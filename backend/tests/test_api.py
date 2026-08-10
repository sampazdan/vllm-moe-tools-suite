import json
from pathlib import Path

from fastapi.testclient import TestClient
from moe_tools_suite.main import create_app
from moe_tools_suite.settings import Settings


def test_custom_benchmark_import_browse_and_ungraded_scoring(
    tmp_path: Path,
) -> None:
    settings = Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "missing-frontend",
    )
    client = TestClient(create_app(settings))
    load_job = client.post(
        "/api/model-sessions", json={"model_id": "Qwen/Qwen3.6-35B-A3B-FP8"}
    ).json()
    assert client.get(f"/api/jobs/{load_job['id']}").json()["status"] == "completed"
    content = "\n".join(
        [
            json.dumps(
                {
                    "id": "sum",
                    "prompt": "Return only the integer result of 2 + 2.",
                    "expected": "4",
                    "category": "correctness",
                }
            ),
            json.dumps(
                {
                    "id": "trace",
                    "prompt": "Describe a repository inspection plan.",
                    "scoring": "ungraded",
                    "category": "agentic",
                }
            ),
        ]
    )
    imported = client.post(
        "/api/benchmarks/custom",
        json={
            "name": "Coding smoke",
            "description": "A tiny harness-development fixture.",
            "content": content,
        },
    )
    assert imported.status_code == 201
    benchmark_id = imported.json()["id"]

    benchmark = next(
        item
        for item in client.get("/api/benchmarks").json()
        if item["id"] == benchmark_id
    )
    assert benchmark["kind"] == "custom"
    page = client.get(
        f"/api/benchmarks/{benchmark_id}/items",
        params={"category": "agentic", "search": "repository"},
    ).json()
    assert page["total"] == 1
    assert page["items"][0]["id"] == "trace"

    empty = client.post(
        "/api/runs",
        json={"benchmark_id": benchmark_id, "item_ids": []},
    )
    assert empty.status_code == 422
    assert "select at least one" in empty.json()["detail"]

    run_job = client.post(
        "/api/runs",
        json={"benchmark_id": benchmark_id, "item_ids": ["sum", "trace"]},
    ).json()
    completed = client.get(f"/api/jobs/{run_job['id']}").json()
    run = client.get(f"/api/runs/{completed['result_id']}").json()
    assert run["score"] == 1.0
    assert run["scored_items"] == 1
    assert run["items"][1]["passed"] is None

    restarted = TestClient(create_app(settings))
    restored = next(
        item
        for item in restarted.get("/api/benchmarks").json()
        if item["id"] == benchmark_id
    )
    assert restored["ready"] is True
    restored_detail = restarted.get(f"/api/runs/{run['id']}/detail").json()
    assert restored_detail["cohort"]["benchmark_id"] == benchmark_id
    assert restored_detail["provenance"]["scoring_version"] == (
        "custom-jsonl-scorer-v1"
    )


def test_mock_vertical_slice_creates_profile_and_masked_run(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    client = TestClient(app)

    assert client.get("/healthz").json() == {"status": "ok", "mode": "mock"}
    assert client.get("/readyz").json() == {
        "status": "ready",
        "database": "ok",
        "mode": "mock",
    }
    model = client.get("/api/models").json()[0]
    assert model["topology"] == {
        "num_layers": 40,
        "num_experts": 256,
        "top_k": 8,
        "routed_layer_ids": list(range(40)),
    }
    runtime = client.get("/api/runtime/status").json()
    assert runtime["managed"] is False
    assert runtime["pid"] is None

    load_job = client.post("/api/model-sessions", json={"model_id": model["id"]})
    assert load_job.status_code == 202
    assert load_job.json()["status"] == "queued"
    assert load_job.json()["progress_total"] == 1
    completed_load = client.get(f"/api/jobs/{load_job.json()['id']}").json()
    assert completed_load["status"] == "completed"
    assert client.get("/api/model-sessions/current").json()["state"] == "ready"

    item_page = client.get("/api/benchmarks/fixture-arithmetic/items").json()
    items = item_page["items"]
    assert item_page["total"] == 12
    baseline_job = client.post(
        "/api/runs",
        json={
            "benchmark_id": "fixture-arithmetic",
            "item_ids": [item["id"] for item in items],
        },
    )
    assert baseline_job.status_code == 202
    assert baseline_job.json()["progress_total"] == len(items)
    completed_baseline = client.get(f"/api/jobs/{baseline_job.json()['id']}").json()
    assert completed_baseline["status"] == "completed"
    assert completed_baseline["progress_current"] == len(items)
    baseline_run = client.get(f"/api/runs/{completed_baseline['result_id']}").json()
    assert baseline_run["score"] == 10 / 12
    baseline_detail = client.get(f"/api/runs/{baseline_run['id']}/detail").json()
    assert baseline_detail["cohort"]["item_ids"] == [item["id"] for item in items]
    assert baseline_detail["provenance"]["dataset_content_hash"]
    assert baseline_detail["provenance"]["generation"] == {
        "temperature": 0.0,
        "max_tokens": 16,
        "seed": 0,
        "enable_thinking": False,
    }
    json_export = client.get(f"/api/runs/{baseline_run['id']}/export")
    assert json_export.status_code == 200
    assert json_export.json()["detail"]["run"]["id"] == baseline_run["id"]
    assert "attachment" in json_export.headers["content-disposition"]
    csv_export = client.get(f"/api/runs/{baseline_run['id']}/export?format=csv")
    assert csv_export.status_code == 200
    assert csv_export.text.startswith("item_id,passed,scoring")

    proposal = client.post(
        "/api/profiles/propose",
        json={
            "run_id": baseline_run["id"],
            "keep_per_layer": 64,
            "metric": "routing_mass",
        },
    )
    assert proposal.status_code == 200
    proposed = proposal.json()
    assert proposed["validation"]["valid"]
    assert proposed["validation"]["retained_fraction"] == 0.25

    saved_profile_response = client.post(
        "/api/profiles",
        json={
            "name": "Arithmetic mass 64",
            "description": "Fixed-budget proposal from the full fixture cohort.",
            "model_id": model["id"],
            "profile": proposed["profile"],
            "source": "proposal",
            "source_run_id": baseline_run["id"],
            "metric": "routing_mass",
            "observed_mass_retained": proposed["observed_mass_retained"],
        },
    )
    assert saved_profile_response.status_code == 201
    saved_profile = saved_profile_response.json()
    assert saved_profile["validation"] == proposed["validation"]
    assert len(saved_profile["profile_fingerprint"]) == 64
    exported = client.get(f"/api/profiles/{saved_profile['id']}/export")
    assert exported.json() == proposed["profile"]
    assert "attachment" in exported.headers["content-disposition"]

    masked_session_job = client.post(
        "/api/model-sessions",
        json={
            "model_id": model["id"],
            "profile": proposed["profile"],
            "profile_id": saved_profile["id"],
        },
    )
    assert masked_session_job.status_code == 202
    completed_masked_load = client.get(
        f"/api/jobs/{masked_session_job.json()['id']}"
    ).json()
    assert completed_masked_load["status"] == "completed"
    masked_model_session = client.get(
        f"/api/model-sessions/{completed_masked_load['result_id']}"
    ).json()
    assert masked_model_session["profile"] == proposed["profile"]
    assert masked_model_session["profile_id"] == saved_profile["id"]
    masked_job = client.post(
        "/api/runs",
        json={
            "benchmark_id": "fixture-arithmetic",
            "item_ids": [item["id"] for item in items],
        },
    )
    assert masked_job.status_code == 202
    completed_masked = client.get(f"/api/jobs/{masked_job.json()['id']}").json()
    routing = client.get(f"/api/runs/{completed_masked['result_id']}/routing").json()
    first_layer_kept = set(proposed["profile"]["layers"]["0"]["keep"])
    assert all(
        count == 0
        for expert_id, count in enumerate(routing["selection_counts"][0])
        if expert_id not in first_layer_kept
    )

    duplicate_profile = client.post(
        "/api/profiles",
        json={
            "name": "Same mask, later research note",
            "model_id": model["id"],
            "profile": proposed["profile"],
            "source": "manual",
            "parent_profile_id": saved_profile["id"],
        },
    ).json()
    assert (
        duplicate_profile["profile_fingerprint"] == saved_profile["profile_fingerprint"]
    )

    masked_run_id = completed_masked["result_id"]
    run_detail = client.get(f"/api/runs/{masked_run_id}/detail").json()
    assert run_detail["saved_profile"]["id"] == saved_profile["id"]
    assert run_detail["model_session"]["profile"] == proposed["profile"]

    comparison_response = client.post(
        "/api/comparisons",
        json={
            "name": "Arithmetic paired check",
            "baseline_run_id": baseline_run["id"],
            "candidate_run_id": masked_run_id,
        },
    )
    assert comparison_response.status_code == 201
    comparison = comparison_response.json()
    assert comparison["profile_id"] == saved_profile["id"]
    assert comparison["cohort_item_ids"] == [item["id"] for item in items]
    assert comparison["regressions"] + comparison["recoveries"] + comparison[
        "retained_passes"
    ] + comparison["retained_failures"] == len(items)
    assert {profile["id"] for profile in client.get("/api/profiles").json()} == {
        saved_profile["id"],
        duplicate_profile["id"],
    }
    assert client.get("/api/comparisons").json()[0]["id"] == comparison["id"]

    generation_job = client.post(
        "/api/runs",
        json={
            "benchmark_id": "fixture-arithmetic",
            "item_ids": [item["id"] for item in items],
            "generation": {
                "temperature": 0,
                "max_tokens": 17,
                "seed": 0,
            },
        },
    ).json()
    generation_run_id = client.get(f"/api/jobs/{generation_job['id']}").json()[
        "result_id"
    ]
    generation_mismatch = client.post(
        "/api/comparisons",
        json={
            "baseline_run_id": baseline_run["id"],
            "candidate_run_id": generation_run_id,
        },
    )
    assert generation_mismatch.status_code == 422
    assert "generation" in generation_mismatch.json()["detail"]

    mismatched_job = client.post(
        "/api/runs",
        json={
            "benchmark_id": "fixture-arithmetic",
            "item_ids": [items[0]["id"]],
        },
    ).json()
    mismatched_run_id = client.get(f"/api/jobs/{mismatched_job['id']}").json()[
        "result_id"
    ]
    incompatible = client.post(
        "/api/comparisons",
        json={
            "baseline_run_id": baseline_run["id"],
            "candidate_run_id": mismatched_run_id,
        },
    )
    assert incompatible.status_code == 422
    assert "cohort fingerprints" in incompatible.json()["detail"]
