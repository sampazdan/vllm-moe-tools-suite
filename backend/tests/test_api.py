import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from moe_tools_suite.agentic.domain import CreateAgentRunRequest
from moe_tools_suite.api import _csv_row
from moe_tools_suite.domain import (
    BenchmarkCohort,
    BenchmarkRun,
    CreateModelSessionRequest,
    GenerationConfig,
    RoutingSummary,
    RunItemResult,
)
from moe_tools_suite.lab import MODEL_ID, RunArtifacts
from moe_tools_suite.main import create_app
from moe_tools_suite.settings import Settings


def test_csv_export_neutralizes_spreadsheet_formulas() -> None:
    row = next(
        csv.reader(
            [
                _csv_row(
                    [
                        "=1+1",
                        "+SUM(A1:A2)",
                        "-2+3",
                        "@IMPORTXML()",
                        "\tcommand",
                        "\rcarriage",
                        -4,
                        "ordinary",
                    ]
                )
            ]
        )
    )
    assert row[:6] == [
        "'=1+1",
        "'+SUM(A1:A2)",
        "'-2+3",
        "'@IMPORTXML()",
        "'\tcommand",
        "'\rcarriage",
    ]
    assert row[6:] == ["-4", "ordinary"]


def test_large_run_archive_and_item_history_are_bounded_and_paginated(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    lab = app.state.lab
    client = TestClient(app)
    load_job = client.post("/api/model-sessions", json={"model_id": MODEL_ID}).json()
    assert client.get(f"/api/jobs/{load_job['id']}").json()["status"] == "completed"
    assert lab.session is not None
    item_count = 12_032
    cohort = BenchmarkCohort(
        id="mmlu-pro-full-cohort",
        benchmark_id="mmlu-pro-full",
        benchmark_revision="v1",
        dataset_content_hash="a" * 64,
        prompt_template_version="v1",
        scoring_version="v1",
        item_ids=[f"mmlu-{index:05d}" for index in range(item_count)],
        fingerprint="b" * 64,
        generation=GenerationConfig(),
    )
    lab.cohorts[cohort.id] = cohort
    run = BenchmarkRun(
        id="mmlu-pro-full-run",
        benchmark_id="mmlu-pro-full",
        model_session_id=lab.session.id,
        status="completed",
        score=0.5,
        scored_items=item_count,
        completed_items=item_count,
        total_items=item_count,
        cohort_id=cohort.id,
        items=[
            RunItemResult(
                item_id=f"mmlu-{index:05d}",
                prompt="p",
                expected="A",
                output="A",
                passed=True,
                latency_ms=1,
                prompt_tokens=1,
                completion_tokens=1,
            )
            for index in range(item_count)
        ],
    )
    lab.runs[run.id] = RunArtifacts(
        run=run,
        routing=RoutingSummary(
            run_id=run.id,
            layer_ids=lab.topology.routed_layer_ids,
            selection_counts=[
                [0] * lab.topology.num_experts for _ in lab.topology.routed_layer_ids
            ],
            routing_mass=[
                [0.0] * lab.topology.num_experts for _ in lab.topology.routed_layer_ids
            ],
            total_routed_slots=0,
        ),
    )

    archive = client.get("/api/runs", params={"limit": 1})
    assert archive.status_code == 200
    assert len(archive.content) < 2_000
    assert archive.json()["total"] == 1
    assert "items" not in archive.json()["items"][0]

    summary = client.get(f"/api/runs/{run.id}")
    detail = client.get(f"/api/runs/{run.id}/detail")
    assert "items" not in summary.json()
    assert "items" not in detail.json()["run"]
    assert len(detail.content) < 5_000
    assert detail.json()["cohort"]["item_count"] == item_count

    first_page = client.get(
        f"/api/runs/{run.id}/items",
        params={"limit": 100},
    ).json()
    final_page = client.get(
        f"/api/runs/{run.id}/items",
        params={"offset": 12_000, "limit": 100},
    ).json()
    assert first_page["total"] == item_count
    assert len(first_page["items"]) == 100
    assert first_page["items"][0]["item_id"] == "mmlu-00000"
    assert len(final_page["items"]) == 32
    assert final_page["items"][-1]["item_id"] == "mmlu-12031"
    assert (
        client.get(f"/api/runs/{run.id}/items", params={"limit": 251}).status_code
        == 422
    )


def test_comparison_rejects_legacy_runs_without_verifiable_cohort(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    client = TestClient(app)
    load = client.post("/api/model-sessions", json={"model_id": MODEL_ID}).json()
    assert client.get(f"/api/jobs/{load['id']}").json()["status"] == "completed"
    baseline_job = client.post(
        "/api/runs",
        json={"benchmark_id": "fixture-arithmetic", "item_ids": ["arith-03"]},
    ).json()
    baseline_id = client.get(f"/api/jobs/{baseline_job['id']}").json()["result_id"]
    proposal = client.post(
        "/api/profiles/propose",
        json={
            "run_id": baseline_id,
            "keep_per_layer": 64,
            "metric": "routing_mass",
        },
    ).json()
    profile = client.post(
        "/api/profiles",
        json={
            "name": "Legacy comparison profile",
            "model_id": MODEL_ID,
            "profile": proposal["profile"],
            "source": "proposal",
            "source_run_id": baseline_id,
            "metric": "routing_mass",
        },
    ).json()
    masked_load = client.post(
        "/api/model-sessions",
        json={"model_id": MODEL_ID, "profile_id": profile["id"]},
    ).json()
    assert client.get(f"/api/jobs/{masked_load['id']}").json()["status"] == "completed"
    candidate_job = client.post(
        "/api/runs",
        json={"benchmark_id": "fixture-arithmetic", "item_ids": ["arith-03"]},
    ).json()
    candidate_id = client.get(f"/api/jobs/{candidate_job['id']}").json()["result_id"]
    for run_id in (baseline_id, candidate_id):
        artifacts = app.state.lab.runs[run_id]
        artifacts.provenance = None
        artifacts.run.cohort_id = None

    response = client.post(
        "/api/comparisons",
        json={
            "baseline_run_id": baseline_id,
            "candidate_run_id": candidate_id,
        },
    )

    assert response.status_code == 422
    assert "legacy run comparability cannot be verified" in response.json()["detail"]


def test_lost_model_load_response_can_be_discovered_and_retried(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    lab = app.state.lab
    submitted = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    client = TestClient(app)

    active = client.get("/api/jobs/active")
    assert active.status_code == 200
    assert active.json() == submitted.model_dump(mode="json")
    current = client.get("/api/model-sessions/current").json()
    assert current["id"] == submitted.result_id
    assert current["state"] == "starting"

    retried = client.post("/api/model-sessions", json={"model_id": MODEL_ID})

    assert retried.status_code == 202
    assert retried.json()["id"] == submitted.id
    assert len(client.get("/api/jobs").json()) == 1
    assert len(client.get("/api/model-sessions").json()) == 1
    assert client.get(f"/api/jobs/{submitted.id}").json()["status"] == "completed"
    assert client.get("/api/jobs/active").json() is None
    assert client.get("/api/model-sessions/current").json()["state"] == "ready"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/model-sessions", {"model_id": MODEL_ID}),
        ("/api/benchmarks/fixture-arithmetic/prepare", None),
        (
            "/api/runs",
            {"benchmark_id": "fixture-arithmetic", "item_ids": ["arith-03"]},
        ),
        (
            "/api/agent-runs",
            {
                "task_pack_id": "pack",
                "task_ids": ["task"],
                "model_session_id": "session",
            },
        ),
    ],
)
def test_active_job_conflict_points_to_the_recoverable_job(
    tmp_path: Path,
    path: str,
    payload: dict[str, object] | None,
) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    lab = app.state.lab
    active = lab.submit_prepare_benchmark("fixture-arithmetic")
    client = TestClient(app)

    response = client.post(path, json=payload)

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "active_job_conflict",
        "message": f"dataset_prepare job {active.id} is already queued",
        "active_job": active.model_dump(mode="json"),
        "recovery_url": f"/api/jobs/{active.id}",
    }
    assert client.get("/api/jobs/active").json()["id"] == active.id
    assert client.post(f"/api/jobs/{active.id}/cancel").json()["status"] == (
        "cancelled"
    )
    assert client.get("/api/jobs/active").json() is None


def test_queued_agent_job_is_globally_visible_and_blocks_model_loading(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    client = TestClient(app)
    load = client.post("/api/model-sessions", json={"model_id": MODEL_ID}).json()
    assert client.get(f"/api/jobs/{load['id']}").json()["status"] == "completed"
    session_id = client.get("/api/model-sessions/current").json()["id"]
    queued_agent = app.state.lab.submit_agent_run(
        CreateAgentRunRequest(
            task_pack_id="smoke-python-v1",
            task_ids=["fix-subtract"],
            model_session_id=session_id,
        )
    )

    active = client.get("/api/jobs/active")
    conflict = client.post("/api/model-sessions", json={"model_id": MODEL_ID})

    assert active.status_code == 200
    assert active.json()["id"] == queued_agent.id
    assert active.json()["kind"] == "agent_run"
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "active_job_conflict"
    assert conflict.json()["detail"]["active_job"]["id"] == queued_agent.id


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
    run_items = client.get(f"/api/runs/{run['id']}/items").json()["items"]
    assert run["score"] == 1.0
    assert run["scored_items"] == 1
    assert run_items[1]["passed"] is None

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
    assert baseline_detail["cohort"]["item_count"] == len(items)
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
    assert len(json_export.json()["detail"]["run"]["items"]) == len(items)
    assert json_export.json()["detail"]["cohort"]["item_ids"] == [
        item["id"] for item in items
    ]
    assert "content-length" not in json_export.headers
    assert "attachment" in json_export.headers["content-disposition"]
    csv_export = client.get(f"/api/runs/{baseline_run['id']}/export?format=csv")
    assert csv_export.status_code == 200
    assert csv_export.text.startswith("item_id,passed,scoring")
    assert "content-length" not in csv_export.headers

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

    explorer_response = client.post(
        "/api/routing/explore",
        json={
            "sources": [{"kind": "benchmark_run", "id": baseline_run["id"]}],
            "comparison_sources": [{"kind": "benchmark_run", "id": masked_run_id}],
            "metric": "selection_count",
        },
    )
    assert explorer_response.status_code == 200
    explorer = explorer_response.json()
    assert explorer["model_id"] == model["id"]
    assert explorer["profile_fingerprint"] is None
    raw_total = sum(sum(row) for row in routing["selection_counts"])
    for normalized_row, raw_row in zip(
        explorer["comparison_selection_counts"],
        routing["selection_counts"],
        strict=True,
    ):
        assert normalized_row == pytest.approx([value / raw_total for value in raw_row])
    assert explorer["comparison_sources"][0]["profile_id"] == saved_profile["id"]
    assert explorer["filter_capabilities"] == {
        "item": False,
        "trial": False,
        "step_type": False,
        "outcome": False,
    }
    unsupported_filter = client.post(
        "/api/routing/explore",
        json={
            "sources": [{"kind": "benchmark_run", "id": baseline_run["id"]}],
            "filters": {"item_ids": [items[0]["id"]]},
        },
    )
    assert unsupported_filter.status_code == 422
    assert "item filtering is unavailable" in unsupported_filter.json()["detail"]

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
