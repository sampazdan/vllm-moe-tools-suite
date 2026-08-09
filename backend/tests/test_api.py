from pathlib import Path

from fastapi.testclient import TestClient
from moe_tools_suite.main import create_app
from moe_tools_suite.settings import Settings


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

    load_job = client.post(
        "/api/model-sessions", json={"model_id": model["id"]}
    )
    assert load_job.status_code == 202
    assert load_job.json()["status"] == "queued"
    assert load_job.json()["progress_total"] == 1
    completed_load = client.get(f"/api/jobs/{load_job.json()['id']}").json()
    assert completed_load["status"] == "completed"
    assert client.get("/api/model-sessions/current").json()["state"] == "ready"

    items = client.get(
        "/api/benchmarks/fixture-arithmetic/items"
    ).json()
    baseline_job = client.post(
        "/api/runs",
        json={
            "benchmark_id": "fixture-arithmetic",
            "item_ids": [item["id"] for item in items],
        },
    )
    assert baseline_job.status_code == 202
    assert baseline_job.json()["progress_total"] == len(items)
    completed_baseline = client.get(
        f"/api/jobs/{baseline_job.json()['id']}"
    ).json()
    assert completed_baseline["status"] == "completed"
    assert completed_baseline["progress_current"] == len(items)
    baseline_run = client.get(
        f"/api/runs/{completed_baseline['result_id']}"
    ).json()
    assert baseline_run["score"] == 10 / 12

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

    masked_session_job = client.post(
        "/api/model-sessions",
        json={"model_id": model["id"], "profile": proposed["profile"]},
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
    masked_job = client.post(
        "/api/runs",
        json={
            "benchmark_id": "fixture-arithmetic",
            "item_ids": [item["id"] for item in items[:2]],
        },
    )
    assert masked_job.status_code == 202
    completed_masked = client.get(
        f"/api/jobs/{masked_job.json()['id']}"
    ).json()
    routing = client.get(
        f"/api/runs/{completed_masked['result_id']}/routing"
    ).json()
    first_layer_kept = set(proposed["profile"]["layers"]["0"]["keep"])
    assert all(
        count == 0
        for expert_id, count in enumerate(routing["selection_counts"][0])
        if expert_id not in first_layer_kept
    )
