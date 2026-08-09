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
    model = client.get("/api/models").json()[0]
    assert model["topology"] == {
        "num_layers": 40,
        "num_experts": 256,
        "top_k": 8,
        "routed_layer_ids": list(range(40)),
    }

    loaded = client.post(
        "/api/model-sessions", json={"model_id": model["id"]}
    )
    assert loaded.status_code == 201
    assert loaded.json()["state"] == "ready"

    items = client.get(
        "/api/benchmarks/fixture-arithmetic/items"
    ).json()
    baseline = client.post(
        "/api/runs",
        json={
            "benchmark_id": "fixture-arithmetic",
            "item_ids": [item["id"] for item in items],
        },
    )
    assert baseline.status_code == 201
    baseline_run = baseline.json()
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

    masked_session = client.post(
        "/api/model-sessions",
        json={"model_id": model["id"], "profile": proposed["profile"]},
    )
    assert masked_session.status_code == 201
    masked = client.post(
        "/api/runs",
        json={
            "benchmark_id": "fixture-arithmetic",
            "item_ids": [item["id"] for item in items[:2]],
        },
    )
    assert masked.status_code == 201
    routing = client.get(f"/api/runs/{masked.json()['id']}/routing").json()
    first_layer_kept = set(proposed["profile"]["layers"]["0"]["keep"])
    assert all(
        count == 0
        for expert_id, count in enumerate(routing["selection_counts"][0])
        if expert_id not in first_layer_kept
    )
