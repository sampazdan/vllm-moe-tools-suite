import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from moe_tools_suite.domain import (
    CreateModelSessionRequest,
    JobKind,
    JobRecord,
    JobStatus,
)
from moe_tools_suite.lab import MODEL_ID, ResearchLab
from moe_tools_suite.main import create_app
from moe_tools_suite.persistence import ModelSessionRow, SqliteStore
from moe_tools_suite.settings import Settings
from sqlalchemy import text


def test_completed_run_and_routing_survive_application_restart(
    tmp_path: Path,
) -> None:
    settings = Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "frontend",
    )
    first_app = create_app(settings)
    first_client = TestClient(first_app)
    load_job = first_client.post(
        "/api/model-sessions", json={"model_id": MODEL_ID}
    ).json()
    assert (
        first_client.get(f"/api/jobs/{load_job['id']}").json()["status"] == "completed"
    )
    run_job = first_client.post(
        "/api/runs",
        json={"benchmark_id": "fixture-arithmetic", "item_ids": ["arith-03"]},
    ).json()
    completed_job = first_client.get(f"/api/jobs/{run_job['id']}").json()
    assert completed_job["status"] == "completed"
    run_id = completed_job["result_id"]
    created = first_client.get(f"/api/runs/{run_id}").json()
    with first_app.state.lab.store.sessions() as database:
        record_json = database.execute(
            text(
                "SELECT record_json FROM benchmark_run_metadata WHERE run_id = :run_id"
            ),
            {"run_id": run_id},
        ).scalar_one()
    assert "items" not in json.loads(record_json)

    proposal = first_client.post(
        "/api/profiles/propose",
        json={
            "run_id": run_id,
            "keep_per_layer": 64,
            "metric": "routing_mass",
        },
    ).json()
    saved_profile = first_client.post(
        "/api/profiles",
        json={
            "name": "Restart-safe profile",
            "model_id": MODEL_ID,
            "profile": proposal["profile"],
            "source": "proposal",
            "source_run_id": run_id,
            "metric": "routing_mass",
            "observed_mass_retained": proposal["observed_mass_retained"],
        },
    ).json()
    masked_load = first_client.post(
        "/api/model-sessions",
        json={
            "model_id": MODEL_ID,
            "profile": proposal["profile"],
            "profile_id": saved_profile["id"],
        },
    ).json()
    completed_masked_load = first_client.get(f"/api/jobs/{masked_load['id']}").json()
    assert completed_masked_load["status"] == "completed"
    masked_job = first_client.post(
        "/api/runs",
        json={"benchmark_id": "fixture-arithmetic", "item_ids": ["arith-03"]},
    ).json()
    masked_run_id = first_client.get(f"/api/jobs/{masked_job['id']}").json()[
        "result_id"
    ]
    comparison = first_client.post(
        "/api/comparisons",
        json={
            "baseline_run_id": run_id,
            "candidate_run_id": masked_run_id,
        },
    ).json()

    artifact = settings.data_dir / "artifacts" / "routing" / f"{run_id}.npz"
    assert artifact.is_file()

    second_app = create_app(settings)
    second_client = TestClient(second_app)

    assert second_client.get("/api/system/status").json()["model_state"] == "unloaded"
    restored = second_client.get("/api/runs").json()
    assert restored["total"] == 2
    assert {run["id"] for run in restored["items"]} == {run_id, masked_run_id}
    routing = second_client.get(f"/api/runs/{run_id}/routing").json()
    assert routing["run_id"] == run_id
    assert routing["total_routed_slots"] > 0
    restored_jobs = second_client.get("/api/jobs").json()
    assert {job["status"] for job in restored_jobs} == {"completed"}
    assert second_client.get("/api/profiles").json()[0]["id"] == saved_profile["id"]
    assert second_client.get("/api/comparisons").json()[0]["id"] == comparison["id"]
    masked_detail = second_client.get(f"/api/runs/{masked_run_id}/detail").json()
    assert masked_detail["saved_profile"]["id"] == saved_profile["id"]
    with second_app.state.lab.store.sessions() as database:
        row = database.get(ModelSessionRow, created["model_session_id"])
        assert row.state == "stopped"
        masked_row = database.get(ModelSessionRow, completed_masked_load["result_id"])
        assert masked_row.state == "failed"
        journal_mode = database.execute(text("PRAGMA journal_mode")).scalar_one()
        assert journal_mode == "delete"


def test_interrupted_job_is_failed_closed_after_restart(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = SqliteStore(data_dir)
    interrupted = JobRecord(
        id="interrupted-job",
        kind=JobKind.MODEL_LOAD,
        status=JobStatus.RUNNING,
        progress_total=1,
    )
    store.save_job(interrupted, {"model_id": MODEL_ID, "profile": None})

    client = TestClient(
        create_app(
            Settings(
                mode="mock",
                data_dir=data_dir,
                frontend_dist=tmp_path / "frontend",
            )
        )
    )

    recovered = client.get("/api/jobs/interrupted-job").json()
    assert recovered["status"] == "failed"
    assert recovered["error"] == "application restarted before the job completed"


def test_interrupted_model_load_keeps_job_and_session_linked_after_restart(
    tmp_path: Path,
) -> None:
    settings = Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "frontend",
    )
    initial = ResearchLab(settings)
    submitted = initial.submit_model_session(
        CreateModelSessionRequest(model_id=MODEL_ID)
    )
    assert submitted.result_id is not None
    assert initial.current_model_session().state == "starting"

    restarted = ResearchLab(settings)

    recovered_job = restarted.get_job(submitted.id)
    recovered_session = restarted.model_sessions[submitted.result_id]
    assert recovered_job.status == "failed"
    assert recovered_job.result_id == recovered_session.id
    assert recovered_session.state == "failed"
    assert recovered_job.completed_at is not None
    assert "linked model session was marked failed" in (recovered_job.error or "")
    assert restarted.active_job() is None
    assert restarted.current_model_session() is None


@pytest.mark.asyncio
async def test_lab_startup_recovers_an_interrupted_managed_process(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "frontend",
        )
    )

    class RecoveryProbe:
        recovered = False

        async def recover_interrupted_process(self) -> None:
            self.recovered = True

    probe = RecoveryProbe()
    lab.server = probe  # type: ignore[assignment]

    await lab.startup()

    assert probe.recovered


@pytest.mark.asyncio
async def test_running_model_load_remains_observable_and_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "frontend",
        )
    )

    class BlockingServer:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def start(self, **kwargs) -> None:
            del kwargs
            self.started.set()
            await self.release.wait()

        async def stop(self) -> None:
            return None

    server = BlockingServer()
    lab.server = server  # type: ignore[assignment]
    request = CreateModelSessionRequest(model_id=MODEL_ID)
    submitted = lab.submit_model_session(request)
    response_waiter = asyncio.create_task(lab.dispatch_job(submitted.id))
    await asyncio.wait_for(server.started.wait(), timeout=1)

    response_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await response_waiter

    active = lab.active_job()
    current = lab.current_model_session()
    retried = lab.submit_model_session(request)
    await lab.execute_job(retried.id)

    assert active is not None
    assert active.id == submitted.id
    assert active.status == "running"
    assert current is not None
    assert current.id == submitted.result_id
    assert current.state == "starting"
    assert retried.id == submitted.id
    assert len(lab.jobs) == 1
    assert len(lab.model_sessions) == 1

    server.release.set()
    await asyncio.wait_for(lab.dispatch_job(retried.id), timeout=1)

    assert lab.get_job(submitted.id).status == "completed"
    assert lab.active_job() is None
    assert lab.current_model_session().state == "ready"
