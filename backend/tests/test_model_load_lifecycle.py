from pathlib import Path

import moe_tools_suite.process_manager as process_manager
import pytest
from fastapi.testclient import TestClient
from moe_tools_suite.domain import (
    CreateModelSessionRequest,
    JobPhaseRecord,
    JobStatus,
    ModelLoadPhase,
)
from moe_tools_suite.lab import MODEL_ID, ResearchLab
from moe_tools_suite.main import create_app
from moe_tools_suite.process_manager import ModelLoadPhaseUpdate
from moe_tools_suite.settings import Settings


def _lab(tmp_path: Path) -> ResearchLab:
    return ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )


def test_structured_runtime_phase_update_is_durable_and_keeps_counters(
    tmp_path: Path,
) -> None:
    lab = _lab(tmp_path)
    job = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))

    lab._update_managed_model_load_phase(
        job.id,
        ModelLoadPhaseUpdate(
            phase=ModelLoadPhase.DOWNLOADING,
            status="started",
            detail="Resolving pinned checkpoint",
        ),
    )
    lab._update_managed_model_load_phase(
        job.id,
        ModelLoadPhaseUpdate(
            phase=ModelLoadPhase.DOWNLOADING,
            status="completed",
            detail="Pinned checkpoint is materialized",
            bytes_current=4096,
            bytes_total=4096,
            files_current=4,
            files_total=4,
        ),
    )

    record = lab.get_job(job.id).phase_history[-1]
    persisted = lab.store.load_jobs()[job.id][0].phase_history[-1]
    assert record.phase is ModelLoadPhase.DOWNLOADING
    assert record.status == "completed"
    assert record.observability == "observed"
    assert record.source == "managed_runtime"
    assert record.completed_at is not None
    assert record.bytes_current == record.bytes_total == 4096
    assert record.files_current == record.files_total == 4
    assert persisted == record


def test_managed_runtime_phase_detail_is_redacted_live_and_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "managed-runtime-secret-token"
    monkeypatch.setenv("MANAGED_RUNTIME_TOKEN", secret)
    lab = _lab(tmp_path)
    job = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))

    lab._update_managed_model_load_phase(
        job.id,
        ModelLoadPhaseUpdate(
            phase=ModelLoadPhase.DOWNLOADING,
            status="started",
            detail=f"Downloading checkpoint with token {secret}",
        ),
    )
    lab._update_managed_model_load_phase(
        job.id,
        ModelLoadPhaseUpdate(
            phase=ModelLoadPhase.DOWNLOADING,
            status="completed",
            detail=f"Downloaded checkpoint with token {secret}",
        ),
    )

    live = lab.get_job(job.id).model_dump_json()
    assert secret not in live
    assert "[REDACTED]" in live

    restarted = _lab(tmp_path)
    reloaded = restarted.get_job(job.id).model_dump_json()
    assert secret not in reloaded
    assert "[REDACTED]" in reloaded


def test_progress_receiver_redacts_generated_tokens_before_persistence(
    tmp_path: Path,
) -> None:
    context_token = "generated-context-control-token"
    lab = _lab(tmp_path)
    job = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    receiver = process_manager._ModelLoadProgressReceiver(
        session_id="runtime-session",
        on_update=lambda update: lab._update_managed_model_load_phase(job.id, update),
        generated_secrets=(context_token,),
    )
    leaked_detail = f"callback={receiver.token} context_control={context_token}"
    for status in ("started", "completed"):
        receiver._accept_event(
            process_manager._RuntimeModelLoadEvent(
                version=1,
                event_id=f"loading-weights-{status}",
                session_id="runtime-session",
                phase="loading_weights",
                status=status,
                detail=leaked_detail,
                process_id=5000,
                rank=0,
                world_size=1,
            )
        )

    live = lab.get_job(job.id).model_dump_json()
    assert receiver.token not in live
    assert context_token not in live
    assert live.count("[REDACTED]") >= 2

    restarted = _lab(tmp_path)
    reloaded = restarted.get_job(job.id).model_dump_json()
    assert receiver.token not in reloaded
    assert context_token not in reloaded
    assert reloaded.count("[REDACTED]") >= 2


def test_compiling_record_completes_only_after_last_tp_worker(
    tmp_path: Path,
) -> None:
    lab = _lab(tmp_path)
    job = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    receiver = process_manager._ModelLoadProgressReceiver(
        session_id="runtime-session",
        on_update=lambda update: lab._update_managed_model_load_phase(job.id, update),
    )

    def emit(status: str, rank: int) -> None:
        receiver._accept_event(
            process_manager._RuntimeModelLoadEvent(
                version=1,
                event_id=f"compiling-{status}-{rank}",
                session_id="runtime-session",
                phase="compiling",
                status=status,
                detail=f"Compile boundary {status} on rank {rank}",
                process_id=5000 + rank,
                rank=rank,
                world_size=2,
            )
        )

    emit("started", 0)
    emit("started", 1)
    emit("completed", 0)
    active = lab.get_job(job.id).phase_history[-1]
    assert active.phase is ModelLoadPhase.COMPILING
    assert active.status == "active"
    assert active.completed_at is None

    emit("completed", 1)
    completed = lab.get_job(job.id).phase_history[-1]
    assert completed.status == "completed"
    assert completed.completed_at is not None


def test_terminal_load_state_closes_every_overlapping_runtime_phase(
    tmp_path: Path,
) -> None:
    lab = _lab(tmp_path)
    job = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    lab._set_model_load_phase(
        job.id,
        ModelLoadPhase.RESOLVING_MODEL,
        detail="Resolving manifest",
    )
    for phase in (ModelLoadPhase.LOADING_WEIGHTS, ModelLoadPhase.WARMING):
        lab._update_managed_model_load_phase(
            job.id,
            ModelLoadPhaseUpdate(
                phase=phase,
                status="started",
                detail=f"Started {phase.value}",
            ),
        )
    lab._set_model_load_phase(
        job.id,
        ModelLoadPhase.WAITING_FOR_READINESS,
        detail="Waiting for readiness",
        source="managed_runtime",
    )

    active_before = [
        record.phase
        for record in lab.get_job(job.id).phase_history
        if record.status == "active"
    ]
    assert active_before == [
        ModelLoadPhase.LOADING_WEIGHTS,
        ModelLoadPhase.WARMING,
        ModelLoadPhase.WAITING_FOR_READINESS,
    ]

    lab._cancel_model_load_phase_unlocked(
        lab.jobs[job.id].record,
        detail="Cleanup completed",
    )

    assert all(
        record.status != "active" for record in lab.get_job(job.id).phase_history
    )


@pytest.mark.asyncio
async def test_failed_load_has_structured_recovery_and_can_retry_same_request(
    tmp_path: Path,
) -> None:
    class FailOnceServer:
        attempts = 0

        async def start(self, **kwargs) -> None:
            del kwargs
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("runtime bootstrap failed\nprivate traceback detail")

        async def stop(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    lab = _lab(tmp_path)
    server = FailOnceServer()
    lab.server = server  # type: ignore[assignment]
    try:
        submitted = lab.submit_model_session(
            CreateModelSessionRequest(model_id=MODEL_ID)
        )
        await lab.execute_job(submitted.id)

        failed = lab.get_job(submitted.id)
        terminal = failed.phase_history[-1]
        assert failed.status is JobStatus.FAILED
        assert failed.error == "runtime bootstrap failed"
        assert terminal.phase is ModelLoadPhase.FAILED
        assert terminal.failure_code == "runtime_start_failed"
        assert terminal.recovery_action == (
            "Inspect the collapsed diagnostics, correct the runtime cause, then retry."
        )
        assert terminal.diagnostics == (
            "runtime bootstrap failed\nprivate traceback detail"
        )

        retried = lab.retry_model_load(failed.id)
        assert retried.id != failed.id
        assert retried.retry_of_job_id == failed.id
        assert retried.result_id != failed.result_id
        await lab.execute_job(retried.id)

        assert lab.get_job(retried.id).status is JobStatus.COMPLETED
        assert lab.get_job(failed.id).status is JobStatus.FAILED
        assert lab.current_model_session() is not None
        assert lab.current_model_session().state == "ready"
    finally:
        await lab.shutdown()


@pytest.mark.asyncio
async def test_failed_load_redacts_secrets_before_persisting_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sentinel-super-secret-token"
    monkeypatch.setenv("CUSTOM_RUNTIME_TOKEN", secret)

    class LeakyServer:
        async def start(self, **kwargs) -> None:
            del kwargs
            raise RuntimeError(f"launcher rejected credential {secret}")

        async def stop(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    lab = _lab(tmp_path)
    lab.server = LeakyServer()  # type: ignore[assignment]
    try:
        submitted = lab.submit_model_session(
            CreateModelSessionRequest(model_id=MODEL_ID)
        )
        await lab.execute_job(submitted.id)

        failed = lab.get_job(submitted.id)
        serialized = failed.model_dump_json()
        assert secret not in serialized
        assert "[REDACTED]" in serialized

        restarted = _lab(tmp_path)
        try:
            assert secret not in restarted.get_job(submitted.id).model_dump_json()
        finally:
            await restarted.shutdown()
    finally:
        await lab.shutdown()


def test_retry_refuses_unproven_runtime_cleanup(tmp_path: Path) -> None:
    lab = _lab(tmp_path)
    submitted = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    artifacts = lab.jobs[submitted.id]
    artifacts.record.status = JobStatus.FAILED
    artifacts.record.phase_history.append(
        JobPhaseRecord(
            phase=ModelLoadPhase.FAILED,
            status="failed",
            detail="managed process cleanup failed",
            failure_code="cleanup_failed",
            recovery_action="Restart the deployment before retrying.",
        )
    )

    with pytest.raises(ValueError, match="restart the deployment"):
        lab.retry_model_load(submitted.id)


def test_retry_refuses_manifest_drift_from_original_load_plan(tmp_path: Path) -> None:
    lab = _lab(tmp_path)
    submitted = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    artifacts = lab.jobs[submitted.id]
    artifacts.record.status = JobStatus.CANCELLED
    original = lab.models[0]
    assert original.runtime_recipe is not None
    changed_revision = "f" * 40
    lab.models[0] = original.model_copy(
        update={
            "revision": changed_revision,
            "runtime_recipe": original.runtime_recipe.model_copy(
                update={"revision": changed_revision}
            ),
        }
    )

    with pytest.raises(ValueError, match="manifest changed"):
        lab.retry_model_load(submitted.id)


def test_retry_lineage_survives_save_and_restart(tmp_path: Path) -> None:
    lab = _lab(tmp_path)
    original = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    lab.cancel_job(original.id)

    retried = lab.retry_model_load(original.id)
    lab.cancel_job(retried.id)

    restarted = _lab(tmp_path)
    restored = restarted.get_job(retried.id)
    assert restored.retry_of_job_id == original.id


def test_restart_failure_keeps_structured_retry_guidance(tmp_path: Path) -> None:
    initial = _lab(tmp_path)
    submitted = initial.submit_model_session(
        CreateModelSessionRequest(model_id=MODEL_ID)
    )

    restarted = _lab(tmp_path)
    failed = restarted.get_job(submitted.id)
    terminal = failed.phase_history[-1]

    assert failed.status is JobStatus.FAILED
    assert terminal.failure_code == "application_restarted"
    assert terminal.recovery_action == (
        "Confirm no model load is active, then retry the same pinned model."
    )
    assert terminal.diagnostics == failed.error


def test_retry_endpoint_queues_a_new_durable_load_and_preserves_conflicts(
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
    original = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    lab.cancel_job(original.id)
    started: list[str] = []

    def track_start(job_id: str) -> None:
        if job_id not in started:
            started.append(job_id)

    lab.start_job = track_start  # type: ignore[method-assign]
    client = TestClient(app)

    response = client.post(f"/api/jobs/{original.id}/retry")

    assert response.status_code == 202
    retried = response.json()
    assert retried["id"] != original.id
    assert retried["status"] == "queued"
    assert started == [retried["id"]]

    adopted = client.post(f"/api/jobs/{original.id}/retry")
    assert adopted.status_code == 202
    assert adopted.json()["id"] == retried["id"]

    lab.cancel_job(retried["id"])
    active = lab.submit_prepare_benchmark("fixture-arithmetic")
    conflict = client.post(f"/api/jobs/{original.id}/retry")
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "active_job_conflict"
    assert conflict.json()["detail"]["active_job"]["id"] == active.id


def test_retry_endpoint_starts_real_background_task_on_request_loop(
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
    original = lab.submit_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    lab.cancel_job(original.id)

    response = TestClient(app).post(f"/api/jobs/{original.id}/retry")

    assert response.status_code == 202
    assert response.json()["id"] != original.id
    assert response.json()["kind"] == "model_load"
