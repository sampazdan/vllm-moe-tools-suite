from __future__ import annotations

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
from moe_tools_suite.lab import MODEL_ID
from moe_tools_suite.main import create_app
from moe_tools_suite.settings import Settings


class SendFailure(OSError):
    pass


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

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_response_send_failure_cannot_orphan_a_queued_model_job(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "frontend",
        )
    )
    lab = app.state.lab
    server = BlockingServer()
    lab.server = server  # type: ignore[assignment]
    body = json.dumps({"model_id": MODEL_ID}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/model-sessions",
        "raw_path": b"/api/model-sessions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
    }
    received = False
    hold_receive = asyncio.Event()

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await hold_receive.wait()
        return {"type": "http.disconnect"}

    async def fail_send(message) -> None:
        del message
        raise SendFailure("simulated response transport failure")

    try:
        with pytest.raises(SendFailure):
            await app(scope, receive, fail_send)
        await asyncio.wait_for(server.started.wait(), timeout=1)

        active = lab.active_job()
        assert active is not None
        assert active.kind is JobKind.MODEL_LOAD
        assert active.status is JobStatus.RUNNING
        assert active.id in lab._job_tasks

        server.release.set()
        await asyncio.wait_for(lab.dispatch_job(active.id), timeout=1)

        completed = lab.get_job(active.id)
        persisted = lab.store.load_jobs()[active.id][0]
        assert completed.status is JobStatus.COMPLETED
        assert persisted.status is JobStatus.COMPLETED
        assert lab.active_job() is None

        replacement = lab.submit_model_session(
            CreateModelSessionRequest.model_validate(lab.jobs[active.id].payload)
        )
        await asyncio.wait_for(lab.execute_job(replacement.id), timeout=1)
        assert lab.get_job(replacement.id).status is JobStatus.COMPLETED
    finally:
        server.release.set()
        await lab.shutdown()


@pytest.mark.parametrize(
    ("submit_method", "path", "payload", "kind"),
    [
        (
            "submit_model_session",
            "/api/model-sessions",
            {"model_id": MODEL_ID},
            JobKind.MODEL_LOAD,
        ),
        (
            "submit_prepare_benchmark",
            "/api/benchmarks/fixture-arithmetic/prepare",
            None,
            JobKind.DATASET_PREPARE,
        ),
        (
            "submit_benchmark",
            "/api/runs",
            {"benchmark_id": "fixture-arithmetic", "item_ids": ["arith-03"]},
            JobKind.BENCHMARK_RUN,
        ),
        (
            "submit_agent_run",
            "/api/agent-runs",
            {
                "task_pack_id": "smoke-python-v1",
                "task_ids": ["fix-subtract"],
                "model_session_id": "session",
            },
            JobKind.AGENT_RUN,
        ),
    ],
)
def test_every_submission_route_starts_its_job_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submit_method: str,
    path: str,
    payload: dict[str, object] | None,
    kind: JobKind,
) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "frontend",
        )
    )
    lab = app.state.lab
    job = JobRecord(id=f"{kind.value}-job", kind=kind, status=JobStatus.QUEUED)
    started: list[str] = []
    monkeypatch.setattr(lab, submit_method, lambda *args: job)
    monkeypatch.setattr(lab, "start_job", started.append)

    response = TestClient(app).post(path, json=payload)

    assert response.status_code == 202
    assert response.json()["id"] == job.id
    assert started == [job.id]
