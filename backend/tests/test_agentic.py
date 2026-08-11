from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
from fastapi.testclient import TestClient
from moe_tools_suite.agentic.catalog import AgentTaskCatalog
from moe_tools_suite.agentic.domain import (
    AgentRunStatus,
    CommandResult,
    CreateAgentRunRequest,
    NetworkPolicy,
    SandboxFile,
    SandboxOwnership,
    SandboxSession,
    SandboxSessionState,
    SandboxSpec,
    TerminationCause,
)
from moe_tools_suite.agentic.providers.base import ownership_labels
from moe_tools_suite.agentic.providers.daytona import (
    DaytonaProviderConfig,
    DaytonaSandboxProvider,
)
from moe_tools_suite.agentic.providers.fake import FakeSandboxProvider
from moe_tools_suite.domain import CreateModelSessionRequest
from moe_tools_suite.lab import MODEL_ID, ResearchLab
from moe_tools_suite.main import create_app
from moe_tools_suite.settings import Settings

PACK_ID = "smoke-python-v1"
TASK_IDS = ["fix-subtract", "implement-slugify", "repair-json-cli"]
PUBLIC_TASK_FIELDS = {
    "id",
    "title",
    "instruction",
    "language",
    "tags",
    "timeout_seconds",
}


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "missing-frontend",
        **overrides,
    )


def _load_model(client: TestClient) -> dict[str, object]:
    response = client.post("/api/model-sessions", json={"model_id": MODEL_ID})
    assert response.status_code == 202
    job = response.json()
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "completed"
    session = client.get("/api/model-sessions/current")
    assert session.status_code == 200
    assert session.json()["state"] == "ready"
    return session.json()


def _deny_local_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_process(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("the fake sandbox attempted a local subprocess")

    async def reject_async_process(*args: object, **kwargs: object) -> NoReturn:
        reject_process(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", reject_process)
    monkeypatch.setattr(subprocess, "run", reject_process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_async_process)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", reject_async_process)


def test_public_task_catalog_never_exposes_execution_material(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        packs_response = client.get("/api/agent-task-packs")
        assert packs_response.status_code == 200
        pack = next(item for item in packs_response.json() if item["id"] == PACK_ID)
        assert pack["task_count"] == 3
        assert pack["ready"] is True
        assert pack["oracle_passed"] is True
        assert pack["noop_failed"] is True
        assert len(pack["fingerprint"]) == 64
        assert "tasks" not in pack

        tasks_response = client.get(f"/api/agent-task-packs/{PACK_ID}/tasks")
        assert tasks_response.status_code == 200
        tasks = tasks_response.json()
        assert [task["id"] for task in tasks] == TASK_IDS
        assert all(set(task) == PUBLIC_TASK_FIELDS for task in tasks)

        private_fields = {
            "files",
            "verifier_file_paths",
            "oracle_commands",
            "verifier_command",
            "image_ref",
            "image_digest",
            "working_directory",
            "network_policy",
            "allowed_hosts",
        }
        assert all(set(task).isdisjoint(private_fields) for task in tasks)
        assert "return sum((left, right))" not in tasks_response.text
        assert "python -m unittest" not in tasks_response.text


def test_bundled_task_pack_oracles_pass_and_noop_repositories_fail(
    tmp_path: Path,
) -> None:
    pack = AgentTaskCatalog().get_pack(PACK_ID)
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
    }

    for task in pack.tasks:
        workspace = tmp_path / task.id
        workspace.mkdir()
        for file in task.files:
            destination = workspace / file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(file.content)

        noop = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert noop.returncode != 0, task.id

        for command in task.oracle_commands:
            subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
                check=True,
            )
        oracle = subprocess.run(
            ["/bin/sh", "-c", task.verifier_command],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            check=False,
        )
        assert oracle.returncode == 0, oracle.stdout + oracle.stderr


class _TamperingFakeProvider(FakeSandboxProvider):
    def __init__(self, verifier_command: str, verifier_path: str) -> None:
        super().__init__()
        self.verifier_command = verifier_command
        self.verifier_path = f"/workspace/task/{verifier_path}"
        self.verifier_snapshots: list[bytes] = []

    async def exec(
        self,
        handle,
        command: str,
        *,
        cwd: str | None = None,
        timeout_seconds: int = 120,
    ) -> CommandResult:
        result = await super().exec(
            handle,
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        sandbox = self._owned_sandbox(handle)
        if command == self.verifier_command:
            snapshot = sandbox.files[self.verifier_path]
            self.verifier_snapshots.append(snapshot)
            canonical = b"class SubtractTests" in snapshot
            return result.model_copy(update={"exit_code": 1 if canonical else 0})
        sandbox.files[self.verifier_path] = b"# agent replaced trusted tests\n"
        return result


def test_canonical_verifier_files_are_restored_after_agent_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    task = AgentTaskCatalog().get_task(PACK_ID, TASK_IDS[0])
    provider = _TamperingFakeProvider(
        task.verifier_command,
        task.verifier_file_paths[0],
    )
    app = create_app(_settings(tmp_path))
    app.state.lab.agentic.providers["fake"] = provider

    with TestClient(app) as client:
        session = _load_model(client)
        response = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [task.id],
                "agent_id": "bash-json-v1",
                "sandbox_provider_id": "fake",
                "model_session_id": session["id"],
            },
        )
        assert response.status_code == 202
        run = client.get(f"/api/agent-runs/{response.json()['result_id']}").json()
        assert run["status"] == "completed"
        assert run["passed_trials"] == 0
        assert run["trials"][0]["status"] == "failed"
        artifacts = client.get(f"/api/trials/{run['trials'][0]['id']}/artifacts").json()
        assert artifacts["verifier"]["status"] == "failed"

    canonical = next(
        file.content.encode()
        for file in task.files
        if file.path == task.verifier_file_paths[0]
    )
    assert provider.verifier_snapshots == [
        b"# agent replaced trusted tests\n",
        canonical,
    ]


def test_fake_three_task_vertical_slice_persists_all_research_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    settings = _settings(tmp_path)
    app = create_app(settings)

    with TestClient(app) as client:
        session = _load_model(client)
        session_id = str(session["id"])
        create_response = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": TASK_IDS,
                "agent_id": "bash-json-v1",
                "sandbox_provider_id": "fake",
                "model_session_id": session_id,
                "budgets": {
                    "max_turns": 8,
                    "max_commands": 8,
                    "max_tokens": 32_768,
                    "timeout_seconds": 60,
                },
            },
        )
        assert create_response.status_code == 202
        queued_job = create_response.json()
        assert queued_job["result_id"]
        assert queued_job["progress_total"] == 3

        job = client.get(f"/api/jobs/{queued_job['id']}").json()
        assert job["status"] == "completed"
        assert job["progress_current"] == job["progress_total"] == 3
        run_id = job["result_id"]

        run_response = client.get(f"/api/agent-runs/{run_id}")
        assert run_response.status_code == 200
        run = run_response.json()
        assert run["status"] == "completed"
        assert run["task_pack_id"] == PACK_ID
        assert run["model_session_id"] == session_id
        assert run["sandbox_provider_id"] == "fake"
        assert run["task_ids"] == TASK_IDS
        assert run["total_trials"] == run["completed_trials"] == 3
        assert run["passed_trials"] == 3
        assert run["mean_reward"] == 1.0
        assert run["active_trial_id"] is None
        assert len(run["compatibility_fingerprint"]) == 64

        summaries = run["trials"]
        assert [trial["task_id"] for trial in summaries] == TASK_IDS
        assert all(trial["status"] == "passed" for trial in summaries)
        assert all(trial["reward"] == 1.0 for trial in summaries)
        assert all(trial["turns"] == 3 for trial in summaries)
        assert all(trial["commands"] == 2 for trial in summaries)
        assert all(trial["inference_calls"] == 3 for trial in summaries)
        assert all(trial["routed_inference_calls"] == 3 for trial in summaries)
        assert all(
            trial["termination_reason"] == "agent_finished" for trial in summaries
        )
        assert all(trial["sandbox_status"] == "deleted" for trial in summaries)

        fake_provider = app.state.lab.agentic.providers["fake"]
        assert fake_provider._sandboxes == {}
        assert len(fake_provider.commands) == 9

        artifact_root = settings.data_dir / "artifacts" / "agent-trials"
        trial_ids = {trial["id"] for trial in summaries}
        for trial in summaries:
            trial_id = trial["id"]
            detail = client.get(f"/api/trials/{trial_id}")
            assert detail.status_code == 200
            assert detail.json() == trial

            trajectory_response = client.get(f"/api/trials/{trial_id}/trajectory")
            assert trajectory_response.status_code == 200
            trajectory = trajectory_response.json()
            assert trajectory["format"] == "ATIF"
            assert trajectory["schema_version"] == "ATIF-v1.7"
            assert [step["sequence"] for step in trajectory["steps"]] == list(
                range(1, len(trajectory["steps"]) + 1)
            )
            step_types = [step["type"] for step in trajectory["steps"]]
            assert step_types.count("system") == 1
            assert step_types.count("user") == 1
            assert step_types.count("assistant") == 3
            assert step_types.count("tool") == 2
            assert step_types.count("observation") == 2
            assert step_types[-1] == "verifier"
            inference_views = [
                step["inference"]
                for step in trajectory["steps"]
                if step["type"] == "assistant"
            ]
            assert all(inference is not None for inference in inference_views)
            assert all(
                inference["routed_layers"] == 40
                for inference in inference_views
                if inference is not None
            )
            assert all(
                inference["total_routed_slots"] > 0
                for inference in inference_views
                if inference is not None
            )

            atif_response = client.get(f"/api/trials/{trial_id}/export/atif")
            assert atif_response.status_code == 200
            assert atif_response.headers["content-type"].startswith("application/json")
            assert "attachment" in atif_response.headers["content-disposition"]
            atif = atif_response.json()
            assert atif["schema_version"] == "ATIF-v1.7"
            assert atif["trajectory_id"] == trial_id
            agent_steps = [step for step in atif["steps"] if step["source"] == "agent"]
            assert len(agent_steps) == trial["inference_calls"]
            assert all(step["llm_call_count"] == 1 for step in agent_steps)
            assert all(step["metrics"]["extra"]["inference_id"] for step in agent_steps)
            assert all(
                step["metrics"]["extra"]["routing_artifact"] for step in agent_steps
            )
            assert atif["final_metrics"]["extra"] == {
                "reward": 1.0,
                "termination_cause": "agent_finished",
            }

            routing_response = client.get(f"/api/trials/{trial_id}/routing")
            assert routing_response.status_code == 200
            routing = routing_response.json()
            assert routing["trial_id"] == trial_id
            assert routing["run_id"] == run_id
            assert routing["model_id"] == MODEL_ID
            assert routing["model_session_id"] == session_id
            assert routing["profile_id"] is None
            assert routing["profile_fingerprint"] is None
            assert routing["layer_ids"] == list(range(40))
            assert len(routing["selection_counts"]) == 40
            assert all(len(row) == 256 for row in routing["selection_counts"])
            assert len(routing["routing_mass"]) == 40
            assert all(len(row) == 256 for row in routing["routing_mass"])
            assert routing["inference_count"] == 3
            assert routing["inference_calls"] == 3
            assert routing["captured_inference_calls"] == 3
            assert routing["served_tokens"] == (
                trial["prompt_tokens"] + trial["completion_tokens"]
            )
            assert routing["total_routed_slots"] == sum(
                artifact["total_routed_slots"] for artifact in routing["artifacts"]
            )
            assert len(routing["artifacts"]) == 3
            for artifact in routing["artifacts"]:
                artifact_path = artifact_root / artifact["relative_path"]
                assert artifact_path.is_file()
                assert (
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    == (artifact["sha256"])
                )

            artifacts_response = client.get(f"/api/trials/{trial_id}/artifacts")
            assert artifacts_response.status_code == 200
            artifacts = artifacts_response.json()
            assert artifacts["trial_id"] == trial_id
            assert artifacts["patch"] is None
            assert artifacts["patch_sha256"] is None
            assert artifacts["files_changed"] == 0
            assert artifacts["additions"] == 0
            assert artifacts["deletions"] == 0
            assert artifacts["verifier"]["status"] == "passed"
            assert artifacts["verifier"]["reward"] == 1.0
            assert artifacts["verifier"]["exit_code"] == 0
            assert artifacts["exports"] == [
                {
                    "name": "ATIF trajectory",
                    "media_type": "application/json",
                    "download_url": f"/api/trials/{trial_id}/export/atif",
                }
            ]

        export_response = client.get(f"/api/agent-runs/{run_id}/export")
        assert export_response.status_code == 200
        assert "attachment" in export_response.headers["content-disposition"]
        exported = export_response.json()
        assert exported["schema_version"] == 1
        assert exported["detail"]["run"]["id"] == run_id
        assert set(exported["trajectories"]) == trial_ids
        assert set(exported["routing"]) == trial_ids
        assert set(exported["manifests"]) == trial_ids
        for manifest in exported["manifests"].values():
            assert {item["name"] for item in manifest["artifacts"]} == {
                "trajectory.json",
                "patch.diff",
                "verifier.json",
            }
            for artifact in manifest["artifacts"]:
                artifact_path = artifact_root / artifact["relative_path"]
                assert artifact_path.is_file()
                assert artifact_path.stat().st_size == artifact["size_bytes"]
                assert (
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    == (artifact["sha256"])
                )

        first_trial_id = summaries[0]["id"]
        proposal_response = client.post(
            f"/api/trials/{first_trial_id}/profile-proposal",
            params={"keep_per_layer": 64, "metric": "routing_mass"},
        )
        assert proposal_response.status_code == 200
        proposal = proposal_response.json()
        assert proposal["validation"]["valid"] is True
        assert proposal["validation"]["retained_fraction"] == 0.25
        assert len(proposal["profile"]["layers"]) == 40
        assert all(
            len(layer["keep"]) == 64 for layer in proposal["profile"]["layers"].values()
        )

        profile_response = client.post(
            "/api/profiles",
            json={
                "name": "Agentic smoke routing mass 64",
                "description": "Generated from a persisted coding trial.",
                "model_id": MODEL_ID,
                "profile": proposal["profile"],
                "source": "agentic",
                "source_trial_id": first_trial_id,
                "metric": "routing_mass",
            },
        )
        assert profile_response.status_code == 201
        profile = profile_response.json()
        assert profile["source"] == "agentic"
        assert profile["source_run_id"] is None
        assert profile["source_trial_id"] == first_trial_id
        assert profile["metric"] == "routing_mass"
        assert profile["observed_mass_retained"] == pytest.approx(
            proposal["observed_mass_retained"]
        )
        assert len(profile["profile_fingerprint"]) == 64
        profile_id = profile["id"]

    with TestClient(create_app(settings)) as restarted:
        restored_run_response = restarted.get(f"/api/agent-runs/{run_id}")
        assert restored_run_response.status_code == 200
        restored_run = restored_run_response.json()
        assert restored_run["status"] == "completed"
        assert restored_run["passed_trials"] == 3
        assert {trial["id"] for trial in restored_run["trials"]} == trial_ids

        assert (
            restarted.get(f"/api/trials/{first_trial_id}/trajectory").status_code == 200
        )
        assert (
            restarted.get(f"/api/trials/{first_trial_id}/routing").json()[
                "captured_inference_calls"
            ]
            == 3
        )
        assert (
            restarted.get(f"/api/trials/{first_trial_id}/artifacts").json()["verifier"][
                "status"
            ]
            == "passed"
        )

        restored_profile = next(
            item
            for item in restarted.get("/api/profiles").json()
            if item["id"] == profile_id
        )
        assert restored_profile["source"] == "agentic"
        assert restored_profile["source_trial_id"] == first_trial_id
        assert restored_profile["profile_fingerprint"] == profile["profile_fingerprint"]


class _EmptyAsyncIterator:
    def __aiter__(self) -> _EmptyAsyncIterator:
        return self

    async def __anext__(self) -> NoReturn:
        raise StopAsyncIteration

    async def aclose(self) -> None:
        return None


class _ReadOnlyDaytonaClient:
    def __init__(self) -> None:
        self.list_calls = 0
        self.create_calls = 0
        self.closed = False

    def list(self) -> _EmptyAsyncIterator:
        self.list_calls += 1
        return _EmptyAsyncIterator()

    async def create(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.create_calls += 1
        raise AssertionError("preflight must never create a billed sandbox")

    async def close(self) -> None:
        self.closed = True


class _DaytonaNotFound(Exception):
    pass


class _DaytonaFileSystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def upload_file(self, content: bytes, path: str, timeout: int) -> None:
        assert timeout > 0
        self.files[path] = content

    async def get_file_info(self, path: str) -> SimpleNamespace:
        content = self.files[path]
        return SimpleNamespace(is_dir=False, size=len(content))

    async def download_file(self, path: str, timeout: int) -> bytes:
        assert timeout > 0
        return self.files[path]


class _DaytonaProcess:
    def __init__(self) -> None:
        self.commands: list[tuple[str, str | None, dict[str, str] | None]] = []

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int,
    ) -> SimpleNamespace:
        assert timeout > 0
        self.commands.append((command, cwd, env))
        if command == "sleep forever":
            raise TimeoutError("provider detail must be sanitized")
        result = "command output\n" if command == "python hello.py" else ""
        return SimpleNamespace(exit_code=0, result=result)


class _LifecycleDaytonaSandbox:
    def __init__(self, sandbox_id: str, labels: dict[str, str]) -> None:
        self.id = sandbox_id
        self.labels = labels
        self.state = "started"
        self.process = _DaytonaProcess()
        self.fs = _DaytonaFileSystem()


class _LifecycleDaytonaClient:
    def __init__(self) -> None:
        self.create_params: dict[str, object] | None = None
        self.create_timeout: float | None = None
        self.deleted = False
        self.closed = False
        self.sandbox: _LifecycleDaytonaSandbox | None = None

    async def create(
        self, params: dict[str, object], *, timeout: float
    ) -> _LifecycleDaytonaSandbox:
        self.create_params = params
        self.create_timeout = timeout
        self.sandbox = _LifecycleDaytonaSandbox(
            "daytona-test-sandbox", dict(params["labels"])
        )
        return self.sandbox

    async def get(self, sandbox_id: str) -> _LifecycleDaytonaSandbox:
        if self.deleted or self.sandbox is None or sandbox_id != self.sandbox.id:
            raise _DaytonaNotFound
        return self.sandbox

    async def delete(
        self, sandbox: _LifecycleDaytonaSandbox, *, timeout: float
    ) -> None:
        assert self.sandbox is sandbox
        assert timeout > 0
        self.deleted = True

    async def close(self) -> None:
        self.closed = True


class _FailedWorkspaceDaytonaProcess(_DaytonaProcess):
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int,
    ) -> SimpleNamespace:
        assert timeout > 0
        self.commands.append((command, cwd, env))
        return SimpleNamespace(exit_code=1, result="workspace setup failed")


class _CreateCleanupDaytonaClient:
    def __init__(self, *, recovery_succeeds: bool) -> None:
        self.recovery_succeeds = recovery_succeeds
        self.sandbox: _LifecycleDaytonaSandbox | None = None
        self.deleted = False
        self.delete_calls = 0
        self.list_queries: list[dict[str, object] | None] = []
        self.closed = False

    async def create(
        self, params: dict[str, object], *, timeout: float
    ) -> _LifecycleDaytonaSandbox:
        assert timeout > 0
        self.sandbox = _LifecycleDaytonaSandbox(
            "daytona-create-failure", dict(params["labels"])
        )
        self.sandbox.process = _FailedWorkspaceDaytonaProcess()
        return self.sandbox

    def list(
        self, query: dict[str, object] | None = None
    ) -> AsyncIterator[_LifecycleDaytonaSandbox]:
        self.list_queries.append(query)

        async def iterate() -> AsyncIterator[_LifecycleDaytonaSandbox]:
            if query is not None and self.sandbox is not None and not self.deleted:
                yield self.sandbox

        return iterate()

    async def get(self, sandbox_id: str) -> _LifecycleDaytonaSandbox:
        if self.deleted or self.sandbox is None or sandbox_id != self.sandbox.id:
            raise _DaytonaNotFound
        return self.sandbox

    async def delete(
        self, sandbox: _LifecycleDaytonaSandbox, *, timeout: float
    ) -> None:
        assert self.sandbox is sandbox
        assert timeout > 0
        self.delete_calls += 1
        if self.delete_calls == 1 or not self.recovery_succeeds:
            raise ConnectionError("simulated delete failure")
        self.deleted = True

    async def close(self) -> None:
        self.closed = True


class _RecoveryDaytonaClient:
    def __init__(self, sandboxes: list[_LifecycleDaytonaSandbox]) -> None:
        self.sandboxes = {sandbox.id: sandbox for sandbox in sandboxes}
        self.deleted_ids: list[str] = []
        self.closed = False

    async def get(self, sandbox_id: str) -> _LifecycleDaytonaSandbox:
        try:
            return self.sandboxes[sandbox_id]
        except KeyError as error:
            raise _DaytonaNotFound from error

    async def delete(
        self, sandbox: _LifecycleDaytonaSandbox, *, timeout: float
    ) -> None:
        assert timeout > 0
        self.deleted_ids.append(sandbox.id)
        del self.sandboxes[sandbox.id]

    async def close(self) -> None:
        self.closed = True


def test_daytona_preflight_is_read_only_and_api_responses_redact_secret(
    tmp_path: Path,
) -> None:
    secret = "daytona-test-secret-never-return-this"
    sdk_client = _ReadOnlyDaytonaClient()
    seen_configs: list[DaytonaProviderConfig] = []

    def client_factory(config: DaytonaProviderConfig) -> _ReadOnlyDaytonaClient:
        seen_configs.append(config)
        return sdk_client

    provider = DaytonaSandboxProvider(
        DaytonaProviderConfig(
            api_key=secret,
            api_url="https://app.daytona.io/api",
            target="us",
        ),
        client_factory=client_factory,
    )
    app = create_app(_settings(tmp_path, daytona_api_key=secret))
    app.state.lab.agentic.providers["daytona"] = provider

    with TestClient(app) as client:
        providers_response = client.get("/api/sandbox-providers")
        assert providers_response.status_code == 200
        daytona = next(
            item for item in providers_response.json() if item["id"] == "daytona"
        )
        assert daytona["configured"] is True
        assert daytona["credential_mode"] == "api_key"
        assert daytona["status"] == "unchecked"
        assert secret not in providers_response.text

        preflight_response = client.post("/api/sandbox-providers/daytona/preflight")
        assert preflight_response.status_code == 200
        preflight = preflight_response.json()
        assert preflight["provider_id"] == "daytona"
        assert preflight["configured"] is True
        assert preflight["reachable"] is True
        assert preflight["authenticated"] is True
        assert preflight["status"] == "ready"
        assert preflight["api_url_host"] == "app.daytona.io"
        assert preflight["region"] == "us"
        assert preflight["error"] is None
        assert secret not in preflight_response.text
        assert sdk_client.list_calls == 1
        assert sdk_client.create_calls == 0
        assert len(seen_configs) == 1
        assert seen_configs[0].api_key is not None
        assert seen_configs[0].api_key.get_secret_value() == secret

    assert sdk_client.closed is True


@pytest.mark.asyncio
async def test_daytona_provider_remote_lifecycle_is_private_bounded_and_deleted() -> (
    None
):
    sdk_client = _LifecycleDaytonaClient()
    provider = DaytonaSandboxProvider(
        DaytonaProviderConfig(api_key="configured-test-key", target="us"),
        client_factory=lambda config: sdk_client,
    )
    ownership = SandboxOwnership(
        controller_id="test-controller",
        run_id="run-1",
        trial_id="trial-1",
    )
    spec = SandboxSpec(
        image_ref="python:3.12-slim",
        working_directory="/workspace/task",
        network_policy=NetworkPolicy.NONE,
        cpu=1,
        memory_mb=1024,
        disk_mb=2048,
    )

    try:
        handle = await provider.create(spec, ownership)
        assert handle.id == "daytona-test-sandbox"
        assert sdk_client.create_timeout == 180
        params = sdk_client.create_params
        assert params is not None
        assert params["public"] is False
        assert params["ephemeral"] is True
        assert params["network_block_all"] is True
        assert params["domain_allow_list"] is None
        assert params["auto_stop_interval"] == 15
        assert "auto_delete_interval" not in params
        assert params["env_vars"] == {}
        assert params["secrets"] == {}
        assert params["resources"] == {"cpu": 1, "memory": 1, "disk": 3}
        assert params["labels"] == {
            "moe-tools-controller": "test-controller",
            "moe-tools-run": "run-1",
            "moe-tools-trial": "trial-1",
            "moe-tools-managed": "true",
            "moe-tools-provider": "daytona",
        }

        await provider.upload(
            handle,
            SandboxFile(
                path="hello.py",
                content='print("hello")\n',
                executable=True,
            ),
        )
        assert sdk_client.sandbox is not None
        assert sdk_client.sandbox.fs.files["/workspace/task/hello.py"] == (
            b'print("hello")\n'
        )
        assert await provider.read(handle, "hello.py") == b'print("hello")\n'

        result = await provider.exec(
            handle,
            "python hello.py",
            cwd="/workspace/task",
            timeout_seconds=20,
        )
        assert result.exit_code == 0
        assert result.stdout == "command output\n"
        assert result.stderr == ""
        assert sdk_client.sandbox.process.commands[-1] == (
            "python hello.py",
            "/workspace/task",
            {},
        )

        timed_out = await provider.exec(
            handle,
            "sleep forever",
            cwd="/workspace/task",
            timeout_seconds=1,
        )
        assert timed_out.exit_code is None
        assert timed_out.timed_out is True
        assert timed_out.stderr == "The Daytona request timed out."
        assert "provider detail" not in timed_out.stderr

        await provider.delete(handle)
        assert sdk_client.deleted is True
        await provider.delete(handle)
    finally:
        await provider.aclose()

    assert sdk_client.closed is True


@pytest.mark.parametrize(
    ("recovery_succeeds", "expected_state", "expected_termination"),
    [
        (True, SandboxSessionState.DELETED, TerminationCause.SANDBOX_ERROR),
        (False, SandboxSessionState.CLEANUP_PENDING, TerminationCause.CLEANUP_ERROR),
    ],
)
def test_failed_daytona_create_uses_label_scoped_cleanup(
    tmp_path: Path,
    recovery_succeeds: bool,
    expected_state: SandboxSessionState,
    expected_termination: TerminationCause,
) -> None:
    sdk_client = _CreateCleanupDaytonaClient(recovery_succeeds=recovery_succeeds)
    provider = DaytonaSandboxProvider(
        DaytonaProviderConfig(api_key="configured-test-key", target="us"),
        client_factory=lambda config: sdk_client,
    )
    app = create_app(_settings(tmp_path, daytona_api_key="configured-test-key"))
    app.state.lab.agentic.providers["daytona"] = provider

    with TestClient(app) as client:
        session = _load_model(client)
        preflight = client.post("/api/sandbox-providers/daytona/preflight")
        assert preflight.status_code == 200
        assert preflight.json()["status"] == "ready"
        response = client.post(
            "/api/agent-runs",
            json={
                "task_pack_id": PACK_ID,
                "task_ids": [TASK_IDS[0]],
                "agent_id": "bash-json-v1",
                "sandbox_provider_id": "daytona",
                "model_session_id": session["id"],
            },
        )
        assert response.status_code == 202
        run_id = response.json()["result_id"]
        trial = next(
            item
            for item in app.state.lab.agentic.trials.values()
            if item.run_id == run_id
        )
        assert trial.sandbox_session_id is not None
        sandbox = app.state.lab.agentic.sandboxes[trial.sandbox_session_id]

        assert sandbox.external_id is None
        assert sandbox.state is expected_state
        assert trial.termination_cause is expected_termination
        persisted = app.state.lab.store.load_sandbox_sessions()[sandbox.id]
        assert persisted.state is expected_state
        assert persisted.cleanup_error == sandbox.cleanup_error

        recovery_query = next(
            query for query in reversed(sdk_client.list_queries) if query is not None
        )
        assert recovery_query == {
            "labels": ownership_labels(sandbox.ownership, "daytona"),
            "limit": 10,
        }
        assert sdk_client.delete_calls == 2
        assert sdk_client.deleted is recovery_succeeds
        if recovery_succeeds:
            assert sandbox.deleted_at is not None
            assert sandbox.cleanup_error is None
        else:
            assert sandbox.deleted_at is None
            assert sandbox.cleanup_error == "The Daytona API could not be reached."

    assert sdk_client.closed is True


@pytest.mark.asyncio
async def test_startup_recovers_only_label_matched_daytona_sandboxes_and_metrics(
    tmp_path: Path,
) -> None:
    secret = "startup-recovery-secret"
    settings = _settings(tmp_path, daytona_api_key=secret)
    initial = ResearchLab(settings)
    session = await initial.create_model_session(
        CreateModelSessionRequest(model_id=MODEL_ID)
    )
    request = CreateAgentRunRequest(
        task_pack_id=PACK_ID,
        task_ids=TASK_IDS,
        agent_id="bash-json-v1",
        sandbox_provider_id="fake",
        model_session_id=session.id,
    )
    run = initial.agentic.create_run(
        run_id="interrupted-run",
        job_id="interrupted-job",
        request=request,
        model_session=session,
    )
    run.status = AgentRunStatus.RUNNING
    initial.store.save_agent_run(run)
    trials = [
        trial for trial in initial.agentic.trials.values() if trial.run_id == run.id
    ]
    owned = SandboxOwnership(
        controller_id=settings.agent_controller_id,
        run_id=run.id,
        trial_id=trials[0].id,
    )
    mismatched = SandboxOwnership(
        controller_id=settings.agent_controller_id,
        run_id=run.id,
        trial_id=trials[1].id,
    )
    task_spec = initial.agentic.catalog.get_task(PACK_ID, TASK_IDS[0]).sandbox_spec()
    sessions = [
        SandboxSession(
            id="owned-session",
            trial_id=owned.trial_id,
            provider_id="daytona",
            external_id="owned-sandbox",
            state=SandboxSessionState.READY,
            ownership=owned,
            spec=task_spec,
        ),
        SandboxSession(
            id="mismatched-session",
            trial_id=mismatched.trial_id,
            provider_id="daytona",
            external_id="mismatched-sandbox",
            state=SandboxSessionState.READY,
            ownership=mismatched,
            spec=task_spec,
        ),
    ]
    for sandbox_session in sessions:
        initial.agentic.sandboxes[sandbox_session.id] = sandbox_session
        initial.store.save_sandbox_session(sandbox_session)
    await initial.shutdown()

    correct_remote = _LifecycleDaytonaSandbox(
        "owned-sandbox",
        ownership_labels(owned, "daytona"),
    )
    wrong_labels = ownership_labels(mismatched, "daytona")
    wrong_labels["moe-tools-trial"] = "some-other-trial"
    mismatched_remote = _LifecycleDaytonaSandbox(
        "mismatched-sandbox",
        wrong_labels,
    )
    sdk_client = _RecoveryDaytonaClient([correct_remote, mismatched_remote])
    provider = DaytonaSandboxProvider(
        DaytonaProviderConfig(api_key=secret, target="us"),
        client_factory=lambda config: sdk_client,
    )
    app = create_app(settings)
    app.state.lab.agentic.providers["daytona"] = provider

    reconciled = app.state.lab.agentic.runs[run.id]
    assert reconciled.status == "failed"
    assert reconciled.completed_trials == reconciled.total_trials == 3
    assert all(
        trial.status == "error"
        for trial in app.state.lab.agentic.trials.values()
        if trial.run_id == run.id
    )

    with TestClient(app):
        owned_session = app.state.lab.agentic.sandboxes["owned-session"]
        mismatched_session = app.state.lab.agentic.sandboxes["mismatched-session"]
        assert owned_session.state == "deleted"
        assert owned_session.deleted_at is not None
        assert owned_session.cleanup_error is None
        assert mismatched_session.state == "error"
        assert mismatched_session.deleted_at is None
        assert mismatched_session.cleanup_error == (
            "The sandbox is not owned by this controller run."
        )
        assert secret not in (mismatched_session.cleanup_error or "")
        assert sdk_client.deleted_ids == ["owned-sandbox"]
        assert "mismatched-sandbox" in sdk_client.sandboxes

        persisted = app.state.lab.store.load_sandbox_sessions()
        assert persisted["owned-session"].state == "deleted"
        assert persisted["mismatched-session"].state == "error"

    assert sdk_client.closed is True


@pytest.mark.asyncio
async def test_controller_cancellation_cleans_an_active_fake_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_local_processes(monkeypatch)
    lab = ResearchLab(_settings(tmp_path))
    try:
        session = await lab.create_model_session(
            CreateModelSessionRequest(model_id=MODEL_ID)
        )
        request = CreateAgentRunRequest(
            task_pack_id=PACK_ID,
            task_ids=[TASK_IDS[0]],
            agent_id="bash-json-v1",
            sandbox_provider_id="fake",
            model_session_id=session.id,
        )
        lab.agentic.create_run(
            run_id="cancel-run",
            job_id="cancel-job",
            request=request,
            model_session=session,
        )
        cancellation_checks = 0

        def should_cancel() -> bool:
            nonlocal cancellation_checks
            cancellation_checks += 1
            return cancellation_checks >= 2

        progress: list[tuple[int, int]] = []
        run = await lab.agentic.execute_run(
            "cancel-run",
            model_session=session,
            on_progress=lambda current, total: progress.append((current, total)),
            should_cancel=should_cancel,
        )

        assert run.status == "cancelled"
        assert run.completed_trials == 1
        assert run.passed_trials == 0
        assert progress == [(1, 1)]
        trial = next(
            item for item in lab.agentic.trials.values() if item.run_id == run.id
        )
        assert trial.status == "cancelled"
        assert trial.termination_cause == "cancelled"
        assert trial.reward is None
        assert trial.turns == 0
        assert trial.completed_at is not None

        assert trial.sandbox_session_id is not None
        sandbox = lab.agentic.sandboxes[trial.sandbox_session_id]
        assert sandbox.state == "deleted"
        assert sandbox.external_id is not None
        assert sandbox.deleted_at is not None
        assert sandbox.cleanup_error is None
        fake_provider = lab.agentic.providers["fake"]
        assert fake_provider._sandboxes == {}
        assert fake_provider.commands == []
    finally:
        await lab.shutdown()
