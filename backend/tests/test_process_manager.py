import asyncio
import json
import os
import signal
from pathlib import Path

import httpx
import pytest
from moe_tools_suite.domain import ExpertProfile, ModelRuntimeRecipe, ProfileLayer
from moe_tools_suite.process_manager import (
    ManagedVllmServer,
    _vllm_child_environment,
)

MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"


class FakeProcess:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.pid = 4242
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        while self.returncode is None:
            await asyncio.sleep(0)
        return self.returncode


def _launcher(tmp_path: Path) -> Path:
    launcher = tmp_path / "runpod-serve"
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n")
    launcher.chmod(0o755)
    return launcher


def _manager(
    tmp_path: Path,
    launcher: Path,
    client: httpx.AsyncClient,
) -> ManagedVllmServer:
    return ManagedVllmServer(
        command=launcher,
        base_url="http://127.0.0.1:8000",
        model_id=MODEL_ID,
        data_dir=tmp_path / "data",
        startup_timeout_seconds=1,
        shutdown_timeout_seconds=1,
        poll_interval_seconds=0.01,
        client=client,
    )


def _mock_owned_process_group(
    manager: ManagedVllmServer,
    process: FakeProcess,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def kill_process_group(pid: int, sent_signal: signal.Signals) -> None:
        assert pid == process.pid
        if sent_signal is signal.SIGTERM:
            process.terminate()
        elif sent_signal is signal.SIGKILL:
            process.kill()

    monkeypatch.setattr(os, "killpg", kill_process_group)
    monkeypatch.setattr(
        manager,
        "_process_group_exists",
        lambda pid: pid == process.pid and process.returncode is None,
    )


def test_vllm_child_environment_has_explicit_allowlist_and_secret_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = {
        "CUDA_VISIBLE_DEVICES": "0",
        "NCCL_DEBUG": "INFO",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "RUNPOD_VLLM_TENSOR_PARALLEL_SIZE": "2",
        "VLLM_USE_V1": "1",
        "HF_HOME": "/tmp/hf-cache",
        "HF_TOKEN": "minimum-model-download-credential",
    }
    denied = {
        "CUDA_SECRET_CHANNEL": "must-not-reach-vllm",
        "NCCL_PASSWORD": "must-not-reach-vllm",
        "PYTORCH_AUTH_TOKEN": "must-not-reach-vllm",
        "RUNPOD_VLLM_CONTROL_TOKEN": "must-not-reach-vllm",
        "VLLM_API_KEY": "must-not-reach-vllm",
        "MOE_TOOLS_AUTH_TOKEN": "must-not-reach-vllm",
        "AWS_ACCESS_KEY_ID": "must-not-reach-vllm",
        "CUSTOM_PASSWORD": "must-not-reach-vllm",
    }
    for key, value in {**allowed, **denied}.items():
        monkeypatch.setenv(key, value)

    environment = _vllm_child_environment()

    assert {key: environment.get(key) for key in allowed} == allowed
    assert all(key not in environment for key in denied)


def test_process_group_validation_fails_closed_for_a_nonleader_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "getpgid", lambda pid: pid + 1)

    with pytest.raises(RuntimeError, match="outside its own process group"):
        ManagedVllmServer._validated_process_group_id(4242)


@pytest.mark.asyncio
async def test_managed_server_writes_profile_and_capture_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess()
    invocation: dict[str, object] = {}

    async def create_subprocess(*args, **kwargs):
        invocation["args"] = args
        invocation["kwargs"] = kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setenv("MOE_TOOLS_AUTH_TOKEN", "must-not-reach-vllm")
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-reach-vllm")
    monkeypatch.setenv("MOE_TOOLS_DAYTONA_API_KEY", "must-not-reach-vllm")
    monkeypatch.setenv("DAYTONA_API_KEY", "must-not-reach-vllm")
    monkeypatch.setenv("DAYTONA_JWT_TOKEN", "must-not-reach-vllm")
    monkeypatch.setenv("DAYTONA_ORGANIZATION_ID", "must-not-reach-vllm")
    monkeypatch.setenv("MOE_TOOLS_ANTHROPIC_API_KEY", "must-not-reach-vllm")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-vllm")
    monkeypatch.setenv("MOE_TOOLS_OPENAI_API_KEY", "must-not-reach-vllm")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-vllm")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-reach-vllm")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-vllm")
    monkeypatch.setenv("CUSTOM_PASSWORD", "must-not-reach-vllm")
    monkeypatch.setenv("HF_TOKEN", "minimum-model-download-credential")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"data": [{"id": MODEL_ID}]}, request=request
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        _mock_owned_process_group(manager, process, monkeypatch)
        monkeypatch.setattr(
            manager,
            "_read_process_identity",
            lambda pid: ("100", f"{manager.command} {pid}"),
        )
        profile = ExpertProfile(layers={"0": ProfileLayer(keep=list(range(8)))})
        revision = "1" * 40
        recipe = ModelRuntimeRecipe(
            revision=revision,
            tensor_parallel_size=2,
            max_model_len=8192,
            reasoning_parser="qwen3",
        )

        await manager.start(
            profile=profile,
            session_id="masked-session",
            model_id=MODEL_ID,
            model_revision=revision,
            runtime_recipe=recipe,
        )

        assert manager.pid == 4242
        assert invocation["args"] == (
            str(manager.command),
            "--tensor-parallel-size",
            "2",
        )
        environment = invocation["kwargs"]["env"]
        assert environment["RUNPOD_CAPTURE_ROUTING"] == "1"
        assert environment["RUNPOD_VLLM_MODEL"] == MODEL_ID
        assert environment["RUNPOD_VLLM_REVISION"] == revision
        assert environment["RUNPOD_VLLM_MAX_MODEL_LEN"] == "8192"
        assert environment["RUNPOD_REASONING_PARSER"] == "qwen3"
        assert "MOE_TOOLS_AUTH_TOKEN" not in environment
        assert "RUNPOD_API_KEY" not in environment
        assert "MOE_TOOLS_DAYTONA_API_KEY" not in environment
        assert "DAYTONA_API_KEY" not in environment
        assert "DAYTONA_JWT_TOKEN" not in environment
        assert "DAYTONA_ORGANIZATION_ID" not in environment
        assert "MOE_TOOLS_ANTHROPIC_API_KEY" not in environment
        assert "ANTHROPIC_API_KEY" not in environment
        assert "MOE_TOOLS_OPENAI_API_KEY" not in environment
        assert "OPENAI_API_KEY" not in environment
        assert "GITHUB_TOKEN" not in environment
        assert "AWS_SECRET_ACCESS_KEY" not in environment
        assert "CUSTOM_PASSWORD" not in environment
        assert environment["HF_TOKEN"] == "minimum-model-download-credential"
        profile_path = Path(environment["MOE_PROFILE"])
        assert json.loads(profile_path.read_text()) == profile.model_dump(mode="json")
        process_record = json.loads((manager.runtime_dir / "vllm.pid").read_text())
        assert process_record == {
            "pid": 4242,
            "session_id": "masked-session",
            "start_ticks": "100",
            "model_id": MODEL_ID,
            "model_revision": revision,
        }

        await manager.stop()

        assert process.terminated
        assert manager.pid is None
        assert not (manager.runtime_dir / "vllm.pid").exists()


@pytest.mark.asyncio
async def test_start_stops_validated_orphan_from_previous_app_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess()

    async def create_subprocess(*args, **kwargs):
        del args, kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"data": [{"id": MODEL_ID}]}, request=request
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        pid_file = manager.runtime_dir / "vllm.pid"
        pid_file.write_text(
            json.dumps({"pid": 991, "session_id": "orphan", "start_ticks": "old"})
        )
        orphan_alive = True

        def process_identity(pid: int):
            if pid == 991 and orphan_alive:
                return "old", f"/opt/vllm/bin/vllm serve {MODEL_ID}"
            if pid == 4242:
                return "new", f"{manager.command}"
            return None

        signals: list[tuple[int, signal.Signals]] = []

        def kill_process_group(pid: int, sent_signal: signal.Signals) -> None:
            nonlocal orphan_alive
            signals.append((pid, sent_signal))
            if pid == 991:
                orphan_alive = False
            elif pid == process.pid:
                if sent_signal is signal.SIGTERM:
                    process.terminate()
                elif sent_signal is signal.SIGKILL:
                    process.kill()

        monkeypatch.setattr(manager, "_read_process_identity", process_identity)
        monkeypatch.setattr(os, "killpg", kill_process_group)
        monkeypatch.setattr(
            manager,
            "_process_group_exists",
            lambda pid: orphan_alive if pid == 991 else process.returncode is None,
        )

        await manager.start(profile=None, session_id="replacement")

        assert signals == [(991, signal.SIGTERM)]
        assert json.loads(pid_file.read_text())["session_id"] == "replacement"
        await manager.stop()


@pytest.mark.asyncio
async def test_startup_recovery_stops_validated_orphan_without_starting_vllm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, request=request)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        pid_file = manager.runtime_dir / "vllm.pid"
        pid_file.write_text(
            json.dumps({"pid": 991, "session_id": "interrupted", "start_ticks": "old"})
        )
        orphan_alive = True

        def process_identity(pid: int):
            if pid == 991 and orphan_alive:
                return "old", f"/opt/vllm/bin/vllm serve {MODEL_ID}"
            return None

        signals: list[tuple[int, signal.Signals]] = []

        def kill_process_group(pid: int, sent_signal: signal.Signals) -> None:
            nonlocal orphan_alive
            signals.append((pid, sent_signal))
            orphan_alive = False

        monkeypatch.setattr(manager, "_read_process_identity", process_identity)
        monkeypatch.setattr(os, "killpg", kill_process_group)
        monkeypatch.setattr(
            manager,
            "_process_group_exists",
            lambda pid: pid == 991 and orphan_alive,
        )

        await manager.recover_interrupted_process()

        assert signals == [(991, signal.SIGTERM)]
        assert manager.pid is None
        assert not pid_file.exists()


@pytest.mark.asyncio
async def test_baseline_start_removes_inherited_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation: dict[str, object] = {}
    process = FakeProcess()

    async def create_subprocess(*args, **kwargs):
        del args
        invocation.update(kwargs)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setenv("MOE_PROFILE", "/stale/profile.json")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"data": [{"id": MODEL_ID}]}, request=request
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        _mock_owned_process_group(manager, process, monkeypatch)

        await manager.start(profile=None, session_id="baseline-session")

        assert "MOE_PROFILE" not in invocation["env"]
        await manager.stop()


@pytest.mark.asyncio
async def test_expert_context_control_uses_internal_token_and_validates_receipt(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert len(request.headers["X-vLLM-Expert-Context-Token"]) >= 32
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(
                200,
                json={
                    "supported": True,
                    "unsupported_reason": None,
                    "topology_fingerprint": "t" * 64,
                    "layers": [],
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "old_context_id": "baseline",
                "old_context_fingerprint": "a" * 64,
                "new_context_id": "mask-a",
                "new_context_fingerprint": "b" * 64,
                "topology_fingerprint": "c" * 64,
                "duration_ms": 12.5,
                "process_id": 4242,
                "weights_reloaded": False,
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        manager._process = FakeProcess()

        capabilities = await manager.context_capabilities()
        receipt = await manager.activate_context("mask-a")

    assert capabilities["supported"] is True
    assert receipt.new_context_id == "mask-a"
    assert receipt.weights_reloaded is False
    assert [request.url.path for request in requests] == [
        "/v1/internal/moe-contexts/capabilities",
        "/v1/internal/moe-contexts/activate",
    ]
    for request in requests:
        assert request.extensions["timeout"]["read"] >= 300


@pytest.mark.asyncio
async def test_startup_failure_includes_log_tail_and_cleans_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def create_subprocess(*args, **kwargs):
        del args
        kwargs["stdout"].write(b"CUDA initialization failed\n")
        return FakeProcess(returncode=17)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, request=request)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)

        with pytest.raises(RuntimeError, match="CUDA initialization failed"):
            await manager.start(profile=None, session_id="failed-session")

        assert manager.pid is None
        assert not (manager.runtime_dir / "vllm.pid").exists()


@pytest.mark.asyncio
async def test_forced_stop_kills_the_validated_process_group_and_reaps_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess()

    async def create_subprocess(*args, **kwargs):
        del args, kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"data": [{"id": MODEL_ID}]}, request=request
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        signals: list[tuple[int, signal.Signals]] = []
        worker_alive = True

        def kill_process_group(pid: int, sent_signal: signal.Signals) -> None:
            nonlocal worker_alive
            signals.append((pid, sent_signal))
            if sent_signal is signal.SIGKILL:
                worker_alive = False
                process.kill()

        async def wait_for_group(pid: int) -> bool:
            assert pid == process.pid
            return not worker_alive

        monkeypatch.setattr(os, "killpg", kill_process_group)
        monkeypatch.setattr(manager, "_wait_for_process_group_exit", wait_for_group)

        await manager.start(profile=None, session_id="forced-stop")
        await manager.stop()

        assert signals == [
            (process.pid, signal.SIGTERM),
            (process.pid, signal.SIGKILL),
        ]
        assert not worker_alive
        assert not process.terminated
        assert process.killed
        assert process.returncode == -9
        assert process.wait_calls == 1
        assert manager.pid is None


@pytest.mark.asyncio
async def test_stop_cleans_workers_after_the_process_group_leader_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess(returncode=17)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, request=request)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        manager._process = process
        pid_file = manager.runtime_dir / "vllm.pid"
        pid_file.write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "session_id": "failed-leader",
                    "start_ticks": "old",
                }
            )
        )
        worker_alive = True
        signals: list[tuple[int, signal.Signals]] = []

        def missing_leader(pid: int) -> int:
            assert pid == process.pid
            raise ProcessLookupError

        def kill_process_group(pid: int, sent_signal: signal.Signals) -> None:
            nonlocal worker_alive
            signals.append((pid, sent_signal))
            worker_alive = False

        monkeypatch.setattr(os, "getpgid", missing_leader)
        monkeypatch.setattr(os, "killpg", kill_process_group)
        monkeypatch.setattr(
            manager,
            "_process_group_exists",
            lambda pid: pid == process.pid and worker_alive,
        )

        await manager.stop()

        assert signals == [(process.pid, signal.SIGTERM)]
        assert not worker_alive
        assert process.wait_calls == 1
        assert manager.pid is None
        assert not pid_file.exists()


@pytest.mark.asyncio
async def test_cancelled_startup_stops_the_spawned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess()

    async def create_subprocess(*args, **kwargs):
        del args, kwargs
        return process

    waiting = asyncio.Event()

    async def wait_until_ready() -> None:
        waiting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, request=request)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        _mock_owned_process_group(manager, process, monkeypatch)
        monkeypatch.setattr(manager, "_wait_until_ready", wait_until_ready)
        startup = asyncio.create_task(
            manager.start(profile=None, session_id="cancelled-session")
        )
        await asyncio.wait_for(waiting.wait(), timeout=1)

        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup

        assert process.terminated
        assert manager.pid is None
        assert not (manager.runtime_dir / "vllm.pid").exists()


@pytest.mark.asyncio
async def test_cancelled_spawn_waits_for_handle_then_stops_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess()
    spawned = asyncio.Event()
    release = asyncio.Event()

    async def create_subprocess(*args, **kwargs):
        del args, kwargs
        spawned.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, request=request)
        )
    ) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        _mock_owned_process_group(manager, process, monkeypatch)
        startup = asyncio.create_task(
            manager.start(profile=None, session_id="cancelled-during-spawn")
        )
        await asyncio.wait_for(spawned.wait(), timeout=1)

        startup.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await startup

        assert process.terminated
        assert manager.pid is None
        assert manager._log_handle is None


@pytest.mark.asyncio
async def test_pid_record_failure_stops_spawned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess()

    async def create_subprocess(*args, **kwargs):
        del args, kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, request=request)
        )
    ) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)
        _mock_owned_process_group(manager, process, monkeypatch)
        monkeypatch.setattr(
            manager,
            "_write_pid_file",
            lambda _pid, _session_id: (_ for _ in ()).throw(OSError("disk full")),
        )

        with pytest.raises(OSError, match="disk full"):
            await manager.start(profile=None, session_id="pid-record-failure")

        assert process.terminated
        assert manager.pid is None
        assert manager._log_handle is None


@pytest.mark.asyncio
async def test_cancelled_startup_during_recorded_process_cleanup_finishes_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()
    calls = 0

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, request=request)
        )
    ) as client:
        manager = _manager(tmp_path, _launcher(tmp_path), client)

        async def stop_recorded_process() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                cleanup_entered.set()
                await cleanup_release.wait()

        monkeypatch.setattr(manager, "_stop_recorded_process", stop_recorded_process)
        startup = asyncio.create_task(
            manager.start(profile=None, session_id="cancelled-before-spawn")
        )
        await asyncio.wait_for(cleanup_entered.wait(), timeout=1)

        startup.cancel()
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await startup

        assert calls == 2
        assert manager.pid is None
        assert manager._log_handle is None
