import asyncio
import json
import os
import signal
from pathlib import Path

import httpx
import pytest
from moe_tools_suite.domain import ExpertProfile, ProfileLayer
from moe_tools_suite.process_manager import ManagedVllmServer

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

        await manager.start(profile=profile, session_id="masked-session")

        assert manager.pid == 4242
        assert invocation["args"] == (str(manager.command),)
        environment = invocation["kwargs"]["env"]
        assert environment["RUNPOD_CAPTURE_ROUTING"] == "1"
        assert environment["RUNPOD_VLLM_MODEL"] == MODEL_ID
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
        profile_path = Path(environment["MOE_PROFILE"])
        assert json.loads(profile_path.read_text()) == profile.model_dump(mode="json")
        process_record = json.loads((manager.runtime_dir / "vllm.pid").read_text())
        assert process_record == {
            "pid": 4242,
            "session_id": "masked-session",
            "start_ticks": "100",
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
