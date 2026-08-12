from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import httpx

from .domain import ExpertProfile, ModelLoadPhase, ModelRuntimeRecipe
from .v2_domain import ContextActivationResult

_VLLM_ENV_NAMES = {
    "HOME",
    "HF_HOME",
    "HF_TOKEN",
    "LANG",
    "LD_LIBRARY_PATH",
    "LOGNAME",
    "PATH",
    "PYTHONPATH",
    "SHELL",
    "TERM",
    "TMPDIR",
    "TRANSFORMERS_CACHE",
    "USER",
    "XDG_CACHE_HOME",
}
_VLLM_ENV_PREFIXES = (
    "CUDA_",
    "NCCL_",
    "NVIDIA_",
    "PYTORCH_",
    "RUNPOD_VLLM_",
    "TORCH_",
    "VLLM_",
)

ModelLoadPhaseCallback = Callable[[ModelLoadPhase, str], None]


def _vllm_child_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if (
            key in _VLLM_ENV_NAMES
            or any(key.startswith(prefix) for prefix in _VLLM_ENV_PREFIXES)
        )
        and (
            key == "HF_TOKEN"
            or not any(
                marker in key.upper()
                for marker in ("TOKEN", "KEY", "SECRET", "PASSWORD")
            )
        )
    }


class ManagedVllmServer:
    """Own one loopback-only vLLM server process."""

    def __init__(
        self,
        *,
        command: Path,
        base_url: str,
        model_id: str,
        data_dir: Path,
        startup_timeout_seconds: float,
        shutdown_timeout_seconds: float,
        poll_interval_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.command = command
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.model_revision: str | None = None
        self.runtime_recipe: ModelRuntimeRecipe | None = None
        self.runtime_dir = data_dir / "runtime"
        self.profile_dir = data_dir / "profiles"
        self.log_dir = data_dir / "logs" / "model-sessions"
        self.startup_timeout_seconds = startup_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._control_timeout = httpx.Timeout(
            connect=10,
            read=max(300, startup_timeout_seconds),
            write=30,
            pool=10,
        )
        self._client = client or httpx.AsyncClient(timeout=5)
        self._owns_client = client is None
        self._process: asyncio.subprocess.Process | None = None
        self._context_control_token = secrets.token_urlsafe(32)
        self._log_handle: BinaryIO | None = None
        self.log_path: Path | None = None
        self.profile_path: Path | None = None
        self.started_at: datetime | None = None

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def pid(self) -> int | None:
        if self._process is None or self._process.returncode is not None:
            return None
        return self._process.pid

    async def start(
        self,
        *,
        profile: ExpertProfile | None,
        session_id: str,
        model_id: str | None = None,
        model_revision: str | None = None,
        runtime_recipe: ModelRuntimeRecipe | None = None,
        on_phase: ModelLoadPhaseCallback | None = None,
    ) -> None:
        try:
            await self.stop()
            await self._stop_recorded_process()
            if not self.command.is_file() or not os.access(self.command, os.X_OK):
                raise RuntimeError(f"vLLM launcher is not executable: {self.command}")
            selected_model_id = model_id or self.model_id
            if runtime_recipe is not None and model_revision != runtime_recipe.revision:
                raise RuntimeError(
                    "managed runtime recipe does not match the selected revision"
                )
            self.model_id = selected_model_id
            self.model_revision = model_revision
            self.runtime_recipe = runtime_recipe

            self.profile_path = self._write_profile(profile, session_id)
            self.log_path = self.log_dir / f"{session_id}.log"
            environment = _vllm_child_environment()
            environment["RUNPOD_CAPTURE_ROUTING"] = "1"
            environment["RUNPOD_VLLM_MODEL"] = self.model_id
            for recipe_key in (
                "RUNPOD_VLLM_REVISION",
                "RUNPOD_VLLM_MAX_MODEL_LEN",
                "RUNPOD_REASONING_PARSER",
            ):
                environment.pop(recipe_key, None)
            if self.model_revision is not None:
                environment["RUNPOD_VLLM_REVISION"] = self.model_revision
            command = [str(self.command)]
            if self.runtime_recipe is not None:
                environment["RUNPOD_VLLM_MAX_MODEL_LEN"] = str(
                    self.runtime_recipe.max_model_len
                )
                if self.runtime_recipe.reasoning_parser is not None:
                    environment["RUNPOD_REASONING_PARSER"] = (
                        self.runtime_recipe.reasoning_parser
                    )
                command.extend(
                    [
                        "--tensor-parallel-size",
                        str(self.runtime_recipe.tensor_parallel_size),
                    ]
                )
            environment["VLLM_MOE_EXPERT_CONTEXT_CONTROL_TOKEN"] = (
                self._context_control_token
            )
            environment.pop("MOE_PROFILE", None)
            if self.profile_path is not None:
                environment["MOE_PROFILE"] = str(self.profile_path)

            self._log_handle = self.log_path.open("ab", buffering=0)
            if on_phase is not None:
                on_phase(
                    ModelLoadPhase.LAUNCHING_PROCESS,
                    f"Launching managed vLLM for {self.model_id}",
                )
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *command,
                    env=environment,
                    stdout=self._log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            try:
                self._process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                while not spawn.done():
                    try:
                        await asyncio.shield(spawn)
                    except asyncio.CancelledError:
                        continue
                if not spawn.cancelled() and spawn.exception() is None:
                    self._process = spawn.result()
                raise

            self.started_at = datetime.now(UTC)
            self._write_pid_file(self._process.pid, session_id)
            if on_phase is not None:
                on_phase(
                    ModelLoadPhase.WAITING_FOR_READINESS,
                    "Waiting for the managed runtime readiness API",
                )
            try:
                await self._wait_until_ready()
            except Exception as error:
                failure = self._failure_context(error)
                raise RuntimeError(failure) from None
        except asyncio.CancelledError:
            await self._finish_abort_start()
            raise
        except Exception:
            await self._abort_start()
            raise

    async def _abort_start(self) -> None:
        try:
            await self.stop()
        finally:
            await self._stop_recorded_process()
            self._close_log()

    async def _finish_abort_start(self) -> None:
        cleanup = asyncio.create_task(self._abort_start())
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        if cleanup.cancelled():
            raise RuntimeError("managed vLLM startup cleanup was cancelled")
        if error := cleanup.exception():
            raise RuntimeError(
                f"managed vLLM startup cleanup failed: {error}"
            ) from error

    async def stop(self) -> None:
        process = self._process
        if process is not None:
            process_group_id = self._validated_process_group_id(process.pid)
            await self._terminate_process_group(
                process_group_id,
                process=process,
            )
            self._remove_pid_file()
        self._process = None
        self.started_at = None
        self._close_log()

    async def aclose(self) -> None:
        await self.stop()
        await self._stop_recorded_process()
        if self._owns_client:
            await self._client.aclose()

    async def recover_interrupted_process(self) -> None:
        """Stop a validated vLLM process left behind by an interrupted app."""
        await self._stop_recorded_process()

    async def context_capabilities(self) -> dict[str, object]:
        return await self._context_request("GET", "/capabilities")

    async def current_context(self) -> dict[str, object]:
        return await self._context_request("GET", "/current")

    async def register_context(
        self,
        *,
        context_id: str,
        layers: dict[str, dict[str, list[int]]],
        creation_source: str,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return await self._context_request(
            "POST",
            "/register",
            payload={
                "context_id": context_id,
                "layers": layers,
                "creation_source": creation_source,
                "metadata": metadata or {},
            },
        )

    async def activate_context(self, context_id: str) -> ContextActivationResult:
        response = await self._context_request(
            "POST",
            "/activate",
            payload={"context_id": context_id},
        )
        return self._validate_activation(response)

    async def reset_context(self) -> ContextActivationResult:
        response = await self._context_request("POST", "/reset", payload={})
        return self._validate_activation(response)

    async def _context_request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if self.pid is None:
            raise RuntimeError("vLLM must be ready before changing expert contexts")
        try:
            response = await self._client.request(
                method,
                f"{self.base_url}/v1/internal/moe-contexts{path}",
                headers={
                    "X-vLLM-Expert-Context-Token": self._context_control_token,
                },
                json=payload,
                timeout=self._control_timeout,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"vLLM expert-context control request failed: {error}"
            ) from error
        if not isinstance(body, dict):
            raise RuntimeError("vLLM expert-context control returned a non-object")
        return body

    def _validate_activation(
        self, response: dict[str, object]
    ) -> ContextActivationResult:
        try:
            result = ContextActivationResult.model_validate(
                {
                    key: value
                    for key, value in response.items()
                    if key in ContextActivationResult.model_fields
                }
            )
        except ValueError as error:
            raise RuntimeError(
                "vLLM expert-context activation returned an invalid receipt"
            ) from error
        if result.process_id is not None and result.process_id != self.pid:
            raise RuntimeError(
                "vLLM expert-context activation receipt came from another process"
            )
        return result

    def read_log_tail(self, max_bytes: int = 12_000) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return ""
        with self.log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - max_bytes))
            return log_file.read().decode(errors="replace").strip()

    async def _wait_until_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout_seconds
        last_error = "vLLM did not answer its readiness endpoint"
        while loop.time() < deadline:
            if self._process is None or self._process.returncode is not None:
                return_code = self._process.returncode if self._process else "unknown"
                raise RuntimeError(f"vLLM exited during startup ({return_code})")
            try:
                response = await self._client.get(f"{self.base_url}/v1/models")
                response.raise_for_status()
                model_ids = {item["id"] for item in response.json().get("data", [])}
                if self.model_id in model_ids:
                    return
                last_error = (
                    f"vLLM is ready but serves {sorted(model_ids)}, not {self.model_id}"
                )
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                last_error = str(error) or error.__class__.__name__
            await asyncio.sleep(self.poll_interval_seconds)
        raise RuntimeError(
            f"vLLM was not ready after {self.startup_timeout_seconds:g}s: {last_error}"
        )

    def _write_profile(
        self, profile: ExpertProfile | None, session_id: str
    ) -> Path | None:
        if profile is None:
            return None
        profile_path = self.profile_dir / f"{session_id}.json"
        temporary_path = profile_path.with_name(
            f".{profile_path.name}.{uuid4().hex}.tmp"
        )
        temporary_path.write_text(profile.model_dump_json(indent=2) + "\n")
        temporary_path.chmod(0o600)
        os.replace(temporary_path, profile_path)
        return profile_path

    def _write_pid_file(self, pid: int, session_id: str) -> None:
        pid_file = self.runtime_dir / "vllm.pid"
        temporary_path = pid_file.with_name(f".{pid_file.name}.{uuid4().hex}.tmp")
        identity = self._read_process_identity(pid)
        temporary_path.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "session_id": session_id,
                    "start_ticks": identity[0] if identity else None,
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        temporary_path.chmod(0o600)
        os.replace(temporary_path, pid_file)

    async def _stop_recorded_process(self) -> None:
        pid_file = self.runtime_dir / "vllm.pid"
        if not pid_file.is_file():
            return
        try:
            record = json.loads(pid_file.read_text())
            pid = int(record["pid"])
            start_ticks = record["start_ticks"]
            recorded_model_id = str(record.get("model_id") or self.model_id)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._remove_pid_file()
            return

        identity = self._read_process_identity(pid)
        if identity is None or start_ticks is None or identity[0] != start_ticks:
            self._remove_pid_file()
            return
        command_line = identity[1]
        is_expected_process = str(self.command) in command_line or (
            "vllm" in command_line and recorded_model_id in command_line
        )
        if not is_expected_process:
            raise RuntimeError(
                f"refusing to signal unexpected process recorded as vLLM PID {pid}"
            )

        process_group_id = self._validated_process_group_id(pid)
        await self._terminate_process_group(process_group_id)
        self._remove_pid_file()

    async def _terminate_process_group(
        self,
        process_group_id: int,
        *,
        process: asyncio.subprocess.Process | None = None,
    ) -> None:
        leader_wait = (
            asyncio.create_task(process.wait()) if process is not None else None
        )
        try:
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                if leader_wait is not None:
                    await leader_wait
                return
            if not await self._wait_for_process_group_exit(process_group_id):
                with suppress(ProcessLookupError):
                    os.killpg(process_group_id, signal.SIGKILL)
                if not await self._wait_for_process_group_exit(process_group_id):
                    raise RuntimeError(
                        f"vLLM process group {process_group_id} did not exit"
                    )
            if leader_wait is not None:
                await asyncio.wait_for(
                    leader_wait,
                    timeout=self.shutdown_timeout_seconds,
                )
        finally:
            if leader_wait is not None and not leader_wait.done():
                leader_wait.cancel()
                with suppress(asyncio.CancelledError):
                    await leader_wait

    async def _wait_for_process_group_exit(self, process_group_id: int) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.shutdown_timeout_seconds
        while loop.time() < deadline:
            if not self._process_group_exists(process_group_id):
                return True
            await asyncio.sleep(0.1)
        return not self._process_group_exists(process_group_id)

    @staticmethod
    def _process_group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _validated_process_group_id(pid: int) -> int:
        if pid <= 1 or pid in {os.getpid(), os.getpgrp()}:
            raise RuntimeError(f"refusing unsafe vLLM process group {pid}")
        try:
            process_group_id = os.getpgid(pid)
        except ProcessLookupError:
            return pid
        if process_group_id != pid:
            raise RuntimeError(f"refusing vLLM PID {pid} outside its own process group")
        return pid

    def _remove_pid_file(self) -> None:
        (self.runtime_dir / "vllm.pid").unlink(missing_ok=True)

    @staticmethod
    def _read_process_identity(pid: int) -> tuple[str, str] | None:
        stat_path = Path(f"/proc/{pid}/stat")
        command_path = Path(f"/proc/{pid}/cmdline")
        try:
            stat = stat_path.read_text()
            after_name = stat[stat.rfind(")") + 2 :].split()
            start_ticks = after_name[19]
            command_line = (
                command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            )
        except (OSError, IndexError):
            return None
        return start_ticks, command_line

    def _failure_context(self, error: Exception) -> str:
        message = f"vLLM failed to start: {error}"
        tail = self.read_log_tail(max_bytes=8192)
        return f"{message}:\n{tail}" if tail else message

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
