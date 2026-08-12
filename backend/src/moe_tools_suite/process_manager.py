from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import os
import secrets
import signal
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, BinaryIO, Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
_CONTROL_ERROR_DETAIL_MAX_LENGTH = 1000

ModelLoadPhaseCallback = Callable[
    [ModelLoadPhase, str, Literal["observed", "unavailable"]], None
]


@dataclass(frozen=True)
class ModelLoadPhaseUpdate:
    phase: ModelLoadPhase
    status: Literal["started", "completed", "failed", "unavailable"]
    detail: str
    bytes_current: int | None = None
    bytes_total: int | None = None
    files_current: int | None = None
    files_total: int | None = None


ModelLoadPhaseUpdateCallback = Callable[[ModelLoadPhaseUpdate], None]

_COORDINATOR_LOAD_PHASES = (
    ModelLoadPhase.CHECKING_CACHE,
    ModelLoadPhase.DOWNLOADING,
)
_WORKER_LOAD_PHASES = (
    ModelLoadPhase.LOADING_WEIGHTS,
    ModelLoadPhase.INITIALIZING_DISTRIBUTED_WORKERS,
    ModelLoadPhase.COMPILING,
    ModelLoadPhase.CAPTURING_GRAPHS,
    ModelLoadPhase.WARMING,
)
_MANAGED_RUNTIME_LOAD_PHASES = _COORDINATOR_LOAD_PHASES + _WORKER_LOAD_PHASES


class _RuntimeModelLoadEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    event_id: Annotated[str, Field(min_length=1, max_length=128)]
    session_id: Annotated[str, Field(min_length=1, max_length=128)]
    phase: ModelLoadPhase
    status: Literal["started", "completed", "failed"]
    detail: Annotated[str, Field(min_length=1, max_length=500)]
    process_id: Annotated[int, Field(gt=1)]
    rank: Annotated[int, Field(ge=0)] | None = None
    world_size: Annotated[int, Field(ge=1, le=64)] = 1
    bytes_current: Annotated[int, Field(ge=0)] | None = None
    bytes_total: Annotated[int, Field(ge=0)] | None = None
    files_current: Annotated[int, Field(ge=0)] | None = None
    files_total: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def validate_runtime_event(self) -> _RuntimeModelLoadEvent:
        if self.phase not in _MANAGED_RUNTIME_LOAD_PHASES:
            raise ValueError("phase is not a managed-runtime startup phase")
        if self.phase in _COORDINATOR_LOAD_PHASES and (
            self.rank != 0 or self.world_size != 1
        ):
            raise ValueError("coordinator phases require rank 0 and world_size 1")
        if self.rank is not None and self.rank >= self.world_size:
            raise ValueError("rank must be lower than world_size")
        if self.rank is None and self.world_size != 1:
            raise ValueError("multi-worker phases require an explicit rank")
        if (
            self.bytes_current is not None
            and self.bytes_total is not None
            and self.bytes_current > self.bytes_total
        ):
            raise ValueError("bytes_current cannot exceed bytes_total")
        if (
            self.files_current is not None
            and self.files_total is not None
            and self.files_current > self.files_total
        ):
            raise ValueError("files_current cannot exceed files_total")
        return self


@dataclass
class _PhaseAggregate:
    world_size: int | None = None
    started_workers: set[int] = field(default_factory=set)
    completed_workers: set[int] = field(default_factory=set)
    started: bool = False
    terminal: bool = False
    detail: str = ""
    bytes_current: int | None = None
    bytes_total: int | None = None
    files_current: int | None = None
    files_total: int | None = None

    def merge(self, event: _RuntimeModelLoadEvent) -> bool:
        if self.world_size is None:
            self.world_size = event.world_size
        elif event.world_size != self.world_size:
            return False
        worker = event.rank if event.rank is not None else event.process_id
        if event.status == "started":
            self.started_workers.add(worker)
        elif event.status == "completed":
            self.completed_workers.add(worker)
        if not self.detail or event.rank in {None, 0}:
            self.detail = event.detail
        for name in (
            "bytes_current",
            "bytes_total",
            "files_current",
            "files_total",
        ):
            value = getattr(event, name)
            current = getattr(self, name)
            if value is not None and (current is None or value > current):
                setattr(self, name, value)
        return True


class _ModelLoadProgressReceiver:
    def __init__(
        self,
        *,
        session_id: str,
        on_update: ModelLoadPhaseUpdateCallback,
        generated_secrets: tuple[str, ...] = (),
        expected_worker_world_size: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.on_update = on_update
        self.token = secrets.token_urlsafe(32)
        self._generated_secrets = (self.token, *generated_secrets)
        self.path = f"/model-load/{secrets.token_urlsafe(18)}"
        self.server: asyncio.AbstractServer | None = None
        self.endpoint: str | None = None
        self._events: set[str] = set()
        self._phases: dict[ModelLoadPhase, _PhaseAggregate] = {}
        self._worker_world_size: int | None = expected_worker_world_size
        self._connection_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._start_connection,
            host="127.0.0.1",
            port=0,
            limit=16_384,
        )
        sockets = self.server.sockets or []
        if len(sockets) != 1:
            await self.aclose()
            raise RuntimeError("model-load progress listener has no bound socket")
        port = sockets[0].getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{port}{self.path}"

    async def aclose(self) -> None:
        if self.server is None:
            return
        self.server.close()
        await self.server.wait_closed()
        self.server = None
        if self._connection_tasks:
            await asyncio.gather(
                *tuple(self._connection_tasks),
                return_exceptions=True,
            )

    def _start_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.create_task(self._handle_connection(reader, writer))
        self._connection_tasks.add(task)
        task.add_done_callback(self._connection_tasks.discard)

    def finalize(self) -> set[ModelLoadPhase]:
        reported = set(self._phases)
        for phase, aggregate in self._phases.items():
            if aggregate.terminal:
                continue
            self.on_update(
                ModelLoadPhaseUpdate(
                    phase=phase,
                    status="unavailable",
                    detail=(
                        "The runtime started this phase but did not deliver a "
                        "complete structured signal before readiness."
                    ),
                    bytes_current=aggregate.bytes_current,
                    bytes_total=aggregate.bytes_total,
                    files_current=aggregate.files_current,
                    files_total=aggregate.files_total,
                )
            )
        return reported

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        status = 400
        try:
            peer = writer.get_extra_info("peername")
            if not peer or not ipaddress.ip_address(peer[0]).is_loopback:
                status = 403
                return
            header_block = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                timeout=2,
            )
            if len(header_block) > 8192:
                status = 413
                return
            request_line, *raw_headers = header_block.decode("ascii").split("\r\n")
            method, target, protocol = request_line.split(" ")
            if method != "POST" or target != self.path or protocol != "HTTP/1.1":
                status = 404
                return
            headers: dict[str, str] = {}
            for raw_header in raw_headers:
                if not raw_header:
                    continue
                name, separator, value = raw_header.partition(":")
                lowered = name.strip().lower()
                if not separator or lowered in headers:
                    return
                headers[lowered] = value.strip()
            supplied_token = headers.get("x-vllm-model-load-token", "")
            if not hmac.compare_digest(supplied_token, self.token):
                status = 403
                return
            if headers.get("content-type") != "application/json":
                status = 415
                return
            content_length = int(headers.get("content-length", "0"))
            if content_length < 2 or content_length > 16_384:
                status = 413
                return
            body = await asyncio.wait_for(
                reader.readexactly(content_length),
                timeout=2,
            )
            event = _RuntimeModelLoadEvent.model_validate_json(body)
            if event.session_id != self.session_id:
                status = 409
                return
            self._accept_event(event)
            status = 204
        except (ValueError, UnicodeDecodeError, asyncio.IncompleteReadError):
            status = 400
        except (asyncio.LimitOverrunError, TimeoutError):
            status = 408
        except Exception:
            status = 500
        finally:
            reason = {
                204: "No Content",
                400: "Bad Request",
                403: "Forbidden",
                404: "Not Found",
                408: "Request Timeout",
                409: "Conflict",
                413: "Content Too Large",
                415: "Unsupported Media Type",
                500: "Internal Server Error",
            }[status]
            writer.write(
                f"HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\n"
                "Connection: close\r\n\r\n".encode()
            )
            with suppress(ConnectionError):
                await writer.drain()
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    def _accept_event(self, event: _RuntimeModelLoadEvent) -> None:
        if event.phase in _WORKER_LOAD_PHASES:
            if self._worker_world_size is None:
                self._worker_world_size = event.world_size
            elif event.world_size != self._worker_world_size:
                return
        if event.event_id in self._events:
            return
        if len(self._events) >= 4096:
            return
        self._events.add(event.event_id)
        aggregate = self._phases.setdefault(event.phase, _PhaseAggregate())
        if aggregate.terminal or not aggregate.merge(event):
            return
        if not aggregate.started:
            aggregate.started = True
            self._send_update(event.phase, "started", aggregate)
        if event.status == "failed":
            aggregate.terminal = True
            self._send_update(event.phase, "failed", aggregate)
        elif (
            not aggregate.terminal
            and aggregate.world_size is not None
            and len(aggregate.completed_workers) >= aggregate.world_size
        ):
            aggregate.terminal = True
            self._send_update(event.phase, "completed", aggregate)

    def _send_update(
        self,
        phase: ModelLoadPhase,
        status: Literal["started", "completed", "failed"],
        aggregate: _PhaseAggregate,
    ) -> None:
        self.on_update(
            ModelLoadPhaseUpdate(
                phase=phase,
                status=status,
                detail=redact_runtime_secrets(
                    aggregate.detail,
                    *self._generated_secrets,
                ),
                bytes_current=aggregate.bytes_current,
                bytes_total=aggregate.bytes_total,
                files_current=aggregate.files_current,
                files_total=aggregate.files_total,
            )
        )


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
        self._model_load_progress_token = ""
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
        on_phase_update: ModelLoadPhaseUpdateCallback | None = None,
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
            self._model_load_progress_token = ""
            for progress_key in (
                "VLLM_MOE_MODEL_LOAD_PROGRESS_URL",
                "VLLM_MOE_MODEL_LOAD_PROGRESS_TOKEN",
                "VLLM_MOE_MODEL_LOAD_PROGRESS_SESSION_ID",
            ):
                environment.pop(progress_key, None)
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
            progress_receiver: _ModelLoadProgressReceiver | None = None
            if on_phase_update is not None:
                candidate = _ModelLoadProgressReceiver(
                    session_id=session_id,
                    on_update=on_phase_update,
                    generated_secrets=(self._context_control_token,),
                    expected_worker_world_size=(
                        self.runtime_recipe.tensor_parallel_size
                        if self.runtime_recipe is not None
                        else None
                    ),
                )
                try:
                    await candidate.start()
                except (OSError, RuntimeError):
                    await candidate.aclose()
                else:
                    progress_receiver = candidate
                    self._model_load_progress_token = candidate.token
                    assert candidate.endpoint is not None
                    environment["VLLM_MOE_MODEL_LOAD_PROGRESS_URL"] = candidate.endpoint
                    environment["VLLM_MOE_MODEL_LOAD_PROGRESS_TOKEN"] = candidate.token
                    environment["VLLM_MOE_MODEL_LOAD_PROGRESS_SESSION_ID"] = session_id
            environment.pop("MOE_PROFILE", None)
            if self.profile_path is not None:
                environment["MOE_PROFILE"] = str(self.profile_path)

            observed_runtime_phases: set[ModelLoadPhase] = set()
            try:
                self._log_handle = self.log_path.open("ab", buffering=0)
                if on_phase is not None:
                    on_phase(
                        ModelLoadPhase.LAUNCHING_PROCESS,
                        f"Launching managed vLLM for {self.model_id}",
                        "observed",
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
                        "observed",
                    )
                try:
                    await self._wait_until_ready()
                except Exception as error:
                    failure = self._failure_context(error)
                    raise RuntimeError(failure) from None
            finally:
                if progress_receiver is not None:
                    await progress_receiver.aclose()
                    observed_runtime_phases = progress_receiver.finalize()
                if on_phase is not None:
                    for phase in _MANAGED_RUNTIME_LOAD_PHASES:
                        if phase in observed_runtime_phases:
                            continue
                        on_phase(
                            phase,
                            (
                                "The managed runtime did not expose a complete "
                                f"structured {phase.value.replace('_', ' ')} signal; "
                                "timing and progress counters are unavailable."
                            ),
                            "unavailable",
                        )
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
            message = f"vLLM expert-context control request failed: {error}"
            if isinstance(error, httpx.HTTPStatusError):
                detail = _bounded_control_error_detail(
                    error.response,
                    self._context_control_token,
                    self._model_load_progress_token,
                )
                if detail is not None:
                    message = f"{message}; server detail: {detail}"
            raise RuntimeError(
                redact_runtime_secrets(
                    message,
                    self._context_control_token,
                    self._model_load_progress_token,
                )
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
            text = log_file.read().decode(errors="replace").strip()
        return redact_runtime_secrets(
            text,
            self._context_control_token,
            self._model_load_progress_token,
        )

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


def redact_runtime_secrets(text: str, *additional_secrets: str) -> str:
    secrets = {
        value
        for key, value in os.environ.items()
        if value
        and len(value) >= 8
        and any(
            marker in key.upper() for marker in ("TOKEN", "KEY", "SECRET", "PASSWORD")
        )
    }
    secrets.update(value for value in additional_secrets if len(value) >= 8)
    redacted = text
    for secret in sorted(secrets, key=len, reverse=True):
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _bounded_control_error_detail(
    response: httpx.Response,
    *additional_secrets: str,
) -> str | None:
    try:
        payload = response.json()
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if not isinstance(detail, str):
        return None
    sanitized = " ".join(redact_runtime_secrets(detail, *additional_secrets).split())
    if not sanitized:
        return None
    if len(sanitized) <= _CONTROL_ERROR_DETAIL_MAX_LENGTH:
        return sanitized
    return sanitized[: _CONTROL_ERROR_DETAIL_MAX_LENGTH - 3].rstrip() + "..."
