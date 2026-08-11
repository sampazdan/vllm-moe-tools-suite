from __future__ import annotations

import asyncio
import json
import os
import signal
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import httpx

from .domain import ExpertProfile


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
        self.runtime_dir = data_dir / "runtime"
        self.profile_dir = data_dir / "profiles"
        self.log_dir = data_dir / "logs" / "model-sessions"
        self.startup_timeout_seconds = startup_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._client = client or httpx.AsyncClient(timeout=5)
        self._owns_client = client is None
        self._process: asyncio.subprocess.Process | None = None
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
    ) -> None:
        await self.stop()
        await self._stop_recorded_process()
        if not self.command.is_file() or not os.access(self.command, os.X_OK):
            raise RuntimeError(f"vLLM launcher is not executable: {self.command}")

        self.profile_path = self._write_profile(profile, session_id)
        self.log_path = self.log_dir / f"{session_id}.log"
        environment = os.environ.copy()
        environment["RUNPOD_CAPTURE_ROUTING"] = "1"
        environment["RUNPOD_VLLM_MODEL"] = self.model_id
        environment.pop("MOE_TOOLS_AUTH_TOKEN", None)
        environment.pop("RUNPOD_API_KEY", None)
        environment.pop("MOE_TOOLS_DAYTONA_API_KEY", None)
        environment.pop("DAYTONA_API_KEY", None)
        environment.pop("DAYTONA_JWT_TOKEN", None)
        environment.pop("DAYTONA_ORGANIZATION_ID", None)
        environment.pop("MOE_PROFILE", None)
        if self.profile_path is not None:
            environment["MOE_PROFILE"] = str(self.profile_path)

        self._log_handle = self.log_path.open("ab", buffering=0)
        try:
            self._process = await asyncio.create_subprocess_exec(
                str(self.command),
                env=environment,
                stdout=self._log_handle,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            self._close_log()
            raise

        self.started_at = datetime.now(UTC)
        self._write_pid_file(self._process.pid, session_id)
        try:
            await self._wait_until_ready()
        except asyncio.CancelledError:
            await asyncio.shield(self.stop())
            raise
        except Exception as error:
            failure = self._failure_context(error)
            await self.stop()
            raise RuntimeError(failure) from None

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
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._remove_pid_file()
            return

        identity = self._read_process_identity(pid)
        if identity is None or start_ticks is None or identity[0] != start_ticks:
            self._remove_pid_file()
            return
        command_line = identity[1]
        is_expected_process = str(self.command) in command_line or (
            "vllm" in command_line and self.model_id in command_line
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
