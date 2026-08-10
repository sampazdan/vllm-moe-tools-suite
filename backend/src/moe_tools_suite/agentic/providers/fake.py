from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic

from ..domain import (
    CommandResult,
    ProviderCapabilities,
    ProviderError,
    ProviderPreflight,
    ProviderStatus,
    SandboxFile,
    SandboxHandle,
    SandboxOwnership,
    SandboxProviderInfo,
    SandboxSpec,
)
from .base import (
    SandboxNotFoundError,
    SandboxOwnershipError,
    SandboxProviderError,
    ownership_labels,
    resolve_sandbox_path,
    truncate_utf8,
)


@dataclass(slots=True)
class _FakeSandbox:
    handle: SandboxHandle
    spec: SandboxSpec
    labels: dict[str, str]
    files: dict[str, bytes] = field(default_factory=dict)
    executable_paths: set[str] = field(default_factory=set)


class FakeSandboxProvider:
    """Deterministic sandbox double that never invokes a local subprocess.

    Commands only return explicitly scripted responses. An unscripted command
    returns exit code 127, making accidental dependence on a host shell visible.
    """

    provider_id = "fake"

    def __init__(
        self,
        *,
        scripted_results: Mapping[str, CommandResult] | None = None,
        max_file_bytes: int = 8 * 1024 * 1024,
        max_output_bytes: int = 200_000,
        preflight_error: ProviderError | None = None,
    ) -> None:
        if max_file_bytes <= 0 or max_output_bytes <= 0:
            raise ValueError("Fake provider byte limits must be positive.")
        self._scripted_results = dict(scripted_results or {})
        self._max_file_bytes = max_file_bytes
        self._max_output_bytes = max_output_bytes
        self._preflight_error = preflight_error
        self._sandboxes: dict[str, _FakeSandbox] = {}
        self._deleted_ids: set[str] = set()
        self._counter = 0
        self._closed = False
        self._last_preflight: ProviderPreflight | None = None
        self.commands: list[tuple[str, str]] = []

    def info(self) -> SandboxProviderInfo:
        preflight = self._last_preflight
        return SandboxProviderInfo(
            id=self.provider_id,
            label="In-memory fake",
            configured=True,
            credential_mode=None,
            required_env=[],
            optional_env=[],
            region=None,
            capabilities=ProviderCapabilities(
                create=True,
                exec=True,
                upload=True,
                download=True,
                delete=True,
            ),
            status=(preflight.status if preflight else ProviderStatus.READY),
            last_checked_at=(preflight.checked_at if preflight else None),
            error=(preflight.error if preflight else None),
        )

    async def preflight(self) -> ProviderPreflight:
        checked_at = datetime.now(UTC)
        error = self._preflight_error
        result = ProviderPreflight(
            provider_id=self.provider_id,
            configured=True,
            reachable=error is None,
            authenticated=error is None,
            status=(ProviderStatus.READY if error is None else ProviderStatus.ERROR),
            latency_ms=0.0,
            checked_at=checked_at,
            api_url_host=None,
            region=None,
            error=error,
        )
        self._last_preflight = result
        return result

    async def create(
        self,
        spec: SandboxSpec,
        ownership: SandboxOwnership,
    ) -> SandboxHandle:
        self._ensure_open()
        self._counter += 1
        sandbox_id = f"fake-{self._counter:06d}"
        handle = SandboxHandle(
            id=sandbox_id,
            provider_id=self.provider_id,
            state="ready",
            image_ref=spec.image_ref,
            image_digest=spec.image_digest,
        )
        self._sandboxes[sandbox_id] = _FakeSandbox(
            handle=handle,
            spec=spec,
            labels=ownership_labels(ownership, self.provider_id),
        )
        return handle

    async def upload(
        self,
        handle: SandboxHandle,
        file: SandboxFile,
    ) -> None:
        sandbox = self._owned_sandbox(handle)
        path = resolve_sandbox_path(sandbox.spec.working_directory, file.path)
        content = file.content.encode("utf-8")
        if len(content) > self._max_file_bytes:
            raise SandboxProviderError(
                "file_too_large",
                "Fake sandbox upload exceeds the configured byte limit.",
            )
        sandbox.files[path] = content
        if file.executable:
            sandbox.executable_paths.add(path)
        else:
            sandbox.executable_paths.discard(path)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        timeout_seconds: int = 120,
    ) -> CommandResult:
        del cwd
        if not command or "\x00" in command or len(command) > 20_000:
            raise SandboxProviderError(
                "invalid_command",
                "Sandbox commands must be non-empty and at most 20000 characters.",
            )
        if timeout_seconds <= 0:
            raise SandboxProviderError(
                "invalid_timeout",
                "Sandbox command timeout must be positive.",
            )
        self._owned_sandbox(handle)
        self.commands.append((handle.id, command))
        started = monotonic()
        scripted = self._scripted_results.get(command)
        if scripted is None:
            scripted = CommandResult(
                command=command,
                exit_code=127,
                stderr=(
                    "FakeSandboxProvider has no scripted response for this command; "
                    "no local process was executed."
                ),
            )
        stdout, stdout_truncated = truncate_utf8(
            scripted.stdout,
            min(self._max_output_bytes, 200_000),
        )
        stderr, stderr_truncated = truncate_utf8(
            scripted.stderr,
            min(self._max_output_bytes, 200_000),
        )
        return scripted.model_copy(
            update={
                "command": command,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": max(
                    scripted.duration_ms,
                    int((monotonic() - started) * 1000),
                ),
                "truncated": (
                    scripted.truncated or stdout_truncated or stderr_truncated
                ),
            }
        )

    async def read(
        self,
        handle: SandboxHandle,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        sandbox = self._owned_sandbox(handle)
        path = resolve_sandbox_path(sandbox.spec.working_directory, path)
        try:
            content = sandbox.files[path]
        except KeyError as error:
            raise SandboxProviderError(
                "file_not_found",
                "The requested fake sandbox file was not found.",
            ) from error
        limit = self._max_file_bytes if max_bytes is None else max_bytes
        if limit < 0 or len(content) > limit:
            raise SandboxProviderError(
                "file_too_large",
                "Fake sandbox read exceeds the configured byte limit.",
            )
        return bytes(content)

    async def delete(self, handle: SandboxHandle) -> None:
        if handle.provider_id != self.provider_id:
            raise SandboxOwnershipError
        if handle.id in self._deleted_ids:
            return
        self._owned_sandbox(handle)
        del self._sandboxes[handle.id]
        self._deleted_ids.add(handle.id)

    async def cleanup_owned(
        self,
        sandbox_id: str | None,
        ownership: SandboxOwnership,
    ) -> None:
        if sandbox_id is None or sandbox_id in self._deleted_ids:
            return
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            self._deleted_ids.add(sandbox_id)
            return
        if sandbox.labels != ownership_labels(ownership, self.provider_id):
            raise SandboxOwnershipError
        del self._sandboxes[sandbox_id]
        self._deleted_ids.add(sandbox_id)

    async def aclose(self) -> None:
        self._closed = True

    def script(self, command: str, result: CommandResult) -> None:
        """Add or replace one exact command response."""

        self._scripted_results[command] = result

    def _owned_sandbox(self, handle: SandboxHandle) -> _FakeSandbox:
        self._ensure_open()
        if handle.provider_id != self.provider_id:
            raise SandboxOwnershipError
        try:
            sandbox = self._sandboxes[handle.id]
        except KeyError as error:
            raise SandboxNotFoundError(handle.id) from error
        return sandbox

    def _ensure_open(self) -> None:
        if self._closed:
            raise SandboxProviderError(
                "provider_closed",
                "The fake sandbox provider is closed.",
            )
