from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from ..domain import (
    CommandResult,
    ProviderError,
    ProviderPreflight,
    SandboxFile,
    SandboxHandle,
    SandboxOwnership,
    SandboxProviderInfo,
    SandboxSpec,
)

MANAGED_LABEL = "moe-tools-managed"
PROVIDER_LABEL = "moe-tools-provider"


class SandboxProviderError(RuntimeError):
    """A bounded, safe-to-display sandbox provider failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def as_provider_error(self) -> ProviderError:
        return ProviderError(
            code=self.code,
            message=self.message,
            retryable=self.retryable,
        )


class SandboxNotFoundError(SandboxProviderError):
    """Raised when a provider cannot find the requested sandbox."""

    def __init__(self, sandbox_id: str) -> None:
        del sandbox_id
        super().__init__(
            "sandbox_not_found",
            "The requested sandbox was not found.",
        )


class SandboxOwnershipError(SandboxProviderError):
    """Raised when a sandbox is outside this provider instance's ownership."""

    def __init__(self) -> None:
        super().__init__(
            "sandbox_ownership_mismatch",
            "The sandbox is not owned by this controller run.",
        )


@runtime_checkable
class SandboxProvider(Protocol):
    """Remote execution boundary used by the agentic trial controller."""

    @property
    def provider_id(self) -> str:
        """Return the provider's stable public identifier."""

    def info(self) -> SandboxProviderInfo:
        """Return local configuration and the latest cached live status."""

    async def preflight(self) -> ProviderPreflight:
        """Perform a bounded, read-only connectivity and authentication check."""

    async def create(
        self,
        spec: SandboxSpec,
        ownership: SandboxOwnership,
    ) -> SandboxHandle:
        """Create an owned remote sandbox."""

    async def upload(
        self,
        handle: SandboxHandle,
        file: SandboxFile,
    ) -> None:
        """Upload one in-memory file to an owned sandbox."""

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        timeout_seconds: int = 120,
    ) -> CommandResult:
        """Execute a command remotely inside an owned sandbox."""

    async def read(
        self,
        handle: SandboxHandle,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        """Read one bounded file from an owned sandbox."""

    async def delete(self, handle: SandboxHandle) -> None:
        """Delete an owned sandbox and wait for confirmed destruction."""

    async def cleanup_owned(
        self,
        sandbox_id: str | None,
        ownership: SandboxOwnership,
    ) -> None:
        """Delete persisted provider state after verifying ownership labels."""

    async def aclose(self) -> None:
        """Close provider transports without implicitly deleting sandboxes."""


def ownership_labels(
    ownership: SandboxOwnership,
    provider_id: str,
) -> dict[str, str]:
    labels = dict(ownership.labels())
    labels[MANAGED_LABEL] = "true"
    labels[PROVIDER_LABEL] = provider_id
    return labels


def labels_match(
    actual: Mapping[str, str] | None,
    expected: Mapping[str, str],
) -> bool:
    if actual is None:
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def validate_remote_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if (
        not candidate.is_absolute()
        or ".." in candidate.parts
        or "\x00" in path
        or len(path) > 4096
    ):
        raise SandboxProviderError(
            "invalid_remote_path",
            "Sandbox file paths must be absolute and no longer than 4096 bytes.",
        )
    return candidate.as_posix()


def resolve_sandbox_path(working_directory: str, path: str) -> str:
    base = PurePosixPath(validate_remote_path(working_directory))
    candidate = PurePosixPath(path)
    if candidate.is_absolute():
        return validate_remote_path(candidate.as_posix())
    if not candidate.parts or ".." in candidate.parts:
        raise SandboxProviderError(
            "invalid_remote_path",
            "Sandbox-relative paths cannot traverse outside the workspace.",
        )
    return validate_remote_path((base / candidate).as_posix())


def truncate_utf8(value: str, max_bytes: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False
    clipped = encoded[:max_bytes]
    return clipped.decode("utf-8", errors="ignore"), True
