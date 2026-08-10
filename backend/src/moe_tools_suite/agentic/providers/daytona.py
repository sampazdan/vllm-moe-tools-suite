from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import math
import os
import re
import shlex
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import PurePosixPath
from time import monotonic
from typing import Any, ClassVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from ..domain import (
    CommandResult,
    CredentialMode,
    ProviderCapabilities,
    ProviderError,
    ProviderPreflight,
    ProviderStatus,
    SandboxFile,
    SandboxHandle,
    SandboxOwnership,
    SandboxProviderInfo,
    SandboxSpec,
    SandboxState,
)
from .base import (
    SandboxNotFoundError,
    SandboxOwnershipError,
    SandboxProviderError,
    labels_match,
    ownership_labels,
    resolve_sandbox_path,
    truncate_utf8,
    validate_remote_path,
)

DAYTONA_SDK_VERSION = "0.192.0"


class DaytonaProviderConfig(BaseModel):
    """Secret-safe configuration for the pinned Daytona SDK adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr | None = Field(default=None, repr=False)
    jwt_token: SecretStr | None = Field(default=None, repr=False)
    organization_id: str | None = None
    api_url: str = "https://app.daytona.io/api"
    target: str | None = None
    preflight_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    create_timeout_seconds: float = Field(default=180.0, gt=0, le=1800)
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    delete_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    file_timeout_seconds: int = Field(default=300, gt=0, le=1800)
    max_exec_timeout_seconds: int = Field(default=900, gt=0, le=3600)
    max_file_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    max_output_bytes: int = Field(default=200_000, gt=0, le=500_000)
    auto_stop_interval_minutes: int = Field(default=15, ge=0, le=60 * 24)

    @field_validator("api_url")
    @classmethod
    def validate_api_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "api_url must be an HTTP(S) URL without credentials, query, or fragment"
            )
        return value.rstrip("/")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> DaytonaProviderConfig:
        source = os.environ if environ is None else environ
        return cls(
            api_key=_secret(source.get("DAYTONA_API_KEY")),
            jwt_token=_secret(source.get("DAYTONA_JWT_TOKEN")),
            organization_id=source.get("DAYTONA_ORGANIZATION_ID") or None,
            api_url=(source.get("DAYTONA_API_URL") or "https://app.daytona.io/api"),
            target=source.get("DAYTONA_TARGET") or None,
        )

    @property
    def configured(self) -> bool:
        return self.credential_mode is not None

    @property
    def credential_mode(self) -> CredentialMode | None:
        if _secret_value(self.api_key):
            return CredentialMode.API_KEY
        if (
            _secret_value(self.jwt_token)
            and self.organization_id
            and self.organization_id.strip()
        ):
            return CredentialMode.JWT
        return None

    @property
    def api_url_host(self) -> str | None:
        return urlsplit(self.api_url).hostname


class DaytonaSandboxProvider:
    """Lazy remote sandbox provider for Daytona Python SDK 0.192.0."""

    provider_id = "daytona"
    sdk_version: ClassVar[str] = DAYTONA_SDK_VERSION

    def __init__(
        self,
        config: DaytonaProviderConfig,
        *,
        client_factory: Callable[[DaytonaProviderConfig], Any] | None = None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory
        self._client_instance: Any | None = None
        self._client_lock = asyncio.Lock()
        self._owned_labels: dict[str, dict[str, str]] = {}
        self._working_directories: dict[str, str] = {}
        self._deleted_ids: set[str] = set()
        self._last_preflight: ProviderPreflight | None = None
        self._closed = False

    def info(self) -> SandboxProviderInfo:
        preflight = self._last_preflight
        configured = self._config.configured
        if preflight is not None:
            status = preflight.status
            checked_at = preflight.checked_at
            error = preflight.error
        else:
            status = (
                ProviderStatus.UNCHECKED
                if configured
                else ProviderStatus.NOT_CONFIGURED
            )
            checked_at = None
            error = None
        mode = self._config.credential_mode
        return SandboxProviderInfo(
            id=self.provider_id,
            label="Daytona",
            configured=configured,
            credential_mode=mode,
            required_env=["MOE_TOOLS_DAYTONA_API_KEY"],
            optional_env=[
                "MOE_TOOLS_DAYTONA_API_URL",
                "MOE_TOOLS_DAYTONA_TARGET",
            ],
            region=self._config.target,
            capabilities=ProviderCapabilities(
                create=True,
                exec=True,
                upload=True,
                download=True,
                delete=True,
            ),
            status=status,
            last_checked_at=checked_at,
            error=error,
        )

    async def preflight(self) -> ProviderPreflight:
        checked_at = datetime.now(UTC)
        started = monotonic()
        if not self._config.configured:
            result = ProviderPreflight(
                provider_id=self.provider_id,
                configured=False,
                reachable=False,
                authenticated=False,
                status=ProviderStatus.NOT_CONFIGURED,
                latency_ms=None,
                checked_at=checked_at,
                api_url_host=self._config.api_url_host,
                region=self._config.target,
                error=ProviderError(
                    code="not_configured",
                    message=(
                        "Configure MOE_TOOLS_DAYTONA_API_KEY and restart the app."
                    ),
                    retryable=False,
                ),
            )
            self._last_preflight = result
            return result

        error: ProviderError | None = None
        try:
            client = await self._client()
            iterator = client.list()
            try:
                await asyncio.wait_for(
                    anext(iterator, None),
                    timeout=self._config.preflight_timeout_seconds,
                )
            finally:
                closer = getattr(iterator, "aclose", None)
                if closer is not None:
                    with suppress(Exception):
                        await closer()
        except Exception as exc:
            error = _provider_error(exc)

        latency_ms = (monotonic() - started) * 1000
        if error is None:
            reachable = True
            authenticated = True
            status = ProviderStatus.READY
        else:
            reachable = error.code in {"authentication_failed", "forbidden"}
            authenticated = error.code == "forbidden"
            status = ProviderStatus.ERROR
        result = ProviderPreflight(
            provider_id=self.provider_id,
            configured=True,
            reachable=reachable,
            authenticated=authenticated,
            status=status,
            latency_ms=latency_ms,
            checked_at=checked_at,
            api_url_host=self._config.api_url_host,
            region=self._config.target,
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
        client = await self._client()
        sdk = _load_daytona_sdk() if self._client_factory is None else None
        expected_labels = ownership_labels(ownership, self.provider_id)
        image = _image_reference(spec)
        working_directory = validate_remote_path(spec.working_directory)
        network_block_all, domain_allow_list = _network_settings(spec)
        resources = {
            "cpu": spec.cpu,
            "memory": max(1, math.ceil(spec.memory_mb / 1024)),
            "disk": max(3, math.ceil(spec.disk_mb / 1024)),
        }
        if sdk is None:
            params = _factory_create_params(
                self._client_factory,
                image=image,
                labels=expected_labels,
                resources=resources,
                network_block_all=network_block_all,
                domain_allow_list=domain_allow_list,
                public=False,
                ephemeral=True,
                auto_stop_interval=self._config.auto_stop_interval_minutes,
                env_vars={},
                secrets={},
            )
        else:
            params = sdk.CreateSandboxFromImageParams(
                image=image,
                labels=expected_labels,
                resources=sdk.Resources(**resources),
                network_block_all=network_block_all,
                domain_allow_list=domain_allow_list,
                public=False,
                ephemeral=True,
                auto_stop_interval=self._config.auto_stop_interval_minutes,
                env_vars={},
                secrets={},
            )

        sandbox: Any | None = None
        try:
            sandbox = await client.create(
                params,
                timeout=self._config.create_timeout_seconds,
            )
            self._owned_labels[sandbox.id] = expected_labels
            self._working_directories[sandbox.id] = working_directory
            setup = await sandbox.process.exec(
                f"mkdir -p -- {shlex.quote(working_directory)}",
                timeout=min(30, self._config.max_exec_timeout_seconds),
            )
            if setup.exit_code != 0:
                raise SandboxProviderError(
                    "workspace_setup_failed",
                    "Daytona created the sandbox but could not prepare its workspace.",
                )
        except BaseException as exc:
            if sandbox is not None:
                await self._best_effort_delete(client, sandbox)
                self._forget(sandbox.id)
            if isinstance(exc, asyncio.CancelledError):
                raise
            _raise_provider_error(exc)

        assert sandbox is not None
        return SandboxHandle(
            id=sandbox.id,
            provider_id=self.provider_id,
            state=_sandbox_state(getattr(sandbox, "state", None)),
            image_ref=spec.image_ref,
            image_digest=spec.image_digest,
        )

    async def upload(
        self,
        handle: SandboxHandle,
        file: SandboxFile,
    ) -> None:
        working_directory = self._working_directories.get(handle.id)
        if working_directory is None:
            raise SandboxOwnershipError
        path = resolve_sandbox_path(working_directory, file.path)
        content = file.content.encode("utf-8")
        if len(content) > self._config.max_file_bytes:
            raise SandboxProviderError(
                "file_too_large",
                "Sandbox upload exceeds the configured byte limit.",
            )
        sandbox = await self._owned_sandbox(handle)
        parent = str(PurePosixPath(path).parent)
        try:
            mkdir = await sandbox.process.exec(
                f"mkdir -p -- {shlex.quote(parent)}",
                timeout=min(30, self._config.max_exec_timeout_seconds),
            )
            if mkdir.exit_code != 0:
                raise SandboxProviderError(
                    "upload_failed",
                    "The remote upload directory could not be created.",
                )
            await sandbox.fs.upload_file(
                content,
                path,
                timeout=self._config.file_timeout_seconds,
            )
            if file.executable:
                chmod = await sandbox.process.exec(
                    f"chmod 700 -- {shlex.quote(path)}",
                    timeout=min(30, self._config.max_exec_timeout_seconds),
                )
                if chmod.exit_code != 0:
                    raise SandboxProviderError(
                        "upload_failed",
                        "The uploaded file could not be made executable.",
                    )
        except Exception as exc:
            _raise_provider_error(exc)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        timeout_seconds: int = 120,
    ) -> CommandResult:
        if not command or "\x00" in command or len(command) > 20_000:
            raise SandboxProviderError(
                "invalid_command",
                "Sandbox commands must be non-empty and at most 20000 characters.",
            )
        if not 0 < timeout_seconds <= self._config.max_exec_timeout_seconds:
            raise SandboxProviderError(
                "invalid_timeout",
                "Sandbox command timeout is outside the configured limit.",
            )
        sandbox = await self._owned_sandbox(handle)
        working_directory = cwd or self._working_directories.get(handle.id)
        if working_directory is not None:
            validate_remote_path(working_directory)
        started = monotonic()
        try:
            response = await sandbox.process.exec(
                command,
                cwd=working_directory,
                env={},
                timeout=timeout_seconds,
            )
        except Exception as exc:
            duration_ms = int((monotonic() - started) * 1000)
            error = _provider_error(exc)
            if error.code == "timeout":
                return CommandResult(
                    command=command,
                    exit_code=None,
                    stderr=error.message,
                    duration_ms=duration_ms,
                    timed_out=True,
                )
            raise SandboxProviderError(
                error.code,
                error.message,
                retryable=error.retryable,
            ) from exc

        output, truncated = truncate_utf8(
            str(response.result or ""),
            min(self._config.max_output_bytes, 200_000),
        )
        return CommandResult(
            command=command,
            exit_code=response.exit_code,
            stdout=output,
            stderr="",
            duration_ms=int((monotonic() - started) * 1000),
            timed_out=False,
            truncated=truncated,
        )

    async def read(
        self,
        handle: SandboxHandle,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        working_directory = self._working_directories.get(handle.id)
        if working_directory is None:
            raise SandboxOwnershipError
        path = resolve_sandbox_path(working_directory, path)
        limit = self._config.max_file_bytes
        if max_bytes is not None:
            if max_bytes < 0:
                raise SandboxProviderError(
                    "invalid_size_limit",
                    "Sandbox read byte limit cannot be negative.",
                )
            limit = min(limit, max_bytes)
        sandbox = await self._owned_sandbox(handle)
        try:
            info = await asyncio.wait_for(
                sandbox.fs.get_file_info(path),
                timeout=self._config.request_timeout_seconds,
            )
            if info.is_dir:
                raise SandboxProviderError(
                    "not_a_file",
                    "The requested sandbox path is a directory.",
                )
            if info.size > limit:
                raise SandboxProviderError(
                    "file_too_large",
                    "Sandbox file exceeds the configured read limit.",
                )
            content = await asyncio.wait_for(
                sandbox.fs.download_file(
                    path,
                    self._config.file_timeout_seconds,
                ),
                timeout=self._config.file_timeout_seconds + 1,
            )
        except Exception as exc:
            _raise_provider_error(exc)
        if content is None:
            raise SandboxProviderError(
                "read_failed",
                "Daytona returned no content for the requested sandbox file.",
            )
        result = content if isinstance(content, bytes) else str(content).encode()
        if len(result) > limit:
            raise SandboxProviderError(
                "file_too_large",
                "Sandbox file exceeds the configured read limit.",
            )
        return result

    async def delete(self, handle: SandboxHandle) -> None:
        self._ensure_open()
        if handle.provider_id != self.provider_id:
            raise SandboxOwnershipError
        if handle.id in self._deleted_ids:
            return
        expected = self._owned_labels.get(handle.id)
        if expected is None:
            raise SandboxOwnershipError
        await self._delete_owned_id(handle.id, expected)

    async def cleanup_owned(
        self,
        sandbox_id: str | None,
        ownership: SandboxOwnership,
    ) -> None:
        """Recover deletion using only durable IDs and exact ownership labels."""

        self._ensure_open()
        expected = ownership_labels(ownership, self.provider_id)
        if sandbox_id is not None:
            await self._delete_owned_id(sandbox_id, expected)
            return

        client = await self._client()
        sdk = _load_daytona_sdk() if self._client_factory is None else None
        query = (
            sdk.ListSandboxesQuery(labels=expected, limit=10)
            if sdk is not None
            else _factory_list_query(self._client_factory, expected)
        )
        iterator = client.list(query)
        matched = 0
        try:
            async for sandbox in iterator:
                if not labels_match(getattr(sandbox, "labels", None), expected):
                    continue
                matched += 1
                if matched > 10:
                    raise SandboxProviderError(
                        "cleanup_match_limit",
                        "Too many Daytona sandboxes matched persisted ownership.",
                    )
                await client.delete(
                    sandbox,
                    timeout=self._config.delete_timeout_seconds,
                )
                await self._wait_until_deleted(client, sandbox.id, expected)
                self._mark_deleted(sandbox.id)
        except Exception as exc:
            _raise_provider_error(exc)
        finally:
            closer = getattr(iterator, "aclose", None)
            if closer is not None:
                with suppress(Exception):
                    await closer()

    async def _delete_owned_id(
        self,
        sandbox_id: str,
        expected: Mapping[str, str],
    ) -> None:
        if sandbox_id in self._deleted_ids:
            return
        client = await self._client()
        try:
            sandbox = await asyncio.wait_for(
                client.get(sandbox_id),
                timeout=self._config.request_timeout_seconds,
            )
        except Exception as exc:
            if _is_not_found(exc):
                self._mark_deleted(sandbox_id)
                return
            _raise_provider_error(exc)
        if not labels_match(getattr(sandbox, "labels", None), expected):
            raise SandboxOwnershipError

        try:
            await client.delete(
                sandbox,
                timeout=self._config.delete_timeout_seconds,
            )
            await self._wait_until_deleted(client, sandbox_id, expected)
        except Exception as exc:
            _raise_provider_error(exc)
        self._mark_deleted(sandbox_id)

    async def aclose(self) -> None:
        async with self._client_lock:
            client = self._client_instance
            self._client_instance = None
            self._closed = True
        if client is not None:
            with suppress(Exception):
                await asyncio.wait_for(
                    client.close(),
                    timeout=self._config.request_timeout_seconds,
                )

    async def _client(self) -> Any:
        self._ensure_open()
        if not self._config.configured:
            raise SandboxProviderError(
                "not_configured",
                "Daytona credentials are not configured.",
            )
        async with self._client_lock:
            if self._client_instance is not None:
                return self._client_instance
            if self._client_factory is not None:
                client = self._client_factory(self._config)
            else:
                sdk = _load_daytona_sdk()
                kwargs: dict[str, Any] = {
                    "api_url": self._config.api_url,
                    "target": self._config.target,
                }
                if self._config.credential_mode is CredentialMode.API_KEY:
                    kwargs["api_key"] = _secret_value(self._config.api_key)
                else:
                    kwargs["jwt_token"] = _secret_value(self._config.jwt_token)
                    kwargs["organization_id"] = self._config.organization_id
                client = sdk.AsyncDaytona(sdk.DaytonaConfig(**kwargs))
            self._client_instance = client
            return client

    async def _owned_sandbox(self, handle: SandboxHandle) -> Any:
        self._ensure_open()
        if handle.provider_id != self.provider_id:
            raise SandboxOwnershipError
        expected = self._owned_labels.get(handle.id)
        if expected is None:
            raise SandboxOwnershipError
        client = await self._client()
        try:
            sandbox = await asyncio.wait_for(
                client.get(handle.id),
                timeout=self._config.request_timeout_seconds,
            )
        except Exception as exc:
            if _is_not_found(exc):
                raise SandboxNotFoundError(handle.id) from exc
            _raise_provider_error(exc)
        if not labels_match(getattr(sandbox, "labels", None), expected):
            raise SandboxOwnershipError
        return sandbox

    async def _wait_until_deleted(
        self,
        client: Any,
        sandbox_id: str,
        expected_labels: Mapping[str, str],
    ) -> None:
        deadline = monotonic() + self._config.delete_timeout_seconds
        while monotonic() < deadline:
            remaining = deadline - monotonic()
            try:
                sandbox = await asyncio.wait_for(
                    client.get(sandbox_id),
                    timeout=min(self._config.request_timeout_seconds, remaining),
                )
            except Exception as exc:
                if _is_not_found(exc):
                    return
                raise
            if _state_value(getattr(sandbox, "state", None)) == "destroyed":
                return
            if not labels_match(
                getattr(sandbox, "labels", None),
                expected_labels,
            ):
                raise SandboxOwnershipError
            await asyncio.sleep(min(1.0, max(0.05, remaining)))
        raise SandboxProviderError(
            "delete_timeout",
            "Daytona accepted deletion but destruction was not confirmed in time.",
            retryable=True,
        )

    async def _best_effort_delete(self, client: Any, sandbox: Any) -> None:
        with suppress(Exception):
            await asyncio.shield(
                client.delete(
                    sandbox,
                    timeout=self._config.delete_timeout_seconds,
                )
            )

    def _mark_deleted(self, sandbox_id: str) -> None:
        self._deleted_ids.add(sandbox_id)
        self._forget(sandbox_id)

    def _forget(self, sandbox_id: str) -> None:
        self._owned_labels.pop(sandbox_id, None)
        self._working_directories.pop(sandbox_id, None)

    def _ensure_open(self) -> None:
        if self._closed:
            raise SandboxProviderError(
                "provider_closed",
                "The Daytona provider is closed.",
            )


def _secret(value: str | None) -> SecretStr | None:
    return SecretStr(value) if value else None


def _secret_value(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value is not None else None


def _load_daytona_sdk() -> Any:
    try:
        installed = importlib.metadata.version("daytona")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SandboxProviderError(
            "dependency_missing",
            f"Daytona SDK {DAYTONA_SDK_VERSION} is not installed.",
        ) from exc
    if installed != DAYTONA_SDK_VERSION:
        raise SandboxProviderError(
            "dependency_version_mismatch",
            (
                f"Daytona SDK {DAYTONA_SDK_VERSION} is required; "
                "the installed version is incompatible."
            ),
        )
    try:
        return importlib.import_module("daytona")
    except ImportError as exc:
        raise SandboxProviderError(
            "dependency_import_failed",
            "The pinned Daytona SDK could not be imported.",
        ) from exc


def _factory_create_params(
    factory: Callable[[DaytonaProviderConfig], Any] | None,
    **kwargs: Any,
) -> Any:
    builder = getattr(factory, "create_params", None)
    if builder is None:
        return kwargs
    return builder(**kwargs)


def _factory_list_query(
    factory: Callable[[DaytonaProviderConfig], Any] | None,
    labels: Mapping[str, str],
) -> Any:
    builder = getattr(factory, "list_query", None)
    if builder is None:
        return {"labels": dict(labels), "limit": 10}
    return builder(labels=dict(labels), limit=10)


def _image_reference(spec: SandboxSpec) -> str:
    if not spec.image_digest:
        return spec.image_ref
    digest = spec.image_digest
    if "@" in spec.image_ref:
        if spec.image_ref.rsplit("@", 1)[1] != digest:
            raise SandboxProviderError(
                "image_digest_mismatch",
                "The sandbox image reference and digest disagree.",
            )
        return spec.image_ref
    return f"{spec.image_ref}@{digest}"


def _network_settings(spec: SandboxSpec) -> tuple[bool | None, str | None]:
    policy = str(spec.network_policy)
    if policy == "none":
        return True, None
    if policy == "public":
        return False, None
    if policy == "allowlist":
        hosts = [_validate_allowed_host(host) for host in spec.allowed_hosts]
        return None, ",".join(hosts)
    raise SandboxProviderError(
        "unsupported_network_policy",
        "The requested Daytona network policy is unsupported.",
    )


_HOST_PATTERN = re.compile(
    r"^(?:\*\.)?(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*"
    r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$"
)


def _validate_allowed_host(host: str) -> str:
    value = host.strip().lower()
    if len(value) > 253 or not _HOST_PATTERN.fullmatch(value):
        raise SandboxProviderError(
            "invalid_allowed_host",
            "Daytona allowlist entries must be plain DNS hostnames.",
        )
    return value


def _sandbox_state(state: Any) -> SandboxState:
    value = _state_value(state)
    if value in {"creating", "building", "pending"}:
        return SandboxState.CREATING
    if value in {"deleting", "destroying"}:
        return SandboxState.DELETING
    if value in {"deleted", "destroyed"}:
        return SandboxState.DELETED
    if value in {"error", "failed"}:
        return SandboxState.ERROR
    return SandboxState.READY


def _state_value(state: Any) -> str | None:
    if state is None:
        return None
    value = getattr(state, "value", state)
    return str(value).lower().rsplit(".", 1)[-1]


def _is_not_found(exc: BaseException) -> bool:
    return "notfound" in type(exc).__name__.lower()


def _provider_error(exc: BaseException) -> ProviderError:
    if isinstance(exc, SandboxProviderError):
        return exc.as_provider_error()
    name = type(exc).__name__.lower()
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name:
        return ProviderError(
            code="timeout",
            message="The Daytona request timed out.",
            retryable=True,
        )
    if "authentication" in name or "unauthorized" in name:
        return ProviderError(
            code="authentication_failed",
            message="Daytona rejected the configured credentials.",
            retryable=False,
        )
    if "authorization" in name or "forbidden" in name:
        return ProviderError(
            code="forbidden",
            message="The Daytona credential lacks the required permission.",
            retryable=False,
        )
    if "notfound" in name:
        return ProviderError(
            code="sandbox_not_found",
            message="The requested Daytona sandbox was not found.",
            retryable=False,
        )
    if any(token in name for token in ("connector", "connection", "dns")):
        return ProviderError(
            code="unreachable",
            message="The Daytona API could not be reached.",
            retryable=True,
        )
    if "validation" in name:
        return ProviderError(
            code="invalid_request",
            message="Daytona rejected an invalid sandbox request.",
            retryable=False,
        )
    return ProviderError(
        code="provider_error",
        message=f"Daytona request failed ({type(exc).__name__[:64]}).",
        retryable=False,
    )


def _raise_provider_error(exc: BaseException) -> None:
    if isinstance(exc, SandboxProviderError):
        raise exc
    error = _provider_error(exc)
    raise SandboxProviderError(
        error.code,
        error.message,
        retryable=error.retryable,
    ) from exc
