#!/opt/vllm-venv/bin/python
"""Emit a bounded, read-only Daytona inventory without exposing credentials.

The helper is intended to run inside the disposable MoE Atelier Pod. It accepts
Daytona configuration only through its own inherited process environment. It
never accepts credentials through arguments, never reads another process,
never serializes SDK objects, and contains no mutation or deletion operation.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import logging
import os
import re
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

DAYTONA_SDK_VERSION = "0.192.0"
SCHEMA_VERSION = "moe-atelier-v2-daytona-inventory/v1"
SAFE_LABEL_KEYS = (
    "moe-tools-managed",
    "moe-tools-provider",
    "moe-tools-controller",
    "moe-tools-run",
    "moe-tools-trial",
)
SAFE_STATE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


class InventoryError(RuntimeError):
    """A secret-free inventory failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _enum_value(value: Any) -> str:
    candidate = getattr(value, "value", value)
    text = str(candidate or "unknown")
    return text if SAFE_STATE_PATTERN.fullmatch(text) else "unknown"


def _safe_text(value: Any, secret: str, *, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).replace(secret, "[REDACTED]")
    text = "".join(character for character in text if character.isprintable())
    return text[:limit]


def _read_process_environment() -> tuple[dict[str, str], str]:
    environment = dict(os.environ)
    secret = environment.get("MOE_TOOLS_DAYTONA_API_KEY", "")
    if not secret or "RUNPOD_SECRET_" in secret or secret.startswith("REPLACE"):
        raise InventoryError("daytona_credential_unavailable")
    return environment, secret


def _daytona_configuration(environment: dict[str, str]) -> tuple[str, str]:
    api_url = environment.get(
        "MOE_TOOLS_DAYTONA_API_URL", "https://app.daytona.io/api"
    ).rstrip("/")
    target = environment.get("MOE_TOOLS_DAYTONA_TARGET", "us")
    parsed = urlsplit(api_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise InventoryError("invalid_daytona_api_url")
    if target not in {"us", "eu"}:
        raise InventoryError("invalid_daytona_target")
    return api_url, target


def _sandbox_record(sandbox: Any, secret: str) -> dict[str, Any]:
    labels = getattr(sandbox, "labels", None)
    if not isinstance(labels, dict):
        labels = {}
    selected_labels = {
        key: _safe_text(labels[key], secret, limit=256)
        for key in SAFE_LABEL_KEYS
        if key in labels
    }
    return {
        "id": _safe_text(getattr(sandbox, "id", None), secret, limit=500),
        "name": _safe_text(getattr(sandbox, "name", None), secret, limit=191),
        "state": _enum_value(getattr(sandbox, "state", None)),
        "target": _safe_text(getattr(sandbox, "target", None), secret, limit=80),
        "created_at": _safe_text(
            getattr(sandbox, "created_at", None), secret, limit=80
        ),
        "labels": selected_labels,
    }


def _is_moe_tools_managed(record: dict[str, Any]) -> bool:
    labels = record["labels"]
    return (
        labels.get("moe-tools-managed") == "true"
        and labels.get("moe-tools-provider") == "daytona"
    )


async def _inventory(
    *,
    api_key: str,
    api_url: str,
    target: str,
    timeout_seconds: float,
    max_sandboxes: int,
) -> list[dict[str, Any]]:
    try:
        installed_version = importlib.metadata.version("daytona")
    except importlib.metadata.PackageNotFoundError as error:
        raise InventoryError("daytona_sdk_missing") from error
    if installed_version != DAYTONA_SDK_VERSION:
        raise InventoryError("daytona_sdk_version_mismatch")

    import daytona

    client = daytona.AsyncDaytona(
        daytona.DaytonaConfig(api_key=api_key, api_url=api_url, target=target)
    )
    records: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            iterator = client.list()
            try:
                async for sandbox in iterator:
                    if len(records) >= max_sandboxes:
                        raise InventoryError("sandbox_inventory_limit_exceeded")
                    records.append(_sandbox_record(sandbox, api_key))
            finally:
                closer = getattr(iterator, "aclose", None)
                if closer is not None:
                    with suppress(Exception):
                        await closer()
    except TimeoutError as error:
        raise InventoryError("daytona_inventory_timeout") from error
    finally:
        with suppress(Exception):
            await client.close()
    return sorted(records, key=lambda record: str(record.get("id") or ""))


def _serialize(report: dict[str, Any], secret: str | None) -> bytes:
    payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    if secret:
        encoded_secret = secret.encode("utf-8")
        escaped_secret = json.dumps(secret, ensure_ascii=False)[1:-1].encode("utf-8")
        if encoded_secret in payload or escaped_secret in payload:
            raise InventoryError("secret_redaction_check_failed")
    return payload


def _write_report(path: Path | None, payload: bytes) -> None:
    if path is None:
        os.write(1, payload)
        return
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a read-only, secret-safe provider-wide Daytona inventory."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-sandboxes", type=int, default=10_000)
    return parser


def main() -> int:
    os.umask(0o077)
    logging.disable(logging.CRITICAL)
    arguments = build_parser().parse_args()
    if not 1 <= arguments.timeout <= 300:
        raise SystemExit("--timeout must be between 1 and 300 seconds")
    if not 1 <= arguments.max_sandboxes <= 100_000:
        raise SystemExit("--max-sandboxes must be between 1 and 100000")

    checked_at = _utc_now()
    secret: str | None = None
    try:
        environment, secret = _read_process_environment()
        api_url, target = _daytona_configuration(environment)
        records = asyncio.run(
            _inventory(
                api_key=secret,
                api_url=api_url,
                target=target,
                timeout_seconds=arguments.timeout,
                max_sandboxes=arguments.max_sandboxes,
            )
        )
        managed = [record for record in records if _is_moe_tools_managed(record)]
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "checked_at": checked_at,
            "success": True,
            "provider": "daytona",
            "sdk_version": DAYTONA_SDK_VERSION,
            "target": target,
            "provider_wide_empty": not records,
            "sandbox_count": len(records),
            "moe_tools_managed_count": len(managed),
            "unmanaged_count": len(records) - len(managed),
            "sandboxes": records,
        }
        exit_code = 0
    except InventoryError as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "checked_at": checked_at,
            "success": False,
            "provider": "daytona",
            "error": {"code": error.code},
        }
        exit_code = 1
    except Exception:
        report = {
            "schema_version": SCHEMA_VERSION,
            "checked_at": checked_at,
            "success": False,
            "provider": "daytona",
            "error": {"code": "daytona_inventory_failed"},
        }
        exit_code = 1

    try:
        payload = _serialize(report, secret)
    except InventoryError:
        payload = _serialize(
            {
                "schema_version": SCHEMA_VERSION,
                "checked_at": checked_at,
                "success": False,
                "provider": "daytona",
                "error": {"code": "secret_redaction_check_failed"},
            },
            None,
        )
        exit_code = 1
    _write_report(arguments.output, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
