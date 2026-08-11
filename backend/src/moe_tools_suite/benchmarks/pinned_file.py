from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_SAFE_COMPONENT = re.compile(r"[a-z0-9][a-z0-9._-]{0,119}")


@dataclass(frozen=True)
class PinnedJsonlSource:
    """Immutable provenance for a small official JSONL artifact."""

    url: str
    revision: str
    expected_size_bytes: int
    sha256: str
    expected_rows: int
    cache_key: str
    file_name: str

    def __post_init__(self) -> None:
        parsed_url = urlparse(self.url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("pinned JSONL URL must use HTTPS")
        if _HEX_40.fullmatch(self.revision) is None:
            raise ValueError("revision must be a full lowercase commit SHA")
        if _HEX_64.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase SHA-256")
        if self.expected_size_bytes < 1 or self.expected_rows < 1:
            raise ValueError("pinned JSONL size and row count must be positive")
        if _SAFE_COMPONENT.fullmatch(self.cache_key) is None:
            raise ValueError("cache_key is not filesystem-safe")
        if _SAFE_COMPONENT.fullmatch(self.file_name) is None:
            raise ValueError("file_name is not filesystem-safe")


class PinnedJsonlFile:
    """Verified lazy cache for a revision-pinned JSONL source."""

    def __init__(self, data_dir: Path, source: PinnedJsonlSource) -> None:
        self.source = source
        self.path = (
            data_dir
            / "datasets"
            / "pinned-jsonl"
            / source.cache_key
            / source.revision
            / source.file_name
        )
        self._ready = self._is_verified_file()
        self._rows: list[Mapping[str, Any]] | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def content_hash(self) -> str:
        return self.source.sha256

    async def prepare(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if self._ready:
            if on_progress is not None:
                on_progress(
                    self.source.expected_size_bytes,
                    self.source.expected_size_bytes,
                )
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        owns_client = client is None
        http_client = client or httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(60, connect=20),
        )
        downloaded = 0
        digest = hashlib.sha256()
        try:
            async with http_client.stream("GET", self.source.url) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as output:
                    async for chunk in response.aiter_bytes():
                        if should_cancel is not None and should_cancel():
                            raise RuntimeError("dataset preparation cancelled")
                        downloaded += len(chunk)
                        if downloaded > self.source.expected_size_bytes:
                            raise ValueError("JSONL download exceeded its pinned size")
                        digest.update(chunk)
                        output.write(chunk)
                        if on_progress is not None:
                            on_progress(downloaded, self.source.expected_size_bytes)
            if downloaded != self.source.expected_size_bytes:
                raise ValueError("JSONL download size did not match pinned provenance")
            if digest.hexdigest() != self.source.sha256:
                raise ValueError(
                    "JSONL download checksum did not match pinned provenance"
                )
            rows = _read_jsonl(temporary_path)
            if len(rows) != self.source.expected_rows:
                raise ValueError("JSONL row count did not match pinned provenance")
            os.replace(temporary_path, self.path)
            self._ready = True
            self._rows = rows
        finally:
            temporary_path.unlink(missing_ok=True)
            if owns_client:
                await http_client.aclose()

    def rows(self) -> list[Mapping[str, Any]]:
        if not self._ready:
            raise RuntimeError("prepare the pinned JSONL source before reading it")
        if self._rows is None:
            self._rows = _read_jsonl(self.path)
            if len(self._rows) != self.source.expected_rows:
                raise ValueError("cached JSONL row count changed after verification")
        return self._rows

    def _is_verified_file(self) -> bool:
        if (
            not self.path.is_file()
            or self.path.stat().st_size != self.source.expected_size_bytes
        ):
            return False
        if _file_sha256(self.path) != self.source.sha256:
            return False
        try:
            return len(_read_jsonl(self.path)) == self.source.expected_rows
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return False


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"JSONL row {line_number} is blank")
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(payload)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
