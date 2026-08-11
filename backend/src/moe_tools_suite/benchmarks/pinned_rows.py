from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from ..domain import BenchmarkInfo, BenchmarkItem

HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
MAX_PAGE_BYTES = 20_000_000
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_SAFE_CACHE_KEY = re.compile(r"[a-z0-9][a-z0-9._-]{0,119}")


@dataclass(frozen=True)
class PinnedRowsSource:
    """Immutable Hugging Face rows source and its official artifact provenance."""

    dataset: str
    revision: str
    config: str
    split: str
    expected_rows: int
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    cache_key: str
    canonical_sha256: str | None = None
    page_size: int = 100

    def __post_init__(self) -> None:
        if not self.dataset or "/" not in self.dataset:
            raise ValueError("dataset must be a namespaced Hugging Face dataset ID")
        if _HEX_40.fullmatch(self.revision) is None:
            raise ValueError("revision must be a full lowercase commit SHA")
        if _HEX_64.fullmatch(self.artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be a lowercase SHA-256")
        if self.canonical_sha256 is not None and (
            _HEX_64.fullmatch(self.canonical_sha256) is None
        ):
            raise ValueError("canonical_sha256 must be a lowercase SHA-256")
        if self.expected_rows < 1:
            raise ValueError("expected_rows must be positive")
        if self.artifact_size_bytes < 1:
            raise ValueError("artifact_size_bytes must be positive")
        if not 1 <= self.page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        if _SAFE_CACHE_KEY.fullmatch(self.cache_key) is None:
            raise ValueError("cache_key is not filesystem-safe")
        if not self.config or not self.split or not self.artifact_path:
            raise ValueError("config, split, and artifact_path cannot be blank")

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(canonical).hexdigest()


class PinnedRowsAdapter(ABC):
    """Lazy, revision-pinned adapter backed by the official HF rows service."""

    def __init__(
        self,
        data_dir: Path,
        source: PinnedRowsSource,
    ) -> None:
        self.source = source
        root = data_dir / "datasets" / "pinned-rows" / source.cache_key
        self.path = root / source.revision / f"{source.split}.jsonl"
        self.manifest_path = self.path.with_suffix(".manifest.json")
        self._ready = self._is_verified_cache()
        self._items: list[BenchmarkItem] | None = None
        self._content_hash: str | None = (
            self._read_cache_manifest().get("canonical_sha256") if self._ready else None
        )

    @property
    @abstractmethod
    def info(self) -> BenchmarkInfo: ...

    @property
    @abstractmethod
    def scoring_version(self) -> str: ...

    @abstractmethod
    def row_to_item(self, row_index: int, row: Mapping[str, Any]) -> BenchmarkItem: ...

    @abstractmethod
    def render_prompt(self, item: BenchmarkItem) -> str: ...

    @abstractmethod
    def score(self, item: BenchmarkItem, output: str) -> bool | None: ...

    @property
    def content_hash(self) -> str:
        return (
            self._content_hash
            or self.source.canonical_sha256
            or self.source.artifact_sha256
        )

    async def prepare(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if self._ready:
            if on_progress is not None:
                on_progress(self.source.expected_rows, self.source.expected_rows)
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        nonce = uuid4().hex
        temporary_path = self.path.with_name(f".{self.path.name}.{nonce}.tmp")
        temporary_manifest = self.manifest_path.with_name(
            f".{self.manifest_path.name}.{nonce}.tmp"
        )
        owns_client = client is None
        http_client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(60, connect=20),
        )
        digest = hashlib.sha256()
        downloaded_rows = 0
        downloaded_bytes = 0
        try:
            with temporary_path.open("wb") as output:
                for offset in range(
                    0,
                    self.source.expected_rows,
                    self.source.page_size,
                ):
                    if should_cancel is not None and should_cancel():
                        raise RuntimeError("dataset preparation cancelled")
                    requested = min(
                        self.source.page_size,
                        self.source.expected_rows - offset,
                    )
                    payload = await self._fetch_page(
                        http_client,
                        offset=offset,
                        length=requested,
                    )
                    rows = self._validated_page(
                        payload,
                        offset=offset,
                        requested=requested,
                    )
                    for row in rows:
                        if should_cancel is not None and should_cancel():
                            raise RuntimeError("dataset preparation cancelled")
                        line = _canonical_row(row)
                        downloaded_bytes += len(line)
                        if downloaded_bytes > self._maximum_cache_bytes:
                            raise ValueError("canonical dataset exceeded its size cap")
                        digest.update(line)
                        output.write(line)
                        downloaded_rows += 1
                    if on_progress is not None:
                        on_progress(downloaded_rows, self.source.expected_rows)

            if downloaded_rows != self.source.expected_rows:
                raise ValueError("dataset row count did not match pinned provenance")
            canonical_sha256 = digest.hexdigest()
            if (
                self.source.canonical_sha256 is not None
                and canonical_sha256 != self.source.canonical_sha256
            ):
                raise ValueError(
                    "canonical dataset checksum did not match pinned provenance"
                )
            cache_manifest = {
                "schema_version": 1,
                "source_fingerprint": self.source.fingerprint,
                "canonical_sha256": canonical_sha256,
                "canonical_size_bytes": downloaded_bytes,
                "row_count": downloaded_rows,
            }
            temporary_manifest.write_text(
                json.dumps(cache_manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary_path, self.path)
            os.replace(temporary_manifest, self.manifest_path)
            self._ready = True
            self._content_hash = canonical_sha256
            self._items = None
        finally:
            temporary_path.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
            if owns_client:
                await http_client.aclose()

    def items(self) -> list[BenchmarkItem]:
        if not self._ready:
            raise RuntimeError(
                f"prepare {self.info.name} before browsing or running it"
            )
        if self._items is None:
            rows = self._load_rows()
            self._items = [
                self.row_to_item(row_index, row) for row_index, row in enumerate(rows)
            ]
            item_ids = [item.id for item in self._items]
            if len(item_ids) != len(set(item_ids)):
                raise ValueError("dataset parser produced duplicate item IDs")
        return self._items

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def _maximum_cache_bytes(self) -> int:
        return max(1_000_000, self.source.artifact_size_bytes * 12)

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        *,
        offset: int,
        length: int,
    ) -> Mapping[str, Any]:
        params = {
            "dataset": self.source.dataset,
            "config": self.source.config,
            "split": self.source.split,
            "offset": offset,
            "length": length,
            "revision": self.source.revision,
        }
        response: httpx.Response | None = None
        for attempt in range(5):
            response = await client.get(HF_ROWS_ENDPOINT, params=params)
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt == 4:
                break
            await asyncio.sleep(min(0.5 * (2**attempt), 4))
        assert response is not None
        response.raise_for_status()
        if len(response.content) > MAX_PAGE_BYTES:
            raise ValueError("dataset page exceeded the response size cap")
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("dataset page must be a JSON object")
        return payload

    def _validated_page(
        self,
        payload: Mapping[str, Any],
        *,
        offset: int,
        requested: int,
    ) -> list[Mapping[str, Any]]:
        if payload.get("num_rows_total") != self.source.expected_rows:
            raise ValueError("dataset service returned an unexpected total row count")
        raw_rows = payload.get("rows")
        if not isinstance(raw_rows, list) or len(raw_rows) != requested:
            raise ValueError("dataset service returned an incomplete page")
        rows: list[Mapping[str, Any]] = []
        for index, entry in enumerate(raw_rows, start=offset):
            if not isinstance(entry, Mapping) or entry.get("row_idx") != index:
                raise ValueError("dataset service returned non-contiguous row indices")
            row = entry.get("row")
            if not isinstance(row, Mapping):
                raise ValueError("dataset service returned a non-object row")
            rows.append(row)
        return rows

    def _load_rows(self) -> list[Mapping[str, Any]]:
        rows: list[Mapping[str, Any]] = []
        with self.path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"cached dataset row {line_number} is invalid JSON"
                    ) from error
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"cached dataset row {line_number} must be an object"
                    )
                rows.append(row)
        if len(rows) != self.source.expected_rows:
            raise ValueError("cached dataset row count changed after verification")
        return rows

    def _is_verified_cache(self) -> bool:
        try:
            manifest = self._read_cache_manifest()
            if not self.path.is_file():
                return False
            if manifest.get("schema_version") != 1:
                return False
            if manifest.get("source_fingerprint") != self.source.fingerprint:
                return False
            if manifest.get("row_count") != self.source.expected_rows:
                return False
            size = manifest.get("canonical_size_bytes")
            digest = manifest.get("canonical_sha256")
            if not isinstance(size, int) or size < 1:
                return False
            if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
                return False
            if self.path.stat().st_size != size:
                return False
            if self.source.canonical_sha256 not in {None, digest}:
                return False
            return _file_sha256(self.path) == digest
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _read_cache_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("cache manifest must contain a JSON object")
        return payload


def _canonical_row(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(row),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
