from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from ..domain import (
    BenchmarkDatasetRecord,
    BenchmarkInfo,
    BenchmarkItemPage,
    CreateCustomBenchmarkRequest,
)
from .base import BenchmarkAdapter
from .custom import CustomBenchmarkAdapter
from .fixture import FixtureArithmeticAdapter
from .gsm8k import Gsm8kAdapter
from .ifeval import ifeval_adapters
from .livebench import LivebenchZebra202406Adapter
from .mmlu_pro import mmlu_pro_adapters


class BenchmarkCatalog:
    """Runtime registry for bundled, downloaded, and imported benchmarks."""

    def __init__(
        self,
        data_dir: Path,
        custom_records: dict[str, BenchmarkDatasetRecord],
        *,
        max_custom_dataset_bytes: int,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.max_custom_dataset_bytes = max_custom_dataset_bytes
        fixture = FixtureArithmeticAdapter()
        gsm8k = Gsm8kAdapter(self.data_dir)
        standard_adapters: list[BenchmarkAdapter] = [
            gsm8k,
            *ifeval_adapters(self.data_dir),
            LivebenchZebra202406Adapter(self.data_dir),
            *mmlu_pro_adapters(self.data_dir),
        ]
        self._adapters: dict[str, BenchmarkAdapter] = {
            fixture.info.id: fixture,
            **{adapter.info.id: adapter for adapter in standard_adapters},
        }
        self._unavailable: dict[str, BenchmarkInfo] = {}
        for record in custom_records.values():
            path = self.custom_path(record.id)
            try:
                adapter = CustomBenchmarkAdapter.from_file(record, path)
            except (OSError, ValueError):
                self._unavailable[record.id] = record.info.model_copy(
                    update={"ready": False}
                )
            else:
                self._adapters[record.id] = adapter

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        infos = [adapter.info for adapter in self._adapters.values()]
        infos.extend(self._unavailable.values())
        return sorted(
            infos,
            key=lambda info: (
                {"fixture": 0, "standard": 1, "custom": 2}[info.kind.value],
                info.name.casefold(),
            ),
        )

    def get_info(self, benchmark_id: str) -> BenchmarkInfo:
        adapter = self._adapters.get(benchmark_id)
        if adapter is not None:
            return adapter.info
        info = self._unavailable.get(benchmark_id)
        if info is not None:
            return info
        raise KeyError(benchmark_id)

    def get_adapter(self, benchmark_id: str) -> BenchmarkAdapter:
        adapter = self._adapters.get(benchmark_id)
        if adapter is not None and adapter.info.ready:
            return adapter
        if benchmark_id in self._unavailable or adapter is not None:
            raise RuntimeError(f"benchmark {benchmark_id!r} is not prepared")
        raise KeyError(benchmark_id)

    def list_items(
        self,
        benchmark_id: str,
        *,
        search: str = "",
        category: str | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> BenchmarkItemPage:
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        adapter = self.get_adapter(benchmark_id)
        all_items = adapter.items()
        normalized_search = search.strip().casefold()
        filtered = [
            item
            for item in all_items
            if (category is None or item.category == category)
            and (
                not normalized_search
                or normalized_search in item.id.casefold()
                or normalized_search in item.prompt.casefold()
                or normalized_search in item.category.casefold()
            )
        ]
        return BenchmarkItemPage(
            benchmark=adapter.info,
            items=filtered[offset : offset + limit],
            total=len(filtered),
            offset=offset,
            limit=limit,
            categories=sorted({item.category for item in all_items}),
        )

    async def prepare(
        self,
        benchmark_id: str,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> BenchmarkInfo:
        adapter = self._adapters.get(benchmark_id)
        if adapter is None:
            raise KeyError(benchmark_id)
        prepare = getattr(adapter, "prepare", None)
        if prepare is None:
            return adapter.info
        await prepare(
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        return adapter.info

    def import_custom(
        self,
        request: CreateCustomBenchmarkRequest,
    ) -> BenchmarkDatasetRecord:
        encoded = request.content.encode()
        if len(encoded) > self.max_custom_dataset_bytes:
            raise ValueError(
                "custom benchmark exceeds the configured upload size limit"
            )
        record, adapter = CustomBenchmarkAdapter.create(request)
        path = self.custom_path(record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_bytes(encoded)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._adapters[record.id] = adapter
        self._unavailable.pop(record.id, None)
        return record

    def custom_path(self, benchmark_id: str) -> Path:
        path = (
            self.data_dir / "datasets" / "custom" / f"{benchmark_id}.jsonl"
        ).resolve()
        custom_root = (self.data_dir / "datasets" / "custom").resolve()
        if not path.is_relative_to(custom_root):
            raise ValueError("custom benchmark path escaped the data directory")
        return path
