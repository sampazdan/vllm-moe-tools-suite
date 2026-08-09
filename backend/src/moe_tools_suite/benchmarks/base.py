from __future__ import annotations

from typing import Protocol

from ..domain import BenchmarkInfo, BenchmarkItem


class BenchmarkAdapter(Protocol):
    """Dataset, prompt, and scoring boundary for one benchmark revision."""

    @property
    def info(self) -> BenchmarkInfo: ...

    @property
    def content_hash(self) -> str: ...

    @property
    def scoring_version(self) -> str: ...

    def items(self) -> list[BenchmarkItem]: ...

    def render_prompt(self, item: BenchmarkItem) -> str: ...

    def score(self, item: BenchmarkItem, output: str) -> bool | None: ...
