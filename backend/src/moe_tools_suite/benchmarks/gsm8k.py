from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

import httpx

from ..domain import (
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkKind,
    GenerationConfig,
    ScoringMode,
)

GSM8K_ID = "gsm8k-main-test"
GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
GSM8K_SIZE_BYTES = 749_738
GSM8K_ITEM_COUNT = 1_319
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_REVISION}/grade_school_math/data/test.jsonl"
)
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


class Gsm8kAdapter:
    """Pinned official GSM8K test split with local verified caching."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "datasets" / "gsm8k" / GSM8K_REVISION / "test.jsonl"
        self._items: list[BenchmarkItem] | None = None
        self._ready = self._is_verified_file()

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            id=GSM8K_ID,
            name="GSM8K · main test",
            description=(
                "The official 1,319-item grade-school math test split. "
                "Final numeric answers are extracted and scored exactly."
            ),
            item_count=GSM8K_ITEM_COUNT,
            categories=["grade-school math"],
            kind=BenchmarkKind.STANDARD,
            source="openai/grade-school-math",
            revision=GSM8K_REVISION,
            split="main/test",
            license="MIT",
            ready=self._ready,
            scoring=ScoringMode.GSM8K,
            prompt_template_version="gsm8k-reasoning-final-marker-v1",
            default_generation=GenerationConfig(max_tokens=768),
        )

    @property
    def content_hash(self) -> str:
        return GSM8K_SHA256

    @property
    def scoring_version(self) -> str:
        return "gsm8k-final-number-v1"

    async def prepare(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if self._ready:
            if on_progress is not None:
                on_progress(GSM8K_SIZE_BYTES, GSM8K_SIZE_BYTES)
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
            async with http_client.stream("GET", GSM8K_URL) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as output:
                    async for chunk in response.aiter_bytes():
                        if should_cancel is not None and should_cancel():
                            raise RuntimeError("dataset preparation cancelled")
                        downloaded += len(chunk)
                        if downloaded > GSM8K_SIZE_BYTES:
                            raise ValueError("GSM8K download exceeded expected size")
                        digest.update(chunk)
                        output.write(chunk)
                        if on_progress is not None:
                            on_progress(downloaded, GSM8K_SIZE_BYTES)
            if downloaded != GSM8K_SIZE_BYTES:
                raise ValueError(
                    "GSM8K download size did not match the pinned artifact"
                )
            if digest.hexdigest() != GSM8K_SHA256:
                raise ValueError(
                    "GSM8K download checksum did not match the pinned artifact"
                )
            os.replace(temporary_path, self.path)
            self._ready = True
            self._items = None
        finally:
            temporary_path.unlink(missing_ok=True)
            if owns_client:
                await http_client.aclose()

    def items(self) -> list[BenchmarkItem]:
        if not self._ready:
            raise RuntimeError("prepare GSM8K before browsing or running it")
        if self._items is None:
            self._items = self._load_items()
        return self._items

    def render_prompt(self, item: BenchmarkItem) -> str:
        return (
            "Solve the following grade-school math problem. Show concise "
            "reasoning, then finish with a final line in the exact form "
            "`#### <number>`.\n\n"
            f"Question: {item.prompt}"
        )

    def score(self, item: BenchmarkItem, output: str) -> bool:
        actual = extract_final_number(output)
        return actual is not None and numbers_equal(actual, item.expected)

    def _is_verified_file(self) -> bool:
        if not self.path.is_file() or self.path.stat().st_size != GSM8K_SIZE_BYTES:
            return False
        return hashlib.sha256(self.path.read_bytes()).hexdigest() == GSM8K_SHA256

    def _load_items(self) -> list[BenchmarkItem]:
        items: list[BenchmarkItem] = []
        with self.path.open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                payload = json.loads(line)
                question = payload["question"]
                answer = payload["answer"]
                expected = answer.rsplit("####", maxsplit=1)[-1].strip()
                question_hash = hashlib.sha256(question.encode()).hexdigest()[:10]
                items.append(
                    BenchmarkItem(
                        id=f"gsm8k-test-{index:04d}-{question_hash}",
                        prompt=question,
                        expected=expected,
                        category="grade-school math",
                        scoring=ScoringMode.GSM8K,
                        metadata={"source_index": index},
                    )
                )
        if len(items) != GSM8K_ITEM_COUNT:
            raise ValueError("GSM8K item count did not match the pinned manifest")
        return items


def extract_final_number(output: str) -> str | None:
    candidate = output.rsplit("####", maxsplit=1)[-1]
    matches = _NUMBER_PATTERN.findall(candidate)
    if not matches and "####" not in output:
        matches = _NUMBER_PATTERN.findall(output)
    return matches[-1] if matches else None


def numbers_equal(left: str, right: str) -> bool:
    normalized_left = left.replace(",", "").strip()
    normalized_right = right.replace(",", "").strip()
    try:
        return Decimal(normalized_left) == Decimal(normalized_right)
    except InvalidOperation:
        return normalized_left == normalized_right
