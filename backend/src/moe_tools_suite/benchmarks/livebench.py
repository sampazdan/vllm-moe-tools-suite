from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from ..domain import (
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkKind,
    GenerationConfig,
    ScoringMode,
)
from .pinned_rows import PinnedRowsAdapter, PinnedRowsSource

LIVEBENCH_REASONING_REVISION = "6fc6498a5dfba553f69f4413feabade1f1a2d384"
LIVEBENCH_EVALUATOR_REVISION = "1fde4a4d3e58b3e1654a0f0dc6de84fead6f0c8e"
LIVEBENCH_ZEBRA_2024_06_ID = "livebench-zebra-2024-06-50"
LIVEBENCH_ZEBRA_KEYS_SHA256 = (
    "8f1119cfdda60f538ccba00d2ae7cbaf75d9a512c64df7c6082b012ab06b2e94"
)
LIVEBENCH_REASONING_SOURCE = PinnedRowsSource(
    dataset="livebench/reasoning",
    revision=LIVEBENCH_REASONING_REVISION,
    config="default",
    split="test",
    expected_rows=200,
    artifact_path="data/test-00000-of-00001.parquet",
    artifact_sha256=(
        "4204bb94c812690ef8ba5f4c1f10b5b1082ca0b7bc532166834f798aa56e2a3c"
    ),
    artifact_size_bytes=88_219,
    cache_key="livebench-reasoning",
    canonical_sha256=(
        "529e96272ca1071e5c900953efee379b4b41ebb144553029eba2912d7345e5c1"
    ),
)


class LivebenchZebra202406Adapter(PinnedRowsAdapter):
    """Pinned 2024-06 LiveBench zebra archive with its release scorer."""

    def __init__(
        self,
        data_dir: Path,
        source: PinnedRowsSource | None = None,
    ) -> None:
        super().__init__(data_dir, source or LIVEBENCH_REASONING_SOURCE)
        self._cohort_items: list[BenchmarkItem] | None = None

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            id=LIVEBENCH_ZEBRA_2024_06_ID,
            name="LiveBench archive · zebra puzzle 2024-06",
            description=(
                "A reproducibly pinned 50-question zebra-puzzle cohort from "
                "LiveBench's 2024-06-24 release, scored with the official "
                "release-specific boolean answer extractor. These questions "
                "were later retired; this application cohort is not a current "
                "LiveBench leaderboard aggregate."
            ),
            item_count=50,
            categories=["zebra puzzle"],
            kind=BenchmarkKind.STANDARD,
            source="livebench/reasoning",
            revision=LIVEBENCH_REASONING_REVISION,
            split="default/test · zebra_puzzle · 2024-06-24",
            license="Apache-2.0",
            ready=self.ready,
            scoring=ScoringMode.LIVEBENCH,
            prompt_template_version="livebench-upstream-prompt-verbatim-v1",
            default_generation=GenerationConfig(max_tokens=2048),
        )

    @property
    def scoring_version(self) -> str:
        return f"livebench-zebra-old-{LIVEBENCH_EVALUATOR_REVISION[:12]}"

    async def prepare(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        await super().prepare(
            on_progress=on_progress,
            should_cancel=should_cancel,
            client=client,
        )
        self._cohort_items = None

    def items(self) -> list[BenchmarkItem]:
        if self._cohort_items is None:
            cohort = [
                item
                for item in super().items()
                if item.metadata.get("task") == "zebra_puzzle"
                and item.metadata.get("release_date") == "2024-06-24T00:00:00"
            ]
            if len(cohort) != 50:
                raise ValueError("LiveBench zebra archive membership changed")
            key_payload = json.dumps(
                [item.metadata["question_id"] for item in cohort],
                separators=(",", ":"),
            ).encode()
            if hashlib.sha256(key_payload).hexdigest() != (LIVEBENCH_ZEBRA_KEYS_SHA256):
                raise ValueError("LiveBench zebra archive fingerprint changed")
            self._cohort_items = cohort
        return self._cohort_items

    def row_to_item(
        self,
        row_index: int,
        row: Mapping[str, Any],
    ) -> BenchmarkItem:
        question_id = row.get("question_id")
        category = row.get("category")
        ground_truth = row.get("ground_truth")
        turns = row.get("turns")
        task = row.get("task")
        release_date = row.get("livebench_release_date")
        removal_date = row.get("livebench_removal_date")
        level = row.get("level")
        if (
            not isinstance(question_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", question_id) is None
        ):
            raise ValueError("LiveBench question_id must be a SHA-256")
        if category != "reasoning":
            raise ValueError("LiveBench reasoning row returned another category")
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise ValueError("LiveBench ground truth must be a non-empty string")
        if (
            not isinstance(turns, list)
            or len(turns) != 1
            or not isinstance(turns[0], str)
            or not turns[0].strip()
        ):
            raise ValueError("LiveBench reasoning row must have one prompt turn")
        if task not in {"zebra_puzzle", "spatial", "web_of_lies_v2"}:
            raise ValueError("LiveBench reasoning row returned an unknown task")
        if not isinstance(release_date, str) or not isinstance(removal_date, str):
            raise ValueError("LiveBench release metadata must be strings")
        if level is not None and not isinstance(level, int):
            raise ValueError("LiveBench level must be an integer or null")
        return BenchmarkItem(
            id=f"livebench-zebra-{question_id}",
            prompt=turns[0],
            expected=ground_truth,
            category="zebra puzzle",
            scoring=ScoringMode.LIVEBENCH,
            metadata={
                "source_index": row_index,
                "question_id": question_id,
                "task": task,
                "release_date": release_date,
                "removal_date": removal_date,
                "level": level,
                "evaluator_revision": LIVEBENCH_EVALUATOR_REVISION,
            },
        )

    def render_prompt(self, item: BenchmarkItem) -> str:
        return item.prompt

    def score(self, item: BenchmarkItem, output: str) -> bool:
        return livebench_zebra_legacy_score(item.expected, output)


def livebench_zebra_legacy_score(ground_truth: str, output: str) -> bool:
    """Port the official pre-2024-11-25 zebra-puzzle boolean scorer."""

    number_to_word = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    bold_words = re.findall(r"\*\*\*(\w+)\*\*\*", output)
    if bold_words:
        candidate = bold_words[-1]
    else:
        words = re.findall(r"\b\w+\b", output)
        candidate = words[-1] if words else ""
    normalized_candidate = candidate.lower()
    normalized_truth = ground_truth.lower()
    return (
        normalized_candidate == normalized_truth
        or (
            candidate in number_to_word
            and number_to_word[candidate].lower() == normalized_truth
        )
        or normalized_candidate + " movies" == normalized_truth
    )
