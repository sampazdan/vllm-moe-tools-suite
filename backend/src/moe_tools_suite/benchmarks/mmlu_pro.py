from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..domain import (
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkKind,
    GenerationConfig,
    ScoringMode,
)
from .pinned_rows import PinnedRowsAdapter, PinnedRowsSource

MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_PRO_CURATED_ID = "mmlu-pro-validation-70"
MMLU_PRO_FULL_ID = "mmlu-pro-test-12032"
MMLU_PRO_CATEGORIES = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]

MMLU_PRO_VALIDATION_SOURCE = PinnedRowsSource(
    dataset="TIGER-Lab/MMLU-Pro",
    revision=MMLU_PRO_REVISION,
    config="default",
    split="validation",
    expected_rows=70,
    artifact_path="data/validation-00000-of-00001.parquet",
    artifact_sha256=(
        "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078"
    ),
    artifact_size_bytes=42_857,
    cache_key="mmlu-pro",
    canonical_sha256=(
        "09f5a3f6933273cfa8221bc8c45a13876dde6dec34280948ecf73f6905c8fd59"
    ),
)

MMLU_PRO_TEST_SOURCE = PinnedRowsSource(
    dataset="TIGER-Lab/MMLU-Pro",
    revision=MMLU_PRO_REVISION,
    config="default",
    split="test",
    expected_rows=12_032,
    artifact_path="data/test-00000-of-00001.parquet",
    artifact_sha256=(
        "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
    ),
    artifact_size_bytes=4_144_185,
    cache_key="mmlu-pro",
)

_FINAL_ANSWER_PATTERNS = (
    re.compile(r"final\s+answer\s*(?:is|:)?\s*[\[(]?([A-J])", re.I),
    re.compile(r"answer\s*:\s*[\[(]?([A-J])", re.I),
    re.compile(r"\\boxed\{\s*([A-J])\s*\}", re.I),
)
_STANDALONE_LETTER = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", re.I)


class MmluProAdapter(PinnedRowsAdapter):
    """Pinned MMLU-Pro validation or test split with answer-letter scoring."""

    def __init__(
        self,
        data_dir: Path,
        *,
        curated: bool,
        source: PinnedRowsSource | None = None,
    ) -> None:
        self.curated = curated
        selected_source = source or (
            MMLU_PRO_VALIDATION_SOURCE if curated else MMLU_PRO_TEST_SOURCE
        )
        super().__init__(data_dir, selected_source)

    @property
    def info(self) -> BenchmarkInfo:
        if self.curated:
            name = "MMLU-Pro · official validation 70"
            description = (
                "The complete official 70-question validation split: five "
                "questions in each of 14 domains. It is a cheap calibration "
                "cohort, not the MMLU-Pro leaderboard test set."
            )
            benchmark_id = MMLU_PRO_CURATED_ID
        else:
            name = "MMLU-Pro · full test"
            description = (
                "The complete official 12,032-question test split. Preparation "
                "is lazy and pages an exact Hugging Face dataset revision; expect "
                "a long, compute-intensive run after download."
            )
            benchmark_id = MMLU_PRO_FULL_ID
        return BenchmarkInfo(
            id=benchmark_id,
            name=name,
            description=description,
            item_count=self.source.expected_rows,
            categories=MMLU_PRO_CATEGORIES,
            kind=BenchmarkKind.STANDARD,
            source="TIGER-Lab/MMLU-Pro",
            revision=MMLU_PRO_REVISION,
            split=f"default/{self.source.split}",
            license="MIT (dataset); Apache-2.0 (reference evaluator)",
            ready=self.ready,
            scoring=ScoringMode.MULTIPLE_CHOICE,
            prompt_template_version="mmlu-pro-zero-shot-letter-v1",
            default_generation=GenerationConfig(max_tokens=1024),
        )

    @property
    def scoring_version(self) -> str:
        return "mmlu-pro-final-letter-v1"

    def row_to_item(
        self,
        row_index: int,
        row: Mapping[str, Any],
    ) -> BenchmarkItem:
        question_id = row.get("question_id")
        question = row.get("question")
        options = row.get("options")
        answer = row.get("answer")
        answer_index = row.get("answer_index")
        category = row.get("category")
        original_source = row.get("src")
        if not isinstance(question_id, int):
            raise ValueError("MMLU-Pro question_id must be an integer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("MMLU-Pro question must be a non-empty string")
        if (
            not isinstance(options, list)
            or not 2 <= len(options) <= 10
            or not all(isinstance(option, str) and option.strip() for option in options)
        ):
            raise ValueError("MMLU-Pro options must contain 2 to 10 strings")
        if not isinstance(answer_index, int) or not 0 <= answer_index < len(options):
            raise ValueError("MMLU-Pro answer_index is outside the option range")
        expected_letter = chr(ord("A") + answer_index)
        if answer != expected_letter:
            raise ValueError("MMLU-Pro answer letter does not match answer_index")
        if category not in MMLU_PRO_CATEGORIES:
            raise ValueError(f"MMLU-Pro returned unknown category {category!r}")
        if not isinstance(original_source, str):
            raise ValueError("MMLU-Pro src must be a string")
        split_label = "validation" if self.curated else "test"
        return BenchmarkItem(
            id=f"mmlu-pro-{split_label}-{question_id}",
            prompt=question.strip(),
            expected=expected_letter,
            category=category,
            scoring=ScoringMode.MULTIPLE_CHOICE,
            metadata={
                "source_index": row_index,
                "question_id": question_id,
                "answer_index": answer_index,
                "source_dataset": original_source,
                "options_json": _compact_json_strings(options),
                "artifact_sha256": self.source.artifact_sha256,
            },
        )

    def render_prompt(self, item: BenchmarkItem) -> str:
        raw_options = item.metadata.get("options_json")
        if not isinstance(raw_options, str):
            raise ValueError("MMLU-Pro item is missing its options")
        options = _parse_options(raw_options)
        rendered_options = "\n".join(
            f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
        )
        return (
            "Answer the following multiple-choice question. Reason carefully, "
            "then end with `Final answer: X`, where X is the single letter of "
            "the best option.\n\n"
            f"Question: {item.prompt}\n\n{rendered_options}"
        )

    def score(self, item: BenchmarkItem, output: str) -> bool:
        actual = extract_mmlu_pro_answer(output)
        return actual == item.expected


def mmlu_pro_adapters(data_dir: Path) -> tuple[MmluProAdapter, MmluProAdapter]:
    return (
        MmluProAdapter(data_dir, curated=True),
        MmluProAdapter(data_dir, curated=False),
    )


def extract_mmlu_pro_answer(output: str) -> str | None:
    for pattern in _FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(output)
        if matches:
            return matches[-1].upper()
    final_line = next(
        (line for line in reversed(output.splitlines()) if line.strip()),
        "",
    )
    matches = _STANDALONE_LETTER.findall(final_line)
    return matches[-1].upper() if matches else None


def _compact_json_strings(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _parse_options(value: str) -> list[str]:
    payload = json.loads(value)
    if not isinstance(payload, list) or not all(
        isinstance(option, str) for option in payload
    ):
        raise ValueError("MMLU-Pro options metadata is invalid")
    return payload
