from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..domain import (
    BenchmarkDatasetRecord,
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkKind,
    CreateCustomBenchmarkRequest,
    GenerationConfig,
    ScoringMode,
)

MAX_CUSTOM_ITEMS = 10_000
MAX_PROMPT_CHARACTERS = 100_000
MAX_EXPECTED_CHARACTERS = 20_000


class CustomBenchmarkAdapter:
    """Validated user-authored JSONL benchmark stored on the data volume."""

    def __init__(
        self,
        record: BenchmarkDatasetRecord,
        content: str,
    ) -> None:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash != record.content_hash:
            raise ValueError("custom benchmark content hash does not match metadata")
        self._record = record
        self._items = parse_custom_jsonl(content)
        if len(self._items) != record.info.item_count:
            raise ValueError("custom benchmark item count does not match metadata")

    @classmethod
    def create(
        cls,
        request: CreateCustomBenchmarkRequest,
    ) -> tuple[BenchmarkDatasetRecord, CustomBenchmarkAdapter]:
        items = parse_custom_jsonl(request.content)
        content_hash = hashlib.sha256(request.content.encode()).hexdigest()
        slug = _slugify(request.name)
        benchmark_id = f"custom-{slug}-{content_hash[:10]}"
        modes = {item.scoring for item in items}
        primary_scoring = next(iter(modes)) if len(modes) == 1 else ScoringMode.EXACT
        info = BenchmarkInfo(
            id=benchmark_id,
            name=request.name,
            description=request.description or "Imported custom JSONL benchmark.",
            item_count=len(items),
            categories=sorted({item.category for item in items}),
            kind=BenchmarkKind.CUSTOM,
            source="user-imported JSONL",
            revision=f"sha256:{content_hash}",
            split="custom",
            license="user supplied",
            scoring=primary_scoring,
            prompt_template_version="custom-jsonl-v1",
            default_generation=GenerationConfig(max_tokens=512),
        )
        record = BenchmarkDatasetRecord(
            id=benchmark_id,
            info=info,
            content_hash=content_hash,
        )
        return record, cls(record, request.content)

    @classmethod
    def from_file(
        cls,
        record: BenchmarkDatasetRecord,
        path: Path,
    ) -> CustomBenchmarkAdapter:
        if not path.is_file():
            raise ValueError("custom benchmark data file is missing")
        return cls(record, path.read_text(encoding="utf-8"))

    @property
    def info(self) -> BenchmarkInfo:
        return self._record.info

    @property
    def content_hash(self) -> str:
        return self._record.content_hash

    @property
    def scoring_version(self) -> str:
        return "custom-jsonl-scorer-v1"

    def items(self) -> list[BenchmarkItem]:
        return self._items

    def render_prompt(self, item: BenchmarkItem) -> str:
        return item.prompt

    def score(self, item: BenchmarkItem, output: str) -> bool | None:
        if item.scoring is ScoringMode.UNGRADED:
            return None
        actual = output.strip()
        expected = item.expected.strip()
        if item.scoring is ScoringMode.EXACT:
            return actual == expected
        if item.scoring is ScoringMode.CONTAINS:
            return expected in actual
        if item.scoring is ScoringMode.REGEX:
            return re.search(expected, actual) is not None
        raise ValueError(f"unsupported custom scoring mode {item.scoring}")


def parse_custom_jsonl(content: str) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    item_ids: set[str] = set()
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if len(items) >= MAX_CUSTOM_ITEMS:
            raise ValueError(f"custom benchmark exceeds {MAX_CUSTOM_ITEMS} items")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"line {line_number} is not valid JSON: {error.msg}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"line {line_number} requires a non-empty prompt")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError(f"line {line_number} prompt is too large")

        raw_id = payload.get("id", f"item-{line_number:05d}")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"line {line_number} has an invalid id")
        item_id = raw_id.strip()
        if len(item_id) > 120 or any(character.isspace() for character in item_id):
            raise ValueError(
                f"line {line_number} id must be at most 120 characters "
                "without whitespace"
            )
        if item_id in item_ids:
            raise ValueError(f"line {line_number} duplicates item id {item_id!r}")

        try:
            scoring = ScoringMode(payload.get("scoring", "exact"))
        except ValueError as error:
            raise ValueError(
                f"line {line_number} has an unsupported scoring mode"
            ) from error
        if scoring is ScoringMode.GSM8K:
            raise ValueError(f"line {line_number} cannot use the internal gsm8k scorer")
        expected = payload.get("expected", "")
        if not isinstance(expected, str):
            raise ValueError(f"line {line_number} expected must be a string")
        if scoring is not ScoringMode.UNGRADED and "expected" not in payload:
            raise ValueError(
                f"line {line_number} requires expected for {scoring.value} scoring"
            )
        if len(expected) > MAX_EXPECTED_CHARACTERS:
            raise ValueError(f"line {line_number} expected value is too large")
        if scoring is ScoringMode.REGEX:
            try:
                re.compile(expected)
            except re.error as error:
                raise ValueError(
                    f"line {line_number} has an invalid regular expression"
                ) from error

        category = payload.get("category", "general")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"line {line_number} category must be a string")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"line {line_number} metadata must be an object")

        item = BenchmarkItem(
            id=item_id,
            prompt=prompt.strip(),
            expected=expected,
            category=category.strip(),
            scoring=scoring,
            metadata=metadata,
        )
        items.append(item)
        item_ids.add(item_id)

    if not items:
        raise ValueError("custom benchmark must contain at least one JSONL item")
    return items


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (slug or "dataset")[:48]
