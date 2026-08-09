from __future__ import annotations

import hashlib
import json

from ..domain import (
    BenchmarkInfo,
    BenchmarkItem,
    BenchmarkKind,
    GenerationConfig,
    ScoringMode,
)


class FixtureArithmeticAdapter:
    """Small deterministic adapter for GPU-free application tests."""

    def __init__(self) -> None:
        self._items = _fixture_items()
        canonical = json.dumps(
            [item.model_dump(mode="json") for item in self._items],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self._content_hash = hashlib.sha256(canonical).hexdigest()

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            id="fixture-arithmetic",
            name="Arithmetic routing fixture",
            description=(
                "Deterministic GPU-free prompts used to exercise scoring, "
                "routing aggregation, and expert-profile creation."
            ),
            item_count=len(self._items),
            categories=sorted({item.category for item in self._items}),
            kind=BenchmarkKind.FIXTURE,
            source="bundled",
            revision="fixture-arithmetic-v1",
            split="fixture",
            license="internal test fixture",
            scoring=ScoringMode.EXACT,
            prompt_template_version="integer-only-v1",
            default_generation=GenerationConfig(max_tokens=16),
        )

    @property
    def content_hash(self) -> str:
        return self._content_hash

    @property
    def scoring_version(self) -> str:
        return "trimmed-exact-v1"

    def items(self) -> list[BenchmarkItem]:
        return self._items

    def render_prompt(self, item: BenchmarkItem) -> str:
        return item.prompt

    def score(self, item: BenchmarkItem, output: str) -> bool:
        return output.strip() == item.expected.strip()


def _fixture_items() -> list[BenchmarkItem]:
    expressions = [
        ("03", "12 + 7", "19", "addition"),
        ("04", "42 - 19", "23", "subtraction"),
        ("05", "9 * 8", "72", "multiplication"),
        ("06", "31 + 46", "77", "addition"),
        ("07", "100 - 37", "63", "subtraction"),
        ("08", "13 * 6", "78", "multiplication"),
        ("09", "-4 + 15", "11", "addition"),
        ("10", "81 - 99", "-18", "subtraction"),
        ("11", "17 * 5", "85", "multiplication"),
        ("12", "128 + 64", "192", "addition"),
        ("13", "72 - 18", "54", "subtraction"),
        ("14", "21 * 4", "84", "multiplication"),
    ]
    return [
        BenchmarkItem(
            id=f"arith-{item_id}",
            prompt=f"Return only the integer result of {expression}.",
            expected=expected,
            category=category,
        )
        for item_id, expression, expected, category in expressions
    ]
