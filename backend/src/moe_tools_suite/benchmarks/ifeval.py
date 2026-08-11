from __future__ import annotations

import collections
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
from .pinned_file import PinnedJsonlFile, PinnedJsonlSource

IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
IFEVAL_EVALUATOR_REVISION = "26d8ccdab6fec61b5c83ad6327ea8bda9e580288"
IFEVAL_STRICT_ID = "ifeval-strict-dependency-free-387"
IFEVAL_FULL_ID = "ifeval-full-541-reference"
IFEVAL_STRICT_KEYS_SHA256 = (
    "b4c91435ca1e16e983256a3ba856a9032c139944922126cdde755ea3728dcad8"
)
IFEVAL_SOURCE = PinnedJsonlSource(
    url=(
        "https://huggingface.co/datasets/google/IFEval/resolve/"
        f"{IFEVAL_REVISION}/ifeval_input_data.jsonl"
    ),
    revision=IFEVAL_REVISION,
    expected_size_bytes=207_111,
    sha256="6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
    expected_rows=541,
    cache_key="google-ifeval",
    file_name="ifeval_input_data.jsonl",
)

UNSUPPORTED_CANONICAL_INSTRUCTIONS = frozenset(
    {
        "change_case:capital_word_frequency",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "language:response_language",
        "length_constraints:number_sentences",
    }
)
SUPPORTED_CANONICAL_INSTRUCTIONS = frozenset(
    {
        "combination:repeat_prompt",
        "combination:two_responses",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
        "detectable_format:constrained_response",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:title",
        "keywords:existence",
        "keywords:forbidden_words",
        "keywords:frequency",
        "keywords:letter_frequency",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
        "length_constraints:number_words",
        "punctuation:no_comma",
        "startend:end_checker",
        "startend:quotation",
    }
)
ALL_CANONICAL_INSTRUCTIONS = (
    SUPPORTED_CANONICAL_INSTRUCTIONS | UNSUPPORTED_CANONICAL_INSTRUCTIONS
)
_CONSTRAINED_RESPONSES = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)


class IfevalAdapter:
    """Pinned IFEval strict subset or full telemetry-only corpus."""

    def __init__(self, corpus: PinnedJsonlFile, *, strict_subset: bool) -> None:
        self.corpus = corpus
        self.strict_subset = strict_subset
        self._items: list[BenchmarkItem] | None = None

    @property
    def info(self) -> BenchmarkInfo:
        if self.strict_subset:
            benchmark_id = IFEVAL_STRICT_ID
            name = "IFEval · dependency-free strict 387"
            description = (
                "An application-specific 387-prompt cohort from the official "
                "541-prompt corpus. It keeps only 20 instruction families whose "
                "upstream strict checks can run deterministically without language "
                "detection or sentence-model assets. All constraints must pass. "
                "This is not an official IFEval aggregate and does not report the "
                "upstream loose or instruction-level metrics."
            )
            item_count = 387
            categories = sorted(SUPPORTED_CANONICAL_INSTRUCTIONS)
            scoring = ScoringMode.IFEVAL
        else:
            benchmark_id = IFEVAL_FULL_ID
            name = "IFEval · full 541 reference"
            description = (
                "The complete official 541-prompt corpus for routing telemetry and "
                "qualitative inspection. It is intentionally ungraded until all 25 "
                "canonical evaluator families and official aggregate metrics are "
                "integrated; do not treat runs as IFEval scores."
            )
            item_count = self.corpus.source.expected_rows
            categories = sorted(ALL_CANONICAL_INSTRUCTIONS)
            scoring = ScoringMode.UNGRADED
        return BenchmarkInfo(
            id=benchmark_id,
            name=name,
            description=description,
            item_count=item_count,
            categories=categories,
            kind=BenchmarkKind.STANDARD,
            source="google/IFEval",
            revision=IFEVAL_REVISION,
            split="official input corpus",
            license="Apache-2.0",
            ready=self.corpus.ready,
            scoring=scoring,
            prompt_template_version="ifeval-upstream-prompt-verbatim-v1",
            default_generation=GenerationConfig(max_tokens=2048),
        )

    @property
    def content_hash(self) -> str:
        return self.corpus.content_hash

    @property
    def scoring_version(self) -> str:
        if self.strict_subset:
            return (
                "ifeval-google-strict-prompt-dependency-free-v1-"
                f"{IFEVAL_EVALUATOR_REVISION[:12]}"
            )
        return "ifeval-reference-ungraded-v1"

    async def prepare(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        await self.corpus.prepare(
            on_progress=on_progress,
            should_cancel=should_cancel,
            client=client,
        )
        self._items = None

    def items(self) -> list[BenchmarkItem]:
        if not self.corpus.ready:
            raise RuntimeError("prepare IFEval before browsing or running it")
        if self._items is None:
            rows = self.corpus.rows()
            parsed = [
                _ifeval_item(row_index, row, scored=self.strict_subset)
                for row_index, row in enumerate(rows)
            ]
            if self.strict_subset:
                parsed = [
                    item
                    for item in parsed
                    if _item_instruction_ids(item) <= SUPPORTED_CANONICAL_INSTRUCTIONS
                ]
                if len(parsed) != 387:
                    raise ValueError("IFEval strict cohort membership changed")
                key_payload = json.dumps(
                    [item.metadata["source_key"] for item in parsed],
                    separators=(",", ":"),
                ).encode()
                if hashlib.sha256(key_payload).hexdigest() != (
                    IFEVAL_STRICT_KEYS_SHA256
                ):
                    raise ValueError("IFEval strict cohort fingerprint changed")
            item_ids = [item.id for item in parsed]
            if len(item_ids) != len(set(item_ids)):
                raise ValueError("IFEval returned duplicate item IDs")
            self._items = parsed
        return self._items

    def render_prompt(self, item: BenchmarkItem) -> str:
        return item.prompt

    def score(self, item: BenchmarkItem, output: str) -> bool | None:
        if not self.strict_subset:
            return None
        instruction_ids = _ordered_item_instruction_ids(item)
        kwargs = _item_kwargs(item)
        if len(instruction_ids) != len(kwargs):
            raise ValueError("IFEval item has misaligned instruction arguments")
        return strict_ifeval_score(instruction_ids, kwargs, output)


def ifeval_adapters(data_dir: Path) -> tuple[IfevalAdapter, IfevalAdapter]:
    corpus = PinnedJsonlFile(data_dir, IFEVAL_SOURCE)
    return (
        IfevalAdapter(corpus, strict_subset=True),
        IfevalAdapter(corpus, strict_subset=False),
    )


def strict_ifeval_score(
    instruction_ids: list[str],
    kwargs: list[Mapping[str, Any]],
    output: str,
) -> bool:
    """Apply the dependency-free subset of Google's strict IFEval semantics.

    The checker behavior is ported from google-research at
    ``IFEVAL_EVALUATOR_REVISION`` under Apache-2.0. Application benchmark runs
    normalize outer response whitespace before reaching this function, so the
    resulting cohort is deliberately not labeled official-score compatible.
    """

    if not output.strip() or len(instruction_ids) != len(kwargs):
        return False
    return all(
        _check_instruction(instruction_id, arguments, output)
        for instruction_id, arguments in zip(instruction_ids, kwargs, strict=True)
    )


def _check_instruction(
    instruction_id: str,
    arguments: Mapping[str, Any],
    output: str,
) -> bool:
    checker = _CHECKERS.get(instruction_id)
    if checker is None:
        raise ValueError(f"unsupported strict IFEval instruction {instruction_id!r}")
    return checker(output, arguments)


def _ifeval_item(
    row_index: int,
    row: Mapping[str, Any],
    *,
    scored: bool,
) -> BenchmarkItem:
    key = row.get("key")
    prompt = row.get("prompt")
    instruction_ids = row.get("instruction_id_list")
    kwargs = row.get("kwargs")
    if not isinstance(key, int):
        raise ValueError("IFEval key must be an integer")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("IFEval prompt must be a non-empty string")
    if (
        not isinstance(instruction_ids, list)
        or not instruction_ids
        or not all(isinstance(value, str) for value in instruction_ids)
    ):
        raise ValueError("IFEval instruction_id_list must contain strings")
    if not set(instruction_ids) <= ALL_CANONICAL_INSTRUCTIONS:
        raise ValueError("IFEval returned an unknown instruction family")
    if (
        not isinstance(kwargs, list)
        or len(kwargs) != len(instruction_ids)
        or not all(isinstance(value, Mapping) for value in kwargs)
    ):
        raise ValueError("IFEval kwargs must align with instruction IDs")
    return BenchmarkItem(
        id=f"ifeval-{key}",
        prompt=prompt,
        expected="Satisfy every declared instruction." if scored else "",
        category=instruction_ids[0],
        scoring=ScoringMode.IFEVAL if scored else ScoringMode.UNGRADED,
        metadata={
            "source_index": row_index,
            "source_key": key,
            "instruction_count": len(instruction_ids),
            "instruction_ids_json": json.dumps(
                instruction_ids,
                separators=(",", ":"),
            ),
            "instruction_kwargs_json": json.dumps(
                kwargs,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "evaluator_revision": IFEVAL_EVALUATOR_REVISION,
        },
    )


def _item_instruction_ids(item: BenchmarkItem) -> frozenset[str]:
    raw = item.metadata.get("instruction_ids_json")
    if not isinstance(raw, str):
        raise ValueError("IFEval item is missing instruction IDs")
    payload = json.loads(raw)
    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise ValueError("IFEval instruction metadata is invalid")
    return frozenset(payload)


def _ordered_item_instruction_ids(item: BenchmarkItem) -> list[str]:
    raw = item.metadata.get("instruction_ids_json")
    if not isinstance(raw, str):
        raise ValueError("IFEval item is missing instruction IDs")
    payload = json.loads(raw)
    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise ValueError("IFEval instruction metadata is invalid")
    return payload


def _item_kwargs(item: BenchmarkItem) -> list[Mapping[str, Any]]:
    raw = item.metadata.get("instruction_kwargs_json")
    if not isinstance(raw, str):
        raise ValueError("IFEval item is missing instruction arguments")
    payload = json.loads(raw)
    if not isinstance(payload, list) or not all(
        isinstance(value, Mapping) for value in payload
    ):
        raise ValueError("IFEval instruction arguments are invalid")
    return payload


def _relation(actual: int, arguments: Mapping[str, Any], key: str) -> bool:
    threshold = arguments.get(key)
    relation = arguments.get("relation")
    if not isinstance(threshold, int) or relation not in {"less than", "at least"}:
        raise ValueError("IFEval relation arguments are invalid")
    return actual < threshold if relation == "less than" else actual >= threshold


def _keywords(output: str, arguments: Mapping[str, Any]) -> bool:
    return all(
        re.search(keyword, output, flags=re.IGNORECASE) is not None
        for keyword in _string_list(arguments, "keywords")
    )


def _forbidden_words(output: str, arguments: Mapping[str, Any]) -> bool:
    return all(
        re.search(r"\b" + word + r"\b", output, flags=re.IGNORECASE) is None
        for word in _string_list(arguments, "forbidden_words")
    )


def _keyword_frequency(output: str, arguments: Mapping[str, Any]) -> bool:
    keyword = _string(arguments, "keyword").strip()
    occurrences = len(re.findall(keyword, output, flags=re.IGNORECASE))
    return _relation(occurrences, arguments, "frequency")


def _letter_frequency(output: str, arguments: Mapping[str, Any]) -> bool:
    letter = _string(arguments, "letter").strip().lower()
    frequency = arguments.get("let_frequency")
    relation = arguments.get("let_relation")
    if len(letter) != 1 or not isinstance(frequency, int):
        raise ValueError("IFEval letter-frequency arguments are invalid")
    if relation not in {"less than", "at least"}:
        raise ValueError("IFEval letter-frequency relation is invalid")
    actual = collections.Counter(output.lower())[letter]
    return actual < frequency if relation == "less than" else actual >= frequency


def _number_words(output: str, arguments: Mapping[str, Any]) -> bool:
    return _relation(len(re.findall(r"\w+", output)), arguments, "num_words")


def _number_placeholders(output: str, arguments: Mapping[str, Any]) -> bool:
    minimum = arguments.get("num_placeholders")
    if not isinstance(minimum, int):
        raise ValueError("IFEval placeholder count is invalid")
    return len(re.findall(r"\[.*?\]", output)) >= minimum


def _number_bullets(output: str, arguments: Mapping[str, Any]) -> bool:
    expected = arguments.get("num_bullets")
    if not isinstance(expected, int):
        raise ValueError("IFEval bullet count is invalid")
    stars = re.findall(r"^\s*\*[^\*].*$", output, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", output, flags=re.MULTILINE)
    return len(stars) + len(dashes) == expected


def _constrained_response(output: str, arguments: Mapping[str, Any]) -> bool:
    if arguments:
        raise ValueError("IFEval constrained response takes no arguments")
    value = output.strip()
    return any(option in value for option in _CONSTRAINED_RESPONSES)


def _number_highlights(output: str, arguments: Mapping[str, Any]) -> bool:
    minimum = arguments.get("num_highlights")
    if not isinstance(minimum, int):
        raise ValueError("IFEval highlight count is invalid")
    count = sum(
        bool(value.strip("*").strip()) for value in re.findall(r"\*[^\n\*]*\*", output)
    )
    count += sum(
        bool(value.removeprefix("**").removesuffix("**").strip())
        for value in re.findall(r"\*\*[^\n\*]*\*\*", output)
    )
    return count >= minimum


def _multiple_sections(output: str, arguments: Mapping[str, Any]) -> bool:
    splitter = _string(arguments, "section_spliter").strip()
    minimum = arguments.get("num_sections")
    if not isinstance(minimum, int):
        raise ValueError("IFEval section count is invalid")
    pattern = r"\s?" + splitter + r"\s?\d+\s?"
    return len(re.split(pattern, output)) - 1 >= minimum


def _paragraphs(output: str, arguments: Mapping[str, Any]) -> bool:
    expected = arguments.get("num_paragraphs")
    if not isinstance(expected, int):
        raise ValueError("IFEval paragraph count is invalid")
    paragraphs = re.split(r"\s?\*\*\*\s?", output)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index in {0, len(paragraphs) - 1}:
                count -= 1
            else:
                return False
    return count == expected


def _nth_paragraph_first_word(output: str, arguments: Mapping[str, Any]) -> bool:
    expected_count = arguments.get("num_paragraphs")
    nth = arguments.get("nth_paragraph")
    expected_word = _string(arguments, "first_word").lower()
    if not isinstance(expected_count, int) or not isinstance(nth, int):
        raise ValueError("IFEval paragraph-first-word arguments are invalid")
    paragraphs = re.split(r"\n\n", output)
    count = sum(bool(paragraph.strip()) for paragraph in paragraphs)
    if nth > count:
        return False
    paragraph = paragraphs[nth - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    first_word = ""
    for letter in word:
        if letter in {".", ",", "?", "!", "'", '"'}:
            break
        first_word += letter.lower()
    return count == expected_count and first_word == expected_word


def _postscript(output: str, arguments: Mapping[str, Any]) -> bool:
    marker = _string(arguments, "postscript_marker").strip()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pattern, output.lower(), flags=re.MULTILINE))


def _json_format(output: str, arguments: Mapping[str, Any]) -> bool:
    if arguments:
        raise ValueError("IFEval JSON format takes no arguments")
    value = (
        output.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _title(output: str, arguments: Mapping[str, Any]) -> bool:
    if arguments:
        raise ValueError("IFEval title takes no arguments")
    return any(
        value.lstrip("<").rstrip(">").strip()
        for value in re.findall(r"<<[^\n]+>>", output)
    )


def _two_responses(output: str, arguments: Mapping[str, Any]) -> bool:
    if arguments:
        raise ValueError("IFEval two-responses takes no arguments")
    valid: list[str] = []
    responses = output.split("******")
    for index, response in enumerate(responses):
        if not response.strip():
            if index not in {0, len(responses) - 1}:
                return False
        else:
            valid.append(response)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _repeat_prompt(output: str, arguments: Mapping[str, Any]) -> bool:
    prompt = _string(arguments, "prompt_to_repeat")
    return output.strip().lower().startswith(prompt.strip().lower())


def _end_checker(output: str, arguments: Mapping[str, Any]) -> bool:
    end_phrase = _string(arguments, "end_phrase").strip().lower()
    return output.strip().strip('"').lower().endswith(end_phrase)


def _no_comma(output: str, arguments: Mapping[str, Any]) -> bool:
    if arguments:
        raise ValueError("IFEval no-comma takes no arguments")
    return re.search(r"\,", output) is None


def _quotation(output: str, arguments: Mapping[str, Any]) -> bool:
    if arguments:
        raise ValueError("IFEval quotation takes no arguments")
    value = output.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


def _string(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ValueError(f"IFEval argument {key!r} must be a string")
    return value


def _string_list(arguments: Mapping[str, Any], key: str) -> list[str]:
    value = arguments.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"IFEval argument {key!r} must contain strings")
    return value


_CHECKERS: dict[str, Callable[[str, Mapping[str, Any]], bool]] = {
    "combination:repeat_prompt": _repeat_prompt,
    "combination:two_responses": _two_responses,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:json_format": _json_format,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:number_bullet_lists": _number_bullets,
    "detectable_format:number_highlighted_sections": _number_highlights,
    "detectable_format:title": _title,
    "keywords:existence": _keywords,
    "keywords:forbidden_words": _forbidden_words,
    "keywords:frequency": _keyword_frequency,
    "keywords:letter_frequency": _letter_frequency,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "length_constraints:number_paragraphs": _paragraphs,
    "length_constraints:number_words": _number_words,
    "punctuation:no_comma": _no_comma,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
}

assert frozenset(_CHECKERS) == SUPPORTED_CANONICAL_INSTRUCTIONS
