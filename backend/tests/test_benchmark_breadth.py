from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from moe_tools_suite.benchmarks.catalog import BenchmarkCatalog
from moe_tools_suite.benchmarks.ifeval import (
    IFEVAL_FULL_ID,
    IFEVAL_SOURCE,
    IFEVAL_STRICT_ID,
    IFEVAL_STRICT_KEYS_SHA256,
    strict_ifeval_score,
)
from moe_tools_suite.benchmarks.livebench import (
    LIVEBENCH_REASONING_SOURCE,
    LIVEBENCH_ZEBRA_2024_06_ID,
    LIVEBENCH_ZEBRA_KEYS_SHA256,
    livebench_zebra_legacy_score,
)
from moe_tools_suite.benchmarks.mmlu_pro import (
    MMLU_PRO_CURATED_ID,
    MMLU_PRO_FULL_ID,
    MmluProAdapter,
    extract_mmlu_pro_answer,
)
from moe_tools_suite.benchmarks.pinned_file import (
    PinnedJsonlFile,
    PinnedJsonlSource,
)
from moe_tools_suite.benchmarks.pinned_rows import PinnedRowsSource
from moe_tools_suite.domain import ScoringMode
from moe_tools_suite.lab import MODEL_ID
from moe_tools_suite.main import create_app
from moe_tools_suite.settings import Settings

REVISION = "1" * 40
ARTIFACT_HASH = "2" * 64


def _row(
    question_id: int,
    *,
    category: str,
    answer_index: int,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "options": ["first", "second", "third"],
        "answer": chr(ord("A") + answer_index),
        "answer_index": answer_index,
        "cot_content": "",
        "category": category,
        "src": f"fixture-{category}",
    }


def _canonical_hash(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            + b"\n"
        )
    return digest.hexdigest()


def _source(rows: list[dict[str, object]]) -> PinnedRowsSource:
    return PinnedRowsSource(
        dataset="example/MMLU-Pro",
        revision=REVISION,
        config="default",
        split="validation",
        expected_rows=len(rows),
        artifact_path="data/validation.parquet",
        artifact_sha256=ARTIFACT_HASH,
        artifact_size_bytes=1024,
        cache_key="mmlu-pro-test",
        canonical_sha256=_canonical_hash(rows),
        page_size=1,
    )


@pytest.mark.asyncio
async def test_mmlu_pro_preparation_is_pinned_paged_and_restart_safe(
    tmp_path: Path,
) -> None:
    rows = [
        _row(10, category="biology", answer_index=1),
        _row(11, category="business", answer_index=2),
    ]
    source = _source(rows)
    seen_offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "datasets-server.huggingface.co"
        assert request.url.params["dataset"] == source.dataset
        assert request.url.params["revision"] == REVISION
        assert request.url.params["split"] == "validation"
        offset = int(request.url.params["offset"])
        seen_offsets.append(offset)
        return httpx.Response(
            200,
            json={
                "num_rows_total": len(rows),
                "rows": [{"row_idx": offset, "row": rows[offset]}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = MmluProAdapter(tmp_path, curated=True, source=source)
    progress: list[tuple[int, int]] = []

    assert adapter.info.ready is False
    await adapter.prepare(
        client=client,
        on_progress=lambda current, total: progress.append((current, total)),
    )

    assert seen_offsets == [0, 1]
    assert progress == [(1, 2), (2, 2)]
    assert adapter.info.ready is True
    assert adapter.content_hash == source.canonical_sha256
    items = adapter.items()
    assert [item.id for item in items] == [
        "mmlu-pro-validation-10",
        "mmlu-pro-validation-11",
    ]
    assert "A. first" in adapter.render_prompt(items[0])
    assert adapter.score(items[0], "Reasoning.\nFinal answer: B") is True
    assert adapter.info.scoring is ScoringMode.MULTIPLE_CHOICE
    assert all(item.scoring is ScoringMode.MULTIPLE_CHOICE for item in items)

    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "app-data",
            frontend_dist=tmp_path / "missing-frontend",
        )
    )
    app.state.lab.benchmark_catalog._adapters[adapter.info.id] = adapter
    with TestClient(app) as test_client:
        model_job = test_client.post(
            "/api/model-sessions",
            json={"model_id": MODEL_ID},
        ).json()
        assert test_client.get(f"/api/jobs/{model_job['id']}").json()["status"] == (
            "completed"
        )
        problem = test_client.get(
            f"/api/benchmarks/{adapter.info.id}/items/{items[0].id}"
        ).json()
        assert problem["benchmark"]["scoring"] == "multiple_choice"
        assert problem["item"]["scoring"] == "multiple_choice"
        assert "final answer letter A–J" in problem["success_criteria"]["summary"]
        assert (
            problem["success_criteria"]["public_contract"]["criteria"][0]["kind"]
            == "benchmark_default"
        )

        run_job = test_client.post(
            "/api/runs",
            json={"benchmark_id": adapter.info.id, "item_ids": [items[0].id]},
        ).json()
        completed = test_client.get(f"/api/jobs/{run_job['id']}").json()
        detail = test_client.get(f"/api/runs/{completed['result_id']}/detail").json()
        assert detail["provenance"]["scoring_version"] == ("mmlu-pro-final-letter-v1")
        assert (
            "final answer letter A–J"
            in (
                detail["provenance"]["evaluation_contract"]["criteria"][0][
                    "description"
                ]
            )
        )

    restarted = MmluProAdapter(tmp_path, curated=True, source=source)
    assert restarted.info.ready is True
    assert restarted.content_hash == source.canonical_sha256
    assert [item.expected for item in restarted.items()] == ["B", "C"]
    await client.aclose()


@pytest.mark.asyncio
async def test_pinned_rows_rejects_source_drift_without_publishing_cache(
    tmp_path: Path,
) -> None:
    rows = [_row(10, category="biology", answer_index=1)]
    source = _source(rows)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "num_rows_total": 2,
                "rows": [{"row_idx": 0, "row": rows[0]}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = MmluProAdapter(tmp_path, curated=True, source=source)

    with pytest.raises(ValueError, match="unexpected total row count"):
        await adapter.prepare(client=client)

    assert adapter.info.ready is False
    assert not adapter.path.exists()
    assert not adapter.manifest_path.exists()
    await client.aclose()


def test_pinned_rows_cache_detects_corruption_on_restart(tmp_path: Path) -> None:
    rows = [_row(10, category="biology", answer_index=1)]
    source = _source(rows)
    adapter = MmluProAdapter(tmp_path, curated=True, source=source)
    adapter.path.parent.mkdir(parents=True)
    canonical = (
        json.dumps(rows[0], sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    adapter.path.write_bytes(canonical)
    adapter.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_fingerprint": source.fingerprint,
                "canonical_sha256": source.canonical_sha256,
                "canonical_size_bytes": len(canonical),
                "row_count": 1,
            }
        )
    )
    adapter.path.write_text("corrupt\n")

    restarted = MmluProAdapter(tmp_path, curated=True, source=source)

    assert restarted.info.ready is False


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("work\nFinal answer: (C)", "C"),
        ("The answer: j", "J"),
        (r"Therefore \\boxed{F}", "F"),
        ("Reasoning mentions A and B\nD", "D"),
        ("No final option", None),
    ],
)
def test_mmlu_pro_answer_extraction(output: str, expected: str | None) -> None:
    assert extract_mmlu_pro_answer(output) == expected


def test_catalog_exposes_curated_and_full_mmlu_pro_as_unprepared(
    tmp_path: Path,
) -> None:
    catalog = BenchmarkCatalog(
        tmp_path,
        {},
        max_custom_dataset_bytes=1_000_000,
    )
    info_by_id = {info.id: info for info in catalog.list_benchmarks()}

    assert info_by_id[MMLU_PRO_CURATED_ID].item_count == 70
    assert info_by_id[MMLU_PRO_CURATED_ID].ready is False
    assert info_by_id[MMLU_PRO_CURATED_ID].scoring is ScoringMode.MULTIPLE_CHOICE
    assert "not the MMLU-Pro leaderboard" in (
        info_by_id[MMLU_PRO_CURATED_ID].description
    )
    assert info_by_id[MMLU_PRO_FULL_ID].item_count == 12_032
    assert info_by_id[MMLU_PRO_FULL_ID].ready is False


@pytest.mark.asyncio
async def test_pinned_jsonl_preparation_verifies_bytes_rows_and_restart(
    tmp_path: Path,
) -> None:
    rows = [
        {"id": 1, "value": "first"},
        {"id": 2, "value": "second"},
    ]
    content = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    ).encode()
    source = PinnedJsonlSource(
        url="https://example.test/dataset.jsonl",
        revision=REVISION,
        expected_size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        expected_rows=2,
        cache_key="fixture-jsonl",
        file_name="dataset.jsonl",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == source.url
        return httpx.Response(200, content=content)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    corpus = PinnedJsonlFile(tmp_path, source)
    progress: list[tuple[int, int]] = []

    await corpus.prepare(
        client=client,
        on_progress=lambda current, total: progress.append((current, total)),
    )

    assert progress[-1] == (len(content), len(content))
    assert corpus.rows() == rows
    assert PinnedJsonlFile(tmp_path, source).ready is True

    corpus.path.write_text("corrupt")
    assert PinnedJsonlFile(tmp_path, source).ready is False
    await client.aclose()


@pytest.mark.parametrize(
    ("instruction_id", "arguments", "passing", "failing"),
    [
        ("keywords:existence", {"keywords": ["alpha"]}, "Alpha here", "none"),
        (
            "keywords:forbidden_words",
            {"forbidden_words": ["cat"]},
            "catalog",
            "a cat",
        ),
        (
            "keywords:frequency",
            {"keyword": "go", "frequency": 2, "relation": "at least"},
            "go go",
            "go",
        ),
        (
            "keywords:letter_frequency",
            {"letter": "a", "let_frequency": 2, "let_relation": "at least"},
            "aardvark",
            "bird",
        ),
        (
            "length_constraints:number_words",
            {"num_words": 3, "relation": "at least"},
            "one two three",
            "one two",
        ),
        (
            "detectable_content:number_placeholders",
            {"num_placeholders": 2},
            "[name] [date]",
            "[name]",
        ),
        (
            "detectable_format:number_bullet_lists",
            {"num_bullets": 2},
            "* first\n- second",
            "* first",
        ),
        (
            "detectable_format:constrained_response",
            {},
            "My answer is yes.",
            "Certainly.",
        ),
        (
            "detectable_format:number_highlighted_sections",
            {"num_highlights": 2},
            "*first* and *second*",
            "*first*",
        ),
        (
            "detectable_format:multiple_sections",
            {"section_spliter": "Section", "num_sections": 2},
            "Section 1 one\nSection 2 two",
            "Section 1 one",
        ),
        (
            "length_constraints:number_paragraphs",
            {"num_paragraphs": 2},
            "first *** second",
            "only one",
        ),
        (
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "hello"},
            "first\n\nHello world",
            "first\n\nGoodbye world",
        ),
        (
            "detectable_content:postscript",
            {"postscript_marker": "P.S."},
            "body\nP.S. note",
            "body only",
        ),
        (
            "detectable_format:json_format",
            {},
            '```json\n{"ok":true}\n```',
            "not json",
        ),
        ("detectable_format:title", {}, "<<A real title>>", "No title"),
        (
            "combination:two_responses",
            {},
            "first******second",
            "first******first",
        ),
        (
            "combination:repeat_prompt",
            {"prompt_to_repeat": "Repeat this"},
            "Repeat this\nanswer",
            "preface Repeat this",
        ),
        (
            "startend:end_checker",
            {"end_phrase": "Done."},
            '"work\nDone."',
            "Done. more",
        ),
        ("punctuation:no_comma", {}, "no commas here", "one, comma"),
        ("startend:quotation", {}, '"wrapped"', "not wrapped"),
    ],
)
def test_dependency_free_ifeval_checkers_match_strict_reference_semantics(
    instruction_id: str,
    arguments: dict[str, object],
    passing: str,
    failing: str,
) -> None:
    assert strict_ifeval_score([instruction_id], [arguments], passing) is True
    assert strict_ifeval_score([instruction_id], [arguments], failing) is False


def test_ifeval_strict_scorer_requires_every_constraint_and_rejects_unknown() -> None:
    instruction_ids = ["keywords:existence", "punctuation:no_comma"]
    kwargs = [{"keywords": ["required"]}, {}]

    assert strict_ifeval_score(instruction_ids, kwargs, "required text") is True
    assert strict_ifeval_score(instruction_ids, kwargs, "required, text") is False
    assert strict_ifeval_score(instruction_ids, kwargs, "") is False
    with pytest.raises(ValueError, match="unsupported strict IFEval"):
        strict_ifeval_score(["language:response_language"], [{}], "English")


def test_catalog_exposes_honest_ifeval_scored_and_reference_modes(
    tmp_path: Path,
) -> None:
    catalog = BenchmarkCatalog(
        tmp_path,
        {},
        max_custom_dataset_bytes=1_000_000,
    )
    info_by_id = {info.id: info for info in catalog.list_benchmarks()}

    strict = info_by_id[IFEVAL_STRICT_ID]
    full = info_by_id[IFEVAL_FULL_ID]
    assert strict.item_count == 387
    assert strict.scoring.value == "ifeval"
    assert "not an official IFEval aggregate" in strict.description
    assert full.item_count == 541
    assert full.scoring.value == "ungraded"
    assert "do not treat runs as IFEval scores" in full.description
    assert strict.ready is full.ready is False
    assert len(IFEVAL_SOURCE.sha256) == 64
    assert len(IFEVAL_STRICT_KEYS_SHA256) == 64


@pytest.mark.parametrize(
    ("ground_truth", "output", "expected"),
    [
        ("rabbit", "reasoning\n***rabbit***", True),
        ("journalist", "The final answer is journalist.", True),
        ("two", "***2***", True),
        ("drama movies", "***drama***", True),
        ("rabbit", "***rabbit*** then ***horse***", False),
        ("rabbit", "No answer supplied.", False),
    ],
)
def test_livebench_zebra_archive_uses_release_specific_boolean_scorer(
    ground_truth: str,
    output: str,
    expected: bool,
) -> None:
    assert livebench_zebra_legacy_score(ground_truth, output) is expected


def test_catalog_exposes_pinned_livebench_archive_without_leaderboard_claim(
    tmp_path: Path,
) -> None:
    catalog = BenchmarkCatalog(
        tmp_path,
        {},
        max_custom_dataset_bytes=1_000_000,
    )
    info_by_id = {info.id: info for info in catalog.list_benchmarks()}
    info = info_by_id[LIVEBENCH_ZEBRA_2024_06_ID]

    assert info.item_count == 50
    assert info.ready is False
    assert info.scoring.value == "livebench"
    assert "not a current LiveBench leaderboard aggregate" in info.description
    assert len(LIVEBENCH_REASONING_SOURCE.canonical_sha256 or "") == 64
    assert len(LIVEBENCH_ZEBRA_KEYS_SHA256) == 64
