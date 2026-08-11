import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from moe_tools_suite.benchmarks.custom import CustomBenchmarkAdapter
from moe_tools_suite.benchmarks.gsm8k import (
    Gsm8kAdapter,
    extract_final_number,
    numbers_equal,
)
from moe_tools_suite.domain import (
    CreateCustomBenchmarkRequest,
    CreateModelSessionRequest,
    ExpertProfile,
    GenerationConfig,
)
from moe_tools_suite.lab import MODEL_ID, ResearchLab
from moe_tools_suite.main import create_app
from moe_tools_suite.runtime import CompletionResult
from moe_tools_suite.settings import Settings
from moe_tools_suite.telemetry import DecodedRouting


def test_custom_jsonl_supports_graded_and_ungraded_items() -> None:
    content = "\n".join(
        [
            json.dumps(
                {
                    "id": "exact",
                    "prompt": "Return 4",
                    "expected": "4",
                    "category": "unit",
                }
            ),
            json.dumps(
                {
                    "id": "contains",
                    "prompt": "Explain",
                    "expected": "needle",
                    "scoring": "contains",
                }
            ),
            json.dumps(
                {
                    "id": "regex",
                    "prompt": "Format",
                    "expected": r"answer: \d+",
                    "scoring": "regex",
                }
            ),
            json.dumps(
                {
                    "id": "agent-trace",
                    "prompt": "Try a workflow",
                    "scoring": "ungraded",
                    "metadata": {"harness": "daytona"},
                }
            ),
        ]
    )
    record, adapter = CustomBenchmarkAdapter.create(
        CreateCustomBenchmarkRequest(name="Mixed workflow", content=content)
    )

    items = adapter.items()
    assert record.id.startswith("custom-mixed-workflow-")
    assert record.content_hash == hashlib.sha256(content.encode()).hexdigest()
    assert adapter.score(items[0], " 4\n") is True
    assert adapter.score(items[1], "found a needle here") is True
    assert adapter.score(items[2], "answer: 123") is True
    assert adapter.score(items[3], "arbitrary trace") is None


def test_custom_benchmark_regex_times_out_fail_closed() -> None:
    content = json.dumps(
        {
            "id": "catastrophic",
            "prompt": "Return text",
            "expected": r"(a+)+$",
            "scoring": "regex",
        }
    )
    _, adapter = CustomBenchmarkAdapter.create(
        CreateCustomBenchmarkRequest(name="Bounded regex", content=content)
    )
    started = time.perf_counter()

    with pytest.raises(ValueError, match="50 ms"):
        adapter.score(adapter.items()[0], "a" * 100_000 + "!")

    assert time.perf_counter() - started < 1


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            '{"id":"same","prompt":"a","expected":"a"}\n'
            '{"id":"same","prompt":"b","expected":"b"}',
            "duplicates item id",
        ),
        (
            '{"prompt":"a","expected":"(","scoring":"regex"}',
            "invalid regular expression",
        ),
        (
            '{"prompt":"a","expected":"a","metadata":{"nested":{}}}',
            "valid string",
        ),
        (
            '{"prompt":"a","expected":"a","scoring":"ifeval"}',
            "internal benchmark scorer",
        ),
        (
            '{"prompt":"a","expected":"a","scoring":"livebench"}',
            "internal benchmark scorer",
        ),
    ],
)
def test_custom_jsonl_rejects_ambiguous_inputs(payload: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CustomBenchmarkAdapter.create(
            CreateCustomBenchmarkRequest(name="Invalid", content=payload)
        )


@pytest.mark.asyncio
async def test_gsm8k_download_is_verified_before_becoming_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (
        "\n".join(
            [
                json.dumps(
                    {
                        "question": "A shop has 1 apple and buys 2. How many?",
                        "answer": "Add them. #### 3",
                    }
                ),
                json.dumps(
                    {
                        "question": "Half of 5 is what?",
                        "answer": "Divide. #### 2.5",
                    }
                ),
            ]
        )
        + "\n"
    )
    encoded = payload.encode()
    import moe_tools_suite.benchmarks.gsm8k as gsm8k_module

    monkeypatch.setattr(gsm8k_module, "GSM8K_SIZE_BYTES", len(encoded))
    monkeypatch.setattr(
        gsm8k_module, "GSM8K_SHA256", hashlib.sha256(encoded).hexdigest()
    )
    monkeypatch.setattr(gsm8k_module, "GSM8K_ITEM_COUNT", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == gsm8k_module.GSM8K_URL
        return httpx.Response(200, content=encoded)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = Gsm8kAdapter(tmp_path)
    progress: list[tuple[int, int]] = []
    assert adapter.info.ready is False

    await adapter.prepare(
        client=client,
        on_progress=lambda current, total: progress.append((current, total)),
    )

    assert adapter.info.ready is True
    assert [item.expected for item in adapter.items()] == ["3", "2.5"]
    assert progress[-1] == (len(encoded), len(encoded))
    assert adapter.score(adapter.items()[0], "Reasoning\n#### 3.0") is True
    await client.aclose()


def test_gsm8k_numeric_extraction_handles_markers_and_commas() -> None:
    assert extract_final_number("work 12 then\n#### 1,200.50") == "1,200.50"
    assert extract_final_number("No marker, final answer -4") == "-4"
    assert numbers_equal("1,200.50", "1200.5")


class SlowRuntime:
    def __init__(self, lab: ResearchLab) -> None:
        self.lab = lab
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(
        self,
        prompt: str,
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
    ) -> CompletionResult:
        del prompt, profile, generation
        self.started.set()
        await self.release.wait()
        topology = self.lab.topology
        ids = np.zeros((1, topology.num_layers, topology.top_k), dtype=np.uint16)
        weights = np.full(ids.shape, 1 / topology.top_k, dtype=np.float32)
        return CompletionResult(
            content="19",
            prompt_tokens=8,
            completion_tokens=1,
            routing=DecodedRouting(expert_ids=ids, expert_weights=weights),
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_running_benchmark_can_be_cancelled_with_partial_results(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "frontend",
        )
    )
    lab = app.state.lab
    await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    slow_runtime = SlowRuntime(lab)
    lab.runtime = slow_runtime
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            submitted = await client.post(
                "/api/runs",
                json={
                    "benchmark_id": "fixture-arithmetic",
                    "item_ids": ["arith-03", "arith-04"],
                },
            )
            assert submitted.status_code == 202
            job_id = submitted.json()["id"]
            await asyncio.wait_for(slow_runtime.started.wait(), timeout=1)

            cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
            assert cancelled.json()["status"] == "cancelling"
            active = await client.get("/api/jobs/active")
            assert active.json()["id"] == job_id
            assert active.json()["status"] == "cancelling"

            blocked = await client.post("/api/benchmarks/fixture-arithmetic/prepare")
            assert blocked.status_code == 409
            assert blocked.json()["detail"]["active_job"]["id"] == job_id
            assert blocked.json()["detail"]["active_job"]["status"] == "cancelling"

            slow_runtime.release.set()
            await asyncio.wait_for(lab.dispatch_job(job_id), timeout=1)

            persisted_job = lab.jobs[job_id].record
            assert persisted_job.status == "cancelled"
            assert persisted_job.result_id is not None
            run = lab.runs[persisted_job.result_id].run
            assert run.status == "cancelled"
            assert run.completed_items == 1
            assert run.total_items == 2
            assert (await client.get("/api/jobs/active")).json() is None
    finally:
        slow_runtime.release.set()
        await lab.shutdown()


@pytest.mark.asyncio
async def test_dataset_cancellation_exception_finishes_cancelled_and_blocks_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "frontend",
        )
    )
    lab = app.state.lab
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_prepare(
        benchmark_id: str,
        *,
        on_progress,
        should_cancel,
    ):
        del on_progress
        started.set()
        await release.wait()
        if should_cancel():
            raise RuntimeError("dataset preparation cancelled")
        return lab.benchmark_catalog.get_info(benchmark_id)

    monkeypatch.setattr(lab.benchmark_catalog, "prepare", slow_prepare)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            submitted = await client.post("/api/benchmarks/fixture-arithmetic/prepare")
            assert submitted.status_code == 202
            job_id = submitted.json()["id"]
            await asyncio.wait_for(started.wait(), timeout=1)

            cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
            assert cancelled.json()["status"] == "cancelling"
            blocked = await client.post("/api/benchmarks/fixture-arithmetic/prepare")
            assert blocked.status_code == 409
            assert blocked.json()["detail"]["active_job"]["id"] == job_id

            release.set()
            await asyncio.wait_for(lab.dispatch_job(job_id), timeout=1)
            terminal = (await client.get(f"/api/jobs/{job_id}")).json()
            assert terminal["status"] == "cancelled"
            assert terminal["error"] is None
            assert (await client.get("/api/jobs/active")).json() is None
    finally:
        release.set()
        await lab.shutdown()
