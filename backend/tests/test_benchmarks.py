import asyncio
import hashlib
import json
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
    RunRequest,
)
from moe_tools_suite.lab import MODEL_ID, ResearchLab
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
    lab = ResearchLab(
        Settings(
            mode="mock",
            data_dir=tmp_path / "data",
            frontend_dist=tmp_path / "frontend",
        )
    )
    await lab.create_model_session(CreateModelSessionRequest(model_id=MODEL_ID))
    slow_runtime = SlowRuntime(lab)
    lab.runtime = slow_runtime
    job = lab.submit_benchmark(
        RunRequest(
            benchmark_id="fixture-arithmetic",
            item_ids=["arith-03", "arith-04"],
        )
    )
    execution = asyncio.create_task(lab.execute_job(job.id))
    await asyncio.wait_for(slow_runtime.started.wait(), timeout=1)

    cancelled = lab.cancel_job(job.id)
    slow_runtime.release.set()
    await asyncio.wait_for(execution, timeout=1)

    persisted_job = lab.jobs[job.id].record
    assert cancelled.status == "cancelled"
    assert persisted_job.status == "cancelled"
    assert persisted_job.result_id is not None
    run = lab.runs[persisted_job.result_id].run
    assert run.status == "cancelled"
    assert run.completed_items == 1
    assert run.total_items == 2
