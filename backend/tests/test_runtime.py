import asyncio
import json

import httpx
import numpy as np
import pytest
from moe_tools_suite.domain import GenerationConfig, ModelTopology
from moe_tools_suite.runtime import VllmRuntime
from moe_tools_suite.telemetry import encode_npy


class _DelayedSseStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[dict[str, object] | str]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            await asyncio.sleep(0.01)
            data = chunk if isinstance(chunk, str) else json.dumps(chunk)
            yield f"data: {data}\n\n".encode()


@pytest.mark.asyncio
async def test_vllm_runtime_reads_custom_choice_telemetry() -> None:
    topology = ModelTopology(
        num_layers=2,
        num_experts=4,
        top_k=2,
        routed_layer_ids=[0, 1],
    )
    ids = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
    weights = np.full(ids.shape, 0.5, dtype=np.float32)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["temperature"] == 0.25
        assert payload["max_tokens"] == 321
        assert payload["seed"] == 7
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["include_reasoning"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "19",
                            "reasoning": "Twelve plus seven is nineteen.",
                        },
                        "finish_reason": "stop",
                        "routed_experts": encode_npy(ids),
                        "routed_expert_weights": encode_npy(weights),
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 1,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                },
                "metrics": {
                    "time_to_first_token_ms": 14.5,
                    "generation_time_ms": 8.0,
                    "queue_time_ms": 1.25,
                    "mean_itl_ms": 4.0,
                    "tokens_per_second": 125.0,
                },
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm"
    )
    runtime = VllmRuntime("http://vllm", "model", topology, client=client)

    result = await runtime.complete(
        "12 + 7",
        request_key="ignored",
        profile=None,
        generation=GenerationConfig(
            temperature=0.25,
            max_tokens=321,
            seed=7,
        ),
    )

    assert result.content == "19"
    assert result.reasoning == "Twelve plus seven is nineteen."
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 8
    assert result.reasoning_tokens == 7
    assert result.performance.time_to_first_token_ms == 14.5
    assert result.performance.generation_time_ms == 8.0
    assert result.performance.queue_time_ms == 1.25
    assert result.performance.mean_inter_token_latency_ms == 4.0
    assert result.performance.tokens_per_second == 125.0
    np.testing.assert_array_equal(result.routing.expert_ids, ids)
    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_runtime_streams_progress_without_changing_final_result() -> None:
    topology = ModelTopology(
        num_layers=2,
        num_experts=4,
        top_k=2,
        routed_layer_ids=[0, 1],
    )
    ids = np.arange(16, dtype=np.uint8).reshape(4, 2, 2) % 4
    weights = np.full(ids.shape, 0.5, dtype=np.float32)
    context_fingerprint = "a" * 64
    prompt_ids = [1, 2, 3]
    completion_ids = [10, 11]
    metrics = {
        "time_to_first_token_ms": 10.0,
        "generation_time_ms": 20.0,
        "tokens_per_second": 100.0,
    }
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if payload.get("stream") is not True:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "19"},
                            "finish_reason": "stop",
                            "token_ids": completion_ids,
                            "routed_experts": encode_npy(ids),
                            "routed_expert_weights": encode_npy(weights),
                            "expert_context_fingerprint": context_fingerprint,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "completion_tokens_details": {"reasoning_tokens": 1},
                    },
                    "prompt_token_ids": prompt_ids,
                    "metrics": metrics,
                    "expert_context_fingerprint": context_fingerprint,
                },
            )

        def usage(completion: int) -> dict[str, object]:
            payload: dict[str, object] = {
                "prompt_tokens": 3,
                "completion_tokens": completion,
                "total_tokens": 3 + completion,
            }
            if completion == 2:
                payload["completion_tokens_details"] = {"reasoning_tokens": 1}
            return payload

        chunks: list[dict[str, object] | str] = [
            {
                "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                "prompt_token_ids": prompt_ids,
                "usage": usage(0),
                "expert_context_fingerprint": context_fingerprint,
            },
            {
                "choices": [{"index": 0, "delta": {"content": "1"}, "token_ids": [10]}],
                "usage": usage(1),
                "expert_context_fingerprint": context_fingerprint,
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "9"},
                        "token_ids": [11],
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage(2),
                "expert_context_fingerprint": context_fingerprint,
            },
            {
                "choices": [],
                "usage": usage(2),
                "metrics": metrics,
                "routed_experts": encode_npy(ids),
                "routed_expert_weights": encode_npy(weights),
                "expert_context_fingerprint": context_fingerprint,
            },
            "[DONE]",
        ]
        return httpx.Response(200, stream=_DelayedSseStream(chunks))

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm"
    )
    runtime = VllmRuntime("http://vllm", "model", topology, client=client)
    non_streamed = await runtime.complete(
        "12 + 7",
        request_key="ordinary-nonstream",
        profile=None,
    )
    progress = []
    streamed = await runtime.complete(
        "12 + 7",
        request_key="ordinary-stream",
        profile=None,
        on_progress=progress.append,
    )

    assert requests[0].get("stream") is None
    assert requests[1]["stream"] is True
    assert requests[1]["return_token_ids"] is True
    assert requests[1]["stream_options"] == {
        "include_usage": True,
        "continuous_usage_stats": True,
    }
    assert [item.completion_tokens for item in progress] == [0, 1, 2]
    assert progress[1].elapsed_ms < 1000
    assert progress[1].current_tps is not None
    assert streamed.content == non_streamed.content == "19"
    assert streamed.prompt_tokens == non_streamed.prompt_tokens == 3
    assert streamed.completion_tokens == non_streamed.completion_tokens == 2
    assert streamed.finish_reason == non_streamed.finish_reason == "stop"
    assert streamed.reasoning_tokens == non_streamed.reasoning_tokens == 1
    assert streamed.context_fingerprint == non_streamed.context_fingerprint
    assert streamed.performance == non_streamed.performance
    np.testing.assert_array_equal(
        streamed.routing.expert_ids, non_streamed.routing.expert_ids
    )
    np.testing.assert_array_equal(
        streamed.routing.expert_weights, non_streamed.routing.expert_weights
    )
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_usage", [None, {"prompt_tokens": 3}])
async def test_vllm_runtime_rejects_missing_or_invalid_continuous_usage(
    invalid_usage: dict[str, int] | None,
) -> None:
    topology = ModelTopology(
        num_layers=1,
        num_experts=2,
        top_k=1,
        routed_layer_ids=[0],
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        chunk: dict[str, object] = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "1"},
                    "token_ids": [10],
                    "finish_reason": "stop",
                }
            ]
        }
        if invalid_usage is not None:
            chunk["usage"] = invalid_usage
        return httpx.Response(
            200,
            stream=_DelayedSseStream([chunk, "[DONE]"]),
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm"
    )
    runtime = VllmRuntime("http://vllm", "model", topology, client=client)
    with pytest.raises(ValueError, match="usage"):
        await runtime.complete(
            "hello",
            request_key="invalid-progress",
            profile=None,
            on_progress=lambda _progress: None,
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_runtime_skips_only_a_prefix_captured_in_the_same_trial() -> None:
    topology = ModelTopology(
        num_layers=2,
        num_experts=4,
        top_k=2,
        routed_layer_ids=[0, 1],
    )
    first_eot = 99
    second_eot = 98
    # A sampled EOT has no routing row, then appears in the next rendered prompt.
    prompt_token_ids = [
        [10, 11, 12],
        [10, 11, 12, 20, 21, first_eot, 30],
        [10, 11, 12, 20, 21, first_eot, 30, 40, second_eot, 50],
        [70, 71],
        [10, 11, 12, 20, 21, first_eot, 30],
    ]
    generated_token_ids = [
        [20, 21, first_eot],
        [40, second_eot],
        [60, 97],
        [80, 96],
        [90, 95],
    ]
    routed_rows = [5, 3, 3, 3, 8]
    contents = [
        '{"action":"shell","command":"pwd"}',
        "next",
        "done",
        "unrelated",
        "separate trial",
    ]
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(
                200,
                json={
                    "count": len(prompt_token_ids[len(calls)]),
                    "max_model_len": 4096,
                    "tokens": prompt_token_ids[len(calls)],
                },
            )
        assert request.url.path == "/v1/chat/completions"
        calls.append(payload)
        call_index = len(calls) - 1
        rows = routed_rows[call_index]
        ids = np.tile(
            np.array([[[0, 1], [2, 3]]], dtype=np.uint8),
            (rows, 1, 1),
        )
        weights = np.full(ids.shape, 0.5, dtype=np.float32)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": contents[call_index]},
                        "token_ids": generated_token_ids[call_index],
                        "routed_experts": encode_npy(ids),
                        "routed_expert_weights": encode_npy(weights),
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2},
                "prompt_token_ids": prompt_token_ids[call_index],
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm"
    )
    runtime = VllmRuntime("http://vllm", "model", topology, client=client)
    first_messages = [
        {"role": "system", "content": "Use JSON."},
        {"role": "user", "content": "Fix it."},
    ]
    first = await runtime.complete_chat(
        first_messages,
        request_key="agent:trial-a:0",
        profile=None,
    )
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": "Command exited 0."},
    ]
    second = await runtime.complete_chat(
        second_messages,
        request_key="agent:trial-a:1",
        profile=None,
    )
    await runtime.complete_chat(
        [
            *second_messages,
            {"role": "assistant", "content": second.content},
            {"role": "user", "content": "Continue."},
        ],
        request_key="agent:trial-a:2",
        profile=None,
    )
    await runtime.complete_chat(
        [
            {"role": "system", "content": "Use JSON."},
            {"role": "user", "content": "A clipped, unrelated tail."},
        ],
        request_key="agent:trial-unrelated:0",
        profile=None,
    )
    await runtime.complete_chat(
        second_messages,
        request_key="agent:trial-b:1",
        profile=None,
    )

    assert "routed_experts_prompt_start" not in calls[0]
    assert calls[1]["routed_experts_prompt_start"] == 5
    assert calls[2]["routed_experts_prompt_start"] == 8
    assert "routed_experts_prompt_start" not in calls[3]
    assert "routed_experts_prompt_start" not in calls[4]
    assert all(call["return_token_ids"] is True for call in calls)
    await client.aclose()
