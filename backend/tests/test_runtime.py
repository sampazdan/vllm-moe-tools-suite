import json

import httpx
import numpy as np
import pytest
from moe_tools_suite.domain import GenerationConfig, ModelTopology
from moe_tools_suite.runtime import VllmRuntime
from moe_tools_suite.telemetry import encode_npy


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
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "19"},
                        "routed_experts": encode_npy(ids),
                        "routed_expert_weights": encode_npy(weights),
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1},
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
    assert result.prompt_tokens == 8
    np.testing.assert_array_equal(result.routing.expert_ids, ids)
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
