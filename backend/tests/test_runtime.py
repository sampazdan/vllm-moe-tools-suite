import httpx
import numpy as np
import pytest
from moe_tools_suite.domain import ModelTopology
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
        "12 + 7", request_key="ignored", profile=None
    )

    assert result.content == "19"
    assert result.prompt_tokens == 8
    np.testing.assert_array_equal(result.routing.expert_ids, ids)
    await client.aclose()
