from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from .domain import ExpertProfile, GenerationConfig, ModelTopology
from .telemetry import DecodedRouting, decode_routing_payloads


@dataclass(frozen=True)
class CompletionResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    routing: DecodedRouting


class ModelRuntime(Protocol):
    async def complete(
        self,
        prompt: str,
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
    ) -> CompletionResult: ...

    async def aclose(self) -> None: ...


class MockModelRuntime:
    """Deterministic GPU-free model and routing fixture."""

    def __init__(self, topology: ModelTopology) -> None:
        self._topology = topology

    async def complete(
        self,
        prompt: str,
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
    ) -> CompletionResult:
        del generation
        seed = int.from_bytes(
            hashlib.sha256(request_key.encode()).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        token_count = 12 + seed % 9
        ids = np.empty(
            (token_count, self._topology.num_layers, self._topology.top_k),
            dtype=np.uint16,
        )
        weights = np.empty(ids.shape, dtype=np.float32)
        for layer_index, layer_id in enumerate(self._topology.routed_layer_ids):
            eligible = _eligible_experts(profile, layer_id, self._topology.num_experts)
            for token_index in range(token_count):
                selected = rng.choice(
                    eligible,
                    size=self._topology.top_k,
                    replace=False,
                )
                raw_weights = rng.random(self._topology.top_k, dtype=np.float32)
                ids[token_index, layer_index] = selected
                weights[token_index, layer_index] = raw_weights / raw_weights.sum()

        answer = _solve_fixture_prompt(prompt)
        if request_key.endswith(("05", "11")):
            answer = str(int(answer) + 1)
        return CompletionResult(
            content=answer,
            prompt_tokens=len(prompt.split()),
            completion_tokens=1,
            routing=DecodedRouting(expert_ids=ids, expert_weights=weights),
        )

    async def aclose(self) -> None:
        return None


class VllmRuntime:
    """Thin client for the custom fork's OpenAI-compatible telemetry fields."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        topology: ModelTopology,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_id = model_id
        self._topology = topology
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=90
        )

    async def complete(
        self,
        prompt: str,
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
    ) -> CompletionResult:
        del request_key, profile
        config = generation or GenerationConfig()
        request_payload: dict[str, object] = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.seed is not None:
            request_payload["seed"] = config.seed
        response = await self._client.post(
            "/v1/chat/completions",
            json=request_payload,
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        if not choice.get("routed_experts") or not choice.get("routed_expert_weights"):
            raise ValueError(
                "vLLM response omitted routing telemetry; start the fork with "
                "both routing capture flags"
            )
        usage = payload.get("usage", {})
        return CompletionResult(
            content=choice["message"]["content"] or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            routing=decode_routing_payloads(
                choice["routed_experts"],
                choice["routed_expert_weights"],
                self._topology,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _eligible_experts(
    profile: ExpertProfile | None,
    layer_id: int,
    num_experts: int,
) -> np.ndarray:
    if profile is not None and (layer := profile.layers.get(str(layer_id))):
        return np.asarray(layer.keep, dtype=np.uint16)
    return np.arange(num_experts, dtype=np.uint16)


def _solve_fixture_prompt(prompt: str) -> str:
    match = re.search(r"(-?\d+)\s*([+*-])\s*(-?\d+)", prompt)
    if match is None:
        return "0"
    left, operator, right = match.groups()
    values = int(left), int(right)
    if operator == "+":
        return str(values[0] + values[1])
    if operator == "-":
        return str(values[0] - values[1])
    return str(values[0] * values[1])
