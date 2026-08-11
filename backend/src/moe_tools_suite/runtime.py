from __future__ import annotations

import hashlib
import json
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

    async def complete_chat(
        self,
        messages: list[dict[str, str]],
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
        return await self.complete_chat(
            [{"role": "user", "content": prompt}],
            request_key=request_key,
            profile=profile,
            generation=generation,
        )

    async def complete_chat(
        self,
        messages: list[dict[str, str]],
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
    ) -> CompletionResult:
        del generation
        prompt = "\n".join(message.get("content", "") for message in messages)
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
        self._routing_prefixes: dict[tuple[str, str], tuple[int, ...]] = {}

    async def complete(
        self,
        prompt: str,
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
    ) -> CompletionResult:
        return await self.complete_chat(
            [{"role": "user", "content": prompt}],
            request_key=request_key,
            profile=profile,
            generation=generation,
        )

    async def complete_chat(
        self,
        messages: list[dict[str, str]],
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
    ) -> CompletionResult:
        del profile
        config = generation or GenerationConfig()
        chat_template_kwargs = {"enable_thinking": config.enable_thinking}
        request_scope = _routing_scope(request_key)
        routed_prompt_start = await self._routed_prompt_start(
            messages,
            chat_template_kwargs,
            request_scope,
        )
        request_payload: dict[str, object] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "chat_template_kwargs": chat_template_kwargs,
            "return_token_ids": True,
        }
        if config.seed is not None:
            request_payload["seed"] = config.seed
        if routed_prompt_start:
            request_payload["routed_experts_prompt_start"] = routed_prompt_start
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
        usage = payload.get("usage") or {}
        content = choice["message"]["content"] or ""
        routing = decode_routing_payloads(
            choice["routed_experts"],
            choice["routed_expert_weights"],
            self._topology,
        )
        self._remember_routing_prefix(
            messages,
            content,
            payload,
            choice,
            routed_prompt_start,
            routing,
            request_scope,
        )
        return CompletionResult(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            routing=routing,
        )

    async def _routed_prompt_start(
        self,
        messages: list[dict[str, str]],
        chat_template_kwargs: dict[str, bool],
        request_scope: str,
    ) -> int:
        previous_tokens: tuple[int, ...] | None = None
        for prefix_length in range(len(messages) - 1, 0, -1):
            fingerprint = _messages_fingerprint(messages[:prefix_length])
            cache_key = (request_scope, fingerprint)
            if cache_key in self._routing_prefixes:
                previous_tokens = self._routing_prefixes[cache_key]
                break
        if previous_tokens is None:
            return 0
        try:
            response = await self._client.post(
                "/tokenize",
                json={
                    "model": self._model_id,
                    "messages": messages,
                    "chat_template_kwargs": chat_template_kwargs,
                },
            )
            response.raise_for_status()
            prompt_tokens = _token_ids(response.json().get("tokens"))
        except (httpx.HTTPError, AttributeError, TypeError, ValueError):
            return 0
        if prompt_tokens is None:
            return 0
        matched = _common_prefix_length(previous_tokens, prompt_tokens)
        return matched if 0 < matched < len(prompt_tokens) else 0

    def _remember_routing_prefix(
        self,
        messages: list[dict[str, str]],
        content: str,
        payload: dict[str, object],
        choice: dict[str, object],
        routed_prompt_start: int,
        routing: DecodedRouting,
        request_scope: str,
    ) -> None:
        prompt_tokens = _token_ids(payload.get("prompt_token_ids"))
        generated_tokens = _token_ids(choice.get("token_ids"))
        if prompt_tokens is None or generated_tokens is None:
            return
        captured_tokens = (*prompt_tokens, *generated_tokens[:-1])
        expected_rows = (
            len(prompt_tokens) - routed_prompt_start + max(0, len(generated_tokens) - 1)
        )
        if expected_rows != routing.expert_ids.shape[0]:
            raise ValueError(
                "routing telemetry length does not match returned token metadata"
            )
        fingerprint = _messages_fingerprint(
            [*messages, {"role": "assistant", "content": content}]
        )
        self._routing_prefixes[(request_scope, fingerprint)] = captured_tokens
        if len(self._routing_prefixes) > 256:
            del self._routing_prefixes[next(iter(self._routing_prefixes))]

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


def _messages_fingerprint(messages: list[dict[str, str]]) -> str:
    serialized = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _routing_scope(request_key: str) -> str:
    scope, separator, turn = request_key.rpartition(":")
    if request_key.startswith("agent:") and separator and turn.isdecimal():
        return scope
    return request_key


def _token_ids(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        return None
    return tuple(value)


def _common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        matched += 1
    return matched


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
