from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np

from .domain import ExpertProfile, GenerationConfig, ModelTopology
from .telemetry import DecodedRouting, decode_routing_payloads
from .v2_domain import InterventionContextRef


@dataclass(frozen=True)
class CompletionPerformance:
    """Timing metrics reported by the serving runtime for one completion."""

    time_to_first_token_ms: float | None = None
    generation_time_ms: float | None = None
    queue_time_ms: float | None = None
    mean_inter_token_latency_ms: float | None = None
    tokens_per_second: float | None = None


@dataclass(frozen=True)
class CompletionProgress:
    """Cumulative token and timing telemetry for an active completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_ms: float
    current_tps: float | None


CompletionProgressCallback = Callable[[CompletionProgress], None]


@dataclass(frozen=True)
class CompletionResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    routing: DecodedRouting
    reasoning_tokens: int | None = None
    reasoning: str | None = None
    finish_reason: str | None = None
    performance: CompletionPerformance = field(default_factory=CompletionPerformance)
    context_id: str | None = None
    context_fingerprint: str | None = None
    topology_fingerprint: str | None = None


class ModelRuntime(Protocol):
    def set_active_context(self, context: InterventionContextRef | None) -> None: ...

    async def complete(
        self,
        prompt: str,
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
        on_progress: CompletionProgressCallback | None = None,
    ) -> CompletionResult: ...

    async def complete_chat(
        self,
        messages: list[dict[str, str]],
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
        on_progress: CompletionProgressCallback | None = None,
    ) -> CompletionResult: ...

    async def aclose(self) -> None: ...


class MockModelRuntime:
    """Deterministic GPU-free model and routing fixture."""

    def __init__(self, topology: ModelTopology) -> None:
        self._topology = topology
        self._active_context: InterventionContextRef | None = None

    def set_active_context(self, context: InterventionContextRef | None) -> None:
        self._active_context = context

    async def complete(
        self,
        prompt: str,
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
        on_progress: CompletionProgressCallback | None = None,
    ) -> CompletionResult:
        return await self.complete_chat(
            [{"role": "user", "content": prompt}],
            request_key=request_key,
            profile=profile,
            generation=generation,
            on_progress=on_progress,
        )

    async def complete_chat(
        self,
        messages: list[dict[str, str]],
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
        on_progress: CompletionProgressCallback | None = None,
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
        result = CompletionResult(
            content=answer,
            prompt_tokens=len(prompt.split()),
            completion_tokens=1,
            routing=DecodedRouting(expert_ids=ids, expert_weights=weights),
            finish_reason="stop",
            context_id=(
                self._active_context.context_id if self._active_context else None
            ),
            context_fingerprint=(
                self._active_context.context_fingerprint
                if self._active_context
                else None
            ),
            topology_fingerprint=(
                self._active_context.topology_fingerprint
                if self._active_context
                else None
            ),
        )
        if on_progress is not None:
            on_progress(
                CompletionProgress(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens,
                    elapsed_ms=0,
                    current_tps=None,
                )
            )
        return result

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
        self._active_context: InterventionContextRef | None = None

    def set_active_context(self, context: InterventionContextRef | None) -> None:
        if context == self._active_context:
            return
        self._active_context = context
        self._routing_prefixes.clear()

    async def complete(
        self,
        prompt: str,
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
        on_progress: CompletionProgressCallback | None = None,
    ) -> CompletionResult:
        return await self.complete_chat(
            [{"role": "user", "content": prompt}],
            request_key=request_key,
            profile=profile,
            generation=generation,
            on_progress=on_progress,
        )

    async def complete_chat(
        self,
        messages: list[dict[str, str]],
        *,
        request_key: str,
        profile: ExpertProfile | None,
        generation: GenerationConfig | None = None,
        on_progress: CompletionProgressCallback | None = None,
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
            "include_reasoning": config.enable_thinking,
            "return_token_ids": True,
        }
        if config.seed is not None:
            request_payload["seed"] = config.seed
        if routed_prompt_start:
            request_payload["routed_experts_prompt_start"] = routed_prompt_start
        if on_progress is not None:
            request_payload.update(
                {
                    "stream": True,
                    "stream_options": {
                        "include_usage": True,
                        "continuous_usage_stats": True,
                    },
                }
            )
            return await self._complete_chat_streaming(
                messages=messages,
                request_payload=request_payload,
                routed_prompt_start=routed_prompt_start,
                request_scope=request_scope,
                on_progress=on_progress,
            )
        response = await self._client.post(
            "/v1/chat/completions",
            json=request_payload,
        )
        response.raise_for_status()
        payload = response.json()
        context_id, context_fingerprint, topology_fingerprint = (
            self._response_context_provenance(payload)
        )
        choice = payload["choices"][0]
        if not choice.get("routed_experts") or not choice.get("routed_expert_weights"):
            raise ValueError(
                "vLLM response omitted routing telemetry; start the fork with "
                "both routing capture flags"
            )
        usage = payload.get("usage") or {}
        completion_details = usage.get("completion_tokens_details")
        reasoning_tokens = (
            _optional_nonnegative_int(completion_details.get("reasoning_tokens"))
            if isinstance(completion_details, dict)
            else None
        )
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("vLLM response returned invalid reasoning content")
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
            reasoning_tokens=reasoning_tokens,
            reasoning=reasoning,
            finish_reason=_optional_string(choice.get("finish_reason")),
            performance=_completion_performance(payload.get("metrics")),
            context_id=context_id,
            context_fingerprint=context_fingerprint,
            topology_fingerprint=topology_fingerprint,
        )

    async def _complete_chat_streaming(
        self,
        *,
        messages: list[dict[str, str]],
        request_payload: dict[str, object],
        routed_prompt_start: int,
        request_scope: str,
        on_progress: CompletionProgressCallback,
    ) -> CompletionResult:
        started = time.perf_counter()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        prompt_token_ids: list[int] | None = None
        completion_token_ids: list[int] = []
        finish_reason: str | None = None
        final_payload: dict[str, object] | None = None
        final_usage: dict[str, object] | None = None
        saw_done = False
        last_progress: tuple[int, int] | None = None
        positive_progress_emitted = False

        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=request_payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError as error:
                    raise ValueError("vLLM stream returned invalid JSON") from error
                if not isinstance(payload, dict):
                    raise ValueError("vLLM stream returned a non-object chunk")
                self._response_context_provenance(payload)
                choices = payload.get("choices")
                if not isinstance(choices, list):
                    raise ValueError("vLLM stream omitted its choices array")
                if choices:
                    if len(choices) != 1 or not isinstance(choices[0], dict):
                        raise ValueError(
                            "vLLM runtime requires a single streaming choice"
                        )
                    choice = choices[0]
                    if choice.get("index") not in {None, 0}:
                        raise ValueError("vLLM stream returned an unexpected choice")
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        raise ValueError("vLLM stream returned an invalid delta")
                    _append_optional_text(
                        content_parts, delta.get("content"), "content"
                    )
                    _append_optional_text(
                        reasoning_parts,
                        delta.get("reasoning") or delta.get("reasoning_content"),
                        "reasoning",
                    )
                    finish = choice.get("finish_reason")
                    if finish is not None:
                        finish_reason = _optional_string(finish)
                        if finish_reason is None:
                            raise ValueError(
                                "vLLM stream returned an invalid finish reason"
                            )
                    token_ids = choice.get("token_ids")
                    if token_ids is not None:
                        parsed_ids = _token_ids(token_ids)
                        if parsed_ids is None:
                            raise ValueError("vLLM stream returned invalid token IDs")
                        completion_token_ids.extend(parsed_ids)
                elif payload.get("usage") is not None:
                    final_payload = payload

                raw_prompt_ids = payload.get("prompt_token_ids")
                if raw_prompt_ids is not None:
                    parsed_prompt_ids = _token_ids(raw_prompt_ids)
                    if parsed_prompt_ids is None:
                        raise ValueError(
                            "vLLM stream returned invalid prompt token IDs"
                        )
                    prompt_token_ids = list(parsed_prompt_ids)

                usage = payload.get("usage")
                if usage is not None:
                    if not isinstance(usage, dict):
                        raise ValueError("vLLM stream returned invalid usage")
                    prompt_tokens, completion_tokens, total_tokens = _stream_usage(
                        usage
                    )
                    progress_key = (prompt_tokens, completion_tokens)
                    if last_progress is not None and (
                        prompt_tokens < last_progress[0]
                        or completion_tokens < last_progress[1]
                    ):
                        raise ValueError("vLLM stream usage counters regressed")
                    if progress_key != last_progress:
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        current_tps = (
                            completion_tokens * 1000 / elapsed_ms
                            if completion_tokens > 0 and elapsed_ms > 0
                            else None
                        )
                        on_progress(
                            CompletionProgress(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens,
                                elapsed_ms=elapsed_ms,
                                current_tps=current_tps,
                            )
                        )
                        positive_progress_emitted = (
                            positive_progress_emitted or current_tps is not None
                        )
                        last_progress = progress_key
                    if not choices:
                        final_usage = usage

        if not saw_done or final_payload is None or final_usage is None:
            raise ValueError("vLLM stream omitted its final usage chunk")
        if not positive_progress_emitted:
            raise ValueError("vLLM stream omitted continuous token progress")
        if finish_reason is None:
            raise ValueError("vLLM stream omitted its finish reason")
        routing_ids = final_payload.get("routed_experts")
        routing_weights = final_payload.get("routed_expert_weights")
        if not isinstance(routing_ids, str) or not isinstance(routing_weights, str):
            raise ValueError(
                "vLLM stream omitted routing telemetry; start the fork with "
                "both routing capture flags"
            )
        routing = decode_routing_payloads(
            routing_ids,
            routing_weights,
            self._topology,
        )
        content = "".join(content_parts)
        prompt_tokens, completion_tokens, _ = _stream_usage(final_usage)
        completion_details = final_usage.get("completion_tokens_details")
        reasoning_tokens = (
            _optional_nonnegative_int(completion_details.get("reasoning_tokens"))
            if isinstance(completion_details, dict)
            else None
        )
        synthetic_payload: dict[str, object] = {
            "prompt_token_ids": (
                prompt_token_ids
                if prompt_token_ids is not None
                and len(prompt_token_ids) == prompt_tokens
                else None
            ),
        }
        synthetic_choice: dict[str, object] = {
            "token_ids": (
                completion_token_ids
                if len(completion_token_ids) == completion_tokens
                else None
            ),
        }
        self._remember_routing_prefix(
            messages,
            content,
            synthetic_payload,
            synthetic_choice,
            routed_prompt_start,
            routing,
            request_scope,
        )
        context_id, context_fingerprint, topology_fingerprint = (
            self._response_context_provenance(final_payload)
        )
        metrics = _completion_performance(final_payload.get("metrics"))
        return CompletionResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            routing=routing,
            reasoning_tokens=reasoning_tokens,
            reasoning="".join(reasoning_parts) or None,
            finish_reason=finish_reason,
            performance=metrics,
            context_id=context_id,
            context_fingerprint=context_fingerprint,
            topology_fingerprint=topology_fingerprint,
        )

    def _response_context_provenance(
        self, payload: dict[str, object]
    ) -> tuple[str | None, str | None, str | None]:
        choice = payload.get("choices")
        first_choice = choice[0] if isinstance(choice, list) and choice else None
        choice_payload = first_choice if isinstance(first_choice, dict) else {}
        fingerprint = _optional_string(
            payload.get("expert_context_fingerprint")
            or choice_payload.get("expert_context_fingerprint")
        )
        context_id = _optional_string(
            payload.get("expert_context_id") or choice_payload.get("expert_context_id")
        )
        topology_fingerprint = _optional_string(
            payload.get("expert_context_topology_fingerprint")
            or choice_payload.get("expert_context_topology_fingerprint")
        )
        active = self._active_context
        if active is None:
            return context_id, fingerprint, topology_fingerprint
        if fingerprint is None:
            raise ValueError(
                "vLLM response omitted expert_context_fingerprint while an "
                "expert context was active"
            )
        if fingerprint != active.context_fingerprint:
            raise ValueError(
                "vLLM response expert-context fingerprint does not match the "
                "active intervention context"
            )
        if context_id is not None and context_id != active.context_id:
            raise ValueError(
                "vLLM response expert-context ID does not match the active context"
            )
        if (
            topology_fingerprint is not None
            and topology_fingerprint != active.topology_fingerprint
        ):
            raise ValueError(
                "vLLM response expert-context topology does not match the active "
                "context"
            )
        return (
            context_id or active.context_id,
            fingerprint,
            topology_fingerprint or active.topology_fingerprint,
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


def _append_optional_text(parts: list[str], value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"vLLM stream returned invalid {field_name}")
    parts.append(value)


def _stream_usage(usage: dict[str, object]) -> tuple[int, int, int]:
    prompt_tokens = _optional_nonnegative_int(usage.get("prompt_tokens"))
    completion_tokens = _optional_nonnegative_int(usage.get("completion_tokens"))
    total_tokens = _optional_nonnegative_int(usage.get("total_tokens"))
    if (
        prompt_tokens is None
        or completion_tokens is None
        or total_tokens != prompt_tokens + completion_tokens
    ):
        raise ValueError("vLLM stream returned inconsistent token usage")
    return prompt_tokens, completion_tokens, total_tokens


def _common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        matched += 1
    return matched


def _completion_performance(value: object) -> CompletionPerformance:
    if not isinstance(value, dict):
        return CompletionPerformance()
    return CompletionPerformance(
        time_to_first_token_ms=_optional_nonnegative_float(
            value.get("time_to_first_token_ms")
        ),
        generation_time_ms=_optional_nonnegative_float(value.get("generation_time_ms")),
        queue_time_ms=_optional_nonnegative_float(value.get("queue_time_ms")),
        mean_inter_token_latency_ms=_optional_nonnegative_float(
            value.get("mean_itl_ms")
        ),
        tokens_per_second=_optional_nonnegative_float(value.get("tokens_per_second")),
    )


def _optional_nonnegative_float(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _optional_nonnegative_int(value: object) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


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
