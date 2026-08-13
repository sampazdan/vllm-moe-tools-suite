from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..domain import ExpertProfile, GenerationConfig, ModelSession, ModelTopology
from ..persistence import SqliteStore
from ..runtime import ModelRuntime
from ..telemetry import AggregatedRouting, DecodedRouting, aggregate_routing
from ..v2_domain import InterventionContextRef
from .artifacts import AgentArtifactStore
from .atif import BASH_JSON_SYSTEM_PROMPT, BASH_TOOL_DEFINITION
from .domain import InferenceCall, utc_now


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["shell", "finish"]
    command: str | None = Field(default=None, max_length=20_000)
    summary: str | None = Field(default=None, max_length=20_000)


@dataclass(frozen=True)
class GatewayResult:
    content: str
    reasoning: str | None
    inference: InferenceCall
    routing: DecodedRouting
    aggregated: AggregatedRouting


class InstrumentedAgentGateway:
    """Run every agent turn through the routing-aware local model runtime."""

    def __init__(
        self,
        *,
        runtime: ModelRuntime,
        topology: ModelTopology,
        store: SqliteStore,
        artifacts: AgentArtifactStore,
    ) -> None:
        self.runtime = runtime
        self.topology = topology
        self.store = store
        self.artifacts = artifacts

    async def infer(
        self,
        *,
        trial_id: str,
        trajectory_step_id: str,
        model_session: ModelSession,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
        request_key: str,
        scripted_content: str | None = None,
        profile: ExpertProfile | None = None,
        context: InterventionContextRef | None = None,
    ) -> GatewayResult:
        inference_id = str(uuid4())
        request_hash = hashlib.sha256(
            json.dumps(
                {"messages": messages, "generation": generation.model_dump()},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        started_at = utc_now()
        started = time.perf_counter()
        try:
            completion = await self.runtime.complete_chat(
                messages,
                request_key=request_key,
                profile=profile,
                generation=generation,
            )
        except Exception as error:
            inference = InferenceCall(
                id=inference_id,
                trial_id=trial_id,
                trajectory_step_id=trajectory_step_id,
                model_session_id=model_session.id,
                context_id=context.context_id if context is not None else None,
                context_fingerprint=(
                    context.context_fingerprint if context is not None else None
                ),
                topology_fingerprint=(
                    context.topology_fingerprint if context is not None else None
                ),
                request_hash=request_hash,
                latency_ms=(time.perf_counter() - started) * 1000,
                started_at=started_at,
                completed_at=utc_now(),
                error=_bounded_error(error),
            )
            self.store.save_agent_inference(inference)
            raise

        if context is not None and (
            completion.context_id != context.context_id
            or completion.context_fingerprint != context.context_fingerprint
            or completion.topology_fingerprint != context.topology_fingerprint
        ):
            error = RuntimeError(
                "model response expert-context provenance did not match the "
                "coding trial context"
            )
            inference = InferenceCall(
                id=inference_id,
                trial_id=trial_id,
                trajectory_step_id=trajectory_step_id,
                model_session_id=model_session.id,
                context_id=context.context_id,
                context_fingerprint=context.context_fingerprint,
                topology_fingerprint=context.topology_fingerprint,
                request_hash=request_hash,
                latency_ms=(time.perf_counter() - started) * 1000,
                started_at=started_at,
                completed_at=utc_now(),
                error=_bounded_error(error),
            )
            self.store.save_agent_inference(inference)
            raise error
        routing_artifact = self.artifacts.save_inference_routing(
            trial_id,
            inference_id,
            completion.routing,
            context_id=context.context_id if context is not None else None,
            context_fingerprint=(
                context.context_fingerprint if context is not None else None
            ),
            topology_fingerprint=(
                context.topology_fingerprint if context is not None else None
            ),
        )
        inference = InferenceCall(
            id=inference_id,
            trial_id=trial_id,
            trajectory_step_id=trajectory_step_id,
            model_session_id=model_session.id,
            context_id=context.context_id if context is not None else None,
            context_fingerprint=(
                context.context_fingerprint if context is not None else None
            ),
            topology_fingerprint=(
                context.topology_fingerprint if context is not None else None
            ),
            request_hash=request_hash,
            prompt_tokens=completion.prompt_tokens,
            reasoning_tokens=completion.reasoning_tokens,
            completion_tokens=completion.completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            time_to_first_token_ms=(completion.performance.time_to_first_token_ms),
            generation_time_ms=completion.performance.generation_time_ms,
            queue_time_ms=completion.performance.queue_time_ms,
            mean_inter_token_latency_ms=(
                completion.performance.mean_inter_token_latency_ms
            ),
            tokens_per_second=completion.performance.tokens_per_second,
            reasoning_content=(
                completion.reasoning[:200_000] if completion.reasoning else None
            ),
            finish_reason=completion.finish_reason,
            started_at=started_at,
            completed_at=utc_now(),
            routing_artifact=routing_artifact,
        )
        self.store.save_agent_inference(inference)
        return GatewayResult(
            content=scripted_content or completion.content,
            reasoning=completion.reasoning,
            inference=inference,
            routing=completion.routing,
            aggregated=aggregate_routing(completion.routing, self.topology),
        )


def parse_agent_action(content: str) -> AgentAction:
    """Parse one bounded shell/finish action from a model response."""

    payload = _extract_json_object(content)
    action = AgentAction.model_validate(payload)
    if action.action == "shell":
        command = (action.command or "").strip()
        if not command:
            raise ValueError("shell action requires a command")
        if "\x00" in command:
            raise ValueError("shell command contains a null byte")
        return action.model_copy(update={"command": command})
    summary = (action.summary or "").strip()
    if not summary:
        raise ValueError("finish action requires a summary")
    return action.model_copy(update={"summary": summary})


def system_prompt_hash() -> str:
    return hashlib.sha256(BASH_JSON_SYSTEM_PROMPT.encode()).hexdigest()


def tool_schema_hash() -> str:
    return hashlib.sha256(
        json.dumps(BASH_TOOL_DEFINITION, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _extract_json_object(content: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response did not contain a JSON action")


def _bounded_error(error: Exception) -> str:
    message = str(error).strip() or error.__class__.__name__
    return message[:1000]
