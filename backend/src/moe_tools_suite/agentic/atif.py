from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from .domain import (
    AgentDescriptor,
    AgenticModel,
    AgentTrajectoryView,
    CommandResult,
    InferenceCallSummary,
    TrajectoryStepType,
    TrajectoryStepView,
    VerifierView,
    utc_now,
)

ATIF_SCHEMA_VERSION = "ATIF-v1.7"

BASH_JSON_SYSTEM_PROMPT = """You are a coding agent working in an isolated Linux
sandbox.
Inspect the task, edit only files in the task workspace, and run useful checks. Respond
with exactly one JSON object per turn and no Markdown. To run a command, return
{"action":"shell","command":"..."}. When the task is complete, return
{"action":"finish","summary":"..."}. Never request credentials, network access,
or files outside the task workspace.
"""

BASH_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run one shell command in the isolated task workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


class AtifSource(StrEnum):
    SYSTEM = "system"
    USER = "user"
    AGENT = "agent"


class AtifAgent(AgenticModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    version: Annotated[str, Field(min_length=1, max_length=120)]
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class AtifToolCall(AgenticModel):
    tool_call_id: Annotated[str, Field(min_length=1, max_length=200)]
    function_name: Annotated[str, Field(min_length=1, max_length=120)]
    arguments: dict[str, Any]
    extra: dict[str, Any] | None = None


class AtifObservationResult(AgenticModel):
    content: Annotated[str, Field(max_length=200_000)]
    source_call_id: str | None = None
    timestamp: datetime | None = None
    extra: dict[str, Any] | None = None


class AtifObservation(AgenticModel):
    results: list[AtifObservationResult]


class AtifMetrics(AgenticModel):
    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    cached_tokens: Annotated[int, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None
    latency_ms: Annotated[float, Field(ge=0)] | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    extra: dict[str, Any] | None = None


class AtifStep(AgenticModel):
    step_id: Annotated[int, Field(ge=1)]
    timestamp: datetime | None = None
    source: AtifSource
    message: str
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    reasoning_content: str | None = None
    tool_calls: list[AtifToolCall] | None = None
    observation: AtifObservation | None = None
    metrics: AtifMetrics | None = None
    is_copied_context: bool | None = None
    llm_call_count: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def validate_source_fields(self) -> Self:
        if self.source is not AtifSource.AGENT:
            agent_only = (
                self.model_name,
                self.reasoning_effort,
                self.reasoning_content,
                self.tool_calls,
                self.llm_call_count,
            )
            if any(value is not None for value in agent_only):
                raise ValueError("agent-only fields require source='agent'")
        if self.tool_calls is not None:
            call_ids = [call.tool_call_id for call in self.tool_calls]
            if len(call_ids) != len(set(call_ids)):
                raise ValueError("tool_call_id values must be unique within a step")
        if self.observation is not None and self.tool_calls is not None:
            call_ids = {call.tool_call_id for call in self.tool_calls}
            linked_ids = {
                result.source_call_id
                for result in self.observation.results
                if result.source_call_id is not None
            }
            if not linked_ids.issubset(call_ids):
                raise ValueError("observation references an unknown tool call")
        return self


class AtifFinalMetrics(AgenticModel):
    total_prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    total_completion_tokens: Annotated[int, Field(ge=0)] | None = None
    total_cached_tokens: Annotated[int, Field(ge=0)] | None = None
    total_cost_usd: Annotated[float, Field(ge=0)] | None = None
    total_steps: Annotated[int, Field(ge=0)] | None = None
    extra: dict[str, Any] | None = None


class AtifTrajectory(AgenticModel):
    schema_version: Literal["ATIF-v1.7"] = ATIF_SCHEMA_VERSION
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: AtifAgent
    steps: list[AtifStep]
    notes: str | None = None
    final_metrics: AtifFinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_step_sequence(self) -> Self:
        expected = list(range(1, len(self.steps) + 1))
        actual = [step.step_id for step in self.steps]
        if actual != expected:
            raise ValueError("ATIF step IDs must be contiguous and start at 1")
        return self


def build_initial_trajectory(
    *,
    trial_id: str,
    agent: AgentDescriptor,
    model_name: str,
    instruction: str,
    system_prompt: str = BASH_JSON_SYSTEM_PROMPT,
    extra: dict[str, Any] | None = None,
) -> AtifTrajectory:
    """Create an ATIF trajectory with system and user task steps."""

    return AtifTrajectory(
        session_id=trial_id,
        trajectory_id=trial_id,
        agent=AtifAgent(
            name=agent.name,
            version=agent.revision,
            model_name=model_name,
            tool_definitions=[BASH_TOOL_DEFINITION],
            extra={"agent_id": agent.id},
        ),
        steps=[
            AtifStep(
                step_id=1,
                timestamp=utc_now(),
                source=AtifSource.SYSTEM,
                message=system_prompt,
            ),
            AtifStep(
                step_id=2,
                timestamp=utc_now(),
                source=AtifSource.USER,
                message=instruction,
            ),
        ],
        extra=extra,
    )


def make_shell_tool_call(
    *,
    tool_call_id: str,
    command: str,
) -> AtifToolCall:
    return AtifToolCall(
        tool_call_id=tool_call_id,
        function_name="bash",
        arguments={"command": command},
    )


def make_command_observation(
    *,
    tool_call_id: str,
    result: CommandResult,
) -> AtifObservation:
    output = result.stdout
    if result.stderr:
        output = f"{output}\n[stderr]\n{result.stderr}" if output else result.stderr
    return AtifObservation(
        results=[
            AtifObservationResult(
                source_call_id=tool_call_id,
                content=output,
                timestamp=utc_now(),
                extra={
                    "exit_code": result.exit_code,
                    "duration_ms": result.duration_ms,
                    "timed_out": result.timed_out,
                    "truncated": result.truncated,
                },
            )
        ]
    )


def append_agent_step(
    trajectory: AtifTrajectory,
    *,
    message: str,
    model_name: str,
    metrics: AtifMetrics,
    tool_calls: list[AtifToolCall] | None = None,
    observation: AtifObservation | None = None,
    reasoning_content: str | None = None,
    timestamp: datetime | None = None,
) -> AtifTrajectory:
    """Return a validated copy with one inference-aligned agent step appended."""

    step = AtifStep(
        step_id=len(trajectory.steps) + 1,
        timestamp=timestamp or utc_now(),
        source=AtifSource.AGENT,
        model_name=model_name,
        message=message,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        observation=observation,
        metrics=metrics,
        llm_call_count=1,
    )
    payload = trajectory.model_dump(mode="python")
    payload["steps"] = [*trajectory.steps, step]
    return AtifTrajectory.model_validate(payload)


def finalize_trajectory(
    trajectory: AtifTrajectory,
    *,
    extra_metrics: dict[str, Any] | None = None,
) -> AtifTrajectory:
    prompt_tokens = sum(
        step.metrics.prompt_tokens or 0
        for step in trajectory.steps
        if step.metrics is not None
    )
    completion_tokens = sum(
        step.metrics.completion_tokens or 0
        for step in trajectory.steps
        if step.metrics is not None
    )
    cached_tokens = sum(
        step.metrics.cached_tokens or 0
        for step in trajectory.steps
        if step.metrics is not None
    )
    payload = trajectory.model_dump(mode="python")
    payload["final_metrics"] = AtifFinalMetrics(
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_cached_tokens=cached_tokens,
        total_steps=len(trajectory.steps),
        extra=extra_metrics,
    )
    return AtifTrajectory.model_validate(payload)


def validate_trajectory(
    value: AtifTrajectory | Mapping[str, Any] | str | bytes,
) -> AtifTrajectory:
    if isinstance(value, AtifTrajectory):
        return AtifTrajectory.model_validate(value.model_dump(mode="python"))
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return AtifTrajectory.model_validate_json(value)
    return AtifTrajectory.model_validate(value)


def trajectory_to_json(trajectory: AtifTrajectory, *, indent: int = 2) -> str:
    validated = validate_trajectory(trajectory)
    return validated.model_dump_json(indent=indent, exclude_none=True) + "\n"


def trajectory_content_hash(trajectory: AtifTrajectory) -> str:
    canonical = json.dumps(
        validate_trajectory(trajectory).model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def trajectory_to_view(
    trajectory: AtifTrajectory,
    *,
    trial_id: str | None = None,
    inference_by_step: Mapping[int, InferenceCallSummary] | None = None,
    verifier: VerifierView | None = None,
    updated_at: datetime | None = None,
) -> AgentTrajectoryView:
    """Flatten ATIF messages, calls, and observations for the trajectory viewer."""

    validated = validate_trajectory(trajectory)
    inference_by_step = inference_by_step or {}
    flattened: list[TrajectoryStepView] = []

    def add_step(**values: Any) -> None:
        flattened.append(TrajectoryStepView(sequence=len(flattened) + 1, **values))

    for step in validated.steps:
        timestamp = step.timestamp or utc_now()
        if step.source is AtifSource.SYSTEM:
            add_step(
                id=f"atif-{step.step_id}-system",
                timestamp=timestamp,
                type=TrajectoryStepType.SYSTEM,
                title="System prompt",
                content=step.message,
            )
            continue
        if step.source is AtifSource.USER:
            add_step(
                id=f"atif-{step.step_id}-task",
                timestamp=timestamp,
                type=TrajectoryStepType.SYSTEM,
                title="Task instruction",
                content=step.message,
            )
            continue

        inference = inference_by_step.get(step.step_id) or _metrics_inference(step)
        add_step(
            id=f"atif-{step.step_id}-assistant",
            timestamp=timestamp,
            type=TrajectoryStepType.ASSISTANT,
            title="Agent response",
            content=step.message,
            inference=inference,
        )
        calls = step.tool_calls or []
        for call_index, call in enumerate(calls, start=1):
            command = (
                call.arguments.get("command")
                if isinstance(call.arguments.get("command"), str)
                else None
            )
            add_step(
                id=f"atif-{step.step_id}-tool-{call_index}",
                timestamp=timestamp,
                type=TrajectoryStepType.TOOL,
                title=f"Tool · {call.function_name}",
                content=json.dumps(call.arguments, indent=2, sort_keys=True),
                tool_name=call.function_name,
                command=command,
            )
        if step.observation is not None:
            for result_index, result in enumerate(step.observation.results, start=1):
                extra = result.extra or {}
                add_step(
                    id=f"atif-{step.step_id}-observation-{result_index}",
                    timestamp=result.timestamp or timestamp,
                    type=TrajectoryStepType.OBSERVATION,
                    title="Command observation",
                    content=result.content,
                    exit_code=_optional_int(extra.get("exit_code")),
                    duration_ms=_optional_float(extra.get("duration_ms")),
                    truncated=bool(extra.get("truncated", False)),
                )

    if verifier is not None:
        add_step(
            id="verifier",
            timestamp=updated_at or utc_now(),
            type=TrajectoryStepType.VERIFIER,
            title=verifier.summary,
            content=verifier.output,
            exit_code=verifier.exit_code,
            duration_ms=verifier.duration_ms,
        )

    latest_timestamp = updated_at or max(
        (step.timestamp for step in flattened),
        default=utc_now(),
    )
    resolved_trial_id = trial_id or validated.session_id or validated.trajectory_id
    if resolved_trial_id is None:
        raise ValueError("trajectory view requires a trial identifier")
    return AgentTrajectoryView(
        trial_id=resolved_trial_id,
        schema_version=validated.schema_version,
        steps=flattened,
        updated_at=latest_timestamp,
    )


def _metrics_inference(step: AtifStep) -> InferenceCallSummary | None:
    if step.metrics is None or step.metrics.extra is None:
        return None
    extra = step.metrics.extra
    inference_id = extra.get("inference_id")
    if not isinstance(inference_id, str) or not inference_id:
        return None
    routing_artifact_id = extra.get("routing_artifact_id")
    return InferenceCallSummary(
        id=inference_id,
        prompt_tokens=step.metrics.prompt_tokens or 0,
        completion_tokens=step.metrics.completion_tokens or 0,
        latency_ms=step.metrics.latency_ms or 0,
        routing_artifact_id=(
            routing_artifact_id if isinstance(routing_artifact_id, str) else None
        ),
        routed_layers=_optional_int(extra.get("routed_layers")) or 0,
        total_routed_slots=_optional_int(extra.get("total_routed_slots")) or 0,
    )


def _optional_int(value: object) -> int | None:
    return value if type(value) is int else None


def _optional_float(value: object) -> float | None:
    if type(value) in {int, float}:
        return float(value)
    return None


ATIFAgent = AtifAgent
ATIFToolCall = AtifToolCall
ATIFObservationResult = AtifObservationResult
ATIFObservation = AtifObservation
ATIFMetrics = AtifMetrics
ATIFStep = AtifStep
ATIFFinalMetrics = AtifFinalMetrics
ATIFTrajectory = AtifTrajectory
