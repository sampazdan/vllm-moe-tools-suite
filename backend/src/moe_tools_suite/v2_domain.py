from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .domain import (
    EvaluationContract,
    EvaluationResult,
    ExecutionPolicy,
    GenerationConfig,
    InferencePerformance,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_fingerprint(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class V2Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkloadKind(StrEnum):
    ANSWER = "answer"
    CODING = "coding"
    STATE_DRIFT = "state_drift"


class ExperimentStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentLaneRole(StrEnum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"


class InterventionKind(StrEnum):
    BASELINE = "baseline"
    EXPERT_MASK = "expert_mask"


class WorkloadRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunUnitStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    UNSCORED = "unscored"
    ERROR = "error"
    CANCELLED = "cancelled"


class RunEventKind(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    CURRENT_UNIT = "current_unit"
    CURRENT_CHECKPOINT = "current_checkpoint"
    MODEL_REQUEST_STARTED = "model_request_started"
    INFERENCE_PROGRESS = "inference_progress"
    MODEL_REQUEST_COMPLETED = "model_request_completed"
    TOOL_COMMAND_STARTED = "tool_command_started"
    TOOL_COMMAND_COMPLETED = "tool_command_completed"
    EVALUATING = "evaluating"
    UNIT_PASSED = "unit_passed"
    UNIT_FAILED = "unit_failed"
    UNIT_UNSCORED = "unit_unscored"
    TRAJECTORY_STEP = "trajectory_step"
    CONTEXT_SWITCHED = "context_switched"
    RETRYING = "retrying"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DriftCondition(StrEnum):
    ORACLE_RESET = "oracle_reset"
    CHAINED = "chained"
    STATE_ANCHORED = "state_anchored"


class DriftParameters(V2Model):
    """Deterministic dimensions applied to every first-party drift scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dependency_span: Annotated[int, Field(ge=1, le=32)] = 1
    branch_count: Annotated[int, Field(ge=1, le=16)] = 1
    rollback_depth: Annotated[int, Field(ge=0, le=16)] = 0
    distractor_ratio: Annotated[float, Field(ge=0, le=0.9)] = 0
    tool_error_rate: Annotated[float, Field(ge=0, le=0.9)] = 0
    state_size: Annotated[int, Field(ge=3, le=128)] = 3

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {"version": "drift-parameters-v1", **self.model_dump(mode="json")}
        )

    @model_validator(mode="after")
    def validate_state_shape(self) -> Self:
        if self.branch_count > self.state_size:
            raise ValueError("branch_count cannot exceed state_size")
        if self.dependency_span >= self.state_size:
            raise ValueError("dependency_span must be smaller than state_size")
        return self


class InterventionContextRef(V2Model):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_id: Annotated[str, Field(min_length=1, max_length=200)]
    kind: InterventionKind
    model_id: Annotated[str, Field(min_length=1, max_length=300)]
    model_revision: str | None = None
    topology_fingerprint: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    profile_id: str | None = None
    profile_fingerprint: Annotated[
        str | None, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = None
    context_fingerprint: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    creation_source: Annotated[str, Field(min_length=1, max_length=120)]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        if self.kind is InterventionKind.BASELINE and (
            self.profile_id is not None or self.profile_fingerprint is not None
        ):
            raise ValueError("baseline context cannot reference an expert profile")
        if self.kind is InterventionKind.EXPERT_MASK and (
            self.profile_id is None or self.profile_fingerprint is None
        ):
            raise ValueError("expert-mask context requires profile provenance")
        return self


class ContextActivationResult(V2Model):
    old_context_id: Annotated[str | None, Field(min_length=1, max_length=200)] = None
    old_context_fingerprint: Annotated[
        str | None, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = None
    new_context_id: Annotated[str, Field(min_length=1, max_length=200)]
    new_context_fingerprint: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    topology_fingerprint: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    duration_ms: Annotated[float, Field(ge=0)]
    process_id: Annotated[int | None, Field(gt=0)] = None
    weights_reloaded: Literal[False] = False
    activated_at: datetime = Field(default_factory=utc_now)


class ActivateExpertContextRequest(V2Model):
    profile_id: str | None = None


class WorkloadDescriptor(V2Model):
    id: Annotated[str, Field(min_length=1, max_length=240)]
    kind: WorkloadKind
    name: Annotated[str, Field(min_length=1, max_length=200)]
    description: Annotated[str, Field(max_length=20_000)] = ""
    source: Annotated[str, Field(min_length=1, max_length=300)]
    revision: Annotated[str, Field(min_length=1, max_length=160)]
    content_fingerprint: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    ready: bool
    blocked_reason: str | None = None
    unit_ids: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    task_id: str | None = None
    language: str | None = None
    difficulty: str | None = None
    expected_horizon: Annotated[int, Field(ge=1)] | None = None
    tools: list[str] = Field(default_factory=list)
    runtime: str | None = None
    preparation_status: str | None = None
    public_problem_statement: str | None = None
    public_success_criteria: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerformanceSnapshot(V2Model):
    id: str
    workload_run_id: str
    run_unit_id: str | None = None
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0
    latency_ms: Annotated[float, Field(ge=0)] = 0
    tokens_per_second: Annotated[float, Field(ge=0)] | None = None
    context_activation_ms: Annotated[float, Field(ge=0)] | None = None
    inference: InferencePerformance | None = None
    captured_at: datetime = Field(default_factory=utc_now)


class EvaluationResultRecord(V2Model):
    id: str
    workload_run_id: str
    run_unit_id: str | None = None
    passed: bool | None = None
    score: Annotated[float, Field(ge=0, le=1)] | None = None
    metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)
    formula_fingerprint: Annotated[
        str | None, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = None
    deterministic: EvaluationResult | None = None
    created_at: datetime = Field(default_factory=utc_now)


class RunUnit(V2Model):
    id: str
    experiment_id: str
    lane_id: str
    workload_run_id: str
    ordinal: Annotated[int, Field(ge=0)]
    unit_key: Annotated[str, Field(min_length=1, max_length=300)]
    workload_unit_id: Annotated[str | None, Field(min_length=1, max_length=240)] = None
    status: RunUnitStatus = RunUnitStatus.QUEUED
    scenario_id: str | None = None
    condition: DriftCondition | None = None
    horizon: Annotated[int, Field(ge=1)] | None = None
    seed: int | None = None
    result: dict[str, Any] | None = None
    evaluation: EvaluationResultRecord | None = None
    performance: PerformanceSnapshot | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WorkloadRun(V2Model):
    id: str
    experiment_id: str
    lane_id: str
    workload: WorkloadDescriptor
    context: InterventionContextRef
    status: WorkloadRunStatus = WorkloadRunStatus.QUEUED
    completed_units: Annotated[int, Field(ge=0)] = 0
    passed_units: Annotated[int, Field(ge=0)] = 0
    total_units: Annotated[int, Field(ge=0)]
    aggregate_metrics: dict[str, str | float | int | bool | list[float] | None] = Field(
        default_factory=dict
    )
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class ExperimentLane(V2Model):
    id: str
    experiment_id: str
    role: ExperimentLaneRole
    label: Annotated[str, Field(min_length=1, max_length=120)]
    context: InterventionContextRef
    workload_run_id: str
    status: WorkloadRunStatus = WorkloadRunStatus.QUEUED


class RunEvent(V2Model):
    experiment_id: str
    sequence: Annotated[int, Field(ge=1)]
    kind: RunEventKind
    phase: Annotated[str, Field(min_length=1, max_length=120)]
    message: Annotated[str, Field(max_length=2000)] = ""
    lane_id: str | None = None
    workload_run_id: str | None = None
    run_unit_id: str | None = None
    checkpoint: Annotated[int, Field(ge=0)] | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RunEventPage(V2Model):
    experiment_id: str
    events: list[RunEvent]
    after: Annotated[int, Field(ge=0)]
    latest_sequence: Annotated[int, Field(ge=0)]
    has_more: bool = False


class Experiment(V2Model):
    id: str
    job_id: str
    name: Annotated[str, Field(min_length=1, max_length=200)]
    model_id: str
    workload: WorkloadDescriptor
    cohort_fingerprint: Annotated[
        str | None, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = None
    contract_provenance: Literal["known", "legacy_unknown"] = "legacy_unknown"
    generation: GenerationConfig | None = None
    evaluation_contract: EvaluationContract | None = None
    execution_policy: ExecutionPolicy | None = None
    drift_parameters: DriftParameters | None = None
    execution_config: dict[str, Any] = Field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.QUEUED
    lanes: list[ExperimentLane]
    completed_units: Annotated[int, Field(ge=0)] = 0
    passed_units: Annotated[int, Field(ge=0)] = 0
    total_units: Annotated[int, Field(ge=1)]
    latest_event_sequence: Annotated[int, Field(ge=0)] = 0
    comparison: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_lanes(self) -> Self:
        roles = [lane.role for lane in self.lanes]
        if roles.count(ExperimentLaneRole.BASELINE) != 1:
            raise ValueError("experiment requires exactly one baseline lane")
        if len(roles) != len(set(roles)):
            raise ValueError("experiment lane roles must be unique")
        contracts = (
            self.generation,
            self.evaluation_contract,
            self.execution_policy,
        )
        if self.contract_provenance == "known" and any(
            contract is None for contract in contracts
        ):
            raise ValueError("known experiment contracts must be fully populated")
        if self.contract_provenance == "legacy_unknown" and any(
            contract is not None for contract in contracts
        ):
            raise ValueError("legacy experiment contract provenance must stay unknown")
        return self


class ExperimentDetail(V2Model):
    experiment: Experiment
    workload_runs: list[WorkloadRun]
    units: list[RunUnit]


class CreateExperimentRequest(V2Model):
    name: Annotated[str | None, Field(max_length=200)] = None
    model_id: Annotated[str, Field(min_length=1, max_length=300)]
    workload_id: Annotated[str, Field(min_length=1, max_length=240)]
    workload_unit_ids: Annotated[
        list[str] | None, Field(min_length=1, max_length=10_000)
    ] = None
    candidate_profile_id: str | None = None
    agent_id: Annotated[str, Field(min_length=1, max_length=128)] = "bash-json-v1"
    sandbox_provider_id: Annotated[str, Field(min_length=1, max_length=128)] = "fake"
    scenario_ids: Annotated[list[str], Field(min_length=1, max_length=20)] = Field(
        default_factory=lambda: ["ledger-reconciliation-v1"]
    )
    conditions: Annotated[list[DriftCondition], Field(min_length=1, max_length=3)] = (
        Field(
            default_factory=lambda: [
                DriftCondition.ORACLE_RESET,
                DriftCondition.CHAINED,
                DriftCondition.STATE_ANCHORED,
            ]
        )
    )
    horizons: Annotated[list[int], Field(min_length=1, max_length=6)] = Field(
        default_factory=lambda: [2, 4, 8]
    )
    seeds: Annotated[list[int], Field(min_length=1, max_length=20)] = Field(
        default_factory=lambda: [0]
    )
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation_contract: EvaluationContract | None = None
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    drift_parameters: DriftParameters = Field(default_factory=DriftParameters)

    @field_validator(
        "scenario_ids", "conditions", "horizons", "seeds", "workload_unit_ids"
    )
    @classmethod
    def require_unique(cls, values: list[Any] | None) -> list[Any] | None:
        if values is None:
            return None
        if len(values) != len(set(values)):
            raise ValueError("experiment dimensions cannot contain duplicates")
        return values

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, values: list[int]) -> list[int]:
        if any(value not in {2, 4, 8, 12, 16, 24} for value in values):
            raise ValueError("supported drift horizons are 2, 4, 8, 12, 16, and 24")
        return values

    @model_validator(mode="after")
    def validate_drift_dimensions(self) -> Self:
        depth = self.drift_parameters.rollback_depth
        if depth and min(self.horizons) < depth + 2:
            raise ValueError(
                "each drift horizon must fit a snapshot, rollback span, and rollback"
            )
        return self


class UpdateProfileMetadataRequest(V2Model):
    name: Annotated[str | None, Field(min_length=1, max_length=120)] = None
    description: Annotated[str | None, Field(max_length=1000)] = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("name cannot be blank")
        return value

    @model_validator(mode="after")
    def require_update(self) -> Self:
        if self.name is None and self.description is None:
            raise ValueError("profile metadata update is empty")
        return self
