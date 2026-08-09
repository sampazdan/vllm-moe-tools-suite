from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelState(StrEnum):
    UNLOADED = "unloaded"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class JobKind(StrEnum):
    MODEL_LOAD = "model_load"
    BENCHMARK_RUN = "benchmark_run"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRecord(BaseModel):
    id: str
    kind: JobKind
    status: JobStatus
    progress_current: Annotated[int, Field(ge=0)] = 0
    progress_total: Annotated[int, Field(ge=0)] = 0
    result_id: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ModelTopology(BaseModel):
    num_layers: Annotated[int, Field(gt=0)]
    num_experts: Annotated[int, Field(gt=0)]
    top_k: Annotated[int, Field(gt=0)]
    routed_layer_ids: list[int]

    @model_validator(mode="after")
    def validate_topology(self) -> ModelTopology:
        if len(self.routed_layer_ids) != self.num_layers:
            raise ValueError("routed layer count must match num_layers")
        if len(set(self.routed_layer_ids)) != len(self.routed_layer_ids):
            raise ValueError("routed layer IDs must be unique")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        return self


class ModelRegistryEntry(BaseModel):
    id: str
    display_name: str
    enabled: bool
    revision: str | None = None
    topology: ModelTopology
    notes: str


class ProfileLayer(BaseModel):
    keep: list[int]

    @field_validator("keep", mode="before")
    @classmethod
    def validate_keep(cls, keep: object) -> object:
        if not isinstance(keep, list):
            raise ValueError("keep must be a list of integer expert IDs")
        if any(type(expert_id) is not int for expert_id in keep):
            raise ValueError("keep must contain only integer expert IDs")
        if len(keep) != len(set(keep)):
            raise ValueError("keep cannot contain duplicate expert IDs")
        return keep


class ExpertProfile(BaseModel):
    version: Literal[1] = 1
    layers: dict[str, ProfileLayer]

    @field_validator("layers")
    @classmethod
    def validate_layer_keys(
        cls, layers: dict[str, ProfileLayer]
    ) -> dict[str, ProfileLayer]:
        if not layers:
            raise ValueError("profile must contain at least one layer")
        normalized_ids: set[int] = set()
        for raw_layer_id in layers:
            if not raw_layer_id.isascii() or not raw_layer_id.isdigit():
                raise ValueError(f"invalid layer ID {raw_layer_id!r}")
            layer_id = int(raw_layer_id)
            if layer_id in normalized_ids:
                raise ValueError(f"duplicate normalized layer ID {layer_id}")
            normalized_ids.add(layer_id)
        return layers


class ProfileValidation(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    eligible_experts: int = 0
    total_experts: int = 0
    retained_fraction: float = 0.0


class ProfileSource(StrEnum):
    PROPOSAL = "proposal"
    MANUAL = "manual"
    IMPORT = "import"


class CreateExpertProfileRequest(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=1000)] = ""
    model_id: str
    profile: ExpertProfile
    source: ProfileSource = ProfileSource.MANUAL
    source_run_id: str | None = None
    parent_profile_id: str | None = None
    metric: Literal["routing_mass", "selection_count"] | None = None
    observed_mass_retained: Annotated[float, Field(ge=0, le=1)] | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("name cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_provenance(self) -> CreateExpertProfileRequest:
        if self.source is ProfileSource.PROPOSAL and self.source_run_id is None:
            raise ValueError("proposal profiles require source_run_id")
        if self.metric is not None and self.source_run_id is None:
            raise ValueError("profile metric requires source_run_id")
        if self.observed_mass_retained is not None and self.source_run_id is None:
            raise ValueError("observed mass requires source_run_id")
        return self


class SavedExpertProfile(BaseModel):
    id: str
    name: str
    description: str
    model_id: str
    profile: ExpertProfile
    profile_fingerprint: str
    source: ProfileSource
    source_run_id: str | None = None
    parent_profile_id: str | None = None
    metric: Literal["routing_mass", "selection_count"] | None = None
    validation: ProfileValidation
    observed_mass_retained: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelSession(BaseModel):
    id: str
    model_id: str
    state: ModelState
    mode: Literal["mock", "vllm"]
    profile: ExpertProfile | None = None
    profile_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BenchmarkItem(BaseModel):
    id: str
    prompt: str
    expected: str
    category: str


class BenchmarkInfo(BaseModel):
    id: str
    name: str
    description: str
    item_count: int
    categories: list[str]


class RunRequest(BaseModel):
    benchmark_id: str = "fixture-arithmetic"
    item_ids: list[str] | None = None


class RunItemResult(BaseModel):
    item_id: str
    prompt: str
    expected: str
    output: str
    passed: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


class BenchmarkRun(BaseModel):
    id: str
    benchmark_id: str
    model_session_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    score: float
    completed_items: int
    total_items: int
    items: list[RunItemResult]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RunDetail(BaseModel):
    run: BenchmarkRun
    model_session: ModelSession
    saved_profile: SavedExpertProfile | None = None


class CreateComparisonRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    name: Annotated[str | None, Field(max_length=120)] = None

    @field_validator("name")
    @classmethod
    def normalize_optional_name(cls, name: str | None) -> str | None:
        if name is None:
            return None
        return name.strip() or None


class ComparisonRecord(BaseModel):
    id: str
    name: str
    baseline_run_id: str
    candidate_run_id: str
    profile_id: str | None = None
    profile_fingerprint: str
    benchmark_id: str
    cohort_item_ids: list[str]
    baseline_score: float
    candidate_score: float
    score_delta: float
    regressions: int
    recoveries: int
    retained_passes: int
    retained_failures: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RoutingSummary(BaseModel):
    run_id: str
    layer_ids: list[int]
    selection_counts: list[list[int]]
    routing_mass: list[list[float]]
    total_routed_slots: int


class ProfileProposalRequest(BaseModel):
    run_id: str
    keep_per_layer: Annotated[int, Field(gt=0)] = 64
    metric: Literal["routing_mass", "selection_count"] = "routing_mass"


class ProfileProposal(BaseModel):
    profile: ExpertProfile
    validation: ProfileValidation
    observed_mass_retained: float


class CreateModelSessionRequest(BaseModel):
    model_id: str
    profile: ExpertProfile | None = None
    profile_id: str | None = None

    @model_validator(mode="after")
    def validate_profile_reference(self) -> CreateModelSessionRequest:
        if self.profile_id is not None and self.profile is None:
            raise ValueError("profile_id requires profile")
        return self


class SystemStatus(BaseModel):
    mode: Literal["mock", "vllm"]
    version: str
    model_state: ModelState
    data_dir: str


class RuntimeStatus(BaseModel):
    managed: bool
    model_id: str
    pid: int | None = None
    session_id: str | None = None
    started_at: datetime | None = None
    log_path: str | None = None
    profile_path: str | None = None
    log_tail: str = ""


class LoginRequest(BaseModel):
    token: str = Field(min_length=1, max_length=4096)


class SessionStatus(BaseModel):
    auth_required: bool
    authenticated: bool
    csrf_token: str | None = None
    expires_at: datetime | None = None
