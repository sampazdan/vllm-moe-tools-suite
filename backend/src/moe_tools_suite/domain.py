from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

_EMPTY_FINGERPRINT = "0" * 64


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
    DATASET_PREPARE = "dataset_prepare"
    AGENT_RUN = "agent_run"
    EXPERIMENT_RUN = "experiment_run"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelLoadPhase(StrEnum):
    QUEUED = "queued"
    RESOLVING_MODEL = "resolving_model"
    STOPPING_PREVIOUS = "stopping_previous"
    CONFIGURING_RUNTIME = "configuring_runtime"
    LAUNCHING_PROCESS = "launching_process"
    WAITING_FOR_READINESS = "waiting_for_readiness"
    STARTING_RUNTIME = "starting_runtime"
    VERIFYING_READY_CONTEXT = "verifying_ready_context"
    READY = "ready"
    CANCELLED = "cancelled"
    FAILED = "failed"


class JobPhaseRecord(BaseModel):
    phase: ModelLoadPhase
    status: Literal["active", "completed", "failed", "cancelled"] = "active"
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    detail: str | None = None


class JobRecord(BaseModel):
    id: str
    kind: JobKind
    status: JobStatus
    progress_current: Annotated[int, Field(ge=0)] = 0
    progress_total: Annotated[int, Field(ge=0)] = 0
    result_id: str | None = None
    error: str | None = None
    phase_history: list[JobPhaseRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ActiveJobConflictDetail(BaseModel):
    code: Literal["active_job_conflict"] = "active_job_conflict"
    message: str
    active_job: JobRecord
    recovery_url: str


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


class ModelRuntimeRecipe(BaseModel):
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    tensor_parallel_size: Annotated[int, Field(ge=1)] = 1
    max_model_len: Annotated[int, Field(ge=1)] = 4096
    reasoning_parser: str | None = None


class ModelRegistryEntry(BaseModel):
    id: str
    display_name: str
    enabled: bool
    revision: str | None = None
    topology: ModelTopology | None
    notes: str
    architecture: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    tensor_parallel_size: Annotated[int, Field(ge=1)] = 1
    context_defaults: dict[str, Any] = Field(default_factory=dict)
    chat_template: str | None = None
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    recommended_hardware: str | None = None
    minimum_memory_gib: Annotated[float, Field(gt=0)] | None = None
    masking_status: Literal["qualified", "experimental", "unsupported"] = "unsupported"
    routing_telemetry_status: Literal["qualified", "experimental", "unsupported"] = (
        "unsupported"
    )
    hot_switch_status: Literal["qualified", "experimental", "unsupported"] = (
        "unsupported"
    )
    qualification_status: Literal[
        "qualified", "experimental", "manifest_only", "unsupported"
    ] = "unsupported"
    last_live_evidence: str | None = None
    failure_reason: str | None = None
    runtime_recipe: ModelRuntimeRecipe | None = None

    @model_validator(mode="after")
    def validate_enabled_runtime(self) -> ModelRegistryEntry:
        if not self.enabled:
            return self
        if self.revision is None:
            raise ValueError("enabled models require an immutable revision")
        if self.topology is None:
            raise ValueError("enabled models require a manifest topology")
        if self.runtime_recipe is None:
            raise ValueError("enabled models require a runtime recipe")
        if self.runtime_recipe.revision != self.revision:
            raise ValueError("runtime recipe revision must match the manifest revision")
        if self.runtime_recipe.tensor_parallel_size != self.tensor_parallel_size:
            raise ValueError(
                "runtime recipe tensor parallel size must match the manifest"
            )
        return self


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
    AGENTIC = "agentic"
    MANUAL = "manual"
    IMPORT = "import"
    MULTI_SOURCE = "multi_source"


class ProfileSourceKind(StrEnum):
    BENCHMARK_RUN = "benchmark_run"
    AGENT_TRIAL = "agent_trial"


class ProfileSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ProfileSourceKind
    id: Annotated[str, Field(min_length=1, max_length=200)]
    weight: Annotated[float, Field(gt=0, le=100)] = 1


class CreateExpertProfileRequest(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=1000)] = ""
    model_id: str
    profile: ExpertProfile
    source: ProfileSource = ProfileSource.MANUAL
    source_run_id: str | None = None
    source_trial_id: str | None = None
    source_refs: list[ProfileSourceRef] = Field(default_factory=list, max_length=500)
    source_fingerprint: Annotated[
        str | None, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = None
    parent_profile_id: str | None = None
    metric: Literal["routing_mass", "selection_count"] | None = None
    selection_strategy: Annotated[str | None, Field(min_length=1, max_length=120)] = (
        None
    )
    selection_config: dict[str, Any] = Field(default_factory=dict)
    observed_mass_retained: Annotated[float, Field(ge=0, le=1)] | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("name cannot be blank")
        return normalized

    @field_validator("selection_config")
    @classmethod
    def validate_selection_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        try:
            encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError("selection_config must contain JSON values") from error
        if len(encoded.encode()) > 200_000:
            raise ValueError("selection_config exceeds 200000 bytes")
        return config

    @model_validator(mode="after")
    def validate_provenance(self) -> CreateExpertProfileRequest:
        if len({(ref.kind, ref.id) for ref in self.source_refs}) != len(
            self.source_refs
        ):
            raise ValueError("profile source references cannot contain duplicates")
        has_benchmark_source = self.source_run_id is not None or any(
            ref.kind is ProfileSourceKind.BENCHMARK_RUN for ref in self.source_refs
        )
        has_agent_source = self.source_trial_id is not None or any(
            ref.kind is ProfileSourceKind.AGENT_TRIAL for ref in self.source_refs
        )
        if self.source is ProfileSource.PROPOSAL and not has_benchmark_source:
            raise ValueError("proposal profiles require source_run_id")
        if self.source is ProfileSource.AGENTIC and not has_agent_source:
            raise ValueError("agentic profiles require source_trial_id")
        if self.source is ProfileSource.MULTI_SOURCE and len(self.source_refs) < 2:
            raise ValueError("multi-source profiles require at least two source_refs")
        if self.source_run_id is not None and self.source_trial_id is not None:
            raise ValueError("profile can reference a run or a trial, not both")
        has_source = bool(
            self.source_run_id or self.source_trial_id or self.source_refs
        )
        if self.metric is not None and not has_source:
            raise ValueError("profile metric requires source routing")
        if self.observed_mass_retained is not None and not has_source:
            raise ValueError("observed mass requires source routing")
        if self.source_fingerprint is not None and not self.source_refs:
            raise ValueError("source_fingerprint requires source_refs")
        if self.selection_config and self.selection_strategy is None:
            raise ValueError("selection_config requires selection_strategy")
        return self


class SavedExpertProfile(BaseModel):
    id: str
    name: str
    description: str
    model_id: str
    profile: ExpertProfile
    profile_fingerprint: str
    profile_fingerprint_version: Literal[1] | None = None
    source: ProfileSource
    source_run_id: str | None = None
    source_trial_id: str | None = None
    source_refs: list[ProfileSourceRef] = Field(default_factory=list)
    source_fingerprint: str | None = None
    parent_profile_id: str | None = None
    metric: Literal["routing_mass", "selection_count"] | None = None
    selection_strategy: str | None = None
    selection_config: dict[str, Any] = Field(default_factory=dict)
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
    model_revision: str | None = None
    topology: ModelTopology | None = None
    runtime_recipe: ModelRuntimeRecipe | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BenchmarkKind(StrEnum):
    FIXTURE = "fixture"
    STANDARD = "standard"
    CUSTOM = "custom"


class ScoringMode(StrEnum):
    EXACT = "exact"
    MULTIPLE_CHOICE = "multiple_choice"
    GSM8K = "gsm8k"
    IFEVAL = "ifeval"
    LIVEBENCH = "livebench"
    REGEX = "regex"
    CONTAINS = "contains"
    UNGRADED = "ungraded"


class ContractModel(BaseModel):
    """Immutable, fail-closed base for reproducible evaluation contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CriterionVisibility(StrEnum):
    PUBLIC = "public"
    HIDDEN = "hidden"


class DeterministicScorerKind(StrEnum):
    BENCHMARK_DEFAULT = "benchmark_default"
    EXACT = "exact"
    CONTAINS = "contains"
    REGEX = "regex"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"
    JSON = "json"
    UNGRADED = "ungraded"
    VERIFIER = "verifier"


class EvaluationCriterion(ContractModel):
    id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=120,
            pattern=r"^[a-z0-9][a-z0-9._-]*$",
        ),
    ] = "correctness"
    label: Annotated[str, Field(min_length=1, max_length=200)] = "Correct answer"
    description: Annotated[str, Field(max_length=2000)] = ""
    kind: DeterministicScorerKind = DeterministicScorerKind.BENCHMARK_DEFAULT
    visibility: CriterionVisibility = CriterionVisibility.PUBLIC
    required: bool = True
    weight: Annotated[float, Field(gt=0, le=100)] = 1
    case_sensitive: bool = True
    strip_whitespace: bool = True
    expected: Annotated[str | None, Field(max_length=100_000)] = None
    pattern: Annotated[str | None, Field(max_length=20_000)] = None
    numeric_tolerance: Annotated[float, Field(ge=0)] = 0
    json_schema: dict[str, Any] | None = None
    verifier_command: Annotated[str | None, Field(max_length=20_000)] = None
    verifier_timeout_seconds: Annotated[float, Field(ge=1, le=7200)] = 300

    @model_validator(mode="after")
    def validate_kind_configuration(self) -> EvaluationCriterion:
        if self.kind is DeterministicScorerKind.REGEX and self.pattern is not None:
            from .bounded_regex import validate_bounded_regex

            validate_bounded_regex(self.pattern)
        if (
            self.kind is DeterministicScorerKind.VERIFIER
            and not (self.verifier_command or "").strip()
        ):
            raise ValueError("verifier criteria require verifier_command")
        if self.kind is not DeterministicScorerKind.JSON and self.json_schema:
            raise ValueError("json_schema is only valid for JSON criteria")
        return self


class JudgeProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    FAKE = "fake"


class JudgeMode(StrEnum):
    SINGLE = "single"
    REFERENCE = "reference"
    PAIRWISE = "pairwise"


class LLMJudgeConfig(ContractModel):
    protocol_version: Literal["judge-json-v2"] = "judge-json-v2"
    provider: JudgeProvider
    model: Annotated[str, Field(min_length=1, max_length=200)]
    mode: JudgeMode = JudgeMode.SINGLE
    rubric: Annotated[str, Field(min_length=1, max_length=20_000)]
    pass_threshold: Annotated[float, Field(ge=0, le=1)] = 0.7
    repetitions: Annotated[int, Field(ge=1, le=9)] = 1
    temperature: Annotated[float, Field(ge=0, le=1)] | None = None
    max_output_tokens: Annotated[int, Field(ge=128, le=16_384)] = 2048
    input_cost_per_million_usd: Annotated[float, Field(ge=0)] | None = None
    output_cost_per_million_usd: Annotated[float, Field(ge=0)] | None = None
    rubric_hash: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = _EMPTY_FINGERPRINT
    fingerprint: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = _EMPTY_FINGERPRINT

    @model_validator(mode="after")
    def populate_fingerprints(self) -> LLMJudgeConfig:
        if (self.input_cost_per_million_usd is None) != (
            self.output_cost_per_million_usd is None
        ):
            raise ValueError("judge input and output prices must be provided together")
        rubric_hash = hashlib.sha256(self.rubric.encode()).hexdigest()
        if self.rubric_hash not in {_EMPTY_FINGERPRINT, rubric_hash}:
            raise ValueError("rubric_hash does not match rubric")
        object.__setattr__(self, "rubric_hash", rubric_hash)
        fingerprint = contract_fingerprint(self)
        if self.fingerprint not in {_EMPTY_FINGERPRINT, fingerprint}:
            raise ValueError("judge config fingerprint does not match configuration")
        object.__setattr__(self, "fingerprint", fingerprint)
        return self


class EvaluationAggregation(StrEnum):
    ALL_REQUIRED = "all_required"
    WEIGHTED_THRESHOLD = "weighted_threshold"


def _default_evaluation_criteria() -> list[EvaluationCriterion]:
    return [EvaluationCriterion()]


class EvaluationContract(ContractModel):
    version: Literal[1] = 1
    name: Annotated[str, Field(min_length=1, max_length=200)] = (
        "Benchmark default evaluation"
    )
    description: Annotated[str, Field(max_length=2000)] = ""
    criteria: Annotated[
        list[EvaluationCriterion], Field(min_length=1, max_length=100)
    ] = Field(default_factory=_default_evaluation_criteria)
    aggregation: EvaluationAggregation = EvaluationAggregation.ALL_REQUIRED
    pass_threshold: Annotated[float, Field(ge=0, le=1)] = 1
    judge: LLMJudgeConfig | None = None
    judge_weight: Annotated[float, Field(ge=0, le=1)] = 0
    judge_can_override_deterministic_failure: bool = False
    fingerprint: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = _EMPTY_FINGERPRINT

    @model_validator(mode="after")
    def validate_contract(self) -> EvaluationContract:
        if len({criterion.id for criterion in self.criteria}) != len(self.criteria):
            raise ValueError("evaluation criterion IDs must be unique")
        if self.judge is None and self.judge_weight != 0:
            raise ValueError("judge_weight requires a judge configuration")
        if self.judge is not None and self.judge_weight == 0:
            raise ValueError("judge configuration requires a positive judge_weight")
        fingerprint = contract_fingerprint(self)
        if self.fingerprint not in {_EMPTY_FINGERPRINT, fingerprint}:
            raise ValueError("evaluation fingerprint does not match contract")
        object.__setattr__(self, "fingerprint", fingerprint)
        return self


class ExecutionPolicy(ContractModel):
    version: Literal[1] = 1
    attempts: Annotated[int, Field(ge=1, le=20)] = 1
    concurrency: Annotated[int, Field(ge=1, le=64)] = 1
    timeout_seconds: Annotated[float, Field(ge=1, le=86_400)] = 3600
    per_item_timeout_seconds: Annotated[float, Field(ge=1, le=7200)] = 300
    max_turns: Annotated[int, Field(ge=1, le=1024)] | None = None
    max_commands: Annotated[int, Field(ge=1, le=4096)] | None = None
    max_tokens: (
        Annotated[
            int,
            Field(
                ge=1,
                le=10_000_000,
                description="Whole-run prompt plus generated-token cap.",
            ),
        ]
        | None
    ) = None
    max_cost_usd: (
        Annotated[
            float,
            Field(
                gt=0,
                le=100_000,
                description="Incremental frontier-judge API spend cap in USD.",
            ),
        ]
        | None
    ) = None
    fail_fast: bool = False
    fingerprint: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = _EMPTY_FINGERPRINT

    @model_validator(mode="after")
    def populate_fingerprint(self) -> ExecutionPolicy:
        fingerprint = contract_fingerprint(self)
        if self.fingerprint not in {_EMPTY_FINGERPRINT, fingerprint}:
            raise ValueError("execution policy fingerprint does not match policy")
        object.__setattr__(self, "fingerprint", fingerprint)
        return self


class CriterionResult(BaseModel):
    criterion_id: str
    kind: DeterministicScorerKind
    required: bool
    weight: float
    score: Annotated[float, Field(ge=0, le=1)] | None = None
    passed: bool | None = None
    explanation: str = ""
    error: str | None = None
    exit_code: int | None = None
    duration_ms: Annotated[float, Field(ge=0)] | None = None
    timed_out: bool = False
    execution_provenance: (
        Literal["trusted_task_verifier", "user_authored_sandbox"] | None
    ) = None
    output_sha256: Annotated[
        str | None, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = None


class JudgeUsage(BaseModel):
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    total_tokens: Annotated[int, Field(ge=0)] | None = None
    # Equivalent cost describes the provider-reported token usage even when the
    # result is later served from cache. Incurred cost is spend caused by this
    # evaluation. The upper bound is conservative budget accounting when actual
    # provider usage is unavailable.
    estimated_cost_usd: Annotated[float, Field(ge=0)] | None = None
    incurred_cost_usd: Annotated[float, Field(ge=0)] | None = None
    cost_upper_bound_usd: Annotated[float, Field(ge=0)] | None = None


class LLMJudgeResult(BaseModel):
    id: str
    provider: JudgeProvider
    model: str
    mode: JudgeMode
    protocol_version: str = "judge-json-v1"
    config_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    rubric_hash: Annotated[str, Field(min_length=64, max_length=64)]
    request_hash: Annotated[str, Field(min_length=64, max_length=64)]
    score: Annotated[float, Field(ge=0, le=1)]
    passed: bool
    confidence: Annotated[float, Field(ge=0, le=1)] | None = None
    rationale: Annotated[str, Field(max_length=20_000)] = ""
    verdict: Annotated[str | None, Field(max_length=80)] = None
    presentation_order: list[Literal["A", "B"]] = Field(default_factory=list)
    usage: JudgeUsage = Field(default_factory=JudgeUsage)
    latency_ms: Annotated[float, Field(ge=0)] = 0
    cached: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationResult(BaseModel):
    deterministic_score: Annotated[float, Field(ge=0, le=1)] | None = None
    judge_score: Annotated[float, Field(ge=0, le=1)] | None = None
    combined_score: Annotated[float, Field(ge=0, le=1)] | None = None
    passed: bool | None = None
    criteria: list[CriterionResult] = Field(default_factory=list)
    judge: LLMJudgeResult | None = None
    error: str | None = None


class InferencePerformance(BaseModel):
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0
    latency_ms: Annotated[float, Field(ge=0)] = 0
    ttft_ms: Annotated[float, Field(ge=0)] | None = None
    prefill_ms: Annotated[float, Field(ge=0)] | None = None
    decode_ms: Annotated[float, Field(ge=0)] | None = None
    queue_time_ms: Annotated[float, Field(ge=0)] | None = None
    mean_inter_token_latency_ms: Annotated[float, Field(ge=0)] | None = None
    tokens_per_second: Annotated[float, Field(ge=0)] | None = None
    estimated_cost_usd: Annotated[float, Field(ge=0)] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunPerformance(BaseModel):
    wall_time_ms: Annotated[float, Field(ge=0)] = 0
    model_time_ms: Annotated[float, Field(ge=0)] = 0
    evaluation_time_ms: Annotated[float, Field(ge=0)] = 0
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0
    mean_tokens_per_second: Annotated[float, Field(ge=0)] | None = None
    p50_tokens_per_second: Annotated[float, Field(ge=0)] | None = None
    p95_tokens_per_second: Annotated[float, Field(ge=0)] | None = None
    inference_cost_usd: Annotated[float, Field(ge=0)] | None = None
    judge_cost_usd: Annotated[float, Field(ge=0)] | None = None
    judge_equivalent_cost_usd: Annotated[float, Field(ge=0)] | None = None
    judge_budget_debit_usd: Annotated[float, Field(ge=0)] = 0
    judge_cost_uncertain: bool = False
    estimated_cost_usd: Annotated[float, Field(ge=0)] | None = None


class JudgeEvaluationRequest(ContractModel):
    config: LLMJudgeConfig
    task: Annotated[str, Field(min_length=1, max_length=100_000)]
    candidate: Annotated[str, Field(max_length=500_000)]
    reference: Annotated[str | None, Field(max_length=500_000)] = None
    baseline: Annotated[str | None, Field(max_length=500_000)] = None
    criteria: list[str] = Field(default_factory=list, max_length=100)
    request_hash: Annotated[
        str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ] = _EMPTY_FINGERPRINT

    @model_validator(mode="after")
    def validate_mode_inputs(self) -> JudgeEvaluationRequest:
        if self.config.mode is JudgeMode.REFERENCE and self.reference is None:
            raise ValueError("reference judge mode requires reference")
        if self.config.mode is JudgeMode.PAIRWISE and self.baseline is None:
            raise ValueError("pairwise judge mode requires baseline")
        request_hash = contract_fingerprint(self)
        if self.request_hash not in {_EMPTY_FINGERPRINT, request_hash}:
            raise ValueError("request_hash does not match judge request")
        object.__setattr__(self, "request_hash", request_hash)
        return self


class JudgeProviderCapability(BaseModel):
    provider: JudgeProvider
    configured: bool
    default_model: str
    supports_single: bool = True
    supports_reference: bool = True
    supports_pairwise: bool = True


class EvaluationCapabilities(BaseModel):
    deterministic_scorers: list[DeterministicScorerKind]
    judge_providers: list[JudgeProviderCapability]


class SuccessCriteriaView(BaseModel):
    summary: str
    public_criteria: list[EvaluationCriterion]
    hidden_criteria_count: Annotated[int, Field(ge=0)] = 0
    hidden_criteria_hash: str | None = None
    evaluation_contract_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    public_contract: EvaluationContract | None = None


class BenchmarkProblemDetail(BaseModel):
    benchmark: BenchmarkInfo
    item: BenchmarkItem
    rendered_prompt: str
    success_criteria: SuccessCriteriaView
    default_execution_policy: ExecutionPolicy


def contract_fingerprint(model: BaseModel) -> str:
    canonical = json.dumps(
        model.model_dump(mode="json", exclude={"fingerprint", "request_hash"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


class GenerationConfig(BaseModel):
    temperature: Annotated[float, Field(ge=0, le=2)] = 0
    max_tokens: Annotated[
        int,
        Field(ge=1, le=4096, description="Maximum generated tokens per response."),
    ] = 512
    seed: int | None = 0
    enable_thinking: bool = False


class BenchmarkItem(BaseModel):
    id: str
    prompt: str
    expected: str = ""
    category: str
    scoring: ScoringMode = ScoringMode.EXACT
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class BenchmarkInfo(BaseModel):
    id: str
    name: str
    description: str
    item_count: int
    categories: list[str]
    kind: BenchmarkKind = BenchmarkKind.FIXTURE
    source: str = "local"
    revision: str = "fixture-v1"
    split: str = "fixture"
    license: str = "internal"
    ready: bool = True
    scoring: ScoringMode = ScoringMode.EXACT
    prompt_template_version: str = "v1"
    default_generation: GenerationConfig = Field(default_factory=GenerationConfig)


class BenchmarkItemPage(BaseModel):
    benchmark: BenchmarkInfo
    items: list[BenchmarkItem]
    total: int
    offset: int
    limit: int
    categories: list[str]


class CreateCustomBenchmarkRequest(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=1000)] = ""
    content: Annotated[str, Field(min_length=1, max_length=5_000_000)]

    @field_validator("name")
    @classmethod
    def normalize_dataset_name(cls, name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("name cannot be blank")
        return normalized


class BenchmarkDatasetRecord(BaseModel):
    id: str
    info: BenchmarkInfo
    content_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BenchmarkCohort(BaseModel):
    id: str
    benchmark_id: str
    benchmark_revision: str
    dataset_content_hash: str
    prompt_template_version: str
    scoring_version: str
    item_ids: list[str]
    fingerprint: str
    generation: GenerationConfig
    evaluation_contract: EvaluationContract | None = None
    execution_policy: ExecutionPolicy | None = None
    evaluation_contract_fingerprint: str | None = None
    execution_policy_fingerprint: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def populate_contract_fingerprints(self) -> BenchmarkCohort:
        expected_evaluation = (
            self.evaluation_contract.fingerprint
            if self.evaluation_contract is not None
            else None
        )
        expected_policy = (
            self.execution_policy.fingerprint
            if self.execution_policy is not None
            else None
        )
        if (
            expected_evaluation is not None
            and self.evaluation_contract_fingerprint is not None
            and self.evaluation_contract_fingerprint != expected_evaluation
        ):
            raise ValueError("evaluation contract fingerprint does not match contract")
        if (
            expected_policy is not None
            and self.execution_policy_fingerprint is not None
            and self.execution_policy_fingerprint != expected_policy
        ):
            raise ValueError("execution policy fingerprint does not match policy")
        if expected_evaluation is not None:
            self.evaluation_contract_fingerprint = expected_evaluation
        if expected_policy is not None:
            self.execution_policy_fingerprint = expected_policy
        return self


class BenchmarkCohortView(BaseModel):
    id: str
    benchmark_id: str
    benchmark_revision: str
    dataset_content_hash: str
    prompt_template_version: str
    scoring_version: str
    item_count: Annotated[int, Field(ge=0)]
    fingerprint: str
    generation: GenerationConfig
    evaluation_contract: EvaluationContract | None = None
    evaluation_contract_fingerprint: str | None = None
    hidden_criteria_count: Annotated[int, Field(ge=0)] = 0
    hidden_criteria_hash: str | None = None
    execution_policy: ExecutionPolicy | None = None
    execution_policy_fingerprint: str | None = None
    created_at: datetime


class RunRequest(BaseModel):
    benchmark_id: str = "fixture-arithmetic"
    item_ids: list[str] | None = None
    generation: GenerationConfig | None = None
    evaluation_contract: EvaluationContract | None = None
    execution_policy: ExecutionPolicy | None = None


class RunItemResult(BaseModel):
    item_id: str
    attempt: Annotated[int, Field(ge=1)] = 1
    prompt: str
    expected: str
    output: str
    passed: bool | None
    scoring: ScoringMode = ScoringMode.EXACT
    error: str | None = None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    total_tokens: Annotated[int, Field(ge=0)] = 0
    judge_budget_debit_usd: Annotated[float, Field(ge=0)] = 0
    judge_cost_uncertain: bool = False
    evaluation: EvaluationResult | None = None
    performance: InferencePerformance | None = None


class BenchmarkRun(BaseModel):
    id: str
    benchmark_id: str
    model_session_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    score: float | None
    scored_items: int = 0
    completed_items: int
    total_items: int
    items: list[RunItemResult]
    cohort_id: str | None = None
    evaluation_contract_fingerprint: str | None = None
    execution_policy_fingerprint: str | None = None
    performance: RunPerformance | None = None
    termination_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BenchmarkRunSummary(BaseModel):
    """Bounded benchmark-run metadata for archives and command-center headers."""

    id: str
    benchmark_id: str
    model_session_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    score: float | None
    scored_items: int = 0
    completed_items: int
    total_items: int
    cohort_id: str | None = None
    evaluation_contract_fingerprint: str | None = None
    execution_policy_fingerprint: str | None = None
    performance: RunPerformance | None = None
    termination_reason: str | None = None
    created_at: datetime


class BenchmarkRunPage(BaseModel):
    items: list[BenchmarkRunSummary]
    total: Annotated[int, Field(ge=0)]
    offset: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1)]


class RunItemResultPage(BaseModel):
    run_id: str
    items: list[RunItemResult]
    total: Annotated[int, Field(ge=0)]
    offset: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1)]


class RunProvenance(BaseModel):
    run_id: str
    cohort_id: str
    cohort_fingerprint: str
    model_id: str
    model_session_id: str
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    benchmark_id: str
    benchmark_revision: str
    dataset_content_hash: str
    prompt_template_version: str
    scoring_version: str
    generation: GenerationConfig
    evaluation_contract: EvaluationContract | None = None
    execution_policy: ExecutionPolicy | None = None
    evaluation_contract_fingerprint: str | None = None
    execution_policy_fingerprint: str | None = None
    app_version: str

    @model_validator(mode="after")
    def validate_contract_fingerprints(self) -> RunProvenance:
        expected_evaluation = (
            self.evaluation_contract.fingerprint
            if self.evaluation_contract is not None
            else None
        )
        expected_policy = (
            self.execution_policy.fingerprint
            if self.execution_policy is not None
            else None
        )
        if (
            expected_evaluation is not None
            and self.evaluation_contract_fingerprint is not None
            and self.evaluation_contract_fingerprint != expected_evaluation
        ):
            raise ValueError("evaluation contract fingerprint does not match contract")
        if (
            expected_policy is not None
            and self.execution_policy_fingerprint is not None
            and self.execution_policy_fingerprint != expected_policy
        ):
            raise ValueError("execution policy fingerprint does not match policy")
        if expected_evaluation is not None:
            self.evaluation_contract_fingerprint = expected_evaluation
        if expected_policy is not None:
            self.execution_policy_fingerprint = expected_policy
        return self


class RunProvenanceView(BaseModel):
    run_id: str
    cohort_id: str
    cohort_fingerprint: str
    model_id: str
    model_session_id: str
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    benchmark_id: str
    benchmark_revision: str
    dataset_content_hash: str
    prompt_template_version: str
    scoring_version: str
    generation: GenerationConfig
    evaluation_contract: EvaluationContract | None = None
    evaluation_contract_fingerprint: str | None = None
    hidden_criteria_count: Annotated[int, Field(ge=0)] = 0
    hidden_criteria_hash: str | None = None
    execution_policy: ExecutionPolicy | None = None
    execution_policy_fingerprint: str | None = None
    app_version: str


class RunDetail(BaseModel):
    run: BenchmarkRunSummary
    model_session: ModelSession
    saved_profile: SavedExpertProfile | None = None
    cohort: BenchmarkCohortView | None = None
    provenance: RunProvenanceView | None = None


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
    baseline_score: float | None
    candidate_score: float | None
    score_delta: float | None
    regressions: int
    recoveries: int
    retained_passes: int
    retained_failures: int
    unscored_items: int = 0
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
    model_id: str | None = None
    profile: ExpertProfile | None = None
    profile_id: str | None = None


class SystemStatus(BaseModel):
    mode: Literal["mock", "vllm"]
    version: str
    model_state: ModelState
    data_dir: str


class RuntimeStatus(BaseModel):
    managed: bool
    model_id: str
    model_revision: str | None = None
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
