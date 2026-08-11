from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..domain import (
    DeterministicScorerKind,
    EvaluationAggregation,
    EvaluationContract,
    EvaluationCriterion,
    EvaluationResult,
    ExecutionPolicy,
    GenerationConfig,
    LLMJudgeConfig,
)

AgenticId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    ),
]


def utc_now() -> datetime:
    return datetime.now(UTC)


class AgenticModel(BaseModel):
    """Base model for fail-closed agentic API and resource contracts."""

    model_config = ConfigDict(extra="forbid")


class CredentialMode(StrEnum):
    API_KEY = "api_key"
    JWT = "jwt"


class ProviderStatus(StrEnum):
    NOT_CONFIGURED = "not_configured"
    UNCHECKED = "unchecked"
    READY = "ready"
    ERROR = "error"


class NetworkPolicy(StrEnum):
    NONE = "none"
    ALLOWLIST = "allowlist"
    PUBLIC = "public"


class SandboxState(StrEnum):
    CREATING = "creating"
    READY = "ready"
    DELETING = "deleting"
    DELETED = "deleted"
    ERROR = "error"


class SandboxSessionState(StrEnum):
    PENDING = "pending"
    CREATING = "creating"
    READY = "ready"
    DELETING = "deleting"
    DELETED = "deleted"
    CLEANUP_PENDING = "cleanup_pending"
    ERROR = "error"


class AgentRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class AgentTrialStatus(StrEnum):
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    VERIFYING = "verifying"
    CLEANING = "cleaning"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    CANCELLED = "cancelled"


class TerminationCause(StrEnum):
    AGENT_FINISHED = "agent_finished"
    TURN_LIMIT = "turn_limit"
    TOKEN_LIMIT = "token_limit"
    COST_LIMIT = "cost_limit"
    TIME_LIMIT = "time_limit"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    MODEL_ERROR = "model_error"
    SANDBOX_ERROR = "sandbox_error"
    VERIFIER_ERROR = "verifier_error"
    CLEANUP_ERROR = "cleanup_error"


class TrajectoryStepType(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    OBSERVATION = "observation"
    VERIFIER = "verifier"
    REASONING = "reasoning"


class ReasoningMode(StrEnum):
    OFF = "off"
    COMPACT = "compact"
    FULL = "full"


class VerifierStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ProviderCapabilities(AgenticModel):
    create: bool = True
    exec: bool = True
    upload: bool = True
    download: bool = True
    delete: bool = True

    @property
    def remote_execution(self) -> bool:
        return self.create and self.exec

    @property
    def file_transfer(self) -> bool:
        return self.upload and self.download


class ProviderError(AgenticModel):
    code: Annotated[str, Field(min_length=1, max_length=80)]
    message: Annotated[str, Field(min_length=1, max_length=500)]
    retryable: bool = False


class SandboxProviderInfo(AgenticModel):
    id: AgenticId
    label: Annotated[str, Field(min_length=1, max_length=120)]
    configured: bool
    credential_mode: CredentialMode | None = None
    required_env: list[str] = Field(default_factory=list)
    optional_env: list[str] = Field(default_factory=list)
    region: str | None = None
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    status: ProviderStatus = ProviderStatus.UNCHECKED
    last_checked_at: datetime | None = None
    error: ProviderError | None = None


class ProviderPreflight(AgenticModel):
    provider_id: AgenticId
    configured: bool
    reachable: bool
    authenticated: bool
    status: ProviderStatus
    latency_ms: Annotated[float, Field(ge=0)] | None = None
    checked_at: datetime = Field(default_factory=utc_now)
    api_url_host: str | None = None
    region: str | None = None
    error: ProviderError | None = None


class SandboxSpec(AgenticModel):
    image_ref: Annotated[str, Field(min_length=1, max_length=500)]
    image_digest: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    working_directory: str = "/workspace/task"
    network_policy: NetworkPolicy = NetworkPolicy.NONE
    allowed_hosts: list[str] = Field(default_factory=list)
    cpu: Annotated[int, Field(ge=1, le=32)] = 1
    memory_mb: Annotated[int, Field(ge=256, le=131_072)] = 1024
    disk_mb: Annotated[int, Field(ge=256, le=1_048_576)] = 2048

    @field_validator("working_directory")
    @classmethod
    def validate_working_directory(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("sandbox working directory must be an absolute safe path")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_network_policy(self) -> Self:
        if self.network_policy is NetworkPolicy.ALLOWLIST and not self.allowed_hosts:
            raise ValueError("allowlist network policy requires allowed_hosts")
        if self.network_policy is not NetworkPolicy.ALLOWLIST and self.allowed_hosts:
            raise ValueError("allowed_hosts requires the allowlist network policy")
        return self


class SandboxOwnership(AgenticModel):
    controller_id: AgenticId
    run_id: str
    trial_id: str

    def labels(self) -> dict[str, str]:
        return {
            "moe-tools-controller": self.controller_id,
            "moe-tools-run": self.run_id,
            "moe-tools-trial": self.trial_id,
        }


class SandboxHandle(AgenticModel):
    id: Annotated[str, Field(min_length=1, max_length=500)]
    provider_id: AgenticId
    state: SandboxState = SandboxState.READY
    image_ref: str
    image_digest: str | None = None


class SandboxFile(AgenticModel):
    path: Annotated[str, Field(min_length=1, max_length=500)]
    content: Annotated[str, Field(max_length=500_000)]
    executable: bool = False

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("sandbox file path must be safe and relative")
        if any(part in {"", "."} for part in path.parts):
            raise ValueError("sandbox file path contains an empty component")
        return path.as_posix()


class CommandResult(AgenticModel):
    command: Annotated[str, Field(min_length=1, max_length=20_000)]
    exit_code: int | None = None
    stdout: Annotated[str, Field(max_length=200_000)] = ""
    stderr: Annotated[str, Field(max_length=200_000)] = ""
    duration_ms: Annotated[float, Field(ge=0)] = 0
    timed_out: bool = False
    truncated: bool = False


class AgentDescriptor(AgenticModel):
    id: AgenticId
    name: Annotated[str, Field(min_length=1, max_length=120)]
    label: Annotated[str, Field(min_length=1, max_length=120)]
    revision: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=1000)] = ""
    available: bool = True
    is_default: bool = False
    tool_names: list[str] = Field(default_factory=lambda: ["bash"])
    default_budgets: AgentBudgets = Field(default_factory=lambda: AgentBudgets())
    system_prompt_hash: Annotated[str, Field(min_length=64, max_length=64)]
    tool_schema_hash: Annotated[str, Field(min_length=64, max_length=64)]


class AgentTask(AgenticModel):
    id: AgenticId
    pack_id: AgenticId
    name: Annotated[str, Field(min_length=1, max_length=120)]
    title: Annotated[str, Field(min_length=1, max_length=120)]
    instruction: Annotated[str, Field(min_length=1, max_length=20_000)]
    category: Annotated[str, Field(min_length=1, max_length=120)] = "python"
    language: Annotated[str, Field(min_length=1, max_length=80)] | None = "python"
    tags: list[str] = Field(default_factory=list)
    success_criteria: Annotated[list[str], Field(min_length=1, max_length=50)]
    image_ref: Annotated[str, Field(min_length=1, max_length=500)]
    image_digest: Annotated[str, Field(min_length=71, max_length=71)]
    working_directory: str = "/workspace/task"
    network_policy: NetworkPolicy = NetworkPolicy.NONE
    allowed_hosts: list[str] = Field(default_factory=list)
    cpu: Annotated[int, Field(ge=1, le=32)] = 1
    memory_mb: Annotated[int, Field(ge=256, le=131_072)] = 1024
    disk_mb: Annotated[int, Field(ge=256, le=1_048_576)] = 2048
    files: Annotated[list[SandboxFile], Field(min_length=1, max_length=100)]
    submission_file_paths: Annotated[list[str], Field(min_length=1, max_length=100)]
    verifier_file_paths: Annotated[list[str], Field(min_length=1, max_length=100)]
    oracle_file_paths: Annotated[list[str], Field(max_length=100)] = Field(
        default_factory=list
    )
    verifier_command: Annotated[str, Field(min_length=1, max_length=20_000)]
    oracle_commands: Annotated[list[str], Field(min_length=1, max_length=32)]
    timeout_seconds: Annotated[float, Field(ge=1, le=7200)] = 300

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_digest) is None:
            raise ValueError("agent task image_digest must pin a sha256 digest")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("agent task file paths must be unique")
        path_groups = {
            "submission": self.submission_file_paths,
            "verifier": self.verifier_file_paths,
            "oracle": self.oracle_file_paths,
        }
        for label, group in path_groups.items():
            if len(group) != len(set(group)):
                raise ValueError(f"{label} file paths must be unique")
            for path in group:
                SandboxFile(path=path, content="")
        protected_paths = set(self.verifier_file_paths) | set(self.oracle_file_paths)
        overlap = set(self.submission_file_paths) & protected_paths
        if overlap:
            raise ValueError(
                "submission paths cannot overlap verifier or oracle paths: "
                f"{sorted(overlap)}"
            )
        verifier_overlap = set(self.verifier_file_paths) & set(self.oracle_file_paths)
        if verifier_overlap:
            raise ValueError(
                "verifier and oracle paths must be disjoint: "
                f"{sorted(verifier_overlap)}"
            )
        unknown_protected_paths = protected_paths - set(paths)
        if unknown_protected_paths:
            raise ValueError(
                "verifier and oracle paths must reference task files: "
                f"{sorted(unknown_protected_paths)}"
            )
        canonical_clean_room_paths = (
            set(paths) - protected_paths - set(self.submission_file_paths)
        )
        materialized_count = (
            len(canonical_clean_room_paths)
            + len(self.verifier_file_paths)
            + len(self.submission_file_paths)
        )
        if materialized_count > 100:
            raise ValueError("clean-room materialization supports at most 100 files")
        if any(not command.strip() for command in self.oracle_commands):
            raise ValueError("oracle commands cannot be blank")
        SandboxSpec(
            image_ref=self.image_ref,
            image_digest=self.image_digest,
            working_directory=self.working_directory,
            network_policy=self.network_policy,
            allowed_hosts=self.allowed_hosts,
            cpu=self.cpu,
            memory_mb=self.memory_mb,
            disk_mb=self.disk_mb,
        )
        return self

    def sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(
            image_ref=self.image_ref,
            image_digest=self.image_digest,
            working_directory=self.working_directory,
            network_policy=self.network_policy,
            allowed_hosts=self.allowed_hosts,
            cpu=self.cpu,
            memory_mb=self.memory_mb,
            disk_mb=self.disk_mb,
        )

    def verifier_fingerprint(self) -> str:
        protected = {
            file.path: hashlib.sha256(file.content.encode()).hexdigest()
            for file in self.files
            if file.path in set(self.verifier_file_paths)
        }
        canonical = json.dumps(
            {
                "command_hash": hashlib.sha256(
                    self.verifier_command.encode()
                ).hexdigest(),
                "protected_files": protected,
                "submission_file_paths": self.submission_file_paths,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(canonical).hexdigest()


class AgentTaskInfo(AgenticModel):
    id: AgenticId
    title: str
    instruction: str
    language: str | None = None
    tags: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    verifier_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    hidden_verifier_file_count: Annotated[int, Field(ge=0)] = 0
    timeout_seconds: Annotated[float, Field(ge=1, le=7200)]


class AgentTaskPackInfo(AgenticModel):
    id: AgenticId
    name: Annotated[str, Field(min_length=1, max_length=120)]
    revision: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=2000)] = ""
    content_hash: Annotated[str, Field(min_length=64, max_length=64)]
    fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    task_count: Annotated[int, Field(ge=1)]
    source: str = "bundled"
    ready: bool = True
    oracle_passed: bool = True
    noop_failed: bool = True
    tags: list[str] = Field(default_factory=list)


class AgentTaskPack(AgenticModel):
    id: AgenticId
    name: Annotated[str, Field(min_length=1, max_length=120)]
    revision: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=2000)] = ""
    content_hash: Annotated[str, Field(min_length=64, max_length=64)]
    fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    tasks: Annotated[list[AgentTask], Field(min_length=1)]
    source: str = "bundled"
    ready: bool = True
    oracle_passed: bool = True
    noop_failed: bool = True
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("agent task IDs must be unique within a pack")
        if any(task.pack_id != self.id for task in self.tasks):
            raise ValueError("agent task pack_id must match its containing pack")
        if self.content_hash != self.fingerprint:
            raise ValueError("task pack fingerprint must match content_hash")
        return self

    def info(self) -> AgentTaskPackInfo:
        return AgentTaskPackInfo(
            id=self.id,
            name=self.name,
            revision=self.revision,
            description=self.description,
            content_hash=self.content_hash,
            fingerprint=self.fingerprint,
            task_count=len(self.tasks),
            source=self.source,
            ready=self.ready,
            oracle_passed=self.oracle_passed,
            noop_failed=self.noop_failed,
            tags=self.tags,
        )


class AgentBudgets(AgenticModel):
    max_turns: Annotated[int, Field(ge=1, le=64)] = 8
    max_commands: Annotated[int, Field(ge=1, le=128)] = 8
    max_tokens: Annotated[int, Field(ge=1, le=262_144)] = 32_768
    timeout_seconds: Annotated[float, Field(ge=1, le=7200)] = 300


def default_agent_evaluation_contract() -> EvaluationContract:
    return EvaluationContract(
        name="Trusted task-pack verifier",
        description=(
            "Success is determined by the task pack's protected verifier running "
            "inside the isolated sandbox."
        ),
        criteria=[
            EvaluationCriterion(
                id="trusted_verifier",
                label="Trusted task-pack verifier",
                description="The protected task verifier exits successfully.",
                kind=DeterministicScorerKind.BENCHMARK_DEFAULT,
            )
        ],
    )


class AgentEvaluationContractView(AgenticModel):
    version: Literal[1] = 1
    name: str
    description: str = ""
    criteria: list[EvaluationCriterion] = Field(default_factory=list)
    aggregation: EvaluationAggregation
    pass_threshold: float
    judge: LLMJudgeConfig | None = None
    judge_weight: float = 0
    judge_can_override_deterministic_failure: bool = False
    fingerprint: Annotated[str, Field(min_length=64, max_length=64)]


class AgentDefinition(AgenticModel):
    id: AgenticId
    label: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=1000)] = ""
    revision: Annotated[str, Field(min_length=1, max_length=120)]
    available: bool
    is_default: bool
    tool_names: list[str]
    default_budgets: AgentBudgets


def agent_definition(agent: AgentDescriptor) -> AgentDefinition:
    return AgentDefinition.model_validate(
        agent.model_dump(
            include={
                "id",
                "label",
                "description",
                "revision",
                "available",
                "is_default",
                "tool_names",
                "default_budgets",
            }
        )
    )


class CreateAgentRunRequest(AgenticModel):
    task_pack_id: AgenticId
    task_ids: Annotated[list[AgenticId], Field(min_length=1, max_length=100)]
    agent_id: AgenticId = "bash-json-v1"
    sandbox_provider_id: AgenticId = "fake"
    model_session_id: Annotated[str, Field(min_length=1, max_length=200)]
    attempts: Annotated[int, Field(ge=1, le=5)] = 1
    seed: int = 0
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    budgets: AgentBudgets = Field(default_factory=AgentBudgets)
    reasoning_mode: ReasoningMode = ReasoningMode.COMPACT
    evaluation_contract: EvaluationContract | None = None
    execution_policy: ExecutionPolicy | None = None

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("agent task IDs cannot contain duplicates")
        return value


class VerifierResult(AgenticModel):
    command: Annotated[str, Field(min_length=1, max_length=20_000)]
    passed: bool
    exit_code: int | None = None
    stdout: Annotated[str, Field(max_length=200_000)] = ""
    stderr: Annotated[str, Field(max_length=200_000)] = ""
    duration_ms: Annotated[float, Field(ge=0)] = 0
    timed_out: bool = False
    truncated: bool = False


class PatchStats(AgenticModel):
    files_changed: Annotated[int, Field(ge=0)] = 0
    insertions: Annotated[int, Field(ge=0)] = 0
    deletions: Annotated[int, Field(ge=0)] = 0
    bytes: Annotated[int, Field(ge=0)] = 0


class TrialArtifact(AgenticModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    relative_path: Annotated[str, Field(min_length=1, max_length=1000)]
    media_type: Annotated[str, Field(min_length=1, max_length=200)]
    sha256: Annotated[str, Field(min_length=64, max_length=64)]
    size_bytes: Annotated[int, Field(ge=0)]


class TrialArtifactManifest(AgenticModel):
    trial_id: str
    artifacts: list[TrialArtifact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class RoutingArtifactInfo(AgenticModel):
    inference_id: str
    relative_path: Annotated[str, Field(min_length=1, max_length=1000)]
    sha256: Annotated[str, Field(min_length=64, max_length=64)]
    token_count: Annotated[int, Field(ge=0)]
    total_routed_slots: Annotated[int, Field(ge=0)]


class TrialRoutingSummary(AgenticModel):
    trial_id: str
    run_id: str
    layer_ids: list[int]
    selection_counts: list[list[int]]
    routing_mass: list[list[float]]
    total_routed_slots: Annotated[int, Field(ge=0)]
    inference_count: Annotated[int, Field(ge=0)]
    inference_calls: Annotated[int, Field(ge=0)]
    captured_inference_calls: Annotated[int, Field(ge=0)]
    served_tokens: Annotated[int, Field(ge=0)]
    model_id: str
    model_session_id: str
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    artifacts: list[RoutingArtifactInfo] = Field(default_factory=list)


class InferenceCall(AgenticModel):
    id: str
    trial_id: str
    trajectory_step_id: str
    model_session_id: str
    request_hash: Annotated[str, Field(min_length=64, max_length=64)]
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    latency_ms: Annotated[float, Field(ge=0)] = 0
    time_to_first_token_ms: Annotated[float, Field(ge=0)] | None = None
    generation_time_ms: Annotated[float, Field(ge=0)] | None = None
    queue_time_ms: Annotated[float, Field(ge=0)] | None = None
    mean_inter_token_latency_ms: Annotated[float, Field(ge=0)] | None = None
    tokens_per_second: Annotated[float, Field(ge=0)] | None = None
    estimated_cost_usd: Annotated[float, Field(ge=0)] | None = None
    reasoning_content: Annotated[str, Field(max_length=200_000)] | None = None
    finish_reason: Annotated[str, Field(max_length=120)] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    routing_artifact: RoutingArtifactInfo | None = None
    error: Annotated[str, Field(max_length=1000)] | None = None
    created_at: datetime = Field(default_factory=utc_now)


class SandboxSession(AgenticModel):
    id: str
    trial_id: str
    provider_id: AgenticId
    external_id: str | None = None
    state: SandboxSessionState = SandboxSessionState.PENDING
    ownership: SandboxOwnership
    spec: SandboxSpec
    effective_network_policy: NetworkPolicy | None = None
    cleanup_error: Annotated[str, Field(max_length=1000)] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None


class TrialPerformance(AgenticModel):
    wall_time_ms: Annotated[float, Field(ge=0)] = 0
    queue_time_ms: Annotated[float, Field(ge=0)] = 0
    provisioning_time_ms: Annotated[float, Field(ge=0)] = 0
    model_time_ms: Annotated[float, Field(ge=0)] = 0
    sandbox_time_ms: Annotated[float, Field(ge=0)] = 0
    verifier_time_ms: Annotated[float, Field(ge=0)] = 0
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0
    mean_tps: Annotated[float, Field(ge=0)] | None = None
    p50_tps: Annotated[float, Field(ge=0)] | None = None
    p95_tps: Annotated[float, Field(ge=0)] | None = None
    inference_cost_usd: Annotated[float, Field(ge=0)] | None = None
    judge_cost_usd: Annotated[float, Field(ge=0)] | None = None
    judge_cost_debit_usd: Annotated[float, Field(ge=0)] = 0
    judge_cost_uncertain: bool = False
    estimated_cost_usd: Annotated[float, Field(ge=0)] | None = None


class AgentTrial(AgenticModel):
    id: str
    run_id: str
    task_id: AgenticId
    task_title_snapshot: Annotated[str, Field(min_length=1, max_length=120)] | None = (
        None
    )
    attempt: Annotated[int, Field(ge=1)] = 1
    seed: int = 0
    model_session_id: str
    status: AgentTrialStatus = AgentTrialStatus.QUEUED
    sandbox_session_id: str | None = None
    reward: Annotated[float, Field(ge=0, le=1)] | None = None
    termination_cause: TerminationCause | None = None
    turns: Annotated[int, Field(ge=0)] = 0
    commands: Annotated[int, Field(ge=0)] = 0
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    verifier: VerifierResult | None = None
    patch_stats: PatchStats | None = None
    evaluation: EvaluationResult | None = None
    judge_cost_debit_usd: Annotated[float, Field(ge=0)] = 0
    judge_cost_uncertain: bool = False
    performance: TrialPerformance | None = None
    artifact_manifest: TrialArtifactManifest | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: Annotated[str, Field(max_length=1000)] | None = None


class AgentTrialSummary(AgenticModel):
    id: str
    agent_run_id: str
    task_id: AgenticId
    title: str
    task_provenance_status: Literal["known", "legacy_unknown"]
    attempt: Annotated[int, Field(ge=1)] = 1
    seed: int = 0
    status: AgentTrialStatus
    reward: float | None = None
    turns: Annotated[int, Field(ge=0)] = 0
    commands: Annotated[int, Field(ge=0)] = 0
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    inference_calls: Annotated[int, Field(ge=0)] = 0
    routed_inference_calls: Annotated[int, Field(ge=0)] = 0
    termination_reason: str | None = None
    sandbox_status: str
    evaluation: EvaluationResult | None = None
    performance: TrialPerformance | None = None
    updated_at: datetime


class AgentRun(AgenticModel):
    id: str
    job_id: str
    status: AgentRunStatus = AgentRunStatus.QUEUED
    task_pack_id: AgenticId
    task_pack_name: str
    task_pack_revision: str
    task_pack_content_hash: Annotated[str, Field(min_length=64, max_length=64)]
    task_ids: list[AgenticId]
    model_session_id: str
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    agent_id: AgenticId
    agent_revision: str
    sandbox_provider_id: AgenticId
    attempts: Annotated[int, Field(ge=1, le=5)] = 1
    generation: GenerationConfig
    budgets: AgentBudgets
    reasoning_mode: ReasoningMode = ReasoningMode.COMPACT
    evaluation_contract: EvaluationContract = Field(
        default_factory=default_agent_evaluation_contract
    )
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    contract_provenance_status: Literal["known", "legacy_unknown"] = "legacy_unknown"
    contract_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    completed_trials: Annotated[int, Field(ge=0)] = 0
    total_trials: Annotated[int, Field(ge=1)]
    passed_trials: Annotated[int, Field(ge=0)] = 0
    mean_reward: Annotated[float, Field(ge=0, le=1)] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: Annotated[str, Field(max_length=1000)] | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if self.completed_trials > self.total_trials:
            raise ValueError("completed trials cannot exceed total trials")
        if self.passed_trials > self.completed_trials:
            raise ValueError("passed trials cannot exceed completed trials")
        return self


class AgentRunDetail(AgenticModel):
    run: AgentRun
    trials: list[AgentTrial]


class AgentRunView(AgenticModel):
    id: str
    job_id: str
    task_pack_id: AgenticId
    task_pack_name: str
    task_pack_revision: str
    task_pack_content_hash: Annotated[str, Field(min_length=64, max_length=64)]
    model_session_id: str
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    agent_id: AgenticId
    agent_revision: str
    sandbox_provider_id: AgenticId
    generation: GenerationConfig
    budgets: AgentBudgets
    contract_provenance_status: Literal["known", "legacy_unknown"]
    reasoning_mode: ReasoningMode | None = None
    evaluation_contract: AgentEvaluationContractView | None = None
    evaluation_contract_fingerprint: (
        Annotated[str, Field(min_length=64, max_length=64)] | None
    ) = None
    hidden_evaluation_criteria_count: Annotated[int, Field(ge=0)] | None = None
    execution_policy: ExecutionPolicy | None = None
    compatibility_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    status: AgentRunStatus
    task_ids: list[AgenticId]
    total_trials: Annotated[int, Field(ge=1)]
    completed_trials: Annotated[int, Field(ge=0)] = 0
    passed_trials: Annotated[int, Field(ge=0)] = 0
    mean_reward: Annotated[float, Field(ge=0, le=1)] | None = None
    active_trial_id: str | None = None
    trials: list[AgentTrialSummary] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class InferenceCallSummary(AgenticModel):
    id: str
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0
    latency_ms: Annotated[float, Field(ge=0)] = 0
    ttft_ms: Annotated[float, Field(ge=0)] | None = None
    prefill_ms: Annotated[float, Field(ge=0)] | None = None
    decode_ms: Annotated[float, Field(ge=0)] | None = None
    tokens_per_second: Annotated[float, Field(ge=0)] | None = None
    estimated_cost_usd: Annotated[float, Field(ge=0)] | None = None
    model_id: str | None = None
    finish_reason: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    routing_artifact_id: str | None = None
    routed_layers: Annotated[int, Field(ge=0)] = 0
    total_routed_slots: Annotated[int, Field(ge=0)] = 0


class TrajectoryStepView(AgenticModel):
    id: str
    sequence: Annotated[int, Field(ge=1)]
    timestamp: datetime
    type: TrajectoryStepType
    title: str
    content: str
    tool_name: str | None = None
    command: str | None = None
    exit_code: int | None = None
    duration_ms: Annotated[float, Field(ge=0)] | None = None
    truncated: bool = False
    phase: str | None = None
    turn: Annotated[int, Field(ge=1)] | None = None
    stream: Literal["stdout", "stderr"] | None = None
    reasoning_visibility: Literal["explicit", "none"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    inference: InferenceCallSummary | None = None


class AgentTrajectoryView(AgenticModel):
    trial_id: str
    format: Literal["ATIF"] = "ATIF"
    schema_version: str
    reasoning_mode: ReasoningMode | None = None
    steps: list[TrajectoryStepView]
    updated_at: datetime


class VerifierView(AgenticModel):
    status: VerifierStatus
    reward: Annotated[float, Field(ge=0, le=1)] | None = None
    summary: str
    output: str
    exit_code: int | None = None
    duration_ms: Annotated[float, Field(ge=0)] | None = None


class ArtifactExport(AgenticModel):
    name: str
    media_type: str
    download_url: str


class AgentTrialArtifactsView(AgenticModel):
    trial_id: str
    patch: str | None = None
    patch_sha256: str | None = None
    files_changed: Annotated[int, Field(ge=0)] = 0
    additions: Annotated[int, Field(ge=0)] = 0
    deletions: Annotated[int, Field(ge=0)] = 0
    verifier: VerifierView | None = None
    evaluation: EvaluationResult | None = None
    exports: list[ArtifactExport] = Field(default_factory=list)


class AgentRunPublicDetail(AgenticModel):
    run: AgentRunView
    trials: list[AgentTrialSummary]


class AgentRunExport(AgenticModel):
    schema_version: int = 1
    exported_at: datetime = Field(default_factory=utc_now)
    detail: AgentRunPublicDetail
    trajectories: dict[str, dict[str, Any]] = Field(default_factory=dict)
    routing: dict[str, TrialRoutingSummary] = Field(default_factory=dict)
    manifests: dict[str, TrialArtifactManifest] = Field(default_factory=dict)
