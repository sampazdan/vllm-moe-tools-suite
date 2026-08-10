from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..domain import GenerationConfig

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
    image_ref: Annotated[str, Field(min_length=1, max_length=500)]
    image_digest: str | None = None
    working_directory: str = "/workspace/task"
    network_policy: NetworkPolicy = NetworkPolicy.NONE
    allowed_hosts: list[str] = Field(default_factory=list)
    cpu: Annotated[int, Field(ge=1, le=32)] = 1
    memory_mb: Annotated[int, Field(ge=256, le=131_072)] = 1024
    disk_mb: Annotated[int, Field(ge=256, le=1_048_576)] = 2048
    files: Annotated[list[SandboxFile], Field(min_length=1, max_length=100)]
    verifier_file_paths: Annotated[list[str], Field(min_length=1, max_length=100)]
    verifier_command: Annotated[str, Field(min_length=1, max_length=20_000)]
    oracle_commands: Annotated[list[str], Field(min_length=1, max_length=32)]
    timeout_seconds: Annotated[float, Field(ge=1, le=7200)] = 300

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("agent task file paths must be unique")
        if len(self.verifier_file_paths) != len(set(self.verifier_file_paths)):
            raise ValueError("verifier file paths must be unique")
        unknown_verifier_paths = set(self.verifier_file_paths) - set(paths)
        if unknown_verifier_paths:
            raise ValueError(
                "verifier file paths must reference task files: "
                f"{sorted(unknown_verifier_paths)}"
            )
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


class AgentTaskInfo(AgenticModel):
    id: AgenticId
    title: str
    instruction: str
    language: str | None = None
    tags: list[str] = Field(default_factory=list)
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
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    latency_ms: Annotated[float, Field(ge=0)] = 0
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


class AgentTrial(AgenticModel):
    id: str
    run_id: str
    task_id: AgenticId
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
    model_session_id: str
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    agent_id: AgenticId
    sandbox_provider_id: AgenticId
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
    completion_tokens: Annotated[int, Field(ge=0)] = 0
    latency_ms: Annotated[float, Field(ge=0)] = 0
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
    inference: InferenceCallSummary | None = None


class AgentTrajectoryView(AgenticModel):
    trial_id: str
    format: Literal["ATIF"] = "ATIF"
    schema_version: str
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
    exports: list[ArtifactExport] = Field(default_factory=list)


class AgentRunExport(AgenticModel):
    schema_version: int = 1
    exported_at: datetime = Field(default_factory=utc_now)
    detail: AgentRunDetail
    trajectories: dict[str, dict[str, Any]] = Field(default_factory=dict)
    routing: dict[str, TrialRoutingSummary] = Field(default_factory=dict)
    manifests: dict[str, TrialArtifactManifest] = Field(default_factory=dict)
