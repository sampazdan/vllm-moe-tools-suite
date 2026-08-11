from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain import AgentTask, AgentTaskPack, NetworkPolicy, SandboxFile
from .external import (
    ExternalTaskPackManifest,
    ExternalTaskPackRegistry,
    load_external_task_pack_registry,
)

_FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ExternalPreparationError(RuntimeError):
    """An external task pack could not be prepared safely."""


class ImportModel(BaseModel):
    """Strict immutable base for external import artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PreparedExternalTask(ImportModel):
    """Separated public and protected assets produced by an upstream adapter."""

    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
    title: Annotated[str, Field(min_length=1, max_length=120)]
    instruction: Annotated[str, Field(min_length=1, max_length=20_000)]
    category: Annotated[str, Field(min_length=1, max_length=120)]
    language: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    tags: list[str] = Field(default_factory=list)
    success_criteria: Annotated[list[str], Field(min_length=1, max_length=50)]
    image_ref: Annotated[str, Field(min_length=1, max_length=500)]
    image_digest: Annotated[str, Field(min_length=71, max_length=71)]
    working_directory: str = "/workspace/task"
    network_policy: NetworkPolicy = NetworkPolicy.NONE
    allowed_hosts: list[str] = Field(default_factory=list)
    cpu: Annotated[int, Field(ge=1, le=32)] = 1
    memory_mb: Annotated[int, Field(ge=256, le=131_072)] = 2048
    disk_mb: Annotated[int, Field(ge=256, le=1_048_576)] = 4096
    timeout_seconds: Annotated[float, Field(ge=1, le=7200)] = 900
    public_files: Annotated[list[SandboxFile], Field(min_length=1, max_length=100)]
    submission_file_paths: Annotated[list[str], Field(min_length=1, max_length=100)]
    protected_files: Annotated[list[SandboxFile], Field(min_length=1, max_length=100)]
    oracle_files: Annotated[list[SandboxFile], Field(max_length=100)] = Field(
        default_factory=list
    )
    verifier_command: Annotated[str, Field(min_length=1, max_length=20_000)]
    oracle_commands: Annotated[list[str], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def validate_asset_boundary(self) -> Self:
        if _IMAGE_DIGEST.fullmatch(self.image_digest) is None:
            raise ValueError("external task image_digest must pin a sha256 digest")
        public_paths = [file.path for file in self.public_files]
        protected_paths = [file.path for file in self.protected_files]
        oracle_paths = [file.path for file in self.oracle_files]
        if len(public_paths) != len(set(public_paths)):
            raise ValueError("external public file paths must be unique")
        if len(protected_paths) != len(set(protected_paths)):
            raise ValueError("external protected file paths must be unique")
        if len(oracle_paths) != len(set(oracle_paths)):
            raise ValueError("external oracle file paths must be unique")
        all_groups = (set(public_paths), set(protected_paths), set(oracle_paths))
        overlap = (all_groups[0] & all_groups[1]) | (all_groups[0] & all_groups[2])
        overlap |= all_groups[1] & all_groups[2]
        if overlap:
            raise ValueError(
                "external public, verifier, and oracle assets overlap: "
                f"{sorted(overlap)}"
            )
        if len(self.submission_file_paths) != len(set(self.submission_file_paths)):
            raise ValueError("external submission file paths must be unique")
        for path in self.submission_file_paths:
            SandboxFile(path=path, content="")
        submission_overlap = set(self.submission_file_paths) & (
            set(protected_paths) | set(oracle_paths)
        )
        if submission_overlap:
            raise ValueError(
                "external submission paths overlap protected assets: "
                f"{sorted(submission_overlap)}"
            )
        if any(not command.strip() for command in self.oracle_commands):
            raise ValueError("external oracle commands cannot be blank")
        return self

    def to_agent_task(self, pack_id: str) -> AgentTask:
        protected_paths = [file.path for file in self.protected_files]
        oracle_paths = [file.path for file in self.oracle_files]
        return AgentTask(
            id=self.id,
            pack_id=pack_id,
            name=self.title,
            title=self.title,
            instruction=self.instruction,
            category=self.category,
            language=self.language,
            tags=self.tags,
            success_criteria=self.success_criteria,
            image_ref=self.image_ref,
            image_digest=self.image_digest,
            working_directory=self.working_directory,
            network_policy=self.network_policy,
            allowed_hosts=self.allowed_hosts,
            cpu=self.cpu,
            memory_mb=self.memory_mb,
            disk_mb=self.disk_mb,
            timeout_seconds=self.timeout_seconds,
            files=[*self.public_files, *self.protected_files, *self.oracle_files],
            submission_file_paths=self.submission_file_paths,
            verifier_file_paths=protected_paths,
            oracle_file_paths=oracle_paths,
            verifier_command=self.verifier_command,
            oracle_commands=self.oracle_commands,
        )


class ExternalImportAttestation(ImportModel):
    """Reproducibility and eligibility evidence from isolated preparation."""

    source_revision: Annotated[str, Field(min_length=40, max_length=40)]
    harness_revision: Annotated[str, Field(min_length=40, max_length=40)]
    source_tree_sha256: Annotated[str, Field(min_length=64, max_length=64)]
    prepared_task_ids: list[str]
    oracle_passed_task_ids: list[str]
    noop_failed_task_ids: list[str]
    license_reviewed: bool = False

    @model_validator(mode="after")
    def validate_attestation(self) -> Self:
        if _FULL_GIT_SHA.fullmatch(self.source_revision) is None:
            raise ValueError("attested source revision must be a full git SHA")
        if _FULL_GIT_SHA.fullmatch(self.harness_revision) is None:
            raise ValueError("attested harness revision must be a full git SHA")
        if _SHA256.fullmatch(self.source_tree_sha256) is None:
            raise ValueError("source_tree_sha256 must be a SHA-256")
        for label, task_ids in (
            ("prepared", self.prepared_task_ids),
            ("oracle-passed", self.oracle_passed_task_ids),
            ("no-op-failed", self.noop_failed_task_ids),
        ):
            if len(task_ids) != len(set(task_ids)):
                raise ValueError(f"{label} external task IDs must be unique")
        return self


class PreparedExternalPack(ImportModel):
    """A staged external pack that is ineligible until every check is attested."""

    manifest_id: str
    tasks: Annotated[list[PreparedExternalTask], Field(min_length=1)]
    attestation: ExternalImportAttestation

    @model_validator(mode="after")
    def validate_task_ids(self) -> Self:
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("prepared external task IDs must be unique")
        return self

    def to_agent_task_pack(
        self,
        registry: ExternalTaskPackRegistry | None = None,
    ) -> AgentTaskPack:
        registry = registry or load_external_task_pack_registry()
        manifest = registry.get(self.manifest_id)
        expected = list(manifest.curated_task_ids)
        actual = [task.id for task in self.tasks]
        if not expected:
            raise ExternalPreparationError(
                "external manifest has no approved application cohort"
            )
        if set(actual) != set(expected) or len(actual) != len(expected):
            raise ExternalPreparationError(
                "prepared tasks do not exactly match the approved curated cohort"
            )
        if self.attestation.source_revision != manifest.source_revision:
            raise ExternalPreparationError("external source revision is not pinned")
        if self.attestation.harness_revision != registry.harness.revision:
            raise ExternalPreparationError("external harness revision is not pinned")
        expected_set = set(expected)
        checks = {
            "prepared": set(self.attestation.prepared_task_ids),
            "oracle passed": set(self.attestation.oracle_passed_task_ids),
            "no-op failed": set(self.attestation.noop_failed_task_ids),
        }
        incomplete = [
            label for label, task_ids in checks.items() if task_ids != expected_set
        ]
        if incomplete:
            raise ExternalPreparationError(
                "external cohort eligibility checks are incomplete: "
                + ", ".join(incomplete)
            )
        if not self.attestation.license_reviewed:
            raise ExternalPreparationError(
                "external cohort license review has not been attested"
            )
        agent_tasks = [task.to_agent_task(manifest.id) for task in self.tasks]
        pack_payload = {
            "id": manifest.id,
            "name": manifest.name,
            "revision": (
                f"{manifest.source_revision[:12]}+harbor."
                f"{registry.harness.revision[:12]}"
            ),
            "description": manifest.curated_selection,
            "tasks": [task.model_dump(mode="json") for task in agent_tasks],
            "source": (
                f"{manifest.source_url}@{manifest.source_revision} via "
                f"{registry.harness.source_url}@{registry.harness.revision}"
            ),
            "ready": True,
            "oracle_passed": True,
            "noop_failed": True,
            "tags": ["external", "agentic", manifest.family],
            "source_tree_sha256": self.attestation.source_tree_sha256,
        }
        fingerprint = _canonical_hash(pack_payload)
        pack_payload.pop("source_tree_sha256")
        pack_payload["content_hash"] = fingerprint
        pack_payload["fingerprint"] = fingerprint
        return AgentTaskPack.model_validate(pack_payload)


class ExternalTaskPackImporter(Protocol):
    """Adapter boundary for lazy, isolated Harbor-compatible preparation."""

    @property
    def manifest(self) -> ExternalTaskPackManifest: ...

    @property
    def requirements(self) -> Sequence[str]: ...

    def prepare(
        self,
        cache_dir: Path,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Awaitable[PreparedExternalPack]: ...


class ManifestOnlyExternalImporter:
    """Fail-closed placeholder until a source-specific converter is installed."""

    def __init__(
        self,
        manifest: ExternalTaskPackManifest,
        *,
        requirements: Sequence[str],
    ) -> None:
        self._manifest = manifest
        self._requirements = tuple(requirements)

    @property
    def manifest(self) -> ExternalTaskPackManifest:
        return self._manifest.model_copy(deep=True)

    @property
    def requirements(self) -> Sequence[str]:
        return self._requirements

    async def prepare(
        self,
        cache_dir: Path,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> PreparedExternalPack:
        del cache_dir, on_progress, should_cancel
        raise ExternalPreparationError(
            f"{self.manifest.name} is provenance-only: " + "; ".join(self.requirements)
        )


def external_task_pack_importers() -> tuple[ManifestOnlyExternalImporter, ...]:
    """Return only cohorts with immutable upstream task assets."""

    registry = load_external_task_pack_registry()
    requirements = {
        "terminal-bench-2.1-engineering-15": (
            "implement the native Harbor task.toml/environment converter",
            "pin every effective container image digest",
            "run each oracle and no-op verifier in isolated Daytona sandboxes",
        ),
        "aider-polyglot-balanced-12": (
            "implement the pinned Harbor aider_polyglot asset converter",
            "review task component licenses before local redistribution",
            "run each oracle and no-op verifier in isolated Daytona sandboxes",
        ),
    }
    return tuple(
        ManifestOnlyExternalImporter(registry.get(pack_id), requirements=items)
        for pack_id, items in requirements.items()
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
