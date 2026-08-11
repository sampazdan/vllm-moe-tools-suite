from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from importlib import resources
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

_FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")


class PreparationState(StrEnum):
    """Readiness stages for an external task pack."""

    REFERENCE_ONLY = "reference_only"
    SOURCE_PINNED = "source_pinned"
    ASSETS_PREPARED = "assets_prepared"
    VALIDATED = "validated"


class ExternalManifestModel(BaseModel):
    """Strict base contract for external harness provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HarnessPin(ExternalManifestModel):
    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
    name: Annotated[str, Field(min_length=1, max_length=120)]
    source_url: HttpUrl
    revision: Annotated[str, Field(min_length=40, max_length=40)]
    license: Annotated[str, Field(min_length=1, max_length=200)]
    trajectory_format: Annotated[str, Field(min_length=1, max_length=80)]

    @model_validator(mode="after")
    def require_immutable_revision(self) -> Self:
        if _FULL_GIT_SHA.fullmatch(self.revision) is None:
            raise ValueError("harness revision must be a full lowercase git SHA")
        return self


class ExternalTaskPackManifest(ExternalManifestModel):
    """Inspectable metadata for a lazily prepared external agentic task pack."""

    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
    name: Annotated[str, Field(min_length=1, max_length=160)]
    family: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
    upstream_version: Annotated[str, Field(min_length=1, max_length=80)]
    source_url: HttpUrl
    source_revision: Annotated[str, Field(min_length=40, max_length=40)]
    license: Annotated[str, Field(min_length=1, max_length=300)]
    upstream_task_count: Annotated[int, Field(ge=1, le=100_000)]
    adapter_path: Annotated[str, Field(min_length=1, max_length=300)]
    distribution: Annotated[str, Field(min_length=1, max_length=120)]
    preparation_state: PreparationState
    curated_selection: Annotated[str, Field(min_length=1, max_length=1000)]
    curated_task_ids: list[str] = Field(max_length=1000)
    public_problem_fields: Annotated[list[str], Field(min_length=1, max_length=30)]
    success_criteria: Annotated[str, Field(min_length=1, max_length=1000)]
    verifier_asset_policy: Annotated[str, Field(min_length=1, max_length=1000)]
    assets_prepared: bool
    oracle_passed: bool
    noop_failed: bool
    limitations: Annotated[list[str], Field(min_length=1, max_length=20)]

    @model_validator(mode="after")
    def validate_eligibility(self) -> Self:
        if _FULL_GIT_SHA.fullmatch(self.source_revision) is None:
            raise ValueError("task-pack source revision must be a full git SHA")
        if len(self.curated_task_ids) != len(set(self.curated_task_ids)):
            raise ValueError("curated task IDs must be unique")
        if len(self.curated_task_ids) > self.upstream_task_count:
            raise ValueError("curated task count exceeds the upstream task count")
        if len(self.public_problem_fields) != len(set(self.public_problem_fields)):
            raise ValueError("public problem fields must be unique")
        if self.preparation_state is PreparationState.ASSETS_PREPARED and (
            not self.assets_prepared
        ):
            raise ValueError("assets_prepared state requires prepared assets")
        if self.preparation_state is PreparationState.VALIDATED and not (
            self.assets_prepared and self.oracle_passed and self.noop_failed
        ):
            raise ValueError(
                "validated task packs require assets, a passing oracle, "
                "and a failing no-op"
            )
        return self

    @property
    def launchable(self) -> bool:
        return self.preparation_state is PreparationState.VALIDATED


class ExternalTaskPackRegistry(ExternalManifestModel):
    """Pinned external harness plus task-pack preparation candidates."""

    schema_version: Annotated[int, Field(ge=1, le=1)]
    harness: HarnessPin
    task_packs: Annotated[list[ExternalTaskPackManifest], Field(min_length=1)]
    content_hash: Annotated[str, Field(min_length=64, max_length=64)]

    @model_validator(mode="after")
    def require_unique_pack_ids(self) -> Self:
        ids = [pack.id for pack in self.task_packs]
        if len(ids) != len(set(ids)):
            raise ValueError("external task-pack IDs must be unique")
        return self

    def get(self, pack_id: str) -> ExternalTaskPackManifest:
        pack = next(
            (candidate for candidate in self.task_packs if candidate.id == pack_id),
            None,
        )
        if pack is None:
            raise KeyError(f"unknown external task pack {pack_id!r}")
        return pack.model_copy(deep=True)


def load_external_task_pack_registry() -> ExternalTaskPackRegistry:
    """Load and fingerprint the bundled provenance-only task-pack manifest."""

    resource = resources.files(__package__).joinpath("external_manifests.json")
    try:
        raw = resource.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid external task-pack manifest resource") from error
    if not isinstance(payload, dict):
        raise ValueError("external task-pack manifest must contain a JSON object")
    payload["content_hash"] = hashlib.sha256(raw).hexdigest()
    return ExternalTaskPackRegistry.model_validate(payload)
