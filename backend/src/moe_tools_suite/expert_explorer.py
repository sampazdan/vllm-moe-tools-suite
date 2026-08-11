from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .domain import ModelTopology


class RoutingSourceKind(StrEnum):
    BENCHMARK_RUN = "benchmark_run"
    AGENT_TRIAL = "agent_trial"


class RoutingSourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: RoutingSourceKind
    id: Annotated[str, Field(min_length=1, max_length=200)]
    weight: Annotated[float, Field(gt=0, le=100)] = 1.0


class RoutingExploreFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_ids: Annotated[list[str], Field(max_length=1000)] = Field(default_factory=list)
    trial_ids: Annotated[list[str], Field(max_length=1000)] = Field(
        default_factory=list
    )
    step_types: Annotated[list[str], Field(max_length=32)] = Field(default_factory=list)
    passed: bool | None = None

    @field_validator("item_ids", "trial_ids", "step_types")
    @classmethod
    def reject_duplicates(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("routing filters cannot contain duplicates")
        return values


class RoutingExploreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: Annotated[
        list[RoutingSourceReference], Field(min_length=1, max_length=100)
    ]
    comparison_sources: Annotated[
        list[RoutingSourceReference], Field(max_length=100)
    ] = Field(default_factory=list)
    metric: Literal["routing_mass", "selection_count"] = "routing_mass"
    filters: RoutingExploreFilters = Field(default_factory=RoutingExploreFilters)

    @field_validator("sources", "comparison_sources")
    @classmethod
    def reject_duplicate_sources(
        cls, values: list[RoutingSourceReference]
    ) -> list[RoutingSourceReference]:
        identities = [(value.kind, value.id) for value in values]
        if len(identities) != len(set(identities)):
            raise ValueError("routing sources cannot contain duplicates")
        return values


class RoutingExploreSource(BaseModel):
    kind: RoutingSourceKind
    id: str
    label: str
    status: str | None = None
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    weight: float = 1.0


class RoutingFilterCapabilities(BaseModel):
    item: bool = False
    trial: bool = False
    step_type: bool = False
    outcome: bool = False


class RoutingExploreResponse(BaseModel):
    fingerprint: str
    aggregation: Literal["weighted_source_normalized"] = "weighted_source_normalized"
    model_id: str
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    layer_ids: list[int]
    num_experts: int
    top_k: int
    selection_counts: list[list[float]]
    routing_mass: list[list[float]]
    comparison_selection_counts: list[list[float]] | None = None
    comparison_routing_mass: list[list[float]] | None = None
    total_routed_slots: Annotated[int, Field(ge=0)]
    captured_inference_calls: Annotated[int, Field(ge=0)] | None = None
    total_inference_calls: Annotated[int, Field(ge=0)] | None = None
    served_tokens: Annotated[int, Field(ge=0)] | None = None
    sources: list[RoutingExploreSource]
    comparison_sources: list[RoutingExploreSource] = Field(default_factory=list)
    filter_capabilities: RoutingFilterCapabilities


@dataclass(frozen=True)
class ResolvedRoutingSource:
    reference: RoutingSourceReference
    label: str
    status: str | None
    model_id: str
    topology: ModelTopology
    profile_id: str | None
    profile_fingerprint: str | None
    selection_counts: np.ndarray
    routing_mass: np.ndarray
    total_routed_slots: int
    captured_inference_calls: int | None = None
    total_inference_calls: int | None = None
    served_tokens: int | None = None
    capabilities: RoutingFilterCapabilities = field(
        default_factory=RoutingFilterCapabilities
    )

    def public(self) -> RoutingExploreSource:
        return RoutingExploreSource(
            kind=self.reference.kind,
            id=self.reference.id,
            label=self.label,
            status=self.status,
            profile_id=self.profile_id,
            profile_fingerprint=self.profile_fingerprint,
            weight=self.reference.weight,
        )


@dataclass(frozen=True)
class _AggregatedGroup:
    counts: np.ndarray
    mass: np.ndarray
    total_routed_slots: int
    captured_inference_calls: int | None
    total_inference_calls: int | None
    served_tokens: int | None
    profile_id: str | None
    profile_fingerprint: str | None


def build_routing_explorer_response(
    request: RoutingExploreRequest,
    sources: list[ResolvedRoutingSource],
    comparison_sources: list[ResolvedRoutingSource] | None = None,
) -> RoutingExploreResponse:
    """Strictly aggregate compatible routing sources for the explorer UI."""

    if not sources:
        raise ValueError("routing filters excluded every primary source")
    comparisons = comparison_sources or []
    _validate_requested_filters(request.filters, [*sources, *comparisons])
    topology = sources[0].topology
    model_id = sources[0].model_id
    _validate_group_compatibility(sources, topology, model_id)
    primary = _aggregate_group(sources, topology)
    comparison = None
    if comparisons:
        _validate_group_compatibility(comparisons, topology, model_id)
        comparison = _aggregate_group(comparisons, topology)

    all_sources = [*sources, *comparisons]
    capabilities = RoutingFilterCapabilities(
        item=all(source.capabilities.item for source in all_sources),
        trial=all(source.capabilities.trial for source in all_sources),
        step_type=all(source.capabilities.step_type for source in all_sources),
        outcome=all(source.capabilities.outcome for source in all_sources),
    )
    fingerprint = _explorer_fingerprint(
        request,
        sources,
        comparisons,
        primary,
        comparison,
    )
    return RoutingExploreResponse(
        fingerprint=fingerprint,
        model_id=model_id,
        profile_id=primary.profile_id,
        profile_fingerprint=primary.profile_fingerprint,
        layer_ids=topology.routed_layer_ids,
        num_experts=topology.num_experts,
        top_k=topology.top_k,
        selection_counts=primary.counts.tolist(),
        routing_mass=primary.mass.tolist(),
        comparison_selection_counts=(
            comparison.counts.tolist() if comparison is not None else None
        ),
        comparison_routing_mass=(
            comparison.mass.tolist() if comparison is not None else None
        ),
        total_routed_slots=primary.total_routed_slots,
        captured_inference_calls=primary.captured_inference_calls,
        total_inference_calls=primary.total_inference_calls,
        served_tokens=primary.served_tokens,
        sources=[source.public() for source in sources],
        comparison_sources=[source.public() for source in comparisons],
        filter_capabilities=capabilities,
    )


def _validate_requested_filters(
    filters: RoutingExploreFilters,
    sources: list[ResolvedRoutingSource],
) -> None:
    checks = (
        (bool(filters.item_ids), "item", "item"),
        (bool(filters.trial_ids), "trial", "trial"),
        (bool(filters.step_types), "step_type", "step type"),
        (filters.passed is not None, "outcome", "outcome"),
    )
    for requested, attribute, label in checks:
        if requested and any(
            not getattr(source.capabilities, attribute) for source in sources
        ):
            raise ValueError(
                f"{label} filtering is unavailable for one or more routing sources"
            )


def _validate_group_compatibility(
    sources: list[ResolvedRoutingSource],
    topology: ModelTopology,
    model_id: str,
) -> None:
    expected_shape = (topology.num_layers, topology.num_experts)
    profile_fingerprint = sources[0].profile_fingerprint
    for source in sources:
        if source.model_id != model_id or source.topology != topology:
            raise ValueError("routing sources use different model topologies")
        if source.profile_fingerprint != profile_fingerprint:
            raise ValueError(
                "routing sources use different expert profiles; use comparison_sources "
                "for cross-profile analysis"
            )
        if source.selection_counts.shape != expected_shape:
            raise ValueError("routing selection-count shape does not match topology")
        if source.routing_mass.shape != expected_shape:
            raise ValueError("routing-mass shape does not match topology")
        for values, label in (
            (source.selection_counts, "selection counts"),
            (source.routing_mass, "routing mass"),
        ):
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"routing {label} must be finite and non-negative")


def _aggregate_group(
    sources: list[ResolvedRoutingSource], topology: ModelTopology
) -> _AggregatedGroup:
    shape = (topology.num_layers, topology.num_experts)
    counts = np.zeros(shape, dtype=np.float64)
    mass = np.zeros(shape, dtype=np.float64)
    for source in sources:
        counts += _normalized_engagement(
            source.selection_counts,
            source.reference.weight,
        )
        mass += _normalized_engagement(
            source.routing_mass,
            source.reference.weight,
        )
    profile_ids = {source.profile_id for source in sources}
    profile_fingerprints = {source.profile_fingerprint for source in sources}
    return _AggregatedGroup(
        counts=counts,
        mass=mass,
        total_routed_slots=sum(source.total_routed_slots for source in sources),
        captured_inference_calls=_sum_optional(
            source.captured_inference_calls for source in sources
        ),
        total_inference_calls=_sum_optional(
            source.total_inference_calls for source in sources
        ),
        served_tokens=_sum_optional(source.served_tokens for source in sources),
        profile_id=next(iter(profile_ids)) if len(profile_ids) == 1 else None,
        profile_fingerprint=(
            next(iter(profile_fingerprints)) if len(profile_fingerprints) == 1 else None
        ),
    )


def _normalized_engagement(values: np.ndarray, weight: float) -> np.ndarray:
    normalized = values.astype(np.float64)
    total = float(normalized.sum())
    if total == 0:
        return np.zeros_like(normalized)
    return normalized * (weight / total)


def _sum_optional(values: Iterable[int | None]) -> int | None:
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return sum(value for value in materialized if value is not None)


def _explorer_fingerprint(
    request: RoutingExploreRequest,
    sources: list[ResolvedRoutingSource],
    comparisons: list[ResolvedRoutingSource],
    primary: _AggregatedGroup,
    comparison: _AggregatedGroup | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"weighted_source_normalized:v1")
    digest.update(
        json.dumps(
            request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for source in [*sources, *comparisons]:
        digest.update(source.model_id.encode())
        digest.update((source.profile_fingerprint or "baseline").encode())
    for group in (primary, comparison):
        if group is None:
            continue
        digest.update(np.ascontiguousarray(group.counts).tobytes())
        digest.update(np.ascontiguousarray(group.mass).tobytes())
    return digest.hexdigest()
