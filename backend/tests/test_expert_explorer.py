from __future__ import annotations

import numpy as np
import pytest
from moe_tools_suite.domain import ModelTopology
from moe_tools_suite.expert_explorer import (
    ResolvedRoutingSource,
    RoutingExploreFilters,
    RoutingExploreRequest,
    RoutingFilterCapabilities,
    RoutingSourceKind,
    RoutingSourceReference,
    build_routing_explorer_response,
)

TOPOLOGY = ModelTopology(
    num_layers=2,
    num_experts=3,
    top_k=2,
    routed_layer_ids=[4, 9],
)


def _source(
    source_id: str,
    *,
    counts: list[list[int]],
    mass: list[list[float]],
    weight: float = 1,
    profile_fingerprint: str | None = None,
    topology: ModelTopology = TOPOLOGY,
    capabilities: RoutingFilterCapabilities | None = None,
) -> ResolvedRoutingSource:
    return ResolvedRoutingSource(
        reference=RoutingSourceReference(
            kind=RoutingSourceKind.BENCHMARK_RUN,
            id=source_id,
            weight=weight,
        ),
        label=source_id,
        status="completed",
        model_id="model-a",
        topology=topology,
        profile_id="profile-a" if profile_fingerprint else None,
        profile_fingerprint=profile_fingerprint,
        selection_counts=np.asarray(counts, dtype=np.int64),
        routing_mass=np.asarray(mass, dtype=np.float64),
        total_routed_slots=int(np.asarray(counts).sum()),
        captured_inference_calls=1,
        total_inference_calls=1,
        served_tokens=3,
        capabilities=capabilities or RoutingFilterCapabilities(),
    )


def test_explorer_aggregates_weighted_sources_and_keeps_comparison_separate() -> None:
    request = RoutingExploreRequest(
        sources=[
            {"kind": "benchmark_run", "id": "base-a"},
            {"kind": "benchmark_run", "id": "base-b", "weight": 2},
        ],
        comparison_sources=[{"kind": "benchmark_run", "id": "masked"}],
    )
    base_a = _source(
        "base-a",
        counts=[[1, 0, 2], [0, 1, 2]],
        mass=[[0.2, 0, 0.8], [0, 0.4, 0.6]],
    )
    base_b = _source(
        "base-b",
        counts=[[0, 3, 0], [2, 0, 1]],
        mass=[[0, 0.7, 0], [0.6, 0, 0.4]],
        weight=2,
    )
    masked = _source(
        "masked",
        counts=[[0, 1, 0], [1, 0, 0]],
        mass=[[0, 0.9, 0], [0.8, 0, 0]],
        profile_fingerprint="f" * 64,
    )

    response = build_routing_explorer_response(
        request,
        [base_a, base_b],
        [masked],
    )

    np.testing.assert_allclose(
        response.selection_counts,
        base_a.selection_counts / base_a.selection_counts.sum()
        + 2 * base_b.selection_counts / base_b.selection_counts.sum(),
    )
    np.testing.assert_allclose(
        response.routing_mass,
        base_a.routing_mass / base_a.routing_mass.sum()
        + 2 * base_b.routing_mass / base_b.routing_mass.sum(),
    )
    np.testing.assert_allclose(
        response.comparison_selection_counts,
        masked.selection_counts / masked.selection_counts.sum(),
    )
    assert response.aggregation == "weighted_source_normalized"
    assert np.asarray(response.routing_mass).sum() == pytest.approx(3)
    assert response.profile_fingerprint is None
    assert response.captured_inference_calls == 2
    assert len(response.fingerprint) == 64


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        (
            [
                _source(
                    "base",
                    counts=[[1, 0, 0], [0, 1, 0]],
                    mass=[[1, 0, 0], [0, 1, 0]],
                ),
                _source(
                    "masked",
                    counts=[[1, 0, 0], [0, 1, 0]],
                    mass=[[1, 0, 0], [0, 1, 0]],
                    profile_fingerprint="a" * 64,
                ),
            ],
            "different expert profiles",
        ),
        (
            [
                _source(
                    "base",
                    counts=[[1, 0, 0], [0, 1, 0]],
                    mass=[[1, 0, 0], [0, 1, 0]],
                ),
                _source(
                    "other-model-shape",
                    counts=[[1, 0], [0, 1]],
                    mass=[[1, 0], [0, 1]],
                    topology=ModelTopology(
                        num_layers=2,
                        num_experts=2,
                        top_k=1,
                        routed_layer_ids=[4, 9],
                    ),
                ),
            ],
            "different model topologies",
        ),
    ],
)
def test_explorer_rejects_incompatible_primary_sources(
    sources: list[ResolvedRoutingSource], message: str
) -> None:
    request = RoutingExploreRequest(sources=[source.reference for source in sources])

    with pytest.raises(ValueError, match=message):
        build_routing_explorer_response(request, sources)


def test_explorer_refuses_filters_that_cannot_be_applied_honestly() -> None:
    source = _source(
        "base",
        counts=[[1, 0, 0], [0, 1, 0]],
        mass=[[1, 0, 0], [0, 1, 0]],
        capabilities=RoutingFilterCapabilities(item=True),
    )
    request = RoutingExploreRequest(
        sources=[source.reference],
        filters=RoutingExploreFilters(step_types=["command"]),
    )

    with pytest.raises(ValueError, match="step type filtering is unavailable"):
        build_routing_explorer_response(request, [source])
