import numpy as np
import pytest
from moe_tools_suite.domain import ModelTopology
from moe_tools_suite.telemetry import (
    aggregate_routing,
    decode_routing_payloads,
    encode_npy,
)


@pytest.fixture
def topology() -> ModelTopology:
    return ModelTopology(
        num_layers=2,
        num_experts=4,
        top_k=2,
        routed_layer_ids=[3, 7],
    )


def test_decodes_fork_wire_format_and_aggregates_pairs(
    topology: ModelTopology,
) -> None:
    ids = np.array([[[0, 1], [2, 3]], [[1, 2], [2, 0]]], dtype=np.uint8)
    weights = np.array(
        [[[0.75, 0.25], [0.6, 0.4]], [[0.8, 0.2], [0.55, 0.45]]],
        dtype=np.float32,
    )

    decoded = decode_routing_payloads(
        encode_npy(ids), encode_npy(weights), topology
    )
    aggregate = aggregate_routing(decoded, topology)

    assert aggregate.selection_counts.tolist() == [[1, 2, 1, 0], [1, 0, 2, 1]]
    np.testing.assert_allclose(
        aggregate.routing_mass,
        [[0.75, 1.05, 0.2, 0.0], [0.45, 0.0, 1.15, 0.4]],
    )
    assert aggregate.total_routed_slots == 8


def test_rejects_unpaired_telemetry_shapes(topology: ModelTopology) -> None:
    ids = np.zeros((3, 2, 2), dtype=np.uint8)
    weights = np.zeros((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="identical shapes"):
        decode_routing_payloads(encode_npy(ids), encode_npy(weights), topology)


def test_rejects_expert_ids_outside_topology(topology: ModelTopology) -> None:
    ids = np.full((1, 2, 2), 4, dtype=np.uint8)
    weights = np.ones((1, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="outside"):
        decode_routing_payloads(encode_npy(ids), encode_npy(weights), topology)
