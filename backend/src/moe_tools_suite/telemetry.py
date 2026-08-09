from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .domain import ModelTopology

ExpertIds = NDArray[np.unsignedinteger]
ExpertWeights = NDArray[np.floating]


@dataclass(frozen=True)
class DecodedRouting:
    expert_ids: ExpertIds
    expert_weights: ExpertWeights


@dataclass(frozen=True)
class AggregatedRouting:
    selection_counts: NDArray[np.int64]
    routing_mass: NDArray[np.float64]
    total_routed_slots: int


def encode_npy(array: NDArray[np.generic]) -> str:
    """Encode an ndarray using the fork's base64 `.npy` wire format."""

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_routing_payloads(
    routed_experts: str,
    routed_expert_weights: str,
    topology: ModelTopology,
) -> DecodedRouting:
    """Decode and validate paired routed-expert telemetry from vLLM."""

    ids = _decode_npy(routed_experts, "routed_experts")
    weights = _decode_npy(routed_expert_weights, "routed_expert_weights")
    if ids.ndim != 3:
        raise ValueError("routed_experts must have shape [tokens, layers, top_k]")
    if ids.shape != weights.shape:
        raise ValueError("routed expert IDs and weights must have identical shapes")
    if ids.shape[1:] != (topology.num_layers, topology.top_k):
        raise ValueError(
            "routing shape does not match topology: "
            f"got {ids.shape[1:]}, expected "
            f"{(topology.num_layers, topology.top_k)}"
        )
    if not np.issubdtype(ids.dtype, np.unsignedinteger):
        raise ValueError("routed expert IDs must use an unsigned integer dtype")
    if not np.issubdtype(weights.dtype, np.floating):
        raise ValueError("routed expert weights must use a floating dtype")
    if ids.size and int(ids.max()) >= topology.num_experts:
        raise ValueError("routed expert ID is outside the model topology")
    if not np.isfinite(weights).all():
        raise ValueError("routed expert weights must be finite")
    return DecodedRouting(expert_ids=ids, expert_weights=weights)


def aggregate_routing(
    decoded: DecodedRouting,
    topology: ModelTopology,
) -> AggregatedRouting:
    """Aggregate selection count and routing mass by routed layer and expert."""

    counts = np.zeros(
        (topology.num_layers, topology.num_experts), dtype=np.int64
    )
    mass = np.zeros((topology.num_layers, topology.num_experts), dtype=np.float64)
    for layer_index in range(topology.num_layers):
        layer_ids = decoded.expert_ids[:, layer_index, :].reshape(-1)
        layer_weights = decoded.expert_weights[:, layer_index, :].reshape(-1)
        np.add.at(counts[layer_index], layer_ids, 1)
        np.add.at(mass[layer_index], layer_ids, layer_weights)
    return AggregatedRouting(
        selection_counts=counts,
        routing_mass=mass,
        total_routed_slots=int(decoded.expert_ids.size),
    )


def _decode_npy(payload: str, field_name: str) -> NDArray[np.generic]:
    try:
        raw = base64.b64decode(payload, validate=True)
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except (ValueError, TypeError, OSError) as error:
        raise ValueError(f"invalid {field_name} payload") from error
