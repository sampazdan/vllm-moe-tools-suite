from __future__ import annotations

import numpy as np

from .domain import (
    ExpertProfile,
    ModelTopology,
    ProfileLayer,
    ProfileProposal,
    ProfileValidation,
    RoutingSummary,
)


def validate_profile(
    profile: ExpertProfile,
    topology: ModelTopology,
) -> ProfileValidation:
    """Validate an expert profile against the loaded model topology."""

    errors: list[str] = []
    routed_layers = set(topology.routed_layer_ids)
    for raw_layer_id, layer in profile.layers.items():
        layer_id = int(raw_layer_id)
        if layer_id not in routed_layers:
            errors.append(f"unknown routed layer {layer_id}")
        invalid = sorted(
            expert_id
            for expert_id in layer.keep
            if not 0 <= expert_id < topology.num_experts
        )
        if invalid:
            errors.append(f"layer {layer_id} has invalid experts {invalid}")
        if len(layer.keep) < topology.top_k:
            errors.append(
                f"layer {layer_id} keeps {len(layer.keep)} experts; "
                f"top_k requires at least {topology.top_k}"
            )

    full_total = topology.num_layers * topology.num_experts
    eligible = full_total
    for raw_layer_id, layer in profile.layers.items():
        if int(raw_layer_id) in routed_layers:
            eligible -= topology.num_experts - len(layer.keep)
    return ProfileValidation(
        valid=not errors,
        errors=errors,
        eligible_experts=eligible,
        total_experts=full_total,
        retained_fraction=eligible / full_total,
    )


def propose_fixed_budget_profile(
    summary: RoutingSummary,
    topology: ModelTopology,
    keep_per_layer: int,
    metric: str,
) -> ProfileProposal:
    """Propose a deterministic top-expert profile from an observed run."""

    if keep_per_layer < topology.top_k:
        raise ValueError(f"keep_per_layer must be at least top_k={topology.top_k}")
    if keep_per_layer > topology.num_experts:
        raise ValueError(
            f"keep_per_layer cannot exceed {topology.num_experts} experts"
        )
    values = np.asarray(
        summary.routing_mass
        if metric == "routing_mass"
        else summary.selection_counts,
        dtype=np.float64,
    )
    layers: dict[str, ProfileLayer] = {}
    for layer_index, layer_id in enumerate(summary.layer_ids):
        order = np.lexsort((np.arange(topology.num_experts), -values[layer_index]))
        keep = np.sort(order[:keep_per_layer]).astype(int).tolist()
        layers[str(layer_id)] = ProfileLayer(keep=keep)

    profile = ExpertProfile(layers=layers)
    return ProfileProposal(
        profile=profile,
        validation=validate_profile(profile, topology),
        observed_mass_retained=calculate_observed_mass_retained(summary, profile),
    )


def calculate_observed_mass_retained(
    summary: RoutingSummary, profile: ExpertProfile
) -> float:
    """Calculate retained routing mass for an arbitrary valid profile."""

    mass_values = np.asarray(summary.routing_mass, dtype=np.float64)
    total_mass = float(mass_values.sum())
    retained_mass = 0.0
    for layer_index, layer_id in enumerate(summary.layer_ids):
        layer = profile.layers.get(str(layer_id))
        if layer is None:
            retained_mass += float(mass_values[layer_index].sum())
        else:
            retained_mass += float(mass_values[layer_index, layer.keep].sum())
    return retained_mass / total_mass if total_mass else 1.0
