from __future__ import annotations

import hashlib
import json

import numpy as np

from .domain import (
    ExpertProfile,
    ModelTopology,
    ProfileLayer,
    ProfileProposal,
    ProfileValidation,
    RoutingSummary,
)

EXPERT_PROFILE_FINGERPRINT_VERSION = 1


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
        valid_experts = {
            expert_id
            for expert_id in layer.keep
            if 0 <= expert_id < topology.num_experts
        }
        invalid = sorted(
            expert_id
            for expert_id in layer.keep
            if not 0 <= expert_id < topology.num_experts
        )
        if invalid:
            errors.append(f"layer {layer_id} has invalid experts {invalid}")
        if len(valid_experts) < topology.top_k:
            errors.append(
                f"layer {layer_id} keeps {len(valid_experts)} valid experts; "
                f"top_k requires at least {topology.top_k}"
            )

    full_total = topology.num_layers * topology.num_experts
    eligible = full_total
    for raw_layer_id, layer in profile.layers.items():
        if int(raw_layer_id) in routed_layers:
            valid_count = sum(
                0 <= expert_id < topology.num_experts for expert_id in layer.keep
            )
            eligible -= topology.num_experts - valid_count
    return ProfileValidation(
        valid=not errors,
        errors=errors,
        eligible_experts=eligible,
        total_experts=full_total,
        retained_fraction=eligible / full_total,
    )


def canonical_full_profile_layers(
    profile: ExpertProfile,
    topology: ModelTopology,
) -> list[dict[str, object]]:
    """Expand a sparse profile into the canonical full runtime mask."""
    validation = validate_profile(profile, topology)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    return [
        {
            "layer_id": layer_id,
            "keep": sorted(
                profile.layers[str(layer_id)].keep
                if str(layer_id) in profile.layers
                else range(topology.num_experts)
            ),
        }
        for layer_id in sorted(topology.routed_layer_ids)
    ]


def canonical_profile_layer_map(
    profile: ExpertProfile,
    topology: ModelTopology,
) -> dict[str, dict[str, list[int]]]:
    return {
        str(layer["layer_id"]): {"keep": list(layer["keep"])}
        for layer in canonical_full_profile_layers(profile, topology)
    }


def expert_profile_fingerprint(
    profile: ExpertProfile,
    topology: ModelTopology,
) -> str:
    payload = {
        "version": EXPERT_PROFILE_FINGERPRINT_VERSION,
        "layers": canonical_full_profile_layers(profile, topology),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def legacy_profile_fingerprint(profile: ExpertProfile) -> str:
    """Reproduce the pre-runtime-mask digest for persisted legacy records."""
    encoded = json.dumps(
        profile.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        raise ValueError(f"keep_per_layer cannot exceed {topology.num_experts} experts")
    if summary.layer_ids != topology.routed_layer_ids:
        raise ValueError("routing layer IDs do not match the model topology")
    if metric == "routing_mass":
        raw_values = summary.routing_mass
    elif metric == "selection_count":
        raw_values = summary.selection_counts
    else:
        raise ValueError("metric must be routing_mass or selection_count")
    values = np.asarray(raw_values, dtype=np.float64)
    expected_shape = (topology.num_layers, topology.num_experts)
    if values.shape != expected_shape:
        raise ValueError(
            f"routing summary has shape {values.shape}, expected {expected_shape}"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("routing summary values must be finite and non-negative")
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
    if mass_values.ndim != 2 or mass_values.shape[0] != len(summary.layer_ids):
        raise ValueError("routing mass rows must match routing layer IDs")
    if not np.isfinite(mass_values).all() or np.any(mass_values < 0):
        raise ValueError("routing mass values must be finite and non-negative")
    total_mass = float(mass_values.sum())
    retained_mass = 0.0
    for layer_index, layer_id in enumerate(summary.layer_ids):
        layer = profile.layers.get(str(layer_id))
        if layer is None:
            retained_mass += float(mass_values[layer_index].sum())
        else:
            if any(
                expert_id < 0 or expert_id >= mass_values.shape[1]
                for expert_id in layer.keep
            ):
                raise ValueError(f"profile layer {layer_id} is outside routing mass")
            retained_mass += float(mass_values[layer_index, layer.keep].sum())
    return retained_mass / total_mass if total_mass else 1.0
