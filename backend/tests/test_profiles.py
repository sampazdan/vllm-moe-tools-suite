import pytest
from moe_tools_suite.domain import (
    ExpertProfile,
    ModelTopology,
    ProfileLayer,
    RoutingSummary,
)
from moe_tools_suite.profiles import propose_fixed_budget_profile, validate_profile
from pydantic import ValidationError


@pytest.fixture
def topology() -> ModelTopology:
    return ModelTopology(
        num_layers=2,
        num_experts=4,
        top_k=2,
        routed_layer_ids=[3, 7],
    )


def test_profile_matches_fork_schema_and_supports_sparse_layers(
    topology: ModelTopology,
) -> None:
    profile = ExpertProfile(layers={"3": ProfileLayer(keep=[0, 2])})

    validation = validate_profile(profile, topology)

    assert validation.valid
    assert validation.eligible_experts == 6
    assert validation.retained_fraction == 0.75
    assert profile.model_dump(mode="json") == {
        "version": 1,
        "layers": {"3": {"keep": [0, 2]}},
    }


@pytest.mark.parametrize("bad_keep", [[0], [0, 4]])
def test_profile_rejects_model_incompatible_experts(
    bad_keep: list[int], topology: ModelTopology
) -> None:
    profile = ExpertProfile(layers={"3": ProfileLayer(keep=bad_keep)})

    assert not validate_profile(profile, topology).valid


def test_invalid_experts_do_not_inflate_retained_fraction(
    topology: ModelTopology,
) -> None:
    profile = ExpertProfile(layers={"3": ProfileLayer(keep=[0, 1, 2, 3, 4])})

    validation = validate_profile(profile, topology)

    assert not validation.valid
    assert validation.eligible_experts == 8
    assert validation.retained_fraction == 1


def test_profile_rejects_boolean_expert_ids() -> None:
    with pytest.raises(ValidationError, match="integer expert IDs"):
        ExpertProfile.model_validate(
            {"version": 1, "layers": {"3": {"keep": [True, 1]}}}
        )


def test_fixed_budget_proposal_is_deterministic(topology: ModelTopology) -> None:
    summary = RoutingSummary(
        run_id="run-1",
        layer_ids=[3, 7],
        selection_counts=[[1, 4, 2, 3], [4, 3, 2, 1]],
        routing_mass=[[0.1, 0.8, 0.3, 0.7], [0.9, 0.8, 0.2, 0.1]],
        total_routed_slots=20,
    )

    proposal = propose_fixed_budget_profile(
        summary, topology, keep_per_layer=2, metric="routing_mass"
    )

    assert proposal.profile.layers["3"].keep == [1, 3]
    assert proposal.profile.layers["7"].keep == [0, 1]
    assert proposal.validation.valid
    assert proposal.observed_mass_retained == pytest.approx(3.2 / 3.9)


def test_fixed_budget_proposal_rejects_mislabeled_routing_rows(
    topology: ModelTopology,
) -> None:
    summary = RoutingSummary(
        run_id="run-1",
        layer_ids=[7, 3],
        selection_counts=[[1, 4, 2, 3], [4, 3, 2, 1]],
        routing_mass=[[0.1, 0.8, 0.3, 0.7], [0.9, 0.8, 0.2, 0.1]],
        total_routed_slots=20,
    )

    with pytest.raises(ValueError, match="layer IDs"):
        propose_fixed_budget_profile(
            summary,
            topology,
            keep_per_layer=2,
            metric="routing_mass",
        )
