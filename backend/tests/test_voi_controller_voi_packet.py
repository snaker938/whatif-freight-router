from __future__ import annotations

import pytest

import app.voi_controller as voi_module
from app.voi_controller import VOIConfig, VOIControllerState

pytestmark = pytest.mark.thesis_modules


def test_uncertified_evidence_plateau_preference_recovers_after_harmful_frontier_only_probe() -> None:
    state = VOIControllerState(
        iteration_index=2,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.3, 10.1, 10.05)},
            {"route_id": "route_c", "objective_vector": (10.5, 10.25, 10.12)},
        ],
        certificate={"route_a": 0.559701, "route_b": 0.440299},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": -0.041802,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_changed": False,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.061869,
                "realized_evidence_uncertainty_delta": -0.043823,
                "realized_productive": True,
            }
        ],
        certificate_margin=0.003398,
        near_tie_mass=0.0,
        search_completeness_score=0.51496,
        search_completeness_gap=0.32504,
        pending_challenger_mass=0.685966,
        best_pending_flip_probability=0.999877,
        corridor_family_recall=1.0,
        frontier_recall_at_budget=0.332389,
        top_refresh_gain=0.268657,
        top_fragility_mass=0.559701,
        competitor_pressure=1.0,
        support_richness=0.728969,
        prior_support_strength=0.728969,
        ambiguity_pressure=0.651489,
        ambiguity_context={
            "selected_candidate_source_stage": "osrm_refined",
            "selected_final_route_source_stage": "osrm_refined",
            "od_hard_case_prior": 0.651489,
            "ambiguity_budget_prior": 0.651489,
            "od_ambiguity_support_ratio": 0.876689,
            "od_ambiguity_source_entropy": 0.96023,
            "od_ambiguity_index": 0.41312,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.437508,
        predicted_delta_certificate=0.440299,
        predicted_delta_margin=0.371087,
        predicted_delta_frontier=0.464152,
        metadata={
            "normalized_objective_gap": 0.421749,
            "normalized_mechanism_gap": 0.112269,
            "normalized_overlap_reduction": 0.904762,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:scenario",
        kind="refresh_top1_vor",
        target="scenario",
        q_score=0.035775,
        predicted_delta_certificate=0.043121,
        predicted_delta_margin=0.027836,
        predicted_delta_frontier=0.0,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.060526,
        predicted_delta_certificate=0.085075,
        predicted_delta_margin=0.027985,
        predicted_delta_frontier=0.01209,
        metadata={"near_tie_mass": 0.0},
    )

    adjusted_actions = voi_module._apply_uncertified_evidence_plateau_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.559701,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_resample = next(
        action for action in adjusted_actions if action.kind == "increase_stochastic_samples"
    )
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_evidence_plateau_preference_applied"] is True
    assert adjusted_resample.metadata["uncertified_evidence_plateau_frontier_probe_recovery"] is True
    assert adjusted_refine.metadata["uncertified_evidence_plateau_search_discount_applied"] is True
    assert adjusted_refine.metadata["uncertified_evidence_plateau_frontier_probe_discount"] is True


def test_strong_winner_side_refresh_preference_bridges_after_productive_frontier_only_refine() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.03, 10.02, 10.01)},
            {"route_id": "route_c", "objective_vector": (10.11, 10.05, 10.03)},
        ],
        certificate={"route_a": 0.692913, "route_b": 0.307087, "route_c": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.141189,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_changed": False,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -1.156986,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": True,
            }
        ],
        certificate_margin=0.098634,
        search_completeness_score=0.538718,
        search_completeness_gap=0.301282,
        pending_challenger_mass=0.62609,
        best_pending_flip_probability=0.996634,
        corridor_family_recall=0.25,
        frontier_recall_at_budget=0.264849,
        support_richness=0.60675,
        prior_support_strength=0.60675,
        ambiguity_pressure=0.665164,
        top_refresh_gain=0.338583,
        top_fragility_mass=0.692913,
        competitor_pressure=1.0,
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.64432,
            "od_ambiguity_source_entropy": 0.78166,
            "od_hard_case_prior": 0.36199,
            "ambiguity_budget_prior": 0.26,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.24890263903260462,
        predicted_delta_certificate=0.30708661417322836,
        predicted_delta_margin=0.20864943200062386,
        predicted_delta_frontier=0.0637551684269397,
        metadata={
            "normalized_objective_gap": 0.029289451040590997,
            "normalized_mechanism_gap": 0.1165980795005731,
            "normalized_overlap_reduction": 0.9047619047619048,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.307087,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.04500089995499909,
        predicted_delta_certificate=0.05496431765893251,
        predicted_delta_margin=0.03503799531949667,
        predicted_delta_frontier=0.0,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
            "empirical_refresh_certificate_delta": -0.354331,
        },
    )

    adjusted_actions = voi_module._apply_strong_winner_side_refresh_preference(
        [refine, refresh],
        state=state,
        current_certificate=0.692913,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_refresh = next(action for action in adjusted_actions if action.kind == "refresh_top1_vor")
    assert adjusted_refresh.q_score > adjusted_refine.q_score
    assert adjusted_refresh.metadata["winner_side_refresh_preference_applied"] is True
    assert adjusted_refresh.metadata["winner_side_refresh_preference_productive_frontier_probe_bridge"] is True
    assert adjusted_refine.metadata["winner_side_refresh_refine_discount_applied"] is True
