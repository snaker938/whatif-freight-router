from __future__ import annotations

import math

import pytest

from scripts.evaluation_metrics import (
    additive_epsilon_indicator,
    ambiguity_alignment,
    ambiguity_prior_overtrigger_rate,
    ambiguity_prior_top_k_precision,
    as_float,
    balanced_gain_score,
    certificate_margin,
    certificate_runner_up_gap,
    controller_cost_per_certificate_point,
    competitor_turnover_rate,
    controller_activation_on_high_ambiguity,
    coverage_of_baseline,
    corridor_family_recall,
    fragility_entropy,
    dominates,
    frontier_diversity,
    frontier_diversity_index,
    frontier_entropy,
    frontier_action_gain,
    frontier_from_rows,
    hypervolume_3d,
    near_tie_mass,
    nominal_winner_margin,
    normalized_weighted_ranking,
    pairwise_weighted_sum_score,
    pearson_binary_correlation,
    refine_cost_mae_ms,
    refine_cost_mape,
    refine_cost_rank_correlation,
    refine_cost_prediction_error,
    refine_cost_positive_sample_count,
    refine_cost_sample_count,
    refine_cost_zero_observed_count,
    percentile,
    robust_win,
    route_metrics,
    memory_per_unit,
    runtime_ratio,
    runtime_per_unit,
    runtime_share,
    score_ranked_recall,
    time_to_best_iteration,
    action_efficiency,
    bytes_to_megabytes,
    cache_reuse_ratio,
    certificate_gain_per_world,
    value_per_second,
    frontier_gain_per_ms,
    pearson_correlation,
    productive_action_rate,
    quality_per_second,
    route_improvement_per_second,
)


pytestmark = [pytest.mark.thesis, pytest.mark.thesis_modules]

EXACT_TWO_POINT_FRONTIER = [
    {"route_id": "r1", "duration_s": 1.0, "monetary_cost": 4.0, "emissions_kg": 4.0},
    {"route_id": "r2", "duration_s": 4.0, "monetary_cost": 1.0, "emissions_kg": 1.0},
]
EXACT_REFERENCE_POINT = {"duration_s": 5.0, "monetary_cost": 5.0, "emissions_kg": 5.0}


def test_route_metrics_and_numeric_coercion_are_stable() -> None:
    route = {
        "metrics": {
            "distance_km": "10.5",
            "duration_s": "120",
            "monetary_cost": 15,
            "emissions_kg": None,
        }
    }
    assert route_metrics(route) == {
        "distance_km": 10.5,
        "duration_s": 120.0,
        "monetary_cost": 15.0,
        "emissions_kg": 0.0,
    }
    assert as_float(float("inf"), default=7.0) == 7.0


def test_dominance_and_balanced_gain_metrics() -> None:
    improved = {"duration_s": 90.0, "monetary_cost": 10.0, "emissions_kg": 5.0}
    baseline = {"duration_s": 100.0, "monetary_cost": 11.0, "emissions_kg": 6.0}
    assert dominates(improved, baseline) is True
    assert dominates(baseline, improved) is False
    assert balanced_gain_score(improved, baseline) > 0.0


def test_pairwise_weighted_sum_prefers_lower_normalized_objectives() -> None:
    route_a = {"duration_s": 90.0, "monetary_cost": 12.0, "emissions_kg": 4.0}
    route_b = {"duration_s": 110.0, "monetary_cost": 18.0, "emissions_kg": 7.0}
    score_a, score_b = pairwise_weighted_sum_score(route_a, route_b, weights=(1.0, 1.0, 1.0))
    assert score_a < score_b
    assert score_a == pytest.approx(0.0)
    assert score_b == pytest.approx(3.0)


def test_pairwise_weighted_sum_matches_exact_pairwise_scores() -> None:
    route_a = {"duration_s": 10.0, "monetary_cost": 5.0, "emissions_kg": 2.0}
    route_b = {"duration_s": 12.0, "monetary_cost": 3.0, "emissions_kg": 4.0}

    score_a, score_b = pairwise_weighted_sum_score(
        route_a,
        route_b,
        weights=(0.5, 0.25, 0.25),
    )

    assert score_a == 0.25
    assert score_b == 0.75


def test_frontier_hypervolume_coverage_and_epsilon_are_hand_checkable() -> None:
    frontier_rows = frontier_from_rows(
        [
            {"route_id": "a", "duration_s": 1.0, "monetary_cost": 2.0, "emissions_kg": 3.0},
            {"route_id": "b", "duration_s": 2.0, "monetary_cost": 4.0, "emissions_kg": 5.0},
            {"route_id": "a", "duration_s": 1.0, "monetary_cost": 2.0, "emissions_kg": 3.0},
        ]
    )
    assert [row["route_id"] for row in frontier_rows] == ["a"]
    reference = {"duration_s": 5.0, "monetary_cost": 5.0, "emissions_kg": 5.0}
    baseline = {"duration_s": 2.0, "monetary_cost": 3.0, "emissions_kg": 4.0}
    assert hypervolume_3d(frontier_rows, reference=reference) == pytest.approx(24.0)
    assert coverage_of_baseline(frontier_rows, baseline) == 1.0
    assert additive_epsilon_indicator(frontier_rows, baseline) == pytest.approx(-1.0)


def test_frontier_from_rows_deduplicates_ids_and_excludes_dominated_rows() -> None:
    frontier_rows = frontier_from_rows(
        [
            {"route_id": "r2", "duration_s": 12.0, "monetary_cost": 4.0, "emissions_kg": 4.0},
            {"route_id": "r1", "duration_s": 10.0, "monetary_cost": 5.0, "emissions_kg": 5.0},
            {"route_id": "r3", "duration_s": 11.0, "monetary_cost": 6.0, "emissions_kg": 6.0},
            {"route_id": "r1", "duration_s": 999.0, "monetary_cost": 999.0, "emissions_kg": 999.0},
        ]
    )

    assert frontier_rows == [
        {"route_id": "r1", "duration_s": 10.0, "monetary_cost": 5.0, "emissions_kg": 5.0},
        {"route_id": "r2", "duration_s": 12.0, "monetary_cost": 4.0, "emissions_kg": 4.0},
    ]


def test_two_point_frontier_hypervolume_matches_exact_union_volume() -> None:
    # Union of [1,5]x[4,5]x[4,5] and [4,5]x[1,5]x[1,5]:
    # 4 + 16 - 1 = 19.
    assert hypervolume_3d(EXACT_TWO_POINT_FRONTIER, reference=EXACT_REFERENCE_POINT) == 19.0


def test_two_point_frontier_epsilon_and_coverage_are_exact() -> None:
    assert additive_epsilon_indicator(EXACT_TWO_POINT_FRONTIER, EXACT_REFERENCE_POINT) == -1.0
    assert coverage_of_baseline(EXACT_TWO_POINT_FRONTIER, EXACT_REFERENCE_POINT) == 1.0
    assert coverage_of_baseline(
        EXACT_TWO_POINT_FRONTIER,
        {"duration_s": 0.5, "monetary_cost": 0.5, "emissions_kg": 0.5},
    ) == 0.0


def test_frontier_diversity_and_binary_correlation_are_positive_for_separated_frontiers() -> None:
    spread, crowding = frontier_diversity(
        [
            {"route_id": "a", "duration_s": 10.0, "monetary_cost": 20.0, "emissions_kg": 30.0},
            {"route_id": "b", "duration_s": 14.0, "monetary_cost": 18.0, "emissions_kg": 26.0},
            {"route_id": "c", "duration_s": 18.0, "monetary_cost": 17.0, "emissions_kg": 22.0},
        ]
    )
    assert spread > 0.0
    assert crowding > 0.0
    assert frontier_diversity([{"route_id": "solo", "duration_s": 1.0, "monetary_cost": 1.0, "emissions_kg": 1.0}]) == (0.0, 0.0)
    corr = pearson_binary_correlation([0.05, 0.1, 0.9, 1.0], [0, 0, 1, 1])
    assert corr is not None and corr > 0.9


def test_frontier_diversity_index_entropy_and_time_to_best_are_deterministic() -> None:
    frontier = [
        {"route_id": "winner", "duration_s": 10.0, "monetary_cost": 20.0, "emissions_kg": 30.0, "score": 0.1},
        {"route_id": "close", "duration_s": 11.0, "monetary_cost": 19.0, "emissions_kg": 29.0, "score": 0.2},
        {"route_id": "far", "duration_s": 18.0, "monetary_cost": 25.0, "emissions_kg": 35.0, "score": 0.9},
    ]
    assert frontier_diversity_index(frontier) == pytest.approx(0.625681)
    entropy = frontier_entropy(frontier)
    assert entropy is not None
    assert 0.0 <= entropy <= 1.0
    assert time_to_best_iteration(
        [
            {"iteration": 0, "selected_route_id": "close"},
            {"iteration": 2, "selected_route_id": "winner"},
            {"iteration": 3, "selected_route_id": "winner"},
        ],
        selected_route_id="winner",
    ) == 2


def test_percentile_uses_linear_interpolation_and_ignores_missing_values() -> None:
    assert percentile([10.0, 20.0, 40.0], 0.50) == 20.0
    assert percentile([10.0, 20.0, 40.0], 0.90) == 36.0
    assert percentile([10.0, None, float("nan"), 40.0], 0.25) == 17.5
    assert percentile([], 0.90) is None


def test_refine_cost_metrics_and_action_efficiency_are_exact() -> None:
    candidates = [
        {"predicted_refine_cost": 12.0, "observed_refine_cost": 10.0},
        {"predicted_refine_cost": 22.0, "observed_refine_cost": 20.0},
    ]
    assert refine_cost_sample_count(candidates) == 2
    assert refine_cost_positive_sample_count(candidates) == 2
    assert refine_cost_zero_observed_count(candidates) == 0
    assert refine_cost_mape(candidates) == 0.15
    assert refine_cost_mae_ms(candidates) == 2.0
    assert refine_cost_rank_correlation(candidates) == 1.0
    assert refine_cost_prediction_error(candidates) == refine_cost_mape(candidates)
    assert frontier_action_gain(frontier_count=3.0, frontier_diversity_index=0.5) == pytest.approx(0.5)
    assert action_efficiency(
        certificate_lift=0.1,
        frontier_gain=0.2,
        action_count=1,
        search_budget_used=1,
        evidence_budget_used=0,
    ) == pytest.approx(0.15)
    assert action_efficiency(
        certificate_lift=0.1,
        frontier_gain=0.2,
        action_count=0,
        search_budget_used=0,
        evidence_budget_used=0,
    ) is None
    assert value_per_second(0.3, 150.0) == 2.0
    assert value_per_second(None, 150.0) is None
    assert quality_per_second(weighted_margin=0.2, balanced_gain=0.1, runtime_ms=250.0) == 0.6
    assert route_improvement_per_second(weighted_margin=0.2, balanced_gain=0.1, runtime_ms=250.0) == 0.8
    assert route_improvement_per_second(weighted_margin=0.1, balanced_gain=0.2, runtime_ms=250.0) == 0.4
    assert frontier_gain_per_ms(0.5, 200.0) == 0.0025
    assert certificate_gain_per_world(0.2, 40.0) == 0.005
    assert controller_cost_per_certificate_point(120.0, 0.2) == 600.0
    assert controller_cost_per_certificate_point(120.0, 0.0) is None
    assert productive_action_rate(3.0, 4.0) == 0.75
    assert cache_reuse_ratio(0.5, 0.25) == 0.375


def test_refine_cost_metrics_keep_zero_observed_rows_visible_without_poisoning_mape() -> None:
    candidates = [
        {"predicted_refine_cost": 155.0, "observed_refine_cost": 0.0},
        {"predicted_refine_cost": 170.0, "observed_refine_cost": 45.0},
    ]

    assert refine_cost_sample_count(candidates) == 2
    assert refine_cost_positive_sample_count(candidates) == 1
    assert refine_cost_zero_observed_count(candidates) == 1
    assert refine_cost_mape(candidates) == pytest.approx(2.777778, rel=0.0, abs=1e-6)
    assert refine_cost_mae_ms(candidates) == 140.0
    assert refine_cost_rank_correlation(candidates) is not None


def test_refine_cost_rank_correlation_handles_inverse_ordering() -> None:
    candidates = [
        {"predicted_refine_cost": 10.0, "observed_refine_cost": 30.0},
        {"predicted_refine_cost": 20.0, "observed_refine_cost": 20.0},
        {"predicted_refine_cost": 30.0, "observed_refine_cost": 10.0},
    ]
    assert refine_cost_rank_correlation(candidates) == -1.0


def test_two_point_frontier_diversity_and_binary_correlation_are_exact() -> None:
    spread, crowding = frontier_diversity(EXACT_TWO_POINT_FRONTIER)

    assert spread == 1.732051
    assert crowding == 1.732051
    assert pearson_binary_correlation([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1]) == 0.894427
    assert pearson_correlation([0.1, 0.5, 0.9], [0.2, 0.6, 1.0]) == 1.0


def test_robust_win_uses_cvar_and_handles_missing_uncertainty() -> None:
    route = {"uncertainty": {"cvar95_duration_s": 110.0}}
    baseline = {"uncertainty": {"cvar95_duration_s": 130.0}}
    assert robust_win(route, baseline) is True
    assert robust_win({"uncertainty": {}}, baseline) is None
    assert robust_win({}, baseline) is None
    assert math.isfinite(as_float("12.5"))


def test_normalized_weighted_ranking_margin_and_near_tie_mass_are_deterministic() -> None:
    frontier = [
        {"route_id": "winner", "duration_s": 100.0, "monetary_cost": 20.0, "emissions_kg": 8.0},
        {"route_id": "close", "duration_s": 101.0, "monetary_cost": 20.1, "emissions_kg": 8.2},
        {"route_id": "far", "duration_s": 118.0, "monetary_cost": 24.0, "emissions_kg": 10.0},
    ]
    ranking = normalized_weighted_ranking(frontier)
    assert [row["route_id"] for row in ranking] == ["winner", "close", "far"]
    margin, winner_id = nominal_winner_margin(frontier)
    assert winner_id == "winner"
    assert margin == 0.180556
    assert near_tie_mass(frontier, tolerance=0.2) == 0.5


def test_certificate_fragility_and_competitor_metrics_capture_stability_shape() -> None:
    assert certificate_margin(0.87, threshold=0.8) == 0.07
    assert certificate_runner_up_gap({"winner": 0.87, "alt": 0.61, "alt2": 0.5}, winner_id="winner") == 0.26
    assert fragility_entropy({"scenario": 3.0, "toll": 1.0, "terrain": 2.0}) == 0.92062
    assert competitor_turnover_rate({"alt-a": {"scenario": 2}, "alt-b": {"terrain": 1, "toll": 1}}) == 0.5


def test_dccs_score_ranked_recall_and_family_recall_reflect_budgeted_selection() -> None:
    candidates = [
        {"candidate_id": "a", "final_score": 0.95, "decision_reason": "frontier_addition", "corridor_family": "north"},
        {"candidate_id": "b", "final_score": 0.85, "decision_reason": "decision_flip", "corridor_family": "west"},
        {"candidate_id": "c", "final_score": 0.10, "decision_reason": "redundant", "corridor_family": "north"},
    ]
    positives = {"frontier_addition", "decision_flip", "challenger_but_not_added"}
    assert score_ranked_recall(candidates, budget=1, positive_labels=positives) == 0.5
    assert score_ranked_recall(candidates, budget=2, positive_labels=positives) == 1.0
    assert corridor_family_recall(candidates, budget=1, positive_labels=positives) == 0.5
    assert corridor_family_recall(candidates, budget=2, positive_labels=positives) == 1.0


def test_runtime_share_is_bounded() -> None:
    assert runtime_share(20.0, 100.0) == 0.2
    assert runtime_share(120.0, 100.0) == 1.0
    assert runtime_share(None, 100.0) is None


def test_ambiguity_prior_precision_and_overtrigger_metrics_are_stable() -> None:
    priors = [0.82, 0.73, 0.44, 0.18]
    observed = [0.21, 0.03, 0.12, 0.01]

    assert ambiguity_prior_top_k_precision(priors, observed, k=2) == 0.5
    assert ambiguity_prior_overtrigger_rate(priors, observed) == 0.5


def test_runtime_ratio_and_memory_conversion_are_stable() -> None:
    assert runtime_ratio(100.0, 20.0) == 5.0
    assert runtime_ratio(None, 20.0) is None
    assert runtime_per_unit(100.0, 4.0) == 25.0
    assert runtime_per_unit(100.0, 0.0) is None
    assert memory_per_unit(64.0, 4.0) == 16.0
    assert ambiguity_alignment(0.8, 0.6) == 0.8
    assert ambiguity_alignment(None, 0.6) is None
    assert controller_activation_on_high_ambiguity(0.7, True) is True
    assert controller_activation_on_high_ambiguity(0.2, True) is None
    assert bytes_to_megabytes(1048576) == 1.0
    assert bytes_to_megabytes(-1) is None
