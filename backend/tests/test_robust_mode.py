from __future__ import annotations

from app.main import _clone_option_with_objectives, _finalize_pareto_options, _pick_best_option, _route_selection_score_map, settings
from app.models import GeoJSONLineString, RouteMetrics, RouteOption


def _option(
    *,
    route_id: str,
    duration_s: float,
    money: float,
    co2: float,
    mean_duration_s: float,
    std_duration_s: float,
    distance_km: float = 30.0,
    cvar95_duration_s: float | None = None,
    mean_money: float | None = None,
    std_money: float = 0.0,
    cvar95_money: float | None = None,
    mean_co2: float | None = None,
    std_co2: float = 0.0,
    cvar95_co2: float | None = None,
) -> RouteOption:
    utility_mean = float(mean_duration_s)
    utility_q95 = float(mean_duration_s + max(0.0, std_duration_s))
    utility_cvar95 = float(cvar95_duration_s if cvar95_duration_s is not None else utility_q95)
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(-1.0, 52.0), (-0.5, 51.8)]),
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_s=duration_s,
            monetary_cost=money,
            emissions_kg=co2,
            avg_speed_kmh=50.0,
        ),
        uncertainty={
            "mean_duration_s": mean_duration_s,
            "std_duration_s": std_duration_s,
            "p95_duration_s": mean_duration_s + std_duration_s,
            "cvar95_duration_s": (
                cvar95_duration_s if cvar95_duration_s is not None else mean_duration_s + std_duration_s
            ),
            "mean_monetary_cost": mean_money if mean_money is not None else money,
            "std_monetary_cost": std_money,
            "cvar95_monetary_cost": cvar95_money if cvar95_money is not None else money + std_money,
            "mean_emissions_kg": mean_co2 if mean_co2 is not None else co2,
            "std_emissions_kg": std_co2,
            "cvar95_emissions_kg": cvar95_co2 if cvar95_co2 is not None else co2 + std_co2,
            "utility_mean": utility_mean,
            "utility_q95": utility_q95,
            "utility_cvar95": utility_cvar95,
            "robust_score": utility_mean + max(0.0, utility_cvar95 - utility_mean),
        },
    )


def _thesis_time_preserving_guard_context() -> dict[str, float | int | str]:
    return {
        "od_ambiguity_index": 0.224232,
        "od_ambiguity_confidence": 0.868936,
        "od_engine_disagreement_prior": 0.391726,
        "od_hard_case_prior": 0.407092,
        "od_ambiguity_support_ratio": 0.574272,
        "od_ambiguity_source_entropy": 0.83723,
        "od_candidate_path_count": 12,
        "od_corridor_family_count": 5,
        "ambiguity_budget_band": "high",
    }


def _nonqualifying_time_preserving_guard_context() -> dict[str, float | int | str]:
    return {
        "od_ambiguity_index": 0.11,
        "od_ambiguity_confidence": 0.31,
        "od_engine_disagreement_prior": 0.18,
        "od_hard_case_prior": 0.14,
        "od_ambiguity_support_ratio": 0.32,
        "od_ambiguity_source_entropy": 0.28,
        "od_candidate_path_count": 1,
        "od_corridor_family_count": 1,
        "ambiguity_budget_band": "low",
    }


def test_pick_best_option_changes_between_expected_and_robust() -> None:
    high_variance_fast = _option(
        route_id="a",
        duration_s=100.0,
        money=100.0,
        co2=100.0,
        mean_duration_s=95.0,
        std_duration_s=40.0,
    )
    low_variance_slower = _option(
        route_id="b",
        duration_s=105.0,
        money=100.0,
        co2=100.0,
        mean_duration_s=105.0,
        std_duration_s=1.0,
    )
    options = [high_variance_fast, low_variance_slower]

    expected_pick = _pick_best_option(
        options,
        w_time=1.0,
        w_money=0.0,
        w_co2=0.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
    )
    robust_pick = _pick_best_option(
        options,
        w_time=1.0,
        w_money=0.0,
        w_co2=0.0,
        optimization_mode="robust",
        risk_aversion=1.0,
    )

    assert expected_pick.id == "a"
    assert robust_pick.id == "b"


def test_finalize_pareto_options_uses_robust_sorting() -> None:
    option_a = _option(
        route_id="a",
        duration_s=100.0,
        money=80.0,
        co2=150.0,
        mean_duration_s=95.0,
        std_duration_s=30.0,
        mean_money=80.0,
        mean_co2=150.0,
    )
    option_b = _option(
        route_id="b",
        duration_s=110.0,
        money=30.0,
        co2=70.0,
        mean_duration_s=110.0,
        std_duration_s=1.0,
        mean_money=30.0,
        mean_co2=70.0,
    )
    options = [option_a, option_b]

    expected = _finalize_pareto_options(
        options,
        max_alternatives=2,
        optimization_mode="expected_value",
        risk_aversion=1.0,
    )
    robust = _finalize_pareto_options(
        options,
        max_alternatives=2,
        optimization_mode="robust",
        risk_aversion=1.0,
    )

    assert expected[0].id == "a"
    assert robust[0].id == "b"


def test_finalize_pareto_options_uses_cvar_tail_for_robust_ordering() -> None:
    option_a = _option(
        route_id="a",
        duration_s=95.0,
        money=100.0,
        co2=100.0,
        mean_duration_s=95.0,
        std_duration_s=2.0,
        cvar95_duration_s=150.0,
        mean_money=100.0,
        mean_co2=100.0,
    )
    option_b = _option(
        route_id="b",
        duration_s=100.0,
        money=100.0,
        co2=100.0,
        mean_duration_s=100.0,
        std_duration_s=12.0,
        cvar95_duration_s=102.0,
        mean_money=100.0,
        mean_co2=100.0,
    )
    options = [option_a, option_b]

    expected = _finalize_pareto_options(
        options,
        max_alternatives=2,
        optimization_mode="expected_value",
        risk_aversion=1.0,
    )
    robust = _finalize_pareto_options(
        options,
        max_alternatives=2,
        optimization_mode="robust",
        risk_aversion=1.0,
    )

    assert expected[0].id == "a"
    assert robust[0].id == "b"


def test_robust_mode_prefers_cvar_over_std_when_available() -> None:
    # Route `a` has lower std but much worse tail; robust mode should use CVaR tail.
    option_a = _option(
        route_id="a",
        duration_s=95.0,
        money=90.0,
        co2=90.0,
        mean_duration_s=100.0,
        std_duration_s=2.0,
        cvar95_duration_s=150.0,
    )
    option_b = _option(
        route_id="b",
        duration_s=100.0,
        money=90.0,
        co2=90.0,
        mean_duration_s=104.0,
        std_duration_s=12.0,
        cvar95_duration_s=106.0,
    )

    picked = _pick_best_option(
        [option_a, option_b],
        w_time=1.0,
        w_money=0.0,
        w_co2=0.0,
        optimization_mode="robust",
        risk_aversion=1.0,
    )
    assert picked.id == "b"


def test_pick_best_option_supports_vikor_profiles() -> None:
    option_a = _option(
        route_id="a",
        duration_s=100.0,
        money=95.0,
        co2=95.0,
        distance_km=65.0,
        mean_duration_s=100.0,
        std_duration_s=1.0,
    )
    option_b = _option(
        route_id="b",
        duration_s=103.0,
        money=95.0,
        co2=95.0,
        distance_km=20.0,
        mean_duration_s=103.0,
        std_duration_s=1.0,
    )
    profiles = (
        "academic_vikor",
        "modified_vikor_distance",
    )
    original_profile = settings.route_selection_math_profile
    try:
        for profile in profiles:
            settings.route_selection_math_profile = profile
            picked = _pick_best_option(
                [option_a, option_b],
                w_time=0.6,
                w_money=0.2,
                w_co2=0.2,
                optimization_mode="expected_value",
                risk_aversion=1.0,
            )
            assert picked.id in {"a", "b"}
    finally:
        settings.route_selection_math_profile = original_profile


def test_route_selection_score_map_matches_robust_pick() -> None:
    option_a = _option(
        route_id="a",
        duration_s=100.0,
        money=80.0,
        co2=110.0,
        mean_duration_s=96.0,
        std_duration_s=22.0,
        cvar95_duration_s=150.0,
        mean_money=82.0,
        cvar95_money=95.0,
        mean_co2=112.0,
        cvar95_co2=120.0,
    )
    option_b = _option(
        route_id="b",
        duration_s=104.0,
        money=79.0,
        co2=109.0,
        mean_duration_s=102.0,
        std_duration_s=4.0,
        cvar95_duration_s=110.0,
        mean_money=80.0,
        cvar95_money=84.0,
        mean_co2=110.0,
        cvar95_co2=114.0,
    )
    option_c = _option(
        route_id="c",
        duration_s=98.0,
        money=92.0,
        co2=118.0,
        mean_duration_s=97.0,
        std_duration_s=10.0,
        cvar95_duration_s=118.0,
        mean_money=93.0,
        cvar95_money=101.0,
        mean_co2=119.0,
        cvar95_co2=127.0,
    )

    options = [option_a, option_b, option_c]
    picked = _pick_best_option(
        options,
        w_time=0.6,
        w_money=0.2,
        w_co2=0.2,
        optimization_mode="robust",
        risk_aversion=1.0,
    )
    score_map = _route_selection_score_map(
        options,
        w_time=0.6,
        w_money=0.2,
        w_co2=0.2,
        optimization_mode="robust",
        risk_aversion=1.0,
    )

    assert picked.id == min(score_map, key=score_map.get)


def test_robust_pick_respects_weighted_tradeoff_across_objectives() -> None:
    motorway_fast = _option(
        route_id="motorway_fast",
        duration_s=96.0,
        money=98.0,
        co2=126.0,
        mean_duration_s=96.0,
        std_duration_s=18.0,
        cvar95_duration_s=125.0,
        mean_money=100.0,
        cvar95_money=122.0,
        mean_co2=129.0,
        cvar95_co2=150.0,
    )
    balanced_corridor = _option(
        route_id="balanced_corridor",
        duration_s=101.0,
        money=84.0,
        co2=108.0,
        mean_duration_s=102.0,
        std_duration_s=4.0,
        cvar95_duration_s=109.0,
        mean_money=84.0,
        cvar95_money=89.0,
        mean_co2=108.0,
        cvar95_co2=114.0,
    )

    picked = _pick_best_option(
        [motorway_fast, balanced_corridor],
        w_time=0.2,
        w_money=0.4,
        w_co2=0.4,
        optimization_mode="robust",
        risk_aversion=1.0,
    )
    score_map = _route_selection_score_map(
        [motorway_fast, balanced_corridor],
        w_time=0.2,
        w_money=0.4,
        w_co2=0.4,
        optimization_mode="robust",
        risk_aversion=1.0,
    )

    assert picked.id == "balanced_corridor"
    assert picked.id == min(score_map, key=score_map.get)


def test_pick_best_option_preserves_time_when_savings_are_marginal() -> None:
    time_preserving = _option(
        route_id="time_preserving",
        duration_s=15433.39,
        money=328.77,
        co2=500.293,
        mean_duration_s=15433.39,
        std_duration_s=5.0,
        cvar95_duration_s=15445.0,
        mean_money=328.77,
        cvar95_money=329.5,
        mean_co2=500.293,
        cvar95_co2=501.0,
        distance_km=311.33,
    )
    marginal_savings = _option(
        route_id="marginal_savings",
        duration_s=16599.31,
        money=327.23,
        co2=492.071,
        mean_duration_s=16599.31,
        std_duration_s=5.0,
        cvar95_duration_s=16612.0,
        mean_money=327.23,
        cvar95_money=328.0,
        mean_co2=492.071,
        cvar95_co2=493.0,
        distance_km=309.193,
    )

    options = [time_preserving, marginal_savings]
    picked = _pick_best_option(
        options,
        w_time=1.05,
        w_money=1.0,
        w_co2=1.05,
        optimization_mode="robust",
        risk_aversion=1.0,
    )
    score_map = _route_selection_score_map(
        options,
        w_time=1.05,
        w_money=1.0,
        w_co2=1.05,
        optimization_mode="robust",
        risk_aversion=1.0,
    )

    assert picked.id == "time_preserving"
    assert picked.id == min(score_map, key=score_map.get)


def test_pick_best_option_redesigned_time_preserving_frontier_guard_prefers_thesis_row_faster_frontier() -> None:
    slower_tradeoff = _option(
        route_id="slow_fallback",
        duration_s=26079.51,
        money=457.02,
        co2=673.632,
        mean_duration_s=26079.51,
        std_duration_s=5.0,
        cvar95_duration_s=26092.0,
        mean_money=457.02,
        cvar95_money=458.0,
        mean_co2=673.632,
        cvar95_co2=674.5,
        distance_km=448.091,
    )
    faster_frontier = _option(
        route_id="fast_frontier",
        duration_s=25556.74,
        money=465.59,
        co2=691.075,
        mean_duration_s=25556.74,
        std_duration_s=5.0,
        cvar95_duration_s=25569.0,
        mean_money=465.59,
        cvar95_money=466.5,
        mean_co2=691.075,
        cvar95_co2=692.0,
        distance_km=454.429,
    )

    options = [slower_tradeoff, faster_frontier]
    unguarded_pick = _pick_best_option(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
    )
    guarded_pick = _pick_best_option(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
        time_preserving_frontier_guard=True,
        ambiguity_context=_thesis_time_preserving_guard_context(),
    )
    guarded_score_map = _route_selection_score_map(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
        time_preserving_frontier_guard=True,
        ambiguity_context=_thesis_time_preserving_guard_context(),
    )

    assert unguarded_pick.id == "slow_fallback"
    assert guarded_pick.id == "fast_frontier"
    assert guarded_pick.id == min(guarded_score_map, key=guarded_score_map.get)


def test_pick_best_option_redesigned_time_preserving_frontier_guard_requires_normalized_supported_context() -> None:
    slower_tradeoff = _option(
        route_id="slow_fallback",
        duration_s=26079.51,
        money=457.02,
        co2=673.632,
        mean_duration_s=26079.51,
        std_duration_s=5.0,
        cvar95_duration_s=26092.0,
        mean_money=457.02,
        cvar95_money=458.0,
        mean_co2=673.632,
        cvar95_co2=674.5,
        distance_km=448.091,
    )
    faster_frontier = _option(
        route_id="fast_frontier",
        duration_s=25556.74,
        money=465.59,
        co2=691.075,
        mean_duration_s=25556.74,
        std_duration_s=5.0,
        cvar95_duration_s=25569.0,
        mean_money=465.59,
        cvar95_money=466.5,
        mean_co2=691.075,
        cvar95_co2=692.0,
        distance_km=454.429,
    )

    options = [slower_tradeoff, faster_frontier]
    unguarded_pick = _pick_best_option(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
    )
    guarded_pick = _pick_best_option(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
        time_preserving_frontier_guard=True,
        ambiguity_context=_nonqualifying_time_preserving_guard_context(),
    )

    assert unguarded_pick.id == "slow_fallback"
    assert guarded_pick.id == "slow_fallback"


def test_pick_best_option_redesigned_time_preserving_frontier_guard_does_not_override_large_sacrifices() -> None:
    slower_tradeoff = _option(
        route_id="slow_fallback",
        duration_s=26079.51,
        money=457.02,
        co2=673.632,
        mean_duration_s=26079.51,
        std_duration_s=5.0,
        cvar95_duration_s=26092.0,
        mean_money=457.02,
        cvar95_money=458.0,
        mean_co2=673.632,
        cvar95_co2=674.5,
        distance_km=448.091,
    )
    faster_but_costly = _option(
        route_id="fast_but_costly",
        duration_s=25556.74,
        money=492.50,
        co2=714.25,
        mean_duration_s=25556.74,
        std_duration_s=5.0,
        cvar95_duration_s=25569.0,
        mean_money=492.50,
        cvar95_money=493.5,
        mean_co2=714.25,
        cvar95_co2=715.0,
        distance_km=462.25,
    )

    options = [slower_tradeoff, faster_but_costly]
    guarded_pick = _pick_best_option(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
        time_preserving_frontier_guard=True,
        ambiguity_context=_thesis_time_preserving_guard_context(),
    )
    guarded_score_map = _route_selection_score_map(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
        time_preserving_frontier_guard=True,
        ambiguity_context=_thesis_time_preserving_guard_context(),
    )

    assert guarded_pick.id == "slow_fallback"
    assert guarded_pick.id == min(guarded_score_map, key=guarded_score_map.get)


def test_pick_best_option_redesigned_time_preserving_frontier_guard_blocks_excessive_time_loss() -> None:
    slower_tradeoff = _option(
        route_id="slow_tradeoff",
        duration_s=25556.74,
        money=465.59,
        co2=691.075,
        mean_duration_s=25556.74,
        std_duration_s=5.0,
        cvar95_duration_s=25569.0,
        mean_money=465.59,
        cvar95_money=466.5,
        mean_co2=691.075,
        cvar95_co2=692.0,
        distance_km=454.429,
    )
    faster_frontier = _option(
        route_id="faster_frontier",
        duration_s=23670.46,
        money=490.0,
        co2=735.0,
        mean_duration_s=23670.46,
        std_duration_s=5.0,
        cvar95_duration_s=23683.0,
        mean_money=490.0,
        cvar95_money=491.0,
        mean_co2=735.0,
        cvar95_co2=736.0,
        distance_km=460.0,
    )

    options = [slower_tradeoff, faster_frontier]
    unguarded_pick = _pick_best_option(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
    )
    guarded_pick = _pick_best_option(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
        time_preserving_frontier_guard=True,
        ambiguity_context=_thesis_time_preserving_guard_context(),
    )
    guarded_score_map = _route_selection_score_map(
        options,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="expected_value",
        risk_aversion=1.0,
        time_preserving_frontier_guard=True,
        ambiguity_context=_thesis_time_preserving_guard_context(),
    )

    assert unguarded_pick.id == "slow_tradeoff"
    assert guarded_pick.id == "faster_frontier"
    assert guarded_pick.id == min(guarded_score_map, key=guarded_score_map.get)


def test_clone_option_with_objectives_scales_uncertainty_tails_for_robust_selection() -> None:
    perturbed = _option(
        route_id="perturbed",
        duration_s=100.0,
        money=80.0,
        co2=90.0,
        mean_duration_s=110.0,
        std_duration_s=10.0,
        cvar95_duration_s=135.0,
        mean_money=88.0,
        std_money=4.0,
        cvar95_money=96.0,
        mean_co2=95.0,
        std_co2=5.0,
        cvar95_co2=103.0,
    )
    stable = _option(
        route_id="stable",
        duration_s=130.0,
        money=95.0,
        co2=100.0,
        mean_duration_s=130.0,
        std_duration_s=2.0,
        cvar95_duration_s=132.0,
        mean_money=95.0,
        std_money=1.0,
        cvar95_money=96.0,
        mean_co2=100.0,
        std_co2=1.0,
        cvar95_co2=101.0,
    )

    cloned = _clone_option_with_objectives(perturbed, (160.0, 120.0, 135.0))

    assert cloned.uncertainty is not None
    assert cloned.uncertainty["mean_duration_s"] == 176.0
    assert cloned.uncertainty["std_duration_s"] == 16.0
    assert cloned.uncertainty["cvar95_duration_s"] == 216.0
    assert cloned.uncertainty["mean_monetary_cost"] == 132.0
    assert cloned.uncertainty["cvar95_monetary_cost"] == 144.0
    assert cloned.uncertainty["mean_emissions_kg"] == 142.5
    assert cloned.uncertainty["cvar95_emissions_kg"] == 154.5

    robust_score_map = _route_selection_score_map(
        [cloned, stable],
        w_time=1.0,
        w_money=0.0,
        w_co2=0.0,
        optimization_mode="robust",
        risk_aversion=1.0,
    )

    assert min(robust_score_map, key=robust_score_map.get) == "stable"
