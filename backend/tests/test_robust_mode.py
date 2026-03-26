from __future__ import annotations

from app.main import _finalize_pareto_options, _pick_best_option, _route_selection_score_map, settings
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
