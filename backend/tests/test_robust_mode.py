from __future__ import annotations

from app.main import _finalize_pareto_options, _pick_best_option
from app.models import GeoJSONLineString, RouteMetrics, RouteOption


def _option(
    *,
    route_id: str,
    duration_s: float,
    money: float,
    co2: float,
    mean_duration_s: float,
    std_duration_s: float,
    mean_money: float | None = None,
    std_money: float = 0.0,
    mean_co2: float | None = None,
    std_co2: float = 0.0,
) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(-1.0, 52.0), (-0.5, 51.8)]),
        metrics=RouteMetrics(
            distance_km=30.0,
            duration_s=duration_s,
            monetary_cost=money,
            emissions_kg=co2,
            avg_speed_kmh=50.0,
        ),
        uncertainty={
            "mean_duration_s": mean_duration_s,
            "std_duration_s": std_duration_s,
            "p95_duration_s": mean_duration_s + std_duration_s,
            "mean_monetary_cost": mean_money if mean_money is not None else money,
            "std_monetary_cost": std_money,
            "mean_emissions_kg": mean_co2 if mean_co2 is not None else co2,
            "std_emissions_kg": std_co2,
            "robust_score": mean_duration_s + std_duration_s,
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
