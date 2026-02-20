from __future__ import annotations

from app.models import GeoJSONLineString, RouteMetrics, RouteOption
from app.pareto_methods import select_pareto_routes


def _opt(route_id: str, duration_s: float, money: float, co2: float) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(-1.0, 52.0), (-0.5, 51.8)]),
        metrics=RouteMetrics(
            distance_km=20.0,
            duration_s=duration_s,
            monetary_cost=money,
            emissions_kg=co2,
            avg_speed_kmh=45.0,
        ),
    )


def test_strict_frontier_does_not_inject_diverse_fallback() -> None:
    # b and c are dominated by a.
    options = [
        _opt("a", 100.0, 100.0, 100.0),
        _opt("b", 110.0, 110.0, 110.0),
        _opt("c", 120.0, 130.0, 140.0),
    ]
    strict = select_pareto_routes(
        options,
        max_alternatives=5,
        pareto_method="dominance",
        epsilon=None,
        strict_frontier=True,
    )
    assert [r.id for r in strict] == ["a"]


def test_non_strict_keeps_pareto_only_without_diversity_injection() -> None:
    options = [
        _opt("a", 100.0, 100.0, 100.0),
        _opt("b", 110.0, 110.0, 110.0),
        _opt("c", 120.0, 130.0, 140.0),
    ]
    non_strict = select_pareto_routes(
        options,
        max_alternatives=5,
        pareto_method="dominance",
        epsilon=None,
        strict_frontier=False,
    )
    assert [r.id for r in non_strict] == ["a"]
