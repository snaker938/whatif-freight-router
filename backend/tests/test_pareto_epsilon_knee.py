from __future__ import annotations

from app.models import EpsilonConstraints, GeoJSONLineString, RouteMetrics, RouteOption
from app.pareto_methods import annotate_knee_scores, select_pareto_routes


def _option(route_id: str, *, duration_s: float, monetary_cost: float, emissions_kg: float) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(-1.0, 52.0), (-0.2, 51.5)]),
        metrics=RouteMetrics(
            distance_km=10.0,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            avg_speed_kmh=42.0,
        ),
    )


def test_epsilon_constraint_filters_candidates() -> None:
    options = [
        _option("r_fast", duration_s=1000, monetary_cost=220, emissions_kg=90),
        _option("r_bal", duration_s=1200, monetary_cost=180, emissions_kg=70),
        _option("r_clean", duration_s=1500, monetary_cost=160, emissions_kg=55),
    ]
    selected = select_pareto_routes(
        options,
        max_alternatives=5,
        pareto_method="epsilon_constraint",
        epsilon=EpsilonConstraints(duration_s=1600, monetary_cost=200, emissions_kg=80),
    )
    selected_ids = {route.id for route in selected}
    assert "r_fast" not in selected_ids
    assert selected_ids == {"r_bal", "r_clean"}


def test_knee_selection_is_deterministic() -> None:
    options = [
        _option("r1", duration_s=1000, monetary_cost=250, emissions_kg=95),
        _option("r2", duration_s=1200, monetary_cost=200, emissions_kg=75),
        _option("r3", duration_s=1500, monetary_cost=170, emissions_kg=55),
    ]
    first = annotate_knee_scores(options)
    second = annotate_knee_scores(options)

    first_knee = [route.id for route in first if route.is_knee]
    second_knee = [route.id for route in second if route.is_knee]
    assert first_knee == second_knee
    assert len(first_knee) == 1
    assert all(route.knee_score is not None for route in first)
