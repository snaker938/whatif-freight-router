from __future__ import annotations

from app.main import _finalize_pareto_options
from app.models import EpsilonConstraints, GeoJSONLineString, RouteMetrics, RouteOption
from app.settings import settings


def _mk_option(idx: int, *, duration_s: float, monetary_cost: float, emissions_kg: float) -> RouteOption:
    return RouteOption(
        id=f"route_{idx}",
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[(0.0, 0.0), (0.01 * idx, 0.01 * idx)],
        ),
        metrics=RouteMetrics(
            distance_km=10.0 + idx,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            avg_speed_kmh=50.0,
        ),
    )


def test_finalize_pareto_options_keeps_strict_frontier_when_backfill_disabled() -> None:
    old_enabled = settings.route_pareto_backfill_enabled
    old_min = settings.route_pareto_backfill_min_alternatives
    settings.route_pareto_backfill_enabled = False
    settings.route_pareto_backfill_min_alternatives = 6
    try:
        options = [
            _mk_option(1, duration_s=100.0, monetary_cost=100.0, emissions_kg=100.0),
            _mk_option(2, duration_s=120.0, monetary_cost=120.0, emissions_kg=120.0),
            _mk_option(3, duration_s=140.0, monetary_cost=140.0, emissions_kg=140.0),
        ]
        out = _finalize_pareto_options(
            options,
            max_alternatives=6,
            pareto_method="dominance",
            epsilon=None,
            optimization_mode="expected_value",
            risk_aversion=1.0,
        )
        assert [row.id for row in out] == ["route_1"]
    finally:
        settings.route_pareto_backfill_enabled = old_enabled
        settings.route_pareto_backfill_min_alternatives = old_min


def test_finalize_pareto_options_backfills_ranked_routes_and_respects_epsilon() -> None:
    old_enabled = settings.route_pareto_backfill_enabled
    old_min = settings.route_pareto_backfill_min_alternatives
    settings.route_pareto_backfill_enabled = True
    settings.route_pareto_backfill_min_alternatives = 4
    try:
        options = [
            _mk_option(1, duration_s=100.0, monetary_cost=100.0, emissions_kg=100.0),
            _mk_option(2, duration_s=110.0, monetary_cost=110.0, emissions_kg=110.0),
            _mk_option(3, duration_s=120.0, monetary_cost=120.0, emissions_kg=120.0),
            _mk_option(4, duration_s=130.0, monetary_cost=130.0, emissions_kg=130.0),
            _mk_option(5, duration_s=140.0, monetary_cost=140.0, emissions_kg=140.0),
        ]
        out = _finalize_pareto_options(
            options,
            max_alternatives=6,
            pareto_method="epsilon_constraint",
            epsilon=EpsilonConstraints(duration_s=130.0, monetary_cost=130.0, emissions_kg=130.0),
            optimization_mode="expected_value",
            risk_aversion=1.0,
        )
        assert [row.id for row in out] == ["route_1", "route_2", "route_3", "route_4"]
    finally:
        settings.route_pareto_backfill_enabled = old_enabled
        settings.route_pareto_backfill_min_alternatives = old_min
