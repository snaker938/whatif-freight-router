from __future__ import annotations

import math
from typing import Any

from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode


def _route(*, distance_m: float, duration_s: float) -> dict[str, Any]:
    coords = [[-1.8904, 52.4862], [-1.2, 52.0], [-0.1276, 51.5072]]
    return {
        "distance": distance_m,
        "duration": duration_s,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [distance_m / 2.0, distance_m / 2.0],
                    "duration": [duration_s / 2.0, duration_s / 2.0],
                }
            }
        ],
    }


def test_scenario_delay_increases_duration_cost_and_emissions() -> None:
    base_route = _route(distance_m=50_000.0, duration_s=3_600.0)

    full = build_option(
        base_route,
        option_id="full",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )
    no_share = build_option(
        base_route,
        option_id="no_sharing",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
    )

    assert no_share.metrics.duration_s > full.metrics.duration_s
    assert no_share.metrics.monetary_cost > full.metrics.monetary_cost
    assert no_share.metrics.emissions_kg > full.metrics.emissions_kg


def test_build_option_uses_segment_totals_when_top_level_missing() -> None:
    route = _route(distance_m=10_000.0, duration_s=1_000.0)
    route["distance"] = 0.0
    route["duration"] = 0.0

    option = build_option(
        route,
        option_id="seg_fallback",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )

    assert math.isclose(option.metrics.distance_km, 10.0, rel_tol=0.0, abs_tol=0.001)
    assert math.isclose(option.metrics.duration_s, 1_000.0, rel_tol=0.0, abs_tol=0.01)


def test_carbon_price_toggle_increases_monetary_cost() -> None:
    route = _route(distance_m=35_000.0, duration_s=2_400.0)

    base = build_option(
        route,
        option_id="base",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )
    priced = build_option(
        route,
        option_id="priced",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(carbon_price_per_kg=0.50),
    )

    assert priced.metrics.monetary_cost > base.metrics.monetary_cost


def test_toll_toggle_only_applies_to_toll_flagged_routes() -> None:
    no_toll_route = _route(distance_m=10_000.0, duration_s=800.0)
    toll_route = _route(distance_m=10_000.0, duration_s=800.0)
    toll_route["contains_toll"] = True

    toggles = CostToggles(use_tolls=True, toll_cost_per_km=0.35)
    no_toll = build_option(
        no_toll_route,
        option_id="no_toll",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )
    with_toll = build_option(
        toll_route,
        option_id="with_toll",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )

    assert with_toll.metrics.monetary_cost > no_toll.metrics.monetary_cost


def test_fuel_multiplier_changes_monetary_ranking() -> None:
    route_fast = _route(distance_m=18_000.0, duration_s=900.0)
    route_slow = _route(distance_m=12_000.0, duration_s=1_500.0)

    base_fast = build_option(
        route_fast,
        option_id="fast",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )
    base_slow = build_option(
        route_slow,
        option_id="slow",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )
    boosted_fast = build_option(
        route_fast,
        option_id="fast_boost",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(fuel_price_multiplier=3.0),
    )
    boosted_slow = build_option(
        route_slow,
        option_id="slow_boost",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(fuel_price_multiplier=3.0),
    )

    base_gap = abs(base_fast.metrics.monetary_cost - base_slow.metrics.monetary_cost)
    boosted_gap = abs(boosted_fast.metrics.monetary_cost - boosted_slow.metrics.monetary_cost)
    assert boosted_gap > base_gap
