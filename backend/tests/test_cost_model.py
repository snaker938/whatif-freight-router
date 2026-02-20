from __future__ import annotations

import math
from typing import Any

import pytest

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
    with pytest.raises(ValueError, match="tariff resolution"):
        build_option(
            toll_route,
            option_id="with_toll",
            vehicle_type="rigid_hgv",
            scenario_mode=ScenarioMode.FULL_SHARING,
            cost_toggles=toggles,
        )

    assert no_toll.metrics.monetary_cost > 0.0


def test_toll_toggle_detects_toll_classes_from_osrm_steps() -> None:
    no_toll_route = _route(distance_m=20_000.0, duration_s=1_400.0)
    toll_class_route = _route(distance_m=20_000.0, duration_s=1_400.0)
    toll_class_route["legs"] = [
        {
            "annotation": {
                "distance": [10_000.0, 10_000.0],
                "duration": [700.0, 700.0],
            },
            "steps": [
                {"distance": 8_000.0, "classes": ["motorway"]},
                {"distance": 12_000.0, "classes": ["toll"]},
            ],
        }
    ]

    toggles = CostToggles(use_tolls=True, toll_cost_per_km=0.50)
    no_toll = build_option(
        no_toll_route,
        option_id="no_toll_step_classes",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )
    tolled = build_option(
        toll_class_route,
        option_id="with_toll_step_classes",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )

    assert no_toll.metrics.monetary_cost > 0.0
    assert sum(float(row.get("toll_cost", 0.0)) for row in tolled.segment_breakdown) > 0.0
    assert bool((tolled.toll_metadata or {}).get("tariff_rule_ids"))
    assert tolled.metrics.monetary_cost > no_toll.metrics.monetary_cost


def test_toll_toggle_detects_toll_classes_from_intersections() -> None:
    no_toll_route = _route(distance_m=20_000.0, duration_s=1_400.0)
    toll_intersection_route = _route(distance_m=20_000.0, duration_s=1_400.0)
    toll_intersection_route["legs"] = [
        {
            "annotation": {
                "distance": [10_000.0, 10_000.0],
                "duration": [700.0, 700.0],
            },
            "steps": [
                {
                    "distance": 20_000.0,
                    "intersections": [{"classes": ["toll"]}],
                }
            ],
        }
    ]

    toggles = CostToggles(use_tolls=True, toll_cost_per_km=0.50)
    no_toll = build_option(
        no_toll_route,
        option_id="no_toll_intersections",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )
    tolled = build_option(
        toll_intersection_route,
        option_id="with_toll_intersections",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toggles,
    )

    assert no_toll.metrics.monetary_cost > 0.0
    assert sum(float(row.get("toll_cost", 0.0)) for row in tolled.segment_breakdown) > 0.0
    assert bool((tolled.toll_metadata or {}).get("tariff_rule_ids"))
    assert tolled.metrics.monetary_cost > no_toll.metrics.monetary_cost


def test_toll_free_candidate_meta_overrides_legacy_toll_flag() -> None:
    route = _route(distance_m=16_000.0, duration_s=1_200.0)
    route["contains_toll"] = True

    toll_toggles = CostToggles(use_tolls=True, toll_cost_per_km=0.60)
    with pytest.raises(ValueError, match="tariff resolution"):
        build_option(
            route,
            option_id="legacy_tolled",
            vehicle_type="rigid_hgv",
            scenario_mode=ScenarioMode.FULL_SHARING,
            cost_toggles=toll_toggles,
        )

    toll_free_route = _route(distance_m=16_000.0, duration_s=1_200.0)
    toll_free_route["contains_toll"] = True
    toll_free_route["_candidate_meta"] = {
        "source_labels": ["exclude:toll"],
        "seen_by_exclude_toll": True,
        "seen_by_non_exclude_toll": False,
        "toll_exclusion_available": True,
    }
    toll_free = build_option(
        toll_free_route,
        option_id="meta_toll_free",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=toll_toggles,
    )

    assert toll_free.metrics.monetary_cost > 0.0


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


def test_segment_breakdown_includes_cost_decomposition_keys() -> None:
    route = _route(distance_m=25_000.0, duration_s=1_600.0)
    route["legs"] = [
        {
            "annotation": {
                "distance": [12_000.0, 13_000.0],
                "duration": [760.0, 840.0],
            },
            "steps": [
                {"distance": 12_000.0, "classes": ["primary"]},
                {"distance": 13_000.0, "classes": ["motorway"]},
            ],
        }
    ]

    option = build_option(
        route,
        option_id="decomp",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(use_tolls=True, toll_cost_per_km=0.55, carbon_price_per_kg=0.12),
    )
    assert option.segment_breakdown
    sample = option.segment_breakdown[0]
    assert "toll_cost" in sample
    assert "fuel_cost" in sample
    assert "carbon_cost" in sample
    assert "energy_kwh" in sample
