from __future__ import annotations

import math
from typing import Any

from app.main import build_option
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
    )
    no_share = build_option(
        base_route,
        option_id="no_sharing",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
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
    )

    assert math.isclose(option.metrics.distance_km, 10.0, rel_tol=0.0, abs_tol=0.001)
    assert math.isclose(option.metrics.duration_s, 1_000.0, rel_tol=0.0, abs_tol=0.01)
