from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode


def _route(*, distance_m: float, duration_s: float) -> dict[str, Any]:
    return {
        "distance": distance_m,
        "duration": duration_s,
        "geometry": {
            "type": "LineString",
            "coordinates": [[-1.8904, 52.4862], [-1.2, 52.0], [-0.1276, 51.5072]],
        },
        "legs": [
            {
                "annotation": {
                    "distance": [distance_m / 2.0, distance_m / 2.0],
                    "duration": [duration_s / 2.0, duration_s / 2.0],
                }
            }
        ],
    }


def test_counterfactuals_are_present_and_consistent() -> None:
    option = build_option(
        _route(distance_m=60_000.0, duration_s=3_600.0),
        option_id="route_0",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.PARTIAL_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="rolling",
        departure_time_utc=datetime(2026, 2, 12, 8, 30, tzinfo=UTC),
    )

    entries = {str(item["id"]): item for item in option.counterfactuals}
    assert {"fuel_plus_10pct", "carbon_price_plus_0_10", "departure_plus_2h", "scenario_shift"} <= set(entries)

    fuel = entries["fuel_plus_10pct"]
    carbon = entries["carbon_price_plus_0_10"]
    depart = entries["departure_plus_2h"]

    assert float(fuel["delta"]) > 0
    assert float(carbon["delta"]) > 0
    assert str(depart["metric"]) == "duration_s"

    expected_carbon_delta = option.metrics.emissions_kg * 0.10
    assert abs(float(carbon["delta"]) - expected_carbon_delta) < 0.01
