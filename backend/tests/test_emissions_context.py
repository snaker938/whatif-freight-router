from __future__ import annotations

import math
from typing import Any

from app.main import build_option
from app.models import CostToggles, EmissionsContext
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


def test_default_emissions_context_matches_explicit_defaults() -> None:
    route = _route(distance_m=24_000.0, duration_s=1_800.0)

    implicit = build_option(
        route,
        option_id="implicit",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
    )
    explicit = build_option(
        route,
        option_id="explicit",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
        emissions_context=EmissionsContext(),
    )

    assert implicit.metrics.energy_kwh is None
    assert explicit.metrics.energy_kwh is None
    assert math.isclose(
        implicit.metrics.emissions_kg,
        explicit.metrics.emissions_kg,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def test_ice_context_petrol_euro4_cold_increases_emissions() -> None:
    route = _route(distance_m=32_000.0, duration_s=2_100.0)

    baseline = build_option(
        route,
        option_id="baseline",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
        emissions_context=EmissionsContext(fuel_type="diesel", euro_class="euro6", ambient_temp_c=15.0),
    )
    stressed = build_option(
        route,
        option_id="stressed",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
        emissions_context=EmissionsContext(fuel_type="petrol", euro_class="euro4", ambient_temp_c=0.0),
    )

    assert stressed.metrics.energy_kwh is None
    assert stressed.metrics.emissions_kg > baseline.metrics.emissions_kg


def test_ev_mode_emits_energy_kwh_and_grid_based_co2() -> None:
    route = _route(distance_m=10_000.0, duration_s=1_000.0)

    option = build_option(
        route,
        option_id="ev_mode",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
        emissions_context=EmissionsContext(fuel_type="ev", euro_class="euro6", ambient_temp_c=20.0),
    )

    assert option.metrics.energy_kwh is not None
    assert option.metrics.energy_kwh > 0
    assert option.metrics.emissions_kg > 0
    # V2 EV pipeline uses speed/grade adjustments; CO2 intensity should still match grid factor.
    assert math.isclose(
        option.metrics.emissions_kg / option.metrics.energy_kwh,
        0.20,
        rel_tol=0.0,
        abs_tol=1e-3,
    )
