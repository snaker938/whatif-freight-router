from __future__ import annotations

import math

from app.objectives_emissions import route_emissions_kg, speed_factor
from app.vehicles import VEHICLES


def test_speed_factor_reference_points() -> None:
    assert speed_factor(55.0) == 1.0
    assert speed_factor(90.0) > 1.0
    assert speed_factor(120.0) > speed_factor(90.0)


def test_route_emissions_reference_example_moderate_speed() -> None:
    # Reference example A:
    #   10 km at 40 km/h for rigid_hgv with no idle-speed penalty expected.
    vehicle = VEHICLES["rigid_hgv"]
    emissions = route_emissions_kg(
        vehicle=vehicle,
        segment_distances_m=[10_000.0],
        segment_durations_s=[900.0],
    )

    # Manual expectation from current formula:
    # factor = 1 + 0.18 * ((55 - 40) / 55)^2
    factor = 1.0 + 0.18 * (((55.0 - 40.0) / 55.0) ** 2)
    expected = vehicle.mass_tonnes * 10.0 * vehicle.emission_factor_kg_per_tkm * factor
    assert math.isclose(emissions, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_route_emissions_reference_example_low_speed_with_idle() -> None:
    # Reference example B:
    #   1 km over 1800 s (2 km/h) for rigid_hgv, including idle add-on.
    vehicle = VEHICLES["rigid_hgv"]
    emissions = route_emissions_kg(
        vehicle=vehicle,
        segment_distances_m=[1_000.0],
        segment_durations_s=[1_800.0],
    )

    factor = 1.0 + 0.18 * (((55.0 - 2.0) / 55.0) ** 2)
    moving = vehicle.mass_tonnes * 1.0 * vehicle.emission_factor_kg_per_tkm * factor
    idle = (1_800.0 / 3_600.0) * vehicle.idle_emissions_kg_per_hour
    expected = moving + idle
    assert math.isclose(emissions, expected, rel_tol=1e-9, abs_tol=1e-9)
