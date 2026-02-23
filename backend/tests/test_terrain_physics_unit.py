from __future__ import annotations

from types import SimpleNamespace

from app.terrain_physics import (
    VehicleTerrainParams,
    params_for_vehicle,
    route_emissions_multiplier,
    segment_duration_multiplier,
)


def test_params_for_vehicle_respects_profile_override() -> None:
    vehicle = SimpleNamespace(
        terrain_params=SimpleNamespace(
            mass_kg=12_000.0,
            c_rr=0.009,
            drag_area_m2=5.8,
            drivetrain_efficiency=0.9,
            regen_efficiency=0.3,
        )
    )

    params = params_for_vehicle(vehicle)
    assert params.mass_kg == 12_000.0
    assert params.drag_area_m2 == 5.8
    assert params.drivetrain_efficiency == 0.9


def test_segment_duration_multiplier_is_profile_and_grade_sensitive() -> None:
    params = VehicleTerrainParams(
        mass_kg=26_000.0,
        c_rr=0.0082,
        drag_area_m2=7.3,
        drivetrain_efficiency=0.88,
        regen_efficiency=0.14,
    )

    flat = segment_duration_multiplier(grade=0.04, speed_kmh=70.0, terrain_profile="flat", params=params)
    rolling = segment_duration_multiplier(grade=0.04, speed_kmh=70.0, terrain_profile="rolling", params=params)
    hilly = segment_duration_multiplier(grade=0.04, speed_kmh=70.0, terrain_profile="hilly", params=params)

    assert flat == 1.0
    assert rolling >= 1.0
    assert hilly >= rolling


def test_route_emissions_multiplier_never_below_one_for_non_flat_profiles() -> None:
    params = params_for_vehicle("rigid_hgv")

    rolling = route_emissions_multiplier(
        mean_grade_up=0.015,
        mean_grade_down=0.010,
        terrain_profile="rolling",
        params=params,
    )
    hilly = route_emissions_multiplier(
        mean_grade_up=0.03,
        mean_grade_down=0.005,
        terrain_profile="hilly",
        params=params,
    )

    assert rolling >= 1.0
    assert hilly >= rolling
