from __future__ import annotations

import math
from typing import Any
from dataclasses import dataclass


G_MPS2 = 9.80665
AIR_DENSITY_KG_M3 = 1.225


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


@dataclass(frozen=True)
class VehicleTerrainParams:
    mass_kg: float
    c_rr: float
    drag_area_m2: float
    drivetrain_efficiency: float
    regen_efficiency: float


def _profile_weight(terrain_profile: str) -> float:
    if terrain_profile == "flat":
        return 0.2
    if terrain_profile == "rolling":
        return 0.7
    return 1.0


def params_for_vehicle(vehicle: str | Any) -> VehicleTerrainParams:
    terrain_params = getattr(vehicle, "terrain_params", None)
    if terrain_params is not None:
        return VehicleTerrainParams(
            mass_kg=float(getattr(terrain_params, "mass_kg", 26_000.0)),
            c_rr=float(getattr(terrain_params, "c_rr", 0.0082)),
            drag_area_m2=float(getattr(terrain_params, "drag_area_m2", 7.3)),
            drivetrain_efficiency=float(getattr(terrain_params, "drivetrain_efficiency", 0.88)),
            regen_efficiency=float(getattr(terrain_params, "regen_efficiency", 0.14)),
        )
    key = str(vehicle).strip().lower()
    if key == "artic_hgv":
        return VehicleTerrainParams(
            mass_kg=38_000.0,
            c_rr=0.0075,
            drag_area_m2=8.2,
            drivetrain_efficiency=0.86,
            regen_efficiency=0.12,
        )
    if key == "van":
        return VehicleTerrainParams(
            mass_kg=3_500.0,
            c_rr=0.0095,
            drag_area_m2=3.9,
            drivetrain_efficiency=0.89,
            regen_efficiency=0.18,
        )
    if key in {"ev", "ev_hgv"}:
        return VehicleTerrainParams(
            mass_kg=18_000.0,
            c_rr=0.0085,
            drag_area_m2=6.7,
            drivetrain_efficiency=0.91,
            regen_efficiency=0.40,
        )
    return VehicleTerrainParams(
        mass_kg=26_000.0,
        c_rr=0.0082,
        drag_area_m2=7.3,
        drivetrain_efficiency=0.88,
        regen_efficiency=0.14,
    )


def segment_duration_multiplier(
    *,
    grade: float,
    speed_kmh: float,
    terrain_profile: str,
    params: VehicleTerrainParams,
) -> float:
    if terrain_profile == "flat":
        return 1.0
    v_ms = max(1.0, float(speed_kmh) / 3.6)
    rolling_force = params.mass_kg * G_MPS2 * params.c_rr
    aero_force = 0.5 * AIR_DENSITY_KG_M3 * params.drag_area_m2 * (v_ms ** 2)
    grade_force = params.mass_kg * G_MPS2 * grade
    baseline_force = max(250.0, rolling_force + aero_force)
    raw_force = rolling_force + aero_force + max(grade_force, -0.85 * baseline_force)
    uplift = raw_force / baseline_force
    if grade < 0.0:
        downhill_recovery = min(abs(grade) * params.regen_efficiency, 0.32)
        uplift = max(0.72, uplift + downhill_recovery)
    weight = _profile_weight(terrain_profile)
    # Profile weighting controls how strongly terrain profile intent applies.
    if uplift >= 1.0:
        weighted = 1.0 + ((uplift - 1.0) * weight)
    else:
        downhill_relief = 1.0 - uplift
        weighted = 1.0 - (downhill_relief * (1.0 - (0.5 * weight)))
    floor_by_profile = {"flat": 1.0, "rolling": 1.02, "hilly": 1.05}
    return _clamp(
        max(floor_by_profile.get(terrain_profile, 1.0), weighted / max(0.70, params.drivetrain_efficiency)),
        0.85,
        1.65,
    )


def route_emissions_multiplier(
    *,
    mean_grade_up: float,
    mean_grade_down: float,
    terrain_profile: str,
    params: VehicleTerrainParams,
) -> float:
    if terrain_profile == "flat":
        return 1.0
    weight = _profile_weight(terrain_profile)
    uphill_penalty = mean_grade_up * (5.3 + (params.mass_kg / 16_000.0))
    downhill_relief = mean_grade_down * (1.7 + (params.regen_efficiency * 1.5))
    mult = 1.0 + ((uphill_penalty - downhill_relief) * weight)
    return _clamp(mult, 1.0, 1.85)
