from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from .calibration_loader import load_fuel_consumption_calibration, load_fuel_price_snapshot
from .models import EmissionsContext
from .vehicles import VehicleProfile


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


FUEL_EMISSIONS_KG_PER_L = {
    "diesel": 2.68,
    "petrol": 2.31,
    "lng": 1.51,
}

@dataclass(frozen=True)
class IceConsumptionParams:
    baseline_l_per_100km: float
    idle_l_per_hour: float
    stop_go_gain: float
    grade_up_gain: float
    grade_down_relief: float
    temp_cold_gain: float


def _ice_params(vehicle: VehicleProfile) -> IceConsumptionParams:
    key = vehicle.id.lower()
    bucket = "default"
    if "van" in key:
        bucket = "van"
    elif "artic" in key:
        bucket = "artic_hgv"
    elif "rigid" in key:
        bucket = "rigid_hgv"
    calibration = load_fuel_consumption_calibration()
    curve = calibration.curves.get(bucket) or calibration.curves.get("default")
    if curve is None:
        curve = next(iter(calibration.curves.values()))
    return IceConsumptionParams(
        baseline_l_per_100km=curve.baseline_l_per_100km,
        idle_l_per_hour=curve.idle_l_per_hour,
        stop_go_gain=curve.stop_go_gain,
        grade_up_gain=curve.grade_up_gain,
        grade_down_relief=curve.grade_down_relief,
        temp_cold_gain=curve.temp_cold_gain,
    )


def _speed_efficiency_multiplier(speed_kmh: float) -> float:
    if speed_kmh <= 0:
        return 1.0
    calibration = load_fuel_consumption_calibration()
    curve = calibration.curves.get("default") or next(iter(calibration.curves.values()))
    delta = speed_kmh - 56.0
    return _clamp(1.0 + (curve.speed_quadratic_gain * delta * delta), 0.88, 1.70)


def _grade_multiplier(grade: float, *, ev_mode: bool) -> float:
    if grade >= 0:
        return 1.0 + min(0.9, grade * (6.0 if ev_mode else 5.0))
    regen_factor = 2.0 if ev_mode else 0.65
    return max(0.72 if ev_mode else 0.85, 1.0 + (grade * regen_factor))


def _region_bucket(lat: float | None, lon: float | None) -> str:
    if lat is None or lon is None:
        return "uk_default"
    if lat >= 56.0:
        return "scotland"
    if lon <= -3.2 and lat < 54.5:
        return "wales_west"
    if lat >= 54.2:
        return "north_england"
    if lat >= 52.6:
        return "midlands"
    if lon > -1.6 and lat <= 52.2:
        return "london_southeast"
    return "south_england"


@dataclass(frozen=True)
class SegmentEnergyResult:
    fuel_liters: float
    energy_kwh: float
    emissions_kg: float
    fuel_cost_gbp: float
    fuel_cost_uncertainty_low_gbp: float
    fuel_cost_uncertainty_high_gbp: float
    emissions_uncertainty_low_kg: float
    emissions_uncertainty_high_kg: float
    source: str
    price_source: str | None = None
    price_as_of: str | None = None


def segment_energy_and_emissions(
    *,
    vehicle: VehicleProfile,
    emissions_context: EmissionsContext,
    distance_km: float,
    duration_s: float,
    grade: float,
    fuel_price_multiplier: float,
    departure_time_utc: datetime | None = None,
    route_centroid_lat: float | None = None,
    route_centroid_lon: float | None = None,
) -> SegmentEnergyResult:
    distance_km = max(0.0, float(distance_km))
    duration_s = max(0.0, float(duration_s))
    speed_kmh = (distance_km / (duration_s / 3600.0)) if duration_s > 0 else 0.0
    price_mult = max(0.0, float(fuel_price_multiplier))
    is_ev = emissions_context.fuel_type == "ev" or vehicle.powertrain == "ev"
    speed_mult = _speed_efficiency_multiplier(speed_kmh)
    grade_mult = _grade_multiplier(grade, ev_mode=is_ev)
    load_factor = _clamp(0.82 + (vehicle.mass_tonnes / 60.0), 0.85, 1.45)

    if is_ev:
        base_kwh_per_km = (
            float(vehicle.ev_kwh_per_km)
            if vehicle.ev_kwh_per_km is not None and vehicle.ev_kwh_per_km > 0
            else 1.25
        )
        temp_mult = _clamp(1.0 + (abs(emissions_context.ambient_temp_c - 18.0) * 0.008), 1.0, 1.35)
        accessory_kw = 0.35 + max(0.0, 16.0 - emissions_context.ambient_temp_c) * 0.015
        idle_kwh = (duration_s / 3600.0) * accessory_kw
        energy_kwh = (distance_km * base_kwh_per_km * speed_mult * grade_mult * temp_mult) + idle_kwh
        grid_kg_per_kwh = (
            float(vehicle.grid_co2_kg_per_kwh)
            if vehicle.grid_co2_kg_per_kwh is not None and vehicle.grid_co2_kg_per_kwh >= 0
            else 0.20
        )
        emissions_kg = energy_kwh * grid_kg_per_kwh
        fuel_snapshot = load_fuel_price_snapshot(as_of_utc=departure_time_utc)
        region = _region_bucket(route_centroid_lat, route_centroid_lon)
        region_mult = fuel_snapshot.regional_multipliers.get(region, fuel_snapshot.regional_multipliers.get("uk_default", 1.0))
        fuel_cost = energy_kwh * fuel_snapshot.grid_price_gbp_per_kwh * region_mult * price_mult
        uncertainty_ratio = _clamp(
            0.04 + (abs(grade) * 0.10) + (max(0.0, abs(speed_kmh - 48.0) / 420.0)),
            0.03,
            0.28,
        )
        emissions_low = max(0.0, emissions_kg * (1.0 - uncertainty_ratio))
        emissions_high = max(emissions_low, emissions_kg * (1.0 + uncertainty_ratio))
        fuel_low = max(0.0, fuel_cost * (1.0 - uncertainty_ratio))
        fuel_high = max(fuel_low, fuel_cost * (1.0 + uncertainty_ratio))
        return SegmentEnergyResult(
            fuel_liters=0.0,
            energy_kwh=max(0.0, energy_kwh),
            emissions_kg=max(0.0, emissions_kg),
            fuel_cost_gbp=max(0.0, fuel_cost),
            fuel_cost_uncertainty_low_gbp=fuel_low,
            fuel_cost_uncertainty_high_gbp=fuel_high,
            emissions_uncertainty_low_kg=emissions_low,
            emissions_uncertainty_high_kg=emissions_high,
            source="ev_energy_v2",
            price_source=fuel_snapshot.source,
            price_as_of=fuel_snapshot.as_of,
        )

    params = _ice_params(vehicle)
    liters_per_100km = params.baseline_l_per_100km
    fuel_snapshot = load_fuel_price_snapshot(as_of_utc=departure_time_utc)
    fuel_prices = fuel_snapshot.prices_gbp_per_l
    fuel_type = emissions_context.fuel_type if emissions_context.fuel_type in fuel_prices else "diesel"
    euro_class_mult = {"euro4": 1.08, "euro5": 1.04, "euro6": 1.0}.get(emissions_context.euro_class, 1.0)
    temp_mult = _clamp(
        1.0 + max(0.0, 14.0 - emissions_context.ambient_temp_c) * params.temp_cold_gain,
        1.0,
        1.30,
    )
    idle_liters = (duration_s / 3600.0) * max(0.5, params.idle_l_per_hour + (vehicle.mass_tonnes * 0.018))
    stop_go_factor = _clamp(1.0 + max(0.0, (52.0 - speed_kmh)) * params.stop_go_gain, 1.0, 1.30)
    if grade >= 0.0:
        grade_profile_mult = 1.0 + min(0.95, grade * params.grade_up_gain)
    else:
        grade_profile_mult = max(0.80, 1.0 + (grade * params.grade_down_relief))
    liters = (
        ((distance_km * liters_per_100km) / 100.0)
        * speed_mult
        * grade_mult
        * grade_profile_mult
        * euro_class_mult
        * temp_mult
        * load_factor
        * stop_go_factor
    )
    liters += idle_liters * 0.11  # bounded idle share to avoid double-counting movement energy
    fuel_price = fuel_prices[fuel_type]
    region = _region_bucket(route_centroid_lat, route_centroid_lon)
    region_mult = fuel_snapshot.regional_multipliers.get(region, fuel_snapshot.regional_multipliers.get("uk_default", 1.0))
    fuel_cost = liters * fuel_price * region_mult * price_mult
    emissions_factor = FUEL_EMISSIONS_KG_PER_L[fuel_type]
    emissions_kg = liters * emissions_factor
    uncertainty_ratio = _clamp(
        0.06
        + (abs(grade) * 0.11)
        + (max(0.0, abs(speed_kmh - 52.0) / 360.0))
        + (max(0.0, vehicle.mass_tonnes - 16.0) / 220.0),
        0.04,
        0.33,
    )
    emissions_low = max(0.0, emissions_kg * (1.0 - uncertainty_ratio))
    emissions_high = max(emissions_low, emissions_kg * (1.0 + uncertainty_ratio))
    fuel_low = max(0.0, fuel_cost * (1.0 - uncertainty_ratio))
    fuel_high = max(fuel_low, fuel_cost * (1.0 + uncertainty_ratio))
    return SegmentEnergyResult(
        fuel_liters=max(0.0, liters),
        energy_kwh=0.0,
        emissions_kg=max(0.0, emissions_kg),
        fuel_cost_gbp=max(0.0, fuel_cost),
        fuel_cost_uncertainty_low_gbp=fuel_low,
        fuel_cost_uncertainty_high_gbp=fuel_high,
        emissions_uncertainty_low_kg=emissions_low,
        emissions_uncertainty_high_kg=emissions_high,
        source="ice_energy_v2",
        price_source=fuel_snapshot.source,
        price_as_of=fuel_snapshot.as_of,
    )
