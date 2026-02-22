from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .calibration_loader import (
    Grid4D,
    load_fuel_consumption_calibration,
    load_fuel_consumption_surface,
    load_fuel_price_snapshot,
    load_fuel_uncertainty_surface,
)
from .model_data_errors import ModelDataError
from .models import EmissionsContext
from .settings import settings
from .vehicles import VehicleProfile


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


FUEL_EMISSIONS_KG_PER_L = {
    "diesel": 2.68,
    "petrol": 2.31,
    "lng": 1.51,
}


@dataclass(frozen=True)
class SegmentEnergyResult:
    fuel_liters: float
    fuel_liters_p10: float
    fuel_liters_p50: float
    fuel_liters_p90: float
    energy_kwh: float
    emissions_kg: float
    fuel_cost_gbp: float
    fuel_cost_p10_gbp: float
    fuel_cost_p50_gbp: float
    fuel_cost_p90_gbp: float
    fuel_cost_uncertainty_low_gbp: float
    fuel_cost_uncertainty_high_gbp: float
    emissions_uncertainty_low_kg: float
    emissions_uncertainty_high_kg: float
    source: str
    price_source: str | None = None
    price_as_of: str | None = None
    consumption_model_source: str | None = None
    consumption_model_version: str | None = None
    consumption_model_as_of_utc: str | None = None


_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def _geohash5(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    bits = [16, 8, 4, 2, 1]
    even = True
    ch = 0
    bit = 0
    out: list[str] = []
    while len(out) < 5:
        if even:
            mid = (lon_min + lon_max) / 2.0
            if lon >= mid:
                ch |= bits[bit]
                lon_min = mid
            else:
                lon_max = mid
        else:
            mid = (lat_min + lat_max) / 2.0
            if lat >= mid:
                ch |= bits[bit]
                lat_min = mid
            else:
                lat_max = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_GEOHASH_BASE32[ch])
            bit = 0
            ch = 0
    return "".join(out)


def _legacy_region_bucket(lat: float | None, lon: float | None) -> str:
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


def _regional_multiplier(
    regional_multipliers: dict[str, float],
    *,
    lat: float | None,
    lon: float | None,
) -> float:
    if not regional_multipliers:
        return 1.0
    normalized = {
        str(key).strip().lower(): float(value)
        for key, value in regional_multipliers.items()
        if str(key).strip()
    }
    geohash5 = _geohash5(lat, lon)
    candidates: list[str] = []
    if geohash5:
        candidates.extend(
            [
                geohash5,
                f"geohash:{geohash5}",
                f"geohash5:{geohash5}",
                geohash5[:4],
                f"geohash:{geohash5[:4]}",
                f"geohash4:{geohash5[:4]}",
            ]
        )
    if not bool(settings.strict_live_data_required):
        legacy_bucket = _legacy_region_bucket(lat, lon)
        candidates.append(legacy_bucket)
    candidates.append("uk_default")
    for key in candidates:
        if key in normalized:
            return max(0.6, min(1.5, float(normalized[key])))
    return 1.0


def _vehicle_class(vehicle: VehicleProfile, emissions_context: EmissionsContext) -> str:
    if emissions_context.fuel_type == "ev" or vehicle.powertrain == "ev":
        return "ev"
    explicit = str(getattr(vehicle, "fuel_surface_class", "")).strip().lower()
    if explicit in {"van", "rigid_hgv", "artic_hgv", "ev"}:
        return explicit
    profile_class = str(getattr(vehicle, "vehicle_class", "")).strip().lower()
    if profile_class in {"van", "rigid_hgv", "artic_hgv", "ev"}:
        return profile_class
    raise ModelDataError(
        reason_code="vehicle_profile_invalid",
        message=(
            "Vehicle profile is missing explicit fuel surface class mapping "
            "(fuel_surface_class/vehicle_class) required for strict fuel model."
        ),
        details={"vehicle_id": str(getattr(vehicle, "id", "unknown"))},
    )


def _derived_load_factor(vehicle: VehicleProfile, *, klass: str) -> float:
    nominal_mass = {
        "van": 3.0,
        "rigid_hgv": 14.0,
        "artic_hgv": 28.0,
        "ev": 14.0,
    }.get(klass, 14.0)
    return max(0.05, float(vehicle.mass_tonnes) / max(0.1, nominal_mass))


def _axis_bracket(axis: tuple[float, ...], value: float) -> tuple[int, int, float]:
    if value <= axis[0]:
        return 0, 0, 0.0
    last = len(axis) - 1
    if value >= axis[last]:
        return last, last, 0.0
    for idx in range(last):
        lo = axis[idx]
        hi = axis[idx + 1]
        if lo <= value <= hi:
            span = hi - lo
            if span <= 0.0:
                return idx, idx, 0.0
            return idx, idx + 1, (value - lo) / span
    return last, last, 0.0


def _interp_grid4d(
    grid: Grid4D,
    *,
    load_axis: tuple[float, ...],
    speed_axis: tuple[float, ...],
    grade_axis: tuple[float, ...],
    temp_axis: tuple[float, ...],
    load_factor: float,
    speed_kmh: float,
    grade_pct: float,
    ambient_temp_c: float,
) -> float:
    i0, i1, tx = _axis_bracket(load_axis, load_factor)
    j0, j1, ty = _axis_bracket(speed_axis, speed_kmh)
    k0, k1, tz = _axis_bracket(grade_axis, grade_pct)
    l0, l1, tw = _axis_bracket(temp_axis, ambient_temp_c)

    value = 0.0
    for i_idx, wi in ((i0, 1.0 - tx), (i1, tx)):
        for j_idx, wj in ((j0, 1.0 - ty), (j1, ty)):
            for k_idx, wk in ((k0, 1.0 - tz), (k1, tz)):
                for l_idx, wl in ((l0, 1.0 - tw), (l1, tw)):
                    weight = wi * wj * wk * wl
                    if weight <= 0.0:
                        continue
                    value += float(grid[i_idx][j_idx][k_idx][l_idx]) * weight
    return max(0.0, value)


def _interp_quantiles(
    quantiles: dict[str, Grid4D],
    *,
    load_axis: tuple[float, ...],
    speed_axis: tuple[float, ...],
    grade_axis: tuple[float, ...],
    temp_axis: tuple[float, ...],
    load_factor: float,
    speed_kmh: float,
    grade_pct: float,
    ambient_temp_c: float,
) -> tuple[float, float, float]:
    q10 = _interp_grid4d(
        quantiles["p10"],
        load_axis=load_axis,
        speed_axis=speed_axis,
        grade_axis=grade_axis,
        temp_axis=temp_axis,
        load_factor=load_factor,
        speed_kmh=speed_kmh,
        grade_pct=grade_pct,
        ambient_temp_c=ambient_temp_c,
    )
    q50 = _interp_grid4d(
        quantiles["p50"],
        load_axis=load_axis,
        speed_axis=speed_axis,
        grade_axis=grade_axis,
        temp_axis=temp_axis,
        load_factor=load_factor,
        speed_kmh=speed_kmh,
        grade_pct=grade_pct,
        ambient_temp_c=ambient_temp_c,
    )
    q90 = _interp_grid4d(
        quantiles["p90"],
        load_axis=load_axis,
        speed_axis=speed_axis,
        grade_axis=grade_axis,
        temp_axis=temp_axis,
        load_factor=load_factor,
        speed_kmh=speed_kmh,
        grade_pct=grade_pct,
        ambient_temp_c=ambient_temp_c,
    )
    # Contract invariant.
    q_mid = max(q10, q50)
    q_hi = max(q_mid, q90)
    return q10, q_mid, q_hi


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
    grade_pct = float(grade) * 100.0

    consumption_surface = load_fuel_consumption_surface()
    uncertainty_surface = load_fuel_uncertainty_surface()
    snapshot = load_fuel_price_snapshot(as_of_utc=departure_time_utc)
    # Compatibility accessor used in weather summary metadata fields.
    _ = load_fuel_consumption_calibration()

    klass = _vehicle_class(vehicle, emissions_context)
    load_factor = _derived_load_factor(vehicle, klass=klass)
    load_factor = _clamp(load_factor, consumption_surface.axes.load_factor[0], consumption_surface.axes.load_factor[-1])
    speed_value = _clamp(speed_kmh, consumption_surface.axes.speed_kmh[0], consumption_surface.axes.speed_kmh[-1])
    grade_value = _clamp(grade_pct, consumption_surface.axes.grade_pct[0], consumption_surface.axes.grade_pct[-1])
    temp_value = _clamp(
        float(emissions_context.ambient_temp_c),
        consumption_surface.axes.ambient_temp_c[0],
        consumption_surface.axes.ambient_temp_c[-1],
    )

    region_mult = _regional_multiplier(
        snapshot.regional_multipliers,
        lat=route_centroid_lat,
        lon=route_centroid_lon,
    )
    price_mult = max(0.0, float(fuel_price_multiplier))

    if klass == "ev":
        grid = consumption_surface.energy_kwh_per_100km.get("ev")
        if grid is None:
            raise ValueError("Fuel surface missing EV energy grid.")
        kwh_per_100km = _interp_grid4d(
            grid,
            load_axis=consumption_surface.axes.load_factor,
            speed_axis=consumption_surface.axes.speed_kmh,
            grade_axis=consumption_surface.axes.grade_pct,
            temp_axis=consumption_surface.axes.ambient_temp_c,
            load_factor=load_factor,
            speed_kmh=speed_value,
            grade_pct=grade_value,
            ambient_temp_c=temp_value,
        )
        q_energy = uncertainty_surface.energy_kwh_multiplier_quantiles.get("ev")
        q_cost = uncertainty_surface.fuel_cost_multiplier_quantiles.get("ev")
        if q_energy is None or q_cost is None:
            raise ValueError("Fuel uncertainty surface missing EV quantile tables.")
        e_q10_mult, e_q50_mult, e_q90_mult = _interp_quantiles(
            q_energy,
            load_axis=uncertainty_surface.axes.load_factor,
            speed_axis=uncertainty_surface.axes.speed_kmh,
            grade_axis=uncertainty_surface.axes.grade_pct,
            temp_axis=uncertainty_surface.axes.ambient_temp_c,
            load_factor=load_factor,
            speed_kmh=speed_value,
            grade_pct=grade_value,
            ambient_temp_c=temp_value,
        )
        c_q10_mult, c_q50_mult, c_q90_mult = _interp_quantiles(
            q_cost,
            load_axis=uncertainty_surface.axes.load_factor,
            speed_axis=uncertainty_surface.axes.speed_kmh,
            grade_axis=uncertainty_surface.axes.grade_pct,
            temp_axis=uncertainty_surface.axes.ambient_temp_c,
            load_factor=load_factor,
            speed_kmh=speed_value,
            grade_pct=grade_value,
            ambient_temp_c=temp_value,
        )

        energy_kwh = max(0.0, (distance_km * kwh_per_100km) / 100.0)
        energy_p10 = energy_kwh * max(0.0, e_q10_mult)
        energy_p50 = energy_kwh * max(0.0, e_q50_mult)
        energy_p90 = max(energy_p50, energy_kwh * max(0.0, e_q90_mult))

        base_cost = energy_kwh * snapshot.grid_price_gbp_per_kwh * region_mult * price_mult
        fuel_cost_p10 = max(0.0, base_cost * max(0.0, c_q10_mult))
        fuel_cost_p50 = max(fuel_cost_p10, base_cost * max(0.0, c_q50_mult))
        fuel_cost_p90 = max(fuel_cost_p50, base_cost * max(0.0, c_q90_mult))

        grid_factor = (
            float(vehicle.grid_co2_kg_per_kwh)
            if vehicle.grid_co2_kg_per_kwh is not None and vehicle.grid_co2_kg_per_kwh >= 0
            else 0.20
        )
        emissions_kg = energy_kwh * grid_factor
        emissions_low = max(0.0, energy_p10 * grid_factor)
        emissions_high = max(emissions_low, energy_p90 * grid_factor)

        return SegmentEnergyResult(
            fuel_liters=0.0,
            fuel_liters_p10=0.0,
            fuel_liters_p50=0.0,
            fuel_liters_p90=0.0,
            energy_kwh=energy_kwh,
            emissions_kg=max(0.0, emissions_kg),
            fuel_cost_gbp=max(0.0, base_cost),
            fuel_cost_p10_gbp=fuel_cost_p10,
            fuel_cost_p50_gbp=fuel_cost_p50,
            fuel_cost_p90_gbp=fuel_cost_p90,
            fuel_cost_uncertainty_low_gbp=fuel_cost_p10,
            fuel_cost_uncertainty_high_gbp=fuel_cost_p90,
            emissions_uncertainty_low_kg=emissions_low,
            emissions_uncertainty_high_kg=emissions_high,
            source=f"surface5d:{consumption_surface.version}",
            price_source=snapshot.source,
            price_as_of=snapshot.as_of,
            consumption_model_source=consumption_surface.source,
            consumption_model_version=consumption_surface.version,
            consumption_model_as_of_utc=consumption_surface.as_of_utc,
        )

    fuel_grid = consumption_surface.fuel_l_per_100km.get(klass)
    if fuel_grid is None:
        raise ValueError(f"Fuel surface missing class '{klass}'.")
    fuel_quantiles = uncertainty_surface.fuel_liters_multiplier_quantiles.get(klass)
    cost_quantiles = uncertainty_surface.fuel_cost_multiplier_quantiles.get(klass)
    if fuel_quantiles is None or cost_quantiles is None:
        raise ValueError(f"Fuel uncertainty surface missing class '{klass}'.")

    liters_per_100km = _interp_grid4d(
        fuel_grid,
        load_axis=consumption_surface.axes.load_factor,
        speed_axis=consumption_surface.axes.speed_kmh,
        grade_axis=consumption_surface.axes.grade_pct,
        temp_axis=consumption_surface.axes.ambient_temp_c,
        load_factor=load_factor,
        speed_kmh=speed_value,
        grade_pct=grade_value,
        ambient_temp_c=temp_value,
    )
    liters = max(0.0, (distance_km * liters_per_100km) / 100.0)

    l_q10_mult, l_q50_mult, l_q90_mult = _interp_quantiles(
        fuel_quantiles,
        load_axis=uncertainty_surface.axes.load_factor,
        speed_axis=uncertainty_surface.axes.speed_kmh,
        grade_axis=uncertainty_surface.axes.grade_pct,
        temp_axis=uncertainty_surface.axes.ambient_temp_c,
        load_factor=load_factor,
        speed_kmh=speed_value,
        grade_pct=grade_value,
        ambient_temp_c=temp_value,
    )
    c_q10_mult, c_q50_mult, c_q90_mult = _interp_quantiles(
        cost_quantiles,
        load_axis=uncertainty_surface.axes.load_factor,
        speed_axis=uncertainty_surface.axes.speed_kmh,
        grade_axis=uncertainty_surface.axes.grade_pct,
        temp_axis=uncertainty_surface.axes.ambient_temp_c,
        load_factor=load_factor,
        speed_kmh=speed_value,
        grade_pct=grade_value,
        ambient_temp_c=temp_value,
    )
    liters_p10 = max(0.0, liters * max(0.0, l_q10_mult))
    liters_p50 = max(liters_p10, liters * max(0.0, l_q50_mult))
    liters_p90 = max(liters_p50, liters * max(0.0, l_q90_mult))

    fuel_prices = snapshot.prices_gbp_per_l
    fuel_type = emissions_context.fuel_type if emissions_context.fuel_type in fuel_prices else "diesel"
    fuel_price = max(0.0, float(fuel_prices.get(fuel_type, fuel_prices.get("diesel", 0.0))))
    base_cost = liters * fuel_price * region_mult * price_mult
    fuel_cost_p10 = max(0.0, base_cost * max(0.0, c_q10_mult))
    fuel_cost_p50 = max(fuel_cost_p10, base_cost * max(0.0, c_q50_mult))
    fuel_cost_p90 = max(fuel_cost_p50, base_cost * max(0.0, c_q90_mult))

    emissions_factor = FUEL_EMISSIONS_KG_PER_L.get(fuel_type, FUEL_EMISSIONS_KG_PER_L["diesel"])
    emissions_kg = liters * emissions_factor
    emissions_low = max(0.0, liters_p10 * emissions_factor)
    emissions_high = max(emissions_low, liters_p90 * emissions_factor)

    return SegmentEnergyResult(
        fuel_liters=liters,
        fuel_liters_p10=liters_p10,
        fuel_liters_p50=liters_p50,
        fuel_liters_p90=liters_p90,
        energy_kwh=0.0,
        emissions_kg=max(0.0, emissions_kg),
        fuel_cost_gbp=max(0.0, base_cost),
        fuel_cost_p10_gbp=fuel_cost_p10,
        fuel_cost_p50_gbp=fuel_cost_p50,
        fuel_cost_p90_gbp=fuel_cost_p90,
        fuel_cost_uncertainty_low_gbp=fuel_cost_p10,
        fuel_cost_uncertainty_high_gbp=fuel_cost_p90,
        emissions_uncertainty_low_kg=emissions_low,
        emissions_uncertainty_high_kg=emissions_high,
        source=f"surface5d:{consumption_surface.version}",
        price_source=snapshot.source,
        price_as_of=snapshot.as_of,
        consumption_model_source=consumption_surface.source,
        consumption_model_version=consumption_surface.version,
        consumption_model_as_of_utc=consumption_surface.as_of_utc,
    )
