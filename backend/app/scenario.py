from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from .model_data_errors import ModelDataError

try:
    UK_TZ = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:
    UK_TZ = UTC


class ScenarioMode(str, Enum):
    NO_SHARING = "no_sharing"
    PARTIAL_SHARING = "partial_sharing"
    FULL_SHARING = "full_sharing"

    # Backward-compatible aliases.
    no_sharing = NO_SHARING
    partial_sharing = PARTIAL_SHARING
    full_sharing = FULL_SHARING


@dataclass(frozen=True)
class ScenarioRouteContext:
    corridor_geohash5: str
    hour_slot_local: int
    day_kind: str
    road_mix_bucket: str
    road_mix_vector: dict[str, float]
    vehicle_class: str
    weather_regime: str
    centroid_lat: float | None = None
    centroid_lon: float | None = None
    road_hint: str | None = None

    @property
    def context_key(self) -> str:
        return (
            f"{self.corridor_geohash5.strip().lower() or 'uk000'}|"
            f"h{max(0, min(23, int(self.hour_slot_local))):02d}|"
            f"{self.day_kind.strip().lower() or 'weekday'}|"
            f"{self.road_mix_bucket.strip().lower() or 'mixed'}|"
            f"{self.vehicle_class.strip().lower() or 'rigid_hgv'}|"
            f"{self.weather_regime.strip().lower() or 'clear'}"
        )


class ScenarioPolicy(BaseModel):
    duration_multiplier: float = Field(..., gt=0.0)
    incident_rate_multiplier: float = Field(..., gt=0.0)
    incident_delay_multiplier: float = Field(..., gt=0.0)
    fuel_consumption_multiplier: float = Field(..., gt=0.0)
    emissions_multiplier: float = Field(..., gt=0.0)
    stochastic_sigma_multiplier: float = Field(..., gt=0.0)
    source: str
    version: str
    as_of_utc: str | None = None
    calibration_basis: str = "empirical"
    context_key: str = "uk_default|mixed|rigid_hgv|weekday|clear"
    live_source_set: dict[str, str] = Field(default_factory=dict)
    live_as_of_utc: str | None = None
    live_coverage: dict[str, float] = Field(default_factory=dict)
    live_traffic_pressure: float = 1.0
    live_incident_pressure: float = 1.0
    live_weather_pressure: float = 1.0
    scenario_edge_scaling_version: str = "v3_live_transform"
    mode_observation_source: str | None = None
    mode_projection_ratio: float | None = None


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def _geohash5(lat: float | None, lon: float | None) -> str:
    if lat is None or lon is None:
        return "uk000"
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return "uk000"
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


def _road_mix_vector(road_class_counts: dict[str, int] | None) -> dict[str, float]:
    if not road_class_counts:
        return {"motorway": 0.0, "trunk": 0.0, "primary": 0.0, "secondary": 0.0, "local": 1.0}
    total = max(sum(max(0, int(v)) for v in road_class_counts.values()), 1)
    motorway_share = max(0, int(road_class_counts.get("motorway", 0))) / total
    trunk_share = max(0, int(road_class_counts.get("trunk", 0))) / total
    primary_share = max(0, int(road_class_counts.get("primary", 0))) / total
    secondary_share = max(0, int(road_class_counts.get("secondary", 0))) / total
    local_share = max(
        0,
        int(road_class_counts.get("local", 0))
        + int(road_class_counts.get("residential", 0))
        + int(road_class_counts.get("unclassified", 0)),
    ) / total
    vec = {
        "motorway": motorway_share,
        "trunk": trunk_share,
        "primary": primary_share,
        "secondary": secondary_share,
        "local": local_share,
    }
    norm = max(sum(vec.values()), 1e-9)
    return {key: float(value) / norm for key, value in vec.items()}


def _dominant_road_mix_bucket(road_mix_vector: dict[str, float]) -> str:
    motorway_share = float(road_mix_vector.get("motorway", 0.0))
    trunk_share = float(road_mix_vector.get("trunk", 0.0))
    local_share = float(road_mix_vector.get("local", 0.0))
    primary_share = float(road_mix_vector.get("primary", 0.0))
    secondary_share = float(road_mix_vector.get("secondary", 0.0))
    if motorway_share >= 0.55:
        return "motorway_heavy"
    if trunk_share >= 0.45:
        return "trunk_heavy"
    if local_share >= 0.50:
        return "urban_local_heavy"
    if (primary_share + secondary_share) >= 0.60:
        return "arterial_heavy"
    return "mixed"


def _day_kind_uk(departure_time_utc: datetime | None) -> str:
    if departure_time_utc is None:
        return "weekday"
    aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
        tzinfo=UTC
    )
    local = aware.astimezone(UK_TZ)
    if local.weekday() >= 5:
        return "weekend"
    return "weekday"


def _hour_slot_local(departure_time_utc: datetime | None) -> int:
    if departure_time_utc is None:
        return 12
    aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
        tzinfo=UTC
    )
    return int(aware.astimezone(UK_TZ).hour)


def build_scenario_route_context(
    *,
    route_points: list[tuple[float, float]] | None,
    road_class_counts: dict[str, int] | None,
    vehicle_class: str,
    departure_time_utc: datetime | None,
    weather_bucket: str,
    road_hint: str | None = None,
) -> ScenarioRouteContext:
    centroid_lat: float | None = None
    centroid_lon: float | None = None
    if route_points:
        centroid_lat = sum(lat for lat, _ in route_points) / len(route_points)
        centroid_lon = sum(lon for _, lon in route_points) / len(route_points)
    road_vector = _road_mix_vector(road_class_counts)
    road_bucket = _dominant_road_mix_bucket(road_vector)
    return ScenarioRouteContext(
        corridor_geohash5=_geohash5(centroid_lat, centroid_lon),
        hour_slot_local=_hour_slot_local(departure_time_utc),
        day_kind=_day_kind_uk(departure_time_utc),
        road_mix_bucket=road_bucket,
        road_mix_vector=road_vector,
        vehicle_class=(vehicle_class or "rigid_hgv"),
        weather_regime=(weather_bucket or "clear"),
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        road_hint=road_hint,
    )


def _adjust_multiplier(
    *,
    base: float,
    pressure: float,
    sensitivity: float,
    gain: float,
    low: float,
    high: float,
) -> float:
    adjusted = base * (1.0 + ((pressure - 1.0) * gain * sensitivity))
    return _clamp(adjusted, low, high)


def _profile_context_distance(
    *,
    context: ScenarioRouteContext,
    context_key: str,
    context_profile: object,
    weights: dict[str, float] | None = None,
) -> float:
    parts = [p.strip().lower() for p in str(context_key or "").split("|") if p.strip()]
    candidate_geo = getattr(context_profile, "corridor_geohash5", None) or getattr(
        context_profile, "corridor_bucket", None
    )
    if not isinstance(candidate_geo, str) or not candidate_geo:
        candidate_geo = parts[0] if parts else "uk000"
    candidate_hour = getattr(context_profile, "hour_slot_local", None)
    if candidate_hour is None and len(parts) >= 2 and parts[1].startswith("h"):
        try:
            candidate_hour = int(parts[1][1:])
        except ValueError:
            candidate_hour = 12
    if candidate_hour is None:
        candidate_hour = 12
    candidate_day = getattr(context_profile, "day_kind", None) or (parts[2] if len(parts) >= 3 else "weekday")
    candidate_road = getattr(context_profile, "road_mix_bucket", None) or (parts[3] if len(parts) >= 4 else "mixed")
    candidate_vehicle = getattr(context_profile, "vehicle_class", None) or (parts[4] if len(parts) >= 5 else "rigid_hgv")
    candidate_weather = getattr(context_profile, "weather_regime", None) or getattr(
        context_profile, "weather_bucket", None
    ) or (parts[5] if len(parts) >= 6 else "clear")
    candidate_road_mix_vector = getattr(context_profile, "road_mix_vector", None)
    if not isinstance(candidate_road_mix_vector, dict):
        candidate_road_mix_vector = {candidate_road: 1.0}

    geo = context.corridor_geohash5.strip().lower()
    cand_geo = str(candidate_geo).strip().lower()
    common_prefix = 0
    for idx in range(min(len(geo), len(cand_geo))):
        if geo[idx] != cand_geo[idx]:
            break
        common_prefix += 1
    geo_distance = 1.0 - (common_prefix / 5.0)
    hour_distance = abs(int(context.hour_slot_local) - int(candidate_hour)) / 24.0
    day_penalty = 0.0 if context.day_kind.strip().lower() == str(candidate_day).strip().lower() else 0.4
    weather_penalty = 0.0 if context.weather_regime.strip().lower() == str(candidate_weather).strip().lower() else 0.35
    vehicle_penalty = 0.0 if context.vehicle_class.strip().lower() == str(candidate_vehicle).strip().lower() else 0.5
    road_penalty = 0.0 if context.road_mix_bucket.strip().lower() == str(candidate_road).strip().lower() else 0.2
    road_mix_distance = 0.0
    for key in ("motorway", "trunk", "primary", "secondary", "local"):
        road_mix_distance += abs(float(context.road_mix_vector.get(key, 0.0)) - float(candidate_road_mix_vector.get(key, 0.0)))
    road_mix_distance *= 0.25
    weight_map = weights or {
        "geo_distance": 0.34,
        "hour_distance": 0.12,
        "day_penalty": 0.12,
        "weather_penalty": 0.12,
        "road_penalty": 0.16,
        "vehicle_penalty": 0.10,
        "road_mix_distance": 0.04,
    }
    total_weight = sum(max(0.0, float(value)) for value in weight_map.values())
    if total_weight <= 0.0:
        total_weight = 1.0
        weight_map = {
            "geo_distance": 0.34,
            "hour_distance": 0.12,
            "day_penalty": 0.12,
            "weather_penalty": 0.12,
            "road_penalty": 0.16,
            "vehicle_penalty": 0.10,
            "road_mix_distance": 0.04,
        }
    norm = {k: max(0.0, float(v)) / total_weight for k, v in weight_map.items()}
    return (
        (float(norm.get("geo_distance", 0.0)) * geo_distance)
        + (float(norm.get("hour_distance", 0.0)) * hour_distance)
        + (float(norm.get("day_penalty", 0.0)) * day_penalty)
        + (float(norm.get("weather_penalty", 0.0)) * weather_penalty)
        + (float(norm.get("road_penalty", 0.0)) * road_penalty)
        + (float(norm.get("vehicle_penalty", 0.0)) * vehicle_penalty)
        + (float(norm.get("road_mix_distance", 0.0)) * road_mix_distance)
    )


def _context_similarity_config(transform_params: dict[str, object] | None) -> tuple[dict[str, float], float]:
    default_weights = {
        "geo_distance": 0.34,
        "hour_distance": 0.12,
        "day_penalty": 0.12,
        "weather_penalty": 0.12,
        "road_penalty": 0.16,
        "vehicle_penalty": 0.10,
        "road_mix_distance": 0.04,
    }
    default_max_distance = 1.25
    if not isinstance(transform_params, dict):
        return default_weights, default_max_distance
    row = transform_params.get("context_similarity")
    if not isinstance(row, dict):
        return default_weights, default_max_distance
    weights_raw = row.get("weights")
    if not isinstance(weights_raw, dict):
        weights_raw = {}
    parsed_weights: dict[str, float] = {}
    for key, default_value in default_weights.items():
        try:
            value = float(weights_raw.get(key, default_value))
        except (TypeError, ValueError):
            value = float(default_value)
        parsed_weights[key] = max(0.0, float(value))
    total = sum(parsed_weights.values())
    if total <= 0.0:
        parsed_weights = dict(default_weights)
        total = sum(parsed_weights.values())
    normalized_weights = {key: float(value) / max(total, 1e-9) for key, value in parsed_weights.items()}
    try:
        max_distance = float(row.get("max_distance", default_max_distance))
    except (TypeError, ValueError):
        max_distance = default_max_distance
    max_distance = max(0.05, min(10.0, float(max_distance)))
    return normalized_weights, max_distance


def resolve_scenario_profile(
    mode: ScenarioMode,
    *,
    context: ScenarioRouteContext | None = None,
) -> ScenarioPolicy:
    # Local import avoids module-cycle sensitivity during startup.
    from .calibration_loader import load_live_scenario_context, load_scenario_profiles

    payload = load_scenario_profiles()
    key = mode.value
    transform = payload.transform_params if isinstance(payload.transform_params, dict) else {}
    context_similarity_weights, context_similarity_max_distance = _context_similarity_config(transform)

    selected = payload.profiles.get(key)
    selected_context_key = "global"
    selected_context_profile: object | None = None
    if context is not None:
        if not payload.contexts:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario profile payload must include contextual profiles in strict runtime.",
            )
        exact_context = payload.contexts.get(context.context_key)
        if exact_context is not None and exact_context.profiles.get(key) is not None:
            selected = exact_context.profiles[key]
            selected_context_key = context.context_key
            selected_context_profile = exact_context
        else:
            best_key = ""
            best_profile = None
            best_context_profile = None
            best_distance = float("inf")
            for ctx_key, ctx_profile in payload.contexts.items():
                mode_profile = ctx_profile.profiles.get(key)
                if mode_profile is None:
                    continue
                distance = _profile_context_distance(
                    context=context,
                    context_key=ctx_key,
                    context_profile=ctx_profile,
                    weights=context_similarity_weights,
                )
                if distance < best_distance:
                    best_distance = distance
                    best_key = ctx_key
                    best_profile = mode_profile
                    best_context_profile = ctx_profile
            if best_profile is None:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=f"Scenario profile payload is missing required mode '{key}'.",
                )
            if best_distance > (float(context_similarity_max_distance) + 1e-9):
                raise ModelDataError(
                    reason_code="scenario_profile_unavailable",
                    message=(
                        "No scenario context match satisfies strict context-similarity distance bound "
                        f"(best_distance={best_distance:.4f}, max_distance={float(context_similarity_max_distance):.4f})."
                    ),
                )
            selected = best_profile
            selected_context_key = best_key
            selected_context_profile = best_context_profile
    if selected is None:
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message=f"Scenario profile payload is missing required mode '{key}'.",
        )

    live_source_set: dict[str, str] = {}
    live_as_of_utc: str | None = None
    live_coverage: dict[str, float] = {}
    traffic_pressure = 1.0
    incident_pressure = 1.0
    weather_pressure = 1.0
    if context is not None:
        transform_params_json: str | None = None
        if isinstance(payload.transform_params, dict):
            transform_params_json = json.dumps(
                payload.transform_params,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        live = load_live_scenario_context(
            corridor_bucket=context.corridor_geohash5,
            road_mix_bucket=context.road_mix_bucket,
            vehicle_class=context.vehicle_class,
            day_kind=context.day_kind,
            hour_slot_local=context.hour_slot_local,
            weather_bucket=context.weather_regime,
            centroid_lat=context.centroid_lat,
            centroid_lon=context.centroid_lon,
            road_hint=context.road_hint,
            transform_params_json=transform_params_json,
        )
        live_source_set = dict(live.source_set)
        live_as_of_utc = live.as_of_utc
        live_coverage = dict(live.coverage)
        traffic_pressure = float(live.traffic_pressure)
        incident_pressure = float(live.incident_pressure)
        weather_pressure = float(live.weather_pressure)

    mode_scale_map = transform.get("mode_effect_scale", {})
    if not isinstance(mode_scale_map, dict) or mode.value not in mode_scale_map:
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message=f"Scenario transform mode_effect_scale missing mode '{mode.value}' in strict runtime.",
        )
    mode_scale = float(mode_scale_map.get(mode.value, 1.0))
    policy_adjustment = transform.get("policy_adjustment", {})
    if not isinstance(policy_adjustment, dict):
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Scenario transform policy_adjustment is missing in strict runtime.",
        )
    pressures = {
        "traffic_pressure": traffic_pressure,
        "incident_pressure": incident_pressure,
        "weather_pressure": weather_pressure,
    }

    def _field_multiplier(field: str, base: float) -> float:
        row = policy_adjustment.get(field, {})
        if not isinstance(row, dict):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario transform policy_adjustment missing field '{field}' in strict runtime.",
            )
        weights = row.get("weights", {})
        if not isinstance(weights, dict):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario transform policy_adjustment.{field}.weights must be an object in strict runtime.",
            )
        normalized_weights = {
            pressure_key: max(0.0, float(weights.get(pressure_key, 0.0)))
            for pressure_key in pressures
        }
        weight_total = sum(normalized_weights.values())
        if weight_total <= 0.0:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario transform policy_adjustment.{field}.weights must be non-empty in strict runtime.",
            )
        blended_pressure = 0.0
        for pressure_key, pressure_value in pressures.items():
            blended_pressure += float(pressure_value) * (float(normalized_weights.get(pressure_key, 0.0)) / weight_total)
        if blended_pressure <= 0.0:
            blended_pressure = 1.0
        if "gain" not in row or "min" not in row or "max" not in row:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario transform policy_adjustment.{field} must include gain/min/max in strict runtime.",
            )
        gain = float(row.get("gain", 0.35))
        low = float(row.get("min", 0.50))
        high = float(row.get("max", 3.20))
        adjusted = _adjust_multiplier(
            base=base,
            pressure=blended_pressure,
            sensitivity=mode_scale,
            gain=gain,
            low=low,
            high=high,
        )
        if mode == ScenarioMode.FULL_SHARING:
            adjusted = min(adjusted, 1.0)
        return adjusted

    duration_multiplier = _field_multiplier(
        "duration_multiplier",
        float(selected.duration_multiplier),
    )
    incident_rate_multiplier = _field_multiplier(
        "incident_rate_multiplier",
        float(selected.incident_rate_multiplier),
    )
    incident_delay_multiplier = _field_multiplier(
        "incident_delay_multiplier",
        float(selected.incident_delay_multiplier),
    )
    fuel_multiplier = _field_multiplier(
        "fuel_consumption_multiplier",
        float(selected.fuel_consumption_multiplier),
    )
    emissions_multiplier = _field_multiplier(
        "emissions_multiplier",
        float(selected.emissions_multiplier),
    )
    sigma_multiplier = _field_multiplier(
        "stochastic_sigma_multiplier",
        float(selected.stochastic_sigma_multiplier),
    )

    return ScenarioPolicy(
        duration_multiplier=duration_multiplier,
        incident_rate_multiplier=incident_rate_multiplier,
        incident_delay_multiplier=incident_delay_multiplier,
        fuel_consumption_multiplier=fuel_multiplier,
        emissions_multiplier=emissions_multiplier,
        stochastic_sigma_multiplier=sigma_multiplier,
        source=payload.source,
        version=payload.version,
        as_of_utc=payload.as_of_utc,
        calibration_basis=payload.calibration_basis,
        context_key=selected_context_key,
        live_source_set=live_source_set,
        live_as_of_utc=live_as_of_utc,
        live_coverage=live_coverage,
        live_traffic_pressure=round(float(traffic_pressure), 6),
        live_incident_pressure=round(float(incident_pressure), 6),
        live_weather_pressure=round(float(weather_pressure), 6),
        scenario_edge_scaling_version=str(transform.get("scenario_edge_scaling_version", "v3_live_transform")),
        mode_observation_source=(
            str(getattr(selected_context_profile, "mode_observation_source", "")).strip() or None
        ),
        mode_projection_ratio=(
            float(getattr(selected_context_profile, "mode_projection_ratio", 0.0))
            if getattr(selected_context_profile, "mode_projection_ratio", None) is not None
            else None
        ),
    )


def scenario_duration_multiplier(
    mode: ScenarioMode,
    *,
    context: ScenarioRouteContext | None = None,
) -> float:
    return float(resolve_scenario_profile(mode, context=context).duration_multiplier)


def scenario_sigma_multiplier(
    mode: ScenarioMode,
    *,
    context: ScenarioRouteContext | None = None,
) -> float:
    return float(resolve_scenario_profile(mode, context=context).stochastic_sigma_multiplier)


def apply_scenario_duration(
    duration_s: float,
    mode: ScenarioMode,
    *,
    context: ScenarioRouteContext | None = None,
) -> float:
    return float(duration_s) * scenario_duration_multiplier(mode, context=context)
