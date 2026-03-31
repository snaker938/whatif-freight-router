from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ._process_cache import ProcessGlobalCacheStore
from .models import (
    CostToggles,
    EmissionsContext,
    IncidentSimulatorConfig,
    OptimizationMode,
    ParetoMethod,
    RouteOption,
    ScenarioMode,
    StochasticConfig,
    TerrainProfile,
    WeatherImpactConfig,
)
from .settings import settings

ROUTE_OPTION_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CachedRouteOptionBuild:
    option: RouteOption
    estimated_build_ms: float = 0.0


@dataclass(frozen=True)
class CachedRouteOptionCore:
    option: RouteOption
    estimated_build_ms: float = 0.0


def _normalize_key_component(value: Any) -> Any:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalize_key_component(model_dump(mode="json"))
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalize_key_component(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_key_component(item) for item in value]
    return str(value)


def _route_geometry_signature(route: Mapping[str, Any]) -> str | None:
    geometry = route.get("geometry")
    if not isinstance(geometry, Mapping):
        return None
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return None
    valid: list[tuple[float, float]] = []
    for point in coordinates:
        if (
            isinstance(point, (list, tuple))
            and len(point) == 2
            and isinstance(point[0], (int, float))
            and isinstance(point[1], (int, float))
        ):
            valid.append((float(point[0]), float(point[1])))
    if len(valid) < 2:
        return None
    step = max(1, len(valid) // 30)
    sample = valid[::step][:40]
    parts = [f"{lon:.4f},{lat:.4f}" for lon, lat in sample]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _road_class_counts(route: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {"motorway": 0, "trunk": 0, "primary": 0, "secondary": 0, "local": 0}
    legs = route.get("legs", [])
    if not isinstance(legs, list):
        return counts
    for leg in legs:
        if not isinstance(leg, Mapping):
            continue
        steps = leg.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            classes = step.get("classes", [])
            class_set = {str(item).strip().lower() for item in classes} if isinstance(classes, list) else set()
            if "motorway" in class_set:
                counts["motorway"] += 1
            elif "trunk" in class_set:
                counts["trunk"] += 1
            elif "primary" in class_set:
                counts["primary"] += 1
            elif "secondary" in class_set:
                counts["secondary"] += 1
            else:
                counts["local"] += 1
    return counts


def _segment_annotation_signature(route: Mapping[str, Any]) -> str | None:
    legs = route.get("legs", [])
    if not isinstance(legs, list) or not legs:
        return None
    segments: list[dict[str, Any]] = []
    for leg in legs:
        if not isinstance(leg, Mapping):
            return None
        annotation = leg.get("annotation", {})
        if not isinstance(annotation, Mapping):
            return None
        distances = annotation.get("distance", [])
        durations = annotation.get("duration", [])
        if not isinstance(distances, list) or not isinstance(durations, list) or len(distances) != len(durations):
            return None
        segments.append(
            {
                "distance": [round(float(value), 3) for value in distances[:512]],
                "duration": [round(float(value), 3) for value in durations[:512]],
            }
        )
    encoded = json.dumps(segments, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _route_identity(route: Mapping[str, Any]) -> str | None:
    signature = _route_geometry_signature(route)
    if signature:
        return signature
    route_id = str(route.get("id") or route.get("route_id") or route.get("option_id") or "").strip()
    return route_id or None


def build_route_option_cache_key(
    route: Mapping[str, Any],
    *,
    vehicle_type: str,
    detail_level: str = "full",
    scenario_mode: ScenarioMode,
    cost_toggles: CostToggles,
    terrain_profile: TerrainProfile = "flat",
    stochastic: StochasticConfig | None = None,
    emissions_context: EmissionsContext | None = None,
    weather: WeatherImpactConfig | None = None,
    incident_simulation: IncidentSimulatorConfig | None = None,
    departure_time_utc: datetime | None = None,
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    risk_aversion: float = 1.0,
    optimization_mode: OptimizationMode = "expected_value",
    pareto_method: ParetoMethod = "dominance",
    epsilon: Any = None,
    max_alternatives: int | None = None,
) -> str | None:
    return _build_route_option_cache_key(
        route,
        vehicle_type=vehicle_type,
        detail_level=detail_level,
        scenario_mode=scenario_mode,
        cost_toggles=cost_toggles,
        terrain_profile=terrain_profile,
        stochastic=stochastic,
        emissions_context=emissions_context,
        weather=weather,
        incident_simulation=incident_simulation,
        departure_time_utc=departure_time_utc,
        utility_weights=utility_weights,
        risk_aversion=risk_aversion,
        optimization_mode=optimization_mode,
        pareto_method=pareto_method,
        epsilon=epsilon,
        max_alternatives=max_alternatives,
        include_detail_level=True,
    )


def build_route_option_core_cache_key(
    route: Mapping[str, Any],
    *,
    vehicle_type: str,
    scenario_mode: ScenarioMode,
    cost_toggles: CostToggles,
    terrain_profile: TerrainProfile = "flat",
    stochastic: StochasticConfig | None = None,
    emissions_context: EmissionsContext | None = None,
    weather: WeatherImpactConfig | None = None,
    incident_simulation: IncidentSimulatorConfig | None = None,
    departure_time_utc: datetime | None = None,
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    risk_aversion: float = 1.0,
    optimization_mode: OptimizationMode = "expected_value",
    pareto_method: ParetoMethod = "dominance",
    epsilon: Any = None,
    max_alternatives: int | None = None,
) -> str | None:
    return _build_route_option_cache_key(
        route,
        vehicle_type=vehicle_type,
        detail_level="core",
        scenario_mode=scenario_mode,
        cost_toggles=cost_toggles,
        terrain_profile=terrain_profile,
        stochastic=stochastic,
        emissions_context=emissions_context,
        weather=weather,
        incident_simulation=incident_simulation,
        departure_time_utc=departure_time_utc,
        utility_weights=utility_weights,
        risk_aversion=risk_aversion,
        optimization_mode=optimization_mode,
        pareto_method=pareto_method,
        epsilon=epsilon,
        max_alternatives=max_alternatives,
        include_detail_level=False,
    )


def _build_route_option_cache_key(
    route: Mapping[str, Any],
    *,
    vehicle_type: str,
    detail_level: str,
    scenario_mode: ScenarioMode,
    cost_toggles: CostToggles,
    terrain_profile: TerrainProfile = "flat",
    stochastic: StochasticConfig | None = None,
    emissions_context: EmissionsContext | None = None,
    weather: WeatherImpactConfig | None = None,
    incident_simulation: IncidentSimulatorConfig | None = None,
    departure_time_utc: datetime | None = None,
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    risk_aversion: float = 1.0,
    optimization_mode: OptimizationMode = "expected_value",
    pareto_method: ParetoMethod = "dominance",
    epsilon: Any = None,
    max_alternatives: int | None = None,
    include_detail_level: bool,
) -> str | None:
    if not bool(settings.route_option_cache_enabled):
        return None

    identity = _route_identity(route)
    if identity is None:
        return None

    segment_signature = _segment_annotation_signature(route)
    route_payload = {
        "route_identity": identity,
        "distance_m": round(float(route.get("distance", 0.0)), 6),
        "duration_s": round(float(route.get("duration", 0.0)), 6),
        "road_class_counts": _road_class_counts(route),
        "segment_annotation_signature": segment_signature,
        "evidence_snapshot_hash": str(route.get("evidence_snapshot_hash") or route.get("snapshot_hash") or ""),
        "evidence_provenance": _normalize_key_component(route.get("evidence_provenance")),
        "evidence_tensor": _normalize_key_component(route.get("evidence_tensor")),
    }
    if segment_signature is None and route_payload["evidence_snapshot_hash"] == "":
        # Route payload is too weak to safely reuse.
        return None

    cache_payload = {
        "schema_version": ROUTE_OPTION_CACHE_SCHEMA_VERSION,
        "route": route_payload,
        "settings": {
            "strict_live_data_required": bool(settings.strict_live_data_required),
            "carbon_policy_scenario": str(settings.carbon_policy_scenario),
            "route_option_segment_cap": int(settings.route_option_segment_cap),
            "route_option_segment_cap_long": int(settings.route_option_segment_cap_long),
            "route_option_long_distance_threshold_km": float(settings.route_option_long_distance_threshold_km),
            "route_option_tod_bucket_s": int(settings.route_option_tod_bucket_s),
            "route_option_energy_speed_bin_kph": float(settings.route_option_energy_speed_bin_kph),
            "route_option_energy_grade_bin_pct": float(settings.route_option_energy_grade_bin_pct),
        },
        "inputs": {
            "detail_level": str(detail_level).strip().lower() or "full",
            "vehicle_type": str(vehicle_type),
            "scenario_mode": str(getattr(scenario_mode, "value", scenario_mode)),
            "cost_toggles": _normalize_key_component(cost_toggles),
            "terrain_profile": str(getattr(terrain_profile, "value", terrain_profile)),
            "stochastic": _normalize_key_component(stochastic),
            "emissions_context": _normalize_key_component(emissions_context),
            "weather": _normalize_key_component(weather),
            "incident_simulation": _normalize_key_component(incident_simulation),
            "departure_time_utc": _normalize_key_component(departure_time_utc),
            "utility_weights": [round(float(value), 6) for value in utility_weights],
            "risk_aversion": round(float(risk_aversion), 6),
            "optimization_mode": str(getattr(optimization_mode, "value", optimization_mode)),
            "pareto_method": str(getattr(pareto_method, "value", pareto_method)),
            "epsilon": _normalize_key_component(epsilon),
            "max_alternatives": int(max_alternatives) if max_alternatives is not None else None,
        },
    }
    if not include_detail_level:
        cache_payload["inputs"]["detail_level"] = None
    encoded = json.dumps(cache_payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


class RouteOptionCacheStore(ProcessGlobalCacheStore[CachedRouteOptionBuild]):
    pass


class RouteOptionCoreCacheStore(ProcessGlobalCacheStore[CachedRouteOptionCore]):
    pass


ROUTE_OPTION_CACHE = RouteOptionCacheStore(
    ttl_s=settings.route_option_cache_ttl_s,
    max_entries=settings.route_option_cache_max_entries,
    max_estimated_bytes=settings.route_option_cache_max_estimated_bytes,
)

ROUTE_OPTION_CORE_CACHE = RouteOptionCoreCacheStore(
    ttl_s=settings.route_option_cache_ttl_s,
    max_entries=settings.route_option_cache_max_entries,
    max_estimated_bytes=settings.route_option_cache_max_estimated_bytes,
)


def get_cached_route_option_build(key: str) -> CachedRouteOptionBuild | None:
    if not bool(settings.route_option_cache_enabled):
        return None
    return ROUTE_OPTION_CACHE.get(key)


def set_cached_route_option_build(key: str, value: CachedRouteOptionBuild) -> bool:
    if not bool(settings.route_option_cache_enabled):
        return False
    return ROUTE_OPTION_CACHE.set(key, value)


def get_cached_route_option_core(key: str) -> CachedRouteOptionCore | None:
    if not bool(settings.route_option_cache_enabled):
        return None
    return ROUTE_OPTION_CORE_CACHE.get(key)


def set_cached_route_option_core(key: str, value: CachedRouteOptionCore) -> bool:
    if not bool(settings.route_option_cache_enabled):
        return False
    return ROUTE_OPTION_CORE_CACHE.set(key, value)


def clear_route_option_cache() -> int:
    if not bool(settings.route_option_cache_enabled):
        return 0
    return ROUTE_OPTION_CACHE.clear() + ROUTE_OPTION_CORE_CACHE.clear()


def route_option_cache_stats() -> dict[str, int]:
    snapshot = ROUTE_OPTION_CACHE.snapshot()
    core_snapshot = ROUTE_OPTION_CORE_CACHE.snapshot()
    snapshot["core_hits"] = int(core_snapshot.get("hits", 0))
    snapshot["core_misses"] = int(core_snapshot.get("misses", 0))
    snapshot["core_rejected"] = int(core_snapshot.get("rejected", 0))
    snapshot["enabled"] = int(bool(settings.route_option_cache_enabled))
    return snapshot
