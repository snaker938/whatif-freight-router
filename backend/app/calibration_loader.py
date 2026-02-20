from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .live_data_sources import (
    live_bank_holidays,
    live_departure_profiles,
    live_fuel_prices,
    live_stochastic_regimes,
    live_toll_tariffs,
    live_toll_topology,
)
from .model_data_errors import ModelDataError
from .settings import settings


def _strict_runtime_required() -> bool:
    # Pass-3 policy: strict runtime is always on for production code paths.
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assets_root() -> Path:
    return _repo_root() / "backend" / "assets" / "uk"


def _model_asset_root() -> Path:
    return Path(settings.model_asset_dir)


def _resolve_asset_path(filename: str) -> Path:
    generated = _model_asset_root() / filename
    if generated.exists():
        return generated
    bundled = _assets_root() / filename
    if bundled.exists():
        return bundled
    raise FileNotFoundError(f"model asset not found: {filename}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(raw: str | None) -> datetime | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _infer_as_of_from_payload(payload: dict[str, Any]) -> datetime | None:
    for key in (
        "as_of_utc",
        "as_of",
        "generated_at_utc",
        "generated_at",
        "refreshed_at_utc",
        "updated_at_utc",
    ):
        parsed = _parse_iso_datetime(str(payload.get(key, "")).strip())
        if parsed is not None:
            return parsed
    return None


def _infer_as_of_from_path(path: Path) -> datetime:
    return datetime.fromtimestamp(float(path.stat().st_mtime), tz=timezone.utc)


def _is_fresh(as_of_utc: datetime | None, *, max_age_days: int) -> bool:
    if as_of_utc is None:
        return False
    return (datetime.now(timezone.utc) - as_of_utc) <= timedelta(days=max(1, int(max_age_days)))


def _raise_if_strict_stale(
    *,
    reason_code: str,
    message: str,
    as_of_utc: datetime | None,
    max_age_days: int,
    enforce: bool = True,
) -> None:
    if not enforce:
        return
    if not _strict_runtime_required():
        return
    if _is_fresh(as_of_utc, max_age_days=max_age_days):
        return
    raise ModelDataError(
        reason_code=reason_code,
        message=message,
        details={
            "as_of_utc": as_of_utc.isoformat() if as_of_utc is not None else None,
            "max_age_days": int(max_age_days),
        },
    )


def _strict_live_dataset_enforced(live_url: str, *, require_auth_token: str | None = None) -> bool:
    if not _strict_runtime_required():
        return False
    # Enforce freshness even when no live URL is configured, so strict mode
    # still requires signed, fresh local assets.
    live_configured = bool(str(live_url or "").strip())
    if live_configured and require_auth_token is not None and not str(require_auth_token or "").strip():
        raise ModelDataError(
            reason_code="fuel_price_auth_unavailable",
            message="Fuel price live source requires authentication token in strict mode.",
        )
    return True


def _strict_empirical_departure_required() -> bool:
    return True


def _strict_empirical_stochastic_required() -> bool:
    return True


def _payload_is_synthetic(payload: dict[str, Any], *, version: str | None = None) -> bool:
    basis = str(payload.get("calibration_basis", "")).strip().lower()
    if basis in {"synthetic", "heuristic", "legacy"}:
        return True
    if version is None:
        version = str(payload.get("version", payload.get("calibration_version", "")))
    lowered = str(version or "").strip().lower()
    return "synthetic" in lowered or "legacy" in lowered


def _parse_departure_profile_payload(
    payload: dict[str, Any],
    *,
    source: str,
) -> DepartureProfile | None:
    version = str(payload.get("version", "uk-v3-contextual"))
    as_of_utc = str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None
    refreshed_at_utc = (
        str(payload.get("refreshed_at_utc", payload.get("generated_at_utc", ""))).strip() or None
    )
    weekday_raw = payload.get("weekday")
    weekend_raw = payload.get("weekend")
    holiday_raw = payload.get("holiday")
    if (
        isinstance(weekday_raw, list)
        and isinstance(weekend_raw, list)
        and isinstance(holiday_raw, list)
        and len(weekday_raw) == 1440
        and len(weekend_raw) == 1440
        and len(holiday_raw) == 1440
    ):
        return DepartureProfile(
            weekday=tuple(float(v) for v in weekday_raw),
            weekend=tuple(float(v) for v in weekend_raw),
            holiday=tuple(float(v) for v in holiday_raw),
            source=source,
            version=version,
            as_of_utc=as_of_utc,
            refreshed_at_utc=refreshed_at_utc,
            contextual={
                "uk_default": {
                    "mixed": {
                        "weekday": tuple(float(v) for v in weekday_raw),
                        "weekend": tuple(float(v) for v in weekend_raw),
                        "holiday": tuple(float(v) for v in holiday_raw),
                    }
                }
            },
        )

    profiles_raw = payload.get("profiles", {})
    contextual: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {}
    if isinstance(profiles_raw, dict):
        for region_key, road_map_raw in profiles_raw.items():
            if not isinstance(road_map_raw, dict):
                continue
            road_map_out: dict[str, dict[str, tuple[float, ...]]] = {}
            for road_key, day_map_raw in road_map_raw.items():
                if not isinstance(day_map_raw, dict):
                    continue
                day_map_out: dict[str, tuple[float, ...]] = {}
                for day_key, values_raw in day_map_raw.items():
                    if not isinstance(values_raw, list) or len(values_raw) != 1440:
                        continue
                    day_map_out[str(day_key)] = tuple(float(v) for v in values_raw)
                if day_map_out:
                    road_map_out[str(road_key)] = day_map_out
            if road_map_out:
                contextual[str(region_key)] = road_map_out
    if not contextual:
        return None

    envelopes_raw = payload.get("envelopes", {})
    contextual_envelopes: dict[str, dict[str, dict[str, dict[str, tuple[float, ...]]]]] = {}
    if isinstance(envelopes_raw, dict):
        for region_key, road_map_raw in envelopes_raw.items():
            if not isinstance(road_map_raw, dict):
                continue
            road_out: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {}
            for road_key, day_map_raw in road_map_raw.items():
                if not isinstance(day_map_raw, dict):
                    continue
                day_out: dict[str, dict[str, tuple[float, ...]]] = {}
                for day_key, env_raw in day_map_raw.items():
                    if not isinstance(env_raw, dict):
                        continue
                    low_raw = env_raw.get("low")
                    high_raw = env_raw.get("high")
                    if (
                        isinstance(low_raw, list)
                        and isinstance(high_raw, list)
                        and len(low_raw) == 1440
                        and len(high_raw) == 1440
                    ):
                        day_out[str(day_key)] = {
                            "low": tuple(float(v) for v in low_raw),
                            "high": tuple(float(v) for v in high_raw),
                        }
                if day_out:
                    road_out[str(road_key)] = day_out
            if road_out:
                contextual_envelopes[str(region_key)] = road_out

    default_road = contextual.get("uk_default", {}).get("mixed", {})
    weekday = default_road.get("weekday")
    weekend = default_road.get("weekend")
    holiday = default_road.get("holiday")
    if not weekday or not weekend or not holiday:
        return None
    return DepartureProfile(
        weekday=weekday,
        weekend=weekend,
        holiday=holiday,
        source=source,
        version=version,
        as_of_utc=as_of_utc,
        refreshed_at_utc=refreshed_at_utc,
        contextual=contextual,
        contextual_envelopes=contextual_envelopes or None,
    )


@dataclass(frozen=True)
class DepartureProfile:
    weekday: tuple[float, ...]
    weekend: tuple[float, ...]
    holiday: tuple[float, ...]
    source: str
    version: str = "uk-v3-contextual"
    as_of_utc: str | None = None
    refreshed_at_utc: str | None = None
    contextual: dict[str, dict[str, dict[str, tuple[float, ...]]]] | None = None
    contextual_envelopes: (
        dict[str, dict[str, dict[str, dict[str, tuple[float, ...]]]]]
        | None
    ) = None

    def resolve(
        self,
        *,
        day_kind: str,
        region: str,
        road_bucket: str,
    ) -> tuple[float, ...]:
        if not self.contextual:
            return {
                "weekday": self.weekday,
                "weekend": self.weekend,
                "holiday": self.holiday,
            }.get(day_kind, self.weekday)
        day = day_kind if day_kind in ("weekday", "weekend", "holiday") else "weekday"
        region_map = self.contextual.get(region) or self.contextual.get("uk_default")
        if not region_map:
            return {
                "weekday": self.weekday,
                "weekend": self.weekend,
                "holiday": self.holiday,
            }.get(day_kind, self.weekday)
        road_map = (
            region_map.get(road_bucket)
            or region_map.get("mixed")
            or next(iter(region_map.values()))
        )
        series = (
            road_map.get(day)
            or road_map.get("weekday")
            or self.weekday
        )
        return series

    def resolve_envelope(
        self,
        *,
        day_kind: str,
        region: str,
        road_bucket: str,
    ) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        if not self.contextual_envelopes:
            return None
        day = day_kind if day_kind in ("weekday", "weekend", "holiday") else "weekday"
        region_map = self.contextual_envelopes.get(region) or self.contextual_envelopes.get("uk_default")
        if not region_map:
            return None
        road_map = region_map.get(road_bucket) or region_map.get("mixed") or next(iter(region_map.values()))
        day_env = road_map.get(day) or road_map.get("weekday")
        if not isinstance(day_env, dict):
            return None
        low = day_env.get("low")
        high = day_env.get("high")
        if (
            isinstance(low, tuple)
            and isinstance(high, tuple)
            and len(low) == 1440
            and len(high) == 1440
        ):
            return low, high
        return None


@dataclass(frozen=True)
class RiskNormalizationReference:
    duration_s_per_km: float
    monetary_gbp_per_km: float
    emissions_kg_per_km: float
    source: str
    version: str = "unknown"
    as_of_utc: str | None = None
    corridor_bucket: str = "uk_default"
    day_kind: str = "weekday"
    local_time_slot: str = "h12"


def _canonical_vehicle_bucket(vehicle_type: str | None) -> str:
    key = (vehicle_type or "").strip().lower()
    if "artic" in key:
        return "artic_hgv"
    if "rigid" in key:
        return "rigid_hgv"
    if "van" in key:
        return "van"
    if "ev" in key:
        return "ev_generic"
    return "default"


def _canonical_corridor_bucket(corridor_bucket: str | None) -> str:
    key = (corridor_bucket or "").strip().lower()
    if not key:
        return "uk_default"
    aliases = {
        "london": "london_southeast",
        "south_east": "london_southeast",
        "south-east": "london_southeast",
        "north": "north_england",
        "midlands_central": "midlands",
        "wales": "wales_west",
    }
    return aliases.get(key, key)


def _canonical_day_kind(day_kind: str | None) -> str:
    key = (day_kind or "").strip().lower()
    if key in {"weekday", "weekend", "holiday"}:
        return key
    return "weekday"


def _canonical_local_time_slot(local_time_slot: str | None) -> str:
    key = (local_time_slot or "").strip().lower()
    if key.startswith("h") and len(key) == 3:
        try:
            hour = int(key[1:])
        except ValueError:
            hour = 12
        return f"h{max(0, min(23, hour)):02d}"
    try:
        hour = int(key)
    except ValueError:
        return "h12"
    return f"h{max(0, min(23, hour)):02d}"


@lru_cache(maxsize=128)
def load_risk_normalization_reference(
    vehicle_type: str | None = None,
    corridor_bucket: str | None = None,
    day_kind: str | None = None,
    local_time_slot: str | None = None,
) -> RiskNormalizationReference:
    path_candidates = [
        _model_asset_root() / "risk_normalization_refs_uk.json",
        _assets_root() / "risk_normalization_refs_uk.json",
    ]
    payload: dict[str, Any] | None = None
    source = ""
    version = ""
    as_of_utc: str | None = None
    selected_path: Path | None = None
    for path in path_candidates:
        if not path.exists():
            continue
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            payload = parsed
            source = str(path)
            version = str(parsed.get("version", parsed.get("calibration_version", "unknown")))
            as_of_utc = str(parsed.get("as_of_utc", parsed.get("as_of", ""))).strip() or None
            selected_path = path
            break
    if payload is None:
        raise ModelDataError(
            reason_code="risk_normalization_unavailable",
            message="Risk normalization reference asset is required in strict runtime.",
        )
    _raise_if_strict_stale(
        reason_code="risk_normalization_unavailable",
        message="Risk normalization reference asset is stale for strict runtime policy.",
        as_of_utc=_infer_as_of_from_payload(payload) or (_infer_as_of_from_path(selected_path) if selected_path else None),
        max_age_days=int(settings.live_stochastic_max_age_days),
        enforce=True,
    )

    vehicle_bucket = _canonical_vehicle_bucket(vehicle_type)
    corridor_key = _canonical_corridor_bucket(corridor_bucket)
    day_key = _canonical_day_kind(day_kind)
    slot_key = _canonical_local_time_slot(local_time_slot)

    selected_ref: dict[str, Any] | None = None
    corridor_vehicle_refs = payload.get("corridor_vehicle_refs", {})
    if isinstance(corridor_vehicle_refs, dict):
        c_item = corridor_vehicle_refs.get(corridor_key)
        if isinstance(c_item, dict):
            by_vehicle = c_item.get(vehicle_bucket)
            if isinstance(by_vehicle, dict):
                by_day = by_vehicle.get(day_key)
                if isinstance(by_day, dict):
                    if isinstance(by_day.get(slot_key), dict):
                        selected_ref = by_day.get(slot_key)
                    elif {"duration_s_per_km", "monetary_gbp_per_km", "emissions_kg_per_km"}.issubset(by_day.keys()):
                        selected_ref = by_day

    if not isinstance(selected_ref, dict):
        raise ModelDataError(
            reason_code="risk_normalization_unavailable",
            message=(
                "No risk normalization entry matched strict route context "
                f"(vehicle={vehicle_bucket}, corridor={corridor_key}, day={day_key}, slot={slot_key})."
            ),
        )

    defaults = {
        "duration_s_per_km": max(
            1.0,
            _safe_float(selected_ref.get("duration_s_per_km"), 0.0),
        ),
        "monetary_gbp_per_km": max(
            0.05,
            _safe_float(selected_ref.get("monetary_gbp_per_km"), 0.0),
        ),
        "emissions_kg_per_km": max(
            0.01,
            _safe_float(selected_ref.get("emissions_kg_per_km"), 0.0),
        ),
    }

    return RiskNormalizationReference(
        duration_s_per_km=defaults["duration_s_per_km"],
        monetary_gbp_per_km=defaults["monetary_gbp_per_km"],
        emissions_kg_per_km=defaults["emissions_kg_per_km"],
        source=source,
        version=version,
        as_of_utc=as_of_utc,
        corridor_bucket=corridor_key,
        day_kind=day_key,
        local_time_slot=slot_key,
    )


def _interpolate_sparse_profile(
    points: list[tuple[int, float]],
    *,
    default_value: float,
) -> tuple[float, ...]:
    if not points:
        return tuple(default_value for _ in range(1440))

    points = sorted((max(0, min(1439, int(m))), float(v)) for m, v in points)
    dense: list[float] = [default_value for _ in range(1440)]

    # Fill leading region.
    first_minute, first_value = points[0]
    for minute in range(0, first_minute + 1):
        dense[minute] = first_value

    # Interpolate between knots.
    for idx in range(1, len(points)):
        prev_minute, prev_value = points[idx - 1]
        next_minute, next_value = points[idx]
        span = max(1, next_minute - prev_minute)
        for minute in range(prev_minute, next_minute + 1):
            t = (minute - prev_minute) / span
            dense[minute] = prev_value + ((next_value - prev_value) * t)

    # Fill trailing region.
    last_minute, last_value = points[-1]
    for minute in range(last_minute, 1440):
        dense[minute] = last_value

    return tuple(dense)


@lru_cache(maxsize=1)
def load_departure_profile() -> DepartureProfile:
    strict_empirical_required = _strict_empirical_departure_required()
    strict_freshness_required = _strict_runtime_required()
    rejected_synthetic = False
    if settings.live_runtime_data_enabled:
        live_payload = live_departure_profiles()
        if isinstance(live_payload, dict):
            parsed = _parse_departure_profile_payload(
                live_payload,
                source="live_runtime:departure_profiles",
            )
            if parsed is not None:
                live_as_of = _infer_as_of_from_payload(live_payload)
                if (
                    strict_empirical_required
                    and not settings.departure_allow_synthetic_profiles
                    and _payload_is_synthetic(
                        live_payload, version=parsed.version
                    )
                ):
                    rejected_synthetic = True
                elif _is_fresh(live_as_of, max_age_days=int(settings.live_departure_max_age_days)):
                    return parsed

    json_candidates = [
        _model_asset_root() / "departure_profiles_uk.json",
        _assets_root() / "departure_profiles_uk.json",
    ]
    for json_path in json_candidates:
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        parsed = _parse_departure_profile_payload(payload, source=str(json_path))
        if parsed is None:
            continue
        if (
            strict_empirical_required
            and not settings.departure_allow_synthetic_profiles
            and _payload_is_synthetic(
                payload, version=parsed.version
            )
        ):
            rejected_synthetic = True
            continue
        payload_as_of = _infer_as_of_from_payload(payload) or _infer_as_of_from_path(json_path)
        _raise_if_strict_stale(
            reason_code="departure_profile_unavailable",
            message="Departure profile fallback asset is stale for strict live-data policy.",
            as_of_utc=payload_as_of,
            max_age_days=int(settings.live_departure_max_age_days),
            enforce=strict_freshness_required,
        )
        return parsed

    csv_candidates = [
        _model_asset_root() / "departure_profile_uk.csv",
        _assets_root() / "departure_profile_uk.csv",
    ]
    # Hard-strict runtime does not permit sparse CSV fallback profiles.
    if strict_empirical_required or _strict_runtime_required():
        csv_candidates = []
    for path in csv_candidates:
        if not path.exists():
            continue
        file_as_of_dt = _infer_as_of_from_path(path)
        _raise_if_strict_stale(
            reason_code="departure_profile_unavailable",
            message="Departure profile CSV fallback asset is stale for strict live-data policy.",
            as_of_utc=file_as_of_dt,
            max_age_days=int(settings.live_departure_max_age_days),
            enforce=strict_freshness_required,
        )
        weekday_points: list[tuple[int, float]] = []
        weekend_points: list[tuple[int, float]] = []
        holiday_points: list[tuple[int, float]] = []

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                minute = int(_safe_float(row.get("minute"), 0.0))
                weekday_points.append((minute, _safe_float(row.get("weekday"), 1.0)))
                weekend_points.append((minute, _safe_float(row.get("weekend"), 1.0)))
                holiday_points.append(
                    (minute, _safe_float(row.get("holiday"), _safe_float(row.get("weekend"), 1.0)))
                )

        weekday = _interpolate_sparse_profile(weekday_points, default_value=1.0)
        weekend = _interpolate_sparse_profile(weekend_points, default_value=1.0)
        holiday = _interpolate_sparse_profile(holiday_points, default_value=1.0)
        file_as_of = file_as_of_dt.isoformat()
        return DepartureProfile(
            weekday=weekday,
            weekend=weekend,
            holiday=holiday,
            source=str(path),
            version="legacy_sparse_csv",
            as_of_utc=file_as_of,
            refreshed_at_utc=file_as_of,
        )

    if rejected_synthetic:
        raise ModelDataError(
            reason_code="departure_profile_unavailable",
            message="Only synthetic departure profiles were available; empirical profile assets are required.",
        )
    raise ModelDataError(
        reason_code="departure_profile_unavailable",
        message="No valid departure profile asset was available for strict routing policy.",
    )


@lru_cache(maxsize=1)
def load_uk_bank_holidays() -> frozenset[str]:
    payload: Any = None
    fallback_path: Path | None = None
    if settings.live_runtime_data_enabled:
        live_payload = live_bank_holidays()
        if isinstance(live_payload, dict):
            live_as_of = _infer_as_of_from_payload(live_payload)
            if _is_fresh(live_as_of, max_age_days=int(settings.live_departure_max_age_days)):
                payload = live_payload
    if payload is None:
        path = _resolve_asset_path("bank_holidays_uk.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        fallback_path = path
    if fallback_path is not None:
        _raise_if_strict_stale(
            reason_code="holiday_data_unavailable",
            message="Bank holiday fallback asset is stale for strict live-data policy.",
            as_of_utc=_infer_as_of_from_payload(payload) or _infer_as_of_from_path(fallback_path),
            max_age_days=int(settings.live_departure_max_age_days),
            enforce=True,
        )

    if not isinstance(payload, dict):
        raise ModelDataError(
            reason_code="holiday_data_unavailable",
            message="No valid bank holiday dataset available.",
        )
    dates: set[str] = set()
    for region_value in payload.values():
        if not isinstance(region_value, dict):
            continue
        events = region_value.get("events", [])
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            date_text = str(event.get("date", "")).strip()
            if date_text:
                dates.add(date_text)
    return frozenset(dates)


@dataclass(frozen=True)
class TerrainManifestInfo:
    version: str
    source: str
    tile_count: int
    bounds: dict[str, float]


@lru_cache(maxsize=1)
def load_terrain_manifest_info() -> TerrainManifestInfo:
    path_candidates = [
        _model_asset_root() / "terrain" / "terrain_manifest.json",
        _model_asset_root() / "terrain_manifest.json",
    ]
    for path in path_candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        raw_bounds = payload.get("bounds", {})
        bounds = {
            "lat_min": _safe_float(raw_bounds.get("lat_min"), 0.0),
            "lat_max": _safe_float(raw_bounds.get("lat_max"), 0.0),
            "lon_min": _safe_float(raw_bounds.get("lon_min"), 0.0),
            "lon_max": _safe_float(raw_bounds.get("lon_max"), 0.0),
        }
        raw_tiles = payload.get("tiles", [])
        tile_count = len(raw_tiles) if isinstance(raw_tiles, list) else 0
        return TerrainManifestInfo(
            version=str(payload.get("version", "unknown")),
            source=str(path),
            tile_count=tile_count,
            bounds=bounds,
        )
    return TerrainManifestInfo(
        version="missing",
        source="none",
        tile_count=0,
        bounds={"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    )


@dataclass(frozen=True)
class TollTariffRule:
    rule_id: str
    operator: str
    crossing_id: str
    road_class: str
    direction: str
    start_minute: int
    end_minute: int
    crossing_fee_gbp: float
    distance_fee_gbp_per_km: float
    vehicle_classes: tuple[str, ...]
    axle_classes: tuple[str, ...]
    payment_classes: tuple[str, ...]
    exemptions: tuple[str, ...]


@dataclass(frozen=True)
class TollTariffTable:
    default_crossing_fee_gbp: float
    default_distance_fee_gbp_per_km: float
    rules: tuple[TollTariffRule, ...]
    source: str


def _parse_tariff_payload(payload: dict[str, Any], *, source: str) -> TollTariffTable:
    defaults = payload.get("defaults", {})
    default_crossing = _safe_float(defaults.get("crossing_fee_gbp"), 0.0)
    default_distance = _safe_float(defaults.get("distance_fee_gbp_per_km"), 0.0)

    rules: list[TollTariffRule] = []
    raw_rules = payload.get("rules", [])
    if isinstance(raw_rules, list):
        for idx, item in enumerate(raw_rules):
            if not isinstance(item, dict):
                continue
            vehicle_classes_raw = item.get("vehicle_classes", ["default"])
            if isinstance(vehicle_classes_raw, list):
                vehicle_classes = tuple(str(v).strip().lower() for v in vehicle_classes_raw if str(v).strip())
            else:
                vehicle_classes = ("default",)
            axle_classes_raw = item.get("axle_classes", ["default"])
            payment_classes_raw = item.get("payment_classes", ["default"])
            exemptions_raw = item.get("exemptions", [])
            axle_classes = (
                tuple(str(v).strip().lower() for v in axle_classes_raw if str(v).strip())
                if isinstance(axle_classes_raw, list)
                else ("default",)
            )
            payment_classes = (
                tuple(str(v).strip().lower() for v in payment_classes_raw if str(v).strip())
                if isinstance(payment_classes_raw, list)
                else ("default",)
            )
            exemptions = (
                tuple(str(v).strip().lower() for v in exemptions_raw if str(v).strip())
                if isinstance(exemptions_raw, list)
                else ()
            )
            rules.append(
                TollTariffRule(
                    rule_id=str(item.get("id", f"rule_{idx}")),
                    operator=str(item.get("operator", "default")).strip().lower(),
                    crossing_id=str(item.get("crossing_id", "*")).strip().lower(),
                    road_class=str(item.get("road_class", "default")).strip().lower(),
                    direction=str(item.get("direction", "both")).strip().lower(),
                    start_minute=max(0, min(1439, int(_safe_float(item.get("start_minute"), 0.0)))),
                    end_minute=max(0, min(1439, int(_safe_float(item.get("end_minute"), 1439.0)))),
                    crossing_fee_gbp=max(0.0, _safe_float(item.get("crossing_fee_gbp"), default_crossing)),
                    distance_fee_gbp_per_km=max(
                        0.0,
                        _safe_float(item.get("distance_fee_gbp_per_km"), default_distance),
                    ),
                    vehicle_classes=vehicle_classes or ("default",),
                    axle_classes=axle_classes or ("default",),
                    payment_classes=payment_classes or ("default",),
                    exemptions=exemptions,
                )
            )

    if _strict_runtime_required():
        strict_rules: list[TollTariffRule] = []
        for rule in rules:
            if rule.operator in {"", "default", "*"}:
                continue
            if rule.crossing_id in {"", "*", "default"}:
                continue
            if rule.road_class in {"", "default", "*"}:
                continue
            if rule.direction in {"", "default", "*"}:
                continue
            if (
                not rule.vehicle_classes
                or any(item in {"", "default", "*"} for item in rule.vehicle_classes)
            ):
                continue
            if (
                not rule.axle_classes
                or any(item in {"", "default", "*"} for item in rule.axle_classes)
            ):
                continue
            if (
                not rule.payment_classes
                or any(item in {"", "default", "*"} for item in rule.payment_classes)
            ):
                continue
            strict_rules.append(rule)
        rules = strict_rules

    return TollTariffTable(
        default_crossing_fee_gbp=max(0.0, default_crossing),
        default_distance_fee_gbp_per_km=max(0.0, default_distance),
        rules=tuple(rules),
        source=source,
    )


@lru_cache(maxsize=1)
def load_toll_tariffs() -> TollTariffTable:
    if settings.live_runtime_data_enabled:
        live_payload = live_toll_tariffs()
        if isinstance(live_payload, dict):
            live_table = _parse_tariff_payload(live_payload, source="live_runtime:toll_tariffs")
            live_as_of = _infer_as_of_from_payload(live_payload)
            live_fresh = _is_fresh(live_as_of, max_age_days=int(settings.live_toll_tariffs_max_age_days))
            if live_table.rules and live_fresh:
                return live_table

    path = _resolve_asset_path("toll_tariffs_uk.yaml")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid toll tariff payload")
    table = _parse_tariff_payload(payload, source=str(path))
    if not table.rules:
        raise ModelDataError(
            reason_code="toll_tariff_unavailable",
            message=f"No toll tariff rules parsed from source '{path}'.",
        )
    _raise_if_strict_stale(
        reason_code="toll_tariff_unavailable",
        message="Toll tariff fallback asset is stale for strict live-data policy.",
        as_of_utc=_infer_as_of_from_payload(payload) or _infer_as_of_from_path(path),
        max_age_days=int(settings.live_toll_tariffs_max_age_days),
        enforce=_strict_live_dataset_enforced(settings.live_toll_tariffs_url),
    )
    return table


@dataclass(frozen=True)
class TollSegmentSeed:
    segment_id: str
    name: str
    operator: str
    road_class: str
    crossing_id: str
    direction: str
    crossing_fee_gbp: float
    distance_fee_gbp_per_km: float
    coordinates: tuple[tuple[float, float], ...]  # (lat, lon)


@lru_cache(maxsize=1)
def load_toll_segments_seed() -> tuple[TollSegmentSeed, ...]:
    payload: Any = None
    source_label = "unknown"
    fallback_path_used: Path | None = None
    if settings.live_runtime_data_enabled:
        live_payload = live_toll_topology()
        if isinstance(live_payload, dict):
            live_as_of = _infer_as_of_from_payload(live_payload)
            live_fresh = _is_fresh(live_as_of, max_age_days=int(settings.live_toll_topology_max_age_days))
            if live_fresh:
                payload = live_payload
                source_label = "live_runtime:toll_topology"

    if payload is None:
        path_candidates = [
            _model_asset_root() / "osm_toll_assets.geojson",
            _assets_root() / "osm_toll_assets.geojson",
        ]
        for candidate in path_candidates:
            if not candidate.exists():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            source_label = str(candidate)
            fallback_path_used = candidate
            break

    if not isinstance(payload, dict):
        if _strict_live_dataset_enforced(settings.live_toll_topology_url):
            raise ModelDataError(
                reason_code="toll_topology_unavailable",
                message="No toll topology payload is available under strict live-data policy.",
            )
        return ()

    features = payload.get("features", [])
    if not isinstance(features, list):
        return ()

    seeds: list[TollSegmentSeed] = []
    for idx, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        if not isinstance(props, dict) or not isinstance(geom, dict):
            continue
        geom_type = str(geom.get("type")).lower()
        raw_coords = geom.get("coordinates", [])
        lat_lon_points: list[tuple[float, float]] = []
        if geom_type == "linestring" and isinstance(raw_coords, list):
            for point in raw_coords:
                if not isinstance(point, list | tuple) or len(point) < 2:
                    continue
                lon = _safe_float(point[0], 0.0)
                lat = _safe_float(point[1], 0.0)
                lat_lon_points.append((lat, lon))
        elif geom_type == "multilinestring" and isinstance(raw_coords, list):
            for line in raw_coords:
                if not isinstance(line, list):
                    continue
                for point in line:
                    if not isinstance(point, list | tuple) or len(point) < 2:
                        continue
                    lon = _safe_float(point[0], 0.0)
                    lat = _safe_float(point[1], 0.0)
                    lat_lon_points.append((lat, lon))
        elif geom_type == "point" and isinstance(raw_coords, list) and len(raw_coords) >= 2:
            lon = _safe_float(raw_coords[0], 0.0)
            lat = _safe_float(raw_coords[1], 0.0)
            # Convert point assets (booths/gantries) into tiny directional stubs so
            # overlap scoring can still evaluate route proximity.
            lat_lon_points = [(lat, lon), (lat + 0.00005, lon + 0.00005)]
        if len(lat_lon_points) < 2:
            continue
        seeds.append(
            TollSegmentSeed(
                segment_id=str(props.get("id", f"segment_{idx}")),
                name=str(props.get("name", "")),
                operator=str(props.get("operator", "default")).strip().lower(),
                road_class=str(props.get("road_class", "default")).strip().lower(),
                crossing_id=str(props.get("crossing_id", props.get("id", f"segment_{idx}"))).strip().lower(),
                direction=str(props.get("direction", "both")).strip().lower(),
                crossing_fee_gbp=max(0.0, _safe_float(props.get("crossing_fee_gbp"), 0.0)),
                distance_fee_gbp_per_km=max(
                    0.0,
                    _safe_float(props.get("distance_fee_gbp_per_km"), 0.0),
                ),
                coordinates=tuple(lat_lon_points),
            )
        )
    if _strict_live_dataset_enforced(settings.live_toll_topology_url) and not seeds:
        raise ModelDataError(
            reason_code="toll_topology_unavailable",
            message=f"No valid toll segments parsed from source '{source_label}'.",
        )
    if fallback_path_used is not None:
        _raise_if_strict_stale(
            reason_code="toll_topology_unavailable",
            message="Toll topology fallback asset is stale for strict live-data policy.",
            as_of_utc=_infer_as_of_from_payload(payload) or _infer_as_of_from_path(fallback_path_used),
            max_age_days=int(settings.live_toll_topology_max_age_days),
            enforce=_strict_live_dataset_enforced(settings.live_toll_topology_url),
        )
    return tuple(seeds)


@dataclass(frozen=True)
class TollConfidenceBin:
    minimum: float
    maximum: float
    calibrated: float


@dataclass(frozen=True)
class TollConfidenceCalibration:
    intercept: float
    class_signal_coef: float
    seed_signal_coef: float
    segment_signal_coef: float
    bonus_both: float
    bonus_class: float
    bins: tuple[TollConfidenceBin, ...]
    source: str
    version: str
    as_of_utc: str | None


@lru_cache(maxsize=1)
def load_toll_confidence_calibration() -> TollConfidenceCalibration:
    candidates = [
        _model_asset_root() / "toll_confidence_calibration_uk.json",
        _assets_root() / "toll_confidence_calibration_uk.json",
    ]
    payload: dict[str, Any] | None = None
    source = ""
    fallback_path: Path | None = None
    for path in candidates:
        if not path.exists():
            continue
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            continue
        payload = parsed
        source = str(path)
        fallback_path = path
        break
    if payload is None:
        raise ModelDataError(
            reason_code="toll_topology_unavailable",
            message="Toll confidence calibration asset is required in strict runtime.",
        )

    if fallback_path is not None:
        _raise_if_strict_stale(
            reason_code="toll_topology_unavailable",
            message="Toll confidence calibration asset is stale for strict live-data policy.",
            as_of_utc=_infer_as_of_from_payload(payload) or _infer_as_of_from_path(fallback_path),
            max_age_days=int(settings.live_toll_topology_max_age_days),
            enforce=True,
        )

    model = payload.get("logit_model", {})
    if not isinstance(model, dict):
        model = {}
    raw_bins = payload.get("reliability_bins", [])
    if _strict_runtime_required():
        required_model_keys = (
            "intercept",
            "class_signal",
            "seed_signal",
            "segment_signal",
            "source_bonus_both",
            "source_bonus_class",
        )
        missing_keys = [key for key in required_model_keys if key not in model]
        if missing_keys:
            raise ModelDataError(
                reason_code="toll_topology_unavailable",
                message=(
                    "Toll confidence calibration asset is missing required logit coefficients: "
                    + ", ".join(missing_keys)
                ),
            )
        if not isinstance(raw_bins, list) or not raw_bins:
            raise ModelDataError(
                reason_code="toll_topology_unavailable",
                message="Toll confidence calibration asset is missing reliability bins in strict runtime.",
            )
    bins: list[TollConfidenceBin] = []
    if isinstance(raw_bins, list):
        for row in raw_bins:
            if not isinstance(row, dict):
                continue
            lo = max(0.0, min(1.0, _safe_float(row.get("min"), 0.0)))
            hi = max(lo, min(1.0, _safe_float(row.get("max"), 1.0)))
            calibrated = max(0.0, min(1.0, _safe_float(row.get("calibrated"), (lo + hi) * 0.5)))
            bins.append(TollConfidenceBin(minimum=lo, maximum=hi, calibrated=calibrated))
    if not bins:
        if _strict_runtime_required():
            raise ModelDataError(
                reason_code="toll_topology_unavailable",
                message="No valid toll confidence reliability bins were parsed for strict runtime.",
            )
        bins = [
            TollConfidenceBin(minimum=0.0, maximum=0.5, calibrated=0.35),
            TollConfidenceBin(minimum=0.5, maximum=1.0, calibrated=0.75),
        ]

    if _strict_runtime_required():
        try:
            intercept = float(model["intercept"])
            class_signal = float(model["class_signal"])
            seed_signal = float(model["seed_signal"])
            segment_signal = float(model["segment_signal"])
            source_bonus_both = float(model["source_bonus_both"])
            source_bonus_class = float(model["source_bonus_class"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ModelDataError(
                reason_code="toll_topology_unavailable",
                message="Toll confidence calibration coefficients must be numeric in strict runtime.",
            ) from exc
    else:
        intercept = _safe_float(model.get("intercept"), -1.05)
        class_signal = _safe_float(model.get("class_signal"), 1.85)
        seed_signal = _safe_float(model.get("seed_signal"), 2.10)
        segment_signal = _safe_float(model.get("segment_signal"), 0.55)
        source_bonus_both = _safe_float(model.get("source_bonus_both"), 0.06)
        source_bonus_class = _safe_float(model.get("source_bonus_class"), 0.02)

    return TollConfidenceCalibration(
        intercept=intercept,
        class_signal_coef=class_signal,
        seed_signal_coef=seed_signal,
        segment_signal_coef=segment_signal,
        bonus_both=source_bonus_both,
        bonus_class=source_bonus_class,
        bins=tuple(bins),
        source=source,
        version=str(payload.get("version", "unknown")),
        as_of_utc=str(payload.get("as_of_utc", "")).strip() or None,
    )


@dataclass(frozen=True)
class FuelPriceSnapshot:
    prices_gbp_per_l: dict[str, float]
    grid_price_gbp_per_kwh: float
    regional_multipliers: dict[str, float]
    as_of: str | None
    source: str


@lru_cache(maxsize=16)
def load_fuel_price_snapshot(as_of_utc: datetime | None = None) -> FuelPriceSnapshot:
    payload: Any = None
    source_label = "unknown"
    fallback_path: Path | None = None
    if settings.live_runtime_data_enabled:
        live_payload = live_fuel_prices(as_of_utc)
        if isinstance(live_payload, dict):
            live_as_of = _infer_as_of_from_payload(live_payload)
            if live_as_of is None:
                history = live_payload.get("history", [])
                if isinstance(history, list):
                    parsed_dates = [
                        _parse_iso_datetime(str(item.get("as_of", "")).strip())
                        for item in history
                        if isinstance(item, dict)
                    ]
                    parsed_dates = [item for item in parsed_dates if item is not None]
                    if parsed_dates:
                        live_as_of = max(parsed_dates)
            if _is_fresh(live_as_of, max_age_days=int(settings.live_fuel_max_age_days)):
                payload = live_payload
                source_label = "live_runtime:fuel_prices"

    if payload is None:
        path = _resolve_asset_path("fuel_prices_uk.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_label = str(path)
        fallback_path = path

    if not isinstance(payload, dict):
        raise ValueError("invalid fuel price payload")
    candidate_rows: list[dict[str, Any]] = []
    history = payload.get("history", [])
    if isinstance(history, list):
        for row in history:
            if isinstance(row, dict):
                candidate_rows.append(row)
    if isinstance(payload.get("prices_gbp_per_l"), dict):
        candidate_rows.append(payload)

    selected_row: dict[str, Any] | None = None
    if as_of_utc is not None and candidate_rows:
        if as_of_utc.tzinfo is None:
            target = as_of_utc.replace(tzinfo=timezone.utc)
        else:
            target = as_of_utc.astimezone(timezone.utc)
        dated_rows: list[tuple[datetime, dict[str, Any]]] = []
        for row in candidate_rows:
            as_of_text = str(row.get("as_of", "")).strip()
            try:
                dt = datetime.fromisoformat(as_of_text.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
            except ValueError:
                continue
            dated_rows.append((dt, row))
        dated_rows.sort(key=lambda item: item[0])
        for dt, row in dated_rows:
            if dt <= target:
                selected_row = row
        if selected_row is None and dated_rows:
            selected_row = dated_rows[-1][1]
    if selected_row is None:
        selected_row = candidate_rows[-1] if candidate_rows else payload

    prices_raw = selected_row.get("prices_gbp_per_l", {})
    prices: dict[str, float] = {}
    if isinstance(prices_raw, dict):
        for key, value in prices_raw.items():
            name = str(key).strip().lower()
            if not name:
                continue
            prices[name] = max(0.0, _safe_float(value, 0.0))
    if not prices:
        raise ModelDataError(
            reason_code="fuel_price_source_unavailable",
            message="Fuel price dataset has no valid prices_gbp_per_l entries in strict runtime.",
        )

    grid_price = _safe_float(selected_row.get("grid_price_gbp_per_kwh"), -1.0)
    if grid_price < 0.0:
        raise ModelDataError(
            reason_code="fuel_price_source_unavailable",
            message="Fuel price dataset is missing grid_price_gbp_per_kwh in strict runtime.",
        )
    grid_price = max(0.0, grid_price)
    regional_raw = selected_row.get("regional_multipliers", payload.get("regional_multipliers", {}))
    regional_multipliers: dict[str, float] = {}
    if isinstance(regional_raw, dict):
        for region_name, mult in regional_raw.items():
            regional_multipliers[str(region_name).strip().lower()] = max(0.6, min(1.5, _safe_float(mult, 1.0)))
    if not regional_multipliers:
        raise ModelDataError(
            reason_code="fuel_price_source_unavailable",
            message="Fuel price dataset has no regional_multipliers entries in strict runtime.",
        )
    as_of = str(selected_row.get("as_of", "")).strip() or None
    if fallback_path is not None:
        # Freshness should reflect dataset recency, not the historical row picked
        # for a scenario timestamp.
        fallback_as_of = _infer_as_of_from_payload(payload) or _parse_iso_datetime(as_of) or _infer_as_of_from_path(
            fallback_path
        )
        _raise_if_strict_stale(
            reason_code="fuel_price_source_unavailable",
            message="Fuel price fallback asset is stale for strict live-data policy.",
            as_of_utc=fallback_as_of,
            max_age_days=int(settings.live_fuel_max_age_days),
            enforce=_strict_live_dataset_enforced(
                settings.live_fuel_price_url,
                require_auth_token=settings.live_fuel_auth_token,
            ),
        )
    return FuelPriceSnapshot(
        prices_gbp_per_l=prices,
        grid_price_gbp_per_kwh=grid_price,
        regional_multipliers=regional_multipliers,
        as_of=as_of,
        source=source_label,
    )


@dataclass(frozen=True)
class StochasticRegime:
    regime_id: str
    sigma_scale: float
    traffic_scale: float
    incident_scale: float
    weather_scale: float
    price_scale: float
    eco_scale: float
    corr: tuple[tuple[float, ...], ...]
    spread_floor: float = 0.05
    spread_cap: float = 1.25
    factor_low: float = 0.55
    factor_high: float = 2.20
    duration_mix: tuple[float, float, float] = (1.0, 1.0, 1.0)
    monetary_mix: tuple[float, float] = (0.62, 0.38)
    emissions_mix: tuple[float, float] = (0.72, 0.28)


@dataclass(frozen=True)
class StochasticRegimeTable:
    copula_id: str
    calibration_version: str
    regimes: dict[str, StochasticRegime]
    source: str
    as_of_utc: str | None = None


@lru_cache(maxsize=1)
def load_stochastic_regimes() -> StochasticRegimeTable:
    strict_empirical_required = _strict_empirical_stochastic_required()
    strict_freshness_required = _strict_runtime_required()
    payload: dict[str, Any] = {}
    source: str = ""
    fallback_path: Path | None = None
    rejected_synthetic = False
    if settings.live_runtime_data_enabled:
        live_payload = live_stochastic_regimes()
        if isinstance(live_payload, dict):
            live_as_of = _infer_as_of_from_payload(live_payload)
            live_is_synthetic = _payload_is_synthetic(
                live_payload,
                version=str(live_payload.get("calibration_version", "")),
            )
            if (
                strict_empirical_required
                and live_is_synthetic
                and not settings.stochastic_allow_synthetic_calibration
            ):
                rejected_synthetic = True
            elif _is_fresh(live_as_of, max_age_days=int(settings.live_stochastic_max_age_days)):
                payload = live_payload
                source = "live_runtime:stochastic_regimes"
    if not payload:
        path_candidates = [
            _model_asset_root() / "stochastic_regimes_uk.json",
            _assets_root() / "stochastic_regimes_uk.json",
        ]
        path = next((candidate for candidate in path_candidates if candidate.exists()), None)
        if path is not None:
            candidate_payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(candidate_payload, dict):
                synthetic = _payload_is_synthetic(
                    candidate_payload,
                    version=str(candidate_payload.get("calibration_version", "")),
                )
                if (
                    strict_empirical_required
                    and synthetic
                    and not settings.stochastic_allow_synthetic_calibration
                ):
                    rejected_synthetic = True
                else:
                    payload = candidate_payload
                    source = str(path)
                    fallback_path = path
    if not isinstance(payload, dict):
        payload = {}
    if strict_empirical_required and not payload:
        raise ModelDataError(
            reason_code="stochastic_calibration_unavailable",
            message="No stochastic calibration payload is available under strict live-data policy.",
        )
    raw_regimes = payload.get("regimes", {})
    regimes: dict[str, StochasticRegime] = {}
    if isinstance(raw_regimes, dict):
        for regime_id, item in raw_regimes.items():
            if not isinstance(item, dict):
                continue
            corr_raw = item.get("corr", [])
            corr: list[tuple[float, ...]] = []
            if isinstance(corr_raw, list):
                for row in corr_raw:
                    if isinstance(row, list) and len(row) == 5:
                        corr.append(tuple(float(v) for v in row))
            if len(corr) != 5:
                if strict_empirical_required:
                    raise ModelDataError(
                        reason_code="stochastic_calibration_unavailable",
                        message=(
                            "Stochastic regime correlation matrix is missing/invalid "
                            f"for regime '{regime_id}' under strict runtime."
                        ),
                    )
                corr = [
                    (1.0, 0.45, 0.35, 0.20, 0.15),
                    (0.45, 1.0, 0.25, 0.30, 0.20),
                    (0.35, 0.25, 1.0, 0.28, 0.32),
                    (0.20, 0.30, 0.28, 1.0, 0.18),
                    (0.15, 0.20, 0.32, 0.18, 1.0),
                ]
            duration_mix_raw = item.get("duration_mix", [1.0, 1.0, 1.0])
            monetary_mix_raw = item.get("monetary_mix", [0.62, 0.38])
            emissions_mix_raw = item.get("emissions_mix", [0.72, 0.28])
            duration_mix = (1.0, 1.0, 1.0)
            monetary_mix = (0.62, 0.38)
            emissions_mix = (0.72, 0.28)
            if isinstance(duration_mix_raw, list) and len(duration_mix_raw) == 3:
                duration_mix = (
                    max(0.0, _safe_float(duration_mix_raw[0], 1.0)),
                    max(0.0, _safe_float(duration_mix_raw[1], 1.0)),
                    max(0.0, _safe_float(duration_mix_raw[2], 1.0)),
                )
            if isinstance(monetary_mix_raw, list) and len(monetary_mix_raw) == 2:
                monetary_mix = (
                    max(0.0, _safe_float(monetary_mix_raw[0], 0.62)),
                    max(0.0, _safe_float(monetary_mix_raw[1], 0.38)),
                )
            if isinstance(emissions_mix_raw, list) and len(emissions_mix_raw) == 2:
                emissions_mix = (
                    max(0.0, _safe_float(emissions_mix_raw[0], 0.72)),
                    max(0.0, _safe_float(emissions_mix_raw[1], 0.28)),
                )
            regimes[str(regime_id)] = StochasticRegime(
                regime_id=str(regime_id),
                sigma_scale=max(0.1, _safe_float(item.get("sigma_scale"), 1.0)),
                traffic_scale=max(0.1, _safe_float(item.get("traffic_scale"), 1.0)),
                incident_scale=max(0.1, _safe_float(item.get("incident_scale"), 1.0)),
                weather_scale=max(0.1, _safe_float(item.get("weather_scale"), 1.0)),
                price_scale=max(0.1, _safe_float(item.get("price_scale"), 1.0)),
                eco_scale=max(0.1, _safe_float(item.get("eco_scale"), 1.0)),
                corr=tuple(corr),
                spread_floor=max(0.01, _safe_float(item.get("spread_floor"), 0.05)),
                spread_cap=max(0.15, _safe_float(item.get("spread_cap"), 1.25)),
                factor_low=max(0.05, _safe_float(item.get("factor_low"), 0.55)),
                factor_high=max(1.0, _safe_float(item.get("factor_high"), 2.20)),
                duration_mix=duration_mix,
                monetary_mix=monetary_mix,
                emissions_mix=emissions_mix,
            )
    if not regimes:
        if rejected_synthetic:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Only synthetic stochastic calibration assets were available.",
            )
        raise ModelDataError(
            reason_code="stochastic_calibration_unavailable",
            message="No valid stochastic calibration regimes were available.",
        )
    as_of_utc: str | None = None
    if fallback_path is not None:
        fallback_as_of = _infer_as_of_from_payload(payload) or _infer_as_of_from_path(fallback_path)
        as_of_utc = fallback_as_of.isoformat()
        _raise_if_strict_stale(
            reason_code="stochastic_calibration_unavailable",
            message="Stochastic calibration fallback asset is stale for strict live-data policy.",
            as_of_utc=fallback_as_of,
            max_age_days=int(settings.live_stochastic_max_age_days),
            enforce=strict_freshness_required,
        )
    else:
        live_as_of = _infer_as_of_from_payload(payload)
        as_of_utc = live_as_of.isoformat() if live_as_of is not None else None
    return StochasticRegimeTable(
        copula_id=str(payload.get("copula_id", "gaussian_5x5_v2")),
        calibration_version=str(payload.get("calibration_version", "v2-uk")),
        regimes=regimes,
        source=source,
        as_of_utc=as_of_utc,
    )


@dataclass(frozen=True)
class StochasticResidualPrior:
    sigma_floor: float
    sample_count: int
    source: str
    calibration_version: str
    as_of_utc: str | None
    prior_id: str


@lru_cache(maxsize=256)
def load_stochastic_residual_prior(
    *,
    day_kind: str,
    road_bucket: str,
    weather_profile: str,
    vehicle_type: str | None,
    corridor_bucket: str | None = None,
    local_time_slot: str | None = None,
) -> StochasticResidualPrior:
    day = _canonical_day_kind(day_kind)
    road = (road_bucket or "mixed").strip().lower() or "mixed"
    weather = (weather_profile or "clear").strip().lower() or "clear"
    vehicle_bucket = _canonical_vehicle_bucket(vehicle_type)
    corridor = _canonical_corridor_bucket(corridor_bucket)
    slot = _canonical_local_time_slot(local_time_slot)

    path_candidates = [
        _model_asset_root() / "stochastic_residual_priors_uk.json",
        _assets_root() / "stochastic_residual_priors_uk.json",
    ]
    payload: dict[str, Any] | None = None
    source = ""
    selected_path: Path | None = None
    for path in path_candidates:
        if not path.exists():
            continue
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            payload = parsed
            source = str(path)
            selected_path = path
            break
    if payload is None:
        raise ModelDataError(
            reason_code="risk_prior_unavailable",
            message="Stochastic residual priors are required in strict runtime.",
        )
    _raise_if_strict_stale(
        reason_code="risk_prior_unavailable",
        message="Stochastic residual priors are stale for strict runtime policy.",
        as_of_utc=_infer_as_of_from_payload(payload) or (_infer_as_of_from_path(selected_path) if selected_path else None),
        max_age_days=int(settings.live_stochastic_max_age_days),
        enforce=True,
    )

    priors = payload.get("priors", {})
    if isinstance(priors, dict):
        exact_key = f"{corridor}_{day}_{slot}_{road}_{weather}_{vehicle_bucket}"
        prior = priors.get(exact_key)
        if isinstance(prior, dict):
            sigma_floor = max(0.005, min(0.75, _safe_float(prior.get("sigma_floor"), 0.0)))
            sample_count = max(1, int(_safe_float(prior.get("sample_count"), 0)))
            return StochasticResidualPrior(
                sigma_floor=sigma_floor,
                sample_count=sample_count,
                source=source,
                calibration_version=str(payload.get("calibration_version", "v2")),
                as_of_utc=str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None,
                prior_id=str(exact_key),
            )
    if isinstance(priors, list):
        indexed: dict[str, dict[str, Any]] = {}
        for row in priors:
            if not isinstance(row, dict):
                continue
            dk = _canonical_day_kind(str(row.get("day_kind", "weekday")))
            rb = str(row.get("road_bucket", "mixed")).strip().lower() or "mixed"
            wp = str(row.get("weather_profile", "clear")).strip().lower() or "clear"
            vb = _canonical_vehicle_bucket(str(row.get("vehicle_type", "default")))
            cb = _canonical_corridor_bucket(str(row.get("corridor_bucket", "uk_default")))
            slot_key = _canonical_local_time_slot(str(row.get("local_time_slot", "h12")))
            key = f"{cb}_{dk}_{slot_key}_{rb}_{wp}_{vb}"
            indexed[key] = row
        exact_key = f"{corridor}_{day}_{slot}_{road}_{weather}_{vehicle_bucket}"
        prior = indexed.get(exact_key)
        if prior is not None:
            sigma_floor = max(0.005, min(0.75, _safe_float(prior.get("sigma_floor"), 0.0)))
            sample_count = max(1, int(_safe_float(prior.get("sample_count"), 0)))
            return StochasticResidualPrior(
                sigma_floor=sigma_floor,
                sample_count=sample_count,
                source=source,
                calibration_version=str(payload.get("calibration_version", "v2")),
                as_of_utc=str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None,
                prior_id=str(prior.get("prior_id", exact_key)),
            )

    raise ModelDataError(
        reason_code="risk_prior_unavailable",
        message=(
            "No stochastic residual prior matched route context "
            f"(corridor={corridor}, day={day}, slot={slot}, road={road}, "
            f"weather={weather}, vehicle={vehicle_bucket})."
        ),
    )


@dataclass(frozen=True)
class FuelConsumptionCurve:
    baseline_l_per_100km: float
    idle_l_per_hour: float
    stop_go_gain: float
    grade_up_gain: float
    grade_down_relief: float
    temp_cold_gain: float
    speed_quadratic_gain: float


@dataclass(frozen=True)
class FuelConsumptionCalibration:
    source: str
    version: str
    as_of_utc: str | None
    curves: dict[str, FuelConsumptionCurve]


@lru_cache(maxsize=1)
def load_fuel_consumption_calibration() -> FuelConsumptionCalibration:
    required_buckets = {"default", "van", "rigid_hgv", "artic_hgv"}
    path_candidates = [
        _model_asset_root() / "fuel_consumption_curves_uk.json",
        _assets_root() / "fuel_consumption_curves_uk.json",
    ]
    source = ""
    version = ""
    as_of_utc: str | None = None
    curves: dict[str, FuelConsumptionCurve] = {}
    for path in path_candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        curves_raw = payload.get("curves", {})
        if not isinstance(curves_raw, dict):
            continue
        out_curves: dict[str, FuelConsumptionCurve] = {}
        for bucket, raw in curves_raw.items():
            if not isinstance(raw, dict):
                continue
            out_curves[str(bucket).strip().lower()] = FuelConsumptionCurve(
                baseline_l_per_100km=max(3.0, _safe_float(raw.get("baseline_l_per_100km"), 27.5)),
                idle_l_per_hour=max(0.2, _safe_float(raw.get("idle_l_per_hour"), 1.6)),
                stop_go_gain=max(0.0, _safe_float(raw.get("stop_go_gain"), 0.004)),
                grade_up_gain=max(0.0, _safe_float(raw.get("grade_up_gain"), 5.0)),
                grade_down_relief=max(0.0, _safe_float(raw.get("grade_down_relief"), 0.45)),
                temp_cold_gain=max(0.0, _safe_float(raw.get("temp_cold_gain"), 0.008)),
                speed_quadratic_gain=max(0.0001, _safe_float(raw.get("speed_quadratic_gain"), 0.00055)),
            )
        if out_curves:
            source = str(path)
            version = str(payload.get("version", payload.get("calibration_version", "unknown")))
            as_of_utc = str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None
            curves = out_curves
            break
    if not curves:
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel consumption calibration asset is required in strict runtime.",
        )
    missing_required = sorted(required_buckets - set(curves.keys()))
    if missing_required:
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message=(
                "Fuel consumption calibration asset is missing required buckets: "
                + ", ".join(missing_required)
            ),
        )
    return FuelConsumptionCalibration(
        source=source,
        version=version,
        as_of_utc=as_of_utc,
        curves=curves,
    )
