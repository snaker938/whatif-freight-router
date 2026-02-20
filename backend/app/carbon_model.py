from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .live_data_sources import live_carbon_schedule
from .model_data_errors import ModelDataError
from .settings import settings

try:
    UK_TZ = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:
    UK_TZ = timezone.utc


@dataclass(frozen=True)
class CarbonPricingContext:
    price_per_kg: float
    source: str
    schedule_year: int
    scope_mode: str
    uncertainty_low: float
    uncertainty_high: float


def _is_fresh_path(path: Path, *, max_age_days: int) -> bool:
    age_seconds = datetime.now(timezone.utc).timestamp() - float(path.stat().st_mtime)
    return age_seconds <= float(max(1, int(max_age_days)) * 86400)


def _strict_live_carbon_enforced() -> bool:
    return settings.live_runtime_data_enabled and settings.strict_live_data_required


def _payload_as_of(payload: dict[str, object]) -> datetime | None:
    for key in ("as_of_utc", "as_of", "generated_at_utc", "updated_at_utc"):
        raw = str(payload.get(key, "")).strip()
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    return None


def _parse_schedule_payload(payload: dict[str, object], *, scenario: str) -> dict[int, float]:
    rows = payload.get("prices_gbp_per_kg", payload)
    if isinstance(rows, dict):
        scenario_rows = rows.get(scenario)
        if isinstance(scenario_rows, dict):
            rows = scenario_rows
        elif isinstance(rows.get("central"), dict):
            rows = rows.get("central")
    if not isinstance(rows, dict):
        return {}
    schedule: dict[int, float] = {}
    for year_raw, price_raw in rows.items():
        try:
            year = int(year_raw)
            price = max(0.0, float(price_raw))
        except (TypeError, ValueError):
            continue
        schedule[year] = price
    return schedule


def _parse_uncertainty_pct(payload: dict[str, object], *, scenario: str, year: int | None = None) -> float | None:
    if year is not None:
        dist_raw = payload.get("uncertainty_distribution_by_year")
        if isinstance(dist_raw, dict):
            row = dist_raw.get(str(int(year)))
            if isinstance(row, dict):
                try:
                    p10 = float(row.get("p10", 0.0))
                    p50 = float(row.get("p50", 0.0))
                    p90 = float(row.get("p90", 0.0))
                    if p50 > 0 and p90 >= p10:
                        pct = max(abs(p90 - p50), abs(p50 - p10)) / p50
                        return max(0.01, min(0.5, float(pct)))
                except (TypeError, ValueError):
                    pass
    raw = payload.get("uncertainty_pct_by_scenario")
    if not isinstance(raw, dict):
        return None
    value = raw.get(scenario)
    if value is None and scenario != "central":
        value = raw.get("central")
    if isinstance(value, dict) and year is not None:
        year_key = str(int(year))
        value = value.get(year_key, value.get("default"))
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.01, min(0.5, pct))


def _load_carbon_payload_for_source(source: str) -> dict[str, object] | None:
    if source == "live_runtime:carbon_schedule":
        live_payload = live_carbon_schedule()
        if isinstance(live_payload, dict):
            return live_payload
        return None
    path = Path(source)
    if not path.exists():
        return None
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(parsed, dict):
        return parsed
    return None


def _load_schedule_from_asset() -> tuple[dict[int, float], str, float]:
    scenario = (settings.carbon_policy_scenario or "central").strip().lower()
    if scenario not in ("central", "high", "low"):
        scenario = "central"
    live_missing_or_invalid = False
    if settings.live_runtime_data_enabled:
        live_payload = live_carbon_schedule()
        if isinstance(live_payload, dict):
            schedule = _parse_schedule_payload(live_payload, scenario=scenario)
            live_as_of = _payload_as_of(live_payload)
            live_fresh = (
                live_as_of is not None
                and (datetime.now(timezone.utc) - live_as_of).days <= int(settings.live_carbon_max_age_days)
            )
            if schedule and live_fresh:
                uncertainty_pct = _parse_uncertainty_pct(live_payload, scenario=scenario)
                if uncertainty_pct is None:
                    raise ModelDataError(
                        reason_code="carbon_policy_unavailable",
                        message="Live carbon policy payload is missing scenario uncertainty configuration.",
                    )
                return schedule, "live_runtime:carbon_schedule", uncertainty_pct
        live_missing_or_invalid = True

    candidates = [
        Path(settings.model_asset_dir) / "carbon_price_schedule_uk.json",
        Path(__file__).resolve().parents[1] / "assets" / "uk" / "carbon_price_schedule_uk.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        schedule = _parse_schedule_payload(payload, scenario=scenario)
        if schedule:
            as_of_dt = _payload_as_of(payload)
            is_fresh = (
                (datetime.now(timezone.utc) - as_of_dt).days <= int(settings.live_carbon_max_age_days)
                if as_of_dt is not None
                else _is_fresh_path(path, max_age_days=int(settings.live_carbon_max_age_days))
            )
            if _strict_live_carbon_enforced() and not is_fresh:
                raise ModelDataError(
                    reason_code="carbon_policy_unavailable",
                    message="Carbon policy fallback asset is stale for strict live-data policy.",
                )
            uncertainty_pct = _parse_uncertainty_pct(payload, scenario=scenario)
            if uncertainty_pct is None and _strict_live_carbon_enforced():
                raise ModelDataError(
                    reason_code="carbon_policy_unavailable",
                    message=(
                        f"Carbon policy asset '{path}' is missing uncertainty distribution configuration "
                        "required in strict runtime."
                    ),
                )
            if uncertainty_pct is None:
                uncertainty_pct = 0.08
            return schedule, str(path), uncertainty_pct
    raise ModelDataError(
        reason_code="carbon_policy_unavailable",
        message=(
            "Live carbon schedule payload was unavailable/invalid and no fresh fallback asset was found."
            if live_missing_or_invalid
            else "No valid carbon schedule asset was available for strict routing policy."
        ),
    )


def _uncertainty_band_from_distribution(
    payload: dict[str, object],
    *,
    year: int,
    scenario_price: float,
) -> tuple[float, float] | None:
    dist_raw = payload.get("uncertainty_distribution_by_year")
    if not isinstance(dist_raw, dict):
        return None
    row = dist_raw.get(str(int(year)))
    if not isinstance(row, dict):
        return None
    try:
        p10 = float(row.get("p10"))
        p50 = float(row.get("p50"))
        p90 = float(row.get("p90"))
    except (TypeError, ValueError):
        return None
    if p50 <= 0:
        return None
    scale = max(0.01, scenario_price / p50)
    low = max(0.0, p10 * scale)
    high = max(low, p90 * scale)
    return low, high


def _load_ev_grid_intensity_from_asset() -> tuple[dict[str, list[float]], str]:
    live_missing_or_invalid = False
    if settings.live_runtime_data_enabled:
        live_payload = live_carbon_schedule()
        if isinstance(live_payload, dict):
            rows = live_payload.get("ev_grid_intensity_kg_per_kwh_by_region")
            if isinstance(rows, dict):
                parsed: dict[str, list[float]] = {}
                for region, values in rows.items():
                    if not isinstance(values, list):
                        continue
                    parsed_values = [max(0.05, float(v)) for v in values[:24]]
                    if len(parsed_values) == 24:
                        parsed[str(region)] = parsed_values
                live_as_of = _payload_as_of(live_payload)
                live_fresh = (
                    live_as_of is not None
                    and (datetime.now(timezone.utc) - live_as_of).days <= int(settings.live_carbon_max_age_days)
                )
                if parsed and live_fresh:
                    return parsed, "live_runtime:carbon_schedule"
        live_missing_or_invalid = True

    candidates = [
        Path(settings.model_asset_dir) / "carbon_intensity_hourly_uk.json",
        Path(__file__).resolve().parents[1] / "assets" / "uk" / "carbon_intensity_hourly_uk.json",
        Path(settings.model_asset_dir) / "carbon_price_schedule_uk.json",
        Path(__file__).resolve().parents[1] / "assets" / "uk" / "carbon_price_schedule_uk.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        rows = payload.get("regions", payload.get("ev_grid_intensity_kg_per_kwh_by_region"))
        if not isinstance(rows, dict):
            continue
        parsed: dict[str, list[float]] = {}
        for region, values in rows.items():
            if not isinstance(values, list):
                continue
            parsed_values = [max(0.05, float(v)) for v in values[:24]]
            if len(parsed_values) == 24:
                parsed[str(region)] = parsed_values
        if parsed:
            as_of_dt = _payload_as_of(payload)
            is_fresh = (
                (datetime.now(timezone.utc) - as_of_dt).days <= int(settings.live_carbon_max_age_days)
                if as_of_dt is not None
                else _is_fresh_path(path, max_age_days=int(settings.live_carbon_max_age_days))
            )
            if _strict_live_carbon_enforced() and not is_fresh:
                raise ModelDataError(
                    reason_code="carbon_intensity_unavailable",
                    message="Carbon intensity fallback asset is stale for strict live-data policy.",
                )
            return parsed, str(path)
    raise ModelDataError(
        reason_code="carbon_intensity_unavailable",
        message=(
            "Live carbon intensity payload was unavailable/invalid and no fresh fallback asset was found."
            if live_missing_or_invalid
            else "No valid carbon intensity asset was available for strict routing policy."
        ),
    )


def _load_non_ev_scope_factors() -> tuple[dict[str, dict[int, float]], str]:
    candidates = [
        ("live_runtime:carbon_schedule", live_carbon_schedule() if settings.live_runtime_data_enabled else None),
        (
            str(Path(settings.model_asset_dir) / "carbon_price_schedule_uk.json"),
            None,
        ),
        (
            str(Path(__file__).resolve().parents[1] / "assets" / "uk" / "carbon_price_schedule_uk.json"),
            None,
        ),
    ]
    for source, payload in candidates:
        if payload is None and source != "live_runtime:carbon_schedule":
            path = Path(source)
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if source != "live_runtime:carbon_schedule":
            path = Path(source)
            as_of_dt = _payload_as_of(payload)
            is_fresh = (
                (datetime.now(timezone.utc) - as_of_dt).days <= int(settings.live_carbon_max_age_days)
                if as_of_dt is not None
                else _is_fresh_path(path, max_age_days=int(settings.live_carbon_max_age_days))
            )
            if _strict_live_carbon_enforced() and not is_fresh:
                raise ModelDataError(
                    reason_code="carbon_policy_unavailable",
                    message="Carbon scope-factor fallback asset is stale for strict live-data policy.",
                )
        raw = payload.get("non_ev_scope_factors")
        if not isinstance(raw, dict):
            continue
        out: dict[str, dict[int, float]] = {}
        for scope_key, rows in raw.items():
            if not isinstance(rows, dict):
                continue
            series: dict[int, float] = {}
            for yr, val in rows.items():
                try:
                    year = int(yr)
                    factor = float(val)
                except (TypeError, ValueError):
                    continue
                series[year] = max(0.8, min(2.5, factor))
            if series:
                out[str(scope_key).strip().lower()] = series
        if out:
            return out, source
    raise ModelDataError(
        reason_code="carbon_policy_unavailable",
        message="No valid non-EV carbon scope factor table is available for strict routing policy.",
    )


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


def resolve_carbon_price(
    *,
    request_price_per_kg: float,
    departure_time_utc: datetime | None,
) -> CarbonPricingContext:
    scope_mode = (settings.carbon_scope_mode or "ttw").strip().lower()
    if scope_mode not in ("ttw", "wtw", "lca"):
        scope_mode = "ttw"
    if request_price_per_kg > 0:
        year = (
            departure_time_utc.astimezone(timezone.utc).year
            if departure_time_utc is not None
            else datetime.now(timezone.utc).year
        )
        base_price = max(0.0, float(request_price_per_kg))
        _schedule, _source, _base_uncertainty = _load_schedule_from_asset()
        schedule_payload = _load_carbon_payload_for_source(
            str(Path(settings.model_asset_dir) / "carbon_price_schedule_uk.json")
        ) or _load_carbon_payload_for_source(
            str(Path(__file__).resolve().parents[1] / "assets" / "uk" / "carbon_price_schedule_uk.json")
        )
        uncertainty_band: tuple[float, float] | None = None
        if isinstance(schedule_payload, dict):
            uncertainty_band = _uncertainty_band_from_distribution(
                schedule_payload,
                year=year,
                scenario_price=base_price,
            )
        if uncertainty_band is None:
            if _strict_live_carbon_enforced():
                raise ModelDataError(
                    reason_code="carbon_policy_unavailable",
                    message="Carbon override requires uncertainty distribution rows in strict runtime.",
                )
            override_low = max(0.0, base_price * 0.92)
            override_high = max(override_low, base_price * 1.08)
        else:
            override_low, override_high = uncertainty_band
        return CarbonPricingContext(
            price_per_kg=base_price,
            source="request_override",
            schedule_year=year,
            scope_mode=scope_mode,
            uncertainty_low=override_low,
            uncertainty_high=override_high,
        )
    schedule, source, uncertainty_pct_base = _load_schedule_from_asset()
    year = (
        departure_time_utc.astimezone(timezone.utc).year
        if departure_time_utc is not None
        else datetime.now(timezone.utc).year
    )
    if year not in schedule:
        nearest = min(schedule.keys(), key=lambda y: abs(y - year))
        year = nearest
    schedule_price = float(schedule[year])
    payload = _load_carbon_payload_for_source(source)
    band: tuple[float, float] | None = None
    if isinstance(payload, dict):
        band = _uncertainty_band_from_distribution(payload, year=year, scenario_price=schedule_price)
    if band is None:
        if _strict_live_carbon_enforced():
            raise ModelDataError(
                reason_code="carbon_policy_unavailable",
                message="Carbon policy uncertainty distribution is missing for strict runtime.",
            )
        uncertainty_pct = max(0.01, float(uncertainty_pct_base))
        band = (
            max(0.0, schedule_price * (1.0 - uncertainty_pct)),
            max(0.0, schedule_price * (1.0 + uncertainty_pct)),
        )
    uncertainty_low, uncertainty_high = band
    return CarbonPricingContext(
        price_per_kg=schedule_price,
        source=source,
        schedule_year=year,
        scope_mode=scope_mode,
        uncertainty_low=uncertainty_low,
        uncertainty_high=uncertainty_high,
    )


def apply_scope_emissions_adjustment(
    *,
    emissions_kg: float,
    is_ev_mode: bool,
    scope_mode: str,
    departure_time_utc: datetime | None = None,
    route_centroid_lat: float | None = None,
    route_centroid_lon: float | None = None,
) -> float:
    scope = (scope_mode or "ttw").strip().lower()
    if scope == "ttw":
        return max(0.0, float(emissions_kg))
    # UK-first WTW/LCA adjustments with EV regional-hourly grid modulation.
    if is_ev_mode:
        profiles, _source = _load_ev_grid_intensity_from_asset()
        region = _region_bucket(route_centroid_lat, route_centroid_lon)
        profile = profiles.get(region) or profiles.get("uk_default")
        if profile is None:
            raise ModelDataError(
                reason_code="carbon_intensity_unavailable",
                message="No EV regional intensity profile matched route context in strict runtime.",
            )
        if departure_time_utc is None:
            hour = datetime.now(timezone.utc).astimezone(UK_TZ).hour
        else:
            aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
                tzinfo=timezone.utc
            )
            hour = aware.astimezone(UK_TZ).hour
        grid_intensity = profile[int(hour) % 24]
        # TTW baseline in current EV path is around 0.20 kg/kWh.
        wtw_factor = max(0.85, min(1.45, grid_intensity / 0.20))
        if scope == "lca":
            # LCA includes upstream generation + embedded lifecycle components.
            factor = max(0.95, min(1.75, wtw_factor * 1.18))
        else:
            factor = wtw_factor
    else:
        factors, _source = _load_non_ev_scope_factors()
        scope_rows = factors.get(scope) or {}
        year = (
            departure_time_utc.astimezone(timezone.utc).year
            if departure_time_utc is not None
            else datetime.now(timezone.utc).year
        )
        if not scope_rows:
            raise ModelDataError(
                reason_code="carbon_policy_unavailable",
                message=f"No non-EV scope factors were available for scope '{scope}' in strict runtime.",
            )
        nearest_year = min(scope_rows.keys(), key=lambda y: abs(int(y) - int(year)))
        factor = scope_rows[nearest_year]
    return max(0.0, float(emissions_kg) * factor)
