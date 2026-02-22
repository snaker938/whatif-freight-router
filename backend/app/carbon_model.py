from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .live_data_sources import live_carbon_schedule
from .model_data_errors import ModelDataError
from .settings import settings

try:
    UK_TZ = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:
    UK_TZ = UTC


@dataclass(frozen=True)
class CarbonPricingContext:
    price_per_kg: float
    source: str
    schedule_year: int
    scope_mode: str
    uncertainty_low: float
    uncertainty_high: float


def _strict_live_carbon_enforced() -> bool:
    return settings.live_runtime_data_enabled and settings.strict_live_data_required


def _strict_live_carbon_policy() -> tuple[bool, bool, bool, bool]:
    strict = _strict_live_carbon_enforced()
    pytest_bypass_enabled = os.environ.get("STRICT_RUNTIME_TEST_BYPASS", "0").strip() == "1"
    require_live_url = bool(settings.live_carbon_require_url_in_strict) if strict else False
    allow_fallback = (bool(settings.live_carbon_allow_signed_fallback) if strict else True) or pytest_bypass_enabled
    return strict, require_live_url, allow_fallback, pytest_bypass_enabled


_PROVENANCE_DISALLOWED_TERMS = (
    "synthetic",
    "heuristic",
    "legacy",
    "interpolated",
    "simulated",
    "wobble",
)


def _reject_non_empirical_payload(
    payload: dict[str, object],
    *,
    reason_code: str,
    source_label: str,
) -> None:
    strict, _require_live_url, _allow_fallback, _pytest_bypass_enabled = _strict_live_carbon_policy()
    if not strict:
        return
    basis = str(payload.get("calibration_basis", payload.get("basis", ""))).strip().lower()
    source = str(payload.get("source", source_label)).strip().lower()
    if any(term in basis for term in _PROVENANCE_DISALLOWED_TERMS):
        raise ModelDataError(
            reason_code=reason_code,
            message="Carbon payload provenance is not empirical enough for strict runtime policy.",
            details={"source": source_label, "calibration_basis": basis},
        )
    if any(term in source for term in _PROVENANCE_DISALLOWED_TERMS):
        raise ModelDataError(
            reason_code=reason_code,
            message="Carbon payload source label indicates non-empirical provenance in strict runtime.",
            details={"source": source_label, "payload_source": source},
        )


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
            parsed = parsed.replace(tzinfo=UTC)
        else:
            parsed = parsed.astimezone(UTC)
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


def _has_uncertainty_distribution(payload: dict[str, object]) -> bool:
    dist_raw = payload.get("uncertainty_distribution_by_year")
    if not isinstance(dist_raw, dict):
        return False
    for row in dist_raw.values():
        if not isinstance(row, dict):
            continue
        if all(key in row for key in ("p10", "p50", "p90")):
            try:
                p10 = float(row["p10"])
                p50 = float(row["p50"])
                p90 = float(row["p90"])
            except (TypeError, ValueError):
                continue
            if p10 >= 0 and p50 > 0 and p90 >= p50:
                return True
    return False


def _load_carbon_payload_for_source(source: str) -> dict[str, object] | None:
    if source != "live_runtime:carbon_schedule":
        return None
    live_payload = live_carbon_schedule()
    if isinstance(live_payload, dict):
        return live_payload
    return None


def _load_schedule_from_asset() -> tuple[dict[int, float], str]:
    strict, require_live_url, _allow_fallback, pytest_bypass_enabled = _strict_live_carbon_policy()
    scenario = (settings.carbon_policy_scenario or "central").strip().lower()
    if scenario not in ("central", "high", "low"):
        scenario = "central"
    live_failure: ModelDataError | None = None
    live_url = str(settings.live_carbon_schedule_url or "").strip()
    if strict and require_live_url and not live_url and not pytest_bypass_enabled:
        raise ModelDataError(
            reason_code="carbon_policy_unavailable",
            message="LIVE_CARBON_SCHEDULE_URL is required in strict runtime.",
        )
    if settings.live_runtime_data_enabled and live_url:
        live_payload = live_carbon_schedule()
        if isinstance(live_payload, dict):
            try:
                _reject_non_empirical_payload(
                    live_payload,
                    reason_code="carbon_policy_unavailable",
                    source_label="live_runtime:carbon_schedule",
                )
                schedule = _parse_schedule_payload(live_payload, scenario=scenario)
                live_as_of = _payload_as_of(live_payload)
                live_fresh = (
                    live_as_of is not None
                    and (datetime.now(UTC) - live_as_of).days <= int(settings.live_carbon_max_age_days)
                )
                if not schedule:
                    live_failure = ModelDataError(
                        reason_code="carbon_policy_unavailable",
                        message="Live carbon policy payload is missing schedule rows.",
                    )
                elif not live_fresh:
                    live_failure = ModelDataError(
                        reason_code="carbon_policy_unavailable",
                        message="Live carbon policy payload is stale for strict runtime policy.",
                        details={
                            "as_of_utc": live_as_of.isoformat() if live_as_of is not None else None,
                            "max_age_days": int(settings.live_carbon_max_age_days),
                        },
                    )
                elif not _has_uncertainty_distribution(live_payload):
                    live_failure = ModelDataError(
                        reason_code="carbon_policy_unavailable",
                        message="Live carbon policy payload is missing uncertainty distribution rows.",
                    )
                else:
                    return schedule, "live_runtime:carbon_schedule"
            except ModelDataError as exc:
                live_failure = exc
    if live_failure is not None:
        raise live_failure
    raise ModelDataError(
        reason_code="carbon_policy_unavailable",
        message="Live carbon schedule payload is unavailable in strict runtime.",
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
    p10_raw = row.get("p10")
    p50_raw = row.get("p50")
    p90_raw = row.get("p90")
    if not isinstance(p10_raw, (int, float, str)):
        return None
    if not isinstance(p50_raw, (int, float, str)):
        return None
    if not isinstance(p90_raw, (int, float, str)):
        return None
    try:
        p10 = float(p10_raw)
        p50 = float(p50_raw)
        p90 = float(p90_raw)
    except (TypeError, ValueError):
        return None
    if p50 <= 0:
        return None
    scale = max(0.01, scenario_price / p50)
    low = max(0.0, p10 * scale)
    high = max(low, p90 * scale)
    return low, high


def _load_ev_grid_intensity_from_asset() -> tuple[dict[str, list[float]], str]:
    strict, require_live_url, _allow_fallback, pytest_bypass_enabled = _strict_live_carbon_policy()
    live_failure: ModelDataError | None = None
    live_url = str(settings.live_carbon_schedule_url or "").strip()
    if strict and require_live_url and not live_url and not pytest_bypass_enabled:
        raise ModelDataError(
            reason_code="carbon_intensity_unavailable",
            message="LIVE_CARBON_SCHEDULE_URL is required in strict runtime.",
        )
    if settings.live_runtime_data_enabled and live_url:
        live_payload = live_carbon_schedule()
        if isinstance(live_payload, dict):
            try:
                _reject_non_empirical_payload(
                    live_payload,
                    reason_code="carbon_intensity_unavailable",
                    source_label="live_runtime:carbon_schedule",
                )
                rows = live_payload.get("ev_grid_intensity_kg_per_kwh_by_region")
                parsed: dict[str, list[float]] = {}
                if isinstance(rows, dict):
                    for region, values in rows.items():
                        if not isinstance(values, list):
                            continue
                        parsed_values = [max(0.05, float(v)) for v in values[:24]]
                        if len(parsed_values) == 24:
                            parsed[str(region)] = parsed_values
                live_as_of = _payload_as_of(live_payload)
                live_fresh = (
                    live_as_of is not None
                    and (datetime.now(UTC) - live_as_of).days <= int(settings.live_carbon_max_age_days)
                )
                if not parsed:
                    live_failure = ModelDataError(
                        reason_code="carbon_intensity_unavailable",
                        message="Live carbon intensity payload is missing regional hourly rows.",
                    )
                elif not live_fresh:
                    live_failure = ModelDataError(
                        reason_code="carbon_intensity_unavailable",
                        message="Live carbon intensity payload is stale for strict runtime policy.",
                        details={
                            "as_of_utc": live_as_of.isoformat() if live_as_of is not None else None,
                            "max_age_days": int(settings.live_carbon_max_age_days),
                        },
                    )
                else:
                    return parsed, "live_runtime:carbon_schedule"
            except ModelDataError as exc:
                live_failure = exc
    if live_failure is not None:
        raise live_failure
    raise ModelDataError(
        reason_code="carbon_intensity_unavailable",
        message="Live carbon intensity payload is unavailable in strict runtime.",
    )


def _load_non_ev_scope_factors() -> tuple[dict[str, dict[int, float]], str]:
    strict, require_live_url, _allow_fallback, pytest_bypass_enabled = _strict_live_carbon_policy()
    live_url = str(settings.live_carbon_schedule_url or "").strip()
    if strict and require_live_url and not live_url and not pytest_bypass_enabled:
        raise ModelDataError(
            reason_code="carbon_policy_unavailable",
            message="LIVE_CARBON_SCHEDULE_URL is required in strict runtime.",
        )
    live_failure: ModelDataError | None = None
    if settings.live_runtime_data_enabled and live_url:
        live_payload = live_carbon_schedule()
        if isinstance(live_payload, dict):
            try:
                _reject_non_empirical_payload(
                    live_payload,
                    reason_code="carbon_policy_unavailable",
                    source_label="live_runtime:carbon_schedule",
                )
                live_as_of = _payload_as_of(live_payload)
                live_fresh = (
                    live_as_of is not None
                    and (datetime.now(UTC) - live_as_of).days <= int(settings.live_carbon_max_age_days)
                )
                if not live_fresh:
                    live_failure = ModelDataError(
                        reason_code="carbon_policy_unavailable",
                        message="Live non-EV carbon scope factor payload is stale for strict runtime policy.",
                        details={
                            "as_of_utc": live_as_of.isoformat() if live_as_of is not None else None,
                            "max_age_days": int(settings.live_carbon_max_age_days),
                        },
                    )
                else:
                    raw = live_payload.get("non_ev_scope_factors")
                    out: dict[str, dict[int, float]] = {}
                    if isinstance(raw, dict):
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
                        return out, "live_runtime:carbon_schedule"
                    live_failure = ModelDataError(
                        reason_code="carbon_policy_unavailable",
                        message="Live non-EV carbon scope factor payload is missing scope rows.",
                    )
            except ModelDataError as exc:
                live_failure = exc
    if live_failure is not None:
        raise live_failure
    raise ModelDataError(
        reason_code="carbon_policy_unavailable",
        message="Live non-EV carbon scope factors are unavailable in strict runtime.",
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


def _interpolate_scope_factor(scope_rows: dict[int, float], year: int) -> float:
    if not scope_rows:
        return 1.0
    years = sorted(int(y) for y in scope_rows.keys())
    if year <= years[0]:
        return float(scope_rows[years[0]])
    if year >= years[-1]:
        return float(scope_rows[years[-1]])
    lower = years[0]
    upper = years[-1]
    for idx in range(1, len(years)):
        if year <= years[idx]:
            lower = years[idx - 1]
            upper = years[idx]
            break
    if upper <= lower:
        return float(scope_rows[lower])
    ratio = (float(year) - float(lower)) / float(upper - lower)
    lo = float(scope_rows[lower])
    hi = float(scope_rows[upper])
    return max(0.8, min(2.5, lo + ((hi - lo) * ratio)))


def _interpolate_schedule_price(schedule: dict[int, float], year: int) -> tuple[int, float]:
    years = sorted(int(y) for y in schedule.keys())
    if not years:
        return year, 0.0
    if year <= years[0]:
        return years[0], float(schedule[years[0]])
    if year >= years[-1]:
        return years[-1], float(schedule[years[-1]])
    lower = years[0]
    upper = years[-1]
    for idx in range(1, len(years)):
        if year <= years[idx]:
            lower = years[idx - 1]
            upper = years[idx]
            break
    if upper <= lower:
        return lower, float(schedule[lower])
    ratio = (float(year) - float(lower)) / float(upper - lower)
    lo = float(schedule[lower])
    hi = float(schedule[upper])
    return year, max(0.0, lo + ((hi - lo) * ratio))


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
            departure_time_utc.astimezone(UTC).year
            if departure_time_utc is not None
            else datetime.now(UTC).year
        )
        base_price = max(0.0, float(request_price_per_kg))
        _schedule, source = _load_schedule_from_asset()
        schedule_payload = _load_carbon_payload_for_source(source)
        uncertainty_band: tuple[float, float] | None = None
        if isinstance(schedule_payload, dict):
            uncertainty_band = _uncertainty_band_from_distribution(
                schedule_payload,
                year=year,
                scenario_price=base_price,
            )
        if uncertainty_band is None:
            raise ModelDataError(
                reason_code="carbon_policy_unavailable",
                message="Carbon override requires uncertainty distribution rows in strict runtime.",
            )
        override_low, override_high = uncertainty_band
        return CarbonPricingContext(
            price_per_kg=base_price,
            source="request_override",
            schedule_year=year,
            scope_mode=scope_mode,
            uncertainty_low=override_low,
            uncertainty_high=override_high,
        )
    schedule, source = _load_schedule_from_asset()
    year = (
        departure_time_utc.astimezone(UTC).year
        if departure_time_utc is not None
        else datetime.now(UTC).year
    )
    schedule_year, schedule_price = _interpolate_schedule_price(schedule, year)
    payload = _load_carbon_payload_for_source(source)
    band: tuple[float, float] | None = None
    if isinstance(payload, dict):
        band = _uncertainty_band_from_distribution(payload, year=year, scenario_price=schedule_price)
    if band is None:
        raise ModelDataError(
            reason_code="carbon_policy_unavailable",
            message="Carbon policy uncertainty distribution is missing for strict runtime.",
        )
    uncertainty_low, uncertainty_high = band
    return CarbonPricingContext(
        price_per_kg=schedule_price,
        source=source,
        schedule_year=schedule_year,
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
            hour = datetime.now(UTC).astimezone(UK_TZ).hour
        else:
            aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
                tzinfo=UTC
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
            departure_time_utc.astimezone(UTC).year
            if departure_time_utc is not None
            else datetime.now(UTC).year
        )
        if not scope_rows:
            raise ModelDataError(
                reason_code="carbon_policy_unavailable",
                message=f"No non-EV scope factors were available for scope '{scope}' in strict runtime.",
            )
        factor = _interpolate_scope_factor(scope_rows, int(year))
    return max(0.0, float(emissions_kg) * factor)
