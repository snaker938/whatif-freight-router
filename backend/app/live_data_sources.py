from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .settings import settings


@dataclass
class _CacheEntry:
    fetched_at_s: float
    payload: Any


_CACHE: dict[str, _CacheEntry] = {}


def _fresh(entry: _CacheEntry, *, ttl_s: int) -> bool:
    return (time.time() - entry.fetched_at_s) <= max(1, int(ttl_s))


def _cache_get(key: str) -> _CacheEntry | None:
    return _CACHE.get(key)


def _cache_put(key: str, payload: Any) -> None:
    _CACHE[key] = _CacheEntry(fetched_at_s=time.time(), payload=payload)


def _strict_or_none(
    *,
    reason_code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    # Strict policy is resolved by calibration loaders after evaluating validated
    # local fallback freshness, so live source helpers should not hard-fail here.
    _ = reason_code
    _ = message
    _ = details


def _fetch_json(
    *,
    key: str,
    url: str,
    headers: dict[str, str] | None = None,
    reason_code_unavailable: str,
    reason_code_auth: str | None = None,
    require_auth_token: str | None = None,
) -> Any | None:
    if not settings.live_runtime_data_enabled:
        return None

    cached = _cache_get(key)
    if cached is not None and _fresh(cached, ttl_s=settings.live_data_cache_ttl_s):
        return cached.payload

    if require_auth_token is not None and not require_auth_token.strip():
        return None

    merged_headers: dict[str, str] = {}
    if headers:
        merged_headers.update(headers)
    if require_auth_token is not None and require_auth_token.strip():
        merged_headers["Authorization"] = f"Bearer {require_auth_token.strip()}"

    try:
        with httpx.Client(timeout=max(2.0, float(settings.live_data_request_timeout_s))) as client:
            resp = client.get(url, headers=merged_headers)
            if resp.status_code in (401, 403):
                _strict_or_none(
                    reason_code=reason_code_auth or reason_code_unavailable,
                    message=f"Authentication failed for live source {key}.",
                    details={"status_code": resp.status_code, "url": url},
                )
                return None
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        # If we have a stale cache entry, return it and let loader freshness
        # policies decide whether it is acceptable.
        stale = _cache_get(key)
        if stale is not None:
            return stale.payload
        _strict_or_none(
            reason_code=reason_code_unavailable,
            message=f"Live source unavailable for {key}: {type(exc).__name__}",
            details={"url": url},
        )
        return None

    _cache_put(key, payload)
    return payload


def live_bank_holidays() -> dict[str, Any] | None:
    return _fetch_json(
        key="bank_holidays",
        url=settings.live_bank_holidays_url,
        reason_code_unavailable="holiday_data_unavailable",
    )


def live_departure_profiles() -> dict[str, Any] | None:
    if not settings.live_departure_profile_url.strip():
        return None
    return _fetch_json(
        key="departure_profiles",
        url=settings.live_departure_profile_url.strip(),
        reason_code_unavailable="departure_profile_unavailable",
    )


def live_stochastic_regimes() -> dict[str, Any] | None:
    if not settings.live_stochastic_regimes_url.strip():
        return None
    return _fetch_json(
        key="stochastic_regimes",
        url=settings.live_stochastic_regimes_url.strip(),
        reason_code_unavailable="stochastic_calibration_unavailable",
    )


def live_toll_topology() -> dict[str, Any] | None:
    if not settings.live_toll_topology_url.strip():
        return None
    return _fetch_json(
        key="toll_topology",
        url=settings.live_toll_topology_url.strip(),
        reason_code_unavailable="toll_topology_unavailable",
    )


def live_toll_tariffs() -> dict[str, Any] | None:
    if not settings.live_toll_tariffs_url.strip():
        return None
    return _fetch_json(
        key="toll_tariffs",
        url=settings.live_toll_tariffs_url.strip(),
        reason_code_unavailable="toll_tariff_unavailable",
    )


def live_fuel_prices(as_of_utc: datetime | None) -> dict[str, Any] | None:
    if not settings.live_fuel_price_url.strip():
        return None
    token = settings.live_fuel_auth_token
    dt = (
        as_of_utc.astimezone(timezone.utc)
        if as_of_utc is not None and as_of_utc.tzinfo is not None
        else as_of_utc.replace(tzinfo=timezone.utc)
        if as_of_utc is not None
        else datetime.now(timezone.utc)
    )
    yyyymm = dt.strftime("%Y-%m")
    sep = "&" if "?" in settings.live_fuel_price_url else "?"
    url = f"{settings.live_fuel_price_url}{sep}month={yyyymm}"
    return _fetch_json(
        key=f"fuel_prices:{yyyymm}",
        url=url,
        reason_code_unavailable="fuel_price_source_unavailable",
        reason_code_auth="fuel_price_auth_unavailable",
        require_auth_token=token,
    )


def live_carbon_schedule() -> dict[str, Any] | None:
    if not settings.live_carbon_schedule_url.strip():
        return None
    return _fetch_json(
        key="carbon_schedule",
        url=settings.live_carbon_schedule_url.strip(),
        reason_code_unavailable="carbon_policy_unavailable",
    )
