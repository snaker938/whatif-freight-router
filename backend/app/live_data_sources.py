from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx

from .settings import settings


@dataclass
class _CacheEntry:
    fetched_at_s: float
    payload: Any


@dataclass
class _RetryResult:
    payload: Any | None
    status_code: int | None
    attempt_count: int
    retry_count: int
    retry_total_backoff_ms: int
    last_error_name: str | None
    last_error_status: int | None
    deadline_exceeded: bool


_CACHE: dict[str, _CacheEntry] = {}


def clear_live_data_source_cache() -> None:
    _CACHE.clear()


def _fresh(entry: _CacheEntry, *, ttl_s: int) -> bool:
    return (time.time() - entry.fetched_at_s) <= max(1, int(ttl_s))


def _cache_get(key: str) -> _CacheEntry | None:
    return _CACHE.get(key)


def _cache_put(key: str, payload: Any) -> None:
    _CACHE[key] = _CacheEntry(fetched_at_s=time.time(), payload=payload)


def _retryable_status_codes() -> set[int]:
    raw = str(settings.live_http_retryable_status_codes or "").strip()
    if not raw:
        return {429, 500, 502, 503, 504}
    parsed: set[int] = set()
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        try:
            code = int(part)
        except ValueError:
            continue
        if 100 <= code <= 599:
            parsed.add(code)
    return parsed or {429, 500, 502, 503, 504}


def _is_retryable_status(code: int) -> bool:
    return int(code) in _retryable_status_codes()


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        if response is None:
            return False
        return _is_retryable_status(int(response.status_code))
    network_error_type = getattr(httpx, "NetworkError", httpx.TransportError)
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, network_error_type)):
        return True
    return False


def _compute_backoff_ms(attempt_index: int) -> int:
    attempt = max(1, int(attempt_index))
    base_ms = max(0, int(settings.live_http_retry_backoff_base_ms))
    max_ms = max(base_ms, int(settings.live_http_retry_backoff_max_ms))
    jitter_ms = max(0, int(settings.live_http_retry_jitter_ms))
    bounded = min(max_ms, base_ms * (2 ** (attempt - 1)))
    if jitter_ms > 0:
        bounded += random.randint(0, jitter_ms)
    return int(min(max_ms, bounded))


def _parse_retry_after_ms(response_headers: Any) -> int | None:
    if response_headers is None:
        return None
    value = None
    try:
        value = response_headers.get("Retry-After")
    except Exception:
        value = None
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = float(text)
        if seconds >= 0.0:
            return int(seconds * 1000.0)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    delta_s = max(0.0, (parsed.astimezone(UTC) - _utc_now()).total_seconds())
    return int(delta_s * 1000.0)


def _extract_status_code(exc: Exception) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return int(exc.response.status_code)
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _request_json_with_bounded_retry(
    *,
    url: str,
    headers: dict[str, str] | None = None,
    deadline_at_monotonic_s: float | None = None,
) -> _RetryResult:
    max_attempts = max(1, int(settings.live_http_max_attempts))
    if deadline_at_monotonic_s is None:
        deadline_ms = max(1, int(settings.live_http_retry_deadline_ms))
        deadline_at_monotonic_s = time.monotonic() + (deadline_ms / 1000.0)
    attempt_count = 0
    retry_count = 0
    retry_total_backoff_ms = 0
    last_error_name: str | None = None
    last_error_status: int | None = None
    deadline_exceeded = False

    while attempt_count < max_attempts:
        remaining_s = deadline_at_monotonic_s - time.monotonic()
        if remaining_s <= 0.0:
            deadline_exceeded = True
            break
        request_timeout_s = min(float(settings.live_data_request_timeout_s), remaining_s)
        if request_timeout_s <= 0.0:
            deadline_exceeded = True
            break
        attempt_count += 1
        try:
            with httpx.Client(timeout=request_timeout_s) as client:
                response = client.get(url, headers=headers)
            status_code = int(response.status_code)
            if status_code >= 400:
                response.raise_for_status()
            if status_code == 204 or not bytes(response.content or b""):
                payload: Any = {}
            else:
                payload = response.json()
            return _RetryResult(
                payload=payload,
                status_code=status_code,
                attempt_count=attempt_count,
                retry_count=retry_count,
                retry_total_backoff_ms=retry_total_backoff_ms,
                last_error_name=None,
                last_error_status=None,
                deadline_exceeded=False,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary for HTTP stack
            last_error_name = type(exc).__name__
            last_error_status = _extract_status_code(exc)
            if not _is_retryable_exception(exc):
                break
            if attempt_count >= max_attempts:
                break
            remaining_s = deadline_at_monotonic_s - time.monotonic()
            if remaining_s <= 0.0:
                deadline_exceeded = True
                break
            backoff_ms = _compute_backoff_ms(retry_count + 1)
            retry_after_ms: int | None = None
            if settings.live_http_retry_respect_retry_after and isinstance(exc, httpx.HTTPStatusError):
                retry_after_ms = _parse_retry_after_ms(exc.response.headers if exc.response is not None else None)
            wait_ms = int(backoff_ms)
            if retry_after_ms is not None:
                wait_ms = int(min(max(0, retry_after_ms), int(settings.live_http_retry_backoff_max_ms)))
            remaining_ms = int(max(0.0, remaining_s * 1000.0))
            wait_ms = min(wait_ms, remaining_ms)
            retry_count += 1
            if wait_ms <= 0:
                deadline_exceeded = remaining_ms <= 0
                continue
            time.sleep(wait_ms / 1000.0)
            retry_total_backoff_ms += wait_ms

    return _RetryResult(
        payload=None,
        status_code=last_error_status,
        attempt_count=attempt_count,
        retry_count=retry_count,
        retry_total_backoff_ms=retry_total_backoff_ms,
        last_error_name=last_error_name,
        last_error_status=last_error_status,
        deadline_exceeded=deadline_exceeded,
    )


def _attach_retry_diagnostics(diagnostics: dict[str, Any], retry: _RetryResult) -> None:
    diagnostics["retry_attempts"] = int(retry.attempt_count)
    diagnostics["retry_count"] = int(retry.retry_count)
    diagnostics["retry_total_backoff_ms"] = int(retry.retry_total_backoff_ms)
    diagnostics["retry_last_error"] = retry.last_error_name
    diagnostics["retry_last_status_code"] = retry.last_error_status
    diagnostics["retry_deadline_exceeded"] = bool(retry.deadline_exceeded)


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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_as_of(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fuel_live_error(
    *,
    reason_code: str,
    message: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "_live_error": {
            "reason_code": reason_code,
            "message": message,
            "diagnostics": diagnostics,
        }
    }


def _scenario_live_error(
    *,
    reason_code: str,
    message: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "_live_error": {
            "reason_code": reason_code,
            "message": message,
            "diagnostics": diagnostics,
        }
    }


def _allowed_hosts(raw: str) -> set[str]:
    text = str(raw or "").strip()
    if not text:
        return set()
    if text == "*":
        return set()
    hosts = {item.strip().lower() for item in text.split(",") if item.strip()}
    return hosts


def _strict_live_runtime_required() -> bool:
    return bool(settings.live_runtime_data_enabled and settings.strict_live_data_required)


def _effective_allowed_hosts_raw(url: str, *, allowed_hosts_raw: str) -> str:
    raw = str(allowed_hosts_raw or "").strip()
    if raw:
        return raw
    # In strict runtime, never allow an unrestricted host policy; at minimum
    # constrain to the configured live URL host.
    if not _strict_live_runtime_required():
        return raw
    try:
        parsed = urlparse(url)
    except Exception:
        return raw
    host = (parsed.hostname or "").strip().lower()
    return host or raw


def _url_allowed(url: str, *, allowed_hosts_raw: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme.lower() != "https":
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    allowed_hosts = _allowed_hosts(
        _effective_allowed_hosts_raw(url, allowed_hosts_raw=allowed_hosts_raw)
    )
    return (not allowed_hosts) or (host in allowed_hosts)


def _scenario_url_allowed(url: str) -> bool:
    return _url_allowed(url, allowed_hosts_raw=str(settings.live_scenario_allowed_hosts or ""))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


_HOUR_TOKEN_RE = re.compile(r"\b([01]?\d|2[0-3])(?::[0-5]\d)?\b")


def _parse_hour_value(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        hour = int(value)
        if 0 <= hour <= 23:
            return hour
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return int(parsed.hour)
    except ValueError:
        pass
    if ":" in text:
        head = text.split(":", 1)[0].strip()
        if head.isdigit():
            hour = int(head)
            if 0 <= hour <= 23:
                return hour
    match = _HOUR_TOKEN_RE.search(text)
    if match:
        try:
            hour = int(match.group(1))
            if 0 <= hour <= 23:
                return hour
        except ValueError:
            return None
    return None


def _extract_hourly_values(payload: Any, keys: tuple[str, ...]) -> list[tuple[int | None, float]]:
    out: list[tuple[int | None, float]] = []
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            hour_raw = (
                item.get("hour")
                or item.get("hour_slot")
                or item.get("time_hour")
                or item.get("period")
                or item.get("time")
                or item.get("timestamp")
                or item.get("dateTime")
            )
            hour = _parse_hour_value(hour_raw)
            for key, value in item.items():
                lowered = str(key).strip().lower()
                if any(marker in lowered for marker in keys):
                    parsed = _safe_float(value, float("nan"))
                    if math.isfinite(parsed):
                        out.append((hour, float(parsed)))
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return out


def _fetch_json_with_ttl(
    *,
    key: str,
    url: str,
    ttl_s: int,
    allowed_hosts_csv: str = "",
    deadline_at_monotonic_s: float | None = None,
) -> tuple[Any | None, str | None]:
    def _annotate(
        payload: Any,
        *,
        cache_hit: bool,
        fetch_error: str | None,
        stale_cache_used: bool,
        status_code: int | None,
        retry: _RetryResult | None = None,
    ) -> Any:
        if not isinstance(payload, dict):
            return payload
        diagnostics = payload.setdefault("_live_diagnostics", {})
        if not isinstance(diagnostics, dict):
            return payload
        diagnostics["source_url"] = url
        diagnostics["cache_hit"] = bool(cache_hit)
        diagnostics["fetch_error"] = fetch_error
        diagnostics["stale_cache_used"] = bool(stale_cache_used)
        diagnostics["status_code"] = int(status_code) if status_code is not None else None
        diagnostics.setdefault("as_of_utc", _iso_utc(_utc_now()))
        if retry is not None:
            _attach_retry_diagnostics(diagnostics, retry)
        return payload

    if not _url_allowed(url, allowed_hosts_raw=allowed_hosts_csv):
        return None, "url_not_allowed"
    cached = _cache_get(key)
    if cached is not None and _fresh(cached, ttl_s=ttl_s):
        return _annotate(
            cached.payload,
            cache_hit=True,
            fetch_error=None,
            stale_cache_used=False,
            status_code=None,
            retry=_RetryResult(
                payload=None,
                status_code=None,
                attempt_count=0,
                retry_count=0,
                retry_total_backoff_ms=0,
                last_error_name=None,
                last_error_status=None,
                deadline_exceeded=False,
            ),
        ), None
    retry_result = _request_json_with_bounded_retry(
        url=url,
        headers=None,
        deadline_at_monotonic_s=deadline_at_monotonic_s,
    )
    if retry_result.payload is None:
        if retry_result.deadline_exceeded:
            error_name = "deadline_exceeded"
        else:
            error_name = retry_result.last_error_name or "source_unavailable"
        if cached is not None:
            return _annotate(
                cached.payload,
                cache_hit=True,
                fetch_error=error_name,
                stale_cache_used=True,
                status_code=retry_result.last_error_status,
                retry=retry_result,
            ), f"stale_cache:{error_name}"
        return None, error_name
    payload = retry_result.payload
    payload = _annotate(
        payload,
        cache_hit=False,
        fetch_error=None,
        stale_cache_used=False,
        status_code=retry_result.status_code,
        retry=retry_result,
    )
    _cache_put(key, payload)
    return payload, None


def _to_iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_weather_bucket(*, weather_code: int | None, precip_mm: float, wind_kph: float) -> str:
    if weather_code is not None:
        # Open-Meteo weather code families.
        if weather_code in {71, 73, 75, 77, 85, 86}:
            return "snow"
        if weather_code in {45, 48}:
            return "fog"
        if weather_code in {95, 96, 99}:
            return "storm"
        if weather_code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
            return "rain"
    if precip_mm >= 4.0 or wind_kph >= 45.0:
        return "storm"
    if precip_mm >= 0.4:
        return "rain"
    if wind_kph >= 28.0:
        return "fog"
    return "clear"


def _scenario_road_hint(route_context: dict[str, Any]) -> str:
    road_hint = str(route_context.get("road_hint", "")).strip().upper()
    if road_hint:
        return road_hint
    corridor = str(route_context.get("corridor_bucket", "")).strip().lower()
    if corridor in {"north_east_corridor", "scotland_south"}:
        return "A1"
    if corridor in {"north_west_corridor", "midlands_west"}:
        return "M6"
    if corridor in {"london_southeast", "south_england"}:
        return "M25"
    if corridor.startswith("wales"):
        return "M4"
    return "A1"


def _webtris_site_lat_lon(site: dict[str, Any]) -> tuple[float, float] | None:
    lat = _safe_float(site.get("Latitude", site.get("latitude")), float("nan"))
    lon = _safe_float(site.get("Longitude", site.get("longitude")), float("nan"))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def _nearest_webtris_site_id(sites_payload: Any, *, lat: float, lon: float) -> int | None:
    nearest = _nearest_webtris_sites(sites_payload, lat=lat, lon=lon, limit=1)
    if not nearest:
        return None
    return int(nearest[0]["site_id"])


def _nearest_webtris_sites(
    sites_payload: Any,
    *,
    lat: float,
    lon: float,
    limit: int = 4,
) -> list[dict[str, float | int]]:
    if isinstance(sites_payload, dict):
        candidates = sites_payload.get("sites", sites_payload.get("items", sites_payload.get("data", [])))
    else:
        candidates = sites_payload
    if not isinstance(candidates, list):
        return []
    out: list[dict[str, float | int]] = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("Status", raw.get("status", ""))).strip().lower()
        if status and status != "active":
            continue
        coords = _webtris_site_lat_lon(raw)
        if coords is None:
            continue
        site_id = raw.get("Id", raw.get("id", raw.get("SiteId", raw.get("site_id"))))
        if not isinstance(site_id, (int, float, str)):
            continue
        try:
            sid = int(site_id)
        except (TypeError, ValueError):
            continue
        d_lat = coords[0] - lat
        d_lon = coords[1] - lon
        dist_deg2 = (d_lat * d_lat) + (d_lon * d_lon)
        out.append(
            {
                "site_id": sid,
                "lat": float(coords[0]),
                "lon": float(coords[1]),
                "dist_deg2": float(dist_deg2),
            }
        )
    out.sort(key=lambda row: float(row.get("dist_deg2", 1e9)))
    return out[: max(1, int(limit))]


def _append_query(url: str, query: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{query}"


def _url_with_query(url: str, patch: dict[str, str]) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({str(k): str(v) for k, v in patch.items()})
    new_query = urlencode(query, doseq=True, safe="[]")
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


def _fetch_dft_rows_paginated(
    *,
    base_url: str,
    ttl_s: int,
    max_pages: int,
    query_patch: dict[str, str] | None = None,
    query_deadline_ms: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    rows: list[dict[str, Any]] = []
    pages_fetched = 0
    fetch_error: str | None = None
    page_diagnostics: list[dict[str, Any]] = []
    query_deadline_exceeded = False
    patch = {str(k): str(v) for k, v in (query_patch or {}).items() if str(k).strip() and str(v).strip()}
    next_url: str | None = _url_with_query(base_url, {"page[number]": "1", "page[size]": "100", **patch})
    page_number = 1
    deadline_ms = (
        max(1, int(query_deadline_ms))
        if query_deadline_ms is not None
        else max(1, int(settings.live_http_retry_deadline_ms))
    )
    query_deadline_at = time.monotonic() + (deadline_ms / 1000.0)
    while next_url and pages_fetched < max(1, int(max_pages)):
        if time.monotonic() >= query_deadline_at:
            fetch_error = "query_deadline_exceeded"
            query_deadline_exceeded = True
            break
        if not _scenario_url_allowed(next_url):
            fetch_error = "dft_url_not_allowed"
            break
        cache_key = f"scenario:dft:raw_counts:{page_number}:{hash(next_url)}"
        payload, page_err = _fetch_json_with_ttl(
            key=cache_key,
            url=next_url,
            ttl_s=ttl_s,
            deadline_at_monotonic_s=query_deadline_at,
        )
        if page_err in {"deadline_exceeded", "stale_cache:deadline_exceeded"}:
            if fetch_error is None:
                fetch_error = "query_deadline_exceeded"
            query_deadline_exceeded = True
        if page_err and fetch_error is None:
            fetch_error = page_err
        if not isinstance(payload, dict):
            break
        live_diag = payload.get("_live_diagnostics")
        if isinstance(live_diag, dict):
            page_diagnostics.append(
                {
                    "page_number": int(page_number),
                    "source_url": str(live_diag.get("source_url", next_url)),
                    "fetch_error": live_diag.get("fetch_error"),
                    "cache_hit": bool(live_diag.get("cache_hit", False)),
                    "stale_cache_used": bool(live_diag.get("stale_cache_used", False)),
                    "status_code": live_diag.get("status_code"),
                    "as_of_utc": live_diag.get("as_of_utc"),
                    "retry_attempts": _safe_int(live_diag.get("retry_attempts"), 0),
                    "retry_count": _safe_int(live_diag.get("retry_count"), 0),
                    "retry_total_backoff_ms": _safe_int(live_diag.get("retry_total_backoff_ms"), 0),
                    "retry_last_error": live_diag.get("retry_last_error"),
                    "retry_last_status_code": live_diag.get("retry_last_status_code"),
                    "retry_deadline_exceeded": bool(live_diag.get("retry_deadline_exceeded", False)),
                }
            )
        pages_fetched += 1
        maybe_rows = payload.get("data", payload.get("results", payload.get("items", [])))
        if isinstance(maybe_rows, list):
            rows.extend(row for row in maybe_rows if isinstance(row, dict))
        links = payload.get("links")
        next_raw: str | None = None
        if isinstance(links, dict):
            value = links.get("next")
            if isinstance(value, str) and value.strip():
                next_raw = value.strip()
        if next_raw is None:
            value = payload.get("next")
            if isinstance(value, str) and value.strip():
                next_raw = value.strip()
        if next_raw:
            next_url = next_raw if next_raw.startswith("http") else urljoin(next_url, next_raw)
        else:
            page_number += 1
            next_url = _url_with_query(
                base_url,
                {"page[number]": str(page_number), "page[size]": "100", **patch},
            )
            # Stop early when API has no further rows.
            if not maybe_rows:
                next_url = None
    return (
        rows,
        {
            "pages_fetched": int(pages_fetched),
            "row_count": int(len(rows)),
            "max_pages": int(max_pages),
            "query_deadline_ms": int(deadline_ms),
            "query_deadline_exceeded": bool(query_deadline_exceeded),
            "query_patch": patch,
            "page_diagnostics": page_diagnostics,
        },
        fetch_error,
    )


def _extract_numeric_series(payload: Any, keys: tuple[str, ...]) -> list[float]:
    out: list[float] = []
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                lowered = str(key).strip().lower()
                if any(marker in lowered for marker in keys):
                    parsed = _safe_float(value, float("nan"))
                    if math.isfinite(parsed):
                        out.append(parsed)
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return out


def _row_lat_lon(row: dict[str, Any]) -> tuple[float, float] | None:
    lat_keys = ("latitude", "lat", "site_lat", "count_point_lat")
    lon_keys = ("longitude", "lon", "lng", "site_lon", "count_point_lon")
    lat = float("nan")
    lon = float("nan")
    for key in lat_keys:
        if key in row:
            lat = _safe_float(row.get(key), float("nan"))
            if math.isfinite(lat):
                break
    for key in lon_keys:
        if key in row:
            lon = _safe_float(row.get(key), float("nan"))
            if math.isfinite(lon):
                break
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return (lat, lon)


def _row_day_kind(row: dict[str, Any]) -> str | None:
    raw = str(
        row.get("day_kind")
        or row.get("day_type")
        or row.get("day")
        or row.get("day_name")
        or ""
    ).strip().lower()
    if not raw:
        return None
    if raw in {"sat", "saturday", "sun", "sunday", "weekend"}:
        return "weekend"
    if raw in {"weekday", "mon", "monday", "tue", "tuesday", "wed", "wednesday", "thu", "thursday", "fri", "friday"}:
        return "weekday"
    return None


def _row_hour_slot(row: dict[str, Any]) -> int | None:
    for key in ("hour", "hour_slot", "hour_of_day", "time_hour"):
        if key in row:
            value = _safe_float(row.get(key), float("nan"))
            if math.isfinite(value):
                return int(max(0, min(23, int(value))))
    time_raw = str(row.get("time", "")).strip()
    if time_raw and ":" in time_raw:
        try:
            return int(max(0, min(23, int(time_raw.split(":")[0]))))
        except ValueError:
            return None
    return None


def _distance_weight_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = lat1 - lat2
    d_lon = lon1 - lon2
    dist = math.sqrt((d_lat * d_lat) + (d_lon * d_lon)) * 111.0
    return 1.0 / max(0.1, dist)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = lat1 - lat2
    d_lon = lon1 - lon2
    return math.sqrt((d_lat * d_lat) + (d_lon * d_lon)) * 111.0


def _scenario_dft_query_candidates(*, road_hint: str, now: datetime) -> list[dict[str, str]]:
    years = [int(now.year) - offset for offset in range(0, 4)]
    road = str(road_hint or "").strip().upper()
    roads: list[str] = []
    if road:
        roads.append(road)
        if road.startswith("M") and road[1:].isdigit():
            roads.append(f"A{road[1:]}")
        elif road.startswith("A") and road[1:].isdigit():
            roads.append(f"M{road[1:]}")
    candidates: list[dict[str, str]] = []
    for road_name in roads:
        for year in years:
            candidates.append(
                {
                    "filter[road_name]": str(road_name),
                    "filter[year]": str(year),
                }
            )
    # Keep year-only fallback, but only after road-targeted pulls.
    for year in years:
        candidates.append({"filter[year]": str(year)})
    unique: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for row in candidates:
        key = tuple(sorted((str(k), str(v)) for k, v in row.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _latest_history_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    history = payload.get("history")
    if not isinstance(history, list):
        return None
    latest_row: dict[str, Any] | None = None
    latest_dt: datetime | None = None
    for row in history:
        if not isinstance(row, dict):
            continue
        dt = _parse_as_of(row.get("as_of") or row.get("as_of_utc"))
        if dt is None:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_row = row
    return latest_row


def _normalize_fuel_payload(
    *,
    payload: Any,
    provider_url: str,
    cache_hit: bool,
    fetch_error: str | None = None,
    stale_cache_used: bool = False,
    status_code: int | None = None,
    retry: _RetryResult | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    diagnostics: dict[str, Any] = {
        "provider_url": provider_url,
        "source_url": provider_url,
        "fetched_at_utc": _iso_utc(now),
        "cache_hit": bool(cache_hit),
        "fetch_error": fetch_error,
        "stale_cache_used": bool(stale_cache_used),
        "status_code": int(status_code) if status_code is not None else None,
    }
    if retry is not None:
        _attach_retry_diagnostics(diagnostics, retry)

    if not isinstance(payload, dict):
        return _fuel_live_error(
            reason_code="fuel_price_source_unavailable",
            message="Live fuel payload was not a JSON object.",
            diagnostics=diagnostics,
        )

    latest_row = _latest_history_row(payload)
    prices_raw = payload.get("prices_gbp_per_l")
    if not isinstance(prices_raw, dict) and latest_row is not None:
        prices_raw = latest_row.get("prices_gbp_per_l")
    if not isinstance(prices_raw, dict):
        prices_raw = {}
    prices: dict[str, float] = {}
    for key, value in prices_raw.items():
        name = str(key).strip().lower()
        if not name:
            continue
        try:
            prices[name] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue

    grid_value = payload.get("grid_price_gbp_per_kwh")
    if grid_value is None and latest_row is not None:
        grid_value = latest_row.get("grid_price_gbp_per_kwh")
    if isinstance(grid_value, (int, float, str)):
        try:
            grid_price = max(0.0, float(grid_value))
        except (TypeError, ValueError):
            grid_price = -1.0
    else:
        grid_price = -1.0

    as_of_dt = (
        _parse_as_of(payload.get("as_of_utc"))
        or _parse_as_of(payload.get("as_of"))
        or _parse_as_of(payload.get("refreshed_at_utc"))
    )
    if as_of_dt is None and latest_row is not None:
        as_of_dt = _parse_as_of(latest_row.get("as_of_utc") or latest_row.get("as_of"))
    as_of_utc = _iso_utc(as_of_dt)
    diagnostics["as_of_utc"] = as_of_utc

    if as_of_dt is not None:
        age_seconds = max(0.0, (now - as_of_dt).total_seconds())
        diagnostics["age_seconds"] = round(age_seconds, 3)
        diagnostics["is_stale"] = age_seconds > (max(1, int(settings.live_fuel_max_age_days)) * 86400.0)
    else:
        diagnostics["is_stale"] = True

    regional_raw = payload.get("regional_multipliers", {})
    regional: dict[str, float] = {}
    if isinstance(regional_raw, dict):
        for key, value in regional_raw.items():
            try:
                regional[str(key).strip().lower()] = float(value)
            except (TypeError, ValueError):
                continue
    if "uk_default" not in regional:
        regional["uk_default"] = 1.0

    if not prices:
        return _fuel_live_error(
            reason_code="fuel_price_source_unavailable",
            message="Live fuel payload had no valid prices_gbp_per_l entries.",
            diagnostics=diagnostics,
        )
    if grid_price < 0.0:
        return _fuel_live_error(
            reason_code="fuel_price_source_unavailable",
            message="Live fuel payload missing grid_price_gbp_per_kwh.",
            diagnostics=diagnostics,
        )
    if as_of_utc is None:
        return _fuel_live_error(
            reason_code="fuel_price_source_unavailable",
            message="Live fuel payload missing as_of_utc/as_of timestamp.",
            diagnostics=diagnostics,
        )

    normalized: dict[str, Any] = {
        "as_of_utc": as_of_utc,
        "source": str(payload.get("source", "")).strip() or f"live:{provider_url}",
        "prices_gbp_per_l": prices,
        "grid_price_gbp_per_kwh": grid_price,
        "regional_multipliers": regional,
        "live_diagnostics": diagnostics,
    }
    signature = payload.get("signature")
    if signature is not None and str(signature).strip():
        normalized["signature"] = str(signature).strip()
    if "history" in payload and isinstance(payload.get("history"), list):
        normalized["history"] = payload["history"]
    return normalized


def _fetch_json(
    *,
    key: str,
    url: str,
    headers: dict[str, str] | None = None,
    reason_code_unavailable: str,
    reason_code_auth: str | None = None,
    require_auth_token: str | None = None,
    allowed_hosts_csv: str = "",
) -> Any | None:
    if not settings.live_runtime_data_enabled:
        return None

    def _annotate(
        payload: Any,
        *,
        cache_hit: bool,
        fetch_error: str | None,
        stale_cache_used: bool,
        status_code: int | None,
        retry: _RetryResult | None = None,
    ) -> Any:
        if not isinstance(payload, dict):
            return payload
        diagnostics = payload.setdefault("_live_diagnostics", {})
        if not isinstance(diagnostics, dict):
            return payload
        diagnostics["source_url"] = url
        diagnostics["cache_hit"] = bool(cache_hit)
        diagnostics["as_of_utc"] = _iso_utc(_utc_now())
        diagnostics["fetch_error"] = fetch_error
        diagnostics["stale_cache_used"] = bool(stale_cache_used)
        diagnostics["status_code"] = int(status_code) if status_code is not None else None
        if retry is not None:
            _attach_retry_diagnostics(diagnostics, retry)
        return payload

    if not _url_allowed(url, allowed_hosts_raw=allowed_hosts_csv):
        _strict_or_none(
            reason_code=reason_code_unavailable,
            message=f"Live source URL host/scheme is not allowed for {key}.",
            details={"url": url},
        )
        return None

    cached = _cache_get(key)
    if cached is not None and _fresh(cached, ttl_s=settings.live_data_cache_ttl_s):
        return _annotate(
            cached.payload,
            cache_hit=True,
            fetch_error=None,
            stale_cache_used=False,
            status_code=None,
            retry=_RetryResult(
                payload=None,
                status_code=None,
                attempt_count=0,
                retry_count=0,
                retry_total_backoff_ms=0,
                last_error_name=None,
                last_error_status=None,
                deadline_exceeded=False,
            ),
        )

    if require_auth_token is not None and not require_auth_token.strip():
        return None

    merged_headers: dict[str, str] = {}
    if headers:
        merged_headers.update(headers)
    if require_auth_token is not None and require_auth_token.strip():
        merged_headers["Authorization"] = f"Bearer {require_auth_token.strip()}"

    retry_result = _request_json_with_bounded_retry(
        url=url,
        headers=merged_headers,
        deadline_at_monotonic_s=None,
    )
    if retry_result.payload is None:
        status_code = retry_result.last_error_status
        if status_code in (401, 403):
            _strict_or_none(
                reason_code=reason_code_auth or reason_code_unavailable,
                message=f"Authentication failed for live source {key}.",
                details={"status_code": status_code, "url": url},
            )
            return None
        stale = _cache_get(key)
        if stale is not None:
            fetch_error = "deadline_exceeded" if retry_result.deadline_exceeded else retry_result.last_error_name
            return _annotate(
                stale.payload,
                cache_hit=True,
                fetch_error=fetch_error or "source_unavailable",
                stale_cache_used=True,
                status_code=status_code,
                retry=retry_result,
            )
        error_name = "deadline_exceeded" if retry_result.deadline_exceeded else (retry_result.last_error_name or "source_unavailable")
        _strict_or_none(
            reason_code=reason_code_unavailable,
            message=f"Live source unavailable for {key}: {error_name}",
            details={"url": url},
        )
        return None
    payload = retry_result.payload

    payload = _annotate(
        payload,
        cache_hit=False,
        fetch_error=None,
        stale_cache_used=False,
        status_code=retry_result.status_code,
        retry=retry_result,
    )
    _cache_put(key, payload)
    return payload


def live_bank_holidays() -> dict[str, Any] | None:
    payload = _fetch_json(
        key="bank_holidays",
        url=settings.live_bank_holidays_url,
        reason_code_unavailable="holiday_data_unavailable",
        allowed_hosts_csv=str(settings.live_bank_holidays_allowed_hosts or ""),
    )
    if not isinstance(payload, dict):
        return payload
    if str(payload.get("as_of_utc", "")).strip() or str(payload.get("as_of", "")).strip():
        return payload
    diag = payload.get("_live_diagnostics")
    if isinstance(diag, dict):
        diag_as_of = str(diag.get("as_of_utc", "")).strip()
        if diag_as_of:
            enriched = dict(payload)
            enriched["as_of_utc"] = diag_as_of
            return enriched
    return payload


def live_departure_profiles() -> dict[str, Any] | None:
    if not settings.live_departure_profile_url.strip():
        return None
    return _fetch_json(
        key="departure_profiles",
        url=settings.live_departure_profile_url.strip(),
        reason_code_unavailable="departure_profile_unavailable",
        allowed_hosts_csv=str(settings.live_departure_allowed_hosts or ""),
    )


def live_scenario_profiles() -> dict[str, Any] | None:
    coeff_url = str(settings.live_scenario_coefficient_url or "").strip()
    if not coeff_url:
        return None
    if not _scenario_url_allowed(coeff_url):
        return _scenario_live_error(
            reason_code="scenario_profile_invalid",
            message="Scenario coefficient URL host/scheme is not allowed.",
            diagnostics={"url": coeff_url},
        )
    payload, fetch_err = _fetch_json_with_ttl(
        key="scenario:coefficients",
        url=coeff_url,
        ttl_s=int(settings.live_scenario_cache_ttl_seconds),
    )
    if payload is None:
        return _scenario_live_error(
            reason_code="scenario_profile_unavailable",
            message="Scenario coefficient payload is unavailable from live source.",
            diagnostics={"url": coeff_url, "fetch_error": fetch_err},
        )
    if not isinstance(payload, dict):
        return _scenario_live_error(
            reason_code="scenario_profile_invalid",
            message="Scenario coefficient payload was not a JSON object.",
            diagnostics={"url": coeff_url, "fetch_error": fetch_err},
        )
    payload.setdefault("_live_source_url", coeff_url)
    if fetch_err is not None:
        payload.setdefault("_live_fetch_warning", fetch_err)
    return payload


def live_scenario_context(
    route_context: dict[str, Any],
    *,
    allow_partial_sources: bool = False,
) -> dict[str, Any] | None:
    if not settings.live_runtime_data_enabled:
        return None
    now = _utc_now()
    centroid_lat = _safe_float(route_context.get("centroid_lat"), float("nan"))
    centroid_lon = _safe_float(route_context.get("centroid_lon"), float("nan"))
    if not math.isfinite(centroid_lat) or not math.isfinite(centroid_lon):
        centroid_lat = 54.2
        centroid_lon = -2.3
    road_hint = _scenario_road_hint(route_context)
    cache_key = (
        "scenario_context:"
        f"{round(centroid_lat, 2)}:{round(centroid_lon, 2)}:"
        f"{road_hint}:{str(route_context.get('day_kind', 'weekday')).strip().lower()}"
    )
    cached = _cache_get(cache_key)
    if cached is not None and _fresh(cached, ttl_s=int(settings.live_scenario_cache_ttl_seconds)):
        if isinstance(cached.payload, dict):
            return cached.payload

    diagnostics: dict[str, Any] = {
        "fetched_at_utc": _to_iso_utc(now),
        "partial_source_mode": bool(allow_partial_sources),
        "route_context": {
            "centroid_lat": round(float(centroid_lat), 6),
            "centroid_lon": round(float(centroid_lon), 6),
            "road_hint": road_hint,
            "corridor_bucket": str(route_context.get("corridor_bucket", "uk_default")),
            "road_mix_bucket": str(route_context.get("road_mix_bucket", "mixed")),
            "day_kind": str(route_context.get("day_kind", "weekday")),
            "hour_slot_local": int(_safe_float(route_context.get("hour_slot_local"), float(now.hour))),
            "weather_bucket": str(route_context.get("weather_bucket", "clear")),
            "vehicle_class": str(route_context.get("vehicle_class", "rigid_hgv")),
        },
    }
    route_hour_local = int(
        max(0, min(23, _safe_int(route_context.get("hour_slot_local"), default=now.hour)))
    )
    day_webtris_candidates = [
        now.strftime("%d%m%Y"),
        (now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=365)).strftime("%d%m%Y"),
        (now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=730)).strftime("%d%m%Y"),
    ]

    def _diag_from_payload(payload: Any, *, fallback_url: str | None = None) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {
                "source_url": fallback_url,
                "fetch_error": None,
                "cache_hit": False,
                "stale_cache_used": False,
                "status_code": None,
                "as_of_utc": None,
                "retry_attempts": 0,
                "retry_count": 0,
                "retry_total_backoff_ms": 0,
                "retry_last_error": None,
                "retry_last_status_code": None,
                "retry_deadline_exceeded": False,
            }
        live_diag = payload.get("_live_diagnostics")
        if not isinstance(live_diag, dict):
            return {
                "source_url": fallback_url,
                "fetch_error": None,
                "cache_hit": False,
                "stale_cache_used": False,
                "status_code": None,
                "as_of_utc": None,
                "retry_attempts": 0,
                "retry_count": 0,
                "retry_total_backoff_ms": 0,
                "retry_last_error": None,
                "retry_last_status_code": None,
                "retry_deadline_exceeded": False,
            }
        return {
            "source_url": str(live_diag.get("source_url", fallback_url or "")).strip() or fallback_url,
            "fetch_error": live_diag.get("fetch_error"),
            "cache_hit": bool(live_diag.get("cache_hit", False)),
            "stale_cache_used": bool(live_diag.get("stale_cache_used", False)),
            "status_code": live_diag.get("status_code"),
            "as_of_utc": live_diag.get("as_of_utc"),
            "retry_attempts": _safe_int(live_diag.get("retry_attempts"), 0),
            "retry_count": _safe_int(live_diag.get("retry_count"), 0),
            "retry_total_backoff_ms": _safe_int(live_diag.get("retry_total_backoff_ms"), 0),
            "retry_last_error": live_diag.get("retry_last_error"),
            "retry_last_status_code": live_diag.get("retry_last_status_code"),
            "retry_deadline_exceeded": bool(live_diag.get("retry_deadline_exceeded", False)),
        }

    # Resolve URLs and enforce trusted-host policy for unsigned live payloads.
    webtris_sites_url = str(settings.live_scenario_webtris_sites_url or "").strip()
    webtris_daily_base = str(settings.live_scenario_webtris_daily_url or "").strip()
    traffic_template = str(settings.live_scenario_traffic_england_url or "").strip()
    dft_url = str(settings.live_scenario_dft_counts_url or "").strip()
    meteo_url_template = str(settings.live_scenario_open_meteo_forecast_url or "").strip()

    for label, url in (
        ("webtris_sites", webtris_sites_url),
        ("webtris_daily", webtris_daily_base),
        ("traffic_england", traffic_template.replace("{road}", road_hint)),
        ("dft_counts", dft_url),
        (
            "open_meteo",
            meteo_url_template.format(lat=f"{centroid_lat:.6f}", lon=f"{centroid_lon:.6f}"),
        ),
    ):
        if not url:
            return _scenario_live_error(
                reason_code="scenario_profile_unavailable",
                message=f"Scenario live source URL missing for {label}.",
                diagnostics={**diagnostics, "missing_source": label},
            )
        if not _scenario_url_allowed(url):
            return _scenario_live_error(
                reason_code="scenario_profile_invalid",
                message=f"Scenario live source URL host/scheme is not allowed for {label}.",
                diagnostics={**diagnostics, "url": url, "source": label},
            )

    # 1) WebTRIS (multi-station weighted extraction).
    webtris_sites, webtris_sites_err = _fetch_json_with_ttl(
        key="scenario:webtris:sites",
        url=webtris_sites_url,
        ttl_s=int(settings.live_scenario_cache_ttl_seconds),
    )
    nearest_sites = _nearest_webtris_sites(
        webtris_sites,
        lat=centroid_lat,
        lon=centroid_lon,
        limit=max(1, int(settings.live_scenario_webtris_nearest_sites)),
    )
    if not nearest_sites:
        return _scenario_live_error(
            reason_code="scenario_profile_unavailable",
            message="Scenario live context unavailable: nearest WebTRIS site could not be resolved.",
            diagnostics={**diagnostics, "webtris_error": webtris_sites_err},
        )
    webtris_flow_weighted_sum = 0.0
    webtris_flow_weight_sum = 0.0
    webtris_speed_weighted_sum = 0.0
    webtris_speed_weight_sum = 0.0
    webtris_daily_err: str | None = None
    webtris_used_sites: list[int] = []
    webtris_site_diagnostics: list[dict[str, float | int | str]] = []
    webtris_fetch_diagnostics: list[dict[str, Any]] = []
    for site in nearest_sites:
        site_id = int(site.get("site_id", 0))
        site_payload: Any = {}
        site_err: str | None = None
        site_flow_hourly: list[tuple[int | None, float]] = []
        site_speed_hourly: list[tuple[int | None, float]] = []
        selected_date_token = ""
        for day_webtris in day_webtris_candidates:
            webtris_daily_url = _append_query(
                webtris_daily_base,
                f"sites={site_id}&start_date={day_webtris}&end_date={day_webtris}&page=1&page_size=96",
            )
            if not _scenario_url_allowed(webtris_daily_url):
                continue
            payload_i, err_i = _fetch_json_with_ttl(
                key=f"scenario:webtris:daily:{site_id}:{day_webtris}",
                url=webtris_daily_url,
                ttl_s=int(settings.live_scenario_cache_ttl_seconds),
            )
            diag_i = _diag_from_payload(payload_i, fallback_url=webtris_daily_url)
            diag_i["site_id"] = int(site_id)
            diag_i["date_token"] = str(day_webtris)
            webtris_fetch_diagnostics.append(diag_i)
            if err_i and webtris_daily_err is None:
                webtris_daily_err = err_i
            flow_i = _extract_hourly_values(payload_i, ("flow", "volume", "count", "vehicles"))
            speed_i = _extract_hourly_values(payload_i, ("speed", "mph", "kph"))
            if flow_i or speed_i:
                site_payload = payload_i
                site_err = err_i
                site_flow_hourly = flow_i
                site_speed_hourly = speed_i
                selected_date_token = str(day_webtris)
                break
            site_payload = payload_i
            site_err = err_i
        if not selected_date_token:
            selected_date_token = day_webtris_candidates[0]

        site_fetch_diag = _diag_from_payload(
            site_payload,
            fallback_url=_append_query(
                webtris_daily_base,
                f"sites={site_id}&start_date={selected_date_token}&end_date={selected_date_token}&page=1&page_size=96",
            ),
        )
        site_fetch_diag["site_id"] = int(site_id)
        site_fetch_diag["date_token"] = selected_date_token
        webtris_fetch_diagnostics.append(site_fetch_diag)
        if site_err and webtris_daily_err is None:
            webtris_daily_err = site_err
        site_flow_vals = [v for _, v in site_flow_hourly]
        site_speed_vals = [v for _, v in site_speed_hourly]

        def _time_weighted_mean(samples: list[tuple[int | None, float]]) -> float:
            if not samples:
                return 0.0
            weighted_sum = 0.0
            weight_sum = 0.0
            for sample_hour, value in samples:
                if sample_hour is None:
                    hour_weight = 0.6
                else:
                    hour_delta = abs(int(sample_hour) - route_hour_local)
                    hour_weight = max(0.20, math.exp(-float(hour_delta) / 3.0))
                weighted_sum += float(value) * hour_weight
                weight_sum += hour_weight
            return max(0.0, weighted_sum / max(weight_sum, 1e-9))

        site_flow = _time_weighted_mean(site_flow_hourly)
        site_speed = _time_weighted_mean(site_speed_hourly)
        if site_flow <= 0.0 and site_flow_vals:
            site_flow = max(0.0, sum(site_flow_vals) / len(site_flow_vals))
        if site_speed <= 0.0 and site_speed_vals:
            site_speed = max(0.0, sum(site_speed_vals) / len(site_speed_vals))
        if site_flow <= 0.0 and site_speed <= 0.0:
            continue
        dist_deg2 = max(1e-8, float(site.get("dist_deg2", 1e-8)))
        weight = 1.0 / math.sqrt(dist_deg2)
        if site_flow > 0.0:
            webtris_flow_weighted_sum += site_flow * weight
            webtris_flow_weight_sum += weight
        if site_speed > 0.0:
            webtris_speed_weighted_sum += site_speed * weight
            webtris_speed_weight_sum += weight
        webtris_used_sites.append(site_id)
        webtris_site_diagnostics.append(
            {
                "site_id": site_id,
                "distance_weight": round(weight, 6),
                "flow": round(site_flow, 3),
                "speed_kph": round(site_speed, 3),
                "sample_count_flow": int(len(site_flow_hourly)),
                "sample_count_speed": int(len(site_speed_hourly)),
            }
        )
    webtris_flow = (
        max(0.0, webtris_flow_weighted_sum / webtris_flow_weight_sum)
        if webtris_flow_weight_sum > 0.0
        else 0.0
    )
    webtris_speed = (
        max(0.0, webtris_speed_weighted_sum / webtris_speed_weight_sum)
        if webtris_speed_weight_sum > 0.0
        else 0.0
    )
    nearest_site_id = webtris_used_sites[0] if webtris_used_sites else None

    # 2) Traffic England incidents/roadworks
    traffic_url = traffic_template.format(road=road_hint)
    traffic_payload, traffic_err = _fetch_json_with_ttl(
        key=f"scenario:trafficengland:{road_hint}",
        url=traffic_url,
        ttl_s=int(settings.live_scenario_cache_ttl_seconds),
    )
    traffic_fetch_diag = _diag_from_payload(traffic_payload, fallback_url=traffic_url)
    events = traffic_payload if isinstance(traffic_payload, list) else []
    if isinstance(traffic_payload, dict):
        maybe_events = traffic_payload.get("events", traffic_payload.get("items", traffic_payload.get("data", [])))
        if isinstance(maybe_events, list):
            events = maybe_events
    incident_count = 0
    roadworks_count = 0
    congestion_count = 0
    severity_total = 0.0
    severity_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(
            event.get("eventType")
            or event.get("type")
            or event.get("event_type")
            or ""
        ).strip().lower()
        severity_raw = str(event.get("severity") or event.get("severityLevel") or "").strip().lower()
        severity_score = 1.0
        if severity_raw in {"severe", "very high", "critical"}:
            severity_score = 3.0
        elif severity_raw in {"high", "major"}:
            severity_score = 2.0
        elif severity_raw in {"medium", "moderate"}:
            severity_score = 1.5
        severity_total += severity_score
        severity_count += 1
        if "incident" in event_type or "accident" in event_type:
            incident_count += 1
        elif "roadwork" in event_type:
            roadworks_count += 1
        elif "congestion" in event_type:
            congestion_count += 1
        else:
            congestion_count += 1

    # 3) DfT raw counts (paginated, road/year targeted first; weighted by distance/day-kind/hour/recency).
    dft_rows: list[dict[str, Any]] = []
    dft_err: str | None = None
    dft_query_meta: list[dict[str, Any]] = []
    if allow_partial_sources:
        # Corpus-building mode may continue without DfT when service availability is poor.
        dft_err = "partial_mode_dft_skipped"
    else:
        dft_candidates_meta = _scenario_dft_query_candidates(road_hint=road_hint, now=now)
        for query_idx, query_patch in enumerate(dft_candidates_meta):
            rows_i, meta_i, err_i = _fetch_dft_rows_paginated(
                base_url=dft_url,
                ttl_s=int(settings.live_scenario_cache_ttl_seconds),
                max_pages=int(settings.live_scenario_dft_max_pages),
                query_patch=query_patch,
                query_deadline_ms=int(settings.live_http_retry_deadline_ms),
            )
            geolocated_count_i = sum(1 for row in rows_i if isinstance(row, dict) and _row_lat_lon(row) is not None)
            dft_query_meta.append(
                {
                    "query_index": int(query_idx),
                    "query_patch": query_patch,
                    "row_count": int(len(rows_i)),
                    "geolocated_count": int(geolocated_count_i),
                    "fetch_error": err_i,
                    "page_meta": meta_i,
                }
            )
            if err_i and dft_err is None:
                dft_err = err_i
            if geolocated_count_i <= 0:
                continue
            dft_rows = rows_i
            # A road+year query with geolocated rows is preferred; no need to fan out.
            break
        if not dft_rows:
            # Final deterministic fallback to unfiltered year pages.
            rows_fallback, meta_fallback, err_fallback = _fetch_dft_rows_paginated(
                base_url=dft_url,
                ttl_s=int(settings.live_scenario_cache_ttl_seconds),
                max_pages=int(settings.live_scenario_dft_max_pages),
                query_patch={"filter[year]": str(int(now.year))},
                query_deadline_ms=int(settings.live_http_retry_deadline_ms),
            )
            if err_fallback and dft_err is None:
                dft_err = err_fallback
            dft_rows = rows_fallback
            dft_query_meta.append(
                {
                    "query_index": int(len(dft_query_meta)),
                    "query_patch": {"filter[year]": str(int(now.year))},
                    "row_count": int(len(rows_fallback)),
                    "geolocated_count": int(
                        sum(1 for row in rows_fallback if isinstance(row, dict) and _row_lat_lon(row) is not None)
                    ),
                    "fetch_error": err_fallback,
                    "page_meta": meta_fallback,
                }
            )
    selected_query_meta: dict[str, Any] = {}
    if dft_query_meta:
        scored = sorted(
            dft_query_meta,
            key=lambda row: (
                -int(_safe_int(row.get("geolocated_count"), 0)),
                -int(_safe_int(row.get("row_count"), 0)),
                int(_safe_int(row.get("query_index"), 0)),
            ),
        )
        selected_query_meta = dict(scored[0])
    dft_page_meta = {
        "query_candidates": dft_query_meta,
        "selected_query": selected_query_meta,
        "row_count": int(len(dft_rows)),
    }
    dft_weighted_sum = 0.0
    dft_weight_sum = 0.0
    dft_station_count = 0
    dft_candidate_count = len(dft_rows)
    max_dft_distance_km = max(1.0, float(settings.live_scenario_dft_max_distance_km))
    max_dft_rows = max(1, int(settings.live_scenario_dft_nearest_limit))
    dft_station_min_count = max(1, int(settings.live_scenario_dft_min_station_count))
    current_day_kind = str(route_context.get("day_kind", "weekday")).strip().lower() or "weekday"
    current_hour = route_hour_local
    day_utc = now.date()
    dft_candidates: list[tuple[float, float, float]] = []
    for row in dft_rows:
        row_count: float | None = None
        for key in (
            "all_motor_vehicles",
            "count",
            "flow",
            "volume",
            "vehicles",
            "count_value",
        ):
            if key in row:
                parsed = _safe_float(row.get(key), float("nan"))
                if math.isfinite(parsed):
                    row_count = parsed
                    break
        if row_count is None:
            continue
        coords = _row_lat_lon(row)
        if coords is None:
            continue
        distance_km = _distance_km(centroid_lat, centroid_lon, coords[0], coords[1])
        if distance_km > max_dft_distance_km:
            continue
        row_day = _row_day_kind(row)
        day_weight = 1.0 if (row_day is None or row_day == current_day_kind) else 0.60
        row_hour = _row_hour_slot(row)
        if row_hour is None:
            hour_weight = 0.75
        else:
            hour_delta = abs(int(row_hour) - current_hour)
            hour_weight = max(0.25, math.exp(-float(hour_delta) / 6.0))
        row_date_raw = str(row.get("count_date") or row.get("date") or row.get("as_of_utc") or "").strip()
        recency_weight = 1.0
        if row_date_raw:
            try:
                row_date = datetime.fromisoformat(row_date_raw.replace("Z", "+00:00")).date()
                recency_days = abs((day_utc - row_date).days)
                recency_weight = max(0.30, math.exp(-float(recency_days) / 14.0))
            except ValueError:
                recency_weight = 0.75
        context_weight = day_weight * hour_weight * recency_weight
        if context_weight <= 0.0:
            continue
        dft_candidates.append((distance_km, float(row_count), context_weight))
    dft_candidates.sort(key=lambda row: row[0])
    selected_dft_candidates = dft_candidates[:max_dft_rows]
    for distance_km, row_count, context_weight in selected_dft_candidates:
        distance_weight = 1.0 / max(0.1, float(distance_km))
        weight = distance_weight * context_weight
        if weight <= 0.0:
            continue
        dft_weighted_sum += float(row_count) * weight
        dft_weight_sum += weight
        dft_station_count += 1
    if not allow_partial_sources and (dft_station_count <= 0 or dft_weight_sum <= 0.0):
        return _scenario_live_error(
            reason_code="scenario_profile_unavailable",
            message="Scenario live context unavailable: no geolocated DfT stations matched route context.",
            diagnostics={
                **diagnostics,
                "dft_error": dft_err,
                "dft_page_meta": dft_page_meta,
                "dft_candidate_count": dft_candidate_count,
                "dft_candidate_filtered_count": int(len(dft_candidates)),
                "dft_selected_count": int(dft_station_count),
            },
        )
    if (not allow_partial_sources) and dft_station_count < dft_station_min_count:
        return _scenario_live_error(
            reason_code="scenario_profile_unavailable",
            message="Scenario live context unavailable: DfT corridor coverage below strict minimum station count.",
            diagnostics={
                **diagnostics,
                "dft_error": dft_err,
                "dft_page_meta": dft_page_meta,
                "dft_station_count": dft_station_count,
                "dft_min_station_count": int(dft_station_min_count),
                "dft_candidate_filtered_count": int(len(dft_candidates)),
                "dft_selected_count": int(dft_station_count),
            },
        )
    dft_count_per_hour = (
        max(0.0, dft_weighted_sum / dft_weight_sum)
        if dft_weight_sum > 0.0
        else 0.0
    )

    # 4) Open-Meteo
    meteo_url = meteo_url_template.format(lat=f"{centroid_lat:.6f}", lon=f"{centroid_lon:.6f}")
    meteo_payload, meteo_err = _fetch_json_with_ttl(
        key=f"scenario:meteo:{round(centroid_lat, 2)}:{round(centroid_lon, 2)}",
        url=meteo_url,
        ttl_s=int(settings.live_scenario_cache_ttl_seconds),
    )
    meteo_fetch_diag = _diag_from_payload(meteo_payload, fallback_url=meteo_url)
    current = meteo_payload.get("current", {}) if isinstance(meteo_payload, dict) else {}
    precip_mm = max(0.0, _safe_float(current.get("precipitation"), 0.0))
    wind_kph = max(0.0, _safe_float(current.get("wind_speed_10m"), 0.0))
    temp_c = _safe_float(current.get("temperature_2m"), 10.0)
    weather_code_raw = current.get("weather_code")
    weather_code: int | None
    if isinstance(weather_code_raw, (int, float, str)):
        try:
            weather_code = int(float(weather_code_raw))
        except (TypeError, ValueError):
            weather_code = None
    else:
        weather_code = None
    weather_bucket = _normalize_weather_bucket(
        weather_code=weather_code,
        precip_mm=precip_mm,
        wind_kph=wind_kph,
    )

    source_set = {
        "webtris": bool(webtris_used_sites),
        "traffic_england": bool(traffic_err is None and isinstance(traffic_payload, (dict, list))),
        "dft": dft_station_count > 0,
        "open_meteo": isinstance(current, dict) and bool(current),
    }
    source_ok_count = sum(1 for ok in source_set.values() if ok)
    coverage_overall = source_ok_count / 4.0
    if source_ok_count < 4 and not allow_partial_sources:
        return _scenario_live_error(
            reason_code="scenario_profile_unavailable",
            message="Scenario live context incomplete: one or more required live sources were unavailable.",
            diagnostics={
                **diagnostics,
                "source_set": source_set,
                "webtris_error": webtris_sites_err or webtris_daily_err,
                "traffic_england_error": traffic_err,
                "dft_error": dft_err,
                "open_meteo_error": meteo_err,
            },
        )

    # Keep live adapter output raw/derived feature vectors; pressure projection is
    # resolved downstream from artifact transform parameters.
    traffic_flow_index = max(0.0, float(webtris_flow))
    traffic_speed_index = max(0.0, float(webtris_speed))
    incident_delay_pressure = max(
        0.0,
        float(incident_count) + float(roadworks_count) + float(congestion_count),
    )
    severity_index = (float(severity_total) / max(1.0, float(severity_count))) if severity_count > 0 else 0.0
    weather_bucket_severity = {
        "clear": 0.0,
        "fog": 0.5,
        "rain": 1.0,
        "storm": 1.5,
        "snow": 1.75,
    }.get(str(weather_bucket).strip().lower(), 0.0)
    weather_severity_index = max(
        0.0,
        float(precip_mm) + (float(wind_kph) / 10.0) + float(weather_bucket_severity),
    )

    as_of_utc = _to_iso_utc(now)
    payload: dict[str, Any] = {
        "as_of_utc": as_of_utc,
        "source_set": {
            "webtris": webtris_sites_url,
            "traffic_england": traffic_url,
            "dft_counts": dft_url,
            "open_meteo": meteo_url,
        },
        "source_diagnostics": {
            "webtris": {
                "nearest_site_ids": [int(site_id) for site_id in webtris_used_sites],
                "site_count_used": int(len(webtris_used_sites)),
                "error": webtris_sites_err or webtris_daily_err,
                "fetch": _diag_from_payload(webtris_sites, fallback_url=webtris_sites_url),
                "daily_fetches": webtris_fetch_diagnostics,
            },
            "traffic_england": {
                "event_count": int(len(events)),
                "error": traffic_err,
                "fetch": traffic_fetch_diag,
            },
            "dft_counts": {
                "error": dft_err,
                "page_meta": dft_page_meta,
                "candidate_count": int(dft_candidate_count),
                "candidate_filtered_count": int(len(dft_candidates)),
                "selected_station_count": int(dft_station_count),
                "fetch": {
                    "source_url": dft_url,
                    "fetch_error": dft_err,
                    "cache_hit": any(
                        bool(page.get("cache_hit", False))
                        for row in (
                            dft_page_meta.get("query_candidates", [])
                            if isinstance(dft_page_meta, dict)
                            else []
                        )
                        if isinstance(row, dict)
                        for page in (
                            row.get("page_meta", {}).get("page_diagnostics", [])
                            if isinstance(row.get("page_meta"), dict)
                            else []
                        )
                        if isinstance(page, dict)
                    ),
                    "stale_cache_used": any(
                        bool(page.get("stale_cache_used", False))
                        for row in (
                            dft_page_meta.get("query_candidates", [])
                            if isinstance(dft_page_meta, dict)
                            else []
                        )
                        if isinstance(row, dict)
                        for page in (
                            row.get("page_meta", {}).get("page_diagnostics", [])
                            if isinstance(row.get("page_meta"), dict)
                            else []
                        )
                        if isinstance(page, dict)
                    ),
                    "status_code": None,
                    "as_of_utc": None,
                },
            },
            "open_meteo": {
                "error": meteo_err,
                "weather_code": weather_code,
                "fetch": meteo_fetch_diag,
            },
        },
        "traffic_features": {
            "flow_index": round(traffic_flow_index, 6),
            "speed_index": round(traffic_speed_index, 6),
            "webtris_flow": round(webtris_flow, 3),
            "webtris_speed_kph": round(webtris_speed, 3),
            "dft_count_per_hour": round(dft_count_per_hour, 3),
            "dft_station_count": int(dft_station_count),
            "dft_selected_count": int(dft_station_count),
            "dft_candidate_filtered_count": int(len(dft_candidates)),
            "dft_weight_sum": round(dft_weight_sum, 6),
            "nearest_site_id": int(nearest_site_id) if nearest_site_id is not None else None,
            "site_count_used": int(len(webtris_used_sites)),
            "site_aggregation": "inverse_distance_weighted",
            "site_diagnostics": webtris_site_diagnostics,
            "dft_page_meta": dft_page_meta,
        },
        "incident_features": {
            "event_count": int(len(events)),
            "incident_count": int(incident_count),
            "roadworks_count": int(roadworks_count),
            "congestion_count": int(congestion_count),
            "severity_index": round(severity_index, 6),
            "delay_pressure": round(incident_delay_pressure, 6),
        },
        "weather_features": {
            "weather_bucket": weather_bucket,
            "temperature_c": round(temp_c, 3),
            "precipitation_mm": round(precip_mm, 4),
            "wind_speed_kph": round(wind_kph, 3),
            "weather_code": weather_code,
            "weather_severity_index": round(weather_severity_index, 6),
        },
        "coverage": {
            "webtris": 1.0 if source_set["webtris"] else 0.0,
            "traffic_england": 1.0 if source_set["traffic_england"] else 0.0,
            "dft": 1.0 if source_set["dft"] else 0.0,
            "open_meteo": 1.0 if source_set["open_meteo"] else 0.0,
            "overall": round(coverage_overall, 6),
        },
        "partial_source_mode": bool(allow_partial_sources),
        "route_context_echo": diagnostics["route_context"],
    }
    _cache_put(cache_key, payload)
    return payload


def live_stochastic_regimes() -> dict[str, Any] | None:
    if not settings.live_stochastic_regimes_url.strip():
        return None
    return _fetch_json(
        key="stochastic_regimes",
        url=settings.live_stochastic_regimes_url.strip(),
        reason_code_unavailable="stochastic_calibration_unavailable",
        allowed_hosts_csv=str(settings.live_stochastic_allowed_hosts or ""),
    )


def live_toll_topology() -> dict[str, Any] | None:
    if not settings.live_toll_topology_url.strip():
        return None
    return _fetch_json(
        key="toll_topology",
        url=settings.live_toll_topology_url.strip(),
        reason_code_unavailable="toll_topology_unavailable",
        allowed_hosts_csv=str(settings.live_toll_allowed_hosts or ""),
    )


def live_toll_tariffs() -> dict[str, Any] | None:
    if not settings.live_toll_tariffs_url.strip():
        return None
    return _fetch_json(
        key="toll_tariffs",
        url=settings.live_toll_tariffs_url.strip(),
        reason_code_unavailable="toll_tariff_unavailable",
        allowed_hosts_csv=str(settings.live_toll_allowed_hosts or ""),
    )


def live_fuel_prices(as_of_utc: datetime | None) -> dict[str, Any] | None:
    if not settings.live_fuel_price_url.strip():
        return None
    token = settings.live_fuel_auth_token.strip()
    api_key = settings.live_fuel_api_key.strip()
    dt = (
        as_of_utc.astimezone(UTC)
        if as_of_utc is not None and as_of_utc.tzinfo is not None
        else as_of_utc.replace(tzinfo=UTC)
        if as_of_utc is not None
        else datetime.now(UTC)
    )
    yyyymm = dt.strftime("%Y-%m")
    sep = "&" if "?" in settings.live_fuel_price_url else "?"
    url = f"{settings.live_fuel_price_url}{sep}month={yyyymm}"
    cache_key = f"fuel_prices:{yyyymm}"
    if not _url_allowed(url, allowed_hosts_raw=str(settings.live_fuel_allowed_hosts or "")):
        return _fuel_live_error(
            reason_code="fuel_price_source_unavailable",
            message="Fuel live source URL host/scheme is not allowed.",
            diagnostics={
                "provider_url": url,
                "source_url": url,
                "fetched_at_utc": _iso_utc(_utc_now()),
            },
        )

    cached = _cache_get(cache_key)
    if cached is not None and _fresh(cached, ttl_s=settings.live_data_cache_ttl_s):
        normalized_cached = _normalize_fuel_payload(
            payload=cached.payload,
            provider_url=url,
            cache_hit=True,
            retry=_RetryResult(
                payload=None,
                status_code=None,
                attempt_count=0,
                retry_count=0,
                retry_total_backoff_ms=0,
                last_error_name=None,
                last_error_status=None,
                deadline_exceeded=False,
            ),
        )
        return normalized_cached

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if api_key:
        api_header = str(settings.live_fuel_api_key_header or "X-API-Key").strip() or "X-API-Key"
        headers[api_header] = api_key

    retry_result = _request_json_with_bounded_retry(
        url=url,
        headers=headers,
        deadline_at_monotonic_s=None,
    )
    if retry_result.payload is None:
        if retry_result.last_error_status in (401, 403):
            return _fuel_live_error(
                reason_code="fuel_price_auth_unavailable",
                message="Authentication failed for live fuel source.",
                diagnostics={
                    "provider_url": url,
                    "status_code": retry_result.last_error_status,
                    "fetched_at_utc": _iso_utc(_utc_now()),
                },
            )
        if cached is not None:
            normalized_stale = _normalize_fuel_payload(
                payload=cached.payload,
                provider_url=url,
                cache_hit=True,
                fetch_error=(
                    "deadline_exceeded"
                    if retry_result.deadline_exceeded
                    else retry_result.last_error_name
                ),
                stale_cache_used=True,
                status_code=retry_result.last_error_status,
                retry=retry_result,
            )
            return normalized_stale
        error_name = "deadline_exceeded" if retry_result.deadline_exceeded else (retry_result.last_error_name or "source_unavailable")
        return _fuel_live_error(
            reason_code="fuel_price_source_unavailable",
            message=f"Live fuel source unavailable: {error_name}",
            diagnostics={
                "provider_url": url,
                "fetched_at_utc": _iso_utc(_utc_now()),
                "retry_attempts": int(retry_result.attempt_count),
                "retry_count": int(retry_result.retry_count),
                "retry_total_backoff_ms": int(retry_result.retry_total_backoff_ms),
                "retry_last_error": retry_result.last_error_name,
                "retry_last_status_code": retry_result.last_error_status,
                "retry_deadline_exceeded": bool(retry_result.deadline_exceeded),
            },
        )

    payload = retry_result.payload
    _cache_put(cache_key, payload)
    return _normalize_fuel_payload(
        payload=payload,
        provider_url=url,
        cache_hit=False,
        status_code=retry_result.status_code,
        retry=retry_result,
    )


def live_carbon_schedule() -> dict[str, Any] | None:
    if not settings.live_carbon_schedule_url.strip():
        return None
    return _fetch_json(
        key="carbon_schedule",
        url=settings.live_carbon_schedule_url.strip(),
        reason_code_unavailable="carbon_policy_unavailable",
        allowed_hosts_csv=str(settings.live_carbon_allowed_hosts or ""),
    )
