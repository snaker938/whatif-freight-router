from __future__ import annotations

import contextvars
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
import threading
import time
from typing import Any

from .logging_utils import log_event
from .settings import settings

_TRACE_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "route_live_trace_request_id",
    default=None,
)


@dataclass
class _TraceRecord:
    request_id: str
    started_at_utc: str
    started_monotonic_s: float
    endpoint: str
    status: str = "running"
    error_reason: str | None = None
    finished_at_utc: str | None = None
    finished_monotonic_s: float | None = None
    expected_calls: list[dict[str, Any]] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    next_entry_id: int = 1
    dropped_entries: int = 0


_TRACE_LOCK = threading.Lock()
_TRACE_STORE: dict[str, _TraceRecord] = {}
_TRACE_ORDER: deque[str] = deque()


def _now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _enabled() -> bool:
    return bool(settings.dev_route_debug_console_enabled)


def _include_sensitive() -> bool:
    return bool(settings.dev_route_debug_include_sensitive)


def _return_raw_payloads() -> bool:
    return bool(settings.dev_route_debug_return_raw_payloads)


def current_trace_request_id() -> str | None:
    return _TRACE_REQUEST_ID.get()


def _canonical_source_family(source_key: str) -> str:
    key = str(source_key or "").strip()
    if not key:
        return "unknown"
    lowered = key.lower()
    if lowered == "scenario:coefficients":
        return "scenario_coefficients"
    if lowered == "scenario:webtris:sites":
        return "scenario_webtris_sites"
    if lowered.startswith("scenario:webtris:daily:"):
        return "scenario_webtris_daily"
    if lowered.startswith("scenario:trafficengland:"):
        return "scenario_traffic_england"
    if lowered.startswith("scenario:dft:raw_counts:"):
        return "scenario_dft_counts"
    if lowered.startswith("scenario:meteo:"):
        return "scenario_open_meteo"
    if lowered.startswith("departure:"):
        return "departure_profiles"
    if lowered.startswith("departure_profiles:"):
        return "departure_profiles"
    if lowered.startswith("stochastic:"):
        return "stochastic_regimes"
    if lowered.startswith("stochastic_regimes:"):
        return "stochastic_regimes"
    if lowered.startswith("toll:topology:"):
        return "toll_topology"
    if lowered.startswith("toll_topology:"):
        return "toll_topology"
    if lowered.startswith("toll:tariffs:"):
        return "toll_tariffs"
    if lowered.startswith("toll_tariffs:"):
        return "toll_tariffs"
    if lowered.startswith("fuel:"):
        return "fuel_prices"
    if lowered.startswith("fuel_prices:"):
        return "fuel_prices"
    if lowered.startswith("carbon:"):
        return "carbon_schedule"
    if lowered.startswith("carbon_schedule:"):
        return "carbon_schedule"
    if lowered.startswith("calendar:bank_holidays:"):
        return "bank_holidays"
    if lowered.startswith("bank_holidays:"):
        return "bank_holidays"
    if lowered.startswith("terrain:live_tile:"):
        return "terrain_live_tile"
    if lowered.startswith("terrain_live_tile:"):
        return "terrain_live_tile"
    return key


def _normalize_expected_call(payload: dict[str, Any]) -> dict[str, Any]:
    source_key = str(payload.get("source_key", "")).strip() or "unknown"
    component = str(payload.get("component", "")).strip() or "route_compute"
    url = str(payload.get("url", "")).strip()
    method = str(payload.get("method", "GET")).strip().upper() or "GET"
    return {
        "source_key": source_key,
        "source_family": _canonical_source_family(source_key),
        "component": component,
        "url": url,
        "method": method,
        "required": bool(payload.get("required", True)),
        "description": str(payload.get("description", "")).strip() or None,
        "phase": str(payload.get("phase", "unknown")).strip() or "unknown",
        "gate": str(payload.get("gate", "none")).strip() or "none",
        "blocked": bool(payload.get("blocked", False)),
        "blocked_reason": str(payload.get("blocked_reason", "")).strip() or None,
        "blocked_stage": str(payload.get("blocked_stage", "")).strip() or None,
        "blocked_detail": str(payload.get("blocked_detail", "")).strip() or None,
    }


def _sanitize_headers(headers: dict[str, Any] | None) -> dict[str, Any] | None:
    if not headers:
        return None
    out: dict[str, Any] = {}
    for key, value in headers.items():
        k = str(key)
        if _include_sensitive():
            out[k] = "" if value is None else str(value)
        else:
            lowered = k.lower()
            if any(token in lowered for token in ("auth", "token", "secret", "key", "cookie")):
                out[k] = "***"
            else:
                out[k] = "" if value is None else str(value)
    return out


def _sanitize_text(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if _include_sensitive():
        return text
    return "***"


def _truncate_response_body(value: str | None) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    max_chars = max(0, int(settings.dev_route_debug_max_raw_body_chars))
    if max_chars <= 0:
        return ("", True) if value else ("", False)
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def _entry_source_family(entry: dict[str, Any]) -> str:
    family = str(entry.get("source_family", "")).strip()
    if family:
        return family
    return _canonical_source_family(str(entry.get("source_key", "")).strip())


def _snapshot_entry(entry: dict[str, Any], *, include_raw_payloads: bool) -> dict[str, Any]:
    out = dict(entry)
    if not include_raw_payloads:
        out.pop("request_headers_raw", None)
        out.pop("response_headers_raw", None)
        out.pop("response_body_raw", None)
    return out


def _rollup_expected(record: _TraceRecord) -> list[dict[str, Any]]:
    entries = record.entries
    expected_rollup: list[dict[str, Any]] = []
    for expected in record.expected_calls:
        source_key = str(expected.get("source_key", "")).strip()
        source_family = str(expected.get("source_family", "")).strip() or _canonical_source_family(source_key)
        matches = [row for row in entries if _entry_source_family(row) == source_family]
        requested = sum(1 for row in matches if bool(row.get("requested")))
        success = sum(1 for row in matches if bool(row.get("success")))
        failures = sum(1 for row in matches if not bool(row.get("success")))
        last_row = matches[-1] if matches else None
        blocked = bool(expected.get("blocked")) and len(matches) == 0
        satisfied = bool(success > 0 or (matches and all(bool(row.get("cache_hit")) for row in matches)))
        if blocked:
            satisfied = False
        status = "ok" if satisfied else "miss"
        if blocked:
            status = "blocked"
        elif len(matches) == 0:
            status = "not_reached" if str(record.status).strip().lower() == "running" else "miss"
        expected_rollup.append(
            {
                **expected,
                "source_family": source_family,
                "observed_calls": len(matches),
                "requested_calls": requested,
                "success_count": success,
                "failure_count": failures,
                "last_status_code": (last_row or {}).get("status_code"),
                "last_fetch_error": (last_row or {}).get("fetch_error"),
                "blocked": blocked,
                "blocked_reason": expected.get("blocked_reason"),
                "blocked_stage": expected.get("blocked_stage"),
                "blocked_detail": expected.get("blocked_detail"),
                "satisfied": satisfied,
                "status": status,
            }
        )
    return expected_rollup


def _purge_locked(now_monotonic_s: float) -> None:
    ttl_s = max(30, int(settings.dev_route_debug_trace_ttl_seconds))
    stale_ids: list[str] = []
    for request_id, record in _TRACE_STORE.items():
        finish = record.finished_monotonic_s
        baseline = finish if isinstance(finish, (int, float)) else record.started_monotonic_s
        if (now_monotonic_s - float(baseline)) > float(ttl_s):
            stale_ids.append(request_id)
    if stale_ids:
        stale_lookup = set(stale_ids)
        for request_id in stale_ids:
            _TRACE_STORE.pop(request_id, None)
        _TRACE_ORDER[:] = deque(
            request_id for request_id in _TRACE_ORDER if request_id not in stale_lookup
        )

    max_traces = max(1, int(settings.dev_route_debug_max_request_traces))
    while len(_TRACE_ORDER) > max_traces:
        oldest = _TRACE_ORDER.popleft()
        _TRACE_STORE.pop(oldest, None)


def _build_summary(record: _TraceRecord) -> dict[str, Any]:
    entries = record.entries
    total_calls = len(entries)
    requested_calls = sum(1 for row in entries if bool(row.get("requested")))
    successful_calls = sum(1 for row in entries if bool(row.get("success")))
    failed_calls = sum(1 for row in entries if not bool(row.get("success")))
    cache_hit_calls = sum(1 for row in entries if bool(row.get("cache_hit")))
    stale_cache_calls = sum(1 for row in entries if bool(row.get("stale_cache_used")))

    expected_rollup = _rollup_expected(record)
    expected_total = len(expected_rollup)
    expected_satisfied = sum(1 for row in expected_rollup if bool(row.get("satisfied")))
    expected_ok_count = sum(1 for row in expected_rollup if str(row.get("status")) == "ok")
    expected_blocked_count = sum(1 for row in expected_rollup if str(row.get("status")) == "blocked")
    expected_not_reached_count = sum(1 for row in expected_rollup if str(row.get("status")) == "not_reached")
    expected_miss_count = sum(1 for row in expected_rollup if str(row.get("status")) == "miss")
    return {
        "total_calls": total_calls,
        "requested_calls": requested_calls,
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "cache_hit_calls": cache_hit_calls,
        "stale_cache_calls": stale_cache_calls,
        "expected_total": expected_total,
        "expected_satisfied": expected_satisfied,
        "expected_ok_count": expected_ok_count,
        "expected_blocked_count": expected_blocked_count,
        "expected_not_reached_count": expected_not_reached_count,
        "expected_miss_count": expected_miss_count,
        "dropped_entries": int(record.dropped_entries),
    }


def _snapshot_locked(record: _TraceRecord) -> dict[str, Any]:
    return _snapshot_with_rollup_locked(record)


def _snapshot_with_rollup_locked(record: _TraceRecord) -> dict[str, Any]:
    entries = record.entries
    expected_rollup = _rollup_expected(record)
    include_raw_payloads = _return_raw_payloads()

    summary = {
        "total_calls": len(entries),
        "requested_calls": sum(1 for row in entries if bool(row.get("requested"))),
        "successful_calls": sum(1 for row in entries if bool(row.get("success"))),
        "failed_calls": sum(1 for row in entries if not bool(row.get("success"))),
        "cache_hit_calls": sum(1 for row in entries if bool(row.get("cache_hit"))),
        "stale_cache_calls": sum(1 for row in entries if bool(row.get("stale_cache_used"))),
        "expected_total": len(expected_rollup),
        "expected_satisfied": sum(1 for row in expected_rollup if bool(row.get("satisfied"))),
        "expected_ok_count": sum(1 for row in expected_rollup if str(row.get("status")) == "ok"),
        "expected_blocked_count": sum(1 for row in expected_rollup if str(row.get("status")) == "blocked"),
        "expected_not_reached_count": sum(1 for row in expected_rollup if str(row.get("status")) == "not_reached"),
        "expected_miss_count": sum(1 for row in expected_rollup if str(row.get("status")) == "miss"),
        "dropped_entries": int(record.dropped_entries),
    }

    return {
        "request_id": record.request_id,
        "endpoint": record.endpoint,
        "status": record.status,
        "error_reason": record.error_reason,
        "started_at_utc": record.started_at_utc,
        "finished_at_utc": record.finished_at_utc,
        "expected_calls": list(record.expected_calls),
        "expected_rollup": expected_rollup,
        "observed_calls": [
            _snapshot_entry(row, include_raw_payloads=include_raw_payloads) for row in entries
        ],
        "summary": summary,
    }


def start_trace(
    request_id: str,
    *,
    endpoint: str,
    expected_calls: list[dict[str, Any]] | None = None,
) -> contextvars.Token[str | None]:
    token = _TRACE_REQUEST_ID.set(request_id)
    if not _enabled():
        return token

    now_monotonic = time.monotonic()
    normalized_expected = [
        _normalize_expected_call(row)
        for row in (expected_calls or [])
        if isinstance(row, dict)
    ]
    with _TRACE_LOCK:
        _purge_locked(now_monotonic)
        _TRACE_STORE[request_id] = _TraceRecord(
            request_id=request_id,
            started_at_utc=_now_utc_iso(),
            started_monotonic_s=now_monotonic,
            endpoint=str(endpoint).strip() or "unknown",
            expected_calls=normalized_expected,
        )
        try:
            _TRACE_ORDER.remove(request_id)
        except ValueError:
            pass
        _TRACE_ORDER.append(request_id)
        _purge_locked(now_monotonic)
    return token


def reset_trace(token: contextvars.Token[str | None] | None) -> None:
    if token is None:
        return
    try:
        _TRACE_REQUEST_ID.reset(token)
    except Exception:
        pass


def record_expected_call(
    *,
    source_key: str,
    component: str,
    url: str,
    method: str = "GET",
    required: bool = True,
    description: str | None = None,
    request_id: str | None = None,
) -> None:
    if not _enabled():
        return
    rid = str(request_id or current_trace_request_id() or "").strip()
    if not rid:
        return
    payload = _normalize_expected_call(
        {
            "source_key": source_key,
            "component": component,
            "url": url,
            "method": method,
            "required": required,
            "description": description,
        }
    )
    with _TRACE_LOCK:
        record = _TRACE_STORE.get(rid)
        if record is None:
            return
        duplicate = next(
            (
                row
                for row in record.expected_calls
                if str(row.get("source_key", "")).strip() == payload["source_key"]
                and str(row.get("url", "")).strip() == payload["url"]
            ),
            None,
        )
        if duplicate is None:
            record.expected_calls.append(payload)


def mark_expected_calls_blocked(
    *,
    reason_code: str,
    stage: str | None = None,
    detail: str | None = None,
    request_id: str | None = None,
) -> None:
    if not _enabled():
        return
    rid = str(request_id or current_trace_request_id() or "").strip()
    if not rid:
        return
    code = str(reason_code or "unknown").strip() or "unknown"
    stage_text = str(stage or "").strip() or None
    detail_text = str(detail or "").strip() or None

    with _TRACE_LOCK:
        record = _TRACE_STORE.get(rid)
        if record is None:
            return
        for expected in record.expected_calls:
            source_family = (
                str(expected.get("source_family", "")).strip()
                or _canonical_source_family(str(expected.get("source_key", "")).strip())
            )
            has_observed = any(
                _entry_source_family(entry) == source_family
                for entry in record.entries
            )
            if has_observed:
                continue
            # Preserve the first block stage/reason so downstream diagnostics stay truthful.
            if bool(expected.get("blocked")) and (
                expected.get("blocked_stage") is not None
                or expected.get("blocked_reason") is not None
                or expected.get("blocked_detail") is not None
            ):
                continue
            expected["blocked"] = True
            expected["blocked_reason"] = code
            expected["blocked_stage"] = stage_text
            expected["blocked_detail"] = detail_text

    log_event(
        "route_live_call_blocked",
        request_id=rid,
        reason_code=code,
        stage=stage_text,
        detail=detail_text,
    )


def record_call(
    *,
    source_key: str,
    component: str,
    url: str,
    method: str = "GET",
    requested: bool,
    success: bool,
    status_code: int | None = None,
    fetch_error: str | None = None,
    cache_hit: bool = False,
    stale_cache_used: bool = False,
    retry_attempts: int = 0,
    retry_count: int = 0,
    retry_total_backoff_ms: int = 0,
    retry_last_error: str | None = None,
    retry_last_status_code: int | None = None,
    retry_deadline_exceeded: bool = False,
    duration_ms: float | None = None,
    headers: dict[str, Any] | None = None,
    request_headers_raw: dict[str, Any] | None = None,
    response_headers_raw: dict[str, Any] | None = None,
    response_body_raw: str | None = None,
    response_body_content_type: str | None = None,
    response_body_bytes: int | None = None,
    extra: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> None:
    if not _enabled():
        return
    rid = str(request_id or current_trace_request_id() or "").strip()
    if not rid:
        return

    safe_headers = _sanitize_headers(headers)
    safe_request_headers = _sanitize_headers(request_headers_raw)
    safe_response_headers = _sanitize_headers(response_headers_raw)
    safe_response_body, response_body_truncated = _truncate_response_body(_sanitize_text(response_body_raw))
    normalized_source_key = str(source_key or "unknown").strip() or "unknown"
    entry: dict[str, Any] = {
        "entry_id": 0,
        "request_id": rid,
        "at_utc": _now_utc_iso(),
        "source_key": normalized_source_key,
        "source_family": _canonical_source_family(normalized_source_key),
        "component": str(component or "route_compute").strip() or "route_compute",
        "url": str(url or "").strip(),
        "method": str(method or "GET").strip().upper() or "GET",
        "requested": bool(requested),
        "success": bool(success),
        "status_code": int(status_code) if isinstance(status_code, int) else None,
        "fetch_error": str(fetch_error).strip() if fetch_error else None,
        "cache_hit": bool(cache_hit),
        "stale_cache_used": bool(stale_cache_used),
        "retry_attempts": int(max(0, retry_attempts)),
        "retry_count": int(max(0, retry_count)),
        "retry_total_backoff_ms": int(max(0, retry_total_backoff_ms)),
        "retry_last_error": str(retry_last_error).strip() if retry_last_error else None,
        "retry_last_status_code": int(retry_last_status_code) if isinstance(retry_last_status_code, int) else None,
        "retry_deadline_exceeded": bool(retry_deadline_exceeded),
        "duration_ms": float(duration_ms) if isinstance(duration_ms, (int, float)) else None,
        "headers": safe_headers,
        "request_headers_raw": safe_request_headers,
        "response_headers_raw": safe_response_headers,
        "response_body_raw": safe_response_body,
        "response_body_truncated": bool(response_body_truncated),
        "response_body_content_type": (
            str(response_body_content_type).strip()
            if isinstance(response_body_content_type, str) and response_body_content_type.strip()
            else None
        ),
        "response_body_bytes": int(response_body_bytes) if isinstance(response_body_bytes, int) else None,
        "extra": dict(extra) if isinstance(extra, dict) else {},
    }

    with _TRACE_LOCK:
        record = _TRACE_STORE.get(rid)
        if record is None:
            return
        max_calls = max(1, int(settings.dev_route_debug_max_calls_per_request))
        if len(record.entries) >= max_calls:
            record.dropped_entries += 1
            return
        entry["entry_id"] = int(record.next_entry_id)
        record.next_entry_id += 1
        record.entries.append(entry)

    log_event(
        "route_live_call",
        request_id=rid,
        source_key=entry["source_key"],
        source_family=entry["source_family"],
        component=entry["component"],
        url=entry["url"],
        requested=entry["requested"],
        success=entry["success"],
        status_code=entry["status_code"],
        fetch_error=entry["fetch_error"],
        cache_hit=entry["cache_hit"],
        stale_cache_used=entry["stale_cache_used"],
        retry_attempts=entry["retry_attempts"],
        retry_count=entry["retry_count"],
        retry_total_backoff_ms=entry["retry_total_backoff_ms"],
        retry_last_error=entry["retry_last_error"],
        retry_last_status_code=entry["retry_last_status_code"],
        retry_deadline_exceeded=entry["retry_deadline_exceeded"],
        duration_ms=entry["duration_ms"],
        headers=entry["headers"],
        extra=entry["extra"],
    )


def finish_trace(
    *,
    request_id: str | None = None,
    status: str,
    endpoint: str | None = None,
    error_reason: str | None = None,
) -> dict[str, Any] | None:
    if not _enabled():
        return None
    rid = str(request_id or current_trace_request_id() or "").strip()
    if not rid:
        return None

    with _TRACE_LOCK:
        record = _TRACE_STORE.get(rid)
        if record is None:
            return None
        record.status = str(status or "finished").strip() or "finished"
        if endpoint:
            record.endpoint = str(endpoint).strip() or record.endpoint
        record.error_reason = str(error_reason).strip() if error_reason else None
        record.finished_at_utc = _now_utc_iso()
        record.finished_monotonic_s = time.monotonic()
        snapshot = _snapshot_with_rollup_locked(record)

    summary = snapshot.get("summary", {}) if isinstance(snapshot, dict) else {}
    log_event(
        "route_live_call_summary",
        request_id=rid,
        endpoint=snapshot.get("endpoint") if isinstance(snapshot, dict) else endpoint,
        status=snapshot.get("status") if isinstance(snapshot, dict) else status,
        error_reason=snapshot.get("error_reason") if isinstance(snapshot, dict) else error_reason,
        total_calls=summary.get("total_calls", 0),
        requested_calls=summary.get("requested_calls", 0),
        successful_calls=summary.get("successful_calls", 0),
        failed_calls=summary.get("failed_calls", 0),
        cache_hit_calls=summary.get("cache_hit_calls", 0),
        stale_cache_calls=summary.get("stale_cache_calls", 0),
        expected_total=summary.get("expected_total", 0),
        expected_satisfied=summary.get("expected_satisfied", 0),
        expected_ok_count=summary.get("expected_ok_count", 0),
        expected_blocked_count=summary.get("expected_blocked_count", 0),
        expected_not_reached_count=summary.get("expected_not_reached_count", 0),
        expected_miss_count=summary.get("expected_miss_count", 0),
        dropped_entries=summary.get("dropped_entries", 0),
    )
    return snapshot


def get_trace(request_id: str) -> dict[str, Any] | None:
    if not _enabled():
        return None
    rid = str(request_id or "").strip()
    if not rid:
        return None
    with _TRACE_LOCK:
        _purge_locked(time.monotonic())
        record = _TRACE_STORE.get(rid)
        if record is None:
            return None
        return _snapshot_with_rollup_locked(record)
