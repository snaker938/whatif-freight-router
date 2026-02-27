from __future__ import annotations

import time
from typing import Any

import httpx

import app.live_data_sources as live_data_sources
from app.settings import settings


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False

    def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:  # noqa: ARG002
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _response(
    status_code: int,
    *,
    url: str = "https://api.example.com/data",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", url)
    if payload is None:
        return httpx.Response(status_code, request=request, headers=headers, content=b"")
    return httpx.Response(status_code, request=request, headers=headers, json=payload)


def _set_retry_defaults(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "live_http_max_attempts", 6)
    monkeypatch.setattr(settings, "live_http_retry_deadline_ms", 15000)
    monkeypatch.setattr(settings, "live_http_retry_backoff_base_ms", 200)
    monkeypatch.setattr(settings, "live_http_retry_backoff_max_ms", 2500)
    monkeypatch.setattr(settings, "live_http_retry_jitter_ms", 0)
    monkeypatch.setattr(settings, "live_http_retry_respect_retry_after", True)
    monkeypatch.setattr(settings, "live_http_retryable_status_codes", "429,500,502,503,504")
    monkeypatch.setattr(settings, "live_data_request_timeout_s", 10.0)


def test_request_json_with_bounded_retry_network_error_then_success(monkeypatch) -> None:
    _set_retry_defaults(monkeypatch)
    request = httpx.Request("GET", "https://api.example.com/data")
    responses: list[Any] = [
        httpx.ConnectError("temporary failure", request=request),
        _response(200, payload={"ok": True}),
    ]

    def _client_factory(*args, **kwargs):  # noqa: ANN001, ARG001
        return _FakeClient(responses)

    monkeypatch.setattr(live_data_sources.httpx, "Client", _client_factory)
    monkeypatch.setattr(live_data_sources.time, "sleep", lambda _seconds: None)

    result = live_data_sources._request_json_with_bounded_retry(url="https://api.example.com/data")

    assert result.payload == {"ok": True}
    assert result.status_code == 200
    assert result.attempt_count == 2
    assert result.retry_count == 1
    assert result.deadline_exceeded is False


def test_request_json_with_bounded_retry_uses_retry_after(monkeypatch) -> None:
    _set_retry_defaults(monkeypatch)
    responses: list[Any] = [
        _response(429, headers={"Retry-After": "1"}),
        _response(200, payload={"ok": 1}),
    ]
    slept_seconds: list[float] = []

    def _client_factory(*args, **kwargs):  # noqa: ANN001, ARG001
        return _FakeClient(responses)

    monkeypatch.setattr(live_data_sources.httpx, "Client", _client_factory)
    monkeypatch.setattr(live_data_sources.time, "sleep", lambda seconds: slept_seconds.append(float(seconds)))

    result = live_data_sources._request_json_with_bounded_retry(url="https://api.example.com/data")

    assert result.payload == {"ok": 1}
    assert result.retry_count == 1
    assert slept_seconds
    assert slept_seconds[0] == 1.0


def test_request_json_with_bounded_retry_non_retryable_status_stops(monkeypatch) -> None:
    _set_retry_defaults(monkeypatch)
    responses: list[Any] = [_response(404)]

    def _client_factory(*args, **kwargs):  # noqa: ANN001, ARG001
        return _FakeClient(responses)

    monkeypatch.setattr(live_data_sources.httpx, "Client", _client_factory)
    monkeypatch.setattr(live_data_sources.time, "sleep", lambda _seconds: None)

    result = live_data_sources._request_json_with_bounded_retry(url="https://api.example.com/data")

    assert result.payload is None
    assert result.attempt_count == 1
    assert result.retry_count == 0
    assert result.last_error_status == 404
    assert result.deadline_exceeded is False


def test_fetch_json_with_ttl_uses_stale_cache_after_retry_exhaustion(monkeypatch) -> None:
    live_data_sources.clear_live_data_source_cache()
    key = "retry-test"
    live_data_sources._cache_put(key, {"snapshot": "cached"})
    live_data_sources._CACHE[key].fetched_at_s = time.time() - 3600.0

    retry_result = live_data_sources._RetryResult(
        payload=None,
        status_code=None,
        attempt_count=6,
        retry_count=5,
        retry_total_backoff_ms=1400,
        last_error_name="ReadTimeout",
        last_error_status=None,
        deadline_exceeded=False,
    )
    monkeypatch.setattr(live_data_sources, "_request_json_with_bounded_retry", lambda **kwargs: retry_result)

    payload, err = live_data_sources._fetch_json_with_ttl(
        key=key,
        url="https://api.example.com/data",
        ttl_s=1,
        allowed_hosts_csv="api.example.com",
    )

    assert err == "stale_cache:ReadTimeout"
    assert isinstance(payload, dict)
    live_diag = payload.get("_live_diagnostics")
    assert isinstance(live_diag, dict)
    assert live_diag.get("stale_cache_used") is True
    assert live_diag.get("retry_attempts") == 6
    assert live_diag.get("retry_count") == 5


def test_fetch_dft_rows_paginated_stops_on_query_deadline(monkeypatch) -> None:
    monkeypatch.setattr(live_data_sources, "_scenario_url_allowed", lambda _url: True)
    monkeypatch.setattr(
        live_data_sources,
        "_fetch_json_with_ttl",
        lambda **kwargs: (None, "deadline_exceeded"),
    )

    rows, meta, err = live_data_sources._fetch_dft_rows_paginated(
        base_url="https://roadtraffic.dft.gov.uk/api/raw-counts",
        ttl_s=1,
        max_pages=5,
        query_patch={"filter[year]": "2026"},
        query_deadline_ms=1,
    )

    assert rows == []
    assert err == "query_deadline_exceeded"
    assert meta.get("query_deadline_exceeded") is True


def test_live_fuel_prices_includes_retry_diagnostics(monkeypatch) -> None:
    live_data_sources.clear_live_data_source_cache()
    _set_retry_defaults(monkeypatch)
    monkeypatch.setattr(settings, "live_fuel_price_url", "https://api.example.com/fuel")
    monkeypatch.setattr(settings, "live_fuel_allowed_hosts", "api.example.com")
    monkeypatch.setattr(settings, "live_fuel_auth_token", "")
    monkeypatch.setattr(settings, "live_fuel_api_key", "")
    monkeypatch.setattr(settings, "live_data_cache_ttl_s", 1)
    retry_result = live_data_sources._RetryResult(
        payload={
            "as_of_utc": "2026-02-23T12:00:00Z",
            "source": "live:test",
            "prices_gbp_per_l": {"diesel": 1.55},
            "grid_price_gbp_per_kwh": 0.30,
            "regional_multipliers": {"uk_default": 1.0},
        },
        status_code=200,
        attempt_count=3,
        retry_count=2,
        retry_total_backoff_ms=600,
        last_error_name=None,
        last_error_status=None,
        deadline_exceeded=False,
    )
    monkeypatch.setattr(live_data_sources, "_request_json_with_bounded_retry", lambda **kwargs: retry_result)

    payload = live_data_sources.live_fuel_prices(None)

    assert isinstance(payload, dict)
    live_diag = payload.get("live_diagnostics")
    assert isinstance(live_diag, dict)
    assert live_diag.get("retry_attempts") == 3
    assert live_diag.get("retry_count") == 2


def test_live_fuel_prices_records_trace_for_network_and_cache_paths(monkeypatch) -> None:
    live_data_sources.clear_live_data_source_cache()
    _set_retry_defaults(monkeypatch)
    monkeypatch.setattr(settings, "live_fuel_price_url", "https://api.example.com/fuel")
    monkeypatch.setattr(settings, "live_fuel_allowed_hosts", "api.example.com")
    monkeypatch.setattr(settings, "live_fuel_auth_token", "")
    monkeypatch.setattr(settings, "live_fuel_api_key", "")
    monkeypatch.setattr(settings, "live_data_cache_ttl_s", 600)
    trace_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(live_data_sources, "_trace_live_call", lambda **kwargs: trace_rows.append(dict(kwargs)))
    retry_result = live_data_sources._RetryResult(
        payload={
            "as_of_utc": "2026-02-23T12:00:00Z",
            "source": "live:test",
            "prices_gbp_per_l": {"diesel": 1.55},
            "grid_price_gbp_per_kwh": 0.30,
            "regional_multipliers": {"uk_default": 1.0},
        },
        status_code=200,
        attempt_count=1,
        retry_count=0,
        retry_total_backoff_ms=0,
        last_error_name=None,
        last_error_status=None,
        deadline_exceeded=False,
    )
    monkeypatch.setattr(live_data_sources, "_request_json_with_bounded_retry", lambda **kwargs: retry_result)

    first_payload = live_data_sources.live_fuel_prices(None)
    second_payload = live_data_sources.live_fuel_prices(None)

    assert isinstance(first_payload, dict)
    assert isinstance(second_payload, dict)
    assert any(
        row.get("source_key") == "fuel_prices"
        and row.get("requested") is True
        and row.get("success") is True
        for row in trace_rows
    )
    assert any(
        row.get("source_key") == "fuel_prices"
        and row.get("requested") is False
        and row.get("cache_hit") is True
        for row in trace_rows
    )
