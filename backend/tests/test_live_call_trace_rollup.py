from __future__ import annotations

from app.live_call_trace import (
    finish_trace,
    get_trace,
    mark_expected_calls_blocked,
    record_call,
    reset_trace,
    start_trace,
)
from app.settings import settings


def test_expected_rollup_matches_namespaced_observed_source_keys(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    monkeypatch.setattr(settings, "dev_route_debug_include_sensitive", True)
    token = start_trace(
        "test-rollup-canonical",
        endpoint="/route",
        expected_calls=[
            {
                "source_key": "scenario_webtris_daily",
                "component": "scenario",
                "url": "https://example.test/reports/daily",
                "method": "GET",
                "required": True,
            }
        ],
    )
    try:
        record_call(
            source_key="scenario:webtris:daily:3555:26022026",
            component="live_data_sources",
            url="https://example.test/reports/daily?site=3555",
            requested=True,
            success=True,
            status_code=200,
        )
        trace = get_trace("test-rollup-canonical")
    finally:
        reset_trace(token)

    assert trace is not None
    rollup = trace.get("expected_rollup", [])
    assert len(rollup) == 1
    row = rollup[0]
    assert row.get("source_family") == "scenario_webtris_daily"
    assert row.get("observed_calls") == 1
    assert row.get("success_count") == 1
    assert row.get("satisfied") is True


def test_live_trace_omits_raw_payloads_by_default(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    monkeypatch.setattr(settings, "dev_route_debug_include_sensitive", True)
    monkeypatch.setattr(settings, "dev_route_debug_return_raw_payloads", False)
    monkeypatch.setattr(settings, "dev_route_debug_max_raw_body_chars", 8)
    token = start_trace(
        "test-rollup-compact",
        endpoint="/route",
        expected_calls=[],
    )
    try:
        record_call(
            source_key="scenario:coefficients",
            component="live_data_sources",
            url="https://example.test/scenario.json",
            requested=True,
            success=True,
            status_code=200,
            request_headers_raw={"Authorization": "Bearer abc"},
            response_headers_raw={"Content-Type": "application/json"},
            response_body_raw="abcdefghijklmnopqrstuvwxyz",
        )
        trace = get_trace("test-rollup-compact")
    finally:
        reset_trace(token)

    assert trace is not None
    observed = trace.get("observed_calls", [])
    assert len(observed) == 1
    row = observed[0]
    assert "request_headers_raw" not in row
    assert "response_headers_raw" not in row
    assert "response_body_raw" not in row
    assert row.get("response_body_truncated") is True


def test_live_trace_returns_truncated_raw_payloads_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    monkeypatch.setattr(settings, "dev_route_debug_include_sensitive", True)
    monkeypatch.setattr(settings, "dev_route_debug_return_raw_payloads", True)
    monkeypatch.setattr(settings, "dev_route_debug_max_raw_body_chars", 8)
    token = start_trace(
        "test-rollup-raw",
        endpoint="/route",
        expected_calls=[],
    )
    try:
        record_call(
            source_key="scenario:coefficients",
            component="live_data_sources",
            url="https://example.test/scenario.json",
            requested=True,
            success=True,
            status_code=200,
            response_body_raw="abcdefghijklmnopqrstuvwxyz",
        )
        trace = get_trace("test-rollup-raw")
    finally:
        reset_trace(token)

    assert trace is not None
    observed = trace.get("observed_calls", [])
    assert len(observed) == 1
    row = observed[0]
    assert row.get("response_body_truncated") is True
    assert row.get("response_body_raw") == "abcdefgh"


def test_expected_rollup_status_progression_not_reached_to_miss(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    token = start_trace(
        "test-rollup-status-not-reached",
        endpoint="/route",
        expected_calls=[
            {
                "source_key": "departure_profiles",
                "component": "calibration",
                "url": "https://example.test/departure.json",
                "method": "GET",
                "required": True,
                "phase": "prefetch",
                "gate": "hard_refresh",
            }
        ],
    )
    try:
        running_trace = get_trace("test-rollup-status-not-reached")
        assert running_trace is not None
        running_row = running_trace["expected_rollup"][0]
        assert running_row.get("status") == "not_reached"
        assert running_row.get("phase") == "prefetch"
        assert running_row.get("gate") == "hard_refresh"

        finish_trace(request_id="test-rollup-status-not-reached", status="error", error_reason="forced")
        finished_trace = get_trace("test-rollup-status-not-reached")
    finally:
        reset_trace(token)

    assert finished_trace is not None
    finished_row = finished_trace["expected_rollup"][0]
    assert finished_row.get("status") == "miss"
    summary = finished_trace.get("summary", {})
    assert summary.get("expected_not_reached_count", 0) == 0
    assert summary.get("expected_miss_count", 0) == 1


def test_expected_rollup_status_blocked(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    token = start_trace(
        "test-rollup-status-blocked",
        endpoint="/route",
        expected_calls=[
            {
                "source_key": "fuel_prices",
                "component": "pricing",
                "url": "https://example.test/fuel.json",
                "method": "GET",
                "required": True,
            }
        ],
    )
    try:
        mark_expected_calls_blocked(
            reason_code="routing_graph_precheck_timeout",
            stage="collecting_candidates",
            detail="precheck_failed",
        )
        finish_trace(
            request_id="test-rollup-status-blocked",
            status="error",
            error_reason="routing_graph_precheck_timeout",
        )
        trace = get_trace("test-rollup-status-blocked")
    finally:
        reset_trace(token)

    assert trace is not None
    row = trace["expected_rollup"][0]
    assert row.get("status") == "blocked"
    summary = trace.get("summary", {})
    assert summary.get("expected_blocked_count", 0) == 1


def test_expected_blocked_stage_preserves_first_block_event(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    token = start_trace(
        "test-rollup-blocked-first-stage",
        endpoint="/route",
        expected_calls=[
            {
                "source_key": "scenario_coefficients",
                "component": "scenario",
                "url": "https://example.test/scenario.json",
                "method": "GET",
                "required": True,
            }
        ],
    )
    try:
        mark_expected_calls_blocked(
            reason_code="routing_graph_disconnected_od",
            stage="collecting_candidates",
            detail="route_graph_precheck_failed",
        )
        mark_expected_calls_blocked(
            reason_code="routing_graph_disconnected_od",
            stage="finalizing_pareto",
            detail="late_finalize_override_should_not_apply",
        )
        finish_trace(
            request_id="test-rollup-blocked-first-stage",
            status="error",
            error_reason="routing_graph_disconnected_od",
        )
        trace = get_trace("test-rollup-blocked-first-stage")
    finally:
        reset_trace(token)

    assert trace is not None
    row = trace["expected_rollup"][0]
    assert row.get("status") == "blocked"
    assert row.get("blocked_stage") == "collecting_candidates"
    assert row.get("blocked_detail") == "route_graph_precheck_failed"


def test_expected_rollup_matches_fuel_cache_key_family(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    token = start_trace(
        "test-rollup-fuel-family",
        endpoint="/route",
        expected_calls=[
            {
                "source_key": "fuel_prices",
                "component": "pricing",
                "url": "https://example.test/fuel.json",
                "method": "GET",
                "required": True,
            }
        ],
    )
    try:
        record_call(
            source_key="fuel_prices:2026-02",
            component="live_data_sources",
            url="https://example.test/fuel.json?month=2026-02",
            requested=False,
            success=True,
            cache_hit=True,
        )
        trace = get_trace("test-rollup-fuel-family")
    finally:
        reset_trace(token)

    assert trace is not None
    row = trace["expected_rollup"][0]
    assert row.get("source_family") == "fuel_prices"
    assert row.get("observed_calls") == 1
    assert row.get("status") == "ok"
