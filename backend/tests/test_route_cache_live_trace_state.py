from __future__ import annotations

from app import route_cache
from app.live_call_trace import (
    finish_trace,
    get_trace,
    record_call,
    record_trace_metadata,
    reset_trace,
    start_trace,
)
from app.settings import settings


def test_route_cache_key_state_is_stable_and_metadata_aware() -> None:
    state = route_cache.build_route_cache_key_state(
        artifact_kind=" decision_region ",
        run_id="run-1",
        lane_id=" focused ",
        variant_id="variant-a",
        cache_mode="hot",
        support_flag=True,
        support_status=" supported ",
        fidelity_class="proxy only",
        terminal_type="certified singleton",
        seed=7,
        extra={"beta": 2, "alpha": 1},
    )
    same_key = route_cache.build_route_cache_key(
        artifact_kind="decision_region",
        run_id="run-1",
        lane_id="focused",
        variant_id="variant-a",
        cache_mode="hot",
        support_flag=True,
        support_status="supported",
        fidelity_class="proxy only",
        terminal_type="certified singleton",
        seed=7,
        extra={"alpha": 1, "beta": 2},
    )

    assert state.as_dict()["artifact_kind"] == "decision_region"
    assert state.as_dict()["extra"] == {"alpha": "1", "beta": "2"}
    assert state.cache_key() == same_key
    assert "support=true" in same_key
    assert "support_status=supported" in same_key
    assert "fidelity=proxy_only" in same_key
    assert "terminal=certified_singleton" in same_key


def test_live_call_trace_carries_fidelity_support_and_terminal_metadata(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    token = start_trace(
        "trace-metadata",
        endpoint="/route",
        expected_calls=[],
        support_flag=False,
        support_status="uncertain",
        fidelity_class="proxy_only",
        terminal_type="uncertified_due_to_budget",
    )
    try:
        record_trace_metadata(
            request_id="trace-metadata",
            support_flag=True,
            support_status="supported",
            fidelity_class="fully_audited",
            terminal_type="certified_singleton",
        )
        record_call(
            request_id="trace-metadata",
            source_key="scenario:coefficients",
            component="live_data_sources",
            url="https://example.test/scenario.json",
            requested=True,
            success=True,
            status_code=200,
        )
        snapshot = finish_trace(
            request_id="trace-metadata",
            status="finished",
        )
        trace = get_trace("trace-metadata")
    finally:
        reset_trace(token)

    assert snapshot is not None
    assert trace is not None
    assert trace["support_flag"] is True
    assert trace["support_status"] == "supported"
    assert trace["fidelity_class"] == "fully_audited"
    assert trace["terminal_type"] == "certified_singleton"
    assert trace["observed_calls"][0]["support_flag"] is True
    assert trace["observed_calls"][0]["fidelity_class"] == "fully_audited"
    assert trace["summary"]["support_flag"] is True
    assert trace["summary"]["terminal_type"] == "certified_singleton"
