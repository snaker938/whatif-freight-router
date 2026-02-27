from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import app.main as main_module
from app.live_call_trace import get_trace, reset_trace, start_trace
from app.models import (
    CostToggles,
    EmissionsContext,
    IncidentSimulatorConfig,
    LatLng,
    StochasticConfig,
    WeatherImpactConfig,
)
from app.settings import settings


def test_route_graph_od_feasibility_timeout_maps_reason_code(monkeypatch) -> None:
    monkeypatch.setattr(settings, "route_graph_od_feasibility_timeout_ms", 100)

    def _slow_feasibility(*, origin_lat: float, origin_lon: float, destination_lat: float, destination_lon: float):
        _ = (origin_lat, origin_lon, destination_lat, destination_lon)
        time.sleep(0.25)
        return {"ok": True, "reason_code": "ok"}

    monkeypatch.setattr(main_module, "route_graph_od_feasibility", _slow_feasibility)
    result = asyncio.run(
        main_module._route_graph_od_feasibility_async(
            origin=LatLng(lat=53.0, lon=-1.0),
            destination=LatLng(lat=52.0, lon=-0.1),
        )
    )
    assert result["ok"] is False
    assert result["reason_code"] == "routing_graph_precheck_timeout"
    assert result["stage"] == "collecting_candidates"
    assert result["stage_detail"] == "route_graph_feasibility_check_timeout"
    assert int(result["timeout_ms"]) == 100


def test_route_graph_od_feasibility_success_reason_code_is_ok(monkeypatch) -> None:
    monkeypatch.setattr(settings, "route_graph_od_feasibility_timeout_ms", 1000)

    def _ok_feasibility(*, origin_lat: float, origin_lon: float, destination_lat: float, destination_lon: float):
        _ = (origin_lat, origin_lon, destination_lat, destination_lon)
        # Successful prechecks should always be logged/returned as reason_code=ok.
        return {"ok": True, "reason_code": "routing_graph_unavailable"}

    monkeypatch.setattr(main_module, "route_graph_od_feasibility", _ok_feasibility)
    result = asyncio.run(
        main_module._route_graph_od_feasibility_async(
            origin=LatLng(lat=53.0, lon=-1.0),
            destination=LatLng(lat=52.0, lon=-0.1),
        )
    )
    assert result["ok"] is True
    assert result["reason_code"] == "ok"


def test_expected_live_calls_marked_blocked_on_precheck_failure(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    token = start_trace(
        "test-precheck-blocked",
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
        main_module._record_expected_calls_blocked(
            reason_code="routing_graph_precheck_timeout",
            stage="collecting_candidates",
            detail="route_graph_precheck_failed",
        )
        trace = get_trace("test-precheck-blocked")
    finally:
        reset_trace(token)

    assert trace is not None
    rollup = trace.get("expected_rollup", [])
    assert len(rollup) == 1
    row = rollup[0]
    assert row.get("blocked") is True
    assert row.get("blocked_reason") == "routing_graph_precheck_timeout"
    assert row.get("blocked_stage") == "collecting_candidates"
    assert row.get("satisfied") is False


class _NoopOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return []


def _collect_kwargs() -> dict[str, Any]:
    return {
        "osrm": _NoopOSRM(),
        "origin": LatLng(lat=52.47574, lon=-1.90201),
        "destination": LatLng(lat=52.28760, lon=-1.77910),
        "waypoints": None,
        "max_alternatives": 3,
        "vehicle_type": "rigid_hgv",
        "scenario_mode": main_module.ScenarioMode.NO_SHARING,
        "cost_toggles": CostToggles(),
        "terrain_profile": "flat",
        "stochastic": StochasticConfig(),
        "emissions_context": EmissionsContext(),
        "weather": WeatherImpactConfig(),
        "incident_simulation": IncidentSimulatorConfig(),
        "departure_time_utc": None,
        "pareto_method": "dominance",
        "epsilon": None,
        "optimization_mode": "expected_value",
        "risk_aversion": 1.0,
        "option_prefix": "route",
        "progress_cb": None,
    }


def test_collect_route_options_precheck_timeout_degraded_continue(monkeypatch) -> None:
    async def _precheck_timeout(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {
            "ok": False,
            "reason_code": "routing_graph_precheck_timeout",
            "message": "Route graph OD feasibility check timed out before completion. (timeout_ms=8000)",
            "timeout_ms": 8000,
            "elapsed_ms": 8012.5,
            "stage": "collecting_candidates",
            "stage_detail": "route_graph_feasibility_check_timeout",
        }

    async def _scenario_context(**_: Any) -> Any:
        return None

    async def _scenario_modifiers(**_: Any) -> dict[str, Any]:
        return {}

    async def _candidate_routes(**_: Any):
        return [], [], 0, main_module.CandidateDiagnostics()

    def _build_options(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return [], [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(settings, "route_graph_precheck_timeout_fail_closed", False)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)
    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _precheck_timeout)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _scenario_context)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _scenario_modifiers)
    monkeypatch.setattr(main_module, "_collect_candidate_routes", _candidate_routes)
    monkeypatch.setattr(main_module, "_build_options", _build_options)

    options, warnings, candidate_fetches, _terrain_diag, candidate_diag = asyncio.run(
        main_module._collect_route_options(**_collect_kwargs())
    )
    assert options == []
    assert candidate_fetches == 0
    assert warnings
    assert "route_graph: routing_graph_precheck_timeout" in warnings[0]
    assert candidate_diag.precheck_reason_code == "routing_graph_precheck_timeout"
    assert candidate_diag.precheck_gate_action == "degraded_continue"


def test_collect_route_options_precheck_timeout_fail_closed(monkeypatch) -> None:
    async def _precheck_timeout(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {
            "ok": False,
            "reason_code": "routing_graph_precheck_timeout",
            "message": "Route graph OD feasibility check timed out before completion. (timeout_ms=8000)",
            "timeout_ms": 8000,
            "elapsed_ms": 8012.5,
            "stage": "collecting_candidates",
            "stage_detail": "route_graph_feasibility_check_timeout",
        }

    monkeypatch.setattr(settings, "route_graph_precheck_timeout_fail_closed", True)
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)
    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _precheck_timeout)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "resolve_vehicle_profile", lambda *_: SimpleNamespace(vehicle_class="rigid_hgv"))

    options, warnings, candidate_fetches, _terrain_diag, candidate_diag = asyncio.run(
        main_module._collect_route_options(**_collect_kwargs())
    )
    assert options == []
    assert candidate_fetches == 0
    assert warnings
    assert "route_graph: routing_graph_precheck_timeout" in warnings[0]
    assert candidate_diag.precheck_reason_code == "routing_graph_precheck_timeout"
    assert candidate_diag.precheck_gate_action == "fail_closed"
