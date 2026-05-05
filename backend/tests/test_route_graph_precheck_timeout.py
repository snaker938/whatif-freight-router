from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import app.main as main_module
import pytest
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


class _NoopORS:
    async def fetch_route(self, **_: Any) -> Any:
        raise AssertionError("local ORS should not be used in deferred-load degraded-continue test")


def _make_ranked_route(*, duration_s: float, lon_seed: float, road_class: str) -> dict[str, Any]:
    return {
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-1.9000 + lon_seed, 52.5000],
                [-1.8500 + lon_seed, 52.5200],
                [-1.7800 + lon_seed, 52.5600],
            ],
        },
        "duration": float(duration_s),
        "distance": 32_000.0,
        "_graph_meta": {
            "road_mix_counts": {road_class: 5},
            "toll_edges": 0,
        },
    }


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


@pytest.mark.parametrize("pipeline_mode", ["dccs", "dccs_refc", "tri_source"])
def test_compute_direct_route_pipeline_precheck_deferred_load_degraded_continue(
    monkeypatch,
    pipeline_mode: str,
) -> None:
    fallback_route = _make_ranked_route(duration_s=27_800.0, lon_seed=0.05, road_class="trunk")
    captured_start_goal: dict[str, Any] = {}

    async def _fake_scenario_context_from_od(**_: Any) -> dict[str, Any]:
        return {"bucket": "clear"}

    async def _fake_scenario_candidate_modifiers_async(**_: Any) -> dict[str, Any]:
        return {}

    def _fake_feasibility(**_: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "reason_code": "routing_graph_deferred_load",
            "message": "Route graph full load is deferred in fast-startup mode; using OSRM family fallback for this request.",
            "elapsed_ms": 2.5,
            "stage": "collecting_candidates",
            "stage_detail": "route_graph_feasibility_check_timeout",
        }

    async def _fake_k_raw_search(
        *,
        start_node_id: str | None = None,
        goal_node_id: str | None = None,
        **_: Any,
    ) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
        captured_start_goal["start_node_id"] = start_node_id
        captured_start_goal["goal_node_id"] = goal_node_id
        return (
            [dict(fallback_route)],
            main_module.GraphCandidateDiagnostics(
                explored_states=0,
                generated_paths=1,
                emitted_paths=1,
                candidate_budget=4,
            ),
            {},
        )

    async def _fake_refine_graph_candidate_batch(**kwargs: Any):
        selected_records = list(kwargs.get("selected_records", []))
        raw_graph_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed_costs = {str(record.candidate_id): 14.0 for record in selected_records}
        refined = [
            dict(raw_graph_routes_by_id.get(str(record.candidate_id), fallback_route))
            for record in selected_records
        ]
        return refined, [], observed_costs, len(selected_records), 25.0

    def _fake_build_options(routes: list[dict[str, Any]], **_: Any):
        options = [
            main_module.RouteOption(
                id=f"route_{idx}",
                geometry=main_module.GeoJSONLineString(
                    type="LineString",
                    coordinates=[
                        (float(point[0]), float(point[1]))
                        for point in route["geometry"]["coordinates"]
                    ],
                ),
                metrics=main_module.RouteMetrics(
                    distance_km=float(route["distance"]) / 1000.0,
                    duration_s=float(route["duration"]),
                    monetary_cost=325.0,
                    emissions_kg=210.0,
                    avg_speed_kmh=70.0,
                ),
            )
            for idx, route in enumerate(routes, start=1)
        ]
        return options, [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(settings, "route_graph_precheck_timeout_fail_closed", False)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _fake_scenario_context_from_od)
    monkeypatch.setattr(
        main_module,
        "_scenario_candidate_modifiers_async",
        _fake_scenario_candidate_modifiers_async,
    )
    monkeypatch.setattr(main_module, "route_graph_od_feasibility", _fake_feasibility)
    monkeypatch.setattr(main_module, "_route_graph_k_raw_search", _fake_k_raw_search)
    monkeypatch.setattr(main_module, "_refine_graph_candidate_batch", _fake_refine_graph_candidate_batch)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: list(options)[0])
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda *_: False)
    monkeypatch.setattr(
        main_module,
        "_route_selection_score_map",
        lambda options, **_: {
            str(option.id): float(index)
            for index, option in enumerate(options, start=1)
        },
    )

    req = main_module.RouteRequest(
        origin=LatLng(lat=52.47574, lon=-1.90201),
        destination=LatLng(lat=52.28760, lon=-1.77910),
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        max_alternatives=3,
        pipeline_mode=pipeline_mode,
        search_budget=1,
        evidence_budget=0,
        cert_world_count=10,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=3,
            pipeline_mode=pipeline_mode,
            run_seed=20260405,
        )
    )

    candidate_diag = result["candidate_diag"]
    assert result["selected"].id == "route_1"
    assert candidate_diag.precheck_reason_code == "routing_graph_deferred_load"
    assert candidate_diag.precheck_gate_action == "degraded_continue"
    assert captured_start_goal["start_node_id"] is None
    assert captured_start_goal["goal_node_id"] is None
    assert any("routing_graph_deferred_load" in warning for warning in result["warnings"])
