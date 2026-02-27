from __future__ import annotations

import asyncio
from typing import Any

import app.main as main_module
from app.models import LatLng
from app.routing_graph import GraphCandidateDiagnostics
from app.settings import settings


class _NoopOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return []


def test_collect_candidate_routes_maps_engine_no_path_reason(monkeypatch) -> None:
    async def _ok_precheck(*, origin: LatLng, destination: LatLng) -> dict[str, Any]:
        _ = (origin, destination)
        return {"ok": True, "reason_code": "ok", "message": "ok"}

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _no_path_routes(**_: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=23859,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=24,
                no_path_reason="no_path",
                no_path_detail="no path",
            ),
        )

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _ok_precheck)
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _no_path_routes)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=52.93571, lon=-1.12610),
            destination=LatLng(lat=51.48892, lon=-0.06592),
            max_routes=12,
            cache_key=None,
            scenario_edge_modifiers={},
            progress_cb=None,
        )
    )
    assert routes == []
    assert spec_count == 0
    assert warnings
    assert "route_graph: routing_graph_no_path" in warnings[0]
    assert "engine_reason=no_path" in warnings[0]
    assert diag.graph_no_path_reason == "routing_graph_no_path"


def _make_graph_route(seed: float = 0.0) -> dict[str, Any]:
    return {
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-1.1000 + seed, 52.1000],
                [-1.0900 + seed, 52.1100],
                [-1.0800 + seed, 52.1200],
                [-1.0700 + seed, 52.1300],
            ],
        },
        "duration": 1200.0,
        "distance": 15000.0,
    }


def test_collect_candidate_routes_retries_once_on_state_budget_exceeded(monkeypatch) -> None:
    call_kwargs: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _state_budget_then_exhausted(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        call_kwargs.append(dict(kwargs))
        if len(call_kwargs) == 1:
            return (
                [],
                GraphCandidateDiagnostics(
                    explored_states=120000,
                    generated_paths=0,
                    emitted_paths=0,
                    candidate_budget=24,
                    effective_max_hops=3345,
                    effective_state_budget=120000,
                    no_path_reason="state_budget_exceeded",
                    no_path_detail="state budget exceeded",
                ),
            )
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=240000,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=24,
                effective_max_hops=3345,
                effective_state_budget=240000,
                no_path_reason="state_budget_exceeded",
                no_path_detail="state budget exceeded after retry",
            ),
        )

    monkeypatch.setattr(settings, "route_graph_max_state_budget", 120000)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_multiplier", 2.0)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_cap", 600000)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", False)
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _state_budget_then_exhausted)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=52.40101, lon=-1.51062),
            destination=LatLng(lat=51.50412, lon=-0.12634),
            max_routes=12,
            cache_key=None,
            scenario_edge_modifiers={},
            progress_cb=None,
        )
    )

    assert routes == []
    assert spec_count == 0
    assert warnings and "route_graph: routing_graph_no_path" in warnings[0]
    assert len(call_kwargs) == 2
    assert call_kwargs[0].get("max_state_budget_override") is None
    assert int(call_kwargs[1].get("max_state_budget_override", 0)) == 240000
    assert diag.graph_retry_attempted is True
    assert diag.graph_retry_state_budget == 240000
    assert diag.graph_retry_outcome == "exhausted"


def test_collect_candidate_routes_scenario_separability_warn_only_returns_routes(monkeypatch) -> None:
    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_with_identical_families(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        modifiers = kwargs.get("scenario_edge_modifiers")
        _ = modifiers
        return (
            [_make_graph_route()],
            GraphCandidateDiagnostics(
                explored_states=9000,
                generated_paths=25,
                emitted_paths=24,
                candidate_budget=24,
                effective_max_hops=220,
                effective_state_budget=120000,
                no_path_reason="",
                no_path_detail="",
            ),
        )

    async def _empty_refine_iter(**_: Any):
        if False:
            yield None

    monkeypatch.setattr(settings, "route_graph_scenario_separability_fail", False)
    monkeypatch.setattr(settings, "route_graph_scenario_jaccard_max", 0.90)
    monkeypatch.setattr(settings, "route_graph_scenario_jaccard_floor", 0.82)
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_with_identical_families)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _empty_refine_iter)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.50051, lon=-0.13835),
            destination=LatLng(lat=51.50074, lon=-0.12750),
            max_routes=12,
            cache_key=None,
            scenario_edge_modifiers={
                "duration_multiplier": 1.25,
                "incident_rate_multiplier": 1.20,
                "incident_delay_multiplier": 1.20,
                "traffic_pressure": 1.30,
                "incident_pressure": 1.25,
                "weather_pressure": 1.15,
                "stochastic_sigma_multiplier": 1.0,
                "scenario_edge_scaling_version": "v4_empirical_transform",
            },
            progress_cb=None,
        )
    )

    assert routes
    assert spec_count >= 1
    assert any("scenario_profile_invalid" in warning for warning in warnings)
    assert diag.scenario_candidate_gate_action == "warned"
