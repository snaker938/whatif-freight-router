from __future__ import annotations

import asyncio
import math
from typing import Any

import app.main as main_module
import app.routing_graph as routing_graph
from app.models import LatLng
from app.routing_graph import GraphCandidateDiagnostics, GraphEdge, RouteGraph
from app.settings import settings


class _NoopOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return []


def _make_graph_route(seed: float = 0.0) -> dict[str, Any]:
    return {
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-0.1300 + seed, 51.5000],
                [-0.1200 + seed, 51.5100],
                [-0.1100 + seed, 51.5200],
                [-0.1000 + seed, 51.5300],
            ],
        },
        "duration": 1200.0,
        "distance": 15000.0,
    }


def _toy_graph() -> RouteGraph:
    edge_ab = GraphEdge(
        to="b",
        cost=120.0,
        distance_m=1200.0,
        highway="primary",
        toll=False,
        maxspeed_kph=50.0,
    )
    edge_ba = GraphEdge(
        to="a",
        cost=120.0,
        distance_m=1200.0,
        highway="primary",
        toll=False,
        maxspeed_kph=50.0,
    )
    return RouteGraph(
        version="test",
        source="test",
        nodes={
            "a": (51.5000, -0.1300),
            "b": (51.5300, -0.1000),
        },
        adjacency={
            "a": (edge_ab,),
            "b": (edge_ba,),
        },
        edge_index={
            ("a", "b"): edge_ab,
            ("b", "a"): edge_ba,
        },
        grid_index={},
        component_by_node={"a": 1, "b": 1},
        component_sizes={1: 2},
        component_count=1,
        largest_component_nodes=2,
        largest_component_ratio=1.0,
        graph_fragmented=False,
    )


def test_effective_max_hops_applies_edge_length_floor(monkeypatch) -> None:
    monkeypatch.setattr(settings, "route_graph_adaptive_hops_enabled", True)
    monkeypatch.setattr(settings, "route_graph_max_hops", 220)
    monkeypatch.setattr(settings, "route_graph_hops_per_km", 1.0)
    monkeypatch.setattr(settings, "route_graph_hops_detour_factor", 1.0)
    monkeypatch.setattr(settings, "route_graph_edge_length_estimate_m", 30.0)
    monkeypatch.setattr(settings, "route_graph_hops_safety_factor", 1.8)
    monkeypatch.setattr(settings, "route_graph_max_hops_cap", 5000)

    origin_lat, origin_lon = 51.5000, -0.1200
    destination_lat, destination_lon = 51.9000, -0.1200
    straight_line_m = routing_graph._haversine_m(origin_lat, origin_lon, destination_lat, destination_lon)
    hop_floor = int(math.ceil((straight_line_m / 30.0) * 1.8))

    hops = routing_graph._effective_route_graph_max_hops(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        destination_lat=destination_lat,
        destination_lon=destination_lon,
    )

    assert hops == min(5000, max(220, hop_floor))
    assert hops > 220


def test_route_graph_candidate_routes_scales_initial_state_budget_with_hops(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_feasibility(**_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "origin_node_id": "a",
            "destination_node_id": "b",
        }

    def _fake_yen_with_stats(*, max_hops: int, max_state_budget: int, **_: Any):
        captured["max_hops"] = int(max_hops)
        captured["max_state_budget"] = int(max_state_budget)
        return [], {
            "explored_states": 42,
            "generated_candidates": 0,
            "no_path_reason": "state_budget_exceeded",
            "termination_reason": "state budget exceeded",
            "first_error": "",
        }

    monkeypatch.setattr(settings, "route_graph_max_state_budget", 100000)
    monkeypatch.setattr(settings, "route_graph_state_budget_per_hop", 1600)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_cap", 900000)
    monkeypatch.setattr(routing_graph, "load_route_graph", lambda: _toy_graph())
    monkeypatch.setattr(routing_graph, "route_graph_od_feasibility", _fake_feasibility)
    monkeypatch.setattr(routing_graph, "_effective_route_graph_max_hops", lambda **_: 400)
    monkeypatch.setattr(routing_graph, "yen_k_shortest_paths_with_stats", _fake_yen_with_stats)

    routes, diag = routing_graph.route_graph_candidate_routes(
        origin_lat=51.5000,
        origin_lon=-0.1300,
        destination_lat=51.5300,
        destination_lon=-0.1000,
        max_paths=4,
        scenario_edge_modifiers={},
    )

    assert routes == []
    assert diag.effective_max_hops == 400
    assert diag.effective_state_budget_initial == 640000
    assert diag.effective_state_budget == 640000
    assert captured["max_hops"] == 400
    assert captured["max_state_budget"] == 640000


def test_collect_candidate_routes_runs_reduced_state_rescue_pass(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _routes_with_rescue(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return (
                [],
                GraphCandidateDiagnostics(
                    explored_states=300000,
                    generated_paths=0,
                    emitted_paths=0,
                    candidate_budget=24,
                    effective_max_hops=700,
                    effective_state_budget=300000,
                    no_path_reason="state_budget_exceeded",
                    no_path_detail="state budget exceeded",
                ),
            )
        if len(calls) == 2:
            return (
                [],
                GraphCandidateDiagnostics(
                    explored_states=600000,
                    generated_paths=0,
                    emitted_paths=0,
                    candidate_budget=24,
                    effective_max_hops=700,
                    effective_state_budget=600000,
                    no_path_reason="state_budget_exceeded",
                    no_path_detail="state budget exceeded after retry",
                ),
            )
        return (
            [_make_graph_route()],
            GraphCandidateDiagnostics(
                explored_states=615000,
                generated_paths=1,
                emitted_paths=1,
                candidate_budget=24,
                effective_max_hops=700,
                effective_state_budget=600000,
                no_path_reason="",
                no_path_detail="",
            ),
        )

    async def _empty_refine_iter(**_: Any):
        if False:
            yield None

    monkeypatch.setattr(settings, "route_graph_max_state_budget", 300000)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_multiplier", 2.0)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_cap", 2500000)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_mode", "reduced")
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _routes_with_rescue)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _empty_refine_iter)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.74933, lon=-0.48340),
            destination=LatLng(lat=51.49899, lon=-0.12360),
            max_routes=12,
            cache_key=None,
            scenario_edge_modifiers={},
            progress_cb=None,
        )
    )

    assert routes
    assert spec_count >= 1
    assert len(calls) == 3
    assert calls[0].get("use_transition_state") is True
    assert calls[0].get("max_state_budget_override") is None
    assert calls[1].get("use_transition_state") is True
    assert int(calls[1].get("max_state_budget_override", 0)) == 600000
    assert calls[2].get("use_transition_state") is False
    assert int(calls[2].get("max_state_budget_override", 0)) == 600000
    assert any("routing_graph_search_rescued" in warning for warning in warnings)
    assert diag.graph_retry_attempted is True
    assert diag.graph_retry_outcome == "exhausted"
    assert diag.graph_rescue_attempted is True
    assert diag.graph_rescue_mode == "reduced"
    assert diag.graph_rescue_state_budget == 600000
    assert diag.graph_rescue_outcome == "succeeded"


def test_collect_candidate_routes_rescue_exhausted_keeps_no_path_reason(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _routes_rescue_exhausted(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=600000,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=24,
                effective_max_hops=700,
                effective_state_budget=600000,
                no_path_reason="state_budget_exceeded" if len(calls) < 3 else "no_path",
                no_path_detail="no path",
            ),
        )

    monkeypatch.setattr(settings, "route_graph_max_state_budget", 300000)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_multiplier", 2.0)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_cap", 2500000)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_mode", "reduced")
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _routes_rescue_exhausted)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.74933, lon=-0.48340),
            destination=LatLng(lat=51.49899, lon=-0.12360),
            max_routes=12,
            cache_key=None,
            scenario_edge_modifiers={},
            progress_cb=None,
        )
    )

    assert routes == []
    assert spec_count == 0
    assert len(calls) == 3
    assert calls[2].get("use_transition_state") is False
    assert warnings and "route_graph: routing_graph_no_path" in warnings[0]
    assert diag.graph_no_path_reason == "routing_graph_no_path"
    assert diag.graph_retry_attempted is True
    assert diag.graph_retry_outcome == "exhausted"
    assert diag.graph_rescue_attempted is True
    assert diag.graph_rescue_outcome == "exhausted"
