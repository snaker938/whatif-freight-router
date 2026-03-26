from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from typing import Any

import pytest

import app.main as main_module
import app.routing_graph as routing_graph
from app.certification_cache import clear_certification_cache
from app.k_raw_cache import clear_k_raw_cache
from app.models import GeoJSONLineString, LatLng, RouteMetrics, RouteOption, RouteRequest
from app.route_cache import clear_route_cache
from app.route_state_cache import clear_route_state_cache
from app.routing_graph import GraphCandidateDiagnostics, GraphEdge, RouteGraph
from app.settings import Settings, settings
from app.voi_dccs_cache import clear_voi_dccs_cache


@pytest.fixture(autouse=True)
def _clear_process_global_caches() -> None:
    clear_route_cache()
    clear_k_raw_cache()
    clear_certification_cache()
    clear_route_state_cache()
    clear_voi_dccs_cache()
    yield
    clear_route_cache()
    clear_k_raw_cache()
    clear_certification_cache()
    clear_route_state_cache()
    clear_voi_dccs_cache()


class _NoopOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return []


class _FailingOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        raise AssertionError("fallback-seeded routes should not be re-fetched from OSRM")


class _NoopORS:
    async def fetch_route(self, **_: Any) -> Any:
        raise AssertionError("local ORS should not be used in tests that do not trigger supplemental rescue")


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


def test_select_ranked_candidate_routes_prefilter_keeps_route_diversity(monkeypatch) -> None:
    monkeypatch.setattr(settings, "route_candidate_alternatives_max", 24)
    monkeypatch.setattr(settings, "route_candidate_prefilter_multiplier", 3)

    routes = [
        _make_ranked_route(duration_s=100.0 + idx, lon_seed=0.001 * idx, road_class="motorway")
        for idx in range(5)
    ] + [
        _make_ranked_route(duration_s=103.0 + idx, lon_seed=0.08 + (0.001 * idx), road_class="primary")
        for idx in range(3)
    ]

    selected = main_module._select_ranked_candidate_routes(routes, max_routes=2)
    families = {
        main_module._graph_family_signature(route)
        for route in selected
        if main_module._graph_family_signature(route)
    }

    assert len(selected) == 6
    assert len(families) >= 2


def test_collect_candidate_routes_skips_baseline_rerun_when_separability_not_enforced(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_once(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [_make_graph_route(seed=0.0 + (0.1 * idx)) for idx in range(5)],
            GraphCandidateDiagnostics(
                explored_states=1200,
                generated_paths=5,
                emitted_paths=5,
                candidate_budget=12,
            ),
        )

    async def _empty_refine_iter(**_: Any):
        if False:
            yield None

    monkeypatch.setattr(settings, "route_graph_scenario_separability_fail", False)
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_once)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _empty_refine_iter)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.74933, lon=-0.48340),
            destination=LatLng(lat=51.49899, lon=-0.12360),
            max_routes=6,
            cache_key=None,
            scenario_edge_modifiers={"duration_multiplier": 1.25},
            progress_cb=None,
        )
    )

    assert routes
    assert spec_count >= 1
    assert len(calls) == 1
    assert diag.scenario_candidate_gate_action == "skipped_non_enforcing"
    assert warnings == []


def test_settings_accept_route_graph_retry_cap_eight_million() -> None:
    instance = Settings(ROUTE_GRAPH_STATE_BUDGET_RETRY_CAP=8_000_000)
    assert instance.route_graph_state_budget_retry_cap == 8_000_000


def test_collect_candidate_routes_uses_reduced_initial_for_long_corridor(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_once(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [_make_graph_route(seed=0.0)],
            GraphCandidateDiagnostics(
                explored_states=90,
                generated_paths=1,
                emitted_paths=1,
                candidate_budget=12,
            ),
        )

    async def _empty_refine_iter(**_: Any):
        if False:
            yield None

    monkeypatch.setattr(settings, "route_graph_reduced_initial_for_long_corridor", True)
    monkeypatch.setattr(settings, "route_graph_skip_initial_search_long_corridor", False)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 10.0)
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_once)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _empty_refine_iter)

    routes, _warnings, _spec_count, _diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=54.9783, lon=-1.6178),
            destination=LatLng(lat=53.0027, lon=-2.1794),
            max_routes=6,
            cache_key=None,
            scenario_edge_modifiers={"duration_multiplier": 1.1},
            progress_cb=None,
        )
    )

    assert routes
    assert calls
    assert calls[0].get("use_transition_state") is False


def test_collect_candidate_routes_uses_reduced_initial_for_medium_long_corridor(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_once(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [_make_graph_route(seed=0.0 + (0.1 * idx)) for idx in range(5)],
            GraphCandidateDiagnostics(
                explored_states=1200,
                generated_paths=5,
                emitted_paths=5,
                candidate_budget=12,
            ),
        )

    async def _empty_refine_iter(**_: Any):
        if False:
            yield None

    monkeypatch.setattr(settings, "route_graph_reduced_initial_for_long_corridor", True)
    monkeypatch.setattr(settings, "route_graph_skip_initial_search_long_corridor", False)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_once)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _empty_refine_iter)

    routes, _warnings, _spec_count, _diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=52.4862, lon=-1.8904),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            max_routes=6,
            cache_key=None,
            scenario_edge_modifiers={"duration_multiplier": 1.1},
            progress_cb=None,
        )
    )

    assert routes
    assert len(calls) == 1
    assert calls[0].get("use_transition_state") is False
    assert calls[0].get("max_state_budget_override") is None


def test_route_graph_k_raw_search_uses_reduced_initial_for_medium_long_corridor(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_once(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [_make_graph_route(seed=0.0 + (0.1 * idx)) for idx in range(5)],
            GraphCandidateDiagnostics(
                explored_states=1200,
                generated_paths=5,
                emitted_paths=5,
                candidate_budget=12,
            ),
        )

    monkeypatch.setattr(settings, "route_graph_reduced_initial_for_long_corridor", True)
    monkeypatch.setattr(settings, "route_graph_skip_initial_search_long_corridor", False)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_once)

    routes, _diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=52.4862, lon=-1.8904),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.1},
            start_node_id="a",
            goal_node_id="b",
        )
    )

    assert routes
    assert len(calls) == 1
    assert calls[0].get("use_transition_state") is False
    assert meta.get("graph_supplemental_probe_attempted") is False


def test_route_graph_k_raw_search_runs_supplemental_probe_for_underfilled_medium_long_corridor(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_probe(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return (
                [_make_graph_route(seed=0.0)],
                GraphCandidateDiagnostics(
                    explored_states=120,
                    generated_paths=1,
                    emitted_paths=1,
                    candidate_budget=12,
                ),
            )
        return (
            [_make_graph_route(seed=0.2)],
            GraphCandidateDiagnostics(
                explored_states=90,
                generated_paths=1,
                emitted_paths=1,
                candidate_budget=12,
            ),
        )

    monkeypatch.setattr(settings, "route_graph_reduced_initial_for_long_corridor", True)
    monkeypatch.setattr(settings, "route_graph_skip_initial_search_long_corridor", False)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_dccs_bootstrap_count", 2)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_probe)

    routes, _diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=52.4862, lon=-1.8904),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.1},
            start_node_id="a",
            goal_node_id="b",
        )
    )

    assert len(routes) == 2
    assert len(calls) == 2
    assert calls[0].get("use_transition_state") is False
    assert calls[1].get("use_transition_state") is True
    assert meta.get("graph_supplemental_probe_attempted") is True
    assert meta.get("graph_supplemental_target") == 4
    assert float(meta.get("graph_search_ms_supplemental", 0.0)) >= 0.0


def test_route_graph_k_raw_search_skips_retry_rescue_for_supported_short_corridor_ambiguity(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_exhausted(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=250000,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=12,
                effective_state_budget=300000,
                no_path_reason="path_search_exhausted",
                no_path_detail="deadline reached",
            ),
        )

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_skip_retry_rescue_reliability_corridor", False)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_exhausted)

    routes, diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=51.5007, lon=-3.2007),
            destination=LatLng(lat=51.4816, lon=-2.5892),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.05},
            start_node_id="a",
            goal_node_id="b",
            od_ambiguity_index=0.72,
            od_engine_disagreement_prior=0.69,
            od_hard_case_prior=0.41,
            od_ambiguity_support_ratio=0.81,
            od_ambiguity_source_entropy=0.63,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert routes == []
    assert len(calls) == 1
    assert diag.no_path_reason == "path_search_exhausted"
    assert meta.get("graph_supported_ambiguity_fast_fallback") is True
    assert meta.get("graph_retry_attempted") is True
    assert meta.get("graph_retry_outcome") == "skipped_supported_ambiguity_fast_fallback"
    assert meta.get("graph_rescue_attempted") is True
    assert meta.get("graph_rescue_mode") == "supported_ambiguity_fast_fallback"
    assert meta.get("graph_rescue_outcome") == "skipped_supported_ambiguity_fast_fallback"


def test_collect_candidate_routes_support_aware_weak_support_skips_initial_search(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_should_not_run(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        raise AssertionError("support-aware weak-support rows should skip the expensive initial graph search")

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_skip_retry_rescue_reliability_corridor", False)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_should_not_run)

    routes, diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=51.5007, lon=-3.2007),
            destination=LatLng(lat=51.4816, lon=-2.5892),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.05},
            start_node_id="a",
            goal_node_id="b",
            od_ambiguity_index=0.16,
            od_engine_disagreement_prior=0.12,
            od_hard_case_prior=0.08,
            od_ambiguity_support_ratio=0.18,
            od_ambiguity_source_entropy=0.10,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert routes == []
    assert calls == []
    assert diag.no_path_reason == "skipped_support_aware_graph_search"
    assert meta.get("graph_supported_ambiguity_fast_fallback") is False
    assert meta.get("graph_low_ambiguity_fast_path") is False
    assert meta.get("graph_retry_attempted") is True
    assert meta.get("graph_retry_outcome") == "skipped_support_aware_graph_search"
    assert meta.get("graph_rescue_attempted") is True
    assert meta.get("graph_rescue_mode") == "support_aware_fast_fallback"
    assert meta.get("graph_rescue_outcome") == "skipped_support_aware_graph_search"
    assert meta.get("graph_search_ms_initial") == 0.0
    assert meta.get("graph_search_ms_retry") == 0.0
    assert meta.get("graph_search_ms_rescue") == 0.0


def test_collect_candidate_routes_support_rich_short_haul_skips_initial_search(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_should_not_run(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        raise AssertionError("support-rich short-haul rows should skip the expensive initial graph search")

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", False)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_should_not_run)

    routes, diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=51.5007, lon=-3.2007),
            destination=LatLng(lat=51.4816, lon=-2.5892),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.05},
            start_node_id="a",
            goal_node_id="b",
            od_ambiguity_index=0.78,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.22,
            od_ambiguity_support_ratio=0.83,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=3,
            od_corridor_family_count=2,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert routes == []
    assert calls == []
    assert diag.no_path_reason == "skipped_support_rich_short_haul_graph_search"
    assert meta.get("graph_supported_ambiguity_fast_fallback") is True
    assert meta.get("graph_low_ambiguity_fast_path") is False
    assert meta.get("graph_retry_attempted") is True
    assert meta.get("graph_retry_outcome") == "skipped_support_rich_short_haul_graph_search"
    assert meta.get("graph_rescue_attempted") is False
    assert meta.get("graph_search_ms_initial") == 0.0
    assert meta.get("graph_search_ms_retry") == 0.0
    assert meta.get("graph_search_ms_rescue") == 0.0


def test_route_graph_k_raw_search_support_rich_short_haul_keeps_graph_search_for_richer_corridor_counts(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_once(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [_make_graph_route(seed=0.15)],
            GraphCandidateDiagnostics(
                explored_states=1200,
                generated_paths=2,
                emitted_paths=1,
                candidate_budget=6,
            ),
        )

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_once)

    routes, diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=51.5007, lon=-3.2007),
            destination=LatLng(lat=51.4816, lon=-2.5892),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.05},
            start_node_id="a",
            goal_node_id="b",
            od_ambiguity_index=0.78,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.22,
            od_ambiguity_support_ratio=0.83,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=10,
            od_corridor_family_count=3,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert routes
    assert len(calls) == 1
    assert diag.no_path_reason == ""
    assert meta.get("graph_support_rich_short_haul_fast_fallback") is False
    assert float(meta.get("graph_search_ms_initial", 0.0)) >= 0.0


def test_route_graph_k_raw_search_weak_support_short_haul_does_not_activate_fast_path(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_once(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [_make_graph_route(seed=0.18)],
            GraphCandidateDiagnostics(
                explored_states=1400,
                generated_paths=2,
                emitted_paths=1,
                candidate_budget=6,
            ),
        )

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_once)

    routes, diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=51.5007, lon=-3.2007),
            destination=LatLng(lat=51.4816, lon=-2.5892),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.05},
            start_node_id="a",
            goal_node_id="b",
            od_ambiguity_index=0.22,
            od_engine_disagreement_prior=0.19,
            od_hard_case_prior=0.16,
            od_ambiguity_support_ratio=0.52,
            od_ambiguity_source_entropy=0.34,
            od_candidate_path_count=2,
            od_corridor_family_count=1,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert routes
    assert len(calls) == 1
    assert diag.no_path_reason == ""
    assert meta.get("graph_support_rich_short_haul_fast_fallback") is False
    assert meta.get("graph_low_ambiguity_fast_path") is False


@pytest.mark.parametrize(
    (
        "od_nominal_margin_proxy",
        "od_objective_spread",
        "od_ambiguity_support_ratio",
        "od_ambiguity_source_entropy",
        "expected_skip",
        "expected_calls",
    ),
    [
        (0.97, 0.18, 0.64, 0.78, True, 0),
        (0.97, 0.18, 0.58, 0.41, False, 1),
        (0.83, 0.34, 0.72, 0.86, False, 1),
    ],
)
def test_route_graph_k_raw_search_support_backed_single_corridor_fast_path_is_cache_safe_and_narrow(
    monkeypatch,
    od_nominal_margin_proxy: float,
    od_objective_spread: float,
    od_ambiguity_support_ratio: float,
    od_ambiguity_source_entropy: float,
    expected_skip: bool,
    expected_calls: int,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_once(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [_make_graph_route(seed=0.21)],
            GraphCandidateDiagnostics(
                explored_states=900,
                generated_paths=2,
                emitted_paths=1,
                candidate_budget=6,
            ),
        )

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_once)

    routes, diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=51.4816, lon=-3.1791),
            destination=LatLng(lat=51.4545, lon=-2.5879),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.05},
            start_node_id="a",
            goal_node_id="b",
            od_ambiguity_index=0.74,
            od_engine_disagreement_prior=0.71,
            od_hard_case_prior=0.24,
            od_ambiguity_support_ratio=od_ambiguity_support_ratio,
            od_ambiguity_source_entropy=od_ambiguity_source_entropy,
            od_candidate_path_count=3,
            od_corridor_family_count=1,
            od_nominal_margin_proxy=od_nominal_margin_proxy,
            od_objective_spread=od_objective_spread,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert len(calls) == expected_calls
    assert bool(diag.no_path_reason == "skipped_support_backed_single_corridor_graph_search") is expected_skip
    assert bool(meta.get("graph_support_backed_single_corridor_fast_fallback")) is expected_skip
    if expected_skip:
        assert routes == []
        assert diag.no_path_reason == "skipped_support_backed_single_corridor_graph_search"
        assert meta.get("graph_retry_outcome") == "skipped_support_backed_single_corridor_graph_search"
    else:
        assert routes
        assert diag.no_path_reason == ""
        assert meta.get("graph_support_backed_single_corridor_fast_fallback") is False
        assert meta.get("graph_low_ambiguity_fast_path") is False


def test_collect_candidate_routes_medium_long_search_exhaustion_uses_osrm_fallback(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    refine_calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_exhausted(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            detail = "deadline reached"
        elif len(calls) == 2:
            detail = "deadline reached after retry"
        else:
            detail = "deadline reached after rescue"
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=500000,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=12,
                effective_state_budget=600000,
                no_path_reason="path_search_exhausted",
                no_path_detail=detail,
            ),
        )

    async def _fallback_iter(**kwargs: Any):
        refine_calls.append(dict(kwargs))
        specs = list(kwargs.get("specs", []))
        assert specs
        yield main_module.CandidateProgress(
            done=1,
            total=len(specs),
            result=main_module.CandidateFetchResult(
                spec=specs[0],
                routes=[_make_graph_route(seed=0.25)],
            ),
        )

    monkeypatch.setattr(settings, "route_graph_reduced_initial_for_long_corridor", True)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_skip_initial_search_long_corridor", False)
    monkeypatch.setattr(settings, "route_graph_skip_retry_rescue_reliability_corridor", False)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_mode", "reduced")
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_multiplier", 2.0)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_cap", 2500000)
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_exhausted)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fallback_iter)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=52.4862, lon=-1.8904),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            max_routes=6,
            cache_key=None,
            scenario_edge_modifiers={"duration_multiplier": 1.1},
            progress_cb=None,
        )
    )

    assert routes
    assert spec_count >= 1
    assert len(calls) == 3
    assert refine_calls
    assert calls[0].get("use_transition_state") is False
    assert calls[1].get("use_transition_state") is False
    assert calls[2].get("use_transition_state") is False
    assert any("routing_graph_search_exhausted_osrm_fallback" in warning for warning in warnings)
    assert diag.scenario_candidate_gate_action == "medium_long_osrm_fallback"
    assert diag.graph_retry_attempted is True
    assert diag.graph_retry_outcome == "exhausted"
    assert diag.graph_rescue_attempted is True
    assert diag.graph_rescue_outcome == "exhausted"


def test_collect_candidate_routes_skips_retry_rescue_for_reliability_corridor_exhaustion(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    refine_calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_exhausted(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=500000,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=12,
                effective_state_budget=600000,
                no_path_reason="path_search_exhausted",
                no_path_detail="deadline reached",
            ),
        )

    async def _fallback_iter(**kwargs: Any):
        refine_calls.append(dict(kwargs))
        specs = list(kwargs.get("specs", []))
        assert specs
        yield main_module.CandidateProgress(
            done=1,
            total=len(specs),
            result=main_module.CandidateFetchResult(
                spec=specs[0],
                routes=[_make_graph_route(seed=0.25)],
            ),
        )

    monkeypatch.setattr(settings, "route_graph_reduced_initial_for_long_corridor", True)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_skip_initial_search_long_corridor", False)
    monkeypatch.setattr(settings, "route_graph_skip_retry_rescue_reliability_corridor", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_mode", "reduced")
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_multiplier", 2.0)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_cap", 2500000)
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_exhausted)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fallback_iter)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=52.4862, lon=-1.8904),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            max_routes=6,
            cache_key=None,
            scenario_edge_modifiers={"duration_multiplier": 1.1},
            progress_cb=None,
        )
    )

    assert routes
    assert spec_count >= 1
    assert len(calls) == 1
    assert refine_calls
    assert any("routing_graph_search_exhausted_osrm_fallback" in warning for warning in warnings)
    assert diag.scenario_candidate_gate_action == "medium_long_osrm_fallback"
    assert diag.graph_retry_attempted is True
    assert diag.graph_retry_outcome == "skipped_reliability_corridor_fast_fallback"
    assert diag.graph_rescue_attempted is True
    assert diag.graph_rescue_outcome == "skipped_reliability_corridor_fast_fallback"


def test_collect_candidate_routes_short_haul_corridor_uniform_no_path_fail_opens_to_osrm_without_retry_rescue(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    refine_calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_no_path(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=2500,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=8,
                effective_state_budget=5000,
                no_path_reason="no_path",
                no_path_detail="short_haul_corridor_uniform_no_path",
            ),
        )

    async def _fallback_iter(**kwargs: Any):
        refine_calls.append(dict(kwargs))
        specs = list(kwargs.get("specs", []))
        assert specs
        yield main_module.CandidateProgress(
            done=1,
            total=len(specs),
            result=main_module.CandidateFetchResult(
                spec=specs[0],
                routes=[_make_graph_route(seed=0.35)],
            ),
        )

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_mode", "reduced")
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_no_path)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fallback_iter)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.4816, lon=-3.1791),
            destination=LatLng(lat=51.3758, lon=-2.3599),
            max_routes=4,
            cache_key=None,
            scenario_edge_modifiers={},
            refinement_policy="corridor_uniform",
            search_budget=2,
            run_seed=20260324,
            progress_cb=None,
        )
    )

    assert routes
    assert spec_count >= 1
    assert len(calls) == 1
    assert refine_calls
    assert any("routing_graph_legacy_corridor_uniform_osrm_fallback" in warning for warning in warnings)
    assert diag.scenario_candidate_gate_action == "legacy_corridor_uniform_osrm_fallback"
    assert diag.graph_no_path_reason == "no_path"
    assert diag.graph_retry_attempted is False
    assert diag.graph_retry_outcome == "not_applicable"
    assert diag.graph_rescue_attempted is True
    assert diag.graph_rescue_mode == "legacy_corridor_uniform_fast_fallback"
    assert diag.graph_rescue_outcome == "no_path"


def test_collect_candidate_routes_short_haul_corridor_uniform_support_rich_fast_fallback_skips_graph_search(
    monkeypatch,
) -> None:
    refine_calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_should_not_run(**_: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        raise AssertionError("support-rich short-haul legacy corridor_uniform rows should skip the expensive graph search")

    async def _fallback_iter(**kwargs: Any):
        refine_calls.append(dict(kwargs))
        specs = list(kwargs.get("specs", []))
        assert specs
        yield main_module.CandidateProgress(
            done=1,
            total=len(specs),
            result=main_module.CandidateFetchResult(
                spec=specs[0],
                routes=[_make_graph_route(seed=0.45)],
            ),
        )

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_mode", "reduced")
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_should_not_run)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fallback_iter)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.4816, lon=-3.1791),
            destination=LatLng(lat=51.4545, lon=-2.5879),
            max_routes=4,
            cache_key=None,
            scenario_edge_modifiers={},
            refinement_policy="corridor_uniform",
            search_budget=2,
            run_seed=20260325,
            od_ambiguity_index=0.78,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.22,
            od_ambiguity_support_ratio=0.83,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=3,
            od_corridor_family_count=2,
            allow_supported_ambiguity_fast_fallback=True,
            progress_cb=None,
        )
    )

    assert routes
    assert spec_count >= 1
    assert refine_calls
    assert any("routing_graph_legacy_corridor_uniform_osrm_fallback" in warning for warning in warnings)
    assert diag.scenario_candidate_gate_action == "legacy_corridor_uniform_osrm_fallback"
    assert diag.graph_no_path_reason == "skipped_support_rich_short_haul_graph_search"
    assert diag.graph_retry_attempted is False
    assert diag.graph_retry_outcome == "not_applicable"
    assert diag.graph_rescue_attempted is True
    assert diag.graph_rescue_mode == "legacy_corridor_uniform_fast_fallback"
    assert diag.graph_rescue_outcome == "skipped_support_rich_short_haul_graph_search"


def test_collect_candidate_routes_short_haul_corridor_uniform_requires_path_family_support(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    refine_calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_no_path(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=2500,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=8,
                effective_state_budget=5000,
                no_path_reason="no_path",
                no_path_detail="short_haul_corridor_uniform_no_path",
            ),
        )

    async def _fallback_iter(**kwargs: Any):
        refine_calls.append(dict(kwargs))
        specs = list(kwargs.get("specs", []))
        assert specs
        yield main_module.CandidateProgress(
            done=1,
            total=len(specs),
            result=main_module.CandidateFetchResult(
                spec=specs[0],
                routes=[_make_graph_route(seed=0.4)],
            ),
        )

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_mode", "reduced")
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_no_path)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fallback_iter)

    routes, warnings, spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.4816, lon=-3.1791),
            destination=LatLng(lat=51.3758, lon=-2.3599),
            max_routes=4,
            cache_key=None,
            scenario_edge_modifiers={},
            refinement_policy="corridor_uniform",
            search_budget=2,
            run_seed=20260324,
            od_ambiguity_index=0.78,
            od_engine_disagreement_prior=0.91,
            od_hard_case_prior=0.22,
            od_ambiguity_support_ratio=0.83,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=10,
            od_corridor_family_count=3,
            allow_supported_ambiguity_fast_fallback=True,
            progress_cb=None,
        )
    )

    assert routes
    assert spec_count >= 1
    assert len(calls) == 1
    assert refine_calls
    assert any("routing_graph_legacy_corridor_uniform_osrm_fallback" in warning for warning in warnings)
    assert diag.graph_no_path_reason == "no_path"
    assert diag.graph_retry_attempted is False
    assert diag.graph_rescue_attempted is True
    assert diag.graph_rescue_mode == "legacy_corridor_uniform_fast_fallback"
    assert diag.graph_rescue_outcome == "no_path"


def test_collect_candidate_routes_retries_on_path_search_exhausted(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_retry(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return (
                [],
                GraphCandidateDiagnostics(
                    explored_states=1000,
                    generated_paths=0,
                    emitted_paths=0,
                    candidate_budget=12,
                    effective_state_budget=1200,
                    no_path_reason="path_search_exhausted",
                    no_path_detail="deadline reached",
                ),
            )
        return (
            [_make_graph_route(seed=0.0)],
            GraphCandidateDiagnostics(
                explored_states=200,
                generated_paths=1,
                emitted_paths=1,
                candidate_budget=12,
                effective_state_budget=2400,
            ),
        )

    async def _empty_refine_iter(**_: Any):
        if False:
            yield None

    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", False)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_multiplier", 2.0)
    monkeypatch.setattr(main_module, "_route_graph_status_async", _ok_status)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_retry)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _empty_refine_iter)

    routes, _warnings, _spec_count, diag = asyncio.run(
        main_module._collect_candidate_routes(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=52.0, lon=-1.0),
            destination=LatLng(lat=52.3, lon=-1.6),
            max_routes=6,
            cache_key=None,
            scenario_edge_modifiers={"duration_multiplier": 1.2},
            progress_cb=None,
        )
    )

    assert routes
    assert len(calls) >= 2
    assert diag.graph_retry_attempted is True
    assert diag.graph_retry_outcome == "succeeded"


@pytest.mark.parametrize(
    ("graph_reason", "graph_detail"),
    [
        (
            "skipped_long_corridor_graph_search",
            "Skipped expensive long-corridor graph search and used bounded fallback.",
        ),
        (
            "path_search_exhausted",
            "Graph search exhausted the bounded long-corridor state budget before emitting candidates.",
        ),
    ],
)
def test_compute_direct_route_pipeline_recovers_long_corridor_k_raw_with_osrm_fallback(
    monkeypatch,
    graph_reason: str,
    graph_detail: str,
) -> None:
    fallback_route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    toll_free_route = _make_ranked_route(duration_s=28_800.0, lon_seed=0.2, road_class="primary")

    async def _fake_scenario_context_from_od(**_: Any) -> dict[str, Any]:
        return {"bucket": "clear"}

    async def _fake_scenario_candidate_modifiers_async(**_: Any) -> dict[str, Any]:
        return {}

    def _fake_feasibility(**_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "origin_node_id": "a",
            "destination_node_id": "b",
        }

    async def _fake_k_raw_search(**_: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics, dict[str, Any]]:
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=0,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=8,
                no_path_reason=graph_reason,
                no_path_detail=graph_detail,
            ),
            {
                "graph_retry_attempted": True,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "skipped_long_corridor_fast_fallback",
                "graph_rescue_attempted": True,
                "graph_rescue_mode": "long_corridor_fast_fallback",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "skipped_long_corridor_fast_fallback",
                "graph_search_ms_initial": 0.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        specs = list(kwargs.get("specs", []))
        assert specs
        alt_spec = specs[0]
        toll_spec = next(spec for spec in specs if "exclude:toll" in spec.label)
        yield main_module.CandidateProgress(
            done=1,
            total=2,
            result=main_module.CandidateFetchResult(spec=alt_spec, routes=[dict(fallback_route)]),
        )
        yield main_module.CandidateProgress(
            done=2,
            total=2,
            result=main_module.CandidateFetchResult(spec=toll_spec, routes=[dict(toll_free_route)]),
        )

    async def _fake_refine_graph_candidate_batch(**kwargs: Any):
        selected_records = list(kwargs.get("selected_records", []))
        raw_graph_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed_costs = {
            str(record.candidate_id): 14.0
            for record in selected_records
        }
        refined = [
            dict(raw_graph_routes_by_id.get(str(record.candidate_id), fallback_route))
            for record in selected_records
        ]
        return refined, [], observed_costs, len(selected_records), 25.0

    def _fake_build_options(routes: list[dict[str, Any]], **_: Any):
        options: list[RouteOption] = []
        for idx, route in enumerate(routes, start=1):
            coords = [
                (float(point[0]), float(point[1]))
                for point in route["geometry"]["coordinates"]
            ]
            options.append(
                RouteOption(
                    id=f"route_{idx}",
                    geometry=GeoJSONLineString(type="LineString", coordinates=coords),
                    metrics=RouteMetrics(
                        distance_km=float(route["distance"]) / 1000.0,
                        duration_s=float(route["duration"]),
                        monetary_cost=325.0,
                        emissions_kg=210.0,
                        avg_speed_kmh=70.0,
                    ),
                )
            )
        return options, [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _fake_scenario_context_from_od)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _fake_scenario_candidate_modifiers_async)
    monkeypatch.setattr(main_module, "route_graph_od_feasibility", _fake_feasibility)
    monkeypatch.setattr(main_module, "_route_graph_k_raw_search", _fake_k_raw_search)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fake_iter_candidate_fetches)
    monkeypatch.setattr(main_module, "_refine_graph_candidate_batch", _fake_refine_graph_candidate_batch)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: list(options)[0])
    monkeypatch.setattr(
        main_module,
        "_route_selection_score_map",
        lambda options, **_: {str(option.id): float(index) for index, option in enumerate(options, start=1)},
    )

    req = RouteRequest(
        origin=LatLng(lat=51.5074, lon=-0.1278),
        destination=LatLng(lat=53.4808, lon=-2.2426),
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        max_alternatives=4,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=4,
            pipeline_mode="dccs",
            run_seed=20260321,
        )
    )

    assert result["selected"].id == "route_1"
    assert result["candidate_fetches"] >= 2
    assert result["candidate_diag"].raw_count == 2
    assert "routing_graph_long_corridor_k_raw_fallback" in " ".join(result["warnings"])
    assert "routing_graph_long_corridor_toll_exclusion_prioritized" in " ".join(result["warnings"])


def test_compute_direct_route_pipeline_recovers_support_aware_k_raw_with_osrm_fallback(
    monkeypatch,
) -> None:
    fallback_route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    toll_free_route = _make_ranked_route(duration_s=28_800.0, lon_seed=0.2, road_class="primary")
    fetched_labels: list[str] = []
    seed_calls: list[dict[str, Any]] = []

    async def _fake_scenario_context_from_od(**_: Any) -> dict[str, Any]:
        return {"bucket": "clear"}

    async def _fake_scenario_candidate_modifiers_async(**_: Any) -> dict[str, Any]:
        return {}

    def _fake_feasibility(**_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "origin_node_id": "a",
            "destination_node_id": "b",
        }

    async def _fake_k_raw_search(**_: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics, dict[str, Any]]:
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=0,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=8,
                no_path_reason="skipped_support_rich_short_haul_graph_search",
                no_path_detail="Skipped expensive graph search because support-rich short-haul evidence made bounded direct fallback preferable.",
            ),
            {
                "graph_retry_attempted": True,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "skipped_support_rich_short_haul_graph_search",
                "graph_rescue_attempted": True,
                "graph_rescue_mode": "support_aware_fast_fallback",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "skipped_support_rich_short_haul_graph_search",
                "graph_supported_ambiguity_fast_fallback": False,
                "graph_search_ms_initial": 0.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _fake_fetch_local_ors_baseline_seed(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        seed_calls.append(dict(kwargs))
        return (
            _make_ranked_route(duration_s=27_900.0, lon_seed=0.35, road_class="trunk"),
            {"provider_mode": "local_service", "engine_profile": "driving-hgv"},
        )

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        specs = list(kwargs.get("specs", []))
        assert specs
        fetched_labels.extend(str(spec.label) for spec in specs)
        alt_spec = specs[0]
        toll_spec = next(spec for spec in specs if "exclude:toll" in spec.label)
        yield main_module.CandidateProgress(
            done=1,
            total=2,
            result=main_module.CandidateFetchResult(spec=alt_spec, routes=[dict(fallback_route)]),
        )
        yield main_module.CandidateProgress(
            done=2,
            total=2,
            result=main_module.CandidateFetchResult(spec=toll_spec, routes=[dict(toll_free_route)]),
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
        options: list[RouteOption] = []
        for idx, route in enumerate(routes, start=1):
            coords = [
                (float(point[0]), float(point[1]))
                for point in route["geometry"]["coordinates"]
            ]
            options.append(
                RouteOption(
                    id=f"route_{idx}",
                    geometry=GeoJSONLineString(type="LineString", coordinates=coords),
                    metrics=RouteMetrics(
                        distance_km=float(route["distance"]) / 1000.0,
                        duration_s=float(route["duration"]),
                        monetary_cost=325.0,
                        emissions_kg=210.0,
                        avg_speed_kmh=70.0,
                    ),
                )
            )
        return options, [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _fake_scenario_context_from_od)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _fake_scenario_candidate_modifiers_async)
    monkeypatch.setattr(main_module, "route_graph_od_feasibility", _fake_feasibility)
    monkeypatch.setattr(main_module, "_route_graph_k_raw_search", _fake_k_raw_search)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fake_iter_candidate_fetches)
    monkeypatch.setattr(main_module, "_fetch_local_ors_baseline_seed", _fake_fetch_local_ors_baseline_seed)
    monkeypatch.setattr(main_module, "_refine_graph_candidate_batch", _fake_refine_graph_candidate_batch)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: list(options)[0])
    monkeypatch.setattr(
        main_module,
        "_route_selection_score_map",
        lambda options, **_: {str(option.id): float(index) for index, option in enumerate(options, start=1)},
    )
    monkeypatch.setattr(settings, "route_graph_direct_k_raw_fallback_include_ors_seed", False)

    req = RouteRequest(
        origin=LatLng(lat=51.5074, lon=-0.1278),
        destination=LatLng(lat=53.4808, lon=-2.2426),
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        max_alternatives=4,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=4,
            pipeline_mode="dccs",
            run_seed=20260321,
        )
    )

    assert result["selected"].id == "route_1"
    assert result["candidate_fetches"] >= 2
    assert result["candidate_diag"].raw_count == 2
    assert result["candidate_diag"].graph_no_path_reason == "skipped_support_rich_short_haul_graph_search"
    assert "routing_graph_unavailable" not in " ".join(result["warnings"])
    assert fetched_labels
    assert all(label.startswith("support_fallback:") for label in fetched_labels)
    assert not any(label.startswith("fallback:") for label in fetched_labels)
    assert "routing_graph_support_aware_k_raw_fallback" in " ".join(result["warnings"])
    assert "routing_graph_long_corridor_k_raw_fallback" not in " ".join(result["warnings"])
    assert seed_calls == []


def test_compute_direct_route_pipeline_suppresses_preemptive_comparator_seed_on_support_aware_fallback_with_usable_raw_corridors(
    monkeypatch,
) -> None:
    fallback_route = _make_ranked_route(duration_s=27_800.0, lon_seed=0.05, road_class="trunk")
    iter_fetch_calls = 0

    async def _fake_scenario_context_from_od(**_: Any) -> dict[str, Any]:
        return {"bucket": "clear"}

    async def _fake_scenario_candidate_modifiers_async(**_: Any) -> dict[str, Any]:
        return {}

    def _fake_feasibility(**_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "origin_node_id": "a",
            "destination_node_id": "b",
        }

    async def _fake_k_raw_search(**_: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics, dict[str, Any]]:
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=0,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=8,
                no_path_reason="skipped_supported_ambiguity_fast_fallback",
                no_path_detail="Skipped expensive graph search because support-aware fallback already had usable raw corridors.",
            ),
            {
                "graph_retry_attempted": True,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "skipped_supported_ambiguity_fast_fallback",
                "graph_rescue_attempted": True,
                "graph_rescue_mode": "support_aware_fast_fallback",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "skipped_supported_ambiguity_fast_fallback",
                "graph_supported_ambiguity_fast_fallback": True,
                "graph_support_rich_short_haul_fast_fallback": False,
                "graph_search_ms_initial": 0.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        nonlocal iter_fetch_calls
        iter_fetch_calls += 1
        specs = list(kwargs.get("specs", []))
        assert specs
        yield main_module.CandidateProgress(
            done=1,
            total=len(specs),
            result=main_module.CandidateFetchResult(spec=specs[0], routes=[dict(fallback_route)]),
        )

    async def _fake_refine_graph_candidate_batch(**kwargs: Any):
        selected_records = list(kwargs.get("selected_records", []))
        raw_graph_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed_costs = {str(record.candidate_id): 13.0 for record in selected_records}
        refined = [
            dict(raw_graph_routes_by_id.get(str(record.candidate_id), fallback_route))
            for record in selected_records
        ]
        return refined, [], observed_costs, len(selected_records), 19.0

    def _fake_build_options(routes: list[dict[str, Any]], **_: Any):
        options: list[RouteOption] = []
        for index, route in enumerate(routes):
            coords = [
                (float(point[0]), float(point[1]))
                for point in route["geometry"]["coordinates"]
            ]
            options.append(
                RouteOption(
                    id=f"route_{index}",
                    geometry=GeoJSONLineString(type="LineString", coordinates=coords),
                    metrics=RouteMetrics(
                        distance_km=float(route["distance"]) / 1000.0,
                        duration_s=float(route["duration"]),
                        monetary_cost=190.0,
                        emissions_kg=115.0,
                        avg_speed_kmh=68.0,
                    ),
                )
            )
        return options, [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(settings, "route_graph_direct_k_raw_fallback_include_ors_seed", False)
    monkeypatch.setattr(settings, "route_dccs_preemptive_comparator_seed_enabled", True)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _fake_scenario_context_from_od)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _fake_scenario_candidate_modifiers_async)
    monkeypatch.setattr(main_module, "route_graph_od_feasibility", _fake_feasibility)
    monkeypatch.setattr(main_module, "_route_graph_k_raw_search", _fake_k_raw_search)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fake_iter_candidate_fetches)
    monkeypatch.setattr(main_module, "_refine_graph_candidate_batch", _fake_refine_graph_candidate_batch)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: list(options)[0])
    monkeypatch.setattr(
        main_module,
        "_route_selection_score_map",
        lambda options, **_: {str(option.id): float(index) for index, option in enumerate(options, start=1)},
    )

    req = RouteRequest(
        origin=LatLng(lat=51.4816, lon=-3.1791),
        destination=LatLng(lat=51.4545, lon=-2.5879),
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        max_alternatives=4,
        od_ambiguity_index=0.82,
        od_engine_disagreement_prior=0.91,
        od_hard_case_prior=0.62,
        od_ambiguity_support_ratio=0.84,
        od_ambiguity_source_entropy=0.78,
        od_candidate_path_count=3,
        od_corridor_family_count=2,
        allow_supported_ambiguity_fast_fallback=True,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=4,
            pipeline_mode="dccs",
            run_seed=20260326,
        )
    )

    dccs_summary = result["extra_json_artifacts"]["dccs_summary.json"]
    assert iter_fetch_calls == 1
    assert dccs_summary["preemptive_comparator_seed_activated"] is False
    assert dccs_summary["preemptive_comparator_candidate_count"] == 0
    assert dccs_summary["selected_from_preemptive_comparator_seed"] is False
    assert dccs_summary["preemptive_comparator_trigger_reason"] == ""


def test_compute_direct_route_pipeline_handles_dropped_refined_routes_without_crashing(
    monkeypatch,
) -> None:
    fallback_route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    toll_free_route = _make_ranked_route(duration_s=28_800.0, lon_seed=0.2, road_class="primary")

    async def _fake_scenario_context_from_od(**_: Any) -> dict[str, Any]:
        return {"bucket": "clear"}

    async def _fake_scenario_candidate_modifiers_async(**_: Any) -> dict[str, Any]:
        return {}

    def _fake_feasibility(**_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "origin_node_id": "a",
            "destination_node_id": "b",
        }

    async def _fake_k_raw_search(**_: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics, dict[str, Any]]:
        return (
            [],
            GraphCandidateDiagnostics(
                explored_states=0,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=8,
                no_path_reason="no_path",
                no_path_detail="simulated_drop_case",
            ),
            {
                "graph_retry_attempted": True,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "skipped_long_corridor_fast_fallback",
                "graph_rescue_attempted": True,
                "graph_rescue_mode": "long_corridor_fast_fallback",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "skipped_long_corridor_fast_fallback",
                "graph_search_ms_initial": 0.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        specs = list(kwargs.get("specs", []))
        assert specs
        alt_spec = specs[0]
        toll_spec = next(spec for spec in specs if "exclude:toll" in spec.label)
        yield main_module.CandidateProgress(
            done=1,
            total=2,
            result=main_module.CandidateFetchResult(spec=alt_spec, routes=[dict(fallback_route)]),
        )
        yield main_module.CandidateProgress(
            done=2,
            total=2,
            result=main_module.CandidateFetchResult(spec=toll_spec, routes=[dict(toll_free_route)]),
        )

    async def _fake_refine_graph_candidate_batch(**kwargs: Any):
        selected_records = list(kwargs.get("selected_records", []))
        raw_graph_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed_costs = {
            str(record.candidate_id): 11.0
            for record in selected_records
        }
        refined = [
            dict(raw_graph_routes_by_id.get(str(record.candidate_id), fallback_route))
            for record in selected_records
        ]
        return refined, [], observed_costs, len(selected_records), 20.0

    def _fake_build_options(routes: list[dict[str, Any]], **_: Any):
        first_route = routes[0]
        first_route["_built_option_id"] = "route_0"
        coords = [
            (float(point[0]), float(point[1]))
            for point in first_route["geometry"]["coordinates"]
        ]
        options = [
            RouteOption(
                id="route_0",
                geometry=GeoJSONLineString(type="LineString", coordinates=coords),
                metrics=RouteMetrics(
                    distance_km=float(first_route["distance"]) / 1000.0,
                    duration_s=float(first_route["duration"]),
                    monetary_cost=310.0,
                    emissions_kg=205.0,
                    avg_speed_kmh=70.0,
                ),
            )
        ]
        return options, ["route_1: terrain_fail_closed (simulated)"], main_module.TerrainDiagnostics(
            fail_closed_count=1,
            dem_version="test-dem",
        )

    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _fake_scenario_context_from_od)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _fake_scenario_candidate_modifiers_async)
    monkeypatch.setattr(main_module, "route_graph_od_feasibility", _fake_feasibility)
    monkeypatch.setattr(main_module, "_route_graph_k_raw_search", _fake_k_raw_search)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fake_iter_candidate_fetches)
    monkeypatch.setattr(main_module, "_refine_graph_candidate_batch", _fake_refine_graph_candidate_batch)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: list(options)[0])
    monkeypatch.setattr(
        main_module,
        "_route_selection_score_map",
        lambda options, **_: {str(option.id): float(index) for index, option in enumerate(options, start=1)},
    )

    req = RouteRequest(
        origin=LatLng(lat=51.5074, lon=-0.1278),
        destination=LatLng(lat=53.4808, lon=-2.2426),
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        max_alternatives=4,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=4,
            pipeline_mode="dccs",
            run_seed=20260321,
        )
    )

    assert result["selected"].id == "route_0"
    assert any("option_build: dropped_refined_routes" in warning for warning in result["warnings"])
    refined_rows = result["extra_jsonl_artifacts"]["refined_routes.jsonl"]
    assert len(refined_rows) == 1


def test_compute_direct_route_pipeline_applies_supplemental_diversity_rescue_on_collapse(
    monkeypatch,
) -> None:
    raw_a = _make_ranked_route(duration_s=4_600.0, lon_seed=0.00, road_class="motorway")
    raw_b = _make_ranked_route(duration_s=4_950.0, lon_seed=0.25, road_class="primary")
    raw_b["_graph_meta"]["toll_edges"] = 2
    raw_c = _make_ranked_route(duration_s=5_200.0, lon_seed=0.55, road_class="secondary")
    collapsed_refined = dict(raw_a)
    rescue_route = _make_ranked_route(duration_s=4_050.0, lon_seed=0.85, road_class="trunk")

    async def _fake_scenario_context_from_od(**_: Any) -> dict[str, Any]:
        return {"bucket": "clear"}

    async def _fake_scenario_candidate_modifiers_async(**_: Any) -> dict[str, Any]:
        return {}

    def _fake_feasibility(**_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "origin_node_id": "a",
            "destination_node_id": "b",
        }

    async def _fake_k_raw_search(**_: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics, dict[str, Any]]:
        return (
            [dict(raw_a), dict(raw_b), dict(raw_c)],
            GraphCandidateDiagnostics(
                explored_states=21,
                generated_paths=5,
                emitted_paths=3,
                candidate_budget=5,
            ),
            {
                "graph_retry_attempted": False,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "not_applicable",
                "graph_rescue_attempted": False,
                "graph_rescue_mode": "not_applicable",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "not_applicable",
                "graph_search_ms_initial": 12.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _fake_refine_graph_candidate_batch(**kwargs: Any):
        selected_records = list(kwargs.get("selected_records", []))
        observed = {str(record.candidate_id): 7.0 for record in selected_records}
        # Collapse all initially refined candidates onto the same realized family.
        return [dict(collapsed_refined)], [], observed, len(selected_records), 14.0

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        spec = list(kwargs.get("specs", []))[0]
        yield main_module.CandidateProgress(
            done=1,
            total=1,
            result=main_module.CandidateFetchResult(spec=spec, routes=[dict(rescue_route)]),
        )

    async def _fake_fetch_local_ors_baseline_seed(**_: Any):
        ors_route = _make_ranked_route(duration_s=4_020.0, lon_seed=1.10, road_class="trunk")
        return ors_route, {"provider_mode": "local_service", "engine_profile": "driving-hgv"}

    class _RescueOSRM:
        async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:
            via = kwargs.get("via")
            if via:
                return [_make_ranked_route(duration_s=3_980.0, lon_seed=1.25, road_class="trunk")]
            return []

    def _fake_build_options(routes: list[dict[str, Any]], **_: Any):
        options: list[RouteOption] = []
        for index, route in enumerate(routes):
            route_id = f"route_{index}"
            route["_built_option_id"] = route_id
            coords = [
                (float(point[0]), float(point[1]))
                for point in route["geometry"]["coordinates"]
            ]
            duration_s = float(route["duration"])
            options.append(
                RouteOption(
                    id=route_id,
                    geometry=GeoJSONLineString(type="LineString", coordinates=coords),
                    metrics=RouteMetrics(
                        distance_km=float(route["distance"]) / 1000.0,
                        duration_s=duration_s,
                        monetary_cost=200.0 + (duration_s / 200.0),
                        emissions_kg=120.0 + (duration_s / 500.0),
                        avg_speed_kmh=70.0,
                    ),
                )
            )
        return options, [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _fake_scenario_context_from_od)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _fake_scenario_candidate_modifiers_async)
    monkeypatch.setattr(main_module, "route_graph_od_feasibility", _fake_feasibility)
    monkeypatch.setattr(main_module, "_route_graph_k_raw_search", _fake_k_raw_search)
    monkeypatch.setattr(main_module, "_refine_graph_candidate_batch", _fake_refine_graph_candidate_batch)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _fake_iter_candidate_fetches)
    monkeypatch.setattr(main_module, "_fetch_local_ors_baseline_seed", _fake_fetch_local_ors_baseline_seed)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options)[:1])
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: min(list(options), key=lambda item: item.metrics.duration_s))
    monkeypatch.setattr(
        main_module,
        "_route_selection_score_map",
        lambda options, **_: {str(option.id): float(option.metrics.duration_s) for option in options},
    )

    req = RouteRequest(
        origin=LatLng(lat=51.59, lon=-2.99),
        destination=LatLng(lat=51.45, lon=-2.58),
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        max_alternatives=5,
        search_budget=3,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_RescueOSRM(),
            ors=_NoopORS(),
            max_alternatives=5,
            pipeline_mode="dccs",
            run_seed=20260322,
        )
        )

    candidate_diag = result["candidate_diag"]
    assert candidate_diag.diversity_collapse_detected is True
    assert candidate_diag.supplemental_challenger_activated is True
    assert candidate_diag.supplemental_selected_count == 1
    assert any(
        source in candidate_diag.supplemental_sources_json
        for source in ("osrm", "ors_local", "ors_local_seed")
    )
    dccs_summary = result["extra_json_artifacts"]["dccs_summary.json"]
    assert dccs_summary["supplemental_challenger_activated"] is True
    assert dccs_summary["supplemental_selected_count"] == 1
    dccs_rows = result["extra_jsonl_artifacts"]["dccs_candidates.jsonl"]
    rescue_rows = [row for row in dccs_rows if row.get("supplemental_diversity_rescue")]
    assert rescue_rows
    assert any(
        row.get("candidate_source_engine") in {"osrm", "ors_local_seed"}
        for row in rescue_rows
    )
    assert result["selected"].id == "route_0"
    assert result["extra_json_artifacts"]["dccs_summary.json"]["selected_route_id"] == "route_0"


def test_is_toll_exclusion_label_accepts_prefixed_and_compound_labels() -> None:
    assert main_module._is_toll_exclusion_label("exclude:toll") is True
    assert main_module._is_toll_exclusion_label("exclude:motorway,toll") is True
    assert main_module._is_toll_exclusion_label("fallback:exclude:toll:direct_k_raw_fallback") is True
    assert main_module._is_toll_exclusion_label("fallback:exclude:motorway,toll:osrm_refined") is True
    assert main_module._is_toll_exclusion_label("fallback:exclude:motorway:direct_k_raw_fallback") is False


def test_graph_route_candidate_payload_uses_geometry_tokens_when_graph_nodes_missing() -> None:
    route_a = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    route_b = _make_ranked_route(duration_s=28_000.0, lon_seed=0.25, road_class="motorway")
    vehicle = main_module.resolve_vehicle_profile("rigid_hgv")

    payload_a = main_module._graph_route_candidate_payload(
        route_a,
        origin=LatLng(lat=51.5074, lon=-0.1278),
        destination=LatLng(lat=53.4808, lon=-2.2426),
        vehicle=vehicle,
        cost_toggles=main_module.CostToggles(),
    )
    payload_b = main_module._graph_route_candidate_payload(
        route_b,
        origin=LatLng(lat=51.5074, lon=-0.1278),
        destination=LatLng(lat=53.4808, lon=-2.2426),
        vehicle=vehicle,
        cost_toggles=main_module.CostToggles(),
    )

    assert payload_a["graph_path"]
    assert payload_b["graph_path"]
    assert payload_a["graph_path"] != payload_b["graph_path"]
    assert payload_a["candidate_id"] != payload_b["candidate_id"]


def test_refine_graph_candidate_batch_reuses_pre_realized_fallback_routes() -> None:
    raw_route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    main_module._annotate_route_candidate_meta(
        raw_route,
        source_labels={"fallback:exclude:toll:direct_k_raw_fallback"},
        toll_exclusion_available=False,
    )

    routes, warnings, observed_costs, fetches, elapsed_ms = asyncio.run(
        main_module._refine_graph_candidate_batch(
            osrm=_FailingOSRM(),
            origin=LatLng(lat=51.5074, lon=-0.1278),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            selected_records=[SimpleNamespace(candidate_id="candidate-1")],
            raw_graph_routes_by_id={"candidate-1": raw_route},
        )
    )

    assert warnings == []
    assert fetches == 0
    assert elapsed_ms >= 0.0
    assert observed_costs == {}
    assert len(routes) == 1
    assert routes[0]["_dccs_candidate_ids"] == ["candidate-1"]


def test_annotate_route_candidate_meta_merges_existing_labels() -> None:
    route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    main_module._annotate_route_candidate_meta(
        route,
        source_labels={"fallback:exclude:toll:direct_k_raw_fallback"},
        toll_exclusion_available=False,
    )
    main_module._annotate_route_candidate_meta(
        route,
        source_labels={"fallback:alternatives:direct_k_raw_fallback"},
        toll_exclusion_available=False,
    )
    meta = route["_candidate_meta"]
    assert meta["seen_by_exclude_toll"] is True
    assert sorted(meta["source_labels"]) == [
        "fallback:alternatives:direct_k_raw_fallback",
        "fallback:exclude:toll:direct_k_raw_fallback",
    ]
