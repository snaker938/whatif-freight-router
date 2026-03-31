from __future__ import annotations

import copy

import asyncio
import math
import time
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


def _make_fallback_route(
    *,
    coordinates: list[list[float]],
    segment_distances_m: list[float],
    segment_durations_s: list[float],
    source_label: str,
    observed_refine_cost_ms: float = 19.0,
) -> dict[str, Any]:
    steps = [
        {
            "distance": float(distance_m),
            "duration": float(duration_s),
            "classes": [],
        }
        for distance_m, duration_s in zip(segment_distances_m, segment_durations_s, strict=True)
    ]
    route = {
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "duration": float(sum(segment_durations_s)),
        "distance": float(sum(segment_distances_m)),
        "legs": [
            {
                "annotation": {
                    "distance": list(segment_distances_m),
                    "duration": list(segment_durations_s),
                },
                "steps": steps,
            }
        ],
    }
    main_module._annotate_route_candidate_meta(
        route,
        source_labels={source_label},
        toll_exclusion_available=False,
        observed_refine_cost_ms=observed_refine_cost_ms,
    )
    return route


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


def test_select_ranked_candidate_routes_prefilter_limits_expensive_redundant_fallback_vias(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "route_candidate_alternatives_max", 24)
    monkeypatch.setattr(settings, "route_candidate_prefilter_multiplier", 2)

    routes = [
        _make_ranked_route(duration_s=100.0, lon_seed=0.0, road_class="motorway"),
        _make_fallback_route(
            coordinates=[[-1.9000, 52.5000], [-1.8500, 52.5200], [-1.7800, 52.5600]],
            segment_distances_m=[12_000.0, 20_000.0],
            segment_durations_s=[40.0, 61.0],
            source_label="fallback:alternatives:direct_k_raw_fallback",
            observed_refine_cost_ms=24.0,
        ),
        _make_fallback_route(
            coordinates=[[-1.9050, 52.5000], [-1.8300, 52.5150], [-1.7300, 52.5550]],
            segment_distances_m=[11_500.0, 20_500.0],
            segment_durations_s=[39.0, 63.0],
            source_label="fallback:via:1:direct_k_raw_fallback",
            observed_refine_cost_ms=24.0,
        ),
        _make_fallback_route(
            coordinates=[[-1.9100, 52.5000], [-1.7900, 52.5050], [-1.6800, 52.5450]],
            segment_distances_m=[11_000.0, 21_000.0],
            segment_durations_s=[39.0, 64.0],
            source_label="fallback:via:2:direct_k_raw_fallback",
            observed_refine_cost_ms=220.0,
        ),
        _make_fallback_route(
            coordinates=[[-1.9150, 52.5000], [-1.8350, 52.5300], [-1.7600, 52.5750]],
            segment_distances_m=[10_500.0, 21_500.0],
            segment_durations_s=[40.0, 64.0],
            source_label="fallback:exclude:motorway:direct_k_raw_fallback",
            observed_refine_cost_ms=22.0,
        ),
        _make_fallback_route(
            coordinates=[[-1.9200, 52.5000], [-1.7650, 52.5100], [-1.6500, 52.5400]],
            segment_distances_m=[10_000.0, 22_000.0],
            segment_durations_s=[40.0, 65.0],
            source_label="fallback:via:3:direct_k_raw_fallback",
            observed_refine_cost_ms=240.0,
        ),
    ]

    selected = main_module._select_ranked_candidate_routes(routes, max_routes=2)
    selected_labels = [
        main_module._selected_route_source_label(main_module._candidate_source_labels(route))
        for route in selected
    ]

    assert len(selected) == 4
    assert "fallback:alternatives:direct_k_raw_fallback" in selected_labels
    assert "fallback:exclude:motorway:direct_k_raw_fallback" in selected_labels
    assert "fallback:via:1:direct_k_raw_fallback" in selected_labels
    assert "fallback:via:2:direct_k_raw_fallback" not in selected_labels
    assert sum(1 for label in selected_labels if label and ":via:" in label) == 1


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


def test_route_graph_k_raw_search_keeps_graph_search_for_supported_stressed_long_corridor(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_once(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [_make_graph_route(seed=0.0), _make_graph_route(seed=0.15)],
            GraphCandidateDiagnostics(
                explored_states=3200,
                generated_paths=2,
                emitted_paths=2,
                candidate_budget=8,
            ),
        )

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_skip_initial_search_long_corridor", True)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_once)

    routes, diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=51.5074, lon=-0.1278),
            destination=LatLng(lat=52.4862, lon=-1.8904),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.05},
            start_node_id="a",
            goal_node_id="b",
            od_ambiguity_index=0.42,
            od_engine_disagreement_prior=0.44,
            od_hard_case_prior=0.48,
            od_ambiguity_support_ratio=0.66,
            od_ambiguity_source_entropy=0.99,
            od_candidate_path_count=14,
            od_corridor_family_count=11,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert routes
    assert len(calls) == 1
    assert diag.no_path_reason == ""
    assert meta.get("graph_long_corridor_stress_probe") is True
    assert meta.get("graph_retry_attempted") is False
    assert meta.get("graph_search_ms_initial", 0.0) >= 0.0


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


def test_route_graph_k_raw_search_support_rich_short_haul_voi_probe_keeps_graph_search(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_once(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        return (
            [_make_graph_route(seed=0.19)],
            GraphCandidateDiagnostics(
                explored_states=2400,
                generated_paths=3,
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
            od_candidate_path_count=3,
            od_corridor_family_count=2,
            allow_supported_ambiguity_fast_fallback=True,
            enable_voi_support_rich_short_haul_graph_probe=True,
        )
    )

    assert routes
    assert len(calls) == 1
    assert diag.no_path_reason == ""
    assert meta.get("graph_support_rich_short_haul_fast_fallback") is True
    assert meta.get("graph_support_rich_short_haul_probe") is True
    assert float(meta.get("graph_search_ms_initial", 0.0)) >= 0.0


def test_route_graph_k_raw_search_support_rich_short_haul_voi_probe_requires_high_stress(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_should_not_run(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        raise AssertionError("high-stress VOI probe should stay disabled on moderate support-rich rows")

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_should_not_run)

    routes, diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=51.5007, lon=-3.2007),
            destination=LatLng(lat=51.4816, lon=-2.5892),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.05},
            start_node_id="a",
            goal_node_id="b",
            od_ambiguity_index=0.34,
            od_engine_disagreement_prior=0.38,
            od_hard_case_prior=0.39,
            od_ambiguity_support_ratio=0.83,
            od_ambiguity_source_entropy=0.71,
            od_candidate_path_count=3,
            od_corridor_family_count=2,
            allow_supported_ambiguity_fast_fallback=True,
            enable_voi_support_rich_short_haul_graph_probe=True,
        )
    )

    assert routes == []
    assert calls == []
    assert diag.no_path_reason == "skipped_support_rich_short_haul_graph_search"
    assert meta.get("graph_support_rich_short_haul_fast_fallback") is True
    assert meta.get("graph_support_rich_short_haul_probe") is False


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


def test_support_rich_short_haul_graph_probe_eligible_requires_high_stress() -> None:
    assert (
        main_module._support_rich_short_haul_graph_probe_eligible(
            enabled=True,
            od_engine_disagreement_prior=0.38,
            od_hard_case_prior=0.39,
        )
        is False
    )
    assert (
        main_module._support_rich_short_haul_graph_probe_eligible(
            enabled=True,
            od_engine_disagreement_prior=0.56,
            od_hard_case_prior=0.39,
        )
        is True
    )


def test_support_rich_short_haul_probe_fail_open_needed_for_underfilled_routes() -> None:
    assert (
        main_module._support_rich_short_haul_probe_fail_open_needed(
            enabled=True,
            routes=[_make_graph_route(seed=0.19)],
            od_candidate_path_count=3,
            od_corridor_family_count=2,
        )
        is True
    )
    assert (
        main_module._support_rich_short_haul_probe_fail_open_needed(
            enabled=True,
            routes=[
                _make_graph_route(seed=0.11),
                _make_graph_route(seed=0.21),
                _make_graph_route(seed=0.31),
                _make_graph_route(seed=0.41),
                _make_graph_route(seed=0.51),
            ],
            od_candidate_path_count=3,
            od_corridor_family_count=1,
        )
        is False
    )


def test_support_backed_single_corridor_probe_fail_open_needed_for_underfilled_routes() -> None:
    assert (
        main_module._support_backed_single_corridor_probe_fail_open_needed(
            enabled=True,
            routes=[_make_graph_route(seed=0.19)],
            od_candidate_path_count=3,
        )
        is True
    )
    assert (
        main_module._support_backed_single_corridor_probe_fail_open_needed(
            enabled=True,
            routes=[
                _make_graph_route(seed=0.11),
                _make_graph_route(seed=0.21),
                _make_graph_route(seed=0.31),
            ],
            od_candidate_path_count=3,
        )
        is False
    )


def test_long_corridor_stress_graph_probe_eligible_requires_real_stress_and_support() -> None:
    assert (
        main_module._long_corridor_stress_graph_probe_eligible(
            long_corridor=True,
            ambiguity_strength=0.42,
            od_engine_disagreement_prior=0.44,
            od_hard_case_prior=0.48,
            od_ambiguity_support_ratio=0.66,
            od_ambiguity_source_entropy=0.99,
            od_candidate_path_count=14,
            od_corridor_family_count=11,
        )
        is True
    )
    assert (
        main_module._long_corridor_stress_graph_probe_eligible(
            long_corridor=True,
            ambiguity_strength=0.18,
            od_engine_disagreement_prior=0.20,
            od_hard_case_prior=0.20,
            od_ambiguity_support_ratio=0.50,
            od_ambiguity_source_entropy=0.99,
            od_candidate_path_count=14,
            od_corridor_family_count=11,
        )
        is False
    )
    assert (
        main_module._long_corridor_stress_graph_probe_eligible(
            long_corridor=True,
            ambiguity_strength=0.03,
            od_engine_disagreement_prior=0.44,
            od_hard_case_prior=0.48,
            od_ambiguity_support_ratio=0.50,
            od_ambiguity_source_entropy=0.99,
            od_candidate_path_count=14,
            od_corridor_family_count=11,
        )
        is False
    )
    assert (
        main_module._long_corridor_stress_graph_probe_eligible(
            long_corridor=True,
            ambiguity_strength=0.32,
            od_engine_disagreement_prior=0.37,
            od_hard_case_prior=0.36,
            od_ambiguity_support_ratio=0.56,
            od_ambiguity_source_entropy=0.68,
            od_candidate_path_count=14,
            od_corridor_family_count=1,
        )
        is False
    )
    assert (
        main_module._long_corridor_stress_graph_probe_eligible(
            long_corridor=True,
            ambiguity_strength=0.18,
            od_engine_disagreement_prior=0.44,
            od_hard_case_prior=0.48,
            od_ambiguity_support_ratio=0.22,
            od_ambiguity_source_entropy=0.18,
            od_candidate_path_count=2,
            od_corridor_family_count=1,
        )
        is False
    )


def test_route_graph_k_raw_search_prior_only_long_corridor_pressure_stays_on_fast_fallback(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _graph_routes_exhausted(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        detail = (
            "deadline reached"
            if len(calls) == 1
            else ("deadline reached after retry" if len(calls) == 2 else "deadline reached after rescue")
        )
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

    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    monkeypatch.setattr(settings, "route_graph_skip_initial_search_long_corridor", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_enabled", True)
    monkeypatch.setattr(settings, "route_graph_state_space_rescue_mode", "reduced")
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_multiplier", 2.0)
    monkeypatch.setattr(settings, "route_graph_state_budget_retry_cap", 2500000)
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_routes_exhausted)

    routes, diag, meta = asyncio.run(
        main_module._route_graph_k_raw_search(
            origin=LatLng(lat=51.5074, lon=-0.1278),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            max_alternatives=6,
            scenario_edge_modifiers={"duration_multiplier": 1.1},
            start_node_id="a",
            goal_node_id="b",
            od_ambiguity_index=0.18,
            od_engine_disagreement_prior=0.44,
            od_hard_case_prior=0.48,
            od_ambiguity_support_ratio=0.50,
            od_ambiguity_source_entropy=0.99,
            od_candidate_path_count=14,
            od_corridor_family_count=11,
            allow_supported_ambiguity_fast_fallback=True,
        )
    )

    assert routes == []
    assert len(calls) == 0
    assert diag.no_path_reason == "skipped_long_corridor_graph_search"
    assert meta.get("graph_long_corridor_stress_probe") is False
    assert meta.get("graph_retry_attempted") is True
    assert meta.get("graph_retry_outcome") == "skipped_long_corridor_fast_fallback"
    assert meta.get("graph_rescue_attempted") is False
    assert meta.get("graph_rescue_mode") == "not_applicable"
    assert meta.get("graph_rescue_outcome") == "not_applicable"


@pytest.mark.parametrize(
    (
        "od_nominal_margin_proxy",
        "od_objective_spread",
        "od_ambiguity_support_ratio",
        "od_ambiguity_source_entropy",
        "od_ambiguity_index",
        "od_engine_disagreement_prior",
        "od_hard_case_prior",
        "expected_skip",
        "expected_calls",
    ),
    [
        (0.97, 0.18, 0.64, 0.78, 0.08, 0.12, 0.18, True, 0),
        (0.97, 0.18, 0.58, 0.41, 0.08, 0.12, 0.18, False, 1),
        (0.83, 0.34, 0.72, 0.86, 0.08, 0.12, 0.18, False, 1),
        (0.97, 0.18, 0.72, 0.86, 0.44, 0.36, 0.34, False, 1),
    ],
)
def test_route_graph_k_raw_search_support_backed_single_corridor_fast_path_is_cache_safe_and_narrow(
    monkeypatch,
    od_nominal_margin_proxy: float,
    od_objective_spread: float,
    od_ambiguity_support_ratio: float,
    od_ambiguity_source_entropy: float,
    od_ambiguity_index: float,
    od_engine_disagreement_prior: float,
    od_hard_case_prior: float,
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
            od_ambiguity_index=od_ambiguity_index,
            od_engine_disagreement_prior=od_engine_disagreement_prior,
            od_hard_case_prior=od_hard_case_prior,
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


def test_collect_candidate_routes_prior_only_long_corridor_pressure_uses_fast_fallback_then_osrm(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []
    refine_calls: list[dict[str, Any]] = []

    async def _ok_status() -> tuple[bool, str]:
        return True, "ok"

    async def _graph_routes_exhausted(**kwargs: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        calls.append(dict(kwargs))
        detail = (
            "deadline reached"
            if len(calls) == 1
            else ("deadline reached after retry" if len(calls) == 2 else "deadline reached after rescue")
        )
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
    monkeypatch.setattr(settings, "route_graph_skip_initial_search_long_corridor", True)
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
            origin=LatLng(lat=51.5074, lon=-0.1278),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            max_routes=6,
            cache_key=None,
            scenario_edge_modifiers={"duration_multiplier": 1.1},
            od_ambiguity_index=0.18,
            od_engine_disagreement_prior=0.44,
            od_hard_case_prior=0.48,
            od_ambiguity_support_ratio=0.50,
            od_ambiguity_source_entropy=0.99,
            od_candidate_path_count=14,
            od_corridor_family_count=11,
            allow_supported_ambiguity_fast_fallback=True,
            progress_cb=None,
        )
    )

    assert routes
    assert spec_count >= 1
    assert len(calls) == 0
    assert refine_calls
    assert any("routing_graph_long_corridor_osrm_fallback" in warning for warning in warnings)
    assert diag.graph_no_path_reason == "skipped_long_corridor_graph_search"
    assert diag.graph_retry_attempted is True
    assert diag.graph_retry_outcome == "skipped_long_corridor_fast_fallback"
    assert diag.graph_rescue_attempted is False
    assert diag.graph_rescue_mode == "not_applicable"
    assert diag.graph_rescue_outcome == "not_applicable"


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
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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


def test_compute_direct_route_pipeline_support_backed_single_corridor_probe_fail_open_merges_support_fallback(
    monkeypatch,
) -> None:
    graph_route = _make_ranked_route(duration_s=27_900.0, lon_seed=0.0, road_class="motorway")
    fallback_route = _make_ranked_route(duration_s=28_150.0, lon_seed=0.16, road_class="trunk")
    fetched_labels: list[str] = []

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
            [dict(graph_route)],
            GraphCandidateDiagnostics(
                explored_states=1200,
                generated_paths=2,
                emitted_paths=1,
                candidate_budget=6,
            ),
            {
                "graph_retry_attempted": False,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "not_applicable",
                "graph_rescue_attempted": False,
                "graph_rescue_mode": "not_applicable",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "not_applicable",
                "graph_supported_ambiguity_fast_fallback": False,
                "graph_support_rich_short_haul_fast_fallback": False,
                "graph_support_rich_short_haul_probe": False,
                "graph_support_backed_single_corridor_diversity_probe": True,
                "graph_support_backed_single_corridor_fast_fallback": False,
                "graph_search_ms_initial": 12.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        specs = list(kwargs.get("specs", []))
        assert specs
        fetched_labels.extend(str(spec.label) for spec in specs)
        yield main_module.CandidateProgress(
            done=1,
            total=len(specs),
            result=main_module.CandidateFetchResult(spec=specs[0], routes=[dict(fallback_route)]),
        )

    async def _fake_refine_graph_candidate_batch(**kwargs: Any):
        selected_records = list(kwargs.get("selected_records", []))
        raw_graph_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed_costs = {str(record.candidate_id): 11.0 for record in selected_records}
        refined = [
            dict(raw_graph_routes_by_id.get(str(record.candidate_id), fallback_route))
            for record in selected_records
        ]
        return refined, [], observed_costs, len(selected_records), 17.0

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
                        monetary_cost=210.0,
                        emissions_kg=130.0,
                        avg_speed_kmh=66.0,
                    ),
                )
            )
        return options, [], main_module.TerrainDiagnostics()

    monkeypatch.setattr(settings, "route_graph_direct_k_raw_fallback_include_ors_seed", False)
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
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
    monkeypatch.setattr(
        main_module,
        "_route_selection_score_map",
        lambda options, **_: {str(option.id): float(index) for index, option in enumerate(options, start=1)},
    )

    req = RouteRequest(
        origin=LatLng(lat=52.4862, lon=-1.8904),
        destination=LatLng(lat=51.4545, lon=-2.5879),
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        max_alternatives=4,
        od_ambiguity_index=0.26,
        od_engine_disagreement_prior=0.36,
        od_hard_case_prior=0.34,
        od_ambiguity_support_ratio=0.78,
        od_ambiguity_source_entropy=0.78,
        od_candidate_path_count=3,
        od_corridor_family_count=1,
        od_nominal_margin_proxy=0.97,
        od_objective_spread=0.18,
        allow_supported_ambiguity_fast_fallback=True,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=4,
            pipeline_mode="dccs",
            run_seed=20260329,
        )
    )

    assert result["candidate_diag"].raw_count == 2
    assert fetched_labels
    assert all(label.startswith("support_fallback:") for label in fetched_labels)
    warnings = " ".join(result["warnings"])
    assert "routing_graph_support_backed_single_corridor_probe_fail_open" in warnings
    assert "routing_graph_support_aware_k_raw_fallback" in warnings
    assert "routing_graph_long_corridor_k_raw_fallback" not in warnings


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
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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


def test_compute_direct_route_pipeline_uses_osrm_only_rescue_for_multi_family_single_frontier_collapse(
    monkeypatch,
) -> None:
    raw_a = _make_ranked_route(duration_s=4_600.0, lon_seed=0.00, road_class="motorway")
    raw_b = _make_ranked_route(duration_s=4_760.0, lon_seed=0.25, road_class="primary")
    raw_c = _make_ranked_route(duration_s=4_890.0, lon_seed=0.55, road_class="secondary")
    raw_d = _make_ranked_route(duration_s=5_010.0, lon_seed=0.90, road_class="trunk")
    raw_e = _make_ranked_route(duration_s=5_160.0, lon_seed=1.15, road_class="motorway")
    raw_f = _make_ranked_route(duration_s=5_280.0, lon_seed=1.40, road_class="primary")
    rescue_route = _make_ranked_route(duration_s=4_020.0, lon_seed=1.20, road_class="trunk")

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
            [dict(raw_a), dict(raw_b), dict(raw_c), dict(raw_d), dict(raw_e), dict(raw_f)],
            GraphCandidateDiagnostics(
                explored_states=21,
                generated_paths=8,
                emitted_paths=6,
                candidate_budget=8,
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
        raw_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed = {str(record.candidate_id): 7.0 for record in selected_records}
        refined_routes = [dict(raw_routes_by_id[str(record.candidate_id)]) for record in selected_records]
        return refined_routes, [], observed, len(selected_records), 14.0

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        spec = list(kwargs.get("specs", []))[0]
        yield main_module.CandidateProgress(
            done=1,
            total=1,
            result=main_module.CandidateFetchResult(spec=spec, routes=[dict(rescue_route)]),
        )

    async def _unexpected_fetch_local_ors_baseline_seed(**_: Any):
        raise AssertionError("local ORS rescue should be skipped for multi-family single-frontier collapse")

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
    monkeypatch.setattr(main_module, "_fetch_local_ors_baseline_seed", _unexpected_fetch_local_ors_baseline_seed)
    monkeypatch.setattr(main_module, "_graph_family_via_points", lambda route, **_: [(51.5, -2.8)])
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options)[:1])
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: min(list(options), key=lambda item: item.metrics.duration_s))
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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
        search_budget=4,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=5,
            pipeline_mode="dccs",
            run_seed=20260329,
        )
    )

    candidate_diag = result["candidate_diag"]
    assert candidate_diag.diversity_collapse_detected is True
    assert candidate_diag.diversity_collapse_reason == "single_frontier_after_multi_family_refine"
    assert candidate_diag.supplemental_challenger_activated is True
    assert candidate_diag.supplemental_selected_count == 1
    assert candidate_diag.supplemental_sources_json == '["osrm"]'
    dccs_summary = result["extra_json_artifacts"]["dccs_summary.json"]
    assert dccs_summary["diversity_collapse_reason"] == "single_frontier_after_multi_family_refine"
    assert dccs_summary["supplemental_challenger_activated"] is True
    rescue_rows = [
        row
        for row in result["extra_jsonl_artifacts"]["dccs_candidates.jsonl"]
        if row.get("supplemental_diversity_rescue")
    ]
    assert rescue_rows
    assert all(row.get("candidate_source_engine") == "osrm" for row in rescue_rows)


def test_compute_direct_route_pipeline_spends_leftover_budget_on_challenger_without_collapse(
    monkeypatch,
) -> None:
    raw_a = _make_ranked_route(duration_s=4_600.0, lon_seed=0.00, road_class="motorway")
    raw_b = _make_ranked_route(duration_s=4_750.0, lon_seed=0.35, road_class="primary")
    raw_c = _make_ranked_route(duration_s=4_050.0, lon_seed=0.80, road_class="trunk")

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
        raw_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed = {str(record.candidate_id): 7.0 for record in selected_records}
        refined_routes = [dict(raw_routes_by_id[str(record.candidate_id)]) for record in selected_records]
        return refined_routes, [], observed, len(selected_records), 14.0

    async def _unexpected_iter_candidate_fetches(**_: Any):
        raise AssertionError("supplemental diversity rescue should not run when leftover challenger fill is sufficient")
        yield  # pragma: no cover

    async def _unexpected_fetch_local_ors_baseline_seed(**_: Any):
        raise AssertionError("local ORS should not be used when diversity rescue is not triggered")

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
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _unexpected_iter_candidate_fetches)
    monkeypatch.setattr(main_module, "_fetch_local_ors_baseline_seed", _unexpected_fetch_local_ors_baseline_seed)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options)[:1])
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: min(list(options), key=lambda item: item.metrics.duration_s))
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=5,
            pipeline_mode="dccs",
            run_seed=20260329,
        )
    )

    candidate_diag = result["candidate_diag"]
    assert candidate_diag.diversity_collapse_detected is False
    assert candidate_diag.leftover_challenger_activated is True
    assert candidate_diag.leftover_challenger_selected_count == 1
    assert candidate_diag.supplemental_challenger_activated is False
    assert candidate_diag.selected_candidate_count == 3
    dccs_summary = result["extra_json_artifacts"]["dccs_summary.json"]
    assert dccs_summary["search_budget_used"] == 3
    assert dccs_summary["leftover_challenger_activated"] is True
    assert dccs_summary["leftover_challenger_selected_count"] == 1
    assert dccs_summary["supplemental_challenger_activated"] is False
    assert any(
        bool(batch.get("leftover_budget_challenger"))
        for batch in dccs_summary["batches"]
        if isinstance(batch, dict)
    )
    dccs_rows = result["extra_jsonl_artifacts"]["dccs_candidates.jsonl"]
    assert any(bool(row.get("leftover_budget_challenger")) for row in dccs_rows)
    assert not any(bool(row.get("supplemental_diversity_rescue")) for row in dccs_rows)


def test_compute_direct_route_pipeline_uses_leftover_challenger_on_multi_family_refine_collapse_when_supplemental_empty(
    monkeypatch,
) -> None:
    raw_a = _make_ranked_route(duration_s=4_700.0, lon_seed=0.00, road_class="motorway")
    raw_b = _make_ranked_route(duration_s=4_850.0, lon_seed=0.35, road_class="primary")
    raw_c = _make_ranked_route(duration_s=4_300.0, lon_seed=0.80, road_class="trunk")
    raw_d = _make_ranked_route(duration_s=4_450.0, lon_seed=1.15, road_class="secondary")

    async def _fake_scenario_context_from_od(**_: Any) -> dict[str, Any]:
        return {"bucket": "clear"}

    async def _fake_scenario_candidate_modifiers_async(**_: Any) -> dict[str, float]:
        return {}

    def _fake_feasibility(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "message": "ok",
            "origin_node_id": "100",
            "destination_node_id": "200",
            "origin_nearest_m": 5.0,
            "destination_nearest_m": 7.0,
            "origin_selected_m": 5.0,
            "destination_selected_m": 7.0,
            "origin_candidate_count": 4,
            "destination_candidate_count": 4,
            "selected_component": 1,
            "selected_component_size": 1000,
            "elapsed_ms": 1.0,
        }

    async def _fake_k_raw_search(**_: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics, dict[str, Any]]:
        return (
            [raw_a, raw_b, raw_c, raw_d],
            GraphCandidateDiagnostics(
                explored_states=18,
                generated_paths=5,
                emitted_paths=4,
                candidate_budget=6,
            ),
            {
                "graph_retry_attempted": False,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "not_applicable",
                "graph_rescue_attempted": False,
                "graph_rescue_mode": "not_applicable",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "not_applicable",
                "graph_search_ms_initial": 10.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _fake_refine_graph_candidate_batch(
        *,
        selected_records,
        raw_graph_routes_by_id,
        **_: Any,
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, float], int, float]:
        refined_routes: list[dict[str, Any]] = []
        observed_costs: dict[str, float] = {}
        for record in selected_records:
            route = copy.deepcopy(raw_graph_routes_by_id[record.candidate_id])
            route["distance"] = float(route["distance"]) * 1.01
            route["duration"] = float(route["duration"]) * 1.01
            route["geometry"]["coordinates"][1][0] += 0.01 * len(refined_routes)
            refined_routes.append(route)
            observed_costs[record.candidate_id] = 6.0 + len(refined_routes)
        return refined_routes, [], observed_costs, len(refined_routes), 12.0

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        spec = list(kwargs.get("specs", []))[0]
        yield main_module.CandidateProgress(
            done=1,
            total=1,
            result=main_module.CandidateFetchResult(spec=spec, routes=[]),
        )

    async def _unexpected_fetch_local_ors_baseline_seed(**_: Any):
        raise AssertionError("local ORS rescue should not be used in this path")

    def _fake_build_options(
        routes: list[dict[str, Any]],
        *,
        option_prefix: str,
        **_: Any,
    ) -> tuple[list[main_module.RouteOption], list[str], main_module.TerrainDiagnostics]:
        options: list[main_module.RouteOption] = []
        for idx, route in enumerate(routes):
            duration_s = float(route["duration"])
            route_id = f"{option_prefix}_{idx}"
            route["_built_option_id"] = route_id
            coords = [
                (float(point[0]), float(point[1]))
                for point in route["geometry"]["coordinates"]
            ]
            options.append(
                main_module.RouteOption(
                    id=route_id,
                    geometry=GeoJSONLineString(type="LineString", coordinates=coords),
                    metrics=main_module.RouteMetrics(
                        distance_km=float(route["distance"]) / 1000.0,
                        duration_s=duration_s,
                        monetary_cost=210.0 + (duration_s / 200.0),
                        emissions_kg=125.0 + (duration_s / 500.0),
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
    monkeypatch.setattr(main_module, "_fetch_local_ors_baseline_seed", _unexpected_fetch_local_ors_baseline_seed)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options)[:1])
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: min(list(options), key=lambda item: item.metrics.duration_s))
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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
        search_budget=4,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=5,
            pipeline_mode="dccs",
            run_seed=20260330,
        )
    )

    candidate_diag = result["candidate_diag"]
    assert candidate_diag.leftover_challenger_activated is True
    assert candidate_diag.leftover_challenger_selected_count == 1
    assert candidate_diag.supplemental_challenger_activated is False
    assert candidate_diag.selected_candidate_count >= 3
    dccs_summary = result["extra_json_artifacts"]["dccs_summary.json"]
    assert dccs_summary["leftover_challenger_activated"] is True
    assert dccs_summary["leftover_challenger_selected_count"] == 1
    assert dccs_summary["supplemental_challenger_activated"] is False
    dccs_rows = result["extra_jsonl_artifacts"]["dccs_candidates.jsonl"]
    assert any(bool(row.get("leftover_budget_challenger")) for row in dccs_rows)
    assert not any(bool(row.get("supplemental_diversity_rescue")) for row in dccs_rows)


def test_compute_direct_route_pipeline_reserves_two_challenger_slots_for_family_rich_budget_four_rows(
    monkeypatch,
) -> None:
    raw_a = _make_ranked_route(duration_s=4_620.0, lon_seed=0.00, road_class="motorway")
    raw_b = _make_ranked_route(duration_s=4_760.0, lon_seed=0.22, road_class="primary")
    raw_c = _make_ranked_route(duration_s=4_910.0, lon_seed=0.45, road_class="secondary")
    raw_d = _make_ranked_route(duration_s=4_140.0, lon_seed=0.70, road_class="trunk")
    raw_e = _make_ranked_route(duration_s=4_280.0, lon_seed=0.98, road_class="motorway")
    raw_f = _make_ranked_route(duration_s=4_990.0, lon_seed=1.24, road_class="primary")

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
            [dict(raw_a), dict(raw_b), dict(raw_c), dict(raw_d), dict(raw_e), dict(raw_f)],
            GraphCandidateDiagnostics(
                explored_states=28,
                generated_paths=8,
                emitted_paths=6,
                candidate_budget=8,
            ),
            {
                "graph_retry_attempted": False,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "not_applicable",
                "graph_rescue_attempted": False,
                "graph_rescue_mode": "not_applicable",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "not_applicable",
                "graph_search_ms_initial": 11.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _fake_refine_graph_candidate_batch(**kwargs: Any):
        selected_records = list(kwargs.get("selected_records", []))
        raw_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed = {str(record.candidate_id): 8.0 for record in selected_records}
        refined_routes = [dict(raw_routes_by_id[str(record.candidate_id)]) for record in selected_records]
        return refined_routes, [], observed, len(selected_records), 15.0

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        spec = list(kwargs.get("specs", []))[0]
        yield main_module.CandidateProgress(
            done=1,
            total=1,
            result=main_module.CandidateFetchResult(spec=spec, routes=[]),
        )

    async def _unexpected_fetch_local_ors_baseline_seed(**_: Any):
        raise AssertionError("local ORS rescue should not be used in this path")

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
                        monetary_cost=205.0 + (duration_s / 200.0),
                        emissions_kg=118.0 + (duration_s / 500.0),
                        avg_speed_kmh=69.0,
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
    monkeypatch.setattr(main_module, "_fetch_local_ors_baseline_seed", _unexpected_fetch_local_ors_baseline_seed)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options)[:1])
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: min(list(options), key=lambda item: item.metrics.duration_s))
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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
        search_budget=4,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=5,
            pipeline_mode="dccs",
            run_seed=20260330,
        )
    )

    candidate_diag = result["candidate_diag"]
    assert candidate_diag.leftover_challenger_activated is True
    assert candidate_diag.leftover_challenger_selected_count == 2
    dccs_summary = result["extra_json_artifacts"]["dccs_summary.json"]
    assert dccs_summary["search_budget_used"] == 4
    assert dccs_summary["leftover_challenger_activated"] is True
    assert dccs_summary["leftover_challenger_selected_count"] == 2
    assert dccs_summary["batches"][0]["selected_count"] == 2
    assert dccs_summary["batches"][1]["selected_count"] == 2


def test_compute_direct_route_pipeline_dccs_refc_spends_full_initial_budget_without_reserved_rescue_slots(
    monkeypatch,
) -> None:
    raw_a = _make_ranked_route(duration_s=4_620.0, lon_seed=0.00, road_class="motorway")
    raw_b = _make_ranked_route(duration_s=4_760.0, lon_seed=0.22, road_class="primary")
    raw_c = _make_ranked_route(duration_s=4_910.0, lon_seed=0.45, road_class="secondary")
    raw_d = _make_ranked_route(duration_s=4_140.0, lon_seed=0.70, road_class="trunk")
    raw_e = _make_ranked_route(duration_s=4_280.0, lon_seed=0.98, road_class="motorway")
    raw_f = _make_ranked_route(duration_s=4_990.0, lon_seed=1.24, road_class="primary")

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
            [dict(raw_a), dict(raw_b), dict(raw_c), dict(raw_d), dict(raw_e), dict(raw_f)],
            GraphCandidateDiagnostics(
                explored_states=28,
                generated_paths=8,
                emitted_paths=6,
                candidate_budget=8,
            ),
            {
                "graph_retry_attempted": False,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "not_applicable",
                "graph_rescue_attempted": False,
                "graph_rescue_mode": "not_applicable",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "not_applicable",
                "graph_search_ms_initial": 11.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _fake_refine_graph_candidate_batch(**kwargs: Any):
        selected_records = list(kwargs.get("selected_records", []))
        raw_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed = {str(record.candidate_id): 8.0 for record in selected_records}
        refined_routes = [dict(raw_routes_by_id[str(record.candidate_id)]) for record in selected_records]
        return refined_routes, [], observed, len(selected_records), 15.0

    async def _fake_iter_candidate_fetches(**kwargs: Any):
        spec = list(kwargs.get("specs", []))[0]
        yield main_module.CandidateProgress(
            done=1,
            total=1,
            result=main_module.CandidateFetchResult(spec=spec, routes=[]),
        )

    async def _unexpected_fetch_local_ors_baseline_seed(**_: Any):
        raise AssertionError("local ORS rescue should not be used in this path")

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
                        monetary_cost=205.0 + (duration_s / 200.0),
                        emissions_kg=118.0 + (duration_s / 500.0),
                        avg_speed_kmh=69.0,
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
    monkeypatch.setattr(main_module, "_fetch_local_ors_baseline_seed", _unexpected_fetch_local_ors_baseline_seed)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options)[:1])
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_pick_best_option", lambda options, **_: min(list(options), key=lambda item: item.metrics.duration_s))
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
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
        search_budget=4,
    )

    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=5,
            pipeline_mode="dccs_refc",
            run_seed=20260331,
        )
    )

    candidate_diag = result["candidate_diag"]
    assert candidate_diag.leftover_challenger_activated is False
    assert candidate_diag.leftover_challenger_selected_count == 0
    assert candidate_diag.supplemental_challenger_activated is False
    assert candidate_diag.selected_candidate_count == 4
    dccs_summary = result["extra_json_artifacts"]["dccs_summary.json"]
    assert dccs_summary["search_budget_used"] == 4
    assert dccs_summary["leftover_challenger_activated"] is False
    assert dccs_summary["leftover_challenger_selected_count"] == 0
    assert dccs_summary["supplemental_challenger_activated"] is False
    assert len(dccs_summary["batches"]) == 1
    assert dccs_summary["batches"][0]["selected_count"] == 4
    dccs_rows = result["extra_jsonl_artifacts"]["dccs_candidates.jsonl"]
    assert not any(bool(row.get("leftover_budget_challenger")) for row in dccs_rows)


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


def test_graph_route_candidate_payload_recovers_fallback_mechanism_signals() -> None:
    route_via = _make_fallback_route(
        coordinates=[
            [-1.90, 52.50],
            [-1.78, 52.58],
            [-1.55, 52.74],
            [-1.24, 52.92],
        ],
        segment_distances_m=[8_000.0, 22_000.0, 18_000.0],
        segment_durations_s=[900.0, 1_020.0, 1_620.0],
        source_label="fallback:via:2:direct_k_raw_fallback",
    )
    route_toll_excluded = _make_fallback_route(
        coordinates=[
            [-1.90, 52.50],
            [-1.88, 52.66],
            [-1.70, 52.79],
            [-1.33, 52.90],
        ],
        segment_distances_m=[7_000.0, 16_000.0, 24_000.0],
        segment_durations_s=[1_080.0, 1_420.0, 1_500.0],
        source_label="fallback:exclude:toll:direct_k_raw_fallback",
    )
    vehicle = main_module.resolve_vehicle_profile("rigid_hgv")

    payload_via = main_module._graph_route_candidate_payload(
        route_via,
        origin=LatLng(lat=52.50, lon=-1.90),
        destination=LatLng(lat=52.90, lon=-1.33),
        vehicle=vehicle,
        cost_toggles=main_module.CostToggles(),
    )
    payload_toll_excluded = main_module._graph_route_candidate_payload(
        route_toll_excluded,
        origin=LatLng(lat=52.50, lon=-1.90),
        destination=LatLng(lat=52.90, lon=-1.33),
        vehicle=vehicle,
        cost_toggles=main_module.CostToggles(),
    )

    assert payload_via["candidate_source_stage"] == "direct_k_raw_fallback"
    assert payload_via["candidate_source_engine"] == "osrm"
    assert payload_via["candidate_source_label"] == "fallback:via:2:direct_k_raw_fallback"
    assert payload_via["seed_observed_refine_cost_ms"] == pytest.approx(19.0, rel=0.0, abs=1e-6)
    assert payload_toll_excluded["candidate_source_label"] == "fallback:exclude:toll:direct_k_raw_fallback"
    assert payload_toll_excluded["seed_observed_refine_cost_ms"] == pytest.approx(19.0, rel=0.0, abs=1e-6)
    assert payload_via["road_class_mix"]["a_road_share"] > 0.0
    assert payload_toll_excluded["road_class_mix"] != payload_via["road_class_mix"]
    assert payload_via["mechanism_descriptor"]["source_via_hint"] == 1.0
    assert payload_toll_excluded["mechanism_descriptor"]["source_exclude_toll_hint"] == 1.0
    assert payload_via["mechanism_descriptor"]["speed_variability"] > 0.0
    assert payload_toll_excluded["mechanism_descriptor"]["shape_bend_density"] > 0.0
    assert payload_via["mechanism_descriptor"] != payload_toll_excluded["mechanism_descriptor"]


def test_candidate_corridor_key_distinguishes_fallback_source_and_shape() -> None:
    vehicle = main_module.resolve_vehicle_profile("rigid_hgv")
    via_route = _make_fallback_route(
        coordinates=[
            [-1.90, 52.50],
            [-1.84, 52.56],
            [-1.58, 52.73],
            [-1.33, 52.90],
        ],
        segment_distances_m=[9_000.0, 19_000.0, 20_000.0],
        segment_durations_s=[840.0, 1_050.0, 1_470.0],
        source_label="fallback:via:1:direct_k_raw_fallback",
    )
    alt_route = _make_fallback_route(
        coordinates=[
            [-1.90, 52.50],
            [-1.70, 52.60],
            [-1.48, 52.74],
            [-1.33, 52.90],
        ],
        segment_distances_m=[15_000.0, 17_000.0, 16_000.0],
        segment_durations_s=[930.0, 1_110.0, 1_080.0],
        source_label="fallback:alternatives:direct_k_raw_fallback",
    )
    via_payload = main_module._graph_route_candidate_payload(
        via_route,
        origin=LatLng(lat=52.50, lon=-1.90),
        destination=LatLng(lat=52.90, lon=-1.33),
        vehicle=vehicle,
        cost_toggles=main_module.CostToggles(),
    )
    alt_payload = main_module._graph_route_candidate_payload(
        alt_route,
        origin=LatLng(lat=52.50, lon=-1.90),
        destination=LatLng(lat=52.90, lon=-1.33),
        vehicle=vehicle,
        cost_toggles=main_module.CostToggles(),
    )

    via_key = main_module._candidate_corridor_key(via_payload)
    alt_key = main_module._candidate_corridor_key(alt_payload)

    assert via_key != alt_key
    assert "src=via" in via_key
    assert "src=alternatives" in alt_key


def test_selected_route_source_prefers_refined_stage_over_raw_fallback_labels() -> None:
    payload = main_module._selected_route_source_payload(
        {
            "_candidate_meta": {
                "source_labels": [
                    "fallback:alternatives:direct_k_raw_fallback",
                    "graph_family:pre_realized_fallback:osrm_refined",
                ]
            }
        }
    )

    assert payload["source_label"] == "graph_family:pre_realized_fallback:osrm_refined"
    assert payload["source_stage"] == "osrm_refined"
    assert payload["source_engine"] == "internal"


def test_refine_graph_candidate_batch_re_realizes_selected_direct_fallback_routes_with_recorded_cost_on_cache_miss(
    monkeypatch,
) -> None:
    raw_route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    main_module._annotate_route_candidate_meta(
        raw_route,
        source_labels={"fallback:exclude:toll:direct_k_raw_fallback"},
        toll_exclusion_available=False,
        observed_refine_cost_ms=27.5,
    )
    realized_route = _make_ranked_route(duration_s=27_550.0, lon_seed=0.35, road_class="primary")

    async def _fake_run_candidate_fetch(**_: Any) -> Any:
        return SimpleNamespace(
            spec=SimpleNamespace(label="graph_family:candidate-1"),
            routes=[realized_route],
            error=None,
        )

    monkeypatch.setattr(main_module, "_run_candidate_fetch", _fake_run_candidate_fetch)

    routes, warnings, observed_costs, fetches, elapsed_ms = asyncio.run(
        main_module._refine_graph_candidate_batch(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.5074, lon=-0.1278),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            selected_records=[SimpleNamespace(candidate_id="candidate-1")],
            raw_graph_routes_by_id={"candidate-1": raw_route},
        )
    )

    assert warnings == []
    assert fetches == 1
    assert elapsed_ms >= 0.0
    assert observed_costs["candidate-1"] >= 0.001
    assert observed_costs["candidate-1"] < 27.5
    assert len(routes) == 1
    assert routes[0]["_dccs_candidate_ids"] == ["candidate-1"]
    assert "graph_family:candidate-1:osrm_refined" in routes[0]["_candidate_meta"]["source_labels"]
    assert "graph_family:pre_realized_fallback:osrm_refined" not in routes[0]["_candidate_meta"]["source_labels"]
    assert routes[0]["_candidate_meta"]["observed_refine_cost_ms"] == pytest.approx(
        observed_costs["candidate-1"],
        rel=0.0,
        abs=1e-6,
    )


def test_refine_graph_candidate_batch_reuses_cached_refined_routes_before_re_realizing_fallback(
    monkeypatch,
) -> None:
    raw_route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    main_module._annotate_route_candidate_meta(
        raw_route,
        source_labels={"fallback:exclude:toll:direct_k_raw_fallback"},
        toll_exclusion_available=False,
        observed_refine_cost_ms=27.5,
    )
    cached_route = _make_ranked_route(duration_s=27_540.0, lon_seed=0.4, road_class="primary")

    async def _unexpected_run_candidate_fetch(**_: Any) -> Any:
        raise AssertionError("cached refine result should be reused before re-realizing fallback candidates")

    monkeypatch.setattr(main_module, "_run_candidate_fetch", _unexpected_run_candidate_fetch)
    monkeypatch.setattr(
        main_module,
        "get_cached_routes",
        lambda _key: ([cached_route], [], 1, {"label": "graph_family:candidate-1"}),
    )

    routes, warnings, observed_costs, fetches, elapsed_ms = asyncio.run(
        main_module._refine_graph_candidate_batch(
            osrm=_NoopOSRM(),
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
    assert "graph_family:candidate-1:osrm_refined" in routes[0]["_candidate_meta"]["source_labels"]


def test_refine_graph_candidate_batch_re_realizes_selected_direct_fallback_routes_without_recorded_cost(
    monkeypatch,
) -> None:
    raw_route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    main_module._annotate_route_candidate_meta(
        raw_route,
        source_labels={"fallback:exclude:toll:direct_k_raw_fallback"},
        toll_exclusion_available=False,
    )
    realized_route = _make_ranked_route(duration_s=27_500.0, lon_seed=0.3, road_class="primary")

    async def _fake_run_candidate_fetch(**_: Any) -> Any:
        return SimpleNamespace(
            spec=SimpleNamespace(label="graph_family:candidate-1"),
            routes=[realized_route],
            error=None,
        )

    monkeypatch.setattr(main_module, "_run_candidate_fetch", _fake_run_candidate_fetch)

    routes, warnings, observed_costs, fetches, elapsed_ms = asyncio.run(
        main_module._refine_graph_candidate_batch(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.5074, lon=-0.1278),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            selected_records=[SimpleNamespace(candidate_id="candidate-1")],
            raw_graph_routes_by_id={"candidate-1": raw_route},
        )
    )

    assert warnings == []
    assert fetches == 1
    assert elapsed_ms >= 0.0
    assert observed_costs["candidate-1"] >= 0.001
    assert len(routes) == 1
    assert routes[0]["_dccs_candidate_ids"] == ["candidate-1"]
    assert "graph_family:candidate-1:osrm_refined" in routes[0]["_candidate_meta"]["source_labels"]


def test_refine_graph_candidate_batch_falls_back_to_pre_realized_route_when_re_realization_fails(
    monkeypatch,
) -> None:
    raw_route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    main_module._annotate_route_candidate_meta(
        raw_route,
        source_labels={"fallback:exclude:toll:direct_k_raw_fallback"},
        toll_exclusion_available=False,
    )

    async def _failed_run_candidate_fetch(**_: Any) -> Any:
        return SimpleNamespace(
            spec=SimpleNamespace(label="graph_family:candidate-1"),
            routes=[],
            error="unavailable",
        )

    monkeypatch.setattr(main_module, "_run_candidate_fetch", _failed_run_candidate_fetch)

    routes, warnings, observed_costs, fetches, elapsed_ms = asyncio.run(
        main_module._refine_graph_candidate_batch(
            osrm=_NoopOSRM(),
            origin=LatLng(lat=51.5074, lon=-0.1278),
            destination=LatLng(lat=53.4808, lon=-2.2426),
            selected_records=[SimpleNamespace(candidate_id="candidate-1")],
            raw_graph_routes_by_id={"candidate-1": raw_route},
        )
    )

    assert warnings == ["graph_family:candidate-1: unavailable"]
    assert fetches == 1
    assert elapsed_ms >= 0.0
    assert observed_costs == {}
    assert len(routes) == 1
    assert routes[0]["_dccs_candidate_ids"] == ["candidate-1"]


def test_run_candidate_fetch_elapsed_excludes_semaphore_wait() -> None:
    class _SlowOSRM:
        async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
            await asyncio.sleep(0.01)
            return [_make_ranked_route(duration_s=27_500.0, lon_seed=0.3, road_class="primary")]

    async def _exercise() -> tuple[Any, float]:
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        task = asyncio.create_task(
            main_module._run_candidate_fetch(
                osrm=_SlowOSRM(),
                origin=LatLng(lat=51.5074, lon=-0.1278),
                destination=LatLng(lat=53.4808, lon=-2.2426),
                spec=main_module.CandidateFetchSpec(
                    label="graph_family:candidate-1",
                    alternatives=False,
                ),
                sem=sem,
            )
        )
        wall_started = time.perf_counter()
        await asyncio.sleep(0.12)
        sem.release()
        result = await task
        wall_elapsed_ms = (time.perf_counter() - wall_started) * 1000.0
        return result, wall_elapsed_ms

    result, wall_elapsed_ms = asyncio.run(_exercise())

    assert result.error is None
    assert wall_elapsed_ms >= 100.0
    assert result.elapsed_ms < 60.0


def test_annotate_route_candidate_meta_merges_existing_labels() -> None:
    route = _make_ranked_route(duration_s=28_000.0, lon_seed=0.0, road_class="motorway")
    main_module._annotate_route_candidate_meta(
        route,
        source_labels={"fallback:exclude:toll:direct_k_raw_fallback"},
        toll_exclusion_available=False,
        observed_refine_cost_ms=33.0,
    )
    main_module._annotate_route_candidate_meta(
        route,
        source_labels={"fallback:alternatives:direct_k_raw_fallback"},
        toll_exclusion_available=False,
        observed_refine_cost_ms=21.0,
    )
    meta = route["_candidate_meta"]
    assert meta["seen_by_exclude_toll"] is True
    assert sorted(meta["source_labels"]) == [
        "fallback:alternatives:direct_k_raw_fallback",
        "fallback:exclude:toll:direct_k_raw_fallback",
    ]
    assert meta["observed_refine_cost_ms"] == 21.0
