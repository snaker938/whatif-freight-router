from __future__ import annotations

import json
import pickle
from pathlib import Path

import app.routing_graph as routing_graph
from app.routing_graph import GraphEdge, RouteGraph, _grid_key, load_route_graph


def _make_graph() -> RouteGraph:
    nodes = {
        "a": (52.5, -1.9),
        "b": (52.6, -1.8),
    }
    edge = GraphEdge(
        to="b",
        cost=10.0,
        distance_m=1000.0,
        highway="primary",
        toll=False,
        maxspeed_kph=70.0,
    )
    adjacency = {"a": (edge,)}
    edge_index = {("a", "b"): edge}
    grid_index = {}
    for node_id, (lat, lon) in nodes.items():
        grid_index.setdefault(_grid_key(lat, lon), []).append(node_id)
    return RouteGraph(
        version="pytest",
        source="pytest.pbf",
        nodes=nodes,
        adjacency=adjacency,
        edge_index=edge_index,
        grid_index={key: tuple(values) for key, values in grid_index.items()},
        component_by_node={"a": 1, "b": 1},
        component_sizes={1: 2},
        component_count=1,
        largest_component_nodes=2,
        largest_component_ratio=1.0,
        graph_fragmented=False,
    )


def test_load_route_graph_uses_streaming_loader(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "routing_graph_uk.json"
    path.write_text("{}", encoding="utf-8")
    expected = _make_graph()
    called: dict[str, Path] = {}

    def _fake_stream_loader(*, path: Path) -> RouteGraph:
        called["path"] = path
        return expected

    monkeypatch.setattr(routing_graph.settings, "route_graph_asset_path", str(path))
    monkeypatch.setattr(routing_graph, "ijson", object())
    monkeypatch.setattr(routing_graph, "_load_route_graph_streaming", _fake_stream_loader)
    load_route_graph.cache_clear()
    try:
        result = load_route_graph()
        assert result is expected
        assert called["path"] == path
    finally:
        load_route_graph.cache_clear()


def test_load_route_graph_requires_streaming_parser(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "routing_graph_uk.json"
    path.write_text("{}", encoding="utf-8")

    def _unexpected_loader(*, path: Path) -> RouteGraph:
        raise AssertionError("stream loader should not run when ijson is unavailable")

    monkeypatch.setattr(routing_graph.settings, "route_graph_asset_path", str(path))
    monkeypatch.setattr(routing_graph, "ijson", None)
    monkeypatch.setattr(routing_graph, "_load_route_graph_streaming", _unexpected_loader)
    load_route_graph.cache_clear()
    try:
        assert load_route_graph() is None
    finally:
        load_route_graph.cache_clear()


def test_streaming_loader_can_skip_edge_index_materialization(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "routing_graph_uk.json"
    path.write_text(
        json.dumps(
            {
                "version": "pytest",
                "source": "pytest.pbf",
                "nodes": [
                    {"id": "a", "lat": 52.5, "lon": -1.9},
                    {"id": "b", "lat": 52.6, "lon": -1.8},
                ],
                "edges": [
                    {
                        "u": "a",
                        "v": "b",
                        "distance_m": 1000.0,
                        "generalized_cost": 1100.0,
                        "oneway": False,
                        "highway": "primary",
                        "toll": False,
                        "maxspeed_kph": 70.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(routing_graph.settings, "route_graph_materialize_edge_index", False)

    graph = routing_graph._load_route_graph_streaming(path=path)

    assert graph is not None
    assert graph.edge_index == {}
    assert routing_graph._graph_edge_lookup(graph, "a", "b") is not None
    assert graph.min_cost_per_meter > 0.0


def test_route_graph_status_returns_fast_ready_without_full_load(monkeypatch) -> None:
    load_route_graph.cache_clear()
    original_state = routing_graph._WARMUP_STATE
    original_ready_mode = routing_graph._WARMUP_READY_MODE
    try:
        monkeypatch.setattr(routing_graph.settings, "route_graph_enabled", True)
        monkeypatch.setattr(routing_graph.settings, "route_graph_fast_startup_enabled", True)
        routing_graph._WARMUP_STATE = "ready"
        routing_graph._WARMUP_READY_MODE = "fast"

        assert routing_graph.route_graph_status() == (True, "ok_fast")
    finally:
        routing_graph._WARMUP_STATE = original_state
        routing_graph._WARMUP_READY_MODE = original_ready_mode
        load_route_graph.cache_clear()


def test_route_graph_k_shortest_paths_defers_under_fast_startup(monkeypatch) -> None:
    load_route_graph.cache_clear()
    original_state = routing_graph._WARMUP_STATE
    original_ready_mode = routing_graph._WARMUP_READY_MODE
    try:
        monkeypatch.setattr(routing_graph.settings, "route_graph_fast_startup_enabled", True)
        routing_graph._WARMUP_STATE = "ready"
        routing_graph._WARMUP_READY_MODE = "fast"

        candidates, diagnostics = routing_graph.route_graph_candidate_routes(
            origin_lat=51.5,
            origin_lon=-0.1,
            destination_lat=52.5,
            destination_lon=-1.1,
        )

        assert candidates == []
        assert diagnostics.no_path_reason == "routing_graph_deferred_load"
    finally:
        routing_graph._WARMUP_STATE = original_state
        routing_graph._WARMUP_READY_MODE = original_ready_mode
        load_route_graph.cache_clear()


def test_fast_startup_warmup_promotes_small_compact_bundle_to_full_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "routing_graph_subset.json"
    path.write_text(
        json.dumps(
            {
                "version": "uk-routing-graph-v1",
                "source": "unit-test-source.osm.pbf#subset:single_od.csv",
                "generated_at_utc": "2026-04-05T00:00:00Z",
                "as_of_utc": "2026-04-05T00:00:00Z",
                "nodes": [
                    {"id": "a", "lat": 52.5, "lon": -1.9},
                    {"id": "b", "lat": 52.6, "lon": -1.8},
                ],
                "edges": [
                    {
                        "u": "a",
                        "v": "b",
                        "distance_m": 1000.0,
                        "generalized_cost": 1100.0,
                        "oneway": False,
                        "highway": "primary",
                        "toll": False,
                        "maxspeed_kph": 70.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    graph = routing_graph._load_route_graph_streaming(path=path)
    assert graph is not None
    routing_graph._save_route_graph_compact_bundle(path, graph)

    load_route_graph.cache_clear()
    original_state = routing_graph._WARMUP_STATE
    original_ready_mode = routing_graph._WARMUP_READY_MODE
    original_thread = routing_graph._WARMUP_THREAD
    try:
        monkeypatch.setattr(routing_graph.settings, "route_graph_asset_path", str(path))
        monkeypatch.setattr(routing_graph.settings, "route_graph_fast_startup_enabled", True)
        monkeypatch.setattr(routing_graph.settings, "route_graph_binary_cache_warmup_max_bytes", 1_000_000)
        monkeypatch.setattr(routing_graph.settings, "route_graph_strict_required", False)
        monkeypatch.setattr(routing_graph.settings, "route_graph_min_giant_component_nodes", 1)
        monkeypatch.setattr(routing_graph.settings, "route_graph_min_giant_component_ratio", 0.0)
        routing_graph._clear_loaded_route_graph_caches()
        routing_graph._mark_warmup_loading()
        routing_graph._warmup_worker()

        assert routing_graph._WARMUP_READY_MODE == "full"
        assert routing_graph.route_graph_status() == (True, "ok")
        assert load_route_graph.cache_info().currsize == 1
    finally:
        routing_graph._WARMUP_STATE = original_state
        routing_graph._WARMUP_READY_MODE = original_ready_mode
        routing_graph._WARMUP_THREAD = original_thread
        routing_graph._clear_loaded_route_graph_caches()


def test_load_route_graph_compact_bundle_rebuilds_missing_derived_topology(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "routing_graph_subset.json"
    path.write_text(
        json.dumps(
            {
                "version": "uk-routing-graph-v1",
                "source": "unit-test-source.osm.pbf#subset:single_od.csv",
                "generated_at_utc": "2026-04-05T00:00:00Z",
                "as_of_utc": "2026-04-05T00:00:00Z",
                "nodes": [
                    {"id": "a", "lat": 52.5, "lon": -1.9},
                    {"id": "b", "lat": 52.6, "lon": -1.8},
                ],
                "edges": [
                    {
                        "u": "a",
                        "v": "b",
                        "distance_m": 1000.0,
                        "generalized_cost": 1100.0,
                        "oneway": False,
                        "highway": "primary",
                        "toll": False,
                        "maxspeed_kph": 70.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    graph = routing_graph._load_route_graph_streaming(path=path)
    assert graph is not None
    routing_graph._save_route_graph_compact_bundle(path, graph)

    bundle_path = routing_graph._graph_compact_bundle_path(path)
    with bundle_path.open("rb") as fh:
        compact_payload = pickle.load(fh)
    graph_payload = compact_payload["graph"]
    for key in (
        "grid_index",
        "component_by_node",
        "component_sizes",
        "component_count",
        "largest_component_nodes",
        "largest_component_ratio",
        "graph_fragmented",
    ):
        graph_payload.pop(key, None)
    with bundle_path.open("wb") as fh:
        pickle.dump(compact_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    monkeypatch.setattr(routing_graph.settings, "route_graph_compact_bundle_enabled", True)
    monkeypatch.setattr(routing_graph.settings, "route_graph_min_giant_component_nodes", 1)
    monkeypatch.setattr(routing_graph.settings, "route_graph_min_giant_component_ratio", 0.0)
    rebuilt = routing_graph._load_route_graph_compact_bundle(path)

    assert rebuilt is not None
    assert rebuilt.grid_index
    assert rebuilt.component_by_node == {"a": 1, "b": 1}
    assert rebuilt.component_sizes == {1: 2}
    assert rebuilt.component_count == 1
    assert rebuilt.largest_component_nodes == 2
    assert rebuilt.largest_component_ratio == 1.0
    assert rebuilt.graph_fragmented is False


def test_save_route_graph_compact_bundle_uses_direct_graph_ready_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "routing_graph_subset.json"
    path.write_text(
        json.dumps(
            {
                "version": "uk-routing-graph-v1",
                "source": "unit-test-source.osm.pbf#subset:single_od.csv",
                "generated_at_utc": "2026-04-05T00:00:00Z",
                "as_of_utc": "2026-04-05T00:00:00Z",
                "nodes": [
                    {"id": "a", "lat": 52.5, "lon": -1.9},
                    {"id": "b", "lat": 52.6, "lon": -1.8},
                ],
                "edges": [
                    {
                        "u": "a",
                        "v": "b",
                        "distance_m": 1000.0,
                        "generalized_cost": 1100.0,
                        "oneway": False,
                        "highway": "primary",
                        "toll": False,
                        "maxspeed_kph": 70.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    graph = routing_graph._load_route_graph_streaming(path=path)
    assert graph is not None
    routing_graph._save_route_graph_compact_bundle(path, graph)

    bundle_path = routing_graph._graph_compact_bundle_path(path)
    with bundle_path.open("rb") as fh:
        compact_payload = pickle.load(fh)

    graph_payload = compact_payload["graph"]
    assert graph_payload["schema_version"] == 2
    assert graph_payload["nodes"] == graph.nodes
    adjacency = graph_payload["adjacency"]
    assert isinstance(adjacency, dict)
    assert isinstance(adjacency["a"], tuple)
    assert isinstance(adjacency["a"][0], GraphEdge)

    rebuilt = routing_graph._load_route_graph_compact_bundle(path)
    assert rebuilt is not None
    assert rebuilt.nodes == graph.nodes
    assert routing_graph._graph_edge_lookup(rebuilt, "a", "b") is not None


def test_load_route_graph_uncached_prefers_binary_cache_for_subset_assets(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "routing_graph_uk.subset.unit.json"
    observed: list[str] = []

    monkeypatch.setattr(
        routing_graph,
        "_load_route_graph_binary_cache",
        lambda _path: observed.append("binary") or {"source": "binary"},
    )
    monkeypatch.setattr(
        routing_graph,
        "_load_route_graph_compact_bundle",
        lambda _path: observed.append("compact") or {"source": "compact"},
    )

    loaded = routing_graph._load_route_graph_uncached(path)

    assert loaded == {"source": "binary"}
    assert observed == ["binary"]


def test_load_route_graph_uncached_keeps_compact_bundle_priority_for_non_subset_assets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "routing_graph_uk.json"
    observed: list[str] = []

    monkeypatch.setattr(
        routing_graph,
        "_load_route_graph_binary_cache",
        lambda _path: observed.append("binary") or {"source": "binary"},
    )
    monkeypatch.setattr(
        routing_graph,
        "_load_route_graph_compact_bundle",
        lambda _path: observed.append("compact") or {"source": "compact"},
    )

    loaded = routing_graph._load_route_graph_uncached(path)

    assert loaded == {"source": "compact"}
    assert observed == ["compact"]


def test_finalize_graph_freezes_adjacency_in_place(monkeypatch) -> None:
    monkeypatch.setattr(routing_graph.settings, "route_graph_min_giant_component_nodes", 1)
    monkeypatch.setattr(routing_graph.settings, "route_graph_min_giant_component_ratio", 0.0)
    nodes = {
        "a": (52.5, -1.9),
        "b": (52.6, -1.8),
        "c": (52.7, -1.7),
    }
    adjacency_mut = {
        "a": [
            routing_graph.GraphEdge(
                to="b",
                cost=1100.0,
                distance_m=1000.0,
                highway="primary",
                toll=False,
                maxspeed_kph=70.0,
            )
        ],
        "b": [
            routing_graph.GraphEdge(
                to="a",
                cost=1100.0,
                distance_m=1000.0,
                highway="primary",
                toll=False,
                maxspeed_kph=70.0,
            )
        ],
        "c": [],
    }

    graph = routing_graph._finalize_graph(
        version="uk-routing-graph-v1",
        source="unit-test-source.osm.pbf#subset:single_od.csv",
        nodes=nodes,
        adjacency_mut=adjacency_mut,
        edge_index={},
        min_cost_per_meter_hint=1.1,
    )

    assert graph is not None
    assert graph.adjacency is adjacency_mut
    assert isinstance(graph.adjacency["a"], tuple)
    assert isinstance(graph.adjacency["b"], tuple)
    assert "c" not in graph.adjacency
    assert all(isinstance(node_ids, tuple) for node_ids in graph.grid_index.values())
    assert graph.component_by_node["a"] == graph.component_by_node["b"]
    assert graph.component_by_node["c"] != graph.component_by_node["a"]
    assert graph.component_sizes == {1: 2, 2: 1}
    assert graph.min_cost_per_meter == 1.1
