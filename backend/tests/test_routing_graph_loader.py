from __future__ import annotations

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
