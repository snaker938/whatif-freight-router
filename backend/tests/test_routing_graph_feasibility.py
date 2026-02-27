from __future__ import annotations

from app.routing_graph import RouteGraph, _grid_key, route_graph_od_feasibility
from app.settings import settings


def _make_graph(
    *,
    nodes: dict[str, tuple[float, float]],
    component_by_node: dict[str, int],
    component_sizes: dict[int, int],
    graph_fragmented: bool,
) -> RouteGraph:
    grid_mut: dict[tuple[int, int], list[str]] = {}
    for node_id, (lat, lon) in nodes.items():
        grid_mut.setdefault(_grid_key(lat, lon), []).append(node_id)
    return RouteGraph(
        version="pytest",
        source="pytest.pbf",
        nodes=nodes,
        adjacency={},
        edge_index={},
        grid_index={key: tuple(values) for key, values in grid_mut.items()},
        component_by_node=component_by_node,
        component_sizes=component_sizes,
        component_count=len(component_sizes),
        largest_component_nodes=max(component_sizes.values(), default=0),
        largest_component_ratio=(
            max(component_sizes.values(), default=0) / float(max(1, len(nodes)))
            if nodes
            else 0.0
        ),
        graph_fragmented=graph_fragmented,
    )


def test_route_graph_feasibility_reports_fragmented(monkeypatch) -> None:
    graph = _make_graph(
        nodes={"a": (52.5, -1.9)},
        component_by_node={"a": 1},
        component_sizes={1: 1},
        graph_fragmented=True,
    )
    monkeypatch.setattr("app.routing_graph.load_route_graph", lambda: graph)
    result = route_graph_od_feasibility(
        origin_lat=52.5,
        origin_lon=-1.9,
        destination_lat=52.6,
        destination_lon=-1.8,
    )
    assert result["ok"] is False
    assert result["reason_code"] == "routing_graph_fragmented"


def test_route_graph_feasibility_reports_coverage_gap(monkeypatch) -> None:
    graph = _make_graph(
        nodes={"a": (52.5, -1.9)},
        component_by_node={"a": 1},
        component_sizes={1: 1},
        graph_fragmented=False,
    )
    monkeypatch.setattr("app.routing_graph.load_route_graph", lambda: graph)
    monkeypatch.setattr(settings, "route_graph_max_nearest_node_distance_m", 1000.0)
    result = route_graph_od_feasibility(
        origin_lat=55.0,
        origin_lon=0.0,
        destination_lat=55.1,
        destination_lon=0.1,
    )
    assert result["ok"] is False
    assert result["reason_code"] == "routing_graph_coverage_gap"


def test_route_graph_feasibility_prefers_component_aligned_candidates(monkeypatch) -> None:
    origin = (52.5, -1.9)
    destination = (52.6, -1.8)
    nodes = {
        "origin_nearest": (52.5002, -1.9),      # nearest to origin but isolated component
        "origin_shared": (52.5010, -1.9),       # slightly farther, shared component
        "destination_nearest": (52.6002, -1.8), # nearest to destination but isolated component
        "destination_shared": (52.5990, -1.8),  # slightly farther, shared component
    }
    graph = _make_graph(
        nodes=nodes,
        component_by_node={
            "origin_nearest": 1,
            "origin_shared": 2,
            "destination_nearest": 3,
            "destination_shared": 2,
        },
        component_sizes={1: 8, 2: 400, 3: 11},
        graph_fragmented=False,
    )
    monkeypatch.setattr("app.routing_graph.load_route_graph", lambda: graph)
    monkeypatch.setattr(settings, "route_graph_max_nearest_node_distance_m", 5_000.0)
    result = route_graph_od_feasibility(
        origin_lat=origin[0],
        origin_lon=origin[1],
        destination_lat=destination[0],
        destination_lon=destination[1],
    )
    assert result["ok"] is True
    assert result["reason_code"] == "ok"
    assert result["selected_component"] == 2
    assert result["origin_node_id"] == "origin_shared"
    assert result["destination_node_id"] == "destination_shared"
    assert result["origin_selected_distance_m"] >= result["origin_nearest_distance_m"]
    assert result["destination_selected_distance_m"] >= result["destination_nearest_distance_m"]


def test_route_graph_feasibility_reports_disconnected_od(monkeypatch) -> None:
    nodes = {
        "a": (52.5, -1.9),
        "b": (52.6, -1.8),
    }
    graph = _make_graph(
        nodes=nodes,
        component_by_node={"a": 1, "b": 2},
        component_sizes={1: 1, 2: 1},
        graph_fragmented=False,
    )
    monkeypatch.setattr("app.routing_graph.load_route_graph", lambda: graph)
    monkeypatch.setattr(settings, "route_graph_max_nearest_node_distance_m", 5_000.0)
    result = route_graph_od_feasibility(
        origin_lat=52.5,
        origin_lon=-1.9,
        destination_lat=52.6,
        destination_lon=-1.8,
    )
    assert result["ok"] is False
    assert result["reason_code"] == "routing_graph_disconnected_od"
