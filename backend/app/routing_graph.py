from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .k_shortest import (
    PathResult,
    TransitionStateFn,
    yen_k_shortest_paths,
    yen_k_shortest_paths_with_stats,
)
from .settings import settings

try:  # pragma: no cover - optional fast path for huge graph assets
    import ijson
except Exception:  # pragma: no cover
    ijson = None  # type: ignore[assignment]


EARTH_RADIUS_M = 6_371_000.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + (math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2))
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _grid_key(lat: float, lon: float, bucket_deg: float = 0.15) -> tuple[int, int]:
    return (int(math.floor(lat / bucket_deg)), int(math.floor(lon / bucket_deg)))


@dataclass(frozen=True)
class RouteGraph:
    version: str
    source: str
    nodes: dict[str, tuple[float, float]]
    adjacency: dict[str, tuple[GraphEdge, ...]]
    edge_index: dict[tuple[str, str], GraphEdge]
    grid_index: dict[tuple[int, int], tuple[str, ...]]


@dataclass(frozen=True)
class GraphCandidateDiagnostics:
    explored_states: int
    generated_paths: int
    emitted_paths: int
    candidate_budget: int


@dataclass(frozen=True)
class GraphEdge:
    to: str
    cost: float
    distance_m: float
    highway: str
    toll: bool
    maxspeed_kph: float | None = None


def _graph_asset_path() -> Path:
    explicit = (settings.route_graph_asset_path or "").strip()
    if explicit:
        return Path(explicit)
    return Path(settings.model_asset_dir) / "routing_graph_uk.json"


def _parse_node(raw: dict[str, object]) -> tuple[str, float, float] | None:
    node_id_raw = raw.get("id")
    if node_id_raw is None:
        return None
    node_id = str(node_id_raw)
    lat_raw = raw.get("lat", 0.0)
    lon_raw = raw.get("lon", 0.0)
    if not isinstance(lat_raw, (int, float, str)):
        return None
    if not isinstance(lon_raw, (int, float, str)):
        return None
    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return (node_id, lat, lon)


def _parse_edge(raw: object) -> tuple[str, str, GraphEdge, bool] | None:
    if isinstance(raw, dict):
        u = raw.get("u")
        v = raw.get("v")
        if u is None or v is None:
            return None
        distance_m = raw.get("distance_m", raw.get("weight", 0.0))
        generalized_cost = raw.get("generalized_cost", distance_m)
        oneway = bool(raw.get("oneway", False))
        highway = str(raw.get("highway", "unclassified")).strip().lower() or "unclassified"
        toll = bool(raw.get("toll", False))
        maxspeed_raw = raw.get("maxspeed_kph")
        if maxspeed_raw is None:
            maxspeed_kph = None
        elif isinstance(maxspeed_raw, (int, float, str)):
            try:
                maxspeed_kph = float(maxspeed_raw)
            except (TypeError, ValueError):
                maxspeed_kph = None
        else:
            maxspeed_kph = None
    elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
        u, v, distance_m = raw[0], raw[1], raw[2]
        generalized_cost = distance_m
        oneway = bool(raw[3]) if len(raw) > 3 else False
        highway = "unclassified"
        toll = False
        maxspeed_kph = None
    else:
        return None
    try:
        dist = max(1.0, float(distance_m))
        cost = max(1.0, float(generalized_cost))
    except (TypeError, ValueError):
        return None
    return (
        str(u),
        str(v),
        GraphEdge(
            to=str(v),
            cost=cost,
            distance_m=dist,
            highway=highway,
            toll=toll,
            maxspeed_kph=maxspeed_kph,
        ),
        oneway,
    )


def _graph_meta_from_head(path: Path) -> tuple[str, str]:
    try:
        with path.open("rb") as fh:
            head = fh.read(65_536).decode("utf-8", errors="ignore")
    except Exception:
        return "unknown", str(path)
    version_match = re.search(r'"version"\s*:\s*"([^"]+)"', head)
    source_match = re.search(r'"source"\s*:\s*"([^"]+)"', head)
    version = version_match.group(1) if version_match else "unknown"
    source = source_match.group(1) if source_match else str(path)
    return version, source


def _finalize_graph(
    *,
    version: str,
    source: str,
    nodes: dict[str, tuple[float, float]],
    adjacency_mut: dict[str, list[GraphEdge]],
    edge_index: dict[tuple[str, str], GraphEdge],
) -> RouteGraph | None:
    adjacency = {k: tuple(v) for k, v in adjacency_mut.items() if v}
    if not nodes or not adjacency:
        return None

    grid_mut: dict[tuple[int, int], list[str]] = {}
    for node_id, (lat, lon) in nodes.items():
        key = _grid_key(lat, lon)
        grid_mut.setdefault(key, []).append(node_id)
    grid_index = {key: tuple(values) for key, values in grid_mut.items()}
    return RouteGraph(
        version=version,
        source=source,
        nodes=nodes,
        adjacency=adjacency,
        edge_index=edge_index,
        grid_index=grid_index,
    )


def _load_route_graph_streaming(
    *,
    path: Path,
    max_nodes: int,
    max_edges: int,
) -> RouteGraph | None:
    if ijson is None:
        return None
    version, source = _graph_meta_from_head(path)

    nodes: dict[str, tuple[float, float]] = {}
    try:
        with path.open("rb") as fh:
            for raw in ijson.items(fh, "nodes.item"):
                if not isinstance(raw, dict):
                    continue
                parsed = _parse_node(raw)
                if parsed is None:
                    continue
                node_id, lat, lon = parsed
                nodes[node_id] = (lat, lon)
                if len(nodes) >= max_nodes:
                    break
    except Exception:
        return None
    if not nodes:
        return None

    adjacency_mut: dict[str, list[GraphEdge]] = {node_id: [] for node_id in nodes}
    edge_index: dict[tuple[str, str], GraphEdge] = {}
    try:
        with path.open("rb") as fh:
            for raw in ijson.items(fh, "edges.item"):
                parsed = _parse_edge(raw)
                if parsed is None:
                    continue
                u, v, edge, oneway = parsed
                if u not in nodes or v not in nodes:
                    continue
                adjacency_mut[u].append(edge)
                edge_index[(u, v)] = edge
                if not oneway:
                    reverse = GraphEdge(
                        to=u,
                        cost=edge.cost,
                        distance_m=edge.distance_m,
                        highway=edge.highway,
                        toll=edge.toll,
                        maxspeed_kph=edge.maxspeed_kph,
                    )
                    adjacency_mut[v].append(reverse)
                    edge_index[(v, u)] = reverse
                if len(edge_index) >= max_edges:
                    break
    except Exception:
        return None
    return _finalize_graph(
        version=version,
        source=source,
        nodes=nodes,
        adjacency_mut=adjacency_mut,
        edge_index=edge_index,
    )


@lru_cache(maxsize=1)
def load_route_graph() -> RouteGraph | None:
    path = _graph_asset_path()
    if not path.exists():
        return None
    try:
        if (
            bool(settings.route_graph_streaming_load_enabled)
            and path.stat().st_size >= int(settings.route_graph_streaming_size_threshold_mb) * 1024 * 1024
        ):
            streamed = _load_route_graph_streaming(
                path=path,
                max_nodes=max(1_000, int(settings.route_graph_streaming_max_nodes)),
                max_edges=max(2_000, int(settings.route_graph_streaming_max_edges)),
            )
            if streamed is not None:
                return streamed
        payload = json.loads(path.read_text(encoding="utf-8"))
    except MemoryError:
        streamed = _load_route_graph_streaming(
            path=path,
            max_nodes=max(1_000, int(settings.route_graph_streaming_max_nodes)),
            max_edges=max(2_000, int(settings.route_graph_streaming_max_edges)),
        )
        if streamed is not None:
            return streamed
        return None
    if not isinstance(payload, dict):
        return None
    raw_nodes = payload.get("nodes", [])
    raw_edges = payload.get("edges", [])
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        return None

    nodes: dict[str, tuple[float, float]] = {}
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        parsed = _parse_node(raw)
        if parsed is None:
            continue
        node_id, lat, lon = parsed
        nodes[node_id] = (lat, lon)
    if not nodes:
        return None

    adjacency_mut: dict[str, list[GraphEdge]] = {node_id: [] for node_id in nodes}
    edge_index: dict[tuple[str, str], GraphEdge] = {}
    for raw in raw_edges:
        parsed = _parse_edge(raw)
        if parsed is None:
            continue
        u, v, edge, oneway = parsed
        if u not in nodes or v not in nodes:
            continue
        adjacency_mut[u].append(edge)
        edge_index[(u, v)] = edge
        if not oneway:
            reverse = GraphEdge(
                to=u,
                cost=edge.cost,
                distance_m=edge.distance_m,
                highway=edge.highway,
                toll=edge.toll,
                maxspeed_kph=edge.maxspeed_kph,
            )
            adjacency_mut[v].append(reverse)
            edge_index[(v, u)] = reverse
    return _finalize_graph(
        version=str(payload.get("version", "unknown")),
        source=str(payload.get("source", str(path))),
        nodes=nodes,
        adjacency_mut=adjacency_mut,
        edge_index=edge_index,
    )


def route_graph_status() -> tuple[bool, str]:
    if not settings.route_graph_enabled:
        return False, "disabled"
    graph = load_route_graph()
    if graph is None:
        return False, "unavailable"
    if bool(settings.route_graph_strict_required):
        src = (graph.source or "").strip().lower()
        if ".pbf" not in src:
            return False, "non_pbf_source"
        if (
            len(graph.nodes) < int(settings.route_graph_min_nodes)
            or len(graph.adjacency) < int(settings.route_graph_min_adjacency)
        ):
            return False, "insufficient_graph_coverage"
    return True, "ok"


def _adjacency_cost_view(
    graph: RouteGraph,
    *,
    scenario_edge_modifiers: dict[str, Any] | None = None,
) -> dict[str, tuple[tuple[str, float], ...]]:
    road_penalty = {
        "motorway": 0.98,
        "motorway_link": 1.04,
        "trunk": 1.02,
        "trunk_link": 1.08,
        "primary": 1.08,
        "primary_link": 1.14,
        "secondary": 1.14,
        "secondary_link": 1.20,
        "tertiary": 1.19,
        "tertiary_link": 1.24,
        "unclassified": 1.27,
        "residential": 1.34,
    }
    toll_propensity_penalty = 26.0
    turn_burden_proxy = {
        "motorway_link": 6.0,
        "trunk_link": 4.5,
        "primary_link": 3.5,
        "secondary_link": 2.5,
        "tertiary_link": 1.8,
    }
    direction_legality_proxy = {
        "residential": 1.8,
        "unclassified": 1.4,
    }
    scenario_mod = scenario_edge_modifiers or {}
    traffic_pressure = max(0.6, min(2.2, float(scenario_mod.get("traffic_pressure", 1.0))))
    incident_pressure = max(0.6, min(2.4, float(scenario_mod.get("incident_pressure", 1.0))))
    weather_pressure = max(0.6, min(2.1, float(scenario_mod.get("weather_pressure", 1.0))))
    duration_multiplier = max(0.6, min(2.5, float(scenario_mod.get("duration_multiplier", 1.0))))
    incident_rate_multiplier = max(0.5, min(2.8, float(scenario_mod.get("incident_rate_multiplier", 1.0))))
    incident_delay_multiplier = max(0.5, min(2.8, float(scenario_mod.get("incident_delay_multiplier", 1.0))))
    sigma_multiplier = max(0.4, min(2.8, float(scenario_mod.get("stochastic_sigma_multiplier", 1.0))))
    weather_regime_factor = max(0.75, min(1.75, float(scenario_mod.get("weather_regime_factor", 1.0))))
    hour_bucket_factor = max(0.75, min(1.75, float(scenario_mod.get("hour_bucket_factor", 1.0))))
    road_class_factors_raw = scenario_mod.get("road_class_factors", {})
    road_class_factors: dict[str, float] = {}
    if isinstance(road_class_factors_raw, dict):
        for key, value in road_class_factors_raw.items():
            k = str(key).strip().lower()
            if not k:
                continue
            try:
                road_class_factors[k] = max(0.70, min(1.85, float(value)))
            except (TypeError, ValueError):
                continue
    weather_regime_factor = max(0.75, min(1.75, float(scenario_mod.get("weather_regime_factor", 1.0))))
    hour_bucket_factor = max(0.75, min(1.75, float(scenario_mod.get("hour_bucket_factor", 1.0))))
    road_class_factors_raw = scenario_mod.get("road_class_factors", {})
    road_class_factors: dict[str, float] = {}
    if isinstance(road_class_factors_raw, dict):
        for key, value in road_class_factors_raw.items():
            k = str(key).strip().lower()
            if not k:
                continue
            try:
                road_class_factors[k] = max(0.70, min(1.85, float(value)))
            except (TypeError, ValueError):
                continue
    mode = str(scenario_mod.get("mode", "no_sharing")).strip().lower()
    mode_toll_bias = 1.0
    if mode == "partial_sharing":
        mode_toll_bias = 0.95
    elif mode == "full_sharing":
        mode_toll_bias = 0.9
    scenario_dynamic_factor = max(
        0.7,
        min(
            2.6,
            (0.34 * duration_multiplier)
            + (0.20 * traffic_pressure)
            + (0.18 * incident_pressure)
            + (0.10 * weather_pressure)
            + (0.08 * weather_regime_factor)
            + (0.06 * hour_bucket_factor)
            + (0.02 * incident_rate_multiplier)
            + (0.01 * incident_delay_multiplier)
            + (0.01 * sigma_multiplier),
        ),
    )
    road_sensitivity = {
        "motorway": 0.75,
        "motorway_link": 0.95,
        "trunk": 0.85,
        "trunk_link": 1.05,
        "primary": 1.10,
        "primary_link": 1.15,
        "secondary": 1.22,
        "secondary_link": 1.30,
        "tertiary": 1.35,
        "tertiary_link": 1.42,
        "unclassified": 1.46,
        "residential": 1.55,
    }
    return {
        node: tuple(
            (
                edge.to,
                max(
                    1.0,
                    (
                        float(edge.cost)
                        * road_penalty.get(edge.highway, 1.22)
                        * road_class_factors.get(edge.highway, 1.0)
                        * (
                            1.0
                            + ((scenario_dynamic_factor - 1.0) * road_sensitivity.get(edge.highway, 1.25) * 0.40)
                        )
                    )
                    + turn_burden_proxy.get(edge.highway, 0.0)
                    + direction_legality_proxy.get(edge.highway, 0.0)
                    + ((toll_propensity_penalty * mode_toll_bias) if edge.toll else 0.0),
                ),
            )
            for edge in edges
        )
        for node, edges in graph.adjacency.items()
}


def _heading_bin(heading_deg: float) -> int:
    normalized = (float(heading_deg) + 360.0) % 360.0
    return int(((normalized + 22.5) % 360.0) // 45.0)


def _segment_heading_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dy = lat2 - lat1
    dx = lon2 - lon1
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return 0.0
    ang = math.degrees(math.atan2(dy, dx))
    return (ang + 360.0) % 360.0


def _heading_delta_deg(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


def _transition_state_callback(
    graph: RouteGraph,
    *,
    scenario_edge_modifiers: dict[str, Any] | None = None,
) -> TransitionStateFn:
    scenario_mod = scenario_edge_modifiers or {}
    traffic_pressure = max(0.6, min(2.2, float(scenario_mod.get("traffic_pressure", 1.0))))
    incident_pressure = max(0.6, min(2.4, float(scenario_mod.get("incident_pressure", 1.0))))
    weather_pressure = max(0.6, min(2.1, float(scenario_mod.get("weather_pressure", 1.0))))
    state_pressure = max(0.7, min(2.8, 0.50 * traffic_pressure + 0.30 * incident_pressure + 0.20 * weather_pressure))

    def _transition(prev_node: str | None, current_node: str, next_node: str) -> tuple[tuple[int, str, str], float] | None:
        current_coords = graph.nodes.get(current_node)
        next_coords = graph.nodes.get(next_node)
        if current_coords is None or next_coords is None:
            return None
        heading = _segment_heading_deg(current_coords[0], current_coords[1], next_coords[0], next_coords[1])
        heading_bin = _heading_bin(heading)
        edge = graph.edge_index.get((current_node, next_node))
        highway = str(edge.highway if edge is not None else "unclassified").strip().lower() or "unclassified"

        turn_class = "start"
        turn_penalty = 0.0
        if prev_node is not None and prev_node in graph.nodes:
            prev_coords = graph.nodes.get(prev_node)
            if prev_coords is not None:
                prev_heading = _segment_heading_deg(prev_coords[0], prev_coords[1], current_coords[0], current_coords[1])
                delta = _heading_delta_deg(prev_heading, heading)
                if delta >= 165.0:
                    # U-turn transitions are treated as illegal in strict graph mode.
                    return None
                if delta >= 120.0:
                    turn_class = "hard_turn"
                    turn_penalty = 22.0
                elif delta >= 70.0:
                    turn_class = "medium_turn"
                    turn_penalty = 10.0
                elif delta >= 30.0:
                    turn_class = "soft_turn"
                    turn_penalty = 3.0
                else:
                    turn_class = "straight"
        if highway in {"residential", "unclassified"} and turn_class in {"hard_turn", "medium_turn"}:
            turn_penalty += 2.5
        return (heading_bin, turn_class, highway), max(0.0, turn_penalty * state_pressure)

    return _transition


def _nearest_node_id(graph: RouteGraph, *, lat: float, lon: float) -> str | None:
    center_key = _grid_key(lat, lon)
    best_id: str | None = None
    best_dist = float("inf")
    for radius in range(0, 6):
        found_any = False
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                key = (center_key[0] + dy, center_key[1] + dx)
                for node_id in graph.grid_index.get(key, ()):
                    found_any = True
                    n_lat, n_lon = graph.nodes[node_id]
                    dist = _haversine_m(lat, lon, n_lat, n_lon)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = node_id
        if found_any and best_id is not None:
            break
    return best_id


def _landmarks_for_path(
    graph: RouteGraph,
    path: PathResult,
    *,
    landmarks_per_path: int,
) -> tuple[tuple[float, float], ...]:
    if len(path.nodes) < 3:
        return ()
    picks: list[tuple[float, float]] = []
    count = max(1, landmarks_per_path)
    for idx in range(1, count + 1):
        ratio = idx / (count + 1)
        node_idx = int(round((len(path.nodes) - 1) * ratio))
        node_idx = max(1, min(len(path.nodes) - 2, node_idx))
        node = path.nodes[node_idx]
        lat, lon = graph.nodes[node]
        picks.append((lat, lon))
    return tuple(picks)


def route_graph_via_points(
    *,
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
    max_paths: int | None = None,
) -> tuple[tuple[float, float], ...]:
    via_paths = route_graph_via_paths(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        destination_lat=destination_lat,
        destination_lon=destination_lon,
        max_paths=max_paths,
    )
    flat: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for path in via_paths:
        for lat, lon in path:
            key = (int(round(lat * 20_000)), int(round(lon * 20_000)))
            if key in seen:
                continue
            seen.add(key)
            flat.append((lat, lon))
            if len(flat) >= int(settings.route_candidate_via_budget):
                return tuple(flat)
    return tuple(flat)


def route_graph_via_paths(
    *,
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
    max_paths: int | None = None,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not settings.route_graph_enabled:
        return ()
    graph = load_route_graph()
    if graph is None:
        return ()
    start = _nearest_node_id(graph, lat=origin_lat, lon=origin_lon)
    goal = _nearest_node_id(graph, lat=destination_lat, lon=destination_lon)
    if not start or not goal or start == goal:
        return ()
    paths = yen_k_shortest_paths(
        adjacency=_adjacency_cost_view(graph),
        start=start,
        goal=goal,
        k=max(1, int(max_paths or settings.route_graph_k_paths)),
        max_hops=max(8, int(settings.route_graph_max_hops)),
        max_state_budget=max(1000, int(settings.route_graph_max_state_budget)),
        max_repeat_per_node=max(0, int(settings.route_graph_max_repeat_per_node)),
        max_detour_ratio=max(1.0, float(settings.route_graph_max_detour_ratio)),
        max_candidate_pool=max(16, int(settings.route_graph_k_paths) * 8),
        transition_state_fn=_transition_state_callback(graph),
    )
    out: list[tuple[tuple[float, float], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for path in paths:
        landmarks = _landmarks_for_path(
            graph,
            path,
            landmarks_per_path=max(1, int(settings.route_graph_via_landmarks_per_path)),
        )
        if not landmarks:
            continue
        key = tuple((int(round(lat * 20_000)), int(round(lon * 20_000))) for lat, lon in landmarks)
        if key in seen:
            continue
        seen.add(key)
        out.append(tuple(landmarks))
        if len(out) >= int(settings.route_candidate_via_budget):
            break
    return tuple(out)


def _path_distance_m(graph: RouteGraph, nodes: tuple[str, ...]) -> float:
    total = 0.0
    for idx in range(1, len(nodes)):
        edge = graph.edge_index.get((nodes[idx - 1], nodes[idx]))
        if edge is not None:
            total += max(0.0, float(edge.distance_m))
            continue
        lat1, lon1 = graph.nodes[nodes[idx - 1]]
        lat2, lon2 = graph.nodes[nodes[idx]]
        total += _haversine_m(lat1, lon1, lat2, lon2)
    return max(0.0, total)


def _segment_classes_from_highway(highway: str, *, toll: bool) -> list[str]:
    out: list[str] = []
    hw = (highway or "").strip().lower()
    if hw:
        out.append(hw)
    if toll:
        out.append("toll")
    return out or ["unclassified"]


def _edge_speed_mps(edge: GraphEdge | None) -> float:
    if edge is not None and edge.maxspeed_kph is not None and edge.maxspeed_kph > 0:
        # Conservative effective speed for routing realism under mixed traffic.
        return max(4.0, min(42.0, (float(edge.maxspeed_kph) / 3.6) * 0.82))
    defaults_kph = {
        "motorway": 90.0,
        "motorway_link": 58.0,
        "trunk": 78.0,
        "trunk_link": 52.0,
        "primary": 64.0,
        "primary_link": 48.0,
        "secondary": 54.0,
        "secondary_link": 42.0,
        "tertiary": 44.0,
        "tertiary_link": 36.0,
        "unclassified": 34.0,
        "residential": 28.0,
    }
    highway = edge.highway if edge is not None else "unclassified"
    return max(4.0, min(40.0, defaults_kph.get(highway, 36.0) / 3.6))


def _turn_burden_penalty(graph: RouteGraph, nodes: tuple[str, ...]) -> float:
    if len(nodes) < 3:
        return 0.0
    burden = 0.0
    for idx in range(2, len(nodes)):
        lat0, lon0 = graph.nodes[nodes[idx - 2]]
        lat1, lon1 = graph.nodes[nodes[idx - 1]]
        lat2, lon2 = graph.nodes[nodes[idx]]
        h1 = _segment_heading_deg(lat0, lon0, lat1, lon1)
        h2 = _segment_heading_deg(lat1, lon1, lat2, lon2)
        delta = abs((h2 - h1 + 180.0) % 360.0 - 180.0)
        burden += min(1.0, delta / 120.0)
    return burden


def route_graph_candidate_routes(
    *,
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
    max_paths: int | None = None,
    scenario_edge_modifiers: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
    budget = max(1, int(max_paths or settings.route_graph_k_paths))
    graph = load_route_graph()
    if graph is None:
        return [], GraphCandidateDiagnostics(
            explored_states=0,
            generated_paths=0,
            emitted_paths=0,
            candidate_budget=budget,
        )
    start = _nearest_node_id(graph, lat=origin_lat, lon=origin_lon)
    goal = _nearest_node_id(graph, lat=destination_lat, lon=destination_lon)
    if not start or not goal or start == goal:
        return [], GraphCandidateDiagnostics(
            explored_states=0,
            generated_paths=0,
            emitted_paths=0,
            candidate_budget=budget,
        )
    paths, stats = yen_k_shortest_paths_with_stats(
        adjacency=_adjacency_cost_view(graph, scenario_edge_modifiers=scenario_edge_modifiers),
        start=start,
        goal=goal,
        k=budget,
        max_hops=max(8, int(settings.route_graph_max_hops)),
        max_state_budget=max(1000, int(settings.route_graph_max_state_budget)),
        max_repeat_per_node=max(0, int(settings.route_graph_max_repeat_per_node)),
        max_detour_ratio=max(1.0, float(settings.route_graph_max_detour_ratio)),
        max_candidate_pool=max(16, int(settings.route_graph_k_paths) * 8),
        transition_state_fn=_transition_state_callback(graph, scenario_edge_modifiers=scenario_edge_modifiers),
    )
    out: list[dict[str, Any]] = []
    scenario_mod = scenario_edge_modifiers or {}
    traffic_pressure = max(0.6, min(2.2, float(scenario_mod.get("traffic_pressure", 1.0))))
    incident_pressure = max(0.6, min(2.4, float(scenario_mod.get("incident_pressure", 1.0))))
    weather_pressure = max(0.6, min(2.1, float(scenario_mod.get("weather_pressure", 1.0))))
    duration_multiplier = max(0.6, min(2.5, float(scenario_mod.get("duration_multiplier", 1.0))))
    incident_rate_multiplier = max(0.5, min(2.8, float(scenario_mod.get("incident_rate_multiplier", 1.0))))
    incident_delay_multiplier = max(0.5, min(2.8, float(scenario_mod.get("incident_delay_multiplier", 1.0))))
    sigma_multiplier = max(0.4, min(2.8, float(scenario_mod.get("stochastic_sigma_multiplier", 1.0))))
    weather_regime_factor = max(0.75, min(1.75, float(scenario_mod.get("weather_regime_factor", 1.0))))
    hour_bucket_factor = max(0.75, min(1.75, float(scenario_mod.get("hour_bucket_factor", 1.0))))
    road_class_factors_raw = scenario_mod.get("road_class_factors", {})
    road_class_factors: dict[str, float] = {}
    if isinstance(road_class_factors_raw, dict):
        for key, value in road_class_factors_raw.items():
            normalized_key = str(key).strip().lower()
            if not normalized_key:
                continue
            try:
                road_class_factors[normalized_key] = max(0.70, min(1.85, float(value)))
            except (TypeError, ValueError):
                continue
    scenario_edge_scaling_version = str(scenario_mod.get("scenario_edge_scaling_version", "v3_live_transform"))
    road_sensitivity = {
        "motorway": 0.75,
        "motorway_link": 0.95,
        "trunk": 0.85,
        "trunk_link": 1.05,
        "primary": 1.10,
        "primary_link": 1.15,
        "secondary": 1.22,
        "secondary_link": 1.30,
        "tertiary": 1.35,
        "tertiary_link": 1.42,
        "unclassified": 1.46,
        "residential": 1.55,
    }
    for idx, path in enumerate(paths, start=1):
        if len(path.nodes) < 2:
            continue
        coords = []
        for node_id in path.nodes:
            lat, lon = graph.nodes[node_id]
            coords.append([float(lon), float(lat)])
        if len(coords) < 2:
            continue
        distance_m = _path_distance_m(graph, path.nodes)
        toll_edges = 0
        road_mix_counts: dict[str, int] = {}
        base_segment_durations_s: list[float] = []
        base_segment_distances_m: list[float] = []
        seg_classes: list[list[str]] = []
        for seg_idx in range(1, len(path.nodes)):
            edge = graph.edge_index.get((path.nodes[seg_idx - 1], path.nodes[seg_idx]))
            lat1, lon1 = graph.nodes[path.nodes[seg_idx - 1]]
            lat2, lon2 = graph.nodes[path.nodes[seg_idx]]
            seg_m = max(
                1.0,
                float(edge.distance_m) if edge is not None else _haversine_m(lat1, lon1, lat2, lon2),
            )
            seg_speed_mps = _edge_speed_mps(edge)
            base_segment_distances_m.append(seg_m)
            edge_highway = edge.highway if edge is not None else "unclassified"
            sensitivity = road_sensitivity.get(edge_highway, 1.25)
            road_factor = road_class_factors.get(edge_highway, 1.0)
            edge_time_scale = max(
                0.55,
                min(
                    2.80,
                    1.0
                    + ((duration_multiplier - 1.0) * sensitivity * 0.36)
                    + ((traffic_pressure - 1.0) * sensitivity * 0.24)
                    + ((incident_pressure - 1.0) * sensitivity * 0.12)
                    + ((weather_pressure - 1.0) * sensitivity * 0.08)
                    + ((weather_regime_factor - 1.0) * sensitivity * 0.08)
                    + ((hour_bucket_factor - 1.0) * sensitivity * 0.07)
                    + ((incident_rate_multiplier - 1.0) * sensitivity * 0.025)
                    + ((incident_delay_multiplier - 1.0) * sensitivity * 0.012)
                    + ((sigma_multiplier - 1.0) * sensitivity * 0.005)
                    + ((road_factor - 1.0) * sensitivity * 0.15),
                ),
            )
            base_segment_durations_s.append((seg_m / max(1.0, seg_speed_mps)) * edge_time_scale)
            if edge is None:
                seg_classes.append(["unclassified"])
                continue
            if edge.toll:
                toll_edges += 1
            road_mix_counts[edge.highway] = road_mix_counts.get(edge.highway, 0) + 1
            seg_classes.append(_segment_classes_from_highway(edge.highway, toll=edge.toll))
        turn_penalty = _turn_burden_penalty(graph, path.nodes)
        toll_penalty = min(0.25, 0.02 * toll_edges)
        generalized_ratio = float(path.cost) / max(1.0, distance_m)
        congestion_ratio = max(0.70, min(2.40, generalized_ratio))
        turn_ratio = 1.0 + min(0.35, 0.04 * turn_penalty)
        penalty_ratio = congestion_ratio * turn_ratio * (1.0 + toll_penalty)
        seg_distances = list(base_segment_distances_m)
        seg_durations = [seg_t * penalty_ratio for seg_t in base_segment_durations_s]
        duration_s = sum(seg_durations)
        route = {
            "distance": float(distance_m),
            "duration": float(duration_s),
            "geometry": {"type": "LineString", "coordinates": coords},
            "legs": [
                {
                    "annotation": {
                        "distance": seg_distances,
                        "duration": seg_durations,
                    },
                    "steps": [
                        {
                            "distance": float(seg_distances[i]),
                            "duration": float(seg_durations[i]),
                            "classes": seg_classes[i] if i < len(seg_classes) else ["unclassified"],
                        }
                        for i in range(len(seg_distances))
                    ],
                }
            ],
            "_graph_meta": {
                "path_id": idx,
                "path_nodes": len(path.nodes),
                "path_cost": float(path.cost),
                "toll_edges": int(toll_edges),
                "turn_burden": round(float(turn_penalty), 6),
                "road_mix_counts": road_mix_counts,
                "scenario_edge_modifiers": dict(scenario_edge_modifiers or {}),
                "scenario_edge_scaling_version": scenario_edge_scaling_version,
            },
        }
        out.append(route)
        if len(out) >= int(settings.route_candidate_via_budget):
            break
    return out, GraphCandidateDiagnostics(
        explored_states=int(stats.get("explored_states", 0)),
        generated_paths=max(0, int(stats.get("generated_candidates", 0))) + len(paths),
        emitted_paths=len(out),
        candidate_budget=budget,
    )
