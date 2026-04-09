from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import csv
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional in local dev
    import osmium
except Exception:  # pragma: no cover
    osmium = None  # type: ignore[assignment]


EARTH_RADIUS_M = 6_371_000.0
UK_BBOX = (49.75, 61.10, -8.75, 2.25)  # lat_min, lat_max, lon_min, lon_max
GRAPH_PROGRESS_EVERY_WAYS = max(1, int(os.environ.get("ROUTING_GRAPH_PROGRESS_EVERY_WAYS", "5000")))
ALLOWED_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
}


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


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _km_to_lat_deg(buffer_km: float) -> float:
    return max(0.0, float(buffer_km)) / 111.0


def _km_to_lon_deg(buffer_km: float, *, latitude_deg: float) -> float:
    scale = max(0.1, math.cos(math.radians(max(-89.9, min(89.9, latitude_deg)))))
    return max(0.0, float(buffer_km)) / (111.0 * scale)


def _load_corpus_bbox(
    *,
    corpus_csv: Path,
    buffer_km: float,
) -> tuple[float, float, float, float]:
    latitudes: list[float] = []
    longitudes: list[float] = []
    with corpus_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for lat_key, lon_key in (
                ("origin_lat", "origin_lon"),
                ("destination_lat", "destination_lon"),
            ):
                raw_lat = row.get(lat_key)
                raw_lon = row.get(lon_key)
                if raw_lat in (None, "") or raw_lon in (None, ""):
                    continue
                latitudes.append(float(raw_lat))
                longitudes.append(float(raw_lon))
    if not latitudes or not longitudes:
        raise RuntimeError(f"No OD coordinates found in corpus csv: {corpus_csv}")
    lat_min = min(latitudes)
    lat_max = max(latitudes)
    lon_min = min(longitudes)
    lon_max = max(longitudes)
    center_lat = (lat_min + lat_max) / 2.0
    lat_buffer_deg = _km_to_lat_deg(float(buffer_km))
    lon_buffer_deg = _km_to_lon_deg(float(buffer_km), latitude_deg=center_lat)
    derived = (
        lat_min - lat_buffer_deg,
        lat_max + lat_buffer_deg,
        lon_min - lon_buffer_deg,
        lon_max + lon_buffer_deg,
    )
    return derived


def _parse_maxspeed(tag: str | None) -> float | None:
    if not tag:
        return None
    raw = str(tag).strip().lower()
    if not raw:
        return None
    parts = raw.split()
    try:
        value = float(parts[0])
    except ValueError:
        return None
    if "mph" in raw:
        return value * 1.60934
    return value


def _highway_penalty(highway: str) -> float:
    penalties = {
        "motorway": 1.00,
        "motorway_link": 1.02,
        "trunk": 1.04,
        "trunk_link": 1.05,
        "primary": 1.08,
        "primary_link": 1.10,
        "secondary": 1.13,
        "secondary_link": 1.15,
        "tertiary": 1.18,
        "tertiary_link": 1.20,
        "unclassified": 1.28,
        "residential": 1.35,
    }
    return penalties.get(highway, 1.25)


def _edge_weight(distance_m: float, *, highway: str, maxspeed_kph: float | None) -> float:
    if distance_m <= 0.0:
        return 1.0
    speed_penalty = 1.0
    if maxspeed_kph is not None and maxspeed_kph > 0:
        speed_penalty = max(0.55, min(1.5, 50.0 / maxspeed_kph))
    return max(1.0, distance_m * _highway_penalty(highway) * speed_penalty)


def _is_toll_from_tags(tags: dict[str, str]) -> bool:
    toll_tag = tags.get("toll", "").strip().lower()
    barrier = tags.get("barrier", "").strip().lower()
    highway = tags.get("highway", "").strip().lower()
    return toll_tag in {"yes", "all", "hgv"} or barrier == "toll_booth" or highway == "toll_gantry"


def _oneway_from_tags(tags: dict[str, str]) -> bool:
    raw = tags.get("oneway", "").strip().lower()
    return raw in {"yes", "true", "1", "forward"}


def _direction_override(tags: dict[str, str]) -> str:
    raw = tags.get("oneway", "").strip().lower()
    if raw in {"-1", "reverse", "backward"}:
        return "reverse"
    return "forward"


def _extract_from_geojson(
    *,
    source: Path,
    bbox: tuple[float, float, float, float],
) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    features = payload.get("features", []) if isinstance(payload, dict) else []
    nodes: dict[str, tuple[float, float]] = {}
    edges: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        if not isinstance(props, dict):
            props = {}
        if not isinstance(geom, dict) or str(geom.get("type", "")).lower() != "linestring":
            continue
        coords = geom.get("coordinates", [])
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        points: list[tuple[float, float]] = []
        for coord in coords:
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            lon = float(coord[0])
            lat = float(coord[1])
            if _in_bbox(lat, lon, bbox):
                points.append((lat, lon))
        if len(points) < 2:
            continue
        for idx in range(1, len(points)):
            lat1, lon1 = points[idx - 1]
            lat2, lon2 = points[idx]
            d_m = _haversine_m(lat1, lon1, lat2, lon2)
            if d_m <= 0.5:
                continue
            u = f"g_{round(lat1, 6)}_{round(lon1, 6)}"
            v = f"g_{round(lat2, 6)}_{round(lon2, 6)}"
            nodes[u] = (lat1, lon1)
            nodes[v] = (lat2, lon2)
            highway = str(props.get("highway", props.get("road_class", "primary"))).strip().lower() or "primary"
            generalized_cost = _edge_weight(d_m, highway=highway, maxspeed_kph=None)
            edges.append(
                {
                    "u": u,
                    "v": v,
                    "distance_m": d_m,
                    "generalized_cost": generalized_cost,
                    "oneway": False,
                    "highway": highway,
                    "toll": bool(str(props.get("toll", "")).strip().lower() in {"yes", "all", "hgv"}),
                    "maxspeed_kph": None,
                }
            )
    return nodes, edges


def _extract_from_pbf(
    *,
    source: Path,
    bbox: tuple[float, float, float, float],
    max_ways: int,
) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    print(
        f"[routing_graph] extracting from {source} (max_ways={max_ways}, progress_every={GRAPH_PROGRESS_EVERY_WAYS})",
        flush=True,
    )
    if osmium is None:
        if source.suffix.lower() == ".osm":
            return _extract_from_osm_xml(source=source, bbox=bbox, max_ways=max_ways)
        raise RuntimeError("pyosmium is required to build routing graph from .pbf inputs.")

    class _GraphHandler(osmium.SimpleHandler):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.nodes: dict[str, tuple[float, float]] = {}
            self.edges: list[dict[str, Any]] = []
            self.ways_seen = 0
            self._last_report = 0

        def way(self, w: Any) -> None:
            if max_ways > 0 and self.ways_seen >= max_ways:
                return
            tags = {str(k): str(v) for k, v in w.tags}
            highway = tags.get("highway", "").strip().lower()
            if highway not in ALLOWED_HIGHWAYS:
                return
            raw_points: list[tuple[str, float, float]] = []
            for n in w.nodes:
                try:
                    lon = float(n.lon)
                    lat = float(n.lat)
                except Exception:
                    continue
                if not _in_bbox(lat, lon, bbox):
                    continue
                raw_points.append((str(n.ref), lat, lon))
            if len(raw_points) < 2:
                return
            self.ways_seen += 1
            if self.ways_seen - self._last_report >= GRAPH_PROGRESS_EVERY_WAYS:
                self._last_report = self.ways_seen
                print(
                    "[routing_graph] progress "
                    f"ways={self.ways_seen} nodes={len(self.nodes)} edges={len(self.edges)}",
                    flush=True,
                )
            oneway = _oneway_from_tags(tags)
            direction = _direction_override(tags)
            maxspeed_kph = _parse_maxspeed(tags.get("maxspeed"))
            toll = _is_toll_from_tags(tags)
            for idx in range(1, len(raw_points)):
                n1, lat1, lon1 = raw_points[idx - 1]
                n2, lat2, lon2 = raw_points[idx]
                d_m = _haversine_m(lat1, lon1, lat2, lon2)
                if d_m <= 0.5:
                    continue
                self.nodes[n1] = (lat1, lon1)
                self.nodes[n2] = (lat2, lon2)
                weight = _edge_weight(d_m, highway=highway, maxspeed_kph=maxspeed_kph)
                u, v = (n2, n1) if direction == "reverse" else (n1, n2)
                self.edges.append(
                    {
                        "u": u,
                        "v": v,
                        "distance_m": d_m,
                        "generalized_cost": weight,
                        "oneway": oneway,
                        "highway": highway,
                        "toll": toll,
                        "maxspeed_kph": maxspeed_kph,
                    }
                )

    handler = _GraphHandler()
    try:
        handler.apply_file(str(source), locations=True, idx="sparse_mem_array")
    except TypeError:
        # Fallback for older pyosmium builds without explicit idx support.
        handler.apply_file(str(source), locations=True)
    print(
        f"[routing_graph] extraction complete ways={handler.ways_seen} nodes={len(handler.nodes)} edges={len(handler.edges)}",
        flush=True,
    )
    return handler.nodes, handler.edges


def _extract_from_osm_xml(
    *,
    source: Path,
    bbox: tuple[float, float, float, float],
    max_ways: int,
) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    print(
        f"[routing_graph] extracting from XML {source} (max_ways={max_ways}, progress_every={GRAPH_PROGRESS_EVERY_WAYS})",
        flush=True,
    )
    tree = ET.parse(source)
    root = tree.getroot()
    node_index: dict[str, tuple[float, float]] = {}
    for node in root.findall("node"):
        node_id = node.attrib.get("id")
        if not node_id:
            continue
        try:
            lat = float(node.attrib.get("lat", "nan"))
            lon = float(node.attrib.get("lon", "nan"))
        except ValueError:
            continue
        node_index[node_id] = (lat, lon)

    nodes: dict[str, tuple[float, float]] = {}
    edges: list[dict[str, Any]] = []
    ways_seen = 0
    for way in root.findall("way"):
        if max_ways > 0 and ways_seen >= max_ways:
            break
        tags: dict[str, str] = {}
        refs: list[str] = []
        for child in list(way):
            if child.tag == "tag":
                key = str(child.attrib.get("k", "")).strip()
                value = str(child.attrib.get("v", "")).strip()
                if key:
                    tags[key] = value
            elif child.tag == "nd":
                ref = str(child.attrib.get("ref", "")).strip()
                if ref:
                    refs.append(ref)
        highway = tags.get("highway", "").strip().lower()
        if highway not in ALLOWED_HIGHWAYS:
            continue
        raw_points: list[tuple[str, float, float]] = []
        for ref in refs:
            coord = node_index.get(ref)
            if coord is None:
                continue
            lat, lon = coord
            if not _in_bbox(lat, lon, bbox):
                continue
            raw_points.append((ref, lat, lon))
        if len(raw_points) < 2:
            continue
        ways_seen += 1
        if ways_seen % GRAPH_PROGRESS_EVERY_WAYS == 0:
            print(
                "[routing_graph] progress "
                f"ways={ways_seen} nodes={len(nodes)} edges={len(edges)}",
                flush=True,
            )
        oneway = _oneway_from_tags(tags)
        direction = _direction_override(tags)
        maxspeed_kph = _parse_maxspeed(tags.get("maxspeed"))
        toll = _is_toll_from_tags(tags)
        for idx in range(1, len(raw_points)):
            n1, lat1, lon1 = raw_points[idx - 1]
            n2, lat2, lon2 = raw_points[idx]
            d_m = _haversine_m(lat1, lon1, lat2, lon2)
            if d_m <= 0.5:
                continue
            nodes[n1] = (lat1, lon1)
            nodes[n2] = (lat2, lon2)
            weight = _edge_weight(d_m, highway=highway, maxspeed_kph=maxspeed_kph)
            u, v = (n2, n1) if direction == "reverse" else (n1, n2)
            edges.append(
                {
                    "u": u,
                    "v": v,
                    "distance_m": d_m,
                    "generalized_cost": weight,
                    "oneway": oneway,
                    "highway": highway,
                    "toll": toll,
                    "maxspeed_kph": maxspeed_kph,
                }
            )
    print(
        f"[routing_graph] extraction complete ways={ways_seen} nodes={len(nodes)} edges={len(edges)}",
        flush=True,
    )
    return nodes, edges


def _component_index_from_edges(
    *,
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[int, int], int, int, float]:
    parent: dict[str, str] = {}
    rank: dict[str, int] = {}

    def _find(node_id: str) -> str:
        root = parent.get(node_id)
        if root is None:
            parent[node_id] = node_id
            rank[node_id] = 0
            return node_id
        path: list[str] = []
        while True:
            next_root = parent.get(root)
            if next_root is None or next_root == root:
                break
            path.append(root)
            root = next_root
        parent[node_id] = root
        for item in path:
            parent[item] = root
        return root

    def _union(a: str, b: str) -> None:
        root_a = _find(a)
        root_b = _find(b)
        if root_a == root_b:
            return
        rank_a = int(rank.get(root_a, 0))
        rank_b = int(rank.get(root_b, 0))
        if rank_a < rank_b:
            root_a, root_b = root_b, root_a
            rank_a, rank_b = rank_b, rank_a
        parent[root_b] = root_a
        if rank_a == rank_b:
            rank[root_a] = rank_a + 1

    for node_id in nodes:
        _find(node_id)
    for edge in edges:
        u = str(edge.get("u", "")).strip()
        v = str(edge.get("v", "")).strip()
        if u in nodes and v in nodes:
            _union(u, v)

    component_by_node: dict[str, int] = {}
    component_sizes: dict[int, int] = {}
    root_to_component: dict[str, int] = {}
    component_idx = 0
    for node_id in nodes:
        root = _find(node_id)
        component_id = root_to_component.get(root)
        if component_id is None:
            component_idx += 1
            component_id = component_idx
            root_to_component[root] = component_id
            component_sizes[component_id] = 0
        component_by_node[node_id] = component_id
        component_sizes[component_id] = int(component_sizes[component_id]) + 1
    largest_component_nodes = max(component_sizes.values(), default=0)
    largest_component_ratio = (
        float(largest_component_nodes) / float(max(1, len(nodes)))
        if nodes
        else 0.0
    )
    return (
        component_by_node,
        component_sizes,
        int(component_idx),
        int(largest_component_nodes),
        float(largest_component_ratio),
    )


def _grid_index_from_nodes(nodes: dict[str, tuple[float, float]]) -> dict[tuple[int, int], tuple[str, ...]]:
    grid_mut: dict[tuple[int, int], list[str]] = {}
    for node_id, (lat, lon) in nodes.items():
        key = (int(math.floor(lat / 0.15)), int(math.floor(lon / 0.15)))
        grid_mut.setdefault(key, []).append(node_id)
    return {key: tuple(values) for key, values in grid_mut.items()}


def _compact_edge_record(edge: dict[str, Any]) -> tuple[str, str, float, float, bool, str, bool, float | None]:
    maxspeed_raw = edge.get("maxspeed_kph")
    maxspeed_kph = None
    if maxspeed_raw is not None:
        try:
            maxspeed_kph = float(maxspeed_raw)
        except (TypeError, ValueError):
            maxspeed_kph = None
    return (
        str(edge.get("u", "")),
        str(edge.get("v", "")),
        float(edge.get("distance_m", 1.0)),
        float(edge.get("generalized_cost", edge.get("distance_m", 1.0))),
        bool(edge.get("oneway", False)),
        str(edge.get("highway", "unclassified")).strip().lower() or "unclassified",
        bool(edge.get("toll", False)),
        maxspeed_kph,
    )


def _write_compact_bundle(
    *,
    graph_output: Path,
    bundle_output: Path,
    source: Path,
    generated_at_utc: str,
    as_of_utc: str,
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, Any]],
    min_giant_component_nodes: int,
    min_giant_component_ratio: float,
) -> dict[str, Any]:
    component_by_node, component_sizes, component_count, largest_component_nodes, largest_component_ratio = (
        _component_index_from_edges(nodes=nodes, edges=edges)
    )
    grid_index = _grid_index_from_nodes(nodes)
    adjacency_mut: dict[str, list[tuple[str, float, float, str, bool, float | None]]] = {}
    min_cost_per_meter = 0.0
    for raw_edge in edges:
        u, v, distance_m, generalized_cost, oneway, highway, toll, maxspeed_kph = _compact_edge_record(raw_edge)
        if u not in nodes or v not in nodes:
            continue
        adjacency_mut.setdefault(u, []).append((v, generalized_cost, distance_m, highway, toll, maxspeed_kph))
        if distance_m > 0.0 and generalized_cost > 0.0:
            ratio = float(generalized_cost) / float(distance_m)
            if min_cost_per_meter <= 0.0 or ratio < min_cost_per_meter:
                min_cost_per_meter = max(0.0, ratio)
        if not oneway:
            adjacency_mut.setdefault(v, []).append((u, generalized_cost, distance_m, highway, toll, maxspeed_kph))
    graph_fragmented = bool(
        largest_component_nodes < max(1, int(min_giant_component_nodes))
        or largest_component_ratio < max(0.0, min(1.0, float(min_giant_component_ratio)))
    )
    bundle_payload = {
        "asset": {
            "path": str(graph_output.resolve()),
            "size": int(graph_output.stat().st_size),
            "mtime_ns": int(getattr(graph_output.stat(), "st_mtime_ns", int(graph_output.stat().st_mtime * 1_000_000_000))),
            "version": "uk-routing-graph-v1",
            "source": str(source),
        },
        "saved_at_utc": generated_at_utc,
        "graph": {
            "version": "uk-routing-graph-v1",
            "source": str(source),
            "generated_at_utc": generated_at_utc,
            "as_of_utc": as_of_utc,
            "nodes": nodes,
            "adjacency": {node: tuple(edges_for_node) for node, edges_for_node in adjacency_mut.items()},
            "grid_index": grid_index,
            "component_by_node": component_by_node,
            "component_sizes": component_sizes,
            "component_count": component_count,
            "largest_component_nodes": largest_component_nodes,
            "largest_component_ratio": largest_component_ratio,
            "graph_fragmented": graph_fragmented,
            "min_cost_per_meter": min_cost_per_meter,
            "edge_count": sum(len(values) for values in adjacency_mut.values()),
        },
    }
    bundle_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = bundle_output.with_suffix(f"{bundle_output.suffix}.tmp")
    with tmp_path.open("wb") as fh:
        pickle.dump(bundle_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(bundle_output)
    with bundle_output.with_suffix(".meta.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "asset": bundle_payload["asset"],
                "saved_at_utc": generated_at_utc,
                "graph": {
                    "version": "uk-routing-graph-v1",
                    "source": str(source),
                    "edge_count": bundle_payload["graph"]["edge_count"],
                    "component_count": component_count,
                },
            },
            fh,
            indent=2,
        )
    return {
        "compact_bundle": str(bundle_output),
        "compact_bundle_meta": str(bundle_output.with_suffix(".meta.json")),
        "compact_bundle_nodes": int(len(nodes)),
        "compact_bundle_edges": int(bundle_payload["graph"]["edge_count"]),
    }


def _write_graph_json_streaming(
    *,
    output: Path,
    source: Path,
    generated_at_utc: str,
    as_of_utc: str,
    bbox: tuple[float, float, float, float],
    nodes: dict[str, tuple[float, float]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    bbox_payload = {
        "lat_min": bbox[0],
        "lat_max": bbox[1],
        "lon_min": bbox[2],
        "lon_max": bbox[3],
    }
    tmp_path = output.with_suffix(f"{output.suffix}.tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write("{")
        fh.write('"version":')
        json.dump("uk-routing-graph-v1", fh, separators=(",", ":"))
        fh.write(',"source":')
        json.dump(str(source), fh, separators=(",", ":"))
        fh.write(',"generated_at_utc":')
        json.dump(generated_at_utc, fh, separators=(",", ":"))
        fh.write(',"as_of_utc":')
        json.dump(as_of_utc, fh, separators=(",", ":"))
        fh.write(',"bbox":')
        json.dump(bbox_payload, fh, separators=(",", ":"))
        fh.write(',"nodes":[')
        first_node = True
        for node_id, (lat, lon) in nodes.items():
            if not first_node:
                fh.write(",")
            json.dump(
                {"id": node_id, "lat": lat, "lon": lon},
                fh,
                separators=(",", ":"),
            )
            first_node = False
        fh.write('],"edges":[')
        first_edge = True
        for edge in edges:
            if not first_edge:
                fh.write(",")
            json.dump(edge, fh, separators=(",", ":"))
            first_edge = False
        fh.write("]}")
    tmp_path.replace(output)
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "version": "uk-routing-graph-v1",
                "source": str(source),
                "generated_at_utc": generated_at_utc,
                "as_of_utc": as_of_utc,
                "nodes": len(nodes),
                "edges": len(edges),
                "bbox": bbox_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "version": "uk-routing-graph-v1",
        "source": str(source),
        "generated_at_utc": generated_at_utc,
        "as_of_utc": as_of_utc,
        "bbox": bbox_payload,
    }


def build(
    *,
    source: Path,
    output: Path,
    bbox: tuple[float, float, float, float] = UK_BBOX,
    max_ways: int = 0,
    compact_bundle_output: Path | None = None,
    compact_bundle_min_giant_component_nodes: int = 1,
    compact_bundle_min_giant_component_ratio: float = 0.0,
) -> dict[str, Any]:
    print(f"[routing_graph] build start source={source} output={output}", flush=True)
    if source.suffix.lower() in {".pbf", ".osm"}:
        nodes, edges = _extract_from_pbf(source=source, bbox=bbox, max_ways=max_ways)
    else:
        nodes, edges = _extract_from_geojson(source=source, bbox=bbox)
    if not nodes or not edges:
        raise RuntimeError("No graph nodes/edges were extracted from source input.")
    print(f"[routing_graph] writing graph payload nodes={len(nodes)} edges={len(edges)}", flush=True)

    generated_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    as_of_utc = datetime.fromtimestamp(source.stat().st_mtime, tz=UTC).isoformat().replace("+00:00", "Z")
    _write_graph_json_streaming(
        output=output,
        source=source,
        generated_at_utc=generated_at_utc,
        as_of_utc=as_of_utc,
        bbox=bbox,
        nodes=nodes,
        edges=edges,
    )
    meta_path = output.with_suffix(".meta.json")
    report = {
        "nodes": len(nodes),
        "edges": len(edges),
        "source": str(source),
        "output": str(output),
        "meta": str(meta_path),
    }
    if compact_bundle_output is not None:
        bundle_report = _write_compact_bundle(
            graph_output=output,
            bundle_output=compact_bundle_output,
            source=source,
            generated_at_utc=generated_at_utc,
            as_of_utc=as_of_utc,
            nodes=nodes,
            edges=edges,
            min_giant_component_nodes=max(1, int(compact_bundle_min_giant_component_nodes)),
            min_giant_component_ratio=max(0.0, float(compact_bundle_min_giant_component_ratio)),
        )
        report.update(bundle_report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build UK routing graph asset from OSM PBF/OSM or GeoJSON.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to source file (.pbf/.osm preferred, GeoJSON fallback supported).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/out/model_assets/routing_graph_uk.json"),
        help="Output graph JSON path.",
    )
    parser.add_argument("--lat-min", type=float, default=UK_BBOX[0])
    parser.add_argument("--lat-max", type=float, default=UK_BBOX[1])
    parser.add_argument("--lon-min", type=float, default=UK_BBOX[2])
    parser.add_argument("--lon-max", type=float, default=UK_BBOX[3])
    parser.add_argument(
        "--od-corpus-csv",
        type=Path,
        default=None,
        help="Optional OD corpus CSV used to derive a tighter bbox from origin/destination coordinates.",
    )
    parser.add_argument(
        "--bbox-buffer-km",
        type=float,
        default=25.0,
        help="Buffer in km applied around a corpus-derived bbox when --od-corpus-csv is used.",
    )
    parser.add_argument(
        "--max-ways",
        type=int,
        default=0,
        help="Optional cap on extracted ways (0 means no cap).",
    )
    parser.add_argument(
        "--compact-bundle-output",
        type=Path,
        default=None,
        help="Optional compact startup bundle path to write alongside the graph JSON.",
    )
    args = parser.parse_args(argv)
    bbox = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    if args.od_corpus_csv is not None:
        bbox = _load_corpus_bbox(
            corpus_csv=args.od_corpus_csv,
            buffer_km=max(0.0, float(args.bbox_buffer_km)),
        )
    report = build(
        source=args.source,
        output=args.output,
        bbox=bbox,
        max_ways=max(0, int(args.max_ways)),
        compact_bundle_output=args.compact_bundle_output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
