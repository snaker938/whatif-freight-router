from __future__ import annotations

import argparse
import json
import math
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
    return handler.nodes, handler.edges


def _extract_from_osm_xml(
    *,
    source: Path,
    bbox: tuple[float, float, float, float],
    max_ways: int,
) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
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
    return nodes, edges


def build(
    *,
    source: Path,
    output: Path,
    bbox: tuple[float, float, float, float] = UK_BBOX,
    max_ways: int = 0,
) -> dict[str, Any]:
    if source.suffix.lower() in {".pbf", ".osm"}:
        nodes, edges = _extract_from_pbf(source=source, bbox=bbox, max_ways=max_ways)
    else:
        nodes, edges = _extract_from_geojson(source=source, bbox=bbox)
    if not nodes or not edges:
        raise RuntimeError("No graph nodes/edges were extracted from source input.")

    generated_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    as_of_utc = datetime.fromtimestamp(source.stat().st_mtime, tz=UTC).isoformat().replace("+00:00", "Z")
    payload = {
        "version": "uk-routing-graph-v1",
        "source": str(source),
        "generated_at_utc": generated_at_utc,
        "as_of_utc": as_of_utc,
        "bbox": {
            "lat_min": bbox[0],
            "lat_max": bbox[1],
            "lon_min": bbox[2],
            "lon_max": bbox[3],
        },
        "nodes": [
            {"id": node_id, "lat": lat, "lon": lon}
            for node_id, (lat, lon) in nodes.items()
        ],
        "edges": edges,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "version": payload["version"],
                "source": str(source),
                "generated_at_utc": generated_at_utc,
                "as_of_utc": as_of_utc,
                "nodes": len(nodes),
                "edges": len(edges),
                "bbox": payload["bbox"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "source": str(source),
        "output": str(output),
        "meta": str(meta_path),
    }


def main() -> None:
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
        "--max-ways",
        type=int,
        default=0,
        help="Optional cap on extracted ways (0 means no cap).",
    )
    args = parser.parse_args()
    bbox = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    report = build(
        source=args.source,
        output=args.output,
        bbox=bbox,
        max_ways=max(0, int(args.max_ways)),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
