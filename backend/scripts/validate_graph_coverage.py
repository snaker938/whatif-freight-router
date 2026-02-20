from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _load_graph(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Routing graph payload is not a JSON object.")
    return payload


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2.0) ** 2) + (math.cos(p1) * math.cos(p2) * (math.sin(dl / 2.0) ** 2))
    return 2.0 * r * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _graph_points(payload: dict[str, Any]) -> list[tuple[float, float]]:
    raw_nodes = payload.get("nodes", [])
    points: list[tuple[float, float]] = []
    if not isinstance(raw_nodes, list):
        return points
    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        try:
            lat = float(item.get("lat", 0.0))
            lon = float(item.get("lon", 0.0))
        except (TypeError, ValueError):
            continue
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            points.append((lat, lon))
    return points


def _fixture_points(fixtures_dir: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not fixtures_dir.exists():
        return points
    for path in sorted(fixtures_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        geometry = payload.get("geometry", {}) if isinstance(payload, dict) else {}
        coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
        if not isinstance(coords, list):
            continue
        for coord in coords:
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            lon = float(coord[0])
            lat = float(coord[1])
            points.append((lat, lon))
    return points


def _nearest_dist_m(points: list[tuple[float, float]], lat: float, lon: float) -> float:
    best = float("inf")
    for p_lat, p_lon in points:
        d = _haversine_m(lat, lon, p_lat, p_lon)
        if d < best:
            best = d
    return best


def _to_xy_m(points: list[tuple[float, float]]) -> tuple[np.ndarray, float]:
    if not points:
        return np.zeros((0, 2), dtype=np.float64), 0.0
    arr = np.asarray(points, dtype=np.float64)
    lat_rad = np.radians(arr[:, 0])
    lon_rad = np.radians(arr[:, 1])
    mean_lat = float(np.mean(lat_rad))
    r = 6_371_000.0
    x = lon_rad * (r * math.cos(mean_lat))
    y = lat_rad * r
    return np.column_stack((x, y)), mean_lat


def validate(
    *,
    graph_path: Path,
    fixtures_dir: Path,
    min_nodes: int,
    min_edges: int,
    max_fixture_dist_m: float,
) -> dict[str, Any]:
    payload = _load_graph(graph_path)
    source = str(payload.get("source", "")).strip().lower()
    if ".pbf" not in source:
        raise RuntimeError("Routing graph source must be OSM PBF-derived in strict mode.")
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    node_count = len(nodes) if isinstance(nodes, list) else 0
    edge_count = len(edges) if isinstance(edges, list) else 0
    if node_count < min_nodes:
        raise RuntimeError(f"Graph node count too low: {node_count} < {min_nodes}")
    if edge_count < min_edges:
        raise RuntimeError(f"Graph edge count too low: {edge_count} < {min_edges}")

    graph_points = _graph_points(payload)
    fixture_points = _fixture_points(fixtures_dir)
    if not graph_points:
        raise RuntimeError("Graph has no valid node coordinates.")
    if not fixture_points:
        raise RuntimeError(f"No fixture route points found in {fixtures_dir}")

    graph_xy, _ = _to_xy_m(graph_points)
    fixture_xy, _ = _to_xy_m(fixture_points)
    if graph_xy.shape[0] == 0 or fixture_xy.shape[0] == 0:
        raise RuntimeError("Coverage validation requires non-empty graph and fixture coordinates.")
    tree = cKDTree(graph_xy)
    distances_m, _indices = tree.query(fixture_xy, k=1, workers=-1)
    worst_fixture_dist = float(np.max(distances_m))
    if worst_fixture_dist > max_fixture_dist_m:
        raise RuntimeError(
            f"Graph coverage check failed: fixture->nearest-node max distance "
            f"{worst_fixture_dist:.2f}m exceeds threshold {max_fixture_dist_m:.2f}m"
        )

    bbox = payload.get("bbox", {})
    return {
        "graph_path": str(graph_path),
        "nodes": node_count,
        "edges": edge_count,
        "worst_fixture_nearest_node_m": round(worst_fixture_dist, 3),
        "bbox": bbox if isinstance(bbox, dict) else {},
        "coverage_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate routing graph coverage against fixture corpus.")
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("backend/out/model_assets/routing_graph_uk.json"),
        help="Graph asset JSON path.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path("backend/tests/fixtures/uk_routes"),
        help="Fixture directory used for nearest-node coverage checks.",
    )
    parser.add_argument("--min-nodes", type=int, default=100000)
    parser.add_argument("--min-edges", type=int, default=100000)
    parser.add_argument(
        "--max-fixture-dist-m",
        type=float,
        default=10_000.0,
        help="Maximum allowed distance from fixture coordinate to nearest graph node.",
    )
    args = parser.parse_args()
    report = validate(
        graph_path=args.graph,
        fixtures_dir=args.fixtures_dir,
        min_nodes=max(1, int(args.min_nodes)),
        min_edges=max(1, int(args.min_edges)),
        max_fixture_dist_m=max(1.0, float(args.max_fixture_dist_m)),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
