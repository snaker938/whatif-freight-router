from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import ijson
except Exception as exc:  # pragma: no cover - dependency boundary
    ijson = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

import scripts.build_routing_graph_uk as build_routing_graph_uk
import app.routing_graph as routing_graph


DEFAULT_CORRIDOR_KM = 35.0


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a route-graph subset around the union of buffered OD corridors "
            "from an evaluation corpus."
        )
    )
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--corpus-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--corridor-km",
        type=float,
        default=DEFAULT_CORRIDOR_KM,
        help=(
            "Maximum distance from the straight-line OD segment kept in the subset. "
            f"Defaults to {DEFAULT_CORRIDOR_KM:.1f} km."
        ),
    )
    parser.add_argument(
        "--buffer-km",
        type=float,
        default=None,
        help="Deprecated alias for --corridor-km. If provided, it overrides --corridor-km.",
    )
    return parser


def _resolve_corridor_km(*, corridor_km: float, buffer_km: float | None) -> float:
    if buffer_km is not None:
        return max(0.0, float(buffer_km))
    return max(0.0, float(corridor_km))


def _bbox_payload_from_tuple(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    lat_min, lat_max, lon_min, lon_max = bbox
    return {
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "lon_min": float(lon_min),
        "lon_max": float(lon_max),
    }


def _bbox_contains(*, lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= float(lat) <= lat_max and lon_min <= float(lon) <= lon_max


def _corridor_bbox_payload(corridors: list[dict[str, float]]) -> dict[str, float]:
    lat_min = min(float(corridor["lat_min"]) for corridor in corridors)
    lat_max = max(float(corridor["lat_max"]) for corridor in corridors)
    lon_min = min(float(corridor["lon_min"]) for corridor in corridors)
    lon_max = max(float(corridor["lon_max"]) for corridor in corridors)
    return {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


def _load_corridors(*, corpus_csv: Path, corridor_km: float) -> list[dict[str, float]]:
    corridors: list[dict[str, float]] = []
    lat_buffer_deg = build_routing_graph_uk._km_to_lat_deg(corridor_km)
    with corpus_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                origin_lat = float(row.get("origin_lat"))
                origin_lon = float(row.get("origin_lon"))
                destination_lat = float(row.get("destination_lat"))
                destination_lon = float(row.get("destination_lon"))
            except (TypeError, ValueError):
                continue
            mid_lat = (origin_lat + destination_lat) / 2.0
            lon_buffer_deg = max(
                build_routing_graph_uk._km_to_lon_deg(corridor_km, latitude_deg=origin_lat),
                build_routing_graph_uk._km_to_lon_deg(corridor_km, latitude_deg=destination_lat),
                build_routing_graph_uk._km_to_lon_deg(corridor_km, latitude_deg=mid_lat),
            )
            lon_scale = 111_000.0 * max(0.1, math.cos(math.radians(max(-89.9, min(89.9, mid_lat)))))
            lat_scale = 111_000.0
            x1 = origin_lon * lon_scale
            y1 = origin_lat * lat_scale
            x2 = destination_lon * lon_scale
            y2 = destination_lat * lat_scale
            dx = x2 - x1
            dy = y2 - y1
            corridors.append(
                {
                    "origin_lat": origin_lat,
                    "origin_lon": origin_lon,
                    "destination_lat": destination_lat,
                    "destination_lon": destination_lon,
                    "lat_scale": lat_scale,
                    "lon_scale": lon_scale,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "dx": dx,
                    "dy": dy,
                    "length_sq": dx * dx + dy * dy,
                    "lat_min": min(origin_lat, destination_lat) - lat_buffer_deg,
                    "lat_max": max(origin_lat, destination_lat) + lat_buffer_deg,
                    "lon_min": min(origin_lon, destination_lon) - lon_buffer_deg,
                    "lon_max": max(origin_lon, destination_lon) + lon_buffer_deg,
                }
            )
    return corridors


def _point_to_corridor_distance_m(*, lat: float, lon: float, corridor: dict[str, float]) -> float:
    x = float(lon) * float(corridor["lon_scale"])
    y = float(lat) * float(corridor["lat_scale"])
    x1 = float(corridor["x1"])
    y1 = float(corridor["y1"])
    dx = float(corridor["dx"])
    dy = float(corridor["dy"])
    length_sq = float(corridor["length_sq"])
    if length_sq <= 1e-9:
        return math.hypot(x - x1, y - y1)
    t = ((x - x1) * dx + (y - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(x - proj_x, y - proj_y)


def _node_within_any_corridor(
    *,
    lat: float,
    lon: float,
    corridors: list[dict[str, float]],
    corridor_km: float,
) -> bool:
    corridor_m = max(0.0, float(corridor_km)) * 1000.0
    for corridor in corridors:
        corridor_bbox = (
            float(corridor["lat_min"]),
            float(corridor["lat_max"]),
            float(corridor["lon_min"]),
            float(corridor["lon_max"]),
        )
        if not _bbox_contains(lat=lat, lon=lon, bbox=corridor_bbox):
            continue
        if _point_to_corridor_distance_m(lat=lat, lon=lon, corridor=corridor) <= corridor_m:
            return True
    return False


def _top_level_item(path: Path, prefix: str, default: Any) -> Any:
    if ijson is None:
        return default
    with path.open("rb") as handle:
        try:
            return next(ijson.items(handle, prefix), default)
        except Exception:
            return default


def _parse_node(raw: Any) -> tuple[str, float, float] | None:
    if not isinstance(raw, dict):
        return None
    node_id = str(raw.get("id", "")).strip()
    if not node_id:
        return None
    try:
        lat = float(raw.get("lat"))
        lon = float(raw.get("lon"))
    except (TypeError, ValueError):
        return None
    return node_id, lat, lon


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _emit_compact_bundle_for_subset(output_json: Path) -> dict[str, Any]:
    if not bool(routing_graph.settings.route_graph_compact_bundle_enabled):
        return {}
    graph = routing_graph._load_route_graph_streaming(path=output_json)
    if graph is None:
        return {}
    routing_graph._save_route_graph_binary_cache(output_json, graph)
    routing_graph._save_route_graph_compact_bundle(output_json, graph)
    binary_cache_path = routing_graph._graph_binary_cache_path(output_json)
    binary_cache_meta_path = routing_graph._graph_binary_cache_meta_path(output_json)
    bundle_path = routing_graph._graph_compact_bundle_path(output_json)
    meta_path = routing_graph._graph_compact_bundle_meta_path(output_json)
    if (
        not binary_cache_path.exists()
        or not binary_cache_meta_path.exists()
        or not bundle_path.exists()
        or not meta_path.exists()
    ):
        return {}
    edge_count = sum(len(edges) for edges in graph.adjacency.values())
    return {
        "binary_cache": str(binary_cache_path),
        "binary_cache_meta": str(binary_cache_meta_path),
        "compact_bundle": str(bundle_path),
        "compact_bundle_meta": str(meta_path),
        "compact_bundle_nodes": int(len(graph.nodes)),
        "compact_bundle_edges": int(edge_count),
    }


def _load_staged_nodes(
    node_staging_path: Path,
    *,
    include_grid_index: bool = True,
) -> tuple[dict[str, tuple[float, float]], dict[tuple[int, int], tuple[str, ...]]]:
    nodes: dict[str, tuple[float, float]] = {}
    grid_mut: dict[tuple[int, int], list[str]] | None = {} if include_grid_index else None
    with node_staging_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                raw = json.loads(payload)
            except json.JSONDecodeError:
                continue
            parsed = _parse_node(raw)
            if parsed is None:
                continue
            node_id, lat, lon = parsed
            nodes[node_id] = (lat, lon)
            if grid_mut is not None:
                key = (int(math.floor(lat / 0.15)), int(math.floor(lon / 0.15)))
                grid_mut.setdefault(key, []).append(node_id)
    if grid_mut is None:
        return nodes, {}
    return nodes, {key: tuple(values) for key, values in grid_mut.items()}


def _component_index_from_adjacency(
    *,
    nodes: dict[str, tuple[float, float]],
    adjacency_mut: dict[str, list[tuple[str, float, float, str, bool, float | None]]],
) -> tuple[dict[str, int], dict[int, int], int, int, float]:
    component_by_node: dict[str, int] = {}
    component_sizes: dict[int, int] = {}
    component_idx = 0
    for node_id in nodes:
        if node_id in component_by_node:
            continue
        component_idx += 1
        size = 0
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in component_by_node:
                continue
            component_by_node[current] = component_idx
            size += 1
            for edge in adjacency_mut.get(current, ()):
                neighbor = str(edge[0])
                if neighbor not in component_by_node:
                    stack.append(neighbor)
        component_sizes[component_idx] = size
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


def _emit_compact_bundle_for_subset_from_staging(
    *,
    graph_json: Path,
    output_json: Path,
    node_staging_path: Path,
    version: str,
    source_label: str,
    generated_at_utc: str,
    as_of_utc: str,
) -> dict[str, Any]:
    if not bool(routing_graph.settings.route_graph_compact_bundle_enabled):
        return {}
    if ijson is None:
        return {}
    if not output_json.exists() or not node_staging_path.exists():
        return {}

    nodes, _ = _load_staged_nodes(node_staging_path, include_grid_index=False)
    if not nodes:
        return {}

    adjacency_mut: dict[str, list[routing_graph.GraphEdge]] = {}
    min_cost_per_meter = 0.0
    with graph_json.open("rb") as source_handle:
        for raw_edge in ijson.items(source_handle, "edges.item"):
            if not isinstance(raw_edge, dict):
                continue
            u, v, distance_m, generalized_cost, oneway, highway, toll, maxspeed_kph = (
                build_routing_graph_uk._compact_edge_record(raw_edge)
            )
            if u not in nodes or v not in nodes:
                continue
            adjacency_mut.setdefault(u, []).append(
                routing_graph.GraphEdge(
                    to=v,
                    cost=float(generalized_cost),
                    distance_m=float(distance_m),
                    highway=str(highway),
                    toll=bool(toll),
                    maxspeed_kph=None if maxspeed_kph is None else float(maxspeed_kph),
                )
            )
            if distance_m > 0.0 and generalized_cost > 0.0:
                ratio = float(generalized_cost) / float(distance_m)
                if min_cost_per_meter <= 0.0 or ratio < min_cost_per_meter:
                    min_cost_per_meter = max(0.0, ratio)
            if not oneway:
                adjacency_mut.setdefault(v, []).append(
                    routing_graph.GraphEdge(
                        to=u,
                        cost=float(generalized_cost),
                        distance_m=float(distance_m),
                        highway=str(highway),
                        toll=bool(toll),
                        maxspeed_kph=None if maxspeed_kph is None else float(maxspeed_kph),
                    )
                )

    graph = routing_graph._finalize_graph(
        version=str(version or "uk-routing-graph-v1"),
        source=str(source_label),
        nodes=nodes,
        adjacency_mut=adjacency_mut,
        edge_index={},
        min_cost_per_meter_hint=min_cost_per_meter,
    )
    if graph is None:
        return {}
    routing_graph._save_route_graph_binary_cache(output_json, graph)
    routing_graph._save_route_graph_compact_bundle(output_json, graph)

    binary_cache_path = routing_graph._graph_binary_cache_path(output_json)
    binary_cache_meta_path = routing_graph._graph_binary_cache_meta_path(output_json)
    bundle_path = routing_graph._graph_compact_bundle_path(output_json)
    meta_path = routing_graph._graph_compact_bundle_meta_path(output_json)
    if (
        not binary_cache_path.exists()
        or not binary_cache_meta_path.exists()
        or not bundle_path.exists()
        or not meta_path.exists()
    ):
        return {}
    edge_count = sum(len(edges) for edges in graph.adjacency.values())
    return {
        "binary_cache": str(binary_cache_path),
        "binary_cache_meta": str(binary_cache_meta_path),
        "compact_bundle": str(bundle_path),
        "compact_bundle_meta": str(meta_path),
        "compact_bundle_nodes": int(len(graph.nodes)),
        "compact_bundle_edges": int(edge_count),
    }


def _source_header(path: Path) -> dict[str, Any]:
    return {
        "version": _top_level_item(path, "version", "uk-routing-graph-v1"),
        "source": _top_level_item(path, "source", str(path)),
        "generated_at_utc": _top_level_item(path, "generated_at_utc", None),
        "as_of_utc": _top_level_item(path, "as_of_utc", None),
        "bbox": _top_level_item(path, "bbox", None),
    }


def _init_membership_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("CREATE TABLE kept_nodes (id TEXT PRIMARY KEY) WITHOUT ROWID")
    return connection


def _flush_membership_batch(
    connection: sqlite3.Connection,
    pending_rows: list[tuple[str]],
) -> None:
    if not pending_rows:
        return
    connection.executemany("INSERT OR IGNORE INTO kept_nodes (id) VALUES (?)", pending_rows)
    pending_rows.clear()


def _membership_contains(cursor: sqlite3.Cursor, node_id: str) -> bool:
    row = cursor.execute("SELECT 1 FROM kept_nodes WHERE id = ? LIMIT 1", (node_id,)).fetchone()
    return row is not None


def _open_existing_membership_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-16384")
    return connection


def _staging_is_resumable(*, node_staging_path: Path, membership_db_path: Path) -> bool:
    if not (
        node_staging_path.exists()
        and membership_db_path.exists()
        and node_staging_path.stat().st_size > 0
        and membership_db_path.stat().st_size > 0
    ):
        return False
    try:
        connection = _open_existing_membership_db(membership_db_path)
        try:
            row = connection.execute("SELECT COUNT(*) FROM kept_nodes").fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return False
    return bool(row and row[0] is not None and int(row[0]) > 0)


def _remove_stale_staging(*, node_staging_path: Path, membership_db_path: Path) -> None:
    for path in (node_staging_path, membership_db_path):
        if path.exists():
            path.unlink()


def build_subset(
    *,
    graph_json: Path,
    corpus_csv: Path,
    output_json: Path,
    corridor_km: float = DEFAULT_CORRIDOR_KM,
    buffer_km: float | None = None,
) -> dict[str, Any]:
    if ijson is None:  # pragma: no cover - dependency boundary
        raise RuntimeError(f"ijson unavailable: {_IMPORT_ERROR}")
    if not graph_json.exists():
        raise FileNotFoundError(graph_json)
    if not corpus_csv.exists():
        raise FileNotFoundError(corpus_csv)

    effective_corridor_km = _resolve_corridor_km(corridor_km=corridor_km, buffer_km=buffer_km)
    corridors = _load_corridors(corpus_csv=corpus_csv, corridor_km=effective_corridor_km)
    if not corridors:
        raise RuntimeError("route_graph_subset_empty_corpus")

    header = _source_header(graph_json)
    corridor_union_bbox = _corridor_bbox_payload(corridors)
    nodes_seen = 0
    nodes_kept = 0
    output_json.parent.mkdir(parents=True, exist_ok=True)
    node_staging_path = output_json.with_suffix(".nodes.tmp.jsonl")
    membership_db_path = output_json.with_suffix(".nodes.tmp.sqlite3")
    tmp_path = output_json.with_suffix(f"{output_json.suffix}.tmp")
    membership_connection: sqlite3.Connection | None = None
    resumed_from_staging = False
    build_completed = False
    used_existing_output_json = False

    try:
        resumable_staging = _staging_is_resumable(
            node_staging_path=node_staging_path,
            membership_db_path=membership_db_path,
        )
        if not resumable_staging:
            _remove_stale_staging(
                node_staging_path=node_staging_path,
                membership_db_path=membership_db_path,
            )
            membership_connection = _init_membership_db(membership_db_path)
            pending_membership_rows: list[tuple[str]] = []
            with graph_json.open("rb") as handle, node_staging_path.open("w", encoding="utf-8") as nodes_tmp:
                for raw_node in ijson.items(handle, "nodes.item"):
                    nodes_seen += 1
                    parsed = _parse_node(raw_node)
                    if parsed is None:
                        continue
                    node_id, lat, lon = parsed
                    if not _node_within_any_corridor(
                        lat=lat,
                        lon=lon,
                        corridors=corridors,
                        corridor_km=effective_corridor_km,
                    ):
                        continue
                    pending_membership_rows.append((node_id,))
                    if len(pending_membership_rows) >= 10_000:
                        _flush_membership_batch(membership_connection, pending_membership_rows)
                    nodes_tmp.write(
                        json.dumps({"id": node_id, "lat": lat, "lon": lon}, separators=(",", ":"))
                    )
                    nodes_tmp.write("\n")
                    nodes_kept += 1
            _flush_membership_batch(membership_connection, pending_membership_rows)
            membership_connection.commit()
        else:
            resumed_from_staging = True
            membership_connection = _open_existing_membership_db(membership_db_path)
            row = membership_connection.execute("SELECT COUNT(*) FROM kept_nodes").fetchone()
            nodes_kept = int(row[0]) if row and row[0] is not None else 0

        if nodes_kept <= 0:
            raise RuntimeError("route_graph_subset_empty_after_corridor_filter")

        source_label = f"{header['source']}#subset:{corpus_csv.name}"
        generated_at_utc = _now()
        as_of_utc = header.get("as_of_utc") or generated_at_utc
        edge_count = 0
        if output_json.exists() and resumed_from_staging:
            used_existing_output_json = True
            membership_cursor = membership_connection.cursor()
            with graph_json.open("rb") as source_handle:
                for raw_edge in ijson.items(source_handle, "edges.item"):
                    if not isinstance(raw_edge, dict):
                        continue
                    u = str(raw_edge.get("u", "")).strip()
                    v = str(raw_edge.get("v", "")).strip()
                    if not u or not v:
                        continue
                    if not _membership_contains(membership_cursor, u):
                        continue
                    if not _membership_contains(membership_cursor, v):
                        continue
                    edge_count += 1
        else:
            if tmp_path.exists():
                tmp_path.unlink()
            membership_cursor = membership_connection.cursor()
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write("{")
                handle.write('"version":')
                json.dump(str(header.get("version") or "uk-routing-graph-v1"), handle, separators=(",", ":"))
                handle.write(',"source":')
                json.dump(source_label, handle, separators=(",", ":"))
                handle.write(',"generated_at_utc":')
                json.dump(generated_at_utc, handle, separators=(",", ":"))
                handle.write(',"as_of_utc":')
                json.dump(as_of_utc, handle, separators=(",", ":"))
                handle.write(',"bbox":')
                json.dump(corridor_union_bbox, handle, separators=(",", ":"))
                handle.write(',"nodes":[')
                first_node = True
                with node_staging_path.open("r", encoding="utf-8") as nodes_tmp:
                    for line in nodes_tmp:
                        node_payload = line.strip()
                        if not node_payload:
                            continue
                        if not first_node:
                            handle.write(",")
                        handle.write(node_payload)
                        first_node = False
                handle.write('],"edges":[')
                first_edge = True
                with graph_json.open("rb") as source_handle:
                    for raw_edge in ijson.items(source_handle, "edges.item"):
                        if not isinstance(raw_edge, dict):
                            continue
                        u = str(raw_edge.get("u", "")).strip()
                        v = str(raw_edge.get("v", "")).strip()
                        if not u or not v:
                            continue
                        if not _membership_contains(membership_cursor, u):
                            continue
                        if not _membership_contains(membership_cursor, v):
                            continue
                        if not first_edge:
                            handle.write(",")
                        json.dump(_json_safe(raw_edge), handle, separators=(",", ":"))
                        first_edge = False
                        edge_count += 1
                handle.write("]}")
            tmp_path.replace(output_json)

        meta_path = output_json.with_suffix(".meta.json")
        report = {
            "version": "uk-routing-graph-subset-v2",
            "graph_json": str(graph_json),
            "corpus_csv": str(corpus_csv),
            "output_json": str(output_json),
            "output_meta_json": str(meta_path),
            "corridor_km": float(effective_corridor_km),
            "selection_mode": "corridor_union_segment_buffer",
            "filter_mode": "per_od_corridor_union",
            "staging_mode": "node_jsonl_staging",
            "od_count": int(len(corridors)),
            "corridor_count": int(len(corridors)),
            "corridor_union_bbox": corridor_union_bbox,
            "source_version": str(header.get("version") or ""),
            "source_generated_at_utc": header.get("generated_at_utc"),
            "source_as_of_utc": header.get("as_of_utc"),
            "source_bbox": header.get("bbox"),
            "nodes_seen": int(nodes_seen),
            "nodes_kept": int(nodes_kept),
            "edges_kept": int(edge_count),
            "resumed_from_staging": bool(resumed_from_staging),
            "used_existing_output_json": bool(used_existing_output_json),
        }
        report.update(
            _emit_compact_bundle_for_subset_from_staging(
                graph_json=graph_json,
                output_json=output_json,
                node_staging_path=node_staging_path,
                version=str(header.get("version") or "uk-routing-graph-v1"),
                source_label=source_label,
                generated_at_utc=generated_at_utc,
                as_of_utc=as_of_utc,
            )
        )
        safe_report = _json_safe(report)
        meta_path.write_text(json.dumps(safe_report, indent=2), encoding="utf-8")
        build_completed = True
        return safe_report
    finally:
        if membership_connection is not None:
            membership_connection.close()
        if build_completed:
            if node_staging_path.exists():
                node_staging_path.unlink()
            if membership_db_path.exists():
                membership_db_path.unlink()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_subset(
        graph_json=Path(args.graph_json),
        corpus_csv=Path(args.corpus_csv),
        output_json=Path(args.output_json),
        corridor_km=float(args.corridor_km),
        buffer_km=args.buffer_km,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
