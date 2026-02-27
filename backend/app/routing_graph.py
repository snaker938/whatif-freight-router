from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from decimal import Decimal
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .k_shortest import (
    PathResult,
    TransitionStateFn,
    yen_k_shortest_paths,
    yen_k_shortest_paths_with_stats,
)
from .logging_utils import log_event
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


def _effective_route_graph_max_hops(
    *,
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
) -> int:
    base_hops = max(8, int(settings.route_graph_max_hops))
    if not bool(settings.route_graph_adaptive_hops_enabled):
        return base_hops
    cap_hops = max(base_hops, int(settings.route_graph_max_hops_cap))
    straight_line_m = max(
        0.0,
        _haversine_m(origin_lat, origin_lon, destination_lat, destination_lon),
    )
    straight_line_km = straight_line_m / 1000.0
    scaled_hops = int(
        math.ceil(
            straight_line_km
            * max(0.1, float(settings.route_graph_hops_per_km))
            * max(1.0, float(settings.route_graph_hops_detour_factor))
        )
    )
    edge_len_estimate_m = max(1.0, float(settings.route_graph_edge_length_estimate_m))
    hops_safety_factor = max(0.1, float(settings.route_graph_hops_safety_factor))
    hop_floor = int(math.ceil((straight_line_m / edge_len_estimate_m) * hops_safety_factor))
    return max(base_hops, min(cap_hops, max(scaled_hops, hop_floor)))


@dataclass(frozen=True)
class RouteGraph:
    version: str
    source: str
    nodes: dict[str, tuple[float, float]]
    adjacency: dict[str, tuple[GraphEdge, ...]]
    edge_index: dict[tuple[str, str], GraphEdge]
    grid_index: dict[tuple[int, int], tuple[str, ...]]
    component_by_node: dict[str, int]
    component_sizes: dict[int, int]
    component_count: int
    largest_component_nodes: int
    largest_component_ratio: float
    graph_fragmented: bool


@dataclass(frozen=True)
class GraphCandidateDiagnostics:
    explored_states: int
    generated_paths: int
    emitted_paths: int
    candidate_budget: int
    effective_max_hops: int = 0
    effective_hops_floor: int = 0
    effective_state_budget_initial: int = 0
    effective_state_budget: int = 0
    no_path_reason: str = ""
    no_path_detail: str = ""


@dataclass(frozen=True)
class GraphEdge:
    to: str
    cost: float
    distance_m: float
    highway: str
    toll: bool
    maxspeed_kph: float | None = None


_WARMUP_LOCK = threading.Lock()
_WARMUP_STATE = "idle"  # idle | loading | ready | failed
_WARMUP_STARTED_AT_UTC: str | None = None
_WARMUP_READY_AT_UTC: str | None = None
_WARMUP_STARTED_MONOTONIC: float | None = None
_WARMUP_LAST_ERROR: str | None = None
_WARMUP_LAST_SOURCE: str | None = None
_WARMUP_LAST_VERSION: str | None = None
_WARMUP_PHASE = "idle"
_WARMUP_TIMED_OUT = False
_WARMUP_PROGRESS_NODES_SEEN = 0
_WARMUP_PROGRESS_NODES_KEPT = 0
_WARMUP_PROGRESS_EDGES_SEEN = 0
_WARMUP_PROGRESS_EDGES_KEPT = 0
_WARMUP_COMPONENT_COUNT = 0
_WARMUP_LARGEST_COMPONENT_NODES = 0
_WARMUP_LARGEST_COMPONENT_RATIO = 0.0
_WARMUP_GRAPH_FRAGMENTED = False
_WARMUP_LAST_PROGRESS_LOG_MONOTONIC: float | None = None
_WARMUP_THREAD: threading.Thread | None = None


def _iso_utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _snapshot_warmup_state() -> dict[str, Any]:
    with _WARMUP_LOCK:
        state = str(_WARMUP_STATE)
        started_at_utc = _WARMUP_STARTED_AT_UTC
        ready_at_utc = _WARMUP_READY_AT_UTC
        started_monotonic = _WARMUP_STARTED_MONOTONIC
        last_error = _WARMUP_LAST_ERROR
        last_source = _WARMUP_LAST_SOURCE
        last_version = _WARMUP_LAST_VERSION
        phase = str(_WARMUP_PHASE)
        timed_out = bool(_WARMUP_TIMED_OUT)
        nodes_seen = int(_WARMUP_PROGRESS_NODES_SEEN)
        nodes_kept = int(_WARMUP_PROGRESS_NODES_KEPT)
        edges_seen = int(_WARMUP_PROGRESS_EDGES_SEEN)
        edges_kept = int(_WARMUP_PROGRESS_EDGES_KEPT)
        component_count = int(_WARMUP_COMPONENT_COUNT)
        largest_component_nodes = int(_WARMUP_LARGEST_COMPONENT_NODES)
        largest_component_ratio = float(_WARMUP_LARGEST_COMPONENT_RATIO)
        graph_fragmented = bool(_WARMUP_GRAPH_FRAGMENTED)
        thread_alive = bool(_WARMUP_THREAD is not None and _WARMUP_THREAD.is_alive())
    elapsed_ms: float | None = None
    if started_monotonic is not None:
        elapsed_ms = round(max(0.0, (time.monotonic() - started_monotonic) * 1000.0), 2)
    asset_path = str(_graph_asset_path())
    asset_exists = False
    asset_size_mb: float | None = None
    try:
        asset_exists = Path(asset_path).exists()
        if asset_exists:
            asset_size_mb = round(Path(asset_path).stat().st_size / (1024.0 * 1024.0), 2)
    except OSError:
        asset_exists = False
        asset_size_mb = None
    return {
        "state": state,
        "phase": phase,
        "started_at_utc": started_at_utc,
        "ready_at_utc": ready_at_utc,
        "elapsed_ms": elapsed_ms,
        "timeout_s": int(max(60, int(settings.route_graph_warmup_timeout_s))),
        "timed_out": timed_out,
        "last_error": last_error,
        "graph_source": last_source,
        "graph_version": last_version,
        "asset_path": asset_path,
        "asset_exists": asset_exists,
        "asset_size_mb": asset_size_mb,
        "nodes_seen": nodes_seen,
        "nodes_kept": nodes_kept,
        "edges_seen": edges_seen,
        "edges_kept": edges_kept,
        "component_count": component_count,
        "largest_component_nodes": largest_component_nodes,
        "largest_component_ratio": largest_component_ratio,
        "graph_fragmented": graph_fragmented,
        "max_nearest_node_distance_m": float(settings.route_graph_max_nearest_node_distance_m),
        "thread_alive": thread_alive,
        "cache_loaded": bool(load_route_graph.cache_info().currsize > 0),
    }


def _mark_warmup_loading() -> None:
    now = _iso_utc_now()
    with _WARMUP_LOCK:
        global _WARMUP_STATE, _WARMUP_STARTED_AT_UTC, _WARMUP_READY_AT_UTC
        global _WARMUP_STARTED_MONOTONIC, _WARMUP_LAST_ERROR, _WARMUP_PHASE
        global _WARMUP_TIMED_OUT, _WARMUP_PROGRESS_NODES_SEEN, _WARMUP_PROGRESS_NODES_KEPT
        global _WARMUP_PROGRESS_EDGES_SEEN, _WARMUP_PROGRESS_EDGES_KEPT
        global _WARMUP_COMPONENT_COUNT, _WARMUP_LARGEST_COMPONENT_NODES
        global _WARMUP_LARGEST_COMPONENT_RATIO, _WARMUP_GRAPH_FRAGMENTED
        global _WARMUP_LAST_PROGRESS_LOG_MONOTONIC
        _WARMUP_STATE = "loading"
        _WARMUP_STARTED_AT_UTC = now
        _WARMUP_READY_AT_UTC = None
        _WARMUP_STARTED_MONOTONIC = time.monotonic()
        _WARMUP_LAST_ERROR = None
        _WARMUP_PHASE = "initializing"
        _WARMUP_TIMED_OUT = False
        _WARMUP_PROGRESS_NODES_SEEN = 0
        _WARMUP_PROGRESS_NODES_KEPT = 0
        _WARMUP_PROGRESS_EDGES_SEEN = 0
        _WARMUP_PROGRESS_EDGES_KEPT = 0
        _WARMUP_COMPONENT_COUNT = 0
        _WARMUP_LARGEST_COMPONENT_NODES = 0
        _WARMUP_LARGEST_COMPONENT_RATIO = 0.0
        _WARMUP_GRAPH_FRAGMENTED = False
        _WARMUP_LAST_PROGRESS_LOG_MONOTONIC = None


def _set_warmup_phase(
    phase: str,
    *,
    nodes_seen: int | None = None,
    nodes_kept: int | None = None,
    edges_seen: int | None = None,
    edges_kept: int | None = None,
    force_log: bool = False,
) -> None:
    now = time.monotonic()
    should_log = force_log
    with _WARMUP_LOCK:
        global _WARMUP_PHASE, _WARMUP_PROGRESS_NODES_SEEN, _WARMUP_PROGRESS_NODES_KEPT
        global _WARMUP_PROGRESS_EDGES_SEEN, _WARMUP_PROGRESS_EDGES_KEPT
        global _WARMUP_LAST_PROGRESS_LOG_MONOTONIC
        _WARMUP_PHASE = str(phase).strip() or _WARMUP_PHASE
        if nodes_seen is not None:
            _WARMUP_PROGRESS_NODES_SEEN = int(max(0, nodes_seen))
        if nodes_kept is not None:
            _WARMUP_PROGRESS_NODES_KEPT = int(max(0, nodes_kept))
        if edges_seen is not None:
            _WARMUP_PROGRESS_EDGES_SEEN = int(max(0, edges_seen))
        if edges_kept is not None:
            _WARMUP_PROGRESS_EDGES_KEPT = int(max(0, edges_kept))
        if not should_log:
            if _WARMUP_LAST_PROGRESS_LOG_MONOTONIC is None:
                should_log = True
            elif now - _WARMUP_LAST_PROGRESS_LOG_MONOTONIC >= 15.0:
                should_log = True
        if should_log:
            _WARMUP_LAST_PROGRESS_LOG_MONOTONIC = now
        state = _WARMUP_STATE
        phase_now = _WARMUP_PHASE
        nodes_seen_now = _WARMUP_PROGRESS_NODES_SEEN
        nodes_kept_now = _WARMUP_PROGRESS_NODES_KEPT
        edges_seen_now = _WARMUP_PROGRESS_EDGES_SEEN
        edges_kept_now = _WARMUP_PROGRESS_EDGES_KEPT
        started_monotonic = _WARMUP_STARTED_MONOTONIC
    if not should_log:
        return
    elapsed_ms: float | None = None
    if started_monotonic is not None:
        elapsed_ms = round(max(0.0, (time.monotonic() - started_monotonic) * 1000.0), 2)
    log_event(
        "route_graph_warmup_progress",
        state=state,
        phase=phase_now,
        elapsed_ms=elapsed_ms,
        nodes_seen=int(nodes_seen_now),
        nodes_kept=int(nodes_kept_now),
        edges_seen=int(edges_seen_now),
        edges_kept=int(edges_kept_now),
    )


def _mark_warmup_ready(graph: RouteGraph) -> None:
    now = _iso_utc_now()
    with _WARMUP_LOCK:
        global _WARMUP_STATE, _WARMUP_READY_AT_UTC, _WARMUP_LAST_ERROR
        global _WARMUP_LAST_SOURCE, _WARMUP_LAST_VERSION
        global _WARMUP_PHASE, _WARMUP_TIMED_OUT
        global _WARMUP_COMPONENT_COUNT, _WARMUP_LARGEST_COMPONENT_NODES
        global _WARMUP_LARGEST_COMPONENT_RATIO, _WARMUP_GRAPH_FRAGMENTED
        _WARMUP_STATE = "ready"
        _WARMUP_READY_AT_UTC = now
        _WARMUP_LAST_ERROR = None
        _WARMUP_LAST_SOURCE = str(graph.source or "")
        _WARMUP_LAST_VERSION = str(graph.version or "")
        _WARMUP_PHASE = "ready"
        _WARMUP_TIMED_OUT = False
        _WARMUP_COMPONENT_COUNT = int(graph.component_count)
        _WARMUP_LARGEST_COMPONENT_NODES = int(graph.largest_component_nodes)
        _WARMUP_LARGEST_COMPONENT_RATIO = float(graph.largest_component_ratio)
        _WARMUP_GRAPH_FRAGMENTED = bool(graph.graph_fragmented)


def _mark_warmup_failed(error: str, *, timed_out: bool = False, phase: str | None = None) -> None:
    with _WARMUP_LOCK:
        global _WARMUP_STATE, _WARMUP_LAST_ERROR, _WARMUP_READY_AT_UTC
        global _WARMUP_PHASE, _WARMUP_TIMED_OUT
        _WARMUP_STATE = "failed"
        _WARMUP_LAST_ERROR = str(error).strip() or "unknown"
        _WARMUP_READY_AT_UTC = None
        if phase is not None and str(phase).strip():
            _WARMUP_PHASE = str(phase).strip()
        _WARMUP_TIMED_OUT = bool(timed_out)


def _warmup_deadline_monotonic() -> float | None:
    if threading.current_thread().name != "route-graph-warmup":
        return None
    with _WARMUP_LOCK:
        if _WARMUP_STATE != "loading" or _WARMUP_STARTED_MONOTONIC is None:
            return None
        timeout_s = max(60, int(settings.route_graph_warmup_timeout_s))
        return _WARMUP_STARTED_MONOTONIC + float(timeout_s)


def _raise_if_warmup_timed_out(deadline_monotonic: float | None, *, phase: str) -> None:
    if deadline_monotonic is None:
        return
    if time.monotonic() <= deadline_monotonic:
        return
    timeout_s = max(60, int(settings.route_graph_warmup_timeout_s))
    raise TimeoutError(f"route_graph_warmup_timeout phase={phase} timeout_s={timeout_s}")


def _warmup_worker() -> None:
    timeout_s = max(60, int(settings.route_graph_warmup_timeout_s))
    log_event(
        "route_graph_warmup_started",
        timeout_s=timeout_s,
        asset_path=str(_graph_asset_path()),
    )
    _set_warmup_phase("initializing", force_log=True)
    try:
        graph = load_route_graph()
        if graph is None:
            _mark_warmup_failed("route_graph_unavailable", timed_out=False, phase="finalizing")
            log_event(
                "route_graph_warmup_failed",
                reason="route_graph_unavailable",
                timeout_s=timeout_s,
                warmup=route_graph_warmup_status(),
            )
            return
        if bool(graph.graph_fragmented):
            with _WARMUP_LOCK:
                global _WARMUP_COMPONENT_COUNT, _WARMUP_LARGEST_COMPONENT_NODES
                global _WARMUP_LARGEST_COMPONENT_RATIO, _WARMUP_GRAPH_FRAGMENTED
                _WARMUP_COMPONENT_COUNT = int(graph.component_count)
                _WARMUP_LARGEST_COMPONENT_NODES = int(graph.largest_component_nodes)
                _WARMUP_LARGEST_COMPONENT_RATIO = float(graph.largest_component_ratio)
                _WARMUP_GRAPH_FRAGMENTED = True
            message = (
                "routing_graph_fragmented "
                f"(component_count={int(graph.component_count)}, "
                f"largest_component_nodes={int(graph.largest_component_nodes)}, "
                f"largest_component_ratio={float(graph.largest_component_ratio):.6f})"
            )
            _mark_warmup_failed(message, timed_out=False, phase="finalizing")
            log_event(
                "route_graph_warmup_failed",
                reason="routing_graph_fragmented",
                timeout_s=timeout_s,
                warmup=route_graph_warmup_status(),
                component_count=int(graph.component_count),
                largest_component_nodes=int(graph.largest_component_nodes),
                largest_component_ratio=round(float(graph.largest_component_ratio), 6),
            )
            return
        _mark_warmup_ready(graph)
        log_event(
            "route_graph_warmup_ready",
            timeout_s=timeout_s,
            warmup=route_graph_warmup_status(),
            node_count=len(graph.nodes),
            adjacency_count=len(graph.adjacency),
            edge_count=len(graph.edge_index),
        )
    except TimeoutError as exc:
        _mark_warmup_failed(str(exc), timed_out=True)
        log_event(
            "route_graph_warmup_timeout",
            timeout_s=timeout_s,
            warmup=route_graph_warmup_status(),
            error_message=str(exc).strip() or type(exc).__name__,
        )
        log_event(
            "route_graph_warmup_failed",
            reason="timeout",
            timeout_s=timeout_s,
            warmup=route_graph_warmup_status(),
        )
    except Exception as exc:  # pragma: no cover - defensive startup boundary
        _mark_warmup_failed(
            f"{type(exc).__name__}: {str(exc).strip()}",
            timed_out=False,
        )
        log_event(
            "route_graph_warmup_failed",
            reason="exception",
            timeout_s=timeout_s,
            warmup=route_graph_warmup_status(),
            error_type=type(exc).__name__,
            error_message=str(exc).strip() or type(exc).__name__,
        )
    finally:
        with _WARMUP_LOCK:
            global _WARMUP_THREAD
            _WARMUP_THREAD = None


def begin_route_graph_warmup(*, force: bool = False) -> None:
    if not bool(settings.route_graph_enabled):
        return
    if not bool(settings.route_graph_warmup_on_startup) and not bool(force):
        return
    thread: threading.Thread | None = None
    with _WARMUP_LOCK:
        global _WARMUP_THREAD, _WARMUP_STATE, _WARMUP_STARTED_AT_UTC, _WARMUP_READY_AT_UTC
        global _WARMUP_STARTED_MONOTONIC, _WARMUP_LAST_ERROR
        if _WARMUP_STATE == "loading" and _WARMUP_THREAD is not None and _WARMUP_THREAD.is_alive():
            return
        if _WARMUP_STATE == "ready" and not bool(force):
            return
        if force:
            load_route_graph.cache_clear()
        _WARMUP_STATE = "loading"
        _WARMUP_STARTED_AT_UTC = _iso_utc_now()
        _WARMUP_READY_AT_UTC = None
        _WARMUP_STARTED_MONOTONIC = time.monotonic()
        _WARMUP_LAST_ERROR = None
        thread = threading.Thread(target=_warmup_worker, name="route-graph-warmup", daemon=True)
        _WARMUP_THREAD = thread
    _mark_warmup_loading()
    if thread is not None:
        thread.start()


def route_graph_warmup_status() -> dict[str, Any]:
    return _snapshot_warmup_state()


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
    if not isinstance(lat_raw, (int, float, str, Decimal)) or isinstance(lat_raw, bool):
        return None
    if not isinstance(lon_raw, (int, float, str, Decimal)) or isinstance(lon_raw, bool):
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
        elif isinstance(maxspeed_raw, (int, float, str, Decimal)) and not isinstance(maxspeed_raw, bool):
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


def _compute_component_index(
    nodes: dict[str, tuple[float, float]],
    adjacency_mut: dict[str, list[GraphEdge]],
) -> tuple[dict[str, int], dict[int, int], int, int, float]:
    undirected: dict[str, set[str]] = {}
    for src, edges in adjacency_mut.items():
        if src not in nodes:
            continue
        out = undirected.setdefault(src, set())
        for edge in edges:
            dst = str(edge.to)
            if dst not in nodes:
                continue
            out.add(dst)
            undirected.setdefault(dst, set()).add(src)
    component_by_node: dict[str, int] = {}
    component_sizes: dict[int, int] = {}
    component_idx = 0
    for node_id in nodes:
        if node_id in component_by_node:
            continue
        component_idx += 1
        q: deque[str] = deque([node_id])
        size = 0
        while q:
            current = q.popleft()
            if current in component_by_node:
                continue
            component_by_node[current] = component_idx
            size += 1
            for nxt in undirected.get(current, ()):
                if nxt not in component_by_node:
                    q.append(nxt)
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
    component_by_node, component_sizes, component_count, largest_component_nodes, largest_component_ratio = (
        _compute_component_index(nodes, adjacency_mut)
    )
    min_giant_nodes = int(max(1, settings.route_graph_min_giant_component_nodes))
    min_giant_ratio = float(max(0.0, min(1.0, settings.route_graph_min_giant_component_ratio)))
    graph_fragmented = bool(
        largest_component_nodes < min_giant_nodes
        or largest_component_ratio < min_giant_ratio
    )
    return RouteGraph(
        version=version,
        source=source,
        nodes=nodes,
        adjacency=adjacency,
        edge_index=edge_index,
        grid_index=grid_index,
        component_by_node=component_by_node,
        component_sizes=component_sizes,
        component_count=component_count,
        largest_component_nodes=largest_component_nodes,
        largest_component_ratio=largest_component_ratio,
        graph_fragmented=graph_fragmented,
    )


def _load_route_graph_streaming(
    *,
    path: Path,
) -> RouteGraph | None:
    if ijson is None:
        return None
    version, source = _graph_meta_from_head(path)
    deadline_monotonic = _warmup_deadline_monotonic()
    nodes_seen = 0
    edges_seen = 0
    directed_edges_kept = 0
    nodes: dict[str, tuple[float, float]] = {}
    adjacency_mut: dict[str, list[GraphEdge]] = {}
    edge_index: dict[tuple[str, str], GraphEdge] = {}

    _set_warmup_phase(
        "parsing_nodes",
        nodes_seen=0,
        nodes_kept=0,
        edges_seen=0,
        edges_kept=0,
        force_log=True,
    )
    try:
        with path.open("rb") as fh:
            for raw_node in ijson.items(fh, "nodes.item"):
                nodes_seen += 1
                parsed = _parse_node(raw_node)
                if parsed is not None:
                    node_id, lat, lon = parsed
                    nodes[node_id] = (lat, lon)
                if nodes_seen % 100_000 == 0:
                    _set_warmup_phase(
                        "parsing_nodes",
                        nodes_seen=nodes_seen,
                        nodes_kept=len(nodes),
                        edges_seen=0,
                        edges_kept=0,
                    )
                _raise_if_warmup_timed_out(deadline_monotonic, phase="parsing_nodes")
    except TimeoutError:
        raise
    except Exception:
        return None

    if not nodes:
        return None

    _set_warmup_phase(
        "parsing_edges",
        nodes_seen=nodes_seen,
        nodes_kept=len(nodes),
        edges_seen=0,
        edges_kept=0,
        force_log=True,
    )
    try:
        with path.open("rb") as fh:
            for raw_edge in ijson.items(fh, "edges.item"):
                edges_seen += 1
                parsed = _parse_edge(raw_edge)
                if parsed is not None:
                    u, v, edge, oneway = parsed
                    if u in nodes and v in nodes:
                        adjacency_mut.setdefault(u, []).append(edge)
                        edge_index[(u, v)] = edge
                        directed_edges_kept += 1
                        if not oneway:
                            reverse = GraphEdge(
                                to=u,
                                cost=edge.cost,
                                distance_m=edge.distance_m,
                                highway=edge.highway,
                                toll=edge.toll,
                                maxspeed_kph=edge.maxspeed_kph,
                            )
                            adjacency_mut.setdefault(v, []).append(reverse)
                            edge_index[(v, u)] = reverse
                            directed_edges_kept += 1
                if edges_seen % 200_000 == 0:
                    _set_warmup_phase(
                        "parsing_edges",
                        nodes_seen=nodes_seen,
                        nodes_kept=len(nodes),
                        edges_seen=edges_seen,
                        edges_kept=directed_edges_kept,
                    )
                _raise_if_warmup_timed_out(deadline_monotonic, phase="parsing_edges")
    except TimeoutError:
        raise
    except Exception:
        return None

    if not edge_index:
        return None

    _set_warmup_phase(
        "finalizing",
        nodes_seen=nodes_seen,
        nodes_kept=len(nodes),
        edges_seen=edges_seen,
        edges_kept=directed_edges_kept,
        force_log=True,
    )
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
    if ijson is None:
        return None
    try:
        return _load_route_graph_streaming(path=path)
    except TimeoutError:
        raise
    except Exception:
        return None


def route_graph_status() -> tuple[bool, str]:
    if not settings.route_graph_enabled:
        return False, "disabled"
    graph = load_route_graph()
    if graph is None:
        return False, "unavailable"
    if bool(graph.graph_fragmented):
        return False, "fragmented"
    if _WARMUP_STATE != "ready":
        _mark_warmup_ready(graph)
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


def _nearest_node_with_distance(
    graph: RouteGraph,
    *,
    lat: float,
    lon: float,
) -> tuple[str | None, float]:
    candidates = _nearest_node_candidates(
        graph,
        lat=lat,
        lon=lon,
        max_candidates=1,
        max_radius=5,
    )
    if not candidates:
        return None, float("inf")
    first = candidates[0]
    return str(first.get("node_id")), float(first.get("distance_m", float("inf")))


def _nearest_node_candidates(
    graph: RouteGraph,
    *,
    lat: float,
    lon: float,
    max_candidates: int = 16,
    max_radius: int = 10,
    max_distance_m: float | None = None,
) -> list[dict[str, Any]]:
    candidate_limit = max(1, int(max_candidates))
    radius_limit = max(0, int(max_radius))
    center_key = _grid_key(lat, lon)
    distance_limit_m = (
        float(max_distance_m)
        if isinstance(max_distance_m, (int, float)) and math.isfinite(float(max_distance_m)) and float(max_distance_m) > 0.0
        else None
    )
    best_by_node: dict[str, float] = {}

    def _ring_offsets(radius: int) -> tuple[tuple[int, int], ...]:
        if radius <= 0:
            return ((0, 0),)
        offsets: list[tuple[int, int]] = []
        for dx in range(-radius, radius + 1):
            offsets.append((dx, -radius))
            offsets.append((dx, radius))
        for dy in range(-radius + 1, radius):
            offsets.append((-radius, dy))
            offsets.append((radius, dy))
        return tuple(offsets)

    for radius in range(0, radius_limit + 1):
        for dx, dy in _ring_offsets(radius):
            key = (center_key[0] + dy, center_key[1] + dx)
            for node_id in graph.grid_index.get(key, ()):
                coords = graph.nodes.get(node_id)
                if coords is None:
                    continue
                n_lat, n_lon = coords
                dist = _haversine_m(lat, lon, n_lat, n_lon)
                if not math.isfinite(dist):
                    continue
                if distance_limit_m is not None and dist > distance_limit_m:
                    continue
                prior = best_by_node.get(node_id)
                if prior is None or dist < prior:
                    best_by_node[node_id] = float(dist)
    if not best_by_node:
        return []
    sorted_nodes = sorted(best_by_node.items(), key=lambda item: item[1])[:candidate_limit]
    out: list[dict[str, Any]] = []
    for node_id, dist_m in sorted_nodes:
        component_raw = graph.component_by_node.get(node_id)
        component_id = int(component_raw) if isinstance(component_raw, int) else None
        component_size = int(graph.component_sizes.get(component_id, 0)) if component_id is not None else 0
        out.append(
            {
                "node_id": str(node_id),
                "distance_m": float(dist_m),
                "component_id": component_id,
                "component_size": component_size,
            }
        )
    return out


def _candidate_summary_rows(
    candidates: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidates[: max(1, int(limit))]:
        rows.append(
            {
                "node_id": str(row.get("node_id", "")),
                "distance_m": float(row.get("distance_m", 0.0)),
                "component_id": (
                    int(row.get("component_id"))
                    if isinstance(row.get("component_id"), int)
                    else None
                ),
                "component_size": int(row.get("component_size", 0)),
            }
        )
    return rows


def _select_component_aligned_od_nodes(
    *,
    origin_candidates: list[dict[str, Any]],
    destination_candidates: list[dict[str, Any]],
    max_nearest_m: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None
    destination_best_by_component: dict[int, dict[str, Any]] = {}
    for destination_row in destination_candidates:
        destination_dist_m = float(destination_row.get("distance_m", float("inf")))
        if not math.isfinite(destination_dist_m) or destination_dist_m > max_nearest_m:
            continue
        destination_component = destination_row.get("component_id")
        if not isinstance(destination_component, int):
            continue
        component_id = int(destination_component)
        prior = destination_best_by_component.get(component_id)
        if prior is None or destination_dist_m < float(prior.get("distance_m", float("inf"))):
            destination_best_by_component[component_id] = destination_row
    for origin_row in origin_candidates:
        origin_dist_m = float(origin_row.get("distance_m", float("inf")))
        if not math.isfinite(origin_dist_m) or origin_dist_m > max_nearest_m:
            continue
        origin_component = origin_row.get("component_id")
        if not isinstance(origin_component, int):
            continue
        component_id = int(origin_component)
        destination_row = destination_best_by_component.get(component_id)
        if destination_row is None:
            continue
        destination_dist_m = float(destination_row.get("distance_m", float("inf")))
        component_size = max(
            int(origin_row.get("component_size", 0)),
            int(destination_row.get("component_size", 0)),
        )
        score = (
            float(origin_dist_m + destination_dist_m),
            float(max(origin_dist_m, destination_dist_m)),
            -float(component_size),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = {
                "origin_node_id": str(origin_row.get("node_id", "")),
                "destination_node_id": str(destination_row.get("node_id", "")),
                "origin_selected_distance_m": float(origin_dist_m),
                "destination_selected_distance_m": float(destination_dist_m),
                "selected_component": component_id,
                "selected_component_size": int(component_size),
            }
    return best


def _candidate_search_radius_for_max_distance(
    *,
    max_distance_m: float,
    radius_cap: int,
) -> int:
    cap = max(1, int(radius_cap))
    if not math.isfinite(max_distance_m) or max_distance_m <= 0.0:
        return cap
    # 0.15deg buckets are ~16-17km in latitude; this keeps scans local to the configured snap radius.
    adaptive = int(math.ceil(float(max_distance_m) / 12_000.0)) + 1
    return max(1, min(cap, adaptive))


def _nearest_node_id(graph: RouteGraph, *, lat: float, lon: float) -> str | None:
    node_id, _distance_m = _nearest_node_with_distance(graph, lat=lat, lon=lon)
    return node_id


def route_graph_od_feasibility(
    *,
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
) -> dict[str, Any]:
    if not bool(settings.route_graph_enabled):
        return {
            "ok": False,
            "reason_code": "routing_graph_unavailable",
            "message": "Route graph is disabled.",
        }
    graph = load_route_graph()
    if graph is None:
        return {
            "ok": False,
            "reason_code": "routing_graph_unavailable",
            "message": "Route graph asset is unavailable.",
        }
    if bool(graph.graph_fragmented):
        return {
            "ok": False,
            "reason_code": "routing_graph_fragmented",
            "message": "Route graph is fragmented for strict runtime policy.",
            "component_count": int(graph.component_count),
            "largest_component_nodes": int(graph.largest_component_nodes),
            "largest_component_ratio": float(graph.largest_component_ratio),
        }
    max_nearest_m = float(max(100.0, settings.route_graph_max_nearest_node_distance_m))
    candidate_limit = max(8, int(settings.route_graph_od_candidate_limit))
    candidate_radius_cap = max(1, int(settings.route_graph_od_candidate_max_radius))
    candidate_radius = _candidate_search_radius_for_max_distance(
        max_distance_m=max_nearest_m,
        radius_cap=candidate_radius_cap,
    )
    origin_candidates = _nearest_node_candidates(
        graph,
        lat=origin_lat,
        lon=origin_lon,
        max_candidates=candidate_limit,
        max_radius=candidate_radius,
        max_distance_m=max_nearest_m,
    )
    destination_candidates = _nearest_node_candidates(
        graph,
        lat=destination_lat,
        lon=destination_lon,
        max_candidates=candidate_limit,
        max_radius=candidate_radius,
        max_distance_m=max_nearest_m,
    )
    origin_nearest_distance_m = (
        float(origin_candidates[0].get("distance_m", float("inf"))) if origin_candidates else float("inf")
    )
    destination_nearest_distance_m = (
        float(destination_candidates[0].get("distance_m", float("inf"))) if destination_candidates else float("inf")
    )
    selected = _select_component_aligned_od_nodes(
        origin_candidates=origin_candidates,
        destination_candidates=destination_candidates,
        max_nearest_m=max_nearest_m,
    )
    if (
        selected is None
        and candidate_radius < candidate_radius_cap
    ):
        fallback_origin_candidates = _nearest_node_candidates(
            graph,
            lat=origin_lat,
            lon=origin_lon,
            max_candidates=candidate_limit,
            max_radius=candidate_radius_cap,
            max_distance_m=max_nearest_m,
        )
        fallback_destination_candidates = _nearest_node_candidates(
            graph,
            lat=destination_lat,
            lon=destination_lon,
            max_candidates=candidate_limit,
            max_radius=candidate_radius_cap,
            max_distance_m=max_nearest_m,
        )
        if len(fallback_origin_candidates) >= len(origin_candidates):
            origin_candidates = fallback_origin_candidates
        if len(fallback_destination_candidates) >= len(destination_candidates):
            destination_candidates = fallback_destination_candidates
        selected = _select_component_aligned_od_nodes(
            origin_candidates=origin_candidates,
            destination_candidates=destination_candidates,
            max_nearest_m=max_nearest_m,
        )
    nearest_origin_component = (
        int(origin_candidates[0].get("component_id"))
        if origin_candidates and isinstance(origin_candidates[0].get("component_id"), int)
        else None
    )
    nearest_destination_component = (
        int(destination_candidates[0].get("component_id"))
        if destination_candidates and isinstance(destination_candidates[0].get("component_id"), int)
        else None
    )
    summary = {
        "origin_candidate_count": int(len(origin_candidates)),
        "destination_candidate_count": int(len(destination_candidates)),
        "origin_candidates_top": _candidate_summary_rows(origin_candidates),
        "destination_candidates_top": _candidate_summary_rows(destination_candidates),
        "candidate_search_radius": int(candidate_radius),
        "candidate_search_radius_cap": int(candidate_radius_cap),
        "candidate_limit": int(candidate_limit),
    }
    if (
        not origin_candidates
        or not destination_candidates
        or not math.isfinite(origin_nearest_distance_m)
        or not math.isfinite(destination_nearest_distance_m)
        or origin_nearest_distance_m > max_nearest_m
        or destination_nearest_distance_m > max_nearest_m
    ):
        return {
            "ok": False,
            "reason_code": "routing_graph_coverage_gap",
            "message": "Route graph coverage gap around origin/destination.",
            "origin_nearest_distance_m": (
                None if not math.isfinite(origin_nearest_distance_m) else float(origin_nearest_distance_m)
            ),
            "destination_nearest_distance_m": (
                None
                if not math.isfinite(destination_nearest_distance_m)
                else float(destination_nearest_distance_m)
            ),
            "max_nearest_node_distance_m": float(max_nearest_m),
            **summary,
        }
    if selected is None:
        return {
            "ok": False,
            "reason_code": "routing_graph_disconnected_od",
            "message": "Origin and destination are disconnected in the loaded graph component map.",
            "origin_node_id": str(origin_candidates[0].get("node_id", "")) if origin_candidates else None,
            "destination_node_id": (
                str(destination_candidates[0].get("node_id", "")) if destination_candidates else None
            ),
            "origin_component": nearest_origin_component,
            "destination_component": nearest_destination_component,
            "origin_nearest_distance_m": float(origin_nearest_distance_m),
            "destination_nearest_distance_m": float(destination_nearest_distance_m),
            "max_nearest_node_distance_m": float(max_nearest_m),
            **summary,
        }
    return {
        "ok": True,
        "reason_code": "ok",
        "message": "Graph coverage/connectivity checks passed.",
        "origin_node_id": str(selected.get("origin_node_id", "")),
        "destination_node_id": str(selected.get("destination_node_id", "")),
        "origin_component": int(selected.get("selected_component", 0)),
        "destination_component": int(selected.get("selected_component", 0)),
        "origin_selected_distance_m": float(selected.get("origin_selected_distance_m", 0.0)),
        "destination_selected_distance_m": float(selected.get("destination_selected_distance_m", 0.0)),
        "origin_nearest_distance_m": float(origin_nearest_distance_m),
        "destination_nearest_distance_m": float(destination_nearest_distance_m),
        "selected_component": int(selected.get("selected_component", 0)),
        "selected_component_size": int(selected.get("selected_component_size", 0)),
        "max_nearest_node_distance_m": float(max_nearest_m),
        **summary,
    }


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
    max_hops_override: int | None = None,
    max_state_budget_override: int | None = None,
    search_deadline_s: float | None = None,
    start_node_id: str | None = None,
    goal_node_id: str | None = None,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not settings.route_graph_enabled:
        return ()
    graph = load_route_graph()
    if graph is None:
        return ()
    if start_node_id is not None and goal_node_id is not None:
        start = str(start_node_id).strip() or None
        goal = str(goal_node_id).strip() or None
        if not start or not goal or start not in graph.nodes or goal not in graph.nodes:
            return ()
    else:
        feasibility = route_graph_od_feasibility(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            destination_lat=destination_lat,
            destination_lon=destination_lon,
        )
        if not bool(feasibility.get("ok")):
            return ()
        start = str(feasibility.get("origin_node_id", "")).strip() or _nearest_node_id(
            graph,
            lat=origin_lat,
            lon=origin_lon,
        )
        goal = str(feasibility.get("destination_node_id", "")).strip() or _nearest_node_id(
            graph,
            lat=destination_lat,
            lon=destination_lon,
        )
    if not start or not goal or start == goal:
        return ()
    effective_max_hops = (
        max(8, int(max_hops_override))
        if max_hops_override is not None
        else _effective_route_graph_max_hops(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            destination_lat=destination_lat,
            destination_lon=destination_lon,
        )
    )
    configured_state_budget = max(1000, int(settings.route_graph_max_state_budget))
    state_budget_per_hop = max(10, int(settings.route_graph_state_budget_per_hop))
    state_budget_cap = max(
        configured_state_budget,
        int(settings.route_graph_state_budget_retry_cap),
    )
    effective_state_budget_initial = max(
        configured_state_budget,
        int(effective_max_hops * state_budget_per_hop),
    )
    effective_state_budget_initial = min(effective_state_budget_initial, state_budget_cap)
    effective_state_budget = (
        min(state_budget_cap, max(1000, int(max_state_budget_override)))
        if max_state_budget_override is not None
        else effective_state_budget_initial
    )
    paths = yen_k_shortest_paths(
        adjacency=_adjacency_cost_view(graph),
        start=start,
        goal=goal,
        k=max(1, int(max_paths or settings.route_graph_k_paths)),
        max_hops=effective_max_hops,
        max_state_budget=effective_state_budget,
        max_repeat_per_node=max(0, int(settings.route_graph_max_repeat_per_node)),
        max_detour_ratio=max(1.0, float(settings.route_graph_max_detour_ratio)),
        max_candidate_pool=max(16, int(settings.route_graph_k_paths) * 8),
        transition_state_fn=_transition_state_callback(graph),
        deadline_monotonic_s=search_deadline_s,
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
    max_hops_override: int | None = None,
    max_state_budget_override: int | None = None,
    use_transition_state: bool = True,
    search_deadline_s: float | None = None,
    start_node_id: str | None = None,
    goal_node_id: str | None = None,
) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
    budget = max(1, int(max_paths or settings.route_graph_k_paths))
    graph = load_route_graph()
    if graph is None:
        return [], GraphCandidateDiagnostics(
            explored_states=0,
            generated_paths=0,
            emitted_paths=0,
            candidate_budget=budget,
            effective_max_hops=0,
            effective_state_budget=0,
            no_path_reason="routing_graph_unavailable",
            no_path_detail="Route graph asset is unavailable.",
        )
    if start_node_id is not None and goal_node_id is not None:
        start = str(start_node_id).strip() or None
        goal = str(goal_node_id).strip() or None
        if not start or not goal or start not in graph.nodes or goal not in graph.nodes:
            return [], GraphCandidateDiagnostics(
                explored_states=0,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=budget,
                effective_max_hops=0,
                effective_state_budget=0,
                no_path_reason="routing_graph_coverage_gap",
                no_path_detail="Pinned route graph endpoint node is unavailable in the loaded graph.",
            )
    else:
        feasibility = route_graph_od_feasibility(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            destination_lat=destination_lat,
            destination_lon=destination_lon,
        )
        if not bool(feasibility.get("ok")):
            return [], GraphCandidateDiagnostics(
                explored_states=0,
                generated_paths=0,
                emitted_paths=0,
                candidate_budget=budget,
                effective_max_hops=0,
                effective_state_budget=0,
                no_path_reason=str(feasibility.get("reason_code", "routing_graph_unavailable")),
                no_path_detail=str(feasibility.get("message", "Graph feasibility check failed.")),
            )
        start = str(feasibility.get("origin_node_id", "")).strip() or _nearest_node_id(
            graph,
            lat=origin_lat,
            lon=origin_lon,
        )
        goal = str(feasibility.get("destination_node_id", "")).strip() or _nearest_node_id(
            graph,
            lat=destination_lat,
            lon=destination_lon,
        )
    if not start or not goal or start == goal:
        return [], GraphCandidateDiagnostics(
            explored_states=0,
            generated_paths=0,
            emitted_paths=0,
            candidate_budget=budget,
            effective_max_hops=0,
            effective_state_budget=0,
            no_path_reason="routing_graph_coverage_gap",
            no_path_detail="Nearest route graph node could not be resolved for one or both endpoints.",
        )
    effective_max_hops = (
        max(8, int(max_hops_override))
        if max_hops_override is not None
        else _effective_route_graph_max_hops(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            destination_lat=destination_lat,
            destination_lon=destination_lon,
        )
    )
    straight_line_m = max(
        0.0,
        _haversine_m(origin_lat, origin_lon, destination_lat, destination_lon),
    )
    edge_len_estimate_m = max(1.0, float(settings.route_graph_edge_length_estimate_m))
    hops_safety_factor = max(0.1, float(settings.route_graph_hops_safety_factor))
    effective_hops_floor = int(math.ceil((straight_line_m / edge_len_estimate_m) * hops_safety_factor))
    configured_state_budget = max(1000, int(settings.route_graph_max_state_budget))
    state_budget_per_hop = max(10, int(settings.route_graph_state_budget_per_hop))
    state_budget_cap = max(
        configured_state_budget,
        int(settings.route_graph_state_budget_retry_cap),
    )
    effective_state_budget_initial = max(
        configured_state_budget,
        int(effective_max_hops * state_budget_per_hop),
    )
    effective_state_budget_initial = min(effective_state_budget_initial, state_budget_cap)
    effective_state_budget = (
        min(state_budget_cap, max(1000, int(max_state_budget_override)))
        if max_state_budget_override is not None
        else effective_state_budget_initial
    )
    paths, stats = yen_k_shortest_paths_with_stats(
        adjacency=_adjacency_cost_view(graph, scenario_edge_modifiers=scenario_edge_modifiers),
        start=start,
        goal=goal,
        k=budget,
        max_hops=effective_max_hops,
        max_state_budget=effective_state_budget,
        max_repeat_per_node=max(0, int(settings.route_graph_max_repeat_per_node)),
        max_detour_ratio=max(1.0, float(settings.route_graph_max_detour_ratio)),
        max_candidate_pool=max(16, int(settings.route_graph_k_paths) * 8),
        transition_state_fn=(
            _transition_state_callback(graph, scenario_edge_modifiers=scenario_edge_modifiers)
            if bool(use_transition_state)
            else None
        ),
        deadline_monotonic_s=search_deadline_s,
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
        effective_max_hops=int(effective_max_hops),
        effective_hops_floor=int(effective_hops_floor),
        effective_state_budget_initial=int(effective_state_budget_initial),
        effective_state_budget=int(effective_state_budget),
        no_path_reason=str(stats.get("no_path_reason", "")).strip(),
        no_path_detail=(
            str(stats.get("first_error", "")).strip()
            or str(stats.get("termination_reason", "")).strip()
        ),
    )
