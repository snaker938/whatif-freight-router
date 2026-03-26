from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.routing_graph import GraphCandidateDiagnostics, route_graph_candidate_routes, route_graph_od_feasibility
from app.settings import settings

EVAL_DATA_DIR = PROJECT_ROOT / "data" / "eval"


@dataclass(frozen=True)
class UKBBox:
    south: float
    north: float
    west: float
    east: float


DISTANCE_BINS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 30.0, "0-30 km"),
    (30.0, 100.0, "30-100 km"),
    (100.0, 250.0, "100-250 km"),
    (250.0, None, "250+ km"),
)

AMBIGUITY_SOURCE_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "routing_graph_probe": 0.72,
    "existing_corpus": 0.60,
}


def _region_bucket(lat: float) -> str:
    if lat < 52.2:
        return "south"
    if lat < 54.2:
        return "midlands"
    if lat < 55.8:
        return "north"
    return "scotland"


def _corridor_bucket(origin: dict[str, float], destination: dict[str, float]) -> str:
    lat_delta = float(destination["lat"]) - float(origin["lat"])
    lon_delta = float(destination["lon"]) - float(origin["lon"])
    abs_lat = abs(lat_delta)
    abs_lon = abs(lon_delta)
    if abs_lat >= (1.75 * abs_lon):
        orientation = "north_south"
    elif abs_lon >= (1.75 * abs_lat):
        orientation = "east_west"
    else:
        orientation = "diagonal"
    return f"{_region_bucket(float(origin['lat']))}_to_{_region_bucket(float(destination['lat']))}|{orientation}"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_bbox(raw: str) -> UKBBox:
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("bbox must contain four comma-separated floats: south,north,west,east")
    south, north, west, east = (float(part) for part in parts)
    if north <= south or east <= west:
        raise ValueError("bbox bounds are invalid")
    return UKBBox(south=south, north=north, west=west, east=east)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _distance_bin(distance_km: float) -> str:
    for lower, upper, label in DISTANCE_BINS:
        if distance_km < lower:
            continue
        if upper is None or distance_km < upper:
            return label
    return DISTANCE_BINS[-1][2]


def _bin_index(distance_km: float) -> int:
    label = _distance_bin(distance_km)
    for idx, (_, _, bin_label) in enumerate(DISTANCE_BINS):
        if bin_label == label:
            return idx
    return len(DISTANCE_BINS) - 1


def _split_evenly(total: int, bucket_count: int) -> list[int]:
    total = max(0, int(total))
    bucket_count = max(1, int(bucket_count))
    base = total // bucket_count
    remainder = total % bucket_count
    return [base + (1 if idx < remainder else 0) for idx in range(bucket_count)]


def _sample_candidate_pair(rng: random.Random, bbox: UKBBox) -> tuple[dict[str, float], dict[str, float]]:
    origin = {
        "lat": rng.uniform(bbox.south, bbox.north),
        "lon": rng.uniform(bbox.west, bbox.east),
    }
    destination = {
        "lat": rng.uniform(bbox.south, bbox.north),
        "lon": rng.uniform(bbox.west, bbox.east),
    }
    return origin, destination


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _row_hash(rows: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(_canonical_json(rows).encode("utf-8"))
    return hasher.hexdigest()


def _safe_distance_m(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_scalar(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _is_nonzero_numeric(value: Any) -> bool:
    parsed = _safe_scalar(value, float("nan"))
    return math.isfinite(parsed) and abs(parsed) > 1e-12


def _route_distance_km(route: dict[str, Any]) -> float:
    if "distance_km" in route:
        return _safe_scalar(route.get("distance_km"))
    raw_distance = _safe_scalar(route.get("distance"))
    if raw_distance > 500.0:
        return raw_distance / 1000.0
    return raw_distance


def _route_duration_s(route: dict[str, Any]) -> float:
    if "duration_s" in route:
        return _safe_scalar(route.get("duration_s"))
    return _safe_scalar(route.get("duration"))


def _route_cost(route: dict[str, Any]) -> float:
    return _safe_scalar(route.get("monetary_cost", route.get("cost", 0.0)))


def _route_emissions(route: dict[str, Any]) -> float:
    return _safe_scalar(route.get("emissions_kg", route.get("co2_kg", route.get("emissions", 0.0))))


def _route_toll_indicator(route: dict[str, Any]) -> float:
    raw_candidates = (
        route.get("has_tolls"),
        route.get("has_toll"),
        route.get("toll_exposure_share"),
        route.get("tolled_share"),
        route.get("toll_cost"),
    )
    for value in raw_candidates:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        scalar = _safe_scalar(value, float("nan"))
        if math.isfinite(scalar) and scalar > 0.0:
            return 1.0
    return 0.0


def _route_family_id(route: dict[str, Any], *, ordinal: int) -> str:
    for key in ("corridor_family", "family_id", "signature", "graph_signature", "route_signature"):
        value = str(route.get(key) or "").strip()
        if value:
            return value
    mechanism = route.get("mechanism_descriptor")
    if isinstance(mechanism, dict):
        for key in ("family", "cluster", "signature"):
            value = str(mechanism.get(key) or "").strip()
            if value:
                return value
    node_ids = route.get("node_ids")
    if isinstance(node_ids, list) and node_ids:
        return f"nodes:{len(node_ids)}:{node_ids[0]}:{node_ids[-1]}"
    polyline = str(route.get("polyline") or "").strip()
    if polyline:
        return f"polyline:{hashlib.sha256(polyline.encode('utf-8')).hexdigest()[:12]}"
    distance_bucket = int(round(_route_distance_km(route) / 25.0))
    duration_bucket = int(round(_route_duration_s(route) / 1800.0))
    toll_bucket = int(_route_toll_indicator(route))
    return f"heuristic:{distance_bucket}:{duration_bucket}:{toll_bucket}:{ordinal}"


def _normalised_spread(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) <= 1:
        return 0.0
    high = max(finite)
    low = min(finite)
    if high <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (high - low) / high))


def _candidate_weighted_margin(routes: list[dict[str, Any]]) -> float:
    if len(routes) <= 1:
        return 1.0
    dims = {
        "distance": [_route_distance_km(route) for route in routes],
        "duration": [_route_duration_s(route) for route in routes],
        "cost": [_route_cost(route) for route in routes],
        "emissions": [_route_emissions(route) for route in routes],
    }
    mins = {key: min(values) for key, values in dims.items()}
    maxs = {key: max(values) for key, values in dims.items()}
    scores: list[float] = []
    for route in routes:
        total = 0.0
        for key, raw in (
            ("distance", _route_distance_km(route)),
            ("duration", _route_duration_s(route)),
            ("cost", _route_cost(route)),
            ("emissions", _route_emissions(route)),
        ):
            low = mins[key]
            high = maxs[key]
            if high > low:
                total += (raw - low) / (high - low)
        scores.append(total / 4.0)
    ordered = sorted(scores)
    return round(max(0.0, min(1.0, ordered[1] - ordered[0])), 6)


def _top_two_relative_gap(values: Sequence[float], *, prefer_low: bool = True) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return 0.0
    ordered = sorted(finite) if prefer_low else sorted(finite, reverse=True)
    best = ordered[0]
    runner_up = ordered[1]
    scale = max(abs(best), abs(runner_up), 1.0)
    return round(max(0.0, min(1.0, abs(runner_up - best) / scale)), 6)


def _path_sufficiency(path_count: int) -> float:
    return max(0.0, min(1.0, max(0, int(path_count) - 1) / 7.0))


def _family_balance(path_count: int, family_count: int) -> float:
    if path_count <= 1:
        return 0.0
    family_ratio = max(0.0, min(1.0, float(max(0, int(family_count))) / float(max(1, int(path_count)))))
    saturation = max(0.0, min(1.0, float(max(0, int(family_count)) - 1) / 4.0))
    return round((0.55 * family_ratio) + (0.45 * saturation), 6)


def _cheap_prior_features(
    *,
    path_count: int,
    family_count: int,
    objective_spread: float,
    nominal_margin: float,
    toll_disagreement: float,
    top2_duration_gap: float = 0.0,
    top2_cost_gap: float = 0.0,
    top2_emissions_gap: float = 0.0,
) -> dict[str, float]:
    clamped_spread = max(0.0, min(1.0, float(objective_spread)))
    clamped_margin = max(0.0, min(1.0, float(nominal_margin)))
    clamped_toll = max(0.0, min(1.0, float(toll_disagreement)))
    path_density = _path_sufficiency(path_count)
    family_diversity = _family_balance(path_count, family_count)
    top2_gap_pressure = max(
        0.0,
        min(
            1.0,
            1.0
            - (
                (
                    max(0.0, min(1.0, float(top2_duration_gap)))
                    + max(0.0, min(1.0, float(top2_cost_gap)))
                    + max(0.0, min(1.0, float(top2_emissions_gap)))
                )
                / 3.0
            ),
        ),
    )
    ambiguity_pressure = (
        (1.0 - clamped_margin) * 0.36
        + clamped_spread * 0.24
        + family_diversity * 0.18
        + top2_gap_pressure * 0.14
        + clamped_toll * 0.05
        + path_density * 0.03
    )
    engine_disagreement_prior = (
        ambiguity_pressure * 0.64
        + top2_gap_pressure * 0.18
        + clamped_spread * 0.08
        + family_diversity * 0.06
        + clamped_toll * 0.04
    )
    hard_case_prior = (
        max(0.0, min(1.0, ambiguity_pressure)) * 0.44
        + max(0.0, min(1.0, engine_disagreement_prior)) * 0.24
        + top2_gap_pressure * 0.14
        + family_diversity * 0.10
        + clamped_spread * 0.06
        + clamped_toll * 0.02
    )
    return {
        "candidate_probe_family_diversity": round(family_diversity, 6),
        "candidate_probe_path_sufficiency": round(path_density, 6),
        "candidate_probe_top2_gap_pressure": round(top2_gap_pressure, 6),
        "candidate_probe_engine_disagreement_prior": round(
            max(0.0, min(1.0, engine_disagreement_prior)),
            6,
        ),
        "hard_case_prior": round(max(0.0, min(1.0, hard_case_prior)), 6),
    }


def _split_prior_sources(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, Mapping):
        return [str(key).strip() for key in value.keys() if str(key).strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seen: set[str] = set()
        tokens: list[str] = []
        for item in value:
            token = str(item).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens
    text = str(value).strip()
    if not text:
        return []
    if text[:1] in {"{", "["}:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        else:
            if isinstance(parsed, Mapping):
                return _split_prior_sources(parsed)
            if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
                return _split_prior_sources(parsed)
    seen: set[str] = set()
    tokens: list[str] = []
    for part in text.replace("+", ",").split(","):
        token = part.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _merge_prior_sources(existing: Any, new_source: str | None) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for token in _split_prior_sources(existing):
        if token in seen:
            continue
        seen.add(token)
        merged.append(token)
    if new_source:
        token = str(new_source).strip()
        if token and token not in seen:
            merged.append(token)
    return ",".join(merged)


def _source_mix_string(source_counts: Mapping[str, Any] | Sequence[str]) -> str:
    if isinstance(source_counts, Mapping):
        counts = {
            str(token).strip(): max(1, int(_safe_scalar(value, 1.0)))
            for token, value in source_counts.items()
            if str(token).strip()
        }
    else:
        counts = {token: 1 for token in _split_prior_sources(source_counts)}
    if not counts:
        return ""
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def _source_mix_counts_for_row(row: Mapping[str, Any], source_tokens: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in source_tokens:
        if token == "routing_graph_probe":
            counts[token] = max(
                1,
                int(_safe_scalar(row.get("candidate_probe_emitted_paths"), 0.0)),
                int(_safe_scalar(row.get("candidate_probe_path_count"), 0.0)),
                int(_safe_scalar(row.get("candidate_probe_corridor_family_count"), 0.0)),
            )
        elif token == "engine_augmented_probe":
            counts[token] = max(
                1,
                sum(
                    1
                    for key in (
                        "candidate_probe_engine_disagreement_prior",
                        "hard_case_prior",
                        "candidate_probe_objective_spread",
                        "candidate_probe_toll_disagreement_rate",
                    )
                    if _is_nonzero_numeric(row.get(key))
                ),
            )
        elif token == "historical_results_bootstrap":
            counts[token] = max(
                1,
                int(_safe_scalar(row.get("ambiguity_prior_sample_count"), 0.0)),
                int(_safe_scalar(row.get("ambiguity_prior_support_count"), 0.0)),
            )
        elif token == "repo_local_geometry_backfill":
            counts[token] = max(
                1,
                sum(
                    1
                    for key in (
                        "candidate_probe_path_count",
                        "candidate_probe_corridor_family_count",
                        "candidate_probe_objective_spread",
                        "candidate_probe_nominal_margin",
                    )
                    if _is_nonzero_numeric(row.get(key))
                ),
            )
        elif token == "existing_corpus":
            counts[token] = max(
                1,
                sum(
                    1
                    for key in (
                        "candidate_probe_objective_spread",
                        "candidate_probe_engine_disagreement_prior",
                        "hard_case_prior",
                        "ambiguity_prior_sample_count",
                    )
                    if _is_nonzero_numeric(row.get(key))
                ),
            )
        else:
            counts[token] = 1
    return counts


def _source_support_map_for_row(row: Mapping[str, Any], source_tokens: Sequence[str]) -> dict[str, float]:
    support: dict[str, float] = {}
    path_count = _path_sufficiency(int(_safe_scalar(row.get("candidate_probe_path_count"), 0.0)))
    family_balance = _family_balance(
        int(_safe_scalar(row.get("candidate_probe_path_count"), 0.0)),
        int(_safe_scalar(row.get("candidate_probe_corridor_family_count"), 0.0)),
    )
    objective_spread = max(0.0, min(1.0, float(_safe_scalar(row.get("candidate_probe_objective_spread"), 0.0))))
    nominal_margin = max(0.0, min(1.0, float(_safe_scalar(row.get("candidate_probe_nominal_margin"), 0.0))))
    engine_prior = max(0.0, min(1.0, float(_safe_scalar(row.get("candidate_probe_engine_disagreement_prior"), 0.0))))
    hard_case_prior = max(0.0, min(1.0, float(_safe_scalar(row.get("hard_case_prior"), 0.0))))
    sample_signal = min(
        1.0,
        math.log1p(max(int(_safe_scalar(row.get("ambiguity_prior_sample_count"), 0.0)), 0))
        / math.log(9.0),
    )
    support_signal = min(1.0, max(int(_safe_scalar(row.get("ambiguity_prior_support_count"), 0.0)), 0) / 4.0)
    ambiguity_support = max(0.0, min(1.0, float(_safe_scalar(row.get("od_ambiguity_support_ratio"), 0.0))))
    for token in source_tokens:
        if token == "routing_graph_probe":
            support[token] = round(
                min(
                    1.0,
                    0.42 * path_count
                    + 0.34 * family_balance
                    + 0.14 * objective_spread
                    + 0.10 * (1.0 - nominal_margin),
                ),
                6,
            )
        elif token == "engine_augmented_probe":
            support[token] = round(
                min(
                    1.0,
                    0.48 * engine_prior
                    + 0.32 * hard_case_prior
                    + 0.20 * max(objective_spread, ambiguity_support),
                ),
                6,
            )
        elif token == "historical_results_bootstrap":
            support[token] = round(
                min(
                    1.0,
                    0.55 * sample_signal + 0.45 * support_signal,
                ),
                6,
            )
        elif token == "repo_local_geometry_backfill":
            support[token] = round(
                min(
                    1.0,
                    0.40 * path_count + 0.26 * family_balance + 0.22 * objective_spread + 0.12 * (1.0 - nominal_margin),
                ),
                6,
            )
        elif token == "existing_corpus":
            support[token] = round(
                min(
                    1.0,
                    0.44 * ambiguity_support + 0.22 * engine_prior + 0.18 * hard_case_prior + 0.16 * sample_signal,
                ),
                6,
            )
        else:
            support[token] = 0.5
    return support


def _source_support_strength(source_support: Mapping[str, Any] | Sequence[Any]) -> float:
    if isinstance(source_support, Mapping):
        values = [max(0.0, min(1.0, _safe_scalar(value, 0.0))) for value in source_support.values()]
    else:
        values = [max(0.0, min(1.0, _safe_scalar(value, 0.0))) for value in source_support]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _fused_ambiguity_sources(row: Mapping[str, Any], *, include_historical: bool = False) -> tuple[list[str], dict[str, int], dict[str, float]]:
    tokens: list[str] = ["routing_graph_probe"]
    if include_historical and (
        _is_nonzero_numeric(row.get("ambiguity_prior_sample_count"))
        or _is_nonzero_numeric(row.get("ambiguity_prior_support_count"))
    ):
        tokens.append("historical_results_bootstrap")
    source_tokens = list(dict.fromkeys(token for token in tokens if token))
    source_counts = _source_mix_counts_for_row(row, source_tokens)
    source_support = _source_support_map_for_row(row, source_tokens)
    return source_tokens, source_counts, source_support


def _candidate_probe_features(routes: list[dict[str, Any]]) -> dict[str, Any]:
    path_count = len(routes)
    distance_values = [_route_distance_km(route) for route in routes]
    duration_values = [_route_duration_s(route) for route in routes]
    cost_values = [_route_cost(route) for route in routes]
    emissions_values = [_route_emissions(route) for route in routes]
    toll_values = [_route_toll_indicator(route) for route in routes]
    family_ids = {_route_family_id(route, ordinal=index) for index, route in enumerate(routes)}
    objective_spread = (
        _normalised_spread(distance_values)
        + _normalised_spread(duration_values)
        + _normalised_spread(cost_values)
        + _normalised_spread(emissions_values)
    ) / 4.0
    toll_disagreement = 0.0
    if toll_values:
        toll_disagreement = 0.0 if all(value == toll_values[0] for value in toll_values) else 1.0
    nominal_margin = _candidate_weighted_margin(routes)
    path_density = _path_sufficiency(path_count)
    top2_duration_gap = _top_two_relative_gap(duration_values)
    top2_cost_gap = _top_two_relative_gap(cost_values)
    top2_emissions_gap = _top_two_relative_gap(emissions_values)
    prior_features = _cheap_prior_features(
        path_count=path_count,
        family_count=len(family_ids),
        objective_spread=objective_spread,
        nominal_margin=nominal_margin,
        toll_disagreement=toll_disagreement,
        top2_duration_gap=top2_duration_gap,
        top2_cost_gap=top2_cost_gap,
        top2_emissions_gap=top2_emissions_gap,
    )
    family_diversity = float(prior_features["candidate_probe_family_diversity"])
    top2_gap_pressure = float(prior_features["candidate_probe_top2_gap_pressure"])
    ambiguity_index = (
        (1.0 - nominal_margin) * 0.32
        + max(0.0, min(1.0, objective_spread)) * 0.22
        + family_diversity * 0.17
        + top2_gap_pressure * 0.19
        + toll_disagreement * 0.05
        + path_density * 0.05
    )
    return {
        "candidate_probe_path_count": path_count,
        "candidate_probe_corridor_family_count": len(family_ids),
        "candidate_probe_distance_spread_km": round(max(distance_values, default=0.0) - min(distance_values, default=0.0), 6),
        "candidate_probe_duration_spread_s": round(max(duration_values, default=0.0) - min(duration_values, default=0.0), 6),
        "candidate_probe_cost_spread": round(max(cost_values, default=0.0) - min(cost_values, default=0.0), 6),
        "candidate_probe_emissions_spread_kg": round(max(emissions_values, default=0.0) - min(emissions_values, default=0.0), 6),
        "candidate_probe_objective_spread": round(objective_spread, 6),
        "candidate_probe_nominal_margin": nominal_margin,
        "candidate_probe_top2_duration_gap": top2_duration_gap,
        "candidate_probe_top2_cost_gap": top2_cost_gap,
        "candidate_probe_top2_emissions_gap": top2_emissions_gap,
        "candidate_probe_toll_disagreement_rate": round(toll_disagreement, 6),
        **prior_features,
        "ambiguity_index": round(max(0.0, min(1.0, ambiguity_index)), 6),
        "od_ambiguity_index": round(max(0.0, min(1.0, ambiguity_index)), 6),
    }


def _ambiguity_source_mix(source_tokens: Sequence[str]) -> str:
    tokens = [str(token).strip() for token in source_tokens if str(token).strip()]
    if not tokens:
        return ""
    return json.dumps({token: 1 for token in sorted(set(tokens))}, sort_keys=True, separators=(",", ":"))


def _ambiguity_source_mix_counts(raw_source_mix: Any) -> dict[str, int]:
    if raw_source_mix in (None, ""):
        return {}
    if isinstance(raw_source_mix, Mapping):
        counts: dict[str, int] = {}
        for key, value in raw_source_mix.items():
            token = str(key).strip()
            if not token:
                continue
            counts[token] = counts.get(token, 0) + max(1, int(_safe_scalar(value, 1.0)))
        return counts
    if isinstance(raw_source_mix, str):
        text = raw_source_mix.strip()
        if not text:
            return {}
        if text[:1] in {"{", "["}:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            else:
                if isinstance(parsed, Mapping):
                    return _ambiguity_source_mix_counts(parsed)
                if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
                    return _ambiguity_source_mix_counts({str(item): 1 for item in parsed if str(item).strip()})
        tokens = [part.strip() for part in text.replace("+", ",").split(",") if part.strip()]
        return {token: tokens.count(token) for token in sorted(set(tokens))}
    if isinstance(raw_source_mix, Sequence) and not isinstance(raw_source_mix, (str, bytes)):
        return _ambiguity_source_mix_counts({str(item): 1 for item in raw_source_mix if str(item).strip()})
    return {}


def _ambiguity_source_entropy(source_counts: Mapping[str, int]) -> float:
    counts = [max(0, int(value)) for value in source_counts.values() if max(0, int(value)) > 0]
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counts:
        probability = count / total
        entropy -= probability * math.log(probability)
    max_entropy = math.log(len(counts))
    if max_entropy <= 0.0:
        return 0.0
    return round(max(0.0, min(1.0, entropy / max_entropy)), 6)


def _ambiguity_derived_fields(
    *,
    ambiguity_index: float,
    ambiguity_confidence: float,
    source_count: int,
    source_mix: Any,
    sample_count: int,
    support_count: int,
    path_count: int,
    family_count: int,
    objective_spread: float,
    nominal_margin: float,
    toll_disagreement: float,
    explicit_prior_strength: float | None = None,
) -> dict[str, float | int]:
    source_counts = _ambiguity_source_mix_counts(source_mix)
    effective_source_count = max(int(source_count), len(source_counts))
    observed_support = (
        max(0.0, min(1.0, float(max(0, int(support_count))) / float(max(1, int(sample_count)))))
        if sample_count > 0
        else 0.0
    )
    source_support = max(0.0, min(1.0, float(effective_source_count) / 3.0))
    family_density = (
        max(0.0, min(1.0, float(max(0, int(family_count))) / float(max(1, int(path_count)))))
        if path_count > 0
        else 0.0
    )
    margin_pressure = max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, float(nominal_margin)))))
    spread_pressure = max(0.0, min(1.0, float(objective_spread)))
    toll_instability = max(0.0, min(1.0, float(toll_disagreement)))
    support_ratio = max(
        0.0,
        min(
            1.0,
            0.40 * observed_support
            + 0.22 * source_support
            + 0.18 * max(0.0, min(1.0, float(ambiguity_confidence)))
            + 0.12 * family_density
            + 0.08 * spread_pressure,
        ),
    )
    if explicit_prior_strength is not None and math.isfinite(float(explicit_prior_strength)):
        prior_strength = max(0.0, min(1.0, float(explicit_prior_strength)))
    else:
        prior_strength = max(0.0, min(1.0, float(ambiguity_index)))
    return {
        "od_ambiguity_support_ratio": round(support_ratio, 6),
        "od_ambiguity_source_mix_count": int(effective_source_count),
        "od_ambiguity_source_entropy": _ambiguity_source_entropy(source_counts),
        "od_ambiguity_prior_strength": round(prior_strength, 6),
        "od_ambiguity_family_density": round(family_density, 6),
        "od_ambiguity_margin_pressure": round(margin_pressure, 6),
        "od_ambiguity_spread_pressure": round(spread_pressure, 6),
        "od_ambiguity_toll_instability": round(toll_instability, 6),
    }


def _ambiguity_confidence(
    *,
    source_tokens: Sequence[str],
    sample_count: int,
    support_count: int,
    ambiguity_nonzero: bool,
) -> float:
    tokens = [str(token).strip() for token in source_tokens if str(token).strip()]
    if not tokens:
        return 0.0
    base_source = max(AMBIGUITY_SOURCE_CONFIDENCE_WEIGHTS.get(token, 0.4) for token in tokens)
    source_bonus = min(1.0, len(set(tokens)) / 3.0)
    sample_signal = min(1.0, math.log1p(max(int(sample_count), 0)) / math.log(9.0))
    support_signal = min(1.0, max(int(support_count), 0) / 4.0)
    confidence = (
        0.45 * base_source
        + 0.15 * source_bonus
        + 0.20 * sample_signal
        + 0.20 * support_signal
    )
    if not ambiguity_nonzero:
        confidence *= 0.85
    return round(min(1.0, max(0.0, confidence)), 6)


def _routing_graph_prior_metadata(
    *,
    candidate_features: Mapping[str, Any],
    ambiguity_index: float,
    generated_paths: int,
    emitted_paths: int,
    path_count: int,
    accepted: bool,
) -> dict[str, Any]:
    source_tokens, source_counts, source_support = _fused_ambiguity_sources(candidate_features)
    sample_count = max(int(generated_paths), int(path_count), 1 if accepted else 0)
    support_count = max(int(emitted_paths), int(path_count), 1 if accepted else 0)
    ambiguity_nonzero = max(0.0, float(ambiguity_index)) > 0.0
    return {
        "ambiguity_prior_source": ",".join(source_tokens),
        "ambiguity_prior_sample_count": int(sample_count),
        "ambiguity_prior_support_count": int(support_count),
        "ambiguity_prior_nonzero": bool(ambiguity_nonzero),
        "od_ambiguity_source_count": len(source_tokens),
        "od_ambiguity_source_mix": _source_mix_string(source_counts),
        "od_ambiguity_source_support": _source_mix_string(source_support),
        "od_ambiguity_source_support_strength": _source_support_strength(source_support),
        "od_ambiguity_confidence": _ambiguity_confidence(
            source_tokens=source_tokens,
            sample_count=sample_count,
            support_count=support_count,
            ambiguity_nonzero=ambiguity_nonzero,
        ),
    }


def _candidate_probe_payload(
    *,
    origin: dict[str, float],
    destination: dict[str, float],
    feasibility_result: dict[str, Any],
    candidate_probe_fn: Callable[..., tuple[list[dict[str, Any]], GraphCandidateDiagnostics]],
    max_paths: int,
) -> dict[str, Any]:
    start_node_id = str(feasibility_result.get("origin_node_id") or "").strip() or None
    goal_node_id = str(feasibility_result.get("destination_node_id") or "").strip() or None
    routes, diagnostics = candidate_probe_fn(
        origin_lat=float(origin["lat"]),
        origin_lon=float(origin["lon"]),
        destination_lat=float(destination["lat"]),
        destination_lon=float(destination["lon"]),
        max_paths=max(1, int(max_paths)),
        start_node_id=start_node_id,
        goal_node_id=goal_node_id,
    )
    emitted_paths = int(max(0, getattr(diagnostics, "emitted_paths", 0)))
    generated_paths = int(max(0, getattr(diagnostics, "generated_paths", 0)))
    explored_states = int(max(0, getattr(diagnostics, "explored_states", 0)))
    candidate_budget = int(max(0, getattr(diagnostics, "candidate_budget", 0)))
    effective_max_hops = int(max(0, getattr(diagnostics, "effective_max_hops", 0)))
    effective_state_budget = int(max(0, getattr(diagnostics, "effective_state_budget", 0)))
    reason_code = str(getattr(diagnostics, "no_path_reason", "") or "ok").strip() or "ok"
    message = str(getattr(diagnostics, "no_path_detail", "") or "").strip()
    accepted = bool(routes) and emitted_paths > 0
    if accepted:
        reason_code = "ok"
        message = f"candidate_probe_ok paths={emitted_paths}"
    elif not message:
        message = reason_code
    features = _candidate_probe_features([route for route in routes if isinstance(route, dict)]) if routes else {
        "candidate_probe_path_count": 0,
        "candidate_probe_corridor_family_count": 0,
        "candidate_probe_distance_spread_km": 0.0,
        "candidate_probe_duration_spread_s": 0.0,
        "candidate_probe_cost_spread": 0.0,
        "candidate_probe_emissions_spread_kg": 0.0,
        "candidate_probe_objective_spread": 0.0,
        "candidate_probe_nominal_margin": 0.0,
        "candidate_probe_top2_duration_gap": 0.0,
        "candidate_probe_top2_cost_gap": 0.0,
        "candidate_probe_top2_emissions_gap": 0.0,
        "candidate_probe_toll_disagreement_rate": 0.0,
        "candidate_probe_family_diversity": 0.0,
        "candidate_probe_path_sufficiency": 0.0,
        "candidate_probe_top2_gap_pressure": 0.0,
        "candidate_probe_engine_disagreement_prior": 0.0,
        "hard_case_prior": 0.0,
        "ambiguity_index": 0.0,
        "od_ambiguity_index": 0.0,
    }
    prior_metadata = _routing_graph_prior_metadata(
        candidate_features=features,
        ambiguity_index=float(features.get("od_ambiguity_index", 0.0)),
        generated_paths=generated_paths,
        emitted_paths=emitted_paths,
        path_count=int(features.get("candidate_probe_path_count", 0) or 0),
        accepted=accepted,
    )
    derived_fields = _ambiguity_derived_fields(
        ambiguity_index=float(features.get("od_ambiguity_index", 0.0)),
        ambiguity_confidence=float(prior_metadata.get("od_ambiguity_confidence", 0.0)),
        source_count=int(prior_metadata.get("od_ambiguity_source_count", 0) or 0),
        source_mix=prior_metadata.get("od_ambiguity_source_mix"),
        sample_count=int(prior_metadata.get("ambiguity_prior_sample_count", 0) or 0),
        support_count=int(prior_metadata.get("ambiguity_prior_support_count", 0) or 0),
        path_count=int(features.get("candidate_probe_path_count", 0) or 0),
        family_count=int(features.get("candidate_probe_corridor_family_count", 0) or 0),
        objective_spread=float(features.get("candidate_probe_objective_spread", 0.0) or 0.0),
        nominal_margin=float(features.get("candidate_probe_nominal_margin", 0.0) or 0.0),
        toll_disagreement=float(features.get("candidate_probe_toll_disagreement_rate", 0.0) or 0.0),
        explicit_prior_strength=_safe_scalar(
            features.get("od_ambiguity_prior_strength", features.get("ambiguity_prior_strength")),
            float("nan"),
        ),
    )
    ambiguity_budget_prior = round(
        max(
            float(features.get("od_ambiguity_index", 0.0)),
            float(features.get("candidate_probe_engine_disagreement_prior", 0.0)),
            float(features.get("hard_case_prior", 0.0)),
        ),
        6,
    )
    return {
        "candidate_probe_accepted": accepted,
        "candidate_probe_reason_code": reason_code,
        "candidate_probe_message": message,
        "candidate_probe_emitted_paths": emitted_paths,
        "candidate_probe_generated_paths": generated_paths,
        "candidate_probe_explored_states": explored_states,
        "candidate_probe_candidate_budget": candidate_budget,
        "candidate_probe_effective_max_hops": effective_max_hops,
        "candidate_probe_effective_state_budget": effective_state_budget,
        **features,
        **prior_metadata,
        **derived_fields,
        "ambiguity_budget_prior": ambiguity_budget_prior,
        "ambiguity_budget_prior_gap": round(
            max(0.0, ambiguity_budget_prior - float(features.get("od_ambiguity_index", 0.0))),
            6,
        ),
        "budget_prior_exceeds_raw": bool(
            ambiguity_budget_prior > float(features.get("od_ambiguity_index", 0.0)) + 1e-9
        ),
    }


def _bootstrap_selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= 1:
        return [dict(row) for row in rows]
    ambiguity_values = [
        _safe_scalar(row.get("od_ambiguity_index", row.get("ambiguity_index")), float("nan"))
        for row in rows
    ]
    ambiguity_values = [value for value in ambiguity_values if math.isfinite(value)]
    path_counts = [int(_safe_scalar(row.get("candidate_probe_path_count"), 0.0)) for row in rows]
    family_counts = [int(_safe_scalar(row.get("candidate_probe_corridor_family_count"), 0.0)) for row in rows]
    objective_spreads = [_safe_scalar(row.get("candidate_probe_objective_spread"), float("nan")) for row in rows]
    objective_spreads = [value for value in objective_spreads if math.isfinite(value)]
    nominal_margins = [_safe_scalar(row.get("candidate_probe_nominal_margin"), float("nan")) for row in rows]
    nominal_margins = [value for value in nominal_margins if math.isfinite(value)]
    engine_priors = [_safe_scalar(row.get("candidate_probe_engine_disagreement_prior"), float("nan")) for row in rows]
    engine_priors = [value for value in engine_priors if math.isfinite(value)]
    hard_case_priors = [_safe_scalar(row.get("hard_case_prior"), float("nan")) for row in rows]
    hard_case_priors = [value for value in hard_case_priors if math.isfinite(value)]
    bootstrap_payload = {
        "ambiguity_index": round(sum(ambiguity_values) / len(ambiguity_values), 6) if ambiguity_values else 0.0,
        "od_ambiguity_index": round(sum(ambiguity_values) / len(ambiguity_values), 6) if ambiguity_values else 0.0,
        "od_ambiguity_prior_strength": round(sum(ambiguity_values) / len(ambiguity_values), 6) if ambiguity_values else 0.0,
        "candidate_probe_path_count": max(path_counts, default=0),
        "candidate_probe_corridor_family_count": max(family_counts, default=0),
        "candidate_probe_objective_spread": round(sum(objective_spreads) / len(objective_spreads), 6) if objective_spreads else 0.0,
        "candidate_probe_nominal_margin": round(sum(nominal_margins) / len(nominal_margins), 6) if nominal_margins else 0.0,
        "candidate_probe_engine_disagreement_prior": round(max(engine_priors, default=0.0), 6),
        "hard_case_prior": round(max(hard_case_priors, default=0.0), 6),
        "ambiguity_prior_source": "historical_results_bootstrap",
        "ambiguity_prior_sample_count": len(rows),
        "ambiguity_prior_support_count": sum(
            1
            for value in (
                max(path_counts, default=0),
                max(family_counts, default=0),
                max(objective_spreads, default=0.0),
                max(nominal_margins, default=0.0),
                max(engine_priors, default=0.0),
                max(hard_case_priors, default=0.0),
            )
            if _is_nonzero_numeric(value)
        ),
    }
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["od_ambiguity_index"] = max(
            _safe_scalar(enriched.get("od_ambiguity_index"), 0.0),
            _safe_scalar(bootstrap_payload["od_ambiguity_index"], 0.0),
        )
        enriched["ambiguity_index"] = enriched["od_ambiguity_index"]
        merged_source = _merge_prior_sources(enriched.get("ambiguity_prior_source"), bootstrap_payload["ambiguity_prior_source"])
        enriched["ambiguity_prior_source"] = merged_source
        enriched["ambiguity_prior_sample_count"] = max(
            int(_safe_scalar(enriched.get("ambiguity_prior_sample_count"), 0.0)),
            int(bootstrap_payload["ambiguity_prior_sample_count"]),
        )
        enriched["ambiguity_prior_support_count"] = max(
            int(_safe_scalar(enriched.get("ambiguity_prior_support_count"), 0.0)),
            int(bootstrap_payload["ambiguity_prior_support_count"]),
        )
        source_tokens = _split_prior_sources(enriched["ambiguity_prior_source"])
        source_counts = _source_mix_counts_for_row(enriched, source_tokens)
        source_support = _source_support_map_for_row(enriched, source_tokens)
        enriched["od_ambiguity_source_count"] = len(source_tokens)
        enriched["od_ambiguity_source_mix"] = _source_mix_string(source_counts)
        enriched["od_ambiguity_source_support"] = _source_mix_string(source_support)
        enriched["od_ambiguity_source_support_strength"] = _source_support_strength(source_support)
        enriched["od_ambiguity_confidence"] = _ambiguity_confidence(
            source_tokens=source_tokens,
            sample_count=int(_safe_scalar(enriched.get("ambiguity_prior_sample_count"), 0.0)),
            support_count=int(_safe_scalar(enriched.get("ambiguity_prior_support_count"), 0.0)),
            ambiguity_nonzero=_is_nonzero_numeric(enriched.get("od_ambiguity_index")),
        )
        enriched.update(
            _ambiguity_derived_fields(
                ambiguity_index=_safe_scalar(enriched.get("od_ambiguity_index"), 0.0),
                ambiguity_confidence=_safe_scalar(enriched.get("od_ambiguity_confidence"), 0.0),
                source_count=int(_safe_scalar(enriched.get("od_ambiguity_source_count"), 0.0)),
                source_mix=enriched.get("od_ambiguity_source_mix"),
                sample_count=int(_safe_scalar(enriched.get("ambiguity_prior_sample_count"), 0.0)),
                support_count=int(_safe_scalar(enriched.get("ambiguity_prior_support_count"), 0.0)),
                path_count=int(_safe_scalar(enriched.get("candidate_probe_path_count"), 0.0)),
                family_count=int(_safe_scalar(enriched.get("candidate_probe_corridor_family_count"), 0.0)),
                objective_spread=_safe_scalar(enriched.get("candidate_probe_objective_spread"), 0.0),
                nominal_margin=_safe_scalar(enriched.get("candidate_probe_nominal_margin"), 0.0),
                toll_disagreement=_safe_scalar(enriched.get("candidate_probe_toll_disagreement_rate"), 0.0),
                explicit_prior_strength=(
                    _safe_scalar(enriched.get("od_ambiguity_prior_strength"), float("nan"))
                    if enriched.get("od_ambiguity_prior_strength") not in (None, "")
                    else None
                ),
            )
        )
        enriched_rows.append(enriched)
    return enriched_rows


def _feasibility_result_to_row(
    *,
    od_id: str,
    sample_index: int,
    origin: dict[str, float],
    destination: dict[str, float],
    distance_km: float,
    accepted: bool,
    feasibility_result: dict[str, Any] | None,
    reason_code: str,
    acceptance_mode: str = "feasibility_only",
    candidate_probe_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = feasibility_result or {}
    row = {
        "od_id": od_id,
        "sample_index": int(sample_index),
        "origin_lat": round(float(origin["lat"]), 6),
        "origin_lon": round(float(origin["lon"]), 6),
        "destination_lat": round(float(destination["lat"]), 6),
        "destination_lon": round(float(destination["lon"]), 6),
        "straight_line_km": round(float(distance_km), 6),
        "distance_bin": _distance_bin(distance_km),
        "bin_index": _bin_index(distance_km),
        "origin_region_bucket": _region_bucket(float(origin["lat"])),
        "destination_region_bucket": _region_bucket(float(destination["lat"])),
        "corridor_bucket": _corridor_bucket(origin, destination),
        "acceptance_mode": str(acceptance_mode),
        "accepted": bool(accepted),
        "reason_code": reason_code,
        "origin_node_id": str(result.get("origin_node_id", "")),
        "destination_node_id": str(result.get("destination_node_id", "")),
        "origin_nearest_distance_m": round(_safe_distance_m(result.get("origin_nearest_distance_m")), 3),
        "destination_nearest_distance_m": round(_safe_distance_m(result.get("destination_nearest_distance_m")), 3),
        "candidate_probe_accepted": False,
        "candidate_probe_reason_code": "",
        "candidate_probe_message": "",
        "candidate_probe_emitted_paths": 0,
        "candidate_probe_generated_paths": 0,
        "candidate_probe_explored_states": 0,
        "candidate_probe_candidate_budget": 0,
        "candidate_probe_effective_max_hops": 0,
        "candidate_probe_effective_state_budget": 0,
        "candidate_probe_path_count": 0,
        "candidate_probe_corridor_family_count": 0,
        "candidate_probe_distance_spread_km": 0.0,
        "candidate_probe_duration_spread_s": 0.0,
        "candidate_probe_cost_spread": 0.0,
        "candidate_probe_emissions_spread_kg": 0.0,
        "candidate_probe_objective_spread": 0.0,
        "candidate_probe_nominal_margin": 0.0,
        "candidate_probe_toll_disagreement_rate": 0.0,
        "candidate_probe_family_diversity": 0.0,
        "candidate_probe_engine_disagreement_prior": 0.0,
        "hard_case_prior": 0.0,
        "ambiguity_index": 0.0,
        "od_ambiguity_index": 0.0,
        "od_ambiguity_confidence": 0.0,
        "od_ambiguity_source_count": 0,
        "od_ambiguity_source_mix": "",
        "ambiguity_prior_source": "",
        "ambiguity_prior_sample_count": 0,
        "ambiguity_prior_support_count": 0,
        "ambiguity_prior_nonzero": False,
        "od_ambiguity_source_support": "",
        "od_ambiguity_source_support_strength": 0.0,
    }
    if candidate_probe_result:
        row.update(candidate_probe_result)
    if accepted:
        row["route_graph_message"] = str(result.get("message", "ok"))
    else:
        row["route_graph_message"] = str(
            (candidate_probe_result or {}).get("candidate_probe_message")
            or result.get("message")
            or reason_code
        )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "od_id",
        "sample_index",
        "origin_lat",
        "origin_lon",
        "destination_lat",
        "destination_lon",
        "straight_line_km",
        "distance_bin",
        "bin_index",
        "origin_region_bucket",
        "destination_region_bucket",
        "corridor_bucket",
        "acceptance_mode",
        "accepted",
        "reason_code",
        "route_graph_message",
        "origin_node_id",
        "destination_node_id",
        "origin_nearest_distance_m",
        "destination_nearest_distance_m",
        "candidate_probe_accepted",
        "candidate_probe_reason_code",
        "candidate_probe_message",
        "candidate_probe_emitted_paths",
        "candidate_probe_generated_paths",
        "candidate_probe_explored_states",
        "candidate_probe_candidate_budget",
        "candidate_probe_effective_max_hops",
        "candidate_probe_effective_state_budget",
        "candidate_probe_path_count",
        "candidate_probe_corridor_family_count",
        "candidate_probe_distance_spread_km",
        "candidate_probe_duration_spread_s",
        "candidate_probe_cost_spread",
        "candidate_probe_emissions_spread_kg",
        "candidate_probe_objective_spread",
        "candidate_probe_nominal_margin",
        "candidate_probe_toll_disagreement_rate",
        "candidate_probe_family_diversity",
        "candidate_probe_engine_disagreement_prior",
        "hard_case_prior",
        "ambiguity_index",
        "od_ambiguity_index",
        "od_ambiguity_confidence",
        "od_ambiguity_source_count",
        "od_ambiguity_source_mix",
        "od_ambiguity_source_mix_count",
        "od_ambiguity_source_support",
        "od_ambiguity_source_support_strength",
        "od_ambiguity_source_entropy",
        "od_ambiguity_support_ratio",
        "od_ambiguity_prior_strength",
        "od_ambiguity_family_density",
        "od_ambiguity_margin_pressure",
        "od_ambiguity_spread_pressure",
        "od_ambiguity_toll_instability",
        "ambiguity_prior_source",
        "ambiguity_prior_sample_count",
        "ambiguity_prior_support_count",
        "ambiguity_prior_nonzero",
        "corpus_kind",
        "selection_rank",
        "selection_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_od_corpus(
    *,
    seed: int,
    pair_count: int,
    bbox: UKBBox,
    max_attempts: int,
    feasibility_fn: Callable[..., dict[str, Any]] = route_graph_od_feasibility,
    candidate_probe_fn: Callable[..., tuple[list[dict[str, Any]], GraphCandidateDiagnostics]] = route_graph_candidate_routes,
    acceptance_mode: str = "graph_candidates",
    probe_max_paths: int = 1,
) -> dict[str, Any]:
    mode = str(acceptance_mode).strip().lower() or "graph_candidates"
    if mode not in {"feasibility_only", "graph_candidates"}:
        raise ValueError("acceptance_mode must be 'feasibility_only' or 'graph_candidates'")
    rng = random.Random(int(seed))
    target_count = max(1, int(pair_count))
    bin_targets = _split_evenly(target_count, len(DISTANCE_BINS))
    accepted_by_bin = [0 for _ in DISTANCE_BINS]
    accepted_by_corridor: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    rejected_preview: list[dict[str, Any]] = []
    reject_stats: dict[str, int] = {}
    accepted_count = 0

    for sample_index in range(max(1, int(max_attempts))):
        if accepted_count >= target_count:
            break
        origin, destination = _sample_candidate_pair(rng, bbox)
        distance_km = _haversine_km(
            float(origin["lat"]),
            float(origin["lon"]),
            float(destination["lat"]),
            float(destination["lon"]),
        )
        bin_idx = _bin_index(distance_km)
        od_id = f"od-{sample_index:06d}"
        if accepted_by_bin[bin_idx] >= bin_targets[bin_idx]:
            reject_stats["bin_quota_full"] = reject_stats.get("bin_quota_full", 0) + 1
            if len(rejected_preview) < 128:
                rejected_preview.append(
                    _feasibility_result_to_row(
                        od_id=od_id,
                        sample_index=sample_index,
                        origin=origin,
                        destination=destination,
                        distance_km=distance_km,
                        accepted=False,
                        feasibility_result={"message": "bin_quota_full"},
                        reason_code="bin_quota_full",
                        acceptance_mode=mode,
                    )
                )
            continue

        result = feasibility_fn(
            origin_lat=float(origin["lat"]),
            origin_lon=float(origin["lon"]),
            destination_lat=float(destination["lat"]),
            destination_lon=float(destination["lon"]),
        )
        feasible = bool(result.get("ok"))
        reason_code = str(result.get("reason_code", "routing_graph_unavailable") if not feasible else "ok")
        if not feasible:
            reject_stats[reason_code] = reject_stats.get(reason_code, 0) + 1
            if len(rejected_preview) < 128:
                rejected_preview.append(
                    _feasibility_result_to_row(
                        od_id=od_id,
                        sample_index=sample_index,
                        origin=origin,
                        destination=destination,
                        distance_km=distance_km,
                        accepted=False,
                        feasibility_result=result,
                        reason_code=reason_code,
                        acceptance_mode=mode,
                    )
                )
            continue

        candidate_probe_result: dict[str, Any] | None = None
        if mode == "graph_candidates":
            candidate_probe_result = _candidate_probe_payload(
                origin=origin,
                destination=destination,
                feasibility_result=result,
                candidate_probe_fn=candidate_probe_fn,
                max_paths=max(1, int(probe_max_paths)),
            )
            probe_ok = bool(candidate_probe_result["candidate_probe_accepted"])
            if not probe_ok:
                probe_reason = str(candidate_probe_result["candidate_probe_reason_code"] or "routing_graph_no_path")
                reject_stats[probe_reason] = reject_stats.get(probe_reason, 0) + 1
                if len(rejected_preview) < 128:
                    rejected_preview.append(
                        _feasibility_result_to_row(
                            od_id=od_id,
                            sample_index=sample_index,
                            origin=origin,
                            destination=destination,
                            distance_km=distance_km,
                            accepted=False,
                            feasibility_result=result,
                            reason_code=probe_reason,
                            acceptance_mode=mode,
                            candidate_probe_result=candidate_probe_result,
                        )
                    )
                continue

        row = _feasibility_result_to_row(
            od_id=od_id,
            sample_index=sample_index,
            origin=origin,
            destination=destination,
            distance_km=distance_km,
            accepted=True,
            feasibility_result=result,
            reason_code="ok",
            acceptance_mode=mode,
            candidate_probe_result=candidate_probe_result,
        )
        rows.append(row)
        accepted_by_bin[bin_idx] += 1
        corridor_bucket = str(row["corridor_bucket"])
        accepted_by_corridor[corridor_bucket] = accepted_by_corridor.get(corridor_bucket, 0) + 1
        accepted_count += 1

    rows = _bootstrap_selected_rows(rows)
    complete = accepted_count >= target_count
    accept_rate = accepted_count / float(max(1, int(max_attempts)))
    rows_hash = _row_hash(rows)
    bin_distribution = {
        DISTANCE_BINS[idx][2]: int(accepted_by_bin[idx])
        for idx in range(len(DISTANCE_BINS))
    }
    summary = {
        "schema_version": "1.0.0",
        "created_at_utc": _utc_now_iso(),
        "seed": int(seed),
        "pair_count": int(pair_count),
        "target_count": target_count,
        "max_attempts": int(max_attempts),
        "acceptance_mode": mode,
        "probe_max_paths": int(max(1, int(probe_max_paths))),
        "bbox": {
            "south": bbox.south,
            "north": bbox.north,
            "west": bbox.west,
            "east": bbox.east,
        },
        "accepted_count": accepted_count,
        "rejected_count": int(max(0, int(max_attempts) - accepted_count)),
        "complete": complete,
        "accept_rate": round(float(accept_rate), 6),
        "accepted_by_bin": bin_distribution,
        "accepted_by_corridor": dict(sorted(accepted_by_corridor.items())),
        "reject_stats": dict(sorted(reject_stats.items())),
        "rejected_samples_preview": rejected_preview,
        "corpus_hash": rows_hash,
        "distance_bins": [
            {"label": label, "lower_km": lower, "upper_km": upper}
            for lower, upper, label in DISTANCE_BINS
        ],
        "rows": rows,
    }
    return summary


def _selected_feature_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "mean_ambiguity_index": 0.0,
            "max_ambiguity_index": 0.0,
            "mean_candidate_paths": 0.0,
            "mean_objective_spread": 0.0,
            "mean_nominal_margin": 0.0,
            "mean_toll_disagreement_rate": 0.0,
            "mean_engine_disagreement_prior": 0.0,
            "mean_hard_case_prior": 0.0,
            "mean_prior_strength": 0.0,
            "mean_budget_prior": 0.0,
            "mean_support_ratio": 0.0,
            "mean_source_mix_count": 0.0,
            "mean_source_support_strength": 0.0,
            "mean_source_entropy": 0.0,
        }
    def _mean(key: str) -> float:
        values = [_safe_scalar(row.get(key), float("nan")) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        return round(sum(values) / len(values), 6) if values else 0.0

    return {
        "mean_ambiguity_index": _mean("ambiguity_index"),
        "max_ambiguity_index": round(max(_safe_scalar(row.get("ambiguity_index"), 0.0) for row in rows), 6),
        "mean_candidate_paths": _mean("candidate_probe_path_count"),
        "mean_objective_spread": _mean("candidate_probe_objective_spread"),
        "mean_nominal_margin": _mean("candidate_probe_nominal_margin"),
        "mean_toll_disagreement_rate": _mean("candidate_probe_toll_disagreement_rate"),
        "mean_engine_disagreement_prior": _mean("candidate_probe_engine_disagreement_prior"),
        "mean_hard_case_prior": _mean("hard_case_prior"),
        "mean_prior_strength": _mean("od_ambiguity_prior_strength"),
        "mean_budget_prior": _mean("ambiguity_budget_prior"),
        "mean_support_ratio": _mean("od_ambiguity_support_ratio"),
        "mean_source_mix_count": _mean("od_ambiguity_source_mix_count"),
        "mean_source_support_strength": _mean("od_ambiguity_source_support_strength"),
        "mean_source_entropy": _mean("od_ambiguity_source_entropy"),
    }


def _corpus_ambiguity_rank_score(row: dict[str, Any]) -> float:
    budget_prior = _safe_scalar(row.get("ambiguity_budget_prior"), _safe_scalar(row.get("od_ambiguity_index"), 0.0))
    ambiguity_index = _safe_scalar(row.get("od_ambiguity_index", row.get("ambiguity_index")), 0.0)
    support_ratio = _safe_scalar(row.get("od_ambiguity_support_ratio"), 0.0)
    source_support_strength = _safe_scalar(row.get("od_ambiguity_source_support_strength"), 0.0)
    source_entropy = _safe_scalar(row.get("od_ambiguity_source_entropy"), 0.0)
    hard_case_prior = _safe_scalar(row.get("hard_case_prior"), 0.0)
    engine_prior = _safe_scalar(row.get("candidate_probe_engine_disagreement_prior"), 0.0)
    path_support = _path_sufficiency(int(_safe_scalar(row.get("candidate_probe_path_count"), 0.0)))
    return round(
        max(
            0.0,
            min(
                1.0,
                0.34 * budget_prior
                + 0.18 * ambiguity_index
                + 0.16 * hard_case_prior
                + 0.12 * engine_prior
                + 0.10 * support_ratio
                + 0.06 * source_entropy
                + 0.04 * path_support
                + 0.02 * source_support_strength,
            ),
        ),
        6,
    )


def _select_rows_for_corpus(
    pool_rows: list[dict[str, Any]],
    *,
    count: int,
    corpus_kind: str,
) -> list[dict[str, Any]]:
    target = max(0, int(count))
    if target <= 0:
        return []
    rows_by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        rows_by_bin[str(row.get("distance_bin") or DISTANCE_BINS[-1][2])].append(dict(row))
    quotas = _split_evenly(target, len(DISTANCE_BINS))
    corridor_frequency = {
        str(row.get("corridor_bucket") or ""): sum(1 for candidate in pool_rows if candidate.get("corridor_bucket") == row.get("corridor_bucket"))
        for row in pool_rows
    }

    def _amb_score(row: dict[str, Any]) -> float:
        return _corpus_ambiguity_rank_score(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for idx, (_, _, bin_label) in enumerate(DISTANCE_BINS):
        candidates = rows_by_bin.get(bin_label, [])
        if not candidates:
            continue
        if corpus_kind == "ambiguous":
            ordered = sorted(
                candidates,
                key=lambda row: (
                    -_amb_score(row),
                    -_safe_scalar(row.get("candidate_probe_path_count"), 0.0),
                    -_safe_scalar(row.get("candidate_probe_objective_spread"), 0.0),
                    str(row.get("od_id") or ""),
                ),
            )
        else:
            ambiguity_values = sorted(_amb_score(row) for row in candidates)
            median = ambiguity_values[len(ambiguity_values) // 2] if ambiguity_values else 0.0
            ordered = sorted(
                candidates,
                key=lambda row: (
                    abs(_amb_score(row) - median),
                    corridor_frequency.get(str(row.get("corridor_bucket") or ""), 0),
                    str(row.get("od_id") or ""),
                ),
            )
        for row in ordered:
            if len([item for item in selected if item.get("distance_bin") == bin_label]) >= quotas[idx]:
                break
            row_id = str(row.get("od_id") or "")
            if row_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id)
            if len(selected) >= target:
                break
        if len(selected) >= target:
            break

    if len(selected) < target:
        if corpus_kind == "ambiguous":
            fallback = sorted(pool_rows, key=lambda row: (-_amb_score(row), str(row.get("od_id") or "")))
        else:
            global_values = sorted(_amb_score(row) for row in pool_rows)
            global_median = global_values[len(global_values) // 2] if global_values else 0.0
            fallback = sorted(
                pool_rows,
                key=lambda row: (
                    abs(_amb_score(row) - global_median),
                    corridor_frequency.get(str(row.get("corridor_bucket") or ""), 0),
                    str(row.get("od_id") or ""),
                ),
            )
        for row in fallback:
            row_id = str(row.get("od_id") or "")
            if row_id in selected_ids:
                continue
            selected.append(dict(row))
            selected_ids.add(row_id)
            if len(selected) >= target:
                break

    enriched: list[dict[str, Any]] = []
    representative_centre = (
        round(sum(_amb_score(row) for row in selected) / float(len(selected)), 6)
        if selected
        else 0.0
    )
    for rank, row in enumerate(selected, start=1):
        enriched_row = dict(row)
        enriched_row["corpus_kind"] = corpus_kind
        enriched_row["selection_rank"] = rank
        enriched_row["selection_score"] = round(
            _amb_score(row)
            if corpus_kind == "ambiguous"
            else abs(_amb_score(row) - representative_centre),
            6,
        )
        enriched.append(enriched_row)
    return enriched


def _selected_corpus_summary(
    *,
    pool_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    corpus_kind: str,
    selection_policy: str,
) -> dict[str, Any]:
    accepted_by_bin: dict[str, int] = defaultdict(int)
    accepted_by_corridor: dict[str, int] = defaultdict(int)
    for row in rows:
        accepted_by_bin[str(row.get("distance_bin") or DISTANCE_BINS[-1][2])] += 1
        accepted_by_corridor[str(row.get("corridor_bucket") or "")] += 1
    return {
        "schema_version": "2.0.0",
        "created_at_utc": _utc_now_iso(),
        "seed": int(pool_summary.get("seed") or 0),
        "corpus_kind": corpus_kind,
        "selection_policy": selection_policy,
        "source_acceptance_mode": pool_summary.get("acceptance_mode"),
        "source_pool_hash": str(pool_summary.get("corpus_hash") or ""),
        "source_pool_count": int(pool_summary.get("accepted_count") or 0),
        "selected_count": len(rows),
        "complete": len(rows) > 0,
        "accepted_by_bin": dict(sorted(accepted_by_bin.items())),
        "accepted_by_corridor": dict(sorted(accepted_by_corridor.items())),
        "ambiguity_feature_stats": _selected_feature_stats(rows),
        "distance_bins": pool_summary.get("distance_bins", []),
        "rows": rows,
    }


def build_dual_od_corpora(
    *,
    seed: int,
    representative_count: int,
    ambiguous_count: int,
    bbox: UKBBox,
    max_attempts: int,
    feasibility_fn: Callable[..., dict[str, Any]] = route_graph_od_feasibility,
    candidate_probe_fn: Callable[..., tuple[list[dict[str, Any]], GraphCandidateDiagnostics]] = route_graph_candidate_routes,
    acceptance_mode: str = "graph_candidates",
    probe_max_paths: int = 6,
) -> dict[str, Any]:
    rep_target = max(1, int(representative_count))
    amb_target = max(1, int(ambiguous_count))
    pool_target = max(rep_target + amb_target, max(rep_target, amb_target) * 2)
    pool_summary = build_od_corpus(
        seed=seed,
        pair_count=pool_target,
        bbox=bbox,
        max_attempts=max_attempts,
        feasibility_fn=feasibility_fn,
        candidate_probe_fn=candidate_probe_fn,
        acceptance_mode=acceptance_mode,
        probe_max_paths=max(2, int(probe_max_paths)),
    )
    accepted_rows = [dict(row) for row in pool_summary.get("rows", [])]
    representative_rows = _select_rows_for_corpus(
        accepted_rows,
        count=min(rep_target, len(accepted_rows)),
        corpus_kind="representative",
    )
    ambiguous_rows = _select_rows_for_corpus(
        accepted_rows,
        count=min(amb_target, len(accepted_rows)),
        corpus_kind="ambiguous",
    )
    representative_rows = _bootstrap_selected_rows(representative_rows)
    ambiguous_rows = _bootstrap_selected_rows(ambiguous_rows)
    return {
        "schema_version": "2.0.0",
        "created_at_utc": _utc_now_iso(),
        "seed": int(seed),
        "source_pool": pool_summary,
        "representative": _selected_corpus_summary(
            pool_summary=pool_summary,
            rows=representative_rows,
            corpus_kind="representative",
            selection_policy="bin_stratified_median_ambiguity_with_corridor_balance",
        ),
        "ambiguous": _selected_corpus_summary(
            pool_summary=pool_summary,
            rows=ambiguous_rows,
            corpus_kind="ambiguous",
            selection_policy="bin_stratified_high_ambiguity_descending",
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic UK OD corpus using route graph feasibility."
    )
    parser.add_argument("--seed", type=int, default=20260212)
    parser.add_argument("--pair-count", type=int, default=100)
    parser.add_argument("--ambiguous-pair-count", type=int, default=40)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--bbox", default=str(settings.terrain_uk_bbox))
    parser.add_argument("--output-dir", default=str(EVAL_DATA_DIR))
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--representative-output-csv", default=None)
    parser.add_argument("--representative-output-json", default=None)
    parser.add_argument("--representative-summary-json", default=None)
    parser.add_argument("--ambiguous-output-csv", default=None)
    parser.add_argument("--ambiguous-output-json", default=None)
    parser.add_argument("--ambiguous-summary-json", default=None)
    parser.add_argument(
        "--acceptance-mode",
        choices=("feasibility_only", "graph_candidates"),
        default="graph_candidates",
    )
    parser.add_argument("--probe-max-paths", type=int, default=6)
    parser.add_argument("--single-corpus", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def _default_output_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    corpus_dir = out_dir / "thesis"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now_compact()
    return (
        corpus_dir / f"od_corpus_uk_{stamp}.csv",
        corpus_dir / f"od_corpus_uk_{stamp}.json",
        corpus_dir / f"od_corpus_uk_{stamp}.summary.json",
    )


def _default_dual_output_paths(out_dir: Path) -> dict[str, tuple[Path, Path, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "representative": (
            out_dir / "uk_od_corpus_representative.csv",
            out_dir / "uk_od_corpus_representative.json",
            out_dir / "uk_od_corpus_representative.summary.json",
        ),
        "ambiguous": (
            out_dir / "uk_od_corpus_ambiguous.csv",
            out_dir / "uk_od_corpus_ambiguous.json",
            out_dir / "uk_od_corpus_ambiguous.summary.json",
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    bbox = _parse_bbox(args.bbox)
    out_dir = Path(args.output_dir).resolve()
    csv_default, json_default, summary_default = _default_output_paths(Path("out").resolve())
    csv_path = Path(args.output_csv).resolve() if args.output_csv else csv_default
    json_path = Path(args.output_json).resolve() if args.output_json else json_default
    summary_path = Path(args.summary_json).resolve() if args.summary_json else summary_default

    wants_single = bool(args.single_corpus or args.output_csv or args.output_json or args.summary_json)
    if not wants_single:
        dual_defaults = _default_dual_output_paths(out_dir)
        representative_csv = Path(args.representative_output_csv).resolve() if args.representative_output_csv else dual_defaults["representative"][0]
        representative_json = Path(args.representative_output_json).resolve() if args.representative_output_json else dual_defaults["representative"][1]
        representative_summary = Path(args.representative_summary_json).resolve() if args.representative_summary_json else dual_defaults["representative"][2]
        ambiguous_csv = Path(args.ambiguous_output_csv).resolve() if args.ambiguous_output_csv else dual_defaults["ambiguous"][0]
        ambiguous_json = Path(args.ambiguous_output_json).resolve() if args.ambiguous_output_json else dual_defaults["ambiguous"][1]
        ambiguous_summary = Path(args.ambiguous_summary_json).resolve() if args.ambiguous_summary_json else dual_defaults["ambiguous"][2]

        bundle = build_dual_od_corpora(
            seed=args.seed,
            representative_count=args.pair_count,
            ambiguous_count=max(1, int(args.ambiguous_pair_count)),
            bbox=bbox,
            max_attempts=args.max_attempts,
            acceptance_mode=args.acceptance_mode,
            probe_max_paths=args.probe_max_paths,
        )
        if not args.allow_partial:
            if len(bundle["representative"]["rows"]) < max(1, int(args.pair_count)):
                raise RuntimeError("Representative corpus builder did not reach the requested pair-count.")
            if len(bundle["ambiguous"]["rows"]) < max(1, int(args.ambiguous_pair_count)):
                raise RuntimeError("Ambiguous corpus builder did not reach the requested pair-count.")

        for path in (
            representative_csv,
            representative_json,
            representative_summary,
            ambiguous_csv,
            ambiguous_json,
            ambiguous_summary,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(representative_csv, bundle["representative"]["rows"])
        representative_json.write_text(json.dumps(bundle["representative"]["rows"], indent=2), encoding="utf-8")
        representative_summary.write_text(json.dumps(bundle["representative"], indent=2), encoding="utf-8")
        _write_csv(ambiguous_csv, bundle["ambiguous"]["rows"])
        ambiguous_json.write_text(json.dumps(bundle["ambiguous"]["rows"], indent=2), encoding="utf-8")
        ambiguous_summary.write_text(json.dumps(bundle["ambiguous"], indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "representative_csv": str(representative_csv),
                    "representative_summary_json": str(representative_summary),
                    "ambiguous_csv": str(ambiguous_csv),
                    "ambiguous_summary_json": str(ambiguous_summary),
                    "source_pool_hash": bundle["source_pool"]["corpus_hash"],
                },
                indent=2,
            )
        )
        return 0

    summary = build_od_corpus(
        seed=args.seed,
        pair_count=args.pair_count,
        bbox=bbox,
        max_attempts=args.max_attempts,
        acceptance_mode=args.acceptance_mode,
        probe_max_paths=args.probe_max_paths,
    )
    if not args.allow_partial and not summary["complete"]:
        raise RuntimeError(
            "Corpus builder did not reach the requested pair-count. "
            "Use --allow-partial if partial corpora are acceptable."
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, summary["rows"])
    json_path.write_text(json.dumps(summary["rows"], indent=2), encoding="utf-8")
    summary_payload = {key: value for key, value in summary.items() if key != "rows"}
    summary_payload["output_csv"] = str(csv_path)
    summary_payload["output_json"] = str(json_path)
    summary_payload["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
