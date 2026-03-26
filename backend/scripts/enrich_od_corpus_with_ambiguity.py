from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import settings
from app.routing_graph import route_graph_candidate_routes, route_graph_od_feasibility
from scripts.build_od_corpus_uk import (
    _ambiguity_derived_fields,
    _candidate_probe_payload,
    _cheap_prior_features,
    _corridor_bucket,
    _distance_bin,
    _haversine_km,
    _region_bucket,
)


PROBE_FIELDS = (
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
    "candidate_probe_top2_duration_gap",
    "candidate_probe_top2_cost_gap",
    "candidate_probe_top2_emissions_gap",
    "candidate_probe_toll_disagreement_rate",
    "candidate_probe_family_diversity",
    "candidate_probe_path_sufficiency",
    "candidate_probe_top2_gap_pressure",
    "candidate_probe_engine_disagreement_prior",
    "engine_probe_geometry_disagreement",
    "engine_probe_duration_delta",
    "engine_probe_distance_delta",
    "engine_probe_request_fingerprint",
    "hard_case_prior",
    "ambiguity_index",
    "od_ambiguity_index",
    "od_ambiguity_confidence",
    "od_ambiguity_source_count",
    "od_ambiguity_source_mix",
    "od_ambiguity_source_support",
    "od_ambiguity_source_support_strength",
    "od_ambiguity_support_ratio",
    "od_ambiguity_source_mix_count",
    "od_ambiguity_source_entropy",
    "od_ambiguity_prior_strength",
    "od_ambiguity_family_density",
    "od_ambiguity_margin_pressure",
    "od_ambiguity_spread_pressure",
    "od_ambiguity_toll_instability",
    "ambiguity_prior_source",
    "ambiguity_prior_sample_count",
    "ambiguity_prior_support_count",
    "ambiguity_prior_nonzero",
    "ambiguity_budget_prior",
    "ambiguity_budget_prior_gap",
    "budget_prior_exceeds_raw",
    "route_graph_reason_code",
    "route_graph_message",
    "origin_node_id",
    "destination_node_id",
    "origin_nearest_distance_m",
    "destination_nearest_distance_m",
    "origin_region_bucket",
    "destination_region_bucket",
    "corridor_bucket",
    "accepted",
)

PRIOR_SOURCE_ORDER = (
    "routing_graph_probe",
    "engine_augmented_probe",
    "repo_local_geometry_backfill",
    "historical_results_bootstrap",
    "existing_corpus",
)

PRIOR_NUMERIC_FIELDS = (
    "candidate_probe_path_count",
    "candidate_probe_corridor_family_count",
    "candidate_probe_objective_spread",
    "candidate_probe_nominal_margin",
    "candidate_probe_toll_disagreement_rate",
    "candidate_probe_engine_disagreement_prior",
    "hard_case_prior",
    "ambiguity_index",
    "od_ambiguity_index",
    "od_ambiguity_prior_strength",
    "od_ambiguity_confidence",
    "od_ambiguity_source_support_strength",
)

SOURCE_CONFIDENCE_WEIGHTS = {
    "routing_graph_probe": 0.92,
    "engine_augmented_probe": 0.78,
    "repo_local_geometry_backfill": 0.56,
    "historical_results_bootstrap": 0.67,
    "existing_corpus": 0.48,
}

RAW_AMBIGUITY_PRIOR_KEYS = (
    "od_ambiguity_index",
    "ambiguity_index",
    "candidate_probe_ambiguity_index",
)

RAW_AMBIGUITY_PRIOR_FALLBACK_KEYS = (
    "candidate_probe_engine_disagreement_prior",
    "od_engine_disagreement_prior",
    "hard_case_prior",
    "od_hard_case_prior",
)

_ENGINE_PROBE_TIMEOUT_S = 8.0


def _ors_profile_for_row(row: Mapping[str, Any]) -> tuple[str, str | None]:
    vehicle_type = str(row.get("vehicle_type") or row.get("profile_id") or "").strip().lower()
    if any(token in vehicle_type for token in ("hgv", "truck", "artic", "rigid")):
        return str(getattr(settings, "ors_directions_profile_hgv", "driving-hgv") or "driving-hgv"), "hgv"
    return str(getattr(settings, "ors_directions_profile_default", "driving-car") or "driving-car"), None


def _sample_polyline(coords: list[tuple[float, float]], *, max_points: int = 24) -> list[tuple[float, float]]:
    if len(coords) <= max_points:
        return list(coords)
    if max_points <= 2:
        return [coords[0], coords[-1]]
    step = max(1, math.ceil((len(coords) - 1) / float(max_points - 1)))
    sampled = coords[::step]
    if sampled[-1] != coords[-1]:
        sampled.append(coords[-1])
    return sampled[: max_points - 1] + [coords[-1]]


def _sampled_geometry_disagreement(
    osrm_coords: list[tuple[float, float]],
    ors_coords: list[tuple[float, float]],
) -> float:
    if len(osrm_coords) < 2 or len(ors_coords) < 2:
        return 0.0
    osrm_sample = _sample_polyline(osrm_coords)
    ors_sample = _sample_polyline(ors_coords)
    horizon_km = max(
        _haversine_km(lat1, lon1, lat2, lon2)
        for (lon1, lat1), (lon2, lat2) in zip(osrm_sample[:-1], osrm_sample[1:], strict=False)
    ) if len(osrm_sample) >= 2 else 0.0
    horizon_km = max(horizon_km, 1.0)
    paired = min(len(osrm_sample), len(ors_sample))
    if paired <= 0:
        return 0.0
    distances = [
        _haversine_km(osrm_sample[idx][1], osrm_sample[idx][0], ors_sample[idx][1], ors_sample[idx][0])
        for idx in range(paired)
    ]
    mean_gap_km = sum(distances) / float(len(distances) or 1)
    return round(min(1.0, max(0.0, mean_gap_km / horizon_km)), 6)


def _support_gated_budget_prior(
    *,
    ambiguity_value: float,
    support_ratio: float,
    source_entropy: float,
    source_count: int,
    source_mix_count: int,
    prior_strength: float,
    family_density: float,
    margin_pressure: float,
    spread_pressure: float,
    toll_instability: float,
    engine_prior: float,
    hard_case_prior: float,
) -> float:
    raw_value = max(0.0, min(1.0, ambiguity_value))
    source_count_strength = max(0.0, min(1.0, source_count / 4.0))
    source_mix_strength = max(0.0, min(1.0, source_mix_count / 4.0))
    support_ratio = max(0.0, min(1.0, support_ratio))
    source_entropy = max(0.0, min(1.0, source_entropy))
    prior_strength = max(0.0, min(1.0, prior_strength))
    family_density = max(0.0, min(1.0, family_density))
    margin_pressure = max(0.0, min(1.0, margin_pressure))
    spread_pressure = max(0.0, min(1.0, spread_pressure))
    toll_instability = max(0.0, min(1.0, toll_instability))
    engine_prior = max(0.0, min(1.0, engine_prior))
    hard_case_prior = max(0.0, min(1.0, hard_case_prior))

    source_support = max(
        0.0,
        min(
            1.0,
            (0.42 * support_ratio)
            + (0.28 * source_entropy)
            + (0.16 * source_count_strength)
            + (0.14 * source_mix_strength),
        ),
    )
    support_gate = max(
        0.0,
        min(
            1.0,
            (0.34 * support_ratio)
            + (0.24 * source_entropy)
            + (0.12 * source_count_strength)
            + (0.10 * source_mix_strength)
            + (0.08 * prior_strength)
            + (0.06 * family_density)
            + (0.04 * margin_pressure)
            + (0.02 * spread_pressure),
        ),
    )
    source_cohesion = max(
        0.0,
        min(
            1.0,
            (0.55 * min(source_count_strength, source_mix_strength))
            + (0.45 * max(0.0, min(1.0, source_entropy))),
        ),
    )
    pressure_gate = max(
        0.0,
        min(1.0, 0.52 * margin_pressure + 0.30 * spread_pressure + 0.18 * toll_instability),
    )
    support_alignment = max(
        0.0,
        min(
            1.0,
            (0.40 * support_ratio)
            + (0.28 * source_entropy)
            + (0.18 * source_cohesion)
            + (0.14 * family_density),
        ),
    )
    support_votes = [
        source_support >= 0.58 and support_ratio >= 0.60 and source_entropy >= 0.52 and source_count >= 2 and source_mix_count >= 2,
        engine_prior >= max(raw_value + 0.06, 0.40) and support_ratio >= 0.54 and source_entropy >= 0.46 and family_density >= 0.24,
        hard_case_prior >= max(raw_value + 0.06, 0.40) and support_ratio >= 0.54 and source_entropy >= 0.46 and family_density >= 0.24,
        pressure_gate >= 0.50 and margin_pressure >= 0.45 and (spread_pressure >= 0.32 or toll_instability >= 0.30),
        family_density >= 0.52 and source_mix_strength >= 0.56 and prior_strength >= 0.46 and support_ratio >= 0.50,
    ]
    if sum(1 for vote in support_votes if bool(vote)) < 2:
        return round(raw_value, 6)
    consensus_score = max(
        0.0,
        min(
            1.0,
            (0.40 * source_support)
            + (0.24 * support_alignment)
            + (0.16 * pressure_gate)
            + (0.10 * max(engine_prior, hard_case_prior))
            + (0.10 * prior_strength),
        ),
    )
    if consensus_score <= raw_value + 0.03:
        return round(raw_value, 6)
    uplift_scale = min(
        0.18,
        max(
            0.03,
            0.04
            + (0.08 * source_support)
            + (0.05 * support_alignment)
            + (0.03 * pressure_gate)
            + (0.03 * source_cohesion),
        ),
    )
    uplift_cap = min(
        0.12,
        0.01
        + (0.04 * source_support)
        + (0.03 * support_alignment)
        + (0.02 * pressure_gate)
        + (0.02 * source_cohesion),
    )
    budget_prior = min(1.0, raw_value + min(uplift_cap, max(0.0, consensus_score - raw_value) * uplift_scale))
    return round(budget_prior, 6)


def _engine_augmented_prior_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    origin_lat = _as_float(row.get("origin_lat"))
    origin_lon = _as_float(row.get("origin_lon"))
    destination_lat = _as_float(row.get("destination_lat"))
    destination_lon = _as_float(row.get("destination_lon"))
    if not all(math.isfinite(value) for value in (origin_lat, origin_lon, destination_lat, destination_lon)):
        return None

    osrm_url = str(getattr(settings, "osrm_base_url", "") or "").rstrip("/")
    ors_url = str(getattr(settings, "ors_base_url", "") or "").rstrip("/")
    if not osrm_url or not ors_url:
        return None

    ors_profile, ors_vehicle_type = _ors_profile_for_row(row)
    osrm_request = (
        f"{osrm_url}/route/v1/driving/{origin_lon:.6f},{origin_lat:.6f};{destination_lon:.6f},{destination_lat:.6f}"
        "?alternatives=false&overview=full&geometries=geojson"
    )
    ors_request = f"{ors_url}/v2/directions/{ors_profile}/geojson"
    ors_body: dict[str, Any] = {
        "coordinates": [[origin_lon, origin_lat], [destination_lon, destination_lat]],
        "instructions": False,
        "elevation": False,
    }
    if ors_vehicle_type:
        ors_body["options"] = {"vehicle_type": ors_vehicle_type}

    try:
        with httpx.Client(timeout=_ENGINE_PROBE_TIMEOUT_S, trust_env=False) as client:
            osrm_resp = client.get(osrm_request)
            osrm_resp.raise_for_status()
            ors_resp = client.post(ors_request, json=ors_body)
            ors_resp.raise_for_status()
    except Exception:
        return None

    try:
        osrm_payload = osrm_resp.json()
        ors_payload = ors_resp.json()
        osrm_route = ((osrm_payload.get("routes") or [])[0]) if isinstance(osrm_payload, dict) else {}
        ors_feature = ((ors_payload.get("features") or [])[0]) if isinstance(ors_payload, dict) else {}
        osrm_coords_raw = ((osrm_route.get("geometry") or {}).get("coordinates") or [])
        ors_coords_raw = (((ors_feature.get("geometry") or {}).get("coordinates")) or [])
        osrm_coords = [
            (float(item[0]), float(item[1]))
            for item in osrm_coords_raw
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
        ors_coords = [
            (float(item[0]), float(item[1]))
            for item in ors_coords_raw
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
        osrm_distance_km = float(osrm_route.get("distance") or 0.0) / 1000.0
        osrm_duration_s = float(osrm_route.get("duration") or 0.0)
        ors_summary = ((ors_feature.get("properties") or {}).get("summary") or {})
        ors_distance_km = float(ors_summary.get("distance") or 0.0) / 1000.0
        ors_duration_s = float(ors_summary.get("duration") or 0.0)
    except Exception:
        return None

    if min(osrm_distance_km, osrm_duration_s, ors_distance_km, ors_duration_s) <= 0.0:
        return None

    duration_delta = abs(osrm_duration_s - ors_duration_s) / max(osrm_duration_s, ors_duration_s, 1.0)
    distance_delta = abs(osrm_distance_km - ors_distance_km) / max(osrm_distance_km, ors_distance_km, 1.0)
    geometry_disagreement = _sampled_geometry_disagreement(osrm_coords, ors_coords)
    graph_path_count = int(_as_float(row.get("candidate_probe_path_count"), 0.0))
    graph_family_count = int(_as_float(row.get("candidate_probe_corridor_family_count"), 0.0))
    graph_objective_spread = _as_float(row.get("candidate_probe_objective_spread"), 0.0)
    graph_nominal_margin = _as_float(row.get("candidate_probe_nominal_margin"), 0.0)
    blended_priors = _cheap_prior_features(
        path_count=max(2, graph_path_count),
        family_count=max(2, graph_family_count),
        objective_spread=max(graph_objective_spread, geometry_disagreement, distance_delta),
        nominal_margin=graph_nominal_margin,
        toll_disagreement=_as_float(row.get("candidate_probe_toll_disagreement_rate"), 0.0),
        top2_duration_gap=duration_delta,
        top2_cost_gap=distance_delta,
        top2_emissions_gap=geometry_disagreement,
    )
    disagreement_prior = round(
        min(
            1.0,
            max(
                0.0,
                0.34 * min(1.0, duration_delta)
                + 0.20 * min(1.0, distance_delta)
                + 0.22 * geometry_disagreement
                + 0.24 * _as_float(blended_priors.get("candidate_probe_engine_disagreement_prior"), 0.0),
            ),
        ),
        6,
    )
    hard_case_prior = round(
        min(
            1.0,
            max(
                _as_float(row.get("hard_case_prior"), 0.0),
                0.54 * disagreement_prior
                + 0.20 * min(1.0, geometry_disagreement + duration_delta)
                + 0.26 * _as_float(blended_priors.get("hard_case_prior"), 0.0),
            ),
        ),
        6,
    )
    ambiguity_value = round(
        max(
            _as_float(row.get("od_ambiguity_index"), 0.0),
            _as_float(row.get("ambiguity_index"), 0.0),
            disagreement_prior,
            hard_case_prior,
        ),
        6,
    )
    support_ratio = _as_float(row.get("od_ambiguity_support_ratio"), 0.0)
    source_entropy = _as_float(row.get("od_ambiguity_source_entropy"), 0.0)
    source_count = int(_as_float(row.get("od_ambiguity_source_count"), 0.0))
    source_mix_count = int(_as_float(row.get("od_ambiguity_source_mix_count"), 0.0))
    prior_strength = _as_float(row.get("od_ambiguity_prior_strength"), ambiguity_value)
    family_density = _as_float(row.get("od_ambiguity_family_density"), 0.0)
    margin_pressure = _as_float(row.get("od_ambiguity_margin_pressure"), 0.0)
    spread_pressure = _as_float(row.get("od_ambiguity_spread_pressure"), 0.0)
    toll_instability = _as_float(row.get("od_ambiguity_toll_instability"), 0.0)
    budget_prior = _support_gated_budget_prior(
        ambiguity_value=ambiguity_value,
        support_ratio=support_ratio,
        source_entropy=source_entropy,
        source_count=source_count,
        source_mix_count=source_mix_count,
        prior_strength=prior_strength,
        family_density=family_density,
        margin_pressure=margin_pressure,
        spread_pressure=spread_pressure,
        toll_instability=toll_instability,
        engine_prior=disagreement_prior,
        hard_case_prior=hard_case_prior,
    )
    support_count = sum(
        1
        for value in (duration_delta, distance_delta, geometry_disagreement, disagreement_prior)
        if _is_nonzero_numeric(value)
    )
    request_fingerprint = hashlib.sha1(
        json.dumps(
            {
                "o": [origin_lon, origin_lat],
                "d": [destination_lon, destination_lat],
                "profile": ors_profile,
                "vehicle_type": ors_vehicle_type,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "candidate_probe_engine_disagreement_prior": disagreement_prior,
        "hard_case_prior": hard_case_prior,
        "ambiguity_index": ambiguity_value,
        "od_ambiguity_index": ambiguity_value,
        "ambiguity_budget_prior": budget_prior,
        "ambiguity_budget_prior_gap": round(max(0.0, budget_prior - ambiguity_value), 6),
        "budget_prior_exceeds_raw": bool(budget_prior > ambiguity_value + 1e-9),
        "candidate_probe_family_diversity": max(
            _as_float(row.get("candidate_probe_family_diversity"), 0.0),
            geometry_disagreement,
        ),
        "candidate_probe_path_sufficiency": max(
            _as_float(row.get("candidate_probe_path_sufficiency"), 0.0),
            _as_float(blended_priors.get("candidate_probe_path_sufficiency"), 0.0),
        ),
        "candidate_probe_top2_gap_pressure": max(
            _as_float(row.get("candidate_probe_top2_gap_pressure"), 0.0),
            _as_float(blended_priors.get("candidate_probe_top2_gap_pressure"), 0.0),
        ),
        "ambiguity_prior_sample_count": max(3, int(_as_float(row.get("ambiguity_prior_sample_count"), 0.0))),
        "ambiguity_prior_support_count": max(2, support_count),
        "engine_probe_geometry_disagreement": geometry_disagreement,
        "engine_probe_duration_delta": round(duration_delta, 6),
        "engine_probe_distance_delta": round(distance_delta, 6),
        "engine_probe_request_fingerprint": request_fingerprint,
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return float(parsed)


def _identity_value(value: Any) -> str:
    return str(value or "").strip()


def _is_nonzero_numeric(value: Any) -> bool:
    parsed = _as_float(value, float("nan"))
    return math.isfinite(parsed) and abs(parsed) > 1e-12


def _is_missing_or_zero(value: Any) -> bool:
    return value in (None, "") or not _is_nonzero_numeric(value)


def _raw_ambiguity_prior_value(row: Mapping[str, Any]) -> float | None:
    raw_candidates: list[float] = []
    for key in RAW_AMBIGUITY_PRIOR_KEYS:
        value = row.get(key)
        if value in (None, ""):
            continue
        parsed = _as_float(value, float("nan"))
        if math.isfinite(parsed):
            raw_candidates.append(min(1.0, max(0.0, parsed)))
    if raw_candidates:
        return max(raw_candidates)

    fallback_candidates: list[float] = []
    for key in RAW_AMBIGUITY_PRIOR_FALLBACK_KEYS:
        value = row.get(key)
        if value in (None, ""):
            continue
        parsed = _as_float(value, float("nan"))
        if math.isfinite(parsed):
            fallback_candidates.append(min(1.0, max(0.0, parsed)))
    if not fallback_candidates:
        return None
    return max(fallback_candidates)


def _has_existing_graph_probe_signal(row: Mapping[str, Any]) -> bool:
    numeric_probe_fields = (
        "candidate_probe_path_count",
        "candidate_probe_corridor_family_count",
        "candidate_probe_explored_states",
        "candidate_probe_generated_paths",
        "candidate_probe_emitted_paths",
    )
    if any(_as_float(row.get(field), 0.0) > 0.0 for field in numeric_probe_fields):
        return True
    reason_code = str(row.get("route_graph_reason_code") or "").strip().lower()
    return bool(reason_code) and reason_code not in {"", "routing_graph_no_path"}


def _merge_prior_sources(*sources: Any) -> str | None:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in sources:
        if raw in (None, ""):
            continue
        parts = _split_prior_sources(raw)
        for part in parts:
            if not part or part in seen:
                continue
            seen.add(part)
            tokens.append(part)
    if not tokens:
        return None
    order = {label: idx for idx, label in enumerate(PRIOR_SOURCE_ORDER)}
    tokens.sort(key=lambda token: (order.get(token, len(order)), token))
    return ",".join(tokens)


def _split_prior_sources(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, Mapping):
        return [str(key).strip() for key in sorted(value) if str(key).strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seen_sequence: set[str] = set()
        sequence_tokens: list[str] = []
        for item in value:
            token = str(item).strip()
            if not token or token in seen_sequence:
                continue
            seen_sequence.add(token)
            sequence_tokens.append(token)
        return sequence_tokens
    text = str(value).strip()
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
    tokens: list[str] = []
    seen: set[str] = set()
    for part in text.replace("+", ",").split(","):
        token = part.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _parse_source_mix_counts(value: Any) -> dict[str, int]:
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        out: dict[str, int] = {}
        for key, raw in value.items():
            token = str(key).strip()
            if not token:
                continue
            count = int(max(0.0, _as_float(raw, 0.0)))
            if count > 0:
                out[token] = count
        return out
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {token: 1 for token in _split_prior_sources(text)}
    if isinstance(parsed, Mapping):
        return _parse_source_mix_counts(parsed)
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        return {token: 1 for token in _split_prior_sources(parsed)}
    return {}


def _apply_ambiguity_derived_fields(row: dict[str, Any]) -> None:
    ambiguity_value = _as_float(
        row.get(
            "od_ambiguity_index",
            row.get("ambiguity_index", row.get("candidate_probe_ambiguity_index")),
        ),
        0.0,
    )
    confidence = _as_float(row.get("od_ambiguity_confidence"), 0.0)
    source_count = int(_as_float(row.get("od_ambiguity_source_count"), 0.0))
    sample_count = int(_as_float(row.get("ambiguity_prior_sample_count"), 0.0))
    support_count = int(_as_float(row.get("ambiguity_prior_support_count"), 0.0))
    path_count = int(_as_float(row.get("candidate_probe_path_count"), 0.0))
    family_count = int(_as_float(row.get("candidate_probe_corridor_family_count"), 0.0))
    objective_spread = _as_float(row.get("candidate_probe_objective_spread"), 0.0)
    nominal_margin = _as_float(row.get("candidate_probe_nominal_margin"), 0.0)
    toll_disagreement = _as_float(row.get("candidate_probe_toll_disagreement_rate"), 0.0)
    explicit_prior_strength = None
    if row.get("od_ambiguity_prior_strength") not in (None, ""):
        candidate_strength = _as_float(row.get("od_ambiguity_prior_strength"), float("nan"))
        if math.isfinite(candidate_strength) and abs(candidate_strength) > 1e-12:
            explicit_prior_strength = candidate_strength
    row.update(
        _ambiguity_derived_fields(
            ambiguity_index=ambiguity_value,
            ambiguity_confidence=confidence,
            source_count=source_count,
            source_mix=row.get("od_ambiguity_source_mix"),
            sample_count=sample_count,
            support_count=support_count,
            path_count=path_count,
            family_count=family_count,
            objective_spread=objective_spread,
            nominal_margin=nominal_margin,
            toll_disagreement=toll_disagreement,
            explicit_prior_strength=explicit_prior_strength,
        )
    )
    engine_prior = _as_float(row.get("candidate_probe_engine_disagreement_prior"), 0.0)
    hard_case_prior = _as_float(row.get("hard_case_prior"), 0.0)
    support_ratio = _as_float(row.get("od_ambiguity_support_ratio"), 0.0)
    source_entropy = _as_float(row.get("od_ambiguity_source_entropy"), 0.0)
    source_count = int(_as_float(row.get("od_ambiguity_source_count"), 0.0))
    source_mix_count = int(_as_float(row.get("od_ambiguity_source_mix_count"), 0.0))
    prior_strength = _as_float(row.get("od_ambiguity_prior_strength"), ambiguity_value)
    family_density = _as_float(row.get("od_ambiguity_family_density"), 0.0)
    margin_pressure = _as_float(row.get("od_ambiguity_margin_pressure"), 0.0)
    spread_pressure = _as_float(row.get("od_ambiguity_spread_pressure"), 0.0)
    toll_instability = _as_float(row.get("od_ambiguity_toll_instability"), 0.0)
    budget_prior = _support_gated_budget_prior(
        ambiguity_value=ambiguity_value,
        support_ratio=support_ratio,
        source_entropy=source_entropy,
        source_count=source_count,
        source_mix_count=source_mix_count,
        prior_strength=prior_strength,
        family_density=family_density,
        margin_pressure=margin_pressure,
        spread_pressure=spread_pressure,
        toll_instability=toll_instability,
        engine_prior=engine_prior,
        hard_case_prior=hard_case_prior,
    )
    row["ambiguity_budget_prior"] = round(budget_prior, 6)
    row["ambiguity_budget_prior_gap"] = round(max(0.0, budget_prior - ambiguity_value), 6)
    row["budget_prior_exceeds_raw"] = bool(budget_prior > ambiguity_value + 1e-9)


def _source_mix_string(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return json.dumps({token: 1 for token in tokens}, sort_keys=True, separators=(",", ":"))


def _source_mix_counts_for_row(row: dict[str, Any], source_tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in source_tokens:
        if token == "routing_graph_probe":
            counts[token] = max(
                1,
                int(_as_float(row.get("candidate_probe_emitted_paths"), 0.0)),
                int(_as_float(row.get("candidate_probe_corridor_family_count"), 0.0)),
            )
        elif token == "engine_augmented_probe":
            counts[token] = max(
                1,
                sum(
                    1
                    for key in (
                        "engine_probe_geometry_disagreement",
                        "engine_probe_duration_delta",
                        "engine_probe_distance_delta",
                        "candidate_probe_engine_disagreement_prior",
                    )
                    if _is_nonzero_numeric(row.get(key))
                ),
            )
        elif token == "historical_results_bootstrap":
            counts[token] = max(1, int(_as_float(row.get("ambiguity_prior_sample_count"), 0.0)))
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
                    )
                    if _is_nonzero_numeric(row.get(key))
                ),
            )
        else:
            counts[token] = 1
    return counts


def _source_support_map_for_row(row: Mapping[str, Any], source_tokens: Sequence[str]) -> dict[str, float]:
    support: dict[str, float] = {}
    path_count = max(0, int(_as_float(row.get("candidate_probe_path_count"), 0.0)))
    family_count = max(0, int(_as_float(row.get("candidate_probe_corridor_family_count"), 0.0)))
    path_density = max(0.0, min(1.0, max(0, path_count - 1) / 7.0))
    family_balance = 0.0 if path_count <= 1 else round((0.55 * min(1.0, family_count / float(max(1, path_count)))) + (0.45 * min(1.0, max(0, family_count - 1) / 4.0)), 6)
    objective_spread = max(0.0, min(1.0, _as_float(row.get("candidate_probe_objective_spread"), 0.0)))
    nominal_margin = max(0.0, min(1.0, _as_float(row.get("candidate_probe_nominal_margin"), 0.0)))
    engine_prior = max(0.0, min(1.0, _as_float(row.get("candidate_probe_engine_disagreement_prior"), 0.0)))
    hard_case_prior = max(0.0, min(1.0, _as_float(row.get("hard_case_prior"), 0.0)))
    sample_signal = min(1.0, math.log1p(max(int(_as_float(row.get("ambiguity_prior_sample_count"), 0.0)), 0)) / math.log(9.0))
    support_signal = min(1.0, max(int(_as_float(row.get("ambiguity_prior_support_count"), 0.0)), 0) / 4.0)
    ambiguity_support = max(0.0, min(1.0, _as_float(row.get("od_ambiguity_support_ratio"), 0.0)))
    for token in source_tokens:
        if token == "routing_graph_probe":
            support[token] = round(
                min(1.0, 0.42 * path_density + 0.34 * family_balance + 0.14 * objective_spread + 0.10 * (1.0 - nominal_margin)),
                6,
            )
        elif token == "engine_augmented_probe":
            support[token] = round(
                min(1.0, 0.48 * engine_prior + 0.32 * hard_case_prior + 0.20 * max(objective_spread, ambiguity_support)),
                6,
            )
        elif token == "historical_results_bootstrap":
            support[token] = round(min(1.0, 0.55 * sample_signal + 0.45 * support_signal), 6)
        elif token == "repo_local_geometry_backfill":
            support[token] = round(
                min(1.0, 0.40 * path_density + 0.26 * family_balance + 0.22 * objective_spread + 0.12 * (1.0 - nominal_margin)),
                6,
            )
        elif token == "existing_corpus":
            support[token] = round(
                min(1.0, 0.44 * ambiguity_support + 0.22 * engine_prior + 0.18 * hard_case_prior + 0.16 * sample_signal),
                6,
            )
        else:
            support[token] = 0.5
    return support


def _source_support_strength(source_support: Mapping[str, Any] | Sequence[Any]) -> float:
    if isinstance(source_support, Mapping):
        values = [max(0.0, min(1.0, _as_float(value, 0.0))) for value in source_support.values()]
    else:
        values = [max(0.0, min(1.0, _as_float(value, 0.0))) for value in source_support]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _ambiguity_confidence(
    *,
    source_tokens: list[str],
    sample_count: int,
    support_count: int,
    ambiguity_nonzero: bool,
) -> float:
    if not source_tokens:
        return 0.0
    base_source = max(SOURCE_CONFIDENCE_WEIGHTS.get(token, 0.4) for token in source_tokens)
    source_bonus = min(1.0, len(source_tokens) / 3.0)
    sample_signal = min(1.0, math.log1p(max(sample_count, 0)) / math.log(9.0))
    support_signal = min(1.0, max(support_count, 0) / 4.0)
    confidence = (
        0.45 * base_source
        + 0.15 * source_bonus
        + 0.20 * sample_signal
        + 0.20 * support_signal
    )
    if not ambiguity_nonzero:
        confidence *= 0.85
    return round(min(1.0, max(0.0, confidence)), 6)


def _update_prior_metadata(
    row: dict[str, Any],
    *,
    source: str | None = None,
    sample_count: int | None = None,
    support_count: int | None = None,
) -> None:
    merged_source = _merge_prior_sources(row.get("ambiguity_prior_source"), source)
    if merged_source:
        row["ambiguity_prior_source"] = merged_source
    if sample_count is not None and int(sample_count) > 0:
        current = int(_as_float(row.get("ambiguity_prior_sample_count"), 0.0))
        row["ambiguity_prior_sample_count"] = max(current, int(sample_count))
    if support_count is not None and int(support_count) > 0:
        current_support = int(_as_float(row.get("ambiguity_prior_support_count"), 0.0))
        row["ambiguity_prior_support_count"] = max(current_support, int(support_count))
    ambiguity_value = row.get("od_ambiguity_index")
    if ambiguity_value in (None, ""):
        ambiguity_value = row.get("ambiguity_index")
    ambiguity_nonzero = _is_nonzero_numeric(ambiguity_value)
    row["ambiguity_prior_nonzero"] = ambiguity_nonzero
    source_tokens = _split_prior_sources(row.get("ambiguity_prior_source"))
    row["od_ambiguity_source_count"] = len(source_tokens)
    source_mix_counts = _source_mix_counts_for_row(row, source_tokens)
    row["od_ambiguity_source_mix"] = json.dumps(source_mix_counts, sort_keys=True, separators=(",", ":")) if source_mix_counts else ""
    source_support = _source_support_map_for_row(row, source_tokens)
    row["od_ambiguity_source_support"] = json.dumps(source_support, sort_keys=True, separators=(",", ":")) if source_support else ""
    row["od_ambiguity_source_support_strength"] = _source_support_strength(source_support)
    row["od_ambiguity_confidence"] = _ambiguity_confidence(
        source_tokens=source_tokens,
        sample_count=int(_as_float(row.get("ambiguity_prior_sample_count"), 0.0)),
        support_count=int(_as_float(row.get("ambiguity_prior_support_count"), 0.0)),
        ambiguity_nonzero=ambiguity_nonzero,
    )
    _apply_ambiguity_derived_fields(row)


def _row_identity_key(row: dict[str, Any]) -> str:
    payload = {
        "od_id": _identity_value(row.get("od_id")),
        "profile_id": _identity_value(row.get("profile_id")),
        "corpus_group": _identity_value(row.get("corpus_group") or row.get("corpus_kind")),
        "origin_lat": _identity_value(row.get("origin_lat")),
        "origin_lon": _identity_value(row.get("origin_lon")),
        "destination_lat": _identity_value(row.get("destination_lat")),
        "destination_lon": _identity_value(row.get("destination_lon")),
        "departure_time_utc": _identity_value(row.get("departure_time_utc")),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _bootstrap_observed_ambiguity(row: dict[str, Any]) -> float | None:
    explicit = _as_float(row.get("observed_ambiguity_index"), float("nan"))
    if math.isfinite(explicit):
        return round(explicit, 6)
    signals: list[float] = []
    frontier_count = _as_float(row.get("frontier_count"), float("nan"))
    if math.isfinite(frontier_count):
        signals.append(min(1.0, max(0.0, (frontier_count - 1.0) / 5.0)))
    near_tie_mass = _as_float(row.get("near_tie_mass"), float("nan"))
    if math.isfinite(near_tie_mass):
        signals.append(min(1.0, max(0.0, near_tie_mass)))
    nominal_margin = _as_float(row.get("nominal_winner_margin"), float("nan"))
    if math.isfinite(nominal_margin):
        signals.append(min(1.0, max(0.0, 1.0 - nominal_margin)))
    if str(row.get("selector_certificate_disagreement") or "").strip():
        lowered = str(row.get("selector_certificate_disagreement") or "").strip().lower()
        signals.append(1.0 if lowered in {"true", "1", "yes"} else 0.0)
    action_count = _as_float(row.get("voi_action_count"), float("nan"))
    if math.isfinite(action_count):
        signals.append(min(1.0, max(0.0, action_count) / 4.0))
    if not signals:
        return None
    return round(sum(signals) / len(signals), 6)


def _summarize_bootstrap_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ambiguity_values = [
        value
        for value in (_bootstrap_observed_ambiguity(row) for row in rows)
        if value is not None and math.isfinite(value)
    ]
    path_counts = [
        int(_as_float(row.get("candidate_count_raw"), 0.0))
        for row in rows
        if int(_as_float(row.get("candidate_count_raw"), 0.0)) > 0
    ]
    corridor_counts = [
        int(_as_float(row.get("frontier_count"), 0.0))
        for row in rows
        if int(_as_float(row.get("frontier_count"), 0.0)) > 0
    ]
    nominal_margins = [
        _as_float(row.get("nominal_winner_margin"), float("nan"))
        for row in rows
    ]
    nominal_margins = [value for value in nominal_margins if math.isfinite(value)]
    near_ties = [
        _as_float(row.get("near_tie_mass"), float("nan"))
        for row in rows
    ]
    near_ties = [value for value in near_ties if math.isfinite(value)]
    mean_nominal_margin = round(sum(nominal_margins) / len(nominal_margins), 6) if nominal_margins else None
    mean_objective_spread = round(sum(near_ties) / len(near_ties), 6) if near_ties else None
    cheap_priors = _cheap_prior_features(
        path_count=max(path_counts, default=0),
        family_count=max(corridor_counts, default=0),
        objective_spread=mean_objective_spread or 0.0,
        nominal_margin=mean_nominal_margin or 0.0,
        toll_disagreement=0.0,
    )
    bootstrap_ambiguity = round(sum(ambiguity_values) / len(ambiguity_values), 6) if ambiguity_values else None
    return {
        "ambiguity_index": bootstrap_ambiguity,
        "od_ambiguity_index": bootstrap_ambiguity,
        "od_ambiguity_prior_strength": bootstrap_ambiguity,
        "candidate_probe_path_count": max(path_counts, default=0),
        "candidate_probe_corridor_family_count": max(corridor_counts, default=0),
        "candidate_probe_nominal_margin": mean_nominal_margin,
        "candidate_probe_objective_spread": mean_objective_spread,
        "candidate_probe_engine_disagreement_prior": cheap_priors["candidate_probe_engine_disagreement_prior"],
        "hard_case_prior": cheap_priors["hard_case_prior"],
        "ambiguity_prior_source": "historical_results_bootstrap",
        "ambiguity_prior_sample_count": len(rows),
        "ambiguity_prior_support_count": sum(
            1
            for value in (
                max(path_counts, default=0),
                max(corridor_counts, default=0),
                mean_nominal_margin,
                mean_objective_spread,
            )
            if _is_nonzero_numeric(value)
        ),
    }


def _historical_bootstrap_index(paths: list[Path]) -> dict[str, Any]:
    grouped_by_identity: dict[str, list[dict[str, Any]]] = {}
    grouped_by_od: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in paths:
        if not path.exists():
            continue
        rows = _read_rows(path)
        for row in rows:
            od_id = str(row.get("od_id") or "").strip()
            if not od_id:
                continue
            identity = _row_identity_key(dict(row))
            grouped_by_identity.setdefault(identity, []).append(dict(row))
            grouped_by_od.setdefault(od_id, {}).setdefault(identity, []).append(dict(row))
    by_identity = {
        identity: _summarize_bootstrap_payload(rows)
        for identity, rows in grouped_by_identity.items()
    }
    by_od_id_unique: dict[str, dict[str, Any]] = {}
    for od_id, identity_groups in grouped_by_od.items():
        if len(identity_groups) == 1:
            rows = next(iter(identity_groups.values()))
            by_od_id_unique[od_id] = _summarize_bootstrap_payload(rows)
    return {
        "by_identity": by_identity,
        "by_od_id_unique": by_od_id_unique,
    }


def _apply_bootstrap_prior(
    row: dict[str, Any],
    *,
    bootstrap_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not bootstrap_index:
        return dict(row)
    payload: dict[str, Any] | None = None
    if "by_identity" in bootstrap_index or "by_od_id_unique" in bootstrap_index:
        identity = _row_identity_key(row)
        by_identity = bootstrap_index.get("by_identity", {})
        if isinstance(by_identity, dict):
            maybe_payload = by_identity.get(identity)
            if isinstance(maybe_payload, dict):
                payload = maybe_payload
        if payload is None:
            od_id = str(row.get("od_id") or "").strip()
            by_od_id_unique = bootstrap_index.get("by_od_id_unique", {})
            if isinstance(by_od_id_unique, dict):
                maybe_payload = by_od_id_unique.get(od_id)
                if isinstance(maybe_payload, dict):
                    payload = maybe_payload
    else:
        od_id = str(row.get("od_id") or "").strip()
        maybe_payload = bootstrap_index.get(od_id)
        if isinstance(maybe_payload, dict):
            payload = maybe_payload
    if not payload:
        return dict(row)
    enriched = dict(row)
    for source_key, target_key in (
        ("ambiguity_index", "ambiguity_index"),
        ("od_ambiguity_index", "od_ambiguity_index"),
        ("od_ambiguity_prior_strength", "od_ambiguity_prior_strength"),
        ("candidate_probe_path_count", "candidate_probe_path_count"),
        ("candidate_probe_corridor_family_count", "candidate_probe_corridor_family_count"),
        ("candidate_probe_objective_spread", "candidate_probe_objective_spread"),
        ("candidate_probe_nominal_margin", "candidate_probe_nominal_margin"),
        ("candidate_probe_engine_disagreement_prior", "candidate_probe_engine_disagreement_prior"),
        ("hard_case_prior", "hard_case_prior"),
        ("ambiguity_prior_source", "ambiguity_prior_source"),
        ("ambiguity_prior_sample_count", "ambiguity_prior_sample_count"),
    ):
        if enriched.get(target_key) not in (None, "", 0, "0", 0.0, "0.0"):
            continue
        value = payload.get(source_key)
        if value not in (None, ""):
            enriched[target_key] = value
    if enriched.get("od_ambiguity_index") in (None, "") and enriched.get("ambiguity_index") not in (None, ""):
        enriched["od_ambiguity_index"] = enriched.get("ambiguity_index")
    _update_prior_metadata(
        enriched,
        source=str(payload.get("ambiguity_prior_source") or "historical_results_bootstrap"),
        sample_count=int(_as_float(payload.get("ambiguity_prior_sample_count"), 0.0)),
        support_count=int(_as_float(payload.get("ambiguity_prior_support_count"), 0.0)),
    )
    return enriched


def _repo_local_geometry_prior_payload(row: dict[str, Any]) -> dict[str, Any]:
    distance_km = _as_float(row.get("straight_line_km"), 0.0)
    if distance_km <= 0.0:
        distance_km = _haversine_km(
            _as_float(row.get("origin_lat")),
            _as_float(row.get("origin_lon")),
            _as_float(row.get("destination_lat")),
            _as_float(row.get("destination_lon")),
        )
    distance_bin = str(row.get("distance_bin") or _distance_bin(distance_km))
    origin_region = str(row.get("origin_region_bucket") or _region_bucket(_as_float(row.get("origin_lat"))))
    destination_region = str(row.get("destination_region_bucket") or _region_bucket(_as_float(row.get("destination_lat"))))
    cross_region = origin_region != destination_region
    base_by_bin = {
        "0-30 km": {"path_count": 2, "family_count": 1, "objective_spread": 0.08, "nominal_margin": 0.46},
        "30-100 km": {"path_count": 3, "family_count": 2, "objective_spread": 0.16, "nominal_margin": 0.30},
        "100-250 km": {"path_count": 4, "family_count": 2, "objective_spread": 0.22, "nominal_margin": 0.20},
        "250+ km": {"path_count": 5, "family_count": 3, "objective_spread": 0.28, "nominal_margin": 0.14},
    }
    base = dict(base_by_bin.get(distance_bin, base_by_bin["30-100 km"]))
    path_count = max(int(base["path_count"]), int(_as_float(row.get("candidate_probe_path_count"), 0.0)))
    family_count = max(int(base["family_count"]), int(_as_float(row.get("candidate_probe_corridor_family_count"), 0.0)))
    objective_spread = max(float(base["objective_spread"]), _as_float(row.get("candidate_probe_objective_spread"), 0.0))
    nominal_margin = _as_float(row.get("candidate_probe_nominal_margin"), float("nan"))
    if not math.isfinite(nominal_margin) or nominal_margin <= 0.0:
        nominal_margin = float(base["nominal_margin"])
    if cross_region:
        family_count += 1
        objective_spread = min(1.0, objective_spread + 0.04)
        nominal_margin = max(0.08, nominal_margin - 0.03)
    toll_disagreement = _as_float(row.get("candidate_probe_toll_disagreement_rate"), 0.0)
    cheap_priors = _cheap_prior_features(
        path_count=path_count,
        family_count=family_count,
        objective_spread=objective_spread,
        nominal_margin=nominal_margin,
        toll_disagreement=toll_disagreement,
        top2_duration_gap=max(0.0, 1.0 - min(1.0, objective_spread)),
        top2_cost_gap=max(0.0, nominal_margin),
        top2_emissions_gap=max(0.0, 1.0 - min(1.0, objective_spread)),
    )
    raw_ambiguity_value = _raw_ambiguity_prior_value(row)
    ambiguity_value = round(raw_ambiguity_value if raw_ambiguity_value is not None else 0.0, 6)
    support_count = sum(
        1
        for value in (path_count, family_count, objective_spread, nominal_margin)
        if _is_nonzero_numeric(value)
    )
    return {
        "candidate_probe_path_count": int(path_count),
        "candidate_probe_corridor_family_count": int(family_count),
        "candidate_probe_objective_spread": round(objective_spread, 6),
        "candidate_probe_nominal_margin": round(nominal_margin, 6),
        "candidate_probe_toll_disagreement_rate": round(toll_disagreement, 6),
        "candidate_probe_path_sufficiency": cheap_priors["candidate_probe_path_sufficiency"],
        "candidate_probe_top2_gap_pressure": cheap_priors["candidate_probe_top2_gap_pressure"],
        "candidate_probe_engine_disagreement_prior": cheap_priors["candidate_probe_engine_disagreement_prior"],
        "hard_case_prior": cheap_priors["hard_case_prior"],
        "ambiguity_index": ambiguity_value,
        "od_ambiguity_index": ambiguity_value,
        "ambiguity_prior_support_count": support_count,
    }


def _probe_is_insufficient(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return True
    if not bool(payload.get("candidate_probe_accepted")):
        return True
    path_count = int(_as_float(payload.get("candidate_probe_path_count"), 0.0))
    family_count = int(_as_float(payload.get("candidate_probe_corridor_family_count"), 0.0))
    objective_spread = _as_float(payload.get("candidate_probe_objective_spread"), 0.0)
    ambiguity_value = _as_float(
        payload.get("od_ambiguity_index")
        if payload.get("od_ambiguity_index") not in (None, "")
        else payload.get("ambiguity_index"),
        0.0,
    )
    return path_count <= 1 or (family_count <= 1 and objective_spread <= 0.02 and ambiguity_value <= 0.02)


def _merge_probe_payload(
    enriched: dict[str, Any],
    payload: dict[str, Any],
    *,
    source_label: str | None,
) -> None:
    passthrough_fields = (
        "candidate_probe_accepted",
        "candidate_probe_reason_code",
        "candidate_probe_message",
        "candidate_probe_emitted_paths",
        "candidate_probe_generated_paths",
        "candidate_probe_explored_states",
        "candidate_probe_candidate_budget",
        "candidate_probe_effective_max_hops",
        "candidate_probe_effective_state_budget",
    )
    for field in passthrough_fields:
        if field in payload:
            enriched[field] = payload[field]
    for field in PRIOR_NUMERIC_FIELDS:
        if field not in payload:
            continue
        incoming = payload.get(field)
        current = enriched.get(field)
        if _is_missing_or_zero(current) or _is_nonzero_numeric(incoming):
            enriched[field] = incoming
    if enriched.get("od_ambiguity_index") in (None, "") and enriched.get("ambiguity_index") not in (None, ""):
        enriched["od_ambiguity_index"] = enriched.get("ambiguity_index")
    _update_prior_metadata(enriched, source=source_label)


def _apply_repo_local_prior_backfill(
    enriched: dict[str, Any],
    *,
    source_label: str,
) -> None:
    payload = _repo_local_geometry_prior_payload(enriched)
    for key, value in payload.items():
        current = enriched.get(key)
        if key in PRIOR_NUMERIC_FIELDS:
            if _is_missing_or_zero(current):
                enriched[key] = value
        elif current in (None, ""):
            enriched[key] = value
    if enriched.get("od_ambiguity_index") in (None, "") and enriched.get("ambiguity_index") not in (None, ""):
        enriched["od_ambiguity_index"] = enriched.get("ambiguity_index")
    _update_prior_metadata(
        enriched,
        source=source_label,
        support_count=int(_as_float(payload.get("ambiguity_prior_support_count"), 0.0)),
    )


def _apply_engine_augmented_prior(
    enriched: dict[str, Any],
    *,
    source_label: str,
) -> bool:
    payload = _engine_augmented_prior_payload(enriched)
    if not isinstance(payload, dict):
        return False
    for key, value in payload.items():
        current = enriched.get(key)
        if key in PRIOR_NUMERIC_FIELDS:
            if _is_missing_or_zero(current) or _is_nonzero_numeric(value):
                enriched[key] = value
        elif current in (None, ""):
            enriched[key] = value
    _update_prior_metadata(
        enriched,
        source=source_label,
        sample_count=int(_as_float(payload.get("ambiguity_prior_sample_count"), 0.0)),
        support_count=int(_as_float(payload.get("ambiguity_prior_support_count"), 0.0)),
    )
    return True


def _enrich_row(
    row: dict[str, Any],
    *,
    probe_max_paths: int,
    bootstrap_index: dict[str, dict[str, Any]] | None,
    use_graph_probe: bool,
    use_engine_probe: bool,
    recompute_cheap_priors: bool = False,
) -> dict[str, Any]:
    origin = {"lat": _as_float(row.get("origin_lat")), "lon": _as_float(row.get("origin_lon"))}
    destination = {"lat": _as_float(row.get("destination_lat")), "lon": _as_float(row.get("destination_lon"))}
    distance_km = _as_float(row.get("straight_line_km"))
    if distance_km <= 0.0:
        distance_km = _haversine_km(origin["lat"], origin["lon"], destination["lat"], destination["lon"])
    enriched = _apply_bootstrap_prior(dict(row), bootstrap_index=bootstrap_index)
    if enriched.get("od_ambiguity_index") in (None, "") and enriched.get("ambiguity_index") not in (None, ""):
        enriched["od_ambiguity_index"] = enriched.get("ambiguity_index")
    enriched["straight_line_km"] = round(distance_km, 6)
    enriched["distance_bin"] = str(row.get("distance_bin") or _distance_bin(distance_km))
    enriched["origin_region_bucket"] = str(row.get("origin_region_bucket") or _region_bucket(origin["lat"]))
    enriched["destination_region_bucket"] = str(row.get("destination_region_bucket") or _region_bucket(destination["lat"]))
    enriched["corridor_bucket"] = str(row.get("corridor_bucket") or _corridor_bucket(origin, destination))

    def _apply_cheap_prior_backfill() -> None:
        if _has_existing_graph_probe_signal(enriched):
            merged_source = _merge_prior_sources(enriched.get("ambiguity_prior_source"), "routing_graph_probe")
            if merged_source:
                enriched["ambiguity_prior_source"] = merged_source
        prior_features = _cheap_prior_features(
            path_count=int(_as_float(enriched.get("candidate_probe_path_count"), 0.0)),
            family_count=int(_as_float(enriched.get("candidate_probe_corridor_family_count"), 0.0)),
            objective_spread=_as_float(enriched.get("candidate_probe_objective_spread"), 0.0),
            nominal_margin=_as_float(enriched.get("candidate_probe_nominal_margin"), 0.0),
            toll_disagreement=_as_float(enriched.get("candidate_probe_toll_disagreement_rate"), 0.0),
        )
        filled_any = False
        for key, value in prior_features.items():
            if recompute_cheap_priors or _is_missing_or_zero(enriched.get(key)):
                enriched[key] = value
                filled_any = True
        if filled_any and not str(enriched.get("ambiguity_prior_source") or "").strip():
            _update_prior_metadata(enriched, source="existing_corpus")

    def _needs_core_prior_backfill() -> bool:
        return any(
            _is_missing_or_zero(enriched.get(key))
            for key in (
                "od_ambiguity_index",
                "candidate_probe_engine_disagreement_prior",
                "hard_case_prior",
            )
        )

    if not use_graph_probe:
        if use_engine_probe:
            _apply_engine_augmented_prior(enriched, source_label="engine_augmented_probe")
        if _needs_core_prior_backfill():
            _apply_repo_local_prior_backfill(enriched, source_label="repo_local_geometry_backfill")
        _apply_cheap_prior_backfill()
        _update_prior_metadata(enriched)
        return enriched
    feasibility = route_graph_od_feasibility(
        origin_lat=origin["lat"],
        origin_lon=origin["lon"],
        destination_lat=destination["lat"],
        destination_lon=destination["lon"],
    )
    reason_code = "ok" if bool(feasibility.get("ok")) else str(feasibility.get("reason_code") or "routing_graph_no_path")
    probe_payload = None
    if bool(feasibility.get("ok")):
        probe_payload = _candidate_probe_payload(
            origin=origin,
            destination=destination,
            feasibility_result=feasibility,
            candidate_probe_fn=route_graph_candidate_routes,
            max_paths=max(2, int(probe_max_paths)),
        )
    else:
        probe_payload = {
            "candidate_probe_accepted": False,
            "candidate_probe_reason_code": reason_code,
            "candidate_probe_message": str(feasibility.get("message") or reason_code),
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
        }
    probe_source_label = "routing_graph_probe" if bool(feasibility.get("ok")) else None
    _merge_probe_payload(enriched, probe_payload, source_label=probe_source_label)
    engine_probe_applied = False
    if use_engine_probe:
        engine_probe_applied = _apply_engine_augmented_prior(enriched, source_label="engine_augmented_probe")
    if not bool(feasibility.get("ok")):
        if _needs_core_prior_backfill():
            _apply_repo_local_prior_backfill(enriched, source_label="repo_local_geometry_backfill")
    elif _probe_is_insufficient(probe_payload):
        if use_engine_probe and not engine_probe_applied:
            _apply_repo_local_prior_backfill(enriched, source_label="repo_local_geometry_backfill")
    _apply_cheap_prior_backfill()
    enriched["accepted"] = bool(feasibility.get("ok"))
    enriched["route_graph_reason_code"] = reason_code
    enriched["route_graph_message"] = str(
        (probe_payload or {}).get("candidate_probe_message")
        or feasibility.get("message")
        or reason_code
    )
    enriched["origin_node_id"] = str(feasibility.get("origin_node_id", ""))
    enriched["destination_node_id"] = str(feasibility.get("destination_node_id", ""))
    enriched["origin_nearest_distance_m"] = round(_as_float(feasibility.get("origin_nearest_distance_m")), 3)
    enriched["destination_nearest_distance_m"] = round(_as_float(feasibility.get("destination_nearest_distance_m")), 3)
    _update_prior_metadata(enriched)
    return enriched


def enrich_rows(
    rows: list[dict[str, Any]],
    *,
    probe_max_paths: int = 6,
    bootstrap_index: dict[str, dict[str, Any]] | None = None,
    use_graph_probe: bool = True,
    use_engine_probe: bool = True,
    recompute_cheap_priors: bool = False,
) -> list[dict[str, Any]]:
    return [
        _enrich_row(
            dict(row),
            probe_max_paths=int(probe_max_paths),
            bootstrap_index=bootstrap_index,
            use_graph_probe=bool(use_graph_probe),
            use_engine_probe=bool(use_engine_probe),
            recompute_cheap_priors=bool(recompute_cheap_priors),
        )
        for row in rows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add deterministic route-graph ambiguity priors to an existing thesis corpus CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--probe-max-paths", type=int, default=6)
    parser.add_argument("--historical-results-csv", action="append", default=[])
    parser.add_argument("--skip-graph-probe", action="store_true")
    parser.add_argument("--skip-engine-probe", action="store_true")
    parser.add_argument("--recompute-cheap-priors", action="store_true")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args(argv)

    rows = _read_rows(Path(args.input_csv))
    bootstrap_index = _historical_bootstrap_index([Path(path) for path in list(args.historical_results_csv or [])])
    enriched = enrich_rows(
        rows,
        probe_max_paths=int(args.probe_max_paths),
        bootstrap_index=bootstrap_index,
        use_graph_probe=not bool(args.skip_graph_probe),
        use_engine_probe=not bool(args.skip_engine_probe),
        recompute_cheap_priors=bool(args.recompute_cheap_priors),
    )
    _write_rows(Path(args.output_csv), enriched)

    if args.summary_json:
        ambiguity_values = [
            _as_float(
                row.get("od_ambiguity_index")
                if row.get("od_ambiguity_index") not in (None, "")
                else row.get("ambiguity_index")
            )
            for row in enriched
        ]
        payload = {
            "row_count": len(enriched),
            "mean_ambiguity_index": round(sum(ambiguity_values) / float(len(ambiguity_values) or 1), 6),
            "max_ambiguity_index": round(max(ambiguity_values, default=0.0), 6),
            "mean_engine_disagreement_prior": round(
                sum(_as_float(row.get("candidate_probe_engine_disagreement_prior"), 0.0) for row in enriched)
                / float(len(enriched) or 1),
                6,
            ),
            "mean_hard_case_prior": round(
                sum(_as_float(row.get("hard_case_prior"), 0.0) for row in enriched) / float(len(enriched) or 1),
                6,
            ),
            "accepted_count": sum(1 for row in enriched if bool(row.get("accepted"))),
            "bootstrap_prior_count": sum(
                1 for row in enriched if "historical_results_bootstrap" in str(row.get("ambiguity_prior_source") or "")
            ),
            "nonzero_ambiguity_prior_count": sum(1 for row in enriched if _is_nonzero_numeric(row.get("od_ambiguity_index")) or _is_nonzero_numeric(row.get("ambiguity_index"))),
            "nonzero_engine_prior_count": sum(1 for row in enriched if _is_nonzero_numeric(row.get("candidate_probe_engine_disagreement_prior"))),
            "nonzero_hard_case_prior_count": sum(1 for row in enriched if _is_nonzero_numeric(row.get("hard_case_prior"))),
            "mean_od_ambiguity_confidence": round(
                sum(_as_float(row.get("od_ambiguity_confidence"), 0.0) for row in enriched) / float(len(enriched) or 1),
                6,
            ),
            "mean_od_ambiguity_source_count": round(
                sum(_as_float(row.get("od_ambiguity_source_count"), 0.0) for row in enriched) / float(len(enriched) or 1),
                6,
            ),
            "mean_od_ambiguity_source_support_strength": round(
                sum(_as_float(row.get("od_ambiguity_source_support_strength"), 0.0) for row in enriched) / float(len(enriched) or 1),
                6,
            ),
            "ambiguity_prior_source_mix": {
                source: sum(1 for row in enriched if source in str(row.get("ambiguity_prior_source") or ""))
                for source in PRIOR_SOURCE_ORDER
                if any(source in str(row.get("ambiguity_prior_source") or "") for row in enriched)
            },
            "od_ambiguity_source_mix": {
                source: sum(
                    _parse_source_mix_counts(row.get("od_ambiguity_source_mix")).get(source, 0)
                    for row in enriched
                )
                for source in PRIOR_SOURCE_ORDER
                if any(
                    _parse_source_mix_counts(row.get("od_ambiguity_source_mix")).get(source, 0) > 0
                    for row in enriched
                )
            },
            "probe_fields": list(PROBE_FIELDS),
        }
        Path(args.summary_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
