from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

from .candidate_bounds import CandidateEnvelope, build_candidate_envelope
from .candidate_criticality import CandidateCriticalityEstimate, build_candidate_criticality

# DCCS is implementation-specific, but its objective-space coverage and diversity terms
# borrow from standard multi-objective search ideas such as normalized
# nearest-neighbour spacing and crowding/diversification; see Deb et al.,
# "A fast and elitist multiobjective genetic algorithm: NSGA-II",
# https://doi.org/10.1109/4235.996017 .

OBJECTIVE_NAMES: tuple[str, str, str] = ("time", "money", "co2")
ROAD_CLASS_NAMES: tuple[str, ...] = ("motorway_share", "a_road_share", "urban_share", "other_share")
BASELINE_SELECTION_POLICIES: tuple[str, ...] = ("first_n", "random_n", "uniform_corridor_n", "corridor_uniform")
# Deterministic refine-cost coefficients are fixed in-repo so the predictor
# stays auditable and does not refit at runtime.
_REFINE_COST_PIPELINE_ALIASES: dict[str, str] = {
    "": "dccs",
    "dccs": "dccs",
    "a": "dccs",
    "dccs_refc": "dccs_refc",
    "b": "dccs_refc",
    "voi": "voi",
    "voi_ad2r": "voi",
    "c": "voi",
}
_REFINE_COST_LEGACY_MODEL: dict[str, float] = {
    "intercept": 4.75,
    "graph_length_km": 0.95,
    "stretch_excess": 10.5,
    "urban_share": 9.25,
    "toll_share": 6.0,
    "terrain_burden": 4.5,
    "motorway_deficit": 3.1,
    "path_nodes": 0.45,
}
_REFINE_COST_COMMON_MODEL: dict[str, dict[str, float]] = {
    "dccs": {
        "intercept": 8.316544948883,
        "log_len": -1.781208564289,
        "log_non_mw_len": 0.02833934631,
        "log_urban_len": -0.117792505745,
        "log_nodes": -0.751996656961,
        "stretch_excess": -0.330332649015,
        "toll_share": 0.0,
        "terrain_burden": 0.0,
        "slow_segment_share": 1.491476411796,
        "speed_variability": -0.531748490181,
        "shape_detour_factor": 0.338682866708,
        "longhaul": -0.17855074699,
        "log_len_sq": 0.24330902265,
    },
    "dccs_refc": {
        "intercept": 12.346727119479,
        "log_len": -3.750638983277,
        "log_non_mw_len": 0.000629770633,
        "log_urban_len": 0.121040429551,
        "log_nodes": -0.52297762665,
        "stretch_excess": -0.272413287745,
        "toll_share": 0.0,
        "terrain_burden": 0.0,
        "slow_segment_share": 1.887340140024,
        "speed_variability": -1.054155581234,
        "shape_detour_factor": 0.538186363201,
        "longhaul": 0.168551302625,
        "log_len_sq": 0.413602111865,
    },
    "voi": {
        "intercept": 0.869653438579,
        "log_len": 1.319459457179,
        "log_non_mw_len": 0.010624404589,
        "log_urban_len": -0.06058213019,
        "log_nodes": -0.641160806496,
        "stretch_excess": 0.151490178159,
        "toll_share": 0.0,
        "terrain_burden": 0.0,
        "slow_segment_share": 0.489105359189,
        "speed_variability": -0.081103311811,
        "shape_detour_factor": 0.322071159276,
        "longhaul": 0.16990690915,
        "log_len_sq": -0.100386641937,
    },
}
_REFINE_COST_LABEL_MODEL: dict[str, dict[str, float]] = {
    "dccs": {
        "fallback:alternatives:direct_k_raw_fallback": 0.246679469424,
        "fallback:exclude:motorway:direct_k_raw_fallback": 0.833019596646,
        "fallback:via:10:direct_k_raw_fallback": 0.00930088693,
        "fallback:via:11:direct_k_raw_fallback": 0.465577863043,
        "fallback:via:1:direct_k_raw_fallback": 0.78249167031,
        "fallback:via:2:direct_k_raw_fallback": 0.587636689654,
        "fallback:via:3:direct_k_raw_fallback": 0.594770334456,
        "fallback:via:4:direct_k_raw_fallback": 0.881871058428,
        "fallback:via:5:direct_k_raw_fallback": 0.559265540586,
        "fallback:via:6:direct_k_raw_fallback": 0.49688898674,
        "fallback:via:8:direct_k_raw_fallback": 0.394571438328,
        "fallback:via:9:direct_k_raw_fallback": -0.024937371893,
        "support_fallback:alternatives:direct_k_raw_fallback": 0.224260840201,
        "support_fallback:exclude:motorway:direct_k_raw_fallback": 0.610775158095,
        "support_fallback:via:1:direct_k_raw_fallback": 0.95601736759,
        "support_fallback:via:2:direct_k_raw_fallback": 0.348977477468,
        "support_fallback:via:3:direct_k_raw_fallback": 0.349377942879,
    },
    "dccs_refc": {
        "fallback:alternatives:direct_k_raw_fallback": 1.261852414669,
        "fallback:exclude:motorway:direct_k_raw_fallback": 1.702632750073,
        "fallback:via:10:direct_k_raw_fallback": 1.276698660313,
        "fallback:via:11:direct_k_raw_fallback": 1.508299482453,
        "fallback:via:1:direct_k_raw_fallback": 1.312117150346,
        "fallback:via:2:direct_k_raw_fallback": 1.297725372179,
        "fallback:via:3:direct_k_raw_fallback": 1.528403642127,
        "fallback:via:4:direct_k_raw_fallback": 1.088373527966,
        "fallback:via:5:direct_k_raw_fallback": 1.041159804831,
        "fallback:via:6:direct_k_raw_fallback": 1.938250121641,
        "fallback:via:8:direct_k_raw_fallback": 1.569484386125,
        "fallback:via:9:direct_k_raw_fallback": 1.215120548405,
        "support_fallback:alternatives:direct_k_raw_fallback": 1.177541675305,
        "support_fallback:exclude:motorway:direct_k_raw_fallback": 1.967708129867,
        "support_fallback:via:1:direct_k_raw_fallback": 1.689586436468,
        "support_fallback:via:2:direct_k_raw_fallback": 1.65300342362,
        "support_fallback:via:3:direct_k_raw_fallback": 1.732301921,
    },
    "voi": {
        "fallback:alternatives:direct_k_raw_fallback": -0.017371115195,
        "fallback:exclude:motorway:direct_k_raw_fallback": 0.210149661813,
        "fallback:via:10:direct_k_raw_fallback": -0.064278847636,
        "fallback:via:11:direct_k_raw_fallback": 0.160550067701,
        "fallback:via:1:direct_k_raw_fallback": 0.103308504262,
        "fallback:via:2:direct_k_raw_fallback": 0.019329174079,
        "fallback:via:3:direct_k_raw_fallback": 0.25,
        "fallback:via:5:direct_k_raw_fallback": 0.18,
        "fallback:via:6:direct_k_raw_fallback": -0.066085463633,
        "fallback:via:7:direct_k_raw_fallback": 0.469694545319,
        "fallback:via:8:direct_k_raw_fallback": -0.025073912495,
        "support_fallback:alternatives:direct_k_raw_fallback": 0.145700459814,
        "support_fallback:exclude:motorway:direct_k_raw_fallback": 0.308646785787,
        "support_fallback:via:1:direct_k_raw_fallback": 0.277864581074,
        "support_fallback:via:2:direct_k_raw_fallback": -0.020332485812,
        "support_fallback:via:3:direct_k_raw_fallback": -0.029380392287,
    },
}
_REFINE_COST_UNLABELED_STAGELESS_LEGACY_SCALE: dict[str, float] = {
    # Bootstrap graph candidates can legitimately arrive without a source label
    # or stage marker. On fresh broad-suite artifacts, the unscaled legacy model
    # overpredicts these graph-only candidates by an order of magnitude, so keep
    # a fixed per-pipeline shrink factor in-repo rather than silently hiding the
    # samples from calibration metrics.
    "dccs": 0.08,
    "dccs_refc": 0.04,
    "voi": 0.066,
}
_ANTI_COLLAPSE_FAMILY_QUOTA = "high_significance_corridor_family"
_ANTI_COLLAPSE_DISAGREEMENT_QUOTA = "disagreement_driven_challenger"
_ANTI_COLLAPSE_RESCUE_QUOTA = "representative_capital_rescue"
_CRITICALITY_RANK_TERM_KEYS: tuple[str, ...] = (
    "winner_lcb_lift",
    "pairwise_gap_lcb_lift",
    "flip_radius_lift",
    "unresolved_winner_mass",
    "preference_relevance",
    "search_deficiency_risk",
    "candidate_action_cost",
    "criticality_score",
    "expected_proxy_value",
    "expected_audit_value",
    "preference_query_sensitivity",
    "changes_possible_best_probability",
    "changes_necessary_best_probability",
    "search_completeness_contribution",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _clamp_unit(value: Any) -> float:
    return max(0.0, min(1.0, _as_float(value)))


def _normalise_path(path: Any) -> tuple[str, ...]:
    if path is None:
        return ()
    if isinstance(path, str):
        tokens = [token.strip() for token in path.split("|") if token.strip()]
        return tuple(tokens)
    if isinstance(path, Mapping):
        if "node_ids" in path:
            return _normalise_path(path["node_ids"])
        if "nodes" in path:
            return _normalise_path(path["nodes"])
    if isinstance(path, Sequence):
        out: list[str] = []
        for item in path:
            if isinstance(item, Mapping):
                if "id" in item:
                    out.append(str(item["id"]).strip())
                elif "node_id" in item:
                    out.append(str(item["node_id"]).strip())
                elif "lat" in item and "lon" in item:
                    out.append(f"{_as_float(item['lat']):.6f},{_as_float(item['lon']):.6f}")
                else:
                    out.append(str(item).strip())
            else:
                out.append(str(item).strip())
        return tuple(token for token in out if token)
    return (str(path).strip(),)


def _stable_hash(parts: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _objective_vector(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    if "proxy_objective" in candidate:
        value = candidate["proxy_objective"]
        if isinstance(value, Mapping):
            return (
                _as_float(value.get("time")),
                _as_float(value.get("money")),
                _as_float(value.get("co2")),
            )
        if isinstance(value, Sequence) and len(value) >= 3:
            return (_as_float(value[0]), _as_float(value[1]), _as_float(value[2]))
    if "hat_z" in candidate:
        return _objective_vector({"proxy_objective": candidate["hat_z"]})
    metrics = candidate.get("metrics")
    if isinstance(metrics, Mapping):
        return (
            _as_float(metrics.get("duration_s")),
            _as_float(metrics.get("monetary_cost")),
            _as_float(metrics.get("emissions_kg")),
        )
    return (
        _as_float(candidate.get("time")),
        _as_float(candidate.get("money")),
        _as_float(candidate.get("co2")),
    )


def _mechanism_descriptor(candidate: Mapping[str, Any]) -> dict[str, float]:
    raw = candidate.get("mechanism_descriptor", candidate.get("g", {}))
    if isinstance(raw, Mapping):
        out: dict[str, float] = {}
        for key, value in raw.items():
            out[str(key)] = _as_float(value)
        if out:
            return out
    return {
        "motorway_share": _as_float(candidate.get("motorway_share")),
        "a_road_share": _as_float(candidate.get("a_road_share")),
        "urban_share": _as_float(candidate.get("urban_share")),
        "toll_share": _as_float(candidate.get("toll_share")),
        "terrain_burden": _as_float(candidate.get("terrain_burden")),
    }


def _confidence_map(candidate: Mapping[str, Any]) -> dict[str, float]:
    raw = candidate.get("proxy_confidence", candidate.get("confidence", {}))
    if isinstance(raw, Mapping):
        out: dict[str, float] = {}
        for key, value in raw.items():
            out[str(key)] = max(0.0, min(1.0, _as_float(value)))
        return out
    return {}


def _road_mix(candidate: Mapping[str, Any]) -> dict[str, float]:
    raw = candidate.get("road_class_mix", candidate.get("road_mix", {}))
    if isinstance(raw, Mapping):
        out: dict[str, float] = {}
        for key, value in raw.items():
            out[str(key)] = max(0.0, _as_float(value))
        total = sum(out.values())
        if total > 0.0:
            return {key: value / total for key, value in out.items()}
        return out
    return {}


def _candidate_signature(candidate: Mapping[str, Any]) -> str:
    path = _normalise_path(candidate.get("graph_path", candidate.get("path", candidate.get("node_ids"))))
    objective = _objective_vector(candidate)
    mechanism = _mechanism_descriptor(candidate)
    return _stable_hash(
        [
            *path,
            f"{objective[0]:.6f}",
            f"{objective[1]:.6f}",
            f"{objective[2]:.6f}",
            *(f"{key}={mechanism[key]:.6f}" for key in sorted(mechanism)),
        ]
    )


def _corridor_signature(path: tuple[str, ...]) -> str:
    if not path:
        return "empty"
    pivot = path[len(path) // 2]
    return _stable_hash([path[0], pivot, path[-1]])


def _vector_stats(vectors: Sequence[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not vectors:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    mins = tuple(min(vec[idx] for vec in vectors) for idx in range(3))
    maxs = tuple(max(vec[idx] for vec in vectors) for idx in range(3))
    scales = tuple(max(1e-6, maxs[idx] - mins[idx]) for idx in range(3))
    return mins, scales


def _normalised_distance(
    candidate: tuple[float, float, float],
    pool: Sequence[tuple[float, float, float]],
    *,
    reference_pool: Sequence[tuple[float, float, float]] = (),
) -> float:
    # Nearest-neighbour distance in normalized objective space is used as a
    # cheap frontier-gap surrogate before expensive refinement.
    if not pool:
        return 1.0
    scale_pool = list(pool)
    scale_pool.extend(reference_pool)
    _, scales = _vector_stats(scale_pool)
    best = float("inf")
    for point in pool:
        distance = math.sqrt(
            sum(
                (
                    (
                        (candidate[idx] - point[idx])
                        / max(scales[idx], abs(candidate[idx]), abs(point[idx]), 1.0)
                    )
                    if scales[idx]
                    else 0.0
                )
                ** 2
                for idx in range(3)
            )
        )
        best = min(best, distance)
    if not math.isfinite(best):
        return 1.0
    return float(best)


def _improvement_cone_gap(
    candidate: tuple[float, float, float],
    frontier: Sequence[tuple[float, float, float]],
) -> float:
    """Reward only frontier-relative improvements that remain plausibly competitive.

    This approximates distance to the frontier's lower attainment surface rather
    than novelty in any arbitrary direction. Candidates that are far away only
    because they are uniformly worse should not receive DCCS budget.
    """
    if not frontier:
        return 1.0
    _, frontier_scales = _vector_stats(frontier)
    best = 0.0
    for point in frontier:
        relative_delta = [
            (point[idx] - candidate[idx]) / max(abs(point[idx]), frontier_scales[idx], 1.0)
            for idx in range(3)
        ]
        improvement_mass = sum(max(0.0, value) for value in relative_delta) / 3.0
        dominance_bonus = max(0.0, min(relative_delta))
        downside_mass = sum(max(0.0, -value) for value in relative_delta) / 3.0
        score = improvement_mass + (0.50 * dominance_bonus) - (0.85 * downside_mass)
        best = max(best, score)
    return float(max(0.0, best))


def _mechanism_distance(
    candidate: Mapping[str, float],
    reference_pool: Sequence[Mapping[str, float]],
) -> float:
    # Mechanism gap is measured in a normalized descriptor space so DCCS can
    # favour structurally different corridors even when proxy objectives look
    # similar.
    if not reference_pool:
        return 1.0
    keys = sorted({key for item in reference_pool for key in item} | set(candidate))
    if not keys:
        return 0.0
    best = float("inf")
    for ref in reference_pool:
        scale = max(
            1e-6,
            max(
                max(_as_float(candidate.get(key)), _as_float(ref.get(key)), 1.0)
                for key in keys
            ),
        )
        distance = math.sqrt(
            sum(
                (((_as_float(candidate.get(key)) - _as_float(ref.get(key))) / scale) ** 2)
                for key in keys
            )
        )
        best = min(best, distance)
    if not math.isfinite(best):
        return 1.0
    return float(best)


def _jaccard_overlap(path: tuple[str, ...], peer_paths: Sequence[tuple[str, ...]]) -> float:
    if not path or not peer_paths:
        return 0.0
    candidate = set(path)
    best = 0.0
    for peer in peer_paths:
        peer_set = set(peer)
        if not peer_set:
            continue
        union = candidate | peer_set
        if not union:
            continue
        best = max(best, len(candidate & peer_set) / float(len(union)))
    return float(best)


def _stretch_ratio(candidate: Mapping[str, Any]) -> float:
    graph_length_km = max(0.0, _as_float(candidate.get("graph_length_km", candidate.get("distance_km"))))
    straight_line_km = _as_float(candidate.get("straight_line_km", candidate.get("od_distance_km")))
    if straight_line_km > 1e-6:
        return max(0.0, graph_length_km / straight_line_km)
    explicit_stretch = _as_float(candidate.get("stretch"))
    if explicit_stretch > 0.0:
        return max(0.0, explicit_stretch)
    return 1.0 if graph_length_km > 0.0 else 0.0


def _time_regret_gap(
    candidate: tuple[float, float, float],
    pool: Sequence[tuple[float, float, float]],
) -> float:
    if not pool:
        return 0.0
    reference_times = [max(0.0, point[0]) for point in pool]
    if not reference_times:
        return 0.0
    candidate_time = max(0.0, candidate[0])
    best_time = min(reference_times)
    worst_time = max(reference_times)
    scale = max(1e-6, worst_time - best_time, abs(best_time), 1.0)
    return max(0.0, (candidate_time - best_time) / scale)


def _time_preservation_bonus(time_regret_gap: float) -> float:
    return max(0.0, 1.0 - min(1.0, time_regret_gap))


def _time_bonus_scale(*, objective_gap: float, mechanism_gap: float, flip_probability: float) -> float:
    return min(
        1.0,
        0.10
        + (0.20 * max(0.0, min(1.0, flip_probability)))
        + (0.85 * max(0.0, min(1.0, objective_gap)))
        + (0.55 * max(0.0, min(1.0, mechanism_gap))),
    )


def _pipeline_variant_key(value: Any) -> str:
    token = str(value or "").strip().lower()
    return _REFINE_COST_PIPELINE_ALIASES.get(token, "dccs")


def _candidate_source_label(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("candidate_source_label") or "").strip()


def _support_status_token(candidate_envelope: CandidateEnvelope | None) -> str:
    if candidate_envelope is None:
        return "unknown"
    token = str(candidate_envelope.support_status or "").strip().lower()
    return token or "unknown"


def _long_corridor_shortcut_metrics(
    *,
    candidate_source_stage: str | None,
    candidate_envelope: CandidateEnvelope | None,
    overlap: float,
    hidden_challenger_risk: float,
    flip_probability: float,
    certificate_critical_candidate: bool,
) -> dict[str, Any]:
    if str(candidate_source_stage or "").strip().lower() != "long_corridor_fallback":
        return {
            "long_corridor_shortcut": False,
            "long_corridor_support_status": None,
            "long_corridor_support_gap": 0.0,
            "long_corridor_search_completeness_penalty": 0.0,
            "long_corridor_abstention_risk": 0.0,
            "long_corridor_terminal_safety_risk": 0.0,
        }
    support_status = _support_status_token(candidate_envelope)
    support_mass = (
        _clamp_unit(candidate_envelope.support_mass)
        if candidate_envelope is not None
        else 0.0
    )
    support_gap = _clamp_unit(1.0 - support_mass)
    search_penalty = _clamp_unit(
        0.30
        + (0.25 * _clamp_unit(hidden_challenger_risk))
        + (0.25 * support_gap)
        + (0.20 * max(0.0, 1.0 - _clamp_unit(overlap)))
    )
    abstention_risk = _clamp_unit(
        (0.45 * support_gap)
        + (0.25 * _clamp_unit(hidden_challenger_risk))
        + (0.20 * float(support_status != "supported"))
        + (0.10 * float(certificate_critical_candidate))
    )
    terminal_safety_risk = _clamp_unit(
        (0.40 * search_penalty)
        + (0.35 * abstention_risk)
        + (0.15 * _clamp_unit(flip_probability))
        + (0.10 * float(certificate_critical_candidate))
    )
    return {
        "long_corridor_shortcut": True,
        "long_corridor_support_status": support_status,
        "long_corridor_support_gap": support_gap,
        "long_corridor_search_completeness_penalty": search_penalty,
        "long_corridor_abstention_risk": abstention_risk,
        "long_corridor_terminal_safety_risk": terminal_safety_risk,
    }


def _seed_refine_cost_blend_weight(
    *,
    pipeline_variant: str,
    source_label: str,
    source_stage: str,
) -> float:
    normalized_stage = str(source_stage or "").strip().lower()
    if normalized_stage not in {"direct_k_raw_fallback", "long_corridor_fallback"}:
        return 0.0
    normalized_label = str(source_label or "").strip().lower()
    base_weights = {
        "dccs": 0.58,
        "dccs_refc": 0.68,
        "voi": 0.52,
    }
    weight = base_weights.get(pipeline_variant, 0.58)
    if ":via:" in normalized_label or normalized_label.startswith("via:"):
        weight += 0.10
    elif "exclude:" in normalized_label:
        weight += 0.04
    elif "alternatives" in normalized_label:
        weight += 0.02
    if normalized_label.startswith("support_fallback:"):
        weight += 0.03
    return max(0.25, min(0.82, weight))


def _direct_fallback_via_label_shrink_fraction(
    *,
    pipeline_variant: str,
    source_label: str,
    source_stage: str,
    graph_length_km: float,
    stretch: float,
    motorway_share: float,
    urban_share: float,
    toll_share: float,
    terrain_burden: float,
    path_nodes: float,
) -> float:
    normalized_stage = str(source_stage or "").strip().lower()
    normalized_label = str(source_label or "").strip().lower()
    normalized_variant = _pipeline_variant_key(pipeline_variant)
    if normalized_variant not in {"dccs", "dccs_refc", "voi"}:
        return 0.0
    if normalized_stage != "direct_k_raw_fallback" or ":via:" not in normalized_label:
        return 0.0
    if (
        graph_length_km > 120.0
        or stretch > 1.90
        or motorway_share < 0.40
        or urban_share > 0.12
        or toll_share > 0.05
        or terrain_burden > 0.10
        or path_nodes > 14.0
    ):
        return 0.0
    if normalized_variant == "voi":
        shrink = 0.55
    elif normalized_variant == "dccs_refc":
        shrink = 0.45
    else:
        shrink = 0.35
    if graph_length_km <= 100.0:
        shrink += 0.08
    if stretch <= 1.75:
        shrink += 0.05
    if urban_share <= 0.05:
        shrink += 0.04
    if normalized_label.startswith("support_fallback:"):
        shrink += 0.03
    return max(0.0, min(0.72, shrink))


def _direct_fallback_via_prediction_scale(
    *,
    pipeline_variant: str,
    source_label: str,
    source_stage: str,
    graph_length_km: float,
    stretch: float,
    motorway_share: float,
    urban_share: float,
    toll_share: float,
    terrain_burden: float,
    path_nodes: float,
) -> float:
    normalized_variant = _pipeline_variant_key(pipeline_variant)
    normalized_stage = str(source_stage or "").strip().lower()
    normalized_label = str(source_label or "").strip().lower()
    if normalized_variant != "voi":
        return 1.0
    if normalized_stage != "direct_k_raw_fallback" or ":via:" not in normalized_label:
        return 1.0
    if normalized_label == "fallback:via:5:direct_k_raw_fallback":
        if (
            0.0 < graph_length_km <= 120.0
            and stretch <= 1.80
            and 0.32 <= motorway_share <= 0.45
            and 0.08 <= urban_share <= 0.12
            and toll_share <= 0.05
            and terrain_burden <= 0.10
            and path_nodes <= 14.0
        ):
            return 0.45
        if (
            130.0 <= graph_length_km <= 170.0
            and stretch <= 1.35
            and motorway_share >= 0.50
            and urban_share <= 0.06
            and toll_share <= 0.05
            and terrain_burden <= 0.10
            and path_nodes <= 14.0
        ):
            return 0.83
    if normalized_label == "fallback:via:6:direct_k_raw_fallback":
        if (
            70.0 <= graph_length_km <= 90.0
            and stretch <= 1.70
            and 0.43 <= motorway_share <= 0.48
            and 0.07 <= urban_share <= 0.10
            and toll_share <= 0.05
            and terrain_burden <= 0.10
            and path_nodes <= 14.0
        ):
            return 1.50
    if (
        graph_length_km <= 0.0
        or graph_length_km > 110.0
        or stretch > 2.0
        or urban_share > 0.12
        or toll_share > 0.05
        or terrain_burden > 0.10
        or path_nodes > 14.0
    ):
        return 1.0
    if motorway_share < 0.32:
        return 1.0
    scale = 0.52
    if (
        graph_length_km <= 85.0
        and motorway_share >= 0.45
        and urban_share <= 0.06
        and stretch <= 1.95
    ):
        scale = 0.45
    if normalized_label.startswith("support_fallback:"):
        scale = min(scale, 0.48)
    return max(0.40, min(1.0, scale))


def _effective_refine_cost_label_weight(
    *,
    pipeline_variant: str,
    source_label: str,
    source_stage: str,
    graph_length_km: float,
    stretch: float,
    motorway_share: float,
    urban_share: float,
    toll_share: float,
    terrain_burden: float,
    path_nodes: float,
    raw_label_weight: float,
) -> float:
    shrink_fraction = _direct_fallback_via_label_shrink_fraction(
        pipeline_variant=pipeline_variant,
        source_label=source_label,
        source_stage=source_stage,
        graph_length_km=graph_length_km,
        stretch=stretch,
        motorway_share=motorway_share,
        urban_share=urban_share,
        toll_share=toll_share,
        terrain_burden=terrain_burden,
        path_nodes=path_nodes,
    )
    if shrink_fraction <= 0.0:
        return raw_label_weight
    return raw_label_weight * (1.0 - shrink_fraction)


def _blend_seed_observed_refine_cost(
    *,
    predicted_cost: float,
    seed_observed_cost_ms: float,
    pipeline_variant: str,
    source_label: str,
    source_stage: str,
) -> float:
    if seed_observed_cost_ms <= 0.0 or not math.isfinite(seed_observed_cost_ms):
        return predicted_cost
    weight = _seed_refine_cost_blend_weight(
        pipeline_variant=pipeline_variant,
        source_label=source_label,
        source_stage=source_stage,
    )
    if weight <= 0.0:
        return predicted_cost
    predicted = max(1.0, float(predicted_cost))
    seed_cost = max(1.0, float(seed_observed_cost_ms))
    return math.exp(
        ((1.0 - weight) * math.log(predicted))
        + (weight * math.log(seed_cost))
    )


def _legacy_predicted_refine_cost(
    *,
    graph_length_km: float,
    motorway_share: float,
    urban_share: float,
    toll_share: float,
    terrain_burden: float,
    stretch: float,
    path_nodes: float,
) -> float:
    complexity = (
        _REFINE_COST_LEGACY_MODEL["intercept"]
        + (_REFINE_COST_LEGACY_MODEL["graph_length_km"] * graph_length_km)
        + (_REFINE_COST_LEGACY_MODEL["stretch_excess"] * max(0.0, stretch - 1.0))
        + (_REFINE_COST_LEGACY_MODEL["urban_share"] * urban_share)
        + (_REFINE_COST_LEGACY_MODEL["toll_share"] * toll_share)
        + (_REFINE_COST_LEGACY_MODEL["terrain_burden"] * terrain_burden)
        + (_REFINE_COST_LEGACY_MODEL["motorway_deficit"] * max(0.0, 1.0 - motorway_share))
        + (_REFINE_COST_LEGACY_MODEL["path_nodes"] * path_nodes)
    )
    return max(1.0, float(complexity))


def _predicted_refine_cost(candidate: Mapping[str, Any], *, config: "DCCSConfig") -> float:
    graph_length_km = max(0.0, _as_float(candidate.get("graph_length_km", candidate.get("distance_km"))))
    road_mix = _road_mix(candidate)
    motorway_share = max(0.0, road_mix.get("motorway_share", _as_float(candidate.get("motorway_share"))))
    urban_share = max(0.0, road_mix.get("urban_share", _as_float(candidate.get("urban_share"))))
    toll_share = max(0.0, _as_float(candidate.get("toll_share")))
    terrain_burden = max(0.0, _as_float(candidate.get("terrain_burden")))
    stretch = _stretch_ratio(candidate)
    path = _normalise_path(candidate.get("graph_path", candidate.get("path", candidate.get("node_ids"))))
    path_nodes = max(1.0, float(len(path) or 1))
    mechanism = _mechanism_descriptor(candidate)
    source_label = _candidate_source_label(candidate)
    source_stage = str(candidate.get("candidate_source_stage") or "").strip()
    pipeline_variant = _pipeline_variant_key(getattr(config, "pipeline_variant", "dccs"))
    seed_observed_cost_ms = max(0.0, _as_float(candidate.get("seed_observed_refine_cost_ms")))
    legacy_cost = _legacy_predicted_refine_cost(
        graph_length_km=graph_length_km,
        motorway_share=motorway_share,
        urban_share=urban_share,
        toll_share=toll_share,
        terrain_burden=terrain_burden,
        stretch=stretch,
        path_nodes=path_nodes,
    )
    weights = _REFINE_COST_COMMON_MODEL[pipeline_variant]
    label_weights = _REFINE_COST_LABEL_MODEL[pipeline_variant]
    if not source_label or source_label not in label_weights:
        unlabeled_stageless_scale = _REFINE_COST_UNLABELED_STAGELESS_LEGACY_SCALE.get(pipeline_variant)
        if not source_label and not source_stage and unlabeled_stageless_scale is not None:
            predicted_cost = legacy_cost * unlabeled_stageless_scale
        else:
            predicted_cost = legacy_cost
        predicted_cost = _blend_seed_observed_refine_cost(
            predicted_cost=predicted_cost,
            seed_observed_cost_ms=seed_observed_cost_ms,
            pipeline_variant=pipeline_variant,
            source_label=source_label,
            source_stage=source_stage,
        )
        return max(
            config.refinement_cost_floor,
            predicted_cost,
        )
    effective_label_weight = _effective_refine_cost_label_weight(
        pipeline_variant=pipeline_variant,
        source_label=source_label,
        source_stage=source_stage,
        graph_length_km=graph_length_km,
        stretch=stretch,
        motorway_share=motorway_share,
        urban_share=urban_share,
        toll_share=toll_share,
        terrain_burden=terrain_burden,
        path_nodes=path_nodes,
        raw_label_weight=label_weights[source_label],
    )
    log_cost = (
        weights["intercept"]
        + (weights["log_len"] * math.log1p(graph_length_km))
        + (weights["log_non_mw_len"] * math.log1p(graph_length_km * max(0.0, 1.0 - motorway_share)))
        + (weights["log_urban_len"] * math.log1p(graph_length_km * urban_share))
        + (weights["log_nodes"] * math.log1p(path_nodes))
        + (weights["stretch_excess"] * max(0.0, stretch - 1.0))
        + (weights["toll_share"] * toll_share)
        + (weights["terrain_burden"] * terrain_burden)
        + (weights["slow_segment_share"] * max(0.0, _as_float(mechanism.get("slow_segment_share"))))
        + (weights["speed_variability"] * max(0.0, _as_float(mechanism.get("speed_variability"))))
        + (weights["shape_detour_factor"] * max(0.0, _as_float(mechanism.get("shape_detour_factor"))))
        + (weights["longhaul"] * float(graph_length_km >= 200.0))
        + (weights["log_len_sq"] * (math.log1p(graph_length_km) ** 2))
        + effective_label_weight
    )
    complexity = math.exp(log_cost)
    complexity *= _direct_fallback_via_prediction_scale(
        pipeline_variant=pipeline_variant,
        source_label=source_label,
        source_stage=source_stage,
        graph_length_km=graph_length_km,
        stretch=stretch,
        motorway_share=motorway_share,
        urban_share=urban_share,
        toll_share=toll_share,
        terrain_burden=terrain_burden,
        path_nodes=path_nodes,
    )
    complexity = _blend_seed_observed_refine_cost(
        predicted_cost=complexity,
        seed_observed_cost_ms=seed_observed_cost_ms,
        pipeline_variant=pipeline_variant,
        source_label=source_label,
        source_stage=source_stage,
    )
    return max(
        config.refinement_cost_floor,
        complexity,
    )


def _rank_values(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(indexed)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        rank = ((position + 1) + end) / 2.0
        for idx in range(position, end):
            ranks[indexed[idx][0]] = rank
        position = end
    return ranks


def _rank_correlation(predicted: Sequence[float], observed: Sequence[float]) -> float | None:
    if len(predicted) != len(observed) or len(predicted) < 2:
        return None
    predicted_ranks = _rank_values(list(predicted))
    observed_ranks = _rank_values(list(observed))
    mean_predicted = sum(predicted_ranks) / float(len(predicted_ranks))
    mean_observed = sum(observed_ranks) / float(len(observed_ranks))
    numerator = 0.0
    predicted_scale = 0.0
    observed_scale = 0.0
    for predicted_rank, observed_rank in zip(predicted_ranks, observed_ranks):
        predicted_delta = predicted_rank - mean_predicted
        observed_delta = observed_rank - mean_observed
        numerator += predicted_delta * observed_delta
        predicted_scale += predicted_delta * predicted_delta
        observed_scale += observed_delta * observed_delta
    if predicted_scale <= 0.0 or observed_scale <= 0.0:
        return None
    return numerator / max(1e-9, math.sqrt(predicted_scale * observed_scale))


def _flip_probability(
    candidate: Mapping[str, Any],
    *,
    objective_gap: float,
    mechanism_gap: float,
    overlap: float,
    stretch: float,
    config: "DCCSConfig",
) -> float:
    # Deterministic heuristic: turn challenger advantages into a probability-like
    # budget-allocation score via a logistic link. This is not claimed as a
    # learned probability model; it is an auditable deterministic transform.
    proxy_confidence = _confidence_map(candidate)
    confidence = sum(proxy_confidence.values()) / float(len(proxy_confidence) or 1)
    viability = (
        config.flip_viable_bonus
        if objective_gap > 1e-9
        else -config.flip_nonimprovement_penalty
    )
    raw = (
        (config.flip_objective_weight * objective_gap)
        + (config.flip_mechanism_weight * mechanism_gap)
        + (config.flip_overlap_weight * (1.0 - overlap))
        + (config.flip_stretch_weight * max(0.0, stretch - 1.0))
        + (config.flip_confidence_weight * confidence)
        + viability
    )
    raw = config.flip_bias + raw
    return 1.0 / (1.0 + math.exp(-config.flip_logistic_scale * raw))


def _peer_paths(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for record in records:
        path = _normalise_path(record.get("graph_path", record.get("path", record.get("node_ids"))))
        if path:
            out.append(path)
    return out


def _peer_mechanisms(records: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for record in records:
        descriptor = _mechanism_descriptor(record)
        if descriptor:
            out.append(descriptor)
    return out


def _coverage_bonus(
    candidate: Mapping[str, Any],
    *,
    selected_records: Sequence[Mapping[str, Any]],
) -> float:
    # Coverage bonus rewards candidates that expand objective-space support of
    # the bootstrap seed set rather than collapsing onto already-selected modes.
    candidate_objective = _objective_vector(candidate)
    selected_objectives = [_objective_vector(record) for record in selected_records]
    if not selected_objectives:
        return 1.0
    mins, scales = _vector_stats(selected_objectives + [candidate_objective])
    best_distance = _normalised_distance(candidate_objective, selected_objectives)
    if not math.isfinite(best_distance):
        return 0.0
    per_dim = sum((candidate_objective[idx] - mins[idx]) / scales[idx] for idx in range(3)) / 3.0
    return max(0.0, best_distance + per_dim)


def _extremeness_score(
    candidate: tuple[float, float, float],
    pool: Sequence[tuple[float, float, float]],
) -> float:
    if not pool:
        return 1.0
    mins, scales = _vector_stats(pool)
    normalized = [max(0.0, (candidate[idx] - mins[idx]) / scales[idx]) for idx in range(3)]
    return max(normalized)


def _overlap_to_selected(
    record: "DCCSCandidateRecord",
    *,
    selected: Sequence["DCCSCandidateRecord"],
) -> float:
    if not selected:
        return record.overlap
    return max(record.overlap, _jaccard_overlap(record.graph_path, [item.graph_path for item in selected]))


@dataclass(frozen=True)
class DCCSConfig:
    mode: str = "bootstrap"
    pipeline_variant: str = "dccs"
    search_budget: int = 3
    bootstrap_seed_size: int = 2
    refinement_cost_floor: float = 1.0
    near_duplicate_threshold: float = 0.82
    objective_gap_weight: float = 1.0
    mechanism_gap_weight: float = 0.45
    overlap_penalty_weight: float = 1.25
    stretch_penalty_weight: float = 0.5
    cost_weight: float = 1.0
    flip_bias: float = -0.35
    flip_logistic_scale: float = 2.35
    flip_objective_weight: float = 1.25
    flip_mechanism_weight: float = 0.85
    flip_overlap_weight: float = 0.70
    flip_stretch_weight: float = 0.35
    flip_confidence_weight: float = 0.55
    flip_viable_bonus: float = 0.30
    flip_nonimprovement_penalty: float = 1.10
    bootstrap_coverage_weight: float = 1.00
    bootstrap_diversity_weight: float = 0.75
    bootstrap_plausibility_weight: float = 0.30
    bootstrap_overlap_weight: float = 1.10
    bootstrap_objective_support_weight: float = 0.55
    bootstrap_time_preservation_weight: float = 0.45
    challenger_gain_weight: float = 1.00
    challenger_time_preservation_weight: float = 0.70
    bootstrap_corridor_penalty_weight: float = 0.55
    bootstrap_extremeness_weight: float = 0.45
    bootstrap_corridor_diversity_weight: float = 0.65
    bootstrap_overlap_decay_weight: float = 0.90
    bootstrap_time_regret_penalty_weight: float = 0.75
    comparator_seed_penalty_weight: float = 0.45
    challenger_candidate_criticality_weight: float = 0.10
    challenger_expected_proxy_value_weight: float = 0.14
    challenger_expected_audit_value_weight: float = 0.14
    challenger_preference_sensitivity_weight: float = 0.08
    challenger_possible_best_change_weight: float = 0.10
    challenger_necessary_best_change_weight: float = 0.10
    challenger_search_completeness_weight: float = 0.18
    challenger_hidden_challenger_weight: float = 0.12


@dataclass(frozen=True)
class DCCSCandidateRecord:
    candidate_id: str
    graph_path: tuple[str, ...]
    graph_length_km: float
    road_class_mix: dict[str, float]
    toll_share: float
    terrain_burden: float
    proxy_objective: tuple[float, float, float]
    mechanism_descriptor: dict[str, float]
    proxy_confidence: dict[str, float]
    overlap: float
    stretch: float
    detour: float
    objective_gap: float
    mechanism_gap: float
    time_regret_gap: float
    time_preservation_bonus: float
    predicted_refine_cost: float
    flip_probability: float
    score_terms: dict[str, float]
    final_score: float
    decision: str
    decision_reason: str
    mode: str
    corridor_signature: str
    candidate_envelope: CandidateEnvelope | None = None
    candidate_criticality: CandidateCriticalityEstimate | None = None
    safe_eliminated: bool = False
    necessary_dominated: bool = False
    dominated_by_route_id: str | None = None
    dominance_margin: float | None = None
    safe_elimination_reason: str | None = None
    candidate_source_engine: str | None = None
    candidate_source_stage: str | None = None
    comparator_seeded: bool = False
    quota_assignment: str = "unassigned"
    time_preserving_likely: bool = False
    dominance_likely: bool = False
    certificate_critical_candidate: bool = False
    hidden_challenger_risk: float = 0.0
    safe_prune_consistent: bool = True
    unresolved_possible_frontier_mass_contribution: float = 0.0
    unresolved_possible_winner_mass_contribution: float = 0.0
    unresolved_certificate_critical_mass_contribution: float = 0.0
    expected_proxy_value: float = 0.0
    expected_audit_value: float = 0.0
    preference_query_sensitivity: float = 0.0
    changes_possible_best_probability: float = 0.0
    changes_necessary_best_probability: float = 0.0
    search_completeness_contribution: float = 0.0
    quota_preserved: bool = False
    long_corridor_shortcut: bool = False
    long_corridor_support_status: str | None = None
    long_corridor_support_gap: float = 0.0
    long_corridor_search_completeness_penalty: float = 0.0
    long_corridor_abstention_risk: float = 0.0
    long_corridor_terminal_safety_risk: float = 0.0
    selection_rank: int | None = None
    observed_refine_cost: float | None = None
    observed_cost_delta: float | None = None
    refine_cost_error: float | None = None
    refine_cost_ratio: float | None = None
    near_duplicate: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DCCSResult:
    mode: str
    search_budget: int
    transition_reason: str
    selected: list[DCCSCandidateRecord]
    skipped: list[DCCSCandidateRecord]
    candidate_ledger: list[DCCSCandidateRecord]
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "search_budget": self.search_budget,
            "transition_reason": self.transition_reason,
            "selected": [item.as_dict() for item in self.selected],
            "skipped": [item.as_dict() for item in self.skipped],
            "candidate_ledger": [item.as_dict() for item in self.candidate_ledger],
            "summary": dict(self.summary),
        }


def stable_candidate_id(candidate: Mapping[str, Any]) -> str:
    explicit = str(candidate.get("candidate_id", "")).strip()
    if explicit:
        return explicit
    return _candidate_signature(candidate)


def build_candidate_record(
    candidate: Mapping[str, Any],
    *,
    frontier: Sequence[Mapping[str, Any]] = (),
    refined: Sequence[Mapping[str, Any]] = (),
    candidate_pool: Sequence[Mapping[str, Any]] = (),
    config: DCCSConfig | None = None,
) -> DCCSCandidateRecord:
    cfg = config or DCCSConfig()
    candidate_id = stable_candidate_id(candidate)
    path = _normalise_path(candidate.get("graph_path", candidate.get("path", candidate.get("node_ids"))))
    objective = _objective_vector(candidate)
    road_mix = _road_mix(candidate)
    mechanism = _mechanism_descriptor(candidate)
    confidence = _confidence_map(candidate)
    frontier_items = [
        item
        for item in (frontier or candidate_pool)
        if stable_candidate_id(item) != candidate_id
    ]
    refined_items = [
        item
        for item in (refined or frontier or candidate_pool)
        if stable_candidate_id(item) != candidate_id
    ]
    frontier_pool = [_objective_vector(item) for item in frontier_items]
    refined_pool = [_mechanism_descriptor(item) for item in refined_items]
    peer_paths = _peer_paths(refined_items)
    overlap = _jaccard_overlap(path, peer_paths)
    stretch = _stretch_ratio(candidate)
    detour = max(0.0, stretch - 1.0)
    objective_reference_pool = frontier_pool or [_objective_vector(item) for item in refined_items]
    objective_gap = _improvement_cone_gap(objective, objective_reference_pool) if objective_reference_pool else 1.0
    mechanism_gap = 0.0 if (refined_items and not refined_pool) else _mechanism_distance(mechanism, refined_pool)
    time_regret_gap = _time_regret_gap(objective, objective_reference_pool)
    time_preservation_bonus = _time_preservation_bonus(time_regret_gap)
    candidate_envelope = build_candidate_envelope(candidate, frontier=frontier_items)
    predicted_cost = _predicted_refine_cost(candidate, config=cfg)
    flip_probability = _flip_probability(
        candidate,
        objective_gap=objective_gap,
        mechanism_gap=mechanism_gap,
        overlap=overlap,
        stretch=stretch,
        config=cfg,
    )
    near_duplicate = overlap >= cfg.near_duplicate_threshold
    candidate_source_engine = str(candidate.get("candidate_source_engine") or "").strip() or None
    candidate_source_stage = str(candidate.get("candidate_source_stage") or "").strip() or None
    comparator_seeded = candidate_source_stage == "preemptive_comparator_seed"
    time_bonus_scale = _time_bonus_scale(
        objective_gap=objective_gap,
        mechanism_gap=mechanism_gap,
        flip_probability=flip_probability,
    )
    candidate_criticality = build_candidate_criticality(
        candidate,
        objective_gap=objective_gap,
        mechanism_gap=mechanism_gap,
        overlap=overlap,
        stretch=stretch,
        time_regret_gap=time_regret_gap,
        predicted_refine_cost=predicted_cost,
        flip_probability=flip_probability,
        candidate_envelope=candidate_envelope,
        near_duplicate=near_duplicate,
    )
    time_preserving_likely = time_preservation_bonus >= 0.25
    dominance_likely = bool(
        (candidate_envelope is not None and candidate_envelope.known_dominance == "dominates_frontier")
        or objective_gap >= 0.15
    )
    hidden_challenger_risk = (
        candidate_criticality.search_deficiency_risk if candidate_criticality is not None else 0.0
    )
    support_mass = candidate_envelope.support_mass if candidate_envelope is not None else 0.0
    certificate_critical_candidate = bool(
        (
            candidate_criticality is not None
            and candidate_criticality.criticality_score >= 0.9
        )
        or (objective_gap > 0.0 and flip_probability >= 0.5)
    )
    long_corridor_metrics = _long_corridor_shortcut_metrics(
        candidate_source_stage=candidate_source_stage,
        candidate_envelope=candidate_envelope,
        overlap=overlap,
        hidden_challenger_risk=hidden_challenger_risk,
        flip_probability=flip_probability,
        certificate_critical_candidate=certificate_critical_candidate,
    )
    unresolved_possible_winner_mass_contribution = _clamp_unit(
        candidate_criticality.unresolved_winner_mass if candidate_criticality is not None else (1.0 - objective_gap)
    )
    unresolved_possible_frontier_mass_contribution = _clamp_unit(
        (0.55 * unresolved_possible_winner_mass_contribution)
        + (0.25 * max(0.0, 1.0 - overlap))
        + (0.20 * max(0.0, 1.0 - support_mass))
    )
    unresolved_certificate_critical_mass_contribution = _clamp_unit(
        (0.45 * unresolved_possible_winner_mass_contribution)
        + (0.25 * _clamp_unit(flip_probability))
        + (0.20 * hidden_challenger_risk)
        + (0.10 * float(certificate_critical_candidate))
    )
    unresolved_possible_frontier_mass_contribution = _clamp_unit(
        unresolved_possible_frontier_mass_contribution
        + (0.40 * long_corridor_metrics["long_corridor_search_completeness_penalty"])
    )
    unresolved_possible_winner_mass_contribution = _clamp_unit(
        unresolved_possible_winner_mass_contribution
        + (0.20 * long_corridor_metrics["long_corridor_search_completeness_penalty"])
    )
    unresolved_certificate_critical_mass_contribution = _clamp_unit(
        unresolved_certificate_critical_mass_contribution
        + (0.25 * long_corridor_metrics["long_corridor_terminal_safety_risk"])
        + (0.10 * long_corridor_metrics["long_corridor_abstention_risk"])
    )
    expected_proxy_value = _clamp_unit(
        (
            candidate_criticality.winner_lcb_lift
            + candidate_criticality.pairwise_gap_lcb_lift
        )
        / 2.0
        if candidate_criticality is not None
        else 0.0
    )
    expected_audit_value = _clamp_unit(
        (
            (0.50 * candidate_criticality.flip_radius_lift)
            + (0.30 * hidden_challenger_risk)
            + (0.20 * max(0.0, 1.0 - support_mass))
        )
        if candidate_criticality is not None
        else hidden_challenger_risk
    )
    preference_query_sensitivity = _clamp_unit(
        candidate_criticality.preference_relevance if candidate_criticality is not None else 0.0
    )
    changes_possible_best_probability = _clamp_unit(
        (0.60 * unresolved_possible_winner_mass_contribution)
        + (0.40 * hidden_challenger_risk)
    )
    changes_necessary_best_probability = _clamp_unit(
        (0.45 * unresolved_certificate_critical_mass_contribution)
        + (0.35 * _clamp_unit(flip_probability))
        + (0.20 * preference_query_sensitivity)
    )
    search_completeness_contribution = _clamp_unit(
        (0.50 * unresolved_possible_frontier_mass_contribution)
        + (0.50 * hidden_challenger_risk)
    )
    search_completeness_contribution = _clamp_unit(
        search_completeness_contribution
        + (0.50 * long_corridor_metrics["long_corridor_search_completeness_penalty"])
    )
    criticality_terms = (
        candidate_criticality.ranking_terms()
        if candidate_criticality is not None
        else {
            "winner_lcb_lift": 0.0,
            "pairwise_gap_lcb_lift": 0.0,
            "flip_radius_lift": 0.0,
            "unresolved_winner_mass": 0.0,
            "preference_relevance": 0.0,
            "search_deficiency_risk": 0.0,
            "candidate_action_cost": predicted_cost,
            "criticality_score": 0.0,
        }
    )
    score_terms = {
        "objective_gap": objective_gap,
        "mechanism_gap": mechanism_gap,
        "overlap_penalty": overlap,
        "stretch_penalty": detour,
        "time_regret_gap": time_regret_gap,
        "time_preservation_bonus": time_preservation_bonus,
        "time_bonus_scale": time_bonus_scale,
        "flip_probability": flip_probability,
        "predicted_refine_cost": predicted_cost,
        "objective_extremeness": _extremeness_score(
            objective,
            [_objective_vector(item) for item in candidate_pool if stable_candidate_id(item) != candidate_id],
        ),
        "comparator_seed_penalty": float(cfg.comparator_seed_penalty_weight if comparator_seeded else 0.0),
        **criticality_terms,
        "expected_proxy_value": expected_proxy_value,
        "expected_audit_value": expected_audit_value,
        "preference_query_sensitivity": preference_query_sensitivity,
        "changes_possible_best_probability": changes_possible_best_probability,
        "changes_necessary_best_probability": changes_necessary_best_probability,
        "search_completeness_contribution": search_completeness_contribution,
    }
    if comparator_seeded:
        quota_assignment = "disagreement_driven_challenger"
    elif time_preserving_likely:
        quota_assignment = "time_preserving_challenger"
    elif dominance_likely:
        quota_assignment = "dominance_likely_challenger"
    elif objective_gap > 0.0 and not near_duplicate:
        quota_assignment = "high_significance_corridor_family"
    elif stretch <= 1.15 and overlap <= 0.50:
        quota_assignment = "representative_capital_rescue"
    else:
        quota_assignment = "unassigned"
    safe_prune_consistent = (not bool(candidate_envelope.safe_eliminated)) or bool(
        candidate_envelope.necessary_dominated
    )
    return DCCSCandidateRecord(
        candidate_id=candidate_id,
        graph_path=path,
        graph_length_km=max(0.0, _as_float(candidate.get("graph_length_km", candidate.get("distance_km")))),
        road_class_mix=road_mix,
        toll_share=max(0.0, _as_float(candidate.get("toll_share"))),
        terrain_burden=max(0.0, _as_float(candidate.get("terrain_burden"))),
        proxy_objective=objective,
        mechanism_descriptor=mechanism,
        proxy_confidence=confidence,
        overlap=overlap,
        stretch=stretch,
        detour=detour,
        objective_gap=objective_gap,
        mechanism_gap=mechanism_gap,
        time_regret_gap=time_regret_gap,
        time_preservation_bonus=time_preservation_bonus,
        predicted_refine_cost=predicted_cost,
        flip_probability=flip_probability,
        score_terms=score_terms,
        final_score=0.0,
        decision="skip",
        decision_reason="pending",
        mode=cfg.mode,
        corridor_signature=_corridor_signature(path),
        candidate_envelope=candidate_envelope,
        candidate_criticality=candidate_criticality,
        safe_eliminated=bool(candidate_envelope.safe_eliminated),
        necessary_dominated=bool(candidate_envelope.necessary_dominated),
        dominated_by_route_id=candidate_envelope.dominated_by_route_id,
        dominance_margin=candidate_envelope.dominance_margin,
        safe_elimination_reason=candidate_envelope.safe_elimination_reason,
        candidate_source_engine=candidate_source_engine,
        candidate_source_stage=candidate_source_stage,
        comparator_seeded=comparator_seeded,
        quota_assignment=quota_assignment,
        time_preserving_likely=time_preserving_likely,
        dominance_likely=dominance_likely,
        certificate_critical_candidate=certificate_critical_candidate,
        hidden_challenger_risk=hidden_challenger_risk,
        safe_prune_consistent=safe_prune_consistent,
        unresolved_possible_frontier_mass_contribution=unresolved_possible_frontier_mass_contribution,
        unresolved_possible_winner_mass_contribution=unresolved_possible_winner_mass_contribution,
        unresolved_certificate_critical_mass_contribution=unresolved_certificate_critical_mass_contribution,
        expected_proxy_value=expected_proxy_value,
        expected_audit_value=expected_audit_value,
        preference_query_sensitivity=preference_query_sensitivity,
        changes_possible_best_probability=changes_possible_best_probability,
        changes_necessary_best_probability=changes_necessary_best_probability,
        search_completeness_contribution=search_completeness_contribution,
        long_corridor_shortcut=bool(long_corridor_metrics["long_corridor_shortcut"]),
        long_corridor_support_status=long_corridor_metrics["long_corridor_support_status"],
        long_corridor_support_gap=long_corridor_metrics["long_corridor_support_gap"],
        long_corridor_search_completeness_penalty=long_corridor_metrics["long_corridor_search_completeness_penalty"],
        long_corridor_abstention_risk=long_corridor_metrics["long_corridor_abstention_risk"],
        long_corridor_terminal_safety_risk=long_corridor_metrics["long_corridor_terminal_safety_risk"],
        near_duplicate=near_duplicate,
    )


def _bootstrap_score(
    record: DCCSCandidateRecord,
    *,
    selected: Sequence[DCCSCandidateRecord],
    candidate_pool: Sequence[DCCSCandidateRecord],
    config: DCCSConfig,
) -> float:
    # Objective-space novelty follows a max-min style dispersion heuristic
    # similar in spirit to crowding-based diversity preservation in NSGA-II:
    # Deb et al. (2002), https://doi.org/10.1109/4235.996017
    selected_objectives = [item.proxy_objective for item in selected]
    selected_mechanisms = [item.mechanism_descriptor for item in selected]
    pool_objectives = [item.proxy_objective for item in candidate_pool]
    coverage = (
        record.objective_gap
        if not selected
        else _normalised_distance(
            record.proxy_objective,
            selected_objectives,
            reference_pool=pool_objectives,
        )
    )
    extremeness = _extremeness_score(record.proxy_objective, pool_objectives)
    diversity = record.mechanism_gap if not selected else _mechanism_distance(record.mechanism_descriptor, selected_mechanisms)
    plausibility = 1.0 / max(1.0, record.stretch)
    objective_support = record.objective_gap
    time_preservation = record.time_preservation_bonus
    overlap_penalty = _overlap_to_selected(record, selected=selected)
    corridor_reuse_count = sum(1 for item in selected if item.corridor_signature == record.corridor_signature)
    corridor_diversity = 1.0 / float(1 + corridor_reuse_count)
    benefit = (
        (config.bootstrap_coverage_weight * coverage)
        + (config.bootstrap_extremeness_weight * extremeness)
        + (config.bootstrap_diversity_weight * diversity)
        + (config.bootstrap_corridor_diversity_weight * corridor_diversity)
        + (config.bootstrap_plausibility_weight * plausibility)
        + (config.bootstrap_objective_support_weight * objective_support)
        + (config.bootstrap_time_preservation_weight * time_preservation)
        + (config.bootstrap_overlap_weight * max(0.0, 1.0 - overlap_penalty))
    )
    cost = (
        1.0
        + (config.cost_weight * record.predicted_refine_cost)
        + (config.bootstrap_overlap_decay_weight * overlap_penalty)
        + (config.bootstrap_time_regret_penalty_weight * record.time_regret_gap)
    )
    if corridor_reuse_count > 0:
        cost += config.bootstrap_corridor_penalty_weight * corridor_reuse_count
    if record.comparator_seeded:
        cost += config.comparator_seed_penalty_weight
    return benefit / max(1e-9, cost)


def _challenger_score(record: DCCSCandidateRecord, *, config: DCCSConfig) -> float:
    time_bonus_scale = _time_bonus_scale(
        objective_gap=record.objective_gap,
        mechanism_gap=record.mechanism_gap,
        flip_probability=record.flip_probability,
    )
    # Budget pressure should favour challengers that are both support-bearing
    # and time-plausible; mechanism-only detours otherwise consume search
    # budget ahead of productive via candidates on collapse-prone rows.
    support_gate = min(
        1.0,
        max(0.0, record.objective_gap + (0.45 * record.time_preservation_bonus)),
    )
    criticality_score = _clamp_unit(
        record.candidate_criticality.criticality_score
        if record.candidate_criticality is not None
        else 0.0
    )
    gain = (
        (config.objective_gap_weight * record.objective_gap)
        + (config.mechanism_gap_weight * record.mechanism_gap * support_gate)
        + (config.challenger_gain_weight * record.flip_probability * support_gate)
        + (0.25 * (1.0 - record.overlap))
        + (config.challenger_time_preservation_weight * record.time_preservation_bonus * time_bonus_scale)
        + (config.challenger_candidate_criticality_weight * criticality_score)
        + (config.challenger_expected_proxy_value_weight * record.expected_proxy_value)
        + (config.challenger_expected_audit_value_weight * record.expected_audit_value)
        + (config.challenger_preference_sensitivity_weight * record.preference_query_sensitivity)
        + (config.challenger_possible_best_change_weight * record.changes_possible_best_probability)
        + (config.challenger_necessary_best_change_weight * record.changes_necessary_best_probability)
        + (config.challenger_search_completeness_weight * record.search_completeness_contribution)
        + (config.challenger_hidden_challenger_weight * record.hidden_challenger_risk)
    )
    if record.long_corridor_shortcut:
        gain += (
            (0.20 * record.long_corridor_search_completeness_penalty)
            + (0.20 * record.long_corridor_abstention_risk)
            + (0.25 * record.long_corridor_terminal_safety_risk)
            + (0.10 * record.long_corridor_support_gap)
        )
    penalty = (
        (config.overlap_penalty_weight * record.overlap)
        + (config.stretch_penalty_weight * max(0.0, record.stretch - 1.0))
        + (config.cost_weight * record.predicted_refine_cost)
    )
    if record.comparator_seeded:
        penalty += config.comparator_seed_penalty_weight
    return gain / max(1e-9, penalty)


def score_candidate(record: DCCSCandidateRecord, *, config: DCCSConfig | None = None) -> float:
    cfg = config or DCCSConfig()
    if cfg.mode == "bootstrap":
        return _bootstrap_score(record, selected=(), candidate_pool=[record], config=cfg)
    return _challenger_score(record, config=cfg)


def record_refine_outcome(
    record: DCCSCandidateRecord,
    *,
    observed_refine_cost: float | None,
    frontier_added: bool = False,
    decision_flip: bool = False,
    dominated_but_close: bool = False,
    redundant: bool = False,
) -> DCCSCandidateRecord:
    if frontier_added:
        label = "frontier_addition"
    elif decision_flip:
        label = "decision_flip"
    elif dominated_but_close:
        label = "challenger_but_not_added"
    elif redundant:
        label = "non_challenger_redundant"
    else:
        label = record.decision_reason
    if observed_refine_cost is None:
        return replace(
            record,
            observed_refine_cost=None,
            observed_cost_delta=None,
            refine_cost_error=None,
            refine_cost_ratio=None,
            decision_reason=label,
        )
    delta = observed_refine_cost - record.predicted_refine_cost
    ratio = observed_refine_cost / max(1e-9, record.predicted_refine_cost)
    return replace(
        record,
        observed_refine_cost=float(observed_refine_cost),
        observed_cost_delta=float(delta),
        refine_cost_error=float(delta),
        refine_cost_ratio=float(ratio),
        decision_reason=label,
    )


def summarize_refine_outcomes(
    records: Sequence[DCCSCandidateRecord],
) -> dict[str, Any]:
    refined = [record for record in records if record.observed_refine_cost is not None]
    if not refined:
        return {
            "observed_metrics_available": False,
            "metric_stage": "pre_refinement_prediction",
            "observed_refinement_count": 0,
            "observed_dc_yield": None,
            "observed_challenger_hit_rate": None,
            "observed_frontier_gain_per_refinement": None,
            "observed_decision_flips": 0,
            "observed_frontier_additions": 0,
            "observed_redundant_count": 0,
            "mean_refine_cost_error": None,
            "mean_refine_cost_ratio": None,
            "refine_cost_mape": None,
            "refine_cost_mae_ms": None,
            "refine_cost_rank_correlation": None,
            "refine_cost_sample_count": 0,
        }
    frontier_additions = sum(1 for record in refined if record.decision_reason == "frontier_addition")
    decision_flips = sum(1 for record in refined if record.decision_reason == "decision_flip")
    challenger_hits = sum(
        1
        for record in refined
        if record.decision_reason in {"frontier_addition", "decision_flip", "challenger_but_not_added"}
    )
    redundant = sum(1 for record in refined if record.decision_reason == "non_challenger_redundant")
    cost_errors = [record.refine_cost_error for record in refined if record.refine_cost_error is not None]
    cost_ratios = [record.refine_cost_ratio for record in refined if record.refine_cost_ratio is not None]
    absolute_errors = [
        abs(_as_float(record.observed_refine_cost) - _as_float(record.predicted_refine_cost))
        for record in refined
    ]
    mape_values = [
        abs(_as_float(record.observed_refine_cost) - _as_float(record.predicted_refine_cost))
        / max(1e-9, _as_float(record.observed_refine_cost))
        for record in refined
        if _as_float(record.observed_refine_cost) > 0.0
    ]
    refined_count = float(len(refined))
    predicted_costs = [max(0.0, _as_float(record.predicted_refine_cost)) for record in refined]
    observed_costs = [max(0.0, _as_float(record.observed_refine_cost)) for record in refined]
    return {
        "observed_metrics_available": True,
        "metric_stage": "post_refinement_observed",
        "observed_refinement_count": len(refined),
        "observed_dc_yield": (decision_flips + frontier_additions) / refined_count,
        "observed_challenger_hit_rate": challenger_hits / refined_count,
        "observed_frontier_gain_per_refinement": frontier_additions / refined_count,
        "observed_decision_flips": decision_flips,
        "observed_frontier_additions": frontier_additions,
        "observed_redundant_count": redundant,
        "mean_refine_cost_error": (sum(cost_errors) / float(len(cost_errors))) if cost_errors else None,
        "mean_refine_cost_ratio": (sum(cost_ratios) / float(len(cost_ratios))) if cost_ratios else None,
        "refine_cost_mape": (sum(mape_values) / float(len(mape_values))) if mape_values else None,
        "refine_cost_mae_ms": (sum(absolute_errors) / float(len(absolute_errors))) if absolute_errors else None,
        "refine_cost_rank_correlation": _rank_correlation(predicted_costs, observed_costs),
        "refine_cost_sample_count": len(refined),
    }


def _dccs_gate_metrics(
    records: Sequence[DCCSCandidateRecord],
    *,
    selected: Sequence[DCCSCandidateRecord],
) -> dict[str, Any]:
    candidate_count = len(records)
    selected_ids = {record.candidate_id for record in selected}
    safe_pruned = [record for record in records if record.safe_eliminated]
    live_records = [record for record in records if not record.safe_eliminated]
    selected_live = [
        record for record in selected if not record.safe_eliminated
    ]
    unresolved_records = [record for record in live_records if record.candidate_id not in selected_ids]
    false_safe_prune_count = sum(1 for record in safe_pruned if not record.safe_prune_consistent)
    available_quota_groups = {
        record.quota_assignment
        for record in records
        if record.quota_assignment != "unassigned"
    }
    selected_quota_groups = {
        record.quota_assignment
        for record in selected
        if record.quota_assignment != "unassigned"
    }
    certificate_critical_candidates = [
        record for record in live_records if record.certificate_critical_candidate
    ]
    time_preserving_candidates = [
        record for record in live_records if record.time_preserving_likely
    ]
    dominance_likely_candidates = [
        record for record in live_records if record.dominance_likely
    ]
    hidden_challenger_candidates = [
        record for record in live_records if record.hidden_challenger_risk >= 0.5
    ]
    hidden_challenger_selected = [
        record for record in selected_live if record.hidden_challenger_risk >= 0.5
    ]
    long_corridor_records = [
        record for record in records if record.long_corridor_shortcut
    ]
    long_corridor_support_status_counts: dict[str, int] = {}
    for record in long_corridor_records:
        token = str(record.long_corridor_support_status or "unknown").strip() or "unknown"
        long_corridor_support_status_counts[token] = long_corridor_support_status_counts.get(token, 0) + 1
    quota_preservation = {
        "high_significance_corridor_families": {},
        "time_preserving_challenger": True,
        "dominance_likely_challenger": True,
        "disagreement_driven_challenger": True,
        "representative_capital_rescue": True,
    }
    high_significance_families = sorted(
        {
            record.corridor_signature
            for record in records
            if record.quota_assignment == _ANTI_COLLAPSE_FAMILY_QUOTA
        }
    )
    selected_ids = {record.candidate_id for record in selected}
    for family in high_significance_families:
        quota_preservation["high_significance_corridor_families"][family] = any(
            record.candidate_id in selected_ids
            for record in selected
            if record.quota_assignment == _ANTI_COLLAPSE_FAMILY_QUOTA
            and record.corridor_signature == family
        )
    if any(record.time_preserving_likely for record in records):
        quota_preservation["time_preserving_challenger"] = any(
            record.time_preserving_likely for record in selected
        )
    if any(record.dominance_likely for record in records):
        quota_preservation["dominance_likely_challenger"] = any(
            record.dominance_likely for record in selected
        )
    if any(record.comparator_seeded for record in records):
        quota_preservation["disagreement_driven_challenger"] = any(
            record.comparator_seeded for record in selected
        )
    if any(record.quota_assignment == _ANTI_COLLAPSE_RESCUE_QUOTA for record in records):
        quota_preservation["representative_capital_rescue"] = any(
            record.quota_assignment == _ANTI_COLLAPSE_RESCUE_QUOTA for record in selected
        )
    live_denominator = float(len(live_records) or 1)
    unresolved_possible_frontier_mass = sum(
        record.unresolved_possible_frontier_mass_contribution for record in unresolved_records
    ) / live_denominator
    unresolved_possible_winner_mass = sum(
        record.unresolved_possible_winner_mass_contribution for record in unresolved_records
    ) / live_denominator
    unresolved_certificate_critical_mass = sum(
        record.unresolved_certificate_critical_mass_contribution for record in unresolved_records
    ) / live_denominator
    search_completeness_gap = _clamp_unit(
        (
            unresolved_possible_frontier_mass
            + unresolved_possible_winner_mass
            + unresolved_certificate_critical_mass
        )
        / 3.0
    )
    return {
        "safe_prune_rate": len(safe_pruned) / float(candidate_count or 1),
        "false_safe_prune_rate": (
            false_safe_prune_count / float(len(safe_pruned))
            if safe_pruned
            else 0.0
        ),
        "anti_collapse_success_rate": (
            len(selected_quota_groups) / float(len(available_quota_groups))
            if available_quota_groups
            else 1.0
        ),
        "certificate_critical_hit_rate": (
            sum(1 for record in selected_live if record.certificate_critical_candidate)
            / float(len(certificate_critical_candidates))
            if certificate_critical_candidates
            else 1.0
        ),
        "time_preserving_challenger_coverage": (
            sum(1 for record in selected_live if record.time_preserving_likely)
            / float(len(time_preserving_candidates))
            if time_preserving_candidates
            else 1.0
        ),
        "dominance_likely_challenger_coverage": (
            sum(1 for record in selected_live if record.dominance_likely)
            / float(len(dominance_likely_candidates))
            if dominance_likely_candidates
            else 1.0
        ),
        "hidden_challenger_miss_diagnostics": {
            "candidate_count": len(hidden_challenger_candidates),
            "selected_count": len(hidden_challenger_selected),
            "miss_count": max(0, len(hidden_challenger_candidates) - len(hidden_challenger_selected)),
            "miss_rate": (
                max(0, len(hidden_challenger_candidates) - len(hidden_challenger_selected))
                / float(len(hidden_challenger_candidates))
                if hidden_challenger_candidates
                else 0.0
            ),
        },
        "unresolved_possible_frontier_mass": unresolved_possible_frontier_mass,
        "unresolved_possible_winner_mass": unresolved_possible_winner_mass,
        "unresolved_certificate_critical_mass": unresolved_certificate_critical_mass,
        "search_completeness_score": 1.0 - search_completeness_gap,
        "search_completeness_gap": search_completeness_gap,
        "anti_collapse_quota_preservation": quota_preservation,
        "quota_preserved_candidate_count": sum(1 for record in selected if record.quota_preserved),
        "long_corridor_shortcut_count": len(long_corridor_records),
        "long_corridor_selected_count": sum(1 for record in selected if record.long_corridor_shortcut),
        "long_corridor_support_status_counts": long_corridor_support_status_counts,
        "long_corridor_search_completeness_penalty_mean": (
            sum(record.long_corridor_search_completeness_penalty for record in long_corridor_records)
            / float(len(long_corridor_records))
            if long_corridor_records
            else 0.0
        ),
        "long_corridor_abstention_risk_mean": (
            sum(record.long_corridor_abstention_risk for record in long_corridor_records)
            / float(len(long_corridor_records))
            if long_corridor_records
            else 0.0
        ),
        "long_corridor_terminal_safety_risk_mean": (
            sum(record.long_corridor_terminal_safety_risk for record in long_corridor_records)
            / float(len(long_corridor_records))
            if long_corridor_records
            else 0.0
        ),
    }


def build_dccs_summary_breadcrumbs(records: Sequence[DCCSCandidateRecord]) -> dict[str, Any]:
    ranking_trace_present = all(
        all(key in record.score_terms for key in _CRITICALITY_RANK_TERM_KEYS)
        for record in records
    ) if records else True
    search_deficiency_trace_present = all(
        0.0 <= record.unresolved_possible_frontier_mass_contribution <= 1.0
        and 0.0 <= record.unresolved_possible_winner_mass_contribution <= 1.0
        and 0.0 <= record.unresolved_certificate_critical_mass_contribution <= 1.0
        and 0.0 <= record.search_completeness_contribution <= 1.0
        for record in records
    ) if records else True
    forecast_trace_present = all(
        0.0 <= record.expected_proxy_value <= 1.0
        and 0.0 <= record.expected_audit_value <= 1.0
        and 0.0 <= record.preference_query_sensitivity <= 1.0
        and 0.0 <= record.changes_possible_best_probability <= 1.0
        and 0.0 <= record.changes_necessary_best_probability <= 1.0
        for record in records
    ) if records else True
    denominator = float(len(records) or 1)
    return {
        "candidate_envelope_schema_version": "candidate_envelope_v1",
        "candidate_criticality_schema_version": "candidate_criticality_v1",
        "candidate_envelope_count": len(records),
        "candidate_criticality_count": len(records),
        "candidate_criticality_ranking_trace_present": ranking_trace_present,
        "search_deficiency_trace_present": search_deficiency_trace_present,
        "forecast_trace_present": forecast_trace_present,
        "criticality_rank_term_keys": list(_CRITICALITY_RANK_TERM_KEYS),
        "mean_candidate_criticality_score": (
            sum(
                _clamp_unit(
                    record.candidate_criticality.criticality_score
                    if record.candidate_criticality is not None
                    else 0.0
                )
                for record in records
            ) / denominator
        ),
        "mean_expected_proxy_value": sum(record.expected_proxy_value for record in records) / denominator,
        "mean_expected_audit_value": sum(record.expected_audit_value for record in records) / denominator,
        "mean_preference_query_sensitivity": (
            sum(record.preference_query_sensitivity for record in records) / denominator
        ),
        "mean_search_completeness_contribution": (
            sum(record.search_completeness_contribution for record in records) / denominator
        ),
        "mean_hidden_challenger_risk": sum(record.hidden_challenger_risk for record in records) / denominator,
        "safe_elimination_provenance_present": bool(records),
        "safe_eliminated_count": sum(1 for record in records if record.safe_eliminated),
        "necessary_dominated_count": sum(1 for record in records if record.necessary_dominated),
        "safe_prune_consistent_count": sum(1 for record in records if record.safe_prune_consistent),
        "dominated_by_route_id_count": sum(
            1 for record in records if str(record.dominated_by_route_id or "").strip()
        ),
    }


def build_candidate_ledger(
    candidates: Sequence[Mapping[str, Any]],
    *,
    frontier: Sequence[Mapping[str, Any]] = (),
    refined: Sequence[Mapping[str, Any]] = (),
    config: DCCSConfig | None = None,
) -> list[DCCSCandidateRecord]:
    cfg = config or DCCSConfig()
    return [
        build_candidate_record(
            candidate,
            frontier=frontier,
            refined=refined,
            candidate_pool=candidates,
            config=cfg,
        )
        for candidate in candidates
    ]


def _resolved_candidate_ledger(
    ledger: Sequence[DCCSCandidateRecord],
    *,
    selected: Sequence[DCCSCandidateRecord],
    skipped: Sequence[DCCSCandidateRecord],
) -> list[DCCSCandidateRecord]:
    resolved: dict[str, DCCSCandidateRecord] = {
        record.candidate_id: record
        for record in [*selected, *skipped]
    }
    return [resolved.get(record.candidate_id, record) for record in ledger]


def _baseline_policy_key(policy: str) -> str:
    key = str(policy or "first_n").strip().lower()
    if key == "corridor_uniform":
        return "uniform_corridor_n"
    if key not in BASELINE_SELECTION_POLICIES:
        raise ValueError(f"unsupported baseline policy: {policy}")
    return key


def select_baseline_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    policy: str,
    seed: int = 0,
) -> list[str]:
    policy_key = _baseline_policy_key(policy)
    ordered_ids = [stable_candidate_id(candidate) for candidate in candidates]
    if budget <= 0 or not ordered_ids:
        return []
    if policy_key == "first_n":
        return ordered_ids[:budget]
    if policy_key == "random_n":
        keyed = sorted(
            ordered_ids,
            key=lambda candidate_id: (
                _stable_hash([str(seed), candidate_id]),
                candidate_id,
            ),
        )
        return keyed[:budget]
    if policy_key == "uniform_corridor_n":
        corridor_to_ids: dict[str, list[str]] = {}
        for candidate in candidates:
            candidate_id = stable_candidate_id(candidate)
            path = _normalise_path(candidate.get("graph_path", candidate.get("path", candidate.get("node_ids"))))
            corridor_to_ids.setdefault(_corridor_signature(path), []).append(candidate_id)
        selected: list[str] = []
        corridor_keys = sorted(corridor_to_ids)
        while corridor_keys and len(selected) < budget:
            next_keys: list[str] = []
            for corridor in corridor_keys:
                ids = corridor_to_ids[corridor]
                if ids:
                    selected.append(ids.pop(0))
                if ids:
                    next_keys.append(corridor)
                if len(selected) >= budget:
                    break
            corridor_keys = next_keys
        return selected
    raise ValueError(f"unsupported baseline policy: {policy}")


def select_baseline_result(
    candidates: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    policy: str,
    seed: int = 0,
    frontier: Sequence[Mapping[str, Any]] = (),
    refined: Sequence[Mapping[str, Any]] = (),
    config: DCCSConfig | None = None,
) -> DCCSResult:
    cfg = config or DCCSConfig(mode="challenger", search_budget=budget)
    policy_key = _baseline_policy_key(policy)
    ledger = build_candidate_ledger(candidates, frontier=frontier, refined=refined, config=cfg)
    ordered_ids = select_baseline_candidates(candidates, budget=len(candidates), policy=policy_key, seed=seed)
    ordered_map = {record.candidate_id: record for record in ledger}
    ordered = [ordered_map[candidate_id] for candidate_id in ordered_ids if candidate_id in ordered_map]
    budget = max(0, int(budget))
    selected: list[DCCSCandidateRecord] = []
    skipped: list[DCCSCandidateRecord] = []
    for rank, record in enumerate(ordered):
        score = score_candidate(record, config=cfg)
        if rank < budget:
            selected.append(
                replace(
                    record,
                    final_score=float(score),
                    decision="refine",
                    decision_reason=f"selected_by_baseline_policy:{policy_key}",
                    selection_rank=rank,
                    mode=f"{cfg.mode}:{policy_key}",
                )
            )
        else:
            skipped.append(
                replace(
                    record,
                    final_score=float(score),
                    decision="skip",
                    decision_reason="budget_exhausted",
                    selection_rank=rank,
                    mode=f"{cfg.mode}:{policy_key}",
                )
            )
    selected_ids = {record.candidate_id for record in selected}
    for record in ledger:
        if record.candidate_id in selected_ids or any(item.candidate_id == record.candidate_id for item in skipped):
            continue
        skipped.append(
            replace(
                record,
                final_score=float(score_candidate(record, config=cfg)),
                decision="skip",
                decision_reason="not_selected",
                mode=f"{cfg.mode}:{policy_key}",
            )
        )
    summary = {
        "mode": f"{cfg.mode}:{policy_key}",
        "transition_reason": f"baseline_policy:{policy_key}",
        "selection_policy": policy_key,
        "search_budget": budget,
        "candidate_count": len(ordered),
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "selected_corridor_count": len({item.corridor_signature for item in selected}),
    }
    summary.update(_dccs_gate_metrics(ordered, selected=selected))
    return DCCSResult(
        mode=f"{cfg.mode}:{policy_key}",
        search_budget=budget,
        transition_reason=f"baseline_policy:{policy_key}",
        selected=selected,
        skipped=skipped,
        candidate_ledger=_resolved_candidate_ledger(ordered, selected=selected, skipped=skipped),
        summary=summary,
    )


def _reserve_anti_collapse_records(
    ledger: Sequence[DCCSCandidateRecord],
    *,
    config: DCCSConfig,
) -> list[DCCSCandidateRecord]:
    if config.mode == "bootstrap" or not ledger:
        return []
    ranked = sorted(
        ledger,
        key=lambda record: (
            -score_candidate(record, config=config),
            record.near_duplicate,
            record.candidate_id,
        ),
    )
    reserved: list[DCCSCandidateRecord] = []
    reserved_ids: set[str] = set()
    reserved_paths: set[tuple[str, ...]] = set()

    def _reserve_matching(predicate: Any) -> None:
        if any(predicate(record) for record in reserved):
            return
        for record in ranked:
            if not predicate(record):
                continue
            if record.candidate_id in reserved_ids or record.graph_path in reserved_paths:
                continue
            reserved.append(record)
            reserved_ids.add(record.candidate_id)
            reserved_paths.add(record.graph_path)
            return

    def _reserve_additional_matching(predicate: Any) -> None:
        for record in ranked:
            if not predicate(record):
                continue
            if record.candidate_id in reserved_ids or record.graph_path in reserved_paths:
                continue
            reserved.append(record)
            reserved_ids.add(record.candidate_id)
            reserved_paths.add(record.graph_path)
            return

    def _is_live_certificate_guard(record: DCCSCandidateRecord) -> bool:
        return (not record.safe_eliminated) and record.certificate_critical_candidate

    def _is_live_hidden_search_guard(record: DCCSCandidateRecord) -> bool:
        return (
            (not record.safe_eliminated)
            and record.hidden_challenger_risk >= 0.5
            and record.search_completeness_contribution > 0.0
        )

    def _reserve_frontier_expansion() -> None:
        reserved_families = {
            record.corridor_signature
            for record in reserved
            if record.corridor_signature
        }
        for record in ranked:
            if record.candidate_id in reserved_ids or record.graph_path in reserved_paths:
                continue
            if record.safe_eliminated or record.near_duplicate:
                continue
            if record.objective_gap <= 0.0:
                continue
            if reserved_families and record.corridor_signature in reserved_families:
                continue
            reserved.append(record)
            reserved_ids.add(record.candidate_id)
            reserved_paths.add(record.graph_path)
            return

    preserved_high_significance_families: set[str] = set()
    for record in ranked:
        if record.quota_assignment != _ANTI_COLLAPSE_FAMILY_QUOTA:
            continue
        if record.corridor_signature in preserved_high_significance_families:
            continue
        if record.candidate_id in reserved_ids or record.graph_path in reserved_paths:
            continue
        reserved.append(record)
        reserved_ids.add(record.candidate_id)
        reserved_paths.add(record.graph_path)
        preserved_high_significance_families.add(record.corridor_signature)
    _reserve_matching(lambda record: record.comparator_seeded)
    _reserve_matching(lambda record: record.time_preserving_likely)
    _reserve_matching(lambda record: record.dominance_likely)
    _reserve_additional_matching(_is_live_certificate_guard)
    _reserve_additional_matching(_is_live_hidden_search_guard)
    _reserve_matching(lambda record: record.quota_assignment == _ANTI_COLLAPSE_RESCUE_QUOTA)
    _reserve_frontier_expansion()
    return reserved


def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    frontier: Sequence[Mapping[str, Any]] = (),
    refined: Sequence[Mapping[str, Any]] = (),
    config: DCCSConfig | None = None,
) -> DCCSResult:
    cfg = config or DCCSConfig()
    ledger = build_candidate_ledger(candidates, frontier=frontier, refined=refined, config=cfg)
    if cfg.mode == "bootstrap" and not frontier:
        transition_reason = "bootstrap_seeding:no_strict_frontier"
    elif cfg.mode == "bootstrap":
        transition_reason = "bootstrap_seeding:frontier_unstable"
    else:
        transition_reason = "challenger_mode:strict_frontier_available"

    selected: list[DCCSCandidateRecord] = []
    skipped: list[DCCSCandidateRecord] = []
    budget = max(0, int(cfg.search_budget))
    selected_ids: set[str] = set()

    if cfg.mode == "bootstrap":
        remaining = list(ledger)
        while remaining and len(selected) < budget:
            ranked = sorted(
                remaining,
                key=lambda record: (
                    -_bootstrap_score(record, selected=selected, candidate_pool=ledger, config=cfg),
                    record.near_duplicate,
                    record.candidate_id,
                ),
            )
            record = ranked[0]
            score = _bootstrap_score(record, selected=selected, candidate_pool=ledger, config=cfg)
            reason = "selected_by_bootstrap"
            if record.graph_path in {item.graph_path for item in selected}:
                reason = "duplicate_signature"
            elif any(item.corridor_signature == record.corridor_signature for item in selected[: max(1, cfg.bootstrap_seed_size - 1)]):
                reason = "duplicate_corridor_bootstrap"
            chosen = replace(
                record,
                final_score=float(score),
                decision="refine" if reason == "selected_by_bootstrap" else "skip",
                decision_reason=reason,
                selection_rank=len(selected) if reason == "selected_by_bootstrap" else None,
            )
            if chosen.decision == "refine":
                selected.append(chosen)
                selected_ids.add(chosen.candidate_id)
            else:
                skipped.append(chosen)
            remaining = [item for item in remaining if item.candidate_id != record.candidate_id]
    else:
        reserved_records = _reserve_anti_collapse_records(ledger, config=cfg)
        effective_budget = max(budget, len(reserved_records))
        for record in reserved_records:
            score = score_candidate(record, config=cfg)
            selected_record = replace(
                record,
                final_score=float(score),
                decision="refine",
                decision_reason="selected_by_anti_collapse_quota",
                selection_rank=len(selected),
                quota_preserved=True,
            )
            selected.append(selected_record)
            selected_ids.add(record.candidate_id)
        sorted_records = sorted(
            ledger,
            key=lambda record: (
                -score_candidate(record, config=cfg),
                record.near_duplicate,
                record.candidate_id,
            ),
        )
        for record in sorted_records:
            if record.candidate_id in selected_ids:
                continue
            if len(selected) >= effective_budget:
                skipped.append(
                    replace(
                        record,
                        final_score=score_candidate(record, config=cfg),
                        decision="skip",
                        decision_reason="budget_exhausted",
                    )
                )
                continue
            score = score_candidate(record, config=cfg)
            reason = "selected_by_challenger"
            if record.graph_path in {item.graph_path for item in selected}:
                reason = "duplicate_signature"
            selected_record = replace(
                record,
                final_score=float(score),
                decision="refine" if reason != "duplicate_signature" else "skip",
                decision_reason=reason,
                selection_rank=len(selected) if reason != "duplicate_signature" else None,
            )
            if reason == "duplicate_signature":
                skipped.append(selected_record)
                continue
            selected.append(selected_record)
            selected_ids.add(record.candidate_id)
        sorted_records = ledger

    skipped_ids = {item.candidate_id for item in skipped}
    for record in ledger:
        if record.candidate_id in selected_ids or record.candidate_id in skipped_ids:
            continue
        skipped.append(
            replace(
                record,
                final_score=float(
                    _bootstrap_score(record, selected=selected, candidate_pool=ledger, config=cfg)
                    if cfg.mode == "bootstrap"
                    else score_candidate(record, config=cfg)
                ),
                decision="skip",
                decision_reason="not_selected",
            )
        )

    if cfg.mode == "bootstrap":
        hit_count = sum(1 for item in selected if item.objective_gap > 0.0)
    else:
        hit_count = sum(1 for item in selected if item.flip_probability >= 0.5)
    frontier_additions = sum(1 for item in selected if item.objective_gap > 0.0)
    decision_flips = sum(1 for item in selected if item.flip_probability >= 0.5)
    dual_critical = sum(
        1 for item in selected if item.objective_gap > 0.0 and item.flip_probability >= 0.5
    )
    unique_critical = sum(
        1 for item in selected if item.objective_gap > 0.0 or item.flip_probability >= 0.5
    )
    dc_yield = unique_critical / float(len(selected) or 1)
    summary = {
        "mode": cfg.mode,
        "transition_reason": transition_reason,
        "search_budget": budget,
        "candidate_count": len(ledger),
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "dc_yield": dc_yield,
        "challenger_hit_rate": hit_count / float(len(selected) or 1),
        "frontier_gain_per_refinement": frontier_additions / float(len(selected) or 1),
        "decision_flips": decision_flips,
        "frontier_additions": frontier_additions,
        "dual_critical_predictions": dual_critical,
        "unique_critical_predictions": unique_critical,
        "dc_yield_is_predicted": True,
        "metric_stage": "pre_refinement_prediction",
        "observed_metrics_available": False,
        "predicted_dc_yield": dc_yield,
        "predicted_challenger_hit_rate": hit_count / float(len(selected) or 1),
        "predicted_frontier_gain_per_refinement": frontier_additions / float(len(selected) or 1),
        "predicted_decision_flips": decision_flips,
        "predicted_frontier_additions": frontier_additions,
        "term_ablation_ready": True,
        "bootstrap_seed_size": int(cfg.bootstrap_seed_size),
        "selected_corridor_count": len({item.corridor_signature for item in selected}),
        "selected_mean_overlap": sum(item.overlap for item in selected) / float(len(selected) or 1),
        "selected_mean_predicted_refine_cost": sum(item.predicted_refine_cost for item in selected) / float(len(selected) or 1),
        "effective_search_budget": max(budget, sum(1 for item in selected if item.quota_preserved)),
    }
    summary.update(_dccs_gate_metrics(ledger, selected=selected))
    summary["refine_cost_calibration_metrics"] = {
        "refine_cost_mape": None,
        "refine_cost_mae_ms": None,
        "refine_cost_rank_correlation": None,
        "refine_cost_sample_count": 0,
    }
    return DCCSResult(
        mode=cfg.mode,
        search_budget=budget,
        transition_reason=transition_reason,
        selected=selected,
        skipped=skipped,
        candidate_ledger=_resolved_candidate_ledger(ledger, selected=selected, skipped=skipped),
        summary=summary,
    )
