from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

# DCCS is thesis-specific, but its objective-space coverage and diversity terms
# borrow from standard multi-objective search ideas such as normalized
# nearest-neighbour spacing and crowding/diversification; see Deb et al.,
# "A fast and elitist multiobjective genetic algorithm: NSGA-II",
# https://doi.org/10.1109/4235.996017 .

OBJECTIVE_NAMES: tuple[str, str, str] = ("time", "money", "co2")
ROAD_CLASS_NAMES: tuple[str, ...] = ("motorway_share", "a_road_share", "urban_share", "other_share")
BASELINE_SELECTION_POLICIES: tuple[str, ...] = ("first_n", "random_n", "uniform_corridor_n", "corridor_uniform")
# Deterministic refine-cost model coefficients are fixed in-repo so the
# predictor remains auditable and does not refit at runtime.
_REFINE_COST_MODEL: dict[str, float] = {
    "intercept": 4.75,
    "graph_length_km": 0.95,
    "stretch_excess": 10.5,
    "urban_share": 9.25,
    "toll_share": 6.0,
    "terrain_burden": 4.5,
    "motorway_deficit": 3.1,
    "path_nodes": 0.45,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


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
) -> float:
    # Nearest-neighbour distance in normalized objective space is used as a
    # cheap frontier-gap surrogate before expensive refinement.
    if not pool:
        return 1.0
    mins, scales = _vector_stats(pool)
    best = float("inf")
    for point in pool:
        distance = math.sqrt(
            sum(
                (((candidate[idx] - point[idx]) / scales[idx]) if scales[idx] else 0.0) ** 2
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
    straight_line_km = max(1e-6, _as_float(candidate.get("straight_line_km", candidate.get("od_distance_km", graph_length_km))))
    return max(0.0, graph_length_km / straight_line_km)


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


def _predicted_refine_cost(candidate: Mapping[str, Any], *, config: "DCCSConfig") -> float:
    graph_length_km = max(0.0, _as_float(candidate.get("graph_length_km", candidate.get("distance_km"))))
    road_mix = _road_mix(candidate)
    motorway_share = road_mix.get("motorway_share", _as_float(candidate.get("motorway_share")))
    urban_share = road_mix.get("urban_share", _as_float(candidate.get("urban_share")))
    toll_share = max(0.0, _as_float(candidate.get("toll_share")))
    terrain_burden = max(0.0, _as_float(candidate.get("terrain_burden")))
    stretch = _stretch_ratio(candidate)
    path = _normalise_path(candidate.get("graph_path", candidate.get("path", candidate.get("node_ids"))))
    path_nodes = max(1.0, float(len(path) or 1))
    # OSRM-like route realization cost is dominated by path length and urban
    # complexity rather than the downstream route-option build stage. This
    # heuristic is therefore calibrated in millisecond-like units so later
    # evaluation can report meaningful MAE/MAPE instead of arbitrary ratios.
    # See the OSRM engine overview for table/routing service behavior:
    # https://github.com/Project-OSRM/osrm-backend/wiki/Library-Usage
    complexity = (
        _REFINE_COST_MODEL["intercept"]
        + (_REFINE_COST_MODEL["graph_length_km"] * graph_length_km)
        + (_REFINE_COST_MODEL["stretch_excess"] * max(0.0, stretch - 1.0))
        + (_REFINE_COST_MODEL["urban_share"] * urban_share)
        + (_REFINE_COST_MODEL["toll_share"] * toll_share)
        + (_REFINE_COST_MODEL["terrain_burden"] * terrain_burden)
        + (_REFINE_COST_MODEL["motorway_deficit"] * max(0.0, 1.0 - motorway_share))
        + (_REFINE_COST_MODEL["path_nodes"] * path_nodes)
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
    # Thesis heuristic: turn challenger advantages into a probability-like
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
    search_budget: int = 3
    bootstrap_seed_size: int = 2
    refinement_cost_floor: float = 1.0
    near_duplicate_threshold: float = 0.82
    objective_gap_weight: float = 1.0
    mechanism_gap_weight: float = 0.8
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
    challenger_gain_weight: float = 1.00
    challenger_time_preservation_weight: float = 0.70
    bootstrap_corridor_penalty_weight: float = 0.55
    bootstrap_extremeness_weight: float = 0.45
    bootstrap_corridor_diversity_weight: float = 0.65
    bootstrap_overlap_decay_weight: float = 0.90
    comparator_seed_penalty_weight: float = 0.45


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
    candidate_source_engine: str | None = None
    candidate_source_stage: str | None = None
    comparator_seeded: bool = False
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
    }
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
        candidate_source_engine=candidate_source_engine,
        candidate_source_stage=candidate_source_stage,
        comparator_seeded=comparator_seeded,
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
    coverage = record.objective_gap if not selected else _normalised_distance(record.proxy_objective, selected_objectives)
    extremeness = _extremeness_score(record.proxy_objective, pool_objectives)
    diversity = record.mechanism_gap if not selected else _mechanism_distance(record.mechanism_descriptor, selected_mechanisms)
    plausibility = 1.0 / max(1.0, record.stretch)
    overlap_penalty = _overlap_to_selected(record, selected=selected)
    corridor_reuse_count = sum(1 for item in selected if item.corridor_signature == record.corridor_signature)
    corridor_diversity = 1.0 / float(1 + corridor_reuse_count)
    benefit = (
        (config.bootstrap_coverage_weight * coverage)
        + (config.bootstrap_extremeness_weight * extremeness)
        + (config.bootstrap_diversity_weight * diversity)
        + (config.bootstrap_corridor_diversity_weight * corridor_diversity)
        + (config.bootstrap_plausibility_weight * plausibility)
        + (config.bootstrap_overlap_weight * max(0.0, 1.0 - overlap_penalty))
    )
    cost = 1.0 + (config.cost_weight * record.predicted_refine_cost) + (config.bootstrap_overlap_decay_weight * overlap_penalty)
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
    gain = (
        (config.objective_gap_weight * record.objective_gap)
        + (config.mechanism_gap_weight * record.mechanism_gap)
        + (config.challenger_gain_weight * record.flip_probability)
        + (0.25 * (1.0 - record.overlap))
        + (config.challenger_time_preservation_weight * record.time_preservation_bonus * time_bonus_scale)
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
    return DCCSResult(
        mode=f"{cfg.mode}:{policy_key}",
        search_budget=budget,
        transition_reason=f"baseline_policy:{policy_key}",
        selected=selected,
        skipped=skipped,
        candidate_ledger=_resolved_candidate_ledger(ordered, selected=selected, skipped=skipped),
        summary={
            "mode": f"{cfg.mode}:{policy_key}",
            "transition_reason": f"baseline_policy:{policy_key}",
            "selection_policy": policy_key,
            "search_budget": budget,
            "candidate_count": len(ordered),
            "selected_count": len(selected),
            "skipped_count": len(skipped),
            "selected_corridor_count": len({item.corridor_signature for item in selected}),
        },
    )


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
        sorted_records = sorted(
            ledger,
            key=lambda record: (
                -score_candidate(record, config=cfg),
                record.near_duplicate,
                record.candidate_id,
            ),
        )
        for record in sorted_records:
            if len(selected) >= budget:
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
