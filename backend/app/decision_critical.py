from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

OBJECTIVE_NAMES: tuple[str, str, str] = ("time", "money", "co2")
ROAD_CLASS_NAMES: tuple[str, ...] = ("motorway_share", "a_road_share", "urban_share", "other_share")


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


def _mechanism_distance(
    candidate: Mapping[str, float],
    reference_pool: Sequence[Mapping[str, float]],
) -> float:
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


def _predicted_refine_cost(candidate: Mapping[str, Any], *, config: "DCCSConfig") -> float:
    graph_length_km = max(0.0, _as_float(candidate.get("graph_length_km", candidate.get("distance_km"))))
    road_mix = _road_mix(candidate)
    motorway_share = road_mix.get("motorway_share", _as_float(candidate.get("motorway_share")))
    urban_share = road_mix.get("urban_share", _as_float(candidate.get("urban_share")))
    toll_share = max(0.0, _as_float(candidate.get("toll_share")))
    terrain_burden = max(0.0, _as_float(candidate.get("terrain_burden")))
    stretch = _stretch_ratio(candidate)
    complexity = 1.0 + (0.50 * urban_share) + (0.35 * toll_share) + (0.25 * terrain_burden) + (0.15 * (1.0 - motorway_share))
    return max(
        config.refinement_cost_floor,
        (0.45 * graph_length_km) + (0.90 * stretch) + complexity,
    )


def _flip_probability(
    candidate: Mapping[str, Any],
    *,
    objective_gap: float,
    mechanism_gap: float,
    overlap: float,
    stretch: float,
    config: "DCCSConfig",
) -> float:
    proxy_confidence = _confidence_map(candidate)
    confidence = sum(proxy_confidence.values()) / float(len(proxy_confidence) or 1)
    raw = (
        (config.flip_objective_weight * objective_gap)
        + (config.flip_mechanism_weight * mechanism_gap)
        + (config.flip_overlap_weight * (1.0 - overlap))
        + (config.flip_stretch_weight * max(0.0, stretch - 1.0))
        + (config.flip_confidence_weight * (1.0 - confidence))
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
    bootstrap_coverage_weight: float = 1.00
    bootstrap_diversity_weight: float = 0.75
    bootstrap_plausibility_weight: float = 0.30
    bootstrap_overlap_weight: float = 1.10
    challenger_gain_weight: float = 1.00


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
    predicted_refine_cost: float
    flip_probability: float
    score_terms: dict[str, float]
    final_score: float
    decision: str
    decision_reason: str
    mode: str
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
    objective_gap = _normalised_distance(objective, frontier_pool)
    mechanism_gap = _mechanism_distance(mechanism, refined_pool)
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
    score_terms = {
        "objective_gap": objective_gap,
        "mechanism_gap": mechanism_gap,
        "overlap_penalty": overlap,
        "stretch_penalty": detour,
        "flip_probability": flip_probability,
        "predicted_refine_cost": predicted_cost,
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
        predicted_refine_cost=predicted_cost,
        flip_probability=flip_probability,
        score_terms=score_terms,
        final_score=0.0,
        decision="skip",
        decision_reason="pending",
        mode=cfg.mode,
        near_duplicate=near_duplicate,
    )


def _bootstrap_score(record: DCCSCandidateRecord, *, config: DCCSConfig) -> float:
    coverage = record.objective_gap
    diversity = record.mechanism_gap
    plausibility = 1.0 / max(1.0, record.stretch)
    low_overlap = 1.0 - record.overlap
    return (
        (config.bootstrap_coverage_weight * coverage)
        + (config.bootstrap_diversity_weight * diversity)
        + (config.bootstrap_plausibility_weight * plausibility)
        + (config.bootstrap_overlap_weight * low_overlap)
        - (config.cost_weight * record.predicted_refine_cost)
    )


def _challenger_score(record: DCCSCandidateRecord, *, config: DCCSConfig) -> float:
    gain = (
        (config.objective_gap_weight * record.objective_gap)
        + (config.mechanism_gap_weight * record.mechanism_gap)
        + (config.challenger_gain_weight * record.flip_probability)
        + (0.25 * (1.0 - record.overlap))
    )
    penalty = (
        (config.overlap_penalty_weight * record.overlap)
        + (config.stretch_penalty_weight * max(0.0, record.stretch - 1.0))
        + (config.cost_weight * record.predicted_refine_cost)
    )
    return gain / max(1e-9, penalty)


def score_candidate(record: DCCSCandidateRecord, *, config: DCCSConfig | None = None) -> float:
    cfg = config or DCCSConfig()
    if cfg.mode == "bootstrap":
        return _bootstrap_score(record, config=cfg)
    return _challenger_score(record, config=cfg)


def record_refine_outcome(
    record: DCCSCandidateRecord,
    *,
    observed_refine_cost: float,
    frontier_added: bool = False,
    decision_flip: bool = False,
    dominated_but_close: bool = False,
    redundant: bool = False,
) -> DCCSCandidateRecord:
    delta = observed_refine_cost - record.predicted_refine_cost
    ratio = observed_refine_cost / max(1e-9, record.predicted_refine_cost)
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
    return replace(
        record,
        observed_refine_cost=float(observed_refine_cost),
        observed_cost_delta=float(delta),
        refine_cost_error=float(delta),
        refine_cost_ratio=float(ratio),
        decision_reason=label,
    )


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


def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    frontier: Sequence[Mapping[str, Any]] = (),
    refined: Sequence[Mapping[str, Any]] = (),
    config: DCCSConfig | None = None,
) -> DCCSResult:
    cfg = config or DCCSConfig()
    ledger = build_candidate_ledger(candidates, frontier=frontier, refined=refined, config=cfg)
    sorted_records = sorted(
        ledger,
        key=lambda record: (
            -score_candidate(record, config=cfg),
            record.near_duplicate,
            record.candidate_id,
        ),
    )
    if cfg.mode == "bootstrap" and not frontier:
        transition_reason = "bootstrap_seeding:no_strict_frontier"
    elif cfg.mode == "bootstrap":
        transition_reason = "bootstrap_seeding:frontier_unstable"
    else:
        transition_reason = "challenger_mode:strict_frontier_available"

    selected: list[DCCSCandidateRecord] = []
    skipped: list[DCCSCandidateRecord] = []
    seen_paths: set[tuple[str, ...]] = set()
    seen_mechanisms: list[dict[str, float]] = []
    budget = max(0, int(cfg.search_budget))

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
        if cfg.mode == "bootstrap":
            reason = "selected_by_bootstrap"
        else:
            reason = "selected_by_challenger"
        if record.graph_path in seen_paths:
            reason = "duplicate_signature"
        selected_record = replace(
            record,
            final_score=float(score),
            decision="refine" if reason != "duplicate_signature" else "skip",
            decision_reason=reason,
        )
        if reason == "duplicate_signature":
            skipped.append(selected_record)
            continue
        selected.append(selected_record)
        seen_paths.add(record.graph_path)
        seen_mechanisms.append(record.mechanism_descriptor)

    for record in sorted_records:
        if record.candidate_id in {item.candidate_id for item in selected}:
            continue
        if record.candidate_id in {item.candidate_id for item in skipped}:
            continue
        skipped.append(
            replace(
                record,
                final_score=float(score_candidate(record, config=cfg)),
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
    dc_yield = (decision_flips + frontier_additions) / float(len(selected) or 1)
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
        "term_ablation_ready": True,
    }
    return DCCSResult(
        mode=cfg.mode,
        search_budget=budget,
        transition_reason=transition_reason,
        selected=selected,
        skipped=skipped,
        candidate_ledger=ledger,
        summary=summary,
    )
