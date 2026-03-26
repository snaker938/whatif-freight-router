from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .decision_critical import DCCSResult, DCCSCandidateRecord
from .evidence_certification import FragilityResult

# VOI-AD2R uses a deterministic myopic value-per-cost controller. The design is
# inspired by decision-theoretic control of computation and one-step
# value-of-information policies rather than by learned RL; see Russell and
# Wefald, "Operational Rationality through Compilation",
# https://www2.eecs.berkeley.edu/Pubs/TechRpts/1993/CSD-93-743.pdf , and
# Frazier, Powell, Dayanik, "The Knowledge-Gradient Policy for Correlated
# Normal Beliefs", https://doi.org/10.1287/ijoc.1080.0314 .


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _as_float(value)))


def _route_id(route: Mapping[str, Any] | DCCSCandidateRecord) -> str:
    if isinstance(route, DCCSCandidateRecord):
        return route.candidate_id
    explicit = str(route.get("route_id", route.get("id", ""))).strip()
    if explicit:
        return explicit
    return str(route.get("candidate_id", "")).strip()


def _objective_vector(route: Mapping[str, Any] | DCCSCandidateRecord) -> tuple[float, float, float]:
    if isinstance(route, DCCSCandidateRecord):
        return route.proxy_objective
    if "objective_vector" in route:
        raw = route["objective_vector"]
        if isinstance(raw, Mapping):
            return (
                _as_float(raw.get("time")),
                _as_float(raw.get("money")),
                _as_float(raw.get("co2")),
            )
        if isinstance(raw, Sequence) and len(raw) >= 3:
            return (_as_float(raw[0]), _as_float(raw[1]), _as_float(raw[2]))
    metrics = route.get("metrics")
    if isinstance(metrics, Mapping):
        return (
            _as_float(metrics.get("duration_s")),
            _as_float(metrics.get("monetary_cost")),
            _as_float(metrics.get("emissions_kg")),
        )
    return (
        _as_float(route.get("time")),
        _as_float(route.get("money")),
        _as_float(route.get("co2")),
    )


def _score_from_objectives(
    route: Mapping[str, Any] | DCCSCandidateRecord,
    *,
    weights: tuple[float, float, float],
) -> float:
    vector = _objective_vector(route)
    return (
        (_as_float(weights[0]) * vector[0])
        + (_as_float(weights[1]) * vector[1])
        + (_as_float(weights[2]) * vector[2])
    )


def _certificate_value_from_map(certificate: Mapping[str, float], route_id: str) -> float:
    return _as_float(certificate.get(route_id, 0.0))


def _near_tie_mass(
    certificate: Mapping[str, float],
    *,
    winner_id: str,
    threshold: float,
) -> float:
    winner_value = _certificate_value_from_map(certificate, winner_id)
    if winner_value <= 0.0:
        return 0.0
    mass = 0.0
    for route_id, value in certificate.items():
        if route_id == winner_id:
            continue
        if winner_value - _as_float(value) <= threshold:
            mass += 1.0
    return mass / float(max(1, len(certificate) - 1))


def _certificate_margin(
    certificate: Mapping[str, float],
    *,
    winner_id: str,
) -> float:
    winner_value = _certificate_value_from_map(certificate, winner_id)
    runner_up = max(
        (
            _as_float(value)
            for route_id, value in certificate.items()
            if route_id != winner_id
        ),
        default=0.0,
    )
    return max(0.0, winner_value - runner_up)


def _saturating_gain(value: float) -> float:
    finite = max(0.0, _as_float(value))
    return 1.0 - math.exp(-finite)


def _prior_strength_from_context(ambiguity_context: Mapping[str, Any] | None) -> float:
    context = ambiguity_context if isinstance(ambiguity_context, Mapping) else {}
    return _clamp01(
        max(
            _as_float(context.get("od_ambiguity_index")),
            _as_float(context.get("od_engine_disagreement_prior")),
            _as_float(context.get("od_hard_case_prior")),
            _as_float(context.get("ambiguity_budget_prior")),
        )
    )


def _source_mix_count(raw_source_mix: Any) -> int:
    if raw_source_mix in (None, ""):
        return 0
    if isinstance(raw_source_mix, Mapping):
        return len([key for key in raw_source_mix if str(key).strip()])
    if isinstance(raw_source_mix, Sequence) and not isinstance(raw_source_mix, (str, bytes)):
        return len([item for item in raw_source_mix if str(item).strip()])
    text = str(raw_source_mix).strip()
    if not text:
        return 0
    if text[:1] in {"{", "["}:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        else:
            return _source_mix_count(parsed)
    return len([item.strip() for item in text.replace("+", ",").split(",") if item.strip()])


def _support_richness_score(state: "VOIControllerState") -> float:
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    prior_strength = _prior_strength_from_context(context)
    support_strength = max(
        _clamp01(state.prior_support_strength),
        _clamp01(context.get("od_ambiguity_prior_strength")),
        _clamp01(context.get("support_strength")),
    )
    support_ratio = _clamp01(context.get("od_ambiguity_support_ratio"))
    source_entropy = _clamp01(context.get("od_ambiguity_source_entropy"))
    source_count = _clamp01(_as_float(context.get("od_ambiguity_source_count")) / 4.0)
    source_mix = _clamp01(
        max(
            _source_mix_count(context.get("od_ambiguity_source_mix")),
            int(max(0.0, _as_float(context.get("od_ambiguity_source_mix_count"), 0.0))),
        )
        / 3.0
    )
    corridor_count = _clamp01(_as_float(context.get("od_corridor_family_count")) / 4.0)
    candidate_paths = _clamp01(_as_float(context.get("od_candidate_path_count")) / 6.0)
    return _clamp01(
        (0.24 * prior_strength)
        + (0.22 * support_strength)
        + (0.18 * support_ratio)
        + (0.14 * source_entropy)
        + (0.08 * source_count)
        + (0.06 * source_mix)
        + (0.04 * corridor_count)
        + (0.04 * candidate_paths)
    )


def _ambiguity_pressure_score(state: "VOIControllerState") -> float:
    completeness_gap = max(0.0, _as_float(state.search_completeness_gap))
    near_tie = _clamp01(state.near_tie_mass)
    pending_mass = _clamp01(state.pending_challenger_mass)
    pending_flip = _clamp01(state.best_pending_flip_probability)
    frontier_recall = _clamp01(state.frontier_recall_at_budget)
    certificate_margin = max(0.0, _as_float(state.certificate_margin))
    certificate_margin_pressure = _clamp01(1.0 - min(1.0, certificate_margin / 0.18))
    return _clamp01(
        (0.30 * pending_flip)
        + (0.22 * pending_mass)
        + (0.18 * near_tie)
        + (0.14 * completeness_gap)
        + (0.10 * max(0.0, 1.0 - frontier_recall))
        + (0.06 * certificate_margin_pressure)
    )


def _stress_world_fraction_from_context(ambiguity_context: Mapping[str, Any] | None) -> float:
    context = ambiguity_context if isinstance(ambiguity_context, Mapping) else {}
    return _clamp01(context.get("refc_stress_world_fraction", context.get("stress_world_fraction")))


def _support_rich_ambiguity_window(
    state: "VOIControllerState",
    *,
    support_strength: float | None = None,
    prior_strength: float | None = None,
) -> bool:
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    effective_support = max(
        _clamp01(support_strength),
        _clamp01(state.support_richness),
        _clamp01(context.get("support_strength")),
    )
    support_ratio = _clamp01(context.get("od_ambiguity_support_ratio"))
    source_entropy = _clamp01(context.get("od_ambiguity_source_entropy"))
    effective_prior = max(
        _clamp01(prior_strength),
        _prior_strength_from_context(context),
        _clamp01(context.get("supported_ambiguity_strength")),
    )
    frontier_pressure = _clamp01(max(0.0, len(state.frontier) - 1.0) / 3.0)
    stress_world_fraction = _stress_world_fraction_from_context(context)
    return bool(
        effective_support >= 0.46
        and support_ratio >= 0.50
        and source_entropy >= 0.42
        and (
            frontier_pressure >= 0.33
            or stress_world_fraction >= 0.08
            or effective_prior >= 0.48
        )
    )


def _recent_no_gain_refine_streak(
    state: "VOIControllerState",
    *,
    max_depth: int = 2,
) -> int:
    trace = state.action_trace if isinstance(state.action_trace, Sequence) else []
    streak = 0
    for raw_entry in reversed(trace):
        if streak >= max_depth:
            break
        if not isinstance(raw_entry, Mapping):
            break
        chosen_action = raw_entry.get("chosen_action")
        if not isinstance(chosen_action, Mapping):
            break
        kind = str(chosen_action.get("kind", "")).strip()
        if kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
            break
        realized_certificate_delta = max(0.0, _as_float(raw_entry.get("realized_certificate_delta")))
        realized_frontier_gain = max(0.0, _as_float(raw_entry.get("realized_frontier_gain")))
        realized_selected_route_improvement = max(
            0.0,
            _as_float(raw_entry.get("realized_selected_route_improvement")),
        )
        realized_runner_up_gap_delta = max(0.0, _as_float(raw_entry.get("realized_runner_up_gap_delta")))
        realized_evidence_uncertainty_delta = _as_float(raw_entry.get("realized_evidence_uncertainty_delta"))
        realized_productive = bool(raw_entry.get("realized_productive", False))
        if (
            realized_productive
            or realized_certificate_delta > 1e-9
            or realized_frontier_gain > 0.0
            or realized_selected_route_improvement > 1e-9
            or realized_runner_up_gap_delta > 1e-9
            or realized_evidence_uncertainty_delta < -1e-9
        ):
            break
        streak += 1
    return streak


def _recent_no_gain_controller_streak(
    state: "VOIControllerState",
    *,
    max_depth: int = 2,
) -> int:
    trace = state.action_trace if isinstance(state.action_trace, Sequence) else []
    streak = 0
    eligible_kinds = {
        "refresh_top1_vor",
        "increase_stochastic_samples",
        "refine_top1_dccs",
        "refine_topk_dccs",
    }
    for raw_entry in reversed(trace):
        if streak >= max_depth:
            break
        if not isinstance(raw_entry, Mapping):
            break
        chosen_action = raw_entry.get("chosen_action")
        kind = ""
        if isinstance(chosen_action, Mapping):
            kind = str(chosen_action.get("kind", "")).strip()
        if not kind:
            kind = str(raw_entry.get("kind", "")).strip()
        if kind not in eligible_kinds:
            break
        realized_certificate_delta = max(0.0, _as_float(raw_entry.get("realized_certificate_delta")))
        realized_frontier_gain = max(0.0, _as_float(raw_entry.get("realized_frontier_gain")))
        realized_selected_route_improvement = max(
            0.0,
            _as_float(raw_entry.get("realized_selected_route_improvement")),
        )
        realized_runner_up_gap_delta = max(0.0, _as_float(raw_entry.get("realized_runner_up_gap_delta")))
        realized_evidence_uncertainty_delta = _as_float(raw_entry.get("realized_evidence_uncertainty_delta"))
        realized_productive = bool(raw_entry.get("realized_productive", False))
        if (
            realized_productive
            or realized_certificate_delta > 1e-9
            or realized_frontier_gain > 0.0
            or realized_selected_route_improvement > 1e-9
            or realized_runner_up_gap_delta > 1e-9
            or realized_evidence_uncertainty_delta < -1e-9
        ):
            break
        streak += 1
    return streak


def _recent_no_gain_evidence_action(
    state: "VOIControllerState",
    *,
    max_depth: int = 1,
) -> bool:
    trace = state.action_trace if isinstance(state.action_trace, Sequence) else []
    inspected = 0
    eligible_kinds = {"refresh_top1_vor", "increase_stochastic_samples"}
    for raw_entry in reversed(trace):
        if inspected >= max_depth:
            break
        if not isinstance(raw_entry, Mapping):
            break
        chosen_action = raw_entry.get("chosen_action")
        kind = ""
        if isinstance(chosen_action, Mapping):
            kind = str(chosen_action.get("kind", "")).strip()
        if not kind:
            kind = str(raw_entry.get("kind", "")).strip()
        if kind not in eligible_kinds:
            break
        realized_certificate_delta = max(0.0, _as_float(raw_entry.get("realized_certificate_delta")))
        realized_frontier_gain = max(0.0, _as_float(raw_entry.get("realized_frontier_gain")))
        realized_selected_route_improvement = max(
            0.0,
            _as_float(raw_entry.get("realized_selected_route_improvement")),
        )
        realized_runner_up_gap_delta = max(0.0, _as_float(raw_entry.get("realized_runner_up_gap_delta")))
        realized_evidence_uncertainty_delta = _as_float(raw_entry.get("realized_evidence_uncertainty_delta"))
        realized_productive = bool(raw_entry.get("realized_productive", False))
        if (
            realized_productive
            or realized_certificate_delta > 1e-9
            or realized_frontier_gain > 0.0
            or realized_selected_route_improvement > 1e-9
            or realized_runner_up_gap_delta > 1e-9
            or realized_evidence_uncertainty_delta < -1e-9
        ):
            return False
        inspected += 1
        return True
    return False


def _trace_has_prior_evidence_action_attempt(state: "VOIControllerState") -> bool:
    trace = state.action_trace if isinstance(state.action_trace, Sequence) else []
    for raw_entry in trace:
        if not isinstance(raw_entry, Mapping):
            continue
        chosen_action = raw_entry.get("chosen_action")
        kind = ""
        if isinstance(chosen_action, Mapping):
            kind = str(chosen_action.get("kind", "")).strip()
        if not kind:
            kind = str(raw_entry.get("kind", "")).strip()
        if kind in {"refresh_top1_vor", "increase_stochastic_samples"}:
            return True
    return False


def _actual_refc_world_count(state: "VOIControllerState") -> float:
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    return max(
        0.0,
        _as_float(context.get("refc_unique_world_count")),
        _as_float(context.get("refc_world_count")),
    )


def _requested_refc_world_count(state: "VOIControllerState") -> float:
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    actual_world_count = _actual_refc_world_count(state)
    return max(
        0.0,
        _as_float(context.get("refc_requested_world_count")),
        _as_float(context.get("requested_world_count")),
        actual_world_count,
    )


def _resample_world_expandable(state: "VOIControllerState") -> bool:
    actual_world_count = _actual_refc_world_count(state)
    requested_world_count = _requested_refc_world_count(state)
    if actual_world_count <= 0.0 or requested_world_count <= 0.0:
        return True
    shortfall = max(0.0, requested_world_count - actual_world_count)
    tolerance = max(2.0, 0.05 * requested_world_count)
    return shortfall < tolerance


@dataclass(frozen=True)
class VOIConfig:
    certificate_threshold: float = 0.67
    stop_threshold: float = 0.02
    search_budget: int = 3
    evidence_budget: int = 2
    max_iterations: int = 8
    top_k_refine: int = 2
    resample_increment: int = 25
    lambda_certificate: float = 0.60
    lambda_margin: float = 0.25
    lambda_frontier: float = 0.15
    epsilon: float = 1e-9
    near_tie_threshold: float = 0.03
    search_completeness_threshold: float = 0.84
    search_completeness_action_bonus: float = 0.22
    evidence_uncertainty_threshold: float = 0.08


@dataclass(frozen=True)
class VOIAction:
    action_id: str
    kind: str
    target: str
    cost_search: int = 0
    cost_evidence: int = 0
    predicted_delta_certificate: float = 0.0
    predicted_delta_margin: float = 0.0
    predicted_delta_frontier: float = 0.0
    q_score: float = 0.0
    feasible: bool = True
    preconditions: tuple[str, ...] = ()
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VOIControllerState:
    iteration_index: int
    frontier: list[Mapping[str, Any]]
    certificate: Mapping[str, float]
    winner_id: str
    selected_route_id: str
    remaining_search_budget: int
    remaining_evidence_budget: int
    action_trace: list[dict[str, Any]] = field(default_factory=list)
    state_trace: list[dict[str, Any]] = field(default_factory=list)
    active_evidence_families: list[str] = field(default_factory=list)
    refreshed_evidence_families: list[str] = field(default_factory=list)
    stochastic_enabled: bool = True
    ambiguity_context: dict[str, Any] = field(default_factory=dict)
    near_tie_mass: float = 0.0
    certificate_margin: float = 0.0
    support_richness: float = 0.0
    ambiguity_pressure: float = 0.0
    search_completeness_score: float = 1.0
    search_completeness_gap: float = 0.0
    prior_support_strength: float = 0.0
    pending_challenger_mass: float = 0.0
    best_pending_flip_probability: float = 0.0
    corridor_family_recall: float = 1.0
    frontier_recall_at_budget: float = 1.0
    top_refresh_gain: float = 0.0
    top_fragility_mass: float = 0.0
    competitor_pressure: float = 0.0
    credible_search_uncertainty: bool = False
    credible_evidence_uncertainty: bool = False
    used_search_budget: int = 0
    used_evidence_budget: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration_index": self.iteration_index,
            "frontier": list(self.frontier),
            "certificate": dict(self.certificate),
            "winner_id": self.winner_id,
            "selected_route_id": self.selected_route_id,
            "remaining_search_budget": self.remaining_search_budget,
            "remaining_evidence_budget": self.remaining_evidence_budget,
            "action_trace": list(self.action_trace),
            "state_trace": list(self.state_trace),
            "active_evidence_families": list(self.active_evidence_families),
            "refreshed_evidence_families": list(self.refreshed_evidence_families),
            "stochastic_enabled": bool(self.stochastic_enabled),
            "ambiguity_context": dict(self.ambiguity_context),
            "near_tie_mass": self.near_tie_mass,
            "certificate_margin": self.certificate_margin,
            "support_richness": self.support_richness,
            "ambiguity_pressure": self.ambiguity_pressure,
            "search_completeness_score": self.search_completeness_score,
            "search_completeness_gap": self.search_completeness_gap,
            "prior_support_strength": self.prior_support_strength,
            "pending_challenger_mass": self.pending_challenger_mass,
            "best_pending_flip_probability": self.best_pending_flip_probability,
            "corridor_family_recall": self.corridor_family_recall,
            "frontier_recall_at_budget": self.frontier_recall_at_budget,
            "top_refresh_gain": self.top_refresh_gain,
            "top_fragility_mass": self.top_fragility_mass,
            "competitor_pressure": self.competitor_pressure,
            "credible_search_uncertainty": bool(self.credible_search_uncertainty),
            "credible_evidence_uncertainty": bool(self.credible_evidence_uncertainty),
            "used_search_budget": self.used_search_budget,
            "used_evidence_budget": self.used_evidence_budget,
        }


@dataclass(frozen=True)
class VOIActionHooks:
    refine: Callable[[VOIControllerState, VOIAction], VOIControllerState] | None = None
    refresh: Callable[[VOIControllerState, VOIAction], VOIControllerState] | None = None
    resample: Callable[[VOIControllerState, VOIAction], VOIControllerState] | None = None


@dataclass(frozen=True)
class VOIStopCertificate:
    final_winner_route_id: str
    final_winner_objective_vector: tuple[float, float, float]
    final_strict_frontier_size: int
    certificate_value: float
    certified: bool
    search_budget_used: int
    search_budget_remaining: int
    evidence_budget_used: int
    evidence_budget_remaining: int
    stop_reason: str
    action_trace: list[dict[str, Any]]
    state_trace: list[dict[str, Any]]
    best_rejected_action: dict[str, Any] | None
    ambiguity_summary: dict[str, Any]
    iteration_count: int = 0
    controller_state: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _best_candidate(
    candidates: Sequence[DCCSCandidateRecord],
    *,
    config: VOIConfig,
    ) -> DCCSCandidateRecord | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda record: (
            record.final_score,
            record.flip_probability,
            record.objective_gap,
            -record.predicted_refine_cost,
            record.candidate_id,
        ),
    )


def _top_k_candidates(
    candidates: Sequence[DCCSCandidateRecord],
    *,
    config: VOIConfig,
) -> list[DCCSCandidateRecord]:
    ranked = sorted(
        candidates,
        key=lambda record: (
            record.final_score,
            record.flip_probability,
            record.objective_gap,
            -record.predicted_refine_cost,
            record.candidate_id,
        ),
        reverse=True,
    )
    return ranked[: max(1, min(config.top_k_refine, len(ranked)))]


def _pending_dccs_candidates(dccs: DCCSResult) -> list[DCCSCandidateRecord]:
    selected_ids = {record.candidate_id for record in dccs.selected}
    ranked_pool = [
        record
        for record in [*dccs.skipped, *dccs.candidate_ledger]
        if record.candidate_id not in selected_ids
        and record.decision_reason not in {"duplicate_signature", "duplicate_corridor_bootstrap"}
    ]
    deduped: dict[str, DCCSCandidateRecord] = {}
    for record in ranked_pool:
        if record.near_duplicate and record.decision_reason not in {"budget_exhausted", "not_selected"}:
            continue
        deduped.setdefault(record.candidate_id, record)
    return sorted(
        deduped.values(),
        key=lambda record: (
            record.final_score,
            record.flip_probability,
            record.objective_gap,
            -record.predicted_refine_cost,
            record.candidate_id,
        ),
        reverse=True,
    )


def _candidate_group_signature(candidates: Sequence[DCCSCandidateRecord]) -> str:
    digest = hashlib.sha1()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        digest.update(candidate.candidate_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _aggregate_candidate_group(
    candidates: Sequence[DCCSCandidateRecord],
) -> dict[str, Any]:
    group = list(candidates)
    return {
        "candidate_ids": [item.candidate_id for item in group],
        "cohort_signature": _candidate_group_signature(group),
        "mean_final_score": sum(item.final_score for item in group) / float(len(group) or 1),
        "mean_flip_probability": sum(item.flip_probability for item in group) / float(len(group) or 1),
        "mean_objective_gap": sum(item.objective_gap for item in group) / float(len(group) or 1),
        "mean_mechanism_gap": sum(item.mechanism_gap for item in group) / float(len(group) or 1),
        "mean_overlap": sum(item.overlap for item in group) / float(len(group) or 1),
        "total_predicted_refine_cost": sum(item.predicted_refine_cost for item in group),
    }


def _best_vor_family(fragility: FragilityResult) -> str | None:
    ranking = fragility.value_of_refresh.get("ranking", [])
    if not isinstance(ranking, list) or not ranking:
        return None
    first = ranking[0]
    if not isinstance(first, Mapping):
        return None
    family = str(first.get("family", "")).strip()
    return family or None


def _family_competitor_pressure(
    fragility: FragilityResult,
    *,
    winner_id: str,
    family: str,
) -> float:
    competitor_map = fragility.competitor_fragility_breakdown.get(winner_id, {})
    total = 0.0
    for family_counts in competitor_map.values():
        if not isinstance(family_counts, Mapping):
            continue
        total += _as_float(family_counts.get(family))
    return total


def _winner_objective_vector(state: VOIControllerState) -> tuple[float, float, float]:
    for route in state.frontier:
        if _route_id(route) == state.winner_id:
            return _objective_vector(route)
    if state.frontier:
        return _objective_vector(state.frontier[0])
    return (0.0, 0.0, 0.0)


def compute_search_completeness_metrics(
    state: VOIControllerState,
    *,
    dccs: DCCSResult,
    config: VOIConfig | None = None,
) -> dict[str, float]:
    cfg = config or VOIConfig()
    pending = _pending_dccs_candidates(dccs)
    candidate_pool = [record for record in dccs.candidate_ledger if isinstance(record, DCCSCandidateRecord)]
    selected = [record for record in dccs.selected if isinstance(record, DCCSCandidateRecord)]
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    prior_strength = _prior_strength_from_context(ambiguity_context)
    prior_confidence = _clamp01(
        ambiguity_context.get("od_ambiguity_confidence", 1.0 if prior_strength > 0.0 else 0.0)
    )
    prior_source_count = max(0, int(_as_float(ambiguity_context.get("od_ambiguity_source_count"), 0.0)))
    source_mix_count = max(
        _source_mix_count(ambiguity_context.get("od_ambiguity_source_mix")),
        int(max(0.0, _as_float(ambiguity_context.get("od_ambiguity_source_mix_count"), 0.0))),
    )
    support_ratio = _clamp01(ambiguity_context.get("od_ambiguity_support_ratio"))
    source_entropy = _clamp01(ambiguity_context.get("od_ambiguity_source_entropy"))
    support_evidence = _clamp01(
        (0.50 * prior_confidence)
        + (0.30 * min(1.0, prior_source_count / 4.0))
        + (0.20 * min(1.0, source_mix_count / 3.0))
    )
    # Upstream ambiguity support should reflect both evidential quality and the
    # actual prior strength carried into the controller; high-confidence weak
    # priors should not keep search alive on already-certified winners.
    prior_support_strength = _clamp01(
        prior_strength * (0.45 + (0.55 * support_evidence))
    )
    support_richness = _clamp01(
        (0.19 * prior_strength)
        + (0.17 * prior_support_strength)
        + (0.15 * support_ratio)
        + (0.12 * source_entropy)
        + (0.10 * prior_confidence)
        + (0.08 * min(1.0, prior_source_count / 4.0))
        + (0.08 * min(1.0, source_mix_count / 3.0))
        + (0.06 * min(1.0, _as_float(ambiguity_context.get("od_candidate_path_count"), 0.0) / 6.0))
        + (0.05 * min(1.0, _as_float(ambiguity_context.get("od_corridor_family_count"), 0.0) / 4.0))
    )
    pending_scores = sorted(
        (
            _clamp01(
                (0.55 * record.flip_probability)
                + (0.25 * _saturating_gain(record.objective_gap))
                + (0.15 * _saturating_gain(record.mechanism_gap))
                + (0.05 * max(0.0, 1.0 - record.overlap))
            )
            for record in pending
        ),
        reverse=True,
    )
    pending_window = pending_scores[:3]
    pending_challenger_mass = (
        sum(pending_window) / float(len(pending_window))
        if pending_window
        else 0.0
    )
    best_pending_flip_probability = max((_clamp01(record.flip_probability) for record in pending), default=0.0)
    critical_total = sum(
        _clamp01(
            (0.60 * record.flip_probability)
            + (0.25 * _saturating_gain(record.objective_gap))
            + (0.15 * _saturating_gain(record.mechanism_gap))
        )
        for record in candidate_pool
    )
    critical_selected = sum(
        _clamp01(
            (0.60 * record.flip_probability)
            + (0.25 * _saturating_gain(record.objective_gap))
            + (0.15 * _saturating_gain(record.mechanism_gap))
        )
        for record in selected
    )
    frontier_recall_at_budget = 1.0 if critical_total <= 0.0 else _clamp01(critical_selected / critical_total)
    frontier_pressure = _clamp01(max(0.0, len(state.frontier) - 1.0) / 4.0)
    certificate_margin_pressure = _clamp01(1.0 - min(1.0, _as_float(state.certificate_margin) / 0.20))
    expected_corridor_count = max(1, int(_as_float(ambiguity_context.get("od_corridor_family_count"), 0.0)))
    if expected_corridor_count <= 1 and candidate_pool:
        expected_corridor_count = max(1, len({record.corridor_signature for record in candidate_pool}))
    selected_corridor_count = len({record.corridor_signature for record in selected}) if selected else 0
    corridor_family_recall = _clamp01(selected_corridor_count / float(max(1, expected_corridor_count)))
    stress_world_fraction = _stress_world_fraction_from_context(ambiguity_context)
    uncertainty_support = 0.25 + (0.75 * prior_support_strength)
    ambiguity_pressure = _clamp01(
        (0.27 * pending_challenger_mass)
        + (0.21 * best_pending_flip_probability)
        + (0.15 * (1.0 - frontier_recall_at_budget))
        + (0.12 * (1.0 - corridor_family_recall))
        + (0.10 * frontier_pressure)
        + (0.08 * certificate_margin_pressure)
        + (0.04 * _clamp01(state.near_tie_mass))
        + (0.04 * stress_world_fraction)
        + (0.03 * support_richness)
        + (0.02 * frontier_pressure * support_richness)
    )
    risk = (
        (0.32 * pending_challenger_mass * (0.40 + (0.60 * prior_support_strength)))
        + (0.22 * best_pending_flip_probability * uncertainty_support)
        + (0.17 * (1.0 - frontier_recall_at_budget))
        + (0.10 * (1.0 - corridor_family_recall))
        + (0.08 * prior_strength * prior_support_strength)
        + (0.06 * frontier_pressure * certificate_margin_pressure)
        + (0.05 * _clamp01(state.near_tie_mass))
        + (0.05 * stress_world_fraction * max(support_richness, prior_support_strength))
        + (0.03 * frontier_pressure * support_richness)
    )
    search_completeness_score = _clamp01(1.0 - risk)
    search_completeness_gap = max(0.0, float(cfg.search_completeness_threshold) - search_completeness_score)
    return {
        "search_completeness_score": round(search_completeness_score, 6),
        "search_completeness_gap": round(search_completeness_gap, 6),
        "pending_challenger_mass": round(pending_challenger_mass, 6),
        "best_pending_flip_probability": round(best_pending_flip_probability, 6),
        "corridor_family_recall": round(corridor_family_recall, 6),
        "frontier_recall_at_budget": round(frontier_recall_at_budget, 6),
        "prior_support_strength": round(prior_support_strength, 6),
        "support_richness": round(support_richness, 6),
        "ambiguity_pressure": round(ambiguity_pressure, 6),
    }


def _with_search_completeness(
    state: VOIControllerState,
    *,
    dccs: DCCSResult,
    config: VOIConfig | None = None,
) -> VOIControllerState:
    return replace(state, **compute_search_completeness_metrics(state, dccs=dccs, config=config))


def _apply_search_completeness_bonus(
    action: VOIAction,
    *,
    state: VOIControllerState,
    config: VOIConfig,
) -> VOIAction:
    if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return action
    gap = max(0.0, _as_float(state.search_completeness_gap))
    if gap <= 0.0:
        return action
    current_certificate = _certificate_value_from_map(state.certificate, state.winner_id)
    certified_surplus = max(0.0, current_certificate - _as_float(config.certificate_threshold))
    pending_flip = _clamp01(state.best_pending_flip_probability)
    pending_mass = _clamp01(state.pending_challenger_mass)
    support_strength = max(_clamp01(state.prior_support_strength), _clamp01(state.support_richness))
    ambiguity_pressure = _clamp01(state.ambiguity_pressure)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    hard_case_prior = max(
        _prior_strength_from_context(ambiguity_context),
        _clamp01(ambiguity_context.get("od_hard_case_prior")),
        _clamp01(ambiguity_context.get("ambiguity_budget_prior")),
    )
    supported_hard_case = _support_rich_ambiguity_window(
        state,
        support_strength=support_strength,
        prior_strength=hard_case_prior,
    )
    suppress_top1_bonus = bool(
        action.kind == "refine_top1_dccs"
        and current_certificate >= _as_float(config.certificate_threshold)
        and supported_hard_case
        and _clamp01(state.top_fragility_mass) >= 0.12
        and _clamp01(state.competitor_pressure) >= 0.75
        and bool(state.credible_evidence_uncertainty)
    )
    if (
        current_certificate >= _as_float(config.certificate_threshold)
        and pending_flip < 0.30
        and pending_mass < 0.25
        and support_strength < 0.30
        and ambiguity_pressure < 0.22
    ):
        return action
    bonus_scale = _as_float(config.search_completeness_action_bonus)
    exploration_gate = max(
        0.05,
        min(
            1.0,
            ((0.45 * pending_flip) + (0.25 * gap) + (0.15 * pending_mass) + (0.15 * ambiguity_pressure))
            * (0.30 + (0.70 * support_strength)),
        ),
    )
    if suppress_top1_bonus:
        exploration_gate = 0.0
    if certified_surplus > 0.0:
        exploration_gate *= max(0.05, 1.0 - (3.0 * certified_surplus))
    frontier_bonus = bonus_scale * gap * exploration_gate
    certificate_headroom = max(0.0, _as_float(config.certificate_threshold) - current_certificate)
    certificate_bonus = frontier_bonus * max(0.05, certificate_headroom + (0.50 * pending_flip))
    margin_bonus = frontier_bonus * max(0.05, 0.5 * gap + 0.5 * _clamp01(state.pending_challenger_mass))
    metadata = dict(action.metadata)
    metadata["search_completeness_gap"] = round(gap, 6)
    metadata["search_completeness_score"] = round(_as_float(state.search_completeness_score), 6)
    metadata["search_completeness_bonus"] = round(frontier_bonus, 6)
    metadata["best_pending_flip_probability"] = round(pending_flip, 6)
    metadata["pending_challenger_mass"] = round(_as_float(state.pending_challenger_mass), 6)
    metadata["corridor_family_recall"] = round(_as_float(state.corridor_family_recall), 6)
    metadata["frontier_recall_at_budget"] = round(_as_float(state.frontier_recall_at_budget), 6)
    metadata["exploration_gate"] = round(exploration_gate, 6)
    metadata["current_certificate"] = round(current_certificate, 6)
    metadata["prior_support_strength"] = round(support_strength, 6)
    metadata["support_richness"] = round(_clamp01(state.support_richness), 6)
    metadata["ambiguity_pressure"] = round(ambiguity_pressure, 6)
    boosted = replace(
        action,
        predicted_delta_certificate=float(action.predicted_delta_certificate + certificate_bonus),
        predicted_delta_margin=float(action.predicted_delta_margin + margin_bonus),
        predicted_delta_frontier=float(action.predicted_delta_frontier + frontier_bonus),
        metadata=metadata,
    )
    return score_action(boosted, config=config)


def _certified_supported_hard_case_refine_penalty(
    action: VOIAction,
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
    stress_world_fraction: float,
    recent_no_gain_refine_streak: int,
) -> VOIAction:
    if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return action
    if current_certificate < _as_float(config.certificate_threshold):
        return action
    if not evidence_uncertainty:
        return action
    top_fragility_mass = _clamp01(state.top_fragility_mass)
    competitor_pressure = _clamp01(state.competitor_pressure)
    if top_fragility_mass < 0.12 or competitor_pressure < 0.75:
        return action
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    support_strength = max(_clamp01(state.support_richness), _clamp01(state.prior_support_strength))
    hard_case_prior = max(
        _prior_strength_from_context(ambiguity_context),
        _clamp01(ambiguity_context.get("od_hard_case_prior")),
        _clamp01(ambiguity_context.get("ambiguity_budget_prior")),
    )
    support_window_ok = _support_rich_ambiguity_window(
        state,
        support_strength=support_strength,
        prior_strength=hard_case_prior,
    )
    hard_case_support = max(support_strength, hard_case_prior)
    if not (support_window_ok or hard_case_support >= 0.60):
        return action
    mechanism_gap = _clamp01(action.metadata.get("normalized_mechanism_gap"))
    objective_gap = _clamp01(action.metadata.get("normalized_objective_gap"))
    if mechanism_gap >= 0.14 and objective_gap < 0.40:
        return action
    flip_probability = _clamp01(action.metadata.get("mean_flip_probability"))
    overlap_reduction = _clamp01(action.metadata.get("normalized_overlap_reduction"))
    ambiguity_pressure = _clamp01(state.ambiguity_pressure)
    evidence_pressure = _clamp01(
        (0.38 * top_fragility_mass)
        + (0.28 * competitor_pressure)
        + (0.12 * _clamp01(state.top_refresh_gain))
        + (0.10 * stress_world_fraction)
        + (0.06 * support_strength)
        + (0.06 * ambiguity_pressure)
    )
    mechanism_weakness = _clamp01(1.0 - min(1.0, mechanism_gap / 0.14))
    penalty = min(
        0.40,
        (
            0.12
            + (0.18 * evidence_pressure)
            + (0.08 * mechanism_weakness)
            + (0.05 * objective_gap)
            + (0.04 * flip_probability)
            + (0.03 * overlap_reduction)
        ),
    )
    if recent_no_gain_refine_streak > 0:
        penalty = min(0.45, penalty + 0.04)
    if action.kind == "refine_topk_dccs":
        penalty = min(0.50, penalty + 0.06)
    if penalty <= 0.0:
        return action
    metadata = dict(action.metadata)
    metadata["certified_supported_hard_case_penalty"] = round(penalty, 6)
    metadata["certified_supported_hard_case_penalized"] = True
    return replace(
        action,
        q_score=max(0.0, _as_float(action.q_score) - penalty),
        metadata=metadata,
    )


def _build_refine_action(
    candidate: DCCSCandidateRecord,
    *,
    kind: str,
    top_k: int = 1,
    cohort: Sequence[DCCSCandidateRecord] | None = None,
    config: VOIConfig,
) -> VOIAction:
    # Search-action gains are deterministic surrogates from the DCCS ledger:
    # flip potential, objective gap, mechanism gap, and overlap reduction.
    candidate_group = list(cohort or [candidate])
    aggregate = _aggregate_candidate_group(candidate_group)
    avg_flip = _as_float(aggregate["mean_flip_probability"])
    avg_gap = _as_float(aggregate["mean_objective_gap"])
    avg_mechanism = _as_float(aggregate["mean_mechanism_gap"])
    avg_overlap = _as_float(aggregate["mean_overlap"])
    normalized_gap = _saturating_gain(avg_gap)
    normalized_mechanism = _saturating_gain(avg_mechanism)
    normalized_overlap_reduction = max(0.0, min(1.0, 1.0 - avg_overlap))
    total_predicted_cost = _as_float(aggregate["total_predicted_refine_cost"])
    predicted_delta_certificate = max(
        0.0,
        (0.35 * avg_flip) + (0.30 * normalized_gap) + (0.15 * normalized_mechanism),
    )
    predicted_delta_margin = max(
        0.0,
        (0.40 * normalized_gap) + (0.20 * normalized_overlap_reduction),
    )
    predicted_delta_frontier = max(0.0, normalized_gap + (0.05 * max(0, len(candidate_group) - 1)))
    cost_search = max(1, int(top_k))
    return VOIAction(
        action_id=f"{kind}:{aggregate['cohort_signature']}",
        kind=kind,
        target=str(aggregate["cohort_signature"]),
        cost_search=cost_search,
        predicted_delta_certificate=predicted_delta_certificate,
        predicted_delta_margin=predicted_delta_margin,
        predicted_delta_frontier=predicted_delta_frontier,
        preconditions=("search_budget_available", "dccs_candidate_available"),
        reason="refine_candidate",
        metadata={
            "top_k": top_k,
            **aggregate,
            "normalized_objective_gap": normalized_gap,
            "normalized_mechanism_gap": normalized_mechanism,
            "normalized_overlap_reduction": normalized_overlap_reduction,
            "predicted_refine_cost": total_predicted_cost,
        },
    )


def _build_refresh_action(
    family: str,
    *,
    fragility: FragilityResult,
    winner_id: str,
    current_certificate: float,
    config: VOIConfig,
) -> VOIAction:
    # Refresh-action gains are derived directly from REFC's value-of-refresh
    # estimates so the controller stays aligned with the fixed certification
    # model.
    ranking = fragility.value_of_refresh.get("ranking", [])
    vor_gain = 0.0
    if isinstance(ranking, list):
        for row in ranking:
            if isinstance(row, Mapping) and str(row.get("family", "")) == family:
                vor_gain = _as_float(row.get("vor"))
                break
    route_fragility = _as_float(fragility.route_fragility_map.get(winner_id, {}).get(family))
    competitor_pressure = _family_competitor_pressure(
        fragility,
        winner_id=winner_id,
        family=family,
    )
    normalized_pressure = _saturating_gain(0.10 * competitor_pressure)
    certificate_ratio = max(0.0, min(1.0, current_certificate / max(config.certificate_threshold, 1e-9)))
    evidence_stability_factor = 0.35 + (0.65 * certificate_ratio)
    return VOIAction(
        action_id=f"refresh:{family}",
        kind="refresh_top1_vor",
        target=family,
        cost_evidence=1,
        predicted_delta_certificate=max(
            0.0,
            evidence_stability_factor * (vor_gain + (0.35 * route_fragility) + (0.10 * normalized_pressure)),
        ),
        predicted_delta_margin=max(
            0.0,
            evidence_stability_factor * ((0.60 * vor_gain) + (0.25 * route_fragility) + (0.10 * normalized_pressure)),
        ),
        predicted_delta_frontier=0.0,
        preconditions=("evidence_budget_available", "vor_available"),
        reason="refresh_evidence_family",
        metadata={
            "vor_gain": vor_gain,
            "route_fragility": route_fragility,
            "competitor_pressure": competitor_pressure,
            "normalized_competitor_pressure": normalized_pressure,
            "current_certificate": current_certificate,
            "evidence_stability_factor": evidence_stability_factor,
        },
    )


def _certified_refresh_priority_bonus(
    *,
    state: VOIControllerState,
    enriched_state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
    recent_no_gain_refine_streak: int,
    stress_world_fraction: float,
) -> float:
    if not evidence_uncertainty or not supported_fragility_uncertainty:
        return 0.0
    if current_certificate < _as_float(config.certificate_threshold):
        return 0.0
    top_fragility_mass = _clamp01(enriched_state.top_fragility_mass)
    competitor_pressure = _clamp01(enriched_state.competitor_pressure)
    if top_fragility_mass < 0.12 or competitor_pressure < 0.75:
        return 0.0
    support_context = _clamp01(
        (0.45 * _clamp01(enriched_state.support_richness))
        + (0.35 * _clamp01(enriched_state.ambiguity_pressure))
        + (0.20 * _clamp01(enriched_state.prior_support_strength))
    )
    evidence_signature = _clamp01(
        (0.50 * top_fragility_mass)
        + (0.30 * competitor_pressure)
        + (0.20 * _clamp01(min(1.0, _as_float(enriched_state.top_refresh_gain) / 0.01)))
        + (0.10 * stress_world_fraction)
    )
    margin_tension = _clamp01(1.0 - min(1.0, max(0.0, _as_float(enriched_state.certificate_margin)) / 0.12))
    decision_pressure = _clamp01(max(_clamp01(enriched_state.near_tie_mass), _clamp01(state.near_tie_mass)))
    bonus = (
        (0.030 + (0.075 * evidence_signature))
        * (0.55 + (0.45 * support_context))
        * (0.70 + (0.30 * margin_tension))
        * (0.85 + (0.15 * decision_pressure))
    )
    if recent_no_gain_refine_streak > 0:
        bonus += 0.015
    return min(0.120, bonus)


def _search_action_shows_strong_decision_movement(
    action: VOIAction | None,
    *,
    state: VOIControllerState,
) -> bool:
    if action is None or action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return False
    flip_probability = _clamp01(action.metadata.get("mean_flip_probability"))
    objective_gap = _clamp01(action.metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(action.metadata.get("normalized_mechanism_gap"))
    overlap_gain = _clamp01(action.metadata.get("normalized_overlap_reduction"))
    predicted_certificate = max(0.0, _as_float(action.predicted_delta_certificate))
    predicted_margin = max(0.0, _as_float(action.predicted_delta_margin))
    predicted_frontier = max(0.0, _as_float(action.predicted_delta_frontier))
    state_pending_signal = max(
        _clamp01(state.best_pending_flip_probability),
        _clamp01(state.pending_challenger_mass),
        _clamp01(1.0 - state.frontier_recall_at_budget),
    )
    structural_movement = bool(
        objective_gap >= 0.18
        or mechanism_gap >= 0.18
        or predicted_frontier >= 0.12
    )
    certificate_movement = bool(
        predicted_certificate >= 0.22
        and (flip_probability >= 0.45 or state_pending_signal >= 0.30)
    )
    margin_movement = bool(
        predicted_margin >= 0.22
        and overlap_gain >= 0.30
        and (flip_probability >= 0.45 or state_pending_signal >= 0.30)
    )
    return bool(structural_movement or certificate_movement or margin_movement)


def _refine_action_has_genuine_novel_search_promise(
    action: VOIAction | None,
    *,
    state: VOIControllerState,
) -> bool:
    if action is None or action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return False
    flip_probability = _clamp01(action.metadata.get("mean_flip_probability"))
    objective_gap = _clamp01(action.metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(action.metadata.get("normalized_mechanism_gap"))
    predicted_frontier = _clamp01(_as_float(action.predicted_delta_frontier) / 0.10)
    competitor_pressure = _clamp01(state.competitor_pressure)
    mechanism_novelty = bool(
        mechanism_gap >= 0.12
        and (
            objective_gap >= 0.08
            or flip_probability >= 0.18
            or predicted_frontier >= 0.08
        )
    )
    competitor_backed_challenge = bool(
        competitor_pressure >= 0.28
        and (
            objective_gap >= 0.14
            or flip_probability >= 0.24
            or predicted_frontier >= 0.10
        )
    )
    return bool(mechanism_novelty or competitor_backed_challenge)


def _should_stop_certified_refine_churn(
    action: VOIAction | None,
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    recent_no_gain_refine_streak: int,
    best_evidence_q: float | None,
    evidence_uncertainty: bool,
    genuine_novel_search_promise: bool,
    strong_decision_movement: bool,
) -> bool:
    if action is None or action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return False
    if current_certificate < _as_float(config.certificate_threshold):
        return False
    if recent_no_gain_refine_streak < 1:
        return False
    if genuine_novel_search_promise:
        return False
    objective_gap = _clamp01(action.metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(action.metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(action.predicted_delta_frontier))
    structural_reopen_signal = bool(
        objective_gap >= 0.16
        or mechanism_gap >= 0.12
        or predicted_frontier >= 0.16
    )
    if strong_decision_movement and structural_reopen_signal:
        return False
    settled_frontier = bool(
        len(state.frontier) <= 1
        or (
            len(state.frontier) <= 2
            and _clamp01(state.near_tie_mass) <= max(config.near_tie_threshold, 0.04)
        )
    )
    if not settled_frontier:
        return False
    predicted_only_search_pressure = bool(
        max(0.0, _as_float(state.search_completeness_gap)) >= 0.10
        or _clamp01(state.pending_challenger_mass) >= 0.35
        or _clamp01(state.best_pending_flip_probability) >= 0.40
    )
    if not predicted_only_search_pressure:
        return False
    no_evidence_signal = bool(
        max(0.0, _as_float(state.top_refresh_gain)) <= 1e-9
        and _clamp01(state.top_fragility_mass) <= 1e-9
        and _clamp01(state.competitor_pressure) <= 0.05
    )
    if not no_evidence_signal:
        return False
    no_evidence_path = bool((not evidence_uncertainty) or best_evidence_q is None or best_evidence_q <= 0.0)
    if not no_evidence_path:
        return False
    return True


def _certified_support_rich_first_refine_discount(
    action: VOIAction,
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> VOIAction:
    if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return action
    if current_certificate < _as_float(config.certificate_threshold):
        return action
    if state.action_trace:
        return action
    if not _support_rich_ambiguity_window(state):
        return action
    top_refresh_gain = max(0.0, _as_float(state.top_refresh_gain))
    top_fragility_mass = _clamp01(state.top_fragility_mass)
    if top_refresh_gain <= 0.0 and top_fragility_mass <= 0.0:
        return action
    competitor_pressure = _clamp01(state.competitor_pressure)
    if competitor_pressure > 0.45:
        return action
    flip_probability = _clamp01(action.metadata.get("mean_flip_probability"))
    objective_gap = _clamp01(action.metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(action.metadata.get("normalized_mechanism_gap"))
    overlap_gain = _clamp01(action.metadata.get("normalized_overlap_reduction"))
    frontier_signal = _clamp01(_as_float(action.predicted_delta_frontier) / 0.10)
    genuine_novel_search_promise = _refine_action_has_genuine_novel_search_promise(
        action,
        state=state,
    )
    if genuine_novel_search_promise:
        return action
    support_context = _clamp01(
        (0.48 * _clamp01(state.support_richness))
        + (0.28 * _clamp01(state.prior_support_strength))
        + (0.24 * _clamp01(state.ambiguity_pressure))
    )
    evidence_signal = _clamp01(
        (0.55 * min(1.0, top_refresh_gain / 0.01))
        + (0.45 * top_fragility_mass)
    )
    low_competitor_signal = _clamp01(max(0.0, 0.45 - competitor_pressure) / 0.45)
    certificate_margin = max(0.0, _as_float(state.certificate_margin))
    margin_tension = _clamp01(1.0 - min(1.0, certificate_margin / 0.10))
    generic_search_signal = _clamp01(
        (0.32 * overlap_gain)
        + (0.22 * (1.0 - min(1.0, objective_gap / 0.12)))
        + (0.18 * (1.0 - min(1.0, mechanism_gap / 0.12)))
        + (0.16 * (1.0 - min(1.0, flip_probability / 0.35)))
        + (0.12 * (1.0 - frontier_signal))
    )
    discount = min(
        0.12,
        0.010
        + (0.016 * evidence_signal)
        + (0.010 * support_context)
        + (0.008 * low_competitor_signal)
        + (0.006 * margin_tension),
    )
    adjusted_q_score = max(0.0, _as_float(action.q_score) - discount)
    evidence_anchor = max(
        0.0,
        (0.55 * top_refresh_gain) + (0.45 * top_fragility_mass),
    )
    if evidence_anchor > 0.0:
        cap_multiplier = max(
            0.08,
            0.78
            - (0.18 * generic_search_signal)
            - (0.08 * support_context)
            - (0.06 * low_competitor_signal)
            - (0.05 * margin_tension),
        )
        adjusted_q_score = min(adjusted_q_score, evidence_anchor * cap_multiplier)
    reduction = max(0.0, _as_float(action.q_score) - adjusted_q_score)
    if reduction <= 0.0:
        return action
    metadata = dict(action.metadata)
    metadata["support_rich_certified_first_refine_discount"] = round(reduction, 6)
    metadata["support_rich_certified_first_refine_discount_q_cap"] = round(adjusted_q_score, 6)
    metadata["support_rich_certified_first_refine_discount_evidence_anchor"] = round(evidence_anchor, 6)
    metadata["support_rich_certified_first_refine_generic_search_signal"] = round(generic_search_signal, 6)
    metadata["support_rich_certified_first_refine_novel_search_promise"] = genuine_novel_search_promise
    metadata["support_rich_certified_first_refine_discount_applied"] = True
    return replace(
        action,
        q_score=adjusted_q_score,
        metadata=metadata,
    )


def _apply_support_rich_certified_refresh_preference(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
    recent_no_gain_refine_streak: int,
) -> list[VOIAction]:
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    support_rich_window = _support_rich_ambiguity_window(state)
    top_refresh_gain = max(0.0, _as_float(state.top_refresh_gain))
    top_fragility_mass = _clamp01(state.top_fragility_mass)
    direct_winner_side_evidence_signal = bool(top_refresh_gain > 0.0 or top_fragility_mass > 0.0)
    hard_case_pressure = max(
        _clamp01(state.ambiguity_pressure),
        _clamp01(ambiguity_context.get("od_hard_case_prior")),
        _clamp01(ambiguity_context.get("ambiguity_budget_prior")),
    )
    first_iteration_search_tension = bool(
        len(state.frontier) >= 2
        and (
            max(0.0, _as_float(state.search_completeness_gap)) >= 0.12
            or _clamp01(state.pending_challenger_mass) >= 0.35
            or max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)) >= 0.20
        )
    )
    first_iteration_supported_hard_case = bool(
        recent_no_gain_refine_streak <= 0
        and evidence_uncertainty
        and supported_fragility_uncertainty
        and support_rich_window
        and direct_winner_side_evidence_signal
        and first_iteration_search_tension
        and hard_case_pressure >= 0.35
    )
    if current_certificate < _as_float(config.certificate_threshold):
        return list(actions)
    if (
        (not evidence_uncertainty or not supported_fragility_uncertainty)
        and not first_iteration_supported_hard_case
    ):
        return list(actions)
    if not support_rich_window:
        return list(actions)
    if not direct_winner_side_evidence_signal:
        return list(actions)
    if recent_no_gain_refine_streak <= 0 and not first_iteration_supported_hard_case:
        return list(actions)
    refresh_index = next((idx for idx, action in enumerate(actions) if action.kind == "refresh_top1_vor"), None)
    if refresh_index is None:
        return list(actions)
    best_resample_action = max(
        (action for action in actions if action.kind == "increase_stochastic_samples"),
        key=lambda action: action.q_score,
        default=None,
    )
    real_near_tie_resample_pressure = bool(
        best_resample_action is not None
        and _clamp01(state.near_tie_mass) >= 0.10
        and top_refresh_gain <= 0.0025
        and top_fragility_mass <= 0.02
    )
    if real_near_tie_resample_pressure and recent_no_gain_refine_streak <= 0:
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if (
        _refine_action_has_genuine_novel_search_promise(best_search_action, state=state)
        and recent_no_gain_refine_streak <= 0
    ):
        return list(actions)
    if (
        _search_action_shows_strong_decision_movement(best_search_action, state=state)
        and recent_no_gain_refine_streak <= 0
        and not first_iteration_supported_hard_case
    ):
        return list(actions)
    best_competing_q = max(
        (
            action.q_score
            for idx, action in enumerate(actions)
            if idx != refresh_index and action.kind != "stop"
        ),
        default=0.0,
    )
    refresh_action = actions[refresh_index]
    evidence_signal = _clamp01(
        (0.45 * top_fragility_mass)
        + (0.35 * min(1.0, top_refresh_gain / 0.01))
        + (0.20 * _clamp01(state.competitor_pressure))
    )
    bonus = max(0.0, best_competing_q - refresh_action.q_score)
    if first_iteration_supported_hard_case:
        threshold_gap = max(0.0, _as_float(config.stop_threshold) - refresh_action.q_score)
        bonus += threshold_gap + 0.002 + (0.010 * evidence_signal)
    else:
        bonus += 0.006 + (0.012 * evidence_signal)
    bonus = min(0.18, bonus)
    if bonus <= 0.0:
        return list(actions)
    metadata = dict(refresh_action.metadata)
    metadata["support_rich_certified_refresh_preference_bonus"] = round(bonus, 6)
    metadata["support_rich_certified_refresh_preference_applied"] = True
    if first_iteration_supported_hard_case:
        metadata["support_rich_certified_refresh_preference_first_action_bridge"] = True
    updated = list(actions)
    updated[refresh_index] = replace(
        refresh_action,
        q_score=refresh_action.q_score + bonus,
        metadata=metadata,
    )
    return updated


def _suppress_stress_only_resample_for_certified_support_rich_row(
    state: "VOIControllerState",
    *,
    current_certificate: float,
    config: VOIConfig,
    stress_world_fraction: float,
) -> bool:
    if current_certificate < _as_float(config.certificate_threshold):
        return False
    if not _support_rich_ambiguity_window(state):
        return False
    if stress_world_fraction < 0.08:
        return False
    if _clamp01(state.near_tie_mass) >= 0.03:
        return False
    if _as_float(state.top_refresh_gain) <= 0.0 and _clamp01(state.top_fragility_mass) <= 0.0:
        return False
    return True


def _build_resample_action(
    *,
    near_tie_mass: float,
    stress_world_fraction: float,
    top_fragility_mass: float,
    config: VOIConfig,
) -> VOIAction:
    predicted_delta_certificate = max(
        0.0,
        (0.16 * near_tie_mass) + (0.10 * stress_world_fraction) + (0.08 * top_fragility_mass),
    )
    return VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        cost_evidence=1,
        predicted_delta_certificate=predicted_delta_certificate,
        predicted_delta_margin=max(0.0, (0.08 * near_tie_mass) + (0.05 * top_fragility_mass)),
        predicted_delta_frontier=max(0.0, 0.03 * max(near_tie_mass, stress_world_fraction)),
        preconditions=("evidence_budget_available", "near_tie_set_nonempty"),
        reason="increase_stochastic_samples",
        metadata={
            "near_tie_mass": near_tie_mass,
            "stress_world_fraction": stress_world_fraction,
            "top_fragility_mass": top_fragility_mass,
            "sample_increment": config.resample_increment,
        },
    )


def _should_offer_evidence_discovery_bridge(
    state: "VOIControllerState",
    *,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
    recent_no_gain_controller_streak: int,
    best_search_action: VOIAction | None,
) -> bool:
    if state.stochastic_enabled:
        return False
    if _requested_refc_world_count(state) <= _actual_refc_world_count(state):
        return False
    if not _resample_world_expandable(state):
        return False
    if current_certificate < _as_float(config.certificate_threshold):
        return False
    if not evidence_uncertainty or not supported_fragility_uncertainty:
        return False
    if not _support_rich_ambiguity_window(state):
        return False
    if _trace_has_prior_evidence_action_attempt(state):
        return False
    if recent_no_gain_controller_streak > 0:
        return False
    if max(0.0, _as_float(state.top_refresh_gain)) > 0.0:
        return False
    if _clamp01(state.top_fragility_mass) > 0.0:
        return False
    if _clamp01(state.near_tie_mass) > 0.03:
        return False
    if _refine_action_has_genuine_novel_search_promise(best_search_action, state=state):
        return False
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    hard_case_pressure = max(
        _clamp01(state.ambiguity_pressure),
        _clamp01(ambiguity_context.get("od_hard_case_prior")),
        _clamp01(ambiguity_context.get("ambiguity_budget_prior")),
    )
    stress_world_fraction = _stress_world_fraction_from_context(ambiguity_context)
    search_tension = bool(
        max(0.0, _as_float(state.search_completeness_gap)) >= 0.12
        or _clamp01(state.pending_challenger_mass) >= 0.35
        or max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)) >= 0.20
    )
    return bool(
        search_tension
        and hard_case_pressure >= 0.35
        and stress_world_fraction >= 0.08
    )


def score_action(action: VOIAction, *, config: VOIConfig | None = None) -> VOIAction:
    cfg = config or VOIConfig()
    cost = max(0.0, float(action.cost_search + action.cost_evidence))
    # Thesis controller heuristic: a deterministic value-per-cost ranking over
    # certificate gain, margin gain, and frontier gain. This is a transparent
    # metareasoning surrogate, not a claim of optimal VOI.
    q_score = (
        (cfg.lambda_certificate * action.predicted_delta_certificate)
        + (cfg.lambda_margin * action.predicted_delta_margin)
        + (cfg.lambda_frontier * action.predicted_delta_frontier)
    ) / (cost + cfg.epsilon)
    return replace(action, q_score=float(q_score))


def build_action_menu(
    state: VOIControllerState,
    *,
    dccs: DCCSResult,
    fragility: FragilityResult,
    config: VOIConfig | None = None,
) -> list[VOIAction]:
    cfg = config or VOIConfig()
    enriched_state = enrich_controller_state_for_actioning(state, dccs=dccs, fragility=fragility, config=cfg)
    actions: list[VOIAction] = []
    pending_candidates = _pending_dccs_candidates(dccs)
    best_candidate = _best_candidate(pending_candidates, config=cfg)
    current_certificate = _certificate_value_from_map(enriched_state.certificate, enriched_state.winner_id)
    initial_search_uncertainty = _credible_search_uncertainty(
        state,
        config=cfg,
        current_certificate=current_certificate,
    )
    search_uncertainty = _credible_search_uncertainty(
        enriched_state,
        config=cfg,
        current_certificate=current_certificate,
    )
    evidence_uncertainty = _credible_evidence_uncertainty(
        enriched_state,
        fragility=fragility,
        config=cfg,
        current_certificate=current_certificate,
    )
    enriched_state = replace(
        enriched_state,
        credible_search_uncertainty=bool(search_uncertainty or initial_search_uncertainty),
        credible_evidence_uncertainty=bool(evidence_uncertainty),
    )
    support_richness = _clamp01(enriched_state.support_richness)
    ambiguity_pressure = _clamp01(enriched_state.ambiguity_pressure)
    stress_world_fraction = _stress_world_fraction_from_context(enriched_state.ambiguity_context)
    raw_support_richness = _support_richness_score(state)
    raw_search_pressure = max(
        max(0.0, _as_float(state.search_completeness_gap)),
        _clamp01(state.pending_challenger_mass),
        _clamp01(state.best_pending_flip_probability),
        max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)),
        _clamp01(state.near_tie_mass),
    )
    preserve_certified_search_actions = bool(
        initial_search_uncertainty
        and raw_search_pressure >= 0.12
        and (
            raw_support_richness >= 0.34
            or _clamp01(state.ambiguity_pressure) >= 0.22
            or _support_rich_ambiguity_window(state)
        )
    )
    strong_live_evidence_signal = bool(
        _clamp01(enriched_state.top_fragility_mass) >= 0.20
        or _clamp01(enriched_state.competitor_pressure) >= 0.75
        or _as_float(enriched_state.top_refresh_gain) >= 0.10
    )
    supported_fragility_uncertainty = bool(
        _support_rich_ambiguity_window(enriched_state)
        and (
            _clamp01(enriched_state.top_fragility_mass) >= 0.08
            or _clamp01(enriched_state.competitor_pressure) >= 0.12
            or _as_float(enriched_state.top_refresh_gain) > 0.0
            or stress_world_fraction >= 0.10
        )
    )
    support_rich_search_pressure = bool(
        _support_rich_ambiguity_window(enriched_state)
        and (
            max(0.0, _as_float(enriched_state.search_completeness_gap)) >= 0.07
            or _clamp01(enriched_state.pending_challenger_mass) >= 0.20
            or _clamp01(enriched_state.best_pending_flip_probability) >= 0.24
            or _clamp01(1.0 - enriched_state.frontier_recall_at_budget) >= 0.10
            or (
                len(enriched_state.frontier) >= 2
                and _clamp01(enriched_state.near_tie_mass) >= 0.05
                and (
                    _clamp01(enriched_state.pending_challenger_mass) >= 0.14
                    or _clamp01(enriched_state.best_pending_flip_probability) >= 0.18
                    or _clamp01(1.0 - enriched_state.frontier_recall_at_budget) >= 0.08
                )
            )
        )
    )
    search_reopen_uncertainty = bool(
        search_uncertainty
        and (
            current_certificate < _as_float(cfg.certificate_threshold)
            or (
                (support_richness >= 0.28 or ambiguity_pressure >= 0.22)
                and (
                    initial_search_uncertainty
                    or support_rich_search_pressure
                    or supported_fragility_uncertainty
                )
            )
        )
    )
    allow_search_actions = (
        current_certificate < _as_float(cfg.certificate_threshold)
        or search_reopen_uncertainty
    )
    evidence_action_support = bool(
        strong_live_evidence_signal
        or (
            supported_fragility_uncertainty
            and (
                _clamp01(enriched_state.top_fragility_mass) >= 0.05
                or _clamp01(enriched_state.competitor_pressure) >= 0.08
                or _as_float(enriched_state.top_refresh_gain) > 0.0
                or stress_world_fraction >= 0.06
                or (
                    current_certificate >= _as_float(cfg.certificate_threshold)
                    and (support_richness >= 0.34 or ambiguity_pressure >= 0.22)
                )
            )
        )
    )
    allow_evidence_actions = (
        state.remaining_evidence_budget > 0
        and (
            current_certificate < _as_float(cfg.certificate_threshold)
            or evidence_action_support
        )
    )
    recent_no_gain_refine_streak = _recent_no_gain_refine_streak(enriched_state)
    recent_no_gain_controller_streak = _recent_no_gain_controller_streak(enriched_state)
    recent_no_gain_evidence_action = _recent_no_gain_evidence_action(enriched_state)
    certified_refresh_priority_bonus = _certified_refresh_priority_bonus(
        state=state,
        enriched_state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
        recent_no_gain_refine_streak=recent_no_gain_refine_streak,
        stress_world_fraction=stress_world_fraction,
    )
    if best_candidate is not None and state.remaining_search_budget > 0 and allow_search_actions:
        refine_top1_action = _apply_search_completeness_bonus(
            score_action(_build_refine_action(best_candidate, kind="refine_top1_dccs", config=cfg), config=cfg),
            state=enriched_state,
            config=cfg,
        )
        refine_top1_action = _certified_supported_hard_case_refine_penalty(
            refine_top1_action,
            state=state,
            current_certificate=current_certificate,
            config=cfg,
            evidence_uncertainty=evidence_uncertainty,
            supported_fragility_uncertainty=supported_fragility_uncertainty,
            stress_world_fraction=stress_world_fraction,
            recent_no_gain_refine_streak=recent_no_gain_refine_streak,
        )
        refine_top1_action = _certified_support_rich_first_refine_discount(
            refine_top1_action,
            state=enriched_state,
            current_certificate=current_certificate,
            config=cfg,
        )
        actions.append(refine_top1_action)
        top_k = min(cfg.top_k_refine, len(pending_candidates), state.remaining_search_budget)
        if top_k > 1:
            cohort = _top_k_candidates(pending_candidates, config=cfg)[:top_k]
            aggregate = max(
                cohort,
                key=lambda record: (
                    record.final_score,
                    record.flip_probability,
                    record.candidate_id,
                ),
            )
            actions.append(
                _apply_search_completeness_bonus(
                    score_action(
                        _build_refine_action(
                            aggregate,
                            kind="refine_topk_dccs",
                            top_k=top_k,
                            cohort=cohort,
                            config=cfg,
                        ),
                        config=cfg,
                    ),
                    state=enriched_state,
                    config=cfg,
                )
            )
            topk_action = actions[-1]
            topk_action = _certified_supported_hard_case_refine_penalty(
                topk_action,
                state=state,
                current_certificate=current_certificate,
                config=cfg,
                evidence_uncertainty=evidence_uncertainty,
                supported_fragility_uncertainty=supported_fragility_uncertainty,
                stress_world_fraction=stress_world_fraction,
                recent_no_gain_refine_streak=recent_no_gain_refine_streak,
            )
            topk_action = _certified_support_rich_first_refine_discount(
                topk_action,
                state=enriched_state,
                current_certificate=current_certificate,
                config=cfg,
            )
            actions[-1] = topk_action
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    best_family = _best_vor_family(fragility)
    refreshed_families = {family for family in state.refreshed_evidence_families if str(family).strip()}
    if (
        best_family is not None
        and best_family not in refreshed_families
        and allow_evidence_actions
    ):
        refresh_action = score_action(
            _build_refresh_action(
                best_family,
                fragility=fragility,
                winner_id=state.winner_id,
                current_certificate=_as_float(state.certificate.get(state.winner_id)),
                config=cfg,
            ),
            config=cfg,
        )
        if certified_refresh_priority_bonus > 0.0:
            refresh_action = replace(
                refresh_action,
                q_score=refresh_action.q_score + certified_refresh_priority_bonus,
            )
        actions.append(refresh_action)
    suppress_stress_only_resample = _suppress_stress_only_resample_for_certified_support_rich_row(
        enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        stress_world_fraction=stress_world_fraction,
    )
    if (
        state.stochastic_enabled
        and allow_evidence_actions
        and max(state.near_tie_mass, stress_world_fraction, enriched_state.top_fragility_mass) > 0.0
        and not suppress_stress_only_resample
        and _resample_world_expandable(enriched_state)
    ):
        actions.append(
            score_action(
                _build_resample_action(
                    near_tie_mass=state.near_tie_mass,
                    stress_world_fraction=stress_world_fraction,
                    top_fragility_mass=enriched_state.top_fragility_mass,
                    config=cfg,
                ),
                config=cfg,
            )
        )
    evidence_discovery_bridge = _should_offer_evidence_discovery_bridge(
        enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
        recent_no_gain_controller_streak=recent_no_gain_controller_streak,
        best_search_action=best_search_action,
    )
    if evidence_discovery_bridge and not any(action.kind == "increase_stochastic_samples" for action in actions):
        bridge_action = _build_resample_action(
            near_tie_mass=state.near_tie_mass,
            stress_world_fraction=stress_world_fraction,
            top_fragility_mass=enriched_state.top_fragility_mass,
            config=cfg,
        )
        bridge_metadata = dict(bridge_action.metadata)
        bridge_metadata["evidence_discovery_bridge"] = True
        bridge_metadata["stochastic_bridge_required"] = True
        bridge_action = replace(
            bridge_action,
            preconditions=("evidence_budget_available", "evidence_discovery_bridge"),
            reason="increase_stochastic_samples_bridge",
            metadata=bridge_metadata,
        )
        actions.append(score_action(bridge_action, config=cfg))
    actions = _apply_support_rich_certified_refresh_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
        recent_no_gain_refine_streak=recent_no_gain_refine_streak,
    )
    resample_shortfall_available = bool(
        _requested_refc_world_count(enriched_state) > _actual_refc_world_count(enriched_state)
    )
    zero_signal_refresh_saturated = bool(
        current_certificate >= _as_float(cfg.certificate_threshold)
        and evidence_uncertainty
        and supported_fragility_uncertainty
        and _support_rich_ambiguity_window(enriched_state)
        and max(0.0, _as_float(enriched_state.top_refresh_gain)) <= 0.0
        and _clamp01(enriched_state.top_fragility_mass) <= 0.0
        and not evidence_discovery_bridge
        and (
            not _resample_world_expandable(enriched_state)
            or not resample_shortfall_available
        )
    )
    if zero_signal_refresh_saturated:
        actions = [action for action in actions if action.kind != "refresh_top1_vor"]
    if (
        recent_no_gain_evidence_action
        and current_certificate >= _as_float(cfg.certificate_threshold)
        and max(0.0, _as_float(enriched_state.top_refresh_gain)) <= 0.0
        and _clamp01(enriched_state.top_fragility_mass) <= 0.0
    ):
        actions = [action for action in actions if action.kind != "refresh_top1_vor"]
    if actions and current_certificate >= _as_float(cfg.certificate_threshold):
        best_evidence_q = max(
            (
                action.q_score
                for action in actions
                if action.kind in {"refresh_top1_vor", "increase_stochastic_samples"}
            ),
            default=None,
        )
        filtered_actions: list[VOIAction] = []
        for action in actions:
            if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
                filtered_actions.append(action)
                continue
            genuine_novel_search_promise = _refine_action_has_genuine_novel_search_promise(
                action,
                state=enriched_state,
            )
            predicted_total = (
                (0.45 * _as_float(action.predicted_delta_certificate))
                + (0.30 * _as_float(action.predicted_delta_margin))
                + (0.25 * _as_float(action.predicted_delta_frontier))
            )
            flip_probability = _clamp01(action.metadata.get("mean_flip_probability"))
            objective_gap = _clamp01(action.metadata.get("normalized_objective_gap"))
            mechanism_gap = _clamp01(action.metadata.get("normalized_mechanism_gap"))
            overlap_gain = _clamp01(action.metadata.get("normalized_overlap_reduction"))
            search_floor = max(
                0.05,
                min(
                    0.24,
                    (0.05 * max(0.0, current_certificate - _as_float(cfg.certificate_threshold)))
                    + (0.07 * _clamp01(enriched_state.pending_challenger_mass))
                    + (0.06 * _clamp01(enriched_state.best_pending_flip_probability))
                    + (0.04 * max(0.0, 1.0 - _clamp01(enriched_state.frontier_recall_at_budget)))
                    + (0.02 * _clamp01(enriched_state.near_tie_mass)),
                ),
            )
            evidence_dominates = bool(
                evidence_uncertainty
                and best_evidence_q is not None
                and best_evidence_q >= (action.q_score * 1.02)
            )
            settled_search = bool(
                _clamp01(enriched_state.frontier_recall_at_budget) >= 0.88
                and _clamp01(enriched_state.pending_challenger_mass) <= 0.22
                and _clamp01(enriched_state.best_pending_flip_probability) <= 0.28
                and _clamp01(enriched_state.near_tie_mass) <= 0.12
                and max(0.0, _as_float(enriched_state.certificate_margin)) >= 0.03
            )
            materially_distinct = bool(
                flip_probability >= 0.32
                or objective_gap >= 0.18
                or mechanism_gap >= 0.18
                or overlap_gain >= 0.22
            )
            limited_decision_upside = bool(
                objective_gap < 0.10
                and _as_float(action.predicted_delta_frontier) < 0.04
                and predicted_total < max(search_floor * 1.5, 0.22)
            )
            credible_refresh_path = bool(
                evidence_uncertainty
                and supported_fragility_uncertainty
                and _clamp01(enriched_state.top_fragility_mass) >= 0.12
                and _clamp01(enriched_state.competitor_pressure) >= 0.75
                and (
                    _clamp01(enriched_state.top_refresh_gain) > 0.0
                    or _clamp01(enriched_state.top_fragility_mass) > 0.0
                    or _clamp01(enriched_state.competitor_pressure) > 0.0
                )
            )
            evidence_clearly_superior = bool(
                evidence_uncertainty
                and best_evidence_q is not None
                and best_evidence_q >= (action.q_score * 1.50)
                and objective_gap < 0.10
                and _as_float(action.predicted_delta_frontier) < 0.04
            )
            certified_evidence_prefers_refresh = bool(
                evidence_uncertainty
                and best_evidence_q is not None
                and best_evidence_q >= max(action.q_score, action.q_score * 1.02)
                and predicted_total < max(search_floor, 0.14)
                and not materially_distinct
            )
            certified_refine_stall = bool(
                recent_no_gain_refine_streak > 0
                and evidence_uncertainty
                and supported_fragility_uncertainty
                and best_evidence_q is not None
                and best_evidence_q > 0.0
            )
            strong_decision_movement = _search_action_shows_strong_decision_movement(
                action,
                state=enriched_state,
            )
            certified_refine_churn_stop = _should_stop_certified_refine_churn(
                action,
                state=enriched_state,
                current_certificate=current_certificate,
                config=cfg,
                recent_no_gain_refine_streak=recent_no_gain_refine_streak,
                best_evidence_q=best_evidence_q,
                evidence_uncertainty=evidence_uncertainty,
                genuine_novel_search_promise=genuine_novel_search_promise,
                strong_decision_movement=strong_decision_movement,
            )
            controller_reopen_stall = bool(
                recent_no_gain_controller_streak > 0
                and current_certificate >= _as_float(cfg.certificate_threshold)
                and not genuine_novel_search_promise
                and not strong_decision_movement
            )
            saturated_zero_signal_search_stall = bool(
                zero_signal_refresh_saturated
                and not genuine_novel_search_promise
                and not strong_decision_movement
            )
            if evidence_discovery_bridge and not genuine_novel_search_promise:
                continue
            if (
                preserve_certified_search_actions
                and search_reopen_uncertainty
                and not certified_refine_stall
                and not controller_reopen_stall
                and not saturated_zero_signal_search_stall
                and not certified_refine_churn_stop
            ):
                filtered_actions.append(action)
                continue
            if evidence_clearly_superior:
                continue
            if certified_refine_churn_stop:
                continue
            if controller_reopen_stall:
                continue
            if saturated_zero_signal_search_stall:
                continue
            if (
                certified_refine_stall
                and current_certificate >= _as_float(cfg.certificate_threshold)
                and not materially_distinct
                and predicted_total <= max(search_floor * 1.10, best_evidence_q * 1.15)
            ):
                continue
            if (
                current_certificate >= _as_float(cfg.certificate_threshold)
                and credible_refresh_path
                and (certified_refine_stall or limited_decision_upside)
                and not materially_distinct
                and predicted_total <= max(search_floor * 1.30, 0.22)
            ):
                continue
            if settled_search and evidence_dominates and (not materially_distinct or limited_decision_upside):
                continue
            if (
                not support_rich_search_pressure
                and (
                    predicted_total < search_floor
                    or certified_evidence_prefers_refresh
                    or (
                        evidence_uncertainty
                        and supported_fragility_uncertainty
                        and evidence_dominates
                        and predicted_total <= max(search_floor, 0.16)
                    )
                )
            ):
                continue
            filtered_actions.append(action)
        actions = filtered_actions
    actions.append(
        VOIAction(
            action_id="stop",
            kind="stop",
            target="stop",
            q_score=0.0,
            feasible=True,
            preconditions=("always",),
            reason="stop",
        )
    )
    return sorted(actions, key=lambda action: (-action.q_score, action.kind, action.target))


def _apply_action(
    state: VOIControllerState,
    action: VOIAction,
    *,
    hooks: VOIActionHooks | None,
) -> VOIControllerState:
    def _normalize_updated_state(updated: VOIControllerState) -> VOIControllerState:
        next_iteration = max(int(updated.iteration_index), int(state.iteration_index) + 1)
        next_search_budget = max(0, int(updated.remaining_search_budget))
        next_evidence_budget = max(0, int(updated.remaining_evidence_budget))
        if action.cost_search:
            next_search_budget = min(
                next_search_budget,
                max(0, int(state.remaining_search_budget) - int(action.cost_search)),
            )
        if action.cost_evidence:
            next_evidence_budget = min(
                next_evidence_budget,
                max(0, int(state.remaining_evidence_budget) - int(action.cost_evidence)),
            )
        return replace(
            updated,
            iteration_index=next_iteration,
            remaining_search_budget=next_search_budget,
            remaining_evidence_budget=next_evidence_budget,
        )

    if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
        if hooks is not None and hooks.refine is not None:
            return _normalize_updated_state(hooks.refine(state, action))
        return _normalize_updated_state(replace(
            state,
            remaining_search_budget=max(0, state.remaining_search_budget - action.cost_search),
        ))
    if action.kind == "refresh_top1_vor":
        if hooks is not None and hooks.refresh is not None:
            refreshed_state = hooks.refresh(state, action)
            refreshed_families = list(dict.fromkeys([*refreshed_state.refreshed_evidence_families, action.target]))
            return _normalize_updated_state(replace(
                refreshed_state,
                refreshed_evidence_families=refreshed_families,
            ))
        refreshed = list(dict.fromkeys([*state.refreshed_evidence_families, action.target]))
        return _normalize_updated_state(replace(
            state,
            remaining_evidence_budget=max(0, state.remaining_evidence_budget - action.cost_evidence),
            refreshed_evidence_families=refreshed,
        ))
    if action.kind == "increase_stochastic_samples":
        if hooks is not None and hooks.resample is not None:
            return _normalize_updated_state(hooks.resample(state, action))
        return _normalize_updated_state(replace(
            state,
            remaining_evidence_budget=max(0, state.remaining_evidence_budget - action.cost_evidence),
        ))
    return state


def _normalize_state(
    state: VOIControllerState,
    *,
    config: VOIConfig,
) -> VOIControllerState:
    return replace(
        state,
        near_tie_mass=_near_tie_mass(state.certificate, winner_id=state.winner_id, threshold=config.near_tie_threshold),
        certificate_margin=_certificate_margin(state.certificate, winner_id=state.winner_id),
        used_search_budget=max(0, int(config.search_budget) - max(0, int(state.remaining_search_budget))),
        used_evidence_budget=max(0, int(config.evidence_budget) - max(0, int(state.remaining_evidence_budget))),
    )


def _ambiguity_summary(fragility: FragilityResult, winner_id: str) -> dict[str, Any]:
    route_fragility = fragility.route_fragility_map.get(winner_id, {})
    ranked_families = sorted(route_fragility.items(), key=lambda item: (-_as_float(item[1]), item[0]))
    competitor_map = fragility.competitor_fragility_breakdown.get(winner_id, {})
    ranked_competitors = sorted(
        (
            (competitor_id, sum(int(value) for value in family_counts.values()))
            for competitor_id, family_counts in competitor_map.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    summary = dict(fragility.value_of_refresh)
    summary["top_fragility_families"] = [family for family, value in ranked_families[:3] if _as_float(value) > 0.0]
    summary["top_competitor_route_id"] = ranked_competitors[0][0] if ranked_competitors and ranked_competitors[0][1] > 0 else None
    summary["top_competitor_pairwise_defeats"] = ranked_competitors[0][1] if ranked_competitors else 0
    return summary


def _with_evidence_signals(
    state: VOIControllerState,
    *,
    fragility: FragilityResult,
) -> VOIControllerState:
    ranking = fragility.value_of_refresh.get("ranking", [])
    top_refresh_gain = 0.0
    if isinstance(ranking, Sequence):
        top_refresh_gain = max(
            (
                _as_float(entry.get("vor"))
                for entry in ranking
                if isinstance(entry, Mapping)
            ),
            default=0.0,
        )
    route_fragility = fragility.route_fragility_map.get(state.winner_id, {})
    top_fragility_mass = max((_as_float(value) for value in route_fragility.values()), default=0.0)
    competitor_pressure = max(
        (
            _family_competitor_pressure(fragility, winner_id=state.winner_id, family=family)
            for family in route_fragility
        ),
        default=0.0,
    )
    return replace(
        state,
        top_refresh_gain=round(top_refresh_gain, 6),
        top_fragility_mass=round(top_fragility_mass, 6),
        competitor_pressure=round(_clamp01(competitor_pressure), 6),
    )


def enrich_controller_state_for_actioning(
    state: VOIControllerState,
    *,
    dccs: DCCSResult,
    fragility: FragilityResult,
    config: VOIConfig,
) -> VOIControllerState:
    # Keep the live loop and the menu builder aligned on the same enriched
    # controller assumptions before any credible uncertainty checks run.
    return _with_evidence_signals(
        _with_search_completeness(state, dccs=dccs, config=config),
        fragility=fragility,
    )


def refresh_controller_state_after_action(
    state: VOIControllerState,
    *,
    dccs: DCCSResult,
    fragility: FragilityResult,
    config: VOIConfig,
    frontier: Sequence[Mapping[str, Any]],
    certificate: Mapping[str, float],
    winner_id: str,
    selected_route_id: str,
    remaining_search_budget: int,
    remaining_evidence_budget: int,
    active_evidence_families: Sequence[str],
    refreshed_evidence_families: Sequence[str] = (),
    stochastic_enabled: bool | None = None,
    ambiguity_context: Mapping[str, Any] | None = None,
    action_trace: Sequence[Mapping[str, Any]] | None = None,
) -> VOIControllerState:
    refreshed_state = replace(
        state,
        frontier=[dict(route) for route in frontier],
        certificate=dict(certificate),
        winner_id=winner_id,
        selected_route_id=selected_route_id,
        remaining_search_budget=max(0, int(remaining_search_budget)),
        remaining_evidence_budget=max(0, int(remaining_evidence_budget)),
        action_trace=list(action_trace or state.action_trace),
        active_evidence_families=list(active_evidence_families),
        refreshed_evidence_families=list(refreshed_evidence_families),
        stochastic_enabled=state.stochastic_enabled if stochastic_enabled is None else bool(stochastic_enabled),
        ambiguity_context=dict(ambiguity_context or state.ambiguity_context or {}),
    )
    return enrich_controller_state_for_actioning(
        refreshed_state,
        dccs=dccs,
        fragility=fragility,
        config=config,
    )


def _credible_search_uncertainty(
    state: VOIControllerState,
    *,
    config: VOIConfig,
    current_certificate: float,
) -> bool:
    threshold = _as_float(config.certificate_threshold)
    if current_certificate < threshold:
        return True
    pending_flip = _clamp01(state.best_pending_flip_probability)
    pending_mass = _clamp01(state.pending_challenger_mass)
    support_strength = max(_clamp01(state.prior_support_strength), _clamp01(state.support_richness))
    prior_strength = _prior_strength_from_context(state.ambiguity_context)
    completeness_gap = max(0.0, _as_float(state.search_completeness_gap))
    near_tie = _clamp01(state.near_tie_mass)
    certificate_surplus = max(0.0, current_certificate - threshold)
    certificate_margin = max(0.0, _as_float(state.certificate_margin))
    frontier_recall = _clamp01(state.frontier_recall_at_budget)
    ambiguity_pressure = _clamp01(state.ambiguity_pressure)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    support_ratio = _clamp01(ambiguity_context.get("od_ambiguity_support_ratio"))
    source_entropy = _clamp01(ambiguity_context.get("od_ambiguity_source_entropy"))
    stress_world_fraction = _stress_world_fraction_from_context(ambiguity_context)
    frontier_pressure = _clamp01(max(0.0, len(state.frontier) - 1.0) / 3.0)
    support_rich_ambiguity = _support_rich_ambiguity_window(
        state,
        support_strength=support_strength,
        prior_strength=prior_strength,
    )
    search_incompleteness_signal = max(
        completeness_gap,
        pending_flip,
        pending_mass,
        max(0.0, 1.0 - frontier_recall),
    )
    supported_search_pressure = _clamp01(
        (0.26 * pending_flip)
        + (0.18 * pending_mass)
        + (0.16 * completeness_gap)
        + (0.14 * ambiguity_pressure)
        + (0.10 * near_tie)
        + (0.08 * max(0.0, 1.0 - frontier_recall))
        + (0.05 * stress_world_fraction)
        + (0.03 * frontier_pressure)
    )
    if (
        current_certificate >= 0.99
        and near_tie <= 0.03
        and certificate_margin >= 0.12
        and not (support_rich_ambiguity and supported_search_pressure >= 0.24)
    ):
        return False
    low_supported_ambiguity = prior_strength <= 0.30 and support_strength <= 0.30
    if (
        current_certificate >= min(1.0, threshold + 0.18)
        and near_tie <= 0.03
        and certificate_margin >= 0.12
        and prior_strength <= 0.20
        and support_strength <= 0.20
        and not (support_rich_ambiguity and supported_search_pressure >= 0.24)
    ):
        return False
    if (
        near_tie <= 0.03
        and certificate_margin >= 0.12
        and frontier_recall >= 0.92
        and completeness_gap <= 0.05
        and pending_flip <= 0.22
        and pending_mass <= 0.18
        and low_supported_ambiguity
        and not (support_rich_ambiguity and supported_search_pressure >= 0.22)
    ):
        return False
    if (
        support_rich_ambiguity
        and search_incompleteness_signal < 0.12
        and certificate_margin >= 0.03
        and near_tie <= 0.15
    ):
        return False
    if support_strength < 0.28 and ambiguity_pressure < 0.18 and stress_world_fraction < 0.08:
        return False
    support_gate = (
        0.20
        + (0.40 * support_strength)
        + (0.25 * prior_strength)
        + (0.15 * ambiguity_pressure)
        + (0.10 * max(stress_world_fraction, 0.6 * support_ratio, 0.5 * source_entropy))
    )
    challenger_risk = (
        (
            (0.34 * pending_flip)
            + (0.18 * pending_mass)
            + (0.18 * near_tie)
            + (0.16 * completeness_gap)
            + (0.14 * prior_strength)
        ) * support_gate
        + (0.10 * max(0.0, 1.0 - frontier_recall))
        + (0.08 * stress_world_fraction)
        + (0.06 * frontier_pressure * max(support_strength, support_ratio))
    )
    certainty_relief = min(0.28, certificate_surplus * 1.25) + min(0.08, certificate_margin * 0.40)
    if support_rich_ambiguity and supported_search_pressure > max(0.17, 0.33 - certainty_relief):
        return True
    return challenger_risk > max(0.16, 0.36 - certainty_relief)


def _credible_evidence_uncertainty(
    state: VOIControllerState,
    *,
    fragility: FragilityResult,
    config: VOIConfig,
    current_certificate: float,
) -> bool:
    threshold = _as_float(config.certificate_threshold)
    top_refresh_gain = max(0.0, _as_float(state.top_refresh_gain))
    top_fragility_mass = max(0.0, _as_float(state.top_fragility_mass))
    competitor_pressure = _clamp01(state.competitor_pressure)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    prior_strength = _prior_strength_from_context(ambiguity_context)
    hard_case_prior = max(
        prior_strength,
        _clamp01(ambiguity_context.get("od_hard_case_prior")),
        _clamp01(ambiguity_context.get("ambiguity_budget_prior")),
    )
    support_strength = max(
        _clamp01(state.prior_support_strength),
        _clamp01(ambiguity_context.get("od_ambiguity_prior_strength")),
    )
    support_richness = max(
        support_strength,
        _clamp01(state.support_richness),
        _clamp01(ambiguity_context.get("od_ambiguity_support_ratio")),
        _clamp01(ambiguity_context.get("od_ambiguity_source_entropy")),
    )
    support_ratio = _clamp01(ambiguity_context.get("od_ambiguity_support_ratio"))
    source_entropy = _clamp01(ambiguity_context.get("od_ambiguity_source_entropy"))
    stress_world_fraction = _stress_world_fraction_from_context(ambiguity_context)
    frontier_pressure = _clamp01(max(0.0, len(state.frontier) - 1.0) / 3.0)
    support_rich_ambiguity = _support_rich_ambiguity_window(
        state,
        support_strength=support_strength,
        prior_strength=hard_case_prior,
    )
    recent_no_gain_refine_streak = _recent_no_gain_refine_streak(state)
    supported_evidence_pressure = _clamp01(
        (0.28 * top_refresh_gain)
        + (0.22 * top_fragility_mass)
        + (0.16 * competitor_pressure)
        + (0.12 * stress_world_fraction)
        + (0.10 * _clamp01(state.near_tie_mass))
        + (0.07 * frontier_pressure)
        + (0.05 * _clamp01(1.0 - min(1.0, _as_float(state.certificate_margin) / 0.16)))
    )
    if top_refresh_gain <= 0.0 and top_fragility_mass <= 0.0 and stress_world_fraction <= 0.0:
        return False
    if support_richness <= 0.24 and support_ratio <= 0.20 and source_entropy <= 0.18:
        return False
    if (
        support_rich_ambiguity
        and recent_no_gain_refine_streak > 0
        and supported_evidence_pressure > 0.0
    ):
        return True
    if (
        current_certificate >= min(1.0, threshold + 0.18)
        and state.certificate_margin >= 0.18
        and top_refresh_gain <= 0.02
        and top_fragility_mass <= 0.08
        and stress_world_fraction <= 0.08
        and not (support_rich_ambiguity and supported_evidence_pressure >= 0.18)
    ):
        return False
    evidence_risk = _clamp01(
        (0.28 * top_refresh_gain)
        + (0.22 * top_fragility_mass)
        + (0.16 * stress_world_fraction)
        + (0.12 * competitor_pressure)
        + (0.10 * hard_case_prior)
        + (0.05 * support_ratio)
        + (0.05 * _clamp01(state.near_tie_mass))
        + (0.04 * _clamp01(1.0 - min(1.0, _as_float(state.certificate_margin) / 0.18)))
        + (0.03 * source_entropy)
        + (0.03 * frontier_pressure)
    )
    evidence_risk *= 0.45 + (0.55 * max(support_richness, support_ratio, 0.75 * source_entropy))
    certificate_surplus = max(0.0, current_certificate - threshold)
    certainty_relief = min(0.16, certificate_surplus * 0.9) + min(0.06, _as_float(state.certificate_margin) * 0.25)
    if support_rich_ambiguity and supported_evidence_pressure > max(0.08, 0.16 - certainty_relief):
        return True
    return evidence_risk > max(_as_float(config.evidence_uncertainty_threshold), 0.18 - certainty_relief)


def credible_search_uncertainty(
    state: VOIControllerState,
    *,
    config: VOIConfig,
    current_certificate: float,
) -> bool:
    return _credible_search_uncertainty(
        state,
        config=config,
        current_certificate=current_certificate,
    )


def credible_evidence_uncertainty(
    state: VOIControllerState,
    *,
    fragility: FragilityResult,
    config: VOIConfig,
    current_certificate: float,
) -> bool:
    return _credible_evidence_uncertainty(
        state,
        fragility=fragility,
        config=config,
        current_certificate=current_certificate,
    )


def _state_snapshot(
    state: VOIControllerState,
    *,
    current_certificate: float,
    feasible_actions: Sequence[VOIAction],
    chosen_action: VOIAction | None,
    best_rejected_action: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "iteration": state.iteration_index,
        "winner_id": state.winner_id,
        "selected_route_id": state.selected_route_id,
        "certificate": dict(state.certificate),
        "current_certificate": current_certificate,
        "frontier_route_ids": [_route_id(route) for route in state.frontier],
        "remaining_search_budget": state.remaining_search_budget,
        "remaining_evidence_budget": state.remaining_evidence_budget,
        "used_search_budget": state.used_search_budget,
        "used_evidence_budget": state.used_evidence_budget,
        "near_tie_mass": state.near_tie_mass,
        "certificate_margin": state.certificate_margin,
        "support_richness": state.support_richness,
        "ambiguity_pressure": state.ambiguity_pressure,
        "search_completeness_score": state.search_completeness_score,
        "search_completeness_gap": state.search_completeness_gap,
        "pending_challenger_mass": state.pending_challenger_mass,
        "best_pending_flip_probability": state.best_pending_flip_probability,
        "corridor_family_recall": state.corridor_family_recall,
        "frontier_recall_at_budget": state.frontier_recall_at_budget,
        "top_refresh_gain": state.top_refresh_gain,
        "top_fragility_mass": state.top_fragility_mass,
        "competitor_pressure": state.competitor_pressure,
        "credible_search_uncertainty": bool(state.credible_search_uncertainty),
        "credible_evidence_uncertainty": bool(state.credible_evidence_uncertainty),
        "frontier_size": len(state.frontier),
        "feasible_actions": [action.as_dict() for action in feasible_actions],
        "chosen_action": chosen_action.as_dict() if chosen_action is not None else None,
        "best_rejected_action": best_rejected_action,
    }


def run_controller(
    *,
    initial_frontier: Sequence[Mapping[str, Any]],
    dccs: DCCSResult,
    fragility: FragilityResult,
    winner_id: str,
    certificate_value: float,
    certificate_map: Mapping[str, float] | None = None,
    selected_route_id: str | None = None,
    ambiguity_context: Mapping[str, Any] | None = None,
    config: VOIConfig | None = None,
    hooks: VOIActionHooks | None = None,
) -> VOIStopCertificate:
    # The stop logic is explicit and auditable: certify if the current route
    # clears the threshold, otherwise act only while some feasible action has
    # positive value above the configured stop threshold.
    cfg = config or VOIConfig()
    certificate_payload = dict(certificate_map or {winner_id: float(certificate_value)})
    state = _normalize_state(VOIControllerState(
        iteration_index=0,
        frontier=list(initial_frontier),
        certificate=certificate_payload,
        winner_id=winner_id,
        selected_route_id=selected_route_id or winner_id,
        remaining_search_budget=max(0, int(cfg.search_budget)),
        remaining_evidence_budget=max(0, int(cfg.evidence_budget)),
        action_trace=[],
        state_trace=[],
        active_evidence_families=[
            str(item.get("family", "")).strip()
            for item in fragility.value_of_refresh.get("ranking", [])
            if isinstance(item, Mapping) and str(item.get("family", "")).strip()
        ],
        refreshed_evidence_families=[],
        stochastic_enabled=bool((ambiguity_context or {}).get("stochastic_enabled", True)),
        ambiguity_context=dict(ambiguity_context or {}),
    ), config=cfg)
    best_rejected_action: dict[str, Any] | None = None

    while state.iteration_index < cfg.max_iterations:
        state = enrich_controller_state_for_actioning(state, dccs=dccs, fragility=fragility, config=cfg)
        current_certificate = _as_float(state.certificate.get(state.winner_id, certificate_value))
        strong_certificate = current_certificate >= cfg.certificate_threshold
        credible_uncertainty = _credible_search_uncertainty(
            state,
            config=cfg,
            current_certificate=current_certificate,
        )
        credible_evidence = _credible_evidence_uncertainty(
            state,
            fragility=fragility,
            config=cfg,
            current_certificate=current_certificate,
        )
        state = replace(
            state,
            credible_search_uncertainty=bool(credible_uncertainty),
            credible_evidence_uncertainty=bool(credible_evidence),
        )
        if strong_certificate and (
            (
                state.search_completeness_score >= cfg.search_completeness_threshold
                or not credible_uncertainty
            )
            and not credible_evidence
        ):
            return VOIStopCertificate(
                final_winner_route_id=state.winner_id,
                final_winner_objective_vector=_winner_objective_vector(state),
                final_strict_frontier_size=len(state.frontier),
                certificate_value=current_certificate,
                certified=True,
                search_budget_used=cfg.search_budget - state.remaining_search_budget,
                search_budget_remaining=state.remaining_search_budget,
                evidence_budget_used=cfg.evidence_budget - state.remaining_evidence_budget,
                evidence_budget_remaining=state.remaining_evidence_budget,
                stop_reason="certified",
                action_trace=list(state.action_trace),
                state_trace=list(state.state_trace),
                best_rejected_action=best_rejected_action,
                ambiguity_summary=_ambiguity_summary(fragility, state.winner_id),
                iteration_count=state.iteration_index,
                controller_state=state.as_dict(),
            )
        if state.remaining_search_budget <= 0 and state.remaining_evidence_budget <= 0:
            return VOIStopCertificate(
                final_winner_route_id=state.winner_id,
                final_winner_objective_vector=_winner_objective_vector(state),
                final_strict_frontier_size=len(state.frontier),
                certificate_value=current_certificate,
                certified=False,
                search_budget_used=cfg.search_budget - state.remaining_search_budget,
                search_budget_remaining=state.remaining_search_budget,
                evidence_budget_used=cfg.evidence_budget - state.remaining_evidence_budget,
                evidence_budget_remaining=state.remaining_evidence_budget,
                stop_reason="budget_exhausted",
                action_trace=list(state.action_trace),
                state_trace=list(state.state_trace),
                best_rejected_action=best_rejected_action,
                ambiguity_summary=_ambiguity_summary(fragility, state.winner_id),
                iteration_count=state.iteration_index,
                controller_state=state.as_dict(),
            )

        actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=cfg)
        feasible_actions = [action for action in actions if action.kind != "stop"]
        if not feasible_actions:
            return VOIStopCertificate(
                final_winner_route_id=state.winner_id,
                final_winner_objective_vector=_winner_objective_vector(state),
                final_strict_frontier_size=len(state.frontier),
                certificate_value=current_certificate,
                certified=False,
                search_budget_used=cfg.search_budget - state.remaining_search_budget,
                search_budget_remaining=state.remaining_search_budget,
                evidence_budget_used=cfg.evidence_budget - state.remaining_evidence_budget,
                evidence_budget_remaining=state.remaining_evidence_budget,
                stop_reason=(
                    "search_incomplete_no_action_worth_it"
                    if state.search_completeness_score < cfg.search_completeness_threshold
                    else "no_action_worth_it"
                ),
                action_trace=list(state.action_trace),
                state_trace=list(state.state_trace),
                best_rejected_action=best_rejected_action,
                ambiguity_summary=_ambiguity_summary(fragility, state.winner_id),
                iteration_count=state.iteration_index,
                controller_state=state.as_dict(),
            )
        top_action = feasible_actions[0]
        if top_action.q_score < cfg.stop_threshold:
            best_rejected_action = top_action.as_dict()
            return VOIStopCertificate(
                final_winner_route_id=state.winner_id,
                final_winner_objective_vector=_winner_objective_vector(state),
                final_strict_frontier_size=len(state.frontier),
                certificate_value=current_certificate,
                certified=False,
                search_budget_used=cfg.search_budget - state.remaining_search_budget,
                search_budget_remaining=state.remaining_search_budget,
                evidence_budget_used=cfg.evidence_budget - state.remaining_evidence_budget,
                evidence_budget_remaining=state.remaining_evidence_budget,
                stop_reason=(
                    "search_incomplete_no_action_worth_it"
                    if state.search_completeness_score < cfg.search_completeness_threshold
                    else "no_action_worth_it"
                ),
                action_trace=list(state.action_trace),
                state_trace=list(state.state_trace),
                best_rejected_action=best_rejected_action,
                ambiguity_summary=_ambiguity_summary(fragility, state.winner_id),
                iteration_count=state.iteration_index,
                controller_state=state.as_dict(),
            )
        if hooks is None and top_action.kind != "stop":
            failed_best_rejected = feasible_actions[1].as_dict() if len(feasible_actions) > 1 else None
            failure_trace_entry = {
                "iteration": state.iteration_index,
                "feasible_actions": [action.as_dict() for action in actions],
                "chosen_action": top_action.as_dict(),
                "certificate_value": current_certificate,
                "remaining_search_budget": state.remaining_search_budget,
                "remaining_evidence_budget": state.remaining_evidence_budget,
                "best_rejected_action": failed_best_rejected,
            }
            failure_state_snapshot = _state_snapshot(
                state,
                current_certificate=current_certificate,
                feasible_actions=actions,
                chosen_action=top_action,
                best_rejected_action=failed_best_rejected,
            )
            return VOIStopCertificate(
                final_winner_route_id=state.winner_id,
                final_winner_objective_vector=_winner_objective_vector(state),
                final_strict_frontier_size=len(state.frontier),
                certificate_value=current_certificate,
                certified=False,
                search_budget_used=cfg.search_budget - state.remaining_search_budget,
                search_budget_remaining=state.remaining_search_budget,
                evidence_budget_used=cfg.evidence_budget - state.remaining_evidence_budget,
                evidence_budget_remaining=state.remaining_evidence_budget,
                stop_reason="error_missing_action_hooks",
                action_trace=[*state.action_trace, failure_trace_entry],
                state_trace=[*state.state_trace, failure_state_snapshot],
                best_rejected_action=failed_best_rejected,
                ambiguity_summary=_ambiguity_summary(fragility, state.winner_id),
                iteration_count=state.iteration_index,
                controller_state=state.as_dict(),
            )

        best_rejected_action = feasible_actions[1].as_dict() if len(feasible_actions) > 1 else None
        previous_winner_id = state.winner_id
        previous_frontier_ids = {_route_id(route) for route in state.frontier}
        previous_certificate = current_certificate
        trace_entry = {
            "iteration": state.iteration_index,
            "feasible_actions": [action.as_dict() for action in actions],
            "chosen_action": top_action.as_dict(),
            "certificate_value": current_certificate,
            "remaining_search_budget": state.remaining_search_budget,
            "remaining_evidence_budget": state.remaining_evidence_budget,
            "best_rejected_action": best_rejected_action,
        }
        state = replace(
            state,
            action_trace=[*state.action_trace, trace_entry],
            state_trace=[
                *state.state_trace,
                _state_snapshot(
                    state,
                    current_certificate=current_certificate,
                    feasible_actions=actions,
                    chosen_action=top_action,
                    best_rejected_action=best_rejected_action,
                ),
            ],
        )
        state = _normalize_state(_apply_action(state, top_action, hooks=hooks), config=cfg)
        post_certificate = _as_float(state.certificate.get(state.winner_id, certificate_value))
        post_frontier_ids = {_route_id(route) for route in state.frontier}
        if (
            post_certificate >= cfg.certificate_threshold
            and state.winner_id == previous_winner_id
            and post_frontier_ids == previous_frontier_ids
            and post_certificate <= (previous_certificate + 1e-6)
        ):
            break
        if top_action.kind == "stop":
            break

        if hooks is None:
            continue

    current_certificate = _as_float(state.certificate.get(state.winner_id, certificate_value))
    if current_certificate >= cfg.certificate_threshold:
        stop_reason = "certified"
    elif state.remaining_search_budget <= 0 and state.remaining_evidence_budget <= 0:
        stop_reason = "budget_exhausted"
    else:
        stop_reason = "iteration_cap_reached"
    return VOIStopCertificate(
        final_winner_route_id=state.winner_id,
        final_winner_objective_vector=_winner_objective_vector(state),
        final_strict_frontier_size=len(state.frontier),
        certificate_value=current_certificate,
        certified=current_certificate >= cfg.certificate_threshold,
        search_budget_used=cfg.search_budget - state.remaining_search_budget,
        search_budget_remaining=state.remaining_search_budget,
        evidence_budget_used=cfg.evidence_budget - state.remaining_evidence_budget,
        evidence_budget_remaining=state.remaining_evidence_budget,
        stop_reason=stop_reason,
        action_trace=list(state.action_trace),
        state_trace=list(state.state_trace),
        best_rejected_action=best_rejected_action,
        ambiguity_summary=_ambiguity_summary(fragility, state.winner_id),
        iteration_count=state.iteration_index,
        controller_state=state.as_dict(),
    )
