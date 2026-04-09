from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .decision_critical import DCCSResult, DCCSCandidateRecord
from .evidence_certification import FragilityResult
from .replay_oracle import (
    ActionReplayRecord,
    ActionValueEstimate,
    ReplayOracleSummary,
    build_action_replay_record as _build_replay_action_record,
    build_action_value_estimate as _build_replay_action_value_estimate,
    replay_oracle_summary_from_trace_rows,
)

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


def _recent_certificate_stalled_search_progress_count(
    state: "VOIControllerState",
    *,
    max_depth: int = 3,
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
        if realized_certificate_delta > 1e-9:
            break
        realized_frontier_gain = max(0.0, _as_float(raw_entry.get("realized_frontier_gain")))
        realized_selected_route_improvement = max(
            0.0,
            _as_float(raw_entry.get("realized_selected_route_improvement")),
        )
        realized_runner_up_gap_delta = max(0.0, _as_float(raw_entry.get("realized_runner_up_gap_delta")))
        realized_productive = bool(raw_entry.get("realized_productive", False))
        search_only_progress = bool(
            realized_productive
            and realized_selected_route_improvement <= 1e-9
            and (
                realized_frontier_gain > 0.0
                or realized_runner_up_gap_delta > 1e-9
            )
        )
        if not search_only_progress:
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


def _trace_has_prior_meaningful_evidence_certificate_lift(
    state: "VOIControllerState",
    *,
    minimum_lift: float = 0.05,
) -> bool:
    trace = state.action_trace if isinstance(state.action_trace, Sequence) else []
    lift_threshold = max(1e-9, _as_float(minimum_lift))
    for raw_entry in trace:
        if not isinstance(raw_entry, Mapping):
            continue
        chosen_action = raw_entry.get("chosen_action")
        kind = ""
        if isinstance(chosen_action, Mapping):
            kind = str(chosen_action.get("kind", "")).strip()
        if not kind:
            kind = str(raw_entry.get("kind", "")).strip()
        if kind not in {"refresh_top1_vor", "increase_stochastic_samples"}:
            continue
        realized_certificate_delta = max(0.0, _as_float(raw_entry.get("realized_certificate_delta")))
        realized_evidence_uncertainty_delta = _as_float(raw_entry.get("realized_evidence_uncertainty_delta"))
        realized_productive = bool(raw_entry.get("realized_productive", False))
        if (
            realized_certificate_delta >= lift_threshold
            and (realized_productive or realized_evidence_uncertainty_delta < -1e-9)
        ):
            return True
    return False


def _recent_no_gain_evidence_discovery_bridge(
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
        metadata: Mapping[str, object] = {}
        if isinstance(chosen_action, Mapping):
            kind = str(chosen_action.get("kind", "")).strip()
            metadata = chosen_action.get("metadata") if isinstance(chosen_action.get("metadata"), Mapping) else {}
        if not kind:
            kind = str(raw_entry.get("kind", "")).strip()
        if kind not in eligible_kinds:
            break
        if not bool(metadata.get("evidence_discovery_bridge")):
            return False
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


def _recent_harmful_evidence_certificate_drift(
    state: "VOIControllerState",
    *,
    minimum_certificate_drop: float = 0.01,
) -> bool:
    trace = state.action_trace if isinstance(state.action_trace, Sequence) else []
    if not trace:
        return False
    raw_entry = trace[-1]
    if not isinstance(raw_entry, Mapping):
        return False
    chosen_action = raw_entry.get("chosen_action")
    kind = ""
    if isinstance(chosen_action, Mapping):
        kind = str(chosen_action.get("kind", "")).strip()
    if not kind:
        kind = str(raw_entry.get("kind", "")).strip()
    if kind not in {"refresh_top1_vor", "increase_stochastic_samples"}:
        return False
    realized_certificate_delta = _as_float(raw_entry.get("realized_certificate_delta"))
    if realized_certificate_delta > -max(1e-9, _as_float(minimum_certificate_drop)):
        return False
    realized_frontier_gain = max(0.0, _as_float(raw_entry.get("realized_frontier_gain")))
    if realized_frontier_gain > 0.0:
        return False
    if bool(raw_entry.get("realized_selected_route_changed", False)):
        return False
    realized_selected_route_improvement = max(
        0.0,
        _as_float(raw_entry.get("realized_selected_route_improvement")),
    )
    if realized_selected_route_improvement > 1e-9:
        return False
    return True


def _recent_productive_refine_route_change(
    state: "VOIControllerState",
    *,
    max_depth: int = 1,
) -> bool:
    trace = state.action_trace if isinstance(state.action_trace, Sequence) else []
    inspected = 0
    for raw_entry in reversed(trace):
        if inspected >= max_depth:
            break
        if not isinstance(raw_entry, Mapping):
            continue
        chosen_action = raw_entry.get("chosen_action")
        kind = ""
        if isinstance(chosen_action, Mapping):
            kind = str(chosen_action.get("kind", "")).strip()
        if not kind:
            kind = str(raw_entry.get("kind", "")).strip()
        if kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        if not bool(raw_entry.get("realized_productive", False)):
            return False
        if not bool(raw_entry.get("realized_selected_route_changed", False)):
            return False
        realized_frontier_gain = max(0.0, _as_float(raw_entry.get("realized_frontier_gain")))
        if realized_frontier_gain <= 0.0:
            return False
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


def _sampler_requested_refc_world_count(state: "VOIControllerState") -> float:
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    sampled_requested_world_count = _as_float(context.get("refc_sampler_requested_world_count"))
    if sampled_requested_world_count > 0.0:
        return sampled_requested_world_count
    return _requested_refc_world_count(state)


def _resample_shortfall_available(state: "VOIControllerState") -> bool:
    actual_world_count = _actual_refc_world_count(state)
    requested_world_count = (
        _sampler_requested_refc_world_count(state)
        if state.stochastic_enabled
        else _requested_refc_world_count(state)
    )
    return requested_world_count > actual_world_count + 1e-9


def _resample_shortfall_ratio(state: "VOIControllerState") -> float:
    actual_world_count = _actual_refc_world_count(state)
    requested_world_count = (
        _sampler_requested_refc_world_count(state)
        if state.stochastic_enabled
        else _requested_refc_world_count(state)
    )
    if requested_world_count <= 0.0:
        return 1.0 if actual_world_count <= 0.0 else 0.0
    return max(0.0, requested_world_count - actual_world_count) / requested_world_count


def _cert_world_shortfall_available(state: "VOIControllerState") -> bool:
    actual_world_count = _actual_refc_world_count(state)
    requested_world_count = _requested_refc_world_count(state)
    return requested_world_count > actual_world_count + 1e-9


def _cert_world_shortfall_ratio(state: "VOIControllerState") -> float:
    actual_world_count = _actual_refc_world_count(state)
    requested_world_count = _requested_refc_world_count(state)
    if requested_world_count <= 0.0:
        return 1.0 if actual_world_count <= 0.0 else 0.0
    return max(0.0, requested_world_count - actual_world_count) / requested_world_count


def _resample_world_expandable(state: "VOIControllerState") -> bool:
    actual_world_count = _actual_refc_world_count(state)
    requested_world_count = _requested_refc_world_count(state)
    if actual_world_count <= 0.0 or requested_world_count <= 0.0:
        return True
    shortfall = max(0.0, requested_world_count - actual_world_count)
    tolerance = max(2.0, 0.05 * requested_world_count)
    return shortfall < tolerance


def _single_frontier_structural_cap_gap(
    state: "VOIControllerState",
) -> tuple[float, float, float, float]:
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    empirical_baseline = _as_float(context.get("empirical_baseline_certificate"), float("nan"))
    controller_baseline = _as_float(
        context.get("controller_baseline_certificate"),
        _certificate_value_from_map(state.certificate, state.winner_id),
    )
    actual_world_count = _actual_refc_world_count(state)
    requested_world_count = _requested_refc_world_count(state)
    if requested_world_count <= 0.0:
        shortfall_ratio = 1.0 if actual_world_count <= 0.0 else 0.0
    else:
        shortfall_ratio = max(0.0, requested_world_count - actual_world_count) / requested_world_count
    return (
        empirical_baseline,
        controller_baseline,
        actual_world_count,
        _clamp01(shortfall_ratio),
    )


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
    replay_oracle_summary: dict[str, Any] | None = None
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


def _frontier_candidate_ids(state: VOIControllerState | None) -> set[str]:
    if state is None:
        return set()
    ids: set[str] = set()
    for route in state.frontier:
        if not isinstance(route, Mapping):
            continue
        candidate_id = str(route.get("candidate_id", "")).strip()
        if candidate_id:
            ids.add(candidate_id)
    return ids


def _dedupe_pending_dccs_candidates(
    records: Sequence[DCCSCandidateRecord],
) -> list[DCCSCandidateRecord]:
    deduped: dict[str, DCCSCandidateRecord] = {}
    for record in records:
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


def _pending_dccs_candidates(
    dccs: DCCSResult,
    *,
    state: VOIControllerState | None = None,
) -> list[DCCSCandidateRecord]:
    selected_ids = {record.candidate_id for record in dccs.selected}
    frontier_candidate_ids = _frontier_candidate_ids(state)
    candidate_pool = [
        record
        for record in [*dccs.skipped, *dccs.candidate_ledger]
        if record.decision_reason not in {"duplicate_signature", "duplicate_corridor_bootstrap"}
    ]
    ranked_pool = [
        record
        for record in candidate_pool
        if record.candidate_id not in selected_ids
        and record.candidate_id not in frontier_candidate_ids
    ]
    if ranked_pool:
        return _dedupe_pending_dccs_candidates(ranked_pool)
    if not frontier_candidate_ids:
        return []
    fallback_pool = [
        record
        for record in candidate_pool
        if record.candidate_id not in frontier_candidate_ids
    ]
    return _dedupe_pending_dccs_candidates(fallback_pool)


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
    controller_family = str(
        fragility.value_of_refresh.get("top_refresh_family_controller", "")
    ).strip()
    if controller_family:
        return controller_family
    controller_ranking = fragility.value_of_refresh.get("controller_ranking", [])
    if isinstance(controller_ranking, list) and controller_ranking:
        first_controller = controller_ranking[0]
        if isinstance(first_controller, Mapping) and _as_float(first_controller.get("controller_score")) > 0.0:
            family = str(first_controller.get("family", "")).strip()
            if family:
                return family
    ranking = fragility.value_of_refresh.get("ranking", [])
    if not isinstance(ranking, list) or not ranking:
        return None
    first = ranking[0]
    if not isinstance(first, Mapping):
        return None
    family = str(first.get("family", "")).strip()
    return family or None


def _controller_refresh_bridge_stats(
    fragility: FragilityResult | Any | None,
) -> tuple[bool, bool, float]:
    value_of_refresh = (
        fragility.value_of_refresh
        if fragility is not None and isinstance(getattr(fragility, "value_of_refresh", None), Mapping)
        else {}
    )
    controller_ranking_basis = str(value_of_refresh.get("controller_ranking_basis", "")).strip()
    controller_ranking = value_of_refresh.get("controller_ranking", [])
    controller_top_family = str(value_of_refresh.get("top_refresh_family_controller", "")).strip()
    controller_top_gain = _as_float(value_of_refresh.get("top_refresh_gain_controller"), float("nan"))
    if isinstance(controller_ranking, list) and controller_ranking:
        first = controller_ranking[0] if isinstance(controller_ranking[0], Mapping) else {}
        if not controller_top_family:
            controller_top_family = str(first.get("family", "")).strip()
        if not math.isfinite(controller_top_gain):
            controller_top_gain = _as_float(first.get("controller_score"), float("nan"))
    if not math.isfinite(controller_top_gain):
        controller_top_gain = 0.0
    empirical_top_family = str(value_of_refresh.get("top_refresh_family", "")).strip()
    empirical_top_gain = max(0.0, _as_float(value_of_refresh.get("top_refresh_gain")))
    fallback_activated = controller_ranking_basis == "raw_refresh_gain_fallback"
    zero_to_nonzero_signal_upgrade = bool(
        fallback_activated
        and controller_top_gain > 1e-9
        and empirical_top_gain <= 1e-9
    )
    disagreement = bool(
        fallback_activated
        and (
            controller_top_family != empirical_top_family
            or zero_to_nonzero_signal_upgrade
            or controller_top_gain > empirical_top_gain + 1e-9
        )
    )
    return fallback_activated, disagreement, max(0.0, controller_top_gain)


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
    pending = _pending_dccs_candidates(dccs, state=state)
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
    threshold = _as_float(config.certificate_threshold)
    frontier_recall = _clamp01(state.frontier_recall_at_budget)
    corridor_family_recall = _clamp01(state.corridor_family_recall)
    search_completeness_gap = max(0.0, _as_float(state.search_completeness_gap))
    pending_challenger_mass = _clamp01(state.pending_challenger_mass)
    pending_flip_probability = _clamp01(state.best_pending_flip_probability)
    predicted_frontier = max(0.0, _as_float(action.predicted_delta_frontier))
    certified_frontier_completion_bridge = bool(
        recent_no_gain_refine_streak <= 0
        and state.remaining_search_budget > 0
        and current_certificate <= min(1.0, threshold + 0.18)
        and _refine_action_has_genuine_novel_search_promise(action, state=state)
        and (
            max(0.0, 1.0 - frontier_recall) >= 0.24
            or max(0.0, 1.0 - corridor_family_recall) >= 0.20
            or search_completeness_gap >= 0.16
        )
        and (
            predicted_frontier >= 0.18
            or objective_gap >= 0.18
        )
        and (
            pending_challenger_mass >= 0.28
            or pending_flip_probability >= 0.35
        )
    )
    if certified_frontier_completion_bridge:
        metadata = dict(action.metadata)
        metadata["certified_supported_hard_case_frontier_bridge"] = True
        return replace(action, metadata=metadata)
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
    value_of_refresh = (
        fragility.value_of_refresh
        if isinstance(fragility.value_of_refresh, Mapping)
        else {}
    )
    structured_refresh_signal = False
    empirical_refresh_certificate_uplift: float | None = None
    empirical_refresh_certificate_delta: float | None = None
    empirical_refresh_certificate_drop: float | None = None
    family_certificate: float | None = None
    empirical_baseline_certificate: float | None = None
    controller_baseline_certificate: float | None = None
    structural_cap_only = False
    per_family_certificate = value_of_refresh.get("per_family_certificate")
    if isinstance(per_family_certificate, Mapping):
        candidate_family_certificate = _as_float(per_family_certificate.get(family), float("nan"))
        baseline_certificate = _as_float(value_of_refresh.get("baseline_certificate"), float("nan"))
        empirical_baseline_value = _as_float(
            value_of_refresh.get("empirical_baseline_certificate"),
            baseline_certificate,
        )
        controller_baseline_value = _as_float(
            value_of_refresh.get("controller_baseline_certificate"),
            current_certificate,
        )
        if math.isfinite(candidate_family_certificate) and math.isfinite(empirical_baseline_value):
            structured_refresh_signal = True
            family_certificate = candidate_family_certificate
            empirical_baseline_certificate = empirical_baseline_value
            if math.isfinite(controller_baseline_value):
                controller_baseline_certificate = controller_baseline_value
            empirical_refresh_certificate_delta = (
                candidate_family_certificate - empirical_baseline_value
            )
            empirical_refresh_certificate_uplift = max(
                0.0,
                float(empirical_refresh_certificate_delta),
            )
            empirical_refresh_certificate_drop = max(
                0.0,
                -float(empirical_refresh_certificate_delta),
            )
            structural_cap_only = bool(
                value_of_refresh.get("single_frontier_certificate_cap_applied")
                and controller_baseline_certificate is not None
                and empirical_baseline_value > controller_baseline_certificate + 1e-9
                and empirical_refresh_certificate_uplift <= 1e-9
            )
    if structured_refresh_signal:
        uncertainty_only_scale = 0.10 if current_certificate < _as_float(config.certificate_threshold) else 0.02
        if structural_cap_only:
            uncertainty_only_scale = 0.0
        predicted_delta_certificate = max(
            0.0,
            float(empirical_refresh_certificate_uplift or 0.0)
            + (
                evidence_stability_factor
                * uncertainty_only_scale
                * ((0.60 * route_fragility) + (0.20 * normalized_pressure))
            ),
        )
        predicted_delta_margin = max(
            0.0,
            (0.75 * float(empirical_refresh_certificate_uplift or 0.0))
            + (
                evidence_stability_factor
                * uncertainty_only_scale
                * ((0.35 * route_fragility) + (0.15 * normalized_pressure))
            ),
        )
    else:
        predicted_delta_certificate = max(
            0.0,
            evidence_stability_factor * (vor_gain + (0.35 * route_fragility) + (0.10 * normalized_pressure)),
        )
        predicted_delta_margin = max(
            0.0,
            evidence_stability_factor * ((0.60 * vor_gain) + (0.25 * route_fragility) + (0.10 * normalized_pressure)),
        )
    return VOIAction(
        action_id=f"refresh:{family}",
        kind="refresh_top1_vor",
        target=family,
        cost_evidence=1,
        predicted_delta_certificate=predicted_delta_certificate,
        predicted_delta_margin=predicted_delta_margin,
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
            "structured_refresh_signal": structured_refresh_signal,
            "empirical_refresh_certificate_uplift": (
                round(empirical_refresh_certificate_uplift, 6)
                if empirical_refresh_certificate_uplift is not None
                else None
            ),
            "empirical_refresh_certificate_delta": (
                round(empirical_refresh_certificate_delta, 6)
                if empirical_refresh_certificate_delta is not None
                else None
            ),
            "empirical_refresh_certificate_drop": (
                round(empirical_refresh_certificate_drop, 6)
                if empirical_refresh_certificate_drop is not None
                else None
            ),
            "family_certificate": round(family_certificate, 6) if family_certificate is not None else None,
            "empirical_baseline_certificate": (
                round(empirical_baseline_certificate, 6)
                if empirical_baseline_certificate is not None
                else None
            ),
            "controller_baseline_certificate": (
                round(controller_baseline_certificate, 6)
                if controller_baseline_certificate is not None
                else None
            ),
            "structural_cap_only": structural_cap_only,
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


def _allow_certified_negative_refresh_revelation(
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    signed_refresh_delta: float,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
) -> bool:
    if current_certificate < _as_float(config.certificate_threshold):
        return False
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    saturation_certificate = min(
        1.0,
        max(0.95, _as_float(config.certificate_threshold) + 0.20),
    )
    if not math.isfinite(signed_refresh_delta) or signed_refresh_delta >= -1e-9:
        return False
    if state.remaining_search_budget <= 0 or state.remaining_evidence_budget <= 0:
        return False
    if not evidence_uncertainty or not supported_fragility_uncertainty:
        return False
    if not _support_rich_ambiguity_window(state):
        return False
    if _trace_has_prior_evidence_action_attempt(state):
        return False
    strong_winner_side_signal = bool(
        max(0.0, _as_float(state.top_refresh_gain)) >= 0.10
        or _clamp01(state.top_fragility_mass) >= 0.15
        or _clamp01(state.competitor_pressure) >= 0.75
    )
    if not strong_winner_side_signal:
        return False
    supported_search_pressure = bool(
        max(0.0, _as_float(state.search_completeness_gap)) >= 0.12
        or _clamp01(state.pending_challenger_mass) >= 0.35
        or _clamp01(state.best_pending_flip_probability) >= 0.40
        or max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)) >= 0.20
        or (
            len(state.frontier) >= 2
            and _clamp01(state.near_tie_mass) >= max(_as_float(config.near_tie_threshold), 0.05)
        )
    )
    if current_certificate >= saturation_certificate - 1e-9:
        tie_like_decision_ambiguity = _has_strong_saturated_reveal_ambiguity(
            state,
            config=config,
        )
        saturation_reveal_pressure = bool(
            max(0.0, _as_float(state.search_completeness_gap)) >= 0.20
            or _clamp01(state.pending_challenger_mass) >= 0.45
            or _clamp01(state.best_pending_flip_probability) >= 0.90
            or max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)) >= 0.25
        )
        return bool(
            strong_winner_side_signal
            and supported_search_pressure
            and saturation_reveal_pressure
            and tie_like_decision_ambiguity
            and max(0.0, _as_float(state.top_refresh_gain)) >= 0.25
            and abs(signed_refresh_delta) >= 0.08
        )
    return supported_search_pressure


def _has_strong_saturated_reveal_ambiguity(
    state: "VOIControllerState",
    *,
    config: VOIConfig,
) -> bool:
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    nominal_margin_proxy = _as_float(ambiguity_context.get("od_nominal_margin_proxy"), float("nan"))
    live_near_tie = bool(
        _clamp01(state.near_tie_mass) >= max(_as_float(config.near_tie_threshold), 0.05)
    )
    strong_structural_ambiguity = bool(
        (math.isfinite(nominal_margin_proxy) and nominal_margin_proxy <= 0.18)
        or _clamp01(ambiguity_context.get("od_objective_spread")) >= 0.25
        or _clamp01(ambiguity_context.get("od_ambiguity_margin_pressure")) >= 0.85
        or _clamp01(ambiguity_context.get("od_ambiguity_spread_pressure")) >= 0.25
    )
    return bool(live_near_tie or strong_structural_ambiguity)


def _has_tie_like_decision_ambiguity(state: "VOIControllerState") -> bool:
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    nominal_margin_proxy = _as_float(ambiguity_context.get("od_nominal_margin_proxy"), float("nan"))
    low_nominal_margin_proxy = bool(
        math.isfinite(nominal_margin_proxy) and nominal_margin_proxy <= 0.40
    )
    return bool(
        _clamp01(state.near_tie_mass) >= 0.10
        or _clamp01(ambiguity_context.get("od_ambiguity_margin_pressure")) >= 0.20
        or _clamp01(ambiguity_context.get("od_ambiguity_spread_pressure")) >= 0.20
        or _clamp01(ambiguity_context.get("od_objective_spread")) >= 0.08
        or low_nominal_margin_proxy
    )


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
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    single_frontier_structural_cap = bool(
        len(state.frontier) <= 1
        and ambiguity_context.get("single_frontier_certificate_cap_applied")
        and _as_float(ambiguity_context.get("empirical_baseline_certificate"), float("nan"))
        > _as_float(ambiguity_context.get("controller_baseline_certificate"), float("nan")) + 1e-9
        and max(0.0, _as_float(state.top_refresh_gain)) <= 0.0
        and _clamp01(state.top_fragility_mass) <= 0.01
        and _clamp01(state.competitor_pressure) <= 0.05
    )
    structural_cap_objective_support = bool(
        objective_gap >= 0.05
        or predicted_frontier >= 0.08
        or (objective_gap >= 0.03 and overlap_gain >= 0.35)
    )
    if single_frontier_structural_cap and not structural_cap_objective_support:
        return False
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
    raw_predicted_frontier = max(0.0, _as_float(action.predicted_delta_frontier))
    predicted_frontier = _clamp01(raw_predicted_frontier / 0.10)
    competitor_pressure = _clamp01(state.competitor_pressure)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    single_frontier_structural_cap = bool(
        len(state.frontier) <= 1
        and ambiguity_context.get("single_frontier_certificate_cap_applied")
        and _as_float(ambiguity_context.get("empirical_baseline_certificate"), float("nan"))
        > _as_float(ambiguity_context.get("controller_baseline_certificate"), float("nan")) + 1e-9
        and max(0.0, _as_float(state.top_refresh_gain)) <= 0.0
        and _clamp01(state.top_fragility_mass) <= 0.01
        and _clamp01(state.competitor_pressure) <= 0.05
    )
    structural_cap_objective_support = bool(
        objective_gap >= 0.05
        or raw_predicted_frontier >= 0.08
        or (objective_gap >= 0.03 and competitor_pressure >= 0.20)
    )
    if single_frontier_structural_cap and not structural_cap_objective_support:
        return False
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


def _should_stop_evidence_exhausted_certified_search_tail(
    action: VOIAction | None,
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> bool:
    if action is None or action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return False
    certificate_threshold = _as_float(config.certificate_threshold)
    saturation_certificate = min(
        1.0,
        max(0.95, certificate_threshold + 0.20),
    )
    headroom_capped_post_refresh_tail = bool(
        current_certificate >= (certificate_threshold + 0.05)
        and current_certificate < saturation_certificate
        and bool(action.metadata.get("certificate_headroom_cap_applied"))
        and _trace_has_prior_meaningful_evidence_certificate_lift(state, minimum_lift=0.03)
    )
    if current_certificate < saturation_certificate and not headroom_capped_post_refresh_tail:
        return False
    if state.remaining_evidence_budget > 0:
        return False
    if not _trace_has_prior_evidence_action_attempt(state):
        return False
    if not _support_rich_ambiguity_window(state):
        return False
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.04):
        return False
    mechanism_gap = _clamp01(action.metadata.get("normalized_mechanism_gap"))
    if headroom_capped_post_refresh_tail:
        if mechanism_gap >= 0.45:
            return False
    elif mechanism_gap >= 0.12:
        return False
    if (
        max(0.0, _as_float(action.predicted_delta_certificate)) > 1e-9
        and not headroom_capped_post_refresh_tail
    ):
        return False
    objective_gap = _clamp01(action.metadata.get("normalized_objective_gap"))
    predicted_frontier = max(0.0, _as_float(action.predicted_delta_frontier))
    if objective_gap >= 0.18 or predicted_frontier >= 0.20:
        return False
    q_score = _as_float(action.q_score)
    if headroom_capped_post_refresh_tail:
        if q_score > max(0.16, _as_float(config.stop_threshold) + 0.12):
            return False
    else:
        low_ranked_search_tail = bool(
            q_score <= max(0.12, _as_float(config.stop_threshold) + 0.10)
        )
        if not low_ranked_search_tail:
            return False
    return True


def _should_stop_saturated_certified_low_decision_ambiguity_reopen(
    action: VOIAction | None,
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    certified_frontier_fill_bridge: bool = False,
) -> bool:
    if action is None or action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return False
    saturation_certificate = min(
        1.0,
        max(0.95, _as_float(config.certificate_threshold) + 0.20),
    )
    if current_certificate < saturation_certificate - 1e-9:
        return False
    if certified_frontier_fill_bridge:
        return False
    if state.action_trace:
        return False
    if len(state.frontier) > 2:
        return False
    if _has_tie_like_decision_ambiguity(state):
        return False
    strong_winner_side_signal = bool(
        max(0.0, _as_float(state.top_refresh_gain)) >= 0.25
        or _clamp01(state.top_fragility_mass) >= 0.25
        or _clamp01(state.competitor_pressure) >= 0.75
    )
    if not strong_winner_side_signal:
        return False
    metadata = action.metadata if isinstance(action.metadata, Mapping) else {}
    if not bool(metadata.get("certificate_headroom_cap_applied")):
        return False
    if max(0.0, _as_float(metadata.get("certificate_headroom_remaining"), 0.0)) > 1e-9:
        return False
    if max(0.0, _as_float(action.predicted_delta_certificate)) > 1e-9:
        return False
    objective_gap = _clamp01(metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(action.predicted_delta_frontier))
    if objective_gap >= 0.14 or predicted_frontier >= 0.14:
        return False
    if mechanism_gap >= 0.45 and predicted_frontier >= 0.10:
        return False
    return True


def _should_stop_evidence_exhausted_uncertified_search_tail(
    action: VOIAction | None,
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> bool:
    if action is None or action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return False
    certificate_threshold = _as_float(config.certificate_threshold)
    if current_certificate >= certificate_threshold:
        return False
    if state.remaining_evidence_budget > 0:
        return False
    if state.remaining_search_budget > 2:
        return False
    if len(state.action_trace) < 5:
        return False
    if current_certificate < max(0.65, certificate_threshold - 0.20):
        return False
    if not _trace_has_prior_meaningful_evidence_certificate_lift(state, minimum_lift=0.05):
        return False
    if not _support_rich_ambiguity_window(state):
        return False
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return False
    strong_winner_side_signal = bool(
        max(0.0, _as_float(state.top_refresh_gain)) >= 0.25
        or _clamp01(state.top_fragility_mass) >= 0.25
        or _clamp01(state.competitor_pressure) >= 0.75
    )
    if not strong_winner_side_signal:
        return False
    objective_gap = _clamp01(action.metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(action.metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(action.predicted_delta_frontier))
    corridor_recall = _clamp01(action.metadata.get("corridor_family_recall", state.corridor_family_recall))
    frontier_recall = _clamp01(action.metadata.get("frontier_recall_at_budget", state.frontier_recall_at_budget))
    low_value_terminal_search_tail = bool(
        _as_float(action.q_score) <= max(0.18, _as_float(config.stop_threshold) + 0.04)
        and objective_gap < 0.26
        and mechanism_gap < 0.12
        and predicted_frontier < 0.36
        and corridor_recall <= 0.25
        and frontier_recall <= 0.30
    )
    if low_value_terminal_search_tail:
        return True
    if objective_gap >= 0.30 or predicted_frontier >= 0.30:
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


def _best_refresh_action_entry(actions: Sequence[VOIAction]) -> tuple[int, VOIAction] | None:
    return max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind == "refresh_top1_vor"
        ),
        key=lambda item: (_as_float(item[1].q_score), -item[0]),
        default=None,
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
    certificate_threshold = _as_float(config.certificate_threshold)
    frontier_probe_progress_count = _recent_certificate_stalled_search_progress_count(
        state,
        max_depth=1,
    )
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
    if current_certificate < certificate_threshold:
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
    refresh_entry = _best_refresh_action_entry(actions)
    if refresh_entry is None:
        return list(actions)
    refresh_index, refresh_action = refresh_entry
    refresh_metadata = refresh_action.metadata if isinstance(refresh_action.metadata, Mapping) else {}
    certified_frontier_probe_refresh_recovery = bool(
        frontier_probe_progress_count >= 1
        and _selected_route_uses_cached_direct_fallback(state)
        and current_certificate >= max(certificate_threshold, 0.90)
        and bool(refresh_metadata.get("structured_refresh_signal"))
        and max(0.0, _as_float(refresh_metadata.get("empirical_refresh_certificate_uplift"))) >= 0.05
        and top_refresh_gain >= 0.05
        and top_fragility_mass >= 0.05
    )
    post_harmful_refresh_recovery = bool(
        _recent_harmful_evidence_certificate_drift(state)
        and bool(refresh_metadata.get("structured_refresh_signal"))
        and max(0.0, _as_float(refresh_metadata.get("empirical_refresh_certificate_uplift"))) >= 0.05
    )
    if (
        recent_no_gain_refine_streak <= 0
        and not first_iteration_supported_hard_case
        and not post_harmful_refresh_recovery
        and not certified_frontier_probe_refresh_recovery
    ):
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
        and not post_harmful_refresh_recovery
        and not certified_frontier_probe_refresh_recovery
    ):
        return list(actions)
    if (
        _search_action_shows_strong_decision_movement(best_search_action, state=state)
        and recent_no_gain_refine_streak <= 0
        and not first_iteration_supported_hard_case
        and not post_harmful_refresh_recovery
        and not certified_frontier_probe_refresh_recovery
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
    if post_harmful_refresh_recovery:
        metadata["support_rich_certified_refresh_preference_post_harmful_refresh_recovery"] = True
    if certified_frontier_probe_refresh_recovery:
        metadata["support_rich_certified_refresh_preference_frontier_probe_recovery"] = True
    updated = list(actions)
    updated[refresh_index] = replace(
        refresh_action,
        q_score=refresh_action.q_score + bonus,
        metadata=metadata,
    )
    return updated


def _apply_strong_winner_side_refresh_preference(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
    recent_no_gain_refine_streak: int,
) -> list[VOIAction]:
    if not evidence_uncertainty or not supported_fragility_uncertainty:
        return list(actions)
    if not _support_rich_ambiguity_window(state):
        return list(actions)
    refresh_entry = _best_refresh_action_entry(actions)
    if refresh_entry is None:
        return list(actions)
    refresh_index, refresh_action = refresh_entry
    best_search_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    if best_search_entry is None:
        return list(actions)
    best_search_index, best_search_action = best_search_entry
    certificate_threshold = _as_float(config.certificate_threshold)
    refresh_metadata = refresh_action.metadata if isinstance(refresh_action.metadata, Mapping) else {}
    top_refresh_gain = max(0.0, _as_float(state.top_refresh_gain))
    top_fragility_mass = _clamp01(state.top_fragility_mass)
    competitor_pressure = _clamp01(state.competitor_pressure)
    strong_structured_uncertified_bridge = bool(
        current_certificate < certificate_threshold
        and bool(refresh_metadata.get("structured_refresh_signal"))
        and max(0.0, _as_float(refresh_metadata.get("empirical_refresh_certificate_uplift"))) >= 0.12
        and _clamp01(best_search_action.metadata.get("normalized_objective_gap")) < 0.08
    )
    empirical_uncertified_refresh_bridge = bool(
        current_certificate < certificate_threshold
        and bool(refresh_metadata.get("structured_refresh_signal"))
        and max(0.0, _as_float(refresh_metadata.get("empirical_refresh_certificate_uplift"))) >= 0.05
        and _clamp01(best_search_action.metadata.get("normalized_objective_gap")) < 0.03
        and _clamp01(best_search_action.metadata.get("normalized_mechanism_gap")) < 0.16
    )
    post_route_change_uncertified_refresh_bridge = bool(
        current_certificate < certificate_threshold
        and _recent_productive_refine_route_change(state)
        and bool(refresh_metadata.get("structured_refresh_signal"))
        and max(0.0, _as_float(refresh_metadata.get("empirical_refresh_certificate_uplift"))) >= 0.20
        and _clamp01(state.pending_challenger_mass) >= 0.55
    )
    cached_direct_fallback_refresh_bridge = bool(
        current_certificate < certificate_threshold
        and current_certificate >= 0.60
        and _selected_route_uses_cached_direct_fallback(state)
        and bool(refresh_metadata.get("structured_refresh_signal"))
        and _clamp01(best_search_action.metadata.get("normalized_objective_gap")) < 0.03
        and max(0.0, _as_float(best_search_action.predicted_delta_frontier)) < 0.08
        and top_refresh_gain >= 0.50
        and top_fragility_mass >= 0.50
        and competitor_pressure >= 0.95
    )
    preferred_uncertified_refresh_bridge = bool(
        strong_structured_uncertified_bridge
        or empirical_uncertified_refresh_bridge
        or post_route_change_uncertified_refresh_bridge
        or cached_direct_fallback_refresh_bridge
    )
    if (
        _clamp01(best_search_action.metadata.get("normalized_mechanism_gap")) >= 0.12
        and not preferred_uncertified_refresh_bridge
    ):
        return list(actions)
    if top_refresh_gain < 0.18 or top_fragility_mass < 0.15 or competitor_pressure < 0.75:
        return list(actions)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    hard_case_pressure = max(
        _clamp01(state.ambiguity_pressure),
        _clamp01(ambiguity_context.get("od_hard_case_prior")),
        _clamp01(ambiguity_context.get("ambiguity_budget_prior")),
    )
    if hard_case_pressure < 0.45:
        return list(actions)
    if (
        current_certificate >= (certificate_threshold + 0.10)
        and recent_no_gain_refine_streak <= 0
    ):
        return list(actions)
    q_gap = max(0.0, _as_float(best_search_action.q_score) - _as_float(refresh_action.q_score))
    if q_gap > 0.12 and recent_no_gain_refine_streak <= 0 and not preferred_uncertified_refresh_bridge:
        return list(actions)
    evidence_signature = _clamp01(
        (0.46 * min(1.0, top_refresh_gain / 0.25))
        + (0.32 * top_fragility_mass)
        + (0.22 * competitor_pressure)
    )
    bonus = min(
        0.24,
        q_gap
        + 0.008
        + (0.024 * evidence_signature)
        + (0.010 * hard_case_pressure),
    )
    if bonus <= 0.0:
        return list(actions)
    updated = list(actions)
    refresh_metadata = dict(refresh_metadata)
    refresh_metadata["winner_side_refresh_preference_bonus"] = round(bonus, 6)
    refresh_metadata["winner_side_refresh_preference_applied"] = True
    refresh_metadata["winner_side_refresh_preference_uncertified_bridge"] = bool(
        current_certificate < certificate_threshold
    )
    refresh_metadata["winner_side_refresh_preference_structured_bridge"] = strong_structured_uncertified_bridge
    refresh_metadata["winner_side_refresh_preference_empirical_bridge"] = empirical_uncertified_refresh_bridge
    refresh_metadata["winner_side_refresh_preference_post_route_change_bridge"] = (
        post_route_change_uncertified_refresh_bridge
    )
    refresh_metadata["winner_side_refresh_preference_cached_direct_fallback_bridge"] = (
        cached_direct_fallback_refresh_bridge
    )
    updated[refresh_index] = replace(
        refresh_action,
        q_score=_as_float(refresh_action.q_score) + bonus,
        metadata=refresh_metadata,
    )
    search_discount = min(0.08, 0.015 + (0.040 * evidence_signature))
    if search_discount > 0.0:
        search_metadata = dict(best_search_action.metadata)
        search_metadata["winner_side_refresh_refine_discount"] = round(search_discount, 6)
        search_metadata["winner_side_refresh_refine_discount_applied"] = True
        updated[best_search_index] = replace(
            best_search_action,
            q_score=max(0.0, _as_float(best_search_action.q_score) - search_discount),
            metadata=search_metadata,
        )
    return updated


def _apply_uncertified_evidence_plateau_preference(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
) -> list[VOIAction]:
    if current_certificate >= _as_float(config.certificate_threshold):
        return list(actions)
    if not evidence_uncertainty:
        return list(actions)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    plateau_progress_count = _recent_certificate_stalled_search_progress_count(state)
    recent_no_gain_refine_streak = _recent_no_gain_refine_streak(state, max_depth=1)
    best_search_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    best_evidence_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind in {"refresh_top1_vor", "increase_stochastic_samples"}
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    if best_search_entry is None or best_evidence_entry is None:
        return list(actions)
    best_search_index, best_search_action = best_search_entry
    best_evidence_index, best_evidence_action = best_evidence_entry
    evidence_q = _as_float(best_evidence_action.q_score)
    search_q = _as_float(best_search_action.q_score)
    if evidence_q <= 0.0 or search_q <= evidence_q:
        return list(actions)
    support_rich_window = _support_rich_ambiguity_window(state)
    top_refresh_gain = max(0.0, _as_float(state.top_refresh_gain))
    top_fragility_mass = _clamp01(state.top_fragility_mass)
    competitor_pressure = _clamp01(state.competitor_pressure)
    evidence_metadata = (
        best_evidence_action.metadata
        if isinstance(best_evidence_action.metadata, Mapping)
        else {}
    )
    near_tie_mass = max(
        _clamp01(state.near_tie_mass),
        _clamp01(evidence_metadata.get("near_tie_mass")),
    )
    objective_spread = _clamp01(ambiguity_context.get("od_objective_spread"))
    budget_prior = _clamp01(ambiguity_context.get("ambiguity_budget_prior"))
    objective_gap = _clamp01(
        (best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}).get(
            "normalized_objective_gap"
        )
    )
    predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
    first_iteration_near_tie_plateau = bool(
        plateau_progress_count <= 0
        and near_tie_mass >= 0.65
        and objective_spread <= 0.02
        and objective_gap < 0.05
        and predicted_frontier < 0.10
        and top_refresh_gain >= 0.18
        and top_fragility_mass >= 0.15
        and competitor_pressure >= 0.75
        and budget_prior <= 0.10
    )
    direct_winner_side_signal = bool(
        top_refresh_gain >= 0.18
        and top_fragility_mass >= 0.15
        and competitor_pressure >= 0.75
    )
    direct_fallback_probe_stall_count = max(
        plateau_progress_count,
        recent_no_gain_refine_streak,
    )
    direct_fallback_frontier_probe_plateau = bool(
        direct_fallback_probe_stall_count >= 1
        and _selected_route_uses_cached_direct_fallback(state)
        and support_rich_window
        and supported_fragility_uncertainty
        and state.remaining_search_budget <= 1
        and len(state.frontier) >= 3
        and best_evidence_action.kind == "increase_stochastic_samples"
        and max(0.0, _as_float(best_evidence_action.predicted_delta_certificate)) >= 0.06
        and direct_winner_side_signal
        and _clamp01(state.pending_challenger_mass) >= 0.55
        and _clamp01(state.best_pending_flip_probability) >= 0.95
        and max(0.0, _as_float(state.certificate_margin)) <= 0.08
    )
    structural_search_incomplete = bool(
        max(0.0, _as_float(state.search_completeness_gap)) >= 0.22
        and _clamp01(state.pending_challenger_mass) >= 0.60
        and _clamp01(state.best_pending_flip_probability) >= 0.95
        and (
            _clamp01(state.corridor_family_recall) < 0.50
            or _clamp01(state.frontier_recall_at_budget) < 0.40
        )
    )
    frontier_completion_search_bridge = bool(
        plateau_progress_count >= 2
        and state.remaining_search_budget > 0
        and structural_search_incomplete
        and _refine_action_has_genuine_novel_search_promise(best_search_action, state=state)
        and (
            predicted_frontier >= 0.16
            or objective_gap >= 0.14
        )
    )
    near_tie_frontier_completion_resample_bridge = bool(
        frontier_completion_search_bridge
        and best_evidence_action.kind == "increase_stochastic_samples"
        and max(0.0, _as_float(best_evidence_action.predicted_delta_certificate)) >= 0.10
        and top_fragility_mass >= 0.45
        and competitor_pressure >= 0.90
        and near_tie_mass >= 0.20
        and max(0.0, _as_float(state.certificate_margin)) <= 0.05
    )
    if first_iteration_near_tie_plateau and direct_winner_side_signal and structural_search_incomplete:
        return list(actions)
    # Keep search in control when repeated stalled refinements still expose
    # credible frontier-completion upside on a support-rich hard row.
    if (
        frontier_completion_search_bridge
        and not near_tie_frontier_completion_resample_bridge
        and not direct_fallback_frontier_probe_plateau
    ):
        return list(actions)
    if not supported_fragility_uncertainty and not first_iteration_near_tie_plateau:
        return list(actions)
    if not support_rich_window and not first_iteration_near_tie_plateau:
        return list(actions)
    if (
        plateau_progress_count < 2
        and not first_iteration_near_tie_plateau
        and not direct_fallback_frontier_probe_plateau
    ):
        return list(actions)
    if top_refresh_gain < 0.18 or top_fragility_mass < 0.15 or competitor_pressure < 0.75:
        return list(actions)
    hard_case_pressure = max(
        _clamp01(state.ambiguity_pressure),
        _clamp01(ambiguity_context.get("od_hard_case_prior")),
        _clamp01(ambiguity_context.get("ambiguity_budget_prior")),
    )
    if hard_case_pressure < 0.45:
        return list(actions)
    evidence_signature = _clamp01(
        (0.40 * min(1.0, top_refresh_gain / 0.25))
        + (0.35 * top_fragility_mass)
        + (0.25 * competitor_pressure)
    )
    q_gap = max(0.0, search_q - evidence_q)
    plateau_signal = _clamp01(min(1.0, plateau_progress_count / 3.0))
    if first_iteration_near_tie_plateau:
        plateau_signal = max(
            plateau_signal,
            _clamp01(
                (0.50 * near_tie_mass)
                + (0.30 * evidence_signature)
                + (0.20 * max(0.0, 1.0 - objective_spread))
            ),
        )
        bonus = min(
            0.32,
            q_gap
            + 0.02
            + (0.05 * evidence_signature)
            + (0.03 * near_tie_mass),
        )
    else:
        bonus = min(
            0.40,
            q_gap
            + 0.012
            + (0.022 * evidence_signature)
            + (0.018 * plateau_signal),
        )
    if bonus <= 0.0:
        return list(actions)
    updated = list(actions)
    evidence_metadata = dict(evidence_metadata)
    evidence_metadata["uncertified_evidence_plateau_preference_bonus"] = round(bonus, 6)
    evidence_metadata["uncertified_evidence_plateau_preference_applied"] = True
    evidence_metadata["uncertified_evidence_plateau_progress_count"] = plateau_progress_count
    if first_iteration_near_tie_plateau:
        evidence_metadata["uncertified_evidence_plateau_first_iteration_near_tie"] = True
    if direct_fallback_frontier_probe_plateau:
        evidence_metadata["uncertified_evidence_plateau_direct_fallback_probe_recovery"] = True
        if recent_no_gain_refine_streak > plateau_progress_count:
            evidence_metadata["uncertified_evidence_plateau_direct_fallback_dead_probe_recovery"] = True
    updated[best_evidence_index] = replace(
        best_evidence_action,
        q_score=evidence_q + bonus,
        metadata=evidence_metadata,
    )
    if first_iteration_near_tie_plateau:
        search_discount = min(
            0.18,
            0.08
            + (0.03 * evidence_signature)
            + (0.02 * near_tie_mass),
        )
    else:
        search_discount = min(
            0.14,
            0.04
            + (0.02 * max(0.0, plateau_progress_count - 1.0))
            + (0.02 * evidence_signature),
        )
    if search_discount > 0.0:
        search_metadata = dict(best_search_action.metadata)
        search_metadata["uncertified_evidence_plateau_search_discount"] = round(search_discount, 6)
        search_metadata["uncertified_evidence_plateau_search_discount_applied"] = True
        if first_iteration_near_tie_plateau:
            search_metadata["uncertified_evidence_plateau_first_iteration_search_discount"] = True
        if direct_fallback_frontier_probe_plateau:
            search_metadata["uncertified_evidence_plateau_direct_fallback_probe_discount"] = True
            if recent_no_gain_refine_streak > plateau_progress_count:
                search_metadata["uncertified_evidence_plateau_direct_fallback_dead_probe_discount"] = True
        updated[best_search_index] = replace(
            best_search_action,
            q_score=max(0.0, search_q - search_discount),
            metadata=search_metadata,
    )
    return updated


def _apply_uncertified_first_iteration_near_tie_resample_preference(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
) -> list[VOIAction]:
    certificate_threshold = _as_float(config.certificate_threshold)
    if current_certificate >= certificate_threshold:
        return list(actions)
    if state.action_trace:
        return list(actions)
    if not evidence_uncertainty:
        return list(actions)
    best_search_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    best_resample_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind == "increase_stochastic_samples"
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    if best_search_entry is None or best_resample_entry is None:
        return list(actions)
    search_index, search_action = best_search_entry
    resample_index, resample_action = best_resample_entry
    search_q = max(0.0, _as_float(search_action.q_score))
    resample_q = max(0.0, _as_float(resample_action.q_score))
    if resample_q <= 0.0 or search_q <= resample_q:
        return list(actions)
    if current_certificate > max(0.68, certificate_threshold - 0.12):
        return list(actions)
    near_tie_mass = max(
        _clamp01(state.near_tie_mass),
        _clamp01(
            (resample_action.metadata if isinstance(resample_action.metadata, Mapping) else {}).get(
                "near_tie_mass"
            )
        ),
    )
    if near_tie_mass < 0.75:
        return list(actions)
    top_refresh_gain = max(0.0, _as_float(state.top_refresh_gain))
    top_fragility_mass = _clamp01(state.top_fragility_mass)
    competitor_pressure = _clamp01(state.competitor_pressure)
    if top_refresh_gain < 0.12 and top_fragility_mass < 0.20 and competitor_pressure < 0.75:
        return list(actions)
    search_metadata = search_action.metadata if isinstance(search_action.metadata, Mapping) else {}
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(search_action.predicted_delta_frontier))
    if objective_gap >= 0.06 or predicted_frontier >= 0.10:
        return list(actions)
    if mechanism_gap >= 0.28 and objective_gap >= 0.05:
        return list(actions)
    resample_certificate = max(0.0, _as_float(resample_action.predicted_delta_certificate))
    q_gap = max(0.0, search_q - resample_q)
    evidence_signature = _clamp01(
        (0.45 * top_fragility_mass)
        + (0.30 * competitor_pressure)
        + (0.15 * min(1.0, top_refresh_gain / 0.20))
        + (0.10 * near_tie_mass)
    )
    bonus = min(
        0.22,
        q_gap
        + 0.02
        + (0.05 * min(1.0, resample_certificate / 0.25))
        + (0.03 * evidence_signature)
        + (0.03 * near_tie_mass),
    )
    if bonus <= 0.0:
        return list(actions)
    updated = list(actions)
    resample_metadata = dict(resample_action.metadata if isinstance(resample_action.metadata, Mapping) else {})
    resample_metadata["uncertified_first_iteration_near_tie_resample_bonus"] = round(bonus, 6)
    resample_metadata["uncertified_first_iteration_near_tie_resample_preference_applied"] = True
    updated[resample_index] = replace(
        resample_action,
        q_score=resample_q + bonus,
        metadata=resample_metadata,
    )
    search_discount = min(
        0.12,
        0.04
        + (0.03 * min(1.0, resample_certificate / 0.25))
        + (0.02 * evidence_signature)
        + (0.02 * near_tie_mass),
    )
    if search_discount > 0.0:
        updated_search_metadata = dict(search_metadata)
        updated_search_metadata["uncertified_first_iteration_near_tie_resample_search_discount"] = round(
            search_discount,
            6,
        )
        updated_search_metadata["uncertified_first_iteration_near_tie_resample_search_discount_applied"] = True
        updated[search_index] = replace(
            search_action,
            q_score=max(0.0, search_q - search_discount),
            metadata=updated_search_metadata,
        )
    return updated


def _apply_uncertified_resample_recovery_preference(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
) -> list[VOIAction]:
    certificate_threshold = _as_float(config.certificate_threshold)
    threshold_gap = certificate_threshold - current_certificate
    if current_certificate >= certificate_threshold:
        return list(actions)
    if threshold_gap <= 0.0 or threshold_gap > 0.08:
        return list(actions)
    if not evidence_uncertainty or not supported_fragility_uncertainty:
        return list(actions)
    if not _support_rich_ambiguity_window(state):
        return list(actions)
    best_search_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    best_resample_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind == "increase_stochastic_samples"
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    if best_search_entry is None or best_resample_entry is None:
        return list(actions)
    search_index, search_action = best_search_entry
    resample_index, resample_action = best_resample_entry
    search_q = max(0.0, _as_float(search_action.q_score))
    resample_q = max(0.0, _as_float(resample_action.q_score))
    if resample_q <= 0.0 or search_q <= resample_q:
        return list(actions)
    search_metadata = search_action.metadata if isinstance(search_action.metadata, Mapping) else {}
    resample_metadata = resample_action.metadata if isinstance(resample_action.metadata, Mapping) else {}
    if not bool(search_metadata.get("certificate_headroom_cap_applied")):
        return list(actions)
    remaining_headroom = max(0.0, 1.0 - _clamp01(current_certificate))
    predicted_before_cap = _as_float(
        search_metadata.get("certificate_headroom_predicted_before_cap"),
        _as_float(search_action.predicted_delta_certificate),
    )
    if predicted_before_cap < remaining_headroom + 0.12:
        return list(actions)
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(search_action.predicted_delta_frontier))
    if objective_gap >= 0.08 or mechanism_gap >= 0.20 or predicted_frontier >= 0.10:
        return list(actions)
    resample_certificate = max(0.0, _as_float(resample_action.predicted_delta_certificate))
    if resample_certificate < 0.06:
        return list(actions)
    top_refresh_gain = max(0.0, _as_float(state.top_refresh_gain))
    top_fragility_mass = _clamp01(state.top_fragility_mass)
    competitor_pressure = _clamp01(state.competitor_pressure)
    if top_refresh_gain < 0.20 and top_fragility_mass < 0.18 and competitor_pressure < 0.75:
        return list(actions)
    structured_negative_refresh = False
    for action in actions:
        if action.kind != "refresh_top1_vor":
            continue
        refresh_metadata = action.metadata if isinstance(action.metadata, Mapping) else {}
        signed_refresh_delta = _as_float(
            refresh_metadata.get("empirical_refresh_certificate_delta"),
            float("nan"),
        )
        if bool(refresh_metadata.get("structured_refresh_signal")) and signed_refresh_delta < -1e-9:
            structured_negative_refresh = True
            break
    if not structured_negative_refresh and top_refresh_gain < 0.28:
        return list(actions)
    q_gap = max(0.0, search_q - resample_q)
    bonus = min(
        0.18,
        q_gap
        + 0.015
        + (0.05 * min(1.0, resample_certificate / 0.18))
        + (0.03 * top_fragility_mass)
        + (0.02 if structured_negative_refresh else 0.0),
    )
    if bonus <= 0.0:
        return list(actions)
    updated = list(actions)
    updated_resample_metadata = dict(resample_metadata)
    updated_resample_metadata["uncertified_resample_recovery_preference_bonus"] = round(bonus, 6)
    updated_resample_metadata["uncertified_resample_recovery_preference_applied"] = True
    updated_resample_metadata["uncertified_resample_recovery_structured_negative_refresh"] = (
        structured_negative_refresh
    )
    updated[resample_index] = replace(
        resample_action,
        q_score=resample_q + bonus,
        metadata=updated_resample_metadata,
    )
    search_discount = min(
        0.10,
        0.03
        + (0.02 * min(1.0, resample_certificate / 0.18))
        + (0.015 * top_fragility_mass)
        + (0.015 if structured_negative_refresh else 0.0),
    )
    if search_discount > 0.0:
        updated_search_metadata = dict(search_metadata)
        updated_search_metadata["uncertified_resample_recovery_search_discount"] = round(
            search_discount,
            6,
        )
        updated_search_metadata["uncertified_resample_recovery_search_discount_applied"] = True
        updated[search_index] = replace(
            search_action,
            q_score=max(0.0, search_q - search_discount),
            metadata=updated_search_metadata,
        )
    return updated


def _apply_uncertified_post_evidence_resample_preference(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
) -> list[VOIAction]:
    certificate_threshold = _as_float(config.certificate_threshold)
    threshold_gap = certificate_threshold - current_certificate
    if current_certificate >= certificate_threshold:
        return list(actions)
    if threshold_gap <= 0.0 or threshold_gap > 0.30:
        return list(actions)
    if not evidence_uncertainty or not supported_fragility_uncertainty:
        return list(actions)
    if not _support_rich_ambiguity_window(state):
        return list(actions)
    if len(state.action_trace) < 3:
        return list(actions)
    if not _trace_has_prior_meaningful_evidence_certificate_lift(state, minimum_lift=0.05):
        return list(actions)
    best_search_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    best_resample_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind == "increase_stochastic_samples"
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    refresh_entry = _best_refresh_action_entry(actions)
    if best_search_entry is None or best_resample_entry is None or refresh_entry is None:
        return list(actions)
    _, refresh_action = refresh_entry
    refresh_metadata = refresh_action.metadata if isinstance(refresh_action.metadata, Mapping) else {}
    signed_refresh_delta = _as_float(
        refresh_metadata.get("empirical_refresh_certificate_delta"),
        float("nan"),
    )
    structured_negative_refresh = bool(
        bool(refresh_metadata.get("structured_refresh_signal"))
        and math.isfinite(signed_refresh_delta)
        and signed_refresh_delta < -1e-9
    )
    if not structured_negative_refresh:
        return list(actions)
    if current_certificate < 0.55:
        return list(actions)
    if _clamp01(state.top_fragility_mass) < 0.45 or _clamp01(state.competitor_pressure) < 0.75:
        return list(actions)
    if _stress_world_fraction_from_context(state.ambiguity_context) < 0.20:
        return list(actions)
    if (
        max(0.0, _as_float(state.search_completeness_gap)) < 0.30
        and _clamp01(state.pending_challenger_mass) < 0.55
        and max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)) < 0.60
    ):
        return list(actions)
    search_index, search_action = best_search_entry
    resample_index, resample_action = best_resample_entry
    search_q = max(0.0, _as_float(search_action.q_score))
    resample_q = max(0.0, _as_float(resample_action.q_score))
    if resample_q <= 0.0 or search_q <= resample_q:
        return list(actions)
    resample_certificate = max(0.0, _as_float(resample_action.predicted_delta_certificate))
    if resample_certificate < 0.05:
        return list(actions)
    q_gap = max(0.0, search_q - resample_q)
    stress_signal = _clamp01(
        (0.40 * _clamp01(state.top_fragility_mass))
        + (0.30 * _clamp01(state.competitor_pressure))
        + (0.20 * _stress_world_fraction_from_context(state.ambiguity_context))
        + (0.10 * min(1.0, max(0.0, threshold_gap) / 0.12))
    )
    bonus = min(
        0.40,
        q_gap
        + 0.02
        + (0.06 * min(1.0, resample_certificate / 0.12))
        + (0.04 * stress_signal),
    )
    if bonus <= 0.0:
        return list(actions)
    updated = list(actions)
    resample_metadata = dict(resample_action.metadata if isinstance(resample_action.metadata, Mapping) else {})
    resample_metadata["uncertified_post_evidence_resample_preference_bonus"] = round(bonus, 6)
    resample_metadata["uncertified_post_evidence_resample_preference_applied"] = True
    resample_metadata["uncertified_post_evidence_resample_structured_negative_refresh"] = True
    updated[resample_index] = replace(
        resample_action,
        q_score=resample_q + bonus,
        metadata=resample_metadata,
    )
    search_discount = min(
        0.22,
        0.10
        + (0.05 * min(1.0, resample_certificate / 0.12))
        + (0.03 * stress_signal),
    )
    if search_discount > 0.0:
        search_metadata = dict(search_action.metadata if isinstance(search_action.metadata, Mapping) else {})
        search_metadata["uncertified_post_evidence_resample_search_discount"] = round(search_discount, 6)
        search_metadata["uncertified_post_evidence_resample_search_discount_applied"] = True
        updated[search_index] = replace(
            search_action,
            q_score=max(0.0, search_q - search_discount),
            metadata=search_metadata,
    )
    return updated


def _apply_uncertified_last_search_token_resample_preference(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
) -> list[VOIAction]:
    certificate_threshold = _as_float(config.certificate_threshold)
    threshold_gap = certificate_threshold - current_certificate
    if current_certificate >= certificate_threshold:
        return list(actions)
    if current_certificate < max(0.45, certificate_threshold - 0.34):
        return list(actions)
    if threshold_gap <= 0.0 or threshold_gap > 0.35:
        return list(actions)
    if not evidence_uncertainty or not supported_fragility_uncertainty:
        return list(actions)
    if not _support_rich_ambiguity_window(state):
        return list(actions)
    if state.remaining_search_budget > 1 or state.remaining_evidence_budget <= 0:
        return list(actions)
    recent_no_gain_refine_streak = _recent_no_gain_refine_streak(state, max_depth=2)
    if recent_no_gain_refine_streak <= 0:
        return list(actions)
    best_search_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    best_resample_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind == "increase_stochastic_samples"
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    if best_search_entry is None or best_resample_entry is None:
        return list(actions)
    search_index, search_action = best_search_entry
    resample_index, resample_action = best_resample_entry
    search_q = max(0.0, _as_float(search_action.q_score))
    resample_q = max(0.0, _as_float(resample_action.q_score))
    if resample_q <= 0.0 or search_q <= resample_q:
        return list(actions)
    search_metadata = search_action.metadata if isinstance(search_action.metadata, Mapping) else {}
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(search_action.predicted_delta_frontier))
    q_gap = max(0.0, search_q - resample_q)
    cached_direct_fallback_resample_exception = bool(
        _selected_route_uses_cached_direct_fallback(state)
        and q_gap <= 0.35
        and current_certificate >= max(0.55, certificate_threshold - 0.24)
        and max(0.0, _as_float(state.top_refresh_gain)) >= 0.20
        and _clamp01(state.top_fragility_mass) >= 0.55
        and _clamp01(state.competitor_pressure) >= 0.95
        and _clamp01(state.corridor_family_recall) >= 0.85
        and _clamp01(state.frontier_recall_at_budget) <= 0.35
    )
    if (
        objective_gap >= 0.30
        or mechanism_gap >= 0.20
        or predicted_frontier >= 0.35
    ) and not cached_direct_fallback_resample_exception:
        return list(actions)
    resample_certificate = max(0.0, _as_float(resample_action.predicted_delta_certificate))
    if resample_certificate < 0.08:
        return list(actions)
    top_fragility_mass = _clamp01(state.top_fragility_mass)
    competitor_pressure = _clamp01(state.competitor_pressure)
    if top_fragility_mass < 0.45 or competitor_pressure < 0.75:
        return list(actions)
    certificate_margin = max(0.0, _as_float(state.certificate_margin))
    near_tie_mass = _clamp01(state.near_tie_mass)
    if certificate_margin > 0.03 and near_tie_mass < 0.20 and not cached_direct_fallback_resample_exception:
        return list(actions)
    stall_signal = _clamp01(
        (0.60 * min(1.0, recent_no_gain_refine_streak / 2.0))
        + (0.40 * min(1.0, max(0, 1 - state.remaining_search_budget)))
    )
    evidence_signature = _clamp01(
        (0.45 * top_fragility_mass)
        + (0.30 * competitor_pressure)
        + (0.15 * min(1.0, threshold_gap / 0.25))
        + (0.10 * max(near_tie_mass, _clamp01(1.0 - min(1.0, certificate_margin / 0.03))))
    )
    bonus = min(
        0.42,
        q_gap
        + 0.05
        + (0.05 * min(1.0, resample_certificate / 0.15))
        + (0.03 * stall_signal)
        + (0.03 * evidence_signature),
    )
    if bonus <= 0.0:
        return list(actions)
    updated = list(actions)
    resample_metadata = dict(resample_action.metadata if isinstance(resample_action.metadata, Mapping) else {})
    resample_metadata["uncertified_last_search_token_resample_preference_bonus"] = round(bonus, 6)
    resample_metadata["uncertified_last_search_token_resample_preference_applied"] = True
    resample_metadata["uncertified_last_search_token_resample_recent_no_gain_refine_streak"] = (
        recent_no_gain_refine_streak
    )
    updated[resample_index] = replace(
        resample_action,
        q_score=resample_q + bonus,
        metadata=resample_metadata,
    )
    search_discount = min(
        0.18,
        0.08
        + (0.03 * min(1.0, resample_certificate / 0.15))
        + (0.02 * stall_signal)
        + (0.02 * evidence_signature),
    )
    if search_discount > 0.0:
        updated_search_metadata = dict(search_metadata)
        updated_search_metadata["uncertified_last_search_token_resample_search_discount"] = round(
            search_discount,
            6,
        )
        updated_search_metadata["uncertified_last_search_token_resample_search_discount_applied"] = True
        updated[search_index] = replace(
            search_action,
            q_score=max(0.0, search_q - search_discount),
            metadata=updated_search_metadata,
        )
    return updated


def _apply_uncertified_structural_cap_bridge_preference(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    if current_certificate >= _as_float(config.certificate_threshold):
        return list(actions)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    if len(state.frontier) > 1:
        return list(actions)
    if not bool(ambiguity_context.get("single_frontier_certificate_cap_applied")):
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    bridge_index = next(
        (
            idx
            for idx, action in enumerate(actions)
            if action.kind == "increase_stochastic_samples"
            and bool((action.metadata if isinstance(action.metadata, Mapping) else {}).get("uncertified_structural_cap_bridge"))
        ),
        None,
    )
    if bridge_index is None:
        return list(actions)
    best_search_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    if best_search_entry is None:
        return list(actions)
    search_index, search_action = best_search_entry
    search_metadata = search_action.metadata if isinstance(search_action.metadata, Mapping) else {}
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    predicted_frontier = max(0.0, _as_float(search_action.predicted_delta_frontier))
    if objective_gap >= 0.08 or predicted_frontier >= 0.14:
        return list(actions)
    bridge_action = actions[bridge_index]
    bridge_metadata = bridge_action.metadata if isinstance(bridge_action.metadata, Mapping) else {}
    if max(0.0, _as_float(bridge_action.predicted_delta_certificate)) < 0.12:
        return list(actions)
    bridge_support = max(
        _clamp01(bridge_metadata.get("search_pressure")),
        _clamp01(bridge_metadata.get("support_signal")),
    )
    bridge_gap = max(0.0, _as_float(search_action.q_score) - _as_float(bridge_action.q_score))
    bridge_bonus = min(
        0.12,
        bridge_gap + 0.01 + (0.02 * bridge_support),
    )
    if bridge_bonus <= 0.0:
        return list(actions)
    updated = list(actions)
    updated_bridge_metadata = dict(bridge_metadata)
    updated_bridge_metadata["structural_cap_bridge_preference_bonus"] = round(bridge_bonus, 6)
    updated_bridge_metadata["structural_cap_bridge_preference_applied"] = True
    updated[bridge_index] = replace(
        bridge_action,
        q_score=_as_float(bridge_action.q_score) + bridge_bonus,
        metadata=updated_bridge_metadata,
    )
    search_discount = min(0.04, 0.01 + (0.015 * bridge_support))
    if search_discount > 0.0:
        updated_search_metadata = dict(search_metadata)
        updated_search_metadata["structural_cap_bridge_search_discount"] = round(search_discount, 6)
        updated_search_metadata["structural_cap_bridge_search_discount_applied"] = True
        updated[search_index] = replace(
            search_action,
            q_score=max(0.0, _as_float(search_action.q_score) - search_discount),
            metadata=updated_search_metadata,
        )
    return updated


def _apply_uncertified_support_rich_zero_signal_bridge_preference(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    if current_certificate >= _as_float(config.certificate_threshold):
        return list(actions)
    bridge_index = next(
        (
            idx
            for idx, action in enumerate(actions)
            if action.kind == "increase_stochastic_samples"
            and bool(
                (action.metadata if isinstance(action.metadata, Mapping) else {}).get(
                    "uncertified_support_rich_zero_signal_bridge"
                )
            )
        ),
        None,
    )
    if bridge_index is None:
        return list(actions)
    bridge_action = actions[bridge_index]
    bridge_metadata = bridge_action.metadata if isinstance(bridge_action.metadata, Mapping) else {}
    live_sampler_bridge = bool(
        bridge_metadata.get("uncertified_support_rich_zero_signal_live_sampler_bridge")
    )
    cert_world_bridge = bool(
        bridge_metadata.get("uncertified_support_rich_zero_signal_cert_world_bridge")
    )
    severe_sampler_undercoverage = bool(
        bridge_metadata.get("uncertified_support_rich_zero_signal_extreme_undercoverage")
    )
    fallback_activated = bool(bridge_metadata.get("controller_refresh_fallback_activated"))
    controller_disagreement = bool(
        bridge_metadata.get("controller_empirical_vs_raw_refresh_disagreement")
    )
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    single_frontier_structural_cap = bool(
        len(state.frontier) <= 1
        and ambiguity_context.get("single_frontier_certificate_cap_applied")
    )
    structural_cap_live_bridge_override = bool(
        (live_sampler_bridge or cert_world_bridge)
        and single_frontier_structural_cap
        and (fallback_activated or controller_disagreement)
        and (_resample_shortfall_available(state) or _cert_world_shortfall_available(state))
    )
    extreme_undercoverage_live_bridge_override = bool(
        live_sampler_bridge
        and severe_sampler_undercoverage
        and _resample_shortfall_available(state)
    )
    if state.stochastic_enabled and not (live_sampler_bridge or cert_world_bridge):
        return list(actions)
    best_search_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}
        ),
        key=lambda item: item[1].q_score,
        default=None,
    )
    if best_search_entry is None:
        return list(actions)
    search_index, search_action = best_search_entry
    search_metadata = search_action.metadata if isinstance(search_action.metadata, Mapping) else {}
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(search_action.predicted_delta_frontier))
    if objective_gap >= 0.12 or predicted_frontier >= 0.15:
        return list(actions)
    if (
        mechanism_gap >= 0.35
        and not structural_cap_live_bridge_override
        and not extreme_undercoverage_live_bridge_override
    ):
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    if max(0.0, _as_float(bridge_action.predicted_delta_certificate)) < 0.12:
        return list(actions)
    bridge_support = max(
        _clamp01(bridge_metadata.get("search_pressure")),
        _clamp01(bridge_metadata.get("support_signal")),
        _clamp01(bridge_metadata.get("hard_case_support")),
    )
    bridge_q = max(0.0, _as_float(bridge_action.q_score))
    search_q = max(0.0, _as_float(search_action.q_score))
    certificate_threshold = _as_float(config.certificate_threshold)
    threshold_gap = max(0.0, certificate_threshold - current_certificate)
    search_certificate = max(0.0, _as_float(search_action.predicted_delta_certificate))
    single_frontier_zero_signal = bool(
        len(state.frontier) <= 1
        and _clamp01(state.near_tie_mass) <= max(_as_float(config.near_tie_threshold), 0.03)
        and max(0.0, _as_float(state.top_refresh_gain)) <= 0.0
        and _clamp01(state.top_fragility_mass) <= 0.01
        and _clamp01(state.competitor_pressure) <= 0.05
    )
    single_frontier_search_finish_supported = bool(
        not single_frontier_zero_signal
        or objective_gap >= 0.04
        or predicted_frontier >= 0.05
    )
    near_threshold_search_finish = bool(
        threshold_gap <= 0.07
        and bridge_q >= search_q - 1e-9
        and abs(search_q - bridge_q) <= 0.08
        and search_q >= max((4.0 * _as_float(config.stop_threshold)), 0.10)
        and search_certificate >= min(0.30, threshold_gap + 0.10)
        and (
            objective_gap >= 0.04
            or mechanism_gap >= 0.45
            or predicted_frontier >= 0.05
        )
        and single_frontier_search_finish_supported
        and not structural_cap_live_bridge_override
    )
    if near_threshold_search_finish:
        updated = list(actions)
        search_finish_signal = min(
            1.0,
            search_certificate / max(0.05, threshold_gap + 0.10),
        )
        search_finish_bonus = min(
            0.18,
            max(0.0, bridge_q - search_q)
            + 0.012
            + (0.03 * bridge_support)
            + (0.02 * search_finish_signal),
        )
        updated_search_metadata = dict(search_metadata)
        updated_search_metadata["uncertified_support_rich_zero_signal_search_finish_bonus"] = round(
            search_finish_bonus,
            6,
        )
        updated_search_metadata["uncertified_support_rich_zero_signal_search_finish_preferred"] = True
        updated[search_index] = replace(
            search_action,
            q_score=search_q + search_finish_bonus,
            metadata=updated_search_metadata,
        )
        bridge_finish_discount = min(
            0.10,
            0.015
            + (0.02 * bridge_support)
            + (0.02 * min(1.0, threshold_gap / 0.07)),
        )
        updated_bridge_metadata = dict(bridge_metadata)
        updated_bridge_metadata["uncertified_support_rich_zero_signal_bridge_finish_discount"] = round(
            bridge_finish_discount,
            6,
        )
        updated_bridge_metadata["uncertified_support_rich_zero_signal_bridge_finish_deferred"] = True
        updated[bridge_index] = replace(
            bridge_action,
            q_score=max(0.0, bridge_q - bridge_finish_discount),
            metadata=updated_bridge_metadata,
        )
        return updated
    controller_bridge_bonus = 0.0
    if fallback_activated:
        controller_bridge_bonus += 0.01
    if controller_disagreement:
        controller_bridge_bonus += 0.02
    if live_sampler_bridge:
        controller_bridge_bonus += 0.01
    if cert_world_bridge:
        controller_bridge_bonus += 0.02
    if severe_sampler_undercoverage:
        controller_bridge_bonus += 0.03
    if structural_cap_live_bridge_override:
        controller_bridge_bonus += 0.03 if controller_disagreement else 0.015
    bridge_gap = max(0.0, search_q - bridge_q)
    bridge_bonus = min(
        (
            0.18
            if structural_cap_live_bridge_override
            else (
                0.16
                if severe_sampler_undercoverage
                else (0.14 if (live_sampler_bridge or cert_world_bridge) else 0.10)
            )
        ),
        bridge_gap + 0.01 + (0.03 * bridge_support) + controller_bridge_bonus,
    )
    if bridge_bonus <= 0.0:
        return list(actions)
    updated = list(actions)
    updated_bridge_metadata = dict(bridge_metadata)
    updated_bridge_metadata["uncertified_support_rich_zero_signal_bridge_preference_bonus"] = round(
        bridge_bonus,
        6,
    )
    updated_bridge_metadata["uncertified_support_rich_zero_signal_bridge_preference_applied"] = True
    updated[bridge_index] = replace(
        bridge_action,
        q_score=_as_float(bridge_action.q_score) + bridge_bonus,
        metadata=updated_bridge_metadata,
    )
    search_discount = min(
        (
            0.08
            if structural_cap_live_bridge_override
            else (
                0.06
                if severe_sampler_undercoverage
                else (0.05 if (live_sampler_bridge or cert_world_bridge) else 0.04)
            )
        ),
        0.01
        + (0.015 * bridge_support)
        + (0.01 if (live_sampler_bridge or cert_world_bridge) else 0.0)
        + (0.015 if severe_sampler_undercoverage else 0.0)
        + (0.02 if structural_cap_live_bridge_override else 0.0),
    )
    if search_discount > 0.0:
        updated_search_metadata = dict(search_metadata)
        updated_search_metadata["uncertified_support_rich_zero_signal_bridge_search_discount"] = round(
            search_discount,
            6,
        )
        updated_search_metadata["uncertified_support_rich_zero_signal_bridge_search_discount_applied"] = True
        updated[search_index] = replace(
            search_action,
            q_score=max(0.0, _as_float(search_action.q_score) - search_discount),
            metadata=updated_search_metadata,
        )
    return updated


def _should_stop_uncertified_weak_search_tail(
    action: VOIAction | None,
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_uncertainty: bool,
) -> bool:
    if action is None or action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return False
    if current_certificate >= _as_float(config.certificate_threshold):
        return False
    if current_certificate < max(0.72, _as_float(config.certificate_threshold) - 0.10):
        return False
    if evidence_uncertainty:
        direct_winner_side_evidence_signal = bool(
            max(0.0, _as_float(state.top_refresh_gain)) > 0.0
            or _clamp01(state.top_fragility_mass) > 0.0
            or _clamp01(state.competitor_pressure) > 0.05
        )
        if direct_winner_side_evidence_signal:
            return False
    if len(state.frontier) > 1:
        return False
    metadata = action.metadata if isinstance(action.metadata, Mapping) else {}
    objective_gap = _clamp01(metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(action.predicted_delta_frontier))
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    world_count_policy = str(ambiguity_context.get("refc_world_count_policy", "")).strip()
    actual_world_count = _actual_refc_world_count(state)
    single_frontier_world_saturation = bool(
        len(state.frontier) <= 1
        and actual_world_count > 0.0
        and not _resample_shortfall_available(state)
        and (
            "single_frontier" in world_count_policy
            or bool(ambiguity_context.get("single_frontier_certificate_cap_applied"))
        )
    )
    if objective_gap >= 0.08 or mechanism_gap >= 0.40 or predicted_frontier >= 0.12:
        if not single_frontier_world_saturation:
            return False
        if objective_gap >= 0.12 or mechanism_gap >= 0.28 or predicted_frontier >= 0.14:
            return False
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return False
    if max(0.0, _as_float(state.top_refresh_gain)) > 0.0 or _clamp01(state.top_fragility_mass) > 0.01:
        return False
    return True


def _suppress_certified_zero_signal_controller_churn(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    evidence_discovery_bridge: bool,
) -> list[VOIAction]:
    saturation_certificate = min(
        1.0,
        max(0.95, _as_float(config.certificate_threshold) + 0.20),
    )
    if current_certificate < saturation_certificate:
        return list(actions)
    support_rich_window = _support_rich_ambiguity_window(state)
    corridor_complete = _clamp01(state.corridor_family_recall) >= 0.75
    if not support_rich_window and not corridor_complete:
        return list(actions)
    if _clamp01(state.near_tie_mass) > 0.03:
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.0
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    if evidence_discovery_bridge:
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if _search_action_shows_strong_decision_movement(best_search_action, state=state):
        if _trace_has_prior_evidence_action_attempt(state):
            return list(actions)
        low_coverage_reopen = bool(
            _clamp01(state.corridor_family_recall) < 0.50
            and _clamp01(state.frontier_recall_at_budget) < 0.40
        )
        if low_coverage_reopen:
            search_metadata = best_search_action.metadata if best_search_action is not None else {}
            structural_reopen_signal = bool(
                _clamp01(search_metadata.get("normalized_objective_gap")) >= 0.10
                or _clamp01(search_metadata.get("normalized_mechanism_gap")) >= 0.08
                or max(
                    0.0,
                    _as_float(best_search_action.predicted_delta_frontier if best_search_action is not None else 0.0),
                ) >= 0.10
            )
            if structural_reopen_signal:
                return list(actions)
    if _refine_action_has_genuine_novel_search_promise(best_search_action, state=state):
        return list(actions)
    if _clamp01((best_search_action.metadata if best_search_action is not None else {}).get("normalized_mechanism_gap")) >= 0.12:
        return list(actions)
    return [
        action
        for action in actions
        if action.kind not in {"refine_top1_dccs", "refine_topk_dccs", "refresh_top1_vor", "increase_stochastic_samples"}
    ]


def _suppress_certified_single_frontier_zero_signal_search_churn(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    threshold = _as_float(config.certificate_threshold)
    near_certified_floor = max(0.75, threshold - 0.05)
    saturation_certificate = min(
        1.0,
        max(0.95, threshold + 0.20),
    )
    if current_certificate < near_certified_floor or current_certificate >= saturation_certificate:
        return list(actions)
    if len(state.frontier) > 1:
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.0
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if best_search_action is None:
        return list(actions)
    metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
    if not bool(metadata.get("certificate_headroom_cap_applied")):
        return list(actions)
    objective_gap = _clamp01(metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
    corridor_recall = _clamp01(metadata.get("corridor_family_recall", state.corridor_family_recall))
    frontier_recall = _clamp01(metadata.get("frontier_recall_at_budget", state.frontier_recall_at_budget))
    if (
        objective_gap >= 0.16
        or mechanism_gap >= 0.22
        or predicted_frontier >= 0.18
        or corridor_recall > 0.15
        or frontier_recall > 0.20
    ):
        return list(actions)
    return [action for action in actions if action.kind == "stop"]


def _suppress_uncertified_structural_cap_churn(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    if current_certificate >= _as_float(config.certificate_threshold):
        return list(actions)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    if len(state.frontier) > 1:
        return list(actions)
    if not bool(ambiguity_context.get("single_frontier_certificate_cap_applied")):
        return list(actions)
    empirical_baseline = _as_float(ambiguity_context.get("empirical_baseline_certificate"), float("nan"))
    controller_baseline = _as_float(ambiguity_context.get("controller_baseline_certificate"), float("nan"))
    if not math.isfinite(empirical_baseline) or not math.isfinite(controller_baseline):
        return list(actions)
    if empirical_baseline <= controller_baseline + 1e-9:
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    bridge_resample_action = next(
        (
            action
            for action in actions
            if action.kind == "increase_stochastic_samples"
            and bool(
                (action.metadata if isinstance(action.metadata, Mapping) else {}).get("uncertified_structural_cap_bridge")
                or (action.metadata if isinstance(action.metadata, Mapping) else {}).get(
                    "uncertified_support_rich_zero_signal_live_sampler_bridge"
                )
                or (action.metadata if isinstance(action.metadata, Mapping) else {}).get(
                    "uncertified_support_rich_zero_signal_cert_world_bridge"
                )
            )
        ),
        None,
    )
    if bridge_resample_action is not None:
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    actual_world_count = _actual_refc_world_count(state)
    if best_search_action is not None and actual_world_count > 0.0:
        search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
        objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
        mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
        predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
        structural_cap_only_refresh = any(
            action.kind == "refresh_top1_vor"
            and bool((action.metadata if isinstance(action.metadata, Mapping) else {}).get("structured_refresh_signal"))
            and bool((action.metadata if isinstance(action.metadata, Mapping) else {}).get("structural_cap_only"))
            and _as_float(
                (action.metadata if isinstance(action.metadata, Mapping) else {}).get(
                    "empirical_refresh_certificate_uplift"
                ),
                float("nan"),
            )
            <= 1e-9
            for action in actions
        )
        if (
            current_certificate >= max(0.72, _as_float(config.certificate_threshold) - 0.10)
            and objective_gap < 0.08
            and predicted_frontier < 0.12
        ):
            return [action for action in actions if action.kind == "stop"]
        if (
            structural_cap_only_refresh
            and actual_world_count <= 1.0
            and not _resample_shortfall_available(state)
            and current_certificate >= max(0.72, _as_float(config.certificate_threshold) - 0.10)
            and objective_gap < 0.12
            and mechanism_gap < 0.24
            and predicted_frontier < 0.14
        ):
            return [action for action in actions if action.kind == "stop"]
    if _refine_action_has_genuine_novel_search_promise(best_search_action, state=state):
        return list(actions)
    return [action for action in actions if action.kind == "stop"]


def _suppress_uncertified_stochastic_disabled_zero_signal_controller_churn(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    if current_certificate >= _as_float(config.certificate_threshold):
        return list(actions)
    if current_certificate < max(0.72, _as_float(config.certificate_threshold) - 0.10):
        return list(actions)
    if state.stochastic_enabled:
        return list(actions)
    if state.remaining_evidence_budget <= 0:
        return list(actions)
    if _trace_has_prior_evidence_action_attempt(state):
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    actual_world_count = _actual_refc_world_count(state)
    if not _resample_shortfall_available(state):
        return list(actions)
    if actual_world_count > 1.0 and not bool(context.get("single_frontier_certificate_cap_applied")):
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if best_search_action is not None:
        search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
        objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
        mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
        predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
        if objective_gap >= 0.12 or mechanism_gap >= 0.35 or predicted_frontier >= 0.15:
            return list(actions)
    preferred_bridge_action = next(
        (
            action
            for action in actions
            if action.kind == "increase_stochastic_samples"
            and bool(
                (action.metadata if isinstance(action.metadata, Mapping) else {}).get(
                    "structural_cap_bridge_preference_applied"
                )
                or (action.metadata if isinstance(action.metadata, Mapping) else {}).get(
                    "uncertified_support_rich_zero_signal_bridge_preference_applied"
                )
            )
        ),
        None,
    )
    if preferred_bridge_action is not None:
        return [
            action
            for action in actions
            if action.kind == "stop" or action.action_id == preferred_bridge_action.action_id
        ]
    if best_search_action is not None:
        search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
        objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
        mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
        predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
        support_rich_hard_case_pressure = max(
            _clamp01(context.get("od_hard_case_prior")),
            _clamp01(context.get("ambiguity_budget_prior")),
            _clamp01(context.get("od_ambiguity_support_ratio")),
            _clamp01(state.support_richness),
            _clamp01(state.ambiguity_pressure),
        )
        structural_search_incomplete = bool(
            max(0.0, _as_float(state.search_completeness_gap)) >= 0.22
            and _clamp01(state.pending_challenger_mass) >= 0.60
            and _clamp01(state.best_pending_flip_probability) >= 0.97
            and (
                _clamp01(state.frontier_recall_at_budget) < 0.40
                or _clamp01(state.corridor_family_recall) < 0.50
            )
        )
        material_search_upside = bool(
            objective_gap >= 0.08
            or mechanism_gap >= 0.18
            or predicted_frontier >= 0.08
        )
        if (
            support_rich_hard_case_pressure >= 0.45
            and structural_search_incomplete
            and material_search_upside
        ):
            return [
                action
                for action in actions
                if action.kind == "stop" or action.action_id == best_search_action.action_id
            ]
    return [action for action in actions if action.kind == "stop"]


def _suppress_uncertified_single_frontier_zero_signal_search_churn(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    certificate_threshold = _as_float(config.certificate_threshold)
    if current_certificate >= certificate_threshold:
        return list(actions)
    if state.action_trace:
        return list(actions)
    if current_certificate < max(0.72, certificate_threshold - 0.10):
        return list(actions)
    if len(state.frontier) > 1:
        return list(actions)
    if _trace_has_prior_evidence_action_attempt(state):
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    evidence_path_available = any(
        action.kind in {"refresh_top1_vor", "increase_stochastic_samples"}
        and max(0.0, _as_float(action.q_score)) > 0.0
        for action in actions
    )
    if evidence_path_available:
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if best_search_action is None:
        return list(actions)
    search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
    if objective_gap >= 0.12 or predicted_frontier >= 0.14:
        return list(actions)
    if mechanism_gap >= 0.30 and objective_gap >= 0.08:
        return list(actions)
    if max(0.0, _as_float(state.search_completeness_gap)) < 0.25:
        return list(actions)
    if _clamp01(state.pending_challenger_mass) < 0.55:
        return list(actions)
    return [action for action in actions if action.kind == "stop"]


def _suppress_uncertified_sampler_only_zero_signal_bridge_tail(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    certificate_threshold = _as_float(config.certificate_threshold)
    if current_certificate >= certificate_threshold:
        return list(actions)
    if state.action_trace:
        return list(actions)
    if len(state.frontier) > 1:
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    if bool(state.credible_evidence_uncertainty):
        return list(actions)
    if current_certificate < max(0.74, certificate_threshold - 0.08):
        return list(actions)
    if max(0.0, _as_float(state.search_completeness_gap)) >= 0.10:
        return list(actions)
    if _clamp01(state.frontier_recall_at_budget) < 0.95:
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    bridge_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind == "increase_stochastic_samples"
            and isinstance(action.metadata, Mapping)
            and action.metadata.get("uncertified_support_rich_zero_signal_bridge")
        ),
        key=lambda item: _as_float(item[1].q_score),
        default=None,
    )
    if bridge_entry is None:
        return list(actions)
    _, bridge_action = bridge_entry
    bridge_metadata = bridge_action.metadata if isinstance(bridge_action.metadata, Mapping) else {}
    if not bool(bridge_metadata.get("uncertified_support_rich_zero_signal_live_sampler_bridge")):
        return list(actions)
    if bool(bridge_metadata.get("uncertified_support_rich_zero_signal_cert_world_bridge")):
        return list(actions)
    if bool(bridge_metadata.get("controller_refresh_fallback_activated")):
        return list(actions)
    if bool(bridge_metadata.get("controller_empirical_vs_raw_refresh_disagreement")):
        return list(actions)
    if (
        state.ambiguity_context.get("single_frontier_certificate_cap_applied")
        if isinstance(state.ambiguity_context, Mapping)
        else False
    ):
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if best_search_action is not None:
        search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
        objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
        mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
        predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
        if (
            max(0.0, _as_float(state.search_completeness_gap)) >= 0.15
            or _clamp01(state.frontier_recall_at_budget) < 0.80
        ) and (
            objective_gap >= 0.12
            or predicted_frontier >= 0.12
            or (mechanism_gap >= 0.55 and objective_gap >= 0.10)
        ):
            return [
                action
                for action in actions
                if action.kind == "stop" or action.action_id == best_search_action.action_id
            ]
    return [action for action in actions if action.kind == "stop"]


def _suppress_uncertified_low_support_cert_world_zero_signal_bridge_tail(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    certificate_threshold = _as_float(config.certificate_threshold)
    if current_certificate >= certificate_threshold:
        return list(actions)
    if state.action_trace:
        return list(actions)
    if len(state.frontier) > 1:
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    if current_certificate < max(0.78, certificate_threshold - 0.06):
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    if _clamp01(state.prior_support_strength) >= 0.18:
        return list(actions)
    if _clamp01(ambiguity_context.get("od_hard_case_prior")) > 0.15:
        return list(actions)
    if _clamp01(ambiguity_context.get("od_engine_disagreement_prior")) > 0.15:
        return list(actions)
    if max(
        _clamp01(ambiguity_context.get("od_ambiguity_index")),
        _clamp01(ambiguity_context.get("ambiguity_budget_prior")),
    ) > 0.10:
        return list(actions)
    bridge_entry = max(
        (
            (idx, action)
            for idx, action in enumerate(actions)
            if action.kind == "increase_stochastic_samples"
            and isinstance(action.metadata, Mapping)
            and action.metadata.get("uncertified_support_rich_zero_signal_bridge")
        ),
        key=lambda item: _as_float(item[1].q_score),
        default=None,
    )
    if bridge_entry is None:
        return list(actions)
    _, bridge_action = bridge_entry
    bridge_metadata = bridge_action.metadata if isinstance(bridge_action.metadata, Mapping) else {}
    if not bool(bridge_metadata.get("uncertified_support_rich_zero_signal_cert_world_bridge")):
        return list(actions)
    if bool(bridge_metadata.get("uncertified_support_rich_zero_signal_live_sampler_bridge")):
        return list(actions)
    return [action for action in actions if action.kind == "stop"]


def _suppress_post_nonproductive_bridge_zero_signal_search_churn(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    if current_certificate >= _as_float(config.certificate_threshold):
        return list(actions)
    if state.remaining_search_budget <= 0:
        return list(actions)
    if not _recent_no_gain_evidence_discovery_bridge(state):
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return list(actions)
    actual_world_count = _actual_refc_world_count(state)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    world_count_policy = str(ambiguity_context.get("refc_world_count_policy", "")).strip()
    bridge_retry_available = any(action.kind == "increase_stochastic_samples" for action in actions)
    sampler_dead_end = bool(
        len(state.frontier) <= 1
        and actual_world_count <= 1.0
        and not bridge_retry_available
        and (
            "single_frontier" in world_count_policy
            or bool(ambiguity_context.get("single_frontier_certificate_cap_applied"))
        )
    )
    if not _resample_shortfall_available(state) and not sampler_dead_end:
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if best_search_action is not None:
        search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
        objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
        mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
        predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
        if (objective_gap >= 0.10 or predicted_frontier >= 0.14) and not sampler_dead_end:
            return list(actions)
        if mechanism_gap >= 0.45 and objective_gap >= 0.06 and not sampler_dead_end:
            return list(actions)
        if sampler_dead_end and (objective_gap >= 0.12 or predicted_frontier >= 0.12):
            return list(actions)
        if sampler_dead_end and mechanism_gap >= 0.72 and objective_gap >= 0.09:
            return list(actions)
    return [action for action in actions if action.kind == "stop"]


def _suppress_post_harmful_evidence_drift_search_churn(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    certificate_threshold = _as_float(config.certificate_threshold)
    if current_certificate >= certificate_threshold:
        return list(actions)
    if current_certificate < max(0.72, certificate_threshold - 0.10):
        return list(actions)
    if not _recent_harmful_evidence_certificate_drift(state):
        return list(actions)
    if not _support_rich_ambiguity_window(state):
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    top_refresh_gain = max(0.0, _as_float(state.top_refresh_gain))
    top_fragility_mass = _clamp01(state.top_fragility_mass)
    competitor_pressure = _clamp01(state.competitor_pressure)
    if top_refresh_gain < 0.20 and top_fragility_mass < 0.15 and competitor_pressure < 0.75:
        return list(actions)
    structured_negative_refresh = any(
        action.kind == "refresh_top1_vor"
        and bool((action.metadata if isinstance(action.metadata, Mapping) else {}).get("structured_refresh_signal"))
        and _as_float(
            (action.metadata if isinstance(action.metadata, Mapping) else {}).get(
                "empirical_refresh_certificate_delta"
            ),
            float("nan"),
        )
        < -1e-9
        for action in actions
    )
    if not structured_negative_refresh:
        return list(actions)
    return [action for action in actions if action.kind == "stop"]


def _suppress_settled_certified_revelation_only_actions(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    saturation_certificate = min(
        1.0,
        max(0.95, _as_float(config.certificate_threshold) + 0.20),
    )
    if current_certificate < saturation_certificate:
        return list(actions)
    if len(state.frontier) > 2:
        return list(actions)
    if max(0.0, _as_float(state.certificate_margin)) < 0.18:
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    refresh_entry = _best_refresh_action_entry(actions)
    if refresh_entry is None:
        return list(actions)
    _, refresh_action = refresh_entry
    refresh_metadata = refresh_action.metadata if isinstance(refresh_action.metadata, Mapping) else {}
    if not bool(refresh_metadata.get("structured_refresh_signal")):
        return list(actions)
    if _as_float(refresh_metadata.get("empirical_refresh_certificate_uplift"), float("nan")) > 1e-9:
        return list(actions)
    strong_winner_side_evidence_signal = bool(
        max(0.0, _as_float(state.top_refresh_gain)) >= 0.10
        or _clamp01(state.top_fragility_mass) >= 0.15
        or _clamp01(state.competitor_pressure) >= 0.75
    )
    supported_hard_case_reopen = bool(
        _support_rich_ambiguity_window(state)
        and strong_winner_side_evidence_signal
        and (
            max(0.0, _as_float(state.search_completeness_gap)) >= 0.12
            or _clamp01(state.pending_challenger_mass) >= 0.35
            or _clamp01(state.best_pending_flip_probability) >= 0.40
            or max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)) >= 0.20
        )
    )
    credible_controller_hard_case_reopen = bool(
        bool(state.credible_search_uncertainty)
        and strong_winner_side_evidence_signal
        and (
            _clamp01(state.support_richness) >= 0.50
            or _clamp01(state.ambiguity_pressure) >= 0.50
            or _clamp01(state.prior_support_strength) >= 0.25
        )
        and (
            max(0.0, _as_float(state.search_completeness_gap)) >= 0.20
            or _clamp01(state.pending_challenger_mass) >= 0.55
            or _clamp01(state.best_pending_flip_probability) >= 0.95
            or max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)) >= 0.45
        )
    )
    if supported_hard_case_reopen or credible_controller_hard_case_reopen:
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if _trace_has_prior_evidence_action_attempt(state):
        if best_search_action is not None and (
            _refine_action_has_genuine_novel_search_promise(best_search_action, state=state)
            or _search_action_shows_strong_decision_movement(best_search_action, state=state)
        ):
            return list(actions)
        return list(actions)
    return [action for action in actions if action.kind == "stop"]


def _search_action_has_substantial_post_evidence_support(
    action: VOIAction | None,
    *,
    state: VOIControllerState,
) -> bool:
    if action is None or action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return False
    objective_gap = _clamp01(action.metadata.get("normalized_objective_gap"))
    overlap_gain = _clamp01(action.metadata.get("normalized_overlap_reduction"))
    predicted_frontier = max(0.0, _as_float(action.predicted_delta_frontier))
    flip_probability = _clamp01(action.metadata.get("mean_flip_probability"))
    pending_signal = max(
        _clamp01(state.pending_challenger_mass),
        _clamp01(state.best_pending_flip_probability),
        max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)),
    )
    return bool(
        objective_gap >= 0.08
        or predicted_frontier >= 0.10
        or (
            objective_gap >= 0.05
            and overlap_gain >= 0.40
            and (flip_probability >= 0.35 or pending_signal >= 0.40)
        )
    )


def _suppress_post_evidence_certified_search_backslide(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    certificate_threshold = _as_float(config.certificate_threshold)
    saturation_certificate = min(
        1.0,
        max(0.95, certificate_threshold + 0.20),
    )
    support_rich_direct_fallback_backslide = bool(
        current_certificate >= certificate_threshold
        and _selected_route_uses_cached_direct_fallback(state)
        and _support_rich_ambiguity_window(state)
        and max(0.0, _as_float(state.top_refresh_gain)) >= 0.18
        and _clamp01(state.top_fragility_mass) >= 0.12
        and _clamp01(state.competitor_pressure) >= 0.25
    )
    if current_certificate < saturation_certificate and not support_rich_direct_fallback_backslide:
        return list(actions)
    if not _trace_has_prior_meaningful_evidence_certificate_lift(state):
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.04):
        return list(actions)
    refresh_entry = _best_refresh_action_entry(actions)
    refresh_action = refresh_entry[1] if refresh_entry is not None else None
    refresh_metadata = (
        refresh_action.metadata
        if refresh_action is not None and isinstance(refresh_action.metadata, Mapping)
        else {}
    )
    if (
        refresh_action is not None
        and bool(refresh_metadata.get("structured_refresh_signal"))
        and _as_float(refresh_metadata.get("empirical_refresh_certificate_uplift"), float("nan")) > 1e-9
    ):
        return [
            action
            for action in actions
            if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}
        ]
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    evidence_actions_available = any(
        action.kind in {"refresh_top1_vor", "increase_stochastic_samples"}
        for action in actions
    )
    if best_search_action is not None and not evidence_actions_available:
        search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
        objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
        mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
        overlap_gain = _clamp01(search_metadata.get("normalized_overlap_reduction"))
        predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
        modest_post_evidence_reopen = bool(
            objective_gap < 0.12
            and predicted_frontier < 0.16
            and not (objective_gap >= 0.08 and predicted_frontier >= 0.10 and overlap_gain >= 0.75)
            and not (mechanism_gap >= 0.45 and predicted_frontier >= 0.12)
        )
        if modest_post_evidence_reopen:
            return [action for action in actions if action.kind == "stop"]
    if _search_action_has_substantial_post_evidence_support(best_search_action, state=state):
        return list(actions)
    return [action for action in actions if action.kind == "stop"]


def _suppress_saturated_certified_search_without_certificate_upside(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    certified_frontier_fill_bridge: bool = False,
) -> list[VOIAction]:
    saturation_certificate = min(
        1.0,
        max(0.95, _as_float(config.certificate_threshold) + 0.20),
    )
    if current_certificate < saturation_certificate:
        return list(actions)
    if certified_frontier_fill_bridge:
        return list(actions)
    if len(state.frontier) > 2:
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    if max(0.0, _as_float(state.certificate_margin)) < 0.18:
        return list(actions)
    if _trace_has_prior_meaningful_evidence_certificate_lift(state):
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if best_search_action is None:
        return list(actions)
    if max(0.0, _as_float(best_search_action.predicted_delta_certificate)) > 1e-9:
        return list(actions)
    search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
    if objective_gap >= 0.12 or predicted_frontier >= 0.12:
        return list(actions)
    refresh_entry = _best_refresh_action_entry(actions)
    refresh_action = refresh_entry[1] if refresh_entry is not None else None
    refresh_metadata = (
        refresh_action.metadata
        if refresh_action is not None and isinstance(refresh_action.metadata, Mapping)
        else {}
    )
    if (
        refresh_action is not None
        and _as_float(refresh_metadata.get("empirical_refresh_certificate_uplift"), float("nan")) > 1e-9
    ):
        return list(actions)
    strong_winner_side_signal = bool(
        max(0.0, _as_float(state.top_refresh_gain)) >= 0.10
        or _clamp01(state.top_fragility_mass) >= 0.15
        or _clamp01(state.competitor_pressure) >= 0.75
    )
    supported_hard_case_reopen = bool(
        _support_rich_ambiguity_window(state)
        and strong_winner_side_signal
        and (
            max(0.0, _as_float(state.search_completeness_gap)) >= 0.12
            or _clamp01(state.pending_challenger_mass) >= 0.35
            or _clamp01(state.best_pending_flip_probability) >= 0.40
            or max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)) >= 0.20
            or (
                len(state.frontier) >= 2
                and _clamp01(state.near_tie_mass) >= max(_as_float(config.near_tie_threshold), 0.05)
            )
        )
    )
    search_only_reveal_reopen = bool(
        supported_hard_case_reopen
        and not _trace_has_prior_evidence_action_attempt(state)
        and refresh_action is None
        and mechanism_gap >= 0.35
        and objective_gap < 0.05
        and predicted_frontier < 0.08
    )
    if search_only_reveal_reopen:
        return [action for action in actions if action.kind == "stop"]
    if supported_hard_case_reopen:
        return list(actions)
    if not strong_winner_side_signal:
        return list(actions)
    return [action for action in actions if action.kind == "stop"]


def _suppress_saturated_certified_zero_headroom_search_probe(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
    certified_frontier_fill_bridge: bool = False,
) -> list[VOIAction]:
    saturation_certificate = min(
        1.0,
        max(0.95, _as_float(config.certificate_threshold) + 0.20),
    )
    if current_certificate < saturation_certificate:
        return list(actions)
    if len(state.frontier) > 3:
        return list(actions)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return list(actions)
    if max(0.0, _as_float(state.certificate_margin)) < 0.18:
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if best_search_action is None:
        return list(actions)
    search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
    if max(0.0, _as_float(best_search_action.predicted_delta_certificate)) > 1e-9:
        return list(actions)
    if not bool(search_metadata.get("certificate_headroom_cap_applied")):
        return list(actions)
    if max(0.0, _as_float(search_metadata.get("certificate_headroom_remaining"), 0.0)) > 1e-9:
        return list(actions)
    if certified_frontier_fill_bridge:
        corridor_recall = _clamp01(
            search_metadata.get("corridor_family_recall", state.corridor_family_recall)
        )
        if corridor_recall <= 0.25:
            return list(actions)
    preserve_failed_reopen = bool(
        (
            _recent_no_gain_evidence_discovery_bridge(state)
            or _recent_no_gain_refine_streak(state) > 0
        )
        and (
            _refine_action_has_genuine_novel_search_promise(best_search_action, state=state)
            or _search_action_shows_strong_decision_movement(best_search_action, state=state)
        )
    )
    if preserve_failed_reopen:
        return list(actions)
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
    evidence_actions = [
        action
        for action in actions
        if action.kind in {"refresh_top1_vor", "increase_stochastic_samples"}
    ]
    support_rich_direct_fallback_zero_headroom_probe = bool(
        _selected_route_uses_cached_direct_fallback(state)
        and _support_rich_ambiguity_window(state)
        and len(state.frontier) <= 3
        and objective_gap < 0.22
        and predicted_frontier < 0.20
        and mechanism_gap < 0.30
        and not certified_frontier_fill_bridge
    )
    if support_rich_direct_fallback_zero_headroom_probe:
        if evidence_actions:
            return [
                action
                for action in actions
                if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}
            ]
        return [action for action in actions if action.kind == "stop"]
    if objective_gap >= 0.05 or predicted_frontier >= 0.05:
        if evidence_actions:
            strong_winner_side_signal = bool(
                max(0.0, _as_float(state.top_refresh_gain)) >= 0.10
                or _clamp01(state.top_fragility_mass) >= 0.15
                or _clamp01(state.competitor_pressure) >= 0.75
            )
            if not strong_winner_side_signal:
                return list(actions)
            preserve_search_reopen = bool(
                objective_gap >= 0.18
                or predicted_frontier >= 0.18
                or (objective_gap >= 0.10 and predicted_frontier >= 0.10 and mechanism_gap >= 0.45)
            )
            if not preserve_search_reopen:
                return [
                    action
                    for action in actions
                    if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}
                ]
        return list(actions)
    if any(action.kind in {"refresh_top1_vor", "increase_stochastic_samples"} for action in actions):
        return [
            action
            for action in actions
            if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}
        ]
    return [action for action in actions if action.kind == "stop"]


def _has_direct_fallback_source_marker(value: object) -> bool:
    marker = str(value or "").strip().lower()
    return "direct_k_raw_fallback" in marker


def _selected_route_uses_cached_direct_fallback(state: "VOIControllerState") -> bool:
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    if bool(ambiguity_context.get("supplemental_challenger_activated")):
        return False
    source_markers = (
        ambiguity_context.get("selected_candidate_source_stage"),
        ambiguity_context.get("selected_final_route_source_stage"),
        ambiguity_context.get("selected_candidate_source_label"),
        ambiguity_context.get("selected_final_route_source_label"),
    )
    return any(_has_direct_fallback_source_marker(marker) for marker in source_markers)


def _support_backed_fallback_is_structurally_thin(state: "VOIControllerState") -> bool:
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    if not bool(ambiguity_context.get("graph_supported_ambiguity_fast_fallback")):
        return False

    candidate_path_count = int(max(0.0, _as_float(ambiguity_context.get("od_candidate_path_count"), 0.0)))
    corridor_family_count = int(max(0.0, _as_float(ambiguity_context.get("od_corridor_family_count"), 0.0)))
    single_frontier_shortcut = "single_frontier_shortcut" in str(
        ambiguity_context.get("refc_world_count_policy") or ""
    )
    low_diversity_counts = bool(
        candidate_path_count > 0
        and candidate_path_count <= 2
        and corridor_family_count > 0
        and corridor_family_count <= 2
    )
    low_diversity_single_frontier = bool(
        len(state.frontier) <= 1
        and (
            (candidate_path_count > 0 and candidate_path_count <= 2)
            or (corridor_family_count > 0 and corridor_family_count <= 2)
            or (
                not bool(ambiguity_context.get("supplemental_challenger_activated", False))
                and _clamp01(state.corridor_family_recall) >= 0.65
                and _clamp01(state.frontier_recall_at_budget) >= 0.65
            )
        )
    )
    return bool(single_frontier_shortcut or low_diversity_counts or low_diversity_single_frontier)


def _suppress_cached_direct_fallback_search_churn(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    if not _selected_route_uses_cached_direct_fallback(state):
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if best_search_action is None:
        return list(actions)
    search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
    threshold = _as_float(config.certificate_threshold)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    search_churn_shortcut = bool(
        ambiguity_context.get("graph_low_ambiguity_fast_path")
        or _support_backed_fallback_is_structurally_thin(state)
    )
    remove_search_only = [
        action
        for action in actions
        if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}
    ]

    if (
        current_certificate >= threshold
        and search_churn_shortcut
        and len(state.frontier) <= 2
        and max(0.0, _as_float(best_search_action.predicted_delta_certificate)) <= 1e-9
    ):
        if not (
            _refine_action_has_genuine_novel_search_promise(best_search_action, state=state)
            and _search_action_shows_strong_decision_movement(best_search_action, state=state)
        ):
            return remove_search_only or [action for action in actions if action.kind == "stop"]

    near_certified_floor = max(0.75, threshold - 0.05)
    very_near_threshold = bool(current_certificate >= max(0.79, threshold - 0.03))
    direct_fallback_objective_cap = 0.12 if very_near_threshold else 0.10
    no_winner_side_signal = bool(
        max(0.0, _as_float(state.top_refresh_gain)) <= 1e-9
        and _clamp01(state.top_fragility_mass) <= 0.01
        and _clamp01(state.competitor_pressure) <= 0.05
    )
    shortcut_single_frontier_reopen = bool(
        current_certificate >= max(0.75, threshold - 0.08)
        and len(state.frontier) <= 1
        and "single_frontier_shortcut" in str(ambiguity_context.get("refc_world_count_policy") or "")
        and int(max(0.0, _as_float(ambiguity_context.get("od_candidate_path_count"), 0.0))) <= 1
        and int(max(0.0, _as_float(ambiguity_context.get("od_corridor_family_count"), 0.0))) <= 1
        and _clamp01(state.corridor_family_recall) >= 0.65
        and _clamp01(state.frontier_recall_at_budget) >= 0.65
        and no_winner_side_signal
    )
    if shortcut_single_frontier_reopen:
        return remove_search_only or [action for action in actions if action.kind == "stop"]
    support_rich_zero_signal_near_threshold_probe = bool(
        near_certified_floor <= current_certificate < threshold
        and len(state.frontier) <= 1
        and no_winner_side_signal
        and _support_rich_ambiguity_window(state)
        and objective_gap < min(0.08, direct_fallback_objective_cap)
        and predicted_frontier < 0.14
        and mechanism_gap < 0.40
    )
    if support_rich_zero_signal_near_threshold_probe:
        return remove_search_only or [action for action in actions if action.kind == "stop"]
    if (
        near_certified_floor <= current_certificate < threshold
        and len(state.frontier) <= 1
        and _clamp01(state.near_tie_mass) <= max(_as_float(config.near_tie_threshold), 0.03)
        and no_winner_side_signal
        and objective_gap < direct_fallback_objective_cap
        and predicted_frontier < 0.16
        and not ((not very_near_threshold) and mechanism_gap >= 0.55 and predicted_frontier >= 0.08)
    ):
        return remove_search_only or [action for action in actions if action.kind == "stop"]
    return list(actions)


def _suppress_post_nonproductive_uncertified_evidence_plateau_churn(
    actions: Sequence[VOIAction],
    *,
    state: VOIControllerState,
    current_certificate: float,
    config: VOIConfig,
) -> list[VOIAction]:
    if current_certificate >= _as_float(config.certificate_threshold):
        return list(actions)
    if not _recent_no_gain_evidence_action(state):
        return list(actions)
    if _clamp01(state.near_tie_mass) < 0.65:
        return list(actions)
    ambiguity_context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    objective_spread = _clamp01(ambiguity_context.get("od_objective_spread"))
    margin_pressure = _clamp01(ambiguity_context.get("od_ambiguity_margin_pressure"))
    low_ambiguity_prior = max(
        _clamp01(ambiguity_context.get("od_ambiguity_index")),
        _clamp01(ambiguity_context.get("od_ambiguity_prior_strength")),
        _clamp01(ambiguity_context.get("ambiguity_budget_prior")),
    )
    if objective_spread > 0.02 or margin_pressure > 0.02 or low_ambiguity_prior > 0.10:
        return list(actions)
    best_search_action = max(
        (action for action in actions if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}),
        key=lambda action: action.q_score,
        default=None,
    )
    if best_search_action is not None:
        search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
        objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
        mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
        predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
        if objective_gap >= 0.08 or predicted_frontier >= 0.10:
            return list(actions)
        if mechanism_gap >= 0.30 and predicted_frontier >= 0.08:
            return list(actions)
    return [action for action in actions if action.kind == "stop"]


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
    if best_search_action is not None and _refine_action_has_genuine_novel_search_promise(best_search_action, state=state):
        search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
        objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
        predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
        if objective_gap >= 0.18 or predicted_frontier >= 0.18:
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


def _should_offer_uncertified_structural_cap_bridge(
    state: "VOIControllerState",
    *,
    current_certificate: float,
    config: VOIConfig,
    recent_no_gain_controller_streak: int,
    best_search_action: VOIAction | None,
) -> bool:
    if current_certificate >= _as_float(config.certificate_threshold):
        return False
    if state.remaining_evidence_budget <= 0:
        return False
    if len(state.frontier) > 1:
        return False
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    if not bool(context.get("single_frontier_certificate_cap_applied")):
        return False
    if not bool(context.get("single_frontier_requires_full_stress")):
        return False
    empirical_baseline, controller_baseline, actual_world_count, shortfall_ratio = _single_frontier_structural_cap_gap(
        state
    )
    if not math.isfinite(empirical_baseline) or not math.isfinite(controller_baseline):
        return False
    # Once any empirical certification worlds already exist, observed structural-cap
    # bridge resamples have been consistently nonproductive. Keep this bridge only
    # for true zero-world evidence gaps instead of reopening on a cap artifact alone.
    if actual_world_count > 0.0:
        return False
    if empirical_baseline <= max(current_certificate, controller_baseline) + 0.05:
        return False
    if actual_world_count > 0.0 and shortfall_ratio <= 0.02:
        return False
    if _trace_has_prior_evidence_action_attempt(state):
        return False
    if recent_no_gain_controller_streak > 0:
        return False
    if (
        _refine_action_has_genuine_novel_search_promise(best_search_action, state=state)
        and not _structural_cap_bridge_can_override_moderate_search_signal(
            state,
            current_certificate=current_certificate,
            config=config,
            best_search_action=best_search_action,
            shortfall_ratio=shortfall_ratio,
        )
    ):
        return False
    support_signal = max(
        _clamp01(state.prior_support_strength),
        _clamp01(state.support_richness),
        _clamp01(state.ambiguity_pressure),
    )
    search_pressure = _clamp01(
        max(
            max(0.0, _as_float(state.search_completeness_gap)),
            _clamp01(state.pending_challenger_mass),
            _clamp01(state.best_pending_flip_probability),
            max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)),
        )
    )
    if search_pressure < 0.10:
        return False
    hard_case_support = max(
        _clamp01(context.get("od_hard_case_prior")),
        _clamp01(context.get("ambiguity_budget_prior")),
        _clamp01(context.get("od_ambiguity_support_ratio")),
    )
    return bool(
        search_pressure >= 0.35
        or (search_pressure >= 0.20 and support_signal >= 0.45)
        or (
            search_pressure >= 0.12
            and hard_case_support >= 0.70
            and shortfall_ratio >= 0.45
        )
    )


def _structural_cap_bridge_can_override_moderate_search_signal(
    state: "VOIControllerState",
    *,
    current_certificate: float,
    config: VOIConfig,
    best_search_action: VOIAction | None,
    shortfall_ratio: float,
) -> bool:
    if best_search_action is None or best_search_action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
        return False
    if state.stochastic_enabled:
        return False
    if current_certificate >= _as_float(config.certificate_threshold):
        return False
    if shortfall_ratio < 0.20:
        return False
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return False
    search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
    if objective_gap >= 0.08 or predicted_frontier >= 0.12:
        return False
    if mechanism_gap >= 0.45 and objective_gap >= 0.06:
        return False
    return True


def _uncertified_support_rich_zero_signal_bridge_features(
    state: "VOIControllerState",
    *,
    current_certificate: float,
    config: VOIConfig,
    shortfall_ratio_override: float | None = None,
) -> tuple[float, float, float, float, float]:
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    shortfall_ratio = _resample_shortfall_ratio(state)
    if shortfall_ratio_override is not None:
        shortfall_ratio = max(shortfall_ratio, _clamp01(shortfall_ratio_override))
    search_pressure = _clamp01(
        max(
            max(0.0, _as_float(state.search_completeness_gap)),
            _clamp01(state.pending_challenger_mass),
            _clamp01(state.best_pending_flip_probability),
            max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)),
        )
    )
    support_signal = max(
        _clamp01(state.prior_support_strength),
        _clamp01(state.support_richness),
        _clamp01(state.ambiguity_pressure),
        _clamp01(context.get("od_ambiguity_support_ratio")),
        _clamp01(context.get("od_ambiguity_source_entropy")),
    )
    hard_case_support = max(
        _clamp01(context.get("od_hard_case_prior")),
        _clamp01(context.get("ambiguity_budget_prior")),
        _clamp01(context.get("od_ambiguity_support_ratio")),
    )
    threshold_gap = max(0.0, _as_float(config.certificate_threshold) - current_certificate)
    return (
        _clamp01(shortfall_ratio),
        search_pressure,
        _clamp01(support_signal),
        _clamp01(hard_case_support),
        threshold_gap,
    )


def _should_use_cert_world_support_rich_zero_signal_bridge(
    state: "VOIControllerState",
    *,
    fragility: FragilityResult | Any | None = None,
) -> bool:
    if len(state.frontier) > 1:
        return False
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    if not bool(context.get("single_frontier_certificate_cap_applied")):
        return False
    actual_world_count = _actual_refc_world_count(state)
    controller_requested_world_count = _requested_refc_world_count(state)
    sampler_requested_world_count = _sampler_requested_refc_world_count(state)
    if controller_requested_world_count <= actual_world_count + 1e-9:
        return False
    if sampler_requested_world_count > actual_world_count + 1e-9:
        return False
    controller_shortfall = max(0.0, controller_requested_world_count - actual_world_count)
    if controller_shortfall < max(8.0, 0.15 * controller_requested_world_count):
        return False
    fallback_activated, controller_disagreement, controller_top_gain = _controller_refresh_bridge_stats(
        fragility
    )
    empirical_baseline = _as_float(context.get("empirical_baseline_certificate"), float("nan"))
    controller_baseline = _as_float(context.get("controller_baseline_certificate"), float("nan"))
    structural_cap_gap = bool(
        math.isfinite(empirical_baseline)
        and math.isfinite(controller_baseline)
        and empirical_baseline > controller_baseline + 1e-9
    )
    return bool(structural_cap_gap and (fallback_activated or controller_disagreement))


def _should_offer_uncertified_support_rich_zero_signal_bridge(
    state: "VOIControllerState",
    *,
    current_certificate: float,
    config: VOIConfig,
    recent_no_gain_controller_streak: int,
    best_search_action: VOIAction | None,
    fragility: FragilityResult | Any | None = None,
) -> bool:
    if current_certificate >= _as_float(config.certificate_threshold):
        return False
    if state.remaining_evidence_budget <= 0:
        return False
    if recent_no_gain_controller_streak > 0:
        return False
    if _trace_has_prior_evidence_action_attempt(state):
        return False
    support_rich_window = _support_rich_ambiguity_window(state)
    if _clamp01(state.near_tie_mass) > max(_as_float(config.near_tie_threshold), 0.03):
        return False
    if (
        max(0.0, _as_float(state.top_refresh_gain)) > 0.0
        or _clamp01(state.top_fragility_mass) > 0.01
        or _clamp01(state.competitor_pressure) > 0.05
    ):
        return False
    if current_certificate < max(0.72, _as_float(config.certificate_threshold) - 0.10):
        return False
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    fallback_activated, controller_disagreement, controller_top_gain = _controller_refresh_bridge_stats(
        fragility
    )
    single_frontier_structural_cap = bool(
        len(state.frontier) <= 1
        and context.get("single_frontier_certificate_cap_applied")
    )
    cert_world_bridge = _should_use_cert_world_support_rich_zero_signal_bridge(
        state,
        fragility=fragility,
    )
    shortfall_ratio, search_pressure, support_signal, hard_case_support, threshold_gap = (
        _uncertified_support_rich_zero_signal_bridge_features(
            state,
            current_certificate=current_certificate,
            config=config,
            shortfall_ratio_override=(
                _cert_world_shortfall_ratio(state) if cert_world_bridge else None
            ),
        )
    )
    if threshold_gap <= 0.0:
        return False
    if search_pressure < 0.24:
        return False
    if support_signal < 0.48 and hard_case_support < 0.35:
        return False
    actual_world_count = _actual_refc_world_count(state)
    sampler_shortfall_available = _resample_shortfall_available(state)
    if not sampler_shortfall_available and not cert_world_bridge:
        return False
    severe_sampler_undercoverage = bool(
        len(state.frontier) <= 1
        and actual_world_count <= 1.0
        and shortfall_ratio >= 0.60
        and search_pressure >= 0.24
        and (support_signal >= 0.48 or hard_case_support >= 0.35)
    )
    structural_cap_cert_world_window = bool(
        cert_world_bridge
        and single_frontier_structural_cap
        and _stress_world_fraction_from_context(context) >= 0.90
        and search_pressure >= 0.24
        and (support_signal >= 0.42 or hard_case_support >= 0.32)
    )
    if (
        not support_rich_window
        and not severe_sampler_undercoverage
        and not structural_cap_cert_world_window
    ):
        return False
    live_cert_world_bridge = bool(
        cert_world_bridge
        and len(state.frontier) <= 1
        and (shortfall_ratio >= 0.35 or structural_cap_cert_world_window)
        and (
            fallback_activated
            or controller_disagreement
            or single_frontier_structural_cap
        )
    )
    live_sampler_bridge = bool(
        len(state.frontier) <= 1
        and sampler_shortfall_available
        and shortfall_ratio >= 0.35
        and (
            fallback_activated
            or controller_disagreement
            or single_frontier_structural_cap
            or severe_sampler_undercoverage
        )
    )
    if (
        severe_sampler_undercoverage
        and not state.stochastic_enabled
        and not fallback_activated
        and not controller_disagreement
        and not cert_world_bridge
    ):
        return False
    if (
        live_cert_world_bridge
        and fallback_activated
        and controller_disagreement
        and controller_top_gain <= 1e-9
        and threshold_gap < 0.02
    ):
        return False
    if state.stochastic_enabled and not (live_sampler_bridge or live_cert_world_bridge):
        return False
    if shortfall_ratio < 0.25 and actual_world_count > 2.0 and not live_cert_world_bridge:
        return False
    if best_search_action is None:
        return live_sampler_bridge or live_cert_world_bridge
    search_metadata = best_search_action.metadata if isinstance(best_search_action.metadata, Mapping) else {}
    objective_gap = _clamp01(search_metadata.get("normalized_objective_gap"))
    mechanism_gap = _clamp01(search_metadata.get("normalized_mechanism_gap"))
    predicted_frontier = max(0.0, _as_float(best_search_action.predicted_delta_frontier))
    if objective_gap >= 0.16 or predicted_frontier >= 0.20:
        return False
    if mechanism_gap >= 0.35 and objective_gap >= 0.10:
        return False
    return True


def _build_uncertified_support_rich_zero_signal_bridge_action(
    state: "VOIControllerState",
    *,
    current_certificate: float,
    config: VOIConfig,
    fragility: FragilityResult | Any | None = None,
) -> VOIAction:
    actual_world_count = _actual_refc_world_count(state)
    controller_requested_world_count = _requested_refc_world_count(state)
    sampler_requested_world_count = (
        _sampler_requested_refc_world_count(state)
        if state.stochastic_enabled
        else controller_requested_world_count
    )
    fallback_activated, controller_disagreement, controller_top_gain = _controller_refresh_bridge_stats(
        fragility
    )
    cert_world_bridge = _should_use_cert_world_support_rich_zero_signal_bridge(
        state,
        fragility=fragility,
    )
    shortfall_ratio, search_pressure, support_signal, hard_case_support, threshold_gap = (
        _uncertified_support_rich_zero_signal_bridge_features(
            state,
            current_certificate=current_certificate,
            config=config,
            shortfall_ratio_override=(
                _cert_world_shortfall_ratio(state) if cert_world_bridge else None
            ),
        )
    )
    requested_world_count = (
        controller_requested_world_count
        if cert_world_bridge
        else sampler_requested_world_count
    )
    sampler_shortfall_available = _resample_shortfall_available(state)
    severe_sampler_undercoverage = bool(
        len(state.frontier) <= 1
        and actual_world_count <= 1.0
        and shortfall_ratio >= 0.60
        and search_pressure >= 0.24
        and (support_signal >= 0.48 or hard_case_support >= 0.35)
    )
    structural_cap_cert_world_window = bool(
        cert_world_bridge
        and len(state.frontier) <= 1
        and bool(
            len(state.frontier) <= 1
            and (
                state.ambiguity_context.get("single_frontier_certificate_cap_applied")
                if isinstance(state.ambiguity_context, Mapping)
                else False
            )
        )
        and _stress_world_fraction_from_context(
            state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
        ) >= 0.90
        and search_pressure >= 0.24
        and (support_signal >= 0.42 or hard_case_support >= 0.32)
    )
    live_sampler_bridge = bool(
        len(state.frontier) <= 1
        and sampler_shortfall_available
        and shortfall_ratio >= 0.35
        and (
            fallback_activated
            or controller_disagreement
            or bool(
                len(state.frontier) <= 1
                and (
                    state.ambiguity_context.get("single_frontier_certificate_cap_applied")
                    if isinstance(state.ambiguity_context, Mapping)
                    else False
                )
            )
            or severe_sampler_undercoverage
        )
    )
    live_cert_world_bridge = bool(
        cert_world_bridge
        and len(state.frontier) <= 1
        and (shortfall_ratio >= 0.35 or structural_cap_cert_world_window)
        and (
            fallback_activated
            or controller_disagreement
            or bool(
                len(state.frontier) <= 1
                and (
                    state.ambiguity_context.get("single_frontier_certificate_cap_applied")
                    if isinstance(state.ambiguity_context, Mapping)
                    else False
                )
            )
        )
    )
    predicted_delta_certificate = max(
        threshold_gap + 0.02,
        (0.12 * search_pressure)
        + (0.10 * shortfall_ratio)
        + (0.08 * support_signal)
        + (0.06 * hard_case_support),
    )
    if live_sampler_bridge or live_cert_world_bridge:
        predicted_delta_certificate += (
            (0.015 if fallback_activated else 0.0)
            + (0.02 if controller_disagreement else 0.0)
            + (0.01 if live_cert_world_bridge else 0.0)
        )
    predicted_delta_certificate = min(
        max(0.0, 1.0 - _clamp01(current_certificate)),
        predicted_delta_certificate,
    )
    predicted_delta_margin = max(
        0.0,
        (0.45 * predicted_delta_certificate)
        + (0.05 * search_pressure)
        + (0.03 * support_signal),
    )
    predicted_delta_frontier = max(
        0.0,
        (0.05 * search_pressure)
        + (0.04 * shortfall_ratio)
        + (0.02 * max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget))),
    )
    return VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        cost_evidence=1,
        predicted_delta_certificate=predicted_delta_certificate,
        predicted_delta_margin=predicted_delta_margin,
        predicted_delta_frontier=predicted_delta_frontier,
        preconditions=("evidence_budget_available", "uncertified_support_rich_zero_signal_bridge"),
        reason="increase_stochastic_samples_uncertified_support_rich_zero_signal_bridge",
        metadata={
            "evidence_discovery_bridge": True,
            "uncertified_support_rich_zero_signal_bridge": True,
            "actual_world_count": actual_world_count,
            "requested_world_count": requested_world_count,
            "sampler_requested_world_count": sampler_requested_world_count,
            "controller_requested_world_count": controller_requested_world_count,
            "world_shortfall_ratio": shortfall_ratio,
            "sampler_world_shortfall_ratio": _resample_shortfall_ratio(state),
            "cert_world_shortfall_ratio": _cert_world_shortfall_ratio(state),
            "search_pressure": search_pressure,
            "support_signal": support_signal,
            "hard_case_support": hard_case_support,
            "threshold_gap": threshold_gap,
            "sample_increment": config.resample_increment,
            "controller_refresh_fallback_activated": fallback_activated,
            "controller_empirical_vs_raw_refresh_disagreement": controller_disagreement,
            "controller_top_refresh_gain": controller_top_gain,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": live_sampler_bridge,
            "uncertified_support_rich_zero_signal_cert_world_bridge": live_cert_world_bridge,
            "uncertified_support_rich_zero_signal_extreme_undercoverage": severe_sampler_undercoverage,
        },
    )


def _build_uncertified_structural_cap_resample_action(
    state: "VOIControllerState",
    *,
    current_certificate: float,
    config: VOIConfig,
) -> VOIAction:
    empirical_baseline, controller_baseline, actual_world_count, shortfall_ratio = _single_frontier_structural_cap_gap(
        state
    )
    support_signal = max(
        _clamp01(state.prior_support_strength),
        _clamp01(state.support_richness),
        _clamp01(state.ambiguity_pressure),
    )
    search_pressure = _clamp01(
        max(
            max(0.0, _as_float(state.search_completeness_gap)),
            _clamp01(state.pending_challenger_mass),
            _clamp01(state.best_pending_flip_probability),
            max(0.0, 1.0 - _clamp01(state.frontier_recall_at_budget)),
        )
    )
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    hard_case_support = max(
        _clamp01(context.get("od_hard_case_prior")),
        _clamp01(context.get("ambiguity_budget_prior")),
        _clamp01(context.get("od_ambiguity_support_ratio")),
    )
    effective_shortfall = max(
        shortfall_ratio,
        1.0 if actual_world_count <= 0.0 else 0.0,
    )
    recoverable_gap = max(0.0, empirical_baseline - max(current_certificate, controller_baseline))
    predicted_delta_certificate = max(
        0.0,
        recoverable_gap
        * (
            0.25
            + (0.30 * search_pressure)
            + (0.25 * effective_shortfall)
            + (0.15 * support_signal)
            + (0.05 * hard_case_support)
        ),
    )
    predicted_delta_certificate = min(
        max(0.0, 1.0 - _clamp01(current_certificate)),
        predicted_delta_certificate,
    )
    predicted_delta_margin = max(
        0.0,
        (0.35 * predicted_delta_certificate)
        + (0.05 * search_pressure)
        + (0.02 * effective_shortfall),
    )
    predicted_delta_frontier = max(
        0.0,
        (0.02 * effective_shortfall) + (0.04 * search_pressure),
    )
    return VOIAction(
        action_id="resample:stochastic:structural_cap_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        cost_evidence=1,
        predicted_delta_certificate=predicted_delta_certificate,
        predicted_delta_margin=predicted_delta_margin,
        predicted_delta_frontier=predicted_delta_frontier,
        preconditions=("evidence_budget_available", "single_frontier_structural_cap"),
        reason="increase_stochastic_samples_structural_cap_bridge",
        metadata={
            "evidence_discovery_bridge": True,
            "structural_cap_bridge": True,
            "empirical_baseline_certificate": empirical_baseline,
            "controller_baseline_certificate": controller_baseline,
            "actual_world_count": actual_world_count,
            "requested_world_count": _requested_refc_world_count(state),
            "world_shortfall_ratio": effective_shortfall,
            "recoverable_certificate_gap": recoverable_gap,
            "support_signal": support_signal,
            "search_pressure": search_pressure,
            "hard_case_support": hard_case_support,
            "sample_increment": config.resample_increment,
        },
    )


def _cap_action_certificate_headroom(
    action: VOIAction,
    *,
    current_certificate: float,
    config: VOIConfig,
) -> VOIAction:
    predicted_certificate = max(0.0, _as_float(action.predicted_delta_certificate))
    remaining_headroom = max(0.0, 1.0 - _clamp01(current_certificate))
    if predicted_certificate <= remaining_headroom + 1e-9:
        return action
    cost = max(0.0, float(action.cost_search + action.cost_evidence))
    certificate_q_penalty = (
        config.lambda_certificate * (predicted_certificate - remaining_headroom)
    ) / (cost + config.epsilon)
    metadata = dict(action.metadata)
    metadata["certificate_headroom_cap_applied"] = True
    metadata["certificate_headroom_remaining"] = round(remaining_headroom, 6)
    metadata["certificate_headroom_predicted_before_cap"] = round(predicted_certificate, 6)
    metadata["certificate_headroom_q_penalty"] = round(certificate_q_penalty, 6)
    return replace(
        action,
        predicted_delta_certificate=remaining_headroom,
        q_score=max(0.0, _as_float(action.q_score) - certificate_q_penalty),
        metadata=metadata,
    )


def build_action_value_estimate(
    action: VOIAction,
    *,
    config: VOIConfig | None = None,
) -> ActionValueEstimate:
    cfg = config or VOIConfig()
    return _build_replay_action_value_estimate(
        action_id=action.action_id,
        action_kind=action.kind,
        action_target=action.target,
        action_reason=action.reason,
        cost_search=action.cost_search,
        cost_evidence=action.cost_evidence,
        predicted_delta_certificate=action.predicted_delta_certificate,
        predicted_delta_margin=action.predicted_delta_margin,
        predicted_delta_frontier=action.predicted_delta_frontier,
        lambda_certificate=cfg.lambda_certificate,
        lambda_margin=cfg.lambda_margin,
        lambda_frontier=cfg.lambda_frontier,
        epsilon=cfg.epsilon,
        ranked_q_score=_as_float(action.q_score),
        metadata=dict(action.metadata),
    )


def action_scoring_primitives(
    action: VOIAction,
    *,
    config: VOIConfig | None = None,
) -> dict[str, float]:
    estimate = build_action_value_estimate(action, config=config)
    return {
        "predicted_delta_certificate": estimate.predicted_delta_certificate,
        "predicted_delta_margin": estimate.predicted_delta_margin,
        "predicted_delta_frontier": estimate.predicted_delta_frontier,
        "weighted_certificate_value": estimate.weighted_certificate_value,
        "weighted_margin_value": estimate.weighted_margin_value,
        "weighted_frontier_value": estimate.weighted_frontier_value,
        "total_predicted_value": estimate.total_predicted_value,
        "total_cost": estimate.total_cost,
        "base_q_score": estimate.base_q_score,
        "ranked_q_score": estimate.ranked_q_score,
        "q_score_adjustment": estimate.ranked_q_score - estimate.base_q_score,
    }


def _low_ambiguity_fast_path(
    state: VOIControllerState,
    *,
    current_certificate: float | None = None,
) -> bool:
    context = state.ambiguity_context if isinstance(state.ambiguity_context, Mapping) else {}
    explicit_fast_path = bool(
        context.get("graph_low_ambiguity_fast_path")
        or context.get("low_ambiguity_fast_path")
        or context.get("controller_low_ambiguity_fast_path")
    )
    if explicit_fast_path:
        return True
    certificate = _clamp01(current_certificate if current_certificate is not None else _certificate_value_from_map(state.certificate, state.winner_id))
    return bool(
        certificate >= 0.82
        and _clamp01(state.search_completeness_score) >= 0.90
        and _clamp01(state.search_completeness_gap) <= 0.10
        and _clamp01(state.support_richness) >= 0.65
        and _clamp01(state.near_tie_mass) <= 0.08
        and _clamp01(state.pending_challenger_mass) <= 0.12
        and _clamp01(state.best_pending_flip_probability) <= 0.18
    )


def build_controller_replay_summary(
    stop_certificate: VOIStopCertificate,
    *,
    trace_source: str = "voi_controller.run_controller",
) -> ReplayOracleSummary:
    replay_summary = replay_oracle_summary_from_trace_rows(
        stop_certificate.action_trace,
        trace_source=trace_source,
        low_ambiguity_fast_path=bool(stop_certificate.ambiguity_summary.get("low_ambiguity_fast_path", False)),
        trace_metadata={
            "stop_reason": stop_certificate.stop_reason,
            "final_route_id": stop_certificate.final_winner_route_id,
            "final_certificate_value": stop_certificate.certificate_value,
            "search_budget_used": stop_certificate.search_budget_used,
            "evidence_budget_used": stop_certificate.evidence_budget_used,
            "iteration_count": stop_certificate.iteration_count,
        },
        replay_metadata={
            "stop_reason": stop_certificate.stop_reason,
            "certified": stop_certificate.certified,
        },
        oracle_metadata={
            "final_strict_frontier_size": stop_certificate.final_strict_frontier_size,
        },
    )
    return replay_summary


def _build_stop_certificate(
    *,
    state: VOIControllerState,
    fragility: FragilityResult,
    cfg: VOIConfig,
    current_certificate: float,
    certified: bool,
    stop_reason: str,
    best_rejected_action: dict[str, Any] | None,
) -> VOIStopCertificate:
    stop_certificate = VOIStopCertificate(
        final_winner_route_id=state.winner_id,
        final_winner_objective_vector=_winner_objective_vector(state),
        final_strict_frontier_size=len(state.frontier),
        certificate_value=current_certificate,
        certified=certified,
        search_budget_used=cfg.search_budget - state.remaining_search_budget,
        search_budget_remaining=state.remaining_search_budget,
        evidence_budget_used=cfg.evidence_budget - state.remaining_evidence_budget,
        evidence_budget_remaining=state.remaining_evidence_budget,
        stop_reason=stop_reason,
        action_trace=list(state.action_trace),
        state_trace=list(state.state_trace),
        best_rejected_action=best_rejected_action,
        ambiguity_summary={
            **_ambiguity_summary(fragility, state.winner_id),
            "low_ambiguity_fast_path": bool(_low_ambiguity_fast_path(state, current_certificate=current_certificate)),
        },
        iteration_count=state.iteration_index,
        controller_state=state.as_dict(),
    )
    return replace(
        stop_certificate,
        replay_oracle_summary=build_controller_replay_summary(stop_certificate).as_dict(),
    )


def build_action_trace_metadata(
    *,
    state: VOIControllerState,
    current_certificate: float,
    action_menu: Sequence[VOIAction],
    chosen_action: VOIAction | None,
    best_rejected_action: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rejected_payload = dict(best_rejected_action) if isinstance(best_rejected_action, Mapping) else {}
    return {
        "trace_version": 1,
        "trace_source": "voi_controller.run_controller",
        "iteration": int(state.iteration_index),
        "winner_id": state.winner_id,
        "selected_route_id": state.selected_route_id,
        "current_certificate": round(_as_float(current_certificate), 6),
        "certificate_margin": round(_as_float(state.certificate_margin), 6),
        "search_completeness_score": round(_as_float(state.search_completeness_score), 6),
        "support_richness": round(_as_float(state.support_richness), 6),
        "ambiguity_pressure": round(_as_float(state.ambiguity_pressure), 6),
        "remaining_search_budget": int(state.remaining_search_budget),
        "remaining_evidence_budget": int(state.remaining_evidence_budget),
        "used_search_budget": int(state.used_search_budget),
        "used_evidence_budget": int(state.used_evidence_budget),
        "frontier_size": len(state.frontier),
        "action_menu_count": len(action_menu),
        "chosen_action_id": chosen_action.action_id if chosen_action is not None else "",
        "chosen_action_kind": chosen_action.kind if chosen_action is not None else "",
        "best_rejected_action_id": str(rejected_payload.get("action_id", "")).strip(),
        "best_rejected_action_kind": str(rejected_payload.get("kind", "")).strip(),
        "credible_search_uncertainty": bool(state.credible_search_uncertainty),
        "credible_evidence_uncertainty": bool(state.credible_evidence_uncertainty),
        "low_ambiguity_fast_path": bool(_low_ambiguity_fast_path(state, current_certificate=current_certificate)),
    }


def build_action_replay_record(
    action: VOIAction,
    *,
    config: VOIConfig | None = None,
    trace_entry: Mapping[str, Any] | None = None,
    trace_metadata: Mapping[str, Any] | None = None,
    replay_metadata: Mapping[str, Any] | None = None,
    oracle_metadata: Mapping[str, Any] | None = None,
) -> ActionReplayRecord:
    estimate = build_action_value_estimate(action, config=config)
    return _build_replay_action_record(
        estimate,
        trace_entry=trace_entry,
        trace_metadata=trace_metadata,
        replay_metadata=replay_metadata,
        oracle_metadata=oracle_metadata,
    )


def _action_value_trace_payload(
    *,
    state: VOIControllerState,
    current_certificate: float,
    action_menu: Sequence[VOIAction],
    chosen_action: VOIAction | None,
    best_rejected_action: Mapping[str, Any] | None,
    config: VOIConfig | None = None,
) -> dict[str, Any]:
    trace_metadata = build_action_trace_metadata(
        state=state,
        current_certificate=current_certificate,
        action_menu=action_menu,
        chosen_action=chosen_action,
        best_rejected_action=best_rejected_action,
    )
    chosen_action_value_record = (
        build_action_replay_record(
            chosen_action,
            config=config,
            trace_metadata=trace_metadata,
            replay_metadata={
                "record_mode": "predicted_only",
                "controller_trace_source": "voi_controller.run_controller",
            },
        ).as_dict()
        if chosen_action is not None
        else None
    )
    return {
        "trace_metadata": trace_metadata,
        "chosen_action_value_record": chosen_action_value_record,
        "action_menu_value_estimates": [
            build_action_value_estimate(action, config=config).as_dict()
            for action in action_menu
        ],
    }


def score_action(action: VOIAction, *, config: VOIConfig | None = None) -> VOIAction:
    # Thesis controller heuristic: a deterministic value-per-cost ranking over
    # certificate gain, margin gain, and frontier gain. This is a transparent
    # metareasoning surrogate, not a claim of optimal VOI.
    estimate = build_action_value_estimate(action, config=config)
    return replace(action, q_score=float(estimate.base_q_score))


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
    pending_candidates = _pending_dccs_candidates(dccs, state=enriched_state)
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
    initial_certified_frontier_fill_bridge = _certified_frontier_fill_bridge_active(
        state,
        config=cfg,
        current_certificate=current_certificate,
    )
    enriched_certified_frontier_fill_bridge = _certified_frontier_fill_bridge_active(
        enriched_state,
        config=cfg,
        current_certificate=current_certificate,
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
    certified_frontier_fill_bridge = bool(
        initial_certified_frontier_fill_bridge
        or enriched_certified_frontier_fill_bridge
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
        initial_certified_frontier_fill_bridge
        or (
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
        refine_top1_action = _cap_action_certificate_headroom(
            refine_top1_action,
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
            topk_action = _cap_action_certificate_headroom(
                topk_action,
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
        refresh_action = _cap_action_certificate_headroom(
            refresh_action,
            current_certificate=current_certificate,
            config=cfg,
        )
        signed_refresh_delta = _as_float(
            refresh_action.metadata.get("empirical_refresh_certificate_delta"),
            float("nan"),
        )
        allow_negative_certified_refresh_reveal = _allow_certified_negative_refresh_revelation(
            state=enriched_state,
            current_certificate=current_certificate,
            config=cfg,
            signed_refresh_delta=signed_refresh_delta,
            evidence_uncertainty=evidence_uncertainty,
            supported_fragility_uncertainty=supported_fragility_uncertainty,
        )
        if allow_negative_certified_refresh_reveal:
            refresh_metadata = dict(refresh_action.metadata)
            refresh_metadata["negative_empirical_refresh_reveal_allowed"] = True
            refresh_action = replace(refresh_action, metadata=refresh_metadata)
        suppress_negative_certified_refresh_reveal = bool(
            current_certificate >= _as_float(cfg.certificate_threshold)
            and math.isfinite(signed_refresh_delta)
            and signed_refresh_delta < -1e-9
            and not allow_negative_certified_refresh_reveal
        )
        if not suppress_negative_certified_refresh_reveal:
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
            _cap_action_certificate_headroom(
                score_action(
                    _build_resample_action(
                        near_tie_mass=state.near_tie_mass,
                        stress_world_fraction=stress_world_fraction,
                        top_fragility_mass=enriched_state.top_fragility_mass,
                        config=cfg,
                    ),
                    config=cfg,
                ),
                current_certificate=current_certificate,
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
    uncertified_structural_cap_bridge = _should_offer_uncertified_structural_cap_bridge(
        enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        recent_no_gain_controller_streak=recent_no_gain_controller_streak,
        best_search_action=best_search_action,
    )
    uncertified_support_rich_zero_signal_bridge = _should_offer_uncertified_support_rich_zero_signal_bridge(
        enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        recent_no_gain_controller_streak=recent_no_gain_controller_streak,
        best_search_action=best_search_action,
        fragility=fragility,
    )
    evidence_discovery_bridge = bool(
        evidence_discovery_bridge
        or uncertified_structural_cap_bridge
        or uncertified_support_rich_zero_signal_bridge
    )
    if evidence_discovery_bridge and not any(action.kind == "increase_stochastic_samples" for action in actions):
        if uncertified_structural_cap_bridge:
            bridge_action = _build_uncertified_structural_cap_resample_action(
                enriched_state,
                current_certificate=current_certificate,
                config=cfg,
            )
        elif uncertified_support_rich_zero_signal_bridge:
            bridge_action = _build_uncertified_support_rich_zero_signal_bridge_action(
                enriched_state,
                current_certificate=current_certificate,
                config=cfg,
                fragility=fragility,
            )
        else:
            bridge_action = _build_resample_action(
                near_tie_mass=state.near_tie_mass,
                stress_world_fraction=stress_world_fraction,
                top_fragility_mass=enriched_state.top_fragility_mass,
                config=cfg,
            )
        bridge_metadata = dict(bridge_action.metadata)
        bridge_metadata["evidence_discovery_bridge"] = True
        bridge_metadata["stochastic_bridge_required"] = True
        if uncertified_structural_cap_bridge:
            bridge_metadata["uncertified_structural_cap_bridge"] = True
        if uncertified_support_rich_zero_signal_bridge:
            bridge_metadata["uncertified_support_rich_zero_signal_bridge"] = True
        bridge_action = replace(
            bridge_action,
            preconditions=(
                ("evidence_budget_available", "single_frontier_structural_cap")
                if uncertified_structural_cap_bridge
                else (
                    ("evidence_budget_available", "uncertified_support_rich_zero_signal_bridge")
                    if uncertified_support_rich_zero_signal_bridge
                    else ("evidence_budget_available", "evidence_discovery_bridge")
                )
            ),
            reason=(
                "increase_stochastic_samples_structural_cap_bridge"
                if uncertified_structural_cap_bridge
                else (
                    "increase_stochastic_samples_uncertified_support_rich_zero_signal_bridge"
                    if uncertified_support_rich_zero_signal_bridge
                    else "increase_stochastic_samples_bridge"
                )
            ),
            metadata=bridge_metadata,
        )
        actions.append(
            _cap_action_certificate_headroom(
                score_action(bridge_action, config=cfg),
                current_certificate=current_certificate,
                config=cfg,
            )
        )
    actions = _apply_support_rich_certified_refresh_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
        recent_no_gain_refine_streak=recent_no_gain_refine_streak,
    )
    actions = _apply_strong_winner_side_refresh_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
        recent_no_gain_refine_streak=recent_no_gain_refine_streak,
    )
    actions = _apply_uncertified_evidence_plateau_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
    )
    actions = _apply_uncertified_last_search_token_resample_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
    )
    actions = _apply_uncertified_first_iteration_near_tie_resample_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
    )
    actions = _apply_uncertified_resample_recovery_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
    )
    actions = _apply_uncertified_post_evidence_resample_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
    )
    actions = _apply_uncertified_structural_cap_bridge_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _apply_uncertified_support_rich_zero_signal_bridge_preference(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_certified_zero_signal_controller_churn(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        evidence_discovery_bridge=evidence_discovery_bridge,
    )
    actions = _suppress_certified_single_frontier_zero_signal_search_churn(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_uncertified_stochastic_disabled_zero_signal_controller_churn(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_uncertified_single_frontier_zero_signal_search_churn(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_uncertified_sampler_only_zero_signal_bridge_tail(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_uncertified_low_support_cert_world_zero_signal_bridge_tail(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_post_nonproductive_bridge_zero_signal_search_churn(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_post_nonproductive_uncertified_evidence_plateau_churn(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_post_harmful_evidence_drift_search_churn(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_uncertified_structural_cap_churn(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_post_evidence_certified_search_backslide(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_settled_certified_revelation_only_actions(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    actions = _suppress_saturated_certified_search_without_certificate_upside(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        certified_frontier_fill_bridge=certified_frontier_fill_bridge,
    )
    actions = _suppress_saturated_certified_zero_headroom_search_probe(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
        certified_frontier_fill_bridge=certified_frontier_fill_bridge,
    )
    actions = _suppress_cached_direct_fallback_search_churn(
        actions,
        state=enriched_state,
        current_certificate=current_certificate,
        config=cfg,
    )
    resample_shortfall_available = bool(_resample_shortfall_available(enriched_state))
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
    if actions:
        filtered_actions: list[VOIAction] = []
        for action in actions:
            if action.kind not in {"refine_top1_dccs", "refine_topk_dccs"}:
                filtered_actions.append(action)
                continue
            if _should_stop_uncertified_weak_search_tail(
                action,
                state=enriched_state,
                current_certificate=current_certificate,
                config=cfg,
                evidence_uncertainty=evidence_uncertainty,
            ):
                continue
            if _should_stop_evidence_exhausted_uncertified_search_tail(
                action,
                state=enriched_state,
                current_certificate=current_certificate,
                config=cfg,
            ):
                continue
            filtered_actions.append(action)
        actions = filtered_actions
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
            support_rich_near_tie_evidence_reopen = bool(
                current_certificate >= _as_float(cfg.certificate_threshold)
                and evidence_uncertainty
                and supported_fragility_uncertainty
                and best_evidence_q is not None
                and best_evidence_q >= max(action.q_score, action.q_score * 1.02)
                and recent_no_gain_refine_streak <= 0
                and _clamp01(enriched_state.near_tie_mass) >= max(_as_float(cfg.near_tie_threshold), 0.10)
                and _clamp01(enriched_state.top_fragility_mass) >= 0.18
                and _clamp01(enriched_state.competitor_pressure) >= 0.25
                and objective_gap < 0.08
                and _as_float(action.predicted_delta_frontier) < 0.08
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
            evidence_exhausted_certified_search_tail_stop = _should_stop_evidence_exhausted_certified_search_tail(
                action,
                state=enriched_state,
                current_certificate=current_certificate,
                config=cfg,
            )
            saturated_low_decision_ambiguity_reopen_stop = _should_stop_saturated_certified_low_decision_ambiguity_reopen(
                action,
                state=enriched_state,
                current_certificate=current_certificate,
                config=cfg,
                certified_frontier_fill_bridge=certified_frontier_fill_bridge,
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
            if evidence_discovery_bridge and not genuine_novel_search_promise and not strong_decision_movement:
                continue
            if evidence_exhausted_certified_search_tail_stop:
                continue
            if saturated_low_decision_ambiguity_reopen_stop:
                continue
            if (
                certified_frontier_fill_bridge
                and best_search_action is not None
                and action.action_id == best_search_action.action_id
                and not supported_fragility_uncertainty
                and not evidence_discovery_bridge
                and (
                    genuine_novel_search_promise
                    or strong_decision_movement
                    or predicted_total >= max(search_floor, 0.04)
                )
                and (
                    best_evidence_q is None
                    or best_evidence_q <= max(action.q_score * 1.15, _as_float(cfg.stop_threshold))
                )
            ):
                bridge_metadata = dict(action.metadata)
                bridge_metadata["certified_frontier_fill_bridge_preserved"] = True
                filtered_actions.append(replace(action, metadata=bridge_metadata))
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
            if support_rich_near_tie_evidence_reopen:
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
    if _certified_frontier_fill_bridge_active(
        state,
        config=config,
        current_certificate=current_certificate,
    ):
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


def _certified_frontier_fill_bridge_active(
    state: VOIControllerState,
    *,
    config: VOIConfig,
    current_certificate: float,
) -> bool:
    threshold = _as_float(config.certificate_threshold)
    if current_certificate < threshold or state.remaining_search_budget <= 0:
        return False
    pending_flip = _clamp01(state.best_pending_flip_probability)
    pending_mass = _clamp01(state.pending_challenger_mass)
    support_strength = max(_clamp01(state.prior_support_strength), _clamp01(state.support_richness))
    prior_strength = _prior_strength_from_context(state.ambiguity_context)
    completeness_gap = max(0.0, _as_float(state.search_completeness_gap))
    near_tie = _clamp01(state.near_tie_mass)
    certificate_margin = max(0.0, _as_float(state.certificate_margin))
    frontier_recall = _clamp01(state.frontier_recall_at_budget)
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
    winner_side_search_fill_signal = bool(
        max(0.0, _as_float(state.top_refresh_gain)) >= 0.08
        or _clamp01(state.top_fragility_mass) >= 0.10
        or _clamp01(state.competitor_pressure) >= 0.20
    )
    return bool(
        current_certificate >= threshold
        and not _trace_has_prior_evidence_action_attempt(state)
        and len(state.frontier) <= 2
        and near_tie <= 0.03
        and certificate_margin >= 0.12
        and frontier_recall <= 0.75
        and search_incompleteness_signal >= 0.18
        and support_strength >= 0.22
        and winner_side_search_fill_signal
        and not support_rich_ambiguity
    )


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
    config: VOIConfig | None = None,
) -> dict[str, Any]:
    value_trace_payload = _action_value_trace_payload(
        state=state,
        current_certificate=current_certificate,
        action_menu=feasible_actions,
        chosen_action=chosen_action,
        best_rejected_action=best_rejected_action,
        config=config,
    )
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
        **value_trace_payload,
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
            return _build_stop_certificate(
                state=state,
                fragility=fragility,
                cfg=cfg,
                current_certificate=current_certificate,
                certified=True,
                stop_reason="certified",
                best_rejected_action=best_rejected_action,
            )
        if state.remaining_search_budget <= 0 and state.remaining_evidence_budget <= 0:
            return _build_stop_certificate(
                state=state,
                fragility=fragility,
                cfg=cfg,
                current_certificate=current_certificate,
                certified=False,
                stop_reason="budget_exhausted",
                best_rejected_action=best_rejected_action,
            )

        actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=cfg)
        feasible_actions = [action for action in actions if action.kind != "stop"]
        if not feasible_actions:
            return _build_stop_certificate(
                state=state,
                fragility=fragility,
                cfg=cfg,
                current_certificate=current_certificate,
                certified=False,
                stop_reason=(
                    "search_incomplete_no_action_worth_it"
                    if state.search_completeness_score < cfg.search_completeness_threshold
                    else "no_action_worth_it"
                ),
                best_rejected_action=best_rejected_action,
            )
        top_action = feasible_actions[0]
        if top_action.q_score < cfg.stop_threshold:
            best_rejected_action = top_action.as_dict()
            return _build_stop_certificate(
                state=state,
                fragility=fragility,
                cfg=cfg,
                current_certificate=current_certificate,
                certified=False,
                stop_reason=(
                    "search_incomplete_no_action_worth_it"
                    if state.search_completeness_score < cfg.search_completeness_threshold
                    else "no_action_worth_it"
                ),
                best_rejected_action=best_rejected_action,
            )
        if hooks is None and top_action.kind != "stop":
            failed_best_rejected = feasible_actions[1].as_dict() if len(feasible_actions) > 1 else None
            failure_value_trace_payload = _action_value_trace_payload(
                state=state,
                current_certificate=current_certificate,
                action_menu=actions,
                chosen_action=top_action,
                best_rejected_action=failed_best_rejected,
                config=cfg,
            )
            failure_trace_entry = {
                "iteration": state.iteration_index,
                "feasible_actions": [action.as_dict() for action in actions],
                "chosen_action": top_action.as_dict(),
                "certificate_value": current_certificate,
                "remaining_search_budget": state.remaining_search_budget,
                "remaining_evidence_budget": state.remaining_evidence_budget,
                "best_rejected_action": failed_best_rejected,
                **failure_value_trace_payload,
            }
            failure_state_snapshot = _state_snapshot(
                state,
                current_certificate=current_certificate,
                feasible_actions=actions,
                chosen_action=top_action,
                best_rejected_action=failed_best_rejected,
                config=cfg,
            )
            state = replace(
                state,
                action_trace=[*state.action_trace, failure_trace_entry],
                state_trace=[*state.state_trace, failure_state_snapshot],
            )
            return _build_stop_certificate(
                state=state,
                fragility=fragility,
                cfg=cfg,
                current_certificate=current_certificate,
                certified=False,
                stop_reason="error_missing_action_hooks",
                best_rejected_action=failed_best_rejected,
            )

        best_rejected_action = feasible_actions[1].as_dict() if len(feasible_actions) > 1 else None
        previous_winner_id = state.winner_id
        previous_frontier_ids = {_route_id(route) for route in state.frontier}
        previous_certificate = current_certificate
        value_trace_payload = _action_value_trace_payload(
            state=state,
            current_certificate=current_certificate,
            action_menu=actions,
            chosen_action=top_action,
            best_rejected_action=best_rejected_action,
            config=cfg,
        )
        trace_entry = {
            "iteration": state.iteration_index,
            "feasible_actions": [action.as_dict() for action in actions],
            "chosen_action": top_action.as_dict(),
            "certificate_value": current_certificate,
            "remaining_search_budget": state.remaining_search_budget,
            "remaining_evidence_budget": state.remaining_evidence_budget,
            "best_rejected_action": best_rejected_action,
            **value_trace_payload,
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
                    config=cfg,
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
    return _build_stop_certificate(
        state=state,
        fragility=fragility,
        cfg=cfg,
        current_certificate=current_certificate,
        certified=current_certificate >= cfg.certificate_threshold,
        stop_reason=stop_reason,
        best_rejected_action=best_rejected_action,
    )
