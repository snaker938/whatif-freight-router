from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .decision_critical import DCCSResult, DCCSCandidateRecord
from .evidence_certification import FragilityResult


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


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


@dataclass(frozen=True)
class VOIConfig:
    certificate_threshold: float = 0.67
    stop_threshold: float = 0.05
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
    active_evidence_families: list[str] = field(default_factory=list)
    near_tie_mass: float = 0.0

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
            "active_evidence_families": list(self.active_evidence_families),
            "near_tie_mass": self.near_tie_mass,
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
    best_rejected_action: dict[str, Any] | None
    ambiguity_summary: dict[str, Any]

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


def _best_vor_family(fragility: FragilityResult) -> str | None:
    ranking = fragility.value_of_refresh.get("ranking", [])
    if not isinstance(ranking, list) or not ranking:
        return None
    first = ranking[0]
    if not isinstance(first, Mapping):
        return None
    family = str(first.get("family", "")).strip()
    return family or None


def _winner_objective_vector(state: VOIControllerState) -> tuple[float, float, float]:
    for route in state.frontier:
        if _route_id(route) == state.winner_id:
            return _objective_vector(route)
    if state.frontier:
        return _objective_vector(state.frontier[0])
    return (0.0, 0.0, 0.0)


def _build_refine_action(
    candidate: DCCSCandidateRecord,
    *,
    kind: str,
    top_k: int = 1,
    config: VOIConfig,
) -> VOIAction:
    predicted_delta_certificate = max(
        0.0,
        (0.35 * candidate.flip_probability) + (0.30 * candidate.objective_gap) + (0.15 * candidate.mechanism_gap),
    )
    predicted_delta_margin = max(
        0.0,
        (0.40 * candidate.objective_gap) + (0.20 * (1.0 - candidate.overlap)),
    )
    predicted_delta_frontier = max(0.0, candidate.objective_gap)
    cost_search = max(1, int(top_k))
    return VOIAction(
        action_id=f"{kind}:{candidate.candidate_id}",
        kind=kind,
        target=candidate.candidate_id,
        cost_search=cost_search,
        predicted_delta_certificate=predicted_delta_certificate,
        predicted_delta_margin=predicted_delta_margin,
        predicted_delta_frontier=predicted_delta_frontier,
        preconditions=("search_budget_available", "dccs_candidate_available"),
        reason="refine_candidate",
        metadata={"top_k": top_k, "predicted_refine_cost": candidate.predicted_refine_cost},
    )


def _build_refresh_action(
    family: str,
    *,
    fragility: FragilityResult,
    config: VOIConfig,
) -> VOIAction:
    ranking = fragility.value_of_refresh.get("ranking", [])
    vor_gain = 0.0
    if isinstance(ranking, list):
        for row in ranking:
            if isinstance(row, Mapping) and str(row.get("family", "")) == family:
                vor_gain = _as_float(row.get("vor"))
                break
    return VOIAction(
        action_id=f"refresh:{family}",
        kind="refresh_top1_vor",
        target=family,
        cost_evidence=1,
        predicted_delta_certificate=max(0.0, vor_gain),
        predicted_delta_margin=max(0.0, vor_gain * 0.5),
        predicted_delta_frontier=0.0,
        preconditions=("evidence_budget_available", "vor_available"),
        reason="refresh_evidence_family",
        metadata={"vor_gain": vor_gain},
    )


def _build_resample_action(
    *,
    near_tie_mass: float,
    config: VOIConfig,
) -> VOIAction:
    predicted_delta_certificate = max(0.0, 0.20 * near_tie_mass)
    return VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        cost_evidence=1,
        predicted_delta_certificate=predicted_delta_certificate,
        predicted_delta_margin=max(0.0, 0.10 * near_tie_mass),
        predicted_delta_frontier=max(0.0, 0.05 * near_tie_mass),
        preconditions=("evidence_budget_available", "near_tie_set_nonempty"),
        reason="increase_stochastic_samples",
        metadata={"near_tie_mass": near_tie_mass, "sample_increment": config.resample_increment},
    )


def score_action(action: VOIAction, *, config: VOIConfig | None = None) -> VOIAction:
    cfg = config or VOIConfig()
    cost = max(0.0, float(action.cost_search + action.cost_evidence))
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
    actions: list[VOIAction] = []
    best_candidate = _best_candidate(dccs.selected, config=cfg)
    if best_candidate is not None and state.remaining_search_budget > 0:
        actions.append(score_action(_build_refine_action(best_candidate, kind="refine_top1_dccs", config=cfg), config=cfg))
        top_k = min(cfg.top_k_refine, len(dccs.selected), state.remaining_search_budget)
        if top_k > 1:
            aggregate = replace(
                best_candidate,
                final_score=best_candidate.final_score + 0.10 * (top_k - 1),
                flip_probability=min(1.0, best_candidate.flip_probability + 0.05 * (top_k - 1)),
                objective_gap=best_candidate.objective_gap + 0.05 * (top_k - 1),
            )
            actions.append(
                score_action(
                    _build_refine_action(aggregate, kind="refine_topk_dccs", top_k=top_k, config=cfg),
                    config=cfg,
                )
            )
    best_family = _best_vor_family(fragility)
    if best_family is not None and state.remaining_evidence_budget > 0:
        actions.append(score_action(_build_refresh_action(best_family, fragility=fragility, config=cfg), config=cfg))
    if state.remaining_evidence_budget > 0 and state.near_tie_mass > 0.0:
        actions.append(score_action(_build_resample_action(near_tie_mass=state.near_tie_mass, config=cfg), config=cfg))
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
    if action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
        if hooks is not None and hooks.refine is not None:
            return hooks.refine(state, action)
        return replace(
            state,
            iteration_index=state.iteration_index + 1,
            remaining_search_budget=max(0, state.remaining_search_budget - action.cost_search),
        )
    if action.kind == "refresh_top1_vor":
        if hooks is not None and hooks.refresh is not None:
            return hooks.refresh(state, action)
        return replace(
            state,
            iteration_index=state.iteration_index + 1,
            remaining_evidence_budget=max(0, state.remaining_evidence_budget - action.cost_evidence),
        )
    if action.kind == "increase_stochastic_samples":
        if hooks is not None and hooks.resample is not None:
            return hooks.resample(state, action)
        return replace(
            state,
            iteration_index=state.iteration_index + 1,
            remaining_evidence_budget=max(0, state.remaining_evidence_budget - action.cost_evidence),
        )
    return state


def run_controller(
    *,
    initial_frontier: Sequence[Mapping[str, Any]],
    dccs: DCCSResult,
    fragility: FragilityResult,
    winner_id: str,
    certificate_value: float,
    selected_route_id: str | None = None,
    config: VOIConfig | None = None,
    hooks: VOIActionHooks | None = None,
) -> VOIStopCertificate:
    cfg = config or VOIConfig()
    state = VOIControllerState(
        iteration_index=0,
        frontier=list(initial_frontier),
        certificate={winner_id: float(certificate_value)},
        winner_id=winner_id,
        selected_route_id=selected_route_id or winner_id,
        remaining_search_budget=max(0, int(cfg.search_budget)),
        remaining_evidence_budget=max(0, int(cfg.evidence_budget)),
        action_trace=[],
        active_evidence_families=[
            str(item.get("family", "")).strip()
            for item in fragility.value_of_refresh.get("ranking", [])
            if isinstance(item, Mapping) and str(item.get("family", "")).strip()
        ],
        near_tie_mass=_near_tie_mass({winner_id: float(certificate_value)}, winner_id=winner_id, threshold=cfg.near_tie_threshold),
    )
    best_rejected_action: dict[str, Any] | None = None

    while state.iteration_index < cfg.max_iterations:
        current_certificate = _as_float(state.certificate.get(state.winner_id, certificate_value))
        if current_certificate >= cfg.certificate_threshold:
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
                best_rejected_action=best_rejected_action,
                ambiguity_summary=fragility.value_of_refresh,
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
                best_rejected_action=best_rejected_action,
                ambiguity_summary=fragility.value_of_refresh,
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
                stop_reason="no_action_worth_it",
                action_trace=list(state.action_trace),
                best_rejected_action=best_rejected_action,
                ambiguity_summary=fragility.value_of_refresh,
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
                stop_reason="no_action_worth_it",
                action_trace=list(state.action_trace),
                best_rejected_action=best_rejected_action,
                ambiguity_summary=fragility.value_of_refresh,
            )

        best_rejected_action = feasible_actions[1].as_dict() if len(feasible_actions) > 1 else top_action.as_dict()
        trace_entry = {
            "iteration": state.iteration_index,
            "feasible_actions": [action.as_dict() for action in actions],
            "chosen_action": top_action.as_dict(),
            "certificate_value": current_certificate,
            "remaining_search_budget": state.remaining_search_budget,
            "remaining_evidence_budget": state.remaining_evidence_budget,
        }
        state = replace(
            state,
            action_trace=[*state.action_trace, trace_entry],
        )
        state = _apply_action(state, top_action, hooks=hooks)
        if top_action.kind == "stop":
            break

        if hooks is None:
            continue

    current_certificate = _as_float(state.certificate.get(state.winner_id, certificate_value))
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
        stop_reason="certified" if current_certificate >= cfg.certificate_threshold else "budget_exhausted",
        action_trace=list(state.action_trace),
        best_rejected_action=best_rejected_action,
        ambiguity_summary=fragility.value_of_refresh,
    )
