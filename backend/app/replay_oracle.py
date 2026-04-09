"""Evaluator-only replay oracle utilities for VOI traces.

This module stays out of the live runtime path. It consumes VOI action traces
and stop-certificate payloads to produce replay/regret summaries for evaluator
checks, artifact validation, and offline debugging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return float(default)
    return float(parsed)


def _action_family(kind: str) -> str:
    if kind in {"refine_top1_dccs", "refine_topk_dccs"}:
        return "search"
    if kind in {"refresh_top1_vor", "increase_stochastic_samples"}:
        return "evidence"
    if kind == "stop":
        return "terminal"
    return "unknown"


def _action_modality(kind: str) -> str:
    if kind == "refine_top1_dccs":
        return "refine_top1"
    if kind == "refine_topk_dccs":
        return "refine_topk"
    if kind == "refresh_top1_vor":
        return "refresh"
    if kind == "increase_stochastic_samples":
        return "resample"
    if kind == "stop":
        return "stop"
    return kind or "unknown"


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


@dataclass(frozen=True)
class ReplayOracleActionSummary:
    iteration: int
    action_id: str
    kind: str
    action_family: str
    action_modality: str
    cost_search: int
    cost_evidence: int
    q_score: float
    predicted_delta_certificate: float
    realized_delta_certificate: float
    predicted_delta_margin: float
    realized_delta_runner_up_gap: float
    predicted_delta_frontier: float
    realized_delta_frontier: float
    predicted_delta_search_completeness: float
    realized_delta_search_completeness: float
    feasible_action_count: int
    productive: bool
    selected: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayOracleSummary:
    iteration_count: int
    action_count: int
    stop_reason: str
    initial_certificate: float
    final_certificate: float
    cumulative_chosen_q: float
    cumulative_best_q: float
    replay_regret: float
    mean_predicted_certificate_delta: float
    mean_realized_certificate_delta: float
    mean_abs_certificate_delta_error: float
    mean_abs_runner_up_gap_error: float
    mean_abs_frontier_delta_error: float
    mean_abs_search_completeness_error: float
    action_family_sequence: list[str] = field(default_factory=list)
    action_modality_sequence: list[str] = field(default_factory=list)
    family_switch_count: int = 0
    modality_switch_count: int = 0
    productive_action_count: int = 0
    productive_action_rate: float = 0.0
    stop_action_seen: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayOracleEvaluation:
    summary: ReplayOracleSummary
    actions: list[ReplayOracleActionSummary] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.as_dict(),
            "actions": [action.as_dict() for action in self.actions],
        }


def summarize_replay_oracle_trace(
    action_trace: Sequence[Mapping[str, Any]],
    *,
    initial_certificate: float | None = None,
    final_certificate: float | None = None,
    stop_reason: str = "",
) -> ReplayOracleEvaluation:
    actions: list[ReplayOracleActionSummary] = []
    cumulative_best_q = 0.0
    cumulative_chosen_q = 0.0
    cumulative_predicted_certificate = 0.0
    cumulative_realized_certificate = 0.0
    abs_certificate_error = 0.0
    abs_runner_up_gap_error = 0.0
    abs_frontier_error = 0.0
    abs_search_completeness_error = 0.0
    productive_action_count = 0
    replay_regret = 0.0
    family_sequence: list[str] = []
    modality_sequence: list[str] = []
    previous_family = ""
    previous_modality = ""
    stop_action_seen = False

    for iteration, entry in enumerate(action_trace):
        if not isinstance(entry, Mapping):
            continue
        chosen = _mapping(entry.get("chosen_action"))
        if chosen is None:
            continue
        feasible_actions = [
            candidate
            for candidate in (_mapping(item) for item in _sequence(entry.get("feasible_actions")))
            if candidate is not None
        ]
        chosen_q = _as_float(chosen.get("q_score"))
        best_q = max((_as_float(candidate.get("q_score")) for candidate in feasible_actions), default=chosen_q)
        cumulative_best_q += best_q
        cumulative_chosen_q += chosen_q
        replay_regret += max(0.0, best_q - chosen_q)

        kind = str(chosen.get("kind", "")).strip()
        action_family = str(chosen.get("action_family") or _action_family(kind))
        action_modality = str(chosen.get("action_modality") or _action_modality(kind))
        family_sequence.append(action_family)
        modality_sequence.append(action_modality)
        if previous_family and previous_family != action_family:
            pass
        if previous_modality and previous_modality != action_modality:
            pass
        previous_family = action_family
        previous_modality = action_modality
        stop_action_seen = stop_action_seen or kind == "stop"

        predicted_delta_certificate = _as_float(chosen.get("predicted_delta_certificate"))
        predicted_delta_margin = _as_float(chosen.get("predicted_delta_margin"))
        predicted_delta_frontier = _as_float(chosen.get("predicted_delta_frontier"))
        predicted_delta_search_completeness = _as_float(chosen.get("predicted_delta_search_completeness"))

        realized_delta_certificate = _as_float(entry.get("realized_certificate_delta"))
        realized_delta_runner_up_gap = _as_float(entry.get("realized_runner_up_gap_delta"))
        realized_delta_frontier = _as_float(entry.get("realized_frontier_gain"))
        realized_delta_search_completeness = _as_float(entry.get("realized_search_completeness_delta"))

        action = ReplayOracleActionSummary(
            iteration=int(_as_float(entry.get("iteration"), iteration)),
            action_id=str(chosen.get("action_id", "")),
            kind=kind,
            action_family=action_family,
            action_modality=action_modality,
            cost_search=int(_as_float(chosen.get("cost_search"))),
            cost_evidence=int(_as_float(chosen.get("cost_evidence"))),
            q_score=chosen_q,
            predicted_delta_certificate=predicted_delta_certificate,
            realized_delta_certificate=realized_delta_certificate,
            predicted_delta_margin=predicted_delta_margin,
            realized_delta_runner_up_gap=realized_delta_runner_up_gap,
            predicted_delta_frontier=predicted_delta_frontier,
            realized_delta_frontier=realized_delta_frontier,
            predicted_delta_search_completeness=predicted_delta_search_completeness,
            realized_delta_search_completeness=realized_delta_search_completeness,
            feasible_action_count=len(feasible_actions),
            productive=bool(entry.get("realized_productive"))
            or realized_delta_certificate > 1e-9
            or realized_delta_frontier > 0.0
            or realized_delta_search_completeness > 1e-9,
            selected=bool(chosen.get("kind") == "stop" or chosen.get("selected", False)),
        )
        actions.append(action)
        cumulative_predicted_certificate += predicted_delta_certificate
        cumulative_realized_certificate += realized_delta_certificate
        abs_certificate_error += abs(realized_delta_certificate - predicted_delta_certificate)
        abs_runner_up_gap_error += abs(realized_delta_runner_up_gap - predicted_delta_margin)
        abs_frontier_error += abs(realized_delta_frontier - predicted_delta_frontier)
        abs_search_completeness_error += abs(realized_delta_search_completeness - predicted_delta_search_completeness)
        if action.productive:
            productive_action_count += 1

    action_count = len(actions)
    if initial_certificate is None:
        initial_certificate = _as_float(action_trace[0].get("certificate_value"), 0.0) if action_trace else 0.0
    if final_certificate is None:
        final_certificate = _as_float(action_trace[-1].get("certificate_value"), initial_certificate) if action_trace else initial_certificate

    summary = ReplayOracleSummary(
        iteration_count=action_count,
        action_count=action_count,
        stop_reason=str(stop_reason),
        initial_certificate=float(initial_certificate),
        final_certificate=float(final_certificate),
        cumulative_chosen_q=cumulative_chosen_q,
        cumulative_best_q=cumulative_best_q,
        replay_regret=replay_regret,
        mean_predicted_certificate_delta=(cumulative_predicted_certificate / action_count) if action_count else 0.0,
        mean_realized_certificate_delta=(cumulative_realized_certificate / action_count) if action_count else 0.0,
        mean_abs_certificate_delta_error=(abs_certificate_error / action_count) if action_count else 0.0,
        mean_abs_runner_up_gap_error=(abs_runner_up_gap_error / action_count) if action_count else 0.0,
        mean_abs_frontier_delta_error=(abs_frontier_error / action_count) if action_count else 0.0,
        mean_abs_search_completeness_error=(abs_search_completeness_error / action_count) if action_count else 0.0,
        action_family_sequence=family_sequence,
        action_modality_sequence=modality_sequence,
        family_switch_count=sum(
            1
            for prev, current in zip(family_sequence, family_sequence[1:], strict=False)
            if prev != current
        ),
        modality_switch_count=sum(
            1
            for prev, current in zip(modality_sequence, modality_sequence[1:], strict=False)
            if prev != current
        ),
        productive_action_count=productive_action_count,
        productive_action_rate=(productive_action_count / action_count) if action_count else 0.0,
        stop_action_seen=stop_action_seen,
    )
    return ReplayOracleEvaluation(summary=summary, actions=actions)


def summarize_replay_oracle_stop_certificate(stop_certificate: Mapping[str, Any]) -> ReplayOracleEvaluation:
    action_trace = _sequence(stop_certificate.get("action_trace"))
    initial_certificate = _as_float(stop_certificate.get("initial_certificate"), 0.0)
    final_certificate = _as_float(
        stop_certificate.get("certificate_value", stop_certificate.get("final_certificate")),
        initial_certificate,
    )
    stop_reason = str(stop_certificate.get("stop_reason", ""))
    return summarize_replay_oracle_trace(
        action_trace,  # type: ignore[arg-type]
        initial_certificate=initial_certificate,
        final_certificate=final_certificate,
        stop_reason=stop_reason,
    )
