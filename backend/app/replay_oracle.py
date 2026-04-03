from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return float(parsed)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class ActionValueEstimate:
    action_id: str
    action_kind: str
    action_target: str
    action_reason: str = ""
    cost_search: int = 0
    cost_evidence: int = 0
    predicted_delta_certificate: float = 0.0
    predicted_delta_margin: float = 0.0
    predicted_delta_frontier: float = 0.0
    weighted_certificate_value: float = 0.0
    weighted_margin_value: float = 0.0
    weighted_frontier_value: float = 0.0
    total_predicted_value: float = 0.0
    total_cost: float = 0.0
    base_q_score: float = 0.0
    ranked_q_score: float = 0.0
    lambda_certificate: float = 0.0
    lambda_margin: float = 0.0
    lambda_frontier: float = 0.0
    epsilon: float = 1e-9
    score_terms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionValueRealization:
    realized_certificate_before: float | None = None
    realized_certificate_after: float | None = None
    realized_certificate_delta: float | None = None
    realized_margin_before: float | None = None
    realized_margin_after: float | None = None
    realized_margin_delta: float | None = None
    realized_frontier_gain: float | None = None
    realized_selected_route_changed: bool | None = None
    realized_selected_score_delta: float | None = None
    realized_runner_up_gap_before: float | None = None
    realized_runner_up_gap_after: float | None = None
    realized_runner_up_gap_delta: float | None = None
    realized_evidence_uncertainty_before: float | None = None
    realized_evidence_uncertainty_after: float | None = None
    realized_evidence_uncertainty_delta: float | None = None
    realized_productive: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionReplayRecord:
    estimate: ActionValueEstimate
    realization: ActionValueRealization | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)
    replay_metadata: dict[str, Any] = field(default_factory=dict)
    oracle_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate.as_dict(),
            "realization": self.realization.as_dict() if self.realization is not None else None,
            "trace_metadata": dict(self.trace_metadata),
            "replay_metadata": dict(self.replay_metadata),
            "oracle_metadata": dict(self.oracle_metadata),
        }


def build_action_value_estimate(
    *,
    action_id: str,
    action_kind: str,
    action_target: str,
    action_reason: str = "",
    cost_search: int = 0,
    cost_evidence: int = 0,
    predicted_delta_certificate: float = 0.0,
    predicted_delta_margin: float = 0.0,
    predicted_delta_frontier: float = 0.0,
    lambda_certificate: float = 0.0,
    lambda_margin: float = 0.0,
    lambda_frontier: float = 0.0,
    epsilon: float = 1e-9,
    ranked_q_score: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ActionValueEstimate:
    certificate_delta = _as_float(predicted_delta_certificate)
    margin_delta = _as_float(predicted_delta_margin)
    frontier_delta = _as_float(predicted_delta_frontier)
    search_cost = max(0, int(cost_search))
    evidence_cost = max(0, int(cost_evidence))
    total_cost = float(search_cost + evidence_cost)
    weighted_certificate = _as_float(lambda_certificate) * certificate_delta
    weighted_margin = _as_float(lambda_margin) * margin_delta
    weighted_frontier = _as_float(lambda_frontier) * frontier_delta
    total_predicted_value = weighted_certificate + weighted_margin + weighted_frontier
    safe_epsilon = max(1e-12, _as_float(epsilon) or 1e-9)
    base_q_score = total_predicted_value / (total_cost + safe_epsilon)
    resolved_ranked_q_score = base_q_score if ranked_q_score is None else _as_float(ranked_q_score)
    return ActionValueEstimate(
        action_id=str(action_id),
        action_kind=str(action_kind),
        action_target=str(action_target),
        action_reason=str(action_reason or ""),
        cost_search=search_cost,
        cost_evidence=evidence_cost,
        predicted_delta_certificate=certificate_delta,
        predicted_delta_margin=margin_delta,
        predicted_delta_frontier=frontier_delta,
        weighted_certificate_value=weighted_certificate,
        weighted_margin_value=weighted_margin,
        weighted_frontier_value=weighted_frontier,
        total_predicted_value=total_predicted_value,
        total_cost=total_cost,
        base_q_score=base_q_score,
        ranked_q_score=resolved_ranked_q_score,
        lambda_certificate=_as_float(lambda_certificate),
        lambda_margin=_as_float(lambda_margin),
        lambda_frontier=_as_float(lambda_frontier),
        epsilon=safe_epsilon,
        score_terms={
            "certificate_delta": certificate_delta,
            "margin_delta": margin_delta,
            "frontier_delta": frontier_delta,
            "certificate_value": weighted_certificate,
            "margin_value": weighted_margin,
            "frontier_value": weighted_frontier,
        },
        metadata=_mapping(metadata),
    )


def realization_from_trace_entry(
    trace_entry: Mapping[str, Any] | None,
) -> ActionValueRealization | None:
    if not isinstance(trace_entry, Mapping):
        return None
    raw_realization = trace_entry.get("realization")
    payload = raw_realization if isinstance(raw_realization, Mapping) else trace_entry
    numeric_fields = (
        "realized_certificate_before",
        "realized_certificate_after",
        "realized_certificate_delta",
        "realized_margin_before",
        "realized_margin_after",
        "realized_margin_delta",
        "realized_frontier_gain",
        "realized_selected_score_delta",
        "realized_runner_up_gap_before",
        "realized_runner_up_gap_after",
        "realized_runner_up_gap_delta",
        "realized_evidence_uncertainty_before",
        "realized_evidence_uncertainty_after",
        "realized_evidence_uncertainty_delta",
    )
    bool_fields = (
        "realized_selected_route_changed",
        "realized_productive",
    )
    if not any(field in payload for field in (*numeric_fields, *bool_fields)):
        return None
    return ActionValueRealization(
        realized_certificate_before=_optional_float(payload.get("realized_certificate_before")),
        realized_certificate_after=_optional_float(payload.get("realized_certificate_after")),
        realized_certificate_delta=_optional_float(payload.get("realized_certificate_delta")),
        realized_margin_before=_optional_float(payload.get("realized_margin_before")),
        realized_margin_after=_optional_float(payload.get("realized_margin_after")),
        realized_margin_delta=_optional_float(payload.get("realized_margin_delta")),
        realized_frontier_gain=_optional_float(payload.get("realized_frontier_gain")),
        realized_selected_route_changed=_optional_bool(payload.get("realized_selected_route_changed")),
        realized_selected_score_delta=_optional_float(payload.get("realized_selected_score_delta")),
        realized_runner_up_gap_before=_optional_float(payload.get("realized_runner_up_gap_before")),
        realized_runner_up_gap_after=_optional_float(payload.get("realized_runner_up_gap_after")),
        realized_runner_up_gap_delta=_optional_float(payload.get("realized_runner_up_gap_delta")),
        realized_evidence_uncertainty_before=_optional_float(payload.get("realized_evidence_uncertainty_before")),
        realized_evidence_uncertainty_after=_optional_float(payload.get("realized_evidence_uncertainty_after")),
        realized_evidence_uncertainty_delta=_optional_float(payload.get("realized_evidence_uncertainty_delta")),
        realized_productive=_optional_bool(payload.get("realized_productive")),
        metadata=_mapping(payload.get("metadata")),
    )


def build_action_replay_record(
    estimate: ActionValueEstimate,
    *,
    trace_entry: Mapping[str, Any] | None = None,
    trace_metadata: Mapping[str, Any] | None = None,
    replay_metadata: Mapping[str, Any] | None = None,
    oracle_metadata: Mapping[str, Any] | None = None,
    realization: ActionValueRealization | None = None,
) -> ActionReplayRecord:
    resolved_realization = realization or realization_from_trace_entry(trace_entry)
    trace_payload = _mapping(trace_entry.get("trace_metadata")) if isinstance(trace_entry, Mapping) else {}
    trace_payload.update(_mapping(trace_metadata))
    replay_payload = _mapping(trace_entry.get("replay_metadata")) if isinstance(trace_entry, Mapping) else {}
    replay_payload.update(_mapping(replay_metadata))
    oracle_payload = _mapping(trace_entry.get("oracle_metadata")) if isinstance(trace_entry, Mapping) else {}
    oracle_payload.update(_mapping(oracle_metadata))
    return ActionReplayRecord(
        estimate=estimate,
        realization=resolved_realization,
        trace_metadata=trace_payload,
        replay_metadata=replay_payload,
        oracle_metadata=oracle_payload,
    )
