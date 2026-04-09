from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence


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


@dataclass(frozen=True)
class ReplayOracleSummary:
    trace_source: str
    record_count: int
    predicted_total_value: float
    predicted_total_certificate_value: float
    predicted_total_margin_value: float
    predicted_total_frontier_value: float
    realized_total_certificate_delta: float
    realized_total_margin_delta: float
    realized_total_frontier_gain: float
    realized_total_evidence_uncertainty_delta: float
    realized_productive_count: int
    realized_predicted_alignment_rate: float
    low_ambiguity_fast_path: bool = False
    best_action_id: str | None = None
    best_action_kind: str | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)
    replay_metadata: dict[str, Any] = field(default_factory=dict)
    oracle_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "0.1.0"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def action_replay_record_from_payload(payload: Mapping[str, Any] | None) -> ActionReplayRecord | None:
    if not isinstance(payload, Mapping):
        return None
    estimate_payload = payload.get("estimate")
    if not isinstance(estimate_payload, Mapping):
        return None
    estimate = ActionValueEstimate(
        action_id=str(estimate_payload.get("action_id", "")),
        action_kind=str(estimate_payload.get("action_kind", "")),
        action_target=str(estimate_payload.get("action_target", "")),
        action_reason=str(estimate_payload.get("action_reason", "")),
        cost_search=int(_as_float(estimate_payload.get("cost_search"))),
        cost_evidence=int(_as_float(estimate_payload.get("cost_evidence"))),
        predicted_delta_certificate=_as_float(estimate_payload.get("predicted_delta_certificate")),
        predicted_delta_margin=_as_float(estimate_payload.get("predicted_delta_margin")),
        predicted_delta_frontier=_as_float(estimate_payload.get("predicted_delta_frontier")),
        weighted_certificate_value=_as_float(estimate_payload.get("weighted_certificate_value")),
        weighted_margin_value=_as_float(estimate_payload.get("weighted_margin_value")),
        weighted_frontier_value=_as_float(estimate_payload.get("weighted_frontier_value")),
        total_predicted_value=_as_float(estimate_payload.get("total_predicted_value")),
        total_cost=_as_float(estimate_payload.get("total_cost")),
        base_q_score=_as_float(estimate_payload.get("base_q_score")),
        ranked_q_score=_as_float(estimate_payload.get("ranked_q_score")),
        lambda_certificate=_as_float(estimate_payload.get("lambda_certificate")),
        lambda_margin=_as_float(estimate_payload.get("lambda_margin")),
        lambda_frontier=_as_float(estimate_payload.get("lambda_frontier")),
        epsilon=max(1e-12, _optional_float(estimate_payload.get("epsilon")) or 1e-9),
        score_terms=dict(estimate_payload.get("score_terms", {}))
        if isinstance(estimate_payload.get("score_terms"), Mapping)
        else {},
        metadata=_mapping(estimate_payload.get("metadata")),
    )
    realization_payload = payload.get("realization")
    realization = None
    if isinstance(realization_payload, Mapping):
        realization = ActionValueRealization(
            realized_certificate_before=_optional_float(realization_payload.get("realized_certificate_before")),
            realized_certificate_after=_optional_float(realization_payload.get("realized_certificate_after")),
            realized_certificate_delta=_optional_float(realization_payload.get("realized_certificate_delta")),
            realized_margin_before=_optional_float(realization_payload.get("realized_margin_before")),
            realized_margin_after=_optional_float(realization_payload.get("realized_margin_after")),
            realized_margin_delta=_optional_float(realization_payload.get("realized_margin_delta")),
            realized_frontier_gain=_optional_float(realization_payload.get("realized_frontier_gain")),
            realized_selected_route_changed=_optional_bool(
                realization_payload.get("realized_selected_route_changed")
            ),
            realized_selected_score_delta=_optional_float(
                realization_payload.get("realized_selected_score_delta")
            ),
            realized_runner_up_gap_before=_optional_float(
                realization_payload.get("realized_runner_up_gap_before")
            ),
            realized_runner_up_gap_after=_optional_float(
                realization_payload.get("realized_runner_up_gap_after")
            ),
            realized_runner_up_gap_delta=_optional_float(
                realization_payload.get("realized_runner_up_gap_delta")
            ),
            realized_evidence_uncertainty_before=_optional_float(
                realization_payload.get("realized_evidence_uncertainty_before")
            ),
            realized_evidence_uncertainty_after=_optional_float(
                realization_payload.get("realized_evidence_uncertainty_after")
            ),
            realized_evidence_uncertainty_delta=_optional_float(
                realization_payload.get("realized_evidence_uncertainty_delta")
            ),
            realized_productive=_optional_bool(realization_payload.get("realized_productive")),
            metadata=_mapping(realization_payload.get("metadata")),
        )
    return ActionReplayRecord(
        estimate=estimate,
        realization=realization,
        trace_metadata=_mapping(payload.get("trace_metadata")),
        replay_metadata=_mapping(payload.get("replay_metadata")),
        oracle_metadata=_mapping(payload.get("oracle_metadata")),
    )


def _sum(values: Iterable[float]) -> float:
    return float(sum(_as_float(value) for value in values))


def build_replay_oracle_summary(
    records: Sequence[ActionReplayRecord | Mapping[str, Any]],
    *,
    trace_source: str = "",
    low_ambiguity_fast_path: bool = False,
    trace_metadata: Mapping[str, Any] | None = None,
    replay_metadata: Mapping[str, Any] | None = None,
    oracle_metadata: Mapping[str, Any] | None = None,
) -> ReplayOracleSummary:
    resolved_records: list[ActionReplayRecord] = []
    for record in records:
        if isinstance(record, ActionReplayRecord):
            resolved_records.append(record)
            continue
        parsed = action_replay_record_from_payload(record)
        if parsed is not None:
            resolved_records.append(parsed)

    predicted_total_value = _sum(record.estimate.total_predicted_value for record in resolved_records)
    predicted_total_certificate_value = _sum(
        record.estimate.weighted_certificate_value for record in resolved_records
    )
    predicted_total_margin_value = _sum(record.estimate.weighted_margin_value for record in resolved_records)
    predicted_total_frontier_value = _sum(record.estimate.weighted_frontier_value for record in resolved_records)
    realized_total_certificate_delta = _sum(
        record.realization.realized_certificate_delta
        for record in resolved_records
        if record.realization is not None and record.realization.realized_certificate_delta is not None
    )
    realized_total_margin_delta = _sum(
        record.realization.realized_margin_delta
        for record in resolved_records
        if record.realization is not None and record.realization.realized_margin_delta is not None
    )
    realized_total_frontier_gain = _sum(
        record.realization.realized_frontier_gain
        for record in resolved_records
        if record.realization is not None and record.realization.realized_frontier_gain is not None
    )
    realized_total_evidence_uncertainty_delta = _sum(
        record.realization.realized_evidence_uncertainty_delta
        for record in resolved_records
        if record.realization is not None
        and record.realization.realized_evidence_uncertainty_delta is not None
    )
    realized_productive_count = sum(
        1
        for record in resolved_records
        if record.realization is not None and bool(record.realization.realized_productive)
    )
    realized_alignment_count = sum(
        1
        for record in resolved_records
        if record.realization is not None
        and (
            bool(record.realization.realized_productive)
            or (
                (record.realization.realized_certificate_delta is not None)
                and record.realization.realized_certificate_delta > 1e-9
            )
            or (
                (record.realization.realized_frontier_gain is not None)
                and record.realization.realized_frontier_gain > 0.0
            )
            or (
                (record.realization.realized_evidence_uncertainty_delta is not None)
                and record.realization.realized_evidence_uncertainty_delta < -1e-9
            )
        )
    )
    record_count = len(resolved_records)
    best_record = max(
        resolved_records,
        key=lambda record: (
            record.estimate.total_predicted_value,
            record.estimate.ranked_q_score,
            record.estimate.base_q_score,
            record.estimate.action_id,
        ),
        default=None,
    )
    return ReplayOracleSummary(
        trace_source=str(trace_source or ""),
        record_count=record_count,
        predicted_total_value=predicted_total_value,
        predicted_total_certificate_value=predicted_total_certificate_value,
        predicted_total_margin_value=predicted_total_margin_value,
        predicted_total_frontier_value=predicted_total_frontier_value,
        realized_total_certificate_delta=realized_total_certificate_delta,
        realized_total_margin_delta=realized_total_margin_delta,
        realized_total_frontier_gain=realized_total_frontier_gain,
        realized_total_evidence_uncertainty_delta=realized_total_evidence_uncertainty_delta,
        realized_productive_count=realized_productive_count,
        realized_predicted_alignment_rate=(
            float(realized_alignment_count) / float(record_count) if record_count > 0 else 0.0
        ),
        low_ambiguity_fast_path=bool(low_ambiguity_fast_path),
        best_action_id=(best_record.estimate.action_id if best_record is not None else None),
        best_action_kind=(best_record.estimate.action_kind if best_record is not None else None),
        trace_metadata=dict(trace_metadata or {}),
        replay_metadata=dict(replay_metadata or {}),
        oracle_metadata=dict(oracle_metadata or {}),
    )


def replay_oracle_summary_from_trace_rows(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    trace_source: str = "",
    low_ambiguity_fast_path: bool = False,
    trace_metadata: Mapping[str, Any] | None = None,
    replay_metadata: Mapping[str, Any] | None = None,
    oracle_metadata: Mapping[str, Any] | None = None,
) -> ReplayOracleSummary:
    records = []
    for row in trace_rows:
        if not isinstance(row, Mapping):
            continue
        chosen = row.get("chosen_action_value_record")
        if isinstance(chosen, Mapping):
            record = action_replay_record_from_payload(chosen)
            if record is not None:
                records.append(record)
    return build_replay_oracle_summary(
        records,
        trace_source=trace_source,
        low_ambiguity_fast_path=low_ambiguity_fast_path,
        trace_metadata=trace_metadata,
        replay_metadata=replay_metadata,
        oracle_metadata=oracle_metadata,
    )
