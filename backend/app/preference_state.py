"""Pipeline stage: carry preference-certification state, compatible sets, and shrinkage history."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .preference_queries import (
    PairwisePreferenceQuery,
    PreferenceQuery,
    RatioPreferenceQuery,
    ThresholdPreferenceQuery,
    TimeGuardPreferenceQuery,
    VetoPreferenceQuery,
)


class CompatibleSetSummary(BaseModel):
    route_ids: list[str] = Field(default_factory=list)
    compatible_set_size: int = Field(default=0, ge=0)
    compatible_set_volume_proxy: float = Field(default=1.0, ge=0.0)
    necessary_best_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    possible_best_prob: float = Field(default=1.0, ge=0.0, le=1.0)
    necessary_best_route_ids: list[str] = Field(default_factory=list)
    possible_best_route_ids: list[str] = Field(default_factory=list)
    support_flag: bool = True
    support_reason: str | None = None

    @field_validator("route_ids", "necessary_best_route_ids", "possible_best_route_ids")
    @classmethod
    def _dedupe_route_ids(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for route_id in value:
            cleaned = str(route_id).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped

    @model_validator(mode="after")
    def _sync_size(self) -> "CompatibleSetSummary":
        if self.compatible_set_size != len(self.route_ids):
            self.compatible_set_size = len(self.route_ids)
        if self.necessary_best_prob > self.possible_best_prob:
            self.necessary_best_prob = self.possible_best_prob
        route_id_set = set(self.route_ids)
        possible_route_ids = [route_id for route_id in self.possible_best_route_ids if route_id in route_id_set]
        necessary_route_ids = [route_id for route_id in self.necessary_best_route_ids if route_id in route_id_set]
        if not possible_route_ids and self.route_ids:
            possible_route_ids = list(self.route_ids)
        for route_id in necessary_route_ids:
            if route_id not in possible_route_ids:
                possible_route_ids.append(route_id)
        self.necessary_best_route_ids = necessary_route_ids
        self.possible_best_route_ids = possible_route_ids
        return self


class PreferenceShrinkageTrace(BaseModel):
    query_index: int = Field(ge=0)
    query_type: Literal["pairwise", "threshold", "ratio", "veto", "time_guard"]
    before_size: int = Field(ge=0)
    after_size: int = Field(ge=0)
    before_volume_proxy: float = Field(ge=0.0)
    after_volume_proxy: float = Field(ge=0.0)
    predicted_shrinkage: float = Field(default=0.0, ge=0.0)
    realized_shrinkage: float = Field(default=0.0, ge=0.0)
    target_route_id: str | None = None
    query_reason: str | None = None
    preference_irrelevance: bool = False


class PreferenceContradictionRecord(BaseModel):
    contradiction_detected: bool = False
    contradiction_reasons: list[str] = Field(default_factory=list)


class PreferenceState(BaseModel):
    compatible_set_summary: CompatibleSetSummary = Field(default_factory=CompatibleSetSummary)
    compatible_weights: list[dict[str, float]] = Field(default_factory=list)
    pairwise_constraints: list[PairwisePreferenceQuery] = Field(default_factory=list)
    threshold_constraints: list[ThresholdPreferenceQuery] = Field(default_factory=list)
    ratio_constraints: list[RatioPreferenceQuery] = Field(default_factory=list)
    veto_rules: list[VetoPreferenceQuery] = Field(default_factory=list)
    time_preserving_guard_rules: list[TimeGuardPreferenceQuery] = Field(default_factory=list)
    query_history: list[PreferenceQuery] = Field(default_factory=list)
    shrinkage_trace: list[PreferenceShrinkageTrace] = Field(default_factory=list)
    contradiction_record: PreferenceContradictionRecord = Field(default_factory=PreferenceContradictionRecord)
    derived_invariants: dict[str, bool] = Field(default_factory=dict)
    terminal_type: Literal["open", "certified", "abstained"] = "open"
    preference_irrelevance_proven: bool = False
    no_query_reason: str | None = None
    query_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _sync_state(self) -> "PreferenceState":
        self.query_count = len(self.query_history)
        self.compatible_set_summary.compatible_set_size = len(self.compatible_set_summary.route_ids)
        self.preference_irrelevance_proven = bool(
            self.preference_irrelevance_proven or self.compatible_set_summary.compatible_set_size <= 1
        )
        self.contradiction_record = detect_preference_contradictions(self)
        sync_route_level_best_markings(self)
        self.no_query_reason = derive_no_query_reason(self)
        self.derived_invariants = {
            "necessary_best_prob_le_possible_best_prob": (
                self.compatible_set_summary.necessary_best_prob
                <= self.compatible_set_summary.possible_best_prob
            ),
            "no_necessary_best_without_possible_best": set(
                self.compatible_set_summary.necessary_best_route_ids
            ).issubset(set(self.compatible_set_summary.possible_best_route_ids)),
            "compatible_set_nonnegative": self.compatible_set_summary.compatible_set_size >= 0,
            "compatible_volume_nonnegative": self.compatible_set_summary.compatible_set_volume_proxy >= 0.0,
            "compatible_set_volume_nonincreasing_after_query": volume_trace_nonincreasing(self),
            "preference_contradiction_free": not self.contradiction_record.contradiction_detected,
            "query_history_matches_trace_or_zero": (
                len(self.shrinkage_trace) == 0 or len(self.shrinkage_trace) <= self.query_count
            ),
        }
        return self


def empty_preference_state(*, route_ids: list[str] | None = None) -> PreferenceState:
    return PreferenceState(
        compatible_set_summary=CompatibleSetSummary(route_ids=list(route_ids or [])),
    )


def volume_trace_nonincreasing(state: PreferenceState) -> bool:
    trace = state.shrinkage_trace
    if len(trace) < 2:
        return True
    return all(
        earlier.after_volume_proxy >= later.after_volume_proxy
        for earlier, later in zip(trace, trace[1:])
    )


def sync_route_level_best_markings(state: PreferenceState) -> None:
    summary = state.compatible_set_summary
    route_ids = list(summary.route_ids)
    route_id_set = set(route_ids)
    possible_route_ids = [route_id for route_id in summary.possible_best_route_ids if route_id in route_id_set]
    if not possible_route_ids and route_ids:
        possible_route_ids = list(route_ids)
    necessary_route_ids = [route_id for route_id in summary.necessary_best_route_ids if route_id in route_id_set]

    if state.contradiction_record.contradiction_detected or not summary.support_flag:
        necessary_route_ids = []

    singleton_route_id = _resolve_singleton_preference_route_id(state, possible_route_ids)
    if singleton_route_id is not None:
        possible_route_ids = [singleton_route_id]
        necessary_route_ids = [singleton_route_id] if summary.support_flag else []
        summary.possible_best_prob = 1.0
        summary.necessary_best_prob = 1.0 if necessary_route_ids else 0.0

    for route_id in necessary_route_ids:
        if route_id not in possible_route_ids:
            possible_route_ids.append(route_id)

    summary.necessary_best_route_ids = necessary_route_ids
    summary.possible_best_route_ids = possible_route_ids


def _resolve_singleton_preference_route_id(
    state: PreferenceState,
    possible_route_ids: list[str],
) -> str | None:
    summary = state.compatible_set_summary
    if summary.compatible_set_size > 1 or not state.preference_irrelevance_proven:
        return None
    if len(summary.route_ids) == 1:
        return summary.route_ids[0]
    if len(possible_route_ids) == 1:
        return possible_route_ids[0]
    if not state.query_history:
        return None
    last_query = state.query_history[-1]
    if isinstance(last_query, PairwisePreferenceQuery):
        preferred_route_id = str(last_query.preferred_route_id).strip()
        if preferred_route_id in set(summary.route_ids):
            return preferred_route_id
    return None


def detect_preference_contradictions(state: PreferenceState) -> PreferenceContradictionRecord:
    reasons: list[str] = []
    pairwise_reason = _pairwise_contradiction_reason(state.pairwise_constraints)
    if pairwise_reason is not None:
        reasons.append(pairwise_reason)
    reasons.extend(_threshold_contradiction_reasons(state.threshold_constraints))
    reasons.extend(_veto_contradiction_reasons(state.veto_rules))
    return PreferenceContradictionRecord(
        contradiction_detected=bool(reasons),
        contradiction_reasons=reasons,
    )


def derive_no_query_reason(state: PreferenceState) -> str | None:
    if state.query_count > 0:
        return None
    if state.contradiction_record.contradiction_detected:
        return "preference_contradiction_detected"
    if not state.compatible_set_summary.support_flag:
        return "preference_support_insufficient"
    if state.preference_irrelevance_proven:
        return "preference_irrelevance_proven"
    if state.compatible_set_summary.compatible_set_size <= 1:
        return "singleton_frontier"
    return "no_preference_query_issued"


def _pairwise_contradiction_reason(
    constraints: list[PairwisePreferenceQuery],
) -> str | None:
    graph: dict[str, set[str]] = {}
    for constraint in constraints:
        graph.setdefault(constraint.preferred_route_id, set()).add(constraint.challenger_route_id)
    seen: set[str] = set()
    stack: list[str] = []
    active: set[str] = set()

    def visit(node: str) -> list[str] | None:
        seen.add(node)
        active.add(node)
        stack.append(node)
        for neighbor in sorted(graph.get(node, ())):
            if neighbor in active:
                idx = stack.index(neighbor)
                return stack[idx:] + [neighbor]
            if neighbor not in seen:
                cycle = visit(neighbor)
                if cycle is not None:
                    return cycle
        stack.pop()
        active.remove(node)
        return None

    for route_id in sorted(graph):
        if route_id in seen:
            continue
        cycle = visit(route_id)
        if cycle is not None:
            path = " > ".join(cycle)
            return f"pairwise_cycle:{path}"
    return None


def _threshold_contradiction_reasons(
    constraints: list[ThresholdPreferenceQuery],
) -> list[str]:
    bounds: dict[tuple[str, str], dict[str, float]] = {}
    reasons: list[str] = []
    for constraint in constraints:
        key = (constraint.route_id, constraint.metric_name)
        row = bounds.setdefault(key, {})
        if constraint.direction == "gte":
            row["lower"] = max(float(constraint.threshold_value), float(row.get("lower", constraint.threshold_value)))
        else:
            row["upper"] = min(float(constraint.threshold_value), float(row.get("upper", constraint.threshold_value)))
    for (route_id, metric_name), row in sorted(bounds.items()):
        lower = row.get("lower")
        upper = row.get("upper")
        if lower is not None and upper is not None and lower > upper:
            reasons.append(f"threshold_conflict:{route_id}:{metric_name}:{lower}>{upper}")
    return reasons


def _veto_contradiction_reasons(
    constraints: list[VetoPreferenceQuery],
) -> list[str]:
    states: dict[tuple[str, str], set[bool]] = {}
    reasons: list[str] = []
    for constraint in constraints:
        key = (constraint.route_id, constraint.veto_name)
        states.setdefault(key, set()).add(bool(constraint.active))
    for (route_id, veto_name), active_states in sorted(states.items()):
        if len(active_states) > 1:
            reasons.append(f"veto_conflict:{route_id}:{veto_name}")
    return reasons
