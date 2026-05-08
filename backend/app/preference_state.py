"""Pipeline stage: carry preference-certification state, compatible sets, and shrinkage history."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

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


class PreferenceWeights(BaseModel):
    values: dict[str, float] = Field(default_factory=dict)

    def dominant_objective(self) -> str | None:
        if not self.values:
            return None
        return max(self.values, key=lambda key: float(self.values[key]))


class CompatibleSet(BaseModel):
    route_ids: tuple[str, ...] = Field(default_factory=tuple)
    blocked_reasons: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class PreferenceStopHint(BaseModel):
    code: str
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferenceState(BaseModel):
    compatible_set_summary: CompatibleSetSummary = Field(default_factory=CompatibleSetSummary)
    compatible_set: CompatibleSet = Field(default_factory=CompatibleSet)
    weights: PreferenceWeights = Field(default_factory=PreferenceWeights)
    frontier: list[dict[str, Any]] = Field(default_factory=list)
    selected_route_id: str | None = None
    stop_reason: str | None = None
    irrelevant_axes: tuple[str, ...] = Field(default_factory=tuple)
    vetoed_targets: tuple[str, ...] = Field(default_factory=tuple)
    certified_only_required: bool = False
    time_guard_required: bool = False
    stop_hints: tuple[PreferenceStopHint, ...] = Field(default_factory=tuple)
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

    def top_route_id(self) -> str | None:
        if self.selected_route_id in self.compatible_set.route_ids:
            return self.selected_route_id
        return self.compatible_set.route_ids[0] if self.compatible_set.route_ids else None

    def has_time_guard(self) -> bool:
        return bool(self.time_guard_required or self.time_preserving_guard_rules)

    def wants_certified_only(self) -> bool:
        return bool(self.certified_only_required or "uncertified" in self.vetoed_targets)


def empty_preference_state(*, route_ids: list[str] | None = None) -> PreferenceState:
    return PreferenceState(
        compatible_set_summary=CompatibleSetSummary(route_ids=list(route_ids or [])),
    )


def build_preference_state(
    request: Mapping[str, Any] | None = None,
    frontier: Sequence[Any] | None = None,
    *,
    elicited_constraints: Sequence[Any] | None = None,
    selected_route_id: str | None = None,
    stop_reason: str | None = None,
    route_ids: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    support_flag: bool = True,
    support_reason: str | None = None,
) -> PreferenceState:
    request_payload = dict(request or {})
    frontier_rows = [_route_payload(route) for route in (frontier or [])]
    route_id_list = _dedupe_route_ids(
        [_route_id(route) for route in frontier_rows] or [str(route_id) for route_id in (route_ids or [])]
    )
    request_weights = _normalize_weights(weights or _mapping_or_empty(request_payload.get("weights")))
    vetoed_targets, time_guard_limit_s = _summarize_elicited_constraints(elicited_constraints or ())
    certified_only_required = "uncertified" in vetoed_targets
    blocked_reasons = {
        route_id: tuple(reasons)
        for route_id in route_id_list
        if (reasons := _blocked_reasons(request_payload, _route_by_id(frontier_rows, route_id), vetoed_targets, time_guard_limit_s))
    }
    compatible_route_ids = tuple(route_id for route_id in route_id_list if route_id not in blocked_reasons)
    singleton_irrelevance = len(compatible_route_ids) <= 1
    certified_count = sum(1 for route in frontier_rows if _route_is_certified(route))
    stop_hints = _build_stop_hints(
        certified_only_required=certified_only_required,
        compatible_route_ids=compatible_route_ids,
        certified_count=certified_count,
        stop_reason=stop_reason,
    )
    summary = CompatibleSetSummary(
        route_ids=list(compatible_route_ids),
        necessary_best_prob=1.0 if singleton_irrelevance and compatible_route_ids and support_flag else 0.0,
        possible_best_prob=1.0,
        necessary_best_route_ids=list(compatible_route_ids[:1]) if singleton_irrelevance and support_flag else [],
        possible_best_route_ids=list(compatible_route_ids[:1] or compatible_route_ids),
        support_flag=bool(support_flag),
        support_reason=support_reason,
    )
    state = PreferenceState(
        compatible_set_summary=summary,
        compatible_set=CompatibleSet(route_ids=compatible_route_ids, blocked_reasons=blocked_reasons),
        weights=PreferenceWeights(values=request_weights),
        frontier=frontier_rows,
        selected_route_id=selected_route_id,
        stop_reason=stop_reason,
        irrelevant_axes=_irrelevant_axes(request_weights, frontier_rows),
        vetoed_targets=tuple(vetoed_targets),
        certified_only_required=certified_only_required,
        time_guard_required=time_guard_limit_s is not None,
        stop_hints=tuple(stop_hints),
        preference_irrelevance_proven=singleton_irrelevance,
    )
    if request_weights:
        state.compatible_weights = [request_weights]
    return state


def _route_payload(route: Any) -> dict[str, Any]:
    if isinstance(route, Mapping):
        return dict(route)
    if hasattr(route, "model_dump"):
        return dict(route.model_dump(mode="json"))
    return {
        "id": getattr(route, "id", None) or getattr(route, "route_id", None),
        "metrics": getattr(route, "metrics", {}),
        "certification": getattr(route, "certification", None),
        "segment_breakdown": getattr(route, "segment_breakdown", []),
        "uncertainty": getattr(route, "uncertainty", {}),
    }


def _route_id(route: Mapping[str, Any]) -> str | None:
    cleaned = str(route.get("id") or route.get("route_id") or "").strip()
    return cleaned or None


def _route_by_id(routes: Sequence[Mapping[str, Any]], route_id: str) -> Mapping[str, Any]:
    return next((route for route in routes if _route_id(route) == route_id), {})


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _mapping_or_attr(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _normalize_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    cleaned = {
        str(key): float(value)
        for key, value in weights.items()
        if value is not None and float(value) >= 0.0 and float(value) == float(value)
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return cleaned
    return {key: round(value / total, 12) for key, value in cleaned.items()}


def _summarize_elicited_constraints(
    constraints: Sequence[Any],
) -> tuple[tuple[str, ...], float | None]:
    vetoed_targets: list[str] = []
    time_guard_limit_s: float | None = None
    for constraint in constraints:
        kind = _constraint_kind(constraint)
        if "veto" in kind:
            target = _constraint_text(constraint, ("target", "veto_name", "name", "value"))
            vetoed_targets.append(target or "uncertified")
        if "time_guard" in kind or "time" in kind:
            guard = _mapping_or_attr(constraint, "guard") or _mapping_or_attr(constraint, "time_guard") or constraint
            limit = _numeric_attr(guard, ("max_duration_s", "max_travel_time_s", "max_travel_time"))
            if limit is not None:
                time_guard_limit_s = limit if time_guard_limit_s is None else min(time_guard_limit_s, limit)
    return tuple(_dedupe_route_ids(vetoed_targets)), time_guard_limit_s


def _constraint_kind(constraint: Any) -> str:
    for key in ("kind", "constraint_type", "type", "query_type"):
        value = _mapping_or_attr(constraint, key)
        if value:
            return str(value).strip().lower()
    if _constraint_text(constraint, ("target", "veto_name", "name")):
        return "veto"
    if _numeric_attr(constraint, ("max_duration_s", "max_travel_time_s", "max_travel_time")) is not None:
        return "time_guard"
    return ""


def _constraint_text(constraint: Any, keys: Sequence[str]) -> str | None:
    for key in keys:
        value = _mapping_or_attr(constraint, key)
        if value:
            cleaned = str(value).strip()
            if cleaned:
                return cleaned
    return None


def _numeric_attr(value: Any, keys: Sequence[str]) -> float | None:
    for key in keys:
        raw_value = _mapping_or_attr(value, key)
        if raw_value is None:
            continue
        return float(raw_value)
    return None


def _blocked_reasons(
    request: Mapping[str, Any],
    route: Mapping[str, Any],
    vetoed_targets: Sequence[str],
    time_guard_limit_s: float | None,
) -> list[str]:
    reasons: list[str] = []
    cost_toggles = _mapping_or_empty(request.get("cost_toggles"))
    if cost_toggles.get("use_tolls") is False and _route_has_toll(route):
        reasons.append("toggle_use_tolls")
    if "uncertified" in vetoed_targets and not _route_is_certified(route):
        reasons.append("veto_uncertified")
    duration_s = _metric_value(route, "duration_s")
    if time_guard_limit_s is not None and duration_s is not None and duration_s > time_guard_limit_s:
        reasons.append("time_guard:max_duration_s")
    return reasons


def _route_has_toll(route: Mapping[str, Any]) -> bool:
    for segment in route.get("segment_breakdown") or []:
        if isinstance(segment, Mapping) and float(segment.get("toll_cost") or 0.0) > 0.0:
            return True
    return float(_metric_value(route, "toll_cost") or 0.0) > 0.0


def _route_is_certified(route: Mapping[str, Any]) -> bool:
    certification = _mapping_or_attr(route, "certification")
    if certification is None:
        return False
    certified = _mapping_or_attr(certification, "certified")
    if certified is not None:
        return bool(certified)
    certificate = _mapping_or_attr(certification, "certificate")
    threshold = _mapping_or_attr(certification, "threshold")
    if certificate is None or threshold is None:
        return False
    return float(certificate) >= float(threshold)


def _metric_value(route: Mapping[str, Any], metric_name: str) -> float | None:
    metrics = _mapping_or_attr(route, "metrics")
    value = _mapping_or_attr(metrics, metric_name)
    if value is None:
        return None
    return float(value)


def _irrelevant_axes(
    weights: Mapping[str, float],
    routes: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    metric_by_axis = {
        "time": "duration_s",
        "money": "monetary_cost",
        "co2": "emissions_kg",
    }
    irrelevant: list[str] = []
    for axis, metric_name in metric_by_axis.items():
        if float(weights.get(axis, 0.0)) <= 0.0:
            irrelevant.append(axis)
            continue
        values = [_metric_value(route, metric_name) for route in routes]
        numeric_values = [value for value in values if value is not None]
        if len(numeric_values) < 2:
            continue
        span = max(numeric_values) - min(numeric_values)
        scale = max(abs(value) for value in numeric_values) or 1.0
        if span / scale <= 0.05:
            irrelevant.append(axis)
    return tuple(irrelevant)


def _build_stop_hints(
    *,
    certified_only_required: bool,
    compatible_route_ids: Sequence[str],
    certified_count: int,
    stop_reason: str | None,
) -> list[PreferenceStopHint]:
    if not certified_only_required or compatible_route_ids:
        return []
    return [
        PreferenceStopHint(
            code="typed_abstention_recommended",
            message="No certified route remains compatible with current preferences.",
            metadata={
                "trigger_metric": "certified_route_count",
                "observed_value": float(certified_count),
                "recommended_action": "expand_worlds",
                "severity": "high",
                "stop_reason": stop_reason,
            },
        )
    ]


def _dedupe_route_ids(values: Sequence[str | None]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


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
