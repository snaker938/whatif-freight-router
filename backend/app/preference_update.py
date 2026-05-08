"""Pipeline stage: apply answered preference queries and record monotone preference-state updates."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field, field_validator

from .preference_model import build_preference_shrinkage_trace
from .preference_queries import PreferenceQuery
from .preference_state import (
    CompatibleSet,
    CompatibleSetSummary,
    PreferenceStopHint,
    PreferenceState,
    PreferenceWeights,
    detect_preference_contradictions,
    sync_route_level_best_markings,
    volume_trace_nonincreasing,
)


class PreferenceUpdate(BaseModel):
    focus_objective: str | None = None
    focus_delta: float = Field(default=0.0, ge=0.0)

    @field_validator("focus_objective")
    @classmethod
    def _normalize_focus_objective(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


def apply_preference_update(
    state: PreferenceState,
    *,
    update: PreferenceUpdate,
) -> PreferenceState:
    updated = state.model_copy(deep=True)
    focus = update.focus_objective
    if focus is None:
        return updated

    weights = dict(getattr(updated.weights, "values", {}) or {})
    if focus not in weights:
        weights[focus] = 0.0
    updated_weights = _shift_weight_focus(weights, focus, float(update.focus_delta))
    updated.weights = PreferenceWeights(values=updated_weights)
    updated.compatible_weights = [updated_weights] if updated_weights else []
    return updated


def apply_query_answer(
    state: PreferenceState,
    query: PreferenceQuery,
    answer: str,
) -> PreferenceState:
    updated = state.model_copy(deep=True)
    query_kind = str(getattr(query, "kind", None) or getattr(query, "query_type", "")).strip()
    answer_value = str(answer or "").strip()
    if query_kind == "certified_focus" and answer_value == "prefer_certified":
        updated.certified_only_required = True
        vetoed_targets = list(getattr(updated, "vetoed_targets", ()) or ())
        if "uncertified" not in vetoed_targets:
            vetoed_targets.append("uncertified")
        updated.vetoed_targets = tuple(vetoed_targets)

        certified_route_ids = {
            route_id
            for route in getattr(updated, "frontier", []) or []
            if (route_id := _route_id(route)) and _route_is_certified(route)
        }
        current_route_ids = tuple(getattr(updated.compatible_set, "route_ids", ()) or ())
        surviving_route_ids = tuple(route_id for route_id in current_route_ids if route_id in certified_route_ids)
        blocked_reasons = dict(getattr(updated.compatible_set, "blocked_reasons", {}) or {})
        for route in getattr(updated, "frontier", []) or []:
            route_id = _route_id(route)
            if not route_id or _route_is_certified(route):
                continue
            reasons = list(blocked_reasons.get(route_id, ()))
            if "veto_uncertified" not in reasons:
                reasons.append("veto_uncertified")
            blocked_reasons[route_id] = tuple(reasons)

        updated.compatible_set = CompatibleSet(
            route_ids=surviving_route_ids,
            blocked_reasons=blocked_reasons,
        )
        updated.compatible_set_summary = CompatibleSetSummary(
            route_ids=list(surviving_route_ids),
            possible_best_route_ids=list(surviving_route_ids),
            necessary_best_route_ids=list(surviving_route_ids[:1]) if len(surviving_route_ids) == 1 else [],
            necessary_best_prob=1.0 if len(surviving_route_ids) == 1 else 0.0,
            possible_best_prob=1.0,
            support_flag=updated.compatible_set_summary.support_flag,
            support_reason=updated.compatible_set_summary.support_reason,
        )
        if not surviving_route_ids:
            updated.stop_hints = _with_stop_hint(updated.stop_hints)

    updated.contradiction_record = detect_preference_contradictions(updated)
    sync_route_level_best_markings(updated)
    updated.derived_invariants = validate_preference_invariants(updated)
    return updated


def append_preference_query(
    state: PreferenceState,
    query: PreferenceQuery,
    *,
    before_size: int | None = None,
    after_size: int | None = None,
    before_volume_proxy: float | None = None,
    after_volume_proxy: float | None = None,
    target_route_id: str | None = None,
    query_reason: str | None = None,
    preference_irrelevance: bool = False,
) -> PreferenceState:
    updated = state.model_copy(deep=True)
    updated.query_history.append(query)
    updated.query_count = len(updated.query_history)

    if query.query_type == "pairwise":
        updated.pairwise_constraints.append(query)
    elif query.query_type == "threshold":
        updated.threshold_constraints.append(query)
    elif query.query_type == "ratio":
        updated.ratio_constraints.append(query)
    elif query.query_type == "veto":
        updated.veto_rules.append(query)
    elif query.query_type == "time_guard":
        updated.time_preserving_guard_rules.append(query)

    if before_size is not None and after_size is not None:
        before_size_int = max(0, int(before_size))
        after_size_int = max(0, int(after_size))
        if after_size_int > before_size_int:
            raise ValueError("valid preference updates cannot increase compatible set size")
        before_volume_value = (
            float(before_volume_proxy)
            if before_volume_proxy is not None
            else float(max(1, before_size_int))
        )
        after_volume_value = (
            float(after_volume_proxy)
            if after_volume_proxy is not None
            else float(max(0, after_size_int))
        )
        if after_volume_value > before_volume_value + 1e-12:
            raise ValueError("valid preference updates cannot increase compatible-set volume")
        updated.shrinkage_trace.append(
            build_preference_shrinkage_trace(
                query_index=len(updated.query_history) - 1,
                query_type=query.query_type,
                before_size=before_size_int,
                after_size=after_size_int,
                before_volume_proxy=before_volume_value,
                after_volume_proxy=after_volume_value,
                target_route_id=target_route_id,
                query_reason=query_reason,
                preference_irrelevance=preference_irrelevance,
            )
        )
        updated.compatible_set_summary.compatible_set_size = after_size_int
        if after_volume_proxy is not None:
            updated.compatible_set_summary.compatible_set_volume_proxy = max(0.0, float(after_volume_proxy))
        updated.preference_irrelevance_proven = bool(
            preference_irrelevance or updated.compatible_set_summary.compatible_set_size <= 1
        )

    updated.contradiction_record = detect_preference_contradictions(updated)
    sync_route_level_best_markings(updated)
    updated.derived_invariants = validate_preference_invariants(updated)
    return updated


def validate_preference_invariants(state: PreferenceState) -> dict[str, bool]:
    summary = state.compatible_set_summary
    return {
        "necessary_best_prob_le_possible_best_prob": summary.necessary_best_prob <= summary.possible_best_prob,
        "no_necessary_best_without_possible_best": set(summary.necessary_best_route_ids).issubset(
            set(summary.possible_best_route_ids)
        ),
        "compatible_set_volume_nonincreasing_after_query": volume_trace_nonincreasing(state),
        "preference_contradiction_free": not state.contradiction_record.contradiction_detected,
        "query_history_matches_trace_or_zero": len(state.shrinkage_trace) == 0
        or len(state.shrinkage_trace) <= len(state.query_history),
    }


def _shift_weight_focus(
    weights: Mapping[str, float],
    focus: str,
    delta: float,
) -> dict[str, float]:
    normalized = _normalize_weights(weights)
    if not normalized:
        return {focus: 1.0}
    focus_value = min(1.0, float(normalized.get(focus, 0.0)) + max(0.0, delta))
    remaining_total = max(0.0, 1.0 - focus_value)
    other_total = sum(value for key, value in normalized.items() if key != focus)
    shifted: dict[str, float] = {focus: round(focus_value, 12)}
    for key, value in normalized.items():
        if key == focus:
            continue
        shifted[key] = round((float(value) / other_total) * remaining_total, 12) if other_total > 0.0 else 0.0
    return shifted


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    cleaned = {
        str(key): float(value)
        for key, value in weights.items()
        if value is not None and float(value) >= 0.0 and float(value) == float(value)
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return cleaned
    return {key: round(value / total, 12) for key, value in cleaned.items()}


def _route_id(route: Any) -> str | None:
    if isinstance(route, Mapping):
        raw_value = route.get("id") or route.get("route_id")
    else:
        raw_value = getattr(route, "id", None) or getattr(route, "route_id", None)
    cleaned = str(raw_value or "").strip()
    return cleaned or None


def _route_is_certified(route: Any) -> bool:
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


def _mapping_or_attr(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _with_stop_hint(existing_hints: tuple[PreferenceStopHint, ...]) -> tuple[PreferenceStopHint, ...]:
    if any(hint.code == "typed_abstention_recommended" for hint in existing_hints):
        return existing_hints
    return (
        *existing_hints,
        PreferenceStopHint(
            code="typed_abstention_recommended",
            message="No certified route remains compatible with current preferences.",
            metadata={
                "trigger_metric": "certified_route_count",
                "observed_value": 0.0,
                "recommended_action": "expand_worlds",
                "severity": "high",
            },
        ),
    )
