"""Pipeline stage: apply answered preference queries and record monotone preference-state updates."""

from __future__ import annotations

from .preference_model import build_preference_shrinkage_trace
from .preference_queries import PreferenceQuery
from .preference_state import (
    PreferenceState,
    detect_preference_contradictions,
    sync_route_level_best_markings,
    volume_trace_nonincreasing,
)


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
