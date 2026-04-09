from __future__ import annotations

from .preference_model import build_preference_shrinkage_trace
from .preference_queries import PreferenceQuery
from .preference_state import PreferenceState


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
        updated.shrinkage_trace.append(
            build_preference_shrinkage_trace(
                query_index=len(updated.query_history) - 1,
                query_type=query.query_type,
                before_size=before_size,
                after_size=after_size,
                before_volume_proxy=(
                    float(before_volume_proxy)
                    if before_volume_proxy is not None
                    else float(max(1, before_size))
                ),
                after_volume_proxy=(
                    float(after_volume_proxy)
                    if after_volume_proxy is not None
                    else float(max(0, after_size))
                ),
                target_route_id=target_route_id,
                query_reason=query_reason,
                preference_irrelevance=preference_irrelevance,
            )
        )
        updated.compatible_set_summary.compatible_set_size = max(0, int(after_size))
        if after_volume_proxy is not None:
            updated.compatible_set_summary.compatible_set_volume_proxy = max(0.0, float(after_volume_proxy))

    updated.derived_invariants = validate_preference_invariants(updated)
    return updated


def validate_preference_invariants(state: PreferenceState) -> dict[str, bool]:
    summary = state.compatible_set_summary
    return {
        "necessary_best_prob_le_possible_best_prob": summary.necessary_best_prob <= summary.possible_best_prob,
        "no_necessary_best_without_possible_best": not (
            summary.necessary_best_prob > 0.0 and summary.possible_best_prob <= 0.0
        ),
        "compatible_set_volume_nonincreasing_after_query": _volume_trace_nonincreasing(state),
        "query_history_matches_trace_or_zero": len(state.shrinkage_trace) == 0
        or len(state.shrinkage_trace) <= len(state.query_history),
    }


def _volume_trace_nonincreasing(state: PreferenceState) -> bool:
    trace = state.shrinkage_trace
    if len(trace) < 2:
        return True
    return all(
        earlier.after_volume_proxy >= later.after_volume_proxy
        for earlier, later in zip(trace, trace[1:])
    )

