"""Backend-backed runtime helpers for frontend preference-elicitation updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from .models import RouteOption, _build_preference_summary
from .preference_state import PreferenceState


class PreferenceRuntimeUpdateRequest(BaseModel):
    candidate_routes: list[RouteOption] = Field(default_factory=list, min_length=1)
    selected_route_id: str | None = None
    selected_certificate_basis: str | None = None
    pipeline_mode: str | None = None
    support_flag: bool | None = None
    support_reason: str | None = None
    preference_state: PreferenceState


class PreferenceRuntimeUpdateResponse(BaseModel):
    selected_route_id: str | None = None
    selected_certificate_basis: str | None = None
    pipeline_mode: str | None = None
    terminal_type: str | None = None
    preference_state: PreferenceState
    preference_query_trace: dict[str, Any] = Field(default_factory=dict)
    preference_summary: dict[str, Any] = Field(default_factory=dict)


def _unique_route_ids(routes: Sequence[RouteOption]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for route in routes:
        route_id = str(route.id).strip()
        if not route_id or route_id in seen:
            continue
        seen.add(route_id)
        ordered.append(route_id)
    return ordered


def _serialize_records(
    values: Sequence[Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for value in values:
        if hasattr(value, "model_dump"):
            out.append(value.model_dump(mode="json"))
        elif isinstance(value, Mapping):
            out.append(dict(value))
    return out


def _resolve_selected_route_id(
    *,
    preference_state: PreferenceState,
    requested_selected_route_id: str | None,
    route_ids: Sequence[str],
) -> str | None:
    route_id_set = set(route_ids)
    candidates = [
        *list(preference_state.compatible_set_summary.necessary_best_route_ids or []),
        *list(preference_state.compatible_set_summary.possible_best_route_ids or []),
        str(requested_selected_route_id or "").strip(),
        *list(route_ids),
    ]
    for route_id in candidates:
        cleaned = str(route_id).strip()
        if cleaned and cleaned in route_id_set:
            return cleaned
    return None


def _normalize_preference_state(
    *,
    preference_state: PreferenceState,
    route_ids: Sequence[str],
    support_flag: bool | None,
    support_reason: str | None,
    selected_route_id: str | None,
) -> PreferenceState:
    normalized = PreferenceState.model_validate(preference_state.model_dump(mode="json"))
    summary = normalized.compatible_set_summary
    route_id_set = set(route_ids)

    active_route_ids = [
        route_id
        for route_id in list(summary.route_ids or [])
        if route_id in route_id_set
    ]
    if normalized.query_count <= 0 or not active_route_ids:
        active_route_ids = list(route_ids)
    if support_flag is not None:
        summary.support_flag = bool(support_flag)
    if support_reason is not None:
        summary.support_reason = support_reason
    summary.necessary_best_route_ids = [
        route_id
        for route_id in list(summary.necessary_best_route_ids or [])
        if route_id in route_id_set
    ]
    summary.possible_best_route_ids = [
        route_id
        for route_id in list(summary.possible_best_route_ids or [])
        if route_id in route_id_set
    ]
    if normalized.query_count > 0:
        normalized.no_query_reason = None
    if normalized.preference_irrelevance_proven and selected_route_id and selected_route_id in route_id_set:
        active_route_ids = [selected_route_id]
        summary.compatible_set_size = 1
        summary.necessary_best_route_ids = [selected_route_id] if bool(summary.support_flag) else []
        summary.possible_best_route_ids = [selected_route_id]
        summary.necessary_best_prob = 1.0 if bool(summary.support_flag) else 0.0
        summary.possible_best_prob = 1.0
    elif not summary.possible_best_route_ids and selected_route_id and selected_route_id in route_id_set:
        summary.possible_best_route_ids = [selected_route_id]
    summary.route_ids = active_route_ids
    return PreferenceState.model_validate(normalized.model_dump(mode="json"))


def build_preference_query_trace_payload(
    *,
    preference_state: PreferenceState,
    selected_route_id: str | None,
    selected_certificate_basis: str | None,
    pipeline_mode: str,
    support_flag: bool | None = None,
    support_reason: str | None = None,
) -> dict[str, Any]:
    effective_selected_route_id = str(selected_route_id or "").strip() or None
    effective_basis = str(selected_certificate_basis or "").strip() or "empirical"
    summary = _build_preference_summary(
        preference_state=preference_state,
        selected_certificate_basis=effective_basis,
        pipeline_mode=pipeline_mode,
    )
    shrinkage_trace = _serialize_records(preference_state.shrinkage_trace)
    last_trace = shrinkage_trace[-1] if shrinkage_trace else {}
    return {
        "schema_version": "preference-query-trace-v1",
        "selected_route_id": effective_selected_route_id,
        "selected_certificate_basis": effective_basis,
        "terminal_type": preference_state.terminal_type,
        "query_count": int(summary.get("query_count", 0) or 0),
        "query_history": _serialize_records(preference_state.query_history),
        "shrinkage_trace": shrinkage_trace,
        "compatible_set_summary": summary.get("compatible_set_summary", {}),
        "derived_invariants": summary.get("derived_invariants", {}),
        "contradiction_record": summary.get("contradiction_record", {}),
        "preference_irrelevance_proven": bool(summary.get("preference_irrelevance_proven", False)),
        "no_query_reason": summary.get("no_query_reason"),
        "no_preference_query_reason": summary.get("no_preference_query_reason")
        or summary.get("no_query_reason"),
        "targeted_challenger_route_id": last_trace.get("target_route_id")
        or summary.get("targeted_challenger_route_id"),
        "query_selection_reason": last_trace.get("query_reason")
        or summary.get("query_selection_reason"),
        "provenance": {
            "selected_route_id": effective_selected_route_id,
            "pipeline_mode": pipeline_mode,
            "support_flag": support_flag,
            "support_reason": support_reason,
        },
    }


def apply_preference_runtime_update(
    request: PreferenceRuntimeUpdateRequest,
) -> PreferenceRuntimeUpdateResponse:
    route_ids = _unique_route_ids(request.candidate_routes)
    if not route_ids:
        raise ValueError("candidate_routes must include at least one route id")

    selected_certificate_basis = str(request.selected_certificate_basis or "").strip() or "empirical"
    pipeline_mode = str(request.pipeline_mode or "").strip() or "dccs_refc"
    selected_route_id = _resolve_selected_route_id(
        preference_state=request.preference_state,
        requested_selected_route_id=request.selected_route_id,
        route_ids=route_ids,
    )
    normalized_state = _normalize_preference_state(
        preference_state=request.preference_state,
        route_ids=route_ids,
        support_flag=request.support_flag,
        support_reason=request.support_reason,
        selected_route_id=selected_route_id,
    )
    resolved_selected_route_id = _resolve_selected_route_id(
        preference_state=normalized_state,
        requested_selected_route_id=selected_route_id,
        route_ids=route_ids,
    )
    preference_summary = _build_preference_summary(
        preference_state=normalized_state,
        selected_certificate_basis=selected_certificate_basis,
        pipeline_mode=pipeline_mode,
    )
    preference_query_trace = build_preference_query_trace_payload(
        preference_state=normalized_state,
        selected_route_id=resolved_selected_route_id,
        selected_certificate_basis=selected_certificate_basis,
        pipeline_mode=pipeline_mode,
        support_flag=request.support_flag,
        support_reason=request.support_reason,
    )
    return PreferenceRuntimeUpdateResponse(
        selected_route_id=resolved_selected_route_id,
        selected_certificate_basis=selected_certificate_basis,
        pipeline_mode=pipeline_mode,
        terminal_type=normalized_state.terminal_type,
        preference_state=normalized_state,
        preference_query_trace=preference_query_trace,
        preference_summary=preference_summary,
    )
