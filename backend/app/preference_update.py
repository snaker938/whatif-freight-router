from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .preference_model import (
    ElicitedConstraint,
    OBJECTIVE_NAMES,
    ObjectiveName,
    PreferenceWeights,
    TimeGuard,
    _as_text,
)
from .preference_queries import PreferenceQuery
from .preference_state import PreferenceState, build_preference_state


@dataclass(frozen=True)
class PreferenceUpdate:
    focus_objective: ObjectiveName | None = None
    focus_delta: float = 0.2
    time_guard: TimeGuard | None = None
    require_certified: bool | None = None
    avoid_tolls: bool | None = None
    optimization_mode: str | None = None
    route_veto_ids: tuple[str, ...] = ()
    extra_constraints: tuple[ElicitedConstraint, ...] = ()
    remove_constraint_keys: tuple[str, ...] = ()


def _rebalance_for_objective(
    weights: PreferenceWeights,
    objective: ObjectiveName,
    *,
    delta: float = 0.2,
) -> PreferenceWeights:
    current = weights.normalized().as_dict()
    remaining_axes = [axis for axis in OBJECTIVE_NAMES if axis != objective]
    transfer = max(0.0, min(delta, sum(current[axis] for axis in remaining_axes)))
    remaining_total = sum(current[axis] for axis in remaining_axes)
    if remaining_total <= 0.0:
        current[objective] = 1.0
        for axis in remaining_axes:
            current[axis] = 0.0
    else:
        for axis in remaining_axes:
            share = current[axis] / remaining_total
            current[axis] = max(0.0, current[axis] - transfer * share)
        current[objective] = current[objective] + transfer
    return PreferenceWeights(
        time=current["time"],
        money=current["money"],
        co2=current["co2"],
    )


def _dedupe_constraints(constraints: Sequence[ElicitedConstraint]) -> tuple[ElicitedConstraint, ...]:
    ordered: dict[str, ElicitedConstraint] = {}
    for constraint in constraints:
        ordered[constraint.key()] = constraint
    return tuple(ordered.values())


def apply_preference_update(state: PreferenceState, update: PreferenceUpdate) -> PreferenceState:
    request_context = dict(state.request_context)
    cost_toggles = dict(request_context.get("cost_toggles") or {})
    constraints = list(state.elicited_constraints)
    remove_keys = set(update.remove_constraint_keys)

    weights = state.weights
    if update.focus_objective is not None:
        weights = _rebalance_for_objective(weights, update.focus_objective, delta=update.focus_delta)
        request_context["weights"] = weights.as_dict()

    if update.optimization_mode is not None:
        request_context["optimization_mode"] = update.optimization_mode

    if update.avoid_tolls is not None:
        cost_toggles["use_tolls"] = not update.avoid_tolls
        request_context["cost_toggles"] = cost_toggles
        if update.avoid_tolls:
            constraints.append(ElicitedConstraint.veto("tolls", source="preference_update"))
        else:
            remove_keys.update({"veto_tolls", "toggle_use_tolls"})

    if update.require_certified is True:
        constraints.append(ElicitedConstraint.veto("uncertified", source="preference_update"))
    elif update.require_certified is False:
        remove_keys.add("veto_uncertified")

    if update.time_guard is not None:
        if update.time_guard.is_active():
            constraints.append(ElicitedConstraint.time_guard(update.time_guard, source="preference_update"))
        else:
            remove_keys.add("time_guard")

    for route_id in update.route_veto_ids:
        constraints.append(ElicitedConstraint.veto(route_id, source="preference_update"))

    constraints.extend(update.extra_constraints)
    retained_constraints = [constraint for constraint in constraints if constraint.key() not in remove_keys]
    deduped_constraints = _dedupe_constraints(retained_constraints)

    return build_preference_state(
        request_context,
        state.frontier,
        elicited_constraints=deduped_constraints,
        selected_route_id=state.selected_route_id,
        selected_certificate=state.selected_certificate,
        stop_reason=state.stop_reason,
    )


def _bool_answer(answer: Any) -> bool | None:
    if isinstance(answer, bool):
        return answer
    answer_text = (_as_text(answer) or "").strip().lower()
    if answer_text in {"yes", "true", "prefer_certified", "robust", "veto", "avoid", "set"}:
        return True
    if answer_text in {"no", "false", "allow_uncertified", "allow", "expected_value", "no_guard"}:
        return False
    return None


def _time_guard_from_answer(answer: Any) -> TimeGuard:
    if isinstance(answer, TimeGuard):
        return answer
    if isinstance(answer, dict):
        latest_arrival = _as_text(answer.get("latest_arrival_utc"))
        max_duration_raw = answer.get("max_duration_s")
        max_weather_raw = answer.get("max_weather_delay_s")
        max_incident_raw = answer.get("max_incident_delay_s")
        return TimeGuard(
            max_duration_s=float(max_duration_raw) if max_duration_raw is not None else None,
            max_weather_delay_s=float(max_weather_raw) if max_weather_raw is not None else None,
            max_incident_delay_s=float(max_incident_raw) if max_incident_raw is not None else None,
            latest_arrival_utc=latest_arrival,
        )
    answer_text = _as_text(answer)
    if answer_text is None:
        return TimeGuard()
    if answer_text.replace(".", "", 1).isdigit():
        return TimeGuard(max_duration_s=float(answer_text))
    return TimeGuard(latest_arrival_utc=answer_text)


def apply_query_answer(state: PreferenceState, query: PreferenceQuery, answer: Any) -> PreferenceState:
    if query.kind == "certified_focus":
        choice = _bool_answer(answer)
        return apply_preference_update(
            state,
            PreferenceUpdate(require_certified=True if choice is None else choice),
        )

    if query.kind == "route_veto":
        choice = _bool_answer(answer)
        if query.target == "tolls":
            return apply_preference_update(state, PreferenceUpdate(avoid_tolls=bool(choice)))
        if choice:
            return apply_preference_update(state, PreferenceUpdate(route_veto_ids=(query.target or "",)))
        return apply_preference_update(
            state,
            PreferenceUpdate(remove_constraint_keys=(f"veto_{query.target}",) if query.target else ()),
        )

    if query.kind == "time_guard":
        choice = _bool_answer(answer)
        if choice is False:
            return apply_preference_update(state, PreferenceUpdate(time_guard=TimeGuard()))
        return apply_preference_update(state, PreferenceUpdate(time_guard=_time_guard_from_answer(answer)))

    if query.kind == "optimization_mode":
        mode = (_as_text(answer) or "").strip().lower()
        if mode not in {"robust", "expected_value"}:
            mode = "robust"
        return apply_preference_update(state, PreferenceUpdate(optimization_mode=mode))

    if query.kind == "objective_tradeoff":
        objective = (_as_text(answer) or "").strip().lower()
        if objective in OBJECTIVE_NAMES:
            return apply_preference_update(state, PreferenceUpdate(focus_objective=objective))

    return state


__all__ = [
    "PreferenceUpdate",
    "_dedupe_constraints",
    "_rebalance_for_objective",
    "apply_preference_update",
    "apply_query_answer",
]
