from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .preference_model import ElicitedConstraint, PreferenceStopHint, RoutePreferenceSummary, _as_text
from .preference_state import PreferenceState

QueryKind = Literal[
    "certified_focus",
    "time_guard",
    "route_veto",
    "objective_tradeoff",
    "optimization_mode",
]


@dataclass(frozen=True)
class PreferenceQuery:
    key: str
    kind: QueryKind
    prompt: str
    rationale: str
    target: str | None = None
    options: tuple[str, ...] = ()
    route_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _route_by_id(state: PreferenceState, route_id: str) -> RoutePreferenceSummary | None:
    for route in state.frontier:
        if route.route_id == route_id:
            return route
    return None


def _top_routes(state: PreferenceState, limit: int = 2) -> tuple[RoutePreferenceSummary, ...]:
    route_ids = state.compatible_set.ranked_route_ids[:limit]
    routes = tuple(route for route_id in route_ids if (route := _route_by_id(state, route_id)) is not None)
    if routes:
        return routes
    return tuple(state.frontier[:limit])


def _has_veto(state: PreferenceState, target: str) -> bool:
    target_key = target.strip().lower()
    return any(
        constraint.kind == "route_veto" and constraint.target.strip().lower() == target_key
        for constraint in state.elicited_constraints
    )


def _dedupe_queries(queries: list[PreferenceQuery], limit: int) -> tuple[PreferenceQuery, ...]:
    ordered: dict[str, PreferenceQuery] = {}
    for query in queries:
        ordered.setdefault(query.key, query)
        if len(ordered) >= limit:
            break
    return tuple(ordered.values())


def preference_stop_hints(state: PreferenceState) -> tuple[PreferenceStopHint, ...]:
    return state.stop_hints


def suggest_preference_queries(state: PreferenceState, *, limit: int = 3) -> tuple[PreferenceQuery, ...]:
    if limit <= 0:
        return ()

    queries: list[PreferenceQuery] = []
    top_routes = _top_routes(state, limit=2)
    top_route = top_routes[0] if top_routes else None
    runner_up = top_routes[1] if len(top_routes) > 1 else None

    uncertified_leader = top_route is not None and not top_route.certified
    no_certified_compatible = bool(state.compatible_set.route_ids) and not state.compatible_set.certified_route_ids
    if uncertified_leader or no_certified_compatible:
        queries.append(
            PreferenceQuery(
                key="certified_focus",
                kind="certified_focus",
                prompt="Should routing require a certified route and abstain if no compatible route is certified?",
                rationale="The current preference leader is uncertified or the compatible set has no certified option.",
                options=("prefer_certified", "allow_uncertified"),
                route_ids=state.compatible_set.ranked_route_ids[:3],
                metadata={
                    "certificate_threshold": state.certificate_threshold,
                    "compatible_certified_route_ids": list(state.compatible_set.certified_route_ids),
                },
            )
        )

    mixed_tolls = any(route.uses_tolls for route in state.frontier) and any(not route.uses_tolls for route in state.frontier)
    if mixed_tolls and not _has_veto(state, "tolls") and state.use_tolls:
        queries.append(
            PreferenceQuery(
                key="route_veto:tolls",
                kind="route_veto",
                prompt="Should tolled routes be vetoed for this decision?",
                rationale="The compatible frontier mixes tolled and toll-free routes.",
                target="tolls",
                options=("veto_tolls", "allow_tolls"),
                route_ids=tuple(route.route_id for route in state.frontier if route.uses_tolls)[:2],
                metadata={"use_tolls": state.use_tolls},
            )
        )

    if not state.time_guard.is_active() and any(
        route.weather_delay_s > 0.0 or route.incident_delay_s > 0.0 for route in state.frontier
    ):
        fastest_route = min(state.frontier, key=lambda route: route.duration_s, default=None)
        suggested_guard = None
        if fastest_route is not None:
            suggested_guard = round(fastest_route.duration_s + 900.0, 3)
        queries.append(
            PreferenceQuery(
                key="time_guard",
                kind="time_guard",
                prompt="Do you want a hard travel-time or arrival-time guard on this route choice?",
                rationale="Delay-sensitive routes remain in the frontier but no explicit time guard is active.",
                options=("set_max_duration", "set_latest_arrival", "no_guard"),
                route_ids=tuple(route.route_id for route in top_routes),
                metadata={"suggested_max_duration_s": suggested_guard},
            )
        )

    if (
        state.optimization_mode != "robust"
        and top_route is not None
        and runner_up is not None
        and top_route.uncertainty_mass > runner_up.uncertainty_mass + 0.15
    ):
        queries.append(
            PreferenceQuery(
                key="optimization_mode",
                kind="optimization_mode",
                prompt="Should the preference model switch to robust mode for this decision?",
                rationale="The current leader carries materially more uncertainty than the runner-up.",
                options=("robust", "expected_value"),
                route_ids=(top_route.route_id, runner_up.route_id),
                metadata={"current_mode": state.optimization_mode},
            )
        )

    if top_route is not None and runner_up is not None:
        material_tradeoff = (
            abs(top_route.duration_s - runner_up.duration_s) > 600.0
            or abs(top_route.monetary_cost - runner_up.monetary_cost) > 10.0
            or abs(top_route.emissions_kg - runner_up.emissions_kg) > 5.0
        )
        if material_tradeoff:
            queries.append(
                PreferenceQuery(
                    key=f"objective_tradeoff:{top_route.route_id}:{runner_up.route_id}",
                    kind="objective_tradeoff",
                    prompt="Which objective should dominate this tradeoff: time, money, or CO2?",
                    rationale="The top compatible routes separate on different objectives under the current weights.",
                    options=("time", "money", "co2"),
                    route_ids=(top_route.route_id, runner_up.route_id),
                    metadata={
                        "dominant_objective": state.weights.dominant_objective(),
                        "irrelevant_axes": list(state.irrelevant_axes),
                    },
                )
            )

    return _dedupe_queries(queries, limit)


def suggest_preference_query(state: PreferenceState) -> PreferenceQuery | None:
    queries = suggest_preference_queries(state, limit=1)
    return queries[0] if queries else None


def constraint_for_query_answer(query: PreferenceQuery, answer: Any) -> ElicitedConstraint | None:
    answer_text = (_as_text(answer) or "").lower()
    if query.kind == "route_veto" and answer_text in {"yes", "veto", "veto_tolls", "avoid"} and query.target:
        return ElicitedConstraint.veto(query.target, source="query_answer")
    return None


__all__ = [
    "PreferenceQuery",
    "QueryKind",
    "constraint_for_query_answer",
    "preference_stop_hints",
    "suggest_preference_queries",
    "suggest_preference_query",
]
