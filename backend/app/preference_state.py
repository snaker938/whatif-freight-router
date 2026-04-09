from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .preference_model import (
    CompatibleSet,
    ElicitedConstraint,
    OBJECTIVE_IRRELEVANCE_FLOORS,
    OBJECTIVE_NAMES,
    ObjectiveName,
    PreferenceStopHint,
    PreferenceWeights,
    RoutePreferenceSummary,
    TimeGuard,
    _as_bool,
    _as_float,
    _as_text,
    _mapping_from_any,
    _parse_iso_utc,
)


def _empty_compatible_set() -> CompatibleSet:
    return CompatibleSet(
        route_ids=(),
        ranked_route_ids=(),
        certified_route_ids=(),
        vetoed_route_ids=(),
    )


@dataclass(frozen=True)
class PreferenceState:
    request_context: dict[str, Any] = field(default_factory=dict, repr=False)
    weights: PreferenceWeights = field(default_factory=PreferenceWeights)
    optimization_mode: str = "expected_value"
    risk_aversion: float = 1.0
    use_tolls: bool = True
    fuel_price_multiplier: float = 1.0
    carbon_price_per_kg: float = 0.0
    toll_cost_per_km: float = 0.0
    departure_time_utc: str | None = None
    time_guard: TimeGuard = field(default_factory=TimeGuard)
    search_budget: int | None = None
    evidence_budget: int | None = None
    certificate_threshold: float | None = None
    tau_stop: float | None = None
    frontier: tuple[RoutePreferenceSummary, ...] = ()
    elicited_constraints: tuple[ElicitedConstraint, ...] = ()
    compatible_set: CompatibleSet = field(default_factory=_empty_compatible_set)
    irrelevant_axes: tuple[ObjectiveName, ...] = ()
    certified_only_required: bool = False
    vetoed_targets: tuple[str, ...] = ()
    selected_route_id: str | None = None
    selected_certificate: float | None = None
    stop_reason: str | None = None
    stop_hints: tuple[PreferenceStopHint, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)

    def top_route_id(self) -> str | None:
        return self.compatible_set.top_route_id()

    def has_certified_compatible_route(self) -> bool:
        return bool(self.compatible_set.certified_route_ids)

    def wants_certified_only(self) -> bool:
        if self.certified_only_required:
            return True
        return any(
            constraint.kind == "route_veto" and constraint.target in {"uncertified", "certified_only"}
            for constraint in self.elicited_constraints
        )

    def has_time_guard(self) -> bool:
        return self.time_guard.is_active()

    def route_veto_targets(self) -> tuple[str, ...]:
        return self.vetoed_targets


def _int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _normalize_timestamp(raw: Any) -> str | None:
    parsed = _parse_iso_utc(raw)
    if parsed is None:
        return _as_text(raw)
    return parsed.isoformat().replace("+00:00", "Z")


def _merge_time_guards(*guards: TimeGuard) -> TimeGuard:
    active = [guard for guard in guards if guard.is_active()]
    if not active:
        return TimeGuard()
    latest_arrivals = [guard.latest_arrival_utc for guard in active if guard.latest_arrival_utc]
    return TimeGuard(
        max_duration_s=min((guard.max_duration_s for guard in active if guard.max_duration_s is not None), default=None),
        max_weather_delay_s=min(
            (guard.max_weather_delay_s for guard in active if guard.max_weather_delay_s is not None),
            default=None,
        ),
        max_incident_delay_s=min(
            (guard.max_incident_delay_s for guard in active if guard.max_incident_delay_s is not None),
            default=None,
        ),
        latest_arrival_utc=min(latest_arrivals) if latest_arrivals else None,
    )


def _constraint_to_time_guard(constraint: ElicitedConstraint) -> TimeGuard:
    if constraint.kind != "time_guard":
        return TimeGuard()
    value = _mapping_from_any(constraint.value)
    return TimeGuard(
        max_duration_s=_as_float(value.get("max_duration_s")) if value.get("max_duration_s") is not None else None,
        max_weather_delay_s=(
            _as_float(value.get("max_weather_delay_s")) if value.get("max_weather_delay_s") is not None else None
        ),
        max_incident_delay_s=(
            _as_float(value.get("max_incident_delay_s")) if value.get("max_incident_delay_s") is not None else None
        ),
        latest_arrival_utc=_normalize_timestamp(value.get("latest_arrival_utc")),
    )


def _time_guard_from_request(request: Mapping[str, Any]) -> TimeGuard:
    embedded = _mapping_from_any(request.get("time_guard") or request.get("preference_time_guard"))
    epsilon = _mapping_from_any(request.get("epsilon"))
    max_duration = None
    if embedded.get("max_duration_s") is not None:
        max_duration = _as_float(embedded.get("max_duration_s"))
    elif request.get("max_duration_s") is not None:
        max_duration = _as_float(request.get("max_duration_s"))
    elif epsilon.get("duration_s") is not None:
        max_duration = _as_float(epsilon.get("duration_s"))
    return TimeGuard(
        max_duration_s=max_duration,
        max_weather_delay_s=(
            _as_float(embedded.get("max_weather_delay_s"))
            if embedded.get("max_weather_delay_s") is not None
            else (
                _as_float(request.get("max_weather_delay_s"))
                if request.get("max_weather_delay_s") is not None
                else None
            )
        ),
        max_incident_delay_s=(
            _as_float(embedded.get("max_incident_delay_s"))
            if embedded.get("max_incident_delay_s") is not None
            else (
                _as_float(request.get("max_incident_delay_s"))
                if request.get("max_incident_delay_s") is not None
                else None
            )
        ),
        latest_arrival_utc=_normalize_timestamp(
            embedded.get("latest_arrival_utc") or request.get("latest_arrival_utc")
        ),
    )


def _route_matches_constraint(
    route: RoutePreferenceSummary,
    constraint: ElicitedConstraint,
    *,
    departure_time_utc: str | None,
) -> tuple[bool, tuple[str, ...], bool]:
    if constraint.kind == "objective_upper_bound":
        objective = constraint.target if constraint.target in OBJECTIVE_NAMES else "time"
        allowed = route.objective_value(objective) <= _as_float(constraint.value, float("inf"))
        return allowed, (() if allowed else (constraint.key(),)), False
    if constraint.kind == "objective_lower_bound":
        objective = constraint.target if constraint.target in OBJECTIVE_NAMES else "time"
        allowed = route.objective_value(objective) >= _as_float(constraint.value, 0.0)
        return allowed, (() if allowed else (constraint.key(),)), False
    if constraint.kind == "time_guard":
        allowed, reasons = _constraint_to_time_guard(constraint).evaluate(route, departure_time_utc=departure_time_utc)
        return allowed, reasons, False
    if constraint.kind == "toggle_preference":
        if constraint.target == "use_tolls" and not _as_bool(constraint.value, True) and route.uses_tolls:
            return False, (constraint.key(),), True
        return True, (), False
    if constraint.kind != "route_veto":
        return True, (), False

    target = constraint.target.strip().lower()
    blocked = False
    if target in {"uncertified", "certified_only"}:
        blocked = not route.certified
    elif target in {"tolls", "use_tolls"}:
        blocked = route.uses_tolls
    elif target in {"weather", "weather_delay"}:
        blocked = route.weather_delay_s > 0.0
    elif target in {"incidents", "incident_delay"}:
        blocked = route.incident_delay_s > 0.0
    else:
        route_key = route.route_id.strip().lower()
        blocked = target in {route_key, f"route:{route_key}"}
    return (not blocked), (() if not blocked else (constraint.key(),)), blocked


def score_routes_for_preference(
    routes: Sequence[RoutePreferenceSummary],
    *,
    weights: PreferenceWeights,
    optimization_mode: str = "expected_value",
    risk_aversion: float = 1.0,
) -> dict[str, float]:
    if not routes:
        return {}
    normalized_weights = weights.normalized()
    mins = {
        objective: min(route.objective_value(objective) for route in routes)
        for objective in OBJECTIVE_NAMES
    }
    spans = {
        objective: max(route.objective_value(objective) for route in routes) - mins[objective]
        for objective in OBJECTIVE_NAMES
    }
    scores: dict[str, float] = {}
    for route in routes:
        score = 0.0
        for objective, weight in normalized_weights.as_dict().items():
            span = spans[objective]
            normalized_value = 0.0 if span <= 0.0 else (route.objective_value(objective) - mins[objective]) / span
            score += float(weight) * normalized_value
        if optimization_mode == "robust":
            score += max(0.0, risk_aversion) * max(0.0, route.uncertainty_mass)
        scores[route.route_id] = score
    return scores


def rank_routes_for_preference(
    routes: Sequence[RoutePreferenceSummary],
    *,
    weights: PreferenceWeights,
    optimization_mode: str = "expected_value",
    risk_aversion: float = 1.0,
) -> tuple[str, ...]:
    scores = score_routes_for_preference(
        routes,
        weights=weights,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    ordered = sorted(routes, key=lambda route: (scores.get(route.route_id, 0.0), route.route_id))
    return tuple(route.route_id for route in ordered)


def compute_compatible_set(
    frontier: Sequence[RoutePreferenceSummary],
    *,
    elicited_constraints: Sequence[ElicitedConstraint] = (),
    departure_time_utc: str | None = None,
    time_guard: TimeGuard | None = None,
    weights: PreferenceWeights | None = None,
    optimization_mode: str = "expected_value",
    risk_aversion: float = 1.0,
) -> CompatibleSet:
    effective_time_guard = time_guard if time_guard is not None else TimeGuard()
    active_constraint_keys: list[str] = []
    if effective_time_guard.is_active():
        active_constraint_keys.append("time_guard")
    active_constraint_keys.extend(constraint.key() for constraint in elicited_constraints if constraint.strict)

    allowed_routes: list[RoutePreferenceSummary] = []
    vetoed_route_ids: list[str] = []
    blocked_reasons: dict[str, tuple[str, ...]] = {}

    for route in frontier:
        reasons: list[str] = []
        route_vetoed = False
        if effective_time_guard.is_active():
            allowed, guard_reasons = effective_time_guard.evaluate(route, departure_time_utc=departure_time_utc)
            if not allowed:
                reasons.extend(guard_reasons)
        for constraint in elicited_constraints:
            if not constraint.strict:
                continue
            allowed, constraint_reasons, blocked_by_veto = _route_matches_constraint(
                route,
                constraint,
                departure_time_utc=departure_time_utc,
            )
            if not allowed:
                reasons.extend(constraint_reasons or (constraint.key(),))
            route_vetoed = route_vetoed or blocked_by_veto
        if reasons:
            blocked_reasons[route.route_id] = tuple(dict.fromkeys(reasons))
            if route_vetoed:
                vetoed_route_ids.append(route.route_id)
            continue
        allowed_routes.append(route)

    ranked_route_ids = rank_routes_for_preference(
        allowed_routes,
        weights=weights or PreferenceWeights(),
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    route_by_id = {route.route_id: route for route in allowed_routes}
    certified_route_ids = tuple(
        route_id for route_id in ranked_route_ids if route_by_id.get(route_id) and route_by_id[route_id].certified
    )
    return CompatibleSet(
        route_ids=tuple(route.route_id for route in allowed_routes),
        ranked_route_ids=ranked_route_ids,
        certified_route_ids=certified_route_ids,
        vetoed_route_ids=tuple(dict.fromkeys(vetoed_route_ids)),
        blocked_reasons=blocked_reasons,
        active_constraint_keys=tuple(dict.fromkeys(active_constraint_keys)),
    )


def detect_irrelevant_preference_axes(frontier: Sequence[RoutePreferenceSummary]) -> tuple[ObjectiveName, ...]:
    if not frontier:
        return OBJECTIVE_NAMES
    irrelevant: list[ObjectiveName] = []
    for objective in OBJECTIVE_NAMES:
        values = [route.objective_value(objective) for route in frontier]
        span = max(values) - min(values)
        if span <= OBJECTIVE_IRRELEVANCE_FLOORS[objective]:
            irrelevant.append(objective)
    return tuple(irrelevant)


def build_preference_stop_hints(
    frontier: Sequence[RoutePreferenceSummary],
    compatible_set: CompatibleSet,
    *,
    elicited_constraints: Sequence[ElicitedConstraint] = (),
    selected_route_id: str | None = None,
    certificate_threshold: float | None = None,
    stop_reason: str | None = None,
    tau_stop: float | None = None,
) -> tuple[PreferenceStopHint, ...]:
    hints: list[PreferenceStopHint] = []
    certified_only = any(
        constraint.kind == "route_veto" and constraint.target in {"uncertified", "certified_only"}
        for constraint in elicited_constraints
    )
    frontier_ids = tuple(route.route_id for route in frontier)

    if not frontier:
        return (
            PreferenceStopHint(
                code="empty_frontier",
                message="No candidate routes are available for preference resolution.",
                severity="block",
            ),
        )

    if not compatible_set.route_ids:
        if certified_only:
            hints.append(
                PreferenceStopHint(
                    code="typed_abstention_recommended",
                    message="No certified route satisfies the current preference constraints; abstention is safer.",
                    severity="block",
                    route_ids=frontier_ids,
                    metadata={
                        "abstention_type": "uncertified_frontier",
                        "stop_reason": stop_reason,
                    },
                )
            )
        hints.append(
            PreferenceStopHint(
                code="preference_infeasible",
                message="No route satisfies the active preference constraints.",
                severity="block",
                route_ids=frontier_ids,
                metadata={"stop_reason": stop_reason},
            )
        )
        return tuple(hints)

    if certificate_threshold is not None and not compatible_set.certified_route_ids:
        hints.append(
            PreferenceStopHint(
                code="typed_abstention_recommended",
                message="No compatible route clears the current certification threshold.",
                severity="block" if certified_only else "warn",
                route_ids=compatible_set.ranked_route_ids,
                metadata={
                    "abstention_type": "certification_gap",
                    "certificate_threshold": certificate_threshold,
                    "stop_reason": stop_reason,
                },
            )
        )
    if selected_route_id and selected_route_id not in compatible_set.route_ids:
        hints.append(
            PreferenceStopHint(
                code="selected_outside_compatible_set",
                message="The currently selected route is outside the compatible preference set.",
                severity="warn",
                route_ids=(selected_route_id,),
                metadata={"stop_reason": stop_reason},
            )
        )
    if stop_reason in {"budget_exhausted", "search_incomplete_no_action_worth_it"}:
        hints.append(
            PreferenceStopHint(
                code="budget_limited_preference_resolution",
                message="Search or evidence budget stopped resolution before preferences fully separated the frontier.",
                severity="warn",
                route_ids=compatible_set.ranked_route_ids,
                metadata={"stop_reason": stop_reason},
            )
        )
    if tau_stop is not None and stop_reason in {"no_action_worth_it", "search_incomplete_no_action_worth_it"}:
        hints.append(
            PreferenceStopHint(
                code="tau_stop_reached",
                message="Preference resolution stopped because the expected value of more search fell below tau.",
                severity="info",
                route_ids=compatible_set.ranked_route_ids,
                metadata={"tau_stop": tau_stop, "stop_reason": stop_reason},
            )
        )
    return tuple(hints)


def build_preference_summary(
    *,
    weights: PreferenceWeights,
    optimization_mode: str,
    risk_aversion: float,
    use_tolls: bool,
    certified_only_required: bool,
    vetoed_targets: Sequence[str],
    time_guard_active: bool,
    compatible_set: CompatibleSet,
    irrelevant_axes: Sequence[ObjectiveName],
    selected_route_id: str | None,
    selected_certificate: float | None,
    stop_reason: str | None,
    stop_hints: Sequence[PreferenceStopHint],
) -> dict[str, Any]:
    return {
        "weights": weights.as_dict(),
        "dominant_objective": weights.dominant_objective(),
        "optimization_mode": optimization_mode,
        "risk_aversion": risk_aversion,
        "use_tolls": use_tolls,
        "certified_only_required": certified_only_required,
        "time_guard_active": time_guard_active,
        "vetoed_targets": list(vetoed_targets),
        "selected_route_id": selected_route_id,
        "selected_certificate": selected_certificate,
        "compatible_route_ids": list(compatible_set.route_ids),
        "ranked_route_ids": list(compatible_set.ranked_route_ids),
        "certified_route_ids": list(compatible_set.certified_route_ids),
        "vetoed_route_ids": list(compatible_set.vetoed_route_ids),
        "active_constraint_keys": list(compatible_set.active_constraint_keys),
        "irrelevant_axes": list(irrelevant_axes),
        "stop_reason": stop_reason,
        "stop_hint_codes": [hint.code for hint in stop_hints],
    }


def _request_constraints(request: Mapping[str, Any]) -> list[ElicitedConstraint]:
    constraints: list[ElicitedConstraint] = []
    cost_toggles = _mapping_from_any(request.get("cost_toggles"))
    if not _as_bool(cost_toggles.get("use_tolls"), True):
        constraints.append(
            ElicitedConstraint(
                kind="toggle_preference",
                target="use_tolls",
                value=False,
                label="toggle_use_tolls",
                source="request",
            )
        )
    epsilon = _mapping_from_any(request.get("epsilon"))
    if epsilon.get("duration_s") is not None:
        constraints.append(ElicitedConstraint.maximum("time", _as_float(epsilon.get("duration_s")), source="request"))
    if epsilon.get("monetary_cost") is not None:
        constraints.append(
            ElicitedConstraint.maximum("money", _as_float(epsilon.get("monetary_cost")), source="request")
        )
    if epsilon.get("emissions_kg") is not None:
        constraints.append(ElicitedConstraint.maximum("co2", _as_float(epsilon.get("emissions_kg")), source="request"))
    return constraints


def _dedupe_constraints(constraints: Sequence[ElicitedConstraint]) -> tuple[ElicitedConstraint, ...]:
    ordered: dict[str, ElicitedConstraint] = {}
    for constraint in constraints:
        ordered[constraint.key()] = constraint
    return tuple(ordered.values())


def build_preference_state(
    request: Mapping[str, Any] | Any,
    frontier: Sequence[RoutePreferenceSummary | Mapping[str, Any] | Any],
    *,
    elicited_constraints: Sequence[ElicitedConstraint] = (),
    selected_route_id: str | None = None,
    selected_certificate: float | None = None,
    stop_reason: str | None = None,
    certificate_map: Mapping[str, float] | None = None,
) -> PreferenceState:
    request_mapping = dict(_mapping_from_any(request))
    weights = PreferenceWeights.from_mapping(request_mapping.get("weights"))
    cost_toggles = _mapping_from_any(request_mapping.get("cost_toggles"))
    departure_time_utc = _normalize_timestamp(request_mapping.get("departure_time_utc"))
    certificate_threshold = (
        _as_float(request_mapping.get("certificate_threshold"))
        if request_mapping.get("certificate_threshold") is not None
        else None
    )
    frontier_routes = tuple(
        RoutePreferenceSummary.from_raw(
            route,
            fallback_threshold=certificate_threshold,
            certificate_map=certificate_map,
        )
        for route in frontier
    )
    all_constraints = _dedupe_constraints((*_request_constraints(request_mapping), *elicited_constraints))
    vetoed_targets = tuple(
        dict.fromkeys(
            _as_text(constraint.target).lower()
            for constraint in all_constraints
            if constraint.kind == "route_veto" and _as_text(constraint.target)
        )
    )
    certified_only_required = any(target in {"uncertified", "certified_only"} for target in vetoed_targets)
    request_guard = _time_guard_from_request(request_mapping)
    elicited_guards = tuple(_constraint_to_time_guard(constraint) for constraint in all_constraints)
    effective_time_guard = _merge_time_guards(request_guard, *elicited_guards)
    optimization_mode = _as_text(request_mapping.get("optimization_mode")) or "expected_value"
    risk_aversion = max(0.0, _as_float(request_mapping.get("risk_aversion"), 1.0))
    compatible_set = compute_compatible_set(
        frontier_routes,
        elicited_constraints=all_constraints,
        departure_time_utc=departure_time_utc,
        time_guard=effective_time_guard,
        weights=weights,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
    )
    irrelevant_axes = detect_irrelevant_preference_axes(frontier_routes)
    route_by_id = {route.route_id: route for route in frontier_routes}
    if selected_certificate is None and selected_route_id and selected_route_id in route_by_id:
        selected_certificate = route_by_id[selected_route_id].certificate
    stop_hints = build_preference_stop_hints(
        frontier_routes,
        compatible_set,
        elicited_constraints=all_constraints,
        selected_route_id=selected_route_id,
        certificate_threshold=certificate_threshold,
        stop_reason=stop_reason,
        tau_stop=_as_float(request_mapping.get("tau_stop")) if request_mapping.get("tau_stop") is not None else None,
    )
    summary = build_preference_summary(
        weights=weights,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
        use_tolls=_as_bool(cost_toggles.get("use_tolls"), True),
        certified_only_required=certified_only_required,
        vetoed_targets=vetoed_targets,
        time_guard_active=effective_time_guard.is_active(),
        compatible_set=compatible_set,
        irrelevant_axes=irrelevant_axes,
        selected_route_id=selected_route_id,
        selected_certificate=selected_certificate,
        stop_reason=stop_reason,
        stop_hints=stop_hints,
    )
    return PreferenceState(
        request_context=request_mapping,
        weights=weights,
        optimization_mode=optimization_mode,
        risk_aversion=risk_aversion,
        use_tolls=_as_bool(cost_toggles.get("use_tolls"), True),
        fuel_price_multiplier=_as_float(cost_toggles.get("fuel_price_multiplier"), 1.0),
        carbon_price_per_kg=_as_float(cost_toggles.get("carbon_price_per_kg"), 0.0),
        toll_cost_per_km=_as_float(cost_toggles.get("toll_cost_per_km"), 0.0),
        departure_time_utc=departure_time_utc,
        time_guard=effective_time_guard,
        search_budget=_int_or_none(request_mapping.get("search_budget")),
        evidence_budget=_int_or_none(request_mapping.get("evidence_budget")),
        certificate_threshold=certificate_threshold,
        tau_stop=_as_float(request_mapping.get("tau_stop")) if request_mapping.get("tau_stop") is not None else None,
        frontier=frontier_routes,
        elicited_constraints=all_constraints,
        compatible_set=compatible_set,
        irrelevant_axes=irrelevant_axes,
        certified_only_required=certified_only_required,
        vetoed_targets=vetoed_targets,
        selected_route_id=selected_route_id,
        selected_certificate=selected_certificate,
        stop_reason=stop_reason,
        stop_hints=stop_hints,
        summary=summary,
    )


__all__ = [
    "PreferenceState",
    "build_preference_state",
    "build_preference_stop_hints",
    "build_preference_summary",
    "compute_compatible_set",
    "detect_irrelevant_preference_axes",
    "rank_routes_for_preference",
    "score_routes_for_preference",
]
