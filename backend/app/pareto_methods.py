from __future__ import annotations

import math

from .models import EpsilonConstraints, ParetoMethod, RouteOption
from .pareto import pareto_filter


def _norm(v: float, vmin: float, vmax: float) -> float:
    return 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)


def filter_by_epsilon(
    options: list[RouteOption],
    epsilon: EpsilonConstraints | None,
) -> list[RouteOption]:
    if epsilon is None:
        return options

    filtered: list[RouteOption] = []
    for route in options:
        m = route.metrics
        if epsilon.duration_s is not None and m.duration_s > epsilon.duration_s:
            continue
        if epsilon.monetary_cost is not None and m.monetary_cost > epsilon.monetary_cost:
            continue
        if epsilon.emissions_kg is not None and m.emissions_kg > epsilon.emissions_kg:
            continue
        filtered.append(route)
    return filtered


def _fallback_diverse_subset(options: list[RouteOption], max_alternatives: int) -> list[RouteOption]:
    desired = min(len(options), max(2, min(max_alternatives, 4)))
    ranked_time = sorted(options, key=lambda o: o.metrics.duration_s)
    ranked_cost = sorted(options, key=lambda o: o.metrics.monetary_cost)
    ranked_co2 = sorted(options, key=lambda o: o.metrics.emissions_kg)

    chosen: dict[str, RouteOption] = {}

    def pick_first(ranked: list[RouteOption]) -> None:
        for route in ranked:
            if route.id not in chosen:
                chosen[route.id] = route
                return

    for ranked in (ranked_time, ranked_cost, ranked_co2):
        pick_first(ranked)

    for route in ranked_time:
        if len(chosen) >= desired:
            break
        if route.id not in chosen:
            chosen[route.id] = route

    return sorted(chosen.values(), key=lambda route: (route.metrics.duration_s, route.id))


def annotate_knee_scores(routes: list[RouteOption]) -> list[RouteOption]:
    if not routes:
        return []

    durations = [r.metrics.duration_s for r in routes]
    costs = [r.metrics.monetary_cost for r in routes]
    co2s = [r.metrics.emissions_kg for r in routes]

    dmin, dmax = min(durations), max(durations)
    cmin, cmax = min(costs), max(costs)
    emin, emax = min(co2s), max(co2s)

    scores: list[tuple[float, RouteOption]] = []
    for route in routes:
        d = _norm(route.metrics.duration_s, dmin, dmax)
        c = _norm(route.metrics.monetary_cost, cmin, cmax)
        e = _norm(route.metrics.emissions_kg, emin, emax)
        score = math.sqrt((d * d) + (c * c) + (e * e))
        scores.append((score, route))

    knee_route_id = min(
        scores,
        key=lambda item: (item[0], item[1].metrics.duration_s, item[1].id),
    )[1].id

    out: list[RouteOption] = []
    for score, route in scores:
        out.append(
            route.model_copy(
                update={
                    "knee_score": round(score, 6),
                    "is_knee": route.id == knee_route_id,
                }
            )
        )
    return out


def select_pareto_routes(
    options: list[RouteOption],
    *,
    max_alternatives: int,
    pareto_method: ParetoMethod,
    epsilon: EpsilonConstraints | None,
) -> list[RouteOption]:
    considered = options
    if pareto_method == "epsilon_constraint":
        constrained = filter_by_epsilon(options, epsilon)
        if constrained:
            considered = constrained

    pareto = pareto_filter(
        considered,
        key=lambda o: (o.metrics.duration_s, o.metrics.monetary_cost, o.metrics.emissions_kg),
    )
    pareto_sorted = sorted(pareto, key=lambda o: (o.metrics.duration_s, o.id))

    if len(pareto_sorted) < 2 and len(considered) > 1:
        pareto_sorted = _fallback_diverse_subset(considered, max_alternatives=max_alternatives)

    return annotate_knee_scores(pareto_sorted)

