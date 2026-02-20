from __future__ import annotations

import math
from collections.abc import Callable

from .models import EpsilonConstraints, ParetoMethod, RouteOption
from .pareto import pareto_filter


def _norm(v: float, vmin: float, vmax: float) -> float:
    return 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)


def filter_by_epsilon(
    options: list[RouteOption],
    epsilon: EpsilonConstraints | None,
    objective_key: Callable[[RouteOption], tuple[float, ...]] | None = None,
) -> list[RouteOption]:
    if epsilon is None:
        return options

    filtered: list[RouteOption] = []
    for route in options:
        # Epsilon bounds are always interpreted against raw physical objectives
        # (duration/money/co2), independent of any robust utility objective keying.
        duration_s = float(route.metrics.duration_s)
        monetary_cost = float(route.metrics.monetary_cost)
        emissions_kg = float(route.metrics.emissions_kg)
        if epsilon.duration_s is not None and duration_s > epsilon.duration_s:
            continue
        if epsilon.monetary_cost is not None and monetary_cost > epsilon.monetary_cost:
            continue
        if epsilon.emissions_kg is not None and emissions_kg > epsilon.emissions_kg:
            continue
        filtered.append(route)
    return filtered


def annotate_knee_scores(
    routes: list[RouteOption],
    objective_key: Callable[[RouteOption], tuple[float, ...]] | None = None,
) -> list[RouteOption]:
    if not routes:
        return []

    objective = objective_key or (
        lambda route: (
            float(route.metrics.duration_s),
            float(route.metrics.monetary_cost),
            float(route.metrics.emissions_kg),
        )
    )
    objective_values = [objective(route) for route in routes]
    dim_count = len(objective_values[0])
    mins = [min(vals[idx] for vals in objective_values) for idx in range(dim_count)]
    maxs = [max(vals[idx] for vals in objective_values) for idx in range(dim_count)]

    scores: list[tuple[float, RouteOption]] = []
    for route, values in zip(routes, objective_values, strict=True):
        score = math.sqrt(
            sum(
                (_norm(values[idx], mins[idx], maxs[idx]) ** 2)
                for idx in range(dim_count)
            )
        )
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
    objective_key: Callable[[RouteOption], tuple[float, ...]] | None = None,
    strict_frontier: bool = False,
) -> list[RouteOption]:
    objective = objective_key or (
        lambda route: (
            float(route.metrics.duration_s),
            float(route.metrics.monetary_cost),
            float(route.metrics.emissions_kg),
        )
    )
    considered = options
    if pareto_method == "epsilon_constraint":
        constrained = filter_by_epsilon(options, epsilon, objective_key=objective)
        considered = constrained

    if not considered:
        return []

    pareto = pareto_filter(considered, key=objective)
    pareto_sorted = sorted(pareto, key=lambda option: (*objective(option), option.id))
    if strict_frontier:
        return annotate_knee_scores(pareto_sorted, objective_key=objective)

    if pareto_sorted:
        return annotate_knee_scores(
            pareto_sorted[: max(1, max_alternatives)],
            objective_key=objective,
        )

    # Non-strict fallback only when strict frontier is disabled and no pareto
    # candidates survived (e.g. corrupted objective payloads).
    non_strict_sorted = sorted(considered, key=lambda option: (*objective(option), option.id))
    return annotate_knee_scores(non_strict_sorted[:max(1, max_alternatives)], objective_key=objective)
