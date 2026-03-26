from __future__ import annotations

import math
from collections.abc import Callable

from .models import EpsilonConstraints, ParetoMethod, RouteOption
from .pareto import pareto_filter


def _norm(v: float, vmin: float, vmax: float) -> float:
    return 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)


def _crowding_distances(values: list[tuple[float, ...]]) -> list[float]:
    count = len(values)
    if count <= 2:
        return [float("inf")] * count
    dim_count = len(values[0])
    distances = [0.0] * count
    for dim in range(dim_count):
        order = sorted(range(count), key=lambda idx: values[idx][dim])
        first = order[0]
        last = order[-1]
        min_v = float(values[first][dim])
        max_v = float(values[last][dim])
        distances[first] = float("inf")
        distances[last] = float("inf")
        if max_v <= min_v:
            continue
        denom = max_v - min_v
        for rank in range(1, count - 1):
            idx = order[rank]
            if math.isinf(distances[idx]):
                continue
            prev_v = float(values[order[rank - 1]][dim])
            next_v = float(values[order[rank + 1]][dim])
            distances[idx] += (next_v - prev_v) / denom
    return distances


def _truncate_pareto_with_crowding(
    routes: list[RouteOption],
    *,
    max_alternatives: int,
    objective_key: Callable[[RouteOption], tuple[float, ...]],
) -> list[RouteOption]:
    if len(routes) <= max_alternatives:
        return list(routes)
    objectives = [objective_key(route) for route in routes]
    crowding = _crowding_distances(objectives)
    ranked_indices = sorted(
        range(len(routes)),
        key=lambda idx: (
            0 if math.isinf(crowding[idx]) else 1,
            -(crowding[idx] if not math.isinf(crowding[idx]) else float("inf")),
            objectives[idx],
            routes[idx].id,
        ),
    )
    selected_indices = ranked_indices[:max_alternatives]
    selected = [routes[idx] for idx in selected_indices]
    return sorted(selected, key=lambda option: (*objective_key(option), option.id))


def filter_by_epsilon(
    options: list[RouteOption],
    epsilon: EpsilonConstraints | None,
    objective_key: Callable[[RouteOption], tuple[float, ...]] | None = None,
) -> list[RouteOption]:
    if epsilon is None:
        return options

    filtered: list[RouteOption] = []
    for route in options:
        # Classical epsilon-constraint filtering treats some objectives as hard
        # upper bounds while optimizing within the feasible subset:
        # Haimes, Lasdon, Wismer (1971) https://doi.org/10.1109/TSSC.1971.233258
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
        # This is a closest-to-ideal heuristic in normalized objective space,
        # not a canonical knee-point detector from the multi-objective
        # optimization literature. See the distinction discussed by
        # Branke et al. (2004): https://publikationen.bibliothek.kit.edu/1000018439
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
        capped = _truncate_pareto_with_crowding(
            pareto_sorted,
            max_alternatives=max(1, max_alternatives),
            objective_key=objective,
        )
        # NSGA-II crowding-distance truncation keeps objective-space spread on
        # the same Pareto frontier before selecting the exported subset:
        # Deb et al. (2002) https://doi.org/10.1109/4235.996017
        return annotate_knee_scores(capped, objective_key=objective)

    # Non-strict fallback only when strict frontier is disabled and no pareto
    # candidates survived (e.g. corrupted objective payloads).
    non_strict_sorted = sorted(considered, key=lambda option: (*objective(option), option.id))
    return annotate_knee_scores(non_strict_sorted[:max(1, max_alternatives)], objective_key=objective)
