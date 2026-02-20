from __future__ import annotations

import math
from typing import Iterable

from .calibration_loader import load_risk_normalization_reference


def normalized_objective_components(
    *,
    duration_s: float,
    monetary_cost: float,
    emissions_kg: float,
    distance_km: float | None,
    vehicle_type: str | None = None,
    corridor_bucket: str | None = None,
    day_kind: str | None = None,
    local_time_slot: str | None = None,
) -> tuple[float, float, float]:
    """Return dimensionless objective components for cross-route utility comparisons."""
    dist = max(1.0, float(distance_km or 0.0))
    refs = load_risk_normalization_reference(
        vehicle_type=vehicle_type,
        corridor_bucket=corridor_bucket,
        day_kind=day_kind,
        local_time_slot=local_time_slot,
    )
    duration_ref = dist * refs.duration_s_per_km
    money_ref = dist * refs.monetary_gbp_per_km
    emissions_ref = dist * refs.emissions_kg_per_km
    return (
        max(0.0, float(duration_s)) / max(duration_ref, 1e-9),
        max(0.0, float(monetary_cost)) / max(money_ref, 1e-9),
        max(0.0, float(emissions_kg)) / max(emissions_ref, 1e-9),
    )


def normalized_weighted_utility(
    *,
    duration_s: float,
    monetary_cost: float,
    emissions_kg: float,
    distance_km: float | None,
    utility_weights: tuple[float, float, float],
    vehicle_type: str | None = None,
    corridor_bucket: str | None = None,
    day_kind: str | None = None,
    local_time_slot: str | None = None,
) -> float:
    w_time, w_money, w_emissions = utility_weights
    w_sum = max(float(w_time) + float(w_money) + float(w_emissions), 1e-9)
    wt = max(0.0, float(w_time)) / w_sum
    wm = max(0.0, float(w_money)) / w_sum
    we = max(0.0, float(w_emissions)) / w_sum
    d_norm, m_norm, e_norm = normalized_objective_components(
        duration_s=duration_s,
        monetary_cost=monetary_cost,
        emissions_kg=emissions_kg,
        distance_km=distance_km,
        vehicle_type=vehicle_type,
        corridor_bucket=corridor_bucket,
        day_kind=day_kind,
        local_time_slot=local_time_slot,
    )
    return (wt * d_norm) + (wm * m_norm) + (we * e_norm)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    q_clamped = min(1.0, max(0.0, float(q)))
    if len(ordered) == 1:
        return ordered[0]
    # Linear interpolation quantile for smoother and less biased tails.
    pos = q_clamped * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    t = pos - lo
    return ordered[lo] + ((ordered[hi] - ordered[lo]) * t)


def cvar(values: Iterable[float], *, alpha: float = 0.95) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    n = len(ordered)
    if n == 1:
        return ordered[0]

    alpha_clamped = min(1.0, max(0.0, float(alpha)))
    if alpha_clamped >= 1.0:
        return ordered[-1]

    # Integrate the interpolated quantile function on [alpha, 1].
    knots: list[float] = [alpha_clamped]
    for idx in range(n):
        p = idx / (n - 1)
        if alpha_clamped < p < 1.0:
            knots.append(p)
    knots.append(1.0)
    knots = sorted(set(knots))

    tail_integral = 0.0
    for start, end in zip(knots, knots[1:]):
        q_start = quantile(ordered, start)
        q_end = quantile(ordered, end)
        tail_integral += 0.5 * (q_start + q_end) * (end - start)

    denom = max(1e-12, 1.0 - alpha_clamped)
    result = tail_integral / denom
    threshold = quantile(ordered, alpha_clamped)
    return max(result, threshold)


def robust_objective(
    *,
    mean_value: float,
    cvar_value: float | None,
    risk_aversion: float,
) -> float:
    if cvar_value is None:
        return float(mean_value)
    tail_excess = max(0.0, float(cvar_value) - float(mean_value))
    return float(mean_value) + (max(0.0, float(risk_aversion)) * tail_excess)
