from __future__ import annotations

import math
from collections.abc import Iterable
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .calibration_loader import load_risk_normalization_reference
from .certification_models import AuditWorldBundle, ProbabilisticWorldBundle, WorldSupportState


def normalized_objective_components(
    *,
    duration_s: float,
    monetary_cost: float,
    emissions_kg: float,
    distance_km: float | None,
    vehicle_type: str | None = None,
    vehicle_bucket: str | None = None,
    corridor_bucket: str | None = None,
    day_kind: str | None = None,
    local_time_slot: str | None = None,
) -> tuple[float, float, float]:
    """Return dimensionless objective components for cross-route utility comparisons."""
    dist = max(1.0, float(distance_km or 0.0))
    refs = load_risk_normalization_reference(
        vehicle_type=vehicle_type,
        vehicle_bucket=vehicle_bucket,
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
    vehicle_bucket: str | None = None,
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
        vehicle_bucket=vehicle_bucket,
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
    # Use the Hyndman-Fan type-7 sample quantile convention:
    # Hyndman & Fan (1996) https://doi.org/10.1080/00031305.1996.10473566
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

    # Empirical CVaR / expected shortfall integrates the tail quantile
    # function on [alpha, 1], following Rockafellar & Uryasev (2000):
    # https://doi.org/10.21314/JOR.2000.038
    knots: list[float] = [alpha_clamped]
    for idx in range(n):
        p = idx / (n - 1)
        if alpha_clamped < p < 1.0:
            knots.append(p)
    knots.append(1.0)
    knots = sorted(set(knots))

    tail_integral = 0.0
    for start, end in zip(knots, knots[1:], strict=False):
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
    risk_family: str = "cvar_excess",
    risk_theta: float = 1.0,
) -> float:
    if cvar_value is None:
        return float(mean_value)
    tail_excess = max(0.0, float(cvar_value) - float(mean_value))
    risk = max(0.0, float(risk_aversion))
    family = str(risk_family or "cvar_excess").strip().lower()
    if family == "entropic":
        theta = max(1e-6, float(risk_theta))
        # This is a thesis-specific entropic-style tail penalty, not the
        # canonical entropic risk measure on the full loss distribution.
        penalty = math.log1p(risk * math.expm1(theta * tail_excess)) / theta
        return float(mean_value) + penalty
    if family == "downside_semivariance":
        scale = max(1.0, abs(float(mean_value)))
        # This is a thesis-specific quadratic penalty on CVaR excess, not the
        # textbook downside semivariance definition over centered tail losses.
        penalty = (tail_excess * tail_excess) / scale
        return float(mean_value) + (risk * penalty)
    return float(mean_value) + (risk * tail_excess)


@dataclass(frozen=True)
class RiskSummary:
    mean_value: float = 0.0
    cvar_value: float | None = None
    robust_score: float = 0.0
    normalized_duration_component: float = 0.0
    normalized_monetary_component: float = 0.0
    normalized_emissions_component: float = 0.0
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    risk_aversion: float = 1.0
    risk_family: str = "cvar_excess"
    risk_theta: float = 1.0
    support_state: WorldSupportState | None = None
    probabilistic_world_bundle: ProbabilisticWorldBundle | None = None
    audit_world_bundle: AuditWorldBundle | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)


@dataclass(frozen=True)
class FragilitySummary:
    route_id: str | None = None
    deterministic_local_flip_radius: float = 0.0
    probabilistic_flip_radius: float = 0.0
    challenger_specific_radii: dict[str, float] = field(default_factory=dict)
    evidence_family_radii: dict[str, float] = field(default_factory=dict)
    dominant_fragility_family: str | None = None
    support_flag: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)


def build_risk_summary(
    *,
    duration_s: float,
    monetary_cost: float,
    emissions_kg: float,
    distance_km: float | None,
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    risk_aversion: float = 1.0,
    risk_family: str | None = None,
    risk_theta: float | None = None,
    cvar_value: float | None = None,
    support_state: WorldSupportState | None = None,
    probabilistic_world_bundle: ProbabilisticWorldBundle | None = None,
    audit_world_bundle: AuditWorldBundle | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> RiskSummary:
    normalized_duration_component, normalized_monetary_component, normalized_emissions_component = (
        normalized_objective_components(
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            distance_km=distance_km,
        )
    )
    weighted_utility = normalized_weighted_utility(
        duration_s=duration_s,
        monetary_cost=monetary_cost,
        emissions_kg=emissions_kg,
        distance_km=distance_km,
        utility_weights=utility_weights,
    )
    resolved_family = str(risk_family or settings.risk_family)
    resolved_theta = float(risk_theta if risk_theta is not None else settings.risk_family_theta)
    return RiskSummary(
        mean_value=weighted_utility,
        cvar_value=cvar_value,
        robust_score=robust_objective(
            mean_value=weighted_utility,
            cvar_value=cvar_value,
            risk_aversion=risk_aversion,
            risk_family=resolved_family,
            risk_theta=resolved_theta,
        ),
        normalized_duration_component=normalized_duration_component,
        normalized_monetary_component=normalized_monetary_component,
        normalized_emissions_component=normalized_emissions_component,
        utility_weights=utility_weights,
        risk_aversion=float(risk_aversion),
        risk_family=resolved_family,
        risk_theta=resolved_theta,
        support_state=support_state,
        probabilistic_world_bundle=probabilistic_world_bundle,
        audit_world_bundle=audit_world_bundle,
        provenance=dict(provenance or {}),
    )


def build_fragility_summary(
    *,
    route_id: str | None = None,
    deterministic_local_flip_radius: float = 0.0,
    probabilistic_flip_radius: float = 0.0,
    challenger_specific_radii: Mapping[str, float] | None = None,
    evidence_family_radii: Mapping[str, float] | None = None,
    dominant_fragility_family: str | None = None,
    support_flag: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> FragilitySummary:
    return FragilitySummary(
        route_id=route_id,
        deterministic_local_flip_radius=float(deterministic_local_flip_radius),
        probabilistic_flip_radius=float(probabilistic_flip_radius),
        challenger_specific_radii={str(key): float(value) for key, value in dict(challenger_specific_radii or {}).items()},
        evidence_family_radii={str(key): float(value) for key, value in dict(evidence_family_radii or {}).items()},
        dominant_fragility_family=dominant_fragility_family,
        support_flag=bool(support_flag),
        provenance=dict(provenance or {}),
    )
