from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _objective_vector(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    value = candidate.get("proxy_objective")
    if isinstance(value, Mapping):
        return (
            _as_float(value.get("time")),
            _as_float(value.get("money")),
            _as_float(value.get("co2")),
        )
    if isinstance(value, Sequence) and len(value) >= 3:
        return (_as_float(value[0]), _as_float(value[1]), _as_float(value[2]))
    metrics = candidate.get("metrics")
    if isinstance(metrics, Mapping):
        return (
            _as_float(metrics.get("duration_s")),
            _as_float(metrics.get("monetary_cost")),
            _as_float(metrics.get("emissions_kg")),
        )
    return (
        _as_float(candidate.get("time")),
        _as_float(candidate.get("money")),
        _as_float(candidate.get("co2")),
    )


def _support_mass(candidate: Mapping[str, Any]) -> float:
    confidence = candidate.get("proxy_confidence", candidate.get("confidence", {}))
    if isinstance(confidence, Mapping) and confidence:
        values = [max(0.0, min(1.0, _as_float(value, 0.0))) for value in confidence.values()]
        return max(0.0, min(1.0, sum(values) / float(len(values))))
    support_ratio = candidate.get("support_mass", candidate.get("support_ratio", None))
    if support_ratio is not None:
        return max(0.0, min(1.0, _as_float(support_ratio)))
    return 0.0


def _dominance_margin(candidate: tuple[float, float, float], dominator: tuple[float, float, float]) -> float:
    return max(0.0, sum(candidate[idx] - dominator[idx] for idx in range(3)) / 3.0)


@dataclass(frozen=True)
class CandidateEnvelope:
    lower_objective: tuple[float, float, float]
    upper_objective: tuple[float, float, float]
    provenance: str
    support_mass: float
    support_status: str
    known_dominance: str | None = None
    safe_eliminated: bool = False
    necessary_dominated: bool = False
    dominated_by_route_id: str | None = None
    dominance_margin: float | None = None
    safe_elimination_reason: str | None = None


def build_candidate_envelope(
    candidate: Mapping[str, Any],
    *,
    frontier: Sequence[Mapping[str, Any]] = (),
) -> CandidateEnvelope:
    objective = _objective_vector(candidate)
    support_mass = _support_mass(candidate)
    slack = 0.03 + (0.12 * (1.0 - support_mass))
    lower_objective = tuple(max(0.0, value * (1.0 - slack)) for value in objective)
    upper_objective = tuple(value * (1.0 + slack) for value in objective)
    support_status = "supported" if support_mass >= 0.50 else ("weak_support" if support_mass > 0.0 else "unknown")

    dominated_by_route_id: str | None = None
    dominance_margin: float | None = None
    known_dominance: str | None = None
    safe_eliminated = False
    necessary_dominated = False
    safe_elimination_reason: str | None = None

    for item in frontier:
        route_id = str(item.get("candidate_id") or item.get("route_id") or "").strip() or None
        frontier_objective = _objective_vector(item)
        dominates_candidate = all(frontier_objective[idx] <= objective[idx] for idx in range(3)) and any(
            frontier_objective[idx] < objective[idx] for idx in range(3)
        )
        if dominates_candidate:
            dominated_by_route_id = route_id
            dominance_margin = _dominance_margin(objective, frontier_objective)
            known_dominance = "dominated_by_frontier"
            safe_eliminated = True
            necessary_dominated = True
            safe_elimination_reason = "objective_dominated_by_frontier_envelope"
            break

        candidate_dominates_frontier = all(objective[idx] <= frontier_objective[idx] for idx in range(3)) and any(
            objective[idx] < frontier_objective[idx] for idx in range(3)
        )
        if candidate_dominates_frontier and known_dominance is None:
            known_dominance = "dominates_frontier"

    return CandidateEnvelope(
        lower_objective=lower_objective,
        upper_objective=upper_objective,
        provenance="proxy_objective_envelope_v1",
        support_mass=support_mass,
        support_status=support_status,
        known_dominance=known_dominance,
        safe_eliminated=safe_eliminated,
        necessary_dominated=necessary_dominated,
        dominated_by_route_id=dominated_by_route_id,
        dominance_margin=dominance_margin,
        safe_elimination_reason=safe_elimination_reason,
    )
