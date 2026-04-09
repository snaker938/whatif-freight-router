from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .candidate_bounds import CandidateEnvelope


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class CandidateCriticalityEstimate:
    winner_lcb_lift: float
    pairwise_gap_lcb_lift: float
    flip_radius_lift: float
    unresolved_winner_mass: float
    preference_relevance: float
    search_deficiency_risk: float
    action_cost: float
    criticality_score: float
    provenance: str


def build_candidate_criticality(
    candidate: Mapping[str, Any],
    *,
    objective_gap: float,
    mechanism_gap: float,
    overlap: float,
    stretch: float,
    time_regret_gap: float,
    predicted_refine_cost: float,
    flip_probability: float,
    candidate_envelope: CandidateEnvelope | None = None,
    near_duplicate: bool = False,
) -> CandidateCriticalityEstimate:
    support_mass = candidate_envelope.support_mass if candidate_envelope is not None else 0.0
    winner_lcb_lift = max(0.0, 0.45 * objective_gap + 0.15 * (1.0 - support_mass))
    pairwise_gap_lcb_lift = max(0.0, 0.35 * mechanism_gap + 0.10 * (1.0 - overlap))
    flip_radius_lift = max(0.0, 0.50 * flip_probability + 0.05 * max(0.0, stretch - 1.0))
    unresolved_winner_mass = max(0.0, 1.0 - objective_gap)
    preference_relevance = _clamp_unit(
        0.40 * mechanism_gap + 0.30 * time_regret_gap + 0.20 * (1.0 - overlap)
    )
    search_deficiency_risk = _clamp_unit(
        0.45 * (1.0 - objective_gap) + 0.25 * overlap + 0.30 * float(near_duplicate)
    )
    action_cost = max(0.0, _as_float(predicted_refine_cost))
    denominator = max(1.0, action_cost)
    criticality_score = (
        winner_lcb_lift
        + pairwise_gap_lcb_lift
        + flip_radius_lift
        + unresolved_winner_mass
        + preference_relevance
        + search_deficiency_risk
    ) / denominator
    return CandidateCriticalityEstimate(
        winner_lcb_lift=winner_lcb_lift,
        pairwise_gap_lcb_lift=pairwise_gap_lcb_lift,
        flip_radius_lift=flip_radius_lift,
        unresolved_winner_mass=unresolved_winner_mass,
        preference_relevance=preference_relevance,
        search_deficiency_risk=search_deficiency_risk,
        action_cost=action_cost,
        criticality_score=criticality_score,
        provenance="candidate_criticality_v1",
    )
