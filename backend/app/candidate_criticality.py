from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from .candidate_bounds import CandidateEnvelope

CANDIDATE_CRITICALITY_SCHEMA_VERSION = "dccs-criticality-v1"


def _confidence_mean(
    proxy_confidence: Mapping[str, float] | None,
    *,
    candidate_envelope: CandidateEnvelope | None,
) -> float:
    if candidate_envelope is not None:
        return float(candidate_envelope.confidence_mean)
    values = [float(value) for value in (proxy_confidence or {}).values()]
    if not values:
        return 0.5
    return max(0.0, min(1.0, sum(values) / float(len(values))))


def classify_criticality_band(score: float, *, decision_critical: bool) -> str:
    if score >= 0.50:
        return "high"
    if decision_critical or score >= 0.20:
        return "medium"
    return "low"


@dataclass(frozen=True)
class CandidateCriticalityEstimate:
    schema_version: str
    criticality_score: float
    value_density: float
    criticality_band: str
    decision_critical: bool
    objective_gap: float
    mechanism_gap: float
    flip_probability: float
    novelty_signal: float
    time_preservation_bonus: float
    confidence_mean: float
    predicted_refine_cost: float
    dominance_margin: float = 0.0
    safe_elimination_reason: str | None = None
    search_deficiency_score: float = 0.0
    hidden_challenger_score: float = 0.0
    anti_collapse_pressure: float = 0.0
    long_corridor_search_completeness: float = 1.0
    observed_refine_cost: float | None = None
    refine_cost_error: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_candidate_criticality_estimate(
    *,
    objective_gap: float,
    mechanism_gap: float,
    flip_probability: float,
    overlap: float,
    stretch: float,
    time_preservation_bonus: float,
    predicted_refine_cost: float,
    dominance_margin: float = 0.0,
    safe_elimination_reason: str | None = None,
    search_deficiency_score: float = 0.0,
    hidden_challenger_score: float = 0.0,
    anti_collapse_pressure: float = 0.0,
    long_corridor_search_completeness: float = 1.0,
    proxy_confidence: Mapping[str, float] | None = None,
    candidate_envelope: CandidateEnvelope | None = None,
    observed_refine_cost: float | None = None,
    refine_cost_error: float | None = None,
) -> CandidateCriticalityEstimate:
    novelty_signal = max(0.0, 1.0 - float(overlap))
    stretch_penalty = max(0.0, float(stretch) - 1.0)
    dominance_signal = max(0.0, min(1.0, float(dominance_margin) / 0.12))
    time_competitiveness = max(
        0.0,
        min(
            1.0,
            (0.70 * max(0.0, float(time_preservation_bonus)))
            + (0.30 * dominance_signal),
        ),
    )
    slow_tradeoff_penalty = max(
        0.0,
        min(
            1.0,
            max(0.0, (0.60 - time_competitiveness) / 0.60)
            * (0.45 + (0.55 * max(0.0, 1.0 - dominance_signal))),
        ),
    )
    long_corridor_gap = max(0.0, min(1.0, 1.0 - float(long_corridor_search_completeness)))
    challenger_pressure = max(
        0.0,
        min(
            1.0,
            (0.34 * max(0.0, float(hidden_challenger_score)))
            + (0.28 * max(0.0, float(search_deficiency_score)))
            + (0.22 * max(0.0, float(anti_collapse_pressure)))
            + (0.16 * long_corridor_gap),
        ),
    )
    confidence_mean = _confidence_mean(
        proxy_confidence,
        candidate_envelope=candidate_envelope,
    )
    benefit = (
        max(0.0, float(objective_gap))
        + (0.75 * max(0.0, float(mechanism_gap)))
        + (0.80 * max(0.0, float(flip_probability)))
        + (0.35 * novelty_signal)
        + (0.45 * max(0.0, float(time_preservation_bonus)))
        + (0.20 * confidence_mean)
        + (0.20 * time_competitiveness)
        + (0.35 * dominance_signal)
        + (0.25 * challenger_pressure)
    )
    penalty = 1.0 + max(0.0, float(predicted_refine_cost)) + stretch_penalty + (0.55 * slow_tradeoff_penalty)
    value_density = benefit / max(1e-9, penalty)
    criticality_score = 1.0 - math.exp(-value_density)
    decision_critical = (
        float(objective_gap) > 1e-9
        or float(flip_probability) >= 0.5
        or dominance_signal >= 0.30
        or challenger_pressure >= 0.45
    )
    return CandidateCriticalityEstimate(
        schema_version=CANDIDATE_CRITICALITY_SCHEMA_VERSION,
        criticality_score=float(max(0.0, min(1.0, criticality_score))),
        value_density=float(max(0.0, value_density)),
        criticality_band=classify_criticality_band(
            criticality_score,
            decision_critical=decision_critical,
        ),
        decision_critical=decision_critical,
        objective_gap=float(objective_gap),
        mechanism_gap=float(mechanism_gap),
        flip_probability=float(flip_probability),
        novelty_signal=float(novelty_signal),
        time_preservation_bonus=float(max(0.0, time_preservation_bonus)),
        confidence_mean=float(confidence_mean),
        predicted_refine_cost=float(max(0.0, predicted_refine_cost)),
        dominance_margin=float(max(0.0, dominance_margin)),
        safe_elimination_reason=(
            str(safe_elimination_reason).strip() or None
            if safe_elimination_reason is not None
            else None
        ),
        search_deficiency_score=float(max(0.0, search_deficiency_score)),
        hidden_challenger_score=float(max(0.0, hidden_challenger_score)),
        anti_collapse_pressure=float(max(0.0, min(1.0, anti_collapse_pressure))),
        long_corridor_search_completeness=float(max(0.0, min(1.0, long_corridor_search_completeness))),
        observed_refine_cost=None if observed_refine_cost is None else float(observed_refine_cost),
        refine_cost_error=None if refine_cost_error is None else float(refine_cost_error),
    )
