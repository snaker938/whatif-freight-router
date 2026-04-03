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
    proxy_confidence: Mapping[str, float] | None = None,
    candidate_envelope: CandidateEnvelope | None = None,
    observed_refine_cost: float | None = None,
    refine_cost_error: float | None = None,
) -> CandidateCriticalityEstimate:
    novelty_signal = max(0.0, 1.0 - float(overlap))
    stretch_penalty = max(0.0, float(stretch) - 1.0)
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
    )
    penalty = 1.0 + max(0.0, float(predicted_refine_cost)) + stretch_penalty
    value_density = benefit / max(1e-9, penalty)
    criticality_score = 1.0 - math.exp(-value_density)
    decision_critical = (
        float(objective_gap) > 1e-9 or float(flip_probability) >= 0.5
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
        observed_refine_cost=None if observed_refine_cost is None else float(observed_refine_cost),
        refine_cost_error=None if refine_cost_error is None else float(refine_cost_error),
    )
