from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

DEFAULT_OBJECTIVE_NAMES: tuple[str, str, str] = ("time", "money", "co2")
CANDIDATE_ENVELOPE_SCHEMA_VERSION = "dccs-envelope-v1"


def _clamp_confidence(value: float | int | None) -> float:
    if value is None:
        return 0.5
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, numeric))


def _confidence_by_objective(
    proxy_confidence: Mapping[str, float] | None,
    *,
    objective_names: Sequence[str],
) -> dict[str, float]:
    raw = dict(proxy_confidence or {})
    if not raw:
        return {name: 0.5 for name in objective_names}
    resolved = {
        name: _clamp_confidence(raw.get(name))
        for name in objective_names
    }
    if all(value <= 0.0 for value in resolved.values()):
        return {name: 0.5 for name in objective_names}
    return resolved


@dataclass(frozen=True)
class CandidateEnvelopeBounds:
    lower: float
    upper: float
    midpoint: float
    width: float
    confidence: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEnvelope:
    schema_version: str
    objective_bounds: dict[str, CandidateEnvelopeBounds]
    refine_cost_bounds: CandidateEnvelopeBounds
    overlap_bounds: CandidateEnvelopeBounds
    stretch_bounds: CandidateEnvelopeBounds
    confidence_floor: float
    confidence_mean: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def bounds_from_confidence(
    center: float,
    *,
    confidence: float,
    minimum_width: float,
    relative_width: float,
    nonnegative: bool = False,
    upper_cap: float | None = None,
) -> CandidateEnvelopeBounds:
    conf = _clamp_confidence(confidence)
    radius = max(
        float(minimum_width),
        abs(float(center)) * float(relative_width) * (0.25 + (1.0 - conf)),
    )
    lower = float(center) - radius
    upper = float(center) + radius
    if nonnegative:
        lower = max(0.0, lower)
    if upper_cap is not None:
        lower = min(float(upper_cap), lower)
        upper = min(float(upper_cap), upper)
    midpoint = (lower + upper) / 2.0
    width = max(0.0, upper - lower)
    return CandidateEnvelopeBounds(
        lower=float(lower),
        upper=float(upper),
        midpoint=float(midpoint),
        width=float(width),
        confidence=conf,
    )


def build_candidate_envelope(
    *,
    proxy_objective: Sequence[float],
    proxy_confidence: Mapping[str, float] | None,
    predicted_refine_cost: float,
    overlap: float,
    stretch: float,
    objective_names: Sequence[str] = DEFAULT_OBJECTIVE_NAMES,
) -> CandidateEnvelope:
    confidences = _confidence_by_objective(proxy_confidence, objective_names=objective_names)
    objective_values = list(proxy_objective)
    while len(objective_values) < len(objective_names):
        objective_values.append(0.0)
    objective_bounds = {
        name: bounds_from_confidence(
            float(objective_values[index]),
            confidence=confidences[name],
            minimum_width=max(0.05, abs(float(objective_values[index])) * 0.01),
            relative_width=0.12,
            nonnegative=True,
        )
        for index, name in enumerate(objective_names)
    }
    confidence_values = list(confidences.values()) or [0.5]
    confidence_floor = min(confidence_values)
    confidence_mean = sum(confidence_values) / float(len(confidence_values))
    return CandidateEnvelope(
        schema_version=CANDIDATE_ENVELOPE_SCHEMA_VERSION,
        objective_bounds=objective_bounds,
        refine_cost_bounds=bounds_from_confidence(
            float(predicted_refine_cost),
            confidence=confidence_mean,
            minimum_width=max(1.0, abs(float(predicted_refine_cost)) * 0.05),
            relative_width=0.25,
            nonnegative=True,
        ),
        overlap_bounds=bounds_from_confidence(
            float(overlap),
            confidence=confidence_floor,
            minimum_width=0.02,
            relative_width=0.45,
            nonnegative=True,
            upper_cap=1.0,
        ),
        stretch_bounds=bounds_from_confidence(
            float(stretch),
            confidence=confidence_mean,
            minimum_width=0.02,
            relative_width=0.18,
            nonnegative=True,
        ),
        confidence_floor=float(confidence_floor),
        confidence_mean=float(confidence_mean),
    )
