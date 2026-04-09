from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .confidence_sequences import WinnerConfidenceState
from .decision_region import DecisionRegionState


@dataclass(frozen=True)
class CertifiedSetState:
    threshold: float
    winner_id: str
    certified_route_ids: tuple[str, ...]
    borderline_route_ids: tuple[str, ...]
    rejected_route_ids: tuple[str, ...]
    minimum_certified_lower_bound: float
    maximum_borderline_upper_bound: float
    safe: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_confidence_states(
        cls,
        confidence_states: Sequence[WinnerConfidenceState],
        *,
        threshold: float,
        winner_id: str,
        decision_region: DecisionRegionState | None = None,
    ) -> "CertifiedSetState":
        certified: list[str] = []
        borderline: list[str] = []
        rejected: list[str] = []
        minimum_certified_lower_bound = 1.0
        maximum_borderline_upper_bound = 0.0
        for state in confidence_states:
            if state.lower_bound >= float(threshold):
                if decision_region is not None and state.route_id == winner_id and not decision_region.certified:
                    borderline.append(state.route_id)
                    maximum_borderline_upper_bound = max(maximum_borderline_upper_bound, state.upper_bound)
                else:
                    certified.append(state.route_id)
                    minimum_certified_lower_bound = min(minimum_certified_lower_bound, state.lower_bound)
            elif state.upper_bound >= float(threshold):
                borderline.append(state.route_id)
                maximum_borderline_upper_bound = max(maximum_borderline_upper_bound, state.upper_bound)
            else:
                rejected.append(state.route_id)
        if not certified:
            minimum_certified_lower_bound = 0.0
        safe = all(
            state.lower_bound >= float(threshold)
            for state in confidence_states
            if state.route_id in certified
        ) and all(
            state.upper_bound < float(threshold)
            for state in confidence_states
            if state.route_id in rejected
        )
        if decision_region is not None and decision_region.certified:
            safe = safe and (winner_id in certified)
        return cls(
            threshold=round(float(threshold), 6),
            winner_id=str(winner_id),
            certified_route_ids=tuple(sorted(certified)),
            borderline_route_ids=tuple(sorted(borderline)),
            rejected_route_ids=tuple(sorted(rejected)),
            minimum_certified_lower_bound=round(minimum_certified_lower_bound, 6),
            maximum_borderline_upper_bound=round(maximum_borderline_upper_bound, 6),
            safe=bool(safe),
        )
