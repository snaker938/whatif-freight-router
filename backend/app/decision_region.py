from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .confidence_sequences import WinnerConfidenceState
from .flip_radius import FlipRadiusState
from .pairwise_gap_model import PairwiseGapState


@dataclass(frozen=True)
class DecisionRegionState:
    winner_id: str
    best_challenger_id: str | None
    certification_threshold: float
    winner_lower_bound: float
    winner_upper_bound: float
    best_challenger_lower_bound: float
    best_challenger_upper_bound: float
    threshold_margin: float
    pairwise_margin: float
    minimum_pairwise_gap: float
    flip_radius: float
    support_strength: float
    status: str
    reason_codes: tuple[str, ...]
    abstain: bool

    @property
    def certified(self) -> bool:
        return self.status == "certified"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_states(
        cls,
        winner_confidence: WinnerConfidenceState,
        challenger_confidences: Sequence[WinnerConfidenceState],
        pairwise_gaps: Sequence[PairwiseGapState],
        flip_radius: FlipRadiusState | None,
        *,
        threshold: float,
    ) -> "DecisionRegionState":
        challengers = list(challenger_confidences)
        best_challenger = None
        if challengers:
            best_challenger = max(
                challengers,
                key=lambda state: (state.upper_bound, state.point_estimate, state.route_id),
            )
        pairwise_by_id = {gap.challenger_id: gap for gap in pairwise_gaps}
        best_pairwise = (
            pairwise_by_id.get(best_challenger.route_id)
            if best_challenger is not None
            else None
        )
        reasons: list[str] = []
        threshold_margin = winner_confidence.lower_bound - float(threshold)
        pairwise_margin = (
            winner_confidence.lower_bound - best_challenger.upper_bound
            if best_challenger is not None
            else winner_confidence.lower_bound
        )
        minimum_pairwise_gap = best_pairwise.lower_gap if best_pairwise is not None else 0.0
        if threshold_margin < 0.0:
            reasons.append("winner_threshold_unproven")
        if best_challenger is not None and pairwise_margin <= 0.0:
            reasons.append("challenger_region_overlap")
        if best_pairwise is not None and best_pairwise.lower_gap <= 0.0:
            reasons.append("pairwise_gap_unproven")
        if flip_radius is not None and flip_radius.proxy_adjusted_radius <= 0.0:
            reasons.append("flip_radius_zero")
        if not reasons:
            status = "certified"
            abstain = False
        elif winner_confidence.support_strength < 0.5 or (flip_radius is not None and flip_radius.fragile):
            status = "abstain"
            abstain = True
        else:
            status = "uncertain"
            abstain = False
        return cls(
            winner_id=winner_confidence.route_id,
            best_challenger_id=best_challenger.route_id if best_challenger is not None else None,
            certification_threshold=round(float(threshold), 6),
            winner_lower_bound=winner_confidence.lower_bound,
            winner_upper_bound=winner_confidence.upper_bound,
            best_challenger_lower_bound=best_challenger.lower_bound if best_challenger is not None else 0.0,
            best_challenger_upper_bound=best_challenger.upper_bound if best_challenger is not None else 0.0,
            threshold_margin=round(threshold_margin, 6),
            pairwise_margin=round(pairwise_margin, 6),
            minimum_pairwise_gap=round(minimum_pairwise_gap, 6),
            flip_radius=round(flip_radius.proxy_adjusted_radius if flip_radius is not None else 0.0, 6),
            support_strength=winner_confidence.support_strength,
            status=status,
            reason_codes=tuple(reasons),
            abstain=abstain,
        )
