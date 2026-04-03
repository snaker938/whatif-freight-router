from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .decision_region import DecisionRegionState

ABSTENTION_REASON_CODES: tuple[str, ...] = (
    "winner_threshold_unproven",
    "challenger_region_overlap",
    "pairwise_gap_unproven",
    "flip_radius_zero",
    "support_insufficient",
)


@dataclass(frozen=True)
class AbstentionRecord:
    reason_code: str
    route_id: str
    winner_id: str
    challenger_id: str | None
    trigger_metric: str
    observed_value: float
    threshold: float
    support_strength: float
    recommended_action: str
    severity: str
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_decision_region(
        cls,
        decision_region: DecisionRegionState,
    ) -> "AbstentionRecord":
        reason_code = (
            decision_region.reason_codes[0]
            if decision_region.reason_codes
            else "support_insufficient"
        )
        if reason_code == "winner_threshold_unproven":
            trigger_metric = "threshold_margin"
            observed_value = decision_region.threshold_margin
            threshold = 0.0
            recommended_action = "expand_worlds"
        elif reason_code == "challenger_region_overlap":
            trigger_metric = "pairwise_margin"
            observed_value = decision_region.pairwise_margin
            threshold = 0.0
            recommended_action = "evaluate_challenger"
        else:
            trigger_metric = "support_strength"
            observed_value = decision_region.support_strength
            threshold = 0.5
            recommended_action = "collect_audit_evidence"
        severity = "high" if decision_region.abstain else "medium"
        return cls(
            reason_code=reason_code,
            route_id=decision_region.winner_id,
            winner_id=decision_region.winner_id,
            challenger_id=decision_region.best_challenger_id,
            trigger_metric=trigger_metric,
            observed_value=round(float(observed_value), 6),
            threshold=round(float(threshold), 6),
            support_strength=round(float(decision_region.support_strength), 6),
            recommended_action=recommended_action,
            severity=severity,
            detail="decision_region_abstention" if decision_region.abstain else "decision_region_uncertain",
        )
