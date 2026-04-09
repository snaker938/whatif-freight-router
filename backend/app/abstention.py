from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .decision_region import DecisionRegionState
from .preference_state import PreferenceState

ABSTENTION_REASON_CODES: tuple[str, ...] = (
    "typed_abstention_recommended",
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
    def from_preference_state(
        cls,
        preference_state: PreferenceState,
        *,
        decision_region: DecisionRegionState | None = None,
    ) -> "AbstentionRecord":
        if decision_region is not None and decision_region.abstain:
            record = cls.from_decision_region(decision_region)
            if preference_state.wants_certified_only() and not preference_state.has_certified_compatible_route():
                return cls(
                    reason_code="typed_abstention_recommended",
                    route_id=record.route_id,
                    winner_id=record.winner_id,
                    challenger_id=record.challenger_id,
                    trigger_metric="certified_route_count",
                    observed_value=float(len(preference_state.compatible_set.certified_route_ids)),
                    threshold=1.0,
                    support_strength=record.support_strength,
                    recommended_action="expand_worlds",
                    severity="high",
                    detail="preference_requires_certified_route",
                )
            return record

        certified_routes = preference_state.compatible_set.certified_route_ids
        if preference_state.wants_certified_only() and not certified_routes:
            route_id = (
                preference_state.selected_route_id
                or preference_state.compatible_set.top_route_id()
                or (preference_state.frontier[0].route_id if preference_state.frontier else "route_unknown")
            )
            return cls(
                reason_code="typed_abstention_recommended",
                route_id=route_id,
                winner_id=route_id,
                challenger_id=(
                    preference_state.compatible_set.ranked_route_ids[1]
                    if len(preference_state.compatible_set.ranked_route_ids) > 1
                    else None
                ),
                trigger_metric="certified_route_count",
                observed_value=float(len(certified_routes)),
                threshold=1.0,
                support_strength=0.0,
                recommended_action="expand_worlds",
                severity="high",
                detail="preference_state_requires_certified_route",
            )

        if not preference_state.compatible_set.route_ids:
            route_id = (
                preference_state.selected_route_id
                or (preference_state.frontier[0].route_id if preference_state.frontier else "route_unknown")
            )
            reason_code = (
                preference_state.stop_hints[0].code
                if preference_state.stop_hints
                else "support_insufficient"
            )
            trigger_metric = "support_strength"
            observed_value = 0.0
            threshold = 0.5
            recommended_action = "collect_audit_evidence"
            if preference_state.has_time_guard():
                recommended_action = "relax_time_guard"
            return cls(
                reason_code=reason_code if reason_code in ABSTENTION_REASON_CODES else "support_insufficient",
                route_id=route_id,
                winner_id=route_id,
                challenger_id=(
                    preference_state.compatible_set.ranked_route_ids[0]
                    if preference_state.compatible_set.ranked_route_ids
                    else None
                ),
                trigger_metric=trigger_metric,
                observed_value=round(float(observed_value), 6),
                threshold=round(float(threshold), 6),
                support_strength=0.0,
                recommended_action=recommended_action,
                severity="high" if preference_state.wants_certified_only() else "medium",
                detail="preference_state_infeasible",
            )

        if preference_state.stop_hints:
            hint = preference_state.stop_hints[0]
            if hint.code == "typed_abstention_recommended":
                abstention_type = str(hint.metadata.get("abstention_type", "support_insufficient"))
                if abstention_type == "uncertified_frontier":
                    trigger_metric = "certified_route_count"
                    observed_value = float(len(preference_state.compatible_set.certified_route_ids))
                    recommended_action = "expand_worlds"
                elif abstention_type == "certification_gap":
                    trigger_metric = "threshold_margin"
                    observed_value = float(preference_state.selected_certificate or 0.0)
                    recommended_action = "evaluate_challenger"
                else:
                    trigger_metric = "support_strength"
                    observed_value = 0.0
                    recommended_action = "collect_audit_evidence"
                return cls(
                    reason_code=hint.code,
                    route_id=preference_state.selected_route_id
                    or preference_state.compatible_set.top_route_id()
                    or preference_state.compatible_set.ranked_route_ids[0],
                    winner_id=preference_state.selected_route_id
                    or preference_state.compatible_set.top_route_id()
                    or preference_state.compatible_set.ranked_route_ids[0],
                    challenger_id=None,
                    trigger_metric=trigger_metric,
                    observed_value=round(float(observed_value), 6),
                    threshold=float(hint.metadata.get("certificate_threshold", 1.0) or 1.0),
                    support_strength=0.0,
                    recommended_action=str(hint.metadata.get("recommended_action", recommended_action)),
                    severity="high" if hint.severity == "block" else "medium",
                    detail=hint.message,
                )

        route_id = (
            preference_state.selected_route_id
            or preference_state.compatible_set.top_route_id()
            or (preference_state.frontier[0].route_id if preference_state.frontier else "route_unknown")
        )
        return cls(
            reason_code="support_insufficient",
            route_id=route_id,
            winner_id=route_id,
            challenger_id=None,
            trigger_metric="support_strength",
            observed_value=0.0,
            threshold=0.5,
            support_strength=0.0,
            recommended_action="collect_audit_evidence",
            severity="medium",
            detail="preference_state_no_abstention_trigger",
        )

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
