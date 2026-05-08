"""Certificate-boundary wrappers for REFC decision-region scaffolding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from .flip_radius import StructuredAdversarialBudget


_TIME_PRESERVING_FAMILIES = {"scenario", "terrain", "weather", "stochastic"}
_MONETARY_FAMILIES = {"toll", "fuel", "carbon"}


def infer_preference_direction_from_family(
    dominant_evidence_family: str | None,
    *,
    support_flag: bool = True,
) -> str | None:
    family = str(dominant_evidence_family or "").strip().lower()
    if not support_flag or not family:
        return None
    if family in _TIME_PRESERVING_FAMILIES:
        return "guard:time_preserving"
    if family in _MONETARY_FAMILIES:
        return "tradeoff:time_vs_money"
    return f"inspect:{family}"


def choose_decision_boundary(
    *,
    support_flag: bool,
    active_challenger_id: str | None,
    minimum_pairwise_gap: float | None,
    minimum_flip_budget: float | None,
    preference_direction: str | None,
    dominant_evidence_family: str | None,
) -> tuple[str | None, str | None]:
    if not support_flag:
        return "support", "evidence"
    if (
        minimum_flip_budget is not None
        and dominant_evidence_family is not None
        and (minimum_pairwise_gap is None or minimum_flip_budget <= minimum_pairwise_gap)
    ):
        return "flip_radius", "evidence"
    if active_challenger_id is not None:
        return "pairwise_gap", "search"
    if preference_direction is not None:
        return "preference", "preference"
    if dominant_evidence_family is not None:
        return "evidence_family", "evidence"
    return "support", "evidence"


@dataclass(frozen=True)
class DecisionRegionState:
    route_id: str
    nearest_certificate_boundary: str | None = None
    active_challenger_id: str | None = None
    dominant_evidence_family: str | None = None
    most_fragile_preference_direction: str | None = None
    minimum_joint_perturbation: float | None = None
    nearest_threat_axis: str | None = None
    selected_certificate_basis: str | None = None
    support_status: str | None = None
    support_bin: str | None = None
    calibration_bin: str | None = None
    calibration_policy_version: str | None = None
    nearest_challenger_gap_lower_bound: float | None = None
    nearest_challenger_audit_sensitivity: float | None = None
    nearest_challenger_radius: float | None = None
    nearest_challenger_flip_budget: float | None = None
    route_fragility_family_count: int = 0
    atlas_kind: str | None = None
    root_cause_tags: list[str] = field(default_factory=list)
    structured_adversarial_budget: StructuredAdversarialBudget | None = None
    support_flag: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)
    certified: bool = False
    abstain: bool = False

    @classmethod
    def from_states(
        cls,
        winner_state: Any,
        challenger_states: list[Any] | tuple[Any, ...],
        pairwise_states: list[Any] | tuple[Any, ...],
        flip_radius_state: Any,
        *,
        threshold: float,
    ) -> "DecisionRegionState":
        route_id = str(getattr(winner_state, "route_id", ""))
        challengers = list(challenger_states or [])
        pairwise_rows = list(pairwise_states or [])
        nearest_pairwise = next(
            (
                state
                for state in pairwise_rows
                if getattr(state, "nearest_challenger", False)
            ),
            None,
        )
        if nearest_pairwise is None and pairwise_rows:
            nearest_pairwise = min(
                pairwise_rows,
                key=lambda state: (
                    _coerce_float(getattr(state, "pairwise_gap_lower_bound", 0.0)),
                    str(getattr(state, "challenger_id", "")),
                ),
            )
        active_challenger_id = (
            None
            if nearest_pairwise is None
            else str(getattr(nearest_pairwise, "challenger_id", ""))
        )
        minimum_pairwise_gap = (
            None
            if nearest_pairwise is None
            else _coerce_float(
                getattr(nearest_pairwise, "pairwise_gap_lower_bound", 0.0)
            )
        )
        minimum_flip_budget = getattr(flip_radius_state, "minimum_flip_budget", None)
        minimum_flip_budget_value = (
            None if minimum_flip_budget is None else _coerce_float(minimum_flip_budget)
        )
        winner_point = _coerce_float(
            getattr(
                winner_state,
                "point_estimate",
                getattr(winner_state, "empirical_win", 0.0),
            )
        )
        threshold_value = _coerce_float(threshold)
        support_flag = bool(getattr(winner_state, "support_flag", True)) and bool(
            getattr(flip_radius_state, "support_flag", True)
        )
        support_flag = support_flag and all(
            bool(getattr(state, "support_flag", True))
            for state in [*challengers, *pairwise_rows]
        )
        nearest_boundary, nearest_axis = choose_decision_boundary(
            support_flag=support_flag,
            active_challenger_id=active_challenger_id,
            minimum_pairwise_gap=minimum_pairwise_gap,
            minimum_flip_budget=minimum_flip_budget_value,
            preference_direction=(
                "certificate_margin" if nearest_pairwise is not None else None
            ),
            dominant_evidence_family=getattr(
                flip_radius_state,
                "dominant_fragility_family",
                None,
            ),
        )
        has_positive_gap = (
            minimum_pairwise_gap is not None and minimum_pairwise_gap > 0.0
        )
        has_positive_budget = (
            minimum_flip_budget_value is not None and minimum_flip_budget_value > 0.0
        )
        certified = bool(
            support_flag
            and winner_point >= threshold_value
            and has_positive_gap
            and has_positive_budget
        )
        root_cause_tags = []
        if active_challenger_id:
            root_cause_tags.append("active_challenger_present")
        if minimum_pairwise_gap is not None and minimum_pairwise_gap <= 0.0:
            root_cause_tags.append("pairwise_gap_unresolved")
        if winner_point < threshold_value:
            root_cause_tags.append("winner_below_threshold")
        if nearest_boundary:
            root_cause_tags.append(f"boundary:{nearest_boundary}")
        return cls(
            route_id=route_id,
            nearest_certificate_boundary=nearest_boundary,
            active_challenger_id=active_challenger_id,
            dominant_evidence_family=getattr(
                flip_radius_state,
                "dominant_fragility_family",
                None,
            ),
            most_fragile_preference_direction=(
                "certificate_margin" if nearest_pairwise is not None else None
            ),
            minimum_joint_perturbation=(
                None
                if minimum_pairwise_gap is None and minimum_flip_budget_value is None
                else round(
                    min(
                        value
                        for value in (minimum_pairwise_gap, minimum_flip_budget_value)
                        if value is not None
                    ),
                    6,
                )
            ),
            nearest_threat_axis=nearest_axis,
            selected_certificate_basis="empirical",
            support_status="supported" if support_flag else "unsupported",
            support_bin="supported" if support_flag else "unsupported",
            nearest_challenger_gap_lower_bound=minimum_pairwise_gap,
            nearest_challenger_audit_sensitivity=(
                None
                if nearest_pairwise is None
                else _coerce_float(
                    getattr(nearest_pairwise, "challenger_audit_sensitivity", 0.0)
                )
            ),
            nearest_challenger_radius=(
                None
                if nearest_pairwise is None
                or getattr(nearest_pairwise, "challenger_radius", None) is None
                else _coerce_float(getattr(nearest_pairwise, "challenger_radius"))
            ),
            nearest_challenger_flip_budget=(
                None
                if nearest_pairwise is None
                or getattr(nearest_pairwise, "flip_budget", None) is None
                else _coerce_float(getattr(nearest_pairwise, "flip_budget"))
            ),
            root_cause_tags=root_cause_tags,
            structured_adversarial_budget=getattr(
                flip_radius_state,
                "structured_adversarial_budget",
                None,
            ),
            support_flag=support_flag,
            provenance={
                "selected_route_id": route_id,
                "threshold": threshold_value,
                "winner_point_estimate": winner_point,
                "minimum_pairwise_gap_lcb": minimum_pairwise_gap,
                "minimum_flip_budget": minimum_flip_budget_value,
            },
            certified=certified,
            abstain=not certified,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
