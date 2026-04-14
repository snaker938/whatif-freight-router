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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
