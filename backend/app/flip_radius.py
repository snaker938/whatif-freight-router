"""Deterministic and probabilistic flip-radius wrappers for REFC scaffolding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Mapping, Sequence


def _rounded_budget(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, float(value)), 6)


@dataclass(frozen=True)
class AdversarialBudgetChannelState:
    budget: float | None = None
    status: str = "not_applicable"
    unit: str = "normalized_margin"
    driver: str | None = None
    source_metric: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredAdversarialBudget:
    schema_version: str = "structured_adversarial_budget.v1"
    evidence_channel: AdversarialBudgetChannelState = field(default_factory=AdversarialBudgetChannelState)
    preference_channel: AdversarialBudgetChannelState = field(default_factory=AdversarialBudgetChannelState)
    search_deficiency_channel: AdversarialBudgetChannelState = field(default_factory=AdversarialBudgetChannelState)
    limiting_channel: str | None = None
    limiting_budget: float | None = None
    tracked_channels: list[str] = field(
        default_factory=lambda: ["evidence", "preference", "search_deficiency"]
    )
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)


def _build_budget_channel(
    *,
    budget: float | None,
    driver: str | None,
    source_metric: str | None,
    details: Mapping[str, Any] | None = None,
) -> AdversarialBudgetChannelState:
    normalized_budget = _rounded_budget(budget)
    if normalized_budget is None:
        status = "not_applicable"
    elif normalized_budget <= 0.0:
        status = "clear"
    else:
        status = "active"
    return AdversarialBudgetChannelState(
        budget=normalized_budget,
        status=status,
        driver=str(driver).strip() or None if driver is not None else None,
        source_metric=str(source_metric).strip() or None if source_metric is not None else None,
        details=dict(details or {}),
    )


def build_structured_adversarial_budget(
    *,
    evidence_budget: float | None,
    evidence_driver: str | None = None,
    evidence_source_metric: str | None = None,
    evidence_details: Mapping[str, Any] | None = None,
    preference_budget: float | None,
    preference_driver: str | None = None,
    preference_source_metric: str | None = None,
    preference_details: Mapping[str, Any] | None = None,
    search_deficiency_budget: float | None,
    search_deficiency_driver: str | None = None,
    search_deficiency_source_metric: str | None = None,
    search_deficiency_details: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> StructuredAdversarialBudget:
    evidence_channel = _build_budget_channel(
        budget=evidence_budget,
        driver=evidence_driver,
        source_metric=evidence_source_metric,
        details=evidence_details,
    )
    preference_channel = _build_budget_channel(
        budget=preference_budget,
        driver=preference_driver,
        source_metric=preference_source_metric,
        details=preference_details,
    )
    search_deficiency_channel = _build_budget_channel(
        budget=search_deficiency_budget,
        driver=search_deficiency_driver,
        source_metric=search_deficiency_source_metric,
        details=search_deficiency_details,
    )
    candidates = [
        ("evidence", evidence_channel.budget),
        ("preference", preference_channel.budget),
        ("search_deficiency", search_deficiency_channel.budget),
    ]
    limiting_channel = None
    limiting_budget = None
    finite_candidates = [
        (channel, float(budget))
        for channel, budget in candidates
        if budget is not None
    ]
    if finite_candidates:
        limiting_channel, limiting_budget_raw = min(
            finite_candidates,
            key=lambda item: (item[1], item[0]),
        )
        limiting_budget = _rounded_budget(limiting_budget_raw)
    return StructuredAdversarialBudget(
        evidence_channel=evidence_channel,
        preference_channel=preference_channel,
        search_deficiency_channel=search_deficiency_channel,
        limiting_channel=limiting_channel,
        limiting_budget=limiting_budget,
        provenance=dict(provenance or {}),
    )


def minimum_positive_value(values: Sequence[float | None]) -> float | None:
    positives = [float(value) for value in values if value is not None and float(value) > 0.0]
    if not positives:
        return None
    return min(positives)


def build_adversarial_degradation_curve(
    *,
    challenger_specific_radii: Mapping[str, float],
    evidence_family_radii: Mapping[str, float],
) -> dict[str, float]:
    pressure = max(
        [float(value) for value in challenger_specific_radii.values()]
        + [float(value) for value in evidence_family_radii.values()]
        + [0.0]
    )
    return {
        str(level): round(max(0.0, 1.0 - (pressure * level)), 6)
        for level in (0.25, 0.5, 0.75, 1.0)
    }


@dataclass(frozen=True)
class FlipRadiusState:
    route_id: str
    deterministic_local_flip_radius: float = 0.0
    probabilistic_flip_radius: float = 0.0
    challenger_specific_radii: dict[str, float] = field(default_factory=dict)
    evidence_family_radii: dict[str, float] = field(default_factory=dict)
    dominant_fragility_family: str | None = None
    minimum_flip_budget: float | None = None
    adversarial_degradation_curve: dict[str, float] = field(default_factory=dict)
    structured_adversarial_budget: StructuredAdversarialBudget | None = None
    support_flag: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pairwise_gap(
        cls,
        pairwise_gap: Any,
        *,
        objective_scale: float = 1.0,
    ) -> "FlipRadiusState":
        pairwise_provenance = dict(getattr(pairwise_gap, "provenance", {}) or {})
        route_id = str(
            pairwise_provenance.get("winner_id")
            or pairwise_provenance.get("selected_route_id")
            or "winner"
        )
        challenger_id = str(getattr(pairwise_gap, "challenger_id", "challenger"))
        mean_gap = _coerce_float(getattr(pairwise_gap, "mean_gap", None))
        lower_bound = _coerce_float(
            getattr(pairwise_gap, "pairwise_gap_lower_bound", mean_gap)
        )
        raw_radius = getattr(pairwise_gap, "challenger_radius", None)
        challenger_radius = _rounded_budget(
            raw_radius if raw_radius is not None else mean_gap
        ) or 0.0
        scale = max(abs(_coerce_float(objective_scale)), 1e-12)
        normalized_radius = round(challenger_radius / scale, 6)
        minimum_flip_budget = (
            challenger_radius if lower_bound > 0.0 and challenger_radius > 0.0 else None
        )
        challenger_specific_radii = {challenger_id: challenger_radius}
        return cls(
            route_id=route_id,
            deterministic_local_flip_radius=normalized_radius,
            probabilistic_flip_radius=normalized_radius,
            challenger_specific_radii=challenger_specific_radii,
            minimum_flip_budget=minimum_flip_budget,
            adversarial_degradation_curve=build_adversarial_degradation_curve(
                challenger_specific_radii=challenger_specific_radii,
                evidence_family_radii={},
            ),
            structured_adversarial_budget=build_structured_adversarial_budget(
                evidence_budget=minimum_flip_budget,
                evidence_driver=challenger_id,
                evidence_source_metric="pairwise_gap",
                evidence_details={"challenger_count": 1},
                preference_budget=None,
                preference_source_metric="most_fragile_preference_direction",
                search_deficiency_budget=None,
                search_deficiency_source_metric="search_completeness_gap",
                provenance={
                    "selected_route_id": route_id,
                    "source": "pairwise_gap",
                },
            ),
            support_flag=bool(getattr(pairwise_gap, "support_flag", True)),
            provenance={
                "selected_route_id": route_id,
                "source": "pairwise_gap",
                "objective_scale": round(scale, 6),
                "pairwise_gap": mean_gap,
            },
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
