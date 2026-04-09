from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .preference_queries import (
    PairwisePreferenceQuery,
    PreferenceQuery,
    RatioPreferenceQuery,
    ThresholdPreferenceQuery,
    TimeGuardPreferenceQuery,
    VetoPreferenceQuery,
)


class CompatibleSetSummary(BaseModel):
    route_ids: list[str] = Field(default_factory=list)
    compatible_set_size: int = Field(default=0, ge=0)
    compatible_set_volume_proxy: float = Field(default=1.0, ge=0.0)
    necessary_best_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    possible_best_prob: float = Field(default=1.0, ge=0.0, le=1.0)
    support_flag: bool = True
    support_reason: str | None = None

    @field_validator("route_ids")
    @classmethod
    def _dedupe_route_ids(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for route_id in value:
            cleaned = str(route_id).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped

    @model_validator(mode="after")
    def _sync_size(self) -> "CompatibleSetSummary":
        if self.compatible_set_size != len(self.route_ids):
            self.compatible_set_size = len(self.route_ids)
        if self.necessary_best_prob > self.possible_best_prob:
            self.necessary_best_prob = self.possible_best_prob
        return self


class PreferenceShrinkageTrace(BaseModel):
    query_index: int = Field(ge=0)
    query_type: Literal["pairwise", "threshold", "ratio", "veto", "time_guard"]
    before_size: int = Field(ge=0)
    after_size: int = Field(ge=0)
    before_volume_proxy: float = Field(ge=0.0)
    after_volume_proxy: float = Field(ge=0.0)
    predicted_shrinkage: float = Field(default=0.0, ge=0.0)
    realized_shrinkage: float = Field(default=0.0, ge=0.0)
    target_route_id: str | None = None
    query_reason: str | None = None
    preference_irrelevance: bool = False


class PreferenceState(BaseModel):
    compatible_set_summary: CompatibleSetSummary = Field(default_factory=CompatibleSetSummary)
    compatible_weights: list[dict[str, float]] = Field(default_factory=list)
    pairwise_constraints: list[PairwisePreferenceQuery] = Field(default_factory=list)
    threshold_constraints: list[ThresholdPreferenceQuery] = Field(default_factory=list)
    ratio_constraints: list[RatioPreferenceQuery] = Field(default_factory=list)
    veto_rules: list[VetoPreferenceQuery] = Field(default_factory=list)
    time_preserving_guard_rules: list[TimeGuardPreferenceQuery] = Field(default_factory=list)
    query_history: list[PreferenceQuery] = Field(default_factory=list)
    shrinkage_trace: list[PreferenceShrinkageTrace] = Field(default_factory=list)
    derived_invariants: dict[str, bool] = Field(default_factory=dict)
    terminal_type: Literal["open", "certified", "abstained"] = "open"
    query_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _sync_state(self) -> "PreferenceState":
        self.query_count = len(self.query_history)
        self.compatible_set_summary.compatible_set_size = len(self.compatible_set_summary.route_ids)
        self.derived_invariants = {
            "necessary_best_prob_le_possible_best_prob": (
                self.compatible_set_summary.necessary_best_prob
                <= self.compatible_set_summary.possible_best_prob
            ),
            "compatible_set_nonnegative": self.compatible_set_summary.compatible_set_size >= 0,
            "compatible_volume_nonnegative": self.compatible_set_summary.compatible_set_volume_proxy >= 0.0,
            "query_history_matches_trace_or_zero": (
                len(self.shrinkage_trace) == 0 or len(self.shrinkage_trace) <= self.query_count
            ),
        }
        return self


def empty_preference_state(*, route_ids: list[str] | None = None) -> PreferenceState:
    return PreferenceState(
        compatible_set_summary=CompatibleSetSummary(route_ids=list(route_ids or [])),
    )

