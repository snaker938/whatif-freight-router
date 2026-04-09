from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .preference_state import CompatibleSetSummary, PreferenceShrinkageTrace, PreferenceState


class CompatibleWeightSetSummary(BaseModel):
    route_ids: list[str] = Field(default_factory=list)
    compatible_weight_count: int = Field(default=0, ge=0)
    volume_proxy: float = Field(default=1.0, ge=0.0)
    support_flag: bool = True
    support_reason: str | None = None


def normalize_weight_vector(weight_vector: dict[str, float] | None) -> dict[str, float]:
    vector = {str(key): float(value) for key, value in (weight_vector or {}).items()}
    cleaned = {key: value for key, value in vector.items() if value >= 0.0 and value == value}
    total = sum(cleaned.values())
    if total <= 0.0:
        return {key: 0.0 for key in cleaned}
    return {key: round(value / total, 12) for key, value in cleaned.items()}


def build_compatible_set_summary(
    *,
    route_ids: list[str] | None = None,
    compatible_set_volume_proxy: float = 1.0,
    necessary_best_prob: float = 0.0,
    possible_best_prob: float = 1.0,
    support_flag: bool = True,
    support_reason: str | None = None,
) -> CompatibleSetSummary:
    return CompatibleSetSummary(
        route_ids=list(route_ids or []),
        compatible_set_volume_proxy=max(0.0, float(compatible_set_volume_proxy)),
        necessary_best_prob=max(0.0, min(1.0, float(necessary_best_prob))),
        possible_best_prob=max(0.0, min(1.0, float(possible_best_prob))),
        support_flag=bool(support_flag),
        support_reason=support_reason,
    )


def summarize_compatible_weight_set(
    *,
    route_ids: list[str] | None = None,
    weights: dict[str, float] | None = None,
    support_flag: bool = True,
    support_reason: str | None = None,
) -> CompatibleWeightSetSummary:
    normalized = normalize_weight_vector(weights)
    active_weight_count = sum(1 for value in normalized.values() if value > 0.0)
    return CompatibleWeightSetSummary(
        route_ids=list(route_ids or []),
        compatible_weight_count=active_weight_count,
        volume_proxy=1.0 if normalized else 0.0,
        support_flag=bool(support_flag),
        support_reason=support_reason,
    )


def build_preference_state(
    *,
    route_ids: list[str] | None = None,
    weights: dict[str, float] | None = None,
    support_flag: bool = True,
    support_reason: str | None = None,
) -> PreferenceState:
    summary = build_compatible_set_summary(
        route_ids=route_ids,
        support_flag=support_flag,
        support_reason=support_reason,
    )
    state = PreferenceState(compatible_set_summary=summary)
    if weights is not None:
        state.compatible_weights = [normalize_weight_vector(weights)]
    return state


def build_preference_shrinkage_trace(
    *,
    query_index: int,
    query_type: str,
    before_size: int,
    after_size: int,
    before_volume_proxy: float,
    after_volume_proxy: float,
    target_route_id: str | None = None,
    query_reason: str | None = None,
    preference_irrelevance: bool = False,
) -> PreferenceShrinkageTrace:
    before_size = max(0, int(before_size))
    after_size = max(0, int(after_size))
    before_volume_proxy = max(0.0, float(before_volume_proxy))
    after_volume_proxy = max(0.0, float(after_volume_proxy))
    predicted = 0.0 if before_size <= 0 else max(0.0, min(1.0, (before_size - after_size) / before_size))
    realized = 0.0 if before_volume_proxy <= 0 else max(
        0.0,
        min(1.0, (before_volume_proxy - after_volume_proxy) / before_volume_proxy),
    )
    return PreferenceShrinkageTrace(
        query_index=max(0, int(query_index)),
        query_type=str(query_type),
        before_size=before_size,
        after_size=after_size,
        before_volume_proxy=before_volume_proxy,
        after_volume_proxy=after_volume_proxy,
        predicted_shrinkage=predicted,
        realized_shrinkage=realized,
        target_route_id=target_route_id,
        query_reason=query_reason,
        preference_irrelevance=bool(preference_irrelevance),
    )
