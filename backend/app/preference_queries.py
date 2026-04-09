from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

PreferenceQueryType = Literal["pairwise", "threshold", "ratio", "veto", "time_guard"]


class PairwisePreferenceQuery(BaseModel):
    query_type: Literal["pairwise"] = "pairwise"
    preferred_route_id: str
    challenger_route_id: str
    reason: str | None = None
    weight_hint: dict[str, float] = Field(default_factory=dict)


class ThresholdPreferenceQuery(BaseModel):
    query_type: Literal["threshold"] = "threshold"
    route_id: str
    metric_name: str
    threshold_value: float
    direction: Literal["gte", "lte"] = "lte"
    reason: str | None = None


class RatioPreferenceQuery(BaseModel):
    query_type: Literal["ratio"] = "ratio"
    route_id: str
    numerator_metric: str
    denominator_metric: str
    minimum_ratio: float = Field(ge=0.0)
    reason: str | None = None


class VetoPreferenceQuery(BaseModel):
    query_type: Literal["veto"] = "veto"
    route_id: str
    veto_name: str
    active: bool = True
    reason: str | None = None


class TimeGuardPreferenceQuery(BaseModel):
    query_type: Literal["time_guard"] = "time_guard"
    route_id: str
    latest_arrival_utc: str | None = None
    max_travel_time_s: float | None = Field(default=None, ge=0.0)
    preserve_time_budget_s: float | None = Field(default=None, ge=0.0)
    reason: str | None = None


PreferenceQuery = Annotated[
    Union[
        PairwisePreferenceQuery,
        ThresholdPreferenceQuery,
        RatioPreferenceQuery,
        VetoPreferenceQuery,
        TimeGuardPreferenceQuery,
    ],
    Field(discriminator="query_type"),
]

