"""Pipeline stage: define typed preference-query payloads exchanged by the controller and API."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

PreferenceQueryType = Literal["pairwise", "threshold", "ratio", "veto", "time_guard"]


def _clean_required_text(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError("value must be non-empty")
    return cleaned


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


class PairwisePreferenceQuery(BaseModel):
    query_type: Literal["pairwise"] = "pairwise"
    preferred_route_id: str
    challenger_route_id: str
    reason: str | None = None
    weight_hint: dict[str, float] = Field(default_factory=dict)

    @field_validator("preferred_route_id", "challenger_route_id")
    @classmethod
    def _require_route_id(cls, value: str) -> str:
        return _clean_required_text(value)

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)

    @model_validator(mode="after")
    def _validate_distinct_routes(self) -> "PairwisePreferenceQuery":
        if self.preferred_route_id == self.challenger_route_id:
            raise ValueError("pairwise preference routes must be distinct")
        return self


class ThresholdPreferenceQuery(BaseModel):
    query_type: Literal["threshold"] = "threshold"
    route_id: str
    metric_name: str
    threshold_value: float
    direction: Literal["gte", "lte"] = "lte"
    reason: str | None = None

    @field_validator("route_id", "metric_name")
    @classmethod
    def _require_text(cls, value: str) -> str:
        return _clean_required_text(value)

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)


class RatioPreferenceQuery(BaseModel):
    query_type: Literal["ratio"] = "ratio"
    route_id: str
    numerator_metric: str
    denominator_metric: str
    minimum_ratio: float = Field(ge=0.0)
    reason: str | None = None

    @field_validator("route_id", "numerator_metric", "denominator_metric")
    @classmethod
    def _require_text(cls, value: str) -> str:
        return _clean_required_text(value)

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)

    @model_validator(mode="after")
    def _validate_distinct_metrics(self) -> "RatioPreferenceQuery":
        if self.numerator_metric == self.denominator_metric:
            raise ValueError("ratio preference metrics must be distinct")
        return self


class VetoPreferenceQuery(BaseModel):
    query_type: Literal["veto"] = "veto"
    route_id: str
    veto_name: str
    active: bool = True
    reason: str | None = None

    @field_validator("route_id", "veto_name")
    @classmethod
    def _require_text(cls, value: str) -> str:
        return _clean_required_text(value)

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)


class TimeGuardPreferenceQuery(BaseModel):
    query_type: Literal["time_guard"] = "time_guard"
    route_id: str
    latest_arrival_utc: str | None = None
    max_travel_time_s: float | None = Field(default=None, ge=0.0)
    preserve_time_budget_s: float | None = Field(default=None, ge=0.0)
    reason: str | None = None

    @field_validator("route_id")
    @classmethod
    def _require_route_id(cls, value: str) -> str:
        return _clean_required_text(value)

    @field_validator("latest_arrival_utc", "reason")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)

    @model_validator(mode="after")
    def _validate_guard_payload(self) -> "TimeGuardPreferenceQuery":
        if (
            self.latest_arrival_utc is None
            and self.max_travel_time_s is None
            and self.preserve_time_budget_s is None
        ):
            raise ValueError("time guard queries must include at least one concrete bound")
        if (
            self.max_travel_time_s is not None
            and self.preserve_time_budget_s is not None
            and self.preserve_time_budget_s > self.max_travel_time_s
        ):
            raise ValueError("preserve_time_budget_s cannot exceed max_travel_time_s")
        return self


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
