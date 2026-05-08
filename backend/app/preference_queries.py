"""Pipeline stage: define typed preference-query payloads exchanged by the controller and API."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping, Sequence, Union

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


TypedPreferenceQuery = Annotated[
    Union[
        PairwisePreferenceQuery,
        ThresholdPreferenceQuery,
        RatioPreferenceQuery,
        VetoPreferenceQuery,
        TimeGuardPreferenceQuery,
    ],
    Field(discriminator="query_type"),
]


class PreferenceQuerySuggestion(BaseModel):
    key: str
    kind: str
    prompt: str
    rationale: str | None = None
    options: tuple[str, ...] = Field(default_factory=tuple)
    route_ids: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key", "kind", "prompt")
    @classmethod
    def _require_text(cls, value: str) -> str:
        return _clean_required_text(value)

    @field_validator("rationale")
    @classmethod
    def _normalize_rationale(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)

    @field_validator("options", "route_ids", mode="before")
    @classmethod
    def _coerce_text_tuple(cls, value: Sequence[object] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        return tuple(cleaned for item in value if (cleaned := str(item).strip()))

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Mapping[str, Any] | None) -> dict[str, Any]:
        return dict(value or {})

    @property
    def query_type(self) -> str:
        return self.kind


class _PreferenceQueryCompat:
    def __call__(self, *args: Any, **kwargs: Any) -> PreferenceQuerySuggestion:
        return PreferenceQuerySuggestion(*args, **kwargs)

    def __get_pydantic_core_schema__(self, source_type: Any, handler: Any) -> Any:
        return handler.generate_schema(TypedPreferenceQuery)

    def __repr__(self) -> str:
        return "PreferenceQuery"


PreferenceQuery = _PreferenceQueryCompat()


def suggest_preference_queries(
    state: Any,
    *,
    limit: int = 3,
) -> list[PreferenceQuerySuggestion]:
    suggestions: list[PreferenceQuerySuggestion] = []
    route_ids, routes = _state_routes(state)
    certified_only_required = _bool_attr_or_method(state, "certified_only_required")
    vetoed_targets = list(_sequence_attr(state, "vetoed_targets"))

    if _should_suggest_certified_focus(state, routes, certified_only_required):
        suggestions.append(
            PreferenceQuerySuggestion(
                key="certified_focus",
                kind="certified_focus",
                prompt="Require a certified route?",
                rationale="Certification preferences can change the compatible route set.",
                options=("prefer_certified", "allow_uncertified"),
                route_ids=tuple(route_ids),
                metadata={
                    "certified_only_required": certified_only_required,
                    "vetoed_targets": vetoed_targets,
                },
            )
        )

    if len(route_ids) >= 2:
        suggestions.append(
            PreferenceQuerySuggestion(
                key="objective_tradeoff",
                kind="objective_tradeoff",
                prompt="Which objective should dominate this tradeoff?",
                rationale="Route alternatives expose a time, money, or emissions tradeoff.",
                options=("prefer_time", "prefer_money", "prefer_co2"),
                route_ids=tuple(route_ids[:2]),
                metadata={"dominant_objective": _dominant_objective(state)},
            )
        )

    cleaned_limit = max(0, int(limit))
    return suggestions[:cleaned_limit] if cleaned_limit else []


def _state_routes(state: Any) -> tuple[list[str], list[Any]]:
    routes = _sequence_attr(state, "frontier")
    if not routes:
        routes = _sequence_attr(state, "routes")
    if not routes:
        routes = _sequence_attr(state, "frontier_routes")

    route_ids = [_route_id(route) for route in routes]
    route_ids = [route_id for route_id in route_ids if route_id]
    if not route_ids:
        compatible_set = getattr(state, "compatible_set", None)
        route_ids = list(_sequence_attr(compatible_set, "route_ids"))
    if not route_ids:
        summary = getattr(state, "compatible_set_summary", None)
        route_ids = list(_sequence_attr(summary, "route_ids"))

    return _dedupe(route_ids), list(routes)


def _should_suggest_certified_focus(
    state: Any,
    routes: Sequence[Any],
    certified_only_required: bool,
) -> bool:
    if certified_only_required:
        return True
    if _bool_attr_or_method(state, "wants_certified_only"):
        return True
    if not routes:
        return False

    certified_flags = [_route_is_certified(route) for route in routes]
    if any(flag is None for flag in certified_flags):
        return False
    if not any(bool(flag) for flag in certified_flags):
        return True

    selected_route_id = getattr(state, "selected_route_id", None)
    selected_route = next((route for route in routes if _route_id(route) == selected_route_id), None)
    if selected_route is not None and _route_is_certified(selected_route) is False:
        return True
    return bool(getattr(state, "stop_reason", None))


def _route_id(route: Any) -> str | None:
    if isinstance(route, Mapping):
        raw_value = route.get("id") or route.get("route_id")
    else:
        raw_value = getattr(route, "id", None) or getattr(route, "route_id", None)
    cleaned = str(raw_value or "").strip()
    return cleaned or None


def _route_is_certified(route: Any) -> bool | None:
    certification = _mapping_or_attr(route, "certification")
    if certification is None:
        return None
    certified = _mapping_or_attr(certification, "certified")
    if certified is not None:
        return bool(certified)
    certificate = _mapping_or_attr(certification, "certificate")
    threshold = _mapping_or_attr(certification, "threshold")
    if certificate is None or threshold is None:
        return None
    return float(certificate) >= float(threshold)


def _mapping_or_attr(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _bool_attr_or_method(value: Any, name: str) -> bool:
    attr = getattr(value, name, False)
    if callable(attr):
        attr = attr()
    return bool(attr)


def _sequence_attr(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    attr = getattr(value, name, None)
    if callable(attr):
        attr = attr()
    if attr is None:
        return []
    return list(attr)


def _dominant_objective(state: Any) -> str | None:
    weights = getattr(state, "weights", None)
    dominant = getattr(weights, "dominant_objective", None)
    if callable(dominant):
        return str(dominant())
    if isinstance(weights, Mapping) and weights:
        return str(max(weights, key=lambda key: float(weights[key])))
    return None


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
