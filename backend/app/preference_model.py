from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import math
from typing import Any, Literal, Mapping, Sequence

ObjectiveName = Literal["time", "money", "co2"]
ConstraintKind = Literal[
    "objective_upper_bound",
    "objective_lower_bound",
    "time_guard",
    "route_veto",
    "toggle_preference",
    "optimization_mode",
]
StopHintSeverity = Literal["info", "warn", "block"]

OBJECTIVE_NAMES: tuple[ObjectiveName, ...] = ("time", "money", "co2")
OBJECTIVE_IRRELEVANCE_FLOORS: dict[ObjectiveName, float] = {
    "time": 300.0,
    "money": 5.0,
    "co2": 2.0,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _mapping_from_any(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    dump = getattr(value, "dict", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = _as_text(raw)
    if text is None:
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class PreferenceWeights:
    time: float = 1.0
    money: float = 1.0
    co2: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Any) -> "PreferenceWeights":
        mapping = _mapping_from_any(raw)
        if not mapping:
            return cls()
        money = mapping.get("money", mapping.get("cost", mapping.get("monetary_cost", 1.0)))
        co2 = mapping.get("co2", mapping.get("emissions", mapping.get("emissions_kg", 1.0)))
        return cls(
            time=max(0.0, _as_float(mapping.get("time"), 1.0)),
            money=max(0.0, _as_float(money, 1.0)),
            co2=max(0.0, _as_float(co2, 1.0)),
        )

    def normalized(self) -> "PreferenceWeights":
        total = self.time + self.money + self.co2
        if total <= 0.0:
            return PreferenceWeights(time=1.0 / 3.0, money=1.0 / 3.0, co2=1.0 / 3.0)
        return PreferenceWeights(
            time=self.time / total,
            money=self.money / total,
            co2=self.co2 / total,
        )

    def as_tuple(self) -> tuple[float, float, float]:
        normalized = self.normalized()
        return (normalized.time, normalized.money, normalized.co2)

    def as_dict(self) -> dict[str, float]:
        normalized = self.normalized()
        return {
            "time": normalized.time,
            "money": normalized.money,
            "co2": normalized.co2,
        }

    def dominant_objective(self) -> ObjectiveName:
        normalized = self.normalized()
        ordered = sorted(
            (
                ("time", normalized.time),
                ("money", normalized.money),
                ("co2", normalized.co2),
            ),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
        return ordered[0][0]  # type: ignore[return-value]

    def is_neutral(self, tolerance: float = 0.05) -> bool:
        normalized = self.normalized()
        values = (normalized.time, normalized.money, normalized.co2)
        return max(values) - min(values) <= max(0.0, tolerance)


@dataclass(frozen=True)
class TimeGuard:
    max_duration_s: float | None = None
    max_weather_delay_s: float | None = None
    max_incident_delay_s: float | None = None
    latest_arrival_utc: str | None = None

    def is_active(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_duration_s,
                self.max_weather_delay_s,
                self.max_incident_delay_s,
                self.latest_arrival_utc,
            )
        )

    def as_constraint_value(self) -> dict[str, float | str]:
        payload: dict[str, float | str] = {}
        if self.max_duration_s is not None:
            payload["max_duration_s"] = float(self.max_duration_s)
        if self.max_weather_delay_s is not None:
            payload["max_weather_delay_s"] = float(self.max_weather_delay_s)
        if self.max_incident_delay_s is not None:
            payload["max_incident_delay_s"] = float(self.max_incident_delay_s)
        if self.latest_arrival_utc is not None:
            payload["latest_arrival_utc"] = self.latest_arrival_utc
        return payload

    def evaluate(
        self,
        route: "RoutePreferenceSummary",
        *,
        departure_time_utc: str | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if self.max_duration_s is not None and route.duration_s > float(self.max_duration_s):
            reasons.append("time_guard:max_duration_s")
        if self.max_weather_delay_s is not None and route.weather_delay_s > float(self.max_weather_delay_s):
            reasons.append("time_guard:max_weather_delay_s")
        if self.max_incident_delay_s is not None and route.incident_delay_s > float(self.max_incident_delay_s):
            reasons.append("time_guard:max_incident_delay_s")
        latest_arrival = _parse_iso_utc(self.latest_arrival_utc)
        departure = _parse_iso_utc(departure_time_utc)
        if latest_arrival is not None and departure is not None:
            arrival = departure + timedelta(seconds=max(0.0, route.duration_s))
            if arrival > latest_arrival:
                reasons.append("time_guard:latest_arrival_utc")
        return (len(reasons) == 0, tuple(reasons))


@dataclass(frozen=True)
class ElicitedConstraint:
    kind: ConstraintKind
    target: str
    value: float | bool | str | Mapping[str, Any] | None = None
    label: str | None = None
    source: str = "user"
    strict: bool = True
    rationale: str | None = None

    @classmethod
    def maximum(cls, objective: ObjectiveName, value: float, *, source: str = "user") -> "ElicitedConstraint":
        return cls(
            kind="objective_upper_bound",
            target=objective,
            value=float(value),
            label=f"max_{objective}",
            source=source,
        )

    @classmethod
    def minimum(cls, objective: ObjectiveName, value: float, *, source: str = "user") -> "ElicitedConstraint":
        return cls(
            kind="objective_lower_bound",
            target=objective,
            value=float(value),
            label=f"min_{objective}",
            source=source,
        )

    @classmethod
    def time_guard(cls, guard: TimeGuard, *, source: str = "user") -> "ElicitedConstraint":
        return cls(
            kind="time_guard",
            target="time_guard",
            value=guard.as_constraint_value(),
            label="time_guard",
            source=source,
        )

    @classmethod
    def veto(cls, target: str, *, source: str = "user", rationale: str | None = None) -> "ElicitedConstraint":
        return cls(
            kind="route_veto",
            target=target,
            value=True,
            label=f"veto_{target}",
            source=source,
            rationale=rationale,
        )

    def key(self) -> str:
        return self.label or f"{self.kind}:{self.target}"


@dataclass(frozen=True)
class RoutePreferenceSummary:
    route_id: str
    duration_s: float
    monetary_cost: float
    emissions_kg: float
    distance_km: float
    certificate: float | None = None
    certified: bool = False
    threshold: float | None = None
    uses_tolls: bool = False
    toll_cost: float = 0.0
    weather_delay_s: float = 0.0
    incident_delay_s: float = 0.0
    uncertainty_mass: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_raw(
        cls,
        route: "RoutePreferenceSummary | Mapping[str, Any] | Any",
        *,
        fallback_threshold: float | None = None,
        certificate_map: Mapping[str, float] | None = None,
    ) -> "RoutePreferenceSummary":
        if isinstance(route, RoutePreferenceSummary):
            return route
        mapping = _mapping_from_any(route)
        metrics = _mapping_from_any(mapping.get("metrics"))
        certification = _mapping_from_any(mapping.get("certification"))
        route_id = _as_text(mapping.get("route_id", mapping.get("id"))) or "route_unknown"
        toll_cost = 0.0
        raw_rows = mapping.get("segment_breakdown", [])
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
            for row in raw_rows:
                row_map = _mapping_from_any(row)
                toll_cost += max(0.0, _as_float(row_map.get("toll_cost")))
        toll_metadata = _mapping_from_any(mapping.get("toll_metadata"))
        uses_tolls = bool(
            toll_cost > 0.0
            or _as_bool(toll_metadata.get("has_tolls"))
            or _as_bool(toll_metadata.get("contains_tolls"))
        )
        if not uses_tolls:
            uses_tolls = _as_float(toll_metadata.get("toll_cost_total"), 0.0) > 0.0
        certificate = None
        if certification and _has_value(certification.get("certificate")):
            certificate = _as_float(certification.get("certificate"))
        elif certificate_map is not None and route_id in certificate_map:
            certificate = _as_float(certificate_map.get(route_id))
        threshold = None
        if certification and _has_value(certification.get("threshold")):
            threshold = _as_float(certification.get("threshold"))
        elif fallback_threshold is not None:
            threshold = float(fallback_threshold)
        uncertainty = _mapping_from_any(mapping.get("uncertainty"))
        duration_s = _as_float(metrics.get("duration_s", mapping.get("duration_s")))
        monetary_cost = _as_float(metrics.get("monetary_cost", mapping.get("money")))
        emissions_kg = _as_float(metrics.get("emissions_kg", mapping.get("co2")))
        std_duration = _as_float(uncertainty.get("std_duration_s"))
        std_money = _as_float(uncertainty.get("std_monetary_cost"))
        std_co2 = _as_float(uncertainty.get("std_emissions_kg"))
        uncertainty_mass = min(
            1.0,
            max(
                std_duration / max(1.0, duration_s),
                std_money / max(1.0, abs(monetary_cost)),
                std_co2 / max(1.0, emissions_kg),
            ),
        )
        certified = False
        if certification:
            certified = _as_bool(certification.get("certified"))
        elif certificate is not None and threshold is not None:
            certified = certificate >= threshold
        return cls(
            route_id=route_id,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            distance_km=max(0.0, _as_float(metrics.get("distance_km", mapping.get("distance_km")))),
            certificate=certificate,
            certified=certified,
            threshold=threshold,
            uses_tolls=uses_tolls,
            toll_cost=max(0.0, toll_cost),
            weather_delay_s=max(0.0, _as_float(metrics.get("weather_delay_s", 0.0))),
            incident_delay_s=max(0.0, _as_float(metrics.get("incident_delay_s", 0.0))),
            uncertainty_mass=uncertainty_mass,
            raw=dict(mapping),
        )

    def objective_value(self, objective: ObjectiveName) -> float:
        if objective == "time":
            return self.duration_s
        if objective == "money":
            return self.monetary_cost
        return self.emissions_kg


@dataclass(frozen=True)
class CompatibleSet:
    route_ids: tuple[str, ...]
    ranked_route_ids: tuple[str, ...]
    certified_route_ids: tuple[str, ...]
    vetoed_route_ids: tuple[str, ...]
    blocked_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    active_constraint_keys: tuple[str, ...] = ()

    def top_route_id(self) -> str | None:
        if not self.ranked_route_ids:
            return None
        return self.ranked_route_ids[0]


@dataclass(frozen=True)
class PreferenceStopHint:
    code: str
    message: str
    severity: StopHintSeverity = "info"
    route_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CompatibleSet",
    "ConstraintKind",
    "ElicitedConstraint",
    "OBJECTIVE_IRRELEVANCE_FLOORS",
    "OBJECTIVE_NAMES",
    "ObjectiveName",
    "PreferenceStopHint",
    "PreferenceWeights",
    "RoutePreferenceSummary",
    "StopHintSeverity",
    "TimeGuard",
    "_as_bool",
    "_as_float",
    "_has_value",
    "_as_text",
    "_mapping_from_any",
    "_parse_iso_utc",
]
