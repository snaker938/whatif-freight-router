"""Anytime-valid confidence-sequence helpers for REFC winner bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Sequence


DEFAULT_CONFIDENCE_DELTA = 0.05
ANYTIME_HOEFFDING_METHOD = "anytime_hoeffding_union_bound"


def _coerce_delta(delta: float | None) -> float:
    try:
        parsed = float(delta) if delta is not None else DEFAULT_CONFIDENCE_DELTA
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE_DELTA
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed >= 1.0:
        return DEFAULT_CONFIDENCE_DELTA
    return parsed


def anytime_hoeffding_interval(
    success_count: int,
    sample_count: int,
    *,
    delta: float = DEFAULT_CONFIDENCE_DELTA,
) -> tuple[float, float]:
    """Return a conservative anytime-valid Bernoulli interval.

    The construction uses Hoeffding's inequality with a summable time schedule
    ``delta_n = delta / (n (n + 1))``. By a union bound over all ``n >= 1``,
    the interval is valid uniformly over optional stopping times.
    """

    n = max(0, int(sample_count))
    if n <= 0:
        return 0.0, 1.0

    successes = min(max(0, int(success_count)), n)
    delta_value = _coerce_delta(delta)
    empirical = successes / float(n)
    log_term = math.log((2.0 * n * (n + 1)) / delta_value)
    radius = math.sqrt(max(0.0, log_term) / (2.0 * n))
    return max(0.0, empirical - radius), min(1.0, empirical + radius)


def _coerce_probability(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return min(1.0, max(0.0, parsed))


@dataclass(frozen=True)
class WinnerConfidenceState:
    route_id: str
    empirical_win: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    method: str = "empirical_winner_frequency"
    delta: float = 0.05
    sample_count: int = 0
    stopping_valid_trace_state: dict[str, Any] = field(default_factory=dict)
    support_flag: bool = True
    support_reason: str | None = None

    @property
    def point_estimate(self) -> float:
        return self.empirical_win

    @property
    def width(self) -> float:
        return max(0.0, self.upper_bound - self.lower_bound)

    @property
    def effective_sample_count(self) -> float:
        value = self.stopping_valid_trace_state.get(
            "effective_sample_count",
            self.sample_count,
        )
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return float(max(0, self.sample_count))

    @classmethod
    def from_point_estimate(
        cls,
        route_id: str,
        *,
        point_estimate: float,
        sample_count: int,
        threshold: float | None = None,
        confidence_level: float | None = None,
        support_strength: float = 1.0,
        proxy_fraction: float = 0.0,
        delta: float | None = None,
        support_flag: bool = True,
        support_reason: str | None = None,
    ) -> "WinnerConfidenceState":
        n = max(0, int(sample_count or 0))
        empirical = _coerce_probability(point_estimate)
        successes = int(round(empirical * n))
        confidence_delta = (
            1.0 - float(confidence_level)
            if confidence_level is not None
            else None
        )
        delta_value = _coerce_delta(delta if delta is not None else confidence_delta)
        lower_bound, upper_bound = anytime_hoeffding_interval(successes, n, delta=delta_value)
        effective_n = (
            n
            * _coerce_probability(support_strength, default=1.0)
            * (1.0 - _coerce_probability(proxy_fraction))
        )
        return cls(
            route_id=str(route_id),
            empirical_win=round(empirical, 6),
            lower_bound=round(lower_bound, 6),
            upper_bound=round(upper_bound, 6),
            method=ANYTIME_HOEFFDING_METHOD,
            delta=delta_value,
            sample_count=n,
            stopping_valid_trace_state={
                "threshold": threshold,
                "confidence_level": 1.0 - delta_value,
                "support_strength": support_strength,
                "proxy_fraction": proxy_fraction,
                "effective_sample_count": round(effective_n, 6),
            },
            support_flag=bool(support_flag),
            support_reason=support_reason,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)


def winner_confidence_sequence(
    route_id: str,
    observations: Sequence[int | float | bool],
    *,
    threshold: float | None = None,
    confidence_level: float = 0.95,
    support_strength: float = 1.0,
    proxy_fraction: float = 0.0,
) -> list[WinnerConfidenceState]:
    delta = _coerce_delta(1.0 - float(confidence_level))
    states: list[WinnerConfidenceState] = []
    success_count = 0
    for sample_count, observation in enumerate(observations, start=1):
        success_count += 1 if _coerce_probability(observation) > 0.0 else 0
        point_estimate = success_count / float(sample_count)
        lower_bound, upper_bound = anytime_hoeffding_interval(success_count, sample_count, delta=delta)
        effective_n = (
            sample_count
            * _coerce_probability(support_strength, default=1.0)
            * (1.0 - _coerce_probability(proxy_fraction))
        )
        states.append(
            WinnerConfidenceState(
                route_id=str(route_id),
                empirical_win=round(point_estimate, 6),
                lower_bound=round(lower_bound, 6),
                upper_bound=round(upper_bound, 6),
                method=ANYTIME_HOEFFDING_METHOD,
                delta=delta,
                sample_count=sample_count,
                stopping_valid_trace_state={
                    "threshold": threshold,
                    "confidence_level": 1.0 - delta,
                    "support_strength": support_strength,
                    "proxy_fraction": proxy_fraction,
                    "effective_sample_count": round(effective_n, 6),
                    "success_count": success_count,
                },
            )
        )
    return states
