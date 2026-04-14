"""Anytime-valid confidence-sequence helpers for REFC winner bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
