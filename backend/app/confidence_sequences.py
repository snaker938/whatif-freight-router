from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

from .world_policies import clamp01


def effective_sample_count(
    sample_count: int,
    *,
    support_strength: float = 1.0,
    proxy_fraction: float = 0.0,
) -> float:
    count = max(1, int(sample_count))
    support = max(0.2, clamp01(support_strength, 1.0))
    proxy_scale = max(0.25, 1.0 - (0.75 * clamp01(proxy_fraction)))
    return float(count) * support * proxy_scale


def confidence_radius(
    sample_count: int,
    *,
    confidence_level: float = 0.95,
    support_strength: float = 1.0,
    proxy_fraction: float = 0.0,
) -> float:
    delta = max(1e-6, min(0.25, 1.0 - clamp01(confidence_level, 0.95)))
    eff_n = max(
        1.0,
        effective_sample_count(
            sample_count,
            support_strength=support_strength,
            proxy_fraction=proxy_fraction,
        ),
    )
    return min(1.0, math.sqrt(math.log(2.0 / delta) / (2.0 * eff_n)))


@dataclass(frozen=True)
class WinnerConfidenceState:
    route_id: str
    wins: float
    sample_count: int
    effective_sample_count: float
    confidence_level: float
    point_estimate: float
    lower_bound: float
    upper_bound: float
    threshold: float
    support_strength: float
    proxy_fraction: float

    @property
    def width(self) -> float:
        return round(max(0.0, self.upper_bound - self.lower_bound), 6)

    @property
    def margin_to_threshold(self) -> float:
        return round(self.lower_bound - self.threshold, 6)

    @property
    def certified(self) -> bool:
        return self.lower_bound >= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_counts(
        cls,
        route_id: str,
        *,
        wins: float,
        sample_count: int,
        threshold: float,
        confidence_level: float = 0.95,
        support_strength: float = 1.0,
        proxy_fraction: float = 0.0,
    ) -> "WinnerConfidenceState":
        total = max(1, int(sample_count))
        bounded_wins = max(0.0, min(float(total), float(wins)))
        point_estimate = bounded_wins / float(total)
        radius = confidence_radius(
            total,
            confidence_level=confidence_level,
            support_strength=support_strength,
            proxy_fraction=proxy_fraction,
        )
        lower_bound = max(0.0, point_estimate - radius)
        upper_bound = min(1.0, point_estimate + radius)
        return cls(
            route_id=str(route_id),
            wins=round(bounded_wins, 6),
            sample_count=total,
            effective_sample_count=round(
                effective_sample_count(
                    total,
                    support_strength=support_strength,
                    proxy_fraction=proxy_fraction,
                ),
                6,
            ),
            confidence_level=round(clamp01(confidence_level, 0.95), 6),
            point_estimate=round(point_estimate, 6),
            lower_bound=round(lower_bound, 6),
            upper_bound=round(upper_bound, 6),
            threshold=round(clamp01(threshold), 6),
            support_strength=round(clamp01(support_strength, 1.0), 6),
            proxy_fraction=round(clamp01(proxy_fraction), 6),
        )

    @classmethod
    def from_point_estimate(
        cls,
        route_id: str,
        *,
        point_estimate: float,
        sample_count: int,
        threshold: float,
        confidence_level: float = 0.95,
        support_strength: float = 1.0,
        proxy_fraction: float = 0.0,
    ) -> "WinnerConfidenceState":
        total = max(1, int(sample_count))
        bounded_point = clamp01(point_estimate)
        return cls.from_counts(
            route_id,
            wins=round(bounded_point * float(total), 6),
            sample_count=total,
            threshold=threshold,
            confidence_level=confidence_level,
            support_strength=support_strength,
            proxy_fraction=proxy_fraction,
        )


def winner_confidence_sequence(
    route_id: str,
    win_indicators: Sequence[int | bool | float],
    *,
    threshold: float,
    confidence_level: float = 0.95,
    support_strength: float = 1.0,
    proxy_fraction: float = 0.0,
) -> list[WinnerConfidenceState]:
    states: list[WinnerConfidenceState] = []
    wins = 0.0
    for idx, value in enumerate(win_indicators, start=1):
        wins += 1.0 if bool(value) else 0.0
        states.append(
            WinnerConfidenceState.from_counts(
                route_id,
                wins=wins,
                sample_count=idx,
                threshold=threshold,
                confidence_level=confidence_level,
                support_strength=support_strength,
                proxy_fraction=proxy_fraction,
            )
        )
    return states
