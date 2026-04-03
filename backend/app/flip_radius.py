from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .pairwise_gap_model import PairwiseGapState
from .world_policies import clamp01


@dataclass(frozen=True)
class FlipRadiusState:
    winner_id: str
    challenger_id: str
    objective_scale: float
    absolute_radius: float
    normalized_radius: float
    support_adjusted_radius: float
    proxy_adjusted_radius: float
    fragile: bool
    recommended_action: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_pairwise_gap(
        cls,
        pairwise_gap: PairwiseGapState,
        *,
        objective_scale: float = 1.0,
        support_strength: float | None = None,
        proxy_fraction: float | None = None,
    ) -> "FlipRadiusState":
        scale = max(abs(float(objective_scale)), 1e-9)
        support = clamp01(
            pairwise_gap.support_strength if support_strength is None else support_strength,
            1.0,
        )
        proxy = clamp01(
            pairwise_gap.proxy_fraction if proxy_fraction is None else proxy_fraction
        )
        absolute_radius = max(0.0, float(pairwise_gap.lower_gap))
        normalized_radius = absolute_radius / scale
        support_adjusted_radius = normalized_radius * max(0.25, support)
        proxy_adjusted_radius = support_adjusted_radius * max(0.25, 1.0 - (0.75 * proxy))
        fragile = bool(
            proxy_adjusted_radius <= 0.02
            or pairwise_gap.challenger_win_share > 0.25
            or pairwise_gap.lower_gap <= 0.0
        )
        recommended_action = "refresh_evidence" if fragile else "hold"
        return cls(
            winner_id=pairwise_gap.winner_id,
            challenger_id=pairwise_gap.challenger_id,
            objective_scale=round(scale, 6),
            absolute_radius=round(absolute_radius, 6),
            normalized_radius=round(normalized_radius, 6),
            support_adjusted_radius=round(support_adjusted_radius, 6),
            proxy_adjusted_radius=round(proxy_adjusted_radius, 6),
            fragile=fragile,
            recommended_action=recommended_action,
        )
