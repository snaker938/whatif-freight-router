from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from .confidence_sequences import confidence_radius, effective_sample_count
from .world_policies import clamp01


@dataclass(frozen=True)
class PairwiseGapState:
    winner_id: str
    challenger_id: str
    sample_count: int
    effective_sample_count: float
    confidence_level: float
    mean_gap: float
    lower_gap: float
    upper_gap: float
    min_gap: float
    max_gap: float
    positive_share: float
    negative_share: float
    tie_share: float
    challenger_win_share: float
    flip_probability_upper: float
    support_strength: float
    proxy_fraction: float

    @property
    def pairwise_safe(self) -> bool:
        return self.lower_gap > 0.0 and self.flip_probability_upper < 0.5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_gap_samples(
        cls,
        winner_id: str,
        challenger_id: str,
        gap_samples: Sequence[float],
        *,
        confidence_level: float = 0.95,
        support_strength: float = 1.0,
        proxy_fraction: float = 0.0,
    ) -> "PairwiseGapState":
        samples = [float(value) for value in gap_samples]
        if not samples:
            samples = [0.0]
        sample_count = len(samples)
        mean_gap = sum(samples) / float(sample_count)
        min_gap = min(samples)
        max_gap = max(samples)
        gap_scale = max(abs(min_gap), abs(max_gap), 1e-6)
        radius = gap_scale * confidence_radius(
            sample_count,
            confidence_level=confidence_level,
            support_strength=support_strength,
            proxy_fraction=proxy_fraction,
        )
        positive_share = sum(1 for gap in samples if gap > 0.0) / float(sample_count)
        negative_share = sum(1 for gap in samples if gap < 0.0) / float(sample_count)
        tie_share = max(0.0, 1.0 - positive_share - negative_share)
        flip_radius = confidence_radius(
            sample_count,
            confidence_level=confidence_level,
            support_strength=support_strength,
            proxy_fraction=proxy_fraction,
        )
        flip_probability_upper = min(1.0, negative_share + flip_radius)
        return cls(
            winner_id=str(winner_id),
            challenger_id=str(challenger_id),
            sample_count=sample_count,
            effective_sample_count=round(
                effective_sample_count(
                    sample_count,
                    support_strength=support_strength,
                    proxy_fraction=proxy_fraction,
                ),
                6,
            ),
            confidence_level=round(clamp01(confidence_level, 0.95), 6),
            mean_gap=round(mean_gap, 6),
            lower_gap=round(mean_gap - radius, 6),
            upper_gap=round(mean_gap + radius, 6),
            min_gap=round(min_gap, 6),
            max_gap=round(max_gap, 6),
            positive_share=round(positive_share, 6),
            negative_share=round(negative_share, 6),
            tie_share=round(tie_share, 6),
            challenger_win_share=round(negative_share, 6),
            flip_probability_upper=round(flip_probability_upper, 6),
            support_strength=round(clamp01(support_strength, 1.0), 6),
            proxy_fraction=round(clamp01(proxy_fraction), 6),
        )

    @classmethod
    def from_score_maps(
        cls,
        winner_id: str,
        challenger_id: str,
        score_maps: Sequence[Mapping[str, float]],
        *,
        confidence_level: float = 0.95,
        support_strength: float = 1.0,
        proxy_fraction: float = 0.0,
    ) -> "PairwiseGapState":
        gaps: list[float] = []
        for score_map in score_maps:
            if not isinstance(score_map, Mapping):
                continue
            winner_score = score_map.get(winner_id)
            challenger_score = score_map.get(challenger_id)
            try:
                winner_value = float(winner_score)
                challenger_value = float(challenger_score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(winner_value) or not math.isfinite(challenger_value):
                continue
            gaps.append(challenger_value - winner_value)
        return cls.from_gap_samples(
            winner_id,
            challenger_id,
            gaps,
            confidence_level=confidence_level,
            support_strength=support_strength,
            proxy_fraction=proxy_fraction,
        )

    @classmethod
    def from_certificate_gap(
        cls,
        winner_id: str,
        challenger_id: str,
        *,
        winner_certificate: float,
        challenger_certificate: float,
        sample_count: int,
        confidence_level: float = 0.95,
        support_strength: float = 1.0,
        proxy_fraction: float = 0.0,
    ) -> "PairwiseGapState":
        gap = clamp01(winner_certificate) - clamp01(challenger_certificate)
        gap_scale = max(abs(gap), 1.0 / float(max(1, int(sample_count))), 1e-6)
        radius = gap_scale * confidence_radius(
            sample_count,
            confidence_level=confidence_level,
            support_strength=support_strength,
            proxy_fraction=proxy_fraction,
        )
        negative_share = 1.0 if gap < 0.0 else 0.0
        positive_share = 1.0 if gap > 0.0 else 0.0
        tie_share = 1.0 if gap == 0.0 else 0.0
        return cls(
            winner_id=str(winner_id),
            challenger_id=str(challenger_id),
            sample_count=max(1, int(sample_count)),
            effective_sample_count=round(
                effective_sample_count(
                    sample_count,
                    support_strength=support_strength,
                    proxy_fraction=proxy_fraction,
                ),
                6,
            ),
            confidence_level=round(clamp01(confidence_level, 0.95), 6),
            mean_gap=round(gap, 6),
            lower_gap=round(gap - radius, 6),
            upper_gap=round(gap + radius, 6),
            min_gap=round(gap, 6),
            max_gap=round(gap, 6),
            positive_share=round(positive_share, 6),
            negative_share=round(negative_share, 6),
            tie_share=round(tie_share, 6),
            challenger_win_share=round(negative_share, 6),
            flip_probability_upper=round(
                min(
                    1.0,
                    negative_share
                    + confidence_radius(
                        sample_count,
                        confidence_level=confidence_level,
                        support_strength=support_strength,
                        proxy_fraction=proxy_fraction,
                    ),
                ),
                6,
            ),
            support_strength=round(clamp01(support_strength, 1.0), 6),
            proxy_fraction=round(clamp01(proxy_fraction), 6),
        )
