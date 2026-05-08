"""Pairwise challenger-gap wrappers for REFC scaffold state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PairwiseGapState:
    challenger_id: str
    pairwise_gap_lower_bound: float = 0.0
    pairwise_gap_upper_bound: float = 0.0
    nearest_challenger: bool = False
    challenger_audit_sensitivity: float = 0.0
    challenger_radius: float | None = None
    flip_budget: float | None = None
    support_flag: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)
    sample_count: int = 0
    min_gap: float = 0.0
    mean_gap: float = 0.0
    max_gap: float = 0.0
    positive_share: float = 0.0
    negative_share: float = 0.0
    tie_share: float = 0.0
    challenger_win_share: float = 0.0

    @classmethod
    def from_score_maps(
        cls,
        winner_id: str,
        challenger_id: str,
        score_maps: Sequence[Mapping[str, Any]],
        *,
        support_strength: float = 1.0,
        proxy_fraction: float = 0.0,
        support_flag: bool = True,
    ) -> "PairwiseGapState":
        gaps = [
            _coerce_finite_float(row[winner_id])
            - _coerce_finite_float(row[challenger_id])
            for row in score_maps
            if winner_id in row and challenger_id in row
        ]
        return cls._from_gaps(
            str(winner_id),
            str(challenger_id),
            gaps,
            support_strength=support_strength,
            proxy_fraction=proxy_fraction,
            support_flag=support_flag,
            source="score_maps",
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
        support_strength: float = 1.0,
        proxy_fraction: float = 0.0,
        support_flag: bool = True,
    ) -> "PairwiseGapState":
        gap = _coerce_finite_float(winner_certificate) - _coerce_finite_float(
            challenger_certificate
        )
        return cls._from_gaps(
            str(winner_id),
            str(challenger_id),
            [gap],
            sample_count=max(0, int(sample_count or 0)),
            support_strength=support_strength,
            proxy_fraction=proxy_fraction,
            support_flag=support_flag,
            source="certificate_gap",
            extra_provenance={
                "winner_certificate": _round_float(winner_certificate),
                "challenger_certificate": _round_float(challenger_certificate),
            },
        )

    @classmethod
    def _from_gaps(
        cls,
        winner_id: str,
        challenger_id: str,
        gaps: Sequence[float],
        *,
        sample_count: int | None = None,
        support_strength: float,
        proxy_fraction: float,
        support_flag: bool,
        source: str,
        extra_provenance: Mapping[str, Any] | None = None,
    ) -> "PairwiseGapState":
        finite_gaps = [_coerce_finite_float(gap) for gap in gaps]
        count = max(
            0,
            int(sample_count if sample_count is not None else len(finite_gaps)),
        )
        min_gap = min(finite_gaps) if finite_gaps else 0.0
        max_gap = max(finite_gaps) if finite_gaps else 0.0
        mean_gap = sum(finite_gaps) / float(len(finite_gaps)) if finite_gaps else 0.0
        positive = sum(1 for gap in finite_gaps if gap > _GAP_TIE_TOLERANCE)
        negative = sum(1 for gap in finite_gaps if gap < -_GAP_TIE_TOLERANCE)
        tied = len(finite_gaps) - positive - negative
        share_denominator = float(len(finite_gaps)) if finite_gaps else 0.0
        positive_share = positive / share_denominator if share_denominator else 0.0
        negative_share = negative / share_denominator if share_denominator else 0.0
        tie_share = tied / share_denominator if share_denominator else 0.0
        provenance = {
            "winner_id": winner_id,
            "source": source,
            "support_strength": _round_float(support_strength),
            "proxy_fraction": _round_float(proxy_fraction),
            **dict(extra_provenance or {}),
        }
        return cls(
            challenger_id=challenger_id,
            pairwise_gap_lower_bound=_round_float(min_gap),
            pairwise_gap_upper_bound=_round_float(max_gap),
            challenger_audit_sensitivity=_round_float(
                1.0 - _clamp_unit(support_strength)
            ),
            challenger_radius=_round_float(mean_gap) if mean_gap > 0.0 else None,
            flip_budget=_round_float(mean_gap) if mean_gap > 0.0 else None,
            support_flag=bool(support_flag),
            provenance=provenance,
            sample_count=count,
            min_gap=_round_float(min_gap),
            mean_gap=_round_float(mean_gap),
            max_gap=_round_float(max_gap),
            positive_share=_round_float(positive_share),
            negative_share=_round_float(negative_share),
            tie_share=_round_float(tie_share),
            challenger_win_share=_round_float(negative_share),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)


_GAP_TIE_TOLERANCE = 1e-12


def _coerce_finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _round_float(value: Any) -> float:
    return round(_coerce_finite_float(value), 6)


def _clamp_unit(value: Any) -> float:
    return min(1.0, max(0.0, _coerce_finite_float(value)))
