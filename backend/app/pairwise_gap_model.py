"""Pairwise challenger-gap wrappers for REFC scaffold state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
