"""Anytime-valid confidence-sequence wrappers for REFC scaffolding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


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
