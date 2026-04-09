"""Deterministic and probabilistic flip-radius wrappers for REFC scaffolding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class FlipRadiusState:
    route_id: str
    deterministic_local_flip_radius: float = 0.0
    probabilistic_flip_radius: float = 0.0
    challenger_specific_radii: dict[str, float] = field(default_factory=dict)
    evidence_family_radii: dict[str, float] = field(default_factory=dict)
    dominant_fragility_family: str | None = None
    minimum_flip_budget: float | None = None
    support_flag: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
