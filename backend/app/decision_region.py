"""Certificate-boundary wrappers for REFC decision-region scaffolding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class DecisionRegionState:
    route_id: str
    nearest_certificate_boundary: str | None = None
    active_challenger_id: str | None = None
    dominant_evidence_family: str | None = None
    most_fragile_preference_direction: str | None = None
    minimum_joint_perturbation: float | None = None
    nearest_threat_axis: str | None = None
    support_flag: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
