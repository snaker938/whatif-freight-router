"""Witness objects for terminal REFC decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class CertificateWitness:
    route_id: str
    active_challenger_ids: list[str] = field(default_factory=list)
    active_evidence_families: list[str] = field(default_factory=list)
    active_preference_constraints: list[str] = field(default_factory=list)
    support_conditions: list[str] = field(default_factory=list)
    action_steps: list[str] = field(default_factory=list)
    witness_sparsity: float | None = None
    witness_size: int = 0
    support_flag: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
