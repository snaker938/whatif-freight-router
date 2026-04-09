"""Certified-set wrappers for REFC scaffold state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class CertifiedSetState:
    member_route_ids: list[str] = field(default_factory=list)
    excluded_route_ids: list[str] = field(default_factory=list)
    exclusion_basis: list[str] = field(default_factory=list)
    certified: bool = False
    threshold: float = 0.0
    support_flag: bool = True
    set_size: int = 0
    witness: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
