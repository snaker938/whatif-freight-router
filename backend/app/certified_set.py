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

    @property
    def safe(self) -> bool:
        return bool(self.certified and self.support_flag)

    @property
    def certified_route_ids(self) -> tuple[str, ...]:
        return tuple(self.member_route_ids)

    @property
    def rejected_route_ids(self) -> tuple[str, ...]:
        return tuple(self.excluded_route_ids)

    @classmethod
    def from_confidence_states(
        cls,
        confidence_states: list[Any] | tuple[Any, ...],
        *,
        threshold: float,
        winner_id: str,
        decision_region: Any | None = None,
    ) -> "CertifiedSetState":
        states = list(confidence_states or [])
        winner_route_id = str(winner_id)
        members = [winner_route_id]
        excluded = [
            route_id
            for route_id in (_state_route_id(state) for state in states)
            if route_id and route_id != winner_route_id
        ]
        winner_state = next(
            (state for state in states if _state_route_id(state) == winner_route_id),
            None,
        )
        threshold_value = _coerce_float(threshold)
        winner_score = _state_point_estimate(winner_state)
        decision_certified = bool(getattr(decision_region, "certified", False))
        support_flag = all(bool(getattr(state, "support_flag", True)) for state in states)
        support_flag = support_flag and bool(getattr(decision_region, "support_flag", True))
        certified = bool(
            support_flag
            and decision_certified
            and winner_score >= threshold_value
            and excluded
        )
        exclusion_basis = [
            "confidence_state_threshold",
            "decision_region_certified" if decision_certified else "decision_region_not_certified",
            "winner_threshold_met" if winner_score >= threshold_value else "winner_threshold_not_met",
        ]
        if not support_flag:
            exclusion_basis.append("support_flag_false")
        return cls(
            member_route_ids=members,
            excluded_route_ids=excluded,
            exclusion_basis=exclusion_basis,
            certified=certified,
            threshold=threshold_value,
            support_flag=support_flag,
            set_size=len(members),
            witness={
                "route_id": winner_route_id,
                "winner_point_estimate": winner_score,
                "decision_region_certified": decision_certified,
                "excluded_route_ids": excluded,
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)


def _state_route_id(state: Any) -> str:
    return str(getattr(state, "route_id", "") or "")


def _state_point_estimate(state: Any) -> float:
    if state is None:
        return 0.0
    return _coerce_float(
        getattr(state, "point_estimate", getattr(state, "empirical_win", 0.0))
    )


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
