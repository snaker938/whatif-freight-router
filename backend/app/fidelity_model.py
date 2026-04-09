from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FidelityClass = Literal["proxy", "audit"]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed != parsed:
        return float(default)
    return float(parsed)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _as_float(value, default)))


@dataclass(frozen=True)
class ActionValueEstimate:
    schema_version: str = "action-value-estimate-v1"
    action_id: str = ""
    fidelity_class: FidelityClass = "proxy"
    expected_gain: float = 0.0
    expected_cost: float = 0.0
    expected_net_gain: float = 0.0
    gain_per_cost: float = 0.0
    confidence: float = 0.0
    support_weight: float = 0.0
    source: str = "conservative"
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_action_value_estimate(
    *,
    action_id: str,
    expected_gain: float,
    expected_cost: float,
    confidence: float = 0.0,
    support_weight: float = 0.0,
    fidelity_class: FidelityClass = "proxy",
    source: str = "conservative",
    provenance: dict[str, Any] | None = None,
) -> ActionValueEstimate:
    cost = max(0.0, _as_float(expected_cost))
    gain = _as_float(expected_gain)
    net_gain = gain - cost
    gain_per_cost = gain / cost if cost > 0.0 else 0.0
    return ActionValueEstimate(
        action_id=str(action_id or ""),
        fidelity_class=fidelity_class,
        expected_gain=gain,
        expected_cost=cost,
        expected_net_gain=net_gain,
        gain_per_cost=gain_per_cost,
        confidence=_clamp01(confidence),
        support_weight=_clamp01(support_weight),
        source=str(source or "conservative"),
        provenance=dict(provenance or {}),
    )
