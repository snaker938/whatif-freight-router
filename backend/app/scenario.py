from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ScenarioMode(str, Enum):
    # Canonical (recommended) enum member names (what your models.py is trying to use)
    NO_SHARING = "no_sharing"
    PARTIAL_SHARING = "partial_sharing"
    FULL_SHARING = "full_sharing"

    # Backwards-compatible aliases (older code used lowercase member names)
    no_sharing = NO_SHARING
    partial_sharing = PARTIAL_SHARING
    full_sharing = FULL_SHARING


class ScenarioPolicy(BaseModel):
    """v0 scenario policy (placeholder).

    The key improvement vs the earlier guide is: scenario_mode now affects outputs through
    one module, so replacing this with a real incident/weather simulator later is clean.
    """

    duration_multiplier: float = Field(..., gt=0)


SCENARIO_POLICIES: dict[ScenarioMode, ScenarioPolicy] = {
    ScenarioMode.NO_SHARING: ScenarioPolicy(duration_multiplier=1.10),
    ScenarioMode.PARTIAL_SHARING: ScenarioPolicy(duration_multiplier=1.05),
    ScenarioMode.FULL_SHARING: ScenarioPolicy(duration_multiplier=1.00),
}


def scenario_duration_multiplier(mode: ScenarioMode) -> float:
    return SCENARIO_POLICIES[mode].duration_multiplier


def apply_scenario_duration(duration_s: float, mode: ScenarioMode) -> float:
    return duration_s * scenario_duration_multiplier(mode)
