from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ScenarioMode(str, Enum):
    no_sharing = "no_sharing"
    partial_sharing = "partial_sharing"
    full_sharing = "full_sharing"


class ScenarioPolicy(BaseModel):
    """v0 scenario policy (placeholder).

    The key improvement vs the earlier guide is: scenario_mode now affects outputs through
    one module, so replacing this with a real incident/weather simulator later is clean.
    """

    duration_multiplier: float = Field(..., gt=0)


SCENARIO_POLICIES: dict[ScenarioMode, ScenarioPolicy] = {
    ScenarioMode.no_sharing: ScenarioPolicy(duration_multiplier=1.10),
    ScenarioMode.partial_sharing: ScenarioPolicy(duration_multiplier=1.05),
    ScenarioMode.full_sharing: ScenarioPolicy(duration_multiplier=1.00),
}


def apply_scenario_duration(duration_s: float, mode: ScenarioMode) -> float:
    return duration_s * SCENARIO_POLICIES[mode].duration_multiplier
