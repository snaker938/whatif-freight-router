from __future__ import annotations

from .models import TerrainProfile


def gradient_multipliers(profile: TerrainProfile) -> tuple[float, float]:
    """Return deterministic (duration_mult, emissions_mult) for terrain profile."""
    if profile == "rolling":
        return 1.03, 1.05
    if profile == "hilly":
        return 1.08, 1.12
    return 1.0, 1.0
