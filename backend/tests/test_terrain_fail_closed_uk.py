from __future__ import annotations

import math

import pytest

import app.terrain_dem as terrain_dem
from app.settings import settings
from app.terrain_dem import TerrainCoverageError, estimate_terrain_summary


def test_uk_fail_closed_raises_when_dem_coverage_too_low(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "terrain_dem_fail_closed_uk", True)
    monkeypatch.setattr(settings, "terrain_dem_coverage_min_uk", 0.98)

    terrain_dem._cached_profile.cache_clear()
    monkeypatch.setattr(terrain_dem, "terrain_runtime_status", lambda: (True, "ok"))
    monkeypatch.setattr(terrain_dem, "sample_elevation_m", lambda lat, lon: (math.nan, False, "test_missing"))

    with pytest.raises(TerrainCoverageError) as exc_info:
        estimate_terrain_summary(
            coordinates_lon_lat=[(-1.7, 55.0), (-1.2, 54.0), (-0.2, 51.6)],
            terrain_profile="hilly",
            avg_speed_kmh=58.0,
            distance_km=40.0,
            vehicle_type="rigid_hgv",
        )
    assert exc_info.value.reason_code == "terrain_dem_coverage_insufficient"
