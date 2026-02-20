from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode
from app.settings import settings


def _route() -> dict[str, Any]:
    coords = [[-3.2, 57.0], [-2.4, 56.2], [-1.6, 54.9], [-0.1, 51.5]]
    return {
        "distance": 120_000.0,
        "duration": 7_400.0,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [30_000.0, 30_000.0, 30_000.0, 30_000.0],
                    "duration": [1_850.0, 1_850.0, 1_850.0, 1_850.0],
                }
            }
        ],
    }


def test_hillier_profile_increases_duration_and_emissions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "terrain_allow_synthetic_grid", True)
    flat = build_option(
        _route(),
        option_id="terrain_uplift_flat",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        departure_time_utc=datetime(2026, 2, 18, 3, 30, tzinfo=UTC),
    )
    hilly = build_option(
        _route(),
        option_id="terrain_uplift_hilly",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="hilly",
        departure_time_utc=datetime(2026, 2, 18, 3, 30, tzinfo=UTC),
    )
    assert hilly.metrics.duration_s >= flat.metrics.duration_s
    assert hilly.metrics.emissions_kg >= flat.metrics.emissions_kg
    assert hilly.terrain_summary is not None
    assert hilly.terrain_summary.source == "dem_real"
