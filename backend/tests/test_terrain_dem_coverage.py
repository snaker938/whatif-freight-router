from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode
from app.settings import settings


def _uk_route() -> dict[str, Any]:
    coords = [[-1.8904, 52.4862], [-1.2, 52.0], [-0.1276, 51.5072]]
    return {
        "distance": 42_000.0,
        "duration": 2_600.0,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [14_000.0, 14_000.0, 14_000.0],
                    "duration": [860.0, 860.0, 880.0],
                }
            }
        ],
    }


def test_terrain_summary_reports_dem_coverage_for_uk_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "terrain_allow_synthetic_grid", True)
    option = build_option(
        _uk_route(),
        option_id="terrain_cov",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="hilly",
        departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
    )
    assert option.terrain_summary is not None
    assert option.terrain_summary.source == "dem_real"
    assert option.terrain_summary.coverage_ratio >= settings.terrain_dem_coverage_min_uk
    assert option.terrain_summary.confidence > 0.0
    assert option.terrain_summary.version
