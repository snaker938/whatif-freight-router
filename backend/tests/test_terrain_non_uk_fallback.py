from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode
from app.terrain_dem import TerrainCoverageError


def _non_uk_route() -> dict[str, Any]:
    coords = [[-122.4194, 37.7749], [-121.8863, 37.3382], [-118.2437, 34.0522]]
    return {
        "distance": 610_000.0,
        "duration": 22_000.0,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [305_000.0, 305_000.0],
                    "duration": [11_000.0, 11_000.0],
                }
            }
        ],
    }


def test_non_uk_route_is_rejected_as_unsupported_region() -> None:
    with pytest.raises(TerrainCoverageError) as exc_info:
        build_option(
            _non_uk_route(),
            option_id="terrain_non_uk",
            vehicle_type="rigid_hgv",
            scenario_mode=ScenarioMode.NO_SHARING,
            cost_toggles=CostToggles(),
            terrain_profile="hilly",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
        )
    assert exc_info.value.reason_code == "terrain_region_unsupported"
