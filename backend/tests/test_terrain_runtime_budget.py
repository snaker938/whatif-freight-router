from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pytest

from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode
from app.settings import settings


def _route() -> dict[str, Any]:
    coords = [[-4.2, 57.5], [-3.0, 56.3], [-1.9, 54.5], [-0.1, 51.5]]
    return {
        "distance": 180_000.0,
        "duration": 11_000.0,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [60_000.0, 60_000.0, 60_000.0],
                    "duration": [3_666.0, 3_667.0, 3_667.0],
                }
            }
        ],
    }


def test_terrain_path_runtime_p95_under_two_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "terrain_allow_synthetic_grid", True)
    timings_ms: list[float] = []
    route = _route()
    for idx in range(24):
        t0 = time.perf_counter()
        build_option(
            route,
            option_id=f"terrain_runtime_{idx}",
            vehicle_type="rigid_hgv",
            scenario_mode=ScenarioMode.NO_SHARING,
            cost_toggles=CostToggles(),
            terrain_profile="hilly",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
        )
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    sorted_timings = sorted(timings_ms)
    p95_idx = max(0, min(len(sorted_timings) - 1, int(0.95 * len(sorted_timings)) - 1))
    assert sorted_timings[p95_idx] < 2000.0
