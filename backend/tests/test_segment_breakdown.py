from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from app.main import build_option
from app.models import CostToggles
from app.scenario import ScenarioMode


def _route(*, distance_m: float, duration_s: float) -> dict[str, Any]:
    coords = [[-1.8904, 52.4862], [-1.2, 52.0], [-0.1276, 51.5072]]
    return {
        "distance": distance_m,
        "duration": duration_s,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [distance_m * 0.45, distance_m * 0.55],
                    "duration": [duration_s * 0.4, duration_s * 0.6],
                }
            }
        ],
    }


def test_segment_breakdown_payload_shape_and_total_consistency() -> None:
    option = build_option(
        _route(distance_m=36_000.0, duration_s=2_400.0),
        option_id="segment_test",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.PARTIAL_SHARING,
        cost_toggles=CostToggles(
            use_tolls=True,
            toll_cost_per_km=0.2,
            fuel_price_multiplier=1.1,
            carbon_price_per_kg=0.1,
        ),
        departure_time_utc=datetime(2026, 2, 12, 8, 30, tzinfo=UTC),
    )

    segments = option.segment_breakdown
    assert len(segments) == 2
    for idx, segment in enumerate(segments):
        assert segment["segment_index"] == idx
        assert "distance_km" in segment
        assert "duration_s" in segment
        assert "monetary_cost" in segment
        assert "emissions_kg" in segment

    total_distance = sum(float(segment["distance_km"]) for segment in segments)
    total_duration = sum(float(segment["duration_s"]) for segment in segments)
    total_cost = sum(float(segment["monetary_cost"]) for segment in segments)
    total_emissions = sum(float(segment["emissions_kg"]) for segment in segments)

    assert math.isclose(total_distance, option.metrics.distance_km, rel_tol=0.0, abs_tol=0.01)
    assert math.isclose(total_duration, option.metrics.duration_s, rel_tol=0.0, abs_tol=0.1)
    assert math.isclose(total_cost, option.metrics.monetary_cost, rel_tol=0.0, abs_tol=0.1)
    assert math.isclose(total_emissions, option.metrics.emissions_kg, rel_tol=0.0, abs_tol=0.05)
