from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient

from app.main import app, build_option, osrm_client
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
                    "distance": [distance_m / 2.0, distance_m / 2.0],
                    "duration": [duration_s / 2.0, duration_s / 2.0],
                }
            }
        ],
    }


class FakeOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return [_route(distance_m=40_000.0, duration_s=2_400.0)]


def test_time_of_day_profile_increases_peak_eta() -> None:
    route = _route(distance_m=45_000.0, duration_s=2_700.0)

    off_peak = build_option(
        route,
        option_id="off_peak",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
        departure_time_utc=datetime(2026, 2, 12, 3, 30, tzinfo=UTC),
    )
    peak = build_option(
        route,
        option_id="peak",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.FULL_SHARING,
        cost_toggles=CostToggles(),
        departure_time_utc=datetime(2026, 2, 12, 8, 30, tzinfo=UTC),
    )

    assert peak.metrics.duration_s > off_peak.metrics.duration_s
    assert any("Time-of-day profile" in msg for msg in peak.eta_explanations)

    stages = [entry["stage"] for entry in peak.eta_timeline]
    assert stages == ["baseline", "time_of_day", "scenario", "gradient"]
    assert float(peak.eta_timeline[-1]["duration_s"]) == peak.metrics.duration_s


def test_hilly_terrain_profile_increases_duration_and_emissions() -> None:
    route = _route(distance_m=45_000.0, duration_s=2_700.0)

    flat = build_option(
        route,
        option_id="flat",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        departure_time_utc=datetime(2026, 2, 12, 3, 30, tzinfo=UTC),
    )
    hilly = build_option(
        route,
        option_id="hilly",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="hilly",
        departure_time_utc=datetime(2026, 2, 12, 3, 30, tzinfo=UTC),
    )

    assert hilly.metrics.duration_s > flat.metrics.duration_s
    assert hilly.metrics.emissions_kg > flat.metrics.emissions_kg
    assert any("Terrain profile 'hilly'" in msg for msg in hilly.eta_explanations)


def test_route_endpoint_returns_eta_explainability_fields() -> None:
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            payload = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "vehicle_type": "rigid_hgv",
                "scenario_mode": "partial_sharing",
                "departure_time_utc": "2026-02-12T08:15:00Z",
                "weights": {"time": 1, "money": 1, "co2": 1},
            }
            resp = client.post("/route", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            selected = data["selected"]
            assert len(selected["eta_explanations"]) >= 2
            assert len(selected["eta_timeline"]) == 4
            assert selected["eta_timeline"][0]["stage"] == "baseline"
    finally:
        app.dependency_overrides.clear()
