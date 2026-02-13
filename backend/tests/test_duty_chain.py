from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app, osrm_client
from app.settings import settings


class FakeOSRM:
    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:
        origin_lat = float(kwargs["origin_lat"])
        dest_lat = float(kwargs["dest_lat"])
        span = abs(origin_lat - dest_lat) + 0.1
        distance_m = 60_000.0 + (span * 1_500.0)
        duration_s = 3_000.0 + (span * 180.0)
        return [
            {
                "distance": distance_m,
                "duration": duration_s,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
                },
                "legs": [
                    {
                        "annotation": {
                            "distance": [distance_m / 2.0, distance_m / 2.0],
                            "duration": [duration_s / 2.0, duration_s / 2.0],
                        }
                    }
                ],
            }
        ]


def test_duty_chain_returns_leg_results_and_aggregate_totals(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            payload = {
                "stops": [
                    {"lat": 52.4862, "lon": -1.8904, "label": "Birmingham"},
                    {"lat": 52.2053, "lon": 0.1218, "label": "Cambridge"},
                    {"lat": 51.5072, "lon": -0.1276, "label": "London"},
                ],
                "vehicle_type": "rigid_hgv",
                "scenario_mode": "no_sharing",
                "max_alternatives": 3,
            }
            resp = client.post("/duty/chain", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["leg_count"] == 2
            assert data["successful_leg_count"] == 2
            assert len(data["legs"]) == 2

            selected_legs = [leg["selected"] for leg in data["legs"] if leg["selected"] is not None]
            assert len(selected_legs) == 2
            distance_sum = sum(leg["metrics"]["distance_km"] for leg in selected_legs)
            duration_sum = sum(leg["metrics"]["duration_s"] for leg in selected_legs)
            cost_sum = sum(leg["metrics"]["monetary_cost"] for leg in selected_legs)
            emissions_sum = sum(leg["metrics"]["emissions_kg"] for leg in selected_legs)

            totals = data["total_metrics"]
            assert abs(totals["distance_km"] - distance_sum) <= 0.01
            assert abs(totals["duration_s"] - duration_sum) <= 0.05
            assert abs(totals["monetary_cost"] - cost_sum) <= 0.05
            assert abs(totals["emissions_kg"] - emissions_sum) <= 0.01
    finally:
        app.dependency_overrides.clear()


def test_duty_chain_requires_at_least_two_stops(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/duty/chain",
                json={"stops": [{"lat": 52.4862, "lon": -1.8904}]},
            )
            assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()

