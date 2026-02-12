from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app, osrm_client
from app.settings import settings


class FakeOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "distance": 65_000.0,
                "duration": 3_600.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
                },
                "legs": [
                    {
                        "annotation": {
                            "distance": [32_500.0, 32_500.0],
                            "duration": [1_800.0, 1_800.0],
                        }
                    }
                ],
            }
        ]


def test_departure_optimize_validation_and_deterministic_best(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            invalid_resp = client.post(
                "/departure/optimize",
                json={
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "window_start_utc": "2026-02-12T12:00:00Z",
                    "window_end_utc": "2026-02-12T11:00:00Z",
                    "step_minutes": 60,
                },
            )
            assert invalid_resp.status_code == 422

            valid_resp = client.post(
                "/departure/optimize",
                json={
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "vehicle_type": "rigid_hgv",
                    "scenario_mode": "no_sharing",
                    "weights": {"time": 1, "money": 1, "co2": 1},
                    "window_start_utc": "2026-02-12T02:00:00Z",
                    "window_end_utc": "2026-02-12T08:00:00Z",
                    "step_minutes": 180,
                },
            )
            assert valid_resp.status_code == 200
            payload = valid_resp.json()
            assert payload["evaluated_count"] >= 3
            assert payload["best"] is not None
            assert payload["best"]["departure_time_utc"].startswith("2026-02-12T02:00:00")
            scores = [item["score"] for item in payload["candidates"]]
            assert scores == sorted(scores)
    finally:
        app.dependency_overrides.clear()


def test_departure_optimize_time_window_feasible_and_infeasible(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            feasible_resp = client.post(
                "/departure/optimize",
                json={
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "window_start_utc": "2026-02-12T02:00:00Z",
                    "window_end_utc": "2026-02-12T08:00:00Z",
                    "step_minutes": 180,
                    "time_window": {
                        "latest_arrival_utc": "2026-02-12T03:05:00Z",
                    },
                },
            )
            assert feasible_resp.status_code == 200
            feasible_payload = feasible_resp.json()
            assert feasible_payload["evaluated_count"] >= 1
            assert feasible_payload["best"]["departure_time_utc"].startswith("2026-02-12T02:00:00")

            infeasible_resp = client.post(
                "/departure/optimize",
                json={
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "window_start_utc": "2026-02-12T02:00:00Z",
                    "window_end_utc": "2026-02-12T08:00:00Z",
                    "step_minutes": 180,
                    "time_window": {
                        "earliest_arrival_utc": "2026-02-12T20:00:00Z",
                        "latest_arrival_utc": "2026-02-12T20:10:00Z",
                    },
                },
            )
            assert infeasible_resp.status_code == 422
            assert infeasible_resp.json()["detail"] == "no feasible departures for provided time window"
    finally:
        app.dependency_overrides.clear()
