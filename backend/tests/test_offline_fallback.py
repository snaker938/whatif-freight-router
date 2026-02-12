from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.fallback_store import clear_route_snapshots
from app.main import app, osrm_client
from app.route_cache import clear_route_cache
from app.routing_osrm import OSRMError
from app.settings import settings


class SuccessOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "distance": 75_000.0,
                "duration": 3_900.0,
                "contains_toll": True,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
                },
                "legs": [
                    {
                        "annotation": {
                            "distance": [37_500.0, 37_500.0],
                            "duration": [1_950.0, 1_950.0],
                        }
                    }
                ],
            }
        ]


class FailingOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        raise OSRMError("simulated outage")


def test_batch_offline_fallback_uses_snapshot(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    monkeypatch.setattr(settings, "offline_fallback_enabled", True)

    clear_route_cache()
    clear_route_snapshots()

    payload = {
        "pairs": [
            {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
            }
        ],
        "vehicle_type": "rigid_hgv",
        "scenario_mode": "no_sharing",
        "max_alternatives": 3,
    }

    try:
        with TestClient(app) as client:
            app.dependency_overrides[osrm_client] = lambda: SuccessOSRM()
            first = client.post("/batch/pareto", json=payload)
            assert first.status_code == 200
            first_result = first.json()["results"][0]
            assert first_result["fallback_used"] is False

            clear_route_cache()
            app.dependency_overrides[osrm_client] = lambda: FailingOSRM()
            second = client.post("/batch/pareto", json=payload)
            assert second.status_code == 200
            second_data = second.json()
            second_result = second_data["results"][0]
            assert second_result["fallback_used"] is True
            assert second_result["error"] is None
            assert len(second_result["routes"]) >= 1

            run_id = second_data["run_id"]
            manifest = client.get(f"/runs/{run_id}/manifest").json()
            assert manifest["execution"]["fallback_used"] is True
            assert manifest["execution"]["fallback_count"] == 1

            metadata = client.get(f"/runs/{run_id}/artifacts/metadata.json").json()
            assert metadata["fallback_used"] is True
            assert metadata["fallback_count"] == 1
    finally:
        app.dependency_overrides.clear()
