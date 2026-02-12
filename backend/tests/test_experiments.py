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
        span = abs(origin_lat - dest_lat) + 0.05
        duration_s = 3_100.0 + (span * 120.0)
        distance_m = 70_000.0 + (span * 2_000.0)
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


def test_experiment_lifecycle_and_compare_replay(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()

    try:
        with TestClient(app) as client:
            payload = {
                "name": "Morning baseline",
                "description": "Primary scenario compare setup",
                "request": {
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "vehicle_type": "rigid_hgv",
                    "weights": {"time": 1, "money": 1, "co2": 1},
                    "max_alternatives": 4,
                    "pareto_method": "epsilon_constraint",
                    "epsilon": {"duration_s": 8_000, "monetary_cost": 8_000, "emissions_kg": 8_000},
                    "terrain_profile": "rolling",
                },
            }

            create_resp = client.post("/experiments", json=payload)
            assert create_resp.status_code == 200
            created = create_resp.json()
            experiment_id = created["id"]
            assert created["name"] == "Morning baseline"
            assert created["request"]["terrain_profile"] == "rolling"

            list_resp = client.get("/experiments")
            assert list_resp.status_code == 200
            listed = list_resp.json()["experiments"]
            assert any(item["id"] == experiment_id for item in listed)

            get_resp = client.get(f"/experiments/{experiment_id}")
            assert get_resp.status_code == 200
            assert get_resp.json()["description"] == "Primary scenario compare setup"

            update_payload = dict(payload)
            update_payload["name"] = "Morning baseline v2"
            update_payload["description"] = "Updated"
            update_resp = client.put(f"/experiments/{experiment_id}", json=update_payload)
            assert update_resp.status_code == 200
            assert update_resp.json()["name"] == "Morning baseline v2"

            replay_resp = client.post(
                f"/experiments/{experiment_id}/compare",
                json={"overrides": {"max_alternatives": 3}},
            )
            assert replay_resp.status_code == 200
            replay = replay_resp.json()
            run_id = replay["run_id"]
            assert len(replay["results"]) == 3
            assert replay["scenario_manifest_endpoint"] == f"/runs/{run_id}/scenario-manifest"

            manifest_resp = client.get(f"/runs/{run_id}/scenario-manifest")
            assert manifest_resp.status_code == 200
            manifest = manifest_resp.json()
            assert manifest["source"]["experiment_id"] == experiment_id
            assert manifest["source"]["experiment_name"] == "Morning baseline v2"

            delete_resp = client.delete(f"/experiments/{experiment_id}")
            assert delete_resp.status_code == 200
            assert delete_resp.json()["deleted"] is True

            missing_resp = client.get(f"/experiments/{experiment_id}")
            assert missing_resp.status_code == 404
    finally:
        app.dependency_overrides.clear()
