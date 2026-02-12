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
                "distance": 50_000.0,
                "duration": 3_000.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
                },
                "legs": [
                    {
                        "annotation": {
                            "distance": [25_000.0, 25_000.0],
                            "duration": [1_500.0, 1_500.0],
                        }
                    }
                ],
            }
        ]


def test_rbac_enforcement_matrix(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    monkeypatch.setattr(settings, "rbac_enabled", True)
    monkeypatch.setattr(settings, "rbac_user_token", "user-token")
    monkeypatch.setattr(settings, "rbac_admin_token", "admin-token")
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()

    try:
        with TestClient(app) as client:
            route_payload = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "weights": {"time": 1, "money": 1, "co2": 1},
            }

            missing = client.post("/route", json=route_payload)
            assert missing.status_code == 401

            invalid = client.post(
                "/route",
                json=route_payload,
                headers={"x-api-token": "bad-token"},
            )
            assert invalid.status_code == 401

            allowed_user = client.post(
                "/route",
                json=route_payload,
                headers={"x-api-token": "user-token"},
            )
            assert allowed_user.status_code == 200

            vehicle_payload = {
                "id": "custom_test",
                "label": "Custom test",
                "mass_tonnes": 10.0,
                "emission_factor_kg_per_tkm": 0.1,
                "cost_per_km": 0.6,
                "cost_per_hour": 25.0,
                "idle_emissions_kg_per_hour": 3.0,
            }
            forbidden_user = client.post(
                "/vehicles/custom",
                json=vehicle_payload,
                headers={"x-api-token": "user-token"},
            )
            assert forbidden_user.status_code == 403

            allowed_admin = client.post(
                "/vehicles/custom",
                json=vehicle_payload,
                headers={"x-api-token": "admin-token"},
            )
            assert allowed_admin.status_code == 200
    finally:
        app.dependency_overrides.clear()
