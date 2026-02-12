from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app, osrm_client
from app.settings import settings


class MultiRouteOSRM:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls += 1
        c = float(self.calls)
        base_duration = 3_200.0 + (c * 8.0)
        routes: list[dict[str, Any]] = []
        for idx, factor in enumerate((0.92, 1.0, 1.14)):
            duration_s = base_duration * factor
            distance_m = 68_000.0 + (idx * 4_000.0) + (c * 40.0)
            lat_shift = 0.005 * idx + (0.0001 * c)
            routes.append(
                {
                    "distance": distance_m,
                    "duration": duration_s,
                    "contains_toll": idx == 0,
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [-1.8904, 52.4862 + lat_shift],
                            [-1.2, 52.0 + lat_shift],
                            [-0.1276, 51.5072 + lat_shift],
                        ],
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
            )
        return routes


def test_full_explainability_and_compare_flow(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    fake = MultiRouteOSRM()
    app.dependency_overrides[osrm_client] = lambda: fake

    try:
        with TestClient(app) as client:
            pareto_payload = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "vehicle_type": "rigid_hgv",
                "scenario_mode": "no_sharing",
                "max_alternatives": 5,
                "pareto_method": "epsilon_constraint",
                "epsilon": {"duration_s": 6000, "monetary_cost": 5000, "emissions_kg": 5000},
                "departure_time_utc": "2026-02-12T08:45:00Z",
            }
            pareto_resp = client.post("/pareto", json=pareto_payload)
            assert pareto_resp.status_code == 200
            routes = pareto_resp.json()["routes"]
            assert routes
            assert all(route["metrics"]["duration_s"] <= 6000 for route in routes)
            assert sum(1 for route in routes if route["is_knee"]) == 1
            for route in routes:
                assert route["knee_score"] is not None
                assert len(route["eta_timeline"]) == 3
                assert len(route["eta_explanations"]) >= 2
                assert len(route["segment_breakdown"]) >= 1

            # Backward compatibility: old request shape without new additive fields still works.
            old_route_payload = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "vehicle_type": "rigid_hgv",
                "scenario_mode": "full_sharing",
                "weights": {"time": 1, "money": 1, "co2": 1},
            }
            old_route_resp = client.post("/route", json=old_route_payload)
            assert old_route_resp.status_code == 200
            assert "selected" in old_route_resp.json()

            compare_payload = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "vehicle_type": "rigid_hgv",
                "weights": {"time": 1, "money": 1, "co2": 1},
                "max_alternatives": 4,
                "pareto_method": "epsilon_constraint",
                "epsilon": {"duration_s": 7000, "monetary_cost": 6000, "emissions_kg": 6000},
                "departure_time_utc": "2026-02-12T08:45:00Z",
            }
            compare_resp = client.post("/scenario/compare", json=compare_payload)
            assert compare_resp.status_code == 200
            compare_data = compare_resp.json()
            run_id = compare_data["run_id"]
            assert len(compare_data["results"]) == 3
            assert compare_data["scenario_manifest_endpoint"] == f"/runs/{run_id}/scenario-manifest"

            scenario_manifest = client.get(f"/runs/{run_id}/scenario-manifest")
            assert scenario_manifest.status_code == 200
            manifest_payload = scenario_manifest.json()
            signature_value = manifest_payload["signature"]["signature"]

            unsigned_payload = dict(manifest_payload)
            unsigned_payload.pop("signature", None)
            verify_resp = client.post(
                "/verify/signature",
                json={"payload": unsigned_payload, "signature": signature_value},
            )
            assert verify_resp.status_code == 200
            assert verify_resp.json()["valid"] is True
    finally:
        app.dependency_overrides.clear()

