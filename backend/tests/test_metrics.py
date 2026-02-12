from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.main import app, osrm_client
from app.metrics_store import reset_metrics
from app.route_cache import clear_route_cache
from app.routing_osrm import OSRMError


def _make_route() -> dict[str, Any]:
    return {
        "distance": 12_000.0,
        "duration": 900.0,
        "geometry": {
            "type": "LineString",
            "coordinates": [[-1.0, 52.0], [-0.8, 51.9], [-0.6, 51.8]],
        },
        "legs": [
            {
                "annotation": {
                    "distance": [6_000.0, 6_000.0],
                    "duration": [450.0, 450.0],
                }
            }
        ],
    }


class SuccessOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return [_make_route()]


class FailingOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        raise OSRMError("forced failure")


def test_metrics_endpoint_tracks_successful_core_requests() -> None:
    reset_metrics()
    clear_route_cache()
    app.dependency_overrides[osrm_client] = lambda: SuccessOSRM()
    try:
        with TestClient(app) as client:
            route_payload = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "vehicle_type": "rigid_hgv",
                "scenario_mode": "full_sharing",
                "weights": {"time": 1, "money": 1, "co2": 1},
            }
            pareto_payload = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "vehicle_type": "rigid_hgv",
                "scenario_mode": "full_sharing",
                "max_alternatives": 3,
            }

            assert client.post("/route", json=route_payload).status_code == 200
            assert client.post("/pareto", json=pareto_payload).status_code == 200

            metrics_resp = client.get("/metrics")
            assert metrics_resp.status_code == 200
            metrics = metrics_resp.json()

            assert metrics["total_requests"] == 2
            assert metrics["total_errors"] == 0
            assert "route" in metrics["endpoints"]
            assert "pareto" in metrics["endpoints"]
            assert metrics["endpoints"]["route"]["request_count"] == 1
            assert metrics["endpoints"]["pareto"]["request_count"] == 1
            assert metrics["endpoints"]["route"]["error_count"] == 0
            assert metrics["endpoints"]["pareto"]["error_count"] == 0
    finally:
        app.dependency_overrides.clear()


def test_metrics_endpoint_tracks_handler_errors() -> None:
    reset_metrics()
    clear_route_cache()
    app.dependency_overrides[osrm_client] = lambda: FailingOSRM()
    try:
        with TestClient(app) as client:
            pareto_payload = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "vehicle_type": "rigid_hgv",
                "scenario_mode": "full_sharing",
                "max_alternatives": 3,
            }
            resp = client.post("/pareto", json=pareto_payload)
            assert resp.status_code == 502

            metrics_resp = client.get("/metrics")
            assert metrics_resp.status_code == 200
            metrics = metrics_resp.json()

            assert metrics["total_requests"] == 1
            assert metrics["total_errors"] == 1
            assert metrics["endpoints"]["pareto"]["request_count"] == 1
            assert metrics["endpoints"]["pareto"]["error_count"] == 1
    finally:
        app.dependency_overrides.clear()
