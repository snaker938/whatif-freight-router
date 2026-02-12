from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

from app import route_cache
from app.main import app, osrm_client


def _make_route(duration_s: float, distance_m: float, lon_offset: float) -> dict[str, Any]:
    return {
        "distance": distance_m,
        "duration": duration_s,
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon_offset, 52.4], [lon_offset + 0.1, 52.0], [lon_offset + 0.3, 51.7]],
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


class CountingOSRM:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        self.calls += 1
        return [_make_route(duration_s=1_000.0, distance_m=24_000.0, lon_offset=-1.3)]


def _payload(*, carbon_price: float = 0.0) -> dict[str, Any]:
    return {
        "origin": {"lat": 52.4862, "lon": -1.8904},
        "destination": {"lat": 51.5072, "lon": -0.1276},
        "vehicle_type": "rigid_hgv",
        "scenario_mode": "full_sharing",
        "weights": {"time": 1, "money": 1, "co2": 1},
        "cost_toggles": {
            "use_tolls": True,
            "fuel_price_multiplier": 1.0,
            "carbon_price_per_kg": carbon_price,
            "toll_cost_per_km": 0.0,
        },
    }


def test_route_cache_hits_and_keying() -> None:
    osrm = CountingOSRM()
    app.dependency_overrides[osrm_client] = lambda: osrm
    route_cache.clear_route_cache()
    try:
        with TestClient(app) as client:
            assert client.delete("/cache").status_code == 200

            first = client.post("/route", json=_payload(carbon_price=0.0))
            assert first.status_code == 200
            assert osrm.calls == 9  # 9 candidate fetch specs on cache miss

            second = client.post("/route", json=_payload(carbon_price=0.0))
            assert second.status_code == 200
            assert osrm.calls == 9  # cache hit: no extra fetches

            third = client.post("/route", json=_payload(carbon_price=0.2))
            assert third.status_code == 200
            assert osrm.calls == 18  # toggles changed -> cache key changed

            stats_resp = client.get("/cache/stats")
            assert stats_resp.status_code == 200
            stats = stats_resp.json()
            assert stats["hits"] >= 1
            assert stats["misses"] >= 2
            assert stats["size"] >= 1
    finally:
        app.dependency_overrides.clear()
        route_cache.clear_route_cache()


def test_route_cache_ttl_expiry_causes_recompute() -> None:
    osrm = CountingOSRM()
    app.dependency_overrides[osrm_client] = lambda: osrm
    route_cache.clear_route_cache()

    old_ttl = route_cache.ROUTE_CACHE._ttl_s
    route_cache.ROUTE_CACHE._ttl_s = 0
    try:
        with TestClient(app) as client:
            first = client.post("/route", json=_payload(carbon_price=0.0))
            assert first.status_code == 200
            assert osrm.calls == 9

            time.sleep(0.02)
            second = client.post("/route", json=_payload(carbon_price=0.0))
            assert second.status_code == 200
            assert osrm.calls == 18
    finally:
        route_cache.ROUTE_CACHE._ttl_s = old_ttl
        app.dependency_overrides.clear()
        route_cache.clear_route_cache()
