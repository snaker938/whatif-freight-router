from __future__ import annotations

import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import app.calibration_loader as calibration_loader
import app.carbon_model as carbon_model
import app.fuel_energy_model as fuel_energy_model
import app.main as main_module
import app.scenario as scenario_module
from fastapi.testclient import TestClient

from app import route_cache
from app.main import app, osrm_client
from app.departure_profile import DepartureMultiplier
from app.routing_graph import GraphCandidateDiagnostics
from app.scenario import ScenarioPolicy
from app.toll_engine import TollComputation


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


@pytest.fixture(autouse=True)
def _runtime_stubs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")

    def _policy(*_args: Any, **_kwargs: Any) -> ScenarioPolicy:
        return ScenarioPolicy(
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="pytest",
            version="pytest",
        )

    def _tod(
        departure_time_utc: datetime | None,
        *,
        route_points: list[tuple[float, float]] | None = None,
        road_class_counts: dict[str, int] | None = None,
    ) -> DepartureMultiplier:
        _ = (route_points, road_class_counts)
        hour = (
            int(departure_time_utc.astimezone(UTC).hour)
            if departure_time_utc is not None
            else 12
        )
        multiplier = 1.20 if 7 <= hour <= 10 else 0.90 if 0 <= hour <= 5 else 1.00
        return DepartureMultiplier(
            multiplier=multiplier,
            profile_source="pytest",
            local_time_iso=departure_time_utc.isoformat() if departure_time_utc is not None else None,
            profile_day="weekday",
            profile_key="uk_default.mixed.weekday",
            profile_version="pytest",
            profile_as_of_utc=datetime.now(UTC).isoformat(),
            profile_refreshed_at_utc=datetime.now(UTC).isoformat(),
        )

    monkeypatch.setattr(main_module, "resolve_scenario_profile", _policy)
    monkeypatch.setattr(scenario_module, "resolve_scenario_profile", _policy)
    monkeypatch.setattr(main_module, "time_of_day_multiplier_uk", _tod)
    monkeypatch.setattr(
        main_module,
        "compute_toll_cost",
        lambda **_kwargs: TollComputation(
            contains_toll=False,
            toll_distance_km=0.0,
            toll_cost_gbp=0.0,
            confidence=1.0,
            source="pytest",
            details={"segments_matched": 0, "classified_steps": 0},
        ),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_carbon_price",
        lambda **_kwargs: carbon_model.CarbonPricingContext(
            price_per_kg=0.10,
            source="pytest",
            schedule_year=2026,
            scope_mode="ttw",
            uncertainty_low=0.08,
            uncertainty_high=0.12,
        ),
    )
    monkeypatch.setattr(
        fuel_energy_model,
        "load_fuel_price_snapshot",
        lambda as_of_utc=None: calibration_loader.FuelPriceSnapshot(
            prices_gbp_per_l={"diesel": 1.52, "petrol": 1.58, "lng": 1.05},
            grid_price_gbp_per_kwh=0.28,
            regional_multipliers={"uk_default": 1.0},
            as_of=datetime.now(UTC).isoformat(),
            source="pytest",
            signature="pytest",
            live_diagnostics={},
        ),
    )
    monkeypatch.setattr(
        main_module,
        "_route_stochastic_uncertainty",
        lambda *args, **kwargs: (
            {"duration_p50_s": 0.0, "monetary_p50_gbp": 0.0, "emissions_p50_kg": 0.0},
            {"sample_count": 1},
        ),
    )
    monkeypatch.setattr(
        main_module,
        "load_risk_normalization_reference",
        lambda **_kwargs: calibration_loader.RiskNormalizationReference(
            duration_s_per_km=90.0,
            monetary_gbp_per_km=1.0,
            emissions_kg_per_km=0.5,
            source="pytest",
            version="pytest",
            as_of_utc=datetime.now(UTC).isoformat(),
            corridor_bucket="uk_default",
            day_kind="weekday",
            local_time_slot="h12",
        ),
    )
    monkeypatch.setattr(
        main_module,
        "load_fuel_consumption_calibration",
        lambda: SimpleNamespace(
            source="pytest",
            version="pytest",
            as_of_utc=datetime.now(UTC).isoformat(),
        ),
    )
    def _fake_graph_candidate_routes(
        *,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        max_paths: int | None = None,
        scenario_edge_modifiers: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        _ = (max_paths, scenario_edge_modifiers)
        routes: list[dict[str, Any]] = []
        for idx in range(9):
            lat_shift = idx * 0.001
            routes.append(
                {
                    "distance": 24_000.0 + (idx * 100.0),
                    "duration": 1_000.0 + (idx * 5.0),
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [origin_lon, origin_lat + lat_shift],
                            [(origin_lon + destination_lon) / 2.0, (origin_lat + destination_lat) / 2.0 + lat_shift],
                            [destination_lon, destination_lat + lat_shift],
                            [destination_lon, destination_lat + lat_shift],
                        ],
                    },
                    "_graph_meta": {
                        "road_mix_counts": {
                            "motorway": 2,
                            "trunk": 1,
                            "primary": 1,
                            "secondary": 0,
                            "local": 0,
                        }
                    },
                }
            )
        return (
            routes,
            GraphCandidateDiagnostics(
                explored_states=30,
                generated_paths=12,
                emitted_paths=len(routes),
                candidate_budget=12,
            ),
        )

    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(main_module, "route_graph_candidate_routes", _fake_graph_candidate_routes)
    calibration_loader.load_scenario_profiles.cache_clear()
    yield
    calibration_loader.load_scenario_profiles.cache_clear()


def test_route_cache_hits_and_keying() -> None:
    osrm = CountingOSRM()
    app.dependency_overrides[osrm_client] = lambda: osrm
    route_cache.clear_route_cache()
    try:
        with TestClient(app) as client:
            assert client.delete("/cache").status_code == 200

            first = client.post("/route", json=_payload(carbon_price=0.0))
            assert first.status_code == 200
            first_fetch_count = osrm.calls
            assert first_fetch_count >= 9

            second = client.post("/route", json=_payload(carbon_price=0.0))
            assert second.status_code == 200
            assert osrm.calls == first_fetch_count  # cache hit: no extra fetches

            third = client.post("/route", json=_payload(carbon_price=0.2))
            assert third.status_code == 200
            assert osrm.calls == first_fetch_count * 2  # toggles changed -> cache key changed

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
            first_fetch_count = osrm.calls
            assert first_fetch_count >= 9

            time.sleep(0.02)
            second = client.post("/route", json=_payload(carbon_price=0.0))
            assert second.status_code == 200
            assert osrm.calls == first_fetch_count * 2
    finally:
        route_cache.ROUTE_CACHE._ttl_s = old_ttl
        app.dependency_overrides.clear()
        route_cache.clear_route_cache()
