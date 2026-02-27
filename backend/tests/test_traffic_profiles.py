from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import app.calibration_loader as calibration_loader
import app.carbon_model as carbon_model
import app.fuel_energy_model as fuel_energy_model
import app.main as main_module
import app.scenario as scenario_module
from app.departure_profile import DepartureMultiplier
from fastapi.testclient import TestClient

from app.main import app, build_option, osrm_client
from app.models import CostToggles
from app.scenario import ScenarioMode, ScenarioPolicy
from app.routing_graph import GraphCandidateDiagnostics
from app.settings import settings
from app.toll_engine import TollComputation


@pytest.fixture(autouse=True)
def _stub_scenario_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)

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

    def _fake_graph_candidate_routes(
        *,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        max_paths: int | None = None,
        scenario_edge_modifiers: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        _ = (max_paths, scenario_edge_modifiers)
        route = {
            "distance": 40_000.0,
            "duration": 2_400.0,
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [origin_lon, origin_lat],
                    [(origin_lon + destination_lon) / 2.0, (origin_lat + destination_lat) / 2.0],
                    [destination_lon, destination_lat],
                    [destination_lon, destination_lat],
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
        return (
            [route],
            GraphCandidateDiagnostics(
                explored_states=6,
                generated_paths=3,
                emitted_paths=1,
                candidate_budget=3,
            ),
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
    async def _fake_graph_precheck(**_kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "reason_code": "ok", "origin_node_id": "pytest_origin", "destination_node_id": "pytest_destination"}

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _fake_graph_precheck)
    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(main_module, "route_graph_candidate_routes", _fake_graph_candidate_routes)
    calibration_loader.load_scenario_profiles.cache_clear()
    yield
    calibration_loader.load_scenario_profiles.cache_clear()


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
