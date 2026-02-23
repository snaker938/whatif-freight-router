from __future__ import annotations

from pathlib import Path
from typing import Any

import app.main as main_module
from fastapi.testclient import TestClient

from app.main import CandidateDiagnostics, TerrainDiagnostics
from app.main import app, osrm_client
from app.models import GeoJSONLineString, RouteMetrics, RouteOption, ScenarioSummary
from app.scenario import ScenarioMode
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

    async def _fake_collect_route_options_with_diagnostics(**kwargs: Any) -> tuple[
        list[RouteOption],
        list[str],
        int,
        TerrainDiagnostics,
        CandidateDiagnostics,
    ]:
        origin = kwargs["origin"]
        destination = kwargs["destination"]
        option_prefix = str(kwargs.get("option_prefix", "leg"))
        scenario_mode = kwargs.get("scenario_mode", ScenarioMode.NO_SHARING)

        coords = [
            (float(origin.lon), float(origin.lat)),
            (
                float((origin.lon + destination.lon) / 2.0),
                float((origin.lat + destination.lat) / 2.0),
            ),
            (float(destination.lon), float(destination.lat)),
        ]
        distance_km = 120.0
        duration_s = 3600.0
        option = RouteOption(
            id=f"{option_prefix}_1",
            geometry=GeoJSONLineString(type="LineString", coordinates=coords),
            metrics=RouteMetrics(
                distance_km=distance_km,
                duration_s=duration_s,
                monetary_cost=distance_km * 1.8,
                emissions_kg=distance_km * 0.75,
                avg_speed_kmh=distance_km / (duration_s / 3600.0),
            ),
            scenario_summary=ScenarioSummary(
                mode=scenario_mode,
                duration_multiplier=1.0,
                incident_rate_multiplier=1.0,
                incident_delay_multiplier=1.0,
                fuel_consumption_multiplier=1.0,
                emissions_multiplier=1.0,
                stochastic_sigma_multiplier=1.0,
                source="pytest",
                version="pytest",
            ),
        )
        return (
            [option],
            [],
            1,
            TerrainDiagnostics(),
            CandidateDiagnostics(
                raw_count=1,
                deduped_count=1,
                graph_explored_states=1,
                graph_generated_paths=1,
                graph_emitted_paths=1,
                candidate_budget=1,
            ),
        )

    monkeypatch.setattr(
        main_module,
        "_collect_route_options_with_diagnostics",
        _fake_collect_route_options_with_diagnostics,
    )
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
