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

    async def _fake_collect_route_options_with_diagnostics(**kwargs: Any) -> tuple[
        list[RouteOption],
        list[str],
        int,
        TerrainDiagnostics,
        CandidateDiagnostics,
    ]:
        origin = kwargs["origin"]
        destination = kwargs["destination"]
        option_prefix = str(kwargs.get("option_prefix", "route"))
        scenario_mode = kwargs.get("scenario_mode", ScenarioMode.NO_SHARING)
        weather_cfg = kwargs.get("weather")
        weather_enabled = bool(getattr(weather_cfg, "enabled", False))

        base_stages: list[dict[str, float | str]] = [
            {"stage": "baseline", "duration_s": 3000.0, "delta_s": 0.0},
            {"stage": "time_of_day", "duration_s": 3150.0, "delta_s": 150.0},
            {"stage": "scenario", "duration_s": 3300.0, "delta_s": 150.0},
        ]
        if weather_enabled:
            base_stages.append({"stage": "weather", "duration_s": 3420.0, "delta_s": 120.0})
        base_stages.append(
            {
                "stage": "gradient",
                "duration_s": 3520.0 if weather_enabled else 3400.0,
                "delta_s": 100.0,
            }
        )

        options: list[RouteOption] = []
        for idx, scale in enumerate((1.0, 1.1, 1.2), start=1):
            coords = [
                (float(origin.lon), float(origin.lat)),
                (
                    float((origin.lon + destination.lon) / 2.0),
                    float((origin.lat + destination.lat) / 2.0),
                ),
                (float(destination.lon), float(destination.lat)),
            ]
            duration_s = 3500.0 * scale
            distance_km = 70.0 * scale
            options.append(
                RouteOption(
                    id=f"{option_prefix}_{idx}",
                    geometry=GeoJSONLineString(type="LineString", coordinates=coords),
                    metrics=RouteMetrics(
                        distance_km=distance_km,
                        duration_s=duration_s,
                        monetary_cost=120.0 * scale,
                        emissions_kg=95.0 * scale,
                        avg_speed_kmh=distance_km / (duration_s / 3600.0),
                        weather_delay_s=120.0 if weather_enabled else 0.0,
                        incident_delay_s=0.0,
                    ),
                    eta_explanations=[
                        "Baseline ETA computed.",
                        "Time-of-day profile applied.",
                        "Scenario multiplier applied.",
                        "Terrain profile applied.",
                    ],
                    eta_timeline=[
                        {
                            "stage": str(stage["stage"]),
                            "duration_s": float(stage["duration_s"]) * scale,
                            "delta_s": float(stage["delta_s"]) * scale,
                        }
                        for stage in base_stages
                    ],
                    segment_breakdown=[
                        {
                            "segment_index": 0,
                            "distance_km": distance_km,
                            "duration_s": duration_s,
                            "incident_delay_s": 0.0,
                            "emissions_kg": 95.0 * scale,
                            "monetary_cost": 120.0 * scale,
                        }
                    ],
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
                    incident_events=[],
                )
            )

        return (
            options,
            [],
            len(options),
            TerrainDiagnostics(),
            CandidateDiagnostics(
                raw_count=len(options),
                deduped_count=len(options),
                graph_explored_states=3,
                graph_generated_paths=3,
                graph_emitted_paths=len(options),
                candidate_budget=3,
            ),
        )

    monkeypatch.setattr(
        main_module,
        "_collect_route_options_with_diagnostics",
        _fake_collect_route_options_with_diagnostics,
    )
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
                "epsilon": {"duration_s": 25000, "monetary_cost": 5000, "emissions_kg": 5000},
                "departure_time_utc": "2026-02-12T08:45:00Z",
                "weather": {
                    "enabled": True,
                    "profile": "storm",
                    "intensity": 1.0,
                    "apply_incident_uplift": True,
                },
                "incident_simulation": {
                    "enabled": True,
                    "seed": 99,
                    "dwell_rate_per_100km": 120.0,
                    "accident_rate_per_100km": 90.0,
                    "closure_rate_per_100km": 60.0,
                },
            }
            pareto_resp = client.post("/pareto", json=pareto_payload)
            assert pareto_resp.status_code == 200
            routes = pareto_resp.json()["routes"]
            assert routes
            assert all(route["metrics"]["duration_s"] <= 25000 for route in routes)
            assert sum(1 for route in routes if route["is_knee"]) == 1
            for route in routes:
                assert route["knee_score"] is not None
                stages = [item["stage"] for item in route["eta_timeline"]]
                assert stages == [
                    "baseline",
                    "time_of_day",
                    "scenario",
                    "weather",
                    "gradient",
                ]
                assert len(route["eta_explanations"]) >= 2
                assert len(route["segment_breakdown"]) >= 1
                assert route["metrics"]["weather_delay_s"] >= 0.0
                assert route["metrics"]["incident_delay_s"] == 0.0
                assert route["incident_events"] == []

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
