from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.calibration_loader as calibration_loader
import app.carbon_model as carbon_model
import app.main as main_module
import app.fuel_energy_model as fuel_energy_model
from app.calibration_loader import FuelPriceSnapshot
from app.main import app, osrm_client
from app.routing_graph import GraphCandidateDiagnostics
from app.settings import settings


class FakeOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "distance": 65_000.0,
                "duration": 3_600.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
                },
                "legs": [
                    {
                        "annotation": {
                            "distance": [32_500.0, 32_500.0],
                            "duration": [1_800.0, 1_800.0],
                        }
                    }
                ],
            }
        ]


def _fake_graph_candidate_routes(**_: Any) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
    return (
        [
            {
                "distance": 65_000.0,
                "duration": 3_600.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
                },
                "legs": [
                    {
                        "annotation": {
                            "distance": [32_500.0, 32_500.0],
                            "duration": [1_800.0, 1_800.0],
                        }
                    }
                ],
            }
        ],
        GraphCandidateDiagnostics(
            explored_states=42,
            generated_paths=3,
            emitted_paths=1,
            candidate_budget=3,
        ),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scenario_profiles_payload(now_iso: str) -> dict[str, Any]:
    transform = calibration_loader._default_scenario_transform_params()
    transform["fit_strategy"] = "empirical_temporal_forward"
    transform["scenario_edge_scaling_version"] = "v4_live_empirical"
    transform["context_similarity"]["max_distance"] = 10.0
    q = lambda v: {"p10": v * 0.97, "p50": v, "p90": v * 1.03}
    base_profiles = {
        "no_sharing": {
            "duration_multiplier": q(1.10),
            "incident_rate_multiplier": q(1.08),
            "incident_delay_multiplier": q(1.12),
            "fuel_consumption_multiplier": q(1.06),
            "emissions_multiplier": q(1.05),
            "stochastic_sigma_multiplier": q(1.10),
        },
        "partial_sharing": {
            "duration_multiplier": q(1.02),
            "incident_rate_multiplier": q(1.01),
            "incident_delay_multiplier": q(1.03),
            "fuel_consumption_multiplier": q(1.01),
            "emissions_multiplier": q(1.01),
            "stochastic_sigma_multiplier": q(1.02),
        },
        "full_sharing": {
            "duration_multiplier": q(0.95),
            "incident_rate_multiplier": q(0.94),
            "incident_delay_multiplier": q(0.95),
            "fuel_consumption_multiplier": q(0.96),
            "emissions_multiplier": q(0.96),
            "stochastic_sigma_multiplier": q(0.95),
        },
    }
    contexts: list[dict[str, Any]] = []
    for corridor_idx in range(8):
        corridor = f"c{corridor_idx:02d}"
        for slot in (0, 4, 8, 12, 16, 20):
            contexts.append(
                {
                    "context_key": f"{corridor}|h{slot:02d}|weekday|mixed|rigid_hgv|clear",
                    "corridor_bucket": corridor,
                    "corridor_geohash5": corridor,
                    "road_mix_bucket": "mixed",
                    "vehicle_class": "rigid_hgv",
                    "day_kind": "weekday",
                    "weather_bucket": "clear",
                    "weather_regime": "clear",
                    "hour_slot_local": slot,
                    "road_mix_vector": {"mixed": 1.0},
                    "mode_observation_source": "observed_live",
                    "mode_projection_ratio": 0.0,
                    "profiles": base_profiles,
                }
            )
    return {
        "version": "scenario_live_test_v1",
        "calibration_basis": "empirical",
        "as_of_utc": now_iso,
        "generated_at_utc": now_iso,
        "split_strategy": "temporal_forward_plus_corridor_block",
        "holdout_metrics": {
            "mode_separation_mean": 0.12,
            "duration_mape": 0.04,
            "monetary_mape": 0.04,
            "emissions_mape": 0.04,
            "coverage": 0.97,
            "hour_slot_coverage": 12.0,
            "corridor_coverage": 10.0,
            "full_identity_share": 0.12,
            "projection_dominant_context_share": 0.0,
            "observed_mode_row_share": 1.0,
        },
        "profiles": base_profiles,
        "contexts": contexts,
        "transform_params": transform,
    }


def _scenario_context_payload(now_iso: str) -> dict[str, Any]:
    fetch = {
        "source_url": "https://live.example/source",
        "fetch_error": None,
        "cache_hit": False,
        "stale_cache_used": False,
        "status_code": 200,
        "as_of_utc": now_iso,
    }
    return {
        "as_of_utc": now_iso,
        "source_set": {
            "webtris": "https://live.example/webtris",
            "traffic_england": "https://live.example/traffic",
            "dft_counts": "https://live.example/dft",
            "open_meteo": "https://live.example/meteo",
        },
        "coverage": {"webtris": 1.0, "traffic_england": 1.0, "dft": 1.0, "open_meteo": 1.0, "overall": 1.0},
        "traffic_features": {"flow_index": 120.0, "speed_index": 62.0},
        "incident_features": {"delay_pressure": 2.5, "severity_index": 0.7},
        "weather_features": {"weather_severity_index": 0.3, "weather_bucket": "clear"},
        "source_diagnostics": {
            "webtris": {"fetch": dict(fetch)},
            "traffic_england": {"fetch": dict(fetch)},
            "dft_counts": {"fetch": dict(fetch)},
            "open_meteo": {"fetch": dict(fetch)},
        },
    }


def _departure_profile_payload(now_iso: str) -> dict[str, Any]:
    return {
        "version": "departure_live_test_v1",
        "calibration_basis": "empirical",
        "as_of_utc": now_iso,
        "weekday": [1.0] * 1440,
        "weekend": [0.95] * 1440,
        "holiday": [0.90] * 1440,
    }


def _bank_holidays_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "england-and-wales": {"events": [{"date": "2026-12-25"}]},
    }


def _fuel_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "prices_gbp_per_l": {"diesel": 1.52, "petrol": 1.58},
        "grid_price_gbp_per_kwh": 0.28,
        "regional_multipliers": {"uk_default": 1.0},
    }


def _toll_tariffs_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "default_crossing_fee_gbp": 0.0,
        "default_distance_fee_gbp_per_km": 0.0,
        "rules": [
            {
                "operator": "nh",
                "crossing_id": "dartford",
                "road_class": "motorway",
                "direction": "both",
                "start_minute": 0,
                "end_minute": 1440,
                "crossing_fee_gbp": 2.5,
                "distance_fee_gbp_per_km": 0.15,
                "vehicle_classes": ["rigid_hgv", "artic_hgv", "van"],
                "axle_classes": ["3to4", "5plus", "2"],
                "payment_classes": ["cash", "electronic"],
                "exemptions": [],
            }
        ],
    }


def _toll_topology_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "dartford_seg",
                    "name": "Dartford Crossing",
                    "operator": "nh",
                    "road_class": "motorway",
                    "crossing_id": "dartford",
                    "direction": "both",
                    "crossing_fee_gbp": 2.5,
                    "distance_fee_gbp_per_km": 0.15,
                },
                "geometry": {"type": "LineString", "coordinates": [[-1.7, 52.3], [-0.9, 51.9]]},
            }
        ],
    }


def _stochastic_payload(now_iso: str) -> dict[str, Any]:
    context_probs: dict[str, dict[str, float]] = {}
    corridors = [f"c{i:02d}" for i in range(8)]
    slots = ["h00", "h04", "h08", "h12", "h16", "h20"]
    for corridor in corridors:
        for slot in slots:
            key = f"{corridor}|weekday|{slot}|mixed|clear|default"
            context_probs[key] = {"weekday_offpeak": 1.0}
    context_probs["*|weekday|h12|mixed|clear|default"] = {"weekday_offpeak": 1.0}
    context_probs["*|weekday|*|mixed|clear|default"] = {"weekday_offpeak": 1.0}
    return {
        "as_of_utc": now_iso,
        "calibration_basis": "empirical",
        "calibration_version": "stochastic_live_test_v1",
        "copula_id": "gaussian_5x5_v2",
        "split_strategy": "temporal_forward_plus_corridor_block",
        "holdout_window": {"start_utc": "2025-01-01T00:00:00Z", "end_utc": "2025-12-31T23:59:59Z"},
        "holdout_metrics": {"pit_mean": 0.5, "coverage": 0.95, "crps_mean": 0.2, "duration_mape": 0.10},
        "coverage_metrics": {"hour_slot_coverage": 12.0, "corridor_coverage": 10.0},
        "posterior_model": {"context_to_regime_probs": context_probs},
        "regimes": {
            "weekday_offpeak": {
                "sigma_scale": 1.0,
                "traffic_scale": 1.0,
                "incident_scale": 1.0,
                "weather_scale": 1.0,
                "price_scale": 1.0,
                "eco_scale": 1.0,
                "corr": [
                    [1.0, 0.20, 0.15, 0.10, 0.08],
                    [0.20, 1.0, 0.20, 0.12, 0.10],
                    [0.15, 0.20, 1.0, 0.14, 0.12],
                    [0.10, 0.12, 0.14, 1.0, 0.20],
                    [0.08, 0.10, 0.12, 0.20, 1.0],
                ],
                "spread_floor": 0.05,
                "spread_cap": 1.25,
                "factor_low": 0.55,
                "factor_high": 2.2,
                "duration_mix": [1.0, 1.0, 1.0],
                "monetary_mix": [0.62, 0.38],
                "emissions_mix": [0.72, 0.28],
                "transform_family": "quantile_mapping_v1",
                "shock_quantile_mapping": {
                    "traffic": [[-2.0, 0.75], [0.0, 1.0], [2.0, 1.35]],
                    "incident": [[-2.0, 0.72], [0.0, 1.0], [2.0, 1.40]],
                    "weather": [[-2.0, 0.78], [0.0, 1.0], [2.0, 1.30]],
                    "price": [[-2.0, 0.82], [0.0, 1.0], [2.0, 1.25]],
                    "eco": [[-2.0, 0.80], [0.0, 1.0], [2.0, 1.22]],
                },
            }
        },
    }


def _carbon_schedule_payload(now_iso: str) -> dict[str, Any]:
    return {
        "as_of_utc": now_iso,
        "source": "desnz_live_schedule",
        "calibration_basis": "empirical",
        "prices_gbp_per_kg": {"central": {"2025": 0.10, "2026": 0.11, "2027": 0.12}},
        "uncertainty_distribution_by_year": {
            "2025": {"p10": 0.08, "p50": 0.10, "p90": 0.12},
            "2026": {"p10": 0.09, "p50": 0.11, "p90": 0.13},
            "2027": {"p10": 0.10, "p50": 0.12, "p90": 0.14},
        },
        "ev_grid_intensity_kg_per_kwh_by_region": {"uk_default": [0.20] * 24},
        "non_ev_scope_factors": {"wtw": {"2026": 1.20}, "lca": {"2026": 1.32}},
    }


@pytest.fixture(autouse=True)
def _strict_runtime_test_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("STRICT_RUNTIME_TEST_BYPASS") is None:
        monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")
    now_iso = _now_iso()
    monkeypatch.setattr(settings, "scenario_require_signature", False)
    monkeypatch.setattr(settings, "live_fuel_require_signature", False)
    monkeypatch.setattr(settings, "live_scenario_coefficient_url", "https://live.example/scenario")
    monkeypatch.setattr(settings, "live_departure_profile_url", "https://live.example/departure")
    monkeypatch.setattr(settings, "live_fuel_price_url", "https://live.example/fuel")
    monkeypatch.setattr(settings, "live_toll_tariffs_url", "https://live.example/tariffs")
    monkeypatch.setattr(settings, "live_toll_topology_url", "https://live.example/topology")
    monkeypatch.setattr(settings, "live_stochastic_regimes_url", "https://live.example/stochastic")
    monkeypatch.setattr(settings, "live_carbon_schedule_url", "https://live.example/carbon")
    monkeypatch.setattr(calibration_loader, "live_scenario_profiles", lambda: _scenario_profiles_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_scenario_context", lambda _ctx: _scenario_context_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_departure_profiles", lambda: _departure_profile_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_bank_holidays", lambda: _bank_holidays_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_fuel_prices", lambda _as_of=None: _fuel_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_toll_tariffs", lambda: _toll_tariffs_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_toll_topology", lambda: _toll_topology_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_stochastic_regimes", lambda: _stochastic_payload(now_iso))
    monkeypatch.setattr(carbon_model, "live_carbon_schedule", lambda: _carbon_schedule_payload(now_iso))
    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(main_module, "route_graph_candidate_routes", _fake_graph_candidate_routes)
    monkeypatch.setattr(settings, "route_graph_scenario_separability_fail", False)
    monkeypatch.setattr(
        fuel_energy_model,
        "load_fuel_price_snapshot",
        lambda as_of_utc=None: FuelPriceSnapshot(
            prices_gbp_per_l={"diesel": 1.52, "petrol": 1.58},
            grid_price_gbp_per_kwh=0.28,
            regional_multipliers={"uk000": 1.0},
            as_of="2026-01-15T00:00:00Z",
            source="pytest_fixture",
            signature="pytest",
            live_diagnostics={"cache_hit": True, "source_url": "pytest://fuel"},
        ),
    )
    calibration_loader.load_fuel_price_snapshot.cache_clear()
    calibration_loader.load_scenario_profiles.cache_clear()
    calibration_loader.load_live_scenario_context.cache_clear()
    calibration_loader.load_departure_profile.cache_clear()
    calibration_loader.load_uk_bank_holidays.cache_clear()
    calibration_loader.load_toll_tariffs.cache_clear()
    calibration_loader.load_toll_segments_seed.cache_clear()
    calibration_loader.load_stochastic_regimes.cache_clear()


def test_departure_optimize_validation_and_deterministic_best(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            invalid_resp = client.post(
                "/departure/optimize",
                json={
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "window_start_utc": "2026-02-12T12:00:00Z",
                    "window_end_utc": "2026-02-12T11:00:00Z",
                    "step_minutes": 60,
                },
            )
            assert invalid_resp.status_code == 422
            invalid_detail = invalid_resp.json()["detail"]
            assert invalid_detail["reason_code"] == "epsilon_infeasible"

            valid_resp = client.post(
                "/departure/optimize",
                json={
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "vehicle_type": "rigid_hgv",
                    "scenario_mode": "no_sharing",
                    "weights": {"time": 1, "money": 1, "co2": 1},
                    "cost_toggles": {"use_tolls": False},
                    "window_start_utc": "2026-02-12T02:00:00Z",
                    "window_end_utc": "2026-02-12T08:00:00Z",
                    "step_minutes": 180,
                },
            )
            assert valid_resp.status_code == 200
            payload = valid_resp.json()
            assert payload["evaluated_count"] >= 3
            assert payload["best"] is not None
            scores = [item["score"] for item in payload["candidates"]]
            assert scores == sorted(scores)
            assert payload["best"]["score"] == min(scores)
            candidate_times = {item["departure_time_utc"] for item in payload["candidates"]}
            assert payload["best"]["departure_time_utc"] in candidate_times
    finally:
        app.dependency_overrides.clear()


def test_departure_optimize_time_window_feasible_and_infeasible(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            feasible_resp = client.post(
                "/departure/optimize",
                json={
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "cost_toggles": {"use_tolls": False},
                    "window_start_utc": "2026-02-12T02:00:00Z",
                    "window_end_utc": "2026-02-12T08:00:00Z",
                    "step_minutes": 180,
                    "time_window": {
                        "latest_arrival_utc": "2026-02-12T03:30:00Z",
                    },
                },
            )
            assert feasible_resp.status_code == 200
            feasible_payload = feasible_resp.json()
            assert feasible_payload["evaluated_count"] >= 1
            assert feasible_payload["best"]["departure_time_utc"].startswith("2026-02-12T02:00:00")

            infeasible_resp = client.post(
                "/departure/optimize",
                json={
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "cost_toggles": {"use_tolls": False},
                    "window_start_utc": "2026-02-12T02:00:00Z",
                    "window_end_utc": "2026-02-12T08:00:00Z",
                    "step_minutes": 180,
                    "time_window": {
                        "earliest_arrival_utc": "2026-02-12T20:00:00Z",
                        "latest_arrival_utc": "2026-02-12T20:10:00Z",
                    },
                },
            )
            assert infeasible_resp.status_code == 422
            detail = infeasible_resp.json()["detail"]
            assert detail["reason_code"] == "epsilon_infeasible"
            assert "no feasible departures" in detail["message"]
    finally:
        app.dependency_overrides.clear()
