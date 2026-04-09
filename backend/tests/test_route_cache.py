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
from app.models import CostToggles, DecisionPackage, GeoJSONLineString, RouteMetrics, RouteOption, ScenarioSummary
from app.routing_graph import GraphCandidateDiagnostics
from app.scenario import ScenarioMode, ScenarioPolicy, ScenarioRouteContext
from app.settings import settings
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


def _route_option_from_raw_route(
    raw_route: dict[str, Any],
    *,
    route_id: str,
    scenario_mode: ScenarioMode,
) -> RouteOption:
    coords = [
        (float(point[0]), float(point[1]))
        for point in raw_route["geometry"]["coordinates"]
    ]
    coords = main_module._downsample_coords(coords)
    distance_km = float(raw_route["distance"]) / 1000.0
    duration_s = float(raw_route["duration"])
    avg_speed_kmh = distance_km / max(duration_s / 3600.0, 1e-9)
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=coords),
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_s=duration_s,
            monetary_cost=distance_km * 2.0,
            emissions_kg=distance_km * 0.8,
            avg_speed_kmh=avg_speed_kmh,
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


@pytest.fixture(autouse=True)
def _runtime_stubs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", False)
    monkeypatch.setattr(settings, "live_route_compute_refresh_mode", "route_compute")
    monkeypatch.setattr(settings, "live_route_compute_probe_terrain", False)
    monkeypatch.setattr(
        main_module,
        "_validate_route_options_evidence",
        lambda options: {
            "status": "ok",
            "issues": [],
            "validations": [
                {"status": "ok", "issues": [], "route_id": getattr(option, "id", "")}
                for option in options
            ],
        },
    )

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
    async def _fake_graph_precheck(**_kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "reason_code": "ok", "origin_node_id": "pytest_origin", "destination_node_id": "pytest_destination"}

    monkeypatch.setattr(main_module, "_route_graph_od_feasibility_async", _fake_graph_precheck)
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


@pytest.fixture
def _bounded_tri_source_cache_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_compute_direct_route_pipeline(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["pipeline_mode"] == "tri_source"
        req = kwargs["req"]
        osrm = kwargs["osrm"]
        raw_routes: list[dict[str, Any]] = []

        for idx in range(9):
            via = [(52.2 + (idx * 0.01), -1.1 + (idx * 0.01))]
            cache_key = main_module._graph_refine_route_cache_key(
                origin=req.origin,
                destination=req.destination,
                alternatives=False,
                via=via,
                vehicle_type=req.vehicle_type,
                scenario_mode=req.scenario_mode,
                cost_toggles=req.cost_toggles,
                terrain_profile=req.terrain_profile,
                departure_time_utc=req.departure_time_utc,
                scenario_cache_token=None,
            )
            cached = route_cache.get_cached_routes(cache_key)
            if cached is not None:
                candidate_routes = cached[0]
            else:
                candidate_routes = await osrm.fetch_routes(
                    origin=req.origin,
                    destination=req.destination,
                    alternatives=False,
                    via=via,
                )
                route_cache.set_cached_routes(
                    cache_key,
                    (
                        candidate_routes,
                        [],
                        1,
                        {
                            "label": f"graph_family:route_{idx}",
                            "cache_kind": "graph_refine",
                        },
                    ),
                )
            raw_routes.append(dict(candidate_routes[0]))

        options = [
            _route_option_from_raw_route(
                raw_route,
                route_id=f"route_{idx}",
                scenario_mode=req.scenario_mode,
            )
            for idx, raw_route in enumerate(raw_routes, start=1)
        ]
        selected = options[0]
        return {
            "selected": selected,
            "candidates": options,
            "warnings": [],
            "candidate_fetches": len(raw_routes),
            "terrain_diag": main_module.TerrainDiagnostics(),
            "candidate_diag": main_module.CandidateDiagnostics(
                raw_count=len(raw_routes),
                deduped_count=len(options),
                candidate_budget=len(options),
            ),
            "selected_certificate": None,
            "voi_stop_summary": None,
            "extra_json_artifacts": {},
            "extra_jsonl_artifacts": {},
            "extra_csv_artifacts": {},
            "extra_text_artifacts": {},
        }

    monkeypatch.setattr(
        main_module,
        "_compute_direct_route_pipeline",
        _fake_compute_direct_route_pipeline,
    )
    monkeypatch.setattr(
        main_module,
        "_build_route_decision_package",
        lambda **kwargs: DecisionPackage(selected_route_id=kwargs["selected"].id),
    )
    monkeypatch.setattr(
        main_module,
        "_write_route_run_bundle",
        lambda **_kwargs: {
            "run_id": "pytest-route-cache",
            "manifest_endpoint": "/runs/pytest-route-cache/manifest",
            "artifacts_endpoint": "/runs/pytest-route-cache/artifacts",
            "provenance_endpoint": "/runs/pytest-route-cache/provenance",
        },
    )


def test_route_cache_hits_and_keying(_bounded_tri_source_cache_runtime: None) -> None:
    osrm = CountingOSRM()
    app.dependency_overrides[osrm_client] = lambda: osrm
    route_cache.clear_route_cache()
    route_cache.clear_route_cache_checkpoint()
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
            stats_payload = stats_resp.json()
            stats = stats_payload["route_cache"]
            assert stats["hits"] >= 1
            assert stats["misses"] >= 2
            assert stats["size"] >= 1
            assert stats["schema_version"] == route_cache.ROUTE_CACHE_SCHEMA_VERSION
            assert stats["checkpoint_operations"] == 0
            assert stats["restore_operations"] == 0
            assert stats["invalidation_counters"]["expired"] == 0
            assert stats["invalidation_counters"]["manual_clear"] >= 0
            assert "hot_rerun_route_cache_checkpoint" in stats_payload
            checkpoint_stats = stats_payload["hot_rerun_route_cache_checkpoint"]
            assert checkpoint_stats["schema_version"] == route_cache.ROUTE_CACHE_SCHEMA_VERSION
            assert checkpoint_stats["checkpoint_operations"] == 0
            assert checkpoint_stats["restore_operations"] == 0
            assert "certification_cache" in stats_payload
    finally:
        app.dependency_overrides.clear()
        route_cache.clear_route_cache()
        route_cache.clear_route_cache_checkpoint()


def test_thesis_cold_cache_scope_preserves_k_raw_and_option_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_calls: list[str] = []

    monkeypatch.setattr(main_module, "clear_route_cache", lambda: observed_calls.append("route") or 11)
    monkeypatch.setattr(main_module, "clear_k_raw_cache", lambda: observed_calls.append("k_raw") or 13)
    monkeypatch.setattr(main_module, "clear_route_option_cache", lambda: observed_calls.append("route_option") or 17)
    monkeypatch.setattr(main_module, "clear_certification_cache", lambda: observed_calls.append("certification") or 19)
    monkeypatch.setattr(main_module, "clear_route_state_cache", lambda: observed_calls.append("route_state") or 23)
    monkeypatch.setattr(main_module, "clear_voi_dccs_cache", lambda: observed_calls.append("voi_dccs") or 29)
    monkeypatch.setattr(main_module, "clear_route_cache_checkpoint", lambda: observed_calls.append("route_checkpoint") or 31)

    with TestClient(app) as client:
        response = client.delete("/cache?scope=thesis_cold")

    assert response.status_code == 200
    assert response.json() == {
        "cleared": 11,
        "cleared_k_raw": 0,
        "cleared_route_option": 0,
        "cleared_certifications": 19,
        "cleared_route_state": 23,
        "cleared_voi_dccs": 29,
    }
    assert observed_calls == ["route_checkpoint", "route", "certification", "route_state", "voi_dccs"]


def test_hot_rerun_cold_source_scope_preserves_certification_and_option_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_calls: list[str] = []

    monkeypatch.setattr(main_module, "checkpoint_route_cache", lambda: observed_calls.append("route_checkpoint") or 7)
    monkeypatch.setattr(main_module, "clear_route_cache", lambda: observed_calls.append("route") or 11)
    monkeypatch.setattr(main_module, "clear_k_raw_cache", lambda: observed_calls.append("k_raw") or 13)
    monkeypatch.setattr(main_module, "clear_route_option_cache", lambda: observed_calls.append("route_option") or 17)
    monkeypatch.setattr(main_module, "clear_certification_cache", lambda: observed_calls.append("certification") or 19)
    monkeypatch.setattr(main_module, "clear_route_state_cache", lambda: observed_calls.append("route_state") or 23)
    monkeypatch.setattr(main_module, "clear_voi_dccs_cache", lambda: observed_calls.append("voi_dccs") or 29)

    with TestClient(app) as client:
        response = client.delete("/cache?scope=hot_rerun_cold_source")

    assert response.status_code == 200
    assert response.json() == {
        "cleared": 11,
        "cleared_k_raw": 0,
        "cleared_route_option": 0,
        "cleared_certifications": 0,
        "cleared_route_state": 23,
        "cleared_voi_dccs": 29,
    }
    assert observed_calls == ["route_checkpoint", "route", "route_state", "voi_dccs"]


def test_route_cache_checkpoint_restore_round_trip() -> None:
    route_cache.clear_route_cache()
    route_cache.clear_route_cache_checkpoint()
    try:
        stored = route_cache.set_cached_routes(
            "checkpoint-key",
            ([{"route_id": "route-1"}], [], 1),
        )
        assert stored is True
        checkpointed = route_cache.checkpoint_route_cache()
        assert checkpointed == 1
        assert route_cache.clear_route_cache() == 1
        assert route_cache.get_cached_routes("checkpoint-key") is None
        restored = route_cache.restore_checkpointed_route_cache(clear_first=False)
        assert restored == 1
        cached = route_cache.get_cached_routes("checkpoint-key")
        assert cached is not None
        assert cached[0][0]["route_id"] == "route-1"
        active_stats = route_cache.route_cache_stats()
        checkpoint_stats = route_cache.route_cache_checkpoint_stats()
        assert active_stats["schema_version"] == route_cache.ROUTE_CACHE_SCHEMA_VERSION
        assert active_stats["checkpoint_operations"] >= 1
        assert active_stats["checkpointed_entries"] >= 1
        assert active_stats["restore_operations"] >= 1
        assert active_stats["restored_entries"] >= 1
        assert checkpoint_stats["checkpoint_operations"] >= 1
        assert checkpoint_stats["checkpointed_entries"] >= 1
        assert checkpoint_stats["restore_operations"] >= 1
        assert checkpoint_stats["restored_entries"] >= 1
    finally:
        route_cache.clear_route_cache()
        route_cache.clear_route_cache_checkpoint()


def test_restore_hot_rerun_cache_endpoint_reports_restore_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "restore_checkpointed_route_cache", lambda clear_first=False: 5)
    monkeypatch.setattr(main_module, "route_cache_checkpoint_stats", lambda: {"size": 7})

    with TestClient(app) as client:
        response = client.post("/cache/hot-rerun/restore")

    assert response.status_code == 200
    assert response.json() == {
        "restored": 5,
        "checkpoint_size": 7,
    }


def test_scenario_policy_cache_is_keyed_by_exact_context(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def _policy(mode: ScenarioMode, *, context: ScenarioRouteContext) -> ScenarioPolicy:
        calls.append((mode.value, context.context_key))
        return ScenarioPolicy(
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="pytest",
            version="pytest",
            context_key=context.context_key,
        )

    monkeypatch.setattr(main_module, "resolve_scenario_profile", _policy)

    cache: dict[tuple[str, str], ScenarioPolicy] = {
        ("__shared__", ScenarioMode.PARTIAL_SHARING.value): ScenarioPolicy(
            duration_multiplier=9.0,
            incident_rate_multiplier=9.0,
            incident_delay_multiplier=9.0,
            fuel_consumption_multiplier=9.0,
            emissions_multiplier=9.0,
            stochastic_sigma_multiplier=9.0,
            source="stale",
            version="stale",
            context_key="stale|shared|context",
        )
    }
    road_mix_vector = {"motorway": 0.2, "trunk": 0.2, "primary": 0.2, "secondary": 0.2, "local": 0.2}
    weekend = ScenarioRouteContext(
        corridor_geohash5="1c816",
        hour_slot_local=8,
        day_kind="weekend",
        road_mix_bucket="mixed",
        road_mix_vector=road_mix_vector,
        vehicle_class="rigid_hgv",
        weather_regime="clear",
    )
    weekday = ScenarioRouteContext(
        corridor_geohash5="1c816",
        hour_slot_local=8,
        day_kind="weekday",
        road_mix_bucket="mixed",
        road_mix_vector=road_mix_vector,
        vehicle_class="rigid_hgv",
        weather_regime="clear",
    )

    weekend_policy = main_module._resolve_route_scenario_policy(
        scenario_mode=ScenarioMode.PARTIAL_SHARING,
        scenario_context=weekend,
        scenario_policy_cache=cache,
    )
    weekend_policy_repeat = main_module._resolve_route_scenario_policy(
        scenario_mode=ScenarioMode.PARTIAL_SHARING,
        scenario_context=weekend,
        scenario_policy_cache=cache,
    )
    weekday_policy = main_module._resolve_route_scenario_policy(
        scenario_mode=ScenarioMode.PARTIAL_SHARING,
        scenario_context=weekday,
        scenario_policy_cache=cache,
    )

    assert weekend_policy.context_key == weekend.context_key
    assert weekend_policy_repeat.context_key == weekend.context_key
    assert weekday_policy.context_key == weekday.context_key
    assert calls == [
        (ScenarioMode.PARTIAL_SHARING.value, weekend.context_key),
        (ScenarioMode.PARTIAL_SHARING.value, weekday.context_key),
    ]
    assert cache[(ScenarioMode.PARTIAL_SHARING.value, weekend.context_key)] is weekend_policy
    assert cache[(ScenarioMode.PARTIAL_SHARING.value, weekday.context_key)] is weekday_policy


def test_route_cache_ttl_expiry_causes_recompute(_bounded_tri_source_cache_runtime: None) -> None:
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
            stats = route_cache.route_cache_stats()
            assert stats["invalidation_counters"]["expired"] >= 1
    finally:
        route_cache.ROUTE_CACHE._ttl_s = old_ttl
        app.dependency_overrides.clear()
        route_cache.clear_route_cache()


def test_graph_refine_route_cache_key_ignores_candidate_label() -> None:
    origin = main_module.LatLng(lat=52.4862, lon=-1.8904)
    destination = main_module.LatLng(lat=51.5072, lon=-0.1276)
    departure = datetime(2026, 3, 30, 9, 0, tzinfo=UTC)
    toggles = CostToggles(use_tolls=True, carbon_price_per_kg=0.10)

    key_a = main_module._graph_refine_route_cache_key(
        origin=origin,
        destination=destination,
        alternatives=False,
        via=[(52.2, -1.1), (51.9, -0.9)],
        vehicle_type="rigid_hgv",
        scenario_mode="full_sharing",
        cost_toggles=toggles,
        terrain_profile="flat",
        departure_time_utc=departure,
        scenario_cache_token="scenario-a",
    )
    key_b = main_module._graph_refine_route_cache_key(
        origin=origin,
        destination=destination,
        alternatives=False,
        via=[(52.2, -1.1), (51.9, -0.9)],
        vehicle_type="rigid_hgv",
        scenario_mode="full_sharing",
        cost_toggles=toggles,
        terrain_profile="flat",
        departure_time_utc=departure,
        scenario_cache_token="scenario-a",
    )
    changed_via_key = main_module._graph_refine_route_cache_key(
        origin=origin,
        destination=destination,
        alternatives=False,
        via=[(52.25, -1.1), (51.9, -0.9)],
        vehicle_type="rigid_hgv",
        scenario_mode="full_sharing",
        cost_toggles=toggles,
        terrain_profile="flat",
        departure_time_utc=departure,
        scenario_cache_token="scenario-a",
    )
    changed_exclude_key = main_module._graph_refine_route_cache_key(
        origin=origin,
        destination=destination,
        alternatives=False,
        exclude="toll",
        via=[(52.2, -1.1), (51.9, -0.9)],
        vehicle_type="rigid_hgv",
        scenario_mode="full_sharing",
        cost_toggles=toggles,
        terrain_profile="flat",
        departure_time_utc=departure,
        scenario_cache_token="scenario-a",
    )

    assert key_a == key_b
    assert changed_via_key != key_a
    assert changed_exclude_key != key_a


def test_graph_refine_route_cache_key_separates_scenario_sensitive_context() -> None:
    origin = main_module.LatLng(lat=52.4862, lon=-1.8904)
    destination = main_module.LatLng(lat=51.5072, lon=-0.1276)
    via = [(52.2, -1.1), (51.9, -0.9)]

    base = main_module._graph_refine_route_cache_key(
        origin=origin,
        destination=destination,
        alternatives=False,
        via=via,
        vehicle_type="rigid_hgv",
        scenario_mode="full_sharing",
        cost_toggles=CostToggles(use_tolls=True, carbon_price_per_kg=0.10),
        terrain_profile="flat",
        departure_time_utc=datetime(2026, 3, 30, 9, 0, tzinfo=UTC),
        scenario_cache_token="scenario-a",
    )
    changed_carbon = main_module._graph_refine_route_cache_key(
        origin=origin,
        destination=destination,
        alternatives=False,
        via=via,
        vehicle_type="rigid_hgv",
        scenario_mode="full_sharing",
        cost_toggles=CostToggles(use_tolls=True, carbon_price_per_kg=0.25),
        terrain_profile="flat",
        departure_time_utc=datetime(2026, 3, 30, 9, 0, tzinfo=UTC),
        scenario_cache_token="scenario-a",
    )
    changed_departure = main_module._graph_refine_route_cache_key(
        origin=origin,
        destination=destination,
        alternatives=False,
        via=via,
        vehicle_type="rigid_hgv",
        scenario_mode="full_sharing",
        cost_toggles=CostToggles(use_tolls=True, carbon_price_per_kg=0.10),
        terrain_profile="flat",
        departure_time_utc=datetime(2026, 3, 30, 10, 0, tzinfo=UTC),
        scenario_cache_token="scenario-a",
    )
    changed_scenario = main_module._graph_refine_route_cache_key(
        origin=origin,
        destination=destination,
        alternatives=False,
        via=via,
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        cost_toggles=CostToggles(use_tolls=True, carbon_price_per_kg=0.10),
        terrain_profile="flat",
        departure_time_utc=datetime(2026, 3, 30, 9, 0, tzinfo=UTC),
        scenario_cache_token="scenario-b",
    )

    assert changed_carbon != base
    assert changed_departure != base
    assert changed_scenario != base
