from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app, ors_client, osrm_client
from app.models import (
    GeoJSONLineString,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    ScenarioMode,
    ScenarioSummary,
)
from app.routing_osrm import OSRMError
from app.settings import settings


def _osrm_route(lon_start: float, lat_start: float) -> dict[str, Any]:
    coords = [
        [lon_start, lat_start],
        [lon_start + 0.03, lat_start + 0.02],
        [lon_start + 0.07, lat_start + 0.05],
    ]
    return {
        "distance": 14_000.0,
        "duration": 1_020.0,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [6_500.0, 7_500.0],
                    "duration": [480.0, 540.0],
                }
            }
        ],
    }


class FakeBaselineOSRM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def fetch_routes(
        self,
        *,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        alternatives: bool | int = True,
        exclude: str | None = None,
        via: list[tuple[float, float]] | None = None,
        max_retries: int = 8,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "origin_lat": origin_lat,
                "origin_lon": origin_lon,
                "dest_lat": dest_lat,
                "dest_lon": dest_lon,
                "alternatives": alternatives,
                "exclude": exclude,
                "via": via,
                "max_retries": max_retries,
            }
        )
        if via:
            coords = [[origin_lon, origin_lat]]
            for via_lat, via_lon in via:
                coords.append([via_lon, via_lat])
            coords.append([dest_lon, dest_lat])
            segment_count = max(1, len(coords) - 1)
            distance_per_segment = 18_000.0 / segment_count
            duration_per_segment = 1_260.0 / segment_count
            return [
                {
                    "distance": 18_000.0,
                    "duration": 1_260.0,
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "legs": [
                        {
                            "annotation": {
                                "distance": [distance_per_segment for _ in range(segment_count)],
                                "duration": [duration_per_segment for _ in range(segment_count)],
                            }
                        }
                    ],
                }
            ]
        primary = _osrm_route(origin_lon, origin_lat)
        alt_count = int(alternatives) if isinstance(alternatives, int) else (2 if alternatives else 1)
        if alt_count > 1:
            secondary = _osrm_route(origin_lon + 0.08, origin_lat + 0.05)
            secondary["geometry"]["coordinates"] = [
                [origin_lon, origin_lat],
                [origin_lon + 0.11, origin_lat + 0.08],
                [dest_lon, dest_lat],
            ]
            secondary["distance"] = 16_500.0
            secondary["duration"] = 1_180.0
            secondary["legs"][0]["annotation"] = {
                "distance": [7_500.0, 9_000.0],
                "duration": [520.0, 660.0],
            }
            return [primary, secondary]
        return [primary]


class FailingBaselineOSRM(FakeBaselineOSRM):
    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:  # type: ignore[override]
        _ = kwargs
        raise OSRMError("forced baseline failure")


class FakeBaselineORS:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.base_url = "http://localhost:8082/ors"

    async def fetch_route(
        self,
        *,
        coordinates_lon_lat: list[tuple[float, float]],
        profile: str,
        vehicle_type: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "coordinates": coordinates_lon_lat,
                "profile": profile,
                "vehicle_type": vehicle_type,
            }
        )
        return main_module.ORSRoute(
            distance_m=15_500.0,
            duration_s=1_140.0,
            coordinates=list(coordinates_lon_lat),
        )


def _payload(with_waypoint: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "origin": {"lat": 52.4862, "lon": -1.8904},
        "destination": {"lat": 51.5072, "lon": -0.1276},
        "vehicle_type": "rigid_hgv",
    }
    if with_waypoint:
        body["waypoints"] = [{"lat": 52.0, "lon": -1.2, "label": "via-1"}]
    return body


def _long_corridor_payload() -> dict[str, Any]:
    return {
        "origin": {"lat": 51.5072, "lon": -0.1276},
        "destination": {"lat": 55.9533, "lon": -3.1883},
        "vehicle_type": "rigid_hgv",
    }


class SingleRouteBaselineOSRM(FakeBaselineOSRM):
    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:  # type: ignore[override]
        routes = await super().fetch_routes(**kwargs)
        return routes[:1]


def _route_response_option(
    route_id: str,
    *,
    monetary_cost: float,
    duration_s: float = 8100.0,
    emissions_kg: float = 144.2,
    certificate: float = 0.84,
    certified: bool = True,
    threshold: float = 0.8,
) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[(-1.8904, 52.4862), (-0.1276, 51.5072)],
        ),
        metrics=RouteMetrics(
            distance_km=185.0,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            avg_speed_kmh=82.2,
        ),
        scenario_summary=ScenarioSummary(
            mode=ScenarioMode.NO_SHARING,
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="pytest",
            version="pytest",
        ),
        certification=RouteCertificationSummary(
            route_id=route_id,
            certificate=certificate,
            certified=certified,
            threshold=threshold,
            active_families=["scenario", "weather"],
            top_fragility_families=["weather"],
            top_competitor_route_id="route_b",
            top_value_of_refresh_family="weather",
        ),
    )


@pytest.fixture
def baseline_osrm() -> FakeBaselineOSRM:
    return FakeBaselineOSRM()


@pytest.fixture
def baseline_client(baseline_osrm: FakeBaselineOSRM):
    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def test_route_defaults_to_tri_source_and_returns_decision_package(
    monkeypatch: pytest.MonkeyPatch,
    baseline_osrm: FakeBaselineOSRM,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))
    monkeypatch.setattr(settings, "route_pipeline_default_mode", "tri_source")
    monkeypatch.setattr(main_module, "_routing_graph_warmup_failfast_detail", lambda: None)

    async def _fail_if_legacy_called(**kwargs: Any) -> tuple[
        list[RouteOption],
        list[str],
        int,
        main_module.TerrainDiagnostics,
        main_module.CandidateDiagnostics,
    ]:
        raise AssertionError("legacy route collector should not run for default tri_source requests")

    selected = _route_response_option("tri_source_route_a", monetary_cost=110.0, certificate=0.86, certified=True)
    challenger = _route_response_option(
        "tri_source_route_b",
        monetary_cost=128.0,
        duration_s=8420.0,
        emissions_kg=150.1,
        certificate=0.73,
        certified=False,
    )

    async def _fake_compute_direct_route_pipeline(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["pipeline_mode"] == "tri_source"
        return {
            "selected": selected,
            "candidates": [selected, challenger],
            "warnings": ["tri_source support placeholder active"],
            "candidate_fetches": 2,
            "terrain_diag": main_module.TerrainDiagnostics(),
            "candidate_diag": main_module.CandidateDiagnostics(raw_count=2, deduped_count=2, candidate_budget=2),
            "selected_certificate": selected.certification,
            "voi_stop_summary": None,
            "extra_json_artifacts": {
                "winner_summary.json": {"route_id": selected.id},
                "certificate_summary.json": {
                    "selected_route_id": selected.id,
                    "selected_certificate": 0.86,
                    "selected_certificate_basis": "empirical",
                },
                "sampled_world_manifest.json": {"world_count": 16},
            },
            "extra_jsonl_artifacts": {
                "strict_frontier.jsonl": [
                    {
                        "route_id": selected.id,
                        "monetary_cost": 110.0,
                        "duration_s": 8100.0,
                        "emissions_kg": 144.2,
                        "certificate": 0.86,
                        "certificate_threshold": 0.8,
                        "certified": True,
                    },
                    {
                        "route_id": challenger.id,
                        "monetary_cost": 128.0,
                        "duration_s": 8420.0,
                        "emissions_kg": 150.1,
                        "certificate": 0.73,
                        "certificate_threshold": 0.8,
                        "certified": False,
                    },
                ],
            },
        }

    monkeypatch.setattr(
        main_module,
        "_collect_route_options_with_diagnostics",
        _fail_if_legacy_called,
    )
    monkeypatch.setattr(
        main_module,
        "_compute_direct_route_pipeline",
        _fake_compute_direct_route_pipeline,
    )
    monkeypatch.setattr(
        main_module,
        "_validate_route_options_evidence",
        lambda options: {
            "status": "ok",
            "issues": [],
            "validations": [{"route_id": option.id, "status": "ok"} for option in options],
        },
    )

    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    app.dependency_overrides[ors_client] = lambda: FakeBaselineORS()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/route",
                json={
                    **_payload(),
                    "od_ambiguity_source_count": 3,
                    "od_ambiguity_source_mix": (
                        "routing_graph_probe,engine_augmented_probe,historical_results_bootstrap"
                    ),
                    "od_ambiguity_source_entropy": 0.92,
                    "od_ambiguity_support_ratio": 0.81,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline_mode"] == "tri_source"
    assert data["selected"]["id"] == selected.id
    assert data["decision_package"]["pipeline_mode"] == "tri_source"
    assert data["decision_package"]["selected_route_id"] == selected.id
    assert data["decision_package"]["support_summary"]["observed_source_count"] == 3
    assert data["decision_package"]["certified_set_summary"]["selected_route_id"] == selected.id
    artifact_path = tmp_path / "artifacts" / data["run_id"] / "decision_package.json"
    assert artifact_path.exists()
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["pipeline_mode"] == "tri_source"


def test_route_waypoints_fall_back_from_tri_source_to_legacy_with_manifest_warning(
    monkeypatch: pytest.MonkeyPatch,
    baseline_osrm: FakeBaselineOSRM,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))
    monkeypatch.setattr(main_module, "_routing_graph_warmup_failfast_detail", lambda: None)

    legacy_route = _route_response_option(
        "legacy_waypoint_route",
        monetary_cost=132.0,
        duration_s=9200.0,
        emissions_kg=152.5,
        certificate=0.62,
        certified=False,
    )

    async def _fake_legacy_collect(**kwargs: Any) -> tuple[
        list[RouteOption],
        list[str],
        int,
        main_module.TerrainDiagnostics,
        main_module.CandidateDiagnostics,
    ]:
        return (
            [legacy_route],
            [],
            1,
            main_module.TerrainDiagnostics(),
            main_module.CandidateDiagnostics(raw_count=1, deduped_count=1, candidate_budget=1),
        )

    async def _fail_if_direct_called(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("direct tri_source runtime should not run for waypoint fallback requests")

    monkeypatch.setattr(
        main_module,
        "_collect_route_options_with_diagnostics",
        _fake_legacy_collect,
    )
    monkeypatch.setattr(
        main_module,
        "_compute_direct_route_pipeline",
        _fail_if_direct_called,
    )
    monkeypatch.setattr(
        main_module,
        "_validate_route_options_evidence",
        lambda options: {
            "status": "ok",
            "issues": [],
            "validations": [{"route_id": option.id, "status": "ok"} for option in options],
        },
    )

    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    app.dependency_overrides[ors_client] = lambda: FakeBaselineORS()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/route",
                json={
                    **_payload(with_waypoint=True),
                    "pipeline_mode": "tri_source",
                    "od_ambiguity_source_count": 2,
                    "od_ambiguity_source_mix": "routing_graph_probe,engine_augmented_probe",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline_mode"] == "legacy"
    assert data["decision_package"]["pipeline_mode"] == "legacy"
    assert data["decision_package"]["abstention_summary"]["reason_code"] == "legacy_runtime_selected"
    manifest_path = tmp_path / "manifests" / f"{data['run_id']}.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert any("tri_source" in warning and "legacy" in warning for warning in manifest_payload["warnings"])


def test_route_baseline_returns_osrm_quick_route(
    baseline_client: TestClient,
    baseline_osrm: FakeBaselineOSRM,
) -> None:
    resp = baseline_client.post("/route/baseline", json=_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "osrm_quick_baseline"
    assert isinstance(data["compute_ms"], (int, float))
    assert data["baseline"]["geometry"]["type"] == "LineString"
    assert data["baseline"]["metrics"]["distance_km"] > 0
    assert data["baseline"]["metrics"]["duration_s"] > 0
    assert data["baseline"]["metrics"]["emissions_kg"] >= 0
    assert data["baseline"]["metrics"]["monetary_cost"] >= 0
    assert len(baseline_osrm.calls) == 1
    assert baseline_osrm.calls[0]["alternatives"] is False


def test_route_baseline_supports_waypoints(
    baseline_client: TestClient,
    baseline_osrm: FakeBaselineOSRM,
) -> None:
    resp = baseline_client.post("/route/baseline", json=_payload(with_waypoint=True))
    assert resp.status_code == 200
    assert len(baseline_osrm.calls) == 1
    assert baseline_osrm.calls[0]["via"] == [(52.0, -1.2)]


def test_route_baseline_bypasses_strict_warmup_gate(
    baseline_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "route_graph_warmup_failfast", True)
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "loading",
            "phase": "parsing_nodes",
            "elapsed_ms": 125_000.0,
            "started_at_utc": "2026-02-26T22:00:00Z",
            "ready_at_utc": None,
            "last_error": None,
        },
    )
    resp = baseline_client.post("/route/baseline", json=_payload())
    assert resp.status_code == 200
    assert resp.json()["method"] == "osrm_quick_baseline"


def test_route_baseline_returns_structured_error_on_osrm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_osrm = FailingBaselineOSRM()
    app.dependency_overrides[osrm_client] = lambda: failing_osrm
    try:
        with TestClient(app) as client:
            resp = client.post("/route/baseline", json=_payload())
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert detail["reason_code"] == "baseline_route_unavailable"
    assert "baseline_route_unavailable" in detail["message"]


def test_route_ors_baseline_defaults_to_local_service_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "local_ors_runtime_manifest",
        lambda **_: {
            "manifest_hash": "sha256:strict-ors-manifest",
            "recorded_at": "2026-03-22T17:00:00+00:00",
            "identity_status": "graph_identity_verified",
            "compose_image": "openrouteservice/openrouteservice:v9.7.1",
            "graph_listing_digest": "abc123",
            "graph_file_count": 12,
            "graph_total_bytes": 937460812,
            "graph_build_info": {
                "graph_build_date": "2026-03-22T16:39:30+0000",
                "osm_date": "2026-02-23T21:21:28+0000",
            },
        },
    )
    baseline_ors = FakeBaselineORS()
    app.dependency_overrides[ors_client] = lambda: baseline_ors
    try:
        with TestClient(app) as client:
            resp = client.post("/route/baseline/ors", json=_payload())
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_local_engine_baseline"
    assert payload["provider_mode"] == "local_service"
    assert payload["baseline_policy"] == "engine_shortest_path"
    assert payload["asset_freshness_status"] == "graph_identity_verified"
    assert payload["asset_manifest_hash"] == "sha256:strict-ors-manifest"
    assert payload["engine_manifest"]["graph_listing_digest"] == "abc123"
    assert baseline_ors.calls


def test_route_ors_baseline_policy_query_overrides_repo_local_default(
    monkeypatch: pytest.MonkeyPatch,
    baseline_osrm: FakeBaselineOSRM,
) -> None:
    monkeypatch.setattr(settings, "route_ors_baseline_mode", "repo_local")
    baseline_ors = FakeBaselineORS()
    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    app.dependency_overrides[ors_client] = lambda: baseline_ors
    try:
        with TestClient(app) as client:
            resp = client.post("/route/baseline/ors?policy=local_service", json=_payload())
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_local_engine_baseline"
    assert payload["provider_mode"] == "local_service"
    assert len(baseline_ors.calls) == 1
    assert len(baseline_osrm.calls) == 0


def test_route_ors_baseline_fails_closed_on_provider_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mismatched_seed(*, req: Any, ors: Any) -> tuple[dict[str, Any], dict[str, Any]]:  # noqa: ARG001
        return _osrm_route(-1.7904, 52.3862), {
            "provider_mode": "repo_local",
            "baseline_policy": "engine_shortest_path",
        }

    monkeypatch.setattr(main_module, "_fetch_local_ors_baseline_seed", _mismatched_seed)
    with TestClient(app) as client:
        resp = client.post("/route/baseline/ors?policy=local_service", json=_payload())
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert detail["reason_code"] == "baseline_route_unavailable"
    assert "provider mismatch" in detail["message"]


def test_route_ors_baseline_waypoint_realization_uses_local_service_path(
    baseline_osrm: FakeBaselineOSRM,
) -> None:
    baseline_ors = FakeBaselineORS()
    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    app.dependency_overrides[ors_client] = lambda: baseline_ors
    try:
        with TestClient(app) as client:
            resp = client.post("/route/baseline/ors", json=_payload(with_waypoint=True))
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_local_engine_baseline"
    assert payload["provider_mode"] == "local_service"
    assert payload["baseline_policy"] == "engine_shortest_path"
    assert payload["baseline"]["metrics"]["distance_km"] > 0
    assert payload["baseline"]["geometry"]["type"] == "LineString"
    assert len(baseline_osrm.calls) == 0
    assert len(baseline_ors.calls) == 1
    assert baseline_ors.calls[0]["coordinates"][1] == (-1.2, 52.0)


def test_route_ors_baseline_uses_repo_local_osrm_alternative_policy(
    monkeypatch: pytest.MonkeyPatch,
    baseline_osrm: FakeBaselineOSRM,
) -> None:
    monkeypatch.setattr(settings, "route_ors_baseline_mode", "repo_local")
    async def _no_graph_candidates(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:  # noqa: ARG001
        return [], object()

    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _no_graph_candidates)
    monkeypatch.setattr(settings, "route_graph_fast_startup_long_corridor_bypass_km", 9_999.0)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 9_999.0)
    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    try:
        with TestClient(app) as client:
            resp = client.post("/route/baseline/ors", json=_payload())
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_repo_local_baseline"
    assert payload["provider_mode"] == "repo_local"
    assert payload["baseline_policy"] in {"osrm_alternative_min_overlap", "direct_fallback"}
    assert payload["baseline"]["metrics"]["distance_km"] > 0
    assert len(baseline_osrm.calls) >= 2
    assert baseline_osrm.calls[0]["alternatives"] is False
    assert any(call["alternatives"] == 6 for call in baseline_osrm.calls[1:])


def test_route_ors_baseline_long_corridor_skips_graph_search_and_uses_bounded_osrm_policy(
    monkeypatch: pytest.MonkeyPatch,
    baseline_osrm: FakeBaselineOSRM,
) -> None:
    monkeypatch.setattr(settings, "route_ors_baseline_mode", "repo_local")
    async def _graph_should_not_run(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:  # noqa: ARG001
        raise AssertionError("long corridor baseline should skip graph candidate search")

    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_should_not_run)
    monkeypatch.setattr(settings, "route_graph_fast_startup_long_corridor_bypass_km", 100.0)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    try:
        with TestClient(app) as client:
            resp = client.post("/route/baseline/ors", json=_long_corridor_payload())
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_repo_local_baseline"
    assert payload["provider_mode"] == "repo_local"
    assert payload["baseline_policy"] == "long_corridor_osrm_alternative_min_overlap"
    assert payload["baseline"]["metrics"]["distance_km"] > 0
    assert len(baseline_osrm.calls) == 1
    assert baseline_osrm.calls[0]["alternatives"] == 2
    assert any("bounded repo-local degrade policy" in note for note in payload["notes"])


def test_route_ors_baseline_long_corridor_falls_back_direct_when_no_distinct_alt_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "route_ors_baseline_mode", "repo_local")
    async def _graph_should_not_run(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:  # noqa: ARG001
        raise AssertionError("long corridor baseline should skip graph candidate search")

    baseline_osrm = SingleRouteBaselineOSRM()
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _graph_should_not_run)
    monkeypatch.setattr(settings, "route_graph_fast_startup_long_corridor_bypass_km", 100.0)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 150.0)
    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    try:
        with TestClient(app) as client:
            resp = client.post("/route/baseline/ors", json=_long_corridor_payload())
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_repo_local_baseline"
    assert payload["provider_mode"] == "repo_local"
    assert payload["baseline_policy"] == "long_corridor_direct_fallback"
    assert len(baseline_osrm.calls) == 1
    assert baseline_osrm.calls[0]["alternatives"] == 2


def test_route_ors_baseline_prefers_graph_realized_min_overlap_candidate(
    monkeypatch: pytest.MonkeyPatch,
    baseline_osrm: FakeBaselineOSRM,
) -> None:
    monkeypatch.setattr(settings, "route_ors_baseline_mode", "repo_local")
    async def _fake_graph_candidates(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:  # noqa: ARG001
        graph_candidate = _osrm_route(-1.8404, 52.4862)
        graph_candidate["geometry"]["coordinates"] = [
            [-1.8904, 52.4862],
            [-1.55, 52.35],
            [-1.10, 52.10],
            [-0.1276, 51.5072],
        ]
        graph_candidate["_graph_meta"] = {
            "road_mix_counts": {"primary": 12, "motorway": 4},
            "toll_edges": 0,
        }
        return [graph_candidate], object()

    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _fake_graph_candidates)
    monkeypatch.setattr(settings, "route_graph_fast_startup_long_corridor_bypass_km", 9_999.0)
    monkeypatch.setattr(settings, "route_graph_long_corridor_threshold_km", 9_999.0)
    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    try:
        with TestClient(app) as client:
            resp = client.post("/route/baseline/ors", json=_payload())
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_repo_local_baseline"
    assert payload["provider_mode"] == "repo_local"
    assert payload["baseline_policy"] == "graph_realized_min_overlap"
    assert payload["baseline"]["metrics"]["distance_km"] > 0
    assert baseline_osrm.calls[0]["alternatives"] is False
    assert any(call["via"] for call in baseline_osrm.calls[1:])


def test_route_ors_baseline_returns_repo_local_secondary_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "route_ors_baseline_mode", "repo_local")
    async def _fake_repo_local_seed(*, req: Any, osrm: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = req
        _ = osrm
        return _osrm_route(-1.7904, 52.3862), {
            "provider_mode": "repo_local",
            "baseline_policy": "corridor_alternative",
            "selected_distinct_corridor": True,
        }

    monkeypatch.setattr(main_module, "_fetch_repo_local_ors_baseline_seed", _fake_repo_local_seed)

    with TestClient(app) as client:
        resp = client.post("/route/baseline/ors", json=_payload(with_waypoint=True))
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_repo_local_baseline"
    assert payload["provider_mode"] == "repo_local"
    assert payload["baseline"]["metrics"]["distance_km"] > 0
    assert payload["baseline"]["geometry"]["type"] == "LineString"


def test_route_ors_baseline_is_distinct_from_quick_osrm_when_secondary_corridor_selected(
    monkeypatch: pytest.MonkeyPatch,
    baseline_osrm: FakeBaselineOSRM,
) -> None:
    monkeypatch.setattr(settings, "route_ors_baseline_mode", "repo_local")
    async def _fake_repo_local_seed(*, req: Any, osrm: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = req
        _ = osrm
        route = _osrm_route(-1.7904, 52.3862)
        route["geometry"]["coordinates"] = [
            [-1.7904, 52.3862],
            [-1.5, 52.1],
            [-0.1276, 51.5072],
        ]
        return route, {"provider_mode": "repo_local", "baseline_policy": "corridor_alternative"}

    monkeypatch.setattr(main_module, "_fetch_repo_local_ors_baseline_seed", _fake_repo_local_seed)
    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    try:
        with TestClient(app) as client:
            primary = client.post("/route/baseline", json=_payload()).json()
            secondary = client.post("/route/baseline/ors", json=_payload()).json()
    finally:
        app.dependency_overrides.clear()
    assert primary["method"] == "osrm_quick_baseline"
    assert secondary["method"] == "ors_repo_local_baseline"
    assert primary["baseline"]["geometry"]["coordinates"] != secondary["baseline"]["geometry"]["coordinates"]
