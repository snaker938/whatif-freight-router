from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import CandidateDiagnostics, TerrainDiagnostics
from app.main import app, osrm_client
from app.model_data_errors import ModelDataError
from app.models import (
    CostToggles,
    EvidenceProvenance,
    EvidenceSourceRecord,
    GeoJSONLineString,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    ScenarioSummary,
    VoiStopSummary,
)
from app.run_store import artifact_dir_for_run
from app.routing_graph import GraphCandidateDiagnostics
from app.routing_osrm import OSRMError
from app.scenario import ScenarioMode
from app.settings import settings


def test_cache_key_component_handles_literals_and_models() -> None:
    assert main_module._cache_key_component("flat") == "flat"
    assert main_module._cache_key_component(3) == 3
    assert main_module._cache_key_component(CostToggles()) == CostToggles().model_dump(mode="json")


def _make_route(
    *,
    point_count: int,
    duration_s: float,
    distance_m: float,
    lon_offset: float,
) -> dict[str, Any]:
    point_count = max(2, point_count)
    coords = [
        [lon_offset + (i * 0.0001), 52.5 + (i * 0.00005)]
        for i in range(point_count)
    ]

    seg_count = point_count - 1
    seg_distance = distance_m / seg_count
    seg_duration = duration_s / seg_count

    return {
        "distance": distance_m,
        "duration": duration_s,
        "geometry": {"type": "LineString", "coordinates": coords},
        "legs": [
            {
                "annotation": {
                    "distance": [seg_distance] * seg_count,
                    "duration": [seg_duration] * seg_count,
                }
            }
        ],
    }


class FakeOSRM:
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
                "alternatives": alternatives,
                "exclude": exclude,
                "via": via,
                "max_retries": max_retries,
            }
        )

        if exclude or via:
            raise OSRMError("forced candidate failed")

        return [
            _make_route(
                point_count=1800,
                duration_s=900.0,
                distance_m=120_000.0,
                lon_offset=-1.9,
            ),
            _make_route(
                point_count=1600,
                duration_s=1050.0,
                distance_m=126_000.0,
                lon_offset=-1.7,
            ),
        ]


@pytest.fixture
def fake_osrm() -> FakeOSRM:
    return FakeOSRM()


@pytest.fixture
def client(fake_osrm: FakeOSRM, monkeypatch: pytest.MonkeyPatch):
    def _fake_graph_candidate_routes(
        *,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        max_paths: int | None = None,
    ) -> tuple[list[dict[str, Any]], GraphCandidateDiagnostics]:
        _ = (origin_lat, origin_lon, destination_lat, destination_lon, max_paths)
        return (
            [
                _make_route(
                    point_count=1800,
                    duration_s=900.0,
                    distance_m=120_000.0,
                    lon_offset=-1.9,
                ),
                _make_route(
                    point_count=1600,
                    duration_s=1050.0,
                    distance_m=126_000.0,
                    lon_offset=-1.7,
                ),
            ],
            GraphCandidateDiagnostics(
                explored_states=320,
                generated_paths=28,
                emitted_paths=2,
                candidate_budget=24,
            ),
        )

    async def _fake_collect_route_options_with_diagnostics(**kwargs: Any) -> tuple[
        list[RouteOption],
        list[str],
        int,
        TerrainDiagnostics,
        CandidateDiagnostics,
    ]:
        scenario_mode = kwargs.get("scenario_mode", ScenarioMode.NO_SHARING)
        raw_routes = [
            _make_route(
                point_count=1800,
                duration_s=900.0,
                distance_m=120_000.0,
                lon_offset=-1.9,
            ),
            _make_route(
                point_count=1600,
                duration_s=1050.0,
                distance_m=126_000.0,
                lon_offset=-1.7,
            ),
        ]

        options: list[RouteOption] = []
        for idx, route in enumerate(raw_routes, start=1):
            raw_coords = route["geometry"]["coordinates"]
            coords = [(float(pt[0]), float(pt[1])) for pt in raw_coords]
            coords = main_module._downsample_coords(coords)
            distance_km = float(route["distance"]) / 1000.0
            duration_s = float(route["duration"])
            avg_speed_kmh = distance_km / (duration_s / 3600.0)
            options.append(
                RouteOption(
                    id=f"route_{idx}",
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
            )

        return (
            options,
            [],
            len(raw_routes),
            TerrainDiagnostics(),
            CandidateDiagnostics(
                raw_count=len(raw_routes),
                deduped_count=len(options),
                graph_explored_states=320,
                graph_generated_paths=28,
                graph_emitted_paths=len(options),
                candidate_budget=24,
            ),
        )

    app.dependency_overrides[osrm_client] = lambda: fake_osrm
    monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_scenario_require_url_in_strict", False)
    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(main_module, "route_graph_candidate_routes", _fake_graph_candidate_routes)
    monkeypatch.setattr(
        main_module,
        "_collect_route_options_with_diagnostics",
        _fake_collect_route_options_with_diagnostics,
    )
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _pareto_payload() -> dict[str, Any]:
    return {
        "origin": {"lat": 52.4862, "lon": -1.8904},
        "destination": {"lat": 51.5072, "lon": -0.1276},
        "vehicle_type": "rigid_hgv",
        "scenario_mode": "no_sharing",
        "max_alternatives": 5,
    }


def test_pareto_computes_all_candidates_and_downsamples_geometry(
    client: TestClient,
    fake_osrm: FakeOSRM,
) -> None:
    resp = client.post("/pareto", json=_pareto_payload())
    assert resp.status_code == 200

    data = resp.json()
    routes = data["routes"]
    assert len(routes) >= 1

    diagnostics = data.get("diagnostics", {})
    assert int(diagnostics.get("candidate_count_raw", 0)) >= 1
    assert "graph_explored_states" in diagnostics
    assert "graph_generated_paths" in diagnostics
    assert "graph_emitted_paths" in diagnostics
    assert "prefetch_failed_details" in diagnostics
    assert "prefetch_missing_expected_sources" in diagnostics
    assert "prefetch_rows_json" in diagnostics

    for route in routes:
        assert len(route["geometry"]["coordinates"]) <= 1200


def test_pareto_stream_emits_progress_events_and_final_sorted_routes(
    client: TestClient,
    fake_osrm: FakeOSRM,
) -> None:
    resp = client.post("/api/pareto/stream", json=_pareto_payload())
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert events
    assert events[0]["type"] == "meta"
    total_candidates = events[0]["total"]
    assert total_candidates >= 1
    assert events[-1]["type"] == "done"
    assert events[-1]["done"] == len(events[-1]["routes"])
    assert events[-1]["total"] == len(events[-1]["routes"])

    route_events = [event for event in events if event["type"] == "route"]
    assert len(route_events) >= 1

    for event in route_events:
        assert 0 < event["done"] <= len(events[-1]["routes"])
        assert len(event["route"]["geometry"]["coordinates"]) <= 1200

    done_event = events[-1]
    assert done_event["warning_count"] == len(done_event["warnings"])

    durations = [route["metrics"]["duration_s"] for route in done_event["routes"]]
    assert durations == sorted(durations)

    # Strict graph-native path should emit routable candidates.
    assert len(done_event["routes"]) >= 1


def test_pareto_stream_emits_stage_metadata_and_stage_logs(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_events: list[tuple[str, dict[str, Any]]] = []

    def _capture_log(event: str, **kwargs: Any) -> None:
        captured_events.append((event, kwargs))

    monkeypatch.setattr(main_module, "log_event", _capture_log)
    resp = client.post("/api/pareto/stream", json=_pareto_payload())
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    meta_events = [event for event in events if event.get("type") == "meta"]
    assert meta_events
    first_meta = meta_events[0]
    assert isinstance(first_meta.get("request_id"), str)
    assert isinstance(first_meta.get("stage"), str)
    assert "elapsed_ms" in first_meta
    assert "candidate_done" in first_meta
    assert "candidate_total" in first_meta

    stage_events = [kwargs for event, kwargs in captured_events if event == "pareto_stream_stage"]
    assert stage_events
    assert any(str(item.get("stage", "")).strip() == "collecting_candidates" for item in stage_events)
    assert any(str(item.get("stage", "")).strip() == "finalizing_pareto" for item in stage_events)


def test_pareto_stream_timeout_emits_route_compute_timeout(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "route_compute_attempt_timeout_s", 1)

    async def _slow_collect_route_options_with_diagnostics(**kwargs: Any) -> tuple[
        list[RouteOption],
        list[str],
        int,
        TerrainDiagnostics,
        CandidateDiagnostics,
    ]:
        progress_cb = kwargs.get("progress_cb")
        if callable(progress_cb):
            await progress_cb(
                {
                    "stage": "collecting_candidates",
                    "stage_detail": "simulated_slow_provider",
                    "candidate_done": 0,
                    "candidate_total": 24,
                }
            )
        await main_module.asyncio.sleep(2.0)
        return (
            [],
            [],
            0,
            TerrainDiagnostics(),
            CandidateDiagnostics(),
        )

    monkeypatch.setattr(
        main_module,
        "_collect_route_options_with_diagnostics",
        _slow_collect_route_options_with_diagnostics,
    )

    resp = client.post("/api/pareto/stream", json=_pareto_payload())
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    fatal_events = [event for event in events if event.get("type") == "fatal"]
    assert fatal_events
    fatal = fatal_events[-1]
    assert fatal.get("reason_code") == "route_compute_timeout"
    assert isinstance(fatal.get("elapsed_ms"), (int, float))
    assert isinstance(fatal.get("stage"), str)
    heartbeat_values = [
        int(event.get("heartbeat"))
        for event in events
        if event.get("type") == "meta" and isinstance(event.get("heartbeat"), int)
    ]
    assert heartbeat_values
    assert max(heartbeat_values) >= 1


def test_route_timeout_returns_route_compute_timeout(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "route_compute_single_attempt_timeout_s", 1)

    async def _slow_collect_route_options_with_diagnostics(**_: Any) -> tuple[
        list[RouteOption],
        list[str],
        int,
        TerrainDiagnostics,
        CandidateDiagnostics,
    ]:
        await main_module.asyncio.sleep(2.0)
        return (
            [],
            [],
            0,
            TerrainDiagnostics(),
            CandidateDiagnostics(),
        )

    monkeypatch.setattr(
        main_module,
        "_collect_route_options_with_diagnostics",
        _slow_collect_route_options_with_diagnostics,
    )

    resp = client.post("/route", json=_pareto_payload())
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["reason_code"] == "route_compute_timeout"
    assert detail["stage"] == "collecting_candidates"
    assert detail["stage_detail"] == "attempt_timeout_reached"
    assert int(detail["timeout_s"]) == 1


def test_route_uses_direct_pipeline_for_voi_mode(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    async def _fail_if_legacy_called(**kwargs: Any) -> tuple[
        list[RouteOption],
        list[str],
        int,
        TerrainDiagnostics,
        CandidateDiagnostics,
    ]:
        raise AssertionError("legacy route collector should not run for VOI pipeline mode")

    route = RouteOption(
        id="voi_route_1",
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[(-1.8904, 52.4862), (-0.1276, 51.5072)],
        ),
        metrics=RouteMetrics(
            distance_km=185.0,
            duration_s=8100.0,
            monetary_cost=312.4,
            emissions_kg=144.2,
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
            route_id="voi_route_1",
            certificate=0.86,
            certified=True,
            threshold=0.8,
            active_families=["scenario", "weather"],
            top_fragility_families=["weather"],
            top_competitor_route_id="voi_route_2",
            top_value_of_refresh_family="weather",
        ),
    )

    async def _fake_compute_direct_route_pipeline(**kwargs: Any) -> dict[str, Any]:
        return {
            "selected": route,
            "candidates": [route],
            "warnings": [],
            "candidate_fetches": 1,
            "terrain_diag": TerrainDiagnostics(),
            "candidate_diag": CandidateDiagnostics(raw_count=1, deduped_count=1, candidate_budget=1),
            "selected_certificate": route.certification,
            "voi_stop_summary": VoiStopSummary(
                final_route_id="voi_route_1",
                certificate=0.86,
                certified=True,
                iteration_count=1,
                search_budget_used=1,
                evidence_budget_used=0,
                stop_reason="certified",
                best_rejected_action="refine_top1_dccs:other",
                best_rejected_q=0.04,
            ),
            "extra_json_artifacts": {
                "certificate_summary.json": {"selected_route_id": "voi_route_1", "selected_certificate": 0.86},
                "voi_stop_certificate.json": {"final_winner_route_id": "voi_route_1", "certificate_value": 0.86},
            },
            "extra_jsonl_artifacts": {
                "dccs_candidates.jsonl": [{"candidate_id": "cand-1", "decision": "refine"}],
                "voi_controller_state.jsonl": [
                    {
                        "iteration_index": 0,
                        "search_completeness_score": 0.74,
                        "search_completeness_gap": 0.11,
                        "prior_support_strength": 0.67,
                        "pending_challenger_mass": 0.28,
                        "best_pending_flip_probability": 0.32,
                        "corridor_family_recall": 0.8,
                        "frontier_recall_at_budget": 0.76,
                        "top_refresh_gain": 0.045,
                        "top_fragility_mass": 0.018,
                        "competitor_pressure": 0.62,
                        "credible_search_uncertainty": True,
                        "credible_evidence_uncertainty": True,
                    }
                ],
            },
            "extra_csv_artifacts": {
                "voi_action_scores.csv": (
                    ["iteration", "action_id", "q_score"],
                    [{"iteration": 0, "action_id": "refine_top1_dccs:cand-1", "q_score": 0.14}],
                ),
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

    resp = client.post(
        "/route",
        json={
            **_pareto_payload(),
            "pipeline_mode": "voi",
            "search_budget": 2,
            "evidence_budget": 1,
            "cert_world_count": 16,
            "certificate_threshold": 0.8,
            "tau_stop": 0.02,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline_mode"] == "voi"
    assert data["selected"]["id"] == "voi_route_1"
    assert data["selected_certificate"]["route_id"] == "voi_route_1"
    assert data["selected_certificate"]["certified"] is True
    assert data["voi_stop_summary"]["stop_reason"] == "certified"
    assert data["run_id"]

    artifact_dir = artifact_dir_for_run(data["run_id"])
    controller_path = artifact_dir / "voi_controller_state.jsonl"
    assert controller_path.exists()
    controller_rows = [
        json.loads(line)
        for line in controller_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert controller_rows
    first_row = controller_rows[0]
    assert first_row["top_refresh_gain"] == pytest.approx(0.045, rel=0.0, abs=1e-6)
    assert first_row["top_fragility_mass"] == pytest.approx(0.018, rel=0.0, abs=1e-6)
    assert first_row["competitor_pressure"] == pytest.approx(0.62, rel=0.0, abs=1e-6)
    assert first_row["credible_search_uncertainty"] is True
    assert first_row["credible_evidence_uncertainty"] is True


def test_route_defaults_to_tri_source_public_mode_and_emits_decision_package(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))
    monkeypatch.setattr(settings, "route_pipeline_default_mode", "tri_source")
    captured: dict[str, Any] = {}

    route = RouteOption(
        id="tri_route_1",
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[(-1.8904, 52.4862), (-0.1276, 51.5072)],
        ),
        metrics=RouteMetrics(
            distance_km=184.5,
            duration_s=8040.0,
            monetary_cost=298.7,
            emissions_kg=141.8,
            avg_speed_kmh=82.6,
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
        evidence_provenance=EvidenceProvenance(
            active_families=["traffic", "weather", "stochastic"],
            families=[
                EvidenceSourceRecord(
                    family="traffic",
                    source="tomtom_live",
                    active=True,
                    freshness_timestamp_utc="2026-04-03T08:00:00Z",
                    confidence=0.94,
                    coverage_ratio=0.98,
                ),
                EvidenceSourceRecord(
                    family="weather",
                    source="metoffice_live",
                    active=True,
                    freshness_timestamp_utc="2026-04-03T08:00:00Z",
                    confidence=0.93,
                    coverage_ratio=0.97,
                ),
                EvidenceSourceRecord(
                    family="stochastic",
                    source="stochastic_model",
                    active=True,
                    freshness_timestamp_utc="2026-04-03T08:00:00Z",
                    confidence=0.9,
                    coverage_ratio=0.95,
                ),
            ],
        ),
        certification=RouteCertificationSummary(
            route_id="tri_route_1",
            certificate=0.91,
            certified=True,
            threshold=0.8,
            active_families=["traffic", "weather", "stochastic"],
            top_fragility_families=["weather"],
            top_competitor_route_id="tri_route_2",
            top_value_of_refresh_family="weather",
            ambiguity_context={
                "od_ambiguity_support_ratio": 0.88,
                "od_ambiguity_source_entropy": 0.74,
            },
        ),
    )

    async def _fake_compute_direct_route_pipeline(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "selected": route,
            "candidates": [route],
            "warnings": [],
            "candidate_fetches": 1,
            "terrain_diag": TerrainDiagnostics(),
            "candidate_diag": CandidateDiagnostics(raw_count=1, deduped_count=1, candidate_budget=1),
            "selected_certificate": route.certification,
            "voi_stop_summary": VoiStopSummary(
                final_route_id="tri_route_1",
                certificate=0.91,
                certified=True,
                iteration_count=1,
                search_budget_used=1,
                evidence_budget_used=0,
                stop_reason="certified",
                best_rejected_action="stop",
                best_rejected_q=0.0,
                search_completeness_score=0.93,
                search_completeness_gap=0.02,
                credible_search_uncertainty=True,
            ),
            "extra_json_artifacts": {
                "certificate_summary.json": {
                    "selected_route_id": "tri_route_1",
                    "selected_certificate": 0.91,
                    "selected_certificate_basis": "threshold_and_pairwise",
                    "route_certificates": {"tri_route_1": 0.91},
                    "frontier_route_ids": ["tri_route_1"],
                },
                "route_fragility_map.json": {
                    "top_refresh_family": "weather",
                    "controller_ranking_basis": "pairwise_gap",
                },
                "sampled_world_manifest.json": {
                    "manifest_hash": "sha256:test-worlds",
                    "selected_route_id": "tri_route_1",
                    "world_count": 16,
                    "requested_world_count": 16,
                    "unique_world_count": 16,
                    "world_reuse_rate": 0.0,
                    "worlds": [
                        {
                            "target_route_id": "tri_route_1",
                            "states": {
                                "traffic": "nominal",
                                "weather": "nominal",
                                "stochastic": "nominal",
                            },
                        }
                    ],
                },
                "evidence_snapshot_manifest.json": {
                    "manifest_hash": "sha256:test-evidence",
                    "family_snapshots": {
                        "traffic": [
                            {"source": "tomtom_live", "confidence": 0.94, "coverage_ratio": 0.98}
                        ],
                        "weather": [
                            {"source": "metoffice_live", "confidence": 0.93, "coverage_ratio": 0.97}
                        ],
                        "stochastic": [
                            {"source": "stochastic_model", "confidence": 0.9, "coverage_ratio": 0.95}
                        ],
                    },
                },
                "voi_action_trace.json": {
                    "pipeline_mode": "tri_source",
                    "selected_route_id": "tri_route_1",
                    "actions": [{"kind": "stop", "action_id": "stop", "q_score": 0.0}],
                },
                "voi_stop_certificate.json": {
                    "pipeline_mode": "tri_source",
                    "final_winner_route_id": "tri_route_1",
                    "certificate_value": 0.91,
                },
                "final_route_trace.json": {"pipeline_mode": "tri_source", "artifact_pointers": {}},
            },
            "extra_jsonl_artifacts": {
                "strict_frontier.jsonl": [
                    {
                        "route_id": "tri_route_1",
                        "selected": True,
                        "certificate": 0.91,
                        "certificate_threshold": 0.8,
                        "monetary_cost": 298.7,
                        "duration_s": 8040.0,
                        "emissions_kg": 141.8,
                    }
                ],
                "voi_controller_state.jsonl": [{"iteration_index": 0, "pipeline_mode": "tri_source"}],
            },
            "extra_csv_artifacts": {},
        }

    monkeypatch.setattr(
        main_module,
        "_compute_direct_route_pipeline",
        _fake_compute_direct_route_pipeline,
    )

    resp = client.post("/route", json=_pareto_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert captured["pipeline_mode"] == "tri_source"
    assert data["pipeline_mode"] == "tri_source"
    assert data["decision_package"]["pipeline_mode"] == "tri_source"

    artifact_dir = artifact_dir_for_run(data["run_id"])
    decision_package = json.loads((artifact_dir / "decision_package.json").read_text(encoding="utf-8"))
    assert decision_package["pipeline_mode"] == "tri_source"
    final_trace = json.loads((artifact_dir / "final_route_trace.json").read_text(encoding="utf-8"))
    assert final_trace["pipeline_mode"] == "tri_source"
    assert final_trace["artifact_pointers"]["decision_package"] == "decision_package.json"
    assert (artifact_dir / "support_summary.json").exists()
    assert (artifact_dir / "certified_set.json").exists()


def test_route_legacy_persists_strict_frontier_and_final_trace(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    resp = client.post(
        "/route",
        json={
            **_pareto_payload(),
            "pipeline_mode": "legacy",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    run_id = data["run_id"]
    artifact_dir = artifact_dir_for_run(run_id)

    frontier_path = artifact_dir / "strict_frontier.jsonl"
    assert frontier_path.exists()
    frontier_rows = [
        json.loads(line)
        for line in frontier_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert frontier_rows
    assert any(row.get("route_id") == data["selected"]["id"] for row in frontier_rows)
    assert any(bool(row.get("selected")) for row in frontier_rows)
    assert all("route_signature" in row for row in frontier_rows)
    assert all("candidate_ids" in row for row in frontier_rows)

    trace_path = artifact_dir / "final_route_trace.json"
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["pipeline_mode"] == "legacy"
    assert trace["selected_route_id"] == data["selected"]["id"]
    assert isinstance(trace.get("stage_timings_ms"), dict)
    assert trace["counts"]["strict_frontier"] == len(frontier_rows)
    assert trace["artifact_pointers"]["strict_frontier"] == "strict_frontier.jsonl"

    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "strict_frontier.jsonl" in metadata["artifact_names"]
    assert "final_route_trace.json" in metadata["artifact_names"]


def test_pareto_timeout_returns_route_compute_timeout(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "route_compute_attempt_timeout_s", 1)

    async def _slow_collect_route_options_with_diagnostics(**_: Any) -> tuple[
        list[RouteOption],
        list[str],
        int,
        TerrainDiagnostics,
        CandidateDiagnostics,
    ]:
        await main_module.asyncio.sleep(2.0)
        return (
            [],
            [],
            0,
            TerrainDiagnostics(),
            CandidateDiagnostics(),
        )

    monkeypatch.setattr(
        main_module,
        "_collect_route_options_with_diagnostics",
        _slow_collect_route_options_with_diagnostics,
    )

    resp = client.post("/pareto", json=_pareto_payload())
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["reason_code"] == "route_compute_timeout"
    assert detail["stage"] == "collecting_candidates"
    assert detail["stage_detail"] == "attempt_timeout_reached"
    assert int(detail["timeout_s"]) == 1


def test_route_failfast_when_graph_warming_up(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "route_graph_warmup_failfast", True)
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "loading",
            "elapsed_ms": 1500.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": None,
            "last_error": None,
        },
    )
    resp = client.post("/route", json=_pareto_payload())
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["reason_code"] == "routing_graph_warming_up"
    assert detail["stage_detail"] == "routing_graph_warming_up"


def test_pareto_failfast_when_graph_warming_up(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "route_graph_warmup_failfast", True)
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "loading",
            "elapsed_ms": 900.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": None,
            "last_error": None,
        },
    )
    resp = client.post("/pareto", json=_pareto_payload())
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["reason_code"] == "routing_graph_warming_up"


def test_pareto_stream_failfast_when_graph_warming_up(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "route_graph_warmup_failfast", True)
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "loading",
            "elapsed_ms": 800.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": None,
            "last_error": None,
        },
    )
    resp = client.post("/api/pareto/stream", json=_pareto_payload())
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    fatal_events = [event for event in events if event.get("type") == "fatal"]
    assert fatal_events
    assert fatal_events[-1]["reason_code"] == "routing_graph_warming_up"


def test_health_ready_reports_route_graph_state(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(
        main_module,
        "_strict_live_readiness_status",
        lambda: {
            "ok": True,
            "status": "ok",
            "reason_code": "ok",
            "message": "Strict live scenario coefficients are ready.",
            "as_of_utc": "2026-02-26T23:00:00+00:00",
            "age_minutes": 42.0,
            "max_age_minutes": 4320,
            "checked_at_utc": "2026-02-26T23:42:00+00:00",
        },
    )
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "ready",
            "elapsed_ms": 2000.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": "2026-02-25T00:00:02Z",
            "last_error": None,
        },
    )
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ready"
    assert payload["strict_route_ready"] is True
    assert payload["recommended_action"] == "ready"
    assert payload["route_graph"]["state"] == "ready"
    assert payload["strict_live"]["ok"] is True
    assert payload["strict_live"]["status"] == "ok"


def test_health_ready_reports_not_ready_when_strict_live_stale(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(
        main_module,
        "_strict_live_readiness_status",
        lambda: {
            "ok": False,
            "status": "stale",
            "reason_code": "scenario_profile_unavailable",
            "message": "Live scenario coefficient payload is stale for strict runtime policy.",
            "as_of_utc": "2026-02-25T23:26:54+00:00",
            "age_minutes": 1460.16,
            "max_age_minutes": 1440,
            "checked_at_utc": "2026-02-26T23:42:44+00:00",
        },
    )
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "ready",
            "elapsed_ms": 2000.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": "2026-02-25T00:00:02Z",
            "last_error": None,
        },
    )
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "not_ready"
    assert payload["strict_route_ready"] is False
    assert payload["recommended_action"] == "refresh_live_sources"
    assert payload["route_graph"]["ok"] is True
    assert payload["strict_live"]["ok"] is False
    assert payload["strict_live"]["status"] == "stale"


def test_health_ready_reports_ready_when_graph_and_strict_live_ok(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(
        main_module,
        "_strict_live_readiness_status",
        lambda: {
            "ok": True,
            "status": "ok",
            "reason_code": "ok",
            "message": "Strict live scenario coefficients are ready.",
            "as_of_utc": "2026-02-26T20:00:00+00:00",
            "age_minutes": 30.0,
            "max_age_minutes": 4320,
            "checked_at_utc": "2026-02-26T20:30:00+00:00",
        },
    )
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "ready",
            "elapsed_ms": 2000.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": "2026-02-25T00:00:02Z",
            "last_error": None,
        },
    )
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ready"
    assert payload["strict_route_ready"] is True
    assert payload["recommended_action"] == "ready"
    assert payload["strict_live"]["ok"] is True


def test_strict_live_readiness_status_uses_uncached_dependency_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"scenario_cached": 0, "scenario_uncached": 0, "fuel_cached": 0, "fuel_uncached": 0}

    def _scenario_uncached():
        state["scenario_uncached"] += 1
        return SimpleNamespace(
            as_of_utc="2026-03-28T12:00:00Z",
            generated_at_utc="2026-03-28T12:00:00Z",
            source="live",
            contexts={"ctx": {}},
        )

    def _scenario_cached():
        state["scenario_cached"] += 1
        return SimpleNamespace(
            as_of_utc="2026-03-28T12:00:00Z",
            generated_at_utc="2026-03-28T12:00:00Z",
            source="cached",
            contexts={"ctx": {}},
        )

    _scenario_cached.__wrapped__ = _scenario_uncached

    def _fuel_uncached(*args, **kwargs):  # noqa: ARG001
        state["fuel_uncached"] += 1
        raise ModelDataError(
            reason_code="fuel_price_source_unavailable",
            message="stale fuel snapshot",
            details={"as_of_utc": "2026-03-16T00:00:00Z", "max_age_days": 7},
        )

    def _fuel_cached(*args, **kwargs):  # noqa: ARG001
        state["fuel_cached"] += 1
        return SimpleNamespace(source="cached", as_of="2026-03-16T00:00:00Z", signature="cached")

    _fuel_cached.__wrapped__ = _fuel_uncached

    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_route_compute_require_all_expected", True)
    monkeypatch.setattr(settings, "live_runtime_data_enabled", True)
    monkeypatch.setattr(main_module, "load_scenario_profiles", _scenario_cached)
    monkeypatch.setattr(main_module, "load_fuel_price_snapshot", _fuel_cached)
    monkeypatch.setattr(
        main_module,
        "load_toll_tariffs",
        lambda: SimpleNamespace(source="live", rules=[{"id": "rule_1"}]),
    )
    monkeypatch.setattr(main_module, "load_toll_segments_seed", lambda: ({"id": "seg_1"},))
    monkeypatch.setattr(
        main_module,
        "load_stochastic_regimes",
        lambda: SimpleNamespace(source="live", regimes={"weekday": {}}),
    )
    monkeypatch.setattr(
        main_module,
        "load_departure_profile",
        lambda: SimpleNamespace(source="live", profiles={"uk_default": {}}),
    )
    monkeypatch.setattr(main_module, "load_uk_bank_holidays", lambda: frozenset({"2026-12-25"}))

    readiness = main_module._strict_live_readiness_status()

    assert readiness["ok"] is False
    assert readiness["reason_code"] == "fuel_price_source_unavailable"
    assert readiness["dependency"] == "fuel_snapshot"
    assert state["scenario_uncached"] == 1
    assert state["scenario_cached"] == 0
    assert state["fuel_uncached"] == 1
    assert state["fuel_cached"] == 0


def test_route_failfast_when_graph_warmup_failed(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "route_graph_warmup_failfast", True)
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "failed",
            "phase": "parsing_nodes",
            "elapsed_ms": 1_250_000.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": None,
            "last_error": "route_graph_warmup_timeout phase=parsing_nodes timeout_s=1200",
            "asset_path": "backend/out/model_assets/routing_graph_uk.json",
            "asset_size_mb": 4122.5,
        },
    )
    resp = client.post("/route", json=_pareto_payload())
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["reason_code"] == "routing_graph_warmup_failed"
    assert detail["stage_detail"] == "routing_graph_warmup_failed"
    assert "retry_hint" in detail


def test_pareto_stream_failfast_when_graph_warmup_failed(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "route_graph_warmup_failfast", True)
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "failed",
            "phase": "parsing_edges",
            "elapsed_ms": 1_250_000.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": None,
            "last_error": "route_graph_warmup_timeout phase=parsing_edges timeout_s=1200",
            "asset_path": "backend/out/model_assets/routing_graph_uk.json",
            "asset_size_mb": 4122.5,
        },
    )
    resp = client.post("/api/pareto/stream", json=_pareto_payload())
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    fatal_events = [event for event in events if event.get("type") == "fatal"]
    assert fatal_events
    assert fatal_events[-1]["reason_code"] == "routing_graph_warmup_failed"
    assert fatal_events[-1].get("stage_detail") == "routing_graph_warmup_failed"


def test_health_ready_loading_skips_graph_status_call(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_status_call() -> tuple[bool, str]:
        raise AssertionError("graph status should not be called while warmup is loading")

    monkeypatch.setattr(main_module, "_route_graph_status_async", _raise_status_call)
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "loading",
            "phase": "parsing_nodes",
            "elapsed_ms": 1500.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": None,
            "last_error": None,
        },
    )
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "not_ready"
    assert payload["recommended_action"] == "wait"
    assert payload["route_graph"]["status"] == "warming_up"


def test_health_ready_failed_reports_rebuild_action(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_status_call() -> tuple[bool, str]:
        raise AssertionError("graph status should not be called while warmup is failed")

    monkeypatch.setattr(main_module, "_route_graph_status_async", _raise_status_call)
    monkeypatch.setattr(
        main_module,
        "route_graph_warmup_status",
        lambda: {
            "state": "failed",
            "phase": "parsing_edges",
            "elapsed_ms": 1_250_000.0,
            "started_at_utc": "2026-02-25T00:00:00Z",
            "ready_at_utc": None,
            "last_error": "route_graph_warmup_timeout",
        },
    )
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "not_ready"
    assert payload["recommended_action"] == "rebuild_graph"
    assert payload["route_graph"]["status"] == "warmup_failed"
