from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import CandidateDiagnostics, TerrainDiagnostics
from app.main import app, osrm_client
from app.models import GeoJSONLineString, RouteMetrics, RouteOption, ScenarioSummary
from app.routing_graph import GraphCandidateDiagnostics
from app.routing_osrm import OSRMError
from app.scenario import ScenarioMode
from app.settings import settings


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

    resp = client.post("/route", json=_pareto_payload())
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["reason_code"] == "route_compute_timeout"
    assert detail["stage"] == "collecting_candidates"
    assert detail["stage_detail"] == "attempt_timeout_reached"
    assert int(detail["timeout_s"]) == 1


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
