from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app, osrm_client
from app.routing_graph import GraphCandidateDiagnostics
from app.routing_osrm import OSRMError


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

    app.dependency_overrides[osrm_client] = lambda: fake_osrm
    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(main_module, "route_graph_candidate_routes", _fake_graph_candidate_routes)
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
