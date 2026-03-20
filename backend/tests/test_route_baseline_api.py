from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app, osrm_client
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
        return [_osrm_route(origin_lon, origin_lat)]


class FailingBaselineOSRM(FakeBaselineOSRM):
    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:  # type: ignore[override]
        _ = kwargs
        raise OSRMError("forced baseline failure")


def _payload(with_waypoint: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "origin": {"lat": 52.4862, "lon": -1.8904},
        "destination": {"lat": 51.5072, "lon": -0.1276},
        "vehicle_type": "rigid_hgv",
    }
    if with_waypoint:
        body["waypoints"] = [{"lat": 52.0, "lon": -1.2, "label": "via-1"}]
    return body


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


def test_route_ors_baseline_requires_provider_key_when_proxy_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ors_directions_api_key", "")
    monkeypatch.setattr(settings, "route_ors_baseline_allow_proxy_fallback", False)
    with TestClient(app) as client:
        resp = client.post("/route/baseline/ors", json=_payload())
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["reason_code"] == "baseline_provider_unconfigured"


def test_route_ors_baseline_uses_proxy_fallback_when_provider_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
    baseline_osrm: FakeBaselineOSRM,
) -> None:
    monkeypatch.setattr(settings, "ors_directions_api_key", "")
    monkeypatch.setattr(settings, "route_ors_baseline_allow_proxy_fallback", True)
    app.dependency_overrides[osrm_client] = lambda: baseline_osrm
    try:
        with TestClient(app) as client:
            resp = client.post("/route/baseline/ors", json=_payload(with_waypoint=True))
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_proxy_baseline"
    assert payload["baseline"]["metrics"]["distance_km"] > 0
    assert payload["baseline"]["geometry"]["type"] == "LineString"
    assert len(baseline_osrm.calls) == 1
    assert baseline_osrm.calls[0]["alternatives"] is False
    assert baseline_osrm.calls[0]["via"] == [(52.0, -1.2)]


def test_route_ors_baseline_returns_reference_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ors_directions_api_key", "dummy-key")

    async def _fake_ors_seed(*, req: Any) -> dict[str, Any]:
        _ = req
        return _osrm_route(-1.8904, 52.4862)

    monkeypatch.setattr(main_module, "_fetch_ors_reference_route_seed", _fake_ors_seed)

    with TestClient(app) as client:
        resp = client.post("/route/baseline/ors", json=_payload(with_waypoint=True))
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["method"] == "ors_reference"
    assert payload["baseline"]["metrics"]["distance_km"] > 0
    assert payload["baseline"]["geometry"]["type"] == "LineString"
