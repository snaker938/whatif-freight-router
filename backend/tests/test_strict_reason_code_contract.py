from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import (
    TerrainDiagnostics,
    app,
    osrm_client,
)


class DummyOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return []


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[osrm_client] = lambda: DummyOSRM()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def _base_od_payload() -> dict[str, Any]:
    return {
        "origin": {"lat": 52.4862, "lon": -1.8904},
        "destination": {"lat": 51.5072, "lon": -0.1276},
        "vehicle_type": "rigid_hgv",
        "scenario_mode": "no_sharing",
        "max_alternatives": 3,
    }


async def _collect_toll_failure(**_: Any) -> tuple[list[Any], list[str], int, TerrainDiagnostics]:
    return (
        [],
        ["route_0: toll_pricing_unresolved (mock unresolved tariff)"],
        0,
        TerrainDiagnostics(),
    )


async def _collect_terrain_region_failure(**_: Any) -> tuple[list[Any], list[str], int, TerrainDiagnostics]:
    return (
        [],
        ["route_0: terrain_region_unsupported (mock unsupported region)"],
        0,
        TerrainDiagnostics(unsupported_region_count=1, dem_version="uk_dem_v4"),
    )


async def _collect_terrain_asset_failure(**_: Any) -> tuple[list[Any], list[str], int, TerrainDiagnostics]:
    return (
        [],
        ["route_0: terrain_dem_asset_unavailable (mock missing assets)"],
        0,
        TerrainDiagnostics(asset_unavailable_count=1, dem_version="missing"),
    )


async def _collect_risk_normalization_failure(**_: Any) -> tuple[list[Any], list[str], int, TerrainDiagnostics]:
    return (
        [],
        ["route_0: risk_normalization_unavailable (mock missing refs)"],
        0,
        TerrainDiagnostics(),
    )


async def _collect_risk_prior_failure(**_: Any) -> tuple[list[Any], list[str], int, TerrainDiagnostics]:
    return (
        [],
        ["route_0: risk_prior_unavailable (mock missing priors)"],
        0,
        TerrainDiagnostics(),
    )


async def _collect_fuel_source_failure(**_: Any) -> tuple[list[Any], list[str], int, TerrainDiagnostics]:
    return (
        [],
        ["route_0: fuel_price_source_unavailable (mock stale live and fallback unavailable)"],
        0,
        TerrainDiagnostics(),
    )


async def _collect_vehicle_profile_failure(**_: Any) -> tuple[list[Any], list[str], int, TerrainDiagnostics]:
    return (
        [],
        ["route_0: vehicle_profile_unavailable (mock unknown vehicle profile)"],
        0,
        TerrainDiagnostics(),
    )


async def _collect_scenario_profile_failure(**_: Any) -> tuple[list[Any], list[str], int, TerrainDiagnostics]:
    return (
        [],
        ["route_0: scenario_profile_unavailable (mock missing scenario profile asset)"],
        0,
        TerrainDiagnostics(),
    )


def test_route_strict_error_uses_reason_code(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_toll_failure)
    response = client.post("/route", json=_base_od_payload())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "toll_tariff_unresolved"
    assert isinstance(detail["warnings"], list)
    assert detail["warnings"]


def test_route_strict_error_uses_terrain_region_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_terrain_region_failure)
    response = client.post("/route", json=_base_od_payload())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "terrain_region_unsupported"
    assert detail["terrain_dem_version"] == "uk_dem_v4"


def test_route_strict_error_uses_terrain_asset_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_terrain_asset_failure)
    response = client.post("/route", json=_base_od_payload())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "terrain_dem_asset_unavailable"
    assert detail["terrain_dem_version"] == "missing"


def test_route_strict_error_uses_risk_normalization_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_risk_normalization_failure)
    response = client.post("/route", json=_base_od_payload())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "risk_normalization_unavailable"


def test_route_strict_error_uses_risk_prior_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_risk_prior_failure)
    response = client.post("/route", json=_base_od_payload())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "risk_prior_unavailable"


def test_route_strict_error_uses_fuel_source_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_fuel_source_failure)
    response = client.post("/route", json=_base_od_payload())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "fuel_price_source_unavailable"


def test_route_strict_error_uses_vehicle_profile_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_vehicle_profile_failure)
    response = client.post("/route", json=_base_od_payload())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "vehicle_profile_unavailable"


def test_route_strict_error_uses_scenario_profile_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_scenario_profile_failure)
    response = client.post("/route", json=_base_od_payload())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "scenario_profile_unavailable"


def test_pareto_stream_multileg_fatal_has_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_toll_failure)
    payload = _base_od_payload()
    payload["waypoints"] = [{"lat": 52.0, "lon": -1.4}]
    response = client.post("/api/pareto/stream", json=payload)
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    fatal = next(event for event in events if event.get("type") == "fatal")
    assert fatal["reason_code"] == "toll_tariff_unresolved"
    assert "message" in fatal and isinstance(fatal["message"], str)


def test_pareto_stream_direct_fatal_has_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_toll_failure)

    response = client.post("/api/pareto/stream", json=_base_od_payload())
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    fatal = next(event for event in events if event.get("type") == "fatal")
    assert fatal["reason_code"] == "toll_tariff_unresolved"


def test_pareto_stream_direct_fatal_has_vehicle_profile_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_vehicle_profile_failure)
    response = client.post("/api/pareto/stream", json=_base_od_payload())
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    fatal = next(event for event in events if event.get("type") == "fatal")
    assert fatal["reason_code"] == "vehicle_profile_unavailable"


def test_pareto_stream_direct_fatal_has_scenario_profile_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_scenario_profile_failure)
    response = client.post("/api/pareto/stream", json=_base_od_payload())
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    fatal = next(event for event in events if event.get("type") == "fatal")
    assert fatal["reason_code"] == "scenario_profile_unavailable"


def test_departure_optimize_surfaces_strict_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_toll_failure)
    payload = {
        **_base_od_payload(),
        "window_start_utc": "2026-02-12T02:00:00Z",
        "window_end_utc": "2026-02-12T04:00:00Z",
        "step_minutes": 60,
    }
    response = client.post("/departure/optimize", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "toll_tariff_unresolved"


def test_pareto_surfaces_strict_reason_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_toll_failure)
    response = client.post("/pareto", json=_base_od_payload())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason_code"] == "toll_tariff_unresolved"


def test_scenario_compare_per_item_errors_are_reason_coded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_toll_failure)
    response = client.post("/scenario/compare", json=_base_od_payload())
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    for result in payload["results"]:
        assert isinstance(result.get("error"), str)
        assert result["error"].startswith("reason_code:toll_tariff_unresolved;")


def test_scenario_compare_per_item_errors_are_scenario_reason_coded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_scenario_profile_failure)
    response = client.post("/scenario/compare", json=_base_od_payload())
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    for result in payload["results"]:
        assert isinstance(result.get("error"), str)
        assert result["error"].startswith("reason_code:scenario_profile_unavailable;")


def test_duty_chain_per_leg_errors_are_reason_coded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_toll_failure)
    payload = {
        "stops": [
            {"lat": 52.4862, "lon": -1.8904, "label": "A"},
            {"lat": 51.5072, "lon": -0.1276, "label": "B"},
        ],
        "vehicle_type": "rigid_hgv",
        "scenario_mode": "no_sharing",
    }
    response = client.post("/duty/chain", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["legs"]
    assert body["legs"][0]["error"].startswith("reason_code:toll_tariff_unresolved;")


def test_duty_chain_per_leg_errors_are_scenario_reason_coded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_scenario_profile_failure)
    payload = {
        "stops": [
            {"lat": 52.4862, "lon": -1.8904, "label": "A"},
            {"lat": 51.5072, "lon": -0.1276, "label": "B"},
        ],
        "vehicle_type": "rigid_hgv",
        "scenario_mode": "no_sharing",
    }
    response = client.post("/duty/chain", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["legs"]
    assert body["legs"][0]["error"].startswith("reason_code:scenario_profile_unavailable;")


def test_batch_pareto_per_pair_errors_are_reason_coded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_toll_failure)
    payload = {
        "pairs": [
            {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
            }
        ],
        "vehicle_type": "rigid_hgv",
        "scenario_mode": "no_sharing",
    }
    response = client.post("/batch/pareto", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["error"].startswith("reason_code:toll_tariff_unresolved;")


def test_batch_pareto_per_pair_errors_are_scenario_reason_coded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "_collect_route_options", _collect_scenario_profile_failure)
    payload = {
        "pairs": [
            {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
            }
        ],
        "vehicle_type": "rigid_hgv",
        "scenario_mode": "no_sharing",
    }
    response = client.post("/batch/pareto", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["error"].startswith("reason_code:scenario_profile_unavailable;")
