from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import CandidateDiagnostics, TerrainDiagnostics
from app.main import app, osrm_client
from app.models import GeoJSONLineString, RouteMetrics, RouteOption, ScenarioSummary
from app.scenario import ScenarioMode
from app.settings import settings
from app.signatures import sign_payload


class FakeOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "distance": 80_000.0,
                "duration": 4_000.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
                },
                "legs": [
                    {
                        "annotation": {
                            "distance": [40_000.0, 40_000.0],
                            "duration": [2_000.0, 2_000.0],
                        }
                    }
                ],
            }
        ]


def test_manifest_signature_and_verify_api(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")
    monkeypatch.setattr(settings, "strict_live_data_required", False)
    monkeypatch.setattr(settings, "live_scenario_require_url_in_strict", False)

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
                "pairs": [
                    {
                        "origin": {"lat": 52.4862, "lon": -1.8904},
                        "destination": {"lat": 51.5072, "lon": -0.1276},
                    }
                ],
                "vehicle_type": "rigid_hgv",
                "scenario_mode": "no_sharing",
            }
            run_resp = client.post("/batch/pareto", json=payload)
            assert run_resp.status_code == 200
            run_id = run_resp.json()["run_id"]

            manifest_resp = client.get(f"/runs/{run_id}/manifest")
            assert manifest_resp.status_code == 200
            manifest = manifest_resp.json()
            assert "signature" in manifest

            sig_resp = client.get(f"/runs/{run_id}/signature")
            assert sig_resp.status_code == 200
            sig_meta = sig_resp.json()["signature"]
            assert sig_meta["algorithm"] == "HMAC-SHA256"
            assert isinstance(sig_meta["signature"], str)

            unsigned_manifest = dict(manifest)
            unsigned_manifest.pop("signature", None)

            verify_ok = client.post(
                "/verify/signature",
                json={"payload": unsigned_manifest, "signature": sig_meta["signature"]},
            )
            assert verify_ok.status_code == 200
            ok_payload = verify_ok.json()
            assert ok_payload["valid"] is True
            assert ok_payload["expected_signature"] == sig_meta["signature"]

            tampered = dict(unsigned_manifest)
            execution = dict(tampered["execution"])
            execution["pair_count"] = 99
            tampered["execution"] = execution
            verify_bad = client.post(
                "/verify/signature",
                json={"payload": tampered, "signature": sig_meta["signature"]},
            )
            assert verify_bad.status_code == 200
            bad_payload = verify_bad.json()
            assert bad_payload["valid"] is False
            assert bad_payload["expected_signature"] != sig_meta["signature"]
    finally:
        app.dependency_overrides.clear()


def test_verify_signature_secret_override() -> None:
    payload = {"source": "feed-a", "value": 7}
    signature = sign_payload(payload, secret="secret-a")

    with TestClient(app) as client:
        verify_ok = client.post(
            "/verify/signature",
            json={
                "payload": payload,
                "signature": signature,
                "secret": "secret-a",
            },
        )
        assert verify_ok.status_code == 200
        assert verify_ok.json()["valid"] is True

        verify_bad = client.post(
            "/verify/signature",
            json={
                "payload": payload,
                "signature": signature,
                "secret": "secret-b",
            },
        )
        assert verify_bad.status_code == 200
        assert verify_bad.json()["valid"] is False
