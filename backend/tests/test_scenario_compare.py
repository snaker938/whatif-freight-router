from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app, osrm_client
from app.settings import settings


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


def test_scenario_compare_manifest_and_signature(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()

    try:
        with TestClient(app) as client:
            payload = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "vehicle_type": "rigid_hgv",
                "max_alternatives": 3,
                "pareto_method": "epsilon_constraint",
                "epsilon": {"duration_s": 10000, "monetary_cost": 5000, "emissions_kg": 5000},
                "departure_time_utc": "2026-02-12T08:30:00Z",
            }
            compare_resp = client.post("/scenario/compare", json=payload)
            assert compare_resp.status_code == 200
            compare_data = compare_resp.json()
            run_id = compare_data["run_id"]

            assert len(compare_data["results"]) == 3
            modes = [item["scenario_mode"] for item in compare_data["results"]]
            assert modes == ["no_sharing", "partial_sharing", "full_sharing"]

            deltas = compare_data["deltas"]
            assert deltas["no_sharing"]["duration_s_delta"] == 0.0
            assert deltas["full_sharing"]["duration_s_delta"] < 0.0

            manifest_resp = client.get(f"/runs/{run_id}/scenario-manifest")
            assert manifest_resp.status_code == 200
            manifest = manifest_resp.json()
            assert manifest["type"] == "scenario_compare"
            assert manifest["execution"]["scenario_count"] == 3
            assert "signature" in manifest

            signature_resp = client.get(f"/runs/{run_id}/scenario-signature")
            assert signature_resp.status_code == 200
            signature = signature_resp.json()["signature"]["signature"]

            unsigned = dict(manifest)
            unsigned.pop("signature", None)

            verify_resp = client.post(
                "/verify/signature",
                json={"payload": unsigned, "signature": signature},
            )
            assert verify_resp.status_code == 200
            assert verify_resp.json()["valid"] is True
    finally:
        app.dependency_overrides.clear()

