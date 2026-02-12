from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app, osrm_client
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
