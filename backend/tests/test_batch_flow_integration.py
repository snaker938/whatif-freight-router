from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app, osrm_client
from app.metrics_store import reset_metrics
from app.settings import settings


class FakeOSRM:
    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:
        origin_lat = float(kwargs["origin_lat"])
        dest_lat = float(kwargs["dest_lat"])
        span = abs(origin_lat - dest_lat) + 0.1
        distance_m = 80_000.0 + (span * 1_000.0)
        duration_s = 4_000.0 + (span * 120.0)

        return [
            {
                "distance": distance_m,
                "duration": duration_s,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
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
        ]


def test_batch_flow_covers_manifest_artifacts_logging_and_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reset_metrics()

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))

    logged: list[dict[str, Any]] = []

    def _capture_log_event(event: str, **fields: Any) -> None:
        logged.append({"event": event, **fields})

    monkeypatch.setattr("app.main.log_event", _capture_log_event)
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            payload = {
                "pairs": [
                    {
                        "origin": {"lat": 52.4862, "lon": -1.8904},
                        "destination": {"lat": 51.5072, "lon": -0.1276},
                    },
                    {
                        "origin": {"lat": 52.52, "lon": -1.5},
                        "destination": {"lat": 51.49, "lon": -0.1},
                    },
                ],
                "vehicle_type": "rigid_hgv",
                "scenario_mode": "no_sharing",
                "max_alternatives": 3,
                "cost_toggles": {
                    "use_tolls": True,
                    "fuel_price_multiplier": 1.2,
                    "carbon_price_per_kg": 0.1,
                    "toll_cost_per_km": 0.4,
                },
                "seed": 42,
                "toggles": {"weather_enabled": False, "incidents_enabled": False},
                "model_version": "demo-model-v1",
            }

            batch_resp = client.post("/batch/pareto", json=payload)
            assert batch_resp.status_code == 200
            batch_data = batch_resp.json()
            run_id = batch_data["run_id"]
            assert len(batch_data["results"]) == 2

            manifest_resp = client.get(f"/runs/{run_id}/manifest")
            assert manifest_resp.status_code == 200
            manifest = manifest_resp.json()
            assert manifest["schema_version"] == "1.0.0"
            assert len(manifest["request"]["pairs"]) == 2
            assert manifest["reproducibility"]["seed"] == 42
            assert manifest["reproducibility"]["toggles"]["weather_enabled"] is False
            assert manifest["model_metadata"]["model_version"] == "demo-model-v1"
            assert manifest["execution"]["pair_count"] == 2
            assert manifest["execution"]["cost_toggles"]["carbon_price_per_kg"] == 0.1

            list_resp = client.get(f"/runs/{run_id}/artifacts")
            assert list_resp.status_code == 200
            artifacts_payload = list_resp.json()
            names = {item["name"] for item in artifacts_payload["artifacts"]}
            assert names == {"results.json", "results.csv", "metadata.json"}

            results_json_resp = client.get(f"/runs/{run_id}/artifacts/results.json")
            assert results_json_resp.status_code == 200
            results_json = results_json_resp.json()
            assert results_json["run_id"] == run_id
            assert len(results_json["results"]) == 2

            results_csv_resp = client.get(f"/runs/{run_id}/artifacts/results.csv")
            assert results_csv_resp.status_code == 200
            csv_text = results_csv_resp.text
            assert "pair_index,origin_lat,origin_lon" in csv_text
            assert "route_id" in csv_text

            metadata_resp = client.get(f"/runs/{run_id}/artifacts/metadata.json")
            assert metadata_resp.status_code == 200
            metadata = metadata_resp.json()
            assert metadata["run_id"] == run_id
            assert metadata["artifact_names"] == ["metadata.json", "results.csv", "results.json"]

            metrics_resp = client.get("/metrics")
            assert metrics_resp.status_code == 200
            metrics = metrics_resp.json()
            assert metrics["endpoints"]["batch_pareto"]["request_count"] == 1
            assert metrics["endpoints"]["runs_manifest_get"]["request_count"] == 1
            assert metrics["endpoints"]["runs_artifacts_list_get"]["request_count"] == 1
            assert metrics["endpoints"]["runs_artifact_get"]["request_count"] == 3

    finally:
        app.dependency_overrides.clear()

    events = [entry["event"] for entry in logged]
    assert "batch_pareto_request" in events
    assert "run_artifacts_list" in events
    assert "run_artifact_get" in events
