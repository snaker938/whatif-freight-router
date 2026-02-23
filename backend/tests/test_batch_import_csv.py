from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app, osrm_client
from app.models import BatchParetoResponse, BatchParetoResult
from app.settings import settings


class FakeOSRM:
    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "distance": 85_000.0,
                "duration": 4_200.0,
                "geometry": {"type": "LineString", "coordinates": [[-1.89, 52.48], [-0.12, 51.50]]},
                "legs": [
                    {
                        "annotation": {
                            "distance": [42_500.0, 42_500.0],
                            "duration": [2_100.0, 2_100.0],
                        }
                    }
                ],
            }
        ]


def test_batch_import_csv_validation_error_on_missing_columns(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))

    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            bad_csv = "origin_lat,origin_lon,destination_lat\n52.4,-1.8,51.5\n"
            resp = client.post("/batch/import/csv", json={"csv_text": bad_csv})
            assert resp.status_code == 422
            assert "missing required columns" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_batch_import_csv_success(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))

    async def _fake_batch_pareto(req, _osrm, _access) -> BatchParetoResponse:
        return BatchParetoResponse(
            run_id="batch_csv_test_run",
            results=[
                BatchParetoResult(origin=pair.origin, destination=pair.destination, routes=[], error=None)
                for pair in req.pairs
            ],
        )

    monkeypatch.setattr(main_module, "batch_pareto", _fake_batch_pareto)

    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()
    try:
        with TestClient(app) as client:
            csv_text = (
                "origin_lat,origin_lon,destination_lat,destination_lon\n"
                "52.4862,-1.8904,51.5072,-0.1276\n"
                "52.5200,-1.5000,51.4900,-0.1000\n"
            )
            resp = client.post(
                "/batch/import/csv",
                json={
                    "csv_text": csv_text,
                    "vehicle_type": "rigid_hgv",
                    "scenario_mode": "no_sharing",
                    "max_alternatives": 3,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["results"]) == 2
            assert data["run_id"] == "batch_csv_test_run"
    finally:
        app.dependency_overrides.clear()
