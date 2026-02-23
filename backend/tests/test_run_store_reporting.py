from __future__ import annotations

import json
from pathlib import Path

from app.reporting import _report_lines, write_report_pdf
from app.run_store import (
    artifact_paths_for_run,
    write_manifest,
    write_run_artifacts,
    write_scenario_manifest,
)
from app.settings import settings


def test_run_store_writes_signed_manifests_and_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    run_manifest = write_manifest("run_1", {"type": "route", "request": {"vehicle_type": "rigid_hgv"}})
    scenario_manifest = write_scenario_manifest("run_1", {"type": "scenario_compare"})
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))

    assert run_manifest.exists()
    assert scenario_manifest.exists()
    assert payload["run_id"] == "run_1"
    assert "signature" in payload
    assert isinstance(payload["signature"], dict)

    artifacts = write_run_artifacts(
        "run_1",
        results_payload={"results": []},
        metadata_payload={"pair_count": 1},
        csv_rows=[{"pair_index": 0, "route_id": "route_0"}],
    )
    by_name = artifact_paths_for_run("run_1")

    assert artifacts["results.json"].exists()
    assert artifacts["metadata.json"].exists()
    assert artifacts["results.csv"].exists()
    assert by_name["results.csv"].exists()
    csv_text = artifacts["results.csv"].read_text(encoding="utf-8")
    assert "pair_index" in csv_text
    assert "route_id" in csv_text


def test_reporting_generates_pdf_and_human_readable_lines(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    manifest = {
        "created_at": "2026-02-23T12:00:00Z",
        "request": {"vehicle_type": "rigid_hgv", "scenario_mode": "full_sharing"},
        "execution": {"duration_ms": 12.3, "pair_count": 1, "error_count": 0},
        "signature": {"algorithm": "HMAC-SHA256", "signature": "abc123"},
    }
    metadata = {"pair_count": 1, "error_count": 0, "duration_ms": 12.3}
    results = {
        "results": [
            {
                "routes": [
                    {
                        "id": "route_1",
                        "metrics": {
                            "duration_s": 1000.0,
                            "monetary_cost": 33.5,
                            "emissions_kg": 11.2,
                        },
                    }
                ]
            }
        ]
    }

    lines = _report_lines("run_2", manifest=manifest, metadata=metadata, results=results)
    path = write_report_pdf("run_2", manifest=manifest, metadata=metadata, results=results)

    assert lines[0] == "Freight Router Run Report"
    assert any("route_1" in line for line in lines)
    assert path.exists()
    pdf_bytes = path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-1.")
