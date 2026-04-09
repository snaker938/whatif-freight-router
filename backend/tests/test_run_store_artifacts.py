from __future__ import annotations

import json
from pathlib import Path

from app.run_store import (
    artifact_paths_for_run,
    write_manifest,
    write_run_artifacts,
    write_scenario_manifest,
)
from app.settings import settings


def test_run_store_writes_signed_manifests_and_artifacts_without_pdf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    run_manifest = write_manifest(
        "run_1",
        {
            "schema_version": "test-v1",
            "type": "route",
            "request": {"vehicle_type": "rigid_hgv"},
        },
    )
    scenario_manifest = write_scenario_manifest("run_1", {"type": "scenario_compare"})
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))

    assert run_manifest.exists()
    assert scenario_manifest.exists()
    assert payload["run_id"] == "run_1"
    assert payload["schema_version"] == "test-v1"
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
    assert {
        "certificate_summary.json",
        "preference_state.json",
        "preference_query_trace.json",
        "world_support_summary.json",
        "voi_stop_certificate.json",
        "thesis_summary.json",
    }.issubset(by_name)
    assert "report.pdf" not in by_name
    csv_text = artifacts["results.csv"].read_text(encoding="utf-8")
    assert "pair_index" in csv_text
    assert "route_id" in csv_text
