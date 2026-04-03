from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.run_store import (
    artifact_paths_for_run,
    schema_version_for_artifact,
    write_json_artifact,
    write_manifest,
    write_run_artifacts,
    write_scenario_manifest,
)
from app.settings import settings


def test_run_store_writes_signed_manifests_and_artifacts_without_pdf(tmp_path: Path, monkeypatch) -> None:
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
    assert "report.pdf" not in by_name
    csv_text = artifacts["results.csv"].read_text(encoding="utf-8")
    assert "pair_index" in csv_text
    assert "route_id" in csv_text


@pytest.mark.parametrize(
    ("artifact_name", "expected_version"),
    [
        ("decision_package.json", "0.1.0"),
        ("preference_summary.json", "0.1.0"),
        ("support_summary.json", "0.1.0"),
        ("controller_summary.json", "0.1.0"),
        ("voi_action_trace.json", "0.1.0"),
        ("voi_stop_certificate.json", "0.1.0"),
        ("final_route_trace.json", "0.1.0"),
    ],
)
def test_schema_version_for_artifact_covers_decision_summary_and_voi_json_surfaces(
    artifact_name: str,
    expected_version: str,
) -> None:
    assert schema_version_for_artifact(artifact_name) == expected_version


@pytest.mark.parametrize(
    ("artifact_name", "payload"),
    [
        ("decision_package.json", {"decision": {"winner": "route_a"}}),
        ("support_summary.json", {"support": {"ambiguity_index": 0.42}}),
        ("voi_action_trace.json", {"actions": [{"iteration": 0, "chosen_action": {"kind": "refine_top1_dccs"}}]}),
        ("voi_stop_certificate.json", {"stop_reason": "certified", "action_trace": []}),
        ("final_route_trace.json", {"pipeline_mode": "voi", "voi": {"stop_reason": "certified"}}),
    ],
)
def test_write_json_artifact_applies_schema_versions_to_decision_summary_and_voi_payloads(
    tmp_path: Path,
    monkeypatch,
    artifact_name: str,
    payload: dict[str, object],
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    path = write_json_artifact("run_schema", artifact_name, payload)
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert persisted["schema_version"] == "0.1.0"
    for key, value in payload.items():
        assert persisted[key] == value
