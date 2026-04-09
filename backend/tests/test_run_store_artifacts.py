from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.run_store import (
    artifact_paths_for_run,
    schema_version_for_artifact,
    write_json_artifact,
    write_jsonl_artifact,
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
    results_payload = json.loads(artifacts["results.json"].read_text(encoding="utf-8"))
    metadata_payload = json.loads(artifacts["metadata.json"].read_text(encoding="utf-8"))
    assert results_payload["schema_version"] == "1.0.0"
    assert metadata_payload["schema_version"] == "1.0.0"
    assert results_payload["results"] == []
    assert metadata_payload["pair_count"] == 1
    assert results_payload["provenance"]["artifact_name"] == "results.json"
    assert metadata_payload["provenance"]["artifact_name"] == "metadata.json"
    assert results_payload["provenance"]["artifact_kind"] == "results_payload"
    assert metadata_payload["provenance"]["artifact_kind"] == "metadata_payload"
    assert results_payload["provenance"]["run_id"] == "run_1"
    assert metadata_payload["provenance"]["run_id"] == "run_1"
    assert results_payload["provenance"]["cache_honesty"]["mode"] == "unspecified"
    assert results_payload["provenance"]["cache_honesty"]["cache_reused"] is None
    csv_text = artifacts["results.csv"].read_text(encoding="utf-8")
    assert "pair_index" in csv_text
    assert "route_id" in csv_text


@pytest.mark.parametrize(
    ("artifact_name", "expected_version"),
    [
        ("dccs_candidates.jsonl", "0.1.0"),
        ("dccs_summary.json", "0.1.0"),
        ("decision_package.json", "0.1.0"),
        ("refined_routes.jsonl", "0.1.0"),
        ("results.json", "1.0.0"),
        ("metadata.json", "1.0.0"),
        ("strict_frontier.jsonl", "0.1.0"),
        ("winner_summary.json", "0.1.0"),
        ("certificate_summary.json", "0.1.0"),
        ("preference_summary.json", "0.1.0"),
        ("support_summary.json", "0.1.0"),
        ("route_fragility_map.json", "0.1.0"),
        ("competitor_fragility_breakdown.json", "0.1.0"),
        ("value_of_refresh.json", "0.1.0"),
        ("sampled_world_manifest.json", "0.1.0"),
        ("evidence_snapshot_manifest.json", "0.1.0"),
        ("controller_summary.json", "0.1.0"),
        ("voi_controller_trace_summary.json", "0.1.0"),
        ("voi_replay_oracle_summary.json", "0.1.0"),
        ("theorem_hook_map.json", "0.1.0"),
        ("lane_manifest.json", "0.1.0"),
        ("voi_action_trace.json", "0.1.0"),
        ("voi_controller_state.jsonl", "0.1.0"),
        ("voi_stop_certificate.json", "0.1.0"),
        ("final_route_trace.json", "0.1.0"),
        ("thesis_metrics.json", "0.1.0"),
        ("thesis_plots.json", "0.1.0"),
        ("evaluation_manifest.json", "0.1.0"),
    ],
)
def test_schema_version_for_artifact_covers_route_evidence_and_voi_surfaces(
    artifact_name: str,
    expected_version: str,
) -> None:
    assert schema_version_for_artifact(artifact_name) == expected_version


@pytest.mark.parametrize(
    ("artifact_name", "payload"),
    [
        ("dccs_summary.json", {"control_state": {"mode": "search"}, "candidate_count": 2}),
        ("decision_package.json", {"decision": {"winner": "route_a"}}),
        ("winner_summary.json", {"winner_route_id": "route_a"}),
        ("certificate_summary.json", {"selected_route_id": "route_a", "selected_certificate": 0.72}),
        ("support_summary.json", {"support": {"ambiguity_index": 0.42}}),
        ("theorem_hook_map.json", {"hooks": [{"hook_id": "winner_summary", "artifact_name": "winner_summary.json"}]}),
        ("lane_manifest.json", {"lane_id": "focused_voi_proof", "artifact_names": ["voi_action_trace.json"]}),
        ("route_fragility_map.json", {"route_a": {"stochastic": 0.16}}),
        ("competitor_fragility_breakdown.json", {"route_a": {"route_b": {"stochastic": 2}}}),
        ("value_of_refresh.json", {"selected_route_id": "route_a", "top_refresh_family": "stochastic"}),
        ("sampled_world_manifest.json", {"selected_route_id": "route_a", "world_count": 16}),
        ("evidence_snapshot_manifest.json", {"route_ids": ["route_a"], "active_families": ["stochastic"]}),
        ("voi_controller_trace_summary.json", {"trace_source": "voi_controller.run_controller", "record_count": 2}),
        ("voi_replay_oracle_summary.json", {"trace_source": "voi_controller.run_controller", "record_count": 2}),
        ("voi_action_trace.json", {"actions": [{"iteration": 0, "chosen_action": {"kind": "refine_top1_dccs"}}]}),
        ("voi_stop_certificate.json", {"stop_reason": "certified", "action_trace": []}),
        ("final_route_trace.json", {"pipeline_mode": "voi", "voi": {"stop_reason": "certified"}}),
        ("thesis_metrics.json", {"metric_family_scaffolding": {"version": "thesis_metric_family_scaffolding_v1"}}),
        ("thesis_plots.json", {"plots": [{"plot_id": "certificate_coverage"}]}),
        ("evaluation_manifest.json", {"evaluation_suite": "thesis", "lane_count": 3}),
    ],
)
def test_write_json_artifact_applies_schema_versions_to_route_evidence_and_voi_payloads(
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


@pytest.mark.parametrize(
    ("artifact_name", "row"),
    [
        ("dccs_candidates.jsonl", {"route_id": "route_a", "search_deficiency_score": 0.18}),
        ("strict_frontier.jsonl", {"route_id": "route_a", "selected": True}),
        ("support_trace.jsonl", {"family": "stochastic", "support_strength": 0.42}),
        ("controller_trace.jsonl", {"iteration": 0, "chosen_action": {"kind": "refresh_weather"}}),
        ("voi_controller_state.jsonl", {"iteration": 0, "winner_id": "route_a"}),
    ],
)
def test_write_jsonl_artifact_applies_schema_versions_and_provenance_to_registered_rows(
    tmp_path: Path,
    monkeypatch,
    artifact_name: str,
    row: dict[str, object],
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    path = write_jsonl_artifact("run_schema", artifact_name, [row])
    persisted = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert persisted["schema_version"] == "0.1.0"
    for key, value in row.items():
        assert persisted[key] == value
    assert persisted["provenance"]["artifact_name"] == artifact_name
    assert persisted["provenance"]["artifact_kind"] == "jsonl_row"
    assert persisted["provenance"]["run_id"] == "run_schema"
    assert persisted["provenance"]["row_index"] == 0
    assert persisted["provenance"]["writer"] == "write_jsonl_artifact"
    assert persisted["provenance"]["cache_honesty"]["mode"] == "unspecified"


def test_write_jsonl_artifact_preserves_existing_schema_version_and_provenance_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    path = write_jsonl_artifact(
        "run_schema",
        "support_trace.jsonl",
        [
            {
                "schema_version": "9.9.9",
                "family": "stochastic",
                "provenance": {"writer": "custom_writer", "artifact_kind": "custom_kind"},
            }
        ],
    )
    persisted = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert persisted["schema_version"] == "9.9.9"
    assert persisted["provenance"]["writer"] == "custom_writer"
    assert persisted["provenance"]["artifact_kind"] == "custom_kind"
    assert persisted["provenance"]["artifact_name"] == "support_trace.jsonl"
