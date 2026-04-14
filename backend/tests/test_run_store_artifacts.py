from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from app.evidence_certification import (
    build_competitor_fragility_artifact_payload,
    build_route_fragility_artifact_payload,
    build_sampled_world_manifest_artifact_payload,
    compute_certificate,
    compute_fragility_maps,
    project_refc_scaffold_states,
)
from app.main import (
    CandidateDiagnostics,
    _assemble_decision_package,
    _build_preference_query_trace_payload,
    _build_route_artifact_summaries,
    _normalize_public_certified_set_summary,
    _write_route_run_bundle,
)
from app.models import (
    GeoJSONLineString,
    LatLng,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    RouteRequest,
)
from app.preference_model import build_preference_state
from app.preference_queries import PairwisePreferenceQuery
from app.preference_update import append_preference_query
from app.risk_model import build_risk_summary
from app.run_store import (
    ROUTE_ARTIFACT_IDENTITY_SCHEMA_VERSION,
    ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION,
    ROUTE_THEOREM_HOOK_SCHEMA_VERSION,
    artifact_paths_for_run,
    list_artifact_paths_for_run,
    route_artifact_theorem_hooks,
    write_csv_artifact,
    write_json_artifact,
    write_manifest,
    write_run_artifacts,
    write_scenario_manifest,
    write_text_artifact,
)
from app.settings import settings
from app.support_model import build_world_support_state
from app.uncertainty_model import build_world_bundle_summary
from app.world_policies import policy_hash as build_policy_hash


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _backend_lockfile_hash() -> str:
    return hashlib.sha256((_repo_root() / "backend" / "uv.lock").read_bytes()).hexdigest()


def _git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return str(result.stdout).strip().lower()


def _route_option(
    route_id: str,
    *,
    distance_km: float,
    duration_s: float,
    monetary_cost: float,
    emissions_kg: float,
) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(0.0, 0.0), (1.0, 1.0)]),
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            avg_speed_kmh=50.0,
        ),
    )


def _evidence_route(
    route_id: str,
    *,
    objective: tuple[float, float, float],
    evidence_tensor: dict[str, dict[str, float]],
) -> dict[str, object]:
    return {
        "route_id": route_id,
        "objective": {
            "time": float(objective[0]),
            "money": float(objective[1]),
            "co2": float(objective[2]),
        },
        "evidence": {
            family: {
                "time": float(weights.get("time", 0.0)),
                "money": float(weights.get("money", 0.0)),
                "co2": float(weights.get("co2", 0.0)),
            }
            for family, weights in evidence_tensor.items()
        },
    }


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
    assert list_artifact_paths_for_run("run_1")["results.summary.md"].exists()
    assert list_artifact_paths_for_run("run_1")["metadata.summary.md"].exists()
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

    refc_payload = {
        "schema_version": "1.0.0",
        "terminal_type": "certified_singleton",
        "selected_route_id": "route_0",
        "selected_certificate_basis": "selected_certificate",
        "world_support_summary": {
            "schema_version": "world-support-summary-v1",
            "selected_route_id": "route_0",
            "selected_certificate_basis": "selected_certificate",
            "support_state": {
                "support_bin": "supported",
                "support_flag": True,
            },
        },
    }
    certificate_summary_path = write_json_artifact(
        "run_1",
        "certificate_summary.json",
        {
            "schema_version": "1.0.0",
            "selected_route_id": "route_0",
            "selected_certificate_basis": "selected_certificate",
            "support_flag": True,
        },
    )
    decision_package_path = write_json_artifact("run_1", "decision_package.json", refc_payload)
    world_support_path = write_json_artifact("run_1", "world_support_summary.json", refc_payload["world_support_summary"])
    discovered = list_artifact_paths_for_run("run_1")

    assert certificate_summary_path.exists()
    assert decision_package_path.exists()
    assert world_support_path.exists()
    assert discovered["decision_package.json"].exists()
    assert discovered["world_support_summary.json"].exists()
    assert discovered["certificate_summary.json"].exists()
    assert discovered["decision_package.summary.md"].exists()
    assert discovered["world_support_summary.summary.md"].exists()
    assert discovered["certificate_summary.summary.md"].exists()
    assert "index.json" not in discovered
    assert "index.md" not in discovered


def test_route_compute_metadata_write_creates_and_refreshes_bundle_index(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    write_json_artifact(
        "run_route",
        "results.json",
        {
            "run_id": "run_route",
            "selected": {"id": "route_0"},
            "candidates": [],
            "warnings": [],
        },
    )
    write_json_artifact(
        "run_route",
        "decision_package.json",
        {
            "schema_version": "1.0.0",
            "terminal_type": "certified_singleton",
            "selected_route_id": "route_0",
            "selected_certificate_basis": "selected_certificate",
            "support_summary": {
                "support_flag": True,
                "support_reason": None,
            },
            "artifact_pointers": {
                "decision_package": "decision_package.json",
                "world_support_summary": "world_support_summary.json",
            },
        },
    )
    write_json_artifact(
        "run_route",
        "world_support_summary.json",
        {
            "schema_version": "world-support-summary-v1",
            "selected_route_id": "route_0",
            "selected_certificate_basis": "selected_certificate",
            "support_state": {
                "support_flag": True,
                "support_bin": "supported",
            },
            "provenance": {
                "support_reason": None,
            },
        },
    )
    write_json_artifact(
        "run_route",
        "final_route_trace.json",
        {
            "artifact_pointers": {
                "final_route_trace": "final_route_trace.json",
            },
            "route_cache_runtime": {
                "cache_hits": 1,
                "cache_misses": 3,
                "reuse_rate": 0.25,
                "last_cache_key": "route-cache-key",
            },
            "option_build_runtime": {
                "cache_hits": 2,
                "cache_misses": 2,
                "reuse_rate": 0.5,
                "last_cache_key": "option-build-key",
            },
        },
    )

    metadata_path = write_json_artifact(
        "run_route",
        "metadata.json",
        {
            "run_id": "run_route",
            "schema_version": "1.0.0",
            "type": "route_compute",
            "request_id": "req-1",
            "pipeline_mode": "voi",
            "run_seed": 7,
            "manifest_endpoint": "/runs/run_route/manifest",
            "artifacts_endpoint": "/runs/run_route/artifacts",
            "provenance_endpoint": "/runs/run_route/provenance",
            "provenance_file": str(tmp_path / "provenance" / "run_route.json"),
            "selected_route_id": "route_0",
            "candidate_count": 2,
            "warning_count": 0,
            "duration_ms": 12.5,
        },
    )

    discovered = list_artifact_paths_for_run("run_route")
    assert metadata_path.exists()
    assert discovered["index.json"].exists()
    assert discovered["index.md"].exists()

    index_payload = json.loads(discovered["index.json"].read_text(encoding="utf-8"))
    assert index_payload["schema_version"] == "run-bundle-index-v1"
    assert index_payload["run_id"] == "run_route"
    assert index_payload["bundle_type"] == "route_compute"
    assert index_payload["pipeline_mode"] == "voi"
    assert index_payload["run_seed"] == 7
    assert index_payload["selected_route_id"] == "route_0"
    assert index_payload["selected_certificate_basis"] == "selected_certificate"
    assert index_payload["terminal_type"] == "certified_singleton"
    assert index_payload["support_flag"] is True
    assert index_payload["artifact_pointers"] == {
        "decision_package": "decision_package.json",
        "final_route_trace": "final_route_trace.json",
        "world_support_summary": "world_support_summary.json",
    }
    assert "index.json" not in index_payload["artifact_names"]
    assert "index.md" not in index_payload["artifact_names"]
    assert {
        "decision_package.json",
        "final_route_trace.json",
        "metadata.json",
        "results.json",
        "world_support_summary.json",
    }.issubset(set(index_payload["artifact_names"]))
    artifact_entry = next(
        item for item in index_payload["artifacts"] if item["name"] == "decision_package.json"
    )
    assert artifact_entry["endpoint"] == "/runs/run_route/artifacts/decision_package.json"
    assert artifact_entry["relative_path"] == "decision_package.json"
    assert artifact_entry["markdown_summary_name"] == "decision_package.summary.md"
    assert artifact_entry["markdown_summary_present"] is True
    assert artifact_entry["markdown_summary_relative_path"] == "decision_package.summary.md"
    assert (
        artifact_entry["markdown_summary_endpoint"]
        == "/runs/run_route/artifacts/decision_package.summary.md"
    )
    assert artifact_entry["artifact_provenance"]["run_id"] == "run_route"
    assert artifact_entry["artifact_provenance"]["lane_id"] == "route_compute_runtime"
    assert artifact_entry["artifact_provenance"]["variant_id"] == "voi"
    assert artifact_entry["artifact_provenance"]["cache_mode"] == "mixed"
    assert artifact_entry["artifact_provenance"]["seed"] == 7
    assert artifact_entry["artifact_provenance"]["schema_version"] == "route-artifact-provenance-v1"
    assert artifact_entry["artifact_provenance"]["payload_schema_version"] == "1.0.0"
    assert artifact_entry["artifact_provenance"]["calibration_policy_version"] == "untracked"
    assert artifact_entry["artifact_provenance"]["controller_policy_version"] == "voi-controller-policy-v1"
    assert artifact_entry["artifact_provenance"]["preference_model_version"] == "preference-elicitation-policy-v1"
    assert artifact_entry["artifact_provenance"]["proxy_correction_version"] == "inactive"
    assert artifact_entry["artifact_provenance"]["support_status"] == "supported"
    assert artifact_entry["artifact_provenance"]["cache_source_id"] == "route-cache-key"
    assert artifact_entry["artifact_provenance"]["reuse_count"] == 3
    assert artifact_entry["artifact_provenance"]["invalidation_reason"] == "not_invalidated"
    assert artifact_entry["artifact_provenance"]["headline_identity"] == {
        "schema_version": "route-artifact-identity-v1",
        "git_commit_hash": _git_commit_hash(),
        "environment_lockfile_hash": _backend_lockfile_hash(),
        "environment_lockfile_path": "backend/uv.lock",
        "container_digest": None,
        "container_digest_source": "unavailable_local_runtime",
        "policy_hashes": {
            "calibration_policy_hash": None,
            "controller_policy_hash": build_policy_hash(
                "voi_controller_policy",
                version="voi-controller-policy-v1",
                configuration={
                    "lane_id": "route_compute_runtime",
                    "pipeline_mode": "voi",
                    "refinement_policy": None,
                },
            ),
            "preference_model_hash": build_policy_hash(
                "preference_model",
                version="preference-elicitation-policy-v1",
                configuration={
                    "lane_id": "route_compute_runtime",
                    "pipeline_mode": "voi",
                    "terminal_type": "certified_singleton",
                },
            ),
            "proxy_correction_hash": None,
            "probabilistic_world_policy_hash": None,
            "audit_world_policy_hash": None,
        },
    }
    assert artifact_entry["artifact_provenance"]["theorem_to_artifact_hooks"] == [
        {
            "hook_id": "decision_package_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        },
        {
            "hook_id": "terminal_outcome_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        },
    ]
    results_entry = next(item for item in index_payload["artifacts"] if item["name"] == "results.json")
    assert results_entry["artifact_provenance"]["payload_schema_version"] == "json-object-v1"
    assert index_payload["artifact_provenance_context"]["cache_source_id"] == "route-cache-key"
    assert index_payload["artifact_provenance_context"]["reuse_count"] == 3
    assert index_payload["artifact_provenance_context"]["headline_identity"]["git_commit_hash"] == _git_commit_hash()
    assert (
        index_payload["artifact_provenance_context"]["headline_identity"]["environment_lockfile_hash"]
        == _backend_lockfile_hash()
    )
    assert index_payload["artifact_provenance_schema_version"] == ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION
    assert index_payload["artifact_provenance"]["payload_schema_version"] == "run-bundle-index-v1"
    assert (
        index_payload["artifact_provenance"]["schema_version"]
        == ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION
    )
    assert (
        index_payload["artifact_provenance"]["headline_identity"]["schema_version"]
        == ROUTE_ARTIFACT_IDENTITY_SCHEMA_VERSION
    )
    assert index_payload["bundle_index_artifacts"][0]["artifact_provenance"]["run_id"] == "run_route"
    assert index_payload["theorem_to_artifact_hook_schema_version"] == ROUTE_THEOREM_HOOK_SCHEMA_VERSION
    expected_hook_map = {
        artifact_name: route_artifact_theorem_hooks(artifact_name)
        for artifact_name in index_payload["artifact_names"]
        if route_artifact_theorem_hooks(artifact_name)
    }
    assert index_payload["theorem_to_artifact_hooks"] == expected_hook_map
    for emitted_artifact in index_payload["artifacts"]:
        provenance = emitted_artifact["artifact_provenance"]
        assert provenance["schema_version"] == ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION
        assert provenance["run_id"] == "run_route"
        assert provenance["lane_id"] == "route_compute_runtime"
        assert provenance["variant_id"] == "voi"
        assert provenance["headline_identity"]["schema_version"] == ROUTE_ARTIFACT_IDENTITY_SCHEMA_VERSION
        assert provenance["headline_identity"]["git_commit_hash"] == _git_commit_hash()
        assert provenance["headline_identity"]["environment_lockfile_hash"] == _backend_lockfile_hash()
        assert provenance["theorem_to_artifact_hooks"] == route_artifact_theorem_hooks(
            emitted_artifact["name"]
        )
    for bundle_index_artifact in index_payload["bundle_index_artifacts"]:
        provenance = bundle_index_artifact["artifact_provenance"]
        assert provenance["schema_version"] == ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION
        assert provenance["run_id"] == "run_route"
        assert provenance["lane_id"] == "route_compute_runtime"
        assert provenance["headline_identity"]["schema_version"] == ROUTE_ARTIFACT_IDENTITY_SCHEMA_VERSION
        assert provenance["headline_identity"]["git_commit_hash"] == _git_commit_hash()
        assert provenance["headline_identity"]["environment_lockfile_hash"] == _backend_lockfile_hash()
        assert provenance["theorem_to_artifact_hooks"] == route_artifact_theorem_hooks(
            bundle_index_artifact["name"]
        )
    assert "decision_package.json" in index_payload["theorem_to_artifact_hooks"]
    assert index_payload["cache_summary"]["route_cache_runtime"]["reuse_rate"] == 0.25
    assert index_payload["cache_summary"]["option_build_runtime"]["reuse_rate"] == 0.5
    assert index_payload["source_artifacts_used"] == [
        "metadata.json",
        "decision_package.json",
        "final_route_trace.json",
        "world_support_summary.json",
    ]

    index_markdown = discovered["index.md"].read_text(encoding="utf-8")
    assert "# Route-Compute Bundle Index" in index_markdown
    assert "Run ID: `run_route`" in index_markdown
    assert "`decision_package` -> `decision_package.json`" in index_markdown
    assert "`decision_package.json`" in index_markdown
    assert "`decision_package.summary.md`" in index_markdown
    decision_summary = discovered["decision_package.summary.md"].read_text(encoding="utf-8")
    assert "# Decision Package Summary" in decision_summary
    assert "## Reviewer Highlights" in decision_summary
    assert "Terminal Type: `certified_singleton`" in decision_summary
    assert "## Top-Level Fields" in decision_summary

    write_json_artifact(
        "run_route",
        "certificate_summary.json",
        {
            "schema_version": "1.0.0",
            "selected_route_id": "route_0",
            "selected_certificate_basis": "selected_certificate",
        },
    )
    refreshed = json.loads(discovered["index.json"].read_text(encoding="utf-8"))
    assert "certificate_summary.json" in refreshed["artifact_names"]


def test_batch_bundle_metadata_creates_generic_bundle_index(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    artifacts = write_run_artifacts(
        "run_batch",
        results_payload={
            "run_id": "run_batch",
            "results": [],
        },
        metadata_payload={
            "run_id": "run_batch",
            "schema_version": "1.0.0",
            "type": "batch_pareto",
            "manifest_endpoint": "/runs/run_batch/manifest",
            "artifacts_endpoint": "/runs/run_batch/artifacts",
            "provenance_endpoint": "/runs/run_batch/provenance",
            "provenance_file": str(tmp_path / "provenance" / "run_batch.json"),
            "pair_count": 3,
            "error_count": 1,
            "duration_ms": 42.5,
        },
        csv_rows=[{"pair_index": 0, "route_id": "route_0"}],
    )

    discovered = list_artifact_paths_for_run("run_batch")
    assert artifacts["metadata.json"].exists()
    assert discovered["index.json"].exists()
    assert discovered["index.md"].exists()

    index_payload = json.loads(discovered["index.json"].read_text(encoding="utf-8"))
    assert index_payload["schema_version"] == "run-bundle-index-v1"
    assert index_payload["run_id"] == "run_batch"
    assert index_payload["bundle_type"] == "batch_pareto"
    assert index_payload["pair_count"] == 3
    assert index_payload["error_count"] == 1
    assert index_payload["duration_ms"] == 42.5
    assert index_payload["artifact_pointers"] == {}
    assert "results.json" in index_payload["artifact_names"]
    assert "metadata.json" in index_payload["artifact_names"]
    assert "results.csv" in index_payload["artifact_names"]
    assert "index.json" not in index_payload["artifact_names"]
    assert "index.md" not in index_payload["artifact_names"]

    index_markdown = discovered["index.md"].read_text(encoding="utf-8")
    assert "# Batch Pareto Bundle Index" in index_markdown
    assert "Pair Count: `3`" in index_markdown
    assert "Error Count: `1`" in index_markdown


def test_thesis_like_bundle_metadata_creates_bundle_index_with_export_status(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    write_json_artifact(
        "run_thesis",
        "metadata.json",
        {
            "run_id": "run_thesis",
            "schema_version": "1.0.0",
            "variant_count": 4,
            "row_count": 80,
            "evaluation_suite": {
                "role": "focused_voi_proof",
                "scope": "focused",
                "focus": "voi",
            },
            "manifest_endpoint": "/runs/run_thesis/manifest",
            "artifacts_endpoint": "/runs/run_thesis/artifacts",
            "provenance_endpoint": "/runs/run_thesis/provenance",
            "provenance_file": str(tmp_path / "provenance" / "run_thesis.json"),
            "duration_ms": 123.45,
        },
    )
    write_csv_artifact(
        "run_thesis",
        "thesis_results.csv",
        fieldnames=["variant_id", "runtime_ms"],
        rows=[{"variant_id": "C", "runtime_ms": 123.45}],
    )
    write_csv_artifact(
        "run_thesis",
        "thesis_summary.csv",
        fieldnames=["variant_id", "mean_runtime_ms"],
        rows=[{"variant_id": "C", "mean_runtime_ms": 123.45}],
    )
    write_csv_artifact(
        "run_thesis",
        "thesis_summary_by_transfer_slice.csv",
        fieldnames=["variant_id", "transfer_slice_kind", "transfer_slice_field"],
        rows=[{"variant_id": "C", "transfer_slice_kind": "leave_one_corridor_family_out", "transfer_slice_field": "corridor_bucket"}],
    )
    write_json_artifact(
        "run_thesis",
        "thesis_summary_by_transfer_slice.json",
        {
            "summary_rows": [
                {
                    "variant_id": "C",
                    "transfer_slice_kind": "leave_one_corridor_family_out",
                    "transfer_slice_field": "corridor_bucket",
                }
            ],
            "transfer_slice_kind": "leave_one_corridor_family_out",
            "slice_field": "corridor_bucket",
        },
    )
    write_csv_artifact(
        "run_thesis",
        "thesis_summary_by_weather_regime_transfer_slice.csv",
        fieldnames=["variant_id", "transfer_slice_kind", "transfer_slice_field"],
        rows=[{"variant_id": "C", "transfer_slice_kind": "leave_one_weather_regime_out", "transfer_slice_field": "weather_profile"}],
    )
    write_json_artifact(
        "run_thesis",
        "thesis_summary_by_weather_regime_transfer_slice.json",
        {
            "summary_rows": [
                {
                    "variant_id": "C",
                    "transfer_slice_kind": "leave_one_weather_regime_out",
                    "transfer_slice_field": "weather_profile",
                }
            ],
            "transfer_slice_kind": "leave_one_weather_regime_out",
            "slice_field": "weather_profile",
        },
    )
    write_text_artifact("run_thesis", "methods_appendix.md", "# methods\n")
    write_text_artifact("run_thesis", "thesis_report.md", "# report\n")
    write_json_artifact(
        "run_thesis",
        "evaluation_manifest.json",
        {
            "run_id": "run_thesis",
            "created_at": "2026-04-10T00:00:00+00:00",
        },
    )

    discovered = list_artifact_paths_for_run("run_thesis")
    assert discovered["index.json"].exists()
    assert discovered["index.md"].exists()

    index_payload = json.loads(discovered["index.json"].read_text(encoding="utf-8"))
    assert index_payload["schema_version"] == "run-bundle-index-v1"
    assert index_payload["run_id"] == "run_thesis"
    assert index_payload["bundle_type"] == "thesis_evaluation"
    assert index_payload["artifact_pointers"] == {}
    assert {
        "metadata.json",
        "thesis_results.csv",
        "thesis_summary.csv",
        "thesis_summary_by_transfer_slice.csv",
        "thesis_summary_by_weather_regime_transfer_slice.csv",
        "methods_appendix.md",
        "thesis_report.md",
        "evaluation_manifest.json",
    }.issubset(set(index_payload["artifact_names"]))

    export_status = {entry["name"]: entry for entry in index_payload["export_status"]}
    assert export_status["thesis_results.csv"]["present"] is True
    assert export_status["thesis_summary.csv"]["present"] is True
    assert export_status["thesis_summary_by_transfer_slice.csv"]["present"] is True
    assert export_status["thesis_summary_by_weather_regime_transfer_slice.csv"]["present"] is True
    assert export_status["methods_appendix.md"]["present"] is True
    assert export_status["thesis_report.md"]["present"] is True
    assert export_status["evaluation_manifest.json"]["present"] is True
    assert export_status["thesis_summary_by_cohort.csv"]["present"] is False
    assert export_status["results.csv"]["present"] is False
    assert export_status["index.json"]["present"] is True
    assert export_status["index.md"]["present"] is True
    assert "report.pdf" not in export_status

    index_markdown = discovered["index.md"].read_text(encoding="utf-8")
    assert "# Thesis Evaluation Bundle Index" in index_markdown
    assert "## Export Status" in index_markdown
    assert "`thesis_report.md`: present" in index_markdown
    assert "`results.csv`: absent" in index_markdown


def test_thesis_headline_json_artifacts_gain_provenance_and_markdown_companions(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    write_json_artifact("run_thesis", "thesis_results.json", {"rows": [{"variant_id": "A", "runtime_ms": 12.5}]})
    write_json_artifact(
        "run_thesis",
        "thesis_summary.json",
        {"summary_rows": [{"variant_id": "A", "row_count": 1, "success_rate": 1.0}]},
    )
    write_json_artifact(
        "run_thesis",
        "thesis_summary_by_cohort.json",
        {
            "summary_rows": [{"variant_id": "A", "cohort_label": "preference_sensitive", "row_count": 1}],
            "cohort_definitions": {"preference_sensitive": "Synthetic test cohort."},
        },
    )
    write_json_artifact(
        "run_thesis",
        "thesis_metrics.json",
        {"runtime_by_variant": [{"variant_id": "A", "mean_runtime_ms": 12.5}]},
    )
    write_json_artifact(
        "run_thesis",
        "thesis_plots.json",
        {"performance_vs_variant": [{"variant_id": "A", "weighted_win_rate_best_baseline": 1.0}]},
    )
    write_json_artifact(
        "run_thesis",
        "metadata.json",
        {
            "run_id": "run_thesis",
            "schema_version": "1.0.0",
            "variant_count": 4,
            "row_count": 1,
            "strict_evidence_policy": "no_synthetic_no_proxy_no_fallback",
            "ors_baseline_policy": "local_service",
            "cache_mode": "hot",
            "cache_reset_scope": "none",
            "cache_reset_policy": "none",
            "cache_carryover_expected": True,
            "strict_proxy_ors_allowed": False,
            "strict_evidence_fallbacks_allowed": False,
            "evaluation_suite": {
                "role": "focused_voi_proof",
                "scope": "focused",
                "focus": "voi",
            },
            "manifest_endpoint": "/runs/run_thesis/manifest",
            "artifacts_endpoint": "/runs/run_thesis/artifacts",
            "provenance_endpoint": "/runs/run_thesis/provenance",
            "provenance_file": str(tmp_path / "provenance" / "run_thesis.json"),
        },
    )
    write_json_artifact(
        "run_thesis",
        "evaluation_manifest.json",
        {
            "run_id": "run_thesis",
            "created_at": "2026-04-11T00:00:00+00:00",
            "model_version": "thesis-script-v3",
            "strict_evidence_policy": "no_synthetic_no_proxy_no_fallback",
            "ors_baseline_policy": "local_service",
            "ors_snapshot_mode": "off",
            "cache_mode": "hot",
            "cache_reset_scope": "none",
            "cache_reset_policy": "none",
            "evaluation_suite": {
                "role": "focused_voi_proof",
                "scope": "focused",
                "focus": "voi",
            },
        },
    )

    discovered = list_artifact_paths_for_run("run_thesis")
    expected_policy_hashes = {
        "strict_evidence_policy_hash": build_policy_hash(
            "strict_evidence_policy",
            version="no_synthetic_no_proxy_no_fallback",
            configuration={"lane_id": "focused_voi_proof"},
        ),
        "baseline_engine_policy_hash": build_policy_hash(
            "baseline_engine_policy",
            version="local_service",
            configuration={
                "lane_id": "focused_voi_proof",
                "ors_snapshot_mode": "off",
            },
        ),
        "evaluation_lane_policy_hash": build_policy_hash(
            "evaluation_lane_policy",
            version="thesis-script-v3",
            configuration={
                "role": "focused_voi_proof",
                "scope": "focused",
                "focus": "voi",
                "strict_proxy_ors_allowed": False,
                "strict_evidence_fallbacks_allowed": False,
            },
        ),
        "cache_policy_hash": build_policy_hash(
            "evaluation_cache_policy",
            version="thesis-eval-cache-policy-v1",
            configuration={
                "cache_mode": "hot",
                "cache_reset_scope": "none",
                "cache_reset_policy": "none",
                "cache_carryover_expected": True,
            },
        ),
    }
    for artifact_name in (
        "thesis_results.json",
        "thesis_summary.json",
        "thesis_summary_by_cohort.json",
        "thesis_metrics.json",
        "thesis_plots.json",
        "evaluation_manifest.json",
    ):
        payload = json.loads(discovered[artifact_name].read_text(encoding="utf-8"))
        provenance = payload["artifact_provenance"]
        identity = provenance["headline_identity"]
        summary_name = f"{Path(artifact_name).stem}.summary.md"

        assert discovered[summary_name].exists()
        assert provenance["schema_version"] == ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION
        assert provenance["run_id"] == "run_thesis"
        assert provenance["lane_id"] == "focused_voi_proof"
        assert provenance["variant_id"] == "aggregate"
        assert provenance["cache_mode"] == "hot"
        assert identity["schema_version"] == ROUTE_ARTIFACT_IDENTITY_SCHEMA_VERSION
        assert identity["git_commit_hash"] == _git_commit_hash()
        assert identity["environment_lockfile_hash"] == _backend_lockfile_hash()
        assert identity["environment_lockfile_path"] == "backend/uv.lock"
        assert "container_digest" in identity
        assert "container_digest_source" in identity
        assert identity["policy_hashes"] == expected_policy_hashes

    index_payload = json.loads(discovered["index.json"].read_text(encoding="utf-8"))
    export_status = {entry["name"]: entry for entry in index_payload["export_status"]}
    assert (
        export_status["thesis_plots.json"]["artifact_provenance"]["headline_identity"]["git_commit_hash"]
        == _git_commit_hash()
    )
    assert (
        export_status["evaluation_manifest.json"]["artifact_provenance"]["headline_identity"]["policy_hashes"]
        == expected_policy_hashes
    )


def test_route_artifact_builders_persist_compact_preference_and_support_fields(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    preference_state = append_preference_query(
        build_preference_state(
            route_ids=["route_0", "route_1"],
            weights={"time": 1.0, "money": 0.5},
            support_flag=True,
            support_reason=None,
        ),
        PairwisePreferenceQuery(preferred_route_id="route_0", challenger_route_id="route_1"),
        before_size=2,
        after_size=1,
        before_volume_proxy=1.0,
        after_volume_proxy=0.4,
        target_route_id="route_1",
        query_reason="reduce ambiguity",
    )
    support_state = build_world_support_state(
        support_score=0.88,
        support_ratio=0.7,
        support_bin="supported",
        calibration_bin="bin_2",
        support_source="unit-test",
    )
    world_bundle_summary = build_world_bundle_summary(
        manifest={
            "world_count": 4,
            "unique_world_count": 3,
            "world_reuse_rate": 0.25,
            "proxy_world_count": 3,
            "audit_world_count": 1,
            "proxy_bias_model_version": "proxy-v5",
            "audit_propensity_version": "audit-v3",
            "proxy_correction_active": True,
            "multi_fidelity_certificate_basis": "corrected_from_residual_model",
            "audit_correction_mass": 2.5,
            "audit_propensity_scores": [0.3, 0.4],
        },
        support_state=support_state,
    )
    risk_summary = build_risk_summary(
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
        distance_km=120.0,
        support_state=support_state,
        probabilistic_world_bundle=world_bundle_summary.probabilistic_world_bundle,
        audit_world_bundle=world_bundle_summary.audit_world_bundle,
    )
    world_support_summary = {
        "schema_version": "world-support-summary-v1",
        "selected_route_id": "route_0",
        "selected_certificate_basis": "selected_certificate",
        "support_flag": True,
        "support_reason": None,
        "support_state": support_state.as_dict(),
        "world_bundle_summary": world_bundle_summary.as_dict(),
        "scenario_summary": None,
        "risk_summary": risk_summary.as_dict(),
    }
    preference_summary, support_summary = _build_route_artifact_summaries(
        preference_state=preference_state,
        world_support_summary=world_support_summary,
        pipeline_mode="dccs_refc",
        selected_certificate_basis="selected_certificate",
        support_flag=True,
        support_reason=None,
        support_state=support_state,
        world_bundle_summary=world_bundle_summary,
        scenario_summary=None,
        risk_summary=risk_summary,
        abstention=None,
        selected_certificate=None,
    )
    preference_query_trace = _build_preference_query_trace_payload(
        preference_state=preference_state,
        selected_route_id="route_0",
        selected_certificate_basis="selected_certificate",
        pipeline_mode="dccs_refc",
        support_flag=True,
        support_reason=None,
    )
    selected = _route_option(
        "route_0",
        distance_km=120.0,
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
    )
    challenger = _route_option(
        "route_1",
        distance_km=123.0,
        duration_s=3720.0,
        monetary_cost=52.0,
        emissions_kg=21.5,
    )
    certified_set_summary = {
        "member_route_ids": ["route_0", "route_1"],
        "excluded_route_ids": ["route_2"],
        "exclusion_basis": [
            "outside_routes_excluded",
            "singleton_not_justified:frontier_pairwise_gap_unresolved",
        ],
        "certified": True,
        "threshold": 0.8,
        "support_flag": True,
        "outside_routes_safely_excluded": True,
        "set_size": 2,
        "witness": {
            "route_id": "route_0",
            "active_challenger_ids": ["route_1"],
            "support_flag": True,
            "singleton_not_justified_reasons": ["frontier_pairwise_gap_unresolved"],
            "outside_routes_excluded": True,
            "outside_routes_safely_excluded": True,
            "excluded_route_safety_reasons": [],
        },
    }
    route_run = _write_route_run_bundle(
        req=RouteRequest(
            origin=LatLng(lat=51.5, lon=-2.6),
            destination=LatLng(lat=51.6, lon=-2.5),
            vehicle_type="rigid_hgv",
            scenario_mode="no_sharing",
            max_alternatives=2,
        ),
        selected=selected,
        candidates=[selected, challenger],
        warnings=[],
        candidate_diag=CandidateDiagnostics(selected_candidate_count=2),
        request_id="req-route-artifact",
        pipeline_mode="dccs_refc",
        run_seed=20260410,
        duration_ms=12.5,
            extra_json_artifacts={
                "decision_package.json": {
                    "schema_version": "1.0.0",
                    "terminal_type": "certified_set",
                    "selected_route_id": "route_0",
                    "selected_certificate_basis": "selected_certificate",
                    "preference_summary": preference_summary,
                    "support_summary": support_summary,
                    "certified_set_summary": copy.deepcopy(certified_set_summary),
                },
                "preference_query_trace.json": preference_query_trace,
                "world_support_summary.json": world_support_summary,
                "certified_set_summary.json": copy.deepcopy(certified_set_summary),
            },
        )

    discovered = list_artifact_paths_for_run(str(route_run["run_id"]))
    emitted_decision = json.loads(discovered["decision_package.json"].read_text(encoding="utf-8"))
    emitted_trace = json.loads(discovered["preference_query_trace.json"].read_text(encoding="utf-8"))
    emitted_certified_set = json.loads(discovered["certified_set_summary.json"].read_text(encoding="utf-8"))
    emitted_metadata = json.loads(discovered["metadata.json"].read_text(encoding="utf-8"))
    emitted_index = json.loads(discovered["index.json"].read_text(encoding="utf-8"))
    decision_summary_text = discovered["decision_package.summary.md"].read_text(encoding="utf-8")

    assert "decision_package.json" in emitted_metadata["artifact_names"]
    assert "preference_query_trace.json" in emitted_metadata["artifact_names"]
    assert "certified_set_summary.json" in emitted_metadata["artifact_names"]
    assert emitted_metadata["artifact_provenance_context"]["run_id"] == route_run["run_id"]
    assert emitted_metadata["artifact_provenance_context"]["lane_id"] == "route_compute_runtime"
    assert emitted_metadata["artifact_provenance_context"]["variant_id"] == "dccs_refc"
    assert emitted_metadata["artifact_provenance_context"]["cache_mode"] == "cold"
    assert emitted_metadata["artifact_provenance_context"]["support_status"] == "supported"
    assert emitted_decision["preference_summary"]["contradiction_record"]["contradiction_detected"] is False
    assert emitted_decision["preference_summary"]["preference_irrelevance_proven"] is True
    assert emitted_decision["preference_summary"]["targeted_challenger_route_id"] == "route_1"
    assert emitted_decision["preference_summary"]["query_selection_reason"] == "reduce ambiguity"
    assert emitted_decision["artifact_provenance"]["run_id"] == route_run["run_id"]
    assert emitted_decision["artifact_provenance"]["lane_id"] == "route_compute_runtime"
    assert emitted_decision["artifact_provenance"]["variant_id"] == "dccs_refc"
    assert emitted_decision["artifact_provenance"]["cache_mode"] == "cold"
    assert emitted_decision["artifact_provenance"]["schema_version"] == "route-artifact-provenance-v1"
    assert emitted_decision["artifact_provenance"]["payload_schema_version"] == "1.0.0"
    assert emitted_decision["artifact_provenance"]["calibration_policy_version"] == "audit-v3"
    assert emitted_decision["artifact_provenance"]["controller_policy_version"] == "voi-controller-policy-v1"
    assert emitted_decision["artifact_provenance"]["preference_model_version"] == "preference-elicitation-policy-v1"
    assert emitted_decision["artifact_provenance"]["proxy_correction_version"] == "proxy-v5"
    assert emitted_decision["artifact_provenance"]["support_status"] == "supported"
    assert emitted_decision["artifact_provenance"]["headline_identity"] == {
        "schema_version": "route-artifact-identity-v1",
        "git_commit_hash": _git_commit_hash(),
        "environment_lockfile_hash": _backend_lockfile_hash(),
        "environment_lockfile_path": "backend/uv.lock",
        "container_digest": None,
        "container_digest_source": "unavailable_local_runtime",
        "policy_hashes": {
            "calibration_policy_hash": build_policy_hash(
                "calibration_policy",
                version="audit-v3",
                configuration={
                    "pipeline_mode": "dccs_refc",
                    "regime_id": None,
                    "copula_id": None,
                    "as_of_utc": None,
                    "probabilistic_world_policy_hash": None,
                    "audit_world_policy_hash": None,
                },
            ),
            "controller_policy_hash": build_policy_hash(
                "voi_controller_policy",
                version="voi-controller-policy-v1",
                configuration={
                    "lane_id": "route_compute_runtime",
                    "pipeline_mode": "dccs_refc",
                    "refinement_policy": None,
                },
            ),
            "preference_model_hash": build_policy_hash(
                "preference_model",
                version="preference-elicitation-policy-v1",
                configuration={
                    "lane_id": "route_compute_runtime",
                    "pipeline_mode": "dccs_refc",
                    "terminal_type": "certified_set",
                },
            ),
            "proxy_correction_hash": build_policy_hash(
                "proxy_correction",
                version="proxy-v5",
                configuration={
                    "active": True,
                    "pipeline_mode": "dccs_refc",
                    "probabilistic_world_policy_hash": None,
                    "audit_world_policy_hash": None,
                },
            ),
            "probabilistic_world_policy_hash": None,
            "audit_world_policy_hash": None,
        },
    }
    assert emitted_decision["support_summary"]["multi_fidelity_summary"]["proxy_world_count"] == 3
    assert emitted_decision["support_summary"]["audit_world_count"] == 1
    assert (
        emitted_decision["support_summary"]["multi_fidelity_certificate_basis"]
        == "corrected_from_residual_model"
    )
    assert emitted_decision["support_summary"]["audit_correction_mass"] == 2.5
    assert emitted_decision["support_summary"]["proxy_only_fraction"] == 0.75
    assert emitted_decision["support_summary"]["positivity_diagnostics"]["positivity_ok"] is True
    assert emitted_decision["certified_set_summary"]["exclusion_basis"] == certified_set_summary["exclusion_basis"]
    assert emitted_decision["certified_set_summary"]["outside_routes_safely_excluded"] is True
    assert (
        emitted_decision["certified_set_summary"]["witness"]["singleton_not_justified_reasons"]
        == ["frontier_pairwise_gap_unresolved"]
    )
    assert emitted_decision["certified_set_summary"]["witness"]["outside_routes_safely_excluded"] is True
    assert emitted_decision["certified_set_summary"]["witness"]["excluded_route_safety_reasons"] == []
    assert emitted_certified_set["exclusion_basis"] == certified_set_summary["exclusion_basis"]
    assert emitted_certified_set["outside_routes_safely_excluded"] is True
    assert emitted_certified_set["witness"]["outside_routes_safely_excluded"] is True
    assert emitted_certified_set["witness"]["excluded_route_safety_reasons"] == []
    assert emitted_trace["contradiction_record"]["contradiction_detected"] is False
    assert emitted_trace["preference_irrelevance_proven"] is True
    assert emitted_trace["no_query_reason"] is None
    assert "no_preference_query_reason" in emitted_trace
    assert emitted_trace["no_preference_query_reason"] is None
    assert emitted_trace["targeted_challenger_route_id"] == "route_1"
    assert emitted_trace["query_selection_reason"] == "reduce ambiguity"
    assert "# Decision Package Summary" in decision_summary_text
    assert "Certified Set Size: `2`" in decision_summary_text
    assert "Cache Mode: `cold`" in decision_summary_text
    certified_set_entry = next(
        item for item in emitted_index["artifacts"] if item["name"] == "certified_set_summary.json"
    )
    decision_entry = next(item for item in emitted_index["artifacts"] if item["name"] == "decision_package.json")
    assert decision_entry["markdown_summary_name"] == "decision_package.summary.md"
    assert decision_entry["markdown_summary_present"] is True
    assert certified_set_entry["artifact_provenance"]["cache_source_id"] is None
    assert certified_set_entry["artifact_provenance"]["reuse_count"] == 0
    assert certified_set_entry["artifact_provenance"]["invalidation_reason"] == "not_recorded"
    assert certified_set_entry["artifact_provenance"]["theorem_to_artifact_hooks"] == [
        {
            "hook_id": "certified_set_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        }
    ]


def test_route_bundle_emits_explicit_fixed_weight_certificate_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    routes = [
        {
            "route_id": "route_a",
            "objective": {"time": 10.0, "money": 9.5, "co2": 8.0},
            "evidence": {"scenario": {"time": 1.0, "money": 0.0, "co2": 0.0}},
        },
        {
            "route_id": "route_b",
            "objective": {"time": 11.0, "money": 10.0, "co2": 9.0},
            "evidence": {"scenario": {"time": 1.0, "money": 0.0, "co2": 0.0}},
        },
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal"}, "support_flag": True},
        {"world_id": "w1", "states": {"scenario": "mildly_stale"}, "support_flag": True},
    ]
    selector_weights = (1.15, 0.9, 1.05)
    certificate = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=selector_weights,
        threshold=0.55,
        active_families=["scenario"],
    )
    certificate.world_manifest["support_flag"] = True
    certificate.world_manifest["selected_certificate_basis"] = "empirical"

    selected = _route_option(
        "route_a",
        distance_km=120.0,
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
    )
    challenger = _route_option(
        "route_b",
        distance_km=122.0,
        duration_s=3660.0,
        monetary_cost=51.0,
        emissions_kg=21.0,
    )
    selected_certificate = RouteCertificationSummary(
        route_id="route_a",
        certificate=float(certificate.certificate["route_a"]),
        certified=bool(certificate.certified),
        threshold=float(certificate.threshold),
        active_families=["scenario"],
        top_fragility_families=[],
        top_competitor_route_id="route_b",
        top_value_of_refresh_family=None,
        ambiguity_context=None,
    )
    decision = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-fixed-weight-state",
        pipeline_mode="dccs_refc",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
        selected_certificate=selected_certificate,
        voi_stop_summary=None,
        preference_state=build_preference_state(
            route_ids=["route_a", "route_b"],
            weights={"time": selector_weights[0], "money": selector_weights[1], "co2": selector_weights[2]},
            support_flag=True,
            support_reason=None,
        ),
        preference_query_trace={},
        world_support_summary={"support_flag": True, "support_reason": None, "support_state": {"support_flag": True}},
        world_manifest=certificate.world_manifest,
        winner_confidence_state=None,
        pairwise_gap_states=[],
        selector_config=certificate.selector_config,
        certified_set=[],
        certified_set_summary=None,
        abstention=None,
    )

    fixed_weight_state = decision.fixed_weight_certificate_state
    assert fixed_weight_state["state_source"] == "selector_config"
    assert fixed_weight_state["objective_order"] == ["time", "money", "co2"]
    assert fixed_weight_state["selector_weights"] == [1.15, 0.9, 1.05]
    assert fixed_weight_state["threshold"] == pytest.approx(0.55)
    assert fixed_weight_state["selected_route_id"] == "route_a"
    assert fixed_weight_state["winner_route_id"] == "route_a"
    assert fixed_weight_state["selected_certificate"] == pytest.approx(float(certificate.certificate["route_a"]))
    assert fixed_weight_state["selected_certificate_basis"] == "empirical"
    assert decision.certificate_summary["fixed_weight_certificate_state"] == fixed_weight_state

    route_run = _write_route_run_bundle(
        req=RouteRequest(
            origin=LatLng(lat=51.5, lon=-2.6),
            destination=LatLng(lat=51.6, lon=-2.5),
            vehicle_type="rigid_hgv",
            scenario_mode="no_sharing",
            max_alternatives=2,
        ),
        selected=selected,
        candidates=[selected, challenger],
        warnings=[],
        candidate_diag=CandidateDiagnostics(selected_candidate_count=2),
        request_id="req-fixed-weight-state",
        pipeline_mode="dccs_refc",
        run_seed=20260412,
        duration_ms=10.0,
        extra_json_artifacts={
            "decision_package.json": decision.model_dump(mode="json"),
            "certificate_summary.json": copy.deepcopy(decision.certificate_summary),
        },
    )

    discovered = list_artifact_paths_for_run(str(route_run["run_id"]))
    emitted_decision = json.loads(discovered["decision_package.json"].read_text(encoding="utf-8"))
    emitted_certificate_summary = json.loads(discovered["certificate_summary.json"].read_text(encoding="utf-8"))

    assert emitted_decision["fixed_weight_certificate_state"] == fixed_weight_state
    assert emitted_decision["certificate_summary"]["fixed_weight_certificate_state"] == fixed_weight_state
    assert emitted_certificate_summary["fixed_weight_certificate_state"] == fixed_weight_state


def test_route_bundle_persists_rich_refc_fragility_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    routes = [
        _evidence_route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 0.0, "co2": 0.0},
                "weather": {"time": 0.0, "money": 0.0, "co2": 0.0},
            },
        ),
        _evidence_route(
            "route_b",
            objective=(10.2, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 0.0, "co2": 0.0},
                "weather": {"time": 0.0, "money": 0.0, "co2": 0.0},
            },
        ),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal", "weather": "nominal"}},
        {"world_id": "w1", "states": {"scenario": "severely_stale", "weather": "nominal"}},
        {"world_id": "w2", "states": {"scenario": "severely_stale", "weather": "nominal"}},
    ]

    certificate = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.60,
        active_families=["scenario", "weather"],
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather"],
        selected_route_id="route_a",
    )

    route_fragility_payload = build_route_fragility_artifact_payload(
        certificate,
        fragility,
        selected_route_id="route_a",
    )
    competitor_payload = build_competitor_fragility_artifact_payload(
        certificate,
        fragility,
        selected_route_id="route_a",
    )
    sampled_world_manifest_payload = build_sampled_world_manifest_artifact_payload(
        {
            "seed": 17,
            "requested_world_count": 4,
            "world_count": 4,
            "unique_world_count": 3,
            "active_families": ["scenario"],
            "state_catalog": ["nominal", "proxy", "refreshed"],
            "support_flag": True,
            "support_bin": "supported",
            "calibration_bin": "empirical",
            "selected_certificate_basis": "empirical",
            "calibration_policy_version": "calibration-policy-v3",
            "worlds": [
                {"world_id": "w0", "states": {"scenario": "nominal"}, "world_kind": "sampled"},
                {"world_id": "w1", "states": {"scenario": "proxy"}, "world_kind": "sampled"},
                {"world_id": "w2", "states": {"scenario": "refreshed"}, "world_kind": "hard_case_targeted"},
                {"world_id": "w2", "states": {"scenario": "refreshed"}, "world_kind": "hard_case_targeted"},
            ],
        }
    )

    selected = _route_option(
        "route_a",
        distance_km=120.0,
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
    )
    challenger = _route_option(
        "route_b",
        distance_km=121.0,
        duration_s=3660.0,
        monetary_cost=51.0,
        emissions_kg=20.5,
    )

    route_run = _write_route_run_bundle(
        req=RouteRequest(
            origin=LatLng(lat=51.5, lon=-2.6),
            destination=LatLng(lat=51.6, lon=-2.5),
            vehicle_type="rigid_hgv",
            scenario_mode="no_sharing",
            max_alternatives=2,
        ),
        selected=selected,
        candidates=[selected, challenger],
        warnings=[],
        candidate_diag=CandidateDiagnostics(selected_candidate_count=2),
        request_id="req-fragility-artifacts",
        pipeline_mode="dccs_refc",
        run_seed=20260411,
        duration_ms=18.0,
        extra_json_artifacts={
            "route_fragility_map.json": route_fragility_payload,
            "competitor_fragility_breakdown.json": competitor_payload,
            "sampled_world_manifest.json": sampled_world_manifest_payload,
        },
    )

    discovered = list_artifact_paths_for_run(str(route_run["run_id"]))
    emitted_route_fragility = json.loads(discovered["route_fragility_map.json"].read_text(encoding="utf-8"))
    emitted_competitor = json.loads(
        discovered["competitor_fragility_breakdown.json"].read_text(encoding="utf-8")
    )
    emitted_manifest = json.loads(discovered["sampled_world_manifest.json"].read_text(encoding="utf-8"))

    route_entry = emitted_route_fragility["route_a"]
    challenger_entry = emitted_competitor["route_a"]["route_b"]

    assert set(route_entry) >= {
        "family_fragility_scores",
        "deterministic_local_flip_radius",
        "probabilistic_flip_radius",
        "family_specific_radii",
        "challenger_specific_radii",
        "dominant_fragility_family",
        "minimum_flip_budget",
        "adversarial_degradation_curve",
    }
    assert route_entry == route_fragility_payload["route_a"]

    assert set(challenger_entry) >= {
        "family_fragility_counts",
        "pairwise_gap_lower_bound",
        "pairwise_gap_upper_bound",
        "challenger_radius",
        "challenger_audit_sensitivity",
        "nearest_challenger",
        "dominant_evidence_family",
        "challenger_family_pressure",
    }
    assert challenger_entry == competitor_payload["route_a"]["route_b"]

    assert set(emitted_manifest) >= {
        "probabilistic_worlds",
        "audit_worlds",
        "proxy_only_worlds",
        "audited_worlds",
        "reused_worlds",
        "support_bins",
        "calibration_policy_version",
    }
    assert emitted_manifest["probabilistic_worlds"] == sampled_world_manifest_payload["probabilistic_worlds"]
    assert emitted_manifest["audit_worlds"] == sampled_world_manifest_payload["audit_worlds"]
    assert emitted_manifest["proxy_only_worlds"] == sampled_world_manifest_payload["proxy_only_worlds"]
    assert emitted_manifest["audited_worlds"] == sampled_world_manifest_payload["audited_worlds"]
    assert emitted_manifest["reused_worlds"] == sampled_world_manifest_payload["reused_worlds"]
    assert emitted_manifest["support_bins"] == sampled_world_manifest_payload["support_bins"]
    assert (
        emitted_manifest["calibration_policy_version"]
        == sampled_world_manifest_payload["calibration_policy_version"]
    )


def test_route_bundle_persists_support_aware_decision_region_and_witness_fields(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    routes = [
        _evidence_route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "fuel": {"time": 0.0, "money": 1.0, "co2": 0.0},
                "scenario": {"time": 1.0, "money": 0.0, "co2": 0.0},
            },
        ),
        _evidence_route(
            "route_b",
            objective=(10.2, 10.0, 10.0),
            evidence_tensor={
                "fuel": {"time": 0.0, "money": 1.0, "co2": 0.0},
                "scenario": {"time": 1.0, "money": 0.0, "co2": 0.0},
            },
        ),
    ]
    worlds = [
        {
            "world_id": "w0",
            "states": {"fuel": "nominal", "scenario": "nominal"},
            "support_flag": True,
        },
        {
            "world_id": "w1",
            "states": {"fuel": "severely_stale", "scenario": "nominal"},
            "support_flag": True,
        },
    ]

    certificate = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.80,
        active_families=["fuel", "scenario"],
    )
    certificate.world_manifest["support_flag"] = True
    certificate.world_manifest["support_bin"] = "supported"
    certificate.world_manifest["calibration_bin"] = "empirical"
    certificate.world_manifest["calibration_policy_version"] = "calibration-policy-v3"
    certificate.world_manifest["selected_certificate_basis"] = "empirical"
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["fuel", "scenario"],
        selected_route_id="route_a",
    )
    projection = project_refc_scaffold_states(
        certificate,
        fragility,
        frontier_route_ids=["route_a"],
        selected_route_id="route_a",
    )

    selected = _route_option(
        "route_a",
        distance_km=120.0,
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
    )
    challenger = _route_option(
        "route_b",
        distance_km=121.0,
        duration_s=3660.0,
        monetary_cost=51.0,
        emissions_kg=20.5,
    )

    route_run = _write_route_run_bundle(
        req=RouteRequest(
            origin=LatLng(lat=51.5, lon=-2.6),
            destination=LatLng(lat=51.6, lon=-2.5),
            vehicle_type="rigid_hgv",
            scenario_mode="no_sharing",
            max_alternatives=2,
        ),
        selected=selected,
        candidates=[selected, challenger],
        warnings=[],
        candidate_diag=CandidateDiagnostics(selected_candidate_count=2),
        request_id="req-decision-region-witness",
        pipeline_mode="dccs_refc",
        run_seed=20260411,
        duration_ms=18.0,
        extra_json_artifacts={
            "pairwise_gap_state.json": {
                "pairwise_gap_states": [state.as_dict() for state in projection["pairwise_gap_states"]],
                "selected_route_id": "route_a",
                "selected_certificate": float(certificate.certificate["route_a"]),
                "support_flag": True,
            },
            "flip_radius_summary.json": projection["flip_radius_state"].as_dict(),
            "decision_region_summary.json": projection["decision_region_state"].as_dict(),
            "certificate_witness.json": projection["certificate_witness"].as_dict(),
            "certified_set_summary.json": projection["certified_set_state"].as_dict(),
        },
    )

    discovered = list_artifact_paths_for_run(str(route_run["run_id"]))
    emitted_decision_region = json.loads(
        discovered["decision_region_summary.json"].read_text(encoding="utf-8")
    )
    emitted_witness = json.loads(discovered["certificate_witness.json"].read_text(encoding="utf-8"))

    assert emitted_decision_region["support_status"] == "supported"
    assert emitted_decision_region["support_bin"] == "supported"
    assert emitted_decision_region["calibration_bin"] == "empirical"
    assert emitted_decision_region["calibration_policy_version"] == "calibration-policy-v3"
    assert emitted_decision_region["selected_certificate_basis"] == "empirical"
    assert emitted_decision_region["nearest_challenger_gap_lower_bound"] is not None
    assert emitted_decision_region["nearest_challenger_audit_sensitivity"] is not None
    assert emitted_decision_region["route_fragility_family_count"] == len(
        emitted_decision_region["provenance"]["route_fragility_families"]
    )
    assert emitted_decision_region["atlas_kind"] is not None
    assert emitted_decision_region["root_cause_tags"]

    assert emitted_witness["support_status"] == "supported"
    assert emitted_witness["support_bin"] == "supported"
    assert emitted_witness["calibration_bin"] == "empirical"
    assert emitted_witness["calibration_policy_version"] == "calibration-policy-v3"
    assert emitted_witness["selected_certificate_basis"] == "empirical"
    assert emitted_witness["nearest_certificate_boundary"] == emitted_decision_region["nearest_certificate_boundary"]
    assert emitted_witness["targeted_challenger_route_id"] == "route_b"
    assert emitted_witness["active_challenger_count"] == 1
    assert emitted_witness["active_evidence_family_count"] >= 1
    assert emitted_witness["active_preference_constraint_count"] >= 1
    assert emitted_witness["support_condition_count"] >= 1
    assert emitted_witness["action_step_count"] >= 1
    assert emitted_witness["explanation_sparsity"] == emitted_witness["witness_sparsity"]
    assert emitted_witness["atlas_kind"] == emitted_decision_region["atlas_kind"]
    assert emitted_witness["root_cause_tags"] == emitted_decision_region["root_cause_tags"]


def test_route_bundle_keeps_rich_certified_set_artifact_but_normalizes_singleton_decision_package(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    selected = _route_option(
        "route_0",
        distance_km=120.0,
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
    )
    challenger = _route_option(
        "route_1",
        distance_km=123.0,
        duration_s=3720.0,
        monetary_cost=52.0,
        emissions_kg=21.5,
    )
    rich_certified_set_summary = {
        "member_route_ids": ["route_0"],
        "excluded_route_ids": ["route_1"],
        "exclusion_basis": ["certificate_threshold", "frontier_selection"],
        "certified": True,
        "threshold": 0.8,
        "support_flag": True,
        "set_size": 1,
        "witness": {
            "route_id": "route_0",
            "active_challenger_ids": ["route_1"],
            "support_flag": True,
        },
    }
    public_singleton_summary = _normalize_public_certified_set_summary(
        terminal_type="certified_singleton",
        selected_route_id="route_0",
        candidate_route_ids=["route_0", "route_1"],
        certified_set_summary=rich_certified_set_summary,
    )

    route_run = _write_route_run_bundle(
        req=RouteRequest(
            origin=LatLng(lat=51.5, lon=-2.6),
            destination=LatLng(lat=51.6, lon=-2.5),
            vehicle_type="rigid_hgv",
            scenario_mode="no_sharing",
            max_alternatives=2,
        ),
        selected=selected,
        candidates=[selected, challenger],
        warnings=[],
        candidate_diag=CandidateDiagnostics(selected_candidate_count=2),
        request_id="req-singleton-artifact",
        pipeline_mode="dccs_refc",
        run_seed=20260410,
        duration_ms=12.5,
        extra_json_artifacts={
            "decision_package.json": {
                "schema_version": "1.0.0",
                "terminal_type": "certified_singleton",
                "selected_route_id": "route_0",
                "selected_certificate_basis": "selected_certificate",
                "preference_summary": {},
                "support_summary": {},
                "certified_set_summary": public_singleton_summary,
            },
            "certified_set_summary.json": copy.deepcopy(rich_certified_set_summary),
        },
    )

    discovered = list_artifact_paths_for_run(str(route_run["run_id"]))
    emitted_decision = json.loads(discovered["decision_package.json"].read_text(encoding="utf-8"))
    emitted_certified_set = json.loads(discovered["certified_set_summary.json"].read_text(encoding="utf-8"))

    assert emitted_decision["terminal_type"] == "certified_singleton"
    assert emitted_decision["certified_set_summary"]["member_route_ids"] == []
    assert emitted_decision["certified_set_summary"]["excluded_route_ids"] == ["route_1"]
    assert emitted_decision["certified_set_summary"]["certified"] is False
    assert emitted_decision["certified_set_summary"]["set_size"] == 0
    assert emitted_decision["certified_set_summary"]["terminal_type"] == "certified_singleton"
    assert emitted_decision["certified_set_summary"]["not_applicable_reason"] == "singleton_terminal"

    assert emitted_certified_set["member_route_ids"] == ["route_0"]
    assert emitted_certified_set["excluded_route_ids"] == ["route_1"]
    assert emitted_certified_set["certified"] is True
    assert emitted_certified_set["set_size"] == 1


def test_route_bundle_keeps_rich_certified_set_artifact_but_normalizes_abstention_decision_package(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    selected = _route_option(
        "route_0",
        distance_km=120.0,
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
    )
    challenger = _route_option(
        "route_1",
        distance_km=123.0,
        duration_s=3720.0,
        monetary_cost=52.0,
        emissions_kg=21.5,
    )
    rich_certified_set_summary = {
        "member_route_ids": ["route_0"],
        "excluded_route_ids": ["route_1"],
        "exclusion_basis": ["certificate_threshold", "frontier_selection"],
        "certified": True,
        "threshold": 0.8,
        "support_flag": True,
        "set_size": 1,
        "witness": {
            "route_id": "route_0",
            "active_challenger_ids": ["route_1"],
            "support_flag": True,
        },
    }
    public_abstention_summary = _normalize_public_certified_set_summary(
        terminal_type="typed_abstention",
        selected_route_id="route_0",
        candidate_route_ids=["route_0", "route_1"],
        certified_set_summary=rich_certified_set_summary,
    )

    route_run = _write_route_run_bundle(
        req=RouteRequest(
            origin=LatLng(lat=51.5, lon=-2.6),
            destination=LatLng(lat=51.6, lon=-2.5),
            vehicle_type="rigid_hgv",
            scenario_mode="no_sharing",
            max_alternatives=2,
        ),
        selected=selected,
        candidates=[selected, challenger],
        warnings=[],
        candidate_diag=CandidateDiagnostics(selected_candidate_count=2),
        request_id="req-abstention-artifact",
        pipeline_mode="dccs_refc",
        run_seed=20260410,
        duration_ms=12.5,
        extra_json_artifacts={
            "decision_package.json": {
                "schema_version": "1.0.0",
                "terminal_type": "typed_abstention",
                "selected_route_id": "route_0",
                "selected_certificate_basis": "selected_certificate",
                "preference_summary": {},
                "support_summary": {},
                "abstention_summary": {
                    "reason_code": "uncertified_due_to_search",
                    "terminal_type": "typed_abstention",
                },
                "certified_set_summary": public_abstention_summary,
            },
            "certified_set_summary.json": copy.deepcopy(rich_certified_set_summary),
        },
    )

    discovered = list_artifact_paths_for_run(str(route_run["run_id"]))
    emitted_decision = json.loads(discovered["decision_package.json"].read_text(encoding="utf-8"))
    emitted_certified_set = json.loads(discovered["certified_set_summary.json"].read_text(encoding="utf-8"))

    assert emitted_decision["terminal_type"] == "typed_abstention"
    assert emitted_decision["certified_set_summary"]["member_route_ids"] == []
    assert emitted_decision["certified_set_summary"]["excluded_route_ids"] == ["route_0", "route_1"]
    assert emitted_decision["certified_set_summary"]["certified"] is False
    assert emitted_decision["certified_set_summary"]["set_size"] == 0
    assert emitted_decision["certified_set_summary"]["terminal_type"] == "typed_abstention"
    assert emitted_decision["certified_set_summary"]["not_applicable_reason"] == "abstention_terminal"

    assert emitted_certified_set["member_route_ids"] == ["route_0"]
    assert emitted_certified_set["excluded_route_ids"] == ["route_1"]
    assert emitted_certified_set["certified"] is True
    assert emitted_certified_set["set_size"] == 1
