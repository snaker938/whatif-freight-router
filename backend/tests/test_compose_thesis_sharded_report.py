from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.compose_thesis_sharded_report as compose_module


pytestmark = pytest.mark.thesis_results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _canonical_corpus(path: Path) -> Path:
    rows = [
        {
            "profile_id": "profile-a",
            "od_id": "od-a",
            "origin_lat": 51.5,
            "origin_lon": -0.1,
            "destination_lat": 54.9,
            "destination_lon": -1.6,
            "corpus_group": "representative",
            "od_ambiguity_index": 0.22,
            "od_ambiguity_confidence": 0.7,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix": "{\"probe\":1}",
        },
        {
            "profile_id": "profile-b",
            "od_id": "od-b",
            "origin_lat": 52.4,
            "origin_lon": -1.9,
            "destination_lat": 53.4,
            "destination_lon": -2.9,
            "corpus_group": "ambiguity",
            "od_ambiguity_index": 0.58,
            "od_ambiguity_confidence": 0.83,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": "{\"probe\":1,\"prior\":1}",
        },
    ]
    return _write_csv(path, rows)


def _write_shard_companions(
    shard_results_csv: Path,
    *,
    run_id: str,
    strict_evidence_policy: str = "no_synthetic_no_proxy_no_fallback",
    repo_asset_preflight_required_ok: bool = True,
    strict_route_ready: bool = True,
    baseline_smoke_required_ok: bool = True,
    strict_live_readiness_pass_rate: float = 1.0,
    evaluation_rerun_success_rate: float = 1.0,
    scenario_profile_unavailable_rate: float = 0.0,
    cache_mode: str = "preserve",
) -> None:
    evaluation_manifest = {
        "run_id": run_id,
        "strict_evidence_policy": strict_evidence_policy,
        "repo_asset_preflight_required_ok": repo_asset_preflight_required_ok,
        "backend_ready_summary": {
            "strict_route_ready": strict_route_ready,
        },
        "baseline_smoke_summary": {
            "required_ok": baseline_smoke_required_ok,
        },
        "run_validity": {
            "strict_live_readiness_pass_rate": strict_live_readiness_pass_rate,
            "evaluation_rerun_success_rate": evaluation_rerun_success_rate,
            "scenario_profile_unavailable_rate": scenario_profile_unavailable_rate,
        },
        "ors_baseline_policy": "repo_local",
        "cache_mode": cache_mode,
        "cache_reset_scope": "none",
        "cache_reset_policy": "none",
        "cache_reset_count": 0,
        "evaluation_suite": {
            "role": "broad_cold_proof",
            "family": "thesis",
        },
        "corpus_hash": f"corpus-{run_id}",
    }
    metadata = {
        "run_id": run_id,
        "strict_evidence_policy": strict_evidence_policy,
        "repo_asset_preflight_required_ok": repo_asset_preflight_required_ok,
        "backend_ready_summary": {
            "strict_route_ready": strict_route_ready,
        },
        "baseline_smoke_summary": {
            "required_ok": baseline_smoke_required_ok,
        },
        "run_validity": {
            "strict_live_readiness_pass_rate": strict_live_readiness_pass_rate,
            "evaluation_rerun_success_rate": evaluation_rerun_success_rate,
            "scenario_profile_unavailable_rate": scenario_profile_unavailable_rate,
        },
        "cache_mode": cache_mode,
        "evaluation_suite": {
            "role": "broad_cold_proof",
            "family": "thesis",
        },
        "corpus_hash": f"corpus-{run_id}",
    }
    _write_json(shard_results_csv.with_name("evaluation_manifest.json"), evaluation_manifest)
    _write_json(shard_results_csv.with_name("metadata.json"), metadata)


def _variant_rows(*, od_id: str, profile_id: str, artifact_run_id: str, corpus_group: str, origin_lat: float, origin_lon: float, destination_lat: float, destination_lon: float) -> list[dict[str, object]]:
    return [
        {
            "artifact_run_id": artifact_run_id,
            "od_id": od_id,
            "profile_id": profile_id,
            "variant_id": "V0",
            "pipeline_mode": "legacy",
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "destination_lat": destination_lat,
            "destination_lon": destination_lon,
            "trip_length_bin": "250-500 km",
            "corpus_group": corpus_group,
            "corpus_kind": corpus_group,
            "ambiguity_prior_source": "probe",
            "weighted_win_osrm": "true",
            "weighted_win_ors": "true",
            "dominates_osrm": "false",
            "dominates_ors": "false",
            "time_preserving_win_osrm": "true",
            "time_preserving_win_ors": "true",
            "runtime_ms": 120.0,
            "algorithm_runtime_ms": 70.0,
            "baseline_osrm_ms": 20.0,
            "baseline_ors_ms": 25.0,
            "route_evidence_ok": "true",
            "artifact_complete": "true",
            "failure_reason": "",
        },
        {
            "artifact_run_id": artifact_run_id,
            "od_id": od_id,
            "profile_id": profile_id,
            "variant_id": "A",
            "pipeline_mode": "dccs",
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "destination_lat": destination_lat,
            "destination_lon": destination_lon,
            "trip_length_bin": "250-500 km",
            "corpus_group": corpus_group,
            "corpus_kind": corpus_group,
            "ambiguity_prior_source": "probe",
            "weighted_win_osrm": "true",
            "weighted_win_ors": "true",
            "dominates_osrm": "false",
            "dominates_ors": "false",
            "time_preserving_win_osrm": "true",
            "time_preserving_win_ors": "true",
            "runtime_ms": 110.0,
            "algorithm_runtime_ms": 60.0,
            "baseline_osrm_ms": 20.0,
            "baseline_ors_ms": 25.0,
            "route_evidence_ok": "true",
            "artifact_complete": "true",
            "failure_reason": "",
        },
        {
            "artifact_run_id": artifact_run_id,
            "od_id": od_id,
            "profile_id": profile_id,
            "variant_id": "B",
            "pipeline_mode": "dccs_refc",
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "destination_lat": destination_lat,
            "destination_lon": destination_lon,
            "trip_length_bin": "250-500 km",
            "corpus_group": corpus_group,
            "corpus_kind": corpus_group,
            "ambiguity_prior_source": "probe",
            "weighted_win_osrm": "true",
            "weighted_win_ors": "true",
            "dominates_osrm": "false",
            "dominates_ors": "false",
            "time_preserving_win_osrm": "true",
            "time_preserving_win_ors": "true",
            "runtime_ms": 105.0,
            "algorithm_runtime_ms": 55.0,
            "baseline_osrm_ms": 20.0,
            "baseline_ors_ms": 25.0,
            "certificate": 1.0,
            "route_evidence_ok": "true",
            "artifact_complete": "true",
            "failure_reason": "",
        },
        {
            "artifact_run_id": artifact_run_id,
            "od_id": od_id,
            "profile_id": profile_id,
            "variant_id": "C",
            "pipeline_mode": "voi",
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "destination_lat": destination_lat,
            "destination_lon": destination_lon,
            "trip_length_bin": "250-500 km",
            "corpus_group": corpus_group,
            "corpus_kind": corpus_group,
            "ambiguity_prior_source": "probe",
            "weighted_win_osrm": "true",
            "weighted_win_ors": "true",
            "dominates_osrm": "false",
            "dominates_ors": "false",
            "time_preserving_win_osrm": "true",
            "time_preserving_win_ors": "true",
            "runtime_ms": 100.0,
            "algorithm_runtime_ms": 50.0,
            "baseline_osrm_ms": 20.0,
            "baseline_ors_ms": 25.0,
            "certificate": 1.0,
            "voi_controller_engaged": "true",
            "route_evidence_ok": "true",
            "artifact_complete": "true",
            "failure_reason": "",
        },
    ]


def test_compose_sharded_report_writes_aggregate_bundle_with_shard_provenance(tmp_path: Path) -> None:
    canonical_csv = _canonical_corpus(tmp_path / "canonical.csv")
    shard_a = _write_csv(
        tmp_path / "run-a" / "thesis_results.csv",
        _variant_rows(
            od_id="od-a",
            profile_id="profile-a",
            artifact_run_id="run-a",
            corpus_group="representative",
            origin_lat=51.5,
            origin_lon=-0.1,
            destination_lat=54.9,
            destination_lon=-1.6,
        ),
    )
    shard_b = _write_csv(
        tmp_path / "run-b" / "thesis_results.csv",
        _variant_rows(
            od_id="od-a",
            profile_id="profile-a",
            artifact_run_id="run-b",
            corpus_group="representative",
            origin_lat=51.5,
            origin_lon=-0.1,
            destination_lat=54.9,
            destination_lon=-1.6,
        )
        + _variant_rows(
            od_id="od-b",
            profile_id="profile-b",
            artifact_run_id="run-b",
            corpus_group="ambiguity",
            origin_lat=52.4,
            origin_lon=-1.9,
            destination_lat=53.4,
            destination_lon=-2.9,
        ),
    )
    _write_shard_companions(shard_a, run_id="run-a")
    _write_shard_companions(
        shard_b,
        run_id="run-b",
        repo_asset_preflight_required_ok=False,
        strict_route_ready=False,
        baseline_smoke_required_ok=False,
        strict_live_readiness_pass_rate=0.5,
        evaluation_rerun_success_rate=0.5,
        scenario_profile_unavailable_rate=0.5,
        cache_mode="cold",
    )

    args = compose_module._build_parser().parse_args(
        [
            "--run-id",
            "sharded-suite-test",
            "--out-dir",
            str(tmp_path / "out"),
            "--canonical-corpus",
            str(canonical_csv),
            "--results-csv",
            str(shard_a),
            "--results-csv",
            str(shard_b),
        ]
    )
    payload = compose_module.compose_sharded_report(args)

    assert payload["run_id"] == "sharded-suite-test"
    assert Path(payload["results_csv"]).exists()
    assert Path(payload["summary_csv"]).exists()
    assert Path(payload["summary_by_cohort_csv"]).exists()
    assert Path(payload["evaluation_manifest"]).exists()
    assert Path(payload["manifest_path"]).exists()
    assert Path(payload["shard_sources_json"]).exists()
    assert payload["output_artifact_validation"]["validated_artifact_count"] >= 8

    shard_sources = json.loads(Path(payload["shard_sources_json"]).read_text(encoding="utf-8"))
    assert shard_sources["composition_type"] == "sharded_results_merge"
    assert shard_sources["shard_count"] == 2
    assert shard_sources["row_count"] == 8
    assert shard_sources["raw_input_row_count"] == 12
    assert shard_sources["pair_count"] == 2
    assert len(shard_sources["shards"]) == 2
    assert [entry["row_count"] for entry in shard_sources["shards"]] == [4, 8]
    assert shard_sources["shards"][0]["artifact_run_ids"] == ["run-a"]
    assert shard_sources["shards"][1]["artifact_run_ids"] == ["run-b"]
    assert shard_sources["shards"][0]["newly_admitted_od_ids"] == ["od-a"]
    assert shard_sources["shards"][0]["replay_od_ids"] == []
    assert shard_sources["shards"][1]["newly_admitted_od_ids"] == ["od-b"]
    assert shard_sources["shards"][1]["replay_od_ids"] == ["od-a"]
    assert shard_sources["shards"][1]["missing_replay_od_ids"] == []
    assert shard_sources["shards"][1]["regression_summary"]["replay_complete"] is True
    assert shard_sources["shards"][1]["regression_summary"]["replay_rows_all_green"] is True
    assert shard_sources["shards"][1]["baseline_comparison_summary"]["A"]["weighted_win_rate_osrm"] == 1.0
    assert shard_sources["shards"][1]["baseline_comparison_summary"]["A"]["weighted_win_rate_ors"] == 1.0
    assert shard_sources["shards"][0]["evaluation_manifest_present"] is True
    assert shard_sources["shards"][0]["metadata_present"] is True
    assert shard_sources["shards"][0]["companion_run_id"] == "run-a"
    assert shard_sources["shards"][1]["repo_asset_preflight_required_ok"] is False
    assert shard_sources["shards"][1]["backend_strict_route_ready"] is False
    assert shard_sources["shards"][1]["baseline_smoke_required_ok"] is False
    assert shard_sources["shard_evidence_rollup"]["evaluation_manifest_present_count"] == 2
    assert shard_sources["shard_evidence_rollup"]["metadata_present_count"] == 2
    assert shard_sources["shard_evidence_rollup"]["strict_evidence_policy_values"] == [
        "no_synthetic_no_proxy_no_fallback"
    ]
    assert shard_sources["shard_evidence_rollup"]["repo_asset_preflight_required_ok"]["all_true"] is False
    assert shard_sources["shard_evidence_rollup"]["backend_strict_route_ready"]["all_true"] is False
    assert shard_sources["shard_evidence_rollup"]["baseline_smoke_required_ok"]["all_true"] is False
    assert shard_sources["shard_evidence_rollup"]["run_validity"]["all_clean"] is False
    assert shard_sources["campaign_rung_summary"]["rung_count"] == 2
    assert shard_sources["campaign_rung_summary"]["all_rungs_replay_complete"] is True
    assert shard_sources["campaign_rung_summary"]["all_rungs_regression_green"] is True
    assert shard_sources["campaign_rung_summary"]["latest_rung_newly_admitted_od_ids"] == ["od-b"]
    assert shard_sources["campaign_rung_summary"]["latest_rung_replay_od_ids"] == ["od-a"]
    assert len(shard_sources["rung_evidence_ledger"]) == 2

    summary_rows = list(csv.DictReader(Path(payload["summary_csv"]).open("r", encoding="utf-8", newline="")))
    assert {row["variant_id"]: int(row["row_count"]) for row in summary_rows} == {
        "V0": 2,
        "A": 2,
        "B": 2,
        "C": 2,
    }

    evaluation_manifest = json.loads(Path(payload["evaluation_manifest"]).read_text(encoding="utf-8"))
    assert evaluation_manifest["composition"]["composition_type"] == "sharded_results_merge"
    assert evaluation_manifest["composition"]["shard_count"] == 2
    assert evaluation_manifest["evaluation_suite"]["role"] == "composed_sharded_report"
    assert evaluation_manifest["shard_preflight_summary"]["required_ok"] is False
    assert evaluation_manifest["shard_readiness_summary"]["strict_route_ready"] is False
    assert evaluation_manifest["shard_baseline_smoke_summary"]["required_ok"] is False
    assert evaluation_manifest["shard_evidence_rollup"]["run_validity"]["all_clean"] is False
    assert evaluation_manifest["campaign_rung_summary"]["all_rungs_replay_complete"] is True
    assert evaluation_manifest["rung_evidence_ledger"][1]["replay_od_ids"] == ["od-a"]

    metadata = json.loads((Path(payload["results_csv"]).parent / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["shard_preflight_summary"]["required_ok"] is False
    assert metadata["shard_readiness_summary"]["strict_route_ready"] is False
    assert metadata["shard_baseline_smoke_summary"]["required_ok"] is False
    assert metadata["campaign_rung_summary"]["latest_rung_newly_admitted_od_ids"] == ["od-b"]

    thesis_report_text = Path(payload["thesis_report"]).read_text(encoding="utf-8")
    assert "- Repo preflight required ok: `False`" in thesis_report_text
    assert "- Backend strict ready: `False`" in thesis_report_text
    assert "- Baseline smoke required ok: `False`" in thesis_report_text
    assert "## Campaign Rung Evidence" in thesis_report_text
    assert 'rung_2: new=["od-b"]; replay=["od-a"]; missing_replay=[]; replay_complete=True; replay_rows_all_green=True; regression_green=True' in thesis_report_text


def test_compose_sharded_report_preserves_missing_replay_in_rung_ledger(tmp_path: Path) -> None:
    canonical_csv = _canonical_corpus(tmp_path / "canonical.csv")
    shard_a = _write_csv(
        tmp_path / "run-a" / "thesis_results.csv",
        _variant_rows(
            od_id="od-a",
            profile_id="profile-a",
            artifact_run_id="run-a",
            corpus_group="representative",
            origin_lat=51.5,
            origin_lon=-0.1,
            destination_lat=54.9,
            destination_lon=-1.6,
        ),
    )
    shard_b = _write_csv(
        tmp_path / "run-b" / "thesis_results.csv",
        _variant_rows(
            od_id="od-b",
            profile_id="profile-b",
            artifact_run_id="run-b",
            corpus_group="ambiguity",
            origin_lat=52.4,
            origin_lon=-1.9,
            destination_lat=53.4,
            destination_lon=-2.9,
        ),
    )
    _write_shard_companions(shard_a, run_id="run-a")
    _write_shard_companions(shard_b, run_id="run-b")

    args = compose_module._build_parser().parse_args(
        [
            "--run-id",
            "sharded-suite-missing-replay",
            "--out-dir",
            str(tmp_path / "out"),
            "--canonical-corpus",
            str(canonical_csv),
            "--results-csv",
            str(shard_a),
            "--results-csv",
            str(shard_b),
        ]
    )
    payload = compose_module.compose_sharded_report(args)

    shard_sources = json.loads(Path(payload["shard_sources_json"]).read_text(encoding="utf-8"))
    assert shard_sources["campaign_rung_summary"]["all_rungs_replay_complete"] is False
    assert shard_sources["campaign_rung_summary"]["all_rungs_regression_green"] is False
    assert shard_sources["rung_evidence_ledger"][1]["missing_replay_od_ids"] == ["od-a"]
    assert shard_sources["rung_evidence_ledger"][1]["regression_summary"]["missing_replay_pair_count"] == 1
    assert shard_sources["rung_evidence_ledger"][1]["regression_summary"]["replay_complete"] is False
    assert shard_sources["rung_evidence_ledger"][1]["regression_summary"]["regression_green"] is False


def test_compose_sharded_report_rejects_duplicate_variant_rows_within_same_shard(tmp_path: Path) -> None:
    canonical_csv = _canonical_corpus(tmp_path / "canonical.csv")
    duplicate_rows = _variant_rows(
        od_id="od-a",
        profile_id="profile-a",
        artifact_run_id="run-a",
        corpus_group="representative",
        origin_lat=51.5,
        origin_lon=-0.1,
        destination_lat=54.9,
        destination_lon=-1.6,
    )
    shard_a = _write_csv(tmp_path / "run-a" / "thesis_results.csv", duplicate_rows + duplicate_rows)

    args = compose_module._build_parser().parse_args(
        [
            "--run-id",
            "sharded-suite-duplicate",
            "--out-dir",
            str(tmp_path / "out"),
            "--canonical-corpus",
            str(canonical_csv),
            "--results-csv",
            str(shard_a),
        ]
    )

    with pytest.raises(RuntimeError, match="duplicate_shard_row:"):
        compose_module.compose_sharded_report(args)
