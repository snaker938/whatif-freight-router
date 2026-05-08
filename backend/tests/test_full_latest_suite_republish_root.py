from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_full_latest_suite as full_suite_module


def test_proxy_audit_lane_uses_focused_corpus_slice() -> None:
    assert full_suite_module._lane_corpus_key("proxy_audit_calibration") == "focused"


def _write_lane_bundle(
    *,
    run_id: str,
    role: str,
    variant_id: str,
    corpus_key: str,
    pipeline_mode: str,
    baseline_identity_verified_rate: float = 1.0,
) -> None:
    artifact_dir = full_suite_module.artifact_dir_for_run(run_id)
    summary_row = {
        "variant_id": variant_id,
        "pipeline_mode": pipeline_mode,
        "row_count": 1,
        "success_rate": 1.0,
        "certified_rate": 1.0,
        "mean_certificate": 0.91,
        "weighted_win_rate_best_baseline": 0.88,
        "dominance_win_rate_best_baseline": 0.91,
        "dominance_win_rate_osrm": 0.9,
        "dominance_win_rate_ors": 0.9,
        "time_preserving_win_rate_best_baseline": 0.8,
        "time_preserving_win_rate_osrm": 0.8,
        "time_preserving_win_rate_ors": 0.8,
        "mean_weighted_margin_vs_best_baseline": 4.2,
        "mean_runtime_ratio_vs_osrm": 0.55,
        "mean_runtime_ratio_vs_ors": 0.60,
        "mean_runtime_p50_ms": 100.0,
        "mean_runtime_p90_ms": 120.0,
        "mean_runtime_p95_ms": 130.0,
        "mean_process_rss_p90_mb": 256.0,
        "median_preference_query_count": 1.0,
        "p90_preference_query_count": 2.0,
        "nontrivial_frontier_rate": 0.9,
        "mean_dccs_false_safe_prune_rate": 0.0,
        "mean_dccs_anti_collapse_success_rate": 0.9,
        "mean_dccs_certificate_critical_hit_rate": 0.9,
        "mean_dccs_time_preserving_challenger_coverage": 0.9,
        "mean_dccs_dominance_likely_challenger_coverage": 0.9,
        "productive_voi_action_rate": 0.8,
        "unnecessary_voi_refine_rate": 0.0,
        "mean_voi_realized_certificate_lift": 0.08,
        "refine_cost_mape": 0.2,
        "refine_cost_rank_correlation": 0.7,
        "mean_route_cache_hit_rate": 0.9,
        "mean_option_build_cache_hit_rate": 0.9,
        "mean_option_build_reuse_rate": 0.9,
        "mean_refc_world_reuse_rate": 0.9,
        "baseline_identity_verified_rate": baseline_identity_verified_rate,
    }
    lane_metadata = {
        "observed_sample_size": {
            "row_count": 4,
            "unique_od_count": 1,
            "unique_row_seed_count": 1,
            "evaluation_size_requirement_met": True,
        },
        "evaluation_size_requirement": {
            "requirement_id": f"{role}:sample-size",
            "unit": "rows",
            "minimum": 1,
            "minimum_description": "at least one row",
        },
        "seed_repeat_plan": {
            "headline_seed_repeat_required": True,
            "requirement_ids": ["P14.7"],
            "minimum_seed_count": 3,
            "configured_seed_count": 3,
            "configured_seeds": [11, 12, 13],
            "meets_minimum": True,
            "status": "complete",
        },
    }
    claim_row = {
        "variant_id": variant_id,
        "pipeline_mode": pipeline_mode,
        "headline_metric_name": "mean_weighted_margin_vs_best_baseline",
        "seed_count": 3,
        "headline_seed_minimum_met": True,
        "majority_agreement_requirement_met": True,
        "paired_comparison_row_count_min": 1,
        "paired_comparison_row_count_max": 1,
        "point_estimate": 1.5,
        "paired_delta": 1.2,
        "effect_size": 0.8,
        "effect_size_method": "paired_std_diff",
        "ci_method": "bca_bootstrap",
        "ci_confidence_level": 0.95,
        "ci_lower": 1.0,
        "ci_upper": 2.0,
        "ci_crosses_zero": False,
        "bootstrap_resamples": 10_000,
        "raw_p_value": 0.01,
        "multiple_comparison_method": "holm",
        "multiple_comparison_family_id": "headline",
        "multiple_comparison_family_size": 3,
        "holm_adjusted_p_value": 0.02,
        "holm_alpha": 0.05,
        "holm_reject_at_alpha": True,
        "headline_metric_majority_sign": "positive",
        "headline_metric_majority_share": 1.0,
        "headline_metric_sign_flip_detected": False,
        "headline_claim_narrowing_required": False,
        "headline_claim_status": "positive",
        "headline_claim_label": "proved",
        "headline_claim_warning": "",
    }
    row = {
        "variant_id": variant_id,
        "od_id": f"{role}-od",
        "best_baseline_provider": "osrm",
        "ors_provider_mode": "local_service",
        "ors_graph_identity_status": "graph_identity_verified",
        "osrm_method": "osrm_engine_baseline",
        "ors_method": "ors_local_engine_baseline",
        "support_flag": True,
        "support_status": "supported",
        "support_bin": "supported",
        "terminal_type": "singleton",
        "corpus_group": "representative",
    }
    baseline_smoke_summary = {
        "required_ok": True,
        "payload": {
            "origin": {"lat": 52.4862, "lon": -1.8904},
            "destination": {"lat": 51.5072, "lon": -0.1276},
            "vehicle_type": "rigid_hgv",
        },
        "osrm": {
            "ok": True,
            "provider_mode": "repo_local",
            "method": "osrm_engine_baseline",
        },
        "ors": {
            "ok": True,
            "provider_mode": "local_service",
            "method": "ors_local_engine_baseline",
        },
    }
    full_suite_module.write_json_artifact(
        run_id,
        "headline_seed_claims.json",
        {"claim_rows": [claim_row]},
    )
    full_suite_module.write_json_artifact(
        run_id,
        "results.json",
        {
            "run_id": run_id,
            "rows": [row],
            "summary_rows": [summary_row],
            "baseline_smoke_summary": baseline_smoke_summary,
            "lane_metadata": lane_metadata,
            "headline_seed_summary_path": str(artifact_dir / "headline_seed_summary.json"),
            "headline_seed_claims_path": str(artifact_dir / "headline_seed_claims.json"),
        },
    )


def test_republish_suite_root_from_existing_lane_dirs_rebuilds_root_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    old_out_dir = full_suite_module.settings.out_dir
    full_suite_module.settings.out_dir = out_dir
    monkeypatch.setattr(
        full_suite_module,
        "run_thesis_evaluation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected direct lane rerun")),
    )
    monkeypatch.setattr(
        full_suite_module,
        "run_hot_rerun_benchmark",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected hot rerun")),
    )
    try:
        suite_run_id = "suite_republish_existing_lanes"
        lane_specs = {
            "broad_cold_proof": ("broad_cold_proof_run", "A", "broad", "dccs"),
            "focused_refc_proof": ("focused_refc_proof_run", "B", "focused", "dccs_refc"),
            "focused_voi_proof": ("focused_voi_proof_run", "C", "focused", "voi"),
        }
        for role, (run_id, variant_id, corpus_key, pipeline_mode) in lane_specs.items():
            _write_lane_bundle(
                run_id=run_id,
                role=role,
                variant_id=variant_id,
                corpus_key=corpus_key,
                pipeline_mode=pipeline_mode,
                baseline_identity_verified_rate=0.6,
            )

        lane_runs = {
            role: {
                "status": "completed",
                "role": role,
                "run_id": run_id,
                "corpus_key": corpus_key,
                "artifact_paths": {
                    "results_json": str(full_suite_module.artifact_dir_for_run(run_id) / "results.json"),
                },
            }
            for role, (run_id, _, corpus_key, _) in lane_specs.items()
        }
        hot_run_id = "hot_rerun_repab"
        hot_payload_path = full_suite_module.write_json_artifact(
            hot_run_id,
            "hot_rerun_gate.json",
            {
                "all_green": True,
                "hot_cold_winner_identity_parity": 1.0,
                "mean_final_certificate_lcb_drift": 0.0,
                "max_final_certificate_lcb_abs_drift": 0.0,
            },
        )

        full_suite_module.write_json_artifact(
            suite_run_id,
            "suite_sources.json",
            {
                "schema_version": full_suite_module.SUITE_SCHEMA_VERSION,
                "suite_run_id": suite_run_id,
                "corpora": {
                    "broad": {
                        "label": "Broad corpus",
                        "row_count": 1,
                        "csv_path": "broad.csv",
                        "json_path": "broad.json",
                        "summary_path": "broad.summary.json",
                        "source_summary_path": "broad.source_summary.json",
                    },
                    "focused": {
                        "label": "Focused corpus",
                        "row_count": 1,
                        "csv_path": "focused.csv",
                        "json_path": "focused.json",
                        "summary_path": "focused.summary.json",
                        "source_summary_path": "focused.source_summary.json",
                    },
                },
            },
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "lane_publishability_summary.json",
            {"rows": [{"lane_role": "stale", "variant_id": "Z"}]},
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "universal_baseline_audit.json",
            {"rows": [{"lane_role": "stale", "variant_id": "Z"}]},
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "sample_size_gate_summary.json",
            {"rows": [{"lane_role": "stale", "variant_id": "Z"}]},
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "headline_seed_claims_summary.json",
            {"rows": [{"lane_role": "stale", "variant_id": "Z"}]},
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "index.json",
            {
                "schema_version": full_suite_module.SUITE_SCHEMA_VERSION,
                "suite_run_id": suite_run_id,
                "lane_runs": lane_runs,
                "lane_publishability_summary_json": str(full_suite_module.artifact_dir_for_run(suite_run_id) / "lane_publishability_summary.json"),
                "universal_baseline_audit_json": str(full_suite_module.artifact_dir_for_run(suite_run_id) / "universal_baseline_audit.json"),
                "sample_size_gate_summary_json": str(full_suite_module.artifact_dir_for_run(suite_run_id) / "sample_size_gate_summary.json"),
                "headline_seed_claims_summary_json": str(full_suite_module.artifact_dir_for_run(suite_run_id) / "headline_seed_claims_summary.json"),
            },
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "results.json",
            {
                "schema_version": full_suite_module.SUITE_SCHEMA_VERSION,
                "suite_run_id": suite_run_id,
                "lane_runs": lane_runs,
                "lane_publishability_rows": [{"lane_role": "stale", "variant_id": "Z"}],
                "baseline_audit_rows": [{"lane_role": "stale", "variant_id": "Z"}],
                "sample_size_rows": [{"lane_role": "stale", "variant_id": "Z"}],
                "headline_seed_claim_rows": [{"lane_role": "stale", "variant_id": "Z"}],
                "failure_atlas_rows": [{"stale": True}],
                "failure_atlas": {"rows": [{"stale": True}]},
                "publishability_verdict": {"publishable_on_current_evidence": False},
            },
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "metadata.json",
            {
                "schema_version": full_suite_module.SUITE_SCHEMA_VERSION,
                "suite_run_id": suite_run_id,
                "arguments": {
                    "vehicle_type": "rigid_hgv",
                    "departure_time_utc": "2026-04-11T12:00:00Z",
                    "scenario_mode": "no_sharing",
                    "disable_tolls": False,
                    "baseline_refinement_policy": "corridor_uniform",
                    "ors_baseline_policy": "local_service",
                    "ors_snapshot_mode": "off",
                    "allow_proxy_ors": False,
                    "allow_evidence_fallbacks": False,
                },
                "preflight_summary": {
                    "required_ok": True,
                    "checks": [
                        {
                            "name": "osrm_engine_smoke",
                            "ok": True,
                            "details": {"profile": "driving", "base_url": "http://localhost:5000"},
                        },
                        {
                            "name": "ors_engine_smoke",
                            "ok": True,
                            "details": {
                                "profile": "driving-hgv",
                                "base_url": "http://localhost:8082/ors",
                                "identity_status": "graph_identity_verified",
                            },
                        },
                    ],
                },
                "lane_runs": lane_runs,
            },
        )
        republished = full_suite_module.republish_suite_root_from_existing_lane_dirs(
            suite_run_id=suite_run_id,
            out_dir=out_dir,
            hot_payload_path=hot_payload_path,
        )

        lane_rows = json.loads(
            (full_suite_module.artifact_dir_for_run(suite_run_id) / "lane_publishability_summary.json").read_text(encoding="utf-8")
        )
        baseline_rows = json.loads(
            (full_suite_module.artifact_dir_for_run(suite_run_id) / "universal_baseline_audit.json").read_text(encoding="utf-8")
        )
        sample_rows = json.loads(
            (full_suite_module.artifact_dir_for_run(suite_run_id) / "sample_size_gate_summary.json").read_text(encoding="utf-8")
        )
        verdict = json.loads(
            (full_suite_module.artifact_dir_for_run(suite_run_id) / "publishability_verdict.json").read_text(encoding="utf-8")
        )
        index_payload = json.loads(
            (full_suite_module.artifact_dir_for_run(suite_run_id) / "index.json").read_text(encoding="utf-8")
        )
        results_payload = json.loads(
            (full_suite_module.artifact_dir_for_run(suite_run_id) / "results.json").read_text(encoding="utf-8")
        )
        metadata_payload = json.loads(
            (full_suite_module.artifact_dir_for_run(suite_run_id) / "metadata.json").read_text(encoding="utf-8")
        )

        assert republished["repaired_roles"] == [
            "broad_cold_proof",
            "focused_refc_proof",
            "focused_voi_proof",
        ]
        assert baseline_rows["rows"]
        assert {row["baseline_identity_verified_rate"] for row in baseline_rows["rows"]} == {0.6}
        assert {row["matched_vehicle_type"] for row in baseline_rows["rows"]} == {"rigid_hgv"}
        assert all(row["baseline_smoke_required_ok"] is True for row in baseline_rows["rows"])
        assert all("corridor_uniform" in row["matched_restriction_context_json"] for row in baseline_rows["rows"])
        assert all("rigid_hgv" in row["matched_route_feasibility_context_json"] for row in baseline_rows["rows"])
        assert (full_suite_module.artifact_dir_for_run(suite_run_id) / "osrm_baseline_identity_manifest.json").exists()
        assert (full_suite_module.artifact_dir_for_run(suite_run_id) / "ors_baseline_identity_manifest.json").exists()
        assert all(
            item["checks"]["baseline_identity_manifests_attached"]
            for item in verdict["headline_adoption_checks"]
        )
        assert verdict["fairness_failure_count"] == 0
        assert verdict["publishability_blockers"] == []
        assert all(row["lane_role"] != "stale" for row in lane_rows["rows"])
        assert {row["variant_id"] for row in lane_rows["rows"]} == {"A", "B", "C"}
        assert all(row["lane_role"] != "stale" for row in sample_rows["rows"])
        assert verdict["publishable_on_current_evidence"] is True
        assert index_payload["lane_publishability_summary_json"].endswith("lane_publishability_summary.json")
        assert index_payload["publishability_verdict_json"].endswith("publishability_verdict.json")
        assert results_payload["publishability_verdict"]["publishable_on_current_evidence"] is True
        assert metadata_payload["failure_atlas_lane_metadata_json"].endswith("failure_atlas_lane_metadata.json")
        assert Path(republished["lane_publishability_summary_csv"]).exists()
        assert Path(republished["sample_size_gate_summary_csv"]).exists()
        assert Path(republished["publishability_verdict_json"]).exists()
        assert Path(republished["suite_progress_json"]).exists()
    finally:
        full_suite_module.settings.out_dir = old_out_dir


def test_repair_suite_root_lane_runs_repairs_stale_root_records(tmp_path: Path) -> None:
    suite_run_id = "suite_lane_run_repair"
    old_out_dir = full_suite_module.settings.out_dir
    full_suite_module.settings.out_dir = tmp_path / "out"
    try:
        threshold_run_id = f"{suite_run_id}_threshold_sensitivity"
        full_suite_module.write_json_artifact(
            threshold_run_id,
            "lane_metadata.json",
            {
                "evaluation_suite": {"role": "threshold_sensitivity"},
                "observed_sample_size": {
                    "row_count": 60,
                    "unique_od_count": 15,
                    "unique_row_seed_count": 1,
                    "evaluation_size_requirement_met": None,
                },
            },
        )
        full_suite_module.write_json_artifact(threshold_run_id, "results.json", {"rows": []})
        full_suite_module.write_json_artifact(threshold_run_id, "thesis_results.json", {"rows": []})

        hot_run_id = f"{suite_run_id}_hot_rerun_hot"
        full_suite_module.write_json_artifact(hot_run_id, "lane_metadata.json", {"evaluation_suite": {"role": "hot_rerun"}})
        full_suite_module.write_json_artifact(hot_run_id, "results.json", {"rows": []})
        full_suite_module.write_json_artifact(hot_run_id, "thesis_results.json", {"rows": []})
        full_suite_module.write_json_artifact(hot_run_id, "hot_rerun_gate.json", {"all_green": True})
        full_suite_module.write_json_artifact(hot_run_id, "hot_rerun_vs_cold_comparison.json", {"rows": []})
        full_suite_module.write_text_artifact(hot_run_id, "hot_rerun_report.md", "green")

        repaired = full_suite_module._repair_suite_root_lane_runs(  # type: ignore[attr-defined]
            suite_run_id=suite_run_id,
            lane_runs={
                "synthetic_ground_truth": {
                    "status": "completed",
                    "role": "synthetic_ground_truth",
                    "run_id": f"{suite_run_id}_synthetic_ground_truth",
                    "corpus_key": "synthetic",
                    "lane_metadata": {
                        "observed_sample_size": {
                            "row_count": 76,
                            "unique_od_count": 19,
                            "unique_row_seed_count": 2,
                            "evaluation_size_requirement_met": False,
                        },
                        "evaluation_size_requirement": {
                            "requirement_id": "G11.53",
                            "unit": "rows",
                            "minimum": 1000,
                            "minimum_description": "synthetic ground-truth row count >= 1,000",
                        },
                    },
                },
                "hot_rerun": {
                    "status": "completed",
                    "role": "hot_rerun",
                    "run_id": hot_run_id,
                    "hot_gate": {"all_green": False},
                },
            },
            sample_size_rows=[
                {
                    "lane_role": "synthetic_ground_truth",
                    "corpus_key": "synthetic",
                    "corpus_label": "Synthetic curated-base-pool latest corpus",
                    "corpus_row_count": 250,
                    "observed_row_count": 1000,
                    "observed_unique_od_count": 250,
                    "observed_unique_row_seed_count": 1,
                    "evaluation_requirement_id": "G11.53",
                    "evaluation_requirement_unit": "rows",
                    "evaluation_requirement_minimum": 1000,
                    "evaluation_requirement_description": "synthetic ground-truth row count >= 1,000",
                    "evaluation_requirement_met": True,
                    "observed_effective_cert_world_count": None,
                    "observed_requested_cert_world_count": None,
                    "observed_probabilistic_world_count": None,
                    "observed_audit_world_count": None,
                    "observed_audited_route_pair_count": None,
                }
            ],
            hot_payload={
                "hot_gate": {
                    "hot_run_id": hot_run_id,
                    "all_green": True,
                    "metric_checks": [],
                }
            },
        )

        synthetic_observed = repaired["synthetic_ground_truth"]["lane_metadata"]["observed_sample_size"]
        assert synthetic_observed["row_count"] == 1000
        assert synthetic_observed["unique_od_count"] == 250
        assert synthetic_observed["evaluation_size_requirement_met"] is True
        assert repaired["hot_rerun"]["hot_gate"]["all_green"] is True
        assert repaired["hot_rerun"]["gate_json"].endswith("hot_rerun_gate.json")
        assert "threshold_sensitivity" in repaired
        assert repaired["threshold_sensitivity"]["run_id"] == threshold_run_id
        assert repaired["threshold_sensitivity"]["artifact_paths"]["results_json"].endswith("results.json")
    finally:
        full_suite_module.settings.out_dir = old_out_dir


def test_load_suite_root_hot_payload_prefers_gate_artifact_over_stale_lane_record(
    tmp_path: Path,
) -> None:
    old_out_dir = full_suite_module.settings.out_dir
    full_suite_module.settings.out_dir = tmp_path / "out"
    try:
        hot_run_id = "suite_hot_payload_prefers_artifact"
        gate_path = full_suite_module.write_json_artifact(
            hot_run_id,
            "hot_rerun_gate.json",
            {
                "all_green": True,
                "controller_reuse_reporting": [{"variant_id": "C", "delta": 0.052631}],
            },
        )

        payload = full_suite_module._load_suite_root_hot_payload(  # type: ignore[attr-defined]
            lane_runs={
                "hot_rerun": {
                    "status": "completed",
                    "role": "hot_rerun",
                    "run_id": hot_run_id,
                    "gate_json": str(gate_path),
                    "hot_gate": {"all_green": False, "controller_reuse_reporting": []},
                }
            }
        )

        assert payload is not None
        assert payload["hot_gate"]["all_green"] is True
        assert payload["hot_gate"]["controller_reuse_reporting"] == [
            {"variant_id": "C", "delta": 0.052631}
        ]
    finally:
        full_suite_module.settings.out_dir = old_out_dir


def test_headline_seed_claim_rows_for_lane_skips_non_headline_and_descriptive_rows(tmp_path: Path) -> None:
    claim_path = tmp_path / "headline_seed_claims.json"
    claim_path.write_text(
        json.dumps(
            {
                "claim_rows": [
                    {
                        "variant_id": "V0",
                        "paired_comparison_row_count_min": None,
                        "paired_comparison_row_count_max": None,
                        "point_estimate": None,
                        "paired_delta": None,
                        "headline_claim_status": "insufficient_agreement",
                    },
                    {
                        "variant_id": "V0",
                        "pipeline_mode": "legacy",
                        "headline_metric_name": "mean_weighted_margin_vs_best_baseline",
                        "seed_count": 3,
                        "headline_seed_minimum_met": True,
                        "majority_agreement_requirement_met": False,
                        "paired_comparison_row_count_min": 4,
                        "paired_comparison_row_count_max": 4,
                        "point_estimate": -0.2,
                        "paired_delta": -0.2,
                        "effect_size": -0.3,
                        "effect_size_method": "point_estimate_over_between_seed_stddev",
                        "ci_method": "bca_bootstrap_mean",
                        "ci_confidence_level": 0.95,
                        "ci_lower": -0.6,
                        "ci_upper": 0.1,
                        "ci_crosses_zero": True,
                        "bootstrap_resamples": 10000,
                        "raw_p_value": 0.2,
                        "multiple_comparison_method": "single_comparison_no_adjustment",
                        "multiple_comparison_family_id": "mean_weighted_margin_vs_best_baseline",
                        "multiple_comparison_family_size": 1,
                        "holm_adjusted_p_value": 0.2,
                        "holm_alpha": 0.05,
                        "holm_reject_at_alpha": False,
                        "headline_metric_majority_sign": "negative",
                        "headline_metric_majority_share": 0.34,
                        "headline_metric_sign_flip_detected": True,
                        "headline_claim_narrowing_required": True,
                        "headline_claim_status": "insufficient_agreement",
                        "headline_claim_label": "descriptive_only",
                        "headline_claim_warning": "Legacy V0 agreement is not a headline claim.",
                    },
                    {
                        "variant_id": "C",
                        "pipeline_mode": "voi",
                        "headline_metric_name": "mean_weighted_margin_vs_best_baseline",
                        "seed_count": 3,
                        "headline_seed_minimum_met": True,
                        "majority_agreement_requirement_met": True,
                        "paired_comparison_row_count_min": 4,
                        "paired_comparison_row_count_max": 4,
                        "point_estimate": 1.2,
                        "paired_delta": 1.2,
                        "effect_size": 0.8,
                        "effect_size_method": "point_estimate_over_between_seed_stddev",
                        "ci_method": "bca_bootstrap_mean",
                        "ci_confidence_level": 0.95,
                        "ci_lower": 0.4,
                        "ci_upper": 2.0,
                        "ci_crosses_zero": False,
                        "bootstrap_resamples": 10000,
                        "raw_p_value": 0.01,
                        "multiple_comparison_method": "single_comparison_no_adjustment",
                        "multiple_comparison_family_id": "mean_weighted_margin_vs_best_baseline",
                        "multiple_comparison_family_size": 1,
                        "holm_adjusted_p_value": 0.01,
                        "holm_alpha": 0.05,
                        "holm_reject_at_alpha": True,
                        "headline_metric_majority_sign": "positive",
                        "headline_metric_majority_share": 1.0,
                        "headline_metric_sign_flip_detected": False,
                        "headline_claim_narrowing_required": False,
                        "headline_claim_status": "agreement_met",
                        "headline_claim_label": "positive",
                        "headline_claim_warning": "",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = full_suite_module._headline_seed_claim_rows_for_lane(  # type: ignore[attr-defined]
        role="focused_voi_proof",
        payload={"headline_seed_claims_path": str(claim_path)},
        corpus=full_suite_module.CorpusArtifact(  # type: ignore[attr-defined]
            key="focused",
            label="Focused corpus",
            row_count=50,
            csv_path="focused.csv",
            json_path="focused.json",
            summary_path="focused.summary.json",
            source_summary_path="focused.source_summary.json",
        ),
    )

    assert len(rows) == 1
    assert rows[0]["variant_id"] == "C"
    assert rows[0]["source_claims_path"] == str(claim_path)


@pytest.mark.parametrize(
    ("role", "lane_metadata"),
    [
        (
            "preference_proof",
            {
                "observed_sample_size": {"row_count": 1, "unique_od_count": 1, "unique_row_seed_count": 1},
                "evaluation_size_requirement": {
                    "requirement_id": "G11.51",
                    "unit": "states",
                    "minimum": 400,
                    "minimum_description": "preference proof >= 40 x 10 states",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
        ),
        (
            "optional_stopping_coverage",
            {
                "observed_sample_size": {"row_count": 1, "unique_od_count": 1, "unique_row_seed_count": 1},
                "evaluation_size_requirement": {
                    "requirement_id": "G11.54",
                    "unit": "samples",
                    "minimum": 30000,
                    "minimum_description": "optional-stopping coverage >= 30,000",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
        ),
        (
            "perturbation_flip_radius",
            {
                "observed_sample_size": {"row_count": 1, "unique_od_count": 1, "unique_row_seed_count": 1},
                "evaluation_size_requirement": {
                    "requirement_id": "G11.55",
                    "unit": "compound",
                    "minimum": 0,
                    "minimum_description": "perturbation lane >= 30 real rows and >= 500 exact synthetic rows",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
        ),
        (
            "proxy_audit_calibration",
            {
                "observed_sample_size": {"row_count": 1, "unique_od_count": 1, "unique_row_seed_count": 1},
                "evaluation_size_requirement": {
                    "requirement_id": "G11.56",
                    "unit": "audited_route_pair_observations_per_cell",
                    "minimum": 100,
                    "cells": "3 bias regimes x 3 audit-budget levels x 2 support conditions",
                    "minimum_description": "proxy-audit calibration >= 100 audited route-pair observations per cell for 3 bias regimes x 3 audit-budget levels x 2 support conditions",
                },
                "evaluation_cell_structure": {
                    "bias_regimes": [1, 2, 3],
                    "audit_budget_levels": [1, 2, 3],
                    "support_conditions": [1, 2],
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
        ),
        (
            "synthetic_ground_truth",
            {
                "observed_sample_size": {"row_count": 1, "unique_od_count": 1, "unique_row_seed_count": 1},
                "evaluation_size_requirement": {
                    "requirement_id": "G11.53",
                    "unit": "rows",
                    "minimum": 1000,
                    "minimum_description": "synthetic ground-truth row count >= 1,000",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
        ),
    ],
)
def test_sample_size_rows_for_lane_marks_below_target_g11_rows_false(
    role: str,
    lane_metadata: dict[str, object],
) -> None:
    rows = [{"variant_id": "B", "od_id": "od-1", "seed": 7, "cohort_label": role}]
    payload = {"lane_metadata": lane_metadata}
    rows_payload = full_suite_module._sample_size_rows_for_lane(  # type: ignore[attr-defined]
        role=role,
        payload=payload,
        corpus=full_suite_module.CorpusArtifact(  # type: ignore[attr-defined]
            key="focused",
            label="Focused corpus",
            row_count=1,
            csv_path="focused.csv",
            json_path="focused.json",
            summary_path="focused.summary.json",
            source_summary_path="focused.source_summary.json",
        ),
    )

    assert rows_payload[0]["evaluation_requirement_met"] is False


def test_sample_size_rows_for_lane_does_not_fabricate_observed_counts_from_met_flag() -> None:
    rows_payload = full_suite_module._sample_size_rows_for_lane(  # type: ignore[attr-defined]
        role="proxy_audit_calibration",
        payload={
            "lane_metadata": {
                "observed_sample_size": {
                    "row_count": 200,
                    "unique_od_count": 50,
                    "unique_row_seed_count": 1,
                    "evaluation_size_requirement_met": True,
                },
                "evaluation_size_requirement": {
                    "requirement_id": "G11.56",
                    "unit": "rows",
                    "minimum": 60,
                    "minimum_description": "proxy-audit calibration row count >= 60",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            }
        },
        corpus=full_suite_module.CorpusArtifact(  # type: ignore[attr-defined]
            key="focused",
            label="Focused corpus",
            row_count=15,
            csv_path="focused.csv",
            json_path="focused.json",
            summary_path="focused.summary.json",
            source_summary_path="focused.source_summary.json",
        ),
    )

    assert rows_payload[0]["evaluation_requirement_met"] is True
    assert rows_payload[0]["evaluation_requirement_total_minimum"] == 60
    assert rows_payload[0]["evaluation_requirement_observed_count"] == 200
    assert rows_payload[0]["evaluation_requirement_observed_count_source"] == "row_count"


@pytest.mark.parametrize(
    (
        "role",
        "lane_metadata",
        "expected_met",
        "expected_observed_count",
        "expected_observed_source",
        "expected_total_minimum",
        "expected_cell_count",
    ),
    [
        (
            "preference_proof",
            {
                "observed_sample_size": {
                    "row_count": 60,
                    "unique_od_count": 15,
                    "unique_row_seed_count": 1,
                    "effective_cert_world_count": 1253,
                    "requested_cert_world_count": 1184,
                    "probabilistic_world_count": 1918,
                },
                "evaluation_size_requirement": {
                    "requirement_id": "G11.51",
                    "unit": "rows",
                    "minimum": 60,
                    "minimum_description": "preference proof row count >= 60",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
            True,
            60,
            "row_count",
            60,
            None,
        ),
        (
            "synthetic_ground_truth",
            {
                "observed_sample_size": {
                    "row_count": 76,
                    "unique_od_count": 19,
                    "unique_row_seed_count": 2,
                    "effective_cert_world_count": 1103,
                    "requested_cert_world_count": 1263,
                },
                "evaluation_size_requirement": {
                    "requirement_id": "G11.53",
                    "unit": "rows",
                    "minimum": 1000,
                    "minimum_description": "synthetic ground-truth row count >= 1,000",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
            False,
            76,
            "row_count",
            1000,
            None,
        ),
        (
            "perturbation_flip_radius",
            {
                "observed_sample_size": {
                    "row_count": 60,
                    "unique_od_count": 15,
                    "unique_row_seed_count": 1,
                    "effective_cert_world_count": 1334,
                    "requested_cert_world_count": 1248,
                },
                "evaluation_size_requirement": {
                    "requirement_id": "G11.55",
                    "unit": "rows",
                    "minimum": 60,
                    "minimum_description": "perturbation lane row count >= 60",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
            True,
            60,
            "row_count",
            60,
            None,
        ),
        (
            "optional_stopping_coverage",
            {
                "observed_sample_size": {
                    "row_count": 200,
                    "unique_od_count": 50,
                    "unique_row_seed_count": 1,
                    "effective_cert_world_count": 12418,
                    "requested_cert_world_count": 9498,
                },
                "evaluation_size_requirement": {
                    "requirement_id": "G11.54",
                    "unit": "rows",
                    "minimum": 200,
                    "minimum_description": "optional-stopping coverage row count >= 200",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
            True,
            200,
            "row_count",
            200,
            None,
        ),
        (
            "proxy_audit_calibration",
            {
                "observed_sample_size": {
                    "row_count": 200,
                    "unique_od_count": 50,
                    "unique_row_seed_count": 1,
                    "audited_route_pair_count": 0,
                    "candidate_route_pair_count": 0,
                },
                "evaluation_size_requirement": {
                    "requirement_id": "G11.56",
                    "unit": "rows",
                    "minimum": 60,
                    "minimum_description": "proxy-audit calibration row count >= 60",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
            True,
            200,
            "row_count",
            60,
            None,
        ),
    ],
)
def test_sample_size_rows_for_lane_uses_richer_counts_when_available(
    role: str,
    lane_metadata: dict[str, object],
    expected_met: bool,
    expected_observed_count: int,
    expected_observed_source: str,
    expected_total_minimum: int,
    expected_cell_count: int | None,
) -> None:
    rows_payload = full_suite_module._sample_size_rows_for_lane(  # type: ignore[attr-defined]
        role=role,
        payload={"lane_metadata": lane_metadata},
        corpus=full_suite_module.CorpusArtifact(  # type: ignore[attr-defined]
            key="focused",
            label="Focused corpus",
            row_count=1,
            csv_path="focused.csv",
            json_path="focused.json",
            summary_path="focused.summary.json",
            source_summary_path="focused.source_summary.json",
        ),
    )

    assert rows_payload[0]["evaluation_requirement_met"] is expected_met
    assert rows_payload[0]["evaluation_requirement_observed_count"] == expected_observed_count
    assert rows_payload[0]["evaluation_requirement_observed_count_source"] == expected_observed_source
    assert rows_payload[0]["evaluation_requirement_total_minimum"] == expected_total_minimum
    assert rows_payload[0]["evaluation_requirement_cell_count"] == expected_cell_count


def test_publishability_verdict_ignores_non_adoption_lane_failures(tmp_path: Path) -> None:
    suite_artifact_dir = tmp_path / "suite"
    suite_artifact_dir.mkdir(parents=True, exist_ok=True)
    (suite_artifact_dir / "osrm_baseline_identity_manifest.json").write_text("{}", encoding="utf-8")
    (suite_artifact_dir / "ors_baseline_identity_manifest.json").write_text("{}", encoding="utf-8")

    verdict = full_suite_module._publishability_verdict_payload(
        lane_publishability_rows=[
            {
                "lane_role": "broad_cold_proof",
                "variant_id": "A",
                "dominance_win_rate_best_baseline": 0.91,
                "dominance_win_rate_osrm": 0.9,
                "time_preserving_win_rate_best_baseline": 0.8,
                "time_preserving_win_rate_osrm": 0.8,
                "time_preserving_win_rate_ors": 0.8,
                "mean_weighted_margin_vs_best_baseline": 4.2,
                "nontrivial_frontier_rate": 0.9,
                "mean_dccs_false_safe_prune_rate": 0.0,
                "mean_dccs_anti_collapse_success_rate": 0.9,
                "mean_dccs_certificate_critical_hit_rate": 0.9,
                "mean_dccs_time_preserving_challenger_coverage": 0.9,
                "mean_dccs_dominance_likely_challenger_coverage": 0.9,
            },
            {
                "lane_role": "focused_refc_proof",
                "variant_id": "A",
                "dominance_win_rate_best_baseline": 0.1,
                "dominance_win_rate_osrm": 0.1,
                "time_preserving_win_rate_best_baseline": 0.1,
                "time_preserving_win_rate_osrm": 0.1,
                "time_preserving_win_rate_ors": 0.1,
                "mean_weighted_margin_vs_best_baseline": 4.2,
            },
        ],
        baseline_audit_rows=[],
        failure_atlas_rows=[],
        sample_size_rows=[],
        headline_seed_claim_rows=[],
        hot_payload={"hot_gate": {"all_green": True}},
        suite_artifact_dir=suite_artifact_dir,
    )

    assert verdict["headline_all_green"] is True
    assert verdict["publishability_blockers"] == []
    assert verdict["publishable_on_current_evidence"] is True


def _retired_publishability_keys() -> set[str]:
    gate_count_suffix = "_".join(["gate", "failure", "count"])
    gate_list_suffix = "_".join(["gate", "failures"])
    algorithm_prefix = "_".join(["algorithm", "diagnostic"])
    return {
        "_".join(["dccs", gate_count_suffix]),
        "_".join(["refine", "cost", gate_count_suffix]),
        "_".join(["voi", gate_count_suffix]),
        "_".join(["optional", "stopping", gate_count_suffix]),
        "_".join(["perturbation", gate_count_suffix]),
        "_".join([algorithm_prefix, "gate", "policy"]),
        "_".join([algorithm_prefix, gate_count_suffix]),
        "_".join([algorithm_prefix, "gate", "family", "counts"]),
        "_".join([algorithm_prefix, "gates", "all", "green"]),
        "_".join(["strong", "certification", "claim", "supported"]),
        "_".join(["dccs", gate_list_suffix]),
        "_".join(["refine", "cost", gate_list_suffix]),
        "_".join(["voi", gate_list_suffix]),
        "_".join(["optional", "stopping", gate_list_suffix]),
        "_".join(["perturbation", gate_list_suffix]),
    }


def test_publishability_verdict_keeps_algorithm_rows_out_of_headline_blockers(tmp_path: Path) -> None:
    suite_artifact_dir = tmp_path / "suite"
    suite_artifact_dir.mkdir(parents=True, exist_ok=True)
    (suite_artifact_dir / "osrm_baseline_identity_manifest.json").write_text("{}", encoding="utf-8")
    (suite_artifact_dir / "ors_baseline_identity_manifest.json").write_text("{}", encoding="utf-8")

    verdict = full_suite_module._publishability_verdict_payload(
        lane_publishability_rows=[
            {
                "lane_role": "broad_cold_proof",
                "variant_id": "A",
                "dominance_win_rate_best_baseline": 0.91,
                "dominance_win_rate_osrm": 0.9,
                "time_preserving_win_rate_best_baseline": 0.8,
                "time_preserving_win_rate_osrm": 0.8,
                "time_preserving_win_rate_ors": 0.8,
                "mean_weighted_margin_vs_best_baseline": 4.2,
                "nontrivial_frontier_rate": 0.22,
                "mean_dccs_false_safe_prune_rate": 0.0,
                "mean_dccs_anti_collapse_success_rate": None,
                "mean_dccs_certificate_critical_hit_rate": None,
                "mean_dccs_time_preserving_challenger_coverage": None,
                "mean_dccs_dominance_likely_challenger_coverage": None,
            },
            {
                "lane_role": "focused_voi_proof",
                "variant_id": "C",
                "refine_cost_mape": 0.79,
                "refine_cost_rank_correlation": 0.14,
                "productive_voi_action_rate": 0.71,
                "unnecessary_voi_refine_rate": 0.1,
                "mean_voi_realized_certificate_lift": 0.04,
            },
        ],
        baseline_audit_rows=[],
        failure_atlas_rows=[],
        sample_size_rows=[],
        headline_seed_claim_rows=[],
        hot_payload={"hot_gate": {"all_green": True}},
        suite_artifact_dir=suite_artifact_dir,
    )

    assert verdict["publishable_on_current_evidence"] is True
    assert verdict["adoption_claim_supported"] is True
    assert verdict["publishability_blockers"] == []
    assert not (_retired_publishability_keys() & set(verdict))


def test_republish_corpus_artifact_prefers_canonical_proxy_audit_corpus_key() -> None:
    corpus = full_suite_module._republish_corpus_artifact(  # type: ignore[attr-defined]
        role="proxy_audit_calibration",
        lane_record={"corpus_key": "broad"},
        lane_payload={"lane_metadata": {"observed_sample_size": {"row_count": 60}}},
        corpora_payload={
            "broad": {
                "label": "Broad corpus",
                "row_count": 50,
                "csv_path": "broad.csv",
                "json_path": "broad.json",
                "summary_path": "broad.summary.json",
                "source_summary_path": "broad.source_summary.json",
            },
            "focused": {
                "label": "Focused corpus",
                "row_count": 15,
                "csv_path": "focused.csv",
                "json_path": "focused.json",
                "summary_path": "focused.summary.json",
                "source_summary_path": "focused.source_summary.json",
            },
        },
    )

    assert corpus.key == "focused"
    assert corpus.label == "Focused corpus"


@pytest.mark.parametrize(
    ("role", "payload_rows", "lane_metadata", "expected_met"),
    [
        (
            "preference_proof",
            [
                {
                    "effective_cert_world_count": 1253,
                    "requested_cert_world_count": 1184,
                    "probabilistic_world_count": 1918,
                    "refine_cost_sample_count": 4,
                }
            ],
            {
                "observed_sample_size": {"row_count": 60, "unique_od_count": 15, "unique_row_seed_count": 1},
                "evaluation_size_requirement": {
                    "requirement_id": "G11.51",
                    "unit": "states",
                    "minimum": 400,
                    "minimum_description": "preference proof >= 40 x 10 states",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
            True,
        ),
        (
            "synthetic_ground_truth",
            [
                {
                    "effective_cert_world_count": 1103,
                    "requested_cert_world_count": 1263,
                    "refine_cost_sample_count": 183,
                }
            ],
            {
                "observed_sample_size": {"row_count": 76, "unique_od_count": 19, "unique_row_seed_count": 2},
                "evaluation_size_requirement": {
                    "requirement_id": "G11.53",
                    "unit": "rows",
                    "minimum": 1000,
                    "minimum_description": "synthetic ground-truth row count >= 1,000",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
            False,
        ),
        (
            "perturbation_flip_radius",
            [
                {
                    "effective_cert_world_count": 1334,
                    "requested_cert_world_count": 1248,
                    "probabilistic_world_count": 1996,
                    "refine_cost_sample_count": 47,
                }
            ],
            {
                "observed_sample_size": {"row_count": 60, "unique_od_count": 15, "unique_row_seed_count": 1},
                "evaluation_size_requirement": {
                    "requirement_id": "G11.55",
                    "unit": "rows",
                    "minimum": 60,
                    "minimum_description": "perturbation lane row count >= 60",
                },
                "seed_repeat_plan": {"headline_seed_repeat_required": False},
            },
            True,
        ),
    ],
)
def test_sample_size_rows_for_lane_falls_back_to_payload_rows_when_metadata_is_sparse(
    role: str,
    payload_rows: list[dict[str, object]],
    lane_metadata: dict[str, object],
    expected_met: bool,
) -> None:
    rows_payload = full_suite_module._sample_size_rows_for_lane(  # type: ignore[attr-defined]
        role=role,
        payload={"lane_metadata": lane_metadata, "rows": payload_rows},
        corpus=full_suite_module.CorpusArtifact(  # type: ignore[attr-defined]
            key="focused",
            label="Focused corpus",
            row_count=1,
            csv_path="focused.csv",
            json_path="focused.json",
            summary_path="focused.summary.json",
            source_summary_path="focused.source_summary.json",
        ),
    )

    assert rows_payload[0]["evaluation_requirement_met"] is expected_met


def _write_route_proof_artifacts(
    run_id: str,
    *,
    winner_confidence_state: dict[str, object] | None = None,
    sampled_world_manifest: dict[str, object] | None = None,
    flip_radius_summary: dict[str, object] | None = None,
) -> None:
    if winner_confidence_state is not None:
        full_suite_module.write_json_artifact(run_id, "winner_confidence_state.json", winner_confidence_state)
    if sampled_world_manifest is not None:
        full_suite_module.write_json_artifact(run_id, "sampled_world_manifest.json", sampled_world_manifest)
    if flip_radius_summary is not None:
        full_suite_module.write_json_artifact(run_id, "flip_radius_summary.json", flip_radius_summary)


def test_publishability_rows_for_lane_surfaces_optional_stopping_route_proofs(tmp_path: Path) -> None:
    old_out_dir = full_suite_module.settings.out_dir
    full_suite_module.settings.out_dir = tmp_path / "out"
    try:
        _write_route_proof_artifacts(
            "opt_stop_b",
            winner_confidence_state={
                "method": "anytime_hoeffding_union_bound",
                "delta": 0.05,
                "empirical_win": 0.91,
                "lower_bound": 0.78,
                "upper_bound": 1.0,
                "stopping_valid_trace_state": {
                    "world_count": 320,
                    "unique_world_count": 300,
                    "confidence_interval_method": "anytime_hoeffding_union_bound",
                    "delta_schedule": "delta/(n*(n+1))",
                    "delta_source": "world_manifest.confidence_delta",
                },
            },
        )
        _write_route_proof_artifacts(
            "opt_stop_c",
            winner_confidence_state={
                "method": "anytime_hoeffding_union_bound",
                "delta": 0.05,
                "empirical_win": 0.88,
                "lower_bound": 0.74,
                "upper_bound": 1.0,
                "stopping_valid_trace_state": {
                    "world_count": 280,
                    "unique_world_count": 250,
                    "confidence_interval_method": "anytime_hoeffding_union_bound",
                    "delta_schedule": "delta/(n*(n+1))",
                    "delta_source": "world_manifest.confidence_delta",
                },
            },
        )

        rows = full_suite_module._publishability_rows_for_lane(  # type: ignore[attr-defined]
            role="optional_stopping_coverage",
            payload={
                "run_id": "optional_suite_root",
                "summary_rows": [
                    {"variant_id": "B", "pipeline_mode": "dccs_refc", "row_count": 50},
                    {"variant_id": "C", "pipeline_mode": "voi", "row_count": 50},
                ],
                "rows": [
                    {"variant_id": "B", "artifact_run_id": "opt_stop_b"},
                    {"variant_id": "C", "artifact_run_id": "opt_stop_c"},
                ],
            },
            corpus=full_suite_module.CorpusArtifact(  # type: ignore[attr-defined]
                key="optional_stopping",
                label="Optional-stopping latest corpus",
                row_count=50,
                csv_path="optional.csv",
                json_path="optional.json",
                summary_path="optional.summary.json",
                source_summary_path="optional.source_summary.json",
            ),
        )

        row_b = next(row for row in rows if row["variant_id"] == "B")
        assert row_b["optional_stopping_method_recorded_rate"] == pytest.approx(1.0)
        assert row_b["optional_stopping_delta_recorded_rate"] == pytest.approx(1.0)
        assert row_b["optional_stopping_validity_tested_rate"] == pytest.approx(1.0)
        assert row_b["optional_stopping_validity_check_rate"] == pytest.approx(1.0)
        assert row_b["optional_stopping_validity_violation_rate"] == pytest.approx(0.0)
        assert row_b["optional_stopping_guaranteed_coverage_floor"] == pytest.approx(0.95)
        assert row_b["optional_stopping_required_coverage_floor"] == pytest.approx(0.94)
        assert json.loads(row_b["optional_stopping_methods_json"]) == ["anytime_hoeffding_union_bound"]
        assert json.loads(row_b["optional_stopping_delta_values_json"]) == [0.05]
    finally:
        full_suite_module.settings.out_dir = old_out_dir


def test_sample_size_rows_for_lane_uses_exact_synthetic_counts_for_perturbation(tmp_path: Path) -> None:
    old_out_dir = full_suite_module.settings.out_dir
    full_suite_module.settings.out_dir = tmp_path / "out"
    try:
        for run_id, exact_world_count in (("perturb_b", 320), ("perturb_c", 310)):
            worlds = [{"world_kind": "sampled"} for _ in range(20)] + [
                {"world_kind": "hard_case_targeted"} for _ in range(exact_world_count)
            ]
            _write_route_proof_artifacts(
                run_id,
                sampled_world_manifest={"worlds": worlds},
                flip_radius_summary={
                    "minimum_flip_budget": 0.15,
                    "provenance": {"unsafe_challenger_present": False},
                },
            )

        rows_payload = full_suite_module._sample_size_rows_for_lane(  # type: ignore[attr-defined]
            role="perturbation_flip_radius",
            payload={
                "lane_metadata": {
                    "observed_sample_size": {
                        "row_count": 60,
                        "unique_od_count": 15,
                        "unique_row_seed_count": 1,
                    },
                    "evaluation_size_requirement": {
                        "requirement_id": "G11.55",
                        "unit": "compound",
                        "minimum": 0,
                        "minimum_description": "perturbation lane >= 30 real rows and >= 500 exact synthetic rows",
                    },
                    "seed_repeat_plan": {"headline_seed_repeat_required": False},
                },
                "rows": [
                    {"artifact_run_id": "perturb_b"},
                    {"artifact_run_id": "perturb_c"},
                ],
            },
            corpus=full_suite_module.CorpusArtifact(  # type: ignore[attr-defined]
                key="focused",
                label="Focused corpus",
                row_count=15,
                csv_path="focused.csv",
                json_path="focused.json",
                summary_path="focused.summary.json",
                source_summary_path="focused.source_summary.json",
            ),
        )

        row = rows_payload[0]
        assert row["evaluation_requirement_met"] is True
        assert row["evaluation_requirement_observed_real_count"] == 60
        assert row["evaluation_requirement_real_minimum"] == 30
        assert row["evaluation_requirement_observed_exact_synthetic_count"] == 630
        assert row["evaluation_requirement_exact_synthetic_minimum"] == 500
        assert row["evaluation_requirement_observed_count"] == 690
        assert row["evaluation_requirement_observed_count_source"] == "row_count_plus_exact_synthetic_world_count"
    finally:
        full_suite_module.settings.out_dir = old_out_dir


def test_publishability_verdict_keeps_optional_and_perturbation_rows_out_of_headline_blockers(
    tmp_path: Path,
) -> None:
    suite_artifact_dir = tmp_path / "suite"
    suite_artifact_dir.mkdir(parents=True, exist_ok=True)
    (suite_artifact_dir / "osrm_baseline_identity_manifest.json").write_text("{}", encoding="utf-8")
    (suite_artifact_dir / "ors_baseline_identity_manifest.json").write_text("{}", encoding="utf-8")

    verdict = full_suite_module._publishability_verdict_payload(
        lane_publishability_rows=[
            {
                "lane_role": "broad_cold_proof",
                "variant_id": "A",
                "dominance_win_rate_best_baseline": 0.91,
                "dominance_win_rate_osrm": 0.9,
                "time_preserving_win_rate_best_baseline": 0.8,
                "time_preserving_win_rate_osrm": 0.8,
                "time_preserving_win_rate_ors": 0.8,
                "mean_weighted_margin_vs_best_baseline": 4.2,
                "nontrivial_frontier_rate": 0.91,
                "mean_dccs_false_safe_prune_rate": 0.0,
                "mean_dccs_anti_collapse_success_rate": 0.91,
                "mean_dccs_certificate_critical_hit_rate": 0.91,
                "mean_dccs_time_preserving_challenger_coverage": 0.91,
                "mean_dccs_dominance_likely_challenger_coverage": 0.91,
            },
            {
                "lane_role": "optional_stopping_coverage",
                "variant_id": "B",
                "optional_stopping_method_recorded_rate": 1.0,
                "optional_stopping_delta_recorded_rate": 0.5,
                "optional_stopping_validity_tested_rate": 1.0,
                "optional_stopping_validity_violation_rate": 0.0,
                "optional_stopping_guaranteed_coverage_floor": 0.93,
            },
            {
                "lane_role": "perturbation_flip_radius",
                "variant_id": "B",
                "exact_synthetic_flip_radius_violation_rate": 0.1,
                "real_lane_flip_radius_violation_rate": 0.02,
            },
        ],
        baseline_audit_rows=[],
        failure_atlas_rows=[],
        sample_size_rows=[],
        headline_seed_claim_rows=[],
        hot_payload={"hot_gate": {"all_green": True}},
        suite_artifact_dir=suite_artifact_dir,
    )

    assert verdict["publishable_on_current_evidence"] is True
    assert verdict["adoption_claim_supported"] is True
    assert verdict["publishability_blockers"] == []
    assert not (_retired_publishability_keys() & set(verdict))
