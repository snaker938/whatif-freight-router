from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
import uuid
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evidence_certification import validate_route_evidence_provenance
from app.model_data_errors import normalize_reason_code
from app.run_store import artifact_dir_for_run, write_csv_artifact, write_json_artifact, write_manifest, write_text_artifact
from app.settings import settings
from scripts.build_od_corpus_uk import (
    _ambiguity_derived_fields,
    _cheap_prior_features,
    _source_support_map_for_row,
    _source_support_strength,
    _split_prior_sources,
)
from scripts.enrich_od_corpus_with_ambiguity import (
    _raw_ambiguity_prior_value as _shared_raw_ambiguity_prior_value,
    _support_gated_budget_prior as _shared_support_gated_ambiguity_budget_prior,
)
from scripts.evaluation_metrics import (
    additive_epsilon_indicator,
    ambiguity_alignment,
    ambiguity_absolute_error,
    ambiguity_prior_overtrigger_rate,
    ambiguity_prior_top_k_precision,
    action_efficiency,
    as_float,
    balanced_gain_score,
    bytes_to_megabytes,
    certificate_margin,
    certificate_runner_up_gap,
    competitor_turnover_rate,
    controller_activation_on_high_ambiguity,
    coverage_of_baseline,
    corridor_family_recall,
    dominates,
    fragility_entropy,
    frontier_action_gain,
    frontier_diversity,
    frontier_diversity_index,
    frontier_entropy,
    frontier_gain_per_ms,
    hypervolume_3d,
    near_tie_mass,
    nominal_winner_margin,
    pairwise_weighted_sum_score,
    pearson_correlation,
    pearson_binary_correlation,
    percentile,
    quality_per_second,
    robust_win,
    route_metrics,
    cache_reuse_ratio,
    certificate_gain_per_world,
    controller_cost_per_certificate_point,
    memory_per_unit,
    productive_action_rate,
    refine_cost_positive_sample_count,
    refine_cost_sample_count,
    route_improvement_per_second,
    runtime_ratio,
    runtime_per_unit,
    runtime_share,
    score_ranked_recall,
    supported_ambiguity_alignment,
    time_to_best_iteration,
    value_per_second,
    refine_cost_mae_ms,
    refine_cost_mape,
    refine_cost_rank_correlation,
    refine_cost_zero_observed_count,
)
from scripts.preflight_live_runtime import run_preflight

VARIANTS: tuple[str, ...] = ("V0", "A", "B", "C")
THESIS_CACHE_MODES: tuple[str, ...] = ("preserve", "cold")
THESIS_COLD_CACHE_SCOPE = "thesis_cold"
HOT_RERUN_COLD_CACHE_SCOPE = "hot_rerun_cold_source"
THESIS_COLD_CACHE_SCOPES: tuple[str, ...] = (
    THESIS_COLD_CACHE_SCOPE,
    HOT_RERUN_COLD_CACHE_SCOPE,
)
VARIANT_PIPELINE_MODE = {"V0": "legacy", "A": "dccs", "B": "dccs_refc", "C": "voi"}
STRICT_EVIDENCE_POLICY = "no_synthetic_no_proxy_no_fallback"
EVALUATION_SUITE_ROLE_DEFAULTS: dict[str, dict[str, str]] = {
    "generic_evaluation": {
        "scope": "generic",
        "focus": "all",
        "label": "Generic evaluation",
    },
    "broad_cold_proof": {
        "scope": "broad",
        "focus": "all",
        "label": "Broad cold thesis proof",
    },
    "focused_refc_proof": {
        "scope": "focused",
        "focus": "refc",
        "label": "Focused REFC proof",
    },
    "focused_voi_proof": {
        "scope": "focused",
        "focus": "voi",
        "label": "Focused VOI proof",
    },
    "dccs_diagnostic_probe": {
        "scope": "probe",
        "focus": "dccs",
        "label": "DCCS diagnostic probe",
    },
    "hot_rerun_cold_source": {
        "scope": "hot_rerun_source",
        "focus": "runtime_reuse",
        "label": "Hot rerun cold source",
    },
    "hot_rerun": {
        "scope": "hot_rerun",
        "focus": "runtime_reuse",
        "label": "Hot rerun proof",
    },
}
FRONTIER_ARTIFACT = "strict_frontier.jsonl"
ARTEFACT_JSON_NAMES = (
    "metadata.json",
    "dccs_summary.json",
    "certificate_summary.json",
    "initial_certificate_summary.json",
    "route_fragility_map.json",
    "initial_route_fragility_map.json",
    "competitor_fragility_breakdown.json",
    "initial_competitor_fragility_breakdown.json",
    "value_of_refresh.json",
    "initial_value_of_refresh.json",
    "sampled_world_manifest.json",
    "initial_sampled_world_manifest.json",
    "voi_action_trace.json",
    "voi_stop_certificate.json",
    "final_route_trace.json",
)
ARTEFACT_JSONL_NAMES = ("dccs_candidates.jsonl", FRONTIER_ARTIFACT, "voi_controller_state.jsonl")
ESSENTIAL_ROUTE_RESPONSE_FIELDS = ("run_id", "manifest_endpoint", "artifacts_endpoint")
REQUIRED_ARTIFACTS_BY_PIPELINE: dict[str, tuple[str, ...]] = {
    "legacy": ("metadata.json", FRONTIER_ARTIFACT, "final_route_trace.json"),
    "dccs": ("metadata.json", FRONTIER_ARTIFACT, "final_route_trace.json", "dccs_summary.json", "dccs_candidates.jsonl"),
    "dccs_refc": (
        "metadata.json",
        FRONTIER_ARTIFACT,
        "final_route_trace.json",
        "dccs_summary.json",
        "dccs_candidates.jsonl",
        "certificate_summary.json",
        "route_fragility_map.json",
        "competitor_fragility_breakdown.json",
        "value_of_refresh.json",
        "sampled_world_manifest.json",
    ),
    "voi": (
        "metadata.json",
        FRONTIER_ARTIFACT,
        "final_route_trace.json",
        "dccs_summary.json",
        "dccs_candidates.jsonl",
        "certificate_summary.json",
        "route_fragility_map.json",
        "competitor_fragility_breakdown.json",
        "value_of_refresh.json",
        "sampled_world_manifest.json",
        "voi_action_trace.json",
        "voi_stop_certificate.json",
    ),
}
SUMMARY_FIELDS = [
    "variant_id",
    "pipeline_mode",
    "row_count",
    "failure_count",
    "success_count",
    "success_rate",
    "artifact_complete_rate",
    "route_evidence_ok_rate",
    "certified_rate",
    "certified_denominator",
    "dominance_win_rate_osrm",
    "dominance_denominator_osrm",
    "dominance_win_rate_ors",
    "dominance_denominator_ors",
    "dominance_win_rate_v0",
    "dominance_denominator_v0",
    "runtime_win_rate_v0",
    "runtime_denominator_v0",
    "algorithm_runtime_win_rate_v0",
    "algorithm_runtime_denominator_v0",
    "dominance_win_rate_best_baseline",
    "dominance_denominator_best_baseline",
    "weighted_win_rate_osrm",
    "weighted_denominator_osrm",
    "weighted_win_rate_ors",
    "weighted_denominator_ors",
    "weighted_win_rate_v0",
    "weighted_denominator_v0",
    "weighted_win_rate_best_baseline",
    "weighted_denominator_best_baseline",
    "balanced_win_rate_osrm",
    "balanced_denominator_osrm",
    "balanced_win_rate_ors",
    "balanced_denominator_ors",
    "balanced_win_rate_v0",
    "balanced_denominator_v0",
    "balanced_win_rate_best_baseline",
    "balanced_denominator_best_baseline",
    "time_preserving_win_rate",
    "time_preserving_denominator",
    "time_preserving_win_rate_osrm",
    "time_preserving_denominator_osrm",
    "time_preserving_win_rate_ors",
    "time_preserving_denominator_ors",
    "time_preserving_win_rate_best_baseline",
    "time_preserving_denominator_best_baseline",
    "time_preserving_dominance_rate",
    "time_preserving_dominance_denominator",
    "time_preserving_dominance_rate_osrm",
    "time_preserving_dominance_denominator_osrm",
    "time_preserving_dominance_rate_ors",
    "time_preserving_dominance_denominator_ors",
    "time_preserving_dominance_rate_best_baseline",
    "time_preserving_dominance_denominator_best_baseline",
    "robust_win_rate_osrm",
    "robust_denominator_osrm",
    "robust_win_rate_ors",
    "robust_denominator_ors",
    "mean_certificate",
    "mean_certificate_denominator",
    "mean_frontier_hypervolume",
    "mean_frontier_hypervolume_denominator",
    "mean_frontier_coverage_osrm",
    "mean_frontier_coverage_ors",
    "mean_frontier_count",
    "nontrivial_frontier_rate",
    "mean_frontier_diversity_index",
    "mean_frontier_entropy",
    "mean_od_ambiguity_index",
    "mean_od_ambiguity_confidence",
    "mean_od_ambiguity_source_count",
    "mean_od_ambiguity_source_mix_count",
    "mean_od_ambiguity_source_support_strength",
    "mean_od_ambiguity_source_entropy",
    "mean_od_ambiguity_support_ratio",
    "mean_od_ambiguity_prior_strength",
    "mean_od_ambiguity_family_density",
    "mean_od_ambiguity_margin_pressure",
    "mean_od_ambiguity_spread_pressure",
    "mean_od_engine_disagreement_prior",
    "mean_od_hard_case_prior",
    "mean_ambiguity_budget_prior",
    "mean_ambiguity_budget_prior_gap",
    "budget_prior_exceeds_raw_rate",
    "upstream_nonzero_od_ambiguity_rate",
    "upstream_high_hard_case_prior_rate",
    "mean_observed_ambiguity_index",
    "mean_ambiguity_alignment",
    "ambiguity_prior_realized_correlation",
    "mean_nominal_winner_margin",
    "mean_near_tie_mass",
    "mean_certificate_margin",
    "mean_certificate_runner_up_gap",
    "mean_fragility_entropy",
    "mean_competitor_turnover_rate",
    "mean_hard_case_rate",
    "mean_hard_case_certificate",
    "mean_hard_case_runtime_ms",
    "mean_hard_case_frontier_diversity_index",
    "mean_hard_case_action_efficiency",
    "mean_hard_case_search_budget_utilization",
    "mean_hard_case_evidence_budget_utilization",
    "mean_hard_case_controller_engagement_rate",
    "mean_certificate_gap_ambiguity_vs_representative",
    "mean_runtime_gap_ambiguity_vs_representative_ms",
    "mean_dccs_dc_yield_gap_ambiguity_vs_representative",
    "mean_time_to_best_gap_ambiguity_vs_representative",
    "mean_search_budget_utilization_gap_ambiguity_vs_representative",
    "mean_evidence_budget_utilization_gap_ambiguity_vs_representative",
    "certificate_selectivity_rate",
    "certificate_selectivity_denominator",
    "productive_voi_action_rate",
    "productive_voi_action_denominator",
    "zero_lift_controller_action_rate",
    "mean_initial_refc_top_vor",
    "mean_initial_refc_top_vor_denominator",
    "mean_final_refc_top_vor",
    "mean_final_refc_top_vor_denominator",
    "mean_initial_winner_fragility_mass",
    "mean_initial_winner_fragility_mass_denominator",
    "mean_final_winner_fragility_mass",
    "mean_final_winner_fragility_mass_denominator",
    "initial_winner_fragility_nonzero_rate",
    "initial_winner_fragility_nonzero_denominator",
    "winner_fragility_nonzero_rate",
    "winner_fragility_nonzero_denominator",
    "initial_refc_top_vor_positive_rate",
    "initial_refc_top_vor_positive_denominator",
    "refc_top_vor_positive_rate",
    "refc_top_vor_positive_denominator",
    "refresh_signal_persistence_rate",
    "refresh_signal_persistence_denominator",
    "refresh_first_productive_rate",
    "refresh_first_productive_denominator",
    "refresh_resolution_honesty_rate",
    "refresh_resolution_honesty_denominator",
    "mean_controller_cost_per_certificate_point",
    "mean_top_refresh_gain",
    "mean_top_refresh_gain_denominator",
    "mean_top_fragility_mass",
    "mean_top_fragility_mass_denominator",
    "mean_competitor_pressure",
    "mean_competitor_pressure_denominator",
    "mean_route_improvement_per_second",
    "ambiguity_prior_top_k_precision",
    "ambiguity_prior_top_k_precision_k",
    "ambiguity_prior_top_k_precision_denominator",
    "ambiguity_prior_overtrigger_rate",
    "mean_dccs_frontier_recall_at_budget",
    "mean_dccs_corridor_family_recall",
    "mean_voi_realized_certificate_lift",
    "mean_voi_realized_runner_up_gap_lift",
    "mean_voi_realized_margin_lift",
    "mean_time_to_best_iteration",
    "mean_action_efficiency",
    "refine_cost_prediction_error_deprecated",
    "refine_cost_mape",
    "refine_cost_sample_count",
    "refine_cost_positive_sample_count",
    "refine_cost_zero_observed_count",
    "refine_cost_mae_ms",
    "refine_cost_rank_correlation",
    "mean_runtime_p50_ms",
    "mean_runtime_p90_ms",
    "mean_runtime_p95_ms",
    "mean_algorithm_runtime_p50_ms",
    "mean_algorithm_runtime_p90_ms",
    "mean_algorithm_runtime_p95_ms",
    "mean_baseline_acquisition_runtime_p90_ms",
    "mean_route_request_ms",
    "mean_baseline_osrm_ms",
    "mean_baseline_ors_ms",
    "mean_stage_k_raw_ms",
    "mean_stage_k_raw_graph_search_initial_ms",
    "mean_stage_k_raw_graph_search_retry_ms",
    "mean_stage_k_raw_graph_search_rescue_ms",
    "mean_stage_k_raw_graph_search_supplemental_ms",
    "mean_stage_k_raw_osrm_fallback_ms",
    "mean_stage_dccs_ms",
    "mean_stage_refinement_ms",
    "mean_stage_pareto_ms",
    "mean_stage_refc_ms",
    "mean_stage_voi_ms",
    "mean_runtime_ratio_vs_osrm",
    "mean_runtime_ratio_vs_ors",
    "mean_algorithm_runtime_ratio_vs_osrm",
    "mean_algorithm_runtime_ratio_vs_ors",
    "mean_runtime_gap_vs_osrm_ms",
    "mean_runtime_gap_vs_ors_ms",
    "mean_algorithm_runtime_gap_vs_osrm_ms",
    "mean_algorithm_runtime_gap_vs_ors_ms",
    "mean_row_local_warmup_ms",
    "warmup_amortized_ms",
    "mean_warmup_overhead_share",
    "mean_global_startup_overhead_ms",
    "mean_global_startup_share_of_algorithm",
    "mean_runtime_per_refined_candidate_ms",
    "mean_runtime_per_frontier_member_ms",
    "mean_memory_per_refined_candidate_mb",
    "mean_quality_per_second",
    "mean_frontier_gain_per_ms",
    "mean_certificate_gain_per_world",
    "mean_cache_reuse_ratio",
    "route_state_cache_hit_rate",
    "route_state_cache_hits",
    "route_state_cache_misses",
    "baseline_identity_verified_rate",
    "mean_ambiguity_prior_gap",
    "realized_diversity_collapse_rate",
    "supplemental_challenger_activation_rate",
    "mean_supplemental_challenger_selected_count",
    "selected_from_supplemental_rescue_rate",
    "selected_from_comparator_engine_rate",
    "preemptive_comparator_activation_rate",
    "mean_preemptive_comparator_candidate_count",
    "selected_from_preemptive_comparator_seed_rate",
    "mean_controller_value_per_second",
    "mean_refc_shortcut_rate",
    "mean_refc_cache_hits",
    "mean_refc_unique_world_count",
    "mean_refc_world_reuse_rate",
    "mean_refc_hard_stress_pack_count",
    "mean_refc_stress_world_fraction",
    "mean_requested_cert_world_count",
    "mean_effective_cert_world_count",
    "mean_world_count_efficiency",
    "mean_refc_ms_per_effective_world",
    "mean_stage_supplemental_rescue_ms",
    "mean_stage_preemptive_comparator_seed_ms",
    "mean_backend_ready_wait_ms",
    "mean_route_graph_warmup_elapsed_ms",
    "mean_preflight_ms",
    "mean_preflight_and_warmup_ms",
    "mean_process_rss_mb",
    "mean_process_vms_mb",
    "mean_process_rss_p90_mb",
    "mean_process_vms_p90_mb",
    "mean_route_cache_hit_rate",
    "mean_k_raw_cache_hit_rate",
    "mean_graph_low_ambiguity_fast_path_rate",
    "mean_graph_supported_ambiguity_fast_fallback_rate",
    "controller_activation_on_high_ambiguity_rate",
    "mean_search_budget_utilization_p90",
    "mean_evidence_budget_utilization_p90",
    "mean_voi_action_density",
    "mean_initial_certificate",
    "initial_certificate_stop_rate",
    "unnecessary_voi_refine_rate",
    "mean_stage_option_build_ms",
    "mean_option_build_reuse_rate",
    "mean_option_build_cache_hits",
    "mean_option_build_rebuild_count",
    "mean_option_build_cache_hit_rate",
    "mean_voi_dccs_cache_hit_rate",
    "voi_dccs_cache_hit_rate",
    "voi_dccs_cache_hits",
    "voi_dccs_cache_misses",
    "option_build_cache_savings_ms_per_row",
    "comparator_independence_rate",
    "strict_failure_elimination_rate",
    "mean_search_completeness_score",
    "mean_search_completeness_gap",
    "mean_prior_support_strength",
    "mean_pending_challenger_mass",
    "mean_best_pending_flip_probability",
    "mean_corridor_family_recall",
    "mean_frontier_recall_at_budget",
    "credible_search_uncertainty_rate",
    "credible_evidence_uncertainty_rate",
    "supported_hard_case_rate",
    "evidence_first_engagement_rate",
    "evidence_only_engagement_rate",
    "mean_ambiguity_absolute_error",
    "mean_supported_ambiguity_alignment",
    "mean_time_to_certification_ms",
    "mean_controller_shortcut_rate",
    "mean_voi_stop_after_certification_rate",
    "mean_controller_stress_rate",
    "controller_stress_row_count",
    "scenario_profile_unavailable_rate",
    "strict_live_readiness_pass_rate",
    "evaluation_rerun_success_rate",
    "controller_refresh_fallback_activation_rate",
    "controller_empirical_vs_raw_refresh_disagreement_rate",
    "broad_hard_case_certificate_selectivity_rate",
    "broad_hard_case_evidence_first_engagement_rate",
    "broad_hard_case_productive_voi_action_rate",
    "broad_hard_case_refc_signal_presence_rate",
    "mean_weighted_margin_gain_vs_v0",
    "mean_balanced_gain_delta_vs_v0_score",
    "mean_duration_gain_vs_v0_s",
    "mean_monetary_gain_vs_v0",
    "mean_emissions_gain_vs_v0_kg",
    "mean_frontier_hypervolume_gain_vs_v0",
    "mean_certificate_lift_vs_v0",
    "mean_hard_case_certificate_lift_vs_v0",
    "mean_weighted_margin_vs_osrm",
    "mean_weighted_margin_vs_ors",
    "mean_weighted_margin_vs_v0",
    "mean_weighted_margin_vs_best_baseline",
    "mean_dccs_dc_yield",
    "mean_dccs_dc_yield_denominator",
    "mean_iteration_count",
    "mean_voi_action_count",
    "mean_voi_refine_action_count",
    "mean_voi_refresh_action_count",
    "mean_voi_resample_action_count",
    "voi_controller_engagement_rate",
    "selector_certificate_disagreement_rate",
    "mean_search_budget_used",
    "mean_evidence_budget_used",
    "mean_search_budget_utilization",
    "mean_evidence_budget_utilization",
    "mean_algorithm_runtime_ms",
    "mean_algorithm_runtime_speedup_vs_v0",
    "mean_runtime_speedup_vs_v0",
    "mean_objective_gain_vs_v0_denominator",
    "mean_certificate_lift_vs_v0_denominator",
    "certificate_availability_gain_vs_v0_rate",
    "mean_baseline_acquisition_runtime_ms",
    "mean_baseline_runtime_share",
    "mean_runtime_ms",
    "mean_runtime_ms_denominator",
    "corpus_kind_counts_json",
    "corpus_group_counts_json",
    "profile_id_counts_json",
]
COHORT_SUMMARY_FIELDS = ["cohort_label", "cohort_total_row_count", "cohort_share_of_variant", *SUMMARY_FIELDS]
RESULT_FIELDS = [
    "od_id", "variant_id", "pipeline_mode", "pipeline_version", "seed", "trip_length_bin",
    "origin_lat", "origin_lon", "destination_lat", "destination_lon", "straight_line_km",
    "profile_id", "corpus_group", "corpus_kind", "cohort_label", "hard_case", "od_ambiguity_index", "observed_ambiguity_index", "od_candidate_path_count", "od_corridor_family_count",
    "od_ambiguity_confidence", "od_ambiguity_source_count", "od_ambiguity_source_mix", "od_ambiguity_source_mix_count",
    "od_ambiguity_source_entropy", "od_ambiguity_support_ratio", "od_ambiguity_prior_strength", "od_ambiguity_family_density",
    "od_ambiguity_margin_pressure", "od_ambiguity_spread_pressure", "od_ambiguity_toll_instability",
    "od_objective_spread", "od_nominal_margin_proxy", "od_toll_disagreement_rate", "od_engine_disagreement_prior", "od_hard_case_prior",
    "ambiguity_prior_sample_count", "ambiguity_prior_support_count",
    "row_override_count", "ambiguity_budget_band", "ambiguity_budget_prior", "ambiguity_budget_prior_gap", "budget_prior_exceeds_raw", "effective_request_config_json",
    "route_id", "route_source", "candidate_count_display", "candidate_count_raw", "refined_count",
    "frontier_count", "iteration_count", "search_budget", "evidence_budget", "search_budget_used",
    "evidence_budget_used", "certificate_threshold", "certificate", "certified", "selected_distance_km",
    "refinement_selection_policy", "refinement_selected_candidate_count", "refinement_selected_candidate_ids_json",
    "selected_duration_s", "selected_monetary_cost", "selected_emissions_kg", "selected_p95_duration_s",
    "selected_cvar95_duration_s", "osrm_method", "osrm_distance_km", "osrm_duration_s",
    "osrm_monetary_cost", "osrm_emissions_kg", "ors_method", "ors_provider_mode", "ors_distance_km",
    "ors_duration_s", "ors_monetary_cost", "ors_emissions_kg", "delta_vs_osrm_distance_km",
    "delta_vs_osrm_duration_s", "delta_vs_osrm_monetary_cost", "delta_vs_osrm_emissions_kg",
    "delta_vs_ors_distance_km", "delta_vs_ors_duration_s", "delta_vs_ors_monetary_cost",
    "delta_vs_ors_emissions_kg", "dominates_osrm", "dominates_ors", "dominates_v0",
    "dominates_best_baseline", "weighted_win_osrm", "weighted_win_ors", "weighted_win_v0",
    "weighted_win_best_baseline", "weighted_margin_vs_osrm", "weighted_margin_vs_ors",
    "weighted_margin_vs_v0", "weighted_margin_vs_best_baseline",
    "balanced_win_osrm", "balanced_win_ors", "balanced_win_v0", "balanced_win_best_baseline",
    "time_preserving_win_osrm", "time_preserving_win_ors", "time_preserving_win_best_baseline",
    "time_preserving_dominance_osrm", "time_preserving_dominance_ors", "time_preserving_dominance_best_baseline",
    "best_baseline_provider", "balanced_gain_vs_osrm_score", "balanced_gain_vs_ors_score",
    "robust_win_osrm", "robust_win_ors",
    "frontier_hypervolume", "frontier_coverage_osrm", "frontier_coverage_ors", "frontier_epsilon_osrm",
    "frontier_epsilon_ors", "frontier_spread", "frontier_crowding_mean", "frontier_diversity_index", "frontier_entropy",
    "time_to_best_iteration", "action_efficiency", "refine_cost_prediction_error_deprecated", "refine_cost_mape", "refine_cost_sample_count", "refine_cost_positive_sample_count", "refine_cost_zero_observed_count", "refine_cost_mae_ms", "refine_cost_rank_correlation", "nominal_winner_margin",
    "near_tie_mass", "certificate_margin", "certificate_runner_up_gap", "fragility_entropy",
    "competitor_turnover_rate", "dccs_dc_yield",
    "dccs_challenger_hit_rate", "dccs_frontier_gain_per_refinement", "dccs_decision_flips",
    "dccs_score_label_correlation", "dccs_frontier_recall_at_budget", "dccs_corridor_family_recall",
    "refc_world_count", "refc_active_family_count",
    "refc_unique_world_count", "refc_world_reuse_rate", "refc_hard_stress_pack_count", "refc_stress_world_fraction",
    "requested_cert_world_count", "effective_cert_world_count", "world_count_policy", "world_count_efficiency",
    "initial_refc_top_fragility_family", "initial_refc_top_refresh_family", "initial_refc_top_vor", "initial_refc_vor_gap",
    "final_refc_top_fragility_family", "final_refc_top_refresh_family", "final_refc_top_vor", "final_refc_vor_gap",
    "initial_winner_fragility_mass", "final_winner_fragility_mass",
    "initial_winner_fragility_nonzero", "winner_fragility_nonzero",
    "initial_refc_top_vor_positive", "refc_top_vor_positive", "refresh_signal_persistent",
    "refresh_first_productive", "refresh_resolution_honest", "refresh_resolution_reason",
    "refc_top_fragility_family", "refc_top_refresh_family", "refc_top_vor", "refc_vor_gap", "certificate_winner_route_id",
    "top_refresh_gain", "top_fragility_mass", "competitor_pressure",
    "selector_certificate_disagreement",
    "refc_top_competitor_route_id", "voi_stop_reason", "voi_best_rejected_action", "voi_best_rejected_q",
    "voi_action_count", "voi_refine_action_count", "voi_refresh_action_count", "voi_resample_action_count",
    "voi_productive_action_count", "voi_nonproductive_action_count", "productive_voi_action_rate",
    "voi_controller_engaged", "controller_stress_row", "nontrivial_frontier", "search_budget_utilization", "evidence_budget_utilization",
    "initial_certificate", "initial_certificate_stop", "unnecessary_voi_refine",
    "time_to_certification_ms", "controller_shortcut", "voi_stop_after_certification",
    "certificate_selective",
    "preemptive_comparator_seeded", "preemptive_comparator_candidate_count", "preemptive_comparator_source_count", "selected_from_preemptive_comparator_seed",
    "comparator_independent", "strict_failure_eliminated",
    "voi_realized_certificate_lift", "voi_realized_frontier_gain", "voi_realized_runtime_delta_ms",
    "weighted_margin_gain_vs_v0", "balanced_gain_delta_vs_v0_score", "duration_gain_vs_v0_s",
    "monetary_gain_vs_v0", "emissions_gain_vs_v0_kg", "frontier_hypervolume_gain_vs_v0", "certificate_lift_vs_v0",
    "certificate_availability_gain_vs_v0",
    "ors_baseline_policy", "ors_asset_manifest_hash", "ors_asset_recorded_at", "ors_asset_freshness_status",
    "ors_graph_identity_status", "ors_engine_image", "ors_graph_build_date", "ors_graph_osm_date",
    "ors_graph_file_count", "ors_graph_total_bytes", "ors_graph_listing_digest",
    "runtime_ms", "algorithm_runtime_ms", "baseline_acquisition_runtime_ms", "baseline_runtime_share",
    "runtime_ratio_vs_osrm", "runtime_ratio_vs_ors", "algorithm_runtime_ratio_vs_osrm", "algorithm_runtime_ratio_vs_ors",
    "runtime_gap_vs_osrm_ms", "runtime_gap_vs_ors_ms", "algorithm_runtime_gap_vs_osrm_ms", "algorithm_runtime_gap_vs_ors_ms",
    "row_local_warmup_ms", "warmup_amortized_ms", "warmup_overhead_share", "global_startup_overhead_ms", "global_startup_share_of_algorithm",
    "runtime_per_refined_candidate_ms", "runtime_per_frontier_member_ms",
    "memory_per_refined_candidate_mb", "quality_per_second", "route_improvement_per_second", "frontier_gain_per_ms", "certificate_gain_per_world", "controller_cost_per_certificate_point", "cache_reuse_ratio", "baseline_identity_verified", "ambiguity_alignment", "ambiguity_prior_gap", "ambiguity_prior_overtrigger", "controller_activation_on_high_ambiguity",
    "preflight_and_warmup_ms", "stage_option_build_ms", "option_build_reuse_rate",
    "option_build_cache_hits", "option_build_rebuild_count", "option_build_cache_hit_rate", "option_build_cache_savings_ms_per_row",
    "search_completeness_score", "search_completeness_gap", "prior_support_strength",
    "pending_challenger_mass", "best_pending_flip_probability", "corridor_family_recall",
    "frontier_recall_at_budget", "credible_search_uncertainty", "credible_evidence_uncertainty",
    "supported_hard_case", "evidence_first_engagement", "evidence_only_engagement", "first_controller_action_kind",
    "controller_refresh_ranking_basis", "controller_top_refresh_family", "controller_top_refresh_gain",
    "controller_refresh_fallback_activated", "controller_empirical_vs_raw_refresh_disagreement",
    "voi_dccs_cache_hits", "voi_dccs_cache_misses", "voi_dccs_cache_hit_rate",
    "ambiguity_absolute_error", "supported_ambiguity_alignment",
    "realized_diversity_collapse", "realized_diversity_collapse_reason", "realized_raw_corridor_family_count",
    "realized_refined_corridor_family_count", "supplemental_challenger_activated", "supplemental_challenger_source_count",
    "supplemental_challenger_candidate_count", "supplemental_challenger_selected_count",
    "supplemental_challenger_budget_used", "selected_candidate_source_label", "selected_candidate_source_engine",
    "selected_candidate_source_stage", "selected_final_route_source_label", "selected_final_route_source_engine",
    "selected_final_route_source_stage", "selected_from_supplemental_rescue", "selected_from_comparator_engine",
    "controller_value_per_second",
    "backend_ready_wait_ms", "backend_ready_probe_ms", "route_graph_warmup_elapsed_ms", "preflight_ms",
    "process_rss_mb", "process_vms_mb", "route_cache_hits", "route_cache_misses", "route_cache_hit_rate",
    "graph_k_raw_cache_hit", "graph_low_ambiguity_fast_path", "graph_supported_ambiguity_fast_fallback",
    "route_state_cache_hits", "route_state_cache_misses", "route_state_cache_hit_rate",
    "refc_shortcut_used", "refc_cache_hits",
    "route_request_ms", "baseline_osrm_ms", "baseline_ors_ms", "stage_dccs_ms",
    "stage_k_raw_ms", "stage_k_raw_graph_search_initial_ms", "stage_k_raw_graph_search_retry_ms",
    "stage_k_raw_graph_search_rescue_ms", "stage_k_raw_graph_search_supplemental_ms",
    "stage_k_raw_osrm_fallback_ms",
    "stage_refc_ms", "stage_voi_ms", "stage_pareto_ms", "stage_refinement_ms", "stage_supplemental_rescue_ms", "stage_preemptive_comparator_seed_ms", "ors_snapshot_mode",
    "ors_snapshot_used", "ors_snapshot_recorded_at", "ors_snapshot_request_hash", "ors_snapshot_response_hash",
    "ors_snapshot_provider_mode", "artifact_complete", "artifact_status", "artifact_missing", "evidence_policy",
    "route_evidence_ok", "route_evidence_status", "route_evidence_issues",
    "failure_reason", "artifact_run_id", "manifest_endpoint", "artifacts_endpoint",
]


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    pipeline_mode: str
    refinement_policy: str | None = None


@dataclass(frozen=True)
class BaselineResult:
    route: dict[str, Any]
    metrics: dict[str, float]
    method: str
    compute_ms: float
    snapshot_used: bool = False
    provider_mode: str = "live"
    baseline_policy: str | None = None
    asset_manifest_hash: str | None = None
    asset_recorded_at: str | None = None
    asset_freshness_status: str | None = None
    engine_manifest: dict[str, Any] | None = None
    snapshot_recorded_at: str | None = None
    snapshot_request_hash: str | None = None
    snapshot_response_hash: str | None = None


ROW_OVERRIDE_FIELDS = (
    "profile_id",
    "corpus_group",
    "weight_time",
    "weight_money",
    "weight_co2",
    "scenario_mode",
    "weather_enabled",
    "weather_profile",
    "weather_intensity",
    "departure_time_utc",
    "stochastic_enabled",
    "stochastic_samples",
    "search_budget",
    "evidence_budget",
    "world_count",
    "certificate_threshold",
    "tau_stop",
    "max_alternatives",
    "optimization_mode",
)


def _empty_baseline_result(*, method: str, provider_mode: str = "strict_rejected") -> BaselineResult:
    metrics = {
        "distance_km": float("nan"),
        "duration_s": float("nan"),
        "monetary_cost": float("nan"),
        "emissions_kg": float("nan"),
    }
    return BaselineResult(route={}, metrics=metrics, method=method, compute_ms=0.0, provider_mode=provider_mode)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canon(value).encode("utf-8")).hexdigest()


def _run_id(
    *,
    seed: int,
    corpus_hash: str,
    model_version: str,
    world_count: int,
    snapshot_mode: str,
    baseline_policy: str,
) -> str:
    payload = {
        "seed": seed,
        "corpus_hash": corpus_hash,
        "model_version": model_version,
        "world_count": world_count,
        "snapshot_mode": snapshot_mode,
        "baseline_policy": baseline_policy,
    }
    return str(uuid.uuid5(uuid.NAMESPACE_URL, _canon(payload)))


def _distance_km(row: dict[str, Any]) -> float:
    if row.get("straight_line_km") not in (None, ""):
        return float(row["straight_line_km"])
    o_lat, o_lon = float(row["origin_lat"]), float(row["origin_lon"])
    d_lat, d_lon = float(row["destination_lat"]), float(row["destination_lon"])
    radius_km = 6371.0
    phi1 = math.radians(o_lat)
    phi2 = math.radians(d_lat)
    dphi = math.radians(d_lat - o_lat)
    dlambda = math.radians(d_lon - o_lon)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _bin(distance_km: float) -> str:
    if distance_km < 30.0:
        return "0-30 km"
    if distance_km < 100.0:
        return "30-100 km"
    if distance_km < 250.0:
        return "100-250 km"
    return "250+ km"


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source_mix_count(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        return len([key for key in value.keys() if str(key).strip()])
    if isinstance(value, (list, tuple, set)):
        return len([item for item in value if str(item).strip()])
    text = _text_or_none(value)
    if text is None:
        return None
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return len([key for key in parsed.keys() if str(key).strip()])
        if isinstance(parsed, list):
            return len([item for item in parsed if str(item).strip()])
    return len([token for token in text.split("|") if token.strip()]) or 1


def _source_support_map(value: Any) -> dict[str, float]:
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        support: dict[str, float] = {}
        for key, raw in value.items():
            token = str(key).strip()
            if not token:
                continue
            support[token] = max(0.0, min(1.0, as_float(raw, 0.0)))
        return support
    text = _text_or_none(value)
    if text is None:
        return {}
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return _source_support_map(parsed)
        if isinstance(parsed, list):
            return {str(item).strip(): 0.5 for item in parsed if str(item).strip()}
    tokens = [token for token in _split_prior_sources(text) if token]
    return {token: 0.5 for token in tokens}


def _source_support_strength(value: Any) -> float | None:
    support = _source_support_map(value)
    if not support:
        return None
    values = [support[token] for token in sorted(support) if math.isfinite(support[token])]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _float_or_default(value: Any, default: float) -> float:
    text = _text_or_none(value)
    if text is None:
        return float(default)
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not math.isfinite(parsed) else float(parsed)


def _int_or_default(value: Any, default: int) -> int:
    text = _text_or_none(value)
    if text is None:
        return int(default)
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return int(default)


def _bool_or_default(value: Any, default: bool) -> bool:
    text = _text_or_none(value)
    if text is None:
        return bool(default)
    lowered = text.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _time_preserving_outcome(*, duration_delta_s: Any, quality_win: Any) -> bool | None:
    duration_delta = as_float(duration_delta_s, float("nan"))
    if not math.isfinite(duration_delta):
        return None
    return bool(quality_win) and duration_delta <= 0.0


def _datetime_or_none(value: Any) -> datetime | None:
    text = _text_or_none(value)
    if text is None:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _route_graph_startup_to_ready_ms(readiness_summary: Mapping[str, Any] | None) -> float | None:
    if not isinstance(readiness_summary, Mapping):
        return None
    route_graph = readiness_summary.get("route_graph")
    if not isinstance(route_graph, Mapping):
        return None
    state = str(route_graph.get("state") or "").strip().lower()
    started_at = _datetime_or_none(route_graph.get("started_at_utc"))
    ready_at = _datetime_or_none(route_graph.get("ready_at_utc"))
    if started_at is not None and ready_at is not None and ready_at >= started_at:
        return round((ready_at - started_at).total_seconds() * 1000.0, 6)
    if state == "ready":
        return None
    elapsed_ms = as_float(route_graph.get("elapsed_ms"), float("nan"))
    if math.isfinite(elapsed_ms):
        return elapsed_ms
    return None


def _startup_components_ms(readiness_summary: Mapping[str, Any] | None) -> list[float]:
    components = [
        as_float(readiness_summary.get("wait_elapsed_ms"), float("nan")) if isinstance(readiness_summary, Mapping) else float("nan"),
        as_float(readiness_summary.get("compute_ms"), float("nan")) if isinstance(readiness_summary, Mapping) else float("nan"),
        as_float(_route_graph_startup_to_ready_ms(readiness_summary), float("nan")),
    ]
    return [component for component in components if math.isfinite(component)]


def _valid_refine_cost_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    valid_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        predicted_cost = as_float(row.get("predicted_refine_cost"), float("nan"))
        observed_cost = as_float(row.get("observed_refine_cost"), float("nan"))
        if not (math.isfinite(predicted_cost) and math.isfinite(observed_cost)):
            continue
        if predicted_cost < 0.0 or observed_cost < 0.0:
            continue
        valid_rows.append(row)
    return valid_rows


def _action_candidate_ids(action_entry: Mapping[str, Any], *, key: str) -> list[str]:
    containers = [action_entry]
    chosen_action = action_entry.get("chosen_action")
    if isinstance(chosen_action, Mapping):
        containers.append(chosen_action)
    for container in containers:
        raw_value = container.get(key)
        if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, bytearray)):
            candidate_ids = [str(item).strip() for item in raw_value if str(item).strip()]
            if candidate_ids:
                return candidate_ids
        if isinstance(raw_value, str):
            candidate_id = raw_value.strip()
            if candidate_id:
                return [candidate_id]
    return []


def _row_override_source(od: dict[str, Any]) -> dict[str, Any]:
    row_overrides = od.get("row_overrides")
    return row_overrides if isinstance(row_overrides, dict) else od


def _parse_row_overrides(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        parsed = None
        for loader in (json.loads, ast.literal_eval):
            try:
                candidate = loader(text)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(candidate, Mapping):
                parsed = candidate
                break
        if parsed is None:
            return {}
    else:
        return {}
    return {
        str(key): parsed[key]
        for key in ROW_OVERRIDE_FIELDS
        if key in parsed and _text_or_none(parsed[key]) is not None
    }


def _support_gated_ambiguity_budget_prior(od: Mapping[str, Any], raw_prior: float) -> float:
    return _shared_support_gated_ambiguity_budget_prior(
        ambiguity_value=raw_prior,
        support_ratio=as_float(od.get("od_ambiguity_support_ratio"), 0.0),
        source_entropy=as_float(od.get("od_ambiguity_source_entropy"), 0.0),
        source_count=int(as_float(od.get("od_ambiguity_source_count"), 0.0)),
        source_mix_count=int(as_float(od.get("od_ambiguity_source_mix_count"), 0.0)),
        prior_strength=as_float(od.get("od_ambiguity_prior_strength"), 0.0),
        family_density=as_float(od.get("od_ambiguity_family_density"), 0.0),
        margin_pressure=as_float(od.get("od_ambiguity_margin_pressure"), 0.0),
        spread_pressure=as_float(od.get("od_ambiguity_spread_pressure"), 0.0),
        toll_instability=as_float(od.get("od_ambiguity_toll_instability"), 0.0),
        engine_prior=as_float(
            od.get("candidate_probe_engine_disagreement_prior"),
            as_float(od.get("od_engine_disagreement_prior"), 0.0),
        ),
        hard_case_prior=as_float(
            od.get("hard_case_prior"),
            as_float(od.get("od_hard_case_prior"), 0.0),
        ),
    )


def _od_ambiguity_prior(od: Mapping[str, Any]) -> float | None:
    raw_prior = _shared_raw_ambiguity_prior_value(od)
    if raw_prior is None:
        return None
    return _support_gated_ambiguity_budget_prior(od, raw_prior)


def _ambiguity_budget_schedule(
    args: argparse.Namespace,
    od: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    base_search = _int_or_default(source.get("search_budget"), int(args.search_budget))
    base_evidence = _int_or_default(source.get("evidence_budget"), int(args.evidence_budget))
    base_world = _int_or_default(source.get("world_count"), int(args.world_count))
    max_alternatives = max(1, _int_or_default(source.get("max_alternatives"), int(args.max_alternatives)))
    prior = _od_ambiguity_prior(od)
    path_count = _int_or_default(od.get("candidate_probe_path_count"), 0)
    corridor_count = _int_or_default(od.get("candidate_probe_corridor_family_count"), 0)
    spread = _float_or_default(od.get("candidate_probe_objective_spread"), 0.0)
    engine_disagreement_prior = _float_or_default(od.get("candidate_probe_engine_disagreement_prior"), 0.0)
    hard_case_prior = _float_or_default(od.get("hard_case_prior"), 0.0)
    if prior is None:
        band = "unspecified"
        search_budget = base_search
        evidence_budget = base_evidence
        world_count = base_world
    elif (
        prior >= 0.65
        or hard_case_prior >= 0.65
        or engine_disagreement_prior >= 0.55
        or path_count >= 4
        or corridor_count >= 3
        or spread >= 0.25
    ):
        band = "high"
        search_budget = min(max_alternatives, max(base_search, min(max_alternatives, base_search + 2)))
        evidence_budget = max(base_evidence, 2)
        world_count = max(base_world, 96)
    elif (
        prior <= 0.25
        and hard_case_prior <= 0.22
        and engine_disagreement_prior <= 0.2
        and path_count <= 2
        and corridor_count <= 1
        and spread <= 0.12
    ):
        band = "low"
        search_budget = max(1, min(base_search, 2))
        evidence_budget = max(1, min(base_evidence, 1))
        world_count = max(16, min(base_world, 24))
    else:
        band = "medium"
        search_budget = min(max_alternatives, max(2, min(base_search, 3)))
        evidence_budget = max(1, min(base_evidence, 2))
        world_count = max(48, min(base_world, 72))
    return {
        "ambiguity_budget_prior": prior,
        "ambiguity_budget_band": band,
        "search_budget": int(search_budget),
        "evidence_budget": int(evidence_budget),
        "world_count": int(world_count),
        "path_count": int(path_count),
        "corridor_count": int(corridor_count),
        "objective_spread": round(float(spread), 6),
        "engine_disagreement_prior": round(float(engine_disagreement_prior), 6),
        "hard_case_prior": round(float(hard_case_prior), 6),
    }


def _effective_request_config(args: argparse.Namespace, od: dict[str, Any], *, variant_seed: int) -> dict[str, Any]:
    source = _row_override_source(od)
    profile_id = _text_or_none(od.get("profile_id"))
    corpus_group = _text_or_none(od.get("corpus_group")) or _text_or_none(od.get("corpus_kind"))
    row_overrides = {
        key: value
        for key, value in (
            ("weight_time", _text_or_none(source.get("weight_time"))),
            ("weight_money", _text_or_none(source.get("weight_money"))),
            ("weight_co2", _text_or_none(source.get("weight_co2"))),
            ("scenario_mode", _text_or_none(source.get("scenario_mode"))),
            ("weather_enabled", _text_or_none(source.get("weather_enabled"))),
            ("weather_profile", _text_or_none(source.get("weather_profile"))),
            ("weather_intensity", _text_or_none(source.get("weather_intensity"))),
            ("departure_time_utc", _text_or_none(source.get("departure_time_utc"))),
            ("stochastic_enabled", _text_or_none(source.get("stochastic_enabled"))),
            ("stochastic_samples", _text_or_none(source.get("stochastic_samples"))),
            ("search_budget", _text_or_none(source.get("search_budget"))),
            ("evidence_budget", _text_or_none(source.get("evidence_budget"))),
            ("world_count", _text_or_none(source.get("world_count"))),
            ("certificate_threshold", _text_or_none(source.get("certificate_threshold"))),
            ("tau_stop", _text_or_none(source.get("tau_stop"))),
            ("max_alternatives", _text_or_none(source.get("max_alternatives"))),
            ("optimization_mode", _text_or_none(source.get("optimization_mode"))),
        )
        if value is not None
    }
    departure_time_utc = _datetime_or_none(source.get("departure_time_utc")) or getattr(args, "departure_time_utc", None)
    if isinstance(departure_time_utc, str):
        departure_time_utc = _datetime_or_none(departure_time_utc)
    stochastic_enabled = _bool_or_default(source.get("stochastic_enabled"), bool(getattr(args, "stochastic_enabled", False)))
    stochastic_samples = _int_or_default(source.get("stochastic_samples"), int(getattr(args, "stochastic_samples", 25)))
    schedule = _ambiguity_budget_schedule(args, od, source=source)
    explicit_search_budget = _text_or_none(source.get("search_budget"))
    explicit_evidence_budget = _text_or_none(source.get("evidence_budget"))
    explicit_world_count = _text_or_none(source.get("world_count"))
    search_budget = (
        _int_or_default(explicit_search_budget, int(schedule["search_budget"]))
        if explicit_search_budget is not None
        else int(schedule["search_budget"])
    )
    evidence_budget = (
        _int_or_default(explicit_evidence_budget, int(schedule["evidence_budget"]))
        if explicit_evidence_budget is not None
        else int(schedule["evidence_budget"])
    )
    world_count = (
        _int_or_default(explicit_world_count, int(schedule["world_count"]))
        if explicit_world_count is not None
        else int(schedule["world_count"])
    )
    certificate_threshold = _float_or_default(source.get("certificate_threshold"), float(args.certificate_threshold))
    tau_stop = _float_or_default(source.get("tau_stop"), float(args.tau_stop))
    max_alternatives = _int_or_default(source.get("max_alternatives"), int(args.max_alternatives))
    weights = {
        "time": _float_or_default(source.get("weight_time"), float(args.weight_time)),
        "money": _float_or_default(source.get("weight_money"), float(args.weight_money)),
        "co2": _float_or_default(source.get("weight_co2"), float(args.weight_co2)),
    }
    weather_profile = _text_or_none(source.get("weather_profile")) or "clear"
    weather_intensity = _float_or_default(source.get("weather_intensity"), 1.0)
    weather_enabled = _bool_or_default(
        source.get("weather_enabled"),
        weather_profile.strip().lower() not in {"", "clear"},
    )
    optimization_mode = _text_or_none(source.get("optimization_mode")) or str(args.optimization_mode)
    scenario_mode = _text_or_none(source.get("scenario_mode")) or str(args.scenario_mode)
    config = {
        "profile_id": profile_id,
        "corpus_group": corpus_group,
        "row_override_keys": sorted(row_overrides),
        "row_override_count": len(row_overrides),
        "ambiguity_budget_band": schedule["ambiguity_budget_band"],
        "ambiguity_budget_prior": schedule["ambiguity_budget_prior"],
        "ambiguity_schedule_reason": {
            "path_count": schedule["path_count"],
            "corridor_count": schedule["corridor_count"],
            "objective_spread": schedule["objective_spread"],
            "engine_disagreement_prior": schedule["engine_disagreement_prior"],
            "hard_case_prior": schedule["hard_case_prior"],
        },
        "scenario_mode": scenario_mode,
        "departure_time_utc": departure_time_utc.isoformat() if isinstance(departure_time_utc, datetime) else None,
        "stochastic": {
            "enabled": stochastic_enabled,
            "seed": int(variant_seed),
            "samples": stochastic_samples,
        },
        "search_budget": search_budget,
        "evidence_budget": evidence_budget,
        "world_count": world_count,
        "certificate_threshold": certificate_threshold,
        "tau_stop": tau_stop,
        "max_alternatives": max_alternatives,
        "optimization_mode": optimization_mode,
        "weights": weights,
        "cost_toggles": {"use_tolls": not bool(args.disable_tolls)},
        "weather": {
            "enabled": weather_enabled,
            "profile": weather_profile,
            "intensity": weather_intensity,
        },
    }
    return config


def load_corpus(path: str) -> list[dict[str, Any]]:
    corpus_path = Path(path)
    if corpus_path.suffix.lower() == ".csv":
        with corpus_path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"]]
    raise ValueError("Unsupported corpus format")


def _corpus_missing_ambiguity_fields(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        ambiguity_present = row.get("od_ambiguity_index") not in (None, "") or row.get("ambiguity_index") not in (None, "")
        if not ambiguity_present:
            return True
        for field in (
            "candidate_probe_path_count",
            "candidate_probe_corridor_family_count",
            "candidate_probe_objective_spread",
            "candidate_probe_nominal_margin",
            "candidate_probe_toll_disagreement_rate",
            "candidate_probe_engine_disagreement_prior",
            "hard_case_prior",
            "od_ambiguity_confidence",
            "od_ambiguity_source_count",
            "od_ambiguity_source_mix",
            "od_ambiguity_source_support",
            "od_ambiguity_source_support_strength",
        ):
            if row.get(field) in (None, ""):
                return True
    return False


def _ensure_corpus_prior_fields(row: dict[str, Any]) -> dict[str, Any]:
    engine_prior_raw = row.get("candidate_probe_engine_disagreement_prior")
    hard_case_raw = row.get("hard_case_prior")
    if engine_prior_raw in (None, "") or hard_case_raw in (None, ""):
        cheap_priors = _cheap_prior_features(
            path_count=int(as_float(row.get("candidate_probe_path_count"), 0.0)),
            family_count=int(as_float(row.get("candidate_probe_corridor_family_count"), 0.0)),
            objective_spread=as_float(row.get("candidate_probe_objective_spread"), 0.0),
            nominal_margin=as_float(row.get("candidate_probe_nominal_margin"), 0.0),
            toll_disagreement=as_float(row.get("candidate_probe_toll_disagreement_rate"), 0.0),
        )
        if engine_prior_raw in (None, ""):
            row["candidate_probe_engine_disagreement_prior"] = cheap_priors["candidate_probe_engine_disagreement_prior"]
        if hard_case_raw in (None, ""):
            row["hard_case_prior"] = cheap_priors["hard_case_prior"]

    ambiguity_value = row.get("od_ambiguity_index")
    if ambiguity_value in (None, ""):
        ambiguity_value = row.get("ambiguity_index")
    ambiguity_float = as_float(ambiguity_value, float("nan"))
    if math.isfinite(ambiguity_float):
        explicit_prior_strength = None
        if row.get("od_ambiguity_prior_strength") not in (None, ""):
            candidate_strength = as_float(row.get("od_ambiguity_prior_strength"), float("nan"))
            if math.isfinite(candidate_strength):
                explicit_prior_strength = candidate_strength
        derived = _ambiguity_derived_fields(
            ambiguity_index=ambiguity_float,
            ambiguity_confidence=as_float(row.get("od_ambiguity_confidence"), 0.0),
            source_count=int(as_float(row.get("od_ambiguity_source_count"), 0.0)),
            source_mix=row.get("od_ambiguity_source_mix"),
            sample_count=int(as_float(row.get("ambiguity_prior_sample_count"), 0.0)),
            support_count=int(as_float(row.get("ambiguity_prior_support_count"), 0.0)),
            path_count=int(as_float(row.get("candidate_probe_path_count"), 0.0)),
            family_count=int(as_float(row.get("candidate_probe_corridor_family_count"), 0.0)),
            objective_spread=as_float(row.get("candidate_probe_objective_spread"), 0.0),
            nominal_margin=as_float(row.get("candidate_probe_nominal_margin"), 0.0),
            toll_disagreement=as_float(row.get("candidate_probe_toll_disagreement_rate"), 0.0),
            explicit_prior_strength=explicit_prior_strength,
        )
        for key, value in derived.items():
            if row.get(key) in (None, ""):
                row[key] = value
        if row.get("od_ambiguity_source_support") in (None, ""):
            source_tokens = _split_prior_sources(row.get("ambiguity_prior_source") or row.get("od_ambiguity_source_mix"))
            source_support = _source_support_map_for_row(row, source_tokens) if source_tokens else {}
            row["od_ambiguity_source_support"] = json.dumps(source_support, sort_keys=True, separators=(",", ":")) if source_support else None
        if row.get("od_ambiguity_source_support_strength") in (None, ""):
            row["od_ambiguity_source_support_strength"] = _source_support_strength(row.get("od_ambiguity_source_support"))
    return row


def _load_rows(rows: list[dict[str, Any]], *, seed: int, max_od: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        normalized_row = _ensure_corpus_prior_fields(dict(row))
        budget_prior = _od_ambiguity_prior(normalized_row)
        raw_ambiguity_prior = _shared_raw_ambiguity_prior_value(normalized_row)
        budget_prior_gap = (
            round(max(0.0, budget_prior - raw_ambiguity_prior), 6)
            if budget_prior is not None and raw_ambiguity_prior is not None
            else None
        )
        distance_km = _distance_km(normalized_row)
        corpus_kind = _text_or_none(normalized_row.get("corpus_kind"))
        corpus_group = _text_or_none(normalized_row.get("corpus_group")) or corpus_kind
        profile_id = _text_or_none(normalized_row.get("profile_id"))
        ambiguity_value = normalized_row.get("ambiguity_index")
        if ambiguity_value in (None, ""):
            ambiguity_value = normalized_row.get("od_ambiguity_index")
        row_overrides = _parse_row_overrides(normalized_row.get("row_overrides"))
        row_overrides.update({key: normalized_row.get(key) for key in ROW_OVERRIDE_FIELDS if _text_or_none(normalized_row.get(key)) is not None})
        out.append(
            {
                "od_id": str(normalized_row.get("od_id") or f"od-{index:06d}"),
                "origin_lat": float(normalized_row["origin_lat"]),
                "origin_lon": float(normalized_row["origin_lon"]),
                "destination_lat": float(normalized_row["destination_lat"]),
                "destination_lon": float(normalized_row["destination_lon"]),
                "straight_line_km": round(distance_km, 6),
                "trip_length_bin": str(normalized_row.get("distance_bin") or _bin(distance_km)),
                "seed": _int_or_default(normalized_row.get("seed"), int(seed)),
                "profile_id": profile_id,
                "corpus_group": corpus_group,
                "corpus_kind": corpus_kind,
                "ambiguity_index": as_float(ambiguity_value, float("nan")) if ambiguity_value not in (None, "") else None,
                "od_ambiguity_index": as_float(ambiguity_value, float("nan")) if ambiguity_value not in (None, "") else None,
                "od_ambiguity_confidence": as_float(normalized_row.get("od_ambiguity_confidence"), float("nan")) if normalized_row.get("od_ambiguity_confidence") not in (None, "") else None,
                "od_ambiguity_source_count": int(as_float(normalized_row.get("od_ambiguity_source_count"), 0.0)),
                "od_ambiguity_source_mix": _text_or_none(normalized_row.get("od_ambiguity_source_mix")),
                "od_ambiguity_source_mix_count": (
                    int(as_float(normalized_row.get("od_ambiguity_source_mix_count"), 0.0))
                    if normalized_row.get("od_ambiguity_source_mix_count") not in (None, "")
                    else _source_mix_count(normalized_row.get("od_ambiguity_source_mix"))
                ),
                "od_ambiguity_source_support": _text_or_none(normalized_row.get("od_ambiguity_source_support")),
                "od_ambiguity_source_support_strength": as_float(normalized_row.get("od_ambiguity_source_support_strength"), float("nan")) if normalized_row.get("od_ambiguity_source_support_strength") not in (None, "") else None,
                "od_ambiguity_source_entropy": as_float(normalized_row.get("od_ambiguity_source_entropy"), float("nan")) if normalized_row.get("od_ambiguity_source_entropy") not in (None, "") else None,
                "od_ambiguity_support_ratio": as_float(normalized_row.get("od_ambiguity_support_ratio"), float("nan")) if normalized_row.get("od_ambiguity_support_ratio") not in (None, "") else None,
                "od_ambiguity_prior_strength": as_float(normalized_row.get("od_ambiguity_prior_strength"), float("nan")) if normalized_row.get("od_ambiguity_prior_strength") not in (None, "") else None,
                "od_ambiguity_family_density": as_float(normalized_row.get("od_ambiguity_family_density"), float("nan")) if normalized_row.get("od_ambiguity_family_density") not in (None, "") else None,
                "od_ambiguity_margin_pressure": as_float(normalized_row.get("od_ambiguity_margin_pressure"), float("nan")) if normalized_row.get("od_ambiguity_margin_pressure") not in (None, "") else None,
                "od_ambiguity_spread_pressure": as_float(normalized_row.get("od_ambiguity_spread_pressure"), float("nan")) if normalized_row.get("od_ambiguity_spread_pressure") not in (None, "") else None,
                "od_ambiguity_toll_instability": as_float(normalized_row.get("od_ambiguity_toll_instability"), float("nan")) if normalized_row.get("od_ambiguity_toll_instability") not in (None, "") else None,
                "candidate_probe_path_count": int(as_float(normalized_row.get("candidate_probe_path_count"), 0.0)),
                "candidate_probe_corridor_family_count": int(as_float(normalized_row.get("candidate_probe_corridor_family_count"), 0.0)),
                "candidate_probe_objective_spread": as_float(normalized_row.get("candidate_probe_objective_spread"), float("nan")) if normalized_row.get("candidate_probe_objective_spread") not in (None, "") else None,
                "candidate_probe_nominal_margin": as_float(normalized_row.get("candidate_probe_nominal_margin"), float("nan")) if normalized_row.get("candidate_probe_nominal_margin") not in (None, "") else None,
                "candidate_probe_toll_disagreement_rate": as_float(normalized_row.get("candidate_probe_toll_disagreement_rate"), float("nan")) if normalized_row.get("candidate_probe_toll_disagreement_rate") not in (None, "") else None,
                "candidate_probe_engine_disagreement_prior": as_float(normalized_row.get("candidate_probe_engine_disagreement_prior"), float("nan")) if normalized_row.get("candidate_probe_engine_disagreement_prior") not in (None, "") else None,
                "hard_case_prior": as_float(normalized_row.get("hard_case_prior"), float("nan")) if normalized_row.get("hard_case_prior") not in (None, "") else None,
                "ambiguity_prior_sample_count": int(as_float(normalized_row.get("ambiguity_prior_sample_count"), 0.0)),
                "ambiguity_prior_support_count": int(as_float(normalized_row.get("ambiguity_prior_support_count"), 0.0)),
                "ambiguity_budget_prior": budget_prior,
                "ambiguity_budget_prior_gap": budget_prior_gap,
                "ambiguity_prior_source": _text_or_none(normalized_row.get("ambiguity_prior_source")),
                "corridor_bucket": _text_or_none(normalized_row.get("corridor_bucket")),
                "budget_prior_exceeds_raw": bool(
                    budget_prior is not None
                    and math.isfinite(raw_ambiguity_prior)
                    and budget_prior > raw_ambiguity_prior + 1e-9
                ),
                "row_overrides": row_overrides,
            }
        )
        if max_od and len(out) >= max_od:
            break
    if not out:
        raise ValueError("Corpus has no usable OD rows")
    return out


def _weights_tuple(args: argparse.Namespace) -> tuple[float, float, float]:
    return (max(float(args.weight_time), 0.0), max(float(args.weight_money), 0.0), max(float(args.weight_co2), 0.0))


def _weights_tuple_from_config(config: Mapping[str, Any]) -> tuple[float, float, float]:
    weights = config.get("weights")
    if not isinstance(weights, Mapping):
        return (1.0, 1.0, 1.0)
    return (
        max(as_float(weights.get("time"), 1.0), 0.0),
        max(as_float(weights.get("money"), 1.0), 0.0),
        max(as_float(weights.get("co2"), 1.0), 0.0),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic thesis evaluation over a fixed OD corpus.")
    corpus = parser.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--corpus-json", default=None)
    corpus.add_argument("--corpus-csv", default=None)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--in-process-backend", action="store_true")
    parser.add_argument("--ready-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--ready-poll-seconds", type=float, default=5.0)
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--max-od", type=int, default=0)
    parser.add_argument("--vehicle-type", default="rigid_hgv")
    parser.add_argument("--scenario-mode", default="no_sharing")
    parser.add_argument("--departure-time-utc", default=None)
    parser.add_argument("--model-version", default="thesis-script-v3")
    parser.add_argument("--optimization-mode", default="expected_value")
    parser.add_argument("--max-alternatives", type=int, default=8)
    parser.add_argument("--search-budget", type=int, default=4)
    parser.add_argument("--evidence-budget", type=int, default=2)
    parser.add_argument("--world-count", type=int, default=64)
    parser.add_argument("--certificate-threshold", type=float, default=0.80)
    parser.add_argument("--tau-stop", type=float, default=0.02)
    parser.add_argument("--stochastic-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stochastic-samples", type=int, default=25)
    parser.add_argument("--route-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--cache-mode",
        choices=THESIS_CACHE_MODES,
        default="preserve",
        help="`cold` clears backend route/process caches before each variant solve; `preserve` keeps intra-run cache carryover.",
    )
    parser.add_argument(
        "--cold-cache-scope",
        choices=THESIS_COLD_CACHE_SCOPES,
        default=THESIS_COLD_CACHE_SCOPE,
        help="Cache-clear scope used when `--cache-mode=cold` is active.",
    )
    parser.add_argument("--weight-time", type=float, default=1.0)
    parser.add_argument("--weight-money", type=float, default=1.0)
    parser.add_argument("--weight-co2", type=float, default=1.0)
    parser.add_argument("--fail-soft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-tolls", action="store_true")
    parser.add_argument("--baseline-refinement-policy", choices=("first_n", "random_n", "corridor_uniform"), default="corridor_uniform")
    parser.add_argument("--ors-baseline-policy", choices=("local_service", "repo_local", "snapshot_record", "snapshot_replay"), default="local_service")
    parser.add_argument("--ors-snapshot-mode", choices=("off", "record", "replay"), default="off")
    parser.add_argument("--ors-snapshot-path", default=None)
    parser.add_argument(
        "--auto-enrich-corpus-ambiguity",
        action="store_true",
        help="If set, backfill missing corpus ambiguity prior columns before evaluation. Disabled by default so thesis runs use explicit checked-in corpus priors and avoid hidden route-graph probe cost.",
    )
    parser.add_argument("--allow-proxy-ors", action="store_true", help="Allow proxy-labelled secondary baseline responses in thesis evaluation output.")
    parser.add_argument("--allow-evidence-fallbacks", action="store_true", help="Allow fallback/proxy evidence provenance in selected thesis routes.")
    parser.add_argument(
        "--evaluation-suite-role",
        default=None,
        help="Optional explicit suite-role label written into metadata/evaluation manifests to distinguish broad, focused, probe, and hot-rerun artifacts.",
    )
    return parser


def _suite_role_descriptor(role: str | None) -> dict[str, str]:
    requested_role = _text_or_none(role)
    normalized_role = (
        requested_role.strip().lower().replace("-", "_").replace(" ", "_")
        if requested_role is not None
        else "generic_evaluation"
    )
    defaults = EVALUATION_SUITE_ROLE_DEFAULTS.get(
        normalized_role,
        EVALUATION_SUITE_ROLE_DEFAULTS["generic_evaluation"],
    )
    return {
        "role": normalized_role,
        "scope": defaults["scope"],
        "focus": defaults["focus"],
        "label": defaults["label"],
    }


def _resolve_evaluation_suite_metadata(
    *,
    args: argparse.Namespace,
    corpus_source_path: str | None,
    run_id: str,
) -> dict[str, str]:
    explicit_role = _text_or_none(getattr(args, "evaluation_suite_role", None))
    if explicit_role is not None:
        return {
            **_suite_role_descriptor(explicit_role),
            "source": "explicit_arg",
        }

    inference_inputs = (
        ("corpus_source_path", _text_or_none(corpus_source_path)),
        ("run_id", _text_or_none(run_id)),
    )
    for source_name, raw_text in inference_inputs:
        hint = (raw_text or "").strip().lower()
        if not hint:
            continue
        if "hot_rerun" in hint:
            role = "hot_rerun"
        elif "dccs_probe" in hint or ("dccs" in hint and "probe" in hint):
            role = "dccs_diagnostic_probe"
        elif "refc_focus" in hint or ("refc" in hint and "focus" in hint):
            role = "focused_refc_proof"
        elif "voi_focus" in hint or ("voi" in hint and "focus" in hint):
            role = "focused_voi_proof"
        elif "thesis_broad" in hint or "broad" in hint:
            role = "broad_cold_proof"
        else:
            role = ""
        if role:
            return {
                **_suite_role_descriptor(role),
                "source": f"inferred_from_{source_name}",
            }
    return {
        **_suite_role_descriptor("generic_evaluation"),
        "source": "default_generic",
    }


def _wait_for_backend_ready(
    client: httpx.Client,
    *,
    backend_url: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    ready_started = time.perf_counter()
    deadline = time.perf_counter() + max(1.0, float(timeout_seconds))
    sleep_seconds = max(0.2, float(poll_seconds))
    last_payload: dict[str, Any] | None = None
    last_error: Exception | None = None
    ready_url = _absolute_url(backend_url, "/health/ready")
    attempts = 0
    while time.perf_counter() <= deadline:
        try:
            attempts += 1
            request_started = time.perf_counter()
            response = client.get(ready_url)
            elapsed_ms = (time.perf_counter() - request_started) * 1000.0
            if response.status_code >= 400:
                raise RuntimeError(f"ready_status_{response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("ready_response_not_object")
            payload.setdefault("compute_ms", round(elapsed_ms, 2))
            payload.setdefault("wait_elapsed_ms", round((time.perf_counter() - ready_started) * 1000.0, 2))
            payload.setdefault("attempt_count", attempts)
            last_payload = payload
            if bool(payload.get("strict_route_ready")):
                return payload
        except Exception as exc:  # pragma: no cover - defensive readiness polling
            last_error = exc
        time.sleep(sleep_seconds)
    if last_payload is not None:
        route_graph = last_payload.get("route_graph", {})
        strict_live = last_payload.get("strict_live", {})
        raise RuntimeError(
            "backend_not_ready: "
            f"route_graph_status={route_graph.get('status')} "
            f"route_graph_state={route_graph.get('state')} "
            f"route_graph_phase={route_graph.get('phase')} "
            f"strict_live_ok={strict_live.get('ok')}"
            f" recommended_action={last_payload.get('recommended_action')}"
        )
    if last_error is not None:
        raise RuntimeError(f"backend_not_ready: {type(last_error).__name__}: {last_error}") from last_error
    raise RuntimeError("backend_not_ready: readiness polling returned no payload")


def _base_payload(
    args: argparse.Namespace,
    od: dict[str, Any],
    *,
    variant_seed: int,
    request_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = request_config or _effective_request_config(args, od, variant_seed=variant_seed)
    payload = {
        "origin": {"lat": od["origin_lat"], "lon": od["origin_lon"]},
        "destination": {"lat": od["destination_lat"], "lon": od["destination_lon"]},
        "vehicle_type": args.vehicle_type,
        "scenario_mode": config["scenario_mode"],
        "weather": config.get("weather", {"enabled": False, "profile": "clear", "intensity": 1.0}),
        "departure_time_utc": config["departure_time_utc"],
        "max_alternatives": int(config["max_alternatives"]),
        "weights": config["weights"],
        "cost_toggles": config["cost_toggles"],
        "stochastic": config["stochastic"],
        "optimization_mode": config["optimization_mode"],
        "pipeline_seed": int(variant_seed),
        "search_budget": int(config["search_budget"]),
        "evidence_budget": int(config["evidence_budget"]),
        "cert_world_count": max(10, int(config["world_count"])),
        "certificate_threshold": float(config["certificate_threshold"]),
        "tau_stop": float(config["tau_stop"]),
        "evaluation_lean_mode": True,
    }
    prior_fields = {
        "od_ambiguity_index": od.get("od_ambiguity_index", od.get("ambiguity_index")),
        "od_ambiguity_confidence": od.get("od_ambiguity_confidence"),
        "od_ambiguity_source_count": od.get("od_ambiguity_source_count"),
        "od_ambiguity_source_mix": od.get("od_ambiguity_source_mix"),
        "od_ambiguity_source_mix_count": od.get("od_ambiguity_source_mix_count"),
        "od_ambiguity_source_entropy": od.get("od_ambiguity_source_entropy"),
        "od_ambiguity_support_ratio": od.get("od_ambiguity_support_ratio"),
        "od_ambiguity_prior_strength": od.get("od_ambiguity_prior_strength"),
        "od_ambiguity_family_density": od.get("od_ambiguity_family_density"),
        "od_ambiguity_margin_pressure": od.get("od_ambiguity_margin_pressure"),
        "od_ambiguity_spread_pressure": od.get("od_ambiguity_spread_pressure"),
        "od_ambiguity_toll_instability": od.get("od_ambiguity_toll_instability"),
        "od_engine_disagreement_prior": od.get("candidate_probe_engine_disagreement_prior"),
        "od_hard_case_prior": od.get("hard_case_prior"),
        "od_candidate_path_count": od.get("candidate_probe_path_count"),
        "od_corridor_family_count": od.get("candidate_probe_corridor_family_count"),
        "od_objective_spread": od.get("candidate_probe_objective_spread"),
        "od_nominal_margin_proxy": od.get("candidate_probe_nominal_margin"),
        "od_toll_disagreement_rate": od.get("candidate_probe_toll_disagreement_rate"),
        "ambiguity_budget_prior": config.get("ambiguity_budget_prior"),
        "ambiguity_budget_band": config.get("ambiguity_budget_band"),
    }
    for key, value in prior_fields.items():
        if value not in (None, ""):
            payload[key] = value
    return payload


def _variant_specs(args: argparse.Namespace) -> list[VariantSpec]:
    return [
        VariantSpec("A", "dccs", "dccs"),
        VariantSpec("B", "dccs_refc", "dccs"),
        VariantSpec("C", "voi", "dccs"),
        # Run the legacy baseline last so the thesis variants can reuse the
        # expensive search/option-build state they extend during suite runs.
        VariantSpec("V0", "legacy", str(args.baseline_refinement_policy)),
    ]


def _variant_specs_for_od(args: argparse.Namespace, *, od_index: int) -> list[VariantSpec]:
    del od_index
    return _variant_specs(args)


def _stable_row_variant_seed(od: Mapping[str, Any], *, base_seed: int) -> int:
    row_overrides = od.get("row_overrides")
    normalized_overrides = (
        {str(key): row_overrides[key] for key in sorted(row_overrides)}
        if isinstance(row_overrides, Mapping)
        else {}
    )
    seed_material = {
        "base_seed": int(base_seed),
        "od_id": str(od.get("od_id") or ""),
        "profile_id": str(od.get("profile_id") or ""),
        "corpus_group": str(od.get("corpus_group") or ""),
        "origin_lat": round(as_float(od.get("origin_lat"), 0.0), 6),
        "origin_lon": round(as_float(od.get("origin_lon"), 0.0), 6),
        "destination_lat": round(as_float(od.get("destination_lat"), 0.0), 6),
        "destination_lon": round(as_float(od.get("destination_lon"), 0.0), 6),
        "row_overrides": normalized_overrides,
    }
    digest = hashlib.sha1(json.dumps(seed_material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _variant_seed(args: argparse.Namespace, od: Mapping[str, Any], *, od_index: int, variant_id: str) -> int:
    del od_index
    del variant_id
    base_seed = _int_or_default(od.get("seed"), int(args.seed))
    return _stable_row_variant_seed(od, base_seed=int(base_seed))


def _variant_payload(
    args: argparse.Namespace,
    od: dict[str, Any],
    spec: VariantSpec,
    *,
    variant_seed: int,
    request_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _base_payload(args, od, variant_seed=variant_seed, request_config=request_config)
    payload["pipeline_mode"] = spec.pipeline_mode
    if spec.refinement_policy:
        payload["refinement_policy"] = spec.refinement_policy
    return payload


def _absolute_url(base_url: str, endpoint: str) -> str:
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def _post_json(client: httpx.Client, url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = client.post(url, json=payload)
    wall_ms = round((time.perf_counter() - started) * 1000.0, 3)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("expected JSON object response")
    compute_ms = data.get("compute_ms")
    if isinstance(compute_ms, (int, float)):
        return data, round(float(compute_ms), 3)
    return data, wall_ms


def _normalize_cache_mode(value: Any) -> str:
    candidate = str(value or "preserve").strip().lower()
    return candidate if candidate in THESIS_CACHE_MODES else "preserve"


def _clear_backend_caches(
    client: Any,
    *,
    backend_url: str,
    scope: str = THESIS_COLD_CACHE_SCOPE,
) -> dict[str, Any]:
    response = client.delete(_absolute_url(backend_url, f"/cache?scope={scope}"))
    status_code = int(getattr(response, "status_code", 500))
    if status_code >= 400:
        raise RuntimeError(f"cache_clear_failed:{status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("cache_clear_invalid_payload")
    return {str(key): payload[key] for key in payload}


def _baseline_smoke_payload() -> dict[str, Any]:
    return {
        "origin": {"lat": 52.4862, "lon": -1.8904},
        "destination": {"lat": 51.5072, "lon": -0.1276},
        "vehicle_type": "rigid_hgv",
    }


def _run_baseline_smoke(
    client: httpx.Client,
    *,
    base_url: str,
    ors_baseline_policy: str,
    snapshot_mode: str,
) -> dict[str, Any]:
    payload = _baseline_smoke_payload()
    normalized_ors_policy = str(ors_baseline_policy or "local_service").strip().lower()
    normalized_snapshot_mode = str(snapshot_mode or "off").strip().lower()
    requested_ors_mode = "repo_local" if normalized_ors_policy == "repo_local" else "local_service"
    summary: dict[str, Any] = {
        "checked_at_utc": _now(),
        "payload": payload,
        "ors_baseline_policy": normalized_ors_policy,
        "ors_snapshot_mode": normalized_snapshot_mode,
    }
    try:
        osrm = _fetch_baseline(
            client,
            _absolute_url(base_url, "/route/baseline?realism=false"),
            payload,
            default_method="osrm_engine_baseline",
        )
        summary["osrm"] = {
            "ok": True,
            "method": osrm.method,
            "provider_mode": osrm.provider_mode,
            "compute_ms": osrm.compute_ms,
            "distance_km": osrm.metrics.get("distance_km"),
            "duration_s": osrm.metrics.get("duration_s"),
        }
    except Exception as exc:
        summary["osrm"] = {
            "ok": False,
            "reason_code": _failure_reason(exc),
            "message": str(exc).strip() or exc.__class__.__name__,
        }
    if normalized_snapshot_mode == "replay":
        summary["ors"] = {
            "ok": True,
            "skipped": True,
            "provider_mode": "snapshot_replay",
            "baseline_policy": normalized_ors_policy,
            "message": "ORS live smoke skipped because snapshot replay mode is active.",
        }
    else:
        try:
            ors, _ = _ors_baseline(
                client,
                base_url,
                payload,
                od_id="baseline_smoke",
                snapshot_mode="off",
                snapshot_bundle=None,
                baseline_policy=requested_ors_mode,
            )
            engine_manifest = ors.engine_manifest if isinstance(ors.engine_manifest, dict) else {}
            summary["ors"] = {
                "ok": True,
                "method": ors.method,
                "provider_mode": ors.provider_mode,
                "baseline_policy": ors.baseline_policy,
                "compute_ms": ors.compute_ms,
                "distance_km": ors.metrics.get("distance_km"),
                "duration_s": ors.metrics.get("duration_s"),
                "asset_manifest_hash": ors.asset_manifest_hash,
                "asset_freshness_status": ors.asset_freshness_status,
                "graph_identity_status": engine_manifest.get("identity_status"),
                "engine_image": engine_manifest.get("compose_image"),
            }
        except Exception as exc:
            summary["ors"] = {
                "ok": False,
                "reason_code": _failure_reason(exc),
                "message": str(exc).strip() or exc.__class__.__name__,
            }
    summary["required_ok"] = bool((summary.get("osrm") or {}).get("ok")) and bool((summary.get("ors") or {}).get("ok"))
    return summary


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail")
        except Exception:
            detail = None
        if isinstance(detail, dict) and detail.get("reason_code"):
            return str(detail["reason_code"])
        return f"http_{exc.response.status_code}"
    if isinstance(exc, httpx.RequestError):
        return "transport_error"
    if isinstance(exc, (RuntimeError, ValueError)):
        message = str(exc).strip()
        if message:
            lowered = message.lower()
            if "evidence_provenance_rejected" in lowered:
                return "evidence_provenance_rejected"
            if "evidence_snapshot_not_allowed" in lowered:
                return "evidence_snapshot_not_allowed"
            if "evidence_freshness_missing" in lowered:
                return "evidence_freshness_missing"
            if "strict_proxy_ors_baseline_forbidden" in lowered or "proxy fallback" in lowered:
                return "strict_proxy_ors_baseline_forbidden"
            if "ors snapshot missing required fields" in lowered:
                return "ors_snapshot_missing_required_fields"
            if "request hash mismatch" in lowered:
                return "ors_snapshot_request_hash_mismatch"
            if "baseline_smoke_failed" in lowered:
                return "baseline_smoke_failed"
            if message.startswith("strict_"):
                return message
            return normalize_reason_code(message)
    return exc.__class__.__name__


def _baseline_route(resp: dict[str, Any], default_method: str) -> dict[str, Any]:
    route = dict(resp.get("baseline") or {})
    route.setdefault("id", f"{default_method}_baseline")
    return route


def _route_evidence_validation(route_payload: dict[str, Any], *, allow_snapshot: bool, require_freshness: bool = True) -> dict[str, Any]:
    validation = validate_route_evidence_provenance(
        route_payload,
        allow_snapshot=allow_snapshot,
        require_freshness=require_freshness,
    ).as_dict()
    validation["route_id"] = str(route_payload.get("id") or route_payload.get("route_id") or "")
    return validation


def _route_response_evidence_validation(route_response: dict[str, Any], *, allow_snapshot: bool) -> dict[str, Any]:
    validations: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    selected = route_response.get("selected")
    if isinstance(selected, dict):
        validation = _route_evidence_validation(selected, allow_snapshot=allow_snapshot)
        validations.append(validation)
        for issue in validation.get("issues", []):
            issue_payload = dict(issue)
            issue_payload["route_id"] = validation["route_id"] or "selected"
            issue_payload["selection_context"] = "selected"
            issues.append(issue_payload)
    candidates = route_response.get("candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            validation = _route_evidence_validation(candidate, allow_snapshot=allow_snapshot)
            validations.append(validation)
            for issue in validation.get("issues", []):
                issue_payload = dict(issue)
                issue_payload["route_id"] = validation["route_id"] or ""
                issue_payload["selection_context"] = "candidate"
                issues.append(issue_payload)
    return {
        "status": "ok" if not issues else "rejected",
        "issues": issues,
        "validations": validations,
        "policy": {
            "mode": STRICT_EVIDENCE_POLICY,
            "allow_snapshot": allow_snapshot,
            "require_freshness": True,
        },
    }


def _strict_failure_reason_from_issues(issues: Sequence[dict[str, Any]]) -> str:
    if not issues:
        return "strict_evidence_rejected"
    reason = str(issues[0].get("reason_code") or "strict_evidence_rejected")
    return reason


def _validate_snapshot_entry(entry: dict[str, Any], *, od_id: str, request_hash: str) -> None:
    required_keys = {
        "request_hash",
        "recorded_at",
        "provider_method",
        "provider_mode",
        "compute_ms",
        "response_hash",
        "response",
    }
    missing = sorted(key for key in required_keys if key not in entry)
    if missing:
        raise ValueError(f"ORS snapshot missing required fields for {od_id}: {', '.join(missing)}")
    if str(entry.get("request_hash")) != request_hash:
        raise ValueError(f"ORS snapshot request hash mismatch for {od_id}")
    provider_method = str(entry.get("provider_method") or "")
    provider_mode = str(entry.get("provider_mode") or "")
    baseline_policy = str(entry.get("baseline_policy") or "")
    if "proxy" in provider_method.lower() or "proxy" in provider_mode.lower():
        raise ValueError(f"ORS snapshot uses proxy provider metadata for {od_id}")
    if "synthetic" in provider_method.lower() or "synthetic" in provider_mode.lower() or "synthetic" in baseline_policy.lower():
        raise ValueError(f"ORS snapshot uses synthetic provider metadata for {od_id}")
    response = entry.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"ORS snapshot response missing for {od_id}")
    response_method = str(response.get("method") or provider_method or "")
    if "proxy" in response_method.lower() or "synthetic" in response_method.lower():
        raise ValueError(f"ORS snapshot response is synthetic/proxy for {od_id}")
    if provider_mode.strip().lower() == "local_service":
        asset_manifest_hash = str(
            entry.get("asset_manifest_hash") or response.get("asset_manifest_hash") or ""
        ).strip()
        if not asset_manifest_hash or asset_manifest_hash.startswith("ors-local:"):
            raise ValueError(f"ORS snapshot missing strict graph provenance hash for {od_id}")
        engine_manifest = response.get("engine_manifest")
        if not isinstance(engine_manifest, dict):
            engine_manifest = entry.get("engine_manifest")
        if not isinstance(engine_manifest, dict):
            raise ValueError(f"ORS snapshot missing engine provenance manifest for {od_id}")
        identity_status = str(
            engine_manifest.get("identity_status")
            or response.get("asset_freshness_status")
            or entry.get("asset_freshness_status")
            or ""
        ).strip()
        if identity_status != "graph_identity_verified":
            raise ValueError(f"ORS snapshot graph provenance is not verified for {od_id}: {identity_status or 'unknown'}")
        if not str(engine_manifest.get("graph_listing_digest") or "").strip():
            raise ValueError(f"ORS snapshot missing graph listing digest for {od_id}")


def _fetch_optional_json(client: httpx.Client, url: str) -> dict[str, Any] | None:
    resp = client.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, dict) else None


def _fetch_optional_jsonl(client: httpx.Client, url: str) -> list[dict[str, Any]] | None:
    resp = client.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    rows: list[dict[str, Any]] = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _fetch_run_artifacts(client: httpx.Client, base_url: str, route_response: dict[str, Any]) -> dict[str, Any]:
    artifacts_endpoint = str(route_response.get("artifacts_endpoint") or "").strip()
    run_id = str(route_response.get("run_id") or "").strip()
    if not artifacts_endpoint and run_id:
        artifacts_endpoint = f"/runs/{run_id}/artifacts"
    if not artifacts_endpoint:
        raise RuntimeError("strict_artifact_contract_missing:artifacts_endpoint")
    base_artifact_url = _absolute_url(base_url, artifacts_endpoint)
    payload: dict[str, Any] = {}
    for name in ARTEFACT_JSON_NAMES:
        value = _fetch_optional_json(client, f"{base_artifact_url.rstrip('/')}/{name}")
        if value is not None:
            payload[name] = value
    for name in ARTEFACT_JSONL_NAMES:
        value = _fetch_optional_jsonl(client, f"{base_artifact_url.rstrip('/')}/{name}")
        if value is not None:
            payload[name] = value
    return payload


def _required_artifacts_for_pipeline(pipeline_mode: str) -> tuple[str, ...]:
    mode = str(pipeline_mode or "").strip().lower()
    return REQUIRED_ARTIFACTS_BY_PIPELINE.get(mode, REQUIRED_ARTIFACTS_BY_PIPELINE["legacy"])


def _validate_artifact_payload(
    *,
    name: str,
    payload: Any,
    route_response: dict[str, Any],
    selected_route_id: str,
) -> None:
    if name.endswith(".json") and not isinstance(payload, dict):
        raise RuntimeError(f"strict_artifact_invalid:{name}")
    if name.endswith(".jsonl") and not isinstance(payload, list):
        raise RuntimeError(f"strict_artifact_invalid:{name}")
    if name == "metadata.json":
        run_id = str(route_response.get("run_id") or "").strip()
        if not run_id or str(payload.get("run_id") or "").strip() != run_id:
            raise RuntimeError("strict_artifact_invalid:metadata.json")
    elif name == FRONTIER_ARTIFACT:
        if not payload:
            raise RuntimeError(f"strict_artifact_invalid:{name}")
        route_ids = {
            str(row.get("route_id") or row.get("id") or "").strip()
            for row in payload
            if isinstance(row, dict)
        }
        if selected_route_id and selected_route_id not in route_ids:
            raise RuntimeError(f"strict_artifact_invalid:{name}")
    elif name == "final_route_trace.json":
        stage_timings = payload.get("stage_timings_ms")
        if not isinstance(stage_timings, dict):
            raise RuntimeError("strict_artifact_invalid:final_route_trace.json")
    elif name == "dccs_summary.json":
        for key in ("candidate_count_raw", "refined_count"):
            if key not in payload:
                raise RuntimeError("strict_artifact_invalid:dccs_summary.json")
    elif name == "dccs_candidates.jsonl":
        if not payload:
            raise RuntimeError("strict_artifact_invalid:dccs_candidates.jsonl")
    elif name == "certificate_summary.json":
        if "selected_certificate" not in payload or not isinstance(payload.get("active_families"), list):
            raise RuntimeError("strict_artifact_invalid:certificate_summary.json")
    elif name == "route_fragility_map.json":
        if selected_route_id and selected_route_id not in payload:
            raise RuntimeError("strict_artifact_invalid:route_fragility_map.json")
    elif name == "competitor_fragility_breakdown.json":
        if selected_route_id and selected_route_id not in payload:
            raise RuntimeError("strict_artifact_invalid:competitor_fragility_breakdown.json")
    elif name == "value_of_refresh.json":
        if not isinstance(payload.get("ranking"), list):
            raise RuntimeError("strict_artifact_invalid:value_of_refresh.json")
    elif name == "sampled_world_manifest.json":
        if as_float(payload.get("world_count"), float("nan")) <= 0.0:
            raise RuntimeError("strict_artifact_invalid:sampled_world_manifest.json")
    elif name == "voi_action_trace.json":
        if not isinstance(payload.get("actions"), list):
            raise RuntimeError("strict_artifact_invalid:voi_action_trace.json")
    elif name == "voi_stop_certificate.json":
        if not str(payload.get("stop_reason") or "").strip():
            raise RuntimeError("strict_artifact_invalid:voi_stop_certificate.json")


def _validate_route_artifacts(
    *,
    spec: VariantSpec,
    route_response: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    for field_name in ESSENTIAL_ROUTE_RESPONSE_FIELDS:
        if not str(route_response.get(field_name) or "").strip():
            raise RuntimeError(f"strict_artifact_contract_missing:{field_name}")
    selected_route = route_response.get("selected")
    selected_route_id = (
        str(selected_route.get("id") or selected_route.get("route_id") or "").strip()
        if isinstance(selected_route, dict)
        else ""
    )
    for name in _required_artifacts_for_pipeline(spec.pipeline_mode):
        if name not in artifacts:
            raise RuntimeError(f"strict_artifact_missing:{name}")
        _validate_artifact_payload(
            name=name,
            payload=artifacts[name],
            route_response=route_response,
            selected_route_id=selected_route_id,
        )


def _normalize_frontier_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        data = route_metrics({"metrics": metrics})
        return {"route_id": str(row.get("id") or row.get("route_id") or ""), "duration_s": data["duration_s"], "monetary_cost": data["monetary_cost"], "emissions_kg": data["emissions_kg"]}
    return {"route_id": str(row.get("route_id") or row.get("id") or ""), "duration_s": as_float(row.get("duration_s")), "monetary_cost": as_float(row.get("monetary_cost")), "emissions_kg": as_float(row.get("emissions_kg"))}


def _reference_point(frontier_rows: list[dict[str, Any]], baselines: Sequence[dict[str, float]]) -> dict[str, float]:
    durations = [as_float(row.get("duration_s")) for row in frontier_rows] + [row["duration_s"] for row in baselines]
    costs = [as_float(row.get("monetary_cost")) for row in frontier_rows] + [row["monetary_cost"] for row in baselines]
    co2s = [as_float(row.get("emissions_kg")) for row in frontier_rows] + [row["emissions_kg"] for row in baselines]
    return {"duration_s": max(durations or [1.0]) * 1.05, "monetary_cost": max(costs or [1.0]) * 1.05, "emissions_kg": max(co2s or [1.0]) * 1.05}


def _top_refresh_stats(value_of_refresh: dict[str, Any]) -> tuple[str | None, float | None, float | None]:
    ranking = value_of_refresh.get("ranking", [])
    if not isinstance(ranking, list) or not ranking:
        return None, None, None
    top = ranking[0] if isinstance(ranking[0], dict) else {}
    second = ranking[1] if len(ranking) > 1 and isinstance(ranking[1], dict) else {}
    top_family = str(top.get("family", "")).strip() or None
    top_vor = as_float(top.get("vor"), float("nan"))
    if not math.isfinite(top_vor):
        return top_family, None, None
    return top_family, round(top_vor, 6), round(top_vor - as_float(second.get("vor"), 0.0), 6)


def _controller_refresh_stats(
    value_of_refresh: Mapping[str, Any],
) -> tuple[str | None, str | None, float | None, bool | None, bool | None]:
    empirical_top_family, empirical_top_gain, _ = _top_refresh_stats(dict(value_of_refresh))
    controller_ranking_basis = str(value_of_refresh.get("controller_ranking_basis") or "").strip() or None
    controller_top_family = str(value_of_refresh.get("top_refresh_family_controller") or "").strip() or None
    controller_top_gain = as_float(value_of_refresh.get("top_refresh_gain_controller"), float("nan"))
    controller_ranking = value_of_refresh.get("controller_ranking", [])
    if isinstance(controller_ranking, list) and controller_ranking:
        first = controller_ranking[0] if isinstance(controller_ranking[0], Mapping) else {}
        if controller_top_family is None:
            controller_top_family = str(first.get("family") or "").strip() or None
        if not math.isfinite(controller_top_gain):
            controller_top_gain = as_float(first.get("controller_score"), float("nan"))
    if not math.isfinite(controller_top_gain):
        controller_top_gain = None
    fallback_activated = (
        controller_ranking_basis == "raw_refresh_gain_fallback"
        if controller_ranking_basis is not None
        else None
    )
    disagreement = None
    if fallback_activated is not None:
        zero_to_nonzero_signal_upgrade = bool(
            fallback_activated
            and controller_top_gain is not None
            and controller_top_gain > 1e-9
            and (empirical_top_gain is None or empirical_top_gain <= 1e-9)
        )
        disagreement = bool(
            fallback_activated
            and (
                controller_top_family != empirical_top_family
                or zero_to_nonzero_signal_upgrade
                or (
                    controller_top_gain is not None
                    and empirical_top_gain is not None
                    and controller_top_gain > empirical_top_gain + 1e-9
                )
            )
        )
    return (
        controller_ranking_basis,
        controller_top_family,
        round(controller_top_gain, 6) if controller_top_gain is not None else None,
        fallback_activated,
        disagreement,
    )


def _certificate_map(certificate_summary: dict[str, Any]) -> dict[str, float]:
    if not isinstance(certificate_summary, dict):
        return {}
    raw_map = certificate_summary.get("certificates")
    if isinstance(raw_map, dict):
        out = {
            str(route_id): as_float(value, float("nan"))
            for route_id, value in raw_map.items()
            if math.isfinite(as_float(value, float("nan")))
        }
        if out:
            return out
    raw_rows = certificate_summary.get("route_certificates")
    if isinstance(raw_rows, dict):
        out = {
            str(route_id): as_float(value, float("nan"))
            for route_id, value in raw_rows.items()
            if math.isfinite(as_float(value, float("nan")))
        }
        if out:
            return out
    if isinstance(raw_rows, list):
        out: dict[str, float] = {}
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            route_id = str(row.get("route_id") or row.get("id") or "").strip()
            value = as_float(row.get("certificate"), float("nan"))
            if route_id and math.isfinite(value):
                out[route_id] = value
        if out:
            return out
    selected_id = str(certificate_summary.get("selected_route_id") or "").strip()
    selected_cert = as_float(certificate_summary.get("selected_certificate"), float("nan"))
    if selected_id and math.isfinite(selected_cert):
        return {selected_id: selected_cert}
    return {}


def _top_fragility(route_fragility_map: dict[str, Any], route_id: str) -> str | None:
    raw = route_fragility_map.get(route_id)
    if not isinstance(raw, dict) or not raw:
        return None
    return max(raw.items(), key=lambda item: (as_float(item[1]), str(item[0])))[0]


def _top_fragility_value(route_fragility_map: dict[str, Any], route_id: str) -> float | None:
    raw = route_fragility_map.get(route_id)
    if not isinstance(raw, dict) or not raw:
        return None
    best_value = as_float(
        max(raw.items(), key=lambda item: (as_float(item[1]), str(item[0])))[1],
        float("nan"),
    )
    return round(best_value, 6) if math.isfinite(best_value) else None


def _top_competitor(competitor_breakdown: dict[str, Any], route_id: str) -> str | None:
    raw = competitor_breakdown.get(route_id)
    if not isinstance(raw, dict) or not raw:
        return None
    totals = {str(competitor_id): sum(int(as_float(value)) for value in family_map.values()) if isinstance(family_map, dict) else 0 for competitor_id, family_map in raw.items()}
    return max(totals.items(), key=lambda item: (item[1], item[0]))[0] if totals else None


def _source_mix_count(raw_source_mix: Any) -> int:
    if raw_source_mix in (None, ""):
        return 0
    if isinstance(raw_source_mix, Mapping):
        return len([key for key in raw_source_mix if str(key).strip()])
    if isinstance(raw_source_mix, Sequence) and not isinstance(raw_source_mix, (str, bytes)):
        return len([item for item in raw_source_mix if str(item).strip()])
    text = str(raw_source_mix).strip()
    if not text:
        return 0
    if text[:1] in {"{", "["}:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        else:
            return _source_mix_count(parsed)
    return len([item.strip() for item in text.replace("+", ",").split(",") if item.strip()])


def _fallback_support_richness(od: Mapping[str, Any]) -> float | None:
    prior_strength = as_float(od.get("od_ambiguity_prior_strength"), float("nan"))
    support_ratio = as_float(od.get("od_ambiguity_support_ratio"), float("nan"))
    source_entropy = as_float(od.get("od_ambiguity_source_entropy"), float("nan"))
    support_strength = as_float(od.get("od_ambiguity_source_support_strength"), float("nan"))
    confidence = as_float(od.get("od_ambiguity_confidence"), float("nan"))
    source_count = min(1.0, max(0.0, as_float(od.get("od_ambiguity_source_count"), 0.0) / 4.0))
    source_mix = min(1.0, max(0.0, _source_mix_count(od.get("od_ambiguity_source_mix")) / 3.0))
    source_mix_count = min(1.0, max(0.0, as_float(od.get("od_ambiguity_source_mix_count"), 0.0) / 4.0))
    family_density = min(1.0, max(0.0, as_float(od.get("od_ambiguity_family_density"), 0.0)))
    candidate_paths = min(1.0, max(0.0, as_float(od.get("candidate_probe_path_count"), 0.0) / 6.0))
    corridor_count = min(1.0, max(0.0, as_float(od.get("candidate_probe_corridor_family_count"), 0.0) / 4.0))
    terms = [
        0.19 * max(0.0, min(1.0, prior_strength)) if math.isfinite(prior_strength) else 0.0,
        0.17 * max(0.0, min(1.0, support_strength)) if math.isfinite(support_strength) else 0.0,
        0.15 * max(0.0, min(1.0, support_ratio)) if math.isfinite(support_ratio) else 0.0,
        0.12 * max(0.0, min(1.0, source_entropy)) if math.isfinite(source_entropy) else 0.0,
        0.10 * max(0.0, min(1.0, confidence)) if math.isfinite(confidence) else 0.0,
        0.08 * source_count,
        0.08 * source_mix,
        0.07 * source_mix_count,
        0.06 * candidate_paths,
        0.05 * corridor_count,
        0.05 * family_density,
    ]
    if not any(term > 0.0 for term in terms):
        return None
    return round(max(0.0, min(1.0, sum(terms))), 6)


def _fallback_ambiguity_pressure(od: Mapping[str, Any]) -> float | None:
    margin_pressure = as_float(od.get("od_ambiguity_margin_pressure"), float("nan"))
    spread_pressure = as_float(od.get("od_ambiguity_spread_pressure"), float("nan"))
    engine_prior = as_float(od.get("candidate_probe_engine_disagreement_prior"), float("nan"))
    hard_case_prior = as_float(od.get("hard_case_prior"), float("nan"))
    toll_instability = as_float(od.get("od_ambiguity_toll_instability"), float("nan"))
    near_tie_proxy = as_float(od.get("candidate_probe_top2_gap_pressure"), float("nan"))
    terms = [
        0.27 * max(0.0, min(1.0, margin_pressure)) if math.isfinite(margin_pressure) else 0.0,
        0.23 * max(0.0, min(1.0, spread_pressure)) if math.isfinite(spread_pressure) else 0.0,
        0.22 * max(0.0, min(1.0, engine_prior)) if math.isfinite(engine_prior) else 0.0,
        0.16 * max(0.0, min(1.0, hard_case_prior)) if math.isfinite(hard_case_prior) else 0.0,
        0.07 * max(0.0, min(1.0, toll_instability)) if math.isfinite(toll_instability) else 0.0,
        0.05 * max(0.0, min(1.0, near_tie_proxy)) if math.isfinite(near_tie_proxy) else 0.0,
    ]
    if not any(term > 0.0 for term in terms):
        return None
    return round(max(0.0, min(1.0, sum(terms))), 6)


def _dccs_score_correlation(rows: list[dict[str, Any]]) -> float | None:
    scores: list[float] = []
    labels: list[int] = []
    for row in rows:
        score = as_float(row.get("final_score"), float("nan"))
        if not math.isfinite(score):
            continue
        labels.append(1 if str(row.get("decision_reason", "")) in {"frontier_addition", "decision_flip", "challenger_but_not_added"} else 0)
        scores.append(score)
    return pearson_binary_correlation(scores, labels)


def _certificate_winner_id(certificates: Mapping[str, float]) -> str | None:
    if not certificates:
        return None
    return max(
        certificates.items(),
        key=lambda item: (as_float(item[1], float("-inf")), str(item[0])),
    )[0]


def _action_counts(action_trace: dict[str, Any]) -> tuple[int, int, int, int]:
    actions = action_trace.get("actions", [])
    if not isinstance(actions, list):
        return 0, 0, 0, 0
    refine = refresh = resample = 0
    for row in actions:
        chosen = row.get("chosen_action") if isinstance(row, dict) else None
        kind = str(chosen.get("kind") or chosen.get("action") or "") if isinstance(chosen, dict) else ""
        if kind.startswith("refine"):
            refine += 1
        elif kind.startswith("refresh"):
            refresh += 1
        elif "stochastic" in kind or "resample" in kind:
            resample += 1
    return len(actions), refine, refresh, resample


def _voi_action_entries(voi_trace: Mapping[str, Any], voi_stop: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace_actions = voi_trace.get("actions")
    if isinstance(trace_actions, list) and trace_actions:
        return [dict(item) for item in trace_actions if isinstance(item, Mapping)]
    stop_actions = voi_stop.get("action_trace")
    if isinstance(stop_actions, list):
        return [dict(item) for item in stop_actions if isinstance(item, Mapping)]
    return []


def _action_realized_certificate_delta(action_entry: Mapping[str, Any]) -> float | None:
    for key in ("realized_certificate_delta",):
        value = as_float(action_entry.get(key), float("nan"))
        if math.isfinite(value):
            return round(value, 6)
    before = as_float(action_entry.get("realized_certificate_before"), float("nan"))
    after = as_float(action_entry.get("realized_certificate_after"), float("nan"))
    if math.isfinite(before) and math.isfinite(after):
        return round(after - before, 6)
    return None


def _action_realized_frontier_gain(action_entry: Mapping[str, Any]) -> int:
    value = as_float(action_entry.get("realized_frontier_gain"), float("nan"))
    if math.isfinite(value):
        return int(round(value))
    return 0


def _action_realized_runner_up_gap_delta(action_entry: Mapping[str, Any]) -> float | None:
    for key in ("realized_runner_up_gap_delta",):
        value = as_float(action_entry.get(key), float("nan"))
        if math.isfinite(value):
            return round(value, 6)
    before = as_float(action_entry.get("realized_runner_up_gap_before"), float("nan"))
    after = as_float(action_entry.get("realized_runner_up_gap_after"), float("nan"))
    if math.isfinite(before) and math.isfinite(after):
        return round(after - before, 6)
    return None


def _action_realized_evidence_uncertainty_delta(action_entry: Mapping[str, Any]) -> float | None:
    for key in ("realized_evidence_uncertainty_delta",):
        value = as_float(action_entry.get(key), float("nan"))
        if math.isfinite(value):
            return round(value, 6)
    before = as_float(action_entry.get("realized_evidence_uncertainty_before"), float("nan"))
    after = as_float(action_entry.get("realized_evidence_uncertainty_after"), float("nan"))
    if math.isfinite(before) and math.isfinite(after):
        return round(after - before, 6)
    return None


def _action_realized_productive(action_entry: Mapping[str, Any]) -> bool | None:
    explicit = action_entry.get("realized_productive")
    if isinstance(explicit, bool):
        return explicit
    certificate_delta = _action_realized_certificate_delta(action_entry)
    selected_changed = bool(action_entry.get("realized_selected_route_changed"))
    frontier_gain = _action_realized_frontier_gain(action_entry)
    selected_score_delta = as_float(action_entry.get("realized_selected_score_delta"), float("nan"))
    runner_up_gap_delta = _action_realized_runner_up_gap_delta(action_entry)
    evidence_uncertainty_delta = _action_realized_evidence_uncertainty_delta(action_entry)
    if (
        certificate_delta is None
        and frontier_gain <= 0
        and not selected_changed
        and not math.isfinite(selected_score_delta)
        and runner_up_gap_delta is None
        and evidence_uncertainty_delta is None
    ):
        return None
    return bool(
        (certificate_delta is not None and certificate_delta > 1e-9)
        or frontier_gain > 0
        or selected_changed
        or (math.isfinite(selected_score_delta) and selected_score_delta < -1e-9)
        or (runner_up_gap_delta is not None and runner_up_gap_delta > 1e-9)
        or (evidence_uncertainty_delta is not None and evidence_uncertainty_delta < -1e-9)
    )


def _refresh_first_resolution_state(
    *,
    first_controller_action_kind: str | None,
    voi_entries: Sequence[Mapping[str, Any]],
    initial_winner_fragility_nonzero: bool,
    initial_refc_top_vor_positive: bool,
    refresh_signal_persistent: bool | None,
) -> tuple[bool | None, bool | None, str | None]:
    if first_controller_action_kind != "refresh_top1_vor":
        return None, None, "not_refresh_first"

    first_entry: Mapping[str, Any] = voi_entries[0] if voi_entries else {}
    refresh_first_productive = bool(_action_realized_productive(first_entry))
    has_initial_signal = bool(initial_winner_fragility_nonzero or initial_refc_top_vor_positive)
    if not has_initial_signal:
        return refresh_first_productive, None, "no_initial_signal"

    if refresh_signal_persistent is True:
        return refresh_first_productive, True, "persistent_signal"

    certificate_delta = _action_realized_certificate_delta(first_entry)
    runner_up_gap_delta = _action_realized_runner_up_gap_delta(first_entry)
    evidence_uncertainty_delta = _action_realized_evidence_uncertainty_delta(first_entry)
    if certificate_delta is not None and certificate_delta > 1e-9:
        return refresh_first_productive, True, "certificate_lift"
    if runner_up_gap_delta is not None and runner_up_gap_delta > 1e-9:
        return refresh_first_productive, True, "runner_up_gap_gain"
    if evidence_uncertainty_delta is not None and evidence_uncertainty_delta < -1e-9:
        return refresh_first_productive, True, "evidence_uncertainty_reduced"
    return refresh_first_productive, False, "unresolved"


def _voi_action_productivity(
    voi_entries: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    total = 0
    productive = 0
    nonproductive_refine = 0
    first_productive_index: int | None = None
    for index, entry in enumerate(voi_entries):
        kind = _chosen_action_kind(entry)
        if not kind or kind == "stop":
            continue
        if _action_realized_productive(entry) is True:
            first_productive_index = index
            break
    leading_refine_probe_credit_remaining = first_productive_index is not None
    for index, entry in enumerate(voi_entries):
        kind = _chosen_action_kind(entry)
        if not kind or kind == "stop":
            continue
        total += 1
        productive_flag = _action_realized_productive(entry)
        if productive_flag is True:
            productive += 1
        elif productive_flag is False and _is_refine_action(kind):
            # A single leading nonproductive refine can be an exploratory probe
            # that exposes the later productive controller phase rather than churn.
            leading_refine_probe = bool(
                leading_refine_probe_credit_remaining
                and first_productive_index is not None
                and index < first_productive_index
                and all(
                    _is_refine_action(_chosen_action_kind(prefix_entry))
                    and _action_realized_productive(prefix_entry) is False
                    for prefix_entry in voi_entries[: index + 1]
                    if _chosen_action_kind(prefix_entry) not in {"", "stop"}
                )
            )
            if leading_refine_probe:
                leading_refine_probe_credit_remaining = False
                continue
            nonproductive_refine += 1
    return productive, total, nonproductive_refine


def _action_snapshot_certificate(action_entry: Mapping[str, Any]) -> float | None:
    for key in ("selected_certificate", "certificate_value", "current_certificate"):
        value = as_float(action_entry.get(key), float("nan"))
        if math.isfinite(value):
            return round(value, 6)
    return None


def _chosen_action_kind(action_entry: Mapping[str, Any]) -> str:
    chosen = action_entry.get("chosen_action")
    if isinstance(chosen, Mapping):
        return str(chosen.get("kind") or "").strip()
    return ""


def _is_evidence_action(kind: str) -> bool:
    normalized = str(kind or "").strip().lower()
    return normalized.startswith("refresh") or "stochastic" in normalized or "resample" in normalized


def _is_refine_action(kind: str) -> bool:
    return str(kind or "").strip().lower().startswith("refine")


def _result_refine_cost_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    pipeline_mode: str,
    voi_entries: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    valid_rows = _valid_refine_cost_rows(candidate_rows)
    if pipeline_mode != "voi" or not valid_rows:
        return valid_rows

    refine_entries = [entry for entry in voi_entries if _is_refine_action(_chosen_action_kind(entry))]
    if not refine_entries:
        return []

    executed_candidate_ids: set[str] = set()
    executed_metadata_seen = False
    for entry in refine_entries:
        entry_candidate_ids = _action_candidate_ids(entry, key="executed_candidate_ids")
        if entry_candidate_ids:
            executed_metadata_seen = True
            executed_candidate_ids.update(entry_candidate_ids)
    if not executed_metadata_seen:
        return valid_rows

    return [
        row
        for row in valid_rows
        if str(row.get("candidate_id") or "").strip() in executed_candidate_ids
    ]


def _first_controller_action_kind(voi_entries: Sequence[Mapping[str, Any]]) -> str | None:
    for entry in voi_entries:
        kind = _chosen_action_kind(entry)
        if kind:
            return kind
    return None


def _initial_certificate(
    *,
    voi_entries: Sequence[Mapping[str, Any]],
    certificate: float | None,
    threshold: float | None,
    stop_reason: str | None,
) -> float | None:
    if voi_entries:
        return _action_snapshot_certificate(voi_entries[0])
    final_certificate = as_float(certificate, float("nan"))
    final_threshold = as_float(threshold, float("nan"))
    if (
        math.isfinite(final_certificate)
        and math.isfinite(final_threshold)
        and final_certificate >= final_threshold
        and str(stop_reason or "").strip() == "certified"
    ):
        return round(final_certificate, 6)
    return None


def _time_to_certification_ms(
    *,
    voi_entries: Sequence[Mapping[str, Any]],
    voi_stop: Mapping[str, Any],
    trace: Mapping[str, Any],
    initial_certificate: float | None,
    certificate: float | None,
    threshold: float | None,
    stage_voi_ms: float | None,
) -> float | None:
    explicit_sources: Sequence[Any] = (
        voi_stop.get("time_to_certification_ms"),
        ((trace.get("certification_runtime") or {}) if isinstance(trace.get("certification_runtime"), Mapping) else {}).get("time_to_certification_ms"),
    )
    for raw_value in explicit_sources:
        value = as_float(raw_value, float("nan"))
        if math.isfinite(value):
            return round(value, 6)
    initial = as_float(initial_certificate, float("nan"))
    final_certificate = as_float(certificate, float("nan"))
    final_threshold = as_float(threshold, float("nan"))
    if math.isfinite(initial) and math.isfinite(final_threshold) and initial >= final_threshold:
        return 0.0
    if not math.isfinite(final_certificate) or not math.isfinite(final_threshold) or final_certificate < final_threshold:
        return None
    voi_stage = as_float(stage_voi_ms, float("nan"))
    if not math.isfinite(voi_stage):
        return None
    for index, entry in enumerate(voi_entries):
        cert_value = _action_snapshot_certificate(entry)
        if cert_value is not None and cert_value >= final_threshold:
            return round(voi_stage * (index / max(1, len(voi_entries))), 6)
    return round(voi_stage, 6)


def _voi_realized_certificate_lift(
    *,
    voi_entries: Sequence[Mapping[str, Any]],
    initial_certificate: float | None,
    certificate: float | None,
) -> float | None:
    initial = as_float(initial_certificate, float("nan"))
    final_certificate = as_float(certificate, float("nan"))
    if math.isfinite(initial) and math.isfinite(final_certificate):
        return round(final_certificate - initial, 6)
    cumulative_delta = 0.0
    observed_delta = False
    for entry in voi_entries:
        delta = _action_realized_certificate_delta(entry)
        if delta is None:
            continue
        cumulative_delta += delta
        observed_delta = True
    if observed_delta:
        return round(cumulative_delta, 6)
    return None


def _option_build_reuse_rate(trace: Mapping[str, Any], trace_candidate_diag: Mapping[str, Any]) -> float | None:
    route_option_runtime = (
        trace.get("route_option_cache_runtime")
        if isinstance(trace.get("route_option_cache_runtime"), Mapping)
        else {}
    )
    option_build_runtime = (
        trace.get("option_build_runtime")
        if isinstance(trace.get("option_build_runtime"), Mapping)
        else {}
    )

    def _runtime_has_signal(runtime: Mapping[str, Any]) -> bool:
        hits = as_float(runtime.get("cache_hits"), float("nan"))
        misses = as_float(runtime.get("cache_misses"), float("nan"))
        reuse_rate = as_float(runtime.get("reuse_rate"), float("nan"))
        return (
            math.isfinite(hits)
            and float(hits) > 0.0
        ) or (
            math.isfinite(misses)
            and float(misses) > 0.0
        ) or (
            math.isfinite(reuse_rate)
        )

    explicit_candidates: Sequence[Any] = (
        route_option_runtime.get("reuse_rate") if _runtime_has_signal(route_option_runtime) else None,
        option_build_runtime.get("reuse_rate") if _runtime_has_signal(option_build_runtime) else None,
        trace_candidate_diag.get("option_build_reuse_rate"),
    )
    for raw_value in explicit_candidates:
        value = as_float(raw_value, float("nan"))
        if math.isfinite(value):
            return round(value, 6)
    return None


def _option_build_cache_runtime(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    route_option_runtime = (
        trace.get("route_option_cache_runtime")
        if isinstance(trace.get("route_option_cache_runtime"), Mapping)
        else {}
    )
    option_build_runtime = (
        trace.get("option_build_runtime")
        if isinstance(trace.get("option_build_runtime"), Mapping)
        else {}
    )

    def _event_count(runtime: Mapping[str, Any]) -> int:
        hits = as_float(runtime.get("cache_hits"), float("nan"))
        misses = as_float(runtime.get("cache_misses"), float("nan"))
        return int(max(0.0, hits if math.isfinite(hits) else 0.0)) + int(
            max(0.0, misses if math.isfinite(misses) else 0.0)
        )

    if _event_count(route_option_runtime) > 0:
        return route_option_runtime
    return option_build_runtime


def _controller_state_rows(artifacts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = artifacts.get("voi_controller_state.jsonl", [])
    if not isinstance(rows, Sequence):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _controller_signal(
    controller_rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    voi_stop: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> float | None:
    explicit_candidates: list[Any] = []
    if controller_rows:
        explicit_candidates.append(controller_rows[0].get(key))
    if isinstance(voi_stop, Mapping):
        explicit_candidates.append(voi_stop.get(key))
    if isinstance(trace, Mapping):
        voi_payload = trace.get("voi", {})
        if isinstance(voi_payload, Mapping):
            explicit_candidates.append(voi_payload.get(key))
    for raw_value in explicit_candidates:
        value = as_float(raw_value, float("nan"))
        if math.isfinite(value):
            return round(value, 6)
    return None


def _controller_flag(
    controller_rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    voi_stop: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> bool | None:
    explicit_candidates: list[Any] = []
    if controller_rows:
        explicit_candidates.append(controller_rows[0].get(key))
    if isinstance(voi_stop, Mapping):
        explicit_candidates.append(voi_stop.get(key))
    if isinstance(trace, Mapping):
        voi_payload = trace.get("voi", {})
        if isinstance(voi_payload, Mapping):
            explicit_candidates.append(voi_payload.get(key))
    for raw_value in explicit_candidates:
        if isinstance(raw_value, bool):
            return raw_value
        numeric = as_float(raw_value, float("nan"))
        if math.isfinite(numeric):
            return bool(numeric)
        text = str(raw_value or "").strip().lower()
        if text in {"true", "false"}:
            return text == "true"
    return None


def _stage_value(trace: dict[str, Any], key: str) -> float | None:
    stage_timings = trace.get("stage_timings_ms", {})
    if not isinstance(stage_timings, dict) or key not in stage_timings:
        return None
    value = as_float(stage_timings.get(key), float("nan"))
    return round(value, 3) if math.isfinite(value) else None


def _stage_value_any(trace: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _stage_value(trace, key)
        if value is not None:
            return value
    return None


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    finite = [as_float(row.get(key), float("nan")) for row in rows]
    finite = [value for value in finite if math.isfinite(value)]
    return round(sum(finite) / len(finite), 6) if finite else None


def _percentile_numeric(rows: list[dict[str, Any]], key: str, quantile: float) -> float | None:
    finite = [as_float(row.get(key), float("nan")) for row in rows]
    finite = [value for value in finite if math.isfinite(value)]
    return percentile(finite, quantile) if finite else None


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return round(sum(1.0 if bool(value) else 0.0 for value in values) / len(values), 6) if values else None


def _mean_numeric_with_denominator(rows: list[dict[str, Any]], key: str) -> tuple[float | None, int]:
    finite = [as_float(row.get(key), float("nan")) for row in rows]
    finite = [value for value in finite if math.isfinite(value)]
    return (round(sum(finite) / len(finite), 6), len(finite)) if finite else (None, 0)


def _mean_ratio(rows: list[dict[str, Any]], numerator_key: str, denominator_key: str) -> float | None:
    ratios: list[float] = []
    for row in rows:
        numerator = as_float(row.get(numerator_key), float("nan"))
        denominator = as_float(row.get(denominator_key), float("nan"))
        if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
            continue
        ratios.append(numerator / denominator)
    return round(sum(ratios) / len(ratios), 6) if ratios else None


def _observed_ambiguity_index(row: Mapping[str, Any]) -> float | None:
    signals: list[float] = []
    frontier_count = as_float(row.get("frontier_count"), float("nan"))
    if math.isfinite(frontier_count):
        signals.append(min(1.0, max(0.0, (frontier_count - 1.0) / 5.0)))
    near_ties = as_float(row.get("near_tie_mass"), float("nan"))
    if math.isfinite(near_ties):
        signals.append(min(1.0, max(0.0, near_ties)))
    winner_margin = as_float(row.get("nominal_winner_margin"), float("nan"))
    if math.isfinite(winner_margin):
        signals.append(min(1.0, max(0.0, 1.0 - winner_margin)))
    certificate_margin_value = as_float(row.get("certificate_margin"), float("nan"))
    certificate_threshold = as_float(row.get("certificate_threshold"), float("nan"))
    if math.isfinite(certificate_margin_value):
        scale = max(0.25, certificate_threshold if math.isfinite(certificate_threshold) else 1.0)
        signals.append(min(1.0, max(0.0, -certificate_margin_value) / scale))
    if row.get("selector_certificate_disagreement") is not None:
        signals.append(1.0 if bool(row.get("selector_certificate_disagreement")) else 0.0)
    action_count = as_float(row.get("voi_action_count"), float("nan"))
    if math.isfinite(action_count):
        signals.append(min(1.0, max(0.0, action_count) / 4.0))
    pending_challenger_mass = as_float(row.get("pending_challenger_mass"), float("nan"))
    if math.isfinite(pending_challenger_mass):
        signals.append(min(1.0, max(0.0, pending_challenger_mass)))
    best_pending_flip_probability = as_float(row.get("best_pending_flip_probability"), float("nan"))
    if math.isfinite(best_pending_flip_probability):
        signals.append(min(1.0, max(0.0, best_pending_flip_probability)))
    search_completeness_gap = as_float(row.get("search_completeness_gap"), float("nan"))
    if math.isfinite(search_completeness_gap):
        signals.append(min(1.0, max(0.0, search_completeness_gap)))
    prior_support_strength = as_float(row.get("prior_support_strength"), float("nan"))
    if math.isfinite(prior_support_strength):
        signals.append(min(1.0, max(0.0, prior_support_strength)))
    if not signals:
        return None
    return round(sum(signals) / len(signals), 6)


def _mean_bool_with_denominator(rows: list[dict[str, Any]], key: str) -> tuple[float | None, int]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return (
        round(sum(1.0 if bool(value) else 0.0 for value in values) / len(values), 6),
        len(values),
    ) if values else (None, 0)


def _validate_csv_header(path: Path, *, expected_fields: list[str], artifact_name: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"strict_output_artifact_missing:{artifact_name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    if list(header) != list(expected_fields):
        raise RuntimeError(f"strict_output_artifact_invalid:{artifact_name}")


def _validate_json_artifact_file(path: Path, *, artifact_name: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"strict_output_artifact_missing:{artifact_name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, (dict, list)):
        raise RuntimeError(f"strict_output_artifact_invalid:{artifact_name}")


def _validate_text_artifact_file(path: Path, *, artifact_name: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"strict_output_artifact_missing:{artifact_name}")
    if not path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"strict_output_artifact_invalid:{artifact_name}")


def _validate_binary_artifact_file(path: Path, *, artifact_name: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"strict_output_artifact_missing:{artifact_name}")


def _validate_written_output_artifacts(
    *,
    results_csv: Path,
    summary_csv: Path,
    methods_path: Path,
    thesis_report_path: Path | None,
    evaluation_manifest_path: Path,
    manifest_path: Path,
    extra_json_paths: dict[str, Path],
    extra_text_paths: dict[str, Path],
    optional_paths: dict[str, Path | None],
) -> dict[str, Any]:
    _validate_csv_header(results_csv, expected_fields=RESULT_FIELDS, artifact_name="thesis_results.csv")
    _validate_csv_header(summary_csv, expected_fields=SUMMARY_FIELDS, artifact_name="thesis_summary.csv")
    _validate_text_artifact_file(methods_path, artifact_name="methods_appendix.md")
    if thesis_report_path is not None:
        _validate_text_artifact_file(thesis_report_path, artifact_name="thesis_report.md")
    _validate_json_artifact_file(evaluation_manifest_path, artifact_name="evaluation_manifest.json")
    _validate_json_artifact_file(manifest_path, artifact_name="manifest.json")
    for artifact_name, path in extra_json_paths.items():
        _validate_json_artifact_file(path, artifact_name=artifact_name)
    for artifact_name, path in extra_text_paths.items():
        _validate_text_artifact_file(path, artifact_name=artifact_name)
    validated: dict[str, dict[str, Any]] = {}
    for artifact_name, path in {
        "thesis_results.csv": results_csv,
        "thesis_summary.csv": summary_csv,
        "methods_appendix.md": methods_path,
        "evaluation_manifest.json": evaluation_manifest_path,
        "manifest.json": manifest_path,
        **extra_json_paths,
        **extra_text_paths,
    }.items():
        validated[artifact_name] = {"path": str(path), "size_bytes": int(path.stat().st_size)}
    if thesis_report_path is not None:
        validated["thesis_report.md"] = {
            "path": str(thesis_report_path),
            "size_bytes": int(thesis_report_path.stat().st_size),
        }
    for artifact_name, path in optional_paths.items():
        if path is None:
            continue
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"strict_output_artifact_missing:{artifact_name}")
        validated[artifact_name] = {"path": str(path), "size_bytes": int(path.stat().st_size)}
    return {"validated_artifact_count": len(validated), "artifacts": validated}


def _baseline_payload(
    args: argparse.Namespace,
    od: dict[str, Any],
    *,
    request_config: dict[str, Any] | None = None,
    variant_seed: int | None = None,
) -> dict[str, Any]:
    seed = int(args.seed) if variant_seed is None else int(variant_seed)
    return _base_payload(args, od, variant_seed=seed, request_config=request_config)


def _fetch_baseline(client: httpx.Client, url: str, payload: dict[str, Any], *, default_method: str, provider_mode: str = "live", snapshot_used: bool = False) -> BaselineResult:
    response, compute_ms = _post_json(client, url, payload)
    route = _baseline_route(response, default_method)
    method = str(response.get("method") or default_method)
    engine_manifest = response.get("engine_manifest")
    return BaselineResult(
        route=route,
        metrics=route_metrics(route),
        method=method,
        compute_ms=compute_ms,
        snapshot_used=snapshot_used,
        provider_mode=str(response.get("provider_mode") or provider_mode),
        baseline_policy=str(response.get("baseline_policy") or ""),
        asset_manifest_hash=str(response.get("asset_manifest_hash") or "") or None,
        asset_recorded_at=str(response.get("asset_recorded_at") or "") or None,
        asset_freshness_status=str(response.get("asset_freshness_status") or "") or None,
        engine_manifest=dict(engine_manifest) if isinstance(engine_manifest, dict) else None,
    )


def _load_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "1.0.0", "recorded_at": _now(), "routes": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ORS snapshot must be a JSON object")
    payload.setdefault("schema_version", "1.0.0")
    payload.setdefault("recorded_at", _now())
    payload.setdefault("routes", {})
    return payload


def _snapshot_route_key(*, od_id: str, request_hash: str) -> str:
    return f"{str(od_id).strip()}::{str(request_hash).strip()}"


def _snapshot_entry_for_request(
    snapshot_bundle: dict[str, Any],
    *,
    od_id: str,
    request_hash: str,
) -> dict[str, Any] | None:
    routes = snapshot_bundle.get("routes", {})
    if not isinstance(routes, dict):
        return None
    preferred_key = _snapshot_route_key(od_id=od_id, request_hash=request_hash)
    preferred = routes.get(preferred_key)
    if isinstance(preferred, dict):
        return preferred
    legacy = routes.get(od_id)
    if isinstance(legacy, dict):
        return legacy
    for entry in routes.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("od_id") or "").strip() == str(od_id).strip() and str(entry.get("request_hash") or "").strip() == str(request_hash).strip():
            return entry
    return None


def _ors_baseline(client: httpx.Client, base_url: str, payload: dict[str, Any], *, od_id: str, snapshot_mode: str, snapshot_bundle: dict[str, Any] | None, baseline_policy: str) -> tuple[BaselineResult, dict[str, Any] | None]:
    request_hash = _digest(payload)
    effective_policy = str(baseline_policy or "local_service").strip().lower()
    effective_snapshot_mode = str(snapshot_mode or "off").strip().lower()
    requested_provider_mode = "repo_local" if effective_policy == "repo_local" else "local_service"
    if effective_policy == "snapshot_replay" or effective_snapshot_mode == "replay":
        if snapshot_bundle is None:
            raise ValueError("ORS snapshot replay requested without a snapshot bundle")
        entry = _snapshot_entry_for_request(snapshot_bundle, od_id=od_id, request_hash=request_hash)
        if not isinstance(entry, dict):
            raise ValueError(f"ORS snapshot missing OD {od_id}")
        _validate_snapshot_entry(entry, od_id=od_id, request_hash=request_hash)
        response = dict(entry.get("response") or {})
        route = _baseline_route(response, "ors_snapshot")
        return (
            BaselineResult(
                route=route,
                metrics=route_metrics(route),
                method=str(entry.get("provider_method") or response.get("method") or "ors_snapshot"),
                compute_ms=as_float(entry.get("compute_ms")),
                snapshot_used=True,
                provider_mode=str(entry.get("provider_mode") or "snapshot_replay"),
                baseline_policy=str(entry.get("baseline_policy") or "snapshot_replay"),
                asset_manifest_hash=str(entry.get("asset_manifest_hash") or "") or None,
                asset_recorded_at=str(entry.get("asset_recorded_at") or "") or None,
                asset_freshness_status=str(entry.get("asset_freshness_status") or "") or None,
                engine_manifest=dict(response.get("engine_manifest") or entry.get("engine_manifest") or {})
                if isinstance(response.get("engine_manifest") or entry.get("engine_manifest"), dict)
                else None,
                snapshot_recorded_at=str(entry.get("recorded_at") or ""),
                snapshot_request_hash=str(entry.get("request_hash") or ""),
                snapshot_response_hash=str(entry.get("response_hash") or ""),
            ),
            snapshot_bundle,
        )
    default_method = "ors_local_engine_baseline" if requested_provider_mode == "local_service" else "ors_repo_local_baseline"
    live = _fetch_baseline(
        client,
        _absolute_url(base_url, f"/route/baseline/ors?realism=false&policy={requested_provider_mode}"),
        payload,
        default_method=default_method,
        provider_mode=requested_provider_mode,
    )
    if "proxy" in str(live.method).lower() or "proxy" in str(live.provider_mode).lower():
        raise ValueError("ORS baseline returned a proxy fallback, which is not allowed in strict thesis evaluation.")
    if str(live.provider_mode or "").strip().lower() != requested_provider_mode:
        raise ValueError(
            f"ORS baseline provider mismatch: requested={requested_provider_mode} returned={live.provider_mode}"
        )
    if requested_provider_mode == "local_service":
        if not str(live.asset_manifest_hash or "").strip() or str(live.asset_manifest_hash).startswith("ors-local:"):
            raise ValueError("ORS local-service baseline is missing a strict graph provenance manifest hash.")
        engine_manifest = live.engine_manifest if isinstance(live.engine_manifest, dict) else None
        if engine_manifest is None:
            raise ValueError("ORS local-service baseline is missing engine provenance metadata.")
        identity_status = str(
            engine_manifest.get("identity_status") or live.asset_freshness_status or ""
        ).strip()
        if identity_status != "graph_identity_verified":
            raise ValueError(f"ORS local-service graph provenance is not verified: {identity_status or 'unknown'}")
        if not str(engine_manifest.get("graph_listing_digest") or "").strip():
            raise ValueError("ORS local-service baseline is missing graph listing provenance.")
        if int(as_float(engine_manifest.get("graph_file_count"), 0.0)) <= 0:
            raise ValueError("ORS local-service baseline graph directory is empty.")
    if (effective_policy == "snapshot_record" or effective_snapshot_mode == "record") and snapshot_bundle is not None:
        snapshot_bundle.setdefault("routes", {})[_snapshot_route_key(od_id=od_id, request_hash=request_hash)] = {
            "od_id": od_id,
            "request_hash": request_hash,
            "recorded_at": _now(),
            "provider_method": live.method,
            "provider_mode": live.provider_mode,
            "baseline_policy": live.baseline_policy,
            "asset_manifest_hash": live.asset_manifest_hash,
            "asset_recorded_at": live.asset_recorded_at,
            "asset_freshness_status": live.asset_freshness_status,
            "engine_manifest": live.engine_manifest,
            "compute_ms": live.compute_ms,
            "response_hash": _digest({"baseline": live.route, "method": live.method}),
            "response": {
                "baseline": live.route,
                "method": live.method,
                "provider_mode": live.provider_mode,
                "baseline_policy": live.baseline_policy,
                "asset_manifest_hash": live.asset_manifest_hash,
                "asset_recorded_at": live.asset_recorded_at,
                "asset_freshness_status": live.asset_freshness_status,
                "engine_manifest": live.engine_manifest,
            },
        }
    return live, snapshot_bundle


def _evidence_fallback_families(route: dict[str, Any]) -> list[str]:
    provenance = route.get("evidence_provenance")
    if not isinstance(provenance, dict):
        return []
    raw_families = provenance.get("families", [])
    if not isinstance(raw_families, list):
        return []
    fallbacks: list[str] = []
    for entry in raw_families:
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("family", "")).strip() or "unknown"
        source = str(entry.get("source", "")).strip().lower()
        fallback_source = str(entry.get("fallback_source", "")).strip().lower()
        if bool(entry.get("fallback_used")) or "proxy" in source or "proxy" in fallback_source:
            fallbacks.append(family)
    return sorted(set(fallbacks))


def _enforce_strict_thesis_inputs(
    *,
    args: argparse.Namespace,
    route_response: dict[str, Any],
    ors: BaselineResult,
) -> dict[str, Any]:
    ors_mode = str(ors.provider_mode or "").strip().lower()
    ors_method = str(ors.method or "").strip().lower()
    ors_policy = str(ors.baseline_policy or "").strip().lower()
    if not bool(args.allow_proxy_ors) and ("proxy" in ors_mode or "proxy" in ors_method or "proxy" in ors_policy):
        raise RuntimeError("strict_proxy_ors_baseline_forbidden")
    if "synthetic" in ors_mode or "synthetic" in ors_method or "synthetic" in ors_policy:
        raise RuntimeError("strict_synthetic_ors_baseline_forbidden")
    route_validation = _route_response_evidence_validation(route_response, allow_snapshot=True)
    if route_validation["status"] != "ok" and not bool(args.allow_evidence_fallbacks):
        raise RuntimeError(_strict_failure_reason_from_issues(route_validation["issues"]))
    return route_validation


def _result_row(
    args: argparse.Namespace,
    od: dict[str, Any],
    spec: VariantSpec,
    route_response: dict[str, Any],
    route_request_ms: float,
    artifacts: dict[str, Any],
    osrm: BaselineResult,
    ors: BaselineResult,
    *,
    readiness_summary: Mapping[str, Any] | None = None,
    request_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_config = request_config or _effective_request_config(args, od, variant_seed=int(args.seed))
    effective_weights = _weights_tuple_from_config(request_config)
    effective_search_budget = int(request_config.get("search_budget") or args.search_budget)
    effective_evidence_budget = int(request_config.get("evidence_budget") or args.evidence_budget)
    effective_certificate_threshold = float(
        request_config.get("certificate_threshold")
        if request_config.get("certificate_threshold") is not None
        else args.certificate_threshold
    )
    selected = dict(route_response.get("selected") or {})
    selected_metrics = route_metrics(selected)
    selected_id = str(selected.get("id") or selected.get("route_id") or "")
    candidates = route_response.get("candidates", [])
    evidence_validation = route_response.get("evidence_validation")
    if not isinstance(evidence_validation, dict):
        evidence_validation = {
            "status": "unknown",
            "issues": [],
            "policy": {"mode": STRICT_EVIDENCE_POLICY, "allow_snapshot": True, "require_freshness": True},
        }
    artifact_validation = route_response.get("artifact_validation")
    if not isinstance(artifact_validation, dict):
        artifact_validation = {
            "status": "unknown",
            "required": list(_required_artifacts_for_pipeline(spec.pipeline_mode)),
            "missing": [],
        }
    dccs_summary = artifacts.get("dccs_summary.json", {})
    certificate_summary = artifacts.get("certificate_summary.json", {})
    initial_certificate_summary = artifacts.get("initial_certificate_summary.json", {})
    route_fragility_map = artifacts.get("route_fragility_map.json", {})
    initial_route_fragility_map = artifacts.get("initial_route_fragility_map.json", {})
    competitor_fragility_breakdown = artifacts.get("competitor_fragility_breakdown.json", {})
    value_of_refresh = artifacts.get("value_of_refresh.json", {})
    initial_value_of_refresh = artifacts.get("initial_value_of_refresh.json", {})
    world_manifest = artifacts.get("sampled_world_manifest.json", {})
    voi_trace = artifacts.get("voi_action_trace.json", {})
    voi_stop = artifacts.get("voi_stop_certificate.json", {})
    trace = artifacts.get("final_route_trace.json", {})
    controller_rows = _controller_state_rows(artifacts)
    trace_candidate_diag = trace.get("candidate_diagnostics", {}) if isinstance(trace, dict) else {}
    trace_resource_usage = trace.get("resource_usage", {}) if isinstance(trace, dict) else {}
    trace_route_cache_runtime = trace.get("route_cache_runtime", {}) if isinstance(trace, dict) else {}
    trace_route_cache = trace.get("route_cache_stats", {}) if isinstance(trace, dict) else {}
    trace_route_state_cache = trace.get("route_state_cache_stats", {}) if isinstance(trace, dict) else {}
    trace_diversity_rescue = trace.get("diversity_rescue", {}) if isinstance(trace, dict) else {}
    trace_cert_runtime = trace.get("certification_runtime", {}) if isinstance(trace, dict) else {}
    trace_option_build_runtime = trace.get("option_build_runtime", {}) if isinstance(trace, dict) else {}
    trace_route_option_cache_runtime = trace.get("route_option_cache_runtime", {}) if isinstance(trace, dict) else {}
    readiness = dict(readiness_summary or {})
    readiness_route_graph = readiness.get("route_graph", {}) if isinstance(readiness, Mapping) else {}
    frontier_rows = [_normalize_frontier_row(row) for row in artifacts.get(FRONTIER_ARTIFACT, []) if isinstance(row, dict)]
    reference_point = _reference_point(frontier_rows, [osrm.metrics, ors.metrics])
    frontier_hypervolume = hypervolume_3d(frontier_rows, reference=reference_point)
    frontier_spread, frontier_crowding = frontier_diversity(frontier_rows)
    frontier_diversity_idx = frontier_diversity_index(frontier_rows)
    frontier_ent = frontier_entropy(frontier_rows)
    frontier_efficiency_gain = frontier_action_gain(
        frontier_count=float(len(frontier_rows)),
        frontier_diversity_index=frontier_diversity_idx,
    )
    candidate_rows = artifacts.get("dccs_candidates.jsonl", [])
    candidate_row_by_id = {
        str(row.get("candidate_id") or "").strip(): row
        for row in candidate_rows
        if isinstance(row, Mapping) and str(row.get("candidate_id") or "").strip()
    }
    selected_candidate_ids = []
    if isinstance(trace, Mapping) and isinstance(trace.get("selected_candidate_ids"), list):
        selected_candidate_ids = [str(item) for item in trace.get("selected_candidate_ids", []) if str(item).strip()]
    elif isinstance(dccs_summary, Mapping) and isinstance(dccs_summary.get("selected_candidate_ids"), list):
        selected_candidate_ids = [str(item) for item in dccs_summary.get("selected_candidate_ids", []) if str(item).strip()]
    selected_primary_candidate = next(
        (
            candidate_row_by_id.get(candidate_id)
            for candidate_id in selected_candidate_ids
            if candidate_row_by_id.get(candidate_id) is not None
        ),
        {},
    )
    top_refresh_family, top_vor, top_vor_gap = _top_refresh_stats(value_of_refresh)
    (
        controller_refresh_ranking_basis,
        controller_top_refresh_family,
        controller_top_refresh_gain,
        controller_refresh_fallback_activated,
        controller_empirical_vs_raw_refresh_disagreement,
    ) = _controller_refresh_stats(value_of_refresh)
    initial_top_refresh_family, initial_top_vor, initial_top_vor_gap = _top_refresh_stats(
        initial_value_of_refresh,
    )
    action_count, refine_count, refresh_count, resample_count = _action_counts(voi_trace)
    voi_entries = _voi_action_entries(
        voi_trace if isinstance(voi_trace, Mapping) else {},
        voi_stop if isinstance(voi_stop, Mapping) else {},
    )
    productive_action_count, productive_action_denominator, nonproductive_refine_count = _voi_action_productivity(
        voi_entries,
    )
    cert_map = _certificate_map(certificate_summary)
    certificate_winner_id = _certificate_winner_id(cert_map)
    requested_world_count = int(world_manifest.get("requested_world_count") or world_manifest.get("world_count") or 0) if world_manifest else 0
    effective_world_count = int(world_manifest.get("effective_world_count") or world_manifest.get("world_count") or 0) if world_manifest else 0
    selected_cert = route_response.get("selected_certificate")
    if isinstance(selected_cert, dict):
        certificate = as_float(selected_cert.get("certificate"), float("nan"))
        certified = bool(selected_cert.get("certified"))
    else:
        certificate = as_float(certificate_summary.get("selected_certificate"), float("nan"))
        certified = bool(certificate_summary.get("certified")) if certificate_summary else None
    win_selected = {k: selected_metrics[k] for k in ("duration_s", "monetary_cost", "emissions_kg")}
    weighted_osrm, weighted_osrm_base = pairwise_weighted_sum_score(
        win_selected,
        osrm.metrics,
        weights=effective_weights,
    )
    weighted_ors, weighted_ors_base = pairwise_weighted_sum_score(
        win_selected,
        ors.metrics,
        weights=effective_weights,
    )
    if weighted_osrm_base <= weighted_ors_base:
        best_baseline_provider = "osrm"
        best_baseline_metrics = osrm.metrics
        best_baseline_weighted_base = weighted_osrm_base
    else:
        best_baseline_provider = "ors"
        best_baseline_metrics = ors.metrics
        best_baseline_weighted_base = weighted_ors_base
    weighted_margin_osrm = round(weighted_osrm_base - weighted_osrm, 6)
    weighted_margin_ors = round(weighted_ors_base - weighted_ors, 6)
    weighted_margin_best_baseline = round(best_baseline_weighted_base - min(weighted_osrm, weighted_ors), 6)
    balanced_gain_osrm = round(balanced_gain_score(win_selected, osrm.metrics), 6)
    balanced_gain_ors = round(balanced_gain_score(win_selected, ors.metrics), 6)
    balanced_gain_best_baseline = round(balanced_gain_score(win_selected, best_baseline_metrics), 6)
    iteration_count = int(voi_stop.get("iteration_count") or action_count or 0)
    search_budget_used = int(voi_stop.get("search_budget_used") or dccs_summary.get("search_budget_used") or dccs_summary.get("refined_count") or 0)
    evidence_budget_used = int(voi_stop.get("evidence_budget_used") or 0)
    refined_count = int(dccs_summary.get("refined_count") or 0)
    search_budget_utilization = round(search_budget_used / max(1, effective_search_budget), 6)
    evidence_budget_utilization = round(evidence_budget_used / max(1, effective_evidence_budget), 6)
    baseline_acquisition_runtime_ms = round(osrm.compute_ms + ors.compute_ms, 3)
    algorithm_runtime_ms = round(max(0.0, route_request_ms), 3)
    runtime_ms = round(route_request_ms + osrm.compute_ms + ors.compute_ms, 3)
    runtime_gap_vs_osrm_ms = round(runtime_ms - osrm.compute_ms, 3)
    runtime_gap_vs_ors_ms = round(runtime_ms - ors.compute_ms, 3)
    algorithm_runtime_gap_vs_osrm_ms = round(algorithm_runtime_ms - osrm.compute_ms, 3)
    algorithm_runtime_gap_vs_ors_ms = round(algorithm_runtime_ms - ors.compute_ms, 3)
    route_cache_hits = (
        int(as_float((trace_route_cache_runtime or {}).get("cache_hits"), float("nan")))
        if isinstance(trace_route_cache_runtime, Mapping)
        and math.isfinite(as_float((trace_route_cache_runtime or {}).get("cache_hits"), float("nan")))
        else int(as_float((trace_route_cache or {}).get("hits"), 0.0))
        if isinstance(trace_route_cache, Mapping)
        else None
    )
    route_cache_misses = (
        int(as_float((trace_route_cache_runtime or {}).get("cache_misses"), float("nan")))
        if isinstance(trace_route_cache_runtime, Mapping)
        and math.isfinite(as_float((trace_route_cache_runtime or {}).get("cache_misses"), float("nan")))
        else int(as_float((trace_route_cache or {}).get("misses"), 0.0))
        if isinstance(trace_route_cache, Mapping)
        else None
    )
    route_cache_hit_rate = runtime_share(route_cache_hits, (route_cache_hits or 0) + (route_cache_misses or 0))
    route_state_cache_hits = (
        int(as_float((trace_option_build_runtime or {}).get("cache_hits_global"), float("nan")))
        if isinstance(trace_option_build_runtime, Mapping)
        and math.isfinite(as_float((trace_option_build_runtime or {}).get("cache_hits_global"), float("nan")))
        else int(as_float((trace_option_build_runtime or {}).get("cache_hits"), 0.0))
        if isinstance(trace_option_build_runtime, Mapping)
        else int(as_float((trace_route_state_cache or {}).get("hits"), 0.0))
        if isinstance(trace_route_state_cache, Mapping)
        else 0
    )
    route_state_cache_misses = (
        int(as_float((trace_option_build_runtime or {}).get("cache_misses"), 0.0))
        if isinstance(trace_option_build_runtime, Mapping)
        else int(as_float((trace_route_state_cache or {}).get("misses"), 0.0))
        if isinstance(trace_route_state_cache, Mapping)
        else 0
    )
    route_state_cache_hit_rate = runtime_share(
        route_state_cache_hits,
        route_state_cache_hits + route_state_cache_misses,
    )
    diversity_collapse = bool(
        dccs_summary.get("diversity_collapse_detected")
        or (trace_diversity_rescue.get("collapse_detected") if isinstance(trace_diversity_rescue, Mapping) else False)
    )
    supplemental_sources = (
        [
            str(item)
            for item in trace_diversity_rescue.get("supplemental_sources", [])
            if str(item).strip()
        ]
        if isinstance(trace_diversity_rescue, Mapping) and isinstance(trace_diversity_rescue.get("supplemental_sources"), list)
        else []
    )
    selected_candidate_source_label = str(
        (dccs_summary.get("selected_candidate_source_label") if isinstance(dccs_summary, Mapping) else None)
        or (selected_primary_candidate.get("candidate_source_label") if isinstance(selected_primary_candidate, Mapping) else None)
        or ""
    ).strip() or None
    selected_candidate_source_engine = str(
        (dccs_summary.get("selected_candidate_source_engine") if isinstance(dccs_summary, Mapping) else None)
        or (selected_primary_candidate.get("candidate_source_engine") if isinstance(selected_primary_candidate, Mapping) else None)
        or ""
    ).strip() or None
    selected_candidate_source_stage = str(
        (dccs_summary.get("selected_candidate_source_stage") if isinstance(dccs_summary, Mapping) else None)
        or (selected_primary_candidate.get("candidate_source_stage") if isinstance(selected_primary_candidate, Mapping) else None)
        or ""
    ).strip() or None
    selected_final_route_source_label = str(
        (dccs_summary.get("selected_final_route_source_label") if isinstance(dccs_summary, Mapping) else None)
        or selected_candidate_source_label
        or ""
    ).strip() or None
    selected_final_route_source_engine = str(
        (dccs_summary.get("selected_final_route_source_engine") if isinstance(dccs_summary, Mapping) else None)
        or selected_candidate_source_engine
        or ""
    ).strip() or None
    selected_final_route_source_stage = str(
        (dccs_summary.get("selected_final_route_source_stage") if isinstance(dccs_summary, Mapping) else None)
        or selected_candidate_source_stage
        or ""
    ).strip() or None
    selected_from_supplemental_rescue = bool(
        (dccs_summary.get("selected_from_supplemental_rescue") if isinstance(dccs_summary, Mapping) else False)
        or (selected_primary_candidate.get("supplemental_diversity_rescue") if isinstance(selected_primary_candidate, Mapping) else False)
        or selected_final_route_source_stage == "supplemental_diversity_rescue"
    )
    selected_from_comparator_engine = bool(
        selected_final_route_source_engine in {"osrm", "ors_local", "ors_local_seed"}
    )
    selected_from_preemptive_comparator_seed = bool(
        (dccs_summary.get("selected_from_preemptive_comparator_seed") if isinstance(dccs_summary, Mapping) else False)
        or selected_final_route_source_stage == "preemptive_comparator_seed"
    )
    preemptive_comparator_seeded = bool(
        as_float((trace_candidate_diag or {}).get("preemptive_comparator_seed_activated"), 0.0) > 0.0
    ) if isinstance(trace_candidate_diag, Mapping) else False
    preemptive_comparator_candidate_count = int(
        as_float((trace_candidate_diag or {}).get("preemptive_comparator_candidate_count"), 0.0)
    ) if isinstance(trace_candidate_diag, Mapping) else 0
    preemptive_comparator_source_count = int(
        as_float((trace_candidate_diag or {}).get("preemptive_comparator_source_count"), 0.0)
    ) if isinstance(trace_candidate_diag, Mapping) else 0
    graph_k_raw_cache_hit = bool(
        isinstance(trace_candidate_diag, Mapping)
        and bool((trace_candidate_diag or {}).get("graph_k_raw_cache_hit"))
    )
    graph_low_ambiguity_fast_path = bool(
        isinstance(trace_candidate_diag, Mapping)
        and bool((trace_candidate_diag or {}).get("graph_low_ambiguity_fast_path"))
    )
    graph_supported_ambiguity_fast_fallback = bool(
        isinstance(trace_candidate_diag, Mapping)
        and bool((trace_candidate_diag or {}).get("graph_supported_ambiguity_fast_fallback"))
    )
    winner_margin, _ = nominal_winner_margin(frontier_rows, weights=effective_weights)
    initial_selected_id = (
        str(initial_certificate_summary.get("selected_route_id") or "").strip()
        if isinstance(initial_certificate_summary, Mapping)
        else ""
    ) or selected_id
    selected_fragility = route_fragility_map.get(selected_id, {}) if isinstance(route_fragility_map, dict) else {}
    selected_competitors = competitor_fragility_breakdown.get(selected_id, {}) if isinstance(competitor_fragility_breakdown, dict) else {}
    initial_refc_top_fragility_family = (
        _top_fragility(initial_route_fragility_map, initial_selected_id)
        if initial_route_fragility_map
        else None
    )
    final_refc_top_fragility_family = _top_fragility(route_fragility_map, selected_id) if route_fragility_map else None
    initial_winner_fragility_mass = (
        _top_fragility_value(initial_route_fragility_map, initial_selected_id)
        if initial_route_fragility_map
        else None
    )
    final_winner_fragility_mass = (
        _top_fragility_value(route_fragility_map, selected_id)
        if route_fragility_map
        else None
    )
    initial_winner_fragility_nonzero = (
        initial_winner_fragility_mass is not None and initial_winner_fragility_mass > 1e-9
    )
    winner_fragility_nonzero = (
        final_winner_fragility_mass is not None and final_winner_fragility_mass > 1e-9
    )
    initial_refc_top_vor_positive = initial_top_vor is not None and initial_top_vor > 1e-9
    refc_top_vor_positive = top_vor is not None and top_vor > 1e-9
    dccs_positive_labels = {"frontier_addition", "decision_flip", "challenger_but_not_added"}
    voi_actions = voi_trace.get("actions", []) if isinstance(voi_trace, dict) else []
    time_to_best = time_to_best_iteration(
        voi_actions,
        selected_route_id=selected_id,
    )
    valid_refine_cost_rows = _result_refine_cost_rows(
        candidate_rows,
        pipeline_mode=spec.pipeline_mode,
        voi_entries=voi_entries,
    )
    refine_sample_count = refine_cost_sample_count(valid_refine_cost_rows) if valid_refine_cost_rows else 0
    refine_positive_sample_count = refine_cost_positive_sample_count(valid_refine_cost_rows) if valid_refine_cost_rows else 0
    refine_zero_observed_count = refine_cost_zero_observed_count(valid_refine_cost_rows) if valid_refine_cost_rows else 0
    refine_mape = refine_cost_mape(valid_refine_cost_rows) if valid_refine_cost_rows else None
    refine_mae_value = refine_cost_mae_ms(valid_refine_cost_rows) if valid_refine_cost_rows else None
    refine_rank_corr = refine_cost_rank_correlation(valid_refine_cost_rows) if valid_refine_cost_rows else None
    option_build_reuse = _option_build_reuse_rate(
        trace if isinstance(trace, Mapping) else {},
        trace_candidate_diag if isinstance(trace_candidate_diag, Mapping) else {},
    )
    option_build_cache_runtime = _option_build_cache_runtime(trace if isinstance(trace, Mapping) else {})
    option_build_cache_hits = int(as_float((option_build_cache_runtime or {}).get("cache_hits"), 0.0)) if isinstance(option_build_cache_runtime, Mapping) else 0
    option_build_rebuild_count = (
        int(as_float((trace_option_build_runtime or {}).get("rebuild_count"), float("nan")))
        if isinstance(trace_option_build_runtime, Mapping)
        and math.isfinite(as_float((trace_option_build_runtime or {}).get("rebuild_count"), float("nan")))
        else int(as_float((option_build_cache_runtime or {}).get("cache_misses"), 0.0))
        if isinstance(option_build_cache_runtime, Mapping)
        else 0
    )
    option_build_cache_hit_rate = runtime_share(
        option_build_cache_hits,
        option_build_cache_hits
        + (
            int(as_float((option_build_cache_runtime or {}).get("cache_misses"), 0.0))
            if isinstance(option_build_cache_runtime, Mapping)
            else 0
        ),
    )
    option_build_cache_savings_ms_per_row = (
        round(
            max(0.0, as_float((option_build_cache_runtime or {}).get("saved_ms_estimate"), 0.0)),
            6,
        )
        if isinstance(option_build_cache_runtime, Mapping)
        and math.isfinite(as_float((option_build_cache_runtime or {}).get("saved_ms_estimate"), float("nan")))
        else None
    )
    action_eff = action_efficiency(
        certificate_lift=as_float(
            certificate_margin(
                certificate if math.isfinite(certificate) else None,
                threshold=effective_certificate_threshold,
            ),
            float("nan"),
        ),
        frontier_gain=frontier_efficiency_gain,
        action_count=action_count,
        search_budget_used=search_budget_used,
        evidence_budget_used=evidence_budget_used,
    )
    certificate_margin_value = certificate_margin(
        certificate if math.isfinite(certificate) else None,
        threshold=effective_certificate_threshold,
    )
    controller_vps = value_per_second(action_eff, algorithm_runtime_ms)
    stage_voi_value = _stage_value(trace, "voi_ms")
    stage_k_raw_value = _stage_value(trace, "k_raw_ms")
    stage_k_raw_graph_search_initial_value = (
        as_float((trace_candidate_diag or {}).get("graph_search_ms_initial"), float("nan"))
        if isinstance(trace_candidate_diag, Mapping)
        else float("nan")
    )
    stage_k_raw_graph_search_retry_value = (
        as_float((trace_candidate_diag or {}).get("graph_search_ms_retry"), float("nan"))
        if isinstance(trace_candidate_diag, Mapping)
        else float("nan")
    )
    stage_k_raw_graph_search_rescue_value = (
        as_float((trace_candidate_diag or {}).get("graph_search_ms_rescue"), float("nan"))
        if isinstance(trace_candidate_diag, Mapping)
        else float("nan")
    )
    stage_k_raw_graph_search_supplemental_value = (
        as_float((trace_candidate_diag or {}).get("graph_search_ms_supplemental"), float("nan"))
        if isinstance(trace_candidate_diag, Mapping)
        else float("nan")
    )
    stage_k_raw_osrm_fallback_value = as_float(
        _stage_value_any(trace, "k_raw_osrm_fallback_ms"),
        float("nan"),
    )
    realized_stage_k_raw_value = as_float(stage_k_raw_value, float("nan"))
    suppress_k_raw_substages = bool(
        graph_k_raw_cache_hit
        or (
            math.isfinite(realized_stage_k_raw_value)
            and realized_stage_k_raw_value <= 1.0
            and route_state_cache_hit_rate is not None
            and route_state_cache_hit_rate >= 1.0 - 1e-9
        )
    )
    if suppress_k_raw_substages:
        # Cached rows should report realized retrieval overhead in `stage_k_raw_ms`,
        # but must not inherit the original graph-search/fallback substage timings.
        stage_k_raw_graph_search_initial_value = 0.0
        stage_k_raw_graph_search_retry_value = 0.0
        stage_k_raw_graph_search_rescue_value = 0.0
        stage_k_raw_graph_search_supplemental_value = 0.0
        stage_k_raw_osrm_fallback_value = 0.0
    if not math.isfinite(realized_stage_k_raw_value):
        stage_k_raw_components = [
            value
            for value in (
                stage_k_raw_graph_search_initial_value,
                stage_k_raw_graph_search_retry_value,
                stage_k_raw_graph_search_rescue_value,
                stage_k_raw_graph_search_supplemental_value,
                stage_k_raw_osrm_fallback_value,
            )
            if math.isfinite(value)
        ]
        stage_k_raw_value = round(sum(stage_k_raw_components), 6) if stage_k_raw_components else None
    stage_option_build_value = as_float(_stage_value_any(trace, "option_build_ms", "build_options_ms"), float("nan"))
    fully_reused_option_build = bool(
        option_build_cache_hits > 0
        and option_build_rebuild_count <= 0
        and (
            (option_build_reuse is not None and float(option_build_reuse) >= 0.999999)
            or (option_build_cache_hit_rate is not None and float(option_build_cache_hit_rate) >= 0.999999)
        )
    )
    if fully_reused_option_build:
        stage_option_build_value = 0.0
    if option_build_cache_savings_ms_per_row is None and math.isfinite(stage_option_build_value):
        reuse_signal = option_build_reuse
        if reuse_signal is None:
            reuse_signal = option_build_cache_hit_rate
        reuse_value = as_float(reuse_signal, float("nan"))
        if math.isfinite(reuse_value):
            option_build_cache_savings_ms_per_row = round(
                max(0.0, stage_option_build_value) * max(0.0, min(1.0, reuse_value)),
                6,
            )
    initial_certificate_value = _initial_certificate(
        voi_entries=voi_entries,
        certificate=certificate if math.isfinite(certificate) else None,
        threshold=effective_certificate_threshold,
        stop_reason=str(voi_stop.get("stop_reason") or "").strip() if isinstance(voi_stop, Mapping) else None,
    )
    initial_certificate_stop = bool(
        spec.pipeline_mode == "voi"
        and action_count == 0
        and str(voi_stop.get("stop_reason") or "").strip() == "certified"
        and initial_certificate_value is not None
        and initial_certificate_value >= effective_certificate_threshold
    )
    unnecessary_voi_refine = bool(
        spec.pipeline_mode == "voi"
        and nonproductive_refine_count > 0
    )
    voi_stop_after_certification = bool(
        spec.pipeline_mode == "voi"
        and certified
        and not any(
            (_action_snapshot_certificate(entry) or -1.0) >= effective_certificate_threshold
            and _chosen_action_kind(entry) not in {"", "stop"}
            for entry in voi_entries
        )
    ) if certified is not None else None
    controller_shortcut = bool(spec.pipeline_mode == "voi" and initial_certificate_stop)
    route_graph_warmup_ms = _route_graph_startup_to_ready_ms(readiness if isinstance(readiness, Mapping) else None)
    search_completeness_score = _controller_signal(
        controller_rows,
        key="search_completeness_score",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    search_completeness_gap = _controller_signal(
        controller_rows,
        key="search_completeness_gap",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    prior_support_strength = _controller_signal(
        controller_rows,
        key="prior_support_strength",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    support_richness = _controller_signal(
        controller_rows,
        key="support_richness",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    ambiguity_pressure = _controller_signal(
        controller_rows,
        key="ambiguity_pressure",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    if support_richness is None:
        support_richness = _fallback_support_richness(od)
    if ambiguity_pressure is None:
        ambiguity_pressure = _fallback_ambiguity_pressure(od)
    pending_challenger_mass = _controller_signal(
        controller_rows,
        key="pending_challenger_mass",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    best_pending_flip_probability = _controller_signal(
        controller_rows,
        key="best_pending_flip_probability",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    corridor_family_recall_value = _controller_signal(
        controller_rows,
        key="corridor_family_recall",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    frontier_recall_at_budget_value = _controller_signal(
        controller_rows,
        key="frontier_recall_at_budget",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    top_refresh_gain_value = _controller_signal(
        controller_rows,
        key="top_refresh_gain",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    top_fragility_mass_value = _controller_signal(
        controller_rows,
        key="top_fragility_mass",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    competitor_pressure_value = _controller_signal(
        controller_rows,
        key="competitor_pressure",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    credible_search_uncertainty_value = _controller_flag(
        controller_rows,
        key="credible_search_uncertainty",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    credible_evidence_uncertainty_value = _controller_flag(
        controller_rows,
        key="credible_evidence_uncertainty",
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
    )
    time_to_certification = _time_to_certification_ms(
        voi_entries=voi_entries,
        voi_stop=voi_stop if isinstance(voi_stop, Mapping) else {},
        trace=trace if isinstance(trace, Mapping) else {},
        initial_certificate=initial_certificate_value,
        certificate=certificate if math.isfinite(certificate) else None,
        threshold=effective_certificate_threshold,
        stage_voi_ms=stage_voi_value,
    )
    row_local_warmup_value = None
    startup_components = _startup_components_ms(readiness if isinstance(readiness, Mapping) else None)
    global_startup_overhead_value = round(sum(startup_components), 6) if startup_components else None
    preflight_value = as_float((trace_candidate_diag or {}).get("precheck_elapsed_ms"), float("nan")) if isinstance(trace_candidate_diag, Mapping) else float("nan")
    preflight_and_warmup_ms = round(
        sum(
            component
            for component in [
                *startup_components,
                preflight_value,
            ]
            if math.isfinite(component)
        ),
        6,
    ) if any(math.isfinite(component) for component in [*startup_components, preflight_value]) else None
    ambiguity_prior = _od_ambiguity_prior(od)
    ambiguity_confidence = as_float(od.get("od_ambiguity_confidence"), float("nan")) if od.get("od_ambiguity_confidence") not in (None, "") else None
    ambiguity_source_count = int(as_float(od.get("od_ambiguity_source_count"), 0.0)) if od.get("od_ambiguity_source_count") not in (None, "") else None
    ambiguity_source_mix = str(od.get("od_ambiguity_source_mix") or "").strip() or None
    ambiguity_source_mix_count = int(as_float(od.get("od_ambiguity_source_mix_count"), 0.0)) if od.get("od_ambiguity_source_mix_count") not in (None, "") else None
    ambiguity_source_entropy = as_float(od.get("od_ambiguity_source_entropy"), float("nan")) if od.get("od_ambiguity_source_entropy") not in (None, "") else None
    ambiguity_support_ratio = as_float(od.get("od_ambiguity_support_ratio"), float("nan")) if od.get("od_ambiguity_support_ratio") not in (None, "") else None
    ambiguity_prior_strength = as_float(od.get("od_ambiguity_prior_strength"), float("nan")) if od.get("od_ambiguity_prior_strength") not in (None, "") else None
    ambiguity_budget_prior_value = (
        as_float(request_config.get("ambiguity_budget_prior"), float("nan"))
        if request_config.get("ambiguity_budget_prior") not in (None, "")
        else float("nan")
    )
    ambiguity_budget_prior = (
        round(ambiguity_budget_prior_value, 6)
        if math.isfinite(ambiguity_budget_prior_value)
        else None
    )
    raw_ambiguity_value = _shared_raw_ambiguity_prior_value(od)
    ambiguity_budget_prior_gap = (
        round(max(0.0, ambiguity_budget_prior_value - raw_ambiguity_value), 6)
        if math.isfinite(ambiguity_budget_prior_value) and raw_ambiguity_value is not None
        else None
    )
    ambiguity_family_density = as_float(od.get("od_ambiguity_family_density"), float("nan")) if od.get("od_ambiguity_family_density") not in (None, "") else None
    ambiguity_margin_pressure = as_float(od.get("od_ambiguity_margin_pressure"), float("nan")) if od.get("od_ambiguity_margin_pressure") not in (None, "") else None
    ambiguity_spread_pressure = as_float(od.get("od_ambiguity_spread_pressure"), float("nan")) if od.get("od_ambiguity_spread_pressure") not in (None, "") else None
    ambiguity_toll_instability = as_float(od.get("od_ambiguity_toll_instability"), float("nan")) if od.get("od_ambiguity_toll_instability") not in (None, "") else None
    first_controller_action_kind = _first_controller_action_kind(voi_entries)
    evidence_first_engagement = bool(spec.pipeline_mode == "voi" and _is_evidence_action(first_controller_action_kind))
    evidence_only_engagement = bool(
        spec.pipeline_mode == "voi"
        and any(_is_evidence_action(_chosen_action_kind(entry)) for entry in voi_entries)
        and not any(_is_refine_action(_chosen_action_kind(entry)) for entry in voi_entries)
    )
    refresh_signal_persistent = (
        bool(refc_top_vor_positive or winner_fragility_nonzero)
        if spec.pipeline_mode == "voi" and first_controller_action_kind == "refresh_top1_vor"
        else None
    )
    refresh_first_productive, refresh_resolution_honest, refresh_resolution_reason = _refresh_first_resolution_state(
        first_controller_action_kind=first_controller_action_kind,
        voi_entries=voi_entries,
        initial_winner_fragility_nonzero=initial_winner_fragility_nonzero,
        initial_refc_top_vor_positive=initial_refc_top_vor_positive,
        refresh_signal_persistent=refresh_signal_persistent,
    )
    supported_hard_case = bool(
        max(
            as_float(ambiguity_support_ratio, 0.0),
            as_float(ambiguity_budget_prior, as_float(ambiguity_prior_strength, 0.0)),
        ) >= 0.45
        and (
            (ambiguity_source_count or 0) >= 2
            or (ambiguity_source_mix_count or 0) >= 2
            or as_float(ambiguity_confidence, 0.0) >= 0.70
        )
    )
    voi_dccs_runtime = trace.get("voi_dccs_runtime", {}) if isinstance(trace, Mapping) else {}
    voi_dccs_cache_hits = int(as_float((voi_dccs_runtime or {}).get("cache_hits"), 0.0)) if isinstance(voi_dccs_runtime, Mapping) else 0
    voi_dccs_cache_misses = int(as_float((voi_dccs_runtime or {}).get("cache_misses"), 0.0)) if isinstance(voi_dccs_runtime, Mapping) else 0
    voi_dccs_cache_hit_rate = runtime_share(
        voi_dccs_cache_hits,
        voi_dccs_cache_hits + voi_dccs_cache_misses,
    )
    manifest_ambiguity_context = world_manifest.get("ambiguity_context", {}) if isinstance(world_manifest, Mapping) else {}
    selected_config = {
        "profile_id": request_config.get("profile_id"),
        "corpus_group": request_config.get("corpus_group"),
        "row_override_count": request_config.get("row_override_count"),
        "row_override_keys": request_config.get("row_override_keys"),
        "ambiguity_budget_band": request_config.get("ambiguity_budget_band"),
        "ambiguity_budget_prior": request_config.get("ambiguity_budget_prior"),
        "ambiguity_schedule_reason": request_config.get("ambiguity_schedule_reason"),
        "scenario_mode": request_config.get("scenario_mode"),
        "weather": request_config.get("weather"),
        "departure_time_utc": request_config.get("departure_time_utc"),
        "stochastic": request_config.get("stochastic"),
        "search_budget": request_config.get("search_budget"),
        "evidence_budget": request_config.get("evidence_budget"),
        "world_count": request_config.get("world_count"),
        "certificate_threshold": request_config.get("certificate_threshold"),
        "tau_stop": request_config.get("tau_stop"),
        "max_alternatives": request_config.get("max_alternatives"),
        "optimization_mode": request_config.get("optimization_mode"),
        "weights": request_config.get("weights"),
    }
    row = {
        "od_id": od["od_id"],
        "variant_id": spec.variant_id,
        "pipeline_mode": spec.pipeline_mode,
        "pipeline_version": args.model_version,
        "seed": od["seed"],
        "trip_length_bin": od["trip_length_bin"],
        "origin_lat": od["origin_lat"],
        "origin_lon": od["origin_lon"],
        "destination_lat": od["destination_lat"],
        "destination_lon": od["destination_lon"],
        "straight_line_km": od["straight_line_km"],
        "profile_id": od.get("profile_id"),
        "corpus_group": od.get("corpus_group"),
        "corpus_kind": od.get("corpus_kind"),
        "od_ambiguity_index": (
            as_float(od.get("od_ambiguity_index"), float("nan"))
            if od.get("od_ambiguity_index") not in (None, "")
            else as_float(od.get("ambiguity_index"), float("nan"))
            if od.get("ambiguity_index") not in (None, "")
            else None
        ),
        "od_ambiguity_confidence": ambiguity_confidence,
        "od_ambiguity_source_count": ambiguity_source_count,
        "od_ambiguity_source_mix": ambiguity_source_mix,
        "od_ambiguity_source_mix_count": ambiguity_source_mix_count,
        "od_ambiguity_source_support": str(od.get("od_ambiguity_source_support") or "").strip() or None,
        "od_ambiguity_source_support_strength": as_float(od.get("od_ambiguity_source_support_strength"), float("nan")) if od.get("od_ambiguity_source_support_strength") not in (None, "") else None,
        "od_ambiguity_source_entropy": ambiguity_source_entropy,
        "od_ambiguity_support_ratio": ambiguity_support_ratio,
        "od_ambiguity_prior_strength": ambiguity_prior_strength,
        "od_ambiguity_family_density": ambiguity_family_density,
        "od_ambiguity_margin_pressure": ambiguity_margin_pressure,
        "od_ambiguity_spread_pressure": ambiguity_spread_pressure,
        "od_ambiguity_toll_instability": ambiguity_toll_instability,
        "observed_ambiguity_index": None,
        "od_candidate_path_count": int(as_float(od.get("candidate_probe_path_count"), 0.0)),
        "od_corridor_family_count": int(as_float(od.get("candidate_probe_corridor_family_count"), 0.0)),
        "od_objective_spread": as_float(od.get("candidate_probe_objective_spread"), float("nan")) if od.get("candidate_probe_objective_spread") not in (None, "") else None,
        "od_nominal_margin_proxy": as_float(od.get("candidate_probe_nominal_margin"), float("nan")) if od.get("candidate_probe_nominal_margin") not in (None, "") else None,
        "od_toll_disagreement_rate": as_float(od.get("candidate_probe_toll_disagreement_rate"), float("nan")) if od.get("candidate_probe_toll_disagreement_rate") not in (None, "") else None,
        "od_engine_disagreement_prior": as_float(od.get("candidate_probe_engine_disagreement_prior"), float("nan")) if od.get("candidate_probe_engine_disagreement_prior") not in (None, "") else None,
        "od_hard_case_prior": as_float(od.get("hard_case_prior"), float("nan")) if od.get("hard_case_prior") not in (None, "") else None,
        "ambiguity_prior_sample_count": int(as_float(od.get("ambiguity_prior_sample_count"), 0.0)) if od.get("ambiguity_prior_sample_count") not in (None, "") else 0,
        "ambiguity_prior_support_count": int(as_float(od.get("ambiguity_prior_support_count"), 0.0)) if od.get("ambiguity_prior_support_count") not in (None, "") else 0,
        "row_override_count": int(request_config.get("row_override_count") or 0),
        "ambiguity_budget_band": request_config.get("ambiguity_budget_band"),
        "ambiguity_budget_prior": request_config.get("ambiguity_budget_prior"),
        "ambiguity_budget_prior_gap": ambiguity_budget_prior_gap,
        "budget_prior_exceeds_raw": bool(
            math.isfinite(ambiguity_budget_prior_value)
            and math.isfinite(raw_ambiguity_value)
            and ambiguity_budget_prior_value > raw_ambiguity_value + 1e-9
        ),
        "effective_request_config_json": json.dumps(selected_config, sort_keys=True, separators=(",", ":")),
        "route_id": selected_id,
        "route_source": spec.pipeline_mode,
        "candidate_count_display": len(candidates) if isinstance(candidates, list) else 0,
        "candidate_count_raw": int(dccs_summary.get("candidate_count_raw") or len(candidates) if isinstance(candidates, list) else 0),
        "refined_count": refined_count,
        "frontier_count": len(frontier_rows),
        "nontrivial_frontier": len(frontier_rows) > 1,
        "iteration_count": iteration_count,
        "search_budget": effective_search_budget,
        "evidence_budget": effective_evidence_budget,
        "search_budget_used": search_budget_used,
        "evidence_budget_used": evidence_budget_used,
        "search_budget_utilization": search_budget_utilization,
        "evidence_budget_utilization": evidence_budget_utilization,
        "initial_certificate": initial_certificate_value,
        "initial_certificate_stop": initial_certificate_stop if spec.pipeline_mode == "voi" else None,
        "unnecessary_voi_refine": unnecessary_voi_refine if spec.pipeline_mode == "voi" else None,
        "time_to_certification_ms": time_to_certification if spec.pipeline_mode == "voi" else None,
        "controller_shortcut": controller_shortcut if spec.pipeline_mode == "voi" else None,
        "voi_stop_after_certification": voi_stop_after_certification if spec.pipeline_mode == "voi" else None,
        "certificate_selective": None,
        "certificate_threshold": effective_certificate_threshold,
        "certificate": round(certificate, 6) if math.isfinite(certificate) else None,
        "certified": certified,
        "refinement_selection_policy": (
            str(trace.get("refinement_policy") or dccs_summary.get("refinement_policy") or spec.refinement_policy or "").strip()
            if isinstance(trace, dict)
            else str(spec.refinement_policy or "").strip()
        ),
        "refinement_selected_candidate_count": int(as_float((trace_candidate_diag or {}).get("selected_candidate_count"), 0.0)) if isinstance(trace_candidate_diag, Mapping) else 0,
        "refinement_selected_candidate_ids_json": (
            json.dumps(trace.get("selected_candidate_ids", []), sort_keys=True)
            if isinstance(trace, dict) and isinstance(trace.get("selected_candidate_ids"), list)
            else json.dumps([], sort_keys=True)
        ),
        "selected_distance_km": selected_metrics["distance_km"],
        "selected_duration_s": selected_metrics["duration_s"],
        "selected_monetary_cost": selected_metrics["monetary_cost"],
        "selected_emissions_kg": selected_metrics["emissions_kg"],
        "selected_p95_duration_s": as_float((selected.get("uncertainty") or {}).get("p95_duration_s"), float("nan")) if isinstance(selected.get("uncertainty"), dict) else None,
        "selected_cvar95_duration_s": as_float((selected.get("uncertainty") or {}).get("cvar95_duration_s"), float("nan")) if isinstance(selected.get("uncertainty"), dict) else None,
        "osrm_method": osrm.method,
        "osrm_distance_km": osrm.metrics["distance_km"],
        "osrm_duration_s": osrm.metrics["duration_s"],
        "osrm_monetary_cost": osrm.metrics["monetary_cost"],
        "osrm_emissions_kg": osrm.metrics["emissions_kg"],
        "ors_method": ors.method,
        "ors_provider_mode": ors.provider_mode,
        "ors_baseline_policy": ors.baseline_policy,
        "ors_asset_manifest_hash": ors.asset_manifest_hash,
        "ors_asset_recorded_at": ors.asset_recorded_at,
        "ors_asset_freshness_status": ors.asset_freshness_status,
        "ors_graph_identity_status": (
            str((ors.engine_manifest or {}).get("identity_status") or ors.asset_freshness_status or "").strip() or None
        ),
        "ors_engine_image": str((ors.engine_manifest or {}).get("compose_image") or "").strip() or None,
        "ors_graph_build_date": str(((ors.engine_manifest or {}).get("graph_build_info") or {}).get("graph_build_date") or "").strip() or None,
        "ors_graph_osm_date": str(((ors.engine_manifest or {}).get("graph_build_info") or {}).get("osm_date") or "").strip() or None,
        "ors_graph_file_count": int(as_float((ors.engine_manifest or {}).get("graph_file_count"), 0.0)) if isinstance(ors.engine_manifest, dict) else None,
        "ors_graph_total_bytes": int(as_float((ors.engine_manifest or {}).get("graph_total_bytes"), 0.0)) if isinstance(ors.engine_manifest, dict) else None,
        "ors_graph_listing_digest": str((ors.engine_manifest or {}).get("graph_listing_digest") or "").strip() or None,
        "ors_distance_km": ors.metrics["distance_km"],
        "ors_duration_s": ors.metrics["duration_s"],
        "ors_monetary_cost": ors.metrics["monetary_cost"],
        "ors_emissions_kg": ors.metrics["emissions_kg"],
        "delta_vs_osrm_distance_km": round(selected_metrics["distance_km"] - osrm.metrics["distance_km"], 6),
        "delta_vs_osrm_duration_s": round(selected_metrics["duration_s"] - osrm.metrics["duration_s"], 6),
        "delta_vs_osrm_monetary_cost": round(selected_metrics["monetary_cost"] - osrm.metrics["monetary_cost"], 6),
        "delta_vs_osrm_emissions_kg": round(selected_metrics["emissions_kg"] - osrm.metrics["emissions_kg"], 6),
        "delta_vs_ors_distance_km": round(selected_metrics["distance_km"] - ors.metrics["distance_km"], 6),
        "delta_vs_ors_duration_s": round(selected_metrics["duration_s"] - ors.metrics["duration_s"], 6),
        "delta_vs_ors_monetary_cost": round(selected_metrics["monetary_cost"] - ors.metrics["monetary_cost"], 6),
        "delta_vs_ors_emissions_kg": round(selected_metrics["emissions_kg"] - ors.metrics["emissions_kg"], 6),
        "dominates_osrm": dominates(win_selected, osrm.metrics),
        "dominates_ors": dominates(win_selected, ors.metrics),
        "dominates_v0": None,
        "dominates_best_baseline": dominates(win_selected, best_baseline_metrics),
        "weighted_win_osrm": weighted_osrm < weighted_osrm_base,
        "weighted_win_ors": weighted_ors < weighted_ors_base,
        "weighted_win_v0": None,
        "weighted_win_best_baseline": min(weighted_osrm, weighted_ors) < best_baseline_weighted_base,
        "weighted_margin_vs_osrm": weighted_margin_osrm,
        "weighted_margin_vs_ors": weighted_margin_ors,
        "weighted_margin_vs_v0": None,
        "weighted_margin_vs_best_baseline": weighted_margin_best_baseline,
        "balanced_win_osrm": balanced_gain_osrm > 0.0,
        "balanced_win_ors": balanced_gain_ors > 0.0,
        "balanced_win_v0": None,
        "balanced_win_best_baseline": balanced_gain_best_baseline > 0.0,
        "time_preserving_win_osrm": _time_preserving_outcome(
            duration_delta_s=round(selected_metrics["duration_s"] - osrm.metrics["duration_s"], 6),
            quality_win=(weighted_osrm < weighted_osrm_base),
        ),
        "time_preserving_win_ors": _time_preserving_outcome(
            duration_delta_s=round(selected_metrics["duration_s"] - ors.metrics["duration_s"], 6),
            quality_win=(weighted_ors < weighted_ors_base),
        ),
        "time_preserving_win_best_baseline": _time_preserving_outcome(
            duration_delta_s=(
                round(selected_metrics["duration_s"] - osrm.metrics["duration_s"], 6)
                if best_baseline_provider == "osrm"
                else round(selected_metrics["duration_s"] - ors.metrics["duration_s"], 6)
            ),
            quality_win=(min(weighted_osrm, weighted_ors) < best_baseline_weighted_base),
        ),
        "time_preserving_dominance_osrm": _time_preserving_outcome(
            duration_delta_s=round(selected_metrics["duration_s"] - osrm.metrics["duration_s"], 6),
            quality_win=dominates(win_selected, osrm.metrics),
        ),
        "time_preserving_dominance_ors": _time_preserving_outcome(
            duration_delta_s=round(selected_metrics["duration_s"] - ors.metrics["duration_s"], 6),
            quality_win=dominates(win_selected, ors.metrics),
        ),
        "time_preserving_dominance_best_baseline": _time_preserving_outcome(
            duration_delta_s=(
                round(selected_metrics["duration_s"] - osrm.metrics["duration_s"], 6)
                if best_baseline_provider == "osrm"
                else round(selected_metrics["duration_s"] - ors.metrics["duration_s"], 6)
            ),
            quality_win=dominates(win_selected, best_baseline_metrics),
        ),
        "best_baseline_provider": best_baseline_provider,
        "balanced_gain_vs_osrm_score": balanced_gain_osrm,
        "balanced_gain_vs_ors_score": balanced_gain_ors,
        "robust_win_osrm": robust_win(selected, osrm.route),
        "robust_win_ors": robust_win(selected, ors.route),
        "frontier_hypervolume": frontier_hypervolume,
        "frontier_coverage_osrm": coverage_of_baseline(frontier_rows, osrm.metrics),
        "frontier_coverage_ors": coverage_of_baseline(frontier_rows, ors.metrics),
        "frontier_epsilon_osrm": additive_epsilon_indicator(frontier_rows, osrm.metrics),
        "frontier_epsilon_ors": additive_epsilon_indicator(frontier_rows, ors.metrics),
        "frontier_spread": frontier_spread,
        "frontier_crowding_mean": frontier_crowding,
        "frontier_diversity_index": frontier_diversity_idx,
        "frontier_entropy": frontier_ent,
        "time_to_best_iteration": time_to_best,
        "refine_cost_prediction_error_deprecated": refine_mape,
        "refine_cost_mape": refine_mape,
        "refine_cost_sample_count": refine_sample_count,
        "refine_cost_positive_sample_count": refine_positive_sample_count,
        "refine_cost_zero_observed_count": refine_zero_observed_count,
        "refine_cost_mae_ms": refine_mae_value,
        "refine_cost_rank_correlation": refine_rank_corr,
        "action_efficiency": action_eff,
        "nominal_winner_margin": winner_margin,
        "near_tie_mass": near_tie_mass(frontier_rows, weights=effective_weights),
        "certificate_margin": certificate_margin(
            certificate if math.isfinite(certificate) else None,
            threshold=effective_certificate_threshold,
        ),
        "certificate_runner_up_gap": certificate_runner_up_gap(cert_map, winner_id=selected_id) if cert_map else None,
        "fragility_entropy": fragility_entropy(selected_fragility) if isinstance(selected_fragility, dict) else None,
        "competitor_turnover_rate": competitor_turnover_rate(selected_competitors) if isinstance(selected_competitors, dict) else None,
        "dccs_dc_yield": as_float(dccs_summary.get("dc_yield"), float("nan")) if dccs_summary else None,
        "dccs_challenger_hit_rate": as_float(dccs_summary.get("challenger_hit_rate"), float("nan")) if dccs_summary else None,
        "dccs_frontier_gain_per_refinement": as_float(dccs_summary.get("frontier_gain_per_refinement"), float("nan")) if dccs_summary else None,
        "dccs_decision_flips": int(dccs_summary.get("decision_flips") or 0) if dccs_summary else None,
        "dccs_score_label_correlation": _dccs_score_correlation(candidate_rows) if candidate_rows else None,
        "dccs_frontier_recall_at_budget": score_ranked_recall(candidate_rows, budget=search_budget_used, positive_labels=dccs_positive_labels) if candidate_rows else None,
        "dccs_corridor_family_recall": corridor_family_recall(candidate_rows, budget=search_budget_used, positive_labels=dccs_positive_labels) if candidate_rows else None,
        "refc_world_count": int(world_manifest.get("world_count") or 0) if world_manifest else 0,
        "refc_unique_world_count": int(world_manifest.get("unique_world_count") or world_manifest.get("world_count") or 0) if world_manifest else 0,
        "refc_world_reuse_rate": as_float(world_manifest.get("world_reuse_rate"), float("nan")) if world_manifest else None,
        "refc_hard_stress_pack_count": int(world_manifest.get("hard_case_stress_pack_count") or 0) if world_manifest else 0,
        "refc_stress_world_fraction": as_float(world_manifest.get("stress_world_fraction"), float("nan")) if world_manifest else None,
        "requested_cert_world_count": requested_world_count,
        "effective_cert_world_count": effective_world_count,
        "world_count_policy": str(world_manifest.get("world_count_policy") or "").strip() or None if world_manifest else None,
        "world_count_efficiency": runtime_ratio(effective_world_count, requested_world_count),
        "refc_active_family_count": len(certificate_summary.get("active_families", [])) if isinstance(certificate_summary.get("active_families"), list) else 0,
        "initial_refc_top_fragility_family": initial_refc_top_fragility_family,
        "initial_refc_top_refresh_family": initial_top_refresh_family,
        "initial_refc_top_vor": initial_top_vor,
        "initial_refc_vor_gap": initial_top_vor_gap,
        "final_refc_top_fragility_family": final_refc_top_fragility_family,
        "final_refc_top_refresh_family": top_refresh_family,
        "final_refc_top_vor": top_vor,
        "final_refc_vor_gap": top_vor_gap,
        "initial_winner_fragility_mass": initial_winner_fragility_mass,
        "final_winner_fragility_mass": final_winner_fragility_mass,
        "initial_winner_fragility_nonzero": initial_winner_fragility_nonzero,
        "winner_fragility_nonzero": winner_fragility_nonzero,
        "initial_refc_top_vor_positive": initial_refc_top_vor_positive,
        "refc_top_vor_positive": refc_top_vor_positive,
        "refresh_signal_persistent": refresh_signal_persistent,
        "refresh_first_productive": refresh_first_productive,
        "refresh_resolution_honest": refresh_resolution_honest,
        "refresh_resolution_reason": refresh_resolution_reason,
        "refc_top_fragility_family": _top_fragility(route_fragility_map, selected_id) if route_fragility_map else None,
        "refc_top_refresh_family": top_refresh_family,
        "refc_top_vor": top_vor,
        "refc_vor_gap": top_vor_gap,
        "certificate_winner_route_id": certificate_winner_id,
        "selector_certificate_disagreement": (
            certificate_winner_id is not None and selected_id != str(certificate_winner_id)
        ),
        "refc_top_competitor_route_id": _top_competitor(competitor_fragility_breakdown, selected_id) if competitor_fragility_breakdown else None,
        "voi_stop_reason": voi_stop.get("stop_reason"),
        "voi_best_rejected_action": voi_stop.get("best_rejected_action"),
        "voi_best_rejected_q": as_float(voi_stop.get("best_rejected_q"), float("nan")) if voi_stop else None,
        "voi_action_count": action_count,
        "voi_refine_action_count": refine_count,
        "voi_refresh_action_count": refresh_count,
        "voi_resample_action_count": resample_count,
        "voi_productive_action_count": productive_action_count if spec.pipeline_mode == "voi" else None,
        "voi_nonproductive_action_count": (
            max(0, productive_action_denominator - productive_action_count)
            if spec.pipeline_mode == "voi"
            else None
        ),
        "productive_voi_action_rate": (
            productive_action_rate(productive_action_count, productive_action_denominator)
            if spec.pipeline_mode == "voi"
            else None
        ),
        "voi_controller_engaged": action_count > 0 or iteration_count > 0,
        "preemptive_comparator_seeded": preemptive_comparator_seeded,
        "preemptive_comparator_candidate_count": preemptive_comparator_candidate_count,
        "preemptive_comparator_source_count": preemptive_comparator_source_count,
        "selected_from_preemptive_comparator_seed": selected_from_preemptive_comparator_seed,
        "voi_realized_certificate_lift": (
            _voi_realized_certificate_lift(
                voi_entries=voi_entries,
                initial_certificate=initial_certificate_value,
                certificate=certificate if math.isfinite(certificate) else None,
            )
            if spec.pipeline_mode == "voi"
            else None
        ),
        "voi_realized_runner_up_gap_lift": None,
        "voi_realized_margin_lift": None,
        "voi_realized_frontier_gain": None,
        "voi_realized_runtime_delta_ms": None,
        "weighted_margin_gain_vs_v0": None,
        "balanced_gain_delta_vs_v0_score": None,
        "duration_gain_vs_v0_s": None,
        "monetary_gain_vs_v0": None,
        "emissions_gain_vs_v0_kg": None,
        "frontier_hypervolume_gain_vs_v0": None,
        "certificate_lift_vs_v0": None,
        "certificate_availability_gain_vs_v0": None,
        "runtime_ms": runtime_ms,
        "algorithm_runtime_ms": algorithm_runtime_ms,
        "baseline_acquisition_runtime_ms": baseline_acquisition_runtime_ms,
        "baseline_runtime_share": runtime_share(baseline_acquisition_runtime_ms, runtime_ms),
        "runtime_ratio_vs_osrm": runtime_ratio(runtime_ms, osrm.compute_ms),
        "runtime_ratio_vs_ors": runtime_ratio(runtime_ms, ors.compute_ms),
        "algorithm_runtime_ratio_vs_osrm": runtime_ratio(algorithm_runtime_ms, osrm.compute_ms),
        "algorithm_runtime_ratio_vs_ors": runtime_ratio(algorithm_runtime_ms, ors.compute_ms),
        "runtime_gap_vs_osrm_ms": runtime_gap_vs_osrm_ms,
        "runtime_gap_vs_ors_ms": runtime_gap_vs_ors_ms,
        "algorithm_runtime_gap_vs_osrm_ms": algorithm_runtime_gap_vs_osrm_ms,
        "algorithm_runtime_gap_vs_ors_ms": algorithm_runtime_gap_vs_ors_ms,
        "row_local_warmup_ms": row_local_warmup_value,
        "warmup_amortized_ms": None,
        "warmup_overhead_share": runtime_share(row_local_warmup_value, runtime_ms),
        "global_startup_overhead_ms": global_startup_overhead_value,
        "global_startup_share_of_algorithm": runtime_ratio(global_startup_overhead_value, algorithm_runtime_ms),
        "runtime_per_refined_candidate_ms": runtime_per_unit(algorithm_runtime_ms, search_budget_used or refined_count),
        "runtime_per_frontier_member_ms": runtime_per_unit(algorithm_runtime_ms, len(frontier_rows)),
        "memory_per_refined_candidate_mb": memory_per_unit(bytes_to_megabytes((trace_resource_usage or {}).get("rss_bytes")) if isinstance(trace_resource_usage, Mapping) else None, search_budget_used or refined_count),
        "quality_per_second": quality_per_second(
            weighted_margin=weighted_margin_best_baseline,
            balanced_gain=balanced_gain_best_baseline,
            runtime_ms=algorithm_runtime_ms,
        ),
        "route_improvement_per_second": route_improvement_per_second(
            weighted_margin=weighted_margin_best_baseline,
            balanced_gain=balanced_gain_best_baseline,
            runtime_ms=algorithm_runtime_ms,
        ),
        "frontier_gain_per_ms": frontier_gain_per_ms(frontier_efficiency_gain, algorithm_runtime_ms),
        "certificate_gain_per_world": certificate_gain_per_world(certificate_margin_value, effective_world_count),
        "controller_cost_per_certificate_point": None,
        "cache_reuse_ratio": cache_reuse_ratio(route_cache_hit_rate, as_float(world_manifest.get("world_reuse_rate"), float("nan")) if world_manifest else None),
        "baseline_identity_verified": (
            str(args.ors_baseline_policy).strip().lower() == "local_service"
            and str((ors.engine_manifest or {}).get("identity_status") or ors.asset_freshness_status or "").strip() == "graph_identity_verified"
            and bool(str(osrm.method or "").strip())
        ) if str(args.ors_baseline_policy).strip().lower() == "local_service" else None,
        "algorithm_runtime_speedup_vs_v0": None,
        "runtime_speedup_vs_v0": None,
        "runtime_win_v0": None,
        "algorithm_runtime_win_v0": None,
        "backend_ready_wait_ms": as_float(readiness.get("wait_elapsed_ms"), float("nan")) if isinstance(readiness, Mapping) else None,
        "backend_ready_probe_ms": as_float(readiness.get("compute_ms"), float("nan")) if isinstance(readiness, Mapping) else None,
        "route_graph_warmup_elapsed_ms": route_graph_warmup_ms,
        "preflight_ms": preflight_value if math.isfinite(preflight_value) else None,
        "preflight_and_warmup_ms": preflight_and_warmup_ms,
        "process_rss_mb": bytes_to_megabytes((trace_resource_usage or {}).get("rss_bytes")) if isinstance(trace_resource_usage, Mapping) else None,
        "process_vms_mb": bytes_to_megabytes((trace_resource_usage or {}).get("vms_bytes")) if isinstance(trace_resource_usage, Mapping) else None,
        "refc_shortcut_used": bool(
            isinstance(trace_cert_runtime, Mapping) and as_float(trace_cert_runtime.get("shortcut_count"), 0.0) > 0.0
        ),
        "refc_cache_hits": int(as_float(trace_cert_runtime.get("cache_hits"), 0.0)) if isinstance(trace_cert_runtime, Mapping) else 0,
        "ambiguity_alignment": ambiguity_alignment(ambiguity_prior, None),
        "ambiguity_absolute_error": None,
        "supported_ambiguity_alignment": None,
        "ambiguity_prior_gap": None,
        "ambiguity_prior_overtrigger": None,
        "controller_activation_on_high_ambiguity": controller_activation_on_high_ambiguity(ambiguity_prior, action_count > 0 or iteration_count > 0),
        "stage_option_build_ms": stage_option_build_value if math.isfinite(stage_option_build_value) else None,
        "option_build_reuse_rate": option_build_reuse,
        "option_build_cache_hits": option_build_cache_hits,
        "option_build_rebuild_count": option_build_rebuild_count,
        "option_build_cache_hit_rate": option_build_cache_hit_rate,
        "option_build_cache_savings_ms_per_row": option_build_cache_savings_ms_per_row,
        "search_completeness_score": search_completeness_score,
        "search_completeness_gap": search_completeness_gap,
        "prior_support_strength": prior_support_strength,
        "support_richness": support_richness,
        "ambiguity_pressure": ambiguity_pressure,
        "pending_challenger_mass": pending_challenger_mass,
        "best_pending_flip_probability": best_pending_flip_probability,
        "corridor_family_recall": corridor_family_recall_value,
        "frontier_recall_at_budget": frontier_recall_at_budget_value,
        "top_refresh_gain": top_refresh_gain_value,
        "top_fragility_mass": top_fragility_mass_value,
        "competitor_pressure": competitor_pressure_value,
        "credible_search_uncertainty": credible_search_uncertainty_value,
        "credible_evidence_uncertainty": credible_evidence_uncertainty_value,
        "supported_hard_case": supported_hard_case,
        "evidence_first_engagement": evidence_first_engagement if spec.pipeline_mode == "voi" else None,
        "evidence_only_engagement": evidence_only_engagement if spec.pipeline_mode == "voi" else None,
        "first_controller_action_kind": first_controller_action_kind if spec.pipeline_mode == "voi" else None,
        "controller_refresh_ranking_basis": (
            controller_refresh_ranking_basis if spec.pipeline_mode in {"dccs_refc", "voi"} else None
        ),
        "controller_top_refresh_family": (
            controller_top_refresh_family if spec.pipeline_mode in {"dccs_refc", "voi"} else None
        ),
        "controller_top_refresh_gain": (
            controller_top_refresh_gain if spec.pipeline_mode in {"dccs_refc", "voi"} else None
        ),
        "controller_refresh_fallback_activated": (
            controller_refresh_fallback_activated if spec.pipeline_mode in {"dccs_refc", "voi"} else None
        ),
        "controller_empirical_vs_raw_refresh_disagreement": (
            controller_empirical_vs_raw_refresh_disagreement if spec.pipeline_mode in {"dccs_refc", "voi"} else None
        ),
        "voi_dccs_cache_hits": voi_dccs_cache_hits if spec.pipeline_mode == "voi" else None,
        "voi_dccs_cache_misses": voi_dccs_cache_misses if spec.pipeline_mode == "voi" else None,
        "voi_dccs_cache_hit_rate": voi_dccs_cache_hit_rate if spec.pipeline_mode == "voi" else None,
        "comparator_independent": not selected_from_comparator_engine and not selected_from_preemptive_comparator_seed,
        "realized_diversity_collapse": diversity_collapse,
        "realized_diversity_collapse_reason": str(
            dccs_summary.get("diversity_collapse_reason")
            or (trace_diversity_rescue.get("collapse_reason") if isinstance(trace_diversity_rescue, Mapping) else "")
            or ""
        ).strip()
        or None,
        "realized_raw_corridor_family_count": int(
            as_float(
                dccs_summary.get("raw_corridor_family_count")
                if dccs_summary.get("raw_corridor_family_count") not in (None, "")
                else (trace_diversity_rescue.get("raw_corridor_family_count") if isinstance(trace_diversity_rescue, Mapping) else 0),
                0.0,
            )
        ),
        "realized_refined_corridor_family_count": int(
            as_float(
                dccs_summary.get("refined_corridor_family_count_after")
                if dccs_summary.get("refined_corridor_family_count_after") not in (None, "")
                else (trace_diversity_rescue.get("refined_corridor_family_count_after") if isinstance(trace_diversity_rescue, Mapping) else 0),
                0.0,
            )
        ),
        "supplemental_challenger_activated": bool(
            dccs_summary.get("supplemental_challenger_activated")
            or (trace_diversity_rescue.get("supplemental_challenger_activated") if isinstance(trace_diversity_rescue, Mapping) else False)
        ),
        "supplemental_challenger_source_count": len(supplemental_sources),
        "supplemental_challenger_candidate_count": int(
            as_float(
                dccs_summary.get("supplemental_candidate_count")
                if dccs_summary.get("supplemental_candidate_count") not in (None, "")
                else (trace_diversity_rescue.get("supplemental_candidate_count") if isinstance(trace_diversity_rescue, Mapping) else 0),
                0.0,
            )
        ),
        "supplemental_challenger_selected_count": int(
            as_float(
                dccs_summary.get("supplemental_selected_count")
                if dccs_summary.get("supplemental_selected_count") not in (None, "")
                else (trace_diversity_rescue.get("supplemental_selected_count") if isinstance(trace_diversity_rescue, Mapping) else 0),
                0.0,
            )
        ),
        "supplemental_challenger_budget_used": int(
            as_float(
                dccs_summary.get("supplemental_budget_used")
                if dccs_summary.get("supplemental_budget_used") not in (None, "")
                else (trace_diversity_rescue.get("supplemental_budget_used") if isinstance(trace_diversity_rescue, Mapping) else 0),
                0.0,
            )
        ),
        "selected_candidate_source_label": selected_candidate_source_label,
        "selected_candidate_source_engine": selected_candidate_source_engine,
        "selected_candidate_source_stage": selected_candidate_source_stage,
        "selected_final_route_source_label": selected_final_route_source_label,
        "selected_final_route_source_engine": selected_final_route_source_engine,
        "selected_final_route_source_stage": selected_final_route_source_stage,
        "selected_from_supplemental_rescue": selected_from_supplemental_rescue,
        "selected_from_comparator_engine": selected_from_comparator_engine,
        "strict_failure_eliminated": None,
        "controller_value_per_second": controller_vps,
        "route_cache_hits": route_cache_hits,
        "route_cache_misses": route_cache_misses,
        "route_cache_hit_rate": route_cache_hit_rate,
        "graph_k_raw_cache_hit": graph_k_raw_cache_hit,
        "graph_low_ambiguity_fast_path": graph_low_ambiguity_fast_path,
        "graph_supported_ambiguity_fast_fallback": graph_supported_ambiguity_fast_fallback,
        "route_state_cache_hits": route_state_cache_hits,
        "route_state_cache_misses": route_state_cache_misses,
        "route_state_cache_hit_rate": route_state_cache_hit_rate,
        "route_request_ms": route_request_ms,
        "baseline_osrm_ms": osrm.compute_ms,
        "baseline_ors_ms": ors.compute_ms,
        "stage_k_raw_ms": stage_k_raw_value if stage_k_raw_value is None or math.isfinite(stage_k_raw_value) else None,
        "stage_k_raw_graph_search_initial_ms": (
            stage_k_raw_graph_search_initial_value if math.isfinite(stage_k_raw_graph_search_initial_value) else None
        ),
        "stage_k_raw_graph_search_retry_ms": (
            stage_k_raw_graph_search_retry_value if math.isfinite(stage_k_raw_graph_search_retry_value) else None
        ),
        "stage_k_raw_graph_search_rescue_ms": (
            stage_k_raw_graph_search_rescue_value if math.isfinite(stage_k_raw_graph_search_rescue_value) else None
        ),
        "stage_k_raw_graph_search_supplemental_ms": (
            stage_k_raw_graph_search_supplemental_value
            if math.isfinite(stage_k_raw_graph_search_supplemental_value)
            else None
        ),
        "stage_k_raw_osrm_fallback_ms": (
            stage_k_raw_osrm_fallback_value if math.isfinite(stage_k_raw_osrm_fallback_value) else None
        ),
        "stage_dccs_ms": _stage_value(trace, "dccs_ms"),
        "stage_refc_ms": _stage_value(trace, "refc_ms"),
        "stage_voi_ms": stage_voi_value,
        "stage_pareto_ms": _stage_value(trace, "pareto_ms"),
        "stage_refinement_ms": _stage_value_any(trace, "refinement_ms", "osrm_refine_ms"),
        "stage_supplemental_rescue_ms": _stage_value(trace, "supplemental_rescue_ms"),
        "stage_preemptive_comparator_seed_ms": _stage_value(trace, "preemptive_comparator_seed_ms"),
        "ors_snapshot_mode": args.ors_snapshot_mode,
        "ors_snapshot_used": ors.snapshot_used,
        "ors_snapshot_recorded_at": ors.snapshot_recorded_at,
        "ors_snapshot_request_hash": ors.snapshot_request_hash,
        "ors_snapshot_response_hash": ors.snapshot_response_hash,
        "ors_snapshot_provider_mode": ors.provider_mode,
        "artifact_complete": str(artifact_validation.get("status") or "unknown") == "ok",
        "artifact_status": str(artifact_validation.get("status") or "unknown"),
        "artifact_missing": json.dumps(list(artifact_validation.get("missing") or []), sort_keys=True),
        "evidence_policy": str((evidence_validation.get("policy") or {}).get("mode") or STRICT_EVIDENCE_POLICY),
        "route_evidence_ok": str(evidence_validation.get("status") or "unknown") == "ok",
        "route_evidence_status": str(evidence_validation.get("status") or "unknown"),
        "route_evidence_issues": json.dumps(evidence_validation.get("issues", []), sort_keys=True),
        "failure_reason": "",
        "artifact_run_id": route_response.get("run_id"),
        "manifest_endpoint": route_response.get("manifest_endpoint"),
        "artifacts_endpoint": route_response.get("artifacts_endpoint"),
    }
    row["observed_ambiguity_index"] = _observed_ambiguity_index(row)
    if row.get("od_ambiguity_index") in (None, "") and isinstance(manifest_ambiguity_context, Mapping):
        manifest_prior = _shared_raw_ambiguity_prior_value(manifest_ambiguity_context)
        if manifest_prior is not None:
            row["od_ambiguity_index"] = manifest_prior
    row["ambiguity_alignment"] = ambiguity_alignment(ambiguity_prior, row["observed_ambiguity_index"])
    row["ambiguity_absolute_error"] = ambiguity_absolute_error(ambiguity_prior, row["observed_ambiguity_index"])
    row["supported_ambiguity_alignment"] = supported_ambiguity_alignment(
        ambiguity_prior,
        row["observed_ambiguity_index"],
        confidence=row.get("od_ambiguity_confidence"),
        support_ratio=row.get("od_ambiguity_support_ratio"),
        source_mix_count=row.get("od_ambiguity_source_mix_count"),
    )
    observed_ambiguity = as_float(row.get("observed_ambiguity_index"), float("nan"))
    row["ambiguity_prior_gap"] = (
        round(abs(float(ambiguity_prior) - observed_ambiguity), 6)
        if ambiguity_prior is not None and math.isfinite(observed_ambiguity)
        else None
    )
    row["cohort_label"] = _cohort_label(row)
    certificate_value = as_float(row.get("certificate"), float("nan"))
    certificate_selective_scope = (
        row.get("pipeline_mode") in {"dccs_refc", "voi"}
        and row.get("cohort_label") != "representative"
        and math.isfinite(certificate_value)
    )
    row["certificate_selective"] = (
        bool(certificate_value < 0.999999)
        if certificate_selective_scope
        else None
    )
    row["ambiguity_prior_overtrigger"] = bool(
        ambiguity_prior is not None
        and math.isfinite(observed_ambiguity)
        and float(ambiguity_prior) >= 0.45
        and observed_ambiguity <= 0.08
    )
    row["controller_stress_row"] = _is_controller_stress_row(row)
    row["hard_case"] = _is_hard_case_row(row)
    row["supported_hard_case"] = bool(row.get("hard_case") and row.get("supported_hard_case"))
    return row


def _failure_row(
    args: argparse.Namespace,
    od: dict[str, Any],
    spec: VariantSpec,
    *,
    failure_reason: str,
    osrm: BaselineResult,
    ors: BaselineResult,
    request_config: dict[str, Any] | None = None,
    artifact_missing: Sequence[str] = (),
    readiness_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request_config = request_config or _effective_request_config(args, od, variant_seed=int(args.seed))
    readiness = dict(readiness_summary or {})
    readiness_route_graph = readiness.get("route_graph", {}) if isinstance(readiness, Mapping) else {}
    ambiguity_prior = _od_ambiguity_prior(od)
    startup_components = _startup_components_ms(readiness if isinstance(readiness, Mapping) else None)
    global_startup_overhead_value = round(sum(startup_components), 6) if startup_components else None
    row = {field: None for field in RESULT_FIELDS}
    row.update(
        {
            "od_id": od["od_id"],
            "variant_id": spec.variant_id,
            "pipeline_mode": spec.pipeline_mode,
            "pipeline_version": args.model_version,
            "seed": od["seed"],
            "trip_length_bin": od["trip_length_bin"],
            "origin_lat": od["origin_lat"],
            "origin_lon": od["origin_lon"],
            "destination_lat": od["destination_lat"],
            "destination_lon": od["destination_lon"],
            "straight_line_km": od["straight_line_km"],
            "profile_id": od.get("profile_id"),
            "corpus_group": od.get("corpus_group"),
            "corpus_kind": od.get("corpus_kind"),
            "od_ambiguity_index": (
                _shared_raw_ambiguity_prior_value(od)
            ),
            "od_ambiguity_confidence": as_float(od.get("od_ambiguity_confidence"), float("nan")) if od.get("od_ambiguity_confidence") not in (None, "") else None,
            "od_ambiguity_source_count": int(as_float(od.get("od_ambiguity_source_count"), 0.0)) if od.get("od_ambiguity_source_count") not in (None, "") else None,
            "od_ambiguity_source_mix": str(od.get("od_ambiguity_source_mix") or "").strip() or None,
            "od_ambiguity_source_mix_count": int(as_float(od.get("od_ambiguity_source_mix_count"), 0.0)) if od.get("od_ambiguity_source_mix_count") not in (None, "") else None,
            "od_ambiguity_source_entropy": as_float(od.get("od_ambiguity_source_entropy"), float("nan")) if od.get("od_ambiguity_source_entropy") not in (None, "") else None,
            "od_ambiguity_support_ratio": as_float(od.get("od_ambiguity_support_ratio"), float("nan")) if od.get("od_ambiguity_support_ratio") not in (None, "") else None,
            "od_ambiguity_prior_strength": as_float(od.get("od_ambiguity_prior_strength"), float("nan")) if od.get("od_ambiguity_prior_strength") not in (None, "") else None,
            "od_ambiguity_family_density": as_float(od.get("od_ambiguity_family_density"), float("nan")) if od.get("od_ambiguity_family_density") not in (None, "") else None,
            "od_ambiguity_margin_pressure": as_float(od.get("od_ambiguity_margin_pressure"), float("nan")) if od.get("od_ambiguity_margin_pressure") not in (None, "") else None,
            "od_ambiguity_spread_pressure": as_float(od.get("od_ambiguity_spread_pressure"), float("nan")) if od.get("od_ambiguity_spread_pressure") not in (None, "") else None,
            "od_ambiguity_toll_instability": as_float(od.get("od_ambiguity_toll_instability"), float("nan")) if od.get("od_ambiguity_toll_instability") not in (None, "") else None,
            "observed_ambiguity_index": None,
            "od_candidate_path_count": int(as_float(od.get("candidate_probe_path_count"), 0.0)),
            "od_corridor_family_count": int(as_float(od.get("candidate_probe_corridor_family_count"), 0.0)),
            "od_objective_spread": as_float(od.get("candidate_probe_objective_spread"), float("nan")) if od.get("candidate_probe_objective_spread") not in (None, "") else None,
            "od_nominal_margin_proxy": as_float(od.get("candidate_probe_nominal_margin"), float("nan")) if od.get("candidate_probe_nominal_margin") not in (None, "") else None,
            "od_toll_disagreement_rate": as_float(od.get("candidate_probe_toll_disagreement_rate"), float("nan")) if od.get("candidate_probe_toll_disagreement_rate") not in (None, "") else None,
            "od_engine_disagreement_prior": as_float(od.get("candidate_probe_engine_disagreement_prior"), float("nan")) if od.get("candidate_probe_engine_disagreement_prior") not in (None, "") else None,
            "od_hard_case_prior": as_float(od.get("hard_case_prior"), float("nan")) if od.get("hard_case_prior") not in (None, "") else None,
            "row_override_count": int(request_config.get("row_override_count") or 0),
            "ambiguity_budget_band": request_config.get("ambiguity_budget_band"),
            "ambiguity_budget_prior": request_config.get("ambiguity_budget_prior"),
            "effective_request_config_json": json.dumps(
                {
                    "profile_id": request_config.get("profile_id"),
                    "corpus_group": request_config.get("corpus_group"),
                    "row_override_count": request_config.get("row_override_count"),
                    "row_override_keys": request_config.get("row_override_keys"),
                    "ambiguity_budget_band": request_config.get("ambiguity_budget_band"),
                    "ambiguity_budget_prior": request_config.get("ambiguity_budget_prior"),
                    "ambiguity_schedule_reason": request_config.get("ambiguity_schedule_reason"),
                    "scenario_mode": request_config.get("scenario_mode"),
                    "weather": request_config.get("weather"),
                    "departure_time_utc": request_config.get("departure_time_utc"),
                    "stochastic": request_config.get("stochastic"),
                    "search_budget": request_config.get("search_budget"),
                    "evidence_budget": request_config.get("evidence_budget"),
                    "world_count": request_config.get("world_count"),
                    "certificate_threshold": request_config.get("certificate_threshold"),
                    "tau_stop": request_config.get("tau_stop"),
                    "max_alternatives": request_config.get("max_alternatives"),
                    "optimization_mode": request_config.get("optimization_mode"),
                    "weights": request_config.get("weights"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "route_id": "",
            "route_source": spec.pipeline_mode,
            "candidate_count_display": 0,
            "candidate_count_raw": 0,
            "refined_count": 0,
            "frontier_count": 0,
            "nontrivial_frontier": False,
            "iteration_count": 0,
            "search_budget": int(request_config.get("search_budget") or args.search_budget),
            "evidence_budget": int(request_config.get("evidence_budget") or args.evidence_budget),
            "search_budget_used": 0,
            "evidence_budget_used": 0,
            "search_budget_utilization": 0.0,
            "evidence_budget_utilization": 0.0,
            "initial_certificate": None,
            "initial_certificate_stop": None,
            "unnecessary_voi_refine": None,
            "time_to_certification_ms": None,
            "controller_shortcut": None,
            "voi_stop_after_certification": None,
            "certificate_threshold": float(
                request_config.get("certificate_threshold")
                if request_config.get("certificate_threshold") is not None
                else args.certificate_threshold
            ),
            "certified": False,
            "refinement_selection_policy": str(spec.refinement_policy or ""),
            "refinement_selected_candidate_count": 0,
            "refinement_selected_candidate_ids_json": json.dumps([], sort_keys=True),
            "dominates_osrm": False,
            "dominates_ors": False,
            "dominates_v0": False,
            "dominates_best_baseline": False,
            "weighted_win_osrm": False,
            "weighted_win_ors": False,
            "weighted_win_v0": False,
            "weighted_win_best_baseline": False,
            "weighted_margin_vs_osrm": None,
            "weighted_margin_vs_ors": None,
            "weighted_margin_vs_v0": None,
            "weighted_margin_vs_best_baseline": None,
            "balanced_win_osrm": False,
            "balanced_win_ors": False,
            "balanced_win_v0": False,
            "balanced_win_best_baseline": False,
            "time_preserving_win_osrm": False,
            "time_preserving_win_ors": False,
            "time_preserving_win_best_baseline": False,
            "time_preserving_dominance_osrm": False,
            "time_preserving_dominance_ors": False,
            "time_preserving_dominance_best_baseline": False,
            "best_baseline_provider": None,
            "balanced_gain_vs_osrm_score": None,
            "balanced_gain_vs_ors_score": None,
            "robust_win_osrm": False,
            "robust_win_ors": False,
            "nominal_winner_margin": None,
            "near_tie_mass": None,
            "certificate_margin": None,
            "certificate_runner_up_gap": None,
            "fragility_entropy": None,
            "competitor_turnover_rate": None,
            "frontier_diversity_index": None,
            "frontier_entropy": None,
            "time_to_best_iteration": None,
            "refine_cost_prediction_error_deprecated": None,
            "refine_cost_mape": None,
            "refine_cost_sample_count": 0,
            "refine_cost_positive_sample_count": 0,
            "refine_cost_zero_observed_count": 0,
            "refine_cost_mae_ms": None,
            "refine_cost_rank_correlation": None,
            "action_efficiency": None,
            "osrm_method": osrm.method,
            "osrm_distance_km": osrm.metrics["distance_km"],
            "osrm_duration_s": osrm.metrics["duration_s"],
            "osrm_monetary_cost": osrm.metrics["monetary_cost"],
            "osrm_emissions_kg": osrm.metrics["emissions_kg"],
            "ors_method": ors.method,
            "ors_provider_mode": ors.provider_mode,
            "ors_baseline_policy": ors.baseline_policy,
            "ors_asset_manifest_hash": ors.asset_manifest_hash,
            "ors_asset_recorded_at": ors.asset_recorded_at,
            "ors_asset_freshness_status": ors.asset_freshness_status,
            "ors_graph_identity_status": (
                str((ors.engine_manifest or {}).get("identity_status") or ors.asset_freshness_status or "").strip() or None
            ),
            "ors_engine_image": str((ors.engine_manifest or {}).get("compose_image") or "").strip() or None,
            "ors_graph_build_date": str(((ors.engine_manifest or {}).get("graph_build_info") or {}).get("graph_build_date") or "").strip() or None,
            "ors_graph_osm_date": str(((ors.engine_manifest or {}).get("graph_build_info") or {}).get("osm_date") or "").strip() or None,
            "ors_graph_file_count": int(as_float((ors.engine_manifest or {}).get("graph_file_count"), 0.0)) if isinstance(ors.engine_manifest, dict) else None,
            "ors_graph_total_bytes": int(as_float((ors.engine_manifest or {}).get("graph_total_bytes"), 0.0)) if isinstance(ors.engine_manifest, dict) else None,
            "ors_graph_listing_digest": str((ors.engine_manifest or {}).get("graph_listing_digest") or "").strip() or None,
            "ors_distance_km": ors.metrics["distance_km"],
            "ors_duration_s": ors.metrics["duration_s"],
            "ors_monetary_cost": ors.metrics["monetary_cost"],
            "ors_emissions_kg": ors.metrics["emissions_kg"],
            "dccs_frontier_recall_at_budget": None,
            "dccs_corridor_family_recall": None,
            "certificate_winner_route_id": None,
            "selector_certificate_disagreement": False,
            "refc_stress_world_fraction": None,
            "voi_realized_certificate_lift": None,
            "voi_realized_runner_up_gap_lift": None,
            "voi_realized_margin_lift": None,
            "voi_realized_frontier_gain": None,
            "voi_realized_runtime_delta_ms": None,
            "voi_productive_action_count": None,
            "voi_nonproductive_action_count": None,
            "productive_voi_action_rate": None,
            "refresh_first_productive": None,
            "refresh_resolution_honest": None,
            "refresh_resolution_reason": None,
            "weighted_margin_gain_vs_v0": None,
            "balanced_gain_delta_vs_v0_score": None,
            "duration_gain_vs_v0_s": None,
            "monetary_gain_vs_v0": None,
            "emissions_gain_vs_v0_kg": None,
            "frontier_hypervolume_gain_vs_v0": None,
            "certificate_lift_vs_v0": None,
            "certificate_availability_gain_vs_v0": False,
            "voi_controller_engaged": False,
            "algorithm_runtime_ms": None,
            "baseline_acquisition_runtime_ms": round(osrm.compute_ms + ors.compute_ms, 3),
            "baseline_runtime_share": 1.0 if (osrm.compute_ms + ors.compute_ms) > 0.0 else None,
            "runtime_ratio_vs_osrm": None,
            "runtime_ratio_vs_ors": None,
            "algorithm_runtime_ratio_vs_osrm": None,
            "algorithm_runtime_ratio_vs_ors": None,
            "runtime_gap_vs_osrm_ms": None,
            "runtime_gap_vs_ors_ms": None,
            "algorithm_runtime_gap_vs_osrm_ms": None,
            "algorithm_runtime_gap_vs_ors_ms": None,
            "row_local_warmup_ms": None,
            "warmup_amortized_ms": None,
            "warmup_overhead_share": None,
            "global_startup_overhead_ms": global_startup_overhead_value,
            "global_startup_share_of_algorithm": None,
            "runtime_per_refined_candidate_ms": None,
            "runtime_per_frontier_member_ms": None,
            "memory_per_refined_candidate_mb": None,
            "quality_per_second": None,
            "route_improvement_per_second": None,
            "frontier_gain_per_ms": None,
            "certificate_gain_per_world": None,
            "controller_cost_per_certificate_point": None,
            "cache_reuse_ratio": None,
            "baseline_identity_verified": (
                str(args.ors_baseline_policy).strip().lower() == "local_service"
                and str((ors.engine_manifest or {}).get("identity_status") or ors.asset_freshness_status or "").strip() == "graph_identity_verified"
                and bool(str(osrm.method or "").strip())
            ) if str(args.ors_baseline_policy).strip().lower() == "local_service" else None,
            "ambiguity_alignment": ambiguity_alignment(ambiguity_prior, None),
            "ambiguity_prior_overtrigger": None,
            "controller_activation_on_high_ambiguity": controller_activation_on_high_ambiguity(ambiguity_prior, False),
            "preflight_and_warmup_ms": None,
            "realized_diversity_collapse": False,
            "realized_diversity_collapse_reason": None,
            "realized_raw_corridor_family_count": 0,
            "realized_refined_corridor_family_count": 0,
            "supplemental_challenger_activated": False,
            "supplemental_challenger_source_count": 0,
            "supplemental_challenger_candidate_count": 0,
            "supplemental_challenger_selected_count": 0,
            "supplemental_challenger_budget_used": 0,
            "selected_candidate_source_label": None,
            "selected_candidate_source_engine": None,
            "selected_candidate_source_stage": None,
            "selected_from_supplemental_rescue": False,
            "selected_from_comparator_engine": False,
            "strict_failure_eliminated": None,
            "controller_value_per_second": None,
            "certificate_selective": None,
            "algorithm_runtime_speedup_vs_v0": None,
            "backend_ready_wait_ms": as_float(readiness.get("wait_elapsed_ms"), float("nan")) if isinstance(readiness, Mapping) else None,
            "backend_ready_probe_ms": as_float(readiness.get("compute_ms"), float("nan")) if isinstance(readiness, Mapping) else None,
            "route_graph_warmup_elapsed_ms": _route_graph_startup_to_ready_ms(readiness if isinstance(readiness, Mapping) else None),
            "preflight_ms": None,
            "process_rss_mb": None,
            "process_vms_mb": None,
            "refc_shortcut_used": False,
            "refc_cache_hits": 0,
            "route_cache_hits": None,
            "route_cache_misses": None,
            "route_cache_hit_rate": None,
            "graph_k_raw_cache_hit": None,
            "graph_low_ambiguity_fast_path": None,
            "graph_supported_ambiguity_fast_fallback": None,
            "route_state_cache_hits": None,
            "route_state_cache_misses": None,
            "route_state_cache_hit_rate": None,
            "baseline_osrm_ms": osrm.compute_ms,
            "baseline_ors_ms": ors.compute_ms,
            "stage_k_raw_ms": None,
            "stage_k_raw_graph_search_initial_ms": None,
            "stage_k_raw_graph_search_retry_ms": None,
            "stage_k_raw_graph_search_rescue_ms": None,
            "stage_k_raw_graph_search_supplemental_ms": None,
            "stage_k_raw_osrm_fallback_ms": None,
            "stage_dccs_ms": None,
            "stage_refc_ms": None,
            "stage_voi_ms": None,
            "stage_pareto_ms": None,
            "stage_refinement_ms": None,
            "stage_option_build_ms": None,
            "option_build_reuse_rate": None,
            "option_build_cache_hits": None,
            "option_build_rebuild_count": None,
            "option_build_cache_hit_rate": None,
            "option_build_cache_savings_ms_per_row": None,
            "search_completeness_score": None,
            "search_completeness_gap": None,
            "prior_support_strength": None,
            "pending_challenger_mass": None,
            "best_pending_flip_probability": None,
            "corridor_family_recall": None,
            "frontier_recall_at_budget": None,
            "credible_search_uncertainty": None,
            "credible_evidence_uncertainty": None,
            "supported_hard_case": None,
            "evidence_first_engagement": None,
            "evidence_only_engagement": None,
            "first_controller_action_kind": None,
            "voi_dccs_cache_hits": None,
            "voi_dccs_cache_misses": None,
            "voi_dccs_cache_hit_rate": None,
            "comparator_independent": None,
            "ambiguity_absolute_error": None,
            "supported_ambiguity_alignment": None,
            "stage_supplemental_rescue_ms": None,
            "ors_snapshot_mode": args.ors_snapshot_mode,
            "ors_snapshot_used": ors.snapshot_used,
            "ors_snapshot_recorded_at": ors.snapshot_recorded_at,
            "ors_snapshot_request_hash": ors.snapshot_request_hash,
            "ors_snapshot_response_hash": ors.snapshot_response_hash,
            "ors_snapshot_provider_mode": ors.provider_mode,
            "artifact_complete": False,
            "artifact_status": "failed",
            "artifact_missing": json.dumps(list(artifact_missing), sort_keys=True),
            "evidence_policy": STRICT_EVIDENCE_POLICY,
            "route_evidence_ok": False,
            "route_evidence_status": "failed",
            "route_evidence_issues": json.dumps([{"reason_code": failure_reason}], sort_keys=True),
            "failure_reason": failure_reason,
        }
    )
    row["observed_ambiguity_index"] = _observed_ambiguity_index(row)
    row["ambiguity_alignment"] = ambiguity_alignment(ambiguity_prior, row["observed_ambiguity_index"])
    row["ambiguity_absolute_error"] = ambiguity_absolute_error(ambiguity_prior, row["observed_ambiguity_index"])
    row["supported_ambiguity_alignment"] = supported_ambiguity_alignment(
        ambiguity_prior,
        row["observed_ambiguity_index"],
        confidence=row.get("od_ambiguity_confidence"),
        support_ratio=row.get("od_ambiguity_support_ratio"),
        source_mix_count=row.get("od_ambiguity_source_mix_count"),
    )
    row["cohort_label"] = _cohort_label(row)
    row["certificate_selective"] = None
    observed_ambiguity = as_float(row.get("observed_ambiguity_index"), float("nan"))
    row["ambiguity_prior_overtrigger"] = bool(
        ambiguity_prior is not None
        and math.isfinite(observed_ambiguity)
        and float(ambiguity_prior) >= 0.45
        and observed_ambiguity <= 0.08
    )
    row["controller_stress_row"] = _is_controller_stress_row(row)
    row["hard_case"] = _is_hard_case_row(row)
    row["supported_hard_case"] = bool(row.get("hard_case") and row.get("supported_hard_case"))
    return row


def _finalize_cross_variant_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[_row_identity_key(row)][str(row.get("variant_id") or "")] = row
    for od_rows in grouped.values():
        refc_row = od_rows.get("B")
        voi_row = od_rows.get("C")
        legacy_row = od_rows.get("V0")
        if refc_row and voi_row:
            current_voi_lift = as_float(voi_row.get("voi_realized_certificate_lift"), float("nan"))
            refc_certificate = as_float(refc_row.get("certificate"), float("nan"))
            voi_certificate = as_float(voi_row.get("certificate"), float("nan"))
            if (
                not math.isfinite(current_voi_lift)
                and math.isfinite(refc_certificate)
                and math.isfinite(voi_certificate)
            ):
                voi_row["voi_realized_certificate_lift"] = round(voi_certificate - refc_certificate, 6)
            refc_runner_up_gap = as_float(refc_row.get("certificate_runner_up_gap"), float("nan"))
            voi_runner_up_gap = as_float(voi_row.get("certificate_runner_up_gap"), float("nan"))
            if math.isfinite(refc_runner_up_gap) and math.isfinite(voi_runner_up_gap):
                voi_row["voi_realized_runner_up_gap_lift"] = round(voi_runner_up_gap - refc_runner_up_gap, 6)
            refc_nominal_margin = as_float(refc_row.get("nominal_winner_margin"), float("nan"))
            voi_nominal_margin = as_float(voi_row.get("nominal_winner_margin"), float("nan"))
            if math.isfinite(refc_nominal_margin) and math.isfinite(voi_nominal_margin):
                voi_row["voi_realized_margin_lift"] = round(voi_nominal_margin - refc_nominal_margin, 6)
            refc_frontier = as_float(refc_row.get("frontier_hypervolume"), float("nan"))
            voi_frontier = as_float(voi_row.get("frontier_hypervolume"), float("nan"))
            if math.isfinite(refc_frontier) and math.isfinite(voi_frontier):
                voi_row["voi_realized_frontier_gain"] = round(voi_frontier - refc_frontier, 6)
        if legacy_row and voi_row:
            legacy_runtime = as_float(legacy_row.get("algorithm_runtime_ms"), float("nan"))
            voi_runtime = as_float(voi_row.get("algorithm_runtime_ms"), float("nan"))
            if math.isfinite(legacy_runtime) and math.isfinite(voi_runtime):
                voi_row["voi_realized_runtime_delta_ms"] = round(voi_runtime - legacy_runtime, 6)
        if voi_row:
            voi_row["controller_cost_per_certificate_point"] = controller_cost_per_certificate_point(
                as_float(voi_row.get("stage_voi_ms"), float("nan")),
                as_float(voi_row.get("voi_realized_certificate_lift"), float("nan")),
            )
        if legacy_row:
            legacy_runtime = as_float(legacy_row.get("algorithm_runtime_ms"), float("nan"))
            legacy_failed = bool(str(legacy_row.get("failure_reason") or "").strip())
            legacy_metrics = {
                "duration_s": as_float(legacy_row.get("selected_duration_s"), float("nan")),
                "monetary_cost": as_float(legacy_row.get("selected_monetary_cost"), float("nan")),
                "emissions_kg": as_float(legacy_row.get("selected_emissions_kg"), float("nan")),
            }
            for variant_id, row in od_rows.items():
                legacy_total_runtime = as_float(legacy_row.get("runtime_ms"), float("nan"))
                current_total_runtime = as_float(row.get("runtime_ms"), float("nan"))
                current_runtime = as_float(row.get("algorithm_runtime_ms"), float("nan"))
                if variant_id == "V0":
                    row["algorithm_runtime_speedup_vs_v0"] = 0.0 if math.isfinite(legacy_runtime) else None
                    row["runtime_speedup_vs_v0"] = 0.0 if math.isfinite(legacy_total_runtime) else None
                    row["weighted_margin_vs_v0"] = 0.0
                    row["weighted_win_v0"] = False
                    row["balanced_win_v0"] = False
                    row["dominates_v0"] = False
                    row["runtime_win_v0"] = False if math.isfinite(legacy_total_runtime) else None
                    row["algorithm_runtime_win_v0"] = False if math.isfinite(legacy_runtime) else None
                    row["certificate_availability_gain_vs_v0"] = False
                elif math.isfinite(legacy_runtime) and legacy_runtime > 0.0 and math.isfinite(current_runtime):
                    row["algorithm_runtime_speedup_vs_v0"] = round((legacy_runtime - current_runtime) / legacy_runtime, 6)
                    row["algorithm_runtime_win_v0"] = current_runtime < legacy_runtime
                else:
                    row["algorithm_runtime_speedup_vs_v0"] = None
                    row["algorithm_runtime_win_v0"] = None
                if variant_id != "V0" and math.isfinite(legacy_total_runtime) and legacy_total_runtime > 0.0 and math.isfinite(current_total_runtime):
                    row["runtime_speedup_vs_v0"] = round((legacy_total_runtime - current_total_runtime) / legacy_total_runtime, 6)
                    row["runtime_win_v0"] = current_total_runtime < legacy_total_runtime
                elif variant_id != "V0":
                    row["runtime_speedup_vs_v0"] = None
                    row["runtime_win_v0"] = None
            for variant_id, row in od_rows.items():
                if variant_id == "V0":
                    row["weighted_margin_gain_vs_v0"] = 0.0
                    row["balanced_gain_delta_vs_v0_score"] = 0.0
                    row["duration_gain_vs_v0_s"] = 0.0
                    row["monetary_gain_vs_v0"] = 0.0
                    row["emissions_gain_vs_v0_kg"] = 0.0
                    row["frontier_hypervolume_gain_vs_v0"] = 0.0
                    row["certificate_lift_vs_v0"] = 0.0
                    row["strict_failure_eliminated"] = False
                    continue
                current_metrics = {
                    "duration_s": as_float(row.get("selected_duration_s"), float("nan")),
                    "monetary_cost": as_float(row.get("selected_monetary_cost"), float("nan")),
                    "emissions_kg": as_float(row.get("selected_emissions_kg"), float("nan")),
                }
                if all(math.isfinite(value) for value in (*legacy_metrics.values(), *current_metrics.values())):
                    current_score, legacy_score = pairwise_weighted_sum_score(
                        current_metrics,
                        legacy_metrics,
                        weights=(1.0, 1.0, 1.0),
                    )
                    row["weighted_margin_vs_v0"] = round(legacy_score - current_score, 6)
                    row["weighted_win_v0"] = bool(current_score < legacy_score)
                    row["balanced_win_v0"] = bool(balanced_gain_score(current_metrics, legacy_metrics) > 0.0)
                    row["dominates_v0"] = dominates(current_metrics, legacy_metrics)
                    row["weighted_margin_gain_vs_v0"] = round(legacy_score - current_score, 6)
                    row["balanced_gain_delta_vs_v0_score"] = round(
                        balanced_gain_score(current_metrics, legacy_metrics),
                        6,
                    )
                    row["duration_gain_vs_v0_s"] = round(legacy_metrics["duration_s"] - current_metrics["duration_s"], 6)
                    row["monetary_gain_vs_v0"] = round(legacy_metrics["monetary_cost"] - current_metrics["monetary_cost"], 6)
                    row["emissions_gain_vs_v0_kg"] = round(legacy_metrics["emissions_kg"] - current_metrics["emissions_kg"], 6)
                if legacy_failed:
                    row["strict_failure_eliminated"] = not bool(str(row.get("failure_reason") or "").strip())
                else:
                    row["strict_failure_eliminated"] = False
                legacy_frontier = as_float(legacy_row.get("frontier_hypervolume"), float("nan"))
                current_frontier = as_float(row.get("frontier_hypervolume"), float("nan"))
                if math.isfinite(legacy_frontier) and math.isfinite(current_frontier):
                    row["frontier_hypervolume_gain_vs_v0"] = round(current_frontier - legacy_frontier, 6)
                legacy_certificate = as_float(legacy_row.get("certificate"), float("nan"))
                current_certificate = as_float(row.get("certificate"), float("nan"))
                row["certificate_availability_gain_vs_v0"] = bool(
                    math.isfinite(current_certificate) and not math.isfinite(legacy_certificate)
                )
                if math.isfinite(current_certificate) and math.isfinite(legacy_certificate):
                    row["certificate_lift_vs_v0"] = round(current_certificate - legacy_certificate, 6)
    return rows


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant_id"])].append(row)
    out: list[dict[str, Any]] = []
    for variant_id in VARIANTS:
        batch = grouped.get(variant_id, [])
        if not batch:
            continue
        certified_rate, certified_denominator = _mean_bool_with_denominator(batch, "certified")
        dominance_win_rate_osrm, dominance_denominator_osrm = _mean_bool_with_denominator(batch, "dominates_osrm")
        dominance_win_rate_ors, dominance_denominator_ors = _mean_bool_with_denominator(batch, "dominates_ors")
        dominance_win_rate_v0, dominance_denominator_v0 = _mean_bool_with_denominator(batch, "dominates_v0")
        runtime_win_rate_v0, runtime_denominator_v0 = _mean_bool_with_denominator(batch, "runtime_win_v0")
        algorithm_runtime_win_rate_v0, algorithm_runtime_denominator_v0 = _mean_bool_with_denominator(batch, "algorithm_runtime_win_v0")
        dominance_win_rate_best_baseline, dominance_denominator_best_baseline = _mean_bool_with_denominator(batch, "dominates_best_baseline")
        weighted_win_rate_osrm, weighted_denominator_osrm = _mean_bool_with_denominator(batch, "weighted_win_osrm")
        weighted_win_rate_ors, weighted_denominator_ors = _mean_bool_with_denominator(batch, "weighted_win_ors")
        weighted_win_rate_v0, weighted_denominator_v0 = _mean_bool_with_denominator(batch, "weighted_win_v0")
        weighted_win_rate_best_baseline, weighted_denominator_best_baseline = _mean_bool_with_denominator(batch, "weighted_win_best_baseline")
        balanced_win_rate_osrm, balanced_denominator_osrm = _mean_bool_with_denominator(batch, "balanced_win_osrm")
        balanced_win_rate_ors, balanced_denominator_ors = _mean_bool_with_denominator(batch, "balanced_win_ors")
        balanced_win_rate_v0, balanced_denominator_v0 = _mean_bool_with_denominator(batch, "balanced_win_v0")
        balanced_win_rate_best_baseline, balanced_denominator_best_baseline = _mean_bool_with_denominator(batch, "balanced_win_best_baseline")
        time_preserving_win_rate_osrm, time_preserving_denominator_osrm = _mean_bool_with_denominator(
            batch,
            "time_preserving_win_osrm",
        )
        time_preserving_win_rate_ors, time_preserving_denominator_ors = _mean_bool_with_denominator(
            batch,
            "time_preserving_win_ors",
        )
        time_preserving_win_rate_best_baseline, time_preserving_denominator_best_baseline = _mean_bool_with_denominator(
            batch,
            "time_preserving_win_best_baseline",
        )
        time_preserving_dominance_rate_osrm, time_preserving_dominance_denominator_osrm = _mean_bool_with_denominator(
            batch,
            "time_preserving_dominance_osrm",
        )
        time_preserving_dominance_rate_ors, time_preserving_dominance_denominator_ors = _mean_bool_with_denominator(
            batch,
            "time_preserving_dominance_ors",
        )
        time_preserving_dominance_rate_best_baseline, time_preserving_dominance_denominator_best_baseline = _mean_bool_with_denominator(
            batch,
            "time_preserving_dominance_best_baseline",
        )
        robust_win_rate_osrm, robust_denominator_osrm = _mean_bool_with_denominator(batch, "robust_win_osrm")
        robust_win_rate_ors, robust_denominator_ors = _mean_bool_with_denominator(batch, "robust_win_ors")
        mean_certificate, mean_certificate_denominator = _mean_numeric_with_denominator(batch, "certificate")
        mean_frontier_hypervolume, mean_frontier_hypervolume_denominator = _mean_numeric_with_denominator(batch, "frontier_hypervolume")
        mean_frontier_diversity_index = _mean_numeric(batch, "frontier_diversity_index")
        mean_frontier_entropy = _mean_numeric(batch, "frontier_entropy")
        mean_od_ambiguity_confidence = _mean_numeric(batch, "od_ambiguity_confidence")
        mean_od_ambiguity_source_count = _mean_numeric(batch, "od_ambiguity_source_count")
        mean_od_ambiguity_source_mix_count = _mean_numeric(batch, "od_ambiguity_source_mix_count")
        mean_od_ambiguity_source_support_strength = _mean_numeric(batch, "od_ambiguity_source_support_strength")
        mean_od_ambiguity_source_entropy = _mean_numeric(batch, "od_ambiguity_source_entropy")
        mean_od_ambiguity_support_ratio = _mean_numeric(batch, "od_ambiguity_support_ratio")
        mean_od_ambiguity_prior_strength = _mean_numeric(batch, "od_ambiguity_prior_strength")
        mean_od_ambiguity_family_density = _mean_numeric(batch, "od_ambiguity_family_density")
        mean_od_ambiguity_margin_pressure = _mean_numeric(batch, "od_ambiguity_margin_pressure")
        mean_od_ambiguity_spread_pressure = _mean_numeric(batch, "od_ambiguity_spread_pressure")
        mean_ambiguity_alignment = _mean_numeric(batch, "ambiguity_alignment")
        mean_ambiguity_absolute_error = _mean_numeric(batch, "ambiguity_absolute_error")
        mean_supported_ambiguity_alignment = _mean_numeric(batch, "supported_ambiguity_alignment")
        mean_ambiguity_budget_prior = _mean_numeric(batch, "ambiguity_budget_prior")
        mean_ambiguity_budget_prior_gap = _mean_numeric(batch, "ambiguity_budget_prior_gap")
        budget_prior_exceeds_raw_rate = _mean_bool(batch, "budget_prior_exceeds_raw")
        ambiguity_prior_realized_correlation = pearson_correlation(
            [
                as_float(
                    row.get("ambiguity_budget_prior"),
                    as_float(row.get("od_ambiguity_index"), float("nan")),
                )
                for row in batch
            ],
            [as_float(row.get("observed_ambiguity_index"), float("nan")) for row in batch],
        )
        mean_od_engine_disagreement_prior = _mean_numeric(batch, "od_engine_disagreement_prior")
        mean_od_hard_case_prior = _mean_numeric(batch, "od_hard_case_prior")
        upstream_nonzero_od_ambiguity_rate = _mean_bool(
            [{**row, "_upstream_nonzero_od_ambiguity": as_float(row.get("od_ambiguity_index"), float("nan")) > 0.0} for row in batch],
            "_upstream_nonzero_od_ambiguity",
        )
        upstream_high_hard_case_prior_rate = _mean_bool(
            [{**row, "_upstream_high_hard_case_prior": as_float(row.get("od_hard_case_prior"), float("nan")) >= 0.35} for row in batch],
            "_upstream_high_hard_case_prior",
        )
        mean_dccs_dc_yield, mean_dccs_dc_yield_denominator = _mean_numeric_with_denominator(batch, "dccs_dc_yield")
        mean_time_to_best_iteration = _mean_numeric(batch, "time_to_best_iteration")
        mean_action_efficiency = _mean_numeric(batch, "action_efficiency")
        mean_refine_cost_prediction_error_deprecated = _mean_numeric(batch, "refine_cost_prediction_error_deprecated")
        mean_refine_cost_mape = _mean_numeric(batch, "refine_cost_mape")
        refine_cost_sample_count_total = sum(int(as_float(row.get("refine_cost_sample_count"), 0.0)) for row in batch)
        refine_cost_positive_sample_count_total = sum(int(as_float(row.get("refine_cost_positive_sample_count"), 0.0)) for row in batch)
        refine_cost_zero_observed_count_total = sum(int(as_float(row.get("refine_cost_zero_observed_count"), 0.0)) for row in batch)
        mean_refine_cost_mae_ms = _mean_numeric(batch, "refine_cost_mae_ms")
        mean_refine_cost_rank_correlation = _mean_numeric(batch, "refine_cost_rank_correlation")
        mean_runtime_p50_ms = _percentile_numeric(batch, "runtime_ms", 0.50)
        mean_runtime_p90_ms = _percentile_numeric(batch, "runtime_ms", 0.90)
        mean_runtime_p95_ms = _percentile_numeric(batch, "runtime_ms", 0.95)
        mean_algorithm_runtime_p50_ms = _percentile_numeric(batch, "algorithm_runtime_ms", 0.50)
        mean_algorithm_runtime_p90_ms = _percentile_numeric(batch, "algorithm_runtime_ms", 0.90)
        mean_algorithm_runtime_p95_ms = _percentile_numeric(batch, "algorithm_runtime_ms", 0.95)
        mean_baseline_acquisition_runtime_p90_ms = _percentile_numeric(batch, "baseline_acquisition_runtime_ms", 0.90)
        mean_route_request_ms = _mean_numeric(batch, "route_request_ms")
        mean_baseline_osrm_ms = _mean_numeric(batch, "baseline_osrm_ms")
        mean_baseline_ors_ms = _mean_numeric(batch, "baseline_ors_ms")
        mean_stage_k_raw_ms = _mean_numeric(batch, "stage_k_raw_ms")
        mean_stage_k_raw_graph_search_initial_ms = _mean_numeric(batch, "stage_k_raw_graph_search_initial_ms")
        mean_stage_k_raw_graph_search_retry_ms = _mean_numeric(batch, "stage_k_raw_graph_search_retry_ms")
        mean_stage_k_raw_graph_search_rescue_ms = _mean_numeric(batch, "stage_k_raw_graph_search_rescue_ms")
        mean_stage_k_raw_graph_search_supplemental_ms = _mean_numeric(batch, "stage_k_raw_graph_search_supplemental_ms")
        mean_stage_k_raw_osrm_fallback_ms = _mean_numeric(batch, "stage_k_raw_osrm_fallback_ms")
        mean_stage_dccs_ms = _mean_numeric(batch, "stage_dccs_ms")
        mean_stage_refinement_ms = _mean_numeric(batch, "stage_refinement_ms")
        mean_stage_pareto_ms = _mean_numeric(batch, "stage_pareto_ms")
        mean_stage_refc_ms = _mean_numeric(batch, "stage_refc_ms")
        mean_stage_voi_ms = _mean_numeric(batch, "stage_voi_ms")
        mean_runtime_ratio_vs_osrm = _mean_numeric(batch, "runtime_ratio_vs_osrm")
        mean_runtime_ratio_vs_ors = _mean_numeric(batch, "runtime_ratio_vs_ors")
        mean_algorithm_runtime_ratio_vs_osrm = _mean_numeric(batch, "algorithm_runtime_ratio_vs_osrm")
        mean_algorithm_runtime_ratio_vs_ors = _mean_numeric(batch, "algorithm_runtime_ratio_vs_ors")
        mean_runtime_gap_vs_osrm_ms = _mean_numeric(batch, "runtime_gap_vs_osrm_ms")
        mean_runtime_gap_vs_ors_ms = _mean_numeric(batch, "runtime_gap_vs_ors_ms")
        mean_algorithm_runtime_gap_vs_osrm_ms = _mean_numeric(batch, "algorithm_runtime_gap_vs_osrm_ms")
        mean_algorithm_runtime_gap_vs_ors_ms = _mean_numeric(batch, "algorithm_runtime_gap_vs_ors_ms")
        mean_row_local_warmup_ms = _mean_numeric(batch, "row_local_warmup_ms")
        mean_warmup_overhead_share = _mean_numeric(batch, "warmup_overhead_share")
        mean_global_startup_overhead_ms = _mean_numeric(batch, "global_startup_overhead_ms")
        warmup_amortized_ms = runtime_per_unit(mean_global_startup_overhead_ms, len(batch))
        mean_global_startup_share_of_algorithm = _mean_numeric(batch, "global_startup_share_of_algorithm")
        mean_runtime_per_refined_candidate_ms = _mean_numeric(batch, "runtime_per_refined_candidate_ms")
        mean_runtime_per_frontier_member_ms = _mean_numeric(batch, "runtime_per_frontier_member_ms")
        mean_memory_per_refined_candidate_mb = _mean_numeric(batch, "memory_per_refined_candidate_mb")
        mean_quality_per_second = _mean_numeric(batch, "quality_per_second")
        mean_route_improvement_per_second = _mean_numeric(batch, "route_improvement_per_second")
        mean_frontier_gain_per_ms = _mean_numeric(batch, "frontier_gain_per_ms")
        mean_certificate_gain_per_world = _mean_numeric(batch, "certificate_gain_per_world")
        mean_controller_cost_per_certificate_point = _mean_numeric(batch, "controller_cost_per_certificate_point")
        mean_top_refresh_gain, mean_top_refresh_gain_denominator = _mean_numeric_with_denominator(batch, "top_refresh_gain")
        mean_top_fragility_mass, mean_top_fragility_mass_denominator = _mean_numeric_with_denominator(batch, "top_fragility_mass")
        mean_competitor_pressure, mean_competitor_pressure_denominator = _mean_numeric_with_denominator(batch, "competitor_pressure")
        mean_initial_refc_top_vor, mean_initial_refc_top_vor_denominator = _mean_numeric_with_denominator(
            batch,
            "initial_refc_top_vor",
        )
        mean_final_refc_top_vor, mean_final_refc_top_vor_denominator = _mean_numeric_with_denominator(
            batch,
            "final_refc_top_vor",
        )
        mean_initial_winner_fragility_mass, mean_initial_winner_fragility_mass_denominator = _mean_numeric_with_denominator(
            batch,
            "initial_winner_fragility_mass",
        )
        mean_final_winner_fragility_mass, mean_final_winner_fragility_mass_denominator = _mean_numeric_with_denominator(
            batch,
            "final_winner_fragility_mass",
        )
        initial_winner_fragility_nonzero_rate, initial_winner_fragility_nonzero_denominator = _mean_bool_with_denominator(
            batch,
            "initial_winner_fragility_nonzero",
        )
        winner_fragility_nonzero_rate, winner_fragility_nonzero_denominator = _mean_bool_with_denominator(
            batch,
            "winner_fragility_nonzero",
        )
        initial_refc_top_vor_positive_rate, initial_refc_top_vor_positive_denominator = _mean_bool_with_denominator(
            batch,
            "initial_refc_top_vor_positive",
        )
        refc_top_vor_positive_rate, refc_top_vor_positive_denominator = _mean_bool_with_denominator(
            batch,
            "refc_top_vor_positive",
        )
        refresh_signal_persistence_rate, refresh_signal_persistence_denominator = _mean_bool_with_denominator(
            batch,
            "refresh_signal_persistent",
        )
        refresh_first_productive_rate, refresh_first_productive_denominator = _mean_bool_with_denominator(
            batch,
            "refresh_first_productive",
        )
        refresh_resolution_honesty_rate, refresh_resolution_honesty_denominator = _mean_bool_with_denominator(
            batch,
            "refresh_resolution_honest",
        )
        zero_lift_controller_rows: list[dict[str, Any]] = []
        for row in batch:
            if str(row.get("variant_id") or "") != "C":
                continue
            lift_value = as_float(row.get("voi_realized_certificate_lift"), float("nan"))
            zero_lift_controller_rows.append(
                {
                    "zero_lift_controller_action": (
                        _bool_or_default(row.get("voi_controller_engaged"), False)
                        and not str(row.get("failure_reason") or "").strip()
                        and (not math.isfinite(lift_value) or lift_value <= 0.0)
                    )
                }
            )
        zero_lift_controller_action_rate = _mean_bool(
            zero_lift_controller_rows,
            "zero_lift_controller_action",
        )
        mean_cache_reuse_ratio = _mean_numeric(batch, "cache_reuse_ratio")
        route_state_cache_hit_rate = _mean_numeric(batch, "route_state_cache_hit_rate")
        route_state_cache_hits = _mean_numeric(batch, "route_state_cache_hits")
        route_state_cache_misses = _mean_numeric(batch, "route_state_cache_misses")
        baseline_identity_verified_rate = _mean_bool(batch, "baseline_identity_verified")
        mean_ambiguity_prior_gap = _mean_numeric(batch, "ambiguity_prior_gap")
        ambiguity_prior_top_k_precision_k = max(1, math.ceil(len(batch) * 0.25))
        ambiguity_prior_top_k_prior_values = [row.get("ambiguity_budget_prior") for row in batch]
        ambiguity_prior_top_k_observed_values = [row.get("observed_ambiguity_index") for row in batch]
        ambiguity_prior_top_k_precision_value = ambiguity_prior_top_k_precision(
            ambiguity_prior_top_k_prior_values,
            ambiguity_prior_top_k_observed_values,
            k=ambiguity_prior_top_k_precision_k,
        )
        ambiguity_prior_top_k_precision_denominator = min(
            ambiguity_prior_top_k_precision_k,
            sum(
                1
                for prior, observed in zip(
                    ambiguity_prior_top_k_prior_values,
                    ambiguity_prior_top_k_observed_values,
                    strict=False,
                )
                if math.isfinite(as_float(prior, float("nan")))
                and math.isfinite(as_float(observed, float("nan")))
            ),
        )
        ambiguity_prior_overtrigger_rate_value = ambiguity_prior_overtrigger_rate(
            ambiguity_prior_top_k_prior_values,
            ambiguity_prior_top_k_observed_values,
        )
        realized_diversity_collapse_rate = _mean_bool(batch, "realized_diversity_collapse")
        supplemental_challenger_activation_rate = _mean_bool(batch, "supplemental_challenger_activated")
        mean_supplemental_challenger_selected_count = _mean_numeric(batch, "supplemental_challenger_selected_count")
        selected_from_supplemental_rescue_rate = _mean_bool(batch, "selected_from_supplemental_rescue")
        selected_from_comparator_engine_rate = _mean_bool(batch, "selected_from_comparator_engine")
        preemptive_comparator_activation_rate = _mean_bool(batch, "preemptive_comparator_seeded")
        mean_preemptive_comparator_candidate_count = _mean_numeric(batch, "preemptive_comparator_candidate_count")
        selected_from_preemptive_comparator_seed_rate = _mean_bool(batch, "selected_from_preemptive_comparator_seed")
        comparator_independence_rate = _mean_bool(batch, "comparator_independent")
        mean_controller_value_per_second = _mean_numeric(batch, "controller_value_per_second")
        mean_refc_cache_hits = _mean_numeric(batch, "refc_cache_hits")
        mean_refc_shortcut_rate = _mean_bool(batch, "refc_shortcut_used")
        mean_refc_unique_world_count = _mean_numeric(batch, "refc_unique_world_count")
        mean_refc_world_reuse_rate = _mean_numeric(batch, "refc_world_reuse_rate")
        mean_refc_hard_stress_pack_count = _mean_numeric(batch, "refc_hard_stress_pack_count")
        mean_refc_stress_world_fraction = _mean_numeric(batch, "refc_stress_world_fraction")
        mean_requested_cert_world_count = _mean_numeric(batch, "requested_cert_world_count")
        mean_effective_cert_world_count = _mean_numeric(batch, "effective_cert_world_count")
        mean_world_count_efficiency = _mean_numeric(batch, "world_count_efficiency")
        mean_refc_ms_per_effective_world = _mean_ratio(batch, "stage_refc_ms", "effective_cert_world_count")
        mean_stage_supplemental_rescue_ms = _mean_numeric(batch, "stage_supplemental_rescue_ms")
        mean_stage_preemptive_comparator_seed_ms = _mean_numeric(batch, "stage_preemptive_comparator_seed_ms")
        mean_backend_ready_wait_ms = _mean_numeric(batch, "backend_ready_wait_ms")
        mean_route_graph_warmup_elapsed_ms = _mean_numeric(batch, "route_graph_warmup_elapsed_ms")
        mean_preflight_ms = _mean_numeric(batch, "preflight_ms")
        mean_process_rss_mb = _mean_numeric(batch, "process_rss_mb")
        mean_process_vms_mb = _mean_numeric(batch, "process_vms_mb")
        mean_process_rss_p90_mb = _percentile_numeric(batch, "process_rss_mb", 0.90)
        mean_process_vms_p90_mb = _percentile_numeric(batch, "process_vms_mb", 0.90)
        mean_route_cache_hit_rate = _mean_numeric(batch, "route_cache_hit_rate")
        mean_k_raw_cache_hit_rate = _mean_bool(batch, "graph_k_raw_cache_hit")
        mean_graph_low_ambiguity_fast_path_rate = _mean_bool(batch, "graph_low_ambiguity_fast_path")
        mean_graph_supported_ambiguity_fast_fallback_rate = _mean_bool(
            batch,
            "graph_supported_ambiguity_fast_fallback",
        )
        mean_search_budget_utilization_p90 = _percentile_numeric(batch, "search_budget_utilization", 0.90)
        mean_evidence_budget_utilization_p90 = _percentile_numeric(batch, "evidence_budget_utilization", 0.90)
        mean_voi_action_density = _mean_ratio(batch, "voi_action_count", "iteration_count")
        mean_initial_certificate = _mean_numeric(batch, "initial_certificate")
        initial_certificate_stop_rate = _mean_bool(batch, "initial_certificate_stop")
        unnecessary_voi_refine_rate = _mean_bool(batch, "unnecessary_voi_refine")
        certificate_selective_rows = [row for row in batch if row.get("certificate_selective") is not None]
        certificate_selectivity_rate, certificate_selectivity_denominator = _mean_bool_with_denominator(
            certificate_selective_rows,
            "certificate_selective",
        )
        mean_stage_option_build_ms = _mean_numeric(batch, "stage_option_build_ms")
        mean_option_build_reuse_rate = _mean_numeric(batch, "option_build_reuse_rate")
        mean_option_build_cache_hits = _mean_numeric(batch, "option_build_cache_hits")
        mean_option_build_rebuild_count = _mean_numeric(batch, "option_build_rebuild_count")
        mean_option_build_cache_hit_rate = _mean_numeric(batch, "option_build_cache_hit_rate")
        mean_option_build_cache_savings_ms_per_row = _mean_numeric(batch, "option_build_cache_savings_ms_per_row")
        mean_search_completeness_score = _mean_numeric(batch, "search_completeness_score")
        mean_search_completeness_gap = _mean_numeric(batch, "search_completeness_gap")
        mean_prior_support_strength = _mean_numeric(batch, "prior_support_strength")
        mean_support_richness = _mean_numeric(batch, "support_richness")
        mean_ambiguity_pressure = _mean_numeric(batch, "ambiguity_pressure")
        mean_pending_challenger_mass = _mean_numeric(batch, "pending_challenger_mass")
        mean_best_pending_flip_probability = _mean_numeric(batch, "best_pending_flip_probability")
        mean_corridor_family_recall = _mean_numeric(batch, "corridor_family_recall")
        mean_frontier_recall_at_budget = _mean_numeric(batch, "frontier_recall_at_budget")
        credible_search_uncertainty_rate = _mean_bool(batch, "credible_search_uncertainty")
        credible_evidence_uncertainty_rate = _mean_bool(batch, "credible_evidence_uncertainty")
        supported_hard_case_rate = _mean_bool(batch, "supported_hard_case")
        evidence_first_engagement_rate = _mean_bool(batch, "evidence_first_engagement")
        evidence_only_engagement_rate = _mean_bool(batch, "evidence_only_engagement")
        productive_voi_action_rows: list[dict[str, Any]] = []
        for row in batch:
            rate_value = row.get("productive_voi_action_rate")
            if rate_value is None:
                productive_count = row.get("voi_productive_action_count")
                nonproductive_count = row.get("voi_nonproductive_action_count")
                if productive_count is None or nonproductive_count is None:
                    continue
                rate_value = productive_action_rate(
                    as_float(productive_count, float("nan")),
                    as_float(productive_count, 0.0) + as_float(nonproductive_count, 0.0),
                )
            rate_value = as_float(rate_value, float("nan"))
            if math.isfinite(rate_value):
                productive_voi_action_rows.append({"productive_voi_action_rate": rate_value})
        productive_voi_action_rate_value, productive_voi_action_denominator = _mean_numeric_with_denominator(
            productive_voi_action_rows,
            "productive_voi_action_rate",
        )
        productive_voi_action_numerator = sum(
            int(as_float(row.get("voi_productive_action_count"), 0.0))
            for row in batch
            if row.get("voi_productive_action_count") is not None
        )
        mean_voi_dccs_cache_hit_rate = _mean_numeric(batch, "voi_dccs_cache_hit_rate")
        mean_voi_dccs_cache_hits = _mean_numeric(batch, "voi_dccs_cache_hits")
        mean_voi_dccs_cache_misses = _mean_numeric(batch, "voi_dccs_cache_misses")
        mean_time_to_certification_ms = _mean_numeric(batch, "time_to_certification_ms")
        mean_controller_shortcut_rate = _mean_bool(batch, "controller_shortcut")
        mean_voi_stop_after_certification_rate = _mean_bool(batch, "voi_stop_after_certification")
        mean_controller_stress_rate = _mean_bool(batch, "controller_stress_row")
        controller_stress_row_count = sum(1 for row in batch if bool(row.get("controller_stress_row")))
        scenario_profile_unavailable_rate = _mean_bool(
            [
                {
                    "_scenario_profile_unavailable": (
                        str(row.get("failure_reason") or "").strip() == "scenario_profile_unavailable"
                    )
                }
                for row in batch
            ],
            "_scenario_profile_unavailable",
        )
        controller_refresh_fallback_activation_rate = _mean_bool(
            batch,
            "controller_refresh_fallback_activated",
        )
        controller_empirical_vs_raw_refresh_disagreement_rate = _mean_bool(
            batch,
            "controller_empirical_vs_raw_refresh_disagreement",
        )
        mean_preflight_and_warmup_ms = _mean_numeric(batch, "preflight_and_warmup_ms")
        mean_runtime_ms, mean_runtime_ms_denominator = _mean_numeric_with_denominator(batch, "runtime_ms")
        controller_activation_high_ambiguity_rate, _ = _mean_bool_with_denominator(batch, "controller_activation_on_high_ambiguity")
        mean_hard_case_certificate_lift_vs_v0 = _mean_numeric([row for row in batch if bool(row.get("hard_case"))], "certificate_lift_vs_v0")
        mean_objective_gain_vs_v0_denominator = sum(1 for row in batch if row.get("weighted_margin_gain_vs_v0") not in (None, ""))
        mean_certificate_lift_vs_v0_denominator = sum(1 for row in batch if row.get("certificate_lift_vs_v0") not in (None, ""))
        certificate_availability_gain_vs_v0_rate = _mean_bool(batch, "certificate_availability_gain_vs_v0")
        success_count = sum(1 for row in batch if not row.get("failure_reason"))
        hard_case_rows = [row for row in batch if bool(row.get("hard_case"))]
        broad_hard_case_certificate_selective_rows = [
            row for row in hard_case_rows if row.get("certificate_selective") is not None
        ]
        broad_hard_case_certificate_selectivity_rate, _ = _mean_bool_with_denominator(
            broad_hard_case_certificate_selective_rows,
            "certificate_selective",
        )
        broad_hard_case_evidence_first_engagement_rate = _mean_bool(
            hard_case_rows,
            "evidence_first_engagement",
        )
        broad_hard_case_productive_rows: list[dict[str, Any]] = []
        for row in hard_case_rows:
            rate_value = row.get("productive_voi_action_rate")
            if rate_value is None:
                productive_count = row.get("voi_productive_action_count")
                nonproductive_count = row.get("voi_nonproductive_action_count")
                if productive_count is None or nonproductive_count is None:
                    continue
                rate_value = productive_action_rate(
                    as_float(productive_count, float("nan")),
                    as_float(productive_count, 0.0) + as_float(nonproductive_count, 0.0),
                )
            rate_value = as_float(rate_value, float("nan"))
            if math.isfinite(rate_value):
                broad_hard_case_productive_rows.append(
                    {"productive_voi_action_rate": rate_value}
                )
        broad_hard_case_productive_voi_action_rate, _ = _mean_numeric_with_denominator(
            broad_hard_case_productive_rows,
            "productive_voi_action_rate",
        )
        broad_hard_case_refc_signal_presence_rate = _mean_bool(
            [
                {
                    "_broad_hard_case_refc_signal_present": bool(
                        row.get("winner_fragility_nonzero") or row.get("refc_top_vor_positive")
                    )
                }
                for row in hard_case_rows
            ],
            "_broad_hard_case_refc_signal_present",
        )
        corpus_kind_counts = Counter(str(row.get("corpus_kind") or "") for row in batch if str(row.get("corpus_kind") or "").strip())
        corpus_group_counts = Counter(str(row.get("corpus_group") or "") for row in batch if str(row.get("corpus_group") or "").strip())
        profile_id_counts = Counter(str(row.get("profile_id") or "") for row in batch if str(row.get("profile_id") or "").strip())
        out.append(
            {
                "variant_id": variant_id,
                "pipeline_mode": VARIANT_PIPELINE_MODE[variant_id],
                "row_count": len(batch),
                "failure_count": sum(1 for row in batch if row.get("failure_reason")),
                "success_count": success_count,
                "success_rate": round(success_count / len(batch), 6),
                "artifact_complete_rate": _mean_bool(batch, "artifact_complete"),
                "route_evidence_ok_rate": _mean_bool(batch, "route_evidence_ok"),
                "certified_rate": certified_rate,
                "certified_denominator": certified_denominator,
                "dominance_win_rate_osrm": dominance_win_rate_osrm,
                "dominance_denominator_osrm": dominance_denominator_osrm,
                "dominance_win_rate_ors": dominance_win_rate_ors,
                "dominance_denominator_ors": dominance_denominator_ors,
                "dominance_win_rate_v0": dominance_win_rate_v0,
                "dominance_denominator_v0": dominance_denominator_v0,
                "runtime_win_rate_v0": runtime_win_rate_v0,
                "runtime_denominator_v0": runtime_denominator_v0,
                "algorithm_runtime_win_rate_v0": algorithm_runtime_win_rate_v0,
                "algorithm_runtime_denominator_v0": algorithm_runtime_denominator_v0,
                "dominance_win_rate_best_baseline": dominance_win_rate_best_baseline,
                "dominance_denominator_best_baseline": dominance_denominator_best_baseline,
                "weighted_win_rate_osrm": weighted_win_rate_osrm,
                "weighted_denominator_osrm": weighted_denominator_osrm,
                "weighted_win_rate_ors": weighted_win_rate_ors,
                "weighted_denominator_ors": weighted_denominator_ors,
                "weighted_win_rate_v0": weighted_win_rate_v0,
                "weighted_denominator_v0": weighted_denominator_v0,
                "weighted_win_rate_best_baseline": weighted_win_rate_best_baseline,
                "weighted_denominator_best_baseline": weighted_denominator_best_baseline,
                "balanced_win_rate_osrm": balanced_win_rate_osrm,
                "balanced_denominator_osrm": balanced_denominator_osrm,
                "balanced_win_rate_ors": balanced_win_rate_ors,
                "balanced_denominator_ors": balanced_denominator_ors,
                "balanced_win_rate_v0": balanced_win_rate_v0,
                "balanced_denominator_v0": balanced_denominator_v0,
                "balanced_win_rate_best_baseline": balanced_win_rate_best_baseline,
                "balanced_denominator_best_baseline": balanced_denominator_best_baseline,
                "time_preserving_win_rate": time_preserving_win_rate_best_baseline,
                "time_preserving_denominator": time_preserving_denominator_best_baseline,
                "time_preserving_win_rate_osrm": time_preserving_win_rate_osrm,
                "time_preserving_denominator_osrm": time_preserving_denominator_osrm,
                "time_preserving_win_rate_ors": time_preserving_win_rate_ors,
                "time_preserving_denominator_ors": time_preserving_denominator_ors,
                "time_preserving_win_rate_best_baseline": time_preserving_win_rate_best_baseline,
                "time_preserving_denominator_best_baseline": time_preserving_denominator_best_baseline,
                "time_preserving_dominance_rate": time_preserving_dominance_rate_best_baseline,
                "time_preserving_dominance_denominator": time_preserving_dominance_denominator_best_baseline,
                "time_preserving_dominance_rate_osrm": time_preserving_dominance_rate_osrm,
                "time_preserving_dominance_denominator_osrm": time_preserving_dominance_denominator_osrm,
                "time_preserving_dominance_rate_ors": time_preserving_dominance_rate_ors,
                "time_preserving_dominance_denominator_ors": time_preserving_dominance_denominator_ors,
                "time_preserving_dominance_rate_best_baseline": time_preserving_dominance_rate_best_baseline,
                "time_preserving_dominance_denominator_best_baseline": time_preserving_dominance_denominator_best_baseline,
                "robust_win_rate_osrm": robust_win_rate_osrm,
                "robust_denominator_osrm": robust_denominator_osrm,
                "robust_win_rate_ors": robust_win_rate_ors,
                "robust_denominator_ors": robust_denominator_ors,
                "mean_certificate": mean_certificate,
                "mean_certificate_denominator": mean_certificate_denominator,
                "mean_frontier_hypervolume": mean_frontier_hypervolume,
                "mean_frontier_hypervolume_denominator": mean_frontier_hypervolume_denominator,
                "mean_frontier_coverage_osrm": _mean_numeric(batch, "frontier_coverage_osrm"),
                "mean_frontier_coverage_ors": _mean_numeric(batch, "frontier_coverage_ors"),
                "mean_frontier_count": _mean_numeric(batch, "frontier_count"),
                "nontrivial_frontier_rate": _mean_bool(batch, "nontrivial_frontier"),
                "mean_frontier_diversity_index": mean_frontier_diversity_index,
                "mean_frontier_entropy": mean_frontier_entropy,
                "mean_od_ambiguity_index": _mean_numeric(batch, "od_ambiguity_index"),
                "mean_od_ambiguity_confidence": mean_od_ambiguity_confidence,
                "mean_od_ambiguity_source_count": mean_od_ambiguity_source_count,
                "mean_od_ambiguity_source_mix_count": mean_od_ambiguity_source_mix_count,
                "mean_od_ambiguity_source_support_strength": mean_od_ambiguity_source_support_strength,
                "mean_od_ambiguity_source_entropy": mean_od_ambiguity_source_entropy,
                "mean_od_ambiguity_support_ratio": mean_od_ambiguity_support_ratio,
                "mean_od_ambiguity_prior_strength": mean_od_ambiguity_prior_strength,
                "mean_od_ambiguity_family_density": mean_od_ambiguity_family_density,
                "mean_od_ambiguity_margin_pressure": mean_od_ambiguity_margin_pressure,
                "mean_od_ambiguity_spread_pressure": mean_od_ambiguity_spread_pressure,
                "mean_od_engine_disagreement_prior": mean_od_engine_disagreement_prior,
                "mean_od_hard_case_prior": mean_od_hard_case_prior,
                "mean_ambiguity_budget_prior": mean_ambiguity_budget_prior,
                "mean_ambiguity_budget_prior_gap": mean_ambiguity_budget_prior_gap,
                "budget_prior_exceeds_raw_rate": budget_prior_exceeds_raw_rate,
                "ambiguity_prior_top_k_precision": ambiguity_prior_top_k_precision_value,
                "ambiguity_prior_top_k_precision_k": ambiguity_prior_top_k_precision_k,
                "ambiguity_prior_top_k_precision_denominator": ambiguity_prior_top_k_precision_denominator,
                "ambiguity_prior_overtrigger_rate": ambiguity_prior_overtrigger_rate_value,
                "upstream_nonzero_od_ambiguity_rate": upstream_nonzero_od_ambiguity_rate,
                "upstream_high_hard_case_prior_rate": upstream_high_hard_case_prior_rate,
                "mean_observed_ambiguity_index": _mean_numeric(batch, "observed_ambiguity_index"),
                "mean_ambiguity_alignment": mean_ambiguity_alignment,
                "mean_ambiguity_absolute_error": mean_ambiguity_absolute_error,
                "mean_supported_ambiguity_alignment": mean_supported_ambiguity_alignment,
                "ambiguity_prior_realized_correlation": ambiguity_prior_realized_correlation,
                "mean_nominal_winner_margin": _mean_numeric(batch, "nominal_winner_margin"),
                "mean_near_tie_mass": _mean_numeric(batch, "near_tie_mass"),
                "mean_certificate_margin": _mean_numeric(batch, "certificate_margin"),
                "mean_certificate_runner_up_gap": _mean_numeric(batch, "certificate_runner_up_gap"),
                "mean_fragility_entropy": _mean_numeric(batch, "fragility_entropy"),
                "mean_competitor_turnover_rate": _mean_numeric(batch, "competitor_turnover_rate"),
                "mean_dccs_frontier_recall_at_budget": _mean_numeric(batch, "dccs_frontier_recall_at_budget"),
                "mean_dccs_corridor_family_recall": _mean_numeric(batch, "dccs_corridor_family_recall"),
                "mean_voi_realized_certificate_lift": _mean_numeric(batch, "voi_realized_certificate_lift"),
                "mean_voi_realized_runner_up_gap_lift": _mean_numeric(batch, "voi_realized_runner_up_gap_lift"),
                "mean_voi_realized_margin_lift": _mean_numeric(batch, "voi_realized_margin_lift"),
                "mean_controller_cost_per_certificate_point": mean_controller_cost_per_certificate_point,
                "mean_initial_refc_top_vor": mean_initial_refc_top_vor,
                "mean_initial_refc_top_vor_denominator": mean_initial_refc_top_vor_denominator,
                "mean_final_refc_top_vor": mean_final_refc_top_vor,
                "mean_final_refc_top_vor_denominator": mean_final_refc_top_vor_denominator,
                "mean_initial_winner_fragility_mass": mean_initial_winner_fragility_mass,
                "mean_initial_winner_fragility_mass_denominator": mean_initial_winner_fragility_mass_denominator,
                "mean_final_winner_fragility_mass": mean_final_winner_fragility_mass,
                "mean_final_winner_fragility_mass_denominator": mean_final_winner_fragility_mass_denominator,
                "initial_winner_fragility_nonzero_rate": initial_winner_fragility_nonzero_rate,
                "initial_winner_fragility_nonzero_denominator": initial_winner_fragility_nonzero_denominator,
                "winner_fragility_nonzero_rate": winner_fragility_nonzero_rate,
                "winner_fragility_nonzero_denominator": winner_fragility_nonzero_denominator,
                "initial_refc_top_vor_positive_rate": initial_refc_top_vor_positive_rate,
                "initial_refc_top_vor_positive_denominator": initial_refc_top_vor_positive_denominator,
                "refc_top_vor_positive_rate": refc_top_vor_positive_rate,
                "refc_top_vor_positive_denominator": refc_top_vor_positive_denominator,
                "refresh_signal_persistence_rate": refresh_signal_persistence_rate,
                "refresh_signal_persistence_denominator": refresh_signal_persistence_denominator,
                "refresh_first_productive_rate": refresh_first_productive_rate,
                "refresh_first_productive_denominator": refresh_first_productive_denominator,
                "refresh_resolution_honesty_rate": refresh_resolution_honesty_rate,
                "refresh_resolution_honesty_denominator": refresh_resolution_honesty_denominator,
                "mean_top_refresh_gain": mean_top_refresh_gain,
                "mean_top_refresh_gain_denominator": mean_top_refresh_gain_denominator,
                "mean_top_fragility_mass": mean_top_fragility_mass,
                "mean_top_fragility_mass_denominator": mean_top_fragility_mass_denominator,
                "mean_competitor_pressure": mean_competitor_pressure,
                "mean_competitor_pressure_denominator": mean_competitor_pressure_denominator,
                "zero_lift_controller_action_rate": zero_lift_controller_action_rate,
                "mean_route_improvement_per_second": mean_route_improvement_per_second,
                "mean_time_to_best_iteration": mean_time_to_best_iteration,
                "mean_action_efficiency": mean_action_efficiency,
                "refine_cost_prediction_error_deprecated": mean_refine_cost_prediction_error_deprecated,
                "refine_cost_mape": mean_refine_cost_mape,
                "refine_cost_mae_ms": mean_refine_cost_mae_ms,
                "refine_cost_rank_correlation": mean_refine_cost_rank_correlation,
                "refine_cost_sample_count": refine_cost_sample_count_total,
                "refine_cost_positive_sample_count": refine_cost_positive_sample_count_total,
                "refine_cost_zero_observed_count": refine_cost_zero_observed_count_total,
                "mean_runtime_p50_ms": mean_runtime_p50_ms,
                "mean_runtime_p90_ms": mean_runtime_p90_ms,
                "mean_runtime_p95_ms": mean_runtime_p95_ms,
                "mean_algorithm_runtime_p50_ms": mean_algorithm_runtime_p50_ms,
                "mean_algorithm_runtime_p90_ms": mean_algorithm_runtime_p90_ms,
                "mean_algorithm_runtime_p95_ms": mean_algorithm_runtime_p95_ms,
                "mean_baseline_acquisition_runtime_p90_ms": mean_baseline_acquisition_runtime_p90_ms,
            "mean_route_request_ms": mean_route_request_ms,
            "mean_baseline_osrm_ms": mean_baseline_osrm_ms,
            "mean_baseline_ors_ms": mean_baseline_ors_ms,
            "mean_stage_k_raw_ms": mean_stage_k_raw_ms,
            "mean_stage_k_raw_graph_search_initial_ms": mean_stage_k_raw_graph_search_initial_ms,
            "mean_stage_k_raw_graph_search_retry_ms": mean_stage_k_raw_graph_search_retry_ms,
            "mean_stage_k_raw_graph_search_rescue_ms": mean_stage_k_raw_graph_search_rescue_ms,
            "mean_stage_k_raw_graph_search_supplemental_ms": mean_stage_k_raw_graph_search_supplemental_ms,
            "mean_stage_k_raw_osrm_fallback_ms": mean_stage_k_raw_osrm_fallback_ms,
            "mean_stage_dccs_ms": mean_stage_dccs_ms,
                "mean_stage_refinement_ms": mean_stage_refinement_ms,
                "mean_stage_pareto_ms": mean_stage_pareto_ms,
                "mean_stage_refc_ms": mean_stage_refc_ms,
                "mean_stage_voi_ms": mean_stage_voi_ms,
                "mean_runtime_ratio_vs_osrm": mean_runtime_ratio_vs_osrm,
                "mean_runtime_ratio_vs_ors": mean_runtime_ratio_vs_ors,
                "mean_algorithm_runtime_ratio_vs_osrm": mean_algorithm_runtime_ratio_vs_osrm,
                "mean_algorithm_runtime_ratio_vs_ors": mean_algorithm_runtime_ratio_vs_ors,
                "mean_runtime_gap_vs_osrm_ms": mean_runtime_gap_vs_osrm_ms,
                "mean_runtime_gap_vs_ors_ms": mean_runtime_gap_vs_ors_ms,
                "mean_algorithm_runtime_gap_vs_osrm_ms": mean_algorithm_runtime_gap_vs_osrm_ms,
                "mean_algorithm_runtime_gap_vs_ors_ms": mean_algorithm_runtime_gap_vs_ors_ms,
                "mean_row_local_warmup_ms": mean_row_local_warmup_ms,
                "warmup_amortized_ms": warmup_amortized_ms,
                "mean_warmup_overhead_share": mean_warmup_overhead_share,
                "mean_global_startup_overhead_ms": mean_global_startup_overhead_ms,
                "mean_global_startup_share_of_algorithm": mean_global_startup_share_of_algorithm,
                "mean_runtime_per_refined_candidate_ms": mean_runtime_per_refined_candidate_ms,
                "mean_runtime_per_frontier_member_ms": mean_runtime_per_frontier_member_ms,
                "mean_memory_per_refined_candidate_mb": mean_memory_per_refined_candidate_mb,
                "mean_quality_per_second": mean_quality_per_second,
                "mean_frontier_gain_per_ms": mean_frontier_gain_per_ms,
                "mean_certificate_gain_per_world": mean_certificate_gain_per_world,
                "mean_cache_reuse_ratio": mean_cache_reuse_ratio,
                "route_state_cache_hit_rate": route_state_cache_hit_rate,
                "route_state_cache_hits": route_state_cache_hits,
                "route_state_cache_misses": route_state_cache_misses,
                "baseline_identity_verified_rate": baseline_identity_verified_rate,
                "mean_ambiguity_prior_gap": mean_ambiguity_prior_gap,
                "realized_diversity_collapse_rate": realized_diversity_collapse_rate,
                "supplemental_challenger_activation_rate": supplemental_challenger_activation_rate,
                "mean_supplemental_challenger_selected_count": mean_supplemental_challenger_selected_count,
                "selected_from_supplemental_rescue_rate": selected_from_supplemental_rescue_rate,
                "selected_from_comparator_engine_rate": selected_from_comparator_engine_rate,
                "preemptive_comparator_activation_rate": preemptive_comparator_activation_rate,
                "mean_preemptive_comparator_candidate_count": mean_preemptive_comparator_candidate_count,
                "selected_from_preemptive_comparator_seed_rate": selected_from_preemptive_comparator_seed_rate,
                "comparator_independence_rate": comparator_independence_rate,
                "strict_failure_elimination_rate": _mean_bool(batch, "strict_failure_eliminated"),
                "mean_controller_value_per_second": mean_controller_value_per_second,
                "mean_refc_shortcut_rate": mean_refc_shortcut_rate,
                "mean_refc_cache_hits": mean_refc_cache_hits,
                "mean_refc_unique_world_count": mean_refc_unique_world_count,
                "mean_refc_world_reuse_rate": mean_refc_world_reuse_rate,
                "mean_refc_hard_stress_pack_count": mean_refc_hard_stress_pack_count,
                "mean_refc_stress_world_fraction": mean_refc_stress_world_fraction,
                "mean_requested_cert_world_count": mean_requested_cert_world_count,
                "mean_effective_cert_world_count": mean_effective_cert_world_count,
                "mean_world_count_efficiency": mean_world_count_efficiency,
                "mean_refc_ms_per_effective_world": mean_refc_ms_per_effective_world,
                "mean_stage_supplemental_rescue_ms": mean_stage_supplemental_rescue_ms,
                "mean_stage_preemptive_comparator_seed_ms": mean_stage_preemptive_comparator_seed_ms,
                "mean_backend_ready_wait_ms": mean_backend_ready_wait_ms,
                "mean_route_graph_warmup_elapsed_ms": mean_route_graph_warmup_elapsed_ms,
                "mean_preflight_ms": mean_preflight_ms,
                "mean_stage_k_raw_ms": mean_stage_k_raw_ms,
                "mean_stage_k_raw_graph_search_initial_ms": mean_stage_k_raw_graph_search_initial_ms,
                "mean_stage_k_raw_graph_search_retry_ms": mean_stage_k_raw_graph_search_retry_ms,
                "mean_stage_k_raw_graph_search_rescue_ms": mean_stage_k_raw_graph_search_rescue_ms,
                "mean_stage_k_raw_graph_search_supplemental_ms": mean_stage_k_raw_graph_search_supplemental_ms,
                "mean_stage_k_raw_osrm_fallback_ms": mean_stage_k_raw_osrm_fallback_ms,
                "mean_process_rss_mb": mean_process_rss_mb,
                "mean_process_vms_mb": mean_process_vms_mb,
                "mean_process_rss_p90_mb": mean_process_rss_p90_mb,
                "mean_process_vms_p90_mb": mean_process_vms_p90_mb,
                "mean_route_cache_hit_rate": mean_route_cache_hit_rate,
                "mean_k_raw_cache_hit_rate": mean_k_raw_cache_hit_rate,
                "mean_graph_low_ambiguity_fast_path_rate": mean_graph_low_ambiguity_fast_path_rate,
                "mean_graph_supported_ambiguity_fast_fallback_rate": (
                    mean_graph_supported_ambiguity_fast_fallback_rate
                ),
                "controller_activation_on_high_ambiguity_rate": controller_activation_high_ambiguity_rate,
                "mean_search_budget_utilization_p90": mean_search_budget_utilization_p90,
                "mean_evidence_budget_utilization_p90": mean_evidence_budget_utilization_p90,
                "mean_voi_action_density": mean_voi_action_density,
                "mean_initial_certificate": mean_initial_certificate,
                "initial_certificate_stop_rate": initial_certificate_stop_rate,
                "unnecessary_voi_refine_rate": unnecessary_voi_refine_rate,
                "mean_stage_option_build_ms": mean_stage_option_build_ms,
                "mean_option_build_reuse_rate": mean_option_build_reuse_rate,
                "mean_option_build_cache_hits": mean_option_build_cache_hits,
                "mean_option_build_rebuild_count": mean_option_build_rebuild_count,
                "mean_option_build_cache_hit_rate": mean_option_build_cache_hit_rate,
                "option_build_cache_savings_ms_per_row": mean_option_build_cache_savings_ms_per_row,
                "mean_search_completeness_score": mean_search_completeness_score,
                "mean_search_completeness_gap": mean_search_completeness_gap,
                "mean_prior_support_strength": mean_prior_support_strength,
                "mean_support_richness": mean_support_richness,
                "mean_ambiguity_pressure": mean_ambiguity_pressure,
                "mean_pending_challenger_mass": mean_pending_challenger_mass,
                "mean_best_pending_flip_probability": mean_best_pending_flip_probability,
                "mean_corridor_family_recall": mean_corridor_family_recall,
                "mean_frontier_recall_at_budget": mean_frontier_recall_at_budget,
                "credible_search_uncertainty_rate": credible_search_uncertainty_rate,
                "credible_evidence_uncertainty_rate": credible_evidence_uncertainty_rate,
                "supported_hard_case_rate": supported_hard_case_rate,
                "evidence_first_engagement_rate": evidence_first_engagement_rate,
                "evidence_only_engagement_rate": evidence_only_engagement_rate,
                "mean_voi_dccs_cache_hit_rate": mean_voi_dccs_cache_hit_rate,
                "voi_dccs_cache_hit_rate": mean_voi_dccs_cache_hit_rate,
                "voi_dccs_cache_hits": mean_voi_dccs_cache_hits,
                "voi_dccs_cache_misses": mean_voi_dccs_cache_misses,
                "mean_time_to_certification_ms": mean_time_to_certification_ms,
                "mean_controller_shortcut_rate": mean_controller_shortcut_rate,
                "mean_voi_stop_after_certification_rate": mean_voi_stop_after_certification_rate,
                "mean_controller_stress_rate": mean_controller_stress_rate,
                "controller_stress_row_count": controller_stress_row_count,
                "scenario_profile_unavailable_rate": scenario_profile_unavailable_rate,
                "strict_live_readiness_pass_rate": None,
                "evaluation_rerun_success_rate": None,
                "controller_refresh_fallback_activation_rate": controller_refresh_fallback_activation_rate,
                "controller_empirical_vs_raw_refresh_disagreement_rate": (
                    controller_empirical_vs_raw_refresh_disagreement_rate
                ),
                "broad_hard_case_certificate_selectivity_rate": (
                    broad_hard_case_certificate_selectivity_rate
                ),
                "broad_hard_case_evidence_first_engagement_rate": (
                    broad_hard_case_evidence_first_engagement_rate
                ),
                "broad_hard_case_productive_voi_action_rate": (
                    broad_hard_case_productive_voi_action_rate
                ),
                "broad_hard_case_refc_signal_presence_rate": (
                    broad_hard_case_refc_signal_presence_rate
                ),
                "mean_preflight_and_warmup_ms": mean_preflight_and_warmup_ms,
                "mean_weighted_margin_gain_vs_v0": _mean_numeric(batch, "weighted_margin_gain_vs_v0"),
                "mean_balanced_gain_delta_vs_v0_score": _mean_numeric(batch, "balanced_gain_delta_vs_v0_score"),
                "mean_duration_gain_vs_v0_s": _mean_numeric(batch, "duration_gain_vs_v0_s"),
                "mean_monetary_gain_vs_v0": _mean_numeric(batch, "monetary_gain_vs_v0"),
                "mean_emissions_gain_vs_v0_kg": _mean_numeric(batch, "emissions_gain_vs_v0_kg"),
                "mean_frontier_hypervolume_gain_vs_v0": _mean_numeric(batch, "frontier_hypervolume_gain_vs_v0"),
                "mean_certificate_lift_vs_v0": _mean_numeric(batch, "certificate_lift_vs_v0"),
                "mean_hard_case_certificate_lift_vs_v0": mean_hard_case_certificate_lift_vs_v0,
                "mean_weighted_margin_vs_osrm": _mean_numeric(batch, "weighted_margin_vs_osrm"),
                "mean_weighted_margin_vs_ors": _mean_numeric(batch, "weighted_margin_vs_ors"),
                "mean_weighted_margin_vs_v0": _mean_numeric(batch, "weighted_margin_vs_v0"),
                "mean_weighted_margin_vs_best_baseline": _mean_numeric(batch, "weighted_margin_vs_best_baseline"),
                "mean_dccs_dc_yield": mean_dccs_dc_yield,
                "mean_dccs_dc_yield_denominator": mean_dccs_dc_yield_denominator,
                "mean_iteration_count": _mean_numeric(batch, "iteration_count"),
                "mean_voi_action_count": _mean_numeric(batch, "voi_action_count"),
                "mean_voi_refine_action_count": _mean_numeric(batch, "voi_refine_action_count"),
                "mean_voi_refresh_action_count": _mean_numeric(batch, "voi_refresh_action_count"),
                "mean_voi_resample_action_count": _mean_numeric(batch, "voi_resample_action_count"),
                "voi_controller_engagement_rate": _mean_bool(batch, "voi_controller_engaged"),
                "selector_certificate_disagreement_rate": _mean_bool(batch, "selector_certificate_disagreement"),
                "certificate_selectivity_rate": certificate_selectivity_rate,
                "productive_voi_action_denominator": productive_voi_action_denominator,
                "productive_voi_action_rate": productive_voi_action_rate_value,
                "certificate_selectivity_denominator": certificate_selectivity_denominator,
                "mean_search_budget_used": _mean_numeric(batch, "search_budget_used"),
                "mean_evidence_budget_used": _mean_numeric(batch, "evidence_budget_used"),
                "mean_search_budget_utilization": _mean_numeric(batch, "search_budget_utilization"),
                "mean_evidence_budget_utilization": _mean_numeric(batch, "evidence_budget_utilization"),
                "mean_algorithm_runtime_ms": _mean_numeric(batch, "algorithm_runtime_ms"),
                "mean_algorithm_runtime_speedup_vs_v0": _mean_numeric(batch, "algorithm_runtime_speedup_vs_v0"),
                "mean_runtime_speedup_vs_v0": _mean_numeric(batch, "runtime_speedup_vs_v0"),
                "mean_objective_gain_vs_v0_denominator": mean_objective_gain_vs_v0_denominator,
                "mean_certificate_lift_vs_v0_denominator": mean_certificate_lift_vs_v0_denominator,
                "certificate_availability_gain_vs_v0_rate": certificate_availability_gain_vs_v0_rate,
                "mean_baseline_acquisition_runtime_ms": _mean_numeric(batch, "baseline_acquisition_runtime_ms"),
                "mean_baseline_runtime_share": _mean_numeric(batch, "baseline_runtime_share"),
                "mean_runtime_ms": mean_runtime_ms,
                "mean_runtime_ms_denominator": mean_runtime_ms_denominator,
                "corpus_kind_counts_json": json.dumps(dict(sorted(corpus_kind_counts.items())), sort_keys=True),
                "corpus_group_counts_json": json.dumps(dict(sorted(corpus_group_counts.items())), sort_keys=True),
                "profile_id_counts_json": json.dumps(dict(sorted(profile_id_counts.items())), sort_keys=True),
            }
        )
    return _augment_summary_rows_with_cohort_metrics(out, rows)


def _methods_appendix(args: argparse.Namespace, *, corpus_hash: str, row_count: int) -> str:
    return "\n".join(
        [
            "# Methods Appendix",
            "",
            f"- Generated rows: {row_count}",
            f"- Corpus hash: `{corpus_hash}`",
            f"- Variants: `{', '.join(f'{key}={VARIANT_PIPELINE_MODE[key]}' for key in VARIANTS)}`",
            f"- Matched search budget: `{args.search_budget}`",
            f"- Evidence budget: `{args.evidence_budget}`",
            f"- Certificate world count: `{args.world_count}`",
            f"- Certificate threshold: `{args.certificate_threshold}`",
            f"- Stop threshold: `{args.tau_stop}`",
            f"- Baseline refinement policy for V0: `{args.baseline_refinement_policy}`",
            f"- Secondary baseline policy: `{args.ors_baseline_policy}`",
            f"- Secondary baseline snapshot mode: `{args.ors_snapshot_mode}`",
            f"- Backend readiness timeout seconds: `{args.ready_timeout_seconds}`",
            f"- Backend readiness poll seconds: `{args.ready_poll_seconds}`",
            f"- Max alternatives: `{args.max_alternatives}`",
            f"- Tolls enabled: `{not bool(args.disable_tolls)}`",
            f"- In-process backend: `{bool(getattr(args, 'in_process_backend', False))}`",
            f"- Strict evidence policy: `{STRICT_EVIDENCE_POLICY}`",
            f"- Backend URL: `{args.backend_url}`",
            f"- OSRM base URL: `{settings.osrm_base_url}`",
            f"- Local ORS base URL: `{settings.ors_base_url}`",
            "",
            "Comparator honesty:",
            "- `V0` is not a full-budget legacy solve in this thesis lane; it is a matched-budget legacy comparator whose expensive refinement stage is capped by the same `search_budget` used by the new pipeline variants.",
            "- The thesis runner passes the same upstream ambiguity/support priors to every variant, including `V0`, so `V0` here should be read as an evaluator-informed legacy comparator rather than an uninformed public-API call with no corpus context.",
            "- The secondary baseline is the self-hosted local openrouteservice engine when `ors_baseline_policy=local_service`; `repo_local` is retained only as an explicit fallback/debug comparator.",
            "- Corpus ambiguity is treated as an upstream route-graph prior when available. Missing corpus priors are not backfilled unless `--auto-enrich-corpus-ambiguity` is explicitly enabled, so thesis runs do not hide extra route-graph probe cost inside evaluation runtime.",
            "- Ambiguity-adaptive budgeting is deterministic and corpus-prior driven: low-ambiguity rows use smaller REFC/VOI budgets, while high-ambiguity rows keep larger search and certification budgets so the controller is still meaningfully stressed.",
            "",
            "Metric definitions:",
            "- Strict dominance compares duration, money, and CO2 with Pareto minimisation.",
            "- Weighted win uses the same fixed selector weights as the route request.",
            "- Balanced win averages clipped relative improvements across the three objectives.",
            "- Frontier metrics include hypervolume, singleton baseline coverage, epsilon indicator, and diversity summaries.",
            "- Ambiguity metrics include corpus-side OD ambiguity, engine-disagreement, and hard-case priors plus an observed ambiguity index derived from realized frontier multiplicity, near-tie mass, winner-margin compression, certificate shortfall, selector disagreement, diversity collapse, and controller intervention.",
            "- Certification metrics include threshold margin, runner-up certificate gap, selector-vs-certificate disagreement, fragility entropy, competitor turnover, ambiguity-prior gap, unique-world reuse, and bounded hard-row stress-pack counts.",
            "- DCCS metrics include score-label correlation, score-ranked frontier recall at used budget, corridor-family recall, realized diversity-collapse rate, supplemental challenger activation, and budget utilization.",
            "- Controller metrics include action counts by type, controller engagement on stressed rows, initial-certificate stop rate, unnecessary VOI refine rate, shortcut rate, time-to-certification, and realized certificate lift.",
        "- Cohort metrics split representative, ambiguity, a broader hard-case ambiguity-pressure cohort, and a stricter controller-stress cohort, then report ambiguity-vs-representative gaps for certificate, runtime, and resource utilization.",
            "- Runtime metrics split algorithm route solve time from baseline acquisition time, report runtime ratios versus OSRM and local ORS, include readiness wait, graph warmup elapsed, warmup-overhead share, preflight-plus-warmup time, process RSS/VMS snapshots, route-cache hit rate, option-build stage cost and reuse, runtime-per-refined-candidate, runtime-per-frontier-member, memory-per-refined-candidate, and p50/p90/p95 distribution summaries plus controller action density.",
            "- Direct incremental-value metrics compare A/B/C against the matched-budget V0 baseline on weighted utility margin, balanced gain, frontier hypervolume, certificate lift, and hard-case certificate lift.",
        ]
    )


def _failure_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        reason = str(row.get("failure_reason") or "").strip()
        if reason:
            counts[reason] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _variant_failure_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        variant_id = str(row.get("variant_id") or "").strip()
        reason = str(row.get("failure_reason") or "").strip()
        if not variant_id or not reason:
            continue
        grouped[variant_id][reason] += 1
    return {
        variant_id: dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])))
        for variant_id, reason_counts in grouped.items()
    }


def _successful_summary_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in summary_rows if int(row.get("success_count") or 0) > 0]


def _failed_summary_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in summary_rows if int(row.get("failure_count") or 0) > 0]


def _cohort_label(row: dict[str, Any]) -> str:
    label = str(row.get("corpus_group") or row.get("corpus_kind") or "").strip().lower()
    if label in {"representative", "ambiguity"}:
        return label
    if label == "ambiguous":
        return "ambiguity"
    return label or "unknown"


def _row_identity_key(row: Mapping[str, Any]) -> str:
    return _digest(
        {
            "od_id": str(row.get("od_id") or "").strip(),
            "profile_id": str(row.get("profile_id") or "").strip(),
            "corpus_group": _cohort_label(dict(row)),
            "origin_lat": str(row.get("origin_lat") or "").strip(),
            "origin_lon": str(row.get("origin_lon") or "").strip(),
            "destination_lat": str(row.get("destination_lat") or "").strip(),
            "destination_lon": str(row.get("destination_lon") or "").strip(),
            "departure_time_utc": str(
                row.get("departure_time_utc")
                or ((row.get("selected_config") or {}).get("departure_time_utc") if isinstance(row.get("selected_config"), dict) else "")
                or ""
            ).strip(),
        }
    )


def _upstream_stress_strength(row: Mapping[str, Any]) -> float:
    return max(
        as_float(row.get("od_ambiguity_index"), 0.0),
        as_float(row.get("od_engine_disagreement_prior"), 0.0),
        as_float(row.get("od_hard_case_prior"), 0.0),
        as_float(row.get("ambiguity_budget_prior"), 0.0),
    )


def _realized_stress_profile(row: Mapping[str, Any]) -> tuple[int, int]:
    major = 0
    minor = 0
    observed_ambiguity_index = as_float(row.get("observed_ambiguity_index"), float("nan"))
    near_tie_mass_value = as_float(row.get("near_tie_mass"), float("nan"))
    nominal_margin = as_float(row.get("nominal_winner_margin"), float("nan"))
    certificate = as_float(row.get("certificate"), float("nan"))
    threshold = as_float(row.get("certificate_threshold"), float("nan"))
    certificate_margin_value = as_float(row.get("certificate_margin"), float("nan"))
    frontier_count = int(as_float(row.get("frontier_count"), 0.0))
    if math.isfinite(observed_ambiguity_index):
        if observed_ambiguity_index >= 0.22:
            major += 1
        elif observed_ambiguity_index >= 0.12:
            minor += 1
    if math.isfinite(near_tie_mass_value):
        if near_tie_mass_value >= 0.18:
            major += 1
        elif near_tie_mass_value >= 0.10:
            minor += 1
    if _bool_or_default(row.get("selector_certificate_disagreement"), False):
        major += 1
    if math.isfinite(certificate) and math.isfinite(threshold):
        if certificate < threshold:
            major += 1
        elif math.isfinite(certificate_margin_value) and certificate_margin_value <= 0.08:
            minor += 1
    if _bool_or_default(row.get("realized_diversity_collapse"), False):
        major += 1
    if _bool_or_default(row.get("nontrivial_frontier"), False) and frontier_count > 1:
        minor += 1
    if math.isfinite(nominal_margin):
        if nominal_margin <= 0.16:
            major += 1
        elif nominal_margin <= 0.24:
            minor += 1
    return major, minor


def _resolved_controller_stress_profile(row: Mapping[str, Any]) -> tuple[int, int]:
    major = 0
    minor = 0
    initial_certificate = as_float(row.get("initial_certificate"), float("nan"))
    threshold = as_float(row.get("certificate_threshold"), float("nan"))
    if math.isfinite(initial_certificate) and math.isfinite(threshold):
        if initial_certificate < threshold:
            major += 1
        elif (initial_certificate - threshold) <= 0.06:
            minor += 1
    if _bool_or_default(row.get("initial_winner_fragility_nonzero"), False):
        minor += 1
    if _bool_or_default(row.get("initial_refc_top_vor_positive"), False):
        minor += 1
    lift_value = as_float(row.get("voi_realized_certificate_lift"), float("nan"))
    if math.isfinite(lift_value):
        if lift_value >= 0.12:
            major += 1
        elif lift_value >= 0.04:
            minor += 1
    time_to_certification = as_float(row.get("time_to_certification_ms"), float("nan"))
    if math.isfinite(time_to_certification) and time_to_certification > 0.0 and (major + minor) > 0:
        minor += 1
    return major, minor


def _is_controller_stress_row(row: dict[str, Any]) -> bool:
    if not _bool_or_default(row.get("voi_controller_engaged"), False):
        return False
    if int(as_float(row.get("voi_action_count"), 0.0)) <= 0:
        return False
    upstream_strength = _upstream_stress_strength(row)
    major_signals, minor_signals = _realized_stress_profile(row)
    resolved_major_signals, resolved_minor_signals = _resolved_controller_stress_profile(row)
    if _bool_or_default(row.get("unnecessary_voi_refine"), False):
        if upstream_strength < 0.55 and resolved_major_signals == 0 and resolved_minor_signals < 2:
            return False
    major_signals += resolved_major_signals
    minor_signals += resolved_minor_signals
    if major_signals >= 1:
        return True
    if upstream_strength >= 0.50 and (major_signals + minor_signals) >= 1:
        return True
    return upstream_strength >= 0.40 and minor_signals >= 2


def _is_hard_case_row(row: dict[str, Any]) -> bool:
    if _is_controller_stress_row(row):
        return True
    upstream_strength = _upstream_stress_strength(row)
    major_signals, minor_signals = _realized_stress_profile(row)
    if major_signals >= 1:
        return True
    if upstream_strength >= 0.65 and (major_signals + minor_signals) >= 1:
        return True
    return upstream_strength >= 0.50 and (major_signals + minor_signals) >= 2


def _cohort_rows(rows: list[dict[str, Any]], cohort_label: str) -> list[dict[str, Any]]:
    if cohort_label == "controller_stress":
        return [row for row in rows if _is_controller_stress_row(row)]
    if cohort_label == "hard_case":
        return [row for row in rows if _is_hard_case_row(row)]
    return [row for row in rows if _cohort_label(row) == cohort_label]


def _finite_delta(lhs: float | None, rhs: float | None) -> float | None:
    left = as_float(lhs, float("nan"))
    right = as_float(rhs, float("nan"))
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return round(left - right, 6)


def _augment_summary_rows_with_cohort_metrics(summary_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("variant_id") or "")].append(row)
    for summary_row in summary_rows:
        variant_id = str(summary_row.get("variant_id") or "")
        batch = grouped.get(variant_id, [])
        representative = _cohort_rows(batch, "representative")
        ambiguity = _cohort_rows(batch, "ambiguity")
        hard_case = _cohort_rows(batch, "hard_case")
        summary_row["mean_hard_case_rate"] = round(len(hard_case) / len(batch), 6) if batch else None
        summary_row["mean_hard_case_certificate"] = _mean_numeric(hard_case, "certificate")
        summary_row["mean_hard_case_runtime_ms"] = _mean_numeric(hard_case, "runtime_ms")
        summary_row["mean_hard_case_frontier_diversity_index"] = _mean_numeric(hard_case, "frontier_diversity_index")
        summary_row["mean_hard_case_action_efficiency"] = _mean_numeric(hard_case, "action_efficiency")
        summary_row["mean_hard_case_search_budget_utilization"] = _mean_numeric(hard_case, "search_budget_utilization")
        summary_row["mean_hard_case_evidence_budget_utilization"] = _mean_numeric(hard_case, "evidence_budget_utilization")
        summary_row["mean_hard_case_controller_engagement_rate"] = _mean_bool(hard_case, "voi_controller_engaged")
        summary_row["mean_certificate_gap_ambiguity_vs_representative"] = _finite_delta(
            _mean_numeric(ambiguity, "certificate"),
            _mean_numeric(representative, "certificate"),
        )
        summary_row["mean_runtime_gap_ambiguity_vs_representative_ms"] = _finite_delta(
            _mean_numeric(ambiguity, "runtime_ms"),
            _mean_numeric(representative, "runtime_ms"),
        )
        summary_row["mean_dccs_dc_yield_gap_ambiguity_vs_representative"] = _finite_delta(
            _mean_numeric(ambiguity, "dccs_dc_yield"),
            _mean_numeric(representative, "dccs_dc_yield"),
        )
        summary_row["mean_time_to_best_gap_ambiguity_vs_representative"] = _finite_delta(
            _mean_numeric(ambiguity, "time_to_best_iteration"),
            _mean_numeric(representative, "time_to_best_iteration"),
        )
        summary_row["mean_search_budget_utilization_gap_ambiguity_vs_representative"] = _finite_delta(
            _mean_numeric(ambiguity, "search_budget_utilization"),
            _mean_numeric(representative, "search_budget_utilization"),
        )
        summary_row["mean_evidence_budget_utilization_gap_ambiguity_vs_representative"] = _finite_delta(
            _mean_numeric(ambiguity, "evidence_budget_utilization"),
            _mean_numeric(representative, "evidence_budget_utilization"),
        )
    return summary_rows


def _cohort_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant_id"])].append(row)
    out: list[dict[str, Any]] = []
    for variant_id in VARIANTS:
        batch = grouped.get(variant_id, [])
        if not batch:
            continue
        total_rows = len(batch)
        for cohort_label in ("representative", "ambiguity", "hard_case", "controller_stress"):
            cohort_rows = _cohort_rows(batch, cohort_label)
            if not cohort_rows:
                continue
            cohort_summary = _summary_rows(cohort_rows)[0]
            cohort_summary.update(
                {
                    "cohort_label": cohort_label,
                    "cohort_total_row_count": total_rows,
                    "cohort_share_of_variant": round(len(cohort_rows) / total_rows, 6),
                }
            )
            out.append(cohort_summary)
    return out


def _run_validity_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    preflight_summary: Mapping[str, Any] | None,
    readiness_summary: Mapping[str, Any] | None,
    evaluation_rerun_success_rate: float,
) -> dict[str, float | int]:
    scenario_profile_unavailable_count = sum(
        1
        for row in rows
        if str(row.get("failure_reason") or "").strip() == "scenario_profile_unavailable"
    )
    row_count = len(rows)
    strict_live_ok = bool((preflight_summary or {}).get("required_ok")) and bool(
        (((readiness_summary or {}).get("strict_live") or {}) if isinstance(readiness_summary, Mapping) else {}).get("ok")
    )
    return {
        "scenario_profile_unavailable_count": scenario_profile_unavailable_count,
        "scenario_profile_unavailable_rate": (
            round(scenario_profile_unavailable_count / row_count, 6)
            if row_count
            else 0.0
        ),
        "strict_live_readiness_pass_rate": 1.0 if strict_live_ok else 0.0,
        "evaluation_rerun_success_rate": round(float(evaluation_rerun_success_rate), 6),
    }


def _apply_run_level_summary_metrics(
    summary_rows: list[dict[str, Any]],
    *,
    run_validity_metrics: Mapping[str, float | int],
) -> list[dict[str, Any]]:
    for row in summary_rows:
        row["strict_live_readiness_pass_rate"] = run_validity_metrics["strict_live_readiness_pass_rate"]
        row["evaluation_rerun_success_rate"] = run_validity_metrics["evaluation_rerun_success_rate"]
        if row.get("scenario_profile_unavailable_rate") is None:
            row["scenario_profile_unavailable_rate"] = run_validity_metrics["scenario_profile_unavailable_rate"]
    return summary_rows


def _cohort_composition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANTS:
        batch = [row for row in rows if str(row.get("variant_id") or "") == variant_id]
        if not batch:
            continue
        cohort_counts = Counter(str(row.get("cohort_label") or "unknown") for row in batch)
        by_variant[variant_id] = {
            "row_count": len(batch),
            "cohort_counts": dict(sorted(cohort_counts.items())),
            "unique_od_ids": sorted({str(row.get("od_id") or "") for row in batch if str(row.get("od_id") or "").strip()}),
        }
    return {
        "total_row_count": len(rows),
        "by_variant": by_variant,
    }


def _rate_text(value: float | None, denominator: int) -> str:
    if value is None:
        return f"n/a (0/{denominator})"
    numerator = int(round(float(value) * denominator))
    return f"{value} ({numerator}/{denominator})"


def _has_any_numeric_metric(rows: Sequence[Mapping[str, Any]], key: str) -> bool:
    for row in rows:
        value = as_float(row.get(key), float("nan"))
        if math.isfinite(value):
            return True
    return False


def _success_variant_line(row: dict[str, Any], *, include_od_ambiguity: bool) -> str:
    row_count = int(row.get("row_count") or 0)
    success_count = int(row.get("success_count") or 0)
    segments = [
        f"success_rows={success_count}/{row_count}",
        f"artifact_complete_rate={row.get('artifact_complete_rate')}",
        f"route_evidence_ok_rate={row.get('route_evidence_ok_rate')}",
        f"weighted_win_best_baseline={_rate_text(row.get('weighted_win_rate_best_baseline'), int(row.get('weighted_denominator_best_baseline') or 0))}",
        f"balanced_win_best_baseline={_rate_text(row.get('balanced_win_rate_best_baseline'), int(row.get('balanced_denominator_best_baseline') or 0))}",
        f"weighted_win_osrm={_rate_text(row.get('weighted_win_rate_osrm'), int(row.get('weighted_denominator_osrm') or 0))}",
        f"weighted_win_ors={_rate_text(row.get('weighted_win_rate_ors'), int(row.get('weighted_denominator_ors') or 0))}",
        f"balanced_win_osrm={_rate_text(row.get('balanced_win_rate_osrm'), int(row.get('balanced_denominator_osrm') or 0))}",
        f"balanced_win_ors={_rate_text(row.get('balanced_win_rate_ors'), int(row.get('balanced_denominator_ors') or 0))}",
        f"time_preserving_win={_rate_text(row.get('time_preserving_win_rate'), int(row.get('time_preserving_denominator') or 0))}",
        f"time_preserving_win_osrm={_rate_text(row.get('time_preserving_win_rate_osrm'), int(row.get('time_preserving_denominator_osrm') or 0))}",
        f"time_preserving_win_ors={_rate_text(row.get('time_preserving_win_rate_ors'), int(row.get('time_preserving_denominator_ors') or 0))}",
        f"mean_weighted_margin_vs_osrm={row.get('mean_weighted_margin_vs_osrm')}",
        f"mean_weighted_margin_vs_best_baseline={row.get('mean_weighted_margin_vs_best_baseline')}",
        f"dominance_win_best_baseline={_rate_text(row.get('dominance_win_rate_best_baseline'), int(row.get('dominance_denominator_best_baseline') or 0))}",
    ]
    if include_od_ambiguity:
        segments.append(f"mean_od_ambiguity_index={row.get('mean_od_ambiguity_index')}")
    segments.extend(
        [
            f"upstream_nonzero_od_ambiguity_rate={row.get('upstream_nonzero_od_ambiguity_rate')}",
            f"upstream_high_hard_case_prior_rate={row.get('upstream_high_hard_case_prior_rate')}",
            f"mean_observed_ambiguity_index={row.get('mean_observed_ambiguity_index')}",
            f"mean_ambiguity_alignment={row.get('mean_ambiguity_alignment')}",
            f"ambiguity_prior_realized_correlation={row.get('ambiguity_prior_realized_correlation')}",
            f"mean_ambiguity_prior_gap={row.get('mean_ambiguity_prior_gap')}",
            f"ambiguity_prior_top_k_precision={row.get('ambiguity_prior_top_k_precision')} (k={row.get('ambiguity_prior_top_k_precision_k')}, n={row.get('ambiguity_prior_top_k_precision_denominator')})",
            f"ambiguity_prior_overtrigger_rate={row.get('ambiguity_prior_overtrigger_rate')}",
            f"mean_frontier_count={row.get('mean_frontier_count')}",
            f"nontrivial_frontier_rate={row.get('nontrivial_frontier_rate')}",
            f"mean_nominal_winner_margin={row.get('mean_nominal_winner_margin')}",
            f"mean_near_tie_mass={row.get('mean_near_tie_mass')}",
            f"mean_certificate={row.get('mean_certificate')} (n={row.get('mean_certificate_denominator')})",
            f"mean_certificate_margin={row.get('mean_certificate_margin')}",
            f"mean_hard_case_rate={row.get('mean_hard_case_rate')}",
            f"mean_hard_case_certificate={row.get('mean_hard_case_certificate')}",
            f"mean_controller_stress_rate={row.get('mean_controller_stress_rate')}",
            f"mean_certificate_gap_ambiguity_vs_representative={row.get('mean_certificate_gap_ambiguity_vs_representative')}",
            f"certificate_selectivity_rate={row.get('certificate_selectivity_rate')} (n={row.get('certificate_selectivity_denominator')})",
            f"mean_runtime_gap_ambiguity_vs_representative_ms={row.get('mean_runtime_gap_ambiguity_vs_representative_ms')}",
            f"mean_hard_case_action_efficiency={row.get('mean_hard_case_action_efficiency')}",
            f"selector_certificate_disagreement_rate={row.get('selector_certificate_disagreement_rate')}",
            f"mean_dccs_frontier_recall_at_budget={row.get('mean_dccs_frontier_recall_at_budget')}",
            f"mean_top_refresh_gain={row.get('mean_top_refresh_gain')} (n={row.get('mean_top_refresh_gain_denominator')})",
            f"mean_top_fragility_mass={row.get('mean_top_fragility_mass')} (n={row.get('mean_top_fragility_mass_denominator')})",
            f"mean_competitor_pressure={row.get('mean_competitor_pressure')} (n={row.get('mean_competitor_pressure_denominator')})",
            f"refresh_first_productive_rate={row.get('refresh_first_productive_rate')} (n={row.get('refresh_first_productive_denominator')})",
            f"refresh_resolution_honesty_rate={row.get('refresh_resolution_honesty_rate')} (n={row.get('refresh_resolution_honesty_denominator')})",
            f"mean_voi_realized_certificate_lift={row.get('mean_voi_realized_certificate_lift')}",
            f"mean_voi_realized_runner_up_gap_lift={row.get('mean_voi_realized_runner_up_gap_lift')}",
            f"mean_voi_realized_margin_lift={row.get('mean_voi_realized_margin_lift')}",
            f"voi_controller_engagement_rate={row.get('voi_controller_engagement_rate')}",
            f"mean_voi_action_count={row.get('mean_voi_action_count')}",
            f"productive_voi_action_rate={row.get('productive_voi_action_rate')} (n={row.get('productive_voi_action_denominator')})",
            f"zero_lift_controller_action_rate={row.get('zero_lift_controller_action_rate')}",
            f"mean_initial_certificate={row.get('mean_initial_certificate')}",
            f"initial_certificate_stop_rate={row.get('initial_certificate_stop_rate')}",
            f"unnecessary_voi_refine_rate={row.get('unnecessary_voi_refine_rate')}",
            f"mean_controller_shortcut_rate={row.get('mean_controller_shortcut_rate')}",
            f"mean_voi_stop_after_certification_rate={row.get('mean_voi_stop_after_certification_rate')}",
            f"mean_time_to_certification_ms={row.get('mean_time_to_certification_ms')}",
            f"mean_search_budget_utilization={row.get('mean_search_budget_utilization')}",
            f"mean_evidence_budget_utilization={row.get('mean_evidence_budget_utilization')}",
            f"controller_activation_on_high_ambiguity_rate={row.get('controller_activation_on_high_ambiguity_rate')}",
            f"selected_from_supplemental_rescue_rate={row.get('selected_from_supplemental_rescue_rate')}",
            f"selected_from_comparator_engine_rate={row.get('selected_from_comparator_engine_rate')}",
            f"preemptive_comparator_activation_rate={row.get('preemptive_comparator_activation_rate')}",
            f"selected_from_preemptive_comparator_seed_rate={row.get('selected_from_preemptive_comparator_seed_rate')}",
            f"mean_weighted_margin_gain_vs_v0={row.get('mean_weighted_margin_gain_vs_v0')}",
            f"mean_frontier_hypervolume_gain_vs_v0={row.get('mean_frontier_hypervolume_gain_vs_v0')}",
            f"mean_certificate_lift_vs_v0={row.get('mean_certificate_lift_vs_v0')} (n={row.get('mean_certificate_lift_vs_v0_denominator')})",
            f"certificate_availability_gain_vs_v0_rate={row.get('certificate_availability_gain_vs_v0_rate')}",
            f"mean_algorithm_runtime_ms={row.get('mean_algorithm_runtime_ms')}",
            f"mean_algorithm_runtime_ratio_vs_osrm={row.get('mean_algorithm_runtime_ratio_vs_osrm')}",
            f"mean_algorithm_runtime_ratio_vs_ors={row.get('mean_algorithm_runtime_ratio_vs_ors')}",
            f"mean_algorithm_runtime_gap_vs_osrm_ms={row.get('mean_algorithm_runtime_gap_vs_osrm_ms')}",
            f"mean_algorithm_runtime_gap_vs_ors_ms={row.get('mean_algorithm_runtime_gap_vs_ors_ms')}",
            f"mean_stage_option_build_ms={row.get('mean_stage_option_build_ms')}",
            f"mean_option_build_reuse_rate={row.get('mean_option_build_reuse_rate')}",
            f"mean_preflight_and_warmup_ms={row.get('mean_preflight_and_warmup_ms')}",
            f"mean_runtime_per_refined_candidate_ms={row.get('mean_runtime_per_refined_candidate_ms')}",
            f"mean_runtime_per_frontier_member_ms={row.get('mean_runtime_per_frontier_member_ms')}",
            f"warmup_amortized_ms={row.get('warmup_amortized_ms')}",
            f"mean_route_improvement_per_second={row.get('mean_route_improvement_per_second')}",
            f"mean_controller_cost_per_certificate_point={row.get('mean_controller_cost_per_certificate_point')}",
            f"refine_cost_sample_count={row.get('refine_cost_sample_count')}",
            f"refine_cost_positive_sample_count={row.get('refine_cost_positive_sample_count')}",
            f"refine_cost_zero_observed_count={row.get('refine_cost_zero_observed_count')}",
            f"mean_certificate_gain_per_world={row.get('mean_certificate_gain_per_world')}",
            f"mean_cache_reuse_ratio={row.get('mean_cache_reuse_ratio')}",
            f"baseline_identity_verified_rate={row.get('baseline_identity_verified_rate')}",
            f"mean_refc_shortcut_rate={row.get('mean_refc_shortcut_rate')}",
            f"mean_refc_cache_hits={row.get('mean_refc_cache_hits')}",
            f"mean_refc_unique_world_count={row.get('mean_refc_unique_world_count')}",
            f"mean_refc_world_reuse_rate={row.get('mean_refc_world_reuse_rate')}",
            f"mean_refc_hard_stress_pack_count={row.get('mean_refc_hard_stress_pack_count')}",
            f"mean_refc_stress_world_fraction={row.get('mean_refc_stress_world_fraction')}",
            f"mean_requested_cert_world_count={row.get('mean_requested_cert_world_count')}",
            f"mean_effective_cert_world_count={row.get('mean_effective_cert_world_count')}",
            f"mean_world_count_efficiency={row.get('mean_world_count_efficiency')}",
            f"mean_refc_ms_per_effective_world={row.get('mean_refc_ms_per_effective_world')}",
            f"mean_stage_supplemental_rescue_ms={row.get('mean_stage_supplemental_rescue_ms')}",
            f"mean_stage_preemptive_comparator_seed_ms={row.get('mean_stage_preemptive_comparator_seed_ms')}",
            f"mean_baseline_runtime_share={row.get('mean_baseline_runtime_share')}",
            f"mean_runtime_ms={row.get('mean_runtime_ms')} (n={row.get('mean_runtime_ms_denominator')})",
        ]
    )
    return f"- {row['variant_id']} / {row['pipeline_mode']}: " + ", ".join(segments)


def _failure_variant_line(row: dict[str, Any], *, reason_counts: dict[str, int]) -> str:
    row_count = int(row.get("row_count") or 0)
    failure_count = int(row.get("failure_count") or 0)
    reason_text = ", ".join(f"{reason}={count}" for reason, count in reason_counts.items()) or "none"
    return (
        f"- {row['variant_id']} / {row['pipeline_mode']}: "
        f"failure_rows={failure_count}/{row_count}, "
        f"success_rows={row.get('success_count')}, "
        f"artifact_complete_rate={row.get('artifact_complete_rate')}, "
        f"route_evidence_ok_rate={row.get('route_evidence_ok_rate')}, "
        f"failure_reasons={reason_text}"
    )


def _thesis_report(
    run_id: str,
    summary_rows: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    corpus_hash: str,
    row_count: int,
    ors_baseline_policy: str,
    ors_snapshot_mode: str,
    preflight_summary: dict[str, Any],
    readiness_summary: dict[str, Any],
    baseline_smoke_summary: dict[str, Any],
    output_validation: dict[str, Any],
) -> str:
    success_rows = _successful_summary_rows(summary_rows)
    failed_rows = _failed_summary_rows(summary_rows)
    summary_by_variant = {
        str(row.get("variant_id") or ""): row
        for row in summary_rows
        if str(row.get("variant_id") or "").strip()
    }
    variant_failures = _variant_failure_breakdown(rows)
    include_od_ambiguity = _has_any_numeric_metric(summary_rows, "mean_od_ambiguity_index")
    legacy_refinement_policy = next(
        (
            str(row.get("refinement_selection_policy") or "").strip()
            for row in rows
            if str(row.get("variant_id") or "") == "V0" and str(row.get("refinement_selection_policy") or "").strip()
        ),
        "n/a",
    )
    def _claim_support_for_variants(variant_ids: set[str], *, cohort: str | None = None) -> bool:
        scoped_rows = [
            row for row in rows
            if str(row.get("variant_id") or "") in variant_ids
            and (cohort is None or str(row.get("cohort_label") or "") == cohort)
        ]
        if not scoped_rows:
            return False
        return all(
            not _text_or_none(row.get("failure_reason"))
            and _bool_or_default(row.get("weighted_win_best_baseline"), False)
            and _bool_or_default(row.get("balanced_win_best_baseline"), False)
            and _bool_or_default(row.get("baseline_identity_verified"), False)
            for row in scoped_rows
        )

    def _hard_case_claim_supported() -> bool:
        scoped_rows = [
            row for row in rows
            if str(row.get("variant_id") or "") == "C"
            and str(row.get("cohort_label") or "") == "hard_case"
        ]
        if not scoped_rows:
            return False
        stressed_rows = [row for row in scoped_rows if _bool_or_default(row.get("controller_stress_row"), False)]
        mean_observed_ambiguity = _mean_numeric(stressed_rows or scoped_rows, "observed_ambiguity_index")
        engagement_rate = _mean_bool(stressed_rows or scoped_rows, "voi_controller_engaged")
        return (
            bool(stressed_rows)
            and engagement_rate is not None
            and engagement_rate >= 0.5
            and mean_observed_ambiguity is not None
            and mean_observed_ambiguity >= 0.12
            and _claim_support_for_variants({"C"}, cohort="hard_case")
        )

    def _controller_stress_claim_supported() -> bool:
        scoped_rows = [
            row
            for row in rows
            if str(row.get("variant_id") or "") == "C"
            and _bool_or_default(row.get("controller_stress_row"), False)
        ]
        if len(scoped_rows) < 2:
            return False
        engagement_rate = _mean_bool(scoped_rows, "voi_controller_engaged")
        return (
            engagement_rate is not None
            and engagement_rate >= 0.5
            and _claim_support_for_variants({"C"}, cohort="ambiguity")
        )

    overall_claim_supported = _claim_support_for_variants({"A", "B", "C"})
    representative_claim_supported = _claim_support_for_variants({"A", "B", "C"}, cohort="representative")
    ambiguity_claim_supported = _claim_support_for_variants({"A", "B", "C"}, cohort="ambiguity")
    hard_case_claim_supported = _hard_case_claim_supported()
    controller_stress_claim_supported = _controller_stress_claim_supported()
    lines = [
        "# Thesis Evaluation Report",
        "",
        f"- Evaluation run id: `{run_id}`",
        f"- Corpus hash: `{corpus_hash}`",
        f"- Requested OD rows: `{row_count}`",
        f"- Result rows: `{len(rows)}`",
        f"- Successful rows: `{sum(1 for row in rows if not row.get('failure_reason'))}`",
        f"- Failure rows: `{sum(1 for row in rows if row.get('failure_reason'))}`",
        f"- Secondary baseline policy: `{ors_baseline_policy}`",
        f"- Secondary baseline snapshot mode: `{ors_snapshot_mode}`",
        f"- Repo preflight required ok: `{bool(preflight_summary.get('required_ok'))}`",
        f"- Backend strict ready: `{bool(readiness_summary.get('strict_route_ready'))}`",
        f"- Backend ready wait ms: `{readiness_summary.get('wait_elapsed_ms')}`",
        f"- Backend ready probe ms: `{readiness_summary.get('compute_ms')}`",
        f"- Route-graph warmup elapsed ms: `{_route_graph_startup_to_ready_ms(readiness_summary if isinstance(readiness_summary, Mapping) else None)}`",
        f"- Baseline smoke required ok: `{bool(baseline_smoke_summary.get('required_ok'))}`",
        f"- Validated output artifacts: `{output_validation.get('validated_artifact_count', 0)}`",
        "",
    ]
    failure_breakdown = _failure_breakdown(rows)
    if failure_breakdown:
        lines.append("## Failure Breakdown")
        lines.append("")
        for reason, count in failure_breakdown.items():
            lines.append(f"- `{reason}`: {count}")
        lines.append("")
    lines.append("## Headline Wins")
    lines.append("")
    if success_rows:
        for row in success_rows:
            lines.append(
                f"- {row['variant_id']} / {row['pipeline_mode']}: "
                f"weighted_win_best_baseline={_rate_text(row.get('weighted_win_rate_best_baseline'), int(row.get('weighted_denominator_best_baseline') or 0))}, "
                f"balanced_win_best_baseline={_rate_text(row.get('balanced_win_rate_best_baseline'), int(row.get('balanced_denominator_best_baseline') or 0))}, "
                f"weighted_win_osrm={_rate_text(row.get('weighted_win_rate_osrm'), int(row.get('weighted_denominator_osrm') or 0))}, "
                f"weighted_win_ors={_rate_text(row.get('weighted_win_rate_ors'), int(row.get('weighted_denominator_ors') or 0))}, "
                f"balanced_win_osrm={_rate_text(row.get('balanced_win_rate_osrm'), int(row.get('balanced_denominator_osrm') or 0))}, "
                f"balanced_win_ors={_rate_text(row.get('balanced_win_rate_ors'), int(row.get('balanced_denominator_ors') or 0))}, "
                f"time_preserving_win={_rate_text(row.get('time_preserving_win_rate'), int(row.get('time_preserving_denominator') or 0))}, "
                f"time_preserving_win_osrm={_rate_text(row.get('time_preserving_win_rate_osrm'), int(row.get('time_preserving_denominator_osrm') or 0))}, "
                f"time_preserving_win_ors={_rate_text(row.get('time_preserving_win_rate_ors'), int(row.get('time_preserving_denominator_ors') or 0))}, "
                f"dominance_win_best_baseline={_rate_text(row.get('dominance_win_rate_best_baseline'), int(row.get('dominance_denominator_best_baseline') or 0))}, "
                f"mean_weighted_margin_vs_osrm={row.get('mean_weighted_margin_vs_osrm')}, "
                f"mean_weighted_margin_vs_ors={row.get('mean_weighted_margin_vs_ors')}, "
                f"mean_weighted_margin_vs_best_baseline={row.get('mean_weighted_margin_vs_best_baseline')}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Residual Resolution")
    lines.append("")
    v0_row = summary_by_variant.get("V0")
    a_row = summary_by_variant.get("A")
    b_row = summary_by_variant.get("B")
    c_row = summary_by_variant.get("C")
    if include_od_ambiguity and c_row is not None:
        lines.append(
            "- Upstream ambiguity is now explicit at corpus level: "
            f"`upstream_nonzero_od_ambiguity_rate={c_row.get('upstream_nonzero_od_ambiguity_rate')}`, "
            f"`mean_ambiguity_budget_prior={c_row.get('mean_ambiguity_budget_prior')}`, "
            f"`budget_prior_exceeds_raw_rate={c_row.get('budget_prior_exceeds_raw_rate')}`, "
            f"`mean_od_ambiguity_support_ratio={c_row.get('mean_od_ambiguity_support_ratio')}`, "
            f"`mean_od_ambiguity_source_support_strength={c_row.get('mean_od_ambiguity_source_support_strength')}`, "
            f"`mean_od_ambiguity_source_entropy={c_row.get('mean_od_ambiguity_source_entropy')}`."
        )
    else:
        lines.append("- Upstream ambiguity still depends on realized routing behavior because the corpus carried no explicit ambiguity prior.")
    if v0_row is not None and a_row is not None and b_row is not None and c_row is not None:
        lines.append(
            "- Legacy quality is no longer maxed out relative to the thesis pipeline: "
            f"`V0 weighted_win_best_baseline={_rate_text(v0_row.get('weighted_win_rate_best_baseline'), int(v0_row.get('weighted_denominator_best_baseline') or 0))}` "
            f"vs `A/B/C={_rate_text(a_row.get('weighted_win_rate_best_baseline'), int(a_row.get('weighted_denominator_best_baseline') or 0))}/"
            f"{_rate_text(b_row.get('weighted_win_rate_best_baseline'), int(b_row.get('weighted_denominator_best_baseline') or 0))}/"
            f"{_rate_text(c_row.get('weighted_win_rate_best_baseline'), int(c_row.get('weighted_denominator_best_baseline') or 0))}`, "
            f"and `C mean_weighted_margin_gain_vs_v0={c_row.get('mean_weighted_margin_gain_vs_v0')}`."
        )
        lines.append(
            "- Runtime is improved on aggregate mean but still mixed on matched rows: "
            f"`V0 mean_algorithm_runtime_ms={v0_row.get('mean_algorithm_runtime_ms')} (n={v0_row.get('mean_runtime_ms_denominator')})` vs "
            f"`C mean_algorithm_runtime_ms={c_row.get('mean_algorithm_runtime_ms')} (n={c_row.get('mean_runtime_ms_denominator')})`, "
            f"`C mean_runtime_ms={c_row.get('mean_runtime_ms')}`, "
            f"`C runtime_win_v0={_rate_text(c_row.get('runtime_win_rate_v0'), int(c_row.get('runtime_denominator_v0') or 0))}`, "
            f"`C algorithm_runtime_win_v0={_rate_text(c_row.get('algorithm_runtime_win_rate_v0'), int(c_row.get('algorithm_runtime_denominator_v0') or 0))}`, "
            f"`C mean_runtime_speedup_vs_v0={c_row.get('mean_runtime_speedup_vs_v0')}`, "
            f"`C mean_algorithm_runtime_speedup_vs_v0={c_row.get('mean_algorithm_runtime_speedup_vs_v0')}`."
        )
    else:
        lines.append("- Variant comparison rows were incomplete, so residual-resolution claims could not be evaluated for this run.")
    lines.append("")
    lines.append("## Comparator Honesty")
    lines.append("")
    lines.append("- `V0` is a matched-budget legacy comparator: its expensive refinement is capped by the configured legacy baseline policy rather than being allowed to refine every candidate.")
    lines.append("- In this thesis lane, `V0` also receives the same upstream ambiguity/support priors carried by the corpus rows, so the comparison is between evaluator-informed variants rather than between informed A/B/C requests and an uninformed legacy request.")
    lines.append(f"- The configured V0 baseline refinement policy for this run is `{legacy_refinement_policy}`.")
    lines.append("- The secondary baseline is the self-hosted local openrouteservice engine when `ors_baseline_policy=local_service`; thesis evaluation fails closed if the backend resolves to any other provider mode.")
    lines.append("- This thesis lane does not require a paid ORS API key and does not silently downgrade to repo-local heuristic artifacts when `local_service` is requested.")
    lines.append("- `baseline_identity_verified_rate` reports how often the self-hosted baseline identity was actually provenance-verified in the recorded rows; missing provenance is surfaced as a metric gap rather than treated as a hidden success.")
    lines.append("- `best_baseline` denotes the stronger of the self-hosted OSRM and self-hosted ORS comparators under the same fixed selector weights used for that row.")
    lines.append("- Supplemental diversity rescue is disclosed separately. If a selected winner comes from a supplemental comparator-engine seed, the report surfaces that rate explicitly instead of hiding it inside headline win counts.")
    if include_od_ambiguity:
        lines.append("- The corpus carries upstream ambiguity priors (`od_ambiguity_index`) in addition to the realized-routing ambiguity signals computed from the final frontier/controller behavior.")
    else:
        lines.append("- No upstream ambiguity prior was available in the corpus, so ambiguity claims in this run rely on realized routing behavior only.")
    lines.append("")
    lines.append("## Baseline Smoke")
    lines.append("")
    osrm_smoke = baseline_smoke_summary.get("osrm", {}) if isinstance(baseline_smoke_summary, dict) else {}
    ors_smoke = baseline_smoke_summary.get("ors", {}) if isinstance(baseline_smoke_summary, dict) else {}
    if osrm_smoke:
        if _bool_or_default(osrm_smoke.get("ok"), False):
            lines.append(
                f"- OSRM smoke: ok, method={osrm_smoke.get('method')}, provider_mode={osrm_smoke.get('provider_mode')}, "
                f"compute_ms={osrm_smoke.get('compute_ms')}, duration_s={osrm_smoke.get('duration_s')}, distance_km={osrm_smoke.get('distance_km')}."
            )
        else:
            lines.append(
                f"- OSRM smoke: failed, reason_code={osrm_smoke.get('reason_code')}, message={osrm_smoke.get('message')}."
            )
    if ors_smoke:
        if _bool_or_default(ors_smoke.get("ok"), False):
            lines.append(
                f"- ORS smoke: ok, method={ors_smoke.get('method')}, provider_mode={ors_smoke.get('provider_mode')}, "
                f"baseline_policy={ors_smoke.get('baseline_policy')}, compute_ms={ors_smoke.get('compute_ms')}, "
                f"graph_identity_status={ors_smoke.get('graph_identity_status')}, asset_manifest_hash={ors_smoke.get('asset_manifest_hash')}."
            )
        else:
            lines.append(
                f"- ORS smoke: failed, reason_code={ors_smoke.get('reason_code')}, message={ors_smoke.get('message')}."
            )
    lines.append("- The baseline smoke checks are deterministic route probes run before row processing. They are intended to prove the benchmark faced live self-hosted OSRM and self-hosted ORS rather than weakened fallback comparators.")
    lines.append("")
    lines.append("## Ambiguity Prior")
    lines.append("")
    lines.append("- `mean_od_engine_disagreement_prior` and `mean_od_hard_case_prior` are deterministic upstream heuristics used for budget shaping; they are priors, not claimed ground-truth labels.")
    lines.append("- `mean_ambiguity_budget_prior` is the effective budget-shaping prior used by the thesis runner after applying a deterministic support-aware weighting rule to the raw corpus ambiguity prior and the engine-disagreement / hard-case priors; it is not a plain maximum.")
    lines.append("- `mean_od_ambiguity_confidence`, `mean_od_ambiguity_support_ratio`, and `mean_od_ambiguity_source_count` describe how much upstream evidence supported those priors before route solving began.")
    lines.append("- `mean_od_ambiguity_source_mix_count`, `mean_od_ambiguity_source_support_strength`, and `mean_od_ambiguity_source_entropy` quantify source diversity and corroboration rather than only source count, so the report distinguishes one-source confidence from multi-source support.")
    lines.append("- `mean_od_ambiguity_prior_strength`, `mean_od_ambiguity_family_density`, `mean_od_ambiguity_margin_pressure`, and `mean_od_ambiguity_spread_pressure` capture why a row looked ambiguous upstream: route-family density, proxy winner-margin pressure, and proxy objective spread.")
    lines.append("- `budget_prior_exceeds_raw_rate` shows how often the effective budget prior still exceeded the raw corpus ambiguity prior after that support-aware weighting step.")
    lines.append("- `upstream_nonzero_od_ambiguity_rate` reports how much of the evaluated suite carried a nonzero corpus-side ambiguity prior before any routing was run.")
    lines.append("- `upstream_high_hard_case_prior_rate` reports how often the corpus already flagged hard rows upstream rather than discovering difficulty only after solving.")
    lines.append("")
    if success_rows:
        for row in success_rows:
            segments = []
            if include_od_ambiguity:
                segments.append(f"mean_od_ambiguity_index={row.get('mean_od_ambiguity_index')}")
            segments.extend(
                [
                    f"mean_od_ambiguity_confidence={row.get('mean_od_ambiguity_confidence')}",
                    f"mean_od_ambiguity_source_count={row.get('mean_od_ambiguity_source_count')}",
                    f"mean_od_ambiguity_source_mix_count={row.get('mean_od_ambiguity_source_mix_count')}",
                    f"mean_od_ambiguity_source_support_strength={row.get('mean_od_ambiguity_source_support_strength')}",
                    f"mean_od_ambiguity_source_entropy={row.get('mean_od_ambiguity_source_entropy')}",
                    f"mean_od_ambiguity_support_ratio={row.get('mean_od_ambiguity_support_ratio')}",
                    f"mean_od_ambiguity_prior_strength={row.get('mean_od_ambiguity_prior_strength')}",
                    f"mean_ambiguity_budget_prior={row.get('mean_ambiguity_budget_prior')}",
                    f"mean_ambiguity_budget_prior_gap={row.get('mean_ambiguity_budget_prior_gap')}",
                    f"budget_prior_exceeds_raw_rate={row.get('budget_prior_exceeds_raw_rate')}",
                    f"ambiguity_prior_top_k_precision={row.get('ambiguity_prior_top_k_precision')} (k={row.get('ambiguity_prior_top_k_precision_k')}, n={row.get('ambiguity_prior_top_k_precision_denominator')})",
                    f"ambiguity_prior_overtrigger_rate={row.get('ambiguity_prior_overtrigger_rate')}",
                    f"mean_od_ambiguity_family_density={row.get('mean_od_ambiguity_family_density')}",
                    f"mean_od_ambiguity_margin_pressure={row.get('mean_od_ambiguity_margin_pressure')}",
                    f"mean_od_ambiguity_spread_pressure={row.get('mean_od_ambiguity_spread_pressure')}",
                    f"mean_od_engine_disagreement_prior={row.get('mean_od_engine_disagreement_prior')}",
                    f"mean_od_hard_case_prior={row.get('mean_od_hard_case_prior')}",
                    f"upstream_nonzero_od_ambiguity_rate={row.get('upstream_nonzero_od_ambiguity_rate')}",
                    f"upstream_high_hard_case_prior_rate={row.get('upstream_high_hard_case_prior_rate')}",
                    f"mean_observed_ambiguity_index={row.get('mean_observed_ambiguity_index')}",
                    f"mean_ambiguity_alignment={row.get('mean_ambiguity_alignment')}",
                    f"mean_ambiguity_absolute_error={row.get('mean_ambiguity_absolute_error')}",
                    f"mean_supported_ambiguity_alignment={row.get('mean_supported_ambiguity_alignment')}",
                    f"ambiguity_prior_realized_correlation={row.get('ambiguity_prior_realized_correlation')}",
                    f"mean_frontier_count={row.get('mean_frontier_count')}",
                    f"mean_near_tie_mass={row.get('mean_near_tie_mass')}",
                    f"selector_certificate_disagreement_rate={row.get('selector_certificate_disagreement_rate')}",
                    f"certificate_selectivity_rate={row.get('certificate_selectivity_rate')} (n={row.get('certificate_selectivity_denominator')})",
                ]
            )
            lines.append(f"- {row['variant_id']}: " + ", ".join(segments))
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Claim Framing")
    lines.append("")
    lines.append(
        "- Overall rows support the broad sampled-suite benchmark claim."
        if overall_claim_supported
        else "- Overall rows do not yet support an unconditional sampled-suite benchmark claim; interpret headline wins as conditional on the successful variants and the recorded failure profile."
    )
    lines.append(
        "- Representative rows support the general UK freight benchmark claim for the sampled suite."
        if representative_claim_supported
        else "- Representative rows should be read as sampled evidence rather than a general UK freight benchmark claim for this run."
    )
    lines.append(
        "- Ambiguity rows support claims about frontier breadth, certification fragility, and challenger discovery under harder trade-offs."
        if ambiguity_claim_supported
        else "- Ambiguity rows are informative for harder trade-offs, but this run does not justify strong ambiguity-specific claims without additional evidence."
    )
    lines.append(
        "- Hard-case rows form a broader ambiguity-pressure cohort derived from upstream ambiguity priors plus realized route difficulty signals; use this cohort for frontier breadth, certification, and runtime under harder trade-offs rather than as a pure controller-activity label."
        if hard_case_claim_supported
        else "- Hard-case rows form the broader ambiguity-pressure cohort for this run; controller conclusions should be drawn from the stricter controller-stress metrics rather than the hard-case count alone."
    )
    lines.append(
        "- Controller-stress rows isolate the stricter subset where VOI actually engaged without being counted as unnecessary refine work, so controller-action claims are tied to explicit admissible interventions."
        if controller_stress_claim_supported
        else f"- Controller-stress rows are sparse in this run ({sum(1 for row in rows if str(row.get('variant_id') or '') == 'C' and _bool_or_default(row.get('controller_stress_row'), False))} row(s)); interpret VOI primarily as a shortcut/stop controller rather than an iteration-heavy controller on this suite."
    )
    lines.append("")
    lines.append("## Startup And Warmup")
    lines.append("")
    lines.append("- Backend ready wait/probe and route-graph warmup are run-level service-start metrics. They are reported here directly and are not treated as per-row variant speed numbers.")
    lines.append(f"- backend_ready_wait_ms={readiness_summary.get('wait_elapsed_ms')}")
    lines.append(f"- backend_ready_probe_ms={readiness_summary.get('compute_ms')}")
    lines.append(f"- route_graph_warmup_elapsed_ms={_route_graph_startup_to_ready_ms(readiness_summary if isinstance(readiness_summary, Mapping) else None)}")
    lines.append("- `warmup_amortized_ms` divides one-time startup cost across the recorded thesis rows so cold-start overhead stays visible without being conflated with per-route algorithm time.")
    lines.append("")
    lines.append("## Direct Vs V0")
    lines.append("")
    if success_rows:
        for row in success_rows:
            lines.append(
                f"- {row['variant_id']}: "
                f"weighted_win_v0={_rate_text(row.get('weighted_win_rate_v0'), int(row.get('weighted_denominator_v0') or 0))}, "
                f"dominance_win_v0={_rate_text(row.get('dominance_win_rate_v0'), int(row.get('dominance_denominator_v0') or 0))}, "
                f"runtime_win_v0={_rate_text(row.get('runtime_win_rate_v0'), int(row.get('runtime_denominator_v0') or 0))}, "
                f"algorithm_runtime_win_v0={_rate_text(row.get('algorithm_runtime_win_rate_v0'), int(row.get('algorithm_runtime_denominator_v0') or 0))}, "
                f"mean_weighted_margin_gain_vs_v0={row.get('mean_weighted_margin_gain_vs_v0')}, "
                f"mean_balanced_gain_delta_vs_v0_score={row.get('mean_balanced_gain_delta_vs_v0_score')}, "
                f"mean_runtime_speedup_vs_v0={row.get('mean_runtime_speedup_vs_v0')}, "
                f"mean_algorithm_runtime_speedup_vs_v0={row.get('mean_algorithm_runtime_speedup_vs_v0')}, "
                f"mean_frontier_hypervolume_gain_vs_v0={row.get('mean_frontier_hypervolume_gain_vs_v0')}, "
                f"objective_gain_rows={row.get('mean_objective_gain_vs_v0_denominator')}/{row.get('row_count')}, "
                f"mean_certificate_lift_vs_v0={row.get('mean_certificate_lift_vs_v0')} (n={row.get('mean_certificate_lift_vs_v0_denominator')}), "
                f"certificate_availability_gain_vs_v0_rate={row.get('certificate_availability_gain_vs_v0_rate')}, "
                f"mean_hard_case_certificate_lift_vs_v0={row.get('mean_hard_case_certificate_lift_vs_v0')}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Controller Admissibility")
    lines.append("")
    if success_rows:
        for row in success_rows:
            lines.append(
                f"- {row['variant_id']}: "
                f"mean_initial_certificate={row.get('mean_initial_certificate')}, "
                f"initial_certificate_stop_rate={row.get('initial_certificate_stop_rate')}, "
                f"unnecessary_voi_refine_rate={row.get('unnecessary_voi_refine_rate')}, "
                f"mean_controller_shortcut_rate={row.get('mean_controller_shortcut_rate')}, "
                f"mean_voi_stop_after_certification_rate={row.get('mean_voi_stop_after_certification_rate')}, "
                f"mean_controller_stress_rate={row.get('mean_controller_stress_rate')}, "
                f"credible_search_uncertainty_rate={row.get('credible_search_uncertainty_rate')}, "
                f"credible_evidence_uncertainty_rate={row.get('credible_evidence_uncertainty_rate')}, "
                f"supported_hard_case_rate={row.get('supported_hard_case_rate')}, "
                f"evidence_first_engagement_rate={row.get('evidence_first_engagement_rate')}, "
                f"evidence_only_engagement_rate={row.get('evidence_only_engagement_rate')}, "
                f"mean_search_completeness_score={row.get('mean_search_completeness_score')}, "
                f"mean_search_completeness_gap={row.get('mean_search_completeness_gap')}, "
                f"mean_prior_support_strength={row.get('mean_prior_support_strength')}, "
                f"mean_support_richness={row.get('mean_support_richness')}, "
                f"mean_ambiguity_pressure={row.get('mean_ambiguity_pressure')}, "
                f"mean_pending_challenger_mass={row.get('mean_pending_challenger_mass')}, "
                f"mean_best_pending_flip_probability={row.get('mean_best_pending_flip_probability')}, "
                f"mean_corridor_family_recall={row.get('mean_corridor_family_recall')}, "
                f"mean_frontier_recall_at_budget={row.get('mean_frontier_recall_at_budget')}, "
                f"mean_voi_dccs_cache_hit_rate={row.get('mean_voi_dccs_cache_hit_rate')}, "
                f"mean_time_to_certification_ms={row.get('mean_time_to_certification_ms')}, "
                f"mean_stage_option_build_ms={row.get('mean_stage_option_build_ms')}, "
                f"mean_option_build_reuse_rate={row.get('mean_option_build_reuse_rate')}, "
                f"mean_option_build_cache_hits={row.get('mean_option_build_cache_hits')}, "
                f"mean_option_build_rebuild_count={row.get('mean_option_build_rebuild_count')}, "
                    f"mean_option_build_cache_hit_rate={row.get('mean_option_build_cache_hit_rate')}, "
                    f"option_build_cache_savings_ms_per_row={row.get('option_build_cache_savings_ms_per_row')}, "
                    f"route_state_cache_hit_rate={row.get('route_state_cache_hit_rate')}, "
                    f"route_state_cache_hits={row.get('route_state_cache_hits')}, "
                    f"route_state_cache_misses={row.get('route_state_cache_misses')}, "
                    f"refine_cost_mape={row.get('refine_cost_mape')}, "
                    f"refine_cost_mae_ms={row.get('refine_cost_mae_ms')}, "
                    f"refine_cost_rank_correlation={row.get('refine_cost_rank_correlation')}, "
                    f"comparator_independence_rate={row.get('comparator_independence_rate')}, "
                    f"strict_failure_elimination_rate={row.get('strict_failure_elimination_rate')}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Successful Variants")
    lines.append("")
    if success_rows:
        for row in success_rows:
            lines.append(_success_variant_line(row, include_od_ambiguity=include_od_ambiguity))
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Failed Variants")
    lines.append("")
    if failed_rows:
        for row in failed_rows:
            lines.append(
                _failure_variant_line(
                    row,
                    reason_counts=variant_failures.get(str(row.get("variant_id") or ""), {}),
                )
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Metric Highlights")
    lines.append("")
    if success_rows:
        for row in success_rows:
            segments = []
            if include_od_ambiguity:
                segments.append(f"corpus_ambiguity={row.get('mean_od_ambiguity_index')}")
            segments.extend(
                [
                    f"observed_ambiguity={row.get('mean_observed_ambiguity_index')}",
                    f"ambiguity_alignment={row.get('mean_ambiguity_alignment')}",
                    f"support_richness={row.get('mean_support_richness')}",
                    f"ambiguity_pressure={row.get('mean_ambiguity_pressure')}",
                    f"frontier_count={row.get('mean_frontier_count')}",
                    f"nontrivial_frontier_rate={row.get('nontrivial_frontier_rate')}",
                    f"winner_margin={row.get('mean_nominal_winner_margin')}",
                    f"near_tie_mass={row.get('mean_near_tie_mass')}",
                    f"certificate_margin={row.get('mean_certificate_margin')}",
                    f"disagreement_rate={row.get('selector_certificate_disagreement_rate')}",
                    f"fragility_entropy={row.get('mean_fragility_entropy')}",
                    f"dccs_frontier_recall={row.get('mean_dccs_frontier_recall_at_budget')}",
                    f"voi_controller_engagement_rate={row.get('voi_controller_engagement_rate')}",
                    f"controller_activation_on_high_ambiguity_rate={row.get('controller_activation_on_high_ambiguity_rate')}",
                    f"realized_diversity_collapse_rate={row.get('realized_diversity_collapse_rate')}",
                    f"supplemental_challenger_activation_rate={row.get('supplemental_challenger_activation_rate')}",
                    f"selected_from_supplemental_rescue_rate={row.get('selected_from_supplemental_rescue_rate')}",
                    f"selected_from_comparator_engine_rate={row.get('selected_from_comparator_engine_rate')}",
                    f"preemptive_comparator_activation_rate={row.get('preemptive_comparator_activation_rate')}",
                    f"selected_from_preemptive_comparator_seed_rate={row.get('selected_from_preemptive_comparator_seed_rate')}",
                    f"voi_action_count={row.get('mean_voi_action_count')}",
        f"voi_certificate_lift={row.get('mean_voi_realized_certificate_lift')}",
        f"voi_runner_up_gap_lift={row.get('mean_voi_realized_runner_up_gap_lift')}",
        f"voi_margin_lift={row.get('mean_voi_realized_margin_lift')}",
                    f"quality_gain_vs_v0={row.get('mean_weighted_margin_gain_vs_v0')}",
                    f"quality_margin_vs_best_baseline={row.get('mean_weighted_margin_vs_best_baseline')}",
                    f"weighted_win_best_baseline={_rate_text(row.get('weighted_win_rate_best_baseline'), int(row.get('weighted_denominator_best_baseline') or 0))}",
                    f"certificate_availability_gain_vs_v0_rate={row.get('certificate_availability_gain_vs_v0_rate')}",
                    f"algorithm_runtime_ms={row.get('mean_algorithm_runtime_ms')}",
                    f"stage_k_raw_ms={row.get('mean_stage_k_raw_ms')}",
                    f"stage_k_raw_graph_search_initial_ms={row.get('mean_stage_k_raw_graph_search_initial_ms')}",
                    f"stage_k_raw_graph_search_retry_ms={row.get('mean_stage_k_raw_graph_search_retry_ms')}",
                    f"stage_k_raw_graph_search_rescue_ms={row.get('mean_stage_k_raw_graph_search_rescue_ms')}",
                    f"runtime_win_v0={_rate_text(row.get('runtime_win_rate_v0'), int(row.get('runtime_denominator_v0') or 0))}",
                    f"algorithm_runtime_win_v0={_rate_text(row.get('algorithm_runtime_win_rate_v0'), int(row.get('algorithm_runtime_denominator_v0') or 0))}",
                    f"mean_runtime_speedup_vs_v0={row.get('mean_runtime_speedup_vs_v0')}",
                    f"mean_algorithm_runtime_speedup_vs_v0={row.get('mean_algorithm_runtime_speedup_vs_v0')}",
                    f"initial_certificate_stop_rate={row.get('initial_certificate_stop_rate')}",
                    f"unnecessary_voi_refine_rate={row.get('unnecessary_voi_refine_rate')}",
                    f"controller_shortcut_rate={row.get('mean_controller_shortcut_rate')}",
                    f"controller_stress_rate={row.get('mean_controller_stress_rate')}",
                    f"stage_option_build_ms={row.get('mean_stage_option_build_ms')}",
                    f"option_build_reuse_rate={row.get('mean_option_build_reuse_rate')}",
                    f"option_build_cache_savings_ms_per_row={row.get('option_build_cache_savings_ms_per_row')}",
                    f"route_state_cache_hit_rate={row.get('route_state_cache_hit_rate')}",
                    f"time_to_certification_ms={row.get('mean_time_to_certification_ms')}",
                    f"refc_shortcut_rate={row.get('mean_refc_shortcut_rate')}",
                    f"refc_cache_hits={row.get('mean_refc_cache_hits')}",
                    f"refc_stress_world_fraction={row.get('mean_refc_stress_world_fraction')}",
                    f"requested_cert_world_count={row.get('mean_requested_cert_world_count')}",
                    f"effective_cert_world_count={row.get('mean_effective_cert_world_count')}",
                    f"world_count_efficiency={row.get('mean_world_count_efficiency')}",
                    f"refc_ms_per_effective_world={row.get('mean_refc_ms_per_effective_world')}",
                    f"supplemental_rescue_ms={row.get('mean_stage_supplemental_rescue_ms')}",
                    f"preemptive_comparator_seed_ms={row.get('mean_stage_preemptive_comparator_seed_ms')}",
                    f"baseline_runtime_ms={row.get('mean_baseline_acquisition_runtime_ms')}",
                ]
            )
            lines.append(f"- {row['variant_id']}: " + ", ".join(segments))
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Hard-Case And Controller Stress")
    lines.append("")
    if success_rows:
        for row in success_rows:
            lines.append(
                f"- {row['variant_id']}: "
                f"mean_hard_case_rate={row.get('mean_hard_case_rate')}, "
                f"mean_hard_case_certificate={row.get('mean_hard_case_certificate')}, "
                f"mean_hard_case_runtime_ms={row.get('mean_hard_case_runtime_ms')}, "
                f"mean_hard_case_action_efficiency={row.get('mean_hard_case_action_efficiency')}, "
                f"mean_hard_case_controller_engagement_rate={row.get('mean_hard_case_controller_engagement_rate')}, "
                f"mean_controller_stress_rate={row.get('mean_controller_stress_rate')}, "
                f"mean_hard_case_certificate_lift_vs_v0={row.get('mean_hard_case_certificate_lift_vs_v0')}, "
                f"supplemental_challenger_activation_rate={row.get('supplemental_challenger_activation_rate')}, "
                f"selected_from_comparator_engine_rate={row.get('selected_from_comparator_engine_rate')}, "
                f"preemptive_comparator_activation_rate={row.get('preemptive_comparator_activation_rate')}, "
                f"selected_from_preemptive_comparator_seed_rate={row.get('selected_from_preemptive_comparator_seed_rate')}, "
                f"comparator_independence_rate={row.get('comparator_independence_rate')}, "
                f"strict_failure_elimination_rate={row.get('strict_failure_elimination_rate')}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Runtime Distribution")
    lines.append("")
    if success_rows:
        for row in success_rows:
            lines.append(
                f"- {row['variant_id']}: runtime_p50_ms={row.get('mean_runtime_p50_ms')}, "
                f"runtime_p90_ms={row.get('mean_runtime_p90_ms')}, runtime_p95_ms={row.get('mean_runtime_p95_ms')}, "
                f"route_request_ms={row.get('mean_route_request_ms')}, "
                f"baseline_osrm_ms={row.get('mean_baseline_osrm_ms')}, "
                f"baseline_ors_ms={row.get('mean_baseline_ors_ms')}, "
                f"algorithm_runtime_p50_ms={row.get('mean_algorithm_runtime_p50_ms')}, "
                f"algorithm_runtime_p90_ms={row.get('mean_algorithm_runtime_p90_ms')}, "
                f"algorithm_runtime_p95_ms={row.get('mean_algorithm_runtime_p95_ms')}, "
                f"stage_k_raw_ms={row.get('mean_stage_k_raw_ms')}, "
                f"stage_k_raw_graph_search_initial_ms={row.get('mean_stage_k_raw_graph_search_initial_ms')}, "
                f"stage_k_raw_graph_search_retry_ms={row.get('mean_stage_k_raw_graph_search_retry_ms')}, "
                f"stage_k_raw_graph_search_rescue_ms={row.get('mean_stage_k_raw_graph_search_rescue_ms')}, "
                f"stage_k_raw_graph_search_supplemental_ms={row.get('mean_stage_k_raw_graph_search_supplemental_ms')}, "
                f"stage_k_raw_osrm_fallback_ms={row.get('mean_stage_k_raw_osrm_fallback_ms')}, "
                f"stage_dccs_ms={row.get('mean_stage_dccs_ms')}, "
                f"stage_refinement_ms={row.get('mean_stage_refinement_ms')}, "
                f"stage_pareto_ms={row.get('mean_stage_pareto_ms')}, "
                f"stage_refc_ms={row.get('mean_stage_refc_ms')}, "
                f"stage_voi_ms={row.get('mean_stage_voi_ms')}, "
                f"runtime_ratio_vs_osrm={row.get('mean_runtime_ratio_vs_osrm')}, "
                f"runtime_ratio_vs_ors={row.get('mean_runtime_ratio_vs_ors')}, "
                f"algorithm_runtime_ratio_vs_osrm={row.get('mean_algorithm_runtime_ratio_vs_osrm')}, "
                f"algorithm_runtime_ratio_vs_ors={row.get('mean_algorithm_runtime_ratio_vs_ors')}, "
                f"runtime_gap_vs_osrm_ms={row.get('mean_runtime_gap_vs_osrm_ms')}, "
                f"runtime_gap_vs_ors_ms={row.get('mean_runtime_gap_vs_ors_ms')}, "
                f"algorithm_runtime_gap_vs_osrm_ms={row.get('mean_algorithm_runtime_gap_vs_osrm_ms')}, "
                f"algorithm_runtime_gap_vs_ors_ms={row.get('mean_algorithm_runtime_gap_vs_ors_ms')}, "
                f"runtime_win_v0={_rate_text(row.get('runtime_win_rate_v0'), int(row.get('runtime_denominator_v0') or 0))}, "
                f"algorithm_runtime_win_v0={_rate_text(row.get('algorithm_runtime_win_rate_v0'), int(row.get('algorithm_runtime_denominator_v0') or 0))}, "
                f"mean_runtime_speedup_vs_v0={row.get('mean_runtime_speedup_vs_v0')}, "
                f"mean_algorithm_runtime_speedup_vs_v0={row.get('mean_algorithm_runtime_speedup_vs_v0')}, "
                f"runtime_per_refined_candidate_ms={row.get('mean_runtime_per_refined_candidate_ms')}, "
                f"runtime_per_frontier_member_ms={row.get('mean_runtime_per_frontier_member_ms')}, "
                f"memory_per_refined_candidate_mb={row.get('mean_memory_per_refined_candidate_mb')}, "
                f"controller_value_per_second={row.get('mean_controller_value_per_second')}, "
                f"baseline_acq_runtime_p90_ms={row.get('mean_baseline_acquisition_runtime_p90_ms')}, "
                f"requested_cert_world_count={row.get('mean_requested_cert_world_count')}, "
                f"effective_cert_world_count={row.get('mean_effective_cert_world_count')}, "
                f"world_count_efficiency={row.get('mean_world_count_efficiency')}, "
                f"refc_ms_per_effective_world={row.get('mean_refc_ms_per_effective_world')}, "
                f"preflight_ms={row.get('mean_preflight_ms')}, "
                f"preflight_and_warmup_ms={row.get('mean_preflight_and_warmup_ms')}, "
                f"process_rss_mb={row.get('mean_process_rss_mb')}, "
                f"process_rss_p90_mb={row.get('mean_process_rss_p90_mb')}, "
                f"process_vms_mb={row.get('mean_process_vms_mb')}, "
                f"process_vms_p90_mb={row.get('mean_process_vms_p90_mb')}, "
                f"route_cache_hit_rate={row.get('mean_route_cache_hit_rate')}, "
                f"k_raw_cache_hit_rate={row.get('mean_k_raw_cache_hit_rate')}, "
                f"graph_low_ambiguity_fast_path_rate={row.get('mean_graph_low_ambiguity_fast_path_rate')}, "
                f"graph_supported_ambiguity_fast_fallback_rate={row.get('mean_graph_supported_ambiguity_fast_fallback_rate')}, "
                f"search_budget_utilization_p90={row.get('mean_search_budget_utilization_p90')}, "
                f"evidence_budget_utilization_p90={row.get('mean_evidence_budget_utilization_p90')}, "
                f"voi_action_density={row.get('mean_voi_action_density')}, "
                f"stage_option_build_ms={row.get('mean_stage_option_build_ms')}, "
                f"option_build_reuse_rate={row.get('mean_option_build_reuse_rate')}, "
                f"option_build_cache_hit_rate={row.get('mean_option_build_cache_hit_rate')}, "
                f"option_build_cache_hits={row.get('mean_option_build_cache_hits')}, "
                f"option_build_rebuild_count={row.get('mean_option_build_rebuild_count')}, "
                f"preemptive_comparator_seed_ms={row.get('mean_stage_preemptive_comparator_seed_ms')}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Cohort Highlights")
    lines.append("")
    cohort_rows = _cohort_summary_rows(rows)
    if cohort_rows:
        for row in cohort_rows:
            lines.append(
                f"- {row['variant_id']} / {row['cohort_label']}: rows={row.get('row_count')}/{row.get('cohort_total_row_count')}, "
                f"share={row.get('cohort_share_of_variant')}, mean_certificate={row.get('mean_certificate')}, "
                f"mean_od_ambiguity_index={row.get('mean_od_ambiguity_index')}, "
                f"mean_observed_ambiguity_index={row.get('mean_observed_ambiguity_index')}, "
                f"mean_runtime_ms={row.get('mean_runtime_ms')}, mean_hard_case_rate={row.get('mean_hard_case_rate')}, "
                f"controller_engagement_rate={row.get('voi_controller_engagement_rate')}, "
                f"mean_search_budget_utilization={row.get('mean_search_budget_utilization')}, "
                f"mean_evidence_budget_utilization={row.get('mean_evidence_budget_utilization')}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def run_thesis_evaluation(args: argparse.Namespace, *, client: httpx.Client | None = None) -> dict[str, Any]:
    own_client = client is None
    should_run_preflight = own_client or bool(getattr(args, "in_process_backend", False))
    old_out_dir = settings.out_dir
    settings.out_dir = str(Path(args.out_dir))
    active_client = client or httpx.Client(base_url=args.backend_url, timeout=args.route_timeout_seconds)
    try:
        corpus_path = str(args.corpus_csv or args.corpus_json)
        corpus_source_path = corpus_path.strip() or None
        corpus_source_resolved_path: str | None = None
        corpus_source_exists = False
        corpus_source_format = "csv" if getattr(args, "corpus_csv", None) else "json" if getattr(args, "corpus_json", None) else None
        if corpus_source_path:
            corpus_source_candidate = Path(corpus_source_path).expanduser()
            corpus_source_exists = corpus_source_candidate.exists()
            corpus_source_resolved_path = str(corpus_source_candidate.resolve(strict=False))
        raw_corpus_rows = load_corpus(corpus_path)
        if bool(getattr(args, "auto_enrich_corpus_ambiguity", False)) and _corpus_missing_ambiguity_fields(raw_corpus_rows):
            from scripts.enrich_od_corpus_with_ambiguity import enrich_rows as enrich_corpus_rows

            raw_corpus_rows = enrich_corpus_rows(
                raw_corpus_rows,
                probe_max_paths=max(2, min(8, int(args.max_alternatives))),
            )
        corpus_rows = _load_rows(raw_corpus_rows, seed=int(args.seed), max_od=int(args.max_od))
        corpus_hash = _digest(corpus_rows)
        effective_snapshot_mode = str(args.ors_snapshot_mode)
        if str(args.ors_baseline_policy) == "snapshot_record":
            effective_snapshot_mode = "record"
        elif str(args.ors_baseline_policy) == "snapshot_replay":
            effective_snapshot_mode = "replay"
        args.ors_snapshot_mode = effective_snapshot_mode
        cache_mode = _normalize_cache_mode(getattr(args, "cache_mode", "preserve"))
        cold_cache_scope = str(getattr(args, "cold_cache_scope", THESIS_COLD_CACHE_SCOPE) or THESIS_COLD_CACHE_SCOPE)
        if cold_cache_scope not in THESIS_COLD_CACHE_SCOPES:
            cold_cache_scope = THESIS_COLD_CACHE_SCOPE
        run_id = str(
            args.run_id
            or _run_id(
                seed=int(args.seed),
                corpus_hash=corpus_hash,
                model_version=str(args.model_version),
                world_count=int(args.world_count),
                snapshot_mode=effective_snapshot_mode,
                baseline_policy=str(args.ors_baseline_policy),
            )
        )
        evaluation_suite = _resolve_evaluation_suite_metadata(
            args=args,
            corpus_source_path=corpus_source_path,
            run_id=run_id,
        )
        artifact_dir_for_run(run_id)
        preflight_path = artifact_dir_for_run(run_id) / "repo_asset_preflight.json"
        if should_run_preflight:
            preflight_summary = run_preflight(output_path=preflight_path)
            if not bool(preflight_summary.get("required_ok")):
                first_failure = next((check for check in preflight_summary.get("checks", []) if not check.get("ok")), None)
                reason = "repo_asset_preflight_failed"
                if isinstance(first_failure, dict):
                    reason = str((first_failure.get("error") or {}).get("reason_code") or reason)
                raise RuntimeError(reason)
        else:
            preflight_summary = {
                "required_ok": True,
                "checks": [],
                "skipped": True,
                "checked_at_utc": _now(),
            }
            preflight_path.write_text(json.dumps(preflight_summary, indent=2), encoding="utf-8")
        readiness_summary = _wait_for_backend_ready(
            active_client,
            backend_url=args.backend_url,
            timeout_seconds=float(args.ready_timeout_seconds),
            poll_seconds=float(args.ready_poll_seconds),
        )
        baseline_smoke_summary = _run_baseline_smoke(
            active_client,
            base_url=args.backend_url,
            ors_baseline_policy=str(args.ors_baseline_policy),
            snapshot_mode=effective_snapshot_mode,
        )
        baseline_smoke_path = write_json_artifact(run_id, "baseline_smoke_summary.json", baseline_smoke_summary)
        if not bool(baseline_smoke_summary.get("required_ok")):
            failed_codes = [
                str(((baseline_smoke_summary.get(name) or {}) if isinstance(baseline_smoke_summary.get(name), dict) else {}).get("reason_code") or f"{name}_baseline_smoke_failed")
                for name in ("osrm", "ors")
                if not bool(((baseline_smoke_summary.get(name) or {}) if isinstance(baseline_smoke_summary.get(name), dict) else {}).get("ok"))
            ]
            raise RuntimeError(f"baseline_smoke_failed:{','.join(failed_codes)}")
        snapshot_path = Path(args.ors_snapshot_path) if args.ors_snapshot_path else artifact_dir_for_run(run_id) / "ors_snapshot.json"
        snapshot_bundle = _load_snapshot(snapshot_path) if effective_snapshot_mode in {"record", "replay"} else None
        rows: list[dict[str, Any]] = []
        baseline_cache: dict[str, tuple[BaselineResult, BaselineResult]] = {}
        cache_reset_count = 0
        for od_index, od in enumerate(corpus_rows):
            od_rows: list[dict[str, Any]] = []
            od_variant_specs = _variant_specs_for_od(args, od_index=od_index)
            base_request_config = _effective_request_config(args, od, variant_seed=int(args.seed))
            baseline_payload = _baseline_payload(args, od, request_config=base_request_config, variant_seed=int(args.seed))
            baseline_cache_key = _digest({"od_id": od.get("od_id"), "payload": baseline_payload, "ors_policy": str(args.ors_baseline_policy), "snapshot_mode": effective_snapshot_mode})
            osrm = _empty_baseline_result(method="osrm_engine_baseline", provider_mode="osrm_failure")
            ors = _empty_baseline_result(method="ors_local_engine_baseline", provider_mode="ors_failure")
            try:
                cached = baseline_cache.get(baseline_cache_key)
                if cached is not None:
                    osrm, ors = cached
                else:
                    try:
                        osrm = _fetch_baseline(
                            active_client,
                            _absolute_url(args.backend_url, "/route/baseline?realism=false"),
                            baseline_payload,
                            default_method="osrm_engine_baseline",
                        )
                    except Exception as exc:
                        failure_reason = _failure_reason(exc)
                        for spec in od_variant_specs:
                            od_rows.append(
                            _failure_row(
                                args,
                                od,
                                spec,
                                failure_reason=failure_reason,
                                osrm=osrm,
                                ors=ors,
                                request_config=base_request_config,
                                readiness_summary=readiness_summary,
                            )
                            )
                        rows.extend(od_rows)
                        continue
                    try:
                        ors, snapshot_bundle = _ors_baseline(
                            active_client,
                            args.backend_url,
                            baseline_payload,
                            od_id=od["od_id"],
                            snapshot_mode=effective_snapshot_mode,
                            snapshot_bundle=snapshot_bundle,
                            baseline_policy=str(args.ors_baseline_policy),
                        )
                    except Exception as exc:
                        failure_reason = _failure_reason(exc)
                        ors = _empty_baseline_result(method="ors_local_engine_baseline", provider_mode="ors_rejected")
                        for spec in od_variant_specs:
                            od_rows.append(
                                _failure_row(
                                    args,
                                    od,
                                    spec,
                                    failure_reason=failure_reason,
                                    osrm=osrm,
                                    ors=ors,
                                    request_config=base_request_config,
                                )
                            )
                        rows.extend(od_rows)
                        continue
                    baseline_cache[baseline_cache_key] = (osrm, ors)
                for spec in od_variant_specs:
                    variant_seed = _variant_seed(args, od, od_index=od_index, variant_id=spec.variant_id)
                    variant_request_config = _effective_request_config(args, od, variant_seed=variant_seed)
                    try:
                        if cache_mode == "cold":
                            _clear_backend_caches(
                                active_client,
                                backend_url=args.backend_url,
                                scope=cold_cache_scope,
                            )
                            cache_reset_count += 1
                        response, route_ms = _post_json(
                            active_client,
                            _absolute_url(args.backend_url, "/route"),
                            _variant_payload(
                                args,
                                od,
                                spec,
                                variant_seed=variant_seed,
                                request_config=variant_request_config,
                            ),
                        )
                        artifacts = _fetch_run_artifacts(active_client, args.backend_url, response)
                        _validate_route_artifacts(spec=spec, route_response=response, artifacts=artifacts)
                        response["artifact_validation"] = {
                            "status": "ok",
                            "required": list(_required_artifacts_for_pipeline(spec.pipeline_mode)),
                            "missing": [],
                        }
                        route_validation = _enforce_strict_thesis_inputs(
                            args=args,
                            route_response=response,
                            ors=ors,
                        )
                        response["evidence_validation"] = route_validation
                        od_rows.append(
                            _result_row(
                                args,
                                od,
                                spec,
                                response,
                                route_ms,
                                artifacts,
                                osrm,
                                ors,
                                readiness_summary=readiness_summary,
                                request_config=variant_request_config,
                            )
                        )
                    except Exception as exc:
                        failure_reason = _failure_reason(exc)
                        artifact_missing: list[str] = []
                        if ":" in failure_reason and failure_reason.split(":", 1)[0] in {
                            "strict_artifact_missing",
                            "strict_artifact_contract_missing",
                            "strict_artifact_invalid",
                        }:
                            failure_reason, artifact_name = failure_reason.split(":", 1)
                            artifact_missing = [artifact_name]
                        od_rows.append(
                            _failure_row(
                                args,
                                od,
                                spec,
                                failure_reason=failure_reason,
                                osrm=osrm,
                                ors=ors,
                                request_config=variant_request_config,
                                artifact_missing=artifact_missing,
                                readiness_summary=readiness_summary,
                            )
                        )
            except Exception as exc:
                failure_reason = _failure_reason(exc)
                for spec in od_variant_specs:
                    od_rows.append(
                        _failure_row(
                            args,
                            od,
                            spec,
                            failure_reason=failure_reason,
                            osrm=osrm,
                            ors=ors,
                            request_config=base_request_config,
                            readiness_summary=readiness_summary,
                        )
                    )
            rows.extend(od_rows)
        rows = _finalize_cross_variant_metrics(rows)
        startup_components = _startup_components_ms(readiness_summary if isinstance(readiness_summary, Mapping) else None)
        warmup_amortized_ms = round(sum(startup_components) / max(1, len(rows)), 6) if startup_components else None
        for row in rows:
            row["warmup_amortized_ms"] = warmup_amortized_ms
        summary_rows = _summary_rows(rows)
        cohort_summary_rows = _cohort_summary_rows(rows)
        run_validity_metrics = _run_validity_metrics(
            rows,
            preflight_summary=preflight_summary,
            readiness_summary=readiness_summary,
            evaluation_rerun_success_rate=1.0,
        )
        _apply_run_level_summary_metrics(
            summary_rows,
            run_validity_metrics=run_validity_metrics,
        )
        _apply_run_level_summary_metrics(
            cohort_summary_rows,
            run_validity_metrics=run_validity_metrics,
        )
        cohort_composition = _cohort_composition(rows)
        metrics_payload = {
            "run_id": run_id,
            "corpus_hash": corpus_hash,
            "rows": rows,
            "summary_rows": summary_rows,
            "summary_by_cohort_rows": cohort_summary_rows,
            "cohort_composition": cohort_composition,
            "run_validity": run_validity_metrics,
            "baseline_smoke": baseline_smoke_summary,
            "startup_and_warmup": {
                "backend_ready_wait_ms": readiness_summary.get("wait_elapsed_ms"),
                "backend_ready_probe_ms": readiness_summary.get("compute_ms"),
                "route_graph_warmup_elapsed_ms": _route_graph_startup_to_ready_ms(
                    readiness_summary if isinstance(readiness_summary, Mapping) else None
                ),
            },
        }
        plots_payload = {
            "certificate_vs_variant": [{"variant_id": row["variant_id"], "mean_certificate": row["mean_certificate"]} for row in summary_rows],
            "runtime_vs_variant": [{"variant_id": row["variant_id"], "mean_runtime_ms": row["mean_runtime_ms"]} for row in summary_rows],
        "win_rate_vs_variant": [{"variant_id": row["variant_id"], "weighted_win_rate_osrm": row["weighted_win_rate_osrm"], "weighted_win_rate_ors": row["weighted_win_rate_ors"], "weighted_win_rate_best_baseline": row["weighted_win_rate_best_baseline"], "weighted_win_rate_v0": row["weighted_win_rate_v0"], "time_preserving_win_rate": row["time_preserving_win_rate"], "time_preserving_win_rate_osrm": row["time_preserving_win_rate_osrm"], "time_preserving_win_rate_ors": row["time_preserving_win_rate_ors"]} for row in summary_rows],
            "ambiguity_vs_variant": [{"variant_id": row["variant_id"], "mean_od_ambiguity_index": row["mean_od_ambiguity_index"], "mean_observed_ambiguity_index": row["mean_observed_ambiguity_index"], "mean_near_tie_mass": row["mean_near_tie_mass"]} for row in summary_rows],
            "ambiguity_prior_vs_variant": [{"variant_id": row["variant_id"], "mean_od_engine_disagreement_prior": row["mean_od_engine_disagreement_prior"], "mean_od_hard_case_prior": row["mean_od_hard_case_prior"], "mean_ambiguity_alignment": row["mean_ambiguity_alignment"]} for row in summary_rows],
            "ambiguity_alignment_vs_variant": [{"variant_id": row["variant_id"], "mean_ambiguity_alignment": row["mean_ambiguity_alignment"], "controller_activation_on_high_ambiguity_rate": row["controller_activation_on_high_ambiguity_rate"]} for row in summary_rows],
            "certificate_margin_vs_variant": [{"variant_id": row["variant_id"], "mean_certificate_margin": row["mean_certificate_margin"], "mean_certificate_runner_up_gap": row["mean_certificate_runner_up_gap"]} for row in summary_rows],
            "dccs_quality_vs_variant": [{"variant_id": row["variant_id"], "mean_dccs_frontier_recall_at_budget": row["mean_dccs_frontier_recall_at_budget"], "mean_dccs_corridor_family_recall": row["mean_dccs_corridor_family_recall"]} for row in summary_rows],
        "runtime_split_vs_variant": [{"variant_id": row["variant_id"], "mean_route_request_ms": row["mean_route_request_ms"], "mean_algorithm_runtime_ms": row["mean_algorithm_runtime_ms"], "mean_algorithm_runtime_delta_ratio_vs_v0": row["mean_algorithm_runtime_speedup_vs_v0"], "mean_baseline_osrm_ms": row["mean_baseline_osrm_ms"], "mean_baseline_ors_ms": row["mean_baseline_ors_ms"], "mean_baseline_acquisition_runtime_ms": row["mean_baseline_acquisition_runtime_ms"], "mean_baseline_runtime_share": row["mean_baseline_runtime_share"], "mean_preflight_and_warmup_ms": row["mean_preflight_and_warmup_ms"], "mean_stage_option_build_ms": row["mean_stage_option_build_ms"], "mean_option_build_reuse_rate": row["mean_option_build_reuse_rate"]} for row in summary_rows],
            "runtime_distribution_vs_variant": [{"variant_id": row["variant_id"], "mean_runtime_p50_ms": row["mean_runtime_p50_ms"], "mean_runtime_p90_ms": row["mean_runtime_p90_ms"], "mean_runtime_p95_ms": row["mean_runtime_p95_ms"], "mean_algorithm_runtime_p90_ms": row["mean_algorithm_runtime_p90_ms"], "mean_algorithm_runtime_p95_ms": row["mean_algorithm_runtime_p95_ms"], "mean_baseline_acquisition_runtime_p90_ms": row["mean_baseline_acquisition_runtime_p90_ms"], "mean_process_rss_p90_mb": row["mean_process_rss_p90_mb"], "mean_process_vms_p90_mb": row["mean_process_vms_p90_mb"]} for row in summary_rows],
        "performance_vs_variant": [{"variant_id": row["variant_id"], "mean_runtime_ratio_vs_osrm": row["mean_runtime_ratio_vs_osrm"], "mean_runtime_ratio_vs_ors": row["mean_runtime_ratio_vs_ors"], "mean_algorithm_runtime_ratio_vs_osrm": row["mean_algorithm_runtime_ratio_vs_osrm"], "mean_algorithm_runtime_ratio_vs_ors": row["mean_algorithm_runtime_ratio_vs_ors"], "mean_runtime_gap_vs_osrm_ms": row["mean_runtime_gap_vs_osrm_ms"], "mean_runtime_gap_vs_ors_ms": row["mean_runtime_gap_vs_ors_ms"], "mean_algorithm_runtime_gap_vs_osrm_ms": row["mean_algorithm_runtime_gap_vs_osrm_ms"], "mean_algorithm_runtime_gap_vs_ors_ms": row["mean_algorithm_runtime_gap_vs_ors_ms"], "mean_stage_k_raw_ms": row["mean_stage_k_raw_ms"], "mean_stage_k_raw_graph_search_initial_ms": row["mean_stage_k_raw_graph_search_initial_ms"], "mean_stage_k_raw_graph_search_retry_ms": row["mean_stage_k_raw_graph_search_retry_ms"], "mean_stage_k_raw_graph_search_rescue_ms": row["mean_stage_k_raw_graph_search_rescue_ms"], "mean_stage_k_raw_graph_search_supplemental_ms": row["mean_stage_k_raw_graph_search_supplemental_ms"], "mean_stage_k_raw_osrm_fallback_ms": row["mean_stage_k_raw_osrm_fallback_ms"], "mean_stage_dccs_ms": row["mean_stage_dccs_ms"], "mean_stage_refinement_ms": row["mean_stage_refinement_ms"], "mean_stage_pareto_ms": row["mean_stage_pareto_ms"], "mean_stage_refc_ms": row["mean_stage_refc_ms"], "mean_stage_voi_ms": row["mean_stage_voi_ms"], "mean_runtime_per_refined_candidate_ms": row["mean_runtime_per_refined_candidate_ms"], "mean_runtime_per_frontier_member_ms": row["mean_runtime_per_frontier_member_ms"], "warmup_amortized_ms": row["warmup_amortized_ms"], "mean_memory_per_refined_candidate_mb": row["mean_memory_per_refined_candidate_mb"], "mean_preflight_ms": row["mean_preflight_ms"], "mean_process_rss_mb": row["mean_process_rss_mb"], "mean_process_vms_mb": row["mean_process_vms_mb"], "mean_route_cache_hit_rate": row["mean_route_cache_hit_rate"], "mean_k_raw_cache_hit_rate": row["mean_k_raw_cache_hit_rate"], "mean_graph_low_ambiguity_fast_path_rate": row["mean_graph_low_ambiguity_fast_path_rate"], "mean_graph_supported_ambiguity_fast_fallback_rate": row["mean_graph_supported_ambiguity_fast_fallback_rate"], "mean_cache_reuse_ratio": row["mean_cache_reuse_ratio"], "mean_quality_per_second": row["mean_quality_per_second"], "mean_frontier_gain_per_ms": row["mean_frontier_gain_per_ms"], "mean_certificate_gain_per_world": row["mean_certificate_gain_per_world"], "baseline_identity_verified_rate": row["baseline_identity_verified_rate"], "mean_ambiguity_prior_gap": row["mean_ambiguity_prior_gap"], "ambiguity_prior_realized_correlation": row["ambiguity_prior_realized_correlation"], "realized_diversity_collapse_rate": row["realized_diversity_collapse_rate"], "supplemental_challenger_activation_rate": row["supplemental_challenger_activation_rate"], "selected_from_supplemental_rescue_rate": row["selected_from_supplemental_rescue_rate"], "selected_from_comparator_engine_rate": row["selected_from_comparator_engine_rate"], "mean_refc_shortcut_rate": row["mean_refc_shortcut_rate"], "mean_refc_cache_hits": row["mean_refc_cache_hits"], "mean_refc_unique_world_count": row["mean_refc_unique_world_count"], "mean_refc_world_reuse_rate": row["mean_refc_world_reuse_rate"], "mean_refc_hard_stress_pack_count": row["mean_refc_hard_stress_pack_count"], "mean_refc_stress_world_fraction": row["mean_refc_stress_world_fraction"], "mean_stage_supplemental_rescue_ms": row["mean_stage_supplemental_rescue_ms"], "mean_controller_value_per_second": row["mean_controller_value_per_second"]} for row in summary_rows],
            "controller_admissibility_vs_variant": [{"variant_id": row["variant_id"], "mean_initial_certificate": row["mean_initial_certificate"], "initial_certificate_stop_rate": row["initial_certificate_stop_rate"], "unnecessary_voi_refine_rate": row["unnecessary_voi_refine_rate"], "mean_controller_shortcut_rate": row["mean_controller_shortcut_rate"], "mean_voi_stop_after_certification_rate": row["mean_voi_stop_after_certification_rate"], "mean_controller_stress_rate": row["mean_controller_stress_rate"], "mean_time_to_certification_ms": row["mean_time_to_certification_ms"]} for row in summary_rows],
            "controller_density_vs_variant": [{"variant_id": row["variant_id"], "mean_voi_action_density": row["mean_voi_action_density"], "mean_voi_action_count": row["mean_voi_action_count"], "voi_controller_engagement_rate": row["voi_controller_engagement_rate"]} for row in summary_rows],
            "gain_vs_v0": [{"variant_id": row["variant_id"], "mean_weighted_margin_gain_vs_v0": row["mean_weighted_margin_gain_vs_v0"], "mean_balanced_gain_delta_vs_v0_score": row["mean_balanced_gain_delta_vs_v0_score"], "mean_frontier_hypervolume_gain_vs_v0": row["mean_frontier_hypervolume_gain_vs_v0"], "mean_certificate_lift_vs_v0": row["mean_certificate_lift_vs_v0"], "mean_hard_case_certificate_lift_vs_v0": row["mean_hard_case_certificate_lift_vs_v0"], "certificate_availability_gain_vs_v0_rate": row["certificate_availability_gain_vs_v0_rate"]} for row in summary_rows],
        "quality_vs_best_baseline": [{"variant_id": row["variant_id"], "weighted_win_rate_best_baseline": row["weighted_win_rate_best_baseline"], "dominance_win_rate_best_baseline": row["dominance_win_rate_best_baseline"], "balanced_win_rate_best_baseline": row["balanced_win_rate_best_baseline"], "time_preserving_win_rate": row["time_preserving_win_rate"], "time_preserving_dominance_rate": row["time_preserving_dominance_rate"], "mean_weighted_margin_vs_best_baseline": row["mean_weighted_margin_vs_best_baseline"]} for row in summary_rows],
            "certificate_vs_cohort": [{"variant_id": row["variant_id"], "cohort_label": row["cohort_label"], "row_count": row["row_count"], "mean_certificate": row["mean_certificate"]} for row in cohort_summary_rows],
            "runtime_vs_cohort": [{"variant_id": row["variant_id"], "cohort_label": row["cohort_label"], "row_count": row["row_count"], "mean_runtime_ms": row["mean_runtime_ms"]} for row in cohort_summary_rows],
            "hard_case_vs_variant": [{"variant_id": row["variant_id"], "mean_hard_case_rate": row["mean_hard_case_rate"], "mean_hard_case_certificate": row["mean_hard_case_certificate"], "mean_hard_case_runtime_ms": row["mean_hard_case_runtime_ms"]} for row in summary_rows],
            "controller_refresh_split_vs_variant": [{"variant_id": row["variant_id"], "controller_refresh_fallback_activation_rate": row["controller_refresh_fallback_activation_rate"], "controller_empirical_vs_raw_refresh_disagreement_rate": row["controller_empirical_vs_raw_refresh_disagreement_rate"]} for row in summary_rows],
            "hard_case_transfer_vs_variant": [{"variant_id": row["variant_id"], "broad_hard_case_certificate_selectivity_rate": row["broad_hard_case_certificate_selectivity_rate"], "broad_hard_case_evidence_first_engagement_rate": row["broad_hard_case_evidence_first_engagement_rate"], "broad_hard_case_productive_voi_action_rate": row["broad_hard_case_productive_voi_action_rate"], "broad_hard_case_refc_signal_presence_rate": row["broad_hard_case_refc_signal_presence_rate"]} for row in summary_rows],
            "run_validity": run_validity_metrics,
            "cohort_composition": cohort_composition,
            "baseline_smoke": baseline_smoke_summary,
            "startup_and_warmup": {
                "backend_ready_wait_ms": readiness_summary.get("wait_elapsed_ms"),
                "backend_ready_probe_ms": readiness_summary.get("compute_ms"),
                "route_graph_warmup_elapsed_ms": _route_graph_startup_to_ready_ms(
                    readiness_summary if isinstance(readiness_summary, Mapping) else None
                ),
            },
        }
        methods_text = _methods_appendix(args, corpus_hash=corpus_hash, row_count=len(corpus_rows))
        write_json_artifact(run_id, "od_corpus.json", {"rows": corpus_rows, "corpus_hash": corpus_hash})
        write_csv_artifact(run_id, "od_corpus.csv", fieldnames=list(corpus_rows[0].keys()), rows=corpus_rows)
        write_json_artifact(
            run_id,
            "od_corpus_summary.json",
            {
                "row_count": len(corpus_rows),
                "corpus_hash": corpus_hash,
                "corpus_kinds": sorted({str(row.get("corpus_kind") or "") for row in corpus_rows if str(row.get("corpus_kind") or "")}),
                "mean_ambiguity_index": _mean_numeric(corpus_rows, "od_ambiguity_index") or _mean_numeric(corpus_rows, "ambiguity_index"),
                "mean_od_ambiguity_confidence": _mean_numeric(corpus_rows, "od_ambiguity_confidence"),
                "mean_od_ambiguity_source_count": _mean_numeric(corpus_rows, "od_ambiguity_source_count"),
                "mean_candidate_probe_path_count": _mean_numeric(corpus_rows, "candidate_probe_path_count"),
                "mean_candidate_probe_objective_spread": _mean_numeric(corpus_rows, "candidate_probe_objective_spread"),
                "mean_candidate_probe_engine_disagreement_prior": _mean_numeric(corpus_rows, "candidate_probe_engine_disagreement_prior"),
                "mean_hard_case_prior": _mean_numeric(corpus_rows, "hard_case_prior"),
            },
        )
        write_json_artifact(run_id, "cohort_composition.json", cohort_composition)
        if snapshot_bundle is not None:
            snapshot_bundle["manifest_hash"] = _digest(snapshot_bundle.get("routes", {}))
            snapshot_path = write_json_artifact(run_id, "ors_snapshot.json", snapshot_bundle)
        results_csv = write_csv_artifact(run_id, "thesis_results.csv", fieldnames=RESULT_FIELDS, rows=rows)
        write_json_artifact(run_id, "thesis_results.json", {"rows": rows})
        summary_csv = write_csv_artifact(run_id, "thesis_summary.csv", fieldnames=SUMMARY_FIELDS, rows=summary_rows)
        write_json_artifact(run_id, "thesis_summary.json", {"summary_rows": summary_rows})
        cohort_summary_csv = write_csv_artifact(run_id, "thesis_summary_by_cohort.csv", fieldnames=COHORT_SUMMARY_FIELDS, rows=cohort_summary_rows)
        write_json_artifact(
            run_id,
            "thesis_summary_by_cohort.json",
            {
                "summary_rows": cohort_summary_rows,
                "cohort_definitions": {
                    "representative": "Rows whose corpus_group is representative.",
                    "ambiguity": "Rows whose corpus_group is ambiguity.",
                    "hard_case": "Rows classified by stronger upstream ambiguity priors together with realized difficulty or controller-stress signals.",
                },
            },
        )
        write_json_artifact(
            run_id,
            "metadata.json",
            {
                "run_id": run_id,
                "variant_count": len(VARIANTS),
                "row_count": len(rows),
                "failure_count": sum(1 for row in rows if row.get("failure_reason")),
                "corpus_hash": corpus_hash,
                "strict_evidence_policy": STRICT_EVIDENCE_POLICY,
                "ors_baseline_policy": str(args.ors_baseline_policy),
                "repo_asset_preflight_required_ok": bool(preflight_summary.get("required_ok")),
                "repo_asset_preflight_path": str(preflight_path),
                "backend_ready_summary": readiness_summary,
                "run_validity": run_validity_metrics,
                "baseline_smoke_summary": baseline_smoke_summary,
                "strict_proxy_ors_allowed": bool(args.allow_proxy_ors),
                "strict_evidence_fallbacks_allowed": bool(args.allow_evidence_fallbacks),
                "evaluation_suite": evaluation_suite,
                "cache_mode": cache_mode,
                "cache_reset_scope": "variant" if cache_mode == "cold" else "none",
                "cache_reset_policy": cold_cache_scope if cache_mode == "cold" else "none",
                "cache_reset_count": cache_reset_count,
                "cache_carryover_expected": cache_mode != "cold",
            },
        )
        write_json_artifact(
            run_id,
            "results.json",
            {"rows": rows, "summary_rows": summary_rows, "summary_by_cohort_rows": cohort_summary_rows},
        )
        thesis_metrics_path = write_json_artifact(run_id, "thesis_metrics.json", metrics_payload)
        thesis_plots_path = write_json_artifact(run_id, "thesis_plots.json", plots_payload)
        methods_path = write_text_artifact(run_id, "methods_appendix.md", methods_text)
        evaluation_manifest_path = write_json_artifact(
            run_id,
            "evaluation_manifest.json",
            {
                "run_id": run_id,
                "created_at": _now(),
                "backend_url": args.backend_url,
                "osrm_base_url": settings.osrm_base_url,
                "corpus_hash": corpus_hash,
                "corpus_source_path": corpus_source_path,
                "corpus_source_resolved_path": corpus_source_resolved_path,
                "corpus_source_format": corpus_source_format,
                "corpus_source_exists": corpus_source_exists,
                "model_version": args.model_version,
                "ors_baseline_policy": str(args.ors_baseline_policy),
                "ors_snapshot_mode": effective_snapshot_mode,
                "ors_snapshot_path": str(snapshot_path) if snapshot_bundle is not None else None,
                "repo_asset_preflight_path": str(preflight_path),
                "repo_asset_preflight_required_ok": bool(preflight_summary.get("required_ok")),
                "backend_ready_summary": readiness_summary,
                "run_validity": run_validity_metrics,
                "baseline_smoke_path": str(baseline_smoke_path),
                "baseline_smoke_summary": baseline_smoke_summary,
                "strict_evidence_policy": STRICT_EVIDENCE_POLICY,
                "evaluation_suite": evaluation_suite,
                "cache_mode": cache_mode,
                "cache_reset_scope": "variant" if cache_mode == "cold" else "none",
                "cache_reset_policy": cold_cache_scope if cache_mode == "cold" else "none",
                "cache_reset_count": cache_reset_count,
                "cache_carryover_expected": cache_mode != "cold",
            },
        )
        manifest_path = write_manifest(
            run_id,
            {
                "request": {
                    "evaluation": {
                        "corpus_hash": corpus_hash,
                        "corpus_source_path": corpus_source_path,
                        "corpus_source_resolved_path": corpus_source_resolved_path,
                        "corpus_source_format": corpus_source_format,
                        "corpus_source_exists": corpus_source_exists,
                        "backend_url": args.backend_url,
                        "evaluation_suite": evaluation_suite,
                        "cache_mode": cache_mode,
                        "cache_reset_scope": "variant" if cache_mode == "cold" else "none",
                        "cache_reset_policy": cold_cache_scope if cache_mode == "cold" else "none",
                        "cache_reset_count": cache_reset_count,
                    }
                },
                "execution": {"pair_count": len(corpus_rows), "variant_count": len(VARIANTS), "duration_ms": _mean_numeric(rows, "runtime_ms")},
            },
        )
        output_validation = _validate_written_output_artifacts(
            results_csv=Path(results_csv),
            summary_csv=Path(summary_csv),
            methods_path=Path(methods_path),
            thesis_report_path=None,
            evaluation_manifest_path=Path(evaluation_manifest_path),
            manifest_path=Path(manifest_path),
            extra_json_paths={
                "thesis_metrics.json": Path(thesis_metrics_path),
                "thesis_plots.json": Path(thesis_plots_path),
                "metadata.json": artifact_dir_for_run(run_id) / "metadata.json",
                "results.json": artifact_dir_for_run(run_id) / "results.json",
                "thesis_results.json": artifact_dir_for_run(run_id) / "thesis_results.json",
                "thesis_summary.json": artifact_dir_for_run(run_id) / "thesis_summary.json",
                "thesis_summary_by_cohort.json": artifact_dir_for_run(run_id) / "thesis_summary_by_cohort.json",
                "od_corpus.json": artifact_dir_for_run(run_id) / "od_corpus.json",
                "od_corpus_summary.json": artifact_dir_for_run(run_id) / "od_corpus_summary.json",
                "cohort_composition.json": artifact_dir_for_run(run_id) / "cohort_composition.json",
                "baseline_smoke_summary.json": Path(baseline_smoke_path),
            },
            extra_text_paths={},
            optional_paths={
                "ors_snapshot.json": Path(snapshot_path) if snapshot_bundle is not None else None,
                "od_corpus.csv": artifact_dir_for_run(run_id) / "od_corpus.csv",
                "thesis_summary_by_cohort.csv": Path(cohort_summary_csv),
            },
        )
        thesis_report_text = _thesis_report(
            run_id,
            summary_rows,
            rows=rows,
            corpus_hash=corpus_hash,
            row_count=len(corpus_rows),
            ors_baseline_policy=str(args.ors_baseline_policy),
            ors_snapshot_mode=effective_snapshot_mode,
            preflight_summary=preflight_summary,
            readiness_summary=readiness_summary,
            baseline_smoke_summary=baseline_smoke_summary,
            output_validation=output_validation,
        )
        thesis_report_path = write_text_artifact(run_id, "thesis_report.md", thesis_report_text)
        _validate_text_artifact_file(Path(thesis_report_path), artifact_name="thesis_report.md")
        output_validation.setdefault("artifacts", {})["thesis_report.md"] = {
            "path": str(thesis_report_path),
            "size_bytes": int(Path(thesis_report_path).stat().st_size),
        }
        output_validation["validated_artifact_count"] = int(output_validation.get("validated_artifact_count", 0)) + 1
        return {
            "run_id": run_id,
            "rows": rows,
            "summary_rows": summary_rows,
            "summary_by_cohort_rows": cohort_summary_rows,
            "success_row_count": sum(1 for row in rows if not row.get("failure_reason")),
            "failure_row_count": sum(1 for row in rows if row.get("failure_reason")),
            "failure_breakdown": _failure_breakdown(rows),
            "successful_variants": [row["variant_id"] for row in _successful_summary_rows(summary_rows)],
            "failed_variants": [row["variant_id"] for row in _failed_summary_rows(summary_rows)],
            "results_csv": str(results_csv),
            "summary_csv": str(summary_csv),
            "summary_by_cohort_csv": str(cohort_summary_csv),
            "methods_appendix": str(methods_path),
            "thesis_report": str(thesis_report_path),
            "evaluation_manifest": str(evaluation_manifest_path),
            "manifest_path": str(manifest_path),
            "output_artifact_validation": output_validation,
            "baseline_smoke_summary": baseline_smoke_summary,
            "baseline_smoke_path": str(baseline_smoke_path),
            "ors_snapshot_path": str(snapshot_path) if snapshot_bundle is not None else None,
            "ors_snapshot_mode": effective_snapshot_mode,
            "repo_asset_preflight_path": str(preflight_path),
            "ors_baseline_policy": str(args.ors_baseline_policy),
            "run_validity": run_validity_metrics,
            "strict_live_readiness_pass_rate": run_validity_metrics["strict_live_readiness_pass_rate"],
            "evaluation_rerun_success_rate": run_validity_metrics["evaluation_rerun_success_rate"],
            "scenario_profile_unavailable_rate": run_validity_metrics["scenario_profile_unavailable_rate"],
        }
    finally:
        settings.out_dir = old_out_dir
        if own_client:
            active_client.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.in_process_backend:
        from app.main import app

        with TestClient(app) as client:
            run_thesis_evaluation(args, client=client)
        return 0
    run_thesis_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
