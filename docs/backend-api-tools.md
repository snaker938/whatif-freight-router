# Backend APIs and Tooling

Last Updated: 2026-04-14
Applies To: `backend/app/main.py`, `backend/app/models.py`, `backend/app/preference_runtime.py`, `backend/app/run_store.py`, `backend/app/model_data_errors.py`, `backend/app/metrics_store.py`, `backend/app/settings.py`

This page is the source-of-truth backend API contract for current strict runtime behavior.

## Runtime Contract (Current)

- Runtime is hard-strict by default via `Settings._enforce_strict_runtime_defaults` in `backend/app/settings.py`.
- Route-producing failures are reason-coded and normalized with `normalize_reason_code` in `backend/app/model_data_errors.py`.
- Streaming and non-streaming flows use aligned reason-code semantics.
- Synthetic fallback paths are blocked in strict production paths unless explicitly test-scoped.
- `/metrics` returns a raw request/error/duration snapshot from `MetricsStore.snapshot()` and then appends cache stats for `route_cache`, `k_raw_cache`, `route_option_cache`, `route_state_cache`, and `voi_dccs_cache`.
- `/health/ready` exposes both graph readiness and strict live-source readiness; the top-level shape includes `status`, `strict_route_ready`, `recommended_action`, `route_graph`, and `strict_live`.
- Route responses carry `selected_certificate` and `voi_stop_summary` when VOI/certification is active, while the persisted run bundle also records `candidate_diagnostics`.
- Batch pair failures are serialized as strict text in the form `reason_code:<code>; message:<message>` and may append a `; warning=<first warning>` suffix.
- Live `POST /route` is certification-facing only: it rejects `pipeline_mode=legacy` and waypoint requests, and comparison traffic belongs on `POST /route/baseline` or `POST /route/baseline/ors`.
- `POST /route/preference` is the runtime-backed preference-elicitation sync surface used by the frontend to re-emit monotone `preference_state`, `preference_query_trace`, and `preference_summary` payloads against the current candidate set.

## Current Metrics And Diagnostics

- `GET /metrics`
  - `created_at`
  - `total_requests`
  - `total_errors`
  - `endpoint_count`
  - `endpoints.<name>.request_count`
  - `endpoints.<name>.error_count`
  - `endpoints.<name>.total_duration_ms`
  - `endpoints.<name>.avg_duration_ms`
  - `endpoints.<name>.max_duration_ms`
- `GET /cache/stats`
  - `route_cache`
  - `hot_rerun_route_cache_checkpoint`
  - `certification_cache`
  - `hot_rerun_certification_cache_checkpoint`
  - `k_raw_cache`
  - `route_option_cache`
  - `route_state_cache`
  - `hot_rerun_route_state_cache_checkpoint`
  - `voi_dccs_cache`
  - `hot_rerun_voi_dccs_cache_checkpoint`
- Route and pareto responses commonly expose these diagnostics families:
  - `stage_timings_ms`
  - `resource_usage`
  - `candidate_diagnostics`
  - `selected_certificate`
  - `voi_stop_summary`
  - `selected_route_id`
  - `selected_candidate_ids`
  - `route_cache_runtime`
  - `route_option_cache_runtime`
  - `voi_dccs_runtime`
  - `counts`
  - `budgets`

## Endpoint Inventory

For endpoint parity, `python scripts/check_docs.py --check-endpoints` compares FastAPI decorators in `backend/app/main.py` with backticked `METHOD /path` entries in this file; the command still also runs the always-on notebook-reference ban. It does not validate frontend Next.js proxy coverage, response-model field details, artifact allowlists, endpoint mentions in `docs/api-cookbook.md`, or evaluation prose.

### System And Admin

- `GET /`
- `GET /health`
- `GET /health/ready`
- `GET /debug/live-calls/{request_id}`
- `GET /metrics`
- `GET /cache/stats`
- `DELETE /cache`
- `POST /cache/hot-rerun/restore`

### Vehicles

- `GET /vehicles`
- `GET /vehicles/custom`
- `POST /vehicles/custom`
- `PUT /vehicles/custom/{vehicle_id}`
- `DELETE /vehicles/custom/{vehicle_id}`

### Routing And Pareto

- `POST /route`
- `POST /route/preference`
- `POST /route/baseline`
- `POST /route/baseline/ors`
- `POST /pareto`
- `POST /pareto/stream`
- `POST /api/pareto/stream`
- `POST /departure/optimize`
- `POST /duty/chain`
- `POST /scenario/compare`

### Experiments

- `GET /experiments`
- `POST /experiments`
- `GET /experiments/{experiment_id}`
- `PUT /experiments/{experiment_id}`
- `DELETE /experiments/{experiment_id}`
- `POST /experiments/{experiment_id}/compare`

### Batch

- `POST /batch/import/csv`
- `POST /batch/pareto`

### Run Artifacts And Signatures

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/provenance`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `POST /verify/signature`
- `GET /runs/{run_id}/artifacts`
- `GET /runs/{run_id}/artifacts/results.json`
- `GET /runs/{run_id}/artifacts/results.csv`
- `GET /runs/{run_id}/artifacts/metadata.json`
- `GET /runs/{run_id}/artifacts/routes.geojson`
- `GET /runs/{run_id}/artifacts/results_summary.csv`
- `GET /runs/{run_id}/artifacts/{artifact_name}`

### Oracle Quality

- `POST /oracle/quality/check`
- `GET /oracle/quality/dashboard`
- `GET /oracle/quality/dashboard.csv`

## Request Surface (Core Models)

All route-producing requests are derived from `RouteRequest`/`ParetoRequest`/`BatchParetoRequest` families in `backend/app/models.py`.

Shared fields:

- `origin`, `destination`, optional `waypoints`
- `vehicle_type`
- `scenario_mode` (`no_sharing`, `partial_sharing`, `full_sharing`)
- `weights` (`time`, `money`, `co2`)
- `cost_toggles` (`use_tolls`, `fuel_price_multiplier`, `carbon_price_per_kg`, `toll_cost_per_km`)
- `terrain_profile` (`flat`, `rolling`, `hilly`)
- `stochastic` (`enabled`, `seed`, `sigma`, `samples`)
- `optimization_mode` (`expected_value`, `robust`)
- `risk_aversion`
- `emissions_context`
- `weather`
- `incident_simulation`
- `departure_time_utc`
- `pareto_method` (`dominance`, `epsilon_constraint`)
- `epsilon` (`duration_s`, `monetary_cost`, `emissions_kg`)
- ambiguity-context fields:
  - `od_ambiguity_index`
  - `od_ambiguity_confidence`
  - `od_engine_disagreement_prior`
  - `od_hard_case_prior`
  - `od_ambiguity_source_count`
  - `od_ambiguity_source_mix`
  - `od_ambiguity_source_mix_count`
  - `od_ambiguity_source_entropy`
  - `od_ambiguity_support_ratio`
  - `od_ambiguity_prior_strength`
  - `od_ambiguity_family_density`
  - `od_ambiguity_margin_pressure`
  - `od_ambiguity_spread_pressure`
  - `od_ambiguity_toll_instability`
  - `od_candidate_path_count`
  - `od_corridor_family_count`
  - `od_objective_spread`
  - `od_nominal_margin_proxy`
  - `od_toll_disagreement_rate`
  - `ambiguity_budget_prior`
  - `ambiguity_budget_band`

Endpoint-specific:

- `POST /route`: `pipeline_mode`, `refinement_policy`, `pipeline_seed`, `search_budget`, `evidence_budget`, `cert_world_count`, `certificate_threshold`, `tau_stop`, `evaluation_lean_mode`
  - live `/route` rejects `pipeline_mode=legacy` and waypoint requests; use `/route/baseline` or `/route/baseline/ors` for comparison, replay, historical-comparison, or waypoint flows
- `POST /route/preference`: `candidate_routes`, `selected_route_id`, `selected_certificate_basis`, `pipeline_mode`, `support_flag`, `support_reason`, `preference_state`
  - request/response models live in `backend/app/preference_runtime.py`
  - validation fails with `422` and `reason_code=preference_runtime_invalid_update` when `candidate_routes` cannot supply at least one usable route id
- `POST /pareto`: `pipeline_mode`, `pipeline_seed`
- `POST /batch/pareto`: `pairs` (1..500), optional `seed`, `toggles`, `model_version`, `pipeline_mode`, `pipeline_seed`, `search_budget`, `evidence_budget`, `cert_world_count`, `certificate_threshold`, `tau_stop`
- `POST /batch/import/csv`: `csv_text` plus the same optional controls as batch
- `POST /departure/optimize`: `window_start_utc`, `window_end_utc`, `step_minutes`, optional `time_window`
- `POST /duty/chain`: `stops` (2..50)
- `POST /scenario/compare`: `scenario_mode` is optional and defaults to the baseline comparison mode when omitted

## Response Surface Notes

- `RouteMetrics`
  - `distance_km`
  - `duration_s`
  - `monetary_cost`
  - `emissions_kg`
  - `avg_speed_kmh`
  - `energy_kwh`
  - `weather_delay_s`
  - `incident_delay_s`
- `RouteOption`
  - `geometry`
  - `metrics`
  - `knee_score`
  - `is_knee`
  - `eta_explanations`
  - `eta_timeline`
  - `segment_breakdown`
  - `counterfactuals`
  - `uncertainty`
  - `uncertainty_samples_meta`
  - `legs`
  - `toll_confidence`
  - `toll_metadata`
  - `vehicle_profile_id`
  - `vehicle_profile_version`
  - `vehicle_profile_source`
  - `scenario_summary`
  - `weather_summary`
  - `terrain_summary`
  - `incident_events`
- `DecisionPackage`
  - `terminal_type`
  - `selected`, `candidates`, `recommended_route`
  - `certified_set`, `abstention`
  - `frontier_summary`, `selected_certificate`, `certificate_summary`
  - `winner_confidence_state`, `pairwise_gap_states`, `flip_radius_state`, `decision_region_state`, `certificate_witness`
  - `preference_summary`, `preference_state`, `preference_query_trace`
  - `support_summary`, `world_support_summary`, `abstention_summary`, `certified_set_summary`
  - `action_trace_summary`, `witness_summary`, `artifact_pointers`
  - `selected_certificate_basis`, `run_id`, `pipeline_mode`
  - `manifest_endpoint`, `artifacts_endpoint`, `provenance_endpoint`
  - `voi_stop_summary`
- `ScenarioSummary`
  - `mode`
  - `context_key`
  - `duration_multiplier`
  - `incident_rate_multiplier`
  - `incident_delay_multiplier`
  - `fuel_consumption_multiplier`
  - `emissions_multiplier`
  - `stochastic_sigma_multiplier`
  - `source`
  - `version`
  - `calibration_basis`
  - `as_of_utc`
  - `live_as_of_utc`
  - `live_sources`
  - `live_coverage_overall`
  - `live_traffic_pressure`
  - `live_incident_pressure`
  - `live_weather_pressure`
  - `scenario_edge_scaling_version`
  - `mode_observation_source`
  - `mode_projection_ratio`
- `RouteBaselineResponse`
  - `baseline`
  - `method`
  - `compute_ms`
  - `provider_mode`
  - `baseline_policy`
  - `asset_manifest_hash`
  - `asset_recorded_at`
  - `asset_freshness_status`
  - `engine_manifest`
  - `notes`
- `ParetoResponse`
  - `routes`
  - `warnings`
  - `diagnostics`
  - common diagnostics keys include `candidate_count_raw`, `candidate_count_deduped`, `graph_explored_states`, `graph_generated_paths`, `graph_emitted_paths`, `candidate_budget`, `graph_effective_max_hops`, `graph_effective_hops_floor`, `graph_effective_state_budget_initial`, `graph_effective_state_budget`, `prefetch_*`, `scenario_gate_*`, `precheck_*`, `graph_retry_*`, `graph_rescue_*`, `pareto_count`, `dominated_count`, and `frontier_certificate`
- `PreferenceRuntimeUpdateResponse`
  - `selected_route_id`
  - `selected_certificate_basis`
  - `pipeline_mode`
  - `terminal_type`
  - `preference_state`
  - `preference_query_trace`
  - `preference_summary`
- `ScenarioCompareResponse`
  - `run_id`
  - `results`
  - `deltas`
  - `baseline_mode`
  - `scenario_manifest_endpoint`
  - `scenario_signature_endpoint`
- `DepartureOptimizeResponse`
  - `best`
  - `candidates`
  - `evaluated_count`
- `DutyChainResponse`
  - `legs`
  - `total_metrics`
  - `leg_count`
  - `successful_leg_count`
- `OracleQualityDashboardResponse`
  - `total_checks`
  - `source_count`
  - `stale_threshold_s`
  - `sources`
  - `updated_at_utc`

## Strict Error Shapes

### Non-stream endpoints (`422`)

```json
{
  "detail": {
    "reason_code": "terrain_dem_coverage_insufficient",
    "message": "Terrain DEM coverage below threshold.",
    "warnings": [
      "Coverage 0.91 < required 0.96"
    ],
    "stage": "collecting_candidates",
    "stage_detail": "terrain_fail_closed",
    "terrain_dem_version": "uk_dem_v1",
    "terrain_coverage_required": 0.96,
    "terrain_coverage_min_observed": 0.91
  }
}
```

### Stream Fatal Event (`POST /pareto/stream`, `POST /api/pareto/stream`)

```json
{
  "type": "fatal",
  "reason_code": "epsilon_infeasible",
  "message": "No routes satisfy epsilon constraints for this request.",
  "warnings": []
}
```

## Scenario Runtime Notes

- Scenario policy is context-conditioned and strict-validated.
- `LIVE_SCENARIO_COEFFICIENT_URL` is required in strict runtime.
- Context uses free UK feeds (WebTRIS, Traffic England, DfT raw counts, Open-Meteo).
- Missing/stale/incomplete strict context resolves to scenario reason codes (`scenario_profile_unavailable` / `scenario_profile_invalid`).
- Signed fallback is blocked by strict defaults (`LIVE_SCENARIO_ALLOW_SIGNED_FALLBACK=false` unless overridden for controlled scenarios).

Successful route payloads include additive scenario fields:

- `route.scenario_summary.*`
- uncertainty metadata keys such as `scenario_mode`, `scenario_profile_version`, `scenario_sigma_multiplier`, `scenario_context_key`

`POST /scenario/compare` specifics:

- `baseline_mode` is fixed to `no_sharing`
- `deltas` carry per-metric nullable deltas plus `*_status`, `*_reason_code`, `*_missing_source`, `*_reason_source`

## Vehicle Runtime Notes

- Built-in profiles are strict asset-backed (`backend/assets/uk/vehicle_profiles_uk.json`).
- Custom profiles are persisted in the runtime output config area (`backend/out/config/`, runtime file: vehicles_v2.json).
- Unknown or invalid `vehicle_type` in strict route-producing flows returns `422` with `vehicle_profile_unavailable` or `vehicle_profile_invalid`.

## Batch And Artifact Runtime Notes

- `POST /batch/pareto` processes pairs with bounded concurrency (`BATCH_CONCURRENCY`).
- Batch outputs include per-pair `error` strings in strict format (`reason_code:...; message:...`) when a pair cannot produce routes.
- The route bundle writes these artifact files by default:
  - results.json
  - results.csv
  - metadata.json
  - index.json
  - index.md
  - routes.geojson
  - results_summary.csv
- The evaluation bundle also emits extended artifacts when the run enables them:
  - dccs_candidates.jsonl
  - dccs_summary.json
  - refined_routes.jsonl
  - strict_frontier.jsonl
  - winner_summary.json
  - certificate_summary.json
  - route_fragility_map.json
  - competitor_fragility_breakdown.json
  - value_of_refresh.json
  - sampled_world_manifest.json
  - evidence_snapshot_manifest.json
  - voi_action_trace.json
  - voi_controller_state.jsonl
  - voi_action_scores.csv
  - voi_stop_certificate.json
  - final_route_trace.json
  - od_corpus.csv
  - od_corpus.json
  - od_corpus_summary.json
  - od_corpus_rejected.json
  - ors_snapshot.json
  - thesis_results.csv
  - thesis_results.json
  - thesis_summary.csv
  - thesis_summary.json
  - thesis_summary_by_transfer_slice.csv
  - thesis_summary_by_transfer_slice.json
  - thesis_summary_by_weather_regime_transfer_slice.csv
  - thesis_summary_by_weather_regime_transfer_slice.json
  - thesis_metrics.json
  - thesis_plots.json
  - methods_appendix.md
  - thesis_report.md
  - evaluation_manifest.json

- `index.json` is the machine-readable route-bundle index. It summarizes run id, terminal type, selected certificate basis, support state, artifact pointers, and per-artifact endpoints for the current run folder.
- `index.md` is the reviewer-readable companion summary for the same route bundle.
- The public `/route` response is the response contract. In the current smoke-covered direct-REFC slice, that response path can normalize to `terminal_type=typed_abstention` even when the emitted local run bundle preserves richer proof artifacts such as `decision_package.json` and `certified_set_summary.json`.
- In that same checked local slice, the public `typed_abstention` response can still carry normalized explanation surfaces such as `certified_set_summary`, `winner_confidence_state`, `certificate_witness`, and `artifact_pointers`; those fields remain part of the response contract even when the emitted bundle keeps a richer local proof surface.
- Operators who need that richer local proof surface should inspect those emitted artifacts through `GET /runs/{run_id}/artifacts/{artifact_name}`. This is a narrow note about the checked local smoke slice, not a global claim about every `/route` abstention.
- Older focused-VOI and legacy-named evaluation export bundles may still carry `thesis_summary_by_cohort.*`; that is an export/history surface, not a current `ARTIFACT_FILES` entry. Current transfer-generalization artifact names are `thesis_summary_by_transfer_slice.*` and `thesis_summary_by_weather_regime_transfer_slice.*`.

For legacy-named evaluation bundles emitted or rewritten through the run store, `index.json` now uses `bundle_type = thesis_evaluation` and includes an `export_status` section for maintained report artifacts; `index.md` mirrors that same presence/absence view as an `Export Status` section. In the maintained set, this can include files such as `thesis_results.csv`, `thesis_summary.csv`, the summary-by-cohort CSV, `thesis_report.md`, and `evaluation_manifest.json`. Read that behavior as forward-looking for new or rewritten bundles only: it does not imply historical backfill for older checked directories that predate the newer bundle-index emission path.
- Signed manifests are written to:
  - `backend/out/manifests/{run_id}.json`
  - `backend/out/scenario_manifests/{run_id}.json`

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [API Cookbook](api-cookbook.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
