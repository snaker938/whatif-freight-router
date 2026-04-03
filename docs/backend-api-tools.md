# Backend APIs and Tooling

Last Updated: 2026-04-03
Applies To: `backend/app/main.py`, `backend/app/models.py`, `backend/app/run_store.py`, `backend/app/settings.py`

This page is the authored summary of the current backend API and stable run-store contract.

## Runtime Contract

- strict runtime defaults are enforced in `backend/app/settings.py`
- route-producing failures are reason-coded and normalized with `normalize_reason_code` in `backend/app/model_data_errors.py`
- streaming and non-streaming route flows use aligned reason-code semantics
- self-hosted OSRM and self-hosted ORS remain the approved comparator engines

## Endpoint Inventory

### System and Admin

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

### Routing and Pareto

- `POST /route`
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

### Run Artifacts and Signatures

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

## Request Surface

All route-producing requests derive from `RouteRequest`, `ParetoRequest`, `BatchParetoRequest`, or related wrappers in `backend/app/models.py`.

### Shared Route and Pareto Fields

- `origin`, `destination`, optional `waypoints`
- `vehicle_type`
- `scenario_mode`
- `max_alternatives`
- `weights`
- `cost_toggles`
- `terrain_profile`
- `stochastic`
- `optimization_mode`
- `risk_aversion`
- `emissions_context`
- `weather`
- `incident_simulation`
- `departure_time_utc`
- `pareto_method`
- `epsilon`

### Thesis / Pipeline Controls

Route-producing requests may also carry:

- `pipeline_mode`
- `pipeline_seed`
- `search_budget`
- `evidence_budget`
- `cert_world_count`
- `certificate_threshold`
- `tau_stop`

`POST /route` additionally supports:

- `refinement_policy`
- `evaluation_lean_mode`
- upstream ambiguity-prior fields such as `od_ambiguity_index`, `od_ambiguity_confidence`, `od_engine_disagreement_prior`, and `od_hard_case_prior`

`POST /batch/pareto` additionally supports:

- `pairs`
- `waypoints`
- `seed`
- `toggles`
- `model_version`
- `pipeline_mode`
- `pipeline_seed`
- `search_budget`
- `evidence_budget`
- `cert_world_count`
- `certificate_threshold`
- `tau_stop`

`POST /batch/import/csv` wraps `csv_text` plus the same route and pipeline controls supported by the batch request family. It also accepts the same route-level fields as `POST /batch/pareto`, except that the OD pairs arrive via CSV rows rather than `pairs`.

`POST /departure/optimize` additionally supports:

- `window_start_utc`
- `window_end_utc`
- `step_minutes`
- optional `time_window`

`POST /duty/chain` uses `stops` rather than a single origin/destination pair.

## Response Surface

### `POST /route`

`RouteResponse` returns:

- `selected`
- `candidates`
- `run_id`
- `pipeline_mode`
- `manifest_endpoint`
- `artifacts_endpoint`
- `provenance_endpoint`
- `decision_package`
- `selected_certificate`
- `voi_stop_summary`

`decision_package` is optional at the response-contract level, but the current route runtime wires it from the same decision/support/controller bundle that is also persisted as `decision_package.json`.

Returned `RouteOption` objects may include richer fields such as:

- `knee_score`, `is_knee`
- `eta_explanations`, `eta_timeline`
- `segment_breakdown`
- `counterfactuals`
- `uncertainty`, `uncertainty_samples_meta`
- `legs`
- `toll_confidence`, `toll_metadata`
- `scenario_summary`, `weather_summary`, `terrain_summary`
- `incident_events`
- `evidence_provenance`
- `certification`

### Baseline Routes

`RouteBaselineResponse` returns:

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

### Scenario Compare

`ScenarioCompareResponse` returns:

- `run_id`
- `results`
- `deltas`
- `baseline_mode`
- `scenario_manifest_endpoint`
- `scenario_signature_endpoint`

### Batch

`BatchParetoResponse` remains intentionally compact:

- `run_id`
- `results`

Per-pair batch failures are serialized into the `error` field using strict text like `reason_code:<code>; message:<message>`.

### Signature Verification

`SignatureVerificationRequest` accepts:

- `payload` as JSON or string
- `signature`
- optional `secret`

`SignatureVerificationResponse` returns:

- `valid`
- `algorithm`
- `signature`
- `expected_signature`

## Stable Artifact Contract

The stable public run-store allowlist comes from `ARTIFACT_FILES` in `backend/app/run_store.py`.

### Core Outputs

- results.json
- results.csv
- metadata.json
- routes.geojson
- results_summary.csv

### DCCS / Frontier Outputs

- dccs_candidates.jsonl
- dccs_summary.json
- refined_routes.jsonl
- strict_frontier.jsonl

### Decision / Controller / Support Outputs

- decision_package.json
- preference_summary.json
- support_summary.json
- support_trace.jsonl
- support_provenance.json
- certified_set.json
- certified_set_routes.jsonl
- abstention_summary.json
- witness_summary.json
- witness_routes.jsonl
- controller_summary.json
- controller_trace.jsonl
- theorem_hook_map.json
- lane_manifest.json

### REFC / VOI Outputs

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

### Corpus / Evaluation Outputs

- od_corpus.csv
- od_corpus.json
- od_corpus_summary.json
- od_corpus_rejected.json
- ors_snapshot.json
- thesis_results.csv
- thesis_results.json
- thesis_summary.csv
- thesis_summary.json
- thesis_metrics.json
- thesis_plots.json
- methods_appendix.md
- thesis_report.md
- evaluation_manifest.json

The run-store schema registry now covers the decision/controller/support family above plus the additive VOI JSON surfaces `voi_action_trace.json`, `voi_stop_certificate.json`, and `final_route_trace.json`. Dict-backed JSON artifacts receive inline `schema_version` values when written. JSONL artifacts remain line-oriented files, but their artifact names are still tracked in the same registry so public artifact contracts can evolve additively without renaming files.

Manifests are signed and written to:

- `backend/out/manifests/<run_id>.json`
- `backend/out/scenario_manifests/<run_id>.json`

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
    "terrain_dem_version": "uk_dem_v1",
    "terrain_coverage_required": 0.96,
    "terrain_coverage_min_observed": 0.91
  }
}
```

### Stream fatal event

```json
{
  "type": "fatal",
  "reason_code": "epsilon_infeasible",
  "message": "No routes satisfy epsilon constraints for this request.",
  "warnings": []
}
```

See the dedicated strict error reference for the exact frozen reason-code set.

## Scenario and Vehicle Notes

- strict scenario context is validated against live/source freshness and schema checks
- successful routes may expose `scenario_summary` and uncertainty metadata
- custom vehicles persist under `backend/out/config/` as a runtime-generated JSON store
- unknown or invalid vehicles map to `vehicle_profile_unavailable` or `vehicle_profile_invalid`

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [API Cookbook](api-cookbook.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
