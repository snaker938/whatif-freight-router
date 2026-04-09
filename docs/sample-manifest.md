# Sample Manifest and Outputs

Last Updated: 2026-04-09
Applies To: `POST /batch/pareto`, run manifests, signatures, artifact endpoints, and thesis evaluation bundles

## Sample Files in Repo

- Request sample: `docs/examples/sample_batch_request.json`
- Response sample: `docs/examples/sample_batch_response.json`
- Manifest sample: `docs/examples/sample_manifest.json`

## Runtime Manifest And Artifact Outputs

On successful batch/scenario compare flows, backend writes:

- manifest: `backend/out/manifests/{run_id}.json`
- scenario manifest: `backend/out/scenario_manifests/{run_id}.json`
- artifact folder: `backend/out/artifacts/{run_id}/`
- provenance stream: `backend/out/provenance/{run_id}.jsonl`

Core route bundle files:

- results.json
- results.csv
- metadata.json
- routes.geojson
- results_summary.csv

Thesis and VOI bundle files currently emitted by the run store:

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
- thesis_summary_by_cohort.csv
- thesis_summary_by_cohort.json
- thesis_metrics.json
- thesis_plots.json
- methods_appendix.md
- thesis_report.md
- evaluation_manifest.json

## Current Manifest Shape

Route compute manifests written by `_write_route_run_bundle()` include:

- `schema_version`
- `type`
- `request`
- `pipeline`
- `selected_route_id`
- `selected_certificate`
- `voi_stop_summary`
- `warnings`
- `candidate_diagnostics`
- `execution`

The companion metadata.json includes:

- `run_id`
- `schema_version`
- `type`
- `request_id`
- `pipeline_mode`
- `run_seed`
- `manifest_endpoint`
- `artifacts_endpoint`
- `provenance_endpoint`
- `provenance_file`
- `artifact_names`
- `selected_route_id`
- `candidate_count`
- `warning_count`
- `duration_ms`

The route-level results.json contains:

- `run_id`
- `selected`
- `candidates`
- `warnings`
- `candidate_diagnostics`

Batch results.csv rows use these columns:

- `pair_index`
- `origin_lat`
- `origin_lon`
- `destination_lat`
- `destination_lon`
- `error`
- `route_id`
- `distance_km`
- `duration_s`
- `monetary_cost`
- `emissions_kg`
- `avg_speed_kmh`

Route bundle results.csv rows add selected as an extra column.

## Retrieval Flow

1. Submit `POST /batch/pareto`.
2. Extract `run_id` from response.
3. Fetch signed metadata:
   - `GET /runs/{run_id}/manifest`
   - `GET /runs/{run_id}/scenario-manifest`
4. Fetch signatures:
   - `GET /runs/{run_id}/signature`
   - `GET /runs/{run_id}/scenario-signature`
5. Enumerate/download outputs:
   - `GET /runs/{run_id}/artifacts`
   - `GET /runs/{run_id}/artifacts/<name>`

## Latest Thesis Bundle Snapshot

A current thesis evaluation bundle in `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/` is one of the most complete currently checked local evidence sets. Its evaluation_manifest.json records:

- `run_id`: `thesis_eval_20260331_r2_focused_voi`
- `created_at`: `2026-04-01T01:09:48.771286+01:00`
- `model_version`: `thesis-script-v3`
- `strict_evidence_policy`: `no_synthetic_no_proxy_no_fallback`
- `cache_mode`: `cold`
- `cache_reset_policy`: `thesis_cold`
- `cache_reset_scope`: `variant`
- `cache_reset_count`: `80`
- `cache_carryover_expected`: `false`
- `run_validity.scenario_profile_unavailable_rate`: `0.0`
- `run_validity.strict_live_readiness_pass_rate`: `1.0`
- `run_validity.evaluation_rerun_success_rate`: `1.0`

The same manifest also records the strict readiness evidence:

- `backend_ready_summary.status`: `ready`
- `backend_ready_summary.strict_route_ready`: `true`
- `backend_ready_summary.route_graph.nodes_seen`: `16782614`
- `backend_ready_summary.route_graph.edges_seen`: `32920150`
- `backend_ready_summary.route_graph.largest_component_ratio`: `0.9274120825277874`
- `backend_ready_summary.route_graph.elapsed_ms`: `290573.79`
- `backend_ready_summary.route_graph.asset_size_mb`: `4123.27`
- `backend_ready_summary.strict_live.dependency_count`: `7`
- `backend_ready_summary.strict_live.dependencies[0].details.contexts`: `192`
- `backend_ready_summary.strict_live.dependencies[1].as_of_utc`: `2026-03-23T00:00:00+00:00`
- `backend_ready_summary.strict_live.dependencies[2].details.rule_count`: `220`
- `backend_ready_summary.strict_live.dependencies[3].details.segment_count`: `28`
- `backend_ready_summary.strict_live.dependencies[4].details.regime_count`: `18`
- `backend_ready_summary.strict_live.dependencies[5].details.region_count`: `11`
- `backend_ready_summary.strict_live.dependencies[6].details.count`: `134`
- `backend_ready_summary.compute_ms`: `2504.54`
- `backend_ready_summary.wait_elapsed_ms`: `284592.65`
- `backend_ready_summary.attempt_count`: `29`

The same bundle’s baseline smoke summary records:

- OSRM `compute_ms`: `943.84`
- OSRM `distance_km`: `189.471`
- OSRM `duration_s`: `13533.31`
- ORS `compute_ms`: `335.31`
- ORS `distance_km`: `203.868`
- ORS `duration_s`: `18898.8`
- ORS `asset_manifest_hash`: `6bbc27f2cff7983598de1ee9fe5272c67b4b3fab6c732dd696d909151261d063`
- ORS `asset_freshness_status`: `graph_identity_verified`
- ORS `engine_image`: `openrouteservice/openrouteservice:v9.7.1`

## Report-Level Outputs In The Thesis Bundle

The thesis evaluation folder includes the derived research outputs as files, not just JSON manifests:

- thesis_summary.json
- thesis_summary.csv
- thesis_summary_by_cohort.json
- thesis_summary_by_cohort.csv
- thesis_metrics.json
- thesis_plots.json
- methods_appendix.md
- thesis_report.md

These files are the best place to inspect the latest aggregate metrics such as:

- success rate
- route evidence completion rate
- dominance and weighted win rates against OSRM, ORS, and the best baseline
- frontier size, diversity, entropy, and hypervolume
- runtime percentiles and runtime ratios vs OSRM/ORS
- ambiguity alignment, budget utilization, and VOI controller engagement
- cache reuse and warmup costs
- baseline acquisition runtime

## Example Commands

```powershell
$body = Get-Content docs/examples/sample_batch_request.json -Raw
$resp = Invoke-RestMethod -Uri "http://localhost:8000/batch/pareto" -Method Post -ContentType "application/json" -Body $body
$runId = $resp.run_id

Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-manifest"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/scenario-signature"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/artifacts"
Invoke-RestMethod -Uri "http://localhost:8000/runs/$runId/artifacts/results.json"
Invoke-WebRequest -Uri "http://localhost:8000/runs/$runId/artifacts/report.pdf" -OutFile ".\report_$runId.pdf"
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [API Cookbook](api-cookbook.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
