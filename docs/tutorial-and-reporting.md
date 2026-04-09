# Tutorial Mode and Reporting

Last Updated: 2026-04-09
Applies To: frontend tutorial flows, devtools reporting surfaces, run artifacts, and signature inspection

This page describes the current frontend tutorial/reporting flow and the backend artifact surfaces it depends on.

The evaluator/reporting pipeline that produces the thesis bundles carries `cohort_scaffolding` and `metric_family_scaffolding` through `evaluation_manifest.json`, `thesis_metrics.json`, `thesis_plots.json`, `results.json`, and `metadata.json`, using `thesis_cohort_scaffolding_v2` and `thesis_metric_family_scaffolding_v1` as the current scaffold versions.

Current checked-in reporting anchors are the repaired broad-cold 2026-04-04 bundle, the focused REFC/VOI and hot-rerun 2026-03-31 bundles, and the 2026-04-06 harder-story / campaign tranche bundles under `backend/out/thesis_campaigns/`. The staged `1200`-OD shard assets under `backend/data/eval/thesis_shards_1200/` are corpus inputs; the matching `backend/out/thesis_1200_s01_repo_local/` directory is still staging-only because it currently contains preflight plus baseline smoke rather than a complete thesis report family.

## Tutorial Scope

The guided flow covers:

- route and Pareto generation
- baseline comparison
- scenario compare
- departure optimization
- duty chain planning
- oracle quality recording
- experiment save/load/replay flows
- run artifact inspection and signature verification

The tutorial overlay is implemented separately from generated thesis reports. It is UI guidance, not thesis proof.

## Frontend Reporting Surfaces

Current reporting-adjacent UI surfaces include:

- Run Inspector
- Signature Verifier
- Ops Diagnostics
- Oracle Quality Dashboard
- Experiment Manager
- Route Certification Panel
- Route Baseline Comparison
- Scenario Comparison
- route readiness and compute trace overlays on the main page
- current-profile vs academic-selection comparison on the main page

## Backend Reporting Endpoints Used By UI

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/provenance`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `GET /runs/{run_id}/artifacts`
- `GET /runs/{run_id}/artifacts/{artifact_name}`
- `POST /verify/signature`
- `GET /oracle/quality/dashboard`
- `GET /oracle/quality/dashboard.csv`
- `POST /oracle/quality/check`

## Run Inspector

The current Run Inspector can:

- inspect manifest, scenario manifest, provenance, signature, and scenario signature
- list the stable public artifact set for a run
- preview bounded allowlisted artifact text inline through the frontend proxy
- download core docs and the same allowlisted artifact subset by name
- render `decision_package` summaries inline when the inspected `run_id` matches the current active route run
- open directly from route certification and scenario comparison flows when a run id is present

This is no longer a PDF-centered flow. Common proxy-backed inspection targets now include:

- dccs_summary.json
- `dccs_summary.json` is the route-side DCCS summary artifact; it now carries `control_state`, and the companion `dccs_candidates.jsonl` / `strict_frontier.jsonl` exports carry the row-level DCCS vocabulary for `safe_elimination_reason`, `dominance_margin`, `dominating_candidate_ids`, `dominated_candidate_ids`, `search_deficiency_score`, `hidden_challenger_score`, `anti_collapse_quota`, and `long_corridor_search_completeness`
- `dccs_candidates.jsonl`
- `refined_routes.jsonl`
- `strict_frontier.jsonl`
- winner_summary.json
- certificate_summary.json
- route_fragility_map.json
- competitor_fragility_breakdown.json
- thesis_report.md
- methods_appendix.md
- value_of_refresh.json
- sampled_world_manifest.json
- voi_action_trace.json
- `voi_action_scores.csv`
- voi_stop_certificate.json
- final_route_trace.json
- `thesis_results.csv`
- thesis_summary.csv
- `od_corpus.csv`
- `od_corpus_summary.json`
- `ors_snapshot.json`

For the currently active route response, the Route Certification Panel and Run Inspector can also surface `decision_package`-derived preference, support, world-support, world-fidelity, certification-state, certified-set, abstention, witness, controller, theorem-hook, and lane-manifest summaries directly from route state. That is a separate live-route inspection path from the bounded proxy-based artifact browsing flow above, and it does not imply that every persisted backend artifact is previewable through the frontend.

Not currently proxy-previewable in detached mode are the persisted decision/support/controller/theorem bundle files such as `decision_package.json`, `preference_summary.json`, `support_summary.json`, `support_provenance.json`, `certified_set.json`, `abstention_summary.json`, `witness_summary.json`, `controller_summary.json`, `controller_trace.jsonl`, `voi_controller_state.jsonl`, `voi_controller_trace_summary.json`, `voi_replay_oracle_summary.json`, `theorem_hook_map.json`, `lane_manifest.json`, and `evaluation_manifest.json`.

In that live-route summary path, `world_fidelity_summary.unique_world_count` remains the deduplicated manifest count and `world_fidelity_summary.effective_world_count` remains the support-aware certification count. The witness panel stays compact and may include a derived `top_fragility_family=...` note sourced from `route_fragility_map.json`, while the detailed per-route family values remain in the artifact bundle.

## Signature Verification

The Signature Verifier accepts:

- JSON payloads
- string payloads
- a required signature
- an optional secret override

It returns the backend verification response so operators can compare `valid` and `expected_signature`.

## Practical Operator Flow

1. trigger a compute action in the UI
2. capture the returned `run_id`
3. inspect the active Route Certification Panel when you need the current route's `decision_package` summaries immediately
4. open Run Inspector and load manifest, provenance, signatures, and any proxy-allowlisted artifacts needed for debugging or reporting
5. use Signature Verifier when you need detached verification of a manifest payload
6. use Route Baseline Comparison when you need OSRM, ORS, or academic-profile deltas in the browser
7. use Route Certification Panel or Scenario Comparison when you need a direct handoff into run-level inspection

## Local Commands

Frontend:

```powershell
pnpm -C frontend dev
```

Backend headless scenario smoke:

```powershell
uv run --project backend python backend/scripts/run_headless_scenario.py --input-json docs/examples/sample_batch_request.json
```

Evaluation/report outputs:

```powershell
uv run --project backend python backend/scripts/run_thesis_evaluation.py
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
uv run --project backend python backend/scripts/compose_thesis_suite_report.py
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Frontend Dev Tools Coverage](frontend-dev-tools.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Backend APIs and Tooling](backend-api-tools.md)
