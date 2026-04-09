# Carbon-Aware Multi-Objective Freight Router

This repository contains a local-first freight-routing stack with:

- self-hosted OSRM and ORS baseline engines
- a FastAPI backend with strict live/runtime gates and signed run artifacts
- a Next.js frontend with route tooling, dual OSRM/ORS baseline comparison, academic comparator overlays, certificate/artifact inspection, diagnostics, tutorial flows, and devtools
- thesis-grade evaluation workflows for DCCS, REFC, and VOI-AD2R
- a separate hot-rerun benchmark for production-style cache/reuse validation

## Repo Layout

- `backend/`: FastAPI app, evaluation scripts, model assets, and runtime outputs
- `frontend/`: Next.js app, API proxy routes, map UI, tutorial system, and devtools
- `docs/`: authored documentation
- `notebooks/`: notebook policy only; the repo is maintained as a script-first workflow
- `osrm/`, `ors/`: local baseline-engine infrastructure and graph data
- `scripts/`: repo-level developer helpers and docs validation

## Prerequisites

- Docker Desktop
- PowerShell
- Python with `uv`
- Node.js with `pnpm`

## Recommended Local Development Flow

From repo root:

```powershell
.\scripts\dev.ps1
```

This is the preferred local startup path. It runs strict live preflight, boots the local routing stack, and starts backend/frontend services when needed.

Key URLs:

- Frontend: `http://localhost:3000`
- Backend OpenAPI docs: `http://localhost:8000/docs`
- Backend readiness: `http://localhost:8000/health/ready`
- OSRM: `http://localhost:5000`
- ORS health: `http://localhost:8082/ors/v2/health`

## Full Container Stack

Run the complete Dockerized stack:

```powershell
docker compose up --build
```

Do not run this at the same time as `.\scripts\dev.ps1`.

## Environment and Runtime Notes

- Copy `.env.example` to `.env` if you do not already have one.
- Strict startup runs `backend/scripts/preflight_live_runtime.py` before backend launch.
- Backend strict route readiness and strict live readiness both surface through `GET /health/ready`; evaluator summary artifacts now add `route_graph_readiness_class`, `route_graph_full_hydration_observed`, `degraded_evaluation_observed`, `degraded_reason_codes_observed`, `precheck_gate_actions_observed`, `route_fallback_observed`, and `strict_full_search_proof_eligible` so `ok_fast` lanes are not mistaken for full-hydration strict-search proof.
- `POST /route` now defaults to the public `tri_source` lane when `pipeline_mode` is omitted. The landed thesis coordinator currently executes the `tri_source` path through `voi` internals for single-leg requests, and waypoint requests still fall back to `legacy`.
- The public `pipeline_mode` reported in `RouteResponse`, `DecisionPackage`, manifests, and rewritten artifacts stays `tri_source` for that default lane even when the internal runtime path is `voi`.
- `RouteResponse` still returns `selected`, `candidates`, `selected_certificate`, and `voi_stop_summary`, and now also includes `decision_package`.
- `decision_package` is the public decision summary mirror. Its public `terminal_kind` contract is now exactly `certified_singleton`, `certified_set`, or `typed_abstention`, and it also carries `preference_summary`, `support_summary`, `world_support_summary`, `world_fidelity_summary`, `certification_state_summary`, `certified_set_summary`, `abstention_summary`, `witness_summary`, `controller_summary`, `theorem_hook_summary`, and `lane_manifest`.
- Waypoint compatibility fallback remains explicit routing behavior. When `/route` has to execute through the legacy runtime for waypoint support, that is surfaced through `pipeline_mode`, provenance, warnings, and `abstention_summary.reason_code=legacy_runtime_selected`, not through a fourth public terminal outcome.
- `decision_package.preference_summary` is the landed preference bridge. It is summary-only selector/runtime metadata, includes suggestion hints, and is not a public preference query/update API.
- The frontend route result now treats `RouteCertificationPanel` plus `RunInspector` as the primary certificate/artifact inspection path. The browser groups artifacts into route-core, decision-package, support, DCCS/REFC, controller/VOI, witness/fragility, theorem/lane, and evaluation families using existing artifact names from the run store rather than any extra backend-only fields.
- Current compute fallback defaults in `.env.example` are intentionally long-running for thesis/evaluation workloads:
  - `ROUTE_COMPUTE_ATTEMPT_TIMEOUT_S=1200`
  - `COMPUTE_ATTEMPT_TIMEOUT_MS=1200000`
  - `COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS=900000`
  - `NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS=1200000`
  - `NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS=900000`
  - `LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT=false`
- Current live refresh and strict route-compute controls are driven by:
  - `LIVE_ROUTE_COMPUTE_REFRESH_MODE`
  - `LIVE_ROUTE_COMPUTE_REQUIRE_ALL_EXPECTED`
  - `LIVE_ROUTE_COMPUTE_FORCE_NO_CACHE_HEADERS`
  - `LIVE_ROUTE_COMPUTE_FORCE_UNCACHED`
  - `LIVE_ROUTE_COMPUTE_PREFETCH_TIMEOUT_MS`
  - `LIVE_ROUTE_COMPUTE_PREFETCH_MAX_CONCURRENCY`
  - `LIVE_ROUTE_COMPUTE_PROBE_TERRAIN`
- Current graph warmup and strict graph controls are driven by:
  - `ROUTE_GRAPH_WARMUP_ON_STARTUP`
  - `ROUTE_GRAPH_WARMUP_FAILFAST`
  - `ROUTE_GRAPH_WARMUP_TIMEOUT_S`
  - `ROUTE_GRAPH_STATUS_CHECK_TIMEOUT_MS`
  - `ROUTE_GRAPH_FAST_STARTUP_ENABLED`
  - `ROUTE_GRAPH_MIN_GIANT_COMPONENT_NODES`
  - `ROUTE_GRAPH_MIN_GIANT_COMPONENT_RATIO`
  - `ROUTE_GRAPH_MAX_NEAREST_NODE_DISTANCE_M`
- In strict runtime, settings validation forces strict live data on, requires expected live route-compute sources, requires strict graph readiness, and disables fast graph startup.

## Important Scripts

Repo-level:

- `.\scripts\dev.ps1`: preferred local startup
- `.\scripts\clean.ps1`: clean Docker/OSRM caches and force rebuild
- `.\scripts\run_backend_tests_safe.ps1`: low-resource backend test runner
- `.\scripts\serve_docs.ps1`: local markdown docs viewer
- `python scripts/check_docs.py`: docs path/link/endpoint/index consistency checks
- `.\scripts\demo_repro_run.ps1`: reproducibility demo path

Backend build and evaluation:

- `uv run --project backend python backend/scripts/build_model_assets.py`
- `uv run --project backend python backend/scripts/publish_live_artifacts_uk.py`
- `uv run --project backend python backend/scripts/preflight_live_runtime.py`
- `uv run --project backend python backend/scripts/run_headless_scenario.py`
- `uv run --project backend python backend/scripts/run_thesis_lane.py`
- `uv run --project backend python backend/scripts/run_thesis_evaluation.py`
- `uv run --project backend python backend/scripts/run_thesis_campaign.py`
- `uv run --project backend python backend/scripts/expand_thesis_broad_corpus.py`
- `uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py`
- `uv run --project backend python backend/scripts/compare_thesis_runs.py`
- `uv run --project backend python backend/scripts/compose_thesis_sharded_report.py`
- `uv run --project backend python backend/scripts/compose_thesis_suite_report.py`
- `uv run --project backend python backend/scripts/check_eta_concept_drift.py`

## Evaluation Workflows

- Use `backend/scripts/run_thesis_lane.py` for smaller lane-local checks and focused iteration.
- Use `backend/scripts/run_thesis_evaluation.py` for the broad cold thesis proof, focused REFC proof, focused VOI proof, and DCCS diagnostic probe. Evaluator runs also emit cohort-scaffolded metadata such as `evaluation_suite`, `thesis_cohort_scaffolding_v2`, the explicit cohort labels (`collapse_prone`, `osrm_brittle`, `ors_brittle`, `refresh_sensitive`, `time_preserving_conflict`, `low_ambiguity_fast_path`, `preference_sensitive`, `support_fragile`, `audit_heavy`, `proxy_friendly`), `thesis_summary_by_cohort.csv/json`, and `cohort_composition.json`.
- Use `backend/scripts/run_thesis_campaign.py` when you need sequential OD widening with regression carry-forward. It bootstraps a known-green corpus, adds only a small batch of unseen ODs per tranche, reruns the prior green ODs on every widening step, treats any expected OD missing from evaluator output as a tranche failure, and writes a persistent campaign ledger with gate config, evaluated-vs-expected OD inventory, artifact-evidence status, tranche-local filtered corpora, and per-OD pass/fail summaries alongside the underlying thesis-run artifacts.
- Use `backend/scripts/run_hot_rerun_benchmark.py` for the dedicated second-run reuse benchmark and hot-vs-cold comparison artifacts.
- Use `backend/scripts/compose_thesis_suite_report.py` to compose completed evaluation lanes into a final suite report with cohort summaries, cohort composition, suite-source provenance, and prior coverage summaries.

## Current Checked-In Result Anchors

- repaired broad cold headline bundle: `backend/out/thesis_eval_core120_green_repaired/artifacts/thesis_eval_core120_green_repaired_20260404/` with `120` ODs / `480` evaluated rows / `0` failures; `weighted_win_rate_v0` is `0.541667` for `A`, `0.55` for `B`, and `0.633333` for `C`, while `C` reports `mean_voi_realized_certificate_lift=0.168752`
- focused REFC anchor: `backend/out/artifacts/refc_focus_20260331_h2/`; variant `C` reports `mean_certificate=0.870634`, `mean_voi_realized_certificate_lift=0.165028`, `voi_controller_engagement_rate=0.9`, and `mean_runtime_ms=8520.1158`
- focused VOI anchor: `backend/out/artifacts/thesis_eval_20260331_r2_focused_voi/`; variant `C` reports `mean_certificate=0.861459`, `mean_voi_realized_certificate_lift=0.097638`, `mean_voi_action_count=1.3`, and `voi_controller_engagement_rate=0.8`
- hot-rerun anchor pair: `backend/out/artifacts/hot_full_20260331_f2_cold/` and `backend/out/artifacts/hot_full_20260331_f2_hot/`; `mean_runtime_ms` drops from `18903.988263 -> 1629.783632` for `V0`, `22722.158474 -> 2537.876947` for `A`, `6407.758 -> 2247.957474` for `B`, and `10213.524474 -> 2122.009263` for `C`, while `mean_refc_world_reuse_rate` rises from `0.0` to `1.0` for `B` and `C`
- harder-story / widening anchors: `backend/out/thesis_campaigns/hard_mixed24_corr12p5_t4_inproc_r4/tranche_001/artifacts/hard_mixed24_corr12p5_t4_inproc_r4_t001/` is the current fully green proof-grade harder-story tranche, while `longcorr_cardiff_newcastle4_corr12p5_r1` and `dominance_cluster5_cardiff_bath_corr12p5_r2` are current partially green but not proof-eligible campaign anchors
- staged broadening assets: `backend/data/eval/uk_od_corpus_thesis_broad_expanded_1200.csv` and `backend/data/eval/thesis_shards_1200/*.csv` are checked in, but `backend/out/thesis_1200_s01_repo_local/artifacts/thesis_1200_s01_repo_local/` currently contains only `repo_asset_preflight.json` and `baseline_smoke_summary.json`, so it should not be cited as a completed thesis result

## Runtime Outputs

Current runtime and evaluation outputs are written under `backend/out/`:

- `backend/out/manifests/`
- `backend/out/scenario_manifests/`
- `backend/out/provenance/`
- `backend/out/artifacts/<run_id>/`
- `backend/out/model_assets/`
- `backend/out/test_runs/<timestamp>/`

Completed runs are typically inspected through signed manifests and the artifact API rather than by opening files directly:

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `GET /runs/{run_id}/artifacts`
- `GET /runs/{run_id}/artifacts/{artifact_name}`

Hot-rerun cache state can be restored separately through `POST /cache/hot-rerun/restore` when you are preparing a dedicated reuse benchmark.

Per-run artifact directories may include:

- core outputs: results.json, results.csv, metadata.json, routes.geojson, results_summary.csv
- decision-package outputs: decision_package.json, preference_summary.json, support_summary.json, support_provenance.json, support_trace.jsonl, certified_set.json, certified_set_routes.jsonl, final_route_trace.json, plus conditional files such as abstention_summary.json, witness_summary.json, witness_routes.jsonl, controller_summary.json, controller_trace.jsonl, voi_action_trace.json, voi_controller_state.jsonl, voi_controller_trace_summary.json, voi_replay_oracle_summary.json, theorem_hook_map.json, and lane_manifest.json when the corresponding runtime state exists
- DCCS outputs: `dccs_summary.json`, `dccs_candidates.jsonl`, and `strict_frontier.jsonl` are the main DCCS export contract; `dccs_summary.json` now carries `control_state`, and `dccs_candidates.jsonl` / `strict_frontier.jsonl` carry the row-level DCCS vocabulary for `safe_elimination_reason`, `dominance_margin`, `dominating_candidate_ids`, `dominated_candidate_ids`, `search_deficiency_score`, `hidden_challenger_score`, `anti_collapse_quota`, and `long_corridor_search_completeness`. `refined_routes.jsonl` remains the downstream refinement output. These surfaces are instrumented for later validation, not empirically cleared in this pass
- REFC outputs: winner_summary.json, certificate_summary.json, route_fragility_map.json, competitor_fragility_breakdown.json, sampled_world_manifest.json, evidence_snapshot_manifest.json
- VOI outputs: value_of_refresh.json, voi_action_trace.json, voi_controller_state.jsonl, voi_action_scores.csv, voi_stop_certificate.json, final_route_trace.json
- evaluation outputs: thesis_results.*, thesis_summary.*, thesis_summary_by_cohort.*, thesis_metrics.json, thesis_plots.json, cohort_composition.json, evaluation_manifest.json, with `metric_family_scaffolding` carried through `metadata.json`, `results.json`, `thesis_metrics.json`, `thesis_plots.json`, and `evaluation_manifest.json`; summary `metadata.json` and `evaluation_manifest.json` also carry `route_graph_readiness_class`, `route_graph_full_hydration_observed`, `degraded_evaluation_observed`, `degraded_reason_codes_observed`, `precheck_gate_actions_observed`, `route_fallback_observed`, and `strict_full_search_proof_eligible`
- report outputs: methods_appendix.md, thesis_report.md
- corpus/baseline helpers: od_corpus.*, ors_snapshot.json
- hot-rerun outputs when applicable: hot_rerun_vs_cold_comparison.json, hot_rerun_vs_cold_comparison.csv, hot_rerun_gate.json, hot_rerun_report.md

The thesis evaluator and run-store registry keep `schema_version` on dict-backed JSON artifacts and version-tracked registry entries for JSONL/JSON outputs such as `voi_action_trace.json`, `voi_controller_trace_summary.json`, `voi_replay_oracle_summary.json`, `theorem_hook_map.json`, `lane_manifest.json`, and `final_route_trace.json`. Runtime `reuse_rate`, `world_reuse_rate`, and related cache/provenance fields are instrumented reporting surfaces unless a later empirical pass verifies them explicitly. Treat `route_graph_readiness_class="fast_startup_metadata_ready"` or `degraded_evaluation_observed=true` in evaluator summaries as degraded-evaluation evidence, not as full-search proof.

The browser-facing inspection workflow should read those files in two tiers: guaranteed route files such as `decision_package.json`, `preference_summary.json`, `support_summary.json`, `support_provenance.json`, `certified_set.json`, `certified_set_routes.jsonl`, and `final_route_trace.json`, then conditional follow-on files for abstention, witnesses, controller traces, theorem hooks, lane manifests, and evaluator outputs when they are actually present for the run.

## Documentation

Start with [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md).

High-value docs:

- [Run and Operations Guide](docs/run-and-operations.md)
- [Backend APIs and Tooling](docs/backend-api-tools.md)
- [API Cookbook](docs/api-cookbook.md)
- [Quality Gates and Benchmarks](docs/quality-gates-and-benchmarks.md)
- [Reproducibility Capsule](docs/reproducibility-capsule.md)
- [VOI Thesis Pipeline Spec](docs/voi-pipeline-spec.md)
- [Notebook Policy](notebooks/NOTEBOOKS_POLICY.md)

`docs/thesis-codebase-report.md` is maintained separately from the general docs set.
