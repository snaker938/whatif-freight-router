# Run and Operations Guide

Last Updated: 2026-04-09
Applies To: local backend/frontend runtime, rebuilds, evaluation runs, hot-rerun cache restore, docs checks, and low-resource test execution

This runbook is the operational reference for running, rebuilding, validating, and reporting on the project safely.

## Prerequisites

- PowerShell
- Docker Desktop
- Python with `uv`
- Node.js with `pnpm`

## Environment Setup

From repo root:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Then review the current strict runtime controls in `.env`:

- strict live sources and auth:
  - `LIVE_SCENARIO_COEFFICIENT_URL`
  - `LIVE_FUEL_PRICE_URL`
  - `LIVE_CARBON_SCHEDULE_URL`
  - departure, stochastic, toll, and terrain URLs
- scenario resiliency:
  - `LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES`
  - `LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT`
  - `LIVE_SCENARIO_MIN_SOURCE_COUNT_STRICT`
  - `LIVE_SCENARIO_MIN_COVERAGE_OVERALL_STRICT`
- backend attempt ceiling:
  - `ROUTE_COMPUTE_ATTEMPT_TIMEOUT_S`
- route seam mode controls:
  - `ROUTE_PIPELINE_DEFAULT_MODE`
- bounded OD probe controls:
  - `ROUTE_CONTEXT_PROBE_TIMEOUT_MS`
  - `ROUTE_CONTEXT_PROBE_MAX_PATHS`
- graph warmup and connectivity controls:
  - `ROUTE_GRAPH_WARMUP_ON_STARTUP`
  - `ROUTE_GRAPH_WARMUP_FAILFAST`
  - `ROUTE_GRAPH_WARMUP_TIMEOUT_S`
  - `ROUTE_GRAPH_STATUS_CHECK_TIMEOUT_MS`
  - `ROUTE_GRAPH_FAST_STARTUP_ENABLED`
  - `ROUTE_GRAPH_MIN_GIANT_COMPONENT_NODES`
  - `ROUTE_GRAPH_MIN_GIANT_COMPONENT_RATIO`
  - `ROUTE_GRAPH_MAX_NEAREST_NODE_DISTANCE_M`
- strict terrain controls:
  - `TERRAIN_DEM_FAIL_CLOSED_UK`
  - `TERRAIN_DEM_COVERAGE_MIN_UK`
- frontend compute fallback controls:
  - `COMPUTE_ATTEMPT_TIMEOUT_MS`
  - `COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS`
  - `NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS`
  - `NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS`
  - `NEXT_PUBLIC_COMPUTE_DEGRADE_STEPS`
- development live-call tracing and raw-body helpers:
  - `DEV_ROUTE_DEBUG_CONSOLE_ENABLED`
  - `DEV_ROUTE_DEBUG_INCLUDE_SENSITIVE`
  - `DEV_ROUTE_DEBUG_MAX_CALLS_PER_REQUEST`
  - `DEV_ROUTE_DEBUG_TRACE_TTL_SECONDS`
  - `DEV_ROUTE_DEBUG_MAX_REQUEST_TRACES`
  - `DEV_ROUTE_DEBUG_RETURN_RAW_PAYLOADS`
  - `DEV_ROUTE_DEBUG_MAX_RAW_BODY_CHARS`
- request-time live refresh controls:
  - `LIVE_ROUTE_COMPUTE_REFRESH_MODE`
  - `LIVE_ROUTE_COMPUTE_REQUIRE_ALL_EXPECTED`
  - `LIVE_ROUTE_COMPUTE_FORCE_NO_CACHE_HEADERS`
  - `LIVE_ROUTE_COMPUTE_FORCE_UNCACHED`
  - `LIVE_ROUTE_COMPUTE_PREFETCH_TIMEOUT_MS`
  - `LIVE_ROUTE_COMPUTE_PREFETCH_MAX_CONCURRENCY`
  - `LIVE_ROUTE_COMPUTE_PROBE_TERRAIN`

Strict runtime defaults are enforced in `backend/app/settings.py`. In practice that means:

- `STRICT_LIVE_DATA_REQUIRED=true`
- `LIVE_ROUTE_COMPUTE_REQUIRE_ALL_EXPECTED=true`
- route graph strict readiness is required
- route graph fast startup is disabled in strict mode
- terrain probing is enabled when strict route-compute requires a full expected-source set
- the public `POST /route` seam defaults to `tri_source` unless `ROUTE_PIPELINE_DEFAULT_MODE` is overridden
- the current single-leg `tri_source` seam keeps the public route contract at `pipeline_mode="tri_source"` while routing internal execution through `voi`
- waypoint-bearing requests use an explicit `legacy` compatibility path with a surfaced warning and `pipeline_mode="legacy"` for that request rather than creating a fourth public route outcome

## Operations Endpoints

Operational endpoints worth knowing:

- backend readiness: `GET /health/ready`
- basic health: `GET /health`
- metrics snapshot: `GET /metrics`
- cache stats: `GET /cache/stats`
- cache clear: `DELETE /cache`
- hot-rerun route-cache restore: `POST /cache/hot-rerun/restore`
- dev-only live call trace: `GET /debug/live-calls/{request_id}`

The frontend devtools proxy some of these routes, but not all of them. The hot-rerun restore route is a backend/operator tool and is not exposed by the current frontend Ops Diagnostics panel.

## Route Compute Fallback Policy

Frontend route compute uses bounded degradation:

1. stream Pareto via `/api/pareto/stream`
2. JSON Pareto via `/api/pareto`
3. single-route fallback via `/api/route`

Default degrade policy: `12 -> 6 -> 3`.

Each fallback aborts the previous attempt before moving on. Strict business failures from route-producing endpoints are terminal and intentionally stop additional fallback attempts.

The frontend fallback step now lands on the public `POST /route` seam with `pipeline_mode="tri_source"` when no explicit override is supplied. For ordinary single-leg OD requests, that public route label stays `tri_source` while the backend currently executes the request through internal `voi` logic. If a request includes waypoints, the backend records an explicit compatibility fallback reason, emits a warning, and returns `pipeline_mode="legacy"` for that request. That legacy path is a compatibility branch, not a fourth public terminal outcome.

Strict live refresh is hybrid:

- scenario coefficients are refreshed on each route attempt
- expensive live context feeds stay short-cached by TTL instead of being force-cleared per request

Route graph warmup lifecycle in strict mode:

- backend starts graph warmup on startup
- if warmup is still running, strict route endpoints fail fast with `routing_graph_warming_up`
- if warmup exceeds timeout or errors, strict route endpoints fail fast with `routing_graph_warmup_failed`
- if the graph fails connectivity quality gates, strict failures use `routing_graph_fragmented`
- OD-specific failures may surface as `routing_graph_coverage_gap`, `routing_graph_disconnected_od`, or `routing_graph_no_path`
- if graph is ready but strict live freshness is not, `/health/ready` may recommend `refresh_live_sources`
- evaluation artifacts now carry a second honesty layer for low-RAM and fast-startup lanes: `route_graph_readiness_class`, `route_graph_full_hydration_observed`, `degraded_evaluation_observed`, `degraded_reason_codes_observed`, `precheck_gate_actions_observed`, `route_fallback_observed`, and `strict_full_search_proof_eligible`
- staged subset assets created by the thesis campaign path now emit both `*.json.pkl` binary caches and `*.compact.pkl` compact bundles; for `.subset.` assets the runtime prefers the binary-cache warmup path before falling back to the compact bundle
- when you inspect low-RAM campaign logs, expect staged subset warmup phases to report `cache_probe`, `cache_signature_check`, `loading_binary_cache`, and `validating_binary_cache_payload` before the backend falls back to `loading_compact_bundle`

Frontend compute remains readiness-gated until `strict_route_ready=true`.

That readiness bit is not enough to interpret thesis evidence by itself. If a run records `backend_ready_summary.route_graph.status="ok_fast"` or `route_graph_readiness_class="fast_startup_metadata_ready"`, the backend was only ready under metadata-only graph warmup. Treat those runs as degraded-evaluation evidence when `degraded_evaluation_observed=true`, not as full-hydration strict-search proof. Full-search proof claims should require `route_graph_full_hydration_observed=true` and `strict_full_search_proof_eligible=true`.

## Start Full Local Stack

From repo root:

```powershell
.\scripts\dev.ps1
```

`dev.ps1` runs strict preflight before backend/frontend launch. If a required live payload is stale or invalid, startup stops and writes:

- `backend/out/model_assets/preflight_live_runtime.json`

Expected services:

- OSRM: `http://localhost:5000`
- Backend API: `http://localhost:8000`
- Backend OpenAPI UI: `http://localhost:8000/docs`
- Backend readiness: `http://localhost:8000/health/ready`
- Frontend: `http://localhost:3000`

## Run Backend Only

From repo root:

```powershell
Set-Location backend
uv sync --dev
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Preflight check before compute:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health/ready" -Method Get
```

Do not run the backend from repo root with `uv run --project backend uvicorn ...` in this repo layout. Use `Set-Location backend` first so `app.main` module resolution stays stable.

When evaluator or thesis runs need a hard Python RAM cap on the backend itself, use the logged launcher with the optional memory-limit wrapper:

```powershell
.\backend\scripts\start_backend_logged.ps1 -Port 8000 -MemoryLimitMB 24576
```

That starts the backend through [run_with_job_memory_limit.ps1](../backend/scripts/run_with_job_memory_limit.ps1), keeps stdout/stderr in `backend/out/backend_stdout.log` and `backend/out/backend_stderr.log`, and records the launcher PID in `backend/out/backend_server.pid`. Stop it with:

```powershell
.\backend\scripts\stop_backend_logged.ps1 -Port 8000
```

## Run Frontend Only

From repo root:

```powershell
Set-Location frontend
pnpm install
pnpm dev
```

## Full Refresh / Rebuild

Use this when OSRM, backend assets, or caches should be rebuilt end-to-end.

1. Stop active terminals.
2. Clean compose/runtime caches:

```powershell
.\scripts\clean.ps1
```

3. Recreate stack:

```powershell
.\scripts\dev.ps1
```

4. Rebuild backend assets:

```powershell
Set-Location backend
uv run python scripts/build_model_assets.py
uv run python scripts/publish_live_artifacts_uk.py
```

5. Optional subsystem rebuilds:

```powershell
uv run python scripts/build_routing_graph_uk.py
uv run python scripts/build_terrain_tiles_uk.py --source-dem-glob "assets/uk/dem/*.tif"
```

## Evaluation and Reporting Commands

From repo root:

```powershell
uv run --project backend python backend/scripts/run_headless_scenario.py --input-json docs/examples/sample_batch_request.json
uv run --project backend python backend/scripts/run_thesis_lane.py --help
uv run --project backend python backend/scripts/run_thesis_evaluation.py
uv run --project backend python backend/scripts/run_thesis_campaign.py --help
uv run --project backend python backend/scripts/run_thesis_hard_story_suite.py --help
uv run --project backend python backend/scripts/expand_thesis_broad_corpus.py --help
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
uv run --project backend python backend/scripts/compare_thesis_runs.py --help
uv run --project backend python backend/scripts/compose_thesis_sharded_report.py --help
uv run --project backend python backend/scripts/compose_thesis_suite_report.py
uv run --project backend python backend/scripts/check_eta_concept_drift.py --help
```

For low-RAM sequential thesis campaigns, the truthful strict-readiness path is a tranche-local
subset graph, not deferred startup on the full UK graph. Build the subset from the tranche corpus,
then point the evaluator or campaign runner at that asset before the in-process backend starts.

Example from repo root:

```powershell
.\backend\scripts\run_with_job_memory_limit.ps1 `
  -MemoryLimitMB 1024 `
  -FilePath .\.venv\Scripts\python.exe `
  -ArgumentList @(
    'backend/scripts/build_route_graph_subset.py',
    '--graph-json', 'backend/out/model_assets/routing_graph_uk.json',
    '--corpus-csv', 'backend/data/eval/uk_od_corpus_london_newcastle_family_4.csv',
    '--output-json', 'backend/out/route_graph_subsets/london_newcastle4.json',
    '--corridor-km', '35'
  )

uv run --project backend python backend/scripts/run_thesis_campaign.py `
  --corpus-csv backend/data/eval/uk_od_corpus_london_newcastle_family_4.csv `
  --max-tranches 1 `
  --route-graph-asset-path backend/out/route_graph_subsets/london_newcastle4.json `
  --route-graph-subset-corridor-km 12.5 `
  --require-balanced-win `
  --require-dominance-win `
  --require-time-preserving-win `
  --require-proof-grade-readiness `
  --evaluation-args --in-process-backend
```

`backend/scripts/build_route_graph_subset.py` now stages kept nodes to disk while it streams the
source graph, and it filters by the union of per-OD corridor-distance checks rather than a single
corpus-wide bounding box. That keeps mixed long-corridor tranches from accidentally retaining an
England-sized slab of graph nodes just because their endpoints span a wide rectangle.

`backend/scripts/run_thesis_campaign.py` can now thread that same bound directly through staged
subset planning with `--route-graph-subset-corridor-km <km>`. The in-process staged-subset path is
only truthful for a single non-resume tranche, because the app must preload the correct subset
before `app.main` imports and later tranches would otherwise keep the wrong graph resident. For
multi-tranche widening, stage the subset first or use a separately managed backend.

If a proof lane includes toll-sensitive rows whose tariff tables are incomplete, the thesis evaluator can be run with
`--toll-cost-per-km <rate>` so the route remains priced under an explicit configured toll model instead of failing the
row on unresolved tariff matches.

Use the evaluation ladder in this order:

1. targeted tests
2. lane-local checks with `backend/scripts/run_thesis_lane.py`
3. harder bounded suite checks with `backend/scripts/run_thesis_hard_story_suite.py`
4. broad-corpus expansion or shard preparation with `backend/scripts/expand_thesis_broad_corpus.py` when the corpus itself is changing
5. full thesis suite with `backend/scripts/run_thesis_evaluation.py`
6. sequential widening with `backend/scripts/run_thesis_campaign.py` when regression carry-forward matters
7. dedicated hot-rerun proof with `backend/scripts/run_hot_rerun_benchmark.py`
8. suite or sharded composition with `backend/scripts/compose_thesis_suite_report.py` / `backend/scripts/compose_thesis_sharded_report.py`
9. explicit before/after diffs with `backend/scripts/compare_thesis_runs.py`

For honest widening, `backend/scripts/run_thesis_campaign.py` now supports a bounded gate overlay on top of the older weighted/balanced checks:

- `--require-dominance-win` requires per-OD dominance wins against OSRM and ORS for every target variant
- `--require-time-preserving-win` requires per-OD time-preserving wins against OSRM and ORS for every target variant
- `--require-proof-grade-readiness` blocks widening unless the tranche metadata records full-hydration proof-grade readiness instead of degraded-evaluation evidence

Replay-ledger semantics stay strict:

- every widening tranche reruns the replay set of previously green ODs
- a tranche is not green if any replay OD regresses, even when newly added ODs pass
- a tranche is not green if an expected OD disappears from evaluator output
- a tranche is not green if the artifact bundle is present but proof-grade readiness is still red while `--require-proof-grade-readiness` is enabled

The thesis runner now emits cohort and metric-family scaffolding into the run payloads it owns, including
`thesis_cohort_scaffolding_v2`, `thesis_metric_family_scaffolding_v1`, `metadata.json`, `results.json`,
`thesis_metrics.json`, `thesis_plots.json`, and `evaluation_manifest.json`. The cohort scaffold includes the named
labels `collapse_prone`, `osrm_brittle`, `ors_brittle`, `refresh_sensitive`, `time_preserving_conflict`,
`low_ambiguity_fast_path`, `preference_sensitive`, `support_fragile`, `audit_heavy`, and `proxy_friendly`.
`evaluation_suite` now distinguishes the explicit runner roles for `broad_cold_proof`, `focused_refc_proof`,
`focused_voi_proof`, `preference_proof`, `optional_stopping_coverage`, `proxy_audit_calibration`,
`perturbation_flip_radius`, `public_transfer`, `synthetic_ground_truth`, `dccs_diagnostic_probe`,
`hot_rerun_cold_source`, and `hot_rerun`.
Within `thesis_metric_family_scaffolding_v1`, `preference` is `metadata_wired` at `cohort_metadata_only`, and
`multi_fidelity_support` is `partially_wired` at `shared_summary_metrics`.
Those surfaces are instrumented for later threshold and coverage tuning; this pass does not claim empirical gate
clearance, and the reuse/cache fields (`reuse_rate`, `world_reuse_rate`, `cache_reuse`) remain honesty surfaces
until verified in a dedicated empirical pass.

Operational note: the latest repaired broad-cold empirical result is
`backend/out/thesis_eval_core120_green_repaired/artifacts/thesis_eval_core120_green_repaired_20260404/`.
That repaired artifact family covers `120` ODs / `480` evaluated rows with `0` failures, but it is a repaired
composition rather than a fresh single-run thesis-runner bundle. Newer result families now also exist under
`backend/out/thesis_campaigns/`, including the green `hard_mixed24_corr12p5_t4_inproc_r4`, the blocked
`dominance_cluster5_cardiff_bath_corr12p5_r2`, the red `longcorr_cardiff4_corr12p5_r1`, and the regression-start
`publishable_seq_fast_bootstrap_v2` campaign ledgers. The latest 1200-row expanded corpus and its 10 shards are
current corpus assets, not completed thesis-result bundles. The low-RAM verification packet used in this pass for
evaluation-related tests ran through `backend/scripts/run_with_job_memory_limit.ps1` with `-MemoryLimitMB 2048`.

Operational note: the April 5 single-OD tuning packet under `backend/out/single_od_london_newcastle_*`
is now the main operational evidence for the split-process memory-cap path. The unstable multi-variant
backend-cap sweeps (`r16` / `r17` / `r18` with backend limits `2368`, `2400`, and `2432` MB) all recorded
backend `limit_exceeded` watchdog exits, while the later isolated `A/B/C` proof-grade runs (`r30` / `r31` / `r32`)
stabilized on the logged split-process topology with backend `2304` MB, evaluator `1024` MB, and
`strict_full_search_proof_eligible=true`.

For a longer but still bounded harder-story suite, `backend/scripts/run_thesis_hard_story_suite.py` runs the checked-in
`backend/data/eval/uk_od_corpus_hard_mixed_24.csv` and `backend/data/eval/uk_od_corpus_longcorr_hard_32.csv` lanes
serially, then composes their completed lane outputs into a single harder-suite bundle. That path is intended to
produce quoteable harder-lane artifacts without pretending it replaces the sequential widening gate.

## Runtime Output Locations

Under `backend/out/`:

- `model_assets/`
- `model_assets/staged_subsets/`
- `route_graph_subsets/`
- `artifacts/<run_id>/`
- `manifests/<run_id>.json`
- `scenario_manifests/<run_id>.json`
- `provenance/<run_id>.jsonl`
- `thesis_campaigns/<campaign>/`
- `thesis_corpus/`
- `test_runs/<timestamp>/`

Common run artifact families:

- route outputs: results.json, results.csv, metadata.json, routes.geojson
- DCCS frontier outputs: `dccs_summary.json`, `dccs_candidates.jsonl`, and `strict_frontier.jsonl` are the DCCS export trio; `dccs_summary.json` carries `control_state`, and the candidate rows in `dccs_candidates.jsonl` / `strict_frontier.jsonl` carry `safe_elimination_reason`, `dominance_margin`, `dominating_candidate_ids`, `dominated_candidate_ids`, `search_deficiency_score`, `hidden_challenger_score`, `anti_collapse_quota`, and `long_corridor_search_completeness`
- certification and fragility outputs: `winner_summary.json`, `certificate_summary.json`, `route_fragility_map.json`, `competitor_fragility_breakdown.json`, `sampled_world_manifest.json`, `evidence_snapshot_manifest.json`, and `value_of_refresh.json`
- public DecisionPackage-derived route bundle outputs wired on the landed seam: decision_package.json, preference_summary.json,
  support_summary.json, support_provenance.json, support_trace.jsonl, certified_set.json, certified_set_routes.jsonl,
  abstention_summary.json, witness_summary.json, witness_routes.jsonl
- that public route bundle mirrors `DecisionPackage` on `POST /route`; its `terminal_kind` is exactly
  `certified_singleton`, `certified_set`, or `typed_abstention`
- controller, trace, and replay outputs when populated: `controller_summary.json`, `controller_trace.jsonl`,
  `voi_action_trace.json`, `voi_controller_state.jsonl`, `voi_action_scores.csv`, `voi_stop_certificate.json`,
  `voi_controller_trace_summary.json`, `voi_replay_oracle_summary.json`, `theorem_hook_map.json`,
  `lane_manifest.json`, and `final_route_trace.json`
- `final_route_trace.json` is the route-trace anchor for the runtime bundle; when present it carries
  `artifact_pointers` to sibling DecisionPackage-derived route-bundle and controller artifacts so Run Inspector or
  direct artifact fetches can discover the emitted family from a completed run
- thesis outputs: thesis_results.*, thesis_summary.*, thesis_summary_by_cohort.*, thesis_metrics.json,
  thesis_plots.json, cohort_composition.json, evaluation_manifest.json, thesis_report.md, with
  `metric_family_scaffolding` exported through `results.json`, `metadata.json`, `thesis_metrics.json`,
  `thesis_plots.json`, and `evaluation_manifest.json`
- campaign outputs: `campaign_state.json`, `campaign_result.json`, `campaign_report.md`, tranche-local
  `tranche_<n>/tranche_od_corpus.csv`, `tranche_<n>/per_od_status.json`, and staged `route_graph_asset_plan`
  entries that point at subset assets and their `.meta.json` reports when the campaign uses low-RAM graph staging
- sharded broad-corpus helpers: `backend/out/thesis_corpus/uk_od_corpus_thesis_broad_expanded_1200.summary.json`,
  `backend/out/thesis_corpus/uk_od_corpus_thesis_broad_expanded_1200.shards.json`, and composed sharded-report
  artifacts such as `shard_sources.json` when `backend/scripts/compose_thesis_sharded_report.py` is used
- hot-rerun outputs: hot_rerun_vs_cold_comparison.json, hot_rerun_vs_cold_comparison.csv, hot_rerun_gate.json, hot_rerun_report.md

## Safe Test Execution

From repo root:

```powershell
.\scripts\run_backend_tests_safe.ps1 -MaxCores 1 -PriorityClass Idle -MaxWorkingSetMB 4096
```

Useful options:

- `-TestPattern test_tooling_scripts.py`
- `-PerFileTimeoutSec 900`
- `-NoOutputStallSec 240`
- `-IncludeCoverage`
- `-ExtraPytestArgs @("-k", "strict")`

Outputs:

- `backend/out/test_runs/<timestamp>/summary.csv`
- `backend/out/test_runs/<timestamp>/summary.json`
- `backend/out/test_runs/<timestamp>/failed_tests.txt`
- `backend/out/test_runs/<timestamp>/rerun_failed.ps1`

## Docs Serve and Validation

From repo root:

```powershell
.\scripts\serve_docs.ps1
python scripts/check_docs.py
```

Docs URL: `http://localhost:8088/`

## Stop Local Stack

1. Stop running terminals.
2. From repo root:

```powershell
docker compose down
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
