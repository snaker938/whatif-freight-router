# Run and Operations Guide

Last Updated: 2026-03-31  
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

Frontend compute remains readiness-gated until `strict_route_ready=true`.

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
uv run --project backend python backend/scripts/run_hot_rerun_benchmark.py
uv run --project backend python backend/scripts/compose_thesis_suite_report.py
uv run --project backend python backend/scripts/check_eta_concept_drift.py --help
```

Use the evaluation ladder in this order:

1. targeted tests
2. lane-local checks with `backend/scripts/run_thesis_lane.py`
3. full thesis suite with `backend/scripts/run_thesis_evaluation.py`
4. dedicated hot-rerun proof with `backend/scripts/run_hot_rerun_benchmark.py`
5. suite composition with `backend/scripts/compose_thesis_suite_report.py`

## Runtime Output Locations

Under `backend/out/`:

- `model_assets/`
- `artifacts/<run_id>/`
- `manifests/<run_id>.json`
- `scenario_manifests/<run_id>.json`
- `provenance/<run_id>.jsonl`
- `test_runs/<timestamp>/`

Common run artifact families:

- route outputs: results.json, results.csv, metadata.json, routes.geojson
- DCCS/REFC/VOI outputs: dccs_summary.json, certificate_summary.json, value_of_refresh.json, voi_action_trace.json, voi_stop_certificate.json
- thesis outputs: thesis_results.*, thesis_summary.*, thesis_metrics.json, thesis_plots.json, evaluation_manifest.json, thesis_report.md
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
