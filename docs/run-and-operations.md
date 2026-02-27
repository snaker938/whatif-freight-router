# Run and Operations Guide

Last Updated: 2026-02-25  
Applies To: local backend/frontend runtime, full rebuilds, docs checks, and low-resource test execution

This runbook is the operational reference for running and rebuilding the project safely.

## Prerequisites

- PowerShell (Windows)
- Docker Desktop (for OSRM and compose workflows)
- Python + `uv` (backend)
- Node.js + `pnpm` (frontend)

## Environment Setup

From repo root:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Then review strict live URL/auth settings in `.env`:

- scenario: `LIVE_SCENARIO_COEFFICIENT_URL`
- scenario coefficient freshness: `LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES` (default `4320`)
- fuel: `LIVE_FUEL_PRICE_URL`, optional auth fields
- carbon: `LIVE_CARBON_SCHEDULE_URL`
- departure/stochastic/toll URLs
- terrain: `LIVE_TERRAIN_DEM_URL_TEMPLATE`
- strict host allow-lists: `LIVE_*_ALLOWED_HOSTS`
- retry/query knobs: `LIVE_HTTP_RETRY_*`, `LIVE_SCENARIO_DFT_MAX_PAGES`
- strict scenario-source resiliency knobs:
  - `LIVE_SCENARIO_COEFFICIENT_URL` (default points to `main` scenario artifact)
  - `LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES` (default `4320`)
  - `LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT`
  - `LIVE_SCENARIO_MIN_SOURCE_COUNT_STRICT`
  - `LIVE_SCENARIO_MIN_COVERAGE_OVERALL_STRICT`
- backend stream attempt ceiling: `ROUTE_COMPUTE_ATTEMPT_TIMEOUT_S`
- bounded OD probe knobs:
  - `ROUTE_CONTEXT_PROBE_TIMEOUT_MS`
  - `ROUTE_CONTEXT_PROBE_MAX_PATHS`
- route graph warmup knobs:
  - `ROUTE_GRAPH_WARMUP_ON_STARTUP`
  - `ROUTE_GRAPH_WARMUP_FAILFAST`
  - `ROUTE_GRAPH_WARMUP_TIMEOUT_S` (default `1200`)
  - `ROUTE_GRAPH_STATUS_CHECK_TIMEOUT_MS`
  - `ROUTE_GRAPH_MIN_GIANT_COMPONENT_NODES` (default `50000`)
  - `ROUTE_GRAPH_MIN_GIANT_COMPONENT_RATIO` (default `0.20`)
  - `ROUTE_GRAPH_MAX_NEAREST_NODE_DISTANCE_M` (default `10000`)
- strict terrain knobs:
  - `TERRAIN_DEM_FAIL_CLOSED_UK` (default `true`)
  - `TERRAIN_DEM_COVERAGE_MIN_UK` (default `0.96`)
- frontend compute fallback knobs:
  - `COMPUTE_ATTEMPT_TIMEOUT_MS` (server route-handler timeout override; default `420000`)
  - `COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS` (server route-handler fallback timeout override; default `180000`)
  - `NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS`
  - `NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS`
  - `NEXT_PUBLIC_COMPUTE_DEGRADE_STEPS` (default `12,6,3`)
- dev live-call tracing knobs (development only; sensitive when enabled):
  - `DEV_ROUTE_DEBUG_CONSOLE_ENABLED`
  - `DEV_ROUTE_DEBUG_INCLUDE_SENSITIVE`
  - `DEV_ROUTE_DEBUG_MAX_CALLS_PER_REQUEST`
  - `DEV_ROUTE_DEBUG_TRACE_TTL_SECONDS`
  - `DEV_ROUTE_DEBUG_MAX_REQUEST_TRACES`

## Route Compute Fallback Policy

Frontend route compute now runs bounded attempts with explicit degradation:

1. Stream Pareto (`/api/pareto/stream`)
2. JSON Pareto fallback (`/api/pareto`)
3. Single-route fallback (`/api/route`)

Default degrade policy: `12 -> 6 -> 3`.

Each fallback aborts the previous attempt before moving on, to avoid orphan backend work.
Strict business failures (`HTTP 4xx` from route/pareto endpoints) are terminal and intentionally halt additional fallback attempts.

Strict-live refresh is hybrid:

- Scenario coefficients are refreshed on each route attempt.
- Expensive live context feeds (DfT/WebTRIS/Traffic/Meteo) are short-cached by TTL instead of being force-cleared per request.

Route graph warmup lifecycle (strict mode):

- Backend starts graph warmup in the background on startup.
- If warmup is still running, strict route endpoints fail fast with `reason_code=routing_graph_warming_up`.
- If warmup exceeds timeout or errors, strict route endpoints fail fast with `reason_code=routing_graph_warmup_failed`.
- If the loaded graph fails connectivity quality gates, strict failures use `reason_code=routing_graph_fragmented`.
- OD-specific strict graph failures surface as:
  - `reason_code=routing_graph_coverage_gap`
  - `reason_code=routing_graph_disconnected_od`
- Check readiness at `GET /health/ready` and retry when `strict_route_ready=true`.
- `/health/ready` now includes `strict_live` diagnostics for scenario coefficients:
  - `ok`, `status`, `reason_code`, `message`
  - `as_of_utc`, `age_minutes`, `max_age_minutes`, `checked_at_utc`
- If graph is ready but strict live freshness is not, `recommended_action=refresh_live_sources`.
- Frontend compute is readiness-gated and remains disabled until `strict_route_ready=true`.

Diagnostics panel (`Route compute diagnostics`) now shows:

- active attempt/endpoint/stage
- backend heartbeat age
- attempt-level timeout and alternative count
- grouped trace entries by attempt
- per-attempt live API call table (URL, requested/success, status/error, cache/retry, headers/extra)

Terrain trace note: repeated same-tile route-cache hits are deduplicated per request in live-call rows; rely on summary counts/terrain diagnostics for full cache-hit volume.

Backend dev endpoint for this trace:

- `GET /debug/live-calls/{request_id}` (dev-gated)

## Start Full Local Stack

From repo root:

```powershell
.\scripts\dev.ps1
```

`dev.ps1` now runs strict live preflight before backend/frontend launch. If a required live payload is stale/invalid, startup stops and writes:

- `backend/out/model_assets/preflight_live_runtime.json`

Expected services:

- OSRM: `http://localhost:5000`
- Backend API: `http://localhost:8000`
- Backend OpenAPI UI: `http://localhost:8000/docs`
- Backend readiness: `http://localhost:8000/health/ready`
- Frontend: `http://localhost:3000`

## Run Backend Only (Known-Good)

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

## Full Refresh / Rebuild (Long Path)

Use this when OSRM + backend assets + caches should be rebuilt end-to-end.

1. Stop active app terminals (`Ctrl+C`).
2. Clean compose/runtime caches:

```powershell
.\scripts\clean.ps1
```

3. Recreate stack:

```powershell
.\scripts\dev.ps1
```

4. Rebuild backend model assets:

```powershell
Set-Location backend
uv run python scripts/build_model_assets.py
uv run python scripts/publish_live_artifacts_uk.py
```

This publish step updates strict runtime tracked artifact targets:

- `backend/assets/uk/departure_profiles_uk.json`
- `backend/assets/uk/stochastic_regimes_uk.json`
- `backend/assets/uk/toll_topology_uk.json`
- `backend/assets/uk/toll_tariffs_uk.json`

5. Optional subsystem rebuilds:

```powershell
uv run python scripts/build_routing_graph_uk.py
uv run python scripts/build_terrain_tiles_uk.py --source-dem-glob "assets/uk/dem/*.tif"
```

## Runtime Output Locations

Under `backend/out/`:

- `model_assets/`
- `artifacts/{run_id}/`
- `manifests/{run_id}.json`
- `scenario_manifests/{run_id}.json`
- `provenance/{run_id}.jsonl`
- `test_runs/{timestamp}/` (safe test runner outputs)

## Safe Test Execution (Low Resource)

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

1. Stop running terminals (`Ctrl+C`).
2. From repo root:

```powershell
docker compose down
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
