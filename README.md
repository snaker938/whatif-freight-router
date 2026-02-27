# Carbon-Aware Multi-Objective Freight Router

This repo runs:
- OSRM in Docker (routing engine)
- FastAPI backend
- Next.js frontend

## Prerequisites

- Docker Desktop running
- PowerShell
- `uv` installed (for backend local dev)
- Node.js and `pnpm` installed (for frontend local dev)

## Recommended workflow (one command, local dev)

From repo root:

```powershell
.\scripts\dev.ps1
```

What `dev.ps1` does:
- creates `.env` from `.env.example` if missing
- starts OSRM with Docker (`docker compose up -d osrm`)
- waits for OSRM to respond on `http://localhost:5000/`
- runs strict live preflight (`backend/scripts/preflight_live_runtime.py`) and fails fast if required live feeds are invalid/stale
- starts backend in a new PowerShell window:
  - `uv sync --dev` (only if `.venv` missing)
  - `uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- starts frontend in a new PowerShell window:
  - `pnpm install` (only if `node_modules` missing)
  - `pnpm dev`
- if port `8000` or `3000` is already in use, that service is skipped

URLs:
- Frontend: `http://localhost:3000`
- Backend docs: `http://localhost:8000/docs`
- Backend readiness: `http://localhost:8000/health/ready`
- OSRM: `http://localhost:5000`

### Route compute timeout/fallback knobs

The frontend compute flow now runs bounded fallback attempts with degraded alternatives (`12 -> 6 -> 3` by default).
Configure in `.env` / `.env.example`:

- `COMPUTE_ATTEMPT_TIMEOUT_MS` (server route-handler timeout override; default `420000`)
- `COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS` (server route-handler fallback timeout override; default `180000`)
- `NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS` (browser fallback; default `420000`)
- `NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS` (browser fallback; default `180000`)
- `NEXT_PUBLIC_COMPUTE_DEGRADE_STEPS` (default `12,6,3`)
- backend attempt ceiling: `ROUTE_COMPUTE_ATTEMPT_TIMEOUT_S` (default `420`)
- bounded OD context probe: `ROUTE_CONTEXT_PROBE_TIMEOUT_MS` (default `2500`)
- bounded OD context probe path budget: `ROUTE_CONTEXT_PROBE_MAX_PATHS` (default `2`)
- strict scenario-source resiliency:
  - `LIVE_SCENARIO_COEFFICIENT_URL` (default tracks `main` scenario artifact)
  - `LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES` (default `4320`)
  - `LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT` (default `true`)
  - `LIVE_SCENARIO_MIN_SOURCE_COUNT_STRICT` (default `3`)
  - `LIVE_SCENARIO_MIN_COVERAGE_OVERALL_STRICT` (default `0.75`)
- graph warmup on backend startup: `ROUTE_GRAPH_WARMUP_ON_STARTUP` (default `1`)
- strict warmup fail-fast gate: `ROUTE_GRAPH_WARMUP_FAILFAST` (default `1`)
- graph warmup timeout: `ROUTE_GRAPH_WARMUP_TIMEOUT_S` (default `1200`)
- graph status check timeout: `ROUTE_GRAPH_STATUS_CHECK_TIMEOUT_MS` (default `1000`)
- strict giant-component floor (nodes): `ROUTE_GRAPH_MIN_GIANT_COMPONENT_NODES` (default `50000`)
- strict giant-component floor (ratio): `ROUTE_GRAPH_MIN_GIANT_COMPONENT_RATIO` (default `0.20`)
- strict OD nearest-node max distance (m): `ROUTE_GRAPH_MAX_NEAREST_NODE_DISTANCE_M` (default `10000`)
- strict terrain fail-closed gate: `TERRAIN_DEM_FAIL_CLOSED_UK` (default `true`)
- strict terrain minimum DEM coverage: `TERRAIN_DEM_COVERAGE_MIN_UK` (default `0.96`)
- dev live-call tracing knobs (development only, sensitive when enabled):
  - `DEV_ROUTE_DEBUG_CONSOLE_ENABLED`
  - `DEV_ROUTE_DEBUG_INCLUDE_SENSITIVE`
  - `DEV_ROUTE_DEBUG_MAX_CALLS_PER_REQUEST`
  - `DEV_ROUTE_DEBUG_TRACE_TTL_SECONDS`
  - `DEV_ROUTE_DEBUG_MAX_REQUEST_TRACES`

Strict-live route compute now uses a hybrid cache refresh strategy: scenario coefficients refresh each attempt, while expensive live context feeds (DfT/WebTRIS/Traffic/Meteo) remain short-cached by TTL for fallback reliability.
Frontend fallback now treats strict business failures (`HTTP 4xx` from route/pareto endpoints) as terminal and stops additional fallback attempts immediately.
Routing graph runtime now always loads the full `routing_graph_uk.json` dataset in strict mode (no streamed/capped graph loading path exposed via env config).
When graph warmup fail-fast is enabled, strict route endpoints return `routing_graph_warming_up` quickly until `GET /health/ready` reports `strict_route_ready=true`. If warmup exceeds timeout, endpoints return `routing_graph_warmup_failed` with rebuild guidance. If the loaded graph is fragmented, strict failures use `routing_graph_fragmented`; OD-specific failures use `routing_graph_disconnected_od` or `routing_graph_coverage_gap`.
`GET /health/ready` now also reports `strict_live` readiness for scenario coefficients; if stale/unavailable under strict policy, `recommended_action=refresh_live_sources`.
The frontend compute button is readiness-gated and remains disabled until strict route readiness is true.
Before triggering compute, confirm `GET /health/ready` is reachable and reports `strict_route_ready=true` and `strict_live.ok=true`.
Route Compute Diagnostics overlay now includes per-attempt live API call tracing (expected sources, observed URL calls, request/success/cache/retry status, headers, and extra diagnostics). Backend dev trace endpoint: `GET /debug/live-calls/{request_id}`.
Repeated same-tile terrain route-cache hits are deduplicated in trace rows per request; use trace summary/terrain diagnostics for total cache-hit volume.


## Stopping the dev workflow

1. In backend window: `Ctrl+C`
2. In frontend window: `Ctrl+C`
3. From repo root:

```powershell
docker compose down
```

## Changing region data (`REGION_PBF_URL`)

Set this in `.env`, for example:

```env
REGION_PBF_URL=https://download.geofabrik.de/europe/united-kingdom-latest.osm.pbf
```

Then run `.\scripts\dev.ps1` again.

You do not need `clean.ps1` for normal region changes. The scripts now handle this automatically:
- changed URL -> re-download `.pbf`
- changed URL -> rebuild OSRM artifacts
- unchanged URL -> reuse cache

## Full Docker stack

Run everything in containers:

```powershell
docker compose up --build
```

Use this for full containerized verification. Do not run this at the same time as `.\scripts\dev.ps1` (port conflicts on `3000`, `5000`, `8000`).

## Utility scripts

- `osrm/scripts/download_pbf.sh`
  - runs automatically via service `osrm_download`
  - downloads `region.osm.pbf` into `osrm/data/pbf`
  - tracks `REGION_PBF_URL` and refreshes cache when URL changes

- `osrm/scripts/run_osrm.sh`
  - runs automatically via service `osrm`
  - reuses existing OSRM graph data when valid
  - rebuilds OSRM graph data when required artifacts are missing or region URL changed

- `scripts/clean.ps1`
  - manual hard reset script
  - runs `docker compose down`
  - removes `osrm/data/pbf` and `osrm/data/osrm` caches
  - recreates those folders with `.gitkeep`
  - use when forcing a fully clean rebuild or recovering from suspected cache corruption

- `scripts/serve_docs.ps1`
  - serves markdown docs from `docs/` over a local HTTP server
  - run from repo root:
  - `.\scripts\serve_docs.ps1`
  - then open `http://localhost:8088/`

- `backend/scripts/benchmark_batch_pareto.py`
  - benchmark harness with runtime/resource logs
  - run from `backend/`:
  - `uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212`

- `backend/scripts/run_headless_scenario.py`
  - headless batch runner with artifact download
  - run from `backend/` with JSON input:
  - `uv run python scripts/run_headless_scenario.py --input-json ../docs/examples/sample_batch_request.json`
  - run from `backend/` with CSV input:
  - `uv run python scripts/run_headless_scenario.py --input-csv .\pairs.csv`

- `backend/scripts/run_robustness_analysis.py`
  - multi-seed robustness analysis runner
  - run from `backend/`:
  - `uv run python scripts/run_robustness_analysis.py --mode inprocess-fake --seeds 11,22,33 --pair-count 100`

- `backend/scripts/run_sensitivity_analysis.py`
  - one-factor sensitivity sweep runner for cost toggles
  - run from `backend/`:
  - `uv run python scripts/run_sensitivity_analysis.py --mode inprocess-fake --pair-count 50 --include-no-tolls`

- `backend/scripts/check_eta_concept_drift.py`
  - compares predicted vs observed ETA CSV values and flags concept drift
  - run from `backend/`:
  - `uv run python scripts/check_eta_concept_drift.py --input-csv .\eta_observations.csv --mae-threshold-s 120 --mape-threshold-pct 10`

- `backend/scripts/generate_run_report.py`
  - regenerate `report.pdf` from manifest/results/metadata for a run
  - run from `backend/`:
  - `uv run python scripts/generate_run_report.py --run-id <run_id> --out-dir out`

- `backend/scripts/publish_live_artifacts_uk.py`
  - publishes strict JSON live artifacts into tracked `backend/assets/uk/` paths
  - converts compiled toll outputs into strict runtime JSON payloads (`toll_topology_uk.json`, `toll_tariffs_uk.json`)
  - validates/regenerates fuel signature when needed
  - run from repo root:
  - `uv run --project backend python backend/scripts/publish_live_artifacts_uk.py`

- `backend/scripts/preflight_live_runtime.py`
  - validates strict live runtime loaders before app startup
  - writes summary to `backend/out/model_assets/preflight_live_runtime.json`
  - run from repo root:
  - `uv run --project backend python backend/scripts/preflight_live_runtime.py`

- `scripts/demo_repro_run.ps1`
  - scripted reproducibility capsule run (fixed seed and pair count)
  - run from repo root:
  - `.\scripts\demo_repro_run.ps1`

## Documentation

- `docs/DOCS_INDEX.md`
  - central docs index and source-of-truth navigation
- `docs/run-and-operations.md`
  - complete local runbook (frontend/backend/full stack/docs/tests)
- run docs checks from repo root:
  - `python scripts/check_docs.py`
- serve docs locally from repo root:
  - `.\scripts\serve_docs.ps1`
- `docs/backend-api-tools.md`
  - backend endpoint inventory, strict contract shape, and tooling commands
- `docs/api-cookbook.md`
  - reproducible CLI/API examples (notebook-free)
- `docs/strict-errors-reference.md`
  - reason-code catalog and stream/non-stream failure shape
- `docs/quality-gates-and-benchmarks.md`
  - quality and performance gate workflow
- `notebooks/NOTEBOOKS_POLICY.md`
  - notebook-free policy and alternatives

## API Quick Commands

Duty chain (multi-leg):

```powershell
$body = @{
  stops = @(
    @{ lat = 52.4862; lon = -1.8904; label = "Birmingham" }
    @{ lat = 52.2053; lon = 0.1218; label = "Cambridge" }
    @{ lat = 51.5072; lon = -0.1276; label = "London" }
  )
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://localhost:8000/duty/chain" -Method Post -ContentType "application/json" -Body $body
```

Oracle quality ingest + dashboard:

```powershell
$check = @{
  source = "oracle_demo"
  schema_valid = $true
  signature_valid = $true
  freshness_s = 45
  latency_ms = 120
  record_count = 10
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/oracle/quality/check" -Method Post -ContentType "application/json" -Body $check
Invoke-RestMethod -Uri "http://localhost:8000/oracle/quality/dashboard" -Method Get
Invoke-WebRequest -Uri "http://localhost:8000/oracle/quality/dashboard.csv" -OutFile ".\oracle_quality_dashboard.csv"
```

Weather + synthetic incidents:

```powershell
$body = @{
  origin = @{ lat = 52.4862; lon = -1.8904 }
  destination = @{ lat = 51.5072; lon = -0.1276 }
  vehicle_type = "rigid_hgv"
  scenario_mode = "no_sharing"
  weather = @{
    enabled = $true
    profile = "rain"
    intensity = 1.2
    apply_incident_uplift = $true
  }
  incident_simulation = @{
    enabled = $true
    seed = 123
    dwell_rate_per_100km = 1.0
    accident_rate_per_100km = 0.3
    closure_rate_per_100km = 0.05
  }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:8000/route" -Method Post -ContentType "application/json" -Body $body
```

## Accessibility and i18n Readiness

- Frontend now includes a language selector (`English`/`Español`) with locale-aware date/number formatting in key panels.
- Keyboard support includes a skip link (`Skip to controls panel`) and focus-visible states for primary controls.
- Screen-reader updates are announced through a live region for compute/compare/optimization/duty-chain status changes.

## Troubleshooting

- `dev.ps1` stuck on "Waiting for OSRM...":
  - check `docker compose logs -f osrm`
- OSRM exits unexpectedly:
  - run `.\scripts\clean.ps1`, then `.\scripts\dev.ps1`
