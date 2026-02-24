# Run and Operations Guide

Last Updated: 2026-02-23  
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
- fuel: `LIVE_FUEL_PRICE_URL`, optional auth fields
- carbon: `LIVE_CARBON_SCHEDULE_URL`
- departure/stochastic/toll URLs
- terrain: `LIVE_TERRAIN_DEM_URL_TEMPLATE`
- strict host allow-lists: `LIVE_*_ALLOWED_HOSTS`
- retry/query knobs: `LIVE_HTTP_RETRY_*`, `LIVE_SCENARIO_DFT_MAX_PAGES`

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
- Frontend: `http://localhost:3000`

## Run Backend Only

From repo root:

```powershell
Set-Location backend
uv sync --dev
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Alternative without changing directory:

```powershell
uv run --project backend uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

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
