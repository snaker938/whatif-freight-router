# Carbon-Aware Multi-Objective Freight Router (Starter v0)

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
- OSRM: `http://localhost:5000`

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

- `scripts/demo_repro_run.ps1`
  - scripted reproducibility capsule run (fixed seed and pair count)
  - run from repo root:
  - `.\scripts\demo_repro_run.ps1`

## Documentation

- `docs/sample-manifest.md`
  - sample batch request/response payloads and manifest retrieval flow
- `docs/dissertation-math-overview.md`
  - concise report-ready overview of Pareto, scoring, and objective formulas
- `docs/math-appendix.md`
  - expanded notation, derivations, complexity, and assumptions
- `docs/appendix-graph-theory-notes.md`
  - graph-theory framing of current candidate-based approach and future MOSP direction
- `docs/co2e-validation.md`
  - reference examples validating the current CO2e formula assumptions
- `docs/performance-profiling-notes.md`
  - benchmark usage, profiling workflow, and optimization notes
- `docs/reproducibility-capsule.md`
  - deterministic demo run and capsule artifact workflow
- `docs/backend-api-tools.md`
  - cost toggles, vehicle CRUD, cache, signatures, provenance, import/export, and analysis tools
- `docs/eta-concept-drift.md`
  - ETA drift-check input schema, metrics, thresholds, and output artifacts
- `docs/jupyter-cookbook.md`
  - notebook setup and guided API/artifact/drift workflows

## Troubleshooting

- `dev.ps1` stuck on "Waiting for OSRM...":
  - check `docker compose logs -f osrm`
- OSRM exits unexpectedly:
  - run `.\scripts\clean.ps1`, then `.\scripts\dev.ps1`
