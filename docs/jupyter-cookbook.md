# Jupyter Cookbook

This repository includes a starter notebook at `notebooks/freight_router_cookbook.ipynb` with step-by-step examples for:

1. Calling routing APIs.
2. Inspecting run artifacts.
3. Verifying signatures.
4. Running ETA concept-drift checks.

## Setup

From repo root:

```powershell
cd backend
uv sync --dev
uv run python -m pip install jupyter
```

## Start Jupyter

From repo root:

```powershell
jupyter notebook notebooks/freight_router_cookbook.ipynb
```

## Notebook Flow

The notebook demonstrates:

- health and vehicle endpoint checks
- route and pareto requests
- scenario comparison request
- manifest/signature retrieval and verification
- local ETA drift-check data generation and script execution

## Prerequisites

- Backend reachable at `http://localhost:8000` (or update base URL cell).
- Optional: run `.\scripts\dev.ps1` first to bring up backend + OSRM.

## Reproducibility Notes

- Notebook examples use fixed coordinates and explicit payload fields.
- For deterministic algorithm experiments, include `seed`, `departure_time_utc`, and `toggles` in requests where supported.
