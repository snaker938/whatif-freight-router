# Backend APIs and Tooling

Last Updated: 2026-02-19  
Applies To: `backend/app/main.py` strict runtime

This page is the backend contract and tooling reference for the current codebase.

## Core Runtime Notes

- Runtime uses strict, reason-coded failure handling for model-data constraints.
- Streaming and non-streaming endpoints use aligned reason-code semantics.
- Hybrid data mode is supported (fresh signed assets and optional live refresh where configured).

## Endpoint Inventory

### System and Admin

- `GET /`
- `GET /health`
- `GET /metrics`
- `GET /cache/stats`
- `DELETE /cache`

### Vehicles

- `GET /vehicles`
- `GET /vehicles/custom`
- `POST /vehicles/custom`
- `PUT /vehicles/custom/{vehicle_id}`
- `DELETE /vehicles/custom/{vehicle_id}`

### Routing and Pareto

- `POST /route`
- `POST /pareto`
- `POST /pareto/stream`
- `POST /api/pareto/stream`
- `POST /departure/optimize`
- `POST /duty/chain`
- `POST /scenario/compare`

### Experiments

- `GET /experiments`
- `POST /experiments`
- `GET /experiments/{experiment_id}`
- `PUT /experiments/{experiment_id}`
- `DELETE /experiments/{experiment_id}`
- `POST /experiments/{experiment_id}/compare`

### Batch

- `POST /batch/pareto`
- `POST /batch/import/csv`

### Run Artifacts and Signatures

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/provenance`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `POST /verify/signature`
- `GET /runs/{run_id}/artifacts`
- `GET /runs/{run_id}/artifacts/results.json`
- `GET /runs/{run_id}/artifacts/results.csv`
- `GET /runs/{run_id}/artifacts/metadata.json`
- `GET /runs/{run_id}/artifacts/routes.geojson`
- `GET /runs/{run_id}/artifacts/results_summary.csv`
- `GET /runs/{run_id}/artifacts/report.pdf`

### Oracle Quality

- `POST /oracle/quality/check`
- `GET /oracle/quality/dashboard`
- `GET /oracle/quality/dashboard.csv`

## Strict Error Contract

### Non-stream endpoints

```json
{
  "detail": {
    "reason_code": "terrain_dem_asset_unavailable",
    "message": "Terrain DEM assets are unavailable.",
    "warnings": ["Build model assets before routing."],
    "terrain_dem_version": "uk_dem_v1",
    "terrain_coverage_required": 0.98,
    "terrain_coverage_min_observed": 0.91
  }
}
```

### Stream fatal event (`POST /pareto/stream`)

```json
{
  "type": "fatal",
  "reason_code": "epsilon_infeasible",
  "message": "No candidates satisfied epsilon constraints.",
  "warnings": []
}
```

## Common Request Features

These are available across route-producing endpoints where applicable:

- `weights`
- `risk_aversion`
- `max_alternatives`
- `pareto_method`
- `epsilon`
- `departure_time_utc`
- `cost_toggles`
- `weather`
- `incident_simulation`
- `emissions_context`

## Model and Artifact Commands

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/score_model_quality.py
uv run python scripts/benchmark_model_v2.py
```

## Related Docs

- [Documentation Index](README.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [API Cookbook](api-cookbook.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
