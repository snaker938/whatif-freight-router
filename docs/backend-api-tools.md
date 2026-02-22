# Backend APIs and Tooling

Last Updated: 2026-02-21  
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

Vehicle profile payloads are v2 in strict runtime and include additive fields:

- `schema_version`
- `vehicle_class`
- `toll_vehicle_class`
- `toll_axle_class`
- `fuel_surface_class`
- `risk_bucket`
- `stochastic_bucket`
- `terrain_params`
- `aliases`
- `profile_source`
- `profile_as_of_utc`

Unknown `vehicle_type` now fails strict route-producing requests with `422 vehicle_profile_unavailable`.

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

Scenario model-data failures follow the same canonical shape with:

- `scenario_profile_unavailable`
- `scenario_profile_invalid`

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

## Scenario Policy Runtime

Scenario mode is strict asset-backed policy (`no_sharing`, `partial_sharing`, `full_sharing`) and applies multipliers to:

- duration
- incident rate and delay
- fuel/energy consumption
- emissions
- stochastic sigma

Scenario policy is context-conditioned by:

- `corridor_geohash5`
- `hour_slot_local`
- `road_mix_vector` / `road_mix_bucket`
- `vehicle_class`
- `day_kind`
- `weather_regime`

Strict runtime requires `LIVE_SCENARIO_COEFFICIENT_URL` and resolves live scenario context from free UK APIs (WebTRIS, Traffic England, DfT raw counts, Open-Meteo). Missing/stale/incomplete live context fails with canonical scenario reason codes. Signed local fallback is blocked by default unless `LIVE_SCENARIO_ALLOW_SIGNED_FALLBACK=true`.

Successful `RouteOption` payloads include additive `scenario_summary`.  
Successful uncertainty metadata includes additive scenario fields:

- `scenario_mode`
- `scenario_profile_version`
- `scenario_sigma_multiplier`
- `scenario_context_key`
- `scenario_live_as_of_utc`

Pareto diagnostics include additive scenario candidate-impact keys:

- `scenario_candidate_family_count`
- `scenario_candidate_jaccard_vs_baseline`
- `scenario_edge_scaling_version`

`POST /scenario/compare` includes additive `baseline_mode` and it is fixed to `no_sharing`.
`/scenario/compare` deltas now include nullable deltas with per-metric status fields (`ok`/`missing`) to avoid false zero-delta masking.
When a metric is missing, the compare payload also carries per-metric provenance:

- `*_reason_code`
- `*_missing_source`
- `*_reason_source`

## Fuel Output Additions

On successful route options, fuel provenance and quantiles are emitted in additive fields (segment rows and route weather summary), including:

- `fuel_price_source`, `fuel_price_as_of`
- `consumption_model_source`, `consumption_model_version`, `consumption_model_as_of_utc`
- `fuel_liters_p10`, `fuel_liters_p50`, `fuel_liters_p90`
- `fuel_cost_p10_gbp`, `fuel_cost_p50_gbp`, `fuel_cost_p90_gbp`

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
