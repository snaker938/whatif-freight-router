# Backend APIs and Tooling

Last Updated: 2026-02-23  
Applies To: `backend/app/main.py`, `backend/app/models.py`, `backend/app/settings.py`

This page is the source-of-truth backend API contract for current strict runtime behavior.

## Runtime Contract (Current)

- Runtime is hard-strict by default via `Settings._enforce_strict_runtime_defaults` in `backend/app/settings.py`.
- Route-producing failures are reason-coded and normalized with `normalize_reason_code` in `backend/app/model_data_errors.py`.
- Streaming and non-streaming flows use aligned reason-code semantics.
- Synthetic fallback paths are blocked in strict production paths unless explicitly test-scoped.

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

- `POST /batch/import/csv`
- `POST /batch/pareto`

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

## Request Surface (Core Models)

All route-producing requests are derived from `RouteRequest`/`ParetoRequest`/`BatchParetoRequest` families in `backend/app/models.py`.

Shared fields:

- `origin`, `destination`, optional `waypoints`
- `vehicle_type`
- `scenario_mode` (`no_sharing`, `partial_sharing`, `full_sharing`)
- `weights` (`time`, `money`, `co2`)
- `cost_toggles` (`use_tolls`, `fuel_price_multiplier`, `carbon_price_per_kg`, `toll_cost_per_km`)
- `terrain_profile` (`flat`, `rolling`, `hilly`)
- `stochastic` (`enabled`, `seed`, `sigma`, `samples`)
- `optimization_mode` (`expected_value`, `robust`)
- `risk_aversion`
- `emissions_context`
- `weather`
- `incident_simulation`
- `departure_time_utc`
- `pareto_method` (`dominance`, `epsilon_constraint`)
- `epsilon` (`duration_s`, `monetary_cost`, `emissions_kg`)

Endpoint-specific:

- `POST /batch/pareto`: `pairs` (1..500), optional `seed`, `toggles`, `model_version`
- `POST /batch/import/csv`: `csv_text` plus the same optional controls as batch
- `POST /departure/optimize`: `window_start_utc`, `window_end_utc`, `step_minutes`, optional `time_window`
- `POST /duty/chain`: `stops` (2..50)

## Strict Error Shapes

### Non-stream endpoints (`422`)

```json
{
  "detail": {
    "reason_code": "terrain_dem_coverage_insufficient",
    "message": "Terrain DEM coverage below threshold.",
    "warnings": [
      "Coverage 0.91 < required 0.96"
    ],
    "terrain_dem_version": "uk_dem_v1",
    "terrain_coverage_required": 0.96,
    "terrain_coverage_min_observed": 0.91
  }
}
```

### Stream fatal event (`POST /pareto/stream`, `POST /api/pareto/stream`)

```json
{
  "type": "fatal",
  "reason_code": "epsilon_infeasible",
  "message": "No routes satisfy epsilon constraints for this request.",
  "warnings": []
}
```

## Scenario Runtime Notes

- Scenario policy is context-conditioned and strict-validated.
- `LIVE_SCENARIO_COEFFICIENT_URL` is required in strict runtime.
- Context uses free UK feeds (WebTRIS, Traffic England, DfT raw counts, Open-Meteo).
- Missing/stale/incomplete strict context resolves to scenario reason codes (`scenario_profile_unavailable` / `scenario_profile_invalid`).
- Signed fallback is blocked by strict defaults (`LIVE_SCENARIO_ALLOW_SIGNED_FALLBACK=false` unless overridden for controlled scenarios).

Successful route payloads include additive scenario fields:

- `route.scenario_summary.*`
- uncertainty metadata keys such as `scenario_mode`, `scenario_profile_version`, `scenario_sigma_multiplier`, `scenario_context_key`

`POST /scenario/compare` specifics:

- `baseline_mode` is fixed to `no_sharing`
- `deltas` carry per-metric nullable deltas plus `*_status`, `*_reason_code`, `*_missing_source`, `*_reason_source`

## Vehicle Runtime Notes

- Built-in profiles are strict asset-backed (`backend/assets/uk/vehicle_profiles_uk.json`).
- Custom profiles are persisted in the runtime output config area (`backend/out/config/`, runtime file: vehicles_v2.json).
- Unknown or invalid `vehicle_type` in strict route-producing flows returns `422` with `vehicle_profile_unavailable` or `vehicle_profile_invalid`.

## Batch and Artifact Runtime Notes

- `POST /batch/pareto` processes pairs with bounded concurrency (`BATCH_CONCURRENCY`).
- Batch outputs include per-pair `error` strings in strict format (`reason_code:...; message:...`) when a pair cannot produce routes.
- Artifact set is fixed by `ARTIFACT_FILES` in `backend/app/run_store.py`:
  - results.json
  - results.csv
  - metadata.json
  - routes.geojson
  - results_summary.csv
  - report.pdf
- Signed manifests are written to:
  - `backend/out/manifests/{run_id}.json`
  - `backend/out/scenario_manifests/{run_id}.json`

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [API Cookbook](api-cookbook.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)

