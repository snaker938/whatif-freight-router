# Backend APIs and Tooling Additions

This document summarizes the backend-facing additions delivered in the latest implementation wave.

## Cost Model Toggles

`POST /route`, `POST /pareto`, and `POST /batch/pareto` accept an optional `cost_toggles` object:

```json
{
  "cost_toggles": {
    "use_tolls": true,
    "fuel_price_multiplier": 1.0,
    "carbon_price_per_kg": 0.0,
    "toll_cost_per_km": 0.0
  }
}
```

Notes:
- `fuel_price_multiplier` scales distance/speed-based fuel cost.
- `carbon_price_per_kg` adds `emissions_kg * carbon_price_per_kg` to monetary cost.
- `use_tolls` and `toll_cost_per_km` control toll add-on cost where routes are toll-flagged.

## Editable Vehicle Profiles

Custom vehicle profile APIs:
- `GET /vehicles/custom`
- `POST /vehicles/custom`
- `PUT /vehicles/custom/{vehicle_id}`
- `DELETE /vehicles/custom/{vehicle_id}`

Persistence file:
- `backend/out/config/vehicles.json`

Merged view:
- `GET /vehicles` now returns built-in + custom profiles.

## Route Cache Management

Cache APIs:
- `GET /cache/stats`
- `DELETE /cache`

Metrics:
- `GET /metrics` includes `route_cache` counters (`size`, `hits`, `misses`, `evictions`, `ttl_s`, `max_entries`).

## Signed Manifests and Verification

Manifest signatures:
- manifests include `signature` metadata with `algorithm`, `signed_at`, and `signature`.
- signing uses `HMAC-SHA256`.

APIs:
- `GET /runs/{run_id}/signature`
- `POST /verify/signature`

Environment variable:
- `MANIFEST_SIGNING_SECRET` (default is dev-only; set explicitly in shared environments).

## Provenance Chain

Per-run provenance file:
- `backend/out/provenance/{run_id}.json`

API:
- `GET /runs/{run_id}/provenance`

Required event sequence:
1. `input_received`
2. `candidates_fetched`
3. `options_built`
4. `pareto_selected`
5. `artifacts_written`

## CSV Import and Export Endpoints

Batch import:
- `POST /batch/import/csv`

Expected CSV columns:
- `origin_lat`
- `origin_lon`
- `destination_lat`
- `destination_lon`

Run artifacts:
- `GET /runs/{run_id}/artifacts`
- `GET /runs/{run_id}/artifacts/results.json`
- `GET /runs/{run_id}/artifacts/results.csv`
- `GET /runs/{run_id}/artifacts/metadata.json`
- `GET /runs/{run_id}/artifacts/routes.geojson`
- `GET /runs/{run_id}/artifacts/results_summary.csv`

## Duty Chaining

Endpoint:
- `POST /duty/chain`

Purpose:
- evaluates an ordered list of stops as sequential legs for one vehicle
- returns per-leg selected route and aggregate totals (`duration_s`, `monetary_cost`, `emissions_kg`, optional `energy_kwh`)

Minimal payload:

```json
{
  "stops": [
    { "lat": 52.4862, "lon": -1.8904, "label": "Birmingham" },
    { "lat": 52.2053, "lon": 0.1218, "label": "Cambridge" },
    { "lat": 51.5072, "lon": -0.1276, "label": "London" }
  ],
  "vehicle_type": "rigid_hgv",
  "scenario_mode": "no_sharing"
}
```

## Oracle Quality Dashboard APIs

Endpoints:
- `POST /oracle/quality/check`
- `GET /oracle/quality/dashboard`
- `GET /oracle/quality/dashboard.csv`

Storage artifacts:
- `backend/out/oracle_quality/checks.ndjson`
- `backend/out/oracle_quality/summary.json`
- `backend/out/oracle_quality/dashboard.csv`

Behavior:
- each check record is appended to NDJSON history
- dashboard aggregates pass-rate, schema/signature failures, freshness, and latency by source

## Offline Fallback Behavior

When OSRM requests fail and fallback is enabled, the backend can reuse a last-known snapshot:
- snapshot store: `backend/out/offline/route_snapshots.json`
- setting: `OFFLINE_FALLBACK_ENABLED` (default `true`)

Output markers:
- per-pair batch result includes `fallback_used`
- manifest `execution` includes `fallback_used` and `fallback_count`
- artifact metadata includes `fallback_used` and `fallback_count`

## Analysis Tooling

Robustness runner:
```powershell
cd backend
uv run python scripts/run_robustness_analysis.py --mode inprocess-fake --seeds 11,22,33 --pair-count 100
```

Sensitivity runner:
```powershell
cd backend
uv run python scripts/run_sensitivity_analysis.py --mode inprocess-fake --pair-count 50 --include-no-tolls
```

Outputs are written under:
- `backend/out/analysis`

## Basic RBAC

RBAC is opt-in and token-based:
- `RBAC_ENABLED` (default `false`)
- `RBAC_USER_TOKEN`
- `RBAC_ADMIN_TOKEN`

Header options:
- `X-API-Token: <token>`
- `Authorization: Bearer <token>`

Role policy:
- `user` (or `admin`) token required for compute endpoints (`/route`, `/pareto`, `/batch/pareto`, `/scenario/compare`, `/departure/optimize`, replay compare).
- `admin` token required for mutating admin endpoints (`/vehicles/custom` mutations, `/cache` clear, experiment create/update/delete).
- read-only endpoints remain public by default.
