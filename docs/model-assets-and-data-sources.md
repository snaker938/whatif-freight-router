# Model Assets and Data Sources

Last Updated: 2026-02-21  
Applies To: `backend/assets/uk/*`, `backend/out/model_assets/*`

## Source and Build Policy

- Source snapshots and small curated files are in `backend/assets/uk/`.
- Generated compiled artifacts are in `backend/out/model_assets/`.
- Generated artifacts are not committed.

## Key Asset Families

- routing graph
- toll topology
- toll tariffs
- departure profiles
- stochastic calibration
- terrain DEM index/tiles
- fuel price tables
- fuel consumption surface (`fuel_consumption_surface_uk.json`)
- fuel uncertainty surface (`fuel_uncertainty_surface_uk.json`)
- scenario profiles (`scenario_profiles_uk.json`)
- scenario live fallback snapshot (`scenario_live_snapshot_uk.json`)
- vehicle profiles (`vehicle_profiles_uk.json`)
- carbon schedule/intensity tables

## Build Commands

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/build_routing_graph_uk.py
uv run python scripts/build_terrain_tiles_uk.py --source-dem-glob "assets/uk/dem/*.tif"
```

## Freshness and Strictness

When strict data requirements fail (missing/stale required model data), route-producing endpoints fail with reason-coded `422`.

Fuel strict policy details:

- live-first lookup uses `LIVE_FUEL_PRICE_URL`
- strict URL requirement: `LIVE_FUEL_REQUIRE_URL_IN_STRICT`
- signed fallback gate: `LIVE_FUEL_ALLOW_SIGNED_FALLBACK`
- signature enforcement: `LIVE_FUEL_REQUIRE_SIGNATURE`
- auth inputs supported: bearer token (`LIVE_FUEL_AUTH_TOKEN`) and API key header (`LIVE_FUEL_API_KEY`, `LIVE_FUEL_API_KEY_HEADER`)

Scenario strict policy details:

- scenario profile asset: `backend/assets/uk/scenario_profiles_uk.json`
- strict live coefficient source: `LIVE_SCENARIO_COEFFICIENT_URL` (required when strict runtime is enabled)
- coefficient freshness bound: `LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES`
- live free-source providers:
  - WebTRIS (`LIVE_SCENARIO_WEBTRIS_SITES_URL`, `LIVE_SCENARIO_WEBTRIS_DAILY_URL`)
  - Traffic England (`LIVE_SCENARIO_TRAFFIC_ENGLAND_URL`)
  - DfT raw counts (`LIVE_SCENARIO_DFT_COUNTS_URL`)
  - Open-Meteo (`LIVE_SCENARIO_OPEN_METEO_FORECAST_URL`)
- trusted host allowlist: `LIVE_SCENARIO_ALLOWED_HOSTS`
- strict URL policy: `LIVE_SCENARIO_REQUIRE_URL_IN_STRICT`
- signed fallback gate: `LIVE_SCENARIO_ALLOW_SIGNED_FALLBACK` (strict default `false`)
- freshness bound: `LIVE_SCENARIO_MAX_AGE_MINUTES`
- signature enforcement for fallback assets: `SCENARIO_REQUIRE_SIGNATURE`
- strict monotonicity is enforced for all factors:
  `no_sharing >= partial_sharing >= full_sharing`
- strict full-mode cap is enforced for p50 pressure multipliers:
  `full_sharing p50 <= 1.0`

Stochastic strict policy details:

- stochastic regimes must include `posterior_model.context_to_regime_probs`
- each regime must use `transform_family: quantile_mapping_v1`
- each regime must include `shock_quantile_mapping` for:
  - `traffic`
  - `incident`
  - `weather`
  - `price`
  - `eco`

Scenario ingestion/build helpers:

- `backend/scripts/fetch_scenario_live_uk.py`
- `backend/scripts/build_scenario_profiles_uk.py`

`fetch_scenario_live_uk.py` supports strict batch pulls for realism corpora:

- `--batch` to sample corridor/day/hour grids
- `--output-jsonl` append-only corpus output (for example `backend/data/raw/uk/scenario_live_observed.jsonl`)
- optional `--project-modes-from-artifact` to emit `modes{}` rows for calibration tooling compatibility

Vehicle profile strict policy details:

- built-in vehicle profiles are loaded from signed `backend/assets/uk/vehicle_profiles_uk.json`
- custom profiles are persisted in `backend/out/config/vehicles_v2.json`
- legacy `vehicles.json` custom payloads are auto-migrated to v2
- unknown profile IDs fail route-producing endpoints with `vehicle_profile_unavailable`

## Related Docs

- [Documentation Index](README.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Backend APIs and Tooling](backend-api-tools.md)
