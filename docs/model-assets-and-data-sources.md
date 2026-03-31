# Model Assets and Data Sources

Last Updated: 2026-03-31  
Applies To: `backend/assets/uk/*`, `backend/out/model_assets/*`, raw live-source captures under `backend/data/raw/uk/*`, and strict runtime settings

This page tracks where backend model inputs come from, where compiled artifacts are written, and which strict gates protect route-producing APIs.

## Asset Locations

- curated reference inputs: `backend/assets/uk/`
- generated runtime assets: `backend/out/model_assets/`
- raw live/source captures: `backend/data/raw/uk/`
- runtime custom vehicle profiles: `backend/out/config/`

Generated files under `backend/out/` are runtime outputs, not source-of-truth configuration.

## Asset Family Map

### Routing Graph

- purpose: graph-native candidate generation and strict route fallback behavior
- build path: `backend/scripts/build_routing_graph_uk.py`
- aggregated build entry: `backend/scripts/build_model_assets.py`
- runtime signals: `routing_graph_unavailable`, `routing_graph_fragmented`, `routing_graph_disconnected_od`, `routing_graph_coverage_gap`

### Toll Topology and Tariffs

- `backend/assets/uk/toll_topology_uk.json`
- `backend/assets/uk/toll_tariffs_uk.json`
- `backend/assets/uk/toll_tariffs_uk.yaml`
- runtime signals: `toll_topology_unavailable`, `toll_tariff_unavailable`, `toll_tariff_unresolved`

### Departure Profiles

- `backend/assets/uk/departure_profiles_uk.json`
- runtime signal: `departure_profile_unavailable`

### Stochastic Calibration

- `backend/assets/uk/stochastic_regimes_uk.json`
- `backend/assets/uk/stochastic_residual_priors_uk.json`
- `backend/assets/uk/stochastic_residuals_empirical.csv`
- `backend/data/raw/uk/stochastic_residuals_raw.csv`
- runtime signal: `stochastic_calibration_unavailable`

### Scenario Profiles and Live Context

- `backend/assets/uk/scenario_profiles_uk.json`
- `backend/data/raw/uk/scenario_live_observed.jsonl`
- `backend/data/raw/uk/scenario_live_observed_strict.jsonl`
- `backend/data/raw/uk/scenario_mode_outcomes_observed.jsonl`
- `backend/data/raw/uk/scenario_mode_outcomes_observed_strict.jsonl`
- runtime signals: `scenario_profile_unavailable`, `scenario_profile_invalid`

### Terrain

- `backend/assets/uk/terrain_dem_grid_uk.json`
- live request-time terrain source is configured through `LIVE_TERRAIN_DEM_URL_TEMPLATE`
- runtime signals: `terrain_dem_asset_unavailable`, `terrain_dem_coverage_insufficient`, `terrain_region_unsupported`

### Fuel and Carbon

- `backend/assets/uk/fuel_prices_uk.json`
- `backend/assets/uk/carbon_price_schedule_uk.json`
- `backend/assets/uk/carbon_intensity_hourly_uk.json`
- raw captures such as `backend/data/raw/uk/fuel_prices_raw.json` and `backend/data/raw/uk/carbon_intensity_hourly_raw.json`
- runtime signals: `fuel_price_auth_unavailable`, `fuel_price_source_unavailable`, `carbon_policy_unavailable`, `carbon_intensity_unavailable`

### Vehicle Profiles

- `backend/assets/uk/vehicle_profiles_uk.json`
- runtime signals: `vehicle_profile_unavailable`, `vehicle_profile_invalid`

## Build and Refresh Commands

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/build_routing_graph_uk.py
uv run python scripts/build_terrain_tiles_uk.py --source-dem-glob "assets/uk/dem/*.tif"
uv run python scripts/build_departure_profiles_uk.py
uv run python scripts/build_stochastic_calibration_uk.py
```

From repo root:

```powershell
uv run --project backend python backend/scripts/fetch_scenario_live_uk.py --batch --output-jsonl backend/data/raw/uk/scenario_live_observed.jsonl
uv run --project backend python backend/scripts/fetch_fuel_history_uk.py --output backend/assets/uk/fuel_prices_uk.json
uv run --project backend python backend/scripts/fetch_carbon_intensity_uk.py --schedule backend/assets/uk/carbon_price_schedule_uk.json
```

## Strict Runtime Policy Notes

The strict-policy story now lives in `backend/app/settings.py`.

Important points:

- `STRICT_LIVE_DATA_REQUIRED` remains the top-level strict gate
- per-family URL requirements exist through `LIVE_*_REQUIRE_URL_IN_STRICT`
- strict route-compute refresh behavior is controlled through `LIVE_ROUTE_COMPUTE_*`
- route graph strict readiness is enforced in settings validation
- strict runtime disables route-graph fast startup

The exact strict-policy behavior depends on settings resolution, especially the live-source policy path, so this page should be read as a map to the source assets and settings, not as a hard-coded claim that every individual switch is always forced the same way in every environment.

## Asset Readiness In Evaluation Outputs

Evaluation and benchmarking artifacts may include readiness evidence such as:

- repo_asset_preflight.json in a completed artifact directory
- evaluation_manifest.json in a completed artifact directory
- signed manifests and provenance logs

These are the primary places to inspect asset freshness and strict readiness in completed runs.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
