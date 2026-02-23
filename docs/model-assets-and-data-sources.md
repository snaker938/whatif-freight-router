# Model Assets and Data Sources

Last Updated: 2026-02-23  
Applies To: `backend/assets/uk/*`, `backend/out/model_assets/*`, live-source settings in `backend/app/settings.py`

This page tracks where backend model inputs come from, where compiled artifacts are written, and which strict gates protect route-producing APIs.

## Asset Locations

- Source snapshots and curated reference inputs:
  - `backend/assets/uk/`
- Generated strict runtime assets:
  - `backend/out/model_assets/`
- Live scenario collection corpus:
  - `backend/data/raw/uk/scenario_live_observed.jsonl`
- Runtime-generated custom vehicle profiles:
  - `backend/out/config/` (runtime file: vehicles_v2.json)

Generated files in `backend/out/` are runtime artifacts and should not be treated as source-of-truth config.

## Asset Family Map

### Routing Graph

- Purpose: graph-native candidate generation and strict fallback when OSRM refinement fails.
- Build script: `backend/scripts/build_routing_graph_uk.py`
- Aggregated build entry: `backend/scripts/build_model_assets.py`
- Runtime gate signals: `routing_graph_unavailable`, `model_asset_unavailable`

### Toll Topology and Tariffs

- Purpose: map route geometry to tolled segments and apply class-aware tariffs.
- Build scripts:
  - `backend/scripts/extract_osm_tolls_uk.py`
  - `backend/scripts/build_pricing_tables_uk.py`
- Live refresh inputs:
  - `LIVE_TOLL_TOPOLOGY_URL`
  - `LIVE_TOLL_TARIFFS_URL`
- Runtime gate signals: `toll_topology_unavailable`, `toll_tariff_unavailable`, `toll_tariff_unresolved`

### Departure Profiles

- Purpose: departure-time uplift and time-window optimization behavior.
- Build script: `backend/scripts/build_departure_profiles_uk.py`
- Live refresh input: `LIVE_DEPARTURE_PROFILE_URL`
- Runtime gate signal: `departure_profile_unavailable`

### Stochastic Calibration

- Purpose: uncertainty posteriors and calibrated regime shocks.
- Build script: `backend/scripts/build_stochastic_calibration_uk.py`
- Strict-required structure:
  - `posterior_model.context_to_regime_probs`
  - per-regime `transform_family: quantile_mapping_v1`
  - per-regime `shock_quantile_mapping` keys for `traffic`, `incident`, `weather`, `price`, `eco`
- Runtime gate signal: `stochastic_calibration_unavailable`

### Scenario Profile + Live Context

- Purpose: strict scenario multipliers for no/partial/full sharing behavior.
- Build/ingest scripts:
  - `backend/scripts/fetch_scenario_live_uk.py`
  - `backend/scripts/build_scenario_profiles_uk.py`
- Strict-required live coefficient source:
  - `LIVE_SCENARIO_COEFFICIENT_URL`
- Core live context sources:
  - WebTRIS (`LIVE_SCENARIO_WEBTRIS_SITES_URL`, `LIVE_SCENARIO_WEBTRIS_DAILY_URL`)
  - Traffic England (`LIVE_SCENARIO_TRAFFIC_ENGLAND_URL`)
  - DfT raw counts (`LIVE_SCENARIO_DFT_COUNTS_URL`)
  - Open-Meteo (`LIVE_SCENARIO_OPEN_METEO_FORECAST_URL`, `LIVE_SCENARIO_OPEN_METEO_ARCHIVE_URL`)
- Runtime gate signals: `scenario_profile_unavailable`, `scenario_profile_invalid`

### Terrain DEM

- Purpose: terrain-aware fuel/emissions uplift and strict fail-closed coverage behavior.
- Build scripts:
  - `backend/scripts/fetch_public_dem_tiles_uk.py`
  - `backend/scripts/build_terrain_tiles_uk.py`
- Live request-time terrain source:
  - `LIVE_TERRAIN_DEM_URL_TEMPLATE`
- Runtime gate signals: `terrain_dem_asset_unavailable`, `terrain_dem_coverage_insufficient`, `terrain_region_unsupported`

### Fuel + Carbon Inputs

- Fuel strict inputs:
  - `LIVE_FUEL_PRICE_URL`
  - auth: `LIVE_FUEL_AUTH_TOKEN` or `LIVE_FUEL_API_KEY` + `LIVE_FUEL_API_KEY_HEADER`
- Carbon strict input:
  - `LIVE_CARBON_SCHEDULE_URL`
- Runtime gate signals:
  - `fuel_price_auth_unavailable`
  - `fuel_price_source_unavailable`
  - `carbon_policy_unavailable`
  - `carbon_intensity_unavailable`

### Vehicle Profiles

- Built-in strict profile asset:
  - `backend/assets/uk/vehicle_profiles_uk.json`
- Custom profile persistence:
  - runtime output config directory `backend/out/config/` (runtime file name: vehicles_v2.json)
- Runtime gate signals:
  - `vehicle_profile_unavailable`
  - `vehicle_profile_invalid`

## Build/Refresh Commands

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/build_routing_graph_uk.py
uv run python scripts/build_terrain_tiles_uk.py --source-dem-glob "assets/uk/dem/*.tif"
uv run python scripts/build_departure_profiles_uk.py
uv run python scripts/build_stochastic_calibration_uk.py
```

Live-source fetch helpers (from repo root):

```powershell
uv run --project backend python backend/scripts/fetch_scenario_live_uk.py --batch --output-jsonl backend/data/raw/uk/scenario_live_observed.jsonl
uv run --project backend python backend/scripts/fetch_fuel_history_uk.py --output backend/assets/uk/fuel_prices_uk.json
uv run --project backend python backend/scripts/fetch_carbon_intensity_uk.py --schedule backend/assets/uk/carbon_price_schedule_uk.json
```

## Strict Runtime Gate Controls (Settings)

Strict defaults are forced in code (`backend/app/settings.py`) during settings validation.

Key enforced controls:

- `STRICT_LIVE_DATA_REQUIRED=true`
- URL-required flags:
  - `LIVE_SCENARIO_REQUIRE_URL_IN_STRICT=true`
  - `LIVE_FUEL_REQUIRE_URL_IN_STRICT=true`
  - `LIVE_CARBON_REQUIRE_URL_IN_STRICT=true`
  - `LIVE_TOLL_TOPOLOGY_REQUIRE_URL_IN_STRICT=true`
  - `LIVE_TOLL_TARIFFS_REQUIRE_URL_IN_STRICT=true`
  - `LIVE_DEPARTURE_REQUIRE_URL_IN_STRICT=true`
  - `LIVE_STOCHASTIC_REQUIRE_URL_IN_STRICT=true`
  - `LIVE_TERRAIN_REQUIRE_URL_IN_STRICT=true`
- Signed fallback controls forced off for strict production paths:
  - `LIVE_*_ALLOW_SIGNED_FALLBACK=false` (all strict live families)
- Signature requirement:
  - `SCENARIO_REQUIRE_SIGNATURE=true`
  - `LIVE_FUEL_REQUIRE_SIGNATURE=true`

Test-only bypass is explicit:

- `STRICT_RUNTIME_TEST_BYPASS=1` (for controlled tests only)

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)

