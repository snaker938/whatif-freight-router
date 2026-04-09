# Model Assets and Data Sources

Last Updated: 2026-04-09
Applies To: `backend/assets/uk/*`, `backend/out/model_assets/*`, live-source settings in `backend/app/settings.py`

This page tracks where backend model inputs come from, where compiled artifacts are written, and which strict gates protect route-producing APIs.

## Current Evidence Snapshot

- `backend/out/model_assets/manifest.json` now reports `version=model-v2-uk`, `source_policy=repo_local_fresh`, `generated_at_utc=2026-03-21T13:09:12.262992Z`, and 19 tracked assets.
- `backend/out/model_assets/preflight_live_runtime.json` passes with `required_ok=true`, `required_failure_count=0`, `strict_live_data_required=true`, `live_runtime_data_enabled=true`, `scenario_contexts=384`, `coverage.overall=1.0`, `toll_rule_count=220`, `toll_topology_segments=28`, `stochastic_regime_count=18`, `departure_region_count=11`, and `bank_holiday_count=134`.
- `backend/out/model_assets/routing_graph_coverage_report.json` passes coverage with `16,782,614` nodes, `17,271,476` edges, `worst_fixture_nearest_node_m=2545.053`, `graph_size_mb=4123.27`, and the UK bounding box `lat 49.75..61.1`, `lon -8.75..2.25`.
- The current compiled fuel surface is `uk_fuel_surface_v1` with axes `vehicle_class=4`, `load_factor=4`, `speed_kmh=5`, `grade_pct=5`, and `ambient_temp_c=5`.
- The current toll-confidence calibration is `uk-toll-confidence-v2-empirical`, with logit coefficients `intercept=0.074138`, `class_signal=0.07386`, `segment_signal=0.107248`, and reliability bins calibrated to `0.1`, `0.3`, `0.5455`, `0.7`, and `0.9`.

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
- Current validation snapshot: the coverage report passes for the full UK graph and the worst fixture is still within 2.55 km of a graph node, so the graph is large but still fixture-covering.

### Toll Topology and Tariffs

- Purpose: map route geometry to tolled segments and apply class-aware tariffs.
- Build scripts:
  - `backend/scripts/extract_osm_tolls_uk.py`
  - `backend/scripts/build_pricing_tables_uk.py`
- Live refresh inputs:
  - `LIVE_TOLL_TOPOLOGY_URL`
  - `LIVE_TOLL_TARIFFS_URL`
- Runtime gate signals: `toll_topology_unavailable`, `toll_tariff_unavailable`, `toll_tariff_unresolved`
- Current calibration snapshot: `backend/assets/uk/toll_confidence_calibration_uk.json` is versioned `uk-toll-confidence-v2-empirical` and is copied into `backend/out/model_assets/` as part of the manifest.

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
- Current compiled snapshot: the empirical calibration yields 18 regimes, 2,832 posterior context keys, 12 hour-slot coverage, 9 corridor coverage, holdout coverage 1.0, PIT mean 0.5149096244101729, CRPS mean 0.47377558811984366, and duration MAPE 0.14058663646867195.

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
- Current compiled snapshot: `scenario_profiles_uk_v2_live` contains 384 contextual profiles, uses `temporal_forward_plus_corridor_block`, and reports a 192-context holdout slice with 6 hour slots and 16 corridors covered. The holdout metrics currently show `mode_separation_mean=0.169764`, `duration_mape=0.0`, `monetary_mape=0.0`, `emissions_mape=0.0`, `full_identity_share=0.291667`, `projection_dominant_context_share=0.5`, `observed_mode_context_share=0.5`, and `observed_mode_row_share=0.5`.

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
- Current raw provenance: fuel history now contains 1,190 rows from two public CSV sources, and the preflight runtime snapshot records fuel as-of `2026-03-23T00:00:00Z` plus `price_per_kg=0.101` and `scope_adjusted_emissions=1.121`.

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

