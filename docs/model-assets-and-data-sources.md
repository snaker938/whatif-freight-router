# Model Assets and Data Sources

Last Updated: 2026-04-09  
Applies To: `backend/assets/uk/*`, `backend/out/model_assets/*`, raw live-source captures under `backend/data/raw/uk/*`, and strict runtime settings

This page tracks where backend model inputs come from, where compiled artifacts are written, and which strict gates protect route-producing APIs.

## Asset Locations

- curated reference inputs: `backend/assets/uk/`
- generated runtime assets: `backend/out/model_assets/`
- staged subset/runtime graph assets: `backend/out/model_assets/staged_subsets/`, `backend/out/route_graph_subsets/`
- raw live/source captures: `backend/data/raw/uk/`
- runtime custom vehicle profiles: `backend/out/config/`

Generated files under `backend/out/` are runtime outputs, not source-of-truth configuration.

## Asset Family Map

### Routing Graph

- purpose: graph-native candidate generation and strict route fallback behavior
- build path: `backend/scripts/build_routing_graph_uk.py`
- aggregated build entry: `backend/scripts/build_model_assets.py`
- runtime signals: `routing_graph_unavailable`, `routing_graph_fragmented`, `routing_graph_disconnected_od`, `routing_graph_coverage_gap`
- staged thesis subsets built through `backend/scripts/build_route_graph_subset.py` now write three sibling runtime artifacts for the same subset graph: the JSON asset, a binary cache at `*.json.pkl`, and a graph-ready compact bundle at `*.compact.pkl`
- the subset builder also writes a sibling `*.meta.json` report with `corridor_km`, `filter_mode`, `corridor_union_bbox`, `nodes_kept`, and `edges_kept`
- runtime load order for `.subset.` assets now prefers the staged binary cache first, then falls back to the compact bundle, then finally to the JSON asset when neither cache is available or valid

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
- `backend/data/raw/uk/scenario_live_observed_20260404.jsonl`
- `backend/data/raw/uk/scenario_live_observed_head_restore.jsonl`
- `backend/data/raw/uk/scenario_live_observed_strict.jsonl`
- `backend/data/raw/uk/scenario_mode_outcomes_observed.jsonl`
- `backend/data/raw/uk/scenario_mode_outcomes_observed.summary.json`
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

## Evaluation Corpus Assets

The checked-in evaluation corpora are now part of the asset story because the current proof and tuning bundles depend on them directly.

- baseline broad corpus: `backend/data/eval/uk_od_corpus_thesis_broad.csv`
- expanded broad corpus: `backend/data/eval/uk_od_corpus_thesis_broad_expanded_120.csv`
- expanded broad summary: `backend/data/eval/uk_od_corpus_thesis_broad_expanded_120.summary.json`
- staged broadening corpus: `backend/data/eval/uk_od_corpus_thesis_broad_expanded_1200.csv`
- staged shard family: `backend/data/eval/thesis_shards_1200/uk_od_corpus_thesis_broad_expanded_1200_s01of10.csv` through `..._s10of10.csv`
- harder-story corpora: `backend/data/eval/uk_od_corpus_hard_mixed_24.csv` and `backend/data/eval/uk_od_corpus_longcorr_hard_32.csv`
- focused / tuning corpora: `backend/data/eval/uk_od_corpus_dominance_cluster_8.csv`, `backend/data/eval/uk_od_corpus_london_newcastle_family_4.csv`, `backend/data/eval/uk_od_corpus_london_newcastle_manchester_liverpool_family_12.csv`, and `backend/data/eval/uk_od_cardiff_liverpool_single.csv`

Current checked-in counts:

- `uk_od_corpus_thesis_broad_expanded_120.csv` carries `120` rows / `120` unique `od_id` values
- `uk_od_corpus_thesis_broad_expanded_1200.csv` carries `1200` rows / `1200` unique `od_id` values
- `backend/data/eval/thesis_shards_1200/` splits that staged `1200`-row corpus into `10` shard CSVs of `120` rows each

The scripts that build or widen these corpora are:

- `backend/scripts/build_od_corpus_uk.py`
- `backend/scripts/expand_thesis_broad_corpus.py`
- `backend/scripts/run_thesis_campaign.py`
- `backend/scripts/compose_thesis_sharded_report.py`

The current `backend/out/thesis_1200_s01_repo_local/` directory should be read as staging evidence only. Its checked-in artifact directory currently contains `repo_asset_preflight.json` and `baseline_smoke_summary.json`, not a completed thesis bundle.

## Build and Refresh Commands

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/build_routing_graph_uk.py
uv run python scripts/build_route_graph_subset.py --help
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
- staged subset assets are part of that strict-runtime story for low-RAM campaigns; when `ROUTE_GRAPH_ASSET_PATH` points at a subset graph, the sibling `*.meta.json`, `*.json.pkl`, and `*.compact.pkl` files are part of the reproducible asset surface

The exact strict-policy behavior depends on settings resolution, especially the live-source policy path, so this page should be read as a map to the source assets and settings, not as a hard-coded claim that every individual switch is always forced the same way in every environment.

## Asset Readiness In Evaluation Outputs

Evaluation and benchmarking artifacts may include readiness evidence such as:

- repo_asset_preflight.json in a completed artifact directory
- evaluation_manifest.json in a completed artifact directory
- signed manifests and provenance logs

These are the primary places to inspect asset freshness and strict readiness in completed runs. The manifest and summary files also carry the current evaluator scaffolding fields when present: `cohort_scaffolding` uses `thesis_cohort_scaffolding_v2`, `metric_family_scaffolding` uses `thesis_metric_family_scaffolding_v1`, and those scaffold payloads are expected to appear in `evaluation_manifest.json`, `thesis_metrics.json`, `thesis_plots.json`, `results.json`, and `metadata.json` for thesis-evaluation runs.

The route-facing artifact layer should be read as an instrumented honesty surface, not as a fresh empirical proof of performance in this pass. That includes the public decision-package family `decision_package.json`, `preference_summary.json`, `support_summary.json`, `support_trace.jsonl`, `support_provenance.json`, `certified_set.json`, `certified_set_routes.jsonl`, and the conditional abstention/witness/controller summaries when they are populated, alongside the route-seam replay and trace artifacts `voi_action_trace.json`, `voi_controller_state.jsonl`, `voi_controller_trace_summary.json`, `voi_replay_oracle_summary.json`, `controller_trace.jsonl`, `theorem_hook_map.json`, `lane_manifest.json`, and `final_route_trace.json`.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
