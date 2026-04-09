# Quality Gates and Benchmarks

Last Updated: 2026-04-09
Applies To: `backend/scripts/preflight_live_runtime.py`, `backend/scripts/score_model_quality.py`, `backend/scripts/benchmark_model_v2.py`, `backend/scripts/benchmark_batch_pareto.py`, `backend/scripts/validate_graph_coverage.py`, thesis evaluation artifacts under `backend/out/thesis_campaigns/*`, and CI lanes in [.github/workflows/backend-ci.yml](../.github/workflows/backend-ci.yml)

This page defines the backend acceptance gates used locally and in CI, and records the latest local evidence currently present in the repo.

## Core Gate Sequence

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/preflight_live_runtime.py
uv run python scripts/score_model_quality.py
uv run python scripts/benchmark_model_v2.py --iterations 8 --p95-gate-ms 2000
```

Targeted subsystem scoring examples:

```powershell
uv run python scripts/score_model_quality.py --subsystem fuel_price
uv run python scripts/score_model_quality.py --subsystem scenario_profile
uv run python scripts/score_model_quality.py --subsystem stochastic_sampling
uv run python scripts/score_model_quality.py --subsystem toll_classification
```

Batch and thesis-oriented benchmark helpers:

```powershell
uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212
uv run python scripts/run_thesis_evaluation.py ...
uv run python scripts/validate_graph_coverage.py
```

## Latest Local Validation

### Strict preflight evidence

`backend/out/model_assets/preflight_live_runtime.json` records a successful strict run at `2026-04-04T15:48:39Z` with:

- `required_ok: true`
- `required_failure_count: 0`
- scenario profile version `scenario_profiles_uk_v2_live`
- scenario live-context coverage `1.0` for WebTRIS, Traffic England, DfT, Open-Meteo, and overall coverage
- 384 scenario contexts recorded by preflight
- 220 toll tariff rules and 28 toll-topology segments
- 18 stochastic regimes and 11 departure-profile regions
- 134 bank holidays
- carbon policy `0.101 GBP/kg` with scope-adjusted emissions factor `1.121`
- OSRM smoke route `189471.0 m / 8794.2 s`
- ORS smoke route `203868.1 m / 12280.8 s`, engine version `9.7.1`, graph identity `graph_identity_verified`

### Routing-graph evidence

`backend/out/model_assets/routing_graph_coverage_report.json` currently records:

- 16,782,614 nodes
- 17,271,476 edges
- graph size about `4123.27 MB`
- worst fixture nearest-node gap `2545.053 m`
- bounding box `lat 49.75..61.1`, `lon -8.75..2.25`
- `coverage_passed: true`

### Latest thesis-lane benchmark evidence

The newest checked thesis campaign bundle is `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2`.

Campaign-level validity:

- 20 evaluation rows and 4 summary rows
- `scenario_profile_unavailable_rate: 0.0`
- `strict_live_readiness_pass_rate: 1.0`
- `evaluation_rerun_success_rate: 0.8`
- backend ready wait `161.38 ms`
- route-graph warmup elapsed `10000.0 ms`

Baseline smoke from `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/thesis_metrics.json`:

- OSRM `140.54 ms`, `189.471 km`, `13306.17 s`
- local ORS `170.91 ms`, `203.868 km`, `18581.61 s`

Variant summary from `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/thesis_summary.json`:

| Variant | Mean runtime ms | Mean algorithm ms | Mean hypervolume | Key result |
| --- | --- | --- | --- | --- |
| `V0` | `9176.741` | `8506.491` | `1954475824.173189` | matched-budget legacy comparator |
| `A` | `5315.91825` | `4645.66825` | `2234025023.091551` | DCCS runtime improvement over `V0` |
| `B` | `5787.824` | `5117.574` | `2234025023.091551` | mean certificate `0.950546` |
| `C` | `2794.579` | `2124.329` | `2234025023.091551` | fastest thesis lane with mean certificate `0.8` |

Across `A`, `B`, and `C`, the current bundle records the same mean objective deltas versus `V0`:

- weighted-margin gain `1.5225`
- balanced-gain delta `0.015621`
- duration gain `1227.32 s`
- monetary gain `2.47`
- emissions gain `-1.4795 kg`

These thesis runtimes are not the same thing as the `backend/scripts/benchmark_model_v2.py` p95 gate. They include thesis-run orchestration, baseline acquisition, certification, and startup overhead.

## CI Lanes (Authoritative)

Workflow: [.github/workflows/backend-ci.yml](../.github/workflows/backend-ci.yml)

### `fast-lane`

- `STRICT_RUNTIME_TEST_BYPASS=1`
- deterministic fixture-first smoke/regression subset
- validates day-to-day behavior with short runtime

### `strict-live-lane`

- `STRICT_RUNTIME_TEST_BYPASS=0`
- signed fallback disabled for key feeds
- validates strict reason-code parity, strict data-path behavior, and fail-closed subsystem behavior

## Minimum Acceptance Gates

- subsystem quality scores pass configured thresholds
- dropped routes do not exceed configured cap, with the strict target effectively `0`
- `backend/scripts/benchmark_model_v2.py` keeps flat and hilly `p95_ms` under the configured gate, which defaults to `2000 ms`
- strict reason-code behavior remains stable for missing, stale, invalid, or unsupported model data
- graph coverage remains inside the UK asset guardrails

## Quality Thresholds

`backend/scripts/score_model_quality.py` currently enforces a score threshold of `95` for:

- `risk_aversion`
- `dominance`
- `scenario_profile`
- `departure_time`
- `stochastic_sampling`
- `terrain_profile`
- `toll_classification`
- `fuel_price`
- `carbon_price`
- `toll_cost`

When `STRICT_LIVE_DATA_REQUIRED=true`, the scorer also expects raw evidence to exist for:

- `backend/data/raw/uk/scenario_live_observed.jsonl`
- `backend/data/raw/uk/stochastic_residuals_raw.csv`
- `backend/data/raw/uk/dft_counts_raw.csv`
- `backend/data/raw/uk/fuel_prices_raw.json`
- `backend/data/raw/uk/carbon_intensity_hourly_raw.json`
- `backend/data/raw/uk/toll_classification/`
- `backend/data/raw/uk/toll_pricing/`
- `backend/data/raw/uk/toll_tariffs_operator_truth.json`

## Benchmark Defaults

`backend/scripts/benchmark_model_v2.py` currently profiles `build_option()` using:

- fixture corpus `backend/tests/fixtures/uk_routes`
- `8` iterations by default
- `rigid_hgv`
- `ScenarioMode.NO_SHARING`
- `use_tolls=false`
- `toll_cost_per_km=0.2`
- `carbon_price_per_kg=0.12`
- stochastic config `enabled=true`, `seed=42`, `sigma=0.08`, `samples=32`
- emissions context `diesel`, `euro6`, `ambient_temp_c=12`
- departure time `2026-02-18T08:30:00Z`
- one flat-terrain pass and one hilly-terrain pass
- p95 success gate `2000 ms`

## Subsystem Expectations

### Fuel

- energy/emissions quantiles remain ordered
- empirical snapshot ingestion stays fresh enough for strict mode
- signature/auth failures map cleanly to canonical fuel reason codes

### Scenario

- sharing modes remain monotone in the expected direction
- holdout separation and MAPE metrics remain inside the current empirical-fit envelope
- `scenario_summary` stays populated on modeled routes
- missing or invalid scenario assets still fail with canonical strict reason codes

### Stochastic

- posterior regime probabilities and `quantile_mapping_v1` transforms remain present
- regime coverage stays grounded in the 18-regime UK corpus
- clipping, coverage, and calibrated factor scales remain bounded

### Vehicle Profiles

- built-in and custom profiles remain schema-compatible
- strict unknown/invalid mapping stays on `vehicle_profile_unavailable` or `vehicle_profile_invalid`

### Terrain

- UK fail-closed coverage stays above the configured cutoff
- unsupported-region behavior stays explicit
- missing DEM assets still map to canonical terrain failures

## Recommended Test Execution Modes

Full local backend suite:

```powershell
uv run --project backend pytest backend/tests
```

Low-resource sequential execution:

```powershell
.\scripts\run_backend_tests_safe.ps1 -MaxCores 1 -PriorityClass Idle
```

## Docs Drift Check

From repo root:

```powershell
python scripts/check_docs.py
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
