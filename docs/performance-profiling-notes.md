# Performance Profiling Notes

Last Updated: 2026-04-09
Applies To: `backend/scripts/benchmark_model_v2.py`, `backend/scripts/benchmark_batch_pareto.py`, `backend/scripts/benchmark_route_graph_warmup.py`, `backend/scripts/run_hot_rerun_benchmark.py`, `backend/scripts/validate_graph_coverage.py`, and thesis evaluation runtime artifacts

## Primary Benchmark Entry Points

From `backend/`:

```powershell
uv run python scripts/benchmark_model_v2.py --iterations 8 --p95-gate-ms 2000
uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212
uv run python scripts/benchmark_route_graph_warmup.py
uv run python scripts/run_hot_rerun_benchmark.py
```

Related analysis scripts:

- `backend/scripts/run_sensitivity_analysis.py`
- `backend/scripts/run_robustness_analysis.py`
- `backend/scripts/validate_graph_coverage.py`

## Benchmark Defaults That Matter

`backend/scripts/benchmark_model_v2.py` currently profiles `build_option()` using:

- fixture corpus `backend/tests/fixtures/uk_routes`
- `8` iterations by default
- `rigid_hgv`
- `ScenarioMode.NO_SHARING`
- `use_tolls=false`
- `toll_cost_per_km=0.2`
- `carbon_price_per_kg=0.12`
- stochastic config `enabled=true`, `seed=42`, `sigma=0.08`, `samples=32`
- emissions context `fuel_type=diesel`, `euro_class=euro6`, `ambient_temp_c=12`
- departure time `2026-02-18T08:30:00Z`
- one flat-terrain pass and one hilly-terrain pass
- p95 success gate `2000 ms`

This benchmark is the steady-state micro-benchmark. It should not be compared directly with thesis-lane timings that include startup, baseline acquisition, certification, and reporting overhead.

## Current Graph-Scale Context

`backend/out/model_assets/routing_graph_coverage_report.json` currently records:

- `16,782,614` nodes
- `17,271,476` edges
- graph size about `4123.27 MB`
- worst fixture nearest-node distance `2545.053 m`
- UK bounding box `lat 49.75..61.1`, `lon -8.75..2.25`
- `coverage_passed: true`

That graph size explains why warmup strategy is material to user-visible latency.

## Latest Local Thesis Runtime Evidence

The newest checked thesis bundle is `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2`.

Route-runtime summary from `backend/out/thesis_campaigns/dominance_cluster5_cardiff_bath_corr12p5_r2/tranche_001/artifacts/dominance_cluster5_cardiff_bath_corr12p5_r2_t001/thesis_summary.json`:

| Variant | Mean runtime ms | Mean algorithm ms | Runtime ratio vs OSRM | Runtime ratio vs ORS |
| --- | --- | --- | --- | --- |
| `V0` | `9176.741` | `8506.491` | `28.244375` | `26.799141` |
| `A` | `5315.91825` | `4645.66825` | `16.358162` | `15.542344` |
| `B` | `5787.824` | `5117.574` | `17.638963` | `16.880822` |
| `C` | `2794.579` | `2124.329` | `8.785593` | `8.246457` |

Additional current runtime evidence:

- global startup overhead `10322.7 ms`
- warmup amortized `2064.54 ms`
- backend ready wait `161.38 ms`
- route-graph warmup elapsed `10000.0 ms`
- `V0` mean route request time `8506.491 ms`
- `C` mean route request time `2124.329 ms`
- `C` mean runtime per refined candidate `961.951375 ms`
- `C` mean option-build cache-hit rate `0.75`
- `C` option-build cache savings `3271.665 ms` per row

Baseline acquisition in the same thesis run:

- OSRM baseline compute `140.54 ms`, `189.471 km`, `13306.17 s`
- ORS baseline compute `170.91 ms`, `203.868 km`, `18581.61 s`

## Runtime Budgets (Operational Targets)

- warm-cache route-model p95 target from `backend/scripts/benchmark_model_v2.py`: `< 2000 ms`
- warm-cache hilly-terrain p95 target from `backend/scripts/benchmark_model_v2.py`: `< 2000 ms`
- batch throughput depends on `backend/scripts/benchmark_batch_pareto.py` mode and backend concurrency
- route-graph startup is bounded operationally by `ROUTE_GRAPH_WARMUP_TIMEOUT_S=1200`

## Route-Graph Search Controls That Affect Performance

Current defaults from `.env.example`:

- `ROUTE_GRAPH_MAX_STATE_BUDGET=1200000`
- `ROUTE_GRAPH_STATE_BUDGET_PER_HOP=1600`
- `ROUTE_GRAPH_STATE_BUDGET_RETRY_MULTIPLIER=2.5`
- `ROUTE_GRAPH_STATE_BUDGET_RETRY_CAP=8000000`
- `ROUTE_GRAPH_SEARCH_INITIAL_TIMEOUT_MS=30000`
- `ROUTE_GRAPH_SEARCH_RETRY_TIMEOUT_MS=120000`
- `ROUTE_GRAPH_SEARCH_RESCUE_TIMEOUT_MS=150000`
- `ROUTE_GRAPH_LONG_CORRIDOR_THRESHOLD_KM=150`
- `ROUTE_GRAPH_LONG_CORRIDOR_MAX_PATHS=4`
- `ROUTE_CANDIDATE_PREFILTER_MULTIPLIER=3`
- `ROUTE_CANDIDATE_PREFILTER_MULTIPLIER_LONG=2`
- `ROUTE_OPTION_SEGMENT_CAP=160`
- `ROUTE_OPTION_SEGMENT_CAP_LONG=40`

These knobs are the main place where algorithmic completeness and latency are deliberately traded off.

## Terrain Live-Fetch Performance Controls

Strict terrain can fetch remote DEM tiles at request time. Key controls:

```powershell
LIVE_TERRAIN_REQUIRE_URL_IN_STRICT=true
LIVE_TERRAIN_ALLOW_SIGNED_FALLBACK=false
LIVE_TERRAIN_ALLOWED_HOSTS=s3.amazonaws.com
LIVE_TERRAIN_TILE_MAX_AGE_DAYS=7
LIVE_TERRAIN_CACHE_MAX_TILES=1024
LIVE_TERRAIN_CACHE_MAX_MB=2048
LIVE_TERRAIN_MAX_REMOTE_TILES_PER_ROUTE=96
LIVE_TERRAIN_FETCH_RETRIES=2
LIVE_TERRAIN_CIRCUIT_BREAKER_FAILURES=8
LIVE_TERRAIN_CIRCUIT_BREAKER_COOLDOWN_S=30
```

Practical notes:

- graph warmup materially affects first-use latency
- repeated route options benefit from option-build reuse and cache hits
- repeated terrain samples benefit from tile-cache reuse when enabled
- strict fetch failures map to canonical terrain reason codes rather than silent fallback

## Low-Risk Local Profiling Strategy

On resource-constrained machines:

1. Run one benchmark script at a time.
2. Keep the backend otherwise idle.
3. Avoid running full pytest and graph-heavy benchmarks together.
4. Profile steady-state route building separately from thesis-lane orchestration.
5. Use `scripts/run_backend_tests_safe.ps1` before profiling if the machine is already under load.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
