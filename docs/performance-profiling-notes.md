# Performance Profiling Notes

Last Updated: 2026-02-23  
Applies To: backend benchmark scripts and strict runtime performance budgets

## Primary Benchmark Entry Points

From `backend/`:

```powershell
uv run python scripts/benchmark_model_v2.py
uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212
```

Related analysis scripts:

- `backend/scripts/run_sensitivity_analysis.py`
- `backend/scripts/run_robustness_analysis.py`
- `backend/scripts/validate_graph_coverage.py`

## Recommended Gate Sequence

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/score_model_quality.py
uv run python scripts/benchmark_model_v2.py
```

## Runtime Budgets (Operational Targets)

- warm-cache `/route` P95 target: `< 2000ms`
- warm-cache `/pareto` P95 target: `< 2000ms`
- batch throughput is bounded by `BATCH_CONCURRENCY`

These are practical operational targets used to detect regressions; exact acceptance may vary by CI runner and dataset freshness.

## Terrain Live-Fetch Performance Controls

Strict terrain can fetch remote DEM tiles at request time. Key controls:

```powershell
LIVE_TERRAIN_DEM_URL_TEMPLATE=https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif
LIVE_TERRAIN_REQUIRE_URL_IN_STRICT=true
LIVE_TERRAIN_ALLOW_SIGNED_FALLBACK=false
LIVE_TERRAIN_ALLOWED_HOSTS=s3.amazonaws.com
LIVE_TERRAIN_CACHE_DIR=backend/out/model_assets/terrain/live_tile_cache
LIVE_TERRAIN_CACHE_MAX_TILES=1024
LIVE_TERRAIN_CACHE_MAX_MB=2048
LIVE_TERRAIN_MAX_REMOTE_TILES_PER_ROUTE=96
LIVE_TERRAIN_FETCH_RETRIES=2
LIVE_TERRAIN_CIRCUIT_BREAKER_FAILURES=8
LIVE_TERRAIN_CIRCUIT_BREAKER_COOLDOWN_S=30
```

Practical notes:

- cache warm-up materially affects first-run latency
- repeated routes benefit from tile cache reuse
- strict fetch failures map to terrain reason codes, not silent fallback

## Low-Risk Local Profiling Strategy

On resource-constrained machines:

1. run one benchmark script at a time
2. keep backend process idle except benchmark
3. avoid running full pytest + benchmark simultaneously
4. use `scripts/run_backend_tests_safe.ps1` for tests to prevent CPU saturation before profiling

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Reproducibility Capsule](reproducibility-capsule.md)

