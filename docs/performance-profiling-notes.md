# Performance Profiling Notes

Last Updated: 2026-02-19  
Applies To: backend benchmark scripts and runtime budgets

## Main Benchmark Entry Points

From `backend/`:

```powershell
uv run python scripts/benchmark_model_v2.py
uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212
```

## Quality + Performance Gate Sequence

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/score_model_quality.py
uv run python scripts/benchmark_model_v2.py
```

## Target Runtime Gates

- Warm-cache `P95 < 2000ms` for `/route`
- Warm-cache `P95 < 2000ms` for `/pareto`

## Terrain Request-Time Live Notes

Terrain sampling now supports request-time live DEM fetch with strict hard-fail behavior.

Required env for strict live terrain:

```powershell
LIVE_TERRAIN_DEM_URL_TEMPLATE=https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif
LIVE_TERRAIN_REQUIRE_URL_IN_STRICT=true
LIVE_TERRAIN_ALLOW_SIGNED_FALLBACK=false
LIVE_TERRAIN_ALLOWED_HOSTS=s3.amazonaws.com
```

Performance controls:

```powershell
LIVE_TERRAIN_CACHE_DIR=backend/out/model_assets/terrain/live_tile_cache
LIVE_TERRAIN_CACHE_MAX_TILES=1024
LIVE_TERRAIN_CACHE_MAX_MB=2048
LIVE_TERRAIN_MAX_REMOTE_TILES_PER_ROUTE=96
LIVE_TERRAIN_FETCH_RETRIES=2
LIVE_TERRAIN_CIRCUIT_BREAKER_FAILURES=8
LIVE_TERRAIN_CIRCUIT_BREAKER_COOLDOWN_S=30
```

Operational notes:
- Runtime fetches each required tile once per route run, then reuses it for remaining samples in that run.
- Missing/blocked URL or live fetch failure in strict mode can surface as `terrain_dem_asset_unavailable`.
- Keep caches warm for benchmark runs to maintain stable P95.

## Related Docs

- [Documentation Index](README.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
