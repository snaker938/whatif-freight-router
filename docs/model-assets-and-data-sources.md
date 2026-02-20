# Model Assets and Data Sources

Last Updated: 2026-02-19  
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

## Related Docs

- [Documentation Index](README.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
- [Backend APIs and Tooling](backend-api-tools.md)
