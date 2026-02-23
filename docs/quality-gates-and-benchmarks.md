# Quality Gates and Benchmarks

Last Updated: 2026-02-23  
Applies To: `backend/scripts/score_model_quality.py`, `backend/scripts/benchmark_model_v2.py`, CI lanes in .github/workflows/backend-ci.yml

This page defines the backend acceptance gates used locally and in CI.

## Core Gate Sequence

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/score_model_quality.py
uv run python scripts/benchmark_model_v2.py
```

Targeted subsystem scoring examples:

```powershell
uv run python scripts/score_model_quality.py --subsystem fuel_price
uv run python scripts/score_model_quality.py --subsystem scenario_profile
uv run python scripts/score_model_quality.py --subsystem stochastic
```

## CI Lanes (Authoritative)

Workflow: .github/workflows/backend-ci.yml

### `fast-lane`

- `STRICT_RUNTIME_TEST_BYPASS=1`
- deterministic fixture-first smoke/regression subset
- validates day-to-day behavior with short runtime

### `strict-live-lane`

- `STRICT_RUNTIME_TEST_BYPASS=0`
- `LIVE_SCENARIO_ALLOW_SIGNED_FALLBACK=0`
- `LIVE_FUEL_ALLOW_SIGNED_FALLBACK=0`
- validates strict reason-code parity and strict data-path behavior

## Minimum Acceptance Gates

- subsystem quality scores pass configured thresholds
- dropped routes do not exceed configured cap (strict default target: `0`)
- warm-cache latency checks remain within configured budgets for `/route` and `/pareto`
- strict reason-code behavior remains stable for missing/stale/invalid model data

## Subsystem Expectations

### Fuel

- consumption sanity and quantile ordering (`p10 <= p50 <= p90`)
- holdout fit quality on empirical samples
- strict live input handling:
  - URL/auth requirements
  - signature policy where required
  - reason-code mapping for auth/source failures

### Scenario

- strict monotonicity across modes (`no_sharing >= partial_sharing >= full_sharing`)
- context-conditioned separability on holdout
- `scenario_summary` completeness on returned options
- uncertainty metadata fields present and coherent
- strict-failure behavior:
  - missing/unavailable profile -> `scenario_profile_unavailable`
  - invalid payload/transform -> `scenario_profile_invalid`

### Stochastic

- posterior context regime probabilities present
- `quantile_mapping_v1` transforms per regime
- required factor mappings: `traffic`, `incident`, `weather`, `price`, `eco`
- clipping/coverage diagnostics remain bounded

### Vehicle Profiles

- v2 profile schema consistency (built-in + custom)
- strict unknown/invalid mapping to:
  - `vehicle_profile_unavailable`
  - `vehicle_profile_invalid`

### Terrain

- UK fail-closed coverage behavior (`terrain_dem_coverage_insufficient`)
- unsupported region behavior (`terrain_region_unsupported`)
- strict DEM availability behavior (`terrain_dem_asset_unavailable`)

## Recommended Test Execution Modes

Full local backend suite (resource intensive):

```powershell
uv run --project backend pytest backend/tests
```

Low-resource sequential execution (preferred on constrained laptops):

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

