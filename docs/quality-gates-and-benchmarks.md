# Quality Gates and Benchmarks

Last Updated: 2026-02-21  
Applies To: backend quality harness and performance gates

## Required Commands

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/score_model_quality.py
uv run python scripts/benchmark_model_v2.py
uv run python scripts/score_model_quality.py --subsystem fuel_price
```

## Minimum Acceptance Gates

- all subsystem quality scores meet configured threshold
- dropped routes stay at configured maximum (strict default is zero)
- warm-cache runtime gate for `/route` and `/pareto` passes

Fuel subsystem scoring emphasizes empirical behavior over metadata-only checks:

- holdout p50 error on consumption outputs
- monotonic sensitivity checks (load/speed/grade/temp)
- quantile validity and coverage (`p10 <= p50 <= p90`)
- strict live-source readiness (URL/auth/signature policy)

Scenario subsystem scoring checks:

- scenario factor monotonicity across modes (`no_sharing >= partial_sharing >= full_sharing`)
- mode separability on holdout routes (duration/money/emissions)
- strict holdout threshold: `mode_separation_mean >= 0.03`
- strict holdout threshold: `duration_mape <= 0.08`
- strict holdout threshold: `monetary_mape <= 0.08`
- strict holdout threshold: `emissions_mape <= 0.08`
- strict holdout threshold: `coverage >= 0.90`
- per-option `scenario_summary` completeness
- uncertainty metadata completeness (`scenario_mode`, `scenario_profile_version`, `scenario_sigma_multiplier`)
- context coverage depth in `scenario_profiles_uk.json`
- holdout metric quality from scenario profile artifact (`holdout_metrics.mode_separation_mean`)
- cross-mode ordering on holdout routes (duration/money/emissions)
- strict-failure behavior when scenario profile assets are missing/invalid
- scenario subsystem score weighting: 70% holdout metrics
- scenario subsystem score weighting: 20% mode ordering/separability
- scenario subsystem score weighting: 10% metadata/contract checks

Stochastic subsystem strictness checks:

- posterior regime model is required (`posterior_model.context_to_regime_probs`)
- transform family is required per regime (`quantile_mapping_v1`)
- shock quantile mappings are required for all factors (`traffic`, `incident`, `weather`, `price`, `eco`)
- clipping diagnostics are scored (`sample_count_clip_ratio`, `sigma_clip_ratio`, `factor_clip_rate`)

Vehicle profile strictness expectations:

- v2 schema coverage for built-in and custom profiles
- explicit class mapping coverage (`fuel_surface_class`, `toll_vehicle_class`, `toll_axle_class`)
- strict unknown-vehicle failure behavior (`vehicle_profile_unavailable`)
- no ID-substring inference reliance in runtime subsystems

CI strictness lanes:

- fast deterministic lane keeps fixture-focused runtime with explicit test bypass where configured
- strict lane runs with `STRICT_RUNTIME_TEST_BYPASS=0` so freshness/fallback-sensitive paths are validated without implicit bypass semantics

## Regression Test Command

From repo root:

```powershell
uv run --project backend pytest backend/tests
```

## Docs Drift Check

From repo root:

```powershell
python scripts/check_docs.py
```

## Related Docs

- [Documentation Index](README.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Strict Error Contract Reference](strict-errors-reference.md)
