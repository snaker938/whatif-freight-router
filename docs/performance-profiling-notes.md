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

## Related Docs

- [Documentation Index](README.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
