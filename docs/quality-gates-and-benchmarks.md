# Quality Gates and Benchmarks

Last Updated: 2026-02-19  
Applies To: backend quality harness and performance gates

## Required Commands

From `backend/`:

```powershell
uv run python scripts/build_model_assets.py
uv run python scripts/score_model_quality.py
uv run python scripts/benchmark_model_v2.py
```

## Minimum Acceptance Gates

- all subsystem quality scores meet configured threshold
- dropped routes stay at configured maximum (strict default is zero)
- warm-cache runtime gate for `/route` and `/pareto` passes

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
