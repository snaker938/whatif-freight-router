# Performance Profiling and Optimization Notes

This note documents how to profile batch Pareto performance in the current v0
stack and where to optimize first.

## Benchmark harness

Use the benchmark harness from `backend/`:

```powershell
uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212
```

Optional live backend run:

```powershell
uv run python scripts/benchmark_batch_pareto.py --mode live --pair-count 100 --backend-url http://localhost:8000
```

## Output log fields

Each benchmark JSON output includes:

- `pair_count`
- `mode`
- `duration_ms`
- `peak_memory_bytes`
- `error_count`
- `timestamp`
- `run_id`
- `log_path`

## Profiling workflow

1. Run in-process fake mode to establish a deterministic baseline.
2. Run live mode against backend+OSRM for realistic runtime variance.
3. Compare:
   - runtime slope versus pair count (50, 100, 200)
   - peak memory growth
   - error count under higher concurrency
4. Store benchmark logs in `backend/out/benchmarks/` for traceability.

## Current likely bottlenecks

- Route fetch fan-out and retries against OSRM.
- Python-level candidate processing and Pareto filtering per OD.
- JSON serialization cost for large batch responses/artifacts.

## Optimization notes

- Keep `batch_concurrency` tuned to avoid oversubscribing OSRM.
- Avoid unnecessary repeated route parsing in batch flows.
- Keep artifact writes buffered and deterministic (single-pass CSV write).
- Prefer measuring before tuning; preserve benchmark logs for each config
  change.

## Assumptions and limitations

- In-process fake mode does not represent full network/OSRM latency.
- Memory readings use process-local `tracemalloc`, not full system telemetry.
- Current algorithm remains candidate-based Pareto, not full MOSP.
