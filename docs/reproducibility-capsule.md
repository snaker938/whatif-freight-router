# Reproducibility Capsule and Scripted Demo

This project includes a deterministic demo run script that captures a benchmark
capsule artifact.

## One-command capsule run

From repo root:

```powershell
.\scripts\demo_repro_run.ps1
```

What it does:

1. Runs the benchmark harness in deterministic `inprocess-fake` mode.
2. Uses fixed inputs:
   - `pair_count=100`
   - `seed=20260212`
   - `max_alternatives=3`
3. Writes output to `backend/out/capsule/repro_capsule_<timestamp>.json`.

## Manual equivalent

From `backend/`:

```powershell
uv run python scripts/benchmark_batch_pareto.py --mode inprocess-fake --pair-count 100 --seed 20260212 --max-alternatives 3 --output out/capsule/repro_capsule_manual.json
```

## Capsule contents

The capsule file is a benchmark JSON log containing:

- run metadata (`timestamp`, `mode`, `pair_count`)
- performance (`duration_ms`, `peak_memory_bytes`)
- quality indicators (`error_count`, `run_id`)

## Recommended reproducibility practice

- Keep the same seed and pair count for before/after comparisons.
- Keep benchmark mode unchanged when comparing commits.
- Attach capsule logs to report milestones.

## Assumptions and limitations

- Capsule currently reflects deterministic fake-mode behavior.
- Live OSRM conditions are expected to vary due environment and network state.
