# ETA Concept Drift Checks

Last Updated: 2026-02-19  
Applies To: `backend/scripts/check_eta_concept_drift.py`

## Purpose

Detect drift between predicted ETA and observed ETA values using threshold checks.

## Input Columns

Required:

- `predicted_eta_s`
- `observed_eta_s`

Optional:

- `trip_id`

## Command

From `backend/`:

```powershell
uv run python scripts/check_eta_concept_drift.py --input-csv .\eta_observations.csv --mae-threshold-s 120 --mape-threshold-pct 10
```

## Output

The script prints metrics and exits non-zero when thresholds are violated.

## Related Docs

- [Documentation Index](README.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
