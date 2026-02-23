# ETA Concept Drift Checks

Last Updated: 2026-02-23  
Applies To: `backend/scripts/check_eta_concept_drift.py`

## Purpose

Checks drift between predicted and observed ETAs and emits machine-readable outputs for monitoring.

## Input Requirements

Required CSV columns:

- `predicted_eta_s`
- `observed_eta_s`

Optional:

- `trip_id`

## Command

From `backend/`:

```powershell
uv run python scripts/check_eta_concept_drift.py `
  --input-csv .\eta_observations.csv `
  --mae-threshold-s 120 `
  --mape-threshold-pct 10 `
  --out-dir out
```

Optional explicit output paths:

```powershell
uv run python scripts/check_eta_concept_drift.py `
  --input-csv .\eta_observations.csv `
  --json-output out\analysis\eta_drift.json `
  --csv-output out\analysis\eta_drift_rows.csv
```

## Output Payload

Script writes:

- JSON summary (metrics + thresholds + alerts)
- CSV with per-row error breakdown

Default locations are auto-generated under:

- `backend/out/analysis/eta_concept_drift_<timestamp>.json`
- `backend/out/analysis/eta_concept_drift_rows_<timestamp>.csv`

## Metrics

- `mae_s`
- `mape_pct`
- `rmse_s`
- `max_abs_error_s`
- `count`

Alert flags:

- `mae_alert`
- `mape_alert`
- `any_alert`

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Performance Profiling Notes](performance-profiling-notes.md)

