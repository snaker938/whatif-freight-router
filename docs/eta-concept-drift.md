# ETA Concept Drift Checks

Last Updated: 2026-03-31  
Applies To: `backend/scripts/check_eta_concept_drift.py`

## Purpose

Checks drift between predicted and observed ETAs and emits machine-readable outputs for offline monitoring.

## Scope

- this is an offline analysis helper
- it is not part of the route API response contract
- it is not itself a thesis proof artifact family
- use it as supporting monitoring or investigative analysis alongside the main evaluation/reporting pipeline

## Input Requirements

Required CSV columns:

- `predicted_eta_s`
- `observed_eta_s`

Optional:

- `trip_id`

## Commands

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

The script writes:

- JSON summary with metrics, thresholds, and alerts
- CSV with per-row error breakdown

The JSON summary includes:

- computed drift metrics
- configured threshold values
- alert booleans
- resolved output paths

Default locations are auto-generated under:

- `backend/out/analysis/eta_concept_drift_<timestamp>.json`
- `backend/out/analysis/eta_concept_drift_rows_<timestamp>.csv`

If `--json-output` or `--csv-output` is supplied, those explicit paths override the defaults.

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
