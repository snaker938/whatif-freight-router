# ETA Concept Drift Checks

Last Updated: 2026-04-09
Applies To: `backend/scripts/check_eta_concept_drift.py`

## Purpose

Checks drift between predicted and observed ETAs and emits machine-readable outputs for monitoring or offline audit.

The script is intentionally simple and deterministic: it does not call the route engine. It scores an external CSV of prediction/observation pairs.

## Input Requirements

Required CSV columns:

- `predicted_eta_s`
- `observed_eta_s`

Optional:

- `trip_id`

Each row is converted into:

- absolute error in seconds
- percent error against observed ETA
- a stable row index for later CSV/JSON reconciliation

## Default Thresholds

`backend/scripts/check_eta_concept_drift.py` currently defaults to:

- `--mae-threshold-s 120`
- `--mape-threshold-pct 10`
- `--out-dir out`

Those thresholds are the current documented alert boundary, not a checked thesis result.

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

The JSON summary currently includes:

- `timestamp`
- `input_csv`
- `thresholds.mae_threshold_s`
- `thresholds.mape_threshold_pct`
- `metrics.count`
- `metrics.mae_s`
- `metrics.mape_pct`
- `metrics.rmse_s`
- `metrics.max_abs_error_s`
- `alerts.mae_alert`
- `alerts.mape_alert`
- `alerts.any_alert`

The CSV output contains one row per observation with:

- `row_index`
- `trip_id`
- `predicted_eta_s`
- `observed_eta_s`
- `abs_error_s`
- `pct_error`

Default output locations are auto-generated under:

- `backend/out/analysis/eta_concept_drift_<timestamp>.json`
- `backend/out/analysis/eta_concept_drift_rows_<timestamp>.csv`

As of `2026-04-09`, there is no checked `backend/out/analysis` directory in the repo, so there is no newer saved drift run to summarize here.

## Interpretation Notes

- `mae_s` is the operationally simplest drift metric and the one most likely to map to user-visible ETA trust.
- `mape_pct` is useful when comparing mixed trip lengths, but it is sensitive to very short observed trips.
- `rmse_s` is more punitive on large misses and is useful when tail failures matter.
- `max_abs_error_s` is important for fail-closed or audit-heavy lanes because one severe miss can matter more than a stable mean.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Reproducibility Capsule](reproducibility-capsule.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Performance Profiling Notes](performance-profiling-notes.md)
