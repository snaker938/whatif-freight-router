# ETA Concept Drift Checks

This project includes a lightweight script to compare predicted ETA against observed ETA and raise threshold-based alerts.

## Script

From `backend/`:

```powershell
uv run python scripts/check_eta_concept_drift.py --input-csv .\eta_observations.csv --mae-threshold-s 120 --mape-threshold-pct 10
```

## Required CSV Columns

- `predicted_eta_s`
- `observed_eta_s`

Optional:
- `trip_id`

Example:

```csv
trip_id,predicted_eta_s,observed_eta_s
A1,900,960
A2,1200,1100
```

## Output Artifacts

By default outputs are written under `backend/out/analysis`:

- `eta_concept_drift_<timestamp>.json`
- `eta_concept_drift_rows_<timestamp>.csv`

JSON output includes:

- `metrics.count`
- `metrics.mae_s`
- `metrics.mape_pct`
- `metrics.rmse_s`
- `metrics.max_abs_error_s`
- `alerts.mae_alert`
- `alerts.mape_alert`
- `alerts.any_alert`

## Alert Logic

- `mae_alert = mae_s > mae_threshold_s`
- `mape_alert = mape_pct > mape_threshold_pct`
- `any_alert = mae_alert OR mape_alert`

## Notes

- This script is an offline drift check over provided samples, not a live telemetry monitor.
- Thresholds should be tuned to the operating context and service-level expectations.
