from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def load_eta_rows(csv_path: str) -> list[dict[str, Any]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"predicted_eta_s", "observed_eta_s"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must include predicted_eta_s and observed_eta_s columns")

        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(reader, start=2):
            try:
                predicted = float(row["predicted_eta_s"])
                observed = float(row["observed_eta_s"])
            except Exception as e:
                raise ValueError(f"invalid numeric ETA value on row {idx}") from e

            abs_error = abs(observed - predicted)
            denom = max(abs(observed), 1e-6)
            pct_error = (abs_error / denom) * 100.0
            rows.append(
                {
                    "row_index": idx,
                    "trip_id": row.get("trip_id", ""),
                    "predicted_eta_s": predicted,
                    "observed_eta_s": observed,
                    "abs_error_s": abs_error,
                    "pct_error": pct_error,
                }
            )
    if not rows:
        raise ValueError("CSV contains no rows")
    return rows


def compute_drift_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("rows must not be empty")

    errors = [float(row["abs_error_s"]) for row in rows]
    pct_errors = [float(row["pct_error"]) for row in rows]
    sq_errors = [err * err for err in errors]

    count = float(len(rows))
    mae = sum(errors) / count
    mape = sum(pct_errors) / count
    rmse = math.sqrt(sum(sq_errors) / count)
    max_abs_error = max(errors)

    return {
        "count": int(count),
        "mae_s": round(mae, 6),
        "mape_pct": round(mape, 6),
        "rmse_s": round(rmse, 6),
        "max_abs_error_s": round(max_abs_error, 6),
    }


def _default_paths(out_dir: Path) -> tuple[Path, Path]:
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now_compact()
    return (
        analysis_dir / f"eta_concept_drift_{stamp}.json",
        analysis_dir / f"eta_concept_drift_rows_{stamp}.csv",
    )


def run_drift_check(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eta_rows(args.input_csv)
    metrics = compute_drift_metrics(rows)

    mae_threshold = float(args.mae_threshold_s)
    mape_threshold = float(args.mape_threshold_pct)
    alerts = {
        "mae_alert": metrics["mae_s"] > mae_threshold,
        "mape_alert": metrics["mape_pct"] > mape_threshold,
    }
    alerts["any_alert"] = alerts["mae_alert"] or alerts["mape_alert"]

    payload: dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "input_csv": str(Path(args.input_csv).resolve()),
        "thresholds": {
            "mae_threshold_s": mae_threshold,
            "mape_threshold_pct": mape_threshold,
        },
        "metrics": metrics,
        "alerts": alerts,
    }

    default_json, default_csv = _default_paths(out_dir)
    json_path = Path(args.json_output) if args.json_output else default_json
    csv_path = Path(args.csv_output) if args.csv_output else default_csv
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "trip_id",
                "predicted_eta_s",
                "observed_eta_s",
                "abs_error_s",
                "pct_error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload["json_output"] = str(json_path)
    payload["csv_output"] = str(csv_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check ETA concept drift from predicted vs observed ETA values."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--mae-threshold-s", type=float, default=120.0)
    parser.add_argument("--mape-threshold-pct", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = run_drift_check(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
