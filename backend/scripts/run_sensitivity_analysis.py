from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeOSRM:
    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:
        origin_lat = float(kwargs["origin_lat"])
        dest_lat = float(kwargs["dest_lat"])
        distance_m = 65_000.0 + (abs(origin_lat - dest_lat) * 2_200.0)
        duration_s = 3_300.0 + (abs(origin_lat - dest_lat) * 220.0)
        return [
            {
                "distance": distance_m,
                "duration": duration_s,
                "contains_toll": True,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
                },
                "legs": [
                    {
                        "annotation": {
                            "distance": [distance_m / 2.0, distance_m / 2.0],
                            "duration": [duration_s / 2.0, duration_s / 2.0],
                        }
                    }
                ],
            }
        ]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_floats(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("at least one numeric value is required")
    return out


def generate_pairs(pair_count: int, seed: int) -> list[dict[str, dict[str, float]]]:
    rng = random.Random(seed)
    pairs: list[dict[str, dict[str, float]]] = []
    for _ in range(max(1, pair_count)):
        origin = {"lat": rng.uniform(50.5, 54.5), "lon": rng.uniform(-3.0, 0.5)}
        destination = {"lat": rng.uniform(50.5, 54.5), "lon": rng.uniform(-3.0, 0.5)}
        pairs.append({"origin": origin, "destination": destination})
    return pairs


def _summarise_batch(batch: dict[str, Any]) -> dict[str, float]:
    results = batch.get("results", [])
    durations: list[float] = []
    moneys: list[float] = []
    emissions: list[float] = []
    error_count = 0

    for item in results:
        if item.get("error"):
            error_count += 1
            continue
        routes = item.get("routes") or []
        if not routes:
            continue
        best = min(routes, key=lambda r: float(r["metrics"]["duration_s"]))
        metrics = best["metrics"]
        durations.append(float(metrics["duration_s"]))
        moneys.append(float(metrics["monetary_cost"]))
        emissions.append(float(metrics["emissions_kg"]))

    return {
        "avg_duration_s": round(statistics.fmean(durations), 6) if durations else 0.0,
        "avg_monetary_cost": round(statistics.fmean(moneys), 6) if moneys else 0.0,
        "avg_emissions_kg": round(statistics.fmean(emissions), 6) if emissions else 0.0,
        "error_count": float(error_count),
    }


def _execute_batch(
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if args.mode == "inprocess-fake":
        from app.main import app, osrm_client
        from app.settings import settings

        old_out_dir = settings.out_dir
        settings.out_dir = str(Path(args.out_dir).resolve())
        app.dependency_overrides[osrm_client] = lambda: _FakeOSRM()
        try:
            with TestClient(app) as client:
                resp = client.post("/batch/pareto", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"batch run failed: {resp.status_code} {resp.text}")
            return resp.json()
        finally:
            app.dependency_overrides.clear()
            settings.out_dir = old_out_dir

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{args.backend_url.rstrip('/')}/batch/pareto", json=payload)
    resp.raise_for_status()
    return resp.json()


def _build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    fuel_values = _parse_floats(args.fuel_multipliers)
    carbon_values = _parse_floats(args.carbon_prices)
    toll_values = _parse_floats(args.toll_costs)

    baseline = {
        "name": "baseline",
        "cost_toggles": {
            "use_tolls": True,
            "fuel_price_multiplier": 1.0,
            "carbon_price_per_kg": 0.0,
            "toll_cost_per_km": 0.0,
        },
    }

    cases: list[dict[str, Any]] = [baseline]
    for value in fuel_values:
        if abs(value - 1.0) < 1e-12:
            continue
        cases.append(
            {
                "name": f"fuel_{value:g}",
                "cost_toggles": {**baseline["cost_toggles"], "fuel_price_multiplier": value},
            }
        )
    for value in carbon_values:
        if abs(value) < 1e-12:
            continue
        cases.append(
            {
                "name": f"carbon_{value:g}",
                "cost_toggles": {**baseline["cost_toggles"], "carbon_price_per_kg": value},
            }
        )
    for value in toll_values:
        if abs(value) < 1e-12:
            continue
        cases.append(
            {
                "name": f"toll_{value:g}",
                "cost_toggles": {**baseline["cost_toggles"], "toll_cost_per_km": value},
            }
        )
    if args.include_no_tolls:
        cases.append(
            {
                "name": "no_tolls",
                "cost_toggles": {**baseline["cost_toggles"], "use_tolls": False, "toll_cost_per_km": 0.5},
            }
        )
    return cases


def _default_paths(out_dir: Path) -> tuple[Path, Path]:
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now_compact()
    return (
        analysis_dir / f"sensitivity_{stamp}.json",
        analysis_dir / f"sensitivity_{stamp}.csv",
    )


def _write_outputs(
    *,
    out_dir: Path,
    payload: dict[str, Any],
    json_output: str | None,
    csv_output: str | None,
) -> tuple[Path, Path]:
    default_json, default_csv = _default_paths(out_dir)
    json_path = Path(json_output) if json_output else default_json
    csv_path = Path(csv_output) if csv_output else default_csv

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "run_id",
                "avg_duration_s",
                "avg_monetary_cost",
                "avg_emissions_kg",
                "delta_duration_s",
                "delta_monetary_cost",
                "delta_emissions_kg",
                "error_count",
                "cost_toggles",
            ],
        )
        writer.writeheader()
        for row in payload["cases"]:
            writer.writerow(row)

    return json_path, csv_path


def run_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = generate_pairs(args.pair_count, args.seed)
    cases = _build_cases(args)

    rows: list[dict[str, Any]] = []
    baseline_metrics: dict[str, float] | None = None
    for case in cases:
        payload = {
            "pairs": pairs,
            "vehicle_type": args.vehicle_type,
            "scenario_mode": args.scenario_mode,
            "max_alternatives": args.max_alternatives,
            "cost_toggles": case["cost_toggles"],
            "seed": args.seed,
            "toggles": {"sensitivity_analysis": True, "case": case["name"]},
            "model_version": args.model_version,
        }
        data = _execute_batch(args, payload)
        metrics = _summarise_batch(data)
        if baseline_metrics is None:
            baseline_metrics = metrics

        row = {
            "case": case["name"],
            "run_id": data.get("run_id", ""),
            "avg_duration_s": metrics["avg_duration_s"],
            "avg_monetary_cost": metrics["avg_monetary_cost"],
            "avg_emissions_kg": metrics["avg_emissions_kg"],
            "delta_duration_s": metrics["avg_duration_s"] - baseline_metrics["avg_duration_s"],
            "delta_monetary_cost": metrics["avg_monetary_cost"]
            - baseline_metrics["avg_monetary_cost"],
            "delta_emissions_kg": metrics["avg_emissions_kg"] - baseline_metrics["avg_emissions_kg"],
            "error_count": int(metrics["error_count"]),
            "cost_toggles": case["cost_toggles"],
        }
        rows.append(row)

    payload: dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "mode": args.mode,
        "pair_count": args.pair_count,
        "seed": args.seed,
        "cases": rows,
    }
    json_path, csv_path = _write_outputs(
        out_dir=out_dir,
        payload=payload,
        json_output=args.json_output,
        csv_output=args.csv_output,
    )
    payload["json_output"] = str(json_path)
    payload["csv_output"] = str(csv_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-factor sensitivity analysis for batch Pareto.")
    parser.add_argument("--pair-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260212)
    parser.add_argument("--max-alternatives", type=int, default=3)
    parser.add_argument("--mode", choices=("inprocess-fake", "live"), default="inprocess-fake")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--vehicle-type", default="rigid_hgv")
    parser.add_argument("--scenario-mode", default="no_sharing")
    parser.add_argument("--model-version", default="sensitivity-v1")
    parser.add_argument("--fuel-multipliers", default="0.9,1.0,1.1")
    parser.add_argument("--carbon-prices", default="0.0,0.1,0.2")
    parser.add_argument("--toll-costs", default="0.0,0.25,0.5")
    parser.add_argument("--include-no-tolls", action="store_true")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = run_sensitivity(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
