from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import httpx
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeOSRM:
    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:
        origin_lat = float(kwargs["origin_lat"])
        dest_lat = float(kwargs["dest_lat"])
        distance_m = 70_000.0 + (abs(origin_lat - dest_lat) * 2_000.0)
        duration_s = 3_600.0 + (abs(origin_lat - dest_lat) * 200.0)
        return [
            {
                "distance": distance_m,
                "duration": duration_s,
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


def _parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("at least one seed is required")
    return out


def generate_pairs(pair_count: int, seed: int) -> list[dict[str, dict[str, float]]]:
    rng = random.Random(seed)
    pairs: list[dict[str, dict[str, float]]] = []
    for _ in range(max(1, pair_count)):
        origin = {"lat": rng.uniform(50.5, 54.5), "lon": rng.uniform(-3.0, 0.5)}
        destination = {"lat": rng.uniform(50.5, 54.5), "lon": rng.uniform(-3.0, 0.5)}
        pairs.append({"origin": origin, "destination": destination})
    return pairs


def _build_payload(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    return {
        "pairs": generate_pairs(args.pair_count, seed),
        "vehicle_type": args.vehicle_type,
        "scenario_mode": args.scenario_mode,
        "max_alternatives": args.max_alternatives,
        "seed": seed,
        "toggles": {"robustness_analysis": True},
        "model_version": args.model_version,
    }


def _summarise_batch(batch: dict[str, Any], seed: int) -> dict[str, Any]:
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
        "seed": seed,
        "run_id": batch.get("run_id", ""),
        "error_count": error_count,
        "avg_duration_s": round(statistics.fmean(durations), 6) if durations else 0.0,
        "avg_monetary_cost": round(statistics.fmean(moneys), 6) if moneys else 0.0,
        "avg_emissions_kg": round(statistics.fmean(emissions), 6) if emissions else 0.0,
    }


def _run_inprocess(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    from app.main import app, osrm_client
    from app.settings import settings

    old_out_dir = settings.out_dir
    settings.out_dir = str(Path(args.out_dir).resolve())
    payload = _build_payload(args, seed)
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


def _run_live(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    payload = _build_payload(args, seed)
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{args.backend_url.rstrip('/')}/batch/pareto", json=payload)
    resp.raise_for_status()
    return resp.json()


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    def _mean_std(key: str) -> tuple[float, float]:
        values = [float(row[key]) for row in rows]
        mean = statistics.fmean(values) if values else 0.0
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        return round(mean, 6), round(stddev, 6)

    duration_mean, duration_std = _mean_std("avg_duration_s")
    cost_mean, cost_std = _mean_std("avg_monetary_cost")
    emissions_mean, emissions_std = _mean_std("avg_emissions_kg")

    return {
        "avg_duration_s_mean": duration_mean,
        "avg_duration_s_stddev": duration_std,
        "avg_monetary_cost_mean": cost_mean,
        "avg_monetary_cost_stddev": cost_std,
        "avg_emissions_kg_mean": emissions_mean,
        "avg_emissions_kg_stddev": emissions_std,
    }


def _default_paths(out_dir: Path) -> tuple[Path, Path]:
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now_compact()
    return (
        analysis_dir / f"robustness_{stamp}.json",
        analysis_dir / f"robustness_{stamp}.csv",
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

    rows = payload["runs"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "run_id",
                "error_count",
                "avg_duration_s",
                "avg_monetary_cost",
                "avg_emissions_kg",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path


def run_robustness(args: argparse.Namespace) -> dict[str, Any]:
    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        data = _run_inprocess(args, seed) if args.mode == "inprocess-fake" else _run_live(args, seed)
        rows.append(_summarise_batch(data, seed))

    payload: dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "mode": args.mode,
        "pair_count": args.pair_count,
        "seeds": seeds,
        "runs": rows,
        "aggregate": _aggregate_rows(rows),
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
    parser = argparse.ArgumentParser(
        description="Run robustness analysis over multiple seeds for batch Pareto."
    )
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--pair-count", type=int, default=25)
    parser.add_argument("--max-alternatives", type=int, default=3)
    parser.add_argument("--mode", choices=("inprocess-fake", "live"), default="inprocess-fake")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--vehicle-type", default="rigid_hgv")
    parser.add_argument("--scenario-mode", default="no_sharing")
    parser.add_argument("--model-version", default="robustness-v1")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = run_robustness(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
