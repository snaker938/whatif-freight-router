from __future__ import annotations

import argparse
import json
import random
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import httpx
from fastapi.testclient import TestClient


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def generate_pairs(pair_count: int, seed: int) -> list[dict[str, dict[str, float]]]:
    rng = random.Random(seed)
    pairs: list[dict[str, dict[str, float]]] = []
    for _ in range(max(1, pair_count)):
        origin = {"lat": rng.uniform(50.5, 54.5), "lon": rng.uniform(-3.0, 0.5)}
        destination = {"lat": rng.uniform(50.5, 54.5), "lon": rng.uniform(-3.0, 0.5)}
        pairs.append({"origin": origin, "destination": destination})
    return pairs


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


def _default_output_path(out_dir: Path) -> Path:
    benchmark_dir = out_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"batch_pareto_benchmark_{_utc_now_compact()}.json"


def _write_record(record: dict[str, Any], out_dir: Path, output: str | None) -> Path:
    path = Path(output) if output else _default_output_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pairs": generate_pairs(args.pair_count, args.seed),
        "vehicle_type": args.vehicle_type,
        "scenario_mode": args.scenario_mode,
        "max_alternatives": args.max_alternatives,
        "seed": args.seed,
        "toggles": {"benchmark_mode": args.mode},
        "model_version": args.model_version,
    }


def _run_inprocess_fake(args: argparse.Namespace) -> dict[str, Any]:
    from app.main import app, osrm_client
    from app.settings import settings

    old_out_dir = settings.out_dir
    settings.out_dir = str(Path(args.out_dir).resolve())
    payload = _build_payload(args)

    tracemalloc.start()
    t0 = perf_counter()
    app.dependency_overrides[osrm_client] = lambda: _FakeOSRM()
    try:
        with TestClient(app) as client:
            resp = client.post("/batch/pareto", json=payload)
        duration_ms = (perf_counter() - t0) * 1000.0
        _current, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        app.dependency_overrides.clear()
        settings.out_dir = old_out_dir

    if resp.status_code != 200:
        raise RuntimeError(f"benchmark request failed: status={resp.status_code} body={resp.text}")

    data = resp.json()
    error_count = sum(1 for item in data["results"] if item.get("error"))
    return {
        "timestamp": _utc_now_iso(),
        "mode": "inprocess-fake",
        "pair_count": args.pair_count,
        "duration_ms": round(duration_ms, 3),
        "peak_memory_bytes": int(peak_bytes),
        "error_count": int(error_count),
        "run_id": data["run_id"],
    }


def _run_live_backend(args: argparse.Namespace) -> dict[str, Any]:
    payload = _build_payload(args)
    url = args.backend_url.rstrip("/")

    tracemalloc.start()
    t0 = perf_counter()
    with httpx.Client(timeout=90.0) as client:
        resp = client.post(f"{url}/batch/pareto", json=payload)
    duration_ms = (perf_counter() - t0) * 1000.0
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    resp.raise_for_status()
    data = resp.json()
    error_count = sum(1 for item in data["results"] if item.get("error"))
    return {
        "timestamp": _utc_now_iso(),
        "mode": "live",
        "pair_count": args.pair_count,
        "duration_ms": round(duration_ms, 3),
        "peak_memory_bytes": int(peak_bytes),
        "error_count": int(error_count),
        "run_id": data["run_id"],
        "backend_url": url,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "inprocess-fake":
        record = _run_inprocess_fake(args)
    else:
        record = _run_live_backend(args)

    path = _write_record(record, out_dir=out_dir, output=args.output)
    record["log_path"] = str(path)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark batch pareto performance with runtime/resource logs."
    )
    parser.add_argument("--pair-count", type=int, default=100)
    parser.add_argument("--max-alternatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260212)
    parser.add_argument(
        "--mode",
        choices=("inprocess-fake", "live"),
        default="inprocess-fake",
    )
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--output", default=None)
    parser.add_argument("--vehicle-type", default="rigid_hgv")
    parser.add_argument("--scenario-mode", default="no_sharing")
    parser.add_argument("--model-version", default="benchmark-v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    record = run_benchmark(args)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
