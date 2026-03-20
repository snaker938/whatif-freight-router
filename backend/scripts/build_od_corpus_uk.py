from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.routing_graph import route_graph_od_feasibility
from app.settings import settings


@dataclass(frozen=True)
class UKBBox:
    south: float
    north: float
    west: float
    east: float


DISTANCE_BINS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 30.0, "0-30 km"),
    (30.0, 100.0, "30-100 km"),
    (100.0, 250.0, "100-250 km"),
    (250.0, None, "250+ km"),
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_bbox(raw: str) -> UKBBox:
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("bbox must contain four comma-separated floats: south,north,west,east")
    south, north, west, east = (float(part) for part in parts)
    if north <= south or east <= west:
        raise ValueError("bbox bounds are invalid")
    return UKBBox(south=south, north=north, west=west, east=east)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _distance_bin(distance_km: float) -> str:
    for lower, upper, label in DISTANCE_BINS:
        if distance_km < lower:
            continue
        if upper is None or distance_km < upper:
            return label
    return DISTANCE_BINS[-1][2]


def _bin_index(distance_km: float) -> int:
    label = _distance_bin(distance_km)
    for idx, (_, _, bin_label) in enumerate(DISTANCE_BINS):
        if bin_label == label:
            return idx
    return len(DISTANCE_BINS) - 1


def _split_evenly(total: int, bucket_count: int) -> list[int]:
    total = max(0, int(total))
    bucket_count = max(1, int(bucket_count))
    base = total // bucket_count
    remainder = total % bucket_count
    return [base + (1 if idx < remainder else 0) for idx in range(bucket_count)]


def _sample_candidate_pair(rng: random.Random, bbox: UKBBox) -> tuple[dict[str, float], dict[str, float]]:
    origin = {
        "lat": rng.uniform(bbox.south, bbox.north),
        "lon": rng.uniform(bbox.west, bbox.east),
    }
    destination = {
        "lat": rng.uniform(bbox.south, bbox.north),
        "lon": rng.uniform(bbox.west, bbox.east),
    }
    return origin, destination


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _row_hash(rows: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(_canonical_json(rows).encode("utf-8"))
    return hasher.hexdigest()


def _feasibility_result_to_row(
    *,
    od_id: str,
    sample_index: int,
    origin: dict[str, float],
    destination: dict[str, float],
    distance_km: float,
    feasible: bool,
    feasibility_result: dict[str, Any] | None,
    reason_code: str,
) -> dict[str, Any]:
    result = feasibility_result or {}
    row = {
        "od_id": od_id,
        "sample_index": int(sample_index),
        "origin_lat": round(float(origin["lat"]), 6),
        "origin_lon": round(float(origin["lon"]), 6),
        "destination_lat": round(float(destination["lat"]), 6),
        "destination_lon": round(float(destination["lon"]), 6),
        "straight_line_km": round(float(distance_km), 6),
        "distance_bin": _distance_bin(distance_km),
        "bin_index": _bin_index(distance_km),
        "accepted": bool(feasible),
        "reason_code": reason_code,
        "origin_node_id": str(result.get("origin_node_id", "")) if feasible else "",
        "destination_node_id": str(result.get("destination_node_id", "")) if feasible else "",
        "origin_nearest_distance_m": round(float(result.get("origin_nearest_distance_m", 0.0)), 3),
        "destination_nearest_distance_m": round(float(result.get("destination_nearest_distance_m", 0.0)), 3),
    }
    if feasible:
        row["route_graph_message"] = str(result.get("message", "ok"))
    else:
        row["route_graph_message"] = str(result.get("message", reason_code))
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "od_id",
        "sample_index",
        "origin_lat",
        "origin_lon",
        "destination_lat",
        "destination_lon",
        "straight_line_km",
        "distance_bin",
        "bin_index",
        "accepted",
        "reason_code",
        "route_graph_message",
        "origin_node_id",
        "destination_node_id",
        "origin_nearest_distance_m",
        "destination_nearest_distance_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_od_corpus(
    *,
    seed: int,
    pair_count: int,
    bbox: UKBBox,
    max_attempts: int,
    feasibility_fn: Callable[..., dict[str, Any]] = route_graph_od_feasibility,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    target_count = max(1, int(pair_count))
    bin_targets = _split_evenly(target_count, len(DISTANCE_BINS))
    accepted_by_bin = [0 for _ in DISTANCE_BINS]
    rows: list[dict[str, Any]] = []
    reject_stats: dict[str, int] = {}
    accepted_count = 0

    for sample_index in range(max(1, int(max_attempts))):
        if accepted_count >= target_count:
            break
        origin, destination = _sample_candidate_pair(rng, bbox)
        distance_km = _haversine_km(
            float(origin["lat"]),
            float(origin["lon"]),
            float(destination["lat"]),
            float(destination["lon"]),
        )
        bin_idx = _bin_index(distance_km)
        od_id = f"od-{sample_index:06d}"
        if accepted_by_bin[bin_idx] >= bin_targets[bin_idx]:
            reject_stats["bin_quota_full"] = reject_stats.get("bin_quota_full", 0) + 1
            continue

        result = feasibility_fn(
            origin_lat=float(origin["lat"]),
            origin_lon=float(origin["lon"]),
            destination_lat=float(destination["lat"]),
            destination_lon=float(destination["lon"]),
        )
        feasible = bool(result.get("ok"))
        reason_code = str(result.get("reason_code", "routing_graph_unavailable") if not feasible else "ok")
        if not feasible:
            reject_stats[reason_code] = reject_stats.get(reason_code, 0) + 1
            continue

        row = _feasibility_result_to_row(
            od_id=od_id,
            sample_index=sample_index,
            origin=origin,
            destination=destination,
            distance_km=distance_km,
            feasible=True,
            feasibility_result=result,
            reason_code="ok",
        )
        rows.append(row)
        accepted_by_bin[bin_idx] += 1
        accepted_count += 1

    complete = accepted_count >= target_count
    accept_rate = accepted_count / float(max(1, int(max_attempts)))
    rows_hash = _row_hash(rows)
    bin_distribution = {
        DISTANCE_BINS[idx][2]: int(accepted_by_bin[idx])
        for idx in range(len(DISTANCE_BINS))
    }
    summary = {
        "schema_version": "1.0.0",
        "created_at_utc": _utc_now_iso(),
        "seed": int(seed),
        "pair_count": int(pair_count),
        "target_count": target_count,
        "max_attempts": int(max_attempts),
        "bbox": {
            "south": bbox.south,
            "north": bbox.north,
            "west": bbox.west,
            "east": bbox.east,
        },
        "accepted_count": accepted_count,
        "rejected_count": int(max(0, int(max_attempts) - accepted_count)),
        "complete": complete,
        "accept_rate": round(float(accept_rate), 6),
        "accepted_by_bin": bin_distribution,
        "reject_stats": dict(sorted(reject_stats.items())),
        "corpus_hash": rows_hash,
        "distance_bins": [
            {"label": label, "lower_km": lower, "upper_km": upper}
            for lower, upper, label in DISTANCE_BINS
        ],
        "rows": rows,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic UK OD corpus using route graph feasibility."
    )
    parser.add_argument("--seed", type=int, default=20260212)
    parser.add_argument("--pair-count", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--bbox", default=str(settings.terrain_uk_bbox))
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def _default_output_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    corpus_dir = out_dir / "thesis"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now_compact()
    return (
        corpus_dir / f"od_corpus_uk_{stamp}.csv",
        corpus_dir / f"od_corpus_uk_{stamp}.json",
        corpus_dir / f"od_corpus_uk_{stamp}.summary.json",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    bbox = _parse_bbox(args.bbox)
    out_dir = Path("out").resolve()
    csv_default, json_default, summary_default = _default_output_paths(out_dir)
    csv_path = Path(args.output_csv).resolve() if args.output_csv else csv_default
    json_path = Path(args.output_json).resolve() if args.output_json else json_default
    summary_path = Path(args.summary_json).resolve() if args.summary_json else summary_default

    summary = build_od_corpus(
        seed=args.seed,
        pair_count=args.pair_count,
        bbox=bbox,
        max_attempts=args.max_attempts,
    )
    if not args.allow_partial and not summary["complete"]:
        raise RuntimeError(
            "Corpus builder did not reach the requested pair-count. "
            "Use --allow-partial if partial corpora are acceptable."
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, summary["rows"])
    json_path.write_text(json.dumps(summary["rows"], indent=2), encoding="utf-8")
    summary_payload = {key: value for key, value in summary.items() if key != "rows"}
    summary_payload["output_csv"] = str(csv_path)
    summary_payload["output_json"] = str(json_path)
    summary_payload["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
