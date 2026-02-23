from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _interpolate_sparse(points: list[tuple[int, float]], default: float = 1.0) -> list[float]:
    if not points:
        return [default for _ in range(1440)]
    points = sorted((max(0, min(1439, int(m))), float(v)) for m, v in points)
    dense = [default for _ in range(1440)]
    first_m, first_v = points[0]
    for i in range(first_m + 1):
        dense[i] = first_v
    for idx in range(1, len(points)):
        pm, pv = points[idx - 1]
        nm, nv = points[idx]
        span = max(1, nm - pm)
        for i in range(pm, nm + 1):
            t = (i - pm) / span
            dense[i] = pv + ((nv - pv) * t)
    last_m, last_v = points[-1]
    for i in range(last_m, 1440):
        dense[i] = last_v
    return dense


def _read_sparse_profile(path: Path) -> dict[str, list[float]]:
    weekday_points: list[tuple[int, float]] = []
    weekend_points: list[tuple[int, float]] = []
    holiday_points: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            minute = int(float(row.get("minute", 0.0)))
            weekday_points.append((minute, float(row.get("weekday", 1.0))))
            weekend_points.append((minute, float(row.get("weekend", 1.0))))
            holiday_points.append((minute, float(row.get("holiday", row.get("weekend", 1.0)))))
    return {
        "weekday": _interpolate_sparse(weekday_points),
        "weekend": _interpolate_sparse(weekend_points),
        "holiday": _interpolate_sparse(holiday_points),
    }


def _scale_profile(series: list[float], scale: float, *, floor: float = 0.72, cap: float = 1.65) -> list[float]:
    return [max(floor, min(cap, float(v) * float(scale))) for v in series]


def _envelope_for_series(
    series: list[float],
    *,
    spread: float = 0.08,
    low_floor: float = 0.55,
    high_cap: float = 2.60,
) -> dict[str, list[float]]:
    low: list[float] = []
    high: list[float] = []
    for value in series:
        v = float(value)
        band = max(0.015, abs(v - 1.0) * spread)
        low.append(max(low_floor, v - band))
        high.append(min(high_cap, v + band))
    return {"low": low, "high": high}


def _build_from_empirical_counts(
    *,
    counts_csv: Path,
) -> tuple[
    dict[str, dict[str, dict[str, list[float]]]],
    dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    str | None,
]:
    # Expected columns:
    # region,road_bucket,day_kind,minute,multiplier,as_of_utc(optional)
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    as_of_candidates: list[str] = []
    with counts_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = str(row.get("region", "uk_default")).strip() or "uk_default"
            road_bucket = str(row.get("road_bucket", "mixed")).strip() or "mixed"
            day_kind = str(row.get("day_kind", "weekday")).strip() or "weekday"
            day_kind = day_kind if day_kind in {"weekday", "weekend", "holiday"} else "weekday"
            minute = int(float(row.get("minute", 0.0)))
            multiplier = float(row.get("multiplier", 1.0))
            grouped[(region, road_bucket, day_kind, max(0, min(1439, minute)))].append(multiplier)
            as_of = str(row.get("as_of_utc", "")).strip()
            if as_of:
                as_of_candidates.append(as_of)

    profiles: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    observed_stdev: dict[tuple[str, str, str], list[float]] = {}
    context_means: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    context_stdev: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for (region, road_bucket, day_kind, minute), values in grouped.items():
        if not values:
            continue
        mean_val = sum(values) / len(values)
        if len(values) > 1:
            var = sum((value - mean_val) ** 2 for value in values) / max(1, len(values) - 1)
            std_val = var ** 0.5
        else:
            std_val = max(0.01, abs(mean_val - 1.0) * 0.08)
        ctx = (region, road_bucket, day_kind)
        context_means[ctx].append((minute, mean_val))
        context_stdev[ctx].append((minute, std_val))
    for (region, road_bucket, day_kind), points in context_means.items():
        profiles[region][road_bucket][day_kind] = _interpolate_sparse(points)
        stdev_series = _interpolate_sparse(context_stdev[(region, road_bucket, day_kind)], default=0.04)
        observed_stdev[(region, road_bucket, day_kind)] = stdev_series

    # Fill any missing day kinds per region/road bucket using weekday defaults.
    for _region, road_map in profiles.items():
        for _road_bucket, day_map in road_map.items():
            weekday = day_map.get("weekday", [1.0 for _ in range(1440)])
            weekend = day_map.get("weekend", weekday)
            holiday = day_map.get("holiday", weekend)
            day_map["weekday"] = weekday
            day_map["weekend"] = weekend
            day_map["holiday"] = holiday
    envelopes: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for region, road_map in profiles.items():
        env_road: dict[str, dict[str, dict[str, list[float]]]] = {}
        for road_bucket, day_map in road_map.items():
            env_day: dict[str, dict[str, list[float]]] = {}
            for day_kind, values in day_map.items():
                stdev_series = observed_stdev.get((region, road_bucket, day_kind), [0.04 for _ in range(1440)])
                low: list[float] = []
                high: list[float] = []
                for value, sigma in zip(values, stdev_series, strict=True):
                    band = max(0.01, min(0.65, 1.64 * max(0.005, float(sigma))))
                    low.append(max(0.50, float(value) - band))
                    high.append(min(2.60, float(value) + band))
                env_day[day_kind] = {"low": low, "high": high}
            env_road[road_bucket] = env_day
        envelopes[region] = env_road

    as_of_utc = max(as_of_candidates) if as_of_candidates else None
    # mypy-style dict conversion for JSON.
    out_profiles: dict[str, dict[str, dict[str, list[float]]]] = {}
    for region, road_map in profiles.items():
        out_profiles[region] = {road: dict(day_map) for road, day_map in road_map.items()}
    return out_profiles, envelopes, as_of_utc


def _build_synthetic_from_sparse(
    *,
    sparse_csv: Path,
) -> tuple[
    dict[str, dict[str, dict[str, list[float]]]],
    dict[str, dict[str, dict[str, dict[str, list[float]]]]],
]:
    base = _read_sparse_profile(sparse_csv)

    region_scales = {
        "uk_default": 1.00,
        "london_southeast": 1.08,
        "midlands": 1.03,
        "north_england": 1.02,
        "scotland": 1.01,
        "wales_west": 1.00,
        "south_england": 1.01,
    }
    road_scales = {
        "motorway_dominant": 0.93,
        "motorway_heavy": 0.95,
        "trunk_dominant": 0.98,
        "trunk_heavy": 1.00,
        "primary_heavy": 1.03,
        "urban_local_heavy": 1.08,
        "mixed": 1.04,
    }

    profiles: dict[str, dict[str, dict[str, list[float]]]] = {}
    for region, region_scale in region_scales.items():
        road_map: dict[str, dict[str, list[float]]] = {}
        for road_bucket, road_scale in road_scales.items():
            day_map: dict[str, list[float]] = {}
            for day_kind, values in base.items():
                day_adj = 0.98 if day_kind == "weekend" else (0.99 if day_kind == "holiday" else 1.0)
                day_map[day_kind] = _scale_profile(
                    values,
                    region_scale * road_scale * day_adj,
                )
            road_map[road_bucket] = day_map
        profiles[region] = road_map
    envelopes: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for region, road_map in profiles.items():
        env_road: dict[str, dict[str, dict[str, list[float]]]] = {}
        for road_bucket, day_map in road_map.items():
            env_day: dict[str, dict[str, list[float]]] = {}
            for day_kind, values in day_map.items():
                env_day[day_kind] = _envelope_for_series(values, spread=0.10)
            env_road[road_bucket] = env_day
        envelopes[region] = env_road
    return profiles, envelopes


def _ci_strict_mode() -> bool:
    return str(os.environ.get("CI", "")).strip().lower() in {"1", "true", "yes"}


def _test_only_synthetic_allowed() -> bool:
    return str(os.environ.get("TEST_ONLY_SYNTHETIC", "")).strip().lower() in {"1", "true", "yes"}


def build(
    *,
    sparse_csv: Path,
    output_json: Path,
    counts_csv: Path | None = None,
    allow_synthetic: bool = False,
) -> None:
    if allow_synthetic and not _test_only_synthetic_allowed():
        raise RuntimeError(
            "Synthetic departure profile generation is disabled in strict runtime. "
            "Set TEST_ONLY_SYNTHETIC=1 for explicit test-only generation."
        )
    if allow_synthetic and _ci_strict_mode():
        raise RuntimeError("Synthetic departure profile generation is disabled in CI strict mode.")
    as_of_utc = None
    source: str | list[str]
    version = "uk-v3-contextual-synthetic"
    calibration_basis = "synthetic"
    envelopes: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    if counts_csv is not None and counts_csv.exists():
        profiles, envelopes, as_of_utc = _build_from_empirical_counts(counts_csv=counts_csv)
        source = [str(sparse_csv), str(counts_csv)]
        version = "uk-v4-contextual-empirical"
        calibration_basis = "empirical"
    else:
        if not allow_synthetic:
            raise FileNotFoundError(
                "Empirical counts CSV is required. Pass --allow-synthetic to build heuristic profiles explicitly."
            )
        profiles, envelopes = _build_synthetic_from_sparse(sparse_csv=sparse_csv)
        source = str(sparse_csv)

    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "source": source,
        "version": version,
        "generated_at_utc": now_iso,
        "as_of_utc": as_of_utc or now_iso,
        "calibration_basis": calibration_basis,
        "profiles": profiles,
        "envelopes": envelopes,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UK contextual departure profiles.")
    parser.add_argument(
        "--sparse-csv",
        type=Path,
        default=Path("backend/assets/uk/departure_profile_uk.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/out/model_assets/departure_profiles_uk.json"),
    )
    parser.add_argument(
        "--counts-csv",
        type=Path,
        default=None,
        help="Optional empirical counts CSV (region,road_bucket,day_kind,minute,multiplier).",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow fallback synthetic scaling when empirical counts are unavailable.",
    )
    args = parser.parse_args()
    build(
        sparse_csv=args.sparse_csv,
        output_json=args.output,
        counts_csv=args.counts_csv,
        allow_synthetic=bool(args.allow_synthetic),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
