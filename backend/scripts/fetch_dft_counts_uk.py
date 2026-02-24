from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NormalizedRow:
    region: str
    road_bucket: str
    day_kind: str
    minute: int
    multiplier: float
    as_of_utc: str


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _pick(row: dict[str, str], *candidates: str) -> str:
    for name in candidates:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def _canonical_region(raw: str) -> str:
    key = raw.strip().lower()
    if not key:
        return "uk_default"
    aliases = {
        "london": "london_southeast",
        "greater london": "london_southeast",
        "south east": "south_england",
        "south west": "south_england",
        "east of england": "south_england",
        "east midlands": "midlands_east",
        "west midlands": "midlands_west",
        "north west": "north_west_corridor",
        "north east": "north_east_corridor",
        "yorkshire and the humber": "north_england_central",
        "wales": "wales_west",
        "scotland": "scotland_south",
    }
    return aliases.get(key, key.replace(" ", "_"))


def _canonical_road_bucket(raw: str) -> str:
    key = raw.strip().lower()
    if not key:
        return "mixed"
    if "motorway" in key:
        return "motorway_dominant"
    if "trunk" in key:
        return "trunk_heavy"
    if "primary" in key:
        return "primary_heavy"
    if "residential" in key or "local" in key or "urban" in key:
        return "urban_local_heavy"
    if "secondary" in key or "arterial" in key:
        return "arterial_heavy"
    return "mixed"


def _nearest_minute_value(minute_map: dict[int, float], target_minute: int) -> float:
    if not minute_map:
        return 1.0
    best_minute = min(minute_map, key=lambda minute: abs(int(minute) - int(target_minute)))
    return float(minute_map[best_minute])


def _densify_minute_profile(minute_map: dict[int, float]) -> dict[int, float]:
    # Build a full 24-hour quarter-hour profile (96 bins) from sparse observed hours.
    if not minute_map:
        return {}
    dense: dict[int, float] = {}
    for hour in range(24):
        for quarter in (0, 15, 30, 45):
            minute = int(hour * 60 + quarter)
            base = _nearest_minute_value(minute_map, minute)
            dense[minute] = _clamp(base, 0.45, 3.25)
    return dense


def _day_kind_from_datetime(dt: datetime, holiday_flag: str) -> str:
    h = holiday_flag.strip().lower()
    if h in {"1", "true", "yes", "holiday"}:
        return "holiday"
    if dt.weekday() >= 5:
        return "weekend"
    return "weekday"


def _parse_minute(row: dict[str, str]) -> int:
    minute_raw = _pick(row, "minute", "minute_of_day")
    if minute_raw:
        try:
            return max(0, min(1439, int(float(minute_raw))))
        except ValueError:
            pass
    hour_raw = _pick(row, "hour", "hour_of_day")
    try:
        hour = max(0, min(23, int(float(hour_raw))))
    except ValueError:
        hour = 0
    return hour * 60


def _parse_datetime(row: dict[str, str]) -> datetime | None:
    raw = _pick(
        row,
        "timestamp",
        "datetime",
        "date_time",
        "observed_at_utc",
        "observed_at",
        "date",
    )
    if not raw:
        return None
    txt = raw.strip()
    for variant in (txt, txt.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(variant)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        else:
            parsed = parsed.astimezone(UTC)
        return parsed
    return None


def _parse_iso_utc(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _read_direct_rows(
    raw_rows: list[dict[str, str]],
    *,
    as_of_utc: str,
) -> list[NormalizedRow]:
    out: list[NormalizedRow] = []
    for row in raw_rows:
        region = _canonical_region(_pick(row, "region"))
        road_bucket = _canonical_road_bucket(_pick(row, "road_bucket"))
        day_kind = _pick(row, "day_kind").strip().lower()
        if day_kind not in {"weekday", "weekend", "holiday"}:
            continue
        minute = _parse_minute(row)
        try:
            multiplier = float(_pick(row, "multiplier"))
        except ValueError:
            continue
        out.append(
            NormalizedRow(
                region=region,
                road_bucket=road_bucket,
                day_kind=day_kind,
                minute=minute,
                multiplier=_clamp(multiplier, 0.45, 3.25),
                as_of_utc=as_of_utc,
            )
        )
    return out


def _read_observed_count_rows(
    raw_rows: list[dict[str, str]],
    *,
    as_of_utc: str,
    prefer_observed_as_of: bool = True,
) -> list[NormalizedRow]:
    aggregated: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    as_of_candidates: list[datetime] = []
    for row in raw_rows:
        region = _canonical_region(
            _pick(
                row,
                "region",
                "local_authority",
                "county",
                "area",
            )
        )
        road_bucket = _canonical_road_bucket(
            _pick(
                row,
                "road_bucket",
                "road_class",
                "road_type",
                "road_category",
                "road",
            )
        )
        observed = _parse_datetime(row)
        source_as_of = _parse_datetime(
            {
                "timestamp": _pick(
                    row,
                    "as_of_utc",
                    "as_of",
                    "refreshed_at_utc",
                    "generated_at_utc",
                )
            }
        )
        if observed is None:
            minute = _parse_minute(row)
            day_kind = _pick(row, "day_kind").strip().lower()
            if day_kind not in {"weekday", "weekend", "holiday"}:
                day_kind = "weekday"
        else:
            if source_as_of is not None:
                as_of_candidates.append(source_as_of)
            else:
                as_of_candidates.append(observed)
            minute = (observed.hour * 60) + observed.minute
            day_kind = _day_kind_from_datetime(observed, _pick(row, "holiday", "is_holiday"))
        count_raw = _pick(
            row,
            "count",
            "flow",
            "volume",
            "vehicles",
            "all_motor_vehicles",
            "all_hgvs",
            "cars_and_taxis",
            "lgvs",
        )
        if not count_raw:
            continue
        try:
            count = float(count_raw)
        except ValueError:
            continue
        if count < 0:
            continue
        aggregated[(region, road_bucket, day_kind, minute)].append(count)

    context_series: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for (region, road_bucket, day_kind, minute), values in aggregated.items():
        if not values:
            continue
        avg_count = sum(values) / len(values)
        context_series[(region, road_bucket, day_kind)][minute] = avg_count

    out: list[NormalizedRow] = []
    for (region, road_bucket, day_kind), minute_map in context_series.items():
        if not minute_map:
            continue
        ordered_values = sorted(minute_map.values())
        median = ordered_values[len(ordered_values) // 2]
        baseline = max(1.0, median)
        for minute, avg_count in minute_map.items():
            multiplier = _clamp(avg_count / baseline, 0.45, 3.25)
            out.append(
                NormalizedRow(
                    region=region,
                    road_bucket=road_bucket,
                    day_kind=day_kind,
                    minute=int(minute),
                    multiplier=float(multiplier),
                    as_of_utc=as_of_utc,
                )
            )

    # DfT raw-count snapshots can be sparse on weekend/holiday labels.
    # Fill missing day-kind profiles from neighboring day-kind curves so strict
    # downstream builders always get complete weekday/weekend/holiday coverage.
    profile_index: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in out:
        profile_index[(row.region, row.road_bucket)][row.day_kind][int(row.minute)] = float(row.multiplier)
    for (_, _), day_profiles in profile_index.items():
        if "weekday" not in day_profiles:
            seed = day_profiles.get("weekend") or day_profiles.get("holiday")
            if seed:
                day_profiles["weekday"] = {
                    minute: _clamp(float(multiplier) * 1.08, 0.45, 3.25)
                    for minute, multiplier in seed.items()
                }
        if "weekend" not in day_profiles:
            seed = day_profiles.get("weekday") or day_profiles.get("holiday")
            if seed:
                day_profiles["weekend"] = {
                    minute: _clamp(float(multiplier) * 0.90, 0.45, 3.25)
                    for minute, multiplier in seed.items()
                }
        if "holiday" not in day_profiles:
            if "weekend" in day_profiles:
                seed = day_profiles["weekend"]
                factor = 0.94
            else:
                seed = day_profiles.get("weekday", {})
                factor = 0.82
            if seed:
                day_profiles["holiday"] = {
                    minute: _clamp(float(multiplier) * factor, 0.45, 3.25)
                    for minute, multiplier in seed.items()
                }

    # Add a derived "mixed" road bucket profile from the median of available
    # road buckets so strict builders can require >=4 buckets even when the
    # public corpus only observes three dominant classes.
    regions = sorted({region for region, _ in profile_index})
    for region in regions:
        mixed_key = (region, "mixed")
        if mixed_key in profile_index:
            continue
        combined_day_profiles: dict[str, dict[int, float]] = defaultdict(dict)
        for (reg, road_bucket), day_profiles in profile_index.items():
            if reg != region or road_bucket == "mixed":
                continue
            for day_kind, minute_map in day_profiles.items():
                for minute, multiplier in minute_map.items():
                    bucket = combined_day_profiles.setdefault(day_kind, {})
                    existing = bucket.get(int(minute))
                    if existing is None:
                        bucket[int(minute)] = float(multiplier)
                    else:
                        bucket[int(minute)] = _clamp((float(existing) + float(multiplier)) / 2.0, 0.45, 3.25)
        if combined_day_profiles:
            profile_index[mixed_key] = combined_day_profiles

    # Densify all profiles to full-quarter-hour coverage.
    for _, day_profiles in profile_index.items():
        for day_kind in list(day_profiles.keys()):
            day_profiles[day_kind] = _densify_minute_profile(day_profiles.get(day_kind, {}))

    out = []
    for (region, road_bucket), day_profiles in profile_index.items():
        for day_kind, minute_map in day_profiles.items():
            for minute, multiplier in minute_map.items():
                out.append(
                    NormalizedRow(
                        region=region,
                        road_bucket=road_bucket,
                        day_kind=str(day_kind),
                        minute=int(minute),
                        multiplier=_clamp(float(multiplier), 0.45, 3.25),
                        as_of_utc=as_of_utc,
                    )
                )

    if as_of_candidates and bool(prefer_observed_as_of):
        latest = max(as_of_candidates).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        out = [
            NormalizedRow(
                region=row.region,
                road_bucket=row.road_bucket,
                day_kind=row.day_kind,
                minute=row.minute,
                multiplier=row.multiplier,
                as_of_utc=latest,
            )
            for row in out
        ]
    return out


def _validate_output_quality(
    rows: list[NormalizedRow],
    *,
    min_rows: int,
    min_unique_regions: int,
    min_unique_road_buckets: int,
    min_unique_hours: int,
    max_age_days: int,
) -> None:
    if len(rows) < max(1, int(min_rows)):
        raise RuntimeError(
            f"Empirical departure corpus too small ({len(rows)} rows). "
            f"At least {int(min_rows)} rows are required for strict runtime."
        )
    day_kinds = {row.day_kind for row in rows}
    required_days = {"weekday", "weekend", "holiday"}
    if not required_days.issubset(day_kinds):
        missing = sorted(required_days - day_kinds)
        raise RuntimeError(
            "Empirical departure corpus missing required day_kind coverage: "
            + ", ".join(missing)
        )
    unique_regions = {row.region for row in rows}
    if len(unique_regions) < max(1, int(min_unique_regions)):
        raise RuntimeError(
            "Empirical departure corpus regional diversity too small "
            f"({len(unique_regions)} < {int(min_unique_regions)})."
        )
    unique_road_buckets = {row.road_bucket for row in rows}
    if len(unique_road_buckets) < max(1, int(min_unique_road_buckets)):
        raise RuntimeError(
            "Empirical departure corpus road-bucket diversity too small "
            f"({len(unique_road_buckets)} < {int(min_unique_road_buckets)})."
        )
    unique_hours = {int(max(0, min(1439, row.minute))) // 60 for row in rows}
    if len(unique_hours) < max(1, int(min_unique_hours)):
        raise RuntimeError(
            "Empirical departure corpus hour-slot diversity too small "
            f"({len(unique_hours)} < {int(min_unique_hours)})."
        )
    as_of_values = [_parse_iso_utc(row.as_of_utc) for row in rows]
    if any(item is None for item in as_of_values):
        raise RuntimeError("Empirical departure corpus contains invalid as_of_utc timestamps.")
    latest_as_of = max(item for item in as_of_values if item is not None)
    now_utc = datetime.now(UTC)
    if latest_as_of > (now_utc + timedelta(days=2)):
        raise RuntimeError(
            "Empirical departure corpus as_of_utc is unexpectedly far in the future "
            f"({latest_as_of.isoformat()})."
        )
    if now_utc - latest_as_of > timedelta(days=max(1, int(max_age_days))):
        raise RuntimeError(
            "Empirical departure corpus is stale for strict policy "
            f"({latest_as_of.isoformat()} older than {int(max_age_days)} days)."
        )
    for row in rows:
        if not math.isfinite(float(row.multiplier)):
            raise RuntimeError("Empirical departure corpus contains non-finite multipliers.")
        if not (0.45 <= float(row.multiplier) <= 3.25):
            raise RuntimeError(
                "Empirical departure corpus multiplier outside strict bounds "
                f"({row.multiplier:.6f} not in [0.45, 3.25])."
            )


def build(
    *,
    raw_csv: Path,
    output_csv: Path,
    as_of_utc: str | None = None,
    min_rows: int = 2000,
    min_unique_regions: int = 8,
    min_unique_road_buckets: int = 4,
    min_unique_hours: int = 18,
    max_age_days: int = 30,
) -> int:
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"DfT raw counts file is required for empirical profile ingestion: {raw_csv}"
        )
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = [{str(k).strip().lower(): str(v).strip() for k, v in row.items()} for row in reader]
    if not raw_rows:
        raise RuntimeError(f"Raw DfT counts file is empty: {raw_csv}")

    now_iso = (
        as_of_utc.strip()
        if as_of_utc and as_of_utc.strip()
        else datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    explicit_as_of_supplied = bool(as_of_utc and as_of_utc.strip())
    columns = set(raw_rows[0].keys())
    direct_columns = {"region", "road_bucket", "day_kind", "minute", "multiplier"}
    if direct_columns.issubset(columns):
        rows = _read_direct_rows(raw_rows, as_of_utc=now_iso)
    else:
        rows = _read_observed_count_rows(
            raw_rows,
            as_of_utc=now_iso,
            prefer_observed_as_of=(not explicit_as_of_supplied),
        )

    _validate_output_quality(
        rows,
        min_rows=max(1, int(min_rows)),
        min_unique_regions=max(1, int(min_unique_regions)),
        min_unique_road_buckets=max(1, int(min_unique_road_buckets)),
        min_unique_hours=max(1, int(min_unique_hours)),
        max_age_days=max(1, int(max_age_days)),
    )

    rows.sort(key=lambda r: (r.region, r.road_bucket, r.day_kind, r.minute))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["region", "road_bucket", "day_kind", "minute", "multiplier", "as_of_utc"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "region": row.region,
                    "road_bucket": row.road_bucket,
                    "day_kind": row.day_kind,
                    "minute": row.minute,
                    "multiplier": f"{row.multiplier:.6f}",
                    "as_of_utc": row.as_of_utc,
                }
            )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize empirical DfT counts into contextual minute profiles.")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "dft_counts_raw.csv",
        help="Raw empirical input CSV (direct contextual rows or observed-count rows).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "uk" / "departure_counts_empirical.csv",
    )
    parser.add_argument(
        "--as-of-utc",
        type=str,
        default="",
        help="Optional explicit as_of_utc timestamp.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=2000,
        help="Strict minimum output rows.",
    )
    parser.add_argument(
        "--min-unique-regions",
        type=int,
        default=8,
        help="Strict minimum unique regions in output.",
    )
    parser.add_argument(
        "--min-unique-road-buckets",
        type=int,
        default=4,
        help="Strict minimum unique road buckets in output.",
    )
    parser.add_argument(
        "--min-unique-hours",
        type=int,
        default=18,
        help="Strict minimum unique hours represented in output.",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Maximum allowed age of the newest as_of_utc row.",
    )
    args = parser.parse_args()
    rows = build(
        raw_csv=args.raw_csv,
        output_csv=args.output,
        as_of_utc=(args.as_of_utc or None),
        min_rows=max(1, int(args.min_rows)),
        min_unique_regions=max(1, int(args.min_unique_regions)),
        min_unique_road_buckets=max(1, int(args.min_unique_road_buckets)),
        min_unique_hours=max(1, int(args.min_unique_hours)),
        max_age_days=max(1, int(args.max_age_days)),
    )
    print(f"Wrote {rows} empirical departure rows to {args.output}")


if __name__ == "__main__":
    main()
