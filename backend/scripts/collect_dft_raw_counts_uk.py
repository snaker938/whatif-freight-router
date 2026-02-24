from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]

REGION_ID_TO_NAME = {
    1: "south_west",
    2: "south_east",
    3: "london",
    4: "east_of_england",
    5: "west_midlands",
    6: "east_midlands",
    7: "yorkshire_humber",
    8: "north_west",
    9: "north_east",
    10: "scotland",
    11: "wales",
}

CORRIDOR_BY_REGION = {
    "london": "london_southeast",
    "south_east": "south_england",
    "south_west": "south_england",
    "east_of_england": "south_england",
    "east_midlands": "midlands_east",
    "west_midlands": "midlands_west",
    "north_west": "north_west_corridor",
    "north_east": "north_east_corridor",
    "yorkshire_humber": "north_england_central",
    "wales": "wales_west",
    "scotland": "scotland_south",
}

RAW_COLUMNS = [
    "dedupe_key",
    "id",
    "count_point_id",
    "observed_at_utc",
    "count_date",
    "year",
    "hour",
    "holiday",
    "day_kind",
    "region_id",
    "region",
    "corridor_bucket",
    "local_authority_id",
    "road_name",
    "road_category",
    "road_type",
    "road_bucket",
    "direction_of_travel",
    "latitude",
    "longitude",
    "all_motor_vehicles",
    "all_hgvs",
    "cars_and_taxis",
    "lgvs",
    "buses_and_coaches",
    "pedal_cycles",
    "two_wheeled_motor_vehicles",
    "source_url",
    "as_of_utc",
]


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_years(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw or "").split(","):
        text = token.strip()
        if not text:
            continue
        try:
            year = int(text)
        except ValueError:
            continue
        if 1900 <= year <= 3000:
            out.append(year)
    return sorted(set(out))


def _parse_day(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _day_kind(*, day: date | None, holiday: bool) -> str:
    if holiday:
        return "holiday"
    if day is None:
        return "weekday"
    return "weekend" if day.weekday() >= 5 else "weekday"


def _dedupe_key(row: dict[str, Any]) -> str:
    parts = (
        str(row.get("id", "")).strip(),
        str(row.get("count_point_id", "")).strip(),
        str(row.get("count_date", "")).strip(),
        str(row.get("hour", "")).strip(),
        str(row.get("direction_of_travel", "")).strip(),
    )
    return "|".join(parts)


def _road_bucket(*, road_category: str, road_type: str, road_name: str) -> str:
    category = str(road_category or "").strip().upper()
    rtype = str(road_type or "").strip().lower()
    name = str(road_name or "").strip().upper()
    if category.startswith("M") or name.startswith("M"):
        return "motorway_dominant"
    if category.startswith("TA") or "trunk" in rtype:
        return "trunk_heavy"
    if category.startswith("PA") or "major" in rtype:
        return "primary_heavy"
    if "minor" in rtype:
        return "urban_local_heavy"
    return "mixed"


def _region_name(region_id: int) -> str:
    return REGION_ID_TO_NAME.get(region_id, "uk_default")


def _corridor(region_name: str) -> str:
    return CORRIDOR_BY_REGION.get(region_name, "uk_default")


def _load_holiday_dates(
    *,
    bank_holidays_url: str,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> set[str]:
    holiday_dates: set[str] = set()
    with httpx.Client(timeout=max(2.0, float(timeout_s))) as client:
        last_error: Exception | None = None
        for attempt in range(max(1, int(retries))):
            try:
                resp = client.get(bank_holidays_url)
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, dict):
                    return set()
                for block in payload.values():
                    if not isinstance(block, dict):
                        continue
                    events = block.get("events", [])
                    if not isinstance(events, list):
                        continue
                    for row in events:
                        if not isinstance(row, dict):
                            continue
                        day = _parse_day(row.get("date"))
                        if day is not None:
                            holiday_dates.add(day.isoformat())
                return holiday_dates
            except Exception as exc:  # pragma: no cover - defensive network fallback
                last_error = exc
                if attempt + 1 < max(1, int(retries)):
                    time.sleep(max(0.05, float(backoff_s)) * float(2**attempt))
        if last_error is not None:
            print(f"[collect_dft_raw_counts_uk] bank holiday fetch failed: {type(last_error).__name__}")
    return holiday_dates


def _load_existing_dedupe_keys(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    out: set[str] = set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = str(row.get("dedupe_key", "")).strip()
            if key:
                out.add(key)
    return out


def _load_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {}
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_state(state_file: Path, payload: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fetch_page(
    *,
    client: httpx.Client,
    base_url: str,
    year: int,
    page_number: int,
    page_size: int,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> dict[str, Any]:
    params = {
        "filter[year]": str(year),
        "page[number]": str(max(1, int(page_number))),
        "page[size]": str(max(1, int(page_size))),
    }
    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            resp = client.get(base_url, params=params, timeout=max(2.0, float(timeout_s)))
            resp.raise_for_status()
            payload = resp.json()
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:  # pragma: no cover - defensive network fallback
            last_error = exc
            if attempt + 1 < max(1, int(retries)):
                time.sleep(max(0.05, float(backoff_s)) * float(2**attempt))
    if last_error is not None:
        raise RuntimeError(
            f"DfT fetch failed for year={year} page={page_number}: {type(last_error).__name__}"
        ) from last_error
    raise RuntimeError(f"DfT fetch failed for year={year} page={page_number}.")


def collect(
    *,
    output_csv: Path,
    years: list[int],
    base_url: str,
    max_pages_per_year: int,
    page_size: int,
    retries: int,
    backoff_s: float,
    timeout_s: float,
    target_min_rows: int,
    append_safe: bool,
    resume: bool,
    state_file: Path,
    bank_holidays_url: str,
) -> dict[str, Any]:
    if not years:
        raise RuntimeError("No valid years were provided.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dedupe_keys = _load_existing_dedupe_keys(output_csv) if append_safe else set()
    holidays = _load_holiday_dates(
        bank_holidays_url=bank_holidays_url,
        timeout_s=timeout_s,
        retries=retries,
        backoff_s=backoff_s,
    )
    state = _load_state(state_file) if resume else {}
    completed_years = {int(item) for item in state.get("completed_years", []) if isinstance(item, int)}
    resume_year = int(state.get("resume_year", 0)) if state.get("resume_year") else None
    resume_page = int(state.get("resume_page", 1)) if state.get("resume_page") else 1

    file_exists = output_csv.exists()
    mode = "a" if file_exists else "w"
    rows_written = 0
    pages_fetched = 0
    year_row_counts: dict[str, int] = {}
    started_at = datetime.now(UTC)
    as_of_utc = _iso_utc(started_at)

    with output_csv.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        if not file_exists:
            writer.writeheader()
        with httpx.Client(timeout=max(2.0, float(timeout_s))) as client:
            for year in years:
                if year in completed_years:
                    continue
                start_page = resume_page if (resume_year == year) else 1
                year_rows = 0
                for page in range(start_page, max(1, int(max_pages_per_year)) + 1):
                    payload = _fetch_page(
                        client=client,
                        base_url=base_url,
                        year=year,
                        page_number=page,
                        page_size=page_size,
                        timeout_s=timeout_s,
                        retries=retries,
                        backoff_s=backoff_s,
                    )
                    pages_fetched += 1
                    rows = payload.get("data", [])
                    if not isinstance(rows, list) or not rows:
                        break
                    for raw in rows:
                        if not isinstance(raw, dict):
                            continue
                        key = _dedupe_key(raw)
                        if not key or key in dedupe_keys:
                            continue
                        dedupe_keys.add(key)
                        day = _parse_day(raw.get("count_date"))
                        day_iso = day.isoformat() if day is not None else ""
                        hour = max(0, min(23, _coerce_int(raw.get("hour"), 0)))
                        holiday = day_iso in holidays if day_iso else False
                        observed_at = (
                            f"{day_iso}T{hour:02d}:00:00Z" if day_iso else f"{year:04d}-01-01T{hour:02d}:00:00Z"
                        )
                        region_id = _coerce_int(raw.get("region_id"), 0)
                        region = _region_name(region_id)
                        row = {
                            "dedupe_key": key,
                            "id": str(raw.get("id", "")).strip(),
                            "count_point_id": str(raw.get("count_point_id", "")).strip(),
                            "observed_at_utc": observed_at,
                            "count_date": day_iso,
                            "year": str(year),
                            "hour": str(hour),
                            "holiday": "1" if holiday else "0",
                            "day_kind": _day_kind(day=day, holiday=holiday),
                            "region_id": str(region_id),
                            "region": region,
                            "corridor_bucket": _corridor(region),
                            "local_authority_id": str(_coerce_int(raw.get("local_authority_id"), 0)),
                            "road_name": str(raw.get("road_name", "")).strip().upper(),
                            "road_category": str(raw.get("road_category", "")).strip().upper(),
                            "road_type": str(raw.get("road_type", "")).strip(),
                            "road_bucket": _road_bucket(
                                road_category=str(raw.get("road_category", "")),
                                road_type=str(raw.get("road_type", "")),
                                road_name=str(raw.get("road_name", "")),
                            ),
                            "direction_of_travel": str(raw.get("direction_of_travel", "")).strip().upper(),
                            "latitude": f"{_coerce_float(raw.get('latitude'), 0.0):.7f}",
                            "longitude": f"{_coerce_float(raw.get('longitude'), 0.0):.7f}",
                            "all_motor_vehicles": str(_coerce_int(raw.get("all_motor_vehicles"), 0)),
                            "all_hgvs": str(_coerce_int(raw.get("all_hgvs"), 0)),
                            "cars_and_taxis": str(_coerce_int(raw.get("cars_and_taxis"), 0)),
                            "lgvs": str(_coerce_int(raw.get("lgvs"), 0)),
                            "buses_and_coaches": str(_coerce_int(raw.get("buses_and_coaches"), 0)),
                            "pedal_cycles": str(_coerce_int(raw.get("pedal_cycles"), 0)),
                            "two_wheeled_motor_vehicles": str(
                                _coerce_int(raw.get("two_wheeled_motor_vehicles"), 0)
                            ),
                            "source_url": base_url,
                            "as_of_utc": as_of_utc,
                        }
                        writer.writerow(row)
                        rows_written += 1
                        year_rows += 1
                    handle.flush()
                    _save_state(
                        state_file,
                        {
                            "resume_year": int(year),
                            "resume_page": int(page + 1),
                            "completed_years": sorted(completed_years),
                            "rows_written": int(rows_written),
                            "updated_at_utc": _iso_utc(datetime.now(UTC)),
                        },
                    )
                    next_page = payload.get("next_page_url")
                    last_page = _coerce_int(payload.get("last_page"), 0)
                    if not next_page and last_page > 0 and page >= last_page:
                        break
                    if not next_page and last_page == 0:
                        break
                    if rows_written >= max(1, int(target_min_rows)):
                        break
                completed_years.add(int(year))
                year_row_counts[str(year)] = int(year_rows)
                _save_state(
                    state_file,
                    {
                        "resume_year": int(year),
                        "resume_page": 1,
                        "completed_years": sorted(completed_years),
                        "rows_written": int(rows_written),
                        "updated_at_utc": _iso_utc(datetime.now(UTC)),
                    },
                )
                if rows_written >= max(1, int(target_min_rows)):
                    break

    finished_at = datetime.now(UTC)
    summary = {
        "source": "dft_raw_counts_public_api",
        "base_url": base_url,
        "years_requested": years,
        "rows_written": int(rows_written),
        "target_min_rows": int(target_min_rows),
        "pages_fetched": int(pages_fetched),
        "year_row_counts": year_row_counts,
        "output_csv": str(output_csv),
        "state_file": str(state_file),
        "append_safe": bool(append_safe),
        "started_at_utc": _iso_utc(started_at),
        "finished_at_utc": _iso_utc(finished_at),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "bank_holiday_count": len(holidays),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect high-volume UK DfT raw counts into empirical CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "dft_counts_raw.csv",
        help="Normalized DfT raw output CSV path.",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2023,2024,2025,2026",
        help="Comma-separated year list.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://roadtraffic.dft.gov.uk/api/raw-counts",
        help="DfT raw-counts API endpoint.",
    )
    parser.add_argument("--max-pages-per-query", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--target-min-rows", type=int, default=200000)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--backoff-s", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--append-safe", action="store_true", default=True)
    parser.add_argument("--no-append-safe", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "dft_counts_raw.state.json",
    )
    parser.add_argument(
        "--bank-holidays-url",
        type=str,
        default="https://www.gov.uk/bank-holidays.json",
    )
    args = parser.parse_args()

    years = _parse_years(args.years)
    if not years:
        raise RuntimeError("No valid years found in --years.")
    append_safe = bool(args.append_safe) and not bool(args.no_append_safe)
    resume = bool(args.resume) and not bool(args.no_resume)
    summary = collect(
        output_csv=args.output,
        years=years,
        base_url=str(args.base_url),
        max_pages_per_year=max(1, int(args.max_pages_per_query)),
        page_size=max(1, int(args.page_size)),
        retries=max(1, int(args.retries)),
        backoff_s=max(0.05, float(args.backoff_s)),
        timeout_s=max(2.0, float(args.timeout_s)),
        target_min_rows=max(1, int(args.target_min_rows)),
        append_safe=append_safe,
        resume=resume,
        state_file=args.state_file,
        bank_holidays_url=str(args.bank_holidays_url),
    )
    print(
        "Collected DfT raw counts "
        f"(rows={summary['rows_written']}, pages={summary['pages_fetched']}, output={summary['output_csv']})."
    )


if __name__ == "__main__":
    main()
