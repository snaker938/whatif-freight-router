from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
CARBON_API_BASE = "https://api.carbonintensity.org.uk"

REGION_IDS = {
    "north_scotland": 1,
    "south_scotland": 2,
    "north_west": 3,
    "north_east": 4,
    "yorkshire_humber": 5,
    "north_wales": 6,
    "wales": 7,
    "west_midlands": 8,
    "east_midlands": 9,
    "east_england": 10,
    "south_west": 11,
    "south_england": 12,
    "london": 13,
    "south_east": 14,
}

REGION_ALIASES = {
    "uk_default": "uk_default",
    "london_southeast": "london",
    "south_england": "south_england",
    "midlands": "west_midlands",
    "scotland": "north_scotland",
    "wales_west": "wales",
}


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _parse_regions(raw: str) -> list[str]:
    out: list[str] = []
    for token in str(raw or "").split(","):
        name = str(token).strip().lower()
        if not name:
            continue
        canonical = REGION_ALIASES.get(name, name)
        out.append(canonical)
    deduped = sorted(set(out))
    return deduped or ["uk_default", "london", "south_england", "west_midlands", "wales", "north_west", "north_east"]


def _parse_utc(raw: str | None) -> datetime | None:
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


def _to_api_timestamp(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def _to_kg_per_kwh(value: float) -> float:
    return _clamp(value / 1000.0, 0.02, 2.5)


def _intensity_value(item: dict[str, Any]) -> float | None:
    intensity = item.get("intensity")
    if isinstance(intensity, dict):
        for key in ("actual", "forecast"):
            val = intensity.get(key)
            if isinstance(val, (int, float)) and float(val) > 0.0:
                return _to_kg_per_kwh(float(val))
    for key in ("intensity", "actual", "forecast"):
        val = item.get(key)
        if isinstance(val, (int, float)) and float(val) > 0.0:
            return _to_kg_per_kwh(float(val))
    return None


def _hourly_profile(values: list[tuple[int, float]]) -> list[float]:
    by_hour: dict[int, list[float]] = {hour: [] for hour in range(24)}
    for hour, value in values:
        if 0 <= hour <= 23:
            by_hour[hour].append(float(value))
    existing = [statistics.median(samples) for samples in by_hour.values() if samples]
    fallback = statistics.median(existing) if existing else 0.22
    out: list[float] = []
    for hour in range(24):
        samples = by_hour[hour]
        median_value = statistics.median(samples) if samples else fallback
        out.append(round(_clamp(float(median_value), 0.02, 2.5), 6))
    return out


def _fetch_json(*, client: httpx.Client, url: str) -> dict[str, Any]:
    resp = client.get(url)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid carbon payload from {url}")
    return payload


def _extract_regional_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if isinstance(data, dict):
        rows = data.get("data")
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _regional_values(rows: list[dict[str, Any]]) -> tuple[list[tuple[int, float]], datetime | None]:
    values: list[tuple[int, float]] = []
    latest: datetime | None = None
    for row in rows:
        from_ts = _parse_utc(str(row.get("from", "")))
        if from_ts is not None and (latest is None or from_ts > latest):
            latest = from_ts
        hour = from_ts.hour if from_ts is not None else None
        value = _intensity_value(row)
        if value is None or hour is None:
            continue
        values.append((int(hour), float(value)))
    return values, latest


def build(
    *,
    output_json: Path,
    regions: list[str],
    window_days: int,
    timeout_s: float,
) -> dict[str, Any]:
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=max(1, int(window_days)))
    start_token = _to_api_timestamp(start)
    end_token = _to_api_timestamp(now)
    region_profiles: dict[str, list[float]] = {}
    latest_as_of: datetime | None = None

    with httpx.Client(timeout=max(2.0, float(timeout_s))) as client:
        for region in regions:
            if region == "uk_default":
                continue
            region_id = REGION_IDS.get(region)
            if region_id is None:
                continue
            url = (
                f"{CARBON_API_BASE}/regional/intensity/"
                f"{start_token}/{end_token}/regionid/{region_id}"
            )
            payload = _fetch_json(client=client, url=url)
            rows = _extract_regional_rows(payload)
            values, latest = _regional_values(rows)
            if latest is not None and (latest_as_of is None or latest > latest_as_of):
                latest_as_of = latest
            if not values:
                continue
            region_profiles[region] = _hourly_profile(values)

        uk_url = f"{CARBON_API_BASE}/intensity/{start_token}/{end_token}"
        uk_payload = _fetch_json(client=client, url=uk_url)
        uk_rows = _extract_regional_rows(uk_payload)
        uk_values, uk_latest = _regional_values(uk_rows)
        if uk_latest is not None and (latest_as_of is None or uk_latest > latest_as_of):
            latest_as_of = uk_latest
        if uk_values:
            region_profiles["uk_default"] = _hourly_profile(uk_values)

    if not region_profiles:
        raise RuntimeError("No carbon intensity regions were collected.")

    if "uk_default" not in region_profiles:
        aggregated: list[float] = []
        for hour in range(24):
            hour_values = [values[hour] for values in region_profiles.values() if len(values) == 24]
            aggregated.append(round(statistics.mean(hour_values), 6) if hour_values else 0.22)
        region_profiles["uk_default"] = aggregated

    latest_token = (
        latest_as_of.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if latest_as_of is not None
        else now.isoformat().replace("+00:00", "Z")
    )
    generated = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = {
        "source": "public_carbon_intensity_api_uk",
        "source_url": CARBON_API_BASE,
        "as_of_utc": latest_token,
        "generated_at_utc": generated,
        "regions": region_profiles,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "regions": sorted(region_profiles),
        "region_count": len(region_profiles),
        "output_json": str(output_json),
        "as_of_utc": latest_token,
    }
    output_json.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect public UK carbon intensity into carbon_intensity_hourly_raw.json."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "carbon_intensity_hourly_raw.json",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="uk_default,london,south_east,midlands,scotland,wales,north_west,north_east",
    )
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    summary = build(
        output_json=args.output,
        regions=_parse_regions(args.regions),
        window_days=max(1, int(args.window_days)),
        timeout_s=max(2.0, float(args.timeout_s)),
    )
    print(
        "Collected carbon intensity raw payload "
        f"(regions={summary['region_count']}, output={summary['output_json']})."
    )


if __name__ == "__main__":
    main()
