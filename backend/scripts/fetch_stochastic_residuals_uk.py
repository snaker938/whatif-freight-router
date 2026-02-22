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
class ResidualRow:
    regime_id: str
    corridor_bucket: str
    day_kind: str
    local_time_slot: str
    road_bucket: str
    weather_profile: str
    vehicle_type: str
    traffic: float
    incident: float
    weather: float
    price: float
    eco: float
    sigma: float
    as_of_utc: str


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def _canonical_day_kind(raw: str) -> str:
    key = raw.strip().lower()
    if key in {"weekday", "weekend", "holiday"}:
        return key
    return "weekday"


def _canonical_road_bucket(raw: str) -> str:
    key = raw.strip().lower()
    if not key:
        return "mixed"
    if "motorway" in key:
        return "motorway_heavy"
    if "trunk" in key:
        return "trunk_heavy"
    if "urban" in key or "local" in key:
        return "urban_local_heavy"
    return key


def _canonical_weather(raw: str) -> str:
    key = raw.strip().lower()
    if key in {"clear", "rain", "storm", "snow", "fog"}:
        return key
    return "clear"


def _canonical_vehicle(raw: str) -> str:
    key = raw.strip().lower()
    if "artic" in key:
        return "artic_hgv"
    if "rigid" in key:
        return "rigid_hgv"
    if "van" in key:
        return "van"
    return "default"


def _canonical_corridor(raw: str) -> str:
    key = raw.strip().lower()
    if not key:
        return "uk_default"
    return key.replace(" ", "_")


def _canonical_local_slot(raw: str) -> str:
    key = raw.strip().lower()
    if key.startswith("h") and len(key) == 3:
        try:
            hour = int(key[1:])
        except ValueError:
            hour = 12
        return f"h{max(0, min(23, hour)):02d}"
    try:
        hour = int(float(key))
    except ValueError:
        return "h12"
    return f"h{max(0, min(23, hour)):02d}"


def _ratio(actual: float, expected: float, *, low: float, high: float) -> float:
    denom = max(1e-6, expected)
    return _clamp(actual / denom, low, high)


def _safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, float(q))) * float(len(ordered) - 1)
    lo = int(rank)
    hi = min(len(ordered) - 1, lo + 1)
    frac = rank - float(lo)
    return ordered[lo] + ((ordered[hi] - ordered[lo]) * frac)


def _parse_rows(raw_csv: Path, *, as_of_utc: str) -> list[ResidualRow]:
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{str(k).strip().lower(): str(v).strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise RuntimeError(f"Raw residual corpus is empty: {raw_csv}")

    direct_cols = {
        "regime_id",
        "traffic",
        "incident",
        "weather",
        "price",
        "eco",
        "sigma",
    }
    output: list[ResidualRow] = []
    if direct_cols.issubset(rows[0].keys()):
        for row in rows:
            try:
                output.append(
                    ResidualRow(
                        regime_id=_pick(row, "regime_id") or "weekday_offpeak",
                        corridor_bucket=_canonical_corridor(_pick(row, "corridor_bucket")),
                        day_kind=_canonical_day_kind(_pick(row, "day_kind")),
                        local_time_slot=_canonical_local_slot(_pick(row, "local_time_slot")),
                        road_bucket=_canonical_road_bucket(_pick(row, "road_bucket")),
                        weather_profile=_canonical_weather(_pick(row, "weather_profile")),
                        vehicle_type=_canonical_vehicle(_pick(row, "vehicle_type")),
                        traffic=_clamp(float(_pick(row, "traffic")), 0.25, 4.5),
                        incident=_clamp(float(_pick(row, "incident")), 0.25, 4.5),
                        weather=_clamp(float(_pick(row, "weather")), 0.25, 4.5),
                        price=_clamp(float(_pick(row, "price")), 0.25, 4.5),
                        eco=_clamp(float(_pick(row, "eco")), 0.25, 4.5),
                        sigma=_clamp(float(_pick(row, "sigma")), 0.01, 2.5),
                        as_of_utc=as_of_utc,
                    )
                )
            except ValueError:
                continue
        return output

    # Observed residual mode.
    # Required observed columns:
    # actual_duration_s, expected_duration_s
    # optional: actual_monetary/expected_monetary, actual_emissions/expected_emissions
    grouped_sigma: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        try:
            actual_duration = float(_pick(row, "actual_duration_s", "observed_duration_s"))
            expected_duration = float(_pick(row, "expected_duration_s", "baseline_duration_s"))
        except ValueError:
            continue
        if actual_duration <= 0 or expected_duration <= 0:
            continue
        duration_ratio = _ratio(actual_duration, expected_duration, low=0.25, high=4.5)
        try:
            actual_money = float(_pick(row, "actual_monetary_cost", "observed_monetary_cost"))
            expected_money = float(_pick(row, "expected_monetary_cost", "baseline_monetary_cost"))
            money_ratio = _ratio(actual_money, expected_money, low=0.25, high=4.5)
        except ValueError:
            money_ratio = duration_ratio
        try:
            actual_emissions = float(_pick(row, "actual_emissions_kg", "observed_emissions_kg"))
            expected_emissions = float(_pick(row, "expected_emissions_kg", "baseline_emissions_kg"))
            emissions_ratio = _ratio(actual_emissions, expected_emissions, low=0.25, high=4.5)
        except ValueError:
            emissions_ratio = duration_ratio

        day_kind = _canonical_day_kind(_pick(row, "day_kind"))
        road_bucket = _canonical_road_bucket(_pick(row, "road_bucket", "road_class"))
        weather_profile = _canonical_weather(_pick(row, "weather_profile"))
        vehicle_type = _canonical_vehicle(_pick(row, "vehicle_type"))
        regime_id = _pick(row, "regime_id")
        if not regime_id:
            regime_id = (
                "holiday"
                if day_kind == "holiday"
                else "weekend"
                if day_kind == "weekend"
                else "weekday_peak" if duration_ratio >= 1.08 else "weekday_offpeak"
            )
        corridor_bucket = _canonical_corridor(_pick(row, "corridor_bucket", "region"))
        local_slot = _canonical_local_slot(_pick(row, "local_time_slot", "hour", "hour_of_day"))
        sigma = abs((duration_ratio - 1.0) * 0.55) + abs((money_ratio - 1.0) * 0.30) + abs((emissions_ratio - 1.0) * 0.15)
        sigma = _clamp(sigma, 0.01, 2.5)
        grouped_sigma[(day_kind, road_bucket, weather_profile, vehicle_type)].append(sigma)

        incident_value = _safe_float(
            _pick(row, "incident_factor", "observed_incident_factor"),
            (duration_ratio * 0.7) + 0.3,
        )
        weather_value = _safe_float(
            _pick(row, "weather_factor", "observed_weather_factor"),
            (duration_ratio * 0.45) + 0.55,
        )
        output.append(
            ResidualRow(
                regime_id=regime_id,
                corridor_bucket=corridor_bucket,
                day_kind=day_kind,
                local_time_slot=local_slot,
                road_bucket=road_bucket,
                weather_profile=weather_profile,
                vehicle_type=vehicle_type,
                traffic=duration_ratio,
                incident=_clamp(incident_value, 0.25, 4.5),
                weather=_clamp(weather_value, 0.25, 4.5),
                price=money_ratio,
                eco=emissions_ratio,
                sigma=sigma,
                as_of_utc=as_of_utc,
            )
        )

    return output


def _validate_output_quality(
    rows: list[ResidualRow],
    *,
    min_rows: int,
    min_unique_regimes: int,
    min_unique_road_buckets: int,
    min_unique_weather_profiles: int,
    min_unique_vehicle_types: int,
    min_unique_local_slots: int,
    min_unique_corridors: int,
    max_age_days: int,
) -> None:
    if len(rows) < max(1, int(min_rows)):
        raise RuntimeError(
            f"Residual corpus too small ({len(rows)} rows). "
            f"At least {int(min_rows)} rows are required for strict runtime."
        )
    day_kinds = {row.day_kind for row in rows}
    required_days = {"weekday", "weekend", "holiday"}
    if not required_days.issubset(day_kinds):
        missing = sorted(required_days - day_kinds)
        raise RuntimeError(
            "Residual corpus missing required day_kind coverage: " + ", ".join(missing)
        )
    regimes = {row.regime_id for row in rows if row.regime_id}
    if len(regimes) < max(1, int(min_unique_regimes)):
        raise RuntimeError(
            "Residual corpus regime diversity too small "
            f"({len(regimes)} < {int(min_unique_regimes)})."
        )
    roads = {row.road_bucket for row in rows if row.road_bucket}
    if len(roads) < max(1, int(min_unique_road_buckets)):
        raise RuntimeError(
            "Residual corpus road-bucket diversity too small "
            f"({len(roads)} < {int(min_unique_road_buckets)})."
        )
    weathers = {row.weather_profile for row in rows if row.weather_profile}
    if len(weathers) < max(1, int(min_unique_weather_profiles)):
        raise RuntimeError(
            "Residual corpus weather-profile diversity too small "
            f"({len(weathers)} < {int(min_unique_weather_profiles)})."
        )
    vehicles = {row.vehicle_type for row in rows if row.vehicle_type}
    if len(vehicles) < max(1, int(min_unique_vehicle_types)):
        raise RuntimeError(
            "Residual corpus vehicle-type diversity too small "
            f"({len(vehicles)} < {int(min_unique_vehicle_types)})."
        )
    slots = {row.local_time_slot for row in rows if row.local_time_slot}
    if len(slots) < max(1, int(min_unique_local_slots)):
        raise RuntimeError(
            "Residual corpus local-time-slot diversity too small "
            f"({len(slots)} < {int(min_unique_local_slots)})."
        )
    corridors = {row.corridor_bucket for row in rows if row.corridor_bucket}
    if len(corridors) < max(1, int(min_unique_corridors)):
        raise RuntimeError(
            "Residual corpus corridor diversity too small "
            f"({len(corridors)} < {int(min_unique_corridors)})."
        )
    as_of_values = [_parse_iso_utc(row.as_of_utc) for row in rows]
    if any(item is None for item in as_of_values):
        raise RuntimeError("Residual corpus contains invalid as_of_utc timestamps.")
    latest_as_of = max(item for item in as_of_values if item is not None)
    now_utc = datetime.now(UTC)
    if latest_as_of > (now_utc + timedelta(days=2)):
        raise RuntimeError(
            "Residual corpus as_of_utc is unexpectedly far in the future "
            f"({latest_as_of.isoformat()})."
        )
    if now_utc - latest_as_of > timedelta(days=max(1, int(max_age_days))):
        raise RuntimeError(
            "Residual corpus is stale for strict policy "
            f"({latest_as_of.isoformat()} older than {int(max_age_days)} days)."
        )

    traffic_vals = [float(row.traffic) for row in rows]
    incident_vals = [float(row.incident) for row in rows]
    weather_vals = [float(row.weather) for row in rows]
    price_vals = [float(row.price) for row in rows]
    eco_vals = [float(row.eco) for row in rows]
    sigma_vals = [float(row.sigma) for row in rows]
    for label, values, low, high in (
        ("traffic", traffic_vals, 0.25, 4.5),
        ("incident", incident_vals, 0.25, 4.5),
        ("weather", weather_vals, 0.25, 4.5),
        ("price", price_vals, 0.25, 4.5),
        ("eco", eco_vals, 0.25, 4.5),
        ("sigma", sigma_vals, 0.01, 2.5),
    ):
        for value in values:
            if not math.isfinite(value):
                raise RuntimeError(f"Residual corpus contains non-finite {label} values.")
            if not (low <= value <= high):
                raise RuntimeError(
                    f"Residual corpus contains out-of-range {label} values "
                    f"({value:.6f} not in [{low}, {high}])."
                )
        span = _quantile(values, 0.95) - _quantile(values, 0.05)
        if span <= 0.01:
            raise RuntimeError(
                f"Residual corpus {label} distribution is nearly collapsed (p95-p05={span:.6f})."
            )


def build(
    *,
    raw_csv: Path,
    output_csv: Path,
    min_rows: int = 5000,
    as_of_utc: str | None = None,
    min_unique_regimes: int = 3,
    min_unique_road_buckets: int = 3,
    min_unique_weather_profiles: int = 3,
    min_unique_vehicle_types: int = 3,
    min_unique_local_slots: int = 4,
    min_unique_corridors: int = 1,
    max_age_days: int = 30,
) -> int:
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Observed residual corpus is required for strict stochastic calibration: {raw_csv}"
        )
    resolved_as_of = (
        as_of_utc.strip()
        if as_of_utc and as_of_utc.strip()
        else datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    rows = _parse_rows(raw_csv, as_of_utc=resolved_as_of)
    _validate_output_quality(
        rows,
        min_rows=max(1, int(min_rows)),
        min_unique_regimes=max(1, int(min_unique_regimes)),
        min_unique_road_buckets=max(1, int(min_unique_road_buckets)),
        min_unique_weather_profiles=max(1, int(min_unique_weather_profiles)),
        min_unique_vehicle_types=max(1, int(min_unique_vehicle_types)),
        min_unique_local_slots=max(1, int(min_unique_local_slots)),
        min_unique_corridors=max(1, int(min_unique_corridors)),
        max_age_days=max(1, int(max_age_days)),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r.regime_id, r.day_kind, r.road_bucket, r.weather_profile, r.vehicle_type))
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "regime_id",
                "corridor_bucket",
                "day_kind",
                "local_time_slot",
                "road_bucket",
                "weather_profile",
                "vehicle_type",
                "traffic",
                "incident",
                "weather",
                "price",
                "eco",
                "sigma",
                "as_of_utc",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "regime_id": row.regime_id,
                    "corridor_bucket": row.corridor_bucket,
                    "day_kind": row.day_kind,
                    "local_time_slot": row.local_time_slot,
                    "road_bucket": row.road_bucket,
                    "weather_profile": row.weather_profile,
                    "vehicle_type": row.vehicle_type,
                    "traffic": f"{row.traffic:.6f}",
                    "incident": f"{row.incident:.6f}",
                    "weather": f"{row.weather:.6f}",
                    "price": f"{row.price:.6f}",
                    "eco": f"{row.eco:.6f}",
                    "sigma": f"{row.sigma:.6f}",
                    "as_of_utc": row.as_of_utc,
                }
            )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize observed residuals into stochastic calibration corpus.")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "stochastic_residuals_raw.csv",
        help="Raw observed residual CSV (direct residual factors or observed-vs-expected columns).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "uk" / "stochastic_residuals_empirical.csv",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=5000,
        help="Strict minimum output rows.",
    )
    parser.add_argument("--min-unique-regimes", type=int, default=3)
    parser.add_argument("--min-unique-road-buckets", type=int, default=3)
    parser.add_argument("--min-unique-weather-profiles", type=int, default=3)
    parser.add_argument("--min-unique-vehicle-types", type=int, default=3)
    parser.add_argument("--min-unique-local-slots", type=int, default=4)
    parser.add_argument("--min-unique-corridors", type=int, default=1)
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument(
        "--as-of-utc",
        type=str,
        default="",
        help="Optional explicit as_of_utc timestamp.",
    )
    args = parser.parse_args()
    rows = build(
        raw_csv=args.raw_csv,
        output_csv=args.output,
        min_rows=max(1, int(args.target_rows)),
        as_of_utc=(args.as_of_utc or None),
        min_unique_regimes=max(1, int(args.min_unique_regimes)),
        min_unique_road_buckets=max(1, int(args.min_unique_road_buckets)),
        min_unique_weather_profiles=max(1, int(args.min_unique_weather_profiles)),
        min_unique_vehicle_types=max(1, int(args.min_unique_vehicle_types)),
        min_unique_local_slots=max(1, int(args.min_unique_local_slots)),
        min_unique_corridors=max(1, int(args.min_unique_corridors)),
        max_age_days=max(1, int(args.max_age_days)),
    )
    print(f"Wrote {rows} residual rows to {args.output}")


if __name__ == "__main__":
    main()
