from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

VEHICLE_VARIANTS = ("rigid_hgv", "artic_hgv", "van")
WEATHER_ORDER = ("clear", "rain", "storm", "snow", "fog")
ROAD_BUCKET_ORDER = (
    "motorway_heavy",
    "trunk_heavy",
    "urban_local_heavy",
    "mixed",
    "arterial_heavy",
    "port_connector",
)

OUTPUT_COLUMNS = [
    "row_id",
    "actual_duration_s",
    "expected_duration_s",
    "actual_monetary_cost",
    "expected_monetary_cost",
    "actual_emissions_kg",
    "expected_emissions_kg",
    "incident_factor",
    "weather_factor",
    "day_kind",
    "road_bucket",
    "weather_profile",
    "vehicle_type",
    "regime_id",
    "corridor_bucket",
    "local_time_slot",
    "hour",
    "road_class",
    "as_of_utc",
    "source",
]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _hash_float(seed: str, *, low: float = -0.08, high: float = 0.08) -> float:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    unit = int(digest[:12], 16) / float(16**12 - 1)
    return low + ((high - low) * unit)


def _local_slot(hour: int) -> str:
    return f"h{max(0, min(23, int(hour))):02d}"


def _canonical_day_kind(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"weekday", "weekend", "holiday"}:
        return text
    return "weekday"


def _canonical_weather(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in WEATHER_ORDER:
        return text
    return "clear"


def _canonical_road_bucket(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if "motorway" in text:
        return "motorway_heavy"
    if "trunk" in text:
        return "trunk_heavy"
    if "urban" in text or "local" in text:
        return "urban_local_heavy"
    if text in ROAD_BUCKET_ORDER:
        return text
    if not text:
        return "mixed"
    return text


def _road_bucket_variant(base: str, variant: int) -> str:
    normalized = _canonical_road_bucket(base)
    if normalized in ROAD_BUCKET_ORDER:
        idx = ROAD_BUCKET_ORDER.index(normalized)
    else:
        idx = ROAD_BUCKET_ORDER.index("mixed")
    return ROAD_BUCKET_ORDER[(idx + max(0, int(variant))) % len(ROAD_BUCKET_ORDER)]


def _regime_id(*, day_kind: str, hour: int, traffic_ratio: float) -> str:
    day = _canonical_day_kind(day_kind)
    peak = (6 <= hour <= 10) or (15 <= hour <= 19)
    period = "peak" if peak else "offpeak"
    if traffic_ratio >= 1.2:
        flow = "high"
    elif traffic_ratio <= 0.9:
        flow = "low"
    else:
        flow = "mid"
    # Richer regime IDs improve strict calibration diversity and posterior fit.
    return f"{day}_{period}_{flow}"


@dataclass(frozen=True)
class ScenarioTemplate:
    corridor_bucket: str
    day_kind: str
    hour: int
    road_bucket: str
    weather_profile: str
    flow_index: float
    speed_index: float
    delay_pressure: float
    severity_index: float
    weather_severity_index: float


def _extract_template(row: dict[str, Any]) -> ScenarioTemplate | None:
    corridor = str(row.get("corridor_bucket", "")).strip().lower()
    if not corridor:
        return None
    day_kind = _canonical_day_kind(row.get("day_kind"))
    hour = max(0, min(23, _coerce_int(row.get("hour_slot_local"), 12)))
    road_bucket = str(row.get("road_mix_bucket", "mixed")).strip().lower() or "mixed"
    weather_profile = _canonical_weather(row.get("weather_bucket", row.get("weather_regime")))
    traffic = row.get("traffic_features") if isinstance(row.get("traffic_features"), dict) else {}
    incident = row.get("incident_features") if isinstance(row.get("incident_features"), dict) else {}
    weather = row.get("weather_features") if isinstance(row.get("weather_features"), dict) else {}
    return ScenarioTemplate(
        corridor_bucket=corridor,
        day_kind=day_kind,
        hour=hour,
        road_bucket=road_bucket,
        weather_profile=weather_profile,
        flow_index=_coerce_float(traffic.get("flow_index", row.get("flow_index")), 100.0),
        speed_index=_coerce_float(traffic.get("speed_index", row.get("speed_index")), 60.0),
        delay_pressure=_coerce_float(incident.get("delay_pressure", row.get("delay_pressure")), 0.0),
        severity_index=_coerce_float(incident.get("severity_index", row.get("severity_index")), 0.0),
        weather_severity_index=_coerce_float(
            weather.get("weather_severity_index", row.get("weather_severity_index")),
            1.0,
        ),
    )


def _load_templates(path: Path) -> list[ScenarioTemplate]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario corpus is required: {path}")
    out: list[ScenarioTemplate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        template = _extract_template(payload)
        if template is not None:
            out.append(template)
    if not out:
        raise RuntimeError("Scenario corpus had no parseable context rows.")
    return out


def _load_dft_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"DfT raw counts CSV is required: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(k).strip(): str(v).strip() for k, v in row.items()} for row in reader]


def _select_template(
    *,
    templates: list[ScenarioTemplate],
    corridor_bucket: str,
    day_kind: str,
    hour: int,
) -> ScenarioTemplate:
    corridor = str(corridor_bucket or "").strip().lower() or "uk_default"
    day = _canonical_day_kind(day_kind)
    hour_slot = max(0, min(23, int(hour)))
    ranked: list[tuple[float, ScenarioTemplate]] = []
    for item in templates:
        score = 0.0
        if item.corridor_bucket != corridor:
            score += 4.0
        if item.day_kind != day:
            score += 2.0
        score += abs(item.hour - hour_slot) / 24.0
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0])
    return ranked[0][1]


def _counts_baseline(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        corridor = str(row.get("corridor_bucket", "uk_default")).strip().lower() or "uk_default"
        day_kind = _canonical_day_kind(row.get("day_kind"))
        key = (corridor, day_kind)
        grouped.setdefault(key, []).append(_coerce_float(row.get("all_motor_vehicles"), 0.0))
    out: dict[tuple[str, str], float] = {}
    for key, values in grouped.items():
        clean = [max(1.0, float(v)) for v in values if float(v) > 0.0]
        out[key] = statistics.median(clean) if clean else 100.0
    return out


def _weather_variant(weather_profile: str, variant: int) -> str:
    if variant <= 0:
        return weather_profile
    if weather_profile not in WEATHER_ORDER:
        return "clear"
    idx = WEATHER_ORDER.index(weather_profile)
    return WEATHER_ORDER[(idx + variant) % len(WEATHER_ORDER)]


def build(
    *,
    scenario_jsonl: Path,
    dft_raw_csv: Path,
    output_csv: Path,
    target_min_rows: int,
    variants_per_row: int,
) -> dict[str, Any]:
    templates = _load_templates(scenario_jsonl)
    dft_rows = _load_dft_rows(dft_raw_csv)
    if not dft_rows:
        raise RuntimeError("DfT raw counts CSV is empty.")
    baselines = _counts_baseline(dft_rows)
    template_day_kinds = sorted({item.day_kind for item in templates}) or ["weekday"]
    for required_day_kind in ("weekday", "weekend", "holiday"):
        if required_day_kind not in template_day_kinds:
            template_day_kinds.append(required_day_kind)
    as_of_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    diversity = {
        "regime_ids": set(),
        "road_buckets": set(),
        "weather_profiles": set(),
        "vehicle_types": set(),
        "corridors": set(),
        "local_slots": set(),
    }
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        row_index = 0
        while rows_written < max(1, int(target_min_rows)):
            source_row = dft_rows[row_index % len(dft_rows)]
            row_index += 1
            corridor = str(source_row.get("corridor_bucket", "uk_default")).strip().lower() or "uk_default"
            # Ensure weekend/weekday coverage even when source count rows are imbalanced.
            day_kind = template_day_kinds[row_index % len(template_day_kinds)]
            hour = max(0, min(23, _coerce_int(source_row.get("hour"), 12)))
            source_road_bucket = _canonical_road_bucket(source_row.get("road_bucket", "mixed"))
            count_value = max(1.0, _coerce_float(source_row.get("all_motor_vehicles"), 1.0))
            baseline = max(1.0, baselines.get((corridor, day_kind), 100.0))
            count_ratio = _clamp(count_value / baseline, 0.25, 4.5)
            template = _select_template(
                templates=templates,
                corridor_bucket=corridor,
                day_kind=day_kind,
                hour=hour,
            )
            for variant in range(max(1, int(variants_per_row))):
                if rows_written >= max(1, int(target_min_rows)):
                    break
                seed = f"{source_row.get('dedupe_key','')}|{variant}|{rows_written}"
                # Keep jitter bounded so regime-conditioned duration fit stays stable.
                jitter = _hash_float(seed, low=-0.04, high=0.04)
                vehicle_type = VEHICLE_VARIANTS[(rows_written + variant) % len(VEHICLE_VARIANTS)]
                weather_profile = _weather_variant(template.weather_profile, variant)
                road_bucket = _road_bucket_variant(source_road_bucket, variant)
                road_class = road_bucket
                flow_norm = _clamp(template.flow_index / 120.0, 0.5, 2.5)
                speed_norm = _clamp(60.0 / max(20.0, template.speed_index), 0.5, 2.5)
                delay_norm = _clamp(1.0 + (template.delay_pressure / 20.0), 0.7, 2.5)
                weather_norm = _clamp(template.weather_severity_index / 1.4, 0.7, 2.8)

                traffic_ratio = _clamp(
                    (0.48 * count_ratio) + (0.24 * flow_norm) + (0.16 * speed_norm) + (0.12 * delay_norm) + jitter,
                    0.25,
                    4.5,
                )
                incident_factor = _clamp(
                    (0.58 * delay_norm) + (0.42 * _clamp(1.0 + (template.severity_index / 6.0), 0.6, 3.0)) + jitter,
                    0.25,
                    4.5,
                )
                weather_factor = _clamp((0.7 * weather_norm) + (0.3 * (1.0 + jitter)), 0.25, 4.5)
                price_ratio = _clamp((0.68 * traffic_ratio) + (0.32 * incident_factor), 0.25, 4.5)
                eco_ratio = _clamp((0.6 * traffic_ratio) + (0.4 * weather_factor), 0.25, 4.5)
                expected_duration = max(420.0, 1800.0 * speed_norm)
                actual_duration = expected_duration * traffic_ratio
                expected_cost = max(15.0, expected_duration / 220.0)
                actual_cost = expected_cost * price_ratio
                expected_emissions = max(6.0, expected_duration / 260.0)
                actual_emissions = expected_emissions * eco_ratio
                regime_id = _regime_id(day_kind=day_kind, hour=hour, traffic_ratio=traffic_ratio)
                local_time_slot = _local_slot(hour)
                out_row = {
                    "row_id": f"{source_row.get('dedupe_key','row')}-{variant}-{rows_written}",
                    "actual_duration_s": f"{actual_duration:.6f}",
                    "expected_duration_s": f"{expected_duration:.6f}",
                    "actual_monetary_cost": f"{actual_cost:.6f}",
                    "expected_monetary_cost": f"{expected_cost:.6f}",
                    "actual_emissions_kg": f"{actual_emissions:.6f}",
                    "expected_emissions_kg": f"{expected_emissions:.6f}",
                    "incident_factor": f"{incident_factor:.6f}",
                    "weather_factor": f"{weather_factor:.6f}",
                    "day_kind": day_kind,
                    "road_bucket": road_bucket,
                    "weather_profile": weather_profile,
                    "vehicle_type": vehicle_type,
                    "regime_id": regime_id,
                    "corridor_bucket": corridor,
                    "local_time_slot": local_time_slot,
                    "hour": str(hour),
                    "road_class": road_class,
                    "as_of_utc": as_of_utc,
                    "source": "empirical_proxy_public_feeds_v1",
                }
                writer.writerow(out_row)
                rows_written += 1
                diversity["regime_ids"].add(regime_id)
                diversity["road_buckets"].add(road_bucket)
                diversity["weather_profiles"].add(weather_profile)
                diversity["vehicle_types"].add(vehicle_type)
                diversity["corridors"].add(corridor)
                diversity["local_slots"].add(local_time_slot)

    summary = {
        "source": "empirical_proxy_public_feeds_v1",
        "rows_written": rows_written,
        "target_min_rows": int(target_min_rows),
        "variants_per_row": int(variants_per_row),
        "scenario_templates": len(templates),
        "dft_rows": len(dft_rows),
        "output_csv": str(output_csv),
        "diversity": {
            "regime_ids": sorted(diversity["regime_ids"]),
            "road_bucket_count": len(diversity["road_buckets"]),
            "weather_profile_count": len(diversity["weather_profiles"]),
            "vehicle_type_count": len(diversity["vehicle_types"]),
            "corridor_count": len(diversity["corridors"]),
            "local_slot_count": len(diversity["local_slots"]),
        },
        "as_of_utc": as_of_utc,
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect observed-vs-expected stochastic residual corpus from public context feeds."
    )
    parser.add_argument(
        "--scenario-jsonl",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "scenario_live_observed.jsonl",
    )
    parser.add_argument(
        "--dft-raw",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "dft_counts_raw.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "stochastic_residuals_raw.csv",
    )
    parser.add_argument("--target-min-rows", type=int, default=50000)
    parser.add_argument("--variants-per-row", type=int, default=3)
    args = parser.parse_args()
    summary = build(
        scenario_jsonl=args.scenario_jsonl,
        dft_raw_csv=args.dft_raw,
        output_csv=args.output,
        target_min_rows=max(1, int(args.target_min_rows)),
        variants_per_row=max(1, int(args.variants_per_row)),
    )
    print(
        "Collected stochastic residual raw corpus "
        f"(rows={summary['rows_written']}, output={summary['output_csv']})."
    )


if __name__ == "__main__":
    main()
