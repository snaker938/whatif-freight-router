from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _load_intensity_source(path: Path) -> tuple[dict[str, list[float]], str | None, str | None]:
    if not path.exists():
        raise FileNotFoundError(f"Carbon intensity source is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Carbon intensity source must be a JSON object.")
    rows = payload.get("regions", payload.get("ev_grid_intensity_kg_per_kwh_by_region"))
    if not isinstance(rows, dict):
        raise RuntimeError("Carbon intensity source must include regional hourly profiles.")
    out: dict[str, list[float]] = {}
    for region, values in rows.items():
        if not isinstance(values, list):
            continue
        parsed = []
        for value in values[:24]:
            try:
                parsed.append(_clamp(float(value), 0.02, 2.5))
            except (TypeError, ValueError):
                parsed = []
                break
        if len(parsed) == 24:
            out[str(region)] = parsed
    if not out:
        raise RuntimeError("Carbon intensity source had no valid 24-hour regional profiles.")
    if "uk_default" not in out:
        first = next(iter(out.values()))
        out["uk_default"] = list(first)
    as_of_utc = str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None
    source = str(payload.get("source", "")).strip() or None
    return out, as_of_utc, source


def _load_schedule(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Carbon price schedule source is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Carbon schedule source must be a JSON object.")
    prices = payload.get("prices_gbp_per_kg", {})
    if not isinstance(prices, dict):
        raise RuntimeError("Carbon schedule source is missing prices_gbp_per_kg.")
    for scenario in ("central", "high", "low"):
        scenario_rows = prices.get(scenario)
        if not isinstance(scenario_rows, dict) or not scenario_rows:
            raise RuntimeError(f"Carbon schedule source missing scenario curve: {scenario}")
    return payload


def _augment_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    prices = schedule.get("prices_gbp_per_kg", {})
    central = prices.get("central", {}) if isinstance(prices, dict) else {}
    if not isinstance(central, dict):
        raise RuntimeError("Carbon schedule central curve is invalid.")
    out = dict(schedule)
    out["uncertainty_distribution_by_year"] = {}
    for year_raw, value_raw in central.items():
        try:
            year = int(year_raw)
            value = max(0.0, float(value_raw))
        except (TypeError, ValueError):
            continue
        # Use asymmetric confidence intervals derived from scenario spread.
        high = prices.get("high", {}) if isinstance(prices.get("high"), dict) else {}
        low = prices.get("low", {}) if isinstance(prices.get("low"), dict) else {}
        try:
            p10 = max(0.0, float(low.get(str(year), value * 0.9)))
            p90 = max(p10, float(high.get(str(year), value * 1.1)))
        except (TypeError, ValueError):
            p10 = max(0.0, value * 0.9)
            p90 = max(p10, value * 1.1)
        sigma = max(0.001, (p90 - p10) / 2.5632)
        out["uncertainty_distribution_by_year"][str(year)] = {
            "p10": round(p10, 6),
            "p50": round(value, 6),
            "p90": round(p90, 6),
            "sigma": round(sigma, 6),
        }
    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    out["generated_at_utc"] = now_iso
    out["as_of_utc"] = str(out.get("as_of_utc", "")).strip() or now_iso
    out["source"] = str(out.get("source", "")).strip() or "uk_policy_and_grid_empirical"
    out["version"] = str(out.get("version", "")).strip() or "uk-carbon-schedule-v3"
    return out


def build_intensity_asset(*, source_json: Path, output_json: Path) -> dict[str, Any]:
    regions, source_as_of, source_label = _load_intensity_source(source_json)
    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = {
        "version": "uk-carbon-intensity-v3",
        "source": source_label or str(source_json),
        "generated_at_utc": now_iso,
        "as_of_utc": source_as_of or now_iso,
        "regions": regions,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def augment_carbon_schedule(*, schedule_json: Path) -> dict[str, Any]:
    schedule = _load_schedule(schedule_json)
    out = _augment_schedule(schedule)
    schedule_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize empirical UK carbon schedule and intensity assets.")
    parser.add_argument(
        "--intensity-source",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "carbon_intensity_hourly_raw.json",
        help="Raw empirical regional-hourly carbon intensity JSON.",
    )
    parser.add_argument(
        "--intensity-output",
        type=Path,
        default=ROOT / "assets" / "uk" / "carbon_intensity_hourly_uk.json",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=ROOT / "assets" / "uk" / "carbon_price_schedule_uk.json",
    )
    args = parser.parse_args()
    intensity = build_intensity_asset(source_json=args.intensity_source, output_json=args.intensity_output)
    schedule = augment_carbon_schedule(schedule_json=args.schedule)
    print(
        f"Wrote intensity asset ({len(intensity.get('regions', {}))} regions) to {args.intensity_output} "
        f"and updated schedule with {len(schedule.get('uncertainty_distribution_by_year', {}))} yearly distributions."
    )


if __name__ == "__main__":
    main()
