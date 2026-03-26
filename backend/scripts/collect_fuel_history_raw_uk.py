from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_TOKENS = ("synthetic", "simulated", "interpolated", "wobble", "fake", "bootstrap")
DEFAULT_GOV_UK_FUEL_CSV_URLS = (
    "https://assets.publishing.service.gov.uk/media/69b81ff2b84f01b2be53a2e2/weekly_road_fuel_prices_160326.csv",
    "https://assets.publishing.service.gov.uk/media/68a3326b32d2c63f869343a3/weekly_road_fuel_prices_2003_to_2017.csv",
)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_day(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if len(text) >= 10:
        text = text[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _source_is_synthetic(raw: str) -> bool:
    text = str(raw or "").strip().lower()
    return any(token in text for token in SYNTHETIC_TOKENS)


def _load_local_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Fuel source payload must be a JSON object: {path}")
    return payload


def _load_remote_json(*, url: str, timeout_s: float) -> dict[str, Any]:
    with httpx.Client(timeout=max(2.0, float(timeout_s))) as client:
        resp = client.get(url)
        resp.raise_for_status()
        payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Fuel source payload must be a JSON object: {url}")
    return payload


def _repo_local_energy_defaults(output_json: Path) -> tuple[float, float, dict[str, float]]:
    candidates = [
        output_json,
        ROOT / "assets" / "uk" / "fuel_prices_uk.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = _load_local_json(path)
        except Exception:
            continue
        prices = payload.get("prices_gbp_per_l", {})
        if not isinstance(prices, dict):
            prices = {}
        lng = _coerce_float(prices.get("lng"), float("nan"))
        grid = _coerce_float(payload.get("grid_price_gbp_per_kwh"), float("nan"))
        regional = payload.get("regional_multipliers", {})
        if isinstance(regional, dict):
            regional_out = {
                str(key).strip().lower(): _clamp(_coerce_float(value, 1.0), 0.7, 1.4)
                for key, value in regional.items()
                if str(key).strip()
            }
        else:
            regional_out = {}
        if lng == lng and lng > 0.0 and grid == grid and grid > 0.0:
            if "uk_default" not in regional_out:
                regional_out["uk_default"] = 1.0
            return lng, grid, regional_out
    return 1.05, 0.29, {"uk_default": 1.0}


def _parse_gov_uk_day(raw: Any) -> date | None:
    text = str(raw or "").strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _gov_uk_history_payload(*, url: str, timeout_s: float, output_json: Path) -> dict[str, Any]:
    with httpx.Client(timeout=max(2.0, float(timeout_s))) as client:
        resp = client.get(url)
        resp.raise_for_status()
        text = resp.text
    lng_default, grid_default, regional = _repo_local_energy_defaults(output_json)
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    history: list[dict[str, Any]] = []
    for row in reader:
        if not isinstance(row, dict):
            continue
        normalized = {str(key or "").strip().lstrip("\ufeff"): value for key, value in row.items()}
        day = _parse_gov_uk_day(normalized.get("Date"))
        if day is None:
            continue
        petrol_ppl = _coerce_float(normalized.get("ULSP (Ultra low sulphur unleaded petrol) Pump price in pence/litre"), float("nan"))
        diesel_ppl = _coerce_float(normalized.get("ULSD (Ultra low sulphur diesel) Pump price in pence/litre"), float("nan"))
        if petrol_ppl != petrol_ppl or diesel_ppl != diesel_ppl:
            continue
        history.append(
            {
                "as_of": day.isoformat(),
                "prices_gbp_per_l": {
                    "diesel": round(_clamp(diesel_ppl / 100.0, 0.6, 4.5), 4),
                    "petrol": round(_clamp(petrol_ppl / 100.0, 0.6, 4.5), 4),
                    "lng": round(_clamp(lng_default, 0.4, 4.5), 4),
                },
                "grid_price_gbp_per_kwh": round(_clamp(grid_default, 0.02, 2.0), 4),
            }
        )
    return {
        "source": "gov_uk_weekly_road_fuel_prices",
        "source_url": url,
        "history": history,
        "regional_multipliers": regional,
    }


def _extract_history(payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload.get("history")
    if isinstance(history, list):
        return [item for item in history if isinstance(item, dict)]
    return []


def _extract_point(row: dict[str, Any], *, source_name: str) -> tuple[date, float, float, float, float] | None:
    day = _parse_day(row.get("as_of") or row.get("as_of_utc"))
    if day is None:
        return None
    prices = row.get("prices_gbp_per_l", {})
    if not isinstance(prices, dict):
        prices = {}
    diesel = _coerce_float(prices.get("diesel"), float("nan"))
    petrol = _coerce_float(prices.get("petrol"), float("nan"))
    lng = _coerce_float(prices.get("lng"), float("nan"))
    grid = _coerce_float(row.get("grid_price_gbp_per_kwh"), float("nan"))
    if any(value != value for value in (diesel, petrol, lng, grid)):
        return None
    if any(value <= 0.0 for value in (diesel, petrol, lng, grid)):
        return None
    if _source_is_synthetic(source_name):
        raise RuntimeError(f"Fuel source appears synthetic and is not accepted: {source_name}")
    return (
        day,
        _clamp(diesel, 0.6, 4.5),
        _clamp(petrol, 0.6, 4.5),
        _clamp(lng, 0.4, 4.5),
        _clamp(grid, 0.02, 2.0),
    )


def build(
    *,
    output_json: Path,
    source_urls: list[str],
    source_jsons: list[Path],
    min_days: int,
    timeout_s: float,
) -> dict[str, Any]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    for path in source_jsons:
        payloads.append((str(path), _load_local_json(path)))
    for url in source_urls:
        payloads.append((url, _load_remote_json(url=url, timeout_s=timeout_s)))

    if not payloads:
        for url in DEFAULT_GOV_UK_FUEL_CSV_URLS:
            payloads.append((url, _gov_uk_history_payload(url=url, timeout_s=timeout_s, output_json=output_json)))

    day_to_values: dict[date, dict[str, list[float]]] = {}
    regional_multipliers: dict[str, float] = {}
    for source_name, payload in payloads:
        source = str(payload.get("source", source_name)).strip() or source_name
        if _source_is_synthetic(source):
            raise RuntimeError(f"Fuel source appears synthetic and is not accepted: {source}")
        for row in _extract_history(payload):
            point = _extract_point(row, source_name=source)
            if point is None:
                continue
            day, diesel, petrol, lng, grid = point
            bucket = day_to_values.setdefault(
                day,
                {
                    "diesel": [],
                    "petrol": [],
                    "lng": [],
                    "grid": [],
                },
            )
            bucket["diesel"].append(diesel)
            bucket["petrol"].append(petrol)
            bucket["lng"].append(lng)
            bucket["grid"].append(grid)
        top_level_point = _extract_point(
            {
                "as_of": payload.get("as_of") or payload.get("as_of_utc"),
                "prices_gbp_per_l": payload.get("prices_gbp_per_l"),
                "grid_price_gbp_per_kwh": payload.get("grid_price_gbp_per_kwh"),
            },
            source_name=source,
        )
        if top_level_point is not None:
            day, diesel, petrol, lng, grid = top_level_point
            bucket = day_to_values.setdefault(
                day,
                {
                    "diesel": [],
                    "petrol": [],
                    "lng": [],
                    "grid": [],
                },
            )
            bucket["diesel"].append(diesel)
            bucket["petrol"].append(petrol)
            bucket["lng"].append(lng)
            bucket["grid"].append(grid)

        multipliers = payload.get("regional_multipliers")
        if isinstance(multipliers, dict):
            for key, value in multipliers.items():
                region = str(key).strip().lower()
                if not region:
                    continue
                regional_multipliers[region] = _clamp(_coerce_float(value, 1.0), 0.7, 1.4)

    ordered_days = sorted(day_to_values)
    if len(ordered_days) < max(1, int(min_days)):
        raise RuntimeError(
            f"Fuel history depth too small ({len(ordered_days)} days). "
            f"At least {int(min_days)} daily rows are required."
        )

    history: list[dict[str, Any]] = []
    for day in ordered_days:
        values = day_to_values[day]
        history.append(
            {
                "as_of": day.isoformat(),
                "prices_gbp_per_l": {
                    "diesel": round(statistics.median(values["diesel"]), 4),
                    "petrol": round(statistics.median(values["petrol"]), 4),
                    "lng": round(statistics.median(values["lng"]), 4),
                },
                "grid_price_gbp_per_kwh": round(statistics.median(values["grid"]), 4),
            }
        )

    if "uk_default" not in regional_multipliers:
        regional_multipliers["uk_default"] = 1.0

    now_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    latest = history[-1]
    payload = {
        "source": "public_fuel_history_uk",
        "source_inputs": [name for name, _ in payloads],
        "as_of": f"{latest['as_of']}T00:00:00Z",
        "as_of_utc": f"{latest['as_of']}T00:00:00Z",
        "generated_at_utc": now_utc,
        "prices_gbp_per_l": latest["prices_gbp_per_l"],
        "grid_price_gbp_per_kwh": latest["grid_price_gbp_per_kwh"],
        "history": history,
        "regional_multipliers": regional_multipliers,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {
        "rows": len(history),
        "sources": [name for name, _ in payloads],
        "output_json": str(output_json),
    }
    output_json.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect public fuel history into backend/data/raw/uk/fuel_prices_raw.json."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "fuel_prices_raw.json",
    )
    parser.add_argument("--source-url", action="append", default=[])
    parser.add_argument("--source-json", action="append", type=Path, default=[])
    parser.add_argument("--min-days", type=int, default=1095)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()

    source_urls = [str(item).strip() for item in list(args.source_url or []) if str(item).strip()]
    env_url = os.environ.get("LIVE_FUEL_PRICE_URL", "").strip()
    env_source_policy = os.environ.get("LIVE_SOURCE_POLICY", "repo_local_fresh").strip().lower() or "repo_local_fresh"
    if env_url and env_source_policy == "strict_external" and env_url not in source_urls:
        source_urls.append(env_url)

    summary = build(
        output_json=args.output,
        source_urls=source_urls,
        source_jsons=list(args.source_json or []),
        min_days=max(1, int(args.min_days)),
        timeout_s=max(2.0, float(args.timeout_s)),
    )
    print(f"Collected fuel raw history (rows={summary['rows']}, output={summary['output_json']}).")


if __name__ == "__main__":
    main()
