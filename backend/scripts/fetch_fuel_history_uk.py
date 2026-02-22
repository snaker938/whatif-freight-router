from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FuelPoint:
    day: date
    diesel: float
    petrol: float
    lng: float
    grid: float


def _parse_day(raw: str) -> date:
    text = str(raw).strip()
    if len(text) >= 10:
        text = text[:10]
    return date.fromisoformat(text)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _signature_material(payload: dict[str, Any]) -> str:
    signed_subset = {
        "as_of_utc": payload.get("as_of_utc", payload.get("as_of")),
        "source": payload.get("source"),
        "prices_gbp_per_l": payload.get("prices_gbp_per_l"),
        "grid_price_gbp_per_kwh": payload.get("grid_price_gbp_per_kwh"),
        "history": payload.get("history"),
        "regional_multipliers": payload.get("regional_multipliers"),
    }
    return json.dumps(signed_subset, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_signature_material(payload).encode("utf-8")).hexdigest()


def _load_history_payload(path: Path) -> tuple[list[FuelPoint], dict[str, float], str, str | None]:
    if not path.exists():
        raise FileNotFoundError(f"Fuel history source file is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Fuel history source payload must be a JSON object.")
    history = payload.get("history", [])
    if not isinstance(history, list) or not history:
        raise RuntimeError("Fuel history source payload must contain non-empty 'history'.")

    points: list[FuelPoint] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        prices = row.get("prices_gbp_per_l", {})
        if not isinstance(prices, dict):
            continue
        try:
            points.append(
                FuelPoint(
                    day=_parse_day(str(row.get("as_of", ""))),
                    diesel=float(prices.get("diesel")),
                    petrol=float(prices.get("petrol")),
                    lng=float(prices.get("lng")),
                    grid=float(row.get("grid_price_gbp_per_kwh")),
                )
            )
        except (TypeError, ValueError):
            continue
    if not points:
        raise RuntimeError("Fuel history source payload had no parseable rows.")

    regional = payload.get("regional_multipliers", {})
    if not isinstance(regional, dict):
        regional = {"uk_default": 1.0}
    regional_out: dict[str, float] = {}
    for key, value in regional.items():
        try:
            regional_out[str(key)] = _clamp(float(value), 0.7, 1.4)
        except (TypeError, ValueError):
            continue
    if "uk_default" not in regional_out:
        regional_out["uk_default"] = 1.0

    source = str(payload.get("source", "")).strip() or "empirical_fuel_history_source"
    lowered_source = source.lower()
    if any(token in lowered_source for token in ("synthetic", "interpolated", "wobble", "simulated")):
        raise RuntimeError(
            "Fuel history source appears synthetic/interpolated; strict runtime requires observed records."
        )
    as_of = str(payload.get("as_of", "")).strip() or None
    return sorted(points, key=lambda p: p.day), regional_out, source, as_of


def build(
    *,
    source_json: Path,
    output_json: Path,
    min_history_days: int = 365,
) -> dict[str, Any]:
    points, regional, source, source_as_of = _load_history_payload(source_json)
    if len(points) < max(1, int(min_history_days)):
        raise RuntimeError(
            f"Fuel history depth too small ({len(points)} rows). "
            f"At least {int(min_history_days)} daily rows are required for strict runtime."
        )
    latest = points[-1]
    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "as_of": source_as_of or f"{latest.day.isoformat()}T00:00:00Z",
        "as_of_utc": source_as_of or f"{latest.day.isoformat()}T00:00:00Z",
        "refreshed_at_utc": now_iso,
        "source": source,
        "provider_contract_version": "fuel-live-v1",
        "prices_gbp_per_l": {
            "diesel": round(latest.diesel, 4),
            "petrol": round(latest.petrol, 4),
            "lng": round(latest.lng, 4),
        },
        "grid_price_gbp_per_kwh": round(latest.grid, 4),
        "history": [
            {
                "as_of": row.day.isoformat(),
                "prices_gbp_per_l": {
                    "diesel": round(_clamp(row.diesel, 0.6, 4.5), 4),
                    "petrol": round(_clamp(row.petrol, 0.6, 4.5), 4),
                    "lng": round(_clamp(row.lng, 0.4, 4.5), 4),
                },
                "grid_price_gbp_per_kwh": round(_clamp(row.grid, 0.02, 2.0), 4),
            }
            for row in points
        ],
        "regional_multipliers": regional,
    }
    payload["signature_algorithm"] = "sha256"
    payload["signature"] = _signature(payload)
    payload["signed"] = True
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize empirical UK fuel history payload for runtime.")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "fuel_prices_raw.json",
        help="Raw empirical fuel history JSON payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "uk" / "fuel_prices_uk.json",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=365,
        help="Strict minimum number of daily rows in source history.",
    )
    args = parser.parse_args()
    payload = build(
        source_json=args.source,
        output_json=args.output,
        min_history_days=max(1, int(args.min_days)),
    )
    print(f"Wrote {len(payload.get('history', []))} empirical fuel rows to {args.output}")


if __name__ == "__main__":
    main()
