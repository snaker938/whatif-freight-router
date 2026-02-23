from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_RULE_FIELDS = {
    "id",
    "operator",
    "crossing_id",
    "road_class",
    "direction",
    "start_minute",
    "end_minute",
    "crossing_fee_gbp",
    "distance_fee_gbp_per_km",
    "vehicle_classes",
    "axle_classes",
    "payment_classes",
    "exemptions",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object: {path}")
    return payload


def _normalize_rule(row: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(field for field in REQUIRED_RULE_FIELDS if field not in row)
    if missing:
        raise RuntimeError(
            f"Tariff rule '{row.get('id', '<missing-id>')}' missing required fields: {', '.join(missing)}"
        )
    try:
        start_minute = int(row["start_minute"])
        end_minute = int(row["end_minute"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid minute window for tariff rule '{row.get('id', '<missing-id>')}'") from exc
    return {
        "id": str(row["id"]).strip(),
        "operator": str(row["operator"]).strip().lower(),
        "crossing_id": str(row["crossing_id"]).strip().lower(),
        "road_class": str(row["road_class"]).strip().lower(),
        "direction": str(row["direction"]).strip().lower(),
        "start_minute": max(0, min(1439, start_minute)),
        "end_minute": max(0, min(1439, end_minute)),
        "crossing_fee_gbp": max(0.0, float(row["crossing_fee_gbp"])),
        "distance_fee_gbp_per_km": max(0.0, float(row["distance_fee_gbp_per_km"])),
        "vehicle_classes": [str(item).strip().lower() for item in list(row["vehicle_classes"])],
        "axle_classes": [str(item).strip().lower() for item in list(row["axle_classes"])],
        "payment_classes": [str(item).strip().lower() for item in list(row["payment_classes"])],
        "exemptions": [str(item).strip().lower() for item in list(row["exemptions"])],
    }


def _load_tariff_truth(path: Path, *, min_rules: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Operator-grade tariff truth file is required for strict build: {path}"
        )
    payload = _load_json(path)
    raw_rules = payload.get("rules", [])
    if not isinstance(raw_rules, list) or not raw_rules:
        raise RuntimeError("Tariff truth payload must include non-empty 'rules'.")
    rules: list[dict[str, Any]] = []
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        rules.append(_normalize_rule(item))
    if len(rules) < max(1, int(min_rules)):
        raise RuntimeError(
            f"Tariff truth depth too small ({len(rules)} rules). "
            f"At least {int(min_rules)} rules are required for strict runtime."
        )
    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "version": str(payload.get("version", "")).strip() or "uk_tariffs_v5_empirical",
        "source": str(payload.get("source", "")).strip() or str(path),
        "generated_at_utc": now_iso,
        "as_of_utc": str(payload.get("as_of_utc", "")).strip() or now_iso,
        "defaults": {
            "crossing_fee_gbp": 0.0,
            "distance_fee_gbp_per_km": 0.0,
        },
        "rules": rules,
    }


def build(
    *,
    fuel_source: Path,
    carbon_source: Path,
    tariff_truth_source: Path,
    toll_tariffs_output: Path,
    output_dir: Path,
    min_tariff_rules: int = 200,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fuel_payload = _load_json(fuel_source)
    carbon_payload = _load_json(carbon_source)
    tariffs_payload = _load_tariff_truth(tariff_truth_source, min_rules=max(1, int(min_tariff_rules)))

    (output_dir / "fuel_prices_uk_compiled.json").write_text(
        json.dumps(fuel_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "carbon_price_schedule_uk.json").write_text(
        json.dumps(carbon_payload, indent=2),
        encoding="utf-8",
    )
    toll_tariffs_output.parent.mkdir(parents=True, exist_ok=True)
    toll_tariffs_output.write_text(json.dumps(tariffs_payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict UK pricing tables from empirical tariff truth.")
    parser.add_argument(
        "--fuel-source",
        type=Path,
        default=ROOT / "assets" / "uk" / "fuel_prices_uk.json",
    )
    parser.add_argument(
        "--carbon-source",
        type=Path,
        default=ROOT / "assets" / "uk" / "carbon_price_schedule_uk.json",
    )
    parser.add_argument(
        "--tariff-truth-source",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "toll_tariffs_operator_truth.json",
    )
    parser.add_argument(
        "--toll-tariffs-output",
        type=Path,
        default=ROOT / "assets" / "uk" / "toll_tariffs_uk.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "out" / "model_assets",
    )
    parser.add_argument(
        "--min-tariff-rules",
        type=int,
        default=200,
    )
    args = parser.parse_args()
    build(
        fuel_source=args.fuel_source,
        carbon_source=args.carbon_source,
        tariff_truth_source=args.tariff_truth_source,
        toll_tariffs_output=args.toll_tariffs_output,
        output_dir=args.output_dir,
        min_tariff_rules=max(1, int(args.min_tariff_rules)),
    )
    print(
        f"Wrote strict pricing assets in {args.output_dir}; "
        f"tariffs => {args.toll_tariffs_output}"
    )


if __name__ == "__main__":
    main()
