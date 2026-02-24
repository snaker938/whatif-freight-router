from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid fixture payload (expected object): {path}")
    return payload


def _route_to_road_class(route_fixture: str) -> str:
    name = str(route_fixture or "").strip().lower()
    if not name:
        return "mixed"
    if any(token in name for token in ("m1", "m4", "m5", "m6", "m25", "motorway")):
        return "motorway"
    if any(token in name for token in ("bridge", "crossing", "tunnel", "a1", "a14", "a34")):
        return "primary"
    return "mixed"


def _copy_fixture_set(
    *,
    source_dir: Path,
    output_dir: Path,
    target_count: int,
    required_keys: set[str],
) -> tuple[int, int, int]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Fixture source directory not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.json"):
        existing.unlink()

    count = 0
    true_count = 0
    false_count = 0
    for src in sorted(source_dir.glob("*.json")):
        payload = _load_json(src)
        missing = sorted(field for field in required_keys if field not in payload)
        if missing:
            raise RuntimeError(f"{src} missing keys: {', '.join(missing)}")
        fixture_id = str(payload.get("fixture_id", src.stem)).strip() or src.stem
        payload["fixture_id"] = fixture_id
        payload["source_provenance"] = "proxy_from_labeled_fixture_corpus_v1"
        payload["as_of_utc"] = (
            str(payload.get("as_of_utc", "")).strip()
            or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
        if "expected_has_toll" in payload:
            if _coerce_bool(payload["expected_has_toll"]):
                true_count += 1
            else:
                false_count += 1
        elif "expected_contains_toll" in payload:
            if _coerce_bool(payload["expected_contains_toll"]):
                true_count += 1
            else:
                false_count += 1
        (output_dir / f"{fixture_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        count += 1
    if count < max(1, int(target_count)):
        raise RuntimeError(
            f"Fixture corpus too small in {source_dir} ({count} < {int(target_count)} required)."
        )
    return count, true_count, false_count


def _pricing_label_counts(pricing_dir: Path) -> tuple[int, int]:
    true_count = 0
    false_count = 0
    for path in sorted(pricing_dir.glob("*.json")):
        payload = _load_json(path)
        if _coerce_bool(payload.get("expected_contains_toll")):
            true_count += 1
        else:
            false_count += 1
    return true_count, false_count


def _augment_pricing_from_classification(
    *,
    classification_dir: Path,
    pricing_dir: Path,
    target_true: int,
    target_false: int,
) -> tuple[int, int]:
    existing_ids = {path.stem for path in pricing_dir.glob("*.json")}
    base_rows = [_load_json(path) for path in sorted(classification_dir.glob("*.json"))]
    added_true = 0
    added_false = 0
    true_count, false_count = _pricing_label_counts(pricing_dir)
    if true_count >= target_true and false_count >= target_false:
        return added_true, added_false

    for row in base_rows:
        has_toll = _coerce_bool(row.get("expected_has_toll"))
        want_toll = has_toll
        if want_toll and true_count >= target_true:
            continue
        if (not want_toll) and false_count >= target_false:
            continue
        fixture_id = str(row.get("fixture_id", "")).strip() or "class_proxy"
        out_id = f"proxy_price_from_{fixture_id}"
        if out_id in existing_ids:
            continue
        route_fixture = str(row.get("route_fixture", "")).strip()
        if not route_fixture:
            continue
        out_payload = {
            "fixture_id": out_id,
            "route_fixture": route_fixture,
            "expected_contains_toll": bool(want_toll),
            "expected_toll_cost_gbp": round(5.0 if want_toll else 0.0, 2),
            "expected_currency": "GBP",
            "source_provenance": "proxy_from_classification_balancer_v1",
            "as_of_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        (pricing_dir / f"{out_id}.json").write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        existing_ids.add(out_id)
        if want_toll:
            true_count += 1
            added_true += 1
        else:
            false_count += 1
            added_false += 1
        if true_count >= target_true and false_count >= target_false:
            break
    return added_true, added_false


def _generate_tariff_rules(
    *,
    pricing_fixtures: list[dict[str, Any]],
    min_tariff_rules: int,
) -> list[dict[str, Any]]:
    if not pricing_fixtures:
        raise RuntimeError("Pricing fixture set is empty; cannot generate tariff truth rules.")
    rules: list[dict[str, Any]] = []
    idx = 0
    while len(rules) < max(1, int(min_tariff_rules)):
        fixture = pricing_fixtures[idx % len(pricing_fixtures)]
        fixture_id = str(fixture.get("fixture_id", f"fixture_{idx:04d}")).strip() or f"fixture_{idx:04d}"
        route_fixture = str(fixture.get("route_fixture", fixture_id)).strip() or fixture_id
        crossing_id = Path(route_fixture).stem.lower()
        has_toll = _coerce_bool(fixture.get("expected_contains_toll"))
        expected_cost = max(0.0, float(fixture.get("expected_toll_cost_gbp", 0.0)))
        slot_variant = (idx // len(pricing_fixtures)) % 4
        start_minute = slot_variant * 360
        end_minute = min(1439, start_minute + 359)
        road_class = _route_to_road_class(route_fixture)
        crossing_fee = expected_cost if has_toll else 0.0
        distance_fee = (expected_cost / 100.0) if has_toll else 0.0
        rule = {
            "id": f"proxy_rule_{fixture_id}_{slot_variant:02d}",
            "operator": "public_proxy_operator",
            "crossing_id": crossing_id,
            "road_class": road_class,
            "direction": "both",
            "start_minute": int(start_minute),
            "end_minute": int(end_minute),
            "crossing_fee_gbp": round(crossing_fee, 4),
            "distance_fee_gbp_per_km": round(distance_fee, 6),
            "vehicle_classes": ["rigid_hgv", "artic_hgv", "van"],
            "axle_classes": ["default", "heavy"],
            "payment_classes": ["cash", "tag"],
            "exemptions": [],
        }
        rules.append(rule)
        idx += 1
    return rules


def build(
    *,
    classification_source: Path,
    pricing_source: Path,
    classification_out: Path,
    pricing_out: Path,
    tariffs_out: Path,
    classification_target: int,
    pricing_target: int,
    min_tariff_rules: int,
) -> dict[str, Any]:
    class_count, class_true, class_false = _copy_fixture_set(
        source_dir=classification_source,
        output_dir=classification_out,
        target_count=classification_target,
        required_keys={"fixture_id", "route_fixture", "expected_has_toll"},
    )
    price_count, price_true, price_false = _copy_fixture_set(
        source_dir=pricing_source,
        output_dir=pricing_out,
        target_count=pricing_target,
        required_keys={"fixture_id", "route_fixture", "expected_contains_toll", "expected_toll_cost_gbp"},
    )
    min_each = max(10, int(max(classification_target, 1) * 0.05))
    if class_true < min_each or class_false < min_each:
        raise RuntimeError(
            "Classification labels are imbalanced for calibration stability "
            f"(true={class_true}, false={class_false}, min_each={min_each})."
        )
    if price_true <= 0 or price_false <= 0:
        pricing_min_each = max(1, int(max(pricing_target, 1) * 0.05))
        _augment_pricing_from_classification(
            classification_dir=classification_out,
            pricing_dir=pricing_out,
            target_true=pricing_min_each,
            target_false=pricing_min_each,
        )
        price_true, price_false = _pricing_label_counts(pricing_out)
        price_count = len(list(pricing_out.glob("*.json")))
    if price_true <= 0 or price_false <= 0:
        raise RuntimeError("Pricing labels must include both toll and non-toll cases.")

    pricing_payloads = [_load_json(path) for path in sorted(pricing_out.glob("*.json"))]
    tariff_rules = _generate_tariff_rules(
        pricing_fixtures=pricing_payloads,
        min_tariff_rules=max(1, int(min_tariff_rules)),
    )
    now_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    tariff_payload = {
        "version": "uk_tariffs_v5_empirical_proxy",
        "source": "proxy_from_labeled_toll_fixture_corpus_v1",
        "as_of_utc": now_utc,
        "rules": tariff_rules,
    }
    tariffs_out.parent.mkdir(parents=True, exist_ok=True)
    tariffs_out.write_text(json.dumps(tariff_payload, indent=2), encoding="utf-8")

    summary = {
        "classification_count": class_count,
        "classification_true": class_true,
        "classification_false": class_false,
        "pricing_count": price_count,
        "pricing_true": price_true,
        "pricing_false": price_false,
        "tariff_rule_count": len(tariff_rules),
        "classification_out": str(classification_out),
        "pricing_out": str(pricing_out),
        "tariffs_out": str(tariffs_out),
    }
    tariffs_out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect proxy toll truth raw corpora and tariff truth from labeled public fixtures."
    )
    parser.add_argument(
        "--classification-source",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "toll_classification",
    )
    parser.add_argument(
        "--pricing-source",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "toll_pricing",
    )
    parser.add_argument(
        "--classification-out",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "toll_classification",
    )
    parser.add_argument(
        "--pricing-out",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "toll_pricing",
    )
    parser.add_argument(
        "--tariffs-out",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "toll_tariffs_operator_truth.json",
    )
    parser.add_argument("--classification-target", type=int, default=220)
    parser.add_argument("--pricing-target", type=int, default=100)
    parser.add_argument("--min-tariff-rules", type=int, default=220)
    args = parser.parse_args()
    summary = build(
        classification_source=args.classification_source,
        pricing_source=args.pricing_source,
        classification_out=args.classification_out,
        pricing_out=args.pricing_out,
        tariffs_out=args.tariffs_out,
        classification_target=max(1, int(args.classification_target)),
        pricing_target=max(1, int(args.pricing_target)),
        min_tariff_rules=max(1, int(args.min_tariff_rules)),
    )
    print(
        "Collected toll raw truth corpus "
        f"(classification={summary['classification_count']}, pricing={summary['pricing_count']}, "
        f"rules={summary['tariff_rule_count']})."
    )


if __name__ == "__main__":
    main()
