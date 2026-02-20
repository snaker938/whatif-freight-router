from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object: {path}")
    return payload


def _copy_fixture_set(*, source_dir: Path, output_dir: Path, required_keys: set[str]) -> int:
    if not source_dir.exists():
        raise FileNotFoundError(f"Labeled toll fixture source directory is required: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.json"):
        existing.unlink()
    count = 0
    for src in sorted(source_dir.glob("*.json")):
        payload = _read_json(src)
        missing = sorted(key for key in required_keys if key not in payload)
        if missing:
            raise RuntimeError(f"{src} is missing required keys: {', '.join(missing)}")
        fixture_id = str(payload.get("fixture_id", "")).strip() or src.stem
        payload["fixture_id"] = fixture_id
        payload["as_of_utc"] = str(payload.get("as_of_utc", "")).strip() or datetime.now(UTC).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z")
        (output_dir / f"{fixture_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        count += 1
    return count


def _fit_confidence_calibration(*, classification_dir: Path, calibration_out_json: Path) -> None:
    samples: list[tuple[list[float], float, float, float]] = []
    for path in sorted(classification_dir.glob("*.json")):
        payload = _read_json(path)
        label = 1.0 if bool(payload.get("expected_has_toll", False)) else 0.0
        reason = str(payload.get("expected_reason", "")).strip().lower()
        class_signal = 1.0 if ("class" in reason or bool(payload.get("class_signal", False))) else 0.0
        seed_signal = 1.0 if (
            "seed" in reason
            or "topology" in reason
            or "overlap" in reason
            or bool(payload.get("seed_signal", False))
        ) else 0.0
        raw_weight = payload.get("weight", 1.0)
        try:
            seg_signal = float(raw_weight)
        except (TypeError, ValueError):
            seg_signal = 1.0
        # Normalize arbitrary fixture weights into [0, 1] segment-strength signal.
        seg_signal = max(0.0, min(1.0, (seg_signal - 0.5) / 1.5))
        samples.append(([1.0, class_signal, seed_signal, seg_signal], label, class_signal, seed_signal))
    if not samples:
        raise RuntimeError("Cannot fit toll confidence calibration without labeled classification fixtures.")

    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    # Fit a compact logistic model from labeled fixtures.
    n = float(len(samples))
    w = [0.0, 0.0, 0.0, 0.0]  # intercept, class_signal, seed_signal, segment_signal
    lr = 0.08
    l2 = 0.02
    for _ in range(500):
        grad = [0.0, 0.0, 0.0, 0.0]
        for features, label, _class_signal, _seed_signal in samples:
            logit = sum(weight * feat for weight, feat in zip(w, features, strict=True))
            p = _sigmoid(logit)
            err = p - label
            for idx in range(4):
                grad[idx] += err * features[idx]
        for idx in range(1, 4):
            grad[idx] += l2 * w[idx]
        for idx in range(4):
            w[idx] -= lr * (grad[idx] / n)

    prevalence = sum(label for _x, label, _c, _s in samples) / n
    prevalence = min(0.98, max(0.02, prevalence))
    class_labels = [label for _x, label, class_signal, _seed in samples if class_signal > 0.5]
    both_labels = [label for _x, label, class_signal, seed_signal in samples if class_signal > 0.5 and seed_signal > 0.5]
    class_rate = (sum(class_labels) / len(class_labels)) if class_labels else prevalence
    both_rate = (sum(both_labels) / len(both_labels)) if both_labels else class_rate
    bonus_class = max(0.0, min(0.25, class_rate - prevalence))
    bonus_both = max(0.0, min(0.35, both_rate - prevalence))

    pred_rows: list[tuple[float, float]] = []
    for features, label, _class_signal, _seed_signal in samples:
        p = _sigmoid(sum(weight * feat for weight, feat in zip(w, features, strict=True)))
        pred_rows.append((p, label))
    bin_edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    reliability_bins: list[dict[str, float]] = []
    for lo, hi in bin_edges:
        labels = [label for p, label in pred_rows if lo <= p <= hi]
        if labels:
            calibrated = sum(labels) / len(labels)
        else:
            calibrated = (lo + hi) * 0.5
        reliability_bins.append(
            {
                "min": round(lo, 2),
                "max": round(hi, 2),
                "calibrated": round(max(0.0, min(1.0, calibrated)), 4),
            }
        )

    calibration_payload = {
        "version": "uk-toll-confidence-v2-empirical",
        "source": str(classification_dir),
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "as_of_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "logit_model": {
            "intercept": round(float(w[0]), 6),
            "class_signal": round(float(w[1]), 6),
            "seed_signal": round(float(w[2]), 6),
            "segment_signal": round(float(w[3]), 6),
            "source_bonus_both": round(bonus_both, 6),
            "source_bonus_class": round(bonus_class, 6),
        },
        "reliability_bins": reliability_bins,
    }
    calibration_out_json.parent.mkdir(parents=True, exist_ok=True)
    calibration_out_json.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")


def build(
    *,
    classification_source_dir: Path,
    pricing_source_dir: Path,
    classification_out_dir: Path,
    pricing_out_dir: Path,
    classification_target: int = 200,
    pricing_target: int = 80,
    calibration_out_json: Path | None = None,
) -> tuple[int, int]:
    class_rows = _copy_fixture_set(
        source_dir=classification_source_dir,
        output_dir=classification_out_dir,
        required_keys={"fixture_id", "route_fixture", "expected_has_toll"},
    )
    price_rows = _copy_fixture_set(
        source_dir=pricing_source_dir,
        output_dir=pricing_out_dir,
        required_keys={"fixture_id", "route_fixture", "expected_contains_toll", "expected_toll_cost_gbp"},
    )
    if class_rows < max(1, int(classification_target)):
        raise RuntimeError(
            f"Classification fixture corpus too small ({class_rows}). "
            f"At least {int(classification_target)} empirical labels are required."
        )
    if price_rows < max(1, int(pricing_target)):
        raise RuntimeError(
            f"Pricing fixture corpus too small ({price_rows}). "
            f"At least {int(pricing_target)} empirical labels are required."
        )
    if calibration_out_json is not None:
        _fit_confidence_calibration(
            classification_dir=classification_out_dir,
            calibration_out_json=calibration_out_json,
        )
    return class_rows, price_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest labeled toll truth corpora and fit confidence calibration.")
    parser.add_argument(
        "--classification-source",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "toll_classification",
        help="Source directory containing labeled toll/no-toll fixtures.",
    )
    parser.add_argument(
        "--pricing-source",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "toll_pricing",
        help="Source directory containing labeled toll pricing fixtures.",
    )
    parser.add_argument(
        "--classification-out",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "toll_classification",
    )
    parser.add_argument(
        "--pricing-out",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "toll_pricing",
    )
    parser.add_argument(
        "--classification-target",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--pricing-target",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=ROOT / "assets" / "uk" / "toll_confidence_calibration_uk.json",
    )
    args = parser.parse_args()
    class_rows, price_rows = build(
        classification_source_dir=args.classification_source,
        pricing_source_dir=args.pricing_source,
        classification_out_dir=args.classification_out,
        pricing_out_dir=args.pricing_out,
        classification_target=max(1, int(args.classification_target)),
        pricing_target=max(1, int(args.pricing_target)),
        calibration_out_json=args.calibration_out,
    )
    print(
        f"Ingested {class_rows} classification labels and {price_rows} pricing labels. "
        f"Calibration => {args.calibration_out}"
    )


if __name__ == "__main__":
    main()
