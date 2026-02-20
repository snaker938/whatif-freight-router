from __future__ import annotations

import argparse
import json
import csv
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_REGIMES = {
    "weekday_peak": {"sigma_scale": 1.18, "traffic_scale": 1.25, "incident_scale": 1.20, "weather_scale": 1.05, "price_scale": 1.00, "eco_scale": 1.00},
    "weekday_offpeak": {"sigma_scale": 0.92, "traffic_scale": 0.90, "incident_scale": 0.95, "weather_scale": 1.00, "price_scale": 1.00, "eco_scale": 1.00},
    "weekend": {"sigma_scale": 0.86, "traffic_scale": 0.82, "incident_scale": 0.90, "weather_scale": 0.98, "price_scale": 1.00, "eco_scale": 1.00},
    "holiday": {"sigma_scale": 0.94, "traffic_scale": 0.88, "incident_scale": 0.95, "weather_scale": 1.02, "price_scale": 1.00, "eco_scale": 1.00},
}

ROAD_MOD = {
    "motorway_heavy": {"sigma_scale": 0.95, "traffic_scale": 0.92},
    "trunk_heavy": {"sigma_scale": 1.00, "traffic_scale": 1.00},
    "mixed": {"sigma_scale": 1.08, "traffic_scale": 1.12},
}

WEATHER_MOD = {
    "clear": {"weather_scale": 0.96, "incident_scale": 0.95},
    "rain": {"weather_scale": 1.16, "incident_scale": 1.12},
    "storm": {"weather_scale": 1.32, "incident_scale": 1.24},
    "snow": {"weather_scale": 1.42, "incident_scale": 1.28},
    "fog": {"weather_scale": 1.18, "incident_scale": 1.10},
}

CORR = [
    [1.0, 0.52, 0.40, 0.24, 0.17],
    [0.52, 1.0, 0.31, 0.33, 0.22],
    [0.40, 0.31, 1.0, 0.30, 0.35],
    [0.24, 0.33, 0.30, 1.0, 0.19],
    [0.17, 0.22, 0.35, 0.19, 1.0],
]


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = _clamp(float(q), 0.0, 1.0)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    t = pos - lo
    return ordered[lo] + ((ordered[hi] - ordered[lo]) * t)


def _merge(base: dict[str, float], mod_a: dict[str, float], mod_b: dict[str, float]) -> dict[str, Any]:
    out = dict(base)
    for key, value in mod_a.items():
        out[key] = max(0.10, float(out.get(key, 1.0)) * float(value))
    for key, value in mod_b.items():
        out[key] = max(0.10, float(out.get(key, 1.0)) * float(value))
    out["spread_floor"] = 0.03
    out["spread_cap"] = 0.38
    out["factor_low"] = 0.60
    out["factor_high"] = 1.95
    out["duration_mix"] = [0.56, 0.29, 0.15]
    out["monetary_mix"] = [0.72, 0.28]
    out["emissions_mix"] = [0.84, 0.16]
    out["corr"] = CORR
    return out


def _corr_from_samples(rows: list[tuple[float, float, float, float, float]]) -> list[list[float]]:
    if len(rows) < 8:
        raise RuntimeError(
            "Each stochastic regime requires at least 8 residual samples for strict calibration."
        )
    means = [0.0 for _ in range(5)]
    for row in rows:
        for i in range(5):
            means[i] += row[i]
    means = [m / len(rows) for m in means]
    cov = [[0.0 for _ in range(5)] for _ in range(5)]
    for row in rows:
        centered = [row[i] - means[i] for i in range(5)]
        for i in range(5):
            for j in range(5):
                cov[i][j] += centered[i] * centered[j]
    denom = max(1, len(rows) - 1)
    for i in range(5):
        for j in range(5):
            cov[i][j] /= denom
    std = [math.sqrt(max(1e-12, cov[i][i])) for i in range(5)]
    corr = [[0.0 for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            corr[i][j] = max(-0.999, min(0.999, cov[i][j] / max(1e-12, std[i] * std[j])))
    for i in range(5):
        corr[i][i] = 1.0
    return corr


def _normalized_triplet(a: float, b: float, c: float, *, fallback: tuple[float, float, float]) -> list[float]:
    total = max(1e-9, a + b + c)
    if total <= 1e-9:
        return [float(fallback[0]), float(fallback[1]), float(fallback[2])]
    return [a / total, b / total, c / total]


def _build_from_residuals(residuals_csv: Path) -> dict[str, dict[str, Any]]:
    # Expected columns:
    # regime_id,traffic,incident,weather,price,eco,sigma(optional)
    grouped: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    sigma_by_regime: dict[str, list[float]] = defaultdict(list)
    with residuals_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            regime = str(row.get("regime_id", "default")).strip() or "default"
            sample = (
                float(row.get("traffic", 1.0)),
                float(row.get("incident", 1.0)),
                float(row.get("weather", 1.0)),
                float(row.get("price", 1.0)),
                float(row.get("eco", 1.0)),
            )
            grouped[regime].append(sample)
            sigma_raw = row.get("sigma")
            if sigma_raw not in (None, ""):
                sigma_by_regime[regime].append(float(sigma_raw))

    regimes: dict[str, dict[str, Any]] = {}
    for regime_id, rows in grouped.items():
        if not rows:
            continue
        if len(rows) < 8:
            raise RuntimeError(
                f"Regime '{regime_id}' has insufficient residual coverage ({len(rows)} < 8)."
            )
        means = [sum(sample[idx] for sample in rows) / len(rows) for idx in range(5)]
        corr = _corr_from_samples(rows)
        duration_samples = [
            _clamp((0.56 * sample[0]) + (0.29 * sample[1]) + (0.15 * sample[2]), 0.35, 3.0)
            for sample in rows
        ]
        spreads = [abs(math.log(max(0.05, sample))) for sample in duration_samples]
        sigma_scale = (
            sum(sigma_by_regime.get(regime_id, [])) / max(1, len(sigma_by_regime.get(regime_id, [])))
            if sigma_by_regime.get(regime_id)
            else 1.0
        )
        duration_mix = _normalized_triplet(
            abs(means[0] - 1.0) + 0.15,
            abs(means[1] - 1.0) + 0.10,
            abs(means[2] - 1.0) + 0.08,
            fallback=(0.56, 0.29, 0.15),
        )
        monetary_mix = _normalized_triplet(
            abs(means[0] - 1.0) + abs(means[1] - 1.0) + abs(means[2] - 1.0) + 0.30,
            abs(means[3] - 1.0) + 0.12,
            0.0,
            fallback=(0.72, 0.28, 0.0),
        )[:2]
        emissions_mix = _normalized_triplet(
            abs(means[0] - 1.0) + abs(means[1] - 1.0) + abs(means[2] - 1.0) + 0.38,
            abs(means[4] - 1.0) + 0.08,
            0.0,
            fallback=(0.84, 0.16, 0.0),
        )[:2]
        spread_floor = _clamp(_percentile(spreads, 0.20), 0.02, 0.18)
        spread_cap = _clamp(_percentile(spreads, 0.90) * 2.15, max(0.18, spread_floor + 0.05), 0.65)
        factor_low = _clamp(_percentile(duration_samples, 0.05), 0.45, 1.0)
        factor_high = _clamp(_percentile(duration_samples, 0.95), 1.0, 2.60)
        regimes[regime_id] = {
            "sigma_scale": max(0.1, sigma_scale),
            "traffic_scale": max(0.1, means[0]),
            "incident_scale": max(0.1, means[1]),
            "weather_scale": max(0.1, means[2]),
            "price_scale": max(0.1, means[3]),
            "eco_scale": max(0.1, means[4]),
            "spread_floor": spread_floor,
            "spread_cap": spread_cap,
            "factor_low": factor_low,
            "factor_high": factor_high,
            "duration_mix": duration_mix,
            "monetary_mix": monetary_mix,
            "emissions_mix": emissions_mix,
            "corr": corr,
        }
    return regimes


def _build_residual_priors(residuals_csv: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    with residuals_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            day_kind = str(row.get("day_kind", "weekday")).strip().lower()
            if day_kind not in {"weekday", "weekend", "holiday"}:
                day_kind = "weekday"
            corridor_bucket = str(row.get("corridor_bucket", "uk_default")).strip().lower() or "uk_default"
            local_time_slot = str(row.get("local_time_slot", "h12")).strip().lower() or "h12"
            road_bucket = str(row.get("road_bucket", "mixed")).strip().lower() or "mixed"
            weather_profile = str(row.get("weather_profile", "clear")).strip().lower() or "clear"
            vehicle_type = str(row.get("vehicle_type", "default")).strip().lower() or "default"
            sigma_raw = row.get("sigma")
            if sigma_raw not in (None, ""):
                try:
                    sigma = abs(float(sigma_raw))
                except ValueError:
                    continue
            else:
                try:
                    traffic = abs(math.log(max(0.05, float(row.get("traffic", 1.0)))))
                    incident = abs(math.log(max(0.05, float(row.get("incident", 1.0)))))
                    weather = abs(math.log(max(0.05, float(row.get("weather", 1.0)))))
                    sigma = (traffic + incident + weather) / 3.0
                except ValueError:
                    sigma = 0.06
            grouped[(corridor_bucket, day_kind, local_time_slot, road_bucket, weather_profile, vehicle_type)].append(sigma)

    priors: list[dict[str, Any]] = []
    for (corridor_bucket, day_kind, local_time_slot, road_bucket, weather_profile, vehicle_type), values in sorted(grouped.items()):
        sigma_floor = _clamp(_percentile(values, 0.30), 0.02, 0.28)
        prior_id = f"{corridor_bucket}:{day_kind}:{local_time_slot}:{road_bucket}:{weather_profile}:{vehicle_type}"
        priors.append(
            {
                "corridor_bucket": corridor_bucket,
                "day_kind": day_kind,
                "local_time_slot": local_time_slot,
                "road_bucket": road_bucket,
                "weather_profile": weather_profile,
                "vehicle_type": vehicle_type,
                "sigma_floor": sigma_floor,
                "sample_count": len(values),
                "prior_id": prior_id,
            }
        )
    return priors


def _ci_strict_mode() -> bool:
    return str(os.environ.get("CI", "")).strip().lower() in {"1", "true", "yes"}


def _test_only_synthetic_allowed() -> bool:
    return str(os.environ.get("TEST_ONLY_SYNTHETIC", "")).strip().lower() in {"1", "true", "yes"}


def build(
    *,
    output_json: Path,
    output_priors_json: Path | None = None,
    residuals_csv: Path | None = None,
    allow_synthetic: bool = False,
) -> None:
    if allow_synthetic and not _test_only_synthetic_allowed():
        raise RuntimeError(
            "Synthetic stochastic calibration generation is disabled in strict runtime. "
            "Set TEST_ONLY_SYNTHETIC=1 for explicit test-only generation."
        )
    if allow_synthetic and _ci_strict_mode():
        raise RuntimeError("Synthetic stochastic calibration generation is disabled in CI strict mode.")
    regimes: dict[str, dict[str, Any]] = {}
    priors: list[dict[str, Any]] = []
    calibration_version = "v3-uk-empirical-2026.02"
    calibration_basis = "synthetic"
    if residuals_csv is not None and residuals_csv.exists():
        regimes = _build_from_residuals(residuals_csv=residuals_csv)
        priors = _build_residual_priors(residuals_csv=residuals_csv)
        calibration_version = "v4-uk-residual-fit"
        calibration_basis = "empirical"
    else:
        if not allow_synthetic:
            raise FileNotFoundError(
                "Residual calibration CSV is required. Pass --allow-synthetic for heuristic fallback."
            )
        for base_name, base_vals in BASE_REGIMES.items():
            for road_name, road_mod in ROAD_MOD.items():
                for weather_name, weather_mod in WEATHER_MOD.items():
                    key = f"{base_name}_{road_name}_{weather_name}"
                    regimes[key] = _merge(base_vals, road_mod, weather_mod)
            # Keep coarse keys too.
            regimes[base_name] = dict(base_vals, corr=CORR)
        calibration_version = "v3-uk-synthetic"
        priors = [
            {
                "day_kind": "weekday",
                "road_bucket": "mixed",
                "weather_profile": "clear",
                "vehicle_type": "default",
                "sigma_floor": 0.08,
                "sample_count": 24,
                "prior_id": "weekday:mixed:clear:default",
            }
        ]

    output_priors = output_priors_json or output_json.with_name("stochastic_residual_priors_uk.json")
    payload = {
        "copula_id": "gaussian_5x5_uk_v3_calibrated",
        "calibration_version": calibration_version,
        "calibration_basis": calibration_basis,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "as_of_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "regimes": regimes,
    }
    priors_payload = {
        "source": str(residuals_csv) if residuals_csv is not None else "synthetic_default",
        "calibration_version": calibration_version,
        "generated_at_utc": payload["generated_at_utc"],
        "as_of_utc": payload["as_of_utc"],
        "priors": priors,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_priors.write_text(json.dumps(priors_payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calibrated UK stochastic regime table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/out/model_assets/stochastic_regimes_uk.json"),
    )
    parser.add_argument(
        "--residuals-csv",
        type=Path,
        default=None,
        help="Optional residual-fit calibration CSV.",
    )
    parser.add_argument(
        "--output-priors",
        type=Path,
        default=None,
        help="Optional output path for residual prior table.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic regime synthesis when residual CSV is unavailable.",
    )
    args = parser.parse_args()
    build(
        output_json=args.output,
        output_priors_json=args.output_priors,
        residuals_csv=args.residuals_csv,
        allow_synthetic=bool(args.allow_synthetic),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
