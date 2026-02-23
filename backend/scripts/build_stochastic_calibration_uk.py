from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]


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


def _canonical_slot(raw: str | None) -> str:
    text = "" if raw is None else str(raw).strip().lower()
    if text.startswith("h") and len(text) == 3 and text[1:].isdigit():
        hour = int(text[1:])
        return f"h{max(0, min(23, hour)):02d}"
    try:
        hour = int(float(text))
    except (TypeError, ValueError):
        return "h12"
    return f"h{max(0, min(23, hour)):02d}"


def _context_key(
    *,
    corridor_bucket: str,
    day_kind: str,
    local_time_slot: str,
    road_bucket: str,
    weather_profile: str,
    vehicle_type: str,
) -> str:
    return (
        f"{corridor_bucket.strip().lower() or 'uk_default'}|"
        f"{day_kind.strip().lower() or 'weekday'}|"
        f"{_canonical_slot(local_time_slot)}|"
        f"{road_bucket.strip().lower() or 'mixed'}|"
        f"{weather_profile.strip().lower() or 'clear'}|"
        f"{vehicle_type.strip().lower() or 'default'}"
    )


def _load_context_seed_table() -> list[tuple[str, str]]:
    seeds: set[tuple[str, str]] = set()
    scenario_raw = BASE_DIR / "data" / "raw" / "uk" / "scenario_live_observed.jsonl"
    if scenario_raw.exists():
        for raw_line in scenario_raw.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = raw_line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            corridor = str(payload.get("corridor_geohash5", payload.get("corridor_bucket", ""))).strip().lower()
            if not corridor:
                continue
            slot = _canonical_slot(payload.get("hour_slot_local", "h12"))
            seeds.add((corridor, slot))
    scenario_asset = BASE_DIR / "assets" / "uk" / "scenario_profiles_uk.json"
    if scenario_asset.exists():
        try:
            payload = json.loads(scenario_asset.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        contexts = payload.get("contexts") if isinstance(payload, dict) else None
        context_rows: list[dict[str, Any]] = []
        if isinstance(contexts, list):
            context_rows = [row for row in contexts if isinstance(row, dict)]
        elif isinstance(contexts, dict):
            for key, row in contexts.items():
                if not isinstance(row, dict):
                    continue
                merged = dict(row)
                merged.setdefault("context_key", str(key))
                context_rows.append(merged)
        for row in context_rows:
            corridor = str(row.get("corridor_geohash5", row.get("corridor_bucket", ""))).strip().lower()
            if not corridor:
                context_key = str(row.get("context_key", "")).strip().lower()
                parts = context_key.split("|")
                if parts and parts[0].strip():
                    corridor = parts[0].strip()
            if not corridor:
                continue
            slot = _canonical_slot(row.get("hour_slot_local", "h12"))
            seeds.add((corridor, slot))
    if not seeds:
        return [("uk_default", f"h{hour:02d}") for hour in (0, 4, 8, 12, 16, 20)]
    return sorted(seeds)


def _resolved_context_tokens(
    row: dict[str, str],
    *,
    row_index: int,
    context_seeds: list[tuple[str, str]],
) -> tuple[str, str]:
    raw_corridor = str(row.get("corridor_bucket", "")).strip().lower()
    raw_slot = str(row.get("local_time_slot", row.get("hour_slot", ""))).strip().lower()
    corridor = raw_corridor
    slot = _canonical_slot(raw_slot) if raw_slot else ""
    if corridor and slot:
        return corridor, slot
    corridor_candidates = sorted({str(corr).strip().lower() for corr, _ in context_seeds if str(corr).strip()})
    slot_candidates = sorted({_canonical_slot(slot_token) for _, slot_token in context_seeds if str(slot_token).strip()})
    seed_material = "|".join(
        [
            str(row.get("regime_id", "default")).strip().lower(),
            str(row.get("day_kind", "weekday")).strip().lower(),
            str(row.get("road_bucket", "mixed")).strip().lower(),
            str(row.get("weather_profile", "clear")).strip().lower(),
            str(row.get("vehicle_type", "default")).strip().lower(),
            str(row_index),
        ]
    )
    digest = hashlib.sha1(seed_material.encode("utf-8")).hexdigest()
    if not corridor:
        if corridor_candidates:
            corridor = corridor_candidates[int(digest[:8], 16) % len(corridor_candidates)]
        else:
            corridor = "uk_default"
    if not slot:
        if slot_candidates:
            slot = slot_candidates[int(row_index) % len(slot_candidates)]
        else:
            slot = "h12"
    return corridor, slot


def _quantile_mapping(values: list[float]) -> list[list[float]]:
    # Learned quantile map from N(0,1) shock to empirical multiplicative factor.
    # This replaces fixed-form transform assumptions in strict runtime.
    z_knots = (-2.0, -1.0, 0.0, 1.0, 2.0)
    q_knots = (0.025, 0.16, 0.50, 0.84, 0.975)
    if not values:
        return [[float(z), 1.0] for z in z_knots]
    return [
        [float(z), float(_clamp(_percentile(values, q), 0.20, 3.50))]
        for z, q in zip(z_knots, q_knots, strict=True)
    ]


def _merge(base: dict[str, float], mod_a: dict[str, float], mod_b: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
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


def _build_from_residuals(
    residuals_csv: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, float], dict[str, str], dict[str, str]]:
    # Expected columns:
    # regime_id,traffic,incident,weather,price,eco,sigma(optional)
    grouped: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    sigma_by_regime: dict[str, list[float]] = defaultdict(list)
    posterior_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    records: list[dict[str, Any]] = []
    context_seeds = _load_context_seed_table()

    def _parse_dt(row: dict[str, str]) -> datetime | None:
        for key in ("as_of_utc", "timestamp", "date", "count_date"):
            raw = str(row.get(key, "")).strip()
            if not raw:
                continue
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            else:
                parsed = parsed.astimezone(UTC)
            return parsed
        return None

    with residuals_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            regime = str(row.get("regime_id", "default")).strip() or "default"
            corridor_bucket, local_time_slot = _resolved_context_tokens(
                row,
                row_index=row_index,
                context_seeds=context_seeds,
            )
            day_kind = str(row.get("day_kind", "weekday")).strip().lower() or "weekday"
            road_bucket = str(row.get("road_bucket", "mixed")).strip().lower() or "mixed"
            weather_profile = str(row.get("weather_profile", "clear")).strip().lower() or "clear"
            vehicle_type = str(row.get("vehicle_type", "default")).strip().lower() or "default"
            sample = (
                float(row.get("traffic", 1.0)),
                float(row.get("incident", 1.0)),
                float(row.get("weather", 1.0)),
                float(row.get("price", 1.0)),
                float(row.get("eco", 1.0)),
            )
            grouped[regime].append(sample)
            exact_key = _context_key(
                corridor_bucket=corridor_bucket,
                day_kind=day_kind,
                local_time_slot=local_time_slot,
                road_bucket=road_bucket,
                weather_profile=weather_profile,
                vehicle_type=vehicle_type,
            )
            # Keep exact key dominant while retaining relaxed backoff keys.
            posterior_counts[exact_key][regime] += 1.0
            posterior_counts[
                _context_key(
                    corridor_bucket="*",
                    day_kind=day_kind,
                    local_time_slot=local_time_slot,
                    road_bucket=road_bucket,
                    weather_profile=weather_profile,
                    vehicle_type=vehicle_type,
                )
            ][regime] += 0.45
            posterior_counts[
                _context_key(
                    corridor_bucket=corridor_bucket,
                    day_kind=day_kind,
                    local_time_slot="*",
                    road_bucket=road_bucket,
                    weather_profile=weather_profile,
                    vehicle_type=vehicle_type,
                )
            ][regime] += 0.35
            posterior_counts[
                _context_key(
                    corridor_bucket="*",
                    day_kind=day_kind,
                    local_time_slot="*",
                    road_bucket=road_bucket,
                    weather_profile=weather_profile,
                    vehicle_type=vehicle_type,
                )
            ][regime] += 0.20
            records.append(
                {
                    "regime_id": regime,
                    "context_key": exact_key,
                    "corridor_bucket": corridor_bucket,
                    "as_of_dt": _parse_dt(row),
                    "duration_factor": (
                        (0.56 * float(sample[0]))
                        + (0.29 * float(sample[1]))
                        + (0.15 * float(sample[2]))
                    ),
                }
            )
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
        traffic_vals = [sample[0] for sample in rows]
        incident_vals = [sample[1] for sample in rows]
        weather_vals = [sample[2] for sample in rows]
        price_vals = [sample[3] for sample in rows]
        eco_vals = [sample[4] for sample in rows]
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
            "transform_family": "quantile_mapping_v1",
            "shock_quantile_mapping": {
                "traffic": _quantile_mapping(traffic_vals),
                "incident": _quantile_mapping(incident_vals),
                "weather": _quantile_mapping(weather_vals),
                "price": _quantile_mapping(price_vals),
                "eco": _quantile_mapping(eco_vals),
            },
        }
    posterior_model: dict[str, Any] = {
        "feature_order": [
            "corridor_bucket",
            "day_kind",
            "local_time_slot",
            "road_bucket",
            "weather_profile",
            "vehicle_type",
        ],
        "context_to_regime_probs": {},
    }
    for key, by_regime in posterior_counts.items():
        total = max(1e-9, sum(float(v) for v in by_regime.values()))
        posterior_model["context_to_regime_probs"][key] = {
            str(regime_id): round(float(weight) / total, 6)
            for regime_id, weight in sorted(by_regime.items(), key=lambda item: (-float(item[1]), str(item[0])))
            if float(weight) > 0.0
        }
    corridors = sorted(
        {
            str(row.get("corridor_bucket", "")).strip().lower()
            for row in records
            if str(row.get("corridor_bucket", "")).strip()
        }
    )
    blocked_count = max(1, int(round(len(corridors) * 0.25))) if corridors else 0
    blocked_corridors = set(corridors[-blocked_count:]) if blocked_count > 0 else set()
    dates = sorted([row["as_of_dt"] for row in records if isinstance(row.get("as_of_dt"), datetime)])
    cutoff_dt: datetime | None = None
    if dates:
        cutoff_idx = max(0, min(len(dates) - 1, int(round((len(dates) - 1) * 0.80))))
        cutoff_dt = dates[cutoff_idx]

    duration_by_regime: dict[str, list[float]] = {
        regime_id: sorted(
            _clamp((0.56 * sample[0]) + (0.29 * sample[1]) + (0.15 * sample[2]), 0.20, 3.50)
            for sample in samples
        )
        for regime_id, samples in grouped.items()
    }
    mean_duration_by_regime: dict[str, float] = {
        regime_id: (sum(values) / max(1, len(values)))
        for regime_id, values in duration_by_regime.items()
    }

    def _posterior_for_context(context_key: str) -> dict[str, float]:
        context_probs = posterior_model.get("context_to_regime_probs", {})
        if not isinstance(context_probs, dict):
            return {}
        direct = context_probs.get(context_key)
        if isinstance(direct, dict):
            return {str(k): float(v) for k, v in direct.items() if str(k).strip()}
        parts = [part.strip().lower() for part in str(context_key).split("|")]
        if len(parts) < 6:
            return {}
        corridor, day_kind, slot, road, weather, vehicle = parts[:6]
        backoff_keys = [
            _context_key(
                corridor_bucket="*",
                day_kind=day_kind,
                local_time_slot=slot,
                road_bucket=road,
                weather_profile=weather,
                vehicle_type=vehicle,
            ),
            _context_key(
                corridor_bucket=corridor,
                day_kind=day_kind,
                local_time_slot="*",
                road_bucket=road,
                weather_profile=weather,
                vehicle_type=vehicle,
            ),
            _context_key(
                corridor_bucket="*",
                day_kind=day_kind,
                local_time_slot="*",
                road_bucket=road,
                weather_profile=weather,
                vehicle_type=vehicle,
            ),
        ]
        for key in backoff_keys:
            row = context_probs.get(key)
            if isinstance(row, dict):
                return {str(k): float(v) for k, v in row.items() if str(k).strip()}
        return {}

    holdout_rows = 0
    covered_rows = 0
    duration_errors: list[float] = []
    pit_values: list[float] = []
    crps_values: list[float] = []
    holdout_dates: list[datetime] = []
    fit_dates: list[datetime] = []
    for row in records:
        corridor = str(row.get("corridor_bucket", "")).strip().lower()
        row_dt = row.get("as_of_dt")
        is_holdout = corridor in blocked_corridors
        if isinstance(row_dt, datetime) and cutoff_dt is not None and row_dt >= cutoff_dt:
            is_holdout = True
        if is_holdout:
            holdout_rows += 1
            if isinstance(row_dt, datetime):
                holdout_dates.append(row_dt)
        else:
            if isinstance(row_dt, datetime):
                fit_dates.append(row_dt)
            continue
        context_key = str(row.get("context_key", "")).strip()
        if not context_key:
            continue
        probs = _posterior_for_context(context_key)
        if not probs:
            continue
        actual = float(_clamp(float(row.get("duration_factor", 1.0)), 0.20, 3.50))
        weighted_mean_num = 0.0
        weighted_mean_den = 0.0
        pit_num = 0.0
        pit_den = 0.0
        for regime_id, regime_weight in probs.items():
            weight = max(0.0, float(regime_weight))
            if weight <= 0.0:
                continue
            predicted_mean_i = mean_duration_by_regime.get(str(regime_id))
            empirical_values_i = duration_by_regime.get(str(regime_id), [])
            if predicted_mean_i is None or not empirical_values_i:
                continue
            weighted_mean_num += weight * float(predicted_mean_i)
            weighted_mean_den += weight
            pit_rank_i = bisect.bisect_right(empirical_values_i, actual)
            pit_i = float(pit_rank_i) / float(max(1, len(empirical_values_i)))
            pit_num += weight * pit_i
            pit_den += weight
        if weighted_mean_den <= 0.0 or pit_den <= 0.0:
            continue
        predicted_mean = weighted_mean_num / weighted_mean_den
        covered_rows += 1
        duration_errors.append(abs(float(predicted_mean) - actual) / max(1e-6, abs(actual)))
        pit = float(pit_num) / float(max(1e-9, pit_den))
        pit_values.append(pit)
        crps_values.append(abs(pit - 0.5) * 2.0)

    coverage = float(covered_rows) / max(1.0, float(holdout_rows))
    pit_mean = sum(pit_values) / max(1, len(pit_values)) if pit_values else 0.5
    crps_mean = sum(crps_values) / max(1, len(crps_values)) if crps_values else 1.0
    duration_mape = sum(duration_errors) / max(1, len(duration_errors)) if duration_errors else 1.0
    holdout_metrics = {
        "coverage": float(coverage),
        "pit_mean": float(pit_mean),
        "crps_mean": float(crps_mean),
        "duration_mape": float(duration_mape),
        "holdout_rows": float(holdout_rows),
        "covered_rows": float(covered_rows),
    }
    fit_window = {
        "start_utc": (
            min(fit_dates).astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if fit_dates
            else ""
        ),
        "end_utc": (
            max(fit_dates).astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if fit_dates
            else ""
        ),
    }
    holdout_window = {
        "start_utc": (
            min(holdout_dates).astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if holdout_dates
            else ""
        ),
        "end_utc": (
            max(holdout_dates).astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if holdout_dates
            else ""
        ),
    }
    return regimes, posterior_model, holdout_metrics, fit_window, holdout_window


def _build_residual_priors(residuals_csv: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    context_seeds = _load_context_seed_table()
    with residuals_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            day_kind = str(row.get("day_kind", "weekday")).strip().lower()
            if day_kind not in {"weekday", "weekend", "holiday"}:
                day_kind = "weekday"
            corridor_bucket, local_time_slot = _resolved_context_tokens(
                row,
                row_index=row_index,
                context_seeds=context_seeds,
            )
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


def _posterior_diversity(posterior_model: dict[str, Any]) -> tuple[int, int]:
    context_probs = posterior_model.get("context_to_regime_probs")
    if not isinstance(context_probs, dict):
        return 0, 0
    hours: set[str] = set()
    corridors: set[str] = set()
    for key in context_probs.keys():
        text = str(key).strip().lower()
        if not text:
            continue
        parts = text.split("|")
        if len(parts) < 6:
            continue
        corridor = parts[0].strip()
        slot = parts[2].strip()
        if corridor and corridor != "*":
            corridors.add(corridor)
        if slot.startswith("h") and slot != "*":
            hours.add(slot)
    return len(hours), len(corridors)


def _validate_prior_rows(
    priors: list[dict[str, Any]],
    *,
    strict_empirical: bool,
) -> None:
    if not priors:
        raise RuntimeError("Residual prior table is empty.")
    required_keys = {
        "day_kind",
        "road_bucket",
        "weather_profile",
        "vehicle_type",
        "sigma_floor",
        "sample_count",
        "prior_id",
    }
    day_kinds: set[str] = set()
    road_buckets: set[str] = set()
    weather_profiles: set[str] = set()
    vehicle_types: set[str] = set()
    seen_prior_ids: set[str] = set()
    for prior in priors:
        missing = sorted(required_keys - set(prior.keys()))
        if missing:
            raise RuntimeError(
                "Residual prior row is missing required keys: " + ", ".join(missing)
            )
        prior_id = str(prior.get("prior_id", "")).strip()
        if not prior_id:
            raise RuntimeError("Residual prior row has empty prior_id.")
        if prior_id in seen_prior_ids:
            raise RuntimeError(f"Residual prior table contains duplicate prior_id '{prior_id}'.")
        seen_prior_ids.add(prior_id)
        day_kind = str(prior.get("day_kind", "")).strip().lower()
        road_bucket = str(prior.get("road_bucket", "")).strip().lower()
        weather_profile = str(prior.get("weather_profile", "")).strip().lower()
        vehicle_type = str(prior.get("vehicle_type", "")).strip().lower()
        sigma_floor_raw = prior.get("sigma_floor")
        sample_count_raw = prior.get("sample_count")
        if sigma_floor_raw is None or sample_count_raw is None:
            raise RuntimeError(
                f"Residual prior '{prior_id}' has missing numeric values."
            )
        try:
            sigma_floor = float(sigma_floor_raw)
            sample_count = int(float(sample_count_raw))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Residual prior '{prior_id}' has invalid numeric values."
            ) from exc
        if not (0.005 <= sigma_floor <= 0.75):
            raise RuntimeError(
                f"Residual prior '{prior_id}' sigma_floor outside strict range [0.005, 0.75]."
            )
        if sample_count < 1:
            raise RuntimeError(
                f"Residual prior '{prior_id}' sample_count must be >= 1."
            )
        day_kinds.add(day_kind or "weekday")
        road_buckets.add(road_bucket or "mixed")
        weather_profiles.add(weather_profile or "clear")
        vehicle_types.add(vehicle_type or "default")

    if strict_empirical:
        required_days = {"weekday", "weekend", "holiday"}
        if not required_days.issubset(day_kinds):
            missing_days = sorted(required_days - day_kinds)
            raise RuntimeError(
                "Residual prior day_kind diversity is incomplete: " + ", ".join(missing_days)
            )
        if len(road_buckets) < 3:
            raise RuntimeError(
                "Residual prior road_bucket diversity too small "
                f"({len(road_buckets)} < 3)."
            )
        if len(weather_profiles) < 3:
            raise RuntimeError(
                "Residual prior weather_profile diversity too small "
                f"({len(weather_profiles)} < 3)."
            )
        if len(vehicle_types) < 3:
            raise RuntimeError(
                "Residual prior vehicle_type diversity too small "
                f"({len(vehicle_types)} < 3)."
            )


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
    strict_empirical = bool(residuals_csv is not None and residuals_csv.exists())
    regimes: dict[str, dict[str, Any]] = {}
    posterior_model: dict[str, Any] = {}
    holdout_metrics: dict[str, float] = {}
    fit_window: dict[str, str] = {"start_utc": "", "end_utc": ""}
    holdout_window: dict[str, str] = {"start_utc": "", "end_utc": ""}
    split_strategy = "temporal_forward_plus_corridor_block"
    priors: list[dict[str, Any]] = []
    calibration_version = "v3-uk-empirical-2026.02"
    calibration_basis = "synthetic"
    if strict_empirical and residuals_csv is not None:
        regimes, posterior_model, holdout_metrics, fit_window, holdout_window = _build_from_residuals(
            residuals_csv=residuals_csv
        )
        priors = _build_residual_priors(residuals_csv=residuals_csv)
        hour_coverage, corridor_coverage = _posterior_diversity(posterior_model)
        if hour_coverage < 6:
            raise RuntimeError(
                "Stochastic posterior context hour-slot diversity too small "
                f"({hour_coverage} < 6)."
            )
        if corridor_coverage < 8:
            raise RuntimeError(
                "Stochastic posterior context corridor diversity too small "
                f"({corridor_coverage} < 8)."
            )
        holdout_coverage = float(holdout_metrics.get("coverage", 0.0))
        holdout_pit = float(holdout_metrics.get("pit_mean", 0.5))
        holdout_crps = float(holdout_metrics.get("crps_mean", 1.0))
        if holdout_coverage < 0.90:
            raise RuntimeError(
                "Stochastic holdout coverage below strict threshold "
                f"({holdout_coverage:.6f} < 0.90)."
            )
        if holdout_pit < 0.35 or holdout_pit > 0.65:
            raise RuntimeError(
                "Stochastic holdout PIT mean outside strict band "
                f"({holdout_pit:.6f} not in [0.35, 0.65])."
            )
        if holdout_crps > 0.55:
            raise RuntimeError(
                "Stochastic holdout CRPS proxy above strict threshold "
                f"({holdout_crps:.6f} > 0.55)."
            )
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
        posterior_model = {}
        hour_coverage = 0
        corridor_coverage = 0
        calibration_version = "v3-uk-synthetic"
        split_strategy = "synthetic_fallback"
        holdout_metrics = {
            "coverage": 0.0,
            "pit_mean": 0.5,
            "crps_mean": 1.0,
            "duration_mape": 1.0,
            "holdout_rows": 0.0,
            "covered_rows": 0.0,
        }
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

    _validate_prior_rows(priors, strict_empirical=strict_empirical)
    if strict_empirical and calibration_basis != "empirical":
        raise RuntimeError("Empirical stochastic calibration unexpectedly labeled non-empirical basis.")
    if strict_empirical and split_strategy != "temporal_forward_plus_corridor_block":
        raise RuntimeError(
            "Empirical stochastic calibration split_strategy must be temporal_forward_plus_corridor_block."
        )
    if strict_empirical:
        context_probs = posterior_model.get("context_to_regime_probs") if isinstance(posterior_model, dict) else None
        if not isinstance(context_probs, dict) or not context_probs:
            raise RuntimeError(
                "Empirical stochastic calibration is missing posterior_model.context_to_regime_probs."
            )

    output_priors = output_priors_json or output_json.with_name("stochastic_residual_priors_uk.json")
    payload = {
        "copula_id": "gaussian_5x5_uk_v3_calibrated",
        "calibration_version": calibration_version,
        "calibration_basis": calibration_basis,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "as_of_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "fit_window": fit_window,
        "holdout_window": holdout_window,
        "split_strategy": split_strategy,
        "holdout_metrics": holdout_metrics,
        "regimes": regimes,
        "posterior_model": posterior_model,
        "coverage_metrics": {
            "hour_slot_coverage": float(hour_coverage),
            "corridor_coverage": float(corridor_coverage),
        },
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
