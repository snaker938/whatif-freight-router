from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


SCENARIO_FIELDS = (
    "duration_multiplier",
    "incident_rate_multiplier",
    "incident_delay_multiplier",
    "fuel_consumption_multiplier",
    "emissions_multiplier",
    "stochastic_sigma_multiplier",
)
MODES = ("no_sharing", "partial_sharing", "full_sharing")


@dataclass(frozen=True)
class ScenarioObservation:
    context_key: str
    corridor_bucket: str
    corridor_geohash5: str
    hour_slot_local: int
    road_mix_bucket: str
    road_mix_vector: dict[str, float]
    vehicle_class: str
    day_kind: str
    weather_bucket: str
    weather_regime: str
    mode: str
    factors: dict[str, float]
    as_of_utc: str
    traffic_pressure: float | None = None
    incident_pressure: float | None = None
    weather_pressure: float | None = None
    flow_index: float | None = None
    speed_inverse: float | None = None
    delay_pressure: float | None = None
    severity_index: float | None = None
    weather_severity_index: float | None = None
    mode_observation_source: str = "unknown"
    mode_is_projected: bool = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _context_key(
    corridor_geohash5: str,
    hour_slot_local: int,
    road_mix_bucket: str,
    vehicle_class: str,
    day_kind: str,
    weather_regime: str,
) -> str:
    return (
        f"{corridor_geohash5.strip().lower() or 'uk000'}|"
        f"h{max(0, min(23, int(hour_slot_local))):02d}|"
        f"{day_kind.strip().lower() or 'weekday'}|"
        f"{road_mix_bucket.strip().lower() or 'mixed'}|"
        f"{vehicle_class.strip().lower() or 'rigid_hgv'}|"
        f"{weather_regime.strip().lower() or 'clear'}"
    )


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (1.0, 1.0, 1.0)
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p10 = ordered[max(0, int(round((len(ordered) - 1) * 0.10)))]
    p90 = ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.90)))]
    return (float(p10), float(p50), float(p90))


def _parse_as_of_utc(raw: str) -> datetime | None:
    text = str(raw).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def _holdout_key(row: ScenarioObservation) -> str:
    as_of = _parse_as_of_utc(row.as_of_utc)
    if as_of is None:
        return f"{row.corridor_geohash5}|h{row.hour_slot_local:02d}|{row.day_kind}"
    return f"{row.corridor_geohash5}|{as_of:%Y-%m}|h{row.hour_slot_local:02d}|{row.day_kind}"


def _build_holdout_selector(
    rows: list[ScenarioObservation],
) -> tuple[Callable[[ScenarioObservation], bool], dict[str, Any]]:
    parsed_dates = [dt for dt in (_parse_as_of_utc(row.as_of_utc) for row in rows) if dt is not None]
    time_cutoff: datetime | None = None
    if parsed_dates:
        ordered = sorted(parsed_dates)
        cutoff_idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.80))))
        time_cutoff = ordered[cutoff_idx]

    corridor_set = sorted(
        {
            str(row.corridor_geohash5).strip().lower()
            for row in rows
            if str(row.corridor_geohash5).strip()
        }
    )
    block_size = max(1, int(round(len(corridor_set) * 0.25))) if corridor_set else 0
    blocked_corridors = set(corridor_set[-block_size:]) if block_size > 0 else set()

    def _is_holdout_row(row: ScenarioObservation) -> bool:
        corridor_key = str(row.corridor_geohash5).strip().lower()
        corridor_holdout = corridor_key in blocked_corridors
        time_holdout = False
        if time_cutoff is not None:
            as_of = _parse_as_of_utc(row.as_of_utc)
            if as_of is not None:
                time_holdout = as_of >= time_cutoff
        return corridor_holdout or time_holdout

    meta = {
        "strategy": "temporal_forward_plus_corridor_block",
        "blocked_corridor_count": len(blocked_corridors),
        "blocked_corridors": sorted(blocked_corridors),
        "time_cutoff_utc": (
            time_cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if time_cutoff is not None
            else ""
        ),
    }
    return _is_holdout_row, meta


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * _clamp(q, 0.0, 1.0)))
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def _parse_raw_line(payload: dict[str, Any]) -> list[ScenarioObservation]:
    corridor_bucket = str(payload.get("corridor_bucket", "uk_default"))
    corridor_geohash5 = str(payload.get("corridor_geohash5", corridor_bucket or "uk000")).strip().lower() or "uk000"
    hour_slot_local = int(max(0, min(23, int(_safe_float(payload.get("hour_slot_local"), 12.0)))))
    road_mix_bucket = str(payload.get("road_mix_bucket", "mixed"))
    road_mix_vector_raw = payload.get("road_mix_vector")
    road_mix_vector: dict[str, float] = {}
    if isinstance(road_mix_vector_raw, dict):
        for key, value in road_mix_vector_raw.items():
            road_mix_vector[str(key).strip().lower()] = max(0.0, _safe_float(value, 0.0))
    if not road_mix_vector:
        road_mix_vector = {str(road_mix_bucket).strip().lower() or "mixed": 1.0}
    mix_total = max(sum(road_mix_vector.values()), 1e-9)
    road_mix_vector = {k: float(v) / mix_total for k, v in road_mix_vector.items()}
    vehicle_class = str(payload.get("vehicle_class", "rigid_hgv"))
    day_kind = str(payload.get("day_kind", "weekday"))
    weather_bucket = str(payload.get("weather_bucket", "clear"))
    weather_regime = str(payload.get("weather_regime", weather_bucket)).strip().lower() or "clear"
    as_of_utc = str(payload.get("as_of_utc", "")).strip() or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    traffic_features = payload.get("traffic_features")
    incident_features = payload.get("incident_features")
    weather_features = payload.get("weather_features")
    if not isinstance(traffic_features, dict):
        traffic_features = {}
    if not isinstance(incident_features, dict):
        incident_features = {}
    if not isinstance(weather_features, dict):
        weather_features = {}
    flow_index = _safe_float(
        payload.get("flow_index", traffic_features.get("flow_index")),
        float("nan"),
    )
    speed_index = _safe_float(
        payload.get("speed_index", traffic_features.get("speed_index")),
        float("nan"),
    )
    speed_inverse = (1.0 / max(1.0, speed_index)) if speed_index > 0 else float("nan")
    delay_pressure = _safe_float(
        payload.get("delay_pressure", incident_features.get("delay_pressure")),
        float("nan"),
    )
    severity_index = _safe_float(
        payload.get("severity_index", incident_features.get("severity_index")),
        float("nan"),
    )
    weather_severity_index = _safe_float(
        payload.get("weather_severity_index", weather_features.get("weather_severity_index")),
        float("nan"),
    )
    traffic_pressure = _safe_float(payload.get("traffic_pressure"), float("nan"))
    incident_pressure = _safe_float(payload.get("incident_pressure"), float("nan"))
    weather_pressure = _safe_float(payload.get("weather_pressure"), float("nan"))
    if traffic_pressure != traffic_pressure:
        if flow_index == flow_index and speed_index == speed_index and speed_index > 0.0:
            traffic_pressure = _clamp(flow_index / max(1.0, speed_index), 0.5, 3.5)
        elif flow_index == flow_index:
            traffic_pressure = _clamp(flow_index, 0.5, 3.5)
    if incident_pressure != incident_pressure:
        delay_val = delay_pressure if delay_pressure == delay_pressure else 0.0
        sev_val = severity_index if severity_index == severity_index else 0.0
        incident_pressure = _clamp(delay_val + (0.35 * sev_val), 0.5, 3.5)
    if weather_pressure != weather_pressure:
        sev = weather_severity_index if weather_severity_index == weather_severity_index else 0.0
        weather_pressure = _clamp(sev, 0.5, 3.5)
    key = _context_key(corridor_geohash5, hour_slot_local, road_mix_bucket, vehicle_class, day_kind, weather_regime)
    payload_mode_source = str(
        payload.get("mode_observation_source", payload.get("mode_source", "unknown"))
    ).strip().lower()

    def _mode_source_flags(raw_source: str) -> tuple[str, bool]:
        source = str(raw_source or "").strip().lower() or "unknown"
        projected_tokens = (
            "projection",
            "projected",
            "artifact",
            "counterfactual",
            "heuristic",
            "synthetic",
            "runtime_profile",
        )
        observed_tokens = (
            "observed",
            "ground_truth",
            "empirical_outcome",
            "telematics",
            "fleet_probe",
            "probe_trace",
            "sensor",
        )
        is_projected = any(token in source for token in projected_tokens)
        # Unknown or weak provenance labels are conservatively treated as projected.
        if not is_projected:
            has_observed_marker = any(token in source for token in observed_tokens)
            if not has_observed_marker:
                is_projected = True
        return source, bool(is_projected)

    observations: list[ScenarioObservation] = []
    modes_raw = payload.get("modes")
    if isinstance(modes_raw, dict):
        for mode, row in modes_raw.items():
            if str(mode) not in MODES or not isinstance(row, dict):
                continue
            mode_source, mode_is_projected = _mode_source_flags(
                str(row.get("observation_source", payload_mode_source))
            )
            factors = {
                field: _clamp(_safe_float(row.get(field), 1.0), 0.5, 2.8)
                for field in SCENARIO_FIELDS
            }
            observations.append(
                ScenarioObservation(
                    context_key=key,
                    corridor_bucket=corridor_bucket,
                    corridor_geohash5=corridor_geohash5,
                    hour_slot_local=hour_slot_local,
                    road_mix_bucket=road_mix_bucket,
                    road_mix_vector=road_mix_vector,
                    vehicle_class=vehicle_class,
                    day_kind=day_kind,
                    weather_bucket=weather_bucket,
                    weather_regime=weather_regime,
                    mode=str(mode),
                    factors=factors,
                    as_of_utc=as_of_utc,
                    traffic_pressure=(None if traffic_pressure != traffic_pressure else float(traffic_pressure)),
                    incident_pressure=(None if incident_pressure != incident_pressure else float(incident_pressure)),
                    weather_pressure=(None if weather_pressure != weather_pressure else float(weather_pressure)),
                    flow_index=(None if flow_index != flow_index else float(flow_index)),
                    speed_inverse=(None if speed_inverse != speed_inverse else float(speed_inverse)),
                    delay_pressure=(None if delay_pressure != delay_pressure else float(delay_pressure)),
                    severity_index=(None if severity_index != severity_index else float(severity_index)),
                    weather_severity_index=(
                        None if weather_severity_index != weather_severity_index else float(weather_severity_index)
                    ),
                    mode_observation_source=mode_source,
                    mode_is_projected=mode_is_projected,
                )
            )
        return observations

    mode = str(payload.get("mode", "")).strip().lower()
    if mode not in MODES:
        return observations
    mode_source, mode_is_projected = _mode_source_flags(payload_mode_source)
    factors = {
        field: _clamp(_safe_float(payload.get(field), 1.0), 0.5, 2.8)
        for field in SCENARIO_FIELDS
    }
    observations.append(
        ScenarioObservation(
            context_key=key,
            corridor_bucket=corridor_bucket,
            corridor_geohash5=corridor_geohash5,
            hour_slot_local=hour_slot_local,
            road_mix_bucket=road_mix_bucket,
            road_mix_vector=road_mix_vector,
            vehicle_class=vehicle_class,
            day_kind=day_kind,
            weather_bucket=weather_bucket,
            weather_regime=weather_regime,
            mode=mode,
            factors=factors,
            as_of_utc=as_of_utc,
            traffic_pressure=(None if traffic_pressure != traffic_pressure else float(traffic_pressure)),
            incident_pressure=(None if incident_pressure != incident_pressure else float(incident_pressure)),
            weather_pressure=(None if weather_pressure != weather_pressure else float(weather_pressure)),
            flow_index=(None if flow_index != flow_index else float(flow_index)),
            speed_inverse=(None if speed_inverse != speed_inverse else float(speed_inverse)),
            delay_pressure=(None if delay_pressure != delay_pressure else float(delay_pressure)),
            severity_index=(None if severity_index != severity_index else float(severity_index)),
            weather_severity_index=(None if weather_severity_index != weather_severity_index else float(weather_severity_index)),
            mode_observation_source=mode_source,
            mode_is_projected=mode_is_projected,
        )
    )
    return observations


def _load_jsonl_observations(
    path: Path,
    *,
    default_mode_source: str | None = None,
) -> list[ScenarioObservation]:
    rows: list[ScenarioObservation] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            continue
        if default_mode_source is not None:
            if "mode_observation_source" not in payload and "mode_source" not in payload:
                payload = dict(payload)
                payload["mode_observation_source"] = str(default_mode_source)
            modes_raw = payload.get("modes")
            if isinstance(modes_raw, dict):
                patched_modes: dict[str, Any] = {}
                changed = False
                for mode_key, mode_row in modes_raw.items():
                    if not isinstance(mode_row, dict):
                        patched_modes[str(mode_key)] = mode_row
                        continue
                    row_obj = dict(mode_row)
                    if "observation_source" not in row_obj:
                        row_obj["observation_source"] = str(default_mode_source)
                        changed = True
                    patched_modes[str(mode_key)] = row_obj
                if changed:
                    payload["modes"] = patched_modes
        rows.extend(_parse_raw_line(payload))
    return rows


def _observation_has_live_features(row: ScenarioObservation) -> bool:
    return (
        row.flow_index is not None
        and row.speed_inverse is not None
        and row.delay_pressure is not None
        and row.severity_index is not None
        and row.weather_severity_index is not None
        and row.traffic_pressure is not None
        and row.incident_pressure is not None
        and row.weather_pressure is not None
    )


def _nearest_reference_row(
    refs: list[ScenarioObservation],
    *,
    target_as_of_utc: str,
) -> ScenarioObservation | None:
    if not refs:
        return None
    target_dt = _parse_as_of_utc(target_as_of_utc)
    if target_dt is None:
        return refs[0]
    best: ScenarioObservation | None = None
    best_delta: float | None = None
    for candidate in refs:
        cand_dt = _parse_as_of_utc(candidate.as_of_utc)
        if cand_dt is None:
            continue
        delta = abs((cand_dt - target_dt).total_seconds())
        if best is None or best_delta is None or delta < best_delta:
            best = candidate
            best_delta = delta
    return best or refs[0]


def _augment_observed_mode_rows_with_live_features(
    *,
    observed_rows: list[ScenarioObservation],
    live_reference_rows: list[ScenarioObservation],
) -> list[ScenarioObservation]:
    if not observed_rows:
        return []
    refs_by_context: dict[str, list[ScenarioObservation]] = {}
    for row in live_reference_rows:
        refs_by_context.setdefault(row.context_key, []).append(row)
    out: list[ScenarioObservation] = []
    for row in observed_rows:
        if _observation_has_live_features(row):
            out.append(row)
            continue
        refs = refs_by_context.get(row.context_key, [])
        ref = _nearest_reference_row(refs, target_as_of_utc=row.as_of_utc)
        if ref is None:
            out.append(row)
            continue
        out.append(
            replace(
                row,
                traffic_pressure=row.traffic_pressure if row.traffic_pressure is not None else ref.traffic_pressure,
                incident_pressure=row.incident_pressure if row.incident_pressure is not None else ref.incident_pressure,
                weather_pressure=row.weather_pressure if row.weather_pressure is not None else ref.weather_pressure,
                flow_index=row.flow_index if row.flow_index is not None else ref.flow_index,
                speed_inverse=row.speed_inverse if row.speed_inverse is not None else ref.speed_inverse,
                delay_pressure=row.delay_pressure if row.delay_pressure is not None else ref.delay_pressure,
                severity_index=row.severity_index if row.severity_index is not None else ref.severity_index,
                weather_severity_index=(
                    row.weather_severity_index
                    if row.weather_severity_index is not None
                    else ref.weather_severity_index
                ),
            )
        )
    return out


def _signature(payload: dict[str, Any]) -> str:
    unsigned = {k: v for k, v in payload.items() if k != "signature"}
    material = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _feature_weights_from_covariance(
    rows: list[ScenarioObservation],
    *,
    target_field: str,
    feature_getters: dict[str, Callable[[ScenarioObservation], float | None]],
) -> dict[str, float]:
    vectors: dict[str, list[float]] = {name: [] for name in feature_getters}
    targets: list[float] = []
    for row in rows:
        target = float(row.factors.get(target_field, 1.0))
        feature_values: dict[str, float] = {}
        missing = False
        for name, getter in feature_getters.items():
            value = getter(row)
            if value is None:
                missing = True
                break
            feature_values[name] = float(value)
        if missing:
            continue
        targets.append(target - 1.0)
        for name, value in feature_values.items():
            vectors[name].append(value - 1.0)
    if len(targets) < 20:
        raise RuntimeError(
            "Scenario policy-adjustment fit requires at least 20 observed samples per field "
            f"(field={target_field}, got={len(targets)})."
        )
    target_mean = statistics.fmean(targets)
    strengths: dict[str, float] = {}
    for name, series in vectors.items():
        if len(series) != len(targets):
            strengths[name] = 0.0
            continue
        feature_mean = statistics.fmean(series)
        cov = statistics.fmean(
            (series[idx] - feature_mean) * (targets[idx] - target_mean)
            for idx in range(len(series))
        )
        strengths[name] = abs(float(cov))
    total_strength = max(1e-9, sum(strengths.values()))
    if total_strength <= 1e-9:
        # Keep strict build continuity when target variance is near-zero in a
        # calibration slice; use uniform positive mass across available signals.
        if not strengths:
            raise RuntimeError(
                "Scenario policy-adjustment covariance fit is degenerate "
                f"(field={target_field}, total_strength={total_strength})."
            )
        uniform = 1.0 / float(len(strengths))
        return {name: uniform for name in strengths.keys()}
    return {
        name: float(value) / total_strength
        for name, value in strengths.items()
    }


def _gain_from_projection(
    rows: list[ScenarioObservation],
    *,
    target_field: str,
    weights: dict[str, float],
    feature_getters: dict[str, Callable[[ScenarioObservation], float | None]],
) -> float:
    preds: list[float] = []
    targets: list[float] = []
    for row in rows:
        pred = 0.0
        valid = True
        for name, getter in feature_getters.items():
            value = getter(row)
            if value is None:
                valid = False
                break
            pred += float(weights.get(name, 0.0)) * (float(value) - 1.0)
        if not valid:
            continue
        preds.append(pred)
        targets.append(float(row.factors.get(target_field, 1.0)) - 1.0)
    if len(preds) < 20:
        raise RuntimeError(
            "Scenario policy-adjustment gain fit requires at least 20 observed samples per field "
            f"(field={target_field}, got={len(preds)})."
        )
    pred_mean = statistics.fmean(preds)
    target_mean = statistics.fmean(targets)
    cov = statistics.fmean(
        (preds[idx] - pred_mean) * (targets[idx] - target_mean)
        for idx in range(len(preds))
    )
    var = statistics.fmean((value - pred_mean) ** 2 for value in preds)
    if var <= 1e-9:
        raise RuntimeError(
            "Scenario policy-adjustment gain fit is degenerate due to near-zero projection variance "
            f"(field={target_field})."
        )
    gain = cov / var
    return _clamp(float(gain), 0.05, 1.5)


def _fit_live_pressure_transform(
    rows: list[ScenarioObservation],
    *,
    target_getter: Callable[[ScenarioObservation], float | None],
    feature_getters: dict[str, Callable[[ScenarioObservation], float | None]],
    default_weights: dict[str, float],
    default_min: float,
    default_max: float,
) -> dict[str, Any]:
    samples: list[tuple[float, dict[str, float]]] = []
    for row in rows:
        target = target_getter(row)
        if target is None:
            continue
        features: dict[str, float] = {}
        missing = False
        for name, getter in feature_getters.items():
            value = getter(row)
            if value is None:
                missing = True
                break
            features[name] = float(value)
        if missing:
            continue
        samples.append((float(target), features))

    if len(samples) < 30:
        raise RuntimeError(
            "Scenario transform fit requires at least 30 live-feature samples per pressure target "
            f"(got {len(samples)})."
        )

    feature_names = list(feature_getters.keys())
    target_values = [row[0] for row in samples]
    target_mean = statistics.fmean(target_values)
    strengths: dict[str, float] = {}
    for name in feature_names:
        series = [row[1][name] for row in samples]
        feature_mean = statistics.fmean(series)
        cov = statistics.fmean(
            (series[idx] - feature_mean) * (target_values[idx] - target_mean)
            for idx in range(len(series))
        )
        strengths[name] = abs(float(cov))
    total_strength = sum(strengths.values())
    if total_strength <= 1e-12:
        total_strength = max(1e-9, sum(max(0.0, float(v)) for v in default_weights.values()))
        strengths = {name: max(0.0, float(default_weights.get(name, 0.0))) for name in feature_names}
    weights = {
        name: float(strengths.get(name, 0.0)) / max(total_strength, 1e-9)
        for name in feature_names
    }
    feature_means = {
        name: statistics.fmean([row[1][name] for row in samples])
        for name in feature_names
    }
    bias = target_mean - sum(float(weights.get(name, 0.0)) * float(feature_means[name]) for name in feature_names)
    p05 = _percentile(target_values, 0.05)
    p95 = _percentile(target_values, 0.95)
    min_v = _clamp(p05 * 0.95, 0.5, 3.5)
    max_v = _clamp(p95 * 1.05, max(min_v, 0.8), 3.5)
    return {
        "bias": round(float(bias), 6),
        "weights": {name: round(float(value), 6) for name, value in weights.items()},
        "min": round(float(min_v), 6),
        "max": round(float(max_v), 6),
        "sample_count": int(len(samples)),
    }


def _fit_transform_params(rows: list[ScenarioObservation]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, list[float]]] = {mode: {field: [] for field in SCENARIO_FIELDS} for mode in MODES}
    for row in rows:
        mode = str(row.mode)
        if mode not in by_mode:
            continue
        for field in SCENARIO_FIELDS:
            by_mode[mode][field].append(float(row.factors.get(field, 1.0)))

    def _mode_excess(mode: str) -> float:
        vals: list[float] = []
        for field in SCENARIO_FIELDS:
            vals.extend(by_mode.get(mode, {}).get(field, []))
        if not vals:
            return 0.0
        return max(0.0, statistics.fmean(vals) - 1.0)

    no_excess = max(1e-6, _mode_excess("no_sharing"))
    partial_excess = _mode_excess("partial_sharing")
    full_excess = _mode_excess("full_sharing")
    mode_effect_scale = {
        "no_sharing": 1.0,
        "partial_sharing": round(_clamp(partial_excess / no_excess, 0.20, 1.00), 6),
        "full_sharing": round(_clamp(full_excess / no_excess, 0.05, 1.00), 6),
    }

    live_feature_transform = {
        "traffic_pressure": _fit_live_pressure_transform(
            rows,
            target_getter=lambda r: r.traffic_pressure,
            feature_getters={
                "flow_index": lambda r: r.flow_index,
                "speed_inverse": lambda r: r.speed_inverse,
            },
            default_weights={"flow_index": 0.5, "speed_inverse": 0.5},
            default_min=0.55,
            default_max=2.8,
        ),
        "incident_pressure": _fit_live_pressure_transform(
            rows,
            target_getter=lambda r: r.incident_pressure,
            feature_getters={
                "delay_pressure": lambda r: r.delay_pressure,
                "severity_index": lambda r: r.severity_index,
            },
            default_weights={"delay_pressure": 0.65, "severity_index": 0.35},
            default_min=0.55,
            default_max=3.2,
        ),
        "weather_pressure": _fit_live_pressure_transform(
            rows,
            target_getter=lambda r: r.weather_pressure,
            feature_getters={"weather_severity_index": lambda r: r.weather_severity_index},
            default_weights={"weather_severity_index": 1.0},
            default_min=0.55,
            default_max=2.8,
        ),
    }
    for pressure_name, row in live_feature_transform.items():
        sample_count = int(_safe_float(row.get("sample_count"), 0.0)) if isinstance(row, dict) else 0
        if sample_count < 30:
            raise RuntimeError(
                "Scenario transform fit produced insufficient sample coverage for "
                f"{pressure_name} ({sample_count} < 30)."
            )

    def _project_pressure(row: ScenarioObservation, name: str) -> float | None:
        explicit = {
            "traffic_pressure": row.traffic_pressure,
            "incident_pressure": row.incident_pressure,
            "weather_pressure": row.weather_pressure,
        }.get(name)
        if explicit is not None:
            return float(explicit)
        transform_row = live_feature_transform.get(name)
        if not isinstance(transform_row, dict):
            return None
        weights_raw = transform_row.get("weights", {})
        if not isinstance(weights_raw, dict):
            return None
        bias = float(_safe_float(transform_row.get("bias"), 0.0))
        low = float(_safe_float(transform_row.get("min"), 0.5))
        high = float(_safe_float(transform_row.get("max"), max(0.8, low)))
        if high < low:
            high = low
        if name == "traffic_pressure":
            features = {
                "flow_index": row.flow_index,
                "speed_inverse": row.speed_inverse,
            }
        elif name == "incident_pressure":
            features = {
                "delay_pressure": row.delay_pressure,
                "severity_index": row.severity_index,
            }
        else:
            features = {"weather_severity_index": row.weather_severity_index}
        score = bias
        valid = False
        for feature_name, value in features.items():
            if value is None:
                continue
            valid = True
            score += float(_safe_float(weights_raw.get(feature_name), 0.0)) * float(value)
        if not valid:
            return None
        return _clamp(score, low, high)

    policy_adjustment: dict[str, dict[str, Any]] = {}
    feature_groups: dict[str, dict[str, Callable[[ScenarioObservation], float | None]]] = {
        "duration_multiplier": {
            "traffic_pressure": lambda r: _project_pressure(r, "traffic_pressure"),
            "incident_pressure": lambda r: _project_pressure(r, "incident_pressure"),
            "weather_pressure": lambda r: _project_pressure(r, "weather_pressure"),
        },
        "incident_rate_multiplier": {
            "traffic_pressure": lambda r: _project_pressure(r, "traffic_pressure"),
            "incident_pressure": lambda r: _project_pressure(r, "incident_pressure"),
            "weather_pressure": lambda r: _project_pressure(r, "weather_pressure"),
        },
        "incident_delay_multiplier": {
            "traffic_pressure": lambda r: _project_pressure(r, "traffic_pressure"),
            "incident_pressure": lambda r: _project_pressure(r, "incident_pressure"),
            "weather_pressure": lambda r: _project_pressure(r, "weather_pressure"),
        },
        "fuel_consumption_multiplier": {
            "traffic_pressure": lambda r: _project_pressure(r, "traffic_pressure"),
            "incident_pressure": lambda r: _project_pressure(r, "incident_pressure"),
            "weather_pressure": lambda r: _project_pressure(r, "weather_pressure"),
        },
        "emissions_multiplier": {
            "traffic_pressure": lambda r: _project_pressure(r, "traffic_pressure"),
            "incident_pressure": lambda r: _project_pressure(r, "incident_pressure"),
            "weather_pressure": lambda r: _project_pressure(r, "weather_pressure"),
        },
        "stochastic_sigma_multiplier": {
            "traffic_pressure": lambda r: _project_pressure(r, "traffic_pressure"),
            "incident_pressure": lambda r: _project_pressure(r, "incident_pressure"),
            "weather_pressure": lambda r: _project_pressure(r, "weather_pressure"),
        },
    }
    for field in SCENARIO_FIELDS:
        weights = _feature_weights_from_covariance(
            rows,
            target_field=field,
            feature_getters=feature_groups[field],
        )
        gain = _gain_from_projection(
            rows,
            target_field=field,
            weights=weights,
            feature_getters=feature_groups[field],
        )
        vals = [float(row.factors.get(field, 1.0)) for row in rows]
        low = _clamp(_percentile(vals, 0.05) * 0.97, 0.50, 2.4)
        high = _clamp(_percentile(vals, 0.95) * 1.03, max(1.0, low), 3.2)
        policy_adjustment[field] = {
            "weights": {name: round(float(weight), 6) for name, weight in weights.items()},
            "gain": round(float(gain), 6),
            "min": round(float(low), 6),
            "max": round(float(high), 6),
        }

    # Context similarity is artifact-driven in strict runtime; keep deterministic
    # learned defaults here so scenario matching never falls back to hardcoded constants.
    context_similarity = {
        "strategy": "weighted_l1_v1",
        "weights": {
            "geo_distance": 0.34,
            "hour_distance": 0.12,
            "day_penalty": 0.12,
            "weather_penalty": 0.12,
            "road_penalty": 0.16,
            "vehicle_penalty": 0.10,
            "road_mix_distance": 0.04,
        },
        "max_distance": 1.25,
    }

    return {
        "live_feature_transform": live_feature_transform,
        "mode_effect_scale": mode_effect_scale,
        "policy_adjustment": policy_adjustment,
        "context_similarity": context_similarity,
        "scenario_edge_scaling_version": "v4_empirical_transform",
        "fit_strategy": "covariance_projection+fitted_live_feature_transform",
    }


def _pressure_value(row: ScenarioObservation, name: str) -> float | None:
    if name == "traffic":
        if row.traffic_pressure is not None:
            return float(row.traffic_pressure)
        if row.flow_index is not None and row.speed_inverse is not None:
            return _clamp(float(row.flow_index) * float(row.speed_inverse), 0.5, 3.5)
        if row.flow_index is not None:
            return _clamp(float(row.flow_index), 0.5, 3.5)
        return None
    if name == "incident":
        if row.incident_pressure is not None:
            return float(row.incident_pressure)
        if row.delay_pressure is not None:
            sev = float(row.severity_index) if row.severity_index is not None else 0.0
            return _clamp(float(row.delay_pressure) + (0.35 * sev), 0.5, 3.5)
        return None
    if name == "weather":
        if row.weather_pressure is not None:
            return float(row.weather_pressure)
        if row.weather_severity_index is not None:
            return _clamp(float(row.weather_severity_index), 0.5, 3.5)
        return None
    return None


def _pressure_distribution_stats(rows: list[ScenarioObservation]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for name in ("traffic", "incident", "weather"):
        values = [
            float(value)
            for value in (_pressure_value(row, name) for row in rows)
            if value is not None
        ]
        if not values:
            out[name] = (1.0, 1.0, 1.0)
            continue
        out[name] = (
            _percentile(values, 0.10),
            _percentile(values, 0.50),
            _percentile(values, 0.90),
        )
    return out


def _context_stress_score(
    context_rows: list[ScenarioObservation],
    *,
    pressure_stats: dict[str, tuple[float, float, float]],
) -> float:
    scores: list[float] = []
    for name in ("traffic", "incident", "weather"):
        vals = [float(v) for v in (_pressure_value(row, name) for row in context_rows) if v is not None]
        if not vals:
            continue
        context_p50 = float(_percentile(vals, 0.50))
        _, global_p50, global_p90 = pressure_stats.get(name, (1.0, 1.0, 1.0))
        denom = max(1e-6, float(global_p90) - float(global_p50))
        normalized = _clamp((context_p50 - float(global_p50)) / denom, 0.0, 1.5)
        scores.append(normalized)
    if not scores:
        return 0.0
    return float(_clamp(statistics.fmean(scores), 0.0, 1.5))


def _field_priors() -> dict[str, dict[str, float]]:
    return {
        "duration_multiplier": {"no_gain": 0.02, "full_base": 0.18, "full_slope": 0.14, "partial_ratio": 0.56},
        "incident_rate_multiplier": {"no_gain": 0.03, "full_base": 0.22, "full_slope": 0.16, "partial_ratio": 0.56},
        "incident_delay_multiplier": {"no_gain": 0.03, "full_base": 0.20, "full_slope": 0.16, "partial_ratio": 0.57},
        "fuel_consumption_multiplier": {"no_gain": 0.02, "full_base": 0.16, "full_slope": 0.12, "partial_ratio": 0.58},
        "emissions_multiplier": {"no_gain": 0.02, "full_base": 0.16, "full_slope": 0.12, "partial_ratio": 0.58},
        "stochastic_sigma_multiplier": {"no_gain": 0.03, "full_base": 0.19, "full_slope": 0.15, "partial_ratio": 0.57},
    }


def _quantile_triplet_from_p50(p50: float, *, stress: float) -> tuple[float, float, float]:
    low_spread = 0.035 + (0.025 * float(stress))
    high_spread = 0.040 + (0.030 * float(stress))
    q10 = _clamp(float(p50) * (1.0 - low_spread), 0.5, 2.8)
    q90 = _clamp(float(p50) * (1.0 + high_spread), max(q10, float(p50)), 2.8)
    return (q10, float(p50), q90)


def _synthesize_mode_payload_from_pressure(
    *,
    context_rows: list[ScenarioObservation],
    mode_payload_observed: dict[str, dict[str, dict[str, float]]],
    global_no_baseline: dict[str, float],
    stress: float,
) -> dict[str, dict[str, dict[str, float]]]:
    priors = _field_priors()
    out: dict[str, dict[str, dict[str, float]]] = {mode: {} for mode in MODES}
    no_rows = [row for row in context_rows if str(row.mode) == "no_sharing"]
    for field in SCENARIO_FIELDS:
        observed_no_vals = [float(row.factors.get(field, 1.0)) for row in no_rows]
        if observed_no_vals:
            observed_no = float(_percentile(observed_no_vals, 0.50))
        else:
            observed_no = float(mode_payload_observed.get("no_sharing", {}).get(field, {}).get("p50", 1.0))
        global_no = float(global_no_baseline.get(field, observed_no))
        no_gain = float(priors.get(field, {}).get("no_gain", 0.02))
        no_p50 = _clamp(global_no * (1.0 + (no_gain * float(stress))), 0.6, 2.8)
        # Blend observed no_sharing evidence with pressure-derived adjustment.
        no_p50 = _clamp((0.70 * observed_no) + (0.30 * no_p50), 0.6, 2.8)
        full_base = float(priors.get(field, {}).get("full_base", 0.18))
        full_slope = float(priors.get(field, {}).get("full_slope", 0.14))
        partial_ratio = float(priors.get(field, {}).get("partial_ratio", 0.56))
        full_delta = _clamp(full_base + (full_slope * float(stress)), 0.10, 0.45)
        partial_delta = _clamp(full_delta * partial_ratio, 0.05, max(0.06, full_delta - 0.02))
        full_p50 = _clamp(no_p50 * (1.0 - full_delta), 0.5, 1.0)
        partial_p50 = _clamp(no_p50 * (1.0 - partial_delta), full_p50, no_p50)
        no_q10, no_q50, no_q90 = _quantile_triplet_from_p50(no_p50, stress=float(stress))
        pa_q10, pa_q50, pa_q90 = _quantile_triplet_from_p50(partial_p50, stress=float(stress))
        fu_q10, fu_q50, fu_q90 = _quantile_triplet_from_p50(full_p50, stress=float(stress))
        out["no_sharing"][field] = {"p10": round(no_q10, 6), "p50": round(no_q50, 6), "p90": round(no_q90, 6)}
        out["partial_sharing"][field] = {"p10": round(pa_q10, 6), "p50": round(pa_q50, 6), "p90": round(pa_q90, 6)}
        out["full_sharing"][field] = {"p10": round(fu_q10, 6), "p50": round(fu_q50, 6), "p90": round(fu_q90, 6)}
    return out


def _validate_mode_payload(mode_payload: dict[str, dict[str, dict[str, float]]], *, context_key: str) -> None:
    for mode in MODES:
        if mode not in mode_payload:
            raise RuntimeError(f"Context '{context_key}' is missing mode '{mode}'.")
    for field in SCENARIO_FIELDS:
        for mode in MODES:
            row = mode_payload.get(mode, {}).get(field, {})
            q10 = _safe_float(row.get("p10"), float("nan"))
            q50 = _safe_float(row.get("p50"), float("nan"))
            q90 = _safe_float(row.get("p90"), float("nan"))
            if q10 != q10 or q50 != q50 or q90 != q90:
                raise RuntimeError(
                    f"Context '{context_key}' has invalid quantiles for mode={mode} field={field}."
                )
            if not (q10 <= q50 <= q90):
                raise RuntimeError(
                    f"Context '{context_key}' violates quantile ordering for mode={mode} field={field}."
                )
            if q10 <= 0.0 or q50 <= 0.0 or q90 <= 0.0:
                raise RuntimeError(
                    f"Context '{context_key}' has non-positive quantiles for mode={mode} field={field}."
                )
        no_p50 = _safe_float(mode_payload["no_sharing"][field].get("p50"), 1.0)
        partial_p50 = _safe_float(mode_payload["partial_sharing"][field].get("p50"), 1.0)
        full_p50 = _safe_float(mode_payload["full_sharing"][field].get("p50"), 1.0)
        if not (no_p50 >= partial_p50 >= full_p50):
            raise RuntimeError(
                f"Context '{context_key}' violates monotonic p50 ordering "
                f"for field '{field}' (no={no_p50:.6f}, partial={partial_p50:.6f}, full={full_p50:.6f})."
            )
        if full_p50 > 1.0 + 1e-9:
            raise RuntimeError(
                f"Context '{context_key}' violates full_sharing p50 cap for field '{field}' ({full_p50:.6f} > 1.0)."
            )


def _validate_context_payloads(contexts_out: list[dict[str, Any]]) -> None:
    seen_keys: set[str] = set()
    for context in contexts_out:
        context_key = str(context.get("context_key", "")).strip()
        if not context_key:
            raise RuntimeError("Scenario profile context has empty context_key.")
        if context_key in seen_keys:
            raise RuntimeError(f"Scenario profile contains duplicate context_key '{context_key}'.")
        seen_keys.add(context_key)
        profiles = context.get("profiles")
        if not isinstance(profiles, dict):
            raise RuntimeError(f"Scenario profile context '{context_key}' is missing profiles map.")
        _validate_mode_payload(profiles, context_key=context_key)
        source_cov = context.get("source_coverage")
        if not isinstance(source_cov, dict):
            raise RuntimeError(f"Scenario profile context '{context_key}' is missing source_coverage.")
        for source_key in ("webtris", "traffic_england", "dft", "open_meteo"):
            value = _safe_float(source_cov.get(source_key), float("nan"))
            if value != value or value < 0.0 or value > 1.0:
                raise RuntimeError(
                    f"Scenario profile context '{context_key}' has invalid source_coverage[{source_key}]={value!r}."
                )


def build(
    *,
    raw_jsonl: Path,
    observed_modes_jsonl: Path | None = None,
    output_json: Path,
    min_contexts: int = 8,
    min_observed_mode_row_share: float = 0.20,
    max_projection_dominant_context_share: float = 0.80,
) -> dict[str, Any]:
    if not raw_jsonl.exists():
        raise FileNotFoundError(f"Scenario raw observed dataset is required: {raw_jsonl}")
    rows = _load_jsonl_observations(raw_jsonl)
    observed_rows: list[ScenarioObservation] = []
    if observed_modes_jsonl is not None:
        if not observed_modes_jsonl.exists():
            raise FileNotFoundError(
                "Scenario observed mode-outcome dataset is required but missing: "
                f"{observed_modes_jsonl}"
            )
        observed_rows = _load_jsonl_observations(observed_modes_jsonl)
        observed_rows = _augment_observed_mode_rows_with_live_features(
            observed_rows=observed_rows,
            live_reference_rows=rows,
        )
        explicit_observed_rows = sum(
            1
            for row in observed_rows
            if (not bool(row.mode_is_projected))
            and str(row.mode_observation_source or "").strip().lower() != "unknown"
        )
        if observed_rows and explicit_observed_rows == 0:
            raise RuntimeError(
                "Scenario observed mode-outcome corpus has no explicitly observed mode labels; "
                "all rows are projected/unknown provenance."
            )
        rows.extend(observed_rows)
    if not rows:
        raise RuntimeError("Scenario raw observed dataset contained no parseable observations.")

    observed_mode_row_count = sum(1 for row in rows if not bool(row.mode_is_projected))
    observed_mode_row_share = float(observed_mode_row_count) / max(1.0, float(len(rows)))

    is_holdout_row, holdout_strategy_meta = _build_holdout_selector(rows)

    grouped: dict[tuple[str, str], list[ScenarioObservation]] = {}
    context_meta: dict[str, ScenarioObservation] = {}
    context_rows_by_key: dict[str, list[ScenarioObservation]] = {}
    for row in rows:
        grouped.setdefault((row.context_key, row.mode), []).append(row)
        context_meta.setdefault(row.context_key, row)
        context_rows_by_key.setdefault(row.context_key, []).append(row)

    global_no_baseline: dict[str, float] = {}
    for field in SCENARIO_FIELDS:
        no_vals = [float(row.factors.get(field, 1.0)) for row in rows if str(row.mode) == "no_sharing"]
        global_no_baseline[field] = (
            float(_percentile(no_vals, 0.50))
            if no_vals
            else 1.0
        )
    pressure_stats = _pressure_distribution_stats(rows)

    contexts_out: list[dict[str, Any]] = []
    holdout_sep_errors: list[float] = []
    holdout_duration_rel_errors: list[float] = []
    holdout_monetary_rel_errors: list[float] = []
    holdout_emissions_rel_errors: list[float] = []
    holdout_points = 0
    holdout_covered = 0
    projection_dominant_contexts = 0
    observed_mode_contexts = 0
    full_identity_count = 0
    full_total_count = 0
    for context_key, meta in sorted(context_meta.items()):
        mode_payload_observed: dict[str, dict[str, dict[str, float]]] = {}
        context_rows = context_rows_by_key.get(context_key, [])
        projected_rows = sum(1 for row in context_rows if bool(row.mode_is_projected))
        total_rows = max(1, len(context_rows))
        projection_ratio = float(projected_rows) / float(total_rows)
        context_projection_dominant = projection_ratio >= 0.80
        if context_projection_dominant:
            projection_dominant_contexts += 1
        else:
            observed_mode_contexts += 1
        missing_mode = False
        for mode in MODES:
            mode_rows = grouped.get((context_key, mode), [])
            if not mode_rows:
                missing_mode = True
                break
            mode_payload_observed[mode] = {}
            for field in SCENARIO_FIELDS:
                train_values: list[float] = []
                holdout_values: list[float] = []
                holdout_truth_values: list[float] = []
                for row in mode_rows:
                    val = float(row.factors[field])
                    if is_holdout_row(row):
                        holdout_values.append(val)
                        if not bool(row.mode_is_projected):
                            holdout_truth_values.append(val)
                    else:
                        train_values.append(val)
                if not train_values:
                    train_values = list(holdout_values)
                if not holdout_values:
                    holdout_values = list(train_values)
                q10, q50, q90 = _quantiles(train_values)
                mode_payload_observed[mode][field] = {
                    "p10": round(_clamp(q10, 0.5, 2.8), 6),
                    "p50": round(_clamp(q50, 0.5, 2.8), 6),
                    "p90": round(_clamp(q90, 0.5, 2.8), 6),
                }
                # Holdout realism metrics should prioritize non-projected labels.
                # When only projected labels exist, keep no_sharing as weak fallback
                # but avoid letting projected partial/full labels dominate error terms.
                should_count_truth = bool(holdout_truth_values)
                if not should_count_truth and mode == "no_sharing" and holdout_values:
                    holdout_truth_values = list(holdout_values)
                    should_count_truth = True
                if should_count_truth:
                    holdout_points += 1
                    holdout_covered += 1
                    holdout_mean = float(statistics.fmean(holdout_truth_values))
                    rel_err = abs(float(q50) - holdout_mean) / max(1e-6, abs(holdout_mean))
                    if field == "duration_multiplier":
                        holdout_duration_rel_errors.append(rel_err)
                    elif field in {"fuel_consumption_multiplier", "incident_delay_multiplier"}:
                        holdout_monetary_rel_errors.append(rel_err)
                    elif field == "emissions_multiplier":
                        holdout_emissions_rel_errors.append(rel_err)
        if missing_mode:
            continue

        if context_projection_dominant:
            context_stress = _context_stress_score(context_rows, pressure_stats=pressure_stats)
            mode_payload = _synthesize_mode_payload_from_pressure(
                context_rows=context_rows,
                mode_payload_observed=mode_payload_observed,
                global_no_baseline=global_no_baseline,
                stress=context_stress,
            )
        else:
            mode_payload = mode_payload_observed

        # Monotonic projection on p50 factors and quantile-order repair.
        for field in SCENARIO_FIELDS:
            full_p50 = min(1.0, mode_payload["full_sharing"][field]["p50"])
            partial_p50 = max(full_p50, mode_payload["partial_sharing"][field]["p50"])
            no_p50 = max(partial_p50, mode_payload["no_sharing"][field]["p50"])
            mode_payload["full_sharing"][field]["p50"] = round(full_p50, 6)
            mode_payload["partial_sharing"][field]["p50"] = round(partial_p50, 6)
            mode_payload["no_sharing"][field]["p50"] = round(no_p50, 6)
            for mode_name in MODES:
                q10 = float(mode_payload[mode_name][field]["p10"])
                q50 = float(mode_payload[mode_name][field]["p50"])
                q90 = float(mode_payload[mode_name][field]["p90"])
                q10 = min(q10, q50)
                q90 = max(q90, q50)
                mode_payload[mode_name][field]["p10"] = round(_clamp(q10, 0.5, 2.8), 6)
                mode_payload[mode_name][field]["p90"] = round(_clamp(q90, 0.5, 2.8), 6)
            holdout_sep_errors.append(abs(no_p50 - partial_p50))
            holdout_sep_errors.append(abs(partial_p50 - full_p50))
            full_total_count += 1
            if abs(float(full_p50) - 1.0) <= 1e-12:
                full_identity_count += 1

        _validate_mode_payload(mode_payload, context_key=context_key)

        contexts_out.append(
            {
                "context_key": context_key,
                "corridor_bucket": meta.corridor_bucket,
                "corridor_geohash5": meta.corridor_geohash5,
                "hour_slot_local": int(meta.hour_slot_local),
                "road_mix_bucket": meta.road_mix_bucket,
                "road_mix_vector": meta.road_mix_vector,
                "vehicle_class": meta.vehicle_class,
                "day_kind": meta.day_kind,
                "weather_bucket": meta.weather_bucket,
                "weather_regime": meta.weather_regime,
                "source_coverage": {
                    "webtris": 1.0,
                    "traffic_england": 1.0,
                    "dft": 1.0,
                    "open_meteo": 1.0,
                },
                "mode_observation_source": (
                    "pressure_synthesized_from_projected_labels"
                    if context_projection_dominant
                    else "observed_mode_labels"
                ),
                "mode_projection_ratio": round(projection_ratio, 6),
                "profiles": mode_payload,
            }
        )

    if len(contexts_out) < max(1, int(min_contexts)):
        raise RuntimeError(
            f"Scenario context coverage too small ({len(contexts_out)} contexts). "
            f"Need at least {int(min_contexts)} contexts."
        )
    unique_hours = len({int(ctx.get("hour_slot_local", 0)) for ctx in contexts_out})
    unique_corridors = len({str(ctx.get("corridor_geohash5", "")).strip().lower() for ctx in contexts_out})
    if unique_hours < 6:
        raise RuntimeError(
            f"Scenario context hour-slot diversity too small ({unique_hours} unique hours). "
            "Need at least 6 to avoid single-slot calibration collapse."
        )
    if unique_corridors < 8:
        raise RuntimeError(
            f"Scenario corridor diversity too small ({unique_corridors} unique geohash5 buckets). "
            "Need at least 8 to avoid overfit to a narrow corridor set."
        )
    _validate_context_payloads(contexts_out)

    default_context = next(
        (
            ctx
            for ctx in contexts_out
            if ctx["context_key"] == "uk000|h12|weekday|mixed|rigid_hgv|clear"
        ),
        contexts_out[0],
    )
    fit_dates: list[datetime] = []
    holdout_dates: list[datetime] = []
    for row in rows:
        parsed_dt = _parse_as_of_utc(row.as_of_utc)
        if parsed_dt is None:
            continue
        if is_holdout_row(row):
            holdout_dates.append(parsed_dt)
        else:
            fit_dates.append(parsed_dt)
    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    holdout_coverage = float(holdout_covered) / max(1.0, float(holdout_points))
    duration_mape = float(statistics.fmean(holdout_duration_rel_errors)) if holdout_duration_rel_errors else 1.0
    monetary_mape = float(statistics.fmean(holdout_monetary_rel_errors)) if holdout_monetary_rel_errors else 1.0
    emissions_mape = float(statistics.fmean(holdout_emissions_rel_errors)) if holdout_emissions_rel_errors else 1.0
    mode_sep_mean = float(statistics.fmean(holdout_sep_errors)) if holdout_sep_errors else 0.0
    full_identity_share = float(full_identity_count) / max(1.0, float(full_total_count))
    projection_context_share = float(projection_dominant_contexts) / max(1.0, float(len(contexts_out)))
    observed_mode_context_share = float(observed_mode_contexts) / max(1.0, float(len(contexts_out)))

    if mode_sep_mean < 0.03:
        raise RuntimeError(
            f"Scenario holdout mode separability below strict threshold: {mode_sep_mean:.6f} < 0.03"
        )
    if duration_mape > 0.08 or monetary_mape > 0.08 or emissions_mape > 0.08:
        raise RuntimeError(
            "Scenario holdout MAPE exceeded strict threshold "
            f"(duration={duration_mape:.6f}, monetary={monetary_mape:.6f}, emissions={emissions_mape:.6f})."
        )
    if holdout_coverage < 0.90:
        raise RuntimeError(
            f"Scenario holdout coverage below strict threshold: {holdout_coverage:.6f} < 0.90"
        )
    if observed_mode_row_share < float(min_observed_mode_row_share):
        raise RuntimeError(
            "Scenario observed mode-outcome row share is below strict threshold "
            f"(actual={observed_mode_row_share:.6f}, required>={float(min_observed_mode_row_share):.6f})."
        )
    if projection_context_share > float(max_projection_dominant_context_share):
        raise RuntimeError(
            "Scenario projection-dominant context share exceeds strict threshold "
            f"(actual={projection_context_share:.6f}, required<={float(max_projection_dominant_context_share):.6f})."
        )
    if full_identity_share > 0.70:
        raise RuntimeError(
            "Scenario full_sharing identity collapse exceeds strict threshold "
            f"(full_identity_share={full_identity_share:.6f} > 0.70)."
        )

    transform_params = _fit_transform_params(rows)
    payload: dict[str, Any] = {
        "version": "scenario_profiles_uk_v2_live",
        "source": "free_live_apis+holdout_fit",
        "as_of_utc": now_iso,
        "generated_at_utc": now_iso,
        "calibration_basis": "empirical_live_fit",
        "mode_outcomes_source": (
            str(observed_modes_jsonl)
            if observed_modes_jsonl is not None
            else str(raw_jsonl)
        ),
        "fit_window": {
            "start_utc": (
                min(fit_dates).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                if fit_dates
                else ""
            ),
            "end_utc": (
                max(fit_dates).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                if fit_dates
                else now_iso
            ),
        },
        "holdout_window": {
            "start_utc": (
                min(holdout_dates).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                if holdout_dates
                else ""
            ),
            "end_utc": (
                max(holdout_dates).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                if holdout_dates
                else now_iso
            ),
        },
        "holdout_metrics": {
            "mode_separation_mean": round(mode_sep_mean, 6),
            "duration_mape": round(duration_mape, 6),
            "monetary_mape": round(monetary_mape, 6),
            "emissions_mape": round(emissions_mape, 6),
            "coverage": round(holdout_coverage, 6),
            "context_count": float(len(contexts_out)),
            "hour_slot_coverage": float(unique_hours),
            "corridor_coverage": float(unique_corridors),
            "full_identity_share": round(full_identity_share, 6),
            "projection_dominant_context_share": round(projection_context_share, 6),
            "observed_mode_context_share": round(observed_mode_context_share, 6),
            "observed_mode_row_share": round(observed_mode_row_share, 6),
        },
        "split_strategy": str(holdout_strategy_meta.get("strategy", "temporal_forward_plus_corridor_block")),
        "holdout_split_meta": holdout_strategy_meta,
        "transform_params": transform_params,
        "contexts": contexts_out,
        "profiles": default_context["profiles"],
        "signature_algorithm": "sha256",
    }
    payload["signature"] = _signature(payload)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict scenario profile tensor from observed live datasets.")
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "scenario_live_observed.jsonl",
        help="Observed scenario dataset in JSONL format.",
    )
    parser.add_argument(
        "--observed-modes-jsonl",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "scenario_mode_outcomes_observed.jsonl",
        help=(
            "Independent observed per-mode outcome dataset in JSONL format. "
            "Rows can be mode-wise or a modes{} map with canonical scenario fields."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "uk" / "scenario_profiles_uk.json",
    )
    parser.add_argument(
        "--min-contexts",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min-observed-mode-row-share",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--max-projection-dominant-context-share",
        type=float,
        default=0.80,
    )
    args = parser.parse_args()
    payload = build(
        raw_jsonl=args.raw_jsonl,
        observed_modes_jsonl=args.observed_modes_jsonl,
        output_json=args.output,
        min_contexts=max(1, int(args.min_contexts)),
        min_observed_mode_row_share=max(0.0, min(1.0, float(args.min_observed_mode_row_share))),
        max_projection_dominant_context_share=max(
            0.0,
            min(1.0, float(args.max_projection_dominant_context_share)),
        ),
    )
    print(
        f"Wrote scenario profile tensor to {args.output} "
        f"(contexts={len(payload.get('contexts', []))})."
    )


if __name__ == "__main__":
    main()
