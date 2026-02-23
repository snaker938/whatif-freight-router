from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from .live_data_sources import (
    clear_live_data_source_cache,
    live_bank_holidays,
    live_departure_profiles,
    live_fuel_prices,
    live_scenario_context,
    live_scenario_profiles,
    live_stochastic_regimes,
    live_toll_tariffs,
    live_toll_topology,
)
from .model_data_errors import ModelDataError, normalize_reason_code
from .settings import settings


def _strict_runtime_required() -> bool:
    # Pass-3 policy: strict runtime is always on for production code paths.
    return True


def _strict_runtime_test_bypass_enabled() -> bool:
    # Explicit opt-in bypass used for deterministic fixture lanes.
    return os.environ.get("STRICT_RUNTIME_TEST_BYPASS", "0").strip() == "1"


def _pytest_active() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def refresh_live_runtime_route_caches() -> None:
    """Force live-source refreshes for strict route computations."""
    clear_live_data_source_cache()
    load_scenario_profiles.cache_clear()
    load_live_scenario_context.cache_clear()
    load_departure_profile.cache_clear()
    load_uk_bank_holidays.cache_clear()
    load_toll_tariffs.cache_clear()
    load_toll_segments_seed.cache_clear()
    load_toll_confidence_calibration.cache_clear()
    load_fuel_price_snapshot.cache_clear()
    load_stochastic_regimes.cache_clear()
    load_stochastic_residual_prior.cache_clear()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assets_root() -> Path:
    return _repo_root() / "backend" / "assets" / "uk"


def _model_asset_root() -> Path:
    return Path(settings.model_asset_dir)


def _resolve_asset_path(filename: str) -> Path:
    generated = _model_asset_root() / filename
    if generated.exists():
        return generated
    bundled = _assets_root() / filename
    if bundled.exists():
        return bundled
    raise FileNotFoundError(f"model asset not found: {filename}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(raw: str | None) -> datetime | None:
    text = (raw or "").strip()
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


def _infer_as_of_from_payload(payload: dict[str, Any]) -> datetime | None:
    for key in (
        "as_of_utc",
        "as_of",
        "generated_at_utc",
        "generated_at",
        "refreshed_at_utc",
        "updated_at_utc",
    ):
        parsed = _parse_iso_datetime(str(payload.get(key, "")).strip())
        if parsed is not None:
            return parsed
    return None


def _infer_as_of_from_path(path: Path) -> datetime:
    return datetime.fromtimestamp(float(path.stat().st_mtime), tz=UTC)


def _is_fresh(as_of_utc: datetime | None, *, max_age_days: int) -> bool:
    if as_of_utc is None:
        return False
    return (datetime.now(UTC) - as_of_utc) <= timedelta(days=max(1, int(max_age_days)))


def _raise_if_strict_stale(
    *,
    reason_code: str,
    message: str,
    as_of_utc: datetime | None,
    max_age_days: int,
    enforce: bool = True,
) -> None:
    if not enforce:
        return
    if not _strict_runtime_required():
        return
    if _is_fresh(as_of_utc, max_age_days=max_age_days):
        return
    raise ModelDataError(
        reason_code=reason_code,
        message=message,
        details={
            "as_of_utc": as_of_utc.isoformat() if as_of_utc is not None else None,
            "max_age_days": int(max_age_days),
        },
    )


def _strict_empirical_departure_required() -> bool:
    return True


def _strict_empirical_stochastic_required() -> bool:
    return True


def _payload_is_synthetic(payload: dict[str, Any], *, version: str | None = None) -> bool:
    basis = str(payload.get("calibration_basis", "")).strip().lower()
    if basis in {"synthetic", "heuristic", "legacy"}:
        return True
    if version is None:
        version = str(payload.get("version", payload.get("calibration_version", "")))
    lowered = str(version or "").strip().lower()
    return "synthetic" in lowered or "legacy" in lowered


@dataclass(frozen=True)
class ScenarioPolicyProfile:
    duration_multiplier: float
    incident_rate_multiplier: float
    incident_delay_multiplier: float
    fuel_consumption_multiplier: float
    emissions_multiplier: float
    stochastic_sigma_multiplier: float
    quantiles: dict[str, tuple[float, float, float]] | None = None


@dataclass(frozen=True)
class ScenarioContextProfile:
    context_key: str
    corridor_bucket: str
    road_mix_bucket: str
    vehicle_class: str
    day_kind: str
    weather_bucket: str
    profiles: dict[str, ScenarioPolicyProfile]
    source_coverage: dict[str, float] | None = None
    corridor_geohash5: str | None = None
    hour_slot_local: int | None = None
    road_mix_vector: dict[str, float] | None = None
    weather_regime: str | None = None
    mode_observation_source: str | None = None
    mode_projection_ratio: float | None = None


@dataclass(frozen=True)
class ScenarioProfiles:
    source: str
    version: str
    as_of_utc: str | None
    generated_at_utc: str | None
    signature: str | None
    calibration_basis: str
    profiles: dict[str, ScenarioPolicyProfile]
    contexts: dict[str, ScenarioContextProfile] | None = None
    fit_window: dict[str, str] | None = None
    holdout_window: dict[str, str] | None = None
    holdout_metrics: dict[str, float] | None = None
    split_strategy: str | None = None
    transform_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScenarioLiveContext:
    as_of_utc: str
    source_set: dict[str, str]
    coverage: dict[str, float]
    traffic_pressure: float
    incident_pressure: float
    weather_pressure: float
    weather_bucket: str
    diagnostics: dict[str, Any]


def _scenario_field_bounds(field: str) -> tuple[float, float]:
    if field in {"stochastic_sigma_multiplier", "incident_rate_multiplier", "incident_delay_multiplier"}:
        return (0.5, 2.5)
    return (0.6, 2.2)


def _parse_scenario_multiplier(
    raw: Any,
    *,
    field: str,
    mode: str,
) -> tuple[float, float, float]:
    low_bound, high_bound = _scenario_field_bounds(field)
    if isinstance(raw, dict):
        q10 = _safe_float(raw.get("p10"), -1.0)
        q50 = _safe_float(raw.get("p50"), -1.0)
        q90 = _safe_float(raw.get("p90"), -1.0)
    else:
        scalar = _safe_float(raw, -1.0)
        q10 = scalar
        q50 = scalar
        q90 = scalar
    if q10 <= 0.0 or q50 <= 0.0 or q90 <= 0.0:
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message=f"Scenario profile field '{field}' must be > 0 for mode '{mode}'.",
        )
    if not (q10 <= q50 <= q90):
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message=f"Scenario profile quantiles for '{field}' must satisfy p10 <= p50 <= p90 for mode '{mode}'.",
        )
    if q50 < low_bound or q50 > high_bound:
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message=(
                f"Scenario profile p50 for '{field}' is outside plausible strict bounds "
                f"[{low_bound}, {high_bound}] for mode '{mode}'."
            ),
        )
    return (q10, q50, q90)


def _parse_scenario_policy_profile(
    row: dict[str, Any],
    *,
    mode: str,
) -> ScenarioPolicyProfile:
    required_fields = (
        "duration_multiplier",
        "incident_rate_multiplier",
        "incident_delay_multiplier",
        "fuel_consumption_multiplier",
        "emissions_multiplier",
        "stochastic_sigma_multiplier",
    )
    parsed_quantiles: dict[str, tuple[float, float, float]] = {}
    for field in required_fields:
        parsed_quantiles[field] = _parse_scenario_multiplier(
            row.get(field),
            field=field,
            mode=mode,
        )
    return ScenarioPolicyProfile(
        duration_multiplier=parsed_quantiles["duration_multiplier"][1],
        incident_rate_multiplier=parsed_quantiles["incident_rate_multiplier"][1],
        incident_delay_multiplier=parsed_quantiles["incident_delay_multiplier"][1],
        fuel_consumption_multiplier=parsed_quantiles["fuel_consumption_multiplier"][1],
        emissions_multiplier=parsed_quantiles["emissions_multiplier"][1],
        stochastic_sigma_multiplier=parsed_quantiles["stochastic_sigma_multiplier"][1],
        quantiles=parsed_quantiles,
    )


def _validate_scenario_monotonicity(
    *,
    profiles: dict[str, ScenarioPolicyProfile],
    label: str,
) -> None:
    required_modes = ("no_sharing", "partial_sharing", "full_sharing")
    for mode in required_modes:
        if mode not in profiles:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario profile context '{label}' is missing required mode '{mode}'.",
            )
    monotonic_fields = (
        "duration_multiplier",
        "incident_rate_multiplier",
        "incident_delay_multiplier",
        "fuel_consumption_multiplier",
        "emissions_multiplier",
        "stochastic_sigma_multiplier",
    )
    no_share = profiles["no_sharing"]
    partial = profiles["partial_sharing"]
    full = profiles["full_sharing"]
    for field in monotonic_fields:
        no_val = float(getattr(no_share, field))
        partial_val = float(getattr(partial, field))
        full_val = float(getattr(full, field))
        if not (no_val >= partial_val >= full_val):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario profile monotonicity check failed for context "
                    f"'{label}' field '{field}' (expected no_sharing >= partial_sharing >= full_sharing)."
                ),
            )
        # Pass-3 policy: full-sharing p50 pressure multipliers must never exceed 1.0.
        if full_val > 1.0 + 1e-9:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario full_sharing cap violated for context "
                    f"'{label}' field '{field}' (p50 must be <= 1.0)."
                ),
            )


def _canonical_context_token(raw: str, *, fallback: str) -> str:
    token = str(raw or "").strip().lower().replace(" ", "_")
    return token or fallback


def _scenario_context_key(
    *,
    corridor_bucket: str,
    road_mix_bucket: str,
    vehicle_class: str,
    day_kind: str,
    weather_bucket: str,
) -> str:
    return (
        f"{_canonical_context_token(corridor_bucket, fallback='uk_default')}|"
        f"{_canonical_context_token(road_mix_bucket, fallback='mixed')}|"
        f"{_canonical_context_token(vehicle_class, fallback='rigid_hgv')}|"
        f"{_canonical_context_token(day_kind, fallback='weekday')}|"
        f"{_canonical_context_token(weather_bucket, fallback='clear')}"
    )


def _parse_scenario_contexts(
    payload: dict[str, Any],
) -> dict[str, ScenarioContextProfile]:
    contexts_raw = payload.get("contexts")
    if contexts_raw is None:
        return {}
    parsed_contexts: dict[str, ScenarioContextProfile] = {}
    context_items: list[dict[str, Any]] = []
    if isinstance(contexts_raw, dict):
        for context_key, row in contexts_raw.items():
            if isinstance(row, dict):
                item = dict(row)
                item.setdefault("context_key", str(context_key))
                context_items.append(item)
    elif isinstance(contexts_raw, list):
        context_items = [row for row in contexts_raw if isinstance(row, dict)]
    else:
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Scenario profile payload 'contexts' must be a list or map when provided.",
        )

    for row in context_items:
        corridor_bucket = _canonical_context_token(row.get("corridor_bucket", "uk_default"), fallback="uk_default")
        corridor_geohash5 = _canonical_context_token(
            row.get("corridor_geohash5", corridor_bucket),
            fallback="uk000",
        )
        road_mix_bucket = _canonical_context_token(row.get("road_mix_bucket", "mixed"), fallback="mixed")
        vehicle_class = _canonical_context_token(row.get("vehicle_class", "rigid_hgv"), fallback="rigid_hgv")
        day_kind = _canonical_context_token(row.get("day_kind", "weekday"), fallback="weekday")
        weather_bucket = _canonical_context_token(row.get("weather_bucket", "clear"), fallback="clear")
        weather_regime = _canonical_context_token(
            row.get("weather_regime", weather_bucket),
            fallback=weather_bucket,
        )
        hour_slot_local = int(max(0, min(23, int(_safe_float(row.get("hour_slot_local"), 12.0)))))
        road_mix_vector_raw = row.get("road_mix_vector")
        road_mix_vector: dict[str, float] | None = None
        if isinstance(road_mix_vector_raw, dict):
            road_mix_vector = {}
            for road_key, road_value in road_mix_vector_raw.items():
                road_mix_vector[str(road_key).strip().lower()] = max(
                    0.0,
                    float(_safe_float(road_value, 0.0)),
                )
        if not road_mix_vector:
            road_mix_vector = {road_mix_bucket: 1.0}
        vec_total = max(sum(max(0.0, float(v)) for v in road_mix_vector.values()), 1e-9)
        road_mix_vector = {k: max(0.0, float(v)) / vec_total for k, v in road_mix_vector.items()}
        context_key = str(row.get("context_key", "")).strip()
        if not context_key:
            context_key = (
                f"{corridor_geohash5}|h{hour_slot_local:02d}|{day_kind}|"
                f"{road_mix_bucket}|{vehicle_class}|{weather_regime}"
            )
        profiles_raw = row.get("profiles", {})
        if not isinstance(profiles_raw, dict):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario profile context '{context_key}' is missing 'profiles'.",
            )
        mode_profiles: dict[str, ScenarioPolicyProfile] = {}
        for mode, mode_row in profiles_raw.items():
            if not isinstance(mode_row, dict):
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=f"Scenario profile context '{context_key}' mode '{mode}' is invalid.",
                )
            mode_profiles[str(mode)] = _parse_scenario_policy_profile(mode_row, mode=str(mode))
        _validate_scenario_monotonicity(profiles=mode_profiles, label=context_key)

        source_cov: dict[str, float] | None = None
        source_cov_raw = row.get("source_coverage")
        if isinstance(source_cov_raw, dict):
            source_cov = {}
            for source_name, source_value in source_cov_raw.items():
                source_cov[str(source_name)] = max(0.0, min(1.0, _safe_float(source_value, 0.0)))

        parsed_contexts[context_key] = ScenarioContextProfile(
            context_key=context_key,
            corridor_bucket=corridor_bucket,
            road_mix_bucket=road_mix_bucket,
            vehicle_class=vehicle_class,
            day_kind=day_kind,
            weather_bucket=weather_bucket,
            profiles=mode_profiles,
            source_coverage=source_cov,
            corridor_geohash5=corridor_geohash5,
            hour_slot_local=hour_slot_local,
            road_mix_vector=road_mix_vector,
            weather_regime=weather_regime,
            mode_observation_source=(
                str(row.get("mode_observation_source", "")).strip() or None
            ),
            mode_projection_ratio=(
                max(0.0, min(1.0, _safe_float(row.get("mode_projection_ratio"), 0.0)))
                if row.get("mode_projection_ratio") is not None
                else None
            ),
        )
    return parsed_contexts


def _default_scenario_transform_params() -> dict[str, Any]:
    return {
        "live_feature_transform": {
            "traffic_pressure": {
                "bias": 0.0,
                "weights": {"flow_index": 0.5, "speed_inverse": 0.5},
                "min": 0.55,
                "max": 2.8,
            },
            "incident_pressure": {
                "bias": 0.0,
                "weights": {"delay_pressure": 0.6, "severity_index": 0.4},
                "min": 0.55,
                "max": 3.2,
            },
            "weather_pressure": {
                "bias": 0.0,
                "weights": {"weather_severity_index": 1.0},
                "min": 0.55,
                "max": 2.8,
            },
        },
        "mode_effect_scale": {
            "no_sharing": 1.00,
            "partial_sharing": 0.80,
            "full_sharing": 0.60,
        },
        "policy_adjustment": {
            "duration_multiplier": {
                "weights": {"traffic_pressure": 0.34, "incident_pressure": 0.33, "weather_pressure": 0.33},
                "gain": 0.35,
                "min": 0.55,
                "max": 3.0,
            },
            "incident_rate_multiplier": {
                "weights": {"traffic_pressure": 0.34, "incident_pressure": 0.33, "weather_pressure": 0.33},
                "gain": 0.35,
                "min": 0.55,
                "max": 3.0,
            },
            "incident_delay_multiplier": {
                "weights": {"traffic_pressure": 0.34, "incident_pressure": 0.33, "weather_pressure": 0.33},
                "gain": 0.35,
                "min": 0.55,
                "max": 3.0,
            },
            "fuel_consumption_multiplier": {
                "weights": {"traffic_pressure": 0.34, "incident_pressure": 0.33, "weather_pressure": 0.33},
                "gain": 0.35,
                "min": 0.55,
                "max": 3.0,
            },
            "emissions_multiplier": {
                "weights": {"traffic_pressure": 0.34, "incident_pressure": 0.33, "weather_pressure": 0.33},
                "gain": 0.35,
                "min": 0.55,
                "max": 3.0,
            },
            "stochastic_sigma_multiplier": {
                "weights": {"traffic_pressure": 0.34, "incident_pressure": 0.33, "weather_pressure": 0.33},
                "gain": 0.35,
                "min": 0.45,
                "max": 3.0,
            },
        },
        "context_similarity": {
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
        },
        "scenario_edge_scaling_version": "v4_neutral_fallback",
        "fit_strategy": "neutral_fallback_non_strict",
    }


def _parse_scenario_transform_params(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("transform_params")
    defaults = _default_scenario_transform_params()
    strict_runtime = _strict_runtime_required()
    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    strict = strict_runtime and not pytest_bypass_enabled
    if not isinstance(raw, dict):
        if strict:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario profile payload must include transform_params in strict runtime.",
            )
        return defaults

    out = json.loads(json.dumps(defaults))
    live_transform_raw = raw.get("live_feature_transform")
    mode_scale_raw = raw.get("mode_effect_scale")
    policy_adjustment_raw = raw.get("policy_adjustment")
    context_similarity_raw = raw.get("context_similarity")
    if strict:
        if (
            not isinstance(live_transform_raw, dict)
            or not isinstance(mode_scale_raw, dict)
            or not isinstance(policy_adjustment_raw, dict)
            or not isinstance(context_similarity_raw, dict)
        ):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario transform_params must include live_feature_transform, "
                    "mode_effect_scale, policy_adjustment, and context_similarity in strict runtime."
                ),
            )
    if isinstance(live_transform_raw, dict):
        expected_pressure_features = {
            "traffic_pressure": {"flow_index", "speed_inverse"},
            "incident_pressure": {"delay_pressure", "severity_index"},
            "weather_pressure": {"weather_severity_index"},
        }
        for name in ("traffic_pressure", "incident_pressure", "weather_pressure"):
            row = live_transform_raw.get(name)
            if not isinstance(row, dict):
                if strict:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=f"Scenario transform_params.live_feature_transform.{name} is missing in strict runtime.",
                    )
                continue
            if strict:
                for key in ("bias", "weights", "min", "max"):
                    if key not in row:
                        raise ModelDataError(
                            reason_code="scenario_profile_invalid",
                            message=(
                                f"Scenario transform_params.live_feature_transform.{name}.{key} "
                                "is required in strict runtime."
                            ),
                        )
            weights = row.get("weights")
            if not isinstance(weights, dict):
                weights = {}
            parsed_weights = {
                str(k): max(0.0, float(_safe_float(v, 0.0)))
                for k, v in weights.items()
                if str(k).strip()
            }
            if strict:
                expected = expected_pressure_features.get(name, set())
                actual = {str(k).strip() for k in parsed_weights.keys() if str(k).strip()}
                if actual != expected:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=(
                            f"Scenario transform_params.live_feature_transform.{name}.weights "
                            "must match expected calibrated feature keys in strict runtime."
                        ),
                        details={"expected": sorted(expected), "actual": sorted(actual)},
                    )
            total = sum(parsed_weights.values())
            if total <= 0.0:
                if strict:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=f"Scenario transform_params.live_feature_transform.{name}.weights must be non-empty in strict runtime.",
                    )
                parsed_weights = dict(out["live_feature_transform"][name]["weights"])
                total = sum(parsed_weights.values())
            min_value = float(_safe_float(row.get("min"), out["live_feature_transform"][name]["min"]))
            max_value = float(_safe_float(row.get("max"), out["live_feature_transform"][name]["max"]))
            if max_value < min_value:
                if strict:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=f"Scenario transform_params.live_feature_transform.{name}.max must be >= min.",
                    )
                max_value = min_value
            out["live_feature_transform"][name] = {
                "bias": float(_safe_float(row.get("bias"), 0.0)),
                "weights": {
                    key: float(value) / max(total, 1e-9)
                    for key, value in parsed_weights.items()
                },
                "min": min_value,
                "max": max_value,
            }
    if isinstance(mode_scale_raw, dict):
        for mode in ("no_sharing", "partial_sharing", "full_sharing"):
            if mode not in mode_scale_raw:
                if strict:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=f"Scenario transform_params.mode_effect_scale missing mode '{mode}'.",
                    )
                continue
            out["mode_effect_scale"][mode] = max(0.0, float(_safe_float(mode_scale_raw.get(mode), out["mode_effect_scale"][mode])))
    if isinstance(policy_adjustment_raw, dict):
        expected_policy_pressure_keys = {"traffic_pressure", "incident_pressure", "weather_pressure"}
        for field in (
            "duration_multiplier",
            "incident_rate_multiplier",
            "incident_delay_multiplier",
            "fuel_consumption_multiplier",
            "emissions_multiplier",
            "stochastic_sigma_multiplier",
        ):
            row = policy_adjustment_raw.get(field)
            if not isinstance(row, dict):
                if strict:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=f"Scenario transform_params.policy_adjustment missing field '{field}'.",
                    )
                continue
            if strict:
                for key in ("weights", "gain", "min", "max"):
                    if key not in row:
                        raise ModelDataError(
                            reason_code="scenario_profile_invalid",
                            message=(
                                f"Scenario transform_params.policy_adjustment.{field}.{key} "
                                "is required in strict runtime."
                            ),
                        )
            weights = row.get("weights")
            if not isinstance(weights, dict):
                weights = {}
            parsed_weights = {
                str(k): max(0.0, float(_safe_float(v, 0.0)))
                for k, v in weights.items()
                if str(k).strip()
            }
            if strict:
                actual = {str(k).strip() for k in parsed_weights.keys() if str(k).strip()}
                if actual != expected_policy_pressure_keys:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=(
                            f"Scenario transform_params.policy_adjustment.{field}.weights "
                            "must match expected pressure keys in strict runtime."
                        ),
                        details={"expected": sorted(expected_policy_pressure_keys), "actual": sorted(actual)},
                    )
            total = sum(parsed_weights.values())
            if total <= 0.0:
                if strict:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=f"Scenario transform_params.policy_adjustment.{field}.weights must be non-empty in strict runtime.",
                    )
                parsed_weights = dict(out["policy_adjustment"][field]["weights"])
                total = sum(parsed_weights.values())
            min_v = float(_safe_float(row.get("min"), out["policy_adjustment"][field]["min"]))
            max_v = float(_safe_float(row.get("max"), out["policy_adjustment"][field]["max"]))
            if max_v < min_v:
                if strict:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=f"Scenario transform_params.policy_adjustment.{field}.max must be >= min.",
                    )
                max_v = min_v
            out["policy_adjustment"][field] = {
                "weights": {
                    key: float(value) / max(total, 1e-9)
                    for key, value in parsed_weights.items()
                },
                "gain": max(0.0, float(_safe_float(row.get("gain"), out["policy_adjustment"][field]["gain"]))),
                "min": min_v,
                "max": max_v,
            }
    if isinstance(context_similarity_raw, dict):
        strategy = str(context_similarity_raw.get("strategy", out["context_similarity"]["strategy"])).strip() or out[
            "context_similarity"
        ]["strategy"]
        weights_raw = context_similarity_raw.get("weights", {})
        if strict and not isinstance(weights_raw, dict):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario transform_params.context_similarity.weights must be an object in strict runtime.",
            )
        if not isinstance(weights_raw, dict):
            weights_raw = {}
        expected_similarity_keys = {
            "geo_distance",
            "hour_distance",
            "day_penalty",
            "weather_penalty",
            "road_penalty",
            "vehicle_penalty",
            "road_mix_distance",
        }
        parsed_weights = {
            key: max(0.0, float(_safe_float(weights_raw.get(key), float("nan"))))
            for key in expected_similarity_keys
        }
        if strict:
            for key, value in parsed_weights.items():
                if not (value == value):
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=(
                            "Scenario transform_params.context_similarity.weights "
                            f"missing key '{key}' in strict runtime."
                        ),
                    )
        normalized_weights = {
            key: (0.0 if not (value == value) else float(value))
            for key, value in parsed_weights.items()
        }
        weight_sum = sum(normalized_weights.values())
        if weight_sum <= 0.0:
            if strict:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message="Scenario transform_params.context_similarity.weights must contain positive mass.",
                )
            normalized_weights = dict(out["context_similarity"]["weights"])
            weight_sum = sum(normalized_weights.values())
        normalized_weights = {key: float(value) / max(weight_sum, 1e-9) for key, value in normalized_weights.items()}
        max_distance = float(
            _safe_float(context_similarity_raw.get("max_distance"), out["context_similarity"]["max_distance"])
        )
        max_distance = max(0.01, min(10.0, max_distance))
        out["context_similarity"] = {
            "strategy": strategy,
            "weights": normalized_weights,
            "max_distance": max_distance,
        }
    if isinstance(raw.get("scenario_edge_scaling_version"), str) and raw.get("scenario_edge_scaling_version"):
        out["scenario_edge_scaling_version"] = str(raw.get("scenario_edge_scaling_version"))
    if isinstance(raw.get("fit_strategy"), str) and raw.get("fit_strategy"):
        out["fit_strategy"] = str(raw.get("fit_strategy"))

    live_transform = out.get("live_feature_transform")
    mode_scale = out.get("mode_effect_scale")
    policy_adjustment = out.get("policy_adjustment")
    context_similarity = out.get("context_similarity")
    if (
        not isinstance(live_transform, dict)
        or not isinstance(mode_scale, dict)
        or not isinstance(policy_adjustment, dict)
        or not isinstance(context_similarity, dict)
    ):
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Scenario transform_params sections are invalid.",
        )
    for mode in ("no_sharing", "partial_sharing", "full_sharing"):
        if mode not in mode_scale:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario transform_params.mode_effect_scale missing mode '{mode}'.",
            )
        mode_scale[mode] = max(0.0, float(_safe_float(mode_scale[mode], 0.0)))

    # Full sharing must never amplify pressure impact above neutral.
    if float(mode_scale["full_sharing"]) > 1.0:
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Scenario transform_params.mode_effect_scale.full_sharing must be <= 1.0.",
        )
    if strict:
        fit_strategy = str(out.get("fit_strategy", "")).strip().lower()
        scaling_version = str(out.get("scenario_edge_scaling_version", "")).strip().lower()
        if not fit_strategy or "fallback" in fit_strategy or "heuristic" in fit_strategy:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario transform fit_strategy must be empirical/non-fallback in strict runtime.",
                details={"fit_strategy": out.get("fit_strategy")},
            )
        if not scaling_version or "fallback" in scaling_version:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario transform scenario_edge_scaling_version cannot be fallback in strict runtime.",
                details={"scenario_edge_scaling_version": out.get("scenario_edge_scaling_version")},
            )

    return out


def _parse_scenario_profiles_payload(
    payload: dict[str, Any],
    *,
    source: str,
) -> ScenarioProfiles:
    version = str(payload.get("version", "")).strip() or "scenario_profiles_unknown"
    if _payload_is_synthetic(payload, version=version):
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Scenario profile payload cannot use synthetic/heuristic/legacy calibration basis in strict runtime.",
        )
    signature = _validate_generic_signature(
        payload=payload,
        reason_code="scenario_profile_invalid",
        message_prefix="Scenario profile payload",
        require_signature=bool(settings.scenario_require_signature and _strict_runtime_required()),
    )

    profiles_raw = payload.get("profiles", {})
    parsed_profiles: dict[str, ScenarioPolicyProfile] = {}
    if isinstance(profiles_raw, dict) and profiles_raw:
        for mode, row in profiles_raw.items():
            if not isinstance(row, dict):
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=f"Scenario profile mode '{mode}' must be an object.",
                )
            parsed_profiles[str(mode)] = _parse_scenario_policy_profile(row, mode=str(mode))
        _validate_scenario_monotonicity(profiles=parsed_profiles, label="global")

    contexts = _parse_scenario_contexts(payload)
    if not parsed_profiles and contexts:
        # Legacy global-profile backfill is disabled in strict runtime.
        if not _strict_runtime_required():
            preferred_keys = [
                "uk000|h12|weekday|mixed|rigid_hgv|clear",
                "uk000|h12|weekday|mixed|default|clear",
            ]
            default_ctx = None
            for key in preferred_keys:
                default_ctx = contexts.get(key)
                if default_ctx is not None:
                    break
            if default_ctx is None:
                default_ctx = next(iter(contexts.values()))
            parsed_profiles = dict(default_ctx.profiles)

    if not parsed_profiles and not contexts:
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Scenario profile payload must provide global profiles or contextual profiles.",
        )

    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    strict_quality_gates = _strict_runtime_required() and not pytest_bypass_enabled
    if strict_quality_gates:
        # Strict live-only policy: scenario modes must be fully observed at train time.
        strict_min_observed_mode_row_share = 1.0
        strict_max_projection_context_share = 0.0
        split_strategy = str(payload.get("split_strategy", "")).strip().lower()
        if split_strategy != "temporal_forward_plus_corridor_block":
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario profile payload split_strategy must be "
                    "'temporal_forward_plus_corridor_block' in strict runtime."
                ),
                details={"split_strategy": split_strategy or None},
            )
        holdout_metrics_raw = payload.get("holdout_metrics", {})
        if not isinstance(holdout_metrics_raw, dict):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario profile payload is missing holdout_metrics in strict runtime.",
            )
        mode_separation = float(_safe_float(holdout_metrics_raw.get("mode_separation_mean"), 0.0))
        duration_mape = float(_safe_float(holdout_metrics_raw.get("duration_mape"), 1.0))
        monetary_mape = float(_safe_float(holdout_metrics_raw.get("monetary_mape"), 1.0))
        emissions_mape = float(_safe_float(holdout_metrics_raw.get("emissions_mape"), 1.0))
        coverage = float(_safe_float(holdout_metrics_raw.get("coverage"), 0.0))
        hour_slot_cov = float(_safe_float(holdout_metrics_raw.get("hour_slot_coverage"), 0.0))
        corridor_cov = float(_safe_float(holdout_metrics_raw.get("corridor_coverage"), 0.0))
        full_identity_share = float(_safe_float(holdout_metrics_raw.get("full_identity_share"), float("nan")))
        projection_context_share = float(
            _safe_float(holdout_metrics_raw.get("projection_dominant_context_share"), float("nan"))
        )
        observed_mode_row_share = float(
            _safe_float(holdout_metrics_raw.get("observed_mode_row_share"), float("nan"))
        )
        if mode_separation < 0.03:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario holdout mode separability is below strict threshold (>= 0.03 required).",
                details={"mode_separation_mean": mode_separation},
            )
        if duration_mape > 0.08 or monetary_mape > 0.08 or emissions_mape > 0.08:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario holdout MAPE exceeds strict threshold "
                    "(duration/monetary/emissions <= 0.08 required)."
                ),
                details={
                    "duration_mape": duration_mape,
                    "monetary_mape": monetary_mape,
                    "emissions_mape": emissions_mape,
                },
            )
        if coverage < 0.90:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario holdout coverage is below strict threshold (>= 0.90 required).",
                details={"coverage": coverage},
            )
        if full_identity_share == full_identity_share and full_identity_share > 0.70:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario full_sharing identity share exceeds strict threshold (<= 0.70 required).",
                details={"full_identity_share": full_identity_share},
            )
        if observed_mode_row_share != observed_mode_row_share:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario holdout_metrics.observed_mode_row_share is required in strict runtime."
                ),
            )
        if observed_mode_row_share < float(strict_min_observed_mode_row_share):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario observed mode row share is below strict threshold "
                    f"(actual={observed_mode_row_share:.6f}, "
                    f"required>={float(strict_min_observed_mode_row_share):.6f})."
                ),
                details={"observed_mode_row_share": observed_mode_row_share},
            )
        if projection_context_share != projection_context_share:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario holdout_metrics.projection_dominant_context_share is required in strict runtime."
                ),
            )
        if projection_context_share > float(strict_max_projection_context_share):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=(
                    "Scenario projection-dominant context share exceeds strict threshold "
                    f"(actual={projection_context_share:.6f}, "
                    f"required<={float(strict_max_projection_context_share):.6f})."
                ),
                details={"projection_dominant_context_share": projection_context_share},
            )
        if hour_slot_cov < 6.0 or corridor_cov < 8.0:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario holdout diversity is below strict thresholds (hour>=6, corridor>=8).",
                details={
                    "hour_slot_coverage": hour_slot_cov,
                    "corridor_coverage": corridor_cov,
                },
            )
        if contexts:
            observed_hours: set[int] = set()
            for ctx in contexts.values():
                if ctx.hour_slot_local is None:
                    continue
                try:
                    observed_hours.add(int(ctx.hour_slot_local))
                except (TypeError, ValueError):
                    continue
            actual_hours = len(observed_hours)
            actual_corridors = len(
                {
                    str(ctx.corridor_geohash5).strip().lower()
                    for ctx in contexts.values()
                    if str(ctx.corridor_geohash5).strip()
                }
            )
            if actual_hours < 6 or actual_corridors < 8:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message="Scenario context tensor coverage is too narrow for strict runtime.",
                    details={
                        "actual_hour_slots": actual_hours,
                        "actual_corridor_coverage": actual_corridors,
                    },
                )
            full_identity_count = 0
            full_total = 0
            for ctx in contexts.values():
                full_profile = ctx.profiles.get("full_sharing")
                if full_profile is None:
                    continue
                for field in (
                    "duration_multiplier",
                    "incident_rate_multiplier",
                    "incident_delay_multiplier",
                    "fuel_consumption_multiplier",
                    "emissions_multiplier",
                    "stochastic_sigma_multiplier",
                ):
                    full_total += 1
                    if abs(float(getattr(full_profile, field)) - 1.0) <= 1e-12:
                        full_identity_count += 1
            if full_total > 0:
                actual_full_identity_share = float(full_identity_count) / float(full_total)
                if actual_full_identity_share > 0.70:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message="Scenario contextual full_sharing identity collapse exceeds strict threshold.",
                        details={
                            "full_identity_share": actual_full_identity_share,
                            "full_identity_count": full_identity_count,
                            "full_total": full_total,
                        },
                    )
            projection_context_count = 0
            for ctx in contexts.values():
                projected_ratio = getattr(ctx, "mode_projection_ratio", None)
                projected_source = str(getattr(ctx, "mode_observation_source", "") or "").strip().lower()
                is_projected_context = False
                if projected_ratio is not None:
                    is_projected_context = float(projected_ratio) > 0.0
                if (not is_projected_context) and projected_source:
                    is_projected_context = any(
                        token in projected_source for token in ("project", "synth", "heuristic", "legacy")
                    )
                if projected_ratio is None and not projected_source:
                    # Missing provenance in strict runtime is treated as projected to avoid overclaiming realism.
                    is_projected_context = True
                if is_projected_context:
                    projection_context_count += 1
            if contexts:
                actual_projection_context_share = float(projection_context_count) / float(len(contexts))
                if actual_projection_context_share > float(strict_max_projection_context_share):
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message="Scenario context projection dominance exceeds strict threshold.",
                        details={
                            "projection_dominant_context_share": actual_projection_context_share,
                            "projection_context_count": projection_context_count,
                            "context_count": len(contexts),
                        },
                    )

    return ScenarioProfiles(
        source=source,
        version=version,
        as_of_utc=str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None,
        generated_at_utc=(
            str(payload.get("generated_at_utc", payload.get("generated_at", ""))).strip() or None
        ),
        signature=signature,
        calibration_basis=str(payload.get("calibration_basis", "")).strip() or "empirical",
        profiles=parsed_profiles,
        contexts=contexts or None,
        fit_window=(payload.get("fit_window") if isinstance(payload.get("fit_window"), dict) else None),
        holdout_window=(payload.get("holdout_window") if isinstance(payload.get("holdout_window"), dict) else None),
        holdout_metrics=(
            {
                str(k): float(v)
                for k, v in payload.get("holdout_metrics", {}).items()
                if isinstance(payload.get("holdout_metrics"), dict)
                and isinstance(v, (int, float))
            }
            if isinstance(payload.get("holdout_metrics"), dict)
            else None
        ),
        split_strategy=str(payload.get("split_strategy", "")).strip() or None,
        transform_params=_parse_scenario_transform_params(payload),
    )


@lru_cache(maxsize=1)
def load_scenario_profiles() -> ScenarioProfiles:
    strict = _strict_runtime_required()
    last_invalid: ModelDataError | None = None
    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    enforce_freshness = strict and not pytest_bypass_enabled
    live_coeff_url = str(settings.live_scenario_coefficient_url or "").strip()
    # Legacy LIVE_SCENARIO_PROFILE_URL compatibility has been removed from the
    # strict scenario coefficient path to keep source policy explicit.

    if strict and bool(settings.live_scenario_require_url_in_strict) and not live_coeff_url and not pytest_bypass_enabled:
        raise ModelDataError(
            reason_code="scenario_profile_unavailable",
            message="LIVE_SCENARIO_COEFFICIENT_URL is required in strict runtime.",
        )

    if settings.live_runtime_data_enabled and live_coeff_url:
        live_payload = live_scenario_profiles()
        if isinstance(live_payload, dict) and "_live_error" in live_payload:
            err = live_payload.get("_live_error", {})
            if isinstance(err, dict):
                last_invalid = ModelDataError(
                    reason_code=normalize_reason_code(
                        str(err.get("reason_code", "scenario_profile_unavailable")),
                        default="scenario_profile_unavailable",
                    ),
                    message=str(err.get("message", "Live scenario coefficient source error")).strip()
                    or "Live scenario coefficient source error",
                    details=(err.get("diagnostics") if isinstance(err.get("diagnostics"), dict) else None),
                )
        elif isinstance(live_payload, dict):
            try:
                parsed = _parse_scenario_profiles_payload(
                    live_payload,
                    source="live_runtime:scenario_coefficients",
                )
                _raise_if_strict_stale_minutes(
                    reason_code="scenario_profile_unavailable",
                    message="Live scenario coefficient payload is stale for strict runtime policy.",
                    as_of_utc=_infer_as_of_from_payload(live_payload),
                    max_age_minutes=int(settings.live_scenario_coefficient_max_age_minutes),
                    enforce=enforce_freshness,
                )
                return parsed
            except ModelDataError as exc:
                last_invalid = exc

    if last_invalid is not None:
        raise last_invalid
    raise ModelDataError(
        reason_code="scenario_profile_unavailable",
        message="Live scenario coefficient profile is unavailable for strict runtime.",
    )


def _is_fresh_minutes(as_of_utc: datetime | None, *, max_age_minutes: int) -> bool:
    if as_of_utc is None:
        return False
    return (datetime.now(UTC) - as_of_utc) <= timedelta(minutes=max(1, int(max_age_minutes)))


def _raise_if_strict_stale_minutes(
    *,
    reason_code: str,
    message: str,
    as_of_utc: datetime | None,
    max_age_minutes: int,
    enforce: bool = True,
) -> None:
    if not enforce:
        return
    if not _strict_runtime_required():
        return
    if _is_fresh_minutes(as_of_utc, max_age_minutes=max_age_minutes):
        return
    raise ModelDataError(
        reason_code=reason_code,
        message=message,
        details={
            "as_of_utc": as_of_utc.isoformat() if as_of_utc is not None else None,
            "max_age_minutes": int(max_age_minutes),
        },
    )


def _parse_live_scenario_context_payload(
    payload: dict[str, Any],
    *,
    source_label: str,
    transform_params: dict[str, Any] | None = None,
) -> ScenarioLiveContext:
    strict = _strict_runtime_required()
    strict_schema = strict
    if _strict_runtime_test_bypass_enabled() and source_label != "live_runtime:scenario_context":
        # Deterministic fixture lanes may load legacy signed snapshots that do
        # not carry full live diagnostics/transform metadata.
        strict_schema = False
    as_of_utc = str(payload.get("as_of_utc", "")).strip()
    if not as_of_utc:
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Live scenario context payload is missing as_of_utc.",
        )
    source_set_raw = payload.get("source_set")
    if not isinstance(source_set_raw, dict):
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Live scenario context payload is missing source_set.",
        )
    source_set = {
        str(key): str(value)
        for key, value in source_set_raw.items()
        if str(key).strip() and str(value).strip()
    }
    required_sources = {"webtris", "traffic_england", "dft_counts", "open_meteo"}
    if not required_sources.issubset(set(source_set.keys())):
        raise ModelDataError(
            reason_code="scenario_profile_unavailable",
            message="Live scenario context payload missing required source coverage.",
            details={"missing_sources": sorted(required_sources - set(source_set.keys()))},
        )
    coverage_raw = payload.get("coverage", {})
    if not isinstance(coverage_raw, dict):
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Live scenario context payload is missing coverage.",
        )
    coverage = {
        "webtris": max(0.0, min(1.0, _safe_float(coverage_raw.get("webtris"), 0.0))),
        "traffic_england": max(0.0, min(1.0, _safe_float(coverage_raw.get("traffic_england"), 0.0))),
        "dft": max(0.0, min(1.0, _safe_float(coverage_raw.get("dft"), 0.0))),
        "open_meteo": max(0.0, min(1.0, _safe_float(coverage_raw.get("open_meteo"), 0.0))),
        "overall": max(0.0, min(1.0, _safe_float(coverage_raw.get("overall"), 0.0))),
    }
    if coverage["overall"] < 0.999:
        raise ModelDataError(
            reason_code="scenario_profile_unavailable",
            message="Live scenario context coverage is incomplete for strict runtime.",
            details={"coverage": coverage},
        )

    traffic_raw = payload.get("traffic_features", {})
    incident_raw = payload.get("incident_features", {})
    weather_raw = payload.get("weather_features", {})
    if not isinstance(traffic_raw, dict) or not isinstance(incident_raw, dict) or not isinstance(weather_raw, dict):
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Live scenario context payload is missing feature blocks.",
        )
    source_diag_raw = payload.get("source_diagnostics", {})
    source_diagnostics = source_diag_raw if isinstance(source_diag_raw, dict) else {}
    if strict_schema:
        required_diag_sources = ("webtris", "traffic_england", "dft_counts", "open_meteo")
        required_fetch_keys = ("source_url", "fetch_error", "cache_hit", "stale_cache_used", "status_code", "as_of_utc")
        for source_key in required_diag_sources:
            source_row = source_diagnostics.get(source_key)
            if not isinstance(source_row, dict):
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=(
                        "Live scenario context payload is missing source diagnostics "
                        f"for '{source_key}' in strict runtime."
                    ),
                )
            fetch_row = source_row.get("fetch")
            if not isinstance(fetch_row, dict):
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=(
                        "Live scenario context payload is missing fetch diagnostics "
                        f"for '{source_key}' in strict runtime."
                    ),
                )
            missing_fetch_fields = [key for key in required_fetch_keys if key not in fetch_row]
            if missing_fetch_fields:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=(
                        "Live scenario context fetch diagnostics are incomplete for strict runtime "
                        f"({source_key}: missing {','.join(missing_fetch_fields)})."
                    ),
                )
    def _required_numeric(
        block: dict[str, Any],
        key: str,
        *,
        default: float,
        low: float,
        high: float,
        block_name: str,
    ) -> float:
        raw_value = block.get(key, None)
        parsed = _safe_float(raw_value, float("nan"))
        if strict_schema and not isinstance(raw_value, (int, float)) and not (isinstance(raw_value, str) and str(raw_value).strip()):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Live scenario context payload missing required numeric field {block_name}.{key}.",
            )
        if strict_schema and not (parsed == parsed):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Live scenario context payload field {block_name}.{key} is non-numeric in strict runtime.",
            )
        if not (parsed == parsed):
            parsed = default
        return max(low, min(high, parsed))

    flow_index = _required_numeric(
        traffic_raw,
        "flow_index",
        default=0.0,
        low=0.0,
        high=1_000_000.0,
        block_name="traffic_features",
    )
    speed_index = _required_numeric(
        traffic_raw,
        "speed_index",
        default=0.0,
        low=0.0,
        high=400.0,
        block_name="traffic_features",
    )
    delay_pressure = _required_numeric(
        incident_raw,
        "delay_pressure",
        default=0.0,
        low=0.0,
        high=100_000.0,
        block_name="incident_features",
    )
    severity_index = _required_numeric(
        incident_raw,
        "severity_index",
        default=0.0,
        low=0.0,
        high=1_000.0,
        block_name="incident_features",
    )
    weather_severity_index = _required_numeric(
        weather_raw,
        "weather_severity_index",
        default=0.0,
        low=0.0,
        high=10_000.0,
        block_name="weather_features",
    )
    weather_bucket = _canonical_context_token(weather_raw.get("weather_bucket", "clear"), fallback="clear")
    speed_inverse = 1.0 / max(1.0, speed_index)

    if transform_params is None:
        if _strict_runtime_required():
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message="Scenario transform parameters are required to parse live scenario context in strict runtime.",
            )
        transform = _default_scenario_transform_params()
    else:
        transform = transform_params
    live_transform_raw = transform.get("live_feature_transform", {})
    if strict_schema and not isinstance(live_transform_raw, dict):
        raise ModelDataError(
            reason_code="scenario_profile_invalid",
            message="Scenario transform live_feature_transform block is required in strict runtime.",
        )
    live_transform = live_transform_raw if isinstance(live_transform_raw, dict) else {}

    def _pressure_transform(
        *,
        name: str,
        feature_names: tuple[str, ...],
        default_min: float,
        default_max: float,
    ) -> tuple[float, dict[str, float], float, float]:
        row = live_transform.get(name, {})
        if strict_schema and not isinstance(row, dict):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario live transform row '{name}' must be an object in strict runtime.",
            )
        if not isinstance(row, dict):
            row = {}
        weights_raw = row.get("weights", {})
        if strict_schema and not isinstance(weights_raw, dict):
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario live transform row '{name}.weights' must be an object in strict runtime.",
            )
        if not isinstance(weights_raw, dict):
            weights_raw = {}

        if strict_schema:
            missing_fields = [field for field in ("bias", "min", "max") if field not in row]
            if missing_fields:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=(
                        f"Scenario live transform row '{name}' is missing required fields "
                        f"({','.join(missing_fields)}) in strict runtime."
                    ),
                )
            missing_weights = [feature for feature in feature_names if feature not in weights_raw]
            if missing_weights:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=(
                        f"Scenario live transform row '{name}.weights' is missing required features "
                        f"({','.join(missing_weights)}) in strict runtime."
                    ),
                )

        bias = _safe_float(row.get("bias"), float("nan"))
        low = _safe_float(row.get("min"), float("nan"))
        high = _safe_float(row.get("max"), float("nan"))
        if not (bias == bias):
            if strict_schema:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=f"Scenario live transform row '{name}.bias' is non-numeric in strict runtime.",
                )
            bias = 0.0
        if not (low == low):
            if strict_schema:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=f"Scenario live transform row '{name}.min' is non-numeric in strict runtime.",
                )
            low = default_min
        if not (high == high):
            if strict_schema:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=f"Scenario live transform row '{name}.max' is non-numeric in strict runtime.",
                )
            high = default_max
        if high < low:
            if strict_schema:
                raise ModelDataError(
                    reason_code="scenario_profile_invalid",
                    message=(
                        f"Scenario live transform row '{name}' has invalid bounds in strict runtime "
                        f"(min={low}, max={high})."
                    ),
                )
            high = low

        weights: dict[str, float] = {}
        for feature_name in feature_names:
            raw_weight = weights_raw.get(feature_name)
            parsed_weight = _safe_float(raw_weight, float("nan"))
            if not (parsed_weight == parsed_weight):
                if strict_schema:
                    raise ModelDataError(
                        reason_code="scenario_profile_invalid",
                        message=(
                            f"Scenario live transform row '{name}.weights.{feature_name}' is non-numeric "
                            "in strict runtime."
                        ),
                    )
                parsed_weight = 0.0
            weights[feature_name] = float(parsed_weight)
        if strict_schema and sum(max(0.0, value) for value in weights.values()) <= 0.0:
            raise ModelDataError(
                reason_code="scenario_profile_invalid",
                message=f"Scenario live transform row '{name}.weights' must include positive weight mass in strict runtime.",
            )
        return float(bias), weights, float(low), float(high)

    pressure_transforms = {
        "traffic_pressure": _pressure_transform(
            name="traffic_pressure",
            feature_names=("flow_index", "speed_inverse"),
            default_min=0.70,
            default_max=2.10,
        ),
        "incident_pressure": _pressure_transform(
            name="incident_pressure",
            feature_names=("delay_pressure", "severity_index"),
            default_min=0.75,
            default_max=2.40,
        ),
        "weather_pressure": _pressure_transform(
            name="weather_pressure",
            feature_names=("weather_severity_index",),
            default_min=0.70,
            default_max=2.00,
        ),
    }

    def _linear_pressure(name: str, values: dict[str, float]) -> float:
        bias, weights, low, high = pressure_transforms[name]
        score = float(bias)
        for key, value in values.items():
            score += float(weights.get(key, 0.0)) * float(value)
        return max(low, min(high, score))

    traffic_pressure = _linear_pressure(
        "traffic_pressure",
        {"flow_index": flow_index, "speed_inverse": speed_inverse},
    )
    incident_pressure = _linear_pressure(
        "incident_pressure",
        {"delay_pressure": delay_pressure, "severity_index": severity_index},
    )
    weather_pressure = _linear_pressure(
        "weather_pressure",
        {"weather_severity_index": weather_severity_index},
    )

    return ScenarioLiveContext(
        as_of_utc=as_of_utc,
        source_set=source_set,
        coverage=coverage,
        traffic_pressure=traffic_pressure,
        incident_pressure=incident_pressure,
        weather_pressure=weather_pressure,
        weather_bucket=weather_bucket,
        diagnostics={
            "source_label": source_label,
            "flow_index": flow_index,
            "speed_index": speed_index,
            "speed_inverse": speed_inverse,
            "delay_pressure": delay_pressure,
            "severity_index": severity_index,
            "weather_severity_index": weather_severity_index,
            "transform_version": str(transform.get("scenario_edge_scaling_version", "v3_live_transform")),
            "source_diagnostics": source_diagnostics,
        },
    )


@lru_cache(maxsize=512)
def load_live_scenario_context(
    *,
    corridor_bucket: str,
    road_mix_bucket: str,
    vehicle_class: str,
    day_kind: str,
    hour_slot_local: int | None,
    weather_bucket: str,
    centroid_lat: float | None,
    centroid_lon: float | None,
    road_hint: str | None = None,
    transform_params_json: str | None = None,
) -> ScenarioLiveContext:
    transform_params: dict[str, Any] | None = None
    if transform_params_json:
        try:
            loaded = json.loads(transform_params_json)
            if isinstance(loaded, dict):
                transform_params = loaded
        except Exception:
            transform_params = None
    strict = _strict_runtime_required()
    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    enforce_freshness = strict and not pytest_bypass_enabled
    live_failure: ModelDataError | None = None
    if settings.live_runtime_data_enabled:
        live_payload = live_scenario_context(
            {
                "corridor_bucket": corridor_bucket,
                "road_mix_bucket": road_mix_bucket,
                "vehicle_class": vehicle_class,
                "day_kind": day_kind,
                "hour_slot_local": int(max(0, min(23, int(_safe_float(hour_slot_local, 12.0))))),
                "weather_bucket": weather_bucket,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "road_hint": road_hint or "",
            }
        )
        if isinstance(live_payload, dict) and "_live_error" in live_payload:
            err = live_payload.get("_live_error", {})
            if isinstance(err, dict):
                live_failure = ModelDataError(
                    reason_code=normalize_reason_code(
                        str(err.get("reason_code", "scenario_profile_unavailable")),
                        default="scenario_profile_unavailable",
                    ),
                    message=str(err.get("message", "Live scenario source error")).strip() or "Live scenario source error",
                    details=(err.get("diagnostics") if isinstance(err.get("diagnostics"), dict) else None),
                )
        elif isinstance(live_payload, dict):
            try:
                parsed = _parse_live_scenario_context_payload(
                    live_payload,
                    source_label="live_runtime:scenario_context",
                    transform_params=transform_params,
                )
                _raise_if_strict_stale_minutes(
                    reason_code="scenario_profile_unavailable",
                    message="Live scenario context is stale for strict runtime policy.",
                    as_of_utc=_parse_iso_datetime(parsed.as_of_utc),
                    max_age_minutes=int(settings.live_scenario_max_age_minutes),
                    enforce=enforce_freshness,
                )
                return parsed
            except ModelDataError as exc:
                live_failure = exc

    if live_failure is not None:
        raise live_failure
    raise ModelDataError(
        reason_code="scenario_profile_unavailable",
        message="Live scenario context is unavailable for strict runtime.",
    )


def _parse_departure_profile_payload(
    payload: dict[str, Any],
    *,
    source: str,
) -> DepartureProfile | None:
    version = str(payload.get("version", "uk-v3-contextual"))
    as_of_utc = str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None
    refreshed_at_utc = (
        str(payload.get("refreshed_at_utc", payload.get("generated_at_utc", ""))).strip() or None
    )
    weekday_raw = payload.get("weekday")
    weekend_raw = payload.get("weekend")
    holiday_raw = payload.get("holiday")
    if (
        isinstance(weekday_raw, list)
        and isinstance(weekend_raw, list)
        and isinstance(holiday_raw, list)
        and len(weekday_raw) == 1440
        and len(weekend_raw) == 1440
        and len(holiday_raw) == 1440
    ):
        return DepartureProfile(
            weekday=tuple(float(v) for v in weekday_raw),
            weekend=tuple(float(v) for v in weekend_raw),
            holiday=tuple(float(v) for v in holiday_raw),
            source=source,
            version=version,
            as_of_utc=as_of_utc,
            refreshed_at_utc=refreshed_at_utc,
            contextual={
                "uk_default": {
                    "mixed": {
                        "weekday": tuple(float(v) for v in weekday_raw),
                        "weekend": tuple(float(v) for v in weekend_raw),
                        "holiday": tuple(float(v) for v in holiday_raw),
                    }
                }
            },
        )

    profiles_raw = payload.get("profiles", {})
    contextual: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {}
    if isinstance(profiles_raw, dict):
        for region_key, road_map_raw in profiles_raw.items():
            if not isinstance(road_map_raw, dict):
                continue
            road_map_out: dict[str, dict[str, tuple[float, ...]]] = {}
            for road_key, day_map_raw in road_map_raw.items():
                if not isinstance(day_map_raw, dict):
                    continue
                day_map_out: dict[str, tuple[float, ...]] = {}
                for day_key, values_raw in day_map_raw.items():
                    if not isinstance(values_raw, list) or len(values_raw) != 1440:
                        continue
                    day_map_out[str(day_key)] = tuple(float(v) for v in values_raw)
                if day_map_out:
                    road_map_out[str(road_key)] = day_map_out
            if road_map_out:
                contextual[str(region_key)] = road_map_out
    if not contextual:
        return None

    envelopes_raw = payload.get("envelopes", {})
    contextual_envelopes: dict[str, dict[str, dict[str, dict[str, tuple[float, ...]]]]] = {}
    if isinstance(envelopes_raw, dict):
        for region_key, road_map_raw in envelopes_raw.items():
            if not isinstance(road_map_raw, dict):
                continue
            road_out: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {}
            for road_key, day_map_raw in road_map_raw.items():
                if not isinstance(day_map_raw, dict):
                    continue
                day_out: dict[str, dict[str, tuple[float, ...]]] = {}
                for day_key, env_raw in day_map_raw.items():
                    if not isinstance(env_raw, dict):
                        continue
                    low_raw = env_raw.get("low")
                    high_raw = env_raw.get("high")
                    if (
                        isinstance(low_raw, list)
                        and isinstance(high_raw, list)
                        and len(low_raw) == 1440
                        and len(high_raw) == 1440
                    ):
                        day_out[str(day_key)] = {
                            "low": tuple(float(v) for v in low_raw),
                            "high": tuple(float(v) for v in high_raw),
                        }
                if day_out:
                    road_out[str(road_key)] = day_out
            if road_out:
                contextual_envelopes[str(region_key)] = road_out

    default_road = contextual.get("uk_default", {}).get("mixed", {})
    weekday = default_road.get("weekday")
    weekend = default_road.get("weekend")
    holiday = default_road.get("holiday")
    if not weekday or not weekend or not holiday:
        return None
    return DepartureProfile(
        weekday=weekday,
        weekend=weekend,
        holiday=holiday,
        source=source,
        version=version,
        as_of_utc=as_of_utc,
        refreshed_at_utc=refreshed_at_utc,
        contextual=contextual,
        contextual_envelopes=contextual_envelopes or None,
    )


@dataclass(frozen=True)
class DepartureProfile:
    weekday: tuple[float, ...]
    weekend: tuple[float, ...]
    holiday: tuple[float, ...]
    source: str
    version: str = "uk-v3-contextual"
    as_of_utc: str | None = None
    refreshed_at_utc: str | None = None
    contextual: dict[str, dict[str, dict[str, tuple[float, ...]]]] | None = None
    contextual_envelopes: (
        dict[str, dict[str, dict[str, dict[str, tuple[float, ...]]]]]
        | None
    ) = None

    def resolve(
        self,
        *,
        day_kind: str,
        region: str,
        road_bucket: str,
    ) -> tuple[float, ...]:
        if not self.contextual:
            return {
                "weekday": self.weekday,
                "weekend": self.weekend,
                "holiday": self.holiday,
            }.get(day_kind, self.weekday)
        day = day_kind if day_kind in ("weekday", "weekend", "holiday") else "weekday"
        region_map = self.contextual.get(region) or self.contextual.get("uk_default")
        if not region_map:
            return {
                "weekday": self.weekday,
                "weekend": self.weekend,
                "holiday": self.holiday,
            }.get(day_kind, self.weekday)
        road_map = (
            region_map.get(road_bucket)
            or region_map.get("mixed")
            or next(iter(region_map.values()))
        )
        series = (
            road_map.get(day)
            or road_map.get("weekday")
            or self.weekday
        )
        return series

    def resolve_envelope(
        self,
        *,
        day_kind: str,
        region: str,
        road_bucket: str,
    ) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        if not self.contextual_envelopes:
            return None
        day = day_kind if day_kind in ("weekday", "weekend", "holiday") else "weekday"
        region_map = self.contextual_envelopes.get(region) or self.contextual_envelopes.get("uk_default")
        if not region_map:
            return None
        road_map = region_map.get(road_bucket) or region_map.get("mixed") or next(iter(region_map.values()))
        day_env = road_map.get(day) or road_map.get("weekday")
        if not isinstance(day_env, dict):
            return None
        low = day_env.get("low")
        high = day_env.get("high")
        if (
            isinstance(low, tuple)
            and isinstance(high, tuple)
            and len(low) == 1440
            and len(high) == 1440
        ):
            return low, high
        return None


@dataclass(frozen=True)
class RiskNormalizationReference:
    duration_s_per_km: float
    monetary_gbp_per_km: float
    emissions_kg_per_km: float
    source: str
    version: str = "unknown"
    as_of_utc: str | None = None
    corridor_bucket: str = "uk_default"
    day_kind: str = "weekday"
    local_time_slot: str = "h12"


def _canonical_vehicle_bucket(
    vehicle_type: str | None,
    *,
    vehicle_bucket: str | None = None,
    bucket_attr: str = "risk_bucket",
) -> str:
    explicit_bucket = str(vehicle_bucket or "").strip().lower()
    if explicit_bucket:
        return explicit_bucket
    key = (vehicle_type or "").strip().lower()
    if not key:
        return "default"
    try:
        from .vehicles import resolve_vehicle_profile  # local import to avoid heavy import cycles

        resolved = resolve_vehicle_profile(key)
        resolved_bucket = str(getattr(resolved, bucket_attr, "")).strip().lower()
        if resolved_bucket:
            return resolved_bucket
    except Exception:
        pass
    return key


def _canonical_corridor_bucket(corridor_bucket: str | None) -> str:
    key = (corridor_bucket or "").strip().lower()
    if not key:
        return "uk_default"
    aliases = {
        "london": "london_southeast",
        "south_east": "london_southeast",
        "south-east": "london_southeast",
        "north": "north_england",
        "midlands_central": "midlands",
        "wales": "wales_west",
    }
    return aliases.get(key, key)


def _canonical_day_kind(day_kind: str | None) -> str:
    key = (day_kind or "").strip().lower()
    if key in {"weekday", "weekend", "holiday"}:
        return key
    return "weekday"


def _canonical_local_time_slot(local_time_slot: str | None) -> str:
    key = (local_time_slot or "").strip().lower()
    if key.startswith("h") and len(key) == 3:
        try:
            hour = int(key[1:])
        except ValueError:
            hour = 12
        return f"h{max(0, min(23, hour)):02d}"
    try:
        hour = int(key)
    except ValueError:
        return "h12"
    return f"h{max(0, min(23, hour)):02d}"


@lru_cache(maxsize=128)
def load_risk_normalization_reference(
    vehicle_type: str | None = None,
    vehicle_bucket: str | None = None,
    corridor_bucket: str | None = None,
    day_kind: str | None = None,
    local_time_slot: str | None = None,
) -> RiskNormalizationReference:
    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    path_candidates = [
        _model_asset_root() / "risk_normalization_refs_uk.json",
        _assets_root() / "risk_normalization_refs_uk.json",
    ]
    payload: dict[str, Any] | None = None
    source = ""
    version = ""
    as_of_utc: str | None = None
    selected_path: Path | None = None
    for path in path_candidates:
        if not path.exists():
            continue
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            payload = parsed
            source = str(path)
            version = str(parsed.get("version", parsed.get("calibration_version", "unknown")))
            as_of_utc = str(parsed.get("as_of_utc", parsed.get("as_of", ""))).strip() or None
            selected_path = path
            break
    if payload is None:
        raise ModelDataError(
            reason_code="risk_normalization_unavailable",
            message="Risk normalization reference asset is required in strict runtime.",
        )
    _raise_if_strict_stale(
        reason_code="risk_normalization_unavailable",
        message="Risk normalization reference asset is stale for strict runtime policy.",
        as_of_utc=_infer_as_of_from_payload(payload) or (_infer_as_of_from_path(selected_path) if selected_path else None),
        max_age_days=int(settings.live_stochastic_max_age_days),
        enforce=not pytest_bypass_enabled,
    )

    vehicle_bucket = _canonical_vehicle_bucket(
        vehicle_type,
        vehicle_bucket=vehicle_bucket,
        bucket_attr="risk_bucket",
    )
    corridor_key = _canonical_corridor_bucket(corridor_bucket)
    day_key = _canonical_day_kind(day_kind)
    slot_key = _canonical_local_time_slot(local_time_slot)

    selected_ref: dict[str, Any] | None = None
    corridor_vehicle_refs = payload.get("corridor_vehicle_refs", {})
    if isinstance(corridor_vehicle_refs, dict):
        c_item = corridor_vehicle_refs.get(corridor_key)
        if isinstance(c_item, dict):
            by_vehicle = c_item.get(vehicle_bucket)
            if isinstance(by_vehicle, dict):
                by_day = by_vehicle.get(day_key)
                if isinstance(by_day, dict):
                    if isinstance(by_day.get(slot_key), dict):
                        selected_ref = by_day.get(slot_key)
                    elif {"duration_s_per_km", "monetary_gbp_per_km", "emissions_kg_per_km"}.issubset(by_day.keys()):
                        selected_ref = by_day

    if not isinstance(selected_ref, dict):
        if pytest_bypass_enabled and isinstance(corridor_vehicle_refs, dict):
            fallback_corridors = [corridor_key, "uk_default", "*"]
            fallback_days = [day_key, "weekday", "weekend", "holiday"]
            fallback_slots = [slot_key, "h12", "*"]
            for fallback_corridor in fallback_corridors:
                c_item = corridor_vehicle_refs.get(fallback_corridor)
                if not isinstance(c_item, dict):
                    continue
                by_vehicle = c_item.get(vehicle_bucket)
                if not isinstance(by_vehicle, dict):
                    continue
                for fallback_day in fallback_days:
                    by_day = by_vehicle.get(fallback_day)
                    if not isinstance(by_day, dict):
                        continue
                    for fallback_slot in fallback_slots:
                        row = by_day.get(fallback_slot)
                        if isinstance(row, dict):
                            selected_ref = row
                            corridor_key = fallback_corridor
                            day_key = fallback_day
                            slot_key = fallback_slot
                            break
                    if isinstance(selected_ref, dict):
                        break
                if isinstance(selected_ref, dict):
                    break
    if not isinstance(selected_ref, dict):
        raise ModelDataError(
            reason_code="risk_normalization_unavailable",
            message=(
                "No risk normalization entry matched strict route context "
                f"(vehicle={vehicle_bucket}, corridor={corridor_key}, day={day_key}, slot={slot_key})."
            ),
        )

    defaults = {
        "duration_s_per_km": max(
            1.0,
            _safe_float(selected_ref.get("duration_s_per_km"), 0.0),
        ),
        "monetary_gbp_per_km": max(
            0.05,
            _safe_float(selected_ref.get("monetary_gbp_per_km"), 0.0),
        ),
        "emissions_kg_per_km": max(
            0.01,
            _safe_float(selected_ref.get("emissions_kg_per_km"), 0.0),
        ),
    }

    return RiskNormalizationReference(
        duration_s_per_km=defaults["duration_s_per_km"],
        monetary_gbp_per_km=defaults["monetary_gbp_per_km"],
        emissions_kg_per_km=defaults["emissions_kg_per_km"],
        source=source,
        version=version,
        as_of_utc=as_of_utc,
        corridor_bucket=corridor_key,
        day_kind=day_key,
        local_time_slot=slot_key,
    )


def _interpolate_sparse_profile(
    points: list[tuple[int, float]],
    *,
    default_value: float,
) -> tuple[float, ...]:
    if not points:
        return tuple(default_value for _ in range(1440))

    points = sorted((max(0, min(1439, int(m))), float(v)) for m, v in points)
    dense: list[float] = [default_value for _ in range(1440)]

    # Fill leading region.
    first_minute, first_value = points[0]
    for minute in range(0, first_minute + 1):
        dense[minute] = first_value

    # Interpolate between knots.
    for idx in range(1, len(points)):
        prev_minute, prev_value = points[idx - 1]
        next_minute, next_value = points[idx]
        span = max(1, next_minute - prev_minute)
        for minute in range(prev_minute, next_minute + 1):
            t = (minute - prev_minute) / span
            dense[minute] = prev_value + ((next_value - prev_value) * t)

    # Fill trailing region.
    last_minute, last_value = points[-1]
    for minute in range(last_minute, 1440):
        dense[minute] = last_value

    return tuple(dense)


@lru_cache(maxsize=1)
def load_departure_profile() -> DepartureProfile:
    strict = _strict_runtime_required()
    strict_empirical_required = _strict_empirical_departure_required()
    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    live_url = str(settings.live_departure_profile_url or "").strip()
    require_live_url = bool(settings.live_departure_require_url_in_strict) if strict else False
    rejected_synthetic = False
    live_failure: ModelDataError | None = None
    if strict and require_live_url and not live_url and not pytest_bypass_enabled:
        raise ModelDataError(
            reason_code="departure_profile_unavailable",
            message="LIVE_DEPARTURE_PROFILE_URL is required in strict runtime.",
        )
    if settings.live_runtime_data_enabled and live_url:
        live_payload = live_departure_profiles()
        if isinstance(live_payload, dict):
            parsed = _parse_departure_profile_payload(
                live_payload,
                source="live_runtime:departure_profiles",
            )
            if parsed is not None:
                live_as_of = _infer_as_of_from_payload(live_payload)
                if (
                    strict_empirical_required
                    and not settings.departure_allow_synthetic_profiles
                    and _payload_is_synthetic(
                        live_payload, version=parsed.version
                    )
                ):
                    rejected_synthetic = True
                elif _is_fresh(live_as_of, max_age_days=int(settings.live_departure_max_age_days)):
                    return parsed
                else:
                    live_failure = ModelDataError(
                        reason_code="departure_profile_unavailable",
                        message="Live departure profile payload is stale for strict runtime policy.",
                        details={
                            "as_of_utc": live_as_of.isoformat() if live_as_of is not None else None,
                            "max_age_days": int(settings.live_departure_max_age_days),
                        },
                    )
            else:
                live_failure = ModelDataError(
                    reason_code="departure_profile_unavailable",
                    message="Live departure profile payload could not be parsed.",
                )
        elif strict and require_live_url:
            live_failure = ModelDataError(
                reason_code="departure_profile_unavailable",
                message="Live departure profile payload is unavailable in strict runtime.",
            )

    if rejected_synthetic:
        raise ModelDataError(
            reason_code="departure_profile_unavailable",
            message="Only synthetic departure profiles were available; empirical profile assets are required.",
        )
    if live_failure is not None:
        raise live_failure
    raise ModelDataError(
        reason_code="departure_profile_unavailable",
        message="Live departure profile is unavailable for strict runtime.",
    )


@lru_cache(maxsize=1)
def load_uk_bank_holidays() -> frozenset[str]:
    payload = live_bank_holidays() if settings.live_runtime_data_enabled else None
    if not isinstance(payload, dict):
        raise ModelDataError(
            reason_code="holiday_data_unavailable",
            message="Live bank holiday dataset is unavailable in strict runtime.",
        )
    _raise_if_strict_stale(
        reason_code="holiday_data_unavailable",
        message="Live bank holiday payload is stale for strict runtime policy.",
        as_of_utc=_infer_as_of_from_payload(payload),
        max_age_days=int(settings.live_departure_max_age_days),
        enforce=True,
    )

    if not payload:
        raise ModelDataError(
            reason_code="holiday_data_unavailable",
            message="No valid bank holiday dataset available.",
        )
    dates: set[str] = set()
    for region_value in payload.values():
        if not isinstance(region_value, dict):
            continue
        events = region_value.get("events", [])
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            date_text = str(event.get("date", "")).strip()
            if date_text:
                dates.add(date_text)
    return frozenset(dates)


@dataclass(frozen=True)
class TerrainManifestInfo:
    version: str
    source: str
    tile_count: int
    bounds: dict[str, float]


@lru_cache(maxsize=1)
def load_terrain_manifest_info() -> TerrainManifestInfo:
    path_candidates = [
        _model_asset_root() / "terrain" / "terrain_manifest.json",
        _model_asset_root() / "terrain_manifest.json",
    ]
    for path in path_candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        raw_bounds = payload.get("bounds", {})
        bounds = {
            "lat_min": _safe_float(raw_bounds.get("lat_min"), 0.0),
            "lat_max": _safe_float(raw_bounds.get("lat_max"), 0.0),
            "lon_min": _safe_float(raw_bounds.get("lon_min"), 0.0),
            "lon_max": _safe_float(raw_bounds.get("lon_max"), 0.0),
        }
        raw_tiles = payload.get("tiles", [])
        tile_count = len(raw_tiles) if isinstance(raw_tiles, list) else 0
        return TerrainManifestInfo(
            version=str(payload.get("version", "unknown")),
            source=str(path),
            tile_count=tile_count,
            bounds=bounds,
        )
    return TerrainManifestInfo(
        version="missing",
        source="none",
        tile_count=0,
        bounds={"lat_min": 0.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 0.0},
    )


@dataclass(frozen=True)
class TollTariffRule:
    rule_id: str
    operator: str
    crossing_id: str
    road_class: str
    direction: str
    start_minute: int
    end_minute: int
    crossing_fee_gbp: float
    distance_fee_gbp_per_km: float
    vehicle_classes: tuple[str, ...]
    axle_classes: tuple[str, ...]
    payment_classes: tuple[str, ...]
    exemptions: tuple[str, ...]


@dataclass(frozen=True)
class TollTariffTable:
    default_crossing_fee_gbp: float
    default_distance_fee_gbp_per_km: float
    rules: tuple[TollTariffRule, ...]
    source: str


def _parse_tariff_payload(payload: dict[str, Any], *, source: str) -> TollTariffTable:
    defaults = payload.get("defaults", {})
    default_crossing = _safe_float(defaults.get("crossing_fee_gbp"), 0.0)
    default_distance = _safe_float(defaults.get("distance_fee_gbp_per_km"), 0.0)

    rules: list[TollTariffRule] = []
    raw_rules = payload.get("rules", [])
    if isinstance(raw_rules, list):
        for idx, item in enumerate(raw_rules):
            if not isinstance(item, dict):
                continue
            vehicle_classes_raw = item.get("vehicle_classes", ["default"])
            if isinstance(vehicle_classes_raw, list):
                vehicle_classes = tuple(str(v).strip().lower() for v in vehicle_classes_raw if str(v).strip())
            else:
                vehicle_classes = ("default",)
            axle_classes_raw = item.get("axle_classes", ["default"])
            payment_classes_raw = item.get("payment_classes", ["default"])
            exemptions_raw = item.get("exemptions", [])
            axle_classes = (
                tuple(str(v).strip().lower() for v in axle_classes_raw if str(v).strip())
                if isinstance(axle_classes_raw, list)
                else ("default",)
            )
            payment_classes = (
                tuple(str(v).strip().lower() for v in payment_classes_raw if str(v).strip())
                if isinstance(payment_classes_raw, list)
                else ("default",)
            )
            exemptions = (
                tuple(str(v).strip().lower() for v in exemptions_raw if str(v).strip())
                if isinstance(exemptions_raw, list)
                else ()
            )
            rules.append(
                TollTariffRule(
                    rule_id=str(item.get("id", f"rule_{idx}")),
                    operator=str(item.get("operator", "default")).strip().lower(),
                    crossing_id=str(item.get("crossing_id", "*")).strip().lower(),
                    road_class=str(item.get("road_class", "default")).strip().lower(),
                    direction=str(item.get("direction", "both")).strip().lower(),
                    start_minute=max(0, min(1439, int(_safe_float(item.get("start_minute"), 0.0)))),
                    end_minute=max(0, min(1439, int(_safe_float(item.get("end_minute"), 1439.0)))),
                    crossing_fee_gbp=max(0.0, _safe_float(item.get("crossing_fee_gbp"), default_crossing)),
                    distance_fee_gbp_per_km=max(
                        0.0,
                        _safe_float(item.get("distance_fee_gbp_per_km"), default_distance),
                    ),
                    vehicle_classes=vehicle_classes or ("default",),
                    axle_classes=axle_classes or ("default",),
                    payment_classes=payment_classes or ("default",),
                    exemptions=exemptions,
                )
            )

    if _strict_runtime_required():
        strict_rules: list[TollTariffRule] = []
        for rule in rules:
            if rule.operator in {"", "default", "*"}:
                continue
            if rule.crossing_id in {"", "*", "default"}:
                continue
            if rule.road_class in {"", "default", "*"}:
                continue
            if rule.direction in {"", "default", "*"}:
                continue
            if (
                not rule.vehicle_classes
                or any(item in {"", "default", "*"} for item in rule.vehicle_classes)
            ):
                continue
            if (
                not rule.axle_classes
                or any(item in {"", "default", "*"} for item in rule.axle_classes)
            ):
                continue
            if (
                not rule.payment_classes
                or any(item in {"", "default", "*"} for item in rule.payment_classes)
            ):
                continue
            strict_rules.append(rule)
        rules = strict_rules

    return TollTariffTable(
        default_crossing_fee_gbp=max(0.0, default_crossing),
        default_distance_fee_gbp_per_km=max(0.0, default_distance),
        rules=tuple(rules),
        source=source,
    )


@lru_cache(maxsize=1)
def load_toll_tariffs() -> TollTariffTable:
    strict = _strict_runtime_required()
    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    live_url = str(settings.live_toll_tariffs_url or "").strip()
    require_live_url = bool(settings.live_toll_tariffs_require_url_in_strict) if strict else False
    live_failure: ModelDataError | None = None
    if strict and require_live_url and not live_url and not pytest_bypass_enabled:
        raise ModelDataError(
            reason_code="toll_tariff_unavailable",
            message="LIVE_TOLL_TARIFFS_URL is required in strict runtime.",
        )
    if settings.live_runtime_data_enabled and live_url:
        live_payload = live_toll_tariffs()
        if isinstance(live_payload, dict):
            live_table = _parse_tariff_payload(live_payload, source="live_runtime:toll_tariffs")
            live_as_of = _infer_as_of_from_payload(live_payload)
            live_fresh = _is_fresh(live_as_of, max_age_days=int(settings.live_toll_tariffs_max_age_days))
            if live_table.rules and live_fresh:
                return live_table
            if strict:
                live_failure = ModelDataError(
                    reason_code="toll_tariff_unavailable",
                    message="Live toll tariff payload is unavailable/stale for strict runtime policy.",
                    details={
                        "as_of_utc": live_as_of.isoformat() if live_as_of is not None else None,
                        "max_age_days": int(settings.live_toll_tariffs_max_age_days),
                    },
                )
    if live_failure is not None:
        raise live_failure
    raise ModelDataError(
        reason_code="toll_tariff_unavailable",
        message="Live toll tariffs are unavailable in strict runtime.",
    )


@dataclass(frozen=True)
class TollSegmentSeed:
    segment_id: str
    name: str
    operator: str
    road_class: str
    crossing_id: str
    direction: str
    crossing_fee_gbp: float
    distance_fee_gbp_per_km: float
    coordinates: tuple[tuple[float, float], ...]  # (lat, lon)


@lru_cache(maxsize=1)
def load_toll_segments_seed() -> tuple[TollSegmentSeed, ...]:
    strict = _strict_runtime_required()
    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    live_url = str(settings.live_toll_topology_url or "").strip()
    require_live_url = bool(settings.live_toll_topology_require_url_in_strict) if strict else False
    payload: Any = None
    source_label = "unknown"
    live_failure: ModelDataError | None = None
    if strict and require_live_url and not live_url and not pytest_bypass_enabled:
        raise ModelDataError(
            reason_code="toll_topology_unavailable",
            message="LIVE_TOLL_TOPOLOGY_URL is required in strict runtime.",
        )
    if settings.live_runtime_data_enabled and live_url:
        live_payload = live_toll_topology()
        if isinstance(live_payload, dict):
            live_as_of = _infer_as_of_from_payload(live_payload)
            live_fresh = _is_fresh(live_as_of, max_age_days=int(settings.live_toll_topology_max_age_days))
            if live_fresh:
                payload = live_payload
                source_label = "live_runtime:toll_topology"
            elif strict:
                live_failure = ModelDataError(
                    reason_code="toll_topology_unavailable",
                    message="Live toll topology payload is stale for strict runtime policy.",
                    details={
                        "as_of_utc": live_as_of.isoformat() if live_as_of is not None else None,
                        "max_age_days": int(settings.live_toll_topology_max_age_days),
                    },
                )
        elif strict and require_live_url:
            live_failure = ModelDataError(
                reason_code="toll_topology_unavailable",
                message="Live toll topology payload is unavailable in strict runtime.",
            )

    if not isinstance(payload, dict):
        if live_failure is not None:
            raise live_failure
        raise ModelDataError(
            reason_code="toll_topology_unavailable",
            message="Live toll topology payload is unavailable in strict runtime.",
        )

    features = payload.get("features", [])
    if not isinstance(features, list):
        return ()

    seeds: list[TollSegmentSeed] = []
    for idx, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        if not isinstance(props, dict) or not isinstance(geom, dict):
            continue
        geom_type = str(geom.get("type")).lower()
        raw_coords = geom.get("coordinates", [])
        lat_lon_points: list[tuple[float, float]] = []
        if geom_type == "linestring" and isinstance(raw_coords, list):
            for point in raw_coords:
                if not isinstance(point, list | tuple) or len(point) < 2:
                    continue
                lon = _safe_float(point[0], 0.0)
                lat = _safe_float(point[1], 0.0)
                lat_lon_points.append((lat, lon))
        elif geom_type == "multilinestring" and isinstance(raw_coords, list):
            for line in raw_coords:
                if not isinstance(line, list):
                    continue
                for point in line:
                    if not isinstance(point, list | tuple) or len(point) < 2:
                        continue
                    lon = _safe_float(point[0], 0.0)
                    lat = _safe_float(point[1], 0.0)
                    lat_lon_points.append((lat, lon))
        elif geom_type == "point" and isinstance(raw_coords, list) and len(raw_coords) >= 2:
            lon = _safe_float(raw_coords[0], 0.0)
            lat = _safe_float(raw_coords[1], 0.0)
            # Convert point assets (booths/gantries) into tiny directional stubs so
            # overlap scoring can still evaluate route proximity.
            lat_lon_points = [(lat, lon), (lat + 0.00005, lon + 0.00005)]
        if len(lat_lon_points) < 2:
            continue
        seeds.append(
            TollSegmentSeed(
                segment_id=str(props.get("id", f"segment_{idx}")),
                name=str(props.get("name", "")),
                operator=str(props.get("operator", "default")).strip().lower(),
                road_class=str(props.get("road_class", "default")).strip().lower(),
                crossing_id=str(props.get("crossing_id", props.get("id", f"segment_{idx}"))).strip().lower(),
                direction=str(props.get("direction", "both")).strip().lower(),
                crossing_fee_gbp=max(0.0, _safe_float(props.get("crossing_fee_gbp"), 0.0)),
                distance_fee_gbp_per_km=max(
                    0.0,
                    _safe_float(props.get("distance_fee_gbp_per_km"), 0.0),
                ),
                coordinates=tuple(lat_lon_points),
            )
        )
    if strict and not seeds:
        raise ModelDataError(
            reason_code="toll_topology_unavailable",
            message=f"No valid toll segments parsed from source '{source_label}'.",
        )
    return tuple(seeds)


@dataclass(frozen=True)
class TollConfidenceBin:
    minimum: float
    maximum: float
    calibrated: float


@dataclass(frozen=True)
class TollConfidenceCalibration:
    intercept: float
    class_signal_coef: float
    seed_signal_coef: float
    segment_signal_coef: float
    bonus_both: float
    bonus_class: float
    bins: tuple[TollConfidenceBin, ...]
    source: str
    version: str
    as_of_utc: str | None


@lru_cache(maxsize=1)
def load_toll_confidence_calibration() -> TollConfidenceCalibration:
    candidates = [
        _model_asset_root() / "toll_confidence_calibration_uk.json",
        _assets_root() / "toll_confidence_calibration_uk.json",
    ]
    payload: dict[str, Any] | None = None
    source = ""
    fallback_path: Path | None = None
    for path in candidates:
        if not path.exists():
            continue
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            continue
        payload = parsed
        source = str(path)
        fallback_path = path
        break
    if payload is None:
        raise ModelDataError(
            reason_code="toll_topology_unavailable",
            message="Toll confidence calibration asset is required in strict runtime.",
        )

    if fallback_path is not None:
        _raise_if_strict_stale(
            reason_code="toll_topology_unavailable",
            message="Toll confidence calibration asset is stale for strict live-data policy.",
            as_of_utc=_infer_as_of_from_payload(payload) or _infer_as_of_from_path(fallback_path),
            max_age_days=int(settings.live_toll_topology_max_age_days),
            enforce=True,
        )

    model = payload.get("logit_model", {})
    if not isinstance(model, dict):
        model = {}
    raw_bins = payload.get("reliability_bins", [])
    if _strict_runtime_required():
        required_model_keys = (
            "intercept",
            "class_signal",
            "seed_signal",
            "segment_signal",
            "source_bonus_both",
            "source_bonus_class",
        )
        missing_keys = [key for key in required_model_keys if key not in model]
        if missing_keys:
            raise ModelDataError(
                reason_code="toll_topology_unavailable",
                message=(
                    "Toll confidence calibration asset is missing required logit coefficients: "
                    + ", ".join(missing_keys)
                ),
            )
        if not isinstance(raw_bins, list) or not raw_bins:
            raise ModelDataError(
                reason_code="toll_topology_unavailable",
                message="Toll confidence calibration asset is missing reliability bins in strict runtime.",
            )
    bins: list[TollConfidenceBin] = []
    if isinstance(raw_bins, list):
        for row in raw_bins:
            if not isinstance(row, dict):
                continue
            lo = max(0.0, min(1.0, _safe_float(row.get("min"), 0.0)))
            hi = max(lo, min(1.0, _safe_float(row.get("max"), 1.0)))
            calibrated = max(0.0, min(1.0, _safe_float(row.get("calibrated"), (lo + hi) * 0.5)))
            bins.append(TollConfidenceBin(minimum=lo, maximum=hi, calibrated=calibrated))
    if not bins:
        if _strict_runtime_required():
            raise ModelDataError(
                reason_code="toll_topology_unavailable",
                message="No valid toll confidence reliability bins were parsed for strict runtime.",
            )
        bins = [
            TollConfidenceBin(minimum=0.0, maximum=0.5, calibrated=0.35),
            TollConfidenceBin(minimum=0.5, maximum=1.0, calibrated=0.75),
        ]

    if _strict_runtime_required():
        try:
            intercept = float(model["intercept"])
            class_signal = float(model["class_signal"])
            seed_signal = float(model["seed_signal"])
            segment_signal = float(model["segment_signal"])
            source_bonus_both = float(model["source_bonus_both"])
            source_bonus_class = float(model["source_bonus_class"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ModelDataError(
                reason_code="toll_topology_unavailable",
                message="Toll confidence calibration coefficients must be numeric in strict runtime.",
            ) from exc
    else:
        intercept = _safe_float(model.get("intercept"), -1.05)
        class_signal = _safe_float(model.get("class_signal"), 1.85)
        seed_signal = _safe_float(model.get("seed_signal"), 2.10)
        segment_signal = _safe_float(model.get("segment_signal"), 0.55)
        source_bonus_both = _safe_float(model.get("source_bonus_both"), 0.06)
        source_bonus_class = _safe_float(model.get("source_bonus_class"), 0.02)

    return TollConfidenceCalibration(
        intercept=intercept,
        class_signal_coef=class_signal,
        seed_signal_coef=seed_signal,
        segment_signal_coef=segment_signal,
        bonus_both=source_bonus_both,
        bonus_class=source_bonus_class,
        bins=tuple(bins),
        source=source,
        version=str(payload.get("version", "unknown")),
        as_of_utc=str(payload.get("as_of_utc", "")).strip() or None,
    )


@dataclass(frozen=True)
class FuelPriceSnapshot:
    prices_gbp_per_l: dict[str, float]
    grid_price_gbp_per_kwh: float
    regional_multipliers: dict[str, float]
    as_of: str | None
    source: str
    signature: str | None = None
    live_diagnostics: dict[str, Any] | None = None


def _fuel_signature_material(payload: dict[str, Any]) -> str:
    material = {
        "as_of_utc": str(payload.get("as_of_utc", payload.get("as_of", ""))).strip(),
        "prices_gbp_per_l": payload.get("prices_gbp_per_l", {}),
        "grid_price_gbp_per_kwh": payload.get("grid_price_gbp_per_kwh"),
        "regional_multipliers": payload.get("regional_multipliers", {}),
        "history": payload.get("history", []),
        "source": str(payload.get("source", "")).strip(),
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fuel_signature(payload: dict[str, Any]) -> str:
    data = _fuel_signature_material(payload).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _validate_fuel_signature(
    *,
    payload: dict[str, Any],
    require_signature: bool,
    reason_code: str,
    message_prefix: str,
) -> str | None:
    signature = str(payload.get("signature", "")).strip() or None
    if not require_signature:
        return signature
    if not signature:
        raise ModelDataError(
            reason_code=reason_code,
            message=f"{message_prefix} signature is required in strict runtime.",
        )
    expected = _fuel_signature(payload)
    if signature.lower() != expected.lower():
        raise ModelDataError(
            reason_code=reason_code,
            message=f"{message_prefix} signature verification failed in strict runtime.",
            details={"expected_signature": expected},
        )
    return signature


def _select_fuel_history_row(
    *,
    payload: dict[str, Any],
    as_of_utc: datetime | None,
) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    history = payload.get("history", [])
    if isinstance(history, list):
        for row in history:
            if isinstance(row, dict):
                candidate_rows.append(row)
    if isinstance(payload.get("prices_gbp_per_l"), dict):
        candidate_rows.append(payload)
    if not candidate_rows:
        return payload

    selected_row: dict[str, Any] | None = None
    if as_of_utc is not None:
        target = as_of_utc if as_of_utc.tzinfo is not None else as_of_utc.replace(tzinfo=UTC)
        target = target.astimezone(UTC)
        dated_rows: list[tuple[datetime, dict[str, Any]]] = []
        for row in candidate_rows:
            row_dt = _parse_iso_datetime(str(row.get("as_of_utc", row.get("as_of", ""))).strip())
            if row_dt is not None:
                dated_rows.append((row_dt, row))
        dated_rows.sort(key=lambda item: item[0])
        for dt, row in dated_rows:
            if dt <= target:
                selected_row = row
        if selected_row is None and dated_rows:
            selected_row = dated_rows[-1][1]
    if selected_row is None:
        selected_row = candidate_rows[-1]
    return selected_row


def _coerce_fuel_snapshot(
    *,
    payload: dict[str, Any],
    source_label: str,
    as_of_utc: datetime | None,
    require_signature: bool,
) -> FuelPriceSnapshot:
    signature = _validate_fuel_signature(
        payload=payload,
        require_signature=require_signature,
        reason_code="fuel_price_source_unavailable",
        message_prefix="Fuel price payload",
    )
    if _strict_runtime_required():
        source_text = str(payload.get("source", source_label)).strip().lower()
        basis_text = str(payload.get("calibration_basis", "")).strip().lower()
        prohibited = ("synthetic", "heuristic", "legacy", "interpolated", "simulated", "wobble")
        if any(token in source_text for token in prohibited) or any(token in basis_text for token in prohibited):
            raise ModelDataError(
                reason_code="fuel_price_source_unavailable",
                message="Fuel price payload provenance is not acceptable for strict runtime policy.",
                details={
                    "source": payload.get("source", source_label),
                    "calibration_basis": payload.get("calibration_basis"),
                },
            )
    selected_row = _select_fuel_history_row(payload=payload, as_of_utc=as_of_utc)

    prices_raw = selected_row.get("prices_gbp_per_l", {})
    prices: dict[str, float] = {}
    if isinstance(prices_raw, dict):
        for key, value in prices_raw.items():
            name = str(key).strip().lower()
            if not name:
                continue
            prices[name] = max(0.0, _safe_float(value, 0.0))
    if not prices:
        raise ModelDataError(
            reason_code="fuel_price_source_unavailable",
            message="Fuel price payload has no valid prices_gbp_per_l entries.",
        )

    grid_price = _safe_float(selected_row.get("grid_price_gbp_per_kwh"), -1.0)
    if grid_price < 0.0:
        raise ModelDataError(
            reason_code="fuel_price_source_unavailable",
            message="Fuel price payload is missing grid_price_gbp_per_kwh.",
        )

    regional_raw = selected_row.get("regional_multipliers", payload.get("regional_multipliers", {}))
    regional_multipliers: dict[str, float] = {}
    if isinstance(regional_raw, dict):
        for region_name, mult in regional_raw.items():
            regional_multipliers[str(region_name).strip().lower()] = max(0.6, min(1.5, _safe_float(mult, 1.0)))
    if "uk_default" not in regional_multipliers:
        regional_multipliers["uk_default"] = 1.0

    snapshot_as_of = str(selected_row.get("as_of_utc", selected_row.get("as_of", ""))).strip() or None
    live_diag = payload.get("live_diagnostics")
    return FuelPriceSnapshot(
        prices_gbp_per_l=prices,
        grid_price_gbp_per_kwh=max(0.0, grid_price),
        regional_multipliers=regional_multipliers,
        as_of=snapshot_as_of,
        source=source_label,
        signature=signature,
        live_diagnostics=(live_diag if isinstance(live_diag, dict) else None),
    )


@lru_cache(maxsize=16)
def load_fuel_price_snapshot(as_of_utc: datetime | None = None) -> FuelPriceSnapshot:
    strict = _strict_runtime_required()
    require_signature = bool(settings.live_fuel_require_signature) if strict else False
    live_url = str(settings.live_fuel_price_url or "").strip()
    require_live_url = bool(settings.live_fuel_require_url_in_strict) if strict else False

    live_failure: ModelDataError | None = None
    if settings.live_runtime_data_enabled and live_url:
        live_payload = live_fuel_prices(as_of_utc)
        if isinstance(live_payload, dict) and "_live_error" in live_payload:
            live_err = live_payload.get("_live_error", {})
            if isinstance(live_err, dict):
                reason = str(live_err.get("reason_code", "fuel_price_source_unavailable")).strip()
                code = "fuel_price_auth_unavailable" if reason == "fuel_price_auth_unavailable" else "fuel_price_source_unavailable"
                live_failure = ModelDataError(
                    reason_code=code,
                    message=str(live_err.get("message", "Live fuel source error")).strip() or "Live fuel source error",
                    details=(live_err.get("diagnostics") if isinstance(live_err.get("diagnostics"), dict) else None),
                )
        elif isinstance(live_payload, dict):
            try:
                live_snapshot = _coerce_fuel_snapshot(
                    payload=live_payload,
                    source_label=str(live_payload.get("source", "")).strip() or "live_runtime:fuel_prices",
                    as_of_utc=as_of_utc,
                    require_signature=require_signature,
                )
                live_as_of = _infer_as_of_from_payload(live_payload) or _parse_iso_datetime(live_snapshot.as_of)
                _raise_if_strict_stale(
                    reason_code="fuel_price_source_unavailable",
                    message="Live fuel price source is stale for strict runtime policy.",
                    as_of_utc=live_as_of,
                    max_age_days=int(settings.live_fuel_max_age_days),
                    enforce=True,
                )
                return live_snapshot
            except ModelDataError as exc:
                live_failure = exc
    elif strict and require_live_url:
        live_failure = ModelDataError(
            reason_code="fuel_price_source_unavailable",
            message="Fuel live price URL is required in strict runtime policy.",
        )

    if live_failure is not None:
        raise live_failure
    raise ModelDataError(
        reason_code="fuel_price_source_unavailable",
        message="Live fuel price source is unavailable in strict runtime.",
    )


@dataclass(frozen=True)
class StochasticRegime:
    regime_id: str
    sigma_scale: float
    traffic_scale: float
    incident_scale: float
    weather_scale: float
    price_scale: float
    eco_scale: float
    corr: tuple[tuple[float, ...], ...]
    spread_floor: float = 0.05
    spread_cap: float = 1.25
    factor_low: float = 0.55
    factor_high: float = 2.20
    duration_mix: tuple[float, float, float] = (1.0, 1.0, 1.0)
    monetary_mix: tuple[float, float] = (0.62, 0.38)
    emissions_mix: tuple[float, float] = (0.72, 0.28)
    transform_family: str = "log_exp"
    shock_quantile_mapping: dict[str, tuple[tuple[float, float], ...]] | None = None


@dataclass(frozen=True)
class StochasticRegimeTable:
    copula_id: str
    calibration_version: str
    regimes: dict[str, StochasticRegime]
    source: str
    as_of_utc: str | None = None
    posterior_model: dict[str, Any] | None = None
    split_strategy: str | None = None
    holdout_window: dict[str, str] | None = None
    holdout_metrics: dict[str, float] | None = None
    coverage_metrics: dict[str, float] | None = None


@lru_cache(maxsize=1)
def load_stochastic_regimes() -> StochasticRegimeTable:
    strict = _strict_runtime_required()
    strict_empirical_required = _strict_empirical_stochastic_required()
    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    live_url = str(settings.live_stochastic_regimes_url or "").strip()
    require_live_url = bool(settings.live_stochastic_require_url_in_strict) if strict else False
    payload: dict[str, Any] = {}
    source: str = ""
    rejected_synthetic = False
    live_failure: ModelDataError | None = None
    if strict and require_live_url and not live_url and not pytest_bypass_enabled:
        raise ModelDataError(
            reason_code="stochastic_calibration_unavailable",
            message="LIVE_STOCHASTIC_REGIMES_URL is required in strict runtime.",
        )
    if settings.live_runtime_data_enabled and live_url:
        live_payload = live_stochastic_regimes()
        if isinstance(live_payload, dict):
            live_as_of = _infer_as_of_from_payload(live_payload)
            live_is_synthetic = _payload_is_synthetic(
                live_payload,
                version=str(live_payload.get("calibration_version", "")),
            )
            if (
                strict_empirical_required
                and live_is_synthetic
                and not settings.stochastic_allow_synthetic_calibration
            ):
                rejected_synthetic = True
                live_failure = ModelDataError(
                    reason_code="stochastic_calibration_unavailable",
                    message="Live stochastic calibration payload uses synthetic/legacy basis in strict runtime.",
                )
            elif _is_fresh(live_as_of, max_age_days=int(settings.live_stochastic_max_age_days)):
                payload = live_payload
                source = "live_runtime:stochastic_regimes"
            else:
                live_failure = ModelDataError(
                    reason_code="stochastic_calibration_unavailable",
                    message="Live stochastic calibration payload is stale for strict runtime policy.",
                    details={
                        "as_of_utc": live_as_of.isoformat() if live_as_of is not None else None,
                        "max_age_days": int(settings.live_stochastic_max_age_days),
                    },
                )
        elif strict and require_live_url:
            live_failure = ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Live stochastic calibration payload is unavailable in strict runtime.",
            )
    if not isinstance(payload, dict):
        payload = {}
    if strict_empirical_required and not payload:
        if live_failure is not None:
            raise live_failure
        raise ModelDataError(
            reason_code="stochastic_calibration_unavailable",
            message="No stochastic calibration payload is available under strict live-data policy.",
        )
    raw_regimes = payload.get("regimes", {})
    regimes: dict[str, StochasticRegime] = {}
    if isinstance(raw_regimes, dict):
        for regime_id, item in raw_regimes.items():
            if not isinstance(item, dict):
                continue
            corr_raw = item.get("corr", [])
            corr: list[tuple[float, ...]] = []
            if isinstance(corr_raw, list):
                for row in corr_raw:
                    if isinstance(row, list) and len(row) == 5:
                        corr.append(tuple(float(v) for v in row))
            if len(corr) != 5:
                if strict_empirical_required:
                    raise ModelDataError(
                        reason_code="stochastic_calibration_unavailable",
                        message=(
                            "Stochastic regime correlation matrix is missing/invalid "
                            f"for regime '{regime_id}' under strict runtime."
                        ),
                    )
                corr = [
                    (1.0, 0.45, 0.35, 0.20, 0.15),
                    (0.45, 1.0, 0.25, 0.30, 0.20),
                    (0.35, 0.25, 1.0, 0.28, 0.32),
                    (0.20, 0.30, 0.28, 1.0, 0.18),
                    (0.15, 0.20, 0.32, 0.18, 1.0),
                ]
            duration_mix_raw = item.get("duration_mix", [1.0, 1.0, 1.0])
            monetary_mix_raw = item.get("monetary_mix", [0.62, 0.38])
            emissions_mix_raw = item.get("emissions_mix", [0.72, 0.28])
            duration_mix = (1.0, 1.0, 1.0)
            monetary_mix = (0.62, 0.38)
            emissions_mix = (0.72, 0.28)
            if isinstance(duration_mix_raw, list) and len(duration_mix_raw) == 3:
                duration_mix = (
                    max(0.0, _safe_float(duration_mix_raw[0], 1.0)),
                    max(0.0, _safe_float(duration_mix_raw[1], 1.0)),
                    max(0.0, _safe_float(duration_mix_raw[2], 1.0)),
                )
            if isinstance(monetary_mix_raw, list) and len(monetary_mix_raw) == 2:
                monetary_mix = (
                    max(0.0, _safe_float(monetary_mix_raw[0], 0.62)),
                    max(0.0, _safe_float(monetary_mix_raw[1], 0.38)),
                )
            if isinstance(emissions_mix_raw, list) and len(emissions_mix_raw) == 2:
                emissions_mix = (
                    max(0.0, _safe_float(emissions_mix_raw[0], 0.72)),
                    max(0.0, _safe_float(emissions_mix_raw[1], 0.28)),
                )
            transform_family = str(item.get("transform_family", "log_exp")).strip().lower() or "log_exp"
            shock_quantile_mapping: dict[str, tuple[tuple[float, float], ...]] | None = None
            raw_mapping = item.get("shock_quantile_mapping")
            if isinstance(raw_mapping, dict):
                parsed_mapping: dict[str, tuple[tuple[float, float], ...]] = {}
                for factor_name, rows in raw_mapping.items():
                    if not isinstance(rows, list):
                        continue
                    points: list[tuple[float, float]] = []
                    for row in rows:
                        if isinstance(row, (list, tuple)) and len(row) >= 2:
                            z = _safe_float(row[0], float("nan"))
                            value = _safe_float(row[1], float("nan"))
                            if z == z and value == value:
                                points.append((float(z), max(0.05, float(value))))
                    points.sort(key=lambda pair: pair[0])
                    if len(points) >= 2:
                        parsed_mapping[str(factor_name).strip().lower()] = tuple(points)
                if parsed_mapping:
                    shock_quantile_mapping = parsed_mapping
            regimes[str(regime_id)] = StochasticRegime(
                regime_id=str(regime_id),
                sigma_scale=max(0.1, _safe_float(item.get("sigma_scale"), 1.0)),
                traffic_scale=max(0.1, _safe_float(item.get("traffic_scale"), 1.0)),
                incident_scale=max(0.1, _safe_float(item.get("incident_scale"), 1.0)),
                weather_scale=max(0.1, _safe_float(item.get("weather_scale"), 1.0)),
                price_scale=max(0.1, _safe_float(item.get("price_scale"), 1.0)),
                eco_scale=max(0.1, _safe_float(item.get("eco_scale"), 1.0)),
                corr=tuple(corr),
                spread_floor=max(0.01, _safe_float(item.get("spread_floor"), 0.05)),
                spread_cap=max(0.15, _safe_float(item.get("spread_cap"), 1.25)),
                factor_low=max(0.05, _safe_float(item.get("factor_low"), 0.55)),
                factor_high=max(1.0, _safe_float(item.get("factor_high"), 2.20)),
                duration_mix=duration_mix,
                monetary_mix=monetary_mix,
                emissions_mix=emissions_mix,
                transform_family=transform_family,
                shock_quantile_mapping=shock_quantile_mapping,
            )
    if not regimes:
        if rejected_synthetic:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Only synthetic stochastic calibration assets were available.",
            )
        raise ModelDataError(
            reason_code="stochastic_calibration_unavailable",
            message="No valid stochastic calibration regimes were available.",
        )
    posterior_model = payload.get("posterior_model") if isinstance(payload.get("posterior_model"), dict) else None
    split_strategy_raw = str(payload.get("split_strategy", "")).strip()
    split_strategy = split_strategy_raw.lower() if split_strategy_raw else ""
    holdout_window_raw = payload.get("holdout_window")
    holdout_window = holdout_window_raw if isinstance(holdout_window_raw, dict) else None
    holdout_metrics_raw = payload.get("holdout_metrics")
    holdout_metrics = holdout_metrics_raw if isinstance(holdout_metrics_raw, dict) else None
    coverage_metrics_raw = payload.get("coverage_metrics")
    coverage_metrics = coverage_metrics_raw if isinstance(coverage_metrics_raw, dict) else None
    if strict_empirical_required:
        if split_strategy != "temporal_forward_plus_corridor_block":
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message=(
                    "Stochastic calibration split_strategy must be "
                    "'temporal_forward_plus_corridor_block' in strict runtime."
                ),
                details={"split_strategy": split_strategy_raw or None},
            )
        if not isinstance(holdout_window, dict):
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Stochastic calibration holdout_window is required in strict runtime.",
            )
        holdout_start = str(holdout_window.get("start_utc", "")).strip()
        holdout_end = str(holdout_window.get("end_utc", "")).strip()
        if not holdout_start or not holdout_end:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Stochastic calibration holdout_window must include start_utc/end_utc in strict runtime.",
            )
        if not isinstance(holdout_metrics, dict):
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Stochastic calibration holdout_metrics are required in strict runtime.",
            )
        pit_mean = float(_safe_float(holdout_metrics.get("pit_mean"), float("nan")))
        holdout_coverage = float(_safe_float(holdout_metrics.get("coverage"), float("nan")))
        crps_mean = float(_safe_float(holdout_metrics.get("crps_mean"), float("nan")))
        duration_mape = float(_safe_float(holdout_metrics.get("duration_mape"), float("nan")))
        if not (pit_mean == pit_mean and holdout_coverage == holdout_coverage and crps_mean == crps_mean):
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message=(
                    "Stochastic calibration holdout_metrics must include numeric "
                    "pit_mean, coverage, and crps_mean in strict runtime."
                ),
            )
        if holdout_coverage < 0.90:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Stochastic holdout coverage is below strict threshold (>= 0.90 required).",
                details={"coverage": holdout_coverage},
            )
        if crps_mean > 0.55:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Stochastic holdout CRPS proxy exceeds strict threshold (<= 0.55 required).",
                details={"crps_mean": crps_mean},
            )
        if pit_mean < 0.35 or pit_mean > 0.65:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Stochastic holdout PIT mean is outside strict calibration band [0.35, 0.65].",
                details={"pit_mean": pit_mean},
            )
        if duration_mape == duration_mape and duration_mape > 0.15:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message="Stochastic holdout duration MAPE exceeds strict threshold (<= 0.15 required).",
                details={"duration_mape": duration_mape},
            )
        if not isinstance(posterior_model, dict):
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message=(
                    "Stochastic calibration payload is missing posterior_model under strict runtime "
                    "(context_to_regime_probs required)."
                ),
            )
        context_probs = posterior_model.get("context_to_regime_probs")
        if not isinstance(context_probs, dict) or not context_probs:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message=(
                    "Stochastic calibration posterior_model.context_to_regime_probs is missing/empty "
                    "under strict runtime."
                ),
            )
        live_slot_tokens: set[str] = set()
        live_corridor_tokens: set[str] = set()
        for context_key in context_probs.keys():
            parts = [part.strip().lower() for part in str(context_key).split("|")]
            if len(parts) >= 1:
                corridor = parts[0]
                if corridor and corridor != "*":
                    live_corridor_tokens.add(corridor)
            if len(parts) >= 3:
                slot = parts[2]
                if slot and slot != "*":
                    live_slot_tokens.add(slot)
        if len(live_slot_tokens) < 6:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message=(
                    "Stochastic posterior context hour-slot diversity is below strict threshold "
                    f"(actual={len(live_slot_tokens)}, required>=6)."
                ),
            )
        if len(live_corridor_tokens) < 8:
            raise ModelDataError(
                reason_code="stochastic_calibration_unavailable",
                message=(
                    "Stochastic posterior context corridor diversity is below strict threshold "
                    f"(actual={len(live_corridor_tokens)}, required>=8)."
                ),
            )
        if isinstance(coverage_metrics, dict):
            reported_hour_cov = float(_safe_float(coverage_metrics.get("hour_slot_coverage"), 0.0))
            reported_corridor_cov = float(_safe_float(coverage_metrics.get("corridor_coverage"), 0.0))
            if reported_hour_cov < 6.0 or reported_corridor_cov < 8.0:
                raise ModelDataError(
                    reason_code="stochastic_calibration_unavailable",
                    message=(
                        "Stochastic coverage_metrics are below strict diversity thresholds "
                        "(hour>=6, corridor>=8)."
                    ),
                    details={
                        "hour_slot_coverage": reported_hour_cov,
                        "corridor_coverage": reported_corridor_cov,
                    },
                )
        required_factors = {"traffic", "incident", "weather", "price", "eco"}
        for regime_id, regime in regimes.items():
            if str(regime.transform_family).strip().lower() != "quantile_mapping_v1":
                raise ModelDataError(
                    reason_code="stochastic_calibration_unavailable",
                    message=(
                        "Stochastic regime transform_family must be 'quantile_mapping_v1' in strict runtime "
                        f"(regime={regime_id})."
                    ),
                )
            mapping = regime.shock_quantile_mapping
            if not isinstance(mapping, dict):
                raise ModelDataError(
                    reason_code="stochastic_calibration_unavailable",
                    message=(
                        "Stochastic regime is missing shock_quantile_mapping in strict runtime "
                        f"(regime={regime_id})."
                    ),
                )
            missing_factors = [name for name in sorted(required_factors) if not mapping.get(name)]
            if missing_factors:
                raise ModelDataError(
                    reason_code="stochastic_calibration_unavailable",
                    message=(
                        "Stochastic regime shock_quantile_mapping is missing required factors in strict runtime "
                        f"(regime={regime_id}, missing={','.join(missing_factors)})."
                    ),
                )
    live_as_of = _infer_as_of_from_payload(payload)
    as_of_utc = live_as_of.isoformat() if live_as_of is not None else None
    return StochasticRegimeTable(
        copula_id=str(payload.get("copula_id", "gaussian_5x5_v2")),
        calibration_version=str(payload.get("calibration_version", "v2-uk")),
        regimes=regimes,
        source=source,
        as_of_utc=as_of_utc,
        posterior_model=posterior_model,
        split_strategy=(split_strategy_raw or None),
        holdout_window=holdout_window,
        holdout_metrics=(
            {
                str(key): float(value)
                for key, value in holdout_metrics.items()
                if isinstance(value, (int, float))
            }
            if isinstance(holdout_metrics, dict)
            else None
        ),
        coverage_metrics=(
            {
                str(key): float(value)
                for key, value in coverage_metrics.items()
                if isinstance(value, (int, float))
            }
            if isinstance(coverage_metrics, dict)
            else None
        ),
    )


@dataclass(frozen=True)
class StochasticResidualPrior:
    sigma_floor: float
    sample_count: int
    source: str
    calibration_version: str
    as_of_utc: str | None
    prior_id: str


@lru_cache(maxsize=256)
def load_stochastic_residual_prior(
    *,
    day_kind: str,
    road_bucket: str,
    weather_profile: str,
    vehicle_type: str | None,
    vehicle_bucket: str | None = None,
    corridor_bucket: str | None = None,
    local_time_slot: str | None = None,
) -> StochasticResidualPrior:
    day = _canonical_day_kind(day_kind)
    road = (road_bucket or "mixed").strip().lower() or "mixed"
    weather = (weather_profile or "clear").strip().lower() or "clear"
    vehicle_bucket = _canonical_vehicle_bucket(
        vehicle_type,
        vehicle_bucket=vehicle_bucket,
        bucket_attr="stochastic_bucket",
    )
    corridor = _canonical_corridor_bucket(corridor_bucket)
    slot = _canonical_local_time_slot(local_time_slot)
    pytest_bypass_enabled = _strict_runtime_test_bypass_enabled()
    pytest_context = _pytest_active()
    exact_key = f"{corridor}_{day}_{slot}_{road}_{weather}_{vehicle_bucket}"
    candidate_keys: list[str] = [exact_key]
    if pytest_bypass_enabled or pytest_context:
        # Keep strict runtime exact-match semantics in production. This fallback
        # exists only for deterministic pytest fixture lanes.
        candidate_keys.extend(
            [
                f"uk_default_{day}_{slot}_{road}_{weather}_{vehicle_bucket}",
                f"uk_default_{day}_h12_{road}_{weather}_{vehicle_bucket}",
                f"uk_default_{day}_{slot}_{road}_{weather}_default",
                f"uk_default_{day}_h12_{road}_{weather}_default",
            ]
        )

    path_candidates = [
        _model_asset_root() / "stochastic_residual_priors_uk.json",
        _assets_root() / "stochastic_residual_priors_uk.json",
    ]
    payload: dict[str, Any] | None = None
    source = ""
    selected_path: Path | None = None
    for path in path_candidates:
        if not path.exists():
            continue
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            payload = parsed
            source = str(path)
            selected_path = path
            break
    if payload is None:
        raise ModelDataError(
            reason_code="risk_prior_unavailable",
            message="Stochastic residual priors are required in strict runtime.",
        )
    _raise_if_strict_stale(
        reason_code="risk_prior_unavailable",
        message="Stochastic residual priors are stale for strict runtime policy.",
        as_of_utc=_infer_as_of_from_payload(payload) or (_infer_as_of_from_path(selected_path) if selected_path else None),
        max_age_days=int(settings.live_stochastic_max_age_days),
        enforce=True,
    )

    priors = payload.get("priors", {})
    if isinstance(priors, dict):
        for key in candidate_keys:
            prior = priors.get(key)
            if isinstance(prior, dict):
                sigma_floor = max(0.005, min(0.75, _safe_float(prior.get("sigma_floor"), 0.0)))
                sample_count = max(1, int(_safe_float(prior.get("sample_count"), 0)))
                return StochasticResidualPrior(
                    sigma_floor=sigma_floor,
                    sample_count=sample_count,
                    source=source,
                    calibration_version=str(payload.get("calibration_version", "v2")),
                    as_of_utc=str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None,
                    prior_id=str(key),
                )
        if pytest_bypass_enabled or pytest_context:
            suffixes = [
                f"_{day}_{slot}_{road}_{weather}_{vehicle_bucket}",
                f"_{day}_h12_{road}_{weather}_{vehicle_bucket}",
                f"_{day}_{slot}_{road}_{weather}_default",
                f"_{day}_h12_{road}_{weather}_default",
            ]
            for suffix in suffixes:
                for key in sorted(priors):
                    prior = priors.get(key)
                    if isinstance(key, str) and isinstance(prior, dict) and key.endswith(suffix):
                        sigma_floor = max(0.005, min(0.75, _safe_float(prior.get("sigma_floor"), 0.0)))
                        sample_count = max(1, int(_safe_float(prior.get("sample_count"), 0)))
                        return StochasticResidualPrior(
                            sigma_floor=sigma_floor,
                            sample_count=sample_count,
                            source=source,
                            calibration_version=str(payload.get("calibration_version", "v2")),
                            as_of_utc=str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None,
                            prior_id=str(key),
                        )
    if isinstance(priors, list):
        indexed: dict[str, dict[str, Any]] = {}
        for row in priors:
            if not isinstance(row, dict):
                continue
            dk = _canonical_day_kind(str(row.get("day_kind", "weekday")))
            rb = str(row.get("road_bucket", "mixed")).strip().lower() or "mixed"
            wp = str(row.get("weather_profile", "clear")).strip().lower() or "clear"
            vb = _canonical_vehicle_bucket(
                str(row.get("vehicle_type", "default")),
                bucket_attr="stochastic_bucket",
            )
            cb = _canonical_corridor_bucket(str(row.get("corridor_bucket", "uk_default")))
            slot_key = _canonical_local_time_slot(str(row.get("local_time_slot", "h12")))
            key = f"{cb}_{dk}_{slot_key}_{rb}_{wp}_{vb}"
            indexed[key] = row
        for key in candidate_keys:
            prior = indexed.get(key)
            if prior is not None:
                sigma_floor = max(0.005, min(0.75, _safe_float(prior.get("sigma_floor"), 0.0)))
                sample_count = max(1, int(_safe_float(prior.get("sample_count"), 0)))
                return StochasticResidualPrior(
                    sigma_floor=sigma_floor,
                    sample_count=sample_count,
                    source=source,
                    calibration_version=str(payload.get("calibration_version", "v2")),
                    as_of_utc=str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None,
                    prior_id=str(prior.get("prior_id", key)),
                )
        if pytest_bypass_enabled or pytest_context:
            suffixes = [
                f"_{day}_{slot}_{road}_{weather}_{vehicle_bucket}",
                f"_{day}_h12_{road}_{weather}_{vehicle_bucket}",
                f"_{day}_{slot}_{road}_{weather}_default",
                f"_{day}_h12_{road}_{weather}_default",
            ]
            for suffix in suffixes:
                for key in sorted(indexed):
                    if not key.endswith(suffix):
                        continue
                    prior = indexed[key]
                    sigma_floor = max(0.005, min(0.75, _safe_float(prior.get("sigma_floor"), 0.0)))
                    sample_count = max(1, int(_safe_float(prior.get("sample_count"), 0)))
                    return StochasticResidualPrior(
                        sigma_floor=sigma_floor,
                        sample_count=sample_count,
                        source=source,
                        calibration_version=str(payload.get("calibration_version", "v2")),
                        as_of_utc=str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None,
                        prior_id=str(prior.get("prior_id", key)),
                    )

    raise ModelDataError(
        reason_code="risk_prior_unavailable",
        message=(
            "No stochastic residual prior matched route context "
            f"(corridor={corridor}, day={day}, slot={slot}, road={road}, "
            f"weather={weather}, vehicle={vehicle_bucket})."
        ),
    )


Grid4D = list[list[list[list[float]]]]


@dataclass(frozen=True)
class FuelSurfaceAxes:
    vehicle_class: tuple[str, ...]
    load_factor: tuple[float, ...]
    speed_kmh: tuple[float, ...]
    grade_pct: tuple[float, ...]
    ambient_temp_c: tuple[float, ...]


@dataclass(frozen=True)
class FuelConsumptionSurface:
    source: str
    version: str
    as_of_utc: str | None
    signature: str | None
    axes: FuelSurfaceAxes
    fuel_l_per_100km: dict[str, Grid4D]
    energy_kwh_per_100km: dict[str, Grid4D]


@dataclass(frozen=True)
class FuelUncertaintySurface:
    source: str
    version: str
    as_of_utc: str | None
    signature: str | None
    axes: FuelSurfaceAxes
    fuel_liters_multiplier_quantiles: dict[str, dict[str, Grid4D]]
    energy_kwh_multiplier_quantiles: dict[str, dict[str, Grid4D]]
    fuel_cost_multiplier_quantiles: dict[str, dict[str, Grid4D]]


@dataclass(frozen=True)
class FuelConsumptionCalibration:
    source: str
    version: str
    as_of_utc: str | None
    axes: FuelSurfaceAxes


def _validate_axis(name: str, raw: Any, *, min_len: int = 2) -> tuple[float, ...]:
    if not isinstance(raw, list) or len(raw) < min_len:
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message=f"Fuel surface axis '{name}' is missing or too short.",
        )
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ModelDataError(
                reason_code="model_asset_unavailable",
                message=f"Fuel surface axis '{name}' contains non-numeric values.",
            ) from exc
    for idx in range(1, len(values)):
        if values[idx] <= values[idx - 1]:
            raise ModelDataError(
                reason_code="model_asset_unavailable",
                message=f"Fuel surface axis '{name}' must be strictly increasing.",
            )
    return tuple(values)


def _validate_vehicle_axis(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel surface vehicle_class axis is missing.",
        )
    items = tuple(str(item).strip().lower() for item in raw if str(item).strip())
    if not items:
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel surface vehicle_class axis is empty.",
        )
    return items


def _parse_grid4d(raw: Any, *, shape: tuple[int, int, int, int], label: str) -> Grid4D:
    if not isinstance(raw, list) or len(raw) != shape[0]:
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message=f"Fuel surface '{label}' has invalid first dimension.",
        )
    out: Grid4D = []
    for i in range(shape[0]):
        dim_i = raw[i]
        if not isinstance(dim_i, list) or len(dim_i) != shape[1]:
            raise ModelDataError(
                reason_code="model_asset_unavailable",
                message=f"Fuel surface '{label}' has invalid second dimension.",
            )
        out_j: list[list[list[float]]] = []
        for j in range(shape[1]):
            dim_j = dim_i[j]
            if not isinstance(dim_j, list) or len(dim_j) != shape[2]:
                raise ModelDataError(
                    reason_code="model_asset_unavailable",
                    message=f"Fuel surface '{label}' has invalid third dimension.",
                )
            out_k: list[list[float]] = []
            for k in range(shape[2]):
                dim_k = dim_j[k]
                if not isinstance(dim_k, list) or len(dim_k) != shape[3]:
                    raise ModelDataError(
                        reason_code="model_asset_unavailable",
                        message=f"Fuel surface '{label}' has invalid fourth dimension.",
                    )
                out_l: list[float] = []
                for value in dim_k:
                    try:
                        out_l.append(float(value))
                    except (TypeError, ValueError) as exc:
                        raise ModelDataError(
                            reason_code="model_asset_unavailable",
                            message=f"Fuel surface '{label}' contains non-numeric values.",
                        ) from exc
                out_k.append(out_l)
            out_j.append(out_k)
        out.append(out_j)
    return out


def _validate_generic_signature(
    *,
    payload: dict[str, Any],
    reason_code: str,
    message_prefix: str,
    require_signature: bool,
) -> str | None:
    signature = str(payload.get("signature", "")).strip() or None
    if not require_signature:
        return signature
    if not signature:
        raise ModelDataError(
            reason_code=reason_code,
            message=f"{message_prefix} signature is required in strict runtime.",
        )
    payload_no_sig = {k: v for k, v in payload.items() if k != "signature"}
    expected = hashlib.sha256(
        json.dumps(payload_no_sig, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if signature.lower() != expected.lower():
        raise ModelDataError(
            reason_code=reason_code,
            message=f"{message_prefix} signature verification failed in strict runtime.",
            details={"expected_signature": expected},
        )
    return signature


def _load_fuel_surface_payload(filename: str) -> tuple[dict[str, Any], str]:
    path_candidates = [
        _model_asset_root() / filename,
        _assets_root() / filename,
    ]
    for path in path_candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload, str(path)
    raise ModelDataError(
        reason_code="model_asset_unavailable",
        message=f"Fuel surface asset '{filename}' is required in strict runtime.",
    )


def _parse_fuel_axes(payload: dict[str, Any]) -> FuelSurfaceAxes:
    axes_raw = payload.get("axes")
    if not isinstance(axes_raw, dict):
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel surface payload is missing 'axes'.",
        )
    axes = FuelSurfaceAxes(
        vehicle_class=_validate_vehicle_axis(axes_raw.get("vehicle_class")),
        load_factor=_validate_axis("load_factor", axes_raw.get("load_factor")),
        speed_kmh=_validate_axis("speed_kmh", axes_raw.get("speed_kmh")),
        grade_pct=_validate_axis("grade_pct", axes_raw.get("grade_pct")),
        ambient_temp_c=_validate_axis("ambient_temp_c", axes_raw.get("ambient_temp_c")),
    )
    required_classes = {"van", "rigid_hgv", "artic_hgv", "ev"}
    if not required_classes.issubset(set(axes.vehicle_class)):
        missing = sorted(required_classes - set(axes.vehicle_class))
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel surface vehicle classes missing required coverage: " + ", ".join(missing),
        )
    return axes


@lru_cache(maxsize=1)
def load_fuel_consumption_surface() -> FuelConsumptionSurface:
    payload, source = _load_fuel_surface_payload("fuel_consumption_surface_uk.json")
    signature = _validate_generic_signature(
        payload=payload,
        reason_code="model_asset_unavailable",
        message_prefix="Fuel consumption surface",
        require_signature=bool(settings.live_fuel_require_signature and _strict_runtime_required()),
    )
    axes = _parse_fuel_axes(payload)
    shape = (
        len(axes.load_factor),
        len(axes.speed_kmh),
        len(axes.grade_pct),
        len(axes.ambient_temp_c),
    )
    fuel_raw = payload.get("fuel_l_per_100km", {})
    if not isinstance(fuel_raw, dict):
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel consumption surface missing fuel_l_per_100km map.",
        )
    energy_raw = payload.get("energy_kwh_per_100km", {})
    if not isinstance(energy_raw, dict):
        energy_raw = {}

    fuel_values: dict[str, Grid4D] = {}
    energy_values: dict[str, Grid4D] = {}
    for klass in axes.vehicle_class:
        if klass in fuel_raw:
            fuel_values[klass] = _parse_grid4d(fuel_raw[klass], shape=shape, label=f"fuel_l_per_100km[{klass}]")
        if klass in energy_raw:
            energy_values[klass] = _parse_grid4d(
                energy_raw[klass],
                shape=shape,
                label=f"energy_kwh_per_100km[{klass}]",
            )
    for required in ("van", "rigid_hgv", "artic_hgv"):
        if required not in fuel_values:
            raise ModelDataError(
                reason_code="model_asset_unavailable",
                message=f"Fuel consumption surface missing '{required}' fuel table.",
            )
    if "ev" not in energy_values:
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel consumption surface missing 'ev' energy table.",
        )
    return FuelConsumptionSurface(
        source=source,
        version=str(payload.get("version", payload.get("calibration_version", "unknown"))),
        as_of_utc=str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None,
        signature=signature,
        axes=axes,
        fuel_l_per_100km=fuel_values,
        energy_kwh_per_100km=energy_values,
    )


def _parse_quantile_table(
    raw: Any,
    *,
    shape: tuple[int, int, int, int],
    label: str,
) -> dict[str, Grid4D]:
    if not isinstance(raw, dict):
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message=f"{label} is missing.",
        )
    out: dict[str, Grid4D] = {}
    for q in ("p10", "p50", "p90"):
        if q not in raw:
            raise ModelDataError(
                reason_code="model_asset_unavailable",
                message=f"{label} missing required quantile '{q}'.",
            )
        out[q] = _parse_grid4d(raw[q], shape=shape, label=f"{label}.{q}")
    return out


@lru_cache(maxsize=1)
def load_fuel_uncertainty_surface() -> FuelUncertaintySurface:
    payload, source = _load_fuel_surface_payload("fuel_uncertainty_surface_uk.json")
    signature = _validate_generic_signature(
        payload=payload,
        reason_code="model_asset_unavailable",
        message_prefix="Fuel uncertainty surface",
        require_signature=bool(settings.live_fuel_require_signature and _strict_runtime_required()),
    )
    axes = _parse_fuel_axes(payload)
    shape = (
        len(axes.load_factor),
        len(axes.speed_kmh),
        len(axes.grade_pct),
        len(axes.ambient_temp_c),
    )
    fuel_raw = payload.get("fuel_liters_multiplier_quantiles", {})
    energy_raw = payload.get("energy_kwh_multiplier_quantiles", {})
    cost_raw = payload.get("fuel_cost_multiplier_quantiles", fuel_raw)
    if not isinstance(fuel_raw, dict) or not isinstance(energy_raw, dict) or not isinstance(cost_raw, dict):
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel uncertainty surface is missing required quantile tables.",
        )

    fuel_q: dict[str, dict[str, Grid4D]] = {}
    energy_q: dict[str, dict[str, Grid4D]] = {}
    cost_q: dict[str, dict[str, Grid4D]] = {}
    for klass in axes.vehicle_class:
        if klass in fuel_raw:
            fuel_q[klass] = _parse_quantile_table(
                fuel_raw[klass],
                shape=shape,
                label=f"fuel_liters_multiplier_quantiles[{klass}]",
            )
        if klass in energy_raw:
            energy_q[klass] = _parse_quantile_table(
                energy_raw[klass],
                shape=shape,
                label=f"energy_kwh_multiplier_quantiles[{klass}]",
            )
        if klass in cost_raw:
            cost_q[klass] = _parse_quantile_table(
                cost_raw[klass],
                shape=shape,
                label=f"fuel_cost_multiplier_quantiles[{klass}]",
            )
    for klass, quantiles in {**fuel_q, **energy_q, **cost_q}.items():
        if not quantiles:
            continue
        p10 = quantiles["p10"]
        p50 = quantiles["p50"]
        p90 = quantiles["p90"]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for idx_l in range(shape[3]):
                        q10 = p10[i][j][k][idx_l]
                        q50 = p50[i][j][k][idx_l]
                        q90 = p90[i][j][k][idx_l]
                        if not (q10 <= q50 <= q90):
                            raise ModelDataError(
                                reason_code="model_asset_unavailable",
                                message=f"Fuel uncertainty surface quantiles invalid for class '{klass}'.",
                            )
    for required in ("van", "rigid_hgv", "artic_hgv"):
        if required not in fuel_q:
            raise ModelDataError(
                reason_code="model_asset_unavailable",
                message=f"Fuel uncertainty surface missing '{required}' fuel quantiles.",
            )
        if required not in cost_q:
            raise ModelDataError(
                reason_code="model_asset_unavailable",
                message=f"Fuel uncertainty surface missing '{required}' cost quantiles.",
            )
    if "ev" not in energy_q:
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel uncertainty surface missing 'ev' energy quantiles.",
        )
    if "ev" not in cost_q:
        raise ModelDataError(
            reason_code="model_asset_unavailable",
            message="Fuel uncertainty surface missing 'ev' cost quantiles.",
        )

    return FuelUncertaintySurface(
        source=source,
        version=str(payload.get("version", payload.get("calibration_version", "unknown"))),
        as_of_utc=str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None,
        signature=signature,
        axes=axes,
        fuel_liters_multiplier_quantiles=fuel_q,
        energy_kwh_multiplier_quantiles=energy_q,
        fuel_cost_multiplier_quantiles=cost_q,
    )


@lru_cache(maxsize=1)
def load_fuel_consumption_calibration() -> FuelConsumptionCalibration:
    surface = load_fuel_consumption_surface()
    return FuelConsumptionCalibration(
        source=surface.source,
        version=surface.version,
        as_of_utc=surface.as_of_utc,
        axes=surface.axes,
    )
