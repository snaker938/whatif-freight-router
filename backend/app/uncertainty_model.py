from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .calibration_loader import StochasticRegime, load_stochastic_regimes, load_uk_bank_holidays
from .model_data_errors import ModelDataError
from .risk_model import cvar, normalized_weighted_utility, quantile, robust_objective
from .settings import settings

try:
    UK_TZ = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:
    UK_TZ = timezone.utc


def _stable_seed(*, route_signature: str, departure_slot: str, user_seed: int) -> int:
    material = f"{route_signature}|{departure_slot}|{user_seed}"
    return int(hashlib.sha1(material.encode("utf-8")).hexdigest()[:16], 16)


def _departure_slot(departure_time_utc: datetime | None) -> str:
    if departure_time_utc is None:
        return "slot:none"
    aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
        tzinfo=timezone.utc
    )
    local = aware.astimezone(UK_TZ)
    # 15-minute deterministic slot for stable replay behavior.
    minute_bucket = (local.minute // 15) * 15
    return local.strftime(f"%Y-%m-%dT%H:{minute_bucket:02d}%z")


def _local_time_slot(departure_time_utc: datetime | None) -> str:
    if departure_time_utc is None:
        return "h12"
    aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
        tzinfo=timezone.utc
    )
    local = aware.astimezone(UK_TZ)
    return f"h{int(local.hour):02d}"


def _local_day_kind(departure_time_utc: datetime | None) -> str:
    if departure_time_utc is None:
        return "weekday"
    aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
        tzinfo=timezone.utc
    )
    local = aware.astimezone(UK_TZ)
    holiday_dates = load_uk_bank_holidays()
    if local.date().isoformat() in holiday_dates:
        return "holiday"
    if local.weekday() >= 5:
        return "weekend"
    return "weekday"


def _canonical_corridor_bucket(corridor_bucket: str | None) -> str:
    key = (corridor_bucket or "").strip().lower()
    if not key:
        return "uk_default"
    return key


def _canonical_vehicle_bucket(vehicle_bucket: str | None) -> str:
    key = (vehicle_bucket or "").strip().lower()
    if not key:
        return "default"
    return key


def _stochastic_context_key(
    *,
    corridor_bucket: str,
    day_kind: str,
    local_time_slot: str,
    road_bucket: str,
    weather_profile: str,
    vehicle_bucket: str,
) -> str:
    return (
        f"{(corridor_bucket or 'uk_default').strip().lower() or 'uk_default'}|"
        f"{(day_kind or 'weekday').strip().lower() or 'weekday'}|"
        f"{(local_time_slot or 'h12').strip().lower() or 'h12'}|"
        f"{(road_bucket or 'mixed').strip().lower() or 'mixed'}|"
        f"{(weather_profile or 'clear').strip().lower() or 'clear'}|"
        f"{(vehicle_bucket or 'default').strip().lower() or 'default'}"
    )


def _posterior_regime_candidates(
    *,
    table: Any,
    corridor_bucket: str,
    day_kind: str,
    local_time_slot: str,
    road_bucket: str,
    weather_profile: str,
    vehicle_bucket: str,
) -> list[str]:
    posterior = getattr(table, "posterior_model", None)
    if not isinstance(posterior, dict):
        return []
    context_probs = posterior.get("context_to_regime_probs", {})
    if not isinstance(context_probs, dict):
        return []
    normalized_slot = (local_time_slot or "h12").strip().lower() or "h12"
    normalized_vehicle = (vehicle_bucket or "default").strip().lower() or "default"
    keys = [
        _stochastic_context_key(
            corridor_bucket=corridor_bucket,
            day_kind=day_kind,
            local_time_slot=normalized_slot,
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket=normalized_vehicle,
        ),
        _stochastic_context_key(
            corridor_bucket="*",
            day_kind=day_kind,
            local_time_slot=normalized_slot,
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket=normalized_vehicle,
        ),
        _stochastic_context_key(
            corridor_bucket=corridor_bucket,
            day_kind=day_kind,
            local_time_slot="h12",
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket=normalized_vehicle,
        ),
        _stochastic_context_key(
            corridor_bucket="*",
            day_kind=day_kind,
            local_time_slot="h12",
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket=normalized_vehicle,
        ),
        _stochastic_context_key(
            corridor_bucket=corridor_bucket,
            day_kind=day_kind,
            local_time_slot="*",
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket=normalized_vehicle,
        ),
        _stochastic_context_key(
            corridor_bucket="*",
            day_kind=day_kind,
            local_time_slot="*",
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket=normalized_vehicle,
        ),
        _stochastic_context_key(
            corridor_bucket=corridor_bucket,
            day_kind=day_kind,
            local_time_slot="h12",
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket="default",
        ),
        _stochastic_context_key(
            corridor_bucket="*",
            day_kind=day_kind,
            local_time_slot="h12",
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket="default",
        ),
        _stochastic_context_key(
            corridor_bucket=corridor_bucket,
            day_kind=day_kind,
            local_time_slot="*",
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket="default",
        ),
        _stochastic_context_key(
            corridor_bucket="*",
            day_kind=day_kind,
            local_time_slot="*",
            road_bucket=road_bucket,
            weather_profile=weather_profile,
            vehicle_bucket="default",
        ),
    ]
    out: list[str] = []
    for key in keys:
        row = context_probs.get(key)
        if not isinstance(row, dict):
            continue
        ranked = sorted(
            (
                (str(regime_id), float(weight))
                for regime_id, weight in row.items()
                if str(regime_id).strip()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        for regime_id, _weight in ranked:
            if regime_id not in out:
                out.append(regime_id)
    return out


@dataclass(frozen=True)
class UncertaintySummary:
    mean_duration_s: float
    std_duration_s: float
    q50_duration_s: float
    q90_duration_s: float
    q95_duration_s: float
    cvar95_duration_s: float
    mean_monetary_cost: float
    std_monetary_cost: float
    q50_monetary_cost: float
    q90_monetary_cost: float
    q95_monetary_cost: float
    cvar95_monetary_cost: float
    mean_emissions_kg: float
    std_emissions_kg: float
    q50_emissions_kg: float
    q90_emissions_kg: float
    q95_emissions_kg: float
    cvar95_emissions_kg: float
    utility_mean: float
    utility_q95: float
    utility_cvar95: float
    robust_score: float
    sample_count_requested: float
    sample_count_used: float
    sample_count_clip_ratio: float
    sigma_requested: float
    sigma_used: float
    sigma_clip_ratio: float
    factor_clip_rate: float
    objective_samples: tuple[tuple[float, float, float, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "mean_duration_s": self.mean_duration_s,
            "std_duration_s": self.std_duration_s,
            "q50_duration_s": self.q50_duration_s,
            "q90_duration_s": self.q90_duration_s,
            "q95_duration_s": self.q95_duration_s,
            "p95_duration_s": self.q95_duration_s,
            "cvar95_duration_s": self.cvar95_duration_s,
            "mean_monetary_cost": self.mean_monetary_cost,
            "std_monetary_cost": self.std_monetary_cost,
            "q50_monetary_cost": self.q50_monetary_cost,
            "q90_monetary_cost": self.q90_monetary_cost,
            "q95_monetary_cost": self.q95_monetary_cost,
            "p95_monetary_cost": self.q95_monetary_cost,
            "cvar95_monetary_cost": self.cvar95_monetary_cost,
            "mean_emissions_kg": self.mean_emissions_kg,
            "std_emissions_kg": self.std_emissions_kg,
            "q50_emissions_kg": self.q50_emissions_kg,
            "q90_emissions_kg": self.q90_emissions_kg,
            "q95_emissions_kg": self.q95_emissions_kg,
            "p95_emissions_kg": self.q95_emissions_kg,
            "cvar95_emissions_kg": self.cvar95_emissions_kg,
            "utility_mean": self.utility_mean,
            "utility_q95": self.utility_q95,
            "utility_cvar95": self.utility_cvar95,
            "robust_score": self.robust_score,
            "sample_count_requested": self.sample_count_requested,
            "sample_count_used": self.sample_count_used,
            "sample_count_clip_ratio": self.sample_count_clip_ratio,
            "sigma_requested": self.sigma_requested,
            "sigma_used": self.sigma_used,
            "sigma_clip_ratio": self.sigma_clip_ratio,
            "factor_clip_rate": self.factor_clip_rate,
        }
        return {key: round(float(value), 6) for key, value in payload.items()}


def _bounded(value: float, *, low: float, high: float) -> float:
    return min(high, max(low, value))


def _cholesky_5x5(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    n = 5
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = matrix[i][j]
            for k in range(j):
                acc -= out[i][k] * out[j][k]
            if i == j:
                out[i][j] = math.sqrt(max(acc, 1e-9))
            else:
                out[i][j] = acc / max(out[j][j], 1e-9)
    return tuple(tuple(row) for row in out)


@lru_cache(maxsize=16)
def _cholesky_cached(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return _cholesky_5x5(matrix)


def _correlated_standard_normals(
    rng: random.Random,
    *,
    l_factor: tuple[tuple[float, ...], ...],
) -> tuple[float, float, float, float, float]:
    z = [rng.gauss(0.0, 1.0) for _ in range(5)]
    out = [0.0] * 5
    for i in range(5):
        total = 0.0
        for j in range(i + 1):
            total += l_factor[i][j] * z[j]
        out[i] = total
    return (out[0], out[1], out[2], out[3], out[4])


def _interp_quantile_mapping(points: tuple[tuple[float, float], ...], z_value: float) -> tuple[float, bool]:
    if not points:
        return 1.0, False
    z = float(z_value)
    if z <= points[0][0]:
        return float(points[0][1]), True
    if z >= points[-1][0]:
        return float(points[-1][1]), True
    for idx in range(1, len(points)):
        z0, v0 = points[idx - 1]
        z1, v1 = points[idx]
        if z <= z1:
            span = max(1e-9, float(z1 - z0))
            t = (z - float(z0)) / span
            return float(v0) + ((float(v1) - float(v0)) * t), False
    return float(points[-1][1]), True


def _dominant_road_bucket(road_class_counts: dict[str, int] | None) -> str:
    if not road_class_counts:
        return "mixed"
    total = max(sum(max(0, int(v)) for v in road_class_counts.values()), 1)
    motorway_share = max(0, int(road_class_counts.get("motorway", 0))) / total
    trunk_share = max(0, int(road_class_counts.get("trunk", 0))) / total
    if motorway_share >= 0.55:
        return "motorway_heavy"
    if trunk_share >= 0.45:
        return "trunk_heavy"
    return "mixed"


def resolve_stochastic_regime(
    departure_time_utc: datetime | None,
    *,
    road_class_counts: dict[str, int] | None = None,
    weather_profile: str | None = None,
    corridor_bucket: str | None = None,
    vehicle_bucket: str | None = None,
) -> tuple[str, str, str, str | None, StochasticRegime]:
    table = load_stochastic_regimes()
    road_bucket = _dominant_road_bucket(road_class_counts)
    weather_key = (weather_profile or "clear").strip().lower()
    corridor_key = _canonical_corridor_bucket(corridor_bucket)
    vehicle_key = _canonical_vehicle_bucket(vehicle_bucket)
    local_slot = _local_time_slot(departure_time_utc)
    local_day_kind = _local_day_kind(departure_time_utc)
    posterior_candidates = _posterior_regime_candidates(
        table=table,
        corridor_bucket=corridor_key,
        day_kind=local_day_kind,
        local_time_slot=local_slot,
        road_bucket=road_bucket,
        weather_profile=weather_key,
        vehicle_bucket=vehicle_key,
    )
    strict_posterior_required = bool(
        getattr(settings, "stochastic_require_empirical_calibration", False)
        and getattr(settings, "strict_live_data_required", False)
    )
    if strict_posterior_required and not isinstance(getattr(table, "posterior_model", None), dict):
        raise ModelDataError(
            reason_code="stochastic_calibration_unavailable",
            message="Stochastic posterior model is required for strict runtime regime resolution.",
        )
    if strict_posterior_required and not posterior_candidates:
        # Strict runtime still permits calibrated posterior backoff through the
        # neutral corridor bucket to avoid brittle geography-key mismatches.
        posterior_candidates = _posterior_regime_candidates(
            table=table,
            corridor_bucket="uk_default",
            day_kind=local_day_kind,
            local_time_slot=local_slot,
            road_bucket=road_bucket,
            weather_profile=weather_key,
            vehicle_bucket=vehicle_key,
        )
    if strict_posterior_required and not posterior_candidates:
        raise ModelDataError(
            reason_code="stochastic_calibration_unavailable",
            message=(
                "No posterior stochastic regime candidate matched strict context "
                f"(corridor={corridor_key}, day={local_day_kind}, slot={local_slot}, "
                f"road={road_bucket}, weather={weather_key}, vehicle={vehicle_key})."
            ),
        )
    for candidate in posterior_candidates:
        picked = table.regimes.get(candidate)
        if picked is not None:
            return candidate, table.copula_id, table.calibration_version, table.as_of_utc, picked
    if strict_posterior_required:
        raise ModelDataError(
            reason_code="stochastic_calibration_unavailable",
            message=(
                "Posterior stochastic regime candidates did not resolve to calibrated regimes "
                f"(corridor={corridor_key}, day={local_day_kind}, slot={local_slot}, "
                f"road={road_bucket}, weather={weather_key}, vehicle={vehicle_key})."
            ),
        )
    holiday_dates = load_uk_bank_holidays()
    if departure_time_utc is None:
        regime_id = "weekday_offpeak"
    else:
        aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
            tzinfo=timezone.utc
        )
        local = aware.astimezone(UK_TZ)
        minute = (local.hour * 60) + local.minute
        local_date = local.date().isoformat()
        is_weekend = local.weekday() >= 5
        in_peak = (420 <= minute <= 630) or (930 <= minute <= 1140)
        if local_date in holiday_dates:
            regime_id = "holiday"
        elif is_weekend:
            regime_id = "weekend"
        elif in_peak:
            regime_id = "weekday_peak"
        else:
            regime_id = "weekday_offpeak"
    candidates = [
        f"{corridor_key}_{regime_id}_{local_slot}_{road_bucket}_{weather_key}_{vehicle_key}",
        f"{corridor_key}_{regime_id}_{road_bucket}_{weather_key}_{vehicle_key}",
        f"{regime_id}_{local_slot}_{road_bucket}_{weather_key}_{vehicle_key}",
        f"{regime_id}_{road_bucket}_{weather_key}_{vehicle_key}",
        f"{corridor_key}_{regime_id}_{local_slot}_{road_bucket}_{weather_key}",
        f"{corridor_key}_{regime_id}_{road_bucket}_{weather_key}",
        f"{regime_id}_{local_slot}_{road_bucket}_{weather_key}",
        f"{regime_id}_{road_bucket}_{weather_key}",
        f"{regime_id}_{road_bucket}",
        f"{regime_id}_{weather_key}",
        regime_id,
    ]
    regime = None
    chosen_regime_id = regime_id
    for candidate in candidates:
        picked = table.regimes.get(candidate)
        if picked is not None:
            regime = picked
            chosen_regime_id = candidate
            break
    if regime is None:
        raise ModelDataError(
            reason_code="stochastic_calibration_unavailable",
            message=(
                "No calibrated stochastic regime matched route context "
                f"(base={regime_id}, road={road_bucket}, weather={weather_key})."
            ),
        )
    return chosen_regime_id, table.copula_id, table.calibration_version, table.as_of_utc, regime


def _factors_from_shocks(
    shocks: tuple[float, float, float, float, float],
    *,
    sigma: float,
    regime: StochasticRegime,
) -> tuple[float, float, float, int]:
    traffic_z, incident_z, weather_z, price_z, eco_z = shocks
    # Calibrated regime parameters are loaded from artifact tables.
    sigma_scaled = sigma * regime.sigma_scale

    clip_events = 0

    def _factor(name: str, z: float, scale: float) -> float:
        nonlocal clip_events
        spread = max(float(regime.spread_floor), sigma_scaled * max(0.1, scale))
        spread = min(float(regime.spread_cap), spread)
        raw = math.exp(spread * z)
        mapping_points = None
        if regime.shock_quantile_mapping is not None:
            mapping_points = regime.shock_quantile_mapping.get(name)
        if mapping_points:
            mapped, edge_clipped = _interp_quantile_mapping(mapping_points, z)
            mapped_scaled = 1.0 + ((float(mapped) - 1.0) * max(0.15, min(1.5, sigma_scaled)))
            raw = max(0.05, mapped_scaled)
            if edge_clipped:
                clip_events += 1
        low = max(0.25, float(regime.factor_low))
        high = max(low + 0.05, float(regime.factor_high))
        if raw < low or raw > high:
            clip_events += 1
        return _bounded(raw, low=low, high=high)

    traffic = _factor("traffic", traffic_z, regime.traffic_scale)
    incident = _factor("incident", incident_z, regime.incident_scale)
    weather = _factor("weather", weather_z, regime.weather_scale)
    price = _factor("price", price_z, regime.price_scale)
    eco = _factor("eco", eco_z, regime.eco_scale)

    traffic_w = max(0.1, regime.duration_mix[0] * regime.traffic_scale)
    incident_w = max(0.1, regime.duration_mix[1] * regime.incident_scale)
    weather_w = max(0.1, regime.duration_mix[2] * regime.weather_scale)
    w_sum = max(1e-9, traffic_w + incident_w + weather_w)
    duration_low = max(0.25, float(regime.factor_low))
    duration_high = max(float(regime.factor_high), float(regime.factor_low) + 0.05)
    duration_raw = ((traffic * traffic_w) + (incident * incident_w) + (weather * weather_w)) / w_sum
    if duration_raw < duration_low or duration_raw > duration_high:
        clip_events += 1
    duration_factor = _bounded(
        duration_raw,
        low=duration_low,
        high=duration_high,
    )
    m_dw = max(0.0, regime.monetary_mix[0])
    m_pw = max(0.0, regime.monetary_mix[1])
    m_sum = max(1e-9, m_dw + m_pw)
    monetary_low = max(0.25, float(regime.factor_low))
    monetary_high = max(float(regime.factor_high), float(regime.factor_low) + 0.05)
    monetary_raw = ((duration_factor * m_dw) + (price * m_pw)) / m_sum
    if monetary_raw < monetary_low or monetary_raw > monetary_high:
        clip_events += 1
    monetary_factor = _bounded(
        monetary_raw,
        low=monetary_low,
        high=monetary_high,
    )
    e_dw = max(0.0, regime.emissions_mix[0])
    e_ew = max(0.0, regime.emissions_mix[1])
    e_sum = max(1e-9, e_dw + e_ew)
    emissions_low = max(0.25, float(regime.factor_low))
    emissions_high = max(float(regime.factor_high), float(regime.factor_low) + 0.05)
    emissions_raw = ((duration_factor * e_dw) + (eco * e_ew)) / e_sum
    if emissions_raw < emissions_low or emissions_raw > emissions_high:
        clip_events += 1
    emissions_factor = _bounded(
        emissions_raw,
        low=emissions_low,
        high=emissions_high,
    )
    return duration_factor, monetary_factor, emissions_factor, clip_events


def compute_uncertainty_summary(
    *,
    base_duration_s: float,
    base_monetary_cost: float,
    base_emissions_kg: float,
    route_signature: str,
    departure_time_utc: datetime | None,
    user_seed: int | None,
    sigma: float,
    samples: int,
    cvar_alpha: float,
    utility_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    risk_aversion: float = 1.0,
    road_class_counts: dict[str, int] | None = None,
    weather_profile: str | None = None,
    base_distance_km: float | None = None,
    vehicle_type: str | None = None,
    vehicle_bucket: str | None = None,
    corridor_bucket: str | None = None,
) -> UncertaintySummary:
    requested_samples = max(1, int(samples))
    requested_sigma = max(0.0, float(sigma))
    sample_count = max(8, min(requested_samples, 600))
    sigma_clamped = _bounded(requested_sigma, low=0.0, high=0.6)
    seed = _stable_seed(
        route_signature=route_signature,
        departure_slot=_departure_slot(departure_time_utc),
        user_seed=0 if user_seed is None else int(user_seed),
    )
    rng = random.Random(seed)
    resolved_vehicle_bucket = _canonical_vehicle_bucket(vehicle_bucket)
    _regime_id, _copula_id, _calibration_version, _as_of_utc, regime = resolve_stochastic_regime(
        departure_time_utc,
        road_class_counts=road_class_counts,
        weather_profile=weather_profile,
        corridor_bucket=corridor_bucket,
        vehicle_bucket=resolved_vehicle_bucket,
    )
    l_factor = _cholesky_cached(regime.corr)

    half = max(1, sample_count // 2)
    duration_samples: list[float] = []
    monetary_samples: list[float] = []
    emissions_samples: list[float] = []
    factor_clip_events = 0
    max_factor_clip_events_per_sample = 8.0

    def _append_from_shocks(shocks: tuple[float, float, float, float, float]) -> None:
        nonlocal factor_clip_events
        duration_factor, monetary_factor, emissions_factor, clip_count = _factors_from_shocks(
            shocks,
            sigma=sigma_clamped,
            regime=regime,
        )
        factor_clip_events += int(max(0, clip_count))
        duration_samples.append(max(1e-6, base_duration_s * duration_factor))
        monetary_samples.append(max(0.0, base_monetary_cost * monetary_factor))
        emissions_samples.append(max(0.0, base_emissions_kg * emissions_factor))

    for _ in range(half):
        z = _correlated_standard_normals(rng, l_factor=l_factor)
        _append_from_shocks(z)
        antithetic = tuple(-v for v in z)
        _append_from_shocks(antithetic)

    while len(duration_samples) < sample_count:
        _append_from_shocks(_correlated_standard_normals(rng, l_factor=l_factor))

    distance_km = max(0.0, float(base_distance_km or 0.0))
    utility_samples: list[float] = []
    day_kind = _local_day_kind(departure_time_utc)
    local_slot = _local_time_slot(departure_time_utc)
    for dur, mon, emi in zip(duration_samples, monetary_samples, emissions_samples, strict=True):
        utility_samples.append(
            normalized_weighted_utility(
                duration_s=dur,
                monetary_cost=mon,
                emissions_kg=emi,
                distance_km=distance_km,
                utility_weights=utility_weights,
                vehicle_type=vehicle_type,
                vehicle_bucket=resolved_vehicle_bucket,
                corridor_bucket=corridor_bucket,
                day_kind=day_kind,
                local_time_slot=local_slot,
            )
        )

    q = lambda arr, p: quantile(arr, p)
    alpha = _bounded(cvar_alpha, low=0.5, high=0.999)
    utility_mean = statistics.fmean(utility_samples)
    utility_cvar = cvar(utility_samples, alpha=alpha)
    utility_robust = robust_objective(
        mean_value=utility_mean,
        cvar_value=utility_cvar,
        risk_aversion=risk_aversion,
        risk_family=settings.risk_family,
        risk_theta=settings.risk_family_theta,
    )
    q95_duration = q(duration_samples, 0.95)
    cvar95_duration = max(cvar(duration_samples, alpha=alpha), q95_duration)
    q95_money = q(monetary_samples, 0.95)
    cvar95_money = max(cvar(monetary_samples, alpha=alpha), q95_money)
    q95_emissions = q(emissions_samples, 0.95)
    cvar95_emissions = max(cvar(emissions_samples, alpha=alpha), q95_emissions)
    utility_q95 = q(utility_samples, 0.95)
    utility_cvar = max(utility_cvar, utility_q95)
    sample_clip_ratio = max(0.0, min(1.0, float(sample_count != requested_samples)))
    sigma_clip_ratio = max(0.0, min(1.0, float(abs(sigma_clamped - requested_sigma) > 1e-9)))
    factor_clip_rate = max(
        0.0,
        min(
            1.0,
            float(factor_clip_events) / max(1.0, float(len(duration_samples)) * max_factor_clip_events_per_sample),
        ),
    )
    objective_samples_full = [
        (
            float(dur),
            float(mon),
            float(emi),
            float(util),
        )
        for dur, mon, emi, util in zip(duration_samples, monetary_samples, emissions_samples, utility_samples, strict=True)
    ]
    sample_cap = max(16, int(getattr(settings, "risk_objective_sample_cap", 160)))
    if len(objective_samples_full) <= sample_cap:
        objective_samples = tuple(objective_samples_full)
    else:
        step = max(1, len(objective_samples_full) // sample_cap)
        sampled = objective_samples_full[::step][:sample_cap]
        if len(sampled) < sample_cap:
            sampled.extend(objective_samples_full[-(sample_cap - len(sampled)) :])
        objective_samples = tuple(sampled[:sample_cap])

    return UncertaintySummary(
        mean_duration_s=statistics.fmean(duration_samples),
        std_duration_s=(statistics.pstdev(duration_samples) if len(duration_samples) > 1 else 0.0),
        q50_duration_s=q(duration_samples, 0.50),
        q90_duration_s=q(duration_samples, 0.90),
        q95_duration_s=q95_duration,
        cvar95_duration_s=cvar95_duration,
        mean_monetary_cost=statistics.fmean(monetary_samples),
        std_monetary_cost=(statistics.pstdev(monetary_samples) if len(monetary_samples) > 1 else 0.0),
        q50_monetary_cost=q(monetary_samples, 0.50),
        q90_monetary_cost=q(monetary_samples, 0.90),
        q95_monetary_cost=q95_money,
        cvar95_monetary_cost=cvar95_money,
        mean_emissions_kg=statistics.fmean(emissions_samples),
        std_emissions_kg=(statistics.pstdev(emissions_samples) if len(emissions_samples) > 1 else 0.0),
        q50_emissions_kg=q(emissions_samples, 0.50),
        q90_emissions_kg=q(emissions_samples, 0.90),
        q95_emissions_kg=q95_emissions,
        cvar95_emissions_kg=cvar95_emissions,
        utility_mean=utility_mean,
        utility_q95=utility_q95,
        utility_cvar95=utility_cvar,
        robust_score=utility_robust,
        sample_count_requested=float(requested_samples),
        sample_count_used=float(sample_count),
        sample_count_clip_ratio=sample_clip_ratio,
        sigma_requested=float(requested_sigma),
        sigma_used=float(sigma_clamped),
        sigma_clip_ratio=sigma_clip_ratio,
        factor_clip_rate=factor_clip_rate,
        objective_samples=objective_samples,
    )
