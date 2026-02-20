from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .calibration_loader import StochasticRegime, load_stochastic_regimes, load_uk_bank_holidays
from .model_data_errors import ModelDataError
from .risk_model import cvar, normalized_weighted_utility, quantile

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

    def as_dict(self) -> dict[str, float]:
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
) -> tuple[str, str, str, str | None, StochasticRegime]:
    table = load_stochastic_regimes()
    road_bucket = _dominant_road_bucket(road_class_counts)
    weather_key = (weather_profile or "clear").strip().lower()
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
        f"{_canonical_corridor_bucket(corridor_bucket)}_{regime_id}_{_local_time_slot(departure_time_utc)}_{road_bucket}_{weather_key}",
        f"{_canonical_corridor_bucket(corridor_bucket)}_{regime_id}_{road_bucket}_{weather_key}",
        f"{regime_id}_{_local_time_slot(departure_time_utc)}_{road_bucket}_{weather_key}",
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
) -> tuple[float, float, float]:
    traffic_z, incident_z, weather_z, price_z, eco_z = shocks
    # Calibrated regime parameters are loaded from artifact tables.
    sigma_scaled = sigma * regime.sigma_scale

    def _factor(z: float, scale: float) -> float:
        spread = max(float(regime.spread_floor), sigma_scaled * max(0.1, scale))
        spread = min(float(regime.spread_cap), spread)
        raw = math.exp(spread * z)
        low = max(0.25, float(regime.factor_low))
        high = max(low + 0.05, float(regime.factor_high))
        return _bounded(raw, low=low, high=high)

    traffic = _factor(traffic_z, regime.traffic_scale)
    incident = _factor(incident_z, regime.incident_scale)
    weather = _factor(weather_z, regime.weather_scale)
    price = _factor(price_z, regime.price_scale)
    eco = _factor(eco_z, regime.eco_scale)

    traffic_w = max(0.1, regime.duration_mix[0] * regime.traffic_scale)
    incident_w = max(0.1, regime.duration_mix[1] * regime.incident_scale)
    weather_w = max(0.1, regime.duration_mix[2] * regime.weather_scale)
    w_sum = max(1e-9, traffic_w + incident_w + weather_w)
    duration_factor = _bounded(
        ((traffic * traffic_w) + (incident * incident_w) + (weather * weather_w)) / w_sum,
        low=max(0.25, float(regime.factor_low)),
        high=max(float(regime.factor_high), float(regime.factor_low) + 0.05),
    )
    m_dw = max(0.0, regime.monetary_mix[0])
    m_pw = max(0.0, regime.monetary_mix[1])
    m_sum = max(1e-9, m_dw + m_pw)
    monetary_factor = _bounded(
        ((duration_factor * m_dw) + (price * m_pw)) / m_sum,
        low=max(0.25, float(regime.factor_low)),
        high=max(float(regime.factor_high), float(regime.factor_low) + 0.05),
    )
    e_dw = max(0.0, regime.emissions_mix[0])
    e_ew = max(0.0, regime.emissions_mix[1])
    e_sum = max(1e-9, e_dw + e_ew)
    emissions_factor = _bounded(
        ((duration_factor * e_dw) + (eco * e_ew)) / e_sum,
        low=max(0.25, float(regime.factor_low)),
        high=max(float(regime.factor_high), float(regime.factor_low) + 0.05),
    )
    return duration_factor, monetary_factor, emissions_factor


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
    corridor_bucket: str | None = None,
) -> UncertaintySummary:
    sample_count = max(8, min(int(samples), 600))
    sigma_clamped = _bounded(float(sigma), low=0.0, high=0.6)
    seed = _stable_seed(
        route_signature=route_signature,
        departure_slot=_departure_slot(departure_time_utc),
        user_seed=0 if user_seed is None else int(user_seed),
    )
    rng = random.Random(seed)
    _regime_id, _copula_id, _calibration_version, _as_of_utc, regime = resolve_stochastic_regime(
        departure_time_utc,
        road_class_counts=road_class_counts,
        weather_profile=weather_profile,
        corridor_bucket=corridor_bucket,
    )
    l_factor = _cholesky_cached(regime.corr)

    half = max(1, sample_count // 2)
    duration_samples: list[float] = []
    monetary_samples: list[float] = []
    emissions_samples: list[float] = []

    def _append_from_shocks(shocks: tuple[float, float, float, float, float]) -> None:
        duration_factor, monetary_factor, emissions_factor = _factors_from_shocks(
            shocks,
            sigma=sigma_clamped,
            regime=regime,
        )
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
                corridor_bucket=corridor_bucket,
                day_kind=day_kind,
                local_time_slot=local_slot,
            )
        )

    q = lambda arr, p: quantile(arr, p)
    alpha = _bounded(cvar_alpha, low=0.5, high=0.999)
    utility_mean = statistics.fmean(utility_samples)
    utility_cvar = cvar(utility_samples, alpha=alpha)
    utility_robust = utility_mean + (max(0.0, float(risk_aversion)) * max(0.0, utility_cvar - utility_mean))
    q95_duration = q(duration_samples, 0.95)
    cvar95_duration = max(cvar(duration_samples, alpha=alpha), q95_duration)
    q95_money = q(monetary_samples, 0.95)
    cvar95_money = max(cvar(monetary_samples, alpha=alpha), q95_money)
    q95_emissions = q(emissions_samples, 0.95)
    cvar95_emissions = max(cvar(emissions_samples, alpha=alpha), q95_emissions)
    utility_q95 = q(utility_samples, 0.95)
    utility_cvar = max(utility_cvar, utility_q95)

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
    )
