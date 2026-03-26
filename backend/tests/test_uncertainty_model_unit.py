from __future__ import annotations

import math
import random
from datetime import UTC, datetime

import app.uncertainty_model as uncertainty_model
from app.calibration_loader import StochasticRegime


def _regime() -> StochasticRegime:
    return StochasticRegime(
        regime_id="weekday_offpeak",
        sigma_scale=1.0,
        traffic_scale=1.0,
        incident_scale=1.0,
        weather_scale=1.0,
        price_scale=1.0,
        eco_scale=1.0,
        corr=(
            (1.0, 0.2, 0.1, 0.1, 0.1),
            (0.2, 1.0, 0.2, 0.1, 0.1),
            (0.1, 0.2, 1.0, 0.1, 0.1),
            (0.1, 0.1, 0.1, 1.0, 0.2),
            (0.1, 0.1, 0.1, 0.2, 1.0),
        ),
        spread_floor=0.05,
        spread_cap=1.25,
        factor_low=0.55,
        factor_high=2.2,
        duration_mix=(1.0, 1.0, 1.0),
        monetary_mix=(0.62, 0.38),
        emissions_mix=(0.72, 0.28),
        transform_family="quantile_mapping_v1",
        shock_quantile_mapping={
            "traffic": ((-2.0, 0.75), (0.0, 1.0), (2.0, 1.35)),
            "incident": ((-2.0, 0.72), (0.0, 1.0), (2.0, 1.40)),
            "weather": ((-2.0, 0.78), (0.0, 1.0), (2.0, 1.30)),
            "price": ((-2.0, 0.82), (0.0, 1.0), (2.0, 1.25)),
            "eco": ((-2.0, 0.80), (0.0, 1.0), (2.0, 1.22)),
        },
    )


def test_stable_seed_is_deterministic_and_slot_sensitive() -> None:
    same_a = uncertainty_model._stable_seed(
        route_signature="route_a",
        departure_slot="2026-02-23T09:00+0000",
        user_seed=7,
    )
    same_b = uncertainty_model._stable_seed(
        route_signature="route_a",
        departure_slot="2026-02-23T09:00+0000",
        user_seed=7,
    )
    changed = uncertainty_model._stable_seed(
        route_signature="route_a",
        departure_slot="2026-02-23T09:15+0000",
        user_seed=7,
    )

    assert same_a == same_b
    assert changed != same_a


def test_quantile_mapping_interpolates_and_clips() -> None:
    points = ((-2.0, 0.8), (0.0, 1.0), (2.0, 1.4))
    lo, lo_clipped = uncertainty_model._interp_quantile_mapping(points, -3.0)
    mid, mid_clipped = uncertainty_model._interp_quantile_mapping(points, 1.0)
    hi, hi_clipped = uncertainty_model._interp_quantile_mapping(points, 3.0)

    assert lo == 0.8
    assert lo_clipped is True
    assert 1.0 < mid < 1.4
    assert mid_clipped is False
    assert hi == 1.4
    assert hi_clipped is True


def test_cholesky_factor_reconstructs_correlation_matrix() -> None:
    corr = _regime().corr
    factor = uncertainty_model._cholesky_5x5(corr)

    reconstructed = []
    for i in range(5):
        row = []
        for j in range(5):
            row.append(sum(factor[i][k] * factor[j][k] for k in range(5)))
        reconstructed.append(tuple(row))

    for i in range(5):
        for j in range(5):
            assert math.isclose(reconstructed[i][j], corr[i][j], rel_tol=0.0, abs_tol=1e-8)


def test_correlated_standard_normals_are_deterministic_for_seeded_rng() -> None:
    factor = uncertainty_model._cholesky_cached(_regime().corr)
    first = uncertainty_model._correlated_standard_normals(random.Random(11), l_factor=factor)
    second = uncertainty_model._correlated_standard_normals(random.Random(11), l_factor=factor)

    assert first == second
    assert len(first) == 5


def test_antithetic_pairing_stabilizes_linearized_sample_means(monkeypatch) -> None:
    monkeypatch.setattr(
        uncertainty_model,
        "resolve_stochastic_regime",
        lambda *_args, **_kwargs: ("weekday_offpeak", "gaussian_5x5_v2", "unit", "2026-02-23T00:00:00Z", _regime()),
    )
    monkeypatch.setattr(uncertainty_model, "load_uk_bank_holidays", lambda: frozenset())
    monkeypatch.setattr(
        uncertainty_model,
        "normalized_weighted_utility",
        lambda **kwargs: float(kwargs["duration_s"]) + float(kwargs["monetary_cost"]) + float(kwargs["emissions_kg"]),
    )
    monkeypatch.setattr(
        uncertainty_model,
        "_correlated_standard_normals",
        lambda *_args, **_kwargs: (1.0, 1.0, 1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(
        uncertainty_model,
        "_factors_from_shocks",
        lambda shocks, **_kwargs: (1.0 + (0.1 * shocks[0]), 1.0 + (0.05 * shocks[1]), 1.0 + (0.02 * shocks[2]), 0),
    )

    summary = uncertainty_model.compute_uncertainty_summary(
        base_duration_s=100.0,
        base_monetary_cost=50.0,
        base_emissions_kg=20.0,
        route_signature="unit_route",
        departure_time_utc=datetime(2026, 2, 23, 9, 30, tzinfo=UTC),
        user_seed=1,
        sigma=0.12,
        samples=8,
        cvar_alpha=0.95,
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
        base_distance_km=100.0,
        vehicle_type="rigid_hgv",
        vehicle_bucket="rigid_hgv",
        corridor_bucket="uk_default",
    )

    assert math.isclose(summary.mean_duration_s, 100.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(summary.mean_monetary_cost, 50.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(summary.mean_emissions_kg, 20.0, rel_tol=0.0, abs_tol=1e-9)


def test_compute_uncertainty_summary_with_stub_regime(monkeypatch) -> None:
    monkeypatch.setattr(
        uncertainty_model,
        "resolve_stochastic_regime",
        lambda *_args, **_kwargs: (
            "weekday_offpeak",
            "gaussian_5x5_v2",
            "stochastic_live_test_v1",
            "2026-02-23T00:00:00Z",
            _regime(),
        ),
    )
    monkeypatch.setattr(uncertainty_model, "load_uk_bank_holidays", lambda: frozenset())
    monkeypatch.setattr(
        uncertainty_model,
        "normalized_weighted_utility",
        lambda **kwargs: float(kwargs["duration_s"]) + float(kwargs["monetary_cost"]) + float(kwargs["emissions_kg"]),
    )

    summary = uncertainty_model.compute_uncertainty_summary(
        base_duration_s=3200.0,
        base_monetary_cost=140.0,
        base_emissions_kg=62.0,
        route_signature="unit_route",
        departure_time_utc=datetime(2026, 2, 23, 9, 30, tzinfo=UTC),
        user_seed=42,
        sigma=0.12,
        samples=48,
        cvar_alpha=0.95,
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
        base_distance_km=120.0,
        vehicle_type="rigid_hgv",
        vehicle_bucket="rigid_hgv",
        corridor_bucket="uk_default",
    )

    payload = summary.as_dict()
    assert payload["q50_duration_s"] <= payload["q90_duration_s"] <= payload["q95_duration_s"]
    assert payload["q50_monetary_cost"] <= payload["q90_monetary_cost"] <= payload["q95_monetary_cost"]
    assert payload["q50_emissions_kg"] <= payload["q90_emissions_kg"] <= payload["q95_emissions_kg"]
    assert payload["sample_count_used"] >= 8
    assert len(summary.objective_samples) > 0


def test_uncertainty_clip_ratios_are_true_relative_clip_amounts(monkeypatch) -> None:
    monkeypatch.setattr(
        uncertainty_model,
        "resolve_stochastic_regime",
        lambda *_args, **_kwargs: (
            "weekday_offpeak",
            "gaussian_5x5_v2",
            "stochastic_live_test_v1",
            "2026-02-23T00:00:00Z",
            _regime(),
        ),
    )
    monkeypatch.setattr(uncertainty_model, "load_uk_bank_holidays", lambda: frozenset())
    monkeypatch.setattr(
        uncertainty_model,
        "normalized_weighted_utility",
        lambda **kwargs: float(kwargs["duration_s"]) + float(kwargs["monetary_cost"]) + float(kwargs["emissions_kg"]),
    )

    summary = uncertainty_model.compute_uncertainty_summary(
        base_duration_s=3200.0,
        base_monetary_cost=140.0,
        base_emissions_kg=62.0,
        route_signature="unit_route",
        departure_time_utc=datetime(2026, 2, 23, 9, 30, tzinfo=UTC),
        user_seed=42,
        sigma=0.8,
        samples=1000,
        cvar_alpha=0.95,
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
        base_distance_km=120.0,
        vehicle_type="rigid_hgv",
        vehicle_bucket="rigid_hgv",
        corridor_bucket="uk_default",
    )

    assert math.isclose(summary.sample_count_clip_ratio, 0.4, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(summary.sigma_clip_ratio, 0.25, rel_tol=0.0, abs_tol=1e-9)
