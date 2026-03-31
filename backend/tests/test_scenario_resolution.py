from __future__ import annotations

from app.calibration_loader import (
    ScenarioContextProfile,
    ScenarioLiveContext,
    ScenarioPolicyProfile,
    ScenarioProfiles,
)
import app.calibration_loader as calibration_loader
from app.scenario import ScenarioMode, ScenarioRouteContext, resolve_scenario_profile


def _profile(*, base: float, spread: float) -> ScenarioPolicyProfile:
    q10 = base - spread
    q50 = base
    q90 = base + spread
    quantiles = {
        field: (q10, q50, q90)
        for field in (
            "duration_multiplier",
            "incident_rate_multiplier",
            "incident_delay_multiplier",
            "fuel_consumption_multiplier",
            "emissions_multiplier",
            "stochastic_sigma_multiplier",
        )
    }
    return ScenarioPolicyProfile(
        duration_multiplier=q50,
        incident_rate_multiplier=q50,
        incident_delay_multiplier=q50,
        fuel_consumption_multiplier=q50,
        emissions_multiplier=q50,
        stochastic_sigma_multiplier=q50,
        quantiles=quantiles,
    )


def _scenario_profiles(
    *,
    weekday_profile: ScenarioPolicyProfile,
    weekend_profile: ScenarioPolicyProfile,
) -> ScenarioProfiles:
    transform = calibration_loader._default_scenario_transform_params()
    transform["context_similarity"]["max_distance"] = 10.0
    weekday_key = "1c816|h08|weekday|mixed|rigid_hgv|clear"
    weekend_key = "1c816|h08|weekend|mixed|rigid_hgv|clear"
    mode_profiles = {"partial_sharing": weekday_profile}
    return ScenarioProfiles(
        source="repo_local:test_scenario_profiles.json",
        version="scenario_profiles_test_v1",
        as_of_utc="2026-03-30T00:00:00Z",
        generated_at_utc="2026-03-30T00:00:00Z",
        signature=None,
        calibration_basis="empirical_live_fit",
        profiles=mode_profiles,
        contexts={
            weekday_key: ScenarioContextProfile(
                context_key=weekday_key,
                corridor_bucket="wales_west",
                corridor_geohash5="1c816",
                hour_slot_local=8,
                road_mix_bucket="mixed",
                road_mix_vector={"mixed": 1.0},
                vehicle_class="rigid_hgv",
                day_kind="weekday",
                weather_bucket="clear",
                weather_regime="clear",
                profiles={"partial_sharing": weekday_profile},
                source_coverage={"webtris": 1.0, "traffic_england": 1.0, "dft": 1.0, "open_meteo": 1.0},
                mode_observation_source="observed_mode_labels",
                mode_projection_ratio=0.0,
            ),
            weekend_key: ScenarioContextProfile(
                context_key=weekend_key,
                corridor_bucket="wales_west",
                corridor_geohash5="1c816",
                hour_slot_local=8,
                road_mix_bucket="mixed",
                road_mix_vector={"mixed": 1.0},
                vehicle_class="rigid_hgv",
                day_kind="weekend",
                weather_bucket="clear",
                weather_regime="clear",
                profiles={"partial_sharing": weekend_profile},
                source_coverage={"webtris": 1.0, "traffic_england": 1.0, "dft": 1.0, "open_meteo": 1.0},
                mode_observation_source="observed_mode_labels",
                mode_projection_ratio=0.0,
            ),
        },
        transform_params=transform,
    )


def _live_context() -> ScenarioLiveContext:
    return ScenarioLiveContext(
        as_of_utc="2026-03-30T00:00:00Z",
        source_set={},
        coverage={},
        traffic_pressure=1.0,
        incident_pressure=1.0,
        weather_pressure=1.0,
        weather_bucket="clear",
        diagnostics={},
    )


def _request_context() -> ScenarioRouteContext:
    return ScenarioRouteContext(
        corridor_geohash5="1c816",
        hour_slot_local=8,
        day_kind="weekend",
        road_mix_bucket="mixed",
        road_mix_vector={"mixed": 1.0},
        vehicle_class="rigid_hgv",
        weather_regime="clear",
    )


def _approximate_request_context() -> ScenarioRouteContext:
    return ScenarioRouteContext(
        corridor_geohash5="1c816",
        hour_slot_local=8,
        day_kind="weekend",
        road_mix_bucket="motorway_heavy",
        road_mix_vector={"motorway": 0.7, "primary": 0.2, "local": 0.1},
        vehicle_class="rigid_hgv",
        weather_regime="clear",
    )


def test_resolve_scenario_profile_prefers_richer_paired_day_context_for_collapsed_exact(
    monkeypatch,
) -> None:
    weekday_profile = _profile(base=1.28, spread=0.06)
    weekend_profile = _profile(base=1.24, spread=0.0)
    profiles = _scenario_profiles(
        weekday_profile=weekday_profile,
        weekend_profile=weekend_profile,
    )
    monkeypatch.setattr(calibration_loader, "load_scenario_profiles", lambda: profiles)
    monkeypatch.setattr(calibration_loader, "load_live_scenario_context", lambda **_: _live_context())

    policy = resolve_scenario_profile(ScenarioMode.PARTIAL_SHARING, context=_request_context())

    assert policy.context_key == "1c816|h08|weekday|mixed|rigid_hgv|clear"
    assert policy.duration_multiplier == weekday_profile.duration_multiplier


def test_resolve_scenario_profile_prefers_richer_paired_day_context_for_collapsed_best_match(
    monkeypatch,
) -> None:
    weekday_profile = _profile(base=1.28, spread=0.06)
    weekend_profile = _profile(base=1.24, spread=0.0)
    profiles = _scenario_profiles(
        weekday_profile=weekday_profile,
        weekend_profile=weekend_profile,
    )
    monkeypatch.setattr(calibration_loader, "load_scenario_profiles", lambda: profiles)
    monkeypatch.setattr(calibration_loader, "load_live_scenario_context", lambda **_: _live_context())

    policy = resolve_scenario_profile(ScenarioMode.PARTIAL_SHARING, context=_approximate_request_context())

    assert policy.context_key == "1c816|h08|weekday|mixed|rigid_hgv|clear"
    assert policy.duration_multiplier == weekday_profile.quantiles["duration_multiplier"][2]
    assert policy.fuel_consumption_multiplier == weekday_profile.fuel_consumption_multiplier


def test_resolve_scenario_profile_keeps_exact_day_context_when_it_has_real_spread(
    monkeypatch,
) -> None:
    weekday_profile = _profile(base=1.28, spread=0.06)
    weekend_profile = _profile(base=1.24, spread=0.03)
    profiles = _scenario_profiles(
        weekday_profile=weekday_profile,
        weekend_profile=weekend_profile,
    )
    monkeypatch.setattr(calibration_loader, "load_scenario_profiles", lambda: profiles)
    monkeypatch.setattr(calibration_loader, "load_live_scenario_context", lambda **_: _live_context())

    policy = resolve_scenario_profile(ScenarioMode.PARTIAL_SHARING, context=_request_context())

    assert policy.context_key == "1c816|h08|weekend|mixed|rigid_hgv|clear"
    assert policy.duration_multiplier == weekend_profile.duration_multiplier
