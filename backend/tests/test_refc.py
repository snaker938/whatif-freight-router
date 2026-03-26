from __future__ import annotations

import json

import pytest

import app.evidence_certification as evidence_certification_module
import app.main as main_module
from app.models import (
    EvidenceProvenance,
    EvidenceSourceRecord,
    GeoJSONLineString,
    RouteMetrics,
    RouteOption,
    ScenarioSummary,
    TerrainSummaryPayload,
)
from app.evidence_certification import (
    EVIDENCE_FAMILIES,
    active_evidence_families,
    compute_certificate,
    compute_fragility_maps,
    dependency_tensor,
    evaluate_world_bundle,
    _route_perturbed_objectives,
    rank_value_of_refresh,
    sample_world_manifest,
    refc_requires_full_stress_worlds,
    validate_route_evidence_provenance,
)

pytestmark = pytest.mark.thesis_modules


def _route(
    route_id: str,
    *,
    objective: tuple[float, float, float],
    evidence_tensor: dict[str, dict[str, float]] | None = None,
    snapshot_payloads: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "route_id": route_id,
        "objective_vector": objective,
        "evidence_tensor": evidence_tensor or {},
        "evidence_provenance": {
            "active_families": list((evidence_tensor or {}).keys()),
            "families": [
                {
                    "family": family,
                    "source": f"{family}_asset",
                    "signature": f"sig-{family}",
                    "snapshot_id": f"snap-{family}",
                    "confidence": 0.9,
                    "coverage_ratio": 1.0,
                    "fallback_used": False,
                    "snapshot_payload": (snapshot_payloads or {}).get(family, {"version": 1, "family": family}),
                }
                for family in (evidence_tensor or {})
            ],
            "dependency_weights": {
                family: {"time": 1.0, "money": 1.0, "co2": 1.0}
                for family in (evidence_tensor or {})
            },
        },
    }


def _route_option(
    route_id: str,
    *,
    duration_s: float,
    money_cost: float,
    co2_kg: float,
    distance_km: float = 100.0,
    toll_cost: float = 0.0,
    fuel_cost: float = 0.0,
    carbon_cost: float = 0.0,
    weather_delay_s: float = 0.0,
    ascent_m: float = 0.0,
    descent_m: float = 0.0,
    scenario_duration_multiplier: float = 1.0,
    scenario_fuel_multiplier: float = 1.0,
    scenario_emissions_multiplier: float = 1.0,
    scenario_sigma_multiplier: float = 1.0,
    std_duration_s: float = 0.0,
    std_money_cost: float = 0.0,
    std_co2_kg: float = 0.0,
    active_families: tuple[str, ...] | None = None,
) -> RouteOption:
    families = active_families or ("scenario", "toll", "terrain", "fuel", "carbon", "weather", "stochastic")
    provenance = EvidenceProvenance(
        active_families=list(families),
        families=[
            EvidenceSourceRecord(
                family=family,
                source=f"{family}_asset",
                active=True,
                confidence=0.95,
                coverage_ratio=1.0,
                fallback_used=False,
            )
            for family in families
        ],
    )
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[(-1.0, 52.0), (-0.5, 52.2)],
        ),
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_s=duration_s,
            monetary_cost=money_cost,
            emissions_kg=co2_kg,
            avg_speed_kmh=max(1.0, distance_km / max(duration_s / 3600.0, 1e-6)),
        ),
        segment_breakdown=[
            {
                "segment_index": 0,
                "toll_cost": toll_cost,
                "fuel_cost": fuel_cost,
                "carbon_cost": carbon_cost,
            }
        ],
        weather_summary={"weather_delay_s": weather_delay_s},
        scenario_summary=ScenarioSummary(
            mode="no_sharing",
            duration_multiplier=scenario_duration_multiplier,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=scenario_fuel_multiplier,
            emissions_multiplier=scenario_emissions_multiplier,
            stochastic_sigma_multiplier=scenario_sigma_multiplier,
            source="test",
            version="test",
        ),
        terrain_summary=TerrainSummaryPayload(
            source="dem_real",
            coverage_ratio=1.0,
            sample_spacing_m=75.0,
            ascent_m=ascent_m,
            descent_m=descent_m,
            confidence=0.95,
            version="test",
        ),
        uncertainty={
            "std_duration_s": std_duration_s,
            "std_monetary_cost": std_money_cost,
            "std_emissions_kg": std_co2_kg,
        },
        evidence_provenance=provenance,
    )


def test_sampled_world_manifest_is_seed_replayable() -> None:
    manifest_a = sample_world_manifest(
        active_families=["scenario", "toll", "weather"],
        seed=17,
        world_count=5,
    )
    manifest_b = sample_world_manifest(
        active_families=["scenario", "toll", "weather"],
        seed=17,
        world_count=5,
    )

    assert manifest_a == manifest_b
    assert manifest_a["active_families"] == ["scenario", "toll", "weather"]
    assert all(set(world["states"]) <= {"scenario", "toll", "weather"} for world in manifest_a["worlds"])


def test_repo_local_fresh_provenance_biases_world_sampling_away_from_proxy_states() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        )
    ]
    manifest = sample_world_manifest(
        active_families=["scenario"],
        seed=17,
        world_count=16,
        routes=routes,
    )

    weights = manifest["state_weights"]["scenario"]
    assert weights["nominal"] > weights["proxy"]
    assert weights["refreshed"] > weights["proxy"]


def test_refc_bypass_helper_matches_refc_classification_thresholds() -> None:
    supported_context = {
        "od_ambiguity_index": 0.62,
        "od_ambiguity_confidence": 0.83,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1}',
        "od_ambiguity_source_entropy": 0.58,
        "od_ambiguity_support_ratio": 0.64,
        "od_ambiguity_prior_strength": 0.66,
        "od_ambiguity_family_density": 0.40,
        "od_ambiguity_margin_pressure": 0.38,
        "od_ambiguity_spread_pressure": 0.24,
        "ambiguity_budget_prior": 0.64,
        "od_candidate_path_count": 3,
        "od_corridor_family_count": 2,
        "od_objective_spread": 0.30,
        "od_nominal_margin_proxy": 0.22,
        "od_toll_disagreement_rate": 0.08,
        "od_engine_disagreement_prior": 0.25,
        "od_hard_case_prior": 0.32,
        "ambiguity_budget_band": "medium",
    }
    weak_context = {
        "od_ambiguity_index": 0.18,
        "od_ambiguity_confidence": 0.12,
        "od_ambiguity_source_count": 1,
        "od_ambiguity_source_mix": "routing_graph_probe",
        "od_ambiguity_source_entropy": 0.18,
        "od_ambiguity_support_ratio": 0.27,
        "od_ambiguity_prior_strength": 0.14,
        "od_ambiguity_family_density": 0.12,
        "od_ambiguity_margin_pressure": 0.09,
        "od_ambiguity_spread_pressure": 0.06,
        "ambiguity_budget_prior": 0.08,
        "od_candidate_path_count": 1,
        "od_corridor_family_count": 1,
        "od_objective_spread": 0.02,
        "od_nominal_margin_proxy": 0.62,
        "od_toll_disagreement_rate": 0.02,
        "od_engine_disagreement_prior": 0.08,
        "od_hard_case_prior": 0.10,
        "ambiguity_budget_band": "low",
    }

    assert main_module._refc_requires_full_stress_worlds(supported_context) is True
    assert refc_requires_full_stress_worlds(supported_context) is True
    assert main_module._refc_requires_full_stress_worlds(weak_context) is False
    assert refc_requires_full_stress_worlds(weak_context) is False


def test_refc_rejects_zero_world_requests_instead_of_fabricating_output() -> None:
    try:
        sample_world_manifest(active_families=["scenario"], seed=17, world_count=0)
    except ValueError as exc:
        assert "world_count" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("zero world_count should be rejected")


def test_evidence_validation_rejects_bootstrap_and_fixture_markers() -> None:
    validation = validate_route_evidence_provenance(
        {
            "active_families": ["scenario", "toll"],
            "families": [
                {
                    "family": "scenario",
                    "source": "live_runtime",
                    "fallback_used": False,
                    "freshness_timestamp_utc": "2026-03-01T00:00:00Z",
                    "details": {"mode_observation_source": "empirical_outcome_bootstrap"},
                },
                {
                    "family": "toll",
                    "source": "live_runtime",
                    "fallback_used": False,
                    "freshness_timestamp_utc": "2026-03-01T00:00:00Z",
                    "details": {"matched_asset_ids": "fixture_001"},
                },
            ],
        }
    )

    assert validation.status == "rejected"
    assert len(validation.issues) == 2


def test_dependency_tensor_is_bounded_and_normalised() -> None:
    route = _route(
        "route_a",
        objective=(10.0, 10.0, 10.0),
        evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
    )
    tensor = dependency_tensor(route, active_families=["scenario"])

    assert set(tensor) == {"time", "money", "co2"}
    assert tensor["time"]["scenario"] == 1.0
    assert tensor["money"]["scenario"] == 1.0
    assert tensor["co2"]["scenario"] == 1.0
    assert active_evidence_families([route]) == ["scenario"]


def test_dependency_tensor_does_not_assign_bias_to_inactive_route_families() -> None:
    route = _route(
        "route_b",
        objective=(10.0, 10.0, 10.0),
        evidence_tensor={},
    )
    tensor = dependency_tensor(route, active_families=["scenario", "weather"])

    assert tensor["time"]["scenario"] == 0.0
    assert tensor["time"]["weather"] == 0.0
    assert tensor["money"]["scenario"] == 0.0
    assert tensor["co2"]["weather"] == 0.0


def test_certificate_and_fragility_outputs_are_hand_checkable() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 1.0, "co2": 1.0},
                "weather": {"time": 0.0, "money": 0.0, "co2": 0.0},
            },
        ),
        _route(
            "route_b",
            objective=(10.05, 10.05, 10.05),
            evidence_tensor={},
        ),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal"}},
        {"world_id": "w1", "states": {"scenario": "severely_stale"}},
        {"world_id": "w2", "states": {"scenario": "refreshed"}},
    ]

    certificate = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.60,
        active_families=["scenario", "weather"],
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather"],
        selected_route_id="route_a",
    )
    vor = rank_value_of_refresh(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather"],
        selected_route_id="route_a",
    )

    assert certificate.winner_id == "route_a"
    assert certificate.certified is True
    assert certificate.certificate["route_a"] == 2 / 3
    assert certificate.certificate["route_b"] == 1 / 3
    assert fragility.route_fragility_map["route_a"]["scenario"] > 0.0
    assert fragility.route_fragility_map["route_a"]["weather"] == 0.0
    assert fragility.route_fragility_details["route_a"]["scenario"]["normalized_drop"] > 0.0
    assert fragility.competitor_fragility_breakdown["route_a"]["route_b"]["scenario"] == len(worlds)
    assert fragility.value_of_refresh["top_refresh_family"] == "scenario"
    assert fragility.value_of_refresh["fragility_stress_state"] == "severely_stale"
    assert fragility.evidence_snapshot_manifest["family_snapshots"]["scenario"][0]["source"] == "scenario_asset"
    assert fragility.evidence_snapshot_manifest["family_snapshots"]["scenario"][0]["snapshot_payload_hash"]
    assert vor["top_refresh_family"] == "scenario"
    assert vor["ranking"][0]["family"] == "scenario"


def test_fragility_is_family_isolated_not_joint_noise() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 0.0, "co2": 0.0},
                "weather": {"time": 0.0, "money": 0.0, "co2": 0.0},
            },
        ),
        _route(
            "route_b",
            objective=(10.2, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 0.0, "money": 0.0, "co2": 0.0},
                "weather": {"time": 1.0, "money": 0.0, "co2": 0.0},
            },
        ),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal"}},
        {"world_id": "w1", "states": {"scenario": "nominal"}},
    ]

    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 0.0, 0.0),
        active_families=["scenario", "weather"],
        selected_route_id="route_a",
    )

    assert fragility.route_fragility_map["route_a"]["scenario"] > 0.0
    assert fragility.competitor_fragility_breakdown["route_a"]["route_b"]["scenario"] == len(worlds)


def test_mixed_targeted_worlds_do_not_change_family_attribution() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 1.0, "co2": 1.0},
                "toll": {"time": 0.5, "money": 0.5, "co2": 0.5},
            },
        ),
        _route(
            "route_b",
            objective=(10.3, 10.1, 10.2),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 1.0, "co2": 1.0},
                "toll": {"time": 0.5, "money": 0.5, "co2": 0.5},
            },
        ),
    ]
    isolated_worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal", "toll": "nominal"}, "world_kind": "sampled"},
        {"world_id": "w1", "states": {"scenario": "severely_stale", "toll": "nominal"}, "world_kind": "hard_case_targeted"},
        {"world_id": "w2", "states": {"scenario": "nominal", "toll": "severely_stale"}, "world_kind": "hard_case_targeted"},
    ]
    mixed_world = {
        "world_id": "w3",
        "states": {"scenario": "severely_stale", "toll": "severely_stale"},
        "world_kind": "hard_case_mixed_targeted",
    }

    isolated = compute_fragility_maps(
        routes,
        worlds=isolated_worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "toll"],
        selected_route_id="route_a",
    )
    with_mixed = compute_fragility_maps(
        routes,
        worlds=isolated_worlds + [mixed_world],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "toll"],
        selected_route_id="route_a",
    )

    assert with_mixed.route_fragility_map == isolated.route_fragility_map
    assert with_mixed.competitor_fragility_breakdown == isolated.competitor_fragility_breakdown


def test_fragility_value_of_refresh_uses_full_targeted_population_when_available() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 1.0, "co2": 1.0},
                "toll": {"time": 0.5, "money": 0.5, "co2": 0.5},
            },
        ),
        _route(
            "route_b",
            objective=(10.28, 10.16, 10.24),
            evidence_tensor={
                "scenario": {"time": 0.0, "money": 0.0, "co2": 0.0},
                "toll": {"time": 0.0, "money": 0.0, "co2": 0.0},
            },
        ),
    ]
    analysis_worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal", "toll": "nominal"}, "world_kind": "sampled"},
        {"world_id": "w1", "states": {"scenario": "nominal", "toll": "nominal"}, "world_kind": "sampled"},
    ]
    mixed_world = {
        "world_id": "w2",
        "states": {"scenario": "severely_stale", "toll": "severely_stale"},
        "world_kind": "hard_case_mixed_targeted",
    }
    full_worlds = analysis_worlds + [mixed_world]

    analysis_certificate = compute_certificate(
        routes,
        worlds=analysis_worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "toll"],
    )
    full_certificate = compute_certificate(
        routes,
        worlds=full_worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "toll"],
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=full_worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "toll"],
        selected_route_id="route_a",
        baseline_certificate=full_certificate,
    )

    assert full_certificate.certificate["route_a"] != analysis_certificate.certificate["route_a"]
    assert fragility.value_of_refresh["baseline_certificate"] == full_certificate.certificate["route_a"]
    assert fragility.route_fragility_details["route_a"]["scenario"]["baseline_certificate"] == analysis_certificate.certificate["route_a"]
    assert fragility.evidence_snapshot_manifest["baseline_unique_world_count"] == analysis_certificate.world_manifest["unique_world_count"]


def test_certificate_degradation_is_monotone_with_stress_severity() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        ),
        _route("route_b", objective=(10.15, 10.15, 10.15), evidence_tensor={}),
    ]
    nominal = compute_certificate(
        routes,
        worlds=[{"world_id": "w0", "states": {"scenario": "nominal"}}],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
    )
    mild = compute_certificate(
        routes,
        worlds=[{"world_id": "w1", "states": {"scenario": "mildly_stale"}}],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
    )
    severe = compute_certificate(
        routes,
        worlds=[{"world_id": "w2", "states": {"scenario": "severely_stale"}}],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
    )

    assert nominal.certificate["route_a"] >= mild.certificate["route_a"] >= severe.certificate["route_a"]


def test_hard_row_manifest_adds_bounded_stress_pack_and_weakens_certificate() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 0.8, "co2": 0.8},
                "weather": {"time": 0.9, "money": 0.1, "co2": 0.1},
                "stochastic": {"time": 0.8, "money": 0.8, "co2": 0.5},
                "toll": {"time": 0.0, "money": 1.0, "co2": 0.0},
            },
        ),
        _route(
            "route_b",
            objective=(10.04, 10.01, 10.02),
            evidence_tensor={},
        ),
    ]
    easy_manifest = sample_world_manifest(
        active_families=["scenario", "weather", "stochastic", "toll"],
        seed=17,
        world_count=8,
        routes=routes,
        ambiguity_context={
            "od_ambiguity_index": 0.12,
            "od_hard_case_prior": 0.10,
            "od_engine_disagreement_prior": 0.08,
            "od_candidate_path_count": 1,
            "od_corridor_family_count": 1,
            "ambiguity_budget_band": "low",
        },
    )
    hard_manifest = sample_world_manifest(
        active_families=["scenario", "weather", "stochastic", "toll"],
        seed=17,
        world_count=8,
        routes=routes,
        ambiguity_context={
            "od_ambiguity_index": 0.78,
            "od_hard_case_prior": 0.82,
            "od_engine_disagreement_prior": 0.66,
            "od_candidate_path_count": 5,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.34,
            "od_nominal_margin_proxy": 0.12,
            "od_toll_disagreement_rate": 0.58,
            "ambiguity_budget_band": "high",
        },
    )

    easy_certificate = compute_certificate(
        routes,
        worlds=easy_manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "stochastic", "toll"],
        ambiguity_context=easy_manifest["ambiguity_context"],
    )
    hard_certificate = compute_certificate(
        routes,
        worlds=hard_manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "stochastic", "toll"],
        ambiguity_context=hard_manifest["ambiguity_context"],
    )

    assert hard_manifest["hard_case_stress_pack_count"] > 0
    assert hard_manifest["world_count"] >= hard_manifest["unique_world_count"]
    assert hard_manifest["stress_world_fraction"] > 0.0
    assert hard_manifest["world_count"] >= easy_manifest["world_count"]
    assert hard_certificate.certificate["route_a"] < easy_certificate.certificate["route_a"]


def test_supported_ambiguity_prior_amplifies_hard_case_stress_and_weakens_certificate() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 0.8, "co2": 0.7},
                "weather": {"time": 0.8, "money": 0.1, "co2": 0.1},
                "stochastic": {"time": 0.9, "money": 0.7, "co2": 0.6},
            },
        ),
        _route("route_b", objective=(10.03, 10.02, 10.04), evidence_tensor={}),
    ]
    low_support_manifest = sample_world_manifest(
        active_families=["scenario", "weather", "stochastic"],
        seed=33,
        world_count=10,
        routes=routes,
        ambiguity_context={
            "od_ambiguity_index": 0.72,
            "od_hard_case_prior": 0.78,
            "od_engine_disagreement_prior": 0.61,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.30,
            "od_nominal_margin_proxy": 0.10,
            "ambiguity_budget_band": "high",
            "od_ambiguity_confidence": 0.20,
            "od_ambiguity_source_count": 1,
        },
    )
    high_support_manifest = sample_world_manifest(
        active_families=["scenario", "weather", "stochastic"],
        seed=33,
        world_count=10,
        routes=routes,
        ambiguity_context={
            "od_ambiguity_index": 0.72,
            "od_hard_case_prior": 0.78,
            "od_engine_disagreement_prior": 0.61,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.30,
            "od_nominal_margin_proxy": 0.10,
            "ambiguity_budget_band": "high",
            "od_ambiguity_confidence": 0.95,
            "od_ambiguity_source_count": 4,
            "od_ambiguity_source_mix": "historical_results_bootstrap,repo_local_geometry_backfill,baseline_disagreement",
        },
    )

    low_support_certificate = compute_certificate(
        routes,
        worlds=low_support_manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "stochastic"],
        ambiguity_context=low_support_manifest["ambiguity_context"],
    )
    high_support_certificate = compute_certificate(
        routes,
        worlds=high_support_manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "stochastic"],
        ambiguity_context=high_support_manifest["ambiguity_context"],
    )

    assert high_support_manifest["stress_world_fraction"] >= low_support_manifest["stress_world_fraction"]
    assert high_support_manifest["ambiguity_context"]["support_strength"] > low_support_manifest["ambiguity_context"]["support_strength"]
    assert high_support_certificate.certificate["route_a"] <= low_support_certificate.certificate["route_a"]


def test_dependency_tensor_reflects_route_specific_operational_exposures() -> None:
    stressed_route = _route_option(
        "route_stressed",
        duration_s=7200.0,
        money_cost=180.0,
        co2_kg=220.0,
        distance_km=140.0,
        toll_cost=48.0,
        fuel_cost=62.0,
        carbon_cost=22.0,
        weather_delay_s=780.0,
        ascent_m=1450.0,
        descent_m=820.0,
        scenario_duration_multiplier=1.18,
        scenario_fuel_multiplier=1.12,
        scenario_emissions_multiplier=1.15,
        std_duration_s=880.0,
        std_money_cost=24.0,
        std_co2_kg=32.0,
    ).model_dump(mode="python")
    baseline_route = _route_option(
        "route_baseline",
        duration_s=7200.0,
        money_cost=180.0,
        co2_kg=220.0,
        distance_km=140.0,
        toll_cost=4.0,
        fuel_cost=28.0,
        carbon_cost=3.0,
        weather_delay_s=40.0,
        ascent_m=120.0,
        descent_m=80.0,
        scenario_duration_multiplier=1.01,
        scenario_fuel_multiplier=1.01,
        scenario_emissions_multiplier=1.01,
        std_duration_s=80.0,
        std_money_cost=2.0,
        std_co2_kg=3.0,
    ).model_dump(mode="python")

    stressed_tensor = dependency_tensor(stressed_route, active_families=EVIDENCE_FAMILIES)
    baseline_tensor = dependency_tensor(baseline_route, active_families=EVIDENCE_FAMILIES)

    assert stressed_tensor["money"]["toll"] > baseline_tensor["money"]["toll"]
    assert stressed_tensor["time"]["weather"] > baseline_tensor["time"]["weather"]
    assert stressed_tensor["time"]["stochastic"] > baseline_tensor["time"]["stochastic"]
    assert stressed_tensor["money"]["stochastic"] > baseline_tensor["money"]["stochastic"]
    assert stressed_tensor["co2"]["weather"] > baseline_tensor["co2"]["weather"]


def test_support_rich_non_hard_case_manifest_adds_targeted_stress_pack_and_weakens_certificate() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 0.7, "co2": 0.5},
                "weather": {"time": 0.8, "money": 0.1, "co2": 0.1},
                "toll": {"time": 0.0, "money": 1.0, "co2": 0.0},
            },
        ),
        _route("route_b", objective=(10.02, 10.03, 10.01), evidence_tensor={}),
    ]
    easy_manifest = sample_world_manifest(
        active_families=["scenario", "weather", "toll"],
        seed=41,
        world_count=10,
        routes=routes,
        ambiguity_context={
            "od_ambiguity_index": 0.18,
            "od_hard_case_prior": 0.10,
            "od_engine_disagreement_prior": 0.08,
            "od_candidate_path_count": 1,
            "od_corridor_family_count": 1,
            "od_objective_spread": 0.12,
            "od_nominal_margin_proxy": 0.72,
            "od_ambiguity_support_ratio": 0.10,
            "od_ambiguity_source_entropy": 0.08,
            "od_ambiguity_confidence": 0.25,
            "od_ambiguity_source_count": 1,
            "ambiguity_budget_prior": 0.12,
            "ambiguity_budget_band": "low",
        },
    )
    supported_manifest = sample_world_manifest(
        active_families=["scenario", "weather", "toll"],
        seed=41,
        world_count=10,
        routes=routes,
        ambiguity_context={
            "od_ambiguity_index": 0.42,
            "od_hard_case_prior": 0.24,
            "od_engine_disagreement_prior": 0.22,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.26,
            "od_nominal_margin_proxy": 0.28,
            "od_ambiguity_family_density": 0.46,
            "od_ambiguity_margin_pressure": 0.42,
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.36,
            "ambiguity_budget_band": "medium",
        },
    )

    easy_certificate = compute_certificate(
        routes,
        worlds=easy_manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "toll"],
        ambiguity_context=easy_manifest["ambiguity_context"],
    )
    supported_certificate = compute_certificate(
        routes,
        worlds=supported_manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "toll"],
        ambiguity_context=supported_manifest["ambiguity_context"],
    )

    assert supported_manifest["ambiguity_context"]["is_hard_case"] is False
    assert supported_manifest["ambiguity_context"]["is_supported_ambiguity_case"] is True
    assert supported_manifest["hard_case_stress_pack_count"] == 0
    assert supported_manifest["supported_ambiguity_stress_pack_count"] > 0
    assert supported_manifest["single_family_targeted_stress_pack_count"] > 0
    assert supported_manifest["refc_stress_world_fraction"] > 0.0
    assert supported_certificate.world_manifest["supported_ambiguity_stress_pack_count"] > 0
    assert supported_certificate.world_manifest["ambiguity_context"]["supported_ambiguity_stress_pack_count"] > 0
    assert supported_certificate.certificate["route_a"] < easy_certificate.certificate["route_a"]


def test_supported_ambiguity_frontier_can_surface_nonzero_fragility_and_refresh_gain() -> None:
    routes = [
        _route_option(
            "route_a",
            duration_s=41.0,
            money_cost=8.5,
            co2_kg=3.2,
            distance_km=12.3,
            toll_cost=1.0,
            fuel_cost=5.1,
            carbon_cost=0.4,
            weather_delay_s=2.5,
            ascent_m=54.0,
            descent_m=41.0,
            scenario_duration_multiplier=1.05,
            scenario_fuel_multiplier=0.97,
            scenario_emissions_multiplier=0.95,
            scenario_sigma_multiplier=1.15,
            std_duration_s=2.1,
            std_money_cost=0.35,
            std_co2_kg=0.16,
            active_families=("scenario", "weather", "toll"),
        ),
        _route_option(
            "route_b",
            duration_s=41.15,
            money_cost=8.56,
            co2_kg=3.24,
            distance_km=12.35,
            toll_cost=0.5,
            fuel_cost=5.45,
            carbon_cost=0.5,
            weather_delay_s=3.35,
            ascent_m=63.0,
            descent_m=44.0,
            scenario_duration_multiplier=1.04,
            scenario_fuel_multiplier=1.0,
            scenario_emissions_multiplier=0.99,
            scenario_sigma_multiplier=1.1,
            std_duration_s=2.35,
            std_money_cost=0.4,
            std_co2_kg=0.18,
            active_families=("scenario", "weather", "toll"),
        ),
        _route_option(
            "route_c",
            duration_s=41.28,
            money_cost=8.63,
            co2_kg=3.29,
            distance_km=12.4,
            toll_cost=0.4,
            fuel_cost=5.7,
            carbon_cost=0.6,
            weather_delay_s=3.6,
            ascent_m=66.0,
            descent_m=48.0,
            scenario_duration_multiplier=1.02,
            scenario_fuel_multiplier=1.01,
            scenario_emissions_multiplier=1.0,
            scenario_sigma_multiplier=1.08,
            std_duration_s=2.6,
            std_money_cost=0.45,
            std_co2_kg=0.2,
            active_families=("scenario", "weather", "toll"),
        ),
    ]
    lightweight_routes = [main_module._route_option_lightweight_copy(route) for route in routes]
    projected_routes = [main_module._route_option_certification_payload(route) for route in lightweight_routes]
    assert projected_routes[0]["metrics"]["distance_km"] == 12.3
    assert len(projected_routes[0]["segment_breakdown"]) == 1
    assert projected_routes[0]["segment_breakdown"][0]["segment_count"] == 1
    assert projected_routes[0]["segment_breakdown"][0]["toll_cost"] > 0.0
    assert projected_routes[0]["weather_summary"]["weather_delay_s"] == 2.5
    assert projected_routes[0]["scenario_summary"]["mode"] == "no_sharing"
    assert projected_routes[0]["terrain_summary"]["source"] == "dem_real"
    assert projected_routes[0]["uncertainty"]["std_duration_s"] == 2.1
    assert projected_routes[0]["evidence_provenance"]["dependency_weights"]["scenario"]["time"] > 0.0
    json.dumps(projected_routes[0], sort_keys=True)

    manifest = sample_world_manifest(
        active_families=["scenario", "weather", "toll"],
        seed=41,
        world_count=12,
        routes=projected_routes,
        ambiguity_context={
            "od_ambiguity_index": 0.42,
            "od_hard_case_prior": 0.24,
            "od_engine_disagreement_prior": 0.22,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.26,
            "od_nominal_margin_proxy": 0.28,
            "od_ambiguity_family_density": 0.46,
            "od_ambiguity_margin_pressure": 0.42,
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.36,
            "ambiguity_budget_band": "medium",
        },
    )

    stress_worlds = [
        world
        for world in manifest["worlds"]
        if str(world.get("world_kind", "")).startswith("supported_ambiguity_")
    ]
    assert stress_worlds
    assert any(world.get("target_route_id") == "route_a" for world in stress_worlds)
    assert any(
        sum(1 for route_id in (world.get("target_route_ids") or {}).values() if route_id == "route_a") >= 2
        for world in stress_worlds
    )

    certificate = compute_certificate(
        projected_routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "toll"],
        ambiguity_context=manifest["ambiguity_context"],
    )
    assert certificate.certificate["route_a"] < 1.0
    fragility = compute_fragility_maps(
        projected_routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather", "toll"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=manifest["ambiguity_context"],
    )

    assert manifest["supported_ambiguity_stress_pack_count"] > 0
    assert fragility.route_fragility_map["route_a"]["scenario"] > 0.0
    assert fragility.value_of_refresh["top_refresh_family"] == "scenario"
    assert fragility.value_of_refresh["top_refresh_gain"] > 0.0
    assert fragility.route_fragility_details["route_a"]["scenario"]["refresh_gain"] > 0.0


def test_supported_ambiguity_saturated_winner_keeps_certificate_at_one_but_surfaces_margin_refresh_gain() -> None:
    routes = [
        _route_option(
            "route_a",
            duration_s=10.0,
            money_cost=10.0,
            co2_kg=10.0,
            distance_km=12.3,
            toll_cost=1.0,
            fuel_cost=5.1,
            carbon_cost=0.4,
            weather_delay_s=2.5,
            ascent_m=54.0,
            descent_m=41.0,
            scenario_duration_multiplier=1.05,
            scenario_fuel_multiplier=0.97,
            scenario_emissions_multiplier=0.95,
            scenario_sigma_multiplier=1.15,
            std_duration_s=2.1,
            std_money_cost=0.35,
            std_co2_kg=0.16,
            active_families=("scenario", "weather", "toll"),
        ),
        _route_option(
            "route_b",
            duration_s=12.0,
            money_cost=12.0,
            co2_kg=12.0,
            distance_km=12.35,
            toll_cost=0.5,
            fuel_cost=5.45,
            carbon_cost=0.5,
            weather_delay_s=3.35,
            ascent_m=63.0,
            descent_m=44.0,
            scenario_duration_multiplier=1.04,
            scenario_fuel_multiplier=1.0,
            scenario_emissions_multiplier=0.99,
            scenario_sigma_multiplier=1.1,
            std_duration_s=2.35,
            std_money_cost=0.4,
            std_co2_kg=0.18,
            active_families=("scenario", "weather", "toll"),
        ),
        _route_option(
            "route_c",
            duration_s=12.2,
            money_cost=12.15,
            co2_kg=12.1,
            distance_km=12.4,
            toll_cost=0.4,
            fuel_cost=5.7,
            carbon_cost=0.6,
            weather_delay_s=3.6,
            ascent_m=66.0,
            descent_m=48.0,
            scenario_duration_multiplier=1.02,
            scenario_fuel_multiplier=1.01,
            scenario_emissions_multiplier=1.0,
            scenario_sigma_multiplier=1.08,
            std_duration_s=2.6,
            std_money_cost=0.45,
            std_co2_kg=0.2,
            active_families=("scenario", "weather", "toll"),
        ),
    ]
    lightweight_routes = [main_module._route_option_lightweight_copy(route) for route in routes]
    projected_routes = [main_module._route_option_certification_payload(route) for route in lightweight_routes]
    manifest = sample_world_manifest(
        active_families=["scenario", "weather", "toll"],
        seed=41,
        world_count=12,
        routes=projected_routes,
        ambiguity_context={
            "od_ambiguity_index": 0.42,
            "od_hard_case_prior": 0.24,
            "od_engine_disagreement_prior": 0.22,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.26,
            "od_nominal_margin_proxy": 0.28,
            "od_ambiguity_family_density": 0.46,
            "od_ambiguity_margin_pressure": 0.42,
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.36,
            "ambiguity_budget_band": "medium",
        },
    )

    certificate = compute_certificate(
        projected_routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "toll"],
        ambiguity_context=manifest["ambiguity_context"],
    )
    assert certificate.certificate["route_a"] < 1.0
    fragility = compute_fragility_maps(
        projected_routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather", "toll"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=manifest["ambiguity_context"],
    )

    assert certificate.certificate["route_a"] == 1.0
    assert fragility.route_fragility_map["route_a"]["scenario"] > 0.0
    assert fragility.value_of_refresh["top_refresh_gain"] > 0.0
    assert fragility.route_fragility_details["route_a"]["scenario"]["margin_fragility"] > 0.0
    assert fragility.value_of_refresh["top_refresh_gain"] <= 0.01
    assert fragility.route_fragility_details["route_a"]["scenario"]["margin_fragility"] <= 0.01


def test_supported_ambiguity_near_tie_floor_surfaces_winner_fragility_without_actionable_refresh_gain() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={},
        ),
        _route("route_b", objective=(10.2, 10.2, 10.2), evidence_tensor={}),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal", "weather": "nominal", "toll": "nominal"}},
        {"world_id": "w1", "states": {"scenario": "nominal", "weather": "nominal", "toll": "nominal"}},
        {"world_id": "w2", "states": {"scenario": "nominal", "weather": "nominal", "toll": "nominal"}},
    ]
    ambiguity_context = {
        "od_ambiguity_index": 0.42,
        "od_hard_case_prior": 0.24,
        "od_engine_disagreement_prior": 0.22,
        "od_candidate_path_count": 4,
        "od_corridor_family_count": 3,
        "od_objective_spread": 0.26,
        "od_nominal_margin_proxy": 0.28,
        "od_ambiguity_family_density": 0.46,
        "od_ambiguity_margin_pressure": 0.42,
        "od_ambiguity_support_ratio": 0.78,
        "od_ambiguity_source_entropy": 0.72,
        "od_ambiguity_confidence": 0.93,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
        "ambiguity_budget_prior": 0.36,
            "ambiguity_budget_band": "medium",
        }
    floor_value = 0.003
    original_floor = evidence_certification_module._winner_recoverable_fragility_floor
    evidence_certification_module._winner_recoverable_fragility_floor = lambda *args, **kwargs: floor_value
    try:
        certificate = compute_certificate(
            routes,
            worlds=worlds,
            selector_weights=(1.0, 1.0, 1.0),
            threshold=0.67,
            active_families=["scenario", "weather", "toll"],
            ambiguity_context=ambiguity_context,
        )
        fragility = compute_fragility_maps(
            routes,
            worlds=worlds,
            selector_weights=(1.0, 1.0, 1.0),
            active_families=["scenario", "weather", "toll"],
            selected_route_id="route_a",
            baseline_certificate=certificate,
            ambiguity_context=ambiguity_context,
        )
    finally:
        evidence_certification_module._winner_recoverable_fragility_floor = original_floor

    detail = fragility.route_fragility_details["route_a"]["scenario"]
    assert detail["absolute_drop"] == pytest.approx(0.0)
    assert detail["margin_fragility"] == pytest.approx(0.0)
    assert detail["winner_recoverable_floor"] == pytest.approx(floor_value)
    assert fragility.route_fragility_map["route_a"]["scenario"] == pytest.approx(detail["winner_recoverable_floor"])
    assert detail["raw_refresh_gain"] == pytest.approx(detail["winner_recoverable_floor"])
    assert detail["refresh_gain"] == pytest.approx(0.0)
    assert fragility.value_of_refresh["top_refresh_gain"] == pytest.approx(0.0)
    assert all(entry["vor"] == pytest.approx(0.0) for entry in fragility.value_of_refresh["ranking"])


def test_supported_ambiguity_stress_recovery_keeps_vor_positive_when_baseline_equals_refreshed() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        ),
        _route("route_b", objective=(10.002, 10.002, 10.002), evidence_tensor={}),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal"}},
        {"world_id": "w1", "states": {"scenario": "nominal"}},
        {"world_id": "w2", "states": {"scenario": "nominal"}},
    ]
    ambiguity_context = {
        "od_ambiguity_index": 0.42,
        "od_hard_case_prior": 0.24,
        "od_engine_disagreement_prior": 0.22,
        "od_candidate_path_count": 4,
        "od_corridor_family_count": 3,
        "od_objective_spread": 0.26,
        "od_nominal_margin_proxy": 0.28,
        "od_ambiguity_family_density": 0.46,
        "od_ambiguity_margin_pressure": 0.42,
        "od_ambiguity_support_ratio": 0.78,
        "od_ambiguity_source_entropy": 0.72,
        "od_ambiguity_confidence": 0.93,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
        "ambiguity_budget_prior": 0.36,
        "ambiguity_budget_band": "medium",
    }
    certificate = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.67,
        active_families=["scenario"],
        ambiguity_context=ambiguity_context,
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=ambiguity_context,
    )

    detail = fragility.route_fragility_details["route_a"]["scenario"]
    assert detail["baseline_certificate"] == detail["refreshed_certificate"]
    assert detail["stressed_certificate"] < detail["refreshed_certificate"]
    assert detail["certificate_refresh_gain"] == pytest.approx(0.0)
    assert detail["certificate_stress_recovery"] > 0.0
    assert detail["refresh_gain"] > 0.0
    assert fragility.value_of_refresh["top_refresh_gain"] > 0.0


def test_fully_refreshed_family_blocks_repeat_refresh_gain_but_keeps_structural_fragility() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        ),
        _route("route_b", objective=(10.002, 10.002, 10.002), evidence_tensor={}),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "refreshed"}},
        {"world_id": "w1", "states": {"scenario": "refreshed"}},
        {"world_id": "w2", "states": {"scenario": "refreshed"}},
    ]
    ambiguity_context = {
        "od_ambiguity_index": 0.42,
        "od_hard_case_prior": 0.24,
        "od_engine_disagreement_prior": 0.22,
        "od_candidate_path_count": 4,
        "od_corridor_family_count": 3,
        "od_objective_spread": 0.26,
        "od_nominal_margin_proxy": 0.28,
        "od_ambiguity_family_density": 0.46,
        "od_ambiguity_margin_pressure": 0.42,
        "od_ambiguity_support_ratio": 0.78,
        "od_ambiguity_source_entropy": 0.72,
        "od_ambiguity_confidence": 0.93,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
        "ambiguity_budget_prior": 0.36,
        "ambiguity_budget_band": "medium",
    }
    certificate = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.67,
        active_families=["scenario"],
        ambiguity_context=ambiguity_context,
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=ambiguity_context,
    )

    detail = fragility.route_fragility_details["route_a"]["scenario"]
    assert detail["family_fully_refreshed"] is True
    assert detail["raw_refresh_gain"] > 0.0
    assert detail["refresh_gain"] == pytest.approx(0.0)
    assert fragility.route_fragility_map["route_a"]["scenario"] > 0.0
    assert fragility.value_of_refresh["top_refresh_gain"] == pytest.approx(0.0)


def test_representative_wide_margin_rows_keep_winner_floor_quiet() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={},
        ),
        _route("route_b", objective=(20.0, 20.0, 20.0), evidence_tensor={}),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal"}},
        {"world_id": "w1", "states": {"scenario": "nominal"}},
        {"world_id": "w2", "states": {"scenario": "nominal"}},
    ]
    ambiguity_context = {
        "od_ambiguity_index": 0.06,
        "od_hard_case_prior": 0.02,
        "od_engine_disagreement_prior": 0.02,
        "od_candidate_path_count": 1,
        "od_corridor_family_count": 1,
        "od_ambiguity_support_ratio": 0.08,
        "od_ambiguity_source_entropy": 0.05,
        "od_ambiguity_source_count": 1,
        "od_ambiguity_source_mix_count": 1,
        "od_nominal_margin_proxy": 0.95,
    }
    certificate = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.67,
        active_families=["scenario"],
        ambiguity_context=ambiguity_context,
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=ambiguity_context,
    )

    detail = fragility.route_fragility_details["route_a"]["scenario"]
    assert detail["winner_recoverable_floor"] == pytest.approx(0.0)
    assert fragility.route_fragility_map["route_a"]["scenario"] == pytest.approx(0.0)
    assert fragility.value_of_refresh["top_refresh_gain"] == pytest.approx(0.0)


def test_supported_ambiguity_stress_pack_scopes_worlds_to_target_routes() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        ),
        _route("route_b", objective=(10.002, 10.002, 10.002), evidence_tensor={}),
    ]
    ambiguity_context = {
        "od_ambiguity_index": 0.42,
        "od_hard_case_prior": 0.24,
        "od_engine_disagreement_prior": 0.22,
        "od_candidate_path_count": 4,
        "od_corridor_family_count": 3,
        "od_objective_spread": 0.26,
        "od_nominal_margin_proxy": 0.28,
        "od_ambiguity_family_density": 0.46,
        "od_ambiguity_margin_pressure": 0.42,
        "od_ambiguity_support_ratio": 0.78,
        "od_ambiguity_source_entropy": 0.72,
        "od_ambiguity_confidence": 0.93,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
        "ambiguity_budget_prior": 0.36,
        "ambiguity_budget_band": "medium",
    }
    manifest = sample_world_manifest(
        active_families=["scenario"],
        seed=7,
        world_count=4,
        routes=routes,
        ambiguity_context=ambiguity_context,
        selector_weights=(1.0, 1.0, 1.0),
    )

    stress_worlds = [
        world
        for world in manifest["worlds"]
        if str(world.get("world_kind", "")).startswith("supported_ambiguity_")
    ]
    assert stress_worlds
    assert any(world.get("target_route_ids") for world in stress_worlds)
    assert any(
        "route_a" in set((world.get("target_route_ids") or {}).values())
        for world in stress_worlds
    )


def test_supported_ambiguity_route_targeted_stress_pack_makes_certificate_nontrivial() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        ),
        _route("route_b", objective=(10.002, 10.002, 10.002), evidence_tensor={}),
    ]
    ambiguity_context = {
        "od_ambiguity_index": 0.42,
        "od_hard_case_prior": 0.24,
        "od_engine_disagreement_prior": 0.22,
        "od_candidate_path_count": 4,
        "od_corridor_family_count": 3,
        "od_objective_spread": 0.26,
        "od_nominal_margin_proxy": 0.28,
        "od_ambiguity_family_density": 0.46,
        "od_ambiguity_margin_pressure": 0.42,
        "od_ambiguity_support_ratio": 0.78,
        "od_ambiguity_source_entropy": 0.72,
        "od_ambiguity_confidence": 0.93,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
        "ambiguity_budget_prior": 0.36,
        "ambiguity_budget_band": "medium",
    }
    manifest = sample_world_manifest(
        active_families=["scenario"],
        seed=11,
        world_count=3,
        routes=routes,
        ambiguity_context=ambiguity_context,
        selector_weights=(1.0, 1.0, 1.0),
    )

    certificate = compute_certificate(
        routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.67,
        active_families=["scenario"],
        ambiguity_context=ambiguity_context,
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=ambiguity_context,
    )

    assert certificate.certificate["route_a"] < 1.0
    assert fragility.route_fragility_map["route_a"]["scenario"] > 0.0


def test_refc_single_frontier_shortcut_is_bypassed_for_supported_rows() -> None:
    assert (
        main_module._refc_requires_full_stress_worlds(
            {
                "od_ambiguity_index": 0.42,
                "od_hard_case_prior": 0.24,
                "od_engine_disagreement_prior": 0.22,
                "od_ambiguity_support_ratio": 0.78,
                "od_ambiguity_source_entropy": 0.72,
                "od_ambiguity_source_count": 3,
                "od_ambiguity_source_mix_count": 3,
            }
        )
        is True
    )
    assert (
        main_module._refc_requires_full_stress_worlds(
            {
                "od_ambiguity_index": 0.18,
                "od_hard_case_prior": 0.08,
                "od_engine_disagreement_prior": 0.08,
                "od_ambiguity_support_ratio": 0.10,
                "od_ambiguity_source_entropy": 0.08,
                "od_ambiguity_source_count": 1,
                "od_ambiguity_source_mix_count": 1,
            }
        )
        is False
    )


def test_fragility_reuses_evaluated_world_bundle_without_changing_results() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        ),
        _route("route_b", objective=(10.1, 10.1, 10.1), evidence_tensor={}),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal"}},
        {"world_id": "w1", "states": {"scenario": "nominal"}},
        {"world_id": "w2", "states": {"scenario": "severely_stale"}},
    ]
    evaluated_bundle = evaluate_world_bundle(
        routes,
        worlds,
        active_families=["scenario"],
        selector_weights=(1.0, 1.0, 1.0),
    )
    certificate = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        evaluated_bundle=evaluated_bundle,
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        selected_route_id="route_a",
        evaluated_bundle=evaluated_bundle,
        baseline_certificate=certificate,
    )

    assert evaluated_bundle.unique_world_count == 2
    assert certificate.world_manifest["unique_world_count"] == 2
    assert certificate.world_manifest["world_reuse_rate"] == pytest.approx(1.0 / 3.0)
    assert fragility.evidence_snapshot_manifest["baseline_unique_world_count"] == 2


def test_snapshot_manifest_captures_payload_hashes_deterministically() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
            snapshot_payloads={"scenario": {"version": 3, "coefficients": [1.0, 1.2]}},
        ),
        _route("route_b", objective=(10.2, 10.2, 10.2), evidence_tensor={}),
    ]
    fragility_a = compute_fragility_maps(
        routes,
        worlds=[{"world_id": "w0", "states": {"scenario": "nominal"}}],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        selected_route_id="route_a",
    )
    fragility_b = compute_fragility_maps(
        routes,
        worlds=[{"world_id": "w0", "states": {"scenario": "nominal"}}],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        selected_route_id="route_a",
    )

    manifest_a = fragility_a.evidence_snapshot_manifest
    manifest_b = fragility_b.evidence_snapshot_manifest
    assert manifest_a == manifest_b
    assert manifest_a["family_snapshots"]["scenario"][0]["snapshot_payload_keys"] == ["coefficients", "version"]


def test_compute_certificate_rejects_empty_world_sets() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        )
    ]

    try:
        compute_certificate(
            routes,
            worlds=[],
            selector_weights=(1.0, 1.0, 1.0),
            active_families=["scenario"],
        )
    except ValueError as exc:
        assert "worlds" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("empty world sets should be rejected")


def test_route_option_dependency_weights_are_route_specific() -> None:
    toll_heavy = _route_option(
        "route_toll",
        duration_s=8_000.0,
        money_cost=220.0,
        co2_kg=260.0,
        toll_cost=80.0,
        fuel_cost=50.0,
        carbon_cost=10.0,
        weather_delay_s=50.0,
        ascent_m=80.0,
        descent_m=75.0,
        scenario_duration_multiplier=1.04,
        scenario_fuel_multiplier=1.02,
        scenario_emissions_multiplier=1.01,
        scenario_sigma_multiplier=1.03,
        std_duration_s=180.0,
        std_money_cost=12.0,
        std_co2_kg=16.0,
    )
    terrain_stochastic = _route_option(
        "route_terrain",
        duration_s=8_400.0,
        money_cost=210.0,
        co2_kg=250.0,
        toll_cost=5.0,
        fuel_cost=85.0,
        carbon_cost=18.0,
        weather_delay_s=420.0,
        ascent_m=2_200.0,
        descent_m=2_150.0,
        scenario_duration_multiplier=1.12,
        scenario_fuel_multiplier=1.08,
        scenario_emissions_multiplier=1.06,
        scenario_sigma_multiplier=1.22,
        std_duration_s=980.0,
        std_money_cost=35.0,
        std_co2_kg=44.0,
    )

    toll_weights = main_module._route_option_dependency_weights(toll_heavy)
    terrain_weights = main_module._route_option_dependency_weights(terrain_stochastic)

    assert toll_weights["toll"]["money"] > terrain_weights["toll"]["money"]
    assert terrain_weights["terrain"]["co2"] > toll_weights["terrain"]["co2"]
    assert terrain_weights["weather"]["time"] > toll_weights["weather"]["time"]
    assert terrain_weights["stochastic"]["time"] > toll_weights["stochastic"]["time"]
    assert terrain_weights["scenario"]["money"] > toll_weights["scenario"]["money"]


def test_route_scoped_world_only_perturbs_target_route() -> None:
    route_a = _route(
        "route_a",
        objective=(10.0, 10.0, 10.0),
        evidence_tensor={
            "scenario": {"time": 0.72, "money": 0.62, "co2": 0.57},
            "weather": {"time": 0.11, "money": 0.12, "co2": 0.10},
        },
    )
    route_b = _route(
        "route_b",
        objective=(10.15, 10.15, 10.15),
        evidence_tensor={
            "scenario": {"time": 0.68, "money": 0.59, "co2": 0.54},
            "weather": {"time": 0.13, "money": 0.14, "co2": 0.12},
        },
    )
    scoped_world = {
        "world_id": "w0",
        "states": {"scenario": "severely_stale", "weather": "nominal"},
        "stress_factor": 1.15,
        "target_route_id": "route_a",
    }

    route_a_perturbed = _route_perturbed_objectives(
        route_a,
        scoped_world,
        active_families=["scenario", "weather"],
    )
    route_b_perturbed = _route_perturbed_objectives(
        route_b,
        scoped_world,
        active_families=["scenario", "weather"],
    )

    assert route_a_perturbed != tuple(route_a["objective_vector"])
    assert route_b_perturbed == tuple(route_b["objective_vector"])


def test_supported_ambiguity_dependency_contrast_surfaces_winner_refresh_signal() -> None:
    routes = [
        _route_option(
            "route_a",
            duration_s=41.0,
            money_cost=8.5,
            co2_kg=3.2,
            distance_km=12.3,
            toll_cost=1.0,
            fuel_cost=5.1,
            carbon_cost=0.4,
            weather_delay_s=2.5,
            ascent_m=54.0,
            descent_m=41.0,
            scenario_duration_multiplier=1.05,
            scenario_fuel_multiplier=0.97,
            scenario_emissions_multiplier=0.95,
            scenario_sigma_multiplier=1.15,
            std_duration_s=2.1,
            std_money_cost=0.35,
            std_co2_kg=0.16,
            active_families=("scenario", "weather", "toll"),
        ),
        _route_option(
            "route_b",
            duration_s=41.15,
            money_cost=8.56,
            co2_kg=3.24,
            distance_km=12.35,
            toll_cost=0.5,
            fuel_cost=5.45,
            carbon_cost=0.5,
            weather_delay_s=3.35,
            ascent_m=63.0,
            descent_m=44.0,
            scenario_duration_multiplier=1.04,
            scenario_fuel_multiplier=1.0,
            scenario_emissions_multiplier=0.99,
            scenario_sigma_multiplier=1.1,
            std_duration_s=2.35,
            std_money_cost=0.4,
            std_co2_kg=0.18,
            active_families=("scenario", "weather", "toll"),
        ),
        _route_option(
            "route_c",
            duration_s=41.28,
            money_cost=8.63,
            co2_kg=3.29,
            distance_km=12.4,
            toll_cost=0.4,
            fuel_cost=5.7,
            carbon_cost=0.6,
            weather_delay_s=3.6,
            ascent_m=66.0,
            descent_m=48.0,
            scenario_duration_multiplier=1.02,
            scenario_fuel_multiplier=1.01,
            scenario_emissions_multiplier=1.0,
            scenario_sigma_multiplier=1.08,
            std_duration_s=2.6,
            std_money_cost=0.45,
            std_co2_kg=0.2,
            active_families=("scenario", "weather", "toll"),
        ),
    ]
    lightweight_routes = [main_module._route_option_lightweight_copy(route) for route in routes]
    projected_routes = [
        main_module._route_option_certification_payload(route) for route in lightweight_routes
    ]
    manifest = sample_world_manifest(
        active_families=["scenario", "weather", "toll"],
        seed=41,
        world_count=12,
        routes=projected_routes,
        ambiguity_context={
            "od_ambiguity_index": 0.42,
            "od_hard_case_prior": 0.24,
            "od_engine_disagreement_prior": 0.22,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.26,
            "od_nominal_margin_proxy": 0.28,
            "od_ambiguity_family_density": 0.46,
            "od_ambiguity_margin_pressure": 0.42,
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.36,
            "ambiguity_budget_band": "medium",
        },
    )

    certificate = compute_certificate(
        projected_routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "toll"],
        ambiguity_context=manifest["ambiguity_context"],
    )
    fragility = compute_fragility_maps(
        projected_routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather", "toll"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=manifest["ambiguity_context"],
    )

    route_a_weights = main_module._route_option_dependency_weights(lightweight_routes[0])
    route_b_weights = main_module._route_option_dependency_weights(lightweight_routes[1])

    assert route_a_weights["scenario"]["time"] > route_b_weights["scenario"]["time"]
    assert route_b_weights["weather"]["time"] > route_a_weights["weather"]["time"]
    assert (
        route_a_weights["scenario"]["time"] - route_a_weights["weather"]["time"]
        > route_b_weights["scenario"]["time"] - route_b_weights["weather"]["time"]
    )
    assert manifest["supported_ambiguity_stress_pack_count"] > 0
    assert fragility.route_fragility_map["route_a"]["scenario"] > 0.0
    assert fragility.value_of_refresh["top_refresh_gain"] > 0.0
