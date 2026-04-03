from __future__ import annotations

import json

import pytest

import app.evidence_certification as evidence_certification_module
import app.main as main_module
from app.abstention import ABSTENTION_REASON_CODES, AbstentionRecord
from app.certificate_witness import CertificateWitness
from app.certification_models import CertificationState
from app.certified_set import CertifiedSetState
from app.confidence_sequences import WinnerConfidenceState, winner_confidence_sequence
from app.decision_region import DecisionRegionState
from app.flip_radius import FlipRadiusState
from app.models import (
    EvidenceProvenance,
    EvidenceSourceRecord,
    GeoJSONLineString,
    RouteMetrics,
    RouteOption,
    ScenarioSummary,
    TerrainSummaryPayload,
)
from app.pairwise_gap_model import PairwiseGapState
from app.evidence_certification import (
    EVIDENCE_FAMILIES,
    annotate_world_manifest_cache_reuse,
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
    coordinates: list[tuple[float, float]] | None = None,
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
            coordinates=coordinates or [(-1.0, 52.0), (-0.5, 52.2)],
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


def test_annotate_world_manifest_cache_reuse_marks_cross_request_hits_without_losing_within_manifest_reuse() -> None:
    manifest = {
        "world_reuse_rate": 0.333333,
        "world_reuse_rate_within_manifest": 0.333333,
        "world_reuse_rate_cross_request": 0.0,
        "certification_cache_reuse_origin": "miss",
    }

    global_hit = annotate_world_manifest_cache_reuse(manifest, cache_reuse_origin="global")
    local_hit = annotate_world_manifest_cache_reuse(manifest, cache_reuse_origin="local")

    assert global_hit["world_reuse_rate_within_manifest"] == pytest.approx(0.333333, rel=0.0, abs=1e-6)
    assert global_hit["world_reuse_rate_cross_request"] == 1.0
    assert global_hit["world_reuse_rate"] == 1.0
    assert global_hit["certification_cache_reuse_origin"] == "global"
    assert global_hit["certification_cache_reuse_applied"] is True
    assert local_hit["world_reuse_rate_cross_request"] == 0.0
    assert local_hit["world_reuse_rate"] == pytest.approx(0.333333, rel=0.0, abs=1e-6)
    assert local_hit["certification_cache_reuse_origin"] == "local"
    assert local_hit["certification_cache_reuse_applied"] is True


def test_annotate_world_manifest_cache_reuse_preserves_existing_cross_request_reuse_on_local_hit() -> None:
    manifest = {
        "world_reuse_rate": 1.0,
        "world_reuse_rate_within_manifest": 0.0,
        "world_reuse_rate_cross_request": 1.0,
        "certification_cache_reuse_origin": "global",
    }

    local_hit = annotate_world_manifest_cache_reuse(manifest, cache_reuse_origin="local")

    assert local_hit["world_reuse_rate_within_manifest"] == 0.0
    assert local_hit["world_reuse_rate_cross_request"] == 1.0
    assert local_hit["world_reuse_rate"] == 1.0
    assert local_hit["certification_cache_reuse_origin"] == "local"
    assert local_hit["certification_cache_reuse_applied"] is True


def test_global_certification_cache_payload_compacts_nonessential_detail_fields() -> None:
    certificate = main_module.CertificateResult(
        winner_id="route_0",
        certificate={"route_0": 0.82, "route_1": 0.18},
        threshold=0.67,
        certified=True,
        selected_route_id="route_0",
        route_scores={"route_0": [0.8, 0.84], "route_1": [0.2, 0.16]},
        world_manifest={"status": "ok", "world_count": 2},
        selector_config={"selector_weights": (1.0, 1.0, 1.0)},
    )
    fragility = main_module.FragilityResult(
        route_fragility_map={"route_0": {"scenario": 0.14}},
        competitor_fragility_breakdown={"route_0": {"route_1": {"scenario": 2}}},
        value_of_refresh={"ranking": [{"family": "scenario", "value": 0.14}]},
        route_fragility_details={"route_0": {"scenario": {"raw_refresh_gain": 0.14}}},
        evidence_snapshot_manifest={"snapshot_hash": "snap-1"},
    )

    cached_certificate, cached_fragility, cached_manifest, cached_families = (
        main_module._global_certification_cache_payload(
            certificate_result=certificate,
            fragility_result=fragility,
            world_manifest_payload={"status": "ok", "world_count": 2, "world_reuse_rate": 1.0},
            active_families=["scenario"],
        )
    )

    assert cached_certificate.certificate == certificate.certificate
    assert cached_certificate.route_scores == {}
    assert cached_certificate.world_manifest["world_reuse_rate"] == pytest.approx(1.0)
    assert cached_fragility.route_fragility_map == fragility.route_fragility_map
    assert cached_fragility.competitor_fragility_breakdown == fragility.competitor_fragility_breakdown
    assert cached_fragility.value_of_refresh == fragility.value_of_refresh
    assert cached_fragility.route_fragility_details == {}
    assert cached_fragility.evidence_snapshot_manifest == {}
    assert cached_manifest["world_reuse_rate"] == pytest.approx(1.0)
    assert cached_families == ["scenario"]


def test_stable_route_signature_map_rebinds_reused_refined_routes_without_built_option_ids() -> None:
    option_a = _route_option(
        "route_a",
        duration_s=100.0,
        money_cost=20.0,
        co2_kg=5.0,
        coordinates=[(-1.0, 52.0), (-0.5, 52.2)],
    )
    option_b = _route_option(
        "route_b",
        duration_s=101.0,
        money_cost=19.5,
        co2_kg=5.2,
        coordinates=[(-2.0, 53.0), (-1.5, 53.2)],
    )
    refined_routes = [
        {"geometry": {"coordinates": [(-2.0, 53.0), (-1.5, 53.2)]}},
        {"geometry": {"coordinates": [(-1.0, 52.0), (-0.5, 52.2)]}},
    ]

    signature_map = main_module._stable_route_signature_map_for_options(
        refined_routes,
        [option_a, option_b],
    )

    assert refined_routes[0]["_built_option_id"] == "route_b"
    assert refined_routes[1]["_built_option_id"] == "route_a"
    assert signature_map == {
        "route_a": main_module._route_option_signature(option_a),
        "route_b": main_module._route_option_signature(option_b),
    }


def test_certification_frontier_signature_map_uses_stable_option_geometry() -> None:
    option_a = _route_option(
        "route_a",
        duration_s=100.0,
        money_cost=20.0,
        co2_kg=5.0,
        coordinates=[(-1.0, 52.0), (-0.5, 52.2)],
    )
    option_b = _route_option(
        "route_b",
        duration_s=101.0,
        money_cost=19.5,
        co2_kg=5.2,
        coordinates=[(-2.0, 53.0), (-1.5, 53.2)],
    )

    signature_map = main_module._certification_frontier_signature_map([option_a, option_b])

    assert signature_map == {
        "route_a": main_module._route_option_signature(option_a),
        "route_b": main_module._route_option_signature(option_b),
    }


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
        "target_route_ids": {"scenario": "route_b", "toll": "route_b"},
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


def test_support_rich_hard_case_mixed_targeted_worlds_influence_route_scoped_family_analysis() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 1.0, "co2": 1.0},
                "toll": {"time": 0.6, "money": 0.6, "co2": 0.6},
            },
        ),
        _route(
            "route_b",
            objective=(10.12, 10.08, 10.10),
            evidence_tensor={},
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
        "target_route_ids": {"scenario": "route_b", "toll": "route_b"},
    }
    ambiguity_context = {
        "od_hard_case_prior": 0.72,
        "od_ambiguity_index": 0.54,
        "od_engine_disagreement_prior": 0.36,
        "od_candidate_path_count": 4,
        "od_corridor_family_count": 3,
        "od_ambiguity_support_ratio": 0.76,
        "od_ambiguity_source_entropy": 0.69,
        "od_ambiguity_confidence": 0.91,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
        "ambiguity_budget_prior": 0.48,
        "ambiguity_budget_band": "high",
    }

    isolated = compute_fragility_maps(
        routes,
        worlds=isolated_worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "toll"],
        selected_route_id="route_a",
        ambiguity_context=ambiguity_context,
    )
    with_mixed = compute_fragility_maps(
        routes,
        worlds=isolated_worlds + [mixed_world],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "toll"],
        selected_route_id="route_a",
        ambiguity_context=ambiguity_context,
    )

    assert with_mixed.route_fragility_details["route_a"]["scenario"]["baseline_certificate"] != pytest.approx(
        isolated.route_fragility_details["route_a"]["scenario"]["baseline_certificate"]
    )
    assert with_mixed.route_fragility_map["route_a"]["scenario"] != pytest.approx(
        isolated.route_fragility_map["route_a"]["scenario"]
    )
    scoped_stressed = evidence_certification_module._stressed_worlds(
        [mixed_world],
        "scenario",
        stress_state="severely_stale",
        target_route_id="route_a",
    )
    scoped_refreshed = evidence_certification_module._refreshed_worlds(
        [mixed_world],
        "scenario",
        target_route_id="route_a",
    )
    assert scoped_stressed[0]["target_route_ids"]["scenario"] == "route_a"
    assert scoped_stressed[0]["target_route_ids"]["toll"] == "route_a"
    assert scoped_refreshed[0]["target_route_ids"]["scenario"] == "route_a"
    assert scoped_refreshed[0]["target_route_ids"]["toll"] == "route_a"


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


def test_support_rich_ambiguity_hard_case_gets_relief_but_low_ambiguity_hard_case_does_not() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 0.8, "co2": 0.7},
                "weather": {"time": 0.8, "money": 0.2, "co2": 0.2},
                "toll": {"time": 0.0, "money": 1.0, "co2": 0.0},
                "stochastic": {"time": 0.7, "money": 0.6, "co2": 0.5},
            },
        ),
        _route("route_b", objective=(10.03, 10.02, 10.04), evidence_tensor={}),
    ]
    low_ambiguity_manifest = sample_world_manifest(
        active_families=["scenario", "weather", "toll", "stochastic"],
        seed=73,
        world_count=16,
        routes=routes,
        ambiguity_context={
            "od_ambiguity_index": 0.04,
            "od_hard_case_prior": 0.72,
            "od_engine_disagreement_prior": 0.28,
            "od_candidate_path_count": 5,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.18,
            "od_nominal_margin_proxy": 0.42,
            "od_ambiguity_family_density": 0.48,
            "od_ambiguity_margin_pressure": 0.36,
            "od_ambiguity_support_ratio": 0.84,
            "od_ambiguity_source_entropy": 0.82,
            "od_ambiguity_confidence": 0.94,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.04,
            "ambiguity_budget_band": "high",
        },
        selector_weights=(1.0, 1.0, 1.0),
    )
    high_ambiguity_manifest = sample_world_manifest(
        active_families=["scenario", "weather", "toll", "stochastic"],
        seed=73,
        world_count=16,
        routes=routes,
        ambiguity_context={
            "od_ambiguity_index": 0.54,
            "od_hard_case_prior": 0.72,
            "od_engine_disagreement_prior": 0.28,
            "od_candidate_path_count": 5,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.18,
            "od_nominal_margin_proxy": 0.42,
            "od_ambiguity_family_density": 0.48,
            "od_ambiguity_margin_pressure": 0.36,
            "od_ambiguity_support_ratio": 0.84,
            "od_ambiguity_source_entropy": 0.82,
            "od_ambiguity_confidence": 0.94,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.46,
            "ambiguity_budget_band": "high",
        },
        selector_weights=(1.0, 1.0, 1.0),
    )

    low_ambiguity_certificate = compute_certificate(
        routes,
        worlds=low_ambiguity_manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.67,
        active_families=["scenario", "weather", "toll", "stochastic"],
        ambiguity_context=low_ambiguity_manifest["ambiguity_context"],
    )
    high_ambiguity_certificate = compute_certificate(
        routes,
        worlds=high_ambiguity_manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.67,
        active_families=["scenario", "weather", "toll", "stochastic"],
        ambiguity_context=high_ambiguity_manifest["ambiguity_context"],
    )

    assert low_ambiguity_manifest["hard_case_stress_pack_count"] > 0
    assert high_ambiguity_manifest["hard_case_stress_pack_count"] > 0
    assert high_ambiguity_manifest["stress_world_fraction"] <= low_ambiguity_manifest["stress_world_fraction"]
    assert high_ambiguity_certificate.certificate["route_a"] >= (
        low_ambiguity_certificate.certificate["route_a"] - 0.01
    )


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

    assert high_support_manifest["stress_world_fraction"] <= (
        low_support_manifest["stress_world_fraction"] + 0.01
    )
    assert high_support_manifest["ambiguity_context"]["support_strength"] > low_support_manifest["ambiguity_context"]["support_strength"]
    assert high_support_certificate.certificate["route_a"] >= (
        low_support_certificate.certificate["route_a"] - 0.02
    )


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


def test_supported_ambiguity_targeted_stress_pack_stays_bounded_without_route_specific_gap() -> None:
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
    low_gap_routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 0.7, "money": 0.2, "co2": 0.2},
                "weather": {"time": 0.5, "money": 0.1, "co2": 0.1},
            },
        ),
        _route(
            "route_b",
            objective=(10.01, 10.01, 10.01),
            evidence_tensor={
                "scenario": {"time": 0.7, "money": 0.2, "co2": 0.2},
                "weather": {"time": 0.5, "money": 0.1, "co2": 0.1},
            },
        ),
        _route(
            "route_c",
            objective=(10.02, 10.02, 10.02),
            evidence_tensor={
                "scenario": {"time": 0.7, "money": 0.2, "co2": 0.2},
                "weather": {"time": 0.5, "money": 0.1, "co2": 0.1},
            },
        ),
    ]
    high_gap_routes = [
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
        _route("route_c", objective=(10.03, 10.04, 10.02), evidence_tensor={}),
    ]

    low_gap_pack = evidence_certification_module._targeted_stress_pack(
        routes=low_gap_routes,
        families=["scenario", "weather", "toll"],
        states=evidence_certification_module.EVIDENCE_STATES,
        seed=41,
        ambiguity_context=ambiguity_context,
        selector_weights=(1.0, 1.0, 1.0),
    )
    high_gap_pack = evidence_certification_module._targeted_stress_pack(
        routes=high_gap_routes,
        families=["scenario", "weather", "toll"],
        states=evidence_certification_module.EVIDENCE_STATES,
        seed=41,
        ambiguity_context=ambiguity_context,
        selector_weights=(1.0, 1.0, 1.0),
    )

    assert low_gap_pack
    assert all(str(world.world_kind).startswith("supported_ambiguity_") for world in low_gap_pack)
    assert len(low_gap_pack) <= 12
    assert len(high_gap_pack) > len(low_gap_pack)


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
    assert certificate.certificate["route_a"] == pytest.approx(1.0)
    fragility = compute_fragility_maps(
        projected_routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather", "toll"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=manifest["ambiguity_context"],
    )

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


def test_controller_refresh_family_uses_raw_refresh_gain_when_empirical_vor_ties() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={},
        ),
        _route("route_b", objective=(10.002, 10.002, 10.002), evidence_tensor={}),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "refreshed", "weather": "refreshed"}},
        {"world_id": "w1", "states": {"scenario": "refreshed", "weather": "refreshed"}},
        {"world_id": "w2", "states": {"scenario": "refreshed", "weather": "refreshed"}},
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
    original_floor = evidence_certification_module._winner_recoverable_fragility_floor
    evidence_certification_module._winner_recoverable_fragility_floor = (
        lambda _route, family, **kwargs: 0.002 if family == "scenario" else 0.006 if family == "weather" else 0.0
    )
    try:
        certificate = compute_certificate(
            routes,
            worlds=worlds,
            selector_weights=(1.0, 1.0, 1.0),
            threshold=0.67,
            active_families=["scenario", "weather"],
            ambiguity_context=ambiguity_context,
        )
        fragility = compute_fragility_maps(
            routes,
            worlds=worlds,
            selector_weights=(1.0, 1.0, 1.0),
            active_families=["scenario", "weather"],
            selected_route_id="route_a",
            baseline_certificate=certificate,
            ambiguity_context=ambiguity_context,
        )
    finally:
        evidence_certification_module._winner_recoverable_fragility_floor = original_floor

    assert fragility.value_of_refresh["top_refresh_gain"] == pytest.approx(0.0)
    assert fragility.value_of_refresh["top_refresh_family"] == "scenario"
    assert fragility.value_of_refresh["controller_ranking_basis"] == "raw_refresh_gain_fallback"
    assert fragility.value_of_refresh["top_refresh_family_controller"] == "weather"
    assert fragility.value_of_refresh["top_refresh_gain_controller"] == pytest.approx(0.006)
    assert fragility.value_of_refresh["controller_ranking"][0]["family"] == "weather"
    assert fragility.value_of_refresh["controller_ranking"][0]["raw_refresh_gain"] == pytest.approx(0.006)


def test_supported_ambiguity_prior_floor_can_surface_without_near_tie_margin_summary() -> None:
    route = _route(
        "route_a",
        objective=(10.0, 10.0, 10.0),
        evidence_tensor={"carbon": {"time": 0.2, "money": 0.4, "co2": 1.0}},
    )
    ambiguity_context = {
        "od_ambiguity_index": 0.41,
        "od_hard_case_prior": 0.18,
        "od_engine_disagreement_prior": 0.22,
        "od_candidate_path_count": 4,
        "od_corridor_family_count": 3,
        "od_objective_spread": 0.26,
        "od_nominal_margin_proxy": 0.30,
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

    floor = evidence_certification_module._winner_recoverable_fragility_floor(
        route,
        "carbon",
        baseline_margin_summary={"margin_stability_signal": 0.0},
        active_families=["carbon"],
        selector_weights=(1.0, 1.0, 1.0),
        ambiguity_context=ambiguity_context,
    )

    assert floor > 0.0
    assert floor <= 0.01


def test_supported_ambiguity_fallback_can_surface_controller_gain_without_empirical_vor() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={"carbon": {"time": 0.2, "money": 0.4, "co2": 1.0}},
        ),
        _route("route_b", objective=(20.0, 20.0, 20.0), evidence_tensor={}),
    ]
    worlds = [
        {"world_id": "w0", "states": {"carbon": "refreshed"}},
        {"world_id": "w1", "states": {"carbon": "refreshed"}},
        {"world_id": "w2", "states": {"carbon": "refreshed"}},
    ]
    ambiguity_context = {
        "od_ambiguity_index": 0.41,
        "od_hard_case_prior": 0.18,
        "od_engine_disagreement_prior": 0.22,
        "od_candidate_path_count": 4,
        "od_corridor_family_count": 3,
        "od_objective_spread": 0.26,
        "od_nominal_margin_proxy": 0.30,
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
        active_families=["carbon"],
        ambiguity_context=ambiguity_context,
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["carbon"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=ambiguity_context,
    )

    detail = fragility.route_fragility_details["route_a"]["carbon"]
    assert detail["raw_refresh_gain"] > 0.0
    assert detail["refresh_gain"] == pytest.approx(0.0)
    assert fragility.value_of_refresh["controller_ranking_basis"] == "raw_refresh_gain_fallback"
    assert fragility.value_of_refresh["top_refresh_family_controller"] == "carbon"
    assert fragility.value_of_refresh["top_refresh_gain_controller"] == pytest.approx(detail["raw_refresh_gain"])


def test_route_scope_by_family_alias_is_rewritten_for_route_scoped_counterfactuals() -> None:
    mixed_world = {
        "world_id": "w_alias",
        "states": {"scenario": "severely_stale", "weather": "severely_stale"},
        "world_kind": "hard_case_mixed_targeted",
        "route_scope_by_family": {"scenario": "route_b", "weather": "route_b"},
    }

    scoped_stressed = evidence_certification_module._stressed_worlds(
        [mixed_world],
        "scenario",
        stress_state="severely_stale",
        target_route_id="route_a",
    )
    scoped_refreshed = evidence_certification_module._refreshed_worlds(
        [mixed_world],
        "scenario",
        target_route_id="route_a",
    )

    assert scoped_stressed[0]["target_route_ids"]["scenario"] == "route_a"
    assert scoped_stressed[0]["target_route_ids"]["weather"] == "route_a"
    assert scoped_stressed[0]["route_scope_by_family"]["scenario"] == "route_a"
    assert scoped_refreshed[0]["target_route_ids"]["scenario"] == "route_a"
    assert scoped_refreshed[0]["target_route_ids"]["weather"] == "route_a"
    assert scoped_refreshed[0]["route_scope_by_family"]["scenario"] == "route_a"


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
    assert fragility.value_of_refresh["top_refresh_family_controller"] is None
    assert fragility.value_of_refresh["top_refresh_gain_controller"] == pytest.approx(0.0)


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


def test_refc_certification_frontier_rescues_supported_single_frontier_with_ranked_challengers() -> None:
    selected = _route_option(
        "route_a",
        duration_s=100.0,
        money_cost=20.0,
        co2_kg=5.0,
        std_duration_s=8.0,
        std_money_cost=2.0,
        std_co2_kg=0.6,
    ).model_copy(
        update={
            "uncertainty": {
                "std_duration_s": 8.0,
                "std_monetary_cost": 2.0,
                "std_emissions_kg": 0.6,
                "utility_mean": 125.0,
                "utility_q95": 132.0,
                "utility_cvar95": 138.0,
                "mean_duration_s": 100.0,
                "q95_duration_s": 108.0,
                "cvar95_duration_s": 112.0,
                "mean_monetary_cost": 20.0,
                "q95_monetary_cost": 22.0,
                "cvar95_monetary_cost": 23.0,
                "mean_emissions_kg": 5.0,
                "q95_emissions_kg": 5.6,
                "cvar95_emissions_kg": 5.9,
            }
        }
    )
    challenger_b = _route_option(
        "route_b",
        duration_s=103.0,
        money_cost=21.0,
        co2_kg=5.3,
        std_duration_s=2.0,
        std_money_cost=0.6,
        std_co2_kg=0.2,
    ).model_copy(
        update={
            "uncertainty": {
                "std_duration_s": 2.0,
                "std_monetary_cost": 0.6,
                "std_emissions_kg": 0.2,
                "utility_mean": 126.0,
                "utility_q95": 127.0,
                "utility_cvar95": 128.0,
                "mean_duration_s": 103.0,
                "q95_duration_s": 105.0,
                "cvar95_duration_s": 106.0,
                "mean_monetary_cost": 21.0,
                "q95_monetary_cost": 21.6,
                "cvar95_monetary_cost": 21.9,
                "mean_emissions_kg": 5.3,
                "q95_emissions_kg": 5.5,
                "cvar95_emissions_kg": 5.6,
            }
        }
    )
    challenger_c = _route_option(
        "route_c",
        duration_s=106.0,
        money_cost=21.4,
        co2_kg=5.5,
        std_duration_s=1.0,
        std_money_cost=0.4,
        std_co2_kg=0.15,
    ).model_copy(
        update={
            "uncertainty": {
                "std_duration_s": 1.0,
                "std_monetary_cost": 0.4,
                "std_emissions_kg": 0.15,
                "utility_mean": 127.0,
                "utility_q95": 127.8,
                "utility_cvar95": 128.4,
                "mean_duration_s": 106.0,
                "q95_duration_s": 107.0,
                "cvar95_duration_s": 107.6,
                "mean_monetary_cost": 21.4,
                "q95_monetary_cost": 21.8,
                "cvar95_monetary_cost": 22.0,
                "mean_emissions_kg": 5.5,
                "q95_emissions_kg": 5.65,
                "cvar95_emissions_kg": 5.75,
            }
        }
    )
    ambiguity_context = {
        "od_ambiguity_index": 0.26,
        "od_hard_case_prior": 0.36,
        "od_engine_disagreement_prior": 0.37,
        "od_ambiguity_support_ratio": 0.68,
        "od_ambiguity_source_entropy": 0.78,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix_count": 3,
        "ambiguity_budget_band": "high",
    }

    certification_frontier, metadata = main_module._refc_certification_frontier_options(
        pipeline_mode="dccs_refc",
        strict_frontier=[selected],
        options=[selected, challenger_b, challenger_c],
        selected=selected,
        ambiguity_context=ambiguity_context,
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="robust",
        risk_aversion=1.0,
    )

    assert [option.id for option in certification_frontier] == ["route_a", "route_b", "route_c"]
    assert metadata["certification_frontier_rescue_applied"] is True
    assert metadata["certification_frontier_rescue_reason"] == "single_frontier_supported_ambiguity_rescue"
    assert metadata["strict_frontier_route_ids"] == ["route_a"]
    assert metadata["certification_frontier_route_ids"] == ["route_a", "route_b", "route_c"]
    assert metadata["certification_frontier_rescue_added_route_ids"] == ["route_b", "route_c"]


def test_refc_certification_frontier_does_not_rescue_low_support_single_frontier() -> None:
    selected = _route_option("route_a", duration_s=100.0, money_cost=20.0, co2_kg=5.0)
    challenger = _route_option("route_b", duration_s=103.0, money_cost=21.0, co2_kg=5.3)

    certification_frontier, metadata = main_module._refc_certification_frontier_options(
        pipeline_mode="dccs_refc",
        strict_frontier=[selected],
        options=[selected, challenger],
        selected=selected,
        ambiguity_context={
            "od_ambiguity_index": 0.12,
            "od_hard_case_prior": 0.08,
            "od_engine_disagreement_prior": 0.06,
            "od_ambiguity_support_ratio": 0.22,
            "od_ambiguity_source_entropy": 0.12,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix_count": 1,
        },
        w_time=1.0,
        w_money=1.0,
        w_co2=1.0,
        optimization_mode="robust",
        risk_aversion=1.0,
    )

    assert [option.id for option in certification_frontier] == ["route_a"]
    assert metadata["certification_frontier_rescue_applied"] is False
    assert metadata["certification_frontier_rescue_reason"] == "ambiguity_support_insufficient"


def test_merge_controller_refresh_overlay_preserves_report_signal_and_restores_controller_fallback() -> None:
    report_fragility = main_module.FragilityResult(
        route_fragility_map={"route_0": {"fuel": 0.32}},
        competitor_fragility_breakdown={"route_0": {"route_1": {"fuel": 1.0}}},
        value_of_refresh={
            "top_refresh_family": "fuel",
            "top_refresh_gain": 0.32,
            "controller_ranking_basis": "empirical_vor",
            "controller_ranking": [
                {"family": "fuel", "controller_score": 0.32, "basis": "empirical_vor"}
            ],
            "top_refresh_family_controller": "fuel",
            "top_refresh_gain_controller": 0.32,
        },
        route_fragility_details={"route_0": {"fuel": {"raw_refresh_gain": 0.32}}},
        evidence_snapshot_manifest={"snapshot_hash": "report"},
    )
    controller_fragility = main_module.FragilityResult(
        route_fragility_map={"route_0": {"weather": 0.0}},
        competitor_fragility_breakdown={"route_0": {}},
        value_of_refresh={
            "controller_ranking_basis": "raw_refresh_gain_fallback",
            "controller_ranking": [
                {
                    "family": "weather",
                    "controller_score": 0.06,
                    "empirical_vor": 0.0,
                    "raw_refresh_gain": 0.06,
                    "basis": "raw_refresh_gain_fallback",
                }
            ],
            "top_refresh_family_controller": "weather",
            "top_refresh_gain_controller": 0.06,
            "empirical_baseline_certificate": 0.68,
            "controller_baseline_certificate": 0.52,
        },
        route_fragility_details={"route_0": {"weather": {"raw_refresh_gain": 0.06}}},
        evidence_snapshot_manifest={"snapshot_hash": "controller"},
    )

    merged = main_module._merge_controller_refresh_overlay(
        report_fragility=report_fragility,
        controller_fragility=controller_fragility,
        controller_frontier_route_ids=["route_0"],
        controller_frontier_mode="strict_frontier",
    )

    assert merged.route_fragility_map == report_fragility.route_fragility_map
    assert merged.competitor_fragility_breakdown == report_fragility.competitor_fragility_breakdown
    assert merged.value_of_refresh["top_refresh_family"] == "fuel"
    assert merged.value_of_refresh["top_refresh_gain"] == pytest.approx(0.32)
    assert merged.value_of_refresh["controller_ranking_basis"] == "raw_refresh_gain_fallback"
    assert merged.value_of_refresh["top_refresh_family_controller"] == "weather"
    assert merged.value_of_refresh["top_refresh_gain_controller"] == pytest.approx(0.06)
    assert merged.value_of_refresh["controller_refresh_frontier_mode"] == "strict_frontier"
    assert merged.value_of_refresh["controller_refresh_frontier_route_ids"] == ["route_0"]
    assert merged.value_of_refresh["controller_refresh_frontier_count"] == 1
    assert merged.value_of_refresh["controller_ranking"][0]["family"] == "weather"
    assert merged.value_of_refresh["controller_ranking"][0]["raw_refresh_gain"] == pytest.approx(0.06)
    assert merged.value_of_refresh["controller_baseline_certificate"] == pytest.approx(0.52)
    assert merged.value_of_refresh["empirical_baseline_certificate"] == pytest.approx(0.68)


def test_single_frontier_shortcut_certificate_value_is_conservative_for_supported_ambiguity() -> None:
    value = main_module._single_frontier_shortcut_certificate_value(
        requested_world_count=96,
        threshold=0.80,
        ambiguity_context={
            "od_ambiguity_index": 0.41,
            "od_hard_case_prior": 0.32,
            "od_engine_disagreement_prior": 0.36,
            "od_ambiguity_confidence": 0.94,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix_count": 3,
            "od_ambiguity_support_ratio": 0.82,
            "od_ambiguity_source_entropy": 0.77,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 2,
            "ambiguity_budget_prior": 0.44,
            "ambiguity_budget_band": "high",
        },
    )

    assert value < 1.0
    assert value == pytest.approx(0.75, abs=0.08)


def test_single_frontier_shortcut_certificate_value_stays_high_for_low_ambiguity_rows() -> None:
    requested_value = main_module._single_frontier_shortcut_certificate_value(
        requested_world_count=72,
        threshold=0.80,
        ambiguity_context={
            "od_ambiguity_index": 0.05,
            "od_hard_case_prior": 0.02,
            "od_engine_disagreement_prior": 0.03,
            "od_ambiguity_confidence": 0.20,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix_count": 1,
            "od_ambiguity_support_ratio": 0.08,
            "od_ambiguity_source_entropy": 0.05,
            "od_candidate_path_count": 1,
            "od_corridor_family_count": 1,
            "ambiguity_budget_prior": 0.04,
            "ambiguity_budget_band": "low",
        },
    )
    single_world_value = main_module._single_frontier_shortcut_certificate_value(
        requested_world_count=1,
        threshold=0.80,
        ambiguity_context={"od_ambiguity_index": 0.45},
    )

    assert requested_value >= 0.89
    assert single_world_value == pytest.approx(1.0)


def test_single_frontier_full_stress_rows_report_actual_world_count_and_apply_structural_cap() -> None:
    ambiguity_context = {
        "od_ambiguity_index": 0.26,
        "od_hard_case_prior": 0.36199,
        "od_engine_disagreement_prior": 0.362655,
        "od_ambiguity_confidence": 0.964,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix_count": 3,
        "od_ambiguity_source_mix": '{"historical_results_bootstrap":8,"repo_local_geometry_backfill":4,"routing_graph_probe":1}',
        "od_ambiguity_support_ratio": 0.64432,
        "od_ambiguity_source_entropy": 0.78166,
        "od_candidate_path_count": 4,
        "od_corridor_family_count": 1,
        "ambiguity_budget_prior": 0.26,
        "ambiguity_budget_band": "high",
    }
    route = _route_option(
        "route_a",
        duration_s=8270.89,
        money_cost=149.95,
        co2_kg=219.081,
        scenario_duration_multiplier=1.0,
        scenario_fuel_multiplier=1.0,
        scenario_emissions_multiplier=1.0,
        scenario_sigma_multiplier=1.0,
        std_duration_s=520.0,
        std_money_cost=4.5,
        std_co2_kg=12.0,
        active_families=("scenario", "toll", "terrain", "fuel", "carbon", "stochastic"),
    )

    certificate, fragility, manifest, active_families = main_module._compute_frontier_certification(
        frontier_options=[route],
        selected_route_id="route_a",
        run_seed=20260320,
        world_count=88,
        threshold=0.83,
        w_time=1.25,
        w_money=1.1,
        w_co2=1.3,
        optimization_mode="robust",
        risk_aversion=0.0,
        ambiguity_context=ambiguity_context,
    )
    baseline_cap = main_module._single_frontier_shortcut_certificate_value(
        requested_world_count=88,
        threshold=0.83,
        ambiguity_context=ambiguity_context,
    )

    assert main_module._refc_requires_full_stress_worlds(ambiguity_context) is True
    assert active_families == ["carbon", "fuel", "scenario", "stochastic", "terrain", "toll"]
    assert manifest["world_count"] > 1
    assert manifest["effective_world_count"] == manifest["world_count"]
    assert manifest["world_count_policy"] == "single_frontier_full_stress"
    assert manifest["single_frontier_certificate_cap_applied"] is True
    assert manifest["selected_certificate_basis"] == "single_frontier_structural_cap"
    assert manifest["single_frontier_empirical_certificate"] == pytest.approx(1.0)
    assert manifest["single_frontier_observed_coverage_ratio"] > 0.5
    assert manifest["single_frontier_observed_coverage_relief"] > 0.0
    assert certificate.certificate["route_a"] == pytest.approx(manifest["single_frontier_certificate_cap"])
    assert certificate.certificate["route_a"] > baseline_cap
    assert certificate.certificate["route_a"] < 0.83
    assert certificate.certified is False
    assert certificate.selector_config["selected_certificate_basis"] == "single_frontier_structural_cap"
    assert fragility.value_of_refresh["baseline_certificate"] == pytest.approx(1.0)
    assert fragility.value_of_refresh["controller_baseline_certificate"] == pytest.approx(
        certificate.certificate["route_a"]
    )


def test_single_frontier_full_stress_can_force_requested_sampler_world_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambiguity_context = {
        "od_ambiguity_index": 0.32,
        "od_hard_case_prior": 0.363736,
        "od_engine_disagreement_prior": 0.370435,
        "od_ambiguity_confidence": 0.910497,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_source_mix_count": 3,
        "od_ambiguity_source_mix": '{"historical_results_bootstrap":4,"repo_local_geometry_backfill":4,"routing_graph_probe":1}',
        "od_ambiguity_support_ratio": 0.818061,
        "od_ambiguity_source_entropy": 0.878347,
        "od_candidate_path_count": 14,
        "od_corridor_family_count": 1,
        "ambiguity_budget_prior": 0.32,
        "ambiguity_budget_band": "high",
    }
    route = _route_option(
        "route_a",
        duration_s=21364.43,
        money_cost=320.11,
        co2_kg=450.345,
        distance_km=320.716,
        scenario_duration_multiplier=1.0,
        scenario_fuel_multiplier=1.0,
        scenario_emissions_multiplier=1.0,
        scenario_sigma_multiplier=1.0,
        std_duration_s=520.0,
        std_money_cost=4.5,
        std_co2_kg=12.0,
        active_families=("scenario", "toll", "terrain", "fuel", "carbon", "stochastic"),
    )
    sampled_counts: list[int] = []

    def _fake_sample_world_manifest(
        *,
        active_families,
        seed,
        world_count,
        state_catalog,
        routes,
        ambiguity_context,
        selector_weights,
    ):
        sampled_counts.append(int(world_count))
        worlds = [
            {
                "world_id": f"w{i}",
                "states": {family: "nominal" for family in active_families},
            }
            for i in range(int(world_count))
        ]
        return {
            "status": "sampled",
            "seed": int(seed),
            "requested_world_count": int(world_count),
            "world_count": int(world_count),
            "effective_world_count": int(world_count),
            "world_count_policy": "configured",
            "unique_world_count": int(world_count),
            "active_families": list(active_families),
            "worlds": worlds,
            "hard_case_stress_pack_count": 0,
            "world_reuse_rate": 0.0,
            "stress_world_fraction": 0.35 if int(world_count) > 1 else 0.0,
        }

    monkeypatch.setattr(main_module, "sample_world_manifest", _fake_sample_world_manifest)

    default_certificate, _, default_manifest, _ = main_module._compute_frontier_certification(
        frontier_options=[route],
        selected_route_id="route_a",
        run_seed=20260320,
        world_count=88,
        threshold=0.85,
        w_time=1.3,
        w_money=1.15,
        w_co2=1.4,
        optimization_mode="robust",
        risk_aversion=1.0,
        ambiguity_context=ambiguity_context,
    )
    forced_certificate, _, forced_manifest, _ = main_module._compute_frontier_certification(
        frontier_options=[route],
        selected_route_id="route_a",
        run_seed=20260320,
        world_count=88,
        threshold=0.85,
        w_time=1.3,
        w_money=1.15,
        w_co2=1.4,
        optimization_mode="robust",
        risk_aversion=1.0,
        ambiguity_context=ambiguity_context,
        force_single_frontier_full_stress_requested_worlds=True,
    )

    assert sampled_counts == [1, 88]
    assert default_manifest["sampler_requested_world_count"] == 1
    assert forced_manifest["sampler_requested_world_count"] == 88
    assert forced_manifest["requested_world_count"] == 88
    assert forced_manifest["effective_world_count"] == 88
    assert forced_manifest["world_count_policy"] == "single_frontier_full_stress"
    assert forced_manifest["single_frontier_observed_coverage_relief"] > 0.0
    assert forced_certificate.certificate["route_a"] > default_certificate.certificate["route_a"]
    assert forced_certificate.certificate["route_a"] < 0.85


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


def test_winner_confidence_sequence_bounds_tighten_for_consistent_winner() -> None:
    states = winner_confidence_sequence(
        "route_a",
        [1, 1, 1, 1, 1, 1, 1, 1],
        threshold=0.7,
        confidence_level=0.95,
        support_strength=0.9,
        proxy_fraction=0.0,
    )

    assert states
    for state in states:
        assert 0.0 <= state.lower_bound <= state.point_estimate <= state.upper_bound <= 1.0

    assert states[-1].lower_bound >= states[0].lower_bound
    assert states[-1].width <= states[0].width
    assert states[-1].effective_sample_count > states[0].effective_sample_count


def test_pairwise_gap_state_from_score_maps_tracks_gap_statistics() -> None:
    state = PairwiseGapState.from_score_maps(
        "route_a",
        "route_b",
        [
            {"route_a": 10.0, "route_b": 10.4},
            {"route_a": 10.1, "route_b": 10.2},
            {"route_a": 10.2, "route_b": 10.2},
            {"route_a": 10.3, "route_b": 10.1},
        ],
        support_strength=0.85,
    )

    assert state.sample_count == 4
    assert state.min_gap <= state.mean_gap <= state.max_gap
    assert state.positive_share + state.negative_share + state.tie_share == pytest.approx(1.0, abs=1e-6)
    assert state.challenger_win_share == state.negative_share


def test_certified_set_marks_safe_certified_winner() -> None:
    threshold = 0.7
    winner = WinnerConfidenceState.from_point_estimate(
        "route_a",
        point_estimate=0.86,
        sample_count=400,
        threshold=threshold,
        support_strength=0.95,
    )
    challenger = WinnerConfidenceState.from_point_estimate(
        "route_b",
        point_estimate=0.58,
        sample_count=400,
        threshold=threshold,
        support_strength=0.95,
    )
    pairwise_gap = PairwiseGapState.from_certificate_gap(
        "route_a",
        "route_b",
        winner_certificate=0.86,
        challenger_certificate=0.58,
        sample_count=400,
        support_strength=0.95,
    )
    flip_radius = FlipRadiusState.from_pairwise_gap(
        pairwise_gap,
        objective_scale=max(abs(pairwise_gap.mean_gap), 1e-6),
    )
    decision_region = DecisionRegionState.from_states(
        winner,
        [challenger],
        [pairwise_gap],
        flip_radius,
        threshold=threshold,
    )
    certified_set = CertifiedSetState.from_confidence_states(
        [winner, challenger],
        threshold=threshold,
        winner_id="route_a",
        decision_region=decision_region,
    )

    assert decision_region.certified
    assert certified_set.safe
    assert certified_set.certified_route_ids == ("route_a",)
    assert certified_set.rejected_route_ids == ("route_b",)


def test_abstention_record_reflects_decision_region_reason() -> None:
    threshold = 0.7
    winner = WinnerConfidenceState.from_point_estimate(
        "route_a",
        point_estimate=0.68,
        sample_count=40,
        threshold=threshold,
        support_strength=0.4,
        proxy_fraction=0.2,
    )
    challenger = WinnerConfidenceState.from_point_estimate(
        "route_b",
        point_estimate=0.66,
        sample_count=40,
        threshold=threshold,
        support_strength=0.4,
        proxy_fraction=0.2,
    )
    pairwise_gap = PairwiseGapState.from_certificate_gap(
        "route_a",
        "route_b",
        winner_certificate=0.68,
        challenger_certificate=0.66,
        sample_count=40,
        support_strength=0.4,
        proxy_fraction=0.2,
    )
    flip_radius = FlipRadiusState.from_pairwise_gap(
        pairwise_gap,
        objective_scale=max(abs(pairwise_gap.mean_gap), 1e-6),
    )
    decision_region = DecisionRegionState.from_states(
        winner,
        [challenger],
        [pairwise_gap],
        flip_radius,
        threshold=threshold,
    )
    abstention = AbstentionRecord.from_decision_region(decision_region)

    assert decision_region.abstain
    assert abstention.reason_code in ABSTENTION_REASON_CODES
    assert abstention.recommended_action
    assert abstention.severity == "high"


def test_certification_state_builds_consistent_witness() -> None:
    state = CertificationState.from_refc_outputs(
        certificate={"route_a": 0.9, "route_b": 0.52, "route_c": 0.33},
        threshold=0.7,
        world_manifest={
            "manifest_hash": "manifest-1",
            "selected_route_id": "route_a",
            "active_families": ["scenario", "weather"],
            "world_count": 400,
            "unique_world_count": 400,
            "requested_world_count": 400,
            "effective_world_count": 400,
            "world_count_policy": "targeted_stress",
            "stress_world_fraction": 0.25,
            "worlds": [
                {
                    "world_kind": "sampled",
                    "states": {"scenario": "nominal", "weather": "nominal"},
                    "target_route_id": "route_a",
                },
                {
                    "world_kind": "sampled",
                    "states": {"scenario": "nominal", "weather": "refreshed"},
                },
                {
                    "world_kind": "stress",
                    "states": {"scenario": "stress", "weather": "nominal"},
                },
                {
                    "world_kind": "sampled",
                    "states": {"scenario": "nominal", "weather": "nominal"},
                },
            ],
        },
        fragility={"controller_ranking_basis": "refc"},
        evidence_snapshot_manifest={
            "manifest_hash": "snapshot-1",
            "family_snapshots": {
                "scenario": [
                    {
                        "source": "live_scenario",
                        "signature": "live",
                        "confidence": 0.98,
                        "coverage_ratio": 1.0,
                        "fallback_used": False,
                    }
                ],
                "weather": [
                    {
                        "source": "live_weather",
                        "signature": "live",
                        "confidence": 0.97,
                        "coverage_ratio": 0.96,
                        "fallback_used": False,
                    }
                ],
            },
        },
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.85,
            "od_ambiguity_confidence": 0.9,
            "od_ambiguity_source_entropy": 0.6,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_family_density": 0.7,
            "od_ambiguity_margin_pressure": 0.55,
            "od_nominal_margin_proxy": 0.75,
        },
        evidence_validation={
            "freshness_coverage": 0.95,
            "live_family_count": 2,
            "snapshot_family_count": 0,
            "model_family_count": 0,
        },
    )
    witness = CertificateWitness.from_state(
        state,
        fragility={"top_refresh_family": "weather"},
    )

    assert state.decision_region.certified
    assert state.certified
    assert witness.is_consistent_with_state(state)
    assert witness.best_challenger_id == state.decision_region.best_challenger_id
    assert witness.top_fragility_family == "weather"
    assert witness.recommended_action == "hold"
