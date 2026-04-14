from __future__ import annotations

import json

from app.abstention import build_abstention_record
from app.decision_region import DecisionRegionState
from app.flip_radius import FlipRadiusState, build_structured_adversarial_budget
from app.settings import Settings, settings
from app.main import _apply_structured_adversarial_budget_channels, _assemble_decision_package
from app.models import (
    DecisionPackage,
    GeoJSONLineString,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
)
from app.preference_model import build_preference_state
from app.risk_model import (
    RiskSummary,
    build_fragility_summary,
    normalized_objective_components,
    normalized_weighted_utility,
    robust_objective,
)
from app.models import ScenarioSummary
from app.support_model import build_world_support_state
from app.uncertainty_model import build_world_bundle_summary


def _make_route(route_id: str, *, duration_s: float, money: float, co2: float) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(-1.0, 52.0), (-0.1, 51.5)]),
        metrics=RouteMetrics(
            distance_km=10.0,
            duration_s=duration_s,
            monetary_cost=money,
            emissions_kg=co2,
            avg_speed_kmh=40.0,
        ),
    )


def test_decision_package_accepts_richer_support_and_preference_summaries() -> None:
    support_state = build_world_support_state(
        support_score=0.9,
        support_ratio=0.4,
        support_bin="in_support",
        calibration_bin="bin_1",
        support_source="world_manifest",
    )
    world_bundle_summary = build_world_bundle_summary(
        manifest={
            "world_count": 3,
            "unique_world_count": 2,
            "active_families": ["scenario", "weather"],
            "state_catalog": ["nominal", "proxy"],
            "state_weights": {"scenario": {"time": 0.7}},
            "worlds": [{"world_id": "w1"}],
            "world_reuse_rate": 0.5,
        },
        support_state=support_state,
    )
    preference_state = build_preference_state(
        route_ids=["route-a", "route-b"],
        weights={"time": 1.0, "money": 0.0, "co2": 0.0},
        support_flag=True,
        support_reason=None,
    )
    preference_state.terminal_type = "certified"
    scenario_summary = ScenarioSummary(
        mode="no_sharing",
        context_key="uk_default|mixed|rigid_hgv|weekday|clear",
        duration_multiplier=1.0,
        incident_rate_multiplier=1.0,
        incident_delay_multiplier=1.0,
        fuel_consumption_multiplier=1.0,
        emissions_multiplier=1.0,
        stochastic_sigma_multiplier=1.0,
        source="fixture",
        version="v1",
        calibration_basis="empirical",
    )
    normalized_duration_component, normalized_monetary_component, normalized_emissions_component = (
        normalized_objective_components(
            duration_s=100.0,
            monetary_cost=20.0,
            emissions_kg=5.0,
            distance_km=12.0,
        )
    )
    weighted_utility = normalized_weighted_utility(
        duration_s=100.0,
        monetary_cost=20.0,
        emissions_kg=5.0,
        distance_km=12.0,
        utility_weights=(1.0, 1.0, 1.0),
    )
    risk_summary = RiskSummary(
        mean_value=weighted_utility,
        cvar_value=None,
        robust_score=robust_objective(
            mean_value=weighted_utility,
            cvar_value=None,
            risk_aversion=1.0,
        ),
        normalized_duration_component=normalized_duration_component,
        normalized_monetary_component=normalized_monetary_component,
        normalized_emissions_component=normalized_emissions_component,
        support_state=support_state,
        probabilistic_world_bundle=world_bundle_summary.probabilistic_world_bundle,
        audit_world_bundle=world_bundle_summary.audit_world_bundle,
    )
    fragility_summary = build_fragility_summary(
        route_id="route-a",
        deterministic_local_flip_radius=0.25,
        probabilistic_flip_radius=0.25,
        challenger_specific_radii={"route-b": 0.2},
        evidence_family_radii={"weather": 0.1},
        dominant_fragility_family="weather",
        support_flag=True,
    )

    payload = DecisionPackage(
        terminal_type="certified_singleton",
        certified_set_summary={
            "member_route_ids": [],
            "excluded_route_ids": ["route-b"],
            "exclusion_basis": [],
            "certified": False,
            "threshold": 0.7,
            "support_flag": True,
            "set_size": 0,
            "terminal_type": "certified_singleton",
            "not_applicable_reason": "singleton_terminal",
            "selected_route_id": "route-a",
            "witness": {
                "route_id": "route-a",
                "active_challenger_ids": ["route-b"],
                "support_flag": True,
            },
        },
        support_summary={
            "support_flag": support_state.support_flag,
            "world_bundle_summary": world_bundle_summary.as_dict(),
            "support_state": support_state.as_dict(),
            "scenario_summary": scenario_summary.model_dump(mode="json"),
            "risk_summary": risk_summary.as_dict(),
        },
        preference_summary={
            "selected_certificate_basis": "empirical",
            "pipeline_mode": "voi",
            "preference_state": preference_state.model_dump(mode="json"),
            "compatible_set_summary": preference_state.compatible_set_summary.model_dump(mode="json"),
            "derived_invariants": dict(preference_state.derived_invariants),
            "query_count": int(preference_state.query_count),
        },
        stability_summary={
            "fragility_summary": fragility_summary.as_dict(),
            "risk_summary": risk_summary.as_dict(),
        },
    )

    encoded = json.loads(payload.model_dump_json())
    assert encoded["support_summary"]["support_state"]["support_flag"] is True
    assert encoded["support_summary"]["world_bundle_summary"]["support_state"]["support_flag"] is True
    assert encoded["certified_set_summary"]["member_route_ids"] == []
    assert encoded["certified_set_summary"]["certified"] is False
    assert encoded["certified_set_summary"]["set_size"] == 0
    assert encoded["certified_set_summary"]["terminal_type"] == "certified_singleton"
    assert encoded["certified_set_summary"]["not_applicable_reason"] == "singleton_terminal"
    assert encoded["certified_set_summary"]["witness"]["active_challenger_ids"] == ["route-b"]
    assert encoded["preference_summary"]["compatible_set_summary"]["compatible_set_size"] == 2
    assert encoded["preference_summary"]["derived_invariants"]["necessary_best_prob_le_possible_best_prob"] is True
    assert encoded["stability_summary"]["fragility_summary"]["route_id"] == "route-a"


def test_assembled_decision_package_preserves_summary_surfaces_and_default_artifact_pointers() -> None:
    selected = _make_route("route-a", duration_s=101.0, money=20.0, co2=5.0)
    challenger = _make_route("route-b", duration_s=106.0, money=22.0, co2=5.5)
    selected_certificate = RouteCertificationSummary(
        route_id=selected.id,
        certificate=0.86,
        certified=True,
        threshold=0.8,
        active_families=["scenario", "weather"],
        top_fragility_families=["weather"],
    )

    payload = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-thesis",
        pipeline_mode="dccs_refc",
        certified_set=[selected],
        selected_certificate=selected_certificate,
        voi_stop_summary=None,
        preference_state=build_preference_state(
            route_ids=[selected.id, challenger.id],
            weights={"time": 1.0, "money": 0.0, "co2": 0.0},
            support_flag=True,
            support_reason=None,
        ),
        preference_query_trace={},
        world_support_summary={
            "support_flag": True,
            "active_families": ["scenario", "weather"],
            "calibration_bin": "bin_1",
        },
        manifest_endpoint="/runs/run-thesis/manifest",
        artifacts_endpoint="/runs/run-thesis/artifacts",
        provenance_endpoint="/runs/run-thesis/provenance",
        abstention=None,
    )

    encoded = json.loads(payload.model_dump_json())

    assert encoded["terminal_type"] == "certified_singleton"
    assert encoded["selected"]["id"] == "route-a"
    assert encoded["recommended_route"]["id"] == "route-a"
    assert encoded["certified_set"] == []
    assert encoded["world_support_summary"]["support_flag"] is True
    assert encoded["world_support_summary"]["calibration_bin"] == "bin_1"
    assert encoded["action_trace_summary"] == {
        "pipeline_mode": "dccs_refc",
        "selected_candidate_count": 2,
    }
    assert encoded["witness_summary"] == {
        "route_id": "route-a",
        "selected_certificate_basis": "selected_certificate",
    }
    assert encoded["artifact_pointers"] == {
        "manifest_endpoint": "/runs/run-thesis/manifest",
        "artifacts_endpoint": "/runs/run-thesis/artifacts",
        "provenance_endpoint": "/runs/run-thesis/provenance",
    }
    assert encoded["selected_certificate_basis"] == "selected_certificate"
    assert encoded["preference_query_trace"]["selected_certificate_basis"] == "selected_certificate"
    assert encoded["certified_set_summary"]["member_route_ids"] == []
    assert encoded["certified_set_summary"]["excluded_route_ids"] == ["route-b"]
    assert encoded["certified_set_summary"]["certified"] is False
    assert encoded["certified_set_summary"]["set_size"] == 0
    assert encoded["certified_set_summary"]["terminal_type"] == "certified_singleton"
    assert encoded["certified_set_summary"]["not_applicable_reason"] == "singleton_terminal"


def test_assembled_decision_package_emits_typed_abstention_summary_contract() -> None:
    selected = _make_route("route-a", duration_s=101.0, money=20.0, co2=5.0)
    challenger = _make_route("route-b", duration_s=106.0, money=22.0, co2=5.5)
    abstention = build_abstention_record(
        stop_reason="search_incomplete_no_action_worth_it",
        support_flag=False,
        support_reason="out_of_support_world_model",
        credible_search_uncertainty=True,
        evidence_family="weather",
        budget_channel="search_budget",
        model_assumption="stationary_world_assumption",
        active_families=["scenario", "weather"],
        top_fragility_families=["weather"],
        detail={
            "winner_confidence_state": {
                "route_id": selected.id,
                "lower_bound": 0.41,
                "threshold": 0.8,
            }
        },
    )

    payload = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-thesis",
        pipeline_mode="dccs_refc",
        certified_set=[selected, challenger],
        selected_certificate=None,
        voi_stop_summary=None,
        preference_state=build_preference_state(
            route_ids=[selected.id, challenger.id],
            weights={"time": 1.0, "money": 0.0, "co2": 0.0},
            support_flag=False,
            support_reason="out_of_support_world_model",
        ),
        preference_query_trace={},
        world_support_summary={
            "support_flag": False,
            "support_reason": "out_of_support_world_model",
            "active_families": ["scenario", "weather"],
        },
        manifest_endpoint="/runs/run-thesis/manifest",
        artifacts_endpoint="/runs/run-thesis/artifacts",
        provenance_endpoint="/runs/run-thesis/provenance",
        abstention=abstention,
    )

    encoded = json.loads(payload.model_dump_json())

    assert encoded["terminal_type"] == "typed_abstention"
    assert encoded["abstention_summary"] == {
        "reason_code": abstention.reason_code,
        "message": abstention.message,
        "terminal_type": "typed_abstention",
        "has_typed_abstention": True,
        "detail": {
            "winner_confidence_state": {
                "route_id": selected.id,
                "lower_bound": 0.41,
                "threshold": 0.8,
            },
            "stop_reason": "search_incomplete_no_action_worth_it",
            "support_reason": "out_of_support_world_model",
            "search_completeness_score": None,
            "search_completeness_gap": None,
            "active_families": ["scenario", "weather"],
            "top_fragility_families": ["weather"],
        },
        "support_flag": False,
        "evidence_family": "weather",
        "budget_channel": "search_budget",
        "model_assumption": "stationary_world_assumption",
    }
    assert encoded["artifact_pointers"] == {
        "manifest_endpoint": "/runs/run-thesis/manifest",
        "artifacts_endpoint": "/runs/run-thesis/artifacts",
        "provenance_endpoint": "/runs/run-thesis/provenance",
    }
    assert encoded["selected_certificate_basis"] is None


def test_assembled_decision_package_surfaces_three_channel_structured_adversarial_budget() -> None:
    selected = _make_route("route-a", duration_s=101.0, money=20.0, co2=5.0)
    challenger = _make_route("route-b", duration_s=106.0, money=22.0, co2=5.5)
    selected_certificate = RouteCertificationSummary(
        route_id=selected.id,
        certificate=0.86,
        certified=True,
        threshold=0.8,
        active_families=["scenario", "weather"],
        top_fragility_families=["weather"],
    )
    preference_state = build_preference_state(
        route_ids=[selected.id, challenger.id],
        weights={"time": 1.0, "money": 0.0, "co2": 0.0},
        support_flag=True,
        support_reason=None,
    )
    preference_state.compatible_set_summary.necessary_best_prob = 0.45
    preference_state.compatible_set_summary.possible_best_prob = 0.85

    base_budget = build_structured_adversarial_budget(
        evidence_budget=0.19,
        evidence_driver="weather",
        evidence_source_metric="evidence_family_radii",
        evidence_details={"family_count": 2},
        preference_budget=None,
        preference_driver=None,
        preference_source_metric="possible_best_minus_necessary_best_probability",
        preference_details={},
        search_deficiency_budget=None,
        search_deficiency_driver=None,
        search_deficiency_source_metric="search_completeness_gap",
        search_deficiency_details={},
        provenance={"selected_route_id": selected.id},
    )
    flip_radius_state = FlipRadiusState(
        route_id=selected.id,
        deterministic_local_flip_radius=0.81,
        probabilistic_flip_radius=0.74,
        evidence_family_radii={"weather": 0.19, "terrain": 0.31},
        dominant_fragility_family="weather",
        minimum_flip_budget=0.19,
        structured_adversarial_budget=base_budget,
        support_flag=True,
    )
    decision_region_state = DecisionRegionState(
        route_id=selected.id,
        nearest_certificate_boundary="preference",
        most_fragile_preference_direction="guard:time_preserving",
        structured_adversarial_budget=base_budget,
        support_flag=True,
    )
    flip_radius_state, decision_region_state = _apply_structured_adversarial_budget_channels(
        flip_radius_state=flip_radius_state,
        decision_region_state=decision_region_state,
        preference_state=preference_state,
        search_completeness_gap=0.27,
    )

    payload = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-thesis",
        pipeline_mode="voi",
        certified_set=[selected],
        selected_certificate=selected_certificate,
        winner_confidence_state=None,
        pairwise_gap_states=[],
        flip_radius_state=flip_radius_state,
        decision_region_state=decision_region_state,
        certificate_witness=None,
        voi_stop_summary=None,
        preference_state=preference_state,
        preference_query_trace={},
        world_support_summary={
            "support_flag": True,
            "active_families": ["scenario", "weather"],
            "calibration_bin": "bin_1",
            "selected_certificate_basis": "selected_certificate",
        },
        manifest_endpoint="/runs/run-thesis/manifest",
        artifacts_endpoint="/runs/run-thesis/artifacts",
        provenance_endpoint="/runs/run-thesis/provenance",
        abstention=None,
    )

    encoded = json.loads(payload.model_dump_json())
    budget = encoded["decision_region_state"]["structured_adversarial_budget"]

    assert budget["evidence_channel"]["budget"] == 0.19
    assert budget["preference_channel"]["budget"] == 0.4
    assert budget["search_deficiency_channel"]["budget"] == 0.27
    assert budget["limiting_channel"] == "evidence"
    assert (
        encoded["flip_radius_state"]["structured_adversarial_budget"]["search_deficiency_channel"]["budget"]
        == 0.27
    )
    assert (
        encoded["certificate_summary"]["structured_adversarial_budget"]["preference_channel"]["budget"]
        == 0.4
    )


def test_settings_accept_route_refc_certified_set_cap_alias(monkeypatch) -> None:
    monkeypatch.delenv("ROUTE_REFC_LOW_AMBIGUITY_WORLD_CAP", raising=False)
    monkeypatch.setenv("ROUTE_REFC_CERTIFIED_SET_CAP", "31")

    cfg = Settings(_env_file=None)

    assert cfg.route_refc_low_ambiguity_world_cap == 31


def test_assembled_decision_package_surfaces_threshold_sensitivity_axes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "route_graph_fast_path_max_ambiguity", 0.18, raising=False)
    monkeypatch.setattr(settings, "route_refc_low_ambiguity_world_cap", 17, raising=False)

    selected = _make_route("route-a", duration_s=100.0, money=20.0, co2=5.0)
    challenger = _make_route("route-b", duration_s=108.0, money=23.0, co2=5.5)
    selected_certificate = RouteCertificationSummary(
        route_id=selected.id,
        certificate=0.86,
        certified=True,
        threshold=0.8,
        active_families=["scenario"],
        top_fragility_families=["scenario"],
    )

    payload = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-threshold-axis",
        pipeline_mode="dccs_refc",
        certified_set=[selected],
        selected_certificate=selected_certificate,
        winner_confidence_state=None,
        pairwise_gap_states=[],
        flip_radius_state=None,
        decision_region_state=None,
        certificate_witness=None,
        voi_stop_summary=None,
        preference_state={},
        preference_query_trace={},
        world_support_summary={
            "support_flag": True,
            "selected_certificate_basis": "selected_certificate",
        },
        world_manifest={
            "world_count": 17,
            "requested_world_count": 64,
            "effective_world_count": 17,
            "world_count_policy": "adaptive_low_ambiguity",
        },
        manifest_endpoint="/runs/run-threshold-axis/manifest",
        artifacts_endpoint="/runs/run-threshold-axis/artifacts",
        provenance_endpoint="/runs/run-threshold-axis/provenance",
        abstention=None,
    )

    encoded = json.loads(payload.model_dump_json())
    axes = encoded["certificate_summary"]["threshold_sensitivity_axes"]

    assert axes["certificate_threshold"]["configured_value"] == 0.8
    assert axes["certificate_threshold"]["request_field"] == "certificate_threshold"
    assert axes["fast_path_max_ambiguity"]["configured_value"] == 0.18
    assert axes["fast_path_max_ambiguity"]["env_field"] == "ROUTE_GRAPH_FAST_PATH_MAX_AMBIGUITY"
    assert axes["certified_set_cap"]["configured_value"] == 17
    assert axes["certified_set_cap"]["is_alias"] is True
    assert axes["certified_set_cap"]["env_alias"] == "ROUTE_REFC_CERTIFIED_SET_CAP"
    assert axes["certified_set_cap"]["mapped_env_field"] == "ROUTE_REFC_LOW_AMBIGUITY_WORLD_CAP"
    assert axes["certified_set_cap"]["truthful_semantics"] == "low_ambiguity_adaptive_refc_world_count_cap"
    assert axes["certified_set_cap"]["directly_caps_certified_set_cardinality"] is False
    assert axes["certified_set_cap"]["active_world_count_policy"] == "adaptive_low_ambiguity"
    assert axes["certified_set_cap"]["requested_world_count"] == 64
    assert axes["certified_set_cap"]["effective_world_count"] == 17
