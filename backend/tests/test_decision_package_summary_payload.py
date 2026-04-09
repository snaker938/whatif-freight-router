from __future__ import annotations

import json

from app.models import (
    DecisionPackage,
    GeoJSONLineString,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    RouteResponse,
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
            "member_route_ids": ["route-a"],
            "excluded_route_ids": ["route-b"],
            "exclusion_basis": ["certificate_threshold", "frontier_selection"],
            "certified": True,
            "threshold": 0.7,
            "support_flag": True,
            "set_size": 1,
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
    assert encoded["certified_set_summary"]["member_route_ids"] == ["route-a"]
    assert encoded["certified_set_summary"]["witness"]["active_challenger_ids"] == ["route-b"]
    assert encoded["preference_summary"]["compatible_set_summary"]["compatible_set_size"] == 2
    assert encoded["preference_summary"]["derived_invariants"]["necessary_best_prob_le_possible_best_prob"] is True
    assert encoded["stability_summary"]["fragility_summary"]["route_id"] == "route-a"


def test_route_response_preserves_summary_surfaces_and_default_artifact_pointers() -> None:
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

    payload = RouteResponse(
        selected=selected,
        candidates=[selected, challenger],
        selected_certificate=selected_certificate,
        world_support_summary={
            "support_flag": True,
            "active_families": ["scenario", "weather"],
            "calibration_bin": "bin_1",
        },
        manifest_endpoint="/runs/run-thesis/manifest",
        artifacts_endpoint="/runs/run-thesis/artifacts",
        provenance_endpoint="/runs/run-thesis/provenance",
    )

    encoded = json.loads(payload.model_dump_json())

    assert encoded["terminal_type"] == "certified_singleton"
    assert encoded["world_support_summary"]["support_flag"] is True
    assert encoded["world_support_summary"]["calibration_bin"] == "bin_1"
    assert encoded["action_trace_summary"] == {
        "pipeline_mode": "legacy",
        "selected_candidate_count": 2,
    }
    assert encoded["witness_summary"] == {
        "route_id": "route-a",
        "selected_certificate_basis": None,
    }
    assert encoded["artifact_pointers"] == {
        "manifest_endpoint": "/runs/run-thesis/manifest",
        "artifacts_endpoint": "/runs/run-thesis/artifacts",
        "provenance_endpoint": "/runs/run-thesis/provenance",
    }
    assert encoded["selected_certificate_basis"] == "selected_certificate"
    assert encoded["preference_query_trace"]["selected_certificate_basis"] == "selected_certificate"
    assert encoded["certified_set_summary"]["member_route_ids"] == ["route-a"]
    assert encoded["certified_set_summary"]["excluded_route_ids"] == ["route-b"]
