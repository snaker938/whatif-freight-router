from __future__ import annotations

import json

import pytest

from app.abstention import build_abstention_record
from app.evidence_certification import (
    compute_certificate,
    compute_fragility_maps,
    project_refc_scaffold_states,
)
from app.main import _assemble_decision_package, _route_terminal_fields
from app.models import (
    DecisionPackage,
    GeoJSONLineString,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    VoiStopSummary,
)
from app.preference_model import build_preference_state
from app.preference_queries import PairwisePreferenceQuery
from app.preference_update import append_preference_query


def _route(route_id: str) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(0.0, 0.0), (1.0, 1.0)]),
        metrics=RouteMetrics(
            distance_km=10.0,
            duration_s=20.0,
            monetary_cost=30.0,
            emissions_kg=4.0,
            avg_speed_kmh=50.0,
        ),
    )


def _response(
    *,
    selected_certificate: RouteCertificationSummary | None,
    voi_stop_summary: VoiStopSummary | None,
    strict_frontier: list[RouteOption],
    support_flag: bool | None = None,
    support_reason: str | None = None,
) -> DecisionPackage:
    selected = strict_frontier[0]
    certified_set, abstention = _route_terminal_fields(
        selected_certificate=selected_certificate,
        voi_stop_summary=voi_stop_summary,
        strict_frontier=strict_frontier,
        support_flag=support_flag,
        support_reason=support_reason,
    )
    world_support_summary = {}
    if support_flag is not None or support_reason is not None:
        world_support_summary = {
            "schema_version": "world-support-summary-v1",
            "support_flag": support_flag,
            "support_reason": support_reason,
            "support_state": {
                "support_flag": support_flag,
                "out_of_support_reason": support_reason,
            },
        }
    response = _assemble_decision_package(
        selected=selected,
        candidates=list(strict_frontier),
        run_id="run-1",
        pipeline_mode="voi",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
        selected_certificate=selected_certificate,
        voi_stop_summary=voi_stop_summary,
        preference_state={},
        preference_query_trace={},
        world_support_summary=world_support_summary,
        certified_set=certified_set,
        abstention=abstention,
    )
    return response


@pytest.mark.parametrize(
    "selected_certificate, voi_stop_summary, strict_frontier, expected_terminal_type, expected_public_set_size, expected_summary_set_size, expected_reason",
    [
        (
            RouteCertificationSummary(
                route_id="route-a",
                certificate=0.92,
                certified=True,
                threshold=0.70,
                active_families=["scenario"],
                top_fragility_families=["weather"],
                top_competitor_route_id="route-b",
                top_value_of_refresh_family="weather",
                ambiguity_context={"support_strength": True},
            ),
            VoiStopSummary(
                final_route_id="route-a",
                certificate=0.92,
                certified=True,
                iteration_count=1,
                search_budget_used=1,
                evidence_budget_used=0,
                stop_reason="certified",
            ),
            [_route("route-a")],
            "certified_singleton",
            0,
            0,
            None,
        ),
        (
            RouteCertificationSummary(
                route_id="route-a",
                certificate=0.92,
                certified=True,
                threshold=0.70,
                active_families=["scenario"],
                top_fragility_families=["weather"],
                top_competitor_route_id="route-b",
                top_value_of_refresh_family="weather",
                ambiguity_context={"support_strength": True},
            ),
            VoiStopSummary(
                final_route_id="route-a",
                certificate=0.92,
                certified=True,
                iteration_count=1,
                search_budget_used=1,
                evidence_budget_used=0,
                stop_reason="certified",
            ),
            [_route("route-a"), _route("route-b")],
            "certified_set",
            2,
            2,
            None,
        ),
        (
            RouteCertificationSummary(
                route_id="route-a",
                certificate=0.43,
                certified=False,
                threshold=0.70,
                active_families=["scenario"],
                top_fragility_families=["weather"],
                top_competitor_route_id="route-b",
                top_value_of_refresh_family="weather",
                ambiguity_context={"support_strength": True},
            ),
            VoiStopSummary(
                final_route_id="route-a",
                certificate=0.43,
                certified=False,
                iteration_count=3,
                search_budget_used=2,
                evidence_budget_used=1,
                stop_reason="search_incomplete_no_action_worth_it",
                credible_search_uncertainty=True,
            ),
            [_route("route-a"), _route("route-b")],
            "typed_abstention",
            0,
            0,
            "uncertified_due_to_search",
        ),
    ],
)
def test_route_terminal_semantics(
    selected_certificate: RouteCertificationSummary | None,
    voi_stop_summary: VoiStopSummary | None,
    strict_frontier: list[RouteOption],
    expected_terminal_type: str,
    expected_public_set_size: int,
    expected_summary_set_size: int,
    expected_reason: str | None,
) -> None:
    response = _response(
        selected_certificate=selected_certificate,
        voi_stop_summary=voi_stop_summary,
        strict_frontier=strict_frontier,
    )

    encoded = json.loads(response.model_dump_json())
    assert encoded["terminal_type"] == expected_terminal_type
    assert len(encoded["certified_set"]) == expected_public_set_size
    assert encoded["certified_set_summary"]["set_size"] == expected_summary_set_size
    assert encoded["certified_set_summary"]["witness"]["route_id"] == "route-a"
    assert response.terminal_type == expected_terminal_type
    assert len(response.certified_set) == expected_public_set_size

    if expected_terminal_type == "certified_singleton":
        assert encoded["selected"]["id"] == "route-a"
        assert encoded["recommended_route"]["id"] == "route-a"
        assert encoded["certified_set_summary"]["member_route_ids"] == []
        assert encoded["certified_set_summary"]["certified"] is False
        assert encoded["certified_set_summary"]["not_applicable_reason"] == "singleton_terminal"
    else:
        assert encoded["selected"] is None
        assert encoded["recommended_route"] is None

    if expected_reason is None:
        assert encoded["abstention"] is None
        assert response.abstention is None
    else:
        assert encoded["abstention"]["reason_code"] == expected_reason
        assert response.abstention is not None
        assert response.abstention.reason_code == expected_reason


def test_route_terminal_abstention_clears_certified_set_and_preserves_summary_fields() -> None:
    selected = _route("route-a")
    challenger = _route("route-b")
    selected_certificate = RouteCertificationSummary(
        route_id="route-a",
        certificate=0.41,
        certified=False,
        threshold=0.70,
        active_families=[],
        top_fragility_families=["weather"],
        top_competitor_route_id="route-b",
        top_value_of_refresh_family="weather",
        ambiguity_context={"support_strength": False},
    )
    abstention = build_abstention_record(
        stop_reason="search_incomplete_no_action_worth_it",
        support_flag=False,
        support_reason="out_of_support_world_model",
        credible_search_uncertainty=True,
        active_families=[],
        top_fragility_families=["weather"],
        detail={"case": "typed_abstention"},
    )
    response = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-2",
        pipeline_mode="dccs_refc",
        manifest_endpoint="/runs/run-2/manifest",
        artifacts_endpoint="/runs/run-2/artifacts",
        provenance_endpoint="/runs/run-2/provenance",
        selected_certificate=selected_certificate,
        voi_stop_summary=VoiStopSummary(
            final_route_id="route-a",
            certificate=0.41,
            certified=False,
            iteration_count=2,
            search_budget_used=2,
            evidence_budget_used=1,
            stop_reason="search_incomplete_no_action_worth_it",
            credible_search_uncertainty=True,
        ),
        preference_state={},
        preference_query_trace={},
        abstention=abstention,
        world_support_summary={
            "support_flag": False,
            "support_reason": "out_of_support_world_model",
            "active_families": [],
        },
        certified_set=[],
    )

    encoded = json.loads(response.model_dump_json())

    assert encoded["terminal_type"] == "typed_abstention"
    assert encoded["certified_set"] == []
    assert encoded["selected"] is None
    assert encoded["recommended_route"] is None
    assert encoded["abstention"]["reason_code"] == "uncertified_due_to_out_of_support_world_model"
    assert encoded["world_support_summary"]["support_flag"] is False
    assert encoded["selected_certificate_basis"] == "selected_certificate"
    assert encoded["artifact_pointers"] == {
        "manifest_endpoint": "/runs/run-2/manifest",
        "artifacts_endpoint": "/runs/run-2/artifacts",
        "provenance_endpoint": "/runs/run-2/provenance",
    }
    assert encoded["certified_set_summary"]["member_route_ids"] == []
    assert encoded["certified_set_summary"]["excluded_route_ids"] == ["route-a", "route-b"]
    assert encoded["witness_summary"]["route_id"] == "route-a"


def test_support_flag_false_forces_typed_abstention_even_when_voi_marks_certified() -> None:
    response = _response(
        selected_certificate=RouteCertificationSummary(
            route_id="route-a",
            certificate=0.92,
            certified=True,
            threshold=0.70,
            active_families=["scenario"],
            top_fragility_families=["weather"],
            top_competitor_route_id="route-b",
            top_value_of_refresh_family="weather",
            ambiguity_context={"support_strength": False},
        ),
        voi_stop_summary=VoiStopSummary(
            final_route_id="route-a",
            certificate=0.92,
            certified=True,
            iteration_count=1,
            search_budget_used=1,
            evidence_budget_used=0,
            stop_reason="certified",
        ),
        strict_frontier=[_route("route-a")],
        support_flag=False,
        support_reason="out_of_support_world_model",
    )

    encoded = json.loads(response.model_dump_json())

    assert encoded["terminal_type"] == "typed_abstention"
    assert encoded["selected"] is None
    assert encoded["recommended_route"] is None
    assert encoded["certified_set"] == []
    assert encoded["abstention"]["reason_code"] == "uncertified_due_to_out_of_support_world_model"
    assert encoded["certificate_summary"]["support_flag"] is False
    assert (
        encoded["certificate_summary"]["out_of_support_reason"]
        == "out_of_support_world_model"
    )
    assert encoded["certificate_summary"]["terminal_type"] == "typed_abstention"


def test_certified_set_terminal_synthesizes_membership_and_exclusion_basis_when_summary_missing() -> None:
    selected = _route("route-a")
    challenger = _route("route-b")

    payload = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-certified-set-contract",
        pipeline_mode="dccs_refc",
        manifest_endpoint="/runs/run-certified-set-contract/manifest",
        artifacts_endpoint="/runs/run-certified-set-contract/artifacts",
        provenance_endpoint="/runs/run-certified-set-contract/provenance",
        selected_certificate=RouteCertificationSummary(
            route_id="route-a",
            certificate=0.92,
            certified=True,
            threshold=0.70,
            active_families=["scenario"],
            top_fragility_families=["weather"],
            top_competitor_route_id="route-b",
            top_value_of_refresh_family="weather",
            ambiguity_context={"support_strength": True},
        ),
        voi_stop_summary=VoiStopSummary(
            final_route_id="route-a",
            certificate=0.92,
            certified=True,
            iteration_count=1,
            search_budget_used=1,
            evidence_budget_used=0,
            stop_reason="certified",
        ),
        preference_state={},
        preference_query_trace={},
        world_support_summary={
            "schema_version": "world-support-summary-v1",
            "support_flag": True,
            "support_state": {
                "support_flag": True,
            },
        },
        certified_set=[selected, challenger],
        certified_set_summary=None,
        abstention=None,
    )

    encoded = json.loads(payload.model_dump_json())

    assert encoded["terminal_type"] == "certified_set"
    assert [route["id"] for route in encoded["certified_set"]] == ["route-a", "route-b"]
    assert encoded["certified_set_summary"]["terminal_type"] == "certified_set"
    assert encoded["certified_set_summary"]["member_route_ids"] == ["route-a", "route-b"]
    assert encoded["certified_set_summary"]["excluded_route_ids"] == []
    assert encoded["certified_set_summary"]["certified"] is True
    assert encoded["certified_set_summary"]["set_size"] == 2
    assert encoded["certified_set_summary"]["exclusion_basis"] == [
        "frontier_selection",
        "no_outside_routes_excluded",
        "explicit_certified_set_summary_missing",
    ]
    assert encoded["certified_set_summary"]["witness"]["route_id"] == "route-a"
    assert encoded["certified_set_summary"]["witness"]["active_challenger_ids"] == ["route-b"]
    assert (
        encoded["certified_set_summary"]["witness"]["summary_status"]
        == "synthesized_without_explicit_certified_set_summary"
    )


def test_assembled_decision_package_preserves_refc_proof_objects_in_live_payload() -> None:
    routes = [
        {
            "route_id": "route-a",
            "objective": {"time": 10.0, "money": 12.0, "co2": 4.0},
            "evidence": {"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        },
        {
            "route_id": "route-b",
            "objective": {"time": 11.0, "money": 11.0, "co2": 5.0},
            "evidence": {"scenario": {"time": 0.9, "money": 1.1, "co2": 1.0}},
        },
    ]
    worlds = [
        {
            "world_id": "w1",
            "states": {"scenario": "nominal"},
            "world_kind": "supported_ambiguity_nominal",
        },
        {
            "world_id": "w2",
            "states": {"scenario": "refreshed"},
            "world_kind": "supported_ambiguity_refreshed",
        },
    ]
    certificate = compute_certificate(routes, worlds=worlds, threshold=0.5)
    fragility = compute_fragility_maps(routes, worlds=worlds, selected_route_id=certificate.selected_route_id)
    projection = project_refc_scaffold_states(
        certificate,
        fragility,
        frontier_route_ids=[certificate.selected_route_id, "route-b"],
    )
    selected = _route("route-a")
    challenger = _route("route-b")
    selected_certificate = RouteCertificationSummary(
        route_id=certificate.selected_route_id,
        certificate=float(certificate.certificate[certificate.selected_route_id]),
        certified=bool(certificate.certified),
        threshold=float(certificate.threshold),
        active_families=list(certificate.world_manifest.get("active_families", [])),
        top_fragility_families=list(fragility.route_fragility_map.get(certificate.selected_route_id, {}).keys())[:3],
        top_competitor_route_id="route-b",
        top_value_of_refresh_family="scenario",
        ambiguity_context={"support_strength": True},
    )
    preference_state = append_preference_query(
        build_preference_state(route_ids=["route-a", "route-b"], weights={"time": 1.0, "money": 0.5}),
        PairwisePreferenceQuery(preferred_route_id="route-a", challenger_route_id="route-b"),
        before_size=2,
        after_size=1,
        before_volume_proxy=1.0,
        after_volume_proxy=0.4,
        target_route_id="route-b",
        query_reason="reduce ambiguity",
    )

    payload = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-proof",
        pipeline_mode="dccs_refc",
        manifest_endpoint="/runs/run-proof/manifest",
        artifacts_endpoint="/runs/run-proof/artifacts",
        provenance_endpoint="/runs/run-proof/provenance",
        selected_certificate=selected_certificate,
        voi_stop_summary=VoiStopSummary(
            final_route_id=certificate.selected_route_id,
            certificate=float(certificate.certificate[certificate.selected_route_id]),
            certified=bool(certificate.certified),
            iteration_count=2,
            search_budget_used=1,
            evidence_budget_used=1,
            stop_reason="certified",
        ),
        preference_state=preference_state,
        preference_query_trace={},
        world_support_summary={
            "schema_version": "world-support-summary-v1",
            "selected_route_id": certificate.selected_route_id,
            "selected_certificate_basis": "selected_certificate",
            "support_flag": True,
            "support_state": {
                "support_flag": True,
                "support_bin": "supported",
            },
            "world_bundle_summary": {
                "multi_fidelity_summary": {
                    "proxy_world_count": 3,
                    "audit_world_count": 1,
                    "proxy_bias_model_version": "proxy-v5",
                    "audit_propensity_version": "audit-v3",
                    "proxy_correction_active": True,
                    "multi_fidelity_certificate_basis": "corrected_from_residual_model",
                    "proxy_only_fraction": 0.75,
                    "audit_correction_mass": 2.5,
                    "positivity_diagnostics": {
                        "positivity_ok": True,
                        "weak_overlap_detected": False,
                    },
                },
            },
        },
        world_manifest=certificate.world_manifest,
        winner_confidence_state=projection["winner_confidence_state"],
        pairwise_gap_states=projection["pairwise_gap_states"],
        flip_radius_state=projection["flip_radius_state"],
        decision_region_state=projection["decision_region_state"],
        certificate_witness=projection["certificate_witness"],
        certified_set=[selected],
        abstention=None,
    )

    encoded = json.loads(payload.model_dump_json())

    assert encoded["terminal_type"] == "certified_singleton"
    assert encoded["winner_confidence_state"]["route_id"] == certificate.selected_route_id
    assert encoded["pairwise_gap_states"][0]["challenger_id"] == "route-b"
    assert encoded["pairwise_gap_states"][0]["nearest_challenger"] is True
    assert encoded["flip_radius_state"]["minimum_flip_budget"] is not None
    assert encoded["flip_radius_state"]["adversarial_degradation_curve"]
    assert encoded["decision_region_state"]["active_challenger_id"] == "route-b"
    assert encoded["decision_region_state"]["nearest_certificate_boundary"] is not None
    assert encoded["certificate_witness"]["active_challenger_ids"] == ["route-b"]
    assert encoded["certificate_witness"]["support_conditions"]
    assert encoded["certificate_witness"]["action_steps"]
    assert encoded["witness_summary"]["witness_size"] == projection["certificate_witness"].witness_size
    assert encoded["witness_summary"]["active_challenger_ids"] == ["route-b"]
    assert encoded["stability_summary"]["minimum_pairwise_gap_lcb"] == projection["pairwise_gap_states"][0].pairwise_gap_lower_bound
    assert encoded["stability_summary"]["minimum_flip_budget"] == projection["flip_radius_state"].minimum_flip_budget
    assert encoded["preference_query_trace"]["contradiction_record"]["contradiction_detected"] is False
    assert encoded["preference_query_trace"]["preference_irrelevance_proven"] is True
    assert encoded["preference_query_trace"]["no_query_reason"] is None
    assert encoded["preference_query_trace"]["targeted_challenger_route_id"] == "route-b"
    assert encoded["preference_query_trace"]["query_selection_reason"] == "reduce ambiguity"
    assert encoded["preference_summary"]["query_count"] == 1
    assert encoded["preference_summary"]["preference_irrelevance_proven"] is True
    assert encoded["preference_summary"]["targeted_challenger_route_id"] == "route-b"
    assert encoded["preference_summary"]["query_selection_reason"] == "reduce ambiguity"
    assert encoded["support_summary"]["multi_fidelity_summary"]["proxy_world_count"] == 3
    assert encoded["support_summary"]["proxy_world_count"] == 3
    assert encoded["support_summary"]["proxy_only_fraction"] == 0.75
    assert encoded["support_summary"]["positivity_diagnostics"]["positivity_ok"] is True
