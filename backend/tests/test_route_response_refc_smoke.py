from __future__ import annotations

import json

from app.evidence_certification import compute_certificate, compute_fragility_maps, project_refc_scaffold_states
from app.main import _route_terminal_fields
from app.models import (
    GeoJSONLineString,
    AbstentionRecord,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    RouteResponse,
    VoiStopSummary,
)


def _route(route_id: str, *, distance_km: float, duration_s: float, monetary_cost: float, emissions_kg: float) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(0.0, 0.0), (1.0, 1.0)]),
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            avg_speed_kmh=50.0,
        ),
    )


def test_route_response_refc_smoke_ties_terminal_shape_to_scaffold_projection() -> None:
    routes = [
        {
            "route_id": "route_a",
            "objective": {"time": 10.0, "money": 12.0, "co2": 4.0},
            "evidence": {"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        },
        {
            "route_id": "route_b",
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
        frontier_route_ids=[certificate.selected_route_id, "route_b"],
    )

    selected_certificate = RouteCertificationSummary(
        route_id=certificate.selected_route_id,
        certificate=float(certificate.certificate[certificate.selected_route_id]),
        certified=bool(certificate.certified),
        threshold=float(certificate.threshold),
        active_families=list(certificate.world_manifest.get("active_families", [])),
        top_fragility_families=list(fragility.route_fragility_map.get(certificate.selected_route_id, {}).keys())[:3],
        top_competitor_route_id=certificate.winner_id if certificate.winner_id != certificate.selected_route_id else "route_b",
        top_value_of_refresh_family="scenario",
        ambiguity_context={"support_strength": True},
    )
    voi_stop_summary = VoiStopSummary(
        final_route_id=certificate.selected_route_id,
        certificate=float(certificate.certificate[certificate.selected_route_id]),
        certified=bool(certificate.certified),
        iteration_count=2,
        search_budget_used=1,
        evidence_budget_used=1,
        stop_reason="certified",
    )

    certified_set, abstention = _route_terminal_fields(
        selected_certificate=selected_certificate,
        voi_stop_summary=voi_stop_summary,
        strict_frontier=[
            _route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
            _route("route_b", distance_km=11.0, duration_s=21.0, monetary_cost=31.0, emissions_kg=5.0),
        ],
    )

    response = RouteResponse(
        selected=_route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
        candidates=[
            _route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
            _route("route_b", distance_km=11.0, duration_s=21.0, monetary_cost=31.0, emissions_kg=5.0),
        ],
        run_id="run-refc-smoke",
        pipeline_mode="voi",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
        selected_certificate=selected_certificate,
        voi_stop_summary=voi_stop_summary,
        world_support_summary={
            "schema_version": "world-support-summary-v1",
            "selected_route_id": certificate.selected_route_id,
            "selected_certificate_basis": "selected_certificate",
            "support_flag": True,
            "support_state": {
                "support_score": 1.0,
                "support_ratio": 1.0,
                "support_bin": "supported",
                "calibration_bin": "empirical",
                "support_source": "world_manifest",
                "out_of_support_reason": None,
                "provenance": {
                    "selected_route_id": certificate.selected_route_id,
                },
            },
            "world_bundle_summary": {
                "schema_version": "world-bundle-summary-v1",
                "support_flag": True,
                "support_state": {
                    "support_score": 1.0,
                    "support_ratio": 1.0,
                    "support_bin": "supported",
                    "calibration_bin": "empirical",
                    "support_source": "world_manifest",
                    "out_of_support_reason": None,
                    "provenance": {
                        "selected_route_id": certificate.selected_route_id,
                    },
                },
                "probabilistic_world_bundle": {
                    "world_count": 2,
                    "selected_route_id": certificate.selected_route_id,
                },
                "audit_world_bundle": {
                    "world_count": 1,
                    "selected_route_id": certificate.selected_route_id,
                },
            },
        },
        certified_set=certified_set,
        abstention=abstention,
        winner_confidence_state=projection["winner_confidence_state"],
        pairwise_gap_states=projection["pairwise_gap_states"],
        flip_radius_state=projection["flip_radius_state"],
        decision_region_state=projection["decision_region_state"],
        certificate_witness=projection["certificate_witness"],
    )

    encoded = json.loads(response.model_dump_json())
    restored = RouteResponse.model_validate(response.model_dump(mode="python"))
    assert response.terminal_type == "certified_set"
    assert encoded["terminal_type"] == "certified_set"
    assert encoded["selected_certificate_basis"] == "selected_certificate"
    assert encoded["world_support_summary"]["selected_route_id"] == certificate.selected_route_id
    assert encoded["world_support_summary"]["schema_version"] == "world-support-summary-v1"
    assert encoded["world_support_summary"]["support_state"]["support_bin"] == "supported"
    assert encoded["world_support_summary"]["world_bundle_summary"]["support_state"]["support_bin"] == "supported"
    assert encoded["artifact_pointers"] == {
        "manifest_endpoint": "/manifest",
        "artifacts_endpoint": "/artifacts",
        "provenance_endpoint": "/provenance",
    }
    assert [route.id for route in response.certified_set] == ["route_a", "route_b"]
    assert response.abstention is None
    assert abstention is None
    assert encoded["abstention"] is None
    assert encoded["winner_confidence_state"]["route_id"] == certificate.selected_route_id
    assert encoded["pairwise_gap_states"][0]["challenger_id"] == "route_b"
    assert encoded["flip_radius_state"]["route_id"] == certificate.selected_route_id
    assert encoded["decision_region_state"]["route_id"] == certificate.selected_route_id
    assert encoded["certificate_witness"]["route_id"] == certificate.selected_route_id
    assert restored.terminal_type == "certified_set"
    assert restored.selected_certificate_basis == "selected_certificate"
    assert restored.world_support_summary["schema_version"] == "world-support-summary-v1"
    assert restored.world_support_summary["selected_route_id"] == certificate.selected_route_id
    assert json.loads(restored.model_dump_json()) == encoded

    assert projection["winner_confidence_state"].route_id == certificate.selected_route_id
    assert projection["pairwise_gap_states"]
    assert projection["flip_radius_state"].route_id == certificate.selected_route_id
    assert projection["decision_region_state"].route_id == certificate.selected_route_id
    assert projection["certificate_witness"].route_id == certificate.selected_route_id
    assert projection["certified_set_state"].member_route_ids == [certificate.selected_route_id, "route_b"]


def test_route_response_revalidates_abstention_mutation_without_certified_set_drift() -> None:
    response = RouteResponse(
        selected=_route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
        candidates=[
            _route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
            _route("route_b", distance_km=11.0, duration_s=21.0, monetary_cost=31.0, emissions_kg=5.0),
        ],
        run_id="run-abstain",
        pipeline_mode="voi",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
    )

    response.abstention = AbstentionRecord(reason_code="uncertified_due_to_search", message="search blocked")

    encoded = json.loads(response.model_dump_json())
    assert response.terminal_type == "typed_abstention"
    assert encoded["terminal_type"] == "typed_abstention"
    assert response.certified_set == []
    assert encoded["certified_set"] == []
    assert response.certified_set_summary["member_route_ids"] == []
    assert response.certified_set_summary["certified"] is False
    assert encoded["abstention"]["reason_code"] == "uncertified_due_to_search"
