from __future__ import annotations

import json

import pytest

from app.abstention import build_abstention_record
from app.main import _route_terminal_fields
from app.models import (
    GeoJSONLineString,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    RouteResponse,
    VoiStopSummary,
)


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
) -> RouteResponse:
    selected = strict_frontier[0]
    certified_set, abstention = _route_terminal_fields(
        selected_certificate=selected_certificate,
        voi_stop_summary=voi_stop_summary,
        strict_frontier=strict_frontier,
    )
    if abstention is not None:
        certified_set = []
    response = RouteResponse(
        selected=selected,
        candidates=list(strict_frontier),
        run_id="run-1",
        pipeline_mode="voi",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
        selected_certificate=selected_certificate,
        voi_stop_summary=voi_stop_summary,
        certified_set=certified_set,
        abstention=abstention,
    )
    return response


@pytest.mark.parametrize(
    "selected_certificate, voi_stop_summary, strict_frontier, expected_terminal_type, expected_set_size, expected_reason",
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
            1,
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
            "uncertified_due_to_search",
        ),
    ],
)
def test_route_terminal_semantics(
    selected_certificate: RouteCertificationSummary | None,
    voi_stop_summary: VoiStopSummary | None,
    strict_frontier: list[RouteOption],
    expected_terminal_type: str,
    expected_set_size: int,
    expected_reason: str | None,
) -> None:
    response = _response(
        selected_certificate=selected_certificate,
        voi_stop_summary=voi_stop_summary,
        strict_frontier=strict_frontier,
    )

    encoded = json.loads(response.model_dump_json())
    assert encoded["terminal_type"] == expected_terminal_type
    assert len(encoded["certified_set"]) == expected_set_size
    assert encoded["certified_set_summary"]["member_route_ids"] == [route.id for route in response.certified_set]
    assert encoded["certified_set_summary"]["set_size"] == expected_set_size
    assert encoded["certified_set_summary"]["witness"]["route_id"] == "route-a"
    assert response.terminal_type == expected_terminal_type
    assert len(response.certified_set) == expected_set_size

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
    response = RouteResponse(
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
        certified_set=[selected, challenger],
        abstention=abstention,
        world_support_summary={
            "support_flag": False,
            "support_reason": "out_of_support_world_model",
            "active_families": [],
        },
    )

    encoded = json.loads(response.model_dump_json())

    assert encoded["terminal_type"] == "typed_abstention"
    assert encoded["certified_set"] == []
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
