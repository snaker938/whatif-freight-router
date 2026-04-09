from __future__ import annotations

import app.main as main_module
from app.models import GeoJSONLineString, LatLng, RouteMetrics, RouteOption, RouteRequest


def _route_option(*, route_id: str) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[
                [-1.0, 51.0],
                [-0.5, 51.5],
                [0.0, 52.0],
            ],
        ),
        metrics=RouteMetrics(
            distance_km=100.0,
            duration_s=3600.0,
            monetary_cost=200.0,
            emissions_kg=50.0,
            avg_speed_kmh=100.0,
        ),
    )


def _route_request() -> RouteRequest:
    return RouteRequest(
        origin=LatLng(lat=51.0, lon=-1.0),
        destination=LatLng(lat=52.0, lon=0.0),
    )


def test_canonical_selected_frontier_row_prefers_frontier_signature_for_selected_route_id() -> None:
    selected = _route_option(route_id="route_0")

    canonical_row = main_module._canonical_selected_frontier_row(
        selected_route_id=selected.id,
        selected_route_signature="signature-from-selected-option",
        frontier_rows=[
            {
                "route_id": "route_0",
                "route_signature": "signature-from-frontier-row",
                "candidate_ids": ["cand-frontier"],
                "selected": True,
            },
            {
                "route_id": "route_1",
                "route_signature": "signature-other",
                "candidate_ids": ["cand-other"],
            },
        ],
    )

    assert canonical_row["route_id"] == "route_0"
    assert canonical_row["route_signature"] == "signature-from-frontier-row"
    assert canonical_row["candidate_ids"] == ["cand-frontier"]


def test_build_route_decision_package_uses_canonical_selected_frontier_route_id() -> None:
    selected = _route_option(route_id="route_local")
    peer = _route_option(route_id="route_peer")

    decision_package = main_module._build_route_decision_package(
        req=_route_request(),
        requested_pipeline_mode="voi",
        actual_pipeline_mode="voi",
        selected=selected,
        candidates=[selected, peer],
        warnings=[],
        selected_certificate=None,
        voi_stop_summary=None,
        evidence_validation={},
        extra_json_artifacts={},
        extra_jsonl_artifacts={
            "strict_frontier.jsonl": [
                {
                    "route_id": "route_frontier",
                    "route_signature": "frontier-signature",
                    "candidate_ids": ["cand-frontier"],
                    "distance_km": 100.0,
                    "duration_s": 3600.0,
                    "monetary_cost": 200.0,
                    "emissions_kg": 50.0,
                    "selected": True,
                },
                {
                    "route_id": "route_peer",
                    "route_signature": "peer-signature",
                    "candidate_ids": ["cand-peer"],
                    "distance_km": 110.0,
                    "duration_s": 3700.0,
                    "monetary_cost": 210.0,
                    "emissions_kg": 55.0,
                    "selected": False,
                },
            ]
        },
    )

    assert decision_package.selected_route_id == "route_frontier"
    assert decision_package.preference_summary.selected_route_id == "route_frontier"
    assert decision_package.certified_set_summary.selected_route_id == "route_frontier"
    assert decision_package.witness_summary is not None
    assert decision_package.witness_summary.primary_witness_route_id == "route_frontier"


def test_string_list_helper_is_available_for_module_scope_route_packaging() -> None:
    assert main_module._string_list('["cand_a", "cand_b"]') == ["cand_a", "cand_b"]
    assert main_module._string_list(["cand_a", "", None, "cand_b"]) == ["cand_a", "cand_b"]
