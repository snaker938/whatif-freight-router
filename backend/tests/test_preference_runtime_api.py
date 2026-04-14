import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import GeoJSONLineString, RouteMetrics, RouteOption
from app.preference_model import build_preference_state
from app.preference_queries import PairwisePreferenceQuery
from app.preference_runtime import (
    PreferenceRuntimeUpdateRequest,
    apply_preference_runtime_update,
)
from app.preference_update import append_preference_query


def _route(route_id: str, *, duration_s: float, monetary_cost: float) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(0.0, 0.0), (1.0, 1.0)]),
        metrics=RouteMetrics(
            distance_km=10.0,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=4.0,
            avg_speed_kmh=50.0,
        ),
    )


def test_apply_preference_runtime_update_reemits_query_trace_and_selected_route() -> None:
    state = build_preference_state(
        route_ids=["route_a", "route_b"],
        weights={"time": 2.0, "money": 1.0, "co2": 0.5},
    )
    updated_state = append_preference_query(
        state,
        PairwisePreferenceQuery(
            preferred_route_id="route_a",
            challenger_route_id="route_b",
            reason="prefer lower cost under the live certificate frontier",
        ),
        before_size=2,
        after_size=1,
        before_volume_proxy=1.0,
        after_volume_proxy=0.4,
        target_route_id="route_b",
        query_reason="prefer lower cost under the live certificate frontier",
        preference_irrelevance=True,
    )
    updated_state.terminal_type = "certified"

    response = apply_preference_runtime_update(
        PreferenceRuntimeUpdateRequest(
            candidate_routes=[
                _route("route_a", duration_s=20.0, monetary_cost=10.0),
                _route("route_b", duration_s=24.0, monetary_cost=12.0),
            ],
            selected_route_id="route_b",
            selected_certificate_basis="selected_certificate",
            pipeline_mode="dccs_refc",
            support_flag=True,
            support_reason="trusted_frontier",
            preference_state=updated_state,
        )
    )

    assert response.selected_route_id == "route_a"
    assert response.terminal_type == "certified"
    assert response.preference_state.compatible_set_summary.compatible_set_size == 1
    assert response.preference_state.compatible_set_summary.possible_best_route_ids == ["route_a"]
    assert response.preference_query_trace["query_count"] == 1
    assert response.preference_query_trace["query_history"][0]["query_type"] == "pairwise"
    assert response.preference_query_trace["query_selection_reason"] == (
        "prefer lower cost under the live certificate frontier"
    )
    assert response.preference_summary["query_count"] == 1
    assert response.preference_summary["selected_certificate_basis"] == "selected_certificate"


def test_apply_preference_runtime_update_filters_missing_routes_and_preserves_support_reason() -> None:
    state = build_preference_state(
        route_ids=["route_a", "missing_route"],
        weights={"time": 1.0, "money": 0.5, "co2": 0.5},
        support_flag=False,
        support_reason="insufficient_overlap",
    )

    response = apply_preference_runtime_update(
        PreferenceRuntimeUpdateRequest(
            candidate_routes=[
                _route("route_a", duration_s=20.0, monetary_cost=10.0),
                _route("route_b", duration_s=24.0, monetary_cost=12.0),
            ],
            selected_route_id="route_a",
            selected_certificate_basis="empirical",
            pipeline_mode="voi",
            support_flag=False,
            support_reason="insufficient_overlap",
            preference_state=state,
        )
    )

    assert response.preference_state.compatible_set_summary.route_ids == ["route_a", "route_b"]
    assert response.preference_state.compatible_set_summary.support_flag is False
    assert response.preference_state.compatible_set_summary.support_reason == "insufficient_overlap"
    assert response.preference_query_trace["no_query_reason"] == "preference_support_insufficient"
    assert response.preference_query_trace["no_preference_query_reason"] == "preference_support_insufficient"
    assert response.preference_summary["no_query_reason"] == "preference_support_insufficient"
    assert response.preference_summary["query_count"] == 0


def test_apply_preference_runtime_update_requires_candidate_routes() -> None:
    state = build_preference_state(route_ids=["route_a"], weights={"time": 1.0})

    with pytest.raises(ValueError, match="candidate_routes"):
        apply_preference_runtime_update(
            PreferenceRuntimeUpdateRequest(
                candidate_routes=[_route("", duration_s=20.0, monetary_cost=10.0)],
                selected_route_id=None,
                preference_state=state,
            )
        )


def test_route_preference_endpoint_reemits_runtime_payload() -> None:
    state = build_preference_state(
        route_ids=["route_a", "route_b"],
        weights={"time": 2.0, "money": 1.0, "co2": 0.5},
    )
    updated_state = append_preference_query(
        state,
        PairwisePreferenceQuery(
            preferred_route_id="route_a",
            challenger_route_id="route_b",
            reason="route_a dominates the current certificate frontier",
        ),
        before_size=2,
        after_size=1,
        before_volume_proxy=1.0,
        after_volume_proxy=0.25,
        target_route_id="route_b",
        query_reason="route_a dominates the current certificate frontier",
        preference_irrelevance=True,
    )
    updated_state.terminal_type = "certified"

    with TestClient(app) as client:
        resp = client.post(
            "/route/preference",
            json={
                "candidate_routes": [
                    _route("route_a", duration_s=20.0, monetary_cost=10.0).model_dump(mode="json"),
                    _route("route_b", duration_s=24.0, monetary_cost=12.0).model_dump(mode="json"),
                ],
                "selected_route_id": "route_b",
                "selected_certificate_basis": "selected_certificate",
                "pipeline_mode": "dccs_refc",
                "support_flag": True,
                "support_reason": "trusted_frontier",
                "preference_state": updated_state.model_dump(mode="json"),
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["selected_route_id"] == "route_a"
    assert payload["selected_certificate_basis"] == "selected_certificate"
    assert payload["terminal_type"] == "certified"
    assert payload["preference_state"]["compatible_set_summary"]["route_ids"] == ["route_a"]
    assert payload["preference_query_trace"]["query_count"] == 1
    assert payload["preference_query_trace"]["query_selection_reason"] == (
        "route_a dominates the current certificate frontier"
    )
    assert payload["preference_summary"]["query_count"] == 1
