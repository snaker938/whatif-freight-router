import json

import pytest

from app.preference_model import (
    build_compatible_set_summary,
    build_preference_state,
    build_preference_shrinkage_trace,
    normalize_weight_vector,
    summarize_compatible_weight_set,
)
from app.preference_queries import (
    PairwisePreferenceQuery,
    PreferenceQuery,
    RatioPreferenceQuery,
    ThresholdPreferenceQuery,
    TimeGuardPreferenceQuery,
    VetoPreferenceQuery,
)
from app.preference_state import CompatibleSetSummary, PreferenceState
from app.preference_update import append_preference_query, validate_preference_invariants
from app.models import GeoJSONLineString, RouteMetrics, RouteOption, RouteResponse


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


def test_preference_queries_round_trip_json_serializable() -> None:
    queries: list[PreferenceQuery] = [
        PairwisePreferenceQuery(
            preferred_route_id="route_a",
            challenger_route_id="route_b",
            reason="closer to preference frontier",
        ),
        ThresholdPreferenceQuery(
            route_id="route_a",
            metric_name="duration_s",
            threshold_value=1200.0,
            direction="lte",
        ),
        RatioPreferenceQuery(
            route_id="route_a",
            numerator_metric="duration_s",
            denominator_metric="distance_km",
            minimum_ratio=10.0,
        ),
        VetoPreferenceQuery(route_id="route_b", veto_name="hazmat_restricted", active=True),
        TimeGuardPreferenceQuery(
            route_id="route_a",
            latest_arrival_utc="2026-04-09T12:00:00Z",
            max_travel_time_s=1800.0,
            preserve_time_budget_s=300.0,
        ),
    ]

    for query in queries:
        payload = query.model_dump(mode="json")
        restored = type(query).model_validate(payload)
        assert restored.model_dump(mode="json") == payload


def test_preference_state_preserves_summary_and_shrinkage() -> None:
    state = build_preference_state(route_ids=["route_a", "route_b"], weights={"time": 2.0, "money": 1.0})
    query = PairwisePreferenceQuery(preferred_route_id="route_a", challenger_route_id="route_b")
    updated = append_preference_query(
        state,
        query,
        before_size=2,
        after_size=1,
        before_volume_proxy=1.0,
        after_volume_proxy=0.4,
        target_route_id="route_b",
        query_reason="reduce ambiguity",
    )

    payload = updated.model_dump(mode="json")
    assert payload["compatible_set_summary"]["compatible_set_size"] == 1
    assert payload["compatible_set_summary"]["route_ids"] == ["route_a", "route_b"]
    assert payload["query_count"] == 1
    assert payload["query_history"][0]["query_type"] == "pairwise"
    assert payload["shrinkage_trace"][0]["predicted_shrinkage"] == 0.5
    assert payload["shrinkage_trace"][0]["realized_shrinkage"] == 0.6
    assert payload["compatible_set_summary"]["necessary_best_route_ids"] == ["route_a"]
    assert payload["compatible_set_summary"]["possible_best_route_ids"] == ["route_a"]
    assert payload["derived_invariants"]["necessary_best_prob_le_possible_best_prob"] is True
    assert payload["derived_invariants"]["no_necessary_best_without_possible_best"] is True
    assert validate_preference_invariants(updated)["compatible_set_volume_nonincreasing_after_query"] is True
    assert payload["contradiction_record"]["contradiction_detected"] is False


def test_preference_state_detects_contradictions_and_irrelevance() -> None:
    singleton = build_preference_state(route_ids=["route_a"], weights={"time": 1.0})
    assert singleton.preference_irrelevance_proven is True
    assert singleton.no_query_reason == "preference_irrelevance_proven"
    assert singleton.compatible_set_summary.necessary_best_prob == 1.0
    assert singleton.compatible_set_summary.possible_best_prob == 1.0
    assert singleton.compatible_set_summary.necessary_best_route_ids == ["route_a"]
    assert singleton.compatible_set_summary.possible_best_route_ids == ["route_a"]

    state = build_preference_state(route_ids=["route_a", "route_b"], weights={"time": 2.0, "money": 1.0})
    updated = append_preference_query(
        state,
        PairwisePreferenceQuery(preferred_route_id="route_a", challenger_route_id="route_b"),
        before_size=2,
        after_size=2,
        before_volume_proxy=1.0,
        after_volume_proxy=1.0,
    )
    contradictory = append_preference_query(
        updated,
        PairwisePreferenceQuery(preferred_route_id="route_b", challenger_route_id="route_a"),
        before_size=2,
        after_size=2,
        before_volume_proxy=1.0,
        after_volume_proxy=1.0,
    )

    assert contradictory.contradiction_record.contradiction_detected is True
    assert contradictory.derived_invariants["preference_contradiction_free"] is False
    assert contradictory.compatible_set_summary.necessary_best_route_ids == []
    assert contradictory.derived_invariants["no_necessary_best_without_possible_best"] is True
    assert any(
        reason.startswith("pairwise_cycle:")
        for reason in contradictory.contradiction_record.contradiction_reasons
    )


def test_preference_queries_reject_invalid_route_and_guard_payloads() -> None:
    with pytest.raises(ValueError, match="distinct"):
        PairwisePreferenceQuery(preferred_route_id="route_a", challenger_route_id="route_a")
    with pytest.raises(ValueError, match="distinct"):
        RatioPreferenceQuery(
            route_id="route_a",
            numerator_metric="duration_s",
            denominator_metric="duration_s",
            minimum_ratio=1.0,
        )
    with pytest.raises(ValueError, match="at least one concrete bound"):
        TimeGuardPreferenceQuery(route_id="route_a")


def test_append_preference_query_rejects_volume_growth() -> None:
    state = build_preference_state(route_ids=["route_a", "route_b"], weights={"time": 2.0, "money": 1.0})

    with pytest.raises(ValueError, match="compatible-set volume"):
        append_preference_query(
            state,
            ThresholdPreferenceQuery(
                route_id="route_a",
                metric_name="duration_s",
                threshold_value=1200.0,
                direction="lte",
            ),
            before_size=2,
            after_size=2,
            before_volume_proxy=0.5,
            after_volume_proxy=0.75,
        )


def test_preference_helpers_produce_conservative_summaries() -> None:
    summary = build_compatible_set_summary(
        route_ids=["route_a", "route_a", "route_b"],
        compatible_set_volume_proxy=1.7,
        necessary_best_prob=0.6,
        possible_best_prob=0.4,
        support_flag=False,
        support_reason="weak overlap",
    )
    weight_summary = summarize_compatible_weight_set(
        route_ids=["route_a", "route_b"],
        weights={"time": 2.0, "money": 0.0, "co2": 2.0},
    )
    shrink = build_preference_shrinkage_trace(
        query_index=0,
        query_type="threshold",
        before_size=3,
        after_size=2,
        before_volume_proxy=1.0,
        after_volume_proxy=0.75,
    )
    normalized = normalize_weight_vector({"time": 2.0, "money": 1.0, "co2": 1.0})

    assert summary.route_ids == ["route_a", "route_b"]
    assert summary.compatible_set_size == 2
    assert summary.necessary_best_prob == summary.possible_best_prob
    assert summary.necessary_best_route_ids == []
    assert summary.possible_best_route_ids == ["route_a", "route_b"]
    assert weight_summary.compatible_weight_count == 2
    assert shrink.predicted_shrinkage == 0.3333333333333333
    assert shrink.realized_shrinkage == 0.25
    assert normalized == {"time": 0.5, "money": 0.25, "co2": 0.25}


def test_preference_state_enforces_route_level_necessary_best_subset() -> None:
    state = PreferenceState(
        compatible_set_summary=CompatibleSetSummary(
            route_ids=["route_a", "route_b"],
            necessary_best_route_ids=["route_a"],
            possible_best_route_ids=["route_b"],
        )
    )

    assert state.compatible_set_summary.necessary_best_route_ids == ["route_a"]
    assert set(state.compatible_set_summary.possible_best_route_ids) == {"route_a", "route_b"}
    assert state.derived_invariants["no_necessary_best_without_possible_best"] is True


def test_route_response_serializes_preference_runtime_visibility() -> None:
    preference_state = build_preference_state(
        route_ids=["route_a", "route_b"],
        weights={"time": 2.0, "money": 1.0, "co2": 0.5},
        support_flag=False,
        support_reason="weak overlap",
    )
    preference_state.terminal_type = "certified"
    preference_query_trace = {
        "schema_version": "preference-query-trace-v1",
        "selected_route_id": "route_a",
        "selected_certificate_basis": "selected_certificate",
        "terminal_type": preference_state.terminal_type,
        "query_count": int(preference_state.query_count),
        "query_history": [],
        "shrinkage_trace": [],
        "compatible_set_summary": preference_state.compatible_set_summary.model_dump(mode="json"),
        "derived_invariants": dict(preference_state.derived_invariants),
        "contradiction_record": preference_state.contradiction_record.model_dump(mode="json"),
        "preference_irrelevance_proven": preference_state.preference_irrelevance_proven,
        "no_query_reason": preference_state.no_query_reason,
        "no_preference_query_reason": preference_state.no_query_reason,
        "targeted_challenger_route_id": None,
        "query_selection_reason": preference_state.no_query_reason,
        "provenance": {"selected_route_id": "route_a", "pipeline_mode": "voi"},
    }

    response = RouteResponse(
        selected=_route("route_a"),
        candidates=[_route("route_a"), _route("route_b")],
        run_id="run-preference",
        pipeline_mode="voi",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
        preference_state=preference_state,
        preference_query_trace=preference_query_trace,
    )

    encoded = json.loads(response.model_dump_json())
    assert encoded["preference_state"]["terminal_type"] == "certified"
    assert encoded["preference_state"]["compatible_set_summary"]["route_ids"] == ["route_a", "route_b"]
    assert encoded["preference_state"]["compatible_set_summary"]["support_flag"] is False
    assert encoded["preference_query_trace"]["schema_version"] == "preference-query-trace-v1"
    assert encoded["preference_query_trace"]["selected_route_id"] == "route_a"
    assert encoded["preference_query_trace"]["compatible_set_summary"]["route_ids"] == ["route_a", "route_b"]
    assert encoded["preference_query_trace"]["contradiction_record"]["contradiction_detected"] is False
    assert encoded["preference_query_trace"]["preference_irrelevance_proven"] is False
    assert encoded["preference_query_trace"]["no_query_reason"] == "preference_support_insufficient"
    assert encoded["preference_query_trace"]["no_preference_query_reason"] == "preference_support_insufficient"
    assert encoded["preference_summary"]["query_count"] == 0
    assert encoded["preference_summary"]["preference_irrelevance_proven"] is False
    assert encoded["preference_summary"]["no_query_reason"] == "preference_support_insufficient"
    assert encoded["preference_summary"]["no_preference_query_reason"] == "preference_support_insufficient"
    assert encoded["preference_summary"]["contradiction_record"]["contradiction_detected"] is False
    assert encoded["preference_summary"]["query_selection_reason"] == "preference_support_insufficient"


def test_route_response_autobuilds_preference_trace_with_no_preference_query_reason() -> None:
    preference_state = build_preference_state(
        route_ids=["route_a", "route_b"],
        weights={"time": 1.0, "money": 0.5},
        support_flag=False,
        support_reason="insufficient_overlap",
    )

    response = RouteResponse(
        selected=_route("route_a"),
        candidates=[_route("route_a"), _route("route_b")],
        run_id="run-preference-autotrace",
        pipeline_mode="voi",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
        preference_state=preference_state,
    )

    encoded = json.loads(response.model_dump_json())

    assert encoded["preference_query_trace"]["preference_irrelevance_proven"] is False
    assert encoded["preference_query_trace"]["no_query_reason"] == "preference_support_insufficient"
    assert encoded["preference_query_trace"]["no_preference_query_reason"] == "preference_support_insufficient"
