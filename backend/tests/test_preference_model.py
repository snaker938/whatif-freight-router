from __future__ import annotations

import pytest

from app.abstention import AbstentionRecord
from app.preference_queries import PreferenceQuery, suggest_preference_queries
from app.preference_state import build_preference_state
from app.preference_update import PreferenceUpdate, apply_preference_update, apply_query_answer
from app.preference_model import ElicitedConstraint, TimeGuard

pytestmark = pytest.mark.thesis_modules


def _route(
    route_id: str,
    *,
    duration_s: float,
    money: float,
    co2: float,
    certified: bool = False,
    certificate: float | None = None,
    threshold: float | None = 0.8,
    toll_cost: float = 0.0,
    weather_delay_s: float = 0.0,
    incident_delay_s: float = 0.0,
    uncertainty_mass: float = 0.0,
) -> dict[str, object]:
    return {
        "id": route_id,
        "metrics": {
            "duration_s": duration_s,
            "monetary_cost": money,
            "emissions_kg": co2,
            "distance_km": 100.0,
            "weather_delay_s": weather_delay_s,
            "incident_delay_s": incident_delay_s,
        },
        "segment_breakdown": ([{"toll_cost": toll_cost}] if toll_cost else []),
        "uncertainty": {
            "std_duration_s": duration_s * uncertainty_mass,
            "std_monetary_cost": max(1.0, money) * uncertainty_mass,
            "std_emissions_kg": max(1.0, co2) * uncertainty_mass,
        },
        "certification": (
            {
                "certificate": certificate if certificate is not None else (0.9 if certified else 0.55),
                "certified": certified,
                "threshold": threshold,
            }
        ),
    }


def test_preference_state_respects_toll_toggle_and_time_guard() -> None:
    request = {
        "weights": {"time": 0.6, "money": 0.3, "co2": 0.1},
        "cost_toggles": {"use_tolls": False},
        "optimization_mode": "expected_value",
        "departure_time_utc": "2026-04-03T09:00:00Z",
    }
    frontier = [
        _route("tolled_fast", duration_s=3200.0, money=110.0, co2=70.0, toll_cost=18.0),
        _route("free_ok", duration_s=3550.0, money=115.0, co2=68.0, certified=True, certificate=0.92),
        _route("free_slow", duration_s=4300.0, money=100.0, co2=62.0),
    ]

    state = build_preference_state(
        request,
        frontier,
        elicited_constraints=(
            ElicitedConstraint.veto("uncertified", source="user"),
            ElicitedConstraint.time_guard(TimeGuard(max_duration_s=3700.0)),
        ),
    )

    assert state.compatible_set.route_ids == ("free_ok",)
    assert state.top_route_id() == "free_ok"
    assert "toggle_use_tolls" in state.compatible_set.blocked_reasons["tolled_fast"]
    assert "veto_uncertified" in state.compatible_set.blocked_reasons["tolled_fast"]
    assert "time_guard:max_duration_s" in state.compatible_set.blocked_reasons["free_slow"]
    assert state.has_time_guard() is True
    assert state.vetoed_targets == ("uncertified",)
    assert state.certified_only_required is True


def test_preference_state_detects_irrelevant_axes() -> None:
    request = {"weights": {"time": 1.0, "money": 1.0, "co2": 1.0}}
    frontier = [
        _route("route_a", duration_s=3600.0, money=100.0, co2=50.0),
        _route("route_b", duration_s=3720.0, money=103.0, co2=51.0),
    ]

    state = build_preference_state(request, frontier)

    assert {"money", "co2"}.issubset(set(state.irrelevant_axes))


def test_query_suggestions_include_certified_focus_and_tradeoff() -> None:
    request = {"weights": {"time": 0.7, "money": 0.2, "co2": 0.1}}
    frontier = [
        _route("fast_uncertain", duration_s=3600.0, money=110.0, co2=78.0, certified=False, certificate=0.52),
        _route("slower_certified", duration_s=4300.0, money=85.0, co2=60.0, certified=True, certificate=0.91),
    ]

    state = build_preference_state(request, frontier, selected_route_id="fast_uncertain")
    queries = suggest_preference_queries(state, limit=5)
    query_kinds = {query.kind for query in queries}

    assert "certified_focus" in query_kinds
    assert "objective_tradeoff" in query_kinds


def test_certified_only_preference_prioritizes_certified_focus_query() -> None:
    request = {
        "weights": {"time": 0.8, "money": 0.1, "co2": 0.1},
        "certificate_threshold": 0.8,
    }
    frontier = [
        _route("route_a", duration_s=3400.0, money=120.0, co2=80.0, certified=False, certificate=0.4),
        _route("route_b", duration_s=3600.0, money=105.0, co2=78.0, certified=False, certificate=0.55),
    ]

    state = build_preference_state(
        request,
        frontier,
        elicited_constraints=(ElicitedConstraint.veto("uncertified", source="user"),),
        stop_reason="budget_exhausted",
    )
    queries = suggest_preference_queries(state, limit=3)

    assert queries
    assert queries[0].kind == "certified_focus"
    assert queries[0].metadata["certified_only_required"] is True
    assert queries[0].metadata["vetoed_targets"] == ["uncertified"]
    assert set(queries[0].route_ids) == {"route_a", "route_b"}


def test_certified_focus_answer_can_trigger_typed_abstention_hint() -> None:
    request = {
        "weights": {"time": 0.8, "money": 0.1, "co2": 0.1},
        "certificate_threshold": 0.8,
    }
    frontier = [
        _route("route_a", duration_s=3400.0, money=120.0, co2=80.0, certified=False, certificate=0.4),
        _route("route_b", duration_s=3600.0, money=105.0, co2=78.0, certified=False, certificate=0.55),
    ]

    state = build_preference_state(request, frontier, stop_reason="budget_exhausted")
    query = PreferenceQuery(
        key="certified_focus",
        kind="certified_focus",
        prompt="Require a certified route?",
        rationale="No certified route currently leads the frontier.",
        options=("prefer_certified", "allow_uncertified"),
    )

    updated = apply_query_answer(state, query, "prefer_certified")

    assert updated.wants_certified_only() is True
    assert updated.compatible_set.route_ids == ()
    assert updated.certified_only_required is True
    assert "typed_abstention_recommended" in {hint.code for hint in updated.stop_hints}


def test_preference_state_derives_actionable_typed_abstention_record() -> None:
    request = {
        "weights": {"time": 0.8, "money": 0.1, "co2": 0.1},
        "certificate_threshold": 0.8,
    }
    frontier = [
        _route("route_a", duration_s=3400.0, money=120.0, co2=80.0, certified=False, certificate=0.4),
        _route("route_b", duration_s=3600.0, money=105.0, co2=78.0, certified=False, certificate=0.55),
    ]

    state = build_preference_state(
        request,
        frontier,
        elicited_constraints=(ElicitedConstraint.veto("uncertified", source="user"),),
        stop_reason="budget_exhausted",
    )
    abstention = AbstentionRecord.from_preference_state(state)

    assert abstention.reason_code == "typed_abstention_recommended"
    assert abstention.trigger_metric == "certified_route_count"
    assert abstention.recommended_action == "expand_worlds"
    assert abstention.severity == "high"
    assert abstention.observed_value == 0.0


def test_preference_update_can_shift_dominant_objective() -> None:
    request = {"weights": {"time": 0.5, "money": 0.3, "co2": 0.2}}
    state = build_preference_state(request, [_route("route_a", duration_s=3600.0, money=100.0, co2=60.0)])

    updated = apply_preference_update(
        state,
        update=PreferenceUpdate(focus_objective="co2", focus_delta=0.25),
    )

    assert updated.weights.dominant_objective() == "co2"
