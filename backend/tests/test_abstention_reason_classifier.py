from __future__ import annotations

import json

from app.abstention import build_abstention_record, classify_typed_abstention_reason


def test_typed_abstention_reason_classifier_covers_all_classes() -> None:
    cases = [
        (
            {"stop_reason": "budget_exhausted"},
            "uncertified_due_to_budget",
        ),
        (
            {
                "stop_reason": "search_incomplete_no_action_worth_it",
                "credible_search_uncertainty": True,
                "search_completeness_score": 0.3,
            },
            "uncertified_due_to_search",
        ),
        (
            {
                "stop_reason": "refresh_revealed_fragility",
                "credible_evidence_uncertainty": True,
                "top_fragility_families": ["weather"],
            },
            "uncertified_due_to_evidence",
        ),
        (
            {
                "stop_reason": "preference_query_blocked",
                "support_reason": "preference ambiguity remained",
            },
            "uncertified_due_to_preference",
        ),
        (
            {
                "stop_reason": "support_unavailable",
                "support_flag": False,
                "support_reason": "out_of_support_world_model",
            },
            "uncertified_due_to_out_of_support_world_model",
        ),
        (
            {
                "stop_reason": "iteration_cap_reached",
                "model_assumption": "terminal_certification_threshold",
                "support_flag": True,
            },
            "uncertified_due_to_model_assumption",
        ),
    ]

    for signals, expected in cases:
        assert classify_typed_abstention_reason(**signals) == expected


def test_typed_abstention_record_round_trips() -> None:
    record = build_abstention_record(
        stop_reason="budget_exhausted",
        support_flag=False,
        support_reason="out_of_support_world_model",
        budget_channel="search/evidence",
        detail={"origin": "unit-test"},
    )
    encoded = json.loads(record.model_dump_json())
    assert encoded["reason_code"] == "uncertified_due_to_budget"
    assert encoded["terminal_type"] == "typed_abstention"
    assert encoded["support_flag"] is False
    assert encoded["detail"]["origin"] == "unit-test"
