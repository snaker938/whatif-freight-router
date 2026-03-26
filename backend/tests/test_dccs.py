from __future__ import annotations

import pytest

from app.decision_critical import (
    DCCSConfig,
    build_candidate_ledger,
    record_refine_outcome,
    select_baseline_candidates,
    select_baseline_result,
    select_candidates,
    summarize_refine_outcomes,
    stable_candidate_id,
)

pytestmark = pytest.mark.thesis_modules


def _candidate(
    candidate_id: str,
    *,
    path: list[str],
    objective: tuple[float, float, float],
    road_mix: dict[str, float],
    toll_share: float,
    terrain_burden: float,
    straight_line_km: float,
    mechanism: dict[str, float],
    confidence: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "graph_path": path,
        "graph_length_km": float(sum(objective) / 9.0),
        "straight_line_km": straight_line_km,
        "road_class_mix": road_mix,
        "toll_share": toll_share,
        "terrain_burden": terrain_burden,
        "proxy_objective": objective,
        "mechanism_descriptor": mechanism,
        "proxy_confidence": confidence or {"time": 0.9, "money": 0.8, "co2": 0.85},
    }


def test_candidate_ledger_is_stable_and_auditable() -> None:
    cand = _candidate(
        "cand_a",
        path=["n1", "n2", "n3"],
        objective=(10.0, 11.0, 12.0),
        road_mix={"motorway_share": 0.7, "a_road_share": 0.2, "urban_share": 0.1},
        toll_share=0.05,
        terrain_burden=0.15,
        straight_line_km=8.5,
        mechanism={"motorway_share": 0.7, "toll_share": 0.05, "terrain_burden": 0.15},
    )
    record = build_candidate_ledger([cand], config=DCCSConfig(mode="bootstrap", search_budget=1))[0]

    assert record.candidate_id == stable_candidate_id(cand)
    assert record.graph_path == ("n1", "n2", "n3")
    assert record.proxy_objective == (10.0, 11.0, 12.0)
    assert 0.0 <= record.flip_probability <= 1.0
    assert record.predicted_refine_cost > 0.0
    assert set(record.score_terms) == {
        "objective_gap",
        "mechanism_gap",
        "overlap_penalty",
        "stretch_penalty",
        "time_regret_gap",
        "time_preservation_bonus",
        "time_bonus_scale",
        "flip_probability",
        "predicted_refine_cost",
        "objective_extremeness",
        "comparator_seed_penalty",
    }
    assert record.comparator_seeded is False


def test_bootstrap_mode_selects_diverse_representatives_under_budget() -> None:
    candidates = [
        _candidate(
            "cand_a",
            path=["a1", "a2", "a3"],
            objective=(10.0, 10.0, 10.0),
            road_mix={"motorway_share": 0.8, "a_road_share": 0.2, "urban_share": 0.0},
            toll_share=0.02,
            terrain_burden=0.05,
            straight_line_km=9.5,
            mechanism={"motorway_share": 0.8, "toll_share": 0.02, "terrain_burden": 0.05},
        ),
        _candidate(
            "cand_b",
            path=["b1", "b2", "b3"],
            objective=(11.5, 11.0, 11.0),
            road_mix={"motorway_share": 0.2, "a_road_share": 0.6, "urban_share": 0.2},
            toll_share=0.10,
            terrain_burden=0.20,
            straight_line_km=9.2,
            mechanism={"motorway_share": 0.2, "toll_share": 0.10, "terrain_burden": 0.20},
        ),
        _candidate(
            "cand_c",
            path=["b1", "b2", "b3"],
            objective=(14.0, 13.5, 13.0),
            road_mix={"motorway_share": 0.2, "a_road_share": 0.6, "urban_share": 0.2},
            toll_share=0.12,
            terrain_burden=0.22,
            straight_line_km=9.2,
            mechanism={"motorway_share": 0.2, "toll_share": 0.12, "terrain_burden": 0.22},
        ),
    ]
    result = select_candidates(
        candidates,
        config=DCCSConfig(mode="bootstrap", search_budget=2, near_duplicate_threshold=0.80),
    )

    selected_ids = {item.candidate_id for item in result.selected}
    assert selected_ids == {"cand_a", "cand_c"}
    assert result.summary["transition_reason"].startswith("bootstrap_seeding")
    assert result.summary["selected_count"] == 2
    assert result.summary["frontier_additions"] >= 0
    assert any(item.candidate_id == "cand_b" for item in result.skipped)
    assert result.summary["selected_corridor_count"] == 2


def test_challenger_mode_prefers_high_flip_probability_per_cost() -> None:
    frontier = [
        _candidate(
            "frontier_a",
            path=["f1", "f2", "f3"],
            objective=(12.0, 12.0, 12.0),
            road_mix={"motorway_share": 0.6, "a_road_share": 0.3, "urban_share": 0.1},
            toll_share=0.04,
            terrain_burden=0.10,
            straight_line_km=10.0,
            mechanism={"motorway_share": 0.6, "toll_share": 0.04, "terrain_burden": 0.10},
        )
    ]
    candidates = [
        _candidate(
            "challenger_fast",
            path=["c1", "c2", "c3"],
            objective=(9.5, 9.4, 9.3),
            road_mix={"motorway_share": 0.4, "a_road_share": 0.5, "urban_share": 0.1},
            toll_share=0.03,
            terrain_burden=0.10,
            straight_line_km=9.3,
            mechanism={"motorway_share": 0.4, "toll_share": 0.03, "terrain_burden": 0.10},
        ),
        _candidate(
            "challenger_slow",
            path=["s1", "s2", "s3", "s4"],
            objective=(9.0, 8.8, 8.7),
            road_mix={"motorway_share": 0.1, "a_road_share": 0.4, "urban_share": 0.5},
            toll_share=0.18,
            terrain_burden=0.25,
            straight_line_km=8.0,
            mechanism={"motorway_share": 0.1, "toll_share": 0.18, "terrain_burden": 0.25},
        ),
    ]
    result = select_candidates(
        candidates,
        frontier=frontier,
        config=DCCSConfig(mode="challenger", search_budget=1),
    )

    assert [item.candidate_id for item in result.selected] == ["challenger_fast"]
    assert result.selected[0].decision == "refine"
    assert result.selected[0].decision_reason == "selected_by_challenger"
    assert result.skipped[0].decision_reason == "budget_exhausted"


def test_challenger_score_prefers_lower_time_regret_when_other_gains_are_similar() -> None:
    frontier = [
        _candidate(
            "frontier_anchor",
            path=["f1", "f2", "f3"],
            objective=(10.0, 10.0, 10.0),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=10.0,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        )
    ]
    candidates = [
        _candidate(
            "time_preserving",
            path=["t1", "t2", "t3"],
            objective=(10.25, 9.60, 9.60),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.7,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        ),
        _candidate(
            "time_regret_heavier",
            path=["s1", "s2", "s3"],
            objective=(10.85, 9.35, 9.35),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.7,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        ),
    ]

    result = select_candidates(
        candidates,
        frontier=frontier,
        config=DCCSConfig(
            mode="challenger",
            search_budget=1,
            challenger_time_preservation_weight=1.5,
        ),
    )

    assert [item.candidate_id for item in result.selected] == ["time_preserving"]
    assert result.selected[0].score_terms["time_preservation_bonus"] > result.skipped[0].score_terms["time_preservation_bonus"]
    assert result.selected[0].score_terms["time_bonus_scale"] > 0.0
    assert result.selected[0].score_terms["time_regret_gap"] < result.skipped[0].score_terms["time_regret_gap"]


def test_challenger_score_still_respects_mechanism_diversity_when_time_is_tied() -> None:
    frontier = [
        _candidate(
            "frontier_anchor",
            path=["f1", "f2", "f3"],
            objective=(10.0, 10.0, 10.0),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=10.0,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        )
    ]
    candidates = [
        _candidate(
            "diverse",
            path=["d1", "d2", "d3"],
            objective=(10.35, 9.55, 9.55),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.6,
            mechanism={"motorway_share": 0.2, "toll_share": 0.15, "terrain_burden": 0.20},
        ),
        _candidate(
            "homogeneous",
            path=["h1", "h2", "h3"],
            objective=(10.35, 9.55, 9.55),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.6,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        ),
    ]

    result = select_candidates(
        candidates,
        frontier=frontier,
        refined=[
            _candidate(
                "ref_anchor",
                path=["r1", "r2", "r3"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=10.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=1),
    )

    assert [item.candidate_id for item in result.selected] == ["diverse"]
    assert result.selected[0].score_terms["mechanism_gap"] > result.skipped[0].score_terms["mechanism_gap"]


def test_challenger_score_does_not_let_time_bonus_override_clearly_stronger_hard_case_challenger() -> None:
    frontier = [
        _candidate(
            "frontier_anchor",
            path=["f1", "f2", "f3"],
            objective=(10.0, 10.0, 10.0),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=10.0,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        )
    ]
    candidates = [
        _candidate(
            "representative_like",
            path=["r1", "r2", "r3"],
            objective=(10.05, 9.88, 9.90),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.9,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        ),
        _candidate(
            "hard_case_challenger",
            path=["h1", "h2", "h3"],
            objective=(10.65, 8.95, 8.95),
            road_mix={"motorway_share": 0.2, "a_road_share": 0.5, "urban_share": 0.3},
            toll_share=0.18,
            terrain_burden=0.25,
            straight_line_km=9.2,
            mechanism={"motorway_share": 0.2, "toll_share": 0.18, "terrain_burden": 0.25},
        ),
    ]

    result = select_candidates(
        candidates,
        frontier=frontier,
        refined=[
            _candidate(
                "ref_anchor",
                path=["x1", "x2", "x3"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=10.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(
            mode="challenger",
            search_budget=1,
            challenger_time_preservation_weight=1.5,
        ),
    )

    assert [item.candidate_id for item in result.selected] == ["hard_case_challenger"]
    assert result.selected[0].score_terms["time_preservation_bonus"] < result.skipped[0].score_terms["time_preservation_bonus"]
    assert result.selected[0].score_terms["time_bonus_scale"] > result.skipped[0].score_terms["time_bonus_scale"]
    assert result.selected[0].score_terms["objective_gap"] > result.skipped[0].score_terms["objective_gap"]
    assert result.selected[0].score_terms["mechanism_gap"] > result.skipped[0].score_terms["mechanism_gap"]


def test_observed_refine_cost_is_attached_to_ledger_records() -> None:
    cand = _candidate(
        "cand_observed",
        path=["o1", "o2", "o3"],
        objective=(10.0, 10.5, 10.8),
        road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
        toll_share=0.08,
        terrain_burden=0.12,
        straight_line_km=8.8,
        mechanism={"motorway_share": 0.5, "toll_share": 0.08, "terrain_burden": 0.12},
    )
    record = build_candidate_ledger([cand], config=DCCSConfig(mode="challenger", search_budget=1))[0]
    updated = record_refine_outcome(
        record,
        observed_refine_cost=record.predicted_refine_cost + 2.5,
        frontier_added=True,
    )

    assert updated.observed_refine_cost == record.predicted_refine_cost + 2.5
    assert updated.refine_cost_ratio > 1.0
    assert updated.decision_reason == "frontier_addition"


def test_missing_observed_refine_cost_does_not_fake_perfect_calibration() -> None:
    cand = _candidate(
        "cand_missing_observed",
        path=["m1", "m2", "m3"],
        objective=(11.0, 11.5, 11.8),
        road_mix={"motorway_share": 0.4, "a_road_share": 0.4, "urban_share": 0.2},
        toll_share=0.0,
        terrain_burden=0.1,
        straight_line_km=3.2,
        mechanism={"motorway_share": 0.4, "a_road_share": 0.4, "urban_share": 0.2},
    )
    record = build_candidate_ledger([cand], config=DCCSConfig(mode="challenger", search_budget=1))[0]
    updated = record_refine_outcome(record, observed_refine_cost=None, redundant=True)

    assert updated.observed_refine_cost is None
    assert updated.refine_cost_ratio is None
    assert updated.refine_cost_error is None
    assert updated.decision_reason == "non_challenger_redundant"


def test_refine_cost_summary_reports_sane_deterministic_metrics() -> None:
    candidates = [
        _candidate(
            "cand_1",
            path=["a1", "a2", "a3"],
            objective=(10.0, 10.2, 10.4),
            road_mix={"motorway_share": 0.7, "a_road_share": 0.2, "urban_share": 0.1},
            toll_share=0.03,
            terrain_burden=0.05,
            straight_line_km=8.6,
            mechanism={"motorway_share": 0.7, "toll_share": 0.03, "terrain_burden": 0.05},
        ),
        _candidate(
            "cand_2",
            path=["b1", "b2", "b3", "b4"],
            objective=(12.0, 12.4, 12.8),
            road_mix={"motorway_share": 0.4, "a_road_share": 0.3, "urban_share": 0.3},
            toll_share=0.08,
            terrain_burden=0.12,
            straight_line_km=9.1,
            mechanism={"motorway_share": 0.4, "toll_share": 0.08, "terrain_burden": 0.12},
        ),
        _candidate(
            "cand_3",
            path=["c1", "c2", "c3", "c4", "c5"],
            objective=(14.0, 14.5, 14.8),
            road_mix={"motorway_share": 0.2, "a_road_share": 0.4, "urban_share": 0.4},
            toll_share=0.15,
            terrain_burden=0.25,
            straight_line_km=9.5,
            mechanism={"motorway_share": 0.2, "toll_share": 0.15, "terrain_burden": 0.25},
        ),
    ]
    ledger = [
        record_refine_outcome(
            build_candidate_ledger([candidate], config=DCCSConfig(mode="challenger", search_budget=1))[0],
            observed_refine_cost=observed,
            frontier_added=index == 0,
        )
        for index, (candidate, observed) in enumerate(zip(candidates, (12.0, 15.0, 18.0), strict=True))
    ]

    summary = summarize_refine_outcomes(ledger)

    assert summary["refine_cost_sample_count"] == 3
    assert summary["refine_cost_mae_ms"] == pytest.approx(1.1057407407407414, rel=0.0, abs=1e-6)
    assert summary["refine_cost_mape"] is not None
    assert 0.0 < summary["refine_cost_mape"] < 0.2
    assert summary["refine_cost_rank_correlation"] == pytest.approx(1.0, rel=0.0, abs=1e-6)


def test_challenger_score_is_monotone_in_overlap_and_cost() -> None:
    frontier = [
        _candidate(
            "frontier_a",
            path=["f1", "f2", "f3"],
            objective=(12.0, 12.0, 12.0),
            road_mix={"motorway_share": 0.6, "a_road_share": 0.3, "urban_share": 0.1},
            toll_share=0.04,
            terrain_burden=0.10,
            straight_line_km=10.0,
            mechanism={"motorway_share": 0.6, "toll_share": 0.04, "terrain_burden": 0.10},
        )
    ]
    low_cost = _candidate(
        "cand_low_cost",
        path=["x1", "x2", "x3"],
        objective=(10.0, 10.1, 10.2),
        road_mix={"motorway_share": 0.8, "a_road_share": 0.2, "urban_share": 0.0},
        toll_share=0.01,
        terrain_burden=0.03,
        straight_line_km=9.8,
        mechanism={"motorway_share": 0.8, "toll_share": 0.01, "terrain_burden": 0.03},
    )
    high_cost = _candidate(
        "cand_high_cost",
        path=["y1", "y2", "y3", "y4", "y5"],
        objective=(10.0, 10.1, 10.2),
        road_mix={"motorway_share": 0.1, "a_road_share": 0.4, "urban_share": 0.5},
        toll_share=0.20,
        terrain_burden=0.30,
        straight_line_km=7.0,
        mechanism={"motorway_share": 0.1, "toll_share": 0.20, "terrain_burden": 0.30},
    )

    low_overlap_score = select_candidates(
        [low_cost],
        frontier=frontier,
        config=DCCSConfig(mode="challenger", search_budget=1),
    ).selected[0].final_score
    high_overlap_score = select_candidates(
        [low_cost],
        refined=[{"candidate_id": "ref", "graph_path": ["x1", "x2", "x3"], "proxy_objective": (10.0, 10.0, 10.0)}],
        config=DCCSConfig(mode="challenger", search_budget=1, near_duplicate_threshold=0.2),
    ).selected[0].final_score
    high_cost_score = select_candidates(
        [high_cost],
        frontier=frontier,
        config=DCCSConfig(mode="challenger", search_budget=1),
    ).selected[0].final_score

    assert low_overlap_score > high_overlap_score
    assert low_overlap_score > high_cost_score


def test_challenger_score_is_monotone_in_objective_gap() -> None:
    frontier = [
        _candidate(
            "frontier_anchor",
            path=["f1", "f2", "f3"],
            objective=(12.0, 12.0, 12.0),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=10.0,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        )
    ]
    better = _candidate(
        "cand_better",
        path=["b1", "b2", "b3"],
        objective=(9.0, 9.0, 9.0),
        road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
        toll_share=0.05,
        terrain_burden=0.10,
        straight_line_km=8.5,
        mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
    )
    weaker = _candidate(
        "cand_weaker",
        path=["w1", "w2", "w3"],
        objective=(11.6, 11.6, 11.6),
        road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
        toll_share=0.05,
        terrain_burden=0.10,
        straight_line_km=8.5,
        mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
    )

    result = select_candidates(
        [better, weaker],
        frontier=frontier,
        config=DCCSConfig(mode="challenger", search_budget=2),
    )

    scores = {row.candidate_id: row.final_score for row in result.selected}
    assert scores["cand_better"] > scores["cand_weaker"]


def test_objective_gap_does_not_reward_uniformly_worse_novelty() -> None:
    frontier = [
        _candidate(
            "frontier_anchor",
            path=["f1", "f2", "f3"],
            objective=(10.0, 10.0, 10.0),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=10.0,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        )
    ]
    worse_and_far = _candidate(
        "cand_far_but_worse",
        path=["z1", "z2", "z3"],
        objective=(13.0, 13.2, 13.4),
        road_mix={"motorway_share": 0.2, "a_road_share": 0.4, "urban_share": 0.4},
        toll_share=0.20,
        terrain_burden=0.25,
        straight_line_km=8.0,
        mechanism={"motorway_share": 0.2, "toll_share": 0.20, "terrain_burden": 0.25},
    )
    slightly_better = _candidate(
        "cand_slightly_better",
        path=["b1", "b2", "b3"],
        objective=(9.8, 10.1, 10.0),
        road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
        toll_share=0.05,
        terrain_burden=0.10,
        straight_line_km=9.5,
        mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
    )

    result = select_candidates(
        [worse_and_far, slightly_better],
        frontier=frontier,
        config=DCCSConfig(mode="challenger", search_budget=2),
    )

    scores = {row.candidate_id: row.final_score for row in result.selected}
    assert scores["cand_slightly_better"] > scores["cand_far_but_worse"]


def test_baseline_candidate_policies_are_deterministic() -> None:
    candidates = [
        _candidate(
            candidate_id=f"cand_{idx}",
            path=[f"n{idx}", f"n{idx+1}", f"n{idx+2}"],
            objective=(10.0 + idx, 11.0 + idx, 12.0 + idx),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.0,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        )
        for idx in range(4)
    ]

    assert select_baseline_candidates(candidates, budget=2, policy="first_n") == ["cand_0", "cand_1"]
    assert select_baseline_candidates(candidates, budget=2, policy="random_n", seed=7) == select_baseline_candidates(
        candidates,
        budget=2,
        policy="random_n",
        seed=7,
    )
    assert select_baseline_candidates(candidates, budget=3, policy="corridor_uniform") == select_baseline_candidates(
        candidates,
        budget=3,
        policy="uniform_corridor_n",
    )


def test_selected_candidate_ledger_carries_scored_decisions() -> None:
    frontier = [
        _candidate(
            "frontier_anchor",
            path=["f1", "f2", "f3"],
            objective=(10.0, 10.0, 10.0),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=10.0,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        )
    ]
    candidates = [
        _candidate(
            "cand_a",
            path=["a1", "a2", "a3"],
            objective=(9.2, 9.3, 9.4),
            road_mix={"motorway_share": 0.7, "a_road_share": 0.2, "urban_share": 0.1},
            toll_share=0.02,
            terrain_burden=0.05,
            straight_line_km=8.8,
            mechanism={"motorway_share": 0.7, "toll_share": 0.02, "terrain_burden": 0.05},
        ),
        _candidate(
            "cand_b",
            path=["b1", "b2", "b3"],
            objective=(9.8, 9.9, 10.0),
            road_mix={"motorway_share": 0.4, "a_road_share": 0.4, "urban_share": 0.2},
            toll_share=0.04,
            terrain_burden=0.09,
            straight_line_km=8.7,
            mechanism={"motorway_share": 0.4, "toll_share": 0.04, "terrain_burden": 0.09},
        ),
    ]

    result = select_candidates(
        candidates,
        frontier=frontier,
        config=DCCSConfig(mode="challenger", search_budget=1),
    )

    ledger = {record.candidate_id: record for record in result.candidate_ledger}
    assert ledger["cand_a"].decision == "refine"
    assert ledger["cand_a"].selection_rank == 0
    assert ledger["cand_a"].final_score > 0.0
    assert ledger["cand_b"].decision == "skip"
    assert ledger["cand_b"].decision_reason in {"budget_exhausted", "not_selected"}


def test_baseline_result_exposes_selection_ranks_and_policy_summary() -> None:
    candidates = [
        _candidate(
            candidate_id=f"cand_{idx}",
            path=[f"n{idx}", f"n{idx+1}", f"n{idx+2}"],
            objective=(10.0 + idx, 11.0 + idx, 12.0 + idx),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.0,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        )
        for idx in range(4)
    ]

    result = select_baseline_result(candidates, budget=2, policy="first_n", seed=11)

    assert [row.candidate_id for row in result.selected] == ["cand_0", "cand_1"]
    assert [row.selection_rank for row in result.selected] == [0, 1]
    assert result.summary["selection_policy"] == "first_n"


def test_dccs_summary_flags_prediction_stage_and_supports_observed_rollup() -> None:
    candidates = [
        _candidate(
            "cand_obs_a",
            path=["a1", "a2", "a3"],
            objective=(10.0, 10.5, 10.8),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.08,
            terrain_burden=0.12,
            straight_line_km=8.8,
            mechanism={"motorway_share": 0.5, "toll_share": 0.08, "terrain_burden": 0.12},
        ),
        _candidate(
            "cand_obs_b",
            path=["b1", "b2", "b3"],
            objective=(10.2, 10.7, 10.9),
            road_mix={"motorway_share": 0.4, "a_road_share": 0.4, "urban_share": 0.2},
            toll_share=0.10,
            terrain_burden=0.10,
            straight_line_km=8.6,
            mechanism={"motorway_share": 0.4, "toll_share": 0.10, "terrain_burden": 0.10},
        ),
    ]

    result = select_candidates(
        candidates,
        config=DCCSConfig(mode="challenger", search_budget=2),
    )

    assert result.summary["dc_yield_is_predicted"] is True
    assert result.summary["metric_stage"] == "pre_refinement_prediction"
    assert result.summary["observed_metrics_available"] is False
    assert result.summary["predicted_dc_yield"] == result.summary["dc_yield"]
    assert 0.0 <= result.summary["dc_yield"] <= 1.0
    assert 0.0 <= result.summary["challenger_hit_rate"] <= 1.0
    assert 0.0 <= result.summary["frontier_gain_per_refinement"] <= 1.0
    assert result.summary["selected_count"] == len(result.selected)

    observed = summarize_refine_outcomes(
        [
            record_refine_outcome(result.selected[0], observed_refine_cost=6.0, frontier_added=True),
            record_refine_outcome(result.selected[1], observed_refine_cost=7.5, redundant=True),
        ]
    )

    assert observed["observed_metrics_available"] is True
    assert observed["metric_stage"] == "post_refinement_observed"
    assert observed["observed_frontier_additions"] == 1
    assert observed["observed_redundant_count"] == 1
    assert observed["observed_dc_yield"] == 0.5
    assert 0.0 <= observed["observed_dc_yield"] <= 1.0
