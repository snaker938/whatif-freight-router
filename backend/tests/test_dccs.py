from __future__ import annotations

from app.decision_critical import (
    DCCSConfig,
    build_candidate_ledger,
    record_refine_outcome,
    select_candidates,
    stable_candidate_id,
)


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
        "flip_probability",
        "predicted_refine_cost",
    }


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
    assert any(item.near_duplicate for item in result.skipped)


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

    assert [item.candidate_id for item in result.selected] == ["challenger_slow"]
    assert result.selected[0].decision == "refine"
    assert result.selected[0].decision_reason == "selected_by_challenger"
    assert result.skipped[0].decision_reason == "budget_exhausted"


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
