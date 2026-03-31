from __future__ import annotations

import math

import pytest

from app.decision_critical import (
    _direct_fallback_via_label_shrink_fraction,
    _legacy_predicted_refine_cost,
    _predicted_refine_cost,
    _bootstrap_score,
    _challenger_score,
    _normalised_distance,
    DCCSCandidateRecord,
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


def _score_only_record(
    candidate_id: str,
    *,
    objective_gap: float,
    mechanism_gap: float,
    overlap: float,
    stretch: float,
    time_regret_gap: float,
    time_preservation_bonus: float,
    predicted_refine_cost: float,
    flip_probability: float,
) -> DCCSCandidateRecord:
    return DCCSCandidateRecord(
        candidate_id=candidate_id,
        graph_path=(candidate_id,),
        graph_length_km=0.0,
        road_class_mix={},
        toll_share=0.0,
        terrain_burden=0.0,
        proxy_objective=(0.0, 0.0, 0.0),
        mechanism_descriptor={},
        proxy_confidence={},
        overlap=overlap,
        stretch=stretch,
        detour=max(0.0, stretch - 1.0),
        objective_gap=objective_gap,
        mechanism_gap=mechanism_gap,
        time_regret_gap=time_regret_gap,
        time_preservation_bonus=time_preservation_bonus,
        predicted_refine_cost=predicted_refine_cost,
        flip_probability=flip_probability,
        score_terms={},
        final_score=0.0,
        decision="skip",
        decision_reason="pending",
        mode="challenger",
        corridor_signature=candidate_id,
    )


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


def test_normalised_distance_stays_bounded_with_single_selected_anchor() -> None:
    anchor = (10.0, 10.0, 10.0)
    close = (10.1, 10.0, 10.0)
    farther = (20.0, 10.0, 10.0)
    reference_pool = [anchor, close, farther]

    close_distance = _normalised_distance(close, [anchor], reference_pool=reference_pool)
    farther_distance = _normalised_distance(farther, [anchor], reference_pool=reference_pool)

    assert math.isfinite(close_distance)
    assert math.isfinite(farther_distance)
    assert 0.0 < close_distance < farther_distance < 2.0


def test_bootstrap_score_does_not_reward_uniformly_worse_single_anchor_outliers() -> None:
    cfg = DCCSConfig(mode="bootstrap", search_budget=2)
    ledger = build_candidate_ledger(
        [
            _candidate(
                "anchor",
                path=["a1", "a2", "a3"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.8, "a_road_share": 0.2, "urban_share": 0.0},
                toll_share=0.02,
                terrain_burden=0.05,
                straight_line_km=3.2,
                mechanism={"motorway_share": 0.8, "toll_share": 0.02, "terrain_burden": 0.05},
            ),
            _candidate(
                "useful_diverse",
                path=["u1", "u2", "u3"],
                objective=(10.6, 9.5, 9.4),
                road_mix={"motorway_share": 0.2, "a_road_share": 0.5, "urban_share": 0.3},
                toll_share=0.10,
                terrain_burden=0.20,
                straight_line_km=3.1,
                mechanism={"motorway_share": 0.2, "toll_share": 0.10, "terrain_burden": 0.20},
            ),
            _candidate(
                "uniformly_worse_outlier",
                path=["w1", "w2", "w3"],
                objective=(45.0, 45.0, 45.0),
                road_mix={"motorway_share": 0.8, "a_road_share": 0.2, "urban_share": 0.0},
                toll_share=0.02,
                terrain_burden=0.05,
                straight_line_km=3.0,
                mechanism={"motorway_share": 0.8, "toll_share": 0.02, "terrain_burden": 0.05},
            ),
        ],
        config=cfg,
    )
    by_id = {record.candidate_id: record for record in ledger}

    useful_score = _bootstrap_score(
        by_id["useful_diverse"],
        selected=[by_id["anchor"]],
        candidate_pool=ledger,
        config=cfg,
    )
    outlier_score = _bootstrap_score(
        by_id["uniformly_worse_outlier"],
        selected=[by_id["anchor"]],
        candidate_pool=ledger,
        config=cfg,
    )

    assert math.isfinite(useful_score)
    assert math.isfinite(outlier_score)
    assert useful_score > outlier_score
    assert outlier_score < 1.0


def test_bootstrap_score_prefers_time_preserving_shorthaul_seed_over_extreme_detour() -> None:
    def _bootstrap_candidate(
        candidate_id: str,
        *,
        path: list[str],
        graph_length_km: float,
        straight_line_km: float,
        objective: tuple[float, float, float],
        road_mix: dict[str, float],
        mechanism: dict[str, float],
        confidence: dict[str, float],
        candidate_source_label: str,
    ) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "graph_path": path,
            "graph_length_km": graph_length_km,
            "straight_line_km": straight_line_km,
            "road_class_mix": road_mix,
            "toll_share": 0.0,
            "terrain_burden": 0.0,
            "proxy_objective": objective,
            "mechanism_descriptor": mechanism,
            "proxy_confidence": confidence,
            "candidate_source_label": candidate_source_label,
        }

    straight_line_km = 41.06
    candidates = [
        _bootstrap_candidate(
            "alternative_seed",
            path=["a1", "a2", "a3", "a4", "a5", "a6"],
            graph_length_km=70.6903,
            straight_line_km=straight_line_km,
            objective=(3949.5, 76.667028, 84.82836),
            road_mix={"motorway_share": 0.5119, "a_road_share": 0.274931, "urban_share": 0.02125, "other_share": 0.191919},
            mechanism={
                "motorway_share": 0.5119,
                "a_road_share": 0.274931,
                "urban_share": 0.02125,
                "toll_share": 0.0,
                "terrain_burden": 0.0,
                "speed_variability": 0.592296,
                "slow_segment_share": 0.212762,
                "shape_bend_density": 1.0,
                "shape_detour_factor": 1.0,
                "source_via_hint": 0.0,
                "source_alternatives_hint": 1.0,
                "source_exclude_toll_hint": 0.0,
                "source_exclude_motorway_hint": 0.0,
                "source_local_ors_hint": 0.0,
            },
            confidence={"time": 0.860952, "money": 0.9, "co2": 0.836175},
            candidate_source_label="fallback:alternatives:direct_k_raw_fallback",
        ),
        _bootstrap_candidate(
            "productive_via2",
            path=["v2a", "v2b", "v2c", "v2d", "v2e", "v2f"],
            graph_length_km=71.2585,
            straight_line_km=straight_line_km,
            objective=(4018.8, 77.575358, 85.5102),
            road_mix={"motorway_share": 0.537926, "a_road_share": 0.298802, "urban_share": 0.030512, "other_share": 0.13276},
            mechanism={
                "motorway_share": 0.537926,
                "a_road_share": 0.298802,
                "urban_share": 0.030512,
                "toll_share": 0.0,
                "terrain_burden": 0.0,
                "speed_variability": 0.576649,
                "slow_segment_share": 0.163272,
                "shape_bend_density": 1.0,
                "shape_detour_factor": 1.0,
                "source_via_hint": 1.0,
                "source_alternatives_hint": 0.0,
                "source_exclude_toll_hint": 0.0,
                "source_exclude_motorway_hint": 0.0,
                "source_local_ors_hint": 0.0,
            },
            confidence={"time": 0.863034, "money": 0.9, "co2": 0.834508},
            candidate_source_label="fallback:via:2:direct_k_raw_fallback",
        ),
        _bootstrap_candidate(
            "redundant_via3",
            path=["v3a", "v3b", "v3c", "v3d", "v3e", "v3f"],
            graph_length_km=126.9848,
            straight_line_km=straight_line_km,
            objective=(6968.5, 136.739564, 152.38176),
            road_mix={"motorway_share": 0.61266, "a_road_share": 0.169841, "urban_share": 0.015474, "other_share": 0.202025},
            mechanism={
                "motorway_share": 0.61266,
                "a_road_share": 0.169841,
                "urban_share": 0.015474,
                "toll_share": 0.0,
                "terrain_burden": 0.0,
                "speed_variability": 0.603893,
                "slow_segment_share": 0.217729,
                "shape_bend_density": 1.0,
                "shape_detour_factor": 1.0,
                "source_via_hint": 1.0,
                "source_alternatives_hint": 0.0,
                "source_exclude_toll_hint": 0.0,
                "source_exclude_motorway_hint": 0.0,
                "source_local_ors_hint": 0.0,
            },
            confidence={"time": 0.869013, "money": 0.9, "co2": 0.837215},
            candidate_source_label="fallback:via:3:direct_k_raw_fallback",
        ),
    ]

    cfg = DCCSConfig(mode="bootstrap", pipeline_variant="voi", search_budget=3)
    ledger = build_candidate_ledger(candidates, config=cfg)
    by_id = {record.candidate_id: record for record in ledger}

    productive_score = _bootstrap_score(
        by_id["productive_via2"],
        selected=[by_id["alternative_seed"]],
        candidate_pool=ledger,
        config=cfg,
    )
    detour_score = _bootstrap_score(
        by_id["redundant_via3"],
        selected=[by_id["alternative_seed"]],
        candidate_pool=ledger,
        config=cfg,
    )

    assert productive_score > detour_score


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


def test_challenger_score_prefers_objective_supported_candidate_over_mechanism_only_novelty() -> None:
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
            "objective_supported",
            path=["o1", "o2", "o3"],
            objective=(10.35, 9.55, 9.55),
            road_mix={"motorway_share": 0.45, "a_road_share": 0.35, "urban_share": 0.20},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.8,
            mechanism={"motorway_share": 0.45, "toll_share": 0.05, "terrain_burden": 0.10},
        ),
        _candidate(
            "mechanism_only_novelty",
            path=["m1", "m2", "m3"],
            objective=(10.08, 9.95, 9.95),
            road_mix={"motorway_share": 0.12, "a_road_share": 0.48, "urban_share": 0.40},
            toll_share=0.18,
            terrain_burden=0.25,
            straight_line_km=9.6,
            mechanism={"motorway_share": 0.12, "toll_share": 0.18, "terrain_burden": 0.25},
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

    assert [item.candidate_id for item in result.selected] == ["objective_supported"]
    assert result.selected[0].score_terms["objective_gap"] > result.skipped[0].score_terms["objective_gap"]
    assert result.selected[0].score_terms["mechanism_gap"] < result.skipped[0].score_terms["mechanism_gap"]
    assert result.selected[0].predicted_refine_cost <= result.skipped[0].predicted_refine_cost


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


def test_challenger_score_deprioritizes_mechanism_only_detour_under_budget_pressure() -> None:
    cfg = DCCSConfig(mode="challenger", pipeline_variant="dccs", search_budget=2)
    via2 = _score_only_record(
        "via2",
        objective_gap=0.07363930214347394,
        mechanism_gap=0.0969994162250475,
        overlap=0.09090909090909091,
        stretch=1.7353698194522296,
        time_regret_gap=0.13930940636162614,
        time_preservation_bonus=0.8606905936383739,
        predicted_refine_cost=44.9409122834497,
        flip_probability=0.9987137950298496,
    )
    via5 = _score_only_record(
        "via5",
        objective_gap=0.05691159682278773,
        mechanism_gap=0.030433786323755402,
        overlap=0.14285714285714285,
        stretch=1.7796219507618507,
        time_regret_gap=0.1532857061858592,
        time_preservation_bonus=0.8467142938141408,
        predicted_refine_cost=42.13225738101606,
        flip_probability=0.9981816224528427,
    )
    exclude_motorway = _score_only_record(
        "exclude_motorway",
        objective_gap=0.0,
        mechanism_gap=1.4626571726265865,
        overlap=0.14285714285714285,
        stretch=3.7968869303859254,
        time_regret_gap=1.7074048874525147,
        time_preservation_bonus=0.0,
        predicted_refine_cost=45.506567496071874,
        flip_probability=0.9999838034769649,
    )

    ranked = sorted(
        (via2, via5, exclude_motorway),
        key=lambda record: (-_challenger_score(record, config=cfg), record.candidate_id),
    )

    assert {record.candidate_id for record in ranked[:2]} == {"via2", "via5"}
    assert _challenger_score(via2, config=cfg) > _challenger_score(exclude_motorway, config=cfg)
    assert _challenger_score(via5, config=cfg) > _challenger_score(exclude_motorway, config=cfg)


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
    assert summary["refine_cost_mae_ms"] is not None
    assert 0.0 < summary["refine_cost_mae_ms"] < 15.0
    assert summary["refine_cost_mape"] is not None
    assert 0.0 < summary["refine_cost_mape"] < 0.25
    assert summary["refine_cost_rank_correlation"] == pytest.approx(1.0, rel=0.0, abs=1e-6)


def test_pipeline_variant_refine_cost_calibration_reflects_cache_sensitivity() -> None:
    candidate = _candidate(
        "cand_pipeline_variant",
        path=["p1", "p2", "p3", "p4", "p5"],
        objective=(125.0, 132.0, 138.0),
        road_mix={"motorway_share": 0.34, "a_road_share": 0.36, "urban_share": 0.30},
        toll_share=0.0,
        terrain_burden=0.0,
        straight_line_km=98.0,
        mechanism={
            "motorway_share": 0.34,
            "a_road_share": 0.36,
            "urban_share": 0.30,
            "toll_share": 0.0,
            "terrain_burden": 0.0,
            "slow_segment_share": 0.19,
            "speed_variability": 0.58,
            "shape_detour_factor": 0.28,
        },
    )
    candidate["candidate_source_label"] = "support_fallback:via:2:direct_k_raw_fallback"
    predicted = {}
    for pipeline_variant in ("dccs", "dccs_refc", "voi"):
        predicted[pipeline_variant] = build_candidate_ledger(
            [candidate],
            config=DCCSConfig(mode="challenger", pipeline_variant=pipeline_variant, search_budget=1),
        )[0].predicted_refine_cost

    assert predicted["dccs"] > 0.0
    assert predicted["dccs_refc"] > 0.0
    assert predicted["voi"] > 0.0
    assert predicted["voi"] < min(predicted["dccs"], predicted["dccs_refc"])
    assert predicted["dccs"] != pytest.approx(predicted["dccs_refc"], rel=0.0, abs=1e-9)


def test_direct_fallback_seed_observed_cost_reanchors_refine_cost_prediction() -> None:
    candidate = _candidate(
        "cand_seed_reanchor",
        path=["p1", "p2", "p3", "p4", "p5", "p6"],
        objective=(225.0, 231.0, 244.0),
        road_mix={"motorway_share": 0.71, "a_road_share": 0.21, "urban_share": 0.08},
        toll_share=0.0,
        terrain_burden=0.0,
        straight_line_km=82.0,
        mechanism={
            "motorway_share": 0.71,
            "a_road_share": 0.21,
            "urban_share": 0.08,
            "toll_share": 0.0,
            "terrain_burden": 0.0,
            "slow_segment_share": 0.14,
            "speed_variability": 0.38,
            "shape_detour_factor": 0.52,
            "source_via_hint": 1.0,
        },
    )
    candidate["candidate_source_label"] = "fallback:via:4:direct_k_raw_fallback"
    candidate["candidate_source_stage"] = "direct_k_raw_fallback"

    unseeded = _predicted_refine_cost(candidate, config=DCCSConfig(pipeline_variant="dccs_refc"))
    seeded_low = _predicted_refine_cost(
        {**candidate, "seed_observed_refine_cost_ms": 48.0},
        config=DCCSConfig(pipeline_variant="dccs_refc"),
    )
    seeded_high = _predicted_refine_cost(
        {**candidate, "seed_observed_refine_cost_ms": 210.0},
        config=DCCSConfig(pipeline_variant="dccs_refc"),
    )

    assert unseeded > 0.0
    assert seeded_low > 0.0
    assert seeded_high > 0.0
    assert seeded_low < seeded_high
    assert seeded_high > unseeded


def test_unlabeled_bootstrap_refine_cost_uses_pipeline_specific_legacy_shrink_factor() -> None:
    candidate = _candidate(
        "cand_unlabeled_bootstrap",
        path=["u1", "u2", "u3", "u4", "u5", "u6"],
        objective=(190.0, 198.0, 204.0),
        road_mix={"motorway_share": 0.78, "a_road_share": 0.20, "urban_share": 0.02},
        toll_share=0.0,
        terrain_burden=0.04,
        straight_line_km=58.0,
        mechanism={
            "motorway_share": 0.78,
            "a_road_share": 0.20,
            "urban_share": 0.02,
            "toll_share": 0.0,
            "terrain_burden": 0.04,
            "slow_segment_share": 0.05,
            "speed_variability": 0.12,
            "shape_detour_factor": 0.19,
        },
    )
    legacy_cost = _legacy_predicted_refine_cost(
        graph_length_km=float(candidate["graph_length_km"]),
        motorway_share=float(candidate["road_class_mix"]["motorway_share"]),
        urban_share=float(candidate["road_class_mix"]["urban_share"]),
        toll_share=float(candidate["toll_share"]),
        terrain_burden=float(candidate["terrain_burden"]),
        stretch=float(candidate["graph_length_km"]) / float(candidate["straight_line_km"]),
        path_nodes=float(len(candidate["graph_path"])),
    )

    predicted_refc = _predicted_refine_cost(candidate, config=DCCSConfig(pipeline_variant="dccs_refc"))
    predicted_voi = _predicted_refine_cost(candidate, config=DCCSConfig(pipeline_variant="voi"))
    labeled_candidate = dict(candidate)
    labeled_candidate["candidate_source_label"] = "support_fallback:alternatives:direct_k_raw_fallback"
    labeled_refc = _predicted_refine_cost(labeled_candidate, config=DCCSConfig(pipeline_variant="dccs_refc"))
    labeled_voi = _predicted_refine_cost(labeled_candidate, config=DCCSConfig(pipeline_variant="voi"))

    assert predicted_refc == pytest.approx(legacy_cost * 0.04, rel=0.0, abs=1e-9)
    assert predicted_voi == pytest.approx(legacy_cost * 0.066, rel=0.0, abs=1e-9)
    assert predicted_refc < legacy_cost
    assert predicted_voi < legacy_cost
    assert predicted_refc < labeled_refc
    assert predicted_voi < labeled_voi

def test_voi_refine_cost_penalizes_detour_heavier_via_fallbacks_over_alternatives() -> None:
    alternatives = _candidate(
        "cand_alt_fallback",
        path=["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12"],
        objective=(210.0, 210.0, 210.0),
        road_mix={"motorway_share": 0.51, "a_road_share": 0.28, "urban_share": 0.02},
        toll_share=0.0,
        terrain_burden=0.0,
        straight_line_km=56.0,
        mechanism={
            "motorway_share": 0.51,
            "a_road_share": 0.28,
            "urban_share": 0.02,
            "toll_share": 0.0,
            "terrain_burden": 0.0,
            "slow_segment_share": 0.19,
            "speed_variability": 0.59,
            "shape_detour_factor": 1.0,
            "source_alternatives_hint": 1.0,
        },
    )
    via = _candidate(
        "cand_via_fallback",
        path=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"],
        objective=(210.0, 210.0, 210.0),
        road_mix={"motorway_share": 0.54, "a_road_share": 0.25, "urban_share": 0.02},
        toll_share=0.0,
        terrain_burden=0.0,
        straight_line_km=45.0,
        mechanism={
            "motorway_share": 0.54,
            "a_road_share": 0.25,
            "urban_share": 0.02,
            "toll_share": 0.0,
            "terrain_burden": 0.0,
            "slow_segment_share": 0.18,
            "speed_variability": 0.46,
            "shape_detour_factor": 1.0,
            "source_via_hint": 1.0,
        },
    )
    alternatives["candidate_source_label"] = "fallback:alternatives:direct_k_raw_fallback"
    via["candidate_source_label"] = "fallback:via:1:direct_k_raw_fallback"

    ledger = build_candidate_ledger(
        [alternatives, via],
        config=DCCSConfig(mode="challenger", pipeline_variant="voi", search_budget=2),
    )
    by_id = {record.candidate_id: record for record in ledger}

    assert by_id["cand_via_fallback"].predicted_refine_cost > by_id["cand_alt_fallback"].predicted_refine_cost


def test_shorthaul_direct_fallback_via_label_shrink_targets_short_routes_only() -> None:
    short_shrink = _direct_fallback_via_label_shrink_fraction(
        pipeline_variant="voi",
        source_label="fallback:via:1:direct_k_raw_fallback",
        source_stage="direct_k_raw_fallback",
        graph_length_km=102.7476,
        stretch=1.7733226193955527,
        motorway_share=0.580317,
        urban_share=0.038172,
        toll_share=0.0,
        terrain_burden=0.0,
        path_nodes=11.0,
    )
    long_shrink = _direct_fallback_via_label_shrink_fraction(
        pipeline_variant="voi",
        source_label="fallback:via:8:direct_k_raw_fallback",
        source_stage="direct_k_raw_fallback",
        graph_length_km=155.4374,
        stretch=2.6826967960325523,
        motorway_share=0.481767,
        urban_share=0.034975,
        toll_share=0.0,
        terrain_burden=0.0,
        path_nodes=11.0,
    )
    alt_shrink = _direct_fallback_via_label_shrink_fraction(
        pipeline_variant="voi",
        source_label="fallback:alternatives:direct_k_raw_fallback",
        source_stage="direct_k_raw_fallback",
        graph_length_km=89.6403,
        stretch=1.5471035002219338,
        motorway_share=0.57947,
        urban_share=0.028826,
        toll_share=0.0,
        terrain_burden=0.0,
        path_nodes=12.0,
    )

    assert short_shrink > 0.5
    assert long_shrink == 0.0
    assert alt_shrink == 0.0


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
