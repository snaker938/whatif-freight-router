from __future__ import annotations

import math

import pytest

from app.candidate_bounds import CandidateEnvelope, CandidateEnvelopeBounds
from app.candidate_criticality import CandidateCriticalityEstimate
from app.decision_critical import (
    _direct_fallback_via_label_shrink_fraction,
    _legacy_predicted_refine_cost,
    _predicted_refine_cost,
    _bootstrap_score,
    _challenger_score,
    initial_bootstrap_budget,
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
    search_deficiency_score: float = 0.0,
    hidden_challenger_score: float = 0.0,
    anti_collapse_quota: float = 0.0,
    long_corridor_search_completeness: float = 1.0,
    dominance_margin: float = 0.0,
    comparator_seeded: bool = False,
    near_duplicate: bool = False,
    corridor_signature: str | None = None,
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
        corridor_signature=corridor_signature or candidate_id,
        candidate_source_stage=("preemptive_comparator_seed" if comparator_seeded else None),
        comparator_seeded=comparator_seeded,
        near_duplicate=near_duplicate,
        dominance_margin=dominance_margin,
        search_deficiency_score=search_deficiency_score,
        hidden_challenger_score=hidden_challenger_score,
        anti_collapse_quota=anti_collapse_quota,
        long_corridor_search_completeness=long_corridor_search_completeness,
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
    assert isinstance(record.candidate_envelope, CandidateEnvelope)
    assert isinstance(record.criticality_estimate, CandidateCriticalityEstimate)
    assert record.candidate_envelope.objective_bounds["time"].lower <= record.proxy_objective[0]
    assert record.candidate_envelope.objective_bounds["time"].upper >= record.proxy_objective[0]
    assert record.criticality_estimate.decision_critical is True
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
    assert {
        "safe_elimination_reason",
        "dominance_margin",
        "search_deficiency_score",
        "hidden_challenger_score",
        "anti_collapse_quota",
        "long_corridor_search_completeness",
    }.isdisjoint(record.score_terms)
    assert record.comparator_seeded is False
    payload = record.as_dict()
    assert payload["candidate_envelope"]["schema_version"] == "dccs-envelope-v1"
    assert payload["criticality_estimate"]["schema_version"] == "dccs-criticality-v1"


def test_direct_record_instantiation_backfills_explicit_state() -> None:
    record = _score_only_record(
        "explicit_state",
        objective_gap=0.3,
        mechanism_gap=0.2,
        overlap=0.15,
        stretch=1.1,
        time_regret_gap=0.1,
        time_preservation_bonus=0.4,
        predicted_refine_cost=6.0,
        flip_probability=0.65,
    )

    assert isinstance(record.candidate_envelope, CandidateEnvelope)
    assert isinstance(record.candidate_envelope.refine_cost_bounds, CandidateEnvelopeBounds)
    assert record.candidate_envelope.refine_cost_bounds.lower >= 0.0
    assert isinstance(record.criticality_estimate, CandidateCriticalityEstimate)
    assert record.criticality_estimate.criticality_band in {"medium", "high"}
    assert record.criticality_estimate.flip_probability == pytest.approx(0.65)


def test_candidate_record_exposes_elimination_and_search_deficiency_signals() -> None:
    dominating = _candidate(
        "dominating",
        path=["d1", "d2", "d3"],
        objective=(9.0, 9.0, 9.0),
        road_mix={"motorway_share": 0.7, "a_road_share": 0.2, "urban_share": 0.1},
        toll_share=0.02,
        terrain_burden=0.05,
        straight_line_km=9.0,
        mechanism={"motorway_share": 0.7, "toll_share": 0.02, "terrain_burden": 0.05},
    )
    dominated = _candidate(
        "dominated",
        path=["w1", "w2", "w3", "w4"],
        objective=(12.0, 12.0, 12.0),
        road_mix={"motorway_share": 0.2, "a_road_share": 0.4, "urban_share": 0.4},
        toll_share=0.14,
        terrain_burden=0.22,
        straight_line_km=6.5,
        mechanism={"motorway_share": 0.2, "toll_share": 0.14, "terrain_burden": 0.22},
    )

    ledger = build_candidate_ledger([dominating, dominated], frontier=[dominating], refined=[dominating])
    record = next(item for item in ledger if item.candidate_id == "dominated")

    assert record.safe_elimination_reason == "dominated_by_frontier"
    assert record.dominance_margin > 0.0
    assert record.dominating_candidate_ids == ("dominating",)
    assert 0.0 <= record.search_deficiency_score <= 1.0
    assert 0.0 <= record.hidden_challenger_score <= 1.0
    assert 0.0 <= record.anti_collapse_quota <= 1.0
    assert 0.0 <= record.long_corridor_search_completeness <= 1.0
    assert record.candidate_envelope.safe_elimination_reason == "dominated_by_frontier"
    assert record.criticality_estimate.safe_elimination_reason == "dominated_by_frontier"
    assert {
        "safe_elimination_reason",
        "dominance_margin",
        "search_deficiency_score",
        "hidden_challenger_score",
        "anti_collapse_quota",
        "long_corridor_search_completeness",
    }.isdisjoint(record.score_terms)
    payload = record.as_dict()
    assert {
        "safe_elimination_reason",
        "dominance_margin",
        "search_deficiency_score",
        "hidden_challenger_score",
        "anti_collapse_quota",
        "long_corridor_search_completeness",
    }.isdisjoint(payload["score_terms"])
    assert payload["dominating_candidate_ids"] == ("dominating",)
    assert payload["candidate_envelope"]["safe_elimination_reason"] == "dominated_by_frontier"
    assert payload["candidate_envelope"]["dominance_margin"] == pytest.approx(record.dominance_margin)
    assert payload["criticality_estimate"]["search_deficiency_score"] == pytest.approx(record.search_deficiency_score)
    assert payload["criticality_estimate"]["anti_collapse_pressure"] == pytest.approx(record.anti_collapse_quota)


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


def test_challenger_score_can_promote_plausible_collapse_prone_long_corridor_seed() -> None:
    cfg = DCCSConfig(mode="challenger", pipeline_variant="dccs_refc", search_budget=2)
    collapse_prone_seed = _score_only_record(
        "collapse_prone_seed",
        objective_gap=0.08,
        mechanism_gap=0.16,
        overlap=0.05,
        stretch=1.18,
        time_regret_gap=0.16,
        time_preservation_bonus=0.84,
        predicted_refine_cost=34.5,
        flip_probability=0.90,
        search_deficiency_score=0.92,
        hidden_challenger_score=0.89,
        anti_collapse_quota=0.87,
        long_corridor_search_completeness=0.28,
        comparator_seeded=True,
    )
    local_incumbent_like = _score_only_record(
        "local_incumbent_like",
        objective_gap=0.10,
        mechanism_gap=0.11,
        overlap=0.06,
        stretch=1.14,
        time_regret_gap=0.22,
        time_preservation_bonus=0.72,
        predicted_refine_cost=31.0,
        flip_probability=0.82,
    )

    assert _challenger_score(collapse_prone_seed, config=cfg) > _challenger_score(
        local_incumbent_like,
        config=cfg,
    )


def test_challenger_score_keeps_unproductive_long_corridor_seed_behind_plausible_candidate() -> None:
    cfg = DCCSConfig(mode="challenger", pipeline_variant="dccs_refc", search_budget=2)
    unproductive_seed = _score_only_record(
        "unproductive_seed",
        objective_gap=0.0,
        mechanism_gap=1.25,
        overlap=0.05,
        stretch=3.1,
        time_regret_gap=1.7,
        time_preservation_bonus=0.0,
        predicted_refine_cost=34.0,
        flip_probability=0.99,
        search_deficiency_score=0.95,
        hidden_challenger_score=0.90,
        anti_collapse_quota=0.88,
        long_corridor_search_completeness=0.20,
        comparator_seeded=True,
    )
    plausible_supported = _score_only_record(
        "plausible_supported",
        objective_gap=0.06,
        mechanism_gap=0.09,
        overlap=0.10,
        stretch=1.35,
        time_regret_gap=0.20,
        time_preservation_bonus=0.80,
        predicted_refine_cost=36.0,
        flip_probability=0.74,
    )

    assert _challenger_score(plausible_supported, config=cfg) > _challenger_score(
        unproductive_seed,
        config=cfg,
    )


def test_bootstrap_score_rewards_dominance_supported_comparator_seed_under_corridor_pressure() -> None:
    cfg = DCCSConfig(mode="bootstrap", pipeline_variant="dccs", search_budget=2)
    incumbent = _score_only_record(
        "incumbent",
        objective_gap=0.04,
        mechanism_gap=0.05,
        overlap=0.10,
        stretch=1.08,
        time_regret_gap=0.18,
        time_preservation_bonus=0.78,
        predicted_refine_cost=23.0,
        flip_probability=0.68,
        corridor_signature="corridor_a",
    )
    dominance_seed = _score_only_record(
        "dominance_seed",
        objective_gap=0.07,
        mechanism_gap=0.11,
        overlap=0.08,
        stretch=1.12,
        time_regret_gap=0.10,
        time_preservation_bonus=0.88,
        predicted_refine_cost=25.0,
        flip_probability=0.84,
        dominance_margin=0.11,
        search_deficiency_score=0.93,
        hidden_challenger_score=0.89,
        anti_collapse_quota=0.86,
        long_corridor_search_completeness=0.24,
        comparator_seeded=True,
        corridor_signature="corridor_a",
    )
    local_same_corridor = _score_only_record(
        "local_same_corridor",
        objective_gap=0.08,
        mechanism_gap=0.08,
        overlap=0.08,
        stretch=1.10,
        time_regret_gap=0.18,
        time_preservation_bonus=0.72,
        predicted_refine_cost=22.0,
        flip_probability=0.79,
        corridor_signature="corridor_a",
    )

    dominance_score = _bootstrap_score(
        dominance_seed,
        selected=[incumbent],
        candidate_pool=[incumbent, dominance_seed, local_same_corridor],
        config=cfg,
    )
    local_score = _bootstrap_score(
        local_same_corridor,
        selected=[incumbent],
        candidate_pool=[incumbent, dominance_seed, local_same_corridor],
        config=cfg,
    )

    assert dominance_score > local_score


def test_bootstrap_score_deprioritizes_slow_non_dominating_tradeoff() -> None:
    cfg = DCCSConfig(mode="bootstrap", pipeline_variant="dccs", search_budget=2)
    incumbent = _score_only_record(
        "incumbent",
        objective_gap=0.03,
        mechanism_gap=0.04,
        overlap=0.10,
        stretch=1.05,
        time_regret_gap=0.18,
        time_preservation_bonus=0.78,
        predicted_refine_cost=18.0,
        flip_probability=0.60,
        corridor_signature="corridor_a",
    )
    slow_tradeoff = _score_only_record(
        "slow_tradeoff",
        objective_gap=0.14,
        mechanism_gap=0.09,
        overlap=0.08,
        stretch=1.12,
        time_regret_gap=0.94,
        time_preservation_bonus=0.06,
        predicted_refine_cost=19.0,
        flip_probability=0.82,
        corridor_signature="corridor_b",
    )
    time_preserving = _score_only_record(
        "time_preserving",
        objective_gap=0.07,
        mechanism_gap=0.08,
        overlap=0.08,
        stretch=1.10,
        time_regret_gap=0.10,
        time_preservation_bonus=0.90,
        predicted_refine_cost=20.0,
        flip_probability=0.70,
        corridor_signature="corridor_c",
    )

    slow_score = _bootstrap_score(
        slow_tradeoff,
        selected=[incumbent],
        candidate_pool=[incumbent, slow_tradeoff, time_preserving],
        config=cfg,
    )
    time_score = _bootstrap_score(
        time_preserving,
        selected=[incumbent],
        candidate_pool=[incumbent, slow_tradeoff, time_preserving],
        config=cfg,
    )

    assert time_score > slow_score


def test_candidate_criticality_rewards_dominance_and_collapse_signals() -> None:
    baseline = _score_only_record(
        "baseline",
        objective_gap=0.0,
        mechanism_gap=0.08,
        overlap=0.18,
        stretch=1.08,
        time_regret_gap=0.22,
        time_preservation_bonus=0.65,
        predicted_refine_cost=18.0,
        flip_probability=0.35,
    )
    pressured = _score_only_record(
        "pressured",
        objective_gap=0.0,
        mechanism_gap=0.08,
        overlap=0.18,
        stretch=1.08,
        time_regret_gap=0.22,
        time_preservation_bonus=0.65,
        predicted_refine_cost=18.0,
        flip_probability=0.35,
        dominance_margin=0.10,
        search_deficiency_score=0.88,
        hidden_challenger_score=0.86,
        anti_collapse_quota=0.81,
        long_corridor_search_completeness=0.18,
    )

    assert pressured.criticality_estimate is not None
    assert baseline.criticality_estimate is not None
    assert pressured.criticality_estimate.criticality_score > baseline.criticality_estimate.criticality_score
    assert pressured.criticality_estimate.decision_critical is True
    assert baseline.criticality_estimate.decision_critical is False


def test_candidate_criticality_deprioritizes_slow_non_dominating_tradeoff() -> None:
    slow_tradeoff = _score_only_record(
        "slow_tradeoff",
        objective_gap=0.12,
        mechanism_gap=0.08,
        overlap=0.16,
        stretch=1.10,
        time_regret_gap=0.92,
        time_preservation_bonus=0.08,
        predicted_refine_cost=18.0,
        flip_probability=0.68,
    )
    time_preserving = _score_only_record(
        "time_preserving",
        objective_gap=0.08,
        mechanism_gap=0.08,
        overlap=0.16,
        stretch=1.10,
        time_regret_gap=0.12,
        time_preservation_bonus=0.88,
        predicted_refine_cost=18.0,
        flip_probability=0.62,
    )

    assert slow_tradeoff.criticality_estimate is not None
    assert time_preserving.criticality_estimate is not None
    assert time_preserving.criticality_estimate.criticality_score > slow_tradeoff.criticality_estimate.criticality_score


def test_challenger_score_deprioritizes_slow_non_dominating_tradeoff_even_with_larger_objective_gap() -> None:
    cfg = DCCSConfig(mode="challenger", pipeline_variant="dccs", search_budget=2)
    slow_tradeoff = _score_only_record(
        "slow_tradeoff",
        objective_gap=0.14,
        mechanism_gap=0.09,
        overlap=0.08,
        stretch=1.12,
        time_regret_gap=0.95,
        time_preservation_bonus=0.05,
        predicted_refine_cost=24.0,
        flip_probability=0.82,
    )
    time_preserving = _score_only_record(
        "time_preserving",
        objective_gap=0.07,
        mechanism_gap=0.08,
        overlap=0.08,
        stretch=1.10,
        time_regret_gap=0.12,
        time_preservation_bonus=0.88,
        predicted_refine_cost=25.0,
        flip_probability=0.72,
    )

    assert _challenger_score(time_preserving, config=cfg) > _challenger_score(
        slow_tradeoff,
        config=cfg,
    )


def test_challenger_selection_spreads_across_corridors_when_scores_are_close() -> None:
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
            "corridor_a_primary",
            path=["s", "a1", "m", "a2", "e"],
            objective=(9.45, 9.45, 9.45),
            road_mix={"motorway_share": 0.46, "a_road_share": 0.34, "urban_share": 0.20},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.6,
            mechanism={"motorway_share": 0.46, "toll_share": 0.05, "terrain_burden": 0.10},
        ),
        _candidate(
            "corridor_a_secondary",
            path=["s", "b1", "m", "b2", "e"],
            objective=(9.48, 9.44, 9.46),
            road_mix={"motorway_share": 0.45, "a_road_share": 0.35, "urban_share": 0.20},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.6,
            mechanism={"motorway_share": 0.45, "toll_share": 0.05, "terrain_burden": 0.10},
        ),
        _candidate(
            "corridor_b_supported",
            path=["q", "c1", "n", "c2", "r"],
            objective=(9.52, 9.40, 9.44),
            road_mix={"motorway_share": 0.30, "a_road_share": 0.45, "urban_share": 0.25},
            toll_share=0.08,
            terrain_burden=0.12,
            straight_line_km=9.4,
            mechanism={"motorway_share": 0.30, "toll_share": 0.08, "terrain_burden": 0.12},
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
        config=DCCSConfig(mode="challenger", search_budget=2),
    )

    assert {item.candidate_id for item in result.selected} == {"corridor_a_primary", "corridor_b_supported"}
    assert len({item.corridor_signature for item in result.selected}) == 2


def test_challenger_selection_keeps_clearly_stronger_same_corridor_candidate() -> None:
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
            "corridor_a_primary",
            path=["s", "a1", "m", "a2", "e"],
            objective=(9.35, 9.30, 9.30),
            road_mix={"motorway_share": 0.48, "a_road_share": 0.32, "urban_share": 0.20},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.6,
            mechanism={"motorway_share": 0.48, "toll_share": 0.05, "terrain_burden": 0.10},
        ),
        _candidate(
            "corridor_a_secondary",
            path=["s", "b1", "m", "b2", "e"],
            objective=(9.40, 9.36, 9.36),
            road_mix={"motorway_share": 0.46, "a_road_share": 0.34, "urban_share": 0.20},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.5,
            mechanism={"motorway_share": 0.46, "toll_share": 0.05, "terrain_burden": 0.10},
        ),
        _candidate(
            "corridor_b_weak",
            path=["q", "c1", "n", "c2", "r"],
            objective=(9.90, 9.82, 9.80),
            road_mix={"motorway_share": 0.24, "a_road_share": 0.41, "urban_share": 0.35},
            toll_share=0.10,
            terrain_burden=0.15,
            straight_line_km=9.1,
            mechanism={"motorway_share": 0.24, "toll_share": 0.10, "terrain_burden": 0.15},
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
        config=DCCSConfig(mode="challenger", search_budget=2),
    )

    assert {item.candidate_id for item in result.selected} == {"corridor_a_primary", "corridor_a_secondary"}
    assert "corridor_b_weak" in {item.candidate_id for item in result.skipped}


def test_initial_bootstrap_budget_preserves_challenger_budget_for_dccs_and_refc() -> None:
    assert initial_bootstrap_budget(
        total_search_budget=2,
        pipeline_variant="dccs",
        bootstrap_seed_size=3,
    ) == 1
    assert initial_bootstrap_budget(
        total_search_budget=3,
        pipeline_variant="dccs_refc",
        bootstrap_seed_size=3,
    ) == 2
    assert initial_bootstrap_budget(
        total_search_budget=6,
        pipeline_variant="dccs_refc",
        bootstrap_seed_size=3,
    ) == 3


def test_initial_bootstrap_budget_respects_reserved_diversity_slots_and_legacy_mode() -> None:
    assert initial_bootstrap_budget(
        total_search_budget=6,
        pipeline_variant="dccs",
        bootstrap_seed_size=5,
        reserve_diversity_slots=2,
    ) == 4
    assert initial_bootstrap_budget(
        total_search_budget=6,
        pipeline_variant="legacy",
        bootstrap_seed_size=3,
        reserve_diversity_slots=0,
    ) == 6


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
    assert updated.criticality_estimate is not None
    assert updated.criticality_estimate.observed_refine_cost == pytest.approx(updated.observed_refine_cost)
    assert updated.criticality_estimate.refine_cost_error == pytest.approx(updated.refine_cost_error)


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


def test_preemptive_comparator_seed_observed_cost_reanchors_refine_cost_prediction() -> None:
    candidate = _candidate(
        "cand_preemptive_seed_reanchor",
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
            "source_alternatives_hint": 1.0,
        },
    )
    candidate["candidate_source_label"] = "preemptive:osrm:alternatives:preemptive_comparator_seed"
    candidate["candidate_source_stage"] = "preemptive_comparator_seed"

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
    assert seeded_low < unseeded
    assert unseeded < seeded_high


def test_comparator_seed_penalty_relents_for_time_competitive_nonduplicate_seed() -> None:
    cfg = DCCSConfig(
        mode="challenger",
        pipeline_variant="dccs_refc",
        comparator_seed_penalty_weight=0.45,
    )
    incumbent = _score_only_record(
        "incumbent",
        objective_gap=0.06,
        mechanism_gap=0.05,
        overlap=0.16,
        stretch=1.04,
        time_regret_gap=0.03,
        time_preservation_bonus=0.97,
        predicted_refine_cost=0.28,
        flip_probability=0.44,
        dominance_margin=0.08,
        corridor_signature="corridor_incumbent",
    )
    competitive_seed = _score_only_record(
        "competitive_seed",
        objective_gap=0.08,
        mechanism_gap=0.07,
        overlap=0.14,
        stretch=1.03,
        time_regret_gap=0.02,
        time_preservation_bonus=0.98,
        predicted_refine_cost=0.30,
        flip_probability=0.46,
        search_deficiency_score=0.20,
        hidden_challenger_score=0.18,
        anti_collapse_quota=0.12,
        long_corridor_search_completeness=0.84,
        dominance_margin=0.14,
        comparator_seeded=True,
        corridor_signature="corridor_fast_seed",
    )
    competitive_nonseed = _score_only_record(
        "competitive_nonseed",
        objective_gap=competitive_seed.objective_gap,
        mechanism_gap=competitive_seed.mechanism_gap,
        overlap=competitive_seed.overlap,
        stretch=competitive_seed.stretch,
        time_regret_gap=competitive_seed.time_regret_gap,
        time_preservation_bonus=competitive_seed.time_preservation_bonus,
        predicted_refine_cost=competitive_seed.predicted_refine_cost,
        flip_probability=competitive_seed.flip_probability,
        search_deficiency_score=competitive_seed.search_deficiency_score,
        hidden_challenger_score=competitive_seed.hidden_challenger_score,
        anti_collapse_quota=competitive_seed.anti_collapse_quota,
        long_corridor_search_completeness=competitive_seed.long_corridor_search_completeness,
        dominance_margin=competitive_seed.dominance_margin,
        comparator_seeded=False,
        corridor_signature="corridor_fast_nonseed",
    )
    weak_duplicate_seed = _score_only_record(
        "weak_duplicate_seed",
        objective_gap=0.02,
        mechanism_gap=0.02,
        overlap=0.91,
        stretch=1.12,
        time_regret_gap=0.19,
        time_preservation_bonus=0.62,
        predicted_refine_cost=0.30,
        flip_probability=0.18,
        search_deficiency_score=0.04,
        hidden_challenger_score=0.02,
        anti_collapse_quota=0.01,
        long_corridor_search_completeness=0.96,
        dominance_margin=0.01,
        comparator_seeded=True,
        near_duplicate=True,
        corridor_signature="corridor_incumbent",
    )

    bootstrap_seed_score = _bootstrap_score(
        competitive_seed,
        selected=[incumbent],
        candidate_pool=[incumbent, competitive_seed, competitive_nonseed, weak_duplicate_seed],
        config=cfg,
    )
    bootstrap_nonseed_score = _bootstrap_score(
        competitive_nonseed,
        selected=[incumbent],
        candidate_pool=[incumbent, competitive_seed, competitive_nonseed, weak_duplicate_seed],
        config=cfg,
    )
    bootstrap_weak_score = _bootstrap_score(
        weak_duplicate_seed,
        selected=[incumbent],
        candidate_pool=[incumbent, competitive_seed, competitive_nonseed, weak_duplicate_seed],
        config=cfg,
    )

    challenger_seed_score = _challenger_score(competitive_seed, config=cfg, selected=[incumbent])
    challenger_nonseed_score = _challenger_score(competitive_nonseed, config=cfg, selected=[incumbent])
    challenger_weak_score = _challenger_score(weak_duplicate_seed, config=cfg, selected=[incumbent])

    assert bootstrap_seed_score == pytest.approx(bootstrap_nonseed_score)
    assert challenger_seed_score == pytest.approx(challenger_nonseed_score)
    assert bootstrap_weak_score < bootstrap_seed_score
    assert challenger_weak_score < challenger_seed_score


def test_time_competitive_nonduplicate_comparator_seed_does_not_lose_solely_on_seed_status() -> None:
    cfg = DCCSConfig(
        mode="challenger",
        pipeline_variant="dccs_refc",
        comparator_seed_penalty_weight=0.45,
    )
    incumbent = _score_only_record(
        "incumbent_edge",
        objective_gap=0.05,
        mechanism_gap=0.04,
        overlap=0.20,
        stretch=1.03,
        time_regret_gap=0.05,
        time_preservation_bonus=0.95,
        predicted_refine_cost=0.28,
        flip_probability=0.40,
        dominance_margin=0.03,
        corridor_signature="corridor_incumbent_edge",
    )
    edge_seed = _score_only_record(
        "edge_seed",
        objective_gap=0.04,
        mechanism_gap=0.04,
        overlap=0.30,
        stretch=1.04,
        time_regret_gap=0.08,
        time_preservation_bonus=0.92,
        predicted_refine_cost=0.30,
        flip_probability=0.34,
        search_deficiency_score=0.70,
        hidden_challenger_score=0.70,
        anti_collapse_quota=0.70,
        long_corridor_search_completeness=0.50,
        dominance_margin=0.02,
        comparator_seeded=True,
        corridor_signature="corridor_fast_seed_edge",
    )
    edge_nonseed = _score_only_record(
        "edge_nonseed",
        objective_gap=edge_seed.objective_gap,
        mechanism_gap=edge_seed.mechanism_gap,
        overlap=edge_seed.overlap,
        stretch=edge_seed.stretch,
        time_regret_gap=edge_seed.time_regret_gap,
        time_preservation_bonus=edge_seed.time_preservation_bonus,
        predicted_refine_cost=edge_seed.predicted_refine_cost,
        flip_probability=edge_seed.flip_probability,
        search_deficiency_score=edge_seed.search_deficiency_score,
        hidden_challenger_score=edge_seed.hidden_challenger_score,
        anti_collapse_quota=edge_seed.anti_collapse_quota,
        long_corridor_search_completeness=edge_seed.long_corridor_search_completeness,
        dominance_margin=edge_seed.dominance_margin,
        comparator_seeded=False,
        corridor_signature="corridor_fast_nonseed_edge",
    )

    bootstrap_seed_score = _bootstrap_score(
        edge_seed,
        selected=[incumbent],
        candidate_pool=[incumbent, edge_seed, edge_nonseed],
        config=cfg,
    )
    bootstrap_nonseed_score = _bootstrap_score(
        edge_nonseed,
        selected=[incumbent],
        candidate_pool=[incumbent, edge_seed, edge_nonseed],
        config=cfg,
    )
    challenger_seed_score = _challenger_score(edge_seed, config=cfg, selected=[incumbent])
    challenger_nonseed_score = _challenger_score(edge_nonseed, config=cfg, selected=[incumbent])

    assert bootstrap_seed_score == pytest.approx(bootstrap_nonseed_score)
    assert challenger_seed_score == pytest.approx(challenger_nonseed_score)


def test_selected_comparator_seed_records_reduced_applied_penalty_when_time_competitive() -> None:
    frontier = [
        _candidate(
            "frontier_anchor",
            path=["f1", "f2", "f3"],
            objective=(10.0, 10.0, 10.0),
            road_mix={"motorway_share": 0.55, "a_road_share": 0.30, "urban_share": 0.15},
            toll_share=0.04,
            terrain_burden=0.08,
            straight_line_km=10.0,
            mechanism={"motorway_share": 0.55, "toll_share": 0.04, "terrain_burden": 0.08},
        )
    ]
    comparator_seed = _candidate(
        "competitive_seed_route",
        path=["c1", "c2", "c3"],
        objective=(10.22, 9.55, 9.55),
        road_mix={"motorway_share": 0.58, "a_road_share": 0.30, "urban_share": 0.12},
        toll_share=0.03,
        terrain_burden=0.08,
        straight_line_km=9.8,
        mechanism={"motorway_share": 0.58, "toll_share": 0.03, "terrain_burden": 0.08},
    )
    comparator_seed["candidate_source_label"] = "preemptive:repo_local_ors:secondary_seed"
    comparator_seed["candidate_source_stage"] = "preemptive_comparator_seed"
    detour = _candidate(
        "detour_candidate",
        path=["d1", "d2", "d3", "d4"],
        objective=(10.90, 9.40, 9.40),
        road_mix={"motorway_share": 0.22, "a_road_share": 0.28, "urban_share": 0.50},
        toll_share=0.16,
        terrain_burden=0.24,
        straight_line_km=9.7,
        mechanism={"motorway_share": 0.22, "toll_share": 0.16, "terrain_burden": 0.24},
    )

    result = select_candidates(
        [comparator_seed, detour],
        frontier=frontier,
        config=DCCSConfig(
            mode="challenger",
            search_budget=1,
            pipeline_variant="dccs_refc",
            comparator_seed_penalty_weight=0.45,
        ),
    )

    ledger = {record.candidate_id: record for record in result.candidate_ledger}
    assert ledger["competitive_seed_route"].final_score > 0.0
    assert 0.0 <= ledger["competitive_seed_route"].score_terms["comparator_seed_penalty"] < 0.45


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


def test_dccs_control_state_reports_collapse_pressure_and_long_corridor_signals() -> None:
    dominating = _candidate(
        "dominating",
        path=["d1", "d2", "d3"],
        objective=(9.0, 9.0, 9.0),
        road_mix={"motorway_share": 0.7, "a_road_share": 0.2, "urban_share": 0.1},
        toll_share=0.02,
        terrain_burden=0.05,
        straight_line_km=9.0,
        mechanism={"motorway_share": 0.7, "toll_share": 0.02, "terrain_burden": 0.05},
    )
    hidden = _candidate(
        "hidden",
        path=["h1", "h2", "h3", "h4", "h5"],
        objective=(8.5, 8.2, 8.1),
        road_mix={"motorway_share": 0.35, "a_road_share": 0.35, "urban_share": 0.30},
        toll_share=0.04,
        terrain_burden=0.08,
        straight_line_km=42.0,
        mechanism={
            "motorway_share": 0.35,
            "a_road_share": 0.35,
            "urban_share": 0.30,
            "toll_share": 0.04,
            "terrain_burden": 0.08,
            "slow_segment_share": 0.22,
            "speed_variability": 0.31,
            "shape_detour_factor": 0.42,
        },
    )
    long_corridor = _candidate(
        "long_corridor",
        path=["l1", "l2", "l3", "l4", "l5", "l6"],
        objective=(13.5, 13.4, 13.6),
        road_mix={"motorway_share": 0.55, "a_road_share": 0.25, "urban_share": 0.20},
        toll_share=0.06,
        terrain_burden=0.18,
        straight_line_km=4.0,
        mechanism={
            "motorway_share": 0.55,
            "a_road_share": 0.25,
            "urban_share": 0.20,
            "toll_share": 0.06,
            "terrain_burden": 0.18,
            "slow_segment_share": 0.25,
            "speed_variability": 0.49,
            "shape_detour_factor": 0.74,
        },
    )

    result = select_candidates(
        [dominating, hidden, long_corridor],
        frontier=[dominating],
        config=DCCSConfig(mode="bootstrap", search_budget=2, bootstrap_seed_size=1),
    )

    assert result.control_state is not None
    assert result.control_state.candidate_count == 3
    assert result.control_state.safe_elimination_count >= 1
    assert result.control_state.hidden_challenger_count >= 1
    assert 0.0 <= result.control_state.anti_collapse_quota <= 1.0
    assert 0.0 <= result.control_state.search_deficiency_score <= 1.0
    assert 0.0 <= result.control_state.long_corridor_search_completeness <= 1.0
    assert result.summary["control_state"]["hidden_challenger_budget"] == result.control_state.hidden_challenger_budget
    assert result.summary["control_state"]["collapse_detected"] is False
