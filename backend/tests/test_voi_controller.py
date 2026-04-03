from __future__ import annotations

import types
from dataclasses import replace

import pytest

import app.main as main_module
import app.replay_oracle as replay_oracle_module
import app.voi_controller as voi_module
from app.decision_critical import DCCSConfig, select_candidates
from app.evidence_certification import compute_certificate, compute_fragility_maps, sample_world_manifest
from app.voi_controller import (
    VOIActionHooks,
    VOIConfig,
    VOIControllerState,
    build_action_menu,
    enrich_controller_state_for_actioning,
    credible_evidence_uncertainty,
    credible_search_uncertainty,
    compute_search_completeness_metrics,
    refresh_controller_state_after_action,
    run_controller,
)

pytestmark = pytest.mark.thesis_modules


def _support_rich_zero_signal_bridge_state(
    *,
    action_trace: list[dict[str, object]] | None = None,
    stochastic_enabled: bool = False,
    requested_world_count: int = 80,
    actual_world_count: int = 56,
    unique_world_count: int = 44,
) -> VOIControllerState:
    world_shortfall = max(0, int(requested_world_count) - int(actual_world_count))
    world_reuse_rate = round(
        max(0, int(actual_world_count) - int(unique_world_count)) / float(max(1, int(actual_world_count))),
        6,
    )
    return VOIControllerState(
        iteration_index=1 if action_trace else 0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.01, 10.0, 10.0)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=action_trace or [],
        active_evidence_families=["scenario"],
        stochastic_enabled=stochastic_enabled,
        ambiguity_context={
            "od_ambiguity_index": 0.74,
            "od_hard_case_prior": 0.76,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.18,
            "od_nominal_margin_proxy": 0.08,
            "od_ambiguity_support_ratio": 0.86,
            "od_ambiguity_prior_strength": 0.74,
            "od_ambiguity_source_entropy": 0.81,
            "ambiguity_budget_prior": 0.74,
            "refc_stress_world_fraction": 0.20,
            "refc_requested_world_count": int(requested_world_count),
            "refc_world_count": int(actual_world_count),
            "refc_unique_world_count": int(unique_world_count),
            "refc_world_reuse_rate": world_reuse_rate,
            "refc_world_count_shortfall": int(world_shortfall),
        },
        certificate_margin=0.0,
        search_completeness_score=0.95,
        search_completeness_gap=0.0,
        prior_support_strength=0.81,
        pending_challenger_mass=0.08,
        best_pending_flip_probability=0.12,
        frontier_recall_at_budget=0.96,
        near_tie_mass=0.02,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )


def test_initial_controller_overconfidence_cap_reduces_overconfident_incomplete_search_baseline() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.1, 10.0, 10.0)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["fuel", "terrain"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.58,
            "ambiguity_budget_prior": 0.45,
            "od_hard_case_prior": 0.69,
        },
        search_completeness_score=0.349457,
        search_completeness_gap=0.490543,
        prior_support_strength=0.707403,
        pending_challenger_mass=0.650685,
        best_pending_flip_probability=0.999986,
        frontier_recall_at_budget=0.279123,
        near_tie_mass=0.0,
        top_refresh_gain=0.851429,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
    )

    capped_value, applied = main_module._initial_controller_overconfidence_cap(
        controller_state=state,
        current_certificate=1.0,
        threshold=0.8,
    )

    assert applied is True
    assert capped_value == pytest.approx(0.45, rel=0.0, abs=1e-6)


def test_initial_controller_overconfidence_cap_skips_rows_without_strong_winner_side_signal() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.1, 10.0, 10.0)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["fuel", "terrain"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.58,
            "ambiguity_budget_prior": 0.45,
            "od_hard_case_prior": 0.69,
        },
        search_completeness_score=0.465338,
        search_completeness_gap=0.374662,
        prior_support_strength=0.804779,
        pending_challenger_mass=0.374,
        best_pending_flip_probability=0.48,
        frontier_recall_at_budget=0.46,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )

    capped_value, applied = main_module._initial_controller_overconfidence_cap(
        controller_state=state,
        current_certificate=1.0,
        threshold=0.8,
    )

    assert applied is False
    assert capped_value == pytest.approx(1.0, rel=0.0, abs=1e-6)


def test_certification_cache_key_changes_when_route_payload_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    route_payload = {
        "route_id": "route-1",
        "objective_vector": (100.0, 20.0, 5.0),
        "distance_km": 12.0,
        "evidence_tensor": {"scenario": {"time": 1.0, "money": 0.4, "co2": 0.2}},
        "evidence_provenance": {
            "active_families": ["scenario"],
            "dependency_weights": {"scenario": {"time": 1.0, "money": 0.4, "co2": 0.2}},
            "families": [{"family": "scenario", "signature": "sig-a"}],
        },
    }
    base_key = main_module._certification_cache_key(
        frontier_route_ids=["route-1"],
        frontier_signatures={"route-1": "sig-route-1"},
        route_payloads=[route_payload],
        evidence_snapshot_hash="snapshot-a",
        selected_route_id="route-1",
        run_seed=11,
        world_count=64,
        threshold=0.8,
        weights=(1.0, 1.0, 1.0),
        optimization_mode="expected_value",
        risk_aversion=1.0,
        forced_refreshed_families=(),
        ambiguity_context={"od_ambiguity_index": 0.42},
    )
    changed_key = main_module._certification_cache_key(
        frontier_route_ids=["route-1"],
        frontier_signatures={"route-1": "sig-route-1"},
        route_payloads=[
            {
                **route_payload,
                "objective_vector": (110.0, 20.0, 5.0),
                "evidence_provenance": {
                    **route_payload["evidence_provenance"],
                    "families": [{"family": "scenario", "signature": "sig-b"}],
                },
            }
        ],
        evidence_snapshot_hash="snapshot-b",
        selected_route_id="route-1",
        run_seed=11,
        world_count=64,
        threshold=0.8,
        weights=(1.0, 1.0, 1.0),
        optimization_mode="expected_value",
        risk_aversion=1.0,
        forced_refreshed_families=(),
        ambiguity_context={"od_ambiguity_index": 0.42},
    )
    monkeypatch.setattr(main_module, "CERTIFICATION_CACHE_VERSION", "refc_margin_refresh_test_salt")
    salted_key = main_module._certification_cache_key(
        frontier_route_ids=["route-1"],
        frontier_signatures={"route-1": "sig-route-1"},
        route_payloads=[route_payload],
        evidence_snapshot_hash="snapshot-a",
        selected_route_id="route-1",
        run_seed=11,
        world_count=64,
        threshold=0.8,
        weights=(1.0, 1.0, 1.0),
        optimization_mode="expected_value",
        risk_aversion=1.0,
        forced_refreshed_families=(),
        ambiguity_context={"od_ambiguity_index": 0.42},
    )

    assert changed_key != base_key
    assert salted_key != base_key


def test_committed_refresh_route_state_cache_key_is_order_invariant() -> None:
    base_key = main_module._committed_refresh_route_state_cache_key(
        base_route_state_cache_key="base-route-state",
        active_families=("terrain", "fuel", "scenario"),
        forced_refreshed_families=("fuel", "terrain"),
    )
    same_key = main_module._committed_refresh_route_state_cache_key(
        base_route_state_cache_key="base-route-state",
        active_families=("scenario", "terrain", "fuel"),
        forced_refreshed_families=("terrain", "fuel"),
    )
    changed_base_key = main_module._committed_refresh_route_state_cache_key(
        base_route_state_cache_key="other-base-route-state",
        active_families=("terrain", "fuel", "scenario"),
        forced_refreshed_families=("fuel", "terrain"),
    )
    changed_refresh_key = main_module._committed_refresh_route_state_cache_key(
        base_route_state_cache_key="base-route-state",
        active_families=("terrain", "fuel", "scenario"),
        forced_refreshed_families=("fuel",),
    )

    assert same_key == base_key
    assert changed_base_key != base_key
    assert changed_refresh_key != base_key


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
        "proxy_confidence": {"time": 0.95, "money": 0.90, "co2": 0.88},
    }


def _route(route_id: str, objective: tuple[float, float, float], evidence_tensor: dict[str, dict[str, float]]) -> dict[str, object]:
    return {
        "route_id": route_id,
        "objective_vector": objective,
        "evidence_tensor": evidence_tensor,
        "evidence_provenance": {
            "active_families": list(evidence_tensor.keys()),
            "dependency_weights": {
                family: {"time": 1.0, "money": 1.0, "co2": 1.0}
                for family in evidence_tensor
            },
        },
    }


def _dccs_result():
    candidates = [
        _candidate(
            "cand_fast",
            path=["c1", "c2", "c3"],
            objective=(9.0, 9.2, 9.1),
            road_mix={"motorway_share": 0.7, "a_road_share": 0.2, "urban_share": 0.1},
            toll_share=0.02,
            terrain_burden=0.05,
            straight_line_km=8.8,
            mechanism={"motorway_share": 0.7, "toll_share": 0.02, "terrain_burden": 0.05},
        ),
        _candidate(
            "cand_slow",
            path=["s1", "s2", "s3", "s4"],
            objective=(12.5, 12.2, 12.1),
            road_mix={"motorway_share": 0.1, "a_road_share": 0.5, "urban_share": 0.4},
            toll_share=0.15,
            terrain_burden=0.25,
            straight_line_km=8.3,
            mechanism={"motorway_share": 0.1, "toll_share": 0.15, "terrain_burden": 0.25},
        ),
    ]
    return select_candidates(
        candidates,
        frontier=[_candidate(
            "frontier_anchor",
            path=["f1", "f2"],
            objective=(10.0, 10.0, 10.0),
            road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
            toll_share=0.05,
            terrain_burden=0.10,
            straight_line_km=9.0,
            mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
        )],
        config=DCCSConfig(mode="challenger", search_budget=1),
    )


def _fragility_result():
    routes = [
        _route(
            "route_a",
            (10.0, 10.0, 10.0),
            {"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        ),
        _route(
            "route_b",
            (10.1, 10.1, 10.1),
            {},
        ),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal"}},
        {"world_id": "w1", "states": {"scenario": "severely_stale"}},
        {"world_id": "w2", "states": {"scenario": "refreshed"}},
    ]
    return compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        selected_route_id="route_a",
    )


def _supported_ambiguity_fragility_result():
    routes = [
        _route(
            "route_a",
            (10.0, 10.0, 10.0),
            {
                "scenario": {"time": 1.0, "money": 0.9, "co2": 0.8},
                "weather": {"time": 0.1, "money": 0.1, "co2": 0.1},
                "toll": {"time": 0.0, "money": 0.1, "co2": 0.0},
            },
        ),
        _route(
            "route_b",
            (10.03, 10.02, 10.01),
            {
                "scenario": {"time": 0.2, "money": 0.1, "co2": 0.1},
                "weather": {"time": 1.0, "money": 0.8, "co2": 0.2},
                "toll": {"time": 0.3, "money": 0.6, "co2": 0.1},
            },
        ),
        _route(
            "route_c",
            (10.07, 10.05, 10.04),
            {
                "scenario": {"time": 0.1, "money": 0.0, "co2": 0.0},
                "weather": {"time": 0.3, "money": 0.2, "co2": 0.2},
                "toll": {"time": 1.0, "money": 1.0, "co2": 0.6},
            },
        ),
    ]
    manifest = sample_world_manifest(
        active_families=["scenario", "weather", "toll"],
        seed=41,
        world_count=12,
        routes=routes,
        ambiguity_context={
            "od_ambiguity_index": 0.42,
            "od_hard_case_prior": 0.24,
            "od_engine_disagreement_prior": 0.22,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.26,
            "od_nominal_margin_proxy": 0.28,
            "od_ambiguity_family_density": 0.46,
            "od_ambiguity_margin_pressure": 0.42,
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.36,
            "ambiguity_budget_band": "medium",
        },
    )
    certificate = compute_certificate(
        routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.6,
        active_families=["scenario", "weather", "toll"],
        ambiguity_context=manifest["ambiguity_context"],
    )
    return compute_fragility_maps(
        routes,
        worlds=manifest["worlds"],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather", "toll"],
        selected_route_id="route_a",
        baseline_certificate=certificate,
        ambiguity_context=manifest["ambiguity_context"],
    )


def _support_rich_certified_reopen_dccs():
    return select_candidates(
        [
            _candidate(
                "cand_live",
                path=["c1", "c2", "c3"],
                objective=(9.0, 9.15, 9.05),
                road_mix={"motorway_share": 0.55, "a_road_share": 0.30, "urban_share": 0.15},
                toll_share=0.04,
                terrain_burden=0.06,
                straight_line_km=8.8,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            ),
            _candidate(
                "cand_peer",
                path=["p1", "p2", "p3", "p4"],
                objective=(11.8, 11.6, 11.7),
                road_mix={"motorway_share": 0.40, "a_road_share": 0.35, "urban_share": 0.25},
                toll_share=0.12,
                terrain_burden=0.15,
                straight_line_km=8.7,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            ),
            _candidate(
                "cand_extra",
                path=["e1", "e2", "e3"],
                objective=(10.8, 10.5, 10.7),
                road_mix={"motorway_share": 0.45, "a_road_share": 0.35, "urban_share": 0.20},
                toll_share=0.08,
                terrain_burden=0.08,
                straight_line_km=8.9,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            ),
            _candidate(
                "cand_more",
                path=["m1", "m2", "m3"],
                objective=(10.4, 10.2, 10.3),
                road_mix={"motorway_share": 0.50, "a_road_share": 0.28, "urban_share": 0.22},
                toll_share=0.06,
                terrain_burden=0.07,
                straight_line_km=8.85,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            ),
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=1),
    )


def _support_rich_certified_reopen_state(
    *,
    action_trace: list[dict[str, object]] | None = None,
) -> VOIControllerState:
    return VOIControllerState(
        iteration_index=1 if action_trace else 0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.06, 9.96, 10.02)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=action_trace or [],
        active_evidence_families=["scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.74,
            "od_hard_case_prior": 0.76,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.36,
            "od_nominal_margin_proxy": 0.08,
            "od_ambiguity_support_ratio": 0.86,
            "od_ambiguity_prior_strength": 0.74,
            "od_ambiguity_source_entropy": 0.81,
        },
        certificate_margin=0.0,
        search_completeness_score=0.83,
        search_completeness_gap=0.13,
        prior_support_strength=0.81,
        pending_challenger_mass=0.34,
        best_pending_flip_probability=0.44,
        frontier_recall_at_budget=0.74,
        near_tie_mass=0.19,
        top_refresh_gain=0.0,
        top_fragility_mass=0.161017,
        competitor_pressure=1.0,
    )


def _support_rich_zero_signal_fragility_result():
    return types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "carbon", "vor": 0.0}],
            "top_refresh_family": "carbon",
            "top_refresh_gain": 0.0,
        },
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )


def _support_rich_zero_signal_bridge_state(
    *,
    action_trace: list[dict[str, object]] | None = None,
    stochastic_enabled: bool = False,
    actual_world_count: int = 92,
    unique_world_count: int | None = None,
    requested_world_count: int = 96,
) -> VOIControllerState:
    base = _support_rich_certified_reopen_state(action_trace=action_trace)
    ambiguity_context = dict(base.ambiguity_context)
    ambiguity_context["refc_stress_world_fraction"] = 0.988372
    ambiguity_context["refc_world_count"] = int(actual_world_count)
    ambiguity_context["refc_unique_world_count"] = int(
        actual_world_count if unique_world_count is None else unique_world_count
    )
    ambiguity_context["refc_requested_world_count"] = int(requested_world_count)
    ambiguity_context["refc_world_count_shortfall"] = max(
        0,
        int(requested_world_count) - int(actual_world_count),
    )
    return replace(
        base,
        stochastic_enabled=stochastic_enabled,
        ambiguity_context=ambiguity_context,
        certificate_margin=1.0,
        search_completeness_score=0.539335,
        search_completeness_gap=0.300665,
        prior_support_strength=0.707403,
        pending_challenger_mass=0.596447,
        best_pending_flip_probability=0.998771,
        frontier_recall_at_budget=0.455377,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )


def test_action_menu_respects_budget_gates() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()

    search_blocked = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.2},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        near_tie_mass=0.2,
    )
    evidence_blocked = replace(search_blocked, remaining_search_budget=1, remaining_evidence_budget=0)

    search_actions = build_action_menu(search_blocked, dccs=dccs, fragility=fragility, config=VOIConfig())
    evidence_actions = build_action_menu(evidence_blocked, dccs=dccs, fragility=fragility, config=VOIConfig())

    assert all(not action.kind.startswith("refine") for action in search_actions if action.kind != "stop")
    assert all(
        action.kind not in {"refresh_top1_vor", "increase_stochastic_samples"}
        for action in evidence_actions
        if action.kind != "stop"
    )


def test_enriched_controller_state_matches_action_menu_assumptions() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    raw_state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.25},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        near_tie_mass=0.2,
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_prior_strength": 0.66,
            "support_strength": 0.58,
        },
    )

    enriched = enrich_controller_state_for_actioning(
        raw_state,
        dccs=dccs,
        fragility=fragility,
        config=VOIConfig(),
    )

    assert enriched.support_richness > raw_state.support_richness
    assert enriched.ambiguity_pressure >= raw_state.ambiguity_pressure
    assert enriched.top_refresh_gain > 0.0
    assert enriched.top_fragility_mass >= 0.0
    assert enriched.competitor_pressure >= 0.0


def test_best_vor_family_prefers_controller_fallback_choice_when_present() -> None:
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "scenario", "vor": 0.0}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.0,
            "controller_ranking": [
                {
                    "family": "weather",
                    "controller_score": 0.006,
                    "empirical_vor": 0.0,
                    "raw_refresh_gain": 0.006,
                    "basis": "raw_refresh_gain_fallback",
                }
            ],
            "top_refresh_family_controller": "weather",
            "top_refresh_gain_controller": 0.006,
        }
    )

    assert voi_module._best_vor_family(fragility) == "weather"


def test_refresh_action_uses_structured_empirical_uplift_when_available() -> None:
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "baseline_certificate": 0.8051282051282052,
            "empirical_baseline_certificate": 0.8051282051282052,
            "controller_baseline_certificate": 0.8051282051282052,
            "per_family_certificate": {
                "scenario": 1.0,
                "fuel": 0.8051282051282052,
            },
            "ranking": [
                {"family": "scenario", "vor": 0.19487179487179485},
                {"family": "fuel", "vor": 0.01},
            ],
        },
        route_fragility_map={"route_a": {"scenario": 0.19487179487179485, "fuel": 0.01}},
        competitor_fragility_breakdown={"route_a": {}},
    )

    uplift_action = voi_module._build_refresh_action(
        "scenario",
        fragility=fragility,
        winner_id="route_a",
        current_certificate=0.8051282051282052,
        config=VOIConfig(certificate_threshold=0.84),
    )
    revelation_only_action = voi_module._build_refresh_action(
        "fuel",
        fragility=fragility,
        winner_id="route_a",
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.84),
    )

    assert uplift_action.metadata["structured_refresh_signal"] is True
    assert uplift_action.metadata["empirical_refresh_certificate_uplift"] == pytest.approx(0.194872, abs=1e-6)
    assert uplift_action.predicted_delta_certificate >= 0.19
    assert revelation_only_action.metadata["structured_refresh_signal"] is True
    assert revelation_only_action.metadata["empirical_refresh_certificate_uplift"] == pytest.approx(0.0, abs=1e-9)
    assert revelation_only_action.metadata["empirical_refresh_certificate_delta"] == pytest.approx(0.0, abs=1e-9)
    assert revelation_only_action.metadata["empirical_refresh_certificate_drop"] == pytest.approx(0.0, abs=1e-9)
    assert revelation_only_action.predicted_delta_certificate < 0.02


def test_uncertified_structural_cap_churn_suppresses_non_novel_reopen() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.747845},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.747845,
        },
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.32,
        predicted_delta_certificate=0.41,
        predicted_delta_margin=0.19,
        predicted_delta_frontier=0.03,
        metadata={
            "mean_flip_probability": 0.99,
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.76,
            "normalized_overlap_reduction": 0.90,
        },
        reason="refine_candidate",
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:carbon",
        kind="refresh_top1_vor",
        target="carbon",
        q_score=0.0,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
        reason="refresh_evidence_family",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_structural_cap_churn(
        [refine, refresh, stop],
        state=state,
        current_certificate=0.747845,
        config=VOIConfig(certificate_threshold=0.82),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_uncertified_structural_cap_churn_keeps_objective_supported_reopen() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.77},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.77,
        },
        top_refresh_gain=0.0,
        top_fragility_mass=0.0002,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.37,
        predicted_delta_certificate=0.51,
        predicted_delta_margin=0.22,
        predicted_delta_frontier=0.094,
        metadata={
            "mean_flip_probability": 0.9999,
            "normalized_objective_gap": 0.056,
            "normalized_mechanism_gap": 0.82,
            "normalized_overlap_reduction": 0.90,
        },
        reason="refine_candidate",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_structural_cap_churn(
        [refine, stop],
        state=state,
        current_certificate=0.77,
        config=VOIConfig(certificate_threshold=0.85),
    )

    assert any(action.kind == "refine_top1_dccs" for action in filtered)


def test_uncertified_structural_cap_churn_stops_observed_zero_signal_mechanism_only_reopen() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.74878},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.74878,
            "refc_world_count": 45,
            "refc_unique_world_count": 45,
            "refc_requested_world_count": 72,
        },
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.225131,
        predicted_delta_certificate=0.25122,
        predicted_delta_margin=0.231084,
        predicted_delta_frontier=0.11085,
        metadata={
            "mean_flip_probability": 0.99997,
            "normalized_objective_gap": 0.069597,
            "normalized_mechanism_gap": 0.825578,
            "normalized_overlap_reduction": 0.904762,
        },
        reason="refine_candidate",
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:carbon",
        kind="refresh_top1_vor",
        target="carbon",
        q_score=0.0,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
        reason="refresh_evidence_family",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_structural_cap_churn(
        [refine, refresh, stop],
        state=state,
        current_certificate=0.74878,
        config=VOIConfig(certificate_threshold=0.82),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_uncertified_structural_cap_churn_stops_structural_cap_only_modest_single_world_reopen() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.756973},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=2,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.756973,
            "refc_world_count": 1,
            "refc_unique_world_count": 1,
            "refc_requested_world_count": 88,
            "refc_sampler_requested_world_count": 1,
            "refc_world_count_policy": "single_frontier_full_stress",
        },
        top_refresh_gain=0.0,
        top_fragility_mass=0.000344,
        competitor_pressure=0.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.217034,
        predicted_delta_certificate=0.243027,
        predicted_delta_margin=0.21658,
        predicted_delta_frontier=0.113819,
        metadata={
            "mean_flip_probability": 0.997643,
            "normalized_objective_gap": 0.107226,
            "normalized_mechanism_gap": 0.190531,
            "normalized_overlap_reduction": 0.857143,
        },
        reason="refine_candidate",
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.0,
        metadata={
            "structured_refresh_signal": True,
            "structural_cap_only": True,
            "empirical_refresh_certificate_uplift": 0.0,
            "empirical_refresh_certificate_delta": 0.0,
        },
        reason="refresh_evidence_family",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_structural_cap_churn(
        [refine, refresh, stop],
        state=state,
        current_certificate=0.756973,
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_settled_certified_revelation_only_refresh_actions_prefer_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.3, 10.2, 10.1)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        certificate_margin=0.77,
        near_tie_mass=0.0,
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.9,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
        reason="refresh_evidence_family",
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.33,
        predicted_delta_certificate=0.43,
        predicted_delta_margin=0.22,
        predicted_delta_frontier=0.10,
        metadata={
            "mean_flip_probability": 0.99,
            "normalized_objective_gap": 0.10,
            "normalized_mechanism_gap": 0.35,
            "normalized_overlap_reduction": 0.90,
        },
        reason="refine_candidate",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_settled_certified_revelation_only_actions(
        [refresh, refine, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_settled_certified_revelation_only_actions_keep_supported_hard_case_reopen() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.3, 10.2, 10.1)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        ambiguity_context={
            "od_ambiguity_index": 0.74,
            "od_hard_case_prior": 0.76,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.36,
            "od_nominal_margin_proxy": 0.08,
            "od_ambiguity_support_ratio": 0.86,
            "od_ambiguity_prior_strength": 0.74,
            "od_ambiguity_source_entropy": 0.81,
            "ambiguity_budget_prior": 0.74,
            "refc_stress_world_fraction": 0.54,
        },
        certificate_margin=1.0,
        near_tie_mass=0.0,
        search_completeness_score=0.473146,
        search_completeness_gap=0.366854,
        prior_support_strength=0.81,
        support_richness=0.7416,
        ambiguity_pressure=0.8124,
        pending_challenger_mass=0.650685,
        best_pending_flip_probability=0.999986,
        frontier_recall_at_budget=0.279421,
        top_refresh_gain=0.942857,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.06,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
        reason="refresh_evidence_family",
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.05,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.06,
        predicted_delta_frontier=0.02,
        metadata={
            "mean_flip_probability": 0.24,
            "normalized_objective_gap": 0.03,
            "normalized_mechanism_gap": 0.04,
            "normalized_overlap_reduction": 0.10,
        },
        reason="refine_candidate",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_settled_certified_revelation_only_actions(
        [refresh, refine, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert [action.kind for action in filtered] == ["refresh_top1_vor", "refine_top1_dccs", "stop"]


def test_settled_certified_revelation_only_actions_keep_credible_controller_hard_case_reopen() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_1", "objective_vector": (10573.77, 163.64, 231.839)},
            {"route_id": "route_0", "objective_vector": (10623.64, 153.82, 214.733)},
        ],
        certificate={"route_1": 0.0, "route_0": 1.0},
        winner_id="route_0",
        selected_route_id="route_0",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        ambiguity_context={
            "od_ambiguity_index": 0.036667,
            "od_hard_case_prior": 0.331127,
            "od_ambiguity_confidence": 0.864,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":8,"routing_graph_probe":2}',
            "od_candidate_path_count": 11,
            "od_corridor_family_count": 2,
            "od_objective_spread": 0.0,
            "od_nominal_margin_proxy": 1.0,
            "od_ambiguity_support_ratio": 0.474005,
            "od_ambiguity_prior_strength": 0.036667,
            "od_ambiguity_source_entropy": 0.721928,
            "ambiguity_budget_prior": 0.036667,
            "refc_stress_world_fraction": 0.6,
        },
        certificate_margin=1.0,
        near_tie_mass=0.0,
        search_completeness_score=0.610626,
        search_completeness_gap=0.229374,
        prior_support_strength=0.279284,
        support_richness=0.532858,
        ambiguity_pressure=0.588967,
        pending_challenger_mass=0.739018,
        best_pending_flip_probability=0.999978,
        frontier_recall_at_budget=0.254755,
        top_refresh_gain=0.15,
        top_fragility_mass=0.15,
        competitor_pressure=1.0,
        credible_search_uncertainty=True,
        credible_evidence_uncertainty=True,
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:terrain",
        kind="refresh_top1_vor",
        target="terrain",
        q_score=0.09,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
        reason="refresh_evidence_family",
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.37,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.06,
        predicted_delta_frontier=0.02,
        metadata={
            "mean_flip_probability": 0.999978,
            "normalized_objective_gap": 0.03,
            "normalized_mechanism_gap": 0.04,
            "normalized_overlap_reduction": 0.10,
        },
        reason="refine_candidate",
    )
    resample = voi_module.VOIAction(
        action_id="resample:test",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.02,
        reason="increase_stochastic_samples",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_settled_certified_revelation_only_actions(
        [refresh, refine, resample, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert [action.kind for action in filtered] == [
        "refresh_top1_vor",
        "refine_top1_dccs",
        "increase_stochastic_samples",
        "stop",
    ]


def test_certified_negative_empirical_refresh_revelation_requires_remaining_certificate_headroom() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.3, 10.2, 10.1)},
        ],
        certificate={"route_a": 0.93, "route_b": 0.07},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        ambiguity_context={
            "od_ambiguity_index": 0.420932,
            "od_hard_case_prior": 0.692693,
            "od_ambiguity_confidence": 0.88087,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":4,"routing_graph_probe":6}',
            "od_candidate_path_count": 3,
            "od_corridor_family_count": 2,
            "od_objective_spread": 0.332366,
            "od_nominal_margin_proxy": 0.099161,
            "od_ambiguity_support_ratio": 0.587187,
            "od_ambiguity_prior_strength": 0.420932,
            "od_ambiguity_source_entropy": 0.845351,
            "ambiguity_budget_prior": 0.535885,
            "refc_stress_world_fraction": 0.542857,
        },
        certificate_margin=1.0,
        near_tie_mass=0.0,
        search_completeness_score=0.473146,
        search_completeness_gap=0.366854,
        prior_support_strength=0.707403,
        support_richness=0.707403,
        ambiguity_pressure=0.605242,
        pending_challenger_mass=0.650685,
        best_pending_flip_probability=0.999986,
        frontier_recall_at_budget=0.279421,
        top_refresh_gain=0.942857,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
    )

    assert voi_module._allow_certified_negative_refresh_revelation(
        state=state,
        current_certificate=0.93,
        config=VOIConfig(certificate_threshold=0.81),
        signed_refresh_delta=-0.057143,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    ) is True
    assert voi_module._allow_certified_negative_refresh_revelation(
        state=replace(state, certificate={"route_a": 1.0, "route_b": 0.0}, certificate_margin=1.0),
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
        signed_refresh_delta=-0.057143,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    ) is False


def test_saturated_certified_negative_refresh_revelation_reopens_supported_high_pressure_row() -> None:
    dccs = replace(
        _dccs_result(),
        selected=[],
        skipped=[],
        candidate_ledger=[],
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "fuel", "vor": 0.851429}],
            "top_refresh_family": "fuel",
            "top_refresh_gain": 0.851429,
        },
        route_fragility_map={"route_a": {"fuel": 1.0}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"fuel": 175.0}}},
    )
    cfg = VOIConfig(certificate_threshold=0.80, search_budget=5, evidence_budget=3)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (4081.59, 75.69, 102.582)},
            {"route_id": "route_b", "objective_vector": (3712.44, 75.82, 104.387)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[],
        active_evidence_families=["carbon", "fuel", "scenario", "stochastic", "terrain", "toll"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.420932,
            "od_hard_case_prior": 0.692693,
            "od_engine_disagreement_prior": 0.701919,
            "od_ambiguity_confidence": 0.914,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":8,"routing_graph_probe":3}',
            "od_candidate_path_count": 10,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.5,
            "od_nominal_margin_proxy": 0.010454,
            "od_ambiguity_support_ratio": 0.587187,
            "od_ambiguity_prior_strength": 0.420932,
            "od_ambiguity_source_entropy": 0.845351,
            "ambiguity_budget_prior": 0.453718,
            "refc_stress_world_fraction": 0.542857,
        },
        certificate_margin=1.0,
        near_tie_mass=0.0,
        search_completeness_score=0.473146,
        search_completeness_gap=0.366854,
        prior_support_strength=0.707403,
        support_richness=0.707403,
        ambiguity_pressure=0.605242,
        pending_challenger_mass=0.650685,
        best_pending_flip_probability=0.999986,
        corridor_family_recall=0.666667,
        frontier_recall_at_budget=0.279421,
        top_refresh_gain=0.851429,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
        credible_search_uncertainty=True,
        credible_evidence_uncertainty=True,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=cfg)  # type: ignore[arg-type]

    assert actions[0].kind == "refresh_top1_vor"
    assert actions[0].q_score > cfg.stop_threshold


def test_saturated_certified_negative_refresh_revelation_stops_wide_margin_row() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_0", "objective_vector": (2557.21, 54.4, 75.36)},
            {"route_id": "route_2", "objective_vector": (2921.6, 54.39, 73.771)},
        ],
        certificate={"route_0": 0.0, "route_2": 1.0},
        winner_id="route_2",
        selected_route_id="route_2",
        remaining_search_budget=1,
        remaining_evidence_budget=3,
        ambiguity_context={
            "od_ambiguity_index": 0.41312,
            "od_hard_case_prior": 0.56246,
            "od_engine_disagreement_prior": 0.557692,
            "od_ambiguity_confidence": 0.910497,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":4,"repo_local_geometry_backfill":4,"routing_graph_probe":2}',
            "od_candidate_path_count": 3,
            "od_corridor_family_count": 2,
            "od_objective_spread": 0.16,
            "od_nominal_margin_proxy": 0.3,
            "od_ambiguity_margin_pressure": 0.7,
            "od_ambiguity_spread_pressure": 0.16,
            "od_ambiguity_support_ratio": 0.876689,
            "od_ambiguity_prior_strength": 0.41312,
            "od_ambiguity_source_entropy": 0.96023,
            "ambiguity_budget_prior": 0.467618,
            "refc_stress_world_fraction": 0.587629,
        },
        certificate_margin=1.0,
        near_tie_mass=0.0,
        search_completeness_score=0.431947,
        search_completeness_gap=0.408053,
        prior_support_strength=0.525415,
        support_richness=0.728969,
        ambiguity_pressure=0.683262,
        pending_challenger_mass=0.760415,
        best_pending_flip_probability=0.999993,
        corridor_family_recall=0.5,
        frontier_recall_at_budget=0.107117,
        top_refresh_gain=0.917526,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
        credible_search_uncertainty=True,
        credible_evidence_uncertainty=True,
    )

    assert voi_module._allow_certified_negative_refresh_revelation(
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.80),
        signed_refresh_delta=-0.082474,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    ) is False


def test_saturated_certified_negative_refresh_revelation_requires_tie_like_decision_ambiguity() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (4081.59, 75.69, 102.582)},
            {"route_id": "route_b", "objective_vector": (4100.11, 75.82, 104.387)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        ambiguity_context={
            "od_ambiguity_index": 0.38083,
            "od_hard_case_prior": 0.38083,
            "od_ambiguity_confidence": 1.0,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":8,"routing_graph_probe":3}',
            "od_candidate_path_count": 10,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.0,
            "od_nominal_margin_proxy": 0.944363,
            "od_ambiguity_margin_pressure": 0.055637,
            "od_ambiguity_spread_pressure": 0.0,
            "od_ambiguity_support_ratio": 0.539459,
            "od_ambiguity_prior_strength": 0.38083,
            "od_ambiguity_source_entropy": 1.0,
            "ambiguity_budget_prior": 0.38083,
            "refc_stress_world_fraction": 0.584431,
        },
        certificate_margin=0.19,
        near_tie_mass=0.0,
        search_completeness_score=0.465874,
        search_completeness_gap=0.374126,
        prior_support_strength=0.652031,
        support_richness=0.652031,
        ambiguity_pressure=0.695783,
        pending_challenger_mass=0.669036,
        best_pending_flip_probability=0.99922,
        frontier_recall_at_budget=0.135125,
        corridor_family_recall=0.125,
        top_refresh_gain=0.904192,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
        credible_search_uncertainty=True,
        credible_evidence_uncertainty=True,
    )

    assert voi_module._allow_certified_negative_refresh_revelation(
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.80),
        signed_refresh_delta=-0.095808,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    ) is False


def test_saturated_certified_low_decision_ambiguity_reopen_prefers_stop_over_headroom_capped_refine() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (4081.59, 75.69, 102.582)},
            {"route_id": "route_b", "objective_vector": (4100.11, 75.82, 104.387)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        ambiguity_context={
            "od_ambiguity_index": 0.38083,
            "od_hard_case_prior": 0.38083,
            "od_ambiguity_support_ratio": 0.539459,
            "od_ambiguity_prior_strength": 0.38083,
            "od_ambiguity_source_entropy": 1.0,
            "od_objective_spread": 0.0,
            "od_nominal_margin_proxy": 0.944363,
            "od_ambiguity_margin_pressure": 0.055637,
            "od_ambiguity_spread_pressure": 0.0,
        },
        certificate_margin=0.19,
        near_tie_mass=0.0,
        search_completeness_score=0.465621,
        search_completeness_gap=0.374379,
        prior_support_strength=0.652031,
        support_richness=0.652031,
        ambiguity_pressure=0.696134,
        pending_challenger_mass=0.67047,
        best_pending_flip_probability=0.999244,
        frontier_recall_at_budget=0.135397,
        corridor_family_recall=0.125,
        top_refresh_gain=0.904192,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
        credible_search_uncertainty=True,
        credible_evidence_uncertainty=True,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.072767,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.225517,
        predicted_delta_frontier=0.109248,
        metadata={
            "mean_flip_probability": 0.999244,
            "normalized_objective_gap": 0.109248,
            "normalized_mechanism_gap": 0.358178,
            "normalized_overlap_reduction": 0.909091,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.0,
        },
    )

    assert voi_module._should_stop_saturated_certified_low_decision_ambiguity_reopen(
        refine,
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.80),
    ) is True


def test_saturated_certified_search_without_certificate_upside_preserves_true_objective_reopen() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.4, 10.3, 10.2)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        certificate_margin=1.0,
        near_tie_mass=0.0,
        top_refresh_gain=0.94,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.12,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.18,
        predicted_delta_frontier=0.14,
        metadata={
            "normalized_objective_gap": 0.14,
            "normalized_mechanism_gap": 0.18,
            "normalized_overlap_reduction": 0.90,
        },
        reason="refine_candidate",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_saturated_certified_search_without_certificate_upside(
        [refine, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert any(action.kind == "refine_top1_dccs" for action in filtered)


def test_cached_direct_fallback_certified_fast_path_suppresses_search_but_keeps_evidence() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.1)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "direct_k_raw_fallback",
            "graph_k_raw_cache_hit": True,
            "graph_low_ambiguity_fast_path": True,
            "graph_supported_ambiguity_fast_fallback": False,
            "supplemental_challenger_activated": False,
        },
        certificate_margin=1.0,
        near_tie_mass=0.0,
        top_refresh_gain=0.15,
        top_fragility_mass=0.15,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.09,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.05,
        predicted_delta_frontier=0.04,
        metadata={
            "normalized_objective_gap": 0.04,
            "normalized_mechanism_gap": 0.11,
            "normalized_overlap_reduction": 0.20,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh_top1_vor:test",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.11,
        predicted_delta_certificate=0.02,
        predicted_delta_margin=0.0,
        predicted_delta_frontier=0.0,
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_cached_direct_fallback_search_churn(
        [refine, refresh, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert [action.kind for action in filtered] == ["refresh_top1_vor", "stop"]


def test_cached_direct_fallback_near_threshold_zero_signal_row_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.79},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "direct_k_raw_fallback",
            "graph_k_raw_cache_hit": True,
            "graph_low_ambiguity_fast_path": False,
            "graph_supported_ambiguity_fast_fallback": False,
            "supplemental_challenger_activated": False,
        },
        certificate_margin=0.79,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.08,
        predicted_delta_certificate=0.01,
        predicted_delta_margin=0.04,
        predicted_delta_frontier=0.03,
        metadata={
            "normalized_objective_gap": 0.02,
            "normalized_mechanism_gap": 0.24,
            "normalized_overlap_reduction": 0.20,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_cached_direct_fallback_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.79,
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_cached_direct_fallback_single_frontier_shortcut_zero_signal_row_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.757826},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "graph_k_raw_cache_hit": True,
            "graph_low_ambiguity_fast_path": False,
            "graph_supported_ambiguity_fast_fallback": True,
            "supplemental_challenger_activated": False,
            "refc_world_count_policy": "single_frontier_shortcut",
            "od_candidate_path_count": 1,
            "od_corridor_family_count": 1,
        },
        certificate_margin=0.757826,
        near_tie_mass=0.0,
        corridor_family_recall=0.75,
        frontier_recall_at_budget=0.716088,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.236861,
        predicted_delta_certificate=0.242174,
        predicted_delta_margin=0.267101,
        predicted_delta_frontier=0.165211,
        metadata={
            "normalized_objective_gap": 0.148212,
            "normalized_mechanism_gap": 0.799122,
            "normalized_overlap_reduction": 1.0,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_cached_direct_fallback_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.757826,
        config=VOIConfig(certificate_threshold=0.82),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_cached_direct_fallback_single_frontier_shortcut_without_explicit_cache_hit_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.757826},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "graph_low_ambiguity_fast_path": False,
            "supplemental_challenger_activated": False,
            "refc_world_count_policy": "single_frontier_shortcut",
            "od_candidate_path_count": 1,
            "od_corridor_family_count": 1,
        },
        certificate_margin=0.757826,
        near_tie_mass=0.0,
        corridor_family_recall=0.75,
        frontier_recall_at_budget=0.716088,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.236861,
        predicted_delta_certificate=0.242174,
        predicted_delta_margin=0.267101,
        predicted_delta_frontier=0.165211,
        metadata={
            "normalized_objective_gap": 0.148212,
            "normalized_mechanism_gap": 0.799122,
            "normalized_overlap_reduction": 1.0,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_cached_direct_fallback_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.757826,
        config=VOIConfig(certificate_threshold=0.82),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_cached_direct_fallback_supported_zero_signal_row_prefers_stop_before_threshold() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.757826},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "graph_low_ambiguity_fast_path": False,
            "graph_supported_ambiguity_fast_fallback": False,
            "supplemental_challenger_activated": False,
        },
        certificate_margin=0.757826,
        near_tie_mass=0.0,
        corridor_family_recall=0.5,
        frontier_recall_at_budget=0.535943,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.216742,
        predicted_delta_certificate=0.18252,
        predicted_delta_margin=0.216742,
        predicted_delta_frontier=0.134959,
        metadata={
            "normalized_objective_gap": 0.054183,
            "normalized_mechanism_gap": 0.243117,
            "normalized_overlap_reduction": 0.904762,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_cached_direct_fallback_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.757826,
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_cached_direct_fallback_support_rich_zero_signal_near_threshold_probe_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.756973},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=0,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "direct_k_raw_fallback",
            "graph_low_ambiguity_fast_path": False,
            "graph_supported_ambiguity_fast_fallback": False,
            "supplemental_challenger_activated": False,
            "od_hard_case_prior": 0.551803,
            "ambiguity_budget_prior": 0.42,
            "od_ambiguity_support_ratio": 0.609138,
            "od_ambiguity_source_entropy": 0.74,
            "od_ambiguity_index": 0.40,
        },
        certificate_margin=0.02,
        near_tie_mass=0.28,
        corridor_family_recall=0.428571,
        frontier_recall_at_budget=0.451056,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
        support_richness=0.609138,
        prior_support_strength=0.609138,
        ambiguity_pressure=0.551803,
    )
    refine = voi_module.VOIAction(
        action_id="refine_topk_dccs:test",
        kind="refine_topk_dccs",
        target="cohort",
        q_score=0.109179,
        predicted_delta_certificate=0.243027,
        predicted_delta_margin=0.213602,
        predicted_delta_frontier=0.127603,
        metadata={
            "normalized_objective_gap": 0.053423,
            "normalized_mechanism_gap": 0.355737,
            "normalized_overlap_reduction": 0.909091,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_cached_direct_fallback_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.756973,
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_cached_direct_fallback_very_near_threshold_zero_signal_row_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.800029},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=1,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "graph_low_ambiguity_fast_path": False,
            "graph_supported_ambiguity_fast_fallback": False,
            "supplemental_challenger_activated": False,
        },
        certificate_margin=0.800029,
        near_tie_mass=0.0,
        corridor_family_recall=0.363636,
        frontier_recall_at_budget=0.364464,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.198715,
        predicted_delta_certificate=0.199971,
        predicted_delta_margin=0.236388,
        predicted_delta_frontier=0.130905,
        metadata={
            "normalized_objective_gap": 0.105905,
            "normalized_mechanism_gap": 0.828916,
            "normalized_overlap_reduction": 0.909091,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_cached_direct_fallback_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.800029,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_cached_direct_fallback_churn_suppression_preserves_far_below_threshold_supported_refine() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.1, 10.2, 10.1)},
        ],
        certificate={"route_a": 0.45, "route_b": 0.32},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "direct_k_raw_fallback",
            "graph_k_raw_cache_hit": True,
            "graph_low_ambiguity_fast_path": False,
            "graph_supported_ambiguity_fast_fallback": False,
            "supplemental_challenger_activated": False,
        },
        certificate_margin=0.13,
        near_tie_mass=0.0,
        top_refresh_gain=0.65,
        top_fragility_mass=0.80,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.18,
        predicted_delta_certificate=0.20,
        predicted_delta_margin=0.21,
        predicted_delta_frontier=0.13,
        metadata={
            "mean_flip_probability": 0.92,
            "normalized_objective_gap": 0.16,
            "normalized_mechanism_gap": 0.22,
            "normalized_overlap_reduction": 0.60,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_cached_direct_fallback_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.45,
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert [action.kind for action in filtered] == ["refine_top1_dccs", "stop"]


def test_cached_direct_fallback_refresh_bridge_can_override_mechanism_only_refine() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.643678, "route_b": 0.356322},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        ambiguity_context={
            "od_ambiguity_index": 0.26,
            "od_hard_case_prior": 0.36199,
            "od_ambiguity_support_ratio": 0.64432,
            "od_ambiguity_source_entropy": 0.78166,
            "ambiguity_budget_prior": 0.26,
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "graph_k_raw_cache_hit": True,
            "graph_supported_ambiguity_fast_fallback": False,
            "supplemental_challenger_activated": False,
        },
        support_richness=0.60675,
        ambiguity_pressure=0.627288,
        top_refresh_gain=0.574713,
        top_fragility_mass=0.62069,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.271442,
        predicted_delta_certificate=0.356322,
        predicted_delta_margin=0.202054,
        predicted_delta_frontier=0.04757,
        metadata={
            "normalized_objective_gap": 0.015353,
            "normalized_mechanism_gap": 0.350522,
            "normalized_overlap_reduction": 0.904762,
        },
        reason="refine_candidate",
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.037172,
        predicted_delta_certificate=0.048886,
        predicted_delta_margin=0.031363,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
        reason="refresh_evidence_family",
    )

    adjusted_actions = voi_module._apply_strong_winner_side_refresh_preference(
        [refine, refresh],
        state=state,
        current_certificate=0.643678,
        config=VOIConfig(certificate_threshold=0.83),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )

    adjusted_refresh = next(action for action in adjusted_actions if action.kind == "refresh_top1_vor")
    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    assert adjusted_refresh.q_score > adjusted_refine.q_score
    assert adjusted_refresh.metadata["winner_side_refresh_preference_applied"] is True
    assert adjusted_refresh.metadata["winner_side_refresh_preference_cached_direct_fallback_bridge"] is True
    assert adjusted_refine.metadata["winner_side_refresh_refine_discount_applied"] is True


def test_cached_direct_fallback_refresh_bridge_uses_support_fallback_label_without_cache_hit() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.643678, "route_b": 0.356322},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        ambiguity_context={
            "od_ambiguity_index": 0.26,
            "od_hard_case_prior": 0.36199,
            "od_ambiguity_support_ratio": 0.64432,
            "od_ambiguity_source_entropy": 0.78166,
            "ambiguity_budget_prior": 0.26,
            "selected_candidate_source_label": "support_fallback:alternatives:direct_k_raw_fallback",
            "selected_candidate_source_stage": "",
            "selected_final_route_source_label": "internal:osrm_refined",
            "selected_final_route_source_stage": "osrm_refined",
            "graph_supported_ambiguity_fast_fallback": False,
            "supplemental_challenger_activated": False,
        },
        support_richness=0.60675,
        ambiguity_pressure=0.627288,
        top_refresh_gain=0.574713,
        top_fragility_mass=0.62069,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.271442,
        predicted_delta_certificate=0.356322,
        predicted_delta_margin=0.202054,
        predicted_delta_frontier=0.04757,
        metadata={
            "normalized_objective_gap": 0.015353,
            "normalized_mechanism_gap": 0.350522,
            "normalized_overlap_reduction": 0.904762,
        },
        reason="refine_candidate",
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.037172,
        predicted_delta_certificate=0.048886,
        predicted_delta_margin=0.031363,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
        reason="refresh_evidence_family",
    )

    adjusted_actions = voi_module._apply_strong_winner_side_refresh_preference(
        [refine, refresh],
        state=state,
        current_certificate=0.643678,
        config=VOIConfig(certificate_threshold=0.83),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )

    adjusted_refresh = next(action for action in adjusted_actions if action.kind == "refresh_top1_vor")
    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    assert adjusted_refresh.q_score > adjusted_refine.q_score
    assert adjusted_refresh.metadata["winner_side_refresh_preference_applied"] is True
    assert adjusted_refresh.metadata["winner_side_refresh_preference_cached_direct_fallback_bridge"] is True
    assert adjusted_refine.metadata["winner_side_refresh_refine_discount_applied"] is True


def test_support_rich_direct_fallback_post_evidence_certified_backslide_prefers_refresh() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.83908, "route_b": 0.16092},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": 0.195402,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.367815,
                "realized_productive": True,
            }
        ],
        ambiguity_context={
            "selected_candidate_source_label": "support_fallback:alternatives:direct_k_raw_fallback",
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "od_hard_case_prior": 0.36199,
            "ambiguity_budget_prior": 0.26,
            "od_ambiguity_support_ratio": 0.64432,
            "od_ambiguity_source_entropy": 0.78166,
        },
        support_richness=0.60675,
        ambiguity_pressure=0.627288,
        top_refresh_gain=0.735632,
        top_fragility_mass=0.712644,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.143959,
        predicted_delta_certificate=0.16092,
        predicted_delta_margin=0.184423,
        predicted_delta_frontier=0.008676,
        metadata={
            "normalized_objective_gap": 0.008676,
            "normalized_mechanism_gap": 0.350522,
            "normalized_overlap_reduction": 0.904762,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:terrain",
        kind="refresh_top1_vor",
        target="terrain",
        q_score=0.076288,
        predicted_delta_certificate=0.035538,
        predicted_delta_margin=0.025228,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.022989,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_post_evidence_certified_search_backslide(
        [refine, refresh, stop],
        state=state,
        current_certificate=0.83908,
        config=VOIConfig(certificate_threshold=0.83),
    )

    assert [action.kind for action in filtered] == ["refresh_top1_vor", "stop"]


def test_certified_negative_empirical_refresh_revelation_stops_when_no_search_remains() -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 1.0,
            "per_family_certificate": {"fuel": 0.994012},
            "ranking": [{"family": "fuel", "vor": 0.994012}],
            "top_refresh_family": "fuel",
            "top_refresh_gain": 0.994012,
        },
        route_fragility_map={"route_a": {"fuel": 1.0}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"fuel": 1.0}}},
    )
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.3, 10.2, 10.1)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.770473,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        ambiguity_context={
            "od_ambiguity_index": 0.38083,
            "od_hard_case_prior": 0.484062,
            "od_ambiguity_confidence": 0.913124,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 2,
            "od_objective_spread": 0.170551,
            "od_nominal_margin_proxy": 0.147797,
            "od_ambiguity_support_ratio": 0.539459,
            "od_ambiguity_prior_strength": 0.38083,
            "od_ambiguity_source_entropy": 1.0,
            "ambiguity_budget_prior": 0.484062,
            "refc_stress_world_fraction": 0.448719,
        },
        certificate_margin=1.0,
        near_tie_mass=0.0,
        search_completeness_score=0.465621,
        search_completeness_gap=0.374379,
        prior_support_strength=0.652031,
        support_richness=0.652031,
        ambiguity_pressure=0.696134,
        pending_challenger_mass=0.67047,
        best_pending_flip_probability=0.999244,
        frontier_recall_at_budget=0.135397,
        top_refresh_gain=0.994012,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.81))

    assert [action.kind for action in actions] == ["stop"]


def test_cap_action_certificate_headroom_removes_impossible_certified_gain() -> None:
    action = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.31,
        cost_search=1,
        predicted_delta_certificate=0.42,
        predicted_delta_margin=0.18,
        predicted_delta_frontier=0.04,
        metadata={},
        reason="refine_candidate",
    )

    capped = voi_module._cap_action_certificate_headroom(
        action,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert capped.predicted_delta_certificate == pytest.approx(0.0)
    assert capped.q_score < action.q_score
    assert capped.metadata["certificate_headroom_cap_applied"] is True


def test_post_evidence_certified_search_backslide_prefers_stop_without_objective_support() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": 0.194872,
                "realized_evidence_uncertainty_delta": -0.184872,
                "realized_productive": True,
            }
        ],
        ambiguity_context={
            "od_ambiguity_index": 0.40,
            "od_hard_case_prior": 0.49,
            "od_ambiguity_support_ratio": 0.68,
            "od_ambiguity_source_entropy": 0.84,
            "ambiguity_budget_prior": 0.42,
        },
        support_richness=0.65,
        ambiguity_pressure=0.62,
        certificate_margin=0.77,
        near_tie_mass=0.0,
        top_refresh_gain=0.194872,
        top_fragility_mass=0.194872,
        competitor_pressure=1.0,
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:terrain",
        kind="refresh_top1_vor",
        target="terrain",
        q_score=0.05,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.004,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
        reason="refresh_evidence_family",
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.28,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.18,
        predicted_delta_frontier=0.0,
        metadata={
            "mean_flip_probability": 0.99,
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.35,
            "normalized_overlap_reduction": 0.91,
        },
        reason="refine_candidate",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_post_evidence_certified_search_backslide(
        [refine, refresh, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_post_evidence_certified_search_backslide_preserves_supported_reopen() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": 0.194872,
                "realized_evidence_uncertainty_delta": -0.184872,
                "realized_productive": True,
            }
        ],
        ambiguity_context={
            "od_ambiguity_index": 0.40,
            "od_hard_case_prior": 0.49,
            "od_ambiguity_support_ratio": 0.68,
            "od_ambiguity_source_entropy": 0.84,
            "ambiguity_budget_prior": 0.42,
        },
        support_richness=0.65,
        ambiguity_pressure=0.62,
        certificate_margin=0.77,
        near_tie_mass=0.0,
        top_refresh_gain=0.194872,
        top_fragility_mass=0.194872,
        competitor_pressure=1.0,
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:terrain",
        kind="refresh_top1_vor",
        target="terrain",
        q_score=0.05,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.004,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
        },
        reason="refresh_evidence_family",
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.28,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.18,
        predicted_delta_frontier=0.11,
        metadata={
            "mean_flip_probability": 0.99,
            "normalized_objective_gap": 0.09,
            "normalized_mechanism_gap": 0.35,
            "normalized_overlap_reduction": 0.91,
        },
        reason="refine_candidate",
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_post_evidence_certified_search_backslide(
        [refine, refresh, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert any(action.kind == "refine_top1_dccs" for action in filtered)


def test_post_evidence_certified_search_backslide_stops_modest_reopen_without_remaining_evidence_path() -> None:
    state = VOIControllerState(
        iteration_index=3,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 10.04, 10.02)},
            {"route_id": "route_c", "objective_vector": (10.2, 10.1, 10.0)},
            {"route_id": "route_d", "objective_vector": (10.35, 10.22, 10.11)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0, "route_c": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=0,
        action_trace=[
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": 0.24031,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": -0.23031,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": -0.046511,
                "realized_productive": True,
            },
        ],
        ambiguity_context={
            "od_ambiguity_index": 0.59,
            "od_hard_case_prior": 0.61,
            "od_ambiguity_support_ratio": 0.73,
            "od_ambiguity_source_entropy": 0.84,
            "ambiguity_budget_prior": 0.58,
        },
        support_richness=0.728969,
        ambiguity_pressure=0.574383,
        certificate_margin=1.0,
        near_tie_mass=0.0,
        top_refresh_gain=0.48062,
        top_fragility_mass=0.24031,
        competitor_pressure=1.0,
        pending_challenger_mass=0.656691,
        best_pending_flip_probability=0.999995,
        frontier_recall_at_budget=0.192142,
    )
    refine = voi_module.VOIAction(
        action_id="refine_topk_dccs:test",
        kind="refine_topk_dccs",
        target="cohort",
        q_score=0.028541,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.151925,
        predicted_delta_frontier=0.127334,
        metadata={
            "normalized_objective_gap": 0.061454,
            "normalized_mechanism_gap": 0.141653,
            "normalized_overlap_reduction": 0.598244,
            "mean_flip_probability": 0.993144,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_post_evidence_certified_search_backslide(
        [refine, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_post_action_controller_state_refresh_uses_updated_frontier_and_certificate() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    raw_state = VOIControllerState(
        iteration_index=2,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.32},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=1,
        action_trace=[{"iteration": 1, "kind": "refresh_top1_vor"}],
        active_evidence_families=["scenario"],
        near_tie_mass=0.12,
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_prior_strength": 0.66,
            "support_strength": 0.58,
        },
    )

    refreshed = refresh_controller_state_after_action(
        raw_state,
        dccs=dccs,
        fragility=fragility,
        config=VOIConfig(),
        frontier=[{"route_id": "route_b", "objective_vector": (9.4, 9.2, 9.1)}],
        certificate={"route_b": 0.91},
        winner_id="route_b",
        selected_route_id="route_b",
        remaining_search_budget=0,
        remaining_evidence_budget=0,
        active_evidence_families=["scenario"],
        refreshed_evidence_families=["scenario"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_prior_strength": 0.66,
            "support_strength": 0.58,
        },
        action_trace=[{"iteration": 1, "kind": "refresh_top1_vor"}],
    )

    assert refreshed.winner_id == "route_b"
    assert refreshed.selected_route_id == "route_b"
    assert refreshed.certificate == {"route_b": 0.91}
    assert refreshed.frontier[0]["route_id"] == "route_b"
    assert refreshed.remaining_search_budget == 0
    assert refreshed.remaining_evidence_budget == 0
    assert refreshed.top_refresh_gain >= raw_state.top_refresh_gain
    assert refreshed.support_richness >= raw_state.support_richness
    assert refreshed.competitor_pressure >= 0.0


def test_action_scoring_is_deterministic() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.25},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        near_tie_mass=0.2,
    )

    first = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig())
    second = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig())

    assert [action.as_dict() for action in first] == [action.as_dict() for action in second]
    assert first[0].kind == "refresh_top1_vor"
    assert first[0].q_score >= first[1].q_score


def test_resample_action_requires_stochastic_mode() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.4},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        stochastic_enabled=False,
        near_tie_mass=0.2,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig())

    assert all(action.kind != "increase_stochastic_samples" for action in actions)


def test_certified_support_rich_zero_refresh_signal_uses_bridge_resample_when_stochastic_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = _support_rich_zero_signal_bridge_state(stochastic_enabled=False)
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]
    bridge = next(action for action in actions if action.kind == "increase_stochastic_samples")

    assert actions[0].kind == "increase_stochastic_samples"
    assert bridge.metadata["evidence_discovery_bridge"] is True
    assert "refine_top1_dccs" not in kinds
    assert "refine_topk_dccs" not in kinds


def test_certified_support_rich_zero_signal_prefers_stop_over_churn_when_sampler_is_already_live() -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = _support_rich_zero_signal_bridge_state(
        stochastic_enabled=True,
        requested_world_count=64,
        actual_world_count=64,
        unique_world_count=48,
    )

    actions = build_action_menu(
        enrich_controller_state_for_actioning(
            state,
            dccs=dccs,
            fragility=fragility,  # type: ignore[arg-type]
            config=VOIConfig(certificate_threshold=0.80),
        ),
        dccs=dccs,
        fragility=fragility,  # type: ignore[arg-type]
        config=VOIConfig(certificate_threshold=0.80),
    )

    kinds = {action.kind for action in actions}
    assert actions[0].kind == "stop"
    assert "refine_top1_dccs" not in kinds
    assert "increase_stochastic_samples" not in kinds
    assert "refresh_top1_vor" not in kinds


def test_certified_support_rich_zero_signal_prefers_stop_with_high_thesis_certificate_threshold() -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = _support_rich_zero_signal_bridge_state(
        stochastic_enabled=True,
        requested_world_count=64,
        actual_world_count=64,
        unique_world_count=48,
    )

    actions = build_action_menu(
        enrich_controller_state_for_actioning(
            state,
            dccs=dccs,
            fragility=fragility,  # type: ignore[arg-type]
            config=VOIConfig(certificate_threshold=0.82),
        ),
        dccs=dccs,
        fragility=fragility,  # type: ignore[arg-type]
        config=VOIConfig(certificate_threshold=0.82),
    )

    kinds = {action.kind for action in actions}
    assert actions[0].kind == "stop"
    assert "refine_top1_dccs" not in kinds
    assert "increase_stochastic_samples" not in kinds
    assert "refresh_top1_vor" not in kinds


def test_uncertified_single_frontier_structural_cap_uses_bridge_resample_even_when_stochastic_is_live() -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.747845},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.31,
            "od_hard_case_prior": 0.36,
            "od_ambiguity_support_ratio": 0.64,
            "od_ambiguity_source_entropy": 0.78,
            "ambiguity_budget_prior": 0.34,
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.747845,
            "refc_world_count": 0,
            "refc_unique_world_count": 0,
            "refc_requested_world_count": 96,
        },
        certificate_margin=0.0,
        search_completeness_score=0.606033,
        search_completeness_gap=0.233967,
        prior_support_strength=0.299975,
        support_richness=0.41,
        ambiguity_pressure=0.42,
        pending_challenger_mass=0.712341,
        best_pending_flip_probability=0.999155,
        frontier_recall_at_budget=0.61,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.82))
    bridge = next(action for action in actions if action.kind == "increase_stochastic_samples")

    assert actions[0].kind == "increase_stochastic_samples"
    assert bridge.reason == "increase_stochastic_samples_structural_cap_bridge"
    assert bridge.metadata["uncertified_structural_cap_bridge"] is True
    assert bridge.metadata["recoverable_certificate_gap"] > 0.20


def test_uncertified_single_frontier_structural_cap_settled_search_prefers_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.72},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.56,
            "od_hard_case_prior": 0.56,
            "od_ambiguity_support_ratio": 0.88,
            "od_ambiguity_source_entropy": 0.96,
            "ambiguity_budget_prior": 0.56,
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.72,
            "refc_world_count": 86,
            "refc_unique_world_count": 86,
            "refc_requested_world_count": 96,
        },
        certificate_margin=0.0,
        search_completeness_score=1.0,
        search_completeness_gap=0.0,
        prior_support_strength=0.525415,
        support_richness=0.728969,
        ambiguity_pressure=0.061404,
        pending_challenger_mass=0.0,
        best_pending_flip_probability=0.0,
        frontier_recall_at_budget=1.0,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.000475,
        competitor_pressure=0.0,
    )

    monkeypatch.setattr(
        voi_module,
        "enrich_controller_state_for_actioning",
        lambda state, **_: state,
    )
    monkeypatch.setattr(
        voi_module,
        "_pending_dccs_candidates",
        lambda _dccs, **_: [],
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.82))

    assert [action.kind for action in actions] == ["stop"]


def test_uncertified_structural_cap_bridge_preference_beats_mechanism_only_refine() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.747845},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
        },
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:structural_cap_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.137978,
        predicted_delta_certificate=0.173321,
        predicted_delta_margin=0.124917,
        predicted_delta_frontier=0.040819,
        metadata={
            "uncertified_structural_cap_bridge": True,
            "search_pressure": 0.850075,
            "support_signal": 0.645344,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.229771,
        predicted_delta_certificate=0.252353,
        predicted_delta_margin=0.224026,
        predicted_delta_frontier=0.126972,
        metadata={
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.0,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    adjusted = voi_module._apply_uncertified_structural_cap_bridge_preference(
        [refine, bridge],
        state=state,
        current_certificate=0.747845,
        config=VOIConfig(certificate_threshold=0.82),
    )

    adjusted_bridge = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    assert adjusted_bridge.q_score > adjusted_refine.q_score
    assert adjusted_bridge.metadata["structural_cap_bridge_preference_applied"] is True
    assert adjusted_refine.metadata["structural_cap_bridge_search_discount_applied"] is True


def test_uncertified_support_rich_zero_signal_bridge_offers_resample_when_sampler_is_disabled() -> None:
    cfg = VOIConfig(certificate_threshold=0.83)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.771728, "route_b": 0.228272},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "od_hard_case_prior": 0.34742,
            "ambiguity_budget_prior": 0.172,
            "od_ambiguity_support_ratio": 0.64632,
            "od_ambiguity_source_entropy": 0.78166,
            "refc_world_count": 1,
            "refc_unique_world_count": 1,
            "refc_requested_world_count": 56,
            "refc_world_count_shortfall": 55,
        },
        certificate_margin=0.771728,
        search_completeness_score=0.588995,
        search_completeness_gap=0.251005,
        prior_support_strength=0.591697,
        support_richness=0.591697,
        ambiguity_pressure=0.569307,
        pending_challenger_mass=0.617901,
        best_pending_flip_probability=0.998199,
        frontier_recall_at_budget=0.365994,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.214823,
        predicted_delta_certificate=0.228272,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.771728,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=refine,
    ) is True

    bridge = voi_module._build_uncertified_support_rich_zero_signal_bridge_action(
        state,
        current_certificate=0.771728,
        config=cfg,
    )

    assert bridge.kind == "increase_stochastic_samples"
    assert bridge.metadata["uncertified_support_rich_zero_signal_bridge"] is True
    assert bridge.predicted_delta_certificate >= 0.12


def test_uncertified_support_rich_zero_signal_bridge_can_reopen_live_sampler_on_fallback_rows() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.77},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_hard_case_prior": 0.363736,
            "ambiguity_budget_prior": 0.32,
            "od_ambiguity_support_ratio": 0.818061,
            "od_ambiguity_source_entropy": 0.878347,
            "single_frontier_certificate_cap_applied": True,
            "refc_stress_world_fraction": 0.18,
            "refc_world_count": 53,
            "refc_unique_world_count": 53,
            "refc_requested_world_count": 104,
            "refc_world_count_shortfall": 51,
        },
        certificate_margin=0.77,
        search_completeness_score=0.691549,
        search_completeness_gap=0.308451,
        prior_support_strength=0.346037,
        support_richness=0.58,
        ambiguity_pressure=0.63,
        pending_challenger_mass=0.732371,
        best_pending_flip_probability=0.999971,
        frontier_recall_at_budget=0.36679,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "carbon", "vor": 0.0}],
            "top_refresh_family": "carbon",
            "top_refresh_gain": 0.0,
            "controller_ranking_basis": "raw_refresh_gain_fallback",
            "top_refresh_family_controller": "fuel",
            "top_refresh_gain_controller": 0.000282,
            "controller_ranking": [
                {
                    "family": "fuel",
                    "controller_score": 0.000282,
                    "empirical_vor": 0.0,
                    "raw_refresh_gain": 0.000282,
                    "basis": "raw_refresh_gain_fallback",
                }
            ],
        },
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.214823,
        predicted_delta_certificate=0.228272,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.77,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=refine,
        fragility=fragility,
    ) is True

    bridge = voi_module._build_uncertified_support_rich_zero_signal_bridge_action(
        state,
        current_certificate=0.77,
        config=cfg,
        fragility=fragility,
    )
    bridge = voi_module.score_action(bridge, config=cfg)

    assert bridge.kind == "increase_stochastic_samples"
    assert bridge.metadata["uncertified_support_rich_zero_signal_bridge"] is True
    assert bridge.metadata["uncertified_support_rich_zero_signal_live_sampler_bridge"] is True
    assert bridge.metadata["controller_refresh_fallback_activated"] is True
    assert bridge.metadata["controller_empirical_vs_raw_refresh_disagreement"] is True
    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.77,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=None,
        fragility=fragility,
    ) is True


def test_uncertified_support_rich_zero_signal_bridge_uses_cert_world_shortfall_when_sampler_request_is_stale() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.77},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_hard_case_prior": 0.363736,
            "ambiguity_budget_prior": 0.32,
            "od_ambiguity_support_ratio": 0.818061,
            "od_ambiguity_source_entropy": 0.878347,
            "single_frontier_certificate_cap_applied": True,
            "refc_stress_world_fraction": 0.18,
            "refc_world_count": 53,
            "refc_unique_world_count": 53,
            "refc_requested_world_count": 104,
            "refc_sampler_requested_world_count": 1,
            "refc_world_count_shortfall": 51,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.77,
        },
        certificate_margin=0.77,
        search_completeness_score=0.691549,
        search_completeness_gap=0.308451,
        prior_support_strength=0.346037,
        support_richness=0.58,
        ambiguity_pressure=0.63,
        pending_challenger_mass=0.732371,
        best_pending_flip_probability=0.999971,
        frontier_recall_at_budget=0.36679,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "carbon", "vor": 0.0}],
            "top_refresh_family": "carbon",
            "top_refresh_gain": 0.0,
            "controller_ranking_basis": "raw_refresh_gain_fallback",
            "top_refresh_family_controller": "fuel",
            "top_refresh_gain_controller": 0.000282,
            "controller_ranking": [
                {
                    "family": "fuel",
                    "controller_score": 0.000282,
                    "empirical_vor": 0.0,
                    "raw_refresh_gain": 0.000282,
                    "basis": "raw_refresh_gain_fallback",
                }
            ],
        },
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.214823,
        predicted_delta_certificate=0.228272,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert voi_module._resample_shortfall_available(state) is False
    assert voi_module._cert_world_shortfall_available(state) is True
    assert voi_module._should_use_cert_world_support_rich_zero_signal_bridge(
        state,
        fragility=fragility,
    ) is True
    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.77,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=refine,
        fragility=fragility,
    ) is True

    bridge = voi_module._build_uncertified_support_rich_zero_signal_bridge_action(
        state,
        current_certificate=0.77,
        config=cfg,
        fragility=fragility,
    )
    bridge = voi_module.score_action(bridge, config=cfg)
    adjusted = voi_module._apply_uncertified_support_rich_zero_signal_bridge_preference(
        [refine, bridge],
        state=state,
        current_certificate=0.77,
        config=cfg,
    )
    adjusted_bridge = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")

    assert bridge.metadata["uncertified_support_rich_zero_signal_bridge"] is True
    assert bridge.metadata["uncertified_support_rich_zero_signal_live_sampler_bridge"] is False
    assert bridge.metadata["uncertified_support_rich_zero_signal_cert_world_bridge"] is True
    assert bridge.metadata["requested_world_count"] == 104
    assert bridge.metadata["sampler_requested_world_count"] == 1
    assert bridge.metadata["controller_requested_world_count"] == 104
    assert bridge.metadata["world_shortfall_ratio"] == pytest.approx(51 / 104, rel=1e-6)
    assert adjusted_bridge.q_score > adjusted_refine.q_score
    assert adjusted_bridge.metadata["uncertified_support_rich_zero_signal_bridge_preference_applied"] is True


def test_uncertified_support_rich_zero_signal_bridge_can_reopen_live_sampler_on_extreme_undercoverage_rows() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.77},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.172,
            "od_ambiguity_confidence": 0.964,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":8,"repo_local_geometry_backfill":4,"routing_graph_probe":1}',
            "od_candidate_path_count": 1,
            "od_corridor_family_count": 1,
            "od_hard_case_prior": 0.30177,
            "ambiguity_budget_prior": 0.32,
            "od_ambiguity_support_ratio": 0.72632,
            "od_ambiguity_source_entropy": 0.78166,
            "refc_world_count": 1,
            "refc_unique_world_count": 1,
            "refc_requested_world_count": 96,
            "refc_sampler_requested_world_count": 96,
            "refc_world_count_shortfall": 95,
        },
        certificate_margin=0.77,
        search_completeness_score=0.773486,
        search_completeness_gap=0.066514,
        prior_support_strength=0.291274,
        support_richness=0.569489,
        ambiguity_pressure=0.397797,
        pending_challenger_mass=0.632488,
        best_pending_flip_probability=0.999718,
        frontier_recall_at_budget=1.0,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "carbon", "vor": 0.0}],
            "top_refresh_family": "carbon",
            "top_refresh_gain": 0.0,
        },
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.212479,
        predicted_delta_certificate=0.242174,
        predicted_delta_margin=0.215858,
        predicted_delta_frontier=0.088066,
        metadata={
            "normalized_objective_gap": 0.081717,
            "normalized_mechanism_gap": 0.640251,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.77,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=refine,
        fragility=fragility,
    ) is True

    bridge = voi_module._build_uncertified_support_rich_zero_signal_bridge_action(
        state,
        current_certificate=0.77,
        config=cfg,
        fragility=fragility,
    )
    bridge = voi_module.score_action(bridge, config=cfg)

    assert bridge.kind == "increase_stochastic_samples"
    assert bridge.metadata["uncertified_support_rich_zero_signal_live_sampler_bridge"] is True
    assert bridge.metadata["uncertified_support_rich_zero_signal_extreme_undercoverage"] is True

    adjusted = voi_module._apply_uncertified_support_rich_zero_signal_bridge_preference(
        [
            refine,
            bridge,
            voi_module.VOIAction(
                action_id="stop",
                kind="stop",
                target="stop",
                q_score=0.0,
                feasible=True,
                preconditions=("always",),
                reason="stop",
            ),
        ],
        state=state,
        current_certificate=0.77,
        config=cfg,
    )

    adjusted_bridge = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    assert adjusted_bridge.q_score > adjusted_refine.q_score
    assert adjusted_bridge.metadata["uncertified_support_rich_zero_signal_bridge_preference_applied"] is True
    assert adjusted_refine.metadata["uncertified_support_rich_zero_signal_bridge_search_discount_applied"] is True


def test_uncertified_support_rich_zero_signal_bridge_rejects_live_sampler_when_only_controller_request_grows() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.77},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_hard_case_prior": 0.363736,
            "ambiguity_budget_prior": 0.32,
            "od_ambiguity_support_ratio": 0.818061,
            "od_ambiguity_source_entropy": 0.878347,
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "refc_stress_world_fraction": 0.18,
            "refc_world_count": 53,
            "refc_unique_world_count": 53,
            "refc_requested_world_count": 104,
            "refc_sampler_requested_world_count": 1,
            "refc_world_count_shortfall": 51,
        },
        certificate_margin=0.77,
        search_completeness_score=0.691549,
        search_completeness_gap=0.308451,
        prior_support_strength=0.346037,
        support_richness=0.58,
        ambiguity_pressure=0.63,
        pending_challenger_mass=0.732371,
        best_pending_flip_probability=0.999971,
        frontier_recall_at_budget=0.36679,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "carbon", "vor": 0.0}],
            "top_refresh_family": "carbon",
            "top_refresh_gain": 0.0,
            "controller_ranking_basis": "raw_refresh_gain_fallback",
            "top_refresh_family_controller": "fuel",
            "top_refresh_gain_controller": 0.000282,
            "controller_ranking": [
                {
                    "family": "fuel",
                    "controller_score": 0.000282,
                    "empirical_vor": 0.0,
                    "raw_refresh_gain": 0.000282,
                    "basis": "raw_refresh_gain_fallback",
                }
            ],
        },
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.214823,
        predicted_delta_certificate=0.228272,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.77,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=refine,
        fragility=fragility,
    ) is False


def test_uncertified_evidence_plateau_preference_can_choose_evidence_first_on_low_spread_near_tie_row() -> None:
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.4, "route_b": 0.6},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        ambiguity_context={
            "od_ambiguity_index": 0.02,
            "od_ambiguity_confidence": 0.864,
            "od_engine_disagreement_prior": 0.357128,
            "od_hard_case_prior": 0.370399,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":12,"routing_graph_probe":3}',
            "od_ambiguity_support_ratio": 0.438187,
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_source_entropy": 0.721928,
            "od_ambiguity_prior_strength": 0.02,
            "od_ambiguity_family_density": 0.3,
            "ambiguity_budget_prior": 0.02,
            "od_objective_spread": 0.0,
            "od_ambiguity_margin_pressure": 0.0,
            "od_candidate_path_count": 10,
            "od_corridor_family_count": 3,
            "od_nominal_margin_proxy": 1.0,
        },
        certificate_margin=0.0,
        search_completeness_score=0.466196,
        search_completeness_gap=0.373804,
        prior_support_strength=0.312407,
        support_richness=0.553078,
        ambiguity_pressure=0.779229,
        pending_challenger_mass=0.641344,
        best_pending_flip_probability=0.997754,
        frontier_recall_at_budget=0.145467,
        near_tie_mass=1.0,
        top_refresh_gain=0.375,
        top_fragility_mass=0.4,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.3275036519,
        predicted_delta_certificate=0.433182,
        predicted_delta_margin=0.219478,
        predicted_delta_frontier=0.084832,
        metadata={
            "normalized_objective_gap": 0.042138,
            "normalized_mechanism_gap": 0.219664,
            "normalized_overlap_reduction": 0.904762,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.1807,
        predicted_delta_certificate=0.252,
        predicted_delta_margin=0.1,
        predicted_delta_frontier=0.03,
        metadata={
            "near_tie_mass": 1.0,
            "stress_world_fraction": 0.6,
            "top_fragility_mass": 0.4,
            "sample_increment": 32,
        },
    )

    adjusted = voi_module._apply_uncertified_evidence_plateau_preference(
        [refine, resample],
        state=state,
        current_certificate=0.4,
        config=cfg,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=False,
    )

    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    adjusted_resample = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    assert adjusted_refine.q_score == pytest.approx(refine.q_score)
    assert adjusted_resample.q_score == pytest.approx(resample.q_score)
    assert "uncertified_evidence_plateau_first_iteration_near_tie" not in adjusted_resample.metadata
    assert "uncertified_evidence_plateau_first_iteration_search_discount" not in adjusted_refine.metadata


def test_uncertified_evidence_plateau_preference_can_use_resample_metadata_near_tie_mass() -> None:
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.4, "route_b": 0.6},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        ambiguity_context={
            "od_ambiguity_index": 0.02,
            "od_ambiguity_confidence": 0.864,
            "od_engine_disagreement_prior": 0.357128,
            "od_hard_case_prior": 0.370399,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":12,"routing_graph_probe":3}',
            "od_ambiguity_support_ratio": 0.438187,
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_source_entropy": 0.721928,
            "od_ambiguity_prior_strength": 0.02,
            "od_ambiguity_family_density": 0.3,
            "ambiguity_budget_prior": 0.02,
            "od_objective_spread": 0.0,
            "od_ambiguity_margin_pressure": 0.0,
            "od_candidate_path_count": 10,
            "od_corridor_family_count": 3,
            "od_nominal_margin_proxy": 1.0,
        },
        certificate_margin=0.0,
        search_completeness_score=0.821196,
        search_completeness_gap=0.178804,
        prior_support_strength=0.312407,
        support_richness=0.553078,
        ambiguity_pressure=0.779229,
        pending_challenger_mass=0.541344,
        best_pending_flip_probability=0.997754,
        frontier_recall_at_budget=0.545467,
        near_tie_mass=0.0,
        top_refresh_gain=0.375,
        top_fragility_mass=0.4,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.3275036519,
        predicted_delta_certificate=0.433182,
        predicted_delta_margin=0.219478,
        predicted_delta_frontier=0.084832,
        metadata={
            "normalized_objective_gap": 0.042138,
            "normalized_mechanism_gap": 0.219664,
            "normalized_overlap_reduction": 0.904762,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.1807,
        predicted_delta_certificate=0.252,
        predicted_delta_margin=0.1,
        predicted_delta_frontier=0.03,
        metadata={
            "near_tie_mass": 1.0,
            "stress_world_fraction": 0.6,
            "top_fragility_mass": 0.4,
            "sample_increment": 32,
        },
    )

    adjusted = voi_module._apply_uncertified_evidence_plateau_preference(
        [refine, resample],
        state=state,
        current_certificate=0.4,
        config=cfg,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    adjusted_resample = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_evidence_plateau_first_iteration_near_tie"] is True
    assert adjusted_refine.metadata["uncertified_evidence_plateau_first_iteration_search_discount"] is True


def test_uncertified_support_rich_zero_signal_bridge_uses_cert_world_shortfall_when_stochastic_is_disabled() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.79},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        stochastic_enabled=False,
        ambiguity_context={
            "od_hard_case_prior": 0.363736,
            "ambiguity_budget_prior": 0.32,
            "od_ambiguity_support_ratio": 0.818061,
            "od_ambiguity_source_entropy": 0.878347,
            "single_frontier_certificate_cap_applied": True,
            "refc_stress_world_fraction": 0.18,
            "refc_world_count": 41,
            "refc_unique_world_count": 41,
            "refc_requested_world_count": 64,
            "refc_sampler_requested_world_count": 1,
            "refc_world_count_shortfall": 23,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.79,
        },
        certificate_margin=0.79,
        search_completeness_score=0.691549,
        search_completeness_gap=0.308451,
        prior_support_strength=0.346037,
        support_richness=0.58,
        ambiguity_pressure=0.63,
        pending_challenger_mass=0.732371,
        best_pending_flip_probability=0.999971,
        frontier_recall_at_budget=0.36679,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "carbon", "vor": 0.0}],
            "top_refresh_family": "carbon",
            "top_refresh_gain": 0.0,
            "controller_ranking_basis": "raw_refresh_gain_fallback",
            "top_refresh_family_controller": "fuel",
            "top_refresh_gain_controller": 0.000282,
            "controller_ranking": [
                {
                    "family": "fuel",
                    "controller_score": 0.000282,
                    "empirical_vor": 0.0,
                    "raw_refresh_gain": 0.000282,
                    "basis": "raw_refresh_gain_fallback",
                }
            ],
        },
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.214823,
        predicted_delta_certificate=0.210001,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert voi_module._resample_shortfall_available(state) is True
    assert voi_module._cert_world_shortfall_available(state) is True
    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.79,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=refine,
        fragility=fragility,
    ) is True

    bridge = voi_module._build_uncertified_support_rich_zero_signal_bridge_action(
        state,
        current_certificate=0.79,
        config=cfg,
        fragility=fragility,
    )
    bridge = voi_module.score_action(bridge, config=cfg)
    adjusted = voi_module._apply_uncertified_support_rich_zero_signal_bridge_preference(
        [refine, bridge],
        state=state,
        current_certificate=0.79,
        config=cfg,
    )
    adjusted_bridge = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")

    assert bridge.metadata["uncertified_support_rich_zero_signal_bridge"] is True
    assert bridge.metadata["uncertified_support_rich_zero_signal_live_sampler_bridge"] is True
    assert bridge.metadata["uncertified_support_rich_zero_signal_cert_world_bridge"] is True
    assert bridge.metadata["controller_refresh_fallback_activated"] is True
    assert bridge.metadata["controller_empirical_vs_raw_refresh_disagreement"] is True
    assert adjusted_bridge.q_score > adjusted_refine.q_score
    assert adjusted_bridge.metadata["uncertified_support_rich_zero_signal_bridge_preference_applied"] is True


def test_uncertified_support_rich_zero_signal_bridge_blocks_disabled_live_sampler_without_controller_evidence() -> None:
    cfg = VOIConfig(certificate_threshold=0.83)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.771728},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "od_hard_case_prior": 0.34742,
            "ambiguity_budget_prior": 0.172,
            "od_ambiguity_support_ratio": 0.64632,
            "od_ambiguity_source_entropy": 0.78166,
            "refc_world_count": 1,
            "refc_requested_world_count": 56,
            "refc_world_count_shortfall": 55,
        },
        certificate_margin=0.771728,
        search_completeness_score=0.588995,
        search_completeness_gap=0.251005,
        prior_support_strength=0.591697,
        support_richness=0.591697,
        ambiguity_pressure=0.569307,
        pending_challenger_mass=0.617901,
        best_pending_flip_probability=0.998199,
        frontier_recall_at_budget=0.365994,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.214823,
        predicted_delta_certificate=0.228272,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert voi_module._support_rich_ambiguity_window(state) is False
    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.771728,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=refine,
    ) is False


def test_uncertified_support_rich_zero_signal_cert_world_bridge_can_bypass_support_window_on_high_stress_structural_cap_rows() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.775},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "od_hard_case_prior": 0.344476,
            "ambiguity_budget_prior": 0.01,
            "od_ambiguity_support_ratio": 0.442187,
            "od_ambiguity_source_entropy": 0.591673,
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.775,
            "refc_world_count": 41,
            "refc_unique_world_count": 41,
            "refc_requested_world_count": 60,
            "refc_sampler_requested_world_count": 1,
            "refc_world_count_shortfall": 19,
            "refc_stress_world_fraction": 0.97561,
        },
        certificate_margin=0.775,
        search_completeness_score=0.548057,
        search_completeness_gap=0.291943,
        prior_support_strength=0.290543,
        support_richness=0.516905,
        ambiguity_pressure=0.630042,
        pending_challenger_mass=0.65969,
        best_pending_flip_probability=0.998801,
        frontier_recall_at_budget=0.14903,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "carbon", "vor": 0.0}],
            "top_refresh_family": "carbon",
            "top_refresh_gain": 0.0,
            "controller_ranking_basis": "raw_refresh_gain_fallback",
            "top_refresh_family_controller": "fuel",
            "top_refresh_gain_controller": 0.000282,
            "controller_ranking": [
                {
                    "family": "fuel",
                    "controller_score": 0.000282,
                    "empirical_vor": 0.0,
                    "raw_refresh_gain": 0.000282,
                    "basis": "raw_refresh_gain_fallback",
                }
            ],
        },
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )

    assert voi_module._support_rich_ambiguity_window(state) is False
    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.775,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=None,
        fragility=fragility,
    ) is True

    bridge = voi_module.score_action(
        voi_module._build_uncertified_support_rich_zero_signal_bridge_action(
            state,
            current_certificate=0.775,
            config=cfg,
            fragility=fragility,
        ),
        config=cfg,
    )

    assert bridge.metadata["uncertified_support_rich_zero_signal_live_sampler_bridge"] is False
    assert bridge.metadata["uncertified_support_rich_zero_signal_cert_world_bridge"] is True
    assert bridge.metadata["controller_refresh_fallback_activated"] is True
    assert bridge.metadata["controller_empirical_vs_raw_refresh_disagreement"] is True


def test_uncertified_support_rich_zero_signal_cert_world_bridge_rejects_tiny_zero_signal_headroom() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.8},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "od_hard_case_prior": 0.344476,
            "ambiguity_budget_prior": 0.01,
            "od_ambiguity_support_ratio": 0.442187,
            "od_ambiguity_source_entropy": 0.591673,
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.8,
            "refc_world_count": 41,
            "refc_unique_world_count": 41,
            "refc_requested_world_count": 60,
            "refc_sampler_requested_world_count": 1,
            "refc_world_count_shortfall": 19,
            "refc_stress_world_fraction": 0.97561,
        },
        certificate_margin=0.8,
        search_completeness_score=0.548057,
        search_completeness_gap=0.291943,
        prior_support_strength=0.290543,
        support_richness=0.516905,
        ambiguity_pressure=0.630042,
        pending_challenger_mass=0.65969,
        best_pending_flip_probability=0.998801,
        frontier_recall_at_budget=0.14903,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "carbon", "vor": 0.0}],
            "top_refresh_family": "carbon",
            "top_refresh_gain": 0.0,
            "controller_ranking_basis": "raw_refresh_gain_fallback",
            "top_refresh_family_controller": "fuel",
            "top_refresh_gain_controller": 0.0,
            "controller_ranking": [
                {
                    "family": "fuel",
                    "controller_score": 0.0,
                    "empirical_vor": 0.0,
                    "raw_refresh_gain": 0.0,
                    "basis": "raw_refresh_gain_fallback",
                }
            ],
        },
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )

    assert voi_module._should_offer_uncertified_support_rich_zero_signal_bridge(
        state,
        current_certificate=0.8,
        config=cfg,
        recent_no_gain_controller_streak=0,
        best_search_action=None,
        fragility=fragility,
    ) is False


def test_uncertified_support_rich_zero_signal_bridge_preference_beats_weak_refine() -> None:
    cfg = VOIConfig(certificate_threshold=0.83)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.771728, "route_b": 0.228272},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "od_hard_case_prior": 0.34742,
            "ambiguity_budget_prior": 0.172,
            "od_ambiguity_support_ratio": 0.64632,
            "od_ambiguity_source_entropy": 0.78166,
            "refc_world_count": 1,
            "refc_unique_world_count": 1,
            "refc_requested_world_count": 56,
            "refc_world_count_shortfall": 55,
        },
        certificate_margin=0.771728,
        search_completeness_score=0.588995,
        search_completeness_gap=0.251005,
        prior_support_strength=0.591697,
        support_richness=0.591697,
        ambiguity_pressure=0.569307,
        pending_challenger_mass=0.617901,
        best_pending_flip_probability=0.998199,
        frontier_recall_at_budget=0.365994,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    bridge = voi_module.score_action(
        voi_module._build_uncertified_support_rich_zero_signal_bridge_action(
            state,
            current_certificate=0.771728,
            config=cfg,
        ),
        config=cfg,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.214823,
        predicted_delta_certificate=0.228272,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    adjusted = voi_module._apply_uncertified_support_rich_zero_signal_bridge_preference(
        [refine, bridge],
        state=state,
        current_certificate=0.771728,
        config=cfg,
    )

    adjusted_bridge = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    assert adjusted_bridge.q_score > adjusted_refine.q_score
    assert adjusted_bridge.metadata["uncertified_support_rich_zero_signal_bridge_preference_applied"] is True
    assert adjusted_refine.metadata["uncertified_support_rich_zero_signal_bridge_search_discount_applied"] is True


def test_uncertified_support_rich_zero_signal_live_sampler_bridge_preference_beats_weak_refine() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.77},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_hard_case_prior": 0.363736,
            "ambiguity_budget_prior": 0.32,
            "od_ambiguity_support_ratio": 0.818061,
            "od_ambiguity_source_entropy": 0.878347,
            "single_frontier_certificate_cap_applied": True,
            "refc_world_count": 53,
            "refc_unique_world_count": 53,
            "refc_requested_world_count": 104,
            "refc_world_count_shortfall": 51,
        },
        certificate_margin=0.77,
        search_completeness_score=0.691549,
        search_completeness_gap=0.308451,
        prior_support_strength=0.346037,
        support_richness=0.58,
        ambiguity_pressure=0.63,
        pending_challenger_mass=0.732371,
        best_pending_flip_probability=0.999971,
        frontier_recall_at_budget=0.36679,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.08,
        predicted_delta_certificate=0.23,
        predicted_delta_margin=0.12,
        predicted_delta_frontier=0.06,
        metadata={
            "uncertified_support_rich_zero_signal_bridge": True,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": True,
            "controller_refresh_fallback_activated": True,
            "controller_empirical_vs_raw_refresh_disagreement": True,
            "search_pressure": 0.999971,
            "support_signal": 0.58,
            "hard_case_support": 0.818061,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.214823,
        predicted_delta_certificate=0.228272,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    adjusted = voi_module._apply_uncertified_support_rich_zero_signal_bridge_preference(
        [refine, bridge],
        state=state,
        current_certificate=0.77,
        config=cfg,
    )

    adjusted_bridge = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    assert adjusted_bridge.q_score > adjusted_refine.q_score
    assert adjusted_bridge.metadata["uncertified_support_rich_zero_signal_bridge_preference_applied"] is True
    assert adjusted_refine.metadata["uncertified_support_rich_zero_signal_bridge_search_discount_applied"] is True


def test_uncertified_support_rich_zero_signal_bridge_keeps_bridge_ahead_for_mechanism_only_single_frontier_row() -> None:
    cfg = VOIConfig(certificate_threshold=0.84)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.805054},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=4,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_hard_case_prior": 0.522223,
            "ambiguity_budget_prior": 0.522223,
            "od_ambiguity_support_ratio": 0.587187,
            "od_ambiguity_source_entropy": 0.845351,
            "refc_world_count": 1,
            "refc_unique_world_count": 1,
            "refc_requested_world_count": 100,
            "refc_world_count_shortfall": 99,
        },
        certificate_margin=0.805054,
        search_completeness_score=0.627983,
        search_completeness_gap=0.212017,
        prior_support_strength=0.342326,
        support_richness=0.337034,
        ambiguity_pressure=0.508066,
        pending_challenger_mass=0.714966,
        best_pending_flip_probability=0.998774,
        frontier_recall_at_budget=0.365524,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.170529,
        predicted_delta_certificate=0.194946,
        predicted_delta_margin=0.152906,
        predicted_delta_frontier=0.102228,
        metadata={
            "uncertified_support_rich_zero_signal_bridge": True,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": True,
            "uncertified_support_rich_zero_signal_extreme_undercoverage": True,
            "search_pressure": 0.998774,
            "support_signal": 0.508066,
            "hard_case_support": 0.522223,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.116576,
        predicted_delta_certificate=0.194946,
        predicted_delta_margin=0.188002,
        predicted_delta_frontier=0.017264,
        metadata={
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.800338,
            "normalized_overlap_reduction": 0.9,
        },
    )

    adjusted = voi_module._apply_uncertified_support_rich_zero_signal_bridge_preference(
        [refine, bridge],
        state=state,
        current_certificate=0.805054,
        config=cfg,
    )

    adjusted_bridge = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    assert adjusted_bridge.q_score > adjusted_refine.q_score
    assert adjusted_bridge.metadata["uncertified_support_rich_zero_signal_bridge_preference_applied"] is True
    assert "uncertified_support_rich_zero_signal_search_finish_preferred" not in adjusted_refine.metadata


def test_uncertified_support_rich_zero_signal_bridge_defers_when_single_frontier_search_has_objective_support() -> None:
    cfg = VOIConfig(certificate_threshold=0.84)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.805054},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=4,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_hard_case_prior": 0.522223,
            "ambiguity_budget_prior": 0.522223,
            "od_ambiguity_support_ratio": 0.587187,
            "od_ambiguity_source_entropy": 0.845351,
            "refc_world_count": 1,
            "refc_unique_world_count": 1,
            "refc_requested_world_count": 100,
            "refc_world_count_shortfall": 99,
        },
        certificate_margin=0.805054,
        search_completeness_score=0.627983,
        search_completeness_gap=0.212017,
        prior_support_strength=0.342326,
        support_richness=0.337034,
        ambiguity_pressure=0.508066,
        pending_challenger_mass=0.714966,
        best_pending_flip_probability=0.998774,
        frontier_recall_at_budget=0.365524,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.170529,
        predicted_delta_certificate=0.194946,
        predicted_delta_margin=0.152906,
        predicted_delta_frontier=0.102228,
        metadata={
            "uncertified_support_rich_zero_signal_bridge": True,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": True,
            "uncertified_support_rich_zero_signal_extreme_undercoverage": True,
            "search_pressure": 0.998774,
            "support_signal": 0.508066,
            "hard_case_support": 0.522223,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.116576,
        predicted_delta_certificate=0.194946,
        predicted_delta_margin=0.188002,
        predicted_delta_frontier=0.052264,
        metadata={
            "normalized_objective_gap": 0.053,
            "normalized_mechanism_gap": 0.800338,
            "normalized_overlap_reduction": 0.9,
        },
    )

    adjusted = voi_module._apply_uncertified_support_rich_zero_signal_bridge_preference(
        [refine, bridge],
        state=state,
        current_certificate=0.805054,
        config=cfg,
    )

    adjusted_bridge = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    assert adjusted_refine.q_score > adjusted_bridge.q_score
    assert adjusted_refine.metadata["uncertified_support_rich_zero_signal_search_finish_preferred"] is True
    assert adjusted_bridge.metadata["uncertified_support_rich_zero_signal_bridge_finish_deferred"] is True


def test_uncertified_support_rich_zero_signal_live_sampler_bridge_override_beats_mechanism_heavy_refine_on_structural_cap_row() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.74878},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        stochastic_enabled=True,
        ambiguity_context={
            "od_hard_case_prior": 0.363736,
            "ambiguity_budget_prior": 0.32,
            "od_ambiguity_support_ratio": 0.685387,
            "od_ambiguity_source_entropy": 0.943545,
            "single_frontier_certificate_cap_applied": True,
            "refc_world_count": 45,
            "refc_unique_world_count": 45,
            "refc_requested_world_count": 72,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.74878,
        },
        certificate_margin=0.74878,
        search_completeness_score=0.493667,
        search_completeness_gap=0.346333,
        prior_support_strength=0.605436,
        support_richness=0.605436,
        ambiguity_pressure=0.67535,
        pending_challenger_mass=0.734447,
        best_pending_flip_probability=0.99997,
        frontier_recall_at_budget=0.223681,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.208713,
        predicted_delta_certificate=0.25122,
        predicted_delta_margin=0.18360911,
        predicted_delta_frontier=0.08052488,
        metadata={
            "uncertified_support_rich_zero_signal_bridge": True,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": True,
            "controller_refresh_fallback_activated": True,
            "controller_empirical_vs_raw_refresh_disagreement": True,
            "search_pressure": 0.99997,
            "support_signal": 0.685387,
            "hard_case_support": 0.561025,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.225131,
        predicted_delta_certificate=0.25122,
        predicted_delta_margin=0.231084,
        predicted_delta_frontier=0.11085,
        metadata={
            "normalized_objective_gap": 0.069597,
            "normalized_mechanism_gap": 0.825578,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    adjusted = voi_module._apply_uncertified_support_rich_zero_signal_bridge_preference(
        [refine, bridge],
        state=state,
        current_certificate=0.74878,
        config=cfg,
    )

    adjusted_bridge = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    assert adjusted_bridge.q_score > adjusted_refine.q_score
    assert adjusted_bridge.metadata["uncertified_support_rich_zero_signal_bridge_preference_applied"] is True
    assert adjusted_refine.metadata["uncertified_support_rich_zero_signal_bridge_search_discount_applied"] is True


def test_uncertified_structural_cap_churn_preserves_live_support_bridge() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.77},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.77,
        },
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.18,
        predicted_delta_certificate=0.23,
        predicted_delta_margin=0.12,
        predicted_delta_frontier=0.06,
        metadata={
            "uncertified_support_rich_zero_signal_bridge": True,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": True,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_structural_cap_churn(
        [bridge, stop],
        state=state,
        current_certificate=0.77,
        config=VOIConfig(certificate_threshold=0.82),
    )

    assert [action.kind for action in filtered] == ["increase_stochastic_samples", "stop"]


def test_uncertified_stochastic_disabled_zero_signal_controller_churn_prefers_stop() -> None:
    cfg = VOIConfig(certificate_threshold=0.83)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.769681},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "od_hard_case_prior": 0.31507,
            "ambiguity_budget_prior": 0.22,
            "od_ambiguity_support_ratio": 0.73112,
            "od_ambiguity_source_entropy": 0.78166,
            "refc_world_count": 1,
            "refc_requested_world_count": 64,
            "refc_world_count_shortfall": 63,
        },
        certificate_margin=0.769681,
        search_completeness_score=0.603316,
        search_completeness_gap=0.236684,
        prior_support_strength=0.575133,
        support_richness=0.575133,
        ambiguity_pressure=0.559784,
        pending_challenger_mass=0.601996,
        best_pending_flip_probability=0.998244,
        frontier_recall_at_budget=0.371663,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.189933,
        predicted_delta_certificate=0.230319,
        predicted_delta_margin=0.192131,
        predicted_delta_frontier=0.024693,
        metadata={
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.073179,
            "normalized_overlap_reduction": 0.909091,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_stochastic_disabled_zero_signal_controller_churn(
        [refine, stop],
        state=state,
        current_certificate=0.769681,
        config=cfg,
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_uncertified_stochastic_disabled_zero_signal_controller_churn_keeps_preferred_bridge() -> None:
    cfg = VOIConfig(certificate_threshold=0.83)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.764219},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "od_hard_case_prior": 0.361974,
            "ambiguity_budget_prior": 0.07,
            "od_ambiguity_support_ratio": 0.429879,
            "od_ambiguity_source_entropy": 0.721928,
            "refc_world_count": 41,
            "refc_requested_world_count": 64,
            "refc_world_count_shortfall": 23,
        },
        certificate_margin=0.764219,
        search_completeness_score=0.510803,
        search_completeness_gap=0.329197,
        prior_support_strength=0.549023,
        support_richness=0.549023,
        ambiguity_pressure=0.668762,
        pending_challenger_mass=0.711813,
        best_pending_flip_probability=0.997734,
        frontier_recall_at_budget=0.124655,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:structural_cap_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.200001,
        predicted_delta_certificate=0.179507,
        predicted_delta_margin=0.123811,
        predicted_delta_frontier=0.047145,
        metadata={
            "uncertified_structural_cap_bridge": True,
            "structural_cap_bridge_preference_applied": True,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.196965,
        predicted_delta_certificate=0.235781,
        predicted_delta_margin=0.201011,
        predicted_delta_frontier=0.03662,
        metadata={
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.0,
            "normalized_overlap_reduction": 0.909091,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_stochastic_disabled_zero_signal_controller_churn(
        [refine, bridge, stop],
        state=state,
        current_certificate=0.764219,
        config=cfg,
    )

    assert [action.kind for action in filtered] == ["increase_stochastic_samples", "stop"]


def test_uncertified_stochastic_disabled_zero_signal_controller_churn_preserves_supported_hard_case_search() -> None:
    cfg = VOIConfig(certificate_threshold=0.83)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.771728},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "od_hard_case_prior": 0.34742,
            "ambiguity_budget_prior": 0.172,
            "od_ambiguity_support_ratio": 0.64632,
            "od_ambiguity_source_entropy": 0.78166,
            "refc_world_count": 1,
            "refc_requested_world_count": 56,
            "refc_world_count_shortfall": 55,
        },
        certificate_margin=0.771728,
        search_completeness_score=0.588995,
        search_completeness_gap=0.251005,
        prior_support_strength=0.591697,
        support_richness=0.591697,
        ambiguity_pressure=0.569307,
        pending_challenger_mass=0.617901,
        best_pending_flip_probability=0.998199,
        frontier_recall_at_budget=0.365994,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.214823,
        predicted_delta_certificate=0.228272,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_stochastic_disabled_zero_signal_controller_churn(
        [refine, stop],
        state=state,
        current_certificate=0.771728,
        config=cfg,
    )

    assert [action.kind for action in filtered] == ["refine_top1_dccs", "stop"]


def test_uncertified_structural_cap_bridge_requires_zero_actual_worlds() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.769002},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.769002,
            "od_hard_case_prior": 0.344476,
            "ambiguity_budget_prior": 0.01,
            "od_ambiguity_support_ratio": 0.442187,
            "od_ambiguity_source_entropy": 0.591673,
            "refc_world_count": 41,
            "refc_unique_world_count": 41,
            "refc_requested_world_count": 60,
            "refc_world_count_shortfall": 19,
        },
        certificate_margin=0.769002,
        search_completeness_score=0.548057,
        search_completeness_gap=0.291943,
        prior_support_strength=0.290543,
        support_richness=0.516905,
        ambiguity_pressure=0.630042,
        pending_challenger_mass=0.65969,
        best_pending_flip_probability=0.998801,
        frontier_recall_at_budget=0.14903,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.210098,
        predicted_delta_certificate=0.230998,
        predicted_delta_margin=0.224355,
        predicted_delta_frontier=0.102738,
        metadata={
            "mean_flip_probability": 0.998801,
            "normalized_objective_gap": 0.072307,
            "normalized_mechanism_gap": 0.358536,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert not voi_module._should_offer_uncertified_structural_cap_bridge(
        state,
        current_certificate=0.769002,
        config=VOIConfig(certificate_threshold=0.81),
        recent_no_gain_controller_streak=0,
        best_search_action=refine,
    )


def test_uncertified_structural_cap_bridge_override_requires_meaningful_shortfall() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.747647},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        stochastic_enabled=False,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.747647,
            "od_hard_case_prior": 0.483837,
            "ambiguity_budget_prior": 0.026667,
            "od_ambiguity_support_ratio": 0.496472,
            "od_ambiguity_source_entropy": 0.998636,
            "refc_world_count": 61,
            "refc_unique_world_count": 61,
            "refc_requested_world_count": 64,
            "refc_world_count_shortfall": 3,
        },
        certificate_margin=0.747647,
        search_completeness_score=0.483261,
        search_completeness_gap=0.356739,
        prior_support_strength=0.337358,
        support_richness=0.516882,
        ambiguity_pressure=0.670941,
        pending_challenger_mass=0.702872,
        best_pending_flip_probability=0.997762,
        frontier_recall_at_budget=0.118349,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.229771,
        predicted_delta_certificate=0.252353,
        predicted_delta_margin=0.224026,
        predicted_delta_frontier=0.126972,
        metadata={
            "mean_flip_probability": 0.997762,
            "normalized_objective_gap": 0.081255,
            "normalized_mechanism_gap": 0.063474,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert not voi_module._structural_cap_bridge_can_override_moderate_search_signal(
        state,
        shortfall_ratio=3.0 / 64.0,
        current_certificate=0.747647,
        config=VOIConfig(certificate_threshold=0.80),
        best_search_action=refine,
    )


def test_post_nonproductive_bridge_zero_signal_search_churn_prefers_stop() -> None:
    cfg = VOIConfig(certificate_threshold=0.81)
    state = VOIControllerState(
        iteration_index=1,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.755669},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=1,
        stochastic_enabled=False,
        action_trace=[
            {
                "chosen_action": {
                    "kind": "increase_stochastic_samples",
                    "metadata": {
                        "evidence_discovery_bridge": True,
                        "structural_cap_bridge": True,
                    },
                },
                "realized_certificate_delta": -0.013333,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.755669,
            "od_hard_case_prior": 0.344476,
            "ambiguity_budget_prior": 0.01,
            "od_ambiguity_support_ratio": 0.442187,
            "od_ambiguity_source_entropy": 0.591673,
            "refc_world_count": 41,
            "refc_unique_world_count": 41,
            "refc_requested_world_count": 92,
            "refc_world_count_shortfall": 51,
        },
        certificate_margin=0.755669,
        search_completeness_score=0.475484,
        search_completeness_gap=0.364516,
        prior_support_strength=0.290543,
        support_richness=0.516905,
        ambiguity_pressure=0.708298,
        pending_challenger_mass=0.645042,
        best_pending_flip_probability=0.998118,
        frontier_recall_at_budget=0.0,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.218098,
        predicted_delta_certificate=0.244331,
        predicted_delta_margin=0.224355,
        predicted_delta_frontier=0.102738,
        metadata={
            "normalized_objective_gap": 0.072307,
            "normalized_mechanism_gap": 0.358536,
            "normalized_overlap_reduction": 0.904762,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_post_nonproductive_bridge_zero_signal_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.755669,
        config=cfg,
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_post_nonproductive_bridge_zero_signal_search_churn_stops_single_frontier_sampler_dead_end() -> None:
    cfg = VOIConfig(certificate_threshold=0.82)
    state = VOIControllerState(
        iteration_index=1,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.757409},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=2,
        stochastic_enabled=True,
        action_trace=[
            {
                "chosen_action": {
                    "kind": "increase_stochastic_samples",
                    "metadata": {
                        "evidence_discovery_bridge": True,
                        "uncertified_support_rich_zero_signal_bridge": True,
                    },
                },
                "realized_certificate_delta": -0.000417,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        ambiguity_context={
            "od_hard_case_prior": 0.72632,
            "ambiguity_budget_prior": 0.172,
            "od_ambiguity_support_ratio": 0.72632,
            "od_ambiguity_source_entropy": 0.78166,
            "refc_world_count": 1,
            "refc_requested_world_count": 128,
            "refc_sampler_requested_world_count": 0,
            "refc_world_count_policy": "single_frontier_shortcut",
        },
        certificate_margin=0.695581,
        search_completeness_score=0.773486,
        search_completeness_gap=0.066514,
        prior_support_strength=0.569489,
        support_richness=0.569489,
        ambiguity_pressure=0.397797,
        pending_challenger_mass=0.632488,
        best_pending_flip_probability=0.999718,
        corridor_family_recall=1.0,
        frontier_recall_at_budget=1.0,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.212729,
        predicted_delta_certificate=0.242591,
        predicted_delta_margin=0.215858,
        predicted_delta_frontier=0.088066,
        metadata={
            "normalized_objective_gap": 0.081717,
            "normalized_mechanism_gap": 0.640251,
            "normalized_overlap_reduction": 0.904762,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_post_nonproductive_bridge_zero_signal_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.757409,
        config=cfg,
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_post_harmful_evidence_drift_search_churn_stops_near_threshold_supported_row() -> None:
    cfg = VOIConfig(certificate_threshold=0.84)
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 9.96, 10.03)},
            {"route_id": "route_c", "objective_vector": (10.11, 10.04, 10.06)},
        ],
        certificate={"route_a": 0.784141, "route_b": 0.215859},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=4,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "increase_stochastic_samples"},
                "realized_certificate_delta": -0.010731,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_changed": False,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": -0.018141,
                "realized_productive": True,
            }
        ],
        ambiguity_context={
            "od_hard_case_prior": 0.492942,
            "ambiguity_budget_prior": 0.423843,
            "od_ambiguity_support_ratio": 0.686842,
            "od_ambiguity_source_entropy": 0.836641,
        },
        near_tie_mass=0.0,
        top_refresh_gain=0.295154,
        top_fragility_mass=0.784141,
        competitor_pressure=1.0,
        support_richness=0.650912,
        ambiguity_pressure=0.641784,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.126987,
        predicted_delta_certificate=0.215859,
        predicted_delta_margin=0.215817,
        predicted_delta_frontier=0.079328,
        metadata={
            "normalized_objective_gap": 0.043569,
            "normalized_mechanism_gap": 0.148094,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_predicted_before_cap": 0.40371,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:terrain",
        kind="refresh_top1_vor",
        target="terrain",
        q_score=0.047,
        predicted_delta_certificate=0.062,
        predicted_delta_margin=0.04,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_delta": -0.492308,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.254434,
        predicted_delta_certificate=0.104581,
        predicted_delta_margin=0.039207,
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_post_harmful_evidence_drift_search_churn(
        [refine, refresh, resample, stop],
        state=state,
        current_certificate=0.784141,
        config=cfg,
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_uncertified_weak_search_tail_stops_low_mechanism_refine() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.794388},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.173791,
        predicted_delta_certificate=0.205612,
        predicted_delta_margin=0.190053,
        predicted_delta_frontier=0.019404,
        metadata={
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.073179,
            "normalized_overlap_reduction": 0.909091,
        },
    )

    assert voi_module._should_stop_uncertified_weak_search_tail(
        refine,
        state=state,
        current_certificate=0.794388,
        config=VOIConfig(certificate_threshold=0.82),
        evidence_uncertainty=False,
    ) is True


def test_uncertified_weak_search_tail_stops_single_frontier_world_saturation_row() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.756973},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=2,
        ambiguity_context={
            "refc_world_count": 1,
            "refc_requested_world_count": 1,
            "refc_sampler_requested_world_count": 1,
            "refc_world_count_policy": "single_frontier_full_stress",
        },
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.217034,
        predicted_delta_certificate=0.243027,
        predicted_delta_margin=0.21658,
        predicted_delta_frontier=0.113819,
        metadata={
            "normalized_objective_gap": 0.107226,
            "normalized_mechanism_gap": 0.190531,
            "normalized_overlap_reduction": 0.857143,
        },
    )

    assert voi_module._should_stop_uncertified_weak_search_tail(
        refine,
        state=state,
        current_certificate=0.756973,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=False,
    ) is True


def test_uncertified_weak_search_tail_keeps_single_frontier_world_saturation_with_material_signal() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.756973},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=2,
        ambiguity_context={
            "refc_world_count": 1,
            "refc_requested_world_count": 1,
            "refc_sampler_requested_world_count": 1,
            "refc_world_count_policy": "single_frontier_full_stress",
        },
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.301,
        predicted_delta_certificate=0.243027,
        predicted_delta_margin=0.249,
        predicted_delta_frontier=0.151,
        metadata={
            "normalized_objective_gap": 0.142,
            "normalized_mechanism_gap": 0.190531,
            "normalized_overlap_reduction": 0.857143,
        },
    )

    assert voi_module._should_stop_uncertified_weak_search_tail(
        refine,
        state=state,
        current_certificate=0.756973,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=False,
    ) is False


def test_uncertified_weak_search_tail_stops_observed_full_coverage_structural_cap_tail() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.775799},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=1,
        ambiguity_context={
            "single_frontier_certificate_cap_applied": True,
            "single_frontier_requires_full_stress": True,
            "empirical_baseline_certificate": 1.0,
            "controller_baseline_certificate": 0.775799,
            "refc_world_count": 120,
            "refc_unique_world_count": 120,
            "refc_requested_world_count": 120,
            "refc_sampler_requested_world_count": 120,
            "refc_world_count_policy": "single_frontier_full_stress",
        },
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.000344,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.205738,
        predicted_delta_certificate=0.224201,
        predicted_delta_margin=0.21658,
        predicted_delta_frontier=0.113819,
        metadata={
            "normalized_objective_gap": 0.107226,
            "normalized_mechanism_gap": 0.190531,
            "normalized_overlap_reduction": 0.857143,
            "mean_flip_probability": 0.997643,
        },
    )

    assert voi_module._should_stop_uncertified_weak_search_tail(
        refine,
        state=state,
        current_certificate=0.775799,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=False,
    ) is True


def test_uncertified_weak_search_tail_keeps_high_mechanism_refine() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.805054},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=4,
        remaining_evidence_budget=3,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.166558,
        predicted_delta_certificate=0.194946,
        predicted_delta_margin=0.188002,
        predicted_delta_frontier=0.017264,
        metadata={
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.800338,
            "normalized_overlap_reduction": 0.9,
        },
    )

    assert voi_module._should_stop_uncertified_weak_search_tail(
        refine,
        state=state,
        current_certificate=0.805054,
        config=VOIConfig(certificate_threshold=0.82),
        evidence_uncertainty=False,
    ) is False


def test_uncertified_weak_search_tail_stops_moderate_single_frontier_refine() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.769002},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.210098,
        predicted_delta_certificate=0.230998,
        predicted_delta_margin=0.224355,
        predicted_delta_frontier=0.102738,
        metadata={
            "normalized_objective_gap": 0.072307,
            "normalized_mechanism_gap": 0.358536,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert voi_module._should_stop_uncertified_weak_search_tail(
        refine,
        state=state,
        current_certificate=0.769002,
        config=VOIConfig(certificate_threshold=0.81),
        evidence_uncertainty=False,
    ) is True


def test_uncertified_weak_search_tail_can_stop_prior_only_evidence_uncertainty() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.769002},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.210098,
        predicted_delta_certificate=0.230998,
        predicted_delta_margin=0.224355,
        predicted_delta_frontier=0.102738,
        metadata={
            "normalized_objective_gap": 0.072307,
            "normalized_mechanism_gap": 0.358536,
            "normalized_overlap_reduction": 0.904762,
        },
    )

    assert voi_module._should_stop_uncertified_weak_search_tail(
        refine,
        state=state,
        current_certificate=0.769002,
        config=VOIConfig(certificate_threshold=0.81),
        evidence_uncertainty=True,
    ) is True


def test_uncertified_weak_search_tail_build_action_menu_prefers_stop() -> None:
    dccs = select_candidates(
        [
            _candidate(
                "cand_tail",
                path=["c1", "c2", "c3"],
                objective=(10.01, 10.0, 10.0),
                road_mix={"motorway_share": 0.49, "a_road_share": 0.31, "urban_share": 0.20},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=8.5,
                mechanism={"motorway_share": 0.49, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2", "f3"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.50, "a_road_share": 0.30, "urban_share": 0.20},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=8.5,
                mechanism={"motorway_share": 0.50, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=1),
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={"ranking": [], "top_refresh_gain": 0.0},
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )

    actions = build_action_menu(
        VOIControllerState(
            iteration_index=0,
            frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
            certificate={"route_a": 0.794388},
            winner_id="route_a",
            selected_route_id="route_a",
            remaining_search_budget=1,
            remaining_evidence_budget=0,
            active_evidence_families=["scenario"],
            near_tie_mass=0.0,
            top_refresh_gain=0.0,
            top_fragility_mass=0.0,
        ),
        dccs=dccs,
        fragility=fragility,  # type: ignore[arg-type]
        config=VOIConfig(certificate_threshold=0.82),
    )

    assert [action.kind for action in actions] == ["stop"]


def test_certified_support_rich_zero_refresh_signal_does_not_bridge_when_world_sampler_saturated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = _support_rich_zero_signal_bridge_state(
        stochastic_enabled=False,
        actual_world_count=86,
        requested_world_count=96,
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise
    original_decision_movement = voi_module._search_action_shows_strong_decision_movement

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    def _suppress_decision_movement(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_decision_movement(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)
    monkeypatch.setattr(voi_module, "_search_action_shows_strong_decision_movement", _suppress_decision_movement)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "increase_stochastic_samples" not in kinds
    assert "refresh_top1_vor" not in kinds
    assert "refine_top1_dccs" not in kinds
    assert "refine_topk_dccs" not in kinds
    assert actions[0].kind == "stop"


def test_certified_support_rich_no_near_tie_prefers_refresh_over_stress_only_resample() -> None:
    dccs = _dccs_result()
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "scenario", "vor": 0.01}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.01,
        },
        route_fragility_map={"route_a": {"scenario": 0.02}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"scenario": 0.2}}},
    )
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.44,
            "od_hard_case_prior": 0.55,
            "od_engine_disagreement_prior": 0.24,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.43,
            "refc_stress_world_fraction": 0.20,
            "is_hard_case": True,
        },
        certificate_margin=0.01,
        search_completeness_score=0.96,
        search_completeness_gap=0.0,
        prior_support_strength=0.79,
        pending_challenger_mass=0.08,
        best_pending_flip_probability=0.10,
        frontier_recall_at_budget=0.98,
        top_refresh_gain=0.01,
        top_fragility_mass=0.02,
        competitor_pressure=0.20,
        near_tie_mass=0.0,
    )

    actions = build_action_menu(
        state,
        dccs=dccs,
        fragility=fragility,  # type: ignore[arg-type]
        config=VOIConfig(certificate_threshold=0.80),
    )

    kinds = [action.kind for action in actions]
    assert "refresh_top1_vor" in kinds
    assert "increase_stochastic_samples" not in kinds
    assert actions[0].kind == "refresh_top1_vor"


def test_certified_support_rich_real_near_tie_keeps_resample_available_and_can_win() -> None:
    dccs = _dccs_result()
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "scenario", "vor": 0.001}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.001,
        },
        route_fragility_map={"route_a": {"scenario": 0.01}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"scenario": 0.1}}},
    )
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.44,
            "od_hard_case_prior": 0.55,
            "od_engine_disagreement_prior": 0.24,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.43,
            "refc_stress_world_fraction": 0.20,
            "is_hard_case": True,
        },
        certificate_margin=0.01,
        search_completeness_score=0.96,
        search_completeness_gap=0.0,
        prior_support_strength=0.79,
        pending_challenger_mass=0.08,
        best_pending_flip_probability=0.10,
        frontier_recall_at_budget=0.98,
        top_refresh_gain=0.001,
        top_fragility_mass=0.01,
        competitor_pressure=0.10,
        near_tie_mass=0.16,
    )

    actions = build_action_menu(
        state,
        dccs=dccs,
        fragility=fragility,  # type: ignore[arg-type]
        config=VOIConfig(certificate_threshold=0.80),
    )

    resample = next(action for action in actions if action.kind == "increase_stochastic_samples")
    assert "near_tie_set_nonempty" in resample.preconditions
    assert actions[0].kind == "increase_stochastic_samples"


def test_support_rich_without_winner_side_refc_signal_does_not_suppress_resample() -> None:
    dccs = _dccs_result()
    fragility = types.SimpleNamespace(
        value_of_refresh={"ranking": [], "top_refresh_gain": 0.0},
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.44,
            "od_hard_case_prior": 0.55,
            "od_engine_disagreement_prior": 0.24,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.43,
            "refc_stress_world_fraction": 0.20,
            "is_hard_case": True,
        },
        certificate_margin=0.01,
        search_completeness_score=0.96,
        search_completeness_gap=0.0,
        prior_support_strength=0.79,
        pending_challenger_mass=0.08,
        best_pending_flip_probability=0.10,
        frontier_recall_at_budget=0.98,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.10,
        near_tie_mass=0.0,
    )

    actions = build_action_menu(
        state,
        dccs=dccs,
        fragility=fragility,  # type: ignore[arg-type]
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert any(action.kind == "increase_stochastic_samples" for action in actions)


def test_refresh_action_is_not_reoffered_once_family_has_been_refreshed() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.4},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        refreshed_evidence_families=["scenario"],
        near_tie_mass=0.2,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig())

    assert all(action.kind != "refresh_top1_vor" for action in actions)


def test_search_completeness_metrics_flag_pending_challengers() -> None:
    dccs = _dccs_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.92, "route_b": 0.15},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=0,
        action_trace=[],
        ambiguity_context={
            "od_ambiguity_index": 0.45,
            "od_ambiguity_confidence": 0.9,
            "od_corridor_family_count": 3,
        },
    )

    metrics = compute_search_completeness_metrics(
        state,
        dccs=dccs,
        config=VOIConfig(search_completeness_threshold=0.90),
    )

    assert 0.0 <= metrics["search_completeness_score"] < 0.90
    assert metrics["search_completeness_gap"] > 0.0
    assert metrics["best_pending_flip_probability"] > 0.0
    assert metrics["frontier_recall_at_budget"] < 1.0


def test_supported_uncertainty_gate_stops_certified_rows_with_lightly_supported_prior() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=0,
        action_trace=[],
        ambiguity_context={
            "od_ambiguity_index": 0.12,
            "od_ambiguity_confidence": 0.35,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix": "historical_results_bootstrap",
        },
        certificate_margin=0.18,
        search_completeness_score=0.61,
        search_completeness_gap=0.23,
        prior_support_strength=0.12,
        pending_challenger_mass=0.55,
        best_pending_flip_probability=0.91,
        frontier_recall_at_budget=0.95,
        near_tie_mass=0.0,
    )

    assert (
        credible_search_uncertainty(
            state,
            config=VOIConfig(certificate_threshold=0.84),
            current_certificate=1.0,
        )
        is False
    )


def test_action_menu_suppresses_refine_actions_when_certified_row_lacks_credible_search_uncertainty() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.12,
            "od_ambiguity_confidence": 0.35,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix": "historical_results_bootstrap",
        },
        certificate_margin=0.18,
        search_completeness_score=0.61,
        search_completeness_gap=0.23,
        prior_support_strength=0.12,
        pending_challenger_mass=0.55,
        best_pending_flip_probability=0.91,
        frontier_recall_at_budget=0.95,
        near_tie_mass=0.0,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.84))

    assert all(action.kind not in {"refine_top1_dccs", "refine_topk_dccs"} for action in actions)


def test_certified_rows_with_only_prior_driven_uncertainty_do_not_reopen_search() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 9.8, 10.1)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.42,
            "od_engine_disagreement_prior": 0.57,
            "od_hard_case_prior": 0.59,
            "od_ambiguity_confidence": 0.75,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix": "historical_results_bootstrap",
        },
        certificate_margin=0.20,
        search_completeness_score=0.636801,
        search_completeness_gap=0.203199,
        prior_support_strength=0.43309,
        pending_challenger_mass=0.590834,
        best_pending_flip_probability=0.998771,
        frontier_recall_at_budget=0.460526,
        near_tie_mass=0.0,
    )
    metrics = compute_search_completeness_metrics(
        state,
        dccs=dccs,
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert metrics["support_richness"] < 0.35
    assert (
        credible_search_uncertainty(
            state,
            config=VOIConfig(certificate_threshold=0.80),
            current_certificate=1.0,
        )
        is False
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    assert all(action.kind not in {"refine_top1_dccs", "refine_topk_dccs"} for action in actions)


def test_certified_underfilled_frontier_with_winner_side_signal_reopens_search() -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "per_family_certificate": {"terrain": 0.85},
            "ranking": [{"family": "terrain", "vor": 0.15}],
            "top_refresh_family": "terrain",
            "top_refresh_gain": 0.15,
        },
        route_fragility_map={"route_a": {"terrain": 0.12}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"terrain": 1.0}}},
    )
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 9.8, 10.1)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["terrain"],
        ambiguity_context={
            "od_ambiguity_index": 0.04,
            "od_hard_case_prior": 0.33,
            "od_engine_disagreement_prior": 0.31,
            "od_ambiguity_confidence": 0.72,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":1,"routing_graph_probe":1}',
            "od_ambiguity_support_ratio": 0.59,
            "od_ambiguity_prior_strength": 0.04,
            "od_ambiguity_source_entropy": 0.72,
            "ambiguity_budget_prior": 0.04,
            "refc_stress_world_fraction": 0.02,
        },
        certificate_margin=0.18,
        search_completeness_score=0.61,
        search_completeness_gap=0.23,
        prior_support_strength=0.27,
        support_richness=0.27,
        ambiguity_pressure=0.23,
        pending_challenger_mass=0.28,
        best_pending_flip_probability=0.74,
        corridor_family_recall=0.67,
        frontier_recall_at_budget=0.67,
        near_tie_mass=0.0,
        top_refresh_gain=0.15,
        top_fragility_mass=0.12,
        competitor_pressure=0.16,
    )

    assert (
        credible_search_uncertainty(
            state,
            config=VOIConfig(certificate_threshold=0.80),
            current_certificate=1.0,
        )
        is True
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    assert any(action.kind in {"refine_top1_dccs", "refine_topk_dccs"} for action in actions)


def test_certified_underfilled_frontier_bridge_requires_winner_side_signal() -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "per_family_certificate": {},
            "ranking": [],
            "top_refresh_family": None,
            "top_refresh_gain": 0.0,
        },
        route_fragility_map={"route_a": {}},
        competitor_fragility_breakdown={"route_a": {}},
    )
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 9.8, 10.1)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["terrain"],
        ambiguity_context={
            "od_ambiguity_index": 0.04,
            "od_hard_case_prior": 0.33,
            "od_engine_disagreement_prior": 0.31,
            "od_ambiguity_confidence": 0.72,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":1,"routing_graph_probe":1}',
            "od_ambiguity_support_ratio": 0.59,
            "od_ambiguity_prior_strength": 0.04,
            "od_ambiguity_source_entropy": 0.72,
            "ambiguity_budget_prior": 0.04,
            "refc_stress_world_fraction": 0.02,
        },
        certificate_margin=0.18,
        search_completeness_score=0.61,
        search_completeness_gap=0.23,
        prior_support_strength=0.27,
        support_richness=0.27,
        ambiguity_pressure=0.23,
        pending_challenger_mass=0.28,
        best_pending_flip_probability=0.74,
        corridor_family_recall=0.67,
        frontier_recall_at_budget=0.67,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )

    assert (
        credible_search_uncertainty(
            state,
            config=VOIConfig(certificate_threshold=0.80),
            current_certificate=1.0,
        )
        is False
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    assert all(action.kind not in {"refine_top1_dccs", "refine_topk_dccs"} for action in actions)


def test_support_rich_hard_case_reopens_search_on_certified_rows() -> None:
    dccs = select_candidates(
        [
            _candidate(
                "cand_live",
                path=["c1", "c2", "c3"],
                objective=(9.0, 9.15, 9.05),
                road_mix={"motorway_share": 0.55, "a_road_share": 0.30, "urban_share": 0.15},
                toll_share=0.04,
                terrain_burden=0.06,
                straight_line_km=8.8,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            ),
            _candidate(
                "cand_peer",
                path=["p1", "p2", "p3", "p4"],
                objective=(11.8, 11.6, 11.7),
                road_mix={"motorway_share": 0.40, "a_road_share": 0.35, "urban_share": 0.25},
                toll_share=0.12,
                terrain_burden=0.15,
                straight_line_km=8.7,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            ),
            _candidate(
                "cand_extra",
                path=["e1", "e2", "e3"],
                objective=(10.8, 10.5, 10.7),
                road_mix={"motorway_share": 0.45, "a_road_share": 0.35, "urban_share": 0.20},
                toll_share=0.08,
                terrain_burden=0.08,
                straight_line_km=8.9,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            ),
            _candidate(
                "cand_more",
                path=["m1", "m2", "m3"],
                objective=(10.4, 10.2, 10.3),
                road_mix={"motorway_share": 0.50, "a_road_share": 0.28, "urban_share": 0.22},
                toll_share=0.06,
                terrain_burden=0.07,
                straight_line_km=8.85,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            ),
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.48, "toll_share": 0.06, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=1),
    )
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.06, 9.96, 10.02)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.74,
            "od_hard_case_prior": 0.76,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.36,
            "od_nominal_margin_proxy": 0.08,
            "od_ambiguity_support_ratio": 0.86,
            "od_ambiguity_prior_strength": 0.74,
            "od_ambiguity_source_entropy": 0.81,
        },
        certificate_margin=0.0,
        search_completeness_score=0.83,
        search_completeness_gap=0.13,
        prior_support_strength=0.81,
        pending_challenger_mass=0.34,
        best_pending_flip_probability=0.44,
        frontier_recall_at_budget=0.74,
        near_tie_mass=0.19,
        top_refresh_gain=0.0,
        top_fragility_mass=0.161017,
        competitor_pressure=1.0,
    )

    assert (
        credible_evidence_uncertainty(
            state,
            fragility=fragility,
            config=VOIConfig(certificate_threshold=0.80),
            current_certificate=1.0,
        )
        is True
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]
    refresh = next(action for action in actions if action.kind == "refresh_top1_vor")
    refine = next(action for action in actions if action.kind == "refine_top1_dccs")
    refine_topk = next(action for action in actions if action.kind == "refine_topk_dccs")
    assert refresh.q_score > refine.q_score
    assert refresh.q_score > refine_topk.q_score
    assert actions[0].kind == "refresh_top1_vor"
    assert "refresh_top1_vor" in kinds
    assert "refine_top1_dccs" in kinds
    assert "refine_topk_dccs" in kinds
    assert refine.metadata["search_completeness_bonus"] == 0.0
    assert refine.metadata["certified_supported_hard_case_penalized"] is True
    assert refine.metadata["certified_supported_hard_case_penalty"] > 0.0
    assert refine_topk.metadata["certified_supported_hard_case_penalized"] is True
    assert refine_topk.metadata["certified_supported_hard_case_penalty"] > 0.0


def test_certified_support_rich_no_gain_refresh_suppresses_refine_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _fragility_result()
    state = _support_rich_certified_reopen_state(
        action_trace=[
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "refresh_top1_vor" in kinds
    assert "refine_top1_dccs" not in kinds
    assert "refine_topk_dccs" not in kinds


def test_certified_support_rich_no_gain_resample_suppresses_refine_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _fragility_result()
    state = _support_rich_certified_reopen_state(
        action_trace=[
            {
                "chosen_action": {"kind": "increase_stochastic_samples"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "refresh_top1_vor" in kinds
    assert "refine_top1_dccs" not in kinds
    assert "refine_topk_dccs" not in kinds


def test_failed_bridge_resample_blocks_zero_signal_refresh_and_generic_refine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = _support_rich_zero_signal_bridge_state(
        action_trace=[
            {
                "chosen_action": {
                    "kind": "increase_stochastic_samples",
                    "metadata": {"evidence_discovery_bridge": True},
                },
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        stochastic_enabled=False,
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "refresh_top1_vor" not in kinds
    assert "refine_top1_dccs" not in kinds
    assert "refine_topk_dccs" not in kinds
    assert actions[0].kind == "stop"


def test_failed_bridge_resample_keeps_genuinely_novel_refine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = _support_rich_zero_signal_bridge_state(
        action_trace=[
            {
                "chosen_action": {
                    "kind": "increase_stochastic_samples",
                    "metadata": {"evidence_discovery_bridge": True},
                },
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        stochastic_enabled=False,
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise

    def _force_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return True
        return original_novelty(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _force_novelty)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "refresh_top1_vor" not in kinds
    assert "refine_top1_dccs" in kinds


def test_failed_bridge_resample_keeps_materially_strong_refine_without_novelty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = _support_rich_zero_signal_bridge_state(
        action_trace=[
            {
                "chosen_action": {
                    "kind": "increase_stochastic_samples",
                    "metadata": {"evidence_discovery_bridge": True},
                },
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        stochastic_enabled=False,
        actual_world_count=86,
        requested_world_count=96,
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise
    original_decision_movement = voi_module._search_action_shows_strong_decision_movement

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    def _force_decision_movement(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return True
        return original_decision_movement(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)
    monkeypatch.setattr(voi_module, "_search_action_shows_strong_decision_movement", _force_decision_movement)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "refresh_top1_vor" not in kinds
    assert "refine_top1_dccs" in kinds


def test_certified_zero_signal_low_coverage_weak_structural_reopen_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = replace(
        _support_rich_zero_signal_bridge_state(
            stochastic_enabled=True,
            requested_world_count=64,
            actual_world_count=64,
            unique_world_count=48,
        ),
        certificate_margin=1.0,
        search_completeness_score=0.557007,
        search_completeness_gap=0.282993,
        prior_support_strength=0.660869,
        support_richness=0.660869,
        ambiguity_pressure=0.602619,
        pending_challenger_mass=0.605033,
        best_pending_flip_probability=0.995172,
        corridor_family_recall=0.363636,
        frontier_recall_at_budget=0.367738,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        cost_search=1,
        predicted_delta_certificate=0.3765479670384516,
        predicted_delta_margin=0.20530576648127485,
        predicted_delta_frontier=0.08268246021216327,
        q_score=0.28965759058555657,
        feasible=True,
        preconditions=("search_budget_available", "dccs_candidate_available"),
        reason="refine_candidate",
        metadata={
            "normalized_objective_gap": 0.06441033679776109,
            "normalized_mechanism_gap": 0.0,
            "mean_flip_probability": 0.994665466279556,
            "normalized_overlap_reduction": 0.8571428571428572,
        },
    )
    stop = voi_module.VOIAction(
        action_id="stop",
        kind="stop",
        target="stop",
        feasible=True,
        preconditions=("always",),
        reason="stop",
    )

    monkeypatch.setattr(
        voi_module,
        "_search_action_shows_strong_decision_movement",
        lambda action, *, state: bool(action and action.kind.startswith("refine")),
    )
    monkeypatch.setattr(
        voi_module,
        "_refine_action_has_genuine_novel_search_promise",
        lambda action, *, state: False,
    )

    filtered = voi_module._suppress_certified_zero_signal_controller_churn(
        [refine, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.82),
        evidence_discovery_bridge=False,
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_certified_repeated_no_gain_refine_loop_stops_instead_of_churning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = replace(
        _support_rich_zero_signal_bridge_state(
            action_trace=[
                {
                    "chosen_action": {"kind": "refine_top1_dccs"},
                    "realized_certificate_delta": 0.0,
                    "realized_frontier_gain": 0.0,
                    "realized_selected_route_improvement": 0.0,
                    "realized_runner_up_gap_delta": 0.0,
                    "realized_evidence_uncertainty_delta": 0.0,
                    "realized_productive": False,
                }
            ],
            stochastic_enabled=False,
            actual_world_count=96,
            requested_world_count=96,
        ),
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        near_tie_mass=0.0,
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise
    original_decision_movement = voi_module._search_action_shows_strong_decision_movement

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    def _suppress_decision_movement(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_decision_movement(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)
    monkeypatch.setattr(voi_module, "_search_action_shows_strong_decision_movement", _suppress_decision_movement)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "refine_top1_dccs" not in kinds
    assert "refine_topk_dccs" not in kinds
    assert actions[0].kind == "stop"


def test_certified_repeated_no_gain_refine_loop_keeps_novel_search_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _support_rich_zero_signal_fragility_result()
    state = replace(
        _support_rich_zero_signal_bridge_state(
            action_trace=[
                {
                    "chosen_action": {"kind": "refine_top1_dccs"},
                    "realized_certificate_delta": 0.0,
                    "realized_frontier_gain": 0.0,
                    "realized_selected_route_improvement": 0.0,
                    "realized_runner_up_gap_delta": 0.0,
                    "realized_evidence_uncertainty_delta": 0.0,
                    "realized_productive": False,
                }
            ],
            stochastic_enabled=False,
            actual_world_count=96,
            requested_world_count=96,
        ),
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        near_tie_mass=0.0,
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise
    original_decision_movement = voi_module._search_action_shows_strong_decision_movement

    def _force_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return True
        return original_novelty(action, state=state)

    def _suppress_decision_movement(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_decision_movement(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _force_novelty)
    monkeypatch.setattr(voi_module, "_search_action_shows_strong_decision_movement", _suppress_decision_movement)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "refine_top1_dccs" in kinds or "refine_topk_dccs" in kinds


def test_evidence_exhausted_certified_search_tail_stops_low_ranked_refine() -> None:
    state = replace(
        _support_rich_certified_reopen_state(
            action_trace=[
                {
                    "chosen_action": {"kind": "refresh_top1_vor"},
                    "realized_certificate_delta": 0.22,
                    "realized_frontier_gain": 0.0,
                    "realized_selected_route_improvement": 0.0,
                    "realized_runner_up_gap_delta": 0.001,
                    "realized_evidence_uncertainty_delta": -0.21,
                    "realized_productive": True,
                }
            ],
        ),
        remaining_evidence_budget=0,
        near_tie_mass=0.0,
        support_richness=0.81,
        top_refresh_gain=0.725714,
        top_fragility_mass=0.725714,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        cost_search=1,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.1169743174148762,
        predicted_delta_frontier=0.1139657935371904,
        q_score=0.08573428374365755,
        feasible=True,
        preconditions=("search_budget_available", "dccs_candidate_available"),
        reason="refine_candidate",
        metadata={
            "normalized_objective_gap": 0.0566540442362925,
            "normalized_mechanism_gap": 0.0,
            "mean_flip_probability": 0.9999375857542858,
            "normalized_overlap_reduction": 0.85,
        },
    )

    assert voi_module._should_stop_evidence_exhausted_certified_search_tail(
        refine,
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.80),
    ) is True


def test_evidence_exhausted_certified_search_tail_keeps_higher_ranked_refine() -> None:
    state = replace(
        _support_rich_certified_reopen_state(
            action_trace=[
                {
                    "chosen_action": {"kind": "refresh_top1_vor"},
                    "realized_certificate_delta": 0.22,
                    "realized_frontier_gain": 0.0,
                    "realized_selected_route_improvement": 0.0,
                    "realized_runner_up_gap_delta": 0.001,
                    "realized_evidence_uncertainty_delta": -0.21,
                    "realized_productive": True,
                }
            ],
        ),
        remaining_evidence_budget=0,
        near_tie_mass=0.0,
        support_richness=0.81,
        top_refresh_gain=0.725714,
        top_fragility_mass=0.725714,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        cost_search=1,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.1169743174148762,
        predicted_delta_frontier=0.1139657935371904,
        q_score=0.14573428374365757,
        feasible=True,
        preconditions=("search_budget_available", "dccs_candidate_available"),
        reason="refine_candidate",
        metadata={
            "normalized_objective_gap": 0.0566540442362925,
            "normalized_mechanism_gap": 0.0,
            "mean_flip_probability": 0.9999375857542858,
            "normalized_overlap_reduction": 0.85,
        },
    )

    assert voi_module._should_stop_evidence_exhausted_certified_search_tail(
        refine,
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.80),
    ) is False


def test_evidence_exhausted_certified_search_tail_stops_headroom_capped_post_refresh_refine() -> None:
    state = VOIControllerState(
        iteration_index=2,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.1, 10.1, 10.1)},
        ],
        certificate={"route_a": 0.886228, "route_b": 0.113772},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=0,
        action_trace=[
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": -0.155689,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": -0.36527,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": 0.041916,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": -0.083832,
                "realized_productive": True,
            },
        ],
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.539459,
            "od_ambiguity_source_entropy": 1.0,
            "od_hard_case_prior": 0.484062,
            "ambiguity_budget_prior": 0.38083,
            "refc_stress_world_fraction": 0.568862,
        },
        certificate_margin=0.076228,
        support_richness=0.652031,
        prior_support_strength=0.652031,
        search_completeness_score=0.465992,
        search_completeness_gap=0.374008,
        pending_challenger_mass=0.668473,
        best_pending_flip_probability=0.999211,
        frontier_recall_at_budget=0.135129,
        top_refresh_gain=0.694611,
        top_fragility_mass=0.57485,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        cost_search=1,
        predicted_delta_certificate=0.1137724551,
        predicted_delta_margin=0.2223982188,
        predicted_delta_frontier=0.1014500925,
        q_score=0.1390805415,
        feasible=True,
        preconditions=("search_budget_available", "dccs_candidate_available"),
        reason="refine_candidate",
        metadata={
            "normalized_objective_gap": 0.1014500925,
            "normalized_mechanism_gap": 0.3581777345,
            "mean_flip_probability": 0.9992109369,
            "normalized_overlap_reduction": 0.9090909091,
            "certificate_headroom_cap_applied": True,
        },
    )

    assert voi_module._should_stop_evidence_exhausted_certified_search_tail(
        refine,
        state=state,
        current_certificate=0.886228,
        config=VOIConfig(certificate_threshold=0.81),
    ) is True


def test_strong_live_evidence_refresh_survives_certified_churn_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _supported_ambiguity_fragility_result()
    state = _support_rich_certified_reopen_state(
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise
    original_decision_movement = voi_module._search_action_shows_strong_decision_movement

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    def _suppress_decision_movement(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_decision_movement(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)
    monkeypatch.setattr(voi_module, "_search_action_shows_strong_decision_movement", _suppress_decision_movement)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "refresh_top1_vor" in kinds
    assert actions[0].kind == "refresh_top1_vor"


def test_post_evidence_certified_search_backslide_keeps_refresh_but_removes_search_tail() -> None:
    state = replace(
        _support_rich_certified_reopen_state(
            action_trace=[
                {
                    "chosen_action": {"kind": "refresh_top1_vor"},
                    "realized_certificate_delta": 0.12,
                    "realized_frontier_gain": 0.0,
                    "realized_selected_route_improvement": 0.0,
                    "realized_runner_up_gap_delta": 0.001,
                    "realized_evidence_uncertainty_delta": -0.16,
                    "realized_productive": True,
                }
            ],
        ),
        remaining_evidence_budget=1,
        near_tie_mass=0.0,
        top_refresh_gain=0.41,
        top_fragility_mass=0.37,
        competitor_pressure=0.92,
    )
    actions = [
        voi_module.VOIAction(
            action_id="refresh_top1_vor:scenario",
            kind="refresh_top1_vor",
            target="scenario",
            q_score=0.21,
            cost_evidence=1,
            predicted_delta_certificate=0.0,
            predicted_delta_margin=0.04,
            predicted_delta_frontier=0.0,
            feasible=True,
            preconditions=("evidence_budget_available",),
            reason="refresh",
            metadata={
                "structured_refresh_signal": True,
                "empirical_refresh_certificate_uplift": 0.11,
            },
        ),
        voi_module.VOIAction(
            action_id="refine_top1_dccs:test",
            kind="refine_top1_dccs",
            target="test",
            q_score=0.08,
            cost_search=1,
            predicted_delta_certificate=0.0,
            predicted_delta_margin=0.01,
            predicted_delta_frontier=0.0,
            feasible=True,
            preconditions=("search_budget_available",),
            reason="refine_candidate",
            metadata={
                "normalized_objective_gap": 0.01,
                "normalized_mechanism_gap": 0.0,
                "normalized_overlap_reduction": 0.0,
                "mean_flip_probability": 0.12,
            },
        ),
        voi_module.VOIAction(
            action_id="stop",
            kind="stop",
            target="stop",
            q_score=0.0,
            feasible=True,
            preconditions=("always",),
            reason="stop",
        ),
    ]

    filtered = voi_module._suppress_post_evidence_certified_search_backslide(
        actions,
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert [action.kind for action in filtered] == ["refresh_top1_vor", "stop"]


def test_certified_support_rich_no_gain_refresh_keeps_genuinely_novel_refine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = _fragility_result()
    state = _support_rich_certified_reopen_state(
        action_trace=[
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise

    def _force_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return True
        return original_novelty(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _force_novelty)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "refine_top1_dccs" in kinds


def test_certified_support_rich_zero_signal_bridge_requires_expandable_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "baseline_margin_summary": {"world_count": 80.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
            "fragility_stress_state": "severely_stale",
            "per_family_certificate": {"scenario": 1.0},
            "per_family_margin_summary": {
                "scenario": {"world_count": 80.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0}
            },
            "ranking": [{"family": "scenario", "vor": 0.0}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.0,
        },
        route_fragility_map={"route_a": {"scenario": 0.0}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"scenario": 0}}},
    )
    state = _support_rich_zero_signal_bridge_state(
        requested_world_count=80,
        actual_world_count=80,
        unique_world_count=80,
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise
    original_movement = voi_module._search_action_shows_strong_decision_movement

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    def _suppress_movement(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_movement(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)
    monkeypatch.setattr(voi_module, "_search_action_shows_strong_decision_movement", _suppress_movement)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert "increase_stochastic_samples" not in kinds
    assert "refresh_top1_vor" not in kinds
    assert actions[0].kind == "stop"


def test_failed_bridge_resample_keeps_strong_decision_refine_even_without_novelty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dccs = _support_rich_certified_reopen_dccs()
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "baseline_margin_summary": {"world_count": 56.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
            "fragility_stress_state": "severely_stale",
            "per_family_certificate": {"scenario": 1.0},
            "per_family_margin_summary": {
                "scenario": {"world_count": 56.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0}
            },
            "ranking": [{"family": "scenario", "vor": 0.0}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.0,
        },
        route_fragility_map={"route_a": {"scenario": 0.0}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"scenario": 0}}},
    )
    state = _support_rich_zero_signal_bridge_state(
        action_trace=[
            {
                "chosen_action": {
                    "kind": "increase_stochastic_samples",
                    "metadata": {"evidence_discovery_bridge": True},
                },
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
    )
    original_novelty = voi_module._refine_action_has_genuine_novel_search_promise
    original_movement = voi_module._search_action_shows_strong_decision_movement

    def _suppress_novelty(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return False
        return original_novelty(action, state=state)

    def _force_movement(action, *, state):
        if action is not None and action.kind in {"refine_top1_dccs", "refine_topk_dccs"}:
            return True
        return original_movement(action, state=state)

    monkeypatch.setattr(voi_module, "_refine_action_has_genuine_novel_search_promise", _suppress_novelty)
    monkeypatch.setattr(voi_module, "_search_action_shows_strong_decision_movement", _force_movement)

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]

    assert any(kind in {"refine_top1_dccs", "refine_topk_dccs"} for kind in kinds)
    assert actions[0].kind in {"refine_top1_dccs", "refine_topk_dccs"}


def test_certified_supported_evidence_hard_case_reopens_evidence_not_search() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.05, 10.02, 10.03)},
        ],
        certificate={"route_a": 0.83, "route_b": 0.79},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.68,
            "od_hard_case_prior": 0.72,
            "od_ambiguity_confidence": 0.91,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1}',
            "od_ambiguity_support_ratio": 0.79,
            "od_ambiguity_prior_strength": 0.68,
            "od_ambiguity_source_entropy": 0.63,
            "refc_stress_world_fraction": 0.34,
        },
        certificate_margin=0.04,
        search_completeness_score=0.91,
        search_completeness_gap=0.0,
        prior_support_strength=0.79,
        pending_challenger_mass=0.08,
        best_pending_flip_probability=0.11,
        frontier_recall_at_budget=0.96,
        top_refresh_gain=0.11,
        top_fragility_mass=0.30,
        competitor_pressure=0.45,
        near_tie_mass=0.12,
    )

    assert credible_search_uncertainty(state, config=VOIConfig(certificate_threshold=0.80), current_certificate=0.83) is False
    assert (
        credible_evidence_uncertainty(
            state,
            fragility=fragility,
            config=VOIConfig(certificate_threshold=0.80),
            current_certificate=0.83,
        )
        is True
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]
    assert "refresh_top1_vor" in kinds
    assert "increase_stochastic_samples" in kinds
    assert "refine_top1_dccs" not in kinds
    assert actions[0].kind == "refresh_top1_vor"


def test_support_rich_ambiguity_row_reopens_search_without_hard_case_label() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.03, 10.04, 10.01)},
        ],
        certificate={"route_a": 0.87, "route_b": 0.82},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.44,
            "od_hard_case_prior": 0.26,
            "od_engine_disagreement_prior": 0.24,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.24,
            "od_nominal_margin_proxy": 0.26,
            "od_ambiguity_support_ratio": 0.81,
            "od_ambiguity_prior_strength": 0.57,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.44,
            "refc_stress_world_fraction": 0.18,
        },
        certificate_margin=0.05,
        search_completeness_score=0.82,
        search_completeness_gap=0.12,
        prior_support_strength=0.78,
        pending_challenger_mass=0.29,
        best_pending_flip_probability=0.36,
        frontier_recall_at_budget=0.79,
        near_tie_mass=0.15,
        top_refresh_gain=0.04,
        top_fragility_mass=0.14,
        competitor_pressure=0.24,
    )

    metrics = compute_search_completeness_metrics(
        state,
        dccs=dccs,
        config=VOIConfig(certificate_threshold=0.80),
    )
    state = replace(state, **metrics)

    assert metrics["support_richness"] > 0.55
    assert metrics["ambiguity_pressure"] > 0.30
    assert credible_search_uncertainty(state, config=VOIConfig(certificate_threshold=0.80), current_certificate=0.87) is True

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]
    assert "refine_top1_dccs" in kinds


def test_support_rich_ambiguity_row_reopens_evidence_without_hard_case_label() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 0.84, "route_b": 0.80},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.43,
            "od_hard_case_prior": 0.24,
            "od_engine_disagreement_prior": 0.22,
            "od_ambiguity_confidence": 0.91,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.76,
            "od_ambiguity_prior_strength": 0.55,
            "od_ambiguity_source_entropy": 0.71,
            "ambiguity_budget_prior": 0.41,
            "refc_stress_world_fraction": 0.20,
        },
        certificate_margin=0.04,
        search_completeness_score=0.92,
        search_completeness_gap=0.0,
        prior_support_strength=0.75,
        pending_challenger_mass=0.06,
        best_pending_flip_probability=0.10,
        frontier_recall_at_budget=0.95,
        top_refresh_gain=0.10,
        top_fragility_mass=0.28,
        competitor_pressure=0.42,
        near_tie_mass=0.11,
    )

    assert credible_search_uncertainty(state, config=VOIConfig(certificate_threshold=0.80), current_certificate=0.84) is False
    assert (
        credible_evidence_uncertainty(
            state,
            fragility=fragility,
            config=VOIConfig(certificate_threshold=0.80),
            current_certificate=0.84,
        )
        is True
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]
    assert "refresh_top1_vor" in kinds
    assert "increase_stochastic_samples" in kinds
    assert "refine_top1_dccs" not in kinds
    assert actions[0].kind == "refresh_top1_vor"


def test_support_rich_ambiguity_prefers_refresh_after_no_gain_refine_streak() -> None:
    dccs = _dccs_result()
    fragility = replace(
        _supported_ambiguity_fragility_result(),
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "baseline_margin_summary": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
            "fragility_stress_state": "severely_stale",
            "per_family_certificate": {"scenario": 1.0, "weather": 0.0, "toll": 0.0},
            "per_family_margin_summary": {
                "scenario": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
                "weather": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
                "toll": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
            },
            "ranking": [{"family": "scenario", "vor": 0.0}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.0,
        },
    )
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        active_evidence_families=["scenario", "weather", "toll"],
        ambiguity_context={
            "od_ambiguity_index": 0.44,
            "od_hard_case_prior": 0.26,
            "od_engine_disagreement_prior": 0.24,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.43,
            "refc_stress_world_fraction": 0.20,
        },
        certificate_margin=0.0,
        search_completeness_score=0.91,
        search_completeness_gap=0.06,
        prior_support_strength=0.79,
        pending_challenger_mass=0.18,
        best_pending_flip_probability=0.22,
        frontier_recall_at_budget=0.86,
        top_refresh_gain=0.0,
        top_fragility_mass=0.161017,
        competitor_pressure=1.0,
        near_tie_mass=0.12,
    )

    assert (
        credible_evidence_uncertainty(
            state,
            fragility=fragility,
            config=VOIConfig(certificate_threshold=0.80),
            current_certificate=0.88,
        )
        is True
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    kinds = [action.kind for action in actions]
    assert actions[0].kind == "refresh_top1_vor"
    refine = next(action for action in actions if action.kind == "refine_top1_dccs")
    assert actions[0].q_score > refine.q_score
    assert "refresh_top1_vor" in kinds


def test_certified_support_rich_small_positive_refresh_signal_reorders_after_no_gain_refine() -> None:
    dccs = select_candidates(
        [
            _candidate(
                "cand_soft",
                path=["c1", "c2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=0),
    )
    fragility = replace(
        _supported_ambiguity_fragility_result(),
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "baseline_margin_summary": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
            "fragility_stress_state": "severely_stale",
            "per_family_certificate": {"scenario": 1.0, "weather": 0.0, "toll": 0.0},
            "per_family_margin_summary": {
                "scenario": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
                "weather": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
                "toll": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
            },
            "ranking": [{"family": "scenario", "vor": 0.0022}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.0022,
        },
        route_fragility_map={
            "route_a": {"scenario": 0.0024, "weather": 0.0, "toll": 0.0},
            "route_b": {"scenario": 0.0, "weather": 0.0, "toll": 0.0},
            "route_c": {"scenario": 0.0, "weather": 0.0, "toll": 0.0},
        },
    )
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        active_evidence_families=["scenario", "weather", "toll"],
        ambiguity_context={
            "od_ambiguity_index": 0.44,
            "od_hard_case_prior": 0.26,
            "od_engine_disagreement_prior": 0.24,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.43,
            "refc_stress_world_fraction": 0.20,
        },
        certificate_margin=0.01,
        search_completeness_score=0.94,
        search_completeness_gap=0.01,
        prior_support_strength=0.79,
        pending_challenger_mass=0.16,
        best_pending_flip_probability=0.20,
        frontier_recall_at_budget=0.95,
        top_refresh_gain=0.0022,
        top_fragility_mass=0.14,
        competitor_pressure=0.36,
        near_tie_mass=0.08,
    )
    enriched = enrich_controller_state_for_actioning(state, dccs=dccs, fragility=fragility, config=cfg)
    pending = voi_module._pending_dccs_candidates(dccs)
    best_candidate = voi_module._best_candidate(pending, config=cfg)
    assert best_candidate is not None
    raw_refine = voi_module._apply_search_completeness_bonus(
        voi_module.score_action(
            voi_module._build_refine_action(best_candidate, kind="refine_top1_dccs", config=cfg),
            config=cfg,
        ),
        state=enriched,
        config=cfg,
    )
    raw_refresh = voi_module.score_action(
        voi_module._build_refresh_action(
            "scenario",
            fragility=fragility,
            winner_id=state.winner_id,
            current_certificate=1.0,
            config=cfg,
        ),
        config=cfg,
    )
    pre_bonus_refresh = replace(
        raw_refresh,
        q_score=raw_refresh.q_score
        + voi_module._certified_refresh_priority_bonus(
            state=state,
            enriched_state=enriched,
            current_certificate=1.0,
            config=cfg,
            evidence_uncertainty=True,
            supported_fragility_uncertainty=True,
            recent_no_gain_refine_streak=1,
            stress_world_fraction=0.20,
        ),
    )

    assert raw_refine.q_score > pre_bonus_refresh.q_score

    actions = build_action_menu(enriched, dccs=dccs, fragility=fragility, config=cfg)
    kinds = [action.kind for action in actions]
    refresh = next(action for action in actions if action.kind == "refresh_top1_vor")
    refine = next(action for action in actions if action.kind == "refine_top1_dccs")
    assert actions[0].kind == "refresh_top1_vor"
    assert refresh.metadata["support_rich_certified_refresh_preference_applied"] is True
    assert refine.metadata.get("support_rich_certified_first_refine_discount_applied") is not True
    assert "refresh_top1_vor" in kinds


def test_low_support_certified_rows_do_not_get_support_rich_refresh_preference() -> None:
    dccs = _dccs_result()
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "scenario", "vor": 0.004}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.004,
        },
        route_fragility_map={"route_a": {"scenario": 0.003}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"scenario": 0.2}}},
    )
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        active_evidence_families=["scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.12,
            "od_hard_case_prior": 0.10,
            "od_engine_disagreement_prior": 0.06,
            "od_ambiguity_confidence": 0.22,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix": "routing_graph_probe",
            "od_ambiguity_support_ratio": 0.16,
            "od_ambiguity_prior_strength": 0.12,
            "od_ambiguity_source_entropy": 0.12,
            "ambiguity_budget_prior": 0.10,
            "refc_stress_world_fraction": 0.04,
        },
        certificate_margin=0.01,
        search_completeness_score=0.94,
        search_completeness_gap=0.01,
        prior_support_strength=0.14,
        pending_challenger_mass=0.16,
        best_pending_flip_probability=0.20,
        frontier_recall_at_budget=0.95,
        top_refresh_gain=0.004,
        top_fragility_mass=0.10,
        competitor_pressure=0.20,
        near_tie_mass=0.08,
    )

    actions = build_action_menu(
        enrich_controller_state_for_actioning(
            state,
            dccs=dccs,
            fragility=fragility,  # type: ignore[arg-type]
            config=cfg,
        ),
        dccs=dccs,
        fragility=fragility,  # type: ignore[arg-type]
        config=cfg,
    )
    assert all(action.kind != "refresh_top1_vor" for action in actions)
    assert all(
        action.metadata.get("support_rich_certified_refresh_preference_applied") is not True
        for action in actions
    )


def test_low_prior_certified_rows_with_strong_live_evidence_signal_still_offer_refresh() -> None:
    dccs = _dccs_result()
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "fuel", "vor": 0.916667}],
            "top_refresh_family": "fuel",
            "top_refresh_gain": 0.916667,
        },
        route_fragility_map={
            "route_a": {"fuel": 0.916667},
            "route_b": {"fuel": 0.0},
            "route_c": {"fuel": 0.0},
        },
        competitor_fragility_breakdown={"route_a": {"route_b": {"fuel": 1.0}}},
    )
    cfg = VOIConfig(certificate_threshold=0.80, stop_threshold=0.02)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.03, 10.02, 10.01)},
            {"route_id": "route_c", "objective_vector": (10.07, 10.05, 10.04)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0, "route_c": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["fuel", "scenario", "toll"],
        stochastic_enabled=False,
        ambiguity_context={
            "od_ambiguity_index": 0.04,
            "od_hard_case_prior": 0.33,
            "od_engine_disagreement_prior": 0.33,
            "od_ambiguity_confidence": 0.86,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":1,"routing_graph_probe":1}',
            "od_ambiguity_support_ratio": 0.47,
            "od_ambiguity_prior_strength": 0.04,
            "od_ambiguity_source_entropy": 0.72,
            "ambiguity_budget_prior": 0.04,
            "refc_stress_world_fraction": 0.60,
        },
        certificate_margin=0.2,
        search_completeness_score=0.653311,
        search_completeness_gap=0.186689,
        prior_support_strength=0.279284,
        pending_challenger_mass=0.624306,
        best_pending_flip_probability=0.997822,
        frontier_recall_at_budget=0.382004,
        top_refresh_gain=0.916667,
        top_fragility_mass=0.916667,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )

    actions = build_action_menu(
        enrich_controller_state_for_actioning(
            state,
            dccs=dccs,
            fragility=fragility,  # type: ignore[arg-type]
            config=cfg,
        ),
        dccs=dccs,
        fragility=fragility,  # type: ignore[arg-type]
        config=cfg,
    )

    refresh = next(action for action in actions if action.kind == "refresh_top1_vor")

    assert actions[0].kind == "refresh_top1_vor"
    assert refresh.q_score > cfg.stop_threshold
    assert refresh.metadata.get("support_rich_certified_refresh_preference_applied") is not True
    assert all(action.kind != "increase_stochastic_samples" for action in actions)
    assert all(
        action.metadata.get("support_rich_certified_first_refine_discount_applied") is not True
        for action in actions
    )


def test_support_rich_hard_case_positive_refresh_signal_can_bridge_above_stop_threshold() -> None:
    dccs = select_candidates(
        [
            _candidate(
                "cand_cardiff",
                path=["c1", "c2"],
                objective=(8.9, 9.0, 9.1),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=0),
    )
    fragility = replace(
        _supported_ambiguity_fragility_result(),
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "baseline_margin_summary": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
            "fragility_stress_state": "severely_stale",
            "per_family_certificate": {"fuel": 1.0},
            "per_family_margin_summary": {"fuel": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0}},
            "ranking": [{"family": "fuel", "vor": 0.001971}],
            "top_refresh_family": "fuel",
            "top_refresh_gain": 0.001971,
        },
        route_fragility_map={
            "route_a": {"fuel": 0.001971},
            "route_b": {"fuel": 0.0},
            "route_c": {"fuel": 0.0},
        },
    )
    cfg = VOIConfig(certificate_threshold=0.80, stop_threshold=0.015)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.01, 10.02, 10.01)},
            {"route_id": "route_c", "objective_vector": (10.02, 10.01, 10.02)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.5, "route_c": 0.5},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[],
        active_evidence_families=["fuel", "scenario", "toll"],
        ambiguity_context={
            "od_ambiguity_index": 0.64,
            "od_hard_case_prior": 0.64,
            "od_engine_disagreement_prior": 0.38,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.77,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.64,
            "refc_stress_world_fraction": 0.20,
        },
        certificate_margin=0.01,
        search_completeness_score=0.458663,
        search_completeness_gap=0.381337,
        prior_support_strength=0.707403,
        pending_challenger_mass=0.705253,
        best_pending_flip_probability=0.99996,
        frontier_recall_at_budget=0.303577,
        top_refresh_gain=0.001971,
        top_fragility_mass=0.001971,
        competitor_pressure=0.0,
        near_tie_mass=0.0,
    )

    enriched = enrich_controller_state_for_actioning(state, dccs=dccs, fragility=fragility, config=cfg)
    pending = voi_module._pending_dccs_candidates(dccs)
    best_candidate = voi_module._best_candidate(pending, config=cfg)
    assert best_candidate is not None
    raw_refine = voi_module._apply_search_completeness_bonus(
        voi_module.score_action(
            voi_module._build_refine_action(best_candidate, kind="refine_top1_dccs", config=cfg),
            config=cfg,
        ),
        state=enriched,
        config=cfg,
    )
    assert voi_module._search_action_shows_strong_decision_movement(raw_refine, state=enriched) is True
    assert voi_module._refine_action_has_genuine_novel_search_promise(raw_refine, state=enriched) is False

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=cfg)
    refresh = next(action for action in actions if action.kind == "refresh_top1_vor")
    refine = next(action for action in actions if action.kind == "refine_top1_dccs")

    assert actions[0].kind == "refresh_top1_vor"
    assert refresh.q_score >= cfg.stop_threshold
    assert refresh.q_score > refine.q_score
    assert refresh.metadata["support_rich_certified_refresh_preference_applied"] is True
    assert refresh.metadata["support_rich_certified_refresh_preference_first_action_bridge"] is True


def test_support_rich_hard_case_positive_fragility_without_vor_can_bridge_above_stop_threshold() -> None:
    dccs = select_candidates(
        [
            _candidate(
                "cand_cardiff_fragility",
                path=["c1", "c2"],
                objective=(8.9, 9.0, 9.1),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=0),
    )
    fragility = replace(
        _supported_ambiguity_fragility_result(),
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "baseline_margin_summary": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
            "fragility_stress_state": "severely_stale",
            "per_family_certificate": {"fuel": 1.0},
            "per_family_margin_summary": {"fuel": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0}},
            "ranking": [{"family": "fuel", "vor": 0.0}],
            "top_refresh_family": "fuel",
            "top_refresh_gain": 0.0,
        },
        route_fragility_map={
            "route_a": {"fuel": 0.006},
            "route_b": {"fuel": 0.0},
            "route_c": {"fuel": 0.0},
        },
    )
    cfg = VOIConfig(certificate_threshold=0.80, stop_threshold=0.015)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.01, 10.02, 10.01)},
            {"route_id": "route_c", "objective_vector": (10.02, 10.01, 10.02)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.5, "route_c": 0.5},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[],
        active_evidence_families=["fuel", "scenario", "toll"],
        ambiguity_context={
            "od_ambiguity_index": 0.64,
            "od_hard_case_prior": 0.64,
            "od_engine_disagreement_prior": 0.38,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.77,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.64,
            "refc_stress_world_fraction": 0.20,
        },
        certificate_margin=0.01,
        search_completeness_score=0.458663,
        search_completeness_gap=0.381337,
        prior_support_strength=0.707403,
        pending_challenger_mass=0.705253,
        best_pending_flip_probability=0.99996,
        frontier_recall_at_budget=0.303577,
        top_refresh_gain=0.0,
        top_fragility_mass=0.006,
        competitor_pressure=0.0,
        near_tie_mass=0.0,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=cfg)
    refresh = next(action for action in actions if action.kind == "refresh_top1_vor")

    assert actions[0].kind == "refresh_top1_vor"
    assert refresh.q_score >= cfg.stop_threshold
    assert refresh.metadata["support_rich_certified_refresh_preference_applied"] is True
    assert refresh.metadata["support_rich_certified_refresh_preference_first_action_bridge"] is True


def test_zero_vor_hard_case_row_does_not_get_first_action_refresh_bridge() -> None:
    dccs = _dccs_result()
    fragility = replace(
        _supported_ambiguity_fragility_result(),
        value_of_refresh={
            "ranking": [],
            "top_refresh_family": None,
            "top_refresh_gain": 0.0,
        },
        route_fragility_map={
            "route_a": {"fuel": 0.0},
            "route_b": {"fuel": 0.0},
        },
    )
    cfg = VOIConfig(certificate_threshold=0.80, stop_threshold=0.015)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.01, 10.02, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.5},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[],
        active_evidence_families=["fuel", "scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.64,
            "od_hard_case_prior": 0.64,
            "od_engine_disagreement_prior": 0.38,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.77,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.64,
            "refc_stress_world_fraction": 0.20,
        },
        certificate_margin=0.01,
        search_completeness_score=0.458663,
        search_completeness_gap=0.381337,
        prior_support_strength=0.707403,
        pending_challenger_mass=0.705253,
        best_pending_flip_probability=0.99996,
        frontier_recall_at_budget=0.303577,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
        near_tie_mass=0.0,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=cfg)
    refresh = next((action for action in actions if action.kind == "refresh_top1_vor"), None)

    assert refresh is None or refresh.metadata.get("support_rich_certified_refresh_preference_first_action_bridge") is not True


def test_positive_refresh_signal_hard_case_row_does_not_bridge_when_search_is_settled() -> None:
    dccs = _dccs_result()
    fragility = replace(
        _supported_ambiguity_fragility_result(),
        value_of_refresh={
            "ranking": [{"family": "fuel", "vor": 0.001971}],
            "top_refresh_family": "fuel",
            "top_refresh_gain": 0.001971,
        },
        route_fragility_map={
            "route_a": {"fuel": 0.001971},
            "route_b": {"fuel": 0.0},
        },
    )
    cfg = VOIConfig(certificate_threshold=0.80, stop_threshold=0.015)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.01, 10.02, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[],
        active_evidence_families=["fuel", "scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.64,
            "od_hard_case_prior": 0.64,
            "od_engine_disagreement_prior": 0.38,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.77,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.64,
            "refc_stress_world_fraction": 0.20,
        },
        certificate_margin=1.0,
        search_completeness_score=0.96,
        search_completeness_gap=0.02,
        prior_support_strength=0.707403,
        pending_challenger_mass=0.08,
        best_pending_flip_probability=0.12,
        frontier_recall_at_budget=0.95,
        top_refresh_gain=0.001971,
        top_fragility_mass=0.001971,
        competitor_pressure=0.0,
        near_tie_mass=0.0,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=cfg)
    refresh = next((action for action in actions if action.kind == "refresh_top1_vor"), None)

    assert refresh is None or refresh.metadata.get("support_rich_certified_refresh_preference_first_action_bridge") is not True


@pytest.mark.parametrize(
    ("evidence_uncertainty", "supported_fragility_uncertainty"),
    [
        (False, True),
        (True, False),
    ],
)
def test_positive_refresh_signal_hard_case_row_does_not_bridge_without_credible_evidence_uncertainty(
    evidence_uncertainty: bool,
    supported_fragility_uncertainty: bool,
) -> None:
    dccs = select_candidates(
        [
            _candidate(
                "cand_cardiff_guardrail",
                path=["c1", "c2"],
                objective=(8.9, 9.0, 9.1),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=0),
    )
    fragility = replace(
        _supported_ambiguity_fragility_result(),
        value_of_refresh={
            "selected_route_id": "route_a",
            "baseline_certificate": 1.0,
            "baseline_margin_summary": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0},
            "fragility_stress_state": "severely_stale",
            "per_family_certificate": {"fuel": 1.0},
            "per_family_margin_summary": {"fuel": {"world_count": 61.0, "mean_runner_up_gap": 0.0, "margin_stability_signal": 0.0}},
            "ranking": [{"family": "fuel", "vor": 0.001971}],
            "top_refresh_family": "fuel",
            "top_refresh_gain": 0.001971,
        },
        route_fragility_map={
            "route_a": {"fuel": 0.001971},
            "route_b": {"fuel": 0.0},
            "route_c": {"fuel": 0.0},
        },
    )
    cfg = VOIConfig(certificate_threshold=0.80, stop_threshold=0.015)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.01, 10.02, 10.01)},
            {"route_id": "route_c", "objective_vector": (10.02, 10.01, 10.02)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0, "route_c": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[],
        active_evidence_families=["fuel", "scenario", "toll"],
        ambiguity_context={
            "od_ambiguity_index": 0.64,
            "od_hard_case_prior": 0.64,
            "od_engine_disagreement_prior": 0.38,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.77,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.64,
            "refc_stress_world_fraction": 0.20,
        },
        certificate_margin=1.0,
        search_completeness_score=0.458663,
        search_completeness_gap=0.381337,
        prior_support_strength=0.707403,
        pending_challenger_mass=0.705253,
        best_pending_flip_probability=0.99996,
        frontier_recall_at_budget=0.303577,
        top_refresh_gain=0.001971,
        top_fragility_mass=0.001971,
        competitor_pressure=0.0,
        near_tie_mass=0.0,
    )

    enriched = enrich_controller_state_for_actioning(state, dccs=dccs, fragility=fragility, config=cfg)
    pending = voi_module._pending_dccs_candidates(dccs)
    best_candidate = voi_module._best_candidate(pending, config=cfg)
    assert best_candidate is not None
    actions = [
        voi_module.score_action(
            voi_module._build_refresh_action(
                "fuel",
                fragility=fragility,
                winner_id=state.winner_id,
                current_certificate=1.0,
                config=cfg,
            ),
            config=cfg,
        ),
        voi_module._certified_support_rich_first_refine_discount(
            voi_module._apply_search_completeness_bonus(
                voi_module.score_action(
                    voi_module._build_refine_action(best_candidate, kind="refine_top1_dccs", config=cfg),
                    config=cfg,
                ),
                state=enriched,
                config=cfg,
            ),
            state=enriched,
            current_certificate=1.0,
            config=cfg,
        ),
        voi_module.VOIAction(
            action_id="stop",
            kind="stop",
            target="stop",
            q_score=0.0,
            feasible=True,
            preconditions=("always",),
            reason="stop",
        ),
    ]

    refreshed = voi_module._apply_support_rich_certified_refresh_preference(
        actions,
        state=enriched,
        current_certificate=1.0,
        config=cfg,
        evidence_uncertainty=evidence_uncertainty,
        supported_fragility_uncertainty=supported_fragility_uncertainty,
        recent_no_gain_refine_streak=0,
    )
    refresh = next(action for action in refreshed if action.kind == "refresh_top1_vor")

    assert refresh.metadata.get("support_rich_certified_refresh_preference_first_action_bridge") is not True
    assert refresh.metadata.get("support_rich_certified_refresh_preference_applied") is not True
    assert refresh.q_score < cfg.stop_threshold


def test_support_rich_certified_refresh_preference_can_recover_after_harmful_refresh_reveal() -> None:
    cfg = VOIConfig(certificate_threshold=0.81)
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.1, 10.1, 10.1)},
        ],
        certificate={"route_a": 0.844311, "route_b": 0.155689},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=1,
        action_trace=[
            {
                "chosen_action": {"kind": "refresh_top1_vor"},
                "realized_certificate_delta": -0.155689,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_changed": False,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": -0.36527,
                "realized_productive": True,
            }
        ],
        ambiguity_context={
            "od_ambiguity_index": 0.38083,
            "od_hard_case_prior": 0.484062,
            "od_ambiguity_support_ratio": 0.539459,
            "od_ambiguity_source_entropy": 1.0,
            "od_ambiguity_prior_strength": 0.38083,
            "ambiguity_budget_prior": 0.38083,
            "refc_stress_world_fraction": 0.568862,
        },
        certificate_margin=0.076228,
        search_completeness_score=0.465874,
        search_completeness_gap=0.374126,
        prior_support_strength=0.652031,
        support_richness=0.652031,
        ambiguity_pressure=0.695783,
        pending_challenger_mass=0.669036,
        best_pending_flip_probability=0.99922,
        frontier_recall_at_budget=0.135125,
        top_refresh_gain=0.904192,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:carbon",
        kind="refresh_top1_vor",
        target="carbon",
        q_score=0.1535837431,
        predicted_delta_certificate=0.1306586625,
        predicted_delta_margin=0.0968442963,
        predicted_delta_frontier=0.0,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.11976,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.1647802522,
        predicted_delta_certificate=0.1556886228,
        predicted_delta_margin=0.2232782351,
        predicted_delta_frontier=0.1036501332,
        metadata={
            "normalized_objective_gap": 0.1036501332,
            "normalized_mechanism_gap": 0.3581777345,
            "normalized_overlap_reduction": 0.9090909091,
        },
    )
    refreshed = voi_module._apply_support_rich_certified_refresh_preference(
        [
            refresh,
            refine,
            voi_module.VOIAction(
                action_id="stop",
                kind="stop",
                target="stop",
                q_score=0.0,
                feasible=True,
                preconditions=("always",),
                reason="stop",
            ),
        ],
        state=state,
        current_certificate=0.844311,
        config=cfg,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )
    adjusted_refresh = next(action for action in refreshed if action.kind == "refresh_top1_vor")
    adjusted_refine = next(action for action in refreshed if action.kind == "refine_top1_dccs")

    assert adjusted_refresh.q_score > adjusted_refine.q_score
    assert adjusted_refresh.metadata["support_rich_certified_refresh_preference_applied"] is True
    assert adjusted_refresh.metadata["support_rich_certified_refresh_preference_post_harmful_refresh_recovery"] is True


def test_support_rich_certified_refresh_preference_can_recover_after_direct_fallback_frontier_probe() -> None:
    cfg = VOIConfig(certificate_threshold=0.83)
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.1, 10.05, 10.02)},
            {"route_id": "route_c", "objective_vector": (10.18, 10.11, 10.06)},
        ],
        certificate={"route_a": 0.95, "route_b": 0.05},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_changed": False,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.214286,
                "realized_productive": True,
            }
        ],
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "od_ambiguity_index": 0.36,
            "od_hard_case_prior": 0.61,
            "od_ambiguity_support_ratio": 0.69,
            "od_ambiguity_source_entropy": 0.84,
            "od_ambiguity_prior_strength": 0.36,
            "ambiguity_budget_prior": 0.36,
            "refc_stress_world_fraction": 0.42,
        },
        certificate_margin=0.049,
        search_completeness_score=0.551607,
        search_completeness_gap=0.288393,
        prior_support_strength=0.650912,
        support_richness=0.650912,
        ambiguity_pressure=0.604121,
        pending_challenger_mass=0.616288,
        best_pending_flip_probability=0.99651,
        frontier_recall_at_budget=0.380789,
        top_refresh_gain=0.107143,
        top_fragility_mass=0.057143,
        competitor_pressure=0.78,
        near_tie_mass=0.0,
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:stochastic",
        kind="refresh_top1_vor",
        target="stochastic",
        q_score=0.0819380443,
        predicted_delta_certificate=0.0559440559,
        predicted_delta_margin=0.0402306095,
        predicted_delta_frontier=0.0,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.055944,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.106534719,
        predicted_delta_certificate=0.05,
        predicted_delta_margin=0.1872,
        predicted_delta_frontier=0.1243206945,
        metadata={
            "normalized_objective_gap": 0.1327553447,
            "normalized_mechanism_gap": 0.1847791086,
            "normalized_overlap_reduction": 0.909091,
            "mean_flip_probability": 0.997963,
        },
    )

    refreshed = voi_module._apply_support_rich_certified_refresh_preference(
        [
            refresh,
            refine,
            voi_module.VOIAction(
                action_id="stop",
                kind="stop",
                target="stop",
                q_score=0.0,
                feasible=True,
                preconditions=("always",),
                reason="stop",
            ),
        ],
        state=state,
        current_certificate=0.95,
        config=cfg,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )
    adjusted_refresh = next(action for action in refreshed if action.kind == "refresh_top1_vor")
    adjusted_refine = next(action for action in refreshed if action.kind == "refine_top1_dccs")

    assert adjusted_refresh.q_score > adjusted_refine.q_score
    assert adjusted_refresh.metadata["support_rich_certified_refresh_preference_applied"] is True
    assert adjusted_refresh.metadata["support_rich_certified_refresh_preference_frontier_probe_recovery"] is True


def test_support_rich_but_settled_rows_do_not_reopen_controller_actions() -> None:
    dccs = _dccs_result()
    fragility = compute_fragility_maps(
        [
            _route("route_a", (10.0, 10.0, 10.0), {"scenario": {"time": 0.0, "money": 0.0, "co2": 0.0}}),
            _route("route_b", (10.4, 10.2, 10.3), {}),
        ],
        worlds=[{"world_id": "w0", "states": {"scenario": "nominal"}}],
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario"],
        selected_route_id="route_a",
    )
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario"],
        ambiguity_context={
            "od_ambiguity_index": 0.40,
            "od_hard_case_prior": 0.22,
            "od_ambiguity_confidence": 0.90,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.54,
            "od_ambiguity_source_entropy": 0.73,
            "ambiguity_budget_prior": 0.40,
            "refc_stress_world_fraction": 0.0,
        },
        certificate_margin=0.20,
        search_completeness_score=0.98,
        search_completeness_gap=0.0,
        prior_support_strength=0.78,
        pending_challenger_mass=0.02,
        best_pending_flip_probability=0.04,
        frontier_recall_at_budget=0.99,
        top_refresh_gain=0.01,
        top_fragility_mass=0.03,
        competitor_pressure=0.02,
        near_tie_mass=0.0,
    )

    assert credible_search_uncertainty(state, config=VOIConfig(certificate_threshold=0.80), current_certificate=1.0) is False
    assert (
        credible_evidence_uncertainty(
            state,
            fragility=fragility,
            config=VOIConfig(certificate_threshold=0.80),
            current_certificate=1.0,
        )
        is False
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(certificate_threshold=0.80))
    assert [action.kind for action in actions] == ["stop"]


def test_action_menu_uses_pending_dccs_candidates_not_already_selected_ones() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.25},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=0,
        action_trace=[],
        active_evidence_families=["scenario"],
        near_tie_mass=0.2,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig())
    refine = next(action for action in actions if action.kind == "refine_top1_dccs")

    assert refine.metadata["candidate_ids"] == ["cand_slow"]


def test_action_menu_falls_back_to_non_frontier_dccs_candidates_when_selected_set_is_overbroad() -> None:
    dccs = _dccs_result()
    dccs = replace(dccs, selected=list(dccs.candidate_ledger))
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{
            "route_id": "route_a",
            "candidate_id": "cand_fast",
            "objective_vector": (10.0, 10.0, 10.0),
        }],
        certificate={"route_a": 0.25},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=0,
        action_trace=[],
        active_evidence_families=["scenario"],
        near_tie_mass=0.2,
    )

    pending = voi_module._pending_dccs_candidates(dccs, state=state)
    assert [record.candidate_id for record in pending] == ["cand_slow"]

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig())
    refine = next(action for action in actions if action.kind == "refine_top1_dccs")

    assert refine.metadata["candidate_ids"] == ["cand_slow"]


def test_refresh_action_uses_fragility_and_competitor_pressure() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.25, "route_b": 0.20},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        near_tie_mass=0.2,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig())
    refresh = next(action for action in actions if action.kind == "refresh_top1_vor")

    assert refresh.metadata["vor_gain"] > 0.0
    assert refresh.metadata["route_fragility"] > 0.0
    assert refresh.metadata["competitor_pressure"] > 0.0


def test_supported_ambiguity_fragility_flows_into_refresh_action_scoring() -> None:
    dccs = _dccs_result()
    fragility = _supported_ambiguity_fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.03, 10.02, 10.01)},
            {"route_id": "route_c", "objective_vector": (10.07, 10.05, 10.04)},
        ],
        certificate={"route_a": 0.34, "route_b": 0.33, "route_c": 0.33},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario", "weather", "toll"],
        near_tie_mass=0.28,
        ambiguity_context={
            "od_ambiguity_index": 0.42,
            "od_hard_case_prior": 0.24,
            "od_engine_disagreement_prior": 0.22,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.26,
            "od_nominal_margin_proxy": 0.28,
            "od_ambiguity_family_density": 0.46,
            "od_ambiguity_margin_pressure": 0.42,
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.36,
            "ambiguity_budget_band": "medium",
        },
    )

    enriched = enrich_controller_state_for_actioning(
        state,
        dccs=dccs,
        fragility=fragility,
        config=VOIConfig(),
    )
    actions = build_action_menu(enriched, dccs=dccs, fragility=fragility, config=VOIConfig())
    refresh = next(action for action in actions if action.kind == "refresh_top1_vor")

    assert enriched.top_refresh_gain > 0.0
    assert refresh.metadata["vor_gain"] > 0.0
    assert refresh.metadata["route_fragility"] > 0.0
    assert refresh.predicted_delta_certificate > 0.0


def test_saturated_winner_refresh_action_can_still_score_positive_margin_signal() -> None:
    dccs = _dccs_result()
    fragility = _supported_ambiguity_fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.03, 10.02, 10.01)},
            {"route_id": "route_c", "objective_vector": (10.07, 10.05, 10.04)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0, "route_c": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario", "weather", "toll"],
        near_tie_mass=0.28,
        ambiguity_context={
            "od_ambiguity_index": 0.42,
            "od_hard_case_prior": 0.24,
            "od_engine_disagreement_prior": 0.22,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.26,
            "od_nominal_margin_proxy": 0.28,
            "od_ambiguity_family_density": 0.46,
            "od_ambiguity_margin_pressure": 0.42,
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.72,
            "od_ambiguity_confidence": 0.93,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.36,
            "ambiguity_budget_band": "medium",
        },
    )

    enriched = enrich_controller_state_for_actioning(
        state,
        dccs=dccs,
        fragility=fragility,
        config=VOIConfig(),
    )
    actions = build_action_menu(enriched, dccs=dccs, fragility=fragility, config=VOIConfig())
    refresh = next(action for action in actions if action.kind == "refresh_top1_vor")

    assert enriched.top_refresh_gain > 0.0
    assert refresh.metadata["vor_gain"] > 0.0
    assert refresh.metadata["route_fragility"] > 0.0


def test_supported_ambiguity_small_positive_refresh_signal_allows_evidence_action() -> None:
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "scenario", "vor": 0.000353}],
            "top_refresh_gain": 0.000353,
        },
        route_fragility_map={
            "route_a": {"scenario": 0.00012, "weather": 0.0, "toll": 0.0},
        },
        competitor_fragility_breakdown={
            "route_a": {"route_b": {"scenario": 2, "weather": 0, "toll": 0}},
        },
    )
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.01, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=0,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario", "weather", "toll"],
        near_tie_mass=0.18,
        ambiguity_context={
            "od_ambiguity_index": 0.41,
            "od_hard_case_prior": 0.24,
            "od_engine_disagreement_prior": 0.22,
            "od_candidate_path_count": 4,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.24,
            "od_nominal_margin_proxy": 0.29,
            "od_ambiguity_family_density": 0.43,
            "od_ambiguity_margin_pressure": 0.39,
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.71,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "ambiguity_budget_prior": 0.34,
            "ambiguity_budget_band": "medium",
        },
    )

    actions = build_action_menu(
        enrich_controller_state_for_actioning(
            state,
            dccs=_dccs_result(),
            fragility=fragility,  # type: ignore[arg-type]
            config=VOIConfig(),
        ),
        dccs=_dccs_result(),
        fragility=fragility,  # type: ignore[arg-type]
        config=VOIConfig(),
    )

    assert any(action.kind == "refresh_top1_vor" for action in actions)


def test_certified_support_rich_first_action_small_refresh_signal_prefers_refresh() -> None:
    dccs = select_candidates(
        [
            _candidate(
                "cand_soft",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=0),
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "scenario", "vor": 0.01}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.01,
        },
        route_fragility_map={"route_a": {"scenario": 0.02}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"scenario": 0.2}}},
    )
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario", "weather", "toll"],
        stochastic_enabled=False,
        ambiguity_context={
            "od_ambiguity_index": 0.44,
            "od_hard_case_prior": 0.55,
            "od_engine_disagreement_prior": 0.24,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.43,
            "refc_stress_world_fraction": 0.20,
            "is_hard_case": True,
        },
        certificate_margin=0.01,
        search_completeness_score=0.96,
        search_completeness_gap=0.0,
        prior_support_strength=0.79,
        pending_challenger_mass=0.08,
        best_pending_flip_probability=0.10,
        frontier_recall_at_budget=0.98,
        top_refresh_gain=0.01,
        top_fragility_mass=0.02,
        competitor_pressure=0.20,
        near_tie_mass=0.06,
    )
    enriched = enrich_controller_state_for_actioning(state, dccs=dccs, fragility=fragility, config=cfg)
    pending = voi_module._pending_dccs_candidates(dccs)
    best_candidate = voi_module._best_candidate(pending, config=cfg)
    assert best_candidate is not None
    raw_refine = voi_module._apply_search_completeness_bonus(
        voi_module.score_action(
            voi_module._build_refine_action(best_candidate, kind="refine_top1_dccs", config=cfg),
            config=cfg,
        ),
        state=enriched,
        config=cfg,
    )
    raw_refresh = voi_module.score_action(
        voi_module._build_refresh_action(
            "scenario",
            fragility=fragility,  # type: ignore[arg-type]
            winner_id=state.winner_id,
            current_certificate=1.0,
            config=cfg,
        ),
        config=cfg,
    )

    assert raw_refine.q_score > raw_refresh.q_score

    actions = build_action_menu(enriched, dccs=dccs, fragility=fragility, config=cfg)  # type: ignore[arg-type]

    refine = next(action for action in actions if action.kind == "refine_top1_dccs")
    assert actions[0].kind == "refresh_top1_vor"
    assert refine.metadata["support_rich_certified_first_refine_discount_applied"] is True


def test_uncertified_support_rich_strong_winner_side_signal_prefers_refresh_first() -> None:
    base_dccs = select_candidates(
        [
            _candidate(
                "cand_soft",
                path=["c1", "c2", "c3", "c4"],
                objective=(16.0, 15.6, 15.4),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=8.5,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=0),
    )
    synthetic_candidate = replace(
        base_dccs.skipped[0],
        objective_gap=0.80,
        mechanism_gap=0.0,
        flip_probability=0.9999,
        final_score=0.03,
    )
    dccs = replace(
        base_dccs,
        selected=[],
        skipped=[synthetic_candidate],
        candidate_ledger=[synthetic_candidate],
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "fuel", "vor": 0.20}],
            "top_refresh_family": "fuel",
            "top_refresh_gain": 0.20,
        },
        route_fragility_map={"route_a": {"fuel": 0.18}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"fuel": 96.0}}},
    )
    cfg = VOIConfig(certificate_threshold=0.67)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
            {"route_id": "route_c", "objective_vector": (11.4, 10.4, 10.2)},
        ],
        certificate={"route_a": 0.457143, "route_b": 0.45, "route_c": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["fuel", "scenario", "weather"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.71,
            "od_hard_case_prior": 0.74,
            "od_engine_disagreement_prior": 0.42,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.71,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.74,
            "refc_stress_world_fraction": 0.54,
            "is_hard_case": True,
        },
        certificate_margin=0.0,
        search_completeness_score=0.40,
        search_completeness_gap=0.44,
        prior_support_strength=0.71,
        pending_challenger_mass=0.70,
        best_pending_flip_probability=0.99,
        frontier_recall_at_budget=0.30,
        top_refresh_gain=0.20,
        top_fragility_mass=0.18,
        competitor_pressure=1.0,
        near_tie_mass=0.5,
    )
    enriched = enrich_controller_state_for_actioning(state, dccs=dccs, fragility=fragility, config=cfg)
    pending = voi_module._pending_dccs_candidates(dccs)
    best_candidate = voi_module._best_candidate(pending, config=cfg)
    assert best_candidate is not None
    raw_refine = voi_module._apply_search_completeness_bonus(
        voi_module.score_action(
            voi_module._build_refine_action(best_candidate, kind="refine_top1_dccs", config=cfg),
            config=cfg,
        ),
        state=enriched,
        config=cfg,
    )
    raw_refresh = voi_module.score_action(
        voi_module._build_refresh_action(
            "fuel",
            fragility=fragility,  # type: ignore[arg-type]
            winner_id=state.winner_id,
            current_certificate=0.457143,
            config=cfg,
        ),
        config=cfg,
    )

    assert raw_refine.q_score > raw_refresh.q_score

    adjusted_actions = voi_module._apply_strong_winner_side_refresh_preference(
        [
            replace(raw_refine, q_score=raw_refresh.q_score + 0.05),
            raw_refresh,
        ],
        state=enriched,
        current_certificate=0.457143,
        config=cfg,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )

    refresh = next(action for action in adjusted_actions if action.kind == "refresh_top1_vor")
    refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    assert refresh.q_score > refine.q_score
    assert refresh.metadata["winner_side_refresh_preference_applied"] is True
    assert refresh.metadata["winner_side_refresh_preference_uncertified_bridge"] is True
    assert refine.metadata["winner_side_refresh_refine_discount_applied"] is True


def test_uncertified_structured_refresh_bridge_can_override_mechanism_gap_veto() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.805128, "route_b": 0.194872},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=4,
        remaining_evidence_budget=3,
        ambiguity_context={
            "od_ambiguity_index": 0.403029,
            "od_hard_case_prior": 0.492942,
            "od_ambiguity_confidence": 0.810497,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":4,"routing_graph_probe":11}',
            "od_ambiguity_support_ratio": 0.686842,
            "od_ambiguity_source_entropy": 0.836641,
            "ambiguity_budget_prior": 0.423843,
        },
        support_richness=0.650912,
        ambiguity_pressure=0.616277,
        top_refresh_gain=0.389744,
        top_fragility_mass=0.194872,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.307431,
        predicted_delta_certificate=0.402886,
        predicted_delta_margin=0.215522,
        predicted_delta_frontier=0.078791,
        metadata={
            "mean_flip_probability": 0.996632,
            "normalized_objective_gap": 0.043569,
            "normalized_mechanism_gap": 0.148094,
            "normalized_overlap_reduction": 0.909091,
        },
        reason="refine_candidate",
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:stochastic",
        kind="refresh_top1_vor",
        target="stochastic",
        q_score=0.177264,
        predicted_delta_certificate=0.225699,
        predicted_delta_margin=0.167378,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.194872,
        },
        reason="refresh_evidence_family",
    )

    adjusted_actions = voi_module._apply_strong_winner_side_refresh_preference(
        [refine, refresh],
        state=state,
        current_certificate=0.805128,
        config=VOIConfig(certificate_threshold=0.84),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )

    adjusted_refresh = next(action for action in adjusted_actions if action.kind == "refresh_top1_vor")
    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    assert adjusted_refresh.q_score > adjusted_refine.q_score
    assert adjusted_refresh.metadata["winner_side_refresh_preference_applied"] is True
    assert adjusted_refresh.metadata["winner_side_refresh_preference_uncertified_bridge"] is True
    assert adjusted_refresh.metadata["winner_side_refresh_preference_structured_bridge"] is True
    assert adjusted_refine.metadata["winner_side_refresh_refine_discount_applied"] is True


def test_uncertified_empirical_refresh_bridge_can_override_large_q_gap_veto() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.457143, "route_b": 0.542857},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        ambiguity_context={
            "od_ambiguity_index": 0.453718,
            "od_hard_case_prior": 0.692693,
            "od_ambiguity_support_ratio": 0.707403,
            "od_ambiguity_source_entropy": 0.836641,
            "ambiguity_budget_prior": 0.453718,
        },
        support_richness=0.707403,
        ambiguity_pressure=0.725242,
        top_refresh_gain=0.531429,
        top_fragility_mass=0.457143,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.304722,
        predicted_delta_certificate=0.405013,
        predicted_delta_margin=0.212208,
        predicted_delta_frontier=0.057745,
        metadata={
            "mean_flip_probability": 0.969257,
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.114019,
            "normalized_overlap_reduction": 0.904762,
        },
        reason="refine_candidate",
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.084621,
        predicted_delta_certificate=0.108502,
        predicted_delta_margin=0.078079,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.074286,
        },
        reason="refresh_evidence_family",
    )

    adjusted_actions = voi_module._apply_strong_winner_side_refresh_preference(
        [refine, refresh],
        state=state,
        current_certificate=0.457143,
        config=VOIConfig(certificate_threshold=0.83),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )

    adjusted_refresh = next(action for action in adjusted_actions if action.kind == "refresh_top1_vor")
    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    assert adjusted_refresh.q_score > adjusted_refine.q_score
    assert adjusted_refresh.metadata["winner_side_refresh_preference_applied"] is True
    assert adjusted_refresh.metadata["winner_side_refresh_preference_empirical_bridge"] is True
    assert adjusted_refine.metadata["winner_side_refresh_refine_discount_applied"] is True


def test_uncertified_post_route_change_refresh_bridge_can_override_second_refine() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_2", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_0", "objective_vector": (10.2, 10.1, 10.05)},
            {"route_id": "route_1", "objective_vector": (10.25, 10.11, 10.08)},
        ],
        certificate={"route_2": 0.412371, "route_0": 0.365979, "route_1": 0.22165},
        winner_id="route_2",
        selected_route_id="route_2",
        remaining_search_budget=1,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": -0.092865,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_changed": True,
                "realized_selected_route_improvement": 0.0,
                "realized_productive": True,
            }
        ],
        ambiguity_context={
            "od_ambiguity_index": 0.41312,
            "od_hard_case_prior": 0.56246,
            "od_ambiguity_support_ratio": 0.876689,
            "od_ambiguity_source_entropy": 0.96023,
            "ambiguity_budget_prior": 0.41312,
        },
        support_richness=0.728969,
        ambiguity_pressure=0.743899,
        top_refresh_gain=0.525773,
        top_fragility_mass=0.154639,
        competitor_pressure=1.0,
        pending_challenger_mass=0.731456,
        best_pending_flip_probability=0.999963,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.521136,
        predicted_delta_certificate=0.571872,
        predicted_delta_margin=0.37009,
        predicted_delta_frontier=0.569938,
        metadata={
            "normalized_objective_gap": 0.515921,
            "normalized_mechanism_gap": 0.127752,
            "normalized_overlap_reduction": 0.666667,
        },
        reason="refine_candidate",
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:scenario",
        kind="refresh_top1_vor",
        target="scenario",
        q_score=0.307798,
        predicted_delta_certificate=0.391191,
        predicted_delta_margin=0.292334,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.371134,
        },
        reason="refresh_evidence_family",
    )

    adjusted_actions = voi_module._apply_strong_winner_side_refresh_preference(
        [refine, refresh],
        state=state,
        current_certificate=0.412371,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )

    adjusted_refresh = next(action for action in adjusted_actions if action.kind == "refresh_top1_vor")
    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    assert adjusted_refresh.q_score > adjusted_refine.q_score
    assert adjusted_refresh.metadata["winner_side_refresh_preference_applied"] is True
    assert adjusted_refresh.metadata["winner_side_refresh_preference_post_route_change_bridge"] is True
    assert adjusted_refine.metadata["winner_side_refresh_refine_discount_applied"] is True


def test_uncertified_first_iteration_near_tie_resample_preference_can_override_non_novel_refine() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.0, 10.0)},
        ],
        certificate={"route_a": 0.6, "route_b": 0.4},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        near_tie_mass=1.0,
        top_refresh_gain=0.375,
        top_fragility_mass=0.4,
        competitor_pressure=1.0,
        search_completeness_score=0.426196,
        search_completeness_gap=0.373804,
        prior_support_strength=0.553078,
        support_richness=0.553078,
        ambiguity_pressure=0.779229,
        pending_challenger_mass=0.641344,
        best_pending_flip_probability=0.997754,
        frontier_recall_at_budget=0.145467,
        corridor_family_recall=0.333333,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.327504,
        predicted_delta_certificate=0.433182,
        predicted_delta_margin=0.219478,
        predicted_delta_frontier=0.084832,
        metadata={
            "normalized_objective_gap": 0.042138,
            "normalized_mechanism_gap": 0.219664,
            "normalized_overlap_reduction": 0.904762,
            "mean_flip_probability": 0.997754,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.1807,
        predicted_delta_certificate=0.252,
        predicted_delta_margin=0.10,
        predicted_delta_frontier=0.03,
        metadata={"near_tie_mass": 1.0, "sample_increment": 32},
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.022714,
        predicted_delta_certificate=0.0297,
        predicted_delta_margin=0.019575,
        predicted_delta_frontier=0.0,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.0,
            "empirical_refresh_certificate_delta": -0.025,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    adjusted = voi_module._apply_uncertified_first_iteration_near_tie_resample_preference(
        [refine, resample, refresh, stop],
        state=state,
        current_certificate=0.6,
        config=VOIConfig(certificate_threshold=0.8),
        evidence_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    adjusted_resample = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_first_iteration_near_tie_resample_preference_applied"] is True
    assert adjusted_refine.metadata["uncertified_first_iteration_near_tie_resample_search_discount_applied"] is True


def test_uncertified_single_frontier_zero_signal_search_churn_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.79},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        search_completeness_score=0.428891,
        search_completeness_gap=0.411109,
        prior_support_strength=0.645344,
        support_richness=0.645344,
        ambiguity_pressure=0.674435,
        pending_challenger_mass=0.628706,
        best_pending_flip_probability=0.997038,
        frontier_recall_at_budget=0.149925,
        corridor_family_recall=0.090909,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.20436,
        predicted_delta_certificate=0.21,
        predicted_delta_margin=0.237255,
        predicted_delta_frontier=0.126972,
        metadata={
            "normalized_objective_gap": 0.081255,
            "normalized_mechanism_gap": 0.063474,
            "normalized_overlap_reduction": 0.909091,
            "mean_flip_probability": 0.996254,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_single_frontier_zero_signal_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.79,
        config=VOIConfig(certificate_threshold=0.8),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_uncertified_single_frontier_zero_signal_search_churn_stops_mechanism_only_reopen() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.771728},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        search_completeness_score=0.488497,
        search_completeness_gap=0.351503,
        prior_support_strength=0.591697,
        support_richness=0.591697,
        ambiguity_pressure=0.569307,
        pending_challenger_mass=0.617901,
        best_pending_flip_probability=0.998199,
        frontier_recall_at_budget=0.365994,
        corridor_family_recall=0.333333,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.214823,
        predicted_delta_certificate=0.228272,
        predicted_delta_margin=0.23371,
        predicted_delta_frontier=0.129549,
        metadata={
            "normalized_objective_gap": 0.102336,
            "normalized_mechanism_gap": 0.26323,
            "normalized_overlap_reduction": 0.904762,
            "mean_flip_probability": 0.998199,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_single_frontier_zero_signal_search_churn(
        [refine, stop],
        state=state,
        current_certificate=0.771728,
        config=VOIConfig(certificate_threshold=0.8),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_uncertified_sampler_only_zero_signal_bridge_tail_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.757826},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_hard_case_prior": 0.30177,
            "ambiguity_budget_prior": 0.172,
            "od_ambiguity_support_ratio": 0.72632,
            "single_frontier_certificate_cap_applied": False,
        },
        certificate_margin=0.757826,
        search_completeness_score=0.773486,
        search_completeness_gap=0.066514,
        prior_support_strength=0.291274,
        support_richness=0.569489,
        ambiguity_pressure=0.397797,
        pending_challenger_mass=0.632488,
        best_pending_flip_probability=0.999718,
        frontier_recall_at_budget=1.0,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
        credible_evidence_uncertainty=False,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.29247,
        predicted_delta_certificate=0.242174,
        predicted_delta_margin=0.182414,
        predicted_delta_frontier=0.089569,
        metadata={
            "uncertified_support_rich_zero_signal_bridge": True,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": True,
            "uncertified_support_rich_zero_signal_cert_world_bridge": False,
            "controller_refresh_fallback_activated": False,
            "controller_empirical_vs_raw_refresh_disagreement": False,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.162483,
        predicted_delta_certificate=0.242174,
        predicted_delta_margin=0.215858,
        predicted_delta_frontier=0.088066,
        metadata={
            "normalized_objective_gap": 0.081717,
            "normalized_mechanism_gap": 0.640251,
            "normalized_overlap_reduction": 0.904762,
            "mean_flip_probability": 0.999718,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_sampler_only_zero_signal_bridge_tail(
        [bridge, refine, stop],
        state=state,
        current_certificate=0.757826,
        config=VOIConfig(certificate_threshold=0.8),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_uncertified_sampler_only_zero_signal_bridge_tail_preserves_cert_world_bridge() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.787574},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_hard_case_prior": 0.36199,
            "ambiguity_budget_prior": 0.26,
            "od_ambiguity_support_ratio": 0.64432,
            "single_frontier_certificate_cap_applied": True,
        },
        certificate_margin=0.787574,
        search_completeness_score=0.728575,
        search_completeness_gap=0.090374,
        prior_support_strength=0.344105,
        support_richness=0.60675,
        ambiguity_pressure=0.434049,
        pending_challenger_mass=0.620313,
        best_pending_flip_probability=0.996088,
        frontier_recall_at_budget=1.0,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.00019,
        competitor_pressure=0.0,
        credible_evidence_uncertainty=False,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.306015,
        predicted_delta_certificate=0.212426,
        predicted_delta_margin=0.168846,
        predicted_delta_frontier=0.067532,
        metadata={
            "uncertified_support_rich_zero_signal_bridge": True,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": False,
            "uncertified_support_rich_zero_signal_cert_world_bridge": True,
            "controller_refresh_fallback_activated": True,
            "controller_empirical_vs_raw_refresh_disagreement": True,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_sampler_only_zero_signal_bridge_tail(
        [bridge, stop],
        state=state,
        current_certificate=0.787574,
        config=VOIConfig(certificate_threshold=0.8),
    )

    assert [action.kind for action in filtered] == ["increase_stochastic_samples", "stop"]


def test_uncertified_low_support_cert_world_zero_signal_bridge_tail_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.800029},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.050984,
            "od_hard_case_prior": 0.12,
            "od_engine_disagreement_prior": 0.12,
            "ambiguity_budget_prior": 0.050984,
        },
        certificate_margin=-0.049971,
        search_completeness_score=0.790368,
        search_completeness_gap=0.209632,
        prior_support_strength=0.112096,
        pending_challenger_mass=0.644018,
        best_pending_flip_probability=0.998393,
        frontier_recall_at_budget=0.369673,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
        credible_evidence_uncertainty=True,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.174099,
        predicted_delta_certificate=0.199971,
        predicted_delta_margin=0.166257,
        predicted_delta_frontier=0.08368,
        metadata={
            "uncertified_support_rich_zero_signal_bridge": True,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": False,
            "uncertified_support_rich_zero_signal_cert_world_bridge": True,
            "controller_refresh_fallback_activated": True,
            "controller_empirical_vs_raw_refresh_disagreement": True,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.192,
        predicted_delta_certificate=0.182,
        predicted_delta_margin=0.148,
        predicted_delta_frontier=0.052,
        metadata={
            "normalized_objective_gap": 0.041,
            "normalized_mechanism_gap": 0.19,
            "normalized_overlap_reduction": 0.12,
            "mean_flip_probability": 0.29,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_low_support_cert_world_zero_signal_bridge_tail(
        [refine, bridge, stop],
        state=state,
        current_certificate=0.800029,
        config=VOIConfig(certificate_threshold=0.85),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_uncertified_low_support_cert_world_zero_signal_bridge_tail_preserves_high_support_bridge() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.81},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=3,
        remaining_evidence_budget=3,
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.043699,
            "od_hard_case_prior": 0.361974,
            "od_engine_disagreement_prior": 0.350457,
            "ambiguity_budget_prior": 0.043699,
        },
        certificate_margin=-0.01,
        search_completeness_score=0.783317,
        search_completeness_gap=0.216683,
        prior_support_strength=0.299975,
        pending_challenger_mass=0.63483,
        best_pending_flip_probability=0.996249,
        frontier_recall_at_budget=0.401175,
        near_tie_mass=0.0,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
        credible_evidence_uncertainty=True,
    )
    bridge = voi_module.VOIAction(
        action_id="resample:stochastic:uncertified_support_rich_zero_signal_bridge",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.188,
        predicted_delta_certificate=0.19,
        predicted_delta_margin=0.16,
        predicted_delta_frontier=0.08,
        metadata={
            "uncertified_support_rich_zero_signal_bridge": True,
            "uncertified_support_rich_zero_signal_live_sampler_bridge": False,
            "uncertified_support_rich_zero_signal_cert_world_bridge": True,
            "controller_refresh_fallback_activated": True,
            "controller_empirical_vs_raw_refresh_disagreement": True,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.205,
        predicted_delta_certificate=0.175,
        predicted_delta_margin=0.152,
        predicted_delta_frontier=0.051,
        metadata={
            "normalized_objective_gap": 0.042,
            "normalized_mechanism_gap": 0.18,
            "normalized_overlap_reduction": 0.12,
            "mean_flip_probability": 0.29,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_uncertified_low_support_cert_world_zero_signal_bridge_tail(
        [refine, bridge, stop],
        state=state,
        current_certificate=0.81,
        config=VOIConfig(certificate_threshold=0.85),
    )

    assert [action.kind for action in filtered] == [
        "refine_top1_dccs",
        "increase_stochastic_samples",
        "stop",
    ]


def test_uncertified_evidence_plateau_preference_can_promote_best_evidence_action() -> None:
    state = VOIControllerState(
        iteration_index=2,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 9.96, 10.03)},
            {"route_id": "route_c", "objective_vector": (10.11, 10.04, 10.06)},
        ],
        certificate={"route_a": 0.512821, "route_b": 0.287179},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.461289,
                "realized_evidence_uncertainty_delta": 0.005129,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.060932,
                "realized_evidence_uncertainty_delta": -0.046154,
                "realized_productive": True,
            },
        ],
        top_refresh_gain=0.287179,
        top_fragility_mass=0.512821,
        competitor_pressure=1.0,
        support_richness=0.650912,
        prior_support_strength=0.650912,
        ambiguity_pressure=0.734275,
        ambiguity_context={
            "od_hard_case_prior": 0.492942,
            "ambiguity_budget_prior": 0.423843,
            "od_ambiguity_support_ratio": 0.686842,
            "od_ambiguity_source_entropy": 0.836641,
            "od_ambiguity_index": 0.403029,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.394893,
        predicted_delta_certificate=0.477371,
        predicted_delta_margin=0.28753,
        predicted_delta_frontier=0.243921,
        metadata={
            "normalized_objective_gap": 0.195,
            "normalized_mechanism_gap": 0.08,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.028901,
        predicted_delta_certificate=0.037916,
        predicted_delta_margin=0.024607,
        predicted_delta_frontier=0.0,
        metadata={"structured_refresh_signal": True},
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.091449,
        predicted_delta_certificate=0.129744,
        predicted_delta_margin=0.045641,
        predicted_delta_frontier=0.014615,
        metadata={"near_tie_mass": 0.333333},
    )

    adjusted_actions = voi_module._apply_uncertified_evidence_plateau_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.512821,
        config=VOIConfig(certificate_threshold=0.84),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_resample = next(
        action for action in adjusted_actions if action.kind == "increase_stochastic_samples"
    )
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_evidence_plateau_preference_applied"] is True
    assert adjusted_resample.metadata["uncertified_evidence_plateau_progress_count"] == 2
    assert adjusted_refine.metadata["uncertified_evidence_plateau_search_discount_applied"] is True


def test_uncertified_evidence_plateau_preference_counts_runner_up_gap_progress_without_certificate_lift() -> None:
    state = VOIControllerState(
        iteration_index=3,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 9.96, 10.03)},
            {"route_id": "route_c", "objective_vector": (10.11, 10.04, 10.06)},
        ],
        certificate={"route_a": 0.512821, "route_b": 0.287179},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.565717,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.042663,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": True,
            },
        ],
        certificate_margin=0.0,
        near_tie_mass=0.333333,
        top_refresh_gain=0.302564,
        top_fragility_mass=0.512821,
        competitor_pressure=1.0,
        support_richness=0.650912,
        prior_support_strength=0.650912,
        ambiguity_pressure=0.824312,
        search_completeness_gap=0.432098,
        pending_challenger_mass=0.624695,
        best_pending_flip_probability=0.997177,
        corridor_family_recall=0.090909,
        frontier_recall_at_budget=0.131393,
        ambiguity_context={
            "od_hard_case_prior": 0.492942,
            "ambiguity_budget_prior": 0.423843,
            "od_ambiguity_support_ratio": 0.686842,
            "od_ambiguity_source_entropy": 0.836641,
            "od_ambiguity_index": 0.403029,
            "od_objective_spread": 0.0,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.325214,
        predicted_delta_certificate=0.432213,
        predicted_delta_margin=0.218792,
        predicted_delta_frontier=0.074586,
        metadata={
            "normalized_objective_gap": 0.01898,
            "normalized_mechanism_gap": 0.210598,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.028901,
        predicted_delta_certificate=0.037916,
        predicted_delta_margin=0.024607,
        predicted_delta_frontier=0.0,
        metadata={"structured_refresh_signal": True},
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.101115,
        predicted_delta_certificate=0.143077,
        predicted_delta_margin=0.052308,
        predicted_delta_frontier=0.014615,
        metadata={"near_tie_mass": 0.333333},
    )

    adjusted_actions = voi_module._apply_uncertified_evidence_plateau_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.512821,
        config=VOIConfig(certificate_threshold=0.84),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_resample = next(
        action for action in adjusted_actions if action.kind == "increase_stochastic_samples"
    )
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_evidence_plateau_preference_applied"] is True
    assert adjusted_resample.metadata["uncertified_evidence_plateau_progress_count"] == 2
    assert adjusted_refine.metadata["uncertified_evidence_plateau_search_discount_applied"] is True


def test_uncertified_evidence_plateau_preference_can_recover_from_single_direct_fallback_frontier_probe() -> None:
    state = VOIControllerState(
        iteration_index=2,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
            {"route_id": "route_c", "objective_vector": (10.4, 10.3, 10.15)},
        ],
        certificate={"route_a": 0.586466, "route_b": 0.330827},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": -0.015038,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.794808,
                "realized_evidence_uncertainty_delta": -0.090226,
                "realized_productive": True,
            }
        ],
        certificate_margin=0.005698,
        near_tie_mass=0.0,
        search_completeness_score=0.456564,
        search_completeness_gap=0.383436,
        pending_challenger_mass=0.693022,
        best_pending_flip_probability=0.999898,
        corridor_family_recall=0.5,
        frontier_recall_at_budget=0.176607,
        top_refresh_gain=0.255639,
        top_fragility_mass=0.586466,
        competitor_pressure=1.0,
        support_richness=0.728969,
        prior_support_strength=0.728969,
        ambiguity_pressure=0.686529,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "od_hard_case_prior": 0.56246,
            "ambiguity_budget_prior": 0.467618,
            "od_ambiguity_support_ratio": 0.876689,
            "od_ambiguity_source_entropy": 0.96023,
            "od_ambiguity_index": 0.41312,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.385652,
        predicted_delta_certificate=0.413534,
        predicted_delta_margin=0.332318,
        predicted_delta_frontier=0.363014,
        metadata={
            "normalized_objective_gap": 0.311563,
            "normalized_mechanism_gap": 0.111901,
            "normalized_overlap_reduction": 0.9,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.034708,
        predicted_delta_certificate=0.045613,
        predicted_delta_margin=0.029363,
        predicted_delta_frontier=0.0,
        metadata={"structured_refresh_signal": True},
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.061184,
        predicted_delta_certificate=0.086767,
        predicted_delta_margin=0.029323,
        predicted_delta_frontier=0.011955,
        metadata={"near_tie_mass": 0.0},
    )

    adjusted_actions = voi_module._apply_uncertified_evidence_plateau_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.586466,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_resample = next(
        action for action in adjusted_actions if action.kind == "increase_stochastic_samples"
    )
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_evidence_plateau_preference_applied"] is True
    assert adjusted_resample.metadata["uncertified_evidence_plateau_direct_fallback_probe_recovery"] is True
    assert adjusted_refine.metadata["uncertified_evidence_plateau_search_discount_applied"] is True
    assert adjusted_refine.metadata["uncertified_evidence_plateau_direct_fallback_probe_discount"] is True


def test_uncertified_evidence_plateau_preference_can_recover_from_single_direct_fallback_dead_probe() -> None:
    state = VOIControllerState(
        iteration_index=2,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
            {"route_id": "route_c", "objective_vector": (10.4, 10.3, 10.15)},
        ],
        certificate={"route_a": 0.586466, "route_b": 0.330827},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        certificate_margin=0.005698,
        near_tie_mass=0.0,
        search_completeness_score=0.456564,
        search_completeness_gap=0.383436,
        pending_challenger_mass=0.693022,
        best_pending_flip_probability=0.999898,
        corridor_family_recall=0.5,
        frontier_recall_at_budget=0.176607,
        top_refresh_gain=0.255639,
        top_fragility_mass=0.586466,
        competitor_pressure=1.0,
        support_richness=0.728969,
        prior_support_strength=0.728969,
        ambiguity_pressure=0.686529,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "od_hard_case_prior": 0.56246,
            "ambiguity_budget_prior": 0.467618,
            "od_ambiguity_support_ratio": 0.876689,
            "od_ambiguity_source_entropy": 0.96023,
            "od_ambiguity_index": 0.41312,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.335945,
        predicted_delta_certificate=0.413534,
        predicted_delta_margin=0.332318,
        predicted_delta_frontier=0.363014,
        metadata={
            "normalized_objective_gap": 0.311563,
            "normalized_mechanism_gap": 0.111901,
            "normalized_overlap_reduction": 0.9,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.034708,
        predicted_delta_certificate=0.045613,
        predicted_delta_margin=0.029363,
        predicted_delta_frontier=0.0,
        metadata={"structured_refresh_signal": True},
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.061184,
        predicted_delta_certificate=0.086767,
        predicted_delta_margin=0.029323,
        predicted_delta_frontier=0.011955,
        metadata={"near_tie_mass": 0.0},
    )

    adjusted_actions = voi_module._apply_uncertified_evidence_plateau_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.586466,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_resample = next(
        action for action in adjusted_actions if action.kind == "increase_stochastic_samples"
    )
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_evidence_plateau_preference_applied"] is True
    assert adjusted_resample.metadata["uncertified_evidence_plateau_direct_fallback_probe_recovery"] is True
    assert adjusted_resample.metadata["uncertified_evidence_plateau_direct_fallback_dead_probe_recovery"] is True
    assert adjusted_refine.metadata["uncertified_evidence_plateau_search_discount_applied"] is True
    assert adjusted_refine.metadata["uncertified_evidence_plateau_direct_fallback_probe_discount"] is True
    assert adjusted_refine.metadata["uncertified_evidence_plateau_direct_fallback_dead_probe_discount"] is True


def test_uncertified_evidence_plateau_preference_defers_to_frontier_completion_search() -> None:
    state = VOIControllerState(
        iteration_index=2,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 9.96, 10.03)},
            {"route_id": "route_c", "objective_vector": (10.11, 10.04, 10.06)},
            {"route_id": "route_d", "objective_vector": (10.24, 10.16, 10.09)},
        ],
        certificate={"route_a": 0.512821, "route_b": 0.287179},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.461289,
                "realized_evidence_uncertainty_delta": 0.005129,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.060932,
                "realized_evidence_uncertainty_delta": -0.046154,
                "realized_productive": True,
            },
        ],
        top_refresh_gain=0.287179,
        top_fragility_mass=0.512821,
        competitor_pressure=1.0,
        support_richness=0.650912,
        prior_support_strength=0.650912,
        ambiguity_pressure=0.734275,
        certificate_margin=0.221498,
        search_completeness_score=0.467012,
        search_completeness_gap=0.372988,
        pending_challenger_mass=0.66337,
        best_pending_flip_probability=0.998715,
        corridor_family_recall=0.181818,
        frontier_recall_at_budget=0.225755,
        ambiguity_context={
            "od_hard_case_prior": 0.492942,
            "ambiguity_budget_prior": 0.423843,
            "od_ambiguity_support_ratio": 0.686842,
            "od_ambiguity_source_entropy": 0.836641,
            "od_ambiguity_index": 0.403029,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.318304,
        predicted_delta_certificate=0.418502,
        predicted_delta_margin=0.28143,
        predicted_delta_frontier=0.235158,
        metadata={
            "normalized_objective_gap": 0.188203,
            "normalized_mechanism_gap": 0.168712,
            "normalized_overlap_reduction": 0.909091,
            "mean_flip_probability": 0.998715,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.028901,
        predicted_delta_certificate=0.037916,
        predicted_delta_margin=0.024607,
        predicted_delta_frontier=0.0,
        metadata={"structured_refresh_signal": True},
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.101115,
        predicted_delta_certificate=0.129744,
        predicted_delta_margin=0.045641,
        predicted_delta_frontier=0.014615,
        metadata={"near_tie_mass": 0.333333},
    )

    adjusted_actions = voi_module._apply_uncertified_evidence_plateau_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.512821,
        config=VOIConfig(certificate_threshold=0.84),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_resample = next(
        action for action in adjusted_actions if action.kind == "increase_stochastic_samples"
    )
    assert adjusted_refine.q_score == pytest.approx(refine.q_score)
    assert adjusted_resample.q_score == pytest.approx(resample.q_score)
    assert "uncertified_evidence_plateau_preference_applied" not in adjusted_resample.metadata
    assert "uncertified_evidence_plateau_search_discount_applied" not in adjusted_refine.metadata


def test_uncertified_evidence_plateau_preference_can_override_frontier_completion_search_on_narrow_margin_near_tie_row() -> None:
    state = VOIControllerState(
        iteration_index=2,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 9.96, 10.03)},
            {"route_id": "route_c", "objective_vector": (10.11, 10.04, 10.06)},
            {"route_id": "route_d", "objective_vector": (10.24, 10.16, 10.09)},
        ],
        certificate={"route_a": 0.512821, "route_b": 0.287179},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.52101,
                "realized_evidence_uncertainty_delta": 0.030769,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.059888,
                "realized_evidence_uncertainty_delta": -0.061538,
                "realized_productive": True,
            },
        ],
        top_refresh_gain=0.271795,
        top_fragility_mass=0.512821,
        competitor_pressure=1.0,
        support_richness=0.650912,
        prior_support_strength=0.650912,
        ambiguity_pressure=0.834935,
        certificate_margin=0.025641,
        near_tie_mass=0.25,
        search_completeness_score=0.411071,
        search_completeness_gap=0.428929,
        pending_challenger_mass=0.66337,
        best_pending_flip_probability=0.998715,
        corridor_family_recall=0.181818,
        frontier_recall_at_budget=0.225755,
        ambiguity_context={
            "od_hard_case_prior": 0.492942,
            "ambiguity_budget_prior": 0.423843,
            "od_ambiguity_support_ratio": 0.686842,
            "od_ambiguity_source_entropy": 0.836641,
            "od_ambiguity_index": 0.403029,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.394893,
        predicted_delta_certificate=0.477371,
        predicted_delta_margin=0.28753,
        predicted_delta_frontier=0.243921,
        metadata={
            "normalized_objective_gap": 0.188203,
            "normalized_mechanism_gap": 0.168712,
            "normalized_overlap_reduction": 0.909091,
            "mean_flip_probability": 0.998715,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.028901,
        predicted_delta_certificate=0.037916,
        predicted_delta_margin=0.024607,
        predicted_delta_frontier=0.0,
        metadata={"structured_refresh_signal": True},
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.091449,
        predicted_delta_certificate=0.129744,
        predicted_delta_margin=0.045641,
        predicted_delta_frontier=0.014615,
        metadata={"near_tie_mass": 0.25},
    )

    adjusted_actions = voi_module._apply_uncertified_evidence_plateau_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.512821,
        config=VOIConfig(certificate_threshold=0.84),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_resample = next(
        action for action in adjusted_actions if action.kind == "increase_stochastic_samples"
    )
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_evidence_plateau_preference_applied"] is True
    assert adjusted_resample.metadata["uncertified_evidence_plateau_progress_count"] == 2
    assert adjusted_refine.metadata["uncertified_evidence_plateau_search_discount_applied"] is True


def test_certified_supported_hard_case_penalty_preserves_frontier_completion_search() -> None:
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=2,
        frontier=[
            {"route_id": "route_0", "objective_vector": (4081.59, 75.69, 102.582)},
            {"route_id": "route_1", "objective_vector": (4125.0, 76.0, 103.1)},
            {"route_id": "route_2", "objective_vector": (4082.0, 75.7, 102.7)},
        ],
        certificate={"route_2": 0.804124, "route_0": 0.756, "route_1": 0.712},
        winner_id="route_2",
        selected_route_id="route_2",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        ambiguity_context={
            "od_ambiguity_index": 0.531948,
            "od_hard_case_prior": 0.728969,
            "od_ambiguity_prior_strength": 0.728969,
            "od_ambiguity_support_ratio": 0.728969,
            "ambiguity_budget_prior": 0.643808,
            "refc_stress_world_fraction": 0.587629,
        },
        certificate_margin=0.047091,
        search_completeness_score=0.501397,
        search_completeness_gap=0.338603,
        prior_support_strength=0.728969,
        support_richness=0.728969,
        ambiguity_pressure=0.643808,
        pending_challenger_mass=0.731138,
        best_pending_flip_probability=0.999963,
        corridor_family_recall=0.5,
        frontier_recall_at_budget=0.508369,
        top_refresh_gain=0.623711,
        top_fragility_mass=0.427835,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:newport_bridge",
        kind="refine_top1_dccs",
        target="newport_bridge",
        cost_search=1,
        predicted_delta_certificate=0.523545,
        predicted_delta_margin=0.339194,
        predicted_delta_frontier=0.514651,
        q_score=0.53598,
        feasible=True,
        preconditions=("search_budget_available", "dccs_candidate_available"),
        reason="refine_candidate",
        metadata={
            "normalized_objective_gap": 0.514651,
            "normalized_mechanism_gap": 0.127752,
            "normalized_overlap_reduction": 0.666667,
            "mean_flip_probability": 0.999963,
            "search_completeness_gap": 0.338603,
            "search_completeness_score": 0.501397,
            "corridor_family_recall": 0.5,
            "frontier_recall_at_budget": 0.508369,
            "pending_challenger_mass": 0.731138,
            "best_pending_flip_probability": 0.999963,
            "current_certificate": 0.804124,
            "prior_support_strength": 0.728969,
            "support_richness": 0.728969,
            "ambiguity_pressure": 0.643808,
        },
    )

    adjusted = voi_module._certified_supported_hard_case_refine_penalty(
        refine,
        state=state,
        current_certificate=0.804124,
        config=cfg,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        stress_world_fraction=0.587629,
        recent_no_gain_refine_streak=0,
    )
    capped = voi_module._cap_action_certificate_headroom(
        adjusted,
        current_certificate=0.804124,
        config=cfg,
    )

    assert adjusted.metadata["certified_supported_hard_case_frontier_bridge"] is True
    assert "certified_supported_hard_case_penalized" not in adjusted.metadata
    assert adjusted.q_score == pytest.approx(refine.q_score)
    assert capped.q_score > 0.30
    assert capped.metadata["certificate_headroom_cap_applied"] is True


def test_uncertified_resample_recovery_preference_promotes_resample_over_near_threshold_refine() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 9.96, 10.03)},
            {"route_id": "route_c", "objective_vector": (10.11, 10.04, 10.06)},
        ],
        certificate={"route_a": 0.794872, "route_b": 0.205128},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=4,
        remaining_evidence_budget=3,
        top_refresh_gain=0.302564,
        top_fragility_mass=0.794872,
        competitor_pressure=1.0,
        support_richness=0.650912,
        prior_support_strength=0.650912,
        ambiguity_pressure=0.644531,
        ambiguity_context={
            "od_hard_case_prior": 0.492942,
            "ambiguity_budget_prior": 0.423843,
            "od_ambiguity_support_ratio": 0.686842,
            "od_ambiguity_source_entropy": 0.836641,
            "od_ambiguity_index": 0.403029,
        },
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="test",
        q_score=0.189024,
        predicted_delta_certificate=0.205128,
        predicted_delta_margin=0.216003,
        predicted_delta_frontier=0.079643,
        metadata={
            "normalized_objective_gap": 0.043569,
            "normalized_mechanism_gap": 0.148094,
            "normalized_overlap_reduction": 0.909091,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.205128,
            "certificate_headroom_predicted_before_cap": 0.40371,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:terrain",
        kind="refresh_top1_vor",
        target="terrain",
        q_score=0.049528,
        predicted_delta_certificate=0.065328,
        predicted_delta_margin=0.041325,
        predicted_delta_frontier=0.0,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_delta": -0.492308,
            "empirical_refresh_certificate_uplift": 0.0,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.079513,
        predicted_delta_certificate=0.112308,
        predicted_delta_margin=0.039744,
        predicted_delta_frontier=0.014615,
        metadata={"sample_increment": 32},
    )

    adjusted_actions = voi_module._apply_uncertified_resample_recovery_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.794872,
        config=VOIConfig(certificate_threshold=0.84),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_resample = next(
        action for action in adjusted_actions if action.kind == "increase_stochastic_samples"
    )
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_resample_recovery_preference_applied"] is True
    assert adjusted_resample.metadata["uncertified_resample_recovery_structured_negative_refresh"] is True
    assert adjusted_refine.metadata["uncertified_resample_recovery_search_discount_applied"] is True


def test_certified_hard_case_generic_surrogate_strength_does_not_preserve_refine() -> None:
    base_dccs = select_candidates(
        [
            _candidate(
                "cand_generic",
                path=["c1", "c2", "c3", "c4"],
                objective=(16.0, 15.6, 15.4),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=8.5,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=0),
    )
    synthetic_candidate = replace(
        base_dccs.skipped[0],
        objective_gap=0.35,
        mechanism_gap=0.0,
        flip_probability=0.58,
        final_score=0.09,
    )
    dccs = replace(
        base_dccs,
        selected=[],
        skipped=[synthetic_candidate],
        candidate_ledger=[synthetic_candidate],
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "scenario", "vor": 0.01}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.01,
        },
        route_fragility_map={"route_a": {"scenario": 0.02}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"scenario": 0.2}}},
    )
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario", "weather", "toll"],
        stochastic_enabled=False,
        ambiguity_context={
            "od_ambiguity_index": 0.44,
            "od_hard_case_prior": 0.55,
            "od_engine_disagreement_prior": 0.24,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.43,
            "refc_stress_world_fraction": 0.20,
            "is_hard_case": True,
        },
        certificate_margin=0.01,
        search_completeness_score=0.94,
        search_completeness_gap=0.01,
        prior_support_strength=0.79,
        pending_challenger_mass=0.20,
        best_pending_flip_probability=0.26,
        frontier_recall_at_budget=0.92,
        top_refresh_gain=0.01,
        top_fragility_mass=0.02,
        competitor_pressure=0.20,
        near_tie_mass=0.08,
    )
    enriched = enrich_controller_state_for_actioning(state, dccs=dccs, fragility=fragility, config=cfg)
    pending = voi_module._pending_dccs_candidates(dccs)
    best_candidate = voi_module._best_candidate(pending, config=cfg)
    assert best_candidate is not None
    raw_refine = voi_module._apply_search_completeness_bonus(
        voi_module.score_action(
            voi_module._build_refine_action(best_candidate, kind="refine_top1_dccs", config=cfg),
            config=cfg,
        ),
        state=enriched,
        config=cfg,
    )

    assert raw_refine.metadata["normalized_mechanism_gap"] < 0.10
    assert voi_module._search_action_shows_strong_decision_movement(raw_refine, state=enriched) is True
    assert voi_module._refine_action_has_genuine_novel_search_promise(raw_refine, state=enriched) is False

    actions = build_action_menu(enriched, dccs=dccs, fragility=fragility, config=cfg)  # type: ignore[arg-type]

    refine = next(action for action in actions if action.kind == "refine_top1_dccs")
    assert actions[0].kind == "refresh_top1_vor"
    assert refine.metadata["support_rich_certified_first_refine_discount_applied"] is True


def test_certified_support_rich_first_action_keeps_refine_when_search_movement_is_strong() -> None:
    dccs = select_candidates(
        [
            _candidate(
                "cand_strong",
                path=["c1", "c2", "c3", "c4"],
                objective=(16.0, 15.6, 15.4),
                road_mix={"motorway_share": 0.05, "a_road_share": 0.25, "urban_share": 0.70},
                toll_share=0.18,
                terrain_burden=0.24,
                straight_line_km=8.5,
                mechanism={"motorway_share": 0.05, "toll_share": 0.18, "terrain_burden": 0.24},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.6, "a_road_share": 0.3, "urban_share": 0.1},
                toll_share=0.02,
                terrain_burden=0.05,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.6, "toll_share": 0.02, "terrain_burden": 0.05},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=0),
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "scenario", "vor": 0.01}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.01,
        },
        route_fragility_map={"route_a": {"scenario": 0.02}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"scenario": 0.2}}},
    )
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario", "weather", "toll"],
        stochastic_enabled=False,
        ambiguity_context={
            "od_ambiguity_index": 0.44,
            "od_hard_case_prior": 0.55,
            "od_engine_disagreement_prior": 0.24,
            "od_ambiguity_confidence": 0.92,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_prior_strength": 0.58,
            "od_ambiguity_source_entropy": 0.74,
            "ambiguity_budget_prior": 0.43,
            "refc_stress_world_fraction": 0.20,
            "is_hard_case": True,
        },
        certificate_margin=0.01,
        search_completeness_score=0.94,
        search_completeness_gap=0.01,
        prior_support_strength=0.79,
        pending_challenger_mass=0.20,
        best_pending_flip_probability=0.26,
        frontier_recall_at_budget=0.92,
        top_refresh_gain=0.01,
        top_fragility_mass=0.02,
        competitor_pressure=0.20,
        near_tie_mass=0.08,
    )
    enriched = enrich_controller_state_for_actioning(state, dccs=dccs, fragility=fragility, config=cfg)
    pending = voi_module._pending_dccs_candidates(dccs)
    best_candidate = voi_module._best_candidate(pending, config=cfg)
    assert best_candidate is not None
    raw_refine = voi_module._apply_search_completeness_bonus(
        voi_module.score_action(
            voi_module._build_refine_action(best_candidate, kind="refine_top1_dccs", config=cfg),
            config=cfg,
        ),
        state=enriched,
        config=cfg,
    )

    assert voi_module._search_action_shows_strong_decision_movement(raw_refine, state=enriched) is True
    assert voi_module._refine_action_has_genuine_novel_search_promise(raw_refine, state=enriched) is True

    actions = build_action_menu(enriched, dccs=dccs, fragility=fragility, config=cfg)  # type: ignore[arg-type]

    refine = next(action for action in actions if action.kind == "refine_top1_dccs")
    assert actions[0].kind == "refine_top1_dccs"
    assert refine.metadata.get("support_rich_certified_first_refine_discount_applied") is not True


def test_low_support_certified_rows_do_not_get_first_action_refine_discount() -> None:
    dccs = select_candidates(
        [
            _candidate(
                "cand_soft",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        frontier=[
            _candidate(
                "frontier_anchor",
                path=["f1", "f2"],
                objective=(10.0, 10.0, 10.0),
                road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
                toll_share=0.05,
                terrain_burden=0.10,
                straight_line_km=9.0,
                mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
            )
        ],
        config=DCCSConfig(mode="challenger", search_budget=0),
    )
    fragility = types.SimpleNamespace(
        value_of_refresh={
            "ranking": [{"family": "scenario", "vor": 0.01}],
            "top_refresh_family": "scenario",
            "top_refresh_gain": 0.01,
        },
        route_fragility_map={"route_a": {"scenario": 0.02}},
        competitor_fragility_breakdown={"route_a": {"route_b": {"scenario": 0.2}}},
    )
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.02, 10.03, 10.01)},
        ],
        certificate={"route_a": 1.0, "route_b": 1.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        active_evidence_families=["scenario"],
        stochastic_enabled=False,
        ambiguity_context={
            "od_ambiguity_index": 0.12,
            "od_hard_case_prior": 0.10,
            "od_engine_disagreement_prior": 0.06,
            "od_ambiguity_confidence": 0.22,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix": "routing_graph_probe",
            "od_ambiguity_support_ratio": 0.16,
            "od_ambiguity_prior_strength": 0.12,
            "od_ambiguity_source_entropy": 0.12,
            "ambiguity_budget_prior": 0.10,
            "refc_stress_world_fraction": 0.04,
        },
        certificate_margin=0.01,
        search_completeness_score=0.96,
        search_completeness_gap=0.0,
        prior_support_strength=0.14,
        pending_challenger_mass=0.08,
        best_pending_flip_probability=0.10,
        frontier_recall_at_budget=0.98,
        top_refresh_gain=0.01,
        top_fragility_mass=0.02,
        competitor_pressure=0.20,
        near_tie_mass=0.06,
    )

    actions = build_action_menu(
        enrich_controller_state_for_actioning(
            state,
            dccs=dccs,
            fragility=fragility,  # type: ignore[arg-type]
            config=cfg,
        ),
        dccs=dccs,
        fragility=fragility,  # type: ignore[arg-type]
        config=cfg,
    )

    assert actions[0].kind != "refresh_top1_vor"
    assert all(
        action.metadata.get("support_rich_certified_first_refine_discount_applied") is not True
        for action in actions
    )


def test_topk_refine_action_aggregates_real_candidate_ids() -> None:
    candidates = [
        _candidate(
            "cand_fast",
            path=["c1", "c2", "c3"],
            objective=(9.0, 9.2, 9.1),
            road_mix={"motorway_share": 0.7, "a_road_share": 0.2, "urban_share": 0.1},
            toll_share=0.02,
            terrain_burden=0.05,
            straight_line_km=8.8,
            mechanism={"motorway_share": 0.7, "toll_share": 0.02, "terrain_burden": 0.05},
        ),
        _candidate(
            "cand_mid",
            path=["m1", "m2", "m3"],
            objective=(9.4, 9.5, 9.6),
            road_mix={"motorway_share": 0.4, "a_road_share": 0.4, "urban_share": 0.2},
            toll_share=0.04,
            terrain_burden=0.08,
            straight_line_km=8.5,
            mechanism={"motorway_share": 0.4, "toll_share": 0.04, "terrain_burden": 0.08},
        ),
        _candidate(
            "cand_alt",
            path=["a1", "a2", "a3"],
            objective=(9.5, 9.7, 9.4),
            road_mix={"motorway_share": 0.3, "a_road_share": 0.5, "urban_share": 0.2},
            toll_share=0.03,
            terrain_burden=0.07,
            straight_line_km=8.4,
            mechanism={"motorway_share": 0.3, "toll_share": 0.03, "terrain_burden": 0.07},
        ),
    ]
    dccs = select_candidates(candidates, config=DCCSConfig(mode="challenger", search_budget=1))
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.25, "route_b": 0.23},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=1,
        action_trace=[],
        state_trace=[],
        active_evidence_families=["scenario"],
        near_tie_mass=0.2,
    )

    actions = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig(top_k_refine=2))
    topk = next(action for action in actions if action.kind == "refine_topk_dccs")
    assert topk.metadata["top_k"] == 2
    assert len(topk.metadata["candidate_ids"]) == 2
    assert topk.target == topk.metadata["cohort_signature"]
    assert topk.metadata["mean_flip_probability"] > 0.0
    assert 0.0 <= topk.metadata["normalized_objective_gap"] <= 1.0
    assert 0.0 <= topk.metadata["normalized_mechanism_gap"] <= 1.0


def test_live_voi_refine_resolution_prefers_chosen_action_candidate_ids() -> None:
    dccs = _dccs_result()
    fallback_candidate_id = dccs.selected[0].candidate_id
    requested_candidate_id = dccs.skipped[0].candidate_id

    assert requested_candidate_id != fallback_candidate_id

    selected_records, execution_diag = main_module._resolve_voi_refine_selected_records(
        action_metadata={"top_k": 1, "candidate_ids": [requested_candidate_id]},
        current_records=[*dccs.selected, *dccs.skipped],
        fallback_records=dccs.selected,
    )

    assert [record.candidate_id for record in selected_records] == [requested_candidate_id]
    assert execution_diag["requested_candidate_ids"] == [requested_candidate_id]
    assert execution_diag["executed_candidate_ids"] == [requested_candidate_id]
    assert execution_diag["execution_used_candidate_metadata"] is True
    assert execution_diag["execution_matches_requested_candidate_ids"] is True
    assert execution_diag["execution_candidate_resolution_source"] == "action_metadata"


def test_live_voi_refine_resolution_falls_back_without_candidate_metadata() -> None:
    dccs = _dccs_result()
    fallback_candidate_id = dccs.selected[0].candidate_id

    selected_records, execution_diag = main_module._resolve_voi_refine_selected_records(
        action_metadata={"top_k": 1},
        current_records=[*dccs.selected, *dccs.skipped],
        fallback_records=dccs.selected,
    )

    assert [record.candidate_id for record in selected_records] == [fallback_candidate_id]
    assert execution_diag["requested_candidate_ids"] == []
    assert execution_diag["executed_candidate_ids"] == [fallback_candidate_id]
    assert execution_diag["execution_used_candidate_metadata"] is False
    assert execution_diag["execution_matches_requested_candidate_ids"] is None
    assert execution_diag["execution_candidate_resolution_source"] == "fallback_top_k"


def test_controller_stop_precedence_is_certified_then_budget_then_threshold() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}]

    certified = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.90,
        selected_route_id="route_a",
        config=VOIConfig(certificate_threshold=0.80, search_completeness_threshold=0.0, search_budget=1, evidence_budget=1),
    )
    exhausted = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        selected_route_id="route_a",
        config=VOIConfig(search_completeness_threshold=0.0, search_budget=0, evidence_budget=0),
    )
    thresholded = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        selected_route_id="route_a",
        config=VOIConfig(search_completeness_threshold=0.0, stop_threshold=10.0, search_budget=1, evidence_budget=1),
    )

    assert certified.stop_reason == "certified"
    assert exhausted.stop_reason == "budget_exhausted"
    assert thresholded.stop_reason == "no_action_worth_it"


def test_controller_emits_audit_ready_stop_certificate() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}]

    def _refine(state: VOIControllerState, action):
        return replace(
            state,
            iteration_index=state.iteration_index + 1,
            remaining_search_budget=max(0, state.remaining_search_budget - action.cost_search),
            certificate={"route_a": 0.92},
        )

    stop_certificate = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        certificate_map={"route_a": 0.20, "route_b": 0.18},
        selected_route_id="route_a",
        config=VOIConfig(certificate_threshold=0.80, search_completeness_threshold=0.0, search_budget=1, evidence_budget=0),
        hooks=VOIActionHooks(refine=_refine),
    )

    assert stop_certificate.final_winner_route_id == "route_a"
    assert stop_certificate.final_strict_frontier_size == 1
    assert stop_certificate.certified is True
    assert stop_certificate.search_budget_used == 1
    assert stop_certificate.action_trace
    assert stop_certificate.state_trace
    assert stop_certificate.state_trace[0]["frontier_route_ids"] == ["route_a"]
    assert stop_certificate.controller_state is not None
    assert stop_certificate.controller_state["used_search_budget"] == 1
    assert stop_certificate.controller_state["certificate_margin"] > 0.0
    assert stop_certificate.iteration_count == 1


def test_controller_stops_immediately_after_certified_no_gain_action() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}]

    def _refine_no_progress(state: VOIControllerState, action):
        return replace(
            state,
            iteration_index=state.iteration_index + 1,
            remaining_search_budget=max(0, state.remaining_search_budget - action.cost_search),
        )

    stop_certificate = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.92,
        selected_route_id="route_a",
        ambiguity_context={
            "od_ambiguity_index": 0.42,
            "od_ambiguity_confidence": 0.72,
            "od_ambiguity_source_count": 1,
        },
        config=VOIConfig(
            certificate_threshold=0.80,
            search_completeness_threshold=0.99,
            search_budget=1,
            evidence_budget=0,
        ),
        hooks=VOIActionHooks(refine=_refine_no_progress),
    )

    assert stop_certificate.certified is True
    assert stop_certificate.stop_reason == "certified"
    assert stop_certificate.iteration_count == 1
    assert len(stop_certificate.action_trace) == 1


def test_controller_state_tracks_margin_and_budget_usage_deterministically() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}]

    def _refresh(state: VOIControllerState, action):
        return replace(
            state,
            iteration_index=state.iteration_index + 1,
            remaining_evidence_budget=max(0, state.remaining_evidence_budget - action.cost_evidence),
            certificate={"route_a": 0.31, "route_b": 0.10},
        )

    stop_certificate = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        certificate_map={"route_a": 0.20, "route_b": 0.10},
        selected_route_id="route_a",
        config=VOIConfig(certificate_threshold=0.80, search_completeness_threshold=0.0, stop_threshold=0.0, search_budget=0, evidence_budget=1),
        hooks=VOIActionHooks(refresh=_refresh),
    )

    assert stop_certificate.controller_state is not None
    assert stop_certificate.controller_state["used_evidence_budget"] == 1
    assert abs(stop_certificate.controller_state["certificate_margin"] - 0.21) < 1e-9


def test_controller_fails_closed_when_action_execution_hooks_are_missing() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}]

    stop_certificate = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        certificate_map={"route_a": 0.20, "route_b": 0.18},
        selected_route_id="route_a",
        config=VOIConfig(certificate_threshold=0.80, stop_threshold=0.0, search_budget=1, evidence_budget=1),
        hooks=None,
    )

    assert stop_certificate.stop_reason == "error_missing_action_hooks"
    assert stop_certificate.action_trace
    assert stop_certificate.best_rejected_action is not None


def test_controller_normalizes_hook_side_effects_for_iteration_and_budget() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}]

    def _refresh_without_budget_bookkeeping(state: VOIControllerState, _action):
        return replace(
            state,
            certificate={"route_a": 0.81, "route_b": 0.10},
        )

    stop_certificate = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        certificate_map={"route_a": 0.20, "route_b": 0.10},
        selected_route_id="route_a",
        config=VOIConfig(certificate_threshold=0.80, search_completeness_threshold=0.0, stop_threshold=0.0, search_budget=0, evidence_budget=1),
        hooks=VOIActionHooks(refresh=_refresh_without_budget_bookkeeping),
    )

    assert stop_certificate.certified is True
    assert stop_certificate.iteration_count == 1
    assert stop_certificate.evidence_budget_used == 1


def test_controller_stops_when_certified_and_residual_search_uncertainty_is_weak() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}]

    stop_certificate = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.95,
        certificate_map={"route_a": 0.95, "route_b": 0.10},
        selected_route_id="route_a",
        config=VOIConfig(
            certificate_threshold=0.80,
            search_completeness_threshold=0.95,
            search_budget=1,
            evidence_budget=0,
        ),
    )

    assert stop_certificate.certified is True
    assert stop_certificate.stop_reason == "certified"
    assert stop_certificate.controller_state is not None
    assert stop_certificate.controller_state["search_completeness_score"] < 0.95


def test_saturated_certified_supported_hard_case_without_refresh_stops_search_only_reveal() -> None:
    cfg = VOIConfig(certificate_threshold=0.8)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.1, 10.0, 10.0)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[],
        ambiguity_context={
            "od_hard_case_prior": 0.692693,
            "ambiguity_budget_prior": 0.453718,
            "od_ambiguity_support_ratio": 0.587187,
            "od_ambiguity_source_entropy": 0.845351,
        },
        certificate_margin=0.650473,
        search_completeness_score=0.473146,
        search_completeness_gap=0.366854,
        prior_support_strength=0.712113,
        support_richness=0.712113,
        ambiguity_pressure=0.601673,
        pending_challenger_mass=0.650685,
        best_pending_flip_probability=0.999986,
        frontier_recall_at_budget=0.279421,
        near_tie_mass=0.0,
        top_refresh_gain=0.942857,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_topk_dccs:test",
        kind="refine_topk_dccs",
        target="test",
        q_score=0.029021,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.190921,
        predicted_delta_frontier=0.068742,
        metadata={
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.546998,
            "normalized_overlap_reduction": 0.906926,
        },
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_saturated_certified_search_without_certificate_upside(
        [refine, stop],
        state=state,
        current_certificate=1.0,
        config=cfg,
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_controller_keeps_searching_when_supported_pending_flip_signal_is_credible() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [
        {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
        {"route_id": "route_b", "objective_vector": (10.08, 10.05, 10.02)},
    ]

    stop_certificate = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.91,
        certificate_map={"route_a": 0.91, "route_b": 0.10},
        selected_route_id="route_a",
        ambiguity_context={
            "od_ambiguity_index": 0.68,
            "od_ambiguity_confidence": 0.95,
            "od_ambiguity_source_count": 4,
            "od_ambiguity_source_mix": "historical_results_bootstrap,repo_local_geometry_backfill,routing_graph_probe,engine_augmented_probe",
        },
        config=VOIConfig(
            certificate_threshold=0.80,
            search_completeness_threshold=0.95,
            stop_threshold=10.0,
            search_budget=1,
            evidence_budget=0,
        ),
        hooks=VOIActionHooks(),
    )

    assert stop_certificate.certified is False
    assert stop_certificate.stop_reason == "search_incomplete_no_action_worth_it"
    assert stop_certificate.controller_state is not None
    assert stop_certificate.controller_state["search_completeness_score"] < 0.95


def test_controller_reports_iteration_cap_distinct_from_budget_exhaustion() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}]

    def _refresh_stays_uncertain(state: VOIControllerState, _action):
        return replace(
            state,
            certificate={"route_a": 0.25, "route_b": 0.22},
        )

    stop_certificate = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        certificate_map={"route_a": 0.20, "route_b": 0.18},
        selected_route_id="route_a",
        config=VOIConfig(
            certificate_threshold=0.90,
            stop_threshold=0.0,
            search_budget=0,
            evidence_budget=3,
            max_iterations=1,
        ),
        hooks=VOIActionHooks(refresh=_refresh_stays_uncertain),
    )

    assert stop_certificate.stop_reason == "iteration_cap_reached"
    assert stop_certificate.evidence_budget_remaining == 2


def test_search_completeness_respects_supported_ambiguity_prior_strength() -> None:
    dccs = _dccs_result()
    base_state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.1, 10.0, 10.0)},
        ],
        certificate={"route_a": 0.88, "route_b": 0.12},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=0,
        action_trace=[],
        near_tie_mass=0.0,
        certificate_margin=0.12,
        ambiguity_context={
            "od_ambiguity_index": 0.60,
            "od_engine_disagreement_prior": 0.42,
            "od_hard_case_prior": 0.55,
            "od_corridor_family_count": 3,
        },
    )

    low_support = compute_search_completeness_metrics(
        replace(
            base_state,
            ambiguity_context={
                **base_state.ambiguity_context,
                "od_ambiguity_confidence": 0.20,
                "od_ambiguity_source_count": 1,
            },
        ),
        dccs=dccs,
        config=VOIConfig(search_completeness_threshold=0.92),
    )
    high_support = compute_search_completeness_metrics(
        replace(
            base_state,
            ambiguity_context={
                **base_state.ambiguity_context,
                "od_ambiguity_confidence": 0.95,
                "od_ambiguity_source_count": 4,
                "od_ambiguity_source_mix": "historical_results_bootstrap,repo_local_geometry_backfill,baseline_disagreement",
            },
        ),
        dccs=dccs,
        config=VOIConfig(search_completeness_threshold=0.92),
    )

    assert high_support["prior_support_strength"] > low_support["prior_support_strength"]
    assert high_support["search_completeness_score"] <= low_support["search_completeness_score"]


def test_uncertified_post_evidence_resample_preference_promotes_resample_after_meaningful_evidence_lift() -> None:
    state = VOIControllerState(
        iteration_index=3,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 10.04, 10.02)},
            {"route_id": "route_c", "objective_vector": (10.2, 10.1, 10.0)},
            {"route_id": "route_d", "objective_vector": (10.35, 10.22, 10.11)},
        ],
        certificate={"route_a": 0.581498, "route_b": 0.36, "route_c": 0.05, "route_d": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.1,
                "realized_runner_up_gap_delta": -0.5,
                "realized_evidence_uncertainty_delta": 0.03,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.05,
                "realized_runner_up_gap_delta": -0.06,
                "realized_evidence_uncertainty_delta": -0.06,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "increase_stochastic_samples"},
                "realized_certificate_delta": 0.068677,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.087631,
                "realized_productive": True,
            },
        ],
        active_evidence_families=["fuel", "scenario", "weather"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.65,
            "od_hard_case_prior": 0.67,
            "od_engine_disagreement_prior": 0.43,
            "od_ambiguity_confidence": 0.88,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.74,
            "od_ambiguity_prior_strength": 0.66,
            "od_ambiguity_source_entropy": 0.79,
            "ambiguity_budget_prior": 0.68,
            "refc_stress_world_fraction": 0.418502,
        },
        certificate_margin=0.221498,
        search_completeness_score=0.467012,
        search_completeness_gap=0.372988,
        prior_support_strength=0.650912,
        support_richness=0.650912,
        ambiguity_pressure=0.767246,
        pending_challenger_mass=0.66337,
        best_pending_flip_probability=0.998715,
        corridor_family_recall=0.181818,
        frontier_recall_at_budget=0.225755,
        top_refresh_gain=0.290749,
        top_fragility_mass=0.581498,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.356733,
        predicted_delta_certificate=0.418502,
        predicted_delta_margin=0.28143,
        predicted_delta_frontier=0.235158,
        metadata={
            "normalized_objective_gap": 0.188203,
            "normalized_mechanism_gap": 0.168712,
            "normalized_overlap_reduction": 0.909091,
            "mean_flip_probability": 0.998715,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.418502,
            "certificate_headroom_predicted_before_cap": 0.466903,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.062174,
        cost_evidence=1,
        predicted_delta_certificate=0.08837,
        predicted_delta_margin=0.029075,
        predicted_delta_frontier=0.012555,
        metadata={"sample_increment": 32},
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.033416,
        cost_evidence=1,
        predicted_delta_certificate=0.04391,
        predicted_delta_margin=0.028281,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_delta": -0.290749,
        },
    )

    adjusted = voi_module._apply_uncertified_post_evidence_resample_preference(
        [refine, resample, refresh],
        state=state,
        current_certificate=0.581498,
        config=VOIConfig(certificate_threshold=0.67),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    adjusted_resample = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_post_evidence_resample_preference_applied"] is True
    assert adjusted_refine.metadata["uncertified_post_evidence_resample_search_discount_applied"] is True


def test_uncertified_last_search_token_resample_preference_promotes_resample_after_no_gain_refine_stall() -> None:
    state = VOIControllerState(
        iteration_index=3,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.021211, 10.0, 10.0)},
            {"route_id": "route_c", "objective_vector": (10.20, 10.08, 10.02)},
            {"route_id": "route_d", "objective_vector": (10.35, 10.18, 10.06)},
            {"route_id": "route_e", "objective_vector": (10.42, 10.25, 10.11)},
        ],
        certificate={"route_a": 0.512821, "route_b": 0.49161, "route_c": 0.0, "route_d": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.52101,
                "realized_evidence_uncertainty_delta": 0.030769,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.030769,
                "realized_productive": True,
            },
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            },
        ],
        active_evidence_families=["fuel", "scenario", "weather"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.66,
            "od_hard_case_prior": 0.67,
            "od_engine_disagreement_prior": 0.43,
            "od_ambiguity_confidence": 0.88,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.74,
            "od_ambiguity_prior_strength": 0.66,
            "od_ambiguity_source_entropy": 0.79,
            "ambiguity_budget_prior": 0.68,
            "refc_stress_world_fraction": 0.487179,
        },
        certificate_margin=0.021211,
        search_completeness_score=0.54561,
        search_completeness_gap=0.45439,
        prior_support_strength=0.650912,
        support_richness=0.650912,
        ambiguity_pressure=0.860184,
        pending_challenger_mass=0.662304,
        best_pending_flip_probability=0.998623,
        corridor_family_recall=0.090909,
        frontier_recall_at_budget=0.128104,
        top_refresh_gain=0.271795,
        top_fragility_mass=0.512821,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.36181,
        predicted_delta_certificate=0.480104,
        predicted_delta_margin=0.310789,
        predicted_delta_frontier=0.298764,
        metadata={
            "normalized_objective_gap": 0.238987,
            "normalized_mechanism_gap": 0.063474,
            "normalized_overlap_reduction": 0.909091,
            "mean_flip_probability": 0.998623,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.268901,
        cost_evidence=1,
        predicted_delta_certificate=0.037916,
        predicted_delta_margin=0.024607,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_delta": -0.241026,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.091449,
        cost_evidence=1,
        predicted_delta_certificate=0.129744,
        predicted_delta_margin=0.045641,
        predicted_delta_frontier=0.014615,
        metadata={"sample_increment": 32},
    )

    adjusted = voi_module._apply_uncertified_last_search_token_resample_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.512821,
        config=VOIConfig(certificate_threshold=0.84),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    adjusted_resample = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_last_search_token_resample_preference_applied"] is True
    assert adjusted_resample.metadata["uncertified_last_search_token_resample_recent_no_gain_refine_streak"] == 1
    assert adjusted_refine.metadata["uncertified_last_search_token_resample_search_discount_applied"] is True


def test_uncertified_last_search_token_resample_preference_allows_direct_fallback_evidence_recovery() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 10.04, 10.02)},
            {"route_id": "route_c", "objective_vector": (10.2, 10.1, 10.0)},
            {"route_id": "route_d", "objective_vector": (10.35, 10.22, 10.11)},
        ],
        certificate={"route_a": 0.601504, "route_b": 0.398496, "route_c": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        active_evidence_families=["stochastic", "scenario"],
        stochastic_enabled=True,
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "od_ambiguity_index": 0.61,
            "od_hard_case_prior": 0.63,
            "od_ambiguity_support_ratio": 0.73,
            "od_ambiguity_source_entropy": 0.84,
            "ambiguity_budget_prior": 0.58,
            "refc_stress_world_fraction": 0.40,
        },
        certificate_margin=0.203008,
        search_completeness_score=0.456479,
        search_completeness_gap=0.383521,
        prior_support_strength=0.728969,
        support_richness=0.728969,
        ambiguity_pressure=0.6805,
        pending_challenger_mass=0.709747,
        best_pending_flip_probability=0.999927,
        corridor_family_recall=1.0,
        frontier_recall_at_budget=0.174765,
        top_refresh_gain=0.278195,
        top_fragility_mass=0.601504,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.37217,
        predicted_delta_certificate=0.398496,
        predicted_delta_margin=0.405863,
        predicted_delta_frontier=0.54337,
        metadata={
            "normalized_objective_gap": 0.491795,
            "normalized_mechanism_gap": 0.112269,
            "normalized_overlap_reduction": 0.904762,
            "mean_flip_probability": 0.999927,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:stochastic",
        kind="refresh_top1_vor",
        target="stochastic",
        q_score=0.275786,
        cost_evidence=1,
        predicted_delta_certificate=0.047044,
        predicted_delta_margin=0.030238,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_delta": -0.278195,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.062094,
        cost_evidence=1,
        predicted_delta_certificate=0.08797,
        predicted_delta_margin=0.030075,
        predicted_delta_frontier=0.011955,
        metadata={"sample_increment": 32},
    )

    adjusted = voi_module._apply_uncertified_last_search_token_resample_preference(
        [refine, refresh, resample],
        state=state,
        current_certificate=0.601504,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    adjusted_resample = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    assert adjusted_resample.q_score > adjusted_refine.q_score
    assert adjusted_resample.metadata["uncertified_last_search_token_resample_preference_applied"] is True
    assert adjusted_refine.metadata["uncertified_last_search_token_resample_search_discount_applied"] is True


def test_uncertified_last_search_token_resample_preference_does_not_override_productive_search() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.40, 10.10, 10.05)},
            {"route_id": "route_c", "objective_vector": (10.58, 10.22, 10.11)},
        ],
        certificate={"route_a": 0.650473, "route_b": 0.265195, "route_c": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=3,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.201685,
                "realized_frontier_gain": 1.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": -0.265195,
                "realized_evidence_uncertainty_delta": 0.0125,
                "realized_productive": True,
            },
        ],
        active_evidence_families=["fuel", "scenario", "weather"],
        stochastic_enabled=True,
        ambiguity_context={
            "od_ambiguity_index": 0.58,
            "od_hard_case_prior": 0.59,
            "od_engine_disagreement_prior": 0.41,
            "od_ambiguity_confidence": 0.88,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.74,
            "od_ambiguity_prior_strength": 0.60,
            "od_ambiguity_source_entropy": 0.79,
            "ambiguity_budget_prior": 0.58,
            "refc_stress_world_fraction": 0.40,
        },
        certificate_margin=0.385278,
        search_completeness_score=0.581818,
        search_completeness_gap=0.418182,
        prior_support_strength=0.650912,
        support_richness=0.650912,
        ambiguity_pressure=0.71,
        pending_challenger_mass=0.58,
        best_pending_flip_probability=0.96,
        corridor_family_recall=0.40,
        frontier_recall_at_budget=0.38,
        top_refresh_gain=0.21,
        top_fragility_mass=0.650473,
        competitor_pressure=0.89,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.270156,
        predicted_delta_certificate=0.312,
        predicted_delta_margin=0.14,
        predicted_delta_frontier=0.18,
        metadata={
            "normalized_objective_gap": 0.22,
            "normalized_mechanism_gap": 0.10,
            "normalized_overlap_reduction": 0.90,
            "mean_flip_probability": 0.99,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.062671,
        cost_evidence=1,
        predicted_delta_certificate=0.08392,
        predicted_delta_margin=0.021,
        predicted_delta_frontier=0.014,
        metadata={"sample_increment": 32},
    )

    adjusted = voi_module._apply_uncertified_last_search_token_resample_preference(
        [refine, resample],
        state=state,
        current_certificate=0.650473,
        config=VOIConfig(certificate_threshold=0.80),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
    )

    adjusted_refine = next(action for action in adjusted if action.kind == "refine_top1_dccs")
    adjusted_resample = next(action for action in adjusted if action.kind == "increase_stochastic_samples")
    assert adjusted_refine.q_score == pytest.approx(refine.q_score)
    assert adjusted_resample.q_score == pytest.approx(resample.q_score)
    assert "uncertified_last_search_token_resample_preference_applied" not in adjusted_resample.metadata
    assert "uncertified_last_search_token_resample_search_discount_applied" not in adjusted_refine.metadata


def test_saturated_certified_zero_headroom_search_probe_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.16, 10.04, 10.02)},
            {"route_id": "route_c", "objective_vector": (10.2, 10.1, 10.04)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.8, "route_c": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        action_trace=[],
        certificate_margin=0.2,
        search_completeness_score=0.607304,
        search_completeness_gap=0.232696,
        prior_support_strength=0.728969,
        support_richness=0.728969,
        ambiguity_pressure=0.525849,
        pending_challenger_mass=0.619623,
        best_pending_flip_probability=0.987514,
        corridor_family_recall=1.0,
        frontier_recall_at_budget=0.677168,
        top_refresh_gain=0.646154,
        top_fragility_mass=0.692308,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.0425,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.17,
        predicted_delta_frontier=0.0,
        metadata={
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.226606,
            "mean_flip_probability": 0.987514,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.0,
            "certificate_headroom_predicted_before_cap": 0.379621,
        },
    )
    stop = voi_module.VOIAction(
        action_id="stop",
        kind="stop",
        target="stop",
        q_score=0.0,
    )

    filtered = voi_module._suppress_saturated_certified_zero_headroom_search_probe(
        [refine, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_saturated_certified_zero_headroom_search_probe_preserves_refresh_and_drops_search() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.16, 10.04, 10.02)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        certificate_margin=1.0,
        search_completeness_score=0.465621,
        search_completeness_gap=0.374379,
        prior_support_strength=0.408274,
        support_richness=0.652031,
        ambiguity_pressure=0.696134,
        pending_challenger_mass=0.67047,
        best_pending_flip_probability=0.999244,
        corridor_family_recall=0.125,
        frontier_recall_at_budget=0.135397,
        top_refresh_gain=0.904192,
        top_fragility_mass=1.0,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.072767,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.225517,
        predicted_delta_frontier=0.109248,
        metadata={
            "normalized_objective_gap": 0.109248,
            "normalized_mechanism_gap": 0.358178,
            "mean_flip_probability": 0.999244,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.0,
            "certificate_headroom_predicted_before_cap": 0.436237,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.054256,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.01,
        predicted_delta_frontier=0.0,
        metadata={},
    )
    stop = voi_module.VOIAction(
        action_id="stop",
        kind="stop",
        target="stop",
        q_score=0.0,
    )

    filtered = voi_module._suppress_saturated_certified_zero_headroom_search_probe(
        [refine, refresh, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert [action.kind for action in filtered] == ["refresh_top1_vor", "stop"]


def test_saturated_certified_zero_headroom_search_probe_drops_weak_bridge_search() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.16, 10.04, 10.02)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        action_trace=[],
        certificate_margin=1.0,
        search_completeness_score=0.619986,
        search_completeness_gap=0.220014,
        prior_support_strength=0.532858,
        support_richness=0.532858,
        ambiguity_pressure=0.575039,
        pending_challenger_mass=0.68779,
        best_pending_flip_probability=0.999586,
        corridor_family_recall=1.0,
        frontier_recall_at_budget=0.254848,
        top_refresh_gain=0.15,
        top_fragility_mass=0.15,
        competitor_pressure=0.834701,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.074848,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.228643,
        predicted_delta_frontier=0.117917,
        metadata={
            "normalized_objective_gap": 0.108192,
            "normalized_mechanism_gap": 0.426721,
            "mean_flip_probability": 0.999561,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.0,
            "certificate_headroom_predicted_before_cap": 0.451172,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:terrain",
        kind="refresh_top1_vor",
        target="terrain",
        q_score=0.000889,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.003554,
        predicted_delta_frontier=0.0,
        metadata={},
    )
    stop = voi_module.VOIAction(
        action_id="stop",
        kind="stop",
        target="stop",
        q_score=0.0,
    )

    filtered = voi_module._suppress_saturated_certified_zero_headroom_search_probe(
        [refine, refresh, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.80),
        certified_frontier_fill_bridge=True,
    )

    assert [action.kind for action in filtered] == ["refresh_top1_vor", "stop"]


def test_saturated_certified_zero_headroom_search_probe_drops_support_rich_direct_fallback_reopen() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.16, 10.04, 10.02)},
            {"route_id": "route_c", "objective_vector": (10.21, 10.09, 10.05)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "realized_certificate_delta": 0.55,
                "realized_frontier_gain": 1.0,
                "realized_productive": True,
            }
        ],
        ambiguity_context={
            "selected_candidate_source_stage": "direct_k_raw_fallback",
            "selected_final_route_source_stage": "osrm_refined",
            "od_ambiguity_support_ratio": 0.71,
            "od_ambiguity_source_entropy": 0.84,
            "od_hard_case_prior": 0.59,
            "od_ambiguity_prior_strength": 0.36,
            "ambiguity_budget_prior": 0.36,
        },
        certificate_margin=1.0,
        search_completeness_score=0.619986,
        search_completeness_gap=0.220014,
        prior_support_strength=0.71,
        support_richness=0.71,
        ambiguity_pressure=0.61,
        pending_challenger_mass=0.68779,
        best_pending_flip_probability=0.999586,
        corridor_family_recall=1.0,
        frontier_recall_at_budget=0.254848,
        top_refresh_gain=0.15,
        top_fragility_mass=0.15,
        competitor_pressure=0.834701,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.090028,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.19456,
        predicted_delta_frontier=0.179161,
        metadata={
            "normalized_objective_gap": 0.197429,
            "normalized_mechanism_gap": 0.210718,
            "mean_flip_probability": 0.99913,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.0,
            "certificate_headroom_predicted_before_cap": 0.388608,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.043295,
        predicted_delta_certificate=0.01,
        predicted_delta_margin=0.01,
        predicted_delta_frontier=0.0,
        metadata={},
    )
    stop = voi_module.VOIAction(
        action_id="stop",
        kind="stop",
        target="stop",
        q_score=0.0,
    )

    filtered = voi_module._suppress_saturated_certified_zero_headroom_search_probe(
        [refine, refresh, stop],
        state=state,
        current_certificate=1.0,
        config=VOIConfig(certificate_threshold=0.81),
    )

    assert [action.kind for action in filtered] == ["refresh_top1_vor", "stop"]


def test_certified_single_frontier_zero_signal_search_churn_prefers_stop() -> None:
    cfg = VOIConfig(certificate_threshold=0.78)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.79},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        certificate_margin=0.79,
        search_completeness_score=0.462809,
        search_completeness_gap=0.377191,
        prior_support_strength=0.645344,
        support_richness=0.645344,
        ambiguity_pressure=0.677396,
        pending_challenger_mass=0.636303,
        best_pending_flip_probability=0.997643,
        corridor_family_recall=0.090909,
        frontier_recall_at_budget=0.144709,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.215273,
        predicted_delta_certificate=0.21,
        predicted_delta_margin=0.254886,
        predicted_delta_frontier=0.170346,
        metadata={
            "normalized_objective_gap": 0.124166,
            "normalized_mechanism_gap": 0.175055,
            "normalized_overlap_reduction": 0.909091,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.21,
            "certificate_headroom_predicted_before_cap": 0.436181,
            "corridor_family_recall": 0.090909,
            "frontier_recall_at_budget": 0.144709,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:none",
        kind="refresh_top1_vor",
        target="None",
        q_score=0.0,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.0,
        predicted_delta_frontier=0.0,
        metadata={},
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_certified_single_frontier_zero_signal_search_churn(
        [refine, refresh, stop],
        state=state,
        current_certificate=0.79,
        config=cfg,
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_near_certified_single_frontier_zero_signal_search_churn_prefers_stop() -> None:
    cfg = VOIConfig(certificate_threshold=0.80)
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.79},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=2,
        action_trace=[],
        certificate_margin=0.79,
        search_completeness_score=0.462809,
        search_completeness_gap=0.377191,
        prior_support_strength=0.408084,
        support_richness=0.408084,
        ambiguity_pressure=0.241505,
        pending_challenger_mass=0.636303,
        best_pending_flip_probability=0.997643,
        corridor_family_recall=0.090909,
        frontier_recall_at_budget=0.144709,
        top_refresh_gain=0.0,
        top_fragility_mass=0.0,
        competitor_pressure=0.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.215273,
        predicted_delta_certificate=0.21,
        predicted_delta_margin=0.254886,
        predicted_delta_frontier=0.170346,
        metadata={
            "normalized_objective_gap": 0.124166,
            "normalized_mechanism_gap": 0.175055,
            "normalized_overlap_reduction": 0.909091,
            "certificate_headroom_cap_applied": True,
            "certificate_headroom_remaining": 0.21,
            "certificate_headroom_predicted_before_cap": 0.436181,
            "corridor_family_recall": 0.090909,
            "frontier_recall_at_budget": 0.144709,
        },
    )
    refresh = voi_module.VOIAction(
        action_id="refresh:none",
        kind="refresh_top1_vor",
        target="None",
        q_score=0.0,
        predicted_delta_certificate=0.0,
        predicted_delta_margin=0.0,
        predicted_delta_frontier=0.0,
        metadata={},
    )
    stop = voi_module.VOIAction(action_id="stop", kind="stop", target="stop", q_score=0.0)

    filtered = voi_module._suppress_certified_single_frontier_zero_signal_search_churn(
        [refine, refresh, stop],
        state=state,
        current_certificate=0.79,
        config=cfg,
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_post_nonproductive_uncertified_evidence_plateau_churn_prefers_stop() -> None:
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.4, "route_b": 0.6},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=1,
        action_trace=[
            {
                "chosen_action": {
                    "kind": "increase_stochastic_samples",
                    "metadata": {
                        "near_tie_mass": 1.0,
                        "stress_world_fraction": 0.6,
                        "top_fragility_mass": 0.4,
                    },
                },
                "realized_certificate_delta": 0.0,
                "realized_frontier_gain": 0.0,
                "realized_selected_route_improvement": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
                "realized_productive": False,
            }
        ],
        ambiguity_context={
            "od_ambiguity_index": 0.02,
            "od_ambiguity_prior_strength": 0.02,
            "od_hard_case_prior": 0.370399,
            "ambiguity_budget_prior": 0.02,
            "od_objective_spread": 0.0,
            "od_ambiguity_margin_pressure": 0.0,
        },
        certificate_margin=0.0,
        search_completeness_score=0.411278,
        search_completeness_gap=0.428722,
        prior_support_strength=0.312407,
        support_richness=0.553078,
        ambiguity_pressure=0.836492,
        pending_challenger_mass=0.625013,
        best_pending_flip_probability=0.997052,
        frontier_recall_at_budget=0.0,
        near_tie_mass=1.0,
        top_refresh_gain=0.375,
        top_fragility_mass=0.4,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.327504,
        predicted_delta_certificate=0.433182,
        predicted_delta_margin=0.219478,
        predicted_delta_frontier=0.084832,
        metadata={
            "normalized_objective_gap": 0.042138,
            "normalized_mechanism_gap": 0.219664,
        },
    )
    resample = voi_module.VOIAction(
        action_id="resample:stochastic",
        kind="increase_stochastic_samples",
        target="stochastic",
        q_score=0.1807,
        predicted_delta_certificate=0.252,
        predicted_delta_margin=0.1,
        predicted_delta_frontier=0.03,
        metadata={},
    )
    stop = voi_module.VOIAction(
        action_id="stop",
        kind="stop",
        target="stop",
        q_score=0.0,
    )

    filtered = voi_module._suppress_post_nonproductive_uncertified_evidence_plateau_churn(
        [refine, resample, stop],
        state=state,
        current_certificate=0.4,
        config=VOIConfig(certificate_threshold=0.80),
    )

    assert [action.kind for action in filtered] == ["stop"]


def test_evidence_exhausted_uncertified_search_tail_prefers_stop_after_productive_evidence_phase() -> None:
    state = VOIControllerState(
        iteration_index=5,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.08, 10.04, 10.02)},
            {"route_id": "route_c", "objective_vector": (10.2, 10.1, 10.0)},
            {"route_id": "route_d", "objective_vector": (10.35, 10.22, 10.11)},
            {"route_id": "route_e", "objective_vector": (10.41, 10.3, 10.18)},
        ],
        certificate={"route_a": 0.67354, "route_b": 0.407131, "route_c": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=0,
        action_trace=[
            {"chosen_action": {"kind": "refine_top1_dccs"}, "realized_certificate_delta": 0.0, "realized_frontier_gain": 1.0, "realized_productive": True},
            {"chosen_action": {"kind": "refine_top1_dccs"}, "realized_certificate_delta": 0.0, "realized_frontier_gain": 1.0, "realized_productive": True},
            {"chosen_action": {"kind": "increase_stochastic_samples"}, "realized_certificate_delta": 0.068677, "realized_productive": True},
            {"chosen_action": {"kind": "increase_stochastic_samples"}, "realized_certificate_delta": 0.051707, "realized_productive": True},
            {"chosen_action": {"kind": "increase_stochastic_samples"}, "realized_certificate_delta": 0.040335, "realized_productive": True},
        ],
        ambiguity_context={
            "od_ambiguity_index": 0.66,
            "od_hard_case_prior": 0.67,
            "od_engine_disagreement_prior": 0.43,
            "od_ambiguity_confidence": 0.88,
            "od_ambiguity_source_count": 3,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1,"historical_results_bootstrap":1}',
            "od_ambiguity_support_ratio": 0.74,
            "od_ambiguity_prior_strength": 0.66,
            "od_ambiguity_source_entropy": 0.79,
            "ambiguity_budget_prior": 0.68,
            "refc_stress_world_fraction": 0.37,
        },
        certificate_margin=0.266409,
        search_completeness_score=0.424113,
        search_completeness_gap=0.415887,
        prior_support_strength=0.650912,
        support_richness=0.650912,
        ambiguity_pressure=0.750376,
        pending_challenger_mass=0.66337,
        best_pending_flip_probability=0.998715,
        corridor_family_recall=0.181818,
        frontier_recall_at_budget=0.225755,
        top_refresh_gain=0.30888,
        top_fragility_mass=0.633205,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="candidate",
        q_score=0.166902,
        predicted_delta_certificate=0.366795,
        predicted_delta_margin=0.280131,
        predicted_delta_frontier=0.233205,
        metadata={
            "normalized_objective_gap": 0.188203,
            "normalized_mechanism_gap": 0.168712,
        },
    )

    assert voi_module._should_stop_evidence_exhausted_uncertified_search_tail(
        refine,
        state=state,
        current_certificate=0.67354,
        config=VOIConfig(certificate_threshold=0.84),
    ) is True


def test_support_rich_certified_refresh_preference_targets_best_refresh_action() -> None:
    cfg = VOIConfig(certificate_threshold=0.81)
    state = VOIControllerState(
        iteration_index=1,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.1, 10.05, 10.0)},
        ],
        certificate={"route_a": 1.0, "route_b": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=2,
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.71,
            "od_ambiguity_source_entropy": 0.83,
            "od_hard_case_prior": 0.64,
            "ambiguity_budget_prior": 0.64,
        },
        support_richness=0.71,
        ambiguity_pressure=0.66,
        top_refresh_gain=0.07,
        top_fragility_mass=0.22,
        competitor_pressure=0.84,
    )
    low_refresh = voi_module.VOIAction(
        action_id="refresh:scenario",
        kind="refresh_top1_vor",
        target="scenario",
        q_score=0.11,
        predicted_delta_certificate=0.02,
        predicted_delta_margin=0.01,
        metadata={},
        reason="refresh_evidence_family",
    )
    high_refresh = voi_module.VOIAction(
        action_id="refresh:terrain",
        kind="refresh_top1_vor",
        target="terrain",
        q_score=0.14,
        predicted_delta_certificate=0.05,
        predicted_delta_margin=0.02,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.07,
        },
        reason="refresh_evidence_family",
    )

    adjusted_actions = voi_module._apply_support_rich_certified_refresh_preference(
        [low_refresh, high_refresh],
        state=state,
        current_certificate=1.0,
        config=cfg,
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=1,
    )

    adjusted_low = next(action for action in adjusted_actions if action.target == "scenario")
    adjusted_high = next(action for action in adjusted_actions if action.target == "terrain")
    assert adjusted_low.metadata.get("support_rich_certified_refresh_preference_applied") is not True
    assert adjusted_high.metadata["support_rich_certified_refresh_preference_applied"] is True
    assert adjusted_high.q_score > adjusted_low.q_score


def test_strong_winner_side_refresh_preference_targets_best_refresh_action() -> None:
    state = VOIControllerState(
        iteration_index=0,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.2, 10.1, 10.05)},
        ],
        certificate={"route_a": 0.457143, "route_b": 0.542857},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=3,
        ambiguity_context={
            "od_ambiguity_support_ratio": 0.707403,
            "od_ambiguity_source_entropy": 0.836641,
            "od_hard_case_prior": 0.692693,
            "ambiguity_budget_prior": 0.453718,
        },
        support_richness=0.707403,
        ambiguity_pressure=0.725242,
        top_refresh_gain=0.531429,
        top_fragility_mass=0.457143,
        competitor_pressure=1.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_top1_dccs:test",
        kind="refine_top1_dccs",
        target="cohort",
        q_score=0.23,
        predicted_delta_certificate=0.405013,
        predicted_delta_margin=0.212208,
        predicted_delta_frontier=0.057745,
        metadata={
            "mean_flip_probability": 0.969257,
            "normalized_objective_gap": 0.0,
            "normalized_mechanism_gap": 0.10,
            "normalized_overlap_reduction": 0.904762,
        },
        reason="refine_candidate",
    )
    low_refresh = voi_module.VOIAction(
        action_id="refresh:carbon",
        kind="refresh_top1_vor",
        target="carbon",
        q_score=0.08,
        predicted_delta_certificate=0.01,
        predicted_delta_margin=0.01,
        metadata={},
        reason="refresh_evidence_family",
    )
    high_refresh = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        q_score=0.16,
        predicted_delta_certificate=0.108502,
        predicted_delta_margin=0.078079,
        metadata={
            "structured_refresh_signal": True,
            "empirical_refresh_certificate_uplift": 0.074286,
        },
        reason="refresh_evidence_family",
    )

    adjusted_actions = voi_module._apply_strong_winner_side_refresh_preference(
        [refine, low_refresh, high_refresh],
        state=state,
        current_certificate=0.457143,
        config=VOIConfig(certificate_threshold=0.83),
        evidence_uncertainty=True,
        supported_fragility_uncertainty=True,
        recent_no_gain_refine_streak=0,
    )

    adjusted_refine = next(action for action in adjusted_actions if action.kind == "refine_top1_dccs")
    adjusted_low = next(action for action in adjusted_actions if action.target == "carbon")
    adjusted_high = next(action for action in adjusted_actions if action.target == "fuel")
    assert adjusted_low.metadata.get("winner_side_refresh_preference_applied") is not True
    assert adjusted_high.metadata["winner_side_refresh_preference_applied"] is True
    assert adjusted_high.metadata["winner_side_refresh_preference_empirical_bridge"] is True
    assert adjusted_refine.metadata["winner_side_refresh_refine_discount_applied"] is True
    assert adjusted_high.q_score > adjusted_refine.q_score


def test_evidence_exhausted_uncertified_search_tail_stops_low_value_terminal_topk_tail() -> None:
    state = VOIControllerState(
        iteration_index=5,
        frontier=[
            {"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)},
            {"route_id": "route_b", "objective_vector": (10.05, 10.02, 10.01)},
            {"route_id": "route_c", "objective_vector": (10.12, 10.07, 10.05)},
            {"route_id": "route_d", "objective_vector": (10.2, 10.12, 10.1)},
            {"route_id": "route_e", "objective_vector": (10.31, 10.2, 10.16)},
        ],
        certificate={"route_a": 0.67354, "route_b": 0.32646, "route_c": 0.0},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=2,
        remaining_evidence_budget=0,
        action_trace=[
            {"chosen_action": {"kind": "refine_top1_dccs"}, "realized_certificate_delta": 0.0, "realized_frontier_gain": 1.0, "realized_productive": True},
            {"chosen_action": {"kind": "refine_top1_dccs"}, "realized_certificate_delta": 0.0, "realized_frontier_gain": 1.0, "realized_productive": True},
            {"chosen_action": {"kind": "increase_stochastic_samples"}, "realized_certificate_delta": 0.068677, "realized_productive": True},
            {"chosen_action": {"kind": "increase_stochastic_samples"}, "realized_certificate_delta": 0.051707, "realized_productive": True},
            {"chosen_action": {"kind": "increase_stochastic_samples"}, "realized_certificate_delta": 0.040335, "realized_productive": True},
        ],
        ambiguity_context={
            "od_ambiguity_index": 0.403029,
            "od_hard_case_prior": 0.492942,
            "od_engine_disagreement_prior": 0.458077,
            "od_ambiguity_confidence": 0.810497,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"historical_results_bootstrap":4,"routing_graph_probe":11}',
            "od_ambiguity_support_ratio": 0.686842,
            "od_ambiguity_prior_strength": 0.403029,
            "od_ambiguity_source_entropy": 0.836641,
            "ambiguity_budget_prior": 0.423843,
            "refc_stress_world_fraction": 0.32646,
        },
        certificate_margin=-0.16646,
        search_completeness_score=0.480265,
        search_completeness_gap=0.359735,
        prior_support_strength=0.650912,
        support_richness=0.650912,
        ambiguity_pressure=0.75001,
        pending_challenger_mass=0.669066,
        best_pending_flip_probability=0.998737,
        corridor_family_recall=0.181818,
        frontier_recall_at_budget=0.227721,
        top_refresh_gain=0.302564,
        top_fragility_mass=0.512821,
        competitor_pressure=1.0,
        near_tie_mass=0.0,
    )
    refine = voi_module.VOIAction(
        action_id="refine_topk_dccs:test",
        kind="refine_topk_dccs",
        target="candidate",
        q_score=0.157286,
        predicted_delta_certificate=0.32646,
        predicted_delta_margin=0.28386,
        predicted_delta_frontier=0.318212,
        metadata={
            "normalized_objective_gap": 0.223227,
            "normalized_mechanism_gap": 0.045457,
            "corridor_family_recall": 0.181818,
            "frontier_recall_at_budget": 0.227721,
        },
    )

    assert voi_module._should_stop_evidence_exhausted_uncertified_search_tail(
        refine,
        state=state,
        current_certificate=0.67354,
        config=VOIConfig(certificate_threshold=0.84),
    ) is True


def test_replay_oracle_action_value_estimate_tracks_tri_source_terms() -> None:
    estimate = replay_oracle_module.build_action_value_estimate(
        action_id="refresh:scenario",
        action_kind="refresh_top1_vor",
        action_target="scenario",
        action_reason="refresh_evidence_family",
        cost_evidence=2,
        predicted_delta_certificate=0.40,
        predicted_delta_margin=0.10,
        predicted_delta_frontier=0.20,
        lambda_certificate=0.50,
        lambda_margin=0.30,
        lambda_frontier=0.20,
        ranked_q_score=0.31,
        metadata={"structured_refresh_signal": True},
    )

    assert estimate.weighted_certificate_value == pytest.approx(0.20)
    assert estimate.weighted_margin_value == pytest.approx(0.03)
    assert estimate.weighted_frontier_value == pytest.approx(0.04)
    assert estimate.total_predicted_value == pytest.approx(0.27)
    assert estimate.base_q_score == pytest.approx(0.135)
    assert estimate.ranked_q_score == pytest.approx(0.31)
    assert estimate.score_terms["certificate_delta"] == pytest.approx(0.40)
    assert estimate.metadata["structured_refresh_signal"] is True


def test_controller_build_action_replay_record_merges_predicted_and_realized_values() -> None:
    cfg = VOIConfig(lambda_certificate=0.50, lambda_margin=0.30, lambda_frontier=0.20)
    action = voi_module.VOIAction(
        action_id="refresh:fuel",
        kind="refresh_top1_vor",
        target="fuel",
        cost_evidence=2,
        predicted_delta_certificate=0.16,
        predicted_delta_margin=0.08,
        predicted_delta_frontier=0.02,
        q_score=0.19,
        metadata={"structured_refresh_signal": True},
        reason="refresh_evidence_family",
    )
    trace_entry = {
        "realized_certificate_before": 0.41,
        "realized_certificate_after": 0.54,
        "realized_certificate_delta": 0.13,
        "realized_frontier_gain": 1.0,
        "realized_selected_route_changed": True,
        "realized_selected_score_delta": 0.07,
        "realized_productive": True,
        "trace_metadata": {"trace_source": "main.voi", "iteration": 2},
        "replay_metadata": {"replay_token": "replay-3"},
        "oracle_metadata": {"oracle_run_id": "oracle-7"},
    }

    record = voi_module.build_action_replay_record(
        action,
        config=cfg,
        trace_entry=trace_entry,
        trace_metadata={"lane": "controller"},
    )
    primitives = voi_module.action_scoring_primitives(action, config=cfg)

    assert record.estimate.base_q_score == pytest.approx(0.054)
    assert primitives["weighted_certificate_value"] == pytest.approx(0.08)
    assert primitives["base_q_score"] == pytest.approx(0.054)
    assert primitives["q_score_adjustment"] == pytest.approx(0.136)
    assert record.realization is not None
    assert record.realization.realized_certificate_delta == pytest.approx(0.13)
    assert record.realization.realized_frontier_gain == pytest.approx(1.0)
    payload = record.as_dict()
    assert payload["trace_metadata"]["trace_source"] == "main.voi"
    assert payload["trace_metadata"]["lane"] == "controller"
    assert payload["replay_metadata"]["replay_token"] == "replay-3"
    assert payload["oracle_metadata"]["oracle_run_id"] == "oracle-7"


def test_run_controller_trace_carries_action_value_records_for_replay() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    frontier = [{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}]

    def _refresh(state: VOIControllerState, action):
        return replace(
            state,
            iteration_index=state.iteration_index + 1,
            remaining_evidence_budget=max(0, state.remaining_evidence_budget - action.cost_evidence),
            certificate={"route_a": 0.81, "route_b": 0.10},
        )

    stop_certificate = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        certificate_map={"route_a": 0.20, "route_b": 0.10},
        selected_route_id="route_a",
        config=VOIConfig(
            certificate_threshold=0.80,
            search_completeness_threshold=0.0,
            stop_threshold=0.0,
            search_budget=0,
            evidence_budget=1,
        ),
        hooks=VOIActionHooks(refresh=_refresh),
    )

    trace_entry = stop_certificate.action_trace[0]
    value_record = trace_entry["chosen_action_value_record"]
    state_value_record = stop_certificate.state_trace[0]["chosen_action_value_record"]

    assert trace_entry["trace_metadata"]["trace_source"] == "voi_controller.run_controller"
    assert trace_entry["trace_metadata"]["action_menu_count"] == len(trace_entry["feasible_actions"])
    assert value_record["estimate"]["action_id"] == trace_entry["chosen_action"]["action_id"]
    assert value_record["realization"] is None
    assert len(trace_entry["action_menu_value_estimates"]) == len(trace_entry["feasible_actions"])
    assert state_value_record["estimate"]["action_id"] == value_record["estimate"]["action_id"]
