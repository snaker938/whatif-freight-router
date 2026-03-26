from __future__ import annotations

import types
from dataclasses import replace

import pytest

import app.main as main_module
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
