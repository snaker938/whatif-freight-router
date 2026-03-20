from __future__ import annotations

from dataclasses import replace

from app.decision_critical import DCCSConfig, select_candidates
from app.evidence_certification import compute_fragility_maps
from app.voi_controller import (
    VOIActionHooks,
    VOIConfig,
    VOIControllerState,
    build_action_menu,
    run_controller,
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


def test_action_scoring_is_deterministic() -> None:
    dccs = _dccs_result()
    fragility = _fragility_result()
    state = VOIControllerState(
        iteration_index=0,
        frontier=[{"route_id": "route_a", "objective_vector": (10.0, 10.0, 10.0)}],
        certificate={"route_a": 0.25},
        winner_id="route_a",
        selected_route_id="route_a",
        remaining_search_budget=1,
        remaining_evidence_budget=1,
        action_trace=[],
        active_evidence_families=["scenario"],
        near_tie_mass=0.2,
    )

    first = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig())
    second = build_action_menu(state, dccs=dccs, fragility=fragility, config=VOIConfig())

    assert [action.as_dict() for action in first] == [action.as_dict() for action in second]
    assert first[0].kind == "refine_top1_dccs"
    assert first[0].q_score >= first[1].q_score


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
        config=VOIConfig(certificate_threshold=0.80, search_budget=1, evidence_budget=1),
    )
    exhausted = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        selected_route_id="route_a",
        config=VOIConfig(search_budget=0, evidence_budget=0),
    )
    thresholded = run_controller(
        initial_frontier=frontier,
        dccs=dccs,
        fragility=fragility,
        winner_id="route_a",
        certificate_value=0.20,
        selected_route_id="route_a",
        config=VOIConfig(stop_threshold=10.0, search_budget=1, evidence_budget=1),
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
        selected_route_id="route_a",
        config=VOIConfig(certificate_threshold=0.80, search_budget=1, evidence_budget=1),
        hooks=VOIActionHooks(refine=_refine),
    )

    assert stop_certificate.final_winner_route_id == "route_a"
    assert stop_certificate.final_strict_frontier_size == 1
    assert stop_certificate.certified is True
    assert stop_certificate.search_budget_used == 1
    assert stop_certificate.action_trace
    assert stop_certificate.best_rejected_action is not None

