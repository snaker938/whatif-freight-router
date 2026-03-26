from __future__ import annotations

import time

from app.k_shortest import PathResult, yen_k_shortest_paths, yen_k_shortest_paths_with_stats


def _graph() -> dict[str, tuple[tuple[str, float], ...]]:
    return {
        "A": (("B", 1.0), ("C", 1.0)),
        "B": (("D", 1.0), ("C", 1.0)),
        "C": (("D", 1.0),),
        "D": (),
    }


def test_yen_k_shortest_returns_sorted_paths_and_stats() -> None:
    paths, stats = yen_k_shortest_paths_with_stats(adjacency=_graph(), start="A", goal="D", k=3)

    assert isinstance(paths, tuple)
    assert len(paths) == 3
    assert all(isinstance(path, PathResult) for path in paths)
    assert paths[0].nodes == ("A", "B", "D")
    assert paths[0].cost <= paths[1].cost <= paths[2].cost
    assert stats["explored_states"] > 0
    assert stats["generated_candidates"] >= 1
    assert stats["pruned_constraints"] >= 0


def test_yen_k_shortest_honors_transition_constraints() -> None:
    def transition_state(_prev: str | None, node: str, nxt: str) -> tuple[str, float] | None:
        if node == "B" and nxt == "D":
            return None
        return nxt, 0.0

    paths = yen_k_shortest_paths(
        adjacency=_graph(),
        start="A",
        goal="D",
        k=2,
        transition_state_fn=transition_state,
    )

    assert len(paths) == 2
    assert all(path.nodes != ("A", "B", "D") for path in paths)
    assert paths[0].nodes == ("A", "C", "D")


def test_yen_k_shortest_returns_empty_when_unreachable() -> None:
    disconnected = {"A": (("B", 1.0),), "B": (), "D": ()}
    paths, stats = yen_k_shortest_paths_with_stats(
        adjacency=disconnected,
        start="A",
        goal="D",
        k=2,
    )

    assert paths == ()
    assert stats["generated_candidates"] == 0
    assert stats["termination_reason"] == "no_initial_path"
    assert stats["no_path_reason"] in {"no_path", "path_search_exhausted"}


def test_yen_k_shortest_heuristic_matches_baseline_shortest_path() -> None:
    adjacency = {
        "A": (("B", 2.0), ("C", 2.5)),
        "B": (("D", 2.0), ("E", 1.0)),
        "C": (("D", 1.4),),
        "E": (("D", 1.0),),
        "D": (),
    }

    baseline, _stats = yen_k_shortest_paths_with_stats(
        adjacency=adjacency,
        start="A",
        goal="D",
        k=1,
    )
    heuristic, _stats_heuristic = yen_k_shortest_paths_with_stats(
        adjacency=adjacency,
        start="A",
        goal="D",
        k=1,
        heuristic_fn=lambda node: {
            "A": 1.2,
            "B": 0.8,
            "C": 0.3,
            "E": 0.3,
            "D": 0.0,
        }.get(node, 0.0),
    )

    assert baseline
    assert heuristic
    assert heuristic[0].nodes == baseline[0].nodes
    assert heuristic[0].cost == baseline[0].cost


def test_admissible_heuristic_preserves_astar_optimality() -> None:
    adjacency = {
        "A": (("B", 1.0), ("C", 1.0)),
        "B": (("D", 1.0),),
        "C": (("D", 1.5),),
        "D": (),
    }

    paths = yen_k_shortest_paths(
        adjacency=adjacency,
        start="A",
        goal="D",
        k=1,
        heuristic_fn=lambda node: {
            "A": 0.0,
            "B": 1.0,
            "C": 1.5,
            "D": 0.0,
        }.get(node, 0.0),
    )

    assert paths[0].nodes == ("A", "B", "D")
    assert paths[0].cost == 2.0


def test_inadmissible_heuristic_can_change_returned_shortest_path() -> None:
    adjacency = {
        "A": (("B", 1.0), ("C", 1.0)),
        "B": (("D", 1.0),),
        "C": (("D", 1.5),),
        "D": (),
    }

    baseline = yen_k_shortest_paths(
        adjacency=adjacency,
        start="A",
        goal="D",
        k=1,
    )
    heuristic = yen_k_shortest_paths(
        adjacency=adjacency,
        start="A",
        goal="D",
        k=1,
        heuristic_fn=lambda node: {
            "A": 0.0,
            "B": 10.0,
            "C": 0.0,
            "D": 0.0,
        }.get(node, 0.0),
    )

    assert baseline[0].nodes == ("A", "B", "D")
    assert baseline[0].cost == 2.0
    assert heuristic[0].nodes == ("A", "C", "D")
    assert heuristic[0].cost == 2.5


def test_yen_k_shortest_normalizes_elapsed_deadline_to_path_search_exhausted() -> None:
    _paths, stats = yen_k_shortest_paths_with_stats(
        adjacency=_graph(),
        start="A",
        goal="D",
        k=2,
        deadline_monotonic_s=time.monotonic() - 1.0,
    )

    assert stats["termination_reason"] == "no_initial_path"
    assert stats["no_path_reason"] == "path_search_exhausted"
    assert stats["first_error"] == "search deadline exceeded"
