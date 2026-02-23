from __future__ import annotations

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
