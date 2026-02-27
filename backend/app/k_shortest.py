from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class PathResult:
    nodes: tuple[str, ...]
    cost: float


class PathNotFoundError(ValueError):
    pass


TransitionStateFn = Callable[[str | None, str, str], tuple[Hashable, float] | None]


def _dijkstra_shortest_path(
    *,
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    start: str,
    goal: str,
    banned_nodes: set[str] | None = None,
    banned_edges: set[tuple[str, str]] | None = None,
    max_hops: int = 220,
    explored_counter: list[int] | None = None,
    max_state_budget: int | None = None,
    max_repeat_per_node: int = 1,
    max_cost_limit: float | None = None,
    transition_state_fn: TransitionStateFn | None = None,
    deadline_monotonic_s: float | None = None,
) -> PathResult:
    banned_nodes = banned_nodes or set()
    banned_edges = banned_edges or set()
    if start in banned_nodes or goal in banned_nodes:
        raise PathNotFoundError("start/goal blocked")
    initial_state: Hashable = "start"
    heap: list[tuple[float, int, str, tuple[str, ...], Hashable]] = [(0.0, 0, start, (start,), initial_state)]
    best_cost_by_state: dict[tuple[str, Hashable], float] = {(start, initial_state): 0.0}
    while heap:
        if deadline_monotonic_s is not None and time.monotonic() >= float(deadline_monotonic_s):
            raise PathNotFoundError("search deadline exceeded")
        if max_state_budget is not None and max_state_budget > 0 and explored_counter is not None:
            if explored_counter[0] >= max_state_budget:
                raise PathNotFoundError("state budget exceeded")
        cost, hops, node, path, state_key = heapq.heappop(heap)
        if explored_counter is not None:
            explored_counter[0] += 1
        if node == goal:
            return PathResult(nodes=path, cost=cost)
        if hops >= max_hops:
            continue
        for nxt, edge_cost in adjacency.get(node, ()):
            if nxt in banned_nodes:
                continue
            if (node, nxt) in banned_edges:
                continue
            if max_repeat_per_node <= 1 and nxt in path:
                continue
            if max_repeat_per_node > 1 and path.count(nxt) >= max_repeat_per_node:
                continue
            next_state_key = state_key
            transition_penalty = 0.0
            if transition_state_fn is not None:
                prev_node = path[-2] if len(path) > 1 else None
                transition = transition_state_fn(prev_node, node, nxt)
                if transition is None:
                    continue
                next_state_key, transition_penalty = transition
            new_cost = cost + max(0.001, float(edge_cost)) + max(0.0, float(transition_penalty))
            if max_cost_limit is not None and new_cost > max_cost_limit:
                continue
            best_key = (nxt, next_state_key)
            prev_best = best_cost_by_state.get(best_key)
            if prev_best is not None and new_cost >= prev_best:
                continue
            best_cost_by_state[best_key] = new_cost
            heapq.heappush(heap, (new_cost, hops + 1, nxt, (*path, nxt), next_state_key))
    raise PathNotFoundError("no path")


def yen_k_shortest_paths_with_stats(
    *,
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    start: str,
    goal: str,
    k: int,
    max_hops: int = 220,
    max_state_budget: int | None = None,
    max_repeat_per_node: int = 1,
    max_detour_ratio: float | None = None,
    max_candidate_pool: int | None = None,
    transition_state_fn: TransitionStateFn | None = None,
    deadline_monotonic_s: float | None = None,
) -> tuple[tuple[PathResult, ...], dict[str, int | str]]:
    if k <= 0:
        return (), {
            "explored_states": 0,
            "generated_candidates": 0,
            "pruned_constraints": 0,
            "termination_reason": "invalid_k",
            "no_path_reason": "invalid_k",
            "first_error": "",
        }
    explored_counter = [0]
    generated_candidates = 0
    pruned_constraints = 0
    first_error = ""
    try:
        first = _dijkstra_shortest_path(
            adjacency=adjacency,
            start=start,
            goal=goal,
            banned_nodes=set(),
            banned_edges=set(),
            max_hops=max_hops,
            explored_counter=explored_counter,
            max_state_budget=max_state_budget,
            max_repeat_per_node=max_repeat_per_node,
            transition_state_fn=transition_state_fn,
            deadline_monotonic_s=deadline_monotonic_s,
        )
    except PathNotFoundError as exc:
        first_error = str(exc).strip() or "no path"
        return (), {
            "explored_states": explored_counter[0],
            "generated_candidates": 0,
            "pruned_constraints": 0,
            "termination_reason": "no_initial_path",
            "no_path_reason": normalize_no_path_reason(first_error),
            "first_error": first_error,
        }

    shortest: list[PathResult] = [first]
    candidates: list[tuple[float, tuple[str, ...]]] = []
    candidate_seen: set[tuple[str, ...]] = {first.nodes}
    detour_cap = (
        max(1.0, float(max_detour_ratio)) * float(first.cost)
        if max_detour_ratio is not None and max_detour_ratio > 0
        else inf
    )
    termination_reason = "k_paths_collected"

    for _ in range(1, k):
        previous = shortest[-1]
        previous_nodes = list(previous.nodes)
        for spur_idx in range(len(previous_nodes) - 1):
            root_path = tuple(previous_nodes[: spur_idx + 1])
            spur_node = root_path[-1]

            banned_edges: set[tuple[str, str]] = set()
            for p in shortest:
                if len(p.nodes) <= spur_idx:
                    continue
                if tuple(p.nodes[: spur_idx + 1]) == root_path and len(p.nodes) > spur_idx + 1:
                    banned_edges.add((p.nodes[spur_idx], p.nodes[spur_idx + 1]))

            banned_nodes = set(root_path[:-1])
            try:
                spur = _dijkstra_shortest_path(
                    adjacency=adjacency,
                    start=spur_node,
                    goal=goal,
                    banned_nodes=banned_nodes,
                    banned_edges=banned_edges,
                    max_hops=max_hops,
                    explored_counter=explored_counter,
                    max_state_budget=max_state_budget,
                    max_repeat_per_node=max_repeat_per_node,
                    transition_state_fn=transition_state_fn,
                    deadline_monotonic_s=deadline_monotonic_s,
                    max_cost_limit=(
                        None
                        if transition_state_fn is not None
                        else (
                            None
                        if detour_cap == inf
                        else max(0.0, detour_cap - sum(
                            next(
                                (edge_cost for nxt, edge_cost in adjacency.get(root_path[idx - 1], ()) if nxt == root_path[idx]),
                                0.0,
                            )
                            for idx in range(1, len(root_path))
                        ))
                        )
                    ),
                )
            except PathNotFoundError as exc:
                if not first_error:
                    first_error = str(exc).strip() or "no path"
                continue

            total_nodes = (*root_path[:-1], *spur.nodes)
            if total_nodes in candidate_seen:
                continue
            root_cost = 0.0
            for idx in range(1, len(root_path)):
                src = root_path[idx - 1]
                dst = root_path[idx]
                for nxt, edge_cost in adjacency.get(src, ()):
                    if nxt == dst:
                        root_cost += edge_cost
                        break
            total_cost = root_cost + spur.cost
            if total_cost > detour_cap:
                pruned_constraints += 1
                continue
            heapq.heappush(candidates, (total_cost, total_nodes))
            generated_candidates += 1
            candidate_seen.add(total_nodes)
            if max_candidate_pool is not None and max_candidate_pool > 0 and len(candidates) > max_candidate_pool:
                # Keep candidate heap bounded for deterministic memory/runtime.
                candidates = heapq.nsmallest(max_candidate_pool, candidates)
                heapq.heapify(candidates)

        if not candidates:
            termination_reason = "candidate_pool_exhausted"
            break
        best_cost, best_nodes = heapq.heappop(candidates)
        shortest.append(PathResult(nodes=best_nodes, cost=best_cost))
    if len(shortest) >= int(max(1, k)):
        termination_reason = "k_paths_collected"
    return tuple(shortest), {
        "explored_states": int(explored_counter[0]),
        "generated_candidates": int(generated_candidates),
        "pruned_constraints": int(pruned_constraints),
        "termination_reason": termination_reason,
        "no_path_reason": (
            ""
            if shortest
            else normalize_no_path_reason(first_error or termination_reason)
        ),
        "first_error": first_error,
    }


def normalize_no_path_reason(message: str) -> str:
    lowered = str(message or "").strip().lower()
    if "state budget" in lowered:
        return "state_budget_exceeded"
    if "start/goal blocked" in lowered:
        return "start_or_goal_blocked"
    if "search deadline exceeded" in lowered:
        return "path_search_exhausted"
    if "no path" in lowered:
        return "no_path"
    return "path_search_exhausted"


def yen_k_shortest_paths(
    *,
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    start: str,
    goal: str,
    k: int,
    max_hops: int = 220,
    max_state_budget: int | None = None,
    max_repeat_per_node: int = 1,
    max_detour_ratio: float | None = None,
    max_candidate_pool: int | None = None,
    transition_state_fn: TransitionStateFn | None = None,
    deadline_monotonic_s: float | None = None,
) -> tuple[PathResult, ...]:
    paths, _stats = yen_k_shortest_paths_with_stats(
        adjacency=adjacency,
        start=start,
        goal=goal,
        k=k,
        max_hops=max_hops,
        max_state_budget=max_state_budget,
        max_repeat_per_node=max_repeat_per_node,
        max_detour_ratio=max_detour_ratio,
        max_candidate_pool=max_candidate_pool,
        transition_state_fn=transition_state_fn,
        deadline_monotonic_s=deadline_monotonic_s,
    )
    return paths
