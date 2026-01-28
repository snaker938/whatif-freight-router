from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def dominates(a: tuple[float, ...], b: tuple[float, ...], *, tol: float = 1e-9) -> bool:
    """Return True if vector a Pareto-dominates b (minimisation).

    a dominates b iff:
      - a_i <= b_i for all i
      - a_i <  b_i for at least one i
    """
    if len(a) != len(b):
        raise ValueError("dimension mismatch")

    le_all = True
    lt_any = False
    for ai, bi in zip(a, b, strict=True):
        if ai > bi + tol:
            le_all = False
            break
        if ai < bi - tol:
            lt_any = True

    return le_all and lt_any


def pareto_filter(items: Iterable[T], key: Callable[[T], tuple[float, ...]]) -> list[T]:
    """Keep only non-dominated items.

    O(n^2) — totally fine for the small candidate sets we get from OSRM alternatives.
    """
    nondominated: list[T] = []
    nondominated_vecs: list[tuple[float, ...]] = []

    for item in items:
        vec = key(item)

        dominated = False
        to_remove: list[int] = []

        for i, nd_vec in enumerate(nondominated_vecs):
            if dominates(nd_vec, vec):
                dominated = True
                break
            if dominates(vec, nd_vec):
                to_remove.append(i)

        if dominated:
            continue

        for idx in reversed(to_remove):
            nondominated.pop(idx)
            nondominated_vecs.pop(idx)

        nondominated.append(item)
        nondominated_vecs.append(vec)

    return nondominated
