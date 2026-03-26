from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._process_cache import ProcessGlobalCacheStore
from .settings import settings

RouteStatePayload = tuple[Any, Any, Any, Any, Any, dict[str, list[str]], dict[str, float]]


@dataclass(frozen=True)
class CachedRouteState:
    state: RouteStatePayload
    estimated_option_build_ms: float = 0.0
    estimated_pareto_ms: float = 0.0


RouteStateCachePayload = CachedRouteState


class RouteStateCacheStore(ProcessGlobalCacheStore[RouteStateCachePayload]):
    pass


ROUTE_STATE_CACHE = RouteStateCacheStore(
    ttl_s=settings.route_state_cache_ttl_s,
    max_entries=settings.route_state_cache_max_entries,
    max_estimated_bytes=settings.route_state_cache_max_estimated_bytes,
)


def get_cached_route_state(key: str) -> RouteStateCachePayload | None:
    return ROUTE_STATE_CACHE.get(key)


def set_cached_route_state(key: str, value: RouteStateCachePayload) -> bool:
    return ROUTE_STATE_CACHE.set(key, value)


def clear_route_state_cache() -> int:
    return ROUTE_STATE_CACHE.clear()


def route_state_cache_stats() -> dict[str, int]:
    return ROUTE_STATE_CACHE.snapshot()
