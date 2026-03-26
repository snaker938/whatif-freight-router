from __future__ import annotations

from typing import Any

from ._process_cache import ProcessGlobalCacheStore
from .settings import settings

RouteCachePayload = tuple[list[dict[str, Any]], list[str], int] | tuple[
    list[dict[str, Any]],
    list[str],
    int,
    dict[str, Any],
]


class RouteCacheStore(ProcessGlobalCacheStore[RouteCachePayload]):
    pass


ROUTE_CACHE = RouteCacheStore(
    ttl_s=settings.route_cache_ttl_s,
    max_entries=settings.route_cache_max_entries,
    max_estimated_bytes=settings.route_cache_max_estimated_bytes,
)


def get_cached_routes(
    key: str,
) -> RouteCachePayload | None:
    return ROUTE_CACHE.get(key)


def set_cached_routes(
    key: str,
    value: RouteCachePayload,
) -> bool:
    return ROUTE_CACHE.set(key, value)


def clear_route_cache() -> int:
    return ROUTE_CACHE.clear()


def route_cache_stats() -> dict[str, int]:
    return ROUTE_CACHE.snapshot()
