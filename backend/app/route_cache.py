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
HOT_RERUN_ROUTE_CACHE_CHECKPOINT = RouteCacheStore(
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


def checkpoint_route_cache() -> int:
    return HOT_RERUN_ROUTE_CACHE_CHECKPOINT.import_items(ROUTE_CACHE.export_items(), clear_first=False)


def restore_checkpointed_route_cache(*, clear_first: bool = False) -> int:
    return ROUTE_CACHE.import_items(
        HOT_RERUN_ROUTE_CACHE_CHECKPOINT.export_items(),
        clear_first=clear_first,
    )


def clear_route_cache_checkpoint() -> int:
    return HOT_RERUN_ROUTE_CACHE_CHECKPOINT.clear()


def route_cache_stats() -> dict[str, int]:
    return ROUTE_CACHE.snapshot()


def route_cache_checkpoint_stats() -> dict[str, int]:
    return HOT_RERUN_ROUTE_CACHE_CHECKPOINT.snapshot()
