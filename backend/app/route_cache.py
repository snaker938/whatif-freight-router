from __future__ import annotations

import copy
from typing import Any

from ._process_cache import ProcessGlobalCacheStore
from .settings import settings

RouteCachePayload = tuple[list[dict[str, Any]], list[str], int] | tuple[
    list[dict[str, Any]],
    list[str],
    int,
    dict[str, Any],
]
ROUTE_CACHE_SCHEMA_VERSION = 1


class RouteCacheStore(ProcessGlobalCacheStore[RouteCachePayload]):
    def __init__(self, *, ttl_s: int, max_entries: int, max_estimated_bytes: int) -> None:
        super().__init__(
            ttl_s=ttl_s,
            max_entries=max_entries,
            max_estimated_bytes=max_estimated_bytes,
        )
        self._schema_version = ROUTE_CACHE_SCHEMA_VERSION
        self._invalidation_counters: dict[str, int] = {
            "expired": 0,
            "manual_clear": 0,
        }
        self._checkpoint_operations = 0
        self._checkpointed_entries = 0
        self._restore_operations = 0
        self._restored_entries = 0

    def _increment_invalidation(self, reason: str, count: int = 1) -> None:
        cleaned = str(reason or "unknown").strip() or "unknown"
        with self._lock:
            self._invalidation_counters[cleaned] = self._invalidation_counters.get(cleaned, 0) + max(0, int(count))

    def get(self, key: str) -> RouteCachePayload | None:
        with self._lock:
            cache_entry = self._items.get(key)
            if cache_entry is None:
                self._misses += 1
                return None
            if self._is_expired(cache_entry):
                self._remove_key(key)
                self._invalidation_counters["expired"] = self._invalidation_counters.get("expired", 0) + 1
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(cache_entry.payload)

    def clear(self) -> int:
        cleared = super().clear()
        if cleared > 0:
            self._increment_invalidation("manual_clear", cleared)
        return cleared

    def record_checkpoint(self, count: int) -> None:
        with self._lock:
            self._checkpoint_operations += 1
            self._checkpointed_entries += max(0, int(count))

    def record_restore(self, count: int) -> None:
        with self._lock:
            self._restore_operations += 1
            self._restored_entries += max(0, int(count))

    def snapshot(self) -> dict[str, Any]:
        stats = super().snapshot()
        with self._lock:
            stats["schema_version"] = self._schema_version
            stats["invalidation_counters"] = dict(self._invalidation_counters)
            stats["checkpoint_operations"] = self._checkpoint_operations
            stats["checkpointed_entries"] = self._checkpointed_entries
            stats["restore_operations"] = self._restore_operations
            stats["restored_entries"] = self._restored_entries
        return stats


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
    checkpointed = HOT_RERUN_ROUTE_CACHE_CHECKPOINT.import_items(ROUTE_CACHE.export_items(), clear_first=False)
    ROUTE_CACHE.record_checkpoint(checkpointed)
    HOT_RERUN_ROUTE_CACHE_CHECKPOINT.record_checkpoint(checkpointed)
    return checkpointed


def restore_checkpointed_route_cache(*, clear_first: bool = False) -> int:
    restored = ROUTE_CACHE.import_items(
        HOT_RERUN_ROUTE_CACHE_CHECKPOINT.export_items(),
        clear_first=clear_first,
    )
    ROUTE_CACHE.record_restore(restored)
    HOT_RERUN_ROUTE_CACHE_CHECKPOINT.record_restore(restored)
    return restored


def clear_route_cache_checkpoint() -> int:
    return HOT_RERUN_ROUTE_CACHE_CHECKPOINT.clear()


def route_cache_stats() -> dict[str, Any]:
    return ROUTE_CACHE.snapshot()


def route_cache_checkpoint_stats() -> dict[str, Any]:
    return HOT_RERUN_ROUTE_CACHE_CHECKPOINT.snapshot()
