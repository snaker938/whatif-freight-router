from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any

from .settings import settings


@dataclass
class _RouteCacheEntry:
    inserted_at: float
    payload: tuple[list[dict[str, Any]], list[str], int] | tuple[list[dict[str, Any]], list[str], int, dict[str, Any]]


class RouteCacheStore:
    def __init__(self, *, ttl_s: int, max_entries: int) -> None:
        self._ttl_s = max(1, int(ttl_s))
        self._max_entries = max(1, int(max_entries))
        self._lock = Lock()
        self._items: OrderedDict[str, _RouteCacheEntry] = OrderedDict()

        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _is_expired(self, entry: _RouteCacheEntry) -> bool:
        return (time.time() - entry.inserted_at) > self._ttl_s

    def get(
        self, key: str
    ) -> tuple[list[dict[str, Any]], list[str], int] | tuple[list[dict[str, Any]], list[str], int, dict[str, Any]] | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self._misses += 1
                return None

            if self._is_expired(entry):
                self._items.pop(key, None)
                self._misses += 1
                return None

            self._items.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(entry.payload)

    def set(
        self,
        key: str,
        value: tuple[list[dict[str, Any]], list[str], int] | tuple[list[dict[str, Any]], list[str], int, dict[str, Any]],
    ) -> None:
        payload = copy.deepcopy(value)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
            self._items[key] = _RouteCacheEntry(inserted_at=time.time(), payload=payload)

            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)
                self._evictions += 1

    def clear(self) -> int:
        with self._lock:
            cleared = len(self._items)
            self._items.clear()
            return cleared

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._items),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "ttl_s": self._ttl_s,
                "max_entries": self._max_entries,
            }


ROUTE_CACHE = RouteCacheStore(
    ttl_s=settings.route_cache_ttl_s,
    max_entries=settings.route_cache_max_entries,
)


def get_cached_routes(
    key: str,
) -> tuple[list[dict[str, Any]], list[str], int] | tuple[list[dict[str, Any]], list[str], int, dict[str, Any]] | None:
    return ROUTE_CACHE.get(key)


def set_cached_routes(
    key: str,
    value: tuple[list[dict[str, Any]], list[str], int] | tuple[list[dict[str, Any]], list[str], int, dict[str, Any]],
) -> None:
    ROUTE_CACHE.set(key, value)


def clear_route_cache() -> int:
    return ROUTE_CACHE.clear()


def route_cache_stats() -> dict[str, int]:
    return ROUTE_CACHE.snapshot()
