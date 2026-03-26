from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any

from .settings import settings


@dataclass
class _KRawCacheEntry:
    inserted_at: float
    payload: tuple[list[dict[str, Any]], Any, dict[str, Any]]


class KRawCacheStore:
    def __init__(self, *, ttl_s: int, max_entries: int) -> None:
        self._ttl_s = max(1, int(ttl_s))
        self._max_entries = max(1, int(max_entries))
        self._lock = Lock()
        self._items: OrderedDict[str, _KRawCacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _is_expired(self, entry: _KRawCacheEntry) -> bool:
        return (time.time() - entry.inserted_at) > self._ttl_s

    def get(self, key: str) -> tuple[list[dict[str, Any]], Any, dict[str, Any]] | None:
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

    def set(self, key: str, value: tuple[list[dict[str, Any]], Any, dict[str, Any]]) -> None:
        payload = copy.deepcopy(value)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
            self._items[key] = _KRawCacheEntry(inserted_at=time.time(), payload=payload)
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


K_RAW_CACHE = KRawCacheStore(
    ttl_s=settings.route_k_raw_cache_ttl_s,
    max_entries=settings.route_k_raw_cache_max_entries,
)


def get_cached_k_raw(key: str) -> tuple[list[dict[str, Any]], Any, dict[str, Any]] | None:
    return K_RAW_CACHE.get(key)


def set_cached_k_raw(key: str, value: tuple[list[dict[str, Any]], Any, dict[str, Any]]) -> None:
    K_RAW_CACHE.set(key, value)


def clear_k_raw_cache() -> int:
    return K_RAW_CACHE.clear()


def k_raw_cache_stats() -> dict[str, int]:
    return K_RAW_CACHE.snapshot()
