from __future__ import annotations

import copy
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from threading import Lock
from typing import Any, Generic, TypeVar

PayloadT = TypeVar("PayloadT")


def estimate_deep_size_bytes(value: Any, *, _seen: set[int] | None = None) -> int:
    """Best-effort deep size estimate for cache admission/eviction decisions."""

    seen = _seen if _seen is not None else set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)

    size = sys.getsizeof(value)

    if isinstance(value, dict):
        for key, nested in value.items():
            size += estimate_deep_size_bytes(key, _seen=seen)
            size += estimate_deep_size_bytes(nested, _seen=seen)
        return size

    if isinstance(value, (list, tuple, set, frozenset)):
        for nested in value:
            size += estimate_deep_size_bytes(nested, _seen=seen)
        return size

    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            size += estimate_deep_size_bytes(getattr(value, field.name), _seen=seen)
        return size

    if hasattr(value, "__dict__"):
        size += estimate_deep_size_bytes(vars(value), _seen=seen)
        return size

    if hasattr(value, "__slots__"):
        for slot in value.__slots__:
            if hasattr(value, slot):
                size += estimate_deep_size_bytes(getattr(value, slot), _seen=seen)
        return size

    return size


@dataclass
class _CacheEntry(Generic[PayloadT]):
    inserted_at: float
    estimated_bytes: int
    payload: PayloadT


class ProcessGlobalCacheStore(Generic[PayloadT]):
    def __init__(
        self,
        *,
        ttl_s: int,
        max_entries: int,
        max_estimated_bytes: int = 0,
    ) -> None:
        self._ttl_s = max(1, int(ttl_s))
        self._max_entries = max(1, int(max_entries))
        self._max_estimated_bytes = max(0, int(max_estimated_bytes))
        self._lock = Lock()
        self._items: OrderedDict[str, _CacheEntry[PayloadT]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversize_rejections = 0
        self._estimated_bytes = 0

    def _is_expired(self, entry: _CacheEntry[PayloadT]) -> bool:
        return (time.time() - entry.inserted_at) > self._ttl_s

    def _remove_key(self, key: str) -> None:
        entry = self._items.pop(key, None)
        if entry is not None:
            self._estimated_bytes = max(0, self._estimated_bytes - entry.estimated_bytes)

    def _trim_locked(self) -> None:
        while self._items and len(self._items) > self._max_entries:
            _, entry = self._items.popitem(last=False)
            self._estimated_bytes = max(0, self._estimated_bytes - entry.estimated_bytes)
            self._evictions += 1

        while (
            self._items
            and self._max_estimated_bytes > 0
            and self._estimated_bytes > self._max_estimated_bytes
        ):
            _, entry = self._items.popitem(last=False)
            self._estimated_bytes = max(0, self._estimated_bytes - entry.estimated_bytes)
            self._evictions += 1

    def get(self, key: str) -> PayloadT | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self._misses += 1
                return None
            if self._is_expired(entry):
                self._remove_key(key)
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(entry.payload)

    def set(self, key: str, value: PayloadT) -> bool:
        payload = copy.deepcopy(value)
        estimated_bytes = estimate_deep_size_bytes(payload)
        with self._lock:
            if self._max_estimated_bytes > 0 and estimated_bytes > self._max_estimated_bytes:
                self._remove_key(key)
                self._oversize_rejections += 1
                return False
            if key in self._items:
                self._remove_key(key)
            self._items[key] = _CacheEntry(
                inserted_at=time.time(),
                estimated_bytes=estimated_bytes,
                payload=payload,
            )
            self._estimated_bytes += estimated_bytes
            self._items.move_to_end(key)
            self._trim_locked()
            return key in self._items

    def clear(self) -> int:
        with self._lock:
            cleared = len(self._items)
            self._items.clear()
            self._estimated_bytes = 0
            return cleared

    def export_items(self) -> list[tuple[str, PayloadT]]:
        with self._lock:
            expired_keys = [
                key
                for key, entry in self._items.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                self._remove_key(key)
            return [
                (key, copy.deepcopy(entry.payload))
                for key, entry in self._items.items()
            ]

    def import_items(
        self,
        items: list[tuple[str, PayloadT]],
        *,
        clear_first: bool = False,
    ) -> int:
        if clear_first:
            self.clear()
        inserted = 0
        for key, payload in items:
            if self.set(key, payload):
                inserted += 1
        return inserted

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._items),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "oversize_rejections": self._oversize_rejections,
                "ttl_s": self._ttl_s,
                "max_entries": self._max_entries,
                "estimated_bytes": self._estimated_bytes,
                "max_estimated_bytes": self._max_estimated_bytes,
            }
