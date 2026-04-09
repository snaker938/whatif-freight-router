from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from ._process_cache import ProcessGlobalCacheStore
from .decision_critical import DCCSResult
from .settings import settings

VOI_DCCS_CACHE_SCHEMA_VERSION = 1
VOI_DCCS_CACHE_ENTRY_KIND = "dccs_selection"
VoiDccsCachePayload: TypeAlias = DCCSResult | dict[str, Any]


@dataclass(frozen=True)
class VoiDccsCacheEntry:
    schema_version: int = VOI_DCCS_CACHE_SCHEMA_VERSION
    entry_kind: str = VOI_DCCS_CACHE_ENTRY_KIND
    payload: VoiDccsCachePayload = field(default_factory=dict)


class VoiDccsCacheStore(ProcessGlobalCacheStore[VoiDccsCacheEntry]):
    def __init__(self, *, ttl_s: int, max_entries: int, max_estimated_bytes: int) -> None:
        super().__init__(
            ttl_s=ttl_s,
            max_entries=max_entries,
            max_estimated_bytes=max_estimated_bytes,
        )
        self._invalidation_counters: dict[str, int] = {
            "expired": 0,
            "schema_mismatch": 0,
            "manual_clear": 0,
        }

    def get_entry(self, key: str) -> VoiDccsCacheEntry | None:
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
            entry = copy.deepcopy(cache_entry.payload)
            if (
                entry.schema_version != VOI_DCCS_CACHE_SCHEMA_VERSION
                or entry.entry_kind != VOI_DCCS_CACHE_ENTRY_KIND
            ):
                self._remove_key(key)
                self._invalidation_counters["schema_mismatch"] = (
                    self._invalidation_counters.get("schema_mismatch", 0) + 1
                )
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return entry

    def get(self, key: str) -> VoiDccsCachePayload | None:
        entry = self.get_entry(key)
        return None if entry is None else entry.payload

    def set_entry(self, key: str, value: VoiDccsCacheEntry) -> bool:
        return super().set(key, value)

    def set(self, key: str, value: VoiDccsCachePayload | VoiDccsCacheEntry) -> bool:
        if isinstance(value, VoiDccsCacheEntry):
            return self.set_entry(key, value)
        return self.set_entry(
            key,
            VoiDccsCacheEntry(payload=value),
        )

    def clear(self) -> int:
        cleared = super().clear()
        if cleared > 0:
            with self._lock:
                self._invalidation_counters["manual_clear"] = (
                    self._invalidation_counters.get("manual_clear", 0) + cleared
                )
        return cleared

    def snapshot(self) -> dict[str, Any]:
        stats = super().snapshot()
        with self._lock:
            stats["schema_version"] = VOI_DCCS_CACHE_SCHEMA_VERSION
            stats["invalidation_counters"] = dict(self._invalidation_counters)
        return stats


VOI_DCCS_CACHE = VoiDccsCacheStore(
    ttl_s=settings.voi_dccs_cache_ttl_s,
    max_entries=settings.voi_dccs_cache_max_entries,
    max_estimated_bytes=settings.voi_dccs_cache_max_estimated_bytes,
)


def get_cached_voi_dccs(key: str) -> VoiDccsCachePayload | None:
    return VOI_DCCS_CACHE.get(key)


def set_cached_voi_dccs(key: str, value: VoiDccsCachePayload | VoiDccsCacheEntry) -> bool:
    return VOI_DCCS_CACHE.set(key, value)


def clear_voi_dccs_cache() -> int:
    return VOI_DCCS_CACHE.clear()


def voi_dccs_cache_stats() -> dict[str, Any]:
    return VOI_DCCS_CACHE.snapshot()
