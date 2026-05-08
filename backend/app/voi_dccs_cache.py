from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ._process_cache import ProcessGlobalCacheStore
from .settings import settings

VOI_DCCS_CACHE_SCHEMA_VERSION = 1
VOI_DCCS_CACHE_ENTRY_KIND = "dccs_selection"


@dataclass(frozen=True)
class VoiDccsCacheEntry:
    schema_version: int
    payload: Any
    entry_kind: str = VOI_DCCS_CACHE_ENTRY_KIND


class VoiDccsCacheStore(ProcessGlobalCacheStore[VoiDccsCacheEntry]):
    def __init__(
        self,
        *,
        ttl_s: int,
        max_entries: int,
        max_estimated_bytes: int = 0,
    ) -> None:
        super().__init__(
            ttl_s=ttl_s,
            max_entries=max_entries,
            max_estimated_bytes=max_estimated_bytes,
        )
        self._invalidation_counters = {
            "expired": 0,
            "schema_mismatch": 0,
            "manual_clear": 0,
        }

    def _entry_for(self, value: Any) -> VoiDccsCacheEntry:
        if isinstance(value, VoiDccsCacheEntry):
            return value
        return VoiDccsCacheEntry(
            schema_version=VOI_DCCS_CACHE_SCHEMA_VERSION,
            payload=value,
        )

    def _is_current_schema(self, entry: VoiDccsCacheEntry) -> bool:
        return (
            entry.schema_version == VOI_DCCS_CACHE_SCHEMA_VERSION
            and entry.entry_kind == VOI_DCCS_CACHE_ENTRY_KIND
        )

    def set(self, key: str, value: Any) -> bool:
        return super().set(key, self._entry_for(value))

    def get_entry(self, key: str) -> VoiDccsCacheEntry | None:
        with self._lock:
            stored = self._items.get(key)
            if stored is None:
                self._misses += 1
                return None
            if self._is_expired(stored):
                self._remove_key(key)
                self._misses += 1
                self._invalidation_counters["expired"] += 1
                return None

            entry = stored.payload
            if not self._is_current_schema(entry):
                self._remove_key(key)
                self._misses += 1
                self._invalidation_counters["schema_mismatch"] += 1
                return None

            self._items.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(entry)

    def get(self, key: str) -> Any | None:
        entry = self.get_entry(key)
        if entry is None:
            return None
        return copy.deepcopy(entry.payload)

    def clear(self) -> int:
        with self._lock:
            cleared = len(self._items)
            self._items.clear()
            self._estimated_bytes = 0
            if cleared:
                self._invalidation_counters["manual_clear"] += cleared
            return cleared

    def export_items(self) -> list[tuple[str, Any]]:
        with self._lock:
            expired_keys: list[str] = []
            stale_keys: list[str] = []
            exported: list[tuple[str, Any]] = []
            for key, stored in self._items.items():
                if self._is_expired(stored):
                    expired_keys.append(key)
                    continue
                entry = stored.payload
                if not self._is_current_schema(entry):
                    stale_keys.append(key)
                    continue
                exported.append((key, copy.deepcopy(entry.payload)))

            for key in expired_keys:
                self._remove_key(key)
            for key in stale_keys:
                self._remove_key(key)
            self._invalidation_counters["expired"] += len(expired_keys)
            self._invalidation_counters["schema_mismatch"] += len(stale_keys)
            return exported

    def import_items(
        self,
        items: list[tuple[str, Any]],
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

    def snapshot(self) -> dict[str, Any]:
        stats = super().snapshot()
        stats["schema_version"] = VOI_DCCS_CACHE_SCHEMA_VERSION
        stats["invalidation_counters"] = self._invalidation_counters.copy()
        return stats


VOI_DCCS_CACHE = VoiDccsCacheStore(
    ttl_s=settings.voi_dccs_cache_ttl_s,
    max_entries=settings.voi_dccs_cache_max_entries,
    max_estimated_bytes=settings.voi_dccs_cache_max_estimated_bytes,
)
HOT_RERUN_VOI_DCCS_CACHE_CHECKPOINT = VoiDccsCacheStore(
    ttl_s=settings.voi_dccs_cache_ttl_s,
    max_entries=settings.voi_dccs_cache_max_entries,
    max_estimated_bytes=settings.voi_dccs_cache_max_estimated_bytes,
)


def get_cached_voi_dccs(key: str) -> Any | None:
    return VOI_DCCS_CACHE.get(key)


def set_cached_voi_dccs(key: str, value: Any) -> bool:
    return VOI_DCCS_CACHE.set(key, value)


def clear_voi_dccs_cache() -> int:
    return VOI_DCCS_CACHE.clear()


def checkpoint_voi_dccs_cache() -> int:
    return HOT_RERUN_VOI_DCCS_CACHE_CHECKPOINT.import_items(
        VOI_DCCS_CACHE.export_items(),
        clear_first=True,
    )


def restore_checkpointed_voi_dccs_cache(*, clear_first: bool = False) -> int:
    return VOI_DCCS_CACHE.import_items(
        HOT_RERUN_VOI_DCCS_CACHE_CHECKPOINT.export_items(),
        clear_first=clear_first,
    )


def clear_voi_dccs_cache_checkpoint() -> int:
    return HOT_RERUN_VOI_DCCS_CACHE_CHECKPOINT.clear()


def voi_dccs_cache_stats() -> dict[str, Any]:
    return VOI_DCCS_CACHE.snapshot()


def voi_dccs_cache_checkpoint_stats() -> dict[str, Any]:
    return HOT_RERUN_VOI_DCCS_CACHE_CHECKPOINT.snapshot()
