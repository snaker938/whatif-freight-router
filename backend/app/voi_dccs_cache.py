from __future__ import annotations

from typing import Any

from ._process_cache import ProcessGlobalCacheStore
from .settings import settings


class VoiDccsCacheStore(ProcessGlobalCacheStore[Any]):
    pass


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


def voi_dccs_cache_stats() -> dict[str, int]:
    return VOI_DCCS_CACHE.snapshot()


def voi_dccs_cache_checkpoint_stats() -> dict[str, int]:
    return HOT_RERUN_VOI_DCCS_CACHE_CHECKPOINT.snapshot()
