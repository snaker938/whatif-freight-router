from __future__ import annotations

from typing import Any

from ._process_cache import ProcessGlobalCacheStore
from .settings import settings

CertificationCachePayload = tuple[Any, Any, dict[str, Any], list[str]]


class CertificationCacheStore(ProcessGlobalCacheStore[CertificationCachePayload]):
    pass


CERTIFICATION_CACHE = CertificationCacheStore(
    ttl_s=settings.route_certification_cache_ttl_s,
    max_entries=settings.route_certification_cache_max_entries,
    max_estimated_bytes=settings.route_certification_cache_max_estimated_bytes,
)
HOT_RERUN_CERTIFICATION_CACHE_CHECKPOINT = CertificationCacheStore(
    ttl_s=settings.route_certification_cache_ttl_s,
    max_entries=settings.route_certification_cache_max_entries,
    max_estimated_bytes=settings.route_certification_cache_max_estimated_bytes,
)


def get_cached_certification(key: str) -> CertificationCachePayload | None:
    return CERTIFICATION_CACHE.get(key)


def set_cached_certification(key: str, value: CertificationCachePayload) -> bool:
    return CERTIFICATION_CACHE.set(key, value)


def clear_certification_cache() -> int:
    return CERTIFICATION_CACHE.clear()


def checkpoint_certification_cache() -> int:
    return HOT_RERUN_CERTIFICATION_CACHE_CHECKPOINT.import_items(
        CERTIFICATION_CACHE.export_items(),
        clear_first=False,
    )


def restore_checkpointed_certification_cache(*, clear_first: bool = False) -> int:
    return CERTIFICATION_CACHE.import_items(
        HOT_RERUN_CERTIFICATION_CACHE_CHECKPOINT.export_items(),
        clear_first=clear_first,
    )


def clear_certification_cache_checkpoint() -> int:
    return HOT_RERUN_CERTIFICATION_CACHE_CHECKPOINT.clear()


def certification_cache_stats() -> dict[str, int]:
    return CERTIFICATION_CACHE.snapshot()


def certification_cache_checkpoint_stats() -> dict[str, int]:
    return HOT_RERUN_CERTIFICATION_CACHE_CHECKPOINT.snapshot()
