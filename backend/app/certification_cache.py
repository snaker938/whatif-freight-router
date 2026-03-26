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


def get_cached_certification(key: str) -> CertificationCachePayload | None:
    return CERTIFICATION_CACHE.get(key)


def set_cached_certification(key: str, value: CertificationCachePayload) -> bool:
    return CERTIFICATION_CACHE.set(key, value)


def clear_certification_cache() -> int:
    return CERTIFICATION_CACHE.clear()


def certification_cache_stats() -> dict[str, int]:
    return CERTIFICATION_CACHE.snapshot()
