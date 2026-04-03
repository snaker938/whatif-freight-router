from __future__ import annotations

import app._process_cache as process_cache
from app.voi_dccs_cache import (
    VOI_DCCS_CACHE_SCHEMA_VERSION,
    VoiDccsCacheEntry,
    VoiDccsCacheStore,
)


def _selection_payload(score: float) -> dict[str, object]:
    return {
        "selected": [{"candidate_id": "candidate_0", "score": score}],
        "summary": {"dc_yield": 0.5, "score": score},
    }


def test_voi_dccs_cache_deep_copies_payload() -> None:
    cache = VoiDccsCacheStore(ttl_s=60, max_entries=8, max_estimated_bytes=1_000_000)

    assert cache.set("selection", _selection_payload(1.5)) is True
    cached = cache.get("selection")
    assert cached is not None
    cached["selected"][0]["score"] = 9.0

    second = cache.get("selection")
    assert second is not None
    assert second["selected"][0]["score"] == 1.5
    entry = cache.get_entry("selection")
    assert entry is not None
    assert entry.schema_version == VOI_DCCS_CACHE_SCHEMA_VERSION
    assert entry.entry_kind == "dccs_selection"


def test_voi_dccs_cache_expires_and_tracks_stats(monkeypatch) -> None:
    cache = VoiDccsCacheStore(ttl_s=3, max_entries=4, max_estimated_bytes=1_000_000)
    fake_now = 200.0

    monkeypatch.setattr(process_cache.time, "time", lambda: fake_now)
    assert cache.set("selection", _selection_payload(2.0)) is True
    assert cache.get("selection") is not None

    fake_now += 4.0
    assert cache.get("selection") is None

    stats = cache.snapshot()
    assert stats["hits"] == 1
    assert stats["misses"] >= 1


def test_voi_dccs_cache_evicts_by_byte_budget() -> None:
    cache = VoiDccsCacheStore(ttl_s=60, max_entries=10, max_estimated_bytes=1024)

    assert cache.set("a", {"blob": "a" * 400}) is True
    assert cache.set("b", {"blob": "b" * 400}) is True
    assert cache.set("c", {"blob": "c" * 400}) is True

    assert cache.get("a") is None
    assert cache.get("c") is not None
    stats = cache.snapshot()
    assert stats["evictions"] >= 1
    assert stats["estimated_bytes"] <= stats["max_estimated_bytes"]


def test_voi_dccs_cache_invalidates_stale_schema_entry() -> None:
    cache = VoiDccsCacheStore(ttl_s=60, max_entries=8, max_estimated_bytes=1_000_000)
    stale = VoiDccsCacheEntry(
        schema_version=VOI_DCCS_CACHE_SCHEMA_VERSION - 1,
        payload=_selection_payload(3.0),
    )

    assert cache.set("stale", stale) is True
    assert cache.get("stale") is None

    stats = cache.snapshot()
    assert stats["size"] == 0
    assert stats["misses"] >= 1
