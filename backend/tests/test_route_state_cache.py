from __future__ import annotations

from dataclasses import dataclass

import app._process_cache as process_cache
from app.route_state_cache import RouteStateCacheStore


@dataclass
class _TerrainDiag:
    coverage_ratio: float


def _payload() -> tuple[object, ...]:
    return (
        [{"id": "route_0", "metrics": {"duration_s": 10.0}}],
        [{"id": "route_0"}],
        [{"id": "route_0"}],
        {"id": "route_0"},
        _TerrainDiag(coverage_ratio=0.95),
        {"route_0": ["candidate_0"]},
        {"route_0": 1.23},
    )


def test_route_state_cache_deep_copies_round_trip() -> None:
    cache = RouteStateCacheStore(ttl_s=60, max_entries=4, max_estimated_bytes=1_000_000)
    original = _payload()

    assert cache.set("state", original) is True
    cached = cache.get("state")
    assert cached is not None

    assert cached[0][0]["metrics"]["duration_s"] == 10.0
    cached[0][0]["metrics"]["duration_s"] = 99.0
    cached[5]["route_0"].append("candidate_1")

    second = cache.get("state")
    assert second is not None
    assert second[0][0]["metrics"]["duration_s"] == 10.0
    assert second[5]["route_0"] == ["candidate_0"]


def test_route_state_cache_expires_entries(monkeypatch) -> None:
    cache = RouteStateCacheStore(ttl_s=5, max_entries=2, max_estimated_bytes=1_000_000)
    fake_now = 1000.0

    monkeypatch.setattr(process_cache.time, "time", lambda: fake_now)
    assert cache.set("state", _payload()) is True
    assert cache.get("state") is not None

    fake_now += 6.0
    assert cache.get("state") is None
    assert cache.snapshot()["misses"] >= 1


def test_route_state_cache_evicts_oldest_entry() -> None:
    cache = RouteStateCacheStore(ttl_s=60, max_entries=2, max_estimated_bytes=1_000_000)

    assert cache.set("a", _payload()) is True
    assert cache.set("b", _payload()) is True
    assert cache.set("c", _payload()) is True

    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert cache.snapshot()["evictions"] == 1


def test_route_state_cache_rejects_oversize_payload() -> None:
    cache = RouteStateCacheStore(ttl_s=60, max_entries=2, max_estimated_bytes=128)

    accepted = cache.set(
        "oversize",
        (
            [{"id": "route_0", "blob": "x" * 2048}],
            [],
            [],
            {"id": "route_0"},
            _TerrainDiag(coverage_ratio=1.0),
            {},
            {},
        ),
    )

    assert accepted is False
    assert cache.get("oversize") is None
    stats = cache.snapshot()
    assert stats["oversize_rejections"] == 1
    assert stats["size"] == 0
