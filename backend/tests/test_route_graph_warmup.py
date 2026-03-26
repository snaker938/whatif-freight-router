from __future__ import annotations

import pickle
from types import SimpleNamespace
from pathlib import Path

from app import routing_graph
from app.settings import settings


def _graph_fixture() -> routing_graph.RouteGraph:
    return routing_graph.RouteGraph(
        version="v-test",
        source="repo_local",
        nodes={"n1": (52.0, -1.0)},
        adjacency={"n1": ()},
        edge_index={},
        grid_index={(346, -7): ("n1",)},
        component_by_node={"n1": 0},
        component_sizes={0: 1},
        component_count=1,
        largest_component_nodes=1,
        largest_component_ratio=1.0,
        graph_fragmented=False,
        min_cost_per_meter=0.0,
    )


def test_binary_cache_is_skipped_for_oversize_warmup(monkeypatch, tmp_path) -> None:
    asset_path = tmp_path / "routing_graph_uk.json"
    cache_path = tmp_path / "routing_graph_uk.json.pkl"
    asset_path.write_text("{}", encoding="utf-8")
    cache_path.write_bytes(b"xx")

    signature = {
        "path": str(asset_path),
        "size": int(asset_path.stat().st_size),
        "mtime_ns": 1,
        "version": "v-test",
        "source": "repo_local",
    }
    phases: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(settings, "route_graph_binary_cache_enabled", True)
    monkeypatch.setattr(settings, "route_graph_binary_cache_warmup_max_bytes", 1)
    monkeypatch.setattr(routing_graph, "_graph_asset_signature", lambda path: signature)
    monkeypatch.setattr(routing_graph, "_is_route_graph_warmup_thread", lambda: True)
    monkeypatch.setattr(routing_graph, "_warmup_deadline_monotonic", lambda: None)
    monkeypatch.setattr(routing_graph, "_set_warmup_phase", lambda phase, **kwargs: phases.append(phase))
    monkeypatch.setattr(routing_graph, "log_event", lambda name, **kwargs: events.append((name, kwargs)))

    graph = routing_graph._load_route_graph_binary_cache(asset_path)

    assert graph is None
    assert phases == ["cache_probe", "cache_signature_check", "binary_cache_skipped_oversize"]
    assert any(name == "route_graph_binary_cache_skipped" for name, _ in events)


def test_binary_cache_uses_large_cache_when_warmup_cap_disabled(monkeypatch, tmp_path) -> None:
    asset_path = tmp_path / "routing_graph_uk.json"
    cache_path = tmp_path / "routing_graph_uk.json.pkl"
    meta_path = tmp_path / "routing_graph_uk.json.pkl.meta.json"
    asset_path.write_text("{}", encoding="utf-8")

    signature = {
        "path": str(asset_path),
        "size": int(asset_path.stat().st_size),
        "mtime_ns": 1,
        "version": "v-test",
        "source": "repo_local",
    }
    with cache_path.open("wb") as fh:
        pickle.dump({"asset": signature, "graph": _graph_fixture()}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(
        '{"asset": {"path": "%s", "size": %d, "mtime_ns": 1, "version": "v-test", "source": "repo_local"}}'
        % (str(asset_path).replace("\\", "\\\\"), int(asset_path.stat().st_size)),
        encoding="utf-8",
    )
    original_stat = Path.stat
    phases: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []

    def _fake_stat(self: Path):
        real = original_stat(self)
        if self == cache_path:
            return SimpleNamespace(
                st_size=5_000_000_000,
                st_mtime=getattr(real, "st_mtime", 0.0),
                st_mtime_ns=getattr(real, "st_mtime_ns", 0),
            )
        return real

    monkeypatch.setattr(Path, "stat", _fake_stat)
    monkeypatch.setattr(settings, "route_graph_binary_cache_enabled", True)
    monkeypatch.setattr(settings, "route_graph_binary_cache_warmup_max_bytes", 0)
    monkeypatch.setattr(routing_graph, "_graph_asset_signature", lambda path: signature)
    monkeypatch.setattr(routing_graph, "_is_route_graph_warmup_thread", lambda: True)
    monkeypatch.setattr(routing_graph, "_warmup_deadline_monotonic", lambda: None)
    monkeypatch.setattr(routing_graph, "_set_warmup_phase", lambda phase, **kwargs: phases.append(phase))
    monkeypatch.setattr(routing_graph, "log_event", lambda name, **kwargs: events.append((name, kwargs)))

    graph = routing_graph._load_route_graph_binary_cache(asset_path)

    assert isinstance(graph, routing_graph.RouteGraph)
    assert "binary_cache_skipped_oversize" not in phases
    assert any(name == "route_graph_binary_cache_loading" for name, _ in events)


def test_binary_cache_warmup_emits_loading_and_validation_phases(monkeypatch, tmp_path) -> None:
    asset_path = tmp_path / "routing_graph_uk.json"
    cache_path = tmp_path / "routing_graph_uk.json.pkl"
    meta_path = tmp_path / "routing_graph_uk.json.pkl.meta.json"
    asset_path.write_text("{}", encoding="utf-8")
    signature = {
        "path": str(asset_path),
        "size": int(asset_path.stat().st_size),
        "mtime_ns": 1,
        "version": "v-test",
        "source": "repo_local",
    }
    with cache_path.open("wb") as fh:
        pickle.dump({"asset": signature, "graph": _graph_fixture()}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text('{"asset": {"path": "%s", "size": %d, "mtime_ns": 1, "version": "v-test", "source": "repo_local"}}' % (str(asset_path).replace("\\", "\\\\"), int(asset_path.stat().st_size)), encoding="utf-8")

    phases: list[str] = []
    monkeypatch.setattr(settings, "route_graph_binary_cache_enabled", True)
    monkeypatch.setattr(settings, "route_graph_binary_cache_warmup_max_bytes", 10_000_000)
    monkeypatch.setattr(routing_graph, "_graph_asset_signature", lambda path: signature)
    monkeypatch.setattr(routing_graph, "_is_route_graph_warmup_thread", lambda: True)
    monkeypatch.setattr(routing_graph, "_warmup_deadline_monotonic", lambda: None)
    monkeypatch.setattr(routing_graph, "_set_warmup_phase", lambda phase, **kwargs: phases.append(phase))
    monkeypatch.setattr(routing_graph, "log_event", lambda *args, **kwargs: None)

    graph = routing_graph._load_route_graph_binary_cache(asset_path)

    assert isinstance(graph, routing_graph.RouteGraph)
    assert phases[:4] == [
        "cache_probe",
        "cache_signature_check",
        "loading_binary_cache",
        "validating_binary_cache_payload",
    ]


def test_load_route_graph_prefers_compact_bundle_over_binary_cache(monkeypatch, tmp_path) -> None:
    path = tmp_path / "routing_graph_uk.json"
    path.write_text("{}", encoding="utf-8")
    expected = _graph_fixture()
    calls: list[str] = []

    monkeypatch.setattr(routing_graph.settings, "route_graph_asset_path", str(path))
    monkeypatch.setattr(routing_graph, "ijson", object())

    def _unexpected_binary_loader(asset_path: Path) -> routing_graph.RouteGraph | None:
        calls.append("binary")
        raise AssertionError("binary cache should not be consulted when the compact bundle is valid")

    def _compact_loader(asset_path: Path) -> routing_graph.RouteGraph | None:
        calls.append("compact")
        return expected

    monkeypatch.setattr(routing_graph, "_load_route_graph_binary_cache", _unexpected_binary_loader)
    monkeypatch.setattr(routing_graph, "_load_route_graph_compact_bundle", _compact_loader)
    load_route_graph = routing_graph.load_route_graph
    load_route_graph.cache_clear()
    try:
        result = load_route_graph()
        assert result is expected
        assert calls == ["compact"]
    finally:
        load_route_graph.cache_clear()


def test_load_route_graph_falls_back_to_binary_cache_when_compact_bundle_is_invalid(monkeypatch, tmp_path) -> None:
    path = tmp_path / "routing_graph_uk.json"
    path.write_text("{}", encoding="utf-8")
    expected = _graph_fixture()
    calls: list[str] = []

    monkeypatch.setattr(routing_graph.settings, "route_graph_asset_path", str(path))
    monkeypatch.setattr(routing_graph, "ijson", object())

    def _invalid_compact_loader(path: Path) -> routing_graph.RouteGraph | None:
        calls.append("compact")
        return None

    def _fallback_binary_loader(path: Path) -> routing_graph.RouteGraph | None:
        calls.append("binary")
        return expected

    monkeypatch.setattr(routing_graph, "_load_route_graph_binary_cache", _fallback_binary_loader)
    monkeypatch.setattr(routing_graph, "_load_route_graph_compact_bundle", _invalid_compact_loader)
    load_route_graph = routing_graph.load_route_graph
    load_route_graph.cache_clear()
    try:
        result = load_route_graph()
        assert result is expected
        assert calls == ["compact", "binary"]
    finally:
        load_route_graph.cache_clear()


def test_compact_bundle_graph_uses_adjacency_lookup_when_edge_index_missing() -> None:
    edge = routing_graph.GraphEdge(
        to="b",
        cost=10.0,
        distance_m=1000.0,
        highway="primary",
        toll=False,
        maxspeed_kph=70.0,
    )
    graph = routing_graph.RouteGraph(
        version="pytest",
        source="pytest.pbf",
        nodes={"a": (52.5, -1.9), "b": (52.6, -1.8)},
        adjacency={"a": (edge,)},
        grid_index={routing_graph._grid_key(52.5, -1.9): ("a", "b")},
        component_by_node={"a": 1, "b": 1},
        component_sizes={1: 2},
        component_count=1,
        largest_component_nodes=2,
        largest_component_ratio=1.0,
        graph_fragmented=False,
        min_cost_per_meter=0.01,
        edge_index={},
    )
    looked_up = routing_graph._graph_edge_lookup(graph, "a", "b")
    assert looked_up == edge
    assert routing_graph._path_distance_m(graph, ("a", "b")) == 1000.0
