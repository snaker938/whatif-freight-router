from __future__ import annotations

import app.routing_graph as routing_graph
from app.settings import settings


def test_effective_max_hops_uses_base_budget_for_short_od(monkeypatch) -> None:
    monkeypatch.setattr(settings, "route_graph_max_hops", 220)
    monkeypatch.setattr(settings, "route_graph_adaptive_hops_enabled", True)
    monkeypatch.setattr(settings, "route_graph_hops_per_km", 18.0)
    monkeypatch.setattr(settings, "route_graph_hops_detour_factor", 1.35)
    monkeypatch.setattr(settings, "route_graph_max_hops_cap", 6000)

    hops = routing_graph._effective_route_graph_max_hops(
        origin_lat=52.0,
        origin_lon=-1.0,
        destination_lat=52.01,
        destination_lon=-1.01,
    )
    assert hops == 220


def test_effective_max_hops_scales_for_long_od_and_respects_cap(monkeypatch) -> None:
    monkeypatch.setattr(settings, "route_graph_max_hops", 220)
    monkeypatch.setattr(settings, "route_graph_adaptive_hops_enabled", True)
    monkeypatch.setattr(settings, "route_graph_hops_per_km", 18.0)
    monkeypatch.setattr(settings, "route_graph_hops_detour_factor", 1.35)
    monkeypatch.setattr(settings, "route_graph_max_hops_cap", 6000)

    hops = routing_graph._effective_route_graph_max_hops(
        origin_lat=52.93571,
        origin_lon=-1.12610,
        destination_lat=51.48892,
        destination_lon=-0.06592,
    )
    assert hops > 220

    capped_hops = routing_graph._effective_route_graph_max_hops(
        origin_lat=50.0,
        origin_lon=-8.0,
        destination_lat=58.0,
        destination_lon=2.0,
    )
    assert capped_hops == 6000


def test_effective_max_hops_returns_base_when_adaptive_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "route_graph_max_hops", 220)
    monkeypatch.setattr(settings, "route_graph_adaptive_hops_enabled", False)
    monkeypatch.setattr(settings, "route_graph_hops_per_km", 18.0)
    monkeypatch.setattr(settings, "route_graph_hops_detour_factor", 1.35)
    monkeypatch.setattr(settings, "route_graph_max_hops_cap", 6000)

    hops = routing_graph._effective_route_graph_max_hops(
        origin_lat=50.0,
        origin_lon=-8.0,
        destination_lat=58.0,
        destination_lon=2.0,
    )
    assert hops == 220
