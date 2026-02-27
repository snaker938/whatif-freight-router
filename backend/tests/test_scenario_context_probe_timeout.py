from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import app.main as main_module
from app.models import LatLng
from app.settings import settings


def test_scenario_context_probe_timeout_falls_back_and_emits_progress(
    monkeypatch,
) -> None:
    captured_logs: list[tuple[str, dict[str, Any]]] = []
    progress_events: list[dict[str, Any]] = []

    async def _slow_probe(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        _ = kwargs
        await main_module.asyncio.sleep(0.25)
        return [], {}

    async def _progress_cb(payload: dict[str, Any]) -> None:
        progress_events.append(dict(payload))

    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _slow_probe)
    monkeypatch.setattr(settings, "route_context_probe_enabled", True)
    monkeypatch.setattr(settings, "route_context_probe_timeout_ms", 100)
    monkeypatch.setattr(settings, "route_context_probe_max_paths", 2)
    monkeypatch.setattr(
        main_module,
        "log_event",
        lambda event, **kwargs: captured_logs.append((str(event), dict(kwargs))),
    )

    context = asyncio.run(
        main_module._scenario_context_from_od(
            origin=LatLng(lat=52.4862, lon=-1.8904),
            destination=LatLng(lat=51.5072, lon=-0.1276),
            vehicle_class="rigid_hgv",
            departure_time_utc=datetime(2026, 2, 25, 12, 0, tzinfo=UTC),
            weather_bucket="clear",
            progress_cb=_progress_cb,
        )
    )

    assert context.road_mix_bucket in {"mixed", "urban_local_heavy"}
    stage_details = [str(event.get("stage_detail", "")) for event in progress_events]
    assert "scenario_context_probe_start" in stage_details
    assert "scenario_context_probe_timeout_fallback" in stage_details
    assert any(event == "scenario_context_probe_timeout_fallback" for event, _ in captured_logs)


def test_scenario_context_probe_complete_uses_configured_max_paths(
    monkeypatch,
) -> None:
    progress_events: list[dict[str, Any]] = []
    observed: dict[str, Any] = {}

    async def _fast_probe(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        observed["max_paths"] = kwargs.get("max_paths")
        return [
            {"_graph_meta": {"road_mix_counts": {"motorway": 12, "local": 2}}},
            {"_graph_meta": {"road_mix_counts": {"motorway": 8, "local": 2}}},
        ], {}

    async def _progress_cb(payload: dict[str, Any]) -> None:
        progress_events.append(dict(payload))

    monkeypatch.setattr(main_module, "route_graph_status", lambda: (True, "ok"))
    monkeypatch.setattr(main_module, "_route_graph_candidate_routes_async", _fast_probe)
    monkeypatch.setattr(settings, "route_context_probe_enabled", True)
    monkeypatch.setattr(settings, "route_context_probe_timeout_ms", 2_500)
    monkeypatch.setattr(settings, "route_context_probe_max_paths", 3)

    context = asyncio.run(
        main_module._scenario_context_from_od(
            origin=LatLng(lat=52.4862, lon=-1.8904),
            destination=LatLng(lat=51.5072, lon=-0.1276),
            vehicle_class="rigid_hgv",
            departure_time_utc=datetime(2026, 2, 25, 12, 0, tzinfo=UTC),
            weather_bucket="clear",
            progress_cb=_progress_cb,
        )
    )

    assert int(observed.get("max_paths", 0)) == 3
    assert context.road_mix_bucket == "motorway_heavy"
    stage_details = [str(event.get("stage_detail", "")) for event in progress_events]
    assert "scenario_context_probe_start" in stage_details
    assert "scenario_context_probe_complete" in stage_details
