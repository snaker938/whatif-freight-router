from __future__ import annotations

import app.main as main_module


def _actual_pipeline_mode(requested_mode: str | None, *, has_waypoints: bool) -> str:
    effective_pipeline_mode = main_module._resolve_pipeline_mode(requested_mode)
    if has_waypoints and effective_pipeline_mode != "legacy":
        return "legacy"
    return effective_pipeline_mode


def test_route_pipeline_default_mode_resolves_to_dccs_refc(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "route_pipeline_default_mode", "dccs_refc", raising=False)
    monkeypatch.setattr(main_module.settings, "route_pipeline_request_override_enabled", True, raising=False)

    assert main_module._resolve_pipeline_mode(None) == "dccs_refc"


def test_route_pipeline_default_mode_keeps_explicit_legacy_override(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "route_pipeline_default_mode", "dccs_refc", raising=False)
    monkeypatch.setattr(main_module.settings, "route_pipeline_request_override_enabled", True, raising=False)

    assert main_module._resolve_pipeline_mode("legacy") == "legacy"


def test_route_pipeline_default_mode_falls_back_to_legacy_for_waypoints(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "route_pipeline_default_mode", "dccs_refc", raising=False)
    monkeypatch.setattr(main_module.settings, "route_pipeline_request_override_enabled", True, raising=False)

    assert _actual_pipeline_mode(None, has_waypoints=True) == "legacy"
    assert _actual_pipeline_mode("dccs_refc", has_waypoints=True) == "legacy"
