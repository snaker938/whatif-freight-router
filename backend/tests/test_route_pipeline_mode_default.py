from __future__ import annotations

import inspect

import app.main as main_module


def test_route_pipeline_default_mode_resolves_to_dccs_refc(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "route_pipeline_default_mode", "dccs_refc", raising=False)
    monkeypatch.setattr(main_module.settings, "route_pipeline_request_override_enabled", True, raising=False)

    assert main_module._resolve_pipeline_mode(None) == "dccs_refc"


def test_route_pipeline_default_mode_rejects_explicit_legacy_on_live_route(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "route_pipeline_default_mode", "dccs_refc", raising=False)
    monkeypatch.setattr(main_module.settings, "route_pipeline_request_override_enabled", True, raising=False)

    effective_mode, error_detail = main_module._resolve_route_request_pipeline_mode(
        requested_mode="legacy",
        waypoint_count=0,
    )

    assert effective_mode == "legacy"
    assert error_detail is not None
    assert error_detail["reason_code"] == "legacy_pipeline_live_route_disabled"
    assert error_detail["supported_live_pipeline_modes"] == ["dccs", "dccs_refc", "voi"]
    assert error_detail["baseline_endpoints"] == ["/route/baseline", "/route/baseline/ors"]


def test_route_pipeline_default_mode_rejects_waypoints_on_live_route(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "route_pipeline_default_mode", "dccs_refc", raising=False)
    monkeypatch.setattr(main_module.settings, "route_pipeline_request_override_enabled", True, raising=False)

    effective_mode, error_detail = main_module._resolve_route_request_pipeline_mode(
        requested_mode=None,
        waypoint_count=1,
    )
    assert effective_mode == "dccs_refc"
    assert error_detail is not None
    assert error_detail["reason_code"] == "waypoints_not_supported_on_live_route"
    assert error_detail["supported_live_pipeline_modes"] == ["dccs", "dccs_refc", "voi"]
    assert error_detail["baseline_endpoints"] == ["/route/baseline", "/route/baseline/ors"]

    effective_mode, error_detail = main_module._resolve_route_request_pipeline_mode(
        requested_mode="dccs_refc",
        waypoint_count=1,
    )
    assert effective_mode == "dccs_refc"
    assert error_detail is not None
    assert error_detail["reason_code"] == "waypoints_not_supported_on_live_route"


def test_route_pipeline_default_mode_rejects_explicit_legacy_waypoint_path(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "route_pipeline_default_mode", "dccs_refc", raising=False)
    monkeypatch.setattr(main_module.settings, "route_pipeline_request_override_enabled", True, raising=False)

    effective_mode, error_detail = main_module._resolve_route_request_pipeline_mode(
        requested_mode="legacy",
        waypoint_count=2,
    )

    assert effective_mode == "legacy"
    assert error_detail is not None
    assert error_detail["reason_code"] == "legacy_pipeline_live_route_disabled"


def test_live_route_handler_no_longer_contains_legacy_pipeline_branch() -> None:
    source = inspect.getsource(main_module.compute_route)

    assert 'actual_pipeline_mode == "legacy"' not in source
    assert "legacy_strict_frontier" not in source
