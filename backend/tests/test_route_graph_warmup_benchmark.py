from __future__ import annotations

from pathlib import Path

from scripts import benchmark_route_graph_warmup


def test_build_child_command_includes_child_mode() -> None:
    script_path = Path("C:/tmp/benchmark_route_graph_warmup.py")
    asset_path = Path("C:/tmp/routing_graph_uk.json")
    command = benchmark_route_graph_warmup._build_child_command(
        script_path=script_path,
        asset_path=asset_path,
        timeout_s=900,
        poll_interval_s=0.5,
    )

    assert command[0]
    assert "--child" in command
    assert "--asset-path" in command
    assert str(asset_path) in command
    assert "--timeout-s" in command
    assert "900" in command


def test_default_asset_path_points_at_model_assets() -> None:
    asset_path = benchmark_route_graph_warmup._default_asset_path()

    assert asset_path.as_posix().endswith("backend/out/model_assets/routing_graph_uk.json")
