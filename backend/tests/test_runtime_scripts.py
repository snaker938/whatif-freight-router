from __future__ import annotations

from pathlib import Path


def test_start_backend_script_hardens_repeated_runs() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "start_backend_logged.ps1"
    ).read_text(encoding="utf-8")

    assert "Get-NetTCPConnection" in script
    assert "app.main:app" in script
    assert "Wait-Process" in script
    assert "Get-ProcessInfoById" in script
    assert "Wait-BackendPortClear" in script
    assert "Backend exited during startup" in script
    assert "refusing to start backend" in script


def test_stop_backend_script_stops_orphaned_listener() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "stop_backend_logged.ps1"
    ).read_text(encoding="utf-8")

    assert "Get-NetTCPConnection" in script
    assert "app.main:app" in script
    assert "Wait-Process" in script
    assert "Get-ProcessInfoById" in script
    assert "Wait-BackendPortClear" in script
    assert "backend_server.pid" in script
