from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


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
    assert "LeaseManifestPath" in script
    assert "-LeaseManifestPath" in script
    assert "-LeaseRole" in script
    assert "-LeaseTopology" in script
    assert '"backend"' in script
    assert 'backend_server_{0}.pid' in script
    assert "Test-BackendProcessMatchesPort" in script
    assert "foreach ($backendProcess in (Get-BackendProcesses))" not in script


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
    assert 'backend_server_{0}.pid' in script
    assert "Test-BackendProcessMatchesPort" in script
    assert "foreach ($backendProcess in (Get-BackendProcesses))" not in script


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell runtime test requires Windows")
def test_run_with_job_memory_limit_script_writes_lease_manifest(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_with_job_memory_limit.ps1"
    manifest_path = tmp_path / "evaluator_lease_manifest.json"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            f"& '{script_path}' "
            f"-MemoryLimitMB 256 "
            f"-WorkingDirectory '{tmp_path}' "
            f"-FilePath '{sys.executable}' "
            f"-ArgumentList @('-c','import time; time.sleep(0.05)') "
            f"-LeaseManifestPath '{manifest_path}' "
            f"-LeaseRole 'evaluator' "
            f"-LeaseTopology 'split_process'"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["role"] == "evaluator"
    assert payload["topology"] == "split_process"
    assert payload["memory_limit_mb"] == 256
    assert Path(payload["resolved_file_path"]) == Path(sys.executable)
    assert payload["argument_list"] == ["-c", "import time; time.sleep(0.05)"]
    assert payload["root_process_pid"] > 0
    assert payload["primary_tracked_process_id"] > 0
    assert payload["status"] == "exited"
    assert payload["exit_code"] == 0


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell runtime test requires Windows")
def test_run_with_job_memory_limit_script_records_limit_exceeded_manifest(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_with_job_memory_limit.ps1"
    manifest_path = tmp_path / "backend_lease_manifest.json"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            f"& '{script_path}' "
            f"-MemoryLimitMB 48 "
            f"-WorkingDirectory '{tmp_path}' "
            f"-FilePath '{sys.executable}' "
            f"-ArgumentList @('-c','import time; chunks=[]; [chunks.append(bytearray(16777216)) or time.sleep(0.15) for _ in range(8)]; time.sleep(5)') "
            f"-LeaseManifestPath '{manifest_path}' "
            f"-LeaseRole 'backend' "
            f"-LeaseTopology 'split_process'"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)

    assert completed.returncode != 0, completed.stderr or completed.stdout
    assert "limit-exceeded" in completed.stdout
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["role"] == "backend"
    assert payload["topology"] == "split_process"
    assert payload["memory_limit_mb"] == 48
    assert payload["status"] == "limit_exceeded"
    assert payload["watchdog_triggered"] is True
    assert payload["exit_code"] == 137
    assert payload["aggregate_working_set_mb"] >= 48.0
    assert payload["root_process_pid"] > 0
    assert payload["primary_tracked_process_id"] > 0
    assert len(payload["tracked_pids"]) >= 1
