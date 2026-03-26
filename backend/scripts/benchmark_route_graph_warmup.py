from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency in local dev envs
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _default_asset_path() -> Path:
    return ROOT / "out" / "model_assets" / "routing_graph_uk.json"


def _normalize_asset_path(asset_path: Path) -> Path:
    if asset_path.is_absolute():
        return asset_path
    return (Path.cwd() / asset_path).resolve()


def _build_child_command(
    *,
    script_path: Path,
    asset_path: Path,
    timeout_s: int,
    poll_interval_s: float,
) -> list[str]:
    normalized_asset_path = _normalize_asset_path(asset_path)
    return [
        sys.executable,
        str(script_path),
        "--child",
        "--asset-path",
        str(normalized_asset_path),
        "--timeout-s",
        str(int(timeout_s)),
        "--poll-interval-s",
        str(float(poll_interval_s)),
    ]


def _process_rss_bytes(pid: int) -> int | None:
    if psutil is not None:
        try:
            return int(psutil.Process(pid).memory_info().rss)
        except Exception:
            pass
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        process = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, int(pid))
        if not process:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            result = ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            if not result:
                return None
            return int(counters.WorkingSetSize)
        finally:
            ctypes.windll.kernel32.CloseHandle(process)
    except Exception:
        return None


def _run_child(
    *,
    asset_path: Path,
    timeout_s: int,
    poll_interval_s: float,
) -> dict[str, Any]:
    from app import routing_graph
    from app.settings import settings

    routing_graph.log_event = lambda *args, **kwargs: None  # type: ignore[assignment]
    asset_path = _normalize_asset_path(asset_path)
    settings.route_graph_asset_path = str(asset_path)
    settings.route_graph_fast_startup_enabled = False
    settings.route_graph_binary_cache_warmup_max_bytes = 0
    settings.route_graph_warmup_timeout_s = max(60, int(timeout_s))
    routing_graph.load_route_graph.cache_clear()
    routing_graph.begin_route_graph_warmup(force=True)

    deadline = time.monotonic() + float(timeout_s)
    status: dict[str, Any] = routing_graph.route_graph_warmup_status()
    child_peak_rss_bytes = 0
    while time.monotonic() < deadline:
        rss = _process_rss_bytes(os.getpid())
        if rss is not None and rss > child_peak_rss_bytes:
            child_peak_rss_bytes = rss
        if str(status.get("state", "")).strip().lower() in {"ready", "failed"}:
            break
        time.sleep(max(0.05, float(poll_interval_s)))
        status = routing_graph.route_graph_warmup_status()
    status = routing_graph.route_graph_warmup_status()
    rss = _process_rss_bytes(os.getpid())
    if rss is not None and rss > child_peak_rss_bytes:
        child_peak_rss_bytes = rss
    payload = {
        "asset_path": str(asset_path),
        "load_mode": str(status.get("load_strategy", "none")),
        "warmup_status": status,
        "graph_edge_count": int(status.get("edge_count") or status.get("edges_kept") or 0),
        "graph_node_count": int(status.get("nodes_kept") or 0),
        "graph_component_count": int(status.get("component_count") or 0),
        "child_peak_rss_bytes": child_peak_rss_bytes,
        "child_peak_rss_mb": round(child_peak_rss_bytes / (1024.0 * 1024.0), 2),
    }
    return payload


def _run_parent(
    *,
    asset_path: Path,
    timeout_s: int,
    poll_interval_s: float,
) -> dict[str, Any]:
    asset_path = _normalize_asset_path(asset_path)
    script_path = Path(__file__).resolve()
    command = _build_child_command(
        script_path=script_path,
        asset_path=asset_path,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
        env=env,
    )
    peak_rss_bytes = 0
    peak_rss_mb = 0.0
    timed_out = False
    try:
        deadline = time.perf_counter() + float(timeout_s) + 10.0
        while proc.poll() is None:
            if time.perf_counter() > deadline:
                timed_out = True
                try:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                break
            rss = _process_rss_bytes(proc.pid)
            if rss is not None and rss > peak_rss_bytes:
                peak_rss_bytes = rss
                peak_rss_mb = peak_rss_bytes / (1024.0 * 1024.0)
            time.sleep(max(0.05, float(poll_interval_s)))
    finally:
        stdout, stderr = proc.communicate()
    wall_time_ms = round((time.perf_counter() - start) * 1000.0, 2)
    child_payload: dict[str, Any] | None = None
    stdout_text = stdout.strip()
    if stdout_text:
        try:
            child_payload = json.loads(stdout_text)
        except json.JSONDecodeError:
            child_payload = {
                "raw_stdout": stdout_text,
            }
    time_to_ready_ms = None
    load_mode = "unknown"
    if isinstance(child_payload, dict):
        warmup_status = child_payload.get("warmup_status")
        if isinstance(warmup_status, dict):
            raw_elapsed_ms = warmup_status.get("elapsed_ms")
            if isinstance(raw_elapsed_ms, (int, float)):
                time_to_ready_ms = round(float(raw_elapsed_ms), 2)
        load_mode = str(child_payload.get("load_mode", "unknown"))
    return {
        "time_to_ready_ms": time_to_ready_ms,
        "load_mode": load_mode,
        "wall_time_ms": wall_time_ms,
        "parent_observed_peak_rss_bytes": peak_rss_bytes,
        "parent_observed_peak_rss_mb": round(peak_rss_mb, 2),
        "child_exit_code": int(proc.returncode or 0),
        "timed_out": timed_out,
        "stderr": stderr.strip(),
        "child": child_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fresh-process route-graph warmup.")
    parser.add_argument(
        "--asset-path",
        type=Path,
        default=_default_asset_path(),
        help="Route graph JSON asset to warm from a fresh process.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=3600,
        help="Maximum warmup runtime for the child process.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=0.25,
        help="Polling interval used both by the child warmup wait loop and the parent RSS monitor.",
    )
    parser.add_argument(
        "--child",
        action="store_true",
        help="Run the warmup inside a fresh child process and emit a JSON payload.",
    )
    args = parser.parse_args()
    asset_path = _normalize_asset_path(args.asset_path)
    if args.child:
        payload = _run_child(
            asset_path=asset_path,
            timeout_s=max(5, int(args.timeout_s)),
            poll_interval_s=max(0.05, float(args.poll_interval_s)),
        )
        print(json.dumps(payload, indent=2), flush=True)
        state = str(payload.get("warmup_status", {}).get("state", "")).strip().lower()
        sys.stdout.flush()
        sys.stderr.flush()
        if state == "ready":
            os._exit(0)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    payload = _run_parent(
        asset_path=asset_path,
        timeout_s=max(5, int(args.timeout_s)),
        poll_interval_s=max(0.05, float(args.poll_interval_s)),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
