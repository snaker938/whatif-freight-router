from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.codex_coord_lib import (  # noqa: E402
    CoordinationError,
    CoordinationStore,
    DEFAULT_PYTHON_MEMORY_CAP_PERCENT,
    load_repo_context,
)


WINDOWS_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
CREATE_NO_WINDOW = 0x08000000


def detect_total_memory_bytes() -> int | None:
    if os.name != "nt":
        for candidate in (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ):
            try:
                raw = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not raw or raw == "max":
                continue
            try:
                limit = int(raw)
            except ValueError:
                continue
            if 0 < limit < 1 << 60:
                return limit
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            return int(state.ullTotalPhys)
        return None
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            return None
    return None


def compute_memory_cap(total_memory_bytes: int | None, requested_bytes: int | None, requested_percent: float | None) -> tuple[int | None, float | None]:
    percent = requested_percent if requested_percent is not None else DEFAULT_PYTHON_MEMORY_CAP_PERCENT
    percent_bytes = None if total_memory_bytes is None else max(1, int(total_memory_bytes * (percent / 100.0)))
    if requested_bytes is not None and percent_bytes is not None:
        cap_bytes = min(requested_bytes, percent_bytes)
    elif requested_bytes is not None:
        cap_bytes = requested_bytes
    else:
        cap_bytes = percent_bytes
    if total_memory_bytes is not None and cap_bytes is not None:
        effective_percent = round((cap_bytes / total_memory_bytes) * 100.0, 3)
    else:
        effective_percent = None
    return cap_bytes, effective_percent


def format_memory_context(
    total_memory_bytes: int | None,
    cap_bytes: int | None,
    cap_percent: float | None,
    requested_bytes: int | None,
    requested_percent: float | None,
) -> str:
    total_text = "unknown" if total_memory_bytes is None else str(total_memory_bytes)
    cap_text = "unknown" if cap_bytes is None else str(cap_bytes)
    percent_text = "unknown" if cap_percent is None else str(cap_percent)
    requested_bits = []
    if requested_bytes is not None:
        requested_bits.append(f"requested_bytes={requested_bytes}")
    if requested_percent is not None:
        requested_bits.append(f"requested_percent={requested_percent}")
    requested_suffix = f" {' '.join(requested_bits)}" if requested_bits else ""
    return f"total_memory_bytes={total_text} cap_bytes={cap_text} cap_percent={percent_text}{requested_suffix}"


def posix_preexec_memory_limit(cap_bytes: int) -> Callable[[], None]:
    import resource

    def configure() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, cap_bytes))

    return configure


class WindowsJobMemoryLimiter:
    def __init__(self, cap_bytes: int) -> None:
        self.cap_bytes = cap_bytes
        self.job_handle: int | None = None

    def apply(self, process: subprocess.Popen[Any]) -> str:
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        job_handle = kernel32.CreateJobObjectW(None, None)
        if not job_handle:
            return "record_only"
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = WINDOWS_JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info.ProcessMemoryLimit = self.cap_bytes
        if not kernel32.SetInformationJobObject(job_handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job_handle)
            return "record_only"
        if not kernel32.AssignProcessToJobObject(job_handle, int(process._handle)):
            kernel32.CloseHandle(job_handle)
            return "record_only"
        self.job_handle = job_handle
        return "windows_job_object_process_memory"

    def close(self) -> None:
        if self.job_handle:
            ctypes.windll.kernel32.CloseHandle(self.job_handle)
            self.job_handle = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Python command with coordination lease tracking")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--session-id", "--child-session-id", dest="session_id", required=True)
    parser.add_argument("--slot-id", type=int, default=None)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--lease-id", default=None)
    parser.add_argument("--memory-cap-bytes", type=int, default=None)
    parser.add_argument("--memory-cap-percent", type=float, default=None)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to launch, after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide the Python command after --")

    store = CoordinationStore(load_repo_context(args.repo_root))
    total_memory_bytes = detect_total_memory_bytes()
    cap_bytes, cap_percent = compute_memory_cap(total_memory_bytes, args.memory_cap_bytes, args.memory_cap_percent)

    child_env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        child_env[key] = "1"
    child_env["TOKENIZERS_PARALLELISM"] = "false"

    preexec_fn = None
    enforcement_method = "record_only"
    windows_job = None
    if cap_bytes is not None and os.name != "nt":
        try:
            preexec_fn = posix_preexec_memory_limit(cap_bytes)
            enforcement_method = "posix_rlimit_as"
        except Exception:
            preexec_fn = None
    elif cap_bytes is not None and os.name == "nt":
        windows_job = WindowsJobMemoryLimiter(cap_bytes)
        enforcement_method = "windows_job_object_process_memory_pending"

    start_note = f"cwd={args.cwd or store.context.repo_root} {format_memory_context(total_memory_bytes, cap_bytes, cap_percent, args.memory_cap_bytes, args.memory_cap_percent)}"
    lease = store.open_python_lease(
        session_id=args.session_id,
        owner_slot=args.slot_id,
        lease_id=args.lease_id,
        purpose=args.purpose,
        command=" ".join(command),
        memory_cap_bytes=cap_bytes,
        memory_cap_percent=cap_percent,
        enforcement_method=enforcement_method,
        status="launching",
        note=start_note,
    )
    lease_id = lease["lease_id"]

    process: subprocess.Popen[Any] | None = None
    signal_state: dict[str, int | str | None] = {"wrapper_signum": None, "wrapper_signal": None, "child_signal": None}
    original_handlers: dict[int, Any] = {}

    def _signal_name(signum: int) -> str:
        try:
            return signal.Signals(signum).name
        except ValueError:
            return str(signum)

    def _forward_signal(signum: int) -> None:
        if process is None or process.poll() is not None:
            return
        signal_name = _signal_name(signum)
        try:
            if signum == signal.SIGINT:
                process.send_signal(signal.SIGINT)
            elif signum == signal.SIGTERM:
                process.terminate()
            elif hasattr(signal, "SIGHUP") and signum == signal.SIGHUP:
                process.terminate()
            elif hasattr(signal, "SIGBREAK") and signum == signal.SIGBREAK:
                process.send_signal(signal.SIGBREAK)
            else:
                process.terminate()
            signal_state["child_signal"] = signal_name
        except Exception:
            try:
                process.terminate()
                signal_state["child_signal"] = "SIGTERM"
            except Exception:
                pass

    def _handle_signal(signum: int, frame: Any) -> None:  # noqa: ARG001
        signal_state["wrapper_signum"] = signum
        signal_state["wrapper_signal"] = _signal_name(signum)
        _forward_signal(signum)

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    if hasattr(signal, "SIGBREAK"):
        handled_signals.append(signal.SIGBREAK)

    try:
        for signum in handled_signals:
            original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        process = subprocess.Popen(
            command,
            cwd=args.cwd or str(store.context.repo_root),
            env=child_env,
            preexec_fn=preexec_fn,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception:
        store.close_python_lease(
            session_id=args.session_id,
            lease_id=lease_id,
            status="launch_failed",
            note=f"Process creation failed before execution. {format_memory_context(total_memory_bytes, cap_bytes, cap_percent, args.memory_cap_bytes, args.memory_cap_percent)}",
        )
        raise

    if windows_job is not None and cap_bytes is not None:
        actual_method = windows_job.apply(process)
        store.touch_python_lease(
            lease_id=lease_id,
            status="running",
            pid=process.pid,
            note=f"enforcement_method={actual_method}",
        )
    else:
        store.touch_python_lease(lease_id=lease_id, status="running", pid=process.pid)
    if signal_state["wrapper_signum"] is not None:
        _forward_signal(int(signal_state["wrapper_signum"]))

    exit_code: int | None = None
    lease_status = "terminated"
    final_note = f"exit_code=unknown wrapper_signal={signal_state['wrapper_signal'] or 'none'} child_signal={signal_state['child_signal'] or 'none'} {format_memory_context(total_memory_bytes, cap_bytes, cap_percent, args.memory_cap_bytes, args.memory_cap_percent)}"
    try:
        last_touch = time.monotonic()
        while True:
            try:
                exit_code = process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - last_touch >= 5:
                    store.touch_python_lease(lease_id=lease_id, status="running", pid=process.pid)
                    last_touch = time.monotonic()
        if signal_state["wrapper_signal"] == "SIGINT":
            lease_status = "interrupted"
        elif signal_state["wrapper_signal"] is not None:
            lease_status = "terminated"
        elif exit_code == 0:
            lease_status = "completed"
        else:
            lease_status = f"failed({exit_code})"
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        if windows_job is not None:
            windows_job.close()

        if exit_code is None:
            try:
                exit_code = process.wait(timeout=1)
            except Exception:
                exit_code = -1
                lease_status = "terminated"
                signal_state["child_signal"] = signal_state["child_signal"] or "wait_failed"
        final_note = (
            f"exit_code={exit_code} "
            f"wrapper_signal={signal_state['wrapper_signal'] or 'none'} "
            f"child_signal={signal_state['child_signal'] or 'none'} "
            f"{format_memory_context(total_memory_bytes, cap_bytes, cap_percent, args.memory_cap_bytes, args.memory_cap_percent)}"
        )
        store.close_python_lease(
            session_id=args.session_id,
            lease_id=lease_id,
            status=lease_status,
            note=final_note,
        )
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CoordinationError as exc:
        print(f"coordination error: {exc}", file=sys.stderr)
        raise SystemExit(1)
