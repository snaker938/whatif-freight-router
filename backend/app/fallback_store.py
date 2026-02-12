from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .settings import settings


_LOCK = Lock()


def _snapshot_store_path() -> Path:
    path = Path(settings.out_dir) / "offline" / "route_snapshots.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_snapshot_store() -> dict[str, dict[str, Any]]:
    path = _snapshot_store_path()
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        updated_at = value.get("updated_at")
        routes = value.get("routes")
        if isinstance(updated_at, str) and isinstance(routes, list):
            out[key] = {"updated_at": updated_at, "routes": routes}
    return out


def _write_snapshot_store(payload: dict[str, dict[str, Any]]) -> None:
    path = _snapshot_store_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_route_snapshot(cache_key: str, routes: list[dict[str, Any]]) -> None:
    if not cache_key:
        return

    with _LOCK:
        payload = _read_snapshot_store()
        payload[cache_key] = {
            "updated_at": datetime.now(UTC).isoformat(),
            "routes": copy.deepcopy(routes),
        }
        _write_snapshot_store(payload)


def load_route_snapshot(cache_key: str) -> tuple[list[dict[str, Any]], str] | None:
    if not cache_key:
        return None

    with _LOCK:
        payload = _read_snapshot_store()
        entry = payload.get(cache_key)
        if not isinstance(entry, dict):
            return None
        updated_at = entry.get("updated_at")
        routes = entry.get("routes")
        if not isinstance(updated_at, str) or not isinstance(routes, list):
            return None
        return copy.deepcopy(routes), updated_at


def clear_route_snapshots() -> int:
    with _LOCK:
        path = _snapshot_store_path()
        if not path.exists():
            return 0

        try:
            payload = _read_snapshot_store()
            count = len(payload)
        except Exception:
            count = 0

        path.write_text("{}", encoding="utf-8")
        return count
