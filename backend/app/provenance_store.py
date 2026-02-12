from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .settings import settings


def provenance_path_for_run(run_id: str) -> Path:
    base = Path(settings.out_dir) / "provenance"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{run_id}.json"


def provenance_event(run_id: str, event: str, **context: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **context,
    }


def write_provenance(run_id: str, events: list[dict[str, Any]]) -> Path:
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "event_count": len(events),
        "events": events,
    }
    path = provenance_path_for_run(run_id)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
