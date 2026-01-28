from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .settings import settings


def write_manifest(run_id: str, manifest: dict[str, Any]) -> Path:
    out_dir = Path(settings.out_dir) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        **manifest,
    }

    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return path
