from __future__ import annotations

import csv
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


ARTIFACT_FILES: tuple[str, ...] = (
    "results.json",
    "results.csv",
    "metadata.json",
    "routes.geojson",
    "results_summary.csv",
)

CSV_COLUMNS: tuple[str, ...] = (
    "pair_index",
    "origin_lat",
    "origin_lon",
    "destination_lat",
    "destination_lon",
    "error",
    "route_id",
    "distance_km",
    "duration_s",
    "monetary_cost",
    "emissions_kg",
    "avg_speed_kmh",
)


def artifact_dir_for_run(run_id: str) -> Path:
    p = Path(settings.out_dir) / "artifacts" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifact_paths_for_run(run_id: str) -> dict[str, Path]:
    base = Path(settings.out_dir) / "artifacts" / run_id
    return {name: base / name for name in ARTIFACT_FILES}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def write_run_artifacts(
    run_id: str,
    *,
    results_payload: dict[str, Any],
    metadata_payload: dict[str, Any],
    csv_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    out_dir = artifact_dir_for_run(run_id)

    results_path = out_dir / "results.json"
    metadata_path = out_dir / "metadata.json"
    csv_path = out_dir / "results.csv"

    _write_json(results_path, results_payload)
    _write_json(metadata_path, metadata_payload)
    _write_csv(csv_path, csv_rows)

    return {
        "results.json": results_path,
        "metadata.json": metadata_path,
        "results.csv": csv_path,
    }
