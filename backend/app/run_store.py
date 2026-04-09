from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.encoders import jsonable_encoder

from .settings import settings
from .signatures import build_signature_metadata


def _write_signed_manifest(run_id: str, manifest: dict[str, Any], *, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        **manifest,
    }
    enriched["signature"] = build_signature_metadata(enriched)

    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return path


def write_manifest(run_id: str, manifest: dict[str, Any]) -> Path:
    out_dir = Path(settings.out_dir) / "manifests"
    return _write_signed_manifest(run_id, manifest, out_dir=out_dir)


def write_scenario_manifest(run_id: str, manifest: dict[str, Any]) -> Path:
    out_dir = Path(settings.out_dir) / "scenario_manifests"
    return _write_signed_manifest(run_id, manifest, out_dir=out_dir)


ARTIFACT_FILES: tuple[str, ...] = (
    "results.json",
    "results.csv",
    "metadata.json",
    "routes.geojson",
    "results_summary.csv",
    "dccs_candidates.jsonl",
    "dccs_summary.json",
    "refined_routes.jsonl",
    "strict_frontier.jsonl",
    "winner_summary.json",
    "certificate_summary.json",
    "route_fragility_map.json",
    "competitor_fragility_breakdown.json",
    "value_of_refresh.json",
    "sampled_world_manifest.json",
    "evidence_snapshot_manifest.json",
    "preference_state.json",
    "preference_query_trace.json",
    "world_support_summary.json",
    "decision_package.json",
    "winner_confidence_state.json",
    "pairwise_gap_state.json",
    "flip_radius_summary.json",
    "decision_region_summary.json",
    "certificate_witness.json",
    "certified_set_summary.json",
    "voi_action_trace.json",
    "voi_controller_state.jsonl",
    "voi_action_scores.csv",
    "voi_stop_certificate.json",
    "final_route_trace.json",
    "od_corpus.csv",
    "od_corpus.json",
    "od_corpus_summary.json",
    "od_corpus_rejected.json",
    "ors_snapshot.json",
    "thesis_results.csv",
    "thesis_results.json",
    "thesis_summary.csv",
    "thesis_summary.json",
    "thesis_metrics.json",
    "thesis_plots.json",
    "methods_appendix.md",
    "thesis_report.md",
    "evaluation_manifest.json",
)

_SAFE_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

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


def is_safe_artifact_name(name: str) -> bool:
    return bool(_SAFE_ARTIFACT_NAME.fullmatch(str(name or "").strip()))


def artifact_path_for_name(run_id: str, artifact_name: str) -> Path:
    cleaned = str(artifact_name or "").strip()
    if not is_safe_artifact_name(cleaned):
        raise ValueError("invalid artifact name")
    return artifact_dir_for_run(run_id) / cleaned


def artifact_paths_for_run(run_id: str) -> dict[str, Path]:
    base = artifact_dir_for_run(run_id)
    return {name: base / name for name in ARTIFACT_FILES}


def list_artifact_paths_for_run(run_id: str) -> dict[str, Path]:
    base = artifact_dir_for_run(run_id)
    found: dict[str, Path] = {}
    for path in sorted(base.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        if not is_safe_artifact_name(path.name):
            continue
        found[path.name] = path
    for name, path in artifact_paths_for_run(run_id).items():
        if path.exists():
            found.setdefault(name, path)
    return found


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    normalized = jsonable_encoder(payload)
    path.write_text(json.dumps(normalized, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        json.dumps(jsonable_encoder(row), separators=(",", ":"), ensure_ascii=False, default=str)
        for row in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def write_json_artifact(run_id: str, artifact_name: str, payload: dict[str, Any] | list[Any]) -> Path:
    path = artifact_path_for_name(run_id, artifact_name)
    normalized = jsonable_encoder(payload)
    path.write_text(json.dumps(normalized, indent=2, default=str), encoding="utf-8")
    return path


def write_jsonl_artifact(run_id: str, artifact_name: str, rows: list[dict[str, Any]]) -> Path:
    path = artifact_path_for_name(run_id, artifact_name)
    _write_jsonl(path, rows)
    return path


def write_csv_artifact(
    run_id: str,
    artifact_name: str,
    *,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> Path:
    path = artifact_path_for_name(run_id, artifact_name)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def write_text_artifact(run_id: str, artifact_name: str, text: str) -> Path:
    path = artifact_path_for_name(run_id, artifact_name)
    path.write_text(str(text), encoding="utf-8")
    return path


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
