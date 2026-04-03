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


MANIFEST_SCHEMA_SURFACE = "manifest"
SCENARIO_MANIFEST_SCHEMA_SURFACE = "scenario_manifest"
DECISION_PACKAGE_SCHEMA_VERSION = "0.1.0"

SCHEMA_VERSIONS: dict[str, str] = {
    MANIFEST_SCHEMA_SURFACE: "1.0.0",
    SCENARIO_MANIFEST_SCHEMA_SURFACE: "1.0.0",
    "decision_package": DECISION_PACKAGE_SCHEMA_VERSION,
    "preference_summary": DECISION_PACKAGE_SCHEMA_VERSION,
    "support_summary": DECISION_PACKAGE_SCHEMA_VERSION,
    "support_trace": DECISION_PACKAGE_SCHEMA_VERSION,
    "support_provenance": DECISION_PACKAGE_SCHEMA_VERSION,
    "certified_set": DECISION_PACKAGE_SCHEMA_VERSION,
    "certified_set_routes": DECISION_PACKAGE_SCHEMA_VERSION,
    "abstention_summary": DECISION_PACKAGE_SCHEMA_VERSION,
    "witness_summary": DECISION_PACKAGE_SCHEMA_VERSION,
    "witness_routes": DECISION_PACKAGE_SCHEMA_VERSION,
    "controller_summary": DECISION_PACKAGE_SCHEMA_VERSION,
    "controller_trace": DECISION_PACKAGE_SCHEMA_VERSION,
    "voi_action_trace": DECISION_PACKAGE_SCHEMA_VERSION,
    "voi_stop_certificate": DECISION_PACKAGE_SCHEMA_VERSION,
    "final_route_trace": DECISION_PACKAGE_SCHEMA_VERSION,
    "theorem_hook_map": DECISION_PACKAGE_SCHEMA_VERSION,
    "lane_manifest": DECISION_PACKAGE_SCHEMA_VERSION,
}

ARTIFACT_SCHEMA_VERSIONS: dict[str, str] = {
    "decision_package.json": SCHEMA_VERSIONS["decision_package"],
    "preference_summary.json": SCHEMA_VERSIONS["preference_summary"],
    "support_summary.json": SCHEMA_VERSIONS["support_summary"],
    "support_trace.jsonl": SCHEMA_VERSIONS["support_trace"],
    "support_provenance.json": SCHEMA_VERSIONS["support_provenance"],
    "certified_set.json": SCHEMA_VERSIONS["certified_set"],
    "certified_set_routes.jsonl": SCHEMA_VERSIONS["certified_set_routes"],
    "abstention_summary.json": SCHEMA_VERSIONS["abstention_summary"],
    "witness_summary.json": SCHEMA_VERSIONS["witness_summary"],
    "witness_routes.jsonl": SCHEMA_VERSIONS["witness_routes"],
    "controller_summary.json": SCHEMA_VERSIONS["controller_summary"],
    "controller_trace.jsonl": SCHEMA_VERSIONS["controller_trace"],
    "voi_action_trace.json": SCHEMA_VERSIONS["voi_action_trace"],
    "voi_stop_certificate.json": SCHEMA_VERSIONS["voi_stop_certificate"],
    "final_route_trace.json": SCHEMA_VERSIONS["final_route_trace"],
    "theorem_hook_map.json": SCHEMA_VERSIONS["theorem_hook_map"],
    "lane_manifest.json": SCHEMA_VERSIONS["lane_manifest"],
}


def schema_version_for_surface(surface: str, *, default: str | None = None) -> str | None:
    cleaned = str(surface or "").strip().lower()
    if not cleaned:
        return default
    return SCHEMA_VERSIONS.get(cleaned, default)


def schema_version_for_artifact(artifact_name: str, *, default: str | None = None) -> str | None:
    cleaned = str(artifact_name or "").strip()
    if not cleaned:
        return default
    return ARTIFACT_SCHEMA_VERSIONS.get(cleaned, default)


def versioned_json_payload(
    payload: dict[str, Any],
    *,
    surface: str | None = None,
    artifact_name: str | None = None,
    default_version: str | None = None,
) -> dict[str, Any]:
    enriched = dict(payload)
    if enriched.get("schema_version") is not None:
        return enriched
    resolved = default_version
    if artifact_name is not None:
        resolved = schema_version_for_artifact(artifact_name, default=resolved)
    if resolved is None and surface is not None:
        resolved = schema_version_for_surface(surface, default=resolved)
    if resolved is not None:
        enriched.setdefault("schema_version", resolved)
    return enriched


def _write_signed_manifest(
    run_id: str,
    manifest: dict[str, Any],
    *,
    out_dir: Path,
    surface: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched = versioned_json_payload(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            **manifest,
        },
        surface=surface,
    )
    enriched["signature"] = build_signature_metadata(enriched)

    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return path


def write_manifest(run_id: str, manifest: dict[str, Any]) -> Path:
    out_dir = Path(settings.out_dir) / "manifests"
    return _write_signed_manifest(
        run_id,
        manifest,
        out_dir=out_dir,
        surface=MANIFEST_SCHEMA_SURFACE,
    )


def write_scenario_manifest(run_id: str, manifest: dict[str, Any]) -> Path:
    out_dir = Path(settings.out_dir) / "scenario_manifests"
    return _write_signed_manifest(
        run_id,
        manifest,
        out_dir=out_dir,
        surface=SCENARIO_MANIFEST_SCHEMA_SURFACE,
    )


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
    "decision_package.json",
    "preference_summary.json",
    "support_summary.json",
    "support_trace.jsonl",
    "support_provenance.json",
    "certified_set.json",
    "certified_set_routes.jsonl",
    "abstention_summary.json",
    "witness_summary.json",
    "witness_routes.jsonl",
    "controller_summary.json",
    "controller_trace.jsonl",
    "theorem_hook_map.json",
    "lane_manifest.json",
    "route_fragility_map.json",
    "competitor_fragility_breakdown.json",
    "value_of_refresh.json",
    "sampled_world_manifest.json",
    "evidence_snapshot_manifest.json",
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
    normalized_payload = payload
    if isinstance(payload, dict):
        normalized_payload = versioned_json_payload(payload, artifact_name=artifact_name)
    normalized = jsonable_encoder(normalized_payload)
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
