from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from fastapi.encoders import jsonable_encoder

from .settings import settings
from .signatures import build_signature_metadata
from .world_policies import policy_hash as build_policy_hash


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
    "thesis_summary_by_transfer_slice.csv",
    "thesis_summary_by_transfer_slice.json",
    "thesis_summary_by_weather_regime_transfer_slice.csv",
    "thesis_summary_by_weather_regime_transfer_slice.json",
    "thesis_metrics.json",
    "thesis_plots.json",
    "methods_appendix.md",
    "thesis_report.md",
    "evaluation_manifest.json",
    "index.json",
    "index.md",
)

_SAFE_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_BUNDLE_INDEX_ARTIFACTS = frozenset({"index.json", "index.md"})
_RUN_BUNDLE_INDEX_SCHEMA_VERSION = "run-bundle-index-v1"
ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION = "route-artifact-provenance-v1"
ROUTE_ARTIFACT_IDENTITY_SCHEMA_VERSION = "route-artifact-identity-v1"
ROUTE_THEOREM_HOOK_SCHEMA_VERSION = "route-theorem-hook-v2"
ROUTE_RUNTIME_LANE_ID = "route_compute_runtime"
ROUTE_CONTROLLER_POLICY_VERSION = "voi-controller-policy-v1"
ROUTE_PREFERENCE_MODEL_VERSION = "preference-elicitation-policy-v1"
_CONTAINER_DIGEST_ENV_VARS: tuple[str, ...] = (
    "CONTAINER_DIGEST",
    "IMAGE_DIGEST",
    "OCI_IMAGE_DIGEST",
    "DOCKER_IMAGE_DIGEST",
)
_THESIS_EXPORT_STATUS_ARTIFACTS: tuple[str, ...] = (
    "thesis_results.csv",
    "thesis_summary.csv",
    "thesis_summary_by_cohort.csv",
    "thesis_summary_by_transfer_slice.csv",
    "thesis_summary_by_weather_regime_transfer_slice.csv",
    "methods_appendix.md",
    "thesis_report.md",
    "evaluation_manifest.json",
    "thesis_plots.json",
    "index.json",
    "index.md",
    "results.csv",
)

_THESIS_HEADLINE_JSON_ARTIFACTS: tuple[str, ...] = (
    "thesis_results.json",
    "thesis_summary.json",
    "thesis_summary_by_cohort.json",
    "thesis_metrics.json",
    "thesis_plots.json",
    "evaluation_manifest.json",
)

_ARTIFACT_SUMMARY_HIGHLIGHT_KEYS: tuple[str, ...] = (
    "run_id",
    "selected_route_id",
    "terminal_type",
    "selected_certificate_basis",
    "support_flag",
    "support_reason",
    "reason_code",
    "message",
    "stop_reason",
    "query_count",
    "world_count",
    "candidate_count",
    "warning_count",
    "duration_ms",
    "set_size",
    "certified",
    "threshold",
    "frontier_count",
)

def _route_artifact_hook(
    hook_id: str,
    *,
    kind: str,
    status: str,
    family_id: str | None = None,
    artifact_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    hook: dict[str, Any] = {
        "hook_id": hook_id,
        "kind": kind,
        "status": status,
    }
    if family_id:
        hook["family_id"] = family_id
    if artifact_fields:
        hook["artifact_fields"] = list(artifact_fields)
    return hook


_ROUTE_ARTIFACT_THEOREM_HOOKS: dict[str, list[dict[str, Any]]] = {
    "decision_package.json": [
        {
            "hook_id": "decision_package_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        },
        {
            "hook_id": "terminal_outcome_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        },
        _route_artifact_hook(
            "THM-09",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="THM-09",
            artifact_fields=(
                "support_summary.multi_fidelity_summary.proxy_world_count",
                "support_summary.audit_world_count",
                "support_summary.multi_fidelity_certificate_basis",
                "support_summary.audit_correction_mass",
                "support_summary.proxy_only_fraction",
                "support_summary.positivity_diagnostics.weak_overlap_detected",
            ),
        ),
        _route_artifact_hook(
            "LB-04",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="LB-04",
            artifact_fields=(
                "support_summary.multi_fidelity_summary.proxy_world_count",
                "support_summary.audit_world_count",
                "support_summary.multi_fidelity_certificate_basis",
                "support_summary.audit_correction_mass",
                "support_summary.proxy_only_fraction",
                "support_summary.positivity_diagnostics.weak_overlap_detected",
            ),
        ),
    ],
    "dccs_candidates.jsonl": [
        _route_artifact_hook(
            "THM-01",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="THM-01",
            artifact_fields=(
                "safe_eliminated",
                "necessary_dominated",
                "dominated_by_route_id",
                "dominance_margin",
            ),
        ),
        _route_artifact_hook(
            "THM-08",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="THM-08",
        ),
    ],
    "dccs_summary.json": [
        _route_artifact_hook(
            "THM-01",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="THM-01",
            artifact_fields=("false_safe_prune_rate",),
        ),
        _route_artifact_hook(
            "THM-08",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="THM-08",
            artifact_fields=("unresolved_possible_winner_mass", "search_completeness_gap"),
        ),
        _route_artifact_hook(
            "LB-03",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="LB-03",
            artifact_fields=("unresolved_possible_winner_mass", "search_completeness_gap"),
        ),
    ],
    "certificate_summary.json": [
        {
            "hook_id": "certificate_summary_scaffold",
            "kind": "theorem_hook",
            "status": "scaffold_only",
        }
    ],
    "initial_certificate_summary.json": [
        _route_artifact_hook(
            "THM-02",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="THM-02",
            artifact_fields=("selected_certificate",),
        ),
        _route_artifact_hook(
            "LB-01",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="LB-01",
            artifact_fields=("selected_certificate",),
        ),
    ],
    "world_support_summary.json": [
        {
            "hook_id": "support_summary_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        }
    ],
    "winner_confidence_state.json": [
        {
            "hook_id": "winner_confidence_scaffold",
            "kind": "theorem_hook",
            "status": "scaffold_only",
        }
    ],
    "pairwise_gap_state.json": [
        {
            "hook_id": "pairwise_gap_scaffold",
            "kind": "theorem_hook",
            "status": "scaffold_only",
        }
    ],
    "flip_radius_summary.json": [
        {
            "hook_id": "flip_radius_scaffold",
            "kind": "theorem_hook",
            "status": "scaffold_only",
        }
    ],
    "decision_region_summary.json": [
        {
            "hook_id": "decision_region_scaffold",
            "kind": "theorem_hook",
            "status": "scaffold_only",
        }
    ],
    "certificate_witness.json": [
        {
            "hook_id": "certificate_witness_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        }
    ],
    "certified_set_summary.json": [
        {
            "hook_id": "certified_set_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        }
    ],
    "preference_state.json": [
        {
            "hook_id": "preference_state_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        }
    ],
    "preference_query_trace.json": [
        {
            "hook_id": "preference_query_trace_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        }
    ],
    "value_of_refresh.json": [
        {
            "hook_id": "voi_refresh_value_scaffold",
            "kind": "theorem_hook",
            "status": "heuristic_measured",
        }
    ],
    "sampled_world_manifest.json": [
        _route_artifact_hook(
            "LB-01",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="LB-01",
            artifact_fields=("world_count",),
        )
    ],
    "replay_oracle_summary.json": [
        _route_artifact_hook(
            "THM-10",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="THM-10",
            artifact_fields=("mean_replay_regret",),
        )
    ],
    "voi_action_trace.json": [
        _route_artifact_hook(
            "THM-10",
            kind="theorem_hook",
            status="runtime_backed",
            family_id="THM-10",
            artifact_fields=("actions",),
        )
    ],
    "voi_stop_certificate.json": [
        {
            "hook_id": "voi_stop_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        }
    ],
    "final_route_trace.json": [
        {
            "hook_id": "pipeline_stage_trace_contract",
            "kind": "runtime_contract",
            "status": "runtime_backed",
        }
    ],
}

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


def _load_json_artifact_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_endpoint(run_id: str, artifact_name: str) -> str:
    return f"/runs/{run_id}/artifacts/{artifact_name}"


def _summary_artifact_name(artifact_name: str) -> str | None:
    cleaned = str(artifact_name or "").strip()
    if cleaned in _BUNDLE_INDEX_ARTIFACTS:
        return None
    if cleaned.endswith(".json"):
        return f"{Path(cleaned).stem}.summary.md"
    if cleaned.endswith(".geojson"):
        return f"{Path(cleaned).stem}.summary.md"
    return None


def _bundle_artifact_names(run_id: str) -> list[str]:
    return [
        name
        for name in sorted(list_artifact_paths_for_run(run_id))
        if name not in _BUNDLE_INDEX_ARTIFACTS
    ]


def _bundle_title(bundle_type: Any) -> str:
    normalized = str(bundle_type or "").strip().lower()
    if normalized == "route_compute":
        return "Route-Compute Bundle Index"
    if normalized == "batch_pareto":
        return "Batch Pareto Bundle Index"
    if normalized == "thesis_evaluation":
        return "Evaluation Bundle Index"
    if normalized:
        return f"{normalized.replace('_', ' ').title()} Bundle Index"
    return "Run Bundle Index"


def route_artifact_theorem_hooks(artifact_name: str) -> list[dict[str, Any]]:
    hooks = _ROUTE_ARTIFACT_THEOREM_HOOKS.get(str(artifact_name or "").strip(), [])
    return [dict(hook) for hook in hooks]


def _json_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_nonempty_text(*values: Any) -> str | None:
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _markdown_scalar(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"`{value}`"
    text = str(value)
    return f"`{text}`" if text else "`\"\"`"


def _value_shape_summary(value: Any) -> str:
    if isinstance(value, Mapping):
        keys = list(value.keys())
        preview = ", ".join(f"`{str(key)}`" for key in keys[:4])
        suffix = "" if len(keys) <= 4 else ", ..."
        return f"object with {len(keys)} field(s)" + (f" ({preview}{suffix})" if preview else "")
    if isinstance(value, list):
        return f"list with {len(value)} item(s)"
    if value is None:
        return "null"
    return f"scalar {_markdown_scalar(value)}"


def _append_highlight(lines: list[str], label: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (dict, list)):
        return
    lines.append(f"- {label}: {_markdown_scalar(value)}")


def _collect_reviewer_highlights(payload: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in _ARTIFACT_SUMMARY_HIGHLIGHT_KEYS:
        if key in payload:
            _append_highlight(lines, key.replace("_", " ").title(), payload.get(key))

    frontier_summary = _json_mapping(payload.get("frontier_summary"))
    if frontier_summary:
        _append_highlight(lines, "Frontier Count", frontier_summary.get("frontier_count"))

    certified_set_summary = _json_mapping(payload.get("certified_set_summary"))
    if certified_set_summary:
        _append_highlight(lines, "Certified Set Size", certified_set_summary.get("set_size"))
        _append_highlight(lines, "Certified Set Certified", certified_set_summary.get("certified"))

    abstention_summary = _json_mapping(payload.get("abstention_summary"))
    if abstention_summary:
        _append_highlight(lines, "Abstention Reason", abstention_summary.get("reason_code"))

    action_trace_summary = _json_mapping(payload.get("action_trace_summary"))
    if action_trace_summary:
        _append_highlight(lines, "Action Trace Stop Reason", action_trace_summary.get("stop_reason"))

    provenance = _json_mapping(payload.get("artifact_provenance"))
    if provenance:
        lines.extend(["", "## Artifact Provenance", ""])
        _append_highlight(lines, "Run Id", provenance.get("run_id"))
        _append_highlight(lines, "Lane Id", provenance.get("lane_id"))
        _append_highlight(lines, "Variant Id", provenance.get("variant_id"))
        _append_highlight(lines, "Cache Mode", provenance.get("cache_mode"))
        _append_highlight(lines, "Support Status", provenance.get("support_status"))
    return lines


def _render_json_artifact_summary_markdown(
    artifact_name: str,
    payload: dict[str, Any] | list[Any],
) -> str:
    title = Path(artifact_name).stem.replace("_", " ").replace("-", " ").title()
    lines = [
        f"# {title} Summary",
        "",
        f"- Companion to: `{artifact_name}`",
    ]
    if isinstance(payload, Mapping):
        schema_version = _first_nonempty_text(payload.get("schema_version"))
        if schema_version:
            lines.append(f"- Payload Schema Version: `{schema_version}`")
        lines.append(f"- JSON Shape: object with {len(payload)} top-level field(s)")
        highlight_lines = _collect_reviewer_highlights(payload)
        if highlight_lines:
            lines.extend(["", "## Reviewer Highlights", ""])
            lines.extend(highlight_lines)
        lines.extend(["", "## Top-Level Fields", ""])
        for key in sorted(payload):
            lines.append(f"- `{key}`: {_value_shape_summary(payload.get(key))}")
        return "\n".join(lines) + "\n"

    lines.append(f"- JSON Shape: list with {len(payload)} item(s)")
    if payload:
        first_item = payload[0]
        lines.extend(["", "## First Item Shape", ""])
        lines.append(f"- {_value_shape_summary(first_item)}")
    return "\n".join(lines) + "\n"


def _write_json_summary_companion(
    run_id: str,
    artifact_name: str,
    payload: dict[str, Any] | list[Any],
) -> Path | None:
    summary_name = _summary_artifact_name(artifact_name)
    if summary_name is None:
        return None
    summary_path = artifact_path_for_name(run_id, summary_name)
    summary_path.write_text(
        _render_json_artifact_summary_markdown(artifact_name, payload),
        encoding="utf-8",
    )
    return summary_path


def _thesis_policy_hashes(
    metadata: Mapping[str, Any],
    evaluation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation_suite = _json_mapping(metadata.get("evaluation_suite")) or _json_mapping(
        evaluation_manifest.get("evaluation_suite")
    )
    lane_role = _first_nonempty_text(evaluation_suite.get("role"), "thesis_evaluation") or "thesis_evaluation"
    strict_evidence_policy = _first_nonempty_text(
        metadata.get("strict_evidence_policy"),
        evaluation_manifest.get("strict_evidence_policy"),
    )
    ors_baseline_policy = _first_nonempty_text(
        metadata.get("ors_baseline_policy"),
        evaluation_manifest.get("ors_baseline_policy"),
    )
    cache_mode = _first_nonempty_text(
        metadata.get("cache_mode"),
        evaluation_manifest.get("cache_mode"),
        "mixed",
    ) or "mixed"
    cache_reset_scope = _first_nonempty_text(
        metadata.get("cache_reset_scope"),
        evaluation_manifest.get("cache_reset_scope"),
        "none",
    ) or "none"
    cache_reset_policy = _first_nonempty_text(
        metadata.get("cache_reset_policy"),
        evaluation_manifest.get("cache_reset_policy"),
        "none",
    ) or "none"
    model_version = _first_nonempty_text(
        evaluation_manifest.get("model_version"),
        "thesis-script-untracked",
    ) or "thesis-script-untracked"
    snapshot_mode = _first_nonempty_text(
        evaluation_manifest.get("ors_snapshot_mode"),
        "off",
    ) or "off"

    return {
        "strict_evidence_policy_hash": _stable_policy_hash(
            "strict_evidence_policy",
            version=strict_evidence_policy,
            configuration={"lane_id": lane_role},
        ),
        "baseline_engine_policy_hash": _stable_policy_hash(
            "baseline_engine_policy",
            version=ors_baseline_policy,
            configuration={
                "lane_id": lane_role,
                "ors_snapshot_mode": snapshot_mode,
            },
        ),
        "evaluation_lane_policy_hash": _stable_policy_hash(
            "evaluation_lane_policy",
            version=model_version,
            configuration={
                "role": lane_role,
                "scope": evaluation_suite.get("scope"),
                "focus": evaluation_suite.get("focus"),
                "strict_proxy_ors_allowed": metadata.get("strict_proxy_ors_allowed"),
                "strict_evidence_fallbacks_allowed": metadata.get("strict_evidence_fallbacks_allowed"),
            },
        ),
        "cache_policy_hash": build_policy_hash(
            "evaluation_cache_policy",
            version="thesis-eval-cache-policy-v1",
            configuration={
                "cache_mode": cache_mode,
                "cache_reset_scope": cache_reset_scope,
                "cache_reset_policy": cache_reset_policy,
                "cache_carryover_expected": metadata.get("cache_carryover_expected"),
            },
        ),
    }


def build_thesis_artifact_provenance_context(
    run_id: str,
    *,
    metadata: Mapping[str, Any],
    evaluation_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_payload = _json_mapping(metadata)
    evaluation_manifest_payload = _json_mapping(evaluation_manifest)
    evaluation_suite = _json_mapping(metadata_payload.get("evaluation_suite")) or _json_mapping(
        evaluation_manifest_payload.get("evaluation_suite")
    )
    lane_id = _first_nonempty_text(evaluation_suite.get("role"), "thesis_evaluation") or "thesis_evaluation"
    cache_mode = _first_nonempty_text(
        metadata_payload.get("cache_mode"),
        evaluation_manifest_payload.get("cache_mode"),
        "mixed",
    ) or "mixed"
    cache_reset_policy = _first_nonempty_text(
        metadata_payload.get("cache_reset_policy"),
        evaluation_manifest_payload.get("cache_reset_policy"),
        "none",
    ) or "none"
    git_commit_hash = _resolve_git_commit_hash()
    environment_lockfile_hash, environment_lockfile_path = _resolve_environment_lockfile_hash()
    container_digest, container_digest_source = _resolve_container_digest()

    return {
        "run_id": run_id,
        "lane_id": lane_id,
        "variant_id": "aggregate",
        "cache_mode": cache_mode,
        "seed": _safe_int(metadata_payload.get("run_seed")),
        "calibration_policy_version": "untracked",
        "controller_policy_version": _first_nonempty_text(
            evaluation_manifest_payload.get("model_version"),
            "thesis-script-untracked",
        )
        or "thesis-script-untracked",
        "preference_model_version": ROUTE_PREFERENCE_MODEL_VERSION,
        "proxy_correction_version": "inactive",
        "support_status": "aggregate",
        "cache_reuse_origin": "bundle_aggregate",
        "cache_source_id": None,
        "reuse_count": _safe_int(metadata_payload.get("cache_reset_count")) or 0,
        "invalidation_reason": cache_reset_policy,
        "headline_identity": {
            "schema_version": ROUTE_ARTIFACT_IDENTITY_SCHEMA_VERSION,
            "git_commit_hash": git_commit_hash,
            "environment_lockfile_hash": environment_lockfile_hash,
            "environment_lockfile_path": environment_lockfile_path,
            "container_digest": container_digest,
            "container_digest_source": container_digest_source,
            "policy_hashes": _thesis_policy_hashes(metadata_payload, evaluation_manifest_payload),
        },
    }


def _thesis_artifact_provenance_context_for_run(run_id: str) -> dict[str, Any] | None:
    metadata = _load_json_artifact_dict(artifact_path_for_name(run_id, "metadata.json"))
    if not metadata:
        return None
    if _infer_bundle_type(run_id, metadata) != "thesis_evaluation":
        return None
    evaluation_manifest = _load_json_artifact_dict(artifact_path_for_name(run_id, "evaluation_manifest.json"))
    return build_thesis_artifact_provenance_context(
        run_id,
        metadata=metadata,
        evaluation_manifest=evaluation_manifest,
    )


def _refresh_thesis_json_artifact_companions_and_provenance(run_id: str) -> None:
    provenance_context = _thesis_artifact_provenance_context_for_run(run_id)
    if provenance_context is None:
        return

    for artifact_name in _THESIS_HEADLINE_JSON_ARTIFACTS:
        path = artifact_path_for_name(run_id, artifact_name)
        if not path.exists() or not path.is_file():
            continue
        payload = _load_json_artifact_dict(path)
        if not payload:
            continue
        payload["artifact_provenance"] = build_route_artifact_provenance(
            artifact_name,
            context=provenance_context,
            payload_schema_version=payload_schema_version_for_artifact(artifact_name, payload),
        )
        _write_json(path, payload)
        _write_json_summary_companion(run_id, artifact_name, payload)


@lru_cache(maxsize=1)
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _backend_lockfile_path() -> Path | None:
    for candidate in ("uv.lock", "poetry.lock", "requirements.lock"):
        path = _repo_root() / "backend" / candidate
        if path.exists() and path.is_file():
            return path
    return None


def _hex_file_hash(path: Path, *, algorithm: str = "sha256") -> str | None:
    try:
        digest = hashlib.new(algorithm)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
    except (OSError, ValueError):
        return None
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _resolve_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit_hash = str(result.stdout or "").strip().lower()
    if len(commit_hash) == 40 and all(ch in "0123456789abcdef" for ch in commit_hash):
        return commit_hash
    return None


@lru_cache(maxsize=1)
def _resolve_environment_lockfile_hash() -> tuple[str | None, str | None]:
    lockfile_path = _backend_lockfile_path()
    if lockfile_path is None:
        return None, None
    return (
        _hex_file_hash(lockfile_path, algorithm="sha256"),
        lockfile_path.relative_to(_repo_root()).as_posix(),
    )


@lru_cache(maxsize=1)
def _resolve_container_digest() -> tuple[str | None, str]:
    for env_name in _CONTAINER_DIGEST_ENV_VARS:
        value = str(os.getenv(env_name, "")).strip()
        if value:
            return value, f"env:{env_name}"
    return None, "unavailable_local_runtime"


def _stable_policy_hash(
    policy_name: str,
    *,
    version: str | None,
    configuration: Mapping[str, Any] | None = None,
) -> str | None:
    normalized_version = _first_nonempty_text(version)
    if normalized_version is None or normalized_version.lower() in {"unknown", "untracked", "inactive"}:
        return None
    return build_policy_hash(policy_name, version=normalized_version, configuration=configuration)


def _route_policy_hashes(
    *,
    pipeline_mode: Any,
    decision_package: Mapping[str, Any],
    final_route_trace: Mapping[str, Any],
    world_bundle_summary: Mapping[str, Any],
    probabilistic_bundle: Mapping[str, Any],
    audit_bundle: Mapping[str, Any],
    multi_fidelity_summary: Mapping[str, Any],
    calibration_policy_version: str,
    controller_policy_version: str,
    preference_model_version: str,
    proxy_correction_version: str,
    lane_id: str,
) -> dict[str, Any]:
    refinement_policy = _first_nonempty_text(final_route_trace.get("refinement_policy"))
    probabilistic_world_policy_hash = _first_nonempty_text(probabilistic_bundle.get("policy_hash"))
    audit_world_policy_hash = _first_nonempty_text(audit_bundle.get("policy_hash"))
    return {
        "calibration_policy_hash": _stable_policy_hash(
            "calibration_policy",
            version=calibration_policy_version,
            configuration={
                "pipeline_mode": pipeline_mode,
                "regime_id": world_bundle_summary.get("regime_id"),
                "copula_id": world_bundle_summary.get("copula_id"),
                "as_of_utc": world_bundle_summary.get("as_of_utc"),
                "probabilistic_world_policy_hash": probabilistic_world_policy_hash,
                "audit_world_policy_hash": audit_world_policy_hash,
            },
        ),
        "controller_policy_hash": _stable_policy_hash(
            "voi_controller_policy",
            version=controller_policy_version,
            configuration={
                "lane_id": lane_id,
                "pipeline_mode": pipeline_mode,
                "refinement_policy": refinement_policy,
            },
        ),
        "preference_model_hash": _stable_policy_hash(
            "preference_model",
            version=preference_model_version,
            configuration={
                "lane_id": lane_id,
                "pipeline_mode": pipeline_mode,
                "terminal_type": decision_package.get("terminal_type"),
            },
        ),
        "proxy_correction_hash": _stable_policy_hash(
            "proxy_correction",
            version=proxy_correction_version,
            configuration={
                "active": bool(multi_fidelity_summary.get("proxy_correction_active")),
                "pipeline_mode": pipeline_mode,
                "probabilistic_world_policy_hash": probabilistic_world_policy_hash,
                "audit_world_policy_hash": audit_world_policy_hash,
            },
        ),
        "probabilistic_world_policy_hash": probabilistic_world_policy_hash,
        "audit_world_policy_hash": audit_world_policy_hash,
    }


def _route_cache_payloads(final_route_trace: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    trace_payload = _json_mapping(final_route_trace)
    return [
        payload
        for payload in (
            _json_mapping(trace_payload.get("route_cache_runtime")),
            _json_mapping(trace_payload.get("option_build_runtime")),
            _json_mapping(trace_payload.get("route_option_cache_runtime")),
            _json_mapping(trace_payload.get("voi_dccs_runtime")),
        )
        if payload
    ]


def _route_support_status(world_support_summary: Mapping[str, Any] | None) -> str:
    support_payload = _json_mapping(world_support_summary)
    support_state = _json_mapping(support_payload.get("support_state"))
    support_bin = _first_nonempty_text(support_state.get("support_bin"))
    if support_bin:
        return support_bin.lower()
    support_flag = support_payload.get("support_flag")
    if support_flag is True:
        return "supported"
    if support_flag is False:
        return "unsupported"
    return "unknown"


def build_route_artifact_provenance_context(
    *,
    run_id: str,
    pipeline_mode: Any,
    run_seed: Any,
    decision_package: Mapping[str, Any] | None = None,
    world_support_summary: Mapping[str, Any] | None = None,
    final_route_trace: Mapping[str, Any] | None = None,
    sampled_world_manifest: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    override_payload = _json_mapping(overrides)
    decision_payload = _json_mapping(decision_package)
    support_payload = _json_mapping(world_support_summary)
    decision_support_summary = _json_mapping(decision_payload.get("support_summary"))
    trace_payload = _json_mapping(final_route_trace)
    manifest_payload = _json_mapping(sampled_world_manifest)
    world_bundle_summary = _json_mapping(
        support_payload.get("world_bundle_summary")
        or {"multi_fidelity_summary": decision_support_summary.get("multi_fidelity_summary")}
    )
    probabilistic_bundle = _json_mapping(world_bundle_summary.get("probabilistic_world_bundle"))
    audit_bundle = _json_mapping(world_bundle_summary.get("audit_world_bundle"))
    multi_fidelity_summary = _json_mapping(world_bundle_summary.get("multi_fidelity_summary"))
    cache_payloads = _route_cache_payloads(trace_payload)

    cache_hits = sum(max(0, _safe_int(payload.get("cache_hits")) or 0) for payload in cache_payloads)
    cache_misses = sum(max(0, _safe_int(payload.get("cache_misses")) or 0) for payload in cache_payloads)
    cache_source_id = _first_nonempty_text(
        override_payload.get("cache_source_id"),
        manifest_payload.get("cache_source_id"),
        *[payload.get("last_cache_key") for payload in cache_payloads],
    )
    cache_reuse_origin = (
        _first_nonempty_text(
            override_payload.get("cache_reuse_origin"),
            manifest_payload.get("certification_cache_reuse_origin"),
        )
        or ("reused" if cache_hits > 0 else "miss")
    ).lower()
    reuse_count = _safe_int(override_payload.get("reuse_count"))
    if reuse_count is None:
        reuse_count = cache_hits
    if reuse_count <= 0 and cache_reuse_origin in {"local", "global", "reused"}:
        reuse_count = 1

    derived_cache_mode = _first_nonempty_text(
        override_payload.get("cache_mode"),
        audit_bundle.get("cache_mode"),
        probabilistic_bundle.get("cache_mode"),
    )
    if not derived_cache_mode:
        if cache_hits > 0 and cache_misses > 0:
            derived_cache_mode = "mixed"
        elif cache_hits > 0:
            derived_cache_mode = "reused"
        else:
            derived_cache_mode = "cold"

    invalidation_reason = _first_nonempty_text(override_payload.get("invalidation_reason"))
    if not invalidation_reason:
        if reuse_count > 0:
            invalidation_reason = "not_invalidated"
        elif cache_misses > 0:
            invalidation_reason = "cache_miss"
        else:
            invalidation_reason = "not_recorded"

    proxy_correction_version = _first_nonempty_text(
        override_payload.get("proxy_correction_version"),
        manifest_payload.get("proxy_correction_version"),
        multi_fidelity_summary.get("proxy_bias_model_version"),
        manifest_payload.get("proxy_bias_model_version"),
    )
    if not proxy_correction_version:
        proxy_correction_version = (
            "inactive"
            if not bool(multi_fidelity_summary.get("proxy_correction_active"))
            else "untracked"
        )

    calibration_policy_version = _first_nonempty_text(
        override_payload.get("calibration_policy_version"),
        world_bundle_summary.get("calibration_version"),
        manifest_payload.get("calibration_version"),
        multi_fidelity_summary.get("audit_propensity_version"),
    ) or "untracked"
    controller_policy_version = _first_nonempty_text(
        override_payload.get("controller_policy_version"),
        ROUTE_CONTROLLER_POLICY_VERSION,
    ) or ROUTE_CONTROLLER_POLICY_VERSION
    preference_model_version = _first_nonempty_text(
        override_payload.get("preference_model_version"),
        ROUTE_PREFERENCE_MODEL_VERSION,
    ) or ROUTE_PREFERENCE_MODEL_VERSION
    lane_id = _first_nonempty_text(override_payload.get("lane_id"), ROUTE_RUNTIME_LANE_ID) or ROUTE_RUNTIME_LANE_ID
    git_commit_hash = _resolve_git_commit_hash()
    environment_lockfile_hash, environment_lockfile_path = _resolve_environment_lockfile_hash()
    container_digest, container_digest_source = _resolve_container_digest()

    return {
        "run_id": run_id,
        "lane_id": lane_id,
        "variant_id": _first_nonempty_text(override_payload.get("variant_id"), pipeline_mode) or "unknown",
        "cache_mode": str(derived_cache_mode).strip().lower() or "cold",
        "seed": _safe_int(override_payload.get("seed"))
        if override_payload.get("seed") is not None
        else _safe_int(run_seed),
        "calibration_policy_version": calibration_policy_version,
        "controller_policy_version": controller_policy_version,
        "preference_model_version": preference_model_version,
        "proxy_correction_version": proxy_correction_version,
        "support_status": _first_nonempty_text(
            override_payload.get("support_status"),
            decision_support_summary.get("support_status"),
            _route_support_status(support_payload),
            "supported"
            if decision_support_summary.get("support_flag") is True
            else "unsupported"
            if decision_support_summary.get("support_flag") is False
            else None,
        )
        or "unknown",
        "cache_reuse_origin": cache_reuse_origin,
        "cache_source_id": cache_source_id,
        "reuse_count": reuse_count,
        "invalidation_reason": invalidation_reason,
        "headline_identity": {
            "schema_version": ROUTE_ARTIFACT_IDENTITY_SCHEMA_VERSION,
            "git_commit_hash": git_commit_hash,
            "environment_lockfile_hash": environment_lockfile_hash,
            "environment_lockfile_path": environment_lockfile_path,
            "container_digest": container_digest,
            "container_digest_source": container_digest_source,
            "policy_hashes": _route_policy_hashes(
                pipeline_mode=pipeline_mode,
                decision_package=decision_payload,
                final_route_trace=trace_payload,
                world_bundle_summary=world_bundle_summary,
                probabilistic_bundle=probabilistic_bundle,
                audit_bundle=audit_bundle,
                multi_fidelity_summary=multi_fidelity_summary,
                calibration_policy_version=calibration_policy_version,
                controller_policy_version=controller_policy_version,
                preference_model_version=preference_model_version,
                proxy_correction_version=proxy_correction_version,
                lane_id=lane_id,
            ),
        },
    }


def _default_payload_schema_version(artifact_name: str) -> str:
    if artifact_name.endswith(".jsonl"):
        return "jsonl-record-sequence-v1"
    if artifact_name.endswith(".csv"):
        return "csv-table-v1"
    if artifact_name.endswith(".md"):
        return "markdown-text-v1"
    if artifact_name.endswith(".geojson"):
        return "geojson-feature-collection-v1"
    if artifact_name.endswith(".json"):
        return "json-object-v1"
    return "artifact-blob-v1"


def payload_schema_version_for_artifact(
    artifact_name: str,
    payload: Mapping[str, Any] | None = None,
) -> str:
    payload_mapping = _json_mapping(payload)
    schema_version = _first_nonempty_text(
        payload_mapping.get("schema_version"),
        _json_mapping(payload_mapping.get("artifact_provenance")).get("payload_schema_version"),
    )
    return schema_version or _default_payload_schema_version(artifact_name)


def build_route_artifact_provenance(
    artifact_name: str,
    *,
    context: Mapping[str, Any],
    payload_schema_version: str | None = None,
) -> dict[str, Any]:
    context_payload = _json_mapping(context)
    return {
        "schema_version": ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "payload_schema_version": payload_schema_version or _default_payload_schema_version(artifact_name),
        "run_id": _first_nonempty_text(context_payload.get("run_id")),
        "lane_id": _first_nonempty_text(context_payload.get("lane_id"), ROUTE_RUNTIME_LANE_ID)
        or ROUTE_RUNTIME_LANE_ID,
        "variant_id": _first_nonempty_text(context_payload.get("variant_id"), "unknown") or "unknown",
        "cache_mode": _first_nonempty_text(context_payload.get("cache_mode"), "cold") or "cold",
        "seed": _safe_int(context_payload.get("seed")),
        "calibration_policy_version": _first_nonempty_text(
            context_payload.get("calibration_policy_version"),
            "untracked",
        )
        or "untracked",
        "controller_policy_version": _first_nonempty_text(
            context_payload.get("controller_policy_version"),
            ROUTE_CONTROLLER_POLICY_VERSION,
        )
        or ROUTE_CONTROLLER_POLICY_VERSION,
        "preference_model_version": _first_nonempty_text(
            context_payload.get("preference_model_version"),
            ROUTE_PREFERENCE_MODEL_VERSION,
        )
        or ROUTE_PREFERENCE_MODEL_VERSION,
        "proxy_correction_version": _first_nonempty_text(
            context_payload.get("proxy_correction_version"),
            "inactive",
        )
        or "inactive",
        "support_status": _first_nonempty_text(context_payload.get("support_status"), "unknown") or "unknown",
        "cache_reuse_origin": _first_nonempty_text(context_payload.get("cache_reuse_origin"), "miss") or "miss",
        "cache_source_id": _first_nonempty_text(context_payload.get("cache_source_id")),
        "reuse_count": _safe_int(context_payload.get("reuse_count")) or 0,
        "invalidation_reason": _first_nonempty_text(
            context_payload.get("invalidation_reason"),
            "not_recorded",
        )
        or "not_recorded",
        "headline_identity": {
            "schema_version": ROUTE_ARTIFACT_IDENTITY_SCHEMA_VERSION,
            **_json_mapping(context_payload.get("headline_identity")),
        },
        "theorem_to_artifact_hooks": route_artifact_theorem_hooks(artifact_name),
    }


def _artifact_entries(
    run_id: str,
    artifact_names: list[str],
    *,
    route_provenance_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    artifact_entries: list[dict[str, Any]] = []
    for artifact_name in artifact_names:
        path = artifact_path_for_name(run_id, artifact_name)
        if not path.exists() or not path.is_file():
            continue
        entry = {
            "name": artifact_name,
            "relative_path": artifact_name,
            "endpoint": _artifact_endpoint(run_id, artifact_name),
            "present": True,
            "size_bytes": int(path.stat().st_size),
        }
        summary_name = _summary_artifact_name(artifact_name)
        if summary_name is not None:
            summary_path = artifact_path_for_name(run_id, summary_name)
            entry["markdown_summary_name"] = summary_name
            entry["markdown_summary_relative_path"] = summary_name if summary_path.exists() else None
            entry["markdown_summary_endpoint"] = (
                _artifact_endpoint(run_id, summary_name) if summary_path.exists() else None
            )
            entry["markdown_summary_present"] = summary_path.exists()
        if route_provenance_context is not None:
            payload = _load_json_artifact_dict(path)
            entry["artifact_provenance"] = build_route_artifact_provenance(
                artifact_name,
                context=route_provenance_context,
                payload_schema_version=payload_schema_version_for_artifact(artifact_name, payload),
            )
        artifact_entries.append(entry)
    return artifact_entries


def _infer_bundle_type(run_id: str, metadata: dict[str, Any] | None) -> str | None:
    explicit = str(metadata.get("type") or "").strip() if metadata else ""
    if explicit:
        return explicit
    if not metadata:
        return None

    has_thesis_markers = (
        isinstance(metadata.get("variant_count"), int)
        or isinstance(metadata.get("evaluation_suite"), dict)
    )
    if not has_thesis_markers:
        return None

    has_thesis_artifacts = any(
        artifact_path_for_name(run_id, artifact_name).exists()
        for artifact_name in ("thesis_results.csv", "thesis_summary.csv", "evaluation_manifest.json")
    )
    if not has_thesis_artifacts:
        return None
    return "thesis_evaluation"


def _export_status_entries(
    run_id: str,
    artifact_names: tuple[str, ...],
    *,
    route_provenance_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for artifact_name in artifact_names:
        path = artifact_path_for_name(run_id, artifact_name)
        present = path.exists()
        entry = {
            "name": artifact_name,
            "present": present,
            "relative_path": artifact_name if present else None,
            "endpoint": _artifact_endpoint(run_id, artifact_name) if present else None,
            "size_bytes": int(path.stat().st_size) if present else None,
        }
        if route_provenance_context is not None and present:
            payload = _load_json_artifact_dict(path)
            entry["artifact_provenance"] = build_route_artifact_provenance(
                artifact_name,
                context=route_provenance_context,
                payload_schema_version=payload_schema_version_for_artifact(artifact_name, payload),
            )
        entries.append(entry)
    return entries


def _build_run_bundle_index_payload(run_id: str) -> dict[str, Any] | None:
    metadata = _load_json_artifact_dict(artifact_path_for_name(run_id, "metadata.json"))
    bundle_type = _infer_bundle_type(run_id, metadata)
    if not metadata or not bundle_type:
        return None

    decision_package = _load_json_artifact_dict(artifact_path_for_name(run_id, "decision_package.json")) or {}
    final_route_trace = _load_json_artifact_dict(artifact_path_for_name(run_id, "final_route_trace.json")) or {}
    world_support_summary = _load_json_artifact_dict(
        artifact_path_for_name(run_id, "world_support_summary.json")
    ) or {}
    sampled_world_manifest = _load_json_artifact_dict(
        artifact_path_for_name(run_id, "sampled_world_manifest.json")
    ) or {}

    artifact_names = _bundle_artifact_names(run_id)
    artifact_provenance_context = (
        _json_mapping(metadata.get("artifact_provenance_context"))
        if bundle_type == "route_compute"
        else {}
    )
    if bundle_type == "route_compute" and not artifact_provenance_context:
        artifact_provenance_context = build_route_artifact_provenance_context(
            run_id=run_id,
            pipeline_mode=metadata.get("pipeline_mode"),
            run_seed=metadata.get("run_seed"),
            decision_package=decision_package,
            world_support_summary=world_support_summary,
            final_route_trace=final_route_trace,
            sampled_world_manifest=sampled_world_manifest,
        )
    elif bundle_type == "thesis_evaluation":
        artifact_provenance_context = _thesis_artifact_provenance_context_for_run(run_id) or {}
    source_artifacts_used = [
        artifact_name
        for artifact_name, payload in (
            ("metadata.json", metadata),
            ("decision_package.json", decision_package),
            ("final_route_trace.json", final_route_trace),
            ("world_support_summary.json", world_support_summary),
        )
        if payload
    ]
    base_payload: dict[str, Any] = {
        "schema_version": _RUN_BUNDLE_INDEX_SCHEMA_VERSION,
        "run_id": run_id,
        "bundle_type": bundle_type,
        "metadata_schema_version": metadata.get("schema_version"),
        "manifest_endpoint": metadata.get("manifest_endpoint"),
        "artifacts_endpoint": metadata.get("artifacts_endpoint"),
        "provenance_endpoint": metadata.get("provenance_endpoint"),
        "provenance_file": metadata.get("provenance_file"),
        "artifact_pointers": {},
        "artifact_names": artifact_names,
        "artifacts": _artifact_entries(
            run_id,
            artifact_names,
            route_provenance_context=artifact_provenance_context or None,
        ),
        "source_artifacts_used": source_artifacts_used,
        "title": _bundle_title(bundle_type),
    }
    if bundle_type != "route_compute":
        base_payload.update(
            {
                "pair_count": metadata.get("pair_count"),
                "error_count": metadata.get("error_count"),
                "duration_ms": metadata.get("duration_ms"),
            }
        )
        if bundle_type == "thesis_evaluation":
            base_payload["export_status"] = _export_status_entries(
                run_id,
                _THESIS_EXPORT_STATUS_ARTIFACTS,
                route_provenance_context=artifact_provenance_context or None,
            )
            if artifact_provenance_context:
                base_payload["artifact_provenance_schema_version"] = (
                    ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION
                )
                base_payload["artifact_provenance_context"] = artifact_provenance_context
                base_payload["artifact_provenance"] = build_route_artifact_provenance(
                    "index.json",
                    context=artifact_provenance_context,
                    payload_schema_version=_RUN_BUNDLE_INDEX_SCHEMA_VERSION,
                )
        return base_payload

    artifact_pointers: dict[str, str] = {}
    for payload in (decision_package, final_route_trace):
        pointers = payload.get("artifact_pointers")
        if not isinstance(pointers, dict):
            continue
        for key, value in pointers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            if value in _BUNDLE_INDEX_ARTIFACTS:
                continue
            artifact_pointers[key] = value

    support_summary = decision_package.get("support_summary")
    if not isinstance(support_summary, dict):
        support_summary = {}
    support_state = world_support_summary.get("support_state")
    if not isinstance(support_state, dict):
        support_state = {}
    support_provenance = world_support_summary.get("provenance")
    if not isinstance(support_provenance, dict):
        support_provenance = {}

    cache_summary: dict[str, Any] = {}
    for cache_name in (
        "route_cache_runtime",
        "option_build_runtime",
        "route_option_cache_runtime",
        "voi_dccs_runtime",
    ):
        cache_payload = final_route_trace.get(cache_name)
        if isinstance(cache_payload, dict):
            cache_summary[cache_name] = {
                key: cache_payload.get(key)
                for key in ("cache_hits", "cache_misses", "reuse_rate", "last_cache_key")
                if key in cache_payload
            }

    selected_route_id = (
        metadata.get("selected_route_id")
        or decision_package.get("selected_route_id")
        or world_support_summary.get("selected_route_id")
    )
    selected_certificate_basis = (
        decision_package.get("selected_certificate_basis")
        or world_support_summary.get("selected_certificate_basis")
    )
    support_flag = support_summary.get("support_flag")
    if support_flag is None:
        support_flag = support_state.get("support_flag")
    support_reason = support_summary.get("support_reason")
    if support_reason is None:
        support_reason = support_provenance.get("support_reason") or support_state.get(
            "out_of_support_reason"
        )

    theorem_to_artifact_hooks = {
        artifact_name: hooks
        for artifact_name in artifact_names
        if (hooks := route_artifact_theorem_hooks(artifact_name))
    }

    base_payload.update(
        {
            "pipeline_mode": metadata.get("pipeline_mode"),
            "run_seed": metadata.get("run_seed"),
            "selected_route_id": selected_route_id,
            "selected_certificate_basis": selected_certificate_basis,
            "terminal_type": decision_package.get("terminal_type"),
            "support_flag": support_flag,
            "support_reason": support_reason,
            "artifact_pointers": artifact_pointers,
            "cache_summary": cache_summary,
            "artifact_provenance_schema_version": ROUTE_ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            "theorem_to_artifact_hook_schema_version": ROUTE_THEOREM_HOOK_SCHEMA_VERSION,
            "artifact_provenance_context": artifact_provenance_context,
            "theorem_to_artifact_hooks": theorem_to_artifact_hooks,
            "bundle_index_artifacts": _export_status_entries(
                run_id,
                ("index.json", "index.md"),
                route_provenance_context=artifact_provenance_context,
            ),
        }
    )
    base_payload["artifact_provenance"] = build_route_artifact_provenance(
        "index.json",
        context=artifact_provenance_context,
        payload_schema_version=_RUN_BUNDLE_INDEX_SCHEMA_VERSION,
    )
    return base_payload


def _render_route_bundle_index_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload.get('title') or _bundle_title(payload.get('bundle_type'))}",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Bundle Type: `{payload.get('bundle_type')}`",
    ]
    if payload.get("pipeline_mode") is not None:
        lines.append(f"- Pipeline Mode: `{payload.get('pipeline_mode')}`")
    if payload.get("run_seed") is not None:
        lines.append(f"- Run Seed: `{payload.get('run_seed')}`")
    if payload.get("selected_route_id") is not None:
        lines.append(f"- Selected Route ID: `{payload.get('selected_route_id')}`")
    if payload.get("selected_certificate_basis") is not None:
        lines.append(f"- Selected Certificate Basis: `{payload.get('selected_certificate_basis')}`")
    if payload.get("terminal_type") is not None:
        lines.append(f"- Terminal Type: `{payload.get('terminal_type')}`")
    if payload.get("support_flag") is not None:
        lines.append(f"- Support Flag: `{payload.get('support_flag')}`")
    if payload.get("pair_count") is not None:
        lines.append(f"- Pair Count: `{payload.get('pair_count')}`")
    if payload.get("error_count") is not None:
        lines.append(f"- Error Count: `{payload.get('error_count')}`")
    if payload.get("duration_ms") is not None:
        lines.append(f"- Duration Ms: `{payload.get('duration_ms')}`")
    lines.extend(["", "## Artifact Pointers", ""])
    artifact_pointers = payload.get("artifact_pointers")
    if isinstance(artifact_pointers, dict) and artifact_pointers:
        for key, value in sorted(artifact_pointers.items()):
            lines.append(f"- `{key}` -> `{value}`")
    else:
        lines.append("- No artifact pointers recorded.")

    lines.extend(["", "## Artifacts", ""])
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            summary_name = artifact.get("markdown_summary_name")
            summary_endpoint = artifact.get("markdown_summary_endpoint")
            summary_suffix = (
                f"; summary `{summary_name}` -> `{summary_endpoint}`"
                if summary_name and summary_endpoint
                else ""
            )
            lines.append(
                f"- `{artifact.get('name')}` ({artifact.get('size_bytes')} bytes) -> `{artifact.get('endpoint')}`{summary_suffix}"
            )
    else:
        lines.append("- No artifacts recorded.")

    export_status = payload.get("export_status")
    if isinstance(export_status, list) and export_status:
        lines.extend(["", "## Export Status", ""])
        for artifact in export_status:
            if not isinstance(artifact, dict):
                continue
            status = "present" if artifact.get("present") else "absent"
            size_bytes = artifact.get("size_bytes")
            if status == "present" and size_bytes is not None:
                lines.append(f"- `{artifact.get('name')}`: {status} ({size_bytes} bytes)")
            else:
                lines.append(f"- `{artifact.get('name')}`: {status}")

    return "\n".join(lines) + "\n"


def _refresh_route_bundle_index_if_ready(run_id: str, *, artifact_name: str) -> None:
    if artifact_name in _BUNDLE_INDEX_ARTIFACTS:
        return

    _refresh_thesis_json_artifact_companions_and_provenance(run_id)
    payload = _build_run_bundle_index_payload(run_id)
    if payload is None:
        return

    out_dir = artifact_dir_for_run(run_id)
    _write_json(out_dir / "index.json", payload)
    (out_dir / "index.md").write_text(
        _render_route_bundle_index_markdown(payload),
        encoding="utf-8",
    )
    refreshed_payload = _build_run_bundle_index_payload(run_id)
    if refreshed_payload is None:
        return
    _write_json(out_dir / "index.json", refreshed_payload)
    (out_dir / "index.md").write_text(
        _render_route_bundle_index_markdown(refreshed_payload),
        encoding="utf-8",
    )


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
    _write_json_summary_companion(run_id, artifact_name, normalized)
    _refresh_route_bundle_index_if_ready(run_id, artifact_name=artifact_name)
    return path


def write_jsonl_artifact(run_id: str, artifact_name: str, rows: list[dict[str, Any]]) -> Path:
    path = artifact_path_for_name(run_id, artifact_name)
    _write_jsonl(path, rows)
    _refresh_route_bundle_index_if_ready(run_id, artifact_name=artifact_name)
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
    _refresh_route_bundle_index_if_ready(run_id, artifact_name=artifact_name)
    return path


def write_text_artifact(run_id: str, artifact_name: str, text: str) -> Path:
    path = artifact_path_for_name(run_id, artifact_name)
    path.write_text(str(text), encoding="utf-8")
    _refresh_route_bundle_index_if_ready(run_id, artifact_name=artifact_name)
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

    normalized_results = jsonable_encoder(results_payload)
    normalized_metadata = jsonable_encoder(metadata_payload)
    _write_json(results_path, normalized_results)
    _write_json(metadata_path, normalized_metadata)
    _write_json_summary_companion(run_id, "results.json", normalized_results)
    _write_json_summary_companion(run_id, "metadata.json", normalized_metadata)
    _write_csv(csv_path, csv_rows)
    _refresh_route_bundle_index_if_ready(run_id, artifact_name="results.csv")

    return {
        "results.json": results_path,
        "metadata.json": metadata_path,
        "results.csv": csv_path,
    }
