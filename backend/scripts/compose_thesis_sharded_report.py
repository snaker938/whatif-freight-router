from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.compose_thesis_suite_report as compose_module
from app.run_store import artifact_dir_for_run, write_csv_artifact, write_json_artifact, write_manifest, write_text_artifact
from app.settings import settings
from scripts.run_thesis_evaluation import (
    COHORT_SUMMARY_FIELDS,
    RESULT_FIELDS,
    SUMMARY_FIELDS,
    _cohort_composition,
    _cohort_scaffolding_payload,
    _cohort_summary_artifact_payload,
    _cohort_summary_rows,
    _finalize_cross_variant_metrics,
    _methods_appendix,
    _now,
    _summary_rows,
    _thesis_report,
    _validate_written_output_artifacts,
)


DEFAULT_CANONICAL_CORPUS = PROJECT_ROOT / "data" / "eval" / "uk_od_corpus_thesis_broad.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose a thesis evaluation bundle from completed shard result CSVs.")
    parser.add_argument("--run-id", default="thesis_sharded_campaign")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "out"))
    parser.add_argument("--canonical-corpus", default=str(DEFAULT_CANONICAL_CORPUS))
    parser.add_argument("--results-csv", action="append", required=True)
    parser.add_argument("--ors-baseline-policy", default="repo_local")
    parser.add_argument("--ors-snapshot-mode", default="off")
    return parser


def _row_merge_key(row: dict[str, Any]) -> str:
    return f"{str(row.get('variant_id') or '').strip()}|{compose_module._row_identity_key(row)}"


def _row_pair_key(row: dict[str, Any]) -> str:
    return compose_module._row_identity_key(row)


def _load_optional_json_artifact(path: Path, *, artifact_name: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid_shard_artifact:{artifact_name}:{path}")
    return payload


def _artifact_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _artifact_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _rate_from_bools(values: list[bool | None]) -> dict[str, Any]:
    present = [value for value in values if isinstance(value, bool)]
    denominator = len(present)
    numerator = sum(1 for value in present if value)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": (numerator / denominator) if denominator else None,
    }


def _baseline_comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted(
        {
            str(row.get("variant_id") or "").strip()
            for row in rows
            if str(row.get("variant_id") or "").strip()
        }
    )
    summary: dict[str, Any] = {}
    for variant_id in variants:
        variant_rows = [
            row for row in rows if str(row.get("variant_id") or "").strip() == variant_id
        ]
        success_count = sum(
            1 for row in variant_rows if not str(row.get("failure_reason") or "").strip()
        )
        failure_count = len(variant_rows) - success_count
        artifact_complete = _rate_from_bools(
            [_coerce_bool(row.get("artifact_complete")) for row in variant_rows]
        )
        route_evidence_ok = _rate_from_bools(
            [_coerce_bool(row.get("route_evidence_ok")) for row in variant_rows]
        )
        weighted_osrm = _rate_from_bools(
            [_coerce_bool(row.get("weighted_win_osrm")) for row in variant_rows]
        )
        weighted_ors = _rate_from_bools(
            [_coerce_bool(row.get("weighted_win_ors")) for row in variant_rows]
        )
        dominance_osrm = _rate_from_bools(
            [_coerce_bool(row.get("dominates_osrm")) for row in variant_rows]
        )
        dominance_ors = _rate_from_bools(
            [_coerce_bool(row.get("dominates_ors")) for row in variant_rows]
        )
        time_preserving_osrm = _rate_from_bools(
            [_coerce_bool(row.get("time_preserving_win_osrm")) for row in variant_rows]
        )
        time_preserving_ors = _rate_from_bools(
            [_coerce_bool(row.get("time_preserving_win_ors")) for row in variant_rows]
        )
        summary[variant_id] = {
            "row_count": len(variant_rows),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": (success_count / len(variant_rows)) if variant_rows else None,
            "artifact_complete_rate": artifact_complete["rate"],
            "route_evidence_ok_rate": route_evidence_ok["rate"],
            "weighted_win_rate_osrm": weighted_osrm["rate"],
            "weighted_win_rate_ors": weighted_ors["rate"],
            "dominance_win_rate_osrm": dominance_osrm["rate"],
            "dominance_win_rate_ors": dominance_ors["rate"],
            "time_preserving_win_rate_osrm": time_preserving_osrm["rate"],
            "time_preserving_win_rate_ors": time_preserving_ors["rate"],
        }
    return summary


def _regression_summary(
    *,
    shard_rows: list[dict[str, Any]],
    replay_pair_keys: set[str],
    expected_replay_pair_keys: set[str],
) -> dict[str, Any]:
    replay_rows = [row for row in shard_rows if _row_pair_key(row) in replay_pair_keys]
    failure_row_count = sum(
        1 for row in replay_rows if str(row.get("failure_reason") or "").strip()
    )
    artifact_incomplete_row_count = sum(
        1 for row in replay_rows if _coerce_bool(row.get("artifact_complete")) is False
    )
    route_evidence_gap_row_count = sum(
        1 for row in replay_rows if _coerce_bool(row.get("route_evidence_ok")) is False
    )
    missing_replay_pair_keys = sorted(expected_replay_pair_keys - replay_pair_keys)
    replay_complete = len(missing_replay_pair_keys) == 0
    replay_rows_all_green = (
        failure_row_count == 0
        and artifact_incomplete_row_count == 0
        and route_evidence_gap_row_count == 0
    )
    return {
        "expected_replay_pair_count": len(expected_replay_pair_keys),
        "replayed_pair_count": len(replay_pair_keys),
        "missing_replay_pair_count": len(missing_replay_pair_keys),
        "missing_replay_pair_keys": missing_replay_pair_keys,
        "replay_row_count": len(replay_rows),
        "replay_failure_row_count": failure_row_count,
        "replay_artifact_incomplete_row_count": artifact_incomplete_row_count,
        "replay_route_evidence_gap_row_count": route_evidence_gap_row_count,
        "replay_complete": replay_complete,
        "replay_rows_all_green": replay_rows_all_green,
        "regression_green": replay_complete and replay_rows_all_green,
    }


def _campaign_rung_summary(
    rung_evidence_ledger: list[dict[str, Any]],
    *,
    final_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    final_od_ids = sorted(
        {
            str(row.get("od_id") or "").strip()
            for row in final_rows
            if str(row.get("od_id") or "").strip()
        }
    )
    return {
        "rung_count": len(rung_evidence_ledger),
        "final_composed_pair_count": len({_row_pair_key(row) for row in final_rows}),
        "final_composed_od_ids": final_od_ids,
        "all_rungs_replay_complete": all(
            bool((_mapping_or_none(entry.get("regression_summary")) or {}).get("replay_complete"))
            for entry in rung_evidence_ledger
        )
        if rung_evidence_ledger
        else True,
        "all_rungs_regression_green": all(
            bool((_mapping_or_none(entry.get("regression_summary")) or {}).get("regression_green"))
            for entry in rung_evidence_ledger
        )
        if rung_evidence_ledger
        else True,
        "latest_rung_newly_admitted_od_ids": list(
            (rung_evidence_ledger[-1].get("newly_admitted_od_ids") or [])
        )
        if rung_evidence_ledger
        else [],
        "latest_rung_replay_od_ids": list(
            (rung_evidence_ledger[-1].get("replay_od_ids") or [])
        )
        if rung_evidence_ledger
        else [],
    }


def _resolve_shard_artifact_evidence(results_path: Path) -> dict[str, Any]:
    evaluation_manifest_path = results_path.with_name("evaluation_manifest.json")
    metadata_path = results_path.with_name("metadata.json")
    evaluation_manifest = _load_optional_json_artifact(
        evaluation_manifest_path,
        artifact_name="evaluation_manifest.json",
    )
    metadata = _load_optional_json_artifact(
        metadata_path,
        artifact_name="metadata.json",
    )
    evaluation_suite = _mapping_or_none(
        (evaluation_manifest or {}).get("evaluation_suite")
    ) or _mapping_or_none((metadata or {}).get("evaluation_suite"))
    readiness_summary = _mapping_or_none(
        (evaluation_manifest or {}).get("backend_ready_summary")
    ) or _mapping_or_none((metadata or {}).get("backend_ready_summary"))
    baseline_smoke_summary = _mapping_or_none(
        (evaluation_manifest or {}).get("baseline_smoke_summary")
    ) or _mapping_or_none((metadata or {}).get("baseline_smoke_summary"))
    run_validity = _mapping_or_none(
        (evaluation_manifest or {}).get("run_validity")
    ) or _mapping_or_none((metadata or {}).get("run_validity"))
    return {
        "evaluation_manifest_path": str(evaluation_manifest_path) if evaluation_manifest is not None else None,
        "evaluation_manifest_present": evaluation_manifest is not None,
        "metadata_path": str(metadata_path) if metadata is not None else None,
        "metadata_present": metadata is not None,
        "companion_run_id": _artifact_text((metadata or {}).get("run_id"))
        or _artifact_text((evaluation_manifest or {}).get("run_id")),
        "evaluation_suite_role": _artifact_text((evaluation_suite or {}).get("role")),
        "evaluation_suite_family": _artifact_text((evaluation_suite or {}).get("family")),
        "strict_evidence_policy": _artifact_text((evaluation_manifest or {}).get("strict_evidence_policy"))
        or _artifact_text((metadata or {}).get("strict_evidence_policy")),
        "ors_baseline_policy": _artifact_text((evaluation_manifest or {}).get("ors_baseline_policy"))
        or _artifact_text((metadata or {}).get("ors_baseline_policy")),
        "corpus_hash": _artifact_text((evaluation_manifest or {}).get("corpus_hash"))
        or _artifact_text((metadata or {}).get("corpus_hash")),
        "repo_asset_preflight_required_ok": _artifact_bool(
            (evaluation_manifest or {}).get("repo_asset_preflight_required_ok")
        )
        if _artifact_bool((evaluation_manifest or {}).get("repo_asset_preflight_required_ok")) is not None
        else _artifact_bool((metadata or {}).get("repo_asset_preflight_required_ok")),
        "backend_strict_route_ready": _artifact_bool((readiness_summary or {}).get("strict_route_ready")),
        "baseline_smoke_required_ok": _artifact_bool((baseline_smoke_summary or {}).get("required_ok")),
        "run_validity": dict(run_validity) if run_validity is not None else None,
        "cache_mode": _artifact_text((evaluation_manifest or {}).get("cache_mode"))
        or _artifact_text((metadata or {}).get("cache_mode")),
        "cache_reset_scope": _artifact_text((evaluation_manifest or {}).get("cache_reset_scope"))
        or _artifact_text((metadata or {}).get("cache_reset_scope")),
        "cache_reset_policy": _artifact_text((evaluation_manifest or {}).get("cache_reset_policy"))
        or _artifact_text((metadata or {}).get("cache_reset_policy")),
        "cache_reset_count": (metadata or {}).get("cache_reset_count")
        if (metadata or {}).get("cache_reset_count") is not None
        else (evaluation_manifest or {}).get("cache_reset_count"),
    }


def _bool_rollup(shards: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    present_values = [entry[field_name] for entry in shards if isinstance(entry.get(field_name), bool)]
    return {
        "present_count": len(present_values),
        "true_count": sum(1 for value in present_values if value),
        "false_count": sum(1 for value in present_values if not value),
        "all_true": all(present_values) if present_values and len(present_values) == len(shards) else None,
    }


def _shard_evidence_rollup(shards: list[dict[str, Any]]) -> dict[str, Any]:
    strict_evidence_policy_values = sorted(
        {
            value
            for value in (_artifact_text(entry.get("strict_evidence_policy")) for entry in shards)
            if value is not None
        }
    )
    ors_baseline_policy_values = sorted(
        {
            value
            for value in (_artifact_text(entry.get("ors_baseline_policy")) for entry in shards)
            if value is not None
        }
    )
    cache_mode_values = sorted(
        {
            value
            for value in (_artifact_text(entry.get("cache_mode")) for entry in shards)
            if value is not None
        }
    )
    run_validity_entries = [
        entry["run_validity"]
        for entry in shards
        if isinstance(entry.get("run_validity"), Mapping)
    ]
    strict_live_rates = sorted(
        {
            float(entry["strict_live_readiness_pass_rate"])
            for entry in run_validity_entries
            if entry.get("strict_live_readiness_pass_rate") is not None
        }
    )
    rerun_success_rates = sorted(
        {
            float(entry["evaluation_rerun_success_rate"])
            for entry in run_validity_entries
            if entry.get("evaluation_rerun_success_rate") is not None
        }
    )
    scenario_unavailable_rates = sorted(
        {
            float(entry["scenario_profile_unavailable_rate"])
            for entry in run_validity_entries
            if entry.get("scenario_profile_unavailable_rate") is not None
        }
    )
    all_run_validity_clean = (
        all(
            float(entry.get("strict_live_readiness_pass_rate", 0.0)) >= 1.0
            and float(entry.get("evaluation_rerun_success_rate", 0.0)) >= 1.0
            and float(entry.get("scenario_profile_unavailable_rate", 1.0)) <= 0.0
            for entry in run_validity_entries
        )
        if run_validity_entries and len(run_validity_entries) == len(shards)
        else None
    )
    return {
        "evaluation_manifest_present_count": sum(
            1 for entry in shards if entry.get("evaluation_manifest_present") is True
        ),
        "metadata_present_count": sum(
            1 for entry in shards if entry.get("metadata_present") is True
        ),
        "missing_evaluation_manifest_count": sum(
            1 for entry in shards if entry.get("evaluation_manifest_present") is not True
        ),
        "missing_metadata_count": sum(
            1 for entry in shards if entry.get("metadata_present") is not True
        ),
        "strict_evidence_policy_values": strict_evidence_policy_values,
        "ors_baseline_policy_values": ors_baseline_policy_values,
        "cache_mode_values": cache_mode_values,
        "repo_asset_preflight_required_ok": _bool_rollup(
            shards,
            "repo_asset_preflight_required_ok",
        ),
        "backend_strict_route_ready": _bool_rollup(
            shards,
            "backend_strict_route_ready",
        ),
        "baseline_smoke_required_ok": _bool_rollup(
            shards,
            "baseline_smoke_required_ok",
        ),
        "run_validity": {
            "present_count": len(run_validity_entries),
            "strict_live_readiness_pass_rates": strict_live_rates,
            "evaluation_rerun_success_rates": rerun_success_rates,
            "scenario_profile_unavailable_rates": scenario_unavailable_rates,
            "all_clean": all_run_validity_clean,
        },
    }


def _aggregate_report_inputs(shard_evidence_rollup: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight_rollup = _mapping_or_none(shard_evidence_rollup.get("repo_asset_preflight_required_ok")) or {}
    readiness_rollup = _mapping_or_none(shard_evidence_rollup.get("backend_strict_route_ready")) or {}
    baseline_rollup = _mapping_or_none(shard_evidence_rollup.get("baseline_smoke_required_ok")) or {}
    run_validity_rollup = _mapping_or_none(shard_evidence_rollup.get("run_validity")) or {}
    preflight_summary = {
        "required_ok": preflight_rollup.get("all_true"),
        "source": "aggregated_shard_companions",
        "present_count": preflight_rollup.get("present_count"),
        "missing_count": shard_evidence_rollup.get("missing_evaluation_manifest_count"),
    }
    readiness_summary = {
        "strict_route_ready": readiness_rollup.get("all_true"),
        "strict_live": {"ok": run_validity_rollup.get("all_clean")},
        "source": "aggregated_shard_companions",
        "present_count": readiness_rollup.get("present_count"),
        "missing_count": shard_evidence_rollup.get("missing_evaluation_manifest_count"),
    }
    baseline_smoke_summary = {
        "required_ok": baseline_rollup.get("all_true"),
        "source": "aggregated_shard_companions",
        "present_count": baseline_rollup.get("present_count"),
        "missing_count": shard_evidence_rollup.get("missing_evaluation_manifest_count"),
    }
    return preflight_summary, readiness_summary, baseline_smoke_summary


def _load_shard_rows(
    results_paths: list[str],
    *,
    canonical_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    merged_rows_by_key: dict[str, dict[str, Any]] = {}
    ordered_merge_keys: list[str] = []
    shard_sources: list[dict[str, Any]] = []
    rung_evidence_ledger: list[dict[str, Any]] = []
    seen_pair_keys: set[str] = set()
    seen_od_ids: set[str] = set()
    for rung_index, raw_path in enumerate(results_paths, start=1):
        path = Path(raw_path).resolve()
        rows = compose_module._load_rows(str(path))
        normalized = compose_module._normalize_rows(rows)
        canonicalized = compose_module._canonicalize_rows(normalized, canonical_index=canonical_index)
        shard_seen_merge_keys: dict[str, str] = {}
        for row in canonicalized:
            merge_key = _row_merge_key(row)
            if merge_key in shard_seen_merge_keys:
                raise RuntimeError(f"duplicate_shard_row:{merge_key}:{path}:{path}")
            shard_seen_merge_keys[merge_key] = str(path)
        artifact_run_ids = sorted(
            {
                str(row.get("artifact_run_id") or "").strip()
                for row in canonicalized
                if str(row.get("artifact_run_id") or "").strip()
            }
        )
        variant_ids = sorted(
            {
                str(row.get("variant_id") or "").strip()
                for row in canonicalized
                if str(row.get("variant_id") or "").strip()
            }
        )
        pair_keys = {_row_pair_key(row) for row in canonicalized}
        pair_count = len(pair_keys)
        shard_od_ids = sorted(
            {
                str(row.get("od_id") or "").strip()
                for row in canonicalized
                if str(row.get("od_id") or "").strip()
            }
        )
        expected_replay_pair_keys = set(seen_pair_keys)
        replay_pair_keys = pair_keys & expected_replay_pair_keys
        new_pair_keys = pair_keys - expected_replay_pair_keys
        newly_admitted_od_ids = sorted(
            {
                str(row.get("od_id") or "").strip()
                for row in canonicalized
                if _row_pair_key(row) in new_pair_keys and str(row.get("od_id") or "").strip()
            }
        )
        replay_od_ids = sorted(
            {
                str(row.get("od_id") or "").strip()
                for row in canonicalized
                if _row_pair_key(row) in replay_pair_keys and str(row.get("od_id") or "").strip()
            }
        )
        missing_replay_od_ids = sorted(set(seen_od_ids) - set(shard_od_ids))
        regression_summary = _regression_summary(
            shard_rows=canonicalized,
            replay_pair_keys=replay_pair_keys,
            expected_replay_pair_keys=expected_replay_pair_keys,
        )
        baseline_comparison_summary = _baseline_comparison_summary(canonicalized)
        shard_evidence = _resolve_shard_artifact_evidence(path)
        rung_entry = {
            "rung_index": rung_index,
            "path": str(path),
            "row_count": len(canonicalized),
            "pair_count": pair_count,
            "od_ids": shard_od_ids,
            "newly_admitted_od_ids": newly_admitted_od_ids,
            "replay_od_ids": replay_od_ids,
            "missing_replay_od_ids": missing_replay_od_ids,
            "new_pair_count": len(new_pair_keys),
            "replay_pair_count": len(replay_pair_keys),
            "regression_summary": regression_summary,
            "baseline_comparison_summary": baseline_comparison_summary,
            "artifact_run_ids": artifact_run_ids,
        }
        rung_evidence_ledger.append(rung_entry)
        shard_sources.append(
            {
                "rung_index": rung_index,
                "path": str(path),
                "row_count": len(canonicalized),
                "pair_count": pair_count,
                "od_ids": shard_od_ids,
                "newly_admitted_od_ids": newly_admitted_od_ids,
                "replay_od_ids": replay_od_ids,
                "missing_replay_od_ids": missing_replay_od_ids,
                "new_pair_count": len(new_pair_keys),
                "replay_pair_count": len(replay_pair_keys),
                "regression_summary": regression_summary,
                "baseline_comparison_summary": baseline_comparison_summary,
                "variant_ids": variant_ids,
                "artifact_run_ids": artifact_run_ids,
                **shard_evidence,
            }
        )
        for row in canonicalized:
            key = _row_merge_key(row)
            if key not in merged_rows_by_key:
                ordered_merge_keys.append(key)
            merged_rows_by_key[key] = row
        seen_pair_keys |= pair_keys
        seen_od_ids |= set(shard_od_ids)
    merged_rows = [merged_rows_by_key[key] for key in ordered_merge_keys]
    campaign_rung_summary = _campaign_rung_summary(
        rung_evidence_ledger,
        final_rows=merged_rows,
    )
    return merged_rows, shard_sources, rung_evidence_ledger, campaign_rung_summary


def _sharded_methods_appendix(source_manifest: dict[str, Any], *, ors_baseline_policy: str, ors_snapshot_mode: str) -> str:
    lines = [
        _methods_appendix(
            argparse.Namespace(
                search_budget="mixed",
                evidence_budget="mixed",
                world_count="mixed",
                certificate_threshold="mixed",
                tau_stop="mixed",
                baseline_refinement_policy="corridor_uniform",
                ors_baseline_policy=ors_baseline_policy,
                ors_snapshot_mode=ors_snapshot_mode,
                ready_timeout_seconds="n/a",
                ready_poll_seconds="n/a",
                max_alternatives="mixed",
                disable_tolls=False,
                in_process_backend=False,
                backend_url="composed-from-shards",
            ),
            corpus_hash=source_manifest["suite_hash"],
            row_count=int(source_manifest["pair_count"]),
        ),
        "",
        "Shard composition:",
        f"- shard_count: `{source_manifest['shard_count']}`",
        f"- total_row_count: `{source_manifest['row_count']}`",
        f"- pair_count: `{source_manifest['pair_count']}`",
        "- All rows are sourced from completed shard `thesis_results.csv` outputs; no synthetic result rows are added here.",
        "",
        "Shard sources:",
    ]
    for shard in source_manifest["shards"]:
        lines.append(
            f"- `{shard['path']}`: rows={shard['row_count']}, pairs={shard['pair_count']}, "
            f"variants={','.join(shard.get('variant_ids', [])) or 'none'}, "
            f"artifact_run_ids={','.join(shard.get('artifact_run_ids', [])) or 'none'}"
        )
    return "\n".join(lines)


def _append_report_sections(
    thesis_report_text: str,
    *,
    source_manifest: Mapping[str, Any],
    prior_coverage: Mapping[str, Any],
    campaign_rung_summary: Mapping[str, Any],
    rung_evidence_ledger: list[dict[str, Any]],
    shard_sources_path: str,
) -> str:
    lines = [
        thesis_report_text.rstrip(),
        "",
        "## Shard Provenance",
        f"- shard_count={source_manifest.get('shard_count')}",
        f"- shard_sources_json={shard_sources_path}",
        f"- ambiguity_prior_source_mix={json.dumps(prior_coverage.get('ambiguity_prior_source_mix', {}), sort_keys=True)}",
        "",
        "## Campaign Rung Evidence",
        f"- rung_count={campaign_rung_summary.get('rung_count')}",
        f"- all_rungs_replay_complete={campaign_rung_summary.get('all_rungs_replay_complete')}",
        f"- all_rungs_regression_green={campaign_rung_summary.get('all_rungs_regression_green')}",
        f"- latest_rung_newly_admitted_od_ids={json.dumps(campaign_rung_summary.get('latest_rung_newly_admitted_od_ids', []))}",
        f"- latest_rung_replay_od_ids={json.dumps(campaign_rung_summary.get('latest_rung_replay_od_ids', []))}",
    ]
    for entry in rung_evidence_ledger:
        regression_summary = _mapping_or_none(entry.get("regression_summary")) or {}
        baseline_summary = _mapping_or_none(entry.get("baseline_comparison_summary")) or {}
        weighted_win_osrm_by_variant = {
            variant_id: variant_summary.get("weighted_win_rate_osrm")
            for variant_id, variant_summary in sorted(baseline_summary.items())
            if isinstance(variant_summary, Mapping)
        }
        weighted_win_ors_by_variant = {
            variant_id: variant_summary.get("weighted_win_rate_ors")
            for variant_id, variant_summary in sorted(baseline_summary.items())
            if isinstance(variant_summary, Mapping)
        }
        lines.append(
            "- "
            + (
                f"rung_{entry.get('rung_index')}: "
                f"new={json.dumps(entry.get('newly_admitted_od_ids', []))}; "
                f"replay={json.dumps(entry.get('replay_od_ids', []))}; "
                f"missing_replay={json.dumps(entry.get('missing_replay_od_ids', []))}; "
                f"replay_complete={regression_summary.get('replay_complete')}; "
                f"replay_rows_all_green={regression_summary.get('replay_rows_all_green')}; "
                f"regression_green={regression_summary.get('regression_green')}; "
                f"weighted_win_rate_osrm_by_variant={json.dumps(weighted_win_osrm_by_variant, sort_keys=True)}; "
                f"weighted_win_rate_ors_by_variant={json.dumps(weighted_win_ors_by_variant, sort_keys=True)}"
            )
        )
    return "\n".join(lines)


def compose_sharded_report(args: argparse.Namespace) -> dict[str, Any]:
    old_out_dir = settings.out_dir
    settings.out_dir = str(Path(args.out_dir))
    try:
        canonical_index = compose_module._load_canonical_corpus_index(str(args.canonical_corpus))
        rows, shard_sources, rung_evidence_ledger, campaign_rung_summary = _load_shard_rows(
            list(args.results_csv or []),
            canonical_index=canonical_index,
        )
        rows = _finalize_cross_variant_metrics(rows)
        summary_rows = _summary_rows(rows)
        cohort_summary_rows = _cohort_summary_rows(rows)
        cohort_composition = _cohort_composition(rows)
        prior_coverage = compose_module._prior_coverage_summary(rows, canonical_index=canonical_index)
        cohort_scaffolding = _cohort_scaffolding_payload()
        evaluation_suite = {
            "role": "composed_sharded_report",
            "family": "evaluation",
            "scope": "campaign",
            "focus": "sharded",
            "source": "compose_sharded_report",
        }

        run_id = str(args.run_id)
        pair_count = len({compose_module._row_identity_key(row) for row in rows})
        suite_hash = compose_module._suite_hash(rows)
        shard_evidence_rollup = _shard_evidence_rollup(shard_sources)
        preflight_summary, readiness_summary, baseline_smoke_summary = _aggregate_report_inputs(
            shard_evidence_rollup
        )
        source_manifest = {
            "composition_type": "sharded_results_merge",
            "row_count": len(rows),
            "raw_input_row_count": sum(int(entry.get("row_count") or 0) for entry in shard_sources),
            "pair_count": pair_count,
            "suite_hash": suite_hash,
            "shard_count": len(shard_sources),
            "shards": shard_sources,
            "rung_evidence_ledger": rung_evidence_ledger,
            "campaign_rung_summary": campaign_rung_summary,
            "canonical_corpus": {"path": str(args.canonical_corpus)},
            "ambiguity_prior_coverage": prior_coverage,
            "shard_evidence_rollup": shard_evidence_rollup,
        }
        shard_sources_path = write_json_artifact(run_id, "shard_sources.json", source_manifest)
        write_json_artifact(run_id, "prior_coverage_summary.json", prior_coverage)
        cohort_composition_path = write_json_artifact(run_id, "cohort_composition.json", cohort_composition)

        results_csv = write_csv_artifact(run_id, "thesis_results.csv", fieldnames=RESULT_FIELDS, rows=rows)
        write_json_artifact(run_id, "thesis_results.json", {"rows": rows})
        summary_csv = write_csv_artifact(run_id, "thesis_summary.csv", fieldnames=SUMMARY_FIELDS, rows=summary_rows)
        write_json_artifact(run_id, "thesis_summary.json", {"summary_rows": summary_rows})
        cohort_summary_csv = write_csv_artifact(run_id, "thesis_summary_by_cohort.csv", fieldnames=COHORT_SUMMARY_FIELDS, rows=cohort_summary_rows)
        cohort_summary_json = write_json_artifact(
            run_id,
            "thesis_summary_by_cohort.json",
            _cohort_summary_artifact_payload(cohort_summary_rows),
        )

        methods_text = _sharded_methods_appendix(
            source_manifest,
            ors_baseline_policy=str(args.ors_baseline_policy),
            ors_snapshot_mode=str(args.ors_snapshot_mode),
        )
        methods_path = write_text_artifact(run_id, "methods_appendix.md", methods_text)
        evaluation_manifest_path = write_json_artifact(
            run_id,
            "evaluation_manifest.json",
            {
                "run_id": run_id,
                "created_at": _now(),
                "composition": source_manifest,
                "evaluation_suite": evaluation_suite,
                "cohort_scaffolding": cohort_scaffolding,
                "ors_baseline_policy": str(args.ors_baseline_policy),
                "ors_snapshot_mode": str(args.ors_snapshot_mode),
                "campaign_rung_summary": campaign_rung_summary,
                "rung_evidence_ledger": rung_evidence_ledger,
                "shard_evidence_rollup": shard_evidence_rollup,
                "strict_evidence_policy_values": shard_evidence_rollup["strict_evidence_policy_values"],
                "shard_preflight_summary": preflight_summary,
                "shard_readiness_summary": readiness_summary,
                "shard_baseline_smoke_summary": baseline_smoke_summary,
            },
        )
        metadata_path = write_json_artifact(
            run_id,
            "metadata.json",
            {
                "run_id": run_id,
                "row_count": len(rows),
                "failure_count": sum(1 for row in rows if row.get("failure_reason")),
                "source_runs": shard_sources,
                "evaluation_suite": evaluation_suite,
                "cohort_scaffolding": cohort_scaffolding,
                "campaign_rung_summary": campaign_rung_summary,
                "rung_evidence_ledger": rung_evidence_ledger,
                "shard_evidence_rollup": shard_evidence_rollup,
                "strict_evidence_policy_values": shard_evidence_rollup["strict_evidence_policy_values"],
                "shard_preflight_summary": preflight_summary,
                "shard_readiness_summary": readiness_summary,
                "shard_baseline_smoke_summary": baseline_smoke_summary,
            },
        )
        manifest_path = write_manifest(
            run_id,
            {
                "request": {
                    "composition": source_manifest,
                    "evaluation_suite": evaluation_suite,
                    "cohort_scaffolding": cohort_scaffolding,
                },
                "execution": {
                    "pair_count": pair_count,
                    "variant_count": len({str(row.get("variant_id") or "").strip() for row in rows if str(row.get("variant_id") or "").strip()}),
                },
            },
        )

        thesis_report_text = _thesis_report(
            run_id,
            summary_rows,
            rows=rows,
            corpus_hash=suite_hash,
            row_count=pair_count,
            ors_baseline_policy=str(args.ors_baseline_policy),
            ors_snapshot_mode=str(args.ors_snapshot_mode),
            preflight_summary=preflight_summary,
            readiness_summary=readiness_summary,
            baseline_smoke_summary=baseline_smoke_summary,
            output_validation={"validated_artifact_count": 0},
        )
        thesis_report_text = _append_report_sections(
            thesis_report_text,
            source_manifest=source_manifest,
            prior_coverage=prior_coverage,
            campaign_rung_summary=campaign_rung_summary,
            rung_evidence_ledger=rung_evidence_ledger,
            shard_sources_path=str(shard_sources_path),
        )
        thesis_report_path = write_text_artifact(run_id, "thesis_report.md", thesis_report_text)

        output_validation = _validate_written_output_artifacts(
            results_csv=Path(results_csv),
            summary_csv=Path(summary_csv),
            methods_path=Path(methods_path),
            thesis_report_path=Path(thesis_report_path),
            evaluation_manifest_path=Path(evaluation_manifest_path),
            manifest_path=Path(manifest_path),
            extra_json_paths={
                "metadata.json": Path(metadata_path),
                "thesis_results.json": artifact_dir_for_run(run_id) / "thesis_results.json",
                "thesis_summary.json": artifact_dir_for_run(run_id) / "thesis_summary.json",
                "thesis_summary_by_cohort.json": Path(cohort_summary_json),
                "shard_sources.json": Path(shard_sources_path),
                "prior_coverage_summary.json": artifact_dir_for_run(run_id) / "prior_coverage_summary.json",
                "cohort_composition.json": Path(cohort_composition_path),
            },
            extra_text_paths={},
            optional_paths={
                "thesis_summary_by_cohort.csv": Path(cohort_summary_csv),
            },
        )
        thesis_report_text = _thesis_report(
            run_id,
            summary_rows,
            rows=rows,
            corpus_hash=suite_hash,
            row_count=pair_count,
            ors_baseline_policy=str(args.ors_baseline_policy),
            ors_snapshot_mode=str(args.ors_snapshot_mode),
            preflight_summary=preflight_summary,
            readiness_summary=readiness_summary,
            baseline_smoke_summary=baseline_smoke_summary,
            output_validation=output_validation,
        )
        thesis_report_text = _append_report_sections(
            thesis_report_text,
            source_manifest=source_manifest,
            prior_coverage=prior_coverage,
            campaign_rung_summary=campaign_rung_summary,
            rung_evidence_ledger=rung_evidence_ledger,
            shard_sources_path=str(shard_sources_path),
        )
        thesis_report_path = write_text_artifact(run_id, "thesis_report.md", thesis_report_text)
        return {
            "run_id": run_id,
            "rows": rows,
            "summary_rows": summary_rows,
            "summary_by_cohort_rows": cohort_summary_rows,
            "summary_by_cohort_json": str(cohort_summary_json),
            "cohort_composition_path": str(cohort_composition_path),
            "cohort_scaffolding": cohort_scaffolding,
            "results_csv": str(results_csv),
            "summary_csv": str(summary_csv),
            "summary_by_cohort_csv": str(cohort_summary_csv),
            "thesis_report": str(thesis_report_path),
            "methods_appendix": str(methods_path),
            "evaluation_manifest": str(evaluation_manifest_path),
            "manifest_path": str(manifest_path),
            "evaluation_suite": evaluation_suite,
            "shard_sources_json": str(shard_sources_path),
            "rung_evidence_ledger": rung_evidence_ledger,
            "campaign_rung_summary": campaign_rung_summary,
            "ambiguity_prior_coverage": prior_coverage,
            "output_artifact_validation": output_validation,
        }
    finally:
        settings.out_dir = old_out_dir


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = compose_sharded_report(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
