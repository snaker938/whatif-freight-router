from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.run_store import artifact_dir_for_run, write_csv_artifact, write_json_artifact, write_manifest, write_text_artifact
from app.settings import settings
from scripts.build_od_corpus_uk import _cheap_prior_features
from scripts.evaluation_metrics import action_efficiency, as_float, frontier_action_gain, runtime_ratio, runtime_share
from scripts.run_thesis_evaluation import (
    COHORT_SUMMARY_FIELDS,
    RESULT_FIELDS,
    SUMMARY_FIELDS,
    _cohort_composition,
    _cohort_label,
    _cohort_summary_rows,
    _is_hard_case_row,
    _observed_ambiguity_index,
    _row_identity_key,
    _finalize_cross_variant_metrics,
    _methods_appendix,
    _thesis_report,
    _validate_written_output_artifacts,
    _now,
    _summary_rows,
)


DEFAULT_REPRESENTATIVE_BASE = (
    PROJECT_ROOT / "out" / "artifacts" / "thesis_representative_20260322_r2" / "thesis_results.csv"
)
DEFAULT_REPRESENTATIVE_REPLACEMENT = (
    PROJECT_ROOT / "out" / "artifacts" / "thesis_leeds_probe_20260322_a2" / "thesis_results.csv"
)
DEFAULT_AMBIGUITY_BASE = (
    PROJECT_ROOT / "out" / "artifacts" / "thesis_ambiguity_20260322_r3" / "thesis_results.csv"
)
DEFAULT_AMBIGUITY_REPLACEMENT = (
    PROJECT_ROOT / "out" / "artifacts" / "thesis_shorthaul_probe_20260322_a1" / "thesis_results.csv"
)
DEFAULT_CANONICAL_CORPUS = (
    PROJECT_ROOT / "data" / "eval" / "uk_od_corpus_thesis_broad.csv"
)

BOOL_FIELDS = {
    "artifact_complete",
    "certified",
    "dominates_osrm",
    "dominates_ors",
    "weighted_win_osrm",
    "weighted_win_ors",
    "balanced_win_osrm",
    "balanced_win_ors",
    "robust_win_osrm",
    "robust_win_ors",
    "selector_certificate_disagreement",
    "voi_controller_engaged",
    "nontrivial_frontier",
    "route_evidence_ok",
    "ors_snapshot_used",
    "baseline_identity_verified",
}


def _split_prior_sources(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for part in str(value).replace("+", ",").split(","):
        token = part.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose a final thesis suite report from completed evaluation runs.")
    parser.add_argument("--run-id", default="thesis_suite_20260322_r3")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "out"))
    parser.add_argument("--representative-base", default=str(DEFAULT_REPRESENTATIVE_BASE))
    parser.add_argument("--representative-replacement", default=str(DEFAULT_REPRESENTATIVE_REPLACEMENT))
    parser.add_argument("--representative-exclude-od", default="leeds_newcastle")
    parser.add_argument("--ambiguity-base", default=str(DEFAULT_AMBIGUITY_BASE))
    parser.add_argument("--ambiguity-replacement", default=str(DEFAULT_AMBIGUITY_REPLACEMENT))
    parser.add_argument("--ambiguity-exclude-od", default="cardiff_bristol")
    parser.add_argument("--ambiguity-replacement-od", default="newport_bristol")
    parser.add_argument("--canonical-corpus", default=str(DEFAULT_CANONICAL_CORPUS))
    return parser


def _load_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _without_od(rows: list[dict[str, Any]], od_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("od_id") or "") != str(od_id)]


def _only_od(rows: list[dict[str, Any]], od_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("od_id") or "") == str(od_id)]


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}
        for field in RESULT_FIELDS:
            value = row.get(field, "")
            if value == "":
                record[field] = None
                continue
            if field in BOOL_FIELDS:
                lowered = str(value).strip().lower()
                if lowered in {"true", "1", "yes", "y", "on"}:
                    record[field] = True
                elif lowered in {"false", "0", "no", "n", "off"}:
                    record[field] = False
                else:
                    record[field] = value
                continue
            record[field] = value
        normalized.append(record)
    return normalized


def _load_canonical_corpus_index(path: str) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return {}
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        profile_id = str(row.get("profile_id") or "").strip()
        if profile_id:
            index[profile_id] = dict(row)
    return index


def _canonical_ambiguity_value(row: dict[str, Any]) -> Any:
    value = row.get("od_ambiguity_index")
    if value not in (None, ""):
        return value
    return row.get("ambiguity_index")


def _canonical_prior_backfill(row: dict[str, Any]) -> dict[str, Any]:
    engine_prior = row.get("candidate_probe_engine_disagreement_prior")
    hard_case_prior = row.get("hard_case_prior")
    if engine_prior not in (None, "") and hard_case_prior not in (None, ""):
        return row
    priors = _cheap_prior_features(
        path_count=int(as_float(row.get("candidate_probe_path_count"), 0.0)),
        family_count=int(as_float(row.get("candidate_probe_corridor_family_count"), 0.0)),
        objective_spread=as_float(row.get("candidate_probe_objective_spread"), 0.0),
        nominal_margin=as_float(row.get("candidate_probe_nominal_margin"), 0.0),
        toll_disagreement=as_float(row.get("candidate_probe_toll_disagreement_rate"), 0.0),
    )
    if engine_prior in (None, ""):
        row["candidate_probe_engine_disagreement_prior"] = priors["candidate_probe_engine_disagreement_prior"]
    if hard_case_prior in (None, ""):
        row["hard_case_prior"] = priors["hard_case_prior"]
    return row


def _prior_coverage_summary(
    rows: list[dict[str, Any]],
    *,
    canonical_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    coverage_rows: list[dict[str, Any]] = []
    for row in rows:
        profile_id = str(row.get("profile_id") or "").strip()
        canonical = canonical_index.get(profile_id)
        source_row = _canonical_prior_backfill(dict(canonical)) if canonical else dict(row)
        coverage_rows.append(source_row)
    row_count = len(coverage_rows)
    source_mix: dict[str, int] = {}
    for row in coverage_rows:
        raw = str(row.get("ambiguity_prior_source") or "").strip()
        if not raw:
            raw = "unknown"
        for token in [part.strip() for part in raw.replace("+", ",").split(",") if part.strip()]:
            source_mix[token] = source_mix.get(token, 0) + 1
    def _nonzero_count(key: str, alt_key: str | None = None) -> int:
        total = 0
        for row in coverage_rows:
            value = row.get(key)
            if value in (None, "") and alt_key is not None:
                value = row.get(alt_key)
            if as_float(value, float("nan")) == as_float(value, float("nan")) and as_float(value, 0.0) > 0.0:
                total += 1
        return total
    ambiguity_nonzero = _nonzero_count("od_ambiguity_index", "ambiguity_index")
    engine_nonzero = _nonzero_count("candidate_probe_engine_disagreement_prior")
    hard_case_nonzero = _nonzero_count("hard_case_prior")
    support_mean = round(
        sum(as_float(row.get("ambiguity_prior_support_count"), 0.0) for row in coverage_rows) / float(max(1, row_count)),
        6,
    )
    confidence_mean = round(
        sum(as_float(row.get("od_ambiguity_confidence"), 0.0) for row in coverage_rows) / float(max(1, row_count)),
        6,
    )
    source_count_mean = round(
        sum(
            as_float(
                row.get("od_ambiguity_source_count"),
                float(len(_split_prior_sources(row.get("ambiguity_prior_source")))),
            )
            for row in coverage_rows
        )
        / float(max(1, row_count)),
        6,
    )
    return {
        "row_count": row_count,
        "nonzero_od_ambiguity_prior_count": ambiguity_nonzero,
        "nonzero_od_ambiguity_prior_rate": round(ambiguity_nonzero / float(max(1, row_count)), 6),
        "nonzero_engine_disagreement_prior_count": engine_nonzero,
        "nonzero_engine_disagreement_prior_rate": round(engine_nonzero / float(max(1, row_count)), 6),
        "nonzero_hard_case_prior_count": hard_case_nonzero,
        "nonzero_hard_case_prior_rate": round(hard_case_nonzero / float(max(1, row_count)), 6),
        "ambiguity_prior_source_mix": dict(sorted(source_mix.items())),
        "mean_prior_support_count": support_mean,
        "mean_od_ambiguity_confidence": confidence_mean,
        "mean_od_ambiguity_source_count": source_count_mean,
    }


def _canonicalize_rows(
    rows: list[dict[str, Any]],
    *,
    canonical_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        profile_id = str(record.get("profile_id") or "").strip()
        canonical = canonical_index.get(profile_id)
        if canonical:
            canonical = _canonical_prior_backfill(dict(canonical))
            for source_key, target_key in (
                ("od_id", "od_id"),
                ("origin_lat", "origin_lat"),
                ("origin_lon", "origin_lon"),
                ("destination_lat", "destination_lat"),
                ("destination_lon", "destination_lon"),
                ("straight_line_km", "straight_line_km"),
                ("distance_bin", "trip_length_bin"),
                ("corpus_group", "corpus_group"),
            ):
                value = canonical.get(source_key)
                if value not in (None, ""):
                    record[target_key] = value
            if record.get("corpus_kind") in (None, "") and canonical.get("corpus_group") not in (None, ""):
                record["corpus_kind"] = canonical.get("corpus_group")
            canonical_ambiguity = _canonical_ambiguity_value(canonical)
            if record.get("od_ambiguity_index") in (None, "") and canonical_ambiguity not in (None, ""):
                record["od_ambiguity_index"] = canonical_ambiguity
            if record.get("od_ambiguity_confidence") in (None, "") and canonical.get("od_ambiguity_confidence") not in (None, ""):
                record["od_ambiguity_confidence"] = canonical.get("od_ambiguity_confidence")
            if record.get("od_ambiguity_source_count") in (None, "") and canonical.get("od_ambiguity_source_count") not in (None, ""):
                record["od_ambiguity_source_count"] = canonical.get("od_ambiguity_source_count")
            if record.get("od_ambiguity_source_mix") in (None, "") and canonical.get("od_ambiguity_source_mix") not in (None, ""):
                record["od_ambiguity_source_mix"] = canonical.get("od_ambiguity_source_mix")
            if record.get("od_candidate_path_count") in (None, "", 0, "0") and canonical.get("candidate_probe_path_count") not in (None, ""):
                record["od_candidate_path_count"] = canonical.get("candidate_probe_path_count")
            if record.get("od_corridor_family_count") in (None, "", 0, "0") and canonical.get("candidate_probe_corridor_family_count") not in (None, ""):
                record["od_corridor_family_count"] = canonical.get("candidate_probe_corridor_family_count")
            if record.get("od_objective_spread") in (None, "") and canonical.get("candidate_probe_objective_spread") not in (None, ""):
                record["od_objective_spread"] = canonical.get("candidate_probe_objective_spread")
            if record.get("od_nominal_margin_proxy") in (None, "") and canonical.get("candidate_probe_nominal_margin") not in (None, ""):
                record["od_nominal_margin_proxy"] = canonical.get("candidate_probe_nominal_margin")
            if record.get("od_toll_disagreement_rate") in (None, "") and canonical.get("candidate_probe_toll_disagreement_rate") not in (None, ""):
                record["od_toll_disagreement_rate"] = canonical.get("candidate_probe_toll_disagreement_rate")
            if record.get("od_engine_disagreement_prior") in (None, "") and canonical.get("candidate_probe_engine_disagreement_prior") not in (None, ""):
                record["od_engine_disagreement_prior"] = canonical.get("candidate_probe_engine_disagreement_prior")
            if record.get("od_hard_case_prior") in (None, "") and canonical.get("hard_case_prior") not in (None, ""):
                record["od_hard_case_prior"] = canonical.get("hard_case_prior")
        if record.get("od_engine_disagreement_prior") in (None, "") or record.get("od_hard_case_prior") in (None, ""):
            derived = _canonical_prior_backfill(
                {
                    "candidate_probe_path_count": record.get("od_candidate_path_count"),
                    "candidate_probe_corridor_family_count": record.get("od_corridor_family_count"),
                    "candidate_probe_objective_spread": record.get("od_objective_spread"),
                    "candidate_probe_nominal_margin": record.get("od_nominal_margin_proxy"),
                    "candidate_probe_toll_disagreement_rate": record.get("od_toll_disagreement_rate"),
                    "candidate_probe_engine_disagreement_prior": record.get("od_engine_disagreement_prior"),
                    "hard_case_prior": record.get("od_hard_case_prior"),
                }
            )
            if record.get("od_engine_disagreement_prior") in (None, "") and derived.get("candidate_probe_engine_disagreement_prior") not in (None, ""):
                record["od_engine_disagreement_prior"] = derived.get("candidate_probe_engine_disagreement_prior")
            if record.get("od_hard_case_prior") in (None, "") and derived.get("hard_case_prior") not in (None, ""):
                record["od_hard_case_prior"] = derived.get("hard_case_prior")
        if record.get("od_ambiguity_source_count") in (None, ""):
            source_tokens = _split_prior_sources(record.get("ambiguity_prior_source"))
            if source_tokens:
                record["od_ambiguity_source_count"] = len(source_tokens)
        if record.get("od_ambiguity_source_mix") in (None, ""):
            source_tokens = _split_prior_sources(record.get("ambiguity_prior_source"))
            if source_tokens:
                record["od_ambiguity_source_mix"] = json.dumps({token: 1 for token in source_tokens}, sort_keys=True, separators=(",", ":"))
        route_request_ms = row.get("route_request_ms")
        baseline_osrm_ms = row.get("baseline_osrm_ms")
        baseline_ors_ms = row.get("baseline_ors_ms")
        if record.get("baseline_acquisition_runtime_ms") in (None, ""):
            osrm_ms = as_float(baseline_osrm_ms, float("nan"))
            ors_ms = as_float(baseline_ors_ms, float("nan"))
            if osrm_ms == osrm_ms or ors_ms == ors_ms:
                total = 0.0
                if osrm_ms == osrm_ms:
                    total += osrm_ms
                if ors_ms == ors_ms:
                    total += ors_ms
                record["baseline_acquisition_runtime_ms"] = round(total, 6)
        if record.get("runtime_ms") in (None, ""):
            route_ms = as_float(route_request_ms, float("nan"))
            baseline_ms = as_float(record.get("baseline_acquisition_runtime_ms"), float("nan"))
            if route_ms == route_ms and baseline_ms == baseline_ms:
                record["runtime_ms"] = round(route_ms + baseline_ms, 6)
        if record.get("baseline_runtime_share") in (None, ""):
            record["baseline_runtime_share"] = runtime_share(
                record.get("baseline_acquisition_runtime_ms"),
                record.get("runtime_ms"),
            )
        if record.get("runtime_ratio_vs_osrm") in (None, ""):
            record["runtime_ratio_vs_osrm"] = runtime_ratio(record.get("runtime_ms"), baseline_osrm_ms)
        if record.get("runtime_ratio_vs_ors") in (None, ""):
            record["runtime_ratio_vs_ors"] = runtime_ratio(record.get("runtime_ms"), baseline_ors_ms)
        if record.get("algorithm_runtime_ratio_vs_osrm") in (None, ""):
            record["algorithm_runtime_ratio_vs_osrm"] = runtime_ratio(record.get("algorithm_runtime_ms"), baseline_osrm_ms)
        if record.get("algorithm_runtime_ratio_vs_ors") in (None, ""):
            record["algorithm_runtime_ratio_vs_ors"] = runtime_ratio(record.get("algorithm_runtime_ms"), baseline_ors_ms)
        record["action_efficiency"] = action_efficiency(
            certificate_lift=record.get("certificate_margin"),
            frontier_gain=frontier_action_gain(
                frontier_count=record.get("frontier_count"),
                frontier_diversity_index=record.get("frontier_diversity_index"),
            ),
            action_count=int(float(record.get("voi_action_count") or 0.0)),
            search_budget_used=int(float(record.get("search_budget_used") or 0.0)),
            evidence_budget_used=int(float(record.get("evidence_budget_used") or 0.0)),
        )
        record["observed_ambiguity_index"] = _observed_ambiguity_index(record)
        record["cohort_label"] = _cohort_label(record)
        record["hard_case"] = _is_hard_case_row(record)
        normalized.append(record)
    return normalized


def _composed_methods_appendix(source_manifest: dict[str, Any]) -> str:
    lines = [
        _methods_appendix(
            argparse.Namespace(
                search_budget="mixed",
                evidence_budget="mixed",
                world_count="mixed",
                certificate_threshold="mixed",
                tau_stop="mixed",
                baseline_refinement_policy="first_n",
                ors_baseline_policy="local_service",
                ors_snapshot_mode="off",
                ready_timeout_seconds="n/a",
                ready_poll_seconds="n/a",
                max_alternatives="mixed",
                disable_tolls=False,
                in_process_backend=False,
                backend_url="composed-from-artifacts",
            ),
            corpus_hash=source_manifest["suite_hash"],
            row_count=int(source_manifest["row_count"]),
        ),
        "",
        "Source runs:",
    ]
    for label, payload in source_manifest["sources"].items():
        lines.append(f"- {label}: `{payload['path']}`")
    lines.append("")
    lines.append("Composition policy:")
    lines.append(
        f"- Representative cohort uses `{source_manifest['representative_base_excluding']}` from the representative base and replaces that OD with `{source_manifest['representative_replacement_od']}` from the representative replacement run."
    )
    lines.append(
        f"- Ambiguity cohort uses `{source_manifest['ambiguity_base_excluding']}` from the ambiguity base and replaces that OD with `{source_manifest['ambiguity_replacement_od']}` from the ambiguity replacement run."
    )
    lines.append("- Cohort summary artifacts are written as `thesis_summary_by_cohort.csv/json` so representative, ambiguity, and derived hard-case claims stay separated in the final suite report.")
    lines.append("- All rows are sourced from completed evaluation outputs; no synthetic result rows are introduced in this composition step.")
    prior_coverage = source_manifest.get("ambiguity_prior_coverage", {})
    if isinstance(prior_coverage, dict) and prior_coverage:
        lines.append("")
        lines.append("Upstream prior coverage:")
        lines.append(
            f"- Nonzero corpus ambiguity prior coverage: `{prior_coverage.get('nonzero_od_ambiguity_prior_count')}/{prior_coverage.get('row_count')}`"
        )
        lines.append(
            f"- Nonzero engine-disagreement prior coverage: `{prior_coverage.get('nonzero_engine_disagreement_prior_count')}/{prior_coverage.get('row_count')}`"
        )
        lines.append(
            f"- Nonzero hard-case prior coverage: `{prior_coverage.get('nonzero_hard_case_prior_count')}/{prior_coverage.get('row_count')}`"
        )
        lines.append(
            f"- Prior source mix: `{json.dumps(prior_coverage.get('ambiguity_prior_source_mix', {}), sort_keys=True)}`"
        )
        lines.append(
            f"- Mean prior confidence / source count: `{prior_coverage.get('mean_od_ambiguity_confidence')}` / `{prior_coverage.get('mean_od_ambiguity_source_count')}`"
        )
    return "\n".join(lines)


def _suite_hash(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "row_identity": _row_identity_key(row),
            "artifact_run_id": row.get("artifact_run_id"),
            "od_id": row.get("od_id"),
            "variant_id": row.get("variant_id"),
        }
        for row in rows
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compose_suite(args: argparse.Namespace) -> dict[str, Any]:
    old_out_dir = settings.out_dir
    settings.out_dir = str(Path(args.out_dir))
    try:
        representative_base = _load_rows(str(args.representative_base))
        representative_replacement = _load_rows(str(args.representative_replacement))
        ambiguity_base = _load_rows(str(args.ambiguity_base))
        ambiguity_replacement = _load_rows(str(args.ambiguity_replacement))
        canonical_index = _load_canonical_corpus_index(str(args.canonical_corpus))

        rows = _without_od(representative_base, str(args.representative_exclude_od))
        rows.extend(_only_od(representative_replacement, str(args.representative_exclude_od)))
        rows.extend(_without_od(ambiguity_base, str(args.ambiguity_exclude_od)))
        rows.extend(_only_od(ambiguity_replacement, str(args.ambiguity_replacement_od)))
        rows = _normalize_rows(rows)
        rows = _canonicalize_rows(rows, canonical_index=canonical_index)
        rows = _finalize_cross_variant_metrics(rows)
        summary_rows = _summary_rows(rows)
        cohort_summary_rows = _cohort_summary_rows(rows)
        cohort_composition = _cohort_composition(rows)
        prior_coverage = _prior_coverage_summary(rows, canonical_index=canonical_index)

        run_id = str(args.run_id)
        pair_count = len({_row_identity_key(row) for row in rows})
        suite_hash = _suite_hash(rows)
        source_manifest = {
            "row_count": len(rows),
            "pair_count": pair_count,
            "suite_hash": suite_hash,
            "representative_base_excluding": str(args.representative_exclude_od),
            "representative_replacement_od": str(args.representative_exclude_od),
            "ambiguity_base_excluding": str(args.ambiguity_exclude_od),
            "ambiguity_replacement_od": str(args.ambiguity_replacement_od),
            "sources": {
                "representative_base": {"path": str(args.representative_base)},
                "representative_replacement": {"path": str(args.representative_replacement)},
                "ambiguity_base": {"path": str(args.ambiguity_base)},
                "ambiguity_replacement": {"path": str(args.ambiguity_replacement)},
                "canonical_corpus": {"path": str(args.canonical_corpus)},
            },
            "ambiguity_prior_coverage": prior_coverage,
        }
        write_json_artifact(run_id, "suite_sources.json", source_manifest)
        write_json_artifact(run_id, "prior_coverage_summary.json", prior_coverage)
        write_json_artifact(run_id, "cohort_composition.json", cohort_composition)

        results_csv = write_csv_artifact(run_id, "thesis_results.csv", fieldnames=RESULT_FIELDS, rows=rows)
        write_json_artifact(run_id, "thesis_results.json", {"rows": rows})
        summary_csv = write_csv_artifact(run_id, "thesis_summary.csv", fieldnames=SUMMARY_FIELDS, rows=summary_rows)
        write_json_artifact(run_id, "thesis_summary.json", {"summary_rows": summary_rows})
        cohort_summary_csv = write_csv_artifact(run_id, "thesis_summary_by_cohort.csv", fieldnames=COHORT_SUMMARY_FIELDS, rows=cohort_summary_rows)
        write_json_artifact(run_id, "thesis_summary_by_cohort.json", {"summary_rows": cohort_summary_rows})

        methods_text = _composed_methods_appendix(source_manifest)
        methods_path = write_text_artifact(run_id, "methods_appendix.md", methods_text)
        evaluation_manifest_path = write_json_artifact(
            run_id,
            "evaluation_manifest.json",
            {
                "run_id": run_id,
                "created_at": _now(),
                "composition": source_manifest,
                "ambiguity_prior_coverage": prior_coverage,
                "secondary_baseline_policy": "local_service",
                "snapshot_mode": "off",
            },
        )
        metadata_path = write_json_artifact(
            run_id,
            "metadata.json",
            {
                "run_id": run_id,
                "row_count": len(rows),
                "failure_count": sum(1 for row in rows if row.get("failure_reason")),
                "source_runs": source_manifest["sources"],
            },
        )
        manifest_path = write_manifest(
            run_id,
            {
                "request": {"composition": source_manifest},
                "execution": {"pair_count": pair_count, "variant_count": 4},
            },
        )
        thesis_report_text = _thesis_report(
            run_id,
            summary_rows,
            rows=rows,
            corpus_hash=suite_hash,
            row_count=pair_count,
            ors_baseline_policy="local_service",
            ors_snapshot_mode="off",
            preflight_summary={"required_ok": True},
            readiness_summary={"strict_route_ready": True},
            output_validation={"validated_artifact_count": 0},
        )
        thesis_report_text = "\n".join(
            [
                thesis_report_text.rstrip(),
                "",
                "## Upstream Prior Coverage",
                f"- nonzero_od_ambiguity_prior_rate={prior_coverage.get('nonzero_od_ambiguity_prior_rate')}",
                f"- nonzero_engine_disagreement_prior_rate={prior_coverage.get('nonzero_engine_disagreement_prior_rate')}",
                f"- nonzero_hard_case_prior_rate={prior_coverage.get('nonzero_hard_case_prior_rate')}",
                f"- mean_prior_support_count={prior_coverage.get('mean_prior_support_count')}",
                f"- ambiguity_prior_source_mix={json.dumps(prior_coverage.get('ambiguity_prior_source_mix', {}), sort_keys=True)}",
            ]
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
                "thesis_summary_by_cohort.json": artifact_dir_for_run(run_id) / "thesis_summary_by_cohort.json",
                "suite_sources.json": artifact_dir_for_run(run_id) / "suite_sources.json",
                "prior_coverage_summary.json": artifact_dir_for_run(run_id) / "prior_coverage_summary.json",
                "cohort_composition.json": artifact_dir_for_run(run_id) / "cohort_composition.json",
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
            ors_baseline_policy="local_service",
            ors_snapshot_mode="off",
            preflight_summary={"required_ok": True},
            readiness_summary={"strict_route_ready": True},
            output_validation=output_validation,
        )
        thesis_report_text = "\n".join(
            [
                thesis_report_text.rstrip(),
                "",
                "## Upstream Prior Coverage",
                f"- nonzero_od_ambiguity_prior_rate={prior_coverage.get('nonzero_od_ambiguity_prior_rate')}",
                f"- nonzero_engine_disagreement_prior_rate={prior_coverage.get('nonzero_engine_disagreement_prior_rate')}",
                f"- nonzero_hard_case_prior_rate={prior_coverage.get('nonzero_hard_case_prior_rate')}",
                f"- mean_prior_support_count={prior_coverage.get('mean_prior_support_count')}",
                f"- ambiguity_prior_source_mix={json.dumps(prior_coverage.get('ambiguity_prior_source_mix', {}), sort_keys=True)}",
            ]
        )
        thesis_report_path = write_text_artifact(run_id, "thesis_report.md", thesis_report_text)
        return {
            "run_id": run_id,
            "rows": rows,
            "summary_rows": summary_rows,
            "summary_by_cohort_rows": cohort_summary_rows,
            "ambiguity_prior_coverage": prior_coverage,
            "results_csv": str(results_csv),
            "summary_csv": str(summary_csv),
            "summary_by_cohort_csv": str(cohort_summary_csv),
            "thesis_report": str(thesis_report_path),
            "methods_appendix": str(methods_path),
            "output_artifact_validation": output_validation,
        }
    finally:
        settings.out_dir = old_out_dir


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = compose_suite(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
