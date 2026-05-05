from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_COMPARE_FIELDS = [
    "pipeline_mode",
    "route_id",
    "failure_reason",
    "artifact_complete",
    "route_evidence_ok",
    "selected_distance_km",
    "selected_duration_s",
    "selected_monetary_cost",
    "selected_emissions_kg",
    "certificate",
    "certified",
    "certificate_margin",
    "certificate_runner_up_gap",
    "frontier_count",
    "frontier_hypervolume",
    "frontier_diversity_index",
    "frontier_entropy",
    "weighted_win_osrm",
    "weighted_win_ors",
    "weighted_win_v0",
    "balanced_win_osrm",
    "balanced_win_ors",
    "balanced_win_v0",
    "dominates_osrm",
    "dominates_ors",
    "dominates_v0",
    "weighted_margin_vs_osrm",
    "weighted_margin_vs_ors",
    "weighted_margin_vs_v0",
    "weighted_margin_vs_best_baseline",
    "runtime_ms",
    "algorithm_runtime_ms",
    "selected_candidate_source_label",
    "selected_final_route_source_label",
    "preemptive_comparator_seeded",
    "selected_from_preemptive_comparator_seed",
    "selected_from_supplemental_rescue",
    "selected_from_comparator_engine",
]

NUMERIC_COMPARE_FIELDS = {
    "selected_distance_km",
    "selected_duration_s",
    "selected_monetary_cost",
    "selected_emissions_kg",
    "certificate",
    "certificate_margin",
    "certificate_runner_up_gap",
    "frontier_count",
    "frontier_hypervolume",
    "frontier_diversity_index",
    "frontier_entropy",
    "weighted_margin_vs_osrm",
    "weighted_margin_vs_ors",
    "weighted_margin_vs_v0",
    "weighted_margin_vs_best_baseline",
    "runtime_ms",
    "algorithm_runtime_ms",
}

DEFAULT_SUMMARY_FIELDS = [
    "success_rate",
    "artifact_complete_rate",
    "route_evidence_ok_rate",
    "weighted_win_rate_osrm",
    "weighted_win_rate_ors",
    "weighted_win_rate_v0",
    "balanced_win_rate_osrm",
    "balanced_win_rate_ors",
    "balanced_win_rate_v0",
    "dominance_win_rate_osrm",
    "dominance_win_rate_ors",
    "dominance_win_rate_v0",
    "mean_certificate",
    "mean_frontier_hypervolume",
    "mean_frontier_count",
    "mean_runtime_ms",
    "mean_algorithm_runtime_ms",
]

FALLBACK_OD_FIELDS = [
    "origin_lat",
    "origin_lon",
    "destination_lat",
    "destination_lon",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two thesis_results.csv files and emit an OD/variant row diff CSV."
    )
    parser.add_argument("--before-results-csv", required=True)
    parser.add_argument("--after-results-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--before-summary-csv")
    parser.add_argument("--after-summary-csv")
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary diff JSON path. Defaults to <out-csv>.summary.json when both summary CSVs are supplied.",
    )
    parser.add_argument(
        "--field",
        action="append",
        dest="extra_fields",
        default=[],
        help="Additional thesis_results.csv fields to compare.",
    )
    return parser


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_float(value: Any) -> float | None:
    text = _normalize_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _canonical_value(field: str, value: Any) -> Any:
    if field in NUMERIC_COMPARE_FIELDS:
        number = _parse_float(value)
        if number is not None:
            return number
    text = _normalize_text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered
    return text


def _parse_bool(value: Any) -> bool | None:
    text = _normalize_text(value).lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _format_delta(before_value: Any, after_value: Any) -> str:
    before_number = _parse_float(before_value)
    after_number = _parse_float(after_value)
    if before_number is None or after_number is None:
        return ""
    delta = after_number - before_number
    return f"{delta:.12g}"


def _unique_fields(extra_fields: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for field in DEFAULT_COMPARE_FIELDS + [_normalize_text(item) for item in extra_fields]:
        if field and field not in seen:
            seen.add(field)
            ordered.append(field)
    return ordered


def _od_display_key(row: dict[str, Any]) -> str:
    od_id = _normalize_text(row.get("od_id"))
    if od_id:
        return od_id
    parts = [_normalize_text(row.get(name)) for name in FALLBACK_OD_FIELDS]
    if any(parts):
        return f"{parts[0]},{parts[1]}->{parts[2]},{parts[3]}"
    return "unknown_od"


def _row_key(row: dict[str, Any]) -> str:
    variant_id = _normalize_text(row.get("variant_id")) or "unknown_variant"
    return f"{_od_display_key(row)}|{variant_id}"


def _rows_by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_row_key(row): row for row in rows}


def _row_is_successful(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    if _normalize_text(row.get("failure_reason")):
        return False
    artifact_complete = _parse_bool(row.get("artifact_complete"))
    if artifact_complete is False:
        return False
    route_evidence_ok = _parse_bool(row.get("route_evidence_ok"))
    if route_evidence_ok is False:
        return False
    return True


def _retained_success_summary(
    *,
    before_by_key: dict[str, dict[str, Any]],
    after_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    all_keys = sorted(set(before_by_key) | set(after_by_key))
    overall = {
        "before_success_row_count": 0,
        "after_success_row_count": 0,
        "retained_success_row_count": 0,
        "regressed_success_row_count": 0,
        "newly_successful_row_count": 0,
        "removed_success_row_count": 0,
        "retained_success_row_keys": [],
        "regressed_success_row_keys": [],
        "newly_successful_row_keys": [],
        "removed_success_row_keys": [],
    }
    variant_summaries: dict[str, dict[str, Any]] = {}

    def _variant_summary(variant_id: str) -> dict[str, Any]:
        summary = variant_summaries.get(variant_id)
        if summary is None:
            summary = {
                "before_success_row_count": 0,
                "after_success_row_count": 0,
                "retained_success_row_count": 0,
                "regressed_success_row_count": 0,
                "newly_successful_row_count": 0,
                "removed_success_row_count": 0,
                "retained_success_row_keys": [],
                "regressed_success_row_keys": [],
                "newly_successful_row_keys": [],
                "removed_success_row_keys": [],
            }
            variant_summaries[variant_id] = summary
        return summary

    for key in all_keys:
        before_row = before_by_key.get(key)
        after_row = after_by_key.get(key)
        template = after_row or before_row or {}
        variant_id = _normalize_text(template.get("variant_id")) or "unknown_variant"
        summary = _variant_summary(variant_id)
        before_success = _row_is_successful(before_row)
        after_success = _row_is_successful(after_row)

        if before_success:
            overall["before_success_row_count"] += 1
            summary["before_success_row_count"] += 1
        if after_success:
            overall["after_success_row_count"] += 1
            summary["after_success_row_count"] += 1
        if before_success and after_row is not None and after_success:
            overall["retained_success_row_count"] += 1
            overall["retained_success_row_keys"].append(key)
            summary["retained_success_row_count"] += 1
            summary["retained_success_row_keys"].append(key)
        elif before_success and after_row is not None and not after_success:
            overall["regressed_success_row_count"] += 1
            overall["regressed_success_row_keys"].append(key)
            summary["regressed_success_row_count"] += 1
            summary["regressed_success_row_keys"].append(key)
        elif before_success and after_row is None:
            overall["removed_success_row_count"] += 1
            overall["removed_success_row_keys"].append(key)
            summary["removed_success_row_count"] += 1
            summary["removed_success_row_keys"].append(key)
        elif not before_success and after_success:
            overall["newly_successful_row_count"] += 1
            overall["newly_successful_row_keys"].append(key)
            summary["newly_successful_row_count"] += 1
            summary["newly_successful_row_keys"].append(key)

    return {
        **overall,
        "variant_summaries": variant_summaries,
    }


def _available_compare_fields(
    fields: list[str],
    *,
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
) -> list[str]:
    available = {
        key
        for row in before_rows + after_rows
        for key in row.keys()
        if _normalize_text(key)
    }
    return [field for field in fields if field in available]


def _compare_row(
    key: str,
    *,
    before_row: dict[str, Any] | None,
    after_row: dict[str, Any] | None,
    compare_fields: list[str],
) -> dict[str, Any]:
    status = "unchanged"
    if before_row is None:
        status = "added"
    elif after_row is None:
        status = "removed"

    template = after_row or before_row or {}
    diff_row: dict[str, Any] = {
        "row_key": key,
        "od_id": _normalize_text(template.get("od_id")),
        "od_display_key": _od_display_key(template),
        "variant_id": _normalize_text(template.get("variant_id")),
        "status": status,
    }

    changed_fields: list[str] = []
    for field in compare_fields:
        before_value = _normalize_text((before_row or {}).get(field))
        after_value = _normalize_text((after_row or {}).get(field))
        diff_row[f"before_{field}"] = before_value
        diff_row[f"after_{field}"] = after_value
        if field in NUMERIC_COMPARE_FIELDS:
            diff_row[f"delta_{field}"] = _format_delta(before_value, after_value)
        if _canonical_value(field, before_value) != _canonical_value(field, after_value):
            changed_fields.append(field)

    if status == "unchanged" and changed_fields:
        status = "changed"
    diff_row["status"] = status
    diff_row["changed_field_count"] = len(changed_fields)
    diff_row["changed_fields"] = ";".join(changed_fields)
    return diff_row


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_rows_by_variant(path: Path) -> dict[str, dict[str, str]]:
    return {
        _normalize_text(row.get("variant_id")): row
        for row in _read_csv_rows(path)
        if _normalize_text(row.get("variant_id"))
    }


def _summary_diff(
    *,
    before_summary_csv: Path | None,
    after_summary_csv: Path | None,
    before_by_key: dict[str, dict[str, Any]],
    after_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if before_summary_csv is None or after_summary_csv is None:
        return None
    before_rows = _summary_rows_by_variant(before_summary_csv)
    after_rows = _summary_rows_by_variant(after_summary_csv)
    variants = sorted(set(before_rows) | set(after_rows))
    variant_diffs: dict[str, Any] = {}
    for variant_id in variants:
        before_row = before_rows.get(variant_id, {})
        after_row = after_rows.get(variant_id, {})
        changed_fields: list[str] = []
        before_payload: dict[str, Any] = {}
        after_payload: dict[str, Any] = {}
        delta_payload: dict[str, Any] = {}
        for field in DEFAULT_SUMMARY_FIELDS:
            before_value = _normalize_text(before_row.get(field))
            after_value = _normalize_text(after_row.get(field))
            if not before_value and not after_value:
                continue
            before_payload[field] = before_value
            after_payload[field] = after_value
            if _canonical_value(field, before_value) != _canonical_value(field, after_value):
                changed_fields.append(field)
            delta = _format_delta(before_value, after_value)
            if delta:
                delta_payload[field] = delta
        variant_diffs[variant_id] = {
            "status": (
                "added"
                if variant_id not in before_rows
                else "removed"
                if variant_id not in after_rows
                else "changed"
                if changed_fields
                else "unchanged"
            ),
            "changed_fields": changed_fields,
            "before": before_payload,
            "after": after_payload,
            "delta": delta_payload,
        }
    return {
        "before_summary_csv": str(before_summary_csv),
        "after_summary_csv": str(after_summary_csv),
        "variant_count": len(variants),
        "variant_diffs": variant_diffs,
        "retained_success_summary": _retained_success_summary(
            before_by_key=before_by_key,
            after_by_key=after_by_key,
        ),
    }


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    before_results_csv = Path(args.before_results_csv)
    after_results_csv = Path(args.after_results_csv)
    out_csv = Path(args.out_csv)
    before_rows = _read_csv_rows(before_results_csv)
    after_rows = _read_csv_rows(after_results_csv)
    compare_fields = _available_compare_fields(
        _unique_fields(list(args.extra_fields or [])),
        before_rows=before_rows,
        after_rows=after_rows,
    )
    before_by_key = _rows_by_key(before_rows)
    after_by_key = _rows_by_key(after_rows)
    all_keys = sorted(set(before_by_key) | set(after_by_key))
    diff_rows = [
        _compare_row(
            key,
            before_row=before_by_key.get(key),
            after_row=after_by_key.get(key),
            compare_fields=compare_fields,
        )
        for key in all_keys
    ]

    fieldnames = [
        "row_key",
        "od_id",
        "od_display_key",
        "variant_id",
        "status",
        "changed_field_count",
        "changed_fields",
    ]
    for field in compare_fields:
        fieldnames.append(f"before_{field}")
        fieldnames.append(f"after_{field}")
        if field in NUMERIC_COMPARE_FIELDS:
            fieldnames.append(f"delta_{field}")
    _write_csv(out_csv, diff_rows, fieldnames=fieldnames)

    summary_payload = _summary_diff(
        before_summary_csv=Path(args.before_summary_csv) if args.before_summary_csv else None,
        after_summary_csv=Path(args.after_summary_csv) if args.after_summary_csv else None,
        before_by_key=before_by_key,
        after_by_key=after_by_key,
    )
    summary_json_path: Path | None = None
    if summary_payload is not None:
        summary_json_path = (
            Path(args.summary_json)
            if _normalize_text(args.summary_json)
            else out_csv.with_suffix(".summary.json")
        )
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    changed_rows = sum(1 for row in diff_rows if row["status"] == "changed")
    added_rows = sum(1 for row in diff_rows if row["status"] == "added")
    removed_rows = sum(1 for row in diff_rows if row["status"] == "removed")
    unchanged_rows = sum(1 for row in diff_rows if row["status"] == "unchanged")
    return {
        "before_results_csv": str(before_results_csv),
        "after_results_csv": str(after_results_csv),
        "out_csv": str(out_csv),
        "summary_json": str(summary_json_path) if summary_json_path is not None else None,
        "row_count": len(diff_rows),
        "changed_row_count": changed_rows,
        "added_row_count": added_rows,
        "removed_row_count": removed_rows,
        "unchanged_row_count": unchanged_rows,
        "compare_fields": compare_fields,
        "retained_success_summary": _retained_success_summary(
            before_by_key=before_by_key,
            after_by_key=after_by_key,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_comparison(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
