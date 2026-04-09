from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .settings import settings


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _oracle_quality_dir() -> Path:
    path = Path(settings.out_dir) / "oracle_quality"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checks_path() -> Path:
    return _oracle_quality_dir() / "checks.ndjson"


def summary_path() -> Path:
    return _oracle_quality_dir() / "summary.json"


def dashboard_csv_path() -> Path:
    return _oracle_quality_dir() / "dashboard.csv"


def replay_oracle_summary_path() -> Path:
    return _oracle_quality_dir() / "replay_oracle_summary.json"


def replay_oracle_dashboard_csv_path() -> Path:
    return _oracle_quality_dir() / "replay_oracle_dashboard.csv"


def append_check_record(record: dict[str, Any]) -> Path:
    path = checks_path()
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(json.dumps(record, separators=(",", ":")))
        f.write("\n")
    return path


def load_check_records() -> list[dict[str, Any]]:
    path = checks_path()
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _source_sort_key(payload: dict[str, Any]) -> tuple[float, str]:
    last_seen = str(payload.get("last_observed_at_utc") or payload.get("source") or "")
    return (0.0 if last_seen else 1.0, str(payload.get("source", "")))


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_source(record: dict[str, Any]) -> dict[str, Any]:
    if isinstance(record.get("replay_oracle_summary"), dict):
        return dict(record["replay_oracle_summary"])
    return record


def _normalize_bool_token(value: Any | None) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "supported"}:
        return "true"
    if text in {"0", "false", "no", "n", "unsupported"}:
        return "false"
    return "unknown"


def _normalize_token(value: Any | None, *, default: str = "unknown") -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text or default


def _mode_count(records: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter = Counter()
    for record in records:
        source = _metadata_source(record)
        token = _normalize_token(source.get(field_name))
        counter[token] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def compute_dashboard_payload(
    records: list[dict[str, Any]],
    *,
    stale_threshold_s: float = 900.0,
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        source = str(record.get("source", "")).strip()
        if not source:
            continue
        by_source.setdefault(source, []).append(record)

    sources: list[dict[str, Any]] = []
    for source, items in by_source.items():
        total = len(items)
        passed = sum(1 for item in items if bool(item.get("passed")))
        schema_failures = sum(1 for item in items if not bool(item.get("schema_valid", False)))
        signature_failures = sum(
            1
            for item in items
            if item.get("signature_valid") is False
        )
        stale_count = sum(
            1
            for item in items
            if (_to_float(item.get("freshness_s")) or 0.0) > stale_threshold_s
        )
        latencies = [
            float(item["latency_ms"])
            for item in items
            if _to_float(item.get("latency_ms")) is not None
        ]
        avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else None
        observed_times = [str(item.get("observed_at_utc", "")) for item in items if item.get("observed_at_utc")]
        last_observed = max(observed_times) if observed_times else None

        sources.append(
            {
                "source": source,
                "check_count": total,
                "pass_rate": round((passed / total) if total else 0.0, 6),
                "schema_failures": schema_failures,
                "signature_failures": signature_failures,
                "stale_count": stale_count,
                "avg_latency_ms": round(avg_latency_ms, 3) if avg_latency_ms is not None else None,
                "last_observed_at_utc": last_observed,
            }
        )

    sources.sort(key=_source_sort_key)
    return {
        "total_checks": len(records),
        "source_count": len(sources),
        "stale_threshold_s": stale_threshold_s,
        "sources": sources,
        "updated_at_utc": _utc_now_iso(),
    }


def compute_replay_oracle_payload(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    sources: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        source = str(record.get("source", "")).strip()
        if not source:
            continue
        sources.setdefault(source, []).append(record)

    source_rows: list[dict[str, Any]] = []
    for source, items in sources.items():
        replay_rows = [_metadata_source(item) for item in items]
        replay_regrets = [
            _to_float(item.get("replay_regret"))
            for item in replay_rows
            if _to_float(item.get("replay_regret")) is not None
        ]
        predicted_certificate_lifts = [
            _to_float(item.get("predicted_certificate_lift"))
            for item in replay_rows
            if _to_float(item.get("predicted_certificate_lift")) is not None
        ]
        realized_certificate_lifts = [
            _to_float(item.get("realized_certificate_lift"))
            for item in replay_rows
            if _to_float(item.get("realized_certificate_lift")) is not None
        ]
        predicted_gap_lifts = [
            _to_float(item.get("predicted_gap_lift"))
            for item in replay_rows
            if _to_float(item.get("predicted_gap_lift")) is not None
        ]
        realized_gap_lifts = [
            _to_float(item.get("realized_gap_lift"))
            for item in replay_rows
            if _to_float(item.get("realized_gap_lift")) is not None
        ]
        support_true = sum(1 for item in replay_rows if _normalize_bool_token(item.get("support_flag")) == "true")
        support_false = sum(1 for item in replay_rows if _normalize_bool_token(item.get("support_flag")) == "false")
        source_rows.append(
            {
                "source": source,
                "row_count": len(items),
                "support_true_count": support_true,
                "support_false_count": support_false,
                "support_unknown_count": len(items) - support_true - support_false,
                "terminal_type_counts": _mode_count(items, "terminal_type"),
                "fidelity_class_counts": _mode_count(items, "fidelity_class"),
                "support_status_counts": _mode_count(items, "support_status"),
                "mean_replay_regret": round(sum(replay_regrets) / len(replay_regrets), 6) if replay_regrets else None,
                "mean_predicted_certificate_lift": round(sum(predicted_certificate_lifts) / len(predicted_certificate_lifts), 6)
                if predicted_certificate_lifts
                else None,
                "mean_realized_certificate_lift": round(sum(realized_certificate_lifts) / len(realized_certificate_lifts), 6)
                if realized_certificate_lifts
                else None,
                "mean_predicted_gap_lift": round(sum(predicted_gap_lifts) / len(predicted_gap_lifts), 6)
                if predicted_gap_lifts
                else None,
                "mean_realized_gap_lift": round(sum(realized_gap_lifts) / len(realized_gap_lifts), 6)
                if realized_gap_lifts
                else None,
            }
        )

    source_rows.sort(key=_source_sort_key)
    return {
        "total_records": len(records),
        "source_count": len(source_rows),
        "sources": source_rows,
        "updated_at_utc": _utc_now_iso(),
    }


def write_summary_artifacts(payload: dict[str, Any]) -> tuple[Path, Path]:
    summary = summary_path()
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = dashboard_csv_path()
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source",
            "check_count",
            "pass_rate",
            "schema_failures",
            "signature_failures",
            "stale_count",
            "avg_latency_ms",
            "last_observed_at_utc",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in payload.get("sources", []):
            writer.writerow({key: source.get(key) for key in fieldnames})
    return summary, csv_path


def write_replay_oracle_artifacts(payload: dict[str, Any]) -> tuple[Path, Path]:
    summary = replay_oracle_summary_path()
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = replay_oracle_dashboard_csv_path()
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source",
            "row_count",
            "support_true_count",
            "support_false_count",
            "support_unknown_count",
            "mean_replay_regret",
            "mean_predicted_certificate_lift",
            "mean_realized_certificate_lift",
            "mean_predicted_gap_lift",
            "mean_realized_gap_lift",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in payload.get("sources", []):
            row = {key: source.get(key) for key in fieldnames}
            writer.writerow(row)
    return summary, csv_path
