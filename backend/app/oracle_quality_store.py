from __future__ import annotations

import csv
import json
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
