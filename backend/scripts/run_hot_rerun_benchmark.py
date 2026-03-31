from __future__ import annotations

import argparse
import copy
import json
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.run_store import artifact_dir_for_run, write_csv_artifact, write_json_artifact, write_manifest, write_text_artifact
from app.settings import settings
from scripts.run_thesis_evaluation import VARIANTS, _build_parser as _build_eval_parser, run_thesis_evaluation

HOT_ROUTE_REUSE_VARIANTS: tuple[str, ...] = ("A", "B", "C")
HOT_REFC_REUSE_VARIANTS: tuple[str, ...] = ("B", "C")
HOT_RUNTIME_IMPROVEMENT_VARIANTS: tuple[str, ...] = ("A", "B", "C")
THESIS_COLD_CACHE_SCOPE = "thesis_cold"
HOT_GATE_THRESHOLDS: dict[str, tuple[float, tuple[str, ...]]] = {
    "mean_route_cache_hit_rate": (0.50, HOT_ROUTE_REUSE_VARIANTS),
    "mean_option_build_cache_hit_rate": (0.70, HOT_ROUTE_REUSE_VARIANTS),
    "mean_option_build_reuse_rate": (0.70, HOT_ROUTE_REUSE_VARIANTS),
    "mean_refc_world_reuse_rate": (0.80, HOT_REFC_REUSE_VARIANTS),
}
HOT_RUNTIME_RATIO_METRICS: tuple[str, ...] = ("mean_runtime_ratio_vs_osrm", "mean_runtime_ratio_vs_ors")
HOT_COMPARISON_FIELDS: list[str] = [
    "variant_id",
    "pipeline_mode",
    "cold_mean_algorithm_runtime_ms",
    "hot_mean_algorithm_runtime_ms",
    "algorithm_runtime_delta_ms",
    "cold_mean_route_cache_hit_rate",
    "hot_mean_route_cache_hit_rate",
    "route_cache_hit_rate_delta",
    "cold_mean_option_build_cache_hit_rate",
    "hot_mean_option_build_cache_hit_rate",
    "option_build_cache_hit_rate_delta",
    "cold_mean_option_build_reuse_rate",
    "hot_mean_option_build_reuse_rate",
    "option_build_reuse_rate_delta",
    "cold_mean_refc_world_reuse_rate",
    "hot_mean_refc_world_reuse_rate",
    "refc_world_reuse_rate_delta",
    "cold_mean_runtime_ratio_vs_osrm",
    "hot_mean_runtime_ratio_vs_osrm",
    "runtime_ratio_vs_osrm_delta",
    "runtime_ratio_vs_osrm_improved",
    "cold_mean_runtime_ratio_vs_ors",
    "hot_mean_runtime_ratio_vs_ors",
    "runtime_ratio_vs_ors_delta",
    "runtime_ratio_vs_ors_improved",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run_label() -> str:
    return datetime.now(UTC).strftime("hot_rerun_%Y%m%d_%H%M%S")


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_eval_parser()
    parser.description = "Run a cold thesis evaluation followed by a true hot rerun on the same backend instance."
    parser.add_argument("--pair-run-id", default=None)
    parser.add_argument("--cold-run-id", default=None)
    parser.add_argument("--hot-run-id", default=None)
    return parser


def _client_response_json(response: Any) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("backend_response_not_object")
    return payload


def _clear_backend_caches(client: Any, *, scope: str = THESIS_COLD_CACHE_SCOPE) -> dict[str, Any]:
    response = client.delete(f"/cache?scope={scope}")
    if int(getattr(response, "status_code", 500)) >= 400:
        raise RuntimeError(f"cache_clear_failed:{getattr(response, 'status_code', 'unknown')}")
    return _client_response_json(response)


def _cache_stats(client: Any) -> dict[str, Any]:
    response = client.get("/cache/stats")
    if int(getattr(response, "status_code", 500)) >= 400:
        raise RuntimeError(f"cache_stats_failed:{getattr(response, 'status_code', 'unknown')}")
    return _client_response_json(response)


def _restore_hot_rerun_route_cache(client: Any) -> dict[str, Any]:
    response = client.post("/cache/hot-rerun/restore")
    if int(getattr(response, "status_code", 500)) >= 400:
        raise RuntimeError(
            f"hot_rerun_route_cache_restore_failed:{getattr(response, 'status_code', 'unknown')}"
        )
    return _client_response_json(response)


def _summary_row_map(summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        variant_id = str(row.get("variant_id") or "").strip()
        if variant_id:
            mapped[variant_id] = dict(row)
    return mapped


def _comparison_row(cold_row: Mapping[str, Any] | None, hot_row: Mapping[str, Any] | None) -> dict[str, Any]:
    cold_row = dict(cold_row or {})
    hot_row = dict(hot_row or {})
    variant_id = str(hot_row.get("variant_id") or cold_row.get("variant_id") or "")
    pipeline_mode = str(hot_row.get("pipeline_mode") or cold_row.get("pipeline_mode") or "")

    def metric_delta(metric: str) -> tuple[float | None, float | None, float | None]:
        cold_value = _as_float(cold_row.get(metric))
        hot_value = _as_float(hot_row.get(metric))
        if cold_value is None or hot_value is None:
            return cold_value, hot_value, None
        return cold_value, hot_value, round(hot_value - cold_value, 6)

    cold_runtime_osrm, hot_runtime_osrm, delta_runtime_osrm = metric_delta("mean_runtime_ratio_vs_osrm")
    cold_runtime_ors, hot_runtime_ors, delta_runtime_ors = metric_delta("mean_runtime_ratio_vs_ors")
    cold_runtime_ms, hot_runtime_ms, delta_runtime_ms = metric_delta("mean_algorithm_runtime_ms")
    cold_route_cache, hot_route_cache, delta_route_cache = metric_delta("mean_route_cache_hit_rate")
    cold_option_hit, hot_option_hit, delta_option_hit = metric_delta("mean_option_build_cache_hit_rate")
    cold_option_reuse, hot_option_reuse, delta_option_reuse = metric_delta("mean_option_build_reuse_rate")
    cold_world_reuse, hot_world_reuse, delta_world_reuse = metric_delta("mean_refc_world_reuse_rate")

    return {
        "variant_id": variant_id,
        "pipeline_mode": pipeline_mode,
        "cold_mean_algorithm_runtime_ms": cold_runtime_ms,
        "hot_mean_algorithm_runtime_ms": hot_runtime_ms,
        "algorithm_runtime_delta_ms": delta_runtime_ms,
        "cold_mean_route_cache_hit_rate": cold_route_cache,
        "hot_mean_route_cache_hit_rate": hot_route_cache,
        "route_cache_hit_rate_delta": delta_route_cache,
        "cold_mean_option_build_cache_hit_rate": cold_option_hit,
        "hot_mean_option_build_cache_hit_rate": hot_option_hit,
        "option_build_cache_hit_rate_delta": delta_option_hit,
        "cold_mean_option_build_reuse_rate": cold_option_reuse,
        "hot_mean_option_build_reuse_rate": hot_option_reuse,
        "option_build_reuse_rate_delta": delta_option_reuse,
        "cold_mean_refc_world_reuse_rate": cold_world_reuse,
        "hot_mean_refc_world_reuse_rate": hot_world_reuse,
        "refc_world_reuse_rate_delta": delta_world_reuse,
        "cold_mean_runtime_ratio_vs_osrm": cold_runtime_osrm,
        "hot_mean_runtime_ratio_vs_osrm": hot_runtime_osrm,
        "runtime_ratio_vs_osrm_delta": delta_runtime_osrm,
        "runtime_ratio_vs_osrm_improved": (
            delta_runtime_osrm is not None and hot_runtime_osrm is not None and cold_runtime_osrm is not None and hot_runtime_osrm < cold_runtime_osrm
        ),
        "cold_mean_runtime_ratio_vs_ors": cold_runtime_ors,
        "hot_mean_runtime_ratio_vs_ors": hot_runtime_ors,
        "runtime_ratio_vs_ors_delta": delta_runtime_ors,
        "runtime_ratio_vs_ors_improved": (
            delta_runtime_ors is not None and hot_runtime_ors is not None and cold_runtime_ors is not None and hot_runtime_ors < cold_runtime_ors
        ),
    }


def build_hot_rerun_comparison(
    *,
    pair_run_id: str,
    cold_run_id: str,
    hot_run_id: str,
    cold_summary_rows: Sequence[Mapping[str, Any]],
    hot_summary_rows: Sequence[Mapping[str, Any]],
    cache_stats: Mapping[str, Any],
) -> dict[str, Any]:
    cold_by_variant = _summary_row_map(cold_summary_rows)
    hot_by_variant = _summary_row_map(hot_summary_rows)
    comparison_rows = [
        _comparison_row(cold_by_variant.get(variant_id), hot_by_variant.get(variant_id))
        for variant_id in VARIANTS
        if variant_id in cold_by_variant or variant_id in hot_by_variant
    ]

    metric_checks: list[dict[str, Any]] = []
    for metric, (threshold, variants) in HOT_GATE_THRESHOLDS.items():
        for variant_id in variants:
            hot_row = hot_by_variant.get(variant_id, {})
            value = _as_float(hot_row.get(metric))
            metric_checks.append(
                {
                    "metric": metric,
                    "variant_id": variant_id,
                    "threshold": threshold,
                    "value": value,
                    "pass": value is not None and value >= threshold - 1e-9,
                }
            )

    runtime_improvement_checks: list[dict[str, Any]] = []
    for metric in HOT_RUNTIME_RATIO_METRICS:
        for variant_id in HOT_RUNTIME_IMPROVEMENT_VARIANTS:
            cold_value = _as_float((cold_by_variant.get(variant_id) or {}).get(metric))
            hot_value = _as_float((hot_by_variant.get(variant_id) or {}).get(metric))
            runtime_improvement_checks.append(
                {
                    "metric": metric,
                    "variant_id": variant_id,
                    "cold_value": cold_value,
                    "hot_value": hot_value,
                    "pass": (
                        cold_value is not None
                        and hot_value is not None
                        and hot_value < cold_value - 1e-9
                    ),
                }
            )

    hot_gate = {
        "pair_run_id": pair_run_id,
        "cold_run_id": cold_run_id,
        "hot_run_id": hot_run_id,
        "gate_scope": {
            "route_reuse_variants": list(HOT_ROUTE_REUSE_VARIANTS),
            "refc_reuse_variants": list(HOT_REFC_REUSE_VARIANTS),
            "runtime_improvement_variants": list(HOT_RUNTIME_IMPROVEMENT_VARIANTS),
        },
        "metric_checks": metric_checks,
        "runtime_improvement_checks": runtime_improvement_checks,
        "all_green": all(check["pass"] for check in [*metric_checks, *runtime_improvement_checks]),
    }
    return {
        "pair_run_id": pair_run_id,
        "created_at": _now(),
        "cold_run_id": cold_run_id,
        "hot_run_id": hot_run_id,
        "cache_stats": dict(cache_stats),
        "cold_summary_rows": [dict(row) for row in cold_summary_rows],
        "hot_summary_rows": [dict(row) for row in hot_summary_rows],
        "comparison_rows": comparison_rows,
        "hot_gate": hot_gate,
    }


def _hot_rerun_report(comparison: Mapping[str, Any]) -> str:
    lines = [
        f"# Hot Rerun Benchmark `{comparison['pair_run_id']}`",
        "",
        f"- cold_run_id={comparison['cold_run_id']}",
        f"- hot_run_id={comparison['hot_run_id']}",
        f"- gate_all_green={comparison['hot_gate']['all_green']}",
        "",
        "## Cache Stats",
        "",
    ]
    cache_stats = comparison.get("cache_stats", {})
    for label in ("before_clear", "after_clear", "after_cold", "restore_response", "after_restore", "after_hot"):
        lines.append(f"- {label}={json.dumps(cache_stats.get(label, {}), sort_keys=True)}")
    lines.append("")
    lines.append("## Variant Comparison")
    lines.append("")
    for row in comparison.get("comparison_rows", []):
        lines.append(
            "- "
            f"{row.get('variant_id')} ({row.get('pipeline_mode')}): "
            f"route_cache={row.get('hot_mean_route_cache_hit_rate')} "
            f"(cold {row.get('cold_mean_route_cache_hit_rate')}), "
            f"option_cache={row.get('hot_mean_option_build_cache_hit_rate')} "
            f"(cold {row.get('cold_mean_option_build_cache_hit_rate')}), "
            f"option_reuse={row.get('hot_mean_option_build_reuse_rate')} "
            f"(cold {row.get('cold_mean_option_build_reuse_rate')}), "
            f"refc_world_reuse={row.get('hot_mean_refc_world_reuse_rate')} "
            f"(cold {row.get('cold_mean_refc_world_reuse_rate')}), "
            f"runtime_ratio_vs_osrm={row.get('hot_mean_runtime_ratio_vs_osrm')} "
            f"(cold {row.get('cold_mean_runtime_ratio_vs_osrm')}), "
            f"runtime_ratio_vs_ors={row.get('hot_mean_runtime_ratio_vs_ors')} "
            f"(cold {row.get('cold_mean_runtime_ratio_vs_ors')})"
        )
    lines.append("")
    lines.append("## Gate Checks")
    lines.append("")
    for check in comparison["hot_gate"]["metric_checks"]:
        lines.append(
            f"- {check['metric']} / {check['variant_id']}: value={check['value']} "
            f"threshold={check['threshold']} pass={check['pass']}"
        )
    for check in comparison["hot_gate"]["runtime_improvement_checks"]:
        lines.append(
            f"- {check['metric']} / {check['variant_id']}: cold={check['cold_value']} "
            f"hot={check['hot_value']} pass={check['pass']}"
        )
    return "\n".join(lines)


def _clone_args(
    args: argparse.Namespace,
    *,
    run_id: str,
    cache_mode: str | None = None,
    cold_cache_scope: str | None = None,
    evaluation_suite_role: str | None = None,
) -> argparse.Namespace:
    cloned = argparse.Namespace(**copy.deepcopy(vars(args)))
    cloned.run_id = run_id
    if cache_mode is not None:
        cloned.cache_mode = cache_mode
    if cold_cache_scope is not None:
        cloned.cold_cache_scope = cold_cache_scope
    if evaluation_suite_role is not None:
        cloned.evaluation_suite_role = evaluation_suite_role
    return cloned


def _pair_run_ids(args: argparse.Namespace) -> tuple[str, str, str]:
    pair_run_id = str(args.pair_run_id or args.run_id or _run_label())
    cold_run_id = str(args.cold_run_id or f"{pair_run_id}_cold")
    hot_run_id = str(args.hot_run_id or f"{pair_run_id}_hot")
    return pair_run_id, cold_run_id, hot_run_id


def _update_json_file(path: Path, updates: Mapping[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} is not a JSON object")
    payload.update(dict(updates))
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _annotate_run_artifact(
    *,
    run_id: str,
    benchmark_phase: str,
    pair_run_id: str,
    paired_run_id: str,
    cache_reset_before_run: bool,
    cache_stats_before_run: Mapping[str, Any],
    cache_stats_after_run: Mapping[str, Any],
    comparison_artifact_name: str | None = None,
) -> None:
    updates: dict[str, Any] = {
        "benchmark_kind": "hot_rerun_benchmark",
        "benchmark_phase": benchmark_phase,
        "pair_run_id": pair_run_id,
        "paired_run_id": paired_run_id,
        "cache_reset_before_run": bool(cache_reset_before_run),
        "cache_stats_before_run": dict(cache_stats_before_run),
        "cache_stats_after_run": dict(cache_stats_after_run),
        "cache_carryover_expected": benchmark_phase == "hot_rerun",
    }
    if comparison_artifact_name:
        updates["hot_rerun_comparison_artifact"] = comparison_artifact_name
    artifact_dir = artifact_dir_for_run(run_id)
    for name in ("metadata.json", "evaluation_manifest.json"):
        path = artifact_dir / name
        if path.exists():
            _update_json_file(path, updates)


def run_hot_rerun_benchmark(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    own_client = client is None
    pair_run_id, cold_run_id, hot_run_id = _pair_run_ids(args)
    old_out_dir = settings.out_dir
    settings.out_dir = str(Path(args.out_dir))
    try:
        with ExitStack() as stack:
            if client is not None:
                active_client = client
            elif bool(getattr(args, "in_process_backend", False)):
                from app.main import app

                active_client = stack.enter_context(TestClient(app))
            else:
                active_client = stack.enter_context(
                    httpx.Client(base_url=args.backend_url, timeout=args.route_timeout_seconds)
                )

            cache_stats = {"before_clear": _cache_stats(active_client)}
            cache_clear_response = _clear_backend_caches(active_client)
            cache_stats["after_clear"] = _cache_stats(active_client)

            cold_payload = run_thesis_evaluation(
                _clone_args(
                    args,
                    run_id=cold_run_id,
                    cache_mode="cold",
                    cold_cache_scope="hot_rerun_cold_source",
                    evaluation_suite_role="hot_rerun_cold_source",
                ),
                client=active_client,
            )
            cache_stats["after_cold"] = _cache_stats(active_client)
            cache_stats["restore_response"] = _restore_hot_rerun_route_cache(active_client)
            cache_stats["after_restore"] = _cache_stats(active_client)

            hot_payload = run_thesis_evaluation(
                _clone_args(
                    args,
                    run_id=hot_run_id,
                    cache_mode="preserve",
                    evaluation_suite_role="hot_rerun",
                ),
                client=active_client,
            )
            cache_stats["after_hot"] = _cache_stats(active_client)

        comparison = build_hot_rerun_comparison(
            pair_run_id=pair_run_id,
            cold_run_id=cold_run_id,
            hot_run_id=hot_run_id,
            cold_summary_rows=cold_payload.get("summary_rows", []),
            hot_summary_rows=hot_payload.get("summary_rows", []),
            cache_stats={
                **cache_stats,
                "cache_clear_response": cache_clear_response,
            },
        )
        comparison_json_path = write_json_artifact(
            hot_run_id,
            "hot_rerun_vs_cold_comparison.json",
            comparison,
        )
        comparison_csv_path = write_csv_artifact(
            hot_run_id,
            "hot_rerun_vs_cold_comparison.csv",
            fieldnames=HOT_COMPARISON_FIELDS,
            rows=list(comparison.get("comparison_rows", [])),
        )
        gate_path = write_json_artifact(hot_run_id, "hot_rerun_gate.json", comparison["hot_gate"])
        report_path = write_text_artifact(
            hot_run_id,
            "hot_rerun_report.md",
            _hot_rerun_report(comparison),
        )
        _annotate_run_artifact(
            run_id=cold_run_id,
            benchmark_phase="cold_rerun_source",
            pair_run_id=pair_run_id,
            paired_run_id=hot_run_id,
            cache_reset_before_run=True,
            cache_stats_before_run=cache_stats.get("after_clear", {}),
            cache_stats_after_run=cache_stats.get("after_cold", {}),
        )
        _annotate_run_artifact(
            run_id=hot_run_id,
            benchmark_phase="hot_rerun",
            pair_run_id=pair_run_id,
            paired_run_id=cold_run_id,
            cache_reset_before_run=False,
            cache_stats_before_run=cache_stats.get("after_restore", cache_stats.get("after_cold", {})),
            cache_stats_after_run=cache_stats.get("after_hot", {}),
            comparison_artifact_name=Path(comparison_json_path).name,
        )
        manifest_path = write_manifest(
            pair_run_id,
            {
                "request": {
                    "hot_rerun_benchmark": {
                        "pair_run_id": pair_run_id,
                        "cold_run_id": cold_run_id,
                        "hot_run_id": hot_run_id,
                        "backend_url": args.backend_url,
                        "corpus_csv": getattr(args, "corpus_csv", None),
                        "corpus_json": getattr(args, "corpus_json", None),
                    }
                },
                "execution": {
                    "hot_gate_all_green": comparison["hot_gate"]["all_green"],
                    "comparison_artifact": str(comparison_json_path),
                    "comparison_csv_artifact": str(comparison_csv_path),
                    "gate_artifact": str(gate_path),
                    "report_artifact": str(report_path),
                },
            },
        )
        return {
            "pair_run_id": pair_run_id,
            "cold_run_id": cold_run_id,
            "hot_run_id": hot_run_id,
            "cold_payload": cold_payload,
            "hot_payload": hot_payload,
            "cache_stats": cache_stats,
            "cache_clear_response": cache_clear_response,
            "comparison": comparison,
            "hot_gate": comparison["hot_gate"],
            "comparison_json": str(comparison_json_path),
            "comparison_csv": str(comparison_csv_path),
            "gate_json": str(gate_path),
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
        }
    finally:
        settings.out_dir = old_out_dir
        if own_client and client is not None and hasattr(client, "close"):
            client.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_hot_rerun_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
