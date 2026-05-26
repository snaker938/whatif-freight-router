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

from app.confidence_sequences import DEFAULT_CONFIDENCE_DELTA, anytime_hoeffding_interval
from app.run_store import artifact_dir_for_run, write_csv_artifact, write_json_artifact, write_manifest, write_text_artifact
from app.settings import settings
from scripts.run_thesis_evaluation import (
    VARIANTS,
    _build_parser as _build_eval_parser,
    in_process_backend_runtime_profile,
    run_thesis_evaluation,
)

HOT_ROUTE_REUSE_VARIANTS: tuple[str, ...] = ("A", "B", "C")
HOT_REFC_REUSE_VARIANTS: tuple[str, ...] = ("B", "C")
HOT_CONTROLLER_REUSE_VARIANTS: tuple[str, ...] = ("C",)
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
    "cold_mean_controller_reuse_rate",
    "hot_mean_controller_reuse_rate",
    "controller_reuse_rate_delta",
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
    "hot_cold_parity_row_count",
    "hot_cold_parity_match_count",
    "hot_cold_parity_rate",
    "route_id_parity_rate",
    "terminal_type_parity_rate",
    "certified_flag_parity_rate",
    "certificate_winner_parity_rate",
    "semantic_drift_count",
    "semantic_drift_rate",
    "certificate_lcb_available_row_count",
    "certificate_lcb_unavailable_row_count",
    "cold_mean_certificate_lcb",
    "hot_mean_certificate_lcb",
    "certificate_lcb_drift",
    "max_final_certificate_lcb_abs_drift",
    "certificate_lcb_source_metric",
]

DIRECT_CERTIFICATE_LCB_SOURCE = "artifact:certificate_summary.json.certificate_lcb"
DERIVED_CERTIFICATE_LCB_SOURCE = (
    "derived:anytime_hoeffding("
    "certificate_summary.empirical_selected_certificate|selected_certificate,"
    "world_count,confidence_delta)"
)


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


def _as_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _normalized_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _stable_route_identity(row: Mapping[str, Any]) -> str | None:
    for field in (
        "selected_candidate_source_label",
        "selected_final_route_source_label",
        "selected_route_signature",
        "certificate_winner_route_signature",
        "route_signature",
        "route_id",
    ):
        value = _normalized_text(row.get(field))
        if value is not None:
            return value
    return None


def _stable_certificate_winner_identity(row: Mapping[str, Any]) -> str | None:
    for field in (
        "selected_candidate_source_label",
        "selected_final_route_source_label",
        "certificate_winner_route_signature",
    ):
        value = _normalized_text(row.get(field))
        if value is not None:
            return value
    stable_route_identity = _stable_route_identity(row)
    if stable_route_identity is not None:
        return stable_route_identity
    return _normalized_text(row.get("certificate_winner_route_id"))


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_eval_parser()
    parser.description = "Run a cold evaluation followed by a true hot rerun on the same backend instance."
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


def _controller_reuse_source_metric(row: Mapping[str, Any]) -> str | None:
    preserved_source = str(row.get("controller_reuse_source_metric") or "").strip()
    if preserved_source:
        return preserved_source
    for field in (
        "mean_controller_reuse_rate",
        "controller_reuse_rate",
        "mean_voi_dccs_cache_hit_rate",
        "voi_dccs_cache_hit_rate",
    ):
        value = _as_float(row.get(field))
        if value is not None:
            return field
    return None


def _controller_reuse_rate(row: Mapping[str, Any]) -> float | None:
    source_metric = _controller_reuse_source_metric(row)
    if source_metric is None:
        return None
    return _as_float(row.get(source_metric))


def _with_controller_reuse_alias(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    source_metric = _controller_reuse_source_metric(normalized)
    controller_reuse_rate = _as_float(normalized.get(source_metric)) if source_metric is not None else None
    if controller_reuse_rate is not None:
        normalized.setdefault("controller_reuse_source_metric", source_metric)
        normalized.setdefault("mean_controller_reuse_rate", controller_reuse_rate)
        normalized.setdefault("controller_reuse_rate", controller_reuse_rate)
    return normalized


def _result_row_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    variant_id = _normalized_text(row.get("variant_id"))
    od_id = _normalized_text(row.get("od_id"))
    if variant_id and od_id:
        return variant_id, od_id
    return None


def _matched_result_rows_by_variant(
    cold_rows: Sequence[Mapping[str, Any]],
    hot_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    cold_by_key = {
        key: dict(row)
        for row in cold_rows
        if (key := _result_row_key(row)) is not None
    }
    hot_by_key = {
        key: dict(row)
        for row in hot_rows
        if (key := _result_row_key(row)) is not None
    }
    pairs_by_variant: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for key in sorted(set(cold_by_key) & set(hot_by_key)):
        variant_id, _ = key
        pairs_by_variant.setdefault(variant_id, []).append((cold_by_key[key], hot_by_key[key]))
    return pairs_by_variant


def _artifact_payload(
    artifact_run_id: str | None,
    artifact_name: str,
    cache: dict[tuple[str, str], Mapping[str, Any] | None],
) -> Mapping[str, Any] | None:
    run_id = _normalized_text(artifact_run_id)
    if run_id is None:
        return None
    cache_key = (run_id, artifact_name)
    if cache_key in cache:
        return cache[cache_key]
    path = artifact_dir_for_run(run_id) / artifact_name
    payload: Mapping[str, Any] | None = None
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            payload = loaded
    cache[cache_key] = payload
    return payload


def _certificate_lcb_from_payload(payload: Mapping[str, Any]) -> tuple[float | None, str | None]:
    for field in ("certificate_lcb", "selected_certificate_lcb"):
        value = _as_float(payload.get(field))
        if value is not None:
            return round(value, 6), DIRECT_CERTIFICATE_LCB_SOURCE

    selected_route_id = _normalized_text(payload.get("selected_route_id")) or _normalized_text(
        payload.get("winner_route_id")
    )
    empirical_certificate = None
    for field in ("empirical_selected_certificate", "empirical_certificate", "selected_certificate"):
        empirical_certificate = _as_float(payload.get(field))
        if empirical_certificate is not None:
            break
    if empirical_certificate is None and selected_route_id is not None:
        route_certificates = payload.get("route_certificates")
        if isinstance(route_certificates, Mapping):
            empirical_certificate = _as_float(route_certificates.get(selected_route_id))

    world_count_value = _as_float(payload.get("world_count"))
    sample_count = max(0, int(world_count_value)) if world_count_value is not None else 0
    if empirical_certificate is None or sample_count <= 0:
        return None, None

    selector_config = payload.get("selector_config")
    selector_config = dict(selector_config) if isinstance(selector_config, Mapping) else {}
    world_manifest = payload.get("world_manifest")
    world_manifest = dict(world_manifest) if isinstance(world_manifest, Mapping) else {}
    delta = DEFAULT_CONFIDENCE_DELTA
    for candidate in (
        world_manifest.get("confidence_delta"),
        world_manifest.get("delta"),
        payload.get("confidence_delta"),
        payload.get("delta"),
        selector_config.get("confidence_delta"),
        selector_config.get("delta"),
    ):
        parsed = _as_float(candidate)
        if parsed is not None and 0.0 < parsed < 1.0:
            delta = parsed
            break

    success_count = min(sample_count, max(0, int(round(empirical_certificate * float(sample_count)))))
    lower_bound, _ = anytime_hoeffding_interval(success_count, sample_count, delta=delta)
    return round(lower_bound, 6), DERIVED_CERTIFICATE_LCB_SOURCE


def _certificate_lcb_from_row_artifact(
    row: Mapping[str, Any],
    cache: dict[tuple[str, str], Mapping[str, Any] | None],
) -> tuple[float | None, str | None]:
    payload = _artifact_payload(
        row.get("artifact_run_id"),
        "certificate_summary.json",
        cache,
    )
    if not isinstance(payload, Mapping):
        return None, None
    return _certificate_lcb_from_payload(payload)


def _merge_lcb_sources(sources: set[str]) -> str | None:
    if not sources:
        return None
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed:" + "|".join(sorted(sources))


def _rounded_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _rounded_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / float(len(values)), 6)


def _variant_parity_and_drift_metrics(
    row_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    artifact_cache: dict[tuple[str, str], Mapping[str, Any] | None],
) -> dict[str, Any]:
    matched_row_count = len(row_pairs)
    route_id_denominator = 0
    route_id_match_count = 0
    terminal_type_denominator = 0
    terminal_type_match_count = 0
    certified_flag_denominator = 0
    certified_flag_match_count = 0
    certificate_winner_denominator = 0
    certificate_winner_match_count = 0
    hot_cold_parity_match_count = 0
    semantic_drift_count = 0
    cold_lcbs: list[float] = []
    hot_lcbs: list[float] = []
    lcb_abs_drifts: list[float] = []
    lcb_sources: set[str] = set()

    for cold_row, hot_row in row_pairs:
        comparable_matches: list[bool] = []

        cold_route_id = _stable_route_identity(cold_row)
        hot_route_id = _stable_route_identity(hot_row)
        if cold_route_id is not None and hot_route_id is not None:
            route_id_denominator += 1
            route_match = cold_route_id == hot_route_id
            comparable_matches.append(route_match)
            if route_match:
                route_id_match_count += 1

        cold_terminal_type = _normalized_text(cold_row.get("preference_terminal_type"))
        hot_terminal_type = _normalized_text(hot_row.get("preference_terminal_type"))
        if cold_terminal_type is not None and hot_terminal_type is not None:
            terminal_type_denominator += 1
            terminal_match = cold_terminal_type == hot_terminal_type
            comparable_matches.append(terminal_match)
            if terminal_match:
                terminal_type_match_count += 1

        cold_certified = _as_bool(cold_row.get("certified"))
        hot_certified = _as_bool(hot_row.get("certified"))
        if cold_certified is not None and hot_certified is not None:
            certified_flag_denominator += 1
            certified_match = cold_certified == hot_certified
            comparable_matches.append(certified_match)
            if certified_match:
                certified_flag_match_count += 1

        cold_certificate_winner = _stable_certificate_winner_identity(cold_row)
        hot_certificate_winner = _stable_certificate_winner_identity(hot_row)
        if cold_certificate_winner is not None and hot_certificate_winner is not None:
            certificate_winner_denominator += 1
            certificate_winner_match = cold_certificate_winner == hot_certificate_winner
            comparable_matches.append(certificate_winner_match)
            if certificate_winner_match:
                certificate_winner_match_count += 1

        if comparable_matches and all(comparable_matches):
            hot_cold_parity_match_count += 1
        if comparable_matches and any(not match for match in comparable_matches):
            semantic_drift_count += 1

        cold_lcb, cold_lcb_source = _certificate_lcb_from_row_artifact(cold_row, artifact_cache)
        hot_lcb, hot_lcb_source = _certificate_lcb_from_row_artifact(hot_row, artifact_cache)
        if cold_lcb is not None and hot_lcb is not None:
            cold_lcbs.append(cold_lcb)
            hot_lcbs.append(hot_lcb)
            lcb_abs_drifts.append(abs(hot_lcb - cold_lcb))
            if cold_lcb_source:
                lcb_sources.add(cold_lcb_source)
            if hot_lcb_source:
                lcb_sources.add(hot_lcb_source)

    cold_mean_certificate_lcb = _rounded_mean(cold_lcbs)
    hot_mean_certificate_lcb = _rounded_mean(hot_lcbs)
    certificate_lcb_drift = (
        round(hot_mean_certificate_lcb - cold_mean_certificate_lcb, 6)
        if cold_mean_certificate_lcb is not None and hot_mean_certificate_lcb is not None
        else None
    )
    certificate_lcb_available_row_count = len(cold_lcbs)
    max_final_certificate_lcb_abs_drift = (
        round(max(lcb_abs_drifts), 6)
        if lcb_abs_drifts
        else None
    )

    return {
        "hot_cold_parity_row_count": matched_row_count,
        "hot_cold_parity_match_count": hot_cold_parity_match_count,
        "hot_cold_parity_rate": _rounded_rate(hot_cold_parity_match_count, matched_row_count),
        "route_id_parity_rate": _rounded_rate(route_id_match_count, route_id_denominator),
        "terminal_type_parity_rate": _rounded_rate(terminal_type_match_count, terminal_type_denominator),
        "certified_flag_parity_rate": _rounded_rate(certified_flag_match_count, certified_flag_denominator),
        "certificate_winner_parity_rate": _rounded_rate(certificate_winner_match_count, certificate_winner_denominator),
        "semantic_drift_count": semantic_drift_count,
        "semantic_drift_rate": _rounded_rate(semantic_drift_count, matched_row_count),
        "certificate_lcb_available_row_count": certificate_lcb_available_row_count,
        "certificate_lcb_unavailable_row_count": max(0, matched_row_count - certificate_lcb_available_row_count),
        "cold_mean_certificate_lcb": cold_mean_certificate_lcb,
        "hot_mean_certificate_lcb": hot_mean_certificate_lcb,
        "certificate_lcb_drift": certificate_lcb_drift,
        "max_final_certificate_lcb_abs_drift": max_final_certificate_lcb_abs_drift,
        "certificate_lcb_source_metric": _merge_lcb_sources(lcb_sources),
    }


def _comparison_row(
    cold_row: Mapping[str, Any] | None,
    hot_row: Mapping[str, Any] | None,
    *,
    row_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]] = (),
    artifact_cache: dict[tuple[str, str], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
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
    cold_controller_reuse, hot_controller_reuse, delta_controller_reuse = metric_delta("mean_controller_reuse_rate")
    cold_world_reuse, hot_world_reuse, delta_world_reuse = metric_delta("mean_refc_world_reuse_rate")
    parity_and_drift = _variant_parity_and_drift_metrics(
        row_pairs,
        artifact_cache or {},
    )

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
        "cold_mean_controller_reuse_rate": cold_controller_reuse,
        "hot_mean_controller_reuse_rate": hot_controller_reuse,
        "controller_reuse_rate_delta": delta_controller_reuse,
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
        **parity_and_drift,
    }


def build_hot_rerun_comparison(
    *,
    pair_run_id: str,
    cold_run_id: str,
    hot_run_id: str,
    cold_summary_rows: Sequence[Mapping[str, Any]],
    hot_summary_rows: Sequence[Mapping[str, Any]],
    cache_stats: Mapping[str, Any],
    cold_rows: Sequence[Mapping[str, Any]] = (),
    hot_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    normalized_cold_summary_rows = [_with_controller_reuse_alias(row) for row in cold_summary_rows]
    normalized_hot_summary_rows = [_with_controller_reuse_alias(row) for row in hot_summary_rows]
    cold_by_variant = _summary_row_map(normalized_cold_summary_rows)
    hot_by_variant = _summary_row_map(normalized_hot_summary_rows)
    matched_rows_by_variant = _matched_result_rows_by_variant(cold_rows, hot_rows)
    artifact_cache: dict[tuple[str, str], Mapping[str, Any] | None] = {}
    comparison_rows = [
        _comparison_row(
            cold_by_variant.get(variant_id),
            hot_by_variant.get(variant_id),
            row_pairs=matched_rows_by_variant.get(variant_id, ()),
            artifact_cache=artifact_cache,
        )
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

    controller_reuse_reporting: list[dict[str, Any]] = []
    for variant_id in HOT_CONTROLLER_REUSE_VARIANTS:
        cold_row = cold_by_variant.get(variant_id, {})
        hot_row = hot_by_variant.get(variant_id, {})
        cold_value = _controller_reuse_rate(cold_row)
        hot_value = _controller_reuse_rate(hot_row)
        if cold_value is None and hot_value is None:
            continue
        controller_reuse_reporting.append(
            {
                "metric": "mean_controller_reuse_rate",
                "variant_id": variant_id,
                "cold_value": cold_value,
                "hot_value": hot_value,
                "delta": (
                    round(hot_value - cold_value, 6)
                    if cold_value is not None and hot_value is not None
                    else None
                ),
                "cold_source_metric": _controller_reuse_source_metric(cold_row),
                "hot_source_metric": _controller_reuse_source_metric(hot_row),
            }
        )

    parity_reporting = [
        {
            "metric": "hot_cold_parity_rate",
            "variant_id": str(row.get("variant_id") or ""),
            "matched_row_count": int(row.get("hot_cold_parity_row_count") or 0),
            "parity_match_count": int(row.get("hot_cold_parity_match_count") or 0),
            "value": row.get("hot_cold_parity_rate"),
            "route_id_parity_rate": row.get("route_id_parity_rate"),
            "terminal_type_parity_rate": row.get("terminal_type_parity_rate"),
            "certified_flag_parity_rate": row.get("certified_flag_parity_rate"),
            "certificate_winner_parity_rate": row.get("certificate_winner_parity_rate"),
            "pass": _as_float(row.get("hot_cold_parity_rate")) == 1.0,
        }
        for row in comparison_rows
        if int(row.get("hot_cold_parity_row_count") or 0) > 0
    ]
    lcb_drift_reporting = [
        {
            "metric": "mean_certificate_lcb_drift",
            "variant_id": str(row.get("variant_id") or ""),
            "matched_row_count": int(row.get("hot_cold_parity_row_count") or 0),
            "available_row_count": int(row.get("certificate_lcb_available_row_count") or 0),
            "unavailable_row_count": int(row.get("certificate_lcb_unavailable_row_count") or 0),
            "cold_value": row.get("cold_mean_certificate_lcb"),
            "hot_value": row.get("hot_mean_certificate_lcb"),
            "delta": row.get("certificate_lcb_drift"),
            "max_abs_delta": row.get("max_final_certificate_lcb_abs_drift"),
            "source_metric": row.get("certificate_lcb_source_metric"),
            "pass": (
                str(row.get("variant_id") or "") in HOT_REFC_REUSE_VARIANTS
                and int(row.get("certificate_lcb_available_row_count") or 0) > 0
                and abs(_as_float(row.get("certificate_lcb_drift")) or 0.0) <= 0.01 + 1e-9
                and (_as_float(row.get("max_final_certificate_lcb_abs_drift")) or 0.0) <= 0.03 + 1e-9
            ),
        }
        for row in comparison_rows
        if int(row.get("hot_cold_parity_row_count") or 0) > 0
    ]
    semantic_drift_reporting = [
        {
            "metric": "semantic_drift_rate",
            "variant_id": str(row.get("variant_id") or ""),
            "matched_row_count": int(row.get("hot_cold_parity_row_count") or 0),
            "drift_count": int(row.get("semantic_drift_count") or 0),
            "value": row.get("semantic_drift_rate"),
            "pass": (
                str(row.get("variant_id") or "") in HOT_ROUTE_REUSE_VARIANTS
                and (_as_float(row.get("semantic_drift_rate")) or 0.0) <= 1e-9
            ),
        }
        for row in comparison_rows
        if int(row.get("hot_cold_parity_row_count") or 0) > 0
    ]

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
            "controller_reuse_variants": list(HOT_CONTROLLER_REUSE_VARIANTS),
            "runtime_improvement_variants": list(HOT_RUNTIME_IMPROVEMENT_VARIANTS),
        },
        "metric_checks": metric_checks,
        "controller_reuse_reporting": controller_reuse_reporting,
        "parity_reporting": parity_reporting,
        "lcb_drift_reporting": lcb_drift_reporting,
        "semantic_drift_reporting": semantic_drift_reporting,
        "runtime_improvement_checks": runtime_improvement_checks,
        "all_green": all(
            check["pass"]
            for check in [
                *metric_checks,
                *runtime_improvement_checks,
                *parity_reporting,
                *[
                    check
                    for check in lcb_drift_reporting
                    if str(check.get("variant_id") or "") in HOT_REFC_REUSE_VARIANTS
                ],
                *[
                    check
                    for check in semantic_drift_reporting
                    if str(check.get("variant_id") or "") in HOT_ROUTE_REUSE_VARIANTS
                ],
            ]
        ),
    }
    return {
        "pair_run_id": pair_run_id,
        "created_at": _now(),
        "cold_run_id": cold_run_id,
        "hot_run_id": hot_run_id,
        "cache_stats": dict(cache_stats),
        "cold_summary_rows": normalized_cold_summary_rows,
        "hot_summary_rows": normalized_hot_summary_rows,
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
            f"controller_reuse={row.get('hot_mean_controller_reuse_rate')} "
            f"(cold {row.get('cold_mean_controller_reuse_rate')}), "
            f"refc_world_reuse={row.get('hot_mean_refc_world_reuse_rate')} "
            f"(cold {row.get('cold_mean_refc_world_reuse_rate')}), "
            f"runtime_ratio_vs_osrm={row.get('hot_mean_runtime_ratio_vs_osrm')} "
            f"(cold {row.get('cold_mean_runtime_ratio_vs_osrm')}), "
            f"runtime_ratio_vs_ors={row.get('hot_mean_runtime_ratio_vs_ors')} "
            f"(cold {row.get('cold_mean_runtime_ratio_vs_ors')})"
        )
    lines.append("")
    lines.append("## Parity And Drift")
    lines.append("")
    for row in comparison.get("comparison_rows", []):
        matched_row_count = int(row.get("hot_cold_parity_row_count") or 0)
        if matched_row_count <= 0:
            continue
        lines.append(
            "- "
            f"{row.get('variant_id')} ({row.get('pipeline_mode')}): "
            f"parity_rate={row.get('hot_cold_parity_rate')} "
            f"route_id_parity_rate={row.get('route_id_parity_rate')}, "
            f"terminal_type_parity_rate={row.get('terminal_type_parity_rate')}, "
            f"certified_flag_parity_rate={row.get('certified_flag_parity_rate')}, "
            f"certificate_winner_parity_rate={row.get('certificate_winner_parity_rate')}, "
            f"semantic_drift_rate={row.get('semantic_drift_rate')}, "
            f"mean_certificate_lcb_drift={row.get('certificate_lcb_drift')} "
            f"max_final_certificate_lcb_abs_drift={row.get('max_final_certificate_lcb_abs_drift')} "
            f"(available_rows={row.get('certificate_lcb_available_row_count')}/{matched_row_count})"
        )
    lines.append("")
    lines.append("## Gate Checks")
    lines.append("")
    for check in comparison["hot_gate"]["metric_checks"]:
        lines.append(
            f"- {check['metric']} / {check['variant_id']}: value={check['value']} "
            f"threshold={check['threshold']} pass={check['pass']}"
        )
    for check in comparison["hot_gate"].get("controller_reuse_reporting", []):
        lines.append(
            f"- {check['metric']} / {check['variant_id']}: cold={check['cold_value']} "
            f"hot={check['hot_value']} delta={check['delta']} "
            f"cold_source={check['cold_source_metric']} hot_source={check['hot_source_metric']}"
        )
    for check in comparison["hot_gate"].get("parity_reporting", []):
        lines.append(
            f"- {check['metric']} / {check['variant_id']}: value={check['value']} "
            f"matched_rows={check['matched_row_count']} parity_matches={check['parity_match_count']} "
            f"route_id_parity_rate={check['route_id_parity_rate']} "
            f"terminal_type_parity_rate={check['terminal_type_parity_rate']} "
            f"certified_flag_parity_rate={check['certified_flag_parity_rate']} "
            f"certificate_winner_parity_rate={check['certificate_winner_parity_rate']}"
        )
    for check in comparison["hot_gate"].get("lcb_drift_reporting", []):
        lines.append(
            f"- {check['metric']} / {check['variant_id']}: cold={check['cold_value']} "
            f"hot={check['hot_value']} delta={check['delta']} "
            f"max_abs_delta={check['max_abs_delta']} "
            f"available_rows={check['available_row_count']} "
            f"unavailable_rows={check['unavailable_row_count']} "
            f"source={check['source_metric']}"
        )
    for check in comparison["hot_gate"].get("semantic_drift_reporting", []):
        lines.append(
            f"- {check['metric']} / {check['variant_id']}: value={check['value']} "
            f"matched_rows={check['matched_row_count']} drift_count={check['drift_count']}"
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
        cache_stats: dict[str, Any]
        cache_clear_response: dict[str, Any]
        cold_payload: dict[str, Any]
        hot_payload: dict[str, Any]

        if client is not None or not bool(getattr(args, "in_process_backend", False)):
            with ExitStack() as stack:
                if client is not None:
                    active_client = client
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
        else:
            with in_process_backend_runtime_profile():
                from app.main import app

                with TestClient(app) as cold_client:
                    cache_stats = {"before_clear": _cache_stats(cold_client)}
                    cache_clear_response = _clear_backend_caches(cold_client)
                    cache_stats["after_clear"] = _cache_stats(cold_client)

                    cold_payload = run_thesis_evaluation(
                        _clone_args(
                            args,
                            run_id=cold_run_id,
                            cache_mode="cold",
                            cold_cache_scope="hot_rerun_cold_source",
                            evaluation_suite_role="hot_rerun_cold_source",
                        ),
                        client=cold_client,
                    )
                    cache_stats["after_cold"] = _cache_stats(cold_client)

                with TestClient(app) as hot_client:
                    cache_stats["restore_response"] = _restore_hot_rerun_route_cache(hot_client)
                    cache_stats["after_restore"] = _cache_stats(hot_client)

                    hot_payload = run_thesis_evaluation(
                        _clone_args(
                            args,
                            run_id=hot_run_id,
                            cache_mode="preserve",
                            evaluation_suite_role="hot_rerun",
                        ),
                        client=hot_client,
                    )
                    cache_stats["after_hot"] = _cache_stats(hot_client)

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
            cold_rows=cold_payload.get("rows", []),
            hot_rows=hot_payload.get("rows", []),
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
