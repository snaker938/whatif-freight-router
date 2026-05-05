from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


thesis_eval: Any | None = None


DEFAULT_TARGET_VARIANTS = ("A", "B", "C")
CANDIDATE_SELECTION_MODES = ("corpus_order", "random_seeded")
DEFAULT_ROUTE_GRAPH_MIN_COVERAGE = 100_000


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the thesis evaluation in small sequential OD tranches with a "
            "persistent green/regression ledger."
        )
    )
    parser.add_argument("--campaign-id", default="thesis_campaign")
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=ROOT / "out" / "thesis_campaigns" / "default",
    )
    parser.add_argument(
        "--corpus-csv",
        type=Path,
        required=True,
        help="Primary candidate corpus that will be widened tranche by tranche.",
    )
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional regression seed corpus. The first tranche runs these ODs "
            "alone; later tranches keep them in the regression set."
        ),
    )
    parser.add_argument(
        "--new-od-batch-size",
        type=int,
        default=1,
        help="How many unseen candidate-corpus ODs to add in each widening tranche.",
    )
    parser.add_argument(
        "--candidate-selection-mode",
        choices=CANDIDATE_SELECTION_MODES,
        default="corpus_order",
        help=(
            "How to admit unseen eligible candidate-corpus ODs when widening. "
            "Replay/regression ODs always keep their preserved order."
        ),
    )
    parser.add_argument(
        "--candidate-selection-seed",
        type=int,
        default=0,
        help=(
            "Seed used for deterministic random candidate admission when "
            "--candidate-selection-mode=random_seeded."
        ),
    )
    parser.add_argument(
        "--max-tranches",
        type=int,
        default=1,
        help="Maximum number of tranches to execute in this invocation.",
    )
    parser.add_argument(
        "--target-variants",
        nargs="+",
        default=list(DEFAULT_TARGET_VARIANTS),
        help="Variants that must clear the tranche gates for an OD to be promoted green.",
    )
    parser.add_argument(
        "--require-balanced-win",
        action="store_true",
        help="Require balanced wins against OSRM and ORS in addition to weighted wins.",
    )
    parser.add_argument(
        "--require-win-v0",
        action="store_true",
        help="Require weighted and balanced wins against V0 as well.",
    )
    parser.add_argument(
        "--min-weighted-margin",
        type=float,
        default=0.0,
        help="Minimum per-row weighted margin required versus OSRM and ORS.",
    )
    parser.add_argument(
        "--min-weighted-margin-v0",
        type=float,
        default=0.0,
        help="Minimum per-row weighted margin required versus V0 when --require-win-v0 is set.",
    )
    parser.add_argument(
        "--require-dominance-win",
        action="store_true",
        help="Require dominance wins against OSRM and ORS before an OD can widen green.",
    )
    parser.add_argument(
        "--require-time-preserving-win",
        action="store_true",
        help="Require time-preserving wins against OSRM and ORS before an OD can widen green.",
    )
    parser.add_argument(
        "--require-proof-grade-readiness",
        action="store_true",
        help=(
            "Require proof-grade readiness metadata: full graph hydration, no degraded "
            "evaluation flag, and strict full-search proof eligibility."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing campaign_state.json if present.",
    )
    parser.add_argument(
        "--evaluation-fast-startup",
        action="store_true",
        help=(
            "Allow evaluation-only deferred route-graph startup when running "
            "the campaign in-process."
        ),
    )
    parser.add_argument(
        "--stage-route-graph-subset",
        action="store_true",
        help=(
            "Stage or reuse a corpus-local route-graph subset asset for each tranche "
            "and record the resulting asset plan in tranche artifacts."
        ),
    )
    parser.add_argument(
        "--route-graph-asset-path",
        default="",
        help=(
            "Override the source ROUTE_GRAPH_ASSET_PATH used when applying or "
            "staging route-graph assets for tranche runs."
        ),
    )
    parser.add_argument(
        "--route-graph-min-nodes",
        type=int,
        default=None,
        help="Override ROUTE_GRAPH_MIN_NODES when applying route-graph runtime overrides.",
    )
    parser.add_argument(
        "--route-graph-min-adjacency",
        type=int,
        default=None,
        help="Override ROUTE_GRAPH_MIN_ADJACENCY when applying route-graph runtime overrides.",
    )
    parser.add_argument(
        "--route-graph-subset-corridor-km",
        type=float,
        default=None,
        help=(
            "Override the staged route-graph subset corridor width in kilometres. "
            "When omitted, the subset builder default is used."
        ),
    )
    parser.add_argument(
        "--stop-on-red-tranche",
        action="store_true",
        default=True,
        help="Stop immediately after a tranche with regressions or failed new ODs.",
    )
    parser.add_argument(
        "--evaluation-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments forwarded verbatim to run_thesis_evaluation.",
    )
    return parser


def _get_thesis_eval() -> Any:
    global thesis_eval
    if thesis_eval is None:
        import scripts.run_thesis_evaluation as thesis_eval_module

        thesis_eval = thesis_eval_module
    return thesis_eval


def _get_route_graph_subset_builder() -> Any:
    import scripts.build_route_graph_subset as route_graph_subset_module

    return route_graph_subset_module


def _default_route_graph_asset_path() -> Path:
    return ROOT / "out" / "model_assets" / "routing_graph_uk.json"


def _route_graph_stage_root() -> Path:
    return ROOT / "out" / "model_assets" / "staged_subsets"


def _corridor_token(corridor_km: float) -> str:
    normalized = f"{float(corridor_km):.3f}".rstrip("0").rstrip(".")
    return normalized.replace(".", "p")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _route_graph_source_signature(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(
        f"{path.resolve()}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}".encode("utf-8")
    ).hexdigest()


def _route_graph_subset_meta_path(asset_path: Path) -> Path:
    return asset_path.with_suffix(".meta.json")


def _read_route_graph_subset_report(asset_path: Path) -> dict[str, Any] | None:
    meta_path = _route_graph_subset_meta_path(asset_path)
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if str(payload.get("version") or "").strip() != "uk-routing-graph-subset-v2":
        return None
    output_json = str(payload.get("output_json") or "").strip()
    if output_json and Path(output_json).expanduser().resolve() != asset_path.expanduser().resolve():
        return None
    return dict(payload)


def _count_route_graph_adjacency_keys(asset_path: Path) -> int | None:
    route_graph_subset = _get_route_graph_subset_builder()
    ijson = getattr(route_graph_subset, "ijson", None)
    if ijson is None:
        return None
    try:
        with asset_path.open("rb") as handle:
            return sum(1 for _key, _value in ijson.kvitems(handle, "adjacency"))
    except Exception:
        return None


def _derive_route_graph_threshold(
    *,
    available_count: int | None,
    requested_count: int | None,
) -> int | None:
    if available_count is None:
        if requested_count is None:
            return None
        return max(1, int(requested_count))
    baseline = int(available_count)
    if requested_count is not None:
        baseline = min(baseline, int(requested_count))
    else:
        baseline = min(baseline, int(DEFAULT_ROUTE_GRAPH_MIN_COVERAGE))
    return max(1, baseline)


def _route_graph_asset_plan_env_overrides(plan: Mapping[str, Any]) -> dict[str, str]:
    asset_path = str(plan.get("asset_path") or "").strip()
    overrides: dict[str, str] = {}
    if asset_path:
        overrides["ROUTE_GRAPH_ASSET_PATH"] = str(Path(asset_path).expanduser().resolve())
    min_nodes = plan.get("route_graph_min_nodes")
    if min_nodes is not None:
        overrides["ROUTE_GRAPH_MIN_NODES"] = str(max(1, int(min_nodes)))
    min_adjacency = plan.get("route_graph_min_adjacency")
    if min_adjacency is not None:
        overrides["ROUTE_GRAPH_MIN_ADJACENCY"] = str(max(1, int(min_adjacency)))
    return overrides


def _apply_route_graph_asset_plan_to_eval_args(
    eval_args: argparse.Namespace,
    asset_plan: Mapping[str, Any] | None,
) -> argparse.Namespace:
    if not isinstance(asset_plan, Mapping):
        return eval_args
    asset_path = str(asset_plan.get("asset_path") or "").strip()
    if asset_path:
        setattr(eval_args, "route_graph_asset_path", asset_path)
    min_nodes = asset_plan.get("route_graph_min_nodes")
    if min_nodes is not None:
        setattr(eval_args, "route_graph_min_nodes", max(1, int(min_nodes)))
    min_adjacency = asset_plan.get("route_graph_min_adjacency")
    if min_adjacency is not None:
        setattr(eval_args, "route_graph_min_adjacency", max(1, int(min_adjacency)))
    return eval_args


def _build_route_graph_asset_plan(
    *,
    corpus_csv: Path | None,
    requested_asset_path: str = "",
    requested_min_nodes: int | None = None,
    requested_min_adjacency: int | None = None,
    requested_subset_corridor_km: float | None = None,
) -> dict[str, Any]:
    requested_asset_text = str(requested_asset_path or "").strip()
    source_asset_path = (
        Path(requested_asset_text).expanduser()
        if requested_asset_text
        else Path(os.environ.get("ROUTE_GRAPH_ASSET_PATH") or "").expanduser()
        if str(os.environ.get("ROUTE_GRAPH_ASSET_PATH") or "").strip()
        else _default_route_graph_asset_path()
    )
    if not source_asset_path.is_absolute():
        source_asset_path = (ROOT / source_asset_path).resolve()
    else:
        source_asset_path = source_asset_path.resolve()
    corpus_path = Path(corpus_csv).expanduser().resolve() if corpus_csv is not None else None
    plan: dict[str, Any] = {
        "generated_at_utc": _now(),
        "mode": "no_subset_requested",
        "source_asset_path": str(source_asset_path),
        "source_asset_exists": bool(source_asset_path.exists()),
        "requested_asset_path": requested_asset_text,
        "requested_min_nodes": int(requested_min_nodes) if requested_min_nodes is not None else None,
        "requested_min_adjacency": int(requested_min_adjacency) if requested_min_adjacency is not None else None,
        "requested_subset_corridor_km": (
            float(requested_subset_corridor_km) if requested_subset_corridor_km is not None else None
        ),
        "corpus_csv": str(corpus_path) if corpus_path is not None else "",
        "asset_path": str(source_asset_path),
        "route_graph_min_nodes": _derive_route_graph_threshold(
            available_count=None,
            requested_count=requested_min_nodes,
        ),
        "route_graph_min_adjacency": _derive_route_graph_threshold(
            available_count=None,
            requested_count=requested_min_adjacency,
        ),
    }
    if corpus_path is None or not corpus_path.exists():
        plan["mode"] = "no_subset_missing_corpus_csv"
        plan["env_overrides"] = _route_graph_asset_plan_env_overrides(plan)
        return plan
    if not source_asset_path.exists():
        plan["mode"] = "source_asset_missing"
        plan["env_overrides"] = _route_graph_asset_plan_env_overrides(plan)
        return plan
    existing_subset_report = _read_route_graph_subset_report(source_asset_path)
    route_graph_subset = _get_route_graph_subset_builder()
    corridor_km = (
        float(requested_subset_corridor_km)
        if requested_subset_corridor_km is not None
        else float(getattr(route_graph_subset, "DEFAULT_CORRIDOR_KM", 35.0))
    )
    report: dict[str, Any] | None = None
    stage_output_path: Path | None = None
    if existing_subset_report is not None:
        plan["mode"] = "explicit_subset_asset"
        report = existing_subset_report
        stage_output_path = source_asset_path
    else:
        corpus_hash = _hash_file(corpus_path)[:12]
        source_sig = _route_graph_source_signature(source_asset_path)[:12]
        stage_root = _route_graph_stage_root()
        stage_root.mkdir(parents=True, exist_ok=True)
        corridor_token = _corridor_token(corridor_km)
        stage_output_path = stage_root / f"{source_asset_path.stem}.subset.c{corridor_token}.{corpus_hash}.{source_sig}.json"
        report = _read_route_graph_subset_report(stage_output_path)
        if report is not None and stage_output_path.exists():
            plan["mode"] = "reused_subset_asset"
        else:
            try:
                report = route_graph_subset.build_subset(
                    graph_json=source_asset_path,
                    corpus_csv=corpus_path,
                    output_json=stage_output_path,
                    corridor_km=corridor_km,
                )
                plan["mode"] = "staged_subset_asset"
            except Exception as exc:
                plan["mode"] = "subset_stage_failed"
                plan["stage_error"] = f"{type(exc).__name__}: {exc}"
                plan["env_overrides"] = _route_graph_asset_plan_env_overrides(plan)
                return plan
    report = dict(report or {})
    adjacency_keys = _count_route_graph_adjacency_keys(stage_output_path or source_asset_path)
    nodes_kept = int(report.get("nodes_kept") or 0) or None
    route_graph_min_nodes = _derive_route_graph_threshold(
        available_count=nodes_kept,
        requested_count=requested_min_nodes,
    )
    route_graph_min_adjacency = _derive_route_graph_threshold(
        available_count=adjacency_keys if adjacency_keys is not None else nodes_kept,
        requested_count=requested_min_adjacency,
    )
    plan.update(
        {
            "asset_path": str((stage_output_path or source_asset_path).resolve()),
            "subset_report_path": str(report.get("output_meta_json") or _route_graph_subset_meta_path(stage_output_path or source_asset_path)),
            "subset_corridor_km": float(report.get("corridor_km") or corridor_km),
            "nodes_kept": nodes_kept,
            "edges_kept": int(report.get("edges_kept") or 0) or None,
            "adjacency_keys": adjacency_keys,
            "adjacency_keys_estimated_from_nodes": bool(adjacency_keys is None and nodes_kept is not None),
            "compact_bundle_path": str(report.get("compact_bundle") or "").strip(),
            "compact_bundle_meta_path": str(report.get("compact_bundle_meta") or "").strip(),
            "used_existing_output_json": bool(report.get("used_existing_output_json")),
            "resumed_from_staging": bool(report.get("resumed_from_staging")),
            "route_graph_min_nodes": route_graph_min_nodes,
            "route_graph_min_adjacency": route_graph_min_adjacency,
        }
    )
    plan["env_overrides"] = _route_graph_asset_plan_env_overrides(plan)
    return plan


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write empty tranche csv")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_od_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        od_id = str(row.get("od_id") or "").strip()
        if not od_id or od_id in seen:
            continue
        seen.add(od_id)
        normalized.append(dict(row))
    return normalized


def _empty_state(*, campaign_id: str, corpus_csv: Path) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "sequential_mode": "retain_green_replay_then_widen",
        "created_at_utc": _now(),
        "updated_at_utc": _now(),
        "corpus_csv": str(corpus_csv),
        "gate_config": {},
        "bootstrap_od_ids": [],
        "bootstrapped": False,
        "green_od_ids": [],
        "red_od_ids": [],
        "completed_candidate_od_ids": [],
        "tranches": [],
    }


def _load_state(path: Path, *, campaign_id: str, corpus_csv: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return _empty_state(campaign_id=campaign_id, corpus_csv=corpus_csv)


def _float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _variant_row_passes(
    row: Mapping[str, Any],
    *,
    require_balanced_win: bool,
    require_win_v0: bool,
    require_dominance_win: bool,
    require_time_preserving_win: bool,
    min_weighted_margin: float,
    min_weighted_margin_v0: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(row.get("failure_reason") or "").strip():
        reasons.append(f"failure:{row.get('failure_reason')}")
    if not _truthy(row.get("weighted_win_osrm")):
        reasons.append("weighted_win_osrm")
    if not _truthy(row.get("weighted_win_ors")):
        reasons.append("weighted_win_ors")
    if _float(row.get("weighted_margin_vs_osrm")) < float(min_weighted_margin):
        reasons.append("weighted_margin_vs_osrm")
    if _float(row.get("weighted_margin_vs_ors")) < float(min_weighted_margin):
        reasons.append("weighted_margin_vs_ors")
    if require_balanced_win:
        if not _truthy(row.get("balanced_win_osrm")):
            reasons.append("balanced_win_osrm")
        if not _truthy(row.get("balanced_win_ors")):
            reasons.append("balanced_win_ors")
    if require_dominance_win:
        if not _truthy(row.get("dominance_win_osrm")):
            reasons.append("dominance_win_osrm")
        if not _truthy(row.get("dominance_win_ors")):
            reasons.append("dominance_win_ors")
    if require_time_preserving_win:
        if not _truthy(row.get("time_preserving_win_osrm")):
            reasons.append("time_preserving_win_osrm")
        if not _truthy(row.get("time_preserving_win_ors")):
            reasons.append("time_preserving_win_ors")
    if require_win_v0:
        if not _truthy(row.get("weighted_win_v0")):
            reasons.append("weighted_win_v0")
        if _float(row.get("weighted_margin_vs_v0")) < float(min_weighted_margin_v0):
            reasons.append("weighted_margin_vs_v0")
        if require_balanced_win and not _truthy(row.get("balanced_win_v0")):
            reasons.append("balanced_win_v0")
    return (not reasons, reasons)


def _evaluate_od_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_variants: Sequence[str],
    require_balanced_win: bool,
    require_win_v0: bool,
    require_dominance_win: bool,
    require_time_preserving_win: bool,
    min_weighted_margin: float,
    min_weighted_margin_v0: float,
) -> dict[str, Any]:
    per_od: dict[str, dict[str, Any]] = {}
    target_variant_set = {str(item) for item in target_variants}
    for row in rows:
        od_id = str(row.get("od_id") or "").strip()
        variant_id = str(row.get("variant_id") or "").strip()
        if not od_id or variant_id not in target_variant_set:
            continue
        passes, reasons = _variant_row_passes(
            row,
            require_balanced_win=require_balanced_win,
            require_win_v0=require_win_v0,
            require_dominance_win=require_dominance_win,
            require_time_preserving_win=require_time_preserving_win,
            min_weighted_margin=min_weighted_margin,
            min_weighted_margin_v0=min_weighted_margin_v0,
        )
        variant_payload = {
            "passes": passes,
            "reasons": reasons,
            "weighted_win_osrm": _truthy(row.get("weighted_win_osrm")),
            "weighted_win_ors": _truthy(row.get("weighted_win_ors")),
            "balanced_win_osrm": _truthy(row.get("balanced_win_osrm")),
            "balanced_win_ors": _truthy(row.get("balanced_win_ors")),
            "dominance_win_osrm": _truthy(row.get("dominance_win_osrm")),
            "dominance_win_ors": _truthy(row.get("dominance_win_ors")),
            "time_preserving_win_osrm": _truthy(row.get("time_preserving_win_osrm")),
            "time_preserving_win_ors": _truthy(row.get("time_preserving_win_ors")),
            "weighted_margin_vs_osrm": _float(row.get("weighted_margin_vs_osrm")),
            "weighted_margin_vs_ors": _float(row.get("weighted_margin_vs_ors")),
            "weighted_win_v0": _truthy(row.get("weighted_win_v0")),
            "balanced_win_v0": _truthy(row.get("balanced_win_v0")),
            "weighted_margin_vs_v0": _float(row.get("weighted_margin_vs_v0")),
            "failure_reason": str(row.get("failure_reason") or ""),
            "selected_final_route_source_label": str(row.get("selected_final_route_source_label") or ""),
            "selected_candidate_source_label": str(row.get("selected_candidate_source_label") or ""),
            "preemptive_comparator_seeded": _truthy(row.get("preemptive_comparator_seeded")),
            "selected_from_preemptive_comparator_seed": _truthy(
                row.get("selected_from_preemptive_comparator_seed")
            ),
            "selected_from_supplemental_rescue": _truthy(row.get("selected_from_supplemental_rescue")),
            "selected_from_comparator_engine": _truthy(row.get("selected_from_comparator_engine")),
            "runtime_ms": _float(row.get("runtime_ms")),
            "algorithm_runtime_ms": _float(row.get("algorithm_runtime_ms")),
        }
        entry = per_od.setdefault(
            od_id,
            {
                "od_id": od_id,
                "evaluated": True,
                "expected_in_tranche": False,
                "variant_results": {},
            },
        )
        entry["variant_results"][variant_id] = variant_payload
    for od_payload in per_od.values():
        missing_variants = sorted(
            variant_id for variant_id in target_variants if variant_id not in od_payload["variant_results"]
        )
        reasons: list[str] = []
        if missing_variants:
            reasons.extend(f"missing_variant:{variant_id}" for variant_id in missing_variants)
        for variant_id in target_variants:
            variant_payload = od_payload["variant_results"].get(variant_id)
            if not variant_payload:
                continue
            if not bool(variant_payload.get("passes")):
                reasons.extend(f"{variant_id}:{reason}" for reason in list(variant_payload.get("reasons") or []))
        od_payload["passes_all_targets"] = not reasons
        od_payload["reasons"] = reasons
    return per_od


def _ensure_expected_od_entries(
    per_od: Mapping[str, Mapping[str, Any]],
    *,
    expected_od_ids: Sequence[str],
    target_variants: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    normalized: dict[str, dict[str, Any]] = {str(od_id): dict(payload) for od_id, payload in per_od.items()}
    ordered_expected_od_ids = list(dict.fromkeys(str(item) for item in expected_od_ids if str(item).strip()))
    expected_set = set(ordered_expected_od_ids)
    evaluated_od_ids = sorted(normalized.keys())
    missing_expected_od_ids: list[str] = []
    for od_id, payload in normalized.items():
        payload["expected_in_tranche"] = od_id in expected_set
        payload["evaluated"] = bool(payload.get("evaluated", True))
    for od_id in ordered_expected_od_ids:
        if od_id in normalized:
            continue
        missing_expected_od_ids.append(od_id)
        normalized[od_id] = {
            "od_id": od_id,
            "evaluated": False,
            "expected_in_tranche": True,
            "variant_results": {},
            "passes_all_targets": False,
            "reasons": [
                "missing_expected_od",
                *[f"missing_variant:{variant_id}" for variant_id in target_variants],
            ],
        }
    return normalized, evaluated_od_ids, missing_expected_od_ids


def _gate_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_variants": list(args.target_variants or []),
        "require_balanced_win": bool(args.require_balanced_win),
        "require_win_v0": bool(args.require_win_v0),
        "require_dominance_win": bool(args.require_dominance_win),
        "require_time_preserving_win": bool(args.require_time_preserving_win),
        "require_proof_grade_readiness": bool(args.require_proof_grade_readiness),
        "min_weighted_margin": float(args.min_weighted_margin),
        "min_weighted_margin_v0": float(args.min_weighted_margin_v0),
        "new_od_batch_size": int(args.new_od_batch_size),
        "candidate_selection_mode": str(args.candidate_selection_mode),
        "candidate_selection_seed": int(args.candidate_selection_seed),
        "stop_on_red_tranche": bool(args.stop_on_red_tranche),
    }


def _artifact_status(raw_path: Any) -> dict[str, Any]:
    path_text = str(raw_path or "").strip()
    path = Path(path_text) if path_text else None
    return {
        "path": path_text,
        "exists": bool(path is not None and path.exists()),
    }


def _path_or_none(raw_path: Any) -> Path | None:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return None
    return Path(path_text)


def _sibling_artifact_path(
    evaluation_payload: Mapping[str, Any],
    *,
    primary_key: str,
    sibling_name: str,
) -> Path | None:
    primary_path = _path_or_none(evaluation_payload.get(primary_key))
    if primary_path is None:
        return None
    return primary_path.parent / sibling_name


def _read_json_artifact(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None or not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "invalid_payload"
    return payload, None


def _proof_grade_readiness_payload(evaluation_payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata_path = _path_or_none(evaluation_payload.get("metadata")) or _path_or_none(
        evaluation_payload.get("metadata_path")
    )
    if metadata_path is None:
        metadata_path = _sibling_artifact_path(
            evaluation_payload,
            primary_key="results_csv",
            sibling_name="metadata.json",
        )
    metadata_artifact = _artifact_status(metadata_path)
    metadata_payload, load_error = _read_json_artifact(metadata_path)
    readiness_summary = (
        metadata_payload.get("backend_ready_summary")
        if isinstance(metadata_payload, Mapping)
        else {}
    )
    route_graph = (
        readiness_summary.get("route_graph")
        if isinstance(readiness_summary, Mapping)
        else {}
    )
    reasons: list[str] = []
    if load_error == "missing":
        reasons.append("missing_metadata_json")
    elif load_error == "invalid_json":
        reasons.append("invalid_metadata_json")
    elif load_error == "invalid_payload":
        reasons.append("invalid_metadata_payload")
    if metadata_payload is not None:
        if not _truthy(readiness_summary.get("strict_route_ready")):
            reasons.append("strict_route_ready")
        if str(route_graph.get("ready_mode") or "").strip().lower() != "full":
            reasons.append("route_graph_ready_mode_full")
        if not _truthy(metadata_payload.get("route_graph_full_hydration_observed")):
            reasons.append("route_graph_full_hydration_observed")
        if _truthy(metadata_payload.get("degraded_evaluation_observed")):
            reasons.append("degraded_evaluation_observed")
        if not _truthy(metadata_payload.get("strict_full_search_proof_eligible")):
            reasons.append("strict_full_search_proof_eligible")
    return {
        "metadata_artifact": metadata_artifact,
        "metadata_load_error": load_error,
        "route_graph_readiness_class": (
            str(metadata_payload.get("route_graph_readiness_class") or "")
            if isinstance(metadata_payload, Mapping)
            else ""
        ),
        "route_graph_ready_mode": str(route_graph.get("ready_mode") or ""),
        "route_graph_load_strategy": str(route_graph.get("load_strategy") or ""),
        "route_graph_full_hydration_observed": _truthy(
            metadata_payload.get("route_graph_full_hydration_observed")
            if isinstance(metadata_payload, Mapping)
            else None
        ),
        "degraded_evaluation_observed": _truthy(
            metadata_payload.get("degraded_evaluation_observed")
            if isinstance(metadata_payload, Mapping)
            else None
        ),
        "strict_full_search_proof_eligible": _truthy(
            metadata_payload.get("strict_full_search_proof_eligible")
            if isinstance(metadata_payload, Mapping)
            else None
        ),
        "strict_route_ready": _truthy(
            readiness_summary.get("strict_route_ready")
            if isinstance(readiness_summary, Mapping)
            else None
        ),
        "proof_grade_readiness_ok": not reasons,
        "proof_grade_readiness_reasons": reasons,
    }


def _evaluation_evidence_payload(evaluation_payload: Mapping[str, Any]) -> dict[str, Any]:
    required_keys = ("results_csv", "summary_csv", "evaluation_manifest")
    optional_keys = (
        "summary_by_cohort_csv",
        "summary_by_cohort_json",
        "cohort_composition_path",
        "thesis_report",
        "methods_appendix",
    )
    required_artifacts = {
        key: _artifact_status(evaluation_payload.get(key))
        for key in required_keys
    }
    optional_artifacts = {
        key: _artifact_status(evaluation_payload.get(key))
        for key in optional_keys
    }
    missing_required_artifacts = sorted(
        key for key, payload in required_artifacts.items() if not bool(payload.get("exists"))
    )
    rows = list(evaluation_payload.get("rows") or [])
    summary_rows = list(evaluation_payload.get("summary_rows") or [])
    run_id = str(evaluation_payload.get("run_id") or "").strip()
    run_validity = evaluation_payload.get("run_validity")
    proof_grade_readiness = _proof_grade_readiness_payload(evaluation_payload)
    return {
        "required_artifacts": required_artifacts,
        "optional_artifacts": optional_artifacts,
        "missing_required_artifacts": missing_required_artifacts,
        "required_artifacts_ok": not missing_required_artifacts,
        "row_payload_present": bool(rows),
        "summary_payload_present": bool(summary_rows),
        "row_count": len(rows),
        "summary_row_count": len(summary_rows),
        "run_id_present": bool(run_id),
        "run_validity_present": run_validity is not None,
        "run_validity": run_validity,
        "proof_grade_readiness": proof_grade_readiness,
    }


def _campaign_markdown(
    *,
    state: Mapping[str, Any],
    last_tranche: Mapping[str, Any] | None,
) -> str:
    gate_config = state.get("gate_config") if isinstance(state, Mapping) else {}
    lines = [
        "# Thesis Campaign Ledger",
        "",
        f"- Campaign id: `{state.get('campaign_id')}`",
        f"- Sequential mode: `{state.get('sequential_mode') or 'retain_green_replay_then_widen'}`",
        f"- Updated at: `{state.get('updated_at_utc')}`",
        f"- Candidate corpus: `{state.get('corpus_csv')}`",
        f"- Bootstrapped: `{state.get('bootstrapped')}`",
        f"- Green OD count: `{len(list(state.get('green_od_ids') or []))}`",
        f"- Red OD count: `{len(list(state.get('red_od_ids') or []))}`",
        f"- Completed candidate OD count: `{len(list(state.get('completed_candidate_od_ids') or []))}`",
    ]
    if gate_config:
        lines.extend(
            [
                "",
                "## Gate Config",
                "",
                f"- Target variants: `{', '.join(list(gate_config.get('target_variants') or [])) or 'none'}`",
                f"- Require balanced win: `{gate_config.get('require_balanced_win')}`",
                f"- Require V0 win: `{gate_config.get('require_win_v0')}`",
                f"- Require dominance win: `{gate_config.get('require_dominance_win')}`",
                f"- Require time-preserving win: `{gate_config.get('require_time_preserving_win')}`",
                f"- Require proof-grade readiness: `{gate_config.get('require_proof_grade_readiness')}`",
                f"- Min weighted margin vs OSRM/ORS: `{gate_config.get('min_weighted_margin')}`",
                f"- Min weighted margin vs V0: `{gate_config.get('min_weighted_margin_v0')}`",
                f"- New OD batch size: `{gate_config.get('new_od_batch_size')}`",
                f"- Candidate selection mode: `{gate_config.get('candidate_selection_mode')}`",
                f"- Candidate selection seed: `{gate_config.get('candidate_selection_seed')}`",
                f"- Stop on non-green tranche: `{gate_config.get('stop_on_red_tranche')}`",
            ]
        )
    if last_tranche:
        replay_od_ids = list(last_tranche.get("replay_od_ids") or [])
        regression_od_ids = list(last_tranche.get("regression_od_ids") or [])
        new_od_ids = list(last_tranche.get("new_od_ids") or [])
        prior_green_od_ids = list(last_tranche.get("prior_green_od_ids") or [])
        replayed_prior_green_od_ids = list(last_tranche.get("replayed_prior_green_od_ids") or [])
        missing_prior_green_replay_od_ids = list(last_tranche.get("missing_prior_green_replay_od_ids") or [])
        retained_green_od_ids = list(last_tranche.get("retained_green_od_ids") or [])
        promoted_green_od_ids = list(last_tranche.get("promoted_green_od_ids") or [])
        unpreserved_green_od_ids = list(last_tranche.get("unpreserved_green_od_ids") or [])
        regression_red_od_ids = list(last_tranche.get("regression_red_od_ids") or [])
        lines.extend(
            [
                "",
                "## Last Tranche",
                "",
                f"- Index: `{last_tranche.get('tranche_index')}`",
                f"- Status: `{last_tranche.get('status')}`",
                f"- Run id: `{last_tranche.get('run_id')}`",
                f"- Selection mode: `{last_tranche.get('selection_mode')}`",
                f"- Candidate selection mode: `{last_tranche.get('candidate_selection_mode')}`",
                f"- Candidate selection seed: `{last_tranche.get('candidate_selection_seed')}`",
                f"- Retrying pending tranche: `{last_tranche.get('retrying_pending_tranche')}`",
                f"- Retry source tranche index: `{last_tranche.get('retry_source_tranche_index')}`",
                f"- Prior green OD count: `{last_tranche.get('prior_green_od_count', len(prior_green_od_ids))}`",
                f"- Prior green OD ids: `{', '.join(prior_green_od_ids) or 'none'}`",
                f"- Replay OD count: `{last_tranche.get('replay_od_count', len(replay_od_ids))}`",
                f"- Replay OD ids: `{', '.join(replay_od_ids) or 'none'}`",
                f"- Replayed prior green OD count: `{last_tranche.get('replayed_prior_green_od_count', len(replayed_prior_green_od_ids))}`",
                f"- Replayed prior green OD ids: `{', '.join(replayed_prior_green_od_ids) or 'none'}`",
                f"- Missing prior green replay OD count: `{last_tranche.get('missing_prior_green_replay_od_count', len(missing_prior_green_replay_od_ids))}`",
                f"- Missing prior green replay OD ids: `{', '.join(missing_prior_green_replay_od_ids) or 'none'}`",
                f"- Regression OD count: `{len(regression_od_ids)}`",
                f"- Regression OD ids: `{', '.join(regression_od_ids) or 'none'}`",
                f"- New OD count: `{len(new_od_ids)}`",
                f"- New OD ids: `{', '.join(new_od_ids) or 'none'}`",
                f"- Evaluated OD ids: `{', '.join(list(last_tranche.get('evaluated_od_ids') or [])) or 'none'}`",
                f"- Missing expected OD ids: `{', '.join(list(last_tranche.get('missing_expected_od_ids') or [])) or 'none'}`",
                f"- Retained green OD count: `{last_tranche.get('retained_green_od_count', len(retained_green_od_ids))}`",
                f"- Retained green OD ids: `{', '.join(retained_green_od_ids) or 'none'}`",
                f"- Promoted green OD count: `{len(promoted_green_od_ids)}`",
                f"- Promoted green OD ids: `{', '.join(promoted_green_od_ids) or 'none'}`",
                f"- Unpreserved green OD count: `{len(unpreserved_green_od_ids)}`",
                f"- Unpreserved green OD ids: `{', '.join(unpreserved_green_od_ids) or 'none'}`",
                f"- Regression red OD count: `{len(regression_red_od_ids)}`",
                f"- Regression red OD ids: `{', '.join(regression_red_od_ids) or 'none'}`",
                f"- Next widening allowed: `{last_tranche.get('next_widening_allowed')}`",
                f"- Next widening block reason: `{last_tranche.get('next_widening_block_reason') or 'none'}`",
                f"- Red OD ids: `{', '.join(list(last_tranche.get('red_od_ids') or [])) or 'none'}`",
                f"- Publication evidence ok: `{last_tranche.get('publication_evidence_ok')}`",
                f"- Missing required artifacts: `{', '.join(list(last_tranche.get('missing_required_artifacts') or [])) or 'none'}`",
                f"- Proof-grade readiness ok: `{last_tranche.get('proof_grade_readiness_ok')}`",
                f"- Proof-grade readiness reasons: `{', '.join(list(last_tranche.get('proof_grade_readiness_reasons') or [])) or 'none'}`",
                f"- Per-OD status JSON: `{last_tranche.get('per_od_status_json')}`",
                f"- Tranche corpus CSV: `{last_tranche.get('tranche_corpus_csv')}`",
                f"- Evaluation results CSV: `{last_tranche.get('results_csv')}`",
                f"- Evaluation summary CSV: `{last_tranche.get('summary_csv')}`",
            ]
        )
        asset_plan = last_tranche.get("route_graph_asset_plan")
        if isinstance(asset_plan, Mapping):
            lines.extend(
                [
                    f"- Route-graph asset plan mode: `{asset_plan.get('mode')}`",
                    f"- Route-graph source asset: `{asset_plan.get('source_asset_path')}`",
                    f"- Route-graph staged asset: `{asset_plan.get('asset_path')}`",
                    f"- Route-graph subset report: `{asset_plan.get('subset_report_path') or 'none'}`",
                    f"- Route-graph min nodes: `{asset_plan.get('route_graph_min_nodes')}`",
                    f"- Route-graph min adjacency: `{asset_plan.get('route_graph_min_adjacency')}`",
                    f"- Route-graph plan error: `{asset_plan.get('stage_error') or 'none'}`",
                ]
            )
    return "\n".join(lines) + "\n"


def _build_eval_args(
    args: argparse.Namespace,
    *,
    tranche_corpus_csv: Path,
    tranche_run_id: str,
    tranche_out_dir: Path,
    route_graph_asset_path_override: str | None = None,
    route_graph_min_nodes_override: int | None = None,
    route_graph_min_adjacency_override: int | None = None,
) -> argparse.Namespace:
    forwarded: list[str] = []
    raw_forwarded = list(args.evaluation_args or [])
    route_graph_override_flags: set[str] = set()
    effective_route_graph_asset_path = (
        str(route_graph_asset_path_override).strip()
        if route_graph_asset_path_override is not None
        else str(getattr(args, "route_graph_asset_path", "") or "").strip()
    )
    effective_route_graph_min_nodes = (
        int(route_graph_min_nodes_override)
        if route_graph_min_nodes_override is not None
        else getattr(args, "route_graph_min_nodes", None)
    )
    effective_route_graph_min_adjacency = (
        int(route_graph_min_adjacency_override)
        if route_graph_min_adjacency_override is not None
        else getattr(args, "route_graph_min_adjacency", None)
    )
    if effective_route_graph_asset_path:
        route_graph_override_flags.add("--route-graph-asset-path")
    if effective_route_graph_min_nodes is not None:
        route_graph_override_flags.add("--route-graph-min-nodes")
    if effective_route_graph_min_adjacency is not None:
        route_graph_override_flags.add("--route-graph-min-adjacency")
    skip_value_for: str | None = None
    for item in raw_forwarded:
        if skip_value_for is not None:
            skip_value_for = None
            continue
        if item in {"--corpus-csv", "--run-id", "--out-dir"} | route_graph_override_flags:
            skip_value_for = item
            continue
        forwarded.append(item)
    if effective_route_graph_asset_path:
        forwarded.extend(["--route-graph-asset-path", str(effective_route_graph_asset_path)])
    if effective_route_graph_min_nodes is not None:
        forwarded.extend(["--route-graph-min-nodes", str(int(effective_route_graph_min_nodes))])
    if effective_route_graph_min_adjacency is not None:
        forwarded.extend(["--route-graph-min-adjacency", str(int(effective_route_graph_min_adjacency))])
    eval_args = _get_thesis_eval()._build_parser().parse_args(
        [
            "--corpus-csv",
            str(tranche_corpus_csv),
            "--run-id",
            tranche_run_id,
            "--out-dir",
            str(tranche_out_dir),
            *forwarded,
        ]
    )
    return eval_args


def _ordered_subset_rows(
    bootstrap_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    regression_od_ids: Sequence[str],
    new_od_ids: Sequence[str],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    desired = list(regression_od_ids) + list(new_od_ids)
    index: dict[str, dict[str, Any]] = {}
    for row in list(bootstrap_rows) + list(candidate_rows):
        od_id = str(row.get("od_id") or "").strip()
        if od_id and od_id not in index:
            index[od_id] = dict(row)
    for od_id in desired:
        row = index.get(od_id)
        if row is not None:
            ordered.append(row)
    return ordered


def _sequential_control_payload(
    *,
    bootstrapping_now: bool,
    last_tranche: Mapping[str, Any] | None,
    prior_green_od_ids: Sequence[str],
    replay_od_ids: Sequence[str],
    regression_od_ids: Sequence[str],
    new_od_ids: Sequence[str],
    tranche_status: str,
    publication_evidence_ok: bool,
    regression_red_od_ids: Sequence[str],
    new_red_od_ids: Sequence[str],
    preserved_green_od_ids: Sequence[str],
    candidate_selection_mode: str,
    candidate_selection_seed: int,
) -> dict[str, Any]:
    prior_green_od_ids = list(dict.fromkeys(str(item) for item in prior_green_od_ids if str(item).strip()))
    replay_od_ids = list(dict.fromkeys(str(item) for item in replay_od_ids if str(item).strip()))
    regression_od_ids = list(dict.fromkeys(str(item) for item in regression_od_ids if str(item).strip()))
    new_od_ids = list(dict.fromkeys(str(item) for item in new_od_ids if str(item).strip()))
    preserved_green_set = {str(item) for item in preserved_green_od_ids if str(item).strip()}
    replay_set = set(replay_od_ids)
    retrying_pending_tranche = bool(last_tranche) and str(last_tranche.get("status") or "").strip().lower() != "green"
    selection_mode = "widen_after_green"
    if bootstrapping_now:
        selection_mode = "bootstrap"
    elif retrying_pending_tranche:
        selection_mode = "retry_pending_tranche"
    replayed_prior_green_od_ids = [od_id for od_id in prior_green_od_ids if od_id in replay_set]
    missing_prior_green_replay_od_ids = [od_id for od_id in prior_green_od_ids if od_id not in replay_set]
    retained_green_od_ids = [od_id for od_id in prior_green_od_ids if od_id in preserved_green_set]
    next_widening_block_reason = ""
    if missing_prior_green_replay_od_ids:
        next_widening_block_reason = "missing_prior_green_replay"
    elif not publication_evidence_ok or tranche_status == "blocked":
        next_widening_block_reason = "publication_evidence_blocked"
    elif regression_red_od_ids:
        next_widening_block_reason = "regression_failed"
    elif new_red_od_ids:
        next_widening_block_reason = "new_batch_failed"
    elif tranche_status != "green":
        next_widening_block_reason = f"tranche_{tranche_status}"
    return {
        "sequential_mode": "retain_green_replay_then_widen",
        "selection_mode": selection_mode,
        "candidate_selection_mode": candidate_selection_mode,
        "candidate_selection_seed": int(candidate_selection_seed),
        "retrying_pending_tranche": retrying_pending_tranche,
        "retry_source_tranche_index": (
            int(last_tranche.get("tranche_index"))
            if retrying_pending_tranche and str(last_tranche.get("tranche_index") or "").strip()
            else None
        ),
        "prior_green_od_ids": prior_green_od_ids,
        "prior_green_od_count": len(prior_green_od_ids),
        "replayed_prior_green_od_ids": replayed_prior_green_od_ids,
        "replayed_prior_green_od_count": len(replayed_prior_green_od_ids),
        "missing_prior_green_replay_od_ids": missing_prior_green_replay_od_ids,
        "missing_prior_green_replay_od_count": len(missing_prior_green_replay_od_ids),
        "retained_green_od_ids": retained_green_od_ids,
        "retained_green_od_count": len(retained_green_od_ids),
        "regression_od_ids": regression_od_ids,
        "regression_od_count": len(regression_od_ids),
        "new_od_ids": new_od_ids,
        "new_od_count": len(new_od_ids),
        "next_widening_allowed": not bool(next_widening_block_reason),
        "next_widening_block_reason": next_widening_block_reason,
    }


def _select_new_od_ids(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    completed_candidate_od_ids: Sequence[str],
    excluded_od_ids: Sequence[str] = (),
    new_od_batch_size: int,
    candidate_selection_mode: str = "corpus_order",
    candidate_selection_seed: int = 0,
) -> list[str]:
    completed = set(str(item) for item in completed_candidate_od_ids)
    excluded = set(str(item) for item in excluded_od_ids)
    eligible: list[str] = []
    for row in candidate_rows:
        od_id = str(row.get("od_id") or "").strip()
        if not od_id or od_id in completed or od_id in excluded:
            continue
        if od_id not in eligible:
            eligible.append(od_id)
    if str(candidate_selection_mode) == "corpus_order":
        ordered = eligible
    elif str(candidate_selection_mode) == "random_seeded":
        original_index = {od_id: index for index, od_id in enumerate(eligible)}
        seed_text = str(int(candidate_selection_seed))
        ordered = sorted(
            eligible,
            key=lambda od_id: (
                hashlib.sha256(f"{seed_text}:{od_id}".encode("utf-8")).hexdigest(),
                original_index[od_id],
            ),
        )
    else:
        raise ValueError(f"unsupported_candidate_selection_mode:{candidate_selection_mode}")
    return ordered[: max(0, int(new_od_batch_size))]


def run_campaign(
    args: argparse.Namespace,
    *,
    shared_client: Any | None = None,
) -> dict[str, Any]:
    campaign_dir = Path(args.campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "campaign_state.json"
    report_path = campaign_dir / "campaign_report.md"
    bootstrap_rows = _normalize_od_rows(
        row for csv_path in list(args.bootstrap_csv or []) for row in _load_csv_rows(Path(csv_path))
    )
    candidate_rows = _normalize_od_rows(_load_csv_rows(Path(args.corpus_csv)))
    state = _load_state(state_path, campaign_id=str(args.campaign_id), corpus_csv=Path(args.corpus_csv))
    if not bool(args.resume) and state_path.exists():
        state = _empty_state(campaign_id=str(args.campaign_id), corpus_csv=Path(args.corpus_csv))
    state["sequential_mode"] = "retain_green_replay_then_widen"
    state["gate_config"] = _gate_config(args)
    state["bootstrap_od_ids"] = [str(row.get("od_id")) for row in bootstrap_rows]
    tranche_outputs: list[dict[str, Any]] = []
    stop_reason = "campaign_complete"

    for tranche_offset in range(int(args.max_tranches)):
        tranche_index = len(list(state.get("tranches") or [])) + 1
        bootstrap_od_ids = [str(item) for item in list(state.get("bootstrap_od_ids") or [])]
        green_od_ids = [str(item) for item in list(state.get("green_od_ids") or [])]
        prior_green_od_ids = list(dict.fromkeys(green_od_ids))
        completed_candidate_od_ids = [str(item) for item in list(state.get("completed_candidate_od_ids") or [])]
        last_tranche = dict(list(state.get("tranches") or [])[-1]) if list(state.get("tranches") or []) else None
        if not bool(state.get("bootstrapped")) and bootstrap_rows:
            regression_od_ids = bootstrap_od_ids
            new_od_ids: list[str] = []
        else:
            regression_od_ids = list(dict.fromkeys(bootstrap_od_ids + green_od_ids))
            if last_tranche and str(last_tranche.get("status") or "").strip().lower() != "green":
                regression_od_ids = list(
                    dict.fromkeys(
                        list(last_tranche.get("replay_od_ids") or [])
                        or list(last_tranche.get("regression_od_ids") or [])
                        or regression_od_ids
                    )
                )
                new_od_ids = list(dict.fromkeys(str(item) for item in list(last_tranche.get("new_od_ids") or [])))
            else:
                new_od_ids = _select_new_od_ids(
                    candidate_rows,
                    completed_candidate_od_ids=completed_candidate_od_ids,
                    excluded_od_ids=regression_od_ids,
                    new_od_batch_size=int(args.new_od_batch_size),
                    candidate_selection_mode=str(args.candidate_selection_mode),
                    candidate_selection_seed=int(args.candidate_selection_seed),
                )
            if not new_od_ids:
                stop_reason = "no_unseen_candidate_ods"
                break
        replay_od_ids = list(regression_od_ids)
        tranche_rows = _ordered_subset_rows(
            bootstrap_rows,
            candidate_rows,
            regression_od_ids=replay_od_ids,
            new_od_ids=new_od_ids,
        )
        if not tranche_rows:
            stop_reason = "empty_tranche"
            break
        tranche_dir = campaign_dir / f"tranche_{tranche_index:03d}"
        tranche_dir.mkdir(parents=True, exist_ok=True)
        tranche_corpus_csv = tranche_dir / "tranche_od_corpus.csv"
        _write_csv(tranche_corpus_csv, tranche_rows)
        run_id = f"{args.campaign_id}_t{tranche_index:03d}"
        bootstrapping_now = (not bool(state.get("bootstrapped"))) and bool(bootstrap_rows)
        route_graph_asset_plan = (
            _build_route_graph_asset_plan(
                corpus_csv=tranche_corpus_csv,
                requested_asset_path=str(getattr(args, "route_graph_asset_path", "") or ""),
                requested_min_nodes=getattr(args, "route_graph_min_nodes", None),
                requested_min_adjacency=getattr(args, "route_graph_min_adjacency", None),
                requested_subset_corridor_km=getattr(args, "route_graph_subset_corridor_km", None),
            )
            if bool(getattr(args, "stage_route_graph_subset", False))
            else None
        )
        eval_args = _build_eval_args(
            args,
            tranche_corpus_csv=tranche_corpus_csv,
            tranche_run_id=run_id,
            tranche_out_dir=tranche_dir,
            route_graph_asset_path_override=(
                str((route_graph_asset_plan or {}).get("asset_path") or "").strip()
                if isinstance(route_graph_asset_plan, Mapping)
                else None
            ),
            route_graph_min_nodes_override=(
                int((route_graph_asset_plan or {}).get("route_graph_min_nodes"))
                if isinstance(route_graph_asset_plan, Mapping)
                and (route_graph_asset_plan or {}).get("route_graph_min_nodes") is not None
                else None
            ),
            route_graph_min_adjacency_override=(
                int((route_graph_asset_plan or {}).get("route_graph_min_adjacency"))
                if isinstance(route_graph_asset_plan, Mapping)
                and (route_graph_asset_plan or {}).get("route_graph_min_adjacency") is not None
                else None
            ),
        )
        eval_args = _apply_route_graph_asset_plan_to_eval_args(eval_args, route_graph_asset_plan)
        if shared_client is not None:
            evaluation_payload = _get_thesis_eval().run_thesis_evaluation(eval_args, client=shared_client)
        else:
            evaluation_payload = _get_thesis_eval().run_thesis_evaluation(eval_args)
        per_od = _evaluate_od_rows(
            list(evaluation_payload.get("rows") or []),
            target_variants=list(args.target_variants or []),
            require_balanced_win=bool(args.require_balanced_win),
            require_win_v0=bool(args.require_win_v0),
            require_dominance_win=bool(args.require_dominance_win),
            require_time_preserving_win=bool(args.require_time_preserving_win),
            min_weighted_margin=float(args.min_weighted_margin),
            min_weighted_margin_v0=float(args.min_weighted_margin_v0),
        )
        expected_od_ids = list(dict.fromkeys(list(regression_od_ids) + list(new_od_ids)))
        per_od, evaluated_od_ids, missing_expected_od_ids = _ensure_expected_od_entries(
            per_od,
            expected_od_ids=expected_od_ids,
            target_variants=list(args.target_variants or []),
        )
        evaluation_evidence = _evaluation_evidence_payload(evaluation_payload)
        proof_grade_readiness = dict(evaluation_evidence.get("proof_grade_readiness") or {})
        proof_grade_required = bool(args.require_proof_grade_readiness)
        proof_grade_readiness_ok = bool(proof_grade_readiness.get("proof_grade_readiness_ok"))
        publication_evidence_ok = bool(evaluation_evidence.get("required_artifacts_ok")) and (
            (not proof_grade_required) or proof_grade_readiness_ok
        )
        regression_set = set(regression_od_ids)
        red_od_ids = sorted(od_id for od_id, payload in per_od.items() if not bool(payload.get("passes_all_targets")))
        regression_red_od_ids = sorted(od_id for od_id in red_od_ids if od_id in regression_set)
        new_red_od_ids = sorted(od_id for od_id in red_od_ids if od_id in set(new_od_ids))
        preserved_green_od_ids = sorted(
            od_id for od_id in regression_od_ids if bool((per_od.get(od_id) or {}).get("passes_all_targets"))
        )
        unpreserved_green_od_ids = sorted(
            od_id for od_id in regression_od_ids if not bool((per_od.get(od_id) or {}).get("passes_all_targets"))
        )
        promoted_green_od_ids = sorted(od_id for od_id in new_od_ids if bool((per_od.get(od_id) or {}).get("passes_all_targets")))
        per_od_path = tranche_dir / "per_od_status.json"
        tranche_status = "green"
        if not publication_evidence_ok:
            tranche_status = "blocked"
        elif regression_red_od_ids:
            tranche_status = "regression"
        elif new_red_od_ids:
            tranche_status = "red"
        sequential_control = _sequential_control_payload(
            bootstrapping_now=bootstrapping_now,
            last_tranche=last_tranche,
            prior_green_od_ids=prior_green_od_ids,
            replay_od_ids=replay_od_ids,
            regression_od_ids=regression_od_ids,
            new_od_ids=new_od_ids,
            tranche_status=tranche_status,
            publication_evidence_ok=publication_evidence_ok,
            regression_red_od_ids=regression_red_od_ids,
            new_red_od_ids=new_red_od_ids,
            preserved_green_od_ids=preserved_green_od_ids,
            candidate_selection_mode=str(args.candidate_selection_mode),
            candidate_selection_seed=int(args.candidate_selection_seed),
        )
        _write_json(
            per_od_path,
            {
                "campaign_id": str(args.campaign_id),
                "tranche_index": tranche_index,
                "generated_at_utc": _now(),
                "gate_config": _gate_config(args),
                "target_variants": list(args.target_variants or []),
                "replay_od_ids": replay_od_ids,
                "replay_od_count": len(replay_od_ids),
                "expected_od_ids": expected_od_ids,
                "evaluated_od_ids": evaluated_od_ids,
                "missing_expected_od_ids": missing_expected_od_ids,
                "publication_evidence": evaluation_evidence,
                "proof_grade_readiness_required": proof_grade_required,
                "route_graph_asset_plan": route_graph_asset_plan,
                "sequential_control": sequential_control,
                "od_status": per_od,
            },
        )
        tranche_payload = {
            "tranche_index": tranche_index,
            "generated_at_utc": _now(),
            "status": tranche_status,
            "run_id": run_id,
            "replay_od_ids": replay_od_ids,
            "replay_od_count": len(replay_od_ids),
            "regression_od_ids": list(regression_od_ids),
            "new_od_ids": list(new_od_ids),
            "expected_od_ids": expected_od_ids,
            "evaluated_od_ids": evaluated_od_ids,
            "missing_expected_od_ids": missing_expected_od_ids,
            "preserved_green_od_ids": preserved_green_od_ids,
            "unpreserved_green_od_ids": unpreserved_green_od_ids,
            "promoted_green_od_ids": promoted_green_od_ids,
            "red_od_ids": red_od_ids,
            "regression_red_od_ids": regression_red_od_ids,
            "new_red_od_ids": new_red_od_ids,
            "tranche_corpus_csv": str(tranche_corpus_csv),
            "per_od_status_json": str(per_od_path),
            "results_csv": str(evaluation_payload.get("results_csv") or ""),
            "summary_csv": str(evaluation_payload.get("summary_csv") or ""),
            "evaluation_manifest": str(evaluation_payload.get("evaluation_manifest") or ""),
            "gate_config": _gate_config(args),
            "publication_evidence_ok": publication_evidence_ok,
            "missing_required_artifacts": list(evaluation_evidence.get("missing_required_artifacts") or []),
            "proof_grade_readiness_ok": proof_grade_readiness_ok,
            "proof_grade_readiness_required": proof_grade_required,
            "proof_grade_readiness_reasons": list(
                proof_grade_readiness.get("proof_grade_readiness_reasons") or []
            ),
            "evaluation_evidence": evaluation_evidence,
            "route_graph_asset_plan": route_graph_asset_plan,
            **sequential_control,
        }
        tranche_outputs.append(tranche_payload)
        state.setdefault("tranches", []).append(tranche_payload)
        if bootstrapping_now:
            state["bootstrapped"] = tranche_status == "green"
        if tranche_status == "green":
            baseline_green_ids = promoted_green_od_ids
            if bootstrapping_now:
                baseline_green_ids = list(regression_od_ids) + promoted_green_od_ids
            merged_green = dict.fromkeys(list(state.get("green_od_ids") or []) + baseline_green_ids)
            state["green_od_ids"] = list(merged_green.keys())
            merged_completed = dict.fromkeys(
                list(state.get("completed_candidate_od_ids") or []) + list(new_od_ids)
            )
            state["completed_candidate_od_ids"] = list(merged_completed.keys())
        elif tranche_status in {"regression", "red"}:
            merged_red = dict.fromkeys(list(state.get("red_od_ids") or []) + red_od_ids)
            state["red_od_ids"] = list(merged_red.keys())
        state["updated_at_utc"] = _now()
        _write_json(state_path, state)
        report_path.write_text(
            _campaign_markdown(state=state, last_tranche=tranche_payload),
            encoding="utf-8",
        )
        if tranche_status != "green" and bool(args.stop_on_red_tranche):
            stop_reason = f"stop_on_{tranche_status}_tranche"
            break
        if tranche_offset + 1 >= int(args.max_tranches):
            stop_reason = "max_tranches_reached"

    final_payload = {
        "campaign_id": str(args.campaign_id),
        "campaign_dir": str(campaign_dir),
        "campaign_state_json": str(state_path),
        "campaign_report_md": str(report_path),
        "stop_reason": stop_reason,
        "tranche_count_executed": len(tranche_outputs),
        "state": state,
        "tranches": tranche_outputs,
    }
    _write_json(campaign_dir / "campaign_result.json", final_payload)
    return final_payload


def _apply_evaluation_fast_startup_env(*, enabled: bool) -> dict[str, str]:
    if not enabled:
        return {}
    updates = {
        "ROUTE_GRAPH_EVALUATION_FAST_STARTUP_ALLOWED": "true",
        "ROUTE_GRAPH_FAST_STARTUP_ENABLED": "true",
    }
    for key, value in updates.items():
        os.environ[key] = value
    return updates


def _apply_evaluation_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    updates = _apply_evaluation_fast_startup_env(enabled=bool(args.evaluation_fast_startup))
    settings_module = sys.modules.get("app.settings")
    runtime_settings = getattr(settings_module, "settings", None) if settings_module is not None else None
    if runtime_settings is not None:
        if "ROUTE_GRAPH_EVALUATION_FAST_STARTUP_ALLOWED" in updates and hasattr(
            runtime_settings,
            "route_graph_evaluation_fast_startup_allowed",
        ):
            runtime_settings.route_graph_evaluation_fast_startup_allowed = True
        if "ROUTE_GRAPH_FAST_STARTUP_ENABLED" in updates and hasattr(
            runtime_settings,
            "route_graph_fast_startup_enabled",
        ):
            runtime_settings.route_graph_fast_startup_enabled = True
    asset_path = str(getattr(args, "route_graph_asset_path", "") or "").strip()
    if asset_path:
        resolved_asset_path = str(Path(asset_path).expanduser().resolve())
        updates["ROUTE_GRAPH_ASSET_PATH"] = resolved_asset_path
        if runtime_settings is not None and hasattr(runtime_settings, "route_graph_asset_path"):
            runtime_settings.route_graph_asset_path = resolved_asset_path
    min_nodes = getattr(args, "route_graph_min_nodes", None)
    if min_nodes is not None:
        resolved_min_nodes = str(max(1, int(min_nodes)))
        updates["ROUTE_GRAPH_MIN_NODES"] = resolved_min_nodes
        if runtime_settings is not None and hasattr(runtime_settings, "route_graph_min_nodes"):
            runtime_settings.route_graph_min_nodes = int(resolved_min_nodes)
    min_adjacency = getattr(args, "route_graph_min_adjacency", None)
    if min_adjacency is not None:
        resolved_min_adjacency = str(max(1, int(min_adjacency)))
        updates["ROUTE_GRAPH_MIN_ADJACENCY"] = resolved_min_adjacency
        if runtime_settings is not None and hasattr(runtime_settings, "route_graph_min_adjacency"):
            runtime_settings.route_graph_min_adjacency = int(resolved_min_adjacency)
    for key, value in updates.items():
        os.environ[key] = value
    return updates


def _prepare_in_process_runtime_env(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "stage_route_graph_subset", False)):
        return {
            "env_updates": _apply_evaluation_runtime_env(args),
            "route_graph_asset_plan": None,
            "preview_tranche_corpus_csv": None,
        }
    if bool(getattr(args, "resume", False)) or int(getattr(args, "max_tranches", 1)) != 1:
        raise RuntimeError("in_process_backend_staged_subset_requires_single_non_resume_tranche")

    bootstrap_rows = _normalize_od_rows(
        row for csv_path in list(args.bootstrap_csv or []) for row in _load_csv_rows(Path(csv_path))
    )
    candidate_rows = _normalize_od_rows(_load_csv_rows(Path(args.corpus_csv)))
    regression_od_ids = [str(row.get("od_id")) for row in bootstrap_rows]
    if bootstrap_rows:
        new_od_ids: list[str] = []
    else:
        new_od_ids = _select_new_od_ids(
            candidate_rows,
            completed_candidate_od_ids=[],
            excluded_od_ids=regression_od_ids,
            new_od_batch_size=int(args.new_od_batch_size),
            candidate_selection_mode=str(args.candidate_selection_mode),
            candidate_selection_seed=int(args.candidate_selection_seed),
        )
    tranche_rows = _ordered_subset_rows(
        bootstrap_rows,
        candidate_rows,
        regression_od_ids=regression_od_ids,
        new_od_ids=new_od_ids,
    )
    if not tranche_rows:
        raise RuntimeError("in_process_backend_empty_initial_tranche")

    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    tranche_dir = campaign_dir / "tranche_001"
    tranche_dir.mkdir(parents=True, exist_ok=True)
    tranche_corpus_csv = tranche_dir / "tranche_od_corpus.csv"
    _write_csv(tranche_corpus_csv, tranche_rows)
    route_graph_asset_plan = _build_route_graph_asset_plan(
        corpus_csv=tranche_corpus_csv,
        requested_asset_path=str(getattr(args, "route_graph_asset_path", "") or ""),
        requested_min_nodes=getattr(args, "route_graph_min_nodes", None),
        requested_min_adjacency=getattr(args, "route_graph_min_adjacency", None),
        requested_subset_corridor_km=getattr(args, "route_graph_subset_corridor_km", None),
    )
    if isinstance(route_graph_asset_plan, Mapping):
        asset_path = str(route_graph_asset_plan.get("asset_path") or "").strip()
        if asset_path:
            setattr(args, "route_graph_asset_path", asset_path)
        min_nodes = route_graph_asset_plan.get("route_graph_min_nodes")
        if min_nodes is not None:
            setattr(args, "route_graph_min_nodes", max(1, int(min_nodes)))
        min_adjacency = route_graph_asset_plan.get("route_graph_min_adjacency")
        if min_adjacency is not None:
            setattr(args, "route_graph_min_adjacency", max(1, int(min_adjacency)))
    return {
        "env_updates": _apply_evaluation_runtime_env(args),
        "route_graph_asset_plan": dict(route_graph_asset_plan) if isinstance(route_graph_asset_plan, Mapping) else None,
        "preview_tranche_corpus_csv": str(tranche_corpus_csv),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evaluation_args = list(args.evaluation_args or [])
    use_in_process = "--in-process-backend" in evaluation_args
    if use_in_process:
        _prepare_in_process_runtime_env(args)
        from app.main import app

        with _get_thesis_eval().TestClient(app) as client:
            run_campaign(args, shared_client=client)
        return 0
    run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
