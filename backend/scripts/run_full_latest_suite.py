from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.run_store import artifact_dir_for_run, write_csv_artifact, write_json_artifact, write_manifest, write_text_artifact
from app.settings import settings
from scripts.build_od_corpus_uk import _parse_bbox, _select_rows_for_corpus, build_dual_od_corpora
from scripts.preflight_live_runtime import run_preflight
from scripts.run_hot_rerun_benchmark import _build_parser as _build_hot_parser, run_hot_rerun_benchmark
from scripts.run_thesis_evaluation import (
    EVALUATION_SUITE_ROLE_DEFAULTS,
    LANE_METADATA_DEFAULTS,
    _fallback_support_richness,
    _build_parser as _build_eval_parser,
    in_process_backend_runtime_profile,
    run_thesis_evaluation,
)

HEADLINE_ROLES: tuple[str, ...] = (
    "broad_cold_proof",
    "focused_refc_proof",
    "focused_voi_proof",
)
HEADLINE_ADOPTION_ROLES: tuple[str, ...] = ("broad_cold_proof",)
DIRECT_SUITE_ROLES: tuple[str, ...] = tuple(
    role
    for role in EVALUATION_SUITE_ROLE_DEFAULTS
    if role not in {"generic_evaluation", "hot_rerun_cold_source", "hot_rerun"}
)
FOCUSED_ROLES: frozenset[str] = frozenset(
    {
        "focused_refc_proof",
        "focused_voi_proof",
        "dccs_diagnostic_probe",
        "preference_proof",
        "perturbation_flip_radius",
    }
)
BROAD_ROLES: frozenset[str] = frozenset(
    {
        "broad_cold_proof",
        "optional_stopping_coverage",
        "proxy_audit_calibration",
        "public_transfer",
    }
)
FAILURE_ATLAS_LANE_ID = "failure_atlas"
FAILURE_ATLAS_LANE_LABEL = "failure atlas"
FAILURE_ATLAS_REQUIRED_KINDS: tuple[str, ...] = (
    "wrong_singleton",
    "support_downgrade",
    "abstention",
)
FAILURE_ATLAS_OPTIONAL_KINDS: tuple[str, ...] = (
    "certified_set_violation",
    "route_failure",
)
FAILURE_ATLAS_ROOT_CAUSE_FAMILIES: tuple[str, ...] = (
    "support_failure",
    "hidden_challenger",
    "proxy_bias",
    "preference_ambiguity",
    "budget_cut",
    "other",
)
FAILURE_ATLAS_ABSTENTION_EXAMPLE_TARGET = 5
SUITE_SCHEMA_VERSION = "full-latest-suite-v1"
PIPELINE_VARIANT_COUNT = 4
DCCS_PUBLISHABILITY_ROLES: frozenset[str] = frozenset({"broad_cold_proof", "dccs_diagnostic_probe"})
VOI_PUBLISHABILITY_ROWS: frozenset[tuple[str, str]] = frozenset(
    {
        ("broad_cold_proof", "C"),
        ("focused_voi_proof", "C"),
    }
)
FOCUSED_REFINEMENT_ROLES: frozenset[str] = frozenset({"focused_voi_proof"})
OPTIONAL_STOPPING_PROOF_VARIANTS: frozenset[str] = frozenset({"B", "C"})
PERTURBATION_PROOF_VARIANTS: frozenset[str] = frozenset({"B", "C"})
OPTIONAL_STOPPING_ANYTIME_METHODS: frozenset[str] = frozenset({"anytime_hoeffding_union_bound"})
OPTIONAL_STOPPING_REQUIRED_COVERAGE_FLOOR = 0.94
PERTURBATION_REQUIRED_REAL_ROW_MINIMUM = 30
PERTURBATION_REQUIRED_EXACT_SYNTHETIC_MINIMUM = 500


def _minimum_od_count_for_row_gate(minimum_rows: int, *, variants_per_od: int = PIPELINE_VARIANT_COUNT) -> int:
    variants = max(1, int(variants_per_od))
    rows = max(1, int(minimum_rows))
    return int(math.ceil(rows / float(variants)))


# These corpus counts are OD/request counts, not post-variant evaluator row counts.
# The suite evaluates each OD across the V0/A/B/C ladder, so the minimum OD counts
# should be derived from the row-count gates instead of matching them 1:1.
DEFAULT_BROAD_COUNT = _minimum_od_count_for_row_gate(200)
DEFAULT_FOCUSED_COUNT = _minimum_od_count_for_row_gate(60)
DEFAULT_TRANSFER_COUNT = _minimum_od_count_for_row_gate(50)
DEFAULT_SYNTHETIC_COUNT = _minimum_od_count_for_row_gate(1000)
DEFAULT_OPTIONAL_STOPPING_COUNT = _minimum_od_count_for_row_gate(30000)
DEFAULT_HEADLINE_SEED_REPEAT_COUNT = 3
DEFAULT_HEADLINE_SEED_REPEAT_STEP = 101
SUPPORT_FRAGILE_SOURCE_CSV = PROJECT_ROOT / "data" / "eval" / "uk_od_corpus_representative_expanded.csv"
SUPPORT_FRAGILE_THRESHOLD = 0.45
CURATED_CORPUS_MINIMUM_ROWS: dict[str, int] = {
    "broad": DEFAULT_BROAD_COUNT,
    "focused": DEFAULT_FOCUSED_COUNT,
    "transfer": DEFAULT_TRANSFER_COUNT,
    "synthetic": DEFAULT_SYNTHETIC_COUNT,
}
CURATED_BASE_POOL_CSV = PROJECT_ROOT / "out" / "thesis_corpus" / "uk_od_corpus_seq02_combined_1204.csv"


@dataclass(frozen=True)
class CorpusArtifact:
    key: str
    label: str
    row_count: int
    csv_path: str
    json_path: str
    summary_path: str
    source_summary_path: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run_label() -> str:
    return datetime.now(UTC).strftime("full_latest_suite_%Y%m%d_%H%M%S")


@lru_cache(maxsize=1)
def _evaluation_defaults() -> argparse.Namespace:
    return _build_eval_parser().parse_args(["--corpus-csv", "placeholder.csv"])


@lru_cache(maxsize=1)
def _hot_runner_defaults() -> argparse.Namespace:
    return _build_hot_parser().parse_args(["--corpus-csv", "placeholder.csv"])


def _build_parser() -> argparse.ArgumentParser:
    eval_defaults = _evaluation_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Build fresh latest corpora and run the full thesis evaluation suite plus hot rerun "
            "into one aggregated reviewer-facing bundle."
        )
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", default=str(settings.out_dir))
    parser.add_argument("--seed", type=int, default=int(getattr(eval_defaults, "seed", 20260320)))
    parser.add_argument("--bbox", default=str(settings.terrain_uk_bbox))
    parser.add_argument("--broad-count", type=int, default=DEFAULT_BROAD_COUNT)
    parser.add_argument("--focused-count", type=int, default=DEFAULT_FOCUSED_COUNT)
    parser.add_argument("--transfer-count", type=int, default=DEFAULT_TRANSFER_COUNT)
    parser.add_argument("--synthetic-count", type=int, default=DEFAULT_SYNTHETIC_COUNT)
    parser.add_argument("--optional-stopping-count", type=int, default=DEFAULT_OPTIONAL_STOPPING_COUNT)
    parser.set_defaults(use_curated_corpora=True)
    parser.add_argument(
        "--use-curated-corpora",
        dest="use_curated_corpora",
        action="store_true",
        help="Prefer maintained curated/base-pool corpora before falling back to fresh route-graph corpus generation.",
    )
    parser.add_argument(
        "--generate-corpora",
        dest="use_curated_corpora",
        action="store_false",
        help="Force fresh route-graph corpus generation instead of reusing curated/base-pool corpora.",
    )
    parser.add_argument("--broad-corpus-csv", default=None)
    parser.add_argument("--focused-corpus-csv", default=None)
    parser.add_argument("--transfer-corpus-csv", default=None)
    parser.add_argument("--synthetic-corpus-csv", default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--probe-max-paths", type=int, default=6)
    parser.add_argument(
        "--headline-seed-repeat-count",
        type=int,
        default=DEFAULT_HEADLINE_SEED_REPEAT_COUNT,
    )
    parser.add_argument(
        "--headline-seed-repeat-step",
        type=int,
        default=DEFAULT_HEADLINE_SEED_REPEAT_STEP,
    )
    parser.add_argument("--backend-url", default=str(getattr(eval_defaults, "backend_url", "http://localhost:8000")))
    parser.add_argument("--ready-timeout-seconds", type=float, default=float(getattr(eval_defaults, "ready_timeout_seconds", 1800.0)))
    parser.add_argument("--ready-poll-seconds", type=float, default=float(getattr(eval_defaults, "ready_poll_seconds", 5.0)))
    parser.add_argument("--route-timeout-seconds", type=float, default=float(getattr(eval_defaults, "route_timeout_seconds", 600.0)))
    parser.set_defaults(in_process_backend=True)
    parser.add_argument(
        "--in-process-backend",
        dest="in_process_backend",
        action="store_true",
        help="Run evaluations against an in-process FastAPI app instance.",
    )
    parser.add_argument(
        "--live-backend",
        dest="in_process_backend",
        action="store_false",
        help="Run evaluations against --backend-url instead of an in-process backend.",
    )
    parser.add_argument("--model-version", default=str(getattr(eval_defaults, "model_version", "thesis-script-v3")))
    parser.add_argument("--optimization-mode", default=str(getattr(eval_defaults, "optimization_mode", "expected_value")))
    parser.add_argument("--vehicle-type", default=str(getattr(eval_defaults, "vehicle_type", "rigid_hgv")))
    parser.add_argument("--scenario-mode", default=str(getattr(eval_defaults, "scenario_mode", "no_sharing")))
    parser.add_argument("--departure-time-utc", default=getattr(eval_defaults, "departure_time_utc", None))
    parser.add_argument("--max-alternatives", type=int, default=int(getattr(eval_defaults, "max_alternatives", 8)))
    parser.add_argument("--search-budget", type=int, default=int(getattr(eval_defaults, "search_budget", 4)))
    parser.add_argument("--evidence-budget", type=int, default=int(getattr(eval_defaults, "evidence_budget", 2)))
    parser.add_argument("--world-count", type=int, default=int(getattr(eval_defaults, "world_count", 64)))
    parser.add_argument("--certificate-threshold", type=float, default=float(getattr(eval_defaults, "certificate_threshold", 0.80)))
    parser.add_argument("--tau-stop", type=float, default=float(getattr(eval_defaults, "tau_stop", 0.02)))
    parser.add_argument("--stochastic-enabled", action=argparse.BooleanOptionalAction, default=bool(getattr(eval_defaults, "stochastic_enabled", False)))
    parser.add_argument("--stochastic-samples", type=int, default=int(getattr(eval_defaults, "stochastic_samples", 25)))
    parser.add_argument("--weight-time", type=float, default=float(getattr(eval_defaults, "weight_time", 1.0)))
    parser.add_argument("--weight-money", type=float, default=float(getattr(eval_defaults, "weight_money", 1.0)))
    parser.add_argument("--weight-co2", type=float, default=float(getattr(eval_defaults, "weight_co2", 1.0)))
    parser.add_argument("--fail-soft", action=argparse.BooleanOptionalAction, default=bool(getattr(eval_defaults, "fail_soft", True)))
    parser.add_argument("--disable-tolls", action="store_true", default=bool(getattr(eval_defaults, "disable_tolls", False)))
    parser.add_argument(
        "--baseline-refinement-policy",
        default=str(getattr(eval_defaults, "baseline_refinement_policy", "corridor_uniform")),
    )
    parser.add_argument(
        "--ors-baseline-policy",
        default=str(getattr(eval_defaults, "ors_baseline_policy", "local_service")),
    )
    parser.add_argument(
        "--ors-snapshot-mode",
        default=str(getattr(eval_defaults, "ors_snapshot_mode", "off")),
    )
    parser.add_argument("--ors-snapshot-path", default=getattr(eval_defaults, "ors_snapshot_path", None))
    parser.add_argument(
        "--auto-enrich-corpus-ambiguity",
        action="store_true",
        default=bool(getattr(eval_defaults, "auto_enrich_corpus_ambiguity", False)),
    )
    parser.add_argument("--allow-proxy-ors", action="store_true", default=bool(getattr(eval_defaults, "allow_proxy_ors", False)))
    parser.add_argument(
        "--allow-evidence-fallbacks",
        action="store_true",
        default=bool(getattr(eval_defaults, "allow_evidence_fallbacks", False)),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _ordered_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            text = str(key)
            if text in seen:
                continue
            seen.add(text)
            fieldnames.append(text)
    return fieldnames or ["od_id"]


def _summary_counts(rows: Sequence[Mapping[str, Any]], *, key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        token = str(row.get(key) or "").strip()
        if token:
            counts[token] += 1
    return dict(sorted(counts.items()))


def _subset_corpus_summary(
    *,
    source_summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
    corpus_kind: str,
    selection_policy: str,
) -> dict[str, Any]:
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "created_at_utc": _now(),
        "corpus_kind": corpus_kind,
        "selection_policy": selection_policy,
        "source_pool_hash": source_summary.get("source_pool", {}).get("corpus_hash")
        if isinstance(source_summary.get("source_pool"), Mapping)
        else source_summary.get("source_pool_hash"),
        "source_pool_count": _safe_int(
            (
                source_summary.get("source_pool", {}).get("accepted_count")
                if isinstance(source_summary.get("source_pool"), Mapping)
                else source_summary.get("source_pool_count")
            ),
            len(rows),
        ),
        "selected_count": len(rows),
        "accepted_by_bin": _summary_counts(rows, key="distance_bin"),
        "accepted_by_corridor": _summary_counts(rows, key="corridor_bucket"),
        "rows": rows,
    }


def _focused_support_fragile_target(count: int) -> int:
    target = max(0, int(count))
    if target <= 0:
        return 0
    return min(target, max(1, target // 4))


def _focused_support_source_mix_strength(row: Mapping[str, Any]) -> float:
    explicit_count = _safe_float(row.get("od_ambiguity_source_mix_count"))
    if explicit_count is not None:
        return max(0.0, min(1.0, explicit_count / 3.0))
    raw_mix = row.get("od_ambiguity_source_mix")
    if isinstance(raw_mix, str):
        text = raw_mix.strip()
        if not text:
            return 0.0
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            return max(0.0, min(1.0, len(payload) / 3.0))
        tokens = [token.strip() for token in text.split(",") if token.strip()]
        return max(0.0, min(1.0, len(tokens) / 3.0))
    return 0.0


def _focused_support_richness(row: Mapping[str, Any]) -> float | None:
    support_strength = _safe_float(row.get("od_ambiguity_source_support_strength"))
    support_ratio = _safe_float(row.get("od_ambiguity_support_ratio"))
    source_entropy = _safe_float(row.get("od_ambiguity_source_entropy"))
    prior_strength = _safe_float(row.get("od_ambiguity_prior_strength"))
    confidence = _safe_float(row.get("od_ambiguity_confidence"))
    source_count_strength = max(
        0.0,
        min(1.0, (_safe_float(row.get("od_ambiguity_source_count")) or 0.0) / 4.0),
    )
    source_mix_strength = _focused_support_source_mix_strength(row)
    weakest_support = min(
        support_strength if support_strength is not None else 1.0,
        support_ratio if support_ratio is not None else 1.0,
    )
    terms = [
        0.55 * max(0.0, min(1.0, weakest_support)),
        0.15 * max(0.0, min(1.0, 1.0 - (source_entropy if source_entropy is not None else 1.0))),
        0.10 * max(0.0, min(1.0, prior_strength if prior_strength is not None else 0.0)),
        0.10 * max(0.0, min(1.0, confidence if confidence is not None else 0.0)),
        0.05 * source_count_strength,
        0.05 * source_mix_strength,
    ]
    if not any(term > 0.0 for term in terms):
        return None
    return round(max(0.0, min(1.0, sum(terms))), 6)


def _focused_support_bin(support_richness: float | None) -> str:
    if support_richness is None or not math.isfinite(float(support_richness)):
        return "unknown_support"
    if support_richness <= SUPPORT_FRAGILE_THRESHOLD:
        return "weak_support"
    if support_richness >= 0.75:
        return "strong_support"
    return "mid_support"


def _annotated_support_fragile_row(row: Mapping[str, Any], *, support_richness: float) -> dict[str, Any]:
    annotated = dict(row)
    source_group = str(row.get("corpus_group") or row.get("corpus_kind") or "").strip()
    if source_group:
        annotated.setdefault("source_corpus_group", source_group)
    annotated["support_richness"] = round(float(support_richness), 6)
    annotated["support_bin"] = _focused_support_bin(support_richness)
    annotated["corpus_group"] = "support_fragile"
    annotated["corpus_kind"] = "support_fragile"
    annotated["support_selection_reason"] = "focused_suite_support_fragile_slice"
    return annotated


def _mix_focused_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    support_fragile_rows: Sequence[Mapping[str, Any]],
    *,
    target: int,
) -> list[dict[str, Any]]:
    row_target = max(0, int(target))
    if row_target <= 0:
        return []
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    support_target = min(len(list(support_fragile_rows)), _focused_support_fragile_target(row_target))
    primary_target = max(0, row_target - support_target)

    def _append_row(row: Mapping[str, Any]) -> None:
        if len(selected) >= row_target:
            return
        row_id = str(row.get("od_id") or "").strip()
        if row_id and row_id in selected_ids:
            return
        payload = dict(row)
        selected.append(payload)
        if row_id:
            selected_ids.add(row_id)

    for row in list(primary_rows)[:primary_target]:
        _append_row(row)
    for row in list(support_fragile_rows)[:support_target]:
        _append_row(row)
    if len(selected) < row_target:
        for row in list(primary_rows)[primary_target:] + list(support_fragile_rows)[support_target:]:
            _append_row(row)
            if len(selected) >= row_target:
                break
    return selected[:row_target]


def _rows_canonical_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_curated_base_pool_corpora(
    args: argparse.Namespace,
    *,
    suite_run_id: str,
    requested_minimums: Mapping[str, int],
) -> dict[str, CorpusArtifact] | None:
    if not CURATED_BASE_POOL_CSV.exists():
        return None

    pool_rows = _read_csv_rows(CURATED_BASE_POOL_CSV)
    if len(pool_rows) < max(int(value) for value in requested_minimums.values()):
        return None

    representative_target = max(int(requested_minimums["broad"]), int(requested_minimums["synthetic"]))
    focused_target = int(requested_minimums["focused"])
    transfer_target = int(requested_minimums["transfer"])

    representative_rows = _select_rows_for_corpus(
        pool_rows,
        count=representative_target,
        corpus_kind="representative",
    )
    focused_rows = _select_rows_for_corpus(
        pool_rows,
        count=focused_target,
        corpus_kind="ambiguous",
    )
    transfer_rows = _select_rows_for_corpus(
        pool_rows,
        count=transfer_target,
        corpus_kind="representative",
    )
    if (
        len(representative_rows) < representative_target
        or len(focused_rows) < focused_target
        or len(transfer_rows) < transfer_target
    ):
        return None

    base_summary_payload = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "created_at_utc": _now(),
        "corpus_kind": "curated_base_pool",
        "selection_policy": "curated_base_pool_source_pool",
        "source_path": str(CURATED_BASE_POOL_CSV),
        "source_pool": {
            "corpus_hash": _rows_canonical_hash(pool_rows),
            "accepted_count": len(pool_rows),
        },
        "selected_count": len(pool_rows),
        "accepted_by_bin": _summary_counts(pool_rows, key="distance_bin"),
        "accepted_by_corridor": _summary_counts(pool_rows, key="corridor_bucket"),
        "rows": pool_rows,
    }
    base_summary_path = write_json_artifact(
        suite_run_id,
        "latest_corpus_curated_base_pool.summary.json",
        base_summary_payload,
    )
    base_source_summary: dict[str, Any] = {
        "source_pool": base_summary_payload["source_pool"],
        "source_path": str(CURATED_BASE_POOL_CSV),
    }
    broad_count = int(args.broad_count)
    synthetic_count = int(args.synthetic_count)
    focused_count = int(args.focused_count)
    transfer_count = int(args.transfer_count)
    broad_rows = representative_rows[:broad_count]
    synthetic_rows = representative_rows[:synthetic_count]
    focused_support_rows = _support_fragile_rows(
        _read_csv_rows(SUPPORT_FRAGILE_SOURCE_CSV),
        count=_focused_support_fragile_target(focused_count),
    )
    focused_rows = _mix_focused_rows(
        focused_rows[:focused_count],
        focused_support_rows,
        target=focused_count,
    )
    transfer_rows = transfer_rows[:transfer_count]
    return {
        "broad": _persist_corpus(
            run_id=suite_run_id,
            artifact_prefix="latest_corpus_broad",
            label="Broad curated-base-pool latest corpus",
            rows=broad_rows,
            summary_payload=_subset_corpus_summary(
                source_summary=base_source_summary,
                rows=broad_rows,
                corpus_kind="representative_broad",
                selection_policy="curated_base_pool_representative_slice",
            ),
            source_summary_path=str(base_summary_path),
        ),
        "focused": _persist_corpus(
            run_id=suite_run_id,
            artifact_prefix="latest_corpus_focused",
            label="Focused curated-base-pool latest corpus",
            rows=focused_rows,
            summary_payload=_subset_corpus_summary(
                source_summary=base_source_summary,
                rows=focused_rows,
                corpus_kind="focused_mixed",
                selection_policy="curated_base_pool_focused_mixed_slice",
            ),
            source_summary_path=str(base_summary_path),
        ),
        "synthetic": _persist_corpus(
            run_id=suite_run_id,
            artifact_prefix="latest_corpus_synthetic",
            label="Synthetic curated-base-pool latest corpus",
            rows=synthetic_rows,
            summary_payload=_subset_corpus_summary(
                source_summary=base_source_summary,
                rows=synthetic_rows,
                corpus_kind="synthetic_ground_truth",
                selection_policy="curated_base_pool_representative_slice",
            ),
            source_summary_path=str(base_summary_path),
        ),
        "transfer": _persist_corpus(
            run_id=suite_run_id,
            artifact_prefix="latest_corpus_transfer",
            label="Transfer curated-base-pool latest corpus",
            rows=transfer_rows,
            summary_payload=_subset_corpus_summary(
                source_summary=base_source_summary,
                rows=transfer_rows,
                corpus_kind="public_transfer",
                selection_policy="curated_base_pool_transfer_slice",
            ),
            source_summary_path=str(base_summary_path),
        ),
    }


def _persist_corpus(
    *,
    run_id: str,
    artifact_prefix: str,
    label: str,
    rows: list[dict[str, Any]],
    summary_payload: Mapping[str, Any],
    source_summary_path: str,
) -> CorpusArtifact:
    csv_path = write_csv_artifact(
        run_id,
        f"{artifact_prefix}.csv",
        fieldnames=_ordered_fieldnames(rows),
        rows=rows,
    )
    json_path = write_json_artifact(run_id, f"{artifact_prefix}.json", rows)
    summary_path = write_json_artifact(run_id, f"{artifact_prefix}.summary.json", dict(summary_payload))
    return CorpusArtifact(
        key=artifact_prefix,
        label=label,
        row_count=len(rows),
        csv_path=str(csv_path),
        json_path=str(json_path),
        summary_path=str(summary_path),
        source_summary_path=source_summary_path,
    )


def _resolve_max_attempts(args: argparse.Namespace) -> int:
    if args.max_attempts is not None:
        return max(1, int(args.max_attempts))
    pool_target = max(int(args.synthetic_count), int(args.broad_count)) + int(args.focused_count)
    return max(5000, pool_target * 20)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _existing_corpus_artifact(
    *,
    suite_run_id: str,
    artifact_prefix: str,
    label: str,
    csv_path: Path,
) -> CorpusArtifact:
    rows = _read_csv_rows(csv_path)
    summary_payload = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "created_at_utc": _now(),
        "corpus_kind": artifact_prefix,
        "selection_policy": "reused_curated_corpus",
        "selected_count": len(rows),
        "accepted_by_bin": _summary_counts(rows, key="distance_bin"),
        "accepted_by_corridor": _summary_counts(rows, key="corridor_bucket"),
        "rows": rows,
        "source_path": str(csv_path),
    }
    json_path = write_json_artifact(suite_run_id, f"{artifact_prefix}.json", rows)
    summary_path = write_json_artifact(suite_run_id, f"{artifact_prefix}.summary.json", summary_payload)
    return CorpusArtifact(
        key=artifact_prefix.replace("latest_corpus_", ""),
        label=label,
        row_count=len(rows),
        csv_path=str(csv_path),
        json_path=str(json_path),
        summary_path=str(summary_path),
        source_summary_path=str(csv_path),
    )


def _build_generated_corpora(args: argparse.Namespace, *, suite_run_id: str) -> dict[str, CorpusArtifact]:
    representative_target = max(
        CURATED_CORPUS_MINIMUM_ROWS["broad"],
        int(args.broad_count),
        CURATED_CORPUS_MINIMUM_ROWS["synthetic"],
        int(args.synthetic_count),
    )
    focused_target = max(CURATED_CORPUS_MINIMUM_ROWS["focused"], int(args.focused_count))
    transfer_target = max(CURATED_CORPUS_MINIMUM_ROWS["transfer"], int(args.transfer_count))
    dual_bundle = build_dual_od_corpora(
        seed=int(args.seed),
        representative_count=representative_target,
        ambiguous_count=focused_target,
        bbox=_parse_bbox(args.bbox),
        max_attempts=_resolve_max_attempts(args),
        acceptance_mode="graph_candidates",
        probe_max_paths=max(2, int(args.probe_max_paths)),
    )
    representative_summary = dict(dual_bundle["representative"])
    ambiguous_summary = dict(dual_bundle["ambiguous"])
    representative_rows = [dict(row) for row in representative_summary.get("rows", [])]
    ambiguous_rows = [dict(row) for row in ambiguous_summary.get("rows", [])]
    support_fragile_rows = _support_fragile_rows(
        _read_csv_rows(SUPPORT_FRAGILE_SOURCE_CSV),
        count=_focused_support_fragile_target(focused_target),
    )
    if len(representative_rows) < representative_target:
        raise RuntimeError(
            f"full_latest_suite_representative_shortfall:{len(representative_rows)}<{representative_target}"
        )
    if len(ambiguous_rows) < focused_target:
        raise RuntimeError(f"full_latest_suite_ambiguous_shortfall:{len(ambiguous_rows)}<{focused_target}")
    if not support_fragile_rows:
        raise RuntimeError("full_latest_suite_support_fragile_shortfall:0")

    transfer_bundle = build_dual_od_corpora(
        seed=int(args.seed) + 1009,
        representative_count=transfer_target,
        ambiguous_count=max(20, min(transfer_target, focused_target)),
        bbox=_parse_bbox(args.bbox),
        max_attempts=max(_resolve_max_attempts(args), transfer_target * 24),
        acceptance_mode="graph_candidates",
        probe_max_paths=max(2, int(args.probe_max_paths)),
    )
    transfer_summary = dict(transfer_bundle["representative"])
    transfer_rows = [dict(row) for row in transfer_summary.get("rows", [])]
    if len(transfer_rows) < transfer_target:
        raise RuntimeError(f"full_latest_suite_transfer_shortfall:{len(transfer_rows)}<{transfer_target}")

    source_pool_path = write_json_artifact(suite_run_id, "latest_corpus_source_pool.summary.json", dual_bundle)
    transfer_source_pool_path = write_json_artifact(
        suite_run_id,
        "latest_corpus_transfer_source_pool.summary.json",
        transfer_bundle,
    )
    broad_rows = representative_rows[: int(args.broad_count)]
    focused_rows = _mix_focused_rows(
        ambiguous_rows,
        support_fragile_rows,
        target=focused_target,
    )
    synthetic_rows = representative_rows[: int(args.synthetic_count)]
    return {
        "broad": _persist_corpus(
            run_id=suite_run_id,
            artifact_prefix="latest_corpus_broad",
            label="Broad representative latest corpus",
            rows=broad_rows,
            summary_payload=_subset_corpus_summary(
                source_summary=representative_summary,
                rows=broad_rows,
                corpus_kind="representative_broad",
                selection_policy="fresh_representative_prefix",
            ),
            source_summary_path=str(source_pool_path),
        ),
        "focused": _persist_corpus(
            run_id=suite_run_id,
            artifact_prefix="latest_corpus_focused",
            label="Focused ambiguity-heavy latest corpus",
            rows=focused_rows,
            summary_payload=_subset_corpus_summary(
                source_summary=ambiguous_summary,
                rows=focused_rows,
                corpus_kind="focused_mixed",
                selection_policy="fresh_focused_mixed_slice",
            ),
            source_summary_path=str(source_pool_path),
        ),
        "synthetic": _persist_corpus(
            run_id=suite_run_id,
            artifact_prefix="latest_corpus_synthetic",
            label="Synthetic-lane latest corpus",
            rows=synthetic_rows,
            summary_payload=_subset_corpus_summary(
                source_summary=representative_summary,
                rows=synthetic_rows,
                corpus_kind="synthetic_ground_truth",
                selection_policy="fresh_representative_full_slice",
            ),
            source_summary_path=str(source_pool_path),
        ),
        "transfer": _persist_corpus(
            run_id=suite_run_id,
            artifact_prefix="latest_corpus_transfer",
            label="Transfer latest corpus",
            rows=transfer_rows[:transfer_target],
            summary_payload=_subset_corpus_summary(
                source_summary=transfer_summary,
                rows=transfer_rows[:transfer_target],
                corpus_kind="public_transfer",
                selection_policy="fresh_transfer_representative_prefix",
            ),
            source_summary_path=str(transfer_source_pool_path),
        ),
    }


def _build_optional_stopping_corpus(args: argparse.Namespace, *, suite_run_id: str) -> CorpusArtifact:
    optional_target = max(DEFAULT_OPTIONAL_STOPPING_COUNT, int(args.optional_stopping_count))
    optional_bundle = build_dual_od_corpora(
        seed=int(args.seed) + 2027,
        representative_count=optional_target,
        ambiguous_count=max(20, min(optional_target, int(args.focused_count))),
        bbox=_parse_bbox(args.bbox),
        max_attempts=max(_resolve_max_attempts(args), optional_target * 24),
        acceptance_mode="graph_candidates",
        probe_max_paths=max(2, int(args.probe_max_paths)),
    )
    representative_summary = dict(optional_bundle["representative"])
    representative_rows = [dict(row) for row in representative_summary.get("rows", [])]
    if len(representative_rows) < optional_target:
        raise RuntimeError(
            f"full_latest_suite_optional_stopping_shortfall:{len(representative_rows)}<{optional_target}"
        )
    optional_source_pool_path = write_json_artifact(
        suite_run_id,
        "latest_corpus_optional_stopping_source_pool.summary.json",
        optional_bundle,
    )
    optional_rows = representative_rows[:optional_target]
    return _persist_corpus(
        run_id=suite_run_id,
        artifact_prefix="latest_corpus_optional_stopping",
        label="Optional-stopping latest corpus",
        rows=optional_rows,
        summary_payload=_subset_corpus_summary(
            source_summary=representative_summary,
            rows=optional_rows,
            corpus_kind="optional_stopping_coverage",
            selection_policy="fresh_optional_stopping_full_slice",
        ),
        source_summary_path=str(optional_source_pool_path),
    )


def _support_fragile_rows(rows: Sequence[Mapping[str, Any]], *, count: int) -> list[dict[str, Any]]:
    scored_rows: list[tuple[float, float, str, dict[str, Any]]] = []
    for row in rows:
        support_richness = _focused_support_richness(row)
        if support_richness is None or support_richness > SUPPORT_FRAGILE_THRESHOLD:
            continue
        ambiguity_index = _safe_float(row.get("od_ambiguity_index")) or 0.0
        annotated = _annotated_support_fragile_row(row, support_richness=support_richness)
        scored_rows.append((float(support_richness), -float(ambiguity_index), str(row.get("od_id") or ""), annotated))
    scored_rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return [dict(row) for _, _, _, row in scored_rows[: max(0, int(count))]]


def _persist_curated_focused_mixed_corpus(
    *,
    suite_run_id: str,
    focused_path: Path,
    focused_target: int,
) -> CorpusArtifact | None:
    curated_rows = _read_csv_rows(focused_path)
    support_rows = _support_fragile_rows(
        _read_csv_rows(SUPPORT_FRAGILE_SOURCE_CSV),
        count=_focused_support_fragile_target(focused_target),
    )
    if not support_rows:
        return None
    mixed_rows = _mix_focused_rows(curated_rows, support_rows, target=focused_target)
    source_summary_payload = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "created_at_utc": _now(),
        "selection_policy": "reused_curated_corpus_plus_support_fragile_slice",
        "source_paths": [str(focused_path), str(SUPPORT_FRAGILE_SOURCE_CSV)],
        "primary_source_path": str(focused_path),
        "support_fragile_source_path": str(SUPPORT_FRAGILE_SOURCE_CSV),
        "primary_source_row_count": len(curated_rows),
        "support_fragile_selected_count": _summary_counts(mixed_rows, key="corpus_group").get("support_fragile", 0),
        "support_fragile_threshold": SUPPORT_FRAGILE_THRESHOLD,
        "rows": mixed_rows,
    }
    source_summary_path = write_json_artifact(
        suite_run_id,
        "latest_corpus_focused.source_summary.json",
        source_summary_payload,
    )
    summary_payload = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "created_at_utc": _now(),
        "corpus_kind": "focused_mixed",
        "selection_policy": "reused_curated_corpus_plus_support_fragile_slice",
        "selected_count": len(mixed_rows),
        "accepted_by_bin": _summary_counts(mixed_rows, key="distance_bin"),
        "accepted_by_corridor": _summary_counts(mixed_rows, key="corridor_bucket"),
        "selected_by_cohort": _summary_counts(mixed_rows, key="corpus_group"),
        "support_fragile_threshold": SUPPORT_FRAGILE_THRESHOLD,
        "rows": mixed_rows,
        "source_path": str(focused_path),
        "support_fragile_source_path": str(SUPPORT_FRAGILE_SOURCE_CSV),
    }
    return _persist_corpus(
        run_id=suite_run_id,
        artifact_prefix="latest_corpus_focused",
        label="Focused curated corpus",
        rows=mixed_rows,
        summary_payload=summary_payload,
        source_summary_path=str(source_summary_path),
    )


def _curated_focused_corpus_needs_support_mix(rows: Sequence[Mapping[str, Any]], *, target: int) -> bool:
    required_support_rows = _focused_support_fragile_target(target)
    if required_support_rows <= 0:
        return False
    return len(_support_fragile_rows(rows, count=required_support_rows)) < required_support_rows


def _build_corpora(args: argparse.Namespace, *, suite_run_id: str) -> dict[str, CorpusArtifact]:
    optional_stopping_corpus = _build_optional_stopping_corpus(args, suite_run_id=suite_run_id)
    if bool(args.use_curated_corpora):
        focused_default = PROJECT_ROOT / "out" / "uk_od_corpus_thesis_ambiguity_subset.csv"
        if not focused_default.exists():
            focused_default = PROJECT_ROOT / "data" / "eval" / "uk_od_corpus_ambiguity_curated.csv"
        broad_path = Path(args.broad_corpus_csv or (PROJECT_ROOT / "data" / "eval" / "uk_od_corpus_thesis_broad.csv"))
        focused_path = Path(args.focused_corpus_csv or focused_default)
        transfer_path = Path(args.transfer_corpus_csv or (PROJECT_ROOT / "data" / "eval" / "uk_od_corpus_representative_expanded.csv"))
        synthetic_path = Path(args.synthetic_corpus_csv or broad_path)
        curated_corpora = {
            "broad": _existing_corpus_artifact(
                suite_run_id=suite_run_id,
                artifact_prefix="latest_corpus_broad",
                label="Broad curated corpus",
                csv_path=broad_path,
            ),
            "focused": _existing_corpus_artifact(
                suite_run_id=suite_run_id,
                artifact_prefix="latest_corpus_focused",
                label="Focused curated corpus",
                csv_path=focused_path,
            ),
            "transfer": _existing_corpus_artifact(
                suite_run_id=suite_run_id,
                artifact_prefix="latest_corpus_transfer",
                label="Transfer curated corpus",
                csv_path=transfer_path,
            ),
            "synthetic": _existing_corpus_artifact(
                suite_run_id=suite_run_id,
                artifact_prefix="latest_corpus_synthetic",
                label="Synthetic fallback curated corpus",
                csv_path=synthetic_path,
            ),
        }
        requested_minimums = {
            "broad": max(CURATED_CORPUS_MINIMUM_ROWS["broad"], int(args.broad_count)),
            "focused": max(CURATED_CORPUS_MINIMUM_ROWS["focused"], int(args.focused_count)),
            "transfer": max(CURATED_CORPUS_MINIMUM_ROWS["transfer"], int(args.transfer_count)),
            "synthetic": max(CURATED_CORPUS_MINIMUM_ROWS["synthetic"], int(args.synthetic_count)),
        }
        if any(
            curated_corpora[key].row_count < requested_minimums[key]
            for key in curated_corpora
        ):
            base_pool_corpora = _build_curated_base_pool_corpora(
                args,
                suite_run_id=suite_run_id,
                requested_minimums=requested_minimums,
            )
            if base_pool_corpora is not None:
                base_pool_corpora["optional_stopping"] = optional_stopping_corpus
                return base_pool_corpora
            generated_corpora = _build_generated_corpora(args, suite_run_id=suite_run_id)
            generated_corpora["optional_stopping"] = optional_stopping_corpus
            return generated_corpora
        focused_rows = _read_csv_rows(focused_path)
        if _curated_focused_corpus_needs_support_mix(
            focused_rows,
            target=int(args.focused_count),
        ):
            mixed_focused_corpus = _persist_curated_focused_mixed_corpus(
                suite_run_id=suite_run_id,
                focused_path=focused_path,
                focused_target=int(args.focused_count),
            )
            if mixed_focused_corpus is not None:
                curated_corpora["focused"] = mixed_focused_corpus
        curated_corpora["optional_stopping"] = optional_stopping_corpus
        return curated_corpora

    generated_corpora = _build_generated_corpora(args, suite_run_id=suite_run_id)
    generated_corpora["optional_stopping"] = optional_stopping_corpus
    return generated_corpora


def _clone_namespace(namespace: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(namespace))


def _lane_corpus_key(role: str) -> str:
    if role == "optional_stopping_coverage":
        return "optional_stopping"
    if role == "public_transfer":
        return "transfer"
    if role == "synthetic_ground_truth":
        return "synthetic"
    if role == "proxy_audit_calibration":
        return "focused"
    if role in BROAD_ROLES:
        return "broad"
    return "focused"


def _lane_cache_mode(role: str) -> str:
    if role in {
        "broad_cold_proof",
        "public_transfer",
        "synthetic_ground_truth",
        "optional_stopping_coverage",
        "proxy_audit_calibration",
    }:
        return "cold"
    return "preserve"


def _lane_seed_repeat_count(role: str, args: argparse.Namespace) -> int:
    if role in HEADLINE_ROLES:
        return max(1, int(args.headline_seed_repeat_count))
    return 1


def _lane_run_id(suite_run_id: str, role: str) -> str:
    return f"{suite_run_id}_{role}"


def _lane_artifact_paths(run_id: str, *, role: str | None = None) -> dict[str, str]:
    artifact_dir = artifact_dir_for_run(run_id)
    artifact_paths = {
        "artifact_dir": str(artifact_dir),
        "results_json": str(artifact_dir / "results.json"),
        "thesis_results_json": str(artifact_dir / "thesis_results.json"),
        "summary_json": str(artifact_dir / "thesis_summary.json"),
        "summary_by_cohort_json": str(artifact_dir / "thesis_summary_by_cohort.json"),
        "metrics_json": str(artifact_dir / "thesis_metrics.json"),
        "plots_json": str(artifact_dir / "thesis_plots.json"),
        "lane_metadata_json": str(artifact_dir / "lane_metadata.json"),
        "cohort_composition_json": str(artifact_dir / "cohort_composition.json"),
        "evaluation_manifest_json": str(artifact_dir / "evaluation_manifest.json"),
        "report_md": str(artifact_dir / "thesis_report.md"),
    }
    if role == "threshold_sensitivity":
        artifact_paths.update(
            {
                "threshold_sensitivity_summary_csv": str(artifact_dir / "threshold_sensitivity_summary.csv"),
                "threshold_sensitivity_summary_json": str(artifact_dir / "threshold_sensitivity_summary.json"),
                "threshold_sensitivity_report_md": str(artifact_dir / "threshold_sensitivity_report.md"),
            }
        )
    return artifact_paths


def _suite_progress_payload(
    *,
    suite_run_id: str,
    lane_runs: Mapping[str, Mapping[str, Any]],
    pending_roles: Sequence[str],
) -> dict[str, Any]:
    completed = sorted(role for role, record in lane_runs.items() if str(record.get("status")) == "completed")
    failed = sorted(role for role, record in lane_runs.items() if str(record.get("status")) == "failed")
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_run_id": suite_run_id,
        "updated_at_utc": _now(),
        "completed_roles": completed,
        "failed_roles": failed,
        "pending_roles": list(pending_roles),
        "lane_runs": {str(role): dict(record) for role, record in lane_runs.items()},
    }


def _write_suite_progress(
    *,
    suite_run_id: str,
    lane_runs: Mapping[str, Mapping[str, Any]],
    pending_roles: Sequence[str],
) -> str:
    path = write_json_artifact(
        suite_run_id,
        "suite_progress.json",
        _suite_progress_payload(
            suite_run_id=suite_run_id,
            lane_runs=lane_runs,
            pending_roles=pending_roles,
        ),
    )
    return str(path)


def _evaluation_namespace(
    args: argparse.Namespace,
    *,
    suite_run_id: str,
    role: str,
    corpus: CorpusArtifact,
) -> argparse.Namespace:
    namespace = _clone_namespace(_evaluation_defaults())
    namespace.corpus_csv = corpus.csv_path
    namespace.corpus_json = None
    namespace.backend_url = str(args.backend_url)
    namespace.in_process_backend = bool(args.in_process_backend)
    namespace.ready_timeout_seconds = float(args.ready_timeout_seconds)
    namespace.ready_poll_seconds = float(args.ready_poll_seconds)
    namespace.out_dir = str(args.out_dir)
    namespace.run_id = _lane_run_id(suite_run_id, role)
    namespace.seed = int(args.seed)
    namespace.seed_repeat_count = _lane_seed_repeat_count(role, args)
    namespace.seed_repeat_step = int(args.headline_seed_repeat_step)
    namespace.max_od = 0
    namespace.vehicle_type = str(args.vehicle_type)
    namespace.scenario_mode = str(args.scenario_mode)
    namespace.departure_time_utc = args.departure_time_utc
    namespace.model_version = str(args.model_version)
    namespace.optimization_mode = str(args.optimization_mode)
    namespace.max_alternatives = int(args.max_alternatives)
    namespace.search_budget = int(args.search_budget)
    namespace.evidence_budget = int(args.evidence_budget)
    namespace.world_count = int(args.world_count)
    namespace.certificate_threshold = float(args.certificate_threshold)
    namespace.tau_stop = float(args.tau_stop)
    namespace.stochastic_enabled = bool(args.stochastic_enabled)
    namespace.stochastic_samples = int(args.stochastic_samples)
    namespace.route_timeout_seconds = float(args.route_timeout_seconds)
    namespace.cache_mode = _lane_cache_mode(role)
    namespace.cold_cache_scope = str(getattr(namespace, "cold_cache_scope", "thesis_cold") or "thesis_cold")
    namespace.weight_time = float(args.weight_time)
    namespace.weight_money = float(args.weight_money)
    namespace.weight_co2 = float(args.weight_co2)
    namespace.fail_soft = bool(args.fail_soft)
    namespace.disable_tolls = bool(args.disable_tolls)
    namespace.baseline_refinement_policy = str(args.baseline_refinement_policy)
    namespace.ors_baseline_policy = str(args.ors_baseline_policy)
    namespace.ors_snapshot_mode = str(args.ors_snapshot_mode)
    namespace.ors_snapshot_path = args.ors_snapshot_path
    namespace.auto_enrich_corpus_ambiguity = bool(args.auto_enrich_corpus_ambiguity)
    namespace.allow_proxy_ors = bool(args.allow_proxy_ors)
    namespace.allow_evidence_fallbacks = bool(args.allow_evidence_fallbacks)
    namespace.evaluation_suite_role = role
    return namespace


def _hot_namespace(
    args: argparse.Namespace,
    *,
    suite_run_id: str,
    corpus: CorpusArtifact,
) -> argparse.Namespace:
    namespace = _clone_namespace(_hot_runner_defaults())
    namespace.corpus_csv = corpus.csv_path
    namespace.corpus_json = None
    namespace.backend_url = str(args.backend_url)
    namespace.in_process_backend = bool(args.in_process_backend)
    namespace.ready_timeout_seconds = float(args.ready_timeout_seconds)
    namespace.ready_poll_seconds = float(args.ready_poll_seconds)
    namespace.out_dir = str(args.out_dir)
    namespace.seed = int(args.seed)
    namespace.seed_repeat_count = 1
    namespace.seed_repeat_step = int(args.headline_seed_repeat_step)
    namespace.max_od = 0
    namespace.vehicle_type = str(args.vehicle_type)
    namespace.scenario_mode = str(args.scenario_mode)
    namespace.departure_time_utc = args.departure_time_utc
    namespace.model_version = str(args.model_version)
    namespace.optimization_mode = str(args.optimization_mode)
    namespace.max_alternatives = int(args.max_alternatives)
    namespace.search_budget = int(args.search_budget)
    namespace.evidence_budget = int(args.evidence_budget)
    namespace.world_count = int(args.world_count)
    namespace.certificate_threshold = float(args.certificate_threshold)
    namespace.tau_stop = float(args.tau_stop)
    namespace.stochastic_enabled = bool(args.stochastic_enabled)
    namespace.stochastic_samples = int(args.stochastic_samples)
    namespace.route_timeout_seconds = float(args.route_timeout_seconds)
    namespace.cache_mode = "preserve"
    namespace.weight_time = float(args.weight_time)
    namespace.weight_money = float(args.weight_money)
    namespace.weight_co2 = float(args.weight_co2)
    namespace.fail_soft = bool(args.fail_soft)
    namespace.disable_tolls = bool(args.disable_tolls)
    namespace.baseline_refinement_policy = str(args.baseline_refinement_policy)
    namespace.ors_baseline_policy = str(args.ors_baseline_policy)
    namespace.ors_snapshot_mode = str(args.ors_snapshot_mode)
    namespace.ors_snapshot_path = args.ors_snapshot_path
    namespace.auto_enrich_corpus_ambiguity = bool(args.auto_enrich_corpus_ambiguity)
    namespace.allow_proxy_ors = bool(args.allow_proxy_ors)
    namespace.allow_evidence_fallbacks = bool(args.allow_evidence_fallbacks)
    namespace.pair_run_id = f"{suite_run_id}_hot_rerun_pair"
    namespace.cold_run_id = f"{suite_run_id}_hot_rerun_cold"
    namespace.hot_run_id = f"{suite_run_id}_hot_rerun_hot"
    return namespace


def _lane_result_record(
    *,
    role: str,
    corpus: CorpusArtifact,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or payload.get("hot_run_id") or payload.get("pair_run_id") or "")
    summary_rows = payload.get("summary_rows", [])
    return {
        "status": "completed",
        "role": role,
        "run_id": run_id,
        "corpus_key": corpus.key,
        "corpus_label": corpus.label,
        "corpus_csv": corpus.csv_path,
        "row_count": corpus.row_count,
        "summary_row_count": len(summary_rows) if isinstance(summary_rows, Sequence) else 0,
        "lane_metadata": dict(payload.get("lane_metadata") or {}) if isinstance(payload.get("lane_metadata"), Mapping) else {},
        "artifact_paths": _lane_artifact_paths(run_id, role=role) if run_id else {},
        "headline_seed_summary_path": str(payload.get("headline_seed_summary_path") or ""),
        "headline_seed_claims_path": str(payload.get("headline_seed_claims_path") or ""),
        "headline_seed_report_table_csv": str(payload.get("headline_seed_report_table_csv") or ""),
    }


def _publishability_rows_for_lane(
    *,
    role: str,
    payload: Mapping[str, Any],
    corpus: CorpusArtifact,
) -> list[dict[str, Any]]:
    lane_rows: list[dict[str, Any]] = []
    summary_rows = payload.get("summary_rows")
    if not isinstance(summary_rows, Sequence):
        return lane_rows
    optional_stopping_rollups = (
        _optional_stopping_proof_rollups(payload)
        if role == "optional_stopping_coverage"
        else {}
    )
    perturbation_rollups = (
        _perturbation_proof_rollups(payload)
        if role == "perturbation_flip_radius"
        else {}
    )
    for row in summary_rows:
        if not isinstance(row, Mapping):
            continue
        variant_id = str(row.get("variant_id") or "")
        proof_rollup = {}
        if variant_id in optional_stopping_rollups:
            proof_rollup.update(optional_stopping_rollups[variant_id])
        if variant_id in perturbation_rollups:
            proof_rollup.update(perturbation_rollups[variant_id])
        lane_rows.append(
            {
                "lane_role": role,
                "variant_id": variant_id,
                "pipeline_mode": str(row.get("pipeline_mode") or ""),
                "corpus_key": corpus.key,
                "row_count": _safe_int(row.get("row_count")),
                "success_rate": _safe_float(row.get("success_rate")),
                "certified_rate": _safe_float(row.get("certified_rate")),
                "mean_certificate": _safe_float(row.get("mean_certificate")),
                "weighted_win_rate_best_baseline": _safe_float(row.get("weighted_win_rate_best_baseline")),
                "dominance_win_rate_best_baseline": _safe_float(row.get("dominance_win_rate_best_baseline")),
                "dominance_win_rate_osrm": _safe_float(row.get("dominance_win_rate_osrm")),
                "dominance_win_rate_ors": _safe_float(row.get("dominance_win_rate_ors")),
                "time_preserving_win_rate_best_baseline": _safe_float(row.get("time_preserving_win_rate_best_baseline")),
                "time_preserving_win_rate_osrm": _safe_float(row.get("time_preserving_win_rate_osrm")),
                "time_preserving_win_rate_ors": _safe_float(row.get("time_preserving_win_rate_ors")),
                "mean_weighted_margin_vs_best_baseline": _safe_float(row.get("mean_weighted_margin_vs_best_baseline")),
                "mean_runtime_ratio_vs_osrm": _safe_float(row.get("mean_runtime_ratio_vs_osrm")),
                "mean_runtime_ratio_vs_ors": _safe_float(row.get("mean_runtime_ratio_vs_ors")),
                "mean_runtime_p50_ms": _safe_float(row.get("mean_runtime_p50_ms")),
                "mean_runtime_p90_ms": _safe_float(row.get("mean_runtime_p90_ms")),
                "mean_runtime_p95_ms": _safe_float(row.get("mean_runtime_p95_ms")),
                "mean_process_rss_p90_mb": _safe_float(row.get("mean_process_rss_p90_mb")),
                "mean_peak_process_rss_mb": _safe_float(row.get("mean_peak_process_rss_mb")),
                "mean_peak_process_rss_p90_mb": _safe_float(row.get("mean_peak_process_rss_p90_mb")),
                "max_peak_process_rss_mb": _safe_float(row.get("max_peak_process_rss_mb")),
                "mean_peak_process_vms_mb": _safe_float(row.get("mean_peak_process_vms_mb")),
                "mean_peak_process_vms_p90_mb": _safe_float(row.get("mean_peak_process_vms_p90_mb")),
                "max_peak_process_vms_mb": _safe_float(row.get("max_peak_process_vms_mb")),
                "median_preference_query_count": _safe_float(row.get("median_preference_query_count")),
                "p90_preference_query_count": _safe_float(row.get("p90_preference_query_count")),
                "nontrivial_frontier_rate": _safe_float(row.get("nontrivial_frontier_rate")),
                "mean_dccs_false_safe_prune_rate": _safe_float(row.get("mean_dccs_false_safe_prune_rate")),
                "mean_dccs_anti_collapse_success_rate": _safe_float(row.get("mean_dccs_anti_collapse_success_rate")),
                "mean_dccs_certificate_critical_hit_rate": _safe_float(row.get("mean_dccs_certificate_critical_hit_rate")),
                "mean_dccs_time_preserving_challenger_coverage": _safe_float(row.get("mean_dccs_time_preserving_challenger_coverage")),
                "mean_dccs_dominance_likely_challenger_coverage": _safe_float(row.get("mean_dccs_dominance_likely_challenger_coverage")),
                "productive_voi_action_rate": _safe_float(row.get("productive_voi_action_rate")),
                "unnecessary_voi_refine_rate": _safe_float(row.get("unnecessary_voi_refine_rate")),
                "mean_voi_realized_certificate_lift": _safe_float(row.get("mean_voi_realized_certificate_lift")),
                "refine_cost_mape": _safe_float(row.get("refine_cost_mape")),
                "refine_cost_rank_correlation": _safe_float(row.get("refine_cost_rank_correlation")),
                "mean_route_cache_hit_rate": _safe_float(row.get("mean_route_cache_hit_rate")),
                "mean_option_build_cache_hit_rate": _safe_float(row.get("mean_option_build_cache_hit_rate")),
                "mean_option_build_reuse_rate": _safe_float(row.get("mean_option_build_reuse_rate")),
                "mean_refc_world_reuse_rate": _safe_float(row.get("mean_refc_world_reuse_rate")),
                "baseline_identity_verified_rate": _safe_float(row.get("baseline_identity_verified_rate")),
                "headline_seed_summary_path": str(payload.get("headline_seed_summary_path") or ""),
                "headline_seed_claims_path": str(payload.get("headline_seed_claims_path") or ""),
                "report_path": _lane_artifact_paths(str(payload.get("run_id") or ""), role=role).get("report_md", ""),
                **proof_rollup,
            }
        )
    return lane_rows


def _baseline_audit_rows_for_lane(
    *,
    role: str,
    payload: Mapping[str, Any],
    corpus: CorpusArtifact,
    suite_args: argparse.Namespace | None = None,
    suite_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = payload.get("rows")
    if isinstance(rows, Sequence):
        for row in rows:
            if isinstance(row, Mapping):
                grouped[str(row.get("variant_id") or "")].append(dict(row))
    summary_rows = payload.get("summary_rows")
    audit_rows: list[dict[str, Any]] = []
    if not isinstance(summary_rows, Sequence):
        return audit_rows
    for summary_row in summary_rows:
        if not isinstance(summary_row, Mapping):
            continue
        variant_id = str(summary_row.get("variant_id") or "")
        batch = grouped.get(variant_id, [])
        metadata_args = (
            suite_metadata.get("arguments")
            if isinstance(suite_metadata, Mapping)
            and isinstance(suite_metadata.get("arguments"), Mapping)
            else {}
        )
        preflight_summary = (
            suite_metadata.get("preflight_summary")
            if isinstance(suite_metadata, Mapping)
            and isinstance(suite_metadata.get("preflight_summary"), Mapping)
            else {}
        )
        preflight_checks = preflight_summary.get("checks") if isinstance(preflight_summary, Mapping) else []
        if not isinstance(preflight_checks, Sequence):
            preflight_checks = []
        preflight_check_map = {
            str(check.get("name") or ""): check
            for check in preflight_checks
            if isinstance(check, Mapping) and str(check.get("name") or "").strip()
        }
        smoke_summary = payload.get("baseline_smoke_summary")
        smoke_payload = (
            smoke_summary.get("payload")
            if isinstance(smoke_summary, Mapping) and isinstance(smoke_summary.get("payload"), Mapping)
            else {}
        )
        matched_od_ids = sorted(
            {
                str(row.get("od_id") or "").strip()
                for row in batch
                if str(row.get("od_id") or "").strip()
            }
        )
        suite_vehicle_type = str(getattr(suite_args, "vehicle_type", None) if suite_args is not None else payload.get("vehicle_type") or "").strip()
        suite_departure_time_utc = str(
            getattr(suite_args, "departure_time_utc", None) if suite_args is not None else payload.get("departure_time_utc") or ""
        ).strip()
        suite_scenario_mode = str(
            getattr(suite_args, "scenario_mode", None) if suite_args is not None else payload.get("scenario_mode") or ""
        ).strip()
        suite_baseline_refinement_policy = str(
            getattr(suite_args, "baseline_refinement_policy", None)
            if suite_args is not None
            else payload.get("baseline_refinement_policy")
            or metadata_args.get("baseline_refinement_policy")
            or ""
        ).strip()
        suite_ors_baseline_policy = str(
            getattr(suite_args, "ors_baseline_policy", None)
            if suite_args is not None
            else payload.get("ors_baseline_policy")
            or metadata_args.get("ors_baseline_policy")
            or ""
        ).strip()
        suite_ors_snapshot_mode = str(
            getattr(suite_args, "ors_snapshot_mode", None)
            if suite_args is not None
            else payload.get("ors_snapshot_mode")
            or metadata_args.get("ors_snapshot_mode")
            or ""
        ).strip()
        suite_allow_proxy_ors = (
            getattr(suite_args, "allow_proxy_ors", None)
            if suite_args is not None and getattr(suite_args, "allow_proxy_ors", None) is not None
            else payload.get("allow_proxy_ors")
            if payload.get("allow_proxy_ors") is not None
            else metadata_args.get("allow_proxy_ors")
        )
        suite_allow_evidence_fallbacks = (
            getattr(suite_args, "allow_evidence_fallbacks", None)
            if suite_args is not None and getattr(suite_args, "allow_evidence_fallbacks", None) is not None
            else payload.get("allow_evidence_fallbacks")
            if payload.get("allow_evidence_fallbacks") is not None
            else metadata_args.get("allow_evidence_fallbacks")
        )
        if not suite_vehicle_type:
            suite_vehicle_type = str(metadata_args.get("vehicle_type") or "").strip()
        if not suite_departure_time_utc:
            suite_departure_time_utc = str(metadata_args.get("departure_time_utc") or "").strip()
        if not suite_scenario_mode:
            suite_scenario_mode = str(metadata_args.get("scenario_mode") or "").strip()
        restriction_context = {
            "scenario_mode": suite_scenario_mode,
            "disable_tolls": bool(
                getattr(suite_args, "disable_tolls", None)
                if suite_args is not None and getattr(suite_args, "disable_tolls", None) is not None
                else payload.get("disable_tolls")
                if payload.get("disable_tolls") is not None
                else metadata_args.get("disable_tolls")
            ),
            "baseline_refinement_policy": suite_baseline_refinement_policy,
            "ors_baseline_policy": suite_ors_baseline_policy,
            "ors_snapshot_mode": suite_ors_snapshot_mode,
            "allow_proxy_ors": bool(suite_allow_proxy_ors),
            "allow_evidence_fallbacks": bool(suite_allow_evidence_fallbacks),
        }
        if isinstance(smoke_summary, Mapping) and isinstance(smoke_summary.get("osrm"), Mapping):
            osrm_smoke = smoke_summary.get("osrm") or {}
        else:
            osrm_smoke = preflight_check_map.get("osrm_engine_smoke") or {}
        if isinstance(smoke_summary, Mapping) and isinstance(smoke_summary.get("ors"), Mapping):
            ors_smoke = smoke_summary.get("ors") or {}
        else:
            ors_smoke = preflight_check_map.get("ors_engine_smoke") or {}
        route_required_ok = (
            bool(smoke_summary.get("required_ok"))
            if isinstance(smoke_summary, Mapping) and smoke_summary.get("required_ok") is not None
            else bool(preflight_summary.get("required_ok"))
            if isinstance(preflight_summary, Mapping) and preflight_summary.get("required_ok") is not None
            else None
        )
        osrm_provider_mode = str((osrm_smoke.get("provider_mode") if isinstance(osrm_smoke, Mapping) else "") or "").strip()
        if not osrm_provider_mode and isinstance(osrm_smoke, Mapping) and osrm_smoke:
            osrm_provider_mode = "repo_local"
        ors_provider_mode = str((ors_smoke.get("provider_mode") if isinstance(ors_smoke, Mapping) else "") or "").strip()
        if not ors_provider_mode:
            ors_provider_mode = next(
                (
                    str(row.get("ors_provider_mode") or "").strip()
                    for row in batch
                    if str(row.get("ors_provider_mode") or "").strip()
                ),
                "",
            )
        osrm_method = str((osrm_smoke.get("method") if isinstance(osrm_smoke, Mapping) else "") or "").strip()
        if not osrm_method:
            osrm_method = next(
                (
                    str(row.get("osrm_method") or "").strip()
                    for row in batch
                    if str(row.get("osrm_method") or "").strip()
                ),
                "",
            )
        ors_method = str((ors_smoke.get("method") if isinstance(ors_smoke, Mapping) else "") or "").strip()
        if not ors_method:
            ors_method = next(
                (
                    str(row.get("ors_method") or "").strip()
                    for row in batch
                    if str(row.get("ors_method") or "").strip()
                ),
                "",
            )
        route_feasibility_context = {
            "required_ok": route_required_ok,
            "osrm_ok": bool(osrm_smoke.get("ok")) if isinstance(osrm_smoke, Mapping) and osrm_smoke else None,
            "ors_ok": bool(ors_smoke.get("ok")) if isinstance(ors_smoke, Mapping) and ors_smoke else None,
            "osrm_provider_mode": osrm_provider_mode,
            "ors_provider_mode": ors_provider_mode,
            "osrm_method": osrm_method,
            "ors_method": ors_method,
            "vehicle_type": suite_vehicle_type or str(smoke_payload.get("vehicle_type") or "").strip(),
            "departure_time_utc": suite_departure_time_utc,
        }
        ors_modes = sorted({str(row.get("ors_provider_mode") or "") for row in batch if str(row.get("ors_provider_mode") or "").strip()})
        ors_identity_rate = 0.0
        if batch:
            ors_identity_rate = sum(
                1
                for row in batch
                if str(row.get("ors_graph_identity_status") or "") == "graph_identity_verified"
            ) / len(batch)
        audit_rows.append(
            {
                "lane_role": role,
                "variant_id": variant_id,
                "pipeline_mode": str(summary_row.get("pipeline_mode") or ""),
                "corpus_key": corpus.key,
                "row_count": _safe_int(summary_row.get("row_count")),
                "matched_od_count": len(matched_od_ids),
                "matched_od_ids_json": json.dumps(matched_od_ids, sort_keys=True),
                "matched_vehicle_type": route_feasibility_context["vehicle_type"],
                "matched_restriction_context_json": json.dumps(restriction_context, sort_keys=True),
                "matched_route_feasibility_context_json": json.dumps(route_feasibility_context, sort_keys=True),
                "baseline_smoke_required_ok": route_feasibility_context["required_ok"],
                "baseline_smoke_vehicle_type": route_feasibility_context["vehicle_type"],
                "baseline_smoke_origin_json": json.dumps((smoke_payload.get("origin") or {}), sort_keys=True),
                "baseline_smoke_destination_json": json.dumps((smoke_payload.get("destination") or {}), sort_keys=True),
                "baseline_identity_verified_rate": _safe_float(summary_row.get("baseline_identity_verified_rate")),
                "ors_graph_identity_verified_rate": round(float(ors_identity_rate), 6) if batch else None,
                "ors_provider_modes": ",".join(ors_modes),
                "best_baseline_provider_counts": json.dumps(dict(Counter(str(row.get("best_baseline_provider") or "") for row in batch)), sort_keys=True),
                "osrm_methods": json.dumps(sorted({str(row.get("osrm_method") or "") for row in batch if str(row.get("osrm_method") or "").strip()})),
                "ors_methods": json.dumps(sorted({str(row.get("ors_method") or "") for row in batch if str(row.get("ors_method") or "").strip()})),
                "weighted_win_rate_best_baseline": _safe_float(summary_row.get("weighted_win_rate_best_baseline")),
                "dominance_win_rate_best_baseline": _safe_float(summary_row.get("dominance_win_rate_best_baseline")),
                "time_preserving_win_rate_best_baseline": _safe_float(summary_row.get("time_preserving_win_rate_best_baseline")),
                "mean_runtime_ratio_vs_osrm": _safe_float(summary_row.get("mean_runtime_ratio_vs_osrm")),
                "mean_runtime_ratio_vs_ors": _safe_float(summary_row.get("mean_runtime_ratio_vs_ors")),
            }
        )
    return audit_rows


def _failure_atlas_rows(
    *,
    role: str,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    atlas_rows: list[dict[str, Any]] = []
    rows = payload.get("rows")
    if not isinstance(rows, Sequence):
        return atlas_rows
    run_id = str(payload.get("run_id") or "")
    artifact_dir = str(artifact_dir_for_run(run_id)) if run_id else ""
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        failure_reason = str(row.get("failure_reason") or "").strip()
        terminal_type = str(row.get("preference_terminal_type") or "").strip().lower()
        uncertified = row.get("certified") is False
        if not failure_reason and terminal_type not in {"abstained"} and not uncertified:
            continue
        atlas_rows.append(
            {
                "lane_role": role,
                "variant_id": str(row.get("variant_id") or ""),
                "od_id": str(row.get("od_id") or ""),
                "cohort": str(row.get("corpus_group") or row.get("corpus_kind") or ""),
                "support_status": str(row.get("support_flag") if row.get("support_flag") is not None else row.get("support_richness") or ""),
                "active_challenger": str(row.get("certificate_winner_route_id") or ""),
                "dominant_fragility_family": str(
                    row.get("final_refc_top_fragility_family")
                    or row.get("refc_top_fragility_family")
                    or row.get("initial_refc_top_fragility_family")
                    or ""
                ),
                "controller_stop_reason": str(row.get("voi_stop_reason") or ""),
                "root_cause_tag": _root_cause_tag(row),
                "failure_reason": failure_reason,
                "preference_terminal_type": terminal_type,
                "artifact_dir": artifact_dir,
                "results_json": str(Path(artifact_dir) / "thesis_results.json") if artifact_dir else "",
            }
        )
    return atlas_rows


def _read_json_payload(path_text: str | None) -> Mapping[str, Any]:
    if not path_text:
        return {}
    normalized_path = _normalize_existing_path(path_text)
    if normalized_path is None:
        return {}
    try:
        payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _normalize_existing_path(path_text: str | Path | None) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    if text.startswith(("\\", "/")):
        anchor = Path.cwd().anchor or "C:\\"
        return Path(anchor) / text.lstrip("\\/")
    return path


def _headline_seed_claim_rows_for_lane(
    *,
    role: str,
    payload: Mapping[str, Any],
    corpus: CorpusArtifact,
) -> list[dict[str, Any]]:
    claim_path = str(payload.get("headline_seed_claims_path") or "")
    claims_payload = _read_json_payload(claim_path)
    claim_rows = claims_payload.get("claim_rows")
    if not isinstance(claim_rows, Sequence):
        return []
    suite_rows: list[dict[str, Any]] = []
    for row in claim_rows:
        if not isinstance(row, Mapping):
            continue
        variant_id = str(row.get("variant_id") or "").strip()
        if variant_id not in {"A", "B", "C"}:
            continue
        paired_row_count_min = _safe_int(row.get("paired_comparison_row_count_min"))
        paired_row_count_max = _safe_int(row.get("paired_comparison_row_count_max"))
        point_estimate = _safe_float(row.get("point_estimate"))
        paired_delta = _safe_float(row.get("paired_delta"))
        if paired_row_count_min <= 0 or paired_row_count_max <= 0:
            continue
        if point_estimate is None or paired_delta is None:
            continue
        suite_rows.append(
            {
                "lane_role": role,
                "corpus_key": corpus.key,
                "variant_id": variant_id,
                "pipeline_mode": str(row.get("pipeline_mode") or ""),
                "headline_metric_name": str(row.get("headline_metric_name") or ""),
                "seed_count": _safe_int(row.get("seed_count")),
                "headline_seed_minimum_met": bool(row.get("headline_seed_minimum_met")),
                "majority_agreement_requirement_met": bool(row.get("majority_agreement_requirement_met")),
                "paired_comparison_row_count_min": _safe_int(row.get("paired_comparison_row_count_min")),
                "paired_comparison_row_count_max": _safe_int(row.get("paired_comparison_row_count_max")),
                "point_estimate": _safe_float(row.get("point_estimate")),
                "paired_delta": _safe_float(row.get("paired_delta")),
                "effect_size": _safe_float(row.get("effect_size")),
                "effect_size_method": str(row.get("effect_size_method") or ""),
                "ci_method": str(row.get("ci_method") or ""),
                "ci_confidence_level": _safe_float(row.get("ci_confidence_level")),
                "ci_lower": _safe_float(row.get("ci_lower")),
                "ci_upper": _safe_float(row.get("ci_upper")),
                "ci_crosses_zero": bool(row.get("ci_crosses_zero")),
                "bootstrap_resamples": _safe_int(row.get("bootstrap_resamples")),
                "raw_p_value": _safe_float(row.get("raw_p_value")),
                "multiple_comparison_method": str(row.get("multiple_comparison_method") or ""),
                "multiple_comparison_family_id": str(row.get("multiple_comparison_family_id") or ""),
                "multiple_comparison_family_size": _safe_int(row.get("multiple_comparison_family_size")),
                "holm_adjusted_p_value": _safe_float(row.get("holm_adjusted_p_value")),
                "holm_alpha": _safe_float(row.get("holm_alpha")),
                "holm_reject_at_alpha": row.get("holm_reject_at_alpha"),
                "headline_metric_majority_sign": str(row.get("headline_metric_majority_sign") or ""),
                "headline_metric_majority_share": _safe_float(row.get("headline_metric_majority_share")),
                "headline_metric_sign_flip_detected": bool(row.get("headline_metric_sign_flip_detected")),
                "headline_claim_narrowing_required": bool(row.get("headline_claim_narrowing_required")),
                "headline_claim_status": str(row.get("headline_claim_status") or ""),
                "headline_claim_label": str(row.get("headline_claim_label") or ""),
                "headline_claim_warning": str(row.get("headline_claim_warning") or ""),
                "source_claims_path": claim_path,
            }
        )
    return suite_rows


def _sample_size_rows_for_lane(
    *,
    role: str,
    payload: Mapping[str, Any],
    corpus: CorpusArtifact,
) -> list[dict[str, Any]]:
    lane_metadata = payload.get("lane_metadata")
    if not isinstance(lane_metadata, Mapping):
        return []
    payload_rows = payload.get("rows")
    if not isinstance(payload_rows, Sequence) or isinstance(payload_rows, (str, bytes)):
        payload_rows = []
    observed = lane_metadata.get("observed_sample_size")
    if not isinstance(observed, Mapping):
        observed = {}
    size_requirement = lane_metadata.get("evaluation_size_requirement")
    if not isinstance(size_requirement, Mapping):
        size_requirement = {}
    seed_plan = lane_metadata.get("seed_repeat_plan")
    if not isinstance(seed_plan, Mapping):
        seed_plan = {}
    def _payload_total(key: str) -> int | None:
        values = [_safe_float(row.get(key)) for row in payload_rows]  # type: ignore[union-attr]
        finite = [value for value in values if value is not None and math.isfinite(value)]
        return int(round(sum(finite))) if finite else None
    stored_evaluation_requirement_met = observed.get("evaluation_size_requirement_met")
    evaluation_requirement_met = None
    evaluation_requirement_observed_count = None
    evaluation_requirement_observed_count_source = ""
    unit = str(size_requirement.get("unit") or "").strip()
    minimum = size_requirement.get("minimum")
    evaluation_requirement_total_minimum = _safe_int(minimum)
    evaluation_requirement_cell_count = None
    evaluation_requirement_real_minimum = None
    evaluation_requirement_exact_synthetic_minimum = None
    evaluation_requirement_observed_real_count = None
    evaluation_requirement_observed_real_count_source = ""
    evaluation_requirement_observed_exact_synthetic_count = None
    evaluation_requirement_observed_exact_synthetic_count_source = ""
    row_count = _safe_int(observed.get("row_count"))
    effective_world_count = _maybe_int(observed.get("effective_cert_world_count"))
    if effective_world_count is None:
        effective_world_count = _payload_total("effective_cert_world_count")
    requested_world_count = _maybe_int(observed.get("requested_cert_world_count"))
    if requested_world_count is None:
        requested_world_count = _payload_total("requested_cert_world_count")
    probabilistic_world_count = _maybe_int(observed.get("probabilistic_world_count"))
    if probabilistic_world_count is None:
        probabilistic_world_count = _payload_total("probabilistic_world_count")
    audit_world_count = _maybe_int(observed.get("audit_world_count"))
    if audit_world_count is None:
        audit_world_count = _payload_total("audit_world_count")
    audited_route_pair_count = _maybe_int(observed.get("audited_route_pair_count"))
    if audited_route_pair_count is None:
        audited_route_pair_count = _payload_total("audited_route_pair_count")
    if unit in {"rows", "samples", "states"} and minimum is not None:
        try:
            observed_count = row_count
            observed_source = "row_count" if row_count is not None else ""
            if unit in {"samples", "states"}:
                observed_candidates = [
                    ("effective_cert_world_count", effective_world_count),
                    ("requested_cert_world_count", requested_world_count),
                    ("probabilistic_world_count", probabilistic_world_count),
                    ("row_count", row_count),
                ]
                populated_candidates = [
                    (source, value)
                    for source, value in observed_candidates
                    if value is not None
                ]
                if populated_candidates:
                    observed_source, observed_count = max(
                        populated_candidates,
                        key=lambda item: int(item[1]),
                    )
            evaluation_requirement_observed_count = observed_count
            evaluation_requirement_observed_count_source = observed_source
            evaluation_requirement_met = observed_count >= int(minimum)
        except (TypeError, ValueError):
            evaluation_requirement_met = None
    elif unit == "audited_route_pair_observations_per_cell" and minimum is not None:
        cell_structure = lane_metadata.get("evaluation_cell_structure")
        if isinstance(cell_structure, Mapping):
            cell_count = 1
            for value in cell_structure.values():
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    cell_count *= max(1, len(value))
            try:
                minimum_total = int(minimum) * max(1, int(cell_count))
            except (TypeError, ValueError):
                minimum_total = None
            if minimum_total is not None:
                evaluation_requirement_cell_count = int(cell_count)
                evaluation_requirement_total_minimum = minimum_total
                observed_count = audited_route_pair_count
                observed_source = "audited_route_pair_count" if observed_count is not None else ""
                if observed_count is None and audit_world_count is not None:
                    observed_count = audit_world_count
                    observed_source = "audit_world_count"
                if observed_count is None and row_count is not None:
                    observed_count = row_count
                    observed_source = "row_count"
                evaluation_requirement_observed_count = observed_count
                evaluation_requirement_observed_count_source = observed_source
                evaluation_requirement_met = observed_count >= minimum_total
    elif unit == "compound":
        description = str(size_requirement.get("minimum_description") or "")
        numbers = [int(token.replace(",", "")) for token in re.findall(r"\d[\d,]*", description)]
        if numbers:
            minimum_total = sum(numbers)
            evaluation_requirement_total_minimum = minimum_total
            observed_total = row_count
            observed_source = "row_count" if row_count is not None else ""
            if "real rows" in description.lower() and "synthetic rows" in description.lower():
                evaluation_requirement_real_minimum = PERTURBATION_REQUIRED_REAL_ROW_MINIMUM
                evaluation_requirement_exact_synthetic_minimum = PERTURBATION_REQUIRED_EXACT_SYNTHETIC_MINIMUM
                evaluation_requirement_observed_real_count = row_count
                evaluation_requirement_observed_real_count_source = "row_count" if row_count is not None else ""
                exact_synthetic_world_count = _perturbation_exact_synthetic_world_count(list(payload_rows))
                if exact_synthetic_world_count > 0:
                    evaluation_requirement_observed_exact_synthetic_count = exact_synthetic_world_count
                    evaluation_requirement_observed_exact_synthetic_count_source = "sampled_world_manifest.world_kind"
                observed_total = (
                    (evaluation_requirement_observed_real_count or 0)
                    + (evaluation_requirement_observed_exact_synthetic_count or 0)
                )
                observed_source = "row_count_plus_exact_synthetic_world_count"
                evaluation_requirement_met = bool(
                    (evaluation_requirement_observed_real_count or 0) >= evaluation_requirement_real_minimum
                    and (evaluation_requirement_observed_exact_synthetic_count or 0) >= evaluation_requirement_exact_synthetic_minimum
                )
            evaluation_requirement_observed_count = observed_total
            evaluation_requirement_observed_count_source = observed_source
            if evaluation_requirement_met is None:
                evaluation_requirement_met = observed_total >= minimum_total
    if evaluation_requirement_met is None:
        evaluation_requirement_met = bool(stored_evaluation_requirement_met) if stored_evaluation_requirement_met is not None else None
    return [
        {
            "lane_role": role,
            "corpus_key": corpus.key,
            "corpus_label": corpus.label,
            "corpus_row_count": corpus.row_count,
            "observed_row_count": _safe_int(observed.get("row_count")),
            "observed_unique_od_count": _safe_int(observed.get("unique_od_count")),
            "observed_unique_row_seed_count": _safe_int(observed.get("unique_row_seed_count")),
            "observed_effective_cert_world_count": effective_world_count,
            "observed_requested_cert_world_count": requested_world_count,
            "observed_probabilistic_world_count": probabilistic_world_count,
            "observed_audit_world_count": audit_world_count,
            "observed_audited_route_pair_count": audited_route_pair_count,
            "evaluation_requirement_id": str(size_requirement.get("requirement_id") or ""),
            "evaluation_requirement_unit": str(size_requirement.get("unit") or ""),
            "evaluation_requirement_minimum": _safe_int(size_requirement.get("minimum")),
            "evaluation_requirement_total_minimum": evaluation_requirement_total_minimum,
            "evaluation_requirement_cell_count": evaluation_requirement_cell_count,
            "evaluation_requirement_observed_count": evaluation_requirement_observed_count,
            "evaluation_requirement_observed_count_source": evaluation_requirement_observed_count_source,
            "evaluation_requirement_real_minimum": evaluation_requirement_real_minimum,
            "evaluation_requirement_exact_synthetic_minimum": evaluation_requirement_exact_synthetic_minimum,
            "evaluation_requirement_observed_real_count": evaluation_requirement_observed_real_count,
            "evaluation_requirement_observed_real_count_source": evaluation_requirement_observed_real_count_source,
            "evaluation_requirement_observed_exact_synthetic_count": evaluation_requirement_observed_exact_synthetic_count,
            "evaluation_requirement_observed_exact_synthetic_count_source": evaluation_requirement_observed_exact_synthetic_count_source,
            "evaluation_requirement_description": str(size_requirement.get("minimum_description") or ""),
            "evaluation_requirement_met": evaluation_requirement_met,
            "headline_seed_repeat_required": bool(seed_plan.get("headline_seed_repeat_required")),
            "headline_seed_requirement_ids": json.dumps(list(seed_plan.get("requirement_ids") or [])),
            "headline_seed_minimum_count": _safe_int(seed_plan.get("minimum_seed_count")),
            "headline_seed_configured_count": _safe_int(seed_plan.get("configured_seed_count")),
            "headline_seed_values": json.dumps(list(seed_plan.get("configured_seeds") or [])),
            "headline_seed_minimum_met": seed_plan.get("meets_minimum"),
            "headline_seed_status": str(seed_plan.get("status") or ""),
            "headline_seed_summary_path": str(payload.get("headline_seed_summary_path") or ""),
            "headline_seed_claims_path": str(payload.get("headline_seed_claims_path") or ""),
        }
    ]


def _root_cause_tag(row: Mapping[str, Any]) -> str:
    reason = str(row.get("failure_reason") or row.get("voi_stop_reason") or "").strip().lower()
    if "support" in reason or row.get("support_flag") is False:
        return "support_failure"
    if "hidden" in reason or (_safe_float(row.get("dccs_hidden_challenger_miss_rate")) or 0.0) > 0.0:
        return "hidden_challenger"
    if "proxy" in reason:
        return "proxy_bias"
    if "preference" in reason or str(row.get("preference_terminal_type") or "").strip().lower() == "abstained":
        return "preference_ambiguity"
    if "budget" in reason:
        return "budget_cut"
    if reason:
        return reason.replace(" ", "_")
    return "uncertified"


def _publishability_verdict_payload(
    *,
    lane_publishability_rows: Sequence[Mapping[str, Any]],
    baseline_audit_rows: Sequence[Mapping[str, Any]],
    failure_atlas_rows: Sequence[Mapping[str, Any]],
    sample_size_rows: Sequence[Mapping[str, Any]],
    headline_seed_claim_rows: Sequence[Mapping[str, Any]],
    hot_payload: Mapping[str, Any] | None,
    suite_artifact_dir: Path | None = None,
) -> dict[str, Any]:
    headline_rows = [
        row
        for row in lane_publishability_rows
        if str(row.get("lane_role") or "") in HEADLINE_ADOPTION_ROLES
        and str(row.get("variant_id") or "") in {"A", "B", "C"}
    ]
    adoption_checks: list[dict[str, Any]] = []
    for row in headline_rows:
        checks = {
            "dominance_win_rate_best_baseline": (_safe_float(row.get("dominance_win_rate_best_baseline")) or -1.0) >= 0.70,
            "dominance_win_rate_osrm": (_safe_float(row.get("dominance_win_rate_osrm")) or -1.0) >= 0.70,
            "time_preserving_win_rate_best_baseline": (_safe_float(row.get("time_preserving_win_rate_best_baseline")) or -1.0) >= 0.60,
            "time_preserving_win_rate_osrm": (_safe_float(row.get("time_preserving_win_rate_osrm")) or -1.0) >= 0.60,
            "time_preserving_win_rate_ors": (_safe_float(row.get("time_preserving_win_rate_ors")) or -1.0) >= 0.60,
            "mean_weighted_margin_vs_best_baseline": (_safe_float(row.get("mean_weighted_margin_vs_best_baseline")) or -1.0) >= 3.0,
            "baseline_identity_manifests_attached": bool(suite_artifact_dir)
            and (suite_artifact_dir / "osrm_baseline_identity_manifest.json").exists()
            and (suite_artifact_dir / "ors_baseline_identity_manifest.json").exists(),
        }
        adoption_checks.append(
            {
                "lane_role": str(row.get("lane_role") or ""),
                "variant_id": str(row.get("variant_id") or ""),
                "checks": checks,
                "all_green": all(checks.values()),
            }
        )
    hot_gate = dict(hot_payload.get("hot_gate") or {}) if isinstance(hot_payload, Mapping) else {}
    fairness_failures = [
        row
        for row in baseline_audit_rows
        if (
            _safe_int(row.get("matched_od_count")) <= 0
            or not str(row.get("matched_vehicle_type") or "").strip()
            or not str(row.get("matched_restriction_context_json") or "").strip()
            or not str(row.get("matched_route_feasibility_context_json") or "").strip()
            or row.get("baseline_smoke_required_ok") is not True
        )
    ]
    sample_size_failures = [
        row
        for row in sample_size_rows
        if str(row.get("evaluation_requirement_id") or "").strip()
        and row.get("evaluation_requirement_met") is False
    ]
    headline_seed_failures = [
        row
        for row in sample_size_rows
        if bool(row.get("headline_seed_repeat_required")) and row.get("headline_seed_minimum_met") is False
    ]
    claim_narrowings = [
        row
        for row in headline_seed_claim_rows
        if bool(row.get("headline_claim_narrowing_required"))
    ]
    inconclusive_claims = [
        row
        for row in headline_seed_claim_rows
        if bool(row.get("ci_crosses_zero"))
    ]
    bootstrap_shortfalls = [
        row
        for row in headline_seed_claim_rows
        if _safe_int(row.get("bootstrap_resamples")) < 10_000
    ]
    all_headline_green = bool(adoption_checks) and all(bool(item.get("all_green")) for item in adoption_checks)
    hot_all_green = bool(hot_gate.get("all_green")) if hot_gate else False
    publishability_blockers: list[str] = []
    if not all_headline_green:
        publishability_blockers.append("headline_adoption_checks_not_all_green")
    if not hot_all_green:
        publishability_blockers.append("hot_rerun_reuse_gates_not_all_green")
    if fairness_failures:
        publishability_blockers.append("baseline_fairness_audit_failures_present")
    if sample_size_failures:
        publishability_blockers.append("evaluation_size_requirements_not_met")
    if headline_seed_failures:
        publishability_blockers.append("headline_seed_repeat_requirements_not_met")
    if claim_narrowings:
        publishability_blockers.append("headline_claim_narrowing_required")
    if bootstrap_shortfalls:
        publishability_blockers.append("headline_bootstrap_resample_shortfall")
    hard_evidence_gates_green = bool(
        all_headline_green
        and hot_all_green
        and len(fairness_failures) == 0
        and len(sample_size_failures) == 0
        and len(headline_seed_failures) == 0
        and len(claim_narrowings) == 0
        and len(bootstrap_shortfalls) == 0
    )
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "generated_at_utc": _now(),
        "headline_rows_evaluated": len(headline_rows),
        "headline_adoption_checks": adoption_checks,
        "headline_all_green": all_headline_green,
        "hot_rerun_all_green": hot_all_green,
        "fairness_failure_count": len(fairness_failures),
        "sample_size_failure_count": len(sample_size_failures),
        "headline_seed_failure_count": len(headline_seed_failures),
        "headline_claim_rows_evaluated": len(headline_seed_claim_rows),
        "headline_claim_narrowing_count": len(claim_narrowings),
        "headline_inconclusive_claim_count": len(inconclusive_claims),
        "headline_bootstrap_shortfall_count": len(bootstrap_shortfalls),
        "failure_atlas_case_count": len(failure_atlas_rows),
        "sample_size_failures": [
            {
                "lane_role": str(row.get("lane_role") or ""),
                "requirement_id": str(row.get("evaluation_requirement_id") or ""),
                "observed_row_count": _safe_int(row.get("observed_row_count")),
                "unique_od_count": _safe_int(row.get("observed_unique_od_count")),
                "minimum": _safe_int(row.get("evaluation_requirement_minimum")),
                "observed_count": _safe_int(row.get("evaluation_requirement_observed_count")),
                "observed_count_source": str(row.get("evaluation_requirement_observed_count_source") or ""),
                "real_minimum": _maybe_int(row.get("evaluation_requirement_real_minimum")),
                "observed_real_count": _maybe_int(row.get("evaluation_requirement_observed_real_count")),
                "exact_synthetic_minimum": _maybe_int(row.get("evaluation_requirement_exact_synthetic_minimum")),
                "observed_exact_synthetic_count": _maybe_int(row.get("evaluation_requirement_observed_exact_synthetic_count")),
            }
            for row in sample_size_failures
        ],
        "headline_seed_failures": [
            {
                "lane_role": str(row.get("lane_role") or ""),
                "configured_seed_count": _safe_int(row.get("headline_seed_configured_count")),
                "minimum_seed_count": _safe_int(row.get("headline_seed_minimum_count")),
                "status": str(row.get("headline_seed_status") or ""),
            }
            for row in headline_seed_failures
        ],
        "headline_claim_warnings": [
            {
                "lane_role": str(row.get("lane_role") or ""),
                "variant_id": str(row.get("variant_id") or ""),
                "metric": str(row.get("headline_metric_name") or ""),
                "claim_status": str(row.get("headline_claim_status") or ""),
                "claim_warning": str(row.get("headline_claim_warning") or ""),
                "ci_crosses_zero": bool(row.get("ci_crosses_zero")),
                "holm_adjusted_p_value": _safe_float(row.get("holm_adjusted_p_value")),
            }
            for row in claim_narrowings
        ],
        "publishability_blockers": publishability_blockers,
        "publishable_on_current_evidence": hard_evidence_gates_green,
        "adoption_claim_supported": hard_evidence_gates_green,
        "hot_rerun_gate": hot_gate,
    }


def _write_baseline_identity_manifests(
    *,
    suite_run_id: str,
    suite_artifact_dir: Path,
    suite_args: argparse.Namespace | None = None,
) -> dict[str, str]:
    osrm_manifest_path = write_json_artifact(
        suite_run_id,
        "osrm_baseline_identity_manifest.json",
        {
            "suite_run_id": suite_run_id,
            "attached": True,
            "manifest_type": "baseline_identity",
            "provider": "osrm",
            "baseline_refinement_policy": str(getattr(suite_args, "baseline_refinement_policy", "") or ""),
            "vehicle_type": str(getattr(suite_args, "vehicle_type", "") or ""),
            "scenario_mode": str(getattr(suite_args, "scenario_mode", "") or ""),
        },
    )
    ors_manifest_path = write_json_artifact(
        suite_run_id,
        "ors_baseline_identity_manifest.json",
        {
            "suite_run_id": suite_run_id,
            "attached": True,
            "manifest_type": "baseline_identity",
            "provider": "ors",
            "baseline_refinement_policy": str(getattr(suite_args, "baseline_refinement_policy", "") or ""),
            "ors_baseline_policy": str(getattr(suite_args, "ors_baseline_policy", "") or ""),
            "ors_snapshot_mode": str(getattr(suite_args, "ors_snapshot_mode", "") or ""),
        },
    )
    return {
        "osrm_baseline_identity_manifest_json": str(osrm_manifest_path),
        "ors_baseline_identity_manifest_json": str(ors_manifest_path),
        "suite_artifact_dir": str(suite_artifact_dir),
    }


def _publishability_markdown(
    *,
    suite_run_id: str,
    verdict: Mapping[str, Any],
    lane_publishability_rows: Sequence[Mapping[str, Any]],
    baseline_audit_rows: Sequence[Mapping[str, Any]],
    sample_size_rows: Sequence[Mapping[str, Any]],
    headline_seed_claim_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# Full Latest Suite Assessment: {suite_run_id}",
        "",
        f"- Generated at UTC: {_now()}",
        f"- Publishable on current evidence: {'yes' if verdict.get('publishable_on_current_evidence') else 'no'}",
        f"- Adoption claim supported on current evidence: {'yes' if verdict.get('adoption_claim_supported') else 'no'}",
        f"- Headline rows evaluated: {int(verdict.get('headline_rows_evaluated') or 0)}",
        f"- Fairness failures: {int(verdict.get('fairness_failure_count') or 0)}",
        f"- Sample-size failures: {int(verdict.get('sample_size_failure_count') or 0)}",
        f"- Headline seed-repeat failures: {int(verdict.get('headline_seed_failure_count') or 0)}",
        f"- Headline claim narrowings: {int(verdict.get('headline_claim_narrowing_count') or 0)}",
        f"- Failure atlas cases: {int(verdict.get('failure_atlas_case_count') or 0)}",
        "",
        "## Publishability Blockers",
        "",
        *([f"- {blocker}" for blocker in list(verdict.get("publishability_blockers") or [])] or ["- none"]),
        "",
    ]
    lines.extend(
        [
            "## Headline Rows",
        "",
        ]
    )
    for row in lane_publishability_rows:
        if str(row.get("lane_role") or "") not in HEADLINE_ROLES:
            continue
        lines.append(
            "- "
            + f"{row.get('lane_role')} / {row.get('variant_id')}: "
            + f"dominance_best={row.get('dominance_win_rate_best_baseline')}, "
            + f"dominance_osrm={row.get('dominance_win_rate_osrm')}, "
            + f"time_best={row.get('time_preserving_win_rate_best_baseline')}, "
            + f"time_osrm={row.get('time_preserving_win_rate_osrm')}, "
            + f"time_ors={row.get('time_preserving_win_rate_ors')}, "
            + f"weighted_margin_best={row.get('mean_weighted_margin_vs_best_baseline')}, "
            + f"frontier={row.get('nontrivial_frontier_rate')}, "
            + f"productive_voi={row.get('productive_voi_action_rate')}, "
            + f"unnecessary_voi={row.get('unnecessary_voi_refine_rate')}, "
            + f"mean_voi_lift={row.get('mean_voi_realized_certificate_lift')}, "
            + f"refine_cost_mape={row.get('refine_cost_mape')}, "
            + f"refine_cost_rank_corr={row.get('refine_cost_rank_correlation')}, "
            + f"runtime_ratio_osrm={row.get('mean_runtime_ratio_vs_osrm')}, "
            + f"runtime_ratio_ors={row.get('mean_runtime_ratio_vs_ors')}"
        )
    optional_stopping_rows = [
        row for row in lane_publishability_rows if str(row.get("lane_role") or "") == "optional_stopping_coverage"
    ]
    if optional_stopping_rows:
        lines.extend(["", "## Optional Stopping Proof", ""])
        for row in optional_stopping_rows:
            lines.append(
                "- "
                + f"{row.get('lane_role')} / {row.get('variant_id')}: "
                + f"method_rate={row.get('optional_stopping_method_recorded_rate')}, "
                + f"delta_rate={row.get('optional_stopping_delta_recorded_rate')}, "
                + f"validity_tested_rate={row.get('optional_stopping_validity_tested_rate')}, "
                + f"validity_violation_rate={row.get('optional_stopping_validity_violation_rate')}, "
                + f"coverage_floor={row.get('optional_stopping_guaranteed_coverage_floor')}, "
                + f"required_floor={row.get('optional_stopping_required_coverage_floor')}, "
                + f"methods={row.get('optional_stopping_methods_json')}, "
                + f"deltas={row.get('optional_stopping_delta_values_json')}"
            )
    perturbation_rows = [
        row for row in lane_publishability_rows if str(row.get("lane_role") or "") == "perturbation_flip_radius"
    ]
    if perturbation_rows:
        lines.extend(["", "## Perturbation Proof", ""])
        for row in perturbation_rows:
            lines.append(
                "- "
                + f"{row.get('lane_role')} / {row.get('variant_id')}: "
                + f"real_violation_rate={row.get('real_lane_flip_radius_violation_rate')}, "
                + f"exact_synthetic_violation_rate={row.get('exact_synthetic_flip_radius_violation_rate')}, "
                + f"exact_synthetic_worlds={row.get('perturbation_exact_synthetic_world_count')}, "
                + f"min_flip_budget={row.get('perturbation_minimum_flip_budget_min')}, "
                + f"world_kinds={row.get('perturbation_world_kind_counts_json')}"
            )
    if baseline_audit_rows:
        lines.extend(["", "## Baseline Audit", ""])
        for row in baseline_audit_rows:
            lines.append(
                "- "
                + f"{row.get('lane_role')} / {row.get('variant_id')}: "
                + f"identity_verified={row.get('baseline_identity_verified_rate')}, "
                + f"ors_graph_identity_verified={row.get('ors_graph_identity_verified_rate')}, "
                + f"ors_modes={row.get('ors_provider_modes')}"
            )
    if isinstance(verdict.get("hot_rerun_gate"), Mapping):
        hot_gate = verdict["hot_rerun_gate"]
        lines.extend(
            [
                "",
                "## Hot Rerun",
                "",
                f"- All hot-rerun reuse gates green: {'yes' if hot_gate.get('all_green') else 'no'}",
                f"- Winner parity: {hot_gate.get('hot_cold_winner_identity_parity')}",
                f"- Mean final LCB drift: {hot_gate.get('mean_final_certificate_lcb_drift')}",
                f"- Max final LCB drift: {hot_gate.get('max_final_certificate_lcb_abs_drift')}",
            ]
        )
    if sample_size_rows:
        lines.extend(["", "## Sample Size And Seed Repeat", ""])
        for row in sample_size_rows:
            lines.append(
                "- "
                + f"{row.get('lane_role')}: "
                + f"observed_rows={row.get('observed_row_count')}, "
                + f"unique_od={row.get('observed_unique_od_count')}, "
                + f"evaluation_requirement={row.get('evaluation_requirement_id') or 'n/a'}, "
                + f"minimum={row.get('evaluation_requirement_minimum')}, "
                + f"observed_count={row.get('evaluation_requirement_observed_count')}, "
                + f"observed_count_source={row.get('evaluation_requirement_observed_count_source')}, "
                + f"real_count={row.get('evaluation_requirement_observed_real_count')}, "
                + f"real_minimum={row.get('evaluation_requirement_real_minimum')}, "
                + f"exact_synthetic_count={row.get('evaluation_requirement_observed_exact_synthetic_count')}, "
                + f"exact_synthetic_minimum={row.get('evaluation_requirement_exact_synthetic_minimum')}, "
                + f"requirement_met={row.get('evaluation_requirement_met')}, "
                + f"seed_repeat_required={row.get('headline_seed_repeat_required')}, "
                + f"configured_seed_count={row.get('headline_seed_configured_count')}, "
                + f"minimum_seed_count={row.get('headline_seed_minimum_count')}, "
                + f"seed_requirement_met={row.get('headline_seed_minimum_met')}"
            )
    if headline_seed_claim_rows:
        lines.extend(["", "## Headline Seed Claims", ""])
        for row in headline_seed_claim_rows:
            lines.append(
                "- "
                + f"{row.get('lane_role')} / {row.get('variant_id')} / {row.get('headline_metric_name')}: "
                + f"point_estimate={row.get('point_estimate')}, "
                + f"delta={row.get('paired_delta')}, "
                + f"effect_size={row.get('effect_size')}, "
                + f"ci=[{row.get('ci_lower')}, {row.get('ci_upper')}], "
                + f"holm_adjusted_p={row.get('holm_adjusted_p_value')}, "
                + f"claim_status={row.get('headline_claim_status')}, "
                + f"claim_warning={row.get('headline_claim_warning')}"
            )
    return "\n".join(lines).strip() + "\n"


def _suite_index_markdown(
    *,
    suite_run_id: str,
    corpora: Mapping[str, CorpusArtifact],
    lane_runs: Mapping[str, Mapping[str, Any]],
    verdict: Mapping[str, Any],
    publication_outputs: Mapping[str, str] | None = None,
) -> str:
    lines = [
        f"# Latest Full Suite Index: {suite_run_id}",
        "",
        "## Corpora",
        "",
    ]
    for key, corpus in corpora.items():
        lines.append(f"- {key}: {corpus.row_count} rows | {corpus.csv_path}")
    lines.extend(["", "## Lane Runs", ""])
    for role, record in lane_runs.items():
        lines.append(
            "- "
            + f"{role}: status={record.get('status')}, "
            + f"run_id={record.get('run_id')}, "
            + f"corpus={record.get('corpus_key')}"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Publishable on current evidence: {'yes' if verdict.get('publishable_on_current_evidence') else 'no'}",
            f"- Adoption claim supported: {'yes' if verdict.get('adoption_claim_supported') else 'no'}",
        ]
    )
    if isinstance(publication_outputs, Mapping) and publication_outputs:
        lines.extend(["", "## REFC Publication Surfaces", ""])
        for label, path in publication_outputs.items():
            path_text = str(path or "").strip()
            if path_text:
                lines.append(f"- {label}: {path_text}")
    return "\n".join(lines).strip() + "\n"


def run_full_latest_suite(args: argparse.Namespace) -> dict[str, Any]:
    suite_run_id = str(args.run_id or _run_label())
    old_out_dir = settings.out_dir
    settings.out_dir = str(Path(args.out_dir))
    lane_runs: dict[str, dict[str, Any]] = {}
    try:
        artifact_dir_for_run(suite_run_id)
        preflight_path = artifact_dir_for_run(suite_run_id) / "preflight_live_runtime.json"
        preflight_summary = run_preflight(output_path=preflight_path)
        corpora = _build_corpora(args, suite_run_id=suite_run_id)
        suite_sources_path = write_json_artifact(
            suite_run_id,
            "suite_sources.json",
            {
                "schema_version": SUITE_SCHEMA_VERSION,
                "generated_at_utc": _now(),
                "suite_run_id": suite_run_id,
                "preflight_path": str(preflight_path),
                "corpora": {
                    key: {
                        "label": corpus.label,
                        "row_count": corpus.row_count,
                        "csv_path": corpus.csv_path,
                        "json_path": corpus.json_path,
                        "summary_path": corpus.summary_path,
                        "source_summary_path": corpus.source_summary_path,
                    }
                    for key, corpus in corpora.items()
                },
            },
        )
        _write_suite_progress(
            suite_run_id=suite_run_id,
            lane_runs=lane_runs,
            pending_roles=[*DIRECT_SUITE_ROLES, "hot_rerun"],
        )

        direct_payloads: dict[str, dict[str, Any]] = {}
        hot_payload: dict[str, Any] | None = None
        with ExitStack() as stack:
            if bool(args.in_process_backend):
                stack.enter_context(in_process_backend_runtime_profile())
                from app.main import app

                active_client: Any = stack.enter_context(TestClient(app))
            else:
                active_client = stack.enter_context(httpx.Client(base_url=args.backend_url, timeout=args.route_timeout_seconds))

            for index, role in enumerate(DIRECT_SUITE_ROLES):
                corpus = corpora[_lane_corpus_key(role)]
                try:
                    payload = run_thesis_evaluation(
                        _evaluation_namespace(args, suite_run_id=suite_run_id, role=role, corpus=corpus),
                        client=active_client,
                    )
                    direct_payloads[role] = dict(payload)
                    lane_runs[role] = _lane_result_record(role=role, corpus=corpus, payload=payload)
                except Exception as exc:
                    lane_runs[role] = {
                        "status": "failed",
                        "role": role,
                        "corpus_key": corpus.key,
                        "corpus_csv": corpus.csv_path,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if bool(args.fail_fast):
                        raise
                _write_suite_progress(
                    suite_run_id=suite_run_id,
                    lane_runs=lane_runs,
                    pending_roles=[*DIRECT_SUITE_ROLES[index + 1 :], "hot_rerun"],
                )

            hot_corpus = corpora["broad"]
            try:
                hot_payload = run_hot_rerun_benchmark(
                    _hot_namespace(args, suite_run_id=suite_run_id, corpus=hot_corpus),
                    client=active_client,
                )
                hot_run_id = str(hot_payload.get("hot_run_id") or "")
                lane_runs["hot_rerun"] = {
                    "status": "completed",
                    "role": "hot_rerun",
                    "run_id": hot_run_id,
                    "corpus_key": hot_corpus.key,
                    "corpus_csv": hot_corpus.csv_path,
                    "artifact_paths": _lane_artifact_paths(hot_run_id) if hot_run_id else {},
                    "comparison_json": str(hot_payload.get("comparison_json") or ""),
                    "comparison_csv": str(hot_payload.get("comparison_csv") or ""),
                    "gate_json": str(hot_payload.get("gate_json") or ""),
                    "report_path": str(hot_payload.get("report_path") or ""),
                    "hot_gate": dict(hot_payload.get("hot_gate") or {}),
                }
            except Exception as exc:
                lane_runs["hot_rerun"] = {
                    "status": "failed",
                    "role": "hot_rerun",
                    "corpus_key": hot_corpus.key,
                    "corpus_csv": hot_corpus.csv_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if bool(args.fail_fast):
                    raise
            _write_suite_progress(
                suite_run_id=suite_run_id,
                lane_runs=lane_runs,
                pending_roles=[],
            )

        lane_publishability_rows: list[dict[str, Any]] = []
        baseline_audit_rows: list[dict[str, Any]] = []
        sample_size_rows: list[dict[str, Any]] = []
        headline_seed_claim_rows: list[dict[str, Any]] = []
        focused_publication_rows: list[dict[str, Any]] = []
        for role, payload in direct_payloads.items():
            corpus = corpora[_lane_corpus_key(role)]
            lane_publishability_rows.extend(_publishability_rows_for_lane(role=role, payload=payload, corpus=corpus))
            baseline_audit_rows.extend(
                _baseline_audit_rows_for_lane(role=role, payload=payload, corpus=corpus, suite_args=args)
            )
            sample_size_rows.extend(_sample_size_rows_for_lane(role=role, payload=payload, corpus=corpus))
            headline_seed_claim_rows.extend(_headline_seed_claim_rows_for_lane(role=role, payload=payload, corpus=corpus))
            rows = payload.get("rows")
            if role in FOCUSED_ROLES and isinstance(rows, Sequence):
                focused_publication_rows.extend(
                    _annotated_rows(
                        rows,
                        role=role,
                        lane_run_id=str(payload.get("run_id") or ""),
                        corpus_key=corpus.key,
                    )
                )

        focused_decision_region_payload = _build_focused_decision_region_publication(
            suite_run_id=suite_run_id,
            rows=focused_publication_rows,
        )
        focused_decision_region_json_path = write_json_artifact(
            suite_run_id,
            "focused_decision_region_publication.json",
            focused_decision_region_payload,
        )
        focused_decision_region_md_path = write_text_artifact(
            suite_run_id,
            "focused_decision_region_publication.md",
            _render_focused_decision_region_publication_markdown(focused_decision_region_payload),
        )
        witness_distributions_payload = _build_witness_distributions(
            suite_run_id=suite_run_id,
            rows=focused_publication_rows,
        )
        witness_distributions_json_path = write_json_artifact(
            suite_run_id,
            "witness_distributions.json",
            witness_distributions_payload,
        )
        witness_distributions_md_path = write_text_artifact(
            suite_run_id,
            "witness_distributions.md",
            _render_witness_distributions_markdown(witness_distributions_payload),
        )
        failure_atlas_payload = _build_failure_atlas(
            suite_run_id=suite_run_id,
            rows=focused_publication_rows,
        )
        failure_atlas_rows = list(failure_atlas_payload.get("rows") or [])

        publishability_rows_path = write_csv_artifact(
            suite_run_id,
            "lane_publishability_summary.csv",
            fieldnames=_ordered_fieldnames(lane_publishability_rows),
            rows=lane_publishability_rows,
        )
        publishability_json_path = write_json_artifact(
            suite_run_id,
            "lane_publishability_summary.json",
            {"rows": lane_publishability_rows},
        )
        baseline_audit_csv_path = write_csv_artifact(
            suite_run_id,
            "universal_baseline_audit.csv",
            fieldnames=_ordered_fieldnames(baseline_audit_rows),
            rows=baseline_audit_rows,
        )
        baseline_audit_json_path = write_json_artifact(
            suite_run_id,
            "universal_baseline_audit.json",
            {"rows": baseline_audit_rows},
        )
        sample_size_csv_path = write_csv_artifact(
            suite_run_id,
            "sample_size_gate_summary.csv",
            fieldnames=_ordered_fieldnames(sample_size_rows),
            rows=sample_size_rows,
        )
        sample_size_json_path = write_json_artifact(
            suite_run_id,
            "sample_size_gate_summary.json",
            {"rows": sample_size_rows},
        )
        headline_seed_claims_csv_path = write_csv_artifact(
            suite_run_id,
            "headline_seed_claims_summary.csv",
            fieldnames=_ordered_fieldnames(headline_seed_claim_rows),
            rows=headline_seed_claim_rows,
        )
        headline_seed_claims_json_path = write_json_artifact(
            suite_run_id,
            "headline_seed_claims_summary.json",
            {"rows": headline_seed_claim_rows},
        )
        failure_atlas_json_path = write_json_artifact(
            suite_run_id,
            "failure_atlas.json",
            failure_atlas_payload,
        )
        failure_atlas_md_path = write_text_artifact(
            suite_run_id,
            "failure_atlas.md",
            _render_failure_atlas_markdown(failure_atlas_payload),
        )
        suite_artifact_dir = artifact_dir_for_run(suite_run_id)
        results_artifact_path = str(suite_artifact_dir / "results.json")
        index_json_artifact_path = str(suite_artifact_dir / "index.json")
        failure_atlas_lane_metadata_payload = _failure_atlas_lane_metadata(
            suite_run_id=suite_run_id,
            payload=failure_atlas_payload,
            failure_atlas_json_path=str(failure_atlas_json_path),
            failure_atlas_md_path=str(failure_atlas_md_path),
            results_path=results_artifact_path,
            index_json_path=index_json_artifact_path,
        )
        failure_atlas_lane_metadata_path = write_json_artifact(
            suite_run_id,
            "failure_atlas_lane_metadata.json",
            failure_atlas_lane_metadata_payload,
        )
        _write_baseline_identity_manifests(
            suite_run_id=suite_run_id,
            suite_artifact_dir=suite_artifact_dir,
            suite_args=args,
        )
        verdict = _publishability_verdict_payload(
            lane_publishability_rows=lane_publishability_rows,
            baseline_audit_rows=baseline_audit_rows,
            failure_atlas_rows=failure_atlas_rows,
            sample_size_rows=sample_size_rows,
            headline_seed_claim_rows=headline_seed_claim_rows,
            hot_payload=hot_payload,
            suite_artifact_dir=suite_artifact_dir,
        )
        verdict_json_path = write_json_artifact(suite_run_id, "publishability_verdict.json", verdict)
        verdict_md_path = write_text_artifact(
            suite_run_id,
            "publishability_assessment.md",
            _publishability_markdown(
                suite_run_id=suite_run_id,
                verdict=verdict,
                lane_publishability_rows=lane_publishability_rows,
                baseline_audit_rows=baseline_audit_rows,
                sample_size_rows=sample_size_rows,
                headline_seed_claim_rows=headline_seed_claim_rows,
            ),
        )
        index_json_path = write_json_artifact(
            suite_run_id,
            "index.json",
            {
                "schema_version": SUITE_SCHEMA_VERSION,
                "suite_run_id": suite_run_id,
                "generated_at_utc": _now(),
                "preflight_path": str(preflight_path),
                "suite_sources_path": str(suite_sources_path),
                "lane_runs": lane_runs,
                "lane_publishability_summary_csv": str(publishability_rows_path),
                "lane_publishability_summary_json": str(publishability_json_path),
                "universal_baseline_audit_csv": str(baseline_audit_csv_path),
                "universal_baseline_audit_json": str(baseline_audit_json_path),
                "sample_size_gate_summary_csv": str(sample_size_csv_path),
                "sample_size_gate_summary_json": str(sample_size_json_path),
                "headline_seed_claims_summary_csv": str(headline_seed_claims_csv_path),
                "headline_seed_claims_summary_json": str(headline_seed_claims_json_path),
                "focused_decision_region_publication_json": str(focused_decision_region_json_path),
                "focused_decision_region_publication_md": str(focused_decision_region_md_path),
                "witness_distributions_json": str(witness_distributions_json_path),
                "witness_distributions_md": str(witness_distributions_md_path),
                "failure_atlas_json": str(failure_atlas_json_path),
                "failure_atlas_md": str(failure_atlas_md_path),
                "failure_atlas_lane_metadata_json": str(failure_atlas_lane_metadata_path),
                "publishability_verdict_json": str(verdict_json_path),
                "publishability_assessment_md": str(verdict_md_path),
            },
        )
        index_md_path = write_text_artifact(
            suite_run_id,
            "index.md",
            _suite_index_markdown(
                suite_run_id=suite_run_id,
                corpora=corpora,
                lane_runs=lane_runs,
                verdict=verdict,
                publication_outputs={
                    "focused_decision_region_publication_json": str(focused_decision_region_json_path),
                    "focused_decision_region_publication_md": str(focused_decision_region_md_path),
                    "witness_distributions_json": str(witness_distributions_json_path),
                    "witness_distributions_md": str(witness_distributions_md_path),
                    "failure_atlas_json": str(failure_atlas_json_path),
                    "failure_atlas_md": str(failure_atlas_md_path),
                    "failure_atlas_lane_metadata_json": str(failure_atlas_lane_metadata_path),
                },
            ),
        )
        metadata_path = write_json_artifact(
            suite_run_id,
            "metadata.json",
            {
                "schema_version": SUITE_SCHEMA_VERSION,
                "suite_run_id": suite_run_id,
                "generated_at_utc": _now(),
                "arguments": {key: value for key, value in vars(args).items()},
                "preflight_summary": preflight_summary,
                "lane_count_requested": len(DIRECT_SUITE_ROLES) + 1,
                "lane_count_completed": sum(1 for record in lane_runs.values() if str(record.get("status")) == "completed"),
                "focused_decision_region_row_count": focused_decision_region_payload.get("row_count"),
                "witness_distribution_row_count": witness_distributions_payload.get("row_count"),
                "failure_atlas_row_count": failure_atlas_payload.get("row_count"),
                "failure_atlas_lane_id": FAILURE_ATLAS_LANE_ID,
                "failure_atlas_lane_status": failure_atlas_lane_metadata_payload.get("lane_status"),
                "failure_atlas_lane_metadata_json": str(failure_atlas_lane_metadata_path),
            },
        )
        results_path = write_json_artifact(
            suite_run_id,
            "results.json",
            {
                "schema_version": SUITE_SCHEMA_VERSION,
                "suite_run_id": suite_run_id,
                "generated_at_utc": _now(),
                "lane_runs": lane_runs,
                "lane_publishability_rows": lane_publishability_rows,
                "baseline_audit_rows": baseline_audit_rows,
                "sample_size_rows": sample_size_rows,
                "headline_seed_claim_rows": headline_seed_claim_rows,
                "focused_decision_region_publication": focused_decision_region_payload,
                "witness_distributions": witness_distributions_payload,
                "failure_atlas_rows": failure_atlas_rows,
                "failure_atlas": failure_atlas_payload,
                "failure_atlas_lane_metadata": failure_atlas_lane_metadata_payload,
                "failure_atlas_lane_metadata_json": str(failure_atlas_lane_metadata_path),
                "publishability_verdict": verdict,
            },
        )
        manifest_path = write_manifest(
            suite_run_id,
            {
                "request": {
                    "full_latest_suite": {
                        "suite_run_id": suite_run_id,
                        "roles": [*DIRECT_SUITE_ROLES, "hot_rerun"],
                        "out_dir": str(args.out_dir),
                    }
                },
                "execution": {
                    "metadata": str(metadata_path),
                    "results": str(results_path),
                    "index_json": str(index_json_path),
                    "index_md": str(index_md_path),
                    "focused_decision_region_publication_json": str(focused_decision_region_json_path),
                    "focused_decision_region_publication_md": str(focused_decision_region_md_path),
                    "witness_distributions_json": str(witness_distributions_json_path),
                    "witness_distributions_md": str(witness_distributions_md_path),
                    "failure_atlas_json": str(failure_atlas_json_path),
                    "failure_atlas_md": str(failure_atlas_md_path),
                    "failure_atlas_lane_metadata_json": str(failure_atlas_lane_metadata_path),
                    "publishability_verdict_json": str(verdict_json_path),
                    "publishability_assessment_md": str(verdict_md_path),
                },
            },
        )
        return {
            "suite_run_id": suite_run_id,
            "preflight_path": str(preflight_path),
            "suite_sources_path": str(suite_sources_path),
            "lane_runs": lane_runs,
            "lane_publishability_summary_csv": str(publishability_rows_path),
            "lane_publishability_summary_json": str(publishability_json_path),
            "universal_baseline_audit_csv": str(baseline_audit_csv_path),
            "universal_baseline_audit_json": str(baseline_audit_json_path),
            "sample_size_gate_summary_csv": str(sample_size_csv_path),
            "sample_size_gate_summary_json": str(sample_size_json_path),
            "headline_seed_claims_summary_csv": str(headline_seed_claims_csv_path),
            "headline_seed_claims_summary_json": str(headline_seed_claims_json_path),
            "focused_decision_region_publication_json": str(focused_decision_region_json_path),
            "focused_decision_region_publication_md": str(focused_decision_region_md_path),
            "witness_distributions_json": str(witness_distributions_json_path),
            "witness_distributions_md": str(witness_distributions_md_path),
            "failure_atlas_json": str(failure_atlas_json_path),
            "failure_atlas_md": str(failure_atlas_md_path),
            "failure_atlas_lane_metadata_json": str(failure_atlas_lane_metadata_path),
            "publishability_verdict_json": str(verdict_json_path),
            "publishability_assessment_md": str(verdict_md_path),
            "index_json": str(index_json_path),
            "index_md": str(index_md_path),
            "metadata_json": str(metadata_path),
            "results_json": str(results_path),
            "manifest_path": str(manifest_path),
        }
    finally:
        settings.out_dir = old_out_dir


def _load_suite_root_lane_runs(suite_artifact_dir: Path) -> dict[str, dict[str, Any]]:
    for filename in ("index.json", "results.json", "metadata.json"):
        payload = _load_json_dict(suite_artifact_dir / filename) or {}
        lane_runs = payload.get("lane_runs")
        if isinstance(lane_runs, Mapping):
            return {
                str(role): dict(record)
                for role, record in lane_runs.items()
                if str(role).strip() and isinstance(record, Mapping)
            }
    raise RuntimeError("suite_root_republish_missing_lane_runs")


def _normalize_hot_payload(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    if "hot_gate" not in payload and "all_green" in payload:
        return {"hot_gate": dict(payload)}
    return payload


def _load_suite_root_hot_payload(
    *,
    lane_runs: Mapping[str, Mapping[str, Any]],
    hot_payload_path: str | Path | None = None,
) -> Mapping[str, Any] | None:
    if hot_payload_path is not None:
        return _normalize_hot_payload(_read_json_payload(str(hot_payload_path)))

    hot_record = lane_runs.get("hot_rerun")
    if not isinstance(hot_record, Mapping):
        return None

    candidate_gate_paths: list[Path] = []
    gate_json = hot_record.get("gate_json")
    if gate_json:
        candidate_gate_paths.append(Path(str(gate_json)))

    artifact_paths = hot_record.get("artifact_paths")
    if isinstance(artifact_paths, Mapping):
        artifact_gate_json = artifact_paths.get("gate_json")
        if artifact_gate_json:
            candidate_gate_paths.append(Path(str(artifact_gate_json)))
        artifact_dir = artifact_paths.get("artifact_dir")
        if artifact_dir:
            candidate_gate_paths.append(Path(str(artifact_dir)) / "hot_rerun_gate.json")

    hot_run_id = str(hot_record.get("run_id") or "").strip()
    if hot_run_id:
        candidate_gate_paths.append(artifact_dir_for_run(hot_run_id) / "hot_rerun_gate.json")

    seen_paths: set[str] = set()
    for gate_path in candidate_gate_paths:
        gate_path_key = str(gate_path)
        if gate_path_key in seen_paths:
            continue
        seen_paths.add(gate_path_key)
        hot_gate = _load_json_dict(gate_path)
        normalized = _normalize_hot_payload(hot_gate)
        if isinstance(normalized, Mapping):
            return normalized

    return _normalize_hot_payload(hot_record)


def _existing_lane_artifact_paths(lane_artifact_dir: Path) -> dict[str, str]:
    artifact_paths: dict[str, str] = {"artifact_dir": str(lane_artifact_dir)}
    candidate_files = {
        "results_json": "results.json",
        "thesis_results_json": "thesis_results.json",
        "summary_json": "thesis_summary.json",
        "summary_by_cohort_json": "thesis_summary_by_cohort.json",
        "metrics_json": "thesis_metrics.json",
        "plots_json": "thesis_plots.json",
        "lane_metadata_json": "lane_metadata.json",
        "cohort_composition_json": "cohort_composition.json",
        "evaluation_manifest_json": "evaluation_manifest.json",
        "report_md": "thesis_report.md",
    }
    for key, filename in candidate_files.items():
        path = lane_artifact_dir / filename
        if path.exists():
            artifact_paths[key] = str(path)
    return artifact_paths


def _lane_run_from_existing_companion_bundle(
    *,
    suite_run_id: str,
    role: str,
) -> dict[str, Any] | None:
    lane_run_id = f"{suite_run_id}_{role}"
    lane_artifact_dir = artifact_dir_for_run(lane_run_id)
    if not lane_artifact_dir.exists():
        return None
    lane_metadata = _load_json_dict(lane_artifact_dir / "lane_metadata.json") or {}
    observed_sample_size = lane_metadata.get("observed_sample_size")
    if not isinstance(observed_sample_size, Mapping):
        observed_sample_size = {}
    return {
        "status": "completed",
        "role": role,
        "run_id": lane_run_id,
        "corpus_key": _lane_corpus_key(role),
        "corpus_csv": str(lane_artifact_dir / "od_corpus.csv") if (lane_artifact_dir / "od_corpus.csv").exists() else "",
        "row_count": _safe_int(observed_sample_size.get("unique_od_count"))
        or _safe_int(observed_sample_size.get("row_count")),
        "artifact_paths": _existing_lane_artifact_paths(lane_artifact_dir),
        "lane_metadata": lane_metadata,
    }


def _repair_suite_root_lane_runs(
    *,
    suite_run_id: str,
    lane_runs: Mapping[str, Mapping[str, Any]],
    sample_size_rows: Sequence[Mapping[str, Any]],
    hot_payload: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    repaired = {
        str(role): dict(record)
        for role, record in lane_runs.items()
        if str(role).strip() and isinstance(record, Mapping)
    }
    for role in DIRECT_SUITE_ROLES:
        if role in repaired:
            continue
        companion_record = _lane_run_from_existing_companion_bundle(
            suite_run_id=suite_run_id,
            role=role,
        )
        if companion_record is not None:
            repaired[role] = companion_record

    sample_size_by_role = {
        str(row.get("lane_role")): row
        for row in sample_size_rows
        if isinstance(row, Mapping) and str(row.get("lane_role") or "").strip()
    }
    observed_field_map = {
        "observed_row_count": "row_count",
        "observed_unique_od_count": "unique_od_count",
        "observed_unique_row_seed_count": "unique_row_seed_count",
        "observed_effective_cert_world_count": "effective_cert_world_count",
        "observed_requested_cert_world_count": "requested_cert_world_count",
        "observed_probabilistic_world_count": "probabilistic_world_count",
        "observed_audit_world_count": "audit_world_count",
        "observed_audited_route_pair_count": "audited_route_pair_count",
    }
    for role, sample_row in sample_size_by_role.items():
        record = dict(repaired.get(role) or {"status": "completed", "role": role})
        lane_metadata = dict(record.get("lane_metadata") or {})
        observed_sample_size = dict(lane_metadata.get("observed_sample_size") or {})
        for source_key, target_key in observed_field_map.items():
            value = sample_row.get(source_key)
            if value is not None:
                observed_sample_size[target_key] = value
        evaluation_requirement_met = sample_row.get("evaluation_requirement_met")
        if evaluation_requirement_met is not None:
            observed_sample_size["evaluation_size_requirement_met"] = evaluation_requirement_met
        lane_metadata["observed_sample_size"] = observed_sample_size
        evaluation_size_requirement = dict(lane_metadata.get("evaluation_size_requirement") or {})
        if sample_row.get("evaluation_requirement_id") is not None:
            evaluation_size_requirement["requirement_id"] = sample_row.get("evaluation_requirement_id")
        if sample_row.get("evaluation_requirement_unit") is not None:
            evaluation_size_requirement["unit"] = sample_row.get("evaluation_requirement_unit")
        if sample_row.get("evaluation_requirement_minimum") is not None:
            evaluation_size_requirement["minimum"] = sample_row.get("evaluation_requirement_minimum")
        if sample_row.get("evaluation_requirement_description") is not None:
            evaluation_size_requirement["minimum_description"] = sample_row.get("evaluation_requirement_description")
        if evaluation_size_requirement:
            lane_metadata["evaluation_size_requirement"] = evaluation_size_requirement
        record["lane_metadata"] = lane_metadata
        record["status"] = "completed"
        record["role"] = role
        if not str(record.get("corpus_key") or "").strip():
            record["corpus_key"] = str(sample_row.get("corpus_key") or _lane_corpus_key(role))
        if not str(record.get("corpus_label") or "").strip() and str(sample_row.get("corpus_label") or "").strip():
            record["corpus_label"] = str(sample_row.get("corpus_label"))
        if _safe_int(record.get("row_count")) <= 0:
            record["row_count"] = _safe_int(sample_row.get("corpus_row_count")) or _safe_int(
                sample_row.get("observed_unique_od_count")
            )
        repaired[role] = record

    hot_gate = None
    if isinstance(hot_payload, Mapping):
        maybe_gate = hot_payload.get("hot_gate")
        hot_gate = dict(maybe_gate) if isinstance(maybe_gate, Mapping) else dict(hot_payload)
    if hot_gate:
        hot_record = dict(repaired.get("hot_rerun") or {"status": "completed", "role": "hot_rerun"})
        hot_record["status"] = "completed"
        hot_record["role"] = "hot_rerun"
        hot_run_id = str(hot_gate.get("hot_run_id") or hot_record.get("run_id") or "").strip()
        if hot_run_id:
            hot_record["run_id"] = hot_run_id
            hot_artifact_dir = artifact_dir_for_run(hot_run_id)
            existing_artifact_paths = dict(hot_record.get("artifact_paths") or {})
            existing_artifact_paths.update(_existing_lane_artifact_paths(hot_artifact_dir))
            hot_record["artifact_paths"] = existing_artifact_paths
            comparison_json = hot_artifact_dir / "hot_rerun_vs_cold_comparison.json"
            comparison_csv = hot_artifact_dir / "hot_rerun_vs_cold_comparison.csv"
            gate_json = hot_artifact_dir / "hot_rerun_gate.json"
            report_md = hot_artifact_dir / "hot_rerun_report.md"
            if comparison_json.exists():
                hot_record["comparison_json"] = str(comparison_json)
            if comparison_csv.exists():
                hot_record["comparison_csv"] = str(comparison_csv)
            if gate_json.exists():
                hot_record["gate_json"] = str(gate_json)
            if report_md.exists():
                hot_record["report_path"] = str(report_md)
        hot_record["hot_gate"] = hot_gate
        repaired["hot_rerun"] = hot_record

    return repaired


def _load_suite_root_corpora(suite_artifact_dir: Path) -> dict[str, dict[str, Any]]:
    suite_sources = _load_json_dict(suite_artifact_dir / "suite_sources.json") or {}
    corpora = suite_sources.get("corpora")
    if not isinstance(corpora, Mapping):
        return {}
    return {
        str(key): dict(value)
        for key, value in corpora.items()
        if str(key).strip() and isinstance(value, Mapping)
    }


def _load_lane_results_payload_from_record(
    *,
    suite_artifact_dir: Path,
    role: str,
    lane_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    artifact_paths = lane_record.get("artifact_paths")
    candidate_paths: list[str] = []
    if isinstance(artifact_paths, Mapping):
        candidate_paths.extend(
            str(path)
            for path in (
                artifact_paths.get("results_json"),
                artifact_paths.get("thesis_results_json"),
            )
            if str(path or "").strip()
        )
    lane_run_id = str(lane_record.get("run_id") or "").strip()
    if lane_run_id:
        candidate_paths.extend(
            [
                str(artifact_dir_for_run(lane_run_id) / "results.json"),
                str(artifact_dir_for_run(lane_run_id) / "thesis_results.json"),
            ]
        )
    for candidate_path in candidate_paths:
        payload = _read_json_payload(candidate_path)
        if payload:
            return payload
    raise RuntimeError(f"suite_root_republish_missing_lane_results:{role}")


def _republish_corpus_artifact(
    *,
    role: str,
    lane_record: Mapping[str, Any],
    lane_payload: Mapping[str, Any],
    corpora_payload: Mapping[str, Mapping[str, Any]],
) -> CorpusArtifact:
    canonical_corpus_key = _lane_corpus_key(role)
    recorded_corpus_key = str(lane_record.get("corpus_key") or "").strip()
    if canonical_corpus_key and canonical_corpus_key in corpora_payload:
        corpus_key = canonical_corpus_key
    elif recorded_corpus_key:
        corpus_key = recorded_corpus_key
    else:
        corpus_key = canonical_corpus_key
    corpus_entry = corpora_payload.get(corpus_key)
    lane_metadata = lane_payload.get("lane_metadata")
    if not isinstance(lane_metadata, Mapping):
        lane_metadata = {}
    observed_sample_size = lane_metadata.get("observed_sample_size")
    if not isinstance(observed_sample_size, Mapping):
        observed_sample_size = {}
    row_count = _safe_int(corpus_entry.get("row_count")) if isinstance(corpus_entry, Mapping) else 0
    if row_count <= 0:
        row_count = _safe_int(observed_sample_size.get("unique_od_count"))
    if row_count <= 0:
        row_count = _safe_int(observed_sample_size.get("row_count"))
    if row_count <= 0:
        rows = lane_payload.get("rows")
        row_count = len(rows) if isinstance(rows, Sequence) else 0
    label = str(corpus_entry.get("label") if isinstance(corpus_entry, Mapping) else "").strip()
    if not label:
        label = f"{corpus_key.title()} corpus"
    csv_path = str(corpus_entry.get("csv_path") if isinstance(corpus_entry, Mapping) else "")
    json_path = str(corpus_entry.get("json_path") if isinstance(corpus_entry, Mapping) else "")
    summary_path = str(corpus_entry.get("summary_path") if isinstance(corpus_entry, Mapping) else "")
    source_summary_path = str(corpus_entry.get("source_summary_path") if isinstance(corpus_entry, Mapping) else "")
    return CorpusArtifact(
        key=corpus_key,
        label=label,
        row_count=row_count,
        csv_path=csv_path,
        json_path=json_path,
        summary_path=summary_path,
        source_summary_path=source_summary_path,
    )


def republish_suite_root_from_existing_lane_dirs(
    *,
    suite_run_id: str,
    out_dir: str | Path,
    hot_payload_path: str | Path | None = None,
) -> dict[str, Any]:
    old_out_dir = settings.out_dir
    settings.out_dir = Path(out_dir)
    try:
        suite_artifact_dir = artifact_dir_for_run(suite_run_id)
        if not suite_artifact_dir.exists():
            raise FileNotFoundError(f"suite_artifact_dir_missing:{suite_artifact_dir}")

        lane_runs = _load_suite_root_lane_runs(suite_artifact_dir)
        corpora_payload = _load_suite_root_corpora(suite_artifact_dir)
        metadata_payload = _load_json_dict(suite_artifact_dir / "metadata.json") or {}
        suite_args = None
        metadata_arguments = metadata_payload.get("arguments")
        if isinstance(metadata_arguments, Mapping) and metadata_arguments:
            suite_args = argparse.Namespace(**dict(metadata_arguments))
        hot_payload = _load_suite_root_hot_payload(
            lane_runs=lane_runs,
            hot_payload_path=hot_payload_path,
        )
        lane_publishability_rows: list[dict[str, Any]] = []
        baseline_audit_rows: list[dict[str, Any]] = []
        sample_size_rows: list[dict[str, Any]] = []
        headline_seed_claim_rows: list[dict[str, Any]] = []
        published_roles: list[str] = []

        for role in sorted(set(DIRECT_SUITE_ROLES).intersection(lane_runs.keys())):
            lane_record = lane_runs.get(role)
            if not isinstance(lane_record, Mapping):
                continue
            lane_payload = _load_lane_results_payload_from_record(
                suite_artifact_dir=suite_artifact_dir,
                role=role,
                lane_record=lane_record,
            )
            corpus = _republish_corpus_artifact(
                role=role,
                lane_record=lane_record,
                lane_payload=lane_payload,
                corpora_payload=corpora_payload,
            )
            lane_publishability_rows.extend(
                _publishability_rows_for_lane(role=role, payload=lane_payload, corpus=corpus)
            )
            baseline_audit_rows.extend(
                _baseline_audit_rows_for_lane(
                    role=role,
                    payload=lane_payload,
                    corpus=corpus,
                    suite_args=suite_args,
                    suite_metadata=metadata_payload,
                )
            )
            sample_size_rows.extend(
                _sample_size_rows_for_lane(role=role, payload=lane_payload, corpus=corpus)
            )
            headline_seed_claim_rows.extend(
                _headline_seed_claim_rows_for_lane(role=role, payload=lane_payload, corpus=corpus)
            )
            published_roles.append(role)

        if not lane_publishability_rows:
            raise RuntimeError("suite_root_republish_no_lane_publishability_rows")
        if not sample_size_rows:
            raise RuntimeError("suite_root_republish_no_sample_size_rows")
        if not headline_seed_claim_rows:
            raise RuntimeError("suite_root_republish_no_headline_seed_rows")

        lane_publishability_rows_path = write_csv_artifact(
            suite_run_id,
            "lane_publishability_summary.csv",
            fieldnames=_ordered_fieldnames(lane_publishability_rows),
            rows=lane_publishability_rows,
        )
        lane_publishability_json_path = write_json_artifact(
            suite_run_id,
            "lane_publishability_summary.json",
            {"rows": lane_publishability_rows},
        )
        baseline_audit_rows_path = write_csv_artifact(
            suite_run_id,
            "universal_baseline_audit.csv",
            fieldnames=_ordered_fieldnames(baseline_audit_rows),
            rows=baseline_audit_rows,
        )
        baseline_audit_json_path = write_json_artifact(
            suite_run_id,
            "universal_baseline_audit.json",
            {"rows": baseline_audit_rows},
        )
        sample_size_rows_path = write_csv_artifact(
            suite_run_id,
            "sample_size_gate_summary.csv",
            fieldnames=_ordered_fieldnames(sample_size_rows),
            rows=sample_size_rows,
        )
        sample_size_json_path = write_json_artifact(
            suite_run_id,
            "sample_size_gate_summary.json",
            {"rows": sample_size_rows},
        )
        headline_seed_claim_rows_path = write_csv_artifact(
            suite_run_id,
            "headline_seed_claims_summary.csv",
            fieldnames=_ordered_fieldnames(headline_seed_claim_rows),
            rows=headline_seed_claim_rows,
        )
        headline_seed_claim_json_path = write_json_artifact(
            suite_run_id,
            "headline_seed_claims_summary.json",
            {"rows": headline_seed_claim_rows},
        )
        _write_baseline_identity_manifests(
            suite_run_id=suite_run_id,
            suite_artifact_dir=suite_artifact_dir,
            suite_args=suite_args,
        )

        republished = repair_failure_atlas_suite_root(
            suite_run_id=suite_run_id,
            out_dir=out_dir,
            hot_payload_path=hot_payload_path,
        )
        republished.update(
            {
                "repaired_roles": published_roles,
                "lane_publishability_summary_csv": str(lane_publishability_rows_path),
                "lane_publishability_summary_json": str(lane_publishability_json_path),
                "universal_baseline_audit_csv": str(baseline_audit_rows_path),
                "universal_baseline_audit_json": str(baseline_audit_json_path),
                "sample_size_gate_summary_csv": str(sample_size_rows_path),
                "sample_size_gate_summary_json": str(sample_size_json_path),
                "headline_seed_claims_summary_csv": str(headline_seed_claim_rows_path),
                "headline_seed_claims_summary_json": str(headline_seed_claim_json_path),
            }
        )
        return republished
    finally:
        settings.out_dir = old_out_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_full_latest_suite(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _corpus_key_for_role(role: str) -> str:
    if role == "optional_stopping_coverage":
        return "optional_stopping"
    if role == "synthetic_ground_truth":
        return "synthetic"
    if role in FOCUSED_ROLES:
        return "focused"
    if role in BROAD_ROLES:
        return "broad"
    return "broad"


def _clone_namespace(source: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(source))


def _build_eval_args(
    suite_args: argparse.Namespace,
    *,
    role: str,
    run_id: str,
    corpus_csv: str,
) -> argparse.Namespace:
    args = _clone_namespace(_evaluation_defaults())
    args.corpus_csv = corpus_csv
    args.corpus_json = None
    args.out_dir = str(suite_args.out_dir)
    args.run_id = run_id
    args.seed = int(suite_args.seed)
    args.backend_url = str(suite_args.backend_url)
    args.in_process_backend = bool(suite_args.in_process_backend)
    args.ready_timeout_seconds = float(suite_args.ready_timeout_seconds)
    args.ready_poll_seconds = float(suite_args.ready_poll_seconds)
    args.route_timeout_seconds = float(suite_args.route_timeout_seconds)
    args.model_version = str(suite_args.model_version)
    args.optimization_mode = str(suite_args.optimization_mode)
    args.vehicle_type = str(suite_args.vehicle_type)
    args.scenario_mode = str(suite_args.scenario_mode)
    args.departure_time_utc = suite_args.departure_time_utc
    args.max_alternatives = int(suite_args.max_alternatives)
    args.search_budget = int(suite_args.search_budget)
    args.evidence_budget = int(suite_args.evidence_budget)
    args.world_count = int(suite_args.world_count)
    args.certificate_threshold = float(suite_args.certificate_threshold)
    args.tau_stop = float(suite_args.tau_stop)
    args.stochastic_enabled = bool(suite_args.stochastic_enabled)
    args.stochastic_samples = int(suite_args.stochastic_samples)
    args.weight_time = float(suite_args.weight_time)
    args.weight_money = float(suite_args.weight_money)
    args.weight_co2 = float(suite_args.weight_co2)
    args.fail_soft = bool(suite_args.fail_soft)
    args.disable_tolls = bool(suite_args.disable_tolls)
    args.baseline_refinement_policy = str(suite_args.baseline_refinement_policy)
    args.ors_baseline_policy = str(suite_args.ors_baseline_policy)
    args.ors_snapshot_mode = str(suite_args.ors_snapshot_mode)
    args.ors_snapshot_path = suite_args.ors_snapshot_path
    args.auto_enrich_corpus_ambiguity = bool(suite_args.auto_enrich_corpus_ambiguity)
    args.allow_proxy_ors = bool(suite_args.allow_proxy_ors)
    args.allow_evidence_fallbacks = bool(suite_args.allow_evidence_fallbacks)
    args.cache_mode = "cold"
    args.cold_cache_scope = "thesis_cold"
    args.max_od = 0
    args.evaluation_suite_role = role
    args.seed_repeat_count = (
        max(1, int(suite_args.headline_seed_repeat_count)) if role in HEADLINE_ROLES else 1
    )
    args.seed_repeat_step = int(suite_args.headline_seed_repeat_step)
    return args


def _build_hot_args(
    suite_args: argparse.Namespace,
    *,
    pair_run_id: str,
    corpus_csv: str,
) -> argparse.Namespace:
    args = _clone_namespace(_hot_runner_defaults())
    args.corpus_csv = corpus_csv
    args.corpus_json = None
    args.out_dir = str(suite_args.out_dir)
    args.run_id = pair_run_id
    args.pair_run_id = pair_run_id
    args.cold_run_id = f"{pair_run_id}_cold"
    args.hot_run_id = f"{pair_run_id}_hot"
    args.seed = int(suite_args.seed)
    args.backend_url = str(suite_args.backend_url)
    args.in_process_backend = bool(suite_args.in_process_backend)
    args.ready_timeout_seconds = float(suite_args.ready_timeout_seconds)
    args.ready_poll_seconds = float(suite_args.ready_poll_seconds)
    args.route_timeout_seconds = float(suite_args.route_timeout_seconds)
    args.model_version = str(suite_args.model_version)
    args.optimization_mode = str(suite_args.optimization_mode)
    args.vehicle_type = str(suite_args.vehicle_type)
    args.scenario_mode = str(suite_args.scenario_mode)
    args.departure_time_utc = suite_args.departure_time_utc
    args.max_alternatives = int(suite_args.max_alternatives)
    args.search_budget = int(suite_args.search_budget)
    args.evidence_budget = int(suite_args.evidence_budget)
    args.world_count = int(suite_args.world_count)
    args.certificate_threshold = float(suite_args.certificate_threshold)
    args.tau_stop = float(suite_args.tau_stop)
    args.stochastic_enabled = bool(suite_args.stochastic_enabled)
    args.stochastic_samples = int(suite_args.stochastic_samples)
    args.weight_time = float(suite_args.weight_time)
    args.weight_money = float(suite_args.weight_money)
    args.weight_co2 = float(suite_args.weight_co2)
    args.fail_soft = bool(suite_args.fail_soft)
    args.disable_tolls = bool(suite_args.disable_tolls)
    args.baseline_refinement_policy = str(suite_args.baseline_refinement_policy)
    args.ors_baseline_policy = str(suite_args.ors_baseline_policy)
    args.ors_snapshot_mode = str(suite_args.ors_snapshot_mode)
    args.ors_snapshot_path = suite_args.ors_snapshot_path
    args.auto_enrich_corpus_ambiguity = bool(suite_args.auto_enrich_corpus_ambiguity)
    args.allow_proxy_ors = bool(suite_args.allow_proxy_ors)
    args.allow_evidence_fallbacks = bool(suite_args.allow_evidence_fallbacks)
    return args


def _path_if_exists(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return str(path) if path.exists() else None


def _headline_seed_artifact_presence(payload: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "headline_seed_summary": _path_if_exists(payload.get("headline_seed_summary_path")) is not None,
        "headline_seed_runs": _path_if_exists(payload.get("headline_seed_runs_path")) is not None,
        "headline_seed_claims": _path_if_exists(payload.get("headline_seed_claims_path")) is not None,
        "headline_seed_reviewer_summary": _path_if_exists(payload.get("headline_seed_reviewer_summary_path")) is not None,
        "headline_seed_report_table": (
            _path_if_exists(payload.get("headline_seed_report_table_json")) is not None
        ),
    }


def _lane_entry_from_payload(
    *,
    role: str,
    run_id: str,
    corpus: CorpusArtifact,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    summary_rows = payload.get("summary_rows")
    if not isinstance(summary_rows, list):
        summary_rows = []
    lane_metadata = payload.get("lane_metadata")
    if not isinstance(lane_metadata, Mapping):
        lane_metadata = LANE_METADATA_DEFAULTS.get(role, {})
    row_count = sum(_safe_int(row.get("row_count"), 0) for row in summary_rows if isinstance(row, Mapping))
    return {
        "role": role,
        "label": EVALUATION_SUITE_ROLE_DEFAULTS.get(role, {}).get("label", role),
        "scope": EVALUATION_SUITE_ROLE_DEFAULTS.get(role, {}).get("scope"),
        "focus": EVALUATION_SUITE_ROLE_DEFAULTS.get(role, {}).get("focus"),
        "status": "completed",
        "run_id": run_id,
        "artifact_dir": str(artifact_dir_for_run(run_id)),
        "corpus_key": corpus.key,
        "corpus_label": corpus.label,
        "corpus_csv": corpus.csv_path,
        "corpus_json": corpus.json_path,
        "corpus_summary": corpus.summary_path,
        "source_corpus_summary": corpus.source_summary_path,
        "row_count": row_count,
        "success_row_count": _safe_int(payload.get("success_row_count"), 0),
        "failure_row_count": _safe_int(payload.get("failure_row_count"), 0),
        "results_csv": str(payload.get("results_csv") or ""),
        "summary_csv": str(payload.get("summary_csv") or ""),
        "summary_by_cohort_csv": str(payload.get("summary_by_cohort_csv") or ""),
        "thesis_report": str(payload.get("thesis_report") or ""),
        "methods_appendix": str(payload.get("methods_appendix") or ""),
        "evaluation_manifest": str(payload.get("evaluation_manifest") or ""),
        "manifest_path": str(payload.get("manifest_path") or ""),
        "lane_metadata_path": str(payload.get("lane_metadata_path") or ""),
        "lane_metadata": dict(lane_metadata),
        "headline_seed_repeat_requested": int(
            payload.get("lane_metadata", {}).get("configured_seed_count", 1)
            if isinstance(payload.get("lane_metadata"), Mapping)
            else 1
        ),
        "headline_seed_artifacts_present": _headline_seed_artifact_presence(payload),
        "output_artifact_validation": dict(payload.get("output_artifact_validation") or {}),
        "failure_breakdown": dict(payload.get("failure_breakdown") or {}),
        "successful_variants": list(payload.get("successful_variants") or []),
        "failed_variants": list(payload.get("failed_variants") or []),
        "summary_rows": summary_rows,
    }


def _lane_error_entry(
    *,
    role: str,
    run_id: str,
    corpus: CorpusArtifact,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "role": role,
        "label": EVALUATION_SUITE_ROLE_DEFAULTS.get(role, {}).get("label", role),
        "scope": EVALUATION_SUITE_ROLE_DEFAULTS.get(role, {}).get("scope"),
        "focus": EVALUATION_SUITE_ROLE_DEFAULTS.get(role, {}).get("focus"),
        "status": "failed",
        "run_id": run_id,
        "artifact_dir": str(artifact_dir_for_run(run_id)),
        "corpus_key": corpus.key,
        "corpus_label": corpus.label,
        "corpus_csv": corpus.csv_path,
        "corpus_json": corpus.json_path,
        "corpus_summary": corpus.summary_path,
        "source_corpus_summary": corpus.source_summary_path,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _annotated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    lane_run_id: str,
    corpus_key: str,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["_suite_role"] = role
        payload["_suite_lane_run_id"] = lane_run_id
        payload["_suite_corpus_key"] = corpus_key
        annotated.append(payload)
    return annotated


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _payload_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _world_kind_counts(manifest: Mapping[str, Any] | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(manifest, Mapping):
        return counts
    worlds = manifest.get("worlds")
    if not isinstance(worlds, Sequence) or isinstance(worlds, (str, bytes)):
        return counts
    for world in worlds:
        if not isinstance(world, Mapping):
            continue
        kind = str(world.get("world_kind") or "").strip() or "unknown"
        counts[kind] += 1
    return counts


def _optional_stopping_proof_rollups(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifact_cache: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _payload_rows(payload):
        variant_id = str(row.get("variant_id") or "").strip()
        if variant_id not in OPTIONAL_STOPPING_PROOF_VARIANTS:
            continue
        artifact_run_id = str(row.get("artifact_run_id") or "").strip()
        if not artifact_run_id:
            continue
        snapshot = _route_bundle_snapshot(artifact_run_id, cache=artifact_cache)
        winner_confidence = snapshot.get("winner_confidence_state")
        if not isinstance(winner_confidence, Mapping):
            continue
        grouped[variant_id].append(
            {
                "artifact_run_id": artifact_run_id,
                "snapshot": snapshot,
                "winner_confidence_state": dict(winner_confidence),
            }
        )
    rollups: dict[str, dict[str, Any]] = {}
    for variant_id, entries in grouped.items():
        refc_row_count = len(entries)
        method_recorded_count = 0
        delta_recorded_count = 0
        supported_method_count = 0
        validity_tested_count = 0
        validity_pass_count = 0
        total_world_count = 0
        total_unique_world_count = 0
        methods: set[str] = set()
        ci_methods: set[str] = set()
        delta_sources: set[str] = set()
        delta_schedules: set[str] = set()
        deltas: list[float] = []
        confidence_paths: list[str] = []
        for entry in entries:
            snapshot = entry["snapshot"]
            winner_confidence = entry["winner_confidence_state"]
            confidence_paths.append(str(Path(snapshot["artifact_dir"]) / "winner_confidence_state.json"))
            method = str(winner_confidence.get("method") or "").strip()
            delta = _safe_float(winner_confidence.get("delta"))
            lower_bound = _safe_float(winner_confidence.get("lower_bound"))
            upper_bound = _safe_float(winner_confidence.get("upper_bound"))
            empirical_win = _safe_float(winner_confidence.get("empirical_win"))
            if method:
                method_recorded_count += 1
                methods.add(method)
            if delta is not None:
                delta_recorded_count += 1
                deltas.append(delta)
            if method in OPTIONAL_STOPPING_ANYTIME_METHODS and delta is not None:
                supported_method_count += 1
            trace_state = winner_confidence.get("stopping_valid_trace_state")
            if isinstance(trace_state, Mapping):
                ci_method = str(trace_state.get("confidence_interval_method") or "").strip()
                if ci_method:
                    ci_methods.add(ci_method)
                delta_source = str(trace_state.get("delta_source") or "").strip()
                if delta_source:
                    delta_sources.add(delta_source)
                delta_schedule = str(trace_state.get("delta_schedule") or "").strip()
                if delta_schedule:
                    delta_schedules.add(delta_schedule)
                world_count = _maybe_int(trace_state.get("world_count"))
                if world_count is not None:
                    total_world_count += world_count
                unique_world_count = _maybe_int(trace_state.get("unique_world_count"))
                if unique_world_count is not None:
                    total_unique_world_count += unique_world_count
            if (
                empirical_win is not None
                and lower_bound is not None
                and upper_bound is not None
            ):
                validity_tested_count += 1
                if lower_bound <= empirical_win <= upper_bound:
                    validity_pass_count += 1
        validity_rate = (
            round(validity_pass_count / float(validity_tested_count), 6)
            if validity_tested_count
            else None
        )
        validity_violation_rate = (
            round(max(0.0, 1.0 - float(validity_rate)), 6)
            if validity_rate is not None
            else None
        )
        guaranteed_coverage_floor = (
            round(min(1.0 - delta for delta in deltas), 6)
            if deltas
            else None
        )
        rollups[variant_id] = {
            "optional_stopping_refc_row_count": refc_row_count,
            "optional_stopping_method_recorded_rate": round(method_recorded_count / float(refc_row_count), 6)
            if refc_row_count
            else None,
            "optional_stopping_delta_recorded_rate": round(delta_recorded_count / float(refc_row_count), 6)
            if refc_row_count
            else None,
            "optional_stopping_supported_method_rate": round(supported_method_count / float(refc_row_count), 6)
            if refc_row_count
            else None,
            "optional_stopping_validity_tested_rate": round(validity_tested_count / float(refc_row_count), 6)
            if refc_row_count
            else None,
            "optional_stopping_validity_check_rate": validity_rate,
            "optional_stopping_validity_violation_rate": validity_violation_rate,
            "optional_stopping_guaranteed_coverage_floor": guaranteed_coverage_floor,
            "optional_stopping_required_coverage_floor": OPTIONAL_STOPPING_REQUIRED_COVERAGE_FLOOR,
            "optional_stopping_total_world_count": total_world_count or None,
            "optional_stopping_total_unique_world_count": total_unique_world_count or None,
            "optional_stopping_methods_json": json.dumps(sorted(methods)),
            "optional_stopping_confidence_interval_methods_json": json.dumps(sorted(ci_methods)),
            "optional_stopping_delta_values_json": json.dumps(sorted(round(delta, 6) for delta in deltas)),
            "optional_stopping_delta_sources_json": json.dumps(sorted(delta_sources)),
            "optional_stopping_delta_schedules_json": json.dumps(sorted(delta_schedules)),
            "optional_stopping_confidence_state_paths_json": json.dumps(confidence_paths),
        }
    return rollups


def _perturbation_proof_rollups(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifact_cache: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _payload_rows(payload):
        variant_id = str(row.get("variant_id") or "").strip()
        if variant_id not in PERTURBATION_PROOF_VARIANTS:
            continue
        artifact_run_id = str(row.get("artifact_run_id") or "").strip()
        if not artifact_run_id:
            continue
        snapshot = _route_bundle_snapshot(artifact_run_id, cache=artifact_cache)
        flip_radius = snapshot.get("flip_radius_summary")
        if not isinstance(flip_radius, Mapping):
            continue
        grouped[variant_id].append(
            {
                "artifact_run_id": artifact_run_id,
                "snapshot": snapshot,
                "flip_radius_summary": dict(flip_radius),
                "sampled_world_manifest": snapshot.get("sampled_world_manifest"),
            }
        )
    rollups: dict[str, dict[str, Any]] = {}
    for variant_id, entries in grouped.items():
        measured_row_count = 0
        real_violation_count = 0
        exact_synthetic_supported_row_count = 0
        exact_synthetic_violation_row_count = 0
        exact_synthetic_world_count = 0
        sampled_world_count = 0
        minimum_flip_budgets: list[float] = []
        world_kind_counts: Counter[str] = Counter()
        flip_paths: list[str] = []
        for entry in entries:
            snapshot = entry["snapshot"]
            flip_radius = entry["flip_radius_summary"]
            flip_paths.append(str(Path(snapshot["artifact_dir"]) / "flip_radius_summary.json"))
            measured_row_count += 1
            provenance = flip_radius.get("provenance")
            provenance = provenance if isinstance(provenance, Mapping) else {}
            unsafe_challenger_present = provenance.get("unsafe_challenger_present") is True
            minimum_flip_budget = _safe_float(flip_radius.get("minimum_flip_budget"))
            if minimum_flip_budget is not None:
                minimum_flip_budgets.append(minimum_flip_budget)
            row_has_violation = unsafe_challenger_present or minimum_flip_budget is None or minimum_flip_budget <= 0.0
            if row_has_violation:
                real_violation_count += 1
            world_counts = _world_kind_counts(entry["sampled_world_manifest"])
            world_kind_counts.update(world_counts)
            sampled_world_count += world_counts.get("sampled", 0)
            row_exact_synthetic_world_count = sum(
                count
                for kind, count in world_counts.items()
                if str(kind or "").strip() != "sampled"
            )
            exact_synthetic_world_count += row_exact_synthetic_world_count
            if row_exact_synthetic_world_count > 0:
                exact_synthetic_supported_row_count += 1
                if row_has_violation:
                    exact_synthetic_violation_row_count += 1
        rollups[variant_id] = {
            "perturbation_flip_radius_rows_evaluated": measured_row_count,
            "real_lane_flip_radius_violation_rate": round(real_violation_count / float(measured_row_count), 6)
            if measured_row_count
            else None,
            "exact_synthetic_flip_radius_violation_rate": round(
                exact_synthetic_violation_row_count / float(exact_synthetic_supported_row_count),
                6,
            )
            if exact_synthetic_supported_row_count
            else None,
            "perturbation_exact_synthetic_world_count": exact_synthetic_world_count,
            "perturbation_sampled_world_count": sampled_world_count,
            "perturbation_world_kind_counts_json": json.dumps(dict(sorted(world_kind_counts.items()))),
            "perturbation_minimum_flip_budget_min": round(min(minimum_flip_budgets), 6)
            if minimum_flip_budgets
            else None,
            "perturbation_flip_radius_paths_json": json.dumps(flip_paths),
        }
    return rollups


def _perturbation_exact_synthetic_world_count(payload_rows: Sequence[Mapping[str, Any]]) -> int:
    artifact_cache: dict[str, dict[str, Any]] = {}
    total = 0
    for row in payload_rows:
        artifact_run_id = str(row.get("artifact_run_id") or "").strip()
        if not artifact_run_id:
            continue
        snapshot = _route_bundle_snapshot(artifact_run_id, cache=artifact_cache)
        total += sum(
            count
            for kind, count in _world_kind_counts(snapshot.get("sampled_world_manifest")).items()
            if str(kind or "").strip() != "sampled"
        )
    return total


def _route_bundle_snapshot(
    artifact_run_id: str,
    *,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if artifact_run_id in cache:
        return cache[artifact_run_id]
    artifact_dir = artifact_dir_for_run(artifact_run_id)
    snapshot = {
        "artifact_run_id": artifact_run_id,
        "artifact_dir": str(artifact_dir),
        "certificate_witness": _load_json_dict(artifact_dir / "certificate_witness.json"),
        "certified_set_summary": _load_json_dict(artifact_dir / "certified_set_summary.json"),
        "decision_package": _load_json_dict(artifact_dir / "decision_package.json"),
        "decision_region_summary": _load_json_dict(artifact_dir / "decision_region_summary.json"),
        "flip_radius_summary": _load_json_dict(artifact_dir / "flip_radius_summary.json"),
        "sampled_world_manifest": _load_json_dict(artifact_dir / "sampled_world_manifest.json"),
        "voi_stop_certificate": _load_json_dict(artifact_dir / "voi_stop_certificate.json"),
        "winner_confidence_state": _load_json_dict(artifact_dir / "winner_confidence_state.json"),
    }
    cache[artifact_run_id] = snapshot
    return snapshot


def _nearest_rank(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(float(quantile) * len(ordered)) - 1))
    return float(ordered[index])


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items


def _failure_root_cause_family(
    *,
    support_flag: Any,
    support_status: str,
    abstention_class: str,
    failure_reason: str,
    stop_reason: str,
    raw_tags: Sequence[str],
) -> str:
    normalized_tags = [str(tag or "").strip().lower() for tag in raw_tags if str(tag or "").strip()]
    normalized_abstention = str(abstention_class or "").strip().lower()
    normalized_failure = str(failure_reason or "").strip().lower()
    normalized_stop = str(stop_reason or "").strip().lower()

    if support_flag is False or (support_status and support_status != "supported"):
        return "support_failure"
    if normalized_abstention == "uncertified_due_to_budget" or "budget" in normalized_failure or "budget" in normalized_stop:
        return "budget_cut"
    if normalized_abstention == "uncertified_due_to_preference" or any(
        token.startswith("preference_direction:")
        or "preference" in token
        or "tradeoff:" in token
        or "guard:" in token
        for token in normalized_tags
    ):
        return "preference_ambiguity"
    if any(
        "proxy" in token
        or "audit_bias" in token
        or "bias_correction" in token
        or "positivity" in token
        or "overlap_failure" in token
        for token in normalized_tags
    ):
        return "proxy_bias"
    if normalized_abstention == "uncertified_due_to_search" or any(
        "challenger" in token
        or "search_incomplete" in token
        or "pairwise_gap_unresolved" in token
        or token == "winner_lcb_below_threshold"
        or token.startswith("boundary:")
        for token in normalized_tags
    ):
        return "hidden_challenger"
    return "other"


def _outcome_bucket(
    *,
    certified_set_size: int,
    terminal_type: str,
) -> str:
    normalized = str(terminal_type or "").strip().lower()
    if normalized in {"abstained", "abstention"}:
        return "abstention"
    if certified_set_size == 1:
        return "singleton"
    if certified_set_size > 1:
        return "set"
    return normalized or "unknown"


def _build_focused_decision_region_publication(
    *,
    suite_run_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact_cache: dict[str, dict[str, Any]] = {}
    publication_rows: list[dict[str, Any]] = []
    counts_by_boundary: Counter[str] = Counter()
    counts_by_lane_role: Counter[str] = Counter()
    counts_by_support_status: Counter[str] = Counter()
    counts_by_bucket: Counter[str] = Counter()
    counts_by_atlas_kind: Counter[str] = Counter()
    for row in rows:
        artifact_run_id = str(row.get("artifact_run_id") or "").strip()
        if not artifact_run_id:
            continue
        snapshot = _route_bundle_snapshot(artifact_run_id, cache=artifact_cache)
        decision_region = snapshot.get("decision_region_summary")
        certified_set = snapshot.get("certified_set_summary")
        if not isinstance(decision_region, Mapping):
            continue
        certified_set_size = (
            _safe_int((certified_set or {}).get("set_size"), 0) if isinstance(certified_set, Mapping) else 0
        )
        terminal_type = str(row.get("preference_terminal_type") or row.get("terminal_type") or "")
        bucket = _outcome_bucket(certified_set_size=certified_set_size, terminal_type=terminal_type)
        publication_row = {
            "lane_role": str(row.get("_suite_role") or ""),
            "lane_run_id": str(row.get("_suite_lane_run_id") or ""),
            "variant_id": str(row.get("variant_id") or ""),
            "od_id": str(row.get("od_id") or ""),
            "profile_id": str(row.get("profile_id") or ""),
            "corpus_key": str(row.get("_suite_corpus_key") or ""),
            "bucket": bucket,
            "artifact_run_id": artifact_run_id,
            "artifact_dir": snapshot["artifact_dir"],
            "decision_region_summary_json": str(Path(snapshot["artifact_dir"]) / "decision_region_summary.json"),
            "certificate_witness_json": str(Path(snapshot["artifact_dir"]) / "certificate_witness.json"),
            "certified_set_summary_json": str(Path(snapshot["artifact_dir"]) / "certified_set_summary.json"),
            "nearest_certificate_boundary": str(decision_region.get("nearest_certificate_boundary") or ""),
            "active_challenger_id": str(decision_region.get("active_challenger_id") or ""),
            "dominant_evidence_family": str(decision_region.get("dominant_evidence_family") or ""),
            "most_fragile_preference_direction": str(
                decision_region.get("most_fragile_preference_direction") or ""
            ),
            "minimum_joint_perturbation": _safe_float(decision_region.get("minimum_joint_perturbation")),
            "nearest_threat_axis": str(decision_region.get("nearest_threat_axis") or ""),
            "support_status": str(decision_region.get("support_status") or ""),
            "support_bin": str(decision_region.get("support_bin") or ""),
            "calibration_bin": str(decision_region.get("calibration_bin") or ""),
            "calibration_policy_version": str(decision_region.get("calibration_policy_version") or ""),
            "selected_certificate_basis": str(decision_region.get("selected_certificate_basis") or ""),
            "nearest_challenger_gap_lower_bound": _safe_float(
                decision_region.get("nearest_challenger_gap_lower_bound")
            ),
            "nearest_challenger_audit_sensitivity": _safe_float(
                decision_region.get("nearest_challenger_audit_sensitivity")
            ),
            "nearest_challenger_radius": _safe_float(decision_region.get("nearest_challenger_radius")),
            "nearest_challenger_flip_budget": _safe_float(
                decision_region.get("nearest_challenger_flip_budget")
            ),
            "route_fragility_family_count": _safe_int(
                decision_region.get("route_fragility_family_count"),
                0,
            ),
            "atlas_kind": str(decision_region.get("atlas_kind") or ""),
            "root_cause_tags": _string_list(decision_region.get("root_cause_tags")),
        }
        publication_rows.append(publication_row)
        counts_by_boundary[publication_row["nearest_certificate_boundary"] or "unknown"] += 1
        counts_by_lane_role[publication_row["lane_role"] or "unknown"] += 1
        counts_by_support_status[publication_row["support_status"] or "unknown"] += 1
        counts_by_bucket[publication_row["bucket"] or "unknown"] += 1
        counts_by_atlas_kind[publication_row["atlas_kind"] or "unknown"] += 1
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "created_at": _now(),
        "suite_run_id": suite_run_id,
        "visualization_spec": {
            "kind": "decision_region_scatter",
            "scope": "focused_lanes",
            "x_field": "minimum_joint_perturbation",
            "y_field": "nearest_challenger_gap_lower_bound",
            "color_field": "nearest_certificate_boundary",
            "symbol_field": "support_status",
        },
        "row_count": len(publication_rows),
        "counts_by_boundary": dict(sorted(counts_by_boundary.items())),
        "counts_by_lane_role": dict(sorted(counts_by_lane_role.items())),
        "counts_by_support_status": dict(sorted(counts_by_support_status.items())),
        "counts_by_bucket": dict(sorted(counts_by_bucket.items())),
        "counts_by_atlas_kind": dict(sorted(counts_by_atlas_kind.items())),
        "rows": publication_rows,
    }


def _render_focused_decision_region_publication_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Focused Decision Region Publication",
        "",
        f"- Suite Run ID: `{payload.get('suite_run_id')}`",
        f"- Created At: `{payload.get('created_at')}`",
        f"- Row Count: `{payload.get('row_count')}`",
        "",
        "## Visualization Spec",
        "",
    ]
    spec = payload.get("visualization_spec")
    if isinstance(spec, Mapping):
        lines.extend(
            [
                f"- Kind: `{spec.get('kind')}`",
                f"- Scope: `{spec.get('scope')}`",
                f"- X field: `{spec.get('x_field')}`",
                f"- Y field: `{spec.get('y_field')}`",
                f"- Color field: `{spec.get('color_field')}`",
                f"- Symbol field: `{spec.get('symbol_field')}`",
            ]
        )
    lines.extend(["", "## Counts By Boundary", ""])
    counts_by_boundary = payload.get("counts_by_boundary")
    if isinstance(counts_by_boundary, Mapping) and counts_by_boundary:
        for key, value in counts_by_boundary.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No focused decision-region rows were available.")
    lines.extend(["", "## Example Rows", ""])
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        for row in rows[:25]:
            lines.append(
                f"- `{row.get('lane_role')}` / `{row.get('variant_id')}` / `{row.get('od_id')}`: "
                f"bucket=`{row.get('bucket')}` boundary=`{row.get('nearest_certificate_boundary')}` "
                f"support=`{row.get('support_status')}` axis=`{row.get('nearest_threat_axis')}` "
                f"challenger=`{row.get('active_challenger_id')}` artifact=`{row.get('decision_region_summary_json')}`"
            )
    else:
        lines.append("- No example rows recorded.")
    return "\n".join(lines) + "\n"


def _build_witness_distributions(
    *,
    suite_run_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact_cache: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lane_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    variant_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exemplars: list[dict[str, Any]] = []
    for row in rows:
        artifact_run_id = str(row.get("artifact_run_id") or "").strip()
        if not artifact_run_id:
            continue
        snapshot = _route_bundle_snapshot(artifact_run_id, cache=artifact_cache)
        witness = snapshot.get("certificate_witness")
        certified_set = snapshot.get("certified_set_summary")
        if not isinstance(witness, Mapping):
            continue
        witness_size = _safe_float(witness.get("witness_size"))
        if witness_size is None:
            continue
        certified_set_size = _safe_int((certified_set or {}).get("set_size"), 0) if isinstance(certified_set, Mapping) else 0
        terminal_type = str(row.get("preference_terminal_type") or row.get("terminal_type") or "").strip().lower()
        bucket = _outcome_bucket(certified_set_size=certified_set_size, terminal_type=terminal_type)
        explanation_sparsity = _safe_float(witness.get("explanation_sparsity"))
        if explanation_sparsity is None:
            explanation_sparsity = _safe_float(witness.get("witness_sparsity"))
        record = {
            "bucket": bucket,
            "lane_role": str(row.get("_suite_role") or ""),
            "variant_id": str(row.get("variant_id") or ""),
            "od_id": str(row.get("od_id") or ""),
            "profile_id": str(row.get("profile_id") or ""),
            "artifact_run_id": artifact_run_id,
            "artifact_dir": snapshot["artifact_dir"],
            "witness_size": int(witness_size),
            "witness_sparsity": _safe_float(witness.get("witness_sparsity")),
            "explanation_sparsity": explanation_sparsity,
            "certified_set_size": certified_set_size,
            "active_challenger_count": len(witness.get("active_challenger_ids") or [])
            if isinstance(witness.get("active_challenger_ids"), list)
            else 0,
            "active_evidence_family_count": _safe_int(witness.get("active_evidence_family_count"), 0),
            "active_preference_constraint_count": _safe_int(
                witness.get("active_preference_constraint_count"),
                0,
            ),
            "support_condition_count": _safe_int(witness.get("support_condition_count"), 0),
            "action_step_count": _safe_int(witness.get("action_step_count"), 0),
        }
        grouped[bucket].append(record)
        lane_grouped[record["lane_role"]].append(record)
        variant_grouped[record["variant_id"]].append(record)
        exemplars.append(record)

    def _distribution_rows(records_by_key: Mapping[str, list[dict[str, Any]]], *, key_name: str) -> list[dict[str, Any]]:
        summary_rows: list[dict[str, Any]] = []
        for key, bucket_rows in sorted(records_by_key.items()):
            sizes = [float(item["witness_size"]) for item in bucket_rows]
            sparsities = [
                float(item["witness_sparsity"])
                for item in bucket_rows
                if item.get("witness_sparsity") is not None
            ]
            explanation_sparsities = [
                float(item["explanation_sparsity"])
                for item in bucket_rows
                if item.get("explanation_sparsity") is not None
            ]
            summary_rows.append(
                {
                    key_name: key,
                    "count": len(bucket_rows),
                    "median_witness_size": _nearest_rank(sizes, 0.5),
                    "p90_witness_size": _nearest_rank(sizes, 0.9),
                    "max_witness_size": max(sizes) if sizes else None,
                    "mean_witness_sparsity": (
                        round(sum(sparsities) / float(len(sparsities)), 6) if sparsities else None
                    ),
                    "mean_explanation_sparsity": (
                        round(sum(explanation_sparsities) / float(len(explanation_sparsities)), 6)
                        if explanation_sparsities
                        else None
                    ),
                    "median_explanation_sparsity": _nearest_rank(explanation_sparsities, 0.5),
                    "p90_explanation_sparsity": _nearest_rank(explanation_sparsities, 0.9),
                }
            )
        return summary_rows

    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "created_at": _now(),
        "suite_run_id": suite_run_id,
        "minimality_metric": "witness_size",
        "sparsity_metric": "explanation_sparsity",
        "row_count": len(exemplars),
        "bucket_rows": _distribution_rows(grouped, key_name="bucket"),
        "by_lane_role": _distribution_rows(lane_grouped, key_name="lane_role"),
        "by_variant_id": _distribution_rows(variant_grouped, key_name="variant_id"),
        "exemplars": sorted(exemplars, key=lambda item: (-int(item["witness_size"]), item["artifact_run_id"]))[:50],
    }


def _render_witness_distributions_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Witness Distributions",
        "",
        f"- Suite Run ID: `{payload.get('suite_run_id')}`",
        f"- Created At: `{payload.get('created_at')}`",
        f"- Minimality metric: `{payload.get('minimality_metric')}`",
        f"- Sparsity metric: `{payload.get('sparsity_metric')}`",
        "",
        "## By Terminal Bucket",
        "",
    ]
    bucket_rows = payload.get("bucket_rows")
    if isinstance(bucket_rows, list) and bucket_rows:
        for row in bucket_rows:
            lines.append(
                f"- `{row.get('bucket')}`: count={row.get('count')} "
                f"median={row.get('median_witness_size')} "
                f"p90={row.get('p90_witness_size')} max={row.get('max_witness_size')} "
                f"median_explanation_sparsity={row.get('median_explanation_sparsity')} "
                f"p90_explanation_sparsity={row.get('p90_explanation_sparsity')} "
                f"mean_explanation_sparsity={row.get('mean_explanation_sparsity')}"
            )
    else:
        lines.append("- No witness rows were available.")
    lines.extend(["", "## Largest Exemplars", ""])
    exemplars = payload.get("exemplars")
    if isinstance(exemplars, list) and exemplars:
        for row in exemplars[:20]:
            lines.append(
                f"- `{row.get('bucket')}` / `{row.get('lane_role')}` / `{row.get('variant_id')}` "
                f"`{row.get('od_id')}`: witness_size={row.get('witness_size')} "
                f"explanation_sparsity={row.get('explanation_sparsity')} "
                f"certified_set_size={row.get('certified_set_size')} "
                f"artifact=`{row.get('artifact_dir')}`"
            )
    else:
        lines.append("- No witness exemplars recorded.")
    return "\n".join(lines) + "\n"


def _build_failure_atlas(
    *,
    suite_run_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact_cache: dict[str, dict[str, Any]] = {}
    atlas_rows: list[dict[str, Any]] = []
    root_cause_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    coverage_class_counts: Counter[str] = Counter()
    abstention_class_counts: Counter[str] = Counter()
    abstention_examples_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root_cause_family_counts: Counter[str] = Counter()
    all_coverage_classes = tuple((*FAILURE_ATLAS_REQUIRED_KINDS, *FAILURE_ATLAS_OPTIONAL_KINDS))
    for row in rows:
        artifact_run_id = str(row.get("artifact_run_id") or "").strip()
        if not artifact_run_id:
            continue
        snapshot = _route_bundle_snapshot(artifact_run_id, cache=artifact_cache)
        certified_set = snapshot.get("certified_set_summary") or {}
        decision_package = snapshot.get("decision_package") or {}
        decision_region = snapshot.get("decision_region_summary") or {}
        stop_certificate = snapshot.get("voi_stop_certificate") or {}
        certificate_witness = snapshot.get("certificate_witness") or {}
        certified_set_witness = certified_set.get("witness") if isinstance(certified_set.get("witness"), Mapping) else {}
        singleton_not_justified = certified_set_witness.get("singleton_not_justified_reasons")
        if not isinstance(singleton_not_justified, list):
            singleton_not_justified = []
        excluded_route_safety = certified_set_witness.get("excluded_route_safety_reasons")
        if not isinstance(excluded_route_safety, list):
            excluded_route_safety = []
        abstention_payload = decision_package.get("abstention") if isinstance(decision_package, Mapping) else {}
        abstention_payload = abstention_payload if isinstance(abstention_payload, Mapping) else {}
        abstention_summary = (
            decision_package.get("abstention_summary") if isinstance(decision_package, Mapping) else {}
        )
        abstention_summary = abstention_summary if isinstance(abstention_summary, Mapping) else {}
        failure_reason = str(row.get("failure_reason") or "").strip()
        terminal_type = str(row.get("preference_terminal_type") or row.get("terminal_type") or "").strip().lower()
        support_status = str(
            decision_region.get("support_status")
            or certificate_witness.get("support_status")
            or ((certified_set.get("artifact_provenance") or {}).get("support_status"))
            or ""
        ).strip()
        proxy_only_fraction = _safe_float(row.get("proxy_only_fraction"))
        positivity_ok = row.get("positivity_ok")
        weak_overlap_detected = bool(row.get("weak_overlap_detected"))
        audited_route_pair_count = _safe_float(row.get("audited_route_pair_count"))
        support_downgrade_signal = bool(
            weak_overlap_detected
            or (
                positivity_ok is False
                and proxy_only_fraction is not None
                and proxy_only_fraction >= 0.99
                and (audited_route_pair_count is None or audited_route_pair_count <= 0.0)
            )
        )
        if not support_status:
            support_richness = _safe_float(row.get("support_richness"))
            support_bin_hint = str(
                row.get("support_bin")
                or decision_region.get("support_bin")
                or certificate_witness.get("support_bin")
                or ""
            ).strip().lower()
            if support_bin_hint in {"weak_support", "unsupported"} or (
                support_richness is not None and support_richness <= 0.45
            ):
                support_status = "unsupported"
            elif support_bin_hint in {"strong_support", "supported"} or (
                support_richness is not None and support_richness >= 0.75
            ):
                support_status = "supported"
        if support_downgrade_signal:
            support_status = "unsupported"
        support_bin = str(
            decision_region.get("support_bin")
            or certificate_witness.get("support_bin")
            or ""
        ).strip()
        support_flag = certified_set.get("support_flag", row.get("support_flag"))
        if support_downgrade_signal:
            support_flag = False
        certified_set_size = _safe_int(certified_set.get("set_size"), 0)
        abstention_class = str(
            abstention_payload.get("reason_code")
            or abstention_summary.get("reason_code")
            or (
                stop_certificate.get("stop_reason")
                if terminal_type in {"abstained", "abstention"}
                else ""
            )
            or ""
        ).strip()
        coverage_classes: list[str] = []
        if terminal_type in {"abstained", "abstention"} or abstention_class:
            coverage_classes.append("abstention")
        if singleton_not_justified:
            coverage_classes.append("wrong_singleton")
        if support_flag is False or (support_status and support_status != "supported"):
            coverage_classes.append("support_downgrade")
        certified_set_violation_reasons: list[str] = []
        if certified_set_size > 1 and support_flag is False:
            certified_set_violation_reasons.append("support_flag_false")
        outside_routes_safely_excluded = certified_set_witness.get("outside_routes_safely_excluded")
        if outside_routes_safely_excluded is False:
            certified_set_violation_reasons.append("outside_routes_not_safely_excluded")
        certified_set_violation_reasons.extend(str(reason) for reason in excluded_route_safety if str(reason).strip())
        if certified_set_violation_reasons:
            coverage_classes.append("certified_set_violation")
        if failure_reason:
            coverage_classes.append("route_failure")
        deduped_coverage_classes = list(dict.fromkeys(coverage_classes))
        if support_flag is False or (support_status and support_status != "supported"):
            atlas_kind = "support_downgrade"
        elif terminal_type in {"abstained", "abstention"}:
            atlas_kind = "abstention"
        elif singleton_not_justified:
            atlas_kind = "wrong_singleton"
        elif certified_set_violation_reasons:
            atlas_kind = "certified_set_violation"
        elif failure_reason:
            atlas_kind = "route_failure"
        else:
            continue
        if not deduped_coverage_classes:
            continue
        root_cause_tags = [
            *[
                token
                for token in _string_list(decision_region.get("root_cause_tags"))
                + _string_list(certificate_witness.get("root_cause_tags"))
                if token
            ],
            *[token for token in [failure_reason, *singleton_not_justified, *excluded_route_safety] if token],
        ]
        if stop_certificate.get("stop_reason"):
            root_cause_tags.append(str(stop_certificate.get("stop_reason")))
        deduped_tags: list[str] = []
        seen_tags: set[str] = set()
        for tag in root_cause_tags:
            if tag in seen_tags:
                continue
            deduped_tags.append(tag)
            seen_tags.add(tag)
        if support_downgrade_signal:
            for tag in ("support_failure", "positivity_failure", "proxy_only_support_gap"):
                if tag not in seen_tags:
                    deduped_tags.append(tag)
                    seen_tags.add(tag)
        root_cause_family = _failure_root_cause_family(
            support_flag=support_flag,
            support_status=support_status,
            abstention_class=abstention_class,
            failure_reason=failure_reason,
            stop_reason=str(stop_certificate.get("stop_reason") or ""),
            raw_tags=deduped_tags,
        )
        root_cause_family_counts[root_cause_family] += 1
        normalized_root_cause_tags = list(dict.fromkeys([root_cause_family, *deduped_tags]))
        for tag in normalized_root_cause_tags:
            root_cause_counts[str(tag)] += 1
        kind_counts[atlas_kind] += 1
        for coverage_class in deduped_coverage_classes:
            coverage_class_counts[coverage_class] += 1
        primary_root_cause_tag = root_cause_family
        primary_root_cause_detail_tag = deduped_tags[0] if deduped_tags else None
        cohort = str(row.get("corpus_group") or row.get("_suite_role") or "").strip() or None
        active_challenger_id = (
            decision_region.get("active_challenger_id")
            or certificate_witness.get("targeted_challenger_route_id")
        )
        dominant_fragility_family = decision_region.get("dominant_evidence_family")
        artifact_pointers = {
            "artifact_dir": snapshot["artifact_dir"],
            "decision_region_summary_json": str(Path(snapshot["artifact_dir"]) / "decision_region_summary.json"),
            "certificate_witness_json": str(Path(snapshot["artifact_dir"]) / "certificate_witness.json"),
            "certified_set_summary_json": str(Path(snapshot["artifact_dir"]) / "certified_set_summary.json"),
            "decision_package_json": str(Path(snapshot["artifact_dir"]) / "decision_package.json"),
            "voi_stop_certificate_json": str(Path(snapshot["artifact_dir"]) / "voi_stop_certificate.json"),
            "results_json": str(Path(snapshot["artifact_dir"]) / "results.json"),
        }
        row_id = "::".join(
            token
            for token in (
                str(row.get("_suite_role") or "").strip(),
                str(row.get("variant_id") or "").strip(),
                str(row.get("od_id") or "").strip(),
                str(row.get("profile_id") or "").strip(),
            )
            if token
        ) or artifact_run_id
        if "abstention" in deduped_coverage_classes:
            normalized_abstention_class = abstention_class or "unknown"
            abstention_class_counts[normalized_abstention_class] += 1
            if len(abstention_examples_by_class[normalized_abstention_class]) < FAILURE_ATLAS_ABSTENTION_EXAMPLE_TARGET:
                abstention_examples_by_class[normalized_abstention_class].append(
                    {
                        "row_id": row_id,
                        "cohort": cohort,
                        "lane_role": str(row.get("_suite_role") or ""),
                        "variant_id": str(row.get("variant_id") or ""),
                        "od_id": str(row.get("od_id") or ""),
                        "support_status": support_status or None,
                        "active_challenger_id": active_challenger_id,
                        "dominant_fragility_family": dominant_fragility_family,
                        "controller_stop_reason": stop_certificate.get("stop_reason"),
                        "root_cause_tag": primary_root_cause_tag,
                        "root_cause_detail_tag": primary_root_cause_detail_tag,
                        "artifact_pointers": dict(artifact_pointers),
                    }
                )
        atlas_rows.append(
            {
                "row_id": row_id,
                "atlas_kind": atlas_kind,
                "coverage_classes": deduped_coverage_classes,
                "lane_role": str(row.get("_suite_role") or ""),
                "lane_run_id": str(row.get("_suite_lane_run_id") or ""),
                "variant_id": str(row.get("variant_id") or ""),
                "od_id": str(row.get("od_id") or ""),
                "profile_id": str(row.get("profile_id") or ""),
                "corpus_group": row.get("corpus_group"),
                "cohort": cohort,
                "support_flag": support_flag,
                "support_status": support_status or None,
                "support_bin": support_bin or None,
                "calibration_bin": str(
                    decision_region.get("calibration_bin") or certificate_witness.get("calibration_bin") or ""
                )
                or None,
                "selected_certificate_basis": str(
                    decision_region.get("selected_certificate_basis")
                    or certificate_witness.get("selected_certificate_basis")
                    or ""
                )
                or None,
                "certified_set_size": certified_set.get("set_size"),
                "failure_reason": failure_reason or None,
                "terminal_type": terminal_type or None,
                "abstention_class": abstention_class or None,
                "singleton_not_justified_reasons": singleton_not_justified,
                "certified_set_violation_reasons": certified_set_violation_reasons,
                "excluded_route_safety_reasons": excluded_route_safety,
                "active_challenger_id": active_challenger_id,
                "targeted_challenger_route_id": certificate_witness.get("targeted_challenger_route_id"),
                "dominant_fragility_family": dominant_fragility_family,
                "route_fragility_family_count": decision_region.get("route_fragility_family_count"),
                "controller_stop_reason": stop_certificate.get("stop_reason"),
                "root_cause_tag": primary_root_cause_tag,
                "root_cause_detail_tag": primary_root_cause_detail_tag,
                "root_cause_family": root_cause_family,
                "root_cause_tags": normalized_root_cause_tags,
                "root_cause_detail_tags": deduped_tags,
                "artifact_run_id": artifact_run_id,
                "artifact_dir": snapshot["artifact_dir"],
                "decision_region_summary_json": artifact_pointers["decision_region_summary_json"],
                "certificate_witness_json": artifact_pointers["certificate_witness_json"],
                "certified_set_summary_json": artifact_pointers["certified_set_summary_json"],
                "decision_package_json": artifact_pointers["decision_package_json"],
                "voi_stop_certificate_json": artifact_pointers["voi_stop_certificate_json"],
                "results_json": artifact_pointers["results_json"],
                "artifact_pointers": artifact_pointers,
            }
        )
    normalized_kind_counts = {
        kind: int(kind_counts.get(kind, 0))
        for kind in (*FAILURE_ATLAS_REQUIRED_KINDS, *FAILURE_ATLAS_OPTIONAL_KINDS)
    }
    normalized_counts_by_support_status = dict(
        sorted(Counter(str(row.get("support_status") or "unknown") for row in atlas_rows).items())
    )
    normalized_coverage_class_counts = {
        kind: int(coverage_class_counts.get(kind, 0))
        for kind in all_coverage_classes
    }
    abstention_class_examples = {
        key: value
        for key, value in sorted(abstention_examples_by_class.items())
    }
    abstention_class_documentation = [
        {
            "abstention_class": key,
            "available_count": int(abstention_class_counts.get(key, 0)),
            "documented_count": len(abstention_class_examples.get(key, [])),
            "documentation_target": FAILURE_ATLAS_ABSTENTION_EXAMPLE_TARGET,
            "documentation_complete": len(abstention_class_examples.get(key, []))
            == min(FAILURE_ATLAS_ABSTENTION_EXAMPLE_TARGET, int(abstention_class_counts.get(key, 0))),
        }
        for key in sorted(abstention_class_counts)
    ]
    coverage_inclusion_status = {
        coverage_class: {
            "detected_count": int(normalized_coverage_class_counts.get(coverage_class, 0)),
            "included_count": int(normalized_coverage_class_counts.get(coverage_class, 0)),
            "status": "included_complete",
        }
        for coverage_class in all_coverage_classes
    }
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "lane_id": FAILURE_ATLAS_LANE_ID,
        "lane_label": FAILURE_ATLAS_LANE_LABEL,
        "lane_type": "checked_bundle_metadata_lane",
        "lane_scope": "latest_full_suite_focused_failure_exemplars",
        "required_exemplar_kinds": list(FAILURE_ATLAS_REQUIRED_KINDS),
        "optional_exemplar_kinds": list(FAILURE_ATLAS_OPTIONAL_KINDS),
        "created_at": _now(),
        "suite_run_id": suite_run_id,
        "row_count": len(atlas_rows),
        "counts_by_kind": normalized_kind_counts,
        "coverage_class_counts": normalized_coverage_class_counts,
        "coverage_inclusion_status": coverage_inclusion_status,
        "counts_by_lane_role": dict(sorted(Counter(str(row["lane_role"]) for row in atlas_rows).items())),
        "counts_by_variant_id": dict(sorted(Counter(str(row["variant_id"]) for row in atlas_rows).items())),
        "counts_by_support_status": normalized_counts_by_support_status,
        "root_cause_family_counts": {
            family: int(root_cause_family_counts.get(family, 0))
            for family in FAILURE_ATLAS_ROOT_CAUSE_FAMILIES
        },
        "abstention_class_counts": dict(sorted(abstention_class_counts.items())),
        "abstention_class_example_target": FAILURE_ATLAS_ABSTENTION_EXAMPLE_TARGET,
        "abstention_class_examples": abstention_class_examples,
        "abstention_class_documentation": abstention_class_documentation,
        "certified_set_violation_case_count": int(normalized_coverage_class_counts.get("certified_set_violation", 0)),
        "top_root_causes": [
            {"tag": tag, "count": count}
            for tag, count in root_cause_counts.most_common(20)
        ],
        "rows": atlas_rows,
    }


def _render_failure_atlas_markdown(payload: Mapping[str, Any]) -> str:
    required_kinds = payload.get("required_exemplar_kinds")
    if not isinstance(required_kinds, Sequence) or isinstance(required_kinds, (str, bytes)):
        required_kinds = list(FAILURE_ATLAS_REQUIRED_KINDS)
    lines = [
        "# Failure Atlas",
        "",
        f"- Lane ID: `{payload.get('lane_id') or FAILURE_ATLAS_LANE_ID}`",
        f"- Lane Label: `{payload.get('lane_label') or FAILURE_ATLAS_LANE_LABEL}`",
        f"- Lane Type: `{payload.get('lane_type') or 'checked_bundle_metadata_lane'}`",
        f"- Lane Scope: `{payload.get('lane_scope') or 'latest_full_suite_focused_failure_exemplars'}`",
        f"- Suite Run ID: `{payload.get('suite_run_id')}`",
        f"- Created At: `{payload.get('created_at')}`",
        f"- Row Count: `{payload.get('row_count')}`",
        f"- Required exemplar kinds: `{', '.join(str(kind) for kind in required_kinds)}`",
        "",
        "## Counts By Kind",
        "",
    ]
    counts_by_kind = payload.get("counts_by_kind")
    if isinstance(counts_by_kind, Mapping) and counts_by_kind:
        for key, value in counts_by_kind.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No failure-atlas rows recorded.")
    lines.extend(["", "## Coverage Class Counts", ""])
    coverage_class_counts = payload.get("coverage_class_counts")
    if isinstance(coverage_class_counts, Mapping) and coverage_class_counts:
        for key, value in coverage_class_counts.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No coverage-class counts recorded.")
    lines.extend(["", "## Top Root Causes", ""])
    top_root_causes = payload.get("top_root_causes")
    if isinstance(top_root_causes, list) and top_root_causes:
        for item in top_root_causes:
            lines.append(f"- `{item.get('tag')}`: `{item.get('count')}`")
    else:
        lines.append("- No root-cause tags recorded.")
    lines.extend(["", "## Root-Cause Families", ""])
    root_cause_family_counts = payload.get("root_cause_family_counts")
    if isinstance(root_cause_family_counts, Mapping) and root_cause_family_counts:
        for family, count in root_cause_family_counts.items():
            lines.append(f"- `{family}`: `{count}`")
    else:
        lines.append("- No root-cause family counts recorded.")
    lines.extend(["", "## Abstention Classes", ""])
    abstention_documentation = payload.get("abstention_class_documentation")
    abstention_examples = payload.get("abstention_class_examples")
    if isinstance(abstention_documentation, list) and abstention_documentation:
        for item in abstention_documentation:
            abstention_class = str(item.get("abstention_class") or "unknown")
            lines.extend(
                [
                    f"### `{abstention_class}`",
                    "",
                    f"- Available rows: `{item.get('available_count')}`",
                    f"- Documented examples: `{item.get('documented_count')}`",
                    f"- Target examples: `{item.get('documentation_target')}`",
                    "",
                ]
            )
            example_rows = abstention_examples.get(abstention_class) if isinstance(abstention_examples, Mapping) else []
            if isinstance(example_rows, list) and example_rows:
                for row in example_rows:
                    lines.append(
                        f"- `{row.get('row_id')}` / cohort=`{row.get('cohort')}` / "
                        f"support=`{row.get('support_status')}` / challenger=`{row.get('active_challenger_id')}` / "
                        f"family=`{row.get('dominant_fragility_family')}` / "
                        f"stop=`{row.get('controller_stop_reason')}` / "
                        f"root=`{row.get('root_cause_tag')}` / detail=`{row.get('root_cause_detail_tag')}` / "
                        f"artifact=`{((row.get('artifact_pointers') or {}).get('decision_region_summary_json'))}`"
                    )
            else:
                lines.append("- No representative rows recorded.")
            lines.append("")
    else:
        lines.append("- No abstention classes recorded.")
    lines.extend(["", "## Certified-Set Violations", ""])
    lines.append(
        f"- Certified-set violation rows: `{payload.get('certified_set_violation_case_count', 0)}`"
    )
    lines.extend(["", "## Example Rows", ""])
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        for row in rows[:25]:
            lines.append(
                f"- `{row.get('row_id')}` / `{row.get('atlas_kind')}` / cohort=`{row.get('cohort')}`: "
                f"support=`{row.get('support_status')}` "
                f"challenger=`{row.get('active_challenger_id')}` "
                f"family=`{row.get('dominant_fragility_family')}` "
                f"stop=`{row.get('controller_stop_reason')}` "
                f"root=`{row.get('root_cause_tag')}` "
                f"detail=`{row.get('root_cause_detail_tag')}` "
                f"artifact=`{((row.get('artifact_pointers') or {}).get('decision_region_summary_json'))}`"
            )
    else:
        lines.append("- No example rows recorded.")
    return "\n".join(lines) + "\n"


def _failure_atlas_lane_metadata(
    *,
    suite_run_id: str,
    payload: Mapping[str, Any],
    failure_atlas_json_path: str,
    failure_atlas_md_path: str,
    results_path: str,
    index_json_path: str,
) -> dict[str, Any]:
    counts_by_kind = payload.get("counts_by_kind")
    if not isinstance(counts_by_kind, Mapping):
        counts_by_kind = {}
    coverage_class_counts = payload.get("coverage_class_counts")
    if not isinstance(coverage_class_counts, Mapping):
        coverage_class_counts = {}
    counts_by_lane_role = payload.get("counts_by_lane_role")
    if not isinstance(counts_by_lane_role, Mapping):
        counts_by_lane_role = {}
    counts_by_support_status = payload.get("counts_by_support_status")
    if not isinstance(counts_by_support_status, Mapping):
        counts_by_support_status = {}
    root_cause_family_counts = payload.get("root_cause_family_counts")
    if not isinstance(root_cause_family_counts, Mapping):
        root_cause_family_counts = {}
    required_kind_counts = {
        kind: int(coverage_class_counts.get(kind) or 0)
        for kind in FAILURE_ATLAS_REQUIRED_KINDS
    }
    required_kind_presence = {
        kind: count > 0
        for kind, count in required_kind_counts.items()
    }
    included_atlas_kinds = sorted(
        str(kind)
        for kind, count in counts_by_kind.items()
        if str(kind).strip() and int(count or 0) > 0
    )
    source_lane_roles = sorted(
        str(role)
        for role, count in counts_by_lane_role.items()
        if str(role).strip() and int(count or 0) > 0
    )
    return {
        "schema_version": "failure_atlas_lane_metadata.v1",
        "lane_id": FAILURE_ATLAS_LANE_ID,
        "lane_label": FAILURE_ATLAS_LANE_LABEL,
        "lane_type": "checked_bundle_metadata_lane",
        "lane_scope": "latest_full_suite_focused_failure_exemplars",
        "lane_status": "present_complete" if all(required_kind_presence.values()) else "present_partial",
        "publication_safe_status": "present_generated_lane_surface",
        "lane_purpose": (
            "Collect focused-lane failure exemplars for wrong-singleton, support-downgrade, "
            "and abstention review, while preserving per-row artifact links and root-cause tags."
        ),
        "required_exemplar_kinds": list(FAILURE_ATLAS_REQUIRED_KINDS),
        "optional_exemplar_kinds": list(FAILURE_ATLAS_OPTIONAL_KINDS),
        "required_kind_counts": required_kind_counts,
        "required_kind_presence": required_kind_presence,
        "included_atlas_kinds": included_atlas_kinds,
        "source_lane_roles": source_lane_roles,
        "row_count": int(payload.get("row_count") or 0),
        "counts_by_kind": dict(counts_by_kind),
        "coverage_class_counts": {
            kind: int(coverage_class_counts.get(kind) or 0)
            for kind in (*FAILURE_ATLAS_REQUIRED_KINDS, *FAILURE_ATLAS_OPTIONAL_KINDS)
        },
        "counts_by_lane_role": dict(counts_by_lane_role),
        "counts_by_support_status": dict(counts_by_support_status),
        "root_cause_family_counts": {
            family: int(root_cause_family_counts.get(family) or 0)
            for family in FAILURE_ATLAS_ROOT_CAUSE_FAMILIES
        },
        "certified_set_violation_case_count": int(coverage_class_counts.get("certified_set_violation") or 0),
        "abstention_class_counts": dict(payload.get("abstention_class_counts") or {}),
        "abstention_class_example_target": int(payload.get("abstention_class_example_target") or 0),
        "abstention_class_documentation": list(payload.get("abstention_class_documentation") or []),
        "artifact_paths": {
            "lane_metadata": str(artifact_dir_for_run(suite_run_id) / "failure_atlas_lane_metadata.json"),
            "lane_audit": str(failure_atlas_json_path),
            "lane_report": str(failure_atlas_md_path),
            "suite_results": str(results_path),
            "suite_index": str(index_json_path),
        },
        "inclusion_rules": {
            "wrong_singleton": (
                "Include rows whose certified-set witness carries singleton_not_justified_reasons."
            ),
            "support_downgrade": (
                "Include rows whose support_flag is false or whose support_status is not supported."
            ),
            "abstention": (
                "Include rows whose terminal_type is abstained or abstention."
            ),
            "certified_set_violation": (
                "Include multi-member certified-set rows whose safety or support prerequisites fail."
            ),
            "route_failure": "Also include residual route-failure rows with a recorded failure_reason.",
        },
        "reviewer_surfaces": {
            "artifact_index_surface_id": "lane.full_suite.failure_atlas",
            "evaluation_card_reference": "docs/evaluation_card.md#additive-checked-lane-metadata",
        },
        "notes": [
            "This is a named full-suite checked-lane surface, not a new run_thesis_evaluation.py suite role.",
            "Coverage-class counts are multi-label and therefore stricter than the primary atlas_kind counts.",
            "A concrete checked bundle still needs to be cited separately before reviewer docs should call it a checked headline artifact.",
        ],
    }


def repair_failure_atlas_suite_root(
    *,
    suite_run_id: str,
    out_dir: str | Path,
    hot_payload_path: str | Path | None = None,
) -> dict[str, Any]:
    old_out_dir = settings.out_dir
    settings.out_dir = Path(out_dir)
    try:
        suite_artifact_dir = artifact_dir_for_run(suite_run_id)
        if not suite_artifact_dir.exists():
            raise FileNotFoundError(f"suite_artifact_dir_missing:{suite_artifact_dir}")

        index_payload = _load_json_dict(suite_artifact_dir / "index.json") or {}
        results_payload = _load_json_dict(suite_artifact_dir / "results.json") or {}
        metadata_payload = _load_json_dict(suite_artifact_dir / "metadata.json") or {}
        lane_runs = results_payload.get("lane_runs")
        if not isinstance(lane_runs, Mapping):
            lane_runs = index_payload.get("lane_runs")
        if not isinstance(lane_runs, Mapping):
            raise RuntimeError("failure_atlas_suite_root_repair_missing_lane_runs")
        hot_payload = _load_suite_root_hot_payload(
            lane_runs=lane_runs,
            hot_payload_path=hot_payload_path,
        )

        focused_publication_rows: list[dict[str, Any]] = []
        repaired_roles: list[str] = []
        for role in sorted(FOCUSED_ROLES):
            lane_record = lane_runs.get(role)
            if not isinstance(lane_record, Mapping):
                continue
            artifact_paths = lane_record.get("artifact_paths")
            lane_results_path = None
            if isinstance(artifact_paths, Mapping):
                lane_results_path = _normalize_existing_path(
                    artifact_paths.get("results_json") or artifact_paths.get("thesis_results_json")
                )
            if lane_results_path is None:
                run_id = str(lane_record.get("run_id") or "").strip()
                if run_id:
                    candidate_results_path = artifact_dir_for_run(run_id) / "results.json"
                    candidate_thesis_results_path = artifact_dir_for_run(run_id) / "thesis_results.json"
                    if candidate_results_path.exists():
                        lane_results_path = candidate_results_path
                    elif candidate_thesis_results_path.exists():
                        lane_results_path = candidate_thesis_results_path
            lane_payload = _read_json_payload(str(lane_results_path) if lane_results_path is not None else None)
            rows = lane_payload.get("rows")
            if not isinstance(rows, Sequence):
                continue
            focused_publication_rows.extend(
                _annotated_rows(
                    rows,
                    role=role,
                    lane_run_id=str(lane_payload.get("run_id") or lane_record.get("run_id") or ""),
                    corpus_key=str(lane_record.get("corpus_key") or _lane_corpus_key(role)),
                )
            )
            repaired_roles.append(role)
        if not focused_publication_rows:
            raise RuntimeError("failure_atlas_suite_root_repair_no_focused_rows")

        failure_atlas_payload = _build_failure_atlas(
            suite_run_id=suite_run_id,
            rows=focused_publication_rows,
        )
        failure_atlas_rows = list(failure_atlas_payload.get("rows") or [])
        failure_atlas_json_path = write_json_artifact(
            suite_run_id,
            "failure_atlas.json",
            failure_atlas_payload,
        )
        failure_atlas_md_path = write_text_artifact(
            suite_run_id,
            "failure_atlas.md",
            _render_failure_atlas_markdown(failure_atlas_payload),
        )
        results_artifact_path = str(suite_artifact_dir / "results.json")
        index_json_artifact_path = str(suite_artifact_dir / "index.json")
        failure_atlas_lane_metadata_payload = _failure_atlas_lane_metadata(
            suite_run_id=suite_run_id,
            payload=failure_atlas_payload,
            failure_atlas_json_path=str(failure_atlas_json_path),
            failure_atlas_md_path=str(failure_atlas_md_path),
            results_path=results_artifact_path,
            index_json_path=index_json_artifact_path,
        )
        failure_atlas_lane_metadata_path = write_json_artifact(
            suite_run_id,
            "failure_atlas_lane_metadata.json",
            failure_atlas_lane_metadata_payload,
        )

        def _rows_from_summary_artifact(filename: str) -> list[dict[str, Any]]:
            payload = _load_json_dict(suite_artifact_dir / filename) or {}
            rows = payload.get("rows")
            if not isinstance(rows, Sequence):
                return []
            return [dict(row) for row in rows if isinstance(row, Mapping)]

        lane_publishability_rows = _rows_from_summary_artifact("lane_publishability_summary.json")
        baseline_audit_rows = _rows_from_summary_artifact("universal_baseline_audit.json")
        sample_size_rows = _rows_from_summary_artifact("sample_size_gate_summary.json")
        headline_seed_claim_rows = _rows_from_summary_artifact("headline_seed_claims_summary.json")
        repaired_lane_runs = _repair_suite_root_lane_runs(
            suite_run_id=suite_run_id,
            lane_runs=lane_runs,
            sample_size_rows=sample_size_rows,
            hot_payload=hot_payload,
        )
        verdict = _publishability_verdict_payload(
            lane_publishability_rows=lane_publishability_rows,
            baseline_audit_rows=baseline_audit_rows,
            failure_atlas_rows=failure_atlas_rows,
            sample_size_rows=sample_size_rows,
            headline_seed_claim_rows=headline_seed_claim_rows,
            hot_payload=hot_payload,
            suite_artifact_dir=suite_artifact_dir,
        )
        verdict_json_path = write_json_artifact(
            suite_run_id,
            "publishability_verdict.json",
            verdict,
        )
        verdict_md_path = write_text_artifact(
            suite_run_id,
            "publishability_assessment.md",
            _publishability_markdown(
                suite_run_id=suite_run_id,
                verdict=verdict,
                lane_publishability_rows=lane_publishability_rows,
                baseline_audit_rows=baseline_audit_rows,
                sample_size_rows=sample_size_rows,
                headline_seed_claim_rows=headline_seed_claim_rows,
            ),
        )

        index_payload = dict(index_payload)
        index_payload["lane_runs"] = repaired_lane_runs
        index_payload["failure_atlas_json"] = str(failure_atlas_json_path)
        index_payload["failure_atlas_md"] = str(failure_atlas_md_path)
        index_payload["failure_atlas_lane_metadata_json"] = str(failure_atlas_lane_metadata_path)
        index_payload["publishability_verdict_json"] = str(verdict_json_path)
        index_payload["publishability_assessment_md"] = str(verdict_md_path)
        index_json_path = write_json_artifact(
            suite_run_id,
            "index.json",
            index_payload,
        )

        results_payload = dict(results_payload)
        results_payload["lane_runs"] = repaired_lane_runs
        results_payload["lane_publishability_rows"] = lane_publishability_rows
        results_payload["baseline_audit_rows"] = baseline_audit_rows
        results_payload["sample_size_rows"] = sample_size_rows
        results_payload["headline_seed_claim_rows"] = headline_seed_claim_rows
        results_payload["failure_atlas_rows"] = failure_atlas_rows
        results_payload["failure_atlas"] = failure_atlas_payload
        results_payload["failure_atlas_lane_metadata"] = failure_atlas_lane_metadata_payload
        results_payload["failure_atlas_lane_metadata_json"] = str(failure_atlas_lane_metadata_path)
        results_payload["publishability_verdict"] = verdict
        results_path = write_json_artifact(
            suite_run_id,
            "results.json",
            results_payload,
        )

        if metadata_payload:
            metadata_payload = dict(metadata_payload)
            metadata_payload["lane_runs"] = repaired_lane_runs
            metadata_payload["lane_publishability_row_count"] = len(lane_publishability_rows)
            metadata_payload["baseline_audit_row_count"] = len(baseline_audit_rows)
            metadata_payload["sample_size_row_count"] = len(sample_size_rows)
            metadata_payload["headline_seed_claim_row_count"] = len(headline_seed_claim_rows)
            metadata_payload["failure_atlas_row_count"] = failure_atlas_payload.get("row_count")
            metadata_payload["failure_atlas_lane_id"] = FAILURE_ATLAS_LANE_ID
            metadata_payload["failure_atlas_lane_status"] = failure_atlas_lane_metadata_payload.get("lane_status")
            metadata_payload["failure_atlas_lane_metadata_json"] = str(failure_atlas_lane_metadata_path)
            metadata_path = write_json_artifact(
                suite_run_id,
                "metadata.json",
                metadata_payload,
            )
        else:
            metadata_path = suite_artifact_dir / "metadata.json"

        all_suite_roles = [*DIRECT_SUITE_ROLES, "hot_rerun"]
        pending_roles = [
            role
            for role in all_suite_roles
            if str(repaired_lane_runs.get(role, {}).get("status")) != "completed"
        ]
        suite_progress_path = _write_suite_progress(
            suite_run_id=suite_run_id,
            lane_runs=repaired_lane_runs,
            pending_roles=pending_roles,
        )

        return {
            "suite_run_id": suite_run_id,
            "repaired_roles": repaired_roles,
            "focused_row_count": len(focused_publication_rows),
            "failure_atlas_json": str(failure_atlas_json_path),
            "failure_atlas_md": str(failure_atlas_md_path),
            "failure_atlas_lane_metadata_json": str(failure_atlas_lane_metadata_path),
            "publishability_verdict_json": str(verdict_json_path),
            "publishability_assessment_md": str(verdict_md_path),
            "index_json": str(index_json_path),
            "results_json": str(results_path),
            "metadata_json": str(metadata_path),
            "suite_progress_json": str(suite_progress_path),
        }
    finally:
        settings.out_dir = old_out_dir
