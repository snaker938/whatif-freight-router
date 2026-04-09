from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.compose_thesis_sharded_report as compose_sharded
import scripts.run_thesis_lane as thesis_lane


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _default_canonical_corpus() -> Path:
    candidates = (
        ROOT / "data" / "eval" / "uk_od_corpus_thesis_broad_expanded_1200.csv",
        ROOT / "data" / "eval" / "uk_od_corpus_thesis_broad_expanded_120.csv",
        ROOT / "data" / "eval" / "uk_od_corpus_thesis_broad.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


DEFAULT_LANE_SPECS: tuple[dict[str, str], ...] = (
    {
        "lane_id": "hard_mixed_24",
        "corpus_csv": str(ROOT / "data" / "eval" / "uk_od_corpus_hard_mixed_24.csv"),
        "evaluation_suite_role": "hard_mixed_story",
        "label": "Hard mixed 24",
    },
    {
        "lane_id": "longcorr_hard_32",
        "corpus_csv": str(ROOT / "data" / "eval" / "uk_od_corpus_longcorr_hard_32.csv"),
        "evaluation_suite_role": "longcorr_story",
        "label": "Long-corridor hard 32",
    },
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a serial harder-story thesis suite over the checked-in hard corpora "
            "and compose the resulting lane outputs into a single quoteable bundle."
        )
    )
    default_suite_dir = ROOT / "out" / "thesis_hard_story_suite"
    parser.add_argument("--suite-id", default="thesis_hard_story_suite")
    parser.add_argument("--suite-dir", type=Path, default=default_suite_dir)
    parser.add_argument("--out-dir", default=str(ROOT / "out"))
    parser.add_argument("--canonical-corpus", default=str(_default_canonical_corpus()))
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--manage-local-backend", action="store_true")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--backend-memory-limit-mb", type=int, default=0)
    parser.add_argument("--powershell-exe", default="powershell.exe")
    parser.add_argument("--backend-start-script", type=Path, default=ROOT / "scripts" / "start_backend_logged.ps1")
    parser.add_argument("--backend-stop-script", type=Path, default=ROOT / "scripts" / "stop_backend_logged.ps1")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--pytest-args", nargs="*", default=[])
    parser.add_argument("--evaluation-args", nargs=argparse.REMAINDER, default=[])
    return parser


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _lane_report_paths(*, suite_dir: Path, lane_id: str) -> tuple[Path, Path]:
    lane_dir = suite_dir / lane_id
    return lane_dir / "report.md", lane_dir / "report.json"


def _lane_argv(args: argparse.Namespace, lane: dict[str, str], *, report_md: Path, report_json: Path) -> list[str]:
    argv = [
        "--report-md",
        str(report_md),
        "--report-json",
        str(report_json),
        "--corpus-csv",
        str(lane["corpus_csv"]),
        "--backend-url",
        str(args.backend_url),
        "--out-dir",
        str(args.out_dir),
    ]
    if args.manage_local_backend:
        argv.extend(
            [
                "--manage-local-backend",
                "--backend-port",
                str(int(args.backend_port)),
                "--backend-memory-limit-mb",
                str(int(args.backend_memory_limit_mb)),
                "--powershell-exe",
                str(args.powershell_exe),
                "--backend-start-script",
                str(args.backend_start_script),
                "--backend-stop-script",
                str(args.backend_stop_script),
            ]
        )
    if args.skip_evaluation:
        argv.append("--skip-evaluation")
    if args.pytest_args:
        argv.extend(["--pytest-args", *list(args.pytest_args)])
    evaluation_args = [
        "--evaluation-suite-role",
        str(lane["evaluation_suite_role"]),
        *list(args.evaluation_args or []),
    ]
    if evaluation_args:
        argv.extend(["--evaluation-args", *evaluation_args])
    return argv


def _load_lane_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _lane_summary_snapshot(report_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    evaluation = report_payload.get("evaluation") if isinstance(report_payload, dict) else None
    summary_rows = evaluation.get("summary_rows") if isinstance(evaluation, dict) else None
    if not isinstance(summary_rows, list):
        return []
    return [row for row in summary_rows if isinstance(row, dict)]


def _successful_results_csv(report_payload: dict[str, Any] | None) -> str | None:
    evaluation = report_payload.get("evaluation") if isinstance(report_payload, dict) else None
    raw = str(evaluation.get("results_csv") or "").strip() if isinstance(evaluation, dict) else ""
    return raw or None


def _markdown_report(
    *,
    suite_id: str,
    lane_results: list[dict[str, Any]],
    composed_payload: dict[str, Any] | None,
) -> str:
    lines = [
        "# Hard Story Thesis Suite",
        "",
        f"- Suite id: `{suite_id}`",
        f"- Generated at: `{_now()}`",
        f"- Lane count: `{len(lane_results)}`",
        f"- Successful lane count: `{sum(1 for lane in lane_results if lane.get('success') is True)}`",
        f"- Failed lane count: `{sum(1 for lane in lane_results if lane.get('success') is not True)}`",
        "",
        "## Lane Results",
    ]
    for lane in lane_results:
        lines.append(
            f"- `{lane.get('lane_id')}` / `{lane.get('evaluation_suite_role')}`: "
            f"success={lane.get('success')}, exit_code={lane.get('exit_code')}, "
            f"corpus_csv=`{lane.get('corpus_csv')}`, "
            f"results_csv=`{lane.get('results_csv') or 'none'}`, "
            f"summary_rows={lane.get('summary_row_count')}"
        )
    lines.append("")
    lines.append("## Composed Bundle")
    if composed_payload is None:
        lines.append("- No composed bundle was written.")
    else:
        lines.append(f"- results_csv=`{composed_payload.get('results_csv')}`")
        lines.append(f"- summary_csv=`{composed_payload.get('summary_csv')}`")
        lines.append(f"- thesis_report=`{composed_payload.get('thesis_report')}`")
        lines.append(f"- evaluation_manifest=`{composed_payload.get('evaluation_manifest')}`")
    return "\n".join(lines) + "\n"


def run_hard_story_suite(args: argparse.Namespace) -> dict[str, Any]:
    suite_dir = Path(args.suite_dir).expanduser().resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    lane_results: list[dict[str, Any]] = []
    successful_results_csvs: list[str] = []

    for lane in DEFAULT_LANE_SPECS:
        report_md, report_json = _lane_report_paths(suite_dir=suite_dir, lane_id=lane["lane_id"])
        argv = _lane_argv(args, lane, report_md=report_md, report_json=report_json)
        exit_code = int(thesis_lane.main(argv))
        report_payload = _load_lane_report(report_json)
        summary_rows = _lane_summary_snapshot(report_payload)
        results_csv = _successful_results_csv(report_payload)
        lane_result = {
            "lane_id": str(lane["lane_id"]),
            "label": str(lane["label"]),
            "corpus_csv": str(lane["corpus_csv"]),
            "evaluation_suite_role": str(lane["evaluation_suite_role"]),
            "report_md": str(report_md),
            "report_json": str(report_json),
            "exit_code": exit_code,
            "success": exit_code == 0 and results_csv is not None,
            "results_csv": results_csv,
            "summary_row_count": len(summary_rows),
            "summary_rows": summary_rows,
        }
        lane_results.append(lane_result)
        if results_csv is not None:
            successful_results_csvs.append(results_csv)

    composed_payload: dict[str, Any] | None = None
    if successful_results_csvs:
        compose_args = argparse.Namespace(
            run_id=f"{args.suite_id}_composed",
            out_dir=str(args.out_dir),
            canonical_corpus=str(args.canonical_corpus),
            results_csv=successful_results_csvs,
            ors_baseline_policy="repo_local",
            ors_snapshot_mode="off",
        )
        composed_payload = compose_sharded.compose_sharded_report(compose_args)

    payload = {
        "suite_id": str(args.suite_id),
        "generated_at_utc": _now(),
        "suite_dir": str(suite_dir),
        "canonical_corpus": str(args.canonical_corpus),
        "lane_results": lane_results,
        "successful_lane_count": sum(1 for lane in lane_results if lane["success"] is True),
        "failed_lane_count": sum(1 for lane in lane_results if lane["success"] is not True),
        "composed_payload": composed_payload,
    }
    report_md = suite_dir / "hard_story_suite_report.md"
    report_json = suite_dir / "hard_story_suite_report.json"
    _write_text(
        report_md,
        _markdown_report(
            suite_id=str(args.suite_id),
            lane_results=lane_results,
            composed_payload=composed_payload,
        ),
    )
    _write_json(report_json, payload)
    payload["report_md"] = str(report_md)
    payload["report_json"] = str(report_json)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_hard_story_suite(args)
    return 0 if int(payload["failed_lane_count"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
