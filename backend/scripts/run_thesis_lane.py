from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_thesis_evaluation as thesis_eval


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the thesis-only test and evaluation lane.")
    default_corpus_csv = ROOT / "data" / "eval" / "uk_od_corpus_thesis_broad.csv"
    parser.add_argument("--report-md", type=Path, default=ROOT / "out" / "thesis_lane_report.md")
    parser.add_argument("--report-json", type=Path, default=ROOT / "out" / "thesis_lane_report.json")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--manage-local-backend", action="store_true")
    parser.add_argument("--keep-backend-running", action="store_true")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--powershell-exe", default="powershell.exe")
    parser.add_argument("--backend-start-script", type=Path, default=ROOT / "scripts" / "start_backend_logged.ps1")
    parser.add_argument("--backend-stop-script", type=Path, default=ROOT / "scripts" / "stop_backend_logged.ps1")
    parser.add_argument("--out-dir", default=str(ROOT / "out"))
    parser.add_argument("--corpus-csv", default=str(default_corpus_csv))
    parser.add_argument("--corpus-json", default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--pytest-args", nargs="*", default=[])
    parser.add_argument("--evaluation-args", nargs=argparse.REMAINDER, default=[])
    return parser


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _validated_artifact_paths(payload: dict[str, Any]) -> dict[str, Path]:
    required_path_keys = (
        "results_csv",
        "summary_csv",
        "thesis_report",
        "methods_appendix",
        "evaluation_manifest",
        "manifest_path",
    )
    validated: dict[str, Path] = {}
    for key in required_path_keys:
        raw = str(payload.get(key) or "").strip()
        if not raw:
            raise RuntimeError(f"thesis_lane_missing_artifact_path:{key}")
        path = Path(raw)
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"thesis_lane_missing_artifact_file:{key}")
        validated[key] = path
    return validated


def _run_backend_script(
    *,
    powershell_exe: str,
    script_path: Path,
    port: int,
) -> dict[str, Any]:
    if not script_path.exists():
        raise RuntimeError(f"backend_lifecycle_script_missing:{script_path}")
    command = [
        str(powershell_exe),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-Port",
        str(int(port)),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    result = {
        "command": command,
        "script": str(script_path),
        "port": int(port),
        "returncode": int(completed.returncode),
        "stdout": stdout,
        "stderr": stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"backend_lifecycle_failed:{script_path.name}:returncode={completed.returncode}:"
            f"stdout={stdout or '<empty>'}:stderr={stderr or '<empty>'}"
        )
    return result


def _variant_summary_lines(summary_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in summary_rows:
        lines.append(
            f"- {row.get('variant_id')} / {row.get('pipeline_mode')}: "
            f"rows={row.get('row_count')}, successes={row.get('success_count')}, failures={row.get('failure_count')}, "
            f"weighted_win_osrm={row.get('weighted_win_rate_osrm')} (n={row.get('weighted_denominator_osrm')}), "
            f"weighted_win_ors={row.get('weighted_win_rate_ors')} (n={row.get('weighted_denominator_ors')}), "
            f"mean_certificate={row.get('mean_certificate')} (n={row.get('mean_certificate_denominator')}), "
            f"nontrivial_frontier_rate={row.get('nontrivial_frontier_rate')}, "
            f"voi_controller_engagement_rate={row.get('voi_controller_engagement_rate')}, "
            f"selector_certificate_disagreement_rate={row.get('selector_certificate_disagreement_rate')}"
        )
    return lines


def _successful_variant_summary_lines(summary_rows: list[dict[str, Any]]) -> list[str]:
    return _variant_summary_lines([row for row in summary_rows if int(row.get("success_count") or 0) > 0])


def _failed_variant_summary_lines(summary_rows: list[dict[str, Any]], *, rows: list[dict[str, Any]]) -> list[str]:
    failure_map: dict[str, dict[str, int]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id") or "").strip()
        reason = str(row.get("failure_reason") or "").strip()
        if not variant_id or not reason:
            continue
        bucket = failure_map.setdefault(variant_id, {})
        bucket[reason] = int(bucket.get(reason, 0)) + 1
    lines: list[str] = []
    for row in summary_rows:
        if int(row.get("failure_count") or 0) <= 0:
            continue
        reason_counts = failure_map.get(str(row.get("variant_id") or ""), {})
        reason_text = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
        lines.append(
            f"- {row.get('variant_id')} / {row.get('pipeline_mode')}: "
            f"failure_rows={row.get('failure_count')}/{row.get('row_count')}, "
            f"success_rows={row.get('success_count')}, "
            f"artifact_complete_rate={row.get('artifact_complete_rate')}, "
            f"route_evidence_ok_rate={row.get('route_evidence_ok_rate')}, "
            f"failure_reasons={reason_text or 'none'}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    pytest_args = [
        "-c",
        str(ROOT / "pyproject.toml"),
        str(ROOT / "tests"),
        "-m",
        "thesis or thesis_results or thesis_modules",
        "-q",
        *list(args.pytest_args or []),
    ]
    pytest_exit = int(pytest.main(pytest_args))
    evaluation_payload: dict[str, Any] | None = None
    evaluation_exit = 0
    backend_lifecycle: dict[str, Any] | None = None
    backend_started_here = False
    if not args.skip_evaluation and (args.corpus_csv or args.corpus_json):
        try:
            eval_args = thesis_eval._build_parser().parse_args(
                [
                    *(
                        ["--corpus-csv", str(args.corpus_csv)]
                        if args.corpus_csv
                        else ["--corpus-json", str(args.corpus_json)]
                    ),
                    "--backend-url",
                    str(args.backend_url),
                    "--out-dir",
                    str(args.out_dir),
                    *list(args.evaluation_args or []),
                ]
            )
            in_process_backend = bool(getattr(eval_args, "in_process_backend", False))
            if bool(args.manage_local_backend):
                if in_process_backend:
                    raise RuntimeError("thesis_lane_invalid_backend_mode:manage_local_backend_with_in_process_backend")
                stop_result = _run_backend_script(
                    powershell_exe=str(args.powershell_exe),
                    script_path=Path(args.backend_stop_script),
                    port=int(args.backend_port),
                )
                start_result = _run_backend_script(
                    powershell_exe=str(args.powershell_exe),
                    script_path=Path(args.backend_start_script),
                    port=int(args.backend_port),
                )
                backend_started_here = True
                backend_lifecycle = {
                    "managed": True,
                    "port": int(args.backend_port),
                    "stop_before_run": stop_result,
                    "start_before_run": start_result,
                }
            if in_process_backend:
                from app.main import app

                with thesis_eval.TestClient(app) as client:
                    evaluation_payload = thesis_eval.run_thesis_evaluation(eval_args, client=client)
            else:
                evaluation_payload = thesis_eval.run_thesis_evaluation(eval_args)
            validated_paths = _validated_artifact_paths(evaluation_payload)
            evaluation_payload["validated_artifacts"] = {
                key: str(path) for key, path in validated_paths.items()
            }
        except Exception as exc:  # pragma: no cover - defensive boundary
            evaluation_exit = 1
            evaluation_payload = {"error": type(exc).__name__, "message": str(exc)}
        finally:
            if backend_started_here and not bool(args.keep_backend_running):
                try:
                    stop_after_run = _run_backend_script(
                        powershell_exe=str(args.powershell_exe),
                        script_path=Path(args.backend_stop_script),
                        port=int(args.backend_port),
                    )
                    if backend_lifecycle is None:
                        backend_lifecycle = {
                            "managed": True,
                            "port": int(args.backend_port),
                        }
                    backend_lifecycle["stop_after_run"] = stop_after_run
                except Exception as exc:  # pragma: no cover - defensive boundary
                    evaluation_exit = 1
                    if evaluation_payload is None:
                        evaluation_payload = {"error": type(exc).__name__, "message": str(exc)}
                    if backend_lifecycle is None:
                        backend_lifecycle = {"managed": True, "port": int(args.backend_port)}
                    backend_lifecycle["stop_after_run_error"] = str(exc)
    summary = {
        "generated_at_utc": _now(),
        "pytest_exit_code": pytest_exit,
        "pytest_args": pytest_args,
        "evaluation_exit_code": evaluation_exit,
        "evaluation": evaluation_payload,
        "backend_lifecycle": backend_lifecycle,
    }
    report_lines = [
        "# Thesis Lane Report",
        "",
        f"- Generated at: `{summary['generated_at_utc']}`",
        f"- Thesis pytest exit code: `{pytest_exit}`",
        f"- Thesis pytest args: `{' '.join(pytest_args)}`",
        f"- Evaluation exit code: `{evaluation_exit}`",
    ]
    if isinstance(backend_lifecycle, dict) and backend_lifecycle.get("managed"):
        report_lines.extend(
            [
                f"- Managed local backend: `true`",
                f"- Managed backend port: `{backend_lifecycle.get('port')}`",
                f"- Start script: `{args.backend_start_script}`",
                f"- Stop script: `{args.backend_stop_script}`",
            ]
        )
        if backend_lifecycle.get("start_before_run"):
            report_lines.append(
                f"- Local backend start stdout: `{(backend_lifecycle['start_before_run'] or {}).get('stdout')}`"
            )
        if backend_lifecycle.get("stop_after_run_error"):
            report_lines.append(f"- Local backend stop error: `{backend_lifecycle.get('stop_after_run_error')}`")
    if isinstance(evaluation_payload, dict):
        if evaluation_payload.get("run_id"):
            report_lines.extend(
                [
                    f"- Evaluation run id: `{evaluation_payload['run_id']}`",
                    f"- Secondary baseline policy: `{evaluation_payload.get('ors_baseline_policy')}`",
                    f"- Secondary baseline snapshot mode: `{evaluation_payload.get('ors_snapshot_mode')}`",
                    f"- Successful rows: `{evaluation_payload.get('success_row_count')}`",
                    f"- Failure rows: `{evaluation_payload.get('failure_row_count')}`",
                    f"- Results CSV: `{evaluation_payload.get('results_csv')}`",
                    f"- Summary CSV: `{evaluation_payload.get('summary_csv')}`",
                    f"- Cohort summary CSV: `{evaluation_payload.get('summary_by_cohort_csv')}`",
                    f"- Thesis report: `{evaluation_payload.get('thesis_report')}`",
                    f"- Methods appendix: `{evaluation_payload.get('methods_appendix')}`",
                    f"- Evaluation manifest: `{evaluation_payload.get('evaluation_manifest')}`",
                    f"- Manifest path: `{evaluation_payload.get('manifest_path')}`",
                    f"- Validated output artifacts: `{(evaluation_payload.get('output_artifact_validation') or {}).get('validated_artifact_count')}`",
                    f"- Corpus source: `{args.corpus_csv or args.corpus_json}`",
                ]
            )
            if evaluation_payload.get("ors_snapshot_path"):
                report_lines.append(f"- Secondary baseline snapshot: `{evaluation_payload.get('ors_snapshot_path')}`")
            summary_rows = evaluation_payload.get("summary_rows")
            if isinstance(summary_rows, list) and summary_rows:
                report_lines.extend(["", "## Successful Variants", ""])
                success_lines = _successful_variant_summary_lines(summary_rows)
                report_lines.extend(success_lines or ["- none"])
                report_lines.extend(["", "## Failed Variants", ""])
                failed_lines = _failed_variant_summary_lines(
                    summary_rows,
                    rows=list(evaluation_payload.get("rows") or []),
                )
                report_lines.extend(failed_lines or ["- none"])
        elif evaluation_payload.get("error"):
            report_lines.append(f"- Evaluation error: `{evaluation_payload['error']}: {evaluation_payload.get('message', '')}`")
    _write(args.report_md, "\n".join(report_lines) + "\n")
    _write(args.report_json, json.dumps(summary, indent=2))
    return 0 if pytest_exit == 0 and evaluation_exit == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
