from __future__ import annotations

import json
import sys
import types
from subprocess import CompletedProcess
from pathlib import Path

import pytest

import scripts.run_thesis_lane as thesis_lane


pytestmark = pytest.mark.thesis_results


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def _seed_lane_artifacts(tmp_path: Path) -> dict[str, Path]:
    artifacts = {
        "results_csv": tmp_path / "results.csv",
        "summary_csv": tmp_path / "summary.csv",
        "summary_by_cohort_csv": tmp_path / "summary_by_cohort.csv",
        "summary_by_cohort_json": tmp_path / "summary_by_cohort.json",
        "cohort_composition_path": tmp_path / "cohort_composition.json",
        "thesis_report": tmp_path / "report.md",
        "methods_appendix": tmp_path / "methods.md",
        "evaluation_manifest": tmp_path / "evaluation_manifest.json",
        "manifest_path": tmp_path / "manifest.json",
    }
    default_contents = {
        "results_csv": "od_id,variant_id\nod-1,V0\n",
        "summary_csv": "variant_id,pipeline_mode,row_count,failure_count,success_count\nV0,legacy,1,0,1\n",
        "summary_by_cohort_csv": "variant_id,pipeline_mode,cohort_label,row_count\nV0,legacy,representative,1\n",
        "summary_by_cohort_json": '{"summary_rows":[]}\n',
        "cohort_composition_path": '{"total_row_count":1,"by_variant":{"V0":{"row_count":1}}}\n',
        "thesis_report": "# report\n",
        "methods_appendix": "# methods\n",
        "evaluation_manifest": "{}\n",
        "manifest_path": "{}\n",
    }
    for key, path in artifacts.items():
        _write(path, default_contents[key])
    return artifacts


def _evaluation_payload(
    artifacts: dict[str, Path],
    *,
    run_id: str = "run-123",
    output_artifact_validation: dict[str, object] | None = None,
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "ors_baseline_policy": "local_service",
        "ors_snapshot_mode": "off",
        "results_csv": str(artifacts["results_csv"]),
        "summary_csv": str(artifacts["summary_csv"]),
        "summary_by_cohort_csv": str(artifacts["summary_by_cohort_csv"]),
        "summary_by_cohort_json": str(artifacts["summary_by_cohort_json"]),
        "cohort_composition_path": str(artifacts["cohort_composition_path"]),
        "thesis_report": str(artifacts["thesis_report"]),
        "methods_appendix": str(artifacts["methods_appendix"]),
        "evaluation_manifest": str(artifacts["evaluation_manifest"]),
        "manifest_path": str(artifacts["manifest_path"]),
        "output_artifact_validation": output_artifact_validation or {"validated_artifact_count": 9},
        "summary_rows": [],
        "rows": [],
        "success_row_count": 0,
        "failure_row_count": 0,
    }
    payload.update(overrides)
    return payload


def test_thesis_lane_script_writes_report_and_summary(tmp_path: Path, monkeypatch) -> None:
    report_md = tmp_path / "thesis_lane_report.md"
    report_json = tmp_path / "thesis_lane_report.json"
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon,distance_bin,seed\n"
        "od-1,52.0,-1.5,51.5,-1.2,30-100 km,7\n",
        encoding="utf-8",
    )
    artifacts = _seed_lane_artifacts(tmp_path)

    monkeypatch.setattr(thesis_lane.pytest, "main", lambda args: 0)
    monkeypatch.setattr(
        thesis_lane.thesis_eval,
        "run_thesis_evaluation",
        lambda args: _evaluation_payload(
            artifacts,
            summary_rows=[
                {
                    "variant_id": "C",
                    "pipeline_mode": "voi",
                    "row_count": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "weighted_win_rate_osrm": 1.0,
                    "weighted_denominator_osrm": 1,
                    "weighted_win_rate_ors": 1.0,
                    "weighted_denominator_ors": 1,
                    "mean_certificate": 0.91,
                    "mean_certificate_denominator": 1,
                }
            ],
            rows=[{"variant_id": "C", "failure_reason": ""}],
            success_row_count=1,
            failure_row_count=0,
            evaluation_suite={
                "role": "broad_cold_proof",
                "family": "evaluation",
                "scope": "broad",
                "focus": "all",
                "source": "corpus_source_path",
            },
            cohort_scaffolding={
                "cohort_scaffolding_version": "thesis_cohort_scaffolding_v1",
                "cohort_labels": ["representative", "ambiguity", "hard_case", "controller_stress"],
                "derived_cohort_labels": ["hard_case", "controller_stress"],
            },
        ),
    )

    exit_code = thesis_lane.main(
        [
            "--report-md",
            str(report_md),
            "--report-json",
            str(report_json),
            "--corpus-csv",
            str(corpus_csv),
        ]
    )

    assert exit_code == 0
    assert report_md.exists()
    assert report_json.exists()
    report_text = report_md.read_text(encoding="utf-8")
    assert "local_service" in report_text
    assert "methods.md" in report_text
    assert "## Successful Variants" in report_text
    assert "## Failed Variants" in report_text
    assert "weighted_win_osrm=1.0 (n=1)" in report_text
    assert f"Corpus source: `{corpus_csv}`" in report_text
    assert f"Cohort summary JSON: `{artifacts['summary_by_cohort_json']}`" in report_text
    assert f"Cohort composition JSON: `{artifacts['cohort_composition_path']}`" in report_text
    assert "Evaluation suite role: `broad_cold_proof`" in report_text
    assert "Evaluation suite family: `evaluation`" in report_text
    assert "Cohort scaffolding version: `thesis_cohort_scaffolding_v1`" in report_text
    assert "Cohort labels: `representative, ambiguity, hard_case, controller_stress`" in report_text
    assert "Derived cohorts: `hard_case, controller_stress`" in report_text
    assert "- none" in report_text
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["pytest_exit_code"] == 0
    assert payload["evaluation"]["run_id"] == "run-123"
    assert payload["evaluation"]["summary_by_cohort_json"] == str(artifacts["summary_by_cohort_json"])
    assert payload["evaluation"]["cohort_composition_path"] == str(artifacts["cohort_composition_path"])
    assert payload["evaluation"]["evaluation_suite"]["family"] == "evaluation"
    assert payload["evaluation"]["cohort_scaffolding"]["derived_cohort_labels"] == [
        "hard_case",
        "controller_stress",
    ]
    assert payload["evaluation"]["validated_artifacts"]["summary_by_cohort_json"] == str(
        artifacts["summary_by_cohort_json"]
    )
    assert payload["evaluation"]["validated_artifacts"]["cohort_composition_path"] == str(
        artifacts["cohort_composition_path"]
    )


def test_thesis_lane_script_separates_successful_and_failed_variants(tmp_path: Path, monkeypatch) -> None:
    report_md = tmp_path / "thesis_lane_report.md"
    report_json = tmp_path / "thesis_lane_report.json"
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon,distance_bin,seed\n"
        "od-1,52.0,-1.5,51.5,-1.2,30-100 km,7\n",
        encoding="utf-8",
    )
    artifacts = _seed_lane_artifacts(tmp_path)

    monkeypatch.setattr(thesis_lane.pytest, "main", lambda args: 0)
    monkeypatch.setattr(
        thesis_lane.thesis_eval,
        "run_thesis_evaluation",
        lambda args: _evaluation_payload(
            artifacts,
            run_id="run-456",
            summary_rows=[
                {
                    "variant_id": "A",
                    "pipeline_mode": "dccs",
                    "row_count": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "artifact_complete_rate": 1.0,
                    "route_evidence_ok_rate": 1.0,
                    "weighted_win_rate_osrm": 1.0,
                    "weighted_denominator_osrm": 1,
                    "weighted_win_rate_ors": 0.0,
                    "weighted_denominator_ors": 1,
                    "mean_certificate": None,
                    "mean_certificate_denominator": 0,
                },
                {
                    "variant_id": "B",
                    "pipeline_mode": "dccs_refc",
                    "row_count": 1,
                    "success_count": 0,
                    "failure_count": 1,
                    "artifact_complete_rate": 0.0,
                    "route_evidence_ok_rate": 0.0,
                    "weighted_win_rate_osrm": 0.0,
                    "weighted_denominator_osrm": 1,
                    "weighted_win_rate_ors": 0.0,
                    "weighted_denominator_ors": 1,
                    "mean_certificate": None,
                    "mean_certificate_denominator": 0,
                },
            ],
            rows=[
                {"variant_id": "A", "failure_reason": ""},
                {"variant_id": "B", "failure_reason": "strict_artifact_missing"},
            ],
            success_row_count=1,
            failure_row_count=1,
        ),
    )

    exit_code = thesis_lane.main(
        [
            "--report-md",
            str(report_md),
            "--report-json",
            str(report_json),
            "--corpus-csv",
            str(corpus_csv),
        ]
    )

    assert exit_code == 0
    report_text = report_md.read_text(encoding="utf-8")
    assert "## Successful Variants" in report_text
    assert "A / dccs" in report_text
    assert "## Failed Variants" in report_text
    assert "B / dccs_refc" in report_text
    assert "failure_reasons=strict_artifact_missing=1" in report_text


def test_thesis_lane_script_supports_in_process_backend(tmp_path: Path, monkeypatch) -> None:
    report_md = tmp_path / "thesis_lane_report.md"
    report_json = tmp_path / "thesis_lane_report.json"
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon,distance_bin,seed\n"
        "od-1,52.0,-1.5,51.5,-1.2,30-100 km,7\n",
        encoding="utf-8",
    )
    artifacts = _seed_lane_artifacts(tmp_path)

    calls: dict[str, object] = {}

    class DummyClient:
        def __enter__(self):
            calls["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            calls["exited"] = True
            return False

    fake_app_module = types.ModuleType("app.main")
    fake_app_module.app = object()
    monkeypatch.setitem(sys.modules, "app.main", fake_app_module)
    monkeypatch.setattr(thesis_lane.pytest, "main", lambda args: 0)
    monkeypatch.setattr(thesis_lane.thesis_eval, "TestClient", lambda app: DummyClient())  # noqa: ARG005

    def fake_run(args, *, client=None):
        calls["client"] = client
        calls["in_process"] = bool(getattr(args, "in_process_backend", False))
        return _evaluation_payload(artifacts, run_id="run-789")

    monkeypatch.setattr(thesis_lane.thesis_eval, "run_thesis_evaluation", fake_run)

    exit_code = thesis_lane.main(
        [
            "--report-md",
            str(report_md),
            "--report-json",
            str(report_json),
            "--corpus-csv",
            str(corpus_csv),
            "--evaluation-args",
            "--in-process-backend",
        ]
    )

    assert exit_code == 0
    assert calls["entered"] is True
    assert calls["exited"] is True
    assert calls["client"] is not None
    assert calls["in_process"] is True


def test_thesis_lane_script_fails_closed_when_evaluation_artifacts_are_missing(tmp_path: Path, monkeypatch) -> None:
    report_md = tmp_path / "thesis_lane_report.md"
    report_json = tmp_path / "thesis_lane_report.json"
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon,distance_bin,seed\n"
        "od-1,52.0,-1.5,51.5,-1.2,30-100 km,7\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(thesis_lane.pytest, "main", lambda args: 0)
    monkeypatch.setattr(
        thesis_lane.thesis_eval,
        "run_thesis_evaluation",
        lambda args: {
            "run_id": "run-123",
            "results_csv": str(tmp_path / "missing-results.csv"),
            "summary_csv": str(tmp_path / "missing-summary.csv"),
            "thesis_report": str(tmp_path / "missing-report.md"),
            "methods_appendix": str(tmp_path / "missing-methods.md"),
            "evaluation_manifest": str(tmp_path / "missing-evaluation-manifest.json"),
            "manifest_path": str(tmp_path / "missing-manifest.json"),
        },
    )

    exit_code = thesis_lane.main(
        [
            "--report-md",
            str(report_md),
            "--report-json",
            str(report_json),
            "--corpus-csv",
            str(corpus_csv),
        ]
    )

    assert exit_code == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["evaluation"]["error"] == "RuntimeError"
    assert "thesis_lane_missing_artifact_file:results_csv" in payload["evaluation"]["message"]


def test_thesis_lane_script_fails_closed_when_required_artifact_path_key_is_missing(tmp_path: Path, monkeypatch) -> None:
    report_md = tmp_path / "thesis_lane_report.md"
    report_json = tmp_path / "thesis_lane_report.json"
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon,distance_bin,seed\n"
        "od-1,52.0,-1.5,51.5,-1.2,30-100 km,7\n",
        encoding="utf-8",
    )

    results_csv = tmp_path / "results.csv"
    thesis_report = tmp_path / "report.md"
    methods_appendix = tmp_path / "methods.md"
    evaluation_manifest = tmp_path / "evaluation_manifest.json"
    manifest_path = tmp_path / "manifest.json"
    _write(results_csv, "od_id,variant_id\nod-1,V0\n")
    _write(thesis_report, "# report\n")
    _write(methods_appendix, "# methods\n")
    _write(evaluation_manifest, "{}\n")
    _write(manifest_path, "{}\n")

    monkeypatch.setattr(thesis_lane.pytest, "main", lambda args: 0)
    monkeypatch.setattr(
        thesis_lane.thesis_eval,
        "run_thesis_evaluation",
        lambda args: {
            "run_id": "run-123",
            "results_csv": str(results_csv),
            "thesis_report": str(thesis_report),
            "methods_appendix": str(methods_appendix),
            "evaluation_manifest": str(evaluation_manifest),
            "manifest_path": str(manifest_path),
        },
    )

    exit_code = thesis_lane.main(
        [
            "--report-md",
            str(report_md),
            "--report-json",
            str(report_json),
            "--corpus-csv",
            str(corpus_csv),
        ]
    )

    assert exit_code == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["evaluation"]["error"] == "RuntimeError"
    assert "thesis_lane_missing_artifact_path:summary_csv" in payload["evaluation"]["message"]


def test_thesis_lane_script_returns_nonzero_when_pytest_fails(tmp_path: Path, monkeypatch) -> None:
    report_md = tmp_path / "thesis_lane_report.md"
    report_json = tmp_path / "thesis_lane_report.json"

    monkeypatch.setattr(thesis_lane.pytest, "main", lambda args: 2)

    exit_code = thesis_lane.main(
        [
            "--report-md",
            str(report_md),
            "--report-json",
            str(report_json),
            "--skip-evaluation",
        ]
    )

    assert exit_code == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["pytest_exit_code"] == 2
    assert payload["evaluation_exit_code"] == 0


def test_thesis_lane_script_reports_evaluation_exception(tmp_path: Path, monkeypatch) -> None:
    report_md = tmp_path / "thesis_lane_report.md"
    report_json = tmp_path / "thesis_lane_report.json"
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon,distance_bin,seed\n"
        "od-1,52.0,-1.5,51.5,-1.2,30-100 km,7\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(thesis_lane.pytest, "main", lambda args: 0)

    def _boom(args):  # noqa: ARG001
        raise ValueError("evaluation exploded")

    monkeypatch.setattr(thesis_lane.thesis_eval, "run_thesis_evaluation", _boom)

    exit_code = thesis_lane.main(
        [
            "--report-md",
            str(report_md),
            "--report-json",
            str(report_json),
            "--corpus-csv",
            str(corpus_csv),
        ]
    )

    assert exit_code == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["evaluation_exit_code"] == 1
    assert payload["evaluation"]["error"] == "ValueError"
    assert "evaluation exploded" in payload["evaluation"]["message"]


def test_thesis_lane_script_can_manage_local_backend_lifecycle(tmp_path: Path, monkeypatch) -> None:
    report_md = tmp_path / "thesis_lane_report.md"
    report_json = tmp_path / "thesis_lane_report.json"
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon,distance_bin,seed\n"
        "od-1,52.0,-1.5,51.5,-1.2,30-100 km,7\n",
        encoding="utf-8",
    )
    start_script = tmp_path / "start_backend_logged.ps1"
    stop_script = tmp_path / "stop_backend_logged.ps1"
    start_script.write_text("Write-Output 12345\n", encoding="utf-8")
    stop_script.write_text("Write-Output stopped\n", encoding="utf-8")
    artifacts = _seed_lane_artifacts(tmp_path)

    calls: list[list[str]] = []

    def _fake_run(command, **kwargs):  # noqa: ANN001
        calls.append(list(command))
        script_name = Path(command[5]).name
        if script_name == "stop_backend_logged.ps1":
            return CompletedProcess(command, 0, stdout="stopped\n", stderr="")
        return CompletedProcess(command, 0, stdout="12345\n", stderr="")

    monkeypatch.setattr(thesis_lane.pytest, "main", lambda args: 0)
    monkeypatch.setattr(thesis_lane.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        thesis_lane.thesis_eval,
        "run_thesis_evaluation",
        lambda args: _evaluation_payload(artifacts),
    )

    exit_code = thesis_lane.main(
        [
            "--report-md",
            str(report_md),
            "--report-json",
            str(report_json),
            "--corpus-csv",
            str(corpus_csv),
            "--manage-local-backend",
            "--backend-start-script",
            str(start_script),
            "--backend-stop-script",
            str(stop_script),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 3
    assert Path(calls[0][5]).name == "stop_backend_logged.ps1"
    assert Path(calls[1][5]).name == "start_backend_logged.ps1"
    assert Path(calls[2][5]).name == "stop_backend_logged.ps1"
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["backend_lifecycle"]["managed"] is True
    assert payload["backend_lifecycle"]["start_before_run"]["stdout"] == "12345"


def test_thesis_lane_defaults_to_checked_in_broad_corpus(tmp_path: Path, monkeypatch) -> None:
    report_md = tmp_path / "thesis_lane_report.md"
    report_json = tmp_path / "thesis_lane_report.json"
    captured: dict[str, object] = {}
    default_corpus = Path(thesis_lane._parser().get_default("corpus_csv"))
    default_corpus.parent.mkdir(parents=True, exist_ok=True)
    if not default_corpus.exists():
        default_corpus.write_text(
            "od_id,origin_lat,origin_lon,destination_lat,destination_lon,distance_bin,seed\n"
            "od-1,52.0,-1.5,51.5,-1.2,30-100 km,7\n",
            encoding="utf-8",
        )
    artifacts = _seed_lane_artifacts(tmp_path)

    monkeypatch.setattr(thesis_lane.pytest, "main", lambda args: 0)

    def _fake_eval(args):
        captured["corpus_csv"] = args.corpus_csv
        return _evaluation_payload(artifacts, run_id="run-default")

    monkeypatch.setattr(thesis_lane.thesis_eval, "run_thesis_evaluation", _fake_eval)

    exit_code = thesis_lane.main(["--report-md", str(report_md), "--report-json", str(report_json)])

    assert exit_code == 0
    assert Path(str(captured["corpus_csv"])) == default_corpus
