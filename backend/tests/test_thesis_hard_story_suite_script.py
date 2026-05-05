from __future__ import annotations

import csv
import json
from pathlib import Path

import scripts.run_thesis_hard_story_suite as hard_suite


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_hard_story_suite_runs_default_lanes_and_composes_successful_results(tmp_path: Path, monkeypatch) -> None:
    lane_calls: list[list[str]] = []
    compose_calls: list[dict[str, object]] = []

    def _fake_lane_main(argv: list[str]) -> int:
        lane_calls.append(list(argv))
        report_md = Path(argv[argv.index("--report-md") + 1])
        report_json = Path(argv[argv.index("--report-json") + 1])
        corpus_csv = Path(argv[argv.index("--corpus-csv") + 1])
        evaluation_args = argv[argv.index("--evaluation-args") + 1 :]
        suite_role = evaluation_args[evaluation_args.index("--evaluation-suite-role") + 1]
        lane_dir = report_json.parent
        results_csv = _write_csv(
            lane_dir / "thesis_results.csv",
            [
                {
                    "artifact_run_id": f"{suite_role}-run",
                    "od_id": f"{corpus_csv.stem}-od",
                    "profile_id": f"{corpus_csv.stem}-profile",
                    "variant_id": "A",
                    "pipeline_mode": "dccs",
                    "origin_lat": 51.5,
                    "origin_lon": -0.1,
                    "destination_lat": 54.9,
                    "destination_lon": -1.6,
                    "trip_length_bin": "250-500 km",
                    "corpus_group": "ambiguity",
                    "corpus_kind": "ambiguity",
                    "ambiguity_prior_source": "probe",
                    "weighted_win_osrm": "true",
                    "weighted_win_ors": "true",
                    "dominates_osrm": "false",
                    "dominates_ors": "false",
                    "time_preserving_win_osrm": "true",
                    "time_preserving_win_ors": "true",
                    "runtime_ms": 100.0,
                    "algorithm_runtime_ms": 80.0,
                    "baseline_osrm_ms": 20.0,
                    "baseline_ors_ms": 25.0,
                    "route_evidence_ok": "true",
                    "artifact_complete": "true",
                    "failure_reason": "",
                }
            ],
        )
        payload = {
            "evaluation": {
                "results_csv": str(results_csv),
                "summary_csv": str(lane_dir / "thesis_summary.csv"),
                "summary_rows": [
                    {
                        "variant_id": "A",
                        "pipeline_mode": "dccs",
                        "row_count": 1,
                        "success_count": 1,
                        "failure_count": 0,
                        "weighted_win_rate_best_baseline": 1.0,
                    }
                ],
            }
        }
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text(f"# {suite_role}\n", encoding="utf-8")
        report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0

    def _fake_compose(args) -> dict[str, object]:
        compose_calls.append(
            {
                "run_id": args.run_id,
                "canonical_corpus": args.canonical_corpus,
                "results_csv": list(args.results_csv),
            }
        )
        return {
            "results_csv": str(tmp_path / "composed" / "thesis_results.csv"),
            "summary_csv": str(tmp_path / "composed" / "thesis_summary.csv"),
            "thesis_report": str(tmp_path / "composed" / "thesis_report.md"),
            "evaluation_manifest": str(tmp_path / "composed" / "evaluation_manifest.json"),
        }

    monkeypatch.setattr(hard_suite.thesis_lane, "main", _fake_lane_main)
    monkeypatch.setattr(hard_suite.compose_sharded, "compose_sharded_report", _fake_compose)

    exit_code = hard_suite.main(
        [
            "--suite-id",
            "hard-story-test",
            "--suite-dir",
            str(tmp_path / "suite"),
            "--out-dir",
            str(tmp_path / "out"),
            "--canonical-corpus",
            str(tmp_path / "canonical.csv"),
        ]
    )

    assert exit_code == 0
    assert len(lane_calls) == len(hard_suite.DEFAULT_LANE_SPECS)
    assert compose_calls[0]["run_id"] == "hard-story-test_composed"
    assert len(compose_calls[0]["results_csv"]) == len(hard_suite.DEFAULT_LANE_SPECS)
    report_payload = json.loads((tmp_path / "suite" / "hard_story_suite_report.json").read_text(encoding="utf-8"))
    assert report_payload["successful_lane_count"] == len(hard_suite.DEFAULT_LANE_SPECS)
    assert report_payload["failed_lane_count"] == 0
    assert report_payload["composed_payload"]["summary_csv"].endswith("thesis_summary.csv")
    report_text = (tmp_path / "suite" / "hard_story_suite_report.md").read_text(encoding="utf-8")
    assert "`hard_mixed_24` / `hard_mixed_story`" in report_text
    assert "`longcorr_hard_32` / `longcorr_story`" in report_text


def test_hard_story_suite_returns_nonzero_when_any_lane_fails(tmp_path: Path, monkeypatch) -> None:
    lane_counter = {"value": 0}

    def _fake_lane_main(argv: list[str]) -> int:
        lane_counter["value"] += 1
        report_md = Path(argv[argv.index("--report-md") + 1])
        report_json = Path(argv[argv.index("--report-json") + 1])
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text("# lane\n", encoding="utf-8")
        if lane_counter["value"] == 1:
            results_csv = _write_csv(
                report_json.parent / "thesis_results.csv",
                [
                    {
                        "artifact_run_id": "lane-1",
                        "od_id": "od-1",
                        "profile_id": "profile-1",
                        "variant_id": "A",
                        "pipeline_mode": "dccs",
                        "origin_lat": 51.5,
                        "origin_lon": -0.1,
                        "destination_lat": 54.9,
                        "destination_lon": -1.6,
                        "trip_length_bin": "250-500 km",
                        "corpus_group": "ambiguity",
                        "corpus_kind": "ambiguity",
                        "ambiguity_prior_source": "probe",
                        "weighted_win_osrm": "true",
                        "weighted_win_ors": "true",
                        "dominates_osrm": "false",
                        "dominates_ors": "false",
                        "time_preserving_win_osrm": "true",
                        "time_preserving_win_ors": "true",
                        "runtime_ms": 100.0,
                        "algorithm_runtime_ms": 80.0,
                        "baseline_osrm_ms": 20.0,
                        "baseline_ors_ms": 25.0,
                        "route_evidence_ok": "true",
                        "artifact_complete": "true",
                        "failure_reason": "",
                    }
                ],
            )
            report_json.write_text(
                json.dumps({"evaluation": {"results_csv": str(results_csv), "summary_rows": []}}, indent=2),
                encoding="utf-8",
            )
            return 0
        report_json.write_text(json.dumps({"evaluation": {"error": "RuntimeError"}}, indent=2), encoding="utf-8")
        return 1

    compose_calls: list[list[str]] = []

    def _fake_compose(args) -> dict[str, object]:
        compose_calls.append(list(args.results_csv))
        return {
            "results_csv": str(tmp_path / "composed" / "thesis_results.csv"),
            "summary_csv": str(tmp_path / "composed" / "thesis_summary.csv"),
            "thesis_report": str(tmp_path / "composed" / "thesis_report.md"),
            "evaluation_manifest": str(tmp_path / "composed" / "evaluation_manifest.json"),
        }

    monkeypatch.setattr(hard_suite.thesis_lane, "main", _fake_lane_main)
    monkeypatch.setattr(hard_suite.compose_sharded, "compose_sharded_report", _fake_compose)

    exit_code = hard_suite.main(
        [
            "--suite-id",
            "partial-hard-story-test",
            "--suite-dir",
            str(tmp_path / "suite"),
            "--out-dir",
            str(tmp_path / "out"),
            "--canonical-corpus",
            str(tmp_path / "canonical.csv"),
        ]
    )

    assert exit_code == 1
    assert len(compose_calls) == 1
    assert len(compose_calls[0]) == 1
    report_payload = json.loads((tmp_path / "suite" / "hard_story_suite_report.json").read_text(encoding="utf-8"))
    assert report_payload["successful_lane_count"] == 1
    assert report_payload["failed_lane_count"] == 1


def test_hard_story_suite_threads_backend_memory_limit_to_lane_invocations(tmp_path: Path, monkeypatch) -> None:
    lane_calls: list[list[str]] = []

    def _fake_lane_main(argv: list[str]) -> int:
        lane_calls.append(list(argv))
        report_md = Path(argv[argv.index("--report-md") + 1])
        report_json = Path(argv[argv.index("--report-json") + 1])
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text("# lane\n", encoding="utf-8")
        results_csv = _write_csv(
            report_json.parent / "thesis_results.csv",
            [
                {
                    "artifact_run_id": "lane-1",
                    "od_id": "od-1",
                    "profile_id": "profile-1",
                    "variant_id": "A",
                    "pipeline_mode": "dccs",
                    "origin_lat": 51.5,
                    "origin_lon": -0.1,
                    "destination_lat": 54.9,
                    "destination_lon": -1.6,
                    "trip_length_bin": "250-500 km",
                    "corpus_group": "ambiguity",
                    "corpus_kind": "ambiguity",
                    "ambiguity_prior_source": "probe",
                    "weighted_win_osrm": "true",
                    "weighted_win_ors": "true",
                    "dominates_osrm": "false",
                    "dominates_ors": "false",
                    "time_preserving_win_osrm": "true",
                    "time_preserving_win_ors": "true",
                    "runtime_ms": 100.0,
                    "algorithm_runtime_ms": 80.0,
                    "baseline_osrm_ms": 20.0,
                    "baseline_ors_ms": 25.0,
                    "route_evidence_ok": "true",
                    "artifact_complete": "true",
                    "failure_reason": "",
                }
            ],
        )
        report_json.write_text(
            json.dumps({"evaluation": {"results_csv": str(results_csv), "summary_rows": []}}, indent=2),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(hard_suite.thesis_lane, "main", _fake_lane_main)
    monkeypatch.setattr(
        hard_suite.compose_sharded,
        "compose_sharded_report",
        lambda args: {
            "results_csv": str(tmp_path / "composed" / "thesis_results.csv"),
            "summary_csv": str(tmp_path / "composed" / "thesis_summary.csv"),
            "thesis_report": str(tmp_path / "composed" / "thesis_report.md"),
            "evaluation_manifest": str(tmp_path / "composed" / "evaluation_manifest.json"),
        },
    )

    exit_code = hard_suite.main(
        [
            "--suite-id",
            "memory-capped-hard-story",
            "--suite-dir",
            str(tmp_path / "suite"),
            "--out-dir",
            str(tmp_path / "out"),
            "--canonical-corpus",
            str(tmp_path / "canonical.csv"),
            "--manage-local-backend",
            "--backend-memory-limit-mb",
            "1792",
        ]
    )

    assert exit_code == 0
    assert len(lane_calls) == len(hard_suite.DEFAULT_LANE_SPECS)
    for argv in lane_calls:
        assert "--backend-memory-limit-mb" in argv
        assert argv[argv.index("--backend-memory-limit-mb") + 1] == "1792"
