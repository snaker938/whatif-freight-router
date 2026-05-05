from __future__ import annotations

import csv
import hashlib
import json
import types
from pathlib import Path

import pytest

import scripts.run_thesis_campaign as campaign_module


pytestmark = pytest.mark.thesis_results


def _write_corpus(path: Path, od_ids: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "od_id",
                "origin_lat",
                "origin_lon",
                "destination_lat",
                "destination_lon",
                "distance_bin",
                "seed",
            ],
        )
        writer.writeheader()
        for index, od_id in enumerate(od_ids):
            writer.writerow(
                {
                    "od_id": od_id,
                    "origin_lat": 51.5 + index,
                    "origin_lon": -0.1,
                    "destination_lat": 52.5 + index,
                    "destination_lon": -1.1,
                    "distance_bin": "30-100 km",
                    "seed": 7 + index,
                }
            )
    return path


def _expected_seeded_random_subset(od_ids: list[str], *, seed: int, count: int) -> list[str]:
    unique_od_ids = list(dict.fromkeys(od_ids))
    original_index = {od_id: index for index, od_id in enumerate(unique_od_ids)}
    ordered = sorted(
        unique_od_ids,
        key=lambda od_id: (
            hashlib.sha256(f"{int(seed)}:{od_id}".encode("utf-8")).hexdigest(),
            original_index[od_id],
        ),
    )
    return ordered[:count]


def _eval_payload(
    run_args,
    *,
    red_pairs: set[tuple[str, str]] | None = None,
    non_dominant_pairs: set[tuple[str, str]] | None = None,
    non_time_preserving_pairs: set[tuple[str, str]] | None = None,
    omitted_od_ids: set[str] | None = None,
    missing_artifact_keys: set[str] | None = None,
    proof_grade_ready: bool = True,
) -> dict[str, object]:
    red_pairs = red_pairs or set()
    non_dominant_pairs = non_dominant_pairs or set()
    non_time_preserving_pairs = non_time_preserving_pairs or set()
    omitted_od_ids = omitted_od_ids or set()
    missing_artifact_keys = missing_artifact_keys or set()
    tranche_rows = campaign_module._load_csv_rows(Path(run_args.corpus_csv))
    rows: list[dict[str, object]] = []
    for od in tranche_rows:
        od_id = str(od["od_id"])
        if od_id in omitted_od_ids:
            continue
        for variant_id in ("A", "B", "C"):
            failed = (od_id, variant_id) in red_pairs
            dominance_failed = (od_id, variant_id) in non_dominant_pairs
            time_preserving_failed = (od_id, variant_id) in non_time_preserving_pairs
            rows.append(
                {
                    "od_id": od_id,
                    "variant_id": variant_id,
                    "failure_reason": "thesis_gate_failed" if failed else "",
                    "weighted_win_osrm": not failed,
                    "weighted_win_ors": not failed,
                    "balanced_win_osrm": not failed,
                    "balanced_win_ors": not failed,
                    "dominance_win_osrm": (not failed) and (not dominance_failed),
                    "dominance_win_ors": (not failed) and (not dominance_failed),
                    "time_preserving_win_osrm": (not failed) and (not time_preserving_failed),
                    "time_preserving_win_ors": (not failed) and (not time_preserving_failed),
                    "weighted_win_v0": not failed,
                    "balanced_win_v0": not failed,
                    "weighted_margin_vs_osrm": 1.25 if not failed else 0.25,
                    "weighted_margin_vs_ors": 1.1 if not failed else 0.1,
                    "weighted_margin_vs_v0": 1.0 if not failed else 0.0,
                    "selected_final_route_source_label": "graph_family:test",
                    "selected_candidate_source_label": "fallback:alternatives:test",
                    "preemptive_comparator_seeded": variant_id in {"B", "C"},
                    "selected_from_preemptive_comparator_seed": False,
                    "selected_from_supplemental_rescue": variant_id == "B",
                    "selected_from_comparator_engine": variant_id == "B",
                    "runtime_ms": 1500.0,
                    "algorithm_runtime_ms": 900.0,
                }
            )
    summary_rows = [
        {
            "variant_id": variant_id,
            "row_count": sum(1 for row in rows if row["variant_id"] == variant_id),
            "success_count": sum(
                1
                for row in rows
                if row["variant_id"] == variant_id and not row["failure_reason"]
            ),
            "failure_count": sum(
                1
                for row in rows
                if row["variant_id"] == variant_id and row["failure_reason"]
            ),
        }
        for variant_id in ("A", "B", "C")
    ]
    run_validity = {
        "strict_live_readiness_pass_rate": 1.0,
        "evaluation_rerun_success_rate": 1.0,
        "scenario_profile_unavailable_rate": 0.0,
    }
    artifact_root = Path(run_args.out_dir) / "artifacts" / str(run_args.run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    results_csv = artifact_root / "thesis_results.csv"
    summary_csv = artifact_root / "thesis_summary.csv"
    metadata_json = artifact_root / "metadata.json"
    if "results_csv" not in missing_artifact_keys:
        results_csv.write_text("od_id,variant_id\n", encoding="utf-8")
    if "summary_csv" not in missing_artifact_keys:
        summary_csv.write_text("variant_id,row_count\n", encoding="utf-8")
    evaluation_manifest = artifact_root / "evaluation_manifest.json"
    if "evaluation_manifest" not in missing_artifact_keys:
        evaluation_manifest.write_text("{}", encoding="utf-8")
    if "metadata" not in missing_artifact_keys:
        metadata_json.write_text(
            json.dumps(
                {
                    "backend_ready_summary": {
                        "strict_route_ready": True,
                        "route_graph": {
                            "ready_mode": "full" if proof_grade_ready else "fast",
                            "load_strategy": "streaming_json" if proof_grade_ready else "fast_startup_metadata",
                        },
                    },
                    "route_graph_full_hydration_observed": proof_grade_ready,
                    "degraded_evaluation_observed": not proof_grade_ready,
                    "strict_full_search_proof_eligible": proof_grade_ready,
                }
            ),
            encoding="utf-8",
        )
    payload: dict[str, object] = {
        "run_id": str(run_args.run_id),
        "rows": rows,
        "summary_rows": summary_rows,
        "run_validity": run_validity,
    }
    if "results_csv" not in missing_artifact_keys:
        payload["results_csv"] = str(results_csv)
    if "summary_csv" not in missing_artifact_keys:
        payload["summary_csv"] = str(summary_csv)
    if "evaluation_manifest" not in missing_artifact_keys:
        payload["evaluation_manifest"] = str(evaluation_manifest)
    if "metadata" not in missing_artifact_keys:
        payload["metadata"] = str(metadata_json)
    return payload


def test_select_new_od_ids_supports_seeded_random_admission_for_eligible_candidates() -> None:
    candidate_rows = [
        {"od_id": "cand-1"},
        {"od_id": "cand-2"},
        {"od_id": "cand-3"},
        {"od_id": "cand-4"},
        {"od_id": "cand-5"},
    ]

    assert campaign_module._select_new_od_ids(
        candidate_rows,
        completed_candidate_od_ids=["cand-2"],
        excluded_od_ids=["cand-5"],
        new_od_batch_size=2,
    ) == ["cand-1", "cand-3"]

    expected = _expected_seeded_random_subset(
        ["cand-1", "cand-3", "cand-4"],
        seed=11,
        count=2,
    )
    assert campaign_module._select_new_od_ids(
        candidate_rows,
        completed_candidate_od_ids=["cand-2"],
        excluded_od_ids=["cand-5"],
        new_od_batch_size=2,
        candidate_selection_mode="random_seeded",
        candidate_selection_seed=11,
    ) == expected


def test_thesis_campaign_bootstraps_then_widens_with_regression_carry_forward(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_csv = _write_corpus(tmp_path / "bootstrap.csv", ["boot-1"])
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1", "cand-2"])
    real_eval = campaign_module._get_thesis_eval()

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=lambda run_args, client=None: _eval_payload(run_args),
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "publishable-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--max-tranches",
                "2",
                "--require-balanced-win",
            ]
        )
    )

    assert payload["stop_reason"] == "max_tranches_reached"
    assert payload["tranche_count_executed"] == 2
    state = payload["state"]
    assert state["bootstrapped"] is True
    assert state["green_od_ids"] == ["boot-1", "cand-1"]
    assert state["completed_candidate_od_ids"] == ["cand-1"]
    first_tranche, second_tranche = payload["tranches"]
    assert first_tranche["new_od_ids"] == []
    assert first_tranche["regression_od_ids"] == ["boot-1"]
    assert second_tranche["regression_od_ids"] == ["boot-1"]
    assert second_tranche["new_od_ids"] == ["cand-1"]
    assert second_tranche["sequential_mode"] == "retain_green_replay_then_widen"
    assert second_tranche["selection_mode"] == "widen_after_green"
    assert second_tranche["retrying_pending_tranche"] is False
    assert second_tranche["retry_source_tranche_index"] is None
    assert second_tranche["prior_green_od_ids"] == ["boot-1"]
    assert second_tranche["replayed_prior_green_od_ids"] == ["boot-1"]
    assert second_tranche["missing_prior_green_replay_od_ids"] == []
    assert second_tranche["retained_green_od_ids"] == ["boot-1"]
    assert second_tranche["next_widening_allowed"] is True
    assert second_tranche["next_widening_block_reason"] == ""
    per_od_status = json.loads(Path(second_tranche["per_od_status_json"]).read_text(encoding="utf-8"))
    assert per_od_status["od_status"]["boot-1"]["passes_all_targets"] is True
    assert per_od_status["od_status"]["cand-1"]["passes_all_targets"] is True
    assert per_od_status["sequential_control"]["selection_mode"] == "widen_after_green"
    assert per_od_status["sequential_control"]["retained_green_od_ids"] == ["boot-1"]


def test_thesis_campaign_random_seeded_widening_preserves_replay_order_and_records_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_csv = _write_corpus(tmp_path / "bootstrap.csv", ["boot-1"])
    candidate_od_ids = ["cand-1", "cand-2", "cand-3", "cand-4"]
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", candidate_od_ids)
    real_eval = campaign_module._get_thesis_eval()

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=lambda run_args, client=None: _eval_payload(run_args),
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "random-seeded-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "2",
                "--candidate-selection-mode",
                "random_seeded",
                "--candidate-selection-seed",
                "11",
                "--max-tranches",
                "2",
                "--require-balanced-win",
            ]
        )
    )

    expected_new_od_ids = _expected_seeded_random_subset(candidate_od_ids, seed=11, count=2)
    second_tranche = payload["tranches"][-1]
    tranche_rows = campaign_module._load_csv_rows(Path(second_tranche["tranche_corpus_csv"]))
    per_od_status = json.loads(Path(second_tranche["per_od_status_json"]).read_text(encoding="utf-8"))
    campaign_report = Path(payload["campaign_report_md"]).read_text(encoding="utf-8")

    assert second_tranche["replay_od_ids"] == ["boot-1"]
    assert second_tranche["new_od_ids"] == expected_new_od_ids
    assert [str(row["od_id"]) for row in tranche_rows] == ["boot-1", *expected_new_od_ids]
    assert second_tranche["candidate_selection_mode"] == "random_seeded"
    assert second_tranche["candidate_selection_seed"] == 11
    assert per_od_status["sequential_control"]["candidate_selection_mode"] == "random_seeded"
    assert per_od_status["sequential_control"]["candidate_selection_seed"] == 11
    assert "- Candidate selection mode: `random_seeded`" in campaign_report
    assert "- Candidate selection seed: `11`" in campaign_report


def test_thesis_campaign_excludes_replay_set_ods_from_new_admission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_csv = _write_corpus(tmp_path / "bootstrap.csv", ["boot-1"])
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["boot-1", "cand-1"])
    real_eval = campaign_module._get_thesis_eval()

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=lambda run_args, client=None: _eval_payload(run_args),
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "exclude-replay-set-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--max-tranches",
                "2",
                "--require-balanced-win",
            ]
        )
    )

    second_tranche = payload["tranches"][-1]
    tranche_rows = campaign_module._load_csv_rows(Path(second_tranche["tranche_corpus_csv"]))
    per_od_status = json.loads(Path(second_tranche["per_od_status_json"]).read_text(encoding="utf-8"))
    campaign_report = Path(payload["campaign_report_md"]).read_text(encoding="utf-8")

    assert second_tranche["replay_od_ids"] == ["boot-1"]
    assert second_tranche["replay_od_count"] == 1
    assert second_tranche["regression_od_ids"] == ["boot-1"]
    assert second_tranche["new_od_ids"] == ["cand-1"]
    assert per_od_status["replay_od_ids"] == ["boot-1"]
    assert per_od_status["replay_od_count"] == 1
    assert [str(row["od_id"]) for row in tranche_rows] == ["boot-1", "cand-1"]
    assert "- Replay OD count: `1`" in campaign_report
    assert "- Replay OD ids: `boot-1`" in campaign_report
    assert "- Regression OD count: `1`" in campaign_report
    assert "- Regression OD ids: `boot-1`" in campaign_report
    assert "- New OD count: `1`" in campaign_report
    assert "- New OD ids: `cand-1`" in campaign_report
    assert "- Selection mode: `widen_after_green`" in campaign_report
    assert "- Prior green OD ids: `boot-1`" in campaign_report
    assert "- Next widening allowed: `True`" in campaign_report
    assert "- Promoted green OD count: `1`" in campaign_report
    assert "- Promoted green OD ids: `cand-1`" in campaign_report
    assert "- Unpreserved green OD count: `0`" in campaign_report
    assert "- Unpreserved green OD ids: `none`" in campaign_report
    assert "- Regression red OD count: `0`" in campaign_report
    assert "- Regression red OD ids: `none`" in campaign_report


def test_thesis_campaign_stops_on_regression_and_records_red_od(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_csv = _write_corpus(tmp_path / "bootstrap.csv", ["boot-1"])
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1"])
    run_counter = {"value": 0}
    real_eval = campaign_module._get_thesis_eval()

    def _fake_run(run_args, client=None):
        run_counter["value"] += 1
        if run_counter["value"] == 1:
            return _eval_payload(run_args)
        return _eval_payload(run_args, red_pairs={("boot-1", "C")})

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=_fake_run,
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "regression-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--max-tranches",
                "3",
                "--require-balanced-win",
            ]
        )
    )

    assert payload["stop_reason"] == "stop_on_regression_tranche"
    assert payload["tranche_count_executed"] == 2
    state = payload["state"]
    assert state["red_od_ids"] == ["boot-1"]
    assert state["green_od_ids"] == ["boot-1"]
    assert payload["tranches"][-1]["status"] == "regression"
    assert payload["tranches"][-1]["regression_red_od_ids"] == ["boot-1"]


def test_thesis_campaign_resume_retries_last_non_green_batch_before_admitting_next_slice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_csv = _write_corpus(tmp_path / "bootstrap.csv", ["boot-1"])
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1", "cand-2"])
    real_eval = campaign_module._get_thesis_eval()
    run_counter = {"value": 0}

    def _fake_run_red_then_green(run_args, client=None):
        run_counter["value"] += 1
        if run_counter["value"] == 2:
            return _eval_payload(run_args, red_pairs={("cand-1", "C")})
        return _eval_payload(run_args)

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=_fake_run_red_then_green,
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    first_payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "resume-retry-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--max-tranches",
                "2",
                "--require-balanced-win",
            ]
        )
    )

    assert first_payload["stop_reason"] == "stop_on_red_tranche"
    assert first_payload["tranches"][-1]["status"] == "red"
    assert first_payload["tranches"][-1]["new_od_ids"] == ["cand-1"]
    assert first_payload["state"]["completed_candidate_od_ids"] == []

    resumed_payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "resume-retry-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--max-tranches",
                "1",
                "--resume",
                "--require-balanced-win",
            ]
        )
    )

    tranche = resumed_payload["tranches"][-1]
    tranche_rows = campaign_module._load_csv_rows(Path(tranche["tranche_corpus_csv"]))
    per_od_status = json.loads(Path(tranche["per_od_status_json"]).read_text(encoding="utf-8"))

    assert tranche["replay_od_ids"] == ["boot-1"]
    assert tranche["new_od_ids"] == ["cand-1"]
    assert tranche["selection_mode"] == "retry_pending_tranche"
    assert tranche["retrying_pending_tranche"] is True
    assert tranche["retry_source_tranche_index"] == 2
    assert tranche["next_widening_allowed"] is True
    assert [str(row["od_id"]) for row in tranche_rows] == ["boot-1", "cand-1"]
    assert resumed_payload["state"]["completed_candidate_od_ids"] == ["cand-1"]
    assert per_od_status["sequential_control"]["selection_mode"] == "retry_pending_tranche"
    assert per_od_status["sequential_control"]["retry_source_tranche_index"] == 2


def test_thesis_campaign_resume_random_seeded_reuses_pending_new_od_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_csv = _write_corpus(tmp_path / "bootstrap.csv", ["boot-1"])
    candidate_od_ids = ["cand-1", "cand-2", "cand-3"]
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", candidate_od_ids)
    real_eval = campaign_module._get_thesis_eval()
    run_counter = {"value": 0}
    first_random_pick = _expected_seeded_random_subset(candidate_od_ids, seed=19, count=1)[0]

    def _fake_run_red_then_green(run_args, client=None):
        run_counter["value"] += 1
        if run_counter["value"] == 2:
            return _eval_payload(run_args, red_pairs={(first_random_pick, "C")})
        return _eval_payload(run_args)

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=_fake_run_red_then_green,
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    first_payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "resume-random-seeded-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--candidate-selection-mode",
                "random_seeded",
                "--candidate-selection-seed",
                "19",
                "--max-tranches",
                "2",
                "--require-balanced-win",
            ]
        )
    )

    assert first_payload["stop_reason"] == "stop_on_red_tranche"
    assert first_payload["tranches"][-1]["new_od_ids"] == [first_random_pick]

    resumed_payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "resume-random-seeded-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--candidate-selection-mode",
                "random_seeded",
                "--candidate-selection-seed",
                "19",
                "--max-tranches",
                "1",
                "--resume",
                "--require-balanced-win",
            ]
        )
    )

    tranche = resumed_payload["tranches"][-1]
    assert tranche["selection_mode"] == "retry_pending_tranche"
    assert tranche["new_od_ids"] == [first_random_pick]
    assert resumed_payload["state"]["completed_candidate_od_ids"] == [first_random_pick]


def test_thesis_campaign_marks_missing_expected_regression_od_as_regression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_csv = _write_corpus(tmp_path / "bootstrap.csv", ["boot-1"])
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1"])
    run_counter = {"value": 0}
    real_eval = campaign_module._get_thesis_eval()

    def _fake_run(run_args, client=None):
        run_counter["value"] += 1
        if run_counter["value"] == 1:
            return _eval_payload(run_args)
        return _eval_payload(run_args, omitted_od_ids={"boot-1"})

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=_fake_run,
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "missing-regression-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--max-tranches",
                "2",
                "--require-balanced-win",
            ]
        )
    )

    tranche = payload["tranches"][-1]
    per_od_status = json.loads(Path(tranche["per_od_status_json"]).read_text(encoding="utf-8"))

    assert tranche["status"] == "regression"
    assert tranche["missing_expected_od_ids"] == ["boot-1"]
    assert tranche["unpreserved_green_od_ids"] == ["boot-1"]
    assert per_od_status["od_status"]["boot-1"]["evaluated"] is False
    assert "missing_expected_od" in per_od_status["od_status"]["boot-1"]["reasons"]


def test_thesis_campaign_blocks_green_when_required_artifacts_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_csv = _write_corpus(tmp_path / "bootstrap.csv", ["boot-1"])
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1"])
    real_eval = campaign_module._get_thesis_eval()

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=lambda run_args, client=None: _eval_payload(
                run_args,
                missing_artifact_keys={"evaluation_manifest"},
            ),
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "blocked-evidence-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--bootstrap-csv",
                str(bootstrap_csv),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--max-tranches",
                "1",
                "--require-balanced-win",
            ]
        )
    )

    tranche = payload["tranches"][-1]
    per_od_status = json.loads(Path(tranche["per_od_status_json"]).read_text(encoding="utf-8"))

    assert tranche["status"] == "blocked"
    assert tranche["publication_evidence_ok"] is False
    assert tranche["missing_required_artifacts"] == ["evaluation_manifest"]
    assert payload["state"]["green_od_ids"] == []
    assert payload["state"]["completed_candidate_od_ids"] == []
    assert per_od_status["publication_evidence"]["required_artifacts_ok"] is False
    assert per_od_status["publication_evidence"]["missing_required_artifacts"] == ["evaluation_manifest"]


def test_thesis_campaign_can_require_dominance_and_time_preserving_wins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1"])
    real_eval = campaign_module._get_thesis_eval()

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=lambda run_args, client=None: _eval_payload(
                run_args,
                non_dominant_pairs={("cand-1", "C")},
                non_time_preserving_pairs={("cand-1", "C")},
            ),
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "strong-route-quality-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--max-tranches",
                "1",
                "--require-balanced-win",
                "--require-dominance-win",
                "--require-time-preserving-win",
            ]
        )
    )

    tranche = payload["tranches"][-1]
    per_od_status = json.loads(Path(tranche["per_od_status_json"]).read_text(encoding="utf-8"))

    assert tranche["status"] == "red"
    assert tranche["new_red_od_ids"] == ["cand-1"]
    assert per_od_status["od_status"]["cand-1"]["passes_all_targets"] is False
    assert "C:dominance_win_osrm" in per_od_status["od_status"]["cand-1"]["reasons"]
    assert "C:time_preserving_win_osrm" in per_od_status["od_status"]["cand-1"]["reasons"]


def test_thesis_campaign_blocks_green_when_proof_grade_readiness_is_required(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1"])
    real_eval = campaign_module._get_thesis_eval()

    monkeypatch.setattr(
        campaign_module,
        "_get_thesis_eval",
        lambda: types.SimpleNamespace(
            _build_parser=real_eval._build_parser,
            run_thesis_evaluation=lambda run_args, client=None: _eval_payload(
                run_args,
                proof_grade_ready=False,
            ),
            TestClient=getattr(real_eval, "TestClient", None),
        ),
    )

    payload = campaign_module.run_campaign(
        campaign_module._parser().parse_args(
            [
                "--campaign-id",
                "proof-grade-loop",
                "--campaign-dir",
                str(tmp_path / "campaign"),
                "--corpus-csv",
                str(candidate_csv),
                "--new-od-batch-size",
                "1",
                "--max-tranches",
                "1",
                "--require-balanced-win",
                "--require-proof-grade-readiness",
            ]
        )
    )

    tranche = payload["tranches"][-1]
    per_od_status = json.loads(Path(tranche["per_od_status_json"]).read_text(encoding="utf-8"))
    proof_grade = per_od_status["publication_evidence"]["proof_grade_readiness"]

    assert tranche["status"] == "blocked"
    assert tranche["publication_evidence_ok"] is False
    assert tranche["proof_grade_readiness_ok"] is False
    assert "degraded_evaluation_observed" in tranche["proof_grade_readiness_reasons"]
    assert proof_grade["proof_grade_readiness_ok"] is False
    assert "strict_full_search_proof_eligible" in proof_grade["proof_grade_readiness_reasons"]


def test_thesis_campaign_build_eval_args_overrides_conflicting_forwarded_values(tmp_path: Path) -> None:
    args = campaign_module._parser().parse_args(
        [
            "--campaign-dir",
            str(tmp_path / "campaign"),
            "--corpus-csv",
            str(tmp_path / "candidate.csv"),
            "--route-graph-asset-path",
            str(tmp_path / "subset-graph.json"),
            "--route-graph-min-nodes",
            "12",
            "--route-graph-min-adjacency",
            "9",
            "--evaluation-args",
            "--corpus-csv",
            "old.csv",
            "--run-id",
            "old-run",
            "--out-dir",
            "old-out",
            "--cache-mode",
            "cold",
            "--route-graph-asset-path",
            "old-graph.json",
            "--route-graph-min-nodes",
            "999",
            "--route-graph-min-adjacency",
            "888",
            "--in-process-backend",
        ]
    )

    eval_args = campaign_module._build_eval_args(
        args,
        tranche_corpus_csv=tmp_path / "tranche.csv",
        tranche_run_id="fresh-run",
        tranche_out_dir=tmp_path / "tranche-out",
    )

    assert Path(eval_args.corpus_csv) == tmp_path / "tranche.csv"
    assert eval_args.run_id == "fresh-run"
    assert Path(eval_args.out_dir) == tmp_path / "tranche-out"
    assert eval_args.cache_mode == "cold"
    assert eval_args.in_process_backend is True
    assert Path(eval_args.route_graph_asset_path) == tmp_path / "subset-graph.json"
    assert eval_args.route_graph_min_nodes == 12
    assert eval_args.route_graph_min_adjacency == 9


def test_apply_evaluation_fast_startup_env_sets_explicit_runtime_flags(monkeypatch) -> None:
    monkeypatch.delenv("ROUTE_GRAPH_EVALUATION_FAST_STARTUP_ALLOWED", raising=False)
    monkeypatch.delenv("ROUTE_GRAPH_FAST_STARTUP_ENABLED", raising=False)

    updates = campaign_module._apply_evaluation_fast_startup_env(enabled=True)

    assert updates == {
        "ROUTE_GRAPH_EVALUATION_FAST_STARTUP_ALLOWED": "true",
        "ROUTE_GRAPH_FAST_STARTUP_ENABLED": "true",
    }
    assert campaign_module.os.environ["ROUTE_GRAPH_EVALUATION_FAST_STARTUP_ALLOWED"] == "true"
    assert campaign_module.os.environ["ROUTE_GRAPH_FAST_STARTUP_ENABLED"] == "true"


def test_apply_evaluation_runtime_env_sets_subset_graph_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROUTE_GRAPH_EVALUATION_FAST_STARTUP_ALLOWED", raising=False)
    monkeypatch.delenv("ROUTE_GRAPH_FAST_STARTUP_ENABLED", raising=False)
    monkeypatch.delenv("ROUTE_GRAPH_ASSET_PATH", raising=False)
    monkeypatch.delenv("ROUTE_GRAPH_MIN_NODES", raising=False)
    monkeypatch.delenv("ROUTE_GRAPH_MIN_ADJACENCY", raising=False)

    args = campaign_module._parser().parse_args(
        [
            "--campaign-dir",
            str(tmp_path / "campaign"),
            "--corpus-csv",
            str(tmp_path / "candidate.csv"),
            "--evaluation-fast-startup",
            "--route-graph-asset-path",
            str(tmp_path / "subset-graph.json"),
            "--route-graph-min-nodes",
            "15",
            "--route-graph-min-adjacency",
            "11",
        ]
    )

    fake_settings = types.SimpleNamespace(
        route_graph_evaluation_fast_startup_allowed=False,
        route_graph_fast_startup_enabled=False,
        route_graph_asset_path="",
        route_graph_min_nodes=0,
        route_graph_min_adjacency=0,
    )
    monkeypatch.setitem(campaign_module.sys.modules, "app.settings", types.SimpleNamespace(settings=fake_settings))

    updates = campaign_module._apply_evaluation_runtime_env(args)

    assert updates == {
        "ROUTE_GRAPH_EVALUATION_FAST_STARTUP_ALLOWED": "true",
        "ROUTE_GRAPH_FAST_STARTUP_ENABLED": "true",
        "ROUTE_GRAPH_ASSET_PATH": str((tmp_path / "subset-graph.json").resolve()),
        "ROUTE_GRAPH_MIN_NODES": "15",
        "ROUTE_GRAPH_MIN_ADJACENCY": "11",
    }
    assert campaign_module.os.environ["ROUTE_GRAPH_ASSET_PATH"] == str((tmp_path / "subset-graph.json").resolve())
    assert campaign_module.os.environ["ROUTE_GRAPH_MIN_NODES"] == "15"
    assert campaign_module.os.environ["ROUTE_GRAPH_MIN_ADJACENCY"] == "11"
    assert fake_settings.route_graph_evaluation_fast_startup_allowed is True
    assert fake_settings.route_graph_fast_startup_enabled is True
    assert fake_settings.route_graph_asset_path == str((tmp_path / "subset-graph.json").resolve())
    assert fake_settings.route_graph_min_nodes == 15
    assert fake_settings.route_graph_min_adjacency == 11


def test_build_route_graph_asset_plan_uses_requested_subset_corridor_km(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROUTE_GRAPH_ASSET_PATH", raising=False)
    source_asset = tmp_path / "routing_graph_uk.json"
    source_asset.write_text("{}", encoding="utf-8")
    corpus_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1", "cand-2"])
    build_calls: list[dict[str, object]] = []

    def _fake_build_subset(*, graph_json, corpus_csv, output_json, corridor_km, buffer_km=None):
        build_calls.append(
            {
                "graph_json": str(graph_json),
                "corpus_csv": str(corpus_csv),
                "output_json": str(output_json),
                "corridor_km": float(corridor_km),
                "buffer_km": buffer_km,
            }
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text("{}", encoding="utf-8")
        meta_path = output_json.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "version": "uk-routing-graph-subset-v2",
                    "output_json": str(output_json),
                    "output_meta_json": str(meta_path),
                    "corridor_km": float(corridor_km),
                    "nodes_kept": 24,
                    "edges_kept": 36,
                    "compact_bundle": "",
                    "compact_bundle_meta": "",
                }
            ),
            encoding="utf-8",
        )
        return json.loads(meta_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(campaign_module, "_default_route_graph_asset_path", lambda: source_asset)
    monkeypatch.setattr(campaign_module, "_route_graph_stage_root", lambda: tmp_path / "staged_subsets")
    monkeypatch.setattr(campaign_module, "_hash_file", lambda path: "c" * 64)
    monkeypatch.setattr(campaign_module, "_route_graph_source_signature", lambda path: "s" * 64)
    monkeypatch.setattr(campaign_module, "_count_route_graph_adjacency_keys", lambda path: 18)
    monkeypatch.setattr(
        campaign_module,
        "_get_route_graph_subset_builder",
        lambda: types.SimpleNamespace(DEFAULT_CORRIDOR_KM=35.0, build_subset=_fake_build_subset),
    )

    plan = campaign_module._build_route_graph_asset_plan(
        corpus_csv=corpus_csv,
        requested_subset_corridor_km=12.5,
    )

    assert build_calls
    assert build_calls[0]["corridor_km"] == pytest.approx(12.5)
    assert ".subset.c12p5." in str(build_calls[0]["output_json"])
    assert plan["mode"] == "staged_subset_asset"
    assert plan["subset_corridor_km"] == pytest.approx(12.5)
    assert plan["route_graph_min_nodes"] == 24
    assert plan["route_graph_min_adjacency"] == 18


def test_prepare_in_process_runtime_env_preloads_single_tranche_subset_asset(tmp_path: Path, monkeypatch) -> None:
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1", "cand-2", "cand-3"])
    captured: dict[str, object] = {}

    def _fake_apply_env(args) -> dict[str, str]:
        captured["route_graph_asset_path"] = str(getattr(args, "route_graph_asset_path", ""))
        captured["route_graph_min_nodes"] = getattr(args, "route_graph_min_nodes", None)
        captured["route_graph_min_adjacency"] = getattr(args, "route_graph_min_adjacency", None)
        return {"ROUTE_GRAPH_ASSET_PATH": str(getattr(args, "route_graph_asset_path", ""))}

    monkeypatch.setattr(
        campaign_module,
        "_build_route_graph_asset_plan",
        lambda **kwargs: {
            "asset_path": str((tmp_path / "subset-graph.json").resolve()),
            "route_graph_min_nodes": 14,
            "route_graph_min_adjacency": 9,
            "subset_corridor_km": 11.0,
        },
    )
    monkeypatch.setattr(campaign_module, "_apply_evaluation_runtime_env", _fake_apply_env)

    args = campaign_module._parser().parse_args(
        [
            "--campaign-dir",
            str(tmp_path / "campaign"),
            "--corpus-csv",
            str(candidate_csv),
            "--new-od-batch-size",
            "2",
            "--max-tranches",
            "1",
            "--stage-route-graph-subset",
            "--route-graph-subset-corridor-km",
            "11",
            "--evaluation-args",
            "--in-process-backend",
        ]
    )

    payload = campaign_module._prepare_in_process_runtime_env(args)

    assert Path(str(payload["preview_tranche_corpus_csv"])) == tmp_path / "campaign" / "tranche_001" / "tranche_od_corpus.csv"
    assert captured["route_graph_asset_path"] == str((tmp_path / "subset-graph.json").resolve())
    assert captured["route_graph_min_nodes"] == 14
    assert captured["route_graph_min_adjacency"] == 9
    assert args.route_graph_asset_path == str((tmp_path / "subset-graph.json").resolve())
    assert args.route_graph_min_nodes == 14
    assert args.route_graph_min_adjacency == 9


def test_prepare_in_process_runtime_env_rejects_multi_tranche_subset_staging(tmp_path: Path) -> None:
    candidate_csv = _write_corpus(tmp_path / "candidate.csv", ["cand-1", "cand-2"])
    args = campaign_module._parser().parse_args(
        [
            "--campaign-dir",
            str(tmp_path / "campaign"),
            "--corpus-csv",
            str(candidate_csv),
            "--new-od-batch-size",
            "1",
            "--max-tranches",
            "2",
            "--stage-route-graph-subset",
            "--evaluation-args",
            "--in-process-backend",
        ]
    )

    with pytest.raises(RuntimeError, match="single_non_resume_tranche"):
        campaign_module._prepare_in_process_runtime_env(args)
