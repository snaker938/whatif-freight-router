from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.compare_thesis_runs as compare_module


pytestmark = pytest.mark.thesis_results


def _write_csv(path: Path, *, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_compare_thesis_runs_writes_rowwise_diff_and_summary_sidecar(tmp_path: Path) -> None:
    result_fields = [
        "od_id",
        "variant_id",
        "pipeline_mode",
        "route_id",
        "selected_distance_km",
        "selected_duration_s",
        "selected_monetary_cost",
        "selected_emissions_kg",
        "certificate",
        "certified",
        "frontier_count",
        "frontier_hypervolume",
        "weighted_win_osrm",
        "weighted_win_ors",
        "weighted_margin_vs_osrm",
        "runtime_ms",
        "algorithm_runtime_ms",
        "selected_candidate_source_label",
        "selected_final_route_source_label",
        "selected_from_supplemental_rescue",
        "selected_from_comparator_engine",
        "selected_from_preemptive_comparator_seed",
        "preemptive_comparator_seeded",
        "failure_reason",
        "artifact_complete",
        "route_evidence_ok",
    ]
    before_results = _write_csv(
        tmp_path / "before_results.csv",
        fieldnames=result_fields,
        rows=[
            {
                "od_id": "od-1",
                "variant_id": "A",
                "pipeline_mode": "dccs",
                "route_id": "route-old",
                "selected_distance_km": 100.0,
                "selected_duration_s": 7200.0,
                "selected_monetary_cost": 80.0,
                "selected_emissions_kg": 150.0,
                "certificate": 0.75,
                "certified": True,
                "frontier_count": 3,
                "frontier_hypervolume": 1200.0,
                "weighted_win_osrm": True,
                "weighted_win_ors": True,
                "weighted_margin_vs_osrm": 2.5,
                "runtime_ms": 5000.0,
                "algorithm_runtime_ms": 3000.0,
                "selected_candidate_source_label": "fallback:old",
                "selected_final_route_source_label": "graph_family:old",
                "selected_from_supplemental_rescue": False,
                "selected_from_comparator_engine": False,
                "selected_from_preemptive_comparator_seed": False,
                "preemptive_comparator_seeded": False,
                "failure_reason": "",
                "artifact_complete": True,
                "route_evidence_ok": True,
            },
            {
                "od_id": "od-2",
                "variant_id": "B",
                "pipeline_mode": "dccs_refc",
                "route_id": "route-stable",
                "selected_distance_km": 200.0,
                "selected_duration_s": 9100.0,
                "selected_monetary_cost": 90.0,
                "selected_emissions_kg": 180.0,
                "certificate": 0.92,
                "certified": True,
                "frontier_count": 4,
                "frontier_hypervolume": 2400.0,
                "weighted_win_osrm": True,
                "weighted_win_ors": True,
                "weighted_margin_vs_osrm": 3.0,
                "runtime_ms": 6100.0,
                "algorithm_runtime_ms": 3400.0,
                "selected_candidate_source_label": "fallback:stable",
                "selected_final_route_source_label": "graph_family:stable",
                "selected_from_supplemental_rescue": False,
                "selected_from_comparator_engine": False,
                "selected_from_preemptive_comparator_seed": False,
                "preemptive_comparator_seeded": True,
                "failure_reason": "",
                "artifact_complete": True,
                "route_evidence_ok": True,
            },
        ],
    )
    after_results = _write_csv(
        tmp_path / "after_results.csv",
        fieldnames=result_fields,
        rows=[
            {
                "od_id": "od-1",
                "variant_id": "A",
                "pipeline_mode": "dccs",
                "route_id": "route-new",
                "selected_distance_km": 98.0,
                "selected_duration_s": 7000.0,
                "selected_monetary_cost": 76.0,
                "selected_emissions_kg": 145.0,
                "certificate": 0.88,
                "certified": True,
                "frontier_count": 4,
                "frontier_hypervolume": 1500.0,
                "weighted_win_osrm": True,
                "weighted_win_ors": True,
                "weighted_margin_vs_osrm": 4.0,
                "runtime_ms": 4300.0,
                "algorithm_runtime_ms": 2500.0,
                "selected_candidate_source_label": "fallback:new",
                "selected_final_route_source_label": "graph_family:new",
                "selected_from_supplemental_rescue": True,
                "selected_from_comparator_engine": False,
                "selected_from_preemptive_comparator_seed": True,
                "preemptive_comparator_seeded": True,
                "failure_reason": "",
                "artifact_complete": True,
                "route_evidence_ok": True,
            },
            {
                "od_id": "od-3",
                "variant_id": "C",
                "pipeline_mode": "voi",
                "route_id": "route-added",
                "selected_distance_km": 140.0,
                "selected_duration_s": 8600.0,
                "selected_monetary_cost": 96.0,
                "selected_emissions_kg": 165.0,
                "certificate": 0.95,
                "certified": True,
                "frontier_count": 5,
                "frontier_hypervolume": 2600.0,
                "weighted_win_osrm": True,
                "weighted_win_ors": True,
                "weighted_margin_vs_osrm": 5.0,
                "runtime_ms": 4800.0,
                "algorithm_runtime_ms": 2700.0,
                "selected_candidate_source_label": "fallback:added",
                "selected_final_route_source_label": "graph_family:added",
                "selected_from_supplemental_rescue": False,
                "selected_from_comparator_engine": True,
                "selected_from_preemptive_comparator_seed": False,
                "preemptive_comparator_seeded": True,
                "failure_reason": "",
                "artifact_complete": True,
                "route_evidence_ok": True,
            },
        ],
    )

    summary_fields = [
        "variant_id",
        "success_rate",
        "weighted_win_rate_osrm",
        "weighted_win_rate_ors",
        "mean_certificate",
        "mean_frontier_hypervolume",
        "mean_runtime_ms",
        "mean_algorithm_runtime_ms",
    ]
    before_summary = _write_csv(
        tmp_path / "before_summary.csv",
        fieldnames=summary_fields,
        rows=[
            {
                "variant_id": "A",
                "success_rate": 1.0,
                "weighted_win_rate_osrm": 1.0,
                "weighted_win_rate_ors": 1.0,
                "mean_certificate": 0.75,
                "mean_frontier_hypervolume": 1200.0,
                "mean_runtime_ms": 5000.0,
                "mean_algorithm_runtime_ms": 3000.0,
            }
        ],
    )
    after_summary = _write_csv(
        tmp_path / "after_summary.csv",
        fieldnames=summary_fields,
        rows=[
            {
                "variant_id": "A",
                "success_rate": 1.0,
                "weighted_win_rate_osrm": 1.0,
                "weighted_win_rate_ors": 1.0,
                "mean_certificate": 0.88,
                "mean_frontier_hypervolume": 1500.0,
                "mean_runtime_ms": 4300.0,
                "mean_algorithm_runtime_ms": 2500.0,
            }
        ],
    )

    out_csv = tmp_path / "comparison.csv"
    payload = compare_module.run_comparison(
        compare_module._build_parser().parse_args(
            [
                "--before-results-csv",
                str(before_results),
                "--after-results-csv",
                str(after_results),
                "--before-summary-csv",
                str(before_summary),
                "--after-summary-csv",
                str(after_summary),
                "--out-csv",
                str(out_csv),
            ]
        )
    )

    assert payload["row_count"] == 3
    assert payload["changed_row_count"] == 1
    assert payload["added_row_count"] == 1
    assert payload["removed_row_count"] == 1
    assert payload["unchanged_row_count"] == 0
    assert payload["summary_json"] == str(out_csv.with_suffix(".summary.json"))

    diff_rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8", newline="")))
    assert [row["row_key"] for row in diff_rows] == ["od-1|A", "od-2|B", "od-3|C"]

    changed_row = diff_rows[0]
    assert changed_row["status"] == "changed"
    assert changed_row["changed_fields"] == (
        "route_id;selected_distance_km;selected_duration_s;selected_monetary_cost;"
        "selected_emissions_kg;certificate;frontier_count;frontier_hypervolume;"
        "weighted_margin_vs_osrm;runtime_ms;algorithm_runtime_ms;"
        "selected_candidate_source_label;selected_final_route_source_label;"
        "preemptive_comparator_seeded;selected_from_preemptive_comparator_seed;"
        "selected_from_supplemental_rescue"
    )
    assert changed_row["delta_selected_distance_km"] == "-2"
    assert changed_row["delta_certificate"] == "0.13"
    assert changed_row["delta_frontier_hypervolume"] == "300"
    assert changed_row["delta_runtime_ms"] == "-700"

    removed_row = diff_rows[1]
    assert removed_row["status"] == "removed"
    assert removed_row["before_route_id"] == "route-stable"
    assert removed_row["after_route_id"] == ""

    added_row = diff_rows[2]
    assert added_row["status"] == "added"
    assert added_row["before_route_id"] == ""
    assert added_row["after_route_id"] == "route-added"

    summary_payload = json.loads(out_csv.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary_payload["variant_count"] == 1
    assert summary_payload["variant_diffs"]["A"]["status"] == "changed"
    assert summary_payload["variant_diffs"]["A"]["changed_fields"] == [
        "mean_certificate",
        "mean_frontier_hypervolume",
        "mean_runtime_ms",
        "mean_algorithm_runtime_ms",
    ]
    assert summary_payload["variant_diffs"]["A"]["delta"]["mean_certificate"] == "0.13"
    retained = summary_payload["retained_success_summary"]
    assert retained["before_success_row_count"] == 2
    assert retained["after_success_row_count"] == 2
    assert retained["retained_success_row_count"] == 1
    assert retained["regressed_success_row_count"] == 0
    assert retained["newly_successful_row_count"] == 1
    assert retained["removed_success_row_count"] == 1
    assert retained["retained_success_row_keys"] == ["od-1|A"]
    assert retained["newly_successful_row_keys"] == ["od-3|C"]
    assert retained["removed_success_row_keys"] == ["od-2|B"]
    assert retained["variant_summaries"]["A"]["retained_success_row_count"] == 1
    assert retained["variant_summaries"]["B"]["removed_success_row_count"] == 1
    assert retained["variant_summaries"]["C"]["newly_successful_row_count"] == 1
    assert payload["retained_success_summary"]["retained_success_row_count"] == 1


def test_compare_thesis_runs_uses_coordinate_fallback_key_when_od_id_missing(
    tmp_path: Path,
) -> None:
    result_fields = [
        "origin_lat",
        "origin_lon",
        "destination_lat",
        "destination_lon",
        "variant_id",
        "selected_distance_km",
    ]
    before_results = _write_csv(
        tmp_path / "before_results.csv",
        fieldnames=result_fields,
        rows=[
            {
                "origin_lat": 51.5,
                "origin_lon": -0.1,
                "destination_lat": 52.5,
                "destination_lon": -1.1,
                "variant_id": "A",
                "selected_distance_km": 100.0,
            }
        ],
    )
    after_results = _write_csv(
        tmp_path / "after_results.csv",
        fieldnames=result_fields,
        rows=[
            {
                "origin_lat": 51.5,
                "origin_lon": -0.1,
                "destination_lat": 52.5,
                "destination_lon": -1.1,
                "variant_id": "A",
                "selected_distance_km": 98.0,
            }
        ],
    )
    out_csv = tmp_path / "comparison.csv"

    payload = compare_module.run_comparison(
        compare_module._build_parser().parse_args(
            [
                "--before-results-csv",
                str(before_results),
                "--after-results-csv",
                str(after_results),
                "--out-csv",
                str(out_csv),
            ]
        )
    )

    assert payload["row_count"] == 1
    row = next(csv.DictReader(out_csv.open("r", encoding="utf-8", newline="")))
    assert row["row_key"] == "51.5,-0.1->52.5,-1.1|A"
    assert row["od_display_key"] == "51.5,-0.1->52.5,-1.1"
    assert row["status"] == "changed"
    assert row["delta_selected_distance_km"] == "-2"
    assert payload["summary_json"] is None
