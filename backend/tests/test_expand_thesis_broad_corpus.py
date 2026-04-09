from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.expand_thesis_broad_corpus as expand_module


pytestmark = pytest.mark.thesis_results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _seed_source_corpus(path: Path) -> Path:
    rows = [
        {
            "od_id": "cardiff_london",
            "origin_lat": 51.4816,
            "origin_lon": -3.1791,
            "destination_lat": 51.5074,
            "destination_lon": -0.1278,
            "origin_region_bucket": "wales",
            "destination_region_bucket": "south",
            "od_ambiguity_index": 0.41,
            "search_budget": 5,
            "evidence_budget": 2,
            "world_count": 72,
            "max_alternatives": 9,
        },
        {
            "od_id": "bristol_leeds",
            "origin_lat": 51.4545,
            "origin_lon": -2.5879,
            "destination_lat": 53.8008,
            "destination_lon": -1.5491,
            "origin_region_bucket": "south",
            "destination_region_bucket": "yorkshire",
            "od_ambiguity_index": 0.23,
            "search_budget": 4,
            "evidence_budget": 2,
            "world_count": 64,
            "max_alternatives": 8,
        },
        {
            "od_id": "manchester_birmingham",
            "origin_lat": 53.4808,
            "origin_lon": -2.2426,
            "destination_lat": 52.4862,
            "destination_lon": -1.8904,
            "origin_region_bucket": "north",
            "destination_region_bucket": "midlands",
            "od_ambiguity_index": 0.35,
            "search_budget": 4,
            "evidence_budget": 2,
            "world_count": 60,
            "max_alternatives": 8,
        },
        {
            "od_id": "newcastle_liverpool",
            "origin_lat": 54.9783,
            "origin_lon": -1.6178,
            "destination_lat": 53.4084,
            "destination_lon": -2.9916,
            "origin_region_bucket": "northeast",
            "destination_region_bucket": "northwest",
            "od_ambiguity_index": 0.52,
            "search_budget": 6,
            "evidence_budget": 3,
            "world_count": 80,
            "max_alternatives": 10,
        },
    ]
    return _write_csv(path, rows)


def _od_ids(path: Path) -> list[str]:
    return [str(row["od_id"]) for row in csv.DictReader(path.open("r", encoding="utf-8", newline=""))]


def test_build_expanded_corpus_supports_1200_rows_from_checked_in_broad_seed(tmp_path: Path) -> None:
    input_csv = Path(__file__).resolve().parents[1] / "data" / "eval" / "uk_od_corpus_thesis_broad.csv"
    output_csv = tmp_path / "expanded_1200.csv"
    summary_json = tmp_path / "expanded_1200.summary.json"
    shard_dir = tmp_path / "shards"
    shard_manifest_json = tmp_path / "expanded_1200.shards.json"

    summary = expand_module.build_expanded_corpus(
        input_csv=input_csv,
        output_csv=output_csv,
        summary_json=summary_json,
        target_count=1200,
        shard_count=10,
        shard_output_dir=shard_dir,
        shard_manifest_json=shard_manifest_json,
    )

    manifest = json.loads(shard_manifest_json.read_text(encoding="utf-8"))
    od_ids = _od_ids(output_csv)

    assert summary["selected_count"] == 1200
    assert summary["candidate_count"] >= 1200
    assert summary["shard_count"] == 10
    assert summary["shard_row_counts"] == [120] * 10
    assert manifest["total_row_count"] == 1200
    assert [entry["row_count"] for entry in manifest["shards"]] == [120] * 10
    assert len(od_ids) == 1200
    assert len(set(od_ids)) == 1200


def test_build_expanded_corpus_writes_deterministic_shards_and_manifest(tmp_path: Path) -> None:
    input_csv = _seed_source_corpus(tmp_path / "source.csv")

    run_one_output = tmp_path / "run_one" / "expanded.csv"
    run_one_summary = tmp_path / "run_one" / "expanded.summary.json"
    run_one_shard_dir = tmp_path / "run_one" / "shards"
    run_one_shard_manifest = tmp_path / "run_one" / "expanded.shards.json"
    run_two_output = tmp_path / "run_two" / "expanded.csv"
    run_two_summary = tmp_path / "run_two" / "expanded.summary.json"
    run_two_shard_dir = tmp_path / "run_two" / "shards"
    run_two_shard_manifest = tmp_path / "run_two" / "expanded.shards.json"

    summary_one = expand_module.build_expanded_corpus(
        input_csv=input_csv,
        output_csv=run_one_output,
        summary_json=run_one_summary,
        target_count=6,
        shard_count=3,
        shard_output_dir=run_one_shard_dir,
        shard_manifest_json=run_one_shard_manifest,
    )
    summary_two = expand_module.build_expanded_corpus(
        input_csv=input_csv,
        output_csv=run_two_output,
        summary_json=run_two_summary,
        target_count=6,
        shard_count=3,
        shard_output_dir=run_two_shard_dir,
        shard_manifest_json=run_two_shard_manifest,
    )

    assert summary_one["selected_count"] == 6
    assert summary_one["shard_count"] == 3
    assert summary_one["shard_row_counts"] == [2, 2, 2]
    assert summary_two["shard_row_counts"] == [2, 2, 2]

    manifest_one = json.loads(run_one_shard_manifest.read_text(encoding="utf-8"))
    manifest_two = json.loads(run_two_shard_manifest.read_text(encoding="utf-8"))

    assert manifest_one["composition_type"] == "corpus_shards"
    assert manifest_one["total_row_count"] == 6
    assert len(manifest_one["shards"]) == 3
    assert [entry["row_count"] for entry in manifest_one["shards"]] == [2, 2, 2]

    assert _od_ids(run_one_output) == _od_ids(run_two_output)
    assert [entry["od_ids"] for entry in manifest_one["shards"]] == [entry["od_ids"] for entry in manifest_two["shards"]]

    expected_master_order = _od_ids(run_one_output)
    actual_master_order = [od_id for entry in manifest_one["shards"] for od_id in entry["od_ids"]]
    assert actual_master_order == expected_master_order
    for entry in manifest_one["shards"]:
        assert _od_ids(Path(entry["path"])) == entry["od_ids"]


def test_build_expanded_corpus_rejects_shard_count_larger_than_row_count(tmp_path: Path) -> None:
    input_csv = _seed_source_corpus(tmp_path / "source.csv")

    with pytest.raises(ValueError, match="shard_count cannot exceed"):
        expand_module.build_expanded_corpus(
            input_csv=input_csv,
            output_csv=tmp_path / "expanded.csv",
            summary_json=tmp_path / "expanded.summary.json",
            target_count=4,
            shard_count=5,
            shard_output_dir=tmp_path / "shards",
            shard_manifest_json=tmp_path / "expanded.shards.json",
        )
