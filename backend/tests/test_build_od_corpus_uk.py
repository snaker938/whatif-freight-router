from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.build_od_corpus_uk as corpus_module

pytestmark = [pytest.mark.thesis, pytest.mark.thesis_modules]


def _feasible(**kwargs):  # noqa: ARG001
    return {
        "ok": True,
        "message": "ok",
        "origin_node_id": "n1",
        "destination_node_id": "n2",
        "origin_nearest_distance_m": 12.0,
        "destination_nearest_distance_m": 18.0,
    }


def _candidate_probe(**kwargs):  # noqa: ARG001
    return [
        {
            "route_id": "probe-a",
            "distance": 1000.0,
            "duration": 120.0,
            "cost": 20.0,
            "emissions": 6.0,
            "corridor_family": "north",
            "has_tolls": False,
        },
        {
            "route_id": "probe-b",
            "distance": 1040.0,
            "duration": 122.0,
            "cost": 19.0,
            "emissions": 6.3,
            "corridor_family": "west",
            "has_tolls": True,
        },
    ], corpus_module.GraphCandidateDiagnostics(
        explored_states=19,
        generated_paths=3,
        emitted_paths=2,
        candidate_budget=4,
        effective_max_hops=40,
        effective_state_budget=8000,
    )


def test_candidate_probe_payload_emits_first_class_ambiguity_prior_metadata() -> None:
    payload = corpus_module._candidate_probe_payload(
        origin={"lat": 51.5, "lon": -2.6},
        destination={"lat": 53.48, "lon": -1.18},
        feasibility_result={"origin_node_id": "n1", "destination_node_id": "n2"},
        candidate_probe_fn=_candidate_probe,
        max_paths=4,
    )

    assert payload["od_ambiguity_index"] > 0.0
    assert payload["od_ambiguity_confidence"] > 0.0
    assert payload["od_ambiguity_source_count"] == 1
    source_mix = json.loads(payload["od_ambiguity_source_mix"])
    assert "routing_graph_probe" in source_mix
    assert "engine_augmented_probe" not in source_mix
    assert payload["od_ambiguity_source_mix_count"] == 1
    assert payload["od_ambiguity_source_support"] not in (None, "")
    support_map = json.loads(payload["od_ambiguity_source_support"])
    assert "routing_graph_probe" in support_map
    assert "engine_augmented_probe" not in support_map
    assert payload["od_ambiguity_source_support_strength"] > 0.0
    assert payload["od_ambiguity_source_entropy"] == 0.0
    assert 0.0 < payload["od_ambiguity_support_ratio"] <= 1.0
    assert payload["od_ambiguity_prior_strength"] == payload["od_ambiguity_index"]
    assert payload["od_ambiguity_family_density"] == 1.0
    assert payload["od_ambiguity_margin_pressure"] > 0.0
    assert payload["od_ambiguity_spread_pressure"] > 0.0
    assert payload["candidate_probe_path_sufficiency"] > 0.0
    assert payload["candidate_probe_top2_gap_pressure"] > 0.0
    assert payload["ambiguity_budget_prior"] >= payload["od_ambiguity_index"]
    ambiguity_sources = set(str(payload["ambiguity_prior_source"]).split(","))
    assert ambiguity_sources == {"routing_graph_probe"}
    assert payload["ambiguity_prior_sample_count"] >= 1
    assert payload["ambiguity_prior_support_count"] >= 1
    assert payload["ambiguity_prior_nonzero"] is True


def test_write_csv_persists_ambiguity_prior_columns(tmp_path: Path) -> None:
    rows = [
        {
            "od_id": "od-000001",
            "sample_index": 1,
            "origin_lat": 51.5,
            "origin_lon": -2.6,
            "destination_lat": 53.48,
            "destination_lon": -1.18,
            "straight_line_km": 10.0,
            "distance_bin": "0-30 km",
            "bin_index": 0,
            "origin_region_bucket": "south",
            "destination_region_bucket": "north",
            "corridor_bucket": "south_to_north|north_south",
            "acceptance_mode": "graph_candidates",
            "accepted": True,
            "reason_code": "ok",
            "route_graph_message": "ok",
            "origin_node_id": "n1",
            "destination_node_id": "n2",
            "origin_nearest_distance_m": 12.0,
            "destination_nearest_distance_m": 18.0,
            "candidate_probe_accepted": True,
            "candidate_probe_reason_code": "ok",
            "candidate_probe_message": "candidate_probe_ok",
            "candidate_probe_emitted_paths": 2,
            "candidate_probe_generated_paths": 3,
            "candidate_probe_explored_states": 19,
            "candidate_probe_candidate_budget": 4,
            "candidate_probe_effective_max_hops": 40,
            "candidate_probe_effective_state_budget": 8000,
            "candidate_probe_path_count": 2,
            "candidate_probe_corridor_family_count": 2,
            "candidate_probe_distance_spread_km": 0.04,
            "candidate_probe_duration_spread_s": 2.0,
            "candidate_probe_cost_spread": 1.0,
            "candidate_probe_emissions_spread_kg": 0.3,
            "candidate_probe_objective_spread": 0.2,
            "candidate_probe_nominal_margin": 0.1,
            "candidate_probe_toll_disagreement_rate": 1.0,
            "candidate_probe_family_diversity": 1.0,
            "candidate_probe_engine_disagreement_prior": 0.4,
            "hard_case_prior": 0.5,
            "ambiguity_index": 0.55,
            "od_ambiguity_index": 0.55,
            "od_ambiguity_confidence": 0.73,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix": '{"routing_graph_probe":1}',
            "od_ambiguity_source_mix_count": 1,
            "od_ambiguity_source_support": '{"routing_graph_probe":0.72}',
            "od_ambiguity_source_support_strength": 0.72,
            "od_ambiguity_source_entropy": 0.0,
            "od_ambiguity_support_ratio": 0.666667,
            "od_ambiguity_prior_strength": 0.55,
            "od_ambiguity_family_density": 1.0,
            "od_ambiguity_margin_pressure": 0.9,
            "od_ambiguity_spread_pressure": 0.2,
            "od_ambiguity_toll_instability": 1.0,
            "ambiguity_prior_source": "routing_graph_probe",
            "ambiguity_prior_sample_count": 3,
            "ambiguity_prior_support_count": 2,
            "ambiguity_prior_nonzero": True,
            "corpus_kind": "representative",
            "selection_rank": 1,
            "selection_score": 0.8,
        }
    ]
    out_csv = tmp_path / "corpus.csv"

    corpus_module._write_csv(out_csv, rows)

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        written = next(csv.DictReader(handle))
    assert written["od_ambiguity_confidence"] == "0.73"
    assert written["od_ambiguity_source_count"] == "1"
    assert written["od_ambiguity_source_mix"] == '{"routing_graph_probe":1}'
    assert written["od_ambiguity_source_mix_count"] == "1"
    assert written["od_ambiguity_source_support"] == '{"routing_graph_probe":0.72}'
    assert written["od_ambiguity_source_support_strength"] == "0.72"
    assert written["od_ambiguity_source_entropy"] == "0.0"
    assert written["od_ambiguity_support_ratio"] == "0.666667"
    assert written["od_ambiguity_prior_strength"] == "0.55"
    assert written["ambiguity_prior_source"] == "routing_graph_probe"


def test_bootstrap_selected_rows_adds_historical_source_support_and_entropy() -> None:
    rows = [
        {
            "od_id": "od-1",
            "distance_bin": "30-100 km",
            "corridor_bucket": "a",
            "ambiguity_index": 0.61,
            "od_ambiguity_index": 0.61,
            "od_ambiguity_confidence": 0.81,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":4,"routing_graph_probe":2}',
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_source_support": '{"engine_augmented_probe":0.75,"routing_graph_probe":0.82}',
            "od_ambiguity_source_support_strength": 0.785,
            "od_ambiguity_source_entropy": 0.63,
            "od_ambiguity_support_ratio": 0.79,
            "od_ambiguity_prior_strength": 0.61,
            "candidate_probe_path_count": 3,
            "candidate_probe_corridor_family_count": 2,
            "candidate_probe_objective_spread": 0.31,
            "candidate_probe_nominal_margin": 0.08,
            "candidate_probe_engine_disagreement_prior": 0.66,
            "hard_case_prior": 0.72,
            "ambiguity_prior_source": "routing_graph_probe,engine_augmented_probe",
            "ambiguity_prior_sample_count": 2,
            "ambiguity_prior_support_count": 2,
        },
        {
            "od_id": "od-2",
            "distance_bin": "30-100 km",
            "corridor_bucket": "b",
            "ambiguity_index": 0.58,
            "od_ambiguity_index": 0.58,
            "od_ambiguity_confidence": 0.74,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":3,"routing_graph_probe":2}',
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_source_support": '{"engine_augmented_probe":0.7,"routing_graph_probe":0.81}',
            "od_ambiguity_source_support_strength": 0.755,
            "od_ambiguity_source_entropy": 0.61,
            "od_ambiguity_support_ratio": 0.74,
            "od_ambiguity_prior_strength": 0.58,
            "candidate_probe_path_count": 4,
            "candidate_probe_corridor_family_count": 3,
            "candidate_probe_objective_spread": 0.28,
            "candidate_probe_nominal_margin": 0.11,
            "candidate_probe_engine_disagreement_prior": 0.61,
            "hard_case_prior": 0.69,
            "ambiguity_prior_source": "routing_graph_probe,engine_augmented_probe",
            "ambiguity_prior_sample_count": 2,
            "ambiguity_prior_support_count": 2,
        },
    ]

    bootstrapped = corpus_module._bootstrap_selected_rows(rows)

    assert len(bootstrapped) == 2
    for row in bootstrapped:
        source_tokens = row["ambiguity_prior_source"].split(",")
        assert "historical_results_bootstrap" in source_tokens
        assert row["od_ambiguity_source_count"] >= 3
        support_map = json.loads(row["od_ambiguity_source_support"])
        assert "historical_results_bootstrap" in support_map
        assert row["od_ambiguity_source_support_strength"] >= 0.5
        assert row["od_ambiguity_source_entropy"] > 0.0


def test_ambiguous_corpus_selection_prefers_stronger_budget_prior() -> None:
    rows = [
        {
            "od_id": "easy-1",
            "distance_bin": "30-100 km",
            "corridor_bucket": "a",
            "ambiguity_index": 0.62,
            "od_ambiguity_index": 0.62,
            "ambiguity_budget_prior": 0.62,
            "hard_case_prior": 0.50,
            "candidate_probe_engine_disagreement_prior": 0.48,
            "od_ambiguity_support_ratio": 0.35,
            "od_ambiguity_source_entropy": 0.0,
            "candidate_probe_path_count": 2,
        },
        {
            "od_id": "hard-1",
            "distance_bin": "30-100 km",
            "corridor_bucket": "b",
            "ambiguity_index": 0.58,
            "od_ambiguity_index": 0.58,
            "ambiguity_budget_prior": 0.84,
            "hard_case_prior": 0.82,
            "candidate_probe_engine_disagreement_prior": 0.76,
            "od_ambiguity_support_ratio": 0.78,
            "od_ambiguity_source_entropy": 0.63,
            "candidate_probe_path_count": 5,
        },
    ]

    selected = corpus_module._select_rows_for_corpus(rows, count=1, corpus_kind="ambiguous")

    assert len(selected) == 1
    assert selected[0]["od_id"] == "hard-1"
    assert selected[0]["selection_score"] > 0.75


def test_build_dual_od_corpora_emits_disjoint_broad_rows(monkeypatch) -> None:
    pool_rows = [
        {
            "od_id": "amb-1",
            "distance_bin": "30-100 km",
            "corridor_bucket": "a",
            "ambiguity_index": 0.91,
            "od_ambiguity_index": 0.91,
            "ambiguity_budget_prior": 0.91,
            "hard_case_prior": 0.88,
            "candidate_probe_engine_disagreement_prior": 0.85,
            "od_ambiguity_support_ratio": 0.82,
            "od_ambiguity_source_entropy": 0.66,
            "candidate_probe_path_count": 5,
        },
        {
            "od_id": "rep-1",
            "distance_bin": "30-100 km",
            "corridor_bucket": "b",
            "ambiguity_index": 0.41,
            "od_ambiguity_index": 0.41,
            "ambiguity_budget_prior": 0.41,
            "hard_case_prior": 0.33,
            "candidate_probe_engine_disagreement_prior": 0.30,
            "od_ambiguity_support_ratio": 0.35,
            "od_ambiguity_source_entropy": 0.05,
            "candidate_probe_path_count": 2,
        },
        {
            "od_id": "amb-2",
            "distance_bin": "30-100 km",
            "corridor_bucket": "c",
            "ambiguity_index": 0.86,
            "od_ambiguity_index": 0.86,
            "ambiguity_budget_prior": 0.86,
            "hard_case_prior": 0.81,
            "candidate_probe_engine_disagreement_prior": 0.78,
            "od_ambiguity_support_ratio": 0.77,
            "od_ambiguity_source_entropy": 0.61,
            "candidate_probe_path_count": 4,
        },
        {
            "od_id": "rep-2",
            "distance_bin": "30-100 km",
            "corridor_bucket": "d",
            "ambiguity_index": 0.47,
            "od_ambiguity_index": 0.47,
            "ambiguity_budget_prior": 0.47,
            "hard_case_prior": 0.38,
            "candidate_probe_engine_disagreement_prior": 0.36,
            "od_ambiguity_support_ratio": 0.29,
            "od_ambiguity_source_entropy": 0.08,
            "candidate_probe_path_count": 2,
        },
    ]

    def _fake_build_od_corpus(**kwargs):  # noqa: ARG001
        return {
            "schema_version": "2.0.0",
            "seed": 7,
            "acceptance_mode": "graph_candidates",
            "corpus_hash": "pool-hash",
            "accepted_count": len(pool_rows),
            "distance_bins": [],
            "rows": [dict(row) for row in pool_rows],
        }

    monkeypatch.setattr(corpus_module, "build_od_corpus", _fake_build_od_corpus)

    bundle = corpus_module.build_dual_od_corpora(
        seed=7,
        representative_count=2,
        ambiguous_count=2,
        bbox=corpus_module.UKBBox(south=50.0, north=56.0, west=-6.0, east=2.0),
        max_attempts=100,
    )

    representative_ids = {row["od_id"] for row in bundle["representative"]["rows"]}
    ambiguous_ids = {row["od_id"] for row in bundle["ambiguous"]["rows"]}
    broad_ids = [row["od_id"] for row in bundle["broad"]["rows"]]

    assert representative_ids == {"rep-1", "rep-2"}
    assert ambiguous_ids == {"amb-1", "amb-2"}
    assert representative_ids.isdisjoint(ambiguous_ids)
    assert len(broad_ids) == 4
    assert len(set(broad_ids)) == 4
    assert bundle["broad"]["component_counts"] == {"representative": 2, "ambiguous": 2}
    assert bundle["broad"]["component_overlap_count"] == 0
    assert bundle["broad"]["distinct_od_id_count"] == 4


def test_main_dual_builder_writes_broad_outputs(tmp_path: Path, monkeypatch) -> None:
    bundle = {
        "source_pool": {"corpus_hash": "pool-hash"},
        "representative": {
            "rows": [{"od_id": "rep-1", "corpus_kind": "representative"}],
        },
        "ambiguous": {
            "rows": [{"od_id": "amb-1", "corpus_kind": "ambiguous"}],
        },
        "broad": {
            "rows": [
                {"od_id": "rep-1", "corpus_kind": "representative"},
                {"od_id": "amb-1", "corpus_kind": "ambiguous"},
            ],
        },
    }

    monkeypatch.setattr(corpus_module, "build_dual_od_corpora", lambda **kwargs: bundle)  # noqa: ARG005

    exit_code = corpus_module.main(
        [
            "--output-dir",
            str(tmp_path),
            "--pair-count",
            "1",
            "--ambiguous-pair-count",
            "1",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "uk_od_corpus_representative.csv").exists()
    assert (tmp_path / "uk_od_corpus_ambiguous.csv").exists()
    assert (tmp_path / "uk_od_corpus_thesis_broad_generated.csv").exists()
    broad_summary = json.loads((tmp_path / "uk_od_corpus_thesis_broad_generated.summary.json").read_text(encoding="utf-8"))
    assert len(broad_summary["rows"]) == 2


def test_harder_eval_corpora_are_curated_from_existing_rows_with_family_caps() -> None:
    backend_dir = Path(__file__).resolve().parents[1]

    def _load_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _canon_od_family(od_id: str) -> str:
        parts = [part for part in od_id.split("_") if not part.startswith("alt")]
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[-1]}"
        return od_id

    hard_mixed_path = backend_dir / "data" / "eval" / "uk_od_corpus_hard_mixed_24.csv"
    longcorr_hard_path = backend_dir / "data" / "eval" / "uk_od_corpus_longcorr_hard_32.csv"

    hard_mixed_rows = _load_rows(hard_mixed_path)
    longcorr_hard_rows = _load_rows(longcorr_hard_path)

    hard_mixed_sources = (
        _load_rows(backend_dir / "data" / "eval" / "uk_od_corpus_dominance_cluster_8.csv")
        + _load_rows(backend_dir / "data" / "eval" / "uk_od_corpus_thesis_broad_expanded_1200.csv")
        + _load_rows(backend_dir / "out" / "thesis_corpus" / "uk_od_corpus_thesis_broad_longcorr_250km.csv")
    )
    hard_mixed_source_ids = {row["od_id"] for row in hard_mixed_sources}
    longcorr_source_ids = {
        row["od_id"]
        for row in _load_rows(backend_dir / "out" / "thesis_corpus" / "uk_od_corpus_thesis_broad_longcorr_250km.csv")
    }

    assert len(hard_mixed_rows) == 24
    assert {row["corpus_group"] for row in hard_mixed_rows} == {"ambiguity", "representative"}
    assert sum(1 for row in hard_mixed_rows if row["trip_length_bin"] in {"250-500 km", "500+ km"}) >= 12
    assert max(
        sum(1 for inner in hard_mixed_rows if _canon_od_family(inner["od_id"]) == family)
        for family in {_canon_od_family(row["od_id"]) for row in hard_mixed_rows}
    ) <= 4
    assert {row["od_id"] for row in hard_mixed_rows}.issubset(hard_mixed_source_ids)

    assert len(longcorr_hard_rows) == 32
    assert {row["corpus_group"] for row in longcorr_hard_rows} == {"ambiguity", "representative"}
    assert all(row["trip_length_bin"] in {"250-500 km", "500+ km"} for row in longcorr_hard_rows)
    assert sum(1 for row in longcorr_hard_rows if float(row["od_ambiguity_index"]) >= 0.30) >= 16
    assert max(
        sum(1 for inner in longcorr_hard_rows if _canon_od_family(inner["od_id"]) == family)
        for family in {_canon_od_family(row["od_id"]) for row in longcorr_hard_rows}
    ) <= 4
    assert {row["od_id"] for row in longcorr_hard_rows}.issubset(longcorr_source_ids)
