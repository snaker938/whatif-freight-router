from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.enrich_od_corpus_with_ambiguity as ambiguity_module

pytestmark = [pytest.mark.thesis, pytest.mark.thesis_modules]


def test_historical_bootstrap_index_derives_ambiguity_when_explicit_field_missing(tmp_path) -> None:
    results_csv = tmp_path / "thesis_results.csv"
    results_csv.write_text(
        "\n".join(
            [
                "od_id,variant_id,observed_ambiguity_index,frontier_count,near_tie_mass,nominal_winner_margin,voi_action_count,candidate_count_raw",
                "cardiff_bristol,V0,,1,0.0,1.0,0,1",
                "cardiff_bristol,A,,3,0.5,0.010454,0,10",
                "cardiff_bristol,B,,3,0.5,0.010454,0,10",
                "cardiff_bristol,C,,3,0.5,0.010454,2,10",
            ]
        ),
        encoding="utf-8",
    )

    index = ambiguity_module._historical_bootstrap_index([results_csv])

    payload = next(iter(index["by_identity"].values()))
    assert payload["ambiguity_index"] == pytest.approx(0.38554, rel=0.0, abs=1e-6)
    assert payload["candidate_probe_path_count"] == 10
    assert payload["candidate_probe_corridor_family_count"] == 3
    assert payload["candidate_probe_nominal_margin"] == pytest.approx(0.25784, rel=0.0, abs=1e-6)
    assert payload["candidate_probe_objective_spread"] == pytest.approx(0.375, rel=0.0, abs=1e-6)
    assert payload["candidate_probe_engine_disagreement_prior"] == pytest.approx(0.615722, rel=0.0, abs=1e-6)
    assert payload["hard_case_prior"] == pytest.approx(0.612119, rel=0.0, abs=1e-6)
    assert payload["ambiguity_prior_source"] == "historical_results_bootstrap"
    assert payload["ambiguity_prior_sample_count"] == 4
    assert payload["ambiguity_prior_support_count"] == 4
    assert index["by_od_id_unique"]["cardiff_bristol"]["ambiguity_index"] == payload["ambiguity_index"]


def test_enrich_rows_can_apply_bootstrap_prior_without_graph_probe(monkeypatch) -> None:
    def _unexpected_graph_probe(**kwargs):  # noqa: ARG001
        raise AssertionError("route graph probe should not run when use_graph_probe=False")

    monkeypatch.setattr(ambiguity_module, "route_graph_od_feasibility", _unexpected_graph_probe)

    rows = [
        {
            "od_id": "od-1",
            "origin_lat": 51.5,
            "origin_lon": -2.6,
            "destination_lat": 51.48,
            "destination_lon": -3.18,
            "distance_bin": "30-100 km",
        }
    ]
    bootstrap_index = {
        "by_identity": {
            ambiguity_module._row_identity_key(rows[0]): {
                "ambiguity_index": 0.61,
                "candidate_probe_path_count": 9,
                "candidate_probe_corridor_family_count": 3,
                "candidate_probe_objective_spread": 0.42,
                "candidate_probe_nominal_margin": 0.18,
                "candidate_probe_engine_disagreement_prior": 0.44,
                "hard_case_prior": 0.52,
                "ambiguity_prior_source": "historical_results_bootstrap",
                "ambiguity_prior_sample_count": 4,
                "ambiguity_prior_support_count": 4,
            }
        },
        "by_od_id_unique": {},
    }

    enriched = ambiguity_module.enrich_rows(
        rows,
        probe_max_paths=6,
        bootstrap_index=bootstrap_index,
        use_graph_probe=False,
    )

    assert len(enriched) == 1
    row = enriched[0]
    assert row["ambiguity_index"] == 0.61
    assert row["candidate_probe_path_count"] == 9
    assert row["candidate_probe_corridor_family_count"] == 3
    assert row["candidate_probe_objective_spread"] == 0.42
    assert row["candidate_probe_nominal_margin"] == 0.18
    assert row["candidate_probe_engine_disagreement_prior"] == 0.44
    assert row["hard_case_prior"] == 0.52
    assert row["ambiguity_prior_source"] == "routing_graph_probe,historical_results_bootstrap"
    assert row["ambiguity_prior_sample_count"] == 4
    assert row["od_ambiguity_index"] == 0.61
    assert row["od_ambiguity_confidence"] > 0.8
    assert row["od_ambiguity_source_count"] == 2
    assert '"historical_results_bootstrap":4' in row["od_ambiguity_source_mix"]
    assert '"routing_graph_probe":' in row["od_ambiguity_source_mix"]
    assert row["od_ambiguity_source_mix_count"] == 2
    assert '"routing_graph_probe"' in row["od_ambiguity_source_support"]
    assert '"historical_results_bootstrap"' in row["od_ambiguity_source_support"]
    assert row["od_ambiguity_source_support_strength"] > 0.7
    assert row["od_ambiguity_source_entropy"] > 0.0
    assert row["od_ambiguity_support_ratio"] > 0.6
    assert row["od_ambiguity_prior_strength"] == 0.61
    assert row["od_ambiguity_family_density"] == pytest.approx(1.0 / 3.0, rel=0.0, abs=1e-6)
    assert row["straight_line_km"] > 0.0
    assert row["corridor_bucket"]


def test_apply_ambiguity_derived_fields_uses_support_gated_budget_prior() -> None:
    weak_row = {
        "od_ambiguity_index": 0.18,
        "candidate_probe_engine_disagreement_prior": 0.28,
        "hard_case_prior": 0.24,
        "od_ambiguity_support_ratio": 0.34,
        "od_ambiguity_source_entropy": 0.31,
        "od_ambiguity_source_count": 2,
        "od_ambiguity_source_mix_count": 2,
        "od_ambiguity_prior_strength": 0.24,
        "od_ambiguity_family_density": 0.25,
        "od_ambiguity_margin_pressure": 0.28,
        "od_ambiguity_spread_pressure": 0.22,
        "od_ambiguity_toll_instability": 0.11,
    }
    ambiguity_module._apply_ambiguity_derived_fields(weak_row)
    assert weak_row["ambiguity_budget_prior"] == pytest.approx(0.18, rel=0.0, abs=1e-6)
    assert weak_row["ambiguity_budget_prior_gap"] == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert weak_row["budget_prior_exceeds_raw"] is False

    rich_row = {
        "od_ambiguity_index": 0.18,
        "candidate_probe_engine_disagreement_prior": 0.48,
        "hard_case_prior": 0.35,
        "candidate_probe_path_count": 9,
        "candidate_probe_corridor_family_count": 3,
        "ambiguity_prior_sample_count": 4,
        "ambiguity_prior_support_count": 4,
        "od_ambiguity_confidence": 0.9,
        "od_ambiguity_source_count": 3,
        "od_ambiguity_support_ratio": 0.79,
        "od_ambiguity_source_entropy": 0.63,
        "od_ambiguity_source_mix_count": 2,
        "od_ambiguity_prior_strength": 0.72,
        "od_ambiguity_family_density": 0.66,
        "od_ambiguity_margin_pressure": 0.92,
        "od_ambiguity_spread_pressure": 0.31,
        "od_ambiguity_toll_instability": 0.12,
    }
    ambiguity_module._apply_ambiguity_derived_fields(rich_row)
    assert rich_row["ambiguity_budget_prior"] > 0.18
    assert rich_row["ambiguity_budget_prior"] <= 0.40
    assert rich_row["ambiguity_budget_prior_gap"] == pytest.approx(
        rich_row["ambiguity_budget_prior"] - 0.18,
        rel=0.0,
        abs=1e-6,
    )
    assert rich_row["budget_prior_exceeds_raw"] is True


def test_apply_ambiguity_derived_fields_accepts_candidate_probe_ambiguity_index_as_raw_prior() -> None:
    row = {
        "candidate_probe_ambiguity_index": 0.21,
        "candidate_probe_engine_disagreement_prior": 0.29,
        "hard_case_prior": 0.24,
        "od_ambiguity_support_ratio": 0.28,
        "od_ambiguity_source_entropy": 0.20,
        "od_ambiguity_source_count": 1,
        "od_ambiguity_source_mix_count": 1,
        "od_ambiguity_prior_strength": 0.21,
        "od_ambiguity_family_density": 0.18,
        "od_ambiguity_margin_pressure": 0.16,
        "od_ambiguity_spread_pressure": 0.11,
        "od_ambiguity_toll_instability": 0.04,
    }

    ambiguity_module._apply_ambiguity_derived_fields(row)

    assert row["od_ambiguity_prior_strength"] == pytest.approx(0.21, rel=0.0, abs=1e-6)
    assert row["ambiguity_budget_prior"] == pytest.approx(0.21, rel=0.0, abs=1e-6)
    assert row["ambiguity_budget_prior_gap"] == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert row["budget_prior_exceeds_raw"] is False


def test_bootstrap_index_does_not_bleed_across_distinct_profiles_with_same_od_id(tmp_path) -> None:
    results_csv = tmp_path / "thesis_results.csv"
    results_csv.write_text(
        "\n".join(
            [
                "od_id,profile_id,corpus_group,origin_lat,origin_lon,destination_lat,destination_lon,departure_time_utc,variant_id,observed_ambiguity_index,frontier_count,near_tie_mass,nominal_winner_margin,voi_action_count,candidate_count_raw",
                "shared_od,profile_a,representative,51.5,-2.6,51.48,-3.18,2026-03-21T08:00:00Z,A,0.2,2,0.2,0.8,0,4",
                "shared_od,profile_b,ambiguity,51.5,-2.6,51.48,-3.18,2026-03-21T09:00:00Z,C,0.8,4,0.6,0.1,2,9",
            ]
        ),
        encoding="utf-8",
    )
    index = ambiguity_module._historical_bootstrap_index([results_csv])

    row_a = {
        "od_id": "shared_od",
        "profile_id": "profile_a",
        "corpus_group": "representative",
        "origin_lat": 51.5,
        "origin_lon": -2.6,
        "destination_lat": 51.48,
        "destination_lon": -3.18,
        "departure_time_utc": "2026-03-21T08:00:00Z",
    }
    row_b = dict(row_a)
    row_b.update({"profile_id": "profile_b", "corpus_group": "ambiguity", "departure_time_utc": "2026-03-21T09:00:00Z"})

    enriched_a = ambiguity_module._apply_bootstrap_prior(row_a, bootstrap_index=index)
    enriched_b = ambiguity_module._apply_bootstrap_prior(row_b, bootstrap_index=index)

    assert enriched_a["ambiguity_index"] == 0.2
    assert enriched_b["ambiguity_index"] == 0.8
    assert "shared_od" not in index["by_od_id_unique"]


def test_enrich_rows_backfills_cheap_engine_and_hard_case_priors_without_graph_probe() -> None:
    rows = [
        {
            "od_id": "od-prior",
            "origin_lat": 51.5,
            "origin_lon": -2.6,
            "destination_lat": 51.48,
            "destination_lon": -3.18,
            "candidate_probe_path_count": 4,
            "candidate_probe_corridor_family_count": 3,
            "candidate_probe_objective_spread": 0.31,
            "candidate_probe_nominal_margin": 0.09,
            "candidate_probe_toll_disagreement_rate": 1.0,
        }
    ]

    enriched = ambiguity_module.enrich_rows(
        rows,
        probe_max_paths=6,
        bootstrap_index=None,
        use_graph_probe=False,
    )

    assert enriched[0]["candidate_probe_path_sufficiency"] == pytest.approx(0.428571, rel=0.0, abs=1e-6)
    assert enriched[0]["candidate_probe_top2_gap_pressure"] == pytest.approx(0.51, rel=0.0, abs=1e-6)
    assert enriched[0]["candidate_probe_engine_disagreement_prior"] == pytest.approx(0.611495, rel=0.0, abs=1e-6)
    assert enriched[0]["hard_case_prior"] == pytest.approx(0.606952, rel=0.0, abs=1e-6)


def test_enrich_rows_can_force_recompute_cheap_priors_without_graph_probe() -> None:
    rows = [
        {
            "od_id": "od-prior-recompute",
            "origin_lat": 51.5,
            "origin_lon": -2.6,
            "destination_lat": 51.48,
            "destination_lon": -3.18,
            "candidate_probe_path_count": 4,
            "candidate_probe_corridor_family_count": 3,
            "candidate_probe_objective_spread": 0.31,
            "candidate_probe_nominal_margin": 0.09,
            "candidate_probe_toll_disagreement_rate": 1.0,
            "candidate_probe_engine_disagreement_prior": 0.10,
            "hard_case_prior": 0.10,
        }
    ]

    enriched = ambiguity_module.enrich_rows(
        rows,
        probe_max_paths=6,
        bootstrap_index=None,
        use_graph_probe=False,
        use_engine_probe=False,
        recompute_cheap_priors=True,
    )

    assert enriched[0]["candidate_probe_engine_disagreement_prior"] == pytest.approx(0.743599, rel=0.0, abs=1e-6)
    assert enriched[0]["hard_case_prior"] == pytest.approx(0.737441, rel=0.0, abs=1e-6)


def test_failed_graph_probe_backfills_repo_local_priors_instead_of_forcing_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        ambiguity_module,
        "route_graph_od_feasibility",
        lambda **kwargs: {"ok": False, "reason_code": "routing_graph_no_path", "message": "no path"},  # noqa: ARG005
    )

    enriched = ambiguity_module.enrich_rows(
        [
            {
                "od_id": "od-fail",
                "origin_lat": 51.5,
                "origin_lon": -2.6,
                "destination_lat": 53.48,
                "destination_lon": -1.18,
                "distance_bin": "100-250 km",
            }
        ],
        probe_max_paths=6,
        bootstrap_index=None,
        use_graph_probe=True,
    )

    row = enriched[0]
    assert row["accepted"] is False
    assert row["route_graph_reason_code"] == "routing_graph_no_path"
    assert row["od_ambiguity_index"] > 0.0
    assert row["candidate_probe_engine_disagreement_prior"] > 0.0
    assert row["hard_case_prior"] > 0.0
    assert "repo_local_geometry_backfill" in str(row["ambiguity_prior_source"])
    assert row["ambiguity_prior_nonzero"] is True
    assert int(row["ambiguity_prior_support_count"]) > 0
    assert row["od_ambiguity_confidence"] > 0.0
    assert row["od_ambiguity_source_count"] >= 1


def test_weak_graph_probe_is_augmented_instead_of_preserving_zero_priors(monkeypatch) -> None:
    monkeypatch.setattr(
        ambiguity_module,
        "route_graph_od_feasibility",
        lambda **kwargs: {"ok": True, "reason_code": "ok", "message": "ok"},  # noqa: ARG005
    )
    monkeypatch.setattr(
        ambiguity_module,
        "_candidate_probe_payload",
        lambda **kwargs: {  # noqa: ARG005
            "candidate_probe_accepted": True,
            "candidate_probe_reason_code": "ok",
            "candidate_probe_message": "weak probe",
            "candidate_probe_emitted_paths": 1,
            "candidate_probe_generated_paths": 1,
            "candidate_probe_explored_states": 10,
            "candidate_probe_candidate_budget": 10,
            "candidate_probe_effective_max_hops": 3,
            "candidate_probe_effective_state_budget": 20,
            "candidate_probe_path_count": 1,
            "candidate_probe_corridor_family_count": 1,
            "candidate_probe_distance_spread_km": 0.0,
            "candidate_probe_duration_spread_s": 0.0,
            "candidate_probe_cost_spread": 0.0,
            "candidate_probe_emissions_spread_kg": 0.0,
            "candidate_probe_objective_spread": 0.0,
            "candidate_probe_nominal_margin": 0.0,
            "candidate_probe_toll_disagreement_rate": 0.0,
            "candidate_probe_family_diversity": 0.0,
            "candidate_probe_engine_disagreement_prior": 0.0,
            "hard_case_prior": 0.0,
            "ambiguity_index": 0.0,
            "od_ambiguity_index": 0.0,
        },
    )
    monkeypatch.setattr(
        ambiguity_module,
        "_engine_augmented_prior_payload",
        lambda row: {  # noqa: ARG005
            "od_ambiguity_index": 0.61,
            "candidate_probe_engine_disagreement_prior": 0.57,
            "hard_case_prior": 0.49,
            "engine_probe_geometry_disagreement": 0.44,
            "engine_probe_duration_delta": 0.21,
            "engine_probe_distance_delta": 0.12,
            "engine_probe_request_fingerprint": "engine-probe-od-weak",
        },
    )

    enriched = ambiguity_module.enrich_rows(
        [
            {
                "od_id": "od-weak",
                "origin_lat": 51.5,
                "origin_lon": -2.6,
                "destination_lat": 53.48,
                "destination_lon": -1.18,
                "distance_bin": "100-250 km",
            }
        ],
        probe_max_paths=6,
        bootstrap_index=None,
        use_graph_probe=True,
    )

    row = enriched[0]
    assert row["accepted"] is True
    assert row["candidate_probe_path_count"] >= 1
    assert row["od_ambiguity_index"] > 0.0
    assert row["candidate_probe_engine_disagreement_prior"] > 0.0
    assert row["hard_case_prior"] > 0.0
    assert "routing_graph_probe" in str(row["ambiguity_prior_source"])
    assert "engine_augmented_probe" in str(row["ambiguity_prior_source"])
    assert row["od_ambiguity_source_count"] >= 2
    assert '"routing_graph_probe"' in row["od_ambiguity_source_mix"]
    assert '"engine_augmented_probe"' in row["od_ambiguity_source_mix"]
    assert row["od_ambiguity_source_mix_count"] >= 2
    assert '"routing_graph_probe"' in row["od_ambiguity_source_support"]
    assert '"engine_augmented_probe"' in row["od_ambiguity_source_support"]
    assert row["od_ambiguity_source_support_strength"] > 0.0
    assert row["od_ambiguity_source_entropy"] > 0.0
    assert row["od_ambiguity_support_ratio"] > 0.0
    assert row["od_ambiguity_prior_strength"] == row["od_ambiguity_index"]


def test_curated_ambiguity_corpus_is_support_rich() -> None:
    curated_path = Path(__file__).resolve().parents[1] / "data" / "eval" / "uk_od_corpus_ambiguity_curated.csv"
    with curated_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert [row["profile_id"] for row in rows] == [
        "ambiguity_longhaul_a",
        "ambiguity_shorthaul_b",
        "ambiguity_midhaul_c",
        "ambiguity_corridor_d",
        "ambiguity_capital_e",
        "ambiguity_capital_f",
        "ambiguity_northbound_g",
        "ambiguity_longhaul_h",
    ]
    assert all(row["corpus_group"] == "ambiguity" for row in rows)
    assert all(row["profile_id"].startswith("ambiguity_") for row in rows)
    assert all(float(row["od_ambiguity_source_count"]) >= 2.0 for row in rows)
    assert all(float(row["od_ambiguity_source_mix_count"]) >= 2.0 for row in rows)
    assert all(float(row["od_ambiguity_support_ratio"]) > 0.0 for row in rows)
    assert all(float(row["od_ambiguity_source_support_strength"]) > 0.0 for row in rows)
    assert any("routing_graph_probe" in row["ambiguity_prior_source"] for row in rows)
    assert any("historical_results_bootstrap" in row["ambiguity_prior_source"] for row in rows)
    assert any("repo_local_geometry_backfill" in row["ambiguity_prior_source"] for row in rows)


def test_weak_graph_probe_without_engine_payload_keeps_repo_local_backfill_label(monkeypatch) -> None:
    monkeypatch.setattr(
        ambiguity_module,
        "route_graph_od_feasibility",
        lambda **kwargs: {"ok": True, "reason_code": "ok", "message": "ok"},  # noqa: ARG005
    )
    monkeypatch.setattr(
        ambiguity_module,
        "_candidate_probe_payload",
        lambda **kwargs: {  # noqa: ARG005
            "candidate_probe_accepted": True,
            "candidate_probe_reason_code": "ok",
            "candidate_probe_message": "weak probe",
            "candidate_probe_emitted_paths": 1,
            "candidate_probe_generated_paths": 1,
            "candidate_probe_explored_states": 10,
            "candidate_probe_candidate_budget": 10,
            "candidate_probe_effective_max_hops": 3,
            "candidate_probe_effective_state_budget": 20,
            "candidate_probe_path_count": 1,
            "candidate_probe_corridor_family_count": 1,
            "candidate_probe_distance_spread_km": 0.0,
            "candidate_probe_duration_spread_s": 0.0,
            "candidate_probe_cost_spread": 0.0,
            "candidate_probe_emissions_spread_kg": 0.0,
            "candidate_probe_objective_spread": 0.0,
            "candidate_probe_nominal_margin": 0.0,
            "candidate_probe_toll_disagreement_rate": 0.0,
            "candidate_probe_family_diversity": 0.0,
            "candidate_probe_engine_disagreement_prior": 0.0,
            "hard_case_prior": 0.0,
            "ambiguity_index": 0.0,
            "od_ambiguity_index": 0.0,
        },
    )
    monkeypatch.setattr(
        ambiguity_module,
        "_engine_augmented_prior_payload",
        lambda row: None,  # noqa: ARG005
    )

    enriched = ambiguity_module.enrich_rows(
        [
            {
                "od_id": "od-weak-no-engine",
                "origin_lat": 51.5,
                "origin_lon": -2.6,
                "destination_lat": 53.48,
                "destination_lon": -1.18,
                "distance_bin": "100-250 km",
            }
        ],
        probe_max_paths=6,
        bootstrap_index=None,
        use_graph_probe=True,
        use_engine_probe=True,
    )

    row = enriched[0]
    source_text = str(row["ambiguity_prior_source"])
    assert "repo_local_geometry_backfill" in source_text
    assert "engine_augmented_probe" not in source_text


def test_split_prior_sources_accepts_json_encoded_source_mix() -> None:
    raw = '{"engine_augmented_probe":1,"routing_graph_probe":1}'
    parsed = ambiguity_module._split_prior_sources(raw)

    assert parsed == ["engine_augmented_probe", "routing_graph_probe"]


def test_skip_graph_probe_preserves_existing_graph_probe_source_signal(monkeypatch) -> None:
    monkeypatch.setattr(
        ambiguity_module,
        "_engine_augmented_prior_payload",
        lambda row: None,  # noqa: ARG005
    )

    enriched = ambiguity_module.enrich_rows(
        [
            {
                "od_id": "od-existing-graph",
                "origin_lat": 51.5,
                "origin_lon": -2.6,
                "destination_lat": 53.48,
                "destination_lon": -1.18,
                "distance_bin": "100-250 km",
                "candidate_probe_path_count": 4,
                "candidate_probe_corridor_family_count": 2,
                "candidate_probe_explored_states": 120,
                "candidate_probe_objective_spread": 0.18,
                "candidate_probe_nominal_margin": 0.22,
                "candidate_probe_toll_disagreement_rate": 0.05,
                "ambiguity_prior_source": "historical_results_bootstrap",
                "ambiguity_prior_sample_count": 6,
                "ambiguity_prior_support_count": 3,
                "od_ambiguity_index": 0.31,
                "hard_case_prior": 0.28,
                "candidate_probe_engine_disagreement_prior": 0.26,
            }
        ],
        bootstrap_index=None,
        use_graph_probe=False,
        use_engine_probe=False,
        recompute_cheap_priors=True,
    )

    row = enriched[0]
    source_text = str(row["ambiguity_prior_source"])
    assert "historical_results_bootstrap" in source_text
    assert "routing_graph_probe" in source_text
    assert '"routing_graph_probe"' in row["od_ambiguity_source_mix"]


def test_main_summary_aggregates_actual_od_source_mix_counts(tmp_path: Path, monkeypatch) -> None:
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    summary_json = tmp_path / "summary.json"
    input_csv.write_text("od_id,origin_lat,origin_lon,destination_lat,destination_lon\nod-1,51.5,-2.6,53.48,-1.18\n", encoding="utf-8")

    monkeypatch.setattr(
        ambiguity_module,
        "enrich_rows",
        lambda rows, **kwargs: [  # noqa: ARG005
            {
                **rows[0],
                "od_ambiguity_index": 0.4,
                "od_ambiguity_confidence": 0.8,
                "od_ambiguity_source_count": 2,
                "od_ambiguity_source_mix_count": 2,
                "od_ambiguity_source_mix": '{"routing_graph_probe":3,"engine_augmented_probe":1}',
                "od_ambiguity_source_support": '{"routing_graph_probe":0.8,"engine_augmented_probe":0.7}',
                "od_ambiguity_source_support_strength": 0.75,
                "od_ambiguity_source_entropy": 0.56,
                "od_ambiguity_support_ratio": 0.75,
                "ambiguity_prior_nonzero": True,
                "ambiguity_prior_source": "routing_graph_probe,engine_augmented_probe",
            },
            {
                **rows[0],
                "od_id": "od-2",
                "od_ambiguity_index": 0.6,
                "od_ambiguity_confidence": 0.9,
                "od_ambiguity_source_count": 2,
                "od_ambiguity_source_mix_count": 2,
                "od_ambiguity_source_mix": '{"routing_graph_probe":2,"historical_results_bootstrap":4}',
                "od_ambiguity_source_support": '{"routing_graph_probe":0.7,"historical_results_bootstrap":0.9}',
                "od_ambiguity_source_support_strength": 0.8,
                "od_ambiguity_source_entropy": 0.62,
                "od_ambiguity_support_ratio": 0.8,
                "ambiguity_prior_nonzero": True,
                "ambiguity_prior_source": "routing_graph_probe,historical_results_bootstrap",
            },
        ],
    )

    exit_code = ambiguity_module.main(
        [
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["od_ambiguity_source_mix"] == {
        "routing_graph_probe": 5,
        "historical_results_bootstrap": 4,
        "engine_augmented_probe": 1,
    }


def test_engine_probe_can_supply_upstream_ambiguity_without_graph_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        ambiguity_module,
        "_engine_augmented_prior_payload",
        lambda row: {  # noqa: ARG005
            "candidate_probe_engine_disagreement_prior": 0.44,
            "hard_case_prior": 0.51,
            "ambiguity_index": 0.51,
            "od_ambiguity_index": 0.51,
            "ambiguity_prior_sample_count": 2,
            "ambiguity_prior_support_count": 4,
            "engine_probe_geometry_disagreement": 0.33,
            "engine_probe_duration_delta": 0.18,
            "engine_probe_distance_delta": 0.09,
            "engine_probe_request_fingerprint": "probe-fingerprint",
        },
    )

    enriched = ambiguity_module.enrich_rows(
        [
            {
                "od_id": "od-engine",
                "origin_lat": 51.5,
                "origin_lon": -2.6,
                "destination_lat": 53.48,
                "destination_lon": -1.18,
                "distance_bin": "100-250 km",
            }
        ],
        probe_max_paths=6,
        bootstrap_index=None,
        use_graph_probe=False,
        use_engine_probe=True,
    )

    row = enriched[0]
    assert row["od_ambiguity_index"] == 0.51
    assert row["candidate_probe_engine_disagreement_prior"] == 0.44
    assert row["hard_case_prior"] == 0.51
    assert row["engine_probe_geometry_disagreement"] == 0.33
    assert row["engine_probe_request_fingerprint"] == "probe-fingerprint"
    assert row["od_ambiguity_source_count"] >= 1
    assert '"engine_augmented_probe"' in row["od_ambiguity_source_support"]
    assert row["od_ambiguity_source_support_strength"] > 0.0
    assert "engine_augmented_probe" in str(row["ambiguity_prior_source"])


def test_enrich_rows_fuses_graph_engine_and_historical_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        ambiguity_module,
        "route_graph_od_feasibility",
        lambda **kwargs: {"ok": True, "reason_code": "ok", "message": "ok"},  # noqa: ARG005
    )
    monkeypatch.setattr(
        ambiguity_module,
        "_candidate_probe_payload",
        lambda **kwargs: {  # noqa: ARG005
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
            "candidate_probe_engine_disagreement_prior": 0.41,
            "hard_case_prior": 0.52,
            "ambiguity_index": 0.55,
            "od_ambiguity_index": 0.55,
            "ambiguity_prior_sample_count": 3,
            "ambiguity_prior_support_count": 2,
        },
    )
    monkeypatch.setattr(
        ambiguity_module,
        "_engine_augmented_prior_payload",
        lambda row: {  # noqa: ARG005
            "candidate_probe_engine_disagreement_prior": 0.44,
            "hard_case_prior": 0.51,
            "ambiguity_index": 0.51,
            "od_ambiguity_index": 0.51,
            "ambiguity_prior_sample_count": 2,
            "ambiguity_prior_support_count": 4,
            "engine_probe_geometry_disagreement": 0.33,
            "engine_probe_duration_delta": 0.18,
            "engine_probe_distance_delta": 0.09,
            "engine_probe_request_fingerprint": "probe-fingerprint",
        },
    )

    row = {
        "od_id": "od-fused",
        "origin_lat": 51.5,
        "origin_lon": -2.6,
        "destination_lat": 51.48,
        "destination_lon": -3.18,
        "distance_bin": "30-100 km",
    }
    bootstrap_index = {
        "by_identity": {
            ambiguity_module._row_identity_key(row): {
                "ambiguity_index": 0.61,
                "candidate_probe_path_count": 9,
                "candidate_probe_corridor_family_count": 3,
                "candidate_probe_objective_spread": 0.42,
                "candidate_probe_nominal_margin": 0.18,
                "candidate_probe_engine_disagreement_prior": 0.44,
                "hard_case_prior": 0.52,
                "ambiguity_prior_source": "historical_results_bootstrap",
                "ambiguity_prior_sample_count": 4,
                "ambiguity_prior_support_count": 4,
            }
        },
        "by_od_id_unique": {},
    }

    enriched = ambiguity_module.enrich_rows(
        [row],
        probe_max_paths=6,
        bootstrap_index=bootstrap_index,
        use_graph_probe=True,
        use_engine_probe=True,
    )

    fused = enriched[0]
    support_map = json.loads(fused["od_ambiguity_source_support"])
    assert "historical_results_bootstrap" in str(fused["ambiguity_prior_source"])
    assert fused["od_ambiguity_source_count"] >= 3
    assert fused["od_ambiguity_source_mix_count"] >= 3
    assert {"routing_graph_probe", "engine_augmented_probe", "historical_results_bootstrap"} <= set(support_map)
    assert fused["od_ambiguity_source_support_strength"] > 0.0
    assert fused["od_ambiguity_source_entropy"] > 0.0
