from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import httpx
import pytest

import scripts.compose_thesis_suite_report as compose_module
import scripts.build_od_corpus_uk as corpus_module
import scripts.enrich_od_corpus_with_ambiguity as ambiguity_module
import scripts.run_thesis_evaluation as thesis_module
import app.main as main_module
from app.models import LatLng, RouteRequest

pytestmark = [pytest.mark.thesis, pytest.mark.thesis_results]


@pytest.fixture(autouse=True)
def _mock_backend_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        thesis_module,
        "_wait_for_backend_ready",
        lambda client, *, backend_url, timeout_seconds, poll_seconds: {
            "strict_route_ready": True,
            "status": "ready",
            "compute_ms": 12.5,
            "wait_elapsed_ms": 42.0,
            "route_graph": {
                "status": "ready",
                "state": "ready",
                "elapsed_ms": 1250.0,
                "started_at_utc": "2026-03-24T12:00:00Z",
                "ready_at_utc": "2026-03-24T12:00:01.250000Z",
            },
            "strict_live": {"ok": True},
        },
    )


def test_route_graph_startup_to_ready_prefers_timestamps_and_ignores_ready_lifetime_elapsed() -> None:
    explicit = thesis_module._route_graph_startup_to_ready_ms(
        {
            "route_graph": {
                "state": "ready",
                "started_at_utc": "2026-03-24T12:00:00Z",
                "ready_at_utc": "2026-03-24T12:00:01.250000Z",
                "elapsed_ms": 999999.0,
            }
        }
    )
    assert explicit == 1250.0

    ready_without_timestamp = thesis_module._route_graph_startup_to_ready_ms(
        {"route_graph": {"state": "ready", "elapsed_ms": 999999.0}}
    )
    assert ready_without_timestamp is None


def test_valid_refine_cost_rows_keeps_small_positive_observed_costs() -> None:
    valid_rows = thesis_module._valid_refine_cost_rows(
        [
            {"predicted_refine_cost": 0.4, "observed_refine_cost": 0.2},
            {"predicted_refine_cost": 5.0, "observed_refine_cost": 0.0},
            {"predicted_refine_cost": -1.0, "observed_refine_cost": 4.0},
        ]
    )

    assert len(valid_rows) == 2
    assert valid_rows[0]["predicted_refine_cost"] == 0.4


def test_cache_mode_defaults_to_preserve_and_normalizes_unknown_values() -> None:
    args = thesis_module._build_parser().parse_args(["--corpus-csv", "dummy.csv"])

    assert args.cache_mode == "preserve"
    assert thesis_module._normalize_cache_mode("cold") == "cold"
    assert thesis_module._normalize_cache_mode("PRESERVE") == "preserve"
    assert thesis_module._normalize_cache_mode("unexpected") == "preserve"


def test_load_rows_preserves_corpus_provenance_fields() -> None:
    rows = thesis_module._load_rows(
        [
            {
                "od_id": "row-1",
                "origin_lat": 51.5,
                "origin_lon": -0.1,
                "destination_lat": 52.5,
                "destination_lon": -1.9,
                "ambiguity_prior_source": "routing_graph_probe,engine_augmented_probe",
                "corridor_bucket": "capital_connector",
            }
        ],
        seed=7,
        max_od=0,
    )

    assert rows[0]["ambiguity_prior_source"] == "routing_graph_probe,engine_augmented_probe"
    assert rows[0]["corridor_bucket"] == "capital_connector"


def test_resolve_evaluation_suite_metadata_prefers_explicit_role_and_infers_focus() -> None:
    explicit_args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "backend/out/uk_od_corpus_thesis_refc_focus_20.csv",
            "--evaluation-suite-role",
            "hot_rerun",
        ]
    )
    explicit_metadata = thesis_module._resolve_evaluation_suite_metadata(
        args=explicit_args,
        corpus_source_path=str(explicit_args.corpus_csv),
        run_id="pair_hot",
    )

    inferred_args = thesis_module._build_parser().parse_args(
        ["--corpus-csv", "backend/out/uk_od_corpus_thesis_refc_focus_20.csv"]
    )
    inferred_metadata = thesis_module._resolve_evaluation_suite_metadata(
        args=inferred_args,
        corpus_source_path=str(inferred_args.corpus_csv),
        run_id="focused_refc_run",
    )

    assert explicit_metadata == {
        "role": "hot_rerun",
        "scope": "hot_rerun",
        "focus": "runtime_reuse",
        "label": "Hot rerun proof",
        "source": "explicit_arg",
    }
    assert inferred_metadata == {
        "role": "focused_refc_proof",
        "scope": "focused",
        "focus": "refc",
        "label": "Focused REFC proof",
        "source": "inferred_from_corpus_source_path",
    }


def test_clear_backend_caches_uses_thesis_cold_scope() -> None:
    observed_paths: list[str] = []

    class _Client:
        def delete(self, path: str):
            observed_paths.append(path)
            return types.SimpleNamespace(status_code=200, json=lambda: {"cleared": 1})

    payload = thesis_module._clear_backend_caches(_Client(), backend_url="http://backend")

    assert payload == {"cleared": 1}
    assert observed_paths == ["http://backend/cache?scope=thesis_cold"]


def test_valid_refine_cost_rows_keeps_zero_observed_and_excludes_missing_costs() -> None:
    rows = thesis_module._valid_refine_cost_rows(
        [
            {"predicted_refine_cost": 155.0, "observed_refine_cost": 0.0},
            {"predicted_refine_cost": 160.0, "observed_refine_cost": None},
            {"predicted_refine_cost": 170.0, "observed_refine_cost": 45.0},
        ]
    )

    assert rows == [
        {"predicted_refine_cost": 155.0, "observed_refine_cost": 0.0},
        {"predicted_refine_cost": 170.0, "observed_refine_cost": 45.0},
    ]


def test_refine_cost_summary_keeps_pre_realized_zero_cost_rows_in_metrics() -> None:
    rows = [
        {"predicted_refine_cost": 155.0, "observed_refine_cost": 0.0},
        {"predicted_refine_cost": 170.0, "observed_refine_cost": 45.0},
    ]
    valid_rows = thesis_module._valid_refine_cost_rows(rows)

    assert len(valid_rows) == 2
    assert thesis_module.refine_cost_sample_count(valid_rows) == 2
    assert thesis_module.refine_cost_positive_sample_count(valid_rows) == 1
    assert thesis_module.refine_cost_zero_observed_count(valid_rows) == 1
    assert thesis_module.refine_cost_mape(valid_rows) == pytest.approx(2.777778, rel=0.0, abs=1e-6)
    assert thesis_module.refine_cost_mae_ms(valid_rows) == 140.0
    assert thesis_module.refine_cost_rank_correlation(valid_rows) is not None


def test_result_refine_cost_rows_drop_voi_rows_without_refine_actions() -> None:
    rows = [
        {"candidate_id": "cand-a", "predicted_refine_cost": 155.0, "observed_refine_cost": 0.0},
        {"candidate_id": "cand-b", "predicted_refine_cost": 170.0, "observed_refine_cost": 45.0},
    ]

    assert thesis_module._result_refine_cost_rows(
        rows,
        pipeline_mode="voi",
        voi_entries=[{"chosen_action": {"kind": "increase_stochastic_samples"}}],
    ) == []


def test_result_refine_cost_rows_scope_voi_metrics_to_executed_candidates() -> None:
    rows = [
        {"candidate_id": "cand-a", "predicted_refine_cost": 155.0, "observed_refine_cost": 0.0},
        {"candidate_id": "cand-b", "predicted_refine_cost": 170.0, "observed_refine_cost": 45.0},
        {"candidate_id": "cand-c", "predicted_refine_cost": 200.0, "observed_refine_cost": 55.0},
    ]

    assert thesis_module._result_refine_cost_rows(
        rows,
        pipeline_mode="voi",
        voi_entries=[
            {
                "chosen_action": {"kind": "refine_top1_dccs"},
                "executed_candidate_ids": ["cand-b", "missing"],
            },
            {"chosen_action": {"kind": "refresh_top1_vor"}},
        ],
    ) == [{"candidate_id": "cand-b", "predicted_refine_cost": 170.0, "observed_refine_cost": 45.0}]


def test_result_refine_cost_rows_preserve_voi_rows_when_execution_ids_are_missing() -> None:
    rows = [
        {"candidate_id": "cand-a", "predicted_refine_cost": 155.0, "observed_refine_cost": 0.0},
        {"candidate_id": "cand-b", "predicted_refine_cost": 170.0, "observed_refine_cost": 45.0},
    ]

    assert thesis_module._result_refine_cost_rows(
        rows,
        pipeline_mode="voi",
        voi_entries=[{"chosen_action": {"kind": "refine_top1_dccs"}}],
    ) == rows


def test_ambiguity_budget_prior_logic_matches_corpus_enricher() -> None:
    row = {
        "od_ambiguity_support_ratio": 0.79,
        "od_ambiguity_source_entropy": 0.63,
        "od_ambiguity_source_count": 2,
        "od_ambiguity_source_mix_count": 2,
        "od_ambiguity_prior_strength": 0.72,
        "od_ambiguity_family_density": 0.66,
        "od_ambiguity_margin_pressure": 0.92,
        "od_ambiguity_spread_pressure": 0.31,
        "od_ambiguity_toll_instability": 0.12,
        "candidate_probe_engine_disagreement_prior": 0.48,
        "hard_case_prior": 0.35,
    }

    corpus_prior = ambiguity_module._support_gated_budget_prior(
        ambiguity_value=0.18,
        support_ratio=row["od_ambiguity_support_ratio"],
        source_entropy=row["od_ambiguity_source_entropy"],
        source_count=row["od_ambiguity_source_count"],
        source_mix_count=row["od_ambiguity_source_mix_count"],
        prior_strength=row["od_ambiguity_prior_strength"],
        family_density=row["od_ambiguity_family_density"],
        margin_pressure=row["od_ambiguity_margin_pressure"],
        spread_pressure=row["od_ambiguity_spread_pressure"],
        toll_instability=row["od_ambiguity_toll_instability"],
        engine_prior=row["candidate_probe_engine_disagreement_prior"],
        hard_case_prior=row["hard_case_prior"],
    )
    evaluator_prior = thesis_module._support_gated_ambiguity_budget_prior(row, 0.18)

    assert evaluator_prior == corpus_prior
    assert evaluator_prior > 0.18

    weak_row = {
        "od_ambiguity_support_ratio": 0.41,
        "od_ambiguity_source_entropy": 0.21,
        "od_ambiguity_source_count": 1,
        "od_ambiguity_source_mix_count": 1,
        "od_ambiguity_prior_strength": 0.19,
        "od_ambiguity_family_density": 0.18,
        "od_ambiguity_margin_pressure": 0.22,
        "od_ambiguity_spread_pressure": 0.11,
        "od_ambiguity_toll_instability": 0.03,
        "candidate_probe_engine_disagreement_prior": 0.18,
        "hard_case_prior": 0.12,
    }
    weak_corpus_prior = ambiguity_module._support_gated_budget_prior(
        ambiguity_value=0.18,
        support_ratio=weak_row["od_ambiguity_support_ratio"],
        source_entropy=weak_row["od_ambiguity_source_entropy"],
        source_count=weak_row["od_ambiguity_source_count"],
        source_mix_count=weak_row["od_ambiguity_source_mix_count"],
        prior_strength=weak_row["od_ambiguity_prior_strength"],
        family_density=weak_row["od_ambiguity_family_density"],
        margin_pressure=weak_row["od_ambiguity_margin_pressure"],
        spread_pressure=weak_row["od_ambiguity_spread_pressure"],
        toll_instability=weak_row["od_ambiguity_toll_instability"],
        engine_prior=weak_row["candidate_probe_engine_disagreement_prior"],
        hard_case_prior=weak_row["hard_case_prior"],
    )
    weak_evaluator_prior = thesis_module._support_gated_ambiguity_budget_prior(weak_row, 0.18)
    assert weak_evaluator_prior == weak_corpus_prior == 0.18


def test_ambiguity_budget_prior_parity_includes_candidate_probe_only_rows() -> None:
    row = {
        "candidate_probe_ambiguity_index": 0.66,
        "od_ambiguity_support_ratio": 0.77,
        "od_ambiguity_source_entropy": 0.64,
        "od_ambiguity_source_count": 2,
        "od_ambiguity_source_mix_count": 2,
        "od_ambiguity_prior_strength": 0.61,
        "od_ambiguity_family_density": 0.58,
        "od_ambiguity_margin_pressure": 0.49,
        "od_ambiguity_spread_pressure": 0.27,
        "od_ambiguity_toll_instability": 0.13,
        "candidate_probe_engine_disagreement_prior": 0.39,
        "hard_case_prior": 0.33,
    }

    raw_prior = ambiguity_module._raw_ambiguity_prior_value(row)
    assert raw_prior == 0.66
    assert thesis_module._od_ambiguity_prior(row) == ambiguity_module._support_gated_budget_prior(
        ambiguity_value=raw_prior,
        support_ratio=row["od_ambiguity_support_ratio"],
        source_entropy=row["od_ambiguity_source_entropy"],
        source_count=row["od_ambiguity_source_count"],
        source_mix_count=row["od_ambiguity_source_mix_count"],
        prior_strength=row["od_ambiguity_prior_strength"],
        family_density=row["od_ambiguity_family_density"],
        margin_pressure=row["od_ambiguity_margin_pressure"],
        spread_pressure=row["od_ambiguity_spread_pressure"],
        toll_instability=row["od_ambiguity_toll_instability"],
        engine_prior=row["candidate_probe_engine_disagreement_prior"],
        hard_case_prior=row["hard_case_prior"],
    )


def test_summary_productive_actions_and_certificate_selectivity_use_scope_denominators() -> None:
    summary_rows = thesis_module._summary_rows(
        [
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "cohort_label": "representative",
                "certificate_selective": None,
                "voi_productive_action_count": None,
                "voi_nonproductive_action_count": None,
                "voi_action_count": 10,
            },
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "cohort_label": "hard_case",
                "certificate_selective": True,
                "voi_productive_action_count": 1,
                "voi_nonproductive_action_count": 1,
                "voi_action_count": 10,
            },
        ]
    )

    row = next(item for item in summary_rows if item["variant_id"] == "C")
    assert row["productive_voi_action_rate"] == 0.5
    assert row["productive_voi_action_denominator"] == 1
    assert row["certificate_selectivity_rate"] == 1.0
    assert row["certificate_selectivity_denominator"] == 1


def test_summary_time_preserving_rates_require_duration_guard_and_surface_best_baseline() -> None:
    summary_rows = thesis_module._summary_rows(
        [
            {
                "variant_id": "A",
                "pipeline_mode": "dccs",
                "cohort_label": "representative",
                "time_preserving_win_osrm": True,
                "time_preserving_win_ors": True,
                "time_preserving_win_best_baseline": True,
                "time_preserving_dominance_osrm": False,
                "time_preserving_dominance_ors": True,
                "time_preserving_dominance_best_baseline": False,
            },
            {
                "variant_id": "A",
                "pipeline_mode": "dccs",
                "cohort_label": "representative",
                "time_preserving_win_osrm": False,
                "time_preserving_win_ors": True,
                "time_preserving_win_best_baseline": False,
                "time_preserving_dominance_osrm": False,
                "time_preserving_dominance_ors": False,
                "time_preserving_dominance_best_baseline": False,
            },
        ]
    )

    row = next(item for item in summary_rows if item["variant_id"] == "A")
    assert row["time_preserving_win_rate"] == 0.5
    assert row["time_preserving_denominator"] == 2
    assert row["time_preserving_win_rate_osrm"] == 0.5
    assert row["time_preserving_denominator_osrm"] == 2
    assert row["time_preserving_win_rate_ors"] == 1.0
    assert row["time_preserving_denominator_ors"] == 2
    assert row["time_preserving_dominance_rate"] == 0.0
    assert row["time_preserving_dominance_denominator"] == 2
    assert row["time_preserving_dominance_rate_ors"] == 0.5
    assert row["time_preserving_dominance_denominator_ors"] == 2


def _candidate_probe_ok(**kwargs):  # noqa: ARG001
    return [
        {
            "route_id": "probe-route",
            "distance": 1000.0,
            "duration": 120.0,
        }
    ], corpus_module.GraphCandidateDiagnostics(
        explored_states=17,
        generated_paths=2,
        emitted_paths=1,
        candidate_budget=1,
        effective_max_hops=32,
        effective_state_budget=5000,
    )


def _candidate_probe_fail(**kwargs):  # noqa: ARG001
    return [], corpus_module.GraphCandidateDiagnostics(
        explored_states=91,
        generated_paths=0,
        emitted_paths=0,
        candidate_budget=1,
        effective_max_hops=48,
        effective_state_budget=12000,
        no_path_reason="routing_graph_no_path",
        no_path_detail="candidate probe exhausted",
    )


def _candidate_probe_ambiguous(**kwargs):  # noqa: ARG001
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
        {
            "route_id": "probe-c",
            "distance": 1350.0,
            "duration": 180.0,
            "cost": 21.0,
            "emissions": 8.0,
            "corridor_family": "north",
            "has_tolls": False,
        },
    ], corpus_module.GraphCandidateDiagnostics(
        explored_states=41,
        generated_paths=5,
        emitted_paths=3,
        candidate_budget=6,
        effective_max_hops=64,
        effective_state_budget=9000,
    )


def _build_corpus_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "od_id",
                "origin_lat",
                "origin_lon",
                "destination_lat",
                "destination_lon",
                "distance_bin",
                "seed",
                "corpus_kind",
                "ambiguity_index",
                "od_ambiguity_confidence",
                "od_ambiguity_source_count",
                "od_ambiguity_source_mix",
                "od_ambiguity_source_mix_count",
                "od_ambiguity_source_support",
                "od_ambiguity_source_support_strength",
                "od_ambiguity_source_entropy",
                "od_ambiguity_support_ratio",
                "od_ambiguity_prior_strength",
                "od_ambiguity_family_density",
                "od_ambiguity_margin_pressure",
                "od_ambiguity_spread_pressure",
                "candidate_probe_path_count",
                "candidate_probe_corridor_family_count",
                "candidate_probe_objective_spread",
                "candidate_probe_nominal_margin",
                "candidate_probe_toll_disagreement_rate",
                "candidate_probe_engine_disagreement_prior",
                "hard_case_prior",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "od_id": "od-000001",
                "origin_lat": 52.0,
                "origin_lon": -1.5,
                "destination_lat": 51.5,
                "destination_lon": -1.2,
                "distance_bin": "30-100 km",
                "seed": 99,
                "corpus_kind": "ambiguous",
                "ambiguity_index": 0.72,
                "od_ambiguity_confidence": 0.81,
                "od_ambiguity_source_count": 2,
                "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1}',
                "od_ambiguity_source_mix_count": 2,
                "od_ambiguity_source_support": '{"engine_augmented_probe":0.75,"routing_graph_probe":0.82}',
                "od_ambiguity_source_support_strength": 0.785,
                "od_ambiguity_source_entropy": 0.63,
                "od_ambiguity_support_ratio": 0.79,
                "od_ambiguity_prior_strength": 0.72,
                "od_ambiguity_family_density": 0.666667,
                "od_ambiguity_margin_pressure": 0.92,
                "od_ambiguity_spread_pressure": 0.31,
                "candidate_probe_path_count": 3,
                "candidate_probe_corridor_family_count": 2,
                "candidate_probe_objective_spread": 0.31,
                "candidate_probe_nominal_margin": 0.08,
                "candidate_probe_toll_disagreement_rate": 1.0,
                "candidate_probe_engine_disagreement_prior": 0.66,
                "hard_case_prior": 0.72,
            }
        )


def _build_override_corpus_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "od_id",
                "origin_lat",
                "origin_lon",
                "destination_lat",
                "destination_lon",
                "distance_bin",
                "seed",
                "corpus_kind",
                "profile_id",
                "corpus_group",
                "weight_time",
                "weight_money",
                "weight_co2",
                "scenario_mode",
                "weather_profile",
                "weather_intensity",
                "departure_time_utc",
                "stochastic_enabled",
                "stochastic_samples",
                "search_budget",
                "evidence_budget",
                "world_count",
                "certificate_threshold",
                "tau_stop",
                "max_alternatives",
                "optimization_mode",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "od_id": "od-override",
                "origin_lat": 52.0,
                "origin_lon": -1.5,
                "destination_lat": 51.5,
                "destination_lon": -1.2,
                "distance_bin": "30-100 km",
                "seed": 99,
                "corpus_kind": "ambiguous",
                "profile_id": "profile_override_a",
                "corpus_group": "ambiguity",
                "weight_time": 1.5,
                "weight_money": 0.8,
                "weight_co2": 1.2,
                "scenario_mode": "partial_sharing",
                "weather_profile": "storm",
                "weather_intensity": 1.25,
                "departure_time_utc": "2026-03-21T10:15:00Z",
                "stochastic_enabled": "true",
                "stochastic_samples": 77,
                "search_budget": 7,
                "evidence_budget": 3,
                "world_count": 88,
                "certificate_threshold": 0.87,
                "tau_stop": 0.025,
                "max_alternatives": 11,
                "optimization_mode": "robust",
            }
        )


def _build_two_row_corpus_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "od_id",
                "origin_lat",
                "origin_lon",
                "destination_lat",
                "destination_lon",
                "distance_bin",
                "seed",
                "corpus_kind",
                "profile_id",
                "corpus_group",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "od_id": "od-fail",
                "origin_lat": 52.0,
                "origin_lon": -1.5,
                "destination_lat": 51.5,
                "destination_lon": -1.2,
                "distance_bin": "30-100 km",
                "seed": 99,
                "corpus_kind": "ambiguous",
                "profile_id": "fail_profile",
                "corpus_group": "ambiguity",
            }
        )
        writer.writerow(
            {
                "od_id": "od-pass",
                "origin_lat": 53.0,
                "origin_lon": -2.0,
                "destination_lat": 53.5,
                "destination_lon": -1.8,
                "distance_bin": "30-100 km",
                "seed": 100,
                "corpus_kind": "representative",
                "profile_id": "pass_profile",
                "corpus_group": "representative",
            }
        )


def test_base_payload_clamps_certificate_world_count_to_route_schema_minimum() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--world-count",
            "8",
        ]
    )
    payload = thesis_module._base_payload(
        args,
        {
            "origin_lat": 52.0,
            "origin_lon": -1.5,
            "destination_lat": 51.5,
            "destination_lon": -1.2,
        },
        variant_seed=7,
    )
    assert payload["cert_world_count"] == 10
    assert payload["evaluation_lean_mode"] is True


def test_base_payload_propagates_ambiguity_context_fields() -> None:
    args = thesis_module._build_parser().parse_args(["--corpus-csv", "dummy.csv"])
    request_config = {
        "scenario_mode": "no_sharing",
        "weather": {"enabled": False, "profile": "clear", "intensity": 1.0},
        "departure_time_utc": "2026-03-23T08:00:00Z",
        "max_alternatives": 8,
        "weights": {"time": 1.0, "money": 1.0, "co2": 1.0},
        "cost_toggles": {"use_tolls": True, "fuel_price_multiplier": 1.0, "carbon_price_per_kg": 0.0, "toll_cost_per_km": 0.0},
        "stochastic": {"enabled": False, "sigma": 0.08, "samples": 25},
        "optimization_mode": "expected_value",
        "search_budget": 4,
        "evidence_budget": 2,
        "world_count": 64,
        "certificate_threshold": 0.8,
        "tau_stop": 0.02,
        "ambiguity_budget_prior": 0.63,
        "ambiguity_budget_band": "high",
    }
    payload = thesis_module._base_payload(
        args,
        {
            "origin_lat": 52.0,
            "origin_lon": -1.5,
            "destination_lat": 51.5,
            "destination_lon": -1.2,
            "od_ambiguity_index": 0.63,
            "od_ambiguity_confidence": 0.81,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1}',
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_source_entropy": 0.63,
            "od_ambiguity_support_ratio": 0.79,
            "od_ambiguity_prior_strength": 0.72,
            "od_ambiguity_family_density": 0.66,
            "od_ambiguity_margin_pressure": 0.92,
            "od_ambiguity_spread_pressure": 0.31,
            "od_ambiguity_toll_instability": 0.12,
            "candidate_probe_engine_disagreement_prior": 0.41,
            "hard_case_prior": 0.72,
            "candidate_probe_path_count": 5,
            "candidate_probe_corridor_family_count": 3,
            "candidate_probe_objective_spread": 0.27,
            "candidate_probe_nominal_margin": 0.18,
            "candidate_probe_toll_disagreement_rate": 0.09,
        },
        variant_seed=7,
        request_config=request_config,
    )

    assert payload["od_ambiguity_index"] == 0.63
    assert payload["od_ambiguity_confidence"] == 0.81
    assert payload["od_ambiguity_source_count"] == 2
    assert payload["od_ambiguity_source_mix"] == '{"engine_augmented_probe":1,"routing_graph_probe":1}'
    assert payload["od_ambiguity_source_mix_count"] == 2
    assert payload["od_ambiguity_source_entropy"] == 0.63
    assert payload["od_ambiguity_support_ratio"] == 0.79
    assert payload["od_ambiguity_prior_strength"] == 0.72
    assert payload["od_ambiguity_family_density"] == 0.66
    assert payload["od_ambiguity_margin_pressure"] == 0.92
    assert payload["od_ambiguity_spread_pressure"] == 0.31
    assert payload["od_ambiguity_toll_instability"] == 0.12
    assert payload["od_engine_disagreement_prior"] == 0.41
    assert payload["od_hard_case_prior"] == 0.72
    assert payload["od_candidate_path_count"] == 5
    assert payload["od_corridor_family_count"] == 3
    assert payload["od_objective_spread"] == 0.27
    assert payload["od_nominal_margin_proxy"] == 0.18
    assert payload["od_toll_disagreement_rate"] == 0.09
    assert payload["ambiguity_budget_prior"] == 0.63
    assert payload["ambiguity_budget_band"] == "high"


def test_route_request_accepts_support_rich_ambiguity_fields() -> None:
    request = RouteRequest(
        origin=LatLng(lat=52.0, lon=-1.5),
        destination=LatLng(lat=51.5, lon=-1.2),
        od_ambiguity_index=0.63,
        od_ambiguity_confidence=0.81,
        od_ambiguity_source_count=2,
        od_ambiguity_source_mix="engine_augmented_probe,routing_graph_probe",
        od_ambiguity_source_mix_count=2,
        od_ambiguity_source_entropy=0.63,
        od_ambiguity_support_ratio=0.79,
        od_ambiguity_prior_strength=0.72,
        od_ambiguity_family_density=0.66,
        od_ambiguity_margin_pressure=0.92,
        od_ambiguity_spread_pressure=0.31,
        od_ambiguity_toll_instability=0.12,
    )

    assert request.od_ambiguity_support_ratio == 0.79
    assert request.od_ambiguity_source_mix_count == 2
    assert request.od_ambiguity_source_entropy == 0.63
    assert request.od_ambiguity_prior_strength == 0.72
    assert request.od_ambiguity_family_density == 0.66
    assert request.od_ambiguity_margin_pressure == 0.92
    assert request.od_ambiguity_spread_pressure == 0.31
    assert request.od_ambiguity_toll_instability == 0.12


def test_od_ambiguity_prior_is_support_aware_and_can_fall_below_raw_prior() -> None:
    prior = thesis_module._od_ambiguity_prior(
        {
            "od_ambiguity_index": 0.22,
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
    )
    assert prior is not None
    assert prior == pytest.approx(0.22, rel=0.0, abs=1e-6)


def test_od_ambiguity_prior_uses_strongest_upstream_prior_when_support_is_rich() -> None:
    prior = thesis_module._od_ambiguity_prior(
        {
            "od_ambiguity_index": 0.22,
            "candidate_probe_engine_disagreement_prior": 0.48,
            "hard_case_prior": 0.35,
            "od_ambiguity_support_ratio": 0.79,
            "od_ambiguity_source_entropy": 0.63,
            "od_ambiguity_source_count": 2,
            "od_ambiguity_source_mix_count": 2,
            "od_ambiguity_prior_strength": 0.72,
            "od_ambiguity_family_density": 0.66,
            "od_ambiguity_margin_pressure": 0.92,
            "od_ambiguity_spread_pressure": 0.31,
            "od_ambiguity_toll_instability": 0.12,
        }
    )
    assert prior is not None
    assert prior > 0.22
    assert prior <= 0.40


def test_load_rows_exports_budget_prior_and_corroboration_counts() -> None:
    rows = thesis_module._load_rows(
        [
            {
                "od_id": "od-1",
                "origin_lat": 52.0,
                "origin_lon": -1.5,
                "destination_lat": 51.5,
                "destination_lon": -1.2,
                "distance_bin": "30-100 km",
                "corpus_kind": "ambiguity",
                "profile_id": "p1",
                "od_ambiguity_index": 0.52,
                "od_ambiguity_confidence": 0.14,
                "od_ambiguity_source_count": 2,
                "od_ambiguity_source_mix": '{"routing_graph_probe":1,"engine_augmented_probe":1}',
                "od_ambiguity_source_mix_count": 2,
                "od_ambiguity_source_entropy": 0.12,
                "od_ambiguity_support_ratio": 0.18,
                "od_ambiguity_prior_strength": 0.22,
                "od_ambiguity_family_density": 0.14,
                "od_ambiguity_margin_pressure": 0.08,
                "od_ambiguity_spread_pressure": 0.05,
                "od_ambiguity_toll_instability": 0.03,
                "ambiguity_prior_sample_count": 6,
                "ambiguity_prior_support_count": 4,
                "candidate_probe_path_count": 1,
                "candidate_probe_corridor_family_count": 1,
                "candidate_probe_objective_spread": 0.21,
                "candidate_probe_nominal_margin": 0.14,
                "candidate_probe_toll_disagreement_rate": 0.05,
                "candidate_probe_engine_disagreement_prior": 0.18,
                "hard_case_prior": 0.12,
            }
        ],
        seed=7,
        max_od=0,
    )
    assert rows[0]["ambiguity_prior_sample_count"] == 6
    assert rows[0]["ambiguity_prior_support_count"] == 4
    assert rows[0]["ambiguity_budget_prior"] is not None
    assert rows[0]["ambiguity_budget_prior"] == pytest.approx(0.52, rel=0.0, abs=1e-6)
    assert rows[0]["ambiguity_budget_prior_gap"] == 0.0
    assert rows[0]["budget_prior_exceeds_raw"] is False


def test_load_rows_preserves_explicit_row_seed_and_falls_back_to_default_seed() -> None:
    rows = thesis_module._load_rows(
        [
            {
                "od_id": "od-explicit",
                "origin_lat": 52.0,
                "origin_lon": -1.5,
                "destination_lat": 51.5,
                "destination_lon": -1.2,
                "distance_bin": "30-100 km",
                "corpus_kind": "ambiguity",
                "profile_id": "p1",
                "seed": 99,
            },
            {
                "od_id": "od-default",
                "origin_lat": 52.1,
                "origin_lon": -1.4,
                "destination_lat": 51.6,
                "destination_lon": -1.1,
                "distance_bin": "30-100 km",
                "corpus_kind": "ambiguity",
                "profile_id": "p2",
            },
        ],
        seed=7,
        max_od=0,
    )

    assert rows[0]["seed"] == 99
    assert rows[1]["seed"] == 7


def test_load_rows_parses_embedded_row_override_blob_and_prefers_explicit_columns() -> None:
    rows = thesis_module._load_rows(
        [
            {
                "od_id": "od-embedded",
                "origin_lat": 52.0,
                "origin_lon": -1.5,
                "destination_lat": 51.5,
                "destination_lon": -1.2,
                "distance_bin": "30-100 km",
                "corpus_kind": "ambiguity",
                "profile_id": "p1",
                "search_budget": "6",
                "row_overrides": "{'profile_id': 'profile_override_a', 'corpus_group': 'ambiguity', 'weight_time': '1.5', 'weight_money': '0.8', 'weight_co2': '1.2', 'scenario_mode': 'partial_sharing', 'weather_profile': 'storm', 'weather_intensity': '1.25', 'departure_time_utc': '2026-03-21T10:15:00Z', 'stochastic_enabled': 'true', 'stochastic_samples': '77', 'search_budget': '7', 'evidence_budget': '3', 'world_count': '88', 'certificate_threshold': '0.87', 'tau_stop': '0.025', 'max_alternatives': '11', 'optimization_mode': 'robust'}",
            }
        ],
        seed=7,
        max_od=0,
    )

    overrides = rows[0]["row_overrides"]
    assert overrides["profile_id"] == "p1"
    assert overrides["corpus_group"] == "ambiguity"
    assert overrides["search_budget"] == "6"
    assert overrides["evidence_budget"] == "3"
    assert overrides["world_count"] == "88"
    assert overrides["certificate_threshold"] == "0.87"
    assert overrides["scenario_mode"] == "partial_sharing"
    assert overrides["weather_profile"] == "storm"
    assert overrides["stochastic_enabled"] == "true"


def test_variant_seed_uses_row_seed_independent_of_corpus_order() -> None:
    args = thesis_module._build_parser().parse_args(["--corpus-csv", "dummy.csv", "--seed", "7"])
    row_a = {
        "od_id": "od-a",
        "origin_lat": 52.0,
        "origin_lon": -1.5,
        "destination_lat": 51.5,
        "destination_lon": -1.2,
        "distance_bin": "30-100 km",
        "corpus_kind": "ambiguity",
        "profile_id": "p1",
        "seed": 99,
    }
    row_b = {
        "od_id": "od-b",
        "origin_lat": 53.0,
        "origin_lon": -2.5,
        "destination_lat": 54.5,
        "destination_lon": -1.2,
        "distance_bin": "100-250 km",
        "corpus_kind": "ambiguity",
        "profile_id": "p2",
        "seed": 123,
    }
    ordered = thesis_module._load_rows([row_a, row_b], seed=7, max_od=0)
    reversed_rows = thesis_module._load_rows([row_b, row_a], seed=7, max_od=0)

    ordered_seed_a = thesis_module._variant_seed(args, ordered[0], od_index=0, variant_id="C")
    reversed_seed_a = thesis_module._variant_seed(args, reversed_rows[1], od_index=1, variant_id="C")
    ordered_seed_b = thesis_module._variant_seed(args, ordered[1], od_index=1, variant_id="C")

    assert ordered_seed_a == reversed_seed_a
    assert ordered_seed_a != ordered[0]["seed"]
    assert ordered_seed_a != ordered_seed_b


def test_od_ambiguity_prior_stays_raw_when_support_signals_are_weak() -> None:
    prior = thesis_module._od_ambiguity_prior(
        {
            "od_ambiguity_index": 0.29,
            "candidate_probe_engine_disagreement_prior": 0.33,
            "hard_case_prior": 0.31,
            "od_ambiguity_support_ratio": 0.09,
            "od_ambiguity_source_entropy": 0.08,
            "od_ambiguity_source_count": 1,
            "od_ambiguity_source_mix_count": 1,
            "od_ambiguity_prior_strength": 0.24,
            "od_ambiguity_family_density": 0.12,
            "od_ambiguity_margin_pressure": 0.17,
            "od_ambiguity_spread_pressure": 0.11,
            "od_ambiguity_toll_instability": 0.07,
        }
    )

    assert prior == pytest.approx(0.29, rel=0.0, abs=1e-6)


def test_row_overrides_take_precedence_in_effective_request_config(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_override_corpus_csv(corpus_csv)
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            str(corpus_csv),
            "--search-budget",
            "4",
            "--evidence-budget",
            "2",
            "--world-count",
            "64",
            "--certificate-threshold",
            "0.80",
            "--tau-stop",
            "0.02",
            "--max-alternatives",
            "8",
            "--scenario-mode",
            "no_sharing",
            "--departure-time-utc",
            "2026-03-21T08:00:00Z",
            "--stochastic-enabled",
            "--stochastic-samples",
            "25",
            "--optimization-mode",
            "expected_value",
            "--weight-time",
            "1.0",
            "--weight-money",
            "1.0",
            "--weight-co2",
            "1.0",
        ]
    )
    loaded = thesis_module._load_rows(thesis_module.load_corpus(str(corpus_csv)), seed=7, max_od=0)
    cfg = thesis_module._effective_request_config(args, loaded[0], variant_seed=13)

    assert cfg["profile_id"] == "profile_override_a"
    assert cfg["corpus_group"] == "ambiguity"
    assert cfg["scenario_mode"] == "partial_sharing"
    assert cfg["weather"] == {"enabled": True, "profile": "storm", "intensity": 1.25}
    assert cfg["departure_time_utc"] == "2026-03-21T10:15:00+00:00"
    assert cfg["stochastic"]["enabled"] is True
    assert cfg["stochastic"]["samples"] == 77
    assert cfg["search_budget"] == 7
    assert cfg["evidence_budget"] == 3
    assert cfg["world_count"] == 88
    assert cfg["certificate_threshold"] == 0.87
    assert cfg["tau_stop"] == 0.025
    assert cfg["max_alternatives"] == 11
    assert cfg["optimization_mode"] == "robust"
    assert cfg["weights"] == {"time": 1.5, "money": 0.8, "co2": 1.2}
    assert cfg["row_override_count"] >= 10
    assert "scenario_mode" in cfg["row_override_keys"]
    assert "weather_profile" in cfg["row_override_keys"]
    assert "weather_intensity" in cfg["row_override_keys"]


def test_effective_request_config_adapts_budgets_from_corpus_ambiguity_prior() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--search-budget",
            "4",
            "--evidence-budget",
            "2",
            "--world-count",
            "64",
            "--max-alternatives",
            "8",
        ]
    )
    low_cfg = thesis_module._effective_request_config(
        args,
        {
            "od_id": "low",
            "origin_lat": 51.5,
            "origin_lon": -0.1,
            "destination_lat": 52.5,
            "destination_lon": -1.0,
            "od_ambiguity_index": 0.12,
            "candidate_probe_path_count": 2,
            "candidate_probe_corridor_family_count": 1,
            "candidate_probe_objective_spread": 0.08,
        },
        variant_seed=7,
    )
    high_cfg = thesis_module._effective_request_config(
        args,
        {
            "od_id": "high",
            "origin_lat": 51.5,
            "origin_lon": -0.1,
            "destination_lat": 52.5,
            "destination_lon": -1.0,
            "od_ambiguity_index": 0.82,
            "candidate_probe_path_count": 5,
            "candidate_probe_corridor_family_count": 3,
            "candidate_probe_objective_spread": 0.28,
        },
        variant_seed=8,
    )

    assert low_cfg["ambiguity_budget_band"] == "low"
    assert low_cfg["search_budget"] == 2
    assert low_cfg["evidence_budget"] == 1
    assert low_cfg["world_count"] == 24
    assert high_cfg["ambiguity_budget_band"] == "high"
    assert high_cfg["search_budget"] == 6
    assert high_cfg["evidence_budget"] == 2
    assert high_cfg["world_count"] == 96


def test_budget_schedule_escalates_on_hard_case_prior_even_when_od_ambiguity_is_moderate() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--search-budget",
            "4",
            "--evidence-budget",
            "1",
            "--world-count",
            "64",
            "--max-alternatives",
            "8",
        ]
    )
    cfg = thesis_module._effective_request_config(
        args,
        {
            "od_id": "prior-heavy",
            "origin_lat": 51.5,
            "origin_lon": -0.1,
            "destination_lat": 52.5,
            "destination_lon": -1.0,
            "od_ambiguity_index": 0.44,
            "candidate_probe_engine_disagreement_prior": 0.58,
            "hard_case_prior": 0.73,
            "candidate_probe_path_count": 3,
            "candidate_probe_corridor_family_count": 2,
            "candidate_probe_objective_spread": 0.18,
        },
        variant_seed=9,
    )

    assert cfg["ambiguity_budget_band"] == "high"
    assert cfg["search_budget"] == 6
    assert cfg["evidence_budget"] == 2
    assert cfg["world_count"] == 96


def test_certificate_map_accepts_dict_shaped_route_certificates() -> None:
    certificate_summary = {
        "selected_route_id": "route_0",
        "selected_certificate": 0.0,
        "route_certificates": {
            "route_0": 0.0,
            "route_10": 1.0,
        },
    }

    assert thesis_module._certificate_map(certificate_summary) == {
        "route_0": 0.0,
        "route_10": 1.0,
    }


def _route_payload(route_id: str, *, duration_s: float, cost: float, emissions: float) -> dict[str, object]:
    return {
        "id": route_id,
        "metrics": {
            "distance_km": 10.0,
            "duration_s": duration_s,
            "monetary_cost": cost,
            "emissions_kg": emissions,
        },
        "uncertainty": {
            "p95_duration_s": duration_s + 12.0,
            "cvar95_duration_s": duration_s + 18.0,
        },
        "evidence_provenance": {
            "active_families": ["scenario", "toll", "terrain"],
            "families": [
                {
                    "family": "scenario",
                    "source": "live_scenario",
                    "active": True,
                    "freshness_timestamp_utc": "2026-03-20T00:00:00Z",
                    "max_age_minutes": 60.0,
                    "signature": f"{route_id}-scenario",
                    "confidence": 0.99,
                    "coverage_ratio": 1.0,
                    "fallback_used": False,
                    "fallback_source": "",
                    "details": {"source_mode": "live"},
                },
                {
                    "family": "toll",
                    "source": "live_toll",
                    "active": True,
                    "freshness_timestamp_utc": "2026-03-20T00:00:00Z",
                    "max_age_minutes": 60.0,
                    "signature": f"{route_id}-toll",
                    "confidence": 0.97,
                    "coverage_ratio": 1.0,
                    "fallback_used": False,
                    "fallback_source": "",
                    "details": {"source_mode": "live"},
                },
                {
                    "family": "terrain",
                    "source": "live_terrain",
                    "active": True,
                    "freshness_timestamp_utc": "2026-03-20T00:00:00Z",
                    "max_age_minutes": 60.0,
                    "signature": f"{route_id}-terrain",
                    "confidence": 0.96,
                    "coverage_ratio": 1.0,
                    "fallback_used": False,
                    "fallback_source": "",
                    "details": {"source_mode": "live"},
                },
            ],
        },
    }


def _artifact_bundle(run_id: str, *, selected_id: str, selected_certificate: float | None, pipeline_mode: str) -> dict[str, object]:
    frontier = [
        {
            "route_id": selected_id,
            "duration_s": 100.0,
            "monetary_cost": 20.0,
            "emissions_kg": 5.0,
        },
        {
            "route_id": "r-alt",
            "duration_s": 102.0,
            "monetary_cost": 19.0,
            "emissions_kg": 5.4,
        },
    ]
    bundle: dict[str, object] = {
        "metadata.json": {"run_id": run_id, "pipeline_mode": pipeline_mode},
        "strict_frontier.jsonl": frontier,
        "final_route_trace.json": {
            "stage_timings_ms": {
                "dccs_ms": 8.0 if pipeline_mode != "legacy" else 0.0,
                "refc_ms": 4.0 if pipeline_mode in {"dccs_refc", "voi"} else 0.0,
                "voi_ms": 6.0 if pipeline_mode == "voi" else 0.0,
                "pareto_ms": 3.0,
                "refinement_ms": 21.0,
                "k_raw_ms": 35.0 if pipeline_mode != "legacy" else 0.0,
                "supplemental_rescue_ms": 5.5 if pipeline_mode != "legacy" else 0.0,
            },
            "diversity_rescue": {
                "collapse_detected": pipeline_mode != "legacy",
                "collapse_reason": "single_frontier_after_diverse_k_raw" if pipeline_mode != "legacy" else "",
                "raw_corridor_family_count": 3 if pipeline_mode != "legacy" else 0,
                "refined_corridor_family_count_after": 2 if pipeline_mode != "legacy" else 0,
                "supplemental_challenger_activated": pipeline_mode in {"dccs", "dccs_refc", "voi"},
                "supplemental_sources": ["osrm", "ors_local"] if pipeline_mode != "legacy" else [],
                "supplemental_candidate_count": 2 if pipeline_mode != "legacy" else 0,
                "supplemental_selected_count": 1 if pipeline_mode != "legacy" else 0,
                "supplemental_budget_used": 1 if pipeline_mode != "legacy" else 0,
            },
            "candidate_diagnostics": {
                "precheck_elapsed_ms": 2.5,
                "selected_candidate_count": 2 if pipeline_mode != "legacy" else 0,
                "graph_search_ms_initial": 12.5 if pipeline_mode != "legacy" else 0.0,
                "graph_search_ms_retry": 9.5 if pipeline_mode != "legacy" else 0.0,
                "graph_search_ms_rescue": 13.0 if pipeline_mode != "legacy" else 0.0,
                "graph_k_raw_cache_hit": pipeline_mode == "voi",
                "graph_low_ambiguity_fast_path": False,
                "graph_supported_ambiguity_fast_fallback": pipeline_mode != "legacy",
            },
            "option_build_runtime": {
                "cache_hits": 2 if pipeline_mode == "voi" else 0,
                "cache_misses": 1 if pipeline_mode != "legacy" else 0,
                "rebuild_count": 1 if pipeline_mode != "legacy" else 0,
                "reuse_rate": 0.666667 if pipeline_mode == "voi" else 0.0,
            },
            "voi_dccs_runtime": {
                "cache_hits": 2 if pipeline_mode == "voi" else 0,
                "cache_misses": 1 if pipeline_mode == "voi" else 0,
                "reuse_rate": 0.666667 if pipeline_mode == "voi" else 0.0,
            },
            "selected_candidate_ids": ["supplemental:ors_local_seed:cand_d"] if pipeline_mode != "legacy" else [],
            "certification_runtime": {
                "cache_hits": 1 if pipeline_mode in {"dccs_refc", "voi"} else 0,
                "cache_misses": 1 if pipeline_mode in {"dccs_refc", "voi"} else 0,
                "shortcut_count": 1 if pipeline_mode == "dccs_refc" else 0,
            },
        },
    }
    if pipeline_mode != "legacy":
        bundle["dccs_summary.json"] = {
            "candidate_count_raw": 5,
            "refined_count": 2,
            "selected_candidate_ids": ["supplemental:ors_local_seed:cand_d"],
            "selected_candidate_source_label": "supplemental:local_ors:osrm_realized_seed",
            "selected_candidate_source_engine": "ors_local_seed",
            "selected_candidate_source_stage": "supplemental_diversity_rescue",
            "selected_from_supplemental_rescue": True,
            "dc_yield": 0.5,
            "challenger_hit_rate": 1.0,
            "frontier_gain_per_refinement": 0.5,
            "decision_flips": 1,
            "search_budget_used": 2,
            "diversity_collapse_detected": True,
            "diversity_collapse_reason": "single_frontier_after_diverse_k_raw",
            "raw_corridor_family_count": 3,
            "refined_corridor_family_count_before": 1,
            "refined_corridor_family_count_after": 2,
            "supplemental_challenger_activated": True,
            "supplemental_candidate_count": 2,
            "supplemental_selected_count": 1,
            "supplemental_budget_used": 1,
        }
        bundle["dccs_candidates.jsonl"] = [
            {"candidate_id": "cand_a", "final_score": 0.9, "decision_reason": "frontier_addition", "corridor_family": "north"},
            {"candidate_id": "cand_b", "final_score": 0.7, "decision_reason": "decision_flip", "corridor_family": "west"},
            {"candidate_id": "cand_c", "final_score": 0.2, "decision_reason": "redundant", "corridor_family": "north"},
            {
                "candidate_id": "supplemental:ors_local_seed:cand_d",
                "final_score": 0.95,
                "decision_reason": "frontier_addition",
                "corridor_family": "south",
                "supplemental_diversity_rescue": True,
                "candidate_source_engine": "ors_local_seed",
                "candidate_source_stage": "supplemental_diversity_rescue",
                "candidate_source_label": "supplemental:local_ors:osrm_realized_seed",
            },
        ]
    if pipeline_mode in {"dccs_refc", "voi"}:
        bundle["certificate_summary.json"] = {
            "selected_route_id": selected_id,
            "selected_certificate": selected_certificate,
            "certified": bool(selected_certificate and selected_certificate >= 0.8),
            "world_count": 12,
            "active_families": ["scenario", "toll", "terrain"],
            "route_certificates": [
                {"route_id": selected_id, "certificate": selected_certificate},
                {"route_id": "r-alt", "certificate": 0.62},
            ],
        }
        bundle["sampled_world_manifest.json"] = {
            "world_count": 12,
            "unique_world_count": 9,
            "seed": 7,
            "hard_case_stress_pack_count": 3,
            "stress_world_fraction": 0.25,
            "worlds": [{"world_id": "w-001"}],
        }
        bundle["initial_certificate_summary.json"] = {
            "selected_route_id": selected_id,
            "selected_certificate": selected_certificate,
            "certified": bool(selected_certificate and selected_certificate >= 0.8),
            "world_count": 12,
            "active_families": ["scenario", "toll", "terrain"],
            "route_certificates": [
                {"route_id": selected_id, "certificate": selected_certificate},
                {"route_id": "r-alt", "certificate": 0.58},
            ],
        }
        bundle["initial_sampled_world_manifest.json"] = {
            "world_count": 12,
            "unique_world_count": 9,
            "seed": 7,
            "hard_case_stress_pack_count": 3,
            "stress_world_fraction": 0.25,
            "worlds": [{"world_id": "w-001"}],
        }
        bundle["initial_route_fragility_map.json"] = {
            selected_id: {"scenario": 0.4, "toll": 0.08, "terrain": 0.16}
        }
        bundle["initial_competitor_fragility_breakdown.json"] = {
            selected_id: {"r-alt": {"scenario": 3, "terrain": 1}}
        }
        bundle["initial_value_of_refresh.json"] = {
            "ranking": [
                {"family": "scenario", "vor": 0.14},
                {"family": "terrain", "vor": 0.06},
            ]
        }
        bundle["route_fragility_map.json"] = {
            selected_id: {"scenario": 0.3, "toll": 0.1, "terrain": 0.2}
        }
        bundle["competitor_fragility_breakdown.json"] = {
            selected_id: {"r-alt": {"scenario": 2, "terrain": 1}}
        }
        bundle["value_of_refresh.json"] = {
            "ranking": [
                {"family": "scenario", "vor": 0.11},
                {"family": "terrain", "vor": 0.04},
            ],
            "controller_ranking_basis": (
                "raw_refresh_gain_fallback" if pipeline_mode == "voi" else "empirical_vor"
            ),
            "controller_ranking": (
                [
                    {
                        "family": "fuel",
                        "controller_score": 0.19,
                        "empirical_vor": 0.0,
                        "raw_refresh_gain": 0.19,
                        "basis": "raw_refresh_gain_fallback",
                    },
                    {
                        "family": "scenario",
                        "controller_score": 0.11,
                        "empirical_vor": 0.11,
                        "raw_refresh_gain": 0.11,
                        "basis": "raw_refresh_gain_fallback",
                    },
                ]
                if pipeline_mode == "voi"
                else [
                    {
                        "family": "scenario",
                        "controller_score": 0.11,
                        "empirical_vor": 0.11,
                        "raw_refresh_gain": 0.11,
                        "basis": "empirical_vor",
                    },
                    {
                        "family": "terrain",
                        "controller_score": 0.04,
                        "empirical_vor": 0.04,
                        "raw_refresh_gain": 0.04,
                        "basis": "empirical_vor",
                    },
                ]
            ),
            "top_refresh_family_controller": "fuel" if pipeline_mode == "voi" else "scenario",
            "top_refresh_gain_controller": 0.19 if pipeline_mode == "voi" else 0.11,
        }
    if pipeline_mode == "voi":
        bundle["voi_action_trace.json"] = {
            "actions": [
                {"iteration": 0, "selected_route_id": "r-alt", "chosen_action": {"kind": "refine_top1_dccs"}},
                {"iteration": 1, "selected_route_id": "r-alt", "chosen_action": {"kind": "refresh_top1_vor"}},
                {"iteration": 2, "selected_route_id": selected_id, "chosen_action": {"kind": "increase_stochastic_samples"}},
            ]
        }
        bundle["voi_controller_state.jsonl"] = [
            {
                "iteration_index": 0,
                "search_completeness_score": 0.74,
                "search_completeness_gap": 0.11,
                "prior_support_strength": 0.67,
                "pending_challenger_mass": 0.28,
                "best_pending_flip_probability": 0.32,
                "corridor_family_recall": 0.8,
                "frontier_recall_at_budget": 0.76,
                "top_refresh_gain": 0.045,
                "top_fragility_mass": 0.018,
                "competitor_pressure": 0.62,
                "credible_search_uncertainty": True,
                "credible_evidence_uncertainty": True,
            }
        ]
        bundle["voi_stop_certificate.json"] = {
            "stop_reason": "certified",
            "iteration_count": 3,
            "search_budget_used": 2,
            "evidence_budget_used": 1,
            "best_rejected_action": "refresh:terrain",
            "best_rejected_q": 0.03,
        }
    return bundle


def test_build_od_corpus_is_deterministic_and_tracks_corridors(monkeypatch) -> None:
    pairs = [
        ({"lat": 51.0, "lon": -1.0}, {"lat": 51.1, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 52.0, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 53.0, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 56.5, "lon": -1.0}),
    ]
    index = {"value": 0}

    def fake_sample_candidate_pair(rng, bbox):  # noqa: ARG001
        pair = pairs[index["value"] % len(pairs)]
        index["value"] += 1
        return pair

    def fake_feasibility_fn(**kwargs):  # noqa: ARG001
        return {
            "ok": True,
            "origin_node_id": "o",
            "destination_node_id": "d",
            "origin_nearest_distance_m": 12.0,
            "destination_nearest_distance_m": 13.0,
            "message": "ok",
        }

    monkeypatch.setattr(corpus_module, "_sample_candidate_pair", fake_sample_candidate_pair)
    monkeypatch.setattr(corpus_module, "route_graph_od_feasibility", fake_feasibility_fn)

    bbox = corpus_module.UKBBox(south=50.0, north=57.0, west=-2.0, east=1.0)
    first = corpus_module.build_od_corpus(
        seed=123,
        pair_count=4,
        bbox=bbox,
        max_attempts=8,
        feasibility_fn=fake_feasibility_fn,
        candidate_probe_fn=_candidate_probe_ok,
    )
    index["value"] = 0
    second = corpus_module.build_od_corpus(
        seed=123,
        pair_count=4,
        bbox=bbox,
        max_attempts=8,
        feasibility_fn=fake_feasibility_fn,
        candidate_probe_fn=_candidate_probe_ok,
    )

    assert first["corpus_hash"] == second["corpus_hash"]
    assert first["rows"] == second["rows"]
    assert first["accepted_count"] == len(first["rows"])
    assert first["accepted_count"] >= 3
    assert first["accepted_by_corridor"]
    assert all("corridor_bucket" in row for row in first["rows"])
    assert all(row["acceptance_mode"] == "graph_candidates" for row in first["rows"])
    assert all(row["candidate_probe_accepted"] for row in first["rows"])


def test_build_dual_od_corpora_surfaces_ambiguity_features_and_two_corpora(monkeypatch) -> None:
    pairs = [
        ({"lat": 51.0, "lon": -1.0}, {"lat": 51.1, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 52.0, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 53.0, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 56.5, "lon": -1.0}),
    ]
    index = {"value": 0}

    def fake_sample_candidate_pair(rng, bbox):  # noqa: ARG001
        pair = pairs[index["value"] % len(pairs)]
        index["value"] += 1
        return pair

    def fake_feasibility_fn(**kwargs):  # noqa: ARG001
        return {
            "ok": True,
            "origin_node_id": "o",
            "destination_node_id": "d",
            "origin_nearest_distance_m": 7.0,
            "destination_nearest_distance_m": 9.0,
            "message": "ok",
        }

    monkeypatch.setattr(corpus_module, "_sample_candidate_pair", fake_sample_candidate_pair)
    monkeypatch.setattr(corpus_module, "route_graph_od_feasibility", fake_feasibility_fn)

    bundle = corpus_module.build_dual_od_corpora(
        seed=123,
        representative_count=3,
        ambiguous_count=2,
        bbox=corpus_module.UKBBox(south=50.0, north=57.0, west=-2.0, east=1.0),
        max_attempts=16,
        feasibility_fn=fake_feasibility_fn,
        candidate_probe_fn=_candidate_probe_ambiguous,
        probe_max_paths=3,
    )

    representative = bundle["representative"]["rows"]
    ambiguous = bundle["ambiguous"]["rows"]
    assert representative and ambiguous
    assert all(row["corpus_kind"] == "representative" for row in representative)
    assert all(row["corpus_kind"] == "ambiguous" for row in ambiguous)
    assert all(row["ambiguity_index"] > 0.0 for row in representative + ambiguous)
    assert all(row["candidate_probe_corridor_family_count"] >= 2 for row in representative + ambiguous)
    assert bundle["representative"]["selection_policy"]
    assert bundle["ambiguous"]["ambiguity_feature_stats"]["mean_ambiguity_index"] >= bundle["representative"]["ambiguity_feature_stats"]["mean_ambiguity_index"]


def test_thesis_broad_corpus_matches_current_cohort_union() -> None:
    data_dir = thesis_module.PROJECT_ROOT / "data" / "eval"
    with (data_dir / "uk_od_corpus_representative_expanded.csv").open("r", encoding="utf-8", newline="") as f:
        representative_rows = list(csv.DictReader(f))
    with (data_dir / "uk_od_corpus_thesis_broad.csv").open("r", encoding="utf-8", newline="") as f:
        broad_rows = list(csv.DictReader(f))

    assert len(broad_rows) == 19
    assert {row["corpus_group"] for row in broad_rows} == {"representative", "ambiguity"}
    assert len({row["od_id"] for row in broad_rows}) == len(broad_rows)
    representative_ids = {row["od_id"] for row in representative_rows}
    broad_representative_rows = [row for row in broad_rows if row["corpus_group"] == "representative"]
    assert len(broad_representative_rows) == len(representative_rows)
    assert {row["od_id"] for row in broad_representative_rows} == representative_ids
    assert all(row.get("od_ambiguity_index") not in (None, "") for row in broad_rows)
    assert all(row.get("candidate_probe_engine_disagreement_prior") not in (None, "") for row in broad_rows)
    assert all(row.get("hard_case_prior") not in (None, "") for row in broad_rows)
    ambiguity_rows = [row for row in broad_rows if row["corpus_group"] == "ambiguity"]
    assert len(ambiguity_rows) == 11
    assert all(row["corpus_group"] == "ambiguity" for row in ambiguity_rows)
    assert {row["distance_bin"] for row in ambiguity_rows} >= {"30-100 km", "100-250 km", "250+ km"}
    assert any(str(row["profile_id"]).startswith("ambiguity_shorthaul_") for row in ambiguity_rows)


def test_is_hard_case_row_requires_observed_difficulty_not_ambiguity_label_only() -> None:
    easy_ambiguity = {
        "corpus_group": "ambiguity",
        "nontrivial_frontier": False,
        "frontier_count": 1,
        "selector_certificate_disagreement": False,
        "voi_controller_engaged": False,
        "voi_action_count": 0,
        "near_tie_mass": 0.0,
        "nominal_winner_margin": 0.8,
        "certificate_threshold": 0.8,
        "certificate": 0.92,
        "od_ambiguity_index": 0.3,
    }
    assert thesis_module._is_hard_case_row(easy_ambiguity) is False

    observed_hard_case = dict(easy_ambiguity)
    observed_hard_case["near_tie_mass"] = 0.2
    assert thesis_module._is_hard_case_row(observed_hard_case) is True


def test_controller_stress_row_requires_real_stress_not_any_controller_action() -> None:
    low_value_controller_row = {
        "voi_controller_engaged": True,
        "voi_action_count": 1,
        "unnecessary_voi_refine": True,
        "od_ambiguity_index": 0.12,
        "od_engine_disagreement_prior": 0.08,
        "od_hard_case_prior": 0.05,
        "observed_ambiguity_index": 0.04,
        "near_tie_mass": 0.01,
        "certificate": 1.0,
        "certificate_threshold": 0.8,
        "certificate_margin": 0.2,
        "selector_certificate_disagreement": False,
        "realized_diversity_collapse": False,
        "nontrivial_frontier": False,
        "frontier_count": 1,
        "nominal_winner_margin": 0.7,
    }
    assert thesis_module._is_controller_stress_row(low_value_controller_row) is False

    stressed_controller_row = dict(low_value_controller_row)
    stressed_controller_row.update(
        {
            "unnecessary_voi_refine": False,
            "od_hard_case_prior": 0.62,
            "near_tie_mass": 0.14,
            "observed_ambiguity_index": 0.16,
            "certificate_margin": 0.07,
        }
    )
    assert thesis_module._is_controller_stress_row(stressed_controller_row) is True


def test_controller_stress_row_counts_resolved_initial_stress_even_with_nonproductive_refine() -> None:
    resolved_controller_row = {
        "voi_controller_engaged": True,
        "voi_action_count": 3,
        "unnecessary_voi_refine": True,
        "od_ambiguity_index": 0.61,
        "od_engine_disagreement_prior": 0.38,
        "od_hard_case_prior": 0.66,
        "observed_ambiguity_index": 0.08,
        "near_tie_mass": 0.02,
        "certificate": 1.0,
        "certificate_threshold": 0.8,
        "certificate_margin": 0.2,
        "selector_certificate_disagreement": False,
        "realized_diversity_collapse": False,
        "nontrivial_frontier": False,
        "frontier_count": 1,
        "nominal_winner_margin": 0.35,
        "initial_certificate": 0.54,
        "initial_winner_fragility_nonzero": True,
        "initial_refc_top_vor_positive": True,
        "voi_realized_certificate_lift": 0.22,
        "time_to_certification_ms": 180.0,
    }

    assert thesis_module._is_controller_stress_row(resolved_controller_row) is True


def test_finalize_cross_variant_metrics_uses_full_row_identity_not_od_id_only() -> None:
    rows = thesis_module._finalize_cross_variant_metrics(
        [
            {
                "od_id": "shared",
                "profile_id": "representative_a",
                "corpus_group": "representative",
                "origin_lat": 51.0,
                "origin_lon": -1.0,
                "destination_lat": 52.0,
                "destination_lon": -2.0,
                "variant_id": "V0",
                "algorithm_runtime_ms": 120.0,
            },
            {
                "od_id": "shared",
                "profile_id": "representative_a",
                "corpus_group": "representative",
                "origin_lat": 51.0,
                "origin_lon": -1.0,
                "destination_lat": 52.0,
                "destination_lon": -2.0,
                "variant_id": "B",
                "certificate": 0.7,
                "frontier_hypervolume": 2.0,
            },
            {
                "od_id": "shared",
                "profile_id": "representative_a",
                "corpus_group": "representative",
                "origin_lat": 51.0,
                "origin_lon": -1.0,
                "destination_lat": 52.0,
                "destination_lon": -2.0,
                "variant_id": "C",
                "certificate": 0.9,
                "frontier_hypervolume": 2.3,
                "algorithm_runtime_ms": 100.0,
            },
            {
                "od_id": "shared",
                "profile_id": "ambiguity_b",
                "corpus_group": "ambiguity",
                "origin_lat": 53.0,
                "origin_lon": -1.5,
                "destination_lat": 54.0,
                "destination_lon": -2.5,
                "variant_id": "V0",
                "algorithm_runtime_ms": 90.0,
            },
            {
                "od_id": "shared",
                "profile_id": "ambiguity_b",
                "corpus_group": "ambiguity",
                "origin_lat": 53.0,
                "origin_lon": -1.5,
                "destination_lat": 54.0,
                "destination_lon": -2.5,
                "variant_id": "B",
                "certificate": 0.4,
                "frontier_hypervolume": 1.5,
            },
            {
                "od_id": "shared",
                "profile_id": "ambiguity_b",
                "corpus_group": "ambiguity",
                "origin_lat": 53.0,
                "origin_lon": -1.5,
                "destination_lat": 54.0,
                "destination_lon": -2.5,
                "variant_id": "C",
                "certificate": 0.55,
                "frontier_hypervolume": 1.7,
                "algorithm_runtime_ms": 95.0,
            },
        ]
    )

    representative_voi = next(row for row in rows if row["variant_id"] == "C" and row["profile_id"] == "representative_a")
    ambiguity_voi = next(row for row in rows if row["variant_id"] == "C" and row["profile_id"] == "ambiguity_b")
    assert representative_voi["voi_realized_certificate_lift"] == 0.2
    assert representative_voi["voi_realized_frontier_gain"] == 0.3
    assert representative_voi["voi_realized_runtime_delta_ms"] == -20.0
    assert ambiguity_voi["voi_realized_certificate_lift"] == 0.15
    assert ambiguity_voi["voi_realized_frontier_gain"] == 0.2
    assert ambiguity_voi["voi_realized_runtime_delta_ms"] == 5.0


def test_strict_failure_elimination_rate_is_derived_from_v0_failure_rows() -> None:
    rows = thesis_module._finalize_cross_variant_metrics(
        [
            {
                "od_id": "shared",
                "profile_id": "stress_case",
                "corpus_group": "ambiguity",
                "origin_lat": 51.0,
                "origin_lon": -1.0,
                "destination_lat": 52.0,
                "destination_lon": -2.0,
                "variant_id": "V0",
                "failure_reason": "routing_graph_no_path",
                "selected_duration_s": 120.0,
                "selected_monetary_cost": 24.0,
                "selected_emissions_kg": 6.0,
                "runtime_ms": 130.0,
                "algorithm_runtime_ms": 120.0,
            },
            {
                "od_id": "shared",
                "profile_id": "stress_case",
                "corpus_group": "ambiguity",
                "origin_lat": 51.0,
                "origin_lon": -1.0,
                "destination_lat": 52.0,
                "destination_lon": -2.0,
                "variant_id": "A",
                "failure_reason": "",
                "selected_duration_s": 110.0,
                "selected_monetary_cost": 21.0,
                "selected_emissions_kg": 5.4,
                "runtime_ms": 112.0,
                "algorithm_runtime_ms": 108.0,
            },
        ]
    )

    v0_row = next(row for row in rows if row["variant_id"] == "V0")
    a_row = next(row for row in rows if row["variant_id"] == "A")
    assert v0_row["strict_failure_eliminated"] is False
    assert a_row["strict_failure_eliminated"] is True

    summary_rows = thesis_module._summary_rows(rows)
    a_summary = next(row for row in summary_rows if row["variant_id"] == "A")
    assert a_summary["strict_failure_elimination_rate"] == 1.0


def test_build_od_corpus_main_writes_dual_repo_local_outputs(monkeypatch, tmp_path: Path) -> None:
    pairs = [
        ({"lat": 51.0, "lon": -1.0}, {"lat": 51.1, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 52.0, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 53.0, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 56.5, "lon": -1.0}),
    ]
    index = {"value": 0}

    def fake_sample_candidate_pair(rng, bbox):  # noqa: ARG001
        pair = pairs[index["value"] % len(pairs)]
        index["value"] += 1
        return pair

    def fake_feasibility_fn(**kwargs):  # noqa: ARG001
        return {
            "ok": True,
            "origin_node_id": "o",
            "destination_node_id": "d",
            "origin_nearest_distance_m": 7.0,
            "destination_nearest_distance_m": 9.0,
            "message": "ok",
        }

    monkeypatch.setattr(corpus_module, "_sample_candidate_pair", fake_sample_candidate_pair)
    monkeypatch.setattr(corpus_module, "route_graph_od_feasibility", fake_feasibility_fn)
    original_build_dual = corpus_module.build_dual_od_corpora
    monkeypatch.setattr(
        corpus_module,
        "build_dual_od_corpora",
        lambda **kwargs: original_build_dual(
            **kwargs,
            feasibility_fn=fake_feasibility_fn,
            candidate_probe_fn=_candidate_probe_ambiguous,
        ),
    )

    exit_code = corpus_module.main(
        [
            "--output-dir",
            str(tmp_path),
            "--pair-count",
            "3",
            "--ambiguous-pair-count",
            "2",
            "--max-attempts",
            "16",
            "--probe-max-paths",
            "3",
            "--allow-partial",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "uk_od_corpus_representative.csv").exists()
    assert (tmp_path / "uk_od_corpus_ambiguous.csv").exists()
    representative_summary = json.loads((tmp_path / "uk_od_corpus_representative.summary.json").read_text(encoding="utf-8"))
    ambiguous_summary = json.loads((tmp_path / "uk_od_corpus_ambiguous.summary.json").read_text(encoding="utf-8"))
    assert representative_summary["selection_policy"]
    assert ambiguous_summary["selection_policy"]


def test_build_od_corpus_rejected_rows_allow_missing_nearest_distances(monkeypatch) -> None:
    pairs = [
        ({"lat": 51.0, "lon": -1.0}, {"lat": 51.02, "lon": -1.01}),
        ({"lat": 52.0, "lon": -1.5}, {"lat": 52.5, "lon": -1.5}),
        ({"lat": 53.2, "lon": -2.1}, {"lat": 55.0, "lon": -2.2}),
        ({"lat": 54.5, "lon": -3.0}, {"lat": 57.0, "lon": -3.2}),
    ]
    index = {"value": 0}

    def fake_sample_candidate_pair(rng, bbox):  # noqa: ARG001
        pair = pairs[index["value"] % len(pairs)]
        index["value"] += 1
        return pair

    def fake_feasibility_fn(**kwargs):  # noqa: ARG001
        if kwargs["origin_lat"] < 52.0:
            return {
                "ok": False,
                "message": "snapping failed",
                "origin_nearest_distance_m": None,
                "destination_nearest_distance_m": None,
            }
        return {
            "ok": True,
            "origin_node_id": "o",
            "destination_node_id": "d",
            "origin_nearest_distance_m": 12.0,
            "destination_nearest_distance_m": 13.0,
            "message": "ok",
        }

    monkeypatch.setattr(corpus_module, "_sample_candidate_pair", fake_sample_candidate_pair)
    monkeypatch.setattr(corpus_module, "route_graph_od_feasibility", fake_feasibility_fn)

    bbox = corpus_module.UKBBox(south=50.0, north=57.0, west=-4.0, east=1.0)
    summary = corpus_module.build_od_corpus(
        seed=123,
        pair_count=2,
        bbox=bbox,
        max_attempts=6,
        feasibility_fn=fake_feasibility_fn,
        acceptance_mode="feasibility_only",
    )

    assert summary["accepted_count"] >= 1
    assert summary["rejected_count"] >= 1
    rejected = [row for row in summary["rejected_samples_preview"] if not row["accepted"]]
    assert rejected
    assert rejected[0]["origin_nearest_distance_m"] == 0.0
    assert rejected[0]["destination_nearest_distance_m"] == 0.0


def test_build_od_corpus_rejects_feasible_pairs_when_candidate_probe_fails(monkeypatch) -> None:
    pairs = [
        ({"lat": 51.0, "lon": -1.0}, {"lat": 51.2, "lon": -1.0}),
        ({"lat": 52.0, "lon": -1.5}, {"lat": 52.4, "lon": -1.3}),
        ({"lat": 53.2, "lon": -2.1}, {"lat": 53.9, "lon": -2.0}),
        ({"lat": 54.5, "lon": -3.0}, {"lat": 55.8, "lon": -3.2}),
    ]
    index = {"value": 0}

    def fake_sample_candidate_pair(rng, bbox):  # noqa: ARG001
        pair = pairs[index["value"] % len(pairs)]
        index["value"] += 1
        return pair

    def fake_feasibility_fn(**kwargs):  # noqa: ARG001
        return {
            "ok": True,
            "origin_node_id": "o",
            "destination_node_id": "d",
            "origin_nearest_distance_m": 8.0,
            "destination_nearest_distance_m": 11.0,
            "message": "ok",
        }

    monkeypatch.setattr(corpus_module, "_sample_candidate_pair", fake_sample_candidate_pair)
    monkeypatch.setattr(corpus_module, "route_graph_od_feasibility", fake_feasibility_fn)

    bbox = corpus_module.UKBBox(south=50.0, north=57.0, west=-4.0, east=1.0)
    summary = corpus_module.build_od_corpus(
        seed=123,
        pair_count=2,
        bbox=bbox,
        max_attempts=6,
        feasibility_fn=fake_feasibility_fn,
        candidate_probe_fn=_candidate_probe_fail,
        acceptance_mode="graph_candidates",
    )

    assert summary["accepted_count"] == 0
    assert summary["reject_stats"]["routing_graph_no_path"] >= 1
    preview = summary["rejected_samples_preview"][0]
    assert preview["candidate_probe_reason_code"] == "routing_graph_no_path"
    assert preview["candidate_probe_emitted_paths"] == 0
    assert preview["origin_node_id"] == "o"


def test_run_thesis_evaluation_executes_real_variant_matrix_and_writes_artifacts(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)

    route_calls: list[str] = []
    run_modes = {
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": "legacy",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb": "dccs",
        "cccccccc-cccc-cccc-cccc-cccccccccccc": "dccs_refc",
        "dddddddd-dddd-dddd-dddd-dddddddddddd": "voi",
    }
    runs = {
        "legacy": {
            "run_id": "11111111-1111-1111-1111-111111111111",
            "selected": _route_payload("legacy-route", duration_s=120.0, cost=24.0, emissions=6.2),
            "candidates": [
                _route_payload("legacy-route", duration_s=120.0, cost=24.0, emissions=6.2),
                _route_payload("legacy-alt", duration_s=122.0, cost=23.0, emissions=6.0),
            ],
            "selected_certificate": None,
            "compute_ms": 120.0,
        },
        "dccs": {
            "run_id": "22222222-2222-2222-2222-222222222222",
            "selected": _route_payload("dccs-route", duration_s=105.0, cost=21.0, emissions=5.3),
            "candidates": [
                _route_payload("dccs-route", duration_s=105.0, cost=21.0, emissions=5.3),
                _route_payload("dccs-alt", duration_s=108.0, cost=20.0, emissions=5.6),
            ],
            "selected_certificate": None,
            "compute_ms": 110.0,
        },
        "dccs_refc": {
            "run_id": "33333333-3333-3333-3333-333333333333",
            "selected": _route_payload("refc-route", duration_s=101.0, cost=20.5, emissions=5.1),
            "candidates": [
                _route_payload("refc-route", duration_s=101.0, cost=20.5, emissions=5.1),
                _route_payload("refc-alt", duration_s=103.0, cost=19.8, emissions=5.4),
            ],
            "selected_certificate": {"certificate": 0.75, "certified": False},
            "compute_ms": 104.0,
        },
        "voi": {
            "run_id": "44444444-4444-4444-4444-444444444444",
            "selected": _route_payload("voi-route", duration_s=98.0, cost=20.0, emissions=4.9),
            "candidates": [
                _route_payload("voi-route", duration_s=98.0, cost=20.0, emissions=4.9),
                _route_payload("voi-alt", duration_s=100.0, cost=19.6, emissions=5.2),
            ],
            "selected_certificate": {"certificate": 0.91, "certified": True},
            "compute_ms": 98.0,
        },
    }
    bundles = {
        value["run_id"]: _artifact_bundle(
            value["run_id"],
            selected_id=str(value["selected"]["id"]),
            selected_certificate=(value["selected_certificate"] or {}).get("certificate"),
            pipeline_mode=mode,
        )
        for mode, value in runs.items()
    }
    osrm_payload = {
        "baseline": _route_payload("osrm-base", duration_s=130.0, cost=18.0, emissions=5.5),
        "method": "osrm_quick_baseline",
        "compute_ms": 11.5,
    }
    ors_payload = {
        "baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3),
        "method": "ors_local_engine_baseline",
        "provider_mode": "local_service",
        "baseline_policy": "engine_shortest_path",
        "asset_manifest_hash": "sha256:ors-manifest",
        "asset_recorded_at": "2026-03-22T17:00:00+00:00",
        "asset_freshness_status": "graph_identity_verified",
        "engine_manifest": {
            "identity_status": "graph_identity_verified",
            "compose_image": "openrouteservice/openrouteservice:v9.7.1",
            "graph_listing_digest": "digest-123",
            "graph_file_count": 12,
            "graph_total_bytes": 937460812,
            "graph_build_info": {
                "graph_build_date": "2026-03-22T16:39:30+0000",
                "osm_date": "2026-02-23T21:21:28+0000",
            },
        },
        "compute_ms": 14.2,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            mode = json.loads(request.content.decode("utf-8"))["pipeline_mode"]
            route_calls.append(mode)
            payload = runs[mode]
            return httpx.Response(
                200,
                json={
                    "selected": payload["selected"],
                    "candidates": payload["candidates"],
                    "run_id": payload["run_id"],
                    "manifest_endpoint": f"/runs/{payload['run_id']}/manifest",
                    "artifacts_endpoint": f"/runs/{payload['run_id']}/artifacts",
                    "selected_certificate": payload["selected_certificate"],
                    "compute_ms": payload["compute_ms"],
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(200, json=osrm_payload)
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(200, json=ors_payload)
        if request.method == "GET" and "/runs/" in request.url.path and "/artifacts/" in request.url.path:
            parts = request.url.path.split("/")
            run_id = parts[2]
            artifact_name = parts[-1]
            artifact_payload = bundles[run_id].get(artifact_name)
            if artifact_payload is None:
                return httpx.Response(404, json={"detail": "missing"})
            if artifact_name.endswith(".jsonl"):
                content = "\n".join(json.dumps(row) for row in artifact_payload)
                return httpx.Response(200, text=content)
            return httpx.Response(200, json=artifact_payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--seed",
                "7",
                "--world-count",
                "12",
                "--certificate-threshold",
                "0.8",
                "--tau-stop",
                "0.01",
                "--ors-snapshot-mode",
                "record",
                "--ors-baseline-policy",
                "local_service",
            ]
        )
        payload = thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    assert route_calls == ["dccs", "dccs_refc", "voi", "legacy"]
    assert Path(payload["results_csv"]).exists()
    assert Path(payload["summary_csv"]).exists()
    assert Path(payload["methods_appendix"]).exists()
    assert Path(payload["thesis_report"]).exists()
    assert Path(payload["ors_snapshot_path"]).exists()
    assert Path(Path(payload["summary_csv"]).parent / "baseline_smoke_summary.json").exists()
    assert (Path(payload["summary_csv"]).parent / "cohort_composition.json").exists()
    assert payload["output_artifact_validation"]["validated_artifact_count"] >= 10
    assert payload["success_row_count"] == 4
    assert payload["failure_row_count"] == 0
    assert payload["successful_variants"] == ["V0", "A", "B", "C"]
    assert payload["failed_variants"] == []
    assert len(payload["rows"]) == 4
    voi_row = next(row for row in payload["rows"] if row["variant_id"] == "C")
    assert voi_row["pipeline_mode"] == "voi"
    assert voi_row["corpus_kind"] == "ambiguous"
    assert voi_row["od_ambiguity_index"] == 0.72
    assert voi_row["od_ambiguity_confidence"] == 0.81
    assert voi_row["od_ambiguity_source_count"] == 2
    assert voi_row["od_ambiguity_source_mix"] == '{"engine_augmented_probe":1,"routing_graph_probe":1}'
    assert voi_row["od_ambiguity_source_mix_count"] == 2
    assert voi_row["od_ambiguity_source_support"] == '{"engine_augmented_probe":0.75,"routing_graph_probe":0.82}'
    assert voi_row["od_ambiguity_source_support_strength"] == pytest.approx(0.785, rel=0.0, abs=1e-6)
    assert voi_row["od_ambiguity_source_entropy"] == 0.63
    assert voi_row["od_ambiguity_support_ratio"] == 0.79
    assert voi_row["od_ambiguity_prior_strength"] == 0.72
    assert voi_row["od_engine_disagreement_prior"] == 0.66
    assert voi_row["od_hard_case_prior"] == 0.72
    assert voi_row["cohort_label"] == "ambiguity"
    assert voi_row["hard_case"] is True
    assert voi_row["certificate"] == 0.91
    assert voi_row["certificate_margin"] == 0.11
    assert voi_row["certificate_runner_up_gap"] == 0.29
    assert voi_row["certificate_winner_route_id"] == "voi-route"
    assert voi_row["selector_certificate_disagreement"] is False
    assert voi_row["frontier_hypervolume"] > 0.0
    assert voi_row["nontrivial_frontier"] is True
    assert voi_row["time_to_best_iteration"] == 2
    assert voi_row["nominal_winner_margin"] == 1.0
    assert voi_row["near_tie_mass"] == 0.0
    assert voi_row["dccs_dc_yield"] == 0.5
    assert voi_row["dccs_frontier_recall_at_budget"] == pytest.approx(0.666667, rel=0.0, abs=1e-6)
    assert voi_row["dccs_corridor_family_recall"] == pytest.approx(0.666667, rel=0.0, abs=1e-6)
    assert voi_row["refc_top_refresh_family"] == "scenario"
    assert voi_row["controller_refresh_ranking_basis"] == "raw_refresh_gain_fallback"
    assert voi_row["controller_top_refresh_family"] == "fuel"
    assert voi_row["controller_top_refresh_gain"] == 0.19
    assert voi_row["controller_refresh_fallback_activated"] is True
    assert voi_row["controller_empirical_vs_raw_refresh_disagreement"] is True
    assert voi_row["fragility_entropy"] == 0.92062
    assert voi_row["competitor_turnover_rate"] == 0.333333
    assert voi_row["voi_action_count"] == 3
    assert voi_row["voi_controller_engaged"] is True
    assert voi_row["credible_evidence_uncertainty"] is True
    assert voi_row["supported_hard_case"] is True
    assert voi_row["evidence_first_engagement"] is False
    assert voi_row["evidence_only_engagement"] is False
    assert voi_row["first_controller_action_kind"] == "refine_top1_dccs"
    assert voi_row["ambiguity_budget_band"] == "high"
    assert voi_row["search_budget_utilization"] == pytest.approx(2.0 / 6.0, rel=0.0, abs=1e-6)
    assert voi_row["evidence_budget_utilization"] == 0.5
    assert voi_row["voi_realized_certificate_lift"] == 0.16
    assert voi_row["voi_realized_runtime_delta_ms"] == -22.0
    assert voi_row["weighted_win_osrm"] is True
    assert voi_row["weighted_win_v0"] is True
    assert voi_row["weighted_win_best_baseline"] is True
    assert voi_row["weighted_margin_vs_osrm"] > 0.0
    assert voi_row["weighted_margin_vs_best_baseline"] > 0.0
    assert voi_row["baseline_acquisition_runtime_ms"] == 25.7
    assert voi_row["algorithm_runtime_ms"] == 98.0
    assert voi_row["warmup_amortized_ms"] == pytest.approx(1304.5 / 4.0, rel=0.0, abs=1e-6)
    assert voi_row["quality_per_second"] is not None and voi_row["quality_per_second"] > 0.0
    assert voi_row["frontier_gain_per_ms"] is not None and voi_row["frontier_gain_per_ms"] > 0.0
    assert voi_row["certificate_gain_per_world"] == pytest.approx(0.11 / 12.0, rel=0.0, abs=1e-9)
    assert voi_row["cache_reuse_ratio"] is None
    assert voi_row["baseline_identity_verified"] is True
    assert voi_row["runtime_ratio_vs_osrm"] == pytest.approx(123.7 / 11.5, rel=0.0, abs=1e-6)
    assert voi_row["algorithm_runtime_ratio_vs_osrm"] == pytest.approx(98.0 / 11.5, rel=0.0, abs=1e-6)
    assert voi_row["runtime_gap_vs_osrm_ms"] == 112.2
    assert voi_row["runtime_gap_vs_ors_ms"] == 109.5
    assert voi_row["algorithm_runtime_gap_vs_osrm_ms"] == 86.5
    assert voi_row["algorithm_runtime_gap_vs_ors_ms"] == 83.8
    assert voi_row["backend_ready_wait_ms"] == 42.0
    assert voi_row["route_graph_warmup_elapsed_ms"] == 1250.0
    assert voi_row["stage_k_raw_ms"] == 35.0
    assert voi_row["stage_k_raw_graph_search_initial_ms"] == 0.0
    assert voi_row["stage_k_raw_graph_search_retry_ms"] == 0.0
    assert voi_row["stage_k_raw_graph_search_rescue_ms"] == 0.0
    assert voi_row["stage_k_raw_osrm_fallback_ms"] == 0.0
    assert voi_row["realized_diversity_collapse"] is True
    assert voi_row["supplemental_challenger_activated"] is True
    assert voi_row["supplemental_challenger_source_count"] == 2
    assert voi_row["supplemental_challenger_selected_count"] == 1
    assert voi_row["selected_candidate_source_engine"] == "ors_local_seed"
    assert voi_row["selected_candidate_source_stage"] == "supplemental_diversity_rescue"
    assert voi_row["selected_from_supplemental_rescue"] is True
    assert voi_row["selected_from_comparator_engine"] is True
    assert voi_row["stage_supplemental_rescue_ms"] == 5.5
    assert voi_row["controller_value_per_second"] is not None and voi_row["controller_value_per_second"] > 0.0
    assert voi_row["refc_stress_world_fraction"] == 0.25
    assert voi_row["voi_dccs_cache_hit_rate"] == pytest.approx(2.0 / 3.0, rel=0.0, abs=1e-6)
    assert voi_row["artifact_complete"] is True
    assert voi_row["artifact_status"] == "ok"
    assert voi_row["route_evidence_ok"] is True
    assert voi_row["route_evidence_status"] == "ok"
    summary_row = next(row for row in payload["summary_rows"] if row["variant_id"] == "C")
    assert summary_row["pipeline_mode"] == "voi"
    assert summary_row["success_count"] == 1
    assert summary_row["mean_certificate"] == 0.91
    assert summary_row["mean_certificate_denominator"] == 1
    assert summary_row["weighted_denominator_osrm"] == 1
    assert summary_row["weighted_denominator_v0"] == 1
    assert summary_row["weighted_denominator_best_baseline"] == 1
    assert summary_row["mean_od_ambiguity_index"] == 0.72
    assert summary_row["mean_od_ambiguity_confidence"] == 0.81
    assert summary_row["mean_od_ambiguity_source_count"] == 2.0
    assert summary_row["mean_od_ambiguity_source_mix_count"] == 2.0
    assert summary_row["mean_od_ambiguity_source_support_strength"] == pytest.approx(0.785, rel=0.0, abs=1e-6)
    assert summary_row["mean_od_ambiguity_source_entropy"] == 0.63
    assert summary_row["mean_od_ambiguity_support_ratio"] == 0.79
    assert summary_row["mean_od_ambiguity_prior_strength"] == 0.72
    assert summary_row["mean_od_engine_disagreement_prior"] == 0.66
    assert summary_row["mean_od_hard_case_prior"] == 0.72
    assert summary_row["mean_ambiguity_budget_prior"] == 0.72
    assert summary_row["mean_ambiguity_budget_prior_gap"] == 0.0
    assert summary_row["budget_prior_exceeds_raw_rate"] == 0.0
    assert summary_row["mean_frontier_count"] == 2.0
    assert summary_row["nontrivial_frontier_rate"] == 1.0
    assert summary_row["mean_certificate_margin"] == 0.11
    assert summary_row["selector_certificate_disagreement_rate"] == 0.0
    assert summary_row["mean_dccs_frontier_recall_at_budget"] == pytest.approx(0.666667, rel=0.0, abs=1e-6)
    assert summary_row["mean_top_refresh_gain"] == pytest.approx(0.045, rel=0.0, abs=1e-6)
    assert summary_row["mean_top_refresh_gain_denominator"] == 1
    assert summary_row["mean_top_fragility_mass"] == pytest.approx(0.018, rel=0.0, abs=1e-6)
    assert summary_row["mean_top_fragility_mass_denominator"] == 1
    assert summary_row["mean_competitor_pressure"] == pytest.approx(0.62, rel=0.0, abs=1e-6)
    assert summary_row["mean_competitor_pressure_denominator"] == 1
    assert summary_row["mean_initial_refc_top_vor"] == pytest.approx(0.14, rel=0.0, abs=1e-6)
    assert summary_row["mean_initial_refc_top_vor_denominator"] == 1
    assert summary_row["mean_final_refc_top_vor"] == pytest.approx(0.11, rel=0.0, abs=1e-6)
    assert summary_row["mean_final_refc_top_vor_denominator"] == 1
    assert summary_row["mean_initial_winner_fragility_mass"] == pytest.approx(0.4, rel=0.0, abs=1e-6)
    assert summary_row["mean_initial_winner_fragility_mass_denominator"] == 1
    assert summary_row["mean_final_winner_fragility_mass"] == pytest.approx(0.3, rel=0.0, abs=1e-6)
    assert summary_row["mean_final_winner_fragility_mass_denominator"] == 1
    assert summary_row["initial_winner_fragility_nonzero_rate"] == 1.0
    assert summary_row["initial_winner_fragility_nonzero_denominator"] == 1
    assert summary_row["winner_fragility_nonzero_rate"] == 1.0
    assert summary_row["winner_fragility_nonzero_denominator"] == 1
    assert summary_row["initial_refc_top_vor_positive_rate"] == 1.0
    assert summary_row["initial_refc_top_vor_positive_denominator"] == 1
    assert summary_row["refc_top_vor_positive_rate"] == 1.0
    assert summary_row["refc_top_vor_positive_denominator"] == 1
    assert summary_row["refresh_signal_persistence_rate"] is None
    assert summary_row["refresh_signal_persistence_denominator"] == 0
    assert summary_row["mean_voi_realized_certificate_lift"] == 0.16
    assert summary_row["voi_controller_engagement_rate"] == 1.0
    assert summary_row["mean_voi_action_count"] == 3.0
    assert summary_row["refine_cost_mape"] is None
    assert summary_row["refine_cost_mae_ms"] is None
    assert summary_row["refine_cost_rank_correlation"] is None
    assert summary_row["refine_cost_prediction_error_deprecated"] is None
    assert summary_row["refine_cost_sample_count"] == 0
    assert summary_row["refine_cost_positive_sample_count"] == 0
    assert summary_row["refine_cost_zero_observed_count"] == 0
    assert summary_row["zero_lift_controller_action_rate"] == 0.0
    assert summary_row["mean_search_budget_utilization"] == pytest.approx(2.0 / 6.0, rel=0.0, abs=1e-6)
    assert summary_row["mean_evidence_budget_utilization"] == 0.5
    assert summary_row["mean_algorithm_runtime_ms"] == 98.0
    assert summary_row["mean_algorithm_runtime_speedup_vs_v0"] == pytest.approx((120.0 - 98.0) / 120.0, rel=0.0, abs=1e-6)
    assert summary_row["mean_runtime_p50_ms"] == 123.7
    assert summary_row["mean_runtime_p90_ms"] == 123.7
    assert summary_row["mean_runtime_p95_ms"] == 123.7
    assert summary_row["mean_algorithm_runtime_p50_ms"] == 98.0
    assert summary_row["mean_algorithm_runtime_p90_ms"] == 98.0
    assert summary_row["mean_algorithm_runtime_p95_ms"] == 98.0
    assert summary_row["mean_baseline_acquisition_runtime_p90_ms"] == 25.7
    assert summary_row["mean_runtime_ratio_vs_osrm"] == pytest.approx(123.7 / 11.5, rel=0.0, abs=1e-6)
    assert summary_row["mean_algorithm_runtime_ratio_vs_osrm"] == pytest.approx(98.0 / 11.5, rel=0.0, abs=1e-6)
    assert summary_row["mean_runtime_gap_vs_osrm_ms"] == 112.2
    assert summary_row["mean_runtime_gap_vs_ors_ms"] == 109.5
    assert summary_row["mean_algorithm_runtime_gap_vs_osrm_ms"] == 86.5
    assert summary_row["mean_algorithm_runtime_gap_vs_ors_ms"] == 83.8
    assert summary_row["warmup_amortized_ms"] == pytest.approx(1304.5, rel=0.0, abs=1e-6)
    assert summary_row["realized_diversity_collapse_rate"] == 1.0
    assert summary_row["supplemental_challenger_activation_rate"] == 1.0
    assert summary_row["mean_supplemental_challenger_selected_count"] == 1.0
    assert summary_row["selected_from_supplemental_rescue_rate"] == 1.0
    assert summary_row["selected_from_comparator_engine_rate"] == 1.0
    assert summary_row["comparator_independence_rate"] == 0.0
    assert summary_row["strict_failure_elimination_rate"] == 0.0
    assert summary_row["mean_stage_supplemental_rescue_ms"] == 5.5
    assert summary_row["mean_stage_k_raw_ms"] == 35.0
    assert summary_row["mean_stage_k_raw_graph_search_initial_ms"] == 0.0
    assert summary_row["mean_stage_k_raw_graph_search_retry_ms"] == 0.0
    assert summary_row["mean_stage_k_raw_graph_search_rescue_ms"] == 0.0
    assert summary_row["mean_stage_k_raw_osrm_fallback_ms"] == 0.0
    assert summary_row["mean_k_raw_cache_hit_rate"] == 1.0
    assert summary_row["mean_graph_low_ambiguity_fast_path_rate"] == 0.0
    assert summary_row["mean_graph_supported_ambiguity_fast_fallback_rate"] == 1.0
    assert summary_row["mean_controller_value_per_second"] is not None and summary_row["mean_controller_value_per_second"] > 0.0
    assert summary_row["mean_quality_per_second"] is not None and summary_row["mean_quality_per_second"] > 0.0
    assert summary_row["mean_frontier_gain_per_ms"] is not None and summary_row["mean_frontier_gain_per_ms"] > 0.0
    assert summary_row["mean_certificate_gain_per_world"] == pytest.approx(0.009167, rel=0.0, abs=1e-6)
    assert summary_row["mean_cache_reuse_ratio"] is None
    assert summary_row["route_state_cache_hit_rate"] == pytest.approx(0.666667, rel=0.0, abs=1e-6)
    assert summary_row["route_state_cache_hits"] == 2.0
    assert summary_row["route_state_cache_misses"] == 1.0
    assert summary_row["option_build_cache_savings_ms_per_row"] is None
    assert summary_row["baseline_identity_verified_rate"] == 1.0
    assert summary_row["mean_backend_ready_wait_ms"] == 42.0
    assert summary_row["mean_route_graph_warmup_elapsed_ms"] == 1250.0
    assert summary_row["weighted_win_rate_v0"] == 1.0
    assert summary_row["weighted_win_rate_best_baseline"] == 1.0
    assert summary_row["mean_weighted_margin_vs_best_baseline"] > 0.0
    assert summary_row["mean_refc_stress_world_fraction"] == 0.25
    assert summary_row["mean_search_budget_utilization_p90"] == pytest.approx(2.0 / 6.0, rel=0.0, abs=1e-6)
    assert summary_row["mean_evidence_budget_utilization_p90"] == 0.5
    assert summary_row["mean_voi_action_density"] == 1.0
    assert summary_row["credible_evidence_uncertainty_rate"] == 1.0
    assert summary_row["supported_hard_case_rate"] == 1.0
    assert summary_row["evidence_first_engagement_rate"] == 0.0
    assert summary_row["evidence_only_engagement_rate"] == 0.0
    assert summary_row["mean_voi_dccs_cache_hit_rate"] == pytest.approx(2.0 / 3.0, rel=0.0, abs=1e-6)
    assert summary_row["voi_dccs_cache_hit_rate"] == pytest.approx(2.0 / 3.0, rel=0.0, abs=1e-6)
    assert summary_row["voi_dccs_cache_hits"] == 2.0
    assert summary_row["voi_dccs_cache_misses"] == 1.0
    assert summary_row["option_build_cache_savings_ms_per_row"] is None
    assert summary_row["controller_activation_on_high_ambiguity_rate"] == 1.0
    assert summary_row["mean_hard_case_rate"] == 1.0
    assert summary_row["mean_hard_case_certificate"] == 0.91
    assert summary_row["mean_hard_case_runtime_ms"] == 123.7
    assert summary_row["mean_hard_case_controller_engagement_rate"] == 1.0
    assert summary_row["controller_stress_row_count"] == 1
    assert summary_row["scenario_profile_unavailable_rate"] == 0.0
    assert summary_row["strict_live_readiness_pass_rate"] == 1.0
    assert summary_row["evaluation_rerun_success_rate"] == 1.0
    assert summary_row["controller_refresh_fallback_activation_rate"] == 1.0
    assert summary_row["controller_empirical_vs_raw_refresh_disagreement_rate"] == 1.0
    assert summary_row["broad_hard_case_certificate_selectivity_rate"] == 1.0
    assert summary_row["broad_hard_case_evidence_first_engagement_rate"] == 0.0
    assert summary_row["broad_hard_case_productive_voi_action_rate"] == 0.0
    assert summary_row["broad_hard_case_refc_signal_presence_rate"] == 1.0
    assert summary_row["mean_certificate_gap_ambiguity_vs_representative"] is None
    assert summary_row["ambiguity_prior_realized_correlation"] is None
    assert summary_row["corpus_kind_counts_json"] == json.dumps({"ambiguous": 1}, sort_keys=True)
    assert summary_row["corpus_group_counts_json"] == json.dumps({"ambiguous": 1}, sort_keys=True)
    assert summary_row["profile_id_counts_json"] == json.dumps({}, sort_keys=True)
    manifest_payload = json.loads(Path(payload["evaluation_manifest"]).read_text(encoding="utf-8"))
    assert manifest_payload["run_validity"]["strict_live_readiness_pass_rate"] == 1.0
    assert manifest_payload["run_validity"]["evaluation_rerun_success_rate"] == 1.0
    assert manifest_payload["run_validity"]["scenario_profile_unavailable_rate"] == 0.0
    assert manifest_payload["corpus_source_format"] == "csv"
    assert manifest_payload["corpus_source_exists"] is True
    assert manifest_payload["corpus_source_path"].endswith(".csv")
    assert Path(manifest_payload["corpus_source_resolved_path"]).is_absolute()
    request_manifest_payload = json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))
    assert request_manifest_payload["request"]["evaluation"]["corpus_source_format"] == "csv"
    assert request_manifest_payload["request"]["evaluation"]["corpus_source_exists"] is True
    assert Path(payload["summary_by_cohort_csv"]).exists()
    cohort_rows = [row for row in payload["summary_by_cohort_rows"] if row["variant_id"] == "C"]
    assert {row["cohort_label"] for row in cohort_rows} == {"ambiguity", "controller_stress", "hard_case"}
    assert all(row["cohort_share_of_variant"] == 1.0 for row in cohort_rows)
    assert all(row["mean_top_refresh_gain"] == pytest.approx(0.045, rel=0.0, abs=1e-6) for row in cohort_rows)
    assert all(row["mean_top_refresh_gain_denominator"] == 1 for row in cohort_rows)
    assert all(row["mean_top_fragility_mass"] == pytest.approx(0.018, rel=0.0, abs=1e-6) for row in cohort_rows)
    assert all(row["mean_top_fragility_mass_denominator"] == 1 for row in cohort_rows)
    assert all(row["mean_competitor_pressure"] == pytest.approx(0.62, rel=0.0, abs=1e-6) for row in cohort_rows)
    assert all(row["mean_competitor_pressure_denominator"] == 1 for row in cohort_rows)
    report_text = Path(payload["thesis_report"]).read_text(encoding="utf-8")
    assert "Failure Breakdown" not in report_text
    assert "## Successful Variants" in report_text
    assert "## Failed Variants" in report_text
    assert "## Headline Wins" in report_text
    assert "## Baseline Smoke" in report_text
    assert "## Ambiguity Prior" in report_text
    assert "## Direct Vs V0" in report_text
    assert "## Hard-Case And Controller Stress" in report_text
    assert "## Metric Highlights" in report_text
    assert "## Runtime Distribution" in report_text
    assert "## Cohort Highlights" in report_text
    assert "mean_od_engine_disagreement_prior=0.66" in report_text
    assert "mean_od_hard_case_prior=0.72" in report_text
    assert "mean_od_ambiguity_confidence=0.81" in report_text
    assert "mean_od_ambiguity_support_ratio=0.79" in report_text
    assert "mean_od_ambiguity_source_mix_count=2.0" in report_text
    assert "weighted_win_osrm=1.0 (1/1)" in report_text
    assert "balanced_win_best_baseline=1.0 (1/1)" in report_text
    assert "time_preserving_win=1.0 (1/1)" in report_text
    assert "time_preserving_win_osrm=1.0 (1/1)" in report_text
    assert "nontrivial_frontier_rate=1.0" in report_text
    assert "voi_controller_engagement_rate=1.0" in report_text
    assert "credible_evidence_uncertainty_rate=1.0" in report_text
    assert "mean_voi_dccs_cache_hit_rate=0.666667" in report_text
    assert "voi_dccs_cache_hit_rate=0.666667" in report_text
    assert "realized_diversity_collapse_rate=1.0" in report_text
    assert "selected_from_comparator_engine_rate=1.0" in report_text
    assert "comparator_independence_rate=0.0" in report_text
    assert "strict_failure_elimination_rate=0.0" in report_text
    assert "mean_certificate_margin=0.11" in report_text
    assert "OSRM smoke: ok" in report_text
    assert "ORS smoke: ok" in report_text
    assert "runtime_p95_ms=123.7" in report_text
    assert "baseline_identity_verified_rate=1.0" in report_text
    assert "controller_value_per_second=" in report_text
    assert "voi_action_density=1.0" in report_text
    assert "supplemental_rescue_ms=5.5" in report_text
    assert "option_build_cache_savings_ms_per_row=None" in report_text
    assert "route_state_cache_hit_rate=0.666667" in report_text
    assert "refine_cost_mape=None" in report_text
    assert "refine_cost_sample_count=0" in report_text
    assert "refine_cost_positive_sample_count=0" in report_text
    assert "refine_cost_zero_observed_count=0" in report_text
    assert "zero_lift_controller_action_rate=0.0" in report_text
    assert "- none" in report_text


def test_run_thesis_evaluation_fails_fast_when_baseline_smoke_fails(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)

    route_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            route_calls.append("route")
            return httpx.Response(200, json={})
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(
                502,
                json={
                    "detail": {
                        "reason_code": "baseline_route_unavailable",
                        "message": "OSRM baseline route is unavailable.",
                    }
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(
                200,
                json={
                    "baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3),
                    "method": "ors_local_engine_baseline",
                    "provider_mode": "local_service",
                    "baseline_policy": "engine_shortest_path",
                    "asset_manifest_hash": "sha256:ors-manifest",
                    "asset_recorded_at": "2026-03-22T17:00:00+00:00",
                    "asset_freshness_status": "graph_identity_verified",
                    "engine_manifest": {
                        "identity_status": "graph_identity_verified",
                        "compose_image": "openrouteservice/openrouteservice:v9.7.1",
                        "graph_listing_digest": "digest-123",
                        "graph_file_count": 12,
                        "graph_total_bytes": 937460812,
                    },
                    "compute_ms": 14.2,
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--run-id",
                "smoke-fail",
                "--ors-baseline-policy",
                "local_service",
                "--ors-snapshot-mode",
                "off",
            ]
        )
        with pytest.raises(RuntimeError, match="baseline_smoke_failed"):
            thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    assert route_calls == []
    smoke_path = tmp_path / "out" / "artifacts" / "smoke-fail" / "baseline_smoke_summary.json"
    assert smoke_path.exists()


def test_thesis_evaluation_main_supports_in_process_backend(tmp_path: Path, monkeypatch) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)
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
    monkeypatch.setattr(thesis_module, "TestClient", lambda app: DummyClient())  # noqa: ARG005

    def fake_run(args, *, client=None):
        calls["client"] = client
        calls["in_process"] = bool(args.in_process_backend)
        return {"run_id": "inproc"}

    monkeypatch.setattr(thesis_module, "run_thesis_evaluation", fake_run)

    exit_code = thesis_module.main(
        [
            "--corpus-csv",
            str(corpus_csv),
            "--out-dir",
            str(tmp_path / "out"),
            "--in-process-backend",
        ]
    )

    assert exit_code == 0
    assert calls["entered"] is True
    assert calls["exited"] is True
    assert calls["client"] is not None
    assert calls["in_process"] is True


def test_run_thesis_evaluation_fails_closed_when_required_route_artifact_is_missing(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)

    route_calls: list[str] = []
    runs = {
        "legacy": {
            "run_id": "11111111-1111-1111-1111-111111111111",
            "selected": _route_payload("legacy-route", duration_s=120.0, cost=24.0, emissions=6.2),
        },
        "dccs": {
            "run_id": "22222222-2222-2222-2222-222222222222",
            "selected": _route_payload("dccs-route", duration_s=105.0, cost=21.0, emissions=5.3),
        },
        "dccs_refc": {
            "run_id": "33333333-3333-3333-3333-333333333333",
            "selected": _route_payload("refc-route", duration_s=101.0, cost=20.5, emissions=5.1),
        },
        "voi": {
            "run_id": "44444444-4444-4444-4444-444444444444",
            "selected": _route_payload("voi-route", duration_s=98.0, cost=20.0, emissions=4.9),
        },
    }
    bundles = {
        value["run_id"]: _artifact_bundle(
            value["run_id"],
            selected_id=str(value["selected"]["id"]),
            selected_certificate=0.9 if mode == "voi" else 0.75 if mode == "dccs_refc" else None,
            pipeline_mode=mode,
        )
        for mode, value in runs.items()
    }
    bundles[runs["dccs_refc"]["run_id"]].pop("certificate_summary.json", None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            mode = json.loads(request.content.decode("utf-8"))["pipeline_mode"]
            route_calls.append(mode)
            payload = runs[mode]
            return httpx.Response(
                200,
                json={
                    "selected": payload["selected"],
                    "candidates": [payload["selected"]],
                    "run_id": payload["run_id"],
                    "manifest_endpoint": f"/runs/{payload['run_id']}/manifest",
                    "artifacts_endpoint": f"/runs/{payload['run_id']}/artifacts",
                    "selected_certificate": {"certificate": 0.9, "certified": True} if mode == "voi" else None,
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(200, json={"baseline": _route_payload("osrm-base", duration_s=130.0, cost=18.0, emissions=5.5), "method": "osrm_quick_baseline", "compute_ms": 11.5})
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(200, json={"baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3), "method": "ors_repo_local_baseline", "provider_mode": "repo_local", "baseline_policy": "corridor_alternative", "compute_ms": 14.2})
        if request.method == "GET" and "/runs/" in request.url.path and "/artifacts/" in request.url.path:
            parts = request.url.path.split("/")
            run_id = parts[2]
            artifact_name = parts[-1]
            artifact_payload = bundles[run_id].get(artifact_name)
            if artifact_payload is None:
                return httpx.Response(404, json={"detail": "missing"})
            if artifact_name.endswith(".jsonl"):
                return httpx.Response(200, text="\n".join(json.dumps(row) for row in artifact_payload))
            return httpx.Response(200, json=artifact_payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--ors-baseline-policy",
                "repo_local",
            ]
        )
        payload = thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    assert route_calls == ["dccs", "dccs_refc", "voi", "legacy"]
    failing_row = next(row for row in payload["rows"] if row["variant_id"] == "B")
    assert str(failing_row["failure_reason"]).startswith("strict_artifact_missing")
    assert failing_row["artifact_complete"] is False
    assert failing_row["artifact_status"] == "failed"
    assert payload["failure_row_count"] >= 1
    assert "B" in payload["failed_variants"]
    summary_row = next(row for row in payload["summary_rows"] if row["variant_id"] == "B")
    assert summary_row["failure_count"] == 1
    assert summary_row["success_count"] == 0
    assert summary_row["mean_certificate_denominator"] == 0
    report_text = Path(payload["thesis_report"]).read_text(encoding="utf-8")
    assert "## Failed Variants" in report_text
    assert "B / dccs_refc" in report_text
    assert "strict_artifact_missing" in report_text


def test_run_thesis_evaluation_replays_ors_snapshot_without_live_ors(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)
    replay_args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            str(corpus_csv),
            "--out-dir",
            str(tmp_path / "out"),
            "--backend-url",
            "http://testserver",
            "--seed",
            "7",
            "--world-count",
            "12",
            "--ors-snapshot-mode",
            "replay",
            "--ors-baseline-policy",
            "repo_local",
        ]
    )
    replay_od = thesis_module._load_rows(thesis_module.load_corpus(str(corpus_csv)), seed=7, max_od=0)[0]
    replay_request_config = thesis_module._effective_request_config(replay_args, replay_od, variant_seed=7)
    replay_baseline_payload = thesis_module._baseline_payload(
        replay_args,
        replay_od,
        request_config=replay_request_config,
        variant_seed=7,
    )

    record_snapshot = {
        "schema_version": "1.0.0",
        "routes": {
            "od-000001": {
                "request_hash": thesis_module._digest(replay_baseline_payload),
                "provider_method": "ors_repo_local_baseline",
                "provider_mode": "repo_local",
                "baseline_policy": "corridor_alternative",
                "recorded_at": "2026-03-20T00:00:00Z",
                "compute_ms": 14.2,
                "response_hash": thesis_module._digest(
                    {
                        "baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3),
                        "method": "ors_repo_local_baseline",
                    }
                ),
                "response": {
                    "baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3),
                    "method": "ors_repo_local_baseline",
                    "provider_mode": "repo_local",
                    "baseline_policy": "corridor_alternative",
                },
            }
        },
    }
    snapshot_path = tmp_path / "ors_snapshot.json"
    snapshot_path.write_text(json.dumps(record_snapshot), encoding="utf-8")

    bundles = {
        "55555555-5555-5555-5555-555555555555": _artifact_bundle(
            "55555555-5555-5555-5555-555555555555",
            selected_id="legacy-route",
            selected_certificate=None,
            pipeline_mode="legacy",
        ),
        "66666666-6666-6666-6666-666666666666": _artifact_bundle(
            "66666666-6666-6666-6666-666666666666",
            selected_id="dccs-route",
            selected_certificate=None,
            pipeline_mode="dccs",
        ),
        "77777777-7777-7777-7777-777777777777": _artifact_bundle(
            "77777777-7777-7777-7777-777777777777",
            selected_id="refc-route",
            selected_certificate=0.76,
            pipeline_mode="dccs_refc",
        ),
        "88888888-8888-8888-8888-888888888888": _artifact_bundle(
            "88888888-8888-8888-8888-888888888888",
            selected_id="voi-route",
            selected_certificate=0.88,
            pipeline_mode="voi",
        ),
    }
    run_ids = iter(bundles.keys())

    def replay_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            run_id = next(run_ids)
            mode = json.loads(request.content.decode("utf-8"))["pipeline_mode"]
            route = _route_payload(f"{mode}-route", duration_s=100.0, cost=20.0, emissions=5.0)
            selected_certificate = {"certificate": 0.88, "certified": True} if mode == "voi" else None
            return httpx.Response(
                200,
                json={
                    "selected": route,
                    "candidates": [route],
                    "run_id": run_id,
                    "manifest_endpoint": f"/runs/{run_id}/manifest",
                    "artifacts_endpoint": f"/runs/{run_id}/artifacts",
                    "selected_certificate": selected_certificate,
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(
                200,
                json={
                    "baseline": _route_payload("osrm-base", duration_s=130.0, cost=18.0, emissions=5.5),
                    "method": "osrm_quick_baseline",
                    "compute_ms": 11.5,
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            raise AssertionError("ORS live endpoint should not be called in replay mode")
        if request.method == "GET" and "/runs/" in request.url.path and "/artifacts/" in request.url.path:
            parts = request.url.path.split("/")
            run_id = parts[2]
            artifact_name = parts[-1]
            artifact_payload = bundles[run_id].get(artifact_name)
            if artifact_payload is None:
                return httpx.Response(404, json={"detail": "missing"})
            if artifact_name.endswith(".jsonl"):
                return httpx.Response(200, text="\n".join(json.dumps(row) for row in artifact_payload))
            return httpx.Response(200, json=artifact_payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(replay_handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--seed",
                "7",
                "--world-count",
                "12",
                    "--ors-snapshot-mode",
                    "replay",
                    "--ors-baseline-policy",
                    "repo_local",
                    "--ors-snapshot-path",
                    str(snapshot_path),
                ]
        )
        payload = thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    assert len(payload["rows"]) == 4
    assert all(row["ors_snapshot_used"] is True for row in payload["rows"])


def test_run_thesis_evaluation_records_failure_rows_per_variant(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            mode = json.loads(request.content.decode("utf-8"))["pipeline_mode"]
            if mode == "dccs_refc":
                return httpx.Response(503, json={"detail": {"reason_code": "strict_dependency_missing"}})
            run_id = {
                "legacy": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "dccs": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "voi": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            }[mode]
            route = _route_payload(f"{mode}-route", duration_s=100.0, cost=20.0, emissions=5.0)
            return httpx.Response(
                200,
                json={
                    "selected": route,
                    "candidates": [route],
                    "run_id": run_id,
                    "manifest_endpoint": f"/runs/{run_id}/manifest",
                    "artifacts_endpoint": f"/runs/{run_id}/artifacts",
                    "selected_certificate": {"certificate": 0.9, "certified": True} if mode == "voi" else None,
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(
                200,
                json={"baseline": _route_payload("osrm-base", duration_s=130.0, cost=18.0, emissions=5.5), "method": "osrm_quick_baseline", "compute_ms": 11.5},
            )
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(
                200,
                json={
                    "baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3),
                    "method": "ors_repo_local_baseline",
                    "provider_mode": "repo_local",
                    "baseline_policy": "corridor_alternative",
                    "compute_ms": 14.2,
                },
            )
        if request.method == "GET" and "/runs/" in request.url.path and "/artifacts/" in request.url.path:
            parts = request.url.path.split("/")
            run_id = parts[2]
            mode = {
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": "legacy",
                "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb": "dccs",
                "cccccccc-cccc-cccc-cccc-cccccccccccc": "voi",
            }[run_id]
            artifact_payload = _artifact_bundle(run_id, selected_id=f"{mode}-route", selected_certificate=0.9 if mode == "voi" else None, pipeline_mode=mode).get(parts[-1])
            if artifact_payload is None:
                return httpx.Response(404, json={"detail": "missing"})
            if parts[-1].endswith(".jsonl"):
                return httpx.Response(200, text="\n".join(json.dumps(row) for row in artifact_payload))
            return httpx.Response(200, json=artifact_payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--ors-baseline-policy",
                "repo_local",
            ]
        )
        payload = thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    failing_row = next(row for row in payload["rows"] if row["variant_id"] == "B")
    assert failing_row["failure_reason"] == "strict_dependency_missing"
    assert failing_row["route_id"] == ""


def test_run_thesis_evaluation_continues_after_one_od_baseline_failure(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_two_row_corpus_csv(corpus_csv)

    route_calls: list[str] = []
    run_modes = {
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": "legacy",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb": "dccs",
        "cccccccc-cccc-cccc-cccc-cccccccccccc": "dccs_refc",
        "dddddddd-dddd-dddd-dddd-dddddddddddd": "voi",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route/baseline":
            payload = json.loads(request.content.decode("utf-8"))
            origin_lat = float(payload["origin"]["lat"])
            if origin_lat == 52.0:
                return httpx.Response(500, json={"detail": {"reason_code": "baseline_transport_error"}})
            return httpx.Response(
                200,
                json={
                    "baseline": _route_payload("osrm-base", duration_s=120.0, cost=18.0, emissions=5.5),
                    "method": "osrm_quick_baseline",
                    "compute_ms": 11.5,
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(
                200,
                json={
                    "baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3),
                    "method": "ors_repo_local_baseline",
                    "provider_mode": "repo_local",
                    "baseline_policy": "corridor_alternative",
                    "compute_ms": 14.2,
                },
            )
        if request.method == "POST" and request.url.path == "/route":
            mode = json.loads(request.content.decode("utf-8"))["pipeline_mode"]
            route_calls.append(mode)
            run_id = {
                "legacy": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "dccs": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "dccs_refc": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                "voi": "dddddddd-dddd-dddd-dddd-dddddddddddd",
            }[mode]
            route = _route_payload(f"{mode}-route", duration_s=100.0, cost=20.0, emissions=5.0)
            return httpx.Response(
                200,
                json={
                    "selected": route,
                    "candidates": [route],
                    "run_id": run_id,
                    "manifest_endpoint": f"/runs/{run_id}/manifest",
                    "artifacts_endpoint": f"/runs/{run_id}/artifacts",
                    "selected_certificate": {"certificate": 0.9, "certified": mode == "voi"} if mode in {"dccs_refc", "voi"} else None,
                },
            )
        if request.method == "GET" and "/runs/" in request.url.path and "/artifacts/" in request.url.path:
            run_id = request.url.path.split("/")[2]
            artifact_name = request.url.path.split("/")[-1]
            mode = run_modes[run_id]
            bundle = _artifact_bundle(
                run_id,
                selected_id=f"{mode}-route",
                selected_certificate=0.9 if mode == "voi" else None,
                pipeline_mode=mode,
            )
            artifact_payload = bundle.get(artifact_name)
            if artifact_payload is None:
                return httpx.Response(404, json={"detail": "missing"})
            if artifact_name.endswith(".jsonl"):
                return httpx.Response(200, text="\n".join(json.dumps(row) for row in artifact_payload))
            return httpx.Response(200, json=artifact_payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--ors-baseline-policy",
                "repo_local",
            ]
        )
        payload = thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    assert len(payload["rows"]) == 8
    assert payload["failure_row_count"] == 4
    assert payload["success_row_count"] == 4
    assert route_calls == ["dccs", "dccs_refc", "voi", "legacy"]
    assert any(row["od_id"] == "od-pass" and not row["failure_reason"] for row in payload["rows"])
    assert all(row["od_id"] == "od-fail" and row["failure_reason"] == "baseline_transport_error" for row in payload["rows"][:4])


def test_run_thesis_evaluation_rejects_proxy_or_synthetic_route_evidence(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)

    bad_route = _route_payload("bad-route", duration_s=101.0, cost=20.2, emissions=5.2)
    bad_route["evidence_provenance"]["families"][0]["source"] = "synthetic_proxy"
    bad_route["evidence_provenance"]["families"][0]["fallback_used"] = True
    bad_route["evidence_provenance"]["families"][0]["fallback_source"] = "proxy"

    route_calls: list[str] = []
    bundles = {
        "11111111-1111-1111-1111-111111111111": _artifact_bundle(
            "11111111-1111-1111-1111-111111111111",
            selected_id="bad-route",
            selected_certificate=None,
            pipeline_mode="legacy",
        ),
        "22222222-2222-2222-2222-222222222222": _artifact_bundle(
            "22222222-2222-2222-2222-222222222222",
            selected_id="bad-route",
            selected_certificate=None,
            pipeline_mode="dccs",
        ),
        "33333333-3333-3333-3333-333333333333": _artifact_bundle(
            "33333333-3333-3333-3333-333333333333",
            selected_id="bad-route",
            selected_certificate=0.75,
            pipeline_mode="dccs_refc",
        ),
        "44444444-4444-4444-4444-444444444444": _artifact_bundle(
            "44444444-4444-4444-4444-444444444444",
            selected_id="bad-route",
            selected_certificate=0.91,
            pipeline_mode="voi",
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            mode = json.loads(request.content.decode("utf-8"))["pipeline_mode"]
            route_calls.append(mode)
            run_id = {
                "legacy": "11111111-1111-1111-1111-111111111111",
                "dccs": "22222222-2222-2222-2222-222222222222",
                "dccs_refc": "33333333-3333-3333-3333-333333333333",
                "voi": "44444444-4444-4444-4444-444444444444",
            }[mode]
            return httpx.Response(
                200,
                json={
                    "selected": bad_route,
                    "candidates": [bad_route, _route_payload(f"{mode}-alt", duration_s=103.0, cost=19.5, emissions=5.4)],
                    "run_id": run_id,
                    "manifest_endpoint": f"/runs/{run_id}/manifest",
                    "artifacts_endpoint": f"/runs/{run_id}/artifacts",
                    "selected_certificate": {"certificate": 0.88, "certified": mode == "voi"},
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(200, json={"baseline": _route_payload("osrm-base", duration_s=120.0, cost=18.0, emissions=5.1), "method": "osrm_quick_baseline", "compute_ms": 12.0})
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(
                200,
                json={
                    "baseline": _route_payload("ors-base", duration_s=125.0, cost=17.5, emissions=5.0),
                    "method": "ors_repo_local_baseline",
                    "provider_mode": "repo_local",
                    "baseline_policy": "corridor_alternative",
                    "compute_ms": 13.0,
                },
            )
        if request.method == "GET" and "/runs/" in request.url.path and "/artifacts/" in request.url.path:
            parts = request.url.path.split("/")
            run_id = parts[2]
            artifact_name = parts[-1]
            artifact_payload = bundles[run_id].get(artifact_name)
            if artifact_payload is None:
                return httpx.Response(404, json={"detail": "missing"})
            if artifact_name.endswith(".jsonl"):
                content = "\n".join(json.dumps(row) for row in artifact_payload)
                return httpx.Response(200, text=content)
            return httpx.Response(200, json=artifact_payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--seed",
                "17",
                "--world-count",
                "12",
                "--certificate-threshold",
                "0.8",
                "--tau-stop",
                "0.01",
                "--ors-snapshot-mode",
                "record",
                "--ors-baseline-policy",
                "repo_local",
            ]
        )
        payload = thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    assert route_calls == ["dccs", "dccs_refc", "voi", "legacy"]
    assert all(
        row["failure_reason"] in {"evidence_provenance_rejected", "strict_evidence_rejected"}
        for row in payload["rows"]
    )


def test_run_thesis_evaluation_rejects_bootstrap_route_evidence_details(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)

    bad_route = _route_payload("bad-route", duration_s=101.0, cost=20.2, emissions=5.2)
    bad_route["evidence_provenance"]["families"][0]["details"] = {
        "mode_observation_source": "empirical_outcome_bootstrap"
    }

    bundles = {
        "11111111-1111-1111-1111-111111111111": _artifact_bundle(
            "11111111-1111-1111-1111-111111111111",
            selected_id="bad-route",
            selected_certificate=None,
            pipeline_mode="legacy",
        ),
        "22222222-2222-2222-2222-222222222222": _artifact_bundle(
            "22222222-2222-2222-2222-222222222222",
            selected_id="bad-route",
            selected_certificate=None,
            pipeline_mode="dccs",
        ),
        "33333333-3333-3333-3333-333333333333": _artifact_bundle(
            "33333333-3333-3333-3333-333333333333",
            selected_id="bad-route",
            selected_certificate=0.75,
            pipeline_mode="dccs_refc",
        ),
        "44444444-4444-4444-4444-444444444444": _artifact_bundle(
            "44444444-4444-4444-4444-444444444444",
            selected_id="bad-route",
            selected_certificate=0.91,
            pipeline_mode="voi",
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            mode = json.loads(request.content.decode("utf-8"))["pipeline_mode"]
            run_id = {
                "legacy": "11111111-1111-1111-1111-111111111111",
                "dccs": "22222222-2222-2222-2222-222222222222",
                "dccs_refc": "33333333-3333-3333-3333-333333333333",
                "voi": "44444444-4444-4444-4444-444444444444",
            }[mode]
            return httpx.Response(
                200,
                json={
                    "selected": bad_route,
                    "candidates": [bad_route],
                    "run_id": run_id,
                    "manifest_endpoint": f"/runs/{run_id}/manifest",
                    "artifacts_endpoint": f"/runs/{run_id}/artifacts",
                    "selected_certificate": {"certificate": 0.88, "certified": mode == "voi"},
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(
                200,
                json={"baseline": _route_payload("osrm-base", duration_s=120.0, cost=21.0, emissions=5.8), "method": "osrm_quick_baseline"},
            )
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(
                200,
                json={
                    "baseline": _route_payload("ors-base", duration_s=118.0, cost=21.2, emissions=5.7),
                    "method": "ors_repo_local_baseline",
                    "provider_mode": "repo_local",
                    "baseline_policy": "corridor_alternative",
                },
            )
        if request.method == "GET" and request.url.path.endswith("/artifacts"):
            run_id = request.url.path.split("/")[2]
            return httpx.Response(200, json={"artifacts": [{"name": name, "path": url} for name, url in bundles[run_id]["artifacts"].items()]})
        if request.method == "GET":
            for bundle in bundles.values():
                for url, payload in bundle["artifact_payloads"].items():
                    if request.url.path == url:
                        return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--ors-baseline-policy",
                "repo_local",
            ]
        )
        payload = thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    assert all(row["failure_reason"] for row in payload["rows"])
    assert all(row["route_evidence_status"] == "failed" for row in payload["rows"])


def test_run_thesis_evaluation_rejects_incomplete_ors_snapshot_metadata(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)

    snapshot_path = tmp_path / "ors_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "routes": {
                    "od-000001": {
                                "request_hash": thesis_module._digest(
                                    {
                                        "origin": {"lat": 52.0, "lon": -1.5},
                                        "destination": {"lat": 51.5, "lon": -1.2},
                                        "vehicle_type": "rigid_hgv",
                                        "scenario_mode": "no_sharing",
                                        "departure_time_utc": None,
                                        "max_alternatives": 8,
                                        "weights": {"time": 1.0, "money": 1.0, "co2": 1.0},
                                        "cost_toggles": {"use_tolls": True},
                                        "stochastic": {"enabled": False, "seed": 7, "samples": 25},
                                        "optimization_mode": "expected_value",
                                        "pipeline_seed": 7,
                                        "search_budget": 4,
                                        "evidence_budget": 2,
                                        "cert_world_count": 12,
                                        "certificate_threshold": 0.8,
                                        "tau_stop": 0.02,
                                    }
                                ),
                        "recorded_at": "2026-03-20T00:00:00Z",
                        "provider_method": "ors_repo_local_baseline",
                        "provider_mode": "repo_local",
                        "compute_ms": 18.0,
                        "response": {
                            "baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3),
                            "method": "ors_repo_local_baseline",
                            "provider_mode": "repo_local",
                            "baseline_policy": "corridor_alternative",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    bundles = {
        "55555555-5555-5555-5555-555555555555": _artifact_bundle("55555555-5555-5555-5555-555555555555", selected_id="legacy-route", selected_certificate=None, pipeline_mode="legacy"),
        "66666666-6666-6666-6666-666666666666": _artifact_bundle("66666666-6666-6666-6666-666666666666", selected_id="dccs-route", selected_certificate=None, pipeline_mode="dccs"),
        "77777777-7777-7777-7777-777777777777": _artifact_bundle("77777777-7777-7777-7777-777777777777", selected_id="refc-route", selected_certificate=0.76, pipeline_mode="dccs_refc"),
        "88888888-8888-8888-8888-888888888888": _artifact_bundle("88888888-8888-8888-8888-888888888888", selected_id="voi-route", selected_certificate=0.88, pipeline_mode="voi"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            mode = json.loads(request.content.decode("utf-8"))["pipeline_mode"]
            run_id = {
                "legacy": "55555555-5555-5555-5555-555555555555",
                "dccs": "66666666-6666-6666-6666-666666666666",
                "dccs_refc": "77777777-7777-7777-7777-777777777777",
                "voi": "88888888-8888-8888-8888-888888888888",
            }[mode]
            return httpx.Response(
                200,
                json={
                    "selected": _route_payload(f"{mode}-route", duration_s=100.0, cost=20.0, emissions=5.0),
                    "candidates": [_route_payload(f"{mode}-route", duration_s=100.0, cost=20.0, emissions=5.0)],
                    "run_id": run_id,
                    "manifest_endpoint": f"/runs/{run_id}/manifest",
                    "artifacts_endpoint": f"/runs/{run_id}/artifacts",
                    "selected_certificate": {"certificate": 0.9, "certified": True} if mode == "voi" else None,
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(200, json={"baseline": _route_payload("osrm-base", duration_s=120.0, cost=18.0, emissions=5.1), "method": "osrm_quick_baseline", "compute_ms": 12.0})
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(
                200,
                json={
                    "baseline": _route_payload("ors-base", duration_s=125.0, cost=17.5, emissions=5.0),
                    "method": "ors_repo_local_baseline",
                    "provider_mode": "repo_local",
                    "baseline_policy": "corridor_alternative",
                    "compute_ms": 13.0,
                },
            )
        if request.method == "GET" and "/runs/" in request.url.path and "/artifacts/" in request.url.path:
            parts = request.url.path.split("/")
            run_id = parts[2]
            artifact_name = parts[-1]
            artifact_payload = bundles[run_id].get(artifact_name)
            if artifact_payload is None:
                return httpx.Response(404, json={"detail": "missing"})
            if artifact_name.endswith(".jsonl"):
                content = "\n".join(json.dumps(row) for row in artifact_payload)
                return httpx.Response(200, text=content)
            return httpx.Response(200, json=artifact_payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--seed",
                "7",
                "--world-count",
                "12",
                "--certificate-threshold",
                "0.8",
                "--tau-stop",
                "0.01",
                "--ors-snapshot-mode",
                "replay",
                "--ors-snapshot-path",
                str(snapshot_path),
            ]
        )
        payload = thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    assert all(row["failure_reason"].startswith("ors_snapshot_missing_required_fields") for row in payload["rows"])


def test_run_thesis_evaluation_fails_closed_on_proxy_ors_and_evidence_fallbacks(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)

    run_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            route = _route_payload("voi-route", duration_s=100.0, cost=20.0, emissions=5.0)
            route["evidence_provenance"] = {
                "active_families": ["scenario"],
                "families": [{"family": "scenario", "source": "live", "fallback_used": True}],
            }
            return httpx.Response(
                200,
                json={
                    "selected": route,
                    "candidates": [route],
                    "run_id": run_id,
                    "manifest_endpoint": f"/runs/{run_id}/manifest",
                    "artifacts_endpoint": f"/runs/{run_id}/artifacts",
                    "selected_certificate": {"certificate": 0.9, "certified": True},
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(
                200,
                json={"baseline": _route_payload("osrm-base", duration_s=130.0, cost=18.0, emissions=5.5), "method": "osrm_quick_baseline", "compute_ms": 11.5},
            )
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(
                200,
                json={"baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3), "method": "ors_proxy_baseline", "compute_ms": 14.2},
            )
        if request.method == "GET" and "/runs/" in request.url.path and "/artifacts/" in request.url.path:
            artifact_payload = _artifact_bundle(run_id, selected_id="voi-route", selected_certificate=0.9, pipeline_mode="voi").get(request.url.path.split("/")[-1])
            if artifact_payload is None:
                return httpx.Response(404, json={"detail": "missing"})
            if request.url.path.endswith(".jsonl"):
                return httpx.Response(200, text="\n".join(json.dumps(row) for row in artifact_payload))
            return httpx.Response(200, json=artifact_payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--run-id",
                "proxy-ors-fail",
            ]
        )
        with pytest.raises(RuntimeError, match="baseline_smoke_failed:strict_proxy_ors_baseline_forbidden"):
            thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    smoke_summary = json.loads(
        (tmp_path / "out" / "artifacts" / "proxy-ors-fail" / "baseline_smoke_summary.json").read_text(encoding="utf-8")
    )
    assert smoke_summary["required_ok"] is False
    assert smoke_summary["ors"]["reason_code"] == "strict_proxy_ors_baseline_forbidden"


def test_run_id_changes_when_secondary_baseline_policy_changes() -> None:
    left = thesis_module._run_id(
        seed=7,
        corpus_hash="corpus-hash",
        model_version="thesis_v1",
        world_count=64,
        snapshot_mode="off",
        baseline_policy="repo_local",
    )
    right = thesis_module._run_id(
        seed=7,
        corpus_hash="corpus-hash",
        model_version="thesis_v1",
        world_count=64,
        snapshot_mode="off",
        baseline_policy="snapshot_replay",
    )
    assert left != right


def test_summary_rows_count_failures_in_boolean_denominators() -> None:
    args = thesis_module._build_parser().parse_args(["--corpus-csv", "dummy.csv"])
    od = {
        "od_id": "od-1",
        "seed": 7,
        "trip_length_bin": "30-100 km",
        "origin_lat": 52.0,
        "origin_lon": -1.5,
        "destination_lat": 51.5,
        "destination_lon": -1.2,
        "straight_line_km": 42.0,
    }
    spec = thesis_module.VariantSpec("C", "voi")
    osrm = thesis_module._empty_baseline_result(method="osrm_quick_baseline", provider_mode="repo_local")
    ors = thesis_module._empty_baseline_result(method="ors_repo_local_baseline", provider_mode="repo_local")
    failing = thesis_module._failure_row(args, od, spec, failure_reason="transport_error", osrm=osrm, ors=ors)
    succeeding = dict(failing)
    succeeding.update(
        {
            "failure_reason": "",
            "artifact_complete": True,
            "route_evidence_ok": True,
            "certified": True,
            "weighted_win_osrm": True,
            "weighted_win_ors": False,
            "dominates_osrm": True,
            "dominates_ors": False,
            "certificate": 0.91,
            "frontier_hypervolume": 1.25,
            "dccs_dc_yield": 0.5,
            "runtime_ms": 123.0,
            "iteration_count": 2,
            "search_budget_used": 1,
            "evidence_budget_used": 1,
        }
    )

    summary = thesis_module._summary_rows([succeeding, failing])
    row = next(item for item in summary if item["variant_id"] == "C")
    assert row["row_count"] == 2
    assert row["success_count"] == 1
    assert row["success_rate"] == 0.5
    assert row["artifact_complete_rate"] == 0.5
    assert row["route_evidence_ok_rate"] == 0.5
    assert row["certified_rate"] == 0.5
    assert row["certified_denominator"] == 2
    assert row["weighted_win_rate_osrm"] == 0.5
    assert row["weighted_denominator_osrm"] == 2


def test_summary_rows_track_numeric_denominators_for_sparse_metrics() -> None:
    summary = thesis_module._summary_rows(
        [
            {
                "variant_id": "C",
                "failure_reason": "",
                "certificate": 0.91,
                "frontier_hypervolume": 1.2,
                "dccs_dc_yield": 0.5,
                "dccs_frontier_recall_at_budget": 0.66,
                "dccs_corridor_family_recall": 0.75,
                "runtime_ms": 100.0,
                "stage_k_raw_ms": 35.0,
                "stage_k_raw_graph_search_initial_ms": 12.5,
                "stage_k_raw_graph_search_retry_ms": 9.5,
                "stage_k_raw_graph_search_rescue_ms": 13.0,
                "search_budget_used": 1,
                "evidence_budget_used": 1,
                "iteration_count": 2,
            },
            {
                "variant_id": "C",
                "failure_reason": "",
                "certificate": None,
                "frontier_hypervolume": 0.8,
                "dccs_dc_yield": None,
                "dccs_frontier_recall_at_budget": None,
                "dccs_corridor_family_recall": None,
                "runtime_ms": None,
                "stage_k_raw_ms": None,
                "stage_k_raw_graph_search_initial_ms": None,
                "stage_k_raw_graph_search_retry_ms": None,
                "stage_k_raw_graph_search_rescue_ms": None,
                "search_budget_used": 2,
                "evidence_budget_used": 0,
                "iteration_count": 1,
            },
            {
                "variant_id": "C",
                "failure_reason": "transport_error",
                "certificate": None,
                "frontier_hypervolume": None,
                "dccs_dc_yield": None,
                "dccs_frontier_recall_at_budget": None,
                "dccs_corridor_family_recall": None,
                "runtime_ms": None,
                "stage_k_raw_ms": None,
                "stage_k_raw_graph_search_initial_ms": None,
                "stage_k_raw_graph_search_retry_ms": None,
                "stage_k_raw_graph_search_rescue_ms": None,
                "search_budget_used": None,
                "evidence_budget_used": None,
                "iteration_count": None,
            },
        ]
    )

    row = next(item for item in summary if item["variant_id"] == "C")
    assert row["row_count"] == 3
    assert row["success_count"] == 2
    assert row["mean_certificate"] == 0.91
    assert row["mean_certificate_denominator"] == 1
    assert row["mean_frontier_hypervolume"] == 1.0
    assert row["mean_frontier_hypervolume_denominator"] == 2
    assert row["mean_dccs_dc_yield"] == 0.5
    assert row["mean_dccs_dc_yield_denominator"] == 1
    assert 0.0 <= row["mean_dccs_dc_yield"] <= 1.0
    assert row["mean_dccs_frontier_recall_at_budget"] == 0.66
    assert 0.0 <= row["mean_dccs_frontier_recall_at_budget"] <= 1.0
    assert row["mean_dccs_corridor_family_recall"] == 0.75
    assert 0.0 <= row["mean_dccs_corridor_family_recall"] <= 1.0
    assert row["mean_runtime_ms"] == 100.0
    assert row["mean_runtime_ms_denominator"] == 1
    assert row["mean_stage_k_raw_ms"] == 35.0
    assert row["mean_stage_k_raw_graph_search_initial_ms"] == 12.5
    assert row["mean_stage_k_raw_graph_search_retry_ms"] == 9.5
    assert row["mean_stage_k_raw_graph_search_rescue_ms"] == 13.0
    assert row["mean_search_budget_used"] == 1.5
    assert row["mean_evidence_budget_used"] == 0.5
    assert row["mean_iteration_count"] == 1.5


def test_summary_rows_capture_hard_case_and_cohort_gap_metrics() -> None:
    representative = {
        "variant_id": "C",
        "failure_reason": "",
        "corpus_group": "representative",
        "corpus_kind": "representative",
        "certificate": 0.96,
        "runtime_ms": 90.0,
        "frontier_diversity_index": 0.25,
        "action_efficiency": 0.20,
        "search_budget_utilization": 0.25,
        "evidence_budget_utilization": 0.0,
        "voi_controller_engaged": False,
        "nontrivial_frontier": False,
        "selector_certificate_disagreement": False,
        "near_tie_mass": 0.0,
        "dccs_dc_yield": 0.40,
        "time_to_best_iteration": 1,
    }
    ambiguity = {
        "variant_id": "C",
        "failure_reason": "",
        "corpus_group": "ambiguity",
        "corpus_kind": "ambiguous",
        "certificate": 0.80,
        "runtime_ms": 120.0,
        "frontier_diversity_index": 0.55,
        "action_efficiency": 0.70,
        "search_budget_utilization": 0.50,
        "evidence_budget_utilization": 0.25,
        "voi_controller_engaged": True,
        "nontrivial_frontier": True,
        "selector_certificate_disagreement": False,
        "near_tie_mass": 0.20,
        "dccs_dc_yield": 0.90,
        "time_to_best_iteration": 3,
    }

    summary = thesis_module._summary_rows([representative, ambiguity])
    row = next(item for item in summary if item["variant_id"] == "C")
    assert row["mean_hard_case_rate"] == 0.5
    assert row["mean_hard_case_certificate"] == 0.8
    assert row["mean_hard_case_runtime_ms"] == 120.0
    assert row["mean_hard_case_frontier_diversity_index"] == 0.55
    assert row["mean_hard_case_action_efficiency"] == 0.7
    assert row["mean_hard_case_search_budget_utilization"] == 0.5
    assert row["mean_hard_case_evidence_budget_utilization"] == 0.25
    assert row["mean_hard_case_controller_engagement_rate"] == 1.0
    assert row["mean_certificate_gap_ambiguity_vs_representative"] == -0.16
    assert row["mean_runtime_gap_ambiguity_vs_representative_ms"] == 30.0
    assert row["mean_dccs_dc_yield_gap_ambiguity_vs_representative"] == 0.5
    assert row["mean_time_to_best_gap_ambiguity_vs_representative"] == 2.0
    assert row["mean_search_budget_utilization_gap_ambiguity_vs_representative"] == 0.25
    assert row["mean_evidence_budget_utilization_gap_ambiguity_vs_representative"] == 0.25

    cohort_rows = thesis_module._cohort_summary_rows([representative, ambiguity])
    labels = {item["cohort_label"] for item in cohort_rows}
    assert labels == {"representative", "ambiguity", "hard_case"}
    hard_case = next(item for item in cohort_rows if item["cohort_label"] == "hard_case")
    assert hard_case["row_count"] == 1
    assert hard_case["cohort_total_row_count"] == 2
    assert hard_case["cohort_share_of_variant"] == 0.5
    assert hard_case["mean_certificate"] == 0.8


def test_summary_rows_capture_runtime_distribution_and_action_density() -> None:
    rows = [
        {
            "variant_id": "C",
            "failure_reason": "",
            "runtime_ms": 90.0,
            "algorithm_runtime_ms": 60.0,
            "baseline_acquisition_runtime_ms": 30.0,
            "global_startup_overhead_ms": 120.0,
            "voi_action_count": 1,
            "iteration_count": 1,
            "search_budget_utilization": 0.25,
            "evidence_budget_utilization": 0.0,
            "quality_per_second": 0.4,
            "frontier_gain_per_ms": 0.002,
            "certificate_gain_per_world": 0.01,
            "cache_reuse_ratio": 0.5,
            "baseline_identity_verified": True,
            "initial_certificate": 0.82,
            "initial_certificate_stop": True,
            "unnecessary_voi_refine": False,
            "time_to_certification_ms": 0.0,
            "controller_shortcut": True,
            "voi_stop_after_certification": True,
            "controller_stress_row": False,
            "preflight_and_warmup_ms": 150.0,
            "stage_option_build_ms": 20.0,
            "option_build_reuse_rate": 0.5,
            "od_ambiguity_index": 0.2,
            "ambiguity_budget_prior": 0.25,
            "observed_ambiguity_index": 0.3,
            "od_ambiguity_confidence": 0.7,
            "od_ambiguity_source_count": 1,
        },
        {
            "variant_id": "C",
            "failure_reason": "",
            "runtime_ms": 120.0,
            "algorithm_runtime_ms": 100.0,
            "baseline_acquisition_runtime_ms": 20.0,
            "global_startup_overhead_ms": 120.0,
            "voi_action_count": 3,
            "iteration_count": 2,
            "search_budget_utilization": 0.50,
            "evidence_budget_utilization": 0.25,
            "quality_per_second": 0.6,
            "frontier_gain_per_ms": 0.004,
            "certificate_gain_per_world": 0.02,
            "cache_reuse_ratio": 0.25,
            "baseline_identity_verified": False,
            "initial_certificate": 0.70,
            "initial_certificate_stop": False,
            "unnecessary_voi_refine": True,
            "time_to_certification_ms": 18.0,
            "controller_shortcut": False,
            "voi_stop_after_certification": False,
            "controller_stress_row": True,
            "preflight_and_warmup_ms": 210.0,
            "stage_option_build_ms": 40.0,
            "option_build_reuse_rate": 0.25,
            "od_ambiguity_index": 0.8,
            "ambiguity_budget_prior": 0.85,
            "observed_ambiguity_index": 0.9,
            "od_ambiguity_confidence": 0.9,
            "od_ambiguity_source_count": 2,
        },
    ]

    summary = thesis_module._summary_rows(rows)
    row = next(item for item in summary if item["variant_id"] == "C")
    assert row["mean_runtime_p50_ms"] == 105.0
    assert row["mean_runtime_p90_ms"] == 117.0
    assert row["mean_runtime_p95_ms"] == 118.5
    assert row["mean_algorithm_runtime_p50_ms"] == 80.0
    assert row["mean_algorithm_runtime_p90_ms"] == 96.0
    assert row["mean_algorithm_runtime_p95_ms"] == 98.0
    assert row["mean_baseline_acquisition_runtime_p90_ms"] == 29.0
    assert row["warmup_amortized_ms"] == 60.0
    assert row["mean_search_budget_utilization_p90"] == 0.475
    assert row["mean_evidence_budget_utilization_p90"] == 0.225
    assert row["mean_voi_action_density"] == 1.25
    assert row["mean_quality_per_second"] == 0.5
    assert row["mean_frontier_gain_per_ms"] == 0.003
    assert row["mean_certificate_gain_per_world"] == 0.015
    assert row["mean_cache_reuse_ratio"] == 0.375
    assert row["baseline_identity_verified_rate"] == 0.5
    assert row["mean_initial_certificate"] == 0.76
    assert row["initial_certificate_stop_rate"] == 0.5
    assert row["unnecessary_voi_refine_rate"] == 0.5
    assert row["mean_time_to_certification_ms"] == 9.0
    assert row["mean_controller_shortcut_rate"] == 0.5
    assert row["mean_voi_stop_after_certification_rate"] == 0.5
    assert row["mean_controller_stress_rate"] == 0.5
    assert row["mean_preflight_and_warmup_ms"] == 180.0
    assert row["mean_stage_option_build_ms"] == 30.0
    assert row["mean_option_build_reuse_rate"] == 0.375
    assert row["mean_od_ambiguity_confidence"] == 0.8
    assert row["mean_od_ambiguity_source_count"] == 1.5
    assert row["ambiguity_prior_realized_correlation"] == 1.0
    assert row["mean_runtime_per_refined_candidate_ms"] is None


def test_summary_rows_prefers_effective_ambiguity_budget_prior_for_correlation() -> None:
    rows = [
        {
            "variant_id": "C",
            "failure_reason": "",
            "runtime_ms": 10.0,
            "algorithm_runtime_ms": 5.0,
            "baseline_acquisition_runtime_ms": 2.0,
            "od_ambiguity_index": 0.9,
            "ambiguity_budget_prior": 0.1,
            "observed_ambiguity_index": 0.1,
        },
        {
            "variant_id": "C",
            "failure_reason": "",
            "runtime_ms": 12.0,
            "algorithm_runtime_ms": 7.0,
            "baseline_acquisition_runtime_ms": 2.0,
            "od_ambiguity_index": 0.1,
            "ambiguity_budget_prior": 0.9,
            "observed_ambiguity_index": 0.9,
        },
    ]

    summary = thesis_module._summary_rows(rows)
    row = next(item for item in summary if item["variant_id"] == "C")

    assert row["ambiguity_prior_realized_correlation"] == 1.0


def test_voi_action_productivity_ignores_single_leading_nonproductive_refine_probe() -> None:
    entries = [
        {
            "chosen_action": {"kind": "refine_top1_dccs"},
            "realized_productive": False,
        },
        {
            "chosen_action": {"kind": "refine_top1_dccs"},
            "realized_productive": True,
        },
        {
            "chosen_action": {"kind": "refresh_top1_vor"},
            "realized_productive": True,
        },
    ]

    productive, total, nonproductive_refine = thesis_module._voi_action_productivity(entries)

    assert productive == 2
    assert total == 3
    assert nonproductive_refine == 0


def test_voi_action_productivity_still_counts_additional_nonproductive_refine_before_value() -> None:
    entries = [
        {
            "chosen_action": {"kind": "refine_top1_dccs"},
            "realized_productive": False,
        },
        {
            "chosen_action": {"kind": "refine_top1_dccs"},
            "realized_productive": False,
        },
        {
            "chosen_action": {"kind": "refresh_top1_vor"},
            "realized_productive": True,
        },
    ]

    productive, total, nonproductive_refine = thesis_module._voi_action_productivity(entries)

    assert productive == 1
    assert total == 3
    assert nonproductive_refine == 1


def test_finalize_cross_variant_metrics_backfills_voi_lift_metrics() -> None:
    rows = thesis_module._finalize_cross_variant_metrics(
        [
            {
                "od_id": "od-1",
                "variant_id": "V0",
                "algorithm_runtime_ms": 120.0,
                "selected_duration_s": 110.0,
                "selected_monetary_cost": 25.0,
                "selected_emissions_kg": 7.0,
                "frontier_hypervolume": 1.8,
            },
            {
                "od_id": "od-1",
                "variant_id": "B",
                "certificate": 0.75,
                "frontier_hypervolume": 2.0,
                "selected_duration_s": 105.0,
                "selected_monetary_cost": 24.0,
                "selected_emissions_kg": 6.5,
            },
            {
                "od_id": "od-1",
                "variant_id": "C",
                "certificate": 0.91,
                "frontier_hypervolume": 2.3,
                "algorithm_runtime_ms": 98.0,
                "selected_duration_s": 100.0,
                "selected_monetary_cost": 23.0,
                "selected_emissions_kg": 6.0,
            },
        ]
    )
    voi_row = next(row for row in rows if row["variant_id"] == "C")
    assert voi_row["voi_realized_certificate_lift"] == 0.16
    assert voi_row["voi_realized_frontier_gain"] == 0.3
    assert voi_row["voi_realized_runtime_delta_ms"] == -22.0
    assert voi_row["duration_gain_vs_v0_s"] == 10.0
    assert voi_row["frontier_hypervolume_gain_vs_v0"] == 0.5
    assert voi_row.get("certificate_lift_vs_v0") is None
    assert voi_row["certificate_availability_gain_vs_v0"] is True


def test_finalize_cross_variant_metrics_preserves_action_level_voi_certificate_lift() -> None:
    rows = thesis_module._finalize_cross_variant_metrics(
        [
            {
                "od_id": "od-1",
                "variant_id": "B",
                "certificate": 1.0,
            },
            {
                "od_id": "od-1",
                "variant_id": "C",
                "certificate": 1.0,
                "voi_realized_certificate_lift": 0.194946,
            },
        ]
    )
    voi_row = next(row for row in rows if row["variant_id"] == "C")
    assert voi_row["voi_realized_certificate_lift"] == 0.194946


def test_finalize_cross_variant_metrics_backfills_margin_lifts_when_certificate_is_saturated() -> None:
    rows = thesis_module._finalize_cross_variant_metrics(
        [
            {
                "od_id": "od-2",
                "variant_id": "B",
                "certificate": 1.0,
                "certificate_runner_up_gap": 0.12,
                "nominal_winner_margin": 0.18,
                "frontier_hypervolume": 2.0,
            },
            {
                "od_id": "od-2",
                "variant_id": "C",
                "certificate": 1.0,
                "certificate_runner_up_gap": 0.2,
                "nominal_winner_margin": 0.26,
                "frontier_hypervolume": 2.1,
                "top_refresh_gain": 0.045,
                "top_fragility_mass": 0.018,
                "competitor_pressure": 0.62,
                "algorithm_runtime_ms": 98.0,
            },
        ]
    )
    voi_row = next(row for row in rows if row["variant_id"] == "C")
    assert voi_row["voi_realized_certificate_lift"] == 0.0
    assert voi_row["voi_realized_runner_up_gap_lift"] == 0.08
    assert voi_row["voi_realized_margin_lift"] == 0.08

    summary_row = next(row for row in thesis_module._summary_rows(rows) if row["variant_id"] == "C")
    assert summary_row["mean_voi_realized_certificate_lift"] == 0.0
    assert summary_row["mean_voi_realized_runner_up_gap_lift"] == 0.08
    assert summary_row["mean_voi_realized_margin_lift"] == 0.08
    assert summary_row["mean_top_refresh_gain"] == pytest.approx(0.045, rel=0.0, abs=1e-6)
    assert summary_row["mean_top_refresh_gain_denominator"] == 1
    assert summary_row["mean_top_fragility_mass"] == pytest.approx(0.018, rel=0.0, abs=1e-6)
    assert summary_row["mean_top_fragility_mass_denominator"] == 1
    assert summary_row["mean_competitor_pressure"] == pytest.approx(0.62, rel=0.0, abs=1e-6)
    assert summary_row["mean_competitor_pressure_denominator"] == 1


def test_refresh_first_resolution_state_distinguishes_persistence_from_honesty() -> None:
    productive, honest, reason = thesis_module._refresh_first_resolution_state(
        first_controller_action_kind="refresh_top1_vor",
        voi_entries=[
            {
                "realized_evidence_uncertainty_delta": -0.03,
            }
        ],
        initial_winner_fragility_nonzero=True,
        initial_refc_top_vor_positive=False,
        refresh_signal_persistent=False,
    )
    assert productive is True
    assert honest is True
    assert reason == "evidence_uncertainty_reduced"

    productive, honest, reason = thesis_module._refresh_first_resolution_state(
        first_controller_action_kind="refresh_top1_vor",
        voi_entries=[
            {
                "realized_productive": False,
                "realized_certificate_delta": 0.0,
                "realized_runner_up_gap_delta": 0.0,
                "realized_evidence_uncertainty_delta": 0.0,
            }
        ],
        initial_winner_fragility_nonzero=False,
        initial_refc_top_vor_positive=False,
        refresh_signal_persistent=False,
    )
    assert productive is False
    assert honest is None
    assert reason == "no_initial_signal"

    productive, honest, reason = thesis_module._refresh_first_resolution_state(
        first_controller_action_kind="refresh_top1_vor",
        voi_entries=[
            {
                "realized_productive": False,
                "realized_selected_score_delta": 0.0,
            }
        ],
        initial_winner_fragility_nonzero=True,
        initial_refc_top_vor_positive=False,
        refresh_signal_persistent=True,
    )
    assert productive is False
    assert honest is True
    assert reason == "persistent_signal"

    productive, honest, reason = thesis_module._refresh_first_resolution_state(
        first_controller_action_kind="refine_top1_dccs",
        voi_entries=[
            {
                "realized_certificate_delta": 0.09,
            }
        ],
        initial_winner_fragility_nonzero=True,
        initial_refc_top_vor_positive=True,
        refresh_signal_persistent=False,
    )
    assert productive is None
    assert honest is None
    assert reason == "not_refresh_first"


def test_summary_rows_roll_up_refresh_first_productivity_and_honesty_rates() -> None:
    summary = thesis_module._summary_rows(
        [
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "refresh_first_productive": True,
                "refresh_resolution_honest": True,
                "refresh_signal_persistent": False,
                "success_count": 1,
                "failure_count": 0,
                "row_count": 1,
            },
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "refresh_first_productive": False,
                "refresh_resolution_honest": True,
                "refresh_signal_persistent": True,
                "success_count": 1,
                "failure_count": 0,
                "row_count": 1,
            },
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "refresh_first_productive": None,
                "refresh_resolution_honest": None,
                "refresh_signal_persistent": None,
                "success_count": 1,
                "failure_count": 0,
                "row_count": 1,
            },
        ]
    )
    row = next(item for item in summary if item["variant_id"] == "C")
    assert row["refresh_signal_persistence_rate"] == 0.5
    assert row["refresh_signal_persistence_denominator"] == 2
    assert row["refresh_first_productive_rate"] == 0.5
    assert row["refresh_first_productive_denominator"] == 2
    assert row["refresh_resolution_honesty_rate"] == 1.0
    assert row["refresh_resolution_honesty_denominator"] == 2


def test_summary_rows_surface_controller_refresh_split_and_hard_case_transfer_metrics() -> None:
    summary = thesis_module._summary_rows(
        [
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "hard_case": True,
                "certificate_selective": True,
                "evidence_first_engagement": True,
                "productive_voi_action_rate": 1.0,
                "winner_fragility_nonzero": True,
                "refc_top_vor_positive": False,
                "controller_refresh_fallback_activated": True,
                "controller_empirical_vs_raw_refresh_disagreement": True,
                "failure_reason": "",
            },
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "hard_case": False,
                "certificate_selective": False,
                "evidence_first_engagement": False,
                "productive_voi_action_rate": 0.0,
                "winner_fragility_nonzero": False,
                "refc_top_vor_positive": False,
                "controller_refresh_fallback_activated": False,
                "controller_empirical_vs_raw_refresh_disagreement": False,
                "failure_reason": "scenario_profile_unavailable",
            },
        ]
    )

    row = next(item for item in summary if item["variant_id"] == "C")
    assert row["scenario_profile_unavailable_rate"] == 0.5
    assert row["controller_refresh_fallback_activation_rate"] == 0.5
    assert row["controller_empirical_vs_raw_refresh_disagreement_rate"] == 0.5
    assert row["broad_hard_case_certificate_selectivity_rate"] == 1.0
    assert row["broad_hard_case_evidence_first_engagement_rate"] == 1.0
    assert row["broad_hard_case_productive_voi_action_rate"] == 1.0
    assert row["broad_hard_case_refc_signal_presence_rate"] == 1.0
    assert row["controller_stress_row_count"] == 0


def test_controller_refresh_stats_counts_zero_to_nonzero_fallback_upgrade_as_disagreement() -> None:
    ranking_basis, top_family, top_gain, fallback_activated, disagreement = thesis_module._controller_refresh_stats(
        {
            "ranking": [{"family": "carbon", "vor": 0.0}],
            "top_refresh_family": "carbon",
            "top_refresh_gain": 0.0,
            "controller_ranking_basis": "raw_refresh_gain_fallback",
            "controller_ranking": [
                {
                    "family": "carbon",
                    "controller_score": 0.027103,
                    "empirical_vor": 0.0,
                    "raw_refresh_gain": 0.027103,
                    "basis": "raw_refresh_gain_fallback",
                }
            ],
            "top_refresh_family_controller": "carbon",
            "top_refresh_gain_controller": 0.027103,
        }
    )

    assert ranking_basis == "raw_refresh_gain_fallback"
    assert top_family == "carbon"
    assert top_gain == 0.027103
    assert fallback_activated is True
    assert disagreement is True


def test_strict_frontier_rows_include_evidence_breadcrumbs() -> None:
    option = types.SimpleNamespace(
        id="route-a",
        geometry=types.SimpleNamespace(coordinates=[(-1.0, 51.0), (-1.1, 51.1)]),
        metrics=types.SimpleNamespace(
            distance_km=12.3,
            duration_s=41.0,
            monetary_cost=8.5,
            emissions_kg=3.2,
        ),
        certification=types.SimpleNamespace(
            certificate=1.0,
            threshold=0.67,
            active_families=["scenario", "weather", "toll"],
            top_fragility_families=["scenario"],
            top_value_of_refresh_family="scenario",
            top_competitor_route_id="route-b",
        ),
    )

    rows = main_module._strict_frontier_rows_from_options(
        [option],
        selected_id="route-a",
        evidence_snapshot_hash="snapshot-abc",
    )
    row = rows[0]
    assert row["active_families"] == ["scenario", "weather", "toll"]
    assert row["evidence_snapshot_hash"] == "snapshot-abc"
    assert row["certificate"] == 1.0
    assert row["certificate_threshold"] == 0.67
    assert row["top_fragility_families"] == ["scenario"]
    assert row["top_value_of_refresh_family"] == "scenario"
    assert row["top_competitor_route_id"] == "route-b"


def test_action_realized_productive_uses_margin_and_uncertainty_signals_when_certificate_is_flat() -> None:
    action_entry = {
        "realized_certificate_delta": 0.0,
        "realized_frontier_gain": 0,
        "realized_selected_route_changed": False,
        "realized_selected_score_delta": 0.0,
        "realized_runner_up_gap_delta": 0.07,
        "realized_evidence_uncertainty_delta": -0.03,
    }

    assert thesis_module._action_realized_productive(action_entry) is True


def test_rate_text_formats_present_and_empty_denominators() -> None:
    assert thesis_module._rate_text(0.5, 2) == "0.5 (1/2)"
    assert thesis_module._rate_text(1.0, 3) == "1.0 (3/3)"
    assert thesis_module._rate_text(None, 0) == "n/a (0/0)"


def test_thesis_report_includes_failure_breakdown_and_variant_rate_text() -> None:
    report = thesis_module._thesis_report(
        "run-123",
        [
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "row_count": 2,
                "success_count": 1,
                "success_rate": 0.5,
                "failure_count": 1,
                "artifact_complete_rate": 0.5,
                "route_evidence_ok_rate": 0.5,
                "certified_rate": 0.5,
                "certified_denominator": 2,
                "weighted_win_rate_osrm": 0.5,
                "weighted_denominator_osrm": 2,
                "weighted_win_rate_v0": 0.5,
                "weighted_denominator_v0": 2,
                "weighted_win_rate_best_baseline": 0.5,
                "weighted_denominator_best_baseline": 2,
                "weighted_win_rate_ors": None,
                "weighted_denominator_ors": 0,
                "time_preserving_win_rate": 0.5,
                "time_preserving_denominator": 2,
                "time_preserving_win_rate_osrm": 0.5,
                "time_preserving_denominator_osrm": 2,
                "time_preserving_win_rate_ors": None,
                "time_preserving_denominator_ors": 0,
                "time_preserving_win_rate_best_baseline": 0.5,
                "time_preserving_denominator_best_baseline": 2,
                "time_preserving_dominance_rate": 0.0,
                "time_preserving_dominance_denominator": 2,
                "time_preserving_dominance_rate_osrm": 0.0,
                "time_preserving_dominance_denominator_osrm": 2,
                "time_preserving_dominance_rate_ors": None,
                "time_preserving_dominance_denominator_ors": 0,
                "time_preserving_dominance_rate_best_baseline": 0.0,
                "time_preserving_dominance_denominator_best_baseline": 2,
                "mean_certificate": 0.91,
                "mean_certificate_denominator": 1,
                "mean_observed_ambiguity_index": 0.44,
                "mean_od_ambiguity_confidence": 0.71,
                "mean_od_ambiguity_source_count": 1.5,
                "mean_od_ambiguity_source_mix_count": 1.5,
                "mean_od_ambiguity_source_entropy": 0.41,
                "mean_od_ambiguity_support_ratio": 0.68,
                "mean_od_ambiguity_prior_strength": 0.44,
                "mean_ambiguity_budget_prior": 0.62,
                "mean_ambiguity_budget_prior_gap": 0.18,
                "budget_prior_exceeds_raw_rate": 0.5,
                "mean_ambiguity_alignment": 0.93,
                "ambiguity_prior_realized_correlation": 0.61,
                "mean_ambiguity_prior_gap": 0.12,
                "mean_runtime_ms": 123.0,
                "mean_runtime_ms_denominator": 1,
                "mean_runtime_p50_ms": 123.0,
                "mean_runtime_p90_ms": 123.0,
                "mean_runtime_p95_ms": 123.0,
                "mean_algorithm_runtime_p50_ms": 98.0,
                "mean_algorithm_runtime_p90_ms": 98.0,
                "mean_algorithm_runtime_p95_ms": 98.0,
                "mean_baseline_acquisition_runtime_p90_ms": 25.7,
                "warmup_amortized_ms": 320.5,
                "mean_warmup_overhead_share": 0.18,
                "mean_runtime_per_refined_candidate_ms": 49.0,
                "mean_runtime_per_frontier_member_ms": 61.5,
                "mean_memory_per_refined_candidate_mb": 32.0,
                "mean_quality_per_second": 0.82,
                "mean_frontier_gain_per_ms": 0.0012,
                "mean_certificate_gain_per_world": 0.007,
                "mean_top_refresh_gain": 0.13,
                "mean_top_refresh_gain_denominator": 1,
                "mean_top_fragility_mass": 0.04,
                "mean_top_fragility_mass_denominator": 1,
                "mean_competitor_pressure": 0.27,
                "mean_competitor_pressure_denominator": 1,
                "refresh_first_productive_rate": 0.5,
                "refresh_first_productive_denominator": 1,
                "refresh_resolution_honesty_rate": 0.5,
                "refresh_resolution_honesty_denominator": 1,
                "mean_cache_reuse_ratio": 0.44,
                "baseline_identity_verified_rate": 1.0,
                "mean_search_budget_utilization_p90": 0.5,
                "mean_evidence_budget_utilization_p90": 0.5,
                "mean_voi_action_density": 1.5,
                "mean_initial_certificate": 0.91,
                "initial_certificate_stop_rate": 0.5,
                "unnecessary_voi_refine_rate": 0.0,
                "mean_stage_option_build_ms": 41.0,
                "mean_option_build_reuse_rate": 0.35,
                "mean_time_to_certification_ms": 12.0,
                "mean_controller_shortcut_rate": 0.5,
                "mean_voi_stop_after_certification_rate": 1.0,
                "mean_controller_stress_rate": 0.5,
                "mean_preflight_and_warmup_ms": 1385.5,
                "mean_weighted_margin_gain_vs_v0": 0.08,
                "mean_weighted_margin_vs_best_baseline": 0.19,
                "mean_frontier_hypervolume_gain_vs_v0": 0.22,
                "mean_certificate_lift_vs_v0": 0.91,
                "certificate_availability_gain_vs_v0_rate": 0.5,
                "mean_hard_case_certificate_lift_vs_v0": 0.35,
                "mean_refc_shortcut_rate": 0.5,
                "mean_refc_cache_hits": 1.0,
                "mean_refc_unique_world_count": 9.0,
                "mean_refc_world_reuse_rate": 0.11,
                "mean_refc_hard_stress_pack_count": 2.0,
                "mean_refc_stress_world_fraction": 0.2,
                "controller_activation_on_high_ambiguity_rate": 1.0,
            }
        ],
        rows=[
            {"variant_id": "C", "failure_reason": ""},
            {"variant_id": "C", "failure_reason": "transport_error"},
            {"variant_id": "A", "failure_reason": "transport_error"},
        ],
        corpus_hash="corpus-hash",
        row_count=2,
        ors_baseline_policy="repo_local",
        ors_snapshot_mode="off",
        preflight_summary={"required_ok": True},
        readiness_summary={"strict_route_ready": True},
        baseline_smoke_summary={
            "required_ok": True,
            "osrm": {"ok": True, "method": "osrm_engine_baseline", "provider_mode": "repo_local"},
            "ors": {"ok": True, "method": "ors_repo_local_baseline", "provider_mode": "repo_local"},
        },
        output_validation={"validated_artifact_count": 7},
    )

    assert "## Failure Breakdown" in report
    assert "## Headline Wins" in report
    assert "## Comparator Honesty" in report
    assert "## Ambiguity Prior" in report
    assert "## Direct Vs V0" in report
    assert "## Hard-Case And Controller Stress" in report
    assert "`transport_error`: 2" in report
    assert "## Claim Framing" in report
    assert "do not yet support an unconditional sampled-suite benchmark claim" in report
    assert "weighted_win_osrm=0.5 (1/2)" in report
    assert "weighted_win_ors=n/a (0/0)" in report
    assert "weighted_win_v0=0.5 (1/2)" in report
    assert "weighted_win_best_baseline=0.5 (1/2)" in report
    assert "time_preserving_win=0.5 (1/2)" in report
    assert "mean_certificate=0.91 (n=1)" in report
    assert "## Metric Highlights" in report
    assert "corpus_ambiguity=" not in report
    assert "observed_ambiguity=0.44" in report
    assert "mean_od_ambiguity_confidence=0.71" in report
    assert "ambiguity_prior_realized_correlation=0.61" in report
    assert "mean_ambiguity_alignment=0.93" in report
    assert "mean_top_refresh_gain=0.13 (n=1)" in report
    assert "mean_top_fragility_mass=0.04 (n=1)" in report
    assert "mean_competitor_pressure=0.27 (n=1)" in report
    assert "refresh_first_productive_rate=0.5 (n=1)" in report
    assert "refresh_resolution_honesty_rate=0.5 (n=1)" in report
    assert "mean_ambiguity_prior_gap=0.12" in report
    assert "mean_ambiguity_budget_prior=0.62" in report
    assert "budget_prior_exceeds_raw_rate=0.5" in report
    assert "## Controller Admissibility" in report
    assert "initial_certificate_stop_rate=0.5" in report
    assert "unnecessary_voi_refine_rate=0.0" in report
    assert "mean_stage_option_build_ms=41.0" in report
    assert "## Runtime Distribution" in report
    assert "runtime_p95_ms=123.0" in report
    assert "runtime_per_refined_candidate_ms=49.0" in report
    assert "warmup_amortized_ms=320.5" in report
    assert "mean_quality_per_second=0.82" not in report
    assert "baseline_identity_verified_rate=1.0" in report
    assert "voi_action_density=1.5" in report
    assert "refc_shortcut_rate=0.5" in report
    assert "refc_cache_hits=1.0" in report
    assert "refc_unique_world_count=9.0" in report
    assert "refc_world_reuse_rate=0.11" in report
    assert "refc_hard_stress_pack_count=2.0" in report
    assert "mean_refc_stress_world_fraction=0.2" in report
    assert "mean_weighted_margin_vs_best_baseline=0.19" in report
    assert "Controller-stress rows are sparse in this run" in report


def test_thesis_report_residual_resolution_downscopes_runtime_claims() -> None:
    report = thesis_module._thesis_report(
        "run-456",
        [
            {
                "variant_id": "V0",
                "pipeline_mode": "legacy",
                "row_count": 2,
                "success_count": 2,
                "success_rate": 1.0,
                "weighted_win_rate_best_baseline": 0.5,
                "weighted_denominator_best_baseline": 2,
                "mean_algorithm_runtime_ms": 100.0,
                "mean_runtime_ms": 110.0,
                "mean_runtime_ms_denominator": 2,
                "runtime_win_rate_v0": 0.0,
                "runtime_denominator_v0": 0,
                "algorithm_runtime_win_rate_v0": 0.0,
                "algorithm_runtime_denominator_v0": 0,
            },
            {
                "variant_id": "A",
                "pipeline_mode": "dccs",
                "row_count": 2,
                "success_count": 2,
                "success_rate": 1.0,
                "weighted_win_rate_best_baseline": 1.0,
                "weighted_denominator_best_baseline": 2,
            },
            {
                "variant_id": "B",
                "pipeline_mode": "dccs_refc",
                "row_count": 2,
                "success_count": 2,
                "success_rate": 1.0,
                "weighted_win_rate_best_baseline": 1.0,
                "weighted_denominator_best_baseline": 2,
            },
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "row_count": 2,
                "success_count": 2,
                "success_rate": 1.0,
                "weighted_win_rate_best_baseline": 1.0,
                "weighted_denominator_best_baseline": 2,
                "mean_algorithm_runtime_ms": 90.0,
                "mean_runtime_ms": 95.0,
                "mean_runtime_ms_denominator": 2,
                "runtime_win_rate_v0": 0.5,
                "runtime_denominator_v0": 2,
                "algorithm_runtime_win_rate_v0": 0.5,
                "algorithm_runtime_denominator_v0": 2,
                "mean_runtime_speedup_vs_v0": -0.1,
                "mean_algorithm_runtime_speedup_vs_v0": -0.2,
                "upstream_nonzero_od_ambiguity_rate": 1.0,
                "mean_ambiguity_budget_prior": 0.61,
                "budget_prior_exceeds_raw_rate": 0.5,
                "mean_od_ambiguity_support_ratio": 0.7,
                "mean_od_ambiguity_source_entropy": 0.45,
                "mean_weighted_margin_gain_vs_v0": 0.15,
            },
        ],
        rows=[],
        corpus_hash="corpus-hash",
        row_count=2,
        ors_baseline_policy="local_service",
        ors_snapshot_mode="off",
        preflight_summary={"required_ok": True},
        readiness_summary={"strict_route_ready": True},
        baseline_smoke_summary={"required_ok": True},
        output_validation={"validated_artifact_count": 5},
    )
    assert "Runtime is improved on aggregate mean but still mixed on matched rows" in report
    assert "mean_algorithm_runtime_speedup_vs_v0=-0.2" in report


def test_compose_parser_defaults_to_latest_suite_run_id() -> None:
    args = compose_module._build_parser().parse_args([])
    assert args.run_id == "thesis_suite_20260322_r3"


def test_composed_methods_appendix_mentions_cohort_summary_artifacts() -> None:
    text = compose_module._composed_methods_appendix(
        {
            "suite_hash": "suite-hash",
            "row_count": 16,
            "representative_base_excluding": "leeds_newcastle",
            "representative_replacement_od": "leeds_newcastle",
            "ambiguity_base_excluding": "cardiff_bristol",
            "ambiguity_replacement_od": "newport_bristol",
            "sources": {
                "representative_base": {"path": "rep-base.csv"},
                "representative_replacement": {"path": "rep-replacement.csv"},
                "ambiguity_base": {"path": "amb-base.csv"},
                "ambiguity_replacement": {"path": "amb-replacement.csv"},
            },
            "ambiguity_prior_coverage": {
                "row_count": 16,
                "nonzero_od_ambiguity_prior_count": 15,
                "nonzero_engine_disagreement_prior_count": 14,
                "nonzero_hard_case_prior_count": 14,
                "ambiguity_prior_source_mix": {
                    "routing_graph_probe": 10,
                    "engine_augmented_probe": 4,
                    "historical_results_bootstrap": 2,
                },
            },
        }
    )
    assert "thesis_summary_by_cohort.csv/json" in text
    assert "hard-case claims stay separated" in text
    assert "rep-base.csv" in text
    assert "Nonzero corpus ambiguity prior coverage" in text
    assert "Prior source mix" in text


def test_compose_prior_coverage_summary_counts_nonzero_priors_and_source_mix() -> None:
    rows = [
        {"profile_id": "p1", "od_ambiguity_index": 0.4},
        {"profile_id": "p2", "od_ambiguity_index": 0.0},
    ]
    canonical_index = {
        "p1": {
            "profile_id": "p1",
            "od_ambiguity_index": 0.4,
            "od_ambiguity_confidence": 0.8,
            "od_ambiguity_source_count": 1,
            "candidate_probe_engine_disagreement_prior": 0.3,
            "hard_case_prior": 0.5,
            "ambiguity_prior_source": "routing_graph_probe",
            "ambiguity_prior_support_count": 4,
        },
        "p2": {
            "profile_id": "p2",
            "od_ambiguity_index": 0.0,
            "od_ambiguity_confidence": 0.6,
            "od_ambiguity_source_count": 2,
            "candidate_probe_engine_disagreement_prior": 0.2,
            "hard_case_prior": 0.0,
            "ambiguity_prior_source": "engine_augmented_probe,historical_results_bootstrap",
            "ambiguity_prior_support_count": 3,
        },
    }

    payload = compose_module._prior_coverage_summary(rows, canonical_index=canonical_index)

    assert payload["row_count"] == 2
    assert payload["nonzero_od_ambiguity_prior_count"] == 1
    assert payload["nonzero_engine_disagreement_prior_count"] == 2
    assert payload["nonzero_hard_case_prior_count"] == 1
    assert payload["nonzero_od_ambiguity_prior_rate"] == 0.5
    assert payload["ambiguity_prior_source_mix"] == {
        "engine_augmented_probe": 1,
        "historical_results_bootstrap": 1,
        "routing_graph_probe": 1,
    }
    assert payload["mean_prior_support_count"] == 3.5
    assert payload["mean_od_ambiguity_confidence"] == 0.7
    assert payload["mean_od_ambiguity_source_count"] == 1.5


def test_observed_ambiguity_index_uses_realized_route_signals() -> None:
    value = thesis_module._observed_ambiguity_index(
        {
            "frontier_count": 5,
            "near_tie_mass": 0.4,
            "nominal_winner_margin": 0.15,
            "certificate_margin": -0.2,
            "certificate_threshold": 0.8,
            "selector_certificate_disagreement": True,
            "voi_action_count": 3,
        }
    )
    assert value == 0.675
    assert thesis_module._observed_ambiguity_index({}) is None


def test_observed_ambiguity_index_respects_controller_side_uncertainty_after_frontier_collapse() -> None:
    value = thesis_module._observed_ambiguity_index(
        {
            "frontier_count": 1,
            "near_tie_mass": 0.0,
            "nominal_winner_margin": 1.0,
            "certificate_margin": 0.18,
            "certificate_threshold": 0.8,
            "selector_certificate_disagreement": False,
            "voi_action_count": 1,
            "pending_challenger_mass": 0.72,
            "best_pending_flip_probability": 1.0,
            "search_completeness_gap": 0.21,
            "prior_support_strength": 0.34,
        }
    )

    assert value == pytest.approx(0.252, rel=0.0, abs=1e-6)
    assert value > 0.2


def test_canonicalize_rows_prefers_current_corpus_labels_and_backfills_observed_ambiguity() -> None:
    canonical_index = {
        "ambiguity_capital_e": {
            "od_id": "london_birmingham_ambiguity",
            "origin_lat": "51.5074",
            "origin_lon": "-0.1278",
            "destination_lat": "52.4862",
            "destination_lon": "-1.8904",
            "straight_line_km": "162.0",
            "distance_bin": "100-250 km",
            "corpus_group": "ambiguity",
        }
    }
    rows = compose_module._canonicalize_rows(
        [
            {
                "od_id": "london_birmingham",
                "profile_id": "ambiguity_capital_e",
                "frontier_count": 4,
                "near_tie_mass": 0.3,
                "nominal_winner_margin": 0.2,
                "certificate_margin": -0.1,
                "certificate_threshold": 0.8,
                "selector_certificate_disagreement": True,
                "voi_action_count": 2,
            }
        ],
        canonical_index=canonical_index,
    )
    assert rows[0]["od_id"] == "london_birmingham_ambiguity"
    assert rows[0]["trip_length_bin"] == "100-250 km"
    assert rows[0]["corpus_group"] == "ambiguity"
    assert rows[0]["cohort_label"] == "ambiguity"
    assert rows[0]["hard_case"] is True
    assert rows[0]["action_efficiency"] == 0.375
    assert rows[0]["observed_ambiguity_index"] == 0.554167


def test_canonicalize_rows_accepts_canonical_od_ambiguity_index() -> None:
    rows = compose_module._canonicalize_rows(
        [
            {
                "profile_id": "canon-od-ambiguity",
                "corpus_kind": "ambiguity",
                "cohort_label": "ambiguity",
                "frontier_count": 1,
                "near_tie_mass": 0.0,
                "nominal_winner_margin": 1.0,
                "certificate_margin": 0.5,
                "voi_action_count": 0,
            }
        ],
        canonical_index={
            "canon-od-ambiguity": {
                "profile_id": "canon-od-ambiguity",
                "od_id": "canon-od-ambiguity",
                "corpus_group": "ambiguity",
                "od_ambiguity_index": 0.67,
            }
        },
    )
    assert rows[0]["od_ambiguity_index"] == 0.67


def test_canonicalize_rows_backfills_engine_and_hard_case_priors_from_canonical_probe_fields() -> None:
    rows = compose_module._canonicalize_rows(
        [
            {
                "profile_id": "canon-priors",
                "corpus_kind": "ambiguity",
                "cohort_label": "ambiguity",
                "frontier_count": 2,
                "near_tie_mass": 0.12,
                "nominal_winner_margin": 0.4,
                "certificate_margin": -0.04,
                "certificate_threshold": 0.8,
                "voi_action_count": 1,
            }
        ],
        canonical_index={
            "canon-priors": {
                "profile_id": "canon-priors",
                "od_id": "canon-priors",
                "corpus_group": "ambiguity",
                "candidate_probe_path_count": 4,
                "candidate_probe_corridor_family_count": 3,
                "candidate_probe_objective_spread": 0.31,
                "candidate_probe_nominal_margin": 0.09,
                "candidate_probe_toll_disagreement_rate": 1.0,
            }
        },
    )

    assert rows[0]["od_engine_disagreement_prior"] == pytest.approx(0.743599, rel=0.0, abs=1e-6)
    assert rows[0]["od_hard_case_prior"] == pytest.approx(0.737441, rel=0.0, abs=1e-6)


def test_canonicalize_rows_backfills_runtime_ratios_from_raw_timing_fields() -> None:
    rows = compose_module._canonicalize_rows(
        [
            {
                "od_id": "od-1",
                "variant_id": "C",
                "pipeline_mode": "voi",
                "profile_id": "profile_a",
                "corpus_group": "ambiguity",
                "route_request_ms": "120.0",
                "baseline_osrm_ms": "12.0",
                "baseline_ors_ms": "15.0",
                "algorithm_runtime_ms": "120.0",
                "baseline_acquisition_runtime_ms": None,
                "runtime_ms": None,
                "runtime_ratio_vs_osrm": None,
                "runtime_ratio_vs_ors": None,
                "algorithm_runtime_ratio_vs_osrm": None,
                "algorithm_runtime_ratio_vs_ors": None,
                "baseline_runtime_share": None,
                "frontier_count": "2",
                "near_tie_mass": "0.1",
                "nominal_winner_margin": "0.7",
                "certificate_margin": "-0.05",
                "certificate_threshold": "0.8",
                "selector_certificate_disagreement": False,
                "voi_action_count": "1",
                "search_budget_used": "1",
                "evidence_budget_used": "1",
                "candidate_count_raw": "3",
            }
        ],
        canonical_index={},
    )

    assert rows[0]["baseline_acquisition_runtime_ms"] == 27.0
    assert rows[0]["runtime_ms"] == 147.0
    assert rows[0]["baseline_runtime_share"] == pytest.approx(27.0 / 147.0, rel=0.0, abs=1e-6)
    assert rows[0]["runtime_ratio_vs_osrm"] == pytest.approx(147.0 / 12.0, rel=0.0, abs=1e-6)
    assert rows[0]["runtime_ratio_vs_ors"] == pytest.approx(147.0 / 15.0, rel=0.0, abs=1e-6)
    assert rows[0]["algorithm_runtime_ratio_vs_osrm"] == pytest.approx(120.0 / 12.0, rel=0.0, abs=1e-6)
    assert rows[0]["algorithm_runtime_ratio_vs_ors"] == pytest.approx(120.0 / 15.0, rel=0.0, abs=1e-6)


def test_run_thesis_evaluation_fails_closed_when_required_artifact_is_missing(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    _build_corpus_csv(corpus_csv)

    runs = {
        "legacy": "11111111-1111-1111-1111-111111111111",
        "dccs": "22222222-2222-2222-2222-222222222222",
        "dccs_refc": "33333333-3333-3333-3333-333333333333",
        "voi": "44444444-4444-4444-4444-444444444444",
    }
    bundles = {
        "legacy": _artifact_bundle(runs["legacy"], selected_id="legacy-route", selected_certificate=None, pipeline_mode="legacy"),
        "dccs": _artifact_bundle(runs["dccs"], selected_id="dccs-route", selected_certificate=None, pipeline_mode="dccs"),
        "dccs_refc": _artifact_bundle(runs["dccs_refc"], selected_id="refc-route", selected_certificate=0.8, pipeline_mode="dccs_refc"),
        "voi": _artifact_bundle(runs["voi"], selected_id="voi-route", selected_certificate=0.9, pipeline_mode="voi"),
    }
    del bundles["voi"]["strict_frontier.jsonl"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            mode = json.loads(request.content.decode("utf-8"))["pipeline_mode"]
            run_id = runs[mode]
            selected = _route_payload(f"{mode}-route", duration_s=100.0, cost=20.0, emissions=5.0)
            return httpx.Response(
                200,
                json={
                    "selected": selected,
                    "candidates": [selected],
                    "run_id": run_id,
                    "manifest_endpoint": f"/runs/{run_id}/manifest",
                    "artifacts_endpoint": f"/runs/{run_id}/artifacts",
                    "selected_certificate": {"certificate": 0.9, "certified": mode == "voi"} if mode in {"dccs_refc", "voi"} else None,
                },
            )
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(200, json={"baseline": _route_payload("osrm-base", duration_s=120.0, cost=18.0, emissions=5.1), "method": "osrm_quick_baseline", "compute_ms": 12.0})
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(
                200,
                json={
                    "baseline": _route_payload("ors-base", duration_s=125.0, cost=17.5, emissions=5.0),
                    "method": "ors_repo_local_baseline",
                    "provider_mode": "repo_local",
                    "baseline_policy": "corridor_alternative",
                    "compute_ms": 13.0,
                },
            )
        if request.method == "GET" and "/runs/" in request.url.path and "/artifacts/" in request.url.path:
            parts = request.url.path.split("/")
            run_id = parts[2]
            artifact_name = parts[-1]
            mode = next(key for key, value in runs.items() if value == run_id)
            artifact_payload = bundles[mode].get(artifact_name)
            if artifact_payload is None:
                return httpx.Response(404, json={"detail": "missing"})
            if artifact_name.endswith(".jsonl"):
                return httpx.Response(200, text="\n".join(json.dumps(row) for row in artifact_payload))
            return httpx.Response(200, json=artifact_payload)
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--ors-snapshot-mode",
                "off",
                "--ors-baseline-policy",
                "repo_local",
            ]
        )
        payload = thesis_module.run_thesis_evaluation(args, client=client)
    finally:
        client.close()

    voi_row = next(row for row in payload["rows"] if row["variant_id"] == "C")
    assert voi_row["failure_reason"] == "strict_artifact_missing"
    assert "strict_frontier.jsonl" in (voi_row["artifact_missing"] or "")
    assert "stage_k_raw_ms" in voi_row
    assert "stage_k_raw_graph_search_initial_ms" in voi_row
    assert "stage_k_raw_graph_search_retry_ms" in voi_row
    assert "stage_k_raw_graph_search_rescue_ms" in voi_row
    assert voi_row["stage_k_raw_ms"] is None
    assert voi_row["stage_k_raw_graph_search_initial_ms"] is None
    assert voi_row["stage_k_raw_graph_search_retry_ms"] is None
    assert voi_row["stage_k_raw_graph_search_rescue_ms"] is None


def test_parser_defaults_favor_local_service_non_snapshot_runs() -> None:
    args = thesis_module._build_parser().parse_args(["--corpus-csv", "dummy.csv"])
    assert args.ors_baseline_policy == "local_service"
    assert args.baseline_refinement_policy == "corridor_uniform"
    assert args.ors_snapshot_mode == "off"
    assert args.route_timeout_seconds == 600.0


def test_load_rows_accepts_od_ambiguity_index_only() -> None:
    rows = thesis_module._load_rows(
        [
            {
                "od_id": "only-od-ambiguity",
                "origin_lat": 51.5,
                "origin_lon": -0.1,
                "destination_lat": 52.5,
                "destination_lon": -1.0,
                "straight_line_km": 120.0,
                "distance_bin": "100-250 km",
                "od_ambiguity_index": 0.61,
                "candidate_probe_path_count": 4,
                "candidate_probe_corridor_family_count": 2,
                "candidate_probe_objective_spread": 0.22,
                "candidate_probe_nominal_margin": 0.11,
                "candidate_probe_toll_disagreement_rate": 0.0,
            }
        ],
        seed=7,
        max_od=0,
    )
    assert rows[0]["ambiguity_index"] == 0.61
    assert rows[0]["od_ambiguity_index"] == 0.61


def test_load_rows_derives_source_support_strength_from_source_mix_when_missing() -> None:
    rows = thesis_module._load_rows(
        [
            {
                "od_id": "support-derivation",
                "origin_lat": 51.5,
                "origin_lon": -0.1,
                "destination_lat": 52.5,
                "destination_lon": -1.0,
                "straight_line_km": 120.0,
                "distance_bin": "100-250 km",
                "od_ambiguity_index": 0.61,
                "od_ambiguity_source_count": 2,
                "od_ambiguity_source_mix": '{"engine_augmented_probe":2,"routing_graph_probe":3}',
                "od_ambiguity_source_mix_count": 2,
                "candidate_probe_path_count": 4,
                "candidate_probe_corridor_family_count": 2,
                "candidate_probe_objective_spread": 0.22,
                "candidate_probe_nominal_margin": 0.11,
                "candidate_probe_toll_disagreement_rate": 0.0,
                "candidate_probe_engine_disagreement_prior": 0.42,
                "hard_case_prior": 0.51,
                "ambiguity_prior_sample_count": 3,
                "ambiguity_prior_support_count": 2,
            }
        ],
        seed=11,
        max_od=0,
    )

    row = rows[0]
    assert row["od_ambiguity_source_support"] is not None
    assert "routing_graph_probe" in row["od_ambiguity_source_support"]
    assert "engine_augmented_probe" in row["od_ambiguity_source_support"]
    assert row["od_ambiguity_source_support_strength"] is not None
    assert row["od_ambiguity_source_support_strength"] > 0.0


def test_load_rows_derives_missing_engine_and_hard_case_priors_from_probe_fields() -> None:
    rows = thesis_module._load_rows(
        [
            {
                "od_id": "derive-priors",
                "origin_lat": 51.5,
                "origin_lon": -0.1,
                "destination_lat": 52.5,
                "destination_lon": -1.0,
                "straight_line_km": 120.0,
                "distance_bin": "100-250 km",
                "od_ambiguity_index": 0.44,
                "candidate_probe_path_count": 4,
                "candidate_probe_corridor_family_count": 3,
                "candidate_probe_objective_spread": 0.31,
                "candidate_probe_nominal_margin": 0.09,
                "candidate_probe_toll_disagreement_rate": 1.0,
            }
        ],
        seed=9,
        max_od=0,
    )

    assert rows[0]["candidate_probe_engine_disagreement_prior"] == pytest.approx(0.743599, rel=0.0, abs=1e-6)
    assert rows[0]["hard_case_prior"] == pytest.approx(0.737441, rel=0.0, abs=1e-6)


def test_ors_baseline_fails_closed_on_provider_mismatch() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3),
                "method": "ors_repo_local_baseline",
                "provider_mode": "repo_local",
                "baseline_policy": "corridor_alternative",
                "compute_ms": 14.2,
            },
        )
    )
    client = httpx.Client(transport=transport, base_url="http://testserver")
    try:
        with pytest.raises(ValueError, match="provider mismatch"):
            thesis_module._ors_baseline(
                client,
                "http://testserver",
                {"origin": {"lat": 51.5, "lon": -0.1}, "destination": {"lat": 52.5, "lon": -1.0}},
                od_id="od-mismatch",
                snapshot_mode="off",
                snapshot_bundle=None,
                baseline_policy="local_service",
            )
    finally:
        client.close()


def test_ors_baseline_fails_closed_on_unverified_local_service_manifest() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "baseline": _route_payload("ors-base", duration_s=135.0, cost=17.0, emissions=5.3),
                "method": "ors_local_engine_baseline",
                "provider_mode": "local_service",
                "baseline_policy": "engine_shortest_path",
                "asset_manifest_hash": "sha256:ors-manifest",
                "asset_freshness_status": "graph_predates_source_pbf",
                "engine_manifest": {
                    "identity_status": "graph_predates_source_pbf",
                    "graph_listing_digest": "digest-123",
                    "graph_file_count": 12,
                },
                "compute_ms": 14.2,
            },
        )
    )
    client = httpx.Client(transport=transport, base_url="http://testserver")
    try:
        with pytest.raises(ValueError, match="graph provenance is not verified"):
            thesis_module._ors_baseline(
                client,
                "http://testserver",
                {"origin": {"lat": 51.5, "lon": -0.1}, "destination": {"lat": 52.5, "lon": -1.0}},
                od_id="od-unverified",
                snapshot_mode="off",
                snapshot_bundle=None,
                baseline_policy="local_service",
            )
    finally:
        client.close()


def test_historical_bootstrap_prior_uses_strong_identity_before_od_id(tmp_path: Path) -> None:
    historical_rows = [
        {
            "od_id": "shared-od",
            "profile_id": "profile-a",
            "origin_lat": "51.5",
            "origin_lon": "-0.1",
            "destination_lat": "52.5",
            "destination_lon": "-1.0",
            "corpus_group": "ambiguity",
            "observed_ambiguity_index": "0.8",
            "candidate_count_raw": "5",
            "frontier_count": "3",
            "near_tie_mass": "0.4",
        },
        {
            "od_id": "shared-od",
            "profile_id": "profile-b",
            "origin_lat": "53.0",
            "origin_lon": "-2.0",
            "destination_lat": "54.0",
            "destination_lon": "-3.0",
            "corpus_group": "representative",
            "observed_ambiguity_index": "0.2",
            "candidate_count_raw": "2",
            "frontier_count": "1",
            "near_tie_mass": "0.1",
        },
    ]
    history_path = tmp_path / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in historical_rows for key in row}))
        writer.writeheader()
        writer.writerows(historical_rows)
    index = ambiguity_module._historical_bootstrap_index([history_path])
    assert "shared-od" not in index["by_od_id_unique"]
    enriched = ambiguity_module._apply_bootstrap_prior(
        {
            "od_id": "shared-od",
            "profile_id": "profile-a",
            "origin_lat": "51.5",
            "origin_lon": "-0.1",
            "destination_lat": "52.5",
            "destination_lon": "-1.0",
            "corpus_group": "ambiguity",
        },
        bootstrap_index=index,
    )
    assert enriched["ambiguity_index"] == 0.8
    assert ambiguity_module._apply_bootstrap_prior(
        {
            "od_id": "shared-od",
            "profile_id": "profile-c",
            "origin_lat": "55.0",
            "origin_lon": "-4.0",
            "destination_lat": "56.0",
            "destination_lon": "-5.0",
            "corpus_group": "ambiguity",
        },
        bootstrap_index=index,
    ).get("ambiguity_index") in (None, "")


def test_result_row_extracts_controller_and_option_build_runtime_metrics() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--backend-url",
            "http://backend.test",
        ]
    )
    od = {
        "od_id": "od-controller",
        "origin_lat": 52.0,
        "origin_lon": -1.5,
        "destination_lat": 51.5,
        "destination_lon": -1.2,
        "straight_line_km": 55.0,
        "trip_length_bin": "30-100 km",
        "seed": 7,
        "profile_id": "controller_profile",
        "corpus_group": "ambiguity",
        "corpus_kind": "ambiguity",
        "od_ambiguity_index": 0.63,
        "od_ambiguity_confidence": 0.81,
        "od_ambiguity_source_count": 2,
        "od_ambiguity_source_mix": '{"engine_augmented_probe":1,"routing_graph_probe":1}',
        "od_ambiguity_source_mix_count": 2,
        "od_ambiguity_source_entropy": 0.63,
        "od_ambiguity_support_ratio": 0.79,
        "od_ambiguity_prior_strength": 0.63,
        "od_ambiguity_family_density": 0.6,
        "od_ambiguity_margin_pressure": 0.82,
        "od_ambiguity_spread_pressure": 0.27,
        "candidate_probe_path_count": 5,
        "candidate_probe_corridor_family_count": 3,
        "candidate_probe_objective_spread": 0.27,
        "candidate_probe_nominal_margin": 0.18,
        "candidate_probe_toll_disagreement_rate": 0.09,
        "candidate_probe_engine_disagreement_prior": 0.41,
        "hard_case_prior": 0.72,
    }
    request_config = thesis_module._effective_request_config(args, od, variant_seed=17)
    selected = _route_payload("r-main", duration_s=100.0, cost=20.0, emissions=5.0)
    route_response = {
        "selected": selected,
        "candidates": [selected],
        "selected_certificate": {"certificate": 0.91, "certified": True},
        "artifact_validation": {"status": "ok", "required": [], "missing": []},
        "route_evidence_validation": {"status": "ok", "issues": []},
        "run_id": "run-controller",
        "manifest_endpoint": "/runs/run-controller/manifest",
        "artifacts_endpoint": "/runs/run-controller/artifacts",
    }
    artifacts = _artifact_bundle(
        "run-controller",
        selected_id="r-main",
        selected_certificate=0.91,
        pipeline_mode="voi",
    )
    artifacts["dccs_candidates.jsonl"] = [
        {"candidate_id": "cand_a", "predicted_refine_cost": 180.0, "observed_refine_cost": 80.0},
        {"candidate_id": "cand_b", "predicted_refine_cost": 155.0, "observed_refine_cost": 45.0},
        {
            "candidate_id": "supplemental:ors_local_seed:cand_d",
            "predicted_refine_cost": 20.0,
            "observed_refine_cost": 10.0,
        },
    ]
    artifacts["voi_action_trace.json"]["actions"][0]["executed_candidate_ids"] = [
        "supplemental:ors_local_seed:cand_d"
    ]
    artifacts["final_route_trace.json"]["stage_timings_ms"]["option_build_ms"] = 21.0
    baseline = thesis_module.BaselineResult(
        route={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        metrics={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        method="baseline",
        compute_ms=10.0,
        provider_mode="repo_local",
    )

    row = thesis_module._result_row(
        args,
        od,
        thesis_module.VariantSpec("C", "voi"),
        route_response,
        120.0,
        artifacts,
        baseline,
        baseline,
        readiness_summary={
            "strict_route_ready": True,
            "wait_elapsed_ms": 1.0,
            "compute_ms": 2.0,
            "route_graph": {"elapsed_ms": 3.0},
        },
        request_config=request_config,
    )

    assert row["search_completeness_score"] == pytest.approx(0.74, rel=0.0, abs=1e-6)
    assert row["search_completeness_gap"] == pytest.approx(0.11, rel=0.0, abs=1e-6)
    assert row["prior_support_strength"] == pytest.approx(0.67, rel=0.0, abs=1e-6)
    assert row["support_richness"] is not None
    assert row["ambiguity_pressure"] is not None
    assert row["pending_challenger_mass"] == pytest.approx(0.28, rel=0.0, abs=1e-6)
    assert row["best_pending_flip_probability"] == pytest.approx(0.32, rel=0.0, abs=1e-6)
    assert row["corridor_family_recall"] == pytest.approx(0.8, rel=0.0, abs=1e-6)
    assert row["frontier_recall_at_budget"] == pytest.approx(0.76, rel=0.0, abs=1e-6)
    assert row["top_refresh_gain"] == pytest.approx(0.045, rel=0.0, abs=1e-6)
    assert row["top_fragility_mass"] == pytest.approx(0.018, rel=0.0, abs=1e-6)
    assert row["competitor_pressure"] == pytest.approx(0.62, rel=0.0, abs=1e-6)
    assert row["initial_refc_top_fragility_family"] == "scenario"
    assert row["initial_refc_top_refresh_family"] == "scenario"
    assert row["initial_refc_top_vor"] == pytest.approx(0.14, rel=0.0, abs=1e-6)
    assert row["initial_refc_vor_gap"] == pytest.approx(0.08, rel=0.0, abs=1e-6)
    assert row["final_refc_top_fragility_family"] == "scenario"
    assert row["final_refc_top_refresh_family"] == "scenario"
    assert row["final_refc_top_vor"] == pytest.approx(0.11, rel=0.0, abs=1e-6)
    assert row["final_refc_vor_gap"] == pytest.approx(0.07, rel=0.0, abs=1e-6)
    assert row["initial_winner_fragility_mass"] == pytest.approx(0.4, rel=0.0, abs=1e-6)
    assert row["final_winner_fragility_mass"] == pytest.approx(0.3, rel=0.0, abs=1e-6)
    assert row["initial_winner_fragility_nonzero"] is True
    assert row["winner_fragility_nonzero"] is True
    assert row["initial_refc_top_vor_positive"] is True
    assert row["refc_top_vor_positive"] is True
    assert row["refresh_signal_persistent"] is None
    assert row["credible_search_uncertainty"] is True
    assert row["credible_evidence_uncertainty"] is True
    assert row["supported_hard_case"] is True
    assert row["evidence_first_engagement"] is False
    assert row["evidence_only_engagement"] is False
    assert row["first_controller_action_kind"] == "refine_top1_dccs"
    assert row["voi_dccs_cache_hits"] == 2
    assert row["voi_dccs_cache_misses"] == 1
    assert row["voi_dccs_cache_hit_rate"] == pytest.approx(2.0 / 3.0, rel=0.0, abs=1e-6)
    assert row["graph_k_raw_cache_hit"] is True
    assert row["graph_low_ambiguity_fast_path"] is False
    assert row["graph_supported_ambiguity_fast_fallback"] is True
    assert row["stage_k_raw_ms"] == pytest.approx(35.0, rel=0.0, abs=1e-6)
    assert row["stage_k_raw_graph_search_initial_ms"] == 0.0
    assert row["stage_k_raw_graph_search_retry_ms"] == 0.0
    assert row["stage_k_raw_graph_search_rescue_ms"] == 0.0
    assert row["stage_k_raw_graph_search_supplemental_ms"] == 0.0
    assert row["stage_k_raw_osrm_fallback_ms"] == 0.0
    assert row["od_ambiguity_source_mix_count"] == 2
    assert row["od_ambiguity_source_entropy"] == 0.63
    assert row["od_ambiguity_support_ratio"] == 0.79
    assert row["od_ambiguity_prior_strength"] == 0.63
    assert row["option_build_reuse_rate"] == pytest.approx(0.666667, rel=0.0, abs=1e-6)
    assert row["option_build_cache_hits"] == 2
    assert row["option_build_rebuild_count"] == 1
    assert row["option_build_cache_hit_rate"] == pytest.approx(2.0 / 3.0, rel=0.0, abs=1e-6)
    assert row["option_build_cache_savings_ms_per_row"] == pytest.approx(14.000007, rel=0.0, abs=1e-6)
    assert row["refine_cost_sample_count"] == 1
    assert row["refine_cost_positive_sample_count"] == 1
    assert row["refine_cost_zero_observed_count"] == 0
    assert row["refine_cost_mape"] == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert row["refine_cost_mae_ms"] == pytest.approx(10.0, rel=0.0, abs=1e-6)
    assert row["refine_cost_rank_correlation"] is None


def test_option_build_reuse_rate_prefers_trace_runtime_and_falls_back_to_candidate_diag() -> None:
    assert thesis_module._option_build_reuse_rate(
        {
            "route_option_cache_runtime": {"reuse_rate": 1.0, "cache_hits": 6, "cache_misses": 0},
            "option_build_runtime": {"reuse_rate": 0.0, "cache_hits": 0, "cache_misses": 1},
        },
        {"option_build_reuse_rate": 0.9},
    ) == 1.0
    assert thesis_module._option_build_reuse_rate({"option_build_runtime": {"reuse_rate": 0.4}}, {"option_build_reuse_rate": 0.9}) == 0.4
    assert thesis_module._option_build_reuse_rate({}, {"option_build_reuse_rate": 0.9}) == 0.9
    assert thesis_module._option_build_reuse_rate({}, {}) is None


def test_result_row_prefers_route_option_cache_runtime_for_option_build_metrics() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--backend-url",
            "http://backend.test",
        ]
    )
    od = {
        "od_id": "od-route-option-runtime",
        "origin_lat": 52.0,
        "origin_lon": -1.5,
        "destination_lat": 51.5,
        "destination_lon": -1.2,
        "straight_line_km": 55.0,
        "trip_length_bin": "30-100 km",
        "seed": 7,
        "profile_id": "route_option_runtime_profile",
        "corpus_group": "ambiguity",
        "corpus_kind": "ambiguity",
    }
    request_config = thesis_module._effective_request_config(args, od, variant_seed=17)
    selected = _route_payload("r-route-option-runtime", duration_s=100.0, cost=20.0, emissions=5.0)
    route_response = {
        "selected": selected,
        "candidates": [selected],
        "selected_certificate": {"certificate": 0.91, "certified": True},
        "artifact_validation": {"status": "ok", "required": [], "missing": []},
        "route_evidence_validation": {"status": "ok", "issues": []},
        "run_id": "run-route-option-runtime",
        "manifest_endpoint": "/runs/run-route-option-runtime/manifest",
        "artifacts_endpoint": "/runs/run-route-option-runtime/artifacts",
    }
    artifacts = _artifact_bundle(
        "run-route-option-runtime",
        selected_id="r-route-option-runtime",
        selected_certificate=0.91,
        pipeline_mode="voi",
    )
    trace = artifacts["final_route_trace.json"]
    assert isinstance(trace, dict)
    trace["option_build_runtime"] = {
        "cache_hits": 0,
        "cache_misses": 1,
        "rebuild_count": 1,
        "reuse_rate": 0.0,
        "saved_ms_estimate": 0.0,
    }
    trace["route_option_cache_runtime"] = {
        "cache_hits": 6,
        "cache_misses": 0,
        "reuse_rate": 1.0,
        "saved_ms_estimate": 18.5,
    }
    baseline = thesis_module.BaselineResult(
        route={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        metrics={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        method="baseline",
        compute_ms=10.0,
        provider_mode="repo_local",
    )

    row = thesis_module._result_row(
        args,
        od,
        thesis_module.VariantSpec("C", "voi"),
        route_response,
        120.0,
        artifacts,
        baseline,
        baseline,
        readiness_summary={
            "strict_route_ready": True,
            "wait_elapsed_ms": 1.0,
            "compute_ms": 2.0,
            "route_graph": {"elapsed_ms": 3.0},
        },
        request_config=request_config,
    )

    assert row["option_build_reuse_rate"] == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert row["option_build_cache_hits"] == 6
    assert row["option_build_rebuild_count"] == 1
    assert row["option_build_cache_hit_rate"] == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert row["option_build_cache_savings_ms_per_row"] == pytest.approx(18.5, rel=0.0, abs=1e-6)


def test_result_row_suppresses_stale_k_raw_substages_for_route_state_reuse() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--backend-url",
            "http://backend.test",
        ]
    )
    od = {
        "od_id": "od-reused",
        "origin_lat": 52.0,
        "origin_lon": -1.5,
        "destination_lat": 51.5,
        "destination_lon": -1.2,
        "straight_line_km": 55.0,
        "trip_length_bin": "30-100 km",
        "seed": 11,
        "profile_id": "reuse_profile",
        "corpus_group": "ambiguity",
        "corpus_kind": "ambiguity",
        "od_ambiguity_index": 0.61,
        "candidate_probe_engine_disagreement_prior": 0.48,
        "hard_case_prior": 0.58,
    }
    request_config = thesis_module._effective_request_config(args, od, variant_seed=23)
    selected = _route_payload("r-reused", duration_s=96.0, cost=18.0, emissions=4.8)
    route_response = {
        "selected": selected,
        "candidates": [selected],
        "selected_certificate": {"certificate": 0.88, "certified": True},
        "artifact_validation": {"status": "ok", "required": [], "missing": []},
        "route_evidence_validation": {"status": "ok", "issues": []},
        "run_id": "run-reused",
        "manifest_endpoint": "/runs/run-reused/manifest",
        "artifacts_endpoint": "/runs/run-reused/artifacts",
    }
    artifacts = _artifact_bundle(
        "run-reused",
        selected_id="r-reused",
        selected_certificate=0.88,
        pipeline_mode="voi",
    )
    trace = artifacts["final_route_trace.json"]
    assert isinstance(trace, dict)
    trace["stage_timings_ms"]["k_raw_ms"] = 0.4
    trace["stage_timings_ms"]["option_build_ms"] = 18.0
    trace["candidate_diagnostics"]["graph_k_raw_cache_hit"] = False
    trace["candidate_diagnostics"]["graph_search_ms_initial"] = 17.0
    trace["candidate_diagnostics"]["graph_search_ms_retry"] = 13.0
    trace["candidate_diagnostics"]["graph_search_ms_rescue"] = 11.0
    trace["candidate_diagnostics"]["graph_search_ms_supplemental"] = 7.0
    trace["option_build_runtime"] = {
        "cache_hits": 1,
        "cache_misses": 0,
        "rebuild_count": 0,
        "reuse_rate": 1.0,
    }
    baseline = thesis_module.BaselineResult(
        route={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        metrics={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        method="baseline",
        compute_ms=10.0,
        provider_mode="repo_local",
    )

    row = thesis_module._result_row(
        args,
        od,
        thesis_module.VariantSpec("C", "voi"),
        route_response,
        101.0,
        artifacts,
        baseline,
        baseline,
        readiness_summary={
            "strict_route_ready": True,
            "wait_elapsed_ms": 1.0,
            "compute_ms": 2.0,
            "route_graph": {"elapsed_ms": 3.0},
        },
        request_config=request_config,
    )

    assert row["route_state_cache_hit_rate"] == 1.0
    assert row["graph_k_raw_cache_hit"] is False
    assert row["stage_k_raw_ms"] == pytest.approx(0.4, rel=0.0, abs=1e-6)
    assert row["stage_k_raw_graph_search_initial_ms"] == 0.0
    assert row["stage_k_raw_graph_search_retry_ms"] == 0.0
    assert row["stage_k_raw_graph_search_rescue_ms"] == 0.0
    assert row["stage_k_raw_graph_search_supplemental_ms"] == 0.0
    assert row["stage_k_raw_osrm_fallback_ms"] == 0.0
    assert row["stage_option_build_ms"] == 0.0


def test_result_row_prefers_request_local_route_cache_runtime_over_global_route_cache_stats() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--backend-url",
            "http://backend.test",
        ]
    )
    od = {
        "od_id": "od-route-cache-runtime",
        "origin_lat": 52.0,
        "origin_lon": -1.5,
        "destination_lat": 51.5,
        "destination_lon": -1.2,
        "straight_line_km": 55.0,
        "trip_length_bin": "30-100 km",
        "seed": 11,
        "profile_id": "reuse_profile",
        "corpus_group": "ambiguity",
        "corpus_kind": "ambiguity",
    }
    request_config = thesis_module._effective_request_config(args, od, variant_seed=23)
    selected = _route_payload("r-route-cache-runtime", duration_s=96.0, cost=18.0, emissions=4.8)
    route_response = {
        "selected": selected,
        "candidates": [selected],
        "selected_certificate": {"certificate": 0.88, "certified": True},
        "artifact_validation": {"status": "ok", "required": [], "missing": []},
        "route_evidence_validation": {"status": "ok", "issues": []},
        "run_id": "run-route-cache-runtime",
        "manifest_endpoint": "/runs/run-route-cache-runtime/manifest",
        "artifacts_endpoint": "/runs/run-route-cache-runtime/artifacts",
    }
    artifacts = _artifact_bundle(
        "run-route-cache-runtime",
        selected_id="r-route-cache-runtime",
        selected_certificate=0.88,
        pipeline_mode="dccs_refc",
    )
    trace = artifacts["final_route_trace.json"]
    assert isinstance(trace, dict)
    trace["route_cache_runtime"] = {"cache_hits": 2, "cache_misses": 0, "reuse_rate": 1.0}
    trace["route_cache_stats"] = {"hits": 50, "misses": 50}
    baseline = thesis_module.BaselineResult(
        route={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        metrics={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        method="baseline",
        compute_ms=10.0,
        provider_mode="repo_local",
    )

    row = thesis_module._result_row(
        args,
        od,
        thesis_module.VariantSpec("B", "dccs_refc"),
        route_response,
        101.0,
        artifacts,
        baseline,
        baseline,
        readiness_summary={
            "strict_route_ready": True,
            "wait_elapsed_ms": 1.0,
            "compute_ms": 2.0,
            "route_graph": {"elapsed_ms": 3.0},
        },
        request_config=request_config,
    )

    assert row["route_cache_hits"] == 2
    assert row["route_cache_misses"] == 0
    assert row["route_cache_hit_rate"] == 1.0


def test_result_row_handles_missing_legacy_ambiguity_budget_prior_without_type_errors() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--backend-url",
            "http://backend.test",
        ]
    )
    od = {
        "od_id": "od-legacy-none",
        "origin_lat": 51.5,
        "origin_lon": -3.2,
        "destination_lat": 51.48,
        "destination_lon": -2.59,
        "straight_line_km": 42.0,
        "trip_length_bin": "30-100 km",
        "seed": 5,
        "profile_id": "legacy_profile",
        "corpus_group": "representative",
        "corpus_kind": "representative",
    }
    request_config = thesis_module._effective_request_config(args, od, variant_seed=5)
    selected = _route_payload("r-legacy", duration_s=105.0, cost=21.0, emissions=5.5)
    route_response = {
        "selected": selected,
        "candidates": [selected],
        "artifact_validation": {"status": "ok", "required": [], "missing": []},
        "route_evidence_validation": {"status": "ok", "issues": []},
        "run_id": "run-legacy",
        "manifest_endpoint": "/runs/run-legacy/manifest",
        "artifacts_endpoint": "/runs/run-legacy/artifacts",
    }
    artifacts = _artifact_bundle(
        "run-legacy",
        selected_id="r-legacy",
        selected_certificate=None,
        pipeline_mode="legacy",
    )
    baseline = thesis_module.BaselineResult(
        route={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        metrics={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        method="baseline",
        compute_ms=10.0,
        provider_mode="repo_local",
    )

    row = thesis_module._result_row(
        args,
        od,
        thesis_module.VariantSpec("V0", "legacy"),
        route_response,
        99.0,
        artifacts,
        baseline,
        baseline,
        readiness_summary={
            "strict_route_ready": True,
            "wait_elapsed_ms": 1.0,
            "compute_ms": 2.0,
            "route_graph": {"elapsed_ms": 3.0},
        },
        request_config=request_config,
    )

    assert row["ambiguity_budget_prior"] is None
    assert row["ambiguity_budget_prior_gap"] is None
    assert row["budget_prior_exceeds_raw"] is False
    assert row["stage_k_raw_osrm_fallback_ms"] is None


def test_result_row_support_fallback_uses_weighted_prior_signals() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--backend-url",
            "http://backend.test",
        ]
    )
    od = {
        "od_id": "od-support-fallback",
        "origin_lat": 51.5,
        "origin_lon": -3.2,
        "destination_lat": 51.48,
        "destination_lon": -2.59,
        "straight_line_km": 42.0,
        "trip_length_bin": "30-100 km",
        "seed": 5,
        "profile_id": "support_profile",
        "corpus_group": "ambiguity",
        "corpus_kind": "ambiguity",
        "od_ambiguity_prior_strength": 0.24,
        "od_ambiguity_support_ratio": 0.82,
        "od_ambiguity_source_entropy": 0.21,
        "od_ambiguity_source_support_strength": 0.63,
        "od_ambiguity_confidence": 0.74,
        "od_ambiguity_source_count": 2,
        "od_ambiguity_source_mix": '{"routing_graph_probe":1,"historical_results_bootstrap":1}',
        "candidate_probe_path_count": 4,
        "candidate_probe_corridor_family_count": 2,
        "od_ambiguity_margin_pressure": 0.48,
        "od_ambiguity_spread_pressure": 0.36,
        "candidate_probe_engine_disagreement_prior": 0.28,
        "hard_case_prior": 0.31,
        "od_ambiguity_toll_instability": 0.12,
        "candidate_probe_top2_gap_pressure": 0.44,
    }
    request_config = thesis_module._effective_request_config(args, od, variant_seed=5)
    selected = _route_payload("r-support", duration_s=105.0, cost=21.0, emissions=5.5)
    route_response = {
        "selected": selected,
        "candidates": [selected],
        "artifact_validation": {"status": "ok", "required": [], "missing": []},
        "route_evidence_validation": {"status": "ok", "issues": []},
        "run_id": "run-support",
        "manifest_endpoint": "/runs/run-support/manifest",
        "artifacts_endpoint": "/runs/run-support/artifacts",
    }
    artifacts = _artifact_bundle(
        "run-support",
        selected_id="r-support",
        selected_certificate=None,
        pipeline_mode="legacy",
    )
    baseline = thesis_module.BaselineResult(
        route={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        metrics={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        method="baseline",
        compute_ms=10.0,
        provider_mode="repo_local",
    )

    row = thesis_module._result_row(
        args,
        od,
        thesis_module.VariantSpec("V0", "legacy"),
        route_response,
        99.0,
        artifacts,
        baseline,
        baseline,
        readiness_summary={
            "strict_route_ready": True,
            "wait_elapsed_ms": 1.0,
            "compute_ms": 2.0,
            "route_graph": {"elapsed_ms": 3.0},
        },
        request_config=request_config,
    )

    assert row["support_richness"] == pytest.approx(0.533233, rel=0.0, abs=1e-6)
    assert row["ambiguity_pressure"] == pytest.approx(0.354, rel=0.0, abs=1e-6)


def test_result_row_suppresses_stale_k_raw_substage_timings_for_tiny_reused_rows() -> None:
    args = thesis_module._build_parser().parse_args(
        [
            "--corpus-csv",
            "dummy.csv",
            "--backend-url",
            "http://backend.test",
        ]
    )
    od = {
        "od_id": "od-reused",
        "origin_lat": 52.0,
        "origin_lon": -1.5,
        "destination_lat": 51.5,
        "destination_lon": -1.2,
        "straight_line_km": 55.0,
        "trip_length_bin": "30-100 km",
        "seed": 7,
        "profile_id": "reused_profile",
        "corpus_group": "representative",
        "corpus_kind": "representative",
        "od_ambiguity_index": 0.18,
        "od_ambiguity_confidence": 0.64,
        "od_ambiguity_source_count": 1,
        "od_ambiguity_source_mix": '{"routing_graph_probe":1}',
        "od_ambiguity_source_mix_count": 1,
        "od_ambiguity_source_entropy": 0.12,
        "od_ambiguity_support_ratio": 1.0,
        "od_ambiguity_prior_strength": 0.18,
        "od_ambiguity_family_density": 0.25,
        "od_ambiguity_margin_pressure": 0.11,
        "od_ambiguity_spread_pressure": 0.04,
        "candidate_probe_path_count": 2,
        "candidate_probe_corridor_family_count": 1,
        "candidate_probe_objective_spread": 0.04,
        "candidate_probe_nominal_margin": 0.32,
        "candidate_probe_toll_disagreement_rate": 0.0,
        "candidate_probe_engine_disagreement_prior": 0.0,
        "hard_case_prior": 0.18,
    }
    request_config = thesis_module._effective_request_config(args, od, variant_seed=17)
    selected = _route_payload("r-main", duration_s=100.0, cost=20.0, emissions=5.0)
    route_response = {
        "selected": selected,
        "candidates": [selected],
        "selected_certificate": {"certificate": 0.91, "certified": True},
        "artifact_validation": {"status": "ok", "required": [], "missing": []},
        "route_evidence_validation": {"status": "ok", "issues": []},
        "run_id": "run-reused",
        "manifest_endpoint": "/runs/run-reused/manifest",
        "artifacts_endpoint": "/runs/run-reused/artifacts",
    }
    artifacts = _artifact_bundle(
        "run-reused",
        selected_id="r-main",
        selected_certificate=0.91,
        pipeline_mode="voi",
    )
    artifacts["final_route_trace.json"]["stage_timings_ms"]["option_build_ms"] = 21.0
    artifacts["final_route_trace.json"]["stage_timings_ms"]["k_raw_ms"] = 0.25
    artifacts["final_route_trace.json"]["candidate_diagnostics"]["graph_k_raw_cache_hit"] = False
    artifacts["final_route_trace.json"]["candidate_diagnostics"]["graph_search_ms_initial"] = 88.0
    artifacts["final_route_trace.json"]["candidate_diagnostics"]["graph_search_ms_retry"] = 77.0
    artifacts["final_route_trace.json"]["candidate_diagnostics"]["graph_search_ms_rescue"] = 66.0
    artifacts["final_route_trace.json"]["candidate_diagnostics"]["graph_search_ms_supplemental"] = 55.0
    artifacts["final_route_trace.json"]["option_build_runtime"]["cache_hits"] = 4
    artifacts["final_route_trace.json"]["option_build_runtime"]["cache_misses"] = 0
    artifacts["final_route_trace.json"]["option_build_runtime"]["reuse_rate"] = 1.0
    artifacts["final_route_trace.json"]["route_state_cache_stats"] = {"hits": 4, "misses": 0}
    baseline = thesis_module.BaselineResult(
        route={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        metrics={"distance_km": 12.0, "duration_s": 140.0, "monetary_cost": 22.0, "emissions_kg": 6.0},
        method="baseline",
        compute_ms=10.0,
        provider_mode="repo_local",
    )

    row = thesis_module._result_row(
        args,
        od,
        thesis_module.VariantSpec("C", "voi"),
        route_response,
        120.0,
        artifacts,
        baseline,
        baseline,
        readiness_summary={
            "strict_route_ready": True,
            "wait_elapsed_ms": 1.0,
            "compute_ms": 2.0,
            "route_graph": {"elapsed_ms": 3.0},
        },
        request_config=request_config,
    )

    assert row["graph_k_raw_cache_hit"] is False
    assert row["route_state_cache_hit_rate"] == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert row["route_state_cache_hits"] == 4.0
    assert row["route_state_cache_misses"] == 0.0
    assert row["stage_k_raw_ms"] == pytest.approx(0.25, rel=0.0, abs=1e-6)
    assert row["stage_k_raw_graph_search_initial_ms"] == 0.0
    assert row["stage_k_raw_graph_search_retry_ms"] == 0.0
    assert row["stage_k_raw_graph_search_rescue_ms"] == 0.0
    assert row["stage_k_raw_graph_search_supplemental_ms"] == 0.0
    assert row["stage_k_raw_osrm_fallback_ms"] == 0.0
