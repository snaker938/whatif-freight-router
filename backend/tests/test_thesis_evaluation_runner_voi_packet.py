from __future__ import annotations

import pytest

import scripts.run_thesis_evaluation as thesis_module

pytestmark = [pytest.mark.thesis, pytest.mark.thesis_results]


def test_summary_refine_cost_rows_use_effective_voi_top_probe_observation() -> None:
    summary_row = thesis_module._summary_rows(
        [
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "refine_cost_prediction_error_deprecated": 0.0,
                "refine_cost_mape": 0.0,
                "refine_cost_sample_count": 3,
                "refine_cost_positive_sample_count": 3,
                "refine_cost_zero_observed_count": 0,
                "refine_cost_mae_ms": 0.0,
                "refine_cost_rank_correlation": 0.0,
                "refine_cost_observations": [
                    {
                        "candidate_id": "cand-a0",
                        "candidate_source_label": "fallback:alternatives:direct_k_raw_fallback",
                        "candidate_source_stage": "direct_k_raw_fallback",
                        "mode": "bootstrap",
                        "selection_rank": 0,
                        "predicted_refine_cost": 40.0,
                        "observed_refine_cost": 20.0,
                    },
                    {
                        "candidate_id": "cand-a1",
                        "candidate_source_label": "fallback:via:3:direct_k_raw_fallback",
                        "candidate_source_stage": "direct_k_raw_fallback",
                        "mode": "bootstrap",
                        "selection_rank": 1,
                        "predicted_refine_cost": 60.0,
                        "observed_refine_cost": 30.0,
                    },
                    {
                        "candidate_id": "cand-a2",
                        "candidate_source_label": "fallback:exclude:motorway:direct_k_raw_fallback",
                        "candidate_source_stage": "direct_k_raw_fallback",
                        "mode": "bootstrap",
                        "selection_rank": 2,
                        "predicted_refine_cost": 200.0,
                        "observed_refine_cost": 100.0,
                    },
                ],
            },
            {
                "variant_id": "C",
                "pipeline_mode": "voi",
                "refine_cost_prediction_error_deprecated": 0.0,
                "refine_cost_mape": 0.0,
                "refine_cost_sample_count": 3,
                "refine_cost_positive_sample_count": 3,
                "refine_cost_zero_observed_count": 0,
                "refine_cost_mae_ms": 0.0,
                "refine_cost_rank_correlation": 0.0,
                "refine_cost_observations": [
                    {
                        "candidate_id": "cand-b0",
                        "candidate_source_label": "fallback:alternatives:direct_k_raw_fallback",
                        "candidate_source_stage": "direct_k_raw_fallback",
                        "mode": "bootstrap",
                        "selection_rank": 0,
                        "predicted_refine_cost": 80.0,
                        "observed_refine_cost": 40.0,
                    },
                    {
                        "candidate_id": "cand-b1",
                        "candidate_source_label": "fallback:via:5:direct_k_raw_fallback",
                        "candidate_source_stage": "direct_k_raw_fallback",
                        "mode": "bootstrap",
                        "selection_rank": 1,
                        "predicted_refine_cost": 120.0,
                        "observed_refine_cost": 60.0,
                    },
                    {
                        "candidate_id": "cand-b2",
                        "candidate_source_label": "fallback:via:7:direct_k_raw_fallback",
                        "candidate_source_stage": "direct_k_raw_fallback",
                        "mode": "challenger",
                        "selection_rank": None,
                        "predicted_refine_cost": 400.0,
                        "observed_refine_cost": 200.0,
                    },
                ],
            },
        ]
    )[0]

    assert summary_row["variant_id"] == "C"
    assert summary_row["refine_cost_sample_count"] == 2
    assert summary_row["refine_cost_positive_sample_count"] == 2
    assert summary_row["refine_cost_zero_observed_count"] == 0
    assert summary_row["refine_cost_mape"] == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert summary_row["refine_cost_mae_ms"] == pytest.approx(36.0, rel=0.0, abs=1e-6)
    assert summary_row["refine_cost_rank_correlation"] == pytest.approx(1.0, rel=0.0, abs=1e-9)


def test_compact_refine_cost_observations_preserve_voi_scope_metadata() -> None:
    observations = thesis_module._compact_refine_cost_observations(
        [
            {
                "candidate_id": "cand-a",
                "candidate_source_label": "fallback:alternatives:direct_k_raw_fallback",
                "candidate_source_stage": "direct_k_raw_fallback",
                "mode": "bootstrap",
                "selection_rank": 0,
                "predicted_refine_cost": 40.0,
                "observed_refine_cost": 20.0,
            }
        ]
    )

    assert observations == [
        {
            "candidate_id": "cand-a",
            "candidate_source_label": "fallback:alternatives:direct_k_raw_fallback",
            "candidate_source_stage": "direct_k_raw_fallback",
            "mode": "bootstrap",
            "selection_rank": 0,
            "predicted_refine_cost": 40.0,
            "observed_refine_cost": 20.0,
        }
    ]
