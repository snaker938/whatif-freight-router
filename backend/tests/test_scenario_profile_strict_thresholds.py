from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

import app.calibration_loader as calibration_loader
from app.model_data_errors import ModelDataError
from app.settings import settings


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scenario_profiles_payload(
    *,
    observed_mode_row_share: float,
    projection_dominant_context_share: float,
) -> dict[str, Any]:
    transform = calibration_loader._default_scenario_transform_params()
    transform["fit_strategy"] = "empirical_temporal_forward"
    transform["scenario_edge_scaling_version"] = "v4_live_empirical"
    transform["context_similarity"]["max_distance"] = 10.0

    def _profile(value: float) -> dict[str, float]:
        return {
            "duration_multiplier": value,
            "incident_rate_multiplier": value,
            "incident_delay_multiplier": value,
            "fuel_consumption_multiplier": value,
            "emissions_multiplier": value,
            "stochastic_sigma_multiplier": value,
        }

    now_iso = _now_iso()
    return {
        "version": "scenario_profiles_thresholds_test_v1",
        "calibration_basis": "empirical",
        "as_of_utc": now_iso,
        "generated_at_utc": now_iso,
        "split_strategy": "temporal_forward_plus_corridor_block",
        "holdout_metrics": {
            "mode_separation_mean": 0.12,
            "duration_mape": 0.04,
            "monetary_mape": 0.04,
            "emissions_mape": 0.04,
            "coverage": 0.97,
            "hour_slot_coverage": 12.0,
            "corridor_coverage": 10.0,
            "full_identity_share": 0.12,
            "projection_dominant_context_share": projection_dominant_context_share,
            "observed_mode_row_share": observed_mode_row_share,
        },
        "profiles": {
            "no_sharing": _profile(1.10),
            "partial_sharing": _profile(1.00),
            "full_sharing": _profile(0.90),
        },
        "transform_params": transform,
    }


def test_scenario_quality_gate_uses_configured_observed_mode_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "scenario_require_signature", False)
    monkeypatch.setattr(settings, "scenario_min_observed_mode_row_share", 0.20)
    monkeypatch.setattr(settings, "scenario_max_projection_dominant_context_share", 0.80)
    payload = _scenario_profiles_payload(
        observed_mode_row_share=0.50,
        projection_dominant_context_share=0.50,
    )

    parsed = calibration_loader._parse_scenario_profiles_payload(
        payload,
        source="live_runtime:scenario_profiles",
    )
    assert parsed.holdout_metrics is not None
    assert float(parsed.holdout_metrics.get("observed_mode_row_share", 0.0)) == pytest.approx(0.50)


def test_scenario_quality_gate_fails_when_observed_mode_threshold_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "scenario_require_signature", False)
    monkeypatch.setattr(settings, "scenario_min_observed_mode_row_share", 0.60)
    monkeypatch.setattr(settings, "scenario_max_projection_dominant_context_share", 0.80)
    payload = _scenario_profiles_payload(
        observed_mode_row_share=0.50,
        projection_dominant_context_share=0.50,
    )

    with pytest.raises(ModelDataError) as excinfo:
        calibration_loader._parse_scenario_profiles_payload(
            payload,
            source="live_runtime:scenario_profiles",
        )
    assert excinfo.value.reason_code == "scenario_profile_invalid"
    details = excinfo.value.details if isinstance(excinfo.value.details, dict) else {}
    assert float(details.get("observed_mode_row_share", 0.0)) == pytest.approx(0.50)
    assert float(details.get("required_min_observed_mode_row_share", 0.0)) == pytest.approx(0.60)


def test_scenario_quality_gate_fails_when_projection_threshold_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "scenario_require_signature", False)
    monkeypatch.setattr(settings, "scenario_min_observed_mode_row_share", 0.20)
    monkeypatch.setattr(settings, "scenario_max_projection_dominant_context_share", 0.40)
    payload = _scenario_profiles_payload(
        observed_mode_row_share=0.50,
        projection_dominant_context_share=0.50,
    )

    with pytest.raises(ModelDataError) as excinfo:
        calibration_loader._parse_scenario_profiles_payload(
            payload,
            source="live_runtime:scenario_profiles",
        )
    assert excinfo.value.reason_code == "scenario_profile_invalid"
    details = excinfo.value.details if isinstance(excinfo.value.details, dict) else {}
    assert float(details.get("projection_dominant_context_share", 0.0)) == pytest.approx(0.50)
    assert float(details.get("required_max_projection_dominant_context_share", 0.0)) == pytest.approx(0.40)
