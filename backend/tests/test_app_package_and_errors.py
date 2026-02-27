from __future__ import annotations

import app
from app.model_data_errors import FROZEN_REASON_CODES, ModelDataError, normalize_reason_code


def test_app_package_imports() -> None:
    # Package marker import should be stable for tooling/tests.
    assert app.__name__ == "app"


def test_model_data_error_string_and_details() -> None:
    err = ModelDataError(
        reason_code="routing_graph_unavailable",
        message="routing graph missing",
        details={"graph_path": "backend/out/model_assets/routing_graph_uk.json"},
    )
    assert str(err) == "routing graph missing"
    assert err.details is not None
    assert err.details["graph_path"].endswith("routing_graph_uk.json")


def test_model_data_error_reason_code_normalization() -> None:
    assert "routing_graph_unavailable" in FROZEN_REASON_CODES
    assert "routing_graph_fragmented" in FROZEN_REASON_CODES
    assert "routing_graph_disconnected_od" in FROZEN_REASON_CODES
    assert "routing_graph_coverage_gap" in FROZEN_REASON_CODES
    assert "routing_graph_no_path" in FROZEN_REASON_CODES
    assert "routing_graph_precheck_timeout" in FROZEN_REASON_CODES
    assert "routing_graph_warmup_failed" in FROZEN_REASON_CODES
    assert "model_asset_unavailable" in FROZEN_REASON_CODES
    assert normalize_reason_code("routing_graph_unavailable") == "routing_graph_unavailable"
    assert normalize_reason_code("routing_graph_fragmented") == "routing_graph_fragmented"
    assert normalize_reason_code("routing_graph_no_path") == "routing_graph_no_path"
    assert normalize_reason_code("routing_graph_precheck_timeout") == "routing_graph_precheck_timeout"
    assert normalize_reason_code("routing_graph_warmup_failed") == "routing_graph_warmup_failed"
    assert normalize_reason_code("unknown_reason") == "model_asset_unavailable"
    assert normalize_reason_code("", default="scenario_profile_unavailable") == "scenario_profile_unavailable"
