from __future__ import annotations

import json
from pathlib import Path

from app.settings import settings

RUN_ID = "6f9c0b65-1f4d-4f2c-9d85-3c91f0cf2d84"


def _artifact_path(name: str) -> Path:
    return Path(settings.out_dir) / "artifacts" / RUN_ID / name


def test_preference_query_worked_example_bundle_is_nonzero_and_consistent() -> None:
    decision_package_path = _artifact_path("decision_package.json")
    preference_state_path = _artifact_path("preference_state.json")
    preference_query_trace_path = _artifact_path("preference_query_trace.json")
    voi_stop_certificate_path = _artifact_path("voi_stop_certificate.json")
    final_route_trace_path = _artifact_path("final_route_trace.json")
    index_path = _artifact_path("index.json")

    for path in (
        decision_package_path,
        preference_state_path,
        preference_query_trace_path,
        voi_stop_certificate_path,
        final_route_trace_path,
        index_path,
    ):
        assert path.exists(), f"missing worked-example artifact: {path.name}"

    decision_package = json.loads(decision_package_path.read_text(encoding="utf-8"))
    preference_state = json.loads(preference_state_path.read_text(encoding="utf-8"))
    preference_query_trace = json.loads(preference_query_trace_path.read_text(encoding="utf-8"))
    voi_stop_certificate = json.loads(voi_stop_certificate_path.read_text(encoding="utf-8"))
    final_route_trace = json.loads(final_route_trace_path.read_text(encoding="utf-8"))
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))

    query_count = int(preference_query_trace["query_count"])
    selected_route_id = str(decision_package["selected"]["id"])
    assert query_count > 0
    assert int(preference_state["query_count"]) == query_count
    assert int(decision_package["preference_summary"]["query_count"]) == query_count
    assert int(decision_package["preference_query_trace"]["query_count"]) == query_count
    assert selected_route_id == preference_query_trace["selected_route_id"]
    assert voi_stop_certificate["final_winner_route_id"] == selected_route_id
    assert final_route_trace["selected_route_id"] == selected_route_id
    assert final_route_trace["terminal_type"] == decision_package["terminal_type"]
    assert final_route_trace["artifact_pointers"]["preference_state"] == "preference_state.json"
    assert final_route_trace["artifact_pointers"]["decision_package"] == "decision_package.json"
    assert "preference_state.json" in index_payload["artifact_names"]
    assert "preference_query_trace.json" in index_payload["artifact_names"]
    assert "decision_package.json" in index_payload["artifact_names"]
