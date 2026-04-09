from __future__ import annotations

import json
from pathlib import Path

from app.evidence_certification import (
    compute_certificate,
    compute_fragility_maps,
    project_refc_scaffold_states,
)
from app.run_store import ARTIFACT_FILES, artifact_paths_for_run, write_json_artifact
from app.settings import settings


def test_refc_artifact_inventory_and_payload_map_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    expected_names = {
        "decision_package.json",
        "winner_confidence_state.json",
        "pairwise_gap_state.json",
        "flip_radius_summary.json",
        "decision_region_summary.json",
        "certificate_witness.json",
        "certified_set_summary.json",
    }

    assert expected_names.issubset(set(ARTIFACT_FILES))

    artifact_paths = artifact_paths_for_run("run_refc")
    assert expected_names.issubset(set(artifact_paths))

    payload = {
        "schema_version": "1.0.0",
        "terminal_type": "certified_singleton",
        "selected_route_id": "route_1",
        "artifact_pointers": {
            "decision_package": "decision_package.json",
            "winner_confidence_state": "winner_confidence_state.json",
            "pairwise_gap_state": "pairwise_gap_state.json",
        },
        "frontier_summary": {
            "frontier_route_ids": ["route_1", "route_2"],
            "frontier_count": 2,
        },
    }

    written = write_json_artifact("run_refc", "decision_package.json", payload)
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == payload


def test_refc_scaffold_projections_serialize_without_changing_certificate_semantics() -> None:
    routes = [
        {
            "route_id": "route_a",
            "objective": {"time": 10.0, "money": 12.0, "co2": 4.0},
            "evidence": {"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
        },
        {
            "route_id": "route_b",
            "objective": {"time": 11.0, "money": 11.0, "co2": 5.0},
            "evidence": {"scenario": {"time": 0.9, "money": 1.1, "co2": 1.0}},
        },
    ]
    worlds = [
        {
            "world_id": "w1",
            "states": {"scenario": "nominal"},
            "world_kind": "supported_ambiguity_nominal",
        },
        {
            "world_id": "w2",
            "states": {"scenario": "refreshed"},
            "world_kind": "supported_ambiguity_refreshed",
        },
    ]

    certificate_before = compute_certificate(routes, worlds=worlds, threshold=0.5)
    fragility = compute_fragility_maps(routes, worlds=worlds, selected_route_id=certificate_before.winner_id)
    projection = project_refc_scaffold_states(
        certificate_before,
        fragility,
        frontier_route_ids=[certificate_before.selected_route_id, "route_b"],
    )
    certificate_after = compute_certificate(routes, worlds=worlds, threshold=0.5)

    assert certificate_before.as_dict() == certificate_after.as_dict()
    assert projection["winner_confidence_state"].route_id == certificate_before.selected_route_id
    assert projection["pairwise_gap_states"]
    assert all(item.challenger_id for item in projection["pairwise_gap_states"])
    assert projection["flip_radius_state"].route_id == certificate_before.selected_route_id
    assert projection["decision_region_state"].route_id == certificate_before.selected_route_id
    assert projection["certificate_witness"].route_id == certificate_before.selected_route_id
    assert projection["certified_set_state"].member_route_ids == [
        certificate_before.selected_route_id,
        "route_b",
    ]

    serialized = {
        key: value.to_json() if hasattr(value, "to_json") else [item.to_json() for item in value]
        for key, value in projection.items()
    }
    assert json.loads(serialized["winner_confidence_state"])["route_id"] == certificate_before.selected_route_id
    assert json.loads(serialized["pairwise_gap_states"][0])["challenger_id"] == "route_b"
    assert json.loads(serialized["flip_radius_state"])["route_id"] == certificate_before.selected_route_id
    assert json.loads(serialized["decision_region_state"])["route_id"] == certificate_before.selected_route_id
    assert json.loads(serialized["certificate_witness"])["route_id"] == certificate_before.selected_route_id
    assert json.loads(serialized["certified_set_state"])["member_route_ids"] == [
        certificate_before.selected_route_id,
        "route_b",
    ]
