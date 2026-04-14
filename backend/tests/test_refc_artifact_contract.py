from __future__ import annotations

import json
from pathlib import Path

import app.evidence_certification as evidence_certification_module
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
        "selected_certificate_basis": "selected_certificate",
        "artifact_pointers": {
            "decision_package": "decision_package.json",
            "winner_confidence_state": "winner_confidence_state.json",
            "pairwise_gap_state": "pairwise_gap_state.json",
        },
        "world_support_summary": {
            "schema_version": "world-support-summary-v1",
            "selected_route_id": "route_1",
            "selected_certificate_basis": "selected_certificate",
            "support_flag": True,
            "support_reason": None,
            "support_state": {
                "support_bin": "supported",
                "support_flag": True,
            },
            "world_bundle_summary": {
                "multi_fidelity_summary": {
                    "proxy_world_count": 3,
                    "audit_world_count": 1,
                    "proxy_bias_model_version": "proxy-v5",
                    "audit_propensity_version": "audit-v3",
                    "proxy_correction_active": True,
                    "multi_fidelity_certificate_basis": "corrected_from_residual_model",
                    "proxy_only_fraction": 0.75,
                    "audit_correction_mass": 2.5,
                    "positivity_diagnostics": {
                        "positivity_ok": True,
                    },
                },
            },
        },
        "preference_summary": {
            "selected_certificate_basis": "selected_certificate",
            "pipeline_mode": "dccs_refc",
            "query_count": 0,
            "contradiction_record": {
                "contradiction_detected": False,
                "contradiction_reasons": [],
            },
            "preference_irrelevance_proven": False,
            "no_query_reason": "no_preference_query_issued",
            "targeted_challenger_route_id": None,
            "query_selection_reason": "no_preference_query_issued",
        },
        "support_summary": {
            "support_flag": True,
            "support_reason": None,
            "multi_fidelity_summary": {
                "proxy_world_count": 3,
                "audit_world_count": 1,
                "proxy_bias_model_version": "proxy-v5",
                "audit_propensity_version": "audit-v3",
                "proxy_correction_active": True,
                "multi_fidelity_certificate_basis": "corrected_from_residual_model",
                "proxy_only_fraction": 0.75,
                "audit_correction_mass": 2.5,
                "positivity_diagnostics": {
                    "positivity_ok": True,
                },
            },
            "proxy_world_count": 3,
            "audit_world_count": 1,
            "proxy_only_fraction": 0.75,
        },
        "frontier_summary": {
            "frontier_route_ids": ["route_1", "route_2"],
            "frontier_count": 2,
        },
        "certificate_summary": {
            "winner_route_id": "route_1",
            "selected_route_id": "route_1",
            "selected_certificate": 0.92,
            "empirical_certificate": 0.91,
            "certificate_lcb": 0.88,
            "certificate_ucb": 0.95,
            "minimum_pairwise_gap_lcb": 0.12,
            "necessary_best_probability": 0.74,
            "possible_best_probability": 0.93,
            "selected_certificate_basis": "selected_certificate",
            "multi_fidelity_basis": "partially_audited",
            "support_flag": True,
            "out_of_support_reason": None,
            "terminal_type": "certified_singleton",
        },
    }

    written = write_json_artifact("run_refc", "decision_package.json", payload)
    assert written.exists()
    written_payload = json.loads(written.read_text(encoding="utf-8"))
    assert written_payload == payload
    assert written_payload["selected_certificate_basis"] == "selected_certificate"
    assert written_payload["world_support_summary"]["selected_route_id"] == "route_1"
    assert written_payload["world_support_summary"]["support_flag"] is True
    assert written_payload["world_support_summary"]["support_state"]["support_bin"] == "supported"
    assert written_payload["preference_summary"]["no_query_reason"] == "no_preference_query_issued"
    assert written_payload["preference_summary"]["contradiction_record"]["contradiction_detected"] is False
    assert written_payload["support_summary"]["multi_fidelity_summary"]["proxy_world_count"] == 3
    assert written_payload["support_summary"]["proxy_only_fraction"] == 0.75
    assert written_payload["certificate_summary"] == payload["certificate_summary"]


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
    assert any(item.nearest_challenger for item in projection["pairwise_gap_states"])
    assert all(item.challenger_radius is not None for item in projection["pairwise_gap_states"])
    assert all(item.flip_budget is not None for item in projection["pairwise_gap_states"])
    assert projection["flip_radius_state"].route_id == certificate_before.selected_route_id
    assert projection["flip_radius_state"].minimum_flip_budget is not None
    assert projection["flip_radius_state"].adversarial_degradation_curve
    assert projection["flip_radius_state"].structured_adversarial_budget is not None
    assert (
        projection["flip_radius_state"].structured_adversarial_budget.evidence_channel.status
        == "active"
    )
    assert (
        projection["flip_radius_state"].structured_adversarial_budget.preference_channel.status
        == "active"
    )
    assert (
        projection["flip_radius_state"].structured_adversarial_budget.search_deficiency_channel.status
        == "not_applicable"
    )
    assert projection["decision_region_state"].route_id == certificate_before.selected_route_id
    assert projection["decision_region_state"].active_challenger_id == "route_b"
    assert projection["decision_region_state"].most_fragile_preference_direction is not None
    assert projection["decision_region_state"].nearest_threat_axis is not None
    assert projection["decision_region_state"].support_status == "supported"
    assert projection["decision_region_state"].support_bin == "supported"
    assert projection["decision_region_state"].calibration_bin == "empirical"
    assert projection["decision_region_state"].selected_certificate_basis == "empirical"
    assert projection["decision_region_state"].nearest_challenger_gap_lower_bound is not None
    assert (
        projection["decision_region_state"].route_fragility_family_count
        == len(projection["decision_region_state"].provenance["route_fragility_families"])
    )
    assert "boundary:pairwise_gap" in projection["decision_region_state"].root_cause_tags
    assert projection["certificate_witness"].route_id == certificate_before.selected_route_id
    assert projection["certificate_witness"].active_challenger_ids
    assert projection["certificate_witness"].support_conditions
    assert projection["certificate_witness"].action_steps
    assert projection["certificate_witness"].support_status == "supported"
    assert projection["certificate_witness"].support_bin == "supported"
    assert projection["certificate_witness"].calibration_bin == "empirical"
    assert projection["certificate_witness"].selected_certificate_basis == "empirical"
    assert projection["certificate_witness"].nearest_certificate_boundary == "pairwise_gap"
    assert projection["certificate_witness"].targeted_challenger_route_id == "route_b"
    assert projection["certificate_witness"].active_challenger_count == 1
    assert projection["certificate_witness"].action_step_count >= 1
    assert projection["certificate_witness"].explanation_sparsity == projection["certificate_witness"].witness_sparsity
    assert "boundary:pairwise_gap" in projection["certificate_witness"].root_cause_tags
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
    flip_payload = json.loads(serialized["flip_radius_state"])
    decision_payload = json.loads(serialized["decision_region_state"])
    witness_payload = json.loads(serialized["certificate_witness"])
    assert flip_payload["route_id"] == certificate_before.selected_route_id
    assert flip_payload["minimum_flip_budget"] is not None
    assert flip_payload["adversarial_degradation_curve"]
    assert flip_payload["structured_adversarial_budget"]["evidence_channel"]["status"] == "active"
    assert flip_payload["structured_adversarial_budget"]["preference_channel"]["status"] == "active"
    assert (
        flip_payload["structured_adversarial_budget"]["search_deficiency_channel"]["status"]
        == "not_applicable"
    )
    assert flip_payload["dominant_fragility_family"] == projection["flip_radius_state"].dominant_fragility_family
    assert (
        flip_payload["provenance"]["unsafe_challenger_present"]
        == projection["flip_radius_state"].provenance["unsafe_challenger_present"]
    )
    assert (
        flip_payload["provenance"]["top_refresh_family"]
        == projection["flip_radius_state"].provenance["top_refresh_family"]
    )
    assert decision_payload["route_id"] == certificate_before.selected_route_id
    assert decision_payload["active_challenger_id"] == "route_b"
    assert (
        decision_payload["nearest_certificate_boundary"]
        == projection["decision_region_state"].nearest_certificate_boundary
    )
    assert decision_payload["nearest_threat_axis"] == projection["decision_region_state"].nearest_threat_axis
    assert (
        decision_payload["minimum_joint_perturbation"]
        == projection["decision_region_state"].minimum_joint_perturbation
    )
    assert decision_payload["most_fragile_preference_direction"] is not None
    assert (
        decision_payload["structured_adversarial_budget"]["preference_channel"]["budget"]
        == projection["decision_region_state"].structured_adversarial_budget.preference_channel.budget
    )
    assert (
        decision_payload["provenance"]["minimum_pairwise_gap_lcb"]
        == projection["pairwise_gap_states"][0].pairwise_gap_lower_bound
    )
    assert (
        decision_payload["provenance"]["minimum_flip_budget"]
        == projection["flip_radius_state"].minimum_flip_budget
    )
    assert (
        decision_payload["provenance"]["route_fragility_families"]
        == projection["decision_region_state"].provenance["route_fragility_families"]
    )
    assert decision_payload["support_status"] == "supported"
    assert decision_payload["support_bin"] == "supported"
    assert decision_payload["calibration_bin"] == "empirical"
    assert decision_payload["selected_certificate_basis"] == "empirical"
    assert decision_payload["nearest_challenger_gap_lower_bound"] is not None
    assert decision_payload["route_fragility_family_count"] == len(
        decision_payload["provenance"]["route_fragility_families"]
    )
    assert "boundary:pairwise_gap" in decision_payload["root_cause_tags"]
    assert witness_payload["route_id"] == certificate_before.selected_route_id
    assert witness_payload["active_challenger_ids"] == ["route_b"]
    assert witness_payload["support_conditions"]
    assert witness_payload["action_steps"]
    assert witness_payload["support_status"] == "supported"
    assert witness_payload["support_bin"] == "supported"
    assert witness_payload["calibration_bin"] == "empirical"
    assert witness_payload["selected_certificate_basis"] == "empirical"
    assert witness_payload["nearest_certificate_boundary"] == "pairwise_gap"
    assert witness_payload["targeted_challenger_route_id"] == "route_b"
    assert witness_payload["active_challenger_count"] == 1
    assert witness_payload["explanation_sparsity"] == witness_payload["witness_sparsity"]
    assert "boundary:pairwise_gap" in witness_payload["root_cause_tags"]
    assert json.loads(serialized["certified_set_state"])["member_route_ids"] == [
        certificate_before.selected_route_id,
        "route_b",
    ]


def test_certified_set_state_artifact_serialization_preserves_exclusion_basis_and_witness_contract() -> None:
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

    certificate = compute_certificate(routes, worlds=worlds, threshold=0.5)
    fragility = compute_fragility_maps(routes, worlds=worlds, selected_route_id=certificate.winner_id)
    projection = project_refc_scaffold_states(
        certificate,
        fragility,
        frontier_route_ids=[certificate.selected_route_id, "route_b"],
    )

    certified_set_state = projection["certified_set_state"]
    certified_set_payload = json.loads(certified_set_state.to_json())

    assert certified_set_payload["member_route_ids"] == certified_set_state.member_route_ids
    assert certified_set_payload["excluded_route_ids"] == certified_set_state.excluded_route_ids
    assert certified_set_payload["exclusion_basis"] == certified_set_state.exclusion_basis
    assert certified_set_payload["set_size"] == certified_set_state.set_size
    assert certified_set_payload["certified"] == certified_set_state.certified
    assert certified_set_payload["witness"]["route_id"] == certified_set_state.witness["route_id"]
    assert (
        certified_set_payload["witness"]["active_challenger_ids"]
        == certified_set_state.witness["active_challenger_ids"]
    )
    assert (
        certified_set_payload["witness"]["outside_routes_excluded"]
        == certified_set_state.witness["outside_routes_excluded"]
    )
    assert (
        certified_set_payload["witness"]["outside_routes_safely_excluded"]
        == certified_set_state.witness["outside_routes_safely_excluded"]
    )
    assert (
        certified_set_payload["witness"]["singleton_justified"]
        == certified_set_state.witness["singleton_justified"]
    )
    assert (
        certified_set_payload["witness"]["singleton_not_justified_reasons"]
        == certified_set_state.witness["singleton_not_justified_reasons"]
    )
    assert (
        certified_set_payload["witness"]["excluded_route_safety_reasons"]
        == certified_set_state.witness["excluded_route_safety_reasons"]
    )


def test_winner_confidence_state_artifact_serialization_records_confidence_trace() -> None:
    certificate = evidence_certification_module.CertificateResult(
        winner_id="route_a",
        certificate={"route_a": 20.0 / 30.0, "route_b": 10.0 / 30.0},
        threshold=0.49,
        certified=True,
        selected_route_id="route_a",
        route_scores={"route_a": [1.0] * 20 + [0.0] * 10, "route_b": [0.0] * 20 + [1.0] * 10},
        world_manifest={
            "world_count": 30,
            "unique_world_count": 30,
            "support_flag": True,
            "selected_certificate_basis": "empirical",
            "confidence_delta": 0.1,
        },
        selector_config={"selector_weights": [1.0, 1.0, 1.0]},
    )

    projection = project_refc_scaffold_states(
        certificate,
        None,
        frontier_route_ids=["route_a", "route_b"],
    )

    winner_payload = json.loads(projection["winner_confidence_state"].to_json())

    assert winner_payload["method"] == "anytime_hoeffding_union_bound"
    assert winner_payload["delta"] == 0.1
    assert winner_payload["stopping_valid_trace_state"]["success_count"] == 20
    assert winner_payload["stopping_valid_trace_state"]["delta_schedule"] == "delta/(n*(n+1))"
    assert winner_payload["stopping_valid_trace_state"]["delta_source"] == "world_manifest.confidence_delta"
