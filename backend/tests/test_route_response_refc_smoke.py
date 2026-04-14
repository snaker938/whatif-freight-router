from __future__ import annotations

import asyncio
import json
from typing import Any

from app.evidence_certification import compute_certificate, compute_fragility_maps, project_refc_scaffold_states
from app import main as main_module
from app.main import (
    CandidateDiagnostics,
    _assemble_decision_package,
    _route_terminal_fields,
    _write_route_run_bundle,
    app,
    ors_client,
    osrm_client,
)
from app.models import (
    GeoJSONLineString,
    AbstentionRecord,
    LatLng,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    RouteRequest,
    RouteResponse,
    VoiStopSummary,
)
from app.preference_model import build_preference_state
from app.run_store import artifact_dir_for_run
from app.settings import settings
from fastapi import Response
from fastapi.testclient import TestClient


def _route(route_id: str, *, distance_km: float, duration_s: float, monetary_cost: float, emissions_kg: float) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(0.0, 0.0), (1.0, 1.0)]),
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            avg_speed_kmh=50.0,
        ),
    )


class _NoopOSRM:
    async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
        return []


class _NoopORS:
    async def fetch_route(self, **_: Any) -> Any:
        raise AssertionError("local ORS should not be used in this direct-pipeline smoke")


def test_route_response_refc_smoke_ties_terminal_shape_to_scaffold_projection() -> None:
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
    fragility = compute_fragility_maps(routes, worlds=worlds, selected_route_id=certificate.selected_route_id)
    projection = project_refc_scaffold_states(
        certificate,
        fragility,
        frontier_route_ids=[certificate.selected_route_id, "route_b"],
    )

    selected_certificate = RouteCertificationSummary(
        route_id=certificate.selected_route_id,
        certificate=float(certificate.certificate[certificate.selected_route_id]),
        certified=bool(certificate.certified),
        threshold=float(certificate.threshold),
        active_families=list(certificate.world_manifest.get("active_families", [])),
        top_fragility_families=list(fragility.route_fragility_map.get(certificate.selected_route_id, {}).keys())[:3],
        top_competitor_route_id=certificate.winner_id if certificate.winner_id != certificate.selected_route_id else "route_b",
        top_value_of_refresh_family="scenario",
        ambiguity_context={"support_strength": True},
    )
    voi_stop_summary = VoiStopSummary(
        final_route_id=certificate.selected_route_id,
        certificate=float(certificate.certificate[certificate.selected_route_id]),
        certified=bool(certificate.certified),
        iteration_count=2,
        search_budget_used=1,
        evidence_budget_used=1,
        stop_reason="certified",
    )

    certified_set, abstention = _route_terminal_fields(
        selected_certificate=selected_certificate,
        voi_stop_summary=voi_stop_summary,
        strict_frontier=[
            _route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
            _route("route_b", distance_km=11.0, duration_s=21.0, monetary_cost=31.0, emissions_kg=5.0),
        ],
    )

    response = RouteResponse(
        selected=_route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
        candidates=[
            _route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
            _route("route_b", distance_km=11.0, duration_s=21.0, monetary_cost=31.0, emissions_kg=5.0),
        ],
        run_id="run-refc-smoke",
        pipeline_mode="voi",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
        selected_certificate=selected_certificate,
        certificate_summary={
            "winner_route_id": certificate.selected_route_id,
            "selected_route_id": certificate.selected_route_id,
            "selected_certificate": float(certificate.certificate[certificate.selected_route_id]),
            "empirical_certificate": float(certificate.certificate[certificate.selected_route_id]),
            "certificate_lcb": projection["winner_confidence_state"].lower_bound,
            "certificate_ucb": projection["winner_confidence_state"].upper_bound,
            "minimum_pairwise_gap_lcb": min(
                state.pairwise_gap_lower_bound for state in projection["pairwise_gap_states"]
            ),
            "necessary_best_probability": 1.0,
            "possible_best_probability": 1.0,
            "selected_certificate_basis": "selected_certificate",
            "multi_fidelity_basis": "partially_audited",
            "support_flag": True,
            "out_of_support_reason": None,
            "terminal_type": "certified_set",
        },
        voi_stop_summary=voi_stop_summary,
        world_support_summary={
            "schema_version": "world-support-summary-v1",
            "selected_route_id": certificate.selected_route_id,
            "selected_certificate_basis": "selected_certificate",
            "support_flag": True,
            "support_state": {
                "support_score": 1.0,
                "support_ratio": 1.0,
                "support_bin": "supported",
                "calibration_bin": "empirical",
                "support_source": "world_manifest",
                "out_of_support_reason": None,
                "provenance": {
                    "selected_route_id": certificate.selected_route_id,
                },
            },
            "world_bundle_summary": {
                "schema_version": "world-bundle-summary-v1",
                "support_flag": True,
                "support_state": {
                    "support_score": 1.0,
                    "support_ratio": 1.0,
                    "support_bin": "supported",
                    "calibration_bin": "empirical",
                    "support_source": "world_manifest",
                    "out_of_support_reason": None,
                    "provenance": {
                        "selected_route_id": certificate.selected_route_id,
                    },
                },
                "probabilistic_world_bundle": {
                    "world_count": 2,
                    "selected_route_id": certificate.selected_route_id,
                },
                "audit_world_bundle": {
                    "world_count": 1,
                    "selected_route_id": certificate.selected_route_id,
                },
            },
        },
        certified_set=certified_set,
        abstention=abstention,
        winner_confidence_state=projection["winner_confidence_state"],
        pairwise_gap_states=projection["pairwise_gap_states"],
        flip_radius_state=projection["flip_radius_state"],
        decision_region_state=projection["decision_region_state"],
        certificate_witness=projection["certificate_witness"],
    )

    encoded = json.loads(response.model_dump_json())
    restored = RouteResponse.model_validate(response.model_dump(mode="python"))
    assert response.terminal_type == "certified_set"
    assert encoded["terminal_type"] == "certified_set"
    assert encoded["selected_certificate_basis"] == "selected_certificate"
    assert encoded["certificate_summary"]["winner_route_id"] == certificate.selected_route_id
    assert encoded["certificate_summary"]["certificate_lcb"] == projection["winner_confidence_state"].lower_bound
    assert encoded["certificate_summary"]["minimum_pairwise_gap_lcb"] == min(
        state.pairwise_gap_lower_bound for state in projection["pairwise_gap_states"]
    )
    assert encoded["certificate_summary"]["multi_fidelity_basis"] == "partially_audited"
    assert encoded["certificate_summary"]["support_flag"] is True
    assert encoded["certificate_summary"]["terminal_type"] == "certified_set"
    assert encoded["world_support_summary"]["selected_route_id"] == certificate.selected_route_id
    assert encoded["world_support_summary"]["schema_version"] == "world-support-summary-v1"
    assert encoded["world_support_summary"]["support_state"]["support_bin"] == "supported"
    assert encoded["world_support_summary"]["world_bundle_summary"]["support_state"]["support_bin"] == "supported"
    assert encoded["artifact_pointers"] == {
        "manifest_endpoint": "/manifest",
        "artifacts_endpoint": "/artifacts",
        "provenance_endpoint": "/provenance",
    }
    assert [route.id for route in response.certified_set] == ["route_a", "route_b"]
    assert response.abstention is None
    assert abstention is None
    assert encoded["abstention"] is None
    assert encoded["winner_confidence_state"]["route_id"] == certificate.selected_route_id
    assert encoded["pairwise_gap_states"][0]["challenger_id"] == "route_b"
    assert any(state["nearest_challenger"] for state in encoded["pairwise_gap_states"])
    assert encoded["pairwise_gap_states"][0]["challenger_radius"] is not None
    assert encoded["flip_radius_state"]["route_id"] == certificate.selected_route_id
    assert encoded["flip_radius_state"]["minimum_flip_budget"] is not None
    assert encoded["flip_radius_state"]["adversarial_degradation_curve"]
    assert encoded["decision_region_state"]["route_id"] == certificate.selected_route_id
    assert encoded["decision_region_state"]["active_challenger_id"] == "route_b"
    assert encoded["decision_region_state"]["most_fragile_preference_direction"] is not None
    assert encoded["decision_region_state"]["nearest_threat_axis"] is not None
    assert encoded["certificate_witness"]["route_id"] == certificate.selected_route_id
    assert encoded["certificate_witness"]["active_challenger_ids"] == ["route_b"]
    assert encoded["certificate_witness"]["support_conditions"]
    assert encoded["certificate_witness"]["action_steps"]
    assert restored.terminal_type == "certified_set"
    assert restored.selected_certificate_basis == "selected_certificate"
    assert restored.world_support_summary["schema_version"] == "world-support-summary-v1"
    assert restored.world_support_summary["selected_route_id"] == certificate.selected_route_id
    assert json.loads(restored.model_dump_json()) == encoded

    assert projection["winner_confidence_state"].route_id == certificate.selected_route_id
    assert projection["pairwise_gap_states"]
    assert projection["flip_radius_state"].route_id == certificate.selected_route_id
    assert projection["decision_region_state"].route_id == certificate.selected_route_id
    assert projection["certificate_witness"].route_id == certificate.selected_route_id
    assert projection["certified_set_state"].member_route_ids == [certificate.selected_route_id, "route_b"]


def test_route_response_revalidates_abstention_mutation_without_certified_set_drift() -> None:
    response = RouteResponse(
        selected=_route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
        candidates=[
            _route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0),
            _route("route_b", distance_km=11.0, duration_s=21.0, monetary_cost=31.0, emissions_kg=5.0),
        ],
        run_id="run-abstain",
        pipeline_mode="voi",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
    )

    response.abstention = AbstentionRecord(reason_code="uncertified_due_to_search", message="search blocked")

    encoded = json.loads(response.model_dump_json())
    assert response.terminal_type == "typed_abstention"
    assert encoded["terminal_type"] == "typed_abstention"
    assert response.certified_set == []
    assert encoded["certified_set"] == []
    assert response.certified_set_summary["member_route_ids"] == []
    assert response.certified_set_summary["certified"] is False
    assert encoded["abstention"]["reason_code"] == "uncertified_due_to_search"


def test_assemble_decision_package_preserves_rich_certified_set_summary_payload() -> None:
    selected = _route(
        "route_a",
        distance_km=10.0,
        duration_s=20.0,
        monetary_cost=30.0,
        emissions_kg=4.0,
    )
    challenger = _route(
        "route_b",
        distance_km=11.0,
        duration_s=21.0,
        monetary_cost=31.0,
        emissions_kg=5.0,
    )
    selected_certificate = RouteCertificationSummary(
        route_id="route_a",
        certificate=0.91,
        certified=True,
        threshold=0.8,
        active_families=["scenario"],
        top_fragility_families=["scenario"],
        top_competitor_route_id="route_b",
        top_value_of_refresh_family="scenario",
        ambiguity_context={"support_strength": True},
    )
    certified_set_summary = {
        "member_route_ids": ["route_a", "route_b"],
        "excluded_route_ids": ["route_c"],
        "exclusion_basis": [
            "outside_routes_excluded",
            "singleton_not_justified:frontier_pairwise_gap_unresolved",
        ],
        "certified": True,
        "threshold": 0.8,
        "support_flag": True,
        "outside_routes_safely_excluded": True,
        "set_size": 2,
        "witness": {
            "route_id": "route_a",
            "active_challenger_ids": ["route_b"],
            "support_flag": True,
            "singleton_not_justified_reasons": ["frontier_pairwise_gap_unresolved"],
            "outside_routes_excluded": True,
            "outside_routes_safely_excluded": True,
            "excluded_route_safety_reasons": [],
        },
    }

    decision = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-certified-set",
        pipeline_mode="dccs_refc",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
        selected_certificate=selected_certificate,
        voi_stop_summary=VoiStopSummary(
            final_route_id="route_a",
            certificate=0.91,
            certified=True,
            iteration_count=1,
            search_budget_used=0,
            evidence_budget_used=0,
            stop_reason="certified_set",
        ),
        preference_state=build_preference_state(
            route_ids=["route_a", "route_b"],
            weights={"time": 1.0, "money": 0.0, "co2": 0.0},
            support_flag=True,
            support_reason=None,
        ),
        preference_query_trace={},
        world_support_summary={
            "schema_version": "world-support-summary-v1",
            "selected_route_id": "route_a",
            "selected_certificate_basis": "selected_certificate",
            "support_flag": True,
            "support_state": {
                "support_flag": True,
                "support_bin": "supported",
                "out_of_support_reason": None,
            },
            "world_bundle_summary": {
                "multi_fidelity_summary": {
                    "audit_world_count": 1,
                    "multi_fidelity_certificate_basis": "corrected_from_residual_model",
                    "audit_correction_mass": 2.5,
                }
            },
        },
        world_manifest={},
        winner_confidence_state=None,
        pairwise_gap_states=[],
        certified_set=[selected, challenger],
        certified_set_summary=certified_set_summary,
        abstention=None,
        flip_radius_state=None,
        decision_region_state=None,
        certificate_witness=None,
    )

    encoded = json.loads(decision.model_dump_json())

    assert decision.terminal_type == "certified_set"
    assert decision.certified_set_summary["exclusion_basis"] == certified_set_summary["exclusion_basis"]
    assert decision.certified_set_summary["outside_routes_safely_excluded"] is True
    assert decision.certified_set_summary["witness"]["singleton_not_justified_reasons"] == [
        "frontier_pairwise_gap_unresolved"
    ]
    assert decision.certified_set_summary["witness"]["outside_routes_excluded"] is True
    assert decision.certified_set_summary["witness"]["outside_routes_safely_excluded"] is True
    assert decision.certified_set_summary["witness"]["excluded_route_safety_reasons"] == []
    assert encoded["certified_set_summary"]["exclusion_basis"] == certified_set_summary["exclusion_basis"]
    assert encoded["certified_set_summary"]["outside_routes_safely_excluded"] is True
    assert encoded["certified_set_summary"]["witness"]["singleton_not_justified_reasons"] == [
        "frontier_pairwise_gap_unresolved"
    ]
    assert encoded["certified_set_summary"]["witness"]["outside_routes_safely_excluded"] is True
    assert encoded["certified_set_summary"]["witness"]["excluded_route_safety_reasons"] == []


def test_assemble_decision_package_synthesizes_marked_certified_set_summary_when_explicit_summary_missing() -> None:
    selected = _route(
        "route_a",
        distance_km=10.0,
        duration_s=20.0,
        monetary_cost=30.0,
        emissions_kg=4.0,
    )
    challenger = _route(
        "route_b",
        distance_km=11.0,
        duration_s=21.0,
        monetary_cost=31.0,
        emissions_kg=5.0,
    )

    decision = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id="run-certified-set-missing-summary",
        pipeline_mode="dccs_refc",
        manifest_endpoint="/manifest",
        artifacts_endpoint="/artifacts",
        provenance_endpoint="/provenance",
        selected_certificate=RouteCertificationSummary(
            route_id="route_a",
            certificate=0.91,
            certified=True,
            threshold=0.8,
            active_families=["scenario"],
            top_fragility_families=["scenario"],
            top_competitor_route_id="route_b",
            top_value_of_refresh_family="scenario",
            ambiguity_context={"support_strength": True},
        ),
        voi_stop_summary=VoiStopSummary(
            final_route_id="route_a",
            certificate=0.91,
            certified=True,
            iteration_count=1,
            search_budget_used=0,
            evidence_budget_used=0,
            stop_reason="certified_set",
        ),
        preference_state=build_preference_state(
            route_ids=["route_a", "route_b"],
            weights={"time": 1.0, "money": 0.0, "co2": 0.0},
            support_flag=True,
            support_reason=None,
        ),
        preference_query_trace={},
        world_support_summary={
            "schema_version": "world-support-summary-v1",
            "selected_route_id": "route_a",
            "selected_certificate_basis": "selected_certificate",
            "support_flag": True,
            "support_state": {
                "support_flag": True,
                "support_bin": "supported",
                "out_of_support_reason": None,
            },
        },
        world_manifest={},
        winner_confidence_state=None,
        pairwise_gap_states=[],
        certified_set=[selected, challenger],
        certified_set_summary=None,
        abstention=None,
        flip_radius_state=None,
        decision_region_state=None,
        certificate_witness=None,
    )

    encoded = json.loads(decision.model_dump_json())

    assert decision.terminal_type == "certified_set"
    assert decision.certified_set_summary["member_route_ids"] == ["route_a", "route_b"]
    assert decision.certified_set_summary["excluded_route_ids"] == []
    assert "explicit_certified_set_summary_missing" in decision.certified_set_summary["exclusion_basis"]
    assert decision.certified_set_summary["certified"] is True
    assert decision.certified_set_summary["support_flag"] is True
    assert decision.certified_set_summary["threshold"] == 0.8
    assert decision.certified_set_summary["witness"]["route_id"] == "route_a"
    assert decision.certified_set_summary["witness"]["active_challenger_ids"] == ["route_b"]
    assert (
        decision.certified_set_summary["witness"]["summary_status"]
        == "synthesized_without_explicit_certified_set_summary"
    )
    assert encoded["certified_set_summary"]["member_route_ids"] == ["route_a", "route_b"]
    assert encoded["certified_set_summary"]["excluded_route_ids"] == []
    assert "explicit_certified_set_summary_missing" in encoded["certified_set_summary"]["exclusion_basis"]
    assert encoded["certified_set_summary"]["certified"] is True
    assert encoded["certified_set_summary"]["support_flag"] is True
    assert encoded["certified_set_summary"]["threshold"] == 0.8
    assert (
        encoded["certified_set_summary"]["witness"]["summary_status"]
        == "synthesized_without_explicit_certified_set_summary"
    )


def test_write_route_run_bundle_emits_direct_refc_certified_set_bundle_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    selected = _route("route_a", distance_km=10.0, duration_s=20.0, monetary_cost=30.0, emissions_kg=4.0)
    challenger = _route("route_b", distance_km=11.0, duration_s=21.0, monetary_cost=31.0, emissions_kg=5.0)
    certified_set_summary = {
        "member_route_ids": ["route_a", "route_b"],
        "excluded_route_ids": ["route_c"],
        "exclusion_basis": [
            "outside_routes_excluded",
            "singleton_not_justified:frontier_pairwise_gap_unresolved",
        ],
        "certified": True,
        "threshold": 0.8,
        "support_flag": True,
        "outside_routes_safely_excluded": True,
        "set_size": 2,
        "witness": {
            "route_id": "route_a",
            "active_challenger_ids": ["route_b"],
            "support_flag": True,
            "singleton_not_justified_reasons": ["frontier_pairwise_gap_unresolved"],
            "outside_routes_excluded": True,
            "outside_routes_safely_excluded": True,
            "excluded_route_safety_reasons": [],
        },
    }

    route_run = _write_route_run_bundle(
        req=RouteRequest(
            origin=LatLng(lat=51.5, lon=-2.6),
            destination=LatLng(lat=51.6, lon=-2.5),
            vehicle_type="rigid_hgv",
            scenario_mode="no_sharing",
            max_alternatives=2,
            pipeline_mode="dccs_refc",
        ),
        selected=selected,
        candidates=[selected, challenger],
        warnings=[],
        candidate_diag=CandidateDiagnostics(selected_candidate_count=2),
        request_id="req-certified-set-bundle",
        pipeline_mode="dccs_refc",
        run_seed=20260410,
        duration_ms=12.5,
        selected_certificate=RouteCertificationSummary(
            route_id="route_a",
            certificate=0.91,
            certified=True,
            threshold=0.8,
            active_families=["scenario"],
            top_fragility_families=["scenario"],
            top_competitor_route_id="route_b",
            top_value_of_refresh_family="scenario",
            ambiguity_context={"support_strength": True},
        ),
        voi_stop_summary=VoiStopSummary(
            final_route_id="route_a",
            certificate=0.91,
            certified=True,
            iteration_count=1,
            search_budget_used=0,
            evidence_budget_used=0,
            stop_reason="certified_set",
        ),
        extra_json_artifacts={
            "decision_package.json": {
                "schema_version": "1.0.0",
                "terminal_type": "certified_set",
                "selected_route_id": "route_a",
                "selected_certificate_basis": "selected_certificate",
                "certified_set_summary": dict(certified_set_summary),
            },
            "certified_set_summary.json": dict(certified_set_summary),
        },
    )

    run_id = str(route_run["run_id"])
    artifact_dir = artifact_dir_for_run(run_id)
    certified_set_path = artifact_dir / "certified_set_summary.json"
    decision_path = artifact_dir / "decision_package.json"
    metadata_path = artifact_dir / "metadata.json"

    emitted_certified_set = json.loads(certified_set_path.read_text(encoding="utf-8"))
    emitted_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    emitted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert artifact_dir.exists()
    assert certified_set_path.exists()
    assert decision_path.exists()
    assert metadata_path.exists()
    assert route_run["manifest_endpoint"] == f"/runs/{run_id}/manifest"
    assert route_run["artifacts_endpoint"] == f"/runs/{run_id}/artifacts"
    assert route_run["provenance_endpoint"] == f"/runs/{run_id}/provenance"
    assert "certified_set_summary.json" in emitted_metadata["artifact_names"]
    assert "decision_package.json" in emitted_metadata["artifact_names"]
    assert emitted_certified_set["member_route_ids"] == ["route_a", "route_b"]
    assert emitted_certified_set["exclusion_basis"] == certified_set_summary["exclusion_basis"]
    assert emitted_certified_set["outside_routes_safely_excluded"] is True
    assert emitted_certified_set["witness"]["route_id"] == "route_a"
    assert emitted_certified_set["witness"]["active_challenger_ids"] == ["route_b"]
    assert emitted_certified_set["witness"]["singleton_not_justified_reasons"] == [
        "frontier_pairwise_gap_unresolved"
    ]
    assert emitted_certified_set["witness"]["outside_routes_safely_excluded"] is True
    assert emitted_certified_set["witness"]["excluded_route_safety_reasons"] == []
    assert emitted_decision["terminal_type"] == "certified_set"
    assert emitted_decision["selected_certificate_basis"] == "selected_certificate"
    assert emitted_decision["certified_set_summary"]["member_route_ids"] == emitted_certified_set["member_route_ids"]
    assert emitted_decision["certified_set_summary"]["exclusion_basis"] == emitted_certified_set["exclusion_basis"]
    assert emitted_decision["certified_set_summary"]["outside_routes_safely_excluded"] is True
    assert emitted_decision["certified_set_summary"]["witness"]["route_id"] == "route_a"


def _install_direct_refc_certified_set_pipeline_stubs(monkeypatch, *, out_dir: str | None = None) -> None:
    def _make_ranked_route(*, route_id: str, duration_s: float, lon_seed: float, road_class: str) -> dict[str, Any]:
        return {
            "route_id": route_id,
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-1.9000 + lon_seed, 52.5000],
                    [-1.8500 + lon_seed, 52.5200],
                    [-1.7800 + lon_seed, 52.5600],
                ],
            },
            "duration": float(duration_s),
            "distance": 32_000.0,
            "_graph_meta": {
                "road_mix_counts": {road_class: 5},
                "toll_edges": 0,
            },
        }

    selected_raw = _make_ranked_route(route_id="route_a", duration_s=1_200.0, lon_seed=0.0, road_class="motorway")
    challenger_raw = _make_ranked_route(route_id="route_b", duration_s=1_260.0, lon_seed=0.2, road_class="trunk")

    async def _fake_scenario_context_from_od(**_: Any) -> dict[str, Any]:
        return {"bucket": "clear"}

    async def _fake_scenario_candidate_modifiers_async(**_: Any) -> dict[str, Any]:
        return {}

    def _fake_feasibility(**_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "reason_code": "ok",
            "origin_node_id": "a",
            "destination_node_id": "b",
        }

    async def _fake_k_raw_search(**_: Any):
        return (
            [dict(selected_raw), dict(challenger_raw)],
            main_module.GraphCandidateDiagnostics(
                explored_states=4,
                generated_paths=2,
                emitted_paths=2,
                candidate_budget=2,
            ),
            {
                "graph_retry_attempted": False,
                "graph_retry_state_budget": 0,
                "graph_retry_outcome": "not_applicable",
                "graph_rescue_attempted": False,
                "graph_rescue_mode": "not_applicable",
                "graph_rescue_state_budget": 0,
                "graph_rescue_outcome": "not_applicable",
                "graph_search_ms_initial": 11.0,
                "graph_search_ms_retry": 0.0,
                "graph_search_ms_rescue": 0.0,
            },
        )

    async def _unexpected_iter_candidate_fetches(**_: Any):
        raise AssertionError("supplemental candidate fetches should not run in this smoke")
        yield

    async def _unexpected_fetch_local_ors_baseline_seed(**_: Any):
        raise AssertionError("local ORS seed should not run in this smoke")

    async def _fake_refine_graph_candidate_batch(**kwargs: Any):
        selected_records = list(kwargs.get("selected_records", []))
        raw_routes_by_id = dict(kwargs.get("raw_graph_routes_by_id", {}))
        observed = {str(record.candidate_id): 6.0 for record in selected_records}
        refined_routes = [dict(raw_routes_by_id[str(record.candidate_id)]) for record in selected_records]
        return refined_routes, [], observed, 0, 5.0

    def _fake_build_options(routes: list[dict[str, Any]], **_: Any):
        options: list[RouteOption] = []
        for route in routes:
            route_id = str(route.get("route_id"))
            route["_built_option_id"] = route_id
            options.append(
                _route(
                    route_id,
                    distance_km=float(route["distance"]) / 1000.0,
                    duration_s=float(route["duration"]),
                    monetary_cost=200.0 if route_id == "route_a" else 210.0,
                    emissions_kg=100.0 if route_id == "route_a" else 110.0,
                )
            )
        return options, [], main_module.TerrainDiagnostics()

    def _fake_compute_frontier_certification(**kwargs: Any):
        frontier_options = list(kwargs["frontier_options"])
        selected_route_id = str(kwargs["selected_route_id"])
        threshold = float(kwargs["threshold"])
        routes = [
            {
                "route_id": option.id,
                "objective": {
                    "time": float(option.metrics.duration_s),
                    "money": float(option.metrics.monetary_cost),
                    "co2": float(option.metrics.emissions_kg),
                },
                "evidence": {
                    "scenario": (
                        {"time": 1.0, "money": 1.0, "co2": 1.0}
                        if option.id == "route_a"
                        else {"time": 0.9, "money": 1.1, "co2": 1.0}
                    )
                },
            }
            for option in frontier_options
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
        certificate = compute_certificate(routes, worlds=worlds, threshold=threshold)
        fragility = compute_fragility_maps(
            routes,
            worlds=worlds,
            selected_route_id=certificate.selected_route_id,
        )
        manifest_payload = dict(certificate.world_manifest)
        manifest_payload.update(
            {
                "worlds": worlds,
                "active_families": list(certificate.world_manifest.get("active_families", ["scenario"])),
                "selected_route_id": selected_route_id,
                "world_count": len(worlds),
                "requested_world_count": len(worlds),
                "effective_world_count": len(worlds),
                "unique_world_count": len(worlds),
                "world_count_policy": "configured",
                "world_reuse_rate": 0.0,
                "support_flag": True,
                "support_reason": None,
                "support_bin": "supported",
                "calibration_bin": "selected_certificate",
                "selected_certificate_basis": "selected_certificate",
                "forced_refreshed_families": [],
            }
        )
        return (
            certificate,
            fragility,
            manifest_payload,
            list(manifest_payload["active_families"]),
        )

    if out_dir is not None:
        monkeypatch.setattr(settings, "out_dir", out_dir)
    monkeypatch.setattr(main_module, "refresh_live_runtime_route_caches", lambda **_: None)
    monkeypatch.setattr(main_module, "_scenario_context_from_od", _fake_scenario_context_from_od)
    monkeypatch.setattr(main_module, "_scenario_candidate_modifiers_async", _fake_scenario_candidate_modifiers_async)
    monkeypatch.setattr(main_module, "route_graph_od_feasibility", _fake_feasibility)
    monkeypatch.setattr(main_module, "_route_graph_k_raw_search", _fake_k_raw_search)
    monkeypatch.setattr(main_module, "_iter_candidate_fetches", _unexpected_iter_candidate_fetches)
    monkeypatch.setattr(main_module, "_fetch_local_ors_baseline_seed", _unexpected_fetch_local_ors_baseline_seed)
    monkeypatch.setattr(main_module, "_refine_graph_candidate_batch", _fake_refine_graph_candidate_batch)
    monkeypatch.setattr(main_module, "_build_options", _fake_build_options)
    monkeypatch.setattr(main_module, "_strict_frontier_options", lambda options, **_: list(options))
    monkeypatch.setattr(main_module, "_finalize_pareto_options", lambda options, **_: list(options))
    monkeypatch.setattr(
        main_module,
        "_pick_best_option",
        lambda options, **_: min(list(options), key=lambda item: float(item.metrics.duration_s)),
    )
    monkeypatch.setattr(main_module, "_should_hydrate_priority_route_options", lambda req: False)
    monkeypatch.setattr(
        main_module,
        "_route_selection_score_map",
        lambda options, **_: {str(option.id): float(option.metrics.duration_s) for option in options},
    )
    monkeypatch.setattr(main_module, "_compute_frontier_certification", _fake_compute_frontier_certification)

def _direct_refc_certified_set_request() -> RouteRequest:
    return RouteRequest(
        origin=LatLng(lat=51.5074, lon=-0.1278),
        destination=LatLng(lat=53.4808, lon=-2.2426),
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        max_alternatives=2,
        search_budget=2,
        certificate_threshold=0.5,
    )


def _run_direct_refc_certified_set_pipeline_smoke(monkeypatch, *, out_dir: str | None = None) -> tuple[RouteRequest, dict[str, Any]]:
    _install_direct_refc_certified_set_pipeline_stubs(monkeypatch, out_dir=out_dir)
    req = _direct_refc_certified_set_request()
    result = asyncio.run(
        main_module._compute_direct_route_pipeline(
            req=req,
            osrm=_NoopOSRM(),
            ors=_NoopORS(),
            max_alternatives=2,
            pipeline_mode="dccs_refc",
            run_seed=20260410,
        )
    )
    return req, result


def test_compute_direct_route_pipeline_synthesize_refc_certified_set_artifacts(monkeypatch) -> None:
    _, result = _run_direct_refc_certified_set_pipeline_smoke(monkeypatch)

    artifacts = result["extra_json_artifacts"]
    decision_package = artifacts["decision_package.json"]
    certified_set_summary = artifacts["certified_set_summary.json"]
    winner_confidence_state = artifacts["winner_confidence_state.json"]
    certificate_witness = artifacts["certificate_witness.json"]
    decision_region_summary = artifacts["decision_region_summary.json"]
    certificate_summary = artifacts["certificate_summary.json"]

    assert result["selected"].id == "route_a"
    assert result["selected_certificate"].certified is True
    assert decision_package["terminal_type"] == "certified_set"
    assert decision_package["artifact_pointers"]["certified_set_summary"] == "certified_set_summary.json"
    assert decision_package["certified_set_summary"]["member_route_ids"] == certified_set_summary["member_route_ids"]
    assert set(certified_set_summary["member_route_ids"]) == {"route_a", "route_b"}
    assert certified_set_summary["set_size"] == 2
    assert certified_set_summary["certified"] is False
    assert certified_set_summary["support_flag"] is True
    assert certified_set_summary["witness"]["route_id"] == "route_a"
    assert certified_set_summary["witness"]["active_challenger_ids"] == ["route_b"]
    assert certified_set_summary["witness"]["singleton_not_justified_reasons"] == [
        "winner_lcb_below_threshold"
    ]
    assert winner_confidence_state["route_id"] == "route_a"
    assert certificate_witness["route_id"] == "route_a"
    assert certificate_witness["active_challenger_ids"] == ["route_b"]
    assert decision_region_summary["route_id"] == "route_a"
    assert decision_region_summary["active_challenger_id"] == "route_b"
    assert (
        decision_package["certificate_summary"]["threshold_sensitivity_axes"]["certified_set_cap"]["is_alias"]
        is True
    )
    assert (
        certificate_summary["threshold_sensitivity_axes"]["certified_set_cap"]["truthful_semantics"]
        == "low_ambiguity_adaptive_refc_world_count_cap"
    )


def test_direct_refc_pipeline_bundle_writer_preserves_certified_set_artifact_family(tmp_path, monkeypatch) -> None:
    req, result = _run_direct_refc_certified_set_pipeline_smoke(monkeypatch, out_dir=str(tmp_path))

    route_run = _write_route_run_bundle(
        req=req,
        selected=result["selected"],
        candidates=result["candidates"],
        warnings=result["warnings"],
        candidate_diag=result["candidate_diag"],
        request_id="req-direct-refc-parity",
        pipeline_mode="dccs_refc",
        run_seed=20260410,
        duration_ms=12.5,
        selected_certificate=result["selected_certificate"],
        voi_stop_summary=result["voi_stop_summary"],
        extra_json_artifacts=result["extra_json_artifacts"],
    )

    artifact_dir = artifact_dir_for_run(str(route_run["run_id"]))
    emitted_decision = json.loads((artifact_dir / "decision_package.json").read_text(encoding="utf-8"))
    emitted_certified_set = json.loads((artifact_dir / "certified_set_summary.json").read_text(encoding="utf-8"))
    emitted_winner_confidence = json.loads((artifact_dir / "winner_confidence_state.json").read_text(encoding="utf-8"))
    emitted_certificate_witness = json.loads((artifact_dir / "certificate_witness.json").read_text(encoding="utf-8"))
    emitted_decision_region = json.loads((artifact_dir / "decision_region_summary.json").read_text(encoding="utf-8"))
    emitted_metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    runtime_artifacts = result["extra_json_artifacts"]
    emitted_decision_without_provenance = dict(emitted_decision)
    emitted_decision_without_provenance.pop("artifact_provenance", None)
    emitted_certified_set_without_provenance = dict(emitted_certified_set)
    emitted_certified_set_without_provenance.pop("artifact_provenance", None)
    emitted_winner_confidence_without_provenance = dict(emitted_winner_confidence)
    emitted_winner_confidence_without_provenance.pop("artifact_provenance", None)
    emitted_certificate_witness_without_provenance = dict(emitted_certificate_witness)
    emitted_certificate_witness_without_provenance.pop("artifact_provenance", None)
    emitted_decision_region_without_provenance = dict(emitted_decision_region)
    emitted_decision_region_without_provenance.pop("artifact_provenance", None)
    assert emitted_decision_without_provenance == json.loads(json.dumps(runtime_artifacts["decision_package.json"]))
    assert emitted_certified_set_without_provenance == json.loads(json.dumps(runtime_artifacts["certified_set_summary.json"]))
    assert emitted_winner_confidence_without_provenance == json.loads(json.dumps(runtime_artifacts["winner_confidence_state.json"]))
    assert emitted_certificate_witness_without_provenance == json.loads(json.dumps(runtime_artifacts["certificate_witness.json"]))
    assert emitted_decision_region_without_provenance == json.loads(json.dumps(runtime_artifacts["decision_region_summary.json"]))
    for artifact_name in [
        "decision_package.json",
        "certified_set_summary.json",
        "winner_confidence_state.json",
        "certificate_witness.json",
        "decision_region_summary.json",
    ]:
        assert artifact_name in emitted_metadata["artifact_names"]


def test_direct_refc_pipeline_bundle_writer_emits_exact_certified_set_disk_artifact_family(
    tmp_path, monkeypatch
) -> None:
    req, result = _run_direct_refc_certified_set_pipeline_smoke(monkeypatch, out_dir=str(tmp_path))

    route_run = _write_route_run_bundle(
        req=req,
        selected=result["selected"],
        candidates=result["candidates"],
        warnings=result["warnings"],
        candidate_diag=result["candidate_diag"],
        request_id="req-direct-refc-disk-artifacts",
        pipeline_mode="dccs_refc",
        run_seed=20260410,
        duration_ms=12.5,
        selected_certificate=result["selected_certificate"],
        voi_stop_summary=result["voi_stop_summary"],
        extra_json_artifacts=result["extra_json_artifacts"],
    )

    artifact_dir = artifact_dir_for_run(str(route_run["run_id"]))
    emitted_metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    runtime_artifacts = result["extra_json_artifacts"]
    expected_artifact_names = [
        "certificate_summary.json",
        "route_fragility_map.json",
        "competitor_fragility_breakdown.json",
        "value_of_refresh.json",
        "sampled_world_manifest.json",
        "evidence_snapshot_manifest.json",
        "world_support_summary.json",
        "flip_radius_summary.json",
        "decision_region_summary.json",
        "certificate_witness.json",
        "certified_set_summary.json",
    ]

    for artifact_name in expected_artifact_names:
        artifact_path = artifact_dir / artifact_name
        emitted_payload = json.loads(artifact_path.read_text(encoding="utf-8"))

        assert artifact_name in runtime_artifacts
        assert artifact_path.exists()
        assert isinstance(emitted_payload, dict)
        emitted_payload_without_provenance = dict(emitted_payload)
        emitted_payload_without_provenance.pop("artifact_provenance", None)
        assert emitted_payload_without_provenance == json.loads(
            json.dumps(runtime_artifacts[artifact_name])
        )
        assert artifact_name in emitted_metadata["artifact_names"]

    emitted_route_fragility = json.loads(
        (artifact_dir / "route_fragility_map.json").read_text(encoding="utf-8")
    )
    emitted_competitor_fragility = json.loads(
        (artifact_dir / "competitor_fragility_breakdown.json").read_text(encoding="utf-8")
    )
    emitted_world_manifest = json.loads(
        (artifact_dir / "sampled_world_manifest.json").read_text(encoding="utf-8")
    )

    assert "deterministic_local_flip_radius" in emitted_route_fragility["route_a"]
    assert "probabilistic_flip_radius" in emitted_route_fragility["route_a"]
    assert "family_specific_radii" in emitted_route_fragility["route_a"]
    assert "dominant_fragility_family" in emitted_route_fragility["route_a"]
    assert "adversarial_degradation_curve" in emitted_route_fragility["route_a"]
    assert "pairwise_gap_lower_bound" in emitted_competitor_fragility["route_a"]["route_b"]
    assert "pairwise_gap_upper_bound" in emitted_competitor_fragility["route_a"]["route_b"]
    assert "challenger_radius" in emitted_competitor_fragility["route_a"]["route_b"]
    assert "challenger_audit_sensitivity" in emitted_competitor_fragility["route_a"]["route_b"]
    assert "probabilistic_worlds" in emitted_world_manifest
    assert "audit_worlds" in emitted_world_manifest
    assert "proxy_only_worlds" in emitted_world_manifest
    assert "audited_worlds" in emitted_world_manifest
    assert "reused_worlds" in emitted_world_manifest
    assert emitted_world_manifest["support_bins"]["support_bin"] == "supported"
    assert "calibration_policy_version" in emitted_world_manifest


def test_compute_route_direct_refc_certified_set_returns_bundle_consistent_decision_package(
    tmp_path, monkeypatch
) -> None:
    req, _ = _run_direct_refc_certified_set_pipeline_smoke(monkeypatch, out_dir=str(tmp_path))

    class _NoopOSRM:
        async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
            return []

    class _NoopORS:
        async def fetch_route(self, **_: Any) -> Any:
            raise AssertionError("local ORS should not be used in this compute_route smoke")

    monkeypatch.setattr(main_module, "_routing_graph_warmup_failfast_detail", lambda: None)
    monkeypatch.setattr(main_module, "_resolve_pipeline_seed", lambda _: 20260410)
    monkeypatch.setattr(main_module, "_validate_route_options_evidence", lambda _: {"status": "ok", "issues": []})

    response = Response()
    decision = asyncio.run(main_module.compute_route(req, response, _NoopOSRM(), _NoopORS(), None))
    encoded = json.loads(decision.model_dump_json())

    artifact_dir = artifact_dir_for_run(str(decision.run_id))
    emitted_decision = json.loads((artifact_dir / "decision_package.json").read_text(encoding="utf-8"))
    emitted_certified_set = json.loads((artifact_dir / "certified_set_summary.json").read_text(encoding="utf-8"))
    emitted_winner_confidence = json.loads((artifact_dir / "winner_confidence_state.json").read_text(encoding="utf-8"))
    emitted_certificate_witness = json.loads((artifact_dir / "certificate_witness.json").read_text(encoding="utf-8"))
    emitted_metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert encoded["terminal_type"] == "typed_abstention"
    assert encoded["selected_certificate_basis"] == "selected_certificate"
    assert response.headers["x-route-request-id"]
    assert encoded["artifact_pointers"] == {
        "manifest_endpoint": f"/runs/{encoded['run_id']}/manifest",
        "artifacts_endpoint": f"/runs/{encoded['run_id']}/artifacts",
        "provenance_endpoint": f"/runs/{encoded['run_id']}/provenance",
    }
    assert emitted_metadata["manifest_endpoint"] == encoded["artifact_pointers"]["manifest_endpoint"]
    assert emitted_metadata["artifacts_endpoint"] == encoded["artifact_pointers"]["artifacts_endpoint"]
    assert emitted_metadata["provenance_endpoint"] == encoded["artifact_pointers"]["provenance_endpoint"]
    assert "evidence_validation.json" in emitted_metadata["artifact_names"]
    assert emitted_decision["terminal_type"] == "certified_set"
    assert emitted_decision["selected_certificate_basis"] == encoded["selected_certificate_basis"]
    assert encoded["certified_set_summary"]["terminal_type"] == "typed_abstention"
    assert encoded["certified_set_summary"]["member_route_ids"] == []
    assert encoded["certified_set_summary"]["not_applicable_reason"] == "abstention_terminal"
    assert set(emitted_certified_set["member_route_ids"]) == {"route_a", "route_b"}
    assert set(encoded["certified_set_summary"]["excluded_route_ids"]) == set(emitted_certified_set["member_route_ids"])
    assert encoded["certified_set_summary"]["exclusion_basis"] == emitted_certified_set["exclusion_basis"]
    assert (
        encoded["certified_set_summary"]["witness"]["singleton_not_justified_reasons"]
        == emitted_certified_set["witness"]["singleton_not_justified_reasons"]
    )
    assert emitted_decision["certified_set_summary"]["member_route_ids"] == emitted_certified_set["member_route_ids"]
    assert emitted_decision["certified_set_summary"]["exclusion_basis"] == emitted_certified_set["exclusion_basis"]
    assert encoded["winner_confidence_state"]["route_id"] == emitted_winner_confidence["route_id"] == "route_a"
    assert encoded["certificate_witness"]["route_id"] == emitted_certificate_witness["route_id"] == "route_a"
    assert encoded["certificate_witness"]["active_challenger_ids"] == emitted_certificate_witness[
        "active_challenger_ids"
    ] == ["route_b"]


def test_route_http_direct_refc_returns_bundle_consistent_decision_package(tmp_path, monkeypatch) -> None:
    class _NoopOSRM:
        async def fetch_routes(self, **_: Any) -> list[dict[str, Any]]:
            return []

    class _NoopORS:
        async def fetch_route(self, **_: Any) -> Any:
            raise AssertionError("local ORS should not be used in this route HTTP smoke")

    _install_direct_refc_certified_set_pipeline_stubs(monkeypatch, out_dir=str(tmp_path))
    monkeypatch.setattr(main_module, "_routing_graph_warmup_failfast_detail", lambda: None)
    monkeypatch.setattr(main_module, "_resolve_pipeline_seed", lambda _: 20260410)
    monkeypatch.setattr(main_module, "_validate_route_options_evidence", lambda _: {"status": "ok", "issues": []})

    payload = _direct_refc_certified_set_request().model_dump(mode="json")
    app.dependency_overrides[osrm_client] = lambda: _NoopOSRM()
    app.dependency_overrides[ors_client] = lambda: _NoopORS()
    try:
        with TestClient(app) as client:
            http_response = client.post("/route", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert http_response.status_code == 200
    assert http_response.headers["x-route-request-id"]
    body = http_response.json()

    artifact_dir = artifact_dir_for_run(str(body["run_id"]))
    emitted_decision = json.loads((artifact_dir / "decision_package.json").read_text(encoding="utf-8"))
    emitted_certified_set = json.loads((artifact_dir / "certified_set_summary.json").read_text(encoding="utf-8"))
    emitted_winner_confidence = json.loads((artifact_dir / "winner_confidence_state.json").read_text(encoding="utf-8"))
    emitted_certificate_witness = json.loads((artifact_dir / "certificate_witness.json").read_text(encoding="utf-8"))
    emitted_metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert body["terminal_type"] == "typed_abstention"
    assert body["selected_certificate_basis"] == "selected_certificate"
    assert body["artifact_pointers"] == {
        "manifest_endpoint": f"/runs/{body['run_id']}/manifest",
        "artifacts_endpoint": f"/runs/{body['run_id']}/artifacts",
        "provenance_endpoint": f"/runs/{body['run_id']}/provenance",
    }
    assert emitted_metadata["manifest_endpoint"] == body["artifact_pointers"]["manifest_endpoint"]
    assert emitted_metadata["artifacts_endpoint"] == body["artifact_pointers"]["artifacts_endpoint"]
    assert emitted_metadata["provenance_endpoint"] == body["artifact_pointers"]["provenance_endpoint"]
    assert "evidence_validation.json" in emitted_metadata["artifact_names"]
    assert emitted_decision["terminal_type"] == "certified_set"
    assert emitted_decision["selected_certificate_basis"] == body["selected_certificate_basis"]
    assert body["certified_set_summary"]["terminal_type"] == "typed_abstention"
    assert body["certified_set_summary"]["member_route_ids"] == []
    assert body["certified_set_summary"]["not_applicable_reason"] == "abstention_terminal"
    assert set(emitted_certified_set["member_route_ids"]) == {"route_a", "route_b"}
    assert set(body["certified_set_summary"]["excluded_route_ids"]) == set(emitted_certified_set["member_route_ids"])
    assert body["certified_set_summary"]["exclusion_basis"] == emitted_certified_set["exclusion_basis"]
    assert (
        body["certified_set_summary"]["witness"]["singleton_not_justified_reasons"]
        == emitted_certified_set["witness"]["singleton_not_justified_reasons"]
    )
    assert body["winner_confidence_state"]["route_id"] == emitted_winner_confidence["route_id"] == "route_a"
    assert body["certificate_witness"]["route_id"] == emitted_certificate_witness["route_id"] == "route_a"
    assert body["certificate_witness"]["active_challenger_ids"] == emitted_certificate_witness[
        "active_challenger_ids"
    ] == ["route_b"]


def test_route_http_direct_refc_degrades_continue_on_deferred_route_graph_precheck(
    tmp_path,
    monkeypatch,
) -> None:
    _install_direct_refc_certified_set_pipeline_stubs(monkeypatch, out_dir=str(tmp_path))
    monkeypatch.setattr(main_module, "_routing_graph_warmup_failfast_detail", lambda: None)
    monkeypatch.setattr(main_module, "_resolve_pipeline_seed", lambda _: 20260410)
    monkeypatch.setattr(main_module, "_validate_route_options_evidence", lambda _: {"status": "ok", "issues": []})
    monkeypatch.setattr(
        main_module,
        "route_graph_od_feasibility",
        lambda **_: {
            "ok": False,
            "reason_code": "routing_graph_deferred_load",
            "message": (
                "Route graph full load is deferred in fast-startup mode; "
                "using OSRM family fallback for this request."
            ),
        },
    )

    payload = _direct_refc_certified_set_request().model_dump(mode="json")
    app.dependency_overrides[osrm_client] = lambda: _NoopOSRM()
    app.dependency_overrides[ors_client] = lambda: _NoopORS()
    try:
        with TestClient(app) as client:
            http_response = client.post("/route", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert http_response.status_code == 200
    body = http_response.json()
    artifact_dir = artifact_dir_for_run(str(body["run_id"]))
    emitted_trace = json.loads((artifact_dir / "final_route_trace.json").read_text(encoding="utf-8"))
    candidate_diagnostics = emitted_trace.get("candidate_diagnostics", {})

    assert candidate_diagnostics["precheck_reason_code"] == "routing_graph_deferred_load"
    assert candidate_diagnostics["precheck_gate_action"] == "degraded_continue"
