from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app, ors_client, osrm_client
from app.models import (
    EvidenceProvenance,
    EvidenceSourceRecord,
    GeoJSONLineString,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    ScenarioMode,
    ScenarioSummary,
)
from app.run_store import artifact_dir_for_run
from app.settings import settings


class FakeDirectOSRM:
    pass


class FakeDirectORS:
    pass


def _payload() -> dict[str, Any]:
    return {
        "origin": {"lat": 52.4862, "lon": -1.8904},
        "destination": {"lat": 51.5072, "lon": -0.1276},
        "vehicle_type": "rigid_hgv",
        "pipeline_mode": "tri_source",
        "certificate_threshold": 0.7,
        "od_ambiguity_index": 0.18,
        "od_ambiguity_confidence": 0.12,
        "od_engine_disagreement_prior": 0.08,
        "od_hard_case_prior": 0.10,
        "od_ambiguity_source_count": 1,
        "od_ambiguity_source_mix": "routing_graph_probe",
        "od_ambiguity_source_mix_count": 1,
        "od_ambiguity_source_entropy": 0.18,
        "od_ambiguity_support_ratio": 0.27,
        "od_ambiguity_prior_strength": 0.14,
        "od_ambiguity_family_density": 0.12,
        "od_ambiguity_margin_pressure": 0.09,
        "od_ambiguity_spread_pressure": 0.06,
        "od_candidate_path_count": 1,
        "od_corridor_family_count": 1,
        "od_objective_spread": 0.02,
        "od_nominal_margin_proxy": 0.62,
        "od_toll_disagreement_rate": 0.02,
        "ambiguity_budget_prior": 0.08,
        "ambiguity_budget_band": "low",
    }


def _model_only_provenance() -> EvidenceProvenance:
    return EvidenceProvenance(
        active_families=["stochastic"],
        families=[
            EvidenceSourceRecord(
                family="stochastic",
                source="stochastic_model",
                active=True,
                confidence=0.9,
                coverage_ratio=0.95,
                fallback_used=False,
            )
        ],
    )


def _route_option(
    route_id: str,
    *,
    monetary_cost: float,
    certificate: float,
    certified: bool,
) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[(-1.8904, 52.4862), (-0.1276, 51.5072)],
        ),
        metrics=RouteMetrics(
            distance_km=185.0,
            duration_s=8100.0,
            monetary_cost=monetary_cost,
            emissions_kg=144.2,
            avg_speed_kmh=82.2,
        ),
        scenario_summary=ScenarioSummary(
            mode=ScenarioMode.NO_SHARING,
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="pytest",
            version="pytest",
        ),
        evidence_provenance=_model_only_provenance(),
        certification=RouteCertificationSummary(
            route_id=route_id,
            certificate=certificate,
            certified=certified,
            threshold=0.7,
            active_families=["stochastic"],
            top_fragility_families=["stochastic"],
            top_competitor_route_id="abstain_route_b" if route_id == "abstain_route_a" else "abstain_route_a",
            top_value_of_refresh_family="stochastic",
        ),
    )


def test_route_tri_source_exposes_typed_abstention_and_persists_abstention_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))
    monkeypatch.setattr(main_module, "_routing_graph_warmup_failfast_detail", lambda: None)

    selected = _route_option(
        "abstain_route_a",
        monetary_cost=110.0,
        certificate=0.68,
        certified=False,
    )
    challenger = _route_option(
        "abstain_route_b",
        monetary_cost=112.0,
        certificate=0.66,
        certified=False,
    )

    async def _fake_compute_direct_route_pipeline(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["pipeline_mode"] == "tri_source"
        return {
            "selected": selected,
            "candidates": [selected, challenger],
            "warnings": [],
            "candidate_fetches": 1,
            "terrain_diag": main_module.TerrainDiagnostics(),
            "candidate_diag": main_module.CandidateDiagnostics(
                raw_count=2,
                deduped_count=2,
                candidate_budget=2,
            ),
            "selected_certificate": selected.certification,
            "voi_stop_summary": None,
            "extra_json_artifacts": {
                "certificate_summary.json": {
                    "selected_route_id": selected.id,
                    "selected_certificate": 0.68,
                    "selected_certificate_basis": "threshold_and_pairwise",
                    "route_certificates": {
                        selected.id: 0.68,
                        challenger.id: 0.66,
                    },
                    "frontier_route_ids": [selected.id, challenger.id],
                },
                "route_fragility_map.json": {
                    selected.id: {
                        "weather": 0.22,
                        "stochastic": 0.11,
                    },
                    challenger.id: {
                        "stochastic": 0.91,
                        "weather": 0.05,
                    },
                },
                "sampled_world_manifest.json": {
                    "manifest_hash": "sha256:typed-abstention-worlds",
                    "selected_route_id": selected.id,
                    "active_families": ["stochastic", "weather"],
                    "world_count": 44,
                    "effective_world_count": 40,
                    "requested_world_count": 44,
                    "unique_world_count": 44,
                    "world_count_policy": "targeted_stress",
                    "stress_world_fraction": 0.1,
                    "world_reuse_rate": 0.09,
                    "worlds": [
                        {
                            "world_id": "w0",
                            "world_kind": "sampled",
                            "target_route_id": selected.id,
                            "states": {"stochastic": "nominal", "weather": "nominal"},
                        }
                    ],
                },
                "evidence_snapshot_manifest.json": {
                    "manifest_hash": "sha256:typed-abstention-evidence",
                    "active_families": ["stochastic", "weather"],
                    "route_ids": [selected.id, challenger.id],
                    "family_snapshots": {
                        "stochastic": [
                            {
                                "source": "stochastic_model",
                                "confidence": 0.9,
                                "coverage_ratio": 0.95,
                                "route_id": selected.id,
                            }
                        ],
                        "weather": [
                            {
                                "source": "weather_model",
                                "confidence": 0.88,
                                "coverage_ratio": 0.91,
                                "route_id": selected.id,
                            }
                        ]
                    },
                },
                "final_route_trace.json": {
                    "pipeline_mode": "tri_source",
                    "selected_route_id": selected.id,
                    "artifact_pointers": {},
                },
            },
            "extra_jsonl_artifacts": {
                "strict_frontier.jsonl": [
                    {
                        "route_id": selected.id,
                        "selected": True,
                        "monetary_cost": 110.0,
                        "duration_s": 8100.0,
                        "emissions_kg": 144.2,
                        "certificate": 0.68,
                        "certificate_threshold": 0.7,
                        "certified": False,
                    },
                    {
                        "route_id": challenger.id,
                        "selected": False,
                        "monetary_cost": 112.0,
                        "duration_s": 8120.0,
                        "emissions_kg": 145.4,
                        "certificate": 0.66,
                        "certificate_threshold": 0.7,
                        "certified": False,
                    },
                ]
            },
            "extra_csv_artifacts": {},
            "extra_text_artifacts": {},
        }

    monkeypatch.setattr(
        main_module,
        "_compute_direct_route_pipeline",
        _fake_compute_direct_route_pipeline,
    )

    app.dependency_overrides[osrm_client] = lambda: FakeDirectOSRM()
    app.dependency_overrides[ors_client] = lambda: FakeDirectORS()
    try:
        with TestClient(app) as client:
            resp = client.post("/route", json=_payload())
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline_mode"] == "tri_source"
    assert data["decision_package"]["pipeline_mode"] == "tri_source"
    assert data["decision_package"]["terminal_kind"] == "typed_abstention"
    assert data["decision_package"]["selected_route_id"] == selected.id
    assert data["decision_package"]["abstention_summary"]["abstained"] is True
    reason_code = data["decision_package"]["abstention_summary"]["reason_code"]
    assert isinstance(reason_code, str)
    assert reason_code
    assert data["decision_package"]["abstention_summary"]["abstention_type"] == "typed_abstention_recommended"
    assert data["decision_package"]["abstention_summary"]["recommended_action"] == "expand_worlds"
    assert data["decision_package"]["certification_state_summary"]["abstained"] is True
    assert data["decision_package"]["world_fidelity_summary"]["unique_world_count"] == 44
    assert data["decision_package"]["world_fidelity_summary"]["effective_world_count"] == 40
    assert "top_fragility_family=weather" in data["decision_package"]["witness_summary"]["notes"]

    artifact_dir = artifact_dir_for_run(data["run_id"])
    decision_package = json.loads((artifact_dir / "decision_package.json").read_text(encoding="utf-8"))
    abstention_summary = json.loads((artifact_dir / "abstention_summary.json").read_text(encoding="utf-8"))
    final_route_trace = json.loads((artifact_dir / "final_route_trace.json").read_text(encoding="utf-8"))

    assert decision_package["terminal_kind"] == "typed_abstention"
    assert decision_package["abstention_summary"]["abstained"] is True
    assert abstention_summary["abstained"] is True
    assert abstention_summary["reason_code"] == reason_code
    assert abstention_summary["abstention_type"] == "typed_abstention_recommended"
    assert decision_package["world_fidelity_summary"]["unique_world_count"] == 44
    assert decision_package["world_fidelity_summary"]["effective_world_count"] == 40
    assert "top_fragility_family=weather" in decision_package["witness_summary"]["notes"]
    assert final_route_trace["artifact_pointers"]["decision_package"] == "decision_package.json"
    assert final_route_trace["artifact_pointers"]["abstention_summary"] == "abstention_summary.json"
