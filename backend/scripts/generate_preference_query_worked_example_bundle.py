from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evidence_certification import (  # noqa: E402
    compute_certificate,
    compute_fragility_maps,
    project_refc_scaffold_states,
)
from app.main import (  # noqa: E402
    CandidateDiagnostics,
    _assemble_decision_package,
    _write_route_run_bundle,
)
from app.models import (  # noqa: E402
    GeoJSONLineString,
    LatLng,
    RouteCertificationSummary,
    RouteMetrics,
    RouteOption,
    RouteRequest,
    VoiStopSummary,
)
from app.preference_model import build_preference_state  # noqa: E402
from app.preference_queries import PairwisePreferenceQuery  # noqa: E402
from app.preference_update import append_preference_query  # noqa: E402
from app.run_store import list_artifact_paths_for_run  # noqa: E402
from app.settings import settings  # noqa: E402


RUN_ID = "6f9c0b65-1f4d-4f2c-9d85-3c91f0cf2d84"
REQUEST_ID = "req-preference-query-worked-example"
RUN_SEED = 20260412


def _route(
    route_id: str,
    *,
    distance_km: float,
    duration_s: float,
    monetary_cost: float,
    emissions_kg: float,
) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[(0.0, 0.0), (1.0, 1.0)],
        ),
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            avg_speed_kmh=(distance_km / max(duration_s / 3600.0, 1e-6)),
        ),
    )


def _artifact_dir() -> Path:
    return Path(settings.out_dir) / "artifacts" / RUN_ID


def _manifest_path() -> Path:
    return Path(settings.out_dir) / "manifests" / f"{RUN_ID}.json"


def _provenance_path() -> Path:
    return Path(settings.out_dir) / "provenance" / f"{RUN_ID}.json"


def _build_bundle_payloads() -> dict[str, object]:
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
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selected_route_id=certificate.selected_route_id,
    )
    projection = project_refc_scaffold_states(
        certificate,
        fragility,
        frontier_route_ids=[certificate.selected_route_id, "route_b"],
    )

    selected = _route(
        "route_a",
        distance_km=10.0,
        duration_s=10.0,
        monetary_cost=12.0,
        emissions_kg=4.0,
    )
    challenger = _route(
        "route_b",
        distance_km=10.8,
        duration_s=11.0,
        monetary_cost=11.0,
        emissions_kg=5.0,
    )

    selected_certificate = RouteCertificationSummary(
        route_id=certificate.selected_route_id,
        certificate=float(certificate.certificate[certificate.selected_route_id]),
        certified=bool(certificate.certified),
        threshold=float(certificate.threshold),
        active_families=list(certificate.world_manifest.get("active_families", [])),
        top_fragility_families=list(
            fragility.route_fragility_map.get(certificate.selected_route_id, {}).keys()
        )[:3],
        top_competitor_route_id="route_b",
        top_value_of_refresh_family="scenario",
        ambiguity_context={"support_strength": True},
    )

    preference_state = append_preference_query(
        build_preference_state(
            route_ids=["route_a", "route_b"],
            weights={"time": 1.0, "money": 0.5, "co2": 0.25},
        ),
        PairwisePreferenceQuery(
            preferred_route_id="route_a",
            challenger_route_id="route_b",
        ),
        before_size=2,
        after_size=1,
        before_volume_proxy=1.0,
        after_volume_proxy=0.4,
        target_route_id="route_b",
        query_reason="reduce ambiguity",
    )
    preference_state.terminal_type = "certified"

    decision_package = _assemble_decision_package(
        selected=selected,
        candidates=[selected, challenger],
        run_id=RUN_ID,
        pipeline_mode="dccs_refc",
        manifest_endpoint=f"/runs/{RUN_ID}/manifest",
        artifacts_endpoint=f"/runs/{RUN_ID}/artifacts",
        provenance_endpoint=f"/runs/{RUN_ID}/provenance",
        selected_certificate=selected_certificate,
        voi_stop_summary=VoiStopSummary(
            final_route_id=certificate.selected_route_id,
            certificate=float(certificate.certificate[certificate.selected_route_id]),
            certified=bool(certificate.certified),
            iteration_count=1,
            search_budget_used=0,
            evidence_budget_used=0,
            stop_reason="certified",
            search_completeness_score=1.0,
            search_completeness_gap=0.0,
            credible_search_uncertainty=False,
        ),
        preference_state=preference_state,
        preference_query_trace={},
        world_support_summary={
            "schema_version": "world-support-summary-v1",
            "selected_route_id": certificate.selected_route_id,
            "selected_certificate_basis": "selected_certificate",
            "support_flag": True,
            "support_state": {
                "support_flag": True,
                "support_bin": "supported",
            },
            "world_bundle_summary": {
                "multi_fidelity_summary": {
                    "proxy_world_count": 2,
                    "audit_world_count": 0,
                    "proxy_bias_model_version": "proxy-v5",
                    "audit_propensity_version": "audit-v3",
                    "proxy_correction_active": True,
                    "multi_fidelity_certificate_basis": "corrected_from_residual_model",
                    "proxy_only_fraction": 1.0,
                    "audit_correction_mass": 0.0,
                    "positivity_diagnostics": {
                        "positivity_ok": True,
                        "weak_overlap_detected": False,
                    },
                },
            },
        },
        world_manifest=certificate.world_manifest,
        winner_confidence_state=projection["winner_confidence_state"],
        pairwise_gap_states=projection["pairwise_gap_states"],
        flip_radius_state=projection["flip_radius_state"],
        decision_region_state=projection["decision_region_state"],
        certificate_witness=projection["certificate_witness"],
        certified_set=[selected],
        abstention=None,
    )
    encoded = json.loads(decision_package.model_dump_json())
    selected_route_id = str(encoded["selected"]["id"])

    now = datetime.now(UTC).isoformat()
    final_route_trace = {
        "pipeline_mode": "dccs_refc",
        "refinement_policy": "dccs",
        "run_seed": RUN_SEED,
        "stage_events": [
            {"stage": "refc", "event": "enter", "timestamp_utc": now},
            {"stage": "preference", "event": "query_appended", "timestamp_utc": now},
            {"stage": "terminal_decision", "event": "exit", "timestamp_utc": now},
        ],
        "selected_route_id": selected_route_id,
        "selected_certificate_basis": encoded["selected_certificate_basis"],
        "terminal_type": encoded["terminal_type"],
        "query_count": encoded["preference_summary"]["query_count"],
        "preference_irrelevance_proven": encoded["preference_summary"]["preference_irrelevance_proven"],
        "stop_reason": "certified",
        "artifact_pointers": {
            "preference_state": "preference_state.json",
            "preference_query_trace": "preference_query_trace.json",
            "world_support_summary": "world_support_summary.json",
            "decision_package": "decision_package.json",
            "winner_confidence_state": "winner_confidence_state.json",
            "pairwise_gap_state": "pairwise_gap_state.json",
            "flip_radius_summary": "flip_radius_summary.json",
            "decision_region_summary": "decision_region_summary.json",
            "certificate_witness": "certificate_witness.json",
            "certified_set_summary": "certified_set_summary.json",
            "voi_stop_certificate": "voi_stop_certificate.json",
        },
    }
    voi_stop_certificate = {
        "final_winner_route_id": selected_route_id,
        "certificate_value": encoded["selected_certificate"]["certificate"],
        "certified": True,
        "search_budget_used": 0,
        "search_budget_remaining": 0,
        "evidence_budget_used": 0,
        "evidence_budget_remaining": 0,
        "search_completeness_score": 1.0,
        "search_completeness_gap": 0.0,
        "credible_search_uncertainty": False,
        "credible_evidence_uncertainty": False,
        "stop_reason": "certified",
        "action_trace": [],
        "best_rejected_action": None,
        "ambiguity_summary": {
            "top_fragility_families": encoded["selected_certificate"]["top_fragility_families"],
            "top_refresh_family": encoded["selected_certificate"]["top_value_of_refresh_family"],
            "top_competitor_route_id": encoded["selected_certificate"]["top_competitor_route_id"],
        },
    }

    return {
        "selected": selected,
        "challenger": challenger,
        "selected_certificate": selected_certificate,
        "decision_package": encoded,
        "preference_state": encoded["preference_state"],
        "preference_query_trace": encoded["preference_query_trace"],
        "world_support_summary": encoded["world_support_summary"],
        "winner_confidence_state": encoded["winner_confidence_state"],
        "pairwise_gap_state": {
            "pairwise_gap_states": encoded["pairwise_gap_states"],
            "selected_route_id": selected_route_id,
            "selected_certificate": encoded["selected_certificate"]["certificate"],
            "support_flag": True,
        },
        "flip_radius_summary": encoded["flip_radius_state"],
        "decision_region_summary": encoded["decision_region_state"],
        "certificate_witness": encoded["certificate_witness"],
        "certified_set_summary": encoded["certified_set_summary"],
        "voi_stop_certificate": voi_stop_certificate,
        "final_route_trace": final_route_trace,
    }


def build_bundle() -> Path:
    artifact_dir = _artifact_dir()
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    _manifest_path().unlink(missing_ok=True)
    _provenance_path().unlink(missing_ok=True)

    payloads = _build_bundle_payloads()
    with patch("app.main.uuid.uuid4", return_value=uuid.UUID(RUN_ID)):
        route_run = _write_route_run_bundle(
            req=RouteRequest(
                origin=LatLng(lat=51.5, lon=-2.6),
                destination=LatLng(lat=51.6, lon=-2.5),
                vehicle_type="rigid_hgv",
                scenario_mode="no_sharing",
                max_alternatives=2,
            ),
            selected=payloads["selected"],
            candidates=[payloads["selected"], payloads["challenger"]],
            warnings=[],
            candidate_diag=CandidateDiagnostics(selected_candidate_count=2),
            request_id=REQUEST_ID,
            pipeline_mode="dccs_refc",
            run_seed=RUN_SEED,
            duration_ms=8.5,
            selected_certificate=payloads["selected_certificate"],
            voi_stop_summary=VoiStopSummary(
                final_route_id="route_a",
                certificate=float(payloads["decision_package"]["selected_certificate"]["certificate"]),
                certified=True,
                iteration_count=1,
                search_budget_used=0,
                evidence_budget_used=0,
                stop_reason="certified",
                search_completeness_score=1.0,
                search_completeness_gap=0.0,
                credible_search_uncertainty=False,
            ),
            extra_json_artifacts={
                "decision_package.json": payloads["decision_package"],
                "preference_state.json": payloads["preference_state"],
                "preference_query_trace.json": payloads["preference_query_trace"],
                "world_support_summary.json": payloads["world_support_summary"],
                "winner_confidence_state.json": payloads["winner_confidence_state"],
                "pairwise_gap_state.json": payloads["pairwise_gap_state"],
                "flip_radius_summary.json": payloads["flip_radius_summary"],
                "decision_region_summary.json": payloads["decision_region_summary"],
                "certificate_witness.json": payloads["certificate_witness"],
                "certified_set_summary.json": payloads["certified_set_summary"],
                "voi_stop_certificate.json": payloads["voi_stop_certificate"],
                "final_route_trace.json": payloads["final_route_trace"],
            },
        )
    if route_run["run_id"] != RUN_ID:
        raise RuntimeError(f"expected run id {RUN_ID}, got {route_run['run_id']}")
    return artifact_dir


def validate_bundle() -> dict[str, object]:
    discovered = list_artifact_paths_for_run(RUN_ID)
    required = {
        "decision_package.json",
        "preference_state.json",
        "preference_query_trace.json",
        "voi_stop_certificate.json",
        "final_route_trace.json",
        "index.json",
    }
    missing = sorted(required.difference(discovered))
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")

    decision_package = json.loads(discovered["decision_package.json"].read_text(encoding="utf-8"))
    preference_state = json.loads(discovered["preference_state.json"].read_text(encoding="utf-8"))
    preference_query_trace = json.loads(
        discovered["preference_query_trace.json"].read_text(encoding="utf-8")
    )
    voi_stop_certificate = json.loads(discovered["voi_stop_certificate.json"].read_text(encoding="utf-8"))
    final_route_trace = json.loads(discovered["final_route_trace.json"].read_text(encoding="utf-8"))

    query_count = int(preference_query_trace.get("query_count", 0) or 0)
    if query_count <= 0:
        raise RuntimeError("preference_query_trace.query_count must be > 0")
    if int(preference_state.get("query_count", 0) or 0) != query_count:
        raise RuntimeError("preference_state.query_count does not match preference_query_trace.query_count")
    if int(decision_package["preference_summary"]["query_count"]) != query_count:
        raise RuntimeError("decision_package.preference_summary.query_count does not match preference_query_trace")
    selected_route_id = str(decision_package["selected"]["id"])
    if selected_route_id != preference_query_trace["selected_route_id"]:
        raise RuntimeError("selected route mismatch between decision_package and preference_query_trace")
    if voi_stop_certificate["final_winner_route_id"] != selected_route_id:
        raise RuntimeError("voi_stop_certificate winner does not match decision package")
    if final_route_trace["selected_route_id"] != selected_route_id:
        raise RuntimeError("final_route_trace selected route does not match decision package")
    if final_route_trace["terminal_type"] != decision_package["terminal_type"]:
        raise RuntimeError("final_route_trace terminal_type does not match decision package")

    return {
        "run_id": RUN_ID,
        "artifact_dir": str(_artifact_dir()),
        "query_count": query_count,
        "terminal_type": decision_package["terminal_type"],
        "selected_route_id": selected_route_id,
    }


def main() -> None:
    artifact_dir = build_bundle()
    summary = validate_bundle()
    print(json.dumps({"artifact_dir": str(artifact_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
