from __future__ import annotations

import copy
import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("OUT_DIR", str((BACKEND_DIR / "out").resolve()))

from app.main import (  # noqa: E402
    CandidateDiagnostics,
    _build_preference_query_trace_payload,
    _build_route_artifact_summaries,
    _write_route_run_bundle,
)
from app.models import GeoJSONLineString, LatLng, RouteMetrics, RouteOption, RouteRequest  # noqa: E402
from app.preference_model import build_preference_state  # noqa: E402
from app.preference_queries import PairwisePreferenceQuery  # noqa: E402
from app.preference_update import append_preference_query  # noqa: E402
from app.risk_model import build_risk_summary  # noqa: E402
from app.run_store import list_artifact_paths_for_run  # noqa: E402
from app.support_model import build_world_support_state  # noqa: E402
from app.uncertainty_model import build_world_bundle_summary  # noqa: E402


FIXED_RUN_ID = str(
    uuid.uuid5(
        uuid.NAMESPACE_URL,
        "whatif-freight-router/p67-certified-set-worked-example-v1",
    )
)


def _route_option(
    route_id: str,
    *,
    distance_km: float,
    duration_s: float,
    monetary_cost: float,
    emissions_kg: float,
) -> RouteOption:
    return RouteOption(
        id=route_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(0.0, 0.0), (1.0, 1.0)]),
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_s=duration_s,
            monetary_cost=monetary_cost,
            emissions_kg=emissions_kg,
            avg_speed_kmh=distance_km / (duration_s / 3600.0),
        ),
    )


def main() -> int:
    preference_state = append_preference_query(
        build_preference_state(
            route_ids=["route_0", "route_1"],
            weights={"time": 1.0, "money": 0.5},
            support_flag=True,
            support_reason=None,
        ),
        PairwisePreferenceQuery(preferred_route_id="route_0", challenger_route_id="route_1"),
        before_size=2,
        after_size=1,
        before_volume_proxy=1.0,
        after_volume_proxy=0.4,
        target_route_id="route_1",
        query_reason="reduce ambiguity",
    )
    support_state = build_world_support_state(
        support_score=0.88,
        support_ratio=0.7,
        support_bin="supported",
        calibration_bin="bin_2",
        support_source="fixture-export",
    )
    world_bundle_summary = build_world_bundle_summary(
        manifest={
            "world_count": 4,
            "unique_world_count": 3,
            "world_reuse_rate": 0.25,
            "proxy_world_count": 3,
            "audit_world_count": 1,
            "proxy_bias_model_version": "proxy-v5",
            "audit_propensity_version": "audit-v3",
            "proxy_correction_active": True,
            "multi_fidelity_certificate_basis": "corrected_from_residual_model",
            "audit_correction_mass": 2.5,
            "audit_propensity_scores": [0.3, 0.4],
        },
        support_state=support_state,
    )
    risk_summary = build_risk_summary(
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
        distance_km=120.0,
        support_state=support_state,
        probabilistic_world_bundle=world_bundle_summary.probabilistic_world_bundle,
        audit_world_bundle=world_bundle_summary.audit_world_bundle,
    )
    world_support_summary = {
        "schema_version": "world-support-summary-v1",
        "selected_route_id": "route_0",
        "selected_certificate_basis": "selected_certificate",
        "support_flag": True,
        "support_reason": None,
        "support_state": support_state.as_dict(),
        "world_bundle_summary": world_bundle_summary.as_dict(),
        "scenario_summary": None,
        "risk_summary": risk_summary.as_dict(),
    }
    preference_summary, support_summary = _build_route_artifact_summaries(
        preference_state=preference_state,
        world_support_summary=world_support_summary,
        pipeline_mode="dccs_refc",
        selected_certificate_basis="selected_certificate",
        support_flag=True,
        support_reason=None,
        support_state=support_state,
        world_bundle_summary=world_bundle_summary,
        scenario_summary=None,
        risk_summary=risk_summary,
        abstention=None,
        selected_certificate=None,
    )
    preference_query_trace = _build_preference_query_trace_payload(
        preference_state=preference_state,
        selected_route_id="route_0",
        selected_certificate_basis="selected_certificate",
        pipeline_mode="dccs_refc",
        support_flag=True,
        support_reason=None,
    )
    selected = _route_option(
        "route_0",
        distance_km=120.0,
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
    )
    challenger = _route_option(
        "route_1",
        distance_km=123.0,
        duration_s=3720.0,
        monetary_cost=52.0,
        emissions_kg=21.5,
    )
    certified_set_summary = {
        "member_route_ids": ["route_0", "route_1"],
        "excluded_route_ids": ["route_2"],
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
            "route_id": "route_0",
            "active_challenger_ids": ["route_1"],
            "support_flag": True,
            "singleton_not_justified_reasons": ["frontier_pairwise_gap_unresolved"],
            "singleton_justified": False,
            "outside_routes_excluded": True,
            "outside_routes_safely_excluded": True,
            "excluded_route_ids": ["route_2"],
            "excluded_route_safety_reasons": [],
            "frontier_member_ids": ["route_0", "route_1"],
        },
    }
    decision_package = {
        "schema_version": "1.0.0",
        "terminal_type": "certified_set",
        "selected_route_id": "route_0",
        "selected_certificate_basis": "selected_certificate",
        "recommended_route": None,
        "certified_set": [
            {"route_id": "route_0", "certificate": 0.86, "selected": True},
            {"route_id": "route_1", "certificate": 0.71, "selected": False},
        ],
        "certificate_summary": {
            "route_id": "route_0",
            "certificate": 0.86,
            "certified": True,
            "threshold": 0.8,
            "selected_route_id": "route_0",
            "winner_route_id": "route_0",
            "selected_certificate": 0.86,
            "selected_certificate_basis": "selected_certificate",
            "support_flag": True,
            "out_of_support_reason": None,
            "terminal_type": "certified_set",
        },
        "frontier_summary": {
            "frontier_route_ids": ["route_0", "route_1"],
            "frontier_count": 2,
            "selected_route_id": "route_0",
            "selected_certificate": 0.86,
        },
        "preference_summary": preference_summary,
        "support_summary": support_summary,
        "certified_set_summary": copy.deepcopy(certified_set_summary),
    }

    with patch("app.main.uuid.uuid4", return_value=uuid.UUID(FIXED_RUN_ID)):
        route_run = _write_route_run_bundle(
            req=RouteRequest(
                origin=LatLng(lat=51.5, lon=-2.6),
                destination=LatLng(lat=51.6, lon=-2.5),
                vehicle_type="rigid_hgv",
                scenario_mode="no_sharing",
                max_alternatives=2,
            ),
            selected=selected,
            candidates=[selected, challenger],
            warnings=[],
            candidate_diag=CandidateDiagnostics(selected_candidate_count=2),
            request_id="req-p67-certified-set-example",
            pipeline_mode="dccs_refc",
            run_seed=20260410,
            duration_ms=12.5,
            extra_json_artifacts={
                "decision_package.json": decision_package,
                "preference_state.json": preference_state.model_dump(mode="json"),
                "preference_query_trace.json": preference_query_trace,
                "world_support_summary.json": world_support_summary,
                "certified_set_summary.json": copy.deepcopy(certified_set_summary),
            },
        )

    if str(route_run["run_id"]) != FIXED_RUN_ID:
        raise RuntimeError(f"unexpected run id: {route_run['run_id']}")

    discovered = list_artifact_paths_for_run(FIXED_RUN_ID)
    decision = json.loads(discovered["decision_package.json"].read_text(encoding="utf-8"))
    certified_set = json.loads(discovered["certified_set_summary.json"].read_text(encoding="utf-8"))
    metadata = json.loads(discovered["metadata.json"].read_text(encoding="utf-8"))
    index = json.loads(discovered["index.json"].read_text(encoding="utf-8"))

    witness = certified_set["witness"]
    if decision["terminal_type"] != "certified_set":
        raise RuntimeError("decision_package terminal_type mismatch")
    if decision["recommended_route"] is not None:
        raise RuntimeError("decision_package unexpectedly recommends a singleton route")
    if certified_set["certified"] is not True:
        raise RuntimeError("certified_set_summary is not certified")
    if certified_set["set_size"] < 2:
        raise RuntimeError("certified_set_summary is not a set-valued outcome")
    if not witness.get("singleton_not_justified_reasons"):
        raise RuntimeError("singleton withholding reasons are missing")
    if witness.get("outside_routes_excluded") is not True:
        raise RuntimeError("outside routes are not marked excluded")
    if witness.get("outside_routes_safely_excluded") is not True:
        raise RuntimeError("outside routes are not marked safely excluded")
    if witness.get("singleton_justified") is not False:
        raise RuntimeError("singleton withholding flag is inconsistent")
    if decision["certified_set_summary"]["certified"] is not True:
        raise RuntimeError("decision package certified_set_summary is inconsistent")
    if metadata["selected_route_id"] != "route_0":
        raise RuntimeError("metadata selected_route_id mismatch")
    if "certified_set_summary.json" not in metadata["artifact_names"]:
        raise RuntimeError("metadata is missing certified_set_summary.json")
    if "decision_package.json" not in index["artifact_names"]:
        raise RuntimeError("index is missing decision_package.json")

    print(FIXED_RUN_ID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
