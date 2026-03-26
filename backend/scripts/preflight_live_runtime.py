from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from app.calibration_loader import (
    load_departure_profile,
    load_fuel_price_snapshot,
    load_live_scenario_context,
    load_scenario_profiles,
    load_stochastic_regimes,
    load_toll_segments_seed,
    load_toll_tariffs,
    load_uk_bank_holidays,
    refresh_live_runtime_route_caches,
)
from app.carbon_model import apply_scope_emissions_adjustment, resolve_carbon_price
from app.model_data_errors import ModelDataError
from app.routing_ors import local_ors_runtime_manifest
from app.settings import settings

STRICT_PROVENANCE_DENY_TOKENS: tuple[str, ...] = (
    "synthetic",
    "proxy",
    "fallback",
    "legacy",
    "bootstrap",
    "fixture",
)

REPO_LOCAL_PROVENANCE_DENY_TOKENS: tuple[str, ...] = (
    "synthetic",
    "legacy",
    "fallback",
)

REPO_LOCAL_RUNTIME_ASSET_PATHS: dict[str, str] = {
    "scenario": "backend/assets/uk/scenario_profiles_uk.json",
    "fuel": "backend/assets/uk/fuel_prices_uk.json",
    "carbon": "backend/assets/uk/carbon_price_schedule_uk.json",
    "departure": "backend/assets/uk/departure_profiles_uk.json",
    "stochastic": "backend/assets/uk/stochastic_regimes_uk.json",
    "toll_topology": "backend/assets/uk/toll_topology_uk.json",
    "toll_tariffs": "backend/assets/uk/toll_tariffs_uk.json",
}

BASELINE_SMOKE_COORDS: tuple[tuple[float, float], tuple[float, float]] = (
    (-1.8904, 52.4862),
    (-0.1276, 51.5072),
)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ModelDataError):
        return {
            "type": type(exc).__name__,
            "reason_code": exc.reason_code,
            "message": exc.message,
            "details": exc.details or {},
        }
    return {
        "type": type(exc).__name__,
        "reason_code": "unexpected_error",
        "message": str(exc),
        "details": {},
    }


def _departure_profile_details() -> dict[str, Any]:
    profile = load_departure_profile()
    contextual = getattr(profile, "contextual", None)
    if isinstance(contextual, dict):
        region_count = len(contextual)
    else:
        # Backward-compatible shape used by some tests and legacy fixtures.
        profiles = getattr(profile, "profiles", {})
        region_count = len(profiles) if isinstance(profiles, dict) else 0
    return {
        "source": str(getattr(profile, "source", "")),
        "calibration_basis": str(getattr(profile, "calibration_basis", "")),
        "region_count": int(region_count),
    }


def _source_policy_name() -> str:
    policy = str(getattr(settings, "live_source_policy", "repo_local_fresh") or "repo_local_fresh").strip().lower()
    if policy not in {"strict_external", "repo_local_fresh"}:
        return "repo_local_fresh"
    return policy


def _active_provenance_deny_tokens() -> tuple[str, ...]:
    return STRICT_PROVENANCE_DENY_TOKENS if _source_policy_name() == "strict_external" else REPO_LOCAL_PROVENANCE_DENY_TOKENS


def _contains_provenance_deny_token(*values: Any) -> bool:
    combined = " ".join(str(value) for value in values if value not in (None, "")).lower()
    return any(token in combined for token in _active_provenance_deny_tokens())


def _validate_check_provenance(name: str, details: dict[str, Any]) -> None:
    serialized = json.dumps(details, sort_keys=True, default=str)
    if _contains_provenance_deny_token(serialized):
        raise ModelDataError(
            reason_code="strict_source_provenance_invalid",
            message=f"{name} exposes synthetic/proxy/bootstrap/fallback provenance markers.",
            details={"check": name},
        )


def _scenario_live_context_details() -> dict[str, Any]:
    scenario_profiles = load_scenario_profiles()
    transform_params_json: str | None = None
    if isinstance(getattr(scenario_profiles, "transform_params", None), dict):
        transform_params_json = json.dumps(
            scenario_profiles.transform_params,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    context = load_live_scenario_context(
        corridor_bucket="gcqrs",
        road_mix_bucket="mixed",
        vehicle_class="rigid_hgv",
        day_kind="weekday",
        hour_slot_local=12,
        weather_bucket="clear",
        centroid_lat=52.4862,
        centroid_lon=-1.8904,
        road_hint="M6",
        transform_params_json=transform_params_json,
    )
    return {
        "as_of_utc": str(context.as_of_utc),
        "coverage": dict(context.coverage),
        "source_set": dict(context.source_set),
    }


def _effective_sources_from_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for check in checks:
        name = str(check.get("name", "")).strip()
        details = check.get("details")
        if not name or not isinstance(details, dict):
            continue
        source = details.get("source")
        if source not in (None, ""):
            sources[name] = source
            continue
        source_set = details.get("source_set")
        if isinstance(source_set, dict) and source_set:
            sources[name] = dict(source_set)
    return sources


def _configured_remote_urls() -> dict[str, str]:
    return {
        "scenario": str(settings.live_scenario_coefficient_url or "").strip(),
        "fuel": str(settings.live_fuel_price_url or "").strip(),
        "carbon": str(settings.live_carbon_schedule_url or "").strip(),
        "departure": str(settings.live_departure_profile_url or "").strip(),
        "stochastic": str(settings.live_stochastic_regimes_url or "").strip(),
        "toll_topology": str(settings.live_toll_topology_url or "").strip(),
        "toll_tariffs": str(settings.live_toll_tariffs_url or "").strip(),
        "bank_holidays": str(settings.live_bank_holidays_url or "").strip(),
    }


def _configured_source_bindings() -> dict[str, str]:
    if _source_policy_name() == "strict_external":
        return _configured_remote_urls()
    bindings = dict(REPO_LOCAL_RUNTIME_ASSET_PATHS)
    # Bank holidays remain a public, no-key source in the current repo-local flow.
    bindings["bank_holidays"] = str(settings.live_bank_holidays_url or "").strip()
    return bindings


def _osrm_engine_smoke_details() -> dict[str, Any]:
    origin, destination = BASELINE_SMOKE_COORDS
    url = (
        f"{str(settings.osrm_base_url).rstrip('/')}/route/v1/{str(settings.osrm_profile).strip() or 'driving'}"
        f"/{origin[0]},{origin[1]};{destination[0]},{destination[1]}?overview=false"
    )
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0, connect=4.0), trust_env=False) as client:
            response = client.get(url)
    except httpx.RequestError as exc:
        raise ModelDataError(
            reason_code="osrm_engine_unreachable",
            message=f"OSRM engine smoke request failed: {exc}",
            details={"base_url": str(settings.osrm_base_url)},
        ) from exc
    if response.status_code >= 400:
        raise ModelDataError(
            reason_code="osrm_engine_unreachable",
            message=f"OSRM engine smoke request returned HTTP {response.status_code}.",
            details={"base_url": str(settings.osrm_base_url), "status_code": int(response.status_code)},
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ModelDataError(
            reason_code="osrm_engine_invalid_response",
            message="OSRM engine smoke request did not return JSON.",
            details={"base_url": str(settings.osrm_base_url)},
        ) from exc
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ModelDataError(
            reason_code="osrm_engine_invalid_response",
            message="OSRM engine smoke request returned no routes.",
            details={"base_url": str(settings.osrm_base_url)},
        )
    route = routes[0] if isinstance(routes[0], dict) else {}
    distance_m = float(route.get("distance") or 0.0)
    duration_s = float(route.get("duration") or 0.0)
    if distance_m <= 0.0 or duration_s <= 0.0:
        raise ModelDataError(
            reason_code="osrm_engine_invalid_response",
            message="OSRM engine smoke route is missing positive distance/duration.",
            details={"base_url": str(settings.osrm_base_url)},
        )
    return {
        "base_url": str(settings.osrm_base_url),
        "profile": str(settings.osrm_profile),
        "status_code": int(response.status_code),
        "distance_m": round(distance_m, 3),
        "duration_s": round(duration_s, 3),
    }


def _ors_engine_smoke_details() -> dict[str, Any]:
    profile = str(settings.ors_directions_profile_hgv or "driving-hgv").strip() or "driving-hgv"
    manifest = local_ors_runtime_manifest(
        base_url=str(settings.ors_base_url),
        profile=profile,
        vehicle_type="hgv",
    )
    identity_status = str(manifest.get("identity_status") or "").strip()
    if identity_status != "graph_identity_verified":
        raise ModelDataError(
            reason_code="ors_graph_identity_unverified",
            message=f"ORS graph provenance is not verified: {identity_status or 'unknown'}.",
            details={
                "base_url": str(settings.ors_base_url),
                "identity_status": identity_status or "unknown",
                "manifest_hash": str(manifest.get("manifest_hash") or ""),
            },
        )
    origin, destination = BASELINE_SMOKE_COORDS
    url = f"{str(settings.ors_base_url).rstrip('/')}/v2/directions/{profile}/geojson"
    body = {
        "coordinates": [[origin[0], origin[1]], [destination[0], destination[1]]],
        "instructions": False,
        "elevation": False,
        "options": {"vehicle_type": "hgv"},
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(25.0, connect=5.0), trust_env=False) as client:
            response = client.post(url, json=body)
    except httpx.RequestError as exc:
        raise ModelDataError(
            reason_code="ors_engine_unreachable",
            message=f"ORS engine smoke request failed: {exc}",
            details={"base_url": str(settings.ors_base_url), "profile": profile},
        ) from exc
    if response.status_code >= 400:
        raise ModelDataError(
            reason_code="ors_engine_unreachable",
            message=f"ORS engine smoke request returned HTTP {response.status_code}.",
            details={
                "base_url": str(settings.ors_base_url),
                "profile": profile,
                "status_code": int(response.status_code),
            },
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ModelDataError(
            reason_code="ors_engine_invalid_response",
            message="ORS engine smoke request did not return JSON.",
            details={"base_url": str(settings.ors_base_url), "profile": profile},
        ) from exc
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ModelDataError(
            reason_code="ors_engine_invalid_response",
            message="ORS engine smoke request returned no features.",
            details={"base_url": str(settings.ors_base_url), "profile": profile},
        )
    feature = features[0] if isinstance(features[0], dict) else {}
    summary = ((feature.get("properties") or {}) if isinstance(feature.get("properties"), dict) else {}).get("summary", {})
    distance_m = float((summary or {}).get("distance") or 0.0)
    duration_s = float((summary or {}).get("duration") or 0.0)
    if distance_m <= 0.0 or duration_s <= 0.0:
        raise ModelDataError(
            reason_code="ors_engine_invalid_response",
            message="ORS engine smoke route is missing positive distance/duration.",
            details={"base_url": str(settings.ors_base_url), "profile": profile},
        )
    engine_meta = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    return {
        "base_url": str(settings.ors_base_url),
        "profile": profile,
        "status_code": int(response.status_code),
        "distance_m": round(distance_m, 3),
        "duration_s": round(duration_s, 3),
        "engine_version": ((engine_meta.get("engine") or {}) if isinstance(engine_meta.get("engine"), dict) else {}).get("version"),
        "graph_date": ((engine_meta.get("engine") or {}) if isinstance(engine_meta.get("engine"), dict) else {}).get("graph_date"),
        "manifest_hash": str(manifest.get("manifest_hash") or ""),
        "identity_status": identity_status,
    }


def _run_required_checks() -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    checks: list[tuple[str, Any]] = [
        (
            "scenario_profiles",
            lambda: {
                "version": load_scenario_profiles().version,
                "source": load_scenario_profiles().source,
                "calibration_basis": getattr(load_scenario_profiles(), "calibration_basis", None),
                "mode_observation_source": getattr(load_scenario_profiles(), "mode_observation_source", None),
                "contexts": len(load_scenario_profiles().contexts),
            },
        ),
        (
            "scenario_live_context",
            _scenario_live_context_details,
        ),
        (
            "fuel_snapshot",
            lambda: {
                "source": load_fuel_price_snapshot().source,
                "as_of": load_fuel_price_snapshot().as_of,
                "signature_prefix": (load_fuel_price_snapshot().signature or "")[:12],
            },
        ),
        (
            "toll_tariffs",
            lambda: {
                "source": load_toll_tariffs().source,
                "rule_count": len(load_toll_tariffs().rules),
                "sample_rule_ids": [
                    str(getattr(rule, "rule_id", rule.get("id", "")) if isinstance(rule, dict) else getattr(rule, "rule_id", ""))
                    for rule in list(load_toll_tariffs().rules)[:3]
                ],
                "sample_operators": [
                    str(getattr(rule, "operator", rule.get("operator", "")) if isinstance(rule, dict) else getattr(rule, "operator", ""))
                    for rule in list(load_toll_tariffs().rules)[:3]
                ],
            },
        ),
        (
            "toll_topology",
            lambda: {
                "segment_count": len(load_toll_segments_seed()),
            },
        ),
        (
            "stochastic_regimes",
            lambda: {
                "source": load_stochastic_regimes().source,
                "calibration_basis": getattr(load_stochastic_regimes(), "calibration_basis", None),
                "regime_count": len(load_stochastic_regimes().regimes),
            },
        ),
        (
            "departure_profiles",
            _departure_profile_details,
        ),
        (
            "bank_holidays",
            lambda: {
                "count": len(load_uk_bank_holidays()),
            },
        ),
        (
            "carbon_policy",
            lambda: {
                "price_per_kg": resolve_carbon_price(
                    request_price_per_kg=0.0,
                    departure_time_utc=now,
                ).price_per_kg,
                "scope_adjusted_emissions": apply_scope_emissions_adjustment(
                    emissions_kg=1.0,
                    is_ev_mode=False,
                    scope_mode="wtw",
                    departure_time_utc=now,
                ),
            },
        ),
        ("osrm_engine_smoke", _osrm_engine_smoke_details),
        ("ors_engine_smoke", _ors_engine_smoke_details),
    ]

    results: list[dict[str, Any]] = []
    for check_name, action in checks:
        started = datetime.now(UTC)
        try:
            details = action()
            if isinstance(details, dict):
                _validate_check_provenance(check_name, details)
            results.append(
                {
                    "name": check_name,
                    "required": True,
                    "ok": True,
                    "duration_ms": int((datetime.now(UTC) - started).total_seconds() * 1000.0),
                    "details": details,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            results.append(
                {
                    "name": check_name,
                    "required": True,
                    "ok": False,
                    "duration_ms": int((datetime.now(UTC) - started).total_seconds() * 1000.0),
                    "error": _error_payload(exc),
                }
            )
    return results


def run_preflight(*, output_path: Path) -> dict[str, Any]:
    refresh_live_runtime_route_caches()
    checks = _run_required_checks()
    required_failures = [check for check in checks if check.get("required") and not check.get("ok")]
    source_policy = _source_policy_name()
    configured_urls = _configured_source_bindings()
    summary = {
        "checked_at_utc": _now_iso(),
        "strict_live_data_required": bool(settings.strict_live_data_required),
        "live_runtime_data_enabled": bool(settings.live_runtime_data_enabled),
        "source_policy": source_policy,
        "required_ok": len(required_failures) == 0,
        "required_failure_count": len(required_failures),
        "checks": checks,
        "configured_urls": configured_urls,
        "urls": configured_urls,
        "effective_sources": _effective_sources_from_checks(checks),
    }
    if source_policy != "strict_external":
        summary["configured_remote_urls"] = _configured_remote_urls()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict live runtime preflight checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "out" / "model_assets" / "preflight_live_runtime.json",
        help="Path to write preflight summary JSON.",
    )
    args = parser.parse_args()

    summary = run_preflight(output_path=args.output)
    if summary["required_ok"]:
        print("Strict live runtime preflight: PASS")
        print(f"Summary: {args.output}")
        return
    print("Strict live runtime preflight: FAIL")
    print(f"Summary: {args.output}")
    failed = [check for check in summary["checks"] if not check.get("ok")]
    for item in failed:
        error = item.get("error", {})
        print(f"- {item.get('name')}: {error.get('reason_code')} | {error.get('message')}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
