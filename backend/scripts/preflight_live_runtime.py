from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from app.calibration_loader import (
    load_departure_profile,
    load_fuel_price_snapshot,
    load_scenario_profiles,
    load_stochastic_regimes,
    load_toll_segments_seed,
    load_toll_tariffs,
    load_uk_bank_holidays,
    refresh_live_runtime_route_caches,
)
from app.carbon_model import apply_scope_emissions_adjustment, resolve_carbon_price
from app.model_data_errors import ModelDataError
from app.settings import settings


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
        "region_count": int(region_count),
    }


def _run_required_checks() -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    checks: list[tuple[str, Any]] = [
        (
            "scenario_profiles",
            lambda: {
                "version": load_scenario_profiles().version,
                "source": load_scenario_profiles().source,
                "contexts": len(load_scenario_profiles().contexts),
            },
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
    ]

    results: list[dict[str, Any]] = []
    for check_name, action in checks:
        started = datetime.now(UTC)
        try:
            details = action()
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
    summary = {
        "checked_at_utc": _now_iso(),
        "strict_live_data_required": bool(settings.strict_live_data_required),
        "live_runtime_data_enabled": bool(settings.live_runtime_data_enabled),
        "required_ok": len(required_failures) == 0,
        "required_failure_count": len(required_failures),
        "checks": checks,
        "urls": {
            "scenario": str(settings.live_scenario_coefficient_url or "").strip(),
            "fuel": str(settings.live_fuel_price_url or "").strip(),
            "carbon": str(settings.live_carbon_schedule_url or "").strip(),
            "departure": str(settings.live_departure_profile_url or "").strip(),
            "stochastic": str(settings.live_stochastic_regimes_url or "").strip(),
            "toll_topology": str(settings.live_toll_topology_url or "").strip(),
            "toll_tariffs": str(settings.live_toll_tariffs_url or "").strip(),
            "bank_holidays": str(settings.live_bank_holidays_url or "").strip(),
        },
    }
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
