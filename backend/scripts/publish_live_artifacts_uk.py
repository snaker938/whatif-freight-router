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

from fetch_fuel_history_uk import build as build_fuel_history


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} payload must be a JSON object: {path}")
    return payload


def _extract_as_of(payload: dict[str, Any]) -> str | None:
    for key in ("as_of_utc", "as_of", "generated_at_utc", "updated_at_utc"):
        value = str(payload.get(key, "")).strip()
        if value:
            return value
    return None


def _ensure_top_level_as_of(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    out = dict(payload)
    as_of = _extract_as_of(out)
    if not as_of:
        raise RuntimeError(f"{label} payload must include as_of_utc/as_of/generated_at_utc/updated_at_utc")
    out["as_of_utc"] = as_of
    return out


def _fuel_signature_material(payload: dict[str, Any]) -> str:
    canonical = {
        "as_of_utc": payload.get("as_of_utc", payload.get("as_of")),
        "source": payload.get("source"),
        "prices_gbp_per_l": payload.get("prices_gbp_per_l"),
        "grid_price_gbp_per_kwh": payload.get("grid_price_gbp_per_kwh"),
        "history": payload.get("history"),
        "regional_multipliers": payload.get("regional_multipliers"),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _fuel_signature(payload: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(_fuel_signature_material(payload).encode("utf-8")).hexdigest()


def _ensure_fuel_signature(
    *,
    fuel_asset_path: Path,
    fuel_raw_path: Path,
) -> tuple[dict[str, Any], bool]:
    fuel_payload = _ensure_top_level_as_of(
        _read_json_object(fuel_asset_path, label="fuel asset"),
        label="fuel asset",
    )
    signature = str(fuel_payload.get("signature", "")).strip()
    expected = _fuel_signature(fuel_payload)
    if signature and signature.lower() == expected.lower():
        return fuel_payload, False
    if not fuel_raw_path.exists():
        raise RuntimeError(
            "Fuel asset signature is missing/invalid and raw source is unavailable for regeneration: "
            f"{fuel_raw_path}"
        )
    build_fuel_history(
        source_json=fuel_raw_path,
        output_json=fuel_asset_path,
        min_history_days=1,
    )
    refreshed = _ensure_top_level_as_of(
        _read_json_object(fuel_asset_path, label="fuel asset"),
        label="fuel asset",
    )
    refreshed_signature = str(refreshed.get("signature", "")).strip()
    refreshed_expected = _fuel_signature(refreshed)
    if not refreshed_signature or refreshed_signature.lower() != refreshed_expected.lower():
        raise RuntimeError("Fuel signature remains invalid after regeneration.")
    return refreshed, True


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN guard
        return None
    return number


def _build_toll_topology_payload(
    *,
    compiled_payload: dict[str, Any],
    as_of_utc: str,
) -> dict[str, Any]:
    raw_segments = compiled_payload.get("segments")
    if not isinstance(raw_segments, list):
        raise RuntimeError("toll_segments_seed_compiled.json must contain a top-level 'segments' array.")
    features: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_segments):
        if not isinstance(row, dict):
            continue
        line_coords: list[list[float]] = []
        raw_coords = row.get("coordinates")
        if isinstance(raw_coords, list):
            for point in raw_coords:
                if not isinstance(point, list | tuple) or len(point) < 2:
                    continue
                lat = _safe_float(point[0])
                lon = _safe_float(point[1])
                if lat is None or lon is None:
                    continue
                line_coords.append([lon, lat])
        if len(line_coords) < 2:
            continue
        segment_id = str(row.get("id", f"segment_{idx}")).strip() or f"segment_{idx}"
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": segment_id,
                    "name": str(row.get("name", "")).strip(),
                    "operator": str(row.get("operator", "")).strip().lower() or "default",
                    "road_class": str(row.get("road_class", "")).strip().lower() or "default",
                    "crossing_id": str(row.get("crossing_id", segment_id)).strip().lower() or segment_id,
                    "direction": str(row.get("direction", "both")).strip().lower() or "both",
                    "crossing_fee_gbp": max(0.0, _safe_float(row.get("crossing_fee_gbp")) or 0.0),
                    "distance_fee_gbp_per_km": max(
                        0.0,
                        _safe_float(row.get("distance_fee_gbp_per_km")) or 0.0,
                    ),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": line_coords,
                },
            }
        )
    if not features:
        raise RuntimeError("No valid topology features could be derived from toll_segments_seed_compiled.json")
    return {
        "type": "FeatureCollection",
        "source": str(compiled_payload.get("source", "live_runtime:toll_topology")).strip()
        or "live_runtime:toll_topology",
        "version": str(compiled_payload.get("version", "uk-v2")).strip() or "uk-v2",
        "generated_at_utc": _utc_now_iso(),
        "as_of_utc": as_of_utc,
        "features": features,
    }


def _build_toll_tariffs_payload(
    *,
    compiled_payload: dict[str, Any],
    as_of_utc: str,
) -> dict[str, Any]:
    rules = compiled_payload.get("rules")
    if not isinstance(rules, list) or not rules:
        raise RuntimeError("toll_tariffs_uk_compiled.json must contain a non-empty top-level 'rules' array.")
    defaults = compiled_payload.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {
            "crossing_fee_gbp": 0.0,
            "distance_fee_gbp_per_km": 0.0,
        }
    return {
        "source": str(compiled_payload.get("source", "live_runtime:toll_tariffs")).strip()
        or "live_runtime:toll_tariffs",
        "version": str(compiled_payload.get("version", "uk-v2")).strip() or "uk-v2",
        "generated_at_utc": _utc_now_iso(),
        "as_of_utc": as_of_utc,
        "defaults": defaults,
        "rules": rules,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def publish(
    *,
    scenario_path: Path,
    fuel_asset_path: Path,
    fuel_raw_path: Path,
    carbon_path: Path,
    departure_in_path: Path,
    stochastic_in_path: Path,
    toll_topology_in_path: Path,
    toll_tariffs_in_path: Path,
    departure_out_path: Path,
    stochastic_out_path: Path,
    toll_topology_out_path: Path,
    toll_tariffs_out_path: Path,
) -> dict[str, Any]:
    scenario_payload = _ensure_top_level_as_of(
        _read_json_object(scenario_path, label="scenario profiles"),
        label="scenario profiles",
    )
    if not str(scenario_payload.get("signature", "")).strip():
        raise RuntimeError("Scenario profile payload must include top-level signature in strict runtime.")
    _ = _ensure_top_level_as_of(
        _read_json_object(carbon_path, label="carbon schedule"),
        label="carbon schedule",
    )
    fuel_payload, fuel_regenerated = _ensure_fuel_signature(
        fuel_asset_path=fuel_asset_path,
        fuel_raw_path=fuel_raw_path,
    )
    departure_payload = _ensure_top_level_as_of(
        _read_json_object(departure_in_path, label="departure profiles"),
        label="departure profiles",
    )
    stochastic_payload = _ensure_top_level_as_of(
        _read_json_object(stochastic_in_path, label="stochastic regimes"),
        label="stochastic regimes",
    )
    toll_topology_compiled = _read_json_object(
        toll_topology_in_path,
        label="compiled toll topology",
    )
    toll_tariffs_compiled = _read_json_object(
        toll_tariffs_in_path,
        label="compiled toll tariffs",
    )

    publish_as_of = _utc_now_iso()
    toll_topology_payload = _build_toll_topology_payload(
        compiled_payload=toll_topology_compiled,
        as_of_utc=publish_as_of,
    )
    toll_tariffs_payload = _build_toll_tariffs_payload(
        compiled_payload=toll_tariffs_compiled,
        as_of_utc=publish_as_of,
    )

    _write_json(departure_out_path, departure_payload)
    _write_json(stochastic_out_path, stochastic_payload)
    _write_json(toll_topology_out_path, toll_topology_payload)
    _write_json(toll_tariffs_out_path, toll_tariffs_payload)

    return {
        "published_at_utc": _utc_now_iso(),
        "fuel_signature_regenerated": bool(fuel_regenerated),
        "inputs": {
            "scenario_profiles": str(scenario_path),
            "fuel_prices": str(fuel_asset_path),
            "carbon_schedule": str(carbon_path),
            "departure_profiles": str(departure_in_path),
            "stochastic_regimes": str(stochastic_in_path),
            "toll_topology_compiled": str(toll_topology_in_path),
            "toll_tariffs_compiled": str(toll_tariffs_in_path),
        },
        "outputs": {
            "departure_profiles": str(departure_out_path),
            "stochastic_regimes": str(stochastic_out_path),
            "toll_topology": str(toll_topology_out_path),
            "toll_tariffs": str(toll_tariffs_out_path),
        },
        "scenario_signature_prefix": str(scenario_payload.get("signature", ""))[:12],
        "fuel_signature_prefix": str(fuel_payload.get("signature", ""))[:12],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish strict live-runtime JSON artifacts into tracked backend/assets/uk files.",
    )
    parser.add_argument("--scenario", type=Path, default=Path("backend/assets/uk/scenario_profiles_uk.json"))
    parser.add_argument("--fuel-asset", type=Path, default=Path("backend/assets/uk/fuel_prices_uk.json"))
    parser.add_argument("--fuel-raw", type=Path, default=Path("backend/data/raw/uk/fuel_prices_raw.json"))
    parser.add_argument("--carbon", type=Path, default=Path("backend/assets/uk/carbon_price_schedule_uk.json"))
    parser.add_argument(
        "--departure-in",
        type=Path,
        default=Path("backend/out/model_assets/departure_profiles_uk.json"),
    )
    parser.add_argument(
        "--stochastic-in",
        type=Path,
        default=Path("backend/out/model_assets/stochastic_regimes_uk.json"),
    )
    parser.add_argument(
        "--toll-topology-in",
        type=Path,
        default=Path("backend/out/model_assets/toll_segments_seed_compiled.json"),
    )
    parser.add_argument(
        "--toll-tariffs-in",
        type=Path,
        default=Path("backend/out/model_assets/toll_tariffs_uk_compiled.json"),
    )
    parser.add_argument(
        "--departure-out",
        type=Path,
        default=Path("backend/assets/uk/departure_profiles_uk.json"),
    )
    parser.add_argument(
        "--stochastic-out",
        type=Path,
        default=Path("backend/assets/uk/stochastic_regimes_uk.json"),
    )
    parser.add_argument(
        "--toll-topology-out",
        type=Path,
        default=Path("backend/assets/uk/toll_topology_uk.json"),
    )
    parser.add_argument(
        "--toll-tariffs-out",
        type=Path,
        default=Path("backend/assets/uk/toll_tariffs_uk.json"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("backend/out/model_assets/live_publish_summary.json"),
    )
    args = parser.parse_args()

    summary = publish(
        scenario_path=args.scenario,
        fuel_asset_path=args.fuel_asset,
        fuel_raw_path=args.fuel_raw,
        carbon_path=args.carbon,
        departure_in_path=args.departure_in,
        stochastic_in_path=args.stochastic_in,
        toll_topology_in_path=args.toll_topology_in,
        toll_tariffs_in_path=args.toll_tariffs_in,
        departure_out_path=args.departure_out,
        stochastic_out_path=args.stochastic_out,
        toll_topology_out_path=args.toll_topology_out,
        toll_tariffs_out_path=args.toll_tariffs_out,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Published strict live artifacts to: {args.departure_out.parent}")
    print(f"Summary: {args.summary_out}")


if __name__ == "__main__":
    main()
