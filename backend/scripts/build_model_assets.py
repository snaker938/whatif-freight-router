from __future__ import annotations

# ruff: noqa: E402
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_departure_profiles_uk import build as build_departure_profiles
from build_pricing_tables_uk import build as build_pricing_tables
from build_routing_graph_uk import build as build_routing_graph
from build_scenario_profiles_uk import build as build_scenario_profiles
from build_stochastic_calibration_uk import build as build_stochastic_regimes
from build_terrain_tiles_uk import build_assets as build_terrain_assets
from extract_osm_tolls_uk import extract as extract_toll_assets
from fetch_carbon_intensity_uk import augment_carbon_schedule, build_intensity_asset
from fetch_dft_counts_uk import build as build_departure_counts_empirical
from fetch_fuel_history_uk import build as build_fuel_history
from fetch_public_dem_tiles_uk import fetch_tiles as fetch_public_dem_tiles
from fetch_stochastic_residuals_uk import build as build_stochastic_residuals
from fetch_toll_truth_uk import build as build_toll_truth_fixtures
from validate_graph_coverage import validate as validate_graph_coverage

from app.calibration_loader import (
    load_departure_profile,
    load_fuel_consumption_surface,
    load_fuel_price_snapshot,
    load_fuel_uncertainty_surface,
    load_scenario_profiles,
    load_stochastic_regimes,
    load_toll_segments_seed,
    load_toll_tariffs,
    load_uk_bank_holidays,
)
from app.settings import settings
from app.vehicles import load_builtin_vehicles


def _ci_strict_mode() -> bool:
    return str(os.environ.get("CI", "")).strip().lower() in {"1", "true", "yes"}


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def _json_array_len(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    value = payload.get(key)
    if isinstance(value, list):
        return len(value)
    return 0


def _fixture_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.json")))


def _graph_meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _load_graph_meta(path: Path) -> dict[str, int] | None:
    meta_path = _graph_meta_path(path)
    payload: dict[str, Any] | None = None
    if meta_path.exists():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta_payload = None
        if isinstance(meta_payload, dict):
            payload = meta_payload
    if payload is None:
        return None
    try:
        nodes = int(payload.get("nodes", 0))
        edges = int(payload.get("edges", 0))
    except (TypeError, ValueError):
        return None
    if nodes <= 0 or edges <= 0:
        return None
    return {"nodes": nodes, "edges": edges}


def _existing_topology_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    features = payload.get("features", None)
    return isinstance(features, list) and len(features) > 0


def _existing_graph_valid(path: Path, *, min_nodes: int, min_edges: int) -> bool:
    if not path.exists():
        return False
    meta = _load_graph_meta(path)
    if meta is not None:
        return (
            int(meta["nodes"]) >= max(1, int(min_nodes))
            and int(meta["edges"]) >= max(1, int(min_edges))
        )
    return False


def _existing_terrain_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    tiles = payload.get("tiles", None)
    if not isinstance(tiles, list) or not tiles:
        return False
    base_dir = path.parent
    for row in tiles:
        if not isinstance(row, dict):
            continue
        rel = str(row.get("path", "")).strip()
        if not rel:
            continue
        tile_path = base_dir / rel
        if not tile_path.exists():
            # Also support manifests copied at out root.
            alt_path = base_dir.parent / rel
            if not alt_path.exists():
                return False
    return True


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _require_fresh_as_of(*, label: str, as_of: datetime, max_age_days: int) -> None:
    now = datetime.now(UTC)
    if as_of > (now + timedelta(days=2)):
        raise RuntimeError(f"{label} as_of_utc is unexpectedly in the future: {as_of.isoformat()}")
    if now - as_of > timedelta(days=max(1, int(max_age_days))):
        raise RuntimeError(
            f"{label} is stale for strict build policy "
            f"({as_of.isoformat()} older than {int(max_age_days)} days)."
        )


def _validate_departure_counts_asset(
    path: Path,
    *,
    min_rows: int,
    min_regions: int,
    min_road_buckets: int,
    min_hours: int,
    max_age_days: int,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Empirical departure counts asset missing: {path}")
    required_columns = {"region", "road_bucket", "day_kind", "minute", "multiplier", "as_of_utc"}
    row_count = 0
    regions: set[str] = set()
    road_buckets: set[str] = set()
    hours: set[int] = set()
    day_kinds: set[str] = set()
    latest_as_of: datetime | None = None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not isinstance(reader.fieldnames, list) or not required_columns.issubset(set(reader.fieldnames)):
            missing = sorted(required_columns - set(reader.fieldnames or []))
            raise RuntimeError(
                f"Empirical departure counts asset schema mismatch ({path}); missing columns: {', '.join(missing)}"
            )
        for row in reader:
            row_count += 1
            region = str(row.get("region", "")).strip().lower()
            road_bucket = str(row.get("road_bucket", "")).strip().lower()
            day_kind = str(row.get("day_kind", "")).strip().lower()
            if region:
                regions.add(region)
            if road_bucket:
                road_buckets.add(road_bucket)
            day_kinds.add(day_kind)
            try:
                minute = int(float(str(row.get("minute", "0")).strip() or "0"))
                multiplier = float(str(row.get("multiplier", "1")).strip() or "1")
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Empirical departure counts asset has invalid numeric row: {row!r}") from exc
            minute = max(0, min(1439, minute))
            hours.add(minute // 60)
            if multiplier <= 0.0:
                raise RuntimeError("Empirical departure counts asset contains non-positive multipliers.")
            as_of = _parse_iso_utc(row.get("as_of_utc"))
            if as_of is None:
                raise RuntimeError("Empirical departure counts asset contains invalid as_of_utc values.")
            latest_as_of = as_of if latest_as_of is None else max(latest_as_of, as_of)
    if row_count < max(1, int(min_rows)):
        raise RuntimeError(
            f"Empirical departure counts corpus too small ({row_count} < {int(min_rows)})."
        )
    required_days = {"weekday", "weekend", "holiday"}
    if not required_days.issubset(day_kinds):
        missing_days = sorted(required_days - day_kinds)
        raise RuntimeError(
            "Empirical departure counts corpus is missing required day_kind values: "
            + ", ".join(missing_days)
        )
    if len(regions) < max(1, int(min_regions)):
        raise RuntimeError(
            f"Empirical departure counts regional diversity too small ({len(regions)} < {int(min_regions)})."
        )
    if len(road_buckets) < max(1, int(min_road_buckets)):
        raise RuntimeError(
            "Empirical departure counts road-bucket diversity too small "
            f"({len(road_buckets)} < {int(min_road_buckets)})."
        )
    if len(hours) < max(1, int(min_hours)):
        raise RuntimeError(
            f"Empirical departure counts hour coverage too small ({len(hours)} < {int(min_hours)})."
        )
    if latest_as_of is None:
        raise RuntimeError("Empirical departure counts corpus is missing as_of_utc coverage.")
    _require_fresh_as_of(
        label="Empirical departure counts corpus",
        as_of=latest_as_of,
        max_age_days=max_age_days,
    )


def _validate_stochastic_residual_asset(
    path: Path,
    *,
    min_rows: int,
    min_regimes: int,
    min_road_buckets: int,
    min_weather_profiles: int,
    min_vehicle_types: int,
    min_local_slots: int,
    min_unique_corridors: int,
    max_age_days: int,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Empirical stochastic residual asset missing: {path}")
    required_columns = {
        "regime_id",
        "corridor_bucket",
        "day_kind",
        "local_time_slot",
        "road_bucket",
        "weather_profile",
        "vehicle_type",
        "traffic",
        "incident",
        "weather",
        "price",
        "eco",
        "sigma",
        "as_of_utc",
    }
    row_count = 0
    regimes: set[str] = set()
    roads: set[str] = set()
    weather_profiles: set[str] = set()
    vehicles: set[str] = set()
    slots: set[str] = set()
    corridors: set[str] = set()
    day_kinds: set[str] = set()
    latest_as_of: datetime | None = None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not isinstance(reader.fieldnames, list) or not required_columns.issubset(set(reader.fieldnames)):
            missing = sorted(required_columns - set(reader.fieldnames or []))
            raise RuntimeError(
                f"Empirical residual asset schema mismatch ({path}); missing columns: {', '.join(missing)}"
            )
        for row in reader:
            row_count += 1
            regimes.add(str(row.get("regime_id", "")).strip().lower())
            roads.add(str(row.get("road_bucket", "")).strip().lower())
            weather_profiles.add(str(row.get("weather_profile", "")).strip().lower())
            vehicles.add(str(row.get("vehicle_type", "")).strip().lower())
            slots.add(str(row.get("local_time_slot", "")).strip().lower())
            corridors.add(str(row.get("corridor_bucket", "")).strip().lower())
            day_kinds.add(str(row.get("day_kind", "")).strip().lower())
            for field, low, high in (
                ("traffic", 0.25, 4.5),
                ("incident", 0.25, 4.5),
                ("weather", 0.25, 4.5),
                ("price", 0.25, 4.5),
                ("eco", 0.25, 4.5),
                ("sigma", 0.01, 2.5),
            ):
                try:
                    value = float(str(row.get(field, "nan")).strip() or "nan")
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"Empirical residual asset has invalid numeric row for '{field}'.") from exc
                if value != value or value < low or value > high:
                    raise RuntimeError(
                        f"Empirical residual asset {field} value out of strict bounds [{low}, {high}]: {value!r}"
                    )
            as_of = _parse_iso_utc(row.get("as_of_utc"))
            if as_of is None:
                raise RuntimeError("Empirical residual asset contains invalid as_of_utc values.")
            latest_as_of = as_of if latest_as_of is None else max(latest_as_of, as_of)
    if row_count < max(1, int(min_rows)):
        raise RuntimeError(
            f"Empirical residual corpus too small ({row_count} < {int(min_rows)})."
        )
    required_days = {"weekday", "weekend", "holiday"}
    if not required_days.issubset(day_kinds):
        missing_days = sorted(required_days - day_kinds)
        raise RuntimeError(
            "Empirical residual corpus missing required day_kind values: " + ", ".join(missing_days)
        )
    if len(regimes - {""}) < max(1, int(min_regimes)):
        raise RuntimeError(
            f"Empirical residual regime diversity too small ({len(regimes - {''})} < {int(min_regimes)})."
        )
    if len(roads - {""}) < max(1, int(min_road_buckets)):
        raise RuntimeError(
            f"Empirical residual road-bucket diversity too small ({len(roads - {''})} < {int(min_road_buckets)})."
        )
    if len(weather_profiles - {""}) < max(1, int(min_weather_profiles)):
        raise RuntimeError(
            "Empirical residual weather-profile diversity too small "
            f"({len(weather_profiles - {''})} < {int(min_weather_profiles)})."
        )
    if len(vehicles - {""}) < max(1, int(min_vehicle_types)):
        raise RuntimeError(
            f"Empirical residual vehicle-type diversity too small ({len(vehicles - {''})} < {int(min_vehicle_types)})."
        )
    if len(slots - {""}) < max(1, int(min_local_slots)):
        raise RuntimeError(
            f"Empirical residual local-slot diversity too small ({len(slots - {''})} < {int(min_local_slots)})."
        )
    if len(corridors - {""}) < max(1, int(min_unique_corridors)):
        raise RuntimeError(
            "Empirical residual corridor diversity too small "
            f"({len(corridors - {''})} < {int(min_unique_corridors)})."
        )
    if latest_as_of is None:
        raise RuntimeError("Empirical residual corpus is missing as_of_utc coverage.")
    _require_fresh_as_of(
        label="Empirical residual corpus",
        as_of=latest_as_of,
        max_age_days=max_age_days,
    )


def _validate_scenario_jsonl_asset(
    path: Path,
    *,
    min_rows: int,
    min_corridors: int,
    min_hours: int,
    max_age_days: int,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Scenario raw corpus missing: {path}")
    row_count = 0
    corridors: set[str] = set()
    hours: set[int] = set()
    day_kinds: set[str] = set()
    latest_as_of: datetime | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception as exc:
                raise RuntimeError(f"Scenario raw corpus contains invalid JSONL row: {text[:120]!r}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("Scenario raw corpus contains non-object JSON rows.")
            row_count += 1
            corridor = str(payload.get("corridor_geohash5", payload.get("corridor_bucket", ""))).strip().lower()
            if corridor:
                corridors.add(corridor)
            try:
                hour_slot = int(float(payload.get("hour_slot_local", 12)))
            except (TypeError, ValueError):
                hour_slot = 12
            hours.add(max(0, min(23, hour_slot)))
            day_kinds.add(str(payload.get("day_kind", "weekday")).strip().lower())
            as_of = _parse_iso_utc(payload.get("as_of_utc"))
            if as_of is not None:
                latest_as_of = as_of if latest_as_of is None else max(latest_as_of, as_of)
    if row_count < max(1, int(min_rows)):
        raise RuntimeError(f"Scenario raw corpus too small ({row_count} < {int(min_rows)}).")
    if len(corridors) < max(1, int(min_corridors)):
        raise RuntimeError(
            f"Scenario raw corpus corridor diversity too small ({len(corridors)} < {int(min_corridors)})."
        )
    if len(hours) < max(1, int(min_hours)):
        raise RuntimeError(
            f"Scenario raw corpus hour coverage too small ({len(hours)} < {int(min_hours)})."
        )
    required_days = {"weekday", "weekend"}
    if not required_days.issubset(day_kinds):
        missing_days = sorted(required_days - day_kinds)
        raise RuntimeError(
            "Scenario raw corpus missing required day kinds: " + ", ".join(missing_days)
        )
    if latest_as_of is None:
        raise RuntimeError("Scenario raw corpus missing parseable as_of_utc timestamps.")
    _require_fresh_as_of(
        label="Scenario raw corpus",
        as_of=latest_as_of,
        max_age_days=max_age_days,
    )


def _validate_scenario_profiles_output(
    path: Path,
    *,
    min_contexts: int,
    max_age_days: int,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Scenario profiles asset missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Scenario profiles asset is not a JSON object.")
    if str(payload.get("calibration_basis", "")).strip().lower().startswith("synthetic"):
        raise RuntimeError("Scenario profiles asset calibration_basis is synthetic.")
    split_strategy = str(payload.get("split_strategy", "")).strip().lower()
    if split_strategy != "temporal_forward_plus_corridor_block":
        raise RuntimeError("Scenario profiles asset split_strategy is not strict-compliant.")
    contexts = payload.get("contexts")
    if not isinstance(contexts, list) or len(contexts) < max(1, int(min_contexts)):
        raise RuntimeError(
            "Scenario profiles asset has insufficient context coverage "
            f"({len(contexts) if isinstance(contexts, list) else 0} < {int(min_contexts)})."
        )
    holdout_metrics = payload.get("holdout_metrics")
    if not isinstance(holdout_metrics, dict):
        raise RuntimeError("Scenario profiles asset missing holdout_metrics.")
    observed_share = float(holdout_metrics.get("observed_mode_row_share", 0.0))
    if observed_share < float(settings.scenario_min_observed_mode_row_share):
        raise RuntimeError(
            "Scenario profiles observed_mode_row_share below strict threshold "
            f"({observed_share:.6f} < {float(settings.scenario_min_observed_mode_row_share):.6f})."
        )
    as_of = _parse_iso_utc(payload.get("as_of_utc", payload.get("generated_at_utc")))
    if as_of is None:
        raise RuntimeError("Scenario profiles asset is missing parseable as_of_utc.")
    _require_fresh_as_of(
        label="Scenario profiles asset",
        as_of=as_of,
        max_age_days=max_age_days,
    )


def _validate_toll_fixture_dir(path: Path, *, required_keys: set[str], min_count: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Toll fixture directory missing: {path}")
    fixture_files = sorted(path.glob("*.json"))
    if len(fixture_files) < max(1, int(min_count)):
        raise RuntimeError(
            f"Toll fixture corpus too small in {path} ({len(fixture_files)} < {int(min_count)})."
        )
    seen_ids: set[str] = set()
    for fixture in fixture_files:
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Toll fixture is not a JSON object: {fixture}")
        missing = sorted(required_keys - set(payload.keys()))
        if missing:
            raise RuntimeError(f"Toll fixture missing required keys ({fixture}): {', '.join(missing)}")
        fixture_id = str(payload.get("fixture_id", "")).strip()
        if not fixture_id:
            raise RuntimeError(f"Toll fixture has empty fixture_id: {fixture}")
        if fixture_id in seen_ids:
            raise RuntimeError(f"Duplicate fixture_id '{fixture_id}' in {path}")
        seen_ids.add(fixture_id)
        route_fixture = str(payload.get("route_fixture", "")).strip()
        if not route_fixture:
            raise RuntimeError(f"Toll fixture has empty route_fixture: {fixture}")
        if "expected_has_toll" in required_keys:
            raw_label = str(payload.get("expected_has_toll", "")).strip().lower()
            if raw_label not in {"true", "false", "1", "0", "yes", "no"} and not isinstance(
                payload.get("expected_has_toll"), bool
            ):
                raise RuntimeError(f"Toll classification fixture has invalid expected_has_toll: {fixture}")
        if "expected_contains_toll" in required_keys:
            raw_label = str(payload.get("expected_contains_toll", "")).strip().lower()
            if raw_label not in {"true", "false", "1", "0", "yes", "no"} and not isinstance(
                payload.get("expected_contains_toll"), bool
            ):
                raise RuntimeError(f"Toll pricing fixture has invalid expected_contains_toll: {fixture}")
            raw_expected_cost = payload.get("expected_toll_cost_gbp")
            if raw_expected_cost is None:
                raise RuntimeError(f"Toll pricing fixture has missing expected_toll_cost_gbp: {fixture}")
            try:
                expected_cost = float(raw_expected_cost)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Toll pricing fixture has invalid expected_toll_cost_gbp: {fixture}") from exc
            if expected_cost < 0.0:
                raise RuntimeError(f"Toll pricing fixture has negative expected_toll_cost_gbp: {fixture}")


def _validate_toll_confidence_asset(path: Path, *, max_age_days: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Toll confidence calibration asset missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Toll confidence calibration asset is not a JSON object.")
    model = payload.get("logit_model")
    bins = payload.get("reliability_bins")
    if not isinstance(model, dict):
        raise RuntimeError("Toll confidence calibration asset is missing logit_model.")
    if not isinstance(bins, list) or len(bins) < 5:
        raise RuntimeError("Toll confidence calibration asset is missing reliability bins.")
    as_of = _parse_iso_utc(payload.get("as_of_utc", payload.get("generated_at_utc")))
    if as_of is None:
        raise RuntimeError("Toll confidence calibration asset is missing parseable as_of_utc.")
    _require_fresh_as_of(
        label="Toll confidence calibration asset",
        as_of=as_of,
        max_age_days=max_age_days,
    )


def _validate_stochastic_outputs(
    *,
    regimes_path: Path,
    priors_path: Path,
    max_age_days: int,
) -> None:
    if not regimes_path.exists():
        raise FileNotFoundError(f"Stochastic regimes output missing: {regimes_path}")
    if not priors_path.exists():
        raise FileNotFoundError(f"Stochastic priors output missing: {priors_path}")
    regimes_payload = json.loads(regimes_path.read_text(encoding="utf-8"))
    priors_payload = json.loads(priors_path.read_text(encoding="utf-8"))
    if not isinstance(regimes_payload, dict) or not isinstance(priors_payload, dict):
        raise RuntimeError("Stochastic outputs are not valid JSON objects.")
    if str(regimes_payload.get("calibration_basis", "")).strip().lower() != "empirical":
        raise RuntimeError("Stochastic regimes output is not empirical.")
    if str(regimes_payload.get("split_strategy", "")).strip().lower() != "temporal_forward_plus_corridor_block":
        raise RuntimeError("Stochastic regimes split_strategy is not strict-compliant.")
    regimes = regimes_payload.get("regimes")
    if not isinstance(regimes, dict) or not regimes:
        raise RuntimeError("Stochastic regimes output is missing regimes map.")
    priors = priors_payload.get("priors")
    if not isinstance(priors, list) or not priors:
        raise RuntimeError("Stochastic priors output is missing priors list.")
    regimes_as_of = _parse_iso_utc(regimes_payload.get("as_of_utc", regimes_payload.get("generated_at_utc")))
    priors_as_of = _parse_iso_utc(priors_payload.get("as_of_utc", priors_payload.get("generated_at_utc")))
    if regimes_as_of is None or priors_as_of is None:
        raise RuntimeError("Stochastic outputs are missing parseable as_of_utc timestamps.")
    _require_fresh_as_of(
        label="Stochastic regimes output",
        as_of=regimes_as_of,
        max_age_days=max_age_days,
    )
    _require_fresh_as_of(
        label="Stochastic priors output",
        as_of=priors_as_of,
        max_age_days=max_age_days,
    )


def build_assets(
    *,
    out_dir: Path,
    departure_counts_csv: Path | None = None,
    stochastic_residuals_csv: Path | None = None,
    routing_graph_source: Path | None = None,
    routing_graph_max_ways: int = 0,
    allow_synthetic: bool = False,
    allow_geojson_routing_graph: bool = False,
    force_rebuild_topology: bool = False,
    force_rebuild_graph: bool = False,
    force_rebuild_terrain: bool = False,
) -> None:
    if allow_synthetic:
        raise ValueError("Synthetic asset generation is disabled in strict runtime.")
    if allow_geojson_routing_graph:
        raise ValueError("GeoJSON routing graph fallback is disabled in strict runtime.")
    raw_root = ROOT / "data" / "raw" / "uk"
    required_raw_paths = [
        raw_root / "dft_counts_raw.csv",
        raw_root / "stochastic_residuals_raw.csv",
        raw_root / "scenario_live_observed.jsonl",
        raw_root / "scenario_mode_outcomes_observed.jsonl",
        raw_root / "fuel_prices_raw.json",
        raw_root / "carbon_intensity_hourly_raw.json",
        raw_root / "toll_tariffs_operator_truth.json",
        raw_root / "toll_classification",
        raw_root / "toll_pricing",
    ]
    missing_raw = [
        str(path)
        for path in required_raw_paths
        if (not path.exists()) or (path.is_dir() and not any(path.glob("*.json")))
    ]
    if missing_raw:
        raise FileNotFoundError(
            "Strict empirical build requires external raw datasets. Missing: "
            + ", ".join(missing_raw)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    departure_counts_default = ROOT / "assets" / "uk" / "departure_counts_empirical.csv"
    stochastic_residuals_default = ROOT / "assets" / "uk" / "stochastic_residuals_empirical.csv"
    departure_counts_input = departure_counts_csv
    if departure_counts_input is None and departure_counts_default.exists():
        departure_counts_input = departure_counts_default
    stochastic_residuals_input = stochastic_residuals_csv
    if stochastic_residuals_input is None and stochastic_residuals_default.exists():
        stochastic_residuals_input = stochastic_residuals_default

    # Strict empirical backbone generation/validation.
    if departure_counts_input is None:
        departure_counts_input = departure_counts_default
    try:
        _validate_departure_counts_asset(
            departure_counts_input,
            min_rows=2000,
            min_regions=8,
            min_road_buckets=4,
            min_hours=18,
            max_age_days=int(settings.live_departure_max_age_days),
        )
    except Exception:
        raw_dft_counts = ROOT / "data" / "raw" / "uk" / "dft_counts_raw.csv"
        build_departure_counts_empirical(
            raw_csv=raw_dft_counts,
            output_csv=departure_counts_input,
            min_rows=2000,
            min_unique_regions=8,
            min_unique_road_buckets=4,
            min_unique_hours=18,
            max_age_days=int(settings.live_departure_max_age_days),
        )
    _validate_departure_counts_asset(
        departure_counts_input,
        min_rows=2000,
        min_regions=8,
        min_road_buckets=4,
        min_hours=18,
        max_age_days=int(settings.live_departure_max_age_days),
    )

    if stochastic_residuals_input is None:
        stochastic_residuals_input = stochastic_residuals_default
    try:
        _validate_stochastic_residual_asset(
            stochastic_residuals_input,
            min_rows=5000,
            min_regimes=3,
            min_road_buckets=3,
            min_weather_profiles=3,
            min_vehicle_types=3,
            min_local_slots=4,
            min_unique_corridors=1,
            max_age_days=int(settings.live_stochastic_max_age_days),
        )
    except Exception:
        raw_stochastic = ROOT / "data" / "raw" / "uk" / "stochastic_residuals_raw.csv"
        build_stochastic_residuals(
            raw_csv=raw_stochastic,
            output_csv=stochastic_residuals_input,
            min_rows=5000,
            min_unique_regimes=3,
            min_unique_road_buckets=3,
            min_unique_weather_profiles=3,
            min_unique_vehicle_types=3,
            min_unique_local_slots=4,
            min_unique_corridors=1,
            max_age_days=int(settings.live_stochastic_max_age_days),
        )
    _validate_stochastic_residual_asset(
        stochastic_residuals_input,
        min_rows=5000,
        min_regimes=3,
        min_road_buckets=3,
        min_weather_profiles=3,
        min_vehicle_types=3,
        min_local_slots=4,
        min_unique_corridors=1,
        max_age_days=int(settings.live_stochastic_max_age_days),
    )

    fuel_asset_path = ROOT / "assets" / "uk" / "fuel_prices_uk.json"
    if _json_array_len(fuel_asset_path, "history") < 365:
        raw_fuel_history = ROOT / "data" / "raw" / "uk" / "fuel_prices_raw.json"
        build_fuel_history(
            source_json=raw_fuel_history,
            output_json=fuel_asset_path,
            min_history_days=365,
        )
    if _json_array_len(fuel_asset_path, "history") < 365:
        raise RuntimeError("Fuel price history must contain at least 365 daily rows.")
    fuel_consumption_surface_path = ROOT / "assets" / "uk" / "fuel_consumption_surface_uk.json"
    fuel_uncertainty_surface_path = ROOT / "assets" / "uk" / "fuel_uncertainty_surface_uk.json"
    scenario_profiles_asset_path = ROOT / "assets" / "uk" / "scenario_profiles_uk.json"
    scenario_live_observed = ROOT / "data" / "raw" / "uk" / "scenario_live_observed.jsonl"
    scenario_mode_outcomes_observed = ROOT / "data" / "raw" / "uk" / "scenario_mode_outcomes_observed.jsonl"
    vehicle_profiles_asset = ROOT / "assets" / "uk" / "vehicle_profiles_uk.json"
    if not fuel_consumption_surface_path.exists():
        raise RuntimeError(
            f"Fuel consumption surface asset missing: {fuel_consumption_surface_path}"
        )
    if not fuel_uncertainty_surface_path.exists():
        raise RuntimeError(
            f"Fuel uncertainty surface asset missing: {fuel_uncertainty_surface_path}"
        )
    _validate_scenario_jsonl_asset(
        scenario_live_observed,
        min_rows=60,
        min_corridors=8,
        min_hours=6,
        max_age_days=max(1, int(settings.live_departure_max_age_days)),
    )
    _validate_scenario_jsonl_asset(
        scenario_mode_outcomes_observed,
        min_rows=30,
        min_corridors=8,
        min_hours=6,
        max_age_days=max(1, int(settings.live_departure_max_age_days)),
    )
    build_scenario_profiles(
        raw_jsonl=scenario_live_observed,
        observed_modes_jsonl=scenario_mode_outcomes_observed,
        output_json=scenario_profiles_asset_path,
        min_contexts=8,
        min_observed_mode_row_share=float(settings.scenario_min_observed_mode_row_share),
        max_projection_dominant_context_share=float(settings.scenario_max_projection_dominant_context_share),
    )
    _validate_scenario_profiles_output(
        scenario_profiles_asset_path,
        min_contexts=8,
        max_age_days=max(1, int(settings.live_departure_max_age_days)),
    )
    if not vehicle_profiles_asset.exists():
        raise RuntimeError(
            f"Vehicle profiles asset missing: {vehicle_profiles_asset}"
        )

    carbon_intensity_asset = ROOT / "assets" / "uk" / "carbon_intensity_hourly_uk.json"
    if not carbon_intensity_asset.exists():
        raw_intensity = ROOT / "data" / "raw" / "uk" / "carbon_intensity_hourly_raw.json"
        build_intensity_asset(source_json=raw_intensity, output_json=carbon_intensity_asset)
    augment_carbon_schedule(schedule_json=ROOT / "assets" / "uk" / "carbon_price_schedule_uk.json")

    toll_classification_fixtures = ROOT / "tests" / "fixtures" / "toll_classification"
    toll_pricing_fixtures = ROOT / "tests" / "fixtures" / "toll_pricing"
    toll_confidence_asset = ROOT / "assets" / "uk" / "toll_confidence_calibration_uk.json"
    try:
        _validate_toll_fixture_dir(
            toll_classification_fixtures,
            required_keys={"fixture_id", "route_fixture", "expected_has_toll"},
            min_count=200,
        )
        _validate_toll_fixture_dir(
            toll_pricing_fixtures,
            required_keys={"fixture_id", "route_fixture", "expected_contains_toll", "expected_toll_cost_gbp"},
            min_count=80,
        )
        _validate_toll_confidence_asset(
            toll_confidence_asset,
            max_age_days=int(settings.live_toll_topology_max_age_days),
        )
    except Exception:
        build_toll_truth_fixtures(
            classification_source_dir=ROOT / "data" / "raw" / "uk" / "toll_classification",
            pricing_source_dir=ROOT / "data" / "raw" / "uk" / "toll_pricing",
            classification_out_dir=toll_classification_fixtures,
            pricing_out_dir=toll_pricing_fixtures,
            classification_target=200,
            pricing_target=80,
            calibration_out_json=toll_confidence_asset,
        )
    _validate_toll_fixture_dir(
        toll_classification_fixtures,
        required_keys={"fixture_id", "route_fixture", "expected_has_toll"},
        min_count=200,
    )
    _validate_toll_fixture_dir(
        toll_pricing_fixtures,
        required_keys={"fixture_id", "route_fixture", "expected_contains_toll", "expected_toll_cost_gbp"},
        min_count=80,
    )
    _validate_toll_confidence_asset(
        toll_confidence_asset,
        max_age_days=int(settings.live_toll_topology_max_age_days),
    )

    # Build contextual departure + stochastic regime assets first so runtime loaders
    # consume compiled model assets rather than bundled defaults.
    sparse_departure_seed = ROOT / "assets" / "uk" / "departure_profile_uk.csv"
    if not sparse_departure_seed.exists():
        sparse_departure_seed = departure_counts_input
    build_departure_profiles(
        sparse_csv=sparse_departure_seed,
        output_json=out_dir / "departure_profiles_uk.json",
        counts_csv=departure_counts_input,
        allow_synthetic=allow_synthetic,
    )
    build_stochastic_regimes(
        output_json=out_dir / "stochastic_regimes_uk.json",
        output_priors_json=out_dir / "stochastic_residual_priors_uk.json",
        residuals_csv=stochastic_residuals_input,
        allow_synthetic=allow_synthetic,
    )
    priors_out = out_dir / "stochastic_residual_priors_uk.json"
    _validate_stochastic_outputs(
        regimes_path=out_dir / "stochastic_regimes_uk.json",
        priors_path=priors_out,
        max_age_days=int(settings.live_stochastic_max_age_days),
    )
    priors_asset = ROOT / "assets" / "uk" / "stochastic_residual_priors_uk.json"
    if priors_out.exists():
        priors_asset.write_text(priors_out.read_text(encoding="utf-8"), encoding="utf-8")
    toll_source_candidates: list[Path] = []
    if routing_graph_source is not None:
        toll_source_candidates.append(routing_graph_source)
    toll_source_candidates.extend(
        [
            out_dir / "osm_uk.pbf",
            out_dir / "osm_uk.osm",
            ROOT / "assets" / "uk" / "uk-latest.osm.pbf",
            ROOT / "assets" / "uk" / "uk-latest.osm",
        ]
    )
    toll_source = next((candidate for candidate in toll_source_candidates if candidate.exists()), None)
    existing_toll_topology = out_dir / "osm_toll_assets.geojson"
    if toll_source is None:
        raise FileNotFoundError(
            "No toll topology source found. Provide OSM PBF via --routing-graph-source or place uk-latest.osm.pbf in backend/assets/uk/."
        )
    else:
        if toll_source.suffix.lower() != ".pbf":
            raise FileNotFoundError(
                "Strict build requires OSM PBF toll topology source."
            )
        if force_rebuild_topology or not _existing_topology_valid(existing_toll_topology):
            extract_toll_assets(
                source_geojson=toll_source,
                output_geojson=existing_toll_topology,
            )
    tariff_truth_source = ROOT / "data" / "raw" / "uk" / "toll_tariffs_operator_truth.json"
    build_pricing_tables(
        fuel_source=ROOT / "assets" / "uk" / "fuel_prices_uk.json",
        carbon_source=ROOT / "assets" / "uk" / "carbon_price_schedule_uk.json",
        tariff_truth_source=tariff_truth_source,
        toll_tariffs_output=ROOT / "assets" / "uk" / "toll_tariffs_uk.yaml",
        output_dir=out_dir,
        min_tariff_rules=200,
    )
    graph_source_candidates: list[Path] = []
    if routing_graph_source is not None:
        graph_source_candidates.append(routing_graph_source)
    graph_source_candidates.extend(
        [
            out_dir / "osm_uk.pbf",
            out_dir / "osm_uk.osm",
            ROOT / "assets" / "uk" / "uk-latest.osm.pbf",
            ROOT / "assets" / "uk" / "uk-latest.osm",
        ]
    )
    if allow_geojson_routing_graph:
        graph_source_candidates.extend(
            [
                out_dir / "osm_toll_assets.geojson",
                ROOT / "assets" / "uk" / "osm_toll_assets.geojson",
            ]
        )
    graph_source = next((candidate for candidate in graph_source_candidates if candidate.exists()), None)
    graph_output = out_dir / "routing_graph_uk.json"
    if graph_source is None:
        raise FileNotFoundError(
            "No routing graph source found. Provide --routing-graph-source or place uk-latest.osm.pbf in backend/assets/uk/."
        )
    else:
        if graph_source.suffix.lower() != ".pbf":
            raise FileNotFoundError(
                "Routing graph requires OSM PBF source in strict mode."
            )
        if force_rebuild_graph or not _existing_graph_valid(
            graph_output,
            min_nodes=max(1, int(settings.route_graph_min_nodes)),
            min_edges=max(1, int(settings.route_graph_min_adjacency)),
        ):
            build_routing_graph(
                source=graph_source,
                output=graph_output,
                max_ways=max(0, int(routing_graph_max_ways)),
            )
    graph_meta = _load_graph_meta(graph_output)
    if graph_meta is None:
        raise RuntimeError(
            "Routing graph metadata sidecar is missing. Rebuild the graph to generate .meta.json."
        )
    if graph_meta["nodes"] < max(1, int(settings.route_graph_min_nodes)):
        raise RuntimeError(
            f"Routing graph node count below strict threshold: {graph_meta['nodes']} < {settings.route_graph_min_nodes}"
        )
    if graph_meta["edges"] < max(1, int(settings.route_graph_min_adjacency)):
        raise RuntimeError(
            f"Routing graph edge count below strict threshold: {graph_meta['edges']} < {settings.route_graph_min_adjacency}"
        )

    graph_size_mb = graph_output.stat().st_size / (1024.0 * 1024.0)
    coverage_report = validate_graph_coverage(
        graph_path=graph_output,
        fixtures_dir=ROOT / "tests" / "fixtures" / "uk_routes",
        min_nodes=max(1, int(settings.route_graph_min_nodes)),
        min_edges=max(1, int(settings.route_graph_min_adjacency)),
        max_fixture_dist_m=max(100.0, float(settings.route_graph_fixture_max_distance_m)),
    )
    coverage_report["graph_size_mb"] = round(graph_size_mb, 2)
    (out_dir / "routing_graph_coverage_report.json").write_text(
        json.dumps(coverage_report, indent=2),
        encoding="utf-8",
    )

    dep = load_departure_profile()
    (out_dir / "departure_profile_uk_compiled.json").write_text(
        json.dumps(
            {
                "source": dep.source,
                "weekday": dep.weekday,
                "weekend": dep.weekend,
                "holiday": dep.holiday,
                "holiday_dates": sorted(load_uk_bank_holidays()),
                "version": "uk-v2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    scenario_profiles = load_scenario_profiles()
    required_modes = {"no_sharing", "partial_sharing", "full_sharing"}
    global_modes = set((scenario_profiles.profiles or {}).keys())
    if required_modes.issubset(global_modes):
        pass
    elif scenario_profiles.contexts:
        for context_key, context in scenario_profiles.contexts.items():
            context_modes = set((context.profiles or {}).keys())
            if not required_modes.issubset(context_modes):
                missing = required_modes - context_modes
                raise RuntimeError(
                    "Scenario profiles context is missing required modes "
                    f"({context_key}: {', '.join(sorted(missing))})"
                )
    else:
        raise RuntimeError(
            "Scenario profiles asset missing required modes: "
            + ", ".join(sorted(required_modes - global_modes))
        )
    if not scenario_profiles.contexts:
        raise RuntimeError("Scenario profiles asset must include contextual profiles in strict build.")
    if len(scenario_profiles.contexts) < 1:
        raise RuntimeError("Scenario profiles asset contextual map is empty in strict build.")
    holdout_metrics = scenario_profiles.holdout_metrics or {}
    split_strategy = str(scenario_profiles.split_strategy or "").strip().lower()
    mode_separation = float(holdout_metrics.get("mode_separation_mean", 0.0))
    duration_mape = float(holdout_metrics.get("duration_mape", 1.0))
    monetary_mape = float(holdout_metrics.get("monetary_mape", 1.0))
    emissions_mape = float(holdout_metrics.get("emissions_mape", 1.0))
    holdout_coverage = float(holdout_metrics.get("coverage", 0.0))
    hour_slot_coverage = float(holdout_metrics.get("hour_slot_coverage", 0.0))
    corridor_coverage = float(holdout_metrics.get("corridor_coverage", 0.0))
    full_identity_share = float(holdout_metrics.get("full_identity_share", float("nan")))
    projection_context_share = float(
        holdout_metrics.get("projection_dominant_context_share", float("nan"))
    )
    observed_mode_row_share = float(
        holdout_metrics.get("observed_mode_row_share", float("nan"))
    )
    context_rows = list((scenario_profiles.contexts or {}).values())
    actual_hour_slots = len(
        {
            int(getattr(ctx, "hour_slot_local", -1))
            for ctx in context_rows
            if isinstance(getattr(ctx, "hour_slot_local", None), int)
        }
    )
    actual_corridors = len(
        {
            str(getattr(ctx, "corridor_geohash5", "")).strip().lower()
            for ctx in context_rows
            if str(getattr(ctx, "corridor_geohash5", "")).strip()
        }
    )
    if mode_separation < 0.03:
        raise RuntimeError(
            "Scenario profiles holdout mode separability below strict threshold (>= 0.03 required)."
        )
    if duration_mape > 0.08 or monetary_mape > 0.08 or emissions_mape > 0.08:
        raise RuntimeError(
            "Scenario profiles holdout MAPE exceeds strict threshold (duration/monetary/emissions <= 0.08 required)."
        )
    if holdout_coverage < 0.90:
        raise RuntimeError(
            "Scenario profiles holdout coverage below strict threshold (>= 0.90 required)."
        )
    if full_identity_share == full_identity_share and full_identity_share > 0.70:
        raise RuntimeError(
            "Scenario profiles full_sharing identity share exceeds strict threshold (<= 0.70 required)."
        )
    if observed_mode_row_share != observed_mode_row_share:
        raise RuntimeError(
            "Scenario profiles holdout_metrics.observed_mode_row_share is required in strict build."
        )
    if observed_mode_row_share < float(settings.scenario_min_observed_mode_row_share):
        raise RuntimeError(
            "Scenario profiles observed mode row share below strict threshold "
            f"(actual={observed_mode_row_share:.6f}, "
            f"required>={float(settings.scenario_min_observed_mode_row_share):.6f})."
        )
    if projection_context_share == projection_context_share and projection_context_share > float(
        settings.scenario_max_projection_dominant_context_share
    ):
        raise RuntimeError(
            "Scenario projection-dominant context share exceeds strict threshold "
            f"(actual={projection_context_share:.6f}, "
            f"required<={float(settings.scenario_max_projection_dominant_context_share):.6f})."
        )
    accepted_split = {"temporal_forward_plus_corridor_block"}
    if split_strategy not in accepted_split:
        raise RuntimeError(
            "Scenario profiles split_strategy must be 'temporal_forward_plus_corridor_block' in strict build."
        )
    if hour_slot_coverage < 6.0:
        raise RuntimeError(
            "Scenario profiles hour-slot diversity below strict threshold (>= 6 unique local-hour slots required)."
        )
    if corridor_coverage < 8.0:
        raise RuntimeError(
            "Scenario profiles corridor diversity below strict threshold (>= 8 unique corridor geohash5 buckets required)."
        )
    if actual_hour_slots < 6:
        raise RuntimeError(
            "Scenario profiles contextual tensor is too narrow by observed hour slots "
            f"(actual={actual_hour_slots}, required>=6)."
        )
    if actual_corridors < 8:
        raise RuntimeError(
            "Scenario profiles contextual tensor is too narrow by observed corridor geohash5 buckets "
            f"(actual={actual_corridors}, required>=8)."
        )

    def _enforce_full_mode_cap(label: str, profile: Any) -> None:
        for field in (
            "duration_multiplier",
            "incident_rate_multiplier",
            "incident_delay_multiplier",
            "fuel_consumption_multiplier",
            "emissions_multiplier",
            "stochastic_sigma_multiplier",
        ):
            if float(getattr(profile, field)) > 1.0 + 1e-9:
                raise RuntimeError(
                    f"Scenario full_sharing p50 cap violated for {label} field {field} (must be <= 1.0)."
                )

    full_global = scenario_profiles.profiles.get("full_sharing")
    if full_global is not None:
        _enforce_full_mode_cap("global", full_global)
    full_identity_count = 0
    full_total_count = 0
    for context_key, context in (scenario_profiles.contexts or {}).items():
        full_ctx = context.profiles.get("full_sharing")
        if full_ctx is not None:
            _enforce_full_mode_cap(f"context:{context_key}", full_ctx)
            for field in (
                "duration_multiplier",
                "incident_rate_multiplier",
                "incident_delay_multiplier",
                "fuel_consumption_multiplier",
                "emissions_multiplier",
                "stochastic_sigma_multiplier",
            ):
                full_total_count += 1
                if abs(float(getattr(full_ctx, field)) - 1.0) <= 1e-12:
                    full_identity_count += 1
    if full_total_count > 0:
        actual_full_identity_share = float(full_identity_count) / float(full_total_count)
        if actual_full_identity_share > 0.70:
            raise RuntimeError(
                "Scenario profiles full_sharing contextual identity collapse exceeds strict threshold "
                f"(actual={actual_full_identity_share:.6f}, required<=0.70)."
            )
    compiled_profiles = dict(scenario_profiles.profiles or {})
    if not compiled_profiles and scenario_profiles.contexts:
        compiled_profiles = dict(next(iter(scenario_profiles.contexts.values())).profiles)
    (out_dir / "scenario_profiles_uk_compiled.json").write_text(
        json.dumps(
            {
                "source": scenario_profiles.source,
                "version": scenario_profiles.version,
                "as_of_utc": scenario_profiles.as_of_utc,
                "generated_at_utc": scenario_profiles.generated_at_utc,
                "calibration_basis": scenario_profiles.calibration_basis,
                "signature": scenario_profiles.signature,
                "fit_window": scenario_profiles.fit_window,
                "holdout_window": scenario_profiles.holdout_window,
                "holdout_metrics": scenario_profiles.holdout_metrics,
                "split_strategy": scenario_profiles.split_strategy,
                "actual_hour_slot_coverage": int(actual_hour_slots),
                "actual_corridor_coverage": int(actual_corridors),
                "transform_params": scenario_profiles.transform_params,
                "profiles": {
                    mode: {
                        "duration_multiplier": profile.duration_multiplier,
                        "incident_rate_multiplier": profile.incident_rate_multiplier,
                        "incident_delay_multiplier": profile.incident_delay_multiplier,
                        "fuel_consumption_multiplier": profile.fuel_consumption_multiplier,
                        "emissions_multiplier": profile.emissions_multiplier,
                        "stochastic_sigma_multiplier": profile.stochastic_sigma_multiplier,
                    }
                    for mode, profile in sorted(compiled_profiles.items())
                },
                "contexts": {
                    context_key: {
                        "corridor_bucket": context.corridor_bucket,
                        "road_mix_bucket": context.road_mix_bucket,
                        "vehicle_class": context.vehicle_class,
                        "day_kind": context.day_kind,
                        "weather_bucket": context.weather_bucket,
                        "source_coverage": context.source_coverage,
                        "profiles": {
                            mode: {
                                "duration_multiplier": profile.duration_multiplier,
                                "incident_rate_multiplier": profile.incident_rate_multiplier,
                                "incident_delay_multiplier": profile.incident_delay_multiplier,
                                "fuel_consumption_multiplier": profile.fuel_consumption_multiplier,
                                "emissions_multiplier": profile.emissions_multiplier,
                                "stochastic_sigma_multiplier": profile.stochastic_sigma_multiplier,
                                "quantiles": profile.quantiles,
                            }
                            for mode, profile in sorted(context.profiles.items())
                        },
                    }
                    for context_key, context in sorted(scenario_profiles.contexts.items())
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "scenario_profiles_uk.json").write_text(
        scenario_profiles_asset_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    toll_tariffs = load_toll_tariffs()
    (out_dir / "toll_tariffs_uk_compiled.json").write_text(
        json.dumps(
            {
                "source": toll_tariffs.source,
                "default_crossing_fee_gbp": toll_tariffs.default_crossing_fee_gbp,
                "default_distance_fee_gbp_per_km": toll_tariffs.default_distance_fee_gbp_per_km,
                "rules": [
                    {
                        "id": rule.rule_id,
                        "operator": rule.operator,
                        "crossing_id": rule.crossing_id,
                        "road_class": rule.road_class,
                        "direction": rule.direction,
                        "start_minute": rule.start_minute,
                        "end_minute": rule.end_minute,
                        "crossing_fee_gbp": rule.crossing_fee_gbp,
                        "distance_fee_gbp_per_km": rule.distance_fee_gbp_per_km,
                        "vehicle_classes": list(rule.vehicle_classes),
                        "axle_classes": list(rule.axle_classes),
                        "payment_classes": list(rule.payment_classes),
                        "exemptions": list(rule.exemptions),
                    }
                    for rule in toll_tariffs.rules
                ],
                "version": "uk-v2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    toll_segments = load_toll_segments_seed()
    (out_dir / "toll_segments_seed_compiled.json").write_text(
        json.dumps(
            {
                "count": len(toll_segments),
                "segments": [
                    {
                        "id": item.segment_id,
                        "name": item.name,
                        "operator": item.operator,
                        "road_class": item.road_class,
                        "direction": item.direction,
                        "crossing_fee_gbp": item.crossing_fee_gbp,
                        "distance_fee_gbp_per_km": item.distance_fee_gbp_per_km,
                        "coordinates": item.coordinates,
                    }
                    for item in toll_segments
                ],
                "version": "uk-v2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fuel_snapshot = load_fuel_price_snapshot()
    (out_dir / "fuel_prices_uk_compiled.json").write_text(
        json.dumps(
            {
                "source": fuel_snapshot.source,
                "as_of": fuel_snapshot.as_of,
                "signature": fuel_snapshot.signature,
                "prices_gbp_per_l": fuel_snapshot.prices_gbp_per_l,
                "grid_price_gbp_per_kwh": fuel_snapshot.grid_price_gbp_per_kwh,
                "regional_multipliers": fuel_snapshot.regional_multipliers,
                "live_diagnostics": fuel_snapshot.live_diagnostics or {},
                "version": "uk-v2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fuel_consumption_surface = load_fuel_consumption_surface()
    fuel_uncertainty_surface = load_fuel_uncertainty_surface()
    vehicle_profiles = load_builtin_vehicles()
    required_vehicle_ids = {"van", "rigid_hgv", "artic_hgv"}
    if not required_vehicle_ids.issubset(set(vehicle_profiles.keys())):
        raise RuntimeError(
            "Vehicle profile asset missing required built-ins: "
            + ", ".join(sorted(required_vehicle_ids - set(vehicle_profiles.keys())))
        )
    for vehicle in vehicle_profiles.values():
        if not str(vehicle.fuel_surface_class).strip():
            raise RuntimeError(f"Vehicle profile '{vehicle.id}' missing fuel_surface_class")
        if not str(vehicle.toll_vehicle_class).strip():
            raise RuntimeError(f"Vehicle profile '{vehicle.id}' missing toll_vehicle_class")
        if not str(vehicle.toll_axle_class).strip():
            raise RuntimeError(f"Vehicle profile '{vehicle.id}' missing toll_axle_class")
        if not str(vehicle.risk_bucket).strip():
            raise RuntimeError(f"Vehicle profile '{vehicle.id}' missing risk_bucket")
        if not str(vehicle.stochastic_bucket).strip():
            raise RuntimeError(f"Vehicle profile '{vehicle.id}' missing stochastic_bucket")
    (out_dir / "fuel_consumption_surface_uk_compiled.json").write_text(
        json.dumps(
            {
                "source": fuel_consumption_surface.source,
                "version": fuel_consumption_surface.version,
                "as_of_utc": fuel_consumption_surface.as_of_utc,
                "signature": fuel_consumption_surface.signature,
                "axes": {
                    "vehicle_class": list(fuel_consumption_surface.axes.vehicle_class),
                    "load_factor": list(fuel_consumption_surface.axes.load_factor),
                    "speed_kmh": list(fuel_consumption_surface.axes.speed_kmh),
                    "grade_pct": list(fuel_consumption_surface.axes.grade_pct),
                    "ambient_temp_c": list(fuel_consumption_surface.axes.ambient_temp_c),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "fuel_uncertainty_surface_uk_compiled.json").write_text(
        json.dumps(
            {
                "source": fuel_uncertainty_surface.source,
                "version": fuel_uncertainty_surface.version,
                "as_of_utc": fuel_uncertainty_surface.as_of_utc,
                "signature": fuel_uncertainty_surface.signature,
                "axes": {
                    "vehicle_class": list(fuel_uncertainty_surface.axes.vehicle_class),
                    "load_factor": list(fuel_uncertainty_surface.axes.load_factor),
                    "speed_kmh": list(fuel_uncertainty_surface.axes.speed_kmh),
                    "grade_pct": list(fuel_uncertainty_surface.axes.grade_pct),
                    "ambient_temp_c": list(fuel_uncertainty_surface.axes.ambient_temp_c),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "vehicle_profiles_uk_compiled.json").write_text(
        json.dumps(
            {
                "count": len(vehicle_profiles),
                "profiles": [vehicle.model_dump(mode="json") for vehicle in sorted(vehicle_profiles.values(), key=lambda p: p.id)],
                "source": str(vehicle_profiles_asset),
                "version": "vehicle_profiles_uk_v2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stochastic_regimes = load_stochastic_regimes()
    posterior_model = (
        stochastic_regimes.posterior_model
        if isinstance(stochastic_regimes.posterior_model, dict)
        else None
    )
    context_probs = (
        posterior_model.get("context_to_regime_probs")
        if isinstance(posterior_model, dict)
        else None
    )
    if not isinstance(context_probs, dict) or not context_probs:
        raise RuntimeError(
            "Stochastic calibration must include posterior_model.context_to_regime_probs in strict build."
        )
    stochastic_hours = set()
    stochastic_corridors = set()
    for key in context_probs.keys():
        parts = str(key).strip().lower().split("|")
        if len(parts) < 6:
            continue
        corridor = parts[0].strip()
        slot = parts[2].strip()
        if corridor and corridor != "*":
            stochastic_corridors.add(corridor)
        if slot.startswith("h") and slot != "*":
            stochastic_hours.add(slot)
    if len(stochastic_hours) < 6:
        raise RuntimeError(
            "Stochastic posterior context hour-slot diversity below strict threshold "
            f"(actual={len(stochastic_hours)}, required>=6)."
        )
    if len(stochastic_corridors) < 8:
        raise RuntimeError(
            "Stochastic posterior context corridor diversity below strict threshold "
            f"(actual={len(stochastic_corridors)}, required>=8)."
        )
    split_strategy = str(stochastic_regimes.split_strategy or "").strip().lower()
    if split_strategy != "temporal_forward_plus_corridor_block":
        raise RuntimeError(
            "Stochastic calibration split_strategy must be temporal_forward_plus_corridor_block in strict build."
        )
    holdout_window = stochastic_regimes.holdout_window or {}
    holdout_start = str(holdout_window.get("start_utc", "")).strip() if isinstance(holdout_window, dict) else ""
    holdout_end = str(holdout_window.get("end_utc", "")).strip() if isinstance(holdout_window, dict) else ""
    if not holdout_start or not holdout_end:
        raise RuntimeError("Stochastic calibration holdout_window must include start_utc/end_utc in strict build.")
    holdout_metrics = stochastic_regimes.holdout_metrics or {}
    holdout_coverage = float(holdout_metrics.get("coverage", 0.0))
    holdout_pit = float(holdout_metrics.get("pit_mean", 0.0))
    holdout_crps = float(holdout_metrics.get("crps_mean", 1.0))
    if holdout_coverage < 0.90:
        raise RuntimeError(
            "Stochastic holdout coverage below strict threshold "
            f"(actual={holdout_coverage:.6f}, required>=0.90)."
        )
    if holdout_pit < 0.35 or holdout_pit > 0.65:
        raise RuntimeError(
            "Stochastic holdout PIT mean outside strict calibration band "
            f"(actual={holdout_pit:.6f}, expected in [0.35, 0.65])."
        )
    if holdout_crps > 0.55:
        raise RuntimeError(
            "Stochastic holdout CRPS proxy above strict threshold "
            f"(actual={holdout_crps:.6f}, required<=0.55)."
        )
    required_shock_factors = {"traffic", "incident", "weather", "price", "eco"}
    for regime_id, regime in stochastic_regimes.regimes.items():
        if str(regime.transform_family).strip().lower() != "quantile_mapping_v1":
            raise RuntimeError(
                "Stochastic regime transform_family must be 'quantile_mapping_v1' in strict build "
                f"(regime={regime_id})."
            )
        mapping = regime.shock_quantile_mapping or {}
        missing_factors = [name for name in sorted(required_shock_factors) if not mapping.get(name)]
        if missing_factors:
            raise RuntimeError(
                "Stochastic regime shock_quantile_mapping missing required factors in strict build "
                f"(regime={regime_id}, missing={','.join(missing_factors)})."
            )
    (out_dir / "stochastic_regimes_uk_compiled.json").write_text(
        json.dumps(
            {
                "source": stochastic_regimes.source,
                "copula_id": stochastic_regimes.copula_id,
                "calibration_version": stochastic_regimes.calibration_version,
                "as_of_utc": stochastic_regimes.as_of_utc,
                "split_strategy": stochastic_regimes.split_strategy,
                "holdout_window": stochastic_regimes.holdout_window,
                "holdout_metrics": stochastic_regimes.holdout_metrics,
                "coverage_metrics": stochastic_regimes.coverage_metrics,
                "posterior_model": posterior_model,
                "regimes": {
                    regime_id: {
                        "sigma_scale": regime.sigma_scale,
                        "traffic_scale": regime.traffic_scale,
                        "incident_scale": regime.incident_scale,
                        "weather_scale": regime.weather_scale,
                        "price_scale": regime.price_scale,
                        "eco_scale": regime.eco_scale,
                        "spread_floor": regime.spread_floor,
                        "spread_cap": regime.spread_cap,
                        "factor_low": regime.factor_low,
                        "factor_high": regime.factor_high,
                        "duration_mix": list(regime.duration_mix),
                        "monetary_mix": list(regime.monetary_mix),
                        "emissions_mix": list(regime.emissions_mix),
                        "transform_family": regime.transform_family,
                        "shock_quantile_mapping": regime.shock_quantile_mapping,
                        "corr": regime.corr,
                    }
                    for regime_id, regime in stochastic_regimes.regimes.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    terrain_dir = out_dir / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    terrain_manifest = terrain_dir / "terrain_manifest.json"
    if force_rebuild_terrain or not _existing_terrain_valid(terrain_manifest):
        dem_glob_candidates = [
            str((out_dir / "dem_source" / "*.tif").resolve()),
            str((ROOT / "assets" / "uk" / "dem" / "*.tif").resolve()),
        ]
        dem_glob = ""
        for candidate in dem_glob_candidates:
            if glob.glob(candidate):
                dem_glob = candidate
                break
        if not dem_glob:
            dem_fetch_dir = out_dir / "dem_source"
            downloaded, requested, _failures = fetch_public_dem_tiles(
                output_dir=dem_fetch_dir,
                zoom=8,
                lat_min=49.75,
                lat_max=61.10,
                lon_min=-8.75,
                lon_max=2.25,
                concurrency=8,
                timeout_s=15.0,
            )
            if downloaded <= 0:
                raise FileNotFoundError(
                    "No DEM GeoTIFF files were available and public DEM bootstrap fetch returned zero tiles."
                )
            dem_glob = str((dem_fetch_dir / "*.tif").resolve())
        source_grid = ROOT / "assets" / "uk" / "terrain_dem_grid_uk.json"
        build_terrain_assets(
            source_dem_glob=dem_glob,
            source_grid=source_grid,
            output_dir=terrain_dir,
            output_root_dir=out_dir,
            version="uk_dem_v4",
            tile_size=1024,
            allow_synthetic_grid=False,
        )

    carbon_schedule_src = ROOT / "assets" / "uk" / "carbon_price_schedule_uk.json"
    if carbon_schedule_src.exists():
        (out_dir / "carbon_price_schedule_uk.json").write_text(
            carbon_schedule_src.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    carbon_intensity_src = ROOT / "assets" / "uk" / "carbon_intensity_hourly_uk.json"
    if carbon_intensity_src.exists():
        (out_dir / "carbon_intensity_hourly_uk.json").write_text(
            carbon_intensity_src.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    risk_norm_src = ROOT / "assets" / "uk" / "risk_normalization_refs_uk.json"
    if risk_norm_src.exists():
        (out_dir / "risk_normalization_refs_uk.json").write_text(
            risk_norm_src.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    toll_conf_cal_src = ROOT / "assets" / "uk" / "toll_confidence_calibration_uk.json"
    if toll_conf_cal_src.exists():
        (out_dir / "toll_confidence_calibration_uk.json").write_text(
            toll_conf_cal_src.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    terrain_asset_candidates = [
        "terrain/terrain_manifest.json",
        "terrain/tiles/uk_dem_main.tif",
    ]
    terrain_assets = [item for item in terrain_asset_candidates if (out_dir / item).exists()]

    manifest_assets = [
        "scenario_profiles_uk.json",
        "scenario_profiles_uk_compiled.json",
        "departure_profiles_uk.json",
        "departure_profile_uk_compiled.json",
        "toll_tariffs_uk_compiled.json",
        "toll_segments_seed_compiled.json",
        "toll_confidence_calibration_uk.json",
        "osm_toll_assets.geojson",
        "fuel_prices_uk_compiled.json",
        "fuel_consumption_surface_uk_compiled.json",
        "fuel_uncertainty_surface_uk_compiled.json",
        "carbon_intensity_hourly_uk.json",
        "risk_normalization_refs_uk.json",
        "stochastic_residual_priors_uk.json",
        "stochastic_regimes_uk_compiled.json",
        "stochastic_regimes_uk.json",
        "routing_graph_uk.json",
        *terrain_assets,
    ]
    checksums: dict[str, str] = {}
    for rel in manifest_assets:
        path = out_dir / rel
        if not path.exists():
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 256), b""):
                h.update(chunk)
        checksums[rel] = h.hexdigest()

    generated_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    signature_seed = json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hashlib.sha256(signature_seed).hexdigest()
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": "model-v2-uk",
                "source": "backend/scripts/build_model_assets.py",
                "generated_at_utc": generated_at_utc,
                "as_of_utc": generated_at_utc,
                "assets": manifest_assets,
                "checksums": checksums,
                "signature": signature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic backend model assets.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compatibility flag. Strict behavior is always enforced for this builder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("backend/out/model_assets"),
        help="Output directory for generated model assets",
    )
    parser.add_argument(
        "--departure-counts-csv",
        type=Path,
        default=None,
        help="Empirical departure profile counts CSV.",
    )
    parser.add_argument(
        "--stochastic-residuals-csv",
        type=Path,
        default=None,
        help="Residual-fit stochastic calibration CSV.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic profile/regime fallback generation.",
    )
    parser.add_argument(
        "--allow-geojson-routing-graph",
        action="store_true",
        help="Allow GeoJSON fallback when no OSM PBF/OSM source is available (test/dev only).",
    )
    parser.add_argument(
        "--routing-graph-source",
        type=Path,
        default=None,
        help="Source path for routing graph build (.pbf/.osm preferred; GeoJSON fallback supported).",
    )
    parser.add_argument(
        "--routing-graph-max-ways",
        type=int,
        default=0,
        help="Optional cap for routing graph way extraction (0 = no cap).",
    )
    parser.add_argument(
        "--force-rebuild-topology",
        action="store_true",
        help="Force rebuild of OSM toll topology even when an existing strict artifact is present.",
    )
    parser.add_argument(
        "--force-rebuild-graph",
        action="store_true",
        help="Force rebuild of routing graph even when an existing strict artifact is present.",
    )
    parser.add_argument(
        "--force-rebuild-terrain",
        action="store_true",
        help="Force rebuild of terrain tiles even when an existing strict artifact is present.",
    )
    args = parser.parse_args()
    build_assets(
        out_dir=args.out_dir,
        departure_counts_csv=args.departure_counts_csv,
        stochastic_residuals_csv=args.stochastic_residuals_csv,
        routing_graph_source=args.routing_graph_source,
        routing_graph_max_ways=max(0, int(args.routing_graph_max_ways)),
        allow_synthetic=bool(args.allow_synthetic),
        allow_geojson_routing_graph=bool(args.allow_geojson_routing_graph),
        force_rebuild_topology=bool(args.force_rebuild_topology),
        force_rebuild_graph=bool(args.force_rebuild_graph),
        force_rebuild_terrain=bool(args.force_rebuild_terrain),
    )
    print(f"Model assets generated at: {args.out_dir}")


if __name__ == "__main__":
    main()
