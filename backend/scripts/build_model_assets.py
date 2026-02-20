from __future__ import annotations

import argparse
import glob
import json
import sys
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.calibration_loader import (
    load_departure_profile,
    load_fuel_price_snapshot,
    load_stochastic_regimes,
    load_uk_bank_holidays,
    load_toll_segments_seed,
    load_toll_tariffs,
)
from app.settings import settings
from build_routing_graph_uk import build as build_routing_graph
from build_terrain_tiles_uk import build_assets as build_terrain_assets
from build_departure_profiles_uk import build as build_departure_profiles
from build_stochastic_calibration_uk import build as build_stochastic_regimes
from build_pricing_tables_uk import build as build_pricing_tables
from extract_osm_tolls_uk import extract as extract_toll_assets
from fetch_public_dem_tiles_uk import fetch_tiles as fetch_public_dem_tiles
from fetch_dft_counts_uk import build as build_departure_counts_empirical
from fetch_fuel_history_uk import build as build_fuel_history
from fetch_carbon_intensity_uk import augment_carbon_schedule, build_intensity_asset
from fetch_stochastic_residuals_uk import build as build_stochastic_residuals
from fetch_toll_truth_uk import build as build_toll_truth_fixtures
from validate_graph_coverage import validate as validate_graph_coverage


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
    if _line_count(departure_counts_input) < 2000:
        raw_dft_counts = ROOT / "data" / "raw" / "uk" / "dft_counts_raw.csv"
        build_departure_counts_empirical(
            raw_csv=raw_dft_counts,
            output_csv=departure_counts_input,
            min_rows=2000,
        )
    if _line_count(departure_counts_input) < 2000:
        raise RuntimeError("Empirical departure counts corpus must contain at least 2000 rows.")

    if stochastic_residuals_input is None:
        stochastic_residuals_input = stochastic_residuals_default
    if _line_count(stochastic_residuals_input) < 5000:
        raw_stochastic = ROOT / "data" / "raw" / "uk" / "stochastic_residuals_raw.csv"
        build_stochastic_residuals(
            raw_csv=raw_stochastic,
            output_csv=stochastic_residuals_input,
            min_rows=5000,
        )
    if _line_count(stochastic_residuals_input) < 5000:
        raise RuntimeError("Empirical stochastic residual corpus must contain at least 5000 rows.")

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

    carbon_intensity_asset = ROOT / "assets" / "uk" / "carbon_intensity_hourly_uk.json"
    if not carbon_intensity_asset.exists():
        raw_intensity = ROOT / "data" / "raw" / "uk" / "carbon_intensity_hourly_raw.json"
        build_intensity_asset(source_json=raw_intensity, output_json=carbon_intensity_asset)
    augment_carbon_schedule(schedule_json=ROOT / "assets" / "uk" / "carbon_price_schedule_uk.json")

    toll_classification_fixtures = ROOT / "tests" / "fixtures" / "toll_classification"
    toll_pricing_fixtures = ROOT / "tests" / "fixtures" / "toll_pricing"
    if _fixture_count(toll_classification_fixtures) < 200 or _fixture_count(toll_pricing_fixtures) < 80:
        build_toll_truth_fixtures(
            classification_source_dir=ROOT / "data" / "raw" / "uk" / "toll_classification",
            pricing_source_dir=ROOT / "data" / "raw" / "uk" / "toll_pricing",
            classification_out_dir=toll_classification_fixtures,
            pricing_out_dir=toll_pricing_fixtures,
            classification_target=200,
            pricing_target=80,
            calibration_out_json=ROOT / "assets" / "uk" / "toll_confidence_calibration_uk.json",
        )
    if _fixture_count(toll_classification_fixtures) < 200:
        raise RuntimeError("Toll classification fixture corpus must include at least 200 labeled cases.")
    if _fixture_count(toll_pricing_fixtures) < 80:
        raise RuntimeError("Toll pricing fixture corpus must include at least 80 labeled cases.")

    # Build contextual departure + stochastic regime assets first so runtime loaders
    # consume compiled model assets rather than bundled defaults.
    build_departure_profiles(
        sparse_csv=ROOT / "assets" / "uk" / "departure_profile_uk.csv",
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
                "prices_gbp_per_l": fuel_snapshot.prices_gbp_per_l,
                "grid_price_gbp_per_kwh": fuel_snapshot.grid_price_gbp_per_kwh,
                "version": "uk-v2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stochastic_regimes = load_stochastic_regimes()
    (out_dir / "stochastic_regimes_uk_compiled.json").write_text(
        json.dumps(
            {
                "source": stochastic_regimes.source,
                "copula_id": stochastic_regimes.copula_id,
                "calibration_version": stochastic_regimes.calibration_version,
                "as_of_utc": stochastic_regimes.as_of_utc,
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
        "departure_profiles_uk.json",
        "departure_profile_uk_compiled.json",
        "toll_tariffs_uk_compiled.json",
        "toll_segments_seed_compiled.json",
        "toll_confidence_calibration_uk.json",
        "osm_toll_assets.geojson",
        "fuel_prices_uk_compiled.json",
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

    generated_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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
