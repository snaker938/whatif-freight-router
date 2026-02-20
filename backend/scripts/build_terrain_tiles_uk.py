from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - this script is exercised in dev/CI build jobs
    import numpy as np
    import rasterio
    from rasterio.merge import merge as raster_merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    rasterio = None  # type: ignore[assignment]
    raster_merge = None  # type: ignore[assignment]
    calculate_default_transform = None  # type: ignore[assignment]
    reproject = None  # type: ignore[assignment]
    Resampling = None  # type: ignore[assignment]


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_grid(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("terrain source grid must be a JSON object")
    rows = int(payload.get("rows", 0))
    cols = int(payload.get("cols", 0))
    values = payload.get("values", [])
    if rows <= 1 or cols <= 1 or not isinstance(values, list) or len(values) != rows:
        raise ValueError("invalid synthetic grid dimensions")
    for row in values:
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError("invalid synthetic grid row")
    if float(payload.get("lat_step", 0.0)) <= 0.0 or float(payload.get("lon_step", 0.0)) <= 0.0:
        raise ValueError("lat_step/lon_step must be positive")
    return payload


def _write_manifest(
    *,
    output_dir: Path,
    output_root_dir: Path,
    version: str,
    source: str,
    bounds: dict[str, float],
    rel_tile_path_terrain_manifest: str,
    rel_tile_path_root_manifest: str,
    tile_format: str,
    tile_checksum: str,
    tile_id: str,
) -> tuple[Path, Path]:
    built_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    terrain_manifest = {
        "version": version,
        "generated_at_utc": built_at,
        "as_of_utc": built_at,
        "source": source,
        "bounds": bounds,
        "tiles": [
            {
                "id": tile_id,
                "path": rel_tile_path_terrain_manifest,
                "format": tile_format,
                "bounds": bounds,
            }
        ],
        "checksums": {
            rel_tile_path_terrain_manifest: tile_checksum,
        },
    }
    terrain_manifest_path = output_dir / "terrain_manifest.json"
    terrain_manifest_path.write_text(json.dumps(terrain_manifest, indent=2), encoding="utf-8")

    root_manifest = dict(terrain_manifest)
    root_manifest["tiles"] = [
        {
            "id": tile_id,
            "path": rel_tile_path_root_manifest,
            "format": tile_format,
            "bounds": bounds,
        }
    ]
    root_manifest["checksums"] = {rel_tile_path_root_manifest: tile_checksum}
    root_manifest_path = output_root_dir / "terrain_manifest.json"
    root_manifest_path.write_text(json.dumps(root_manifest, indent=2), encoding="utf-8")
    return terrain_manifest_path, root_manifest_path


def _build_from_geotiff_sources(
    *,
    dem_paths: list[Path],
    output_dir: Path,
    output_root_dir: Path,
    version: str,
    tile_size: int,
) -> tuple[Path, Path]:
    if (
        rasterio is None
        or raster_merge is None
        or np is None
        or calculate_default_transform is None
        or reproject is None
        or Resampling is None
    ):
        raise RuntimeError("rasterio is required to build GeoTIFF terrain assets")
    sources = [rasterio.open(path) for path in dem_paths]
    source_material = "|".join(sorted(path.name for path in dem_paths))
    source_fingerprint = hashlib.sha1(source_material.encode("utf-8")).hexdigest()[:12]
    try:
        mosaic, transform = raster_merge(sources)
        band = mosaic[0]
        src_crs = sources[0].crs
        if src_crs is None:
            raise RuntimeError("DEM source CRS is missing; cannot build terrain assets.")

        dst_crs = "EPSG:4326"
        if str(src_crs).upper() != dst_crs:
            src_height = int(band.shape[0])
            src_width = int(band.shape[1])
            src_bounds = rasterio.transform.array_bounds(src_height, src_width, transform)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs,
                dst_crs,
                src_width,
                src_height,
                *src_bounds,
            )
            reprojected = np.empty((dst_height, dst_width), dtype=band.dtype)
            reproject(
                source=band,
                destination=reprojected,
                src_transform=transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            band = reprojected
            transform = dst_transform
            output_crs = dst_crs
        else:
            output_crs = src_crs
        out_tiles_dir = output_dir / "tiles"
        out_tiles_dir.mkdir(parents=True, exist_ok=True)
        out_tif = out_tiles_dir / "uk_dem_main.tif"

        meta = sources[0].meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": int(band.shape[0]),
                "width": int(band.shape[1]),
                "transform": transform,
                "count": 1,
                "crs": output_crs,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": max(128, min(2048, tile_size)),
                "blockysize": max(128, min(2048, tile_size)),
            }
        )
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(band, 1)

        west, south, east, north = rasterio.transform.array_bounds(
            int(band.shape[0]),
            int(band.shape[1]),
            transform,
        )
        bounds = {
            "lat_min": float(south),
            "lat_max": float(north),
            "lon_min": float(west),
            "lon_max": float(east),
        }
        checksum = _checksum(out_tif)
        return _write_manifest(
            output_dir=output_dir,
            output_root_dir=output_root_dir,
            version=version,
            source=f"dem_geotiff_compiled_local:{source_fingerprint}",
            bounds=bounds,
            rel_tile_path_terrain_manifest="tiles/uk_dem_main.tif",
            rel_tile_path_root_manifest="terrain/tiles/uk_dem_main.tif",
            tile_format="geotiff",
            tile_checksum=checksum,
            tile_id="uk_dem_main",
        )
    finally:
        for src in sources:
            src.close()


def _build_from_synthetic_grid(
    *,
    source_grid: Path,
    output_dir: Path,
    output_root_dir: Path,
    version: str,
) -> tuple[Path, Path]:
    payload = _read_grid(source_grid)
    out_synth_dir = output_dir / "synthetic"
    out_synth_dir.mkdir(parents=True, exist_ok=True)
    out_grid = out_synth_dir / "uk_dem_grid_synthetic.json"
    out_grid_payload = {
        **payload,
        "version": version,
        "built_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_grid": str(source_grid),
        "synthetic": True,
    }
    out_grid.write_text(json.dumps(out_grid_payload, indent=2), encoding="utf-8")

    rows = int(payload["rows"])
    cols = int(payload["cols"])
    lat_min = float(payload["lat_min"])
    lon_min = float(payload["lon_min"])
    lat_step = float(payload["lat_step"])
    lon_step = float(payload["lon_step"])
    bounds = {
        "lat_min": lat_min,
        "lat_max": lat_min + (lat_step * (rows - 1)),
        "lon_min": lon_min,
        "lon_max": lon_min + (lon_step * (cols - 1)),
    }
    checksum = _checksum(out_grid)
    return _write_manifest(
        output_dir=output_dir,
        output_root_dir=output_root_dir,
        version=version,
        source="synthetic_grid_test_only",
        bounds=bounds,
        rel_tile_path_terrain_manifest="synthetic/uk_dem_grid_synthetic.json",
        rel_tile_path_root_manifest="terrain/synthetic/uk_dem_grid_synthetic.json",
        tile_format="grid_json",
        tile_checksum=checksum,
        tile_id="uk_dem_synthetic",
    )


def build_assets(
    *,
    source_dem_glob: str,
    source_grid: Path,
    output_dir: Path,
    output_root_dir: Path,
    version: str,
    tile_size: int,
    allow_synthetic_grid: bool,
) -> tuple[Path, Path]:
    test_only_synth = str(os.environ.get("TEST_ONLY_SYNTHETIC", "")).strip().lower() in {"1", "true", "yes"}
    if allow_synthetic_grid and not test_only_synth:
        raise RuntimeError(
            "Synthetic terrain grid generation is disabled in strict runtime. "
            "Set TEST_ONLY_SYNTHETIC=1 for explicit test-only generation."
        )
    if allow_synthetic_grid and str(os.environ.get("CI", "")).strip().lower() in {"1", "true", "yes"}:
        raise RuntimeError("Synthetic terrain grid generation is disabled in CI strict mode.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root_dir.mkdir(parents=True, exist_ok=True)
    dem_paths = [Path(path) for path in sorted(glob.glob(source_dem_glob))]
    if dem_paths:
        return _build_from_geotiff_sources(
            dem_paths=dem_paths,
            output_dir=output_dir,
            output_root_dir=output_root_dir,
            version=version,
            tile_size=tile_size,
        )
    if allow_synthetic_grid:
        return _build_from_synthetic_grid(
            source_grid=source_grid,
            output_dir=output_dir,
            output_root_dir=output_root_dir,
            version=version,
        )
    raise FileNotFoundError(
        f"No DEM sources matched --source-dem-glob='{source_dem_glob}'. "
        "Provide DEM GeoTIFF inputs or pass --allow-synthetic-grid for test-only assets."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UK terrain DEM runtime assets.")
    parser.add_argument(
        "--source-dem-glob",
        type=str,
        default="assets/uk/dem/*.tif",
    )
    parser.add_argument(
        "--source-grid",
        type=Path,
        default=Path("assets/uk/terrain_dem_grid_uk.json"),
    )
    parser.add_argument(
        "--allow-synthetic-grid",
        action="store_true",
        help="Allow fallback generation from synthetic grid JSON (test/dev only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/model_assets/terrain"),
    )
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=Path("out/model_assets"),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="uk_dem_v4",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
    )
    args = parser.parse_args()

    terrain_manifest_path, root_manifest_path = build_assets(
        source_dem_glob=args.source_dem_glob,
        source_grid=args.source_grid,
        output_dir=args.output_dir,
        output_root_dir=args.output_root_dir,
        version=args.version,
        tile_size=max(128, int(args.tile_size)),
        allow_synthetic_grid=bool(args.allow_synthetic_grid),
    )
    print(f"Wrote terrain manifest: {terrain_manifest_path}")
    print(f"Wrote root manifest: {root_manifest_path}")


if __name__ == "__main__":
    main()
