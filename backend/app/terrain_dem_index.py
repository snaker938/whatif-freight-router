from __future__ import annotations

import json
import math
import os
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .settings import settings

try:  # pragma: no cover - exercised in integration where rasterio is installed
    import rasterio
except Exception:  # pragma: no cover - strict runtime check handles missing backend
    rasterio = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TerrainGridTile:
    tile_id: str
    path: str
    lat_min: float
    lat_step: float
    lon_min: float
    lon_step: float
    rows: int
    cols: int
    values: tuple[tuple[float, ...], ...]

    @property
    def lat_max(self) -> float:
        return self.lat_min + (self.lat_step * (self.rows - 1))

    @property
    def lon_max(self) -> float:
        return self.lon_min + (self.lon_step * (self.cols - 1))


@dataclass(frozen=True)
class TerrainRasterTile:
    tile_id: str
    path: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


TerrainTile = TerrainGridTile | TerrainRasterTile


@dataclass(frozen=True)
class TerrainManifest:
    version: str
    source: str
    tiles: tuple[TerrainTile, ...]


def _coerce_grid(payload: dict[str, object], *, tile_id: str, path: str) -> TerrainGridTile | None:
    rows = int(payload.get("rows", 0))
    cols = int(payload.get("cols", 0))
    values_raw = payload.get("values", [])
    if rows <= 1 or cols <= 1 or not isinstance(values_raw, list) or len(values_raw) != rows:
        return None
    rows_out: list[tuple[float, ...]] = []
    for row in values_raw:
        if not isinstance(row, list) or len(row) != cols:
            return None
        rows_out.append(tuple(float(v) for v in row))
    lat_step = float(payload.get("lat_step", 0.0))
    lon_step = float(payload.get("lon_step", 0.0))
    if lat_step <= 0 or lon_step <= 0:
        return None
    return TerrainGridTile(
        tile_id=tile_id,
        path=path,
        lat_min=float(payload.get("lat_min", 0.0)),
        lat_step=lat_step,
        lon_min=float(payload.get("lon_min", 0.0)),
        lon_step=lon_step,
        rows=rows,
        cols=cols,
        values=tuple(rows_out),
    )


def _coerce_raster_row(
    row: dict[str, object],
    *,
    tile_id: str,
    path: str,
) -> TerrainRasterTile | None:
    bounds = row.get("bounds", {})
    if not isinstance(bounds, dict):
        return None
    lat_min = float(bounds.get("lat_min", 0.0))
    lat_max = float(bounds.get("lat_max", 0.0))
    lon_min = float(bounds.get("lon_min", 0.0))
    lon_max = float(bounds.get("lon_max", 0.0))
    if lat_max <= lat_min or lon_max <= lon_min:
        return None
    tile_path = Path(path)
    if rasterio is not None:
        try:
            with rasterio.open(tile_path) as ds:
                if ds.width <= 1 or ds.height <= 1:
                    return None
                if ds.crs is None:
                    return None
                crs_text = str(ds.crs).strip().lower()
                if "4326" not in crs_text and "wgs84" not in crs_text:
                    return None
                ds_bounds = ds.bounds
                tolerance = 0.25
                if (
                    abs(float(ds_bounds.bottom) - lat_min) > tolerance
                    or abs(float(ds_bounds.top) - lat_max) > tolerance
                    or abs(float(ds_bounds.left) - lon_min) > tolerance
                    or abs(float(ds_bounds.right) - lon_max) > tolerance
                ):
                    return None
                # Force a tiny read to validate tile readability.
                _ = ds.read(1, window=((0, 1), (0, 1)))
        except Exception:
            return None
    return TerrainRasterTile(
        tile_id=tile_id,
        path=path,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


def _manifest_candidates() -> list[Path]:
    root = Path(settings.model_asset_dir)
    return [
        root / "terrain" / "terrain_manifest.json",
        root / "terrain_manifest.json",
    ]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_manifest_checksums(
    payload: dict[str, Any],
    *,
    manifest_dir: Path,
) -> bool:
    checksums = payload.get("checksums", {})
    if not isinstance(checksums, dict):
        return True
    for rel_path, expected in checksums.items():
        if not isinstance(rel_path, str) or not isinstance(expected, str):
            continue
        tile_path = _resolve_tile_path(manifest_dir=manifest_dir, rel_path=rel_path)
        if not tile_path.exists():
            return False
        actual = _sha256(tile_path)
        if actual.lower() != expected.strip().lower():
            return False
    return True


def _resolve_tile_path(*, manifest_dir: Path, rel_path: str) -> Path:
    path = manifest_dir / rel_path
    if path.exists():
        return path
    # Support manifests copied to model_asset root while tiles remain in terrain/.
    root_path = manifest_dir.parent / rel_path
    if root_path.exists():
        return root_path
    return path


def _tile_from_manifest_row(
    row: dict[str, object],
    *,
    manifest_dir: Path,
) -> TerrainTile | None:
    tile_id = str(row.get("id", "tile"))
    rel_path = str(row.get("path", "")).strip()
    if not rel_path:
        return None
    path = _resolve_tile_path(manifest_dir=manifest_dir, rel_path=rel_path)
    if not path.exists():
        return None

    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return _coerce_raster_row(row, tile_id=tile_id, path=str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return _coerce_grid(payload, tile_id=tile_id, path=str(path))


@lru_cache(maxsize=1)
def load_terrain_manifest() -> TerrainManifest:
    for manifest_path in _manifest_candidates():
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if not _verify_manifest_checksums(payload, manifest_dir=manifest_path.parent):
            raise RuntimeError(
                f"Terrain manifest checksum verification failed for '{manifest_path}'."
            )
        raw_tiles = payload.get("tiles", [])
        tiles: list[TerrainTile] = []
        if isinstance(raw_tiles, list):
            for raw_tile in raw_tiles:
                if not isinstance(raw_tile, dict):
                    continue
                tile = _tile_from_manifest_row(raw_tile, manifest_dir=manifest_path.parent)
                if tile is not None:
                    tiles.append(tile)
        if tiles:
            if (
                any(isinstance(tile, TerrainGridTile) for tile in tiles)
                and not settings.terrain_allow_synthetic_grid
                and "PYTEST_CURRENT_TEST" not in os.environ
            ):
                raise RuntimeError(
                    "Synthetic grid terrain assets are disabled in strict runtime. "
                    "Provide real DEM tiles or enable TERRAIN_ALLOW_SYNTHETIC_GRID for test-only flows."
                )
            return TerrainManifest(
                version=str(payload.get("version", "uk_dem_v3")),
                source=str(manifest_path),
                tiles=tuple(tiles),
            )
    return TerrainManifest(version="missing", source="none", tiles=())


@dataclass(frozen=True)
class _RasterSampleData:
    values: Any
    transform: Any
    nodata: float | None
    width: int
    height: int


def _raster_cache_slots() -> int:
    # Approximate one moderately sized tile per ~128MB.
    return max(1, int(settings.terrain_dem_tile_cache_max_mb) // 128)


def _load_raster_data_impl(path: str) -> _RasterSampleData:  # pragma: no cover - IO heavy, covered in integration
    if rasterio is None:
        raise RuntimeError("rasterio is required to sample GeoTIFF terrain assets")
    with rasterio.open(path) as ds:
        band = ds.read(1)
        return _RasterSampleData(
            values=band,
            transform=ds.transform,
            nodata=(float(ds.nodata) if ds.nodata is not None else None),
            width=int(ds.width),
            height=int(ds.height),
        )


def _sample_bilinear_grid(tile: TerrainGridTile, *, lat: float, lon: float) -> float:
    y = (lat - tile.lat_min) / tile.lat_step
    x = (lon - tile.lon_min) / tile.lon_step
    if x < 0 or y < 0 or x > (tile.cols - 1) or y > (tile.rows - 1):
        return math.nan

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(tile.cols - 1, x0 + 1)
    y1 = min(tile.rows - 1, y0 + 1)
    tx = x - x0
    ty = y - y0

    q11 = tile.values[y0][x0]
    q21 = tile.values[y0][x1]
    q12 = tile.values[y1][x0]
    q22 = tile.values[y1][x1]
    top = q11 + ((q21 - q11) * tx)
    bottom = q12 + ((q22 - q12) * tx)
    return top + ((bottom - top) * ty)


def _sample_bilinear_raster(tile: TerrainRasterTile, *, lat: float, lon: float) -> float:
    data = _load_raster_data(tile.path)
    col_f, row_f = (~data.transform) * (lon, lat)
    if col_f < 0 or row_f < 0 or col_f > (data.width - 1) or row_f > (data.height - 1):
        return math.nan

    c0 = int(math.floor(col_f))
    r0 = int(math.floor(row_f))
    c1 = min(data.width - 1, c0 + 1)
    r1 = min(data.height - 1, r0 + 1)
    tx = col_f - c0
    ty = row_f - r0

    q11 = float(data.values[r0, c0])
    q21 = float(data.values[r0, c1])
    q12 = float(data.values[r1, c0])
    q22 = float(data.values[r1, c1])

    if data.nodata is not None:
        nodata = float(data.nodata)
        if any(abs(v - nodata) <= 1e-6 for v in (q11, q21, q12, q22)):
            return math.nan

    top = q11 + ((q21 - q11) * tx)
    bottom = q12 + ((q22 - q12) * tx)
    return top + ((bottom - top) * ty)


def terrain_runtime_status() -> tuple[bool, str]:
    try:
        manifest = load_terrain_manifest()
    except RuntimeError:
        return False, "terrain_dem_asset_unavailable"
    if manifest.version == "missing" or not manifest.tiles:
        return False, "terrain_dem_asset_unavailable"
    if "synthetic" in str(manifest.source).lower() and "PYTEST_CURRENT_TEST" not in os.environ:
        return False, "terrain_dem_asset_unavailable"
    if any(isinstance(tile, TerrainRasterTile) for tile in manifest.tiles) and rasterio is None:
        return False, "terrain_dem_asset_unavailable"
    return True, "ok"


def sample_elevation_m(lat: float, lon: float) -> tuple[float, bool, str]:
    manifest = load_terrain_manifest()
    for tile in manifest.tiles:
        if lat < tile.lat_min or lat > tile.lat_max or lon < tile.lon_min or lon > tile.lon_max:
            continue
        if isinstance(tile, TerrainGridTile):
            sampled = _sample_bilinear_grid(tile, lat=lat, lon=lon)
        else:
            sampled = _sample_bilinear_raster(tile, lat=lat, lon=lon)
        if math.isfinite(sampled):
            return sampled, True, manifest.version
    return math.nan, False, manifest.version


# Bind runtime LRU cache size once settings are loaded.
_load_raster_data = lru_cache(maxsize=_raster_cache_slots())(_load_raster_data_impl)
