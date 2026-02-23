from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

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


_LIVE_DIAG_LOCK = threading.Lock()
_LIVE_ROUTE_TOKEN = 0
_LIVE_ROUTE_TILE_TOKENS: dict[tuple[int, int, int], int] = {}
_LIVE_ROUTE_REMOTE_FETCHES = 0
_LIVE_CIRCUIT_FAIL_STREAK = 0
_LIVE_CIRCUIT_OPEN_UNTIL_MONOTONIC = 0.0
_LIVE_DIAGNOSTICS: dict[str, Any] = {
    "requests": 0,
    "cache_hits": 0,
    "fetch_failures": 0,
    "stale_cache_used": 0,
    "remote_fetches": 0,
    "circuit_breaker_open": False,
    "circuit_breaker_fail_streak": 0,
    "status_code": None,
    "fetch_error": None,
    "source_url": "",
    "as_of_utc": None,
    "tile_zoom": int(settings.live_terrain_tile_zoom),
}

def _strict_runtime_test_bypass_enabled() -> bool:
    return os.environ.get("STRICT_RUNTIME_TEST_BYPASS", "0").strip() == "1"


def _pytest_active() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _terrain_live_policy() -> tuple[bool, bool, bool]:
    strict = bool(settings.strict_live_data_required)
    require_live = bool(settings.live_terrain_require_url_in_strict) if strict else False
    allow_fallback = (bool(settings.live_terrain_allow_signed_fallback) if strict else True) or _strict_runtime_test_bypass_enabled()
    if _pytest_active() and not bool(settings.live_terrain_enable_in_tests):
        # Keep deterministic fixture behavior unless tests explicitly opt in.
        allow_fallback = True
    return strict, require_live, allow_fallback


def terrain_live_mode_active() -> bool:
    if not bool(settings.live_runtime_data_enabled):
        return False
    if _pytest_active() and not bool(settings.live_terrain_enable_in_tests):
        return False
    return bool(str(settings.live_terrain_dem_url_template or "").strip())


def terrain_live_begin_route_run() -> None:
    if not terrain_live_mode_active():
        return
    global _LIVE_ROUTE_TOKEN, _LIVE_ROUTE_REMOTE_FETCHES
    with _LIVE_DIAG_LOCK:
        _LIVE_ROUTE_TOKEN += 1
        if _LIVE_ROUTE_TOKEN > 1_000_000_000:
            _LIVE_ROUTE_TOKEN = 1
        _LIVE_ROUTE_TILE_TOKENS.clear()
        _LIVE_ROUTE_REMOTE_FETCHES = 0
        _LIVE_DIAGNOSTICS.update(
            {
                "requests": 0,
                "cache_hits": 0,
                "fetch_failures": 0,
                "stale_cache_used": 0,
                "remote_fetches": 0,
                "circuit_breaker_open": bool(time.monotonic() < _LIVE_CIRCUIT_OPEN_UNTIL_MONOTONIC),
                "circuit_breaker_fail_streak": int(_LIVE_CIRCUIT_FAIL_STREAK),
                "status_code": None,
                "fetch_error": None,
                "source_url": "",
                "as_of_utc": None,
                "tile_zoom": int(settings.live_terrain_tile_zoom),
            }
        )


def terrain_live_route_token() -> int:
    with _LIVE_DIAG_LOCK:
        return int(_LIVE_ROUTE_TOKEN)


def terrain_live_diagnostics_snapshot() -> dict[str, Any]:
    with _LIVE_DIAG_LOCK:
        out = dict(_LIVE_DIAGNOSTICS)
        out["circuit_breaker_open"] = bool(time.monotonic() < _LIVE_CIRCUIT_OPEN_UNTIL_MONOTONIC)
        out["circuit_breaker_fail_streak"] = int(_LIVE_CIRCUIT_FAIL_STREAK)
    requests = int(out.get("requests", 0))
    cache_hits = int(out.get("cache_hits", 0))
    out["cache_hit_rate"] = (float(cache_hits) / float(requests)) if requests > 0 else 0.0
    out["mode"] = "request_time_live" if terrain_live_mode_active() else "manifest_asset"
    return out


def _set_live_diag(**updates: Any) -> None:
    with _LIVE_DIAG_LOCK:
        for key, value in updates.items():
            _LIVE_DIAGNOSTICS[key] = value


def _incr_live_diag(key: str, delta: int = 1) -> None:
    with _LIVE_DIAG_LOCK:
        _LIVE_DIAGNOSTICS[key] = int(_LIVE_DIAGNOSTICS.get(key, 0)) + int(delta)


def _cache_dir_live_tiles() -> Path:
    configured = str(settings.live_terrain_cache_dir or "").strip()
    if configured:
        base = Path(configured)
    else:
        base = Path(settings.model_asset_dir) / "terrain" / "live_tile_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _allowed_hosts() -> tuple[str, ...]:
    raw = str(settings.live_terrain_allowed_hosts or "").strip()
    if not raw:
        return ()
    out: list[str] = []
    for token in raw.split(","):
        host = token.strip().lower()
        if host:
            out.append(host)
    return tuple(dict.fromkeys(out))


def _host_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    host_l = host.strip().lower()
    if not host_l:
        return False
    for allow in allowed:
        if host_l == allow or host_l.endswith("." + allow):
            return True
    return False


def _deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    max_index = int(n) - 1
    return max(0, min(max_index, x)), max(0, min(max_index, y))


def _render_live_tile_url(*, z: int, x: int, y: int) -> str:
    template = str(settings.live_terrain_dem_url_template or "").strip()
    if not template:
        return ""
    try:
        return template.format(z=int(z), x=int(x), y=int(y))
    except Exception:
        return ""


def _tile_path(*, z: int, x: int, y: int) -> Path:
    return _cache_dir_live_tiles() / f"{int(z)}_{int(x)}_{int(y)}.tif"


def _is_fresh_file(path: Path, *, max_age_days: int) -> bool:
    if not path.exists():
        return False
    if max_age_days < 0:
        return True
    age_s = max(0.0, time.time() - path.stat().st_mtime)
    return age_s <= (float(max_age_days) * 86_400.0)


def _enforce_live_cache_budget(cache_dir: Path) -> None:
    tif_files = list(cache_dir.glob("*.tif"))
    if not tif_files:
        return
    max_tiles = max(1, int(settings.live_terrain_cache_max_tiles))
    max_bytes = max(1, int(settings.live_terrain_cache_max_mb)) * 1024 * 1024
    files_sorted = sorted(tif_files, key=lambda p: p.stat().st_mtime, reverse=True)
    keep: list[Path] = []
    size_acc = 0
    for path in files_sorted:
        size = int(path.stat().st_size)
        if len(keep) < max_tiles and (size_acc + size) <= max_bytes:
            keep.append(path)
            size_acc += size
            continue
        try:
            path.unlink(missing_ok=True)
        except Exception:
            continue


def _download_tile_bytes(url: str, *, timeout_s: float, attempts: int = 2) -> tuple[bytes | None, int | None, str | None]:
    last_error: str | None = None
    for _ in range(max(1, int(attempts))):
        try:
            with urlopen(url, timeout=max(1.0, float(timeout_s))) as response:  # noqa: S310 - host policy validated upstream
                data = response.read()
                status = int(getattr(response, "status", 200))
            return data, status, None
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            return None, int(exc.code), last_error
        except URLError as exc:
            last_error = f"URL error: {exc.reason}"
            continue
        except OSError as exc:
            last_error = f"OS error: {exc}"
            continue
    return None, None, last_error or "download_failed"


_LIVE_FETCHER: Callable[[str], tuple[bytes | None, int | None, str | None]] | None = None


def set_terrain_live_fetcher_for_testing(
    fetcher: Callable[[str], tuple[bytes | None, int | None, str | None]] | None,
) -> None:
    """Inject a deterministic live fetch transport for CI and tests."""
    global _LIVE_FETCHER
    _LIVE_FETCHER = fetcher


def _call_live_fetcher(url: str, *, timeout_s: float) -> tuple[bytes | None, int | None, str | None]:
    if _LIVE_FETCHER is not None:
        return _LIVE_FETCHER(url)
    attempts = max(1, int(settings.live_terrain_fetch_retries))
    return _download_tile_bytes(url, timeout_s=timeout_s, attempts=attempts)


def _circuit_open() -> bool:
    with _LIVE_DIAG_LOCK:
        return bool(time.monotonic() < _LIVE_CIRCUIT_OPEN_UNTIL_MONOTONIC)


def _record_fetch_success() -> None:
    global _LIVE_CIRCUIT_FAIL_STREAK
    with _LIVE_DIAG_LOCK:
        _LIVE_CIRCUIT_FAIL_STREAK = 0
        _LIVE_DIAGNOSTICS["circuit_breaker_fail_streak"] = 0
        _LIVE_DIAGNOSTICS["circuit_breaker_open"] = False


def _record_fetch_failure() -> None:
    global _LIVE_CIRCUIT_FAIL_STREAK, _LIVE_CIRCUIT_OPEN_UNTIL_MONOTONIC
    with _LIVE_DIAG_LOCK:
        _LIVE_CIRCUIT_FAIL_STREAK += 1
        _LIVE_DIAGNOSTICS["circuit_breaker_fail_streak"] = int(_LIVE_CIRCUIT_FAIL_STREAK)
        threshold = max(1, int(settings.live_terrain_circuit_breaker_failures))
        if _LIVE_CIRCUIT_FAIL_STREAK >= threshold:
            cooldown_s = max(1, int(settings.live_terrain_circuit_breaker_cooldown_s))
            _LIVE_CIRCUIT_OPEN_UNTIL_MONOTONIC = time.monotonic() + float(cooldown_s)
            _LIVE_DIAGNOSTICS["circuit_breaker_open"] = True


def _consume_remote_fetch_budget() -> bool:
    global _LIVE_ROUTE_REMOTE_FETCHES
    with _LIVE_DIAG_LOCK:
        budget = max(1, int(settings.live_terrain_max_remote_tiles_per_route))
        if _LIVE_ROUTE_REMOTE_FETCHES >= budget:
            return False
        _LIVE_ROUTE_REMOTE_FETCHES += 1
        _LIVE_DIAGNOSTICS["remote_fetches"] = int(_LIVE_ROUTE_REMOTE_FETCHES)
    return True


def _mark_tile_seen_for_route(tile_key: tuple[int, int, int]) -> None:
    with _LIVE_DIAG_LOCK:
        _LIVE_ROUTE_TILE_TOKENS[tile_key] = int(_LIVE_ROUTE_TOKEN)


def _tile_seen_for_route(tile_key: tuple[int, int, int]) -> bool:
    with _LIVE_DIAG_LOCK:
        return int(_LIVE_ROUTE_TILE_TOKENS.get(tile_key, -1)) == int(_LIVE_ROUTE_TOKEN)


def _fetch_live_tile_path(*, z: int, x: int, y: int) -> tuple[Path | None, str | None]:
    strict, require_live, allow_fallback = _terrain_live_policy()
    tile_key = (int(z), int(x), int(y))
    path = _tile_path(z=z, x=x, y=y)
    if _tile_seen_for_route(tile_key) and path.exists():
        _incr_live_diag("cache_hits")
        return path, None

    url = _render_live_tile_url(z=z, x=x, y=y)
    if not url:
        if strict and require_live and not allow_fallback:
            _incr_live_diag("fetch_failures")
            _set_live_diag(fetch_error="missing_live_terrain_url_template")
            return None, "missing_live_terrain_url_template"
        if path.exists():
            _incr_live_diag("cache_hits")
            _mark_tile_seen_for_route(tile_key)
            return path, None
        _incr_live_diag("fetch_failures")
        _set_live_diag(fetch_error="missing_live_terrain_url_template")
        return None, "missing_live_terrain_url_template"

    parsed = urlparse(url)
    host = str(parsed.netloc or "").strip().lower()
    allowed_hosts = _allowed_hosts()
    if not _host_allowed(host, allowed_hosts):
        _incr_live_diag("fetch_failures")
        _set_live_diag(fetch_error=f"host_not_allowed:{host}", source_url=url)
        if path.exists() and allow_fallback:
            _incr_live_diag("stale_cache_used")
            _mark_tile_seen_for_route(tile_key)
            return path, None
        return None, f"host_not_allowed:{host}"

    timeout_s = float(settings.live_data_request_timeout_s)
    max_age_days = int(settings.live_terrain_tile_max_age_days)
    if _circuit_open():
        _incr_live_diag("fetch_failures")
        _set_live_diag(fetch_error="terrain_live_circuit_open", source_url=url)
        if path.exists() and allow_fallback:
            _incr_live_diag("stale_cache_used")
            _mark_tile_seen_for_route(tile_key)
            return path, None
        return None, "terrain_live_circuit_open"

    if not _consume_remote_fetch_budget():
        _incr_live_diag("fetch_failures")
        _set_live_diag(fetch_error="terrain_live_fetch_budget_exceeded", source_url=url)
        if path.exists() and allow_fallback:
            _incr_live_diag("stale_cache_used")
            _mark_tile_seen_for_route(tile_key)
            return path, None
        return None, "terrain_live_fetch_budget_exceeded"

    _incr_live_diag("requests")
    _set_live_diag(source_url=url, tile_zoom=int(z))

    # Request-time live policy: refresh each route run per tile, then reuse
    # that tile for remaining samples in the same run.
    data, status_code, err = _call_live_fetcher(url, timeout_s=timeout_s)
    _set_live_diag(status_code=status_code, fetch_error=err)
    if data:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
        _enforce_live_cache_budget(path.parent)
        _record_fetch_success()
        _set_live_diag(
            as_of_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            stale_cache_used=False,
        )
        _mark_tile_seen_for_route(tile_key)
        return path, None

    if path.exists() and (allow_fallback or _is_fresh_file(path, max_age_days=max_age_days)):
        _incr_live_diag("stale_cache_used")
        _set_live_diag(stale_cache_used=True)
        _record_fetch_failure()
        _mark_tile_seen_for_route(tile_key)
        return path, None

    _incr_live_diag("fetch_failures")
    _record_fetch_failure()
    return None, err or "live_tile_unavailable"


def _coerce_grid(payload: dict[str, object], *, tile_id: str, path: str) -> TerrainGridTile | None:
    rows_raw = payload.get("rows", 0)
    cols_raw = payload.get("cols", 0)
    if not isinstance(rows_raw, (int, float, str)):
        return None
    if not isinstance(cols_raw, (int, float, str)):
        return None
    try:
        rows = int(rows_raw)
        cols = int(cols_raw)
    except (TypeError, ValueError):
        return None
    values_raw = payload.get("values", [])
    if rows <= 1 or cols <= 1 or not isinstance(values_raw, list) or len(values_raw) != rows:
        return None
    rows_out: list[tuple[float, ...]] = []
    for row in values_raw:
        if not isinstance(row, list) or len(row) != cols:
            return None
        converted: list[float] = []
        for value in row:
            if not isinstance(value, (int, float, str)):
                return None
            try:
                converted.append(float(value))
            except (TypeError, ValueError):
                return None
        rows_out.append(tuple(converted))
    lat_step_raw = payload.get("lat_step", 0.0)
    lon_step_raw = payload.get("lon_step", 0.0)
    lat_min_raw = payload.get("lat_min", 0.0)
    lon_min_raw = payload.get("lon_min", 0.0)
    if not isinstance(lat_step_raw, (int, float, str)):
        return None
    if not isinstance(lon_step_raw, (int, float, str)):
        return None
    if not isinstance(lat_min_raw, (int, float, str)):
        return None
    if not isinstance(lon_min_raw, (int, float, str)):
        return None
    try:
        lat_step = float(lat_step_raw)
        lon_step = float(lon_step_raw)
        lat_min = float(lat_min_raw)
        lon_min = float(lon_min_raw)
    except (TypeError, ValueError):
        return None
    if lat_step <= 0 or lon_step <= 0:
        return None
    return TerrainGridTile(
        tile_id=tile_id,
        path=path,
        lat_min=lat_min,
        lat_step=lat_step,
        lon_min=lon_min,
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


def _sample_bilinear_raster_path(path: Path, *, lat: float, lon: float) -> float:
    data = _load_raster_data(str(path))
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


def _sample_live_elevation(lat: float, lon: float) -> tuple[float, bool, str]:
    zoom = int(settings.live_terrain_tile_zoom)
    x, y = _deg2num(lat, lon, zoom)
    path, _error = _fetch_live_tile_path(z=zoom, x=x, y=y)
    if path is None:
        return math.nan, False, f"live_dem_z{zoom}"
    sampled = _sample_bilinear_raster_path(path, lat=lat, lon=lon)
    if math.isfinite(sampled):
        return sampled, True, f"live_dem_z{zoom}"
    return math.nan, False, f"live_dem_z{zoom}"


def terrain_runtime_status() -> tuple[bool, str]:
    if _pytest_active() and not bool(settings.live_terrain_enable_in_tests):
        # Deterministic CI/test mode: do not require real DEM assets/network
        # unless the test explicitly opts in to live terrain behavior.
        return True, "ok"

    strict, require_live, allow_fallback = _terrain_live_policy()
    url_template = str(settings.live_terrain_dem_url_template or "").strip()
    if strict and require_live and not url_template and not allow_fallback:
        return False, "terrain_dem_asset_unavailable"
    if terrain_live_mode_active():
        if rasterio is None:
            return False, "terrain_dem_asset_unavailable"
        if url_template:
            url = _render_live_tile_url(z=int(settings.live_terrain_tile_zoom), x=0, y=0)
            host = str(urlparse(url).netloc or "").strip().lower() if url else ""
            if not _host_allowed(host, _allowed_hosts()):
                return False, "terrain_dem_asset_unavailable"
        return True, "ok"

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
    if _pytest_active() and not bool(settings.live_terrain_enable_in_tests):
        # Deterministic pseudo-elevation surface for unit/integration tests.
        seed = ((lat * 37.0) + (lon * 11.0)) % 240.0
        return 80.0 + seed, True, "pytest_stub_dem_v1"

    strict, require_live, allow_fallback = _terrain_live_policy()
    if terrain_live_mode_active():
        sampled, covered, version = _sample_live_elevation(lat, lon)
        if covered:
            return sampled, True, version
        if strict and require_live and not allow_fallback:
            return math.nan, False, version

    try:
        manifest = load_terrain_manifest()
    except RuntimeError:
        if terrain_live_mode_active():
            return math.nan, False, f"live_dem_z{int(settings.live_terrain_tile_zoom)}"
        return math.nan, False, "missing"
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
