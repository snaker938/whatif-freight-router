from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

TILE_URL_TEMPLATE = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"


def _deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    max_index = int(n) - 1
    return max(0, min(max_index, x)), max(0, min(max_index, y))


def _tile_coords_for_bbox(
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom: int,
) -> list[tuple[int, int]]:
    nw_x, nw_y = _deg2num(lat_max, lon_min, zoom)
    se_x, se_y = _deg2num(lat_min, lon_max, zoom)
    x_min, x_max = min(nw_x, se_x), max(nw_x, se_x)
    y_min, y_max = min(nw_y, se_y), max(nw_y, se_y)
    tiles: list[tuple[int, int]] = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((x, y))
    return tiles


def _download_tile(
    *,
    zoom: int,
    x: int,
    y: int,
    output_dir: Path,
    timeout_s: float,
) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{zoom}_{x}_{y}.tif"
    if path.exists() and path.stat().st_size > 0:
        return True, str(path)
    url = TILE_URL_TEMPLATE.format(z=zoom, x=x, y=y)
    try:
        with urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - fixed trusted host
            data = response.read()
        path.write_bytes(data)
        return True, str(path)
    except HTTPError as exc:
        return False, f"{url} -> HTTP {exc.code}"
    except URLError as exc:
        return False, f"{url} -> URL error: {exc.reason}"
    except OSError as exc:
        return False, f"{url} -> OS error: {exc}"


def fetch_tiles(
    *,
    output_dir: Path,
    zoom: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    concurrency: int,
    timeout_s: float,
) -> tuple[int, int, list[str]]:
    coords = _tile_coords_for_bbox(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        zoom=zoom,
    )
    if not coords:
        return 0, 0, ["No tile coordinates resolved from bbox."]
    failures: list[str] = []
    ok_count = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [
            executor.submit(
                _download_tile,
                zoom=zoom,
                x=x,
                y=y,
                output_dir=output_dir,
                timeout_s=timeout_s,
            )
            for x, y in coords
        ]
        for future in as_completed(futures):
            ok, message = future.result()
            if ok:
                ok_count += 1
            else:
                failures.append(message)

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": "elevation-tiles-prod-geotiff",
        "zoom": int(zoom),
        "requested_tiles": len(coords),
        "downloaded_tiles": ok_count,
        "bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "failures": failures,
    }
    (output_dir / "fetch_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return ok_count, len(coords), failures


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch UK DEM GeoTIFF tiles from public sources.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/out/model_assets/dem_source"),
    )
    parser.add_argument("--zoom", type=int, default=8)
    parser.add_argument("--lat-min", type=float, default=49.75)
    parser.add_argument("--lat-max", type=float, default=61.10)
    parser.add_argument("--lon-min", type=float, default=-8.75)
    parser.add_argument("--lon-max", type=float, default=2.25)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    ok_count, requested, failures = fetch_tiles(
        output_dir=args.output_dir,
        zoom=max(0, int(args.zoom)),
        lat_min=float(args.lat_min),
        lat_max=float(args.lat_max),
        lon_min=float(args.lon_min),
        lon_max=float(args.lon_max),
        concurrency=max(1, int(args.concurrency)),
        timeout_s=max(2.0, float(args.timeout_s)),
    )
    print(f"Downloaded {ok_count}/{requested} DEM tiles into {args.output_dir}")
    if failures:
        print(f"Tile download failures: {len(failures)}")


if __name__ == "__main__":
    main()
