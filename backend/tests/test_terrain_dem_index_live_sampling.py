from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from affine import Affine
import numpy as np

from app.live_call_trace import get_trace, reset_trace, start_trace
from app.settings import settings
import app.terrain_dem_index as terrain_dem_index


def test_sample_bilinear_raster_path_projects_wgs84_to_dataset_crs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 2x2 raster where center interpolation should yield 25.0.
    sample_data = terrain_dem_index._RasterSampleData(
        values=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        transform=(Affine.translation(0.0, 2.0) * Affine.scale(1.0, -1.0)),
        nodata=None,
        width=2,
        height=2,
        crs="EPSG:3857",
    )
    monkeypatch.setattr(terrain_dem_index, "_load_raster_data", lambda _path: sample_data)
    monkeypatch.setattr(
        terrain_dem_index,
        "_rasterio_transform",
        lambda src, dst, xs, ys: ([0.5], [1.5]),
    )
    monkeypatch.setattr(
        terrain_dem_index,
        "rasterio",
        SimpleNamespace(warp=SimpleNamespace(transform=lambda *_args, **_kwargs: ([0.5], [1.5]))),
    )

    sampled = terrain_dem_index._sample_bilinear_raster_path(
        Path("dummy_live_tile.tif"),
        lat=53.208445,
        lon=-0.977783,
    )
    assert math.isfinite(sampled)
    assert sampled == pytest.approx(25.0, rel=1e-6, abs=1e-6)


def test_fetch_live_tile_path_deduplicates_route_cache_hit_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "dev_route_debug_console_enabled", True)
    monkeypatch.setattr(settings, "dev_route_debug_include_sensitive", True)
    monkeypatch.setattr(settings, "live_runtime_data_enabled", True)
    monkeypatch.setattr(settings, "strict_live_data_required", True)
    monkeypatch.setattr(settings, "live_terrain_enable_in_tests", True)
    monkeypatch.setattr(settings, "live_terrain_require_url_in_strict", True)
    monkeypatch.setattr(settings, "live_terrain_allow_signed_fallback", False)
    monkeypatch.setattr(settings, "live_terrain_cache_dir", str(tmp_path))
    monkeypatch.setattr(
        settings,
        "live_terrain_dem_url_template",
        "https://example.test/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif",
    )
    monkeypatch.setattr(settings, "live_terrain_allowed_hosts", "example.test")
    monkeypatch.setattr(settings, "live_terrain_fetch_retries", 1)
    monkeypatch.setattr(settings, "live_data_request_timeout_s", 1.0)
    monkeypatch.setattr(settings, "live_route_compute_force_no_cache_headers", False)

    request_id = "test-terrain-trace-dedup"
    token = start_trace(request_id, endpoint="/route", expected_calls=[])
    terrain_dem_index.set_terrain_live_fetcher_for_testing(
        lambda _url: (
            b"fake-dem-bytes",
            200,
            None,
            {"Content-Type": "image/tiff"},
            "image/tiff",
        )
    )
    try:
        terrain_dem_index.terrain_live_begin_route_run()
        first_path, first_err = terrain_dem_index._fetch_live_tile_path(z=8, x=126, y=84)
        second_path, second_err = terrain_dem_index._fetch_live_tile_path(z=8, x=126, y=84)
        third_path, third_err = terrain_dem_index._fetch_live_tile_path(z=8, x=126, y=84)

        assert first_err is None
        assert second_err is None
        assert third_err is None
        assert first_path is not None and first_path.exists()
        assert second_path == first_path
        assert third_path == first_path

        trace = get_trace(request_id)
        assert isinstance(trace, dict)
        observed = trace.get("observed_calls", [])
        assert isinstance(observed, list)
        terrain_rows = [
            row
            for row in observed
            if isinstance(row, dict) and str(row.get("source_key", "")) == "terrain_live_tile"
        ]
        route_cache_rows = [
            row
            for row in terrain_rows
            if isinstance(row.get("extra"), dict)
            and str((row.get("extra") or {}).get("served_from", "")) == "route_tile_cache"
        ]
        assert len(route_cache_rows) == 1

        diagnostics = terrain_dem_index.terrain_live_diagnostics_snapshot()
        assert int(diagnostics.get("cache_hits", 0)) == 2
        assert int(diagnostics.get("cache_hit_trace_suppressed", 0)) == 1
    finally:
        terrain_dem_index.set_terrain_live_fetcher_for_testing(None)
        reset_trace(token)
