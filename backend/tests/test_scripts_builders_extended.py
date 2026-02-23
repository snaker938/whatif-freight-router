from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.benchmark_model_v2 as benchmark_model_v2
import scripts.build_departure_profiles_uk as build_departure_profiles_uk
import scripts.build_model_assets as build_model_assets
import scripts.build_pricing_tables_uk as build_pricing_tables_uk
import scripts.build_routing_graph_uk as build_routing_graph_uk
import scripts.build_scenario_profiles_uk as build_scenario_profiles_uk
import scripts.build_stochastic_calibration_uk as build_stochastic_calibration_uk
import scripts.build_terrain_tiles_uk as build_terrain_tiles_uk


def test_benchmark_model_v2_load_fixture_routes_and_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "route_a.json").write_text(
        json.dumps(
            {
                "distance": 1000.0,
                "duration": 120.0,
                "geometry": {"type": "LineString", "coordinates": [[-1.5, 52.4], [-1.4, 52.3]]},
            }
        ),
        encoding="utf-8",
    )
    routes = benchmark_model_v2._load_fixture_routes(fixtures_dir)
    assert len(routes) == 1

    monkeypatch.setattr(benchmark_model_v2, "build_option", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        benchmark_model_v2,
        "route_graph_candidate_routes",
        lambda **kwargs: (
            [{"geometry": {"type": "LineString", "coordinates": [[-1.5, 52.4], [-1.4, 52.3]]}}],
            SimpleNamespace(explored_states=12, generated_paths=8, emitted_paths=3, candidate_budget=24),
        ),
    )
    report = benchmark_model_v2.benchmark(fixtures_dir=fixtures_dir, iterations=1, p95_gate_ms=2000.0)
    assert report["samples"] >= 1
    assert report["routes_per_iter"] == 1
    assert report["p95_gate_passed"] is True
    assert report["graph_explored_states_p95"] >= 0


def test_benchmark_model_v2_main_enforce_gate_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark_model_v2.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            fixtures_dir=Path("unused"),
            iterations=1,
            p95_gate_ms=100.0,
            enforce_gate=True,
        ),
    )
    monkeypatch.setattr(
        benchmark_model_v2,
        "benchmark",
        lambda **kwargs: {"p95_gate_passed": False, "samples": 1},
    )
    with pytest.raises(SystemExit):
        benchmark_model_v2.main()


def test_build_departure_profiles_interpolate_and_build_empirical(tmp_path: Path) -> None:
    dense = build_departure_profiles_uk._interpolate_sparse([(0, 1.0), (60, 1.5), (1439, 0.9)])
    assert len(dense) == 1440
    assert dense[0] == pytest.approx(1.0)
    assert dense[1439] == pytest.approx(0.9)

    sparse_csv = tmp_path / "sparse.csv"
    sparse_csv.write_text("minute,weekday,weekend,holiday\n0,1.0,0.9,0.95\n60,1.2,1.0,1.05\n", encoding="utf-8")
    counts_csv = tmp_path / "counts.csv"
    counts_csv.write_text(
        "region,road_bucket,day_kind,minute,multiplier,as_of_utc\n"
        "uk_default,mixed,weekday,0,1.00,2026-02-01T00:00:00Z\n"
        "uk_default,mixed,weekend,0,0.95,2026-02-01T00:00:00Z\n"
        "uk_default,mixed,holiday,0,0.92,2026-02-01T00:00:00Z\n"
        "uk_default,mixed,weekday,60,1.15,2026-02-01T00:00:00Z\n",
        encoding="utf-8",
    )
    out_json = tmp_path / "departure_profiles.json"
    build_departure_profiles_uk.build(
        sparse_csv=sparse_csv,
        counts_csv=counts_csv,
        output_json=out_json,
        allow_synthetic=False,
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["calibration_basis"] == "empirical"
    assert "uk_default" in payload["profiles"]
    assert "mixed" in payload["profiles"]["uk_default"]


def test_build_pricing_tables_normalize_and_write_outputs(tmp_path: Path) -> None:
    row = {
        "id": "r1",
        "operator": "op",
        "crossing_id": "x1",
        "road_class": "motorway",
        "direction": "both",
        "start_minute": 0,
        "end_minute": 1439,
        "crossing_fee_gbp": 2.5,
        "distance_fee_gbp_per_km": 0.1,
        "vehicle_classes": ["rigid_hgv"],
        "axle_classes": ["default"],
        "payment_classes": ["cash"],
        "exemptions": [],
    }
    normalized = build_pricing_tables_uk._normalize_rule(row)
    assert normalized["id"] == "r1"
    assert normalized["start_minute"] == 0

    fuel_source = tmp_path / "fuel.json"
    carbon_source = tmp_path / "carbon.json"
    tariff_source = tmp_path / "tariff.json"
    fuel_source.write_text(json.dumps({"fuel": "ok"}), encoding="utf-8")
    carbon_source.write_text(json.dumps({"carbon": "ok"}), encoding="utf-8")
    tariff_source.write_text(
        json.dumps(
            {
                "version": "v1",
                "source": "unit-test",
                "as_of_utc": "2026-02-01T00:00:00Z",
                "rules": [row],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    toll_tariffs = tmp_path / "toll_tariffs.json"
    build_pricing_tables_uk.build(
        fuel_source=fuel_source,
        carbon_source=carbon_source,
        tariff_truth_source=tariff_source,
        toll_tariffs_output=toll_tariffs,
        output_dir=out_dir,
        min_tariff_rules=1,
    )
    assert (out_dir / "fuel_prices_uk_compiled.json").exists()
    assert (out_dir / "carbon_price_schedule_uk.json").exists()
    assert toll_tariffs.exists()


def test_build_routing_graph_geojson_builds_nodes_edges_and_meta(tmp_path: Path) -> None:
    assert build_routing_graph_uk._parse_maxspeed("60 mph") == pytest.approx(96.5604, rel=1e-4)
    assert build_routing_graph_uk._direction_override({"oneway": "-1"}) == "reverse"

    source = tmp_path / "roads.geojson"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"highway": "primary", "toll": "yes"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-1.5000, 52.4000], [-1.4900, 52.3950], [-1.4800, 52.3900]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "routing_graph_uk.json"
    report = build_routing_graph_uk.build(source=source, output=output)
    assert report["nodes"] > 0
    assert report["edges"] > 0
    assert output.exists()
    assert output.with_suffix(".meta.json").exists()


def test_build_scenario_profiles_helpers_cover_core_behaviors() -> None:
    assert build_scenario_profiles_uk._safe_float("1.25") == pytest.approx(1.25)
    assert build_scenario_profiles_uk._safe_float("bad", 7.0) == pytest.approx(7.0)
    key = build_scenario_profiles_uk._context_key("uk001", 8, "mixed", "rigid_hgv", "weekday", "clear")
    assert key.startswith("uk001|h08|weekday|")
    p10, p50, p90 = build_scenario_profiles_uk._quantiles([0.8, 1.0, 1.2, 1.4])
    assert p10 <= p50 <= p90

    obs = build_scenario_profiles_uk._parse_raw_line(
        {
            "corridor_bucket": "uk001",
            "hour_slot_local": 12,
            "road_mix_bucket": "mixed",
            "vehicle_class": "rigid_hgv",
            "day_kind": "weekday",
            "weather_bucket": "clear",
            "as_of_utc": "2026-02-01T00:00:00Z",
            "mode": "no_sharing",
            "mode_observation_source": "observed_telematics",
            "duration_multiplier": 1.1,
            "incident_rate_multiplier": 1.0,
            "incident_delay_multiplier": 1.0,
            "fuel_consumption_multiplier": 1.0,
            "emissions_multiplier": 1.0,
            "stochastic_sigma_multiplier": 1.0,
        }
    )
    assert len(obs) == 1
    assert obs[0].mode == "no_sharing"
    assert obs[0].mode_is_projected is False

    selector, meta = build_scenario_profiles_uk._build_holdout_selector(obs)
    assert meta["strategy"] == "temporal_forward_plus_corridor_block"
    assert isinstance(selector(obs[0]), bool)


def test_build_stochastic_calibration_helpers_and_synthetic_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert build_stochastic_calibration_uk._canonical_slot("9") == "h09"
    assert (
        build_stochastic_calibration_uk._context_key(
            corridor_bucket="uk_default",
            day_kind="weekday",
            local_time_slot="8",
            road_bucket="mixed",
            weather_profile="clear",
            vehicle_type="rigid_hgv",
        )
        == "uk_default|weekday|h08|mixed|clear|rigid_hgv"
    )
    corr = build_stochastic_calibration_uk._corr_from_samples(
        [
            (1.0, 1.1, 0.9, 1.0, 1.0),
            (1.1, 1.0, 1.0, 0.9, 1.1),
            (0.9, 1.2, 1.1, 1.0, 1.0),
            (1.2, 0.9, 1.0, 1.1, 0.9),
            (1.0, 1.0, 1.2, 0.8, 1.1),
            (1.1, 1.1, 0.8, 1.2, 0.9),
            (0.8, 0.9, 1.1, 1.0, 1.2),
            (1.2, 1.0, 1.0, 1.1, 0.8),
        ]
    )
    assert len(corr) == 5
    assert all(len(row) == 5 for row in corr)
    assert all(corr[idx][idx] == pytest.approx(1.0) for idx in range(5))

    monkeypatch.setenv("TEST_ONLY_SYNTHETIC", "1")
    monkeypatch.delenv("CI", raising=False)
    output_json = tmp_path / "stochastic_regimes_uk.json"
    build_stochastic_calibration_uk.build(
        output_json=output_json,
        output_priors_json=None,
        residuals_csv=None,
        allow_synthetic=True,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    priors_payload = json.loads(
        output_json.with_name("stochastic_residual_priors_uk.json").read_text(encoding="utf-8")
    )
    assert payload["calibration_basis"] == "synthetic"
    assert isinstance(payload.get("regimes"), dict) and payload["regimes"]
    assert isinstance(priors_payload.get("priors"), list) and priors_payload["priors"]


def test_build_terrain_tiles_synthetic_build_and_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError):
        build_terrain_tiles_uk._coerce_positive_int(None, field_name="dst_width")

    source_grid = tmp_path / "grid.json"
    source_grid.write_text(
        json.dumps(
            {
                "rows": 3,
                "cols": 3,
                "lat_min": 52.0,
                "lon_min": -2.0,
                "lat_step": 0.1,
                "lon_step": 0.1,
                "values": [
                    [100.0, 101.0, 102.0],
                    [103.0, 104.0, 105.0],
                    [106.0, 107.0, 108.0],
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "terrain"
    output_root_dir = tmp_path / "root"
    monkeypatch.setenv("TEST_ONLY_SYNTHETIC", "1")
    monkeypatch.delenv("CI", raising=False)
    terrain_manifest, root_manifest = build_terrain_tiles_uk.build_assets(
        source_dem_glob=str(tmp_path / "dem" / "*.tif"),
        source_grid=source_grid,
        output_dir=output_dir,
        output_root_dir=output_root_dir,
        version="uk_dem_v_test",
        tile_size=512,
        allow_synthetic_grid=True,
    )
    assert terrain_manifest.exists()
    assert root_manifest.exists()
    manifest_payload = json.loads(terrain_manifest.read_text(encoding="utf-8"))
    assert manifest_payload["tiles"][0]["format"] == "grid_json"


def test_build_model_assets_helpers_and_main_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("a\n1\n2\n3\n", encoding="utf-8")
    assert build_model_assets._line_count(csv_path) == 4

    arr_path = tmp_path / "arr.json"
    arr_path.write_text(json.dumps({"items": [1, 2, 3]}), encoding="utf-8")
    assert build_model_assets._json_array_len(arr_path, "items") == 3
    assert build_model_assets._parse_iso_utc("2026-02-01T00:00:00Z") is not None

    terrain_dir = tmp_path / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    tile_path = terrain_dir / "tiles" / "a.tif"
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    tile_path.write_bytes(b"abc")
    manifest = terrain_dir / "terrain_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tiles": [
                    {"path": "tiles/a.tif"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert build_model_assets._existing_terrain_valid(manifest) is True

    captured: dict[str, object] = {}

    def _fake_build_assets(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(build_model_assets, "build_assets", _fake_build_assets)
    monkeypatch.setattr(
        build_model_assets.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            out_dir=Path("out/model_assets"),
            departure_counts_csv=Path("dep.csv"),
            stochastic_residuals_csv=Path("res.csv"),
            allow_synthetic=False,
            allow_geojson_routing_graph=False,
            routing_graph_source=Path("graph.pbf"),
            routing_graph_max_ways=250,
            force_rebuild_topology=True,
            force_rebuild_graph=False,
            force_rebuild_terrain=True,
        ),
    )
    build_model_assets.main()
    assert captured["routing_graph_max_ways"] == 250
    assert captured["force_rebuild_topology"] is True
    assert captured["force_rebuild_terrain"] is True
