from __future__ import annotations

import json
import os
import pickle
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.routing_graph as routing_graph
import scripts.benchmark_model_v2 as benchmark_model_v2
import scripts.build_departure_profiles_uk as build_departure_profiles_uk
import scripts.build_model_assets as build_model_assets
import scripts.build_pricing_tables_uk as build_pricing_tables_uk
import scripts.build_route_graph_subset as build_route_graph_subset
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
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["version"] == "uk-routing-graph-v1"
    assert isinstance(payload["nodes"], list) and payload["nodes"]
    assert isinstance(payload["edges"], list) and payload["edges"]


def test_build_routing_graph_streams_payload_and_writes_compact_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "roads.geojson"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"highway": "primary"},
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
    compact_bundle_output = tmp_path / "routing_graph_uk.compact.pkl"

    original_dumps = build_routing_graph_uk.json.dumps

    def _guarded_dumps(value: object, *args: object, **kwargs: object) -> str:
        if (
            isinstance(value, dict)
            and isinstance(value.get("nodes"), list)
            and isinstance(value.get("edges"), list)
        ):
            raise AssertionError("graph build should stream the full payload instead of json.dumps on it")
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(build_routing_graph_uk.json, "dumps", _guarded_dumps)

    report = build_routing_graph_uk.build(
        source=source,
        output=output,
        compact_bundle_output=compact_bundle_output,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["version"] == "uk-routing-graph-v1"
    assert len(payload["nodes"]) == report["nodes"]
    assert len(payload["edges"]) == report["edges"]
    assert compact_bundle_output.exists()
    assert compact_bundle_output.with_suffix(".meta.json").exists()
    assert report["compact_bundle"] == str(compact_bundle_output)
    assert report["compact_bundle_edges"] >= report["edges"]


def test_build_routing_graph_corpus_bbox_derivation_applies_buffer(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "campaign.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon\n"
        "od-1,51.5000,-3.2000,54.9700,-1.6000\n"
        "od-2,51.8000,-4.0000,55.9000,-2.5000\n",
        encoding="utf-8",
    )

    bbox = build_routing_graph_uk._load_corpus_bbox(corpus_csv=corpus_csv, buffer_km=25.0)

    lat_min, lat_max, lon_min, lon_max = bbox
    assert lat_min < 51.5
    assert lat_max > 55.9
    assert lon_min < -4.0
    assert lon_max > -1.6


def test_build_routing_graph_main_uses_corpus_bbox_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "roads.geojson"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"highway": "primary"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-3.2000, 51.5000], [-1.6000, 54.9700]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus_csv = tmp_path / "campaign.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon\n"
        "od-1,51.5000,-3.2000,54.9700,-1.6000\n",
        encoding="utf-8",
    )
    output = tmp_path / "routing_graph_uk.json"
    observed: dict[str, object] = {}

    original_build = build_routing_graph_uk.build

    def _recording_build(**kwargs):
        observed["bbox"] = kwargs.get("bbox")
        return original_build(**kwargs)

    monkeypatch.setattr(build_routing_graph_uk, "build", _recording_build)
    argv = [
        "--source",
        str(source),
        "--output",
        str(output),
        "--od-corpus-csv",
        str(corpus_csv),
        "--bbox-buffer-km",
        "10",
    ]
    build_routing_graph_uk.main(argv)

    assert output.exists()
    assert observed["bbox"] is not None
    lat_min, lat_max, lon_min, lon_max = observed["bbox"]
    assert lat_min < 51.5
    assert lat_max > 54.97
    assert lon_min < -3.2
    assert lon_max > -1.6


def test_build_scenario_profiles_helpers_cover_core_behaviors() -> None:
    assert build_scenario_profiles_uk._safe_float("1.25") == pytest.approx(1.25)
    assert build_scenario_profiles_uk._safe_float("bad", 7.0) == pytest.approx(7.0)
    assert build_scenario_profiles_uk._iso_utc(
        build_scenario_profiles_uk._parse_as_of_utc("2026-02-01T00:00:00Z")
    ) == "2026-02-01T00:00:00Z"
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
    assert build_scenario_profiles_uk._dft_flow_index(3000.0) == pytest.approx(1.0)

    dft_backfilled = build_scenario_profiles_uk._parse_raw_line(
        {
            "corridor_bucket": "uk002",
            "hour_slot_local": 16,
            "road_mix_bucket": "mixed",
            "vehicle_class": "rigid_hgv",
            "day_kind": "weekday",
            "weather_bucket": "clear",
            "weather_regime": "clear",
            "as_of_utc": "2026-02-01T16:00:00Z",
            "traffic_features": {
                "flow_index": 0.0,
                "speed_index": 0.0,
                "dft_count_per_hour": 3300.0,
            },
            "incident_features": {"delay_pressure": 0.8, "severity_index": 0.9},
            "weather_features": {"weather_severity_index": 0.7},
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
    assert len(dft_backfilled) == 1
    assert dft_backfilled[0].flow_index is not None
    assert dft_backfilled[0].speed_inverse is not None
    assert dft_backfilled[0].traffic_pressure is not None

    selector, meta = build_scenario_profiles_uk._build_holdout_selector(obs)
    assert meta["strategy"] == "temporal_forward_plus_corridor_block"
    assert isinstance(selector(obs[0]), bool)


def test_build_scenario_profiles_uses_latest_observation_time_for_as_of(tmp_path: Path) -> None:
    raw_jsonl = tmp_path / "scenario_live_observed.jsonl"
    observed_modes_jsonl = tmp_path / "scenario_mode_outcomes_observed.jsonl"
    output_json = tmp_path / "scenario_profiles_uk.json"

    corridor_keys = [f"uk{idx:03d}" for idx in range(8)]
    hour_slots = [0, 4, 8, 12, 16, 20]
    weather_by_hour = {0: "clear", 4: "clear", 8: "rain", 12: "clear", 16: "rain", 20: "clear"}
    latest_observation = "2026-02-22T23:47:40Z"
    rows: list[dict[str, object]] = []
    for corridor_index, corridor in enumerate(corridor_keys):
        for hour_index, hour in enumerate(hour_slots):
            observed_as_of = (
                latest_observation
                if corridor_index == len(corridor_keys) - 1 and hour == hour_slots[-1]
                else f"2026-02-{20 + ((corridor_index + hour_index) % 3):02d}T{hour:02d}:00:00Z"
            )
            for mode_name, multiplier in (
                ("no_sharing", 1.22),
                ("partial_sharing", 1.08),
                ("full_sharing", 0.94),
            ):
                rows.append(
                    {
                        "corridor_bucket": corridor,
                        "corridor_geohash5": corridor,
                        "hour_slot_local": hour,
                        "road_mix_bucket": "mixed",
                        "road_mix_vector": {"mixed": 1.0},
                        "vehicle_class": "rigid_hgv",
                        "day_kind": "weekday" if hour < 20 else "weekend",
                        "weather_bucket": weather_by_hour[hour],
                        "weather_regime": weather_by_hour[hour],
                        "as_of_utc": observed_as_of,
                        "flow_index": 1.2 + (0.01 * corridor_index),
                        "speed_index": 1.0,
                        "delay_pressure": 0.8 + (0.01 * hour_index),
                        "severity_index": 0.9,
                        "weather_severity_index": 0.7 if weather_by_hour[hour] == "clear" else 1.2,
                        "mode": mode_name,
                        "mode_observation_source": "observed_telematics",
                        "duration_multiplier": multiplier,
                        "incident_rate_multiplier": multiplier,
                        "incident_delay_multiplier": multiplier,
                        "fuel_consumption_multiplier": multiplier,
                        "emissions_multiplier": multiplier,
                        "stochastic_sigma_multiplier": multiplier,
                    }
                )

    encoded_rows = "\n".join(json.dumps(row) for row in rows) + "\n"
    raw_jsonl.write_text(encoded_rows, encoding="utf-8")
    observed_modes_jsonl.write_text(encoded_rows, encoding="utf-8")

    payload = build_scenario_profiles_uk.build(
        raw_jsonl=raw_jsonl,
        observed_modes_jsonl=observed_modes_jsonl,
        output_json=output_json,
        min_contexts=8,
        min_observed_mode_row_share=0.2,
        max_projection_dominant_context_share=0.8,
    )

    assert output_json.exists()
    assert payload["as_of_utc"] == latest_observation
    assert payload["generated_at_utc"] >= payload["as_of_utc"]
    assert payload["source_observation_window"]["start_utc"] == "2026-02-20T00:00:00Z"
    assert payload["source_observation_window"]["end_utc"] == latest_observation
    assert payload["source_observation_window"]["row_count"] == len(rows) * 2
    assert payload["source_observation_window"]["observed_mode_row_count"] == len(rows) * 2
    assert payload["source_observation_filter"]["selected_row_count"] == len(rows) * 2
    assert payload["source_observation_filter"]["dropped_row_count"] == 0


def test_build_scenario_profiles_filters_to_recent_observation_window(tmp_path: Path) -> None:
    raw_jsonl = tmp_path / "scenario_live_observed.jsonl"
    observed_modes_jsonl = tmp_path / "scenario_mode_outcomes_observed.jsonl"
    output_json = tmp_path / "scenario_profiles_uk.json"

    corridor_keys = [f"uk{idx:03d}" for idx in range(8)]
    hour_slots = [0, 4, 8, 12, 16, 20]
    weather_by_hour = {0: "clear", 4: "clear", 8: "rain", 12: "clear", 16: "rain", 20: "clear"}

    stale_rows: list[dict[str, object]] = []
    fresh_rows: list[dict[str, object]] = []
    for corridor_index, corridor in enumerate(corridor_keys):
        for hour_index, hour in enumerate(hour_slots):
            stale_as_of = f"2026-02-01T{hour:02d}:00:00Z"
            fresh_as_of = (
                    "2026-02-22T23:47:40Z"
                    if corridor_index == len(corridor_keys) - 1 and hour == hour_slots[-1]
                    else f"2026-02-22T{hour:02d}:{(corridor_index + hour_index) % 50:02d}:00Z"
            )
            for target_rows, observed_as_of, no_multiplier in (
                (stale_rows, stale_as_of, 1.18),
                (fresh_rows, fresh_as_of, 1.24),
            ):
                for mode_name, multiplier in (
                    ("no_sharing", no_multiplier),
                    ("partial_sharing", no_multiplier - 0.14),
                    ("full_sharing", no_multiplier - 0.28),
                ):
                    target_rows.append(
                        {
                            "corridor_bucket": corridor,
                            "corridor_geohash5": corridor,
                            "hour_slot_local": hour,
                            "road_mix_bucket": "mixed",
                            "road_mix_vector": {"mixed": 1.0},
                            "vehicle_class": "rigid_hgv",
                            "day_kind": "weekday" if hour < 20 else "weekend",
                            "weather_bucket": weather_by_hour[hour],
                            "weather_regime": weather_by_hour[hour],
                            "as_of_utc": observed_as_of,
                            "flow_index": 0.0,
                            "speed_index": 0.0,
                            "dft_count_per_hour": 3200.0 + (40.0 * corridor_index) + (15.0 * hour_index),
                            "delay_pressure": 0.7 + (0.01 * hour_index),
                            "severity_index": 0.9,
                            "weather_severity_index": 0.7 if weather_by_hour[hour] == "clear" else 1.2,
                            "mode": mode_name,
                            "mode_observation_source": "observed_telematics",
                            "duration_multiplier": multiplier,
                            "incident_rate_multiplier": multiplier,
                            "incident_delay_multiplier": multiplier,
                            "fuel_consumption_multiplier": multiplier,
                            "emissions_multiplier": multiplier,
                            "stochastic_sigma_multiplier": multiplier,
                        }
                    )

    all_rows = stale_rows + fresh_rows
    encoded_rows = "\n".join(json.dumps(row) for row in all_rows) + "\n"
    raw_jsonl.write_text(encoded_rows, encoding="utf-8")
    observed_modes_jsonl.write_text(encoded_rows, encoding="utf-8")

    payload = build_scenario_profiles_uk.build(
        raw_jsonl=raw_jsonl,
        observed_modes_jsonl=observed_modes_jsonl,
        output_json=output_json,
        min_contexts=8,
        min_observed_mode_row_share=0.2,
        max_projection_dominant_context_share=0.8,
    )

    assert output_json.exists()
    assert payload["as_of_utc"] == "2026-02-22T23:47:40Z"
    assert payload["source_observation_window"]["start_utc"] == "2026-02-22T00:00:00Z"
    assert payload["source_observation_window"]["end_utc"] == "2026-02-22T23:47:40Z"
    assert payload["source_observation_window"]["row_count"] == len(fresh_rows) * 2
    assert payload["source_observation_window"]["observed_mode_row_count"] == len(fresh_rows) * 2
    assert payload["source_observation_filter"]["input_row_count"] == len(all_rows) * 2
    assert payload["source_observation_filter"]["selected_row_count"] == len(fresh_rows) * 2
    assert payload["source_observation_filter"]["dropped_row_count"] == len(stale_rows) * 2


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


def test_build_model_assets_existing_coverage_report_reuse_requires_fresh_matching_graph(tmp_path: Path) -> None:
    graph_output = tmp_path / "routing_graph_uk.json"
    graph_output.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    report_path = tmp_path / "routing_graph_coverage_report.json"
    report_payload = {
        "coverage_passed": True,
        "nodes": 12,
        "edges": 24,
        "graph_path": str(graph_output),
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    report_path.touch()

    reused = build_model_assets._existing_coverage_report_valid(
        report_path,
        graph_output=graph_output,
        min_nodes=10,
        min_edges=20,
    )
    assert reused == report_payload

    graph_output.write_text(json.dumps({"nodes": [1], "edges": [2]}), encoding="utf-8")
    os.utime(graph_output, (report_path.stat().st_mtime + 5.0, report_path.stat().st_mtime + 5.0))
    invalidated = build_model_assets._existing_coverage_report_valid(
        report_path,
        graph_output=graph_output,
        min_nodes=10,
        min_edges=20,
    )
    assert invalidated is None


def test_build_model_assets_prefers_valid_non_strict_scenario_corpus_when_strict_override_is_bad(
    tmp_path: Path,
) -> None:
    strict_path = tmp_path / "scenario_live_observed_strict.jsonl"
    default_path = tmp_path / "scenario_live_observed.jsonl"
    strict_path.write_text(
        json.dumps(
            {
                "source": "free_live_apis",
                "mode_observation_source": "empirical_outcome_bootstrap",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    default_path.write_text(
        json.dumps(
            {
                "source": "free_live_apis",
                "mode_observation_source": "empirical_outcome_public_feeds_v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    selected, summary = build_model_assets._preferred_validated_jsonl_source(
        strict_path=strict_path,
        default_path=default_path,
        label="Scenario live observed",
        source_policy="strict_external",
    )

    assert selected == default_path
    assert summary["selected_via"] == "default_fallback_after_invalid_strict_override"


def test_build_model_assets_rejects_proxy_toll_raw_inputs(tmp_path: Path) -> None:
    classification_dir = tmp_path / "classification"
    pricing_dir = tmp_path / "pricing"
    classification_dir.mkdir(parents=True, exist_ok=True)
    pricing_dir.mkdir(parents=True, exist_ok=True)
    (classification_dir / "class_0001.json").write_text(
        json.dumps({"source_provenance": "proxy_from_labeled_fixture_corpus_v1"}),
        encoding="utf-8",
    )
    (pricing_dir / "price_0001.json").write_text(
        json.dumps({"source_provenance": "proxy_from_labeled_fixture_corpus_v1"}),
        encoding="utf-8",
    )
    tariffs_path = tmp_path / "toll_tariffs_operator_truth.json"
    tariffs_path.write_text(
        json.dumps({"source": "proxy_from_labeled_toll_fixture_corpus_v1"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        build_model_assets._validate_toll_raw_provenance(
            classification_dir=classification_dir,
            pricing_dir=pricing_dir,
            tariffs_path=tariffs_path,
            source_policy="strict_external",
        )


def test_build_route_graph_subset_filters_by_corridor_not_global_bbox(tmp_path: Path) -> None:
    graph_json = tmp_path / "routing_graph.json"
    graph_json.write_text(
        json.dumps(
            {
                "version": "uk-routing-graph-v1",
                "source": "unit-test-source.osm.pbf",
                "generated_at_utc": "2026-04-05T00:00:00Z",
                "as_of_utc": "2026-04-05T00:00:00Z",
                "bbox": {"lat_min": 0.0, "lat_max": 2.0, "lon_min": 0.0, "lon_max": 2.0},
                "nodes": [
                    {"id": "a", "lat": 0.0, "lon": 0.0},
                    {"id": "b", "lat": 1.0, "lon": 1.0},
                    {"id": "c", "lat": 2.0, "lon": 2.0},
                    {"id": "d", "lat": 0.0, "lon": 2.0},
                ],
                "edges": [
                    {"u": "a", "v": "b", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                    {"u": "b", "v": "c", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                    {"u": "a", "v": "d", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon\n"
        "od-1,0.0,0.0,2.0,2.0\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "subset.json"

    report = build_route_graph_subset.build_subset(
        graph_json=graph_json,
        corpus_csv=corpus_csv,
        output_json=output_json,
        buffer_km=40.0,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    kept_ids = {str(node["id"]) for node in payload["nodes"]}
    kept_edges = {(str(edge["u"]), str(edge["v"])) for edge in payload["edges"]}
    assert kept_ids == {"a", "b", "c"}
    assert kept_edges == {("a", "b"), ("b", "c")}
    assert report["selection_mode"] == "corridor_union_segment_buffer"
    assert report["corridor_count"] == 1
    assert report["nodes_kept"] == 3
    assert report["edges_kept"] == 2


def test_build_route_graph_subset_supports_multiple_disjoint_corridors(tmp_path: Path) -> None:
    graph_json = tmp_path / "routing_graph.json"
    graph_json.write_text(
        json.dumps(
            {
                "version": "uk-routing-graph-v1",
                "source": "unit-test-source.osm.pbf",
                "generated_at_utc": "2026-04-05T00:00:00Z",
                "as_of_utc": "2026-04-05T00:00:00Z",
                "bbox": {"lat_min": 0.0, "lat_max": 4.0, "lon_min": 0.0, "lon_max": 4.0},
                "nodes": [
                    {"id": "a", "lat": 0.0, "lon": 0.0},
                    {"id": "b", "lat": 0.0, "lon": 1.0},
                    {"id": "c", "lat": 4.0, "lon": 4.0},
                    {"id": "d", "lat": 4.0, "lon": 3.0},
                    {"id": "x", "lat": 2.0, "lon": 2.0},
                ],
                "edges": [
                    {"u": "a", "v": "b", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                    {"u": "d", "v": "c", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                    {"u": "b", "v": "x", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon\n"
        "od-1,0.0,0.0,0.0,1.0\n"
        "od-2,4.0,3.0,4.0,4.0\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "subset.json"

    report = build_route_graph_subset.build_subset(
        graph_json=graph_json,
        corpus_csv=corpus_csv,
        output_json=output_json,
        buffer_km=20.0,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    kept_ids = {str(node["id"]) for node in payload["nodes"]}
    kept_edges = {(str(edge["u"]), str(edge["v"])) for edge in payload["edges"]}
    assert kept_ids == {"a", "b", "c", "d"}
    assert kept_edges == {("a", "b"), ("d", "c")}
    assert report["corridor_count"] == 2


def test_build_route_graph_subset_main_uses_explicit_corridor_km_and_cleans_staging(tmp_path: Path) -> None:
    graph_json = tmp_path / "routing_graph.json"
    graph_json.write_text(
        json.dumps(
            {
                "version": "uk-routing-graph-v1",
                "source": "unit-test-source.osm.pbf",
                "generated_at_utc": "2026-04-05T00:00:00Z",
                "as_of_utc": "2026-04-05T00:00:00Z",
                "bbox": {"lat_min": 0.0, "lat_max": 5.0, "lon_min": 0.0, "lon_max": 5.0},
                "nodes": [
                    {"id": "keep-1", "lat": 0.0, "lon": 0.0},
                    {"id": "keep-2", "lat": 0.3, "lon": 0.3},
                    {"id": "drop-1", "lat": 2.0, "lon": 2.0},
                ],
                "edges": [
                    {"u": "keep-1", "v": "keep-2", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                    {"u": "keep-2", "v": "drop-1", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon\n"
        "od-1,0.0,0.0,0.3,0.3\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "subset.json"

    exit_code = build_route_graph_subset.main(
        [
            "--graph-json",
            str(graph_json),
            "--corpus-csv",
            str(corpus_csv),
            "--output-json",
            str(output_json),
            "--corridor-km",
            "12",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    report = json.loads(output_json.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert {str(node["id"]) for node in payload["nodes"]} == {"keep-1", "keep-2"}
    assert {(str(edge["u"]), str(edge["v"])) for edge in payload["edges"]} == {("keep-1", "keep-2")}
    assert report["corridor_km"] == pytest.approx(12.0)
    assert report["filter_mode"] == "per_od_corridor_union"
    assert report["staging_mode"] == "node_jsonl_staging"
    assert Path(report["binary_cache"]).exists()
    assert Path(report["binary_cache_meta"]).exists()
    assert Path(report["compact_bundle"]).exists()
    assert Path(report["compact_bundle_meta"]).exists()
    assert report["compact_bundle_nodes"] == 2
    assert report["compact_bundle_edges"] >= 1
    assert output_json.with_suffix(".nodes.tmp.jsonl").exists() is False
    assert output_json.with_suffix(".nodes.tmp.sqlite3").exists() is False


def test_build_route_graph_subset_resumes_from_existing_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_json = tmp_path / "routing_graph.json"
    graph_json.write_text(
        json.dumps(
            {
                "version": "uk-routing-graph-v1",
                "source": "unit-test-source.osm.pbf",
                "generated_at_utc": "2026-04-05T00:00:00Z",
                "as_of_utc": "2026-04-05T00:00:00Z",
                "bbox": {"lat_min": 0.0, "lat_max": 5.0, "lon_min": 0.0, "lon_max": 5.0},
                "nodes": [],
                "edges": [
                    {"u": "keep-1", "v": "keep-2", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                    {"u": "keep-2", "v": "drop-1", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus_csv = tmp_path / "corpus.csv"
    corpus_csv.write_text(
        "od_id,origin_lat,origin_lon,destination_lat,destination_lon\n"
        "od-1,0.0,0.0,0.3,0.3\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "subset.json"
    output_json.write_text(
        json.dumps(
            {
                "version": "uk-routing-graph-v1",
                "source": "unit-test-source.osm.pbf#subset:corpus.csv",
                "generated_at_utc": "2026-04-05T00:00:00Z",
                "as_of_utc": "2026-04-05T00:00:00Z",
                "bbox": {"lat_min": 0.0, "lat_max": 1.0, "lon_min": 0.0, "lon_max": 1.0},
                "nodes": [
                    {"id": "keep-1", "lat": 0.0, "lon": 0.0},
                    {"id": "keep-2", "lat": 0.3, "lon": 0.3},
                ],
                "edges": [
                    {"u": "keep-1", "v": "keep-2", "distance_m": 1000.0, "generalized_cost": 1000.0, "oneway": False, "highway": "primary", "toll": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    node_staging_path = output_json.with_suffix(".nodes.tmp.jsonl")
    membership_db_path = output_json.with_suffix(".nodes.tmp.sqlite3")
    node_staging_path.write_text(
        "\n".join(
            (
                json.dumps({"id": "keep-1", "lat": 0.0, "lon": 0.0}, separators=(",", ":")),
                json.dumps({"id": "keep-2", "lat": 0.3, "lon": 0.3}, separators=(",", ":")),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    connection = sqlite3.connect(membership_db_path)
    try:
        connection.execute("CREATE TABLE kept_nodes (id TEXT PRIMARY KEY) WITHOUT ROWID")
        connection.executemany(
            "INSERT OR IGNORE INTO kept_nodes (id) VALUES (?)",
            [("keep-1",), ("keep-2",)],
        )
        connection.commit()
    finally:
        connection.close()

    original_load_staged_nodes = build_route_graph_subset._load_staged_nodes
    load_staged_nodes_calls = 0

    def _counting_load_staged_nodes(*args, **kwargs):
        nonlocal load_staged_nodes_calls
        load_staged_nodes_calls += 1
        return original_load_staged_nodes(*args, **kwargs)

    monkeypatch.setattr(build_route_graph_subset, "_load_staged_nodes", _counting_load_staged_nodes)

    report = build_route_graph_subset.build_subset(
        graph_json=graph_json,
        corpus_csv=corpus_csv,
        output_json=output_json,
        corridor_km=12.0,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["resumed_from_staging"] is True
    assert report["used_existing_output_json"] is True
    assert {str(node["id"]) for node in payload["nodes"]} == {"keep-1", "keep-2"}
    assert {(str(edge["u"]), str(edge["v"])) for edge in payload["edges"]} == {("keep-1", "keep-2")}
    assert Path(report["binary_cache"]).exists()
    assert Path(report["binary_cache_meta"]).exists()
    assert Path(report["compact_bundle"]).exists()
    assert Path(report["compact_bundle_meta"]).exists()
    with Path(report["compact_bundle"]).open("rb") as fh:
        compact_payload = pickle.load(fh)
    graph_payload = compact_payload["graph"]
    assert graph_payload["schema_version"] == 2
    assert isinstance(graph_payload["adjacency"]["keep-1"], tuple)
    assert isinstance(graph_payload["adjacency"]["keep-1"][0], routing_graph.GraphEdge)
    for key in (
        "grid_index",
        "component_by_node",
        "component_sizes",
        "component_count",
        "largest_component_nodes",
        "largest_component_ratio",
        "graph_fragmented",
    ):
        assert key in graph_payload
    rebuilt = routing_graph._load_route_graph_compact_bundle(output_json)
    assert rebuilt is not None
    assert rebuilt.grid_index
    assert rebuilt.component_by_node == {"keep-1": 1, "keep-2": 1}
    assert rebuilt.component_sizes == {1: 2}
    assert load_staged_nodes_calls == 1
    assert output_json.with_suffix(".nodes.tmp.jsonl").exists() is False
    assert output_json.with_suffix(".nodes.tmp.sqlite3").exists() is False
