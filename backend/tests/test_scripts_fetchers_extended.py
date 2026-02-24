from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import scripts.collect_carbon_intensity_raw_uk as collect_carbon_intensity_raw_uk
import scripts.collect_dft_raw_counts_uk as collect_dft_raw_counts_uk
import scripts.collect_fuel_history_raw_uk as collect_fuel_history_raw_uk
import scripts.collect_scenario_mode_outcomes_proxy_uk as collect_scenario_mode_outcomes_proxy_uk
import scripts.collect_stochastic_residuals_raw_uk as collect_stochastic_residuals_raw_uk
import scripts.collect_toll_truth_raw_uk as collect_toll_truth_raw_uk
import scripts.extract_osm_tolls_uk as extract_osm_tolls_uk
import scripts.fetch_carbon_intensity_uk as fetch_carbon_intensity_uk
import scripts.fetch_dft_counts_uk as fetch_dft_counts_uk
import scripts.fetch_fuel_history_uk as fetch_fuel_history_uk
import scripts.fetch_public_dem_tiles_uk as fetch_public_dem_tiles_uk
import scripts.fetch_scenario_live_uk as fetch_scenario_live_uk
import scripts.fetch_stochastic_residuals_uk as fetch_stochastic_residuals_uk
import scripts.fetch_toll_truth_uk as fetch_toll_truth_uk


def test_extract_osm_tolls_geojson_and_xml(tmp_path: Path) -> None:
    assert extract_osm_tolls_uk._is_tolled_props({"name": "Toll Bridge"}) is True
    assert extract_osm_tolls_uk._is_tolled_props({"name": "Normal Road", "toll": "no"}) is False

    source_geojson = tmp_path / "source.geojson"
    source_geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "A Toll Crossing", "toll": "yes"},
                        "geometry": {"type": "LineString", "coordinates": [[-1.0, 52.0], [-1.1, 52.1]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Free Road", "toll": "no"},
                        "geometry": {"type": "LineString", "coordinates": [[-2.0, 53.0], [-2.1, 53.1]]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    filtered = extract_osm_tolls_uk._extract_from_geojson(source_geojson=source_geojson)
    assert len(filtered) == 1
    out_geojson = tmp_path / "out.geojson"
    extract_osm_tolls_uk.extract(source_geojson=source_geojson, output_geojson=out_geojson)
    out_payload = json.loads(out_geojson.read_text(encoding="utf-8"))
    assert len(out_payload["features"]) == 1

    source_osm = tmp_path / "sample.osm"
    source_osm.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="52.0000" lon="-1.0000" />
  <node id="2" lat="52.0100" lon="-1.0100" />
  <way id="100">
    <nd ref="1" />
    <nd ref="2" />
    <tag k="name" v="Test Toll Way" />
    <tag k="toll" v="yes" />
    <tag k="highway" v="primary" />
  </way>
</osm>""",
        encoding="utf-8",
    )
    features = extract_osm_tolls_uk._extract_from_osm_xml(source_osm=source_osm)
    assert features
    assert features[0]["geometry"]["type"] == "LineString"


def test_fetch_carbon_intensity_build_and_schedule_augment(tmp_path: Path) -> None:
    source = tmp_path / "intensity_raw.json"
    source.write_text(
        json.dumps(
            {
                "source": "unit_test",
                "as_of_utc": "2026-02-01T00:00:00Z",
                "regions": {"uk_default": [0.2 for _ in range(24)], "london": [0.22 for _ in range(24)]},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "intensity_out.json"
    built = fetch_carbon_intensity_uk.build_intensity_asset(source_json=source, output_json=output)
    assert output.exists()
    assert built["version"] == "uk-carbon-intensity-v3"
    assert "uk_default" in built["regions"]

    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                "source": "policy",
                "prices_gbp_per_kg": {
                    "central": {"2026": 0.11, "2027": 0.12},
                    "low": {"2026": 0.09, "2027": 0.1},
                    "high": {"2026": 0.13, "2027": 0.14},
                },
            }
        ),
        encoding="utf-8",
    )
    augmented = fetch_carbon_intensity_uk.augment_carbon_schedule(schedule_json=schedule_path)
    assert "uncertainty_distribution_by_year" in augmented
    assert "2026" in augmented["uncertainty_distribution_by_year"]


def test_fetch_dft_counts_builds_contextual_rows(tmp_path: Path) -> None:
    as_of = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    raw_csv = tmp_path / "dft_raw.csv"
    raw_csv.write_text(
        "region,road_bucket,day_kind,minute,multiplier,as_of_utc\n"
        f"london,motorway,weekday,0,1.10,{as_of}\n"
        f"london,motorway,weekday,60,1.20,{as_of}\n"
        f"wales,primary,weekend,120,0.95,{as_of}\n"
        f"wales,primary,holiday,180,0.90,{as_of}\n"
        f"scotland,urban_local,weekday,240,1.05,{as_of}\n"
        f"scotland,urban_local,weekend,300,1.00,{as_of}\n",
        encoding="utf-8",
    )
    output = tmp_path / "departure_counts_empirical.csv"
    rows = fetch_dft_counts_uk.build(
        raw_csv=raw_csv,
        output_csv=output,
        as_of_utc=as_of,
        min_rows=3,
        min_unique_regions=2,
        min_unique_road_buckets=2,
        min_unique_hours=3,
        max_age_days=3650,
    )
    assert rows >= 6
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "region,road_bucket,day_kind,minute,multiplier,as_of_utc" in content


def test_fetch_fuel_history_builds_signed_payload(tmp_path: Path) -> None:
    source = tmp_path / "fuel_raw.json"
    source.write_text(
        json.dumps(
            {
                "source": "observed_market_feed",
                "as_of": "2026-02-03T00:00:00Z",
                "regional_multipliers": {"uk_default": 1.0, "london_southeast": 1.1},
                "history": [
                    {
                        "as_of": "2026-02-01",
                        "prices_gbp_per_l": {"diesel": 1.55, "petrol": 1.62, "lng": 1.10},
                        "grid_price_gbp_per_kwh": 0.28,
                    },
                    {
                        "as_of": "2026-02-02",
                        "prices_gbp_per_l": {"diesel": 1.56, "petrol": 1.63, "lng": 1.09},
                        "grid_price_gbp_per_kwh": 0.29,
                    },
                    {
                        "as_of": "2026-02-03",
                        "prices_gbp_per_l": {"diesel": 1.57, "petrol": 1.64, "lng": 1.08},
                        "grid_price_gbp_per_kwh": 0.30,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "fuel_prices_uk.json"
    payload = fetch_fuel_history_uk.build(source_json=source, output_json=output, min_history_days=2)
    assert output.exists()
    assert payload["signed"] is True
    assert payload["signature_algorithm"] == "sha256"
    assert len(payload["history"]) == 3


def test_fetch_public_dem_tiles_fetch_tiles_with_stubbed_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "tiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        fetch_public_dem_tiles_uk,
        "_download_tile",
        lambda **kwargs: (True, str(Path(kwargs["output_dir"]) / f"{kwargs['zoom']}_{kwargs['x']}_{kwargs['y']}.tif")),
    )
    ok_count, requested, failures = fetch_public_dem_tiles_uk.fetch_tiles(
        output_dir=output_dir,
        zoom=2,
        lat_min=51.0,
        lat_max=52.0,
        lon_min=-1.0,
        lon_max=0.0,
        concurrency=2,
        timeout_s=5.0,
    )
    assert requested >= 1
    assert ok_count == requested
    assert failures == []
    assert (output_dir / "fetch_manifest.json").exists()


def test_fetch_scenario_live_load_match_and_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed_jsonl = tmp_path / "observed_modes.jsonl"
    observed_jsonl.write_text(
        json.dumps(
            {
                "corridor_bucket": "uk_default",
                "day_kind": "weekday",
                "hour_slot_local": 12,
                "road_mix_bucket": "mixed",
                "vehicle_class": "rigid_hgv",
                "weather_bucket": "clear",
                "as_of_utc": "2026-02-01T12:00:00Z",
                "mode_observation_source": "observed_telematics",
                "modes": {
                    "no_sharing": {
                        "duration_multiplier": 1.2,
                        "incident_rate_multiplier": 1.1,
                        "incident_delay_multiplier": 1.1,
                        "fuel_consumption_multiplier": 1.05,
                        "emissions_multiplier": 1.03,
                        "stochastic_sigma_multiplier": 1.08,
                    },
                    "partial_sharing": {
                        "duration_multiplier": 1.1,
                        "incident_rate_multiplier": 1.0,
                        "incident_delay_multiplier": 1.0,
                        "fuel_consumption_multiplier": 1.0,
                        "emissions_multiplier": 0.98,
                        "stochastic_sigma_multiplier": 1.0,
                    },
                    "full_sharing": {
                        "duration_multiplier": 1.0,
                        "incident_rate_multiplier": 0.95,
                        "incident_delay_multiplier": 0.95,
                        "fuel_consumption_multiplier": 0.95,
                        "emissions_multiplier": 0.94,
                        "stochastic_sigma_multiplier": 0.95,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outcomes = fetch_scenario_live_uk._load_observed_mode_outcomes(observed_jsonl)
    assert len(outcomes) == 1
    matched = fetch_scenario_live_uk._match_observed_mode_outcome(
        outcomes=outcomes,
        corridor_bucket="uk_default",
        road_mix_bucket="mixed",
        vehicle_class="rigid_hgv",
        day_kind="weekday",
        weather_bucket="clear",
        hour_slot_local=12,
    )
    assert matched is not None
    assert fetch_scenario_live_uk._parse_hour_slots("1,2,25,-1") == [0, 1, 2, 23]

    monkeypatch.setattr(
        fetch_scenario_live_uk,
        "live_scenario_context",
        lambda route_context, allow_partial_sources=False: {
            "coverage": {"overall": 1.0},
            "source_coverage": {"webtris": 1.0, "traffic_england": 1.0, "dft": 1.0, "open_meteo": 1.0},
            "as_of_utc": "2026-02-01T12:00:00Z",
        },
    )
    output = tmp_path / "scenario_snapshot.json"
    snapshot = fetch_scenario_live_uk.build_snapshot(
        output_json=output,
        corridor_bucket="uk_default",
        road_mix_bucket="mixed",
        vehicle_class="rigid_hgv",
        day_kind="weekday",
        weather_bucket="clear",
        centroid_lat=54.2,
        centroid_lon=-2.3,
        road_hint="A1",
        hour_slot_local=12,
        observed_mode_outcomes=outcomes,
        require_observed_modes=True,
    )
    assert output.exists()
    assert snapshot["mode_is_projected"] is False
    assert set(snapshot["modes"].keys()) == {"no_sharing", "partial_sharing", "full_sharing"}


def test_fetch_scenario_live_mode_source_projection_classifier() -> None:
    assert fetch_scenario_live_uk._mode_source_is_projected("empirical_proxy_public_feeds_v1") is True
    assert fetch_scenario_live_uk._mode_source_is_projected("bootstrap_replay_v1") is True
    assert fetch_scenario_live_uk._mode_source_is_projected("resample_snapshot_v1") is True
    assert fetch_scenario_live_uk._mode_source_is_projected("observed_telematics") is False


def test_collect_dft_fetch_page_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(collect_dft_raw_counts_uk.time, "sleep", lambda seconds: sleep_calls.append(float(seconds)))

    class _Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
            _ = args, kwargs
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return _Response({"data": [{"id": 1}]})

    payload = collect_dft_raw_counts_uk._fetch_page(
        client=_FlakyClient(),
        base_url="https://example.test/raw-counts",
        year=2026,
        page_number=1,
        page_size=10,
        timeout_s=2.0,
        retries=3,
        backoff_s=0.1,
    )
    assert payload["data"] == [{"id": 1}]
    assert len(sleep_calls) == 1


def test_collect_dft_raw_collect_paginates_dedupes_and_writes_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_csv = tmp_path / "dft_counts_raw.csv"
    state_file = tmp_path / "dft_counts_raw.state.json"
    monkeypatch.setattr(
        collect_dft_raw_counts_uk,
        "_load_holiday_dates",
        lambda **kwargs: {"2026-01-01"},
    )

    rows_page_1 = [
        {
            "id": "1",
            "count_point_id": "100",
            "count_date": "2026-01-01",
            "hour": 8,
            "direction_of_travel": "N",
            "region_id": 3,
            "local_authority_id": 11,
            "road_name": "M1",
            "road_category": "M",
            "road_type": "major",
            "latitude": 52.4,
            "longitude": -1.9,
            "all_motor_vehicles": 1200,
            "all_hgvs": 140,
            "cars_and_taxis": 1000,
            "lgvs": 160,
            "buses_and_coaches": 20,
            "pedal_cycles": 5,
            "two_wheeled_motor_vehicles": 12,
        }
    ]
    rows_page_2 = [
        rows_page_1[0],
        {
            "id": "2",
            "count_point_id": "200",
            "count_date": "2026-01-02",
            "hour": 9,
            "direction_of_travel": "S",
            "region_id": 11,
            "local_authority_id": 22,
            "road_name": "A55",
            "road_category": "PA",
            "road_type": "primary",
            "latitude": 53.3,
            "longitude": -3.5,
            "all_motor_vehicles": 700,
            "all_hgvs": 80,
            "cars_and_taxis": 560,
            "lgvs": 90,
            "buses_and_coaches": 8,
            "pedal_cycles": 3,
            "two_wheeled_motor_vehicles": 7,
        },
    ]

    def _fake_fetch_page(**kwargs) -> dict[str, object]:
        page = int(kwargs["page_number"])
        if page == 1:
            return {"data": rows_page_1, "next_page_url": "next", "last_page": 2}
        if page == 2:
            return {"data": rows_page_2, "next_page_url": None, "last_page": 2}
        return {"data": []}

    monkeypatch.setattr(collect_dft_raw_counts_uk, "_fetch_page", _fake_fetch_page)
    summary = collect_dft_raw_counts_uk.collect(
        output_csv=output_csv,
        years=[2026],
        base_url="https://example.test/raw-counts",
        max_pages_per_year=5,
        page_size=50,
        retries=1,
        backoff_s=0.01,
        timeout_s=2.0,
        target_min_rows=2,
        append_safe=True,
        resume=True,
        state_file=state_file,
        bank_holidays_url="https://example.test/bank-holidays.json",
    )
    assert summary["rows_written"] == 2
    assert summary["pages_fetched"] == 2
    assert output_csv.exists()
    assert state_file.exists()

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["day_kind"] == "holiday"
    assert rows[0]["corridor_bucket"] == "london_southeast"
    assert rows[1]["corridor_bucket"] == "wales_west"


def test_collect_stochastic_residuals_raw_builds_target_rows(tmp_path: Path) -> None:
    scenario_jsonl = tmp_path / "scenario_live_observed.jsonl"
    scenario_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "corridor_bucket": "uk_default",
                        "day_kind": "weekday",
                        "hour_slot_local": 8,
                        "road_mix_bucket": "mixed",
                        "weather_bucket": "clear",
                        "traffic_features": {"flow_index": 120.0, "speed_index": 58.0},
                        "incident_features": {"delay_pressure": 0.8, "severity_index": 0.4},
                        "weather_features": {"weather_severity_index": 1.1},
                    }
                ),
                json.dumps(
                    {
                        "corridor_bucket": "wales_west",
                        "day_kind": "weekend",
                        "hour_slot_local": 16,
                        "road_mix_bucket": "primary_heavy",
                        "weather_bucket": "rain",
                        "traffic_features": {"flow_index": 90.0, "speed_index": 66.0},
                        "incident_features": {"delay_pressure": 0.5, "severity_index": 0.3},
                        "weather_features": {"weather_severity_index": 1.3},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dft_raw_csv = tmp_path / "dft_counts_raw.csv"
    dft_raw_csv.write_text(
        "dedupe_key,corridor_bucket,day_kind,hour,road_bucket,road_category,all_motor_vehicles\n"
        "k1,uk_default,weekday,8,mixed,m,1300\n"
        "k2,wales_west,weekend,16,primary_heavy,pa,700\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "stochastic_residuals_raw.csv"
    summary = collect_stochastic_residuals_raw_uk.build(
        scenario_jsonl=scenario_jsonl,
        dft_raw_csv=dft_raw_csv,
        output_csv=output_csv,
        target_min_rows=12,
        variants_per_row=2,
    )
    assert summary["rows_written"] == 12
    assert summary["diversity"]["vehicle_type_count"] >= 2
    assert output_csv.exists()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {"actual_duration_s", "expected_duration_s", "regime_id", "source"} <= set(rows[0].keys())


def test_collect_fuel_history_raw_builds_payload_and_rejects_synthetic(tmp_path: Path) -> None:
    source_a = tmp_path / "fuel_a.json"
    source_b = tmp_path / "fuel_b.json"
    source_a.write_text(
        json.dumps(
            {
                "source": "public_feed_a",
                "regional_multipliers": {"uk_default": 1.0, "london": 1.1},
                "history": [
                    {
                        "as_of": "2026-02-01",
                        "prices_gbp_per_l": {"diesel": 1.50, "petrol": 1.58, "lng": 1.04},
                        "grid_price_gbp_per_kwh": 0.28,
                    },
                    {
                        "as_of": "2026-02-02",
                        "prices_gbp_per_l": {"diesel": 1.52, "petrol": 1.60, "lng": 1.05},
                        "grid_price_gbp_per_kwh": 0.29,
                    },
                    {
                        "as_of": "2026-02-03",
                        "prices_gbp_per_l": {"diesel": 1.53, "petrol": 1.61, "lng": 1.06},
                        "grid_price_gbp_per_kwh": 0.30,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    source_b.write_text(
        json.dumps(
            {
                "source": "public_feed_b",
                "history": [
                    {
                        "as_of": "2026-02-01",
                        "prices_gbp_per_l": {"diesel": 1.51, "petrol": 1.59, "lng": 1.03},
                        "grid_price_gbp_per_kwh": 0.27,
                    },
                    {
                        "as_of": "2026-02-02",
                        "prices_gbp_per_l": {"diesel": 1.53, "petrol": 1.61, "lng": 1.06},
                        "grid_price_gbp_per_kwh": 0.28,
                    },
                    {
                        "as_of": "2026-02-03",
                        "prices_gbp_per_l": {"diesel": 1.54, "petrol": 1.62, "lng": 1.07},
                        "grid_price_gbp_per_kwh": 0.29,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "fuel_prices_raw.json"
    summary = collect_fuel_history_raw_uk.build(
        output_json=output_json,
        source_urls=[],
        source_jsons=[source_a, source_b],
        min_days=3,
        timeout_s=2.0,
    )
    assert summary["rows"] == 3
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["source"] == "public_fuel_history_uk"
    assert len(payload["history"]) == 3
    assert "uk_default" in payload["regional_multipliers"]

    synthetic_source = tmp_path / "fuel_synthetic.json"
    synthetic_source.write_text(
        json.dumps(
            {
                "source": "synthetic_generator",
                "history": [
                    {
                        "as_of": "2026-02-01",
                        "prices_gbp_per_l": {"diesel": 1.2, "petrol": 1.3, "lng": 0.9},
                        "grid_price_gbp_per_kwh": 0.2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        collect_fuel_history_raw_uk.build(
            output_json=tmp_path / "should_fail.json",
            source_urls=[],
            source_jsons=[synthetic_source],
            min_days=1,
            timeout_s=2.0,
        )


def test_collect_carbon_intensity_raw_builds_profiles_with_stubbed_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_fetch_json(*, client, url: str):  # noqa: ANN001
        _ = client
        if "/regional/intensity/" in url:
            return {
                "data": {
                    "data": [
                        {"from": "2026-02-01T00:00Z", "intensity": {"actual": 190}},
                        {"from": "2026-02-01T01:00Z", "intensity": {"forecast": 210}},
                    ]
                }
            }
        return {
            "data": [
                {"from": "2026-02-01T00:00Z", "intensity": {"actual": 200}},
                {"from": "2026-02-01T01:00Z", "intensity": {"actual": 220}},
            ]
        }

    monkeypatch.setattr(collect_carbon_intensity_raw_uk, "_fetch_json", _fake_fetch_json)
    output_json = tmp_path / "carbon_intensity_hourly_raw.json"
    summary = collect_carbon_intensity_raw_uk.build(
        output_json=output_json,
        regions=["uk_default", "london"],
        window_days=7,
        timeout_s=2.0,
    )
    assert summary["region_count"] >= 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert "uk_default" in payload["regions"]
    assert "london" in payload["regions"]
    assert len(payload["regions"]["uk_default"]) == 24


def test_collect_toll_truth_raw_builds_proxy_corpora(tmp_path: Path) -> None:
    classification_source = tmp_path / "classification_source"
    pricing_source = tmp_path / "pricing_source"
    classification_source.mkdir(parents=True, exist_ok=True)
    pricing_source.mkdir(parents=True, exist_ok=True)

    for idx in range(20):
        is_toll = idx % 2 == 0
        (classification_source / f"class_{idx:03d}.json").write_text(
            json.dumps(
                {
                    "fixture_id": f"class_{idx:03d}",
                    "route_fixture": f"route_{idx:03d}.json",
                    "expected_has_toll": is_toll,
                }
            ),
            encoding="utf-8",
        )
    for idx in range(4):
        is_toll = idx % 2 == 0
        (pricing_source / f"price_{idx:03d}.json").write_text(
            json.dumps(
                {
                    "fixture_id": f"price_{idx:03d}",
                    "route_fixture": f"route_{idx:03d}.json",
                    "expected_contains_toll": is_toll,
                    "expected_toll_cost_gbp": 8.5 if is_toll else 0.0,
                }
            ),
            encoding="utf-8",
        )

    classification_out = tmp_path / "classification_out"
    pricing_out = tmp_path / "pricing_out"
    tariffs_out = tmp_path / "toll_tariffs_operator_truth.json"
    summary = collect_toll_truth_raw_uk.build(
        classification_source=classification_source,
        pricing_source=pricing_source,
        classification_out=classification_out,
        pricing_out=pricing_out,
        tariffs_out=tariffs_out,
        classification_target=20,
        pricing_target=4,
        min_tariff_rules=12,
    )
    assert summary["classification_count"] == 20
    assert summary["pricing_count"] == 4
    assert summary["tariff_rule_count"] == 12
    tariff_payload = json.loads(tariffs_out.read_text(encoding="utf-8"))
    assert tariff_payload["source"] == "proxy_from_labeled_toll_fixture_corpus_v1"
    assert len(tariff_payload["rules"]) == 12


def test_collect_scenario_mode_outcomes_proxy_builds_projected_rows(tmp_path: Path) -> None:
    scenario_jsonl = tmp_path / "scenario_live_observed.jsonl"
    base_row = {
        "as_of_utc": "2026-02-23T12:00:00Z",
        "corridor_bucket": "uk_default",
        "corridor_geohash5": "gcpvj",
        "hour_slot_local": 12,
        "road_mix_bucket": "mixed",
        "vehicle_class": "rigid_hgv",
        "day_kind": "weekday",
        "weather_bucket": "clear",
        "traffic_features": {"flow_index": 125.0, "speed_index": 55.0},
        "incident_features": {"delay_pressure": 1.0, "severity_index": 0.5},
        "weather_features": {"weather_severity_index": 1.2},
    }
    scenario_jsonl.write_text(
        "\n".join([json.dumps(base_row), json.dumps(base_row)]) + "\n",
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "scenario_mode_outcomes_observed.jsonl"
    summary = collect_scenario_mode_outcomes_proxy_uk.build(
        scenario_jsonl=scenario_jsonl,
        output_jsonl=output_jsonl,
    )
    assert summary["rows"] == 1
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["mode_observation_source"] == "empirical_proxy_public_feeds_v1"
    assert row["mode_is_projected"] is True
    no_mode = row["modes"]["no_sharing"]
    partial_mode = row["modes"]["partial_sharing"]
    full_mode = row["modes"]["full_sharing"]
    assert no_mode["duration_multiplier"] > partial_mode["duration_multiplier"] > full_mode["duration_multiplier"]


def test_fetch_stochastic_residuals_build_writes_output(tmp_path: Path) -> None:
    as_of = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    raw_csv = tmp_path / "residuals_raw.csv"
    raw_csv.write_text(
        "regime_id,corridor_bucket,day_kind,local_time_slot,road_bucket,weather_profile,vehicle_type,traffic,incident,weather,price,eco,sigma\n"
        "weekday_peak,uk_default,weekday,h08,motorway_heavy,clear,rigid_hgv,1.20,1.10,1.00,1.05,1.02,0.20\n"
        "weekday_offpeak,uk_default,weekday,h12,mixed,rain,rigid_hgv,0.95,0.90,1.10,0.98,0.96,0.15\n"
        "weekend,uk_default,weekend,h16,trunk_heavy,clear,van,1.05,1.00,0.95,1.00,1.01,0.18\n"
        "holiday,uk_default,holiday,h10,mixed,snow,artic_hgv,1.30,1.25,1.35,1.20,1.18,0.35\n"
        "weekday_peak,uk_default,weekday,h18,motorway_heavy,fog,van,1.15,1.05,1.08,1.10,1.06,0.22\n"
        "weekend,uk_default,weekend,h20,trunk_heavy,storm,artic_hgv,1.25,1.22,1.28,1.19,1.15,0.30\n",
        encoding="utf-8",
    )
    output = tmp_path / "stochastic_residuals_empirical.csv"
    rows = fetch_stochastic_residuals_uk.build(
        raw_csv=raw_csv,
        output_csv=output,
        min_rows=3,
        as_of_utc=as_of,
        min_unique_regimes=1,
        min_unique_road_buckets=1,
        min_unique_weather_profiles=1,
        min_unique_vehicle_types=1,
        min_unique_local_slots=1,
        min_unique_corridors=1,
        max_age_days=3650,
    )
    assert rows >= 6
    assert output.exists()


def test_fetch_toll_truth_build_copies_fixtures_and_writes_calibration(tmp_path: Path) -> None:
    classification_source = tmp_path / "classification_source"
    pricing_source = tmp_path / "pricing_source"
    classification_source.mkdir(parents=True, exist_ok=True)
    pricing_source.mkdir(parents=True, exist_ok=True)

    for idx in range(20):
        is_toll = idx < 10
        (classification_source / f"class_{idx:03d}.json").write_text(
            json.dumps(
                {
                    "fixture_id": f"class_{idx:03d}",
                    "route_fixture": f"route_{idx:03d}.json",
                    "expected_has_toll": is_toll,
                    "expected_reason": "class_and_seed" if is_toll else "none",
                    "class_signal": is_toll,
                    "seed_signal": is_toll,
                    "weight": 1.0 + (idx % 3) * 0.1,
                }
            ),
            encoding="utf-8",
        )

    for idx in range(4):
        has_toll = idx % 2 == 0
        (pricing_source / f"price_{idx:03d}.json").write_text(
            json.dumps(
                {
                    "fixture_id": f"price_{idx:03d}",
                    "route_fixture": f"route_{idx:03d}.json",
                    "expected_contains_toll": has_toll,
                    "expected_toll_cost_gbp": 12.5 if has_toll else 0.0,
                }
            ),
            encoding="utf-8",
        )

    classification_out = tmp_path / "classification_out"
    pricing_out = tmp_path / "pricing_out"
    calibration_out = tmp_path / "toll_confidence_calibration_uk.json"

    class_rows, price_rows = fetch_toll_truth_uk.build(
        classification_source_dir=classification_source,
        pricing_source_dir=pricing_source,
        classification_out_dir=classification_out,
        pricing_out_dir=pricing_out,
        classification_target=20,
        pricing_target=4,
        calibration_out_json=calibration_out,
    )
    assert class_rows == 20
    assert price_rows == 4
    assert calibration_out.exists()
    calibration_payload = json.loads(calibration_out.read_text(encoding="utf-8"))
    assert "logit_model" in calibration_payload
    assert "reliability_bins" in calibration_payload
