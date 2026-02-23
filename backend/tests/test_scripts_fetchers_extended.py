from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
