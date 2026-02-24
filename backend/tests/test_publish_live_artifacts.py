from __future__ import annotations

import json
from pathlib import Path

import scripts.publish_live_artifacts_uk as publish_live_artifacts_uk


def test_publish_live_artifacts_builds_strict_json_outputs(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets" / "uk"
    out_dir = tmp_path / "out" / "model_assets"
    raw_dir = tmp_path / "data" / "raw" / "uk"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    scenario_path = assets_dir / "scenario_profiles_uk.json"
    scenario_path.write_text(
        json.dumps(
            {
                "version": "scenario_profiles_uk_v2_live",
                "as_of_utc": "2026-02-24T00:00:00Z",
                "signature": "abc123",
                "contexts": {"ctx": {}},
            }
        ),
        encoding="utf-8",
    )
    carbon_path = assets_dir / "carbon_price_schedule_uk.json"
    carbon_path.write_text(
        json.dumps(
            {
                "source": "empirical",
                "as_of_utc": "2026-02-24T00:00:00Z",
                "prices_gbp_per_kg": {"central": {"2026": 0.12}},
                "uncertainty_distribution_by_year": {
                    "2026": {"p10": 0.10, "p50": 0.12, "p90": 0.14}
                },
            }
        ),
        encoding="utf-8",
    )
    fuel_asset_path = assets_dir / "fuel_prices_uk.json"
    fuel_asset_path.write_text(
        json.dumps(
            {
                "source": "public_fuel_history_uk",
                "as_of_utc": "2026-02-24T00:00:00Z",
                "prices_gbp_per_l": {"diesel": 1.55, "petrol": 1.62, "lng": 1.1},
                "grid_price_gbp_per_kwh": 0.30,
                "history": [
                    {
                        "as_of": "2026-02-24",
                        "prices_gbp_per_l": {"diesel": 1.55, "petrol": 1.62, "lng": 1.1},
                        "grid_price_gbp_per_kwh": 0.30,
                    }
                ],
                "regional_multipliers": {"uk_default": 1.0},
            }
        ),
        encoding="utf-8",
    )
    fuel_raw_path = raw_dir / "fuel_prices_raw.json"
    fuel_raw_path.write_text(
        json.dumps(
            {
                "source": "public_fuel_history_uk",
                "as_of": "2026-02-24T00:00:00Z",
                "history": [
                    {
                        "as_of": "2026-02-24",
                        "prices_gbp_per_l": {"diesel": 1.55, "petrol": 1.62, "lng": 1.1},
                        "grid_price_gbp_per_kwh": 0.30,
                    }
                ],
                "regional_multipliers": {"uk_default": 1.0},
            }
        ),
        encoding="utf-8",
    )
    departure_in_path = out_dir / "departure_profiles_uk.json"
    departure_in_path.write_text(
        json.dumps({"as_of_utc": "2026-02-24T00:00:00Z", "profiles": {"uk_default": {}}}),
        encoding="utf-8",
    )
    stochastic_in_path = out_dir / "stochastic_regimes_uk.json"
    stochastic_in_path.write_text(
        json.dumps({"as_of_utc": "2026-02-24T00:00:00Z", "regimes": {"weekday": {}}}),
        encoding="utf-8",
    )
    toll_topology_in_path = out_dir / "toll_segments_seed_compiled.json"
    toll_topology_in_path.write_text(
        json.dumps(
            {
                "version": "uk-v2",
                "segments": [
                    {
                        "id": "seg_1",
                        "name": "Test Crossing",
                        "operator": "operator_a",
                        "road_class": "motorway",
                        "direction": "both",
                        "crossing_fee_gbp": 3.5,
                        "distance_fee_gbp_per_km": 0.2,
                        "coordinates": [[52.0, -1.0], [52.1, -1.1]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    toll_tariffs_in_path = out_dir / "toll_tariffs_uk_compiled.json"
    toll_tariffs_in_path.write_text(
        json.dumps(
            {
                "version": "uk-v2",
                "source": "empirical_tariffs",
                "defaults": {"crossing_fee_gbp": 0.0, "distance_fee_gbp_per_km": 0.0},
                "rules": [
                    {
                        "id": "rule_1",
                        "operator": "operator_a",
                        "crossing_id": "seg_1",
                        "road_class": "motorway",
                        "direction": "both",
                        "start_minute": 0,
                        "end_minute": 1439,
                        "crossing_fee_gbp": 3.5,
                        "distance_fee_gbp_per_km": 0.2,
                        "vehicle_classes": ["rigid_hgv"],
                        "axle_classes": ["heavy"],
                        "payment_classes": ["cash"],
                        "exemptions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    departure_out_path = assets_dir / "departure_profiles_uk.json"
    stochastic_out_path = assets_dir / "stochastic_regimes_uk.json"
    toll_topology_out_path = assets_dir / "toll_topology_uk.json"
    toll_tariffs_out_path = assets_dir / "toll_tariffs_uk.json"

    summary = publish_live_artifacts_uk.publish(
        scenario_path=scenario_path,
        fuel_asset_path=fuel_asset_path,
        fuel_raw_path=fuel_raw_path,
        carbon_path=carbon_path,
        departure_in_path=departure_in_path,
        stochastic_in_path=stochastic_in_path,
        toll_topology_in_path=toll_topology_in_path,
        toll_tariffs_in_path=toll_tariffs_in_path,
        departure_out_path=departure_out_path,
        stochastic_out_path=stochastic_out_path,
        toll_topology_out_path=toll_topology_out_path,
        toll_tariffs_out_path=toll_tariffs_out_path,
    )

    assert summary["fuel_signature_regenerated"] is True
    assert departure_out_path.exists()
    assert stochastic_out_path.exists()
    assert toll_topology_out_path.exists()
    assert toll_tariffs_out_path.exists()

    topology_payload = json.loads(toll_topology_out_path.read_text(encoding="utf-8"))
    assert topology_payload["type"] == "FeatureCollection"
    assert topology_payload["features"]
    assert topology_payload["as_of_utc"]

    tariffs_payload = json.loads(toll_tariffs_out_path.read_text(encoding="utf-8"))
    assert tariffs_payload["rules"]
    assert tariffs_payload["as_of_utc"]

    fuel_payload = json.loads(fuel_asset_path.read_text(encoding="utf-8"))
    assert fuel_payload.get("signature")


def test_publish_live_artifacts_sanitizes_default_tariff_classes(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets" / "uk"
    out_dir = tmp_path / "out" / "model_assets"
    raw_dir = tmp_path / "data" / "raw" / "uk"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    (assets_dir / "scenario_profiles_uk.json").write_text(
        json.dumps(
            {
                "version": "scenario_profiles_uk_v2_live",
                "as_of_utc": "2026-02-24T00:00:00Z",
                "signature": "abc123",
                "contexts": {"ctx": {}},
            }
        ),
        encoding="utf-8",
    )
    (assets_dir / "carbon_price_schedule_uk.json").write_text(
        json.dumps(
            {
                "source": "empirical",
                "as_of_utc": "2026-02-24T00:00:00Z",
                "prices_gbp_per_kg": {"central": {"2026": 0.12}},
                "uncertainty_distribution_by_year": {
                    "2026": {"p10": 0.10, "p50": 0.12, "p90": 0.14}
                },
            }
        ),
        encoding="utf-8",
    )
    fuel_asset_path = assets_dir / "fuel_prices_uk.json"
    fuel_asset_path.write_text(
        json.dumps(
            {
                "source": "public_fuel_history_uk",
                "as_of_utc": "2026-02-24T00:00:00Z",
                "prices_gbp_per_l": {"diesel": 1.55, "petrol": 1.62, "lng": 1.1},
                "grid_price_gbp_per_kwh": 0.30,
                "history": [
                    {
                        "as_of": "2026-02-24",
                        "prices_gbp_per_l": {"diesel": 1.55, "petrol": 1.62, "lng": 1.1},
                        "grid_price_gbp_per_kwh": 0.30,
                    }
                ],
                "regional_multipliers": {"uk_default": 1.0},
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "fuel_prices_raw.json").write_text(
        json.dumps(
            {
                "source": "public_fuel_history_uk",
                "as_of": "2026-02-24T00:00:00Z",
                "history": [
                    {
                        "as_of": "2026-02-24",
                        "prices_gbp_per_l": {"diesel": 1.55, "petrol": 1.62, "lng": 1.1},
                        "grid_price_gbp_per_kwh": 0.30,
                    }
                ],
                "regional_multipliers": {"uk_default": 1.0},
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "departure_profiles_uk.json").write_text(
        json.dumps({"as_of_utc": "2026-02-24T00:00:00Z", "profiles": {"uk_default": {}}}),
        encoding="utf-8",
    )
    (out_dir / "stochastic_regimes_uk.json").write_text(
        json.dumps({"as_of_utc": "2026-02-24T00:00:00Z", "regimes": {"weekday": {}}}),
        encoding="utf-8",
    )
    (out_dir / "toll_segments_seed_compiled.json").write_text(
        json.dumps(
            {
                "version": "uk-v2",
                "segments": [
                    {
                        "id": "seg_1",
                        "name": "Test Crossing",
                        "operator": "operator_a",
                        "road_class": "motorway",
                        "direction": "both",
                        "crossing_fee_gbp": 3.5,
                        "distance_fee_gbp_per_km": 0.2,
                        "coordinates": [[52.0, -1.0], [52.1, -1.1]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "toll_tariffs_uk_compiled.json").write_text(
        json.dumps(
            {
                "version": "uk-v2",
                "source": "empirical_tariffs",
                "defaults": {"crossing_fee_gbp": 0.0, "distance_fee_gbp_per_km": 0.0},
                "rules": [
                    {
                        "id": "rule_1",
                        "operator": "operator_a",
                        "crossing_id": "seg_1",
                        "road_class": "motorway",
                        "direction": "both",
                        "start_minute": 0,
                        "end_minute": 1439,
                        "crossing_fee_gbp": 3.5,
                        "distance_fee_gbp_per_km": 0.2,
                        "vehicle_classes": ["default", "rigid_hgv"],
                        "axle_classes": ["default", "heavy"],
                        "payment_classes": ["default", "cash"],
                        "exemptions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    publish_live_artifacts_uk.publish(
        scenario_path=assets_dir / "scenario_profiles_uk.json",
        fuel_asset_path=fuel_asset_path,
        fuel_raw_path=raw_dir / "fuel_prices_raw.json",
        carbon_path=assets_dir / "carbon_price_schedule_uk.json",
        departure_in_path=out_dir / "departure_profiles_uk.json",
        stochastic_in_path=out_dir / "stochastic_regimes_uk.json",
        toll_topology_in_path=out_dir / "toll_segments_seed_compiled.json",
        toll_tariffs_in_path=out_dir / "toll_tariffs_uk_compiled.json",
        departure_out_path=assets_dir / "departure_profiles_uk.json",
        stochastic_out_path=assets_dir / "stochastic_regimes_uk.json",
        toll_topology_out_path=assets_dir / "toll_topology_uk.json",
        toll_tariffs_out_path=assets_dir / "toll_tariffs_uk.json",
    )

    tariffs_payload = json.loads((assets_dir / "toll_tariffs_uk.json").read_text(encoding="utf-8"))
    row = tariffs_payload["rules"][0]
    assert row["vehicle_classes"] == ["rigid_hgv"]
    assert row["axle_classes"] == ["heavy"]
    assert row["payment_classes"] == ["cash"]
