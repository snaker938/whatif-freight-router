from __future__ import annotations

import hashlib
from pathlib import Path

import app.terrain_dem_index as terrain_dem_index
from app.experiment_store import (
    create_experiment,
    delete_experiment,
    get_experiment,
    list_experiments,
    update_experiment,
)
from app.models import ExperimentBundleInput, LatLng, ScenarioCompareRequest
from app.oracle_quality_store import (
    append_check_record,
    compute_dashboard_payload,
    load_check_records,
    write_summary_artifacts,
)
from app.settings import settings


def _experiment_payload(name: str) -> ExperimentBundleInput:
    return ExperimentBundleInput(
        name=name,
        description="unit test bundle",
        request=ScenarioCompareRequest(
            origin=LatLng(lat=52.4862, lon=-1.8904),
            destination=LatLng(lat=51.5072, lon=-0.1276),
            vehicle_type="rigid_hgv",
        ),
    )


def test_experiment_store_crud_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    created, created_path = create_experiment(_experiment_payload("Exp A"))
    assert created_path.exists()
    assert created.name == "Exp A"

    fetched = get_experiment(created.id)
    assert fetched.id == created.id

    updated, updated_path = update_experiment(created.id, _experiment_payload("Exp B"))
    assert updated_path.exists()
    assert updated.id == created.id
    assert updated.name == "Exp B"

    listed = list_experiments(q="exp")
    assert any(item.id == created.id for item in listed)

    deleted_id, index_path = delete_experiment(created.id)
    assert deleted_id == created.id
    assert index_path.exists()
    assert all(item.id != created.id for item in list_experiments())


def test_oracle_quality_store_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    append_check_record(
        {
            "source": "feed_a",
            "passed": True,
            "schema_valid": True,
            "signature_valid": True,
            "freshness_s": 100.0,
            "latency_ms": 10.0,
            "observed_at_utc": "2026-02-23T09:00:00Z",
        }
    )
    append_check_record(
        {
            "source": "feed_a",
            "passed": False,
            "schema_valid": False,
            "signature_valid": True,
            "freshness_s": 3600.0,
            "latency_ms": 30.0,
            "observed_at_utc": "2026-02-23T10:00:00Z",
        }
    )

    records = load_check_records()
    payload = compute_dashboard_payload(records, stale_threshold_s=900.0)
    summary_path, csv_path = write_summary_artifacts(payload)

    assert len(records) == 2
    assert payload["source_count"] == 1
    assert payload["total_checks"] == 2
    assert summary_path.exists()
    assert csv_path.exists()
    assert "feed_a" in csv_path.read_text(encoding="utf-8")


def test_terrain_dem_index_grid_and_checksum_helpers(tmp_path: Path) -> None:
    grid_payload = {
        "rows": 2,
        "cols": 2,
        "lat_min": 50.0,
        "lon_min": -2.0,
        "lat_step": 0.1,
        "lon_step": 0.1,
        "values": [[100.0, 101.0], [102.0, 103.0]],
    }
    tile = terrain_dem_index._coerce_grid(grid_payload, tile_id="tile_1", path="dummy.json")
    assert tile is not None
    assert tile.rows == 2
    assert tile.cols == 2
    assert tile.values[1][1] == 103.0

    invalid = terrain_dem_index._coerce_grid(
        {"rows": 1, "cols": 2, "values": [[1, 2]], "lat_step": 0.1, "lon_step": 0.1},
        tile_id="bad",
        path="bad.json",
    )
    assert invalid is None

    tile_file = tmp_path / "tile_a.bin"
    tile_file.write_bytes(b"terrain-tile")
    checksum = hashlib.sha256(tile_file.read_bytes()).hexdigest()
    payload_ok = {"checksums": {"tile_a.bin": checksum}}
    payload_bad = {"checksums": {"tile_a.bin": "deadbeef"}}

    assert terrain_dem_index._verify_manifest_checksums(payload_ok, manifest_dir=tmp_path) is True
    assert terrain_dem_index._verify_manifest_checksums(payload_bad, manifest_dir=tmp_path) is False
