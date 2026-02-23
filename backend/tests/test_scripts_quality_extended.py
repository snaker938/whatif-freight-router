from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.score_model_quality as score_model_quality
import scripts.validate_graph_coverage as validate_graph_coverage


def test_validate_graph_coverage_pass_and_fail_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_path = tmp_path / "routing_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "source": "uk-network.osm.pbf",
                "nodes": [
                    {"id": "n1", "lat": 52.4862, "lon": -1.8904},
                    {"id": "n2", "lat": 51.5072, "lon": -0.1276},
                ],
                "edges": [{"u": "n1", "v": "n2"}],
                "bbox": {"lat_min": 49.0, "lat_max": 61.0, "lon_min": -9.0, "lon_max": 3.0},
            }
        ),
        encoding="utf-8",
    )
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "route.json").write_text(
        json.dumps(
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.8904, 52.4862], [-0.1276, 51.5072]],
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(validate_graph_coverage, "_SciPyKDTree", None)
    report = validate_graph_coverage.validate(
        graph_path=graph_path,
        fixtures_dir=fixtures_dir,
        min_nodes=1,
        min_edges=1,
        max_fixture_dist_m=50_000.0,
    )
    assert report["coverage_passed"] is True
    assert report["nodes"] == 2
    assert report["edges"] == 1

    graph_bad = tmp_path / "routing_graph_bad.json"
    graph_bad.write_text(
        json.dumps(
            {
                "source": "roads.geojson",
                "nodes": [{"id": "n1", "lat": 52.0, "lon": -1.0}],
                "edges": [{"u": "n1", "v": "n1"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        validate_graph_coverage.validate(
            graph_path=graph_bad,
            fixtures_dir=fixtures_dir,
            min_nodes=1,
            min_edges=1,
            max_fixture_dist_m=50_000.0,
        )


def test_validate_graph_coverage_helpers_are_numerically_stable() -> None:
    dist = validate_graph_coverage._haversine_m(52.4862, -1.8904, 51.5072, -0.1276)
    assert dist > 100_000.0
    arr, mean_lat = validate_graph_coverage._to_xy_m([(52.4862, -1.8904), (51.5072, -0.1276)])
    assert arr.shape == (2, 2)
    assert isinstance(mean_lat, float)


def test_score_model_quality_helpers_and_missing_evidence_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert score_model_quality._ratio_to_target(50.0, 100.0) == pytest.approx(0.5)
    assert score_model_quality._clamp_score(101.0) == 100

    monkeypatch.setattr(score_model_quality.settings, "strict_live_data_required", True)
    missing_path = tmp_path / "missing_raw.csv"
    monkeypatch.setattr(
        score_model_quality,
        "_strict_raw_evidence_requirements",
        lambda: {"fuel_price": [missing_path]},
    )
    missing = score_model_quality._strict_missing_raw_evidence(subsystem="fuel_price")
    assert missing == [str(missing_path)]

    monkeypatch.setattr(
        score_model_quality.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(subsystem=None),
    )
    monkeypatch.setattr(score_model_quality, "_strict_missing_raw_evidence", lambda subsystem: [str(missing_path)])
    score_model_quality.main()
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert payload["all_passed"] is False
    assert "missing_raw_evidence" in payload
    assert payload["scores"]["overall"] == 0.0


def test_score_model_quality_subsystem_view_serializes_without_full_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(score_model_quality.settings, "strict_live_data_required", False)
    monkeypatch.setattr(
        score_model_quality.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(subsystem="fuel_price"),
    )
    monkeypatch.setattr(score_model_quality, "_strict_missing_raw_evidence", lambda subsystem: [])
    monkeypatch.setattr(
        score_model_quality,
        "_load_fixture_routes",
        lambda fixtures_dir: [
            {
                "geometry": {"type": "LineString", "coordinates": [[-1.8904, 52.4862], [-0.1276, 51.5072]]},
                "distance": 100_000.0,
                "duration": 7_000.0,
                "legs": [{"annotation": {"distance": [50_000.0, 50_000.0], "duration": [3_500.0, 3_500.0]}}],
            }
        ],
    )
    monkeypatch.setattr(
        score_model_quality,
        "_provenance_summary",
        lambda model_assets_dir: {"assets": {}, "model_asset_dir": str(model_assets_dir)},
    )
    monkeypatch.setattr(score_model_quality, "_synthetic_manifest_violations", lambda model_assets_dir: [])
    monkeypatch.setattr(score_model_quality.settings, "quality_min_fixture_routes", 1)
    monkeypatch.setattr(score_model_quality.settings, "quality_min_unique_corridors", 1)
    monkeypatch.setattr(score_model_quality, "_score_fuel_price", lambda routes: (97, {"mae": 0.01}))

    score_model_quality.main()
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["subsystem"] == "fuel_price"
    assert payload["score"] == 97
    assert payload["passed"] is True
