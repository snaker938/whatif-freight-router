from __future__ import annotations

import os
from pathlib import Path

from app import routing_ors


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_local_ors_runtime_manifest_reports_verified_graph(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    docker_compose = _write(
        repo_root / "docker-compose.yml",
        "\n".join(
            [
                "services:",
                "  ors:",
                "    image: openrouteservice/openrouteservice:v9.7.1",
            ]
        ),
    )
    config_path = _write(repo_root / "ors" / "config" / "ors-config.yml", "ors:\n  engine:\n    init_threads: 1\n")
    pbf_path = _write(repo_root / "osrm" / "data" / "pbf" / "region.osm.pbf", "pbf")
    _write(repo_root / "osrm" / "data" / "pbf" / ".region_pbf_url", "https://example.test/region.osm.pbf\n")
    graph_dir = repo_root / "ors" / "data" / "driving-hgv"
    _write(
        graph_dir / "graph_build_info.yml",
        "\n".join(
            [
                "graph_build_date: 2026-03-22T16:39:30+0000",
                "osm_date: 2026-02-23T21:21:28+0000",
                "graph_version: 4",
            ]
        ),
    )
    _write(graph_dir / "stamp.txt", "2164892613\n")
    _write(graph_dir / "properties", "graph.properties")
    _write(graph_dir / "edges", "edges")

    monkeypatch.setattr(routing_ors, "_DOCKER_COMPOSE_PATH", docker_compose)
    monkeypatch.setattr(routing_ors, "_ORS_CONFIG_PATH", config_path)
    monkeypatch.setattr(routing_ors, "_SOURCE_PBF_PATH", pbf_path)
    monkeypatch.setattr(routing_ors, "_SOURCE_PBF_URL_PATH", repo_root / "osrm" / "data" / "pbf" / ".region_pbf_url")
    monkeypatch.setattr(routing_ors, "_ORS_GRAPH_ROOT", repo_root / "ors" / "data")

    manifest = routing_ors.local_ors_runtime_manifest(
        base_url="http://localhost:8082/ors",
        profile="driving-hgv",
        vehicle_type="hgv",
    )

    assert manifest["identity_status"] == "graph_identity_verified"
    assert manifest["compose_image"] == "openrouteservice/openrouteservice:v9.7.1"
    assert manifest["graph_file_count"] == 4
    assert manifest["graph_listing_digest"]
    assert manifest["manifest_hash"]
    assert manifest["graph_build_info"]["graph_build_date"] == "2026-03-22T16:39:30+0000"
    assert manifest["source_pbf"]["source_url"] == "https://example.test/region.osm.pbf"


def test_local_ors_runtime_manifest_detects_graph_older_than_source_pbf(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    config_path = _write(repo_root / "ors" / "config" / "ors-config.yml", "ors:\n")
    pbf_path = _write(repo_root / "osrm" / "data" / "pbf" / "region.osm.pbf", "pbf")
    graph_dir = repo_root / "ors" / "data" / "driving-hgv"
    build_info_path = _write(graph_dir / "graph_build_info.yml", "graph_build_date: 2026-03-20T10:00:00+0000\n")
    _write(graph_dir / "edges", "edges")

    old_graph_time = 100
    new_source_time = 200
    os.utime(config_path, ns=(old_graph_time, old_graph_time))
    os.utime(build_info_path, ns=(old_graph_time, old_graph_time))
    os.utime(pbf_path, ns=(new_source_time, new_source_time))

    monkeypatch.setattr(routing_ors, "_ORS_CONFIG_PATH", config_path)
    monkeypatch.setattr(routing_ors, "_SOURCE_PBF_PATH", pbf_path)
    monkeypatch.setattr(routing_ors, "_SOURCE_PBF_URL_PATH", repo_root / "osrm" / "data" / "pbf" / ".region_pbf_url")
    monkeypatch.setattr(routing_ors, "_ORS_GRAPH_ROOT", repo_root / "ors" / "data")

    manifest = routing_ors.local_ors_runtime_manifest(
        base_url="http://localhost:8082/ors",
        profile="driving-hgv",
        vehicle_type="hgv",
    )

    assert manifest["identity_status"] == "graph_predates_source_pbf"
