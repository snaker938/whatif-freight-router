from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx

import scripts.build_od_corpus_uk as corpus_module
import scripts.run_thesis_evaluation as thesis_module


def test_build_od_corpus_is_deterministic_and_stratified(monkeypatch) -> None:
    pairs = [
        ({"lat": 51.0, "lon": -1.0}, {"lat": 51.1, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 52.0, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 53.0, "lon": -1.0}),
        ({"lat": 51.0, "lon": -1.0}, {"lat": 56.5, "lon": -1.0}),
    ]
    index = {"value": 0}

    def fake_sample_candidate_pair(rng, bbox):  # noqa: ARG001
        pair = pairs[index["value"] % len(pairs)]
        index["value"] += 1
        return pair

    def fake_feasibility_fn(**kwargs):  # noqa: ARG001
        return {
            "ok": True,
            "origin_node_id": "o",
            "destination_node_id": "d",
            "origin_nearest_distance_m": 12.0,
            "destination_nearest_distance_m": 13.0,
            "message": "ok",
        }

    monkeypatch.setattr(corpus_module, "_sample_candidate_pair", fake_sample_candidate_pair)
    monkeypatch.setattr(corpus_module, "route_graph_od_feasibility", fake_feasibility_fn)

    bbox = corpus_module.UKBBox(south=50.0, north=57.0, west=-2.0, east=1.0)
    first = corpus_module.build_od_corpus(
        seed=123,
        pair_count=4,
        bbox=bbox,
        max_attempts=8,
        feasibility_fn=fake_feasibility_fn,
    )
    index["value"] = 0
    second = corpus_module.build_od_corpus(
        seed=123,
        pair_count=4,
        bbox=bbox,
        max_attempts=8,
        feasibility_fn=fake_feasibility_fn,
    )

    assert first["corpus_hash"] == second["corpus_hash"]
    assert first["rows"] == second["rows"]
    assert first["accepted_count"] == len(first["rows"])
    assert first["accepted_count"] >= 3
    assert sum(1 for count in first["accepted_by_bin"].values() if count) >= 3


def test_run_thesis_evaluation_records_and_replays_ors_snapshot(tmp_path: Path) -> None:
    corpus_csv = tmp_path / "corpus.csv"
    with corpus_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["od_id", "origin_lat", "origin_lon", "destination_lat", "destination_lon", "distance_bin", "seed"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "od_id": "od-000001",
                "origin_lat": 52.0,
                "origin_lon": -1.5,
                "destination_lat": 51.5,
                "destination_lon": -1.2,
                "distance_bin": "30-100 km",
                "seed": 99,
            }
        )

    route_payload = {
        "selected": {
            "id": "r-selected",
            "metrics": {"distance_km": 10.0, "duration_s": 100.0, "monetary_cost": 20.0, "emissions_kg": 5.0},
            "evidence_provenance": {
                "active_families": ["scenario", "toll", "terrain"],
                "families": [
                    {"family": "scenario", "active": True},
                    {"family": "toll", "active": True},
                    {"family": "terrain", "active": True},
                ],
            },
        },
        "candidates": [
            {
                "id": "r-selected",
                "metrics": {"distance_km": 10.0, "duration_s": 100.0, "monetary_cost": 20.0, "emissions_kg": 5.0},
                "evidence_provenance": {
                    "active_families": ["scenario", "toll", "terrain"],
                    "families": [
                        {"family": "scenario", "active": True},
                        {"family": "toll", "active": True},
                        {"family": "terrain", "active": True},
                    ],
                },
            },
            {
                "id": "r-alt",
                "metrics": {"distance_km": 12.0, "duration_s": 110.0, "monetary_cost": 19.0, "emissions_kg": 6.0},
                "evidence_provenance": {
                    "active_families": ["scenario", "toll", "terrain"],
                    "families": [
                        {"family": "scenario", "active": True},
                        {"family": "toll", "active": True},
                        {"family": "terrain", "active": True},
                    ],
                },
            },
        ],
    }

    osrm_payload = {
        "baseline": {
            "id": "osrm-base",
            "metrics": {"distance_km": 11.0, "duration_s": 130.0, "monetary_cost": 18.0, "emissions_kg": 5.5},
        },
        "method": "osrm_quick_baseline",
        "compute_ms": 11.5,
    }
    ors_payload = {
        "baseline": {
            "id": "ors-base",
            "metrics": {"distance_km": 11.5, "duration_s": 135.0, "monetary_cost": 17.0, "emissions_kg": 5.3},
        },
        "method": "ors_reference",
        "compute_ms": 14.2,
    }

    def record_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            return httpx.Response(200, json=route_payload)
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(200, json=osrm_payload)
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            return httpx.Response(200, json=ors_payload)
        return httpx.Response(404, json={"detail": "not found"})

    record_client = httpx.Client(transport=httpx.MockTransport(record_handler), base_url="http://testserver")
    try:
        record_args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--seed",
                "7",
                "--world-count",
                "12",
                "--certificate-threshold",
                "0.95",
                "--tau-stop",
                "0.0",
                "--ors-snapshot-mode",
                "record",
            ]
        )
        record_payload = thesis_module.run_thesis_evaluation(record_args, client=record_client)
    finally:
        record_client.close()

    snapshot_path = Path(record_payload["ors_snapshot_path"])
    assert snapshot_path.exists()
    assert Path(record_payload["results_csv"]).exists()
    assert Path(record_payload["summary_csv"]).exists()
    assert Path(record_payload["certificate_summary"]).exists()
    assert Path(record_payload["voi_action_trace"]).exists()
    assert Path(record_payload["voi_stop_certificate"]).exists()

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["routes"]["od-000001"]["baseline"]["id"] == "ors-base"

    def replay_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/route":
            return httpx.Response(200, json=route_payload)
        if request.method == "POST" and request.url.path == "/route/baseline":
            return httpx.Response(200, json=osrm_payload)
        if request.method == "POST" and request.url.path == "/route/baseline/ors":
            raise AssertionError("ORS live endpoint should not be called in replay mode")
        return httpx.Response(404, json={"detail": "not found"})

    replay_client = httpx.Client(transport=httpx.MockTransport(replay_handler), base_url="http://testserver")
    try:
        replay_args = thesis_module._build_parser().parse_args(
            [
                "--corpus-csv",
                str(corpus_csv),
                "--out-dir",
                str(tmp_path / "out"),
                "--backend-url",
                "http://testserver",
                "--seed",
                "7",
                "--world-count",
                "12",
                "--certificate-threshold",
                "0.95",
                "--tau-stop",
                "0.0",
                "--ors-snapshot-mode",
                "replay",
                "--ors-snapshot-path",
                str(snapshot_path),
            ]
        )
        replay_payload = thesis_module.run_thesis_evaluation(replay_args, client=replay_client)
    finally:
        replay_client.close()

    assert replay_payload["corpus_hash"] == record_payload["corpus_hash"]
    stable_keys = [
        "od_id",
        "variant_id",
        "route_id",
        "certificate",
        "certified",
        "selected_duration_s",
        "selected_monetary_cost",
        "selected_emissions_kg",
        "delta_vs_osrm_duration_s",
        "delta_vs_ors_duration_s",
        "dominates_osrm",
        "dominates_ors",
        "search_budget_used",
        "evidence_budget_used",
    ]
    assert [
        {key: row[key] for key in stable_keys}
        for row in replay_payload["rows"]
    ] == [
        {key: row[key] for key in stable_keys}
        for row in record_payload["rows"]
    ]
    assert replay_payload["summary_rows"] == record_payload["summary_rows"]
    assert Path(replay_payload["results_json"]).exists()
    assert Path(replay_payload["summary_json"]).exists()
