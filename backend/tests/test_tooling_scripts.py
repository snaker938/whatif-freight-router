from __future__ import annotations

from pathlib import Path

import httpx

from scripts.benchmark_batch_pareto import build_parser as benchmark_parser
from scripts.benchmark_batch_pareto import run_benchmark
from scripts.run_headless_scenario import execute_headless_run, load_payload_from_csv


def test_benchmark_parser_defaults() -> None:
    args = benchmark_parser().parse_args([])
    assert args.pair_count == 100
    assert args.mode == "inprocess-fake"
    assert args.max_alternatives == 3


def test_run_benchmark_inprocess_fake_writes_schema(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    output_file = out_dir / "bench.json"
    args = benchmark_parser().parse_args(
        [
            "--mode",
            "inprocess-fake",
            "--pair-count",
            "12",
            "--out-dir",
            str(out_dir),
            "--output",
            str(output_file),
        ]
    )

    record = run_benchmark(args)
    assert record["mode"] == "inprocess-fake"
    assert record["pair_count"] == 12
    assert record["duration_ms"] >= 0
    assert record["peak_memory_bytes"] >= 0
    assert record["error_count"] >= 0
    assert Path(record["log_path"]).exists()


def test_load_payload_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "origin_lat,origin_lon,destination_lat,destination_lon\n"
        "52.4862,-1.8904,51.5072,-0.1276\n",
        encoding="utf-8",
    )

    payload = load_payload_from_csv(
        str(csv_path),
        vehicle_type="rigid_hgv",
        scenario_mode="no_sharing",
        max_alternatives=5,
        seed=42,
        model_version="headless-v1",
    )
    assert len(payload["pairs"]) == 1
    assert payload["seed"] == 42
    assert payload["toggles"]["headless_mode"] is True


def test_execute_headless_run_with_mock_transport(tmp_path: Path) -> None:
    run_id = "11111111-1111-1111-1111-111111111111"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/batch/pareto":
            return httpx.Response(
                200,
                json={"run_id": run_id, "results": [{"error": None}]},
            )
        if request.method == "GET" and path == f"/runs/{run_id}/manifest":
            return httpx.Response(200, content=b'{"schema_version":"1.0.0"}')
        if request.method == "GET" and path == f"/runs/{run_id}/artifacts":
            return httpx.Response(
                200,
                json={
                    "artifacts": [
                        {"name": "results.json"},
                        {"name": "results.csv"},
                        {"name": "metadata.json"},
                    ]
                },
            )
        if request.method == "GET" and path == f"/runs/{run_id}/artifacts/results.json":
            return httpx.Response(200, content=b'{"run_id":"11111111-1111-1111-1111-111111111111"}')
        if request.method == "GET" and path == f"/runs/{run_id}/artifacts/results.csv":
            return httpx.Response(200, content=b"pair_index,route_id\n0,r0\n")
        if request.method == "GET" and path == f"/runs/{run_id}/artifacts/metadata.json":
            return httpx.Response(200, content=b'{"artifact_names":["results.json"]}')
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://testserver")
    try:
        payload = {
            "pairs": [
                {
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                }
            ],
            "vehicle_type": "rigid_hgv",
            "scenario_mode": "no_sharing",
            "max_alternatives": 5,
        }
        summary = execute_headless_run(
            payload,
            backend_url="http://testserver",
            save_dir=str(tmp_path / "headless"),
            client=client,
        )
    finally:
        client.close()

    summary_path = Path(summary["summary_file"])
    assert summary["run_id"] == run_id
    assert summary_path.exists()
    assert Path(summary["manifest_file"]).exists()
    assert summary["downloaded_artifacts"] == ["metadata.json", "results.csv", "results.json"]
