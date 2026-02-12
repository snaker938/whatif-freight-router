from __future__ import annotations

from pathlib import Path

import httpx

from scripts.benchmark_batch_pareto import build_parser as benchmark_parser
from scripts.benchmark_batch_pareto import run_benchmark
from scripts.check_eta_concept_drift import build_parser as drift_parser
from scripts.check_eta_concept_drift import compute_drift_metrics, run_drift_check
from scripts.generate_run_report import build_parser as report_parser
from scripts.generate_run_report import run_generate_report
from scripts.run_headless_scenario import execute_headless_run, load_payload_from_csv
from scripts.run_robustness_analysis import build_parser as robustness_parser
from scripts.run_robustness_analysis import run_robustness
from scripts.run_sensitivity_analysis import build_parser as sensitivity_parser
from scripts.run_sensitivity_analysis import run_sensitivity


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


def test_run_robustness_inprocess_fake_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    args = robustness_parser().parse_args(
        [
            "--mode",
            "inprocess-fake",
            "--seeds",
            "101,202",
            "--pair-count",
            "6",
            "--out-dir",
            str(out_dir),
        ]
    )
    payload = run_robustness(args)
    assert payload["mode"] == "inprocess-fake"
    assert payload["pair_count"] == 6
    assert payload["seeds"] == [101, 202]
    assert len(payload["runs"]) == 2
    assert "avg_duration_s_mean" in payload["aggregate"]
    assert Path(payload["json_output"]).exists()
    assert Path(payload["csv_output"]).exists()


def test_run_sensitivity_inprocess_fake_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    args = sensitivity_parser().parse_args(
        [
            "--mode",
            "inprocess-fake",
            "--pair-count",
            "5",
            "--seed",
            "303",
            "--include-no-tolls",
            "--out-dir",
            str(out_dir),
        ]
    )
    payload = run_sensitivity(args)
    assert payload["mode"] == "inprocess-fake"
    assert payload["pair_count"] == 5
    assert payload["seed"] == 303
    assert len(payload["cases"]) >= 2
    baseline = payload["cases"][0]
    assert baseline["case"] == "baseline"
    assert baseline["delta_monetary_cost"] == 0
    assert Path(payload["json_output"]).exists()
    assert Path(payload["csv_output"]).exists()


def test_eta_concept_drift_metrics_and_alerts(tmp_path: Path) -> None:
    csv_path = tmp_path / "eta.csv"
    csv_path.write_text(
        "trip_id,predicted_eta_s,observed_eta_s\n"
        "a,100,110\n"
        "b,200,180\n",
        encoding="utf-8",
    )

    args = drift_parser().parse_args(
        [
            "--input-csv",
            str(csv_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--mae-threshold-s",
            "10",
            "--mape-threshold-pct",
            "8",
        ]
    )
    payload = run_drift_check(args)
    assert payload["metrics"]["count"] == 2
    assert payload["metrics"]["mae_s"] == 15.0
    assert payload["alerts"]["mae_alert"] is True
    assert payload["alerts"]["any_alert"] is True
    assert Path(payload["json_output"]).exists()
    assert Path(payload["csv_output"]).exists()


def test_compute_drift_metrics_empty_rows() -> None:
    try:
        compute_drift_metrics([])
    except ValueError:
        assert True
    else:
        assert False


def test_generate_run_report_script_writes_pdf(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    run_id = "11111111-1111-1111-1111-111111111111"
    manifest_dir = out_dir / "manifests"
    artifact_dir = out_dir / "artifacts" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (manifest_dir / f"{run_id}.json").write_text(
        '{"run_id":"11111111-1111-1111-1111-111111111111","created_at":"2026-02-12T00:00:00Z","request":{"vehicle_type":"rigid_hgv","scenario_mode":"no_sharing"},"execution":{"pair_count":1},"signature":{"algorithm":"HMAC-SHA256","signature":"abc"}}',
        encoding="utf-8",
    )
    (artifact_dir / "results.json").write_text(
        '{"run_id":"11111111-1111-1111-1111-111111111111","results":[{"routes":[{"id":"r0","metrics":{"duration_s":3600,"monetary_cost":120,"emissions_kg":85}}]}]}',
        encoding="utf-8",
    )
    (artifact_dir / "metadata.json").write_text(
        '{"run_id":"11111111-1111-1111-1111-111111111111","manifest_endpoint":"/runs/11111111-1111-1111-1111-111111111111/manifest","artifacts_endpoint":"/runs/11111111-1111-1111-1111-111111111111/artifacts","pair_count":1}',
        encoding="utf-8",
    )

    args = report_parser().parse_args(
        [
            "--run-id",
            run_id,
            "--out-dir",
            str(out_dir),
        ]
    )
    payload = run_generate_report(args)
    report_path = Path(payload["report_pdf"])

    assert payload["run_id"] == run_id
    assert report_path.exists()
    assert report_path.read_bytes().startswith(b"%PDF")
