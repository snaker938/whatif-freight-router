from __future__ import annotations

import json
from pathlib import Path

import scripts.run_hot_rerun_benchmark as hot_module


def _summary_row(
    variant_id: str,
    pipeline_mode: str,
    *,
    route_cache: float,
    option_cache: float,
    option_reuse: float,
    runtime_osrm: float,
    runtime_ors: float,
    algorithm_runtime: float,
    refc_world_reuse: float | None = None,
) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "pipeline_mode": pipeline_mode,
        "mean_route_cache_hit_rate": route_cache,
        "mean_option_build_cache_hit_rate": option_cache,
        "mean_option_build_reuse_rate": option_reuse,
        "mean_refc_world_reuse_rate": refc_world_reuse,
        "mean_runtime_ratio_vs_osrm": runtime_osrm,
        "mean_runtime_ratio_vs_ors": runtime_ors,
        "mean_algorithm_runtime_ms": algorithm_runtime,
    }


def test_build_hot_rerun_comparison_scopes_gate_to_applicable_variants() -> None:
    cold_summary_rows = [
        _summary_row("V0", "legacy", route_cache=0.0, option_cache=0.0, option_reuse=0.0, runtime_osrm=10.0, runtime_ors=8.0, algorithm_runtime=100.0),
        _summary_row("A", "dccs", route_cache=0.0, option_cache=0.2, option_reuse=0.2, runtime_osrm=12.0, runtime_ors=9.0, algorithm_runtime=120.0),
        _summary_row("B", "dccs_refc", route_cache=0.0, option_cache=0.1, option_reuse=0.1, runtime_osrm=6.0, runtime_ors=4.0, algorithm_runtime=60.0, refc_world_reuse=0.0),
        _summary_row("C", "voi", route_cache=0.0, option_cache=0.1, option_reuse=0.1, runtime_osrm=9.0, runtime_ors=7.0, algorithm_runtime=90.0, refc_world_reuse=0.0),
    ]
    hot_summary_rows = [
        _summary_row("V0", "legacy", route_cache=0.75, option_cache=0.0, option_reuse=0.0, runtime_osrm=11.0, runtime_ors=7.5, algorithm_runtime=95.0),
        _summary_row("A", "dccs", route_cache=0.9, option_cache=0.8, option_reuse=0.8, runtime_osrm=8.0, runtime_ors=6.5, algorithm_runtime=90.0),
        _summary_row("B", "dccs_refc", route_cache=0.95, option_cache=0.9, option_reuse=0.9, runtime_osrm=3.0, runtime_ors=2.5, algorithm_runtime=40.0, refc_world_reuse=0.85),
        _summary_row("C", "voi", route_cache=0.92, option_cache=0.88, option_reuse=0.88, runtime_osrm=5.0, runtime_ors=4.0, algorithm_runtime=50.0, refc_world_reuse=0.9),
    ]

    comparison = hot_module.build_hot_rerun_comparison(
        pair_run_id="pair-1",
        cold_run_id="pair-1_cold",
        hot_run_id="pair-1_hot",
        cold_summary_rows=cold_summary_rows,
        hot_summary_rows=hot_summary_rows,
        cache_stats={"after_hot": {"route_cache": {"hits": 10}}},
    )

    assert comparison["hot_gate"]["all_green"] is True
    checks = comparison["hot_gate"]["metric_checks"]
    assert all(check["variant_id"] != "V0" for check in checks)
    a_row = next(row for row in comparison["comparison_rows"] if row["variant_id"] == "A")
    assert a_row["runtime_ratio_vs_osrm_improved"] is True
    assert a_row["hot_mean_route_cache_hit_rate"] == 0.9
    b_world_check = next(
        check
        for check in checks
        if check["metric"] == "mean_refc_world_reuse_rate" and check["variant_id"] == "B"
    )
    assert b_world_check["pass"] is True


class _DummyResponse:
    def __init__(self, payload: dict[str, object], *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return dict(self._payload)


class _DummyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get(self, path: str) -> _DummyResponse:
        self.calls.append(("GET", path))
        return _DummyResponse(
            {
                "route_cache": {"hits": 4, "misses": 1},
                "hot_rerun_route_cache_checkpoint": {"size": 3, "hits": 0, "misses": 0},
                "route_option_cache": {"hits": 8, "misses": 2},
            }
        )

    def delete(self, path: str) -> _DummyResponse:
        self.calls.append(("DELETE", path))
        return _DummyResponse({"route_cache": 1, "route_option_cache": 1})

    def post(self, path: str) -> _DummyResponse:
        self.calls.append(("POST", path))
        return _DummyResponse({"restored": 3, "checkpoint_size": 3})


def test_run_hot_rerun_benchmark_writes_hot_comparison_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.csv"
    corpus_path.write_text("od_id\nrow-1\n", encoding="utf-8")
    args = hot_module._build_parser().parse_args(
        [
            "--corpus-csv",
            str(corpus_path),
            "--out-dir",
            str(tmp_path),
            "--pair-run-id",
            "bench-1",
        ]
    )
    client = _DummyClient()
    observed_run_ids: list[str] = []
    observed_cache_modes: list[str] = []
    observed_cold_cache_scopes: list[str | None] = []
    observed_suite_roles: list[str | None] = []

    def fake_run_thesis_evaluation(run_args, *, client):
        observed_run_ids.append(str(run_args.run_id))
        observed_cache_modes.append(str(getattr(run_args, "cache_mode", "")))
        observed_cold_cache_scopes.append(getattr(run_args, "cold_cache_scope", None))
        observed_suite_roles.append(getattr(run_args, "evaluation_suite_role", None))
        artifact_dir = tmp_path / "artifacts" / str(run_args.run_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_id": str(run_args.run_id),
            "cache_reset_policy": str(getattr(run_args, "cold_cache_scope", "none")) if str(getattr(run_args, "cache_mode", "")) == "cold" else "none",
        }
        for name in ("metadata.json", "evaluation_manifest.json"):
            (artifact_dir / name).write_text(json.dumps(metadata), encoding="utf-8")
        summary_rows = [
            _summary_row("A", "dccs", route_cache=0.0, option_cache=0.2, option_reuse=0.2, runtime_osrm=10.0, runtime_ors=8.0, algorithm_runtime=100.0),
            _summary_row("B", "dccs_refc", route_cache=0.0, option_cache=0.1, option_reuse=0.1, runtime_osrm=7.0, runtime_ors=5.0, algorithm_runtime=70.0, refc_world_reuse=0.0),
            _summary_row("C", "voi", route_cache=0.0, option_cache=0.1, option_reuse=0.1, runtime_osrm=9.0, runtime_ors=6.0, algorithm_runtime=90.0, refc_world_reuse=0.0),
        ]
        if str(run_args.run_id).endswith("_hot"):
            summary_rows = [
                _summary_row("A", "dccs", route_cache=0.85, option_cache=0.8, option_reuse=0.8, runtime_osrm=6.0, runtime_ors=5.0, algorithm_runtime=60.0),
                _summary_row("B", "dccs_refc", route_cache=0.9, option_cache=0.9, option_reuse=0.9, runtime_osrm=3.0, runtime_ors=2.0, algorithm_runtime=30.0, refc_world_reuse=0.88),
                _summary_row("C", "voi", route_cache=0.92, option_cache=0.9, option_reuse=0.9, runtime_osrm=4.0, runtime_ors=3.0, algorithm_runtime=40.0, refc_world_reuse=0.9),
            ]
        return {"run_id": str(run_args.run_id), "summary_rows": summary_rows}

    monkeypatch.setattr(hot_module, "run_thesis_evaluation", fake_run_thesis_evaluation)
    cache_snapshots = iter(
        [
            {"stage": "before_clear", "route_cache": {"size": 1}},
            {"stage": "after_clear", "route_cache": {"size": 0}},
            {"stage": "after_cold", "route_cache": {"size": 1}},
            {"stage": "after_restore", "route_cache": {"size": 155}},
            {"stage": "after_hot", "route_cache": {"size": 155}},
        ]
    )
    monkeypatch.setattr(hot_module, "_cache_stats", lambda _client: next(cache_snapshots))

    result = hot_module.run_hot_rerun_benchmark(args, client=client)

    assert observed_run_ids == ["bench-1_cold", "bench-1_hot"]
    assert observed_cache_modes == ["cold", "preserve"]
    assert observed_cold_cache_scopes[0] == "hot_rerun_cold_source"
    assert observed_cold_cache_scopes[1] == "thesis_cold"
    assert observed_suite_roles == ["hot_rerun_cold_source", "hot_rerun"]
    assert ("DELETE", "/cache?scope=thesis_cold") in client.calls
    assert ("POST", "/cache/hot-rerun/restore") in client.calls
    assert result["hot_gate"]["all_green"] is True
    assert Path(result["comparison_json"]).exists()
    assert Path(result["comparison_csv"]).exists()
    assert Path(result["gate_json"]).exists()
    assert Path(result["report_path"]).exists()
    cold_metadata = json.loads((tmp_path / "artifacts" / "bench-1_cold" / "metadata.json").read_text(encoding="utf-8"))
    hot_metadata = json.loads((tmp_path / "artifacts" / "bench-1_hot" / "metadata.json").read_text(encoding="utf-8"))
    hot_manifest = json.loads((tmp_path / "artifacts" / "bench-1_hot" / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert cold_metadata["benchmark_phase"] == "cold_rerun_source"
    assert cold_metadata["pair_run_id"] == "bench-1"
    assert cold_metadata["paired_run_id"] == "bench-1_hot"
    assert cold_metadata["cache_reset_policy"] == "hot_rerun_cold_source"
    assert hot_metadata["benchmark_phase"] == "hot_rerun"
    assert hot_metadata["cache_carryover_expected"] is True
    assert hot_metadata["cache_stats_before_run"]["stage"] == "after_restore"
    assert hot_metadata["cache_stats_after_run"]["stage"] == "after_hot"
    assert hot_manifest["cache_stats_before_run"]["stage"] == "after_restore"
    assert hot_manifest["hot_rerun_comparison_artifact"] == "hot_rerun_vs_cold_comparison.json"
    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "restore_response=" in report_text
    assert "after_restore=" in report_text
