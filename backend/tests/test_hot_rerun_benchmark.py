from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app as backend_app
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
    controller_reuse: float | None = None,
    refc_world_reuse: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
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
    if controller_reuse is not None:
        row["mean_voi_dccs_cache_hit_rate"] = controller_reuse
    return row


def _result_row(
    od_id: str,
    variant_id: str,
    pipeline_mode: str,
    *,
    route_id: str,
    terminal_type: str,
    certified: bool,
    certificate_winner_route_id: str | None,
    artifact_run_id: str,
    selected_final_route_source_label: str | None = None,
    selected_candidate_source_label: str | None = None,
    selected_route_signature: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "od_id": od_id,
        "variant_id": variant_id,
        "pipeline_mode": pipeline_mode,
        "route_id": route_id,
        "preference_terminal_type": terminal_type,
        "certified": certified,
        "certificate_winner_route_id": certificate_winner_route_id,
        "artifact_run_id": artifact_run_id,
    }
    if selected_final_route_source_label is not None:
        row["selected_final_route_source_label"] = selected_final_route_source_label
    if selected_candidate_source_label is not None:
        row["selected_candidate_source_label"] = selected_candidate_source_label
    if selected_route_signature is not None:
        row["selected_route_signature"] = selected_route_signature
    return row


def test_build_hot_rerun_comparison_reports_controller_reuse_from_voi_cache_hits() -> None:
    comparison = hot_module.build_hot_rerun_comparison(
        pair_run_id="pair-controller",
        cold_run_id="pair-controller_cold",
        hot_run_id="pair-controller_hot",
        cold_summary_rows=[
            _summary_row("A", "dccs", route_cache=0.0, option_cache=0.2, option_reuse=0.2, runtime_osrm=12.0, runtime_ors=9.0, algorithm_runtime=120.0),
            _summary_row("B", "dccs_refc", route_cache=0.0, option_cache=0.1, option_reuse=0.1, runtime_osrm=6.0, runtime_ors=4.0, algorithm_runtime=60.0, refc_world_reuse=0.0),
            _summary_row(
                "C",
                "voi",
                route_cache=0.0,
                option_cache=0.1,
                option_reuse=0.1,
                runtime_osrm=9.0,
                runtime_ors=7.0,
                algorithm_runtime=90.0,
                controller_reuse=0.25,
                refc_world_reuse=0.0,
            ),
        ],
        hot_summary_rows=[
            _summary_row("A", "dccs", route_cache=0.85, option_cache=0.8, option_reuse=0.8, runtime_osrm=8.0, runtime_ors=6.5, algorithm_runtime=90.0),
            _summary_row("B", "dccs_refc", route_cache=0.9, option_cache=0.9, option_reuse=0.9, runtime_osrm=3.0, runtime_ors=2.5, algorithm_runtime=40.0, refc_world_reuse=0.85),
            _summary_row(
                "C",
                "voi",
                route_cache=0.92,
                option_cache=0.88,
                option_reuse=0.88,
                runtime_osrm=5.0,
                runtime_ors=4.0,
                algorithm_runtime=50.0,
                controller_reuse=0.75,
                refc_world_reuse=0.9,
            ),
        ],
        cache_stats={"after_hot": {"route_cache": {"hits": 10}}},
    )

    c_row = next(row for row in comparison["comparison_rows"] if row["variant_id"] == "C")
    assert c_row["cold_mean_controller_reuse_rate"] == pytest.approx(0.25)
    assert c_row["hot_mean_controller_reuse_rate"] == pytest.approx(0.75)
    assert c_row["controller_reuse_rate_delta"] == pytest.approx(0.5)
    hot_summary_row = next(row for row in comparison["hot_summary_rows"] if row["variant_id"] == "C")
    assert hot_summary_row["mean_controller_reuse_rate"] == pytest.approx(0.75)
    assert hot_summary_row["controller_reuse_rate"] == pytest.approx(0.75)
    assert hot_summary_row["mean_voi_dccs_cache_hit_rate"] == pytest.approx(0.75)

    controller_reporting = comparison["hot_gate"]["controller_reuse_reporting"]
    assert controller_reporting == [
        {
            "metric": "mean_controller_reuse_rate",
            "variant_id": "C",
            "cold_value": 0.25,
            "hot_value": 0.75,
            "delta": 0.5,
            "cold_source_metric": "mean_voi_dccs_cache_hit_rate",
            "hot_source_metric": "mean_voi_dccs_cache_hit_rate",
        }
    ]

    report_text = hot_module._hot_rerun_report(comparison)
    assert "controller_reuse=0.75 (cold 0.25)" in report_text
    assert (
        "mean_controller_reuse_rate / C: cold=0.25 hot=0.75 delta=0.5 "
        "cold_source=mean_voi_dccs_cache_hit_rate hot_source=mean_voi_dccs_cache_hit_rate"
    ) in report_text


def test_build_hot_rerun_comparison_reports_parity_lcb_and_semantic_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(hot_module.settings, "out_dir", str(tmp_path))

    for artifact_run_id, certificate_lcb in (
        ("artifact-cold-1", 0.61),
        ("artifact-hot-1", 0.72),
        ("artifact-cold-2", 0.40),
        ("artifact-hot-2", 0.35),
    ):
        artifact_dir = hot_module.artifact_dir_for_run(artifact_run_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "certificate_summary.json").write_text(
            json.dumps(
                {
                    "selected_route_id": "route_0",
                    "certificate_lcb": certificate_lcb,
                }
            ),
            encoding="utf-8",
        )

    comparison = hot_module.build_hot_rerun_comparison(
        pair_run_id="pair-drift",
        cold_run_id="pair-drift_cold",
        hot_run_id="pair-drift_hot",
        cold_summary_rows=[
            _summary_row("A", "dccs", route_cache=0.0, option_cache=0.2, option_reuse=0.2, runtime_osrm=12.0, runtime_ors=9.0, algorithm_runtime=120.0),
            _summary_row("B", "dccs_refc", route_cache=0.0, option_cache=0.1, option_reuse=0.1, runtime_osrm=6.0, runtime_ors=4.0, algorithm_runtime=60.0, refc_world_reuse=0.0),
            _summary_row(
                "C",
                "voi",
                route_cache=0.0,
                option_cache=0.1,
                option_reuse=0.1,
                runtime_osrm=9.0,
                runtime_ors=7.0,
                algorithm_runtime=90.0,
                controller_reuse=0.25,
                refc_world_reuse=0.0,
            ),
        ],
        hot_summary_rows=[
            _summary_row("A", "dccs", route_cache=0.85, option_cache=0.8, option_reuse=0.8, runtime_osrm=8.0, runtime_ors=6.5, algorithm_runtime=90.0),
            _summary_row("B", "dccs_refc", route_cache=0.9, option_cache=0.9, option_reuse=0.9, runtime_osrm=3.0, runtime_ors=2.5, algorithm_runtime=40.0, refc_world_reuse=0.85),
            _summary_row(
                "C",
                "voi",
                route_cache=0.92,
                option_cache=0.88,
                option_reuse=0.88,
                runtime_osrm=5.0,
                runtime_ors=4.0,
                algorithm_runtime=50.0,
                controller_reuse=0.75,
                refc_world_reuse=0.9,
            ),
        ],
        cold_rows=[
            _result_row(
                "od-1",
                "C",
                "voi",
                route_id="route_0",
                terminal_type="certified_singleton",
                certified=True,
                certificate_winner_route_id="route_0",
                artifact_run_id="artifact-cold-1",
            ),
            _result_row(
                "od-2",
                "C",
                "voi",
                route_id="route_1",
                terminal_type="typed_abstention",
                certified=False,
                certificate_winner_route_id="route_1",
                artifact_run_id="artifact-cold-2",
            ),
        ],
        hot_rows=[
            _result_row(
                "od-1",
                "C",
                "voi",
                route_id="route_0",
                terminal_type="certified_singleton",
                certified=True,
                certificate_winner_route_id="route_0",
                artifact_run_id="artifact-hot-1",
            ),
            _result_row(
                "od-2",
                "C",
                "voi",
                route_id="route_2",
                terminal_type="certified_set",
                certified=True,
                certificate_winner_route_id="route_2",
                artifact_run_id="artifact-hot-2",
            ),
        ],
        cache_stats={"after_hot": {"route_cache": {"hits": 10}}},
    )

    assert comparison["hot_gate"]["all_green"] is False
    c_row = next(row for row in comparison["comparison_rows"] if row["variant_id"] == "C")
    assert c_row["hot_cold_parity_row_count"] == 2
    assert c_row["hot_cold_parity_match_count"] == 1
    assert c_row["hot_cold_parity_rate"] == pytest.approx(0.5)
    assert c_row["route_id_parity_rate"] == pytest.approx(0.5)
    assert c_row["terminal_type_parity_rate"] == pytest.approx(0.5)
    assert c_row["certified_flag_parity_rate"] == pytest.approx(0.5)
    assert c_row["certificate_winner_parity_rate"] == pytest.approx(0.5)
    assert c_row["semantic_drift_count"] == 1
    assert c_row["semantic_drift_rate"] == pytest.approx(0.5)
    assert c_row["certificate_lcb_available_row_count"] == 2
    assert c_row["certificate_lcb_unavailable_row_count"] == 0
    assert c_row["cold_mean_certificate_lcb"] == pytest.approx(0.505)
    assert c_row["hot_mean_certificate_lcb"] == pytest.approx(0.535)
    assert c_row["certificate_lcb_drift"] == pytest.approx(0.03)
    assert c_row["max_final_certificate_lcb_abs_drift"] == pytest.approx(0.11)

    parity_reporting = comparison["hot_gate"]["parity_reporting"]
    assert any(
        check["variant_id"] == "C"
        and check["metric"] == "hot_cold_parity_rate"
        and check["value"] == pytest.approx(0.5)
        for check in parity_reporting
    )
    lcb_reporting = comparison["hot_gate"]["lcb_drift_reporting"]
    assert any(
        check["variant_id"] == "C"
        and check["metric"] == "mean_certificate_lcb_drift"
        and check["delta"] == pytest.approx(0.03)
        and check["max_abs_delta"] == pytest.approx(0.11)
        and check["source_metric"] == "artifact:certificate_summary.json.certificate_lcb"
        for check in lcb_reporting
    )
    semantic_reporting = comparison["hot_gate"]["semantic_drift_reporting"]
    assert any(
        check["variant_id"] == "C"
        and check["metric"] == "semantic_drift_rate"
        and check["value"] == pytest.approx(0.5)
        and check["drift_count"] == 1
        for check in semantic_reporting
    )

    report_text = hot_module._hot_rerun_report(comparison)
    assert "## Parity And Drift" in report_text
    assert (
        "C (voi): parity_rate=0.5 route_id_parity_rate=0.5, terminal_type_parity_rate=0.5, "
        "certified_flag_parity_rate=0.5, certificate_winner_parity_rate=0.5, "
        "semantic_drift_rate=0.5, mean_certificate_lcb_drift=0.03 "
        "max_final_certificate_lcb_abs_drift=0.11 (available_rows=2/2)"
    ) in report_text
    assert (
        "hot_cold_parity_rate / C: value=0.5 matched_rows=2 parity_matches=1 "
        "route_id_parity_rate=0.5 terminal_type_parity_rate=0.5 certified_flag_parity_rate=0.5 "
        "certificate_winner_parity_rate=0.5"
    ) in report_text
    assert (
        "mean_certificate_lcb_drift / C: cold=0.505 hot=0.535 delta=0.03 "
        "max_abs_delta=0.11 available_rows=2 unavailable_rows=0 "
        "source=artifact:certificate_summary.json.certificate_lcb"
    ) in report_text
    assert "semantic_drift_rate / C: value=0.5 matched_rows=2 drift_count=1" in report_text


def test_certificate_lcb_falls_back_to_anytime_bound_from_certificate_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(hot_module.settings, "out_dir", str(tmp_path))

    artifact_dir = hot_module.artifact_dir_for_run("artifact-derived")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "certificate_summary.json").write_text(
        json.dumps(
            {
                "selected_route_id": "route_0",
                "selected_certificate": 1.0,
                "empirical_selected_certificate": 1.0,
                "world_count": 87,
                "selector_config": {"threshold": 0.83},
            }
        ),
        encoding="utf-8",
    )

    value, source_metric = hot_module._certificate_lcb_from_row_artifact(
        {"artifact_run_id": "artifact-derived"},
        {},
    )
    expected_lcb, _ = hot_module.anytime_hoeffding_interval(
        87,
        87,
        delta=hot_module.DEFAULT_CONFIDENCE_DELTA,
    )

    assert value == pytest.approx(round(expected_lcb, 6))
    assert source_metric == hot_module.DERIVED_CERTIFICATE_LCB_SOURCE


def test_build_hot_rerun_comparison_uses_stable_route_identity_when_route_ids_renumber() -> None:
    comparison = hot_module.build_hot_rerun_comparison(
        pair_run_id="pair-stable-identity",
        cold_run_id="pair-stable-identity_cold",
        hot_run_id="pair-stable-identity_hot",
        cold_summary_rows=[
            _summary_row(
                "C",
                "voi",
                route_cache=0.0,
                option_cache=0.1,
                option_reuse=0.1,
                runtime_osrm=9.0,
                runtime_ors=7.0,
                algorithm_runtime=90.0,
                controller_reuse=0.25,
                refc_world_reuse=0.0,
            ),
        ],
        hot_summary_rows=[
            _summary_row(
                "C",
                "voi",
                route_cache=0.92,
                option_cache=0.88,
                option_reuse=0.88,
                runtime_osrm=5.0,
                runtime_ors=4.0,
                algorithm_runtime=50.0,
                controller_reuse=0.75,
                refc_world_reuse=0.9,
            ),
        ],
        cold_rows=[
            _result_row(
                "od-1",
                "C",
                "voi",
                route_id="route_0",
                terminal_type="certified_singleton",
                certified=True,
                certificate_winner_route_id="route_0",
                artifact_run_id="artifact-cold-1",
                selected_final_route_source_label="graph_family:sig-route-a:osrm_refined",
                selected_route_signature="cold-geometry-signature",
            ),
        ],
        hot_rows=[
            _result_row(
                "od-1",
                "C",
                "voi",
                route_id="route_2",
                terminal_type="certified_singleton",
                certified=True,
                certificate_winner_route_id="route_2",
                artifact_run_id="artifact-hot-1",
                selected_final_route_source_label="graph_family:sig-route-a:osrm_refined",
                selected_route_signature="hot-geometry-signature",
            ),
        ],
        cache_stats={"after_hot": {"route_cache": {"hits": 10}}},
    )

    c_row = next(row for row in comparison["comparison_rows"] if row["variant_id"] == "C")
    assert c_row["hot_cold_parity_row_count"] == 1
    assert c_row["hot_cold_parity_match_count"] == 1
    assert c_row["hot_cold_parity_rate"] == pytest.approx(1.0)
    assert c_row["route_id_parity_rate"] == pytest.approx(1.0)
    assert c_row["certificate_winner_parity_rate"] == pytest.approx(1.0)
    assert c_row["semantic_drift_count"] == 0
    assert c_row["semantic_drift_rate"] == pytest.approx(0.0)


def test_build_hot_rerun_comparison_prefers_stable_candidate_source_over_final_stage_signature() -> None:
    comparison = hot_module.build_hot_rerun_comparison(
        pair_run_id="pair-stable-candidate-source",
        cold_run_id="pair-stable-candidate-source_cold",
        hot_run_id="pair-stable-candidate-source_hot",
        cold_summary_rows=[
            _summary_row(
                "C",
                "voi",
                route_cache=0.0,
                option_cache=0.1,
                option_reuse=0.1,
                runtime_osrm=9.0,
                runtime_ors=7.0,
                algorithm_runtime=90.0,
                controller_reuse=0.25,
                refc_world_reuse=0.0,
            ),
        ],
        hot_summary_rows=[
            _summary_row(
                "C",
                "voi",
                route_cache=0.92,
                option_cache=0.88,
                option_reuse=0.88,
                runtime_osrm=5.0,
                runtime_ors=4.0,
                algorithm_runtime=50.0,
                controller_reuse=0.75,
                refc_world_reuse=0.9,
            ),
        ],
        cold_rows=[
            _result_row(
                "od-1",
                "C",
                "voi",
                route_id="route_0",
                terminal_type="typed_abstention",
                certified=False,
                certificate_winner_route_id="route_0",
                artifact_run_id="artifact-cold-1",
                selected_final_route_source_label="graph_family:candidate-a:osrm_refined",
                selected_candidate_source_label="fallback:alternatives:direct_k_raw_fallback",
                selected_route_signature="cold-geometry-signature",
            ),
        ],
        hot_rows=[
            _result_row(
                "od-1",
                "C",
                "voi",
                route_id="route_0",
                terminal_type="typed_abstention",
                certified=False,
                certificate_winner_route_id="route_0",
                artifact_run_id="artifact-hot-1",
                selected_final_route_source_label="fallback:alternatives:direct_k_raw_fallback",
                selected_candidate_source_label="fallback:alternatives:direct_k_raw_fallback",
                selected_route_signature="hot-geometry-signature",
            ),
        ],
        cache_stats={"after_hot": {"route_cache": {"hits": 10}}},
    )

    c_row = next(row for row in comparison["comparison_rows"] if row["variant_id"] == "C")
    assert c_row["hot_cold_parity_row_count"] == 1
    assert c_row["hot_cold_parity_match_count"] == 1
    assert c_row["hot_cold_parity_rate"] == pytest.approx(1.0)
    assert c_row["route_id_parity_rate"] == pytest.approx(1.0)
    assert c_row["certificate_winner_parity_rate"] == pytest.approx(1.0)
    assert c_row["semantic_drift_count"] == 0
    assert c_row["semantic_drift_rate"] == pytest.approx(0.0)


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
                "hot_rerun_route_state_cache_checkpoint": {"size": 2, "hits": 0, "misses": 0},
                "hot_rerun_voi_dccs_cache_checkpoint": {"size": 5, "hits": 0, "misses": 0},
                "route_option_cache": {"hits": 8, "misses": 2},
            }
        )

    def delete(self, path: str) -> _DummyResponse:
        self.calls.append(("DELETE", path))
        return _DummyResponse({"route_cache": 1, "route_option_cache": 1})

    def post(self, path: str) -> _DummyResponse:
        self.calls.append(("POST", path))
        return _DummyResponse(
            {
                "restored": 10,
                "checkpoint_size": 10,
                "restored_route_cache": 3,
                "route_checkpoint_size": 3,
                "restored_certification_cache": 0,
                "certification_checkpoint_size": 0,
                "restored_route_state_cache": 2,
                "route_state_checkpoint_size": 2,
                "restored_voi_dccs_cache": 5,
                "voi_dccs_checkpoint_size": 5,
            }
        )


class _FakeInProcessClient:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[str, str]] = []

    def __enter__(self) -> "_FakeInProcessClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def get(self, path: str) -> _DummyResponse:
        self.calls.append(("GET", path))
        return _DummyResponse({"client_label": self.label, "path": path})

    def delete(self, path: str) -> _DummyResponse:
        self.calls.append(("DELETE", path))
        return _DummyResponse({"cleared": 1})

    def post(self, path: str) -> _DummyResponse:
        self.calls.append(("POST", path))
        return _DummyResponse(
            {
                "restored": 10,
                "checkpoint_size": 10,
                "restored_route_cache": 3,
                "route_checkpoint_size": 3,
                "restored_certification_cache": 0,
                "certification_checkpoint_size": 0,
                "restored_route_state_cache": 2,
                "route_state_checkpoint_size": 2,
                "restored_voi_dccs_cache": 5,
                "voi_dccs_checkpoint_size": 5,
            }
        )


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
            _summary_row("C", "voi", route_cache=0.0, option_cache=0.1, option_reuse=0.1, runtime_osrm=9.0, runtime_ors=6.0, algorithm_runtime=90.0, controller_reuse=0.25, refc_world_reuse=0.0),
        ]
        if str(run_args.run_id).endswith("_hot"):
            summary_rows = [
                _summary_row("A", "dccs", route_cache=0.85, option_cache=0.8, option_reuse=0.8, runtime_osrm=6.0, runtime_ors=5.0, algorithm_runtime=60.0),
                _summary_row("B", "dccs_refc", route_cache=0.9, option_cache=0.9, option_reuse=0.9, runtime_osrm=3.0, runtime_ors=2.0, algorithm_runtime=30.0, refc_world_reuse=0.88),
                _summary_row("C", "voi", route_cache=0.92, option_cache=0.9, option_reuse=0.9, runtime_osrm=4.0, runtime_ors=3.0, algorithm_runtime=40.0, controller_reuse=0.75, refc_world_reuse=0.9),
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
    assert result["cache_stats"]["restore_response"]["restored_route_state_cache"] == 2
    assert result["cache_stats"]["restore_response"]["restored_voi_dccs_cache"] == 5
    controller_reporting = result["comparison"]["hot_gate"]["controller_reuse_reporting"]
    assert controller_reporting == [
        {
            "metric": "mean_controller_reuse_rate",
            "variant_id": "C",
            "cold_value": 0.25,
            "hot_value": 0.75,
            "delta": 0.5,
            "cold_source_metric": "mean_voi_dccs_cache_hit_rate",
            "hot_source_metric": "mean_voi_dccs_cache_hit_rate",
        }
    ]
    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "restore_response=" in report_text
    assert "after_restore=" in report_text
    assert "mean_controller_reuse_rate / C" in report_text


def test_run_hot_rerun_benchmark_reopens_in_process_client_for_hot_phase(
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
            "bench-in-process",
            "--in-process-backend",
        ]
    )
    monkeypatch.setattr(hot_module, "in_process_backend_runtime_profile", lambda: nullcontext())
    created_clients: list[_FakeInProcessClient] = []

    def fake_test_client(_app):
        client_obj = _FakeInProcessClient(f"client-{len(created_clients)}")
        created_clients.append(client_obj)
        return client_obj

    monkeypatch.setattr(hot_module, "TestClient", fake_test_client)
    monkeypatch.setattr(
        hot_module,
        "run_thesis_evaluation",
        lambda run_args, *, client: _fake_in_process_hot_rerun_payload(tmp_path, run_args),
    )

    result = hot_module.run_hot_rerun_benchmark(args)

    assert len(created_clients) == 2
    assert created_clients[0] is not created_clients[1]
    assert ("DELETE", "/cache?scope=thesis_cold") in created_clients[0].calls
    assert ("POST", "/cache/hot-rerun/restore") in created_clients[1].calls
    assert Path(result["comparison_json"]).exists()
    assert Path(result["comparison_csv"]).exists()
    assert Path(result["gate_json"]).exists()
    assert Path(result["report_path"]).exists()
    assert result["hot_gate"]["all_green"] is True
    assert result["comparison"]["hot_gate"]["controller_reuse_reporting"] == [
        {
            "metric": "mean_controller_reuse_rate",
            "variant_id": "C",
            "cold_value": 0.25,
            "hot_value": 0.75,
            "delta": 0.5,
            "cold_source_metric": "mean_voi_dccs_cache_hit_rate",
            "hot_source_metric": "mean_voi_dccs_cache_hit_rate",
        }
    ]


def test_hot_rerun_restore_exposes_certification_checkpoint_stats(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "require_role", lambda *args, **kwargs: None)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        main_module,
        "restore_checkpointed_route_cache",
        lambda *, clear_first=False: calls.append(("route_restore", clear_first)) or 4,
    )
    monkeypatch.setattr(
        main_module,
        "route_cache_checkpoint_stats",
        lambda: {"size": 4, "hits": 0, "misses": 0},
    )
    monkeypatch.setattr(
        main_module,
        "restore_checkpointed_certification_cache",
        lambda *, clear_first=False: calls.append(("cert_restore", clear_first)) or 7,
    )
    monkeypatch.setattr(
        main_module,
        "certification_cache_checkpoint_stats",
        lambda: {"size": 7, "hits": 0, "misses": 0},
    )
    monkeypatch.setattr(
        main_module,
        "restore_checkpointed_route_state_cache",
        lambda *, clear_first=False: calls.append(("route_state_restore", clear_first)) or 2,
    )
    monkeypatch.setattr(
        main_module,
        "route_state_cache_checkpoint_stats",
        lambda: {"size": 2, "hits": 0, "misses": 0},
    )
    monkeypatch.setattr(
        main_module,
        "restore_checkpointed_voi_dccs_cache",
        lambda *, clear_first=False: calls.append(("voi_dccs_restore", clear_first)) or 3,
    )
    monkeypatch.setattr(
        main_module,
        "voi_dccs_cache_checkpoint_stats",
        lambda: {"size": 3, "hits": 0, "misses": 0},
    )

    with TestClient(backend_app) as client:
        stats_response = client.get("/cache/stats")
        restore_response = client.post("/cache/hot-rerun/restore")

    stats_payload = stats_response.json()
    restore_payload = restore_response.json()

    assert stats_payload["hot_rerun_certification_cache_checkpoint"]["size"] == 7
    assert restore_payload["checkpoint_size"] == 16
    assert restore_payload["restored"] == 16
    assert restore_payload["restored_route_cache"] == 4
    assert restore_payload["route_checkpoint_size"] == 4
    assert restore_payload["certification_checkpoint_size"] == 7
    assert restore_payload["restored_certification_cache"] == 7
    assert restore_payload["route_state_checkpoint_size"] == 2
    assert restore_payload["restored_route_state_cache"] == 2
    assert restore_payload["voi_dccs_checkpoint_size"] == 3
    assert restore_payload["restored_voi_dccs_cache"] == 3
    assert stats_payload["hot_rerun_route_state_cache_checkpoint"]["size"] == 2
    assert stats_payload["hot_rerun_voi_dccs_cache_checkpoint"]["size"] == 3
    assert calls == [
        ("route_restore", False),
        ("cert_restore", False),
        ("route_state_restore", False),
        ("voi_dccs_restore", False),
    ]


def _fake_in_process_hot_rerun_payload(tmp_path: Path, run_args) -> dict[str, object]:
    artifact_dir = tmp_path / "artifacts" / str(run_args.run_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    is_hot = str(run_args.run_id).endswith("_hot")
    metadata = {
        "run_id": str(run_args.run_id),
        "cache_reset_policy": str(getattr(run_args, "cold_cache_scope", "none")) if str(getattr(run_args, "cache_mode", "")) == "cold" else "none",
    }
    for name in ("metadata.json", "evaluation_manifest.json"):
        (artifact_dir / name).write_text(json.dumps(metadata), encoding="utf-8")
    summary_rows = [
        _summary_row(
            "A",
            "dccs",
            route_cache=0.0 if not is_hot else 0.9,
            option_cache=0.2 if not is_hot else 0.85,
            option_reuse=0.2 if not is_hot else 0.85,
            runtime_osrm=12.0 if not is_hot else 8.0,
            runtime_ors=9.0 if not is_hot else 6.0,
            algorithm_runtime=100.0,
        ),
        _summary_row(
            "B",
            "dccs_refc",
            route_cache=0.0 if not is_hot else 0.92,
            option_cache=0.1 if not is_hot else 0.88,
            option_reuse=0.1 if not is_hot else 0.88,
            runtime_osrm=6.0 if not is_hot else 3.0,
            runtime_ors=4.0 if not is_hot else 2.0,
            algorithm_runtime=60.0,
            refc_world_reuse=0.0 if not is_hot else 0.85,
        ),
        _summary_row(
            "C",
            "voi",
            route_cache=0.0 if not is_hot else 0.95,
            option_cache=0.1 if not is_hot else 0.9,
            option_reuse=0.1 if not is_hot else 0.9,
            runtime_osrm=9.0 if not is_hot else 4.0,
            runtime_ors=7.0 if not is_hot else 3.0,
            algorithm_runtime=90.0,
            controller_reuse=0.25 if not is_hot else 0.75,
            refc_world_reuse=0.0 if not is_hot else 0.9,
        ),
    ]
    return {"run_id": str(run_args.run_id), "summary_rows": summary_rows}
