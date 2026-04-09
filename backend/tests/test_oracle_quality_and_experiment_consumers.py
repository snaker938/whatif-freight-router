from __future__ import annotations

import json

from app import experiment_store, oracle_quality_store
from app.models import ExperimentBundle, ScenarioCompareRequest
from app.models import LatLng
from app.scenario import ScenarioMode
from app.settings import settings


def test_replay_oracle_payload_includes_trace_metadata_and_writes_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path / "out"))
    payload = oracle_quality_store.compute_replay_oracle_payload(
        [
            {
                "source": "voi:run-1",
                "replay_oracle_summary": {
                    "support_flag": True,
                    "support_status": "supported",
                    "fidelity_class": "fully_audited",
                    "terminal_type": "certified_singleton",
                    "replay_regret": 0.125,
                    "predicted_certificate_lift": 0.2,
                    "realized_certificate_lift": 0.15,
                    "predicted_gap_lift": 0.4,
                    "realized_gap_lift": 0.35,
                },
            },
            {
                "source": "voi:run-1",
                "support_flag": False,
                "support_status": "unsupported",
                "fidelity_class": "proxy_only",
                "terminal_type": "typed_abstention",
                "replay_regret": 0.25,
            },
        ]
    )

    assert payload["total_records"] == 2
    row = payload["sources"][0]
    assert row["source"] == "voi:run-1"
    assert row["support_true_count"] == 1
    assert row["support_false_count"] == 1
    assert row["terminal_type_counts"]["certified_singleton"] == 1
    assert row["terminal_type_counts"]["typed_abstention"] == 1
    assert row["fidelity_class_counts"]["fully_audited"] == 1
    assert row["fidelity_class_counts"]["proxy_only"] == 1
    assert row["mean_replay_regret"] == 0.1875
    assert row["mean_predicted_certificate_lift"] == 0.2
    assert row["mean_realized_gap_lift"] == 0.35

    summary_path, csv_path = oracle_quality_store.write_replay_oracle_artifacts(payload)
    assert summary_path.exists()
    assert csv_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["sources"][0]["support_true_count"] == 1
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "support_true_count" in csv_text
    assert "mean_replay_regret" in csv_text


def test_experiment_inventory_payload_exposes_pipeline_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path / "out"))
    request = ScenarioCompareRequest(
        origin=LatLng(lat=52.0, lon=-1.0),
        destination=LatLng(lat=51.5, lon=-0.1),
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.PARTIAL_SHARING,
        optimization_mode="robust",
        terrain_profile="hilly",
        pareto_method="epsilon_constraint",
        max_alternatives=12,
        risk_aversion=1.5,
    )
    bundle = ExperimentBundle(
        id="exp-1",
        name="experiment one",
        description="test",
        request=request,
        created_at="2026-04-09T00:00:00Z",
        updated_at="2026-04-09T00:10:00Z",
    )

    payload = experiment_store.compute_experiment_inventory_payload([bundle])
    assert payload["total_experiments"] == 1
    assert payload["scenario_mode_counts"]["partial_sharing"] == 1
    assert payload["optimization_mode_counts"]["robust"] == 1
    assert payload["terrain_profile_counts"]["hilly"] == 1
    assert payload["pareto_method_counts"]["epsilon_constraint"] == 1
    row = payload["rows"][0]
    assert row["scenario_mode"] == "partial_sharing"
    assert row["optimization_mode"] == "robust"
    assert row["terrain_profile"] == "hilly"
    assert row["pareto_method"] == "epsilon_constraint"
    assert row["max_alternatives"] == 12
    assert row["risk_aversion"] == 1.5

    summary_path, csv_path = experiment_store.write_experiment_inventory_artifacts(payload)
    assert summary_path.exists()
    assert csv_path.exists()
    assert "scenario_mode_counts" in summary_path.read_text(encoding="utf-8")
    assert "optimization_mode" in csv_path.read_text(encoding="utf-8")
