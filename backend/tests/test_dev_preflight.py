from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scripts.preflight_live_runtime as preflight_live_runtime
from app.model_data_errors import ModelDataError


def test_preflight_live_runtime_success(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight_live_runtime, "refresh_live_runtime_route_caches", lambda: None)
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_scenario_profiles",
        lambda: SimpleNamespace(version="v2", source="live", contexts={"ctx": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_fuel_price_snapshot",
        lambda: SimpleNamespace(source="live", as_of="2026-02-24T00:00:00Z", signature="abc123"),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_toll_tariffs",
        lambda: SimpleNamespace(source="live", rules=[{"id": "rule_1"}]),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_toll_segments_seed",
        lambda: [{"id": "seg_1"}],
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_stochastic_regimes",
        lambda: SimpleNamespace(source="live", regimes={"weekday": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_departure_profile",
        lambda: SimpleNamespace(source="live", profiles={"uk_default": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_uk_bank_holidays",
        lambda: frozenset({"2026-12-25"}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "resolve_carbon_price",
        lambda **kwargs: SimpleNamespace(price_per_kg=0.12),  # noqa: ARG005
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "apply_scope_emissions_adjustment",
        lambda **kwargs: 1.05,  # noqa: ARG005
    )

    output_path = tmp_path / "preflight.json"
    summary = preflight_live_runtime.run_preflight(output_path=output_path)
    assert summary["required_ok"] is True
    assert summary["required_failure_count"] == 0
    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["required_ok"] is True


def test_preflight_live_runtime_captures_required_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight_live_runtime, "refresh_live_runtime_route_caches", lambda: None)
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_scenario_profiles",
        lambda: (_ for _ in ()).throw(
            ModelDataError(
                reason_code="scenario_profile_unavailable",
                message="scenario payload unavailable",
            )
        ),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_fuel_price_snapshot",
        lambda: SimpleNamespace(source="live", as_of="2026-02-24T00:00:00Z", signature="abc123"),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_toll_tariffs",
        lambda: SimpleNamespace(source="live", rules=[{"id": "rule_1"}]),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_toll_segments_seed",
        lambda: [{"id": "seg_1"}],
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_stochastic_regimes",
        lambda: SimpleNamespace(source="live", regimes={"weekday": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_departure_profile",
        lambda: SimpleNamespace(source="live", profiles={"uk_default": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_uk_bank_holidays",
        lambda: frozenset({"2026-12-25"}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "resolve_carbon_price",
        lambda **kwargs: SimpleNamespace(price_per_kg=0.12),  # noqa: ARG005
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "apply_scope_emissions_adjustment",
        lambda **kwargs: 1.05,  # noqa: ARG005
    )

    output_path = tmp_path / "preflight.json"
    summary = preflight_live_runtime.run_preflight(output_path=output_path)
    assert summary["required_ok"] is False
    assert summary["required_failure_count"] >= 1
    failed = [check for check in summary["checks"] if not check["ok"]]
    assert failed
    assert failed[0]["error"]["reason_code"] == "scenario_profile_unavailable"


def test_dev_script_invokes_preflight_runtime_check() -> None:
    dev_script = Path(__file__).resolve().parents[2] / "scripts" / "dev.ps1"
    content = dev_script.read_text(encoding="utf-8")
    assert "preflight_live_runtime.py" in content
    assert "Invoke-StrictLivePreflight" in content
