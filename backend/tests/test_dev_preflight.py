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
        "_osrm_engine_smoke_details",
        lambda: {"base_url": "http://localhost:5000", "profile": "driving", "distance_m": 1000.0, "duration_s": 100.0},
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "_ors_engine_smoke_details",
        lambda: {
            "base_url": "http://localhost:8082/ors",
            "profile": "driving-hgv",
            "distance_m": 1200.0,
            "duration_s": 110.0,
            "identity_status": "graph_identity_verified",
            "manifest_hash": "sha256:ors",
        },
    )
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
        "load_live_scenario_context",
        lambda **kwargs: SimpleNamespace(  # noqa: ARG005
            as_of_utc="2026-02-24T00:00:00Z",
            coverage={"overall": 1.0},
            source_set={"webtris": "ok", "traffic_england": "ok", "dft_counts": "ok", "open_meteo": "ok"},
        ),
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
        "_osrm_engine_smoke_details",
        lambda: {"base_url": "http://localhost:5000", "profile": "driving", "distance_m": 1000.0, "duration_s": 100.0},
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "_ors_engine_smoke_details",
        lambda: {
            "base_url": "http://localhost:8082/ors",
            "profile": "driving-hgv",
            "distance_m": 1200.0,
            "duration_s": 110.0,
            "identity_status": "graph_identity_verified",
            "manifest_hash": "sha256:ors",
        },
    )
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


def test_preflight_live_runtime_uses_uncached_fuel_loader(tmp_path: Path, monkeypatch) -> None:
    state = {"cached_calls": 0, "uncached_calls": 0}

    def _uncached_loader():
        state["uncached_calls"] += 1
        raise ModelDataError(
            reason_code="fuel_price_source_unavailable",
            message="stale uncached fuel snapshot",
            details={"as_of_utc": "2026-03-16T00:00:00Z", "max_age_days": 7},
        )

    def _cached_loader():
        state["cached_calls"] += 1
        return SimpleNamespace(source="cached", as_of="2026-03-16T00:00:00Z", signature="cached")

    _cached_loader.__wrapped__ = _uncached_loader

    monkeypatch.setattr(preflight_live_runtime, "refresh_live_runtime_route_caches", lambda: None)
    monkeypatch.setattr(
        preflight_live_runtime,
        "_osrm_engine_smoke_details",
        lambda: {"base_url": "http://localhost:5000", "profile": "driving", "distance_m": 1000.0, "duration_s": 100.0},
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "_ors_engine_smoke_details",
        lambda: {
            "base_url": "http://localhost:8082/ors",
            "profile": "driving-hgv",
            "distance_m": 1200.0,
            "duration_s": 110.0,
            "identity_status": "graph_identity_verified",
            "manifest_hash": "sha256:ors",
        },
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_scenario_profiles",
        lambda: SimpleNamespace(version="v2", source="live", calibration_basis="empirical", mode_observation_source="observed", contexts={"ctx": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_live_scenario_context",
        lambda **kwargs: SimpleNamespace(  # noqa: ARG005
            as_of_utc="2026-03-24T00:00:00Z",
            coverage={"overall": 1.0},
            source_set={"webtris": "ok"},
        ),
    )
    monkeypatch.setattr(preflight_live_runtime, "load_fuel_price_snapshot", _cached_loader)
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_toll_tariffs",
        lambda: SimpleNamespace(source="live", rules=[{"id": "rule_1"}]),
    )
    monkeypatch.setattr(preflight_live_runtime, "load_toll_segments_seed", lambda: [{"id": "seg_1"}])
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_stochastic_regimes",
        lambda: SimpleNamespace(source="live", calibration_basis="empirical", regimes={"weekday": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_departure_profile",
        lambda: SimpleNamespace(source="live", calibration_basis="empirical", profiles={"uk_default": {}}),
    )
    monkeypatch.setattr(preflight_live_runtime, "load_uk_bank_holidays", lambda: frozenset({"2026-12-25"}))
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

    summary = preflight_live_runtime.run_preflight(output_path=tmp_path / "preflight.json")

    failed = [check for check in summary["checks"] if not check["ok"]]
    assert summary["required_ok"] is False
    assert any(item["name"] == "fuel_snapshot" for item in failed)
    assert state["uncached_calls"] == 1
    assert state["cached_calls"] == 0


def test_preflight_live_runtime_rejects_proxy_provenance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight_live_runtime.settings, "live_source_policy", "strict_external")
    monkeypatch.setattr(preflight_live_runtime, "refresh_live_runtime_route_caches", lambda: None)
    monkeypatch.setattr(
        preflight_live_runtime,
        "_osrm_engine_smoke_details",
        lambda: {"base_url": "http://localhost:5000", "profile": "driving", "distance_m": 1000.0, "duration_s": 100.0},
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "_ors_engine_smoke_details",
        lambda: {
            "base_url": "http://localhost:8082/ors",
            "profile": "driving-hgv",
            "distance_m": 1200.0,
            "duration_s": 110.0,
            "identity_status": "graph_identity_verified",
            "manifest_hash": "sha256:ors",
        },
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_scenario_profiles",
        lambda: SimpleNamespace(
            version="v2",
            source="live",
            calibration_basis="empirical_live_fit",
            mode_observation_source="observed_mode_labels",
            contexts={"ctx": {}},
        ),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_fuel_price_snapshot",
        lambda: SimpleNamespace(source="live", as_of="2026-02-24T00:00:00Z", signature="abc123"),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_live_scenario_context",
        lambda **kwargs: SimpleNamespace(  # noqa: ARG005
            as_of_utc="2026-02-24T00:00:00Z",
            coverage={"overall": 1.0},
            source_set={"webtris": "ok", "traffic_england": "ok", "dft_counts": "ok", "open_meteo": "ok"},
        ),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_toll_tariffs",
        lambda: SimpleNamespace(
            source="live_runtime:toll_tariffs",
            rules=[SimpleNamespace(rule_id="proxy_rule_0001", operator="public_proxy_operator")],
        ),
    )
    monkeypatch.setattr(preflight_live_runtime, "load_toll_segments_seed", lambda: [{"id": "seg_1"}])
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_stochastic_regimes",
        lambda: SimpleNamespace(source="live", calibration_basis="empirical", regimes={"weekday": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_departure_profile",
        lambda: SimpleNamespace(source="live", calibration_basis="empirical", profiles={"uk_default": {}}),
    )
    monkeypatch.setattr(preflight_live_runtime, "load_uk_bank_holidays", lambda: frozenset({"2026-12-25"}))
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

    summary = preflight_live_runtime.run_preflight(output_path=tmp_path / "preflight.json")
    failed = [check for check in summary["checks"] if not check["ok"]]

    assert summary["required_ok"] is False
    assert any(item["error"]["reason_code"] == "strict_source_provenance_invalid" for item in failed)


def test_preflight_live_runtime_reports_repo_local_bindings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight_live_runtime.settings, "live_source_policy", "repo_local_fresh")
    monkeypatch.setattr(
        preflight_live_runtime.settings,
        "live_scenario_coefficient_url",
        "https://example.invalid/scenario_profiles_uk.json",
    )
    monkeypatch.setattr(
        preflight_live_runtime.settings,
        "live_bank_holidays_url",
        "https://www.gov.uk/bank-holidays.json",
    )
    monkeypatch.setattr(preflight_live_runtime, "refresh_live_runtime_route_caches", lambda: None)
    monkeypatch.setattr(
        preflight_live_runtime,
        "_osrm_engine_smoke_details",
        lambda: {"base_url": "http://localhost:5000", "profile": "driving", "distance_m": 1000.0, "duration_s": 100.0},
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "_ors_engine_smoke_details",
        lambda: {
            "base_url": "http://localhost:8082/ors",
            "profile": "driving-hgv",
            "distance_m": 1200.0,
            "duration_s": 110.0,
            "identity_status": "graph_identity_verified",
            "manifest_hash": "sha256:ors",
        },
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_scenario_profiles",
        lambda: SimpleNamespace(version="v2", source="repo_local:scenario_profiles_uk.json", contexts={"ctx": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_fuel_price_snapshot",
        lambda: SimpleNamespace(
            source="repo_local:fuel_prices_uk.json",
            as_of="2026-02-24T00:00:00Z",
            signature="abc123",
        ),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_live_scenario_context",
        lambda **kwargs: SimpleNamespace(  # noqa: ARG005
            as_of_utc="2026-02-24T00:00:00Z",
            coverage={"overall": 1.0},
            source_set={
                "webtris": "repo_local:scenario_profiles_context",
                "traffic_england": "repo_local:scenario_profiles_context",
                "dft_counts": "repo_local:scenario_profiles_context",
                "open_meteo": "repo_local:scenario_profiles_context",
            },
        ),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_toll_tariffs",
        lambda: SimpleNamespace(source="repo_local:toll_tariffs_uk.json", rules=[{"id": "rule_1"}]),
    )
    monkeypatch.setattr(preflight_live_runtime, "load_toll_segments_seed", lambda: [{"id": "seg_1"}])
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_stochastic_regimes",
        lambda: SimpleNamespace(source="repo_local:stochastic_regimes_uk.json", regimes={"weekday": {}}),
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "load_departure_profile",
        lambda: SimpleNamespace(source="repo_local:departure_profiles_uk.json", profiles={"uk_default": {}}),
    )
    monkeypatch.setattr(preflight_live_runtime, "load_uk_bank_holidays", lambda: frozenset({"2026-12-25"}))
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

    summary = preflight_live_runtime.run_preflight(output_path=tmp_path / "preflight.json")

    assert summary["required_ok"] is True
    assert summary["configured_urls"]["scenario"] == "backend/assets/uk/scenario_profiles_uk.json"
    assert summary["configured_urls"]["fuel"] == "backend/assets/uk/fuel_prices_uk.json"
    assert summary["configured_urls"]["bank_holidays"] == "https://www.gov.uk/bank-holidays.json"
    assert summary["configured_remote_urls"]["scenario"] == "https://example.invalid/scenario_profiles_uk.json"


def test_preflight_live_runtime_fails_when_ors_engine_smoke_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight_live_runtime, "refresh_live_runtime_route_caches", lambda: None)
    monkeypatch.setattr(
        preflight_live_runtime,
        "_osrm_engine_smoke_details",
        lambda: {"base_url": "http://localhost:5000", "profile": "driving", "distance_m": 1000.0, "duration_s": 100.0},
    )
    monkeypatch.setattr(
        preflight_live_runtime,
        "_ors_engine_smoke_details",
        lambda: (_ for _ in ()).throw(
            ModelDataError(
                reason_code="ors_engine_unreachable",
                message="ORS engine smoke request failed.",
                details={},
            )
        ),
    )
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
        "load_live_scenario_context",
        lambda **kwargs: SimpleNamespace(  # noqa: ARG005
            as_of_utc="2026-02-24T00:00:00Z",
            coverage={"overall": 1.0},
            source_set={"webtris": "ok", "traffic_england": "ok", "dft_counts": "ok", "open_meteo": "ok"},
        ),
    )
    monkeypatch.setattr(preflight_live_runtime, "load_toll_tariffs", lambda: SimpleNamespace(source="live", rules=[{"id": "rule_1"}]))
    monkeypatch.setattr(preflight_live_runtime, "load_toll_segments_seed", lambda: [{"id": "seg_1"}])
    monkeypatch.setattr(preflight_live_runtime, "load_stochastic_regimes", lambda: SimpleNamespace(source="live", regimes={"weekday": {}}))
    monkeypatch.setattr(preflight_live_runtime, "load_departure_profile", lambda: SimpleNamespace(source="live", profiles={"uk_default": {}}))
    monkeypatch.setattr(preflight_live_runtime, "load_uk_bank_holidays", lambda: frozenset({"2026-12-25"}))
    monkeypatch.setattr(preflight_live_runtime, "resolve_carbon_price", lambda **kwargs: SimpleNamespace(price_per_kg=0.12))  # noqa: ARG005
    monkeypatch.setattr(preflight_live_runtime, "apply_scope_emissions_adjustment", lambda **kwargs: 1.05)  # noqa: ARG005

    summary = preflight_live_runtime.run_preflight(output_path=tmp_path / "preflight.json")

    assert summary["required_ok"] is False
    failed = {check["name"]: check for check in summary["checks"] if not check["ok"]}
    assert failed["ors_engine_smoke"]["error"]["reason_code"] == "ors_engine_unreachable"


def test_dev_script_invokes_preflight_runtime_check() -> None:
    dev_script = Path(__file__).resolve().parents[2] / "scripts" / "dev.ps1"
    content = dev_script.read_text(encoding="utf-8")
    assert "preflight_live_runtime.py" in content
    assert "Invoke-StrictLivePreflight" in content
