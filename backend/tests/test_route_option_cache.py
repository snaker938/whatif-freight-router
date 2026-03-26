from __future__ import annotations

from app.models import (
    CostToggles,
    EmissionsContext,
    EvidenceProvenance,
    EvidenceSourceRecord,
    GeoJSONLineString,
    IncidentSimulatorConfig,
    RouteMetrics,
    RouteOption,
    ScenarioMode,
    StochasticConfig,
    TerrainProfile,
    WeatherImpactConfig,
)
import app.route_option_cache as route_option_cache


def _route_payload(*, snapshot_hash: str = "snapshot-1", evidence_tensor: dict | None = None) -> dict[str, object]:
    return {
        "id": "route_0",
        "distance": 12345.0,
        "duration": 678.0,
        "geometry": {
            "type": "LineString",
            "coordinates": [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        },
        "legs": [
            {
                "steps": [{"classes": ["motorway"]}, {"classes": ["primary"]}],
                "annotation": {
                    "distance": [100.0, 200.0],
                    "duration": [10.0, 20.0],
                },
            }
        ],
        "evidence_provenance": {
            "active_families": ["scenario", "toll"],
            "families": [
                {"family": "scenario", "source": "repo_local", "active": True, "confidence": 0.8},
                {"family": "toll", "source": "repo_local", "active": True, "confidence": 0.9},
            ],
        },
        "evidence_tensor": evidence_tensor or {"scenario": {"time": 0.7}, "toll": {"money": 0.6}},
        "snapshot_hash": snapshot_hash,
    }


def _route_option() -> RouteOption:
    return RouteOption(
        id="option_0",
        geometry=GeoJSONLineString(type="LineString", coordinates=[(0.0, 0.0), (1.0, 1.0)]),
        metrics=RouteMetrics(
            distance_km=12.3,
            duration_s=45.6,
            monetary_cost=7.89,
            emissions_kg=1.23,
            avg_speed_kmh=48.7,
        ),
        evidence_provenance=EvidenceProvenance(
            active_families=["scenario", "toll"],
            families=[
                EvidenceSourceRecord(family="scenario", source="repo_local", active=True, confidence=0.8),
                EvidenceSourceRecord(family="toll", source="repo_local", active=True, confidence=0.9),
            ],
        ),
    )


def test_route_option_cache_key_is_stable_for_equivalent_inputs() -> None:
    route_a = _route_payload()
    route_b = {
        "duration": route_a["duration"],
        "geometry": route_a["geometry"],
        "evidence_tensor": {"toll": {"money": 0.6}, "scenario": {"time": 0.7}},
        "evidence_provenance": {
            "families": list(route_a["evidence_provenance"]["families"]),
            "active_families": list(route_a["evidence_provenance"]["active_families"]),
        },
        "id": "route_0",
        "legs": route_a["legs"],
        "distance": route_a["distance"],
        "snapshot_hash": "snapshot-1",
    }

    key_a = route_option_cache.build_route_option_cache_key(
        route_a,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
    )
    key_b = route_option_cache.build_route_option_cache_key(
        route_b,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
    )

    assert key_a is not None
    assert key_a == key_b


def test_route_option_cache_key_changes_with_evidence_snapshot() -> None:
    route_a = _route_payload(snapshot_hash="snapshot-a")
    route_b = _route_payload(snapshot_hash="snapshot-b")

    key_a = route_option_cache.build_route_option_cache_key(
        route_a,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
    )
    key_b = route_option_cache.build_route_option_cache_key(
        route_b,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
    )

    assert key_a is not None
    assert key_b is not None
    assert key_a != key_b


def test_route_option_cache_key_changes_with_detail_level() -> None:
    route = _route_payload()

    full_key = route_option_cache.build_route_option_cache_key(
        route,
        detail_level="full",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
    )
    summary_key = route_option_cache.build_route_option_cache_key(
        route,
        detail_level="summary",
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
    )

    assert full_key is not None
    assert summary_key is not None
    assert full_key != summary_key


def test_route_option_core_cache_key_is_shared_across_detail_levels_but_separates_snapshots() -> None:
    route_a = _route_payload(snapshot_hash="snapshot-a")
    route_b = _route_payload(snapshot_hash="snapshot-b")

    key_a = route_option_cache.build_route_option_core_cache_key(
        route_a,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
    )
    key_a_repeat = route_option_cache.build_route_option_core_cache_key(
        route_a,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
    )
    key_b = route_option_cache.build_route_option_core_cache_key(
        route_b,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        terrain_profile="flat",
        stochastic=StochasticConfig(),
        emissions_context=EmissionsContext(),
        weather=WeatherImpactConfig(),
        incident_simulation=IncidentSimulatorConfig(),
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=1.0,
    )

    assert key_a is not None
    assert key_a_repeat == key_a
    assert key_b is not None
    assert key_b != key_a


def test_route_option_cache_key_separates_expected_value_and_robust_builds() -> None:
    route = _route_payload()

    expected_key = route_option_cache.build_route_option_cache_key(
        route,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        optimization_mode="expected_value",
        stochastic=StochasticConfig(enabled=False, sigma=0.08, samples=25),
    )
    robust_key = route_option_cache.build_route_option_cache_key(
        route,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
        optimization_mode="robust",
        stochastic=StochasticConfig(enabled=True, sigma=0.08, samples=25),
    )

    assert expected_key is not None
    assert robust_key is not None
    assert expected_key != robust_key


def test_route_option_cache_deep_copies_on_round_trip() -> None:
    cache = route_option_cache.RouteOptionCacheStore(ttl_s=60, max_entries=2, max_estimated_bytes=1_000_000)
    payload = route_option_cache.CachedRouteOptionBuild(option=_route_option(), estimated_build_ms=42.5)

    assert cache.set("route-option", payload) is True
    cached = cache.get("route-option")
    assert cached is not None
    cached.option.evidence_provenance.active_families.append("weather")
    cached.option.metrics.duration_s = 999.0

    second = cache.get("route-option")
    assert second is not None
    assert second.option.evidence_provenance.active_families == ["scenario", "toll"]
    assert second.option.metrics.duration_s == 45.6


def test_route_option_cache_is_bounded_and_evicts_oldest() -> None:
    cache = route_option_cache.RouteOptionCacheStore(ttl_s=60, max_entries=1, max_estimated_bytes=1_000_000)
    payload = route_option_cache.CachedRouteOptionBuild(option=_route_option(), estimated_build_ms=12.0)

    assert cache.set("first", payload) is True
    assert cache.set("second", payload) is True

    assert cache.get("first") is None
    assert cache.get("second") is not None
    assert cache.snapshot()["evictions"] == 1


def test_route_option_cache_bypasses_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(route_option_cache.settings, "route_option_cache_enabled", False)

    key = route_option_cache.build_route_option_cache_key(
        _route_payload(),
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(),
    )
    assert key is None
    assert route_option_cache.set_cached_route_option_build("disabled", route_option_cache.CachedRouteOptionBuild(option=_route_option())) is False
    assert route_option_cache.get_cached_route_option_build("disabled") is None
