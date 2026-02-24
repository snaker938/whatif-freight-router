from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
TESTS_DIR = Path(__file__).resolve().parent


APP_MODULE_TO_TEST_FILES = {
    "__init__.py": ["test_app_package_and_errors.py", "test_app_smoke_all.py"],
    "calibration_loader.py": ["test_cost_model.py", "test_terrain_runtime_budget.py"],
    "carbon_model.py": ["test_cost_model.py", "test_emissions_context.py"],
    "departure_profile.py": ["test_departure_profile_v2.py", "test_departure_optimize.py"],
    "experiment_store.py": ["test_experiments.py"],
    "fuel_energy_model.py": ["test_cost_model.py", "test_emissions_context.py"],
    "incident_simulator.py": ["test_incident_simulator.py"],
    "k_shortest.py": ["test_k_shortest.py"],
    "live_data_sources.py": [
        "test_rbac_logging_live_sources.py",
        "test_scripts_fetchers_extended.py",
        "test_live_retry_policy.py",
    ],
    "logging_utils.py": ["test_rbac_logging_live_sources.py"],
    "main.py": ["test_batch_flow_integration.py", "test_api_streaming.py"],
    "metrics_store.py": ["test_metrics.py", "test_batch_flow_integration.py"],
    "model_data_errors.py": ["test_app_package_and_errors.py", "test_cost_model.py"],
    "models.py": ["test_property_invariants.py", "test_emissions_context.py"],
    "multileg_engine.py": ["test_multileg_engine.py"],
    "objectives_emissions.py": ["test_emissions_models.py"],
    "objectives_selection.py": ["test_weights.py", "test_property_invariants.py"],
    "oracle_quality_store.py": ["test_oracle_quality.py", "test_stores_and_terrain_index_unit.py"],
    "pareto.py": ["test_pareto.py", "test_pareto_strict_frontier.py"],
    "pareto_methods.py": ["test_pareto_epsilon_knee.py", "test_pareto_strict_frontier.py"],
    "provenance_store.py": ["test_run_store_reporting.py", "test_property_invariants.py"],
    "rbac.py": ["test_rbac_logging_live_sources.py"],
    "reporting.py": ["test_run_store_reporting.py"],
    "risk_model.py": ["test_scenario_compare.py", "test_stochastic_uncertainty.py"],
    "route_cache.py": ["test_route_cache.py", "test_property_invariants.py"],
    "routing_graph.py": ["test_traffic_profiles.py", "test_route_cache.py"],
    "routing_osrm.py": ["test_api_streaming.py", "test_metrics.py"],
    "run_store.py": ["test_run_store_reporting.py"],
    "scenario.py": ["test_scenario_compare.py", "test_counterfactuals.py"],
    "settings.py": ["test_property_invariants.py", "test_terrain_fail_closed_uk.py"],
    "signatures.py": ["test_signatures_api.py", "test_property_invariants.py"],
    "terrain_dem.py": ["test_terrain_fail_closed_uk.py", "test_terrain_segment_grades.py"],
    "terrain_dem_index.py": ["test_stores_and_terrain_index_unit.py"],
    "terrain_physics.py": ["test_terrain_physics_unit.py", "test_terrain_physics_uplift.py"],
    "toll_engine.py": ["test_toll_engine_unit.py", "test_traffic_profiles.py"],
    "uncertainty_model.py": ["test_uncertainty_model_unit.py"],
    "vehicles.py": ["test_vehicle_custom.py", "test_emissions_models.py"],
    "weather_adapter.py": ["test_weather_adapter.py"],
}


def test_every_app_module_has_declared_tests() -> None:
    app_files = sorted(path.name for path in APP_DIR.glob("*.py"))
    assert sorted(APP_MODULE_TO_TEST_FILES) == app_files


def test_declared_app_test_files_exist() -> None:
    for module_name, test_files in APP_MODULE_TO_TEST_FILES.items():
        assert test_files, f"{module_name} has no mapped tests"
        for test_file in test_files:
            assert (TESTS_DIR / test_file).exists(), f"{module_name} mapped missing test file: {test_file}"
