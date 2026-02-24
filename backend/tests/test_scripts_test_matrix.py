from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
TESTS_DIR = Path(__file__).resolve().parent


SCRIPT_TO_TEST_FILES = {
    "__init__.py": ["test_scripts_smoke_all.py"],
    "benchmark_batch_pareto.py": ["test_tooling_scripts.py", "test_scripts_smoke_all.py"],
    "benchmark_model_v2.py": ["test_scripts_builders_extended.py", "test_scripts_smoke_all.py"],
    "build_departure_profiles_uk.py": ["test_scripts_builders_extended.py", "test_scripts_smoke_all.py"],
    "build_model_assets.py": ["test_scripts_builders_extended.py", "test_scripts_smoke_all.py"],
    "build_pricing_tables_uk.py": ["test_scripts_builders_extended.py", "test_scripts_smoke_all.py"],
    "build_routing_graph_uk.py": ["test_scripts_builders_extended.py", "test_scripts_smoke_all.py"],
    "build_scenario_profiles_uk.py": ["test_scripts_builders_extended.py", "test_scripts_smoke_all.py"],
    "build_stochastic_calibration_uk.py": ["test_scripts_builders_extended.py", "test_scripts_smoke_all.py"],
    "build_terrain_tiles_uk.py": ["test_scripts_builders_extended.py", "test_scripts_smoke_all.py"],
    "check_eta_concept_drift.py": ["test_tooling_scripts.py", "test_scripts_smoke_all.py"],
    "collect_carbon_intensity_raw_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "collect_dft_raw_counts_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "collect_fuel_history_raw_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "collect_scenario_mode_outcomes_proxy_uk.py": [
        "test_scripts_fetchers_extended.py",
        "test_scripts_smoke_all.py",
    ],
    "collect_stochastic_residuals_raw_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "collect_toll_truth_raw_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "extract_osm_tolls_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "fetch_carbon_intensity_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "fetch_dft_counts_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "fetch_fuel_history_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "fetch_public_dem_tiles_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "fetch_scenario_live_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "fetch_stochastic_residuals_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "fetch_toll_truth_uk.py": ["test_scripts_fetchers_extended.py", "test_scripts_smoke_all.py"],
    "generate_run_report.py": ["test_tooling_scripts.py", "test_scripts_smoke_all.py"],
    "preflight_live_runtime.py": ["test_dev_preflight.py", "test_scripts_smoke_all.py"],
    "publish_live_artifacts_uk.py": ["test_publish_live_artifacts.py", "test_scripts_smoke_all.py"],
    "run_headless_scenario.py": ["test_tooling_scripts.py", "test_scripts_smoke_all.py"],
    "run_robustness_analysis.py": ["test_tooling_scripts.py", "test_scripts_smoke_all.py"],
    "run_sensitivity_analysis.py": ["test_tooling_scripts.py", "test_scripts_smoke_all.py"],
    "score_model_quality.py": ["test_scripts_quality_extended.py", "test_scripts_smoke_all.py"],
    "validate_graph_coverage.py": ["test_scripts_quality_extended.py", "test_scripts_smoke_all.py"],
}


def test_every_script_has_declared_test_files() -> None:
    scripts = sorted(path.name for path in SCRIPTS_DIR.glob("*.py"))
    assert sorted(SCRIPT_TO_TEST_FILES) == scripts


def test_declared_test_files_exist() -> None:
    for script, tests in SCRIPT_TO_TEST_FILES.items():
        assert tests, f"{script} has no mapped tests"
        for test_file in tests:
            assert (TESTS_DIR / test_file).exists(), f"{script} mapped missing test file: {test_file}"
