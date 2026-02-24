from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

EXPECTED_SCRIPT_FILES = {
    "__init__.py",
    "benchmark_batch_pareto.py",
    "benchmark_model_v2.py",
    "build_departure_profiles_uk.py",
    "build_model_assets.py",
    "build_pricing_tables_uk.py",
    "build_routing_graph_uk.py",
    "build_scenario_profiles_uk.py",
    "build_stochastic_calibration_uk.py",
    "build_terrain_tiles_uk.py",
    "check_eta_concept_drift.py",
    "collect_carbon_intensity_raw_uk.py",
    "collect_dft_raw_counts_uk.py",
    "collect_fuel_history_raw_uk.py",
    "collect_scenario_mode_outcomes_proxy_uk.py",
    "collect_stochastic_residuals_raw_uk.py",
    "collect_toll_truth_raw_uk.py",
    "extract_osm_tolls_uk.py",
    "fetch_carbon_intensity_uk.py",
    "fetch_dft_counts_uk.py",
    "fetch_fuel_history_uk.py",
    "fetch_public_dem_tiles_uk.py",
    "fetch_scenario_live_uk.py",
    "fetch_stochastic_residuals_uk.py",
    "fetch_toll_truth_uk.py",
    "generate_run_report.py",
    "preflight_live_runtime.py",
    "publish_live_artifacts_uk.py",
    "run_headless_scenario.py",
    "run_robustness_analysis.py",
    "run_sensitivity_analysis.py",
    "score_model_quality.py",
    "validate_graph_coverage.py",
}


def _all_script_paths() -> list[Path]:
    return sorted(path for path in SCRIPTS_DIR.glob("*.py") if path.is_file())


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def test_script_inventory_is_complete() -> None:
    discovered = {path.name for path in _all_script_paths()}
    assert discovered == EXPECTED_SCRIPT_FILES


@pytest.mark.parametrize("script_path", _all_script_paths(), ids=lambda p: p.name)
def test_script_source_parses(script_path: Path) -> None:
    source = script_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(script_path))


@pytest.mark.parametrize(
    "script_path",
    [path for path in _all_script_paths() if path.name != "__init__.py"],
    ids=lambda p: p.name,
)
def test_executable_scripts_have_main_contract(script_path: Path) -> None:
    source = script_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(script_path))
    has_main_function = any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in module.body
    )
    assert has_main_function
    has_main_guard = any(_is_main_guard(node) for node in module.body)
    assert has_main_guard
