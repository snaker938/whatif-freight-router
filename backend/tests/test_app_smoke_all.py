from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"

EXPECTED_APP_FILES = {
    "__init__.py",
    "calibration_loader.py",
    "carbon_model.py",
    "departure_profile.py",
    "experiment_store.py",
    "fuel_energy_model.py",
    "incident_simulator.py",
    "k_shortest.py",
    "live_call_trace.py",
    "live_data_sources.py",
    "logging_utils.py",
    "main.py",
    "metrics_store.py",
    "model_data_errors.py",
    "models.py",
    "multileg_engine.py",
    "objectives_emissions.py",
    "objectives_selection.py",
    "oracle_quality_store.py",
    "pareto.py",
    "pareto_methods.py",
    "provenance_store.py",
    "rbac.py",
    "reporting.py",
    "risk_model.py",
    "route_cache.py",
    "routing_graph.py",
    "routing_osrm.py",
    "run_store.py",
    "scenario.py",
    "settings.py",
    "signatures.py",
    "terrain_dem.py",
    "terrain_dem_index.py",
    "terrain_physics.py",
    "toll_engine.py",
    "uncertainty_model.py",
    "vehicles.py",
    "weather_adapter.py",
}


def _all_app_paths() -> list[Path]:
    return sorted(path for path in APP_DIR.glob("*.py") if path.is_file())


def test_app_inventory_is_complete() -> None:
    discovered = {path.name for path in _all_app_paths()}
    assert discovered == EXPECTED_APP_FILES


@pytest.mark.parametrize("module_path", _all_app_paths(), ids=lambda p: p.name)
def test_app_module_parses(module_path: Path) -> None:
    source = module_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(module_path))
