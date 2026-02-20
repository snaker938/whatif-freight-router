from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import build_option
from app.models import CostToggles, EmissionsContext, StochasticConfig
from app.routing_graph import route_graph_candidate_routes
from app.scenario import ScenarioMode


def _load_fixture_routes(fixtures_dir: Path) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    if not fixtures_dir.exists():
        return routes
    for path in sorted(fixtures_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            routes.append(payload)
    return routes


def benchmark(*, fixtures_dir: Path, iterations: int, p95_gate_ms: float = 2000.0) -> dict[str, float | int | bool]:
    routes = _load_fixture_routes(fixtures_dir)
    if not routes:
        raise RuntimeError(f"No fixture routes found in {fixtures_dir}")

    timings_ms: list[float] = []
    terrain_timings_ms: list[float] = []
    graph_explored_states: list[int] = []
    graph_generated_paths: list[int] = []
    graph_emitted_paths: list[int] = []
    graph_candidate_budget: list[int] = []
    dropped = 0
    tracemalloc.start()
    for i in range(max(1, iterations)):
        for idx, route in enumerate(routes):
            geometry = route.get("geometry", {})
            coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
            if (
                isinstance(coords, list)
                and len(coords) >= 2
                and isinstance(coords[0], (list, tuple))
                and isinstance(coords[-1], (list, tuple))
                and len(coords[0]) >= 2
                and len(coords[-1]) >= 2
            ):
                try:
                    graph_routes, graph_diag = route_graph_candidate_routes(
                        origin_lat=float(coords[0][1]),
                        origin_lon=float(coords[0][0]),
                        destination_lat=float(coords[-1][1]),
                        destination_lon=float(coords[-1][0]),
                        max_paths=24,
                    )
                    if graph_routes:
                        graph_explored_states.append(int(graph_diag.explored_states))
                        graph_generated_paths.append(int(graph_diag.generated_paths))
                        graph_emitted_paths.append(int(graph_diag.emitted_paths))
                        graph_candidate_budget.append(int(graph_diag.candidate_budget))
                except Exception:
                    pass
            t0 = time.perf_counter()
            try:
                build_option(
                    route,
                    option_id=f"bench_{i}_{idx}",
                    vehicle_type="rigid_hgv",
                    scenario_mode=ScenarioMode.NO_SHARING,
                    cost_toggles=CostToggles(
                        use_tolls=False,
                        toll_cost_per_km=0.2,
                        carbon_price_per_kg=0.12,
                    ),
                    terrain_profile="flat",
                    stochastic=StochasticConfig(enabled=True, seed=42, sigma=0.08, samples=32),
                    emissions_context=EmissionsContext(fuel_type="diesel", euro_class="euro6", ambient_temp_c=12),
                    departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                )
            except Exception:
                dropped += 1
                continue
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
            t1 = time.perf_counter()
            try:
                build_option(
                    route,
                    option_id=f"bench_terrain_{i}_{idx}",
                    vehicle_type="rigid_hgv",
                    scenario_mode=ScenarioMode.NO_SHARING,
                    cost_toggles=CostToggles(
                        use_tolls=False,
                        toll_cost_per_km=0.2,
                        carbon_price_per_kg=0.12,
                    ),
                    terrain_profile="hilly",
                    stochastic=StochasticConfig(enabled=True, seed=42, sigma=0.08, samples=32),
                    emissions_context=EmissionsContext(fuel_type="diesel", euro_class="euro6", ambient_temp_c=12),
                    departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                )
            except Exception:
                dropped += 1
                continue
            terrain_timings_ms.append((time.perf_counter() - t1) * 1000.0)

    if not timings_ms:
        raise RuntimeError("No successful benchmark samples; all route builds failed.")
    sorted_timings = sorted(timings_ms)
    p95_idx = max(0, min(len(sorted_timings) - 1, int(0.95 * len(sorted_timings)) - 1))
    terrain_sorted = sorted(terrain_timings_ms) if terrain_timings_ms else [0.0]
    terrain_p95_idx = max(0, min(len(terrain_sorted) - 1, int(0.95 * len(terrain_sorted)) - 1))
    p95_ms = round(sorted_timings[p95_idx], 4)
    terrain_p95_ms = round(terrain_sorted[terrain_p95_idx], 4)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    explored_sorted = sorted(graph_explored_states) if graph_explored_states else [0]
    generated_sorted = sorted(graph_generated_paths) if graph_generated_paths else [0]
    emitted_sorted = sorted(graph_emitted_paths) if graph_emitted_paths else [0]
    budget_sorted = sorted(graph_candidate_budget) if graph_candidate_budget else [0]
    diag_p95_idx = max(0, min(len(explored_sorted) - 1, int(0.95 * len(explored_sorted)) - 1))
    return {
        "samples": len(timings_ms),
        "dropped_samples": dropped,
        "routes_per_iter": len(routes),
        "mean_ms": round(statistics.fmean(timings_ms), 4),
        "p95_ms": p95_ms,
        "max_ms": round(max(timings_ms), 4),
        "terrain_mean_ms": round(statistics.fmean(terrain_timings_ms), 4) if terrain_timings_ms else 0.0,
        "terrain_p95_ms": terrain_p95_ms,
        "terrain_max_ms": round(max(terrain_timings_ms), 4) if terrain_timings_ms else 0.0,
        "p95_gate_ms": round(float(p95_gate_ms), 3),
        "p95_gate_passed": bool(p95_ms <= float(p95_gate_ms)),
        "terrain_p95_gate_passed": bool(terrain_p95_ms <= float(p95_gate_ms)),
        "graph_explored_states_p95": int(explored_sorted[diag_p95_idx]),
        "graph_generated_paths_p95": int(generated_sorted[diag_p95_idx]),
        "graph_emitted_paths_p95": int(emitted_sorted[diag_p95_idx]),
        "graph_candidate_budget_p95": int(budget_sorted[diag_p95_idx]),
        "memory_current_mb": round(float(current_mem) / (1024.0 * 1024.0), 3),
        "memory_peak_mb": round(float(peak_mem) / (1024.0 * 1024.0), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark backend model-v2 route option runtime.")
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "uk_routes",
        help="Path to deterministic route fixture corpus",
    )
    parser.add_argument("--iterations", type=int, default=8, help="Number of benchmark iterations")
    parser.add_argument(
        "--p95-gate-ms",
        type=float,
        default=2000.0,
        help="P95 runtime gate for success.",
    )
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help="Exit non-zero when p95 gate fails.",
    )
    args = parser.parse_args()

    report = benchmark(
        fixtures_dir=args.fixtures_dir,
        iterations=args.iterations,
        p95_gate_ms=max(1.0, float(args.p95_gate_ms)),
    )
    print(json.dumps(report, indent=2))
    if args.enforce_gate and (not report.get("p95_gate_passed", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
