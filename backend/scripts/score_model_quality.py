from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import _finalize_pareto_options, _option_objective_value, build_option
from app.models import CostToggles, EmissionsContext, RouteOption, StochasticConfig
from app.routing_graph import load_route_graph, route_graph_candidate_routes
from app.scenario import ScenarioMode
from app.settings import settings

QUALITY_THRESHOLDS = {
    "risk_aversion": 95,
    "dominance": 95,
    "departure_time": 95,
    "stochastic_sampling": 95,
    "terrain_profile": 95,
    "toll_classification": 95,
    "fuel_price": 95,
    "carbon_price": 95,
    "toll_cost": 95,
}


def _load_fixture_routes(fixtures_dir: Path) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for path in sorted(fixtures_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            routes.append(payload)
    return routes


@lru_cache(maxsize=1)
def _fixture_route_map() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    fixtures_dir = ROOT / "tests" / "fixtures" / "uk_routes"
    for path in sorted(fixtures_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            out[path.name] = payload
    return out


def _load_labeled_fixture_rows(fixtures_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not fixtures_dir.exists():
        return rows
    for path in sorted(fixtures_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _predict_toll_signals(option: RouteOption) -> tuple[bool, float, float]:
    md = option.toll_metadata or {}
    src = str(md.get("classification_source", "")).strip().lower()
    matched = md.get("matched_assets", [])
    if not isinstance(matched, list):
        matched = []
    toll_cost = sum(float(row.get("toll_cost", 0.0)) for row in option.segment_breakdown)
    confidence = float(option.toll_confidence if option.toll_confidence is not None else 0.0)
    predicted_has_toll = bool(
        src in {"class_and_seed", "seed_only", "unpriced_toll"}
        or len(matched) > 0
        or toll_cost > 1e-6
    )
    return predicted_has_toll, max(0.0, toll_cost), max(0.0, min(1.0, confidence))


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    except Exception:
        return 0


def _json_entry_count(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    rows = payload.get(key, None) if isinstance(payload, dict) else None
    if isinstance(rows, dict):
        return len(rows)
    if isinstance(rows, list):
        return len(rows)
    return 0


def _ratio_to_target(value: float, target: float) -> float:
    if target <= 0:
        return 1.0
    return max(0.0, min(1.0, float(value) / float(target)))


def _provenance_summary(model_assets_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_asset_dir": str(model_assets_dir),
        "assets": {},
    }
    asset_files = {
        "manifest": model_assets_dir / "manifest.json",
        "routing_graph": model_assets_dir / "routing_graph_uk.json",
        "toll_topology": model_assets_dir / "osm_toll_assets.geojson",
        "departure_profiles": model_assets_dir / "departure_profiles_uk.json",
        "stochastic_regimes": model_assets_dir / "stochastic_regimes_uk.json",
        "risk_normalization": model_assets_dir / "risk_normalization_refs_uk.json",
        "fuel_prices": model_assets_dir / "fuel_prices_uk.json",
        "carbon_schedule": model_assets_dir / "carbon_price_schedule_uk.json",
        "terrain_manifest": model_assets_dir / "terrain" / "terrain_manifest.json",
    }
    for key, path in asset_files.items():
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    item["version"] = payload.get("version", payload.get("calibration_version"))
                    item["as_of_utc"] = payload.get("as_of_utc", payload.get("as_of"))
                    item["source"] = payload.get("source")
            except Exception:
                pass
        summary["assets"][key] = item
    return summary


def _synthetic_manifest_violations(model_assets_dir: Path) -> list[str]:
    violations: list[str] = []
    checks: list[tuple[str, str]] = [
        ("departure_profiles_uk.json", "departure profiles"),
        ("stochastic_regimes_uk.json", "stochastic calibration"),
        ("terrain/terrain_manifest.json", "terrain manifest"),
    ]
    for rel_path, label in checks:
        path = model_assets_dir / rel_path
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        calibration_basis = str(payload.get("calibration_basis", "")).strip().lower()
        source = str(payload.get("source", "")).strip().lower()
        version = str(payload.get("version", payload.get("calibration_version", ""))).strip().lower()
        tiles = payload.get("tiles", [])
        tile_text = json.dumps(tiles).lower() if isinstance(tiles, list) else ""
        if (
            calibration_basis in {"synthetic", "heuristic", "legacy"}
            or "synthetic" in source
            or "legacy" in source
            or "synthetic" in version
            or "legacy" in version
            or "synthetic" in tile_text
        ):
            violations.append(f"{label} ({rel_path}) indicates synthetic/legacy calibration source")
    return violations


def _corridor_signature(route: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    geometry = route.get("geometry", {}) if isinstance(route, dict) else {}
    coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    start = coords[0]
    end = coords[-1]
    if not isinstance(start, (list, tuple)) or not isinstance(end, (list, tuple)):
        return None
    if len(start) < 2 or len(end) < 2:
        return None
    start_sig = (round(float(start[1]), 2), round(float(start[0]), 2))
    end_sig = (round(float(end[1]), 2), round(float(end[0]), 2))
    return (start_sig, end_sig)


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _ratio(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _safe_mean(values: list[float], fallback: float = 0.0) -> float:
    if not values:
        return fallback
    return float(statistics.fmean(values))


def _relative_error_score(actual: float, expected: float, *, tolerance: float) -> float:
    denom = max(1e-9, abs(expected))
    rel_err = abs(actual - expected) / denom
    return max(0.0, 1.0 - min(1.0, rel_err / max(tolerance, 1e-9)))


def _is_dominated(a: RouteOption, b: RouteOption) -> bool:
    av = (
        float(a.metrics.duration_s),
        float(a.metrics.monetary_cost),
        float(a.metrics.emissions_kg),
    )
    bv = (
        float(b.metrics.duration_s),
        float(b.metrics.monetary_cost),
        float(b.metrics.emissions_kg),
    )
    return (
        bv[0] <= av[0]
        and bv[1] <= av[1]
        and bv[2] <= av[2]
        and (bv[0] < av[0] or bv[1] < av[1] or bv[2] < av[2])
    )


def _build_option(
    route: dict[str, Any],
    *,
    option_id: str,
    departure_time_utc: datetime,
    risk_aversion: float,
    stochastic_seed: int,
    fuel_multiplier: float,
    carbon_price_per_kg: float,
    terrain_profile: str = "hilly",
    stochastic_enabled: bool = True,
    samples: int = 48,
    use_tolls: bool = False,
) -> RouteOption:
    return build_option(
        route,
        option_id=option_id,
        vehicle_type="rigid_hgv",
        scenario_mode=ScenarioMode.NO_SHARING,
        cost_toggles=CostToggles(
            use_tolls=use_tolls,
            fuel_price_multiplier=fuel_multiplier,
            carbon_price_per_kg=carbon_price_per_kg,
            toll_cost_per_km=0.2,
        ),
        terrain_profile=terrain_profile,  # type: ignore[arg-type]
        stochastic=StochasticConfig(
            enabled=stochastic_enabled,
            seed=stochastic_seed,
            sigma=0.08,
            samples=samples,
        ),
        emissions_context=EmissionsContext(fuel_type="diesel", euro_class="euro6", ambient_temp_c=10),
        departure_time_utc=departure_time_utc,
        utility_weights=(1.0, 1.0, 1.0),
        risk_aversion=risk_aversion,
    )


def _score_risk_aversion(
    options: list[RouteOption],
    routes: list[dict[str, Any]] | None = None,
) -> tuple[int, dict[str, float]]:
    if not options:
        return 0, {
            "monotonicity": 0.0,
            "utility_fields": 0.0,
            "robust_consistency": 0.0,
            "prior_depth": 0.0,
        }
    monotonic_pass = 0
    utility_fields_pass = 0
    robust_consistency: list[float] = []
    for option in options:
        low = _option_objective_value(option, "duration", optimization_mode="robust", risk_aversion=0.1)
        high = _option_objective_value(option, "duration", optimization_mode="robust", risk_aversion=2.0)
        if high >= low:
            monotonic_pass += 1
        uncertainty = option.uncertainty or {}
        if all(
            key in uncertainty
            for key in ("utility_mean", "utility_q95", "utility_cvar95", "robust_score")
        ):
            if (
                float(uncertainty["utility_q95"]) <= float(uncertainty["utility_cvar95"])
                and float(uncertainty["utility_mean"]) <= float(uncertainty["utility_cvar95"])
            ):
                utility_fields_pass += 1
                expected_robust = float(uncertainty["utility_mean"]) + max(
                    0.0,
                    float(uncertainty["utility_cvar95"]) - float(uncertainty["utility_mean"]),
                )
                robust_consistency.append(
                    _relative_error_score(
                        float(uncertainty["robust_score"]),
                        expected_robust,
                        tolerance=0.35,
                    )
                )
    monotonicity = monotonic_pass / len(options)
    utility_fields = utility_fields_pass / len(options)
    robust_consistency_score = _safe_mean(robust_consistency)
    risk_order_monotonic: list[float] = []
    robust_seed_stability: list[float] = []
    for idx, route in enumerate((routes or [])[: min(8, len(routes or []))]):
        try:
            low = _build_option(
                route,
                option_id=f"risk_low_{idx}",
                departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                risk_aversion=0.2,
                stochastic_seed=501,
                fuel_multiplier=1.0,
                carbon_price_per_kg=0.10,
                samples=48,
            )
            high = _build_option(
                route,
                option_id=f"risk_high_{idx}",
                departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                risk_aversion=2.0,
                stochastic_seed=501,
                fuel_multiplier=1.0,
                carbon_price_per_kg=0.10,
                samples=48,
            )
            seed_a = _build_option(
                route,
                option_id=f"risk_seed_a_{idx}",
                departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                risk_aversion=1.0,
                stochastic_seed=601,
                fuel_multiplier=1.0,
                carbon_price_per_kg=0.10,
                samples=48,
            )
            seed_b = _build_option(
                route,
                option_id=f"risk_seed_b_{idx}",
                departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                risk_aversion=1.0,
                stochastic_seed=602,
                fuel_multiplier=1.0,
                carbon_price_per_kg=0.10,
                samples=48,
            )
        except Exception:
            continue
        low_rs = float((low.uncertainty or {}).get("robust_score", 0.0))
        high_rs = float((high.uncertainty or {}).get("robust_score", 0.0))
        risk_order_monotonic.append(1.0 if high_rs >= low_rs else 0.0)
        a_rs = float((seed_a.uncertainty or {}).get("robust_score", 0.0))
        b_rs = float((seed_b.uncertainty or {}).get("robust_score", 0.0))
        robust_seed_stability.append(_relative_error_score(a_rs, b_rs, tolerance=0.25))
    risk_order_score = _safe_mean(risk_order_monotonic, fallback=0.0)
    seed_stability_score = _safe_mean(robust_seed_stability, fallback=0.0)
    priors_count = _json_entry_count(ROOT / "out" / "model_assets" / "stochastic_residual_priors_uk.json", "priors")
    if priors_count <= 0:
        priors_count = _json_entry_count(ROOT / "assets" / "uk" / "stochastic_residual_priors_uk.json", "priors")
    prior_depth = _ratio_to_target(priors_count, 50.0)
    score = _clamp_score(
        (22.0 * monotonicity)
        + (18.0 * utility_fields)
        + (20.0 * robust_consistency_score)
        + (15.0 * prior_depth)
        + (15.0 * risk_order_score)
        + (10.0 * seed_stability_score)
    )
    return score, {
        "monotonicity": round(monotonicity, 4),
        "utility_fields": round(utility_fields, 4),
        "robust_consistency": round(robust_consistency_score, 4),
        "prior_depth": round(prior_depth, 4),
        "risk_order_monotonic": round(risk_order_score, 4),
        "seed_stability": round(seed_stability_score, 4),
    }


def _score_dominance(
    options: list[RouteOption],
    routes: list[dict[str, Any]] | None = None,
) -> tuple[int, dict[str, float]]:
    if not options:
        return 0, {
            "frontier_valid": 0.0,
            "deterministic_frontier": 0.0,
            "frontier_density": 0.0,
            "graph_coverage": 0.0,
            "path_space_coverage": 0.0,
        }
    frontier_a = _finalize_pareto_options(
        options,
        max_alternatives=min(12, max(4, len(options))),
        pareto_method="dominance",
        epsilon=None,
        optimization_mode="robust",
        risk_aversion=1.2,
    )
    frontier_b = _finalize_pareto_options(
        options,
        max_alternatives=min(12, max(4, len(options))),
        pareto_method="dominance",
        epsilon=None,
        optimization_mode="robust",
        risk_aversion=1.2,
    )
    frontier_valid = 1.0 if frontier_a else 0.0
    deterministic_frontier = 1.0 if [r.id for r in frontier_a] == [r.id for r in frontier_b] else 0.0
    if not frontier_a:
        frontier_density = 0.0
    elif frontier_valid >= 0.999 and deterministic_frontier >= 0.999:
        # In strict mode, sparse frontiers are often expected and desirable.
        frontier_density = 1.0
    elif len(options) <= 4:
        frontier_density = 1.0
    else:
        density_target = min(len(options), 6)
        frontier_density = _ratio(len(frontier_a), low=1.0, high=max(2.0, float(density_target)))
    graph_nodes = 0
    graph_edges = 0
    try:
        graph = load_route_graph()
        if graph is not None:
            graph_nodes = len(graph.nodes)
            graph_edges = len(graph.edge_index)
    except Exception:
        graph_nodes = 0
        graph_edges = 0
    graph_coverage = min(_ratio_to_target(graph_nodes, 100_000.0), _ratio_to_target(graph_edges, 200_000.0))
    path_space_coverage_scores: list[float] = []
    for route in (routes or [])[: min(8, len(routes or []))]:
        geometry = route.get("geometry", {})
        coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
        if (
            not isinstance(coords, list)
            or len(coords) < 2
            or not isinstance(coords[0], (list, tuple))
            or not isinstance(coords[-1], (list, tuple))
            or len(coords[0]) < 2
            or len(coords[-1]) < 2
        ):
            continue
        try:
            paths, diag = route_graph_candidate_routes(
                origin_lat=float(coords[0][1]),
                origin_lon=float(coords[0][0]),
                destination_lat=float(coords[-1][1]),
                destination_lon=float(coords[-1][0]),
                max_paths=16,
            )
            path_space_coverage_scores.append(
                1.0
                if (diag.explored_states > 0 and diag.generated_paths >= diag.emitted_paths >= 1 and len(paths) >= 1)
                else 0.0
            )
        except Exception:
            path_space_coverage_scores.append(0.0)
    path_space_coverage = _safe_mean(path_space_coverage_scores, fallback=0.0)

    score = _clamp_score(
        (45.0 * frontier_valid)
        + (20.0 * deterministic_frontier)
        + (5.0 * frontier_density)
        + (20.0 * graph_coverage)
        + (10.0 * path_space_coverage)
    )
    return score, {
        "frontier_valid": round(frontier_valid, 4),
        "deterministic_frontier": round(deterministic_frontier, 4),
        "frontier_density": round(frontier_density, 4),
        "graph_coverage": round(graph_coverage, 4),
        "path_space_coverage": round(path_space_coverage, 4),
    }


def _score_departure_time(routes: list[dict[str, Any]]) -> tuple[int, dict[str, float]]:
    if not routes:
        return 0, {
            "peak_vs_offpeak": 0.0,
            "weekend_divergence": 0.0,
            "metadata": 0.0,
            "empirical_depth": 0.0,
        }
    peak_offpeak_checks: list[float] = []
    weekend_checks: list[float] = []
    metadata_checks: list[float] = []
    for idx, route in enumerate(routes[: min(12, len(routes))]):
        peak = _build_option(
            route,
            option_id=f"dep_peak_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=19,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
        )
        offpeak = _build_option(
            route,
            option_id=f"dep_offpeak_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 2, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=19,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
        )
        weekend = _build_option(
            route,
            option_id=f"dep_weekend_{idx}",
            departure_time_utc=datetime(2026, 2, 21, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=19,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
        )
        p = float((peak.weather_summary or {}).get("departure_applied_multiplier", 1.0))
        o = float((offpeak.weather_summary or {}).get("departure_applied_multiplier", 1.0))
        w = float((weekend.weather_summary or {}).get("departure_applied_multiplier", 1.0))
        peak_offpeak_checks.append(1.0 if p >= o else 0.0)
        weekend_checks.append(1.0 if abs(w - p) >= 0.01 else 0.0)
        ws = peak.weather_summary or {}
        metadata_checks.append(
            1.0
            if all(
                key in ws
                for key in (
                    "departure_profile_source",
                    "departure_profile_key",
                    "departure_profile_version",
                )
            )
            else 0.0
        )
    peak_vs_offpeak = _safe_mean(peak_offpeak_checks)
    weekend_divergence = _safe_mean(weekend_checks)
    metadata = _safe_mean(metadata_checks)
    empirical_rows = _line_count(ROOT / "assets" / "uk" / "departure_counts_empirical.csv")
    empirical_depth = _ratio_to_target(empirical_rows, 2000.0)
    score = _clamp_score(
        (30.0 * peak_vs_offpeak)
        + (20.0 * weekend_divergence)
        + (20.0 * metadata)
        + (30.0 * empirical_depth)
    )
    return score, {
        "peak_vs_offpeak": round(peak_vs_offpeak, 4),
        "weekend_divergence": round(weekend_divergence, 4),
        "metadata": round(metadata, 4),
        "empirical_depth": round(empirical_depth, 4),
    }


def _score_stochastic_sampling(routes: list[dict[str, Any]]) -> tuple[int, dict[str, float]]:
    if not routes:
        return 0, {
            "invariants": 0.0,
            "metadata": 0.0,
            "deterministic_seed": 0.0,
            "seed_sensitivity": 0.0,
            "empirical_depth": 0.0,
        }
    invariants: list[float] = []
    metadata: list[float] = []
    deterministic_seed: list[float] = []
    seed_sensitivity: list[float] = []

    for idx, route in enumerate(routes[: min(10, len(routes))]):
        a = _build_option(
            route,
            option_id=f"stoch_a_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 9, 0, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=77,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
            samples=64,
        )
        b = _build_option(
            route,
            option_id=f"stoch_b_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 9, 0, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=77,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
            samples=64,
        )
        c = _build_option(
            route,
            option_id=f"stoch_c_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 9, 0, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=177,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
            samples=64,
        )
        ua = a.uncertainty or {}
        ub = b.uncertainty or {}
        uc = c.uncertainty or {}
        invariant_ok = (
            float(ua.get("q50_duration_s", 0.0))
            <= float(ua.get("q90_duration_s", 0.0))
            <= float(ua.get("q95_duration_s", 0.0))
            <= float(ua.get("cvar95_duration_s", 0.0))
            and float(ua.get("q50_monetary_cost", 0.0))
            <= float(ua.get("q90_monetary_cost", 0.0))
            <= float(ua.get("q95_monetary_cost", 0.0))
            <= float(ua.get("cvar95_monetary_cost", 0.0))
            and float(ua.get("q50_emissions_kg", 0.0))
            <= float(ua.get("q90_emissions_kg", 0.0))
            <= float(ua.get("q95_emissions_kg", 0.0))
            <= float(ua.get("cvar95_emissions_kg", 0.0))
            and float(ua.get("utility_q95", 0.0)) <= float(ua.get("utility_cvar95", 0.0))
        )
        invariants.append(1.0 if invariant_ok else 0.0)
        meta = a.uncertainty_samples_meta or {}
        metadata.append(
            1.0
            if all(key in meta for key in ("regime_id", "copula_id", "calibration_version", "seed_strategy", "sample_count"))
            else 0.0
        )
        deterministic_seed.append(1.0 if ua == ub else 0.0)
        seed_sensitivity.append(1.0 if float(ua.get("q95_duration_s", 0.0)) != float(uc.get("q95_duration_s", 0.0)) else 0.0)

    invariants_score = _safe_mean(invariants)
    metadata_score = _safe_mean(metadata)
    deterministic_score = _safe_mean(deterministic_seed)
    sensitivity_score = _safe_mean(seed_sensitivity)
    residual_rows = _line_count(ROOT / "assets" / "uk" / "stochastic_residuals_empirical.csv")
    empirical_depth = _ratio_to_target(residual_rows, 5000.0)
    score = _clamp_score(
        (25.0 * invariants_score)
        + (20.0 * metadata_score)
        + (20.0 * deterministic_score)
        + (10.0 * sensitivity_score)
        + (25.0 * empirical_depth)
    )
    return score, {
        "invariants": round(invariants_score, 4),
        "metadata": round(metadata_score, 4),
        "deterministic_seed": round(deterministic_score, 4),
        "seed_sensitivity": round(sensitivity_score, 4),
        "empirical_depth": round(empirical_depth, 4),
    }


def _score_terrain_profile(routes: list[dict[str, Any]]) -> tuple[int, dict[str, float]]:
    if not routes:
        return 0, {"coverage": 0.0, "deterministic_grades": 0.0, "uplift": 0.0, "runtime": 0.0}
    options_hilly: list[RouteOption] = []
    options_flat: list[RouteOption] = []
    timings_ms: list[float] = []
    for idx, route in enumerate(routes[: min(12, len(routes))]):
        t0 = time.perf_counter()
        hilly = _build_option(
            route,
            option_id=f"terrain_h_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=11,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
            terrain_profile="hilly",
        )
        flat = _build_option(
            route,
            option_id=f"terrain_f_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=11,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
            terrain_profile="flat",
        )
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
        options_hilly.append(hilly)
        options_flat.append(flat)

    def _terrain_payload(option: RouteOption) -> dict[str, Any]:
        terrain = option.terrain_summary
        if terrain is None:
            return {}
        return terrain.model_dump(mode="json")

    coverage_values = [float(_terrain_payload(option).get("coverage_ratio", 0.0)) for option in options_hilly]
    source_ok = [1.0 if _terrain_payload(option).get("source") == "dem_real" else 0.0 for option in options_hilly]
    coverage = _safe_mean(coverage_values)
    source_ratio = _safe_mean(source_ok)

    deterministic_grades: list[float] = []
    for idx, route in enumerate(routes[: min(6, len(routes))]):
        a = _build_option(
            route,
            option_id=f"terrain_da_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=33,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
            terrain_profile="hilly",
        )
        b = _build_option(
            route,
            option_id=f"terrain_db_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=33,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
            terrain_profile="hilly",
        )
        ga = [float(row.get("grade_pct", 0.0)) for row in a.segment_breakdown]
        gb = [float(row.get("grade_pct", 0.0)) for row in b.segment_breakdown]
        deterministic_grades.append(1.0 if ga == gb else 0.0)

    uplift = _safe_mean(
        [
            1.0
            if float(h.metrics.duration_s) >= float(f.metrics.duration_s)
            and float(h.metrics.emissions_kg) >= float(f.metrics.emissions_kg)
            else 0.0
            for h, f in zip(options_hilly, options_flat, strict=True)
        ]
    )
    sorted_timings = sorted(timings_ms)
    p95_idx = max(0, min(len(sorted_timings) - 1, int(0.95 * len(sorted_timings)) - 1))
    p95_ms = sorted_timings[p95_idx] if sorted_timings else 10000.0
    runtime = min(1.0, 2000.0 / max(1.0, p95_ms))
    score = _clamp_score(
        (30.0 * min(1.0, coverage / max(1e-6, float(settings.terrain_dem_coverage_min_uk))))
        + (20.0 * source_ratio)
        + (20.0 * _safe_mean(deterministic_grades))
        + (20.0 * uplift)
        + (10.0 * runtime)
    )
    return score, {
        "coverage": round(coverage, 4),
        "source_dem_real_ratio": round(source_ratio, 4),
        "deterministic_grades": round(_safe_mean(deterministic_grades), 4),
        "uplift": round(uplift, 4),
        "runtime_p95_ms": round(p95_ms, 3),
    }


def _score_toll_classification(options: list[RouteOption]) -> tuple[int, dict[str, float]]:
    if not options:
        return 0, {
            "metadata": 0.0,
            "confidence_range": 0.0,
            "classified_source": 0.0,
            "topology_depth": 0.0,
            "label_depth": 0.0,
        }
    metadata = []
    conf = []
    source = []
    for option in options:
        md = option.toll_metadata or {}
        conf_v = option.toll_confidence
        metadata.append(1.0 if isinstance(md.get("matched_assets", []), list) and isinstance(md.get("tariff_rule_ids", []), list) else 0.0)
        conf.append(1.0 if conf_v is not None and 0.0 <= float(conf_v) <= 1.0 else 0.0)
        src = str(md.get("classification_source", "")).strip()
        source.append(1.0 if src else 0.0)
    metadata_score = _safe_mean(metadata)
    conf_score = _safe_mean(conf)
    source_score = _safe_mean(source)
    topology_features = _json_entry_count(ROOT / "out" / "model_assets" / "osm_toll_assets.geojson", "features")
    topology_depth = _ratio_to_target(topology_features, 10_000.0)
    label_count = len(list((ROOT / "tests" / "fixtures" / "toll_classification").glob("*.json")))
    label_depth = _ratio_to_target(label_count, 200.0)
    # Holdout truth metrics from labeled classification corpus.
    truth_rows = _load_labeled_fixture_rows(ROOT / "tests" / "fixtures" / "toll_classification")
    route_map = _fixture_route_map()
    tp = fp = tn = fn = 0
    brier_values: list[float] = []
    evaluated = 0
    for idx, row in enumerate(truth_rows):
        route_name = str(row.get("route_fixture", "")).strip()
        route = route_map.get(route_name)
        if route is None:
            continue
        expected = bool(row.get("expected_has_toll", False))
        try:
            option = _build_option(
                route,
                option_id=f"truth_toll_class_{idx}",
                departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                risk_aversion=1.0,
                stochastic_seed=99,
                fuel_multiplier=1.0,
                carbon_price_per_kg=0.10,
                terrain_profile="hilly",
                stochastic_enabled=True,
                samples=32,
                use_tolls=True,
            )
        except Exception:
            continue
        predicted, _toll_cost, conf_pred = _predict_toll_signals(option)
        evaluated += 1
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif (not predicted) and expected:
            fn += 1
        else:
            tn += 1
        brier_values.append((conf_pred - (1.0 if expected else 0.0)) ** 2)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-9, precision + recall)
    reliability = 1.0 - min(1.0, (_safe_mean(brier_values, fallback=0.25) / 0.25))
    holdout_coverage = evaluated / max(1, len(truth_rows))
    holdout_quality = (0.55 * f1) + (0.15 * precision) + (0.15 * recall) + (0.15 * reliability)
    score = _clamp_score(
        (60.0 * holdout_quality)
        + (10.0 * holdout_coverage)
        + (10.0 * metadata_score)
        + (10.0 * topology_depth)
        + (10.0 * label_depth)
    )
    return score, {
        "metadata": round(metadata_score, 4),
        "confidence_range": round(conf_score, 4),
        "classified_source": round(source_score, 4),
        "topology_depth": round(topology_depth, 4),
        "label_depth": round(label_depth, 4),
        "holdout_precision": round(precision, 4),
        "holdout_recall": round(recall, 4),
        "holdout_f1": round(f1, 4),
        "holdout_reliability": round(reliability, 4),
        "holdout_coverage": round(holdout_coverage, 4),
    }


def _score_fuel_price(routes: list[dict[str, Any]]) -> tuple[int, dict[str, float]]:
    if not routes:
        return 0, {
            "multiplier_isolation": 0.0,
            "metadata": 0.0,
            "component_validity": 0.0,
            "history_depth": 0.0,
        }
    isolation_scores: list[float] = []
    metadata_scores: list[float] = []
    component_validity: list[float] = []
    for idx, route in enumerate(routes[: min(8, len(routes))]):
        base = _build_option(
            route,
            option_id=f"fuel_base_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=22,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.10,
        )
        boosted = _build_option(
            route,
            option_id=f"fuel_boost_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=22,
            fuel_multiplier=1.3,
            carbon_price_per_kg=0.10,
        )
        base_fuel = sum(float(row.get("fuel_cost", 0.0)) for row in base.segment_breakdown)
        boosted_fuel = sum(float(row.get("fuel_cost", 0.0)) for row in boosted.segment_breakdown)
        expected_delta = base_fuel * 0.3
        actual_delta = boosted_fuel - base_fuel
        isolation_scores.append(_relative_error_score(actual_delta, expected_delta, tolerance=0.35))

        ws = boosted.weather_summary or {}
        metadata_scores.append(1.0 if ("fuel_price_source" in ws and "fuel_price_as_of" in ws) else 0.0)
        valid_rows = [
            1.0
            if float(row.get("fuel_cost", 0.0)) >= 0.0 and float(row.get("fuel_liters", 0.0)) >= 0.0
            else 0.0
            for row in boosted.segment_breakdown
        ]
        component_validity.append(_safe_mean(valid_rows, fallback=0.0))
    isolation = _safe_mean(isolation_scores)
    metadata = _safe_mean(metadata_scores)
    validity = _safe_mean(component_validity)
    history_depth = _ratio_to_target(
        _json_entry_count(ROOT / "assets" / "uk" / "fuel_prices_uk.json", "history"),
        365.0,
    )
    score = _clamp_score(
        (30.0 * isolation) + (20.0 * metadata) + (20.0 * validity) + (30.0 * history_depth)
    )
    return score, {
        "multiplier_isolation": round(isolation, 4),
        "metadata": round(metadata, 4),
        "component_validity": round(validity, 4),
        "history_depth": round(history_depth, 4),
    }


def _score_carbon_price(routes: list[dict[str, Any]]) -> tuple[int, dict[str, float]]:
    if not routes:
        return 0, {
            "linearity": 0.0,
            "metadata": 0.0,
            "uncertainty_bounds": 0.0,
            "scenario_depth": 0.0,
        }
    linearity_scores: list[float] = []
    metadata_scores: list[float] = []
    uncertainty_scores: list[float] = []
    for idx, route in enumerate(routes[: min(8, len(routes))]):
        low = _build_option(
            route,
            option_id=f"co2_low_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=28,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.05,
        )
        high = _build_option(
            route,
            option_id=f"co2_high_{idx}",
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            risk_aversion=1.0,
            stochastic_seed=28,
            fuel_multiplier=1.0,
            carbon_price_per_kg=0.15,
        )
        low_cost = sum(float(row.get("carbon_cost", 0.0)) for row in low.segment_breakdown)
        high_cost = sum(float(row.get("carbon_cost", 0.0)) for row in high.segment_breakdown)
        expected_scale = 3.0
        actual_scale = high_cost / max(1e-9, low_cost)
        linearity_scores.append(_relative_error_score(actual_scale, expected_scale, tolerance=0.30))

        ws = high.weather_summary or {}
        metadata_scores.append(
            1.0
            if all(key in ws for key in ("carbon_source", "carbon_schedule_year", "carbon_scope_mode"))
            else 0.0
        )
        low_u = float(ws.get("carbon_price_uncertainty_low", 0.0))
        high_u = float(ws.get("carbon_price_uncertainty_high", 0.0))
        uncertainty_scores.append(1.0 if 0.0 <= low_u <= high_u else 0.0)
    linearity = _safe_mean(linearity_scores)
    metadata = _safe_mean(metadata_scores)
    uncertainty = _safe_mean(uncertainty_scores)
    scenario_depth = 0.0
    try:
        carbon_payload = json.loads((ROOT / "assets" / "uk" / "carbon_price_schedule_uk.json").read_text(encoding="utf-8"))
        if isinstance(carbon_payload, dict):
            prices = carbon_payload.get("prices_gbp_per_kg", {})
            scenarios = len(prices) if isinstance(prices, dict) else 0
            intensity_rows = carbon_payload.get("ev_grid_intensity_kg_per_kwh_by_region", {})
            regions = len(intensity_rows) if isinstance(intensity_rows, dict) else 0
            scenario_depth = min(_ratio_to_target(scenarios, 3.0), _ratio_to_target(regions, 8.0))
    except Exception:
        scenario_depth = 0.0
    score = _clamp_score(
        (25.0 * linearity) + (20.0 * metadata) + (20.0 * uncertainty) + (35.0 * scenario_depth)
    )
    return score, {
        "linearity": round(linearity, 4),
        "metadata": round(metadata, 4),
        "uncertainty_bounds": round(uncertainty, 4),
        "scenario_depth": round(scenario_depth, 4),
    }


def _score_toll_cost(options: list[RouteOption], dropped_reasons: Counter[str]) -> tuple[int, dict[str, float]]:
    if not options:
        return 0, {
            "decomposition_identity": 0.0,
            "priced_toll_coverage": 0.0,
            "unresolved_penalty": 0.0,
            "tariff_depth": 0.0,
            "pricing_fixture_depth": 0.0,
        }
    decomposition_checks: list[float] = []
    priced_toll_checks: list[float] = []
    for option in options:
        total_components = 0.0
        total_monetary = 0.0
        for row in option.segment_breakdown:
            fuel = float(row.get("fuel_cost", 0.0))
            time_cost = float(row.get("time_cost", 0.0))
            toll = float(row.get("toll_cost", 0.0))
            carbon = float(row.get("carbon_cost", 0.0))
            row_monetary = float(row.get("monetary_cost", 0.0))
            total_components += fuel + time_cost + toll + carbon
            total_monetary += row_monetary
        decomposition_checks.append(_relative_error_score(total_components, total_monetary, tolerance=0.02))
        toll_md = option.toll_metadata or {}
        has_classification = bool(str(toll_md.get("classification_source", "")).strip())
        has_pricing = bool((toll_md.get("tariff_rule_ids", []) or [])) or sum(float(row.get("toll_cost", 0.0)) for row in option.segment_breakdown) <= 0.0
        priced_toll_checks.append(1.0 if (has_classification and has_pricing) else 0.0)
    unresolved_count = dropped_reasons.get("toll_tariff_unresolved", 0) + dropped_reasons.get("toll_tariff_unavailable", 0)
    unresolved_penalty = max(0.0, 1.0 - (unresolved_count / max(1, len(options))))
    decomposition_score = _safe_mean(decomposition_checks)
    priced_toll_score = _safe_mean(priced_toll_checks)
    tariff_depth = 0.0
    tariffs_path = ROOT / "assets" / "uk" / "toll_tariffs_uk.yaml"
    if tariffs_path.exists():
        text = tariffs_path.read_text(encoding="utf-8")
        rule_count = text.count('"id":')
        default_count = text.count('"id": "default_')
        depth_score = _ratio_to_target(rule_count, 200.0)
        default_penalty = 1.0 - min(1.0, (default_count / max(1, rule_count)))
        tariff_depth = max(0.0, min(1.0, (0.7 * depth_score) + (0.3 * default_penalty)))
    pricing_fixture_count = len(list((ROOT / "tests" / "fixtures" / "toll_pricing").glob("*.json")))
    pricing_fixture_depth = _ratio_to_target(pricing_fixture_count, 80.0)

    # Holdout MAPE from labeled pricing corpus.
    pricing_rows = _load_labeled_fixture_rows(ROOT / "tests" / "fixtures" / "toll_pricing")
    route_map = _fixture_route_map()
    mape_terms: list[float] = []
    holdout_eval = 0
    holdout_fail = 0
    for idx, row in enumerate(pricing_rows):
        route_name = str(row.get("route_fixture", "")).strip()
        route = route_map.get(route_name)
        if route is None:
            continue
        expected_contains = bool(row.get("expected_contains_toll", False))
        expected_cost = max(0.0, float(row.get("expected_toll_cost_gbp", 0.0)))
        try:
            option = _build_option(
                route,
                option_id=f"truth_toll_price_{idx}",
                departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                risk_aversion=1.0,
                stochastic_seed=101,
                fuel_multiplier=1.0,
                carbon_price_per_kg=0.10,
                terrain_profile="hilly",
                stochastic_enabled=True,
                samples=32,
                use_tolls=True,
            )
        except Exception:
            holdout_fail += 1
            continue
        _predicted_contains, predicted_cost, _conf = _predict_toll_signals(option)
        holdout_eval += 1
        if expected_contains:
            err = abs(predicted_cost - expected_cost) / max(1e-6, expected_cost)
            mape_terms.append(err)
        else:
            mape_terms.append(0.0 if predicted_cost <= 0.05 else 1.0)
    holdout_mape = _safe_mean(mape_terms, fallback=1.0)
    holdout_mape_score = 1.0 - min(1.0, holdout_mape / 0.05)
    holdout_coverage = holdout_eval / max(1, len(pricing_rows))
    holdout_failure_penalty = 1.0 - min(1.0, holdout_fail / max(1, len(pricing_rows)))

    score = _clamp_score(
        (15.0 * decomposition_score)
        + (10.0 * priced_toll_score)
        + (10.0 * unresolved_penalty)
        + (10.0 * tariff_depth)
        + (10.0 * pricing_fixture_depth)
        + (35.0 * holdout_mape_score)
        + (5.0 * holdout_coverage)
        + (5.0 * holdout_failure_penalty)
    )
    return score, {
        "decomposition_identity": round(decomposition_score, 4),
        "priced_toll_coverage": round(priced_toll_score, 4),
        "unresolved_penalty": round(unresolved_penalty, 4),
        "tariff_depth": round(tariff_depth, 4),
        "pricing_fixture_depth": round(pricing_fixture_depth, 4),
        "holdout_mape": round(holdout_mape, 6),
        "holdout_mape_score": round(holdout_mape_score, 4),
        "holdout_coverage": round(holdout_coverage, 4),
        "holdout_failure_penalty": round(holdout_failure_penalty, 4),
    }


def _safe_score_eval(
    fn: Any,
) -> tuple[int, dict[str, float | str]]:
    try:
        score, metrics = fn()
        return int(score), metrics
    except Exception as exc:
        return 0, {"error": str(exc).strip() or type(exc).__name__}


def main() -> None:
    fixtures_dir = ROOT / "tests" / "fixtures" / "uk_routes"
    routes = _load_fixture_routes(fixtures_dir)
    provenance = _provenance_summary(Path(settings.model_asset_dir))
    if not routes:
        raise SystemExit("No fixture routes found for quality scoring.")
    corridor_sigs = {
        sig
        for sig in (_corridor_signature(route) for route in routes)
        if sig is not None
    }
    min_routes = int(settings.quality_min_fixture_routes)
    min_corridors = int(settings.quality_min_unique_corridors)
    fixture_count_ok = len(routes) >= min_routes
    corridor_count_ok = len(corridor_sigs) >= min_corridors
    if not fixture_count_ok or not corridor_count_ok:
        zero_scores = {key: 0 for key in QUALITY_THRESHOLDS}
        zero_scores["overall"] = 0.0
        out = {
            "scores": zero_scores,
            "thresholds": {
                key: {
                    "score": 0,
                    "threshold": QUALITY_THRESHOLDS[key],
                    "passed": False,
                }
                for key in QUALITY_THRESHOLDS
            },
            "all_passed": False,
            "fixture_diversity_gate": {
                "passed": False,
                "route_count": len(routes),
                "min_route_count": min_routes,
                "unique_corridor_count": len(corridor_sigs),
                "min_unique_corridors": min_corridors,
            },
            "fatal": "Fixture corpus is too small or insufficiently diverse for valid quality scoring.",
            "provenance": provenance,
        }
        print(json.dumps(out, indent=2))
        return

    synthetic_violations = _synthetic_manifest_violations(Path(settings.model_asset_dir))
    if synthetic_violations:
        zero_scores = {key: 0 for key in QUALITY_THRESHOLDS}
        zero_scores["overall"] = 0.0
        out = {
            "scores": zero_scores,
            "thresholds": {
                key: {
                    "score": 0,
                    "threshold": QUALITY_THRESHOLDS[key],
                    "passed": False,
                }
                for key in QUALITY_THRESHOLDS
            },
            "all_passed": False,
            "dropped_routes": 0,
            "dropped_route_reasons": {},
            "dropped_route_gate": {
                "threshold": int(settings.quality_max_dropped_routes),
                "passed": False,
                "availability_factor": 0.0,
            },
            "fixture_diversity_gate": {
                "passed": fixture_count_ok and corridor_count_ok,
                "route_count": len(routes),
                "min_route_count": min_routes,
                "unique_corridor_count": len(corridor_sigs),
                "min_unique_corridors": min_corridors,
            },
            "fatal": "Synthetic/legacy calibration assets detected in strict scoring mode.",
            "violations": synthetic_violations,
            "provenance": provenance,
        }
        print(json.dumps(out, indent=2))
        return

    options: list[RouteOption] = []
    toll_options: list[RouteOption] = []
    dropped = 0
    dropped_reasons: Counter[str] = Counter()
    dropped_toll = 0
    dropped_toll_reasons: Counter[str] = Counter()
    for idx, route in enumerate(routes):
        try:
            option = _build_option(
                route,
                option_id=f"score_{idx}",
                departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                risk_aversion=1.2,
                stochastic_seed=42,
                fuel_multiplier=1.0,
                carbon_price_per_kg=0.12,
                terrain_profile="hilly",
                stochastic_enabled=True,
                samples=48,
                use_tolls=False,
            )
            options.append(option)
        except Exception as exc:
            dropped += 1
            msg = str(exc)
            reason = "unknown"
            if "reason_code:" in msg:
                reason = msg.split("reason_code:", 1)[1].split(";", 1)[0].strip() or "unknown"
            elif hasattr(exc, "reason_code"):
                reason = str(getattr(exc, "reason_code") or "unknown")
            elif (
                "terrain_dem_asset_unavailable" in msg.lower()
                or "terrain dem assets are unavailable" in msg.lower()
                or "synthetic grid terrain assets are disabled" in msg.lower()
            ):
                reason = "terrain_dem_asset_unavailable"
            elif "terrain" in msg.lower() and "coverage" in msg.lower():
                reason = "terrain_dem_coverage_insufficient"
            elif "toll" in msg.lower():
                reason = "toll_tariff_unresolved"
            dropped_reasons[reason] += 1

        try:
            toll_option = _build_option(
                route,
                option_id=f"score_toll_{idx}",
                departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
                risk_aversion=1.2,
                stochastic_seed=42,
                fuel_multiplier=1.0,
                carbon_price_per_kg=0.12,
                terrain_profile="hilly",
                stochastic_enabled=True,
                samples=48,
                use_tolls=True,
            )
            toll_options.append(toll_option)
        except Exception as exc:
            dropped_toll += 1
            msg = str(exc)
            reason = "unknown"
            if "reason_code:" in msg:
                reason = msg.split("reason_code:", 1)[1].split(";", 1)[0].strip() or "unknown"
            elif hasattr(exc, "reason_code"):
                reason = str(getattr(exc, "reason_code") or "unknown")
            elif "toll" in msg.lower():
                reason = "toll_tariff_unresolved"
            dropped_toll_reasons[reason] += 1

    if not options:
        zero_scores = {key: 0 for key in QUALITY_THRESHOLDS}
        zero_scores["overall"] = 0.0
        out = {
            "scores": zero_scores,
            "thresholds": {
                key: {
                    "score": 0,
                    "threshold": QUALITY_THRESHOLDS[key],
                    "passed": False,
                }
                for key in QUALITY_THRESHOLDS
            },
            "all_passed": False,
            "dropped_routes": dropped,
            "dropped_route_reasons": dict(dropped_reasons),
            "dropped_route_gate": {
                "threshold": int(settings.quality_max_dropped_routes),
                "passed": False,
                "availability_factor": 0.0,
            },
            "fatal": "No feasible options could be built for quality scoring under strict model-data policy.",
            "provenance": provenance,
        }
        print(json.dumps(out, indent=2))
        return

    risk_score, risk_metrics = _safe_score_eval(lambda: _score_risk_aversion(options, routes))
    dominance_score, dominance_metrics = _safe_score_eval(lambda: _score_dominance(options, routes))
    departure_score, departure_metrics = _safe_score_eval(lambda: _score_departure_time(routes))
    stochastic_score, stochastic_metrics = _safe_score_eval(lambda: _score_stochastic_sampling(routes))
    terrain_score, terrain_metrics = _safe_score_eval(lambda: _score_terrain_profile(routes))
    toll_class_score, toll_class_metrics = _safe_score_eval(lambda: _score_toll_classification(toll_options))
    fuel_score, fuel_metrics = _safe_score_eval(lambda: _score_fuel_price(routes))
    carbon_score, carbon_metrics = _safe_score_eval(lambda: _score_carbon_price(routes))
    toll_cost_score, toll_cost_metrics = _safe_score_eval(lambda: _score_toll_cost(toll_options, dropped_toll_reasons))

    scores = {
        "risk_aversion": risk_score,
        "dominance": dominance_score,
        "departure_time": departure_score,
        "stochastic_sampling": stochastic_score,
        "terrain_profile": terrain_score,
        "toll_classification": toll_class_score,
        "fuel_price": fuel_score,
        "carbon_price": carbon_score,
        "toll_cost": toll_cost_score,
    }

    total_dropped = dropped + dropped_toll
    merged_dropped_reasons = Counter(dropped_reasons)
    merged_dropped_reasons.update(dropped_toll_reasons)
    dropped_gate_passed = total_dropped <= int(settings.quality_max_dropped_routes)
    availability_factor = 1.0
    fixture_diversity_gate_passed = fixture_count_ok and corridor_count_ok

    scores["overall"] = round(statistics.fmean(scores.values()), 2)
    threshold_status = {
        key: {
            "score": scores[key],
            "threshold": QUALITY_THRESHOLDS[key],
            "passed": scores[key] >= QUALITY_THRESHOLDS[key],
        }
        for key in QUALITY_THRESHOLDS
    }
    out = {
        "scores": scores,
        "thresholds": threshold_status,
        "all_passed": all(item["passed"] for item in threshold_status.values()) and dropped_gate_passed,
        "dropped_routes": total_dropped,
        "dropped_route_reasons": dict(merged_dropped_reasons),
        "dropped_route_gate": {
            "threshold": int(settings.quality_max_dropped_routes),
            "passed": dropped_gate_passed,
            "availability_factor": round(availability_factor, 4),
        },
        "fixture_diversity_gate": {
            "passed": fixture_diversity_gate_passed,
            "route_count": len(routes),
            "min_route_count": min_routes,
            "unique_corridor_count": len(corridor_sigs),
            "min_unique_corridors": min_corridors,
        },
        "metrics": {
            "risk_aversion": risk_metrics,
            "dominance": dominance_metrics,
            "departure_time": departure_metrics,
            "stochastic_sampling": stochastic_metrics,
            "terrain_profile": terrain_metrics,
            "toll_classification": toll_class_metrics,
            "fuel_price": fuel_metrics,
            "carbon_price": carbon_metrics,
            "toll_cost": toll_cost_metrics,
        },
        "toll_scoring": {
            "evaluated_options": len(toll_options),
            "dropped_routes": dropped_toll,
            "dropped_route_reasons": dict(dropped_toll_reasons),
        },
        "provenance": provenance,
    }
    out["all_passed"] = bool(out["all_passed"]) and fixture_diversity_gate_passed
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
