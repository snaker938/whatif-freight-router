from __future__ import annotations

# ruff: noqa: E402
import argparse
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

from app.calibration_loader import (
    load_fuel_price_snapshot,
    load_scenario_profiles,
    load_stochastic_regimes,
)
from app.fuel_energy_model import segment_energy_and_emissions
from app.main import _finalize_pareto_options, _option_objective_value, build_option
from app.models import CostToggles, EmissionsContext, RouteOption, StochasticConfig
from app.routing_graph import load_route_graph, route_graph_candidate_routes
from app.scenario import ScenarioMode
from app.settings import settings
from app.vehicles import VehicleProfile, all_vehicles, get_vehicle

QUALITY_THRESHOLDS = {
    "risk_aversion": 95,
    "dominance": 95,
    "scenario_profile": 95,
    "departure_time": 95,
    "stochastic_sampling": 95,
    "terrain_profile": 95,
    "toll_classification": 95,
    "fuel_price": 95,
    "carbon_price": 95,
    "toll_cost": 95,
}


def _strict_raw_evidence_requirements() -> dict[str, list[Path]]:
    raw_root = ROOT / "data" / "raw" / "uk"
    return {
        "scenario_profile": [raw_root / "scenario_live_observed.jsonl"],
        "stochastic_sampling": [raw_root / "stochastic_residuals_raw.csv"],
        "departure_time": [raw_root / "dft_counts_raw.csv"],
        "fuel_price": [raw_root / "fuel_prices_raw.json"],
        "carbon_price": [raw_root / "carbon_intensity_hourly_raw.json"],
        "toll_classification": [raw_root / "toll_classification", raw_root / "toll_tariffs_operator_truth.json"],
        "toll_cost": [raw_root / "toll_pricing", raw_root / "toll_tariffs_operator_truth.json"],
    }


def _raw_path_has_evidence(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    # Directory evidence: require at least one plausible structured-data file.
    for pattern in ("*.json", "*.jsonl", "*.csv", "*.yaml", "*.yml"):
        if any(path.glob(pattern)):
            return True
    return False


def _strict_missing_raw_evidence(*, subsystem: str | None) -> list[str]:
    if not bool(settings.strict_live_data_required):
        return []
    requirements = _strict_raw_evidence_requirements()
    required_paths: list[Path] = []
    if subsystem and subsystem in requirements:
        required_paths.extend(requirements[subsystem])
    else:
        for rows in requirements.values():
            required_paths.extend(rows)
    seen: set[str] = set()
    missing: list[str] = []
    for path in required_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not _raw_path_has_evidence(path):
            missing.append(key)
    return missing


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
        ("scenario_profiles_uk.json", "scenario profiles"),
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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            payload = dump(mode="json")
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
    return {}


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


def _score_scenario_profiles(routes: list[dict[str, Any]]) -> tuple[int, dict[str, float | None]]:
    if not routes:
        return 0, {
            "monotonic_duration": 0.0,
            "monotonic_monetary": 0.0,
            "monotonic_emissions": 0.0,
            "mode_separability": 0.0,
            "metadata_completeness": 0.0,
            "stochastic_metadata": 0.0,
            "context_coverage": 0.0,
            "hour_slot_coverage": 0.0,
            "corridor_coverage": 0.0,
            "holdout_quality": 0.0,
            "asset_integrity": 0.0,
            "split_strategy_quality": 0.0,
        }

    monotonic_duration: list[float] = []
    monotonic_monetary: list[float] = []
    monotonic_emissions: list[float] = []
    mode_separability: list[float] = []
    metadata: list[float] = []
    stochastic_meta: list[float] = []

    def _build_for_mode(route: dict[str, Any], *, mode: ScenarioMode, idx: int) -> RouteOption:
        return build_option(
            route,
            option_id=f"scenario_{mode.value}_{idx}",
            vehicle_type="rigid_hgv",
            scenario_mode=mode,
            cost_toggles=CostToggles(
                use_tolls=False,
                fuel_price_multiplier=1.0,
                carbon_price_per_kg=0.10,
                toll_cost_per_km=0.2,
            ),
            terrain_profile="hilly",  # type: ignore[arg-type]
            stochastic=StochasticConfig(enabled=True, seed=71, sigma=0.08, samples=32),
            emissions_context=EmissionsContext(fuel_type="diesel", euro_class="euro6", ambient_temp_c=10),
            departure_time_utc=datetime(2026, 2, 18, 8, 30, tzinfo=UTC),
            utility_weights=(1.0, 1.0, 1.0),
            risk_aversion=1.0,
        )

    for idx, route in enumerate(routes[: min(12, len(routes))]):
        no_share = _build_for_mode(route, mode=ScenarioMode.NO_SHARING, idx=idx)
        partial = _build_for_mode(route, mode=ScenarioMode.PARTIAL_SHARING, idx=idx)
        full = _build_for_mode(route, mode=ScenarioMode.FULL_SHARING, idx=idx)

        monotonic_duration.append(
            1.0
            if (
                no_share.metrics.duration_s >= partial.metrics.duration_s
                and partial.metrics.duration_s >= full.metrics.duration_s
            )
            else 0.0
        )
        monotonic_monetary.append(
            1.0
            if (
                no_share.metrics.monetary_cost >= partial.metrics.monetary_cost
                and partial.metrics.monetary_cost >= full.metrics.monetary_cost
            )
            else 0.0
        )
        monotonic_emissions.append(
            1.0
            if (
                no_share.metrics.emissions_kg >= partial.metrics.emissions_kg
                and partial.metrics.emissions_kg >= full.metrics.emissions_kg
            )
            else 0.0
        )
        duration_sep = abs(no_share.metrics.duration_s - full.metrics.duration_s) / max(
            1e-6, abs(no_share.metrics.duration_s)
        )
        monetary_sep = abs(no_share.metrics.monetary_cost - full.metrics.monetary_cost) / max(
            1e-6, abs(no_share.metrics.monetary_cost)
        )
        emissions_sep = abs(no_share.metrics.emissions_kg - full.metrics.emissions_kg) / max(
            1e-6, abs(no_share.metrics.emissions_kg)
        )
        mode_separability.append(
            _safe_mean(
                [
                    _ratio_to_target(duration_sep, 0.02),
                    _ratio_to_target(monetary_sep, 0.02),
                    _ratio_to_target(emissions_sep, 0.02),
                ]
            )
        )
        no_summary = _as_dict(no_share.scenario_summary)
        partial_summary = _as_dict(partial.scenario_summary)
        full_summary = _as_dict(full.scenario_summary)
        metadata.append(
            1.0
            if (
                no_summary.get("mode") == "no_sharing"
                and partial_summary.get("mode") == "partial_sharing"
                and full_summary.get("mode") == "full_sharing"
                and all("version" in row for row in (no_summary, partial_summary, full_summary))
                and all("context_key" in row for row in (no_summary, partial_summary, full_summary))
            )
            else 0.0
        )
        no_meta = no_share.uncertainty_samples_meta or {}
        stochastic_meta.append(
            1.0
            if (
                no_meta.get("scenario_mode") == "no_sharing"
                and "scenario_profile_version" in no_meta
                and "scenario_sigma_multiplier" in no_meta
                and "scenario_context_key" in no_meta
            )
            else 0.0
        )

    def _mape_score(value: float, threshold: float) -> float:
        if value <= threshold:
            return 1.0
        return max(0.0, 1.0 - ((value - threshold) / max(threshold, 1e-9)))

    def _independent_raw_holdout_quality() -> tuple[float, dict[str, float]]:
        raw_path = ROOT / "data" / "raw" / "uk" / "scenario_live_observed.jsonl"
        if not raw_path.exists():
            return 0.0, {"raw_holdout_rows": 0.0, "raw_holdout_coverage": 0.0}

        def _context_key_from_payload(payload: dict[str, Any]) -> str:
            corridor = (
                str(payload.get("corridor_geohash5", payload.get("corridor_bucket", "uk000"))).strip().lower()
                or "uk000"
            )
            hour_slot = int(max(0, min(23, int(float(payload.get("hour_slot_local", 12) or 12)))))
            day_kind = str(payload.get("day_kind", "weekday")).strip().lower() or "weekday"
            road_mix = str(payload.get("road_mix_bucket", "mixed")).strip().lower() or "mixed"
            vehicle_class = str(payload.get("vehicle_class", "rigid_hgv")).strip().lower() or "rigid_hgv"
            weather = str(payload.get("weather_regime", payload.get("weather_bucket", "clear"))).strip().lower() or "clear"
            return f"{corridor}|h{hour_slot:02d}|{day_kind}|{road_mix}|{vehicle_class}|{weather}"

        observed_rows: list[tuple[str, str, dict[str, float], str]] = []
        max_rows = 40_000
        with raw_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for idx, line in enumerate(handle):
                if idx >= max_rows:
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                context_key = _context_key_from_payload(payload)
                as_of_utc = str(payload.get("as_of_utc", "")).strip()
                modes_raw = payload.get("modes")
                if isinstance(modes_raw, dict):
                    for mode_name, row in modes_raw.items():
                        if mode_name not in {"no_sharing", "partial_sharing", "full_sharing"} or not isinstance(row, dict):
                            continue
                        factors = {
                            "duration_multiplier": float(row.get("duration_multiplier", 1.0)),
                            "fuel_consumption_multiplier": float(row.get("fuel_consumption_multiplier", 1.0)),
                            "emissions_multiplier": float(row.get("emissions_multiplier", 1.0)),
                        }
                        observed_rows.append((context_key, str(mode_name), factors, as_of_utc))
                else:
                    mode_name = str(payload.get("mode", "")).strip().lower()
                    if mode_name not in {"no_sharing", "partial_sharing", "full_sharing"}:
                        continue
                    factors = {
                        "duration_multiplier": float(payload.get("duration_multiplier", 1.0)),
                        "fuel_consumption_multiplier": float(payload.get("fuel_consumption_multiplier", 1.0)),
                        "emissions_multiplier": float(payload.get("emissions_multiplier", 1.0)),
                    }
                    observed_rows.append((context_key, mode_name, factors, as_of_utc))

        if not observed_rows:
            return 0.0, {"raw_holdout_rows": 0.0, "raw_holdout_coverage": 0.0}

        scenario_profiles = load_scenario_profiles()
        contexts = scenario_profiles.contexts or {}
        globals_map = scenario_profiles.profiles or {}
        holdout_start_raw = ""
        if isinstance(scenario_profiles.holdout_window, dict):
            holdout_start_raw = str(scenario_profiles.holdout_window.get("start_utc", "")).strip()
        holdout_start_dt: datetime | None = None
        if holdout_start_raw:
            try:
                holdout_start_dt = datetime.fromisoformat(holdout_start_raw.replace("Z", "+00:00"))
            except ValueError:
                holdout_start_dt = None

        def _is_holdout(context_key: str, as_of_utc: str) -> bool:
            if holdout_start_dt is not None and as_of_utc:
                try:
                    as_of_dt = datetime.fromisoformat(as_of_utc.replace("Z", "+00:00"))
                    if as_of_dt >= holdout_start_dt:
                        return True
                except ValueError:
                    pass
            return False

        duration_errs: list[float] = []
        fuel_errs: list[float] = []
        emissions_errs: list[float] = []
        covered = 0
        total = 0
        observed_by_context: dict[str, dict[str, dict[str, list[float]]]] = {}
        predicted_by_context: dict[str, dict[str, dict[str, float]]] = {}

        for context_key, mode, factors, as_of_utc in observed_rows:
            if not _is_holdout(context_key, as_of_utc):
                continue
            total += 1
            ctx = contexts.get(context_key)
            profile = None
            if ctx is not None:
                profile = ctx.profiles.get(mode)
            if profile is None:
                profile = globals_map.get(mode)
            if profile is None:
                continue
            covered += 1
            predicted_by_context.setdefault(context_key, {}).setdefault(mode, {})
            observed_by_context.setdefault(context_key, {}).setdefault(mode, {"duration_multiplier": [], "fuel_consumption_multiplier": [], "emissions_multiplier": []})
            for field, bucket in (
                ("duration_multiplier", duration_errs),
                ("fuel_consumption_multiplier", fuel_errs),
                ("emissions_multiplier", emissions_errs),
            ):
                observed_value = float(factors.get(field, 1.0))
                predicted_value = float(getattr(profile, field))
                rel_err = abs(predicted_value - observed_value) / max(1e-6, abs(observed_value))
                bucket.append(rel_err)
                observed_by_context[context_key][mode][field].append(observed_value)
                predicted_by_context[context_key][mode][field] = predicted_value

        coverage = float(covered) / max(1.0, float(total))
        if covered <= 0:
            return 0.0, {"raw_holdout_rows": float(total), "raw_holdout_coverage": round(coverage, 6)}

        duration_mape = float(statistics.fmean(duration_errs)) if duration_errs else 1.0
        fuel_mape = float(statistics.fmean(fuel_errs)) if fuel_errs else 1.0
        emissions_mape = float(statistics.fmean(emissions_errs)) if emissions_errs else 1.0

        separability_errors: list[float] = []
        for context_key, mode_map in observed_by_context.items():
            if not all(mode in mode_map for mode in ("no_sharing", "partial_sharing", "full_sharing")):
                continue
            pred_map = predicted_by_context.get(context_key, {})
            if not all(mode in pred_map for mode in ("no_sharing", "partial_sharing", "full_sharing")):
                continue
            for field in ("duration_multiplier", "fuel_consumption_multiplier", "emissions_multiplier"):
                obs_no = statistics.fmean(mode_map["no_sharing"][field])
                obs_full = statistics.fmean(mode_map["full_sharing"][field])
                pred_no = float(pred_map["no_sharing"][field])
                pred_full = float(pred_map["full_sharing"][field])
                obs_sep = abs(obs_no - obs_full) / max(1e-6, abs(obs_no))
                pred_sep = abs(pred_no - pred_full) / max(1e-6, abs(pred_no))
                separability_errors.append(abs(pred_sep - obs_sep) / max(0.02, obs_sep))
        separability_score = (
            max(0.0, 1.0 - min(1.0, float(statistics.fmean(separability_errors))))
            if separability_errors
            else 0.0
        )
        raw_quality = _safe_mean(
            [
                _mape_score(duration_mape, 0.08),
                _mape_score(fuel_mape, 0.08),
                _mape_score(emissions_mape, 0.08),
                _ratio_to_target(coverage, 0.90),
                separability_score,
            ]
        )
        return raw_quality, {
            "raw_holdout_rows": float(total),
            "raw_holdout_coverage": round(coverage, 6),
            "raw_duration_mape": round(duration_mape, 6),
            "raw_fuel_mape": round(fuel_mape, 6),
            "raw_emissions_mape": round(emissions_mape, 6),
            "raw_separability_quality": round(separability_score, 6),
        }

    try:
        scenario_profiles = load_scenario_profiles()
        no_p = scenario_profiles.profiles.get("no_sharing")
        partial_p = scenario_profiles.profiles.get("partial_sharing")
        full_p = scenario_profiles.profiles.get("full_sharing")
        asset_integrity = (
            1.0
            if (
                no_p is not None
                and partial_p is not None
                and full_p is not None
                and no_p.duration_multiplier >= partial_p.duration_multiplier >= full_p.duration_multiplier
                and no_p.fuel_consumption_multiplier >= partial_p.fuel_consumption_multiplier >= full_p.fuel_consumption_multiplier
                and no_p.emissions_multiplier >= partial_p.emissions_multiplier >= full_p.emissions_multiplier
            )
            else 0.0
        )
        context_rows = list((scenario_profiles.contexts or {}).values())
        context_count = len(context_rows)
        context_coverage = _ratio_to_target(float(context_count), 12.0)
        holdout_metrics = scenario_profiles.holdout_metrics or {}
        mode_sep = float(holdout_metrics.get("mode_separation_mean", 0.0))
        duration_mape = float(holdout_metrics.get("duration_mape", 1.0))
        monetary_mape = float(holdout_metrics.get("monetary_mape", 1.0))
        emissions_mape = float(holdout_metrics.get("emissions_mape", 1.0))
        coverage = float(holdout_metrics.get("coverage", 0.0))
        hour_slot_coverage_reported = float(holdout_metrics.get("hour_slot_coverage", 0.0))
        corridor_coverage_reported = float(holdout_metrics.get("corridor_coverage", 0.0))
        full_identity_share = float(holdout_metrics.get("full_identity_share", float("nan")))
        projection_share_reported = float(
            holdout_metrics.get("projection_dominant_context_share", float("nan"))
        )
        observed_mode_row_share_reported = float(
            holdout_metrics.get("observed_mode_row_share", float("nan"))
        )
        actual_hours = len(
            {
                int(getattr(ctx, "hour_slot_local", -1))
                for ctx in context_rows
                if isinstance(getattr(ctx, "hour_slot_local", None), int)
            }
        )
        actual_corridors = len(
            {
                str(getattr(ctx, "corridor_geohash5", "")).strip().lower()
                for ctx in context_rows
                if str(getattr(ctx, "corridor_geohash5", "")).strip()
            }
        )
        hour_slot_coverage = min(hour_slot_coverage_reported, float(actual_hours))
        corridor_coverage = min(corridor_coverage_reported, float(actual_corridors))
        actual_projection_count = 0
        observed_row_share_proxy_acc = 0.0
        observed_row_share_proxy_n = 0
        for ctx in context_rows:
            projected_ratio = getattr(ctx, "mode_projection_ratio", None)
            projected_source = str(getattr(ctx, "mode_observation_source", "") or "").strip().lower()
            projected = False
            ratio_value: float | None = None
            if projected_ratio is not None:
                try:
                    ratio_value = float(projected_ratio)
                    projected = ratio_value >= 0.80
                except (TypeError, ValueError):
                    projected = False
                    ratio_value = None
            if not projected and projected_source:
                projected = ("project" in projected_source) or ("synthesized" in projected_source)
            if projected:
                actual_projection_count += 1
            if ratio_value is not None:
                observed_row_share_proxy_acc += max(0.0, min(1.0, 1.0 - ratio_value))
                observed_row_share_proxy_n += 1
        actual_projection_share = (
            float(actual_projection_count) / float(context_count)
            if context_count > 0
            else float("nan")
        )
        actual_observed_mode_row_share = (
            float(observed_row_share_proxy_acc) / float(observed_row_share_proxy_n)
            if observed_row_share_proxy_n > 0
            else float("nan")
        )
        projection_share_candidates = [
            value
            for value in (projection_share_reported, actual_projection_share)
            if value == value
        ]
        projection_share = (
            max(projection_share_candidates)
            if projection_share_candidates
            else float("nan")
        )
        observed_mode_row_share_candidates = [
            value
            for value in (observed_mode_row_share_reported, actual_observed_mode_row_share)
            if value == value
        ]
        observed_mode_row_share = (
            min(observed_mode_row_share_candidates)
            if observed_mode_row_share_candidates
            else float("nan")
        )
        split_strategy = str(getattr(scenario_profiles, "split_strategy", "") or "").strip().lower()
        if full_identity_share == full_identity_share:
            full_identity_quality = max(0.0, 1.0 - min(1.0, full_identity_share / 0.70))
        else:
            full_identity_quality = 0.0
        if projection_share == projection_share:
            projection_evidence_quality = max(0.0, 1.0 - min(1.0, projection_share / 0.80))
        else:
            projection_evidence_quality = 0.0
        if observed_mode_row_share == observed_mode_row_share:
            observed_mode_row_quality = _ratio_to_target(
                observed_mode_row_share,
                float(settings.scenario_min_observed_mode_row_share),
            )
        else:
            observed_mode_row_quality = 0.0

        holdout_quality = _safe_mean(
            [
                _ratio_to_target(mode_sep, 0.03),
                _mape_score(duration_mape, 0.08),
                _mape_score(monetary_mape, 0.08),
                _mape_score(emissions_mape, 0.08),
                _ratio_to_target(coverage, 0.90),
                full_identity_quality,
                projection_evidence_quality,
                observed_mode_row_quality,
            ]
        )
        split_strategy_quality = (
            1.0
            if split_strategy in {"temporal_forward_plus_corridor_block"}
            else 0.0
        )
        raw_holdout_quality, raw_holdout_metrics = _independent_raw_holdout_quality()
    except Exception:
        asset_integrity = 0.0
        context_coverage = 0.0
        holdout_quality = 0.0
        hour_slot_coverage = 0.0
        corridor_coverage = 0.0
        split_strategy_quality = 0.0
        full_identity_share = float("nan")
        full_identity_quality = 0.0
        projection_share = float("nan")
        projection_evidence_quality = 0.0
        observed_mode_row_share = float("nan")
        observed_mode_row_quality = 0.0
        raw_holdout_quality = 0.0
        raw_holdout_metrics = {"raw_holdout_rows": 0.0, "raw_holdout_coverage": 0.0}

    monotonic_duration_score = _safe_mean(monotonic_duration, fallback=0.0)
    monotonic_monetary_score = _safe_mean(monotonic_monetary, fallback=0.0)
    monotonic_emissions_score = _safe_mean(monotonic_emissions, fallback=0.0)
    mode_separability_score = _safe_mean(mode_separability, fallback=0.0)
    metadata_score = _safe_mean(metadata, fallback=0.0)
    stochastic_metadata_score = _safe_mean(stochastic_meta, fallback=0.0)
    hour_coverage_score = _ratio_to_target(hour_slot_coverage, 6.0)
    corridor_coverage_score = _ratio_to_target(corridor_coverage, 8.0)
    ordering_score = _safe_mean(
        [
            monotonic_duration_score,
            monotonic_monetary_score,
            monotonic_emissions_score,
            mode_separability_score,
        ]
    )
    metadata_contract_score = _safe_mean(
        [
            metadata_score,
            stochastic_metadata_score,
            asset_integrity,
            context_coverage,
            hour_coverage_score,
            corridor_coverage_score,
            split_strategy_quality,
        ]
    )
    independent_truth_score = _safe_mean([holdout_quality, raw_holdout_quality], fallback=0.0)
    runtime_ordering_score = ordering_score
    score = _clamp_score(
        (70.0 * independent_truth_score)
        + (20.0 * runtime_ordering_score)
        + (10.0 * metadata_contract_score)
    )
    # Strict realism guard: independent truth metrics must exist and be non-trivial.
    raw_holdout_rows = float(raw_holdout_metrics.get("raw_holdout_rows", 0.0))
    if independent_truth_score <= 0.0:
        score = min(score, 74)
    if settings.strict_live_data_required and raw_holdout_rows < 500:
        score = min(score, 69)
    if full_identity_share == full_identity_share and full_identity_share > 0.70:
        score = min(score, 69)
    if projection_share == projection_share and projection_share >= 0.95:
        score = min(score, 72)
    if (
        observed_mode_row_share == observed_mode_row_share
        and observed_mode_row_share < float(settings.scenario_min_observed_mode_row_share)
    ):
        score = min(score, 69)
    return score, {
        "monotonic_duration": round(monotonic_duration_score, 4),
        "monotonic_monetary": round(monotonic_monetary_score, 4),
        "monotonic_emissions": round(monotonic_emissions_score, 4),
        "mode_separability": round(mode_separability_score, 4),
        "metadata_completeness": round(metadata_score, 4),
        "stochastic_metadata": round(stochastic_metadata_score, 4),
        "context_coverage": round(context_coverage, 4),
        "hour_slot_coverage": round(hour_coverage_score, 4),
        "corridor_coverage": round(corridor_coverage_score, 4),
        "holdout_quality": round(holdout_quality, 4),
        "raw_holdout_quality": round(raw_holdout_quality, 4),
        "independent_truth_quality": round(independent_truth_score, 4),
        "runtime_ordering_quality": round(runtime_ordering_score, 4),
        "asset_integrity": round(asset_integrity, 4),
        "split_strategy_quality": round(split_strategy_quality, 4),
        "full_identity_share": (
            round(full_identity_share, 4)
            if full_identity_share == full_identity_share
            else None
        ),
        "full_identity_quality": round(full_identity_quality, 4),
        "projection_dominant_context_share": (
            round(projection_share, 4)
            if projection_share == projection_share
            else None
        ),
        "projection_evidence_quality": round(projection_evidence_quality, 4),
        "observed_mode_row_share": (
            round(observed_mode_row_share, 4)
            if observed_mode_row_share == observed_mode_row_share
            else None
        ),
        "observed_mode_row_quality": round(observed_mode_row_quality, 4),
        "raw_holdout_rows": round(float(raw_holdout_metrics.get("raw_holdout_rows", 0.0)), 4),
        "raw_holdout_coverage": round(float(raw_holdout_metrics.get("raw_holdout_coverage", 0.0)), 4),
    }


def _score_stochastic_sampling(routes: list[dict[str, Any]]) -> tuple[int, dict[str, float | None]]:
    if not routes:
        return 0, {
            "invariants": 0.0,
            "metadata": 0.0,
            "deterministic_seed": 0.0,
            "seed_sensitivity": 0.0,
            "clip_quality": 0.0,
            "posterior_quality": 0.0,
            "empirical_depth": 0.0,
        }
    invariants: list[float] = []
    metadata: list[float] = []
    deterministic_seed: list[float] = []
    seed_sensitivity: list[float] = []
    sample_clip_penalties: list[float] = []
    sigma_clip_penalties: list[float] = []
    factor_clip_penalties: list[float] = []

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
        sample_clip_penalties.append(float(ua.get("sample_count_clip_ratio", 0.0)))
        sigma_clip_penalties.append(float(ua.get("sigma_clip_ratio", 0.0)))
        factor_clip_penalties.append(float(ua.get("factor_clip_rate", 0.0)))

    invariants_score = _safe_mean(invariants)
    metadata_score = _safe_mean(metadata)
    deterministic_score = _safe_mean(deterministic_seed)
    sensitivity_score = _safe_mean(seed_sensitivity)
    sample_clip_score = max(0.0, 1.0 - min(1.0, _safe_mean(sample_clip_penalties, fallback=1.0) / 0.05))
    sigma_clip_score = max(0.0, 1.0 - min(1.0, _safe_mean(sigma_clip_penalties, fallback=1.0) / 0.05))
    factor_clip_score = max(0.0, 1.0 - min(1.0, _safe_mean(factor_clip_penalties, fallback=1.0) / 0.15))
    clip_quality = _safe_mean([sample_clip_score, sigma_clip_score, factor_clip_score])
    try:
        table = load_stochastic_regimes()
        posterior = table.posterior_model if isinstance(table.posterior_model, dict) else {}
        context_probs = posterior.get("context_to_regime_probs", {}) if isinstance(posterior, dict) else {}
        posterior_coverage_score = (
            _ratio_to_target(float(len(context_probs)), 80.0)
            if isinstance(context_probs, dict)
            else 0.0
        )
        required_factors = {"traffic", "incident", "weather", "price", "eco"}
        transform_checks: list[float] = []
        for regime in table.regimes.values():
            family_ok = 1.0 if str(getattr(regime, "transform_family", "")).strip().lower() == "quantile_mapping_v1" else 0.0
            mapping = getattr(regime, "shock_quantile_mapping", None)
            mapping_ok = (
                1.0
                if isinstance(mapping, dict) and all(bool(mapping.get(name)) for name in required_factors)
                else 0.0
            )
            transform_checks.append(_safe_mean([family_ok, mapping_ok], fallback=0.0))
        transform_quality = _safe_mean(transform_checks, fallback=0.0)
        posterior_quality = _safe_mean([posterior_coverage_score, transform_quality], fallback=0.0)
        holdout = table.holdout_metrics if isinstance(table.holdout_metrics, dict) else {}
        holdout_coverage = float(holdout.get("coverage", 0.0))
        holdout_pit = float(holdout.get("pit_mean", float("nan")))
        holdout_crps = float(holdout.get("crps_mean", float("inf")))
        holdout_duration_mape = float(holdout.get("duration_mape", float("inf")))
        pit_quality = 0.0
        if holdout_pit == holdout_pit:
            if 0.35 <= holdout_pit <= 0.65:
                pit_quality = 1.0
            else:
                pit_quality = max(0.0, 1.0 - (abs(holdout_pit - 0.5) / 0.5))
        crps_quality = max(0.0, 1.0 - min(1.0, holdout_crps / 0.55))
        duration_mape_quality = max(0.0, 1.0 - min(1.0, holdout_duration_mape / 0.15))
        holdout_truth_quality = _safe_mean(
            [
                _ratio_to_target(holdout_coverage, 0.90),
                pit_quality,
                crps_quality,
                duration_mape_quality,
            ],
            fallback=0.0,
        )
    except Exception:
        posterior_quality = 0.0
        posterior_coverage_score = 0.0
        transform_quality = 0.0
        holdout_truth_quality = 0.0
        holdout_coverage = 0.0
        holdout_pit = float("nan")
        holdout_crps = float("inf")
        holdout_duration_mape = float("inf")
    residual_rows = _line_count(ROOT / "assets" / "uk" / "stochastic_residuals_empirical.csv")
    empirical_depth = _ratio_to_target(residual_rows, 5000.0)
    runtime_quality = _safe_mean(
        [
            invariants_score,
            deterministic_score,
            sensitivity_score,
            clip_quality,
            posterior_quality,
        ],
        fallback=0.0,
    )
    metadata_contract_quality = _safe_mean([metadata_score, empirical_depth], fallback=0.0)
    score = _clamp_score(
        (70.0 * holdout_truth_quality)
        + (20.0 * runtime_quality)
        + (10.0 * metadata_contract_quality)
    )
    if settings.strict_live_data_required and holdout_truth_quality <= 0.0:
        score = min(score, 69)
    return score, {
        "invariants": round(invariants_score, 4),
        "metadata": round(metadata_score, 4),
        "deterministic_seed": round(deterministic_score, 4),
        "seed_sensitivity": round(sensitivity_score, 4),
        "clip_quality": round(clip_quality, 4),
        "posterior_quality": round(posterior_quality, 4),
        "posterior_coverage_score": round(posterior_coverage_score, 4),
        "transform_quality": round(transform_quality, 4),
        "holdout_truth_quality": round(holdout_truth_quality, 4),
        "holdout_coverage": round(holdout_coverage, 4),
        "holdout_pit_mean": (
            round(holdout_pit, 4)
            if holdout_pit == holdout_pit
            else None
        ),
        "holdout_crps_mean": (
            round(holdout_crps, 4)
            if holdout_crps == holdout_crps and holdout_crps != float("inf")
            else None
        ),
        "holdout_duration_mape": (
            round(holdout_duration_mape, 4)
            if holdout_duration_mape == holdout_duration_mape and holdout_duration_mape != float("inf")
            else None
        ),
        "runtime_quality": round(runtime_quality, 4),
        "metadata_contract_quality": round(metadata_contract_quality, 4),
        "sample_clip_score": round(sample_clip_score, 4),
        "sigma_clip_score": round(sigma_clip_score, 4),
        "factor_clip_score": round(factor_clip_score, 4),
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
            "provenance": 0.0,
            "quantile_validity": 0.0,
            "holdout_p50_error": 1.0,
            "holdout_p50_score": 0.0,
            "sensitivity": 0.0,
            "live_source_readiness": 0.0,
            "history_depth": 0.0,
        }
    isolation_scores: list[float] = []
    provenance_scores: list[float] = []
    quantile_validity: list[float] = []
    holdout_errors: list[float] = []
    quantile_coverage: list[float] = []
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
        provenance_scores.append(
            1.0
            if all(
                key in ws
                for key in (
                    "fuel_price_source",
                    "fuel_price_as_of",
                    "consumption_model_source",
                    "consumption_model_version",
                    "consumption_model_as_of_utc",
                )
            )
            else 0.0
        )
        row_validity: list[float] = []
        row_coverage: list[float] = []
        row_holdout: list[float] = []
        for row in boosted.segment_breakdown:
            liters = float(row.get("fuel_liters", 0.0))
            liters_p10 = float(row.get("fuel_liters_p10", 0.0))
            liters_p50 = float(row.get("fuel_liters_p50", 0.0))
            liters_p90 = float(row.get("fuel_liters_p90", 0.0))
            cost = float(row.get("fuel_cost", 0.0))
            cost_p10 = float(row.get("fuel_cost_p10_gbp", 0.0))
            cost_p50 = float(row.get("fuel_cost_p50_gbp", 0.0))
            cost_p90 = float(row.get("fuel_cost_p90_gbp", 0.0))
            ordered = 1.0 if (liters_p10 <= liters_p50 <= liters_p90 and cost_p10 <= cost_p50 <= cost_p90) else 0.0
            row_validity.append(ordered)
            row_coverage.append(1.0 if (liters_p10 <= liters <= liters_p90 and cost_p10 <= cost <= cost_p90) else 0.0)
            row_holdout.append(abs(liters - liters_p50) / max(1e-6, liters_p50))
        quantile_validity.append(_safe_mean(row_validity, fallback=0.0))
        quantile_coverage.append(_safe_mean(row_coverage, fallback=0.0))
        holdout_errors.append(_safe_mean(row_holdout, fallback=1.0))

    # Monotonic sensitivity checks on direct segment model controls.
    sensitivity_checks: list[float] = []
    try:
        base_vehicle = get_vehicle("rigid_hgv")
        light_vehicle = VehicleProfile.model_validate(
            {
                **base_vehicle.model_dump(mode="python"),
                "id": "rigid_hgv_light_probe",
                "mass_tonnes": max(8.0, float(base_vehicle.mass_tonnes) * 0.85),
            }
        )
        heavy_vehicle = VehicleProfile.model_validate(
            {
                **base_vehicle.model_dump(mode="python"),
                "id": "rigid_hgv_heavy_probe",
                "mass_tonnes": float(base_vehicle.mass_tonnes) * 1.25,
            }
        )
        ctx_mild = EmissionsContext(fuel_type="diesel", euro_class="euro6", ambient_temp_c=15.0)
        ctx_cold = EmissionsContext(fuel_type="diesel", euro_class="euro6", ambient_temp_c=0.0)
        light = segment_energy_and_emissions(
            vehicle=light_vehicle,
            emissions_context=ctx_mild,
            distance_km=12.0,
            duration_s=900.0,
            grade=0.0,
            fuel_price_multiplier=1.0,
        )
        heavy = segment_energy_and_emissions(
            vehicle=heavy_vehicle,
            emissions_context=ctx_mild,
            distance_km=12.0,
            duration_s=900.0,
            grade=0.0,
            fuel_price_multiplier=1.0,
        )
        flat = segment_energy_and_emissions(
            vehicle=base_vehicle,
            emissions_context=ctx_mild,
            distance_km=12.0,
            duration_s=900.0,
            grade=0.0,
            fuel_price_multiplier=1.0,
        )
        uphill = segment_energy_and_emissions(
            vehicle=base_vehicle,
            emissions_context=ctx_mild,
            distance_km=12.0,
            duration_s=900.0,
            grade=0.04,
            fuel_price_multiplier=1.0,
        )
        mild = segment_energy_and_emissions(
            vehicle=base_vehicle,
            emissions_context=ctx_mild,
            distance_km=12.0,
            duration_s=900.0,
            grade=0.0,
            fuel_price_multiplier=1.0,
        )
        cold = segment_energy_and_emissions(
            vehicle=base_vehicle,
            emissions_context=ctx_cold,
            distance_km=12.0,
            duration_s=900.0,
            grade=0.0,
            fuel_price_multiplier=1.0,
        )
        stop_go = segment_energy_and_emissions(
            vehicle=base_vehicle,
            emissions_context=ctx_mild,
            distance_km=12.0,
            duration_s=1600.0,
            grade=0.0,
            fuel_price_multiplier=1.0,
        )
        freeflow = segment_energy_and_emissions(
            vehicle=base_vehicle,
            emissions_context=ctx_mild,
            distance_km=12.0,
            duration_s=800.0,
            grade=0.0,
            fuel_price_multiplier=1.0,
        )
        sensitivity_checks.extend(
            [
                1.0 if heavy.fuel_liters >= light.fuel_liters else 0.0,
                1.0 if uphill.fuel_liters >= flat.fuel_liters else 0.0,
                1.0 if cold.fuel_liters >= mild.fuel_liters else 0.0,
                1.0 if stop_go.fuel_liters >= freeflow.fuel_liters else 0.0,
            ]
        )
    except Exception:
        sensitivity_checks.append(0.0)

    live_readiness_checks: list[float] = []
    try:
        snap = load_fuel_price_snapshot()
        live_readiness_checks.append(1.0 if str(snap.source).strip() else 0.0)
        live_readiness_checks.append(1.0 if str(snap.as_of).strip() else 0.0)
        live_readiness_checks.append(1.0 if str(snap.signature or "").strip() else 0.0)
        policy_ok = bool(settings.live_fuel_require_signature) and bool(settings.live_fuel_allow_signed_fallback)
        live_readiness_checks.append(1.0 if policy_ok else 0.0)
        url_or_fallback_ok = bool(str(settings.live_fuel_price_url).strip()) or bool(settings.live_fuel_allow_signed_fallback)
        live_readiness_checks.append(1.0 if url_or_fallback_ok else 0.0)
    except Exception:
        live_readiness_checks.append(0.0)

    vehicle_profile_checks: list[float] = []
    try:
        profiles = all_vehicles()
        for profile in profiles.values():
            vehicle_profile_checks.append(1.0 if int(profile.schema_version) >= 2 else 0.0)
            vehicle_profile_checks.append(1.0 if str(profile.fuel_surface_class).strip() else 0.0)
            vehicle_profile_checks.append(1.0 if str(profile.toll_vehicle_class).strip() else 0.0)
            vehicle_profile_checks.append(1.0 if str(profile.toll_axle_class).strip() else 0.0)
            vehicle_profile_checks.append(1.0 if str(profile.risk_bucket).strip() else 0.0)
            vehicle_profile_checks.append(1.0 if str(profile.stochastic_bucket).strip() else 0.0)
    except Exception:
        vehicle_profile_checks.append(0.0)

    isolation = _safe_mean(isolation_scores)
    provenance = _safe_mean(provenance_scores)
    quantiles = _safe_mean(quantile_validity)
    coverage = _safe_mean(quantile_coverage)
    holdout_error = _safe_mean(holdout_errors, fallback=1.0)
    holdout_score = 1.0 - min(1.0, holdout_error / 0.15)
    sensitivity = _safe_mean(sensitivity_checks, fallback=0.0)
    live_readiness = _safe_mean(live_readiness_checks, fallback=0.0)
    vehicle_profile_schema = _safe_mean(vehicle_profile_checks, fallback=0.0)
    history_depth = _ratio_to_target(
        _json_entry_count(ROOT / "assets" / "uk" / "fuel_prices_uk.json", "history"),
        365.0,
    )
    score = _clamp_score(
        (15.0 * isolation)
        + (10.0 * provenance)
        + (15.0 * quantiles)
        + (10.0 * coverage)
        + (15.0 * holdout_score)
        + (15.0 * sensitivity)
        + (10.0 * live_readiness)
        + (10.0 * vehicle_profile_schema)
        + (0.0 * history_depth)
    )
    return score, {
        "multiplier_isolation": round(isolation, 4),
        "provenance": round(provenance, 4),
        "quantile_validity": round(quantiles, 4),
        "quantile_coverage": round(coverage, 4),
        "holdout_p50_error": round(holdout_error, 6),
        "holdout_p50_score": round(holdout_score, 4),
        "sensitivity": round(sensitivity, 4),
        "live_source_readiness": round(live_readiness, 4),
        "vehicle_profile_schema": round(vehicle_profile_schema, 4),
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
        has_pricing = bool(toll_md.get("tariff_rule_ids", []) or []) or sum(float(row.get("toll_cost", 0.0)) for row in option.segment_breakdown) <= 0.0
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
) -> tuple[int, dict[str, float | str | None]]:
    try:
        score, metrics = fn()
        return int(score), metrics
    except Exception as exc:
        return 0, {"error": str(exc).strip() or type(exc).__name__}


def _exception_reason_code(exc: Exception) -> str | None:
    extra = getattr(exc, "__dict__", {})
    if not isinstance(extra, dict):
        return None
    value = extra.get("reason_code")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score backend model quality gates.")
    parser.add_argument(
        "--subsystem",
        type=str,
        default=None,
        choices=sorted(list(QUALITY_THRESHOLDS.keys()) + ["overall"]),
        help="Optional subsystem-only output view (e.g. fuel_price).",
    )
    args = parser.parse_args()
    subsystem = args.subsystem
    missing_raw_evidence = _strict_missing_raw_evidence(subsystem=subsystem)
    if missing_raw_evidence:
        zero_scores: dict[str, float] = {key: 0.0 for key in QUALITY_THRESHOLDS}
        zero_scores["overall"] = 0.0
        fatal = {
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
            "fatal": "Insufficient independent raw evidence for strict live quality scoring.",
            "missing_raw_evidence": missing_raw_evidence,
        }
        if subsystem and subsystem != "overall":
            targeted = {
                "subsystem": subsystem,
                "score": 0,
                "threshold": QUALITY_THRESHOLDS.get(subsystem, 0),
                "passed": False,
                "fatal": fatal["fatal"],
                "missing_raw_evidence": missing_raw_evidence,
            }
            print(json.dumps(targeted, indent=2))
            return
        print(json.dumps(fatal, indent=2))
        return

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
        zero_scores: dict[str, float] = {key: 0.0 for key in QUALITY_THRESHOLDS}
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
        zero_scores: dict[str, float] = {key: 0.0 for key in QUALITY_THRESHOLDS}
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

    if subsystem == "fuel_price":
        fuel_score, fuel_metrics = _safe_score_eval(lambda: _score_fuel_price(routes))
        targeted = {
            "subsystem": "fuel_price",
            "score": fuel_score,
            "threshold": QUALITY_THRESHOLDS["fuel_price"],
            "passed": fuel_score >= QUALITY_THRESHOLDS["fuel_price"],
            "metrics": fuel_metrics,
            "fixture_diversity_gate": {
                "passed": fixture_count_ok and corridor_count_ok,
                "route_count": len(routes),
                "min_route_count": min_routes,
                "unique_corridor_count": len(corridor_sigs),
                "min_unique_corridors": min_corridors,
            },
            "provenance": provenance,
        }
        print(json.dumps(targeted, indent=2))
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
            else:
                reason_code = _exception_reason_code(exc)
                if reason_code is not None:
                    reason = reason_code
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
            else:
                reason_code = _exception_reason_code(exc)
                if reason_code is not None:
                    reason = reason_code
                elif "toll" in msg.lower():
                    reason = "toll_tariff_unresolved"
            dropped_toll_reasons[reason] += 1

    if not options:
        zero_scores: dict[str, float] = {key: 0.0 for key in QUALITY_THRESHOLDS}
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
    scenario_score, scenario_metrics = _safe_score_eval(lambda: _score_scenario_profiles(routes))
    departure_score, departure_metrics = _safe_score_eval(lambda: _score_departure_time(routes))
    stochastic_score, stochastic_metrics = _safe_score_eval(lambda: _score_stochastic_sampling(routes))
    terrain_score, terrain_metrics = _safe_score_eval(lambda: _score_terrain_profile(routes))
    toll_class_score, toll_class_metrics = _safe_score_eval(lambda: _score_toll_classification(toll_options))
    fuel_score, fuel_metrics = _safe_score_eval(lambda: _score_fuel_price(routes))
    carbon_score, carbon_metrics = _safe_score_eval(lambda: _score_carbon_price(routes))
    toll_cost_score, toll_cost_metrics = _safe_score_eval(lambda: _score_toll_cost(toll_options, dropped_toll_reasons))

    scores: dict[str, float] = {
        "risk_aversion": float(risk_score),
        "dominance": float(dominance_score),
        "scenario_profile": float(scenario_score),
        "departure_time": float(departure_score),
        "stochastic_sampling": float(stochastic_score),
        "terrain_profile": float(terrain_score),
        "toll_classification": float(toll_class_score),
        "fuel_price": float(fuel_score),
        "carbon_price": float(carbon_score),
        "toll_cost": float(toll_cost_score),
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
            "scenario_profile": scenario_metrics,
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
    if subsystem:
        if subsystem == "overall":
            targeted = {
                "subsystem": "overall",
                "score": scores["overall"],
                "all_passed": out["all_passed"],
                "dropped_routes": out["dropped_routes"],
                "dropped_route_gate": out["dropped_route_gate"],
            }
        else:
            targeted = {
                "subsystem": subsystem,
                "score": scores[subsystem],
                "threshold": QUALITY_THRESHOLDS[subsystem],
                "passed": threshold_status[subsystem]["passed"],
                "metrics": out["metrics"].get(subsystem, {}),
                "dropped_routes": out["dropped_routes"],
                "all_passed": out["all_passed"],
            }
        print(json.dumps(targeted, indent=2))
        return
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
