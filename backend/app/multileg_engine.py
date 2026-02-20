from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from .models import GeoJSONLineString, LatLng, RouteMetrics, RouteOption, Waypoint


@dataclass(frozen=True)
class MultiLegComposeResult:
    options: list[RouteOption]
    warnings: list[str]
    candidate_fetches: int


@dataclass(frozen=True)
class _ChainState:
    id: str
    leg_options: tuple[RouteOption, ...]
    composed: RouteOption


LegSolver = Callable[
    [int, LatLng, LatLng],
    Awaitable[
        tuple[list[RouteOption], list[str], int]
        | tuple[list[RouteOption], list[str], int, Any]
        | tuple[list[RouteOption], list[str], int, Any, Any]
    ],
]
ParetoSelector = Callable[[list[RouteOption], int], list[RouteOption]]


def _compose_nodes(origin: LatLng, destination: LatLng, waypoints: Sequence[Waypoint]) -> list[LatLng]:
    nodes: list[LatLng] = [origin]
    for waypoint in waypoints:
        nodes.append(LatLng(lat=waypoint.lat, lon=waypoint.lon))
    nodes.append(destination)
    return nodes


def _build_chain_id(leg_options: Sequence[RouteOption]) -> str:
    if not leg_options:
        return "chain_empty"
    return "chain_" + "__".join(option.id for option in leg_options)


def _merge_geometry(leg_options: Sequence[RouteOption]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for index, option in enumerate(leg_options):
        coords = option.geometry.coordinates
        if not coords:
            continue
        if index == 0 or not merged:
            merged.extend((float(lon), float(lat)) for lon, lat in coords)
            continue
        first = coords[0]
        if merged and math.isclose(merged[-1][0], float(first[0]), abs_tol=1e-9) and math.isclose(
            merged[-1][1],
            float(first[1]),
            abs_tol=1e-9,
        ):
            merged.extend((float(lon), float(lat)) for lon, lat in coords[1:])
        else:
            merged.extend((float(lon), float(lat)) for lon, lat in coords)
    return merged


def _aggregate_uncertainty(options: Sequence[RouteOption]) -> dict[str, float] | None:
    uncertainty_rows = [option.uncertainty for option in options if option.uncertainty]
    if not uncertainty_rows:
        return None

    sum_keys = (
        "mean_duration_s",
        "q50_duration_s",
        "q90_duration_s",
        "q95_duration_s",
        "p95_duration_s",
        "cvar95_duration_s",
        "mean_monetary_cost",
        "q50_monetary_cost",
        "q90_monetary_cost",
        "q95_monetary_cost",
        "p95_monetary_cost",
        "cvar95_monetary_cost",
        "mean_emissions_kg",
        "q50_emissions_kg",
        "q90_emissions_kg",
        "q95_emissions_kg",
        "p95_emissions_kg",
        "cvar95_emissions_kg",
    )
    mean_keys = (
        "utility_mean",
        "utility_q95",
        "utility_cvar95",
        "robust_score",
    )
    std_keys = (
        "std_duration_s",
        "std_monetary_cost",
        "std_emissions_kg",
    )

    out: dict[str, float] = {}
    for key in sum_keys:
        out[key] = round(
            sum(float(row.get(key, 0.0)) for row in uncertainty_rows if isinstance(row, dict)),
            6,
        )

    for key in std_keys:
        variance = sum(float(row.get(key, 0.0)) ** 2 for row in uncertainty_rows if isinstance(row, dict))
        out[key] = round(math.sqrt(max(0.0, variance)), 6)
    for key in mean_keys:
        values = [float(row.get(key, 0.0)) for row in uncertainty_rows if isinstance(row, dict) and key in row]
        if values:
            out[key] = round(sum(values) / len(values), 6)

    return out


def _aggregate_uncertainty_meta(options: Sequence[RouteOption]) -> dict[str, str | float | int | bool] | None:
    rows = [option.uncertainty_samples_meta for option in options if option.uncertainty_samples_meta]
    if not rows:
        return None
    sample_count = 0
    seed_material: list[str] = []
    regime_ids: list[str] = []
    copula_ids: list[str] = []
    calibration_versions: list[str] = []
    seed_strategies: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_count += int(row.get("sample_count", 0))
        seed_material.append(str(row.get("seed", "none")))
        if row.get("regime_id") is not None:
            regime_ids.append(str(row.get("regime_id")))
        if row.get("copula_id") is not None:
            copula_ids.append(str(row.get("copula_id")))
        if row.get("calibration_version") is not None:
            calibration_versions.append(str(row.get("calibration_version")))
        if row.get("seed_strategy") is not None:
            seed_strategies.append(str(row.get("seed_strategy")))
    return {
        "sample_count": sample_count,
        "seed": "|".join(seed_material),
        "leg_count": len(options),
        "regime_id": "|".join(sorted(set(regime_ids))),
        "copula_id": "|".join(sorted(set(copula_ids))),
        "calibration_version": "|".join(sorted(set(calibration_versions))),
        "seed_strategy": "|".join(sorted(set(seed_strategies))),
    }


def _compose_route_option(leg_options: Sequence[RouteOption], *, option_id: str) -> RouteOption:
    geometry = _merge_geometry(leg_options)
    if len(geometry) < 2:
        geometry = [(0.0, 0.0), (0.0, 0.0)]

    distance_km = sum(float(option.metrics.distance_km) for option in leg_options)
    duration_s = sum(float(option.metrics.duration_s) for option in leg_options)
    monetary_cost = sum(float(option.metrics.monetary_cost) for option in leg_options)
    emissions_kg = sum(float(option.metrics.emissions_kg) for option in leg_options)
    weather_delay_s = sum(float(option.metrics.weather_delay_s or 0.0) for option in leg_options)
    incident_delay_s = sum(float(option.metrics.incident_delay_s or 0.0) for option in leg_options)
    energy_values = [option.metrics.energy_kwh for option in leg_options if option.metrics.energy_kwh is not None]
    energy_kwh = sum(float(value) for value in energy_values) if energy_values else None
    avg_speed_kmh = (distance_km / (duration_s / 3600.0)) if duration_s > 0 else 0.0

    leg_meta: list[dict[str, str | float | int | bool]] = []
    segment_breakdown: list[dict[str, float | int]] = []
    segment_index = 0
    toll_conf_rows: list[float] = []
    toll_assets: list[str] = []
    toll_rule_ids: list[str] = []
    incident_events = []
    eta_explanations: list[str] = []
    terrain_sources: list[str] = []
    terrain_coverage: list[float] = []
    terrain_spacing: list[float] = []
    terrain_ascent = 0.0
    terrain_descent = 0.0
    terrain_confidence: list[float] = []
    terrain_fail_closed = False
    terrain_versions: set[str] = set()
    terrain_hist_weighted: dict[str, float] = {}
    terrain_hist_weight_total = 0.0

    for leg_index, option in enumerate(leg_options):
        leg_meta.append(
            {
                "leg_index": leg_index,
                "option_id": option.id,
                "distance_km": round(float(option.metrics.distance_km), 6),
                "duration_s": round(float(option.metrics.duration_s), 6),
                "monetary_cost": round(float(option.metrics.monetary_cost), 6),
                "emissions_kg": round(float(option.metrics.emissions_kg), 6),
            }
        )
        if option.toll_confidence is not None:
            toll_conf_rows.append(float(option.toll_confidence))
        if option.toll_metadata:
            assets = option.toll_metadata.get("matched_assets")
            if isinstance(assets, list):
                toll_assets.extend(str(item) for item in assets)
            rule_ids = option.toll_metadata.get("tariff_rule_ids")
            if isinstance(rule_ids, list):
                toll_rule_ids.extend(str(item) for item in rule_ids)

        if option.segment_breakdown:
            for row in option.segment_breakdown:
                next_row = dict(row)
                next_row["segment_index"] = segment_index
                segment_index += 1
                segment_breakdown.append(next_row)
        if option.terrain_summary is not None:
            terrain_sources.append(str(option.terrain_summary.source))
            terrain_coverage.append(float(option.terrain_summary.coverage_ratio))
            terrain_spacing.append(float(option.terrain_summary.sample_spacing_m))
            terrain_ascent += float(option.terrain_summary.ascent_m)
            terrain_descent += float(option.terrain_summary.descent_m)
            terrain_confidence.append(float(option.terrain_summary.confidence))
            terrain_fail_closed = terrain_fail_closed or bool(option.terrain_summary.fail_closed_applied)
            terrain_versions.add(str(option.terrain_summary.version))
            hist = option.terrain_summary.grade_histogram or {}
            leg_distance_weight = max(0.001, float(option.metrics.distance_km))
            for key, value in hist.items():
                terrain_hist_weighted[key] = terrain_hist_weighted.get(key, 0.0) + (
                    float(value) * leg_distance_weight
                )
            terrain_hist_weight_total += leg_distance_weight
        if option.incident_events:
            incident_events.extend(option.incident_events)
        if option.eta_explanations:
            eta_explanations.extend([f"Leg {leg_index + 1}: {item}" for item in option.eta_explanations])

    uncertainty = _aggregate_uncertainty(leg_options)
    uncertainty_meta = _aggregate_uncertainty_meta(leg_options)
    toll_confidence = (sum(toll_conf_rows) / len(toll_conf_rows)) if toll_conf_rows else None

    terrain_summary = None
    if terrain_sources:
        hist = (
            {
                key: round(val / max(1e-9, terrain_hist_weight_total), 6)
                for key, val in terrain_hist_weighted.items()
            }
            if terrain_hist_weight_total > 0
            else {}
        )
        source_rank = {"dem_real": 2, "missing": 1, "unsupported_region": 0}
        terrain_summary = {
            "source": sorted(terrain_sources, key=lambda item: source_rank.get(item, 0), reverse=True)[0],
            "coverage_ratio": round(min(terrain_coverage), 6),
            "sample_spacing_m": round(sum(terrain_spacing) / max(1, len(terrain_spacing)), 3),
            "ascent_m": round(terrain_ascent, 3),
            "descent_m": round(terrain_descent, 3),
            "grade_histogram": hist,
            "confidence": round(sum(terrain_confidence) / max(1, len(terrain_confidence)), 6),
            "fail_closed_applied": terrain_fail_closed,
            "version": "|".join(sorted(terrain_versions)) if terrain_versions else "unknown",
        }

    return RouteOption(
        id=option_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=geometry),
        metrics=RouteMetrics(
            distance_km=round(distance_km, 3),
            duration_s=round(duration_s, 2),
            monetary_cost=round(monetary_cost, 2),
            emissions_kg=round(emissions_kg, 3),
            avg_speed_kmh=round(avg_speed_kmh, 2),
            energy_kwh=round(energy_kwh, 3) if energy_kwh is not None else None,
            weather_delay_s=round(weather_delay_s, 2),
            incident_delay_s=round(incident_delay_s, 2),
        ),
        eta_explanations=eta_explanations,
        segment_breakdown=segment_breakdown,
        uncertainty=uncertainty,
        uncertainty_samples_meta=uncertainty_meta,
        legs=leg_meta,
        toll_confidence=round(toll_confidence, 6) if toll_confidence is not None else None,
        toll_metadata={
            "matched_assets": sorted(set(toll_assets)),
            "tariff_rule_ids": sorted(set(toll_rule_ids)),
            "fallback_policy_used": False,
        },
        terrain_summary=terrain_summary,
        incident_events=incident_events,
    )


async def compose_multileg_route_options(
    *,
    origin: LatLng,
    destination: LatLng,
    waypoints: Sequence[Waypoint],
    max_alternatives: int,
    leg_solver: LegSolver,
    pareto_selector: ParetoSelector,
    option_prefix: str,
) -> MultiLegComposeResult:
    nodes = _compose_nodes(origin, destination, waypoints)
    if len(nodes) < 2:
        return MultiLegComposeResult(options=[], warnings=[], candidate_fetches=0)

    warnings: list[str] = []
    candidate_fetches = 0

    leg_options_all: list[list[RouteOption]] = []
    for leg_index in range(len(nodes) - 1):
        leg_origin = nodes[leg_index]
        leg_destination = nodes[leg_index + 1]
        leg_result = await leg_solver(leg_index, leg_origin, leg_destination)
        if len(leg_result) == 5:
            leg_options, leg_warnings, leg_fetches, _leg_diag, _leg_candidate_diag = leg_result
        elif len(leg_result) == 4:
            leg_options, leg_warnings, leg_fetches, _leg_diag = leg_result
        else:
            leg_options, leg_warnings, leg_fetches = leg_result
        warnings.extend(leg_warnings)
        candidate_fetches += leg_fetches
        if not leg_options:
            return MultiLegComposeResult(
                options=[],
                warnings=warnings,
                candidate_fetches=candidate_fetches,
            )
        leg_options_all.append(leg_options)

    beam_limit = max(max_alternatives, min(64, max_alternatives * 12))
    states: list[_ChainState] = []
    first_leg = leg_options_all[0]
    for option in first_leg:
        chain = (option,)
        chain_id = _build_chain_id(chain)
        states.append(
            _ChainState(
                id=chain_id,
                leg_options=chain,
                composed=_compose_route_option(chain, option_id=f"{option_prefix}_{chain_id}"),
            )
        )

    for leg_index in range(1, len(leg_options_all)):
        expanded: list[_ChainState] = []
        for state in states:
            for option in leg_options_all[leg_index]:
                chain = tuple([*state.leg_options, option])
                chain_id = _build_chain_id(chain)
                expanded.append(
                    _ChainState(
                        id=chain_id,
                        leg_options=chain,
                        composed=_compose_route_option(chain, option_id=f"{option_prefix}_{chain_id}"),
                    )
                )
        if not expanded:
            break

        expanded_options = [state.composed for state in expanded]
        pruned_options = pareto_selector(expanded_options, beam_limit)
        keep_ids = {option.id for option in pruned_options}
        states = [state for state in expanded if state.composed.id in keep_ids]
        states.sort(
            key=lambda state: (
                state.composed.metrics.duration_s,
                state.composed.metrics.monetary_cost,
                state.composed.metrics.emissions_kg,
                state.composed.id,
            )
        )
        if len(states) > beam_limit:
            states = states[:beam_limit]

    final_options = [state.composed for state in states]
    final_options = pareto_selector(final_options, max_alternatives)
    return MultiLegComposeResult(
        options=final_options,
        warnings=warnings,
        candidate_fetches=candidate_fetches,
    )
