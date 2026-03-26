from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

# REFC uses bounded stress-world analysis rather than an unconstrained noise
# model. The construction is closer to global sensitivity analysis and explicit
# stress testing than to learned simulation; see Saltelli et al., "Global
# Sensitivity Analysis: The Primer", https://doi.org/10.1002/9780470725184 .
# The route certificate itself is an empirical winner-frequency estimate over
# those worlds, following the sample-average-approximation perspective used in
# stochastic programming; see Shapiro, Dentcheva, Ruszczynski, "Lectures on
# Stochastic Programming", https://doi.org/10.1137/1.9780898718751 .

EVIDENCE_FAMILIES: tuple[str, ...] = (
    "scenario",
    "toll",
    "terrain",
    "fuel",
    "carbon",
    "weather",
    "stochastic",
)
EVIDENCE_STATES: tuple[str, ...] = (
    "nominal",
    "mildly_stale",
    "severely_stale",
    "low_confidence",
    "proxy",
    "refreshed",
)

OBJECTIVE_NAMES: tuple[str, str, str] = ("time", "money", "co2")

DEFAULT_STATE_EFFECTS: dict[str, tuple[float, float, float]] = {
    "nominal": (0.00, 0.00, 0.00),
    "mildly_stale": (0.040, 0.030, 0.032),
    "severely_stale": (0.140, 0.110, 0.095),
    "low_confidence": (0.078, 0.062, 0.056),
    "proxy": (0.185, 0.152, 0.132),
    "refreshed": (-0.038, -0.030, -0.024),
}

# Thesis-defined bounded perturbation priors. These are explicit scenario
# states used for replayable fragility analysis, not a learned stochastic model.
DEFAULT_FAMILY_SENSITIVITY: dict[str, tuple[float, float, float]] = {
    "scenario": (0.82, 0.34, 0.24),
    "toll": (0.05, 1.00, 0.12),
    "terrain": (0.52, 0.22, 0.78),
    "fuel": (0.12, 0.94, 0.30),
    "carbon": (0.00, 0.90, 0.78),
    "weather": (0.72, 0.18, 0.18),
    "stochastic": (0.68, 0.62, 0.52),
}

DEFAULT_OBJECTIVE_BIAS: dict[str, dict[str, float]] = {
    "time": {
        "scenario": 0.35,
        "weather": 0.25,
        "terrain": 0.20,
        "stochastic": 0.15,
        "toll": 0.05,
        "fuel": 0.05,
        "carbon": 0.02,
    },
    "money": {
        "fuel": 0.35,
        "toll": 0.30,
        "carbon": 0.20,
        "scenario": 0.05,
        "weather": 0.02,
        "terrain": 0.03,
        "stochastic": 0.05,
    },
    "co2": {
        "terrain": 0.30,
        "fuel": 0.25,
        "carbon": 0.25,
        "scenario": 0.10,
        "weather": 0.03,
        "stochastic": 0.07,
        "toll": 0.00,
    },
}

SelectorScoreMapFn = Callable[
    [Sequence[Mapping[str, Any]], Mapping[str, tuple[float, float, float]]],
    Mapping[str, float],
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _as_float(value, default)))


def _stable_hash(parts: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _route_id(route: Mapping[str, Any]) -> str:
    explicit = str(route.get("route_id", route.get("id", ""))).strip()
    if explicit:
        return explicit
    payload = json.dumps(route, sort_keys=True, separators=(",", ":"), default=str)
    return _stable_hash([payload])


def _objective_vector(route: Mapping[str, Any]) -> tuple[float, float, float]:
    if "objective_vector" in route:
        raw = route["objective_vector"]
        if isinstance(raw, Mapping):
            return (
                _as_float(raw.get("time")),
                _as_float(raw.get("money")),
                _as_float(raw.get("co2")),
            )
        if isinstance(raw, Sequence) and len(raw) >= 3:
            return (_as_float(raw[0]), _as_float(raw[1]), _as_float(raw[2]))
    metrics = route.get("metrics")
    if isinstance(metrics, Mapping):
        return (
            _as_float(metrics.get("duration_s")),
            _as_float(metrics.get("monetary_cost")),
            _as_float(metrics.get("emissions_kg")),
        )
    return (
        _as_float(route.get("time")),
        _as_float(route.get("money")),
        _as_float(route.get("co2")),
    )


def _evidence_tensor(route: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    raw = route.get("evidence_tensor", route.get("dependency_tensor", {}))
    if isinstance(raw, Mapping):
        out: dict[str, dict[str, float]] = {}
        for family, objective_map in raw.items():
            if not isinstance(objective_map, Mapping):
                continue
            inner: dict[str, float] = {}
            for objective, weight in objective_map.items():
                inner[str(objective)] = max(0.0, min(1.0, _as_float(weight)))
            if inner:
                out[str(family)] = inner
        return out
    return {}


def _route_provenance(route: Mapping[str, Any]) -> dict[str, Any]:
    raw = route.get("evidence_provenance", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def _route_active_families(route: Mapping[str, Any]) -> set[str]:
    tensor = _evidence_tensor(route)
    provenance = _route_provenance(route)
    active: set[str] = set()
    for family, objective_map in tensor.items():
        if any(_as_float(weight) > 0.0 for weight in objective_map.values()):
            active.add(str(family))
    active_families = provenance.get("active_families", [])
    if isinstance(active_families, Sequence) and not isinstance(active_families, (str, bytes)):
        active.update(str(family) for family in active_families)
    raw_route_specific = provenance.get("dependency_weights", {})
    if isinstance(raw_route_specific, Mapping):
        for family, family_weights in raw_route_specific.items():
            if isinstance(family_weights, Mapping) and any(_as_float(value) > 0.0 for value in family_weights.values()):
                active.add(str(family))
    raw_families = provenance.get("families", [])
    if isinstance(raw_families, Sequence) and not isinstance(raw_families, (str, bytes)):
        for entry in raw_families:
            if not isinstance(entry, Mapping):
                continue
            family = str(entry.get("family", "")).strip()
            if family and bool(entry.get("active", True)):
                active.add(family)
    return active


def _family_provenance_rows(
    routes: Sequence[Mapping[str, Any]],
    family: str,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for route in routes:
        provenance = _route_provenance(route)
        raw_families = provenance.get("families", [])
        if not isinstance(raw_families, Sequence) or isinstance(raw_families, (str, bytes)):
            continue
        for entry in raw_families:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("family", "")).strip() == family:
                rows.append(entry)
    return rows


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _route_segment_cost_totals(route: Mapping[str, Any]) -> tuple[float, float, float]:
    raw_rows = route.get("segment_breakdown", [])
    rows = raw_rows if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)) else []
    toll_cost = 0.0
    fuel_cost = 0.0
    carbon_cost = 0.0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        toll_cost += _as_float(row.get("toll_cost"))
        fuel_cost += _as_float(row.get("fuel_cost"))
        carbon_cost += _as_float(row.get("carbon_cost"))
    return toll_cost, fuel_cost, carbon_cost


def _route_operational_dependency_overrides(
    route: Mapping[str, Any],
    *,
    active_families: Sequence[str],
) -> dict[str, dict[str, float]]:
    duration_s, money_cost, emissions_kg = _objective_vector(route)
    metrics = _mapping_or_empty(route.get("metrics"))
    weather_summary = _mapping_or_empty(route.get("weather_summary"))
    terrain_summary = _mapping_or_empty(route.get("terrain_summary"))
    scenario_summary = _mapping_or_empty(route.get("scenario_summary"))
    uncertainty = _mapping_or_empty(route.get("uncertainty"))

    distance_km = max(
        1.0,
        _as_float(metrics.get("distance_km"), _as_float(route.get("distance_km"), 0.0)),
    )
    cost_scale = max(
        1.0,
        abs(money_cost),
        _as_float(metrics.get("monetary_cost"), 0.0),
    )
    emissions_scale = max(1.0, abs(emissions_kg))

    toll_cost, fuel_cost, carbon_cost = _route_segment_cost_totals(route)
    toll_share = _clamp01(toll_cost / cost_scale)
    fuel_share = _clamp01(fuel_cost / cost_scale)
    carbon_share = _clamp01(carbon_cost / cost_scale)

    weather_delay_s = max(
        _as_float(metrics.get("weather_delay_s"), 0.0),
        _as_float(weather_summary.get("weather_delay_s"), 0.0),
    )
    incident_delay_s = _as_float(metrics.get("incident_delay_s"), 0.0)
    weather_time_pressure = _clamp01(
        (weather_delay_s + (0.50 * incident_delay_s)) / max(1.0, max(duration_s, 0.0) * 0.25),
    )

    ascent_m = max(0.0, _as_float(terrain_summary.get("ascent_m"), 0.0))
    descent_m = max(0.0, _as_float(terrain_summary.get("descent_m"), 0.0))
    terrain_pressure = _clamp01(
        (ascent_m + (0.35 * descent_m)) / max(1.0, distance_km * 120.0),
    )

    duration_multiplier = _as_float(scenario_summary.get("duration_multiplier"), 1.0)
    incident_rate_multiplier = _as_float(scenario_summary.get("incident_rate_multiplier"), 1.0)
    incident_delay_multiplier = _as_float(scenario_summary.get("incident_delay_multiplier"), 1.0)
    fuel_multiplier = _as_float(scenario_summary.get("fuel_consumption_multiplier"), 1.0)
    emissions_multiplier = _as_float(scenario_summary.get("emissions_multiplier"), 1.0)

    scenario_time_pressure = _clamp01(
        abs(duration_multiplier - 1.0)
        + (0.35 * abs(incident_rate_multiplier - 1.0))
        + (0.35 * abs(incident_delay_multiplier - 1.0)),
    )
    scenario_money_pressure = _clamp01(
        abs(fuel_multiplier - 1.0)
        + (0.30 * abs(duration_multiplier - 1.0)),
    )
    scenario_co2_pressure = _clamp01(
        abs(emissions_multiplier - 1.0)
        + (0.25 * abs(fuel_multiplier - 1.0)),
    )

    std_duration_s = _as_float(uncertainty.get("std_duration_s"), 0.0)
    std_money_cost = _as_float(uncertainty.get("std_monetary_cost"), 0.0)
    std_emissions_kg = _as_float(uncertainty.get("std_emissions_kg"), 0.0)
    stochastic_time_pressure = _clamp01(std_duration_s / max(1.0, max(duration_s, 0.0) * 0.25))
    stochastic_money_pressure = _clamp01(std_money_cost / max(1.0, cost_scale * 0.25))
    stochastic_co2_pressure = _clamp01(std_emissions_kg / max(1.0, emissions_scale * 0.25))

    emissions_intensity = _clamp01(emissions_kg / max(1.0, distance_km * 2.5))
    fuel_exposure = max(fuel_share, _clamp01((fuel_cost + (0.10 * emissions_kg)) / cost_scale))
    carbon_exposure = max(carbon_share, emissions_intensity)

    overrides = {
        objective: {family: 0.0 for family in active_families}
        for objective in OBJECTIVE_NAMES
    }
    if "scenario" in overrides["time"]:
        overrides["time"]["scenario"] += 0.32 * scenario_time_pressure
        overrides["money"]["scenario"] += 0.22 * scenario_money_pressure
        overrides["co2"]["scenario"] += 0.28 * scenario_co2_pressure
    if "weather" in overrides["time"]:
        overrides["time"]["weather"] += 0.52 * weather_time_pressure
        overrides["money"]["weather"] += 0.08 * weather_time_pressure
        overrides["co2"]["weather"] += 0.12 * weather_time_pressure
    if "terrain" in overrides["time"]:
        overrides["time"]["terrain"] += 0.34 * terrain_pressure
        overrides["money"]["terrain"] += 0.12 * terrain_pressure
        overrides["co2"]["terrain"] += 0.52 * terrain_pressure
    if "toll" in overrides["time"]:
        overrides["time"]["toll"] += 0.08 * toll_share
        overrides["money"]["toll"] += 0.66 * toll_share
    if "fuel" in overrides["time"]:
        overrides["time"]["fuel"] += 0.04 * fuel_exposure
        overrides["money"]["fuel"] += 0.56 * fuel_exposure
        overrides["co2"]["fuel"] += 0.38 * fuel_exposure
    if "carbon" in overrides["money"]:
        overrides["money"]["carbon"] += 0.46 * carbon_exposure
        overrides["co2"]["carbon"] += 0.34 * carbon_exposure
    if "stochastic" in overrides["time"]:
        overrides["time"]["stochastic"] += 0.32 * stochastic_time_pressure
        overrides["money"]["stochastic"] += 0.22 * stochastic_money_pressure
        overrides["co2"]["stochastic"] += 0.22 * stochastic_co2_pressure
    return overrides


def _family_state_weights(
    routes: Sequence[Mapping[str, Any]],
    family: str,
    *,
    state_catalog: Sequence[str],
    ambiguity_context: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    base = {
        "nominal": 0.36,
        "mildly_stale": 0.18,
        "severely_stale": 0.10,
        "low_confidence": 0.10,
        "proxy": 0.06,
        "refreshed": 0.20,
    }
    rows = _family_provenance_rows(routes, family)
    confidences = [_as_float(row.get("confidence"), 0.85) for row in rows]
    coverage = [_as_float(row.get("coverage_ratio"), 1.0) for row in rows]
    freshness_present = sum(1 for row in rows if str(row.get("freshness_timestamp_utc", "")).strip())
    proxy_marked = sum(
        1
        for row in rows
        if _contains_strict_deny_token(
            row.get("source"),
            row.get("signature"),
            row.get("snapshot_id"),
            row.get("fallback_source"),
        )
    )
    fallback_used = sum(1 for row in rows if bool(row.get("fallback_used", False)))
    confidence_mean = sum(confidences) / float(len(confidences) or 1)
    coverage_mean = sum(coverage) / float(len(coverage) or 1)
    freshness_share = freshness_present / float(len(rows) or 1)
    weakness = max(
        0.0,
        min(
            1.0,
            (0.45 * (1.0 - confidence_mean))
            + (0.25 * (1.0 - coverage_mean))
            + (0.20 * (1.0 - freshness_share))
            + (0.10 * min(1.0, (proxy_marked + fallback_used) / float(len(rows) or 1))),
        ),
    )
    context = _normalised_ambiguity_context(ambiguity_context)
    context_boost = _family_context_boost(family, context)
    support_strength = _clamp01(context.get("support_strength"), 0.0)
    margin_pressure = 1.0 - _clamp01(context.get("od_nominal_margin_proxy"), 1.0)
    path_pressure = min(1.0, max(0.0, (_as_float(context.get("od_candidate_path_count")) - 1.0) / 5.0))
    family_pressure = min(1.0, max(0.0, (_as_float(context.get("od_corridor_family_count")) - 1.0) / 3.0))
    stress_selectivity = _clamp01(
        (0.42 * support_strength)
        + (0.22 * margin_pressure)
        + (0.14 * path_pressure)
        + (0.12 * family_pressure)
        + (0.10 * _clamp01(context.get("od_objective_spread"), 0.0))
    )
    supported_case = bool(context.get("is_hard_case") or context.get("is_supported_ambiguity_case"))
    weights = dict(base)
    weights["nominal"] += (0.18 * (1.0 - weakness))
    weights["refreshed"] += (0.10 * (1.0 - weakness))
    weights["mildly_stale"] += (0.06 * weakness)
    weights["low_confidence"] += (0.08 * weakness)
    weights["severely_stale"] += (0.12 * weakness)
    weights["proxy"] += (0.10 * weakness)
    if context_boost > 0.0:
        stress_scale = (0.45 + (0.55 * support_strength)) * (0.55 + (0.45 * stress_selectivity))
        nominal_floor = 0.08 if supported_case else 0.14
        refreshed_floor = 0.20 if supported_case else 0.30
        nominal_penalty = 0.88 if supported_case else 0.70
        refreshed_penalty = 0.30 if supported_case else 0.18
        weights["nominal"] *= max(nominal_floor, 1.0 - (nominal_penalty * context_boost * stress_scale))
        weights["refreshed"] *= max(refreshed_floor, 1.0 - (refreshed_penalty * context_boost * stress_scale))
        weights["mildly_stale"] += (0.07 if supported_case else 0.06) * context_boost * stress_scale
        weights["low_confidence"] += (0.12 if supported_case else 0.10) * context_boost * stress_scale
        weights["severely_stale"] += (0.28 if supported_case else 0.22) * context_boost * stress_scale
        weights["proxy"] += (0.04 if supported_case else 0.028) * context_boost * stress_scale
    allowed = {str(state): max(0.0, float(weights.get(str(state), 0.0))) for state in state_catalog}
    total = sum(allowed.values())
    if total <= 0.0:
        return {str(state): 1.0 / float(len(state_catalog)) for state in state_catalog}
    return {state: value / total for state, value in allowed.items()}


def _normalised_ambiguity_context(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    path_count = max(0, int(_as_float(payload.get("od_candidate_path_count"), 0.0)))
    family_count = max(0, int(_as_float(payload.get("od_corridor_family_count"), 0.0)))
    source_count = max(0, int(_as_float(payload.get("od_ambiguity_source_count"), 0.0)))
    explicit_source_mix_count = max(0, int(_as_float(payload.get("od_ambiguity_source_mix_count"), 0.0)))
    raw_source_mix = payload.get("od_ambiguity_source_mix")
    source_mix_items: list[str] = []
    if isinstance(raw_source_mix, str):
        text = raw_source_mix.strip()
        if text[:1] in {"{", "["}:
            try:
                parsed_source_mix = json.loads(text)
            except json.JSONDecodeError:
                parsed_source_mix = None
            else:
                if isinstance(parsed_source_mix, Mapping):
                    source_mix_items = [str(key).strip() for key in parsed_source_mix if str(key).strip()]
                elif isinstance(parsed_source_mix, Sequence) and not isinstance(parsed_source_mix, (str, bytes)):
                    source_mix_items = [str(item).strip() for item in parsed_source_mix if str(item).strip()]
        if not source_mix_items:
            source_mix_items = [item.strip() for item in raw_source_mix.replace("+", ",").split(",") if item.strip()]
    elif isinstance(raw_source_mix, Sequence) and not isinstance(raw_source_mix, (str, bytes)):
        source_mix_items = [str(item).strip() for item in raw_source_mix if str(item).strip()]
    budget_band = str(payload.get("ambiguity_budget_band") or "unspecified").strip().lower() or "unspecified"
    ambiguity_confidence = _clamp01(
        payload.get("od_ambiguity_confidence"),
        1.0 if _clamp01(payload.get("od_ambiguity_index")) > 0.0 else 0.0,
    )
    support_ratio = _clamp01(payload.get("od_ambiguity_support_ratio"))
    source_entropy = _clamp01(payload.get("od_ambiguity_source_entropy"))
    source_count_strength = min(1.0, source_count / 4.0) if source_count > 0 else 0.0
    derived_source_mix_count = max(len(source_mix_items), explicit_source_mix_count)
    source_mix_strength = min(1.0, derived_source_mix_count / 3.0) if derived_source_mix_count > 0 else 0.0
    support_strength = _clamp01(
        (0.35 * ambiguity_confidence)
        + (0.18 * source_count_strength)
        + (0.12 * source_mix_strength)
        + (0.20 * support_ratio)
        + (0.15 * source_entropy),
        0.0,
    )
    raw_context_strength = max(
        _clamp01(payload.get("od_ambiguity_index")),
        _clamp01(payload.get("od_hard_case_prior")),
        _clamp01(payload.get("od_engine_disagreement_prior")),
        _clamp01(payload.get("od_toll_disagreement_rate")),
    )
    normalized = {
        "od_ambiguity_index": _clamp01(payload.get("od_ambiguity_index")),
        "od_ambiguity_confidence": ambiguity_confidence,
        "od_ambiguity_source_count": source_count,
        "od_ambiguity_source_mix_count": derived_source_mix_count,
        "od_ambiguity_support_ratio": support_ratio,
        "od_ambiguity_source_entropy": source_entropy,
        "od_ambiguity_prior_strength": _clamp01(payload.get("od_ambiguity_prior_strength")),
        "od_ambiguity_family_density": _clamp01(payload.get("od_ambiguity_family_density")),
        "od_ambiguity_margin_pressure": _clamp01(payload.get("od_ambiguity_margin_pressure")),
        "od_ambiguity_spread_pressure": _clamp01(payload.get("od_ambiguity_spread_pressure")),
        "od_ambiguity_toll_instability": _clamp01(payload.get("od_ambiguity_toll_instability")),
        "od_engine_disagreement_prior": _clamp01(payload.get("od_engine_disagreement_prior")),
        "od_hard_case_prior": _clamp01(payload.get("od_hard_case_prior")),
        "od_candidate_path_count": path_count,
        "od_corridor_family_count": family_count,
        "od_objective_spread": _clamp01(payload.get("od_objective_spread")),
        "od_nominal_margin_proxy": _clamp01(payload.get("od_nominal_margin_proxy")),
        "od_toll_disagreement_rate": _clamp01(payload.get("od_toll_disagreement_rate")),
        "ambiguity_budget_prior": _clamp01(payload.get("ambiguity_budget_prior")),
        "ambiguity_budget_band": budget_band,
        "support_strength": round(support_strength, 6),
        "stress_world_fraction": _clamp01(payload.get("stress_world_fraction")),
        "refc_stress_world_fraction": _clamp01(
            payload.get("refc_stress_world_fraction", payload.get("stress_world_fraction"))
        ),
        "hard_case_stress_world_fraction": _clamp01(payload.get("hard_case_stress_world_fraction")),
        "supported_ambiguity_stress_world_fraction": _clamp01(
            payload.get("supported_ambiguity_stress_world_fraction")
        ),
        "hard_case_stress_pack_count": max(0, int(_as_float(payload.get("hard_case_stress_pack_count"), 0.0))),
        "supported_ambiguity_stress_pack_count": max(
            0,
            int(_as_float(payload.get("supported_ambiguity_stress_pack_count"), 0.0)),
        ),
        "targeted_stress_pack_count": max(0, int(_as_float(payload.get("targeted_stress_pack_count"), 0.0))),
    }
    projected_context_strength = round(
        _clamp01(raw_context_strength * (0.65 + (0.35 * support_strength))),
        6,
    )
    supported_ambiguity_strength = round(
        _clamp01(
            (0.22 * normalized["od_ambiguity_index"])
            + (0.16 * normalized["ambiguity_budget_prior"])
            + (0.12 * normalized["od_engine_disagreement_prior"])
            + (0.12 * normalized["od_ambiguity_family_density"])
            + (0.10 * normalized["od_ambiguity_margin_pressure"])
            + (0.08 * normalized["od_objective_spread"])
            + (0.08 * support_ratio)
            + (0.06 * source_entropy)
            + (0.06 * support_strength)
        ),
        6,
    )
    normalized["is_hard_case"] = bool(
        normalized["od_hard_case_prior"] >= 0.6
        or normalized["od_ambiguity_index"] >= 0.65
        or budget_band == "high"
        or (
            support_ratio >= 0.65
            and source_entropy >= 0.55
            and projected_context_strength >= 0.45
        )
    )
    normalized["supported_ambiguity_strength"] = supported_ambiguity_strength
    support_rich_structural_support = bool(
        path_count >= 2
        or family_count >= 2
        or derived_source_mix_count >= 2
        or source_count >= 3
        or normalized["od_ambiguity_family_density"] >= 0.28
        or normalized["od_ambiguity_margin_pressure"] >= 0.25
    )
    normalized["is_supported_ambiguity_case"] = bool(
        not normalized["is_hard_case"]
        and support_strength >= 0.45
        and support_ratio >= 0.50
        and source_entropy >= 0.42
        and (
            (
                supported_ambiguity_strength >= 0.42
                and support_rich_structural_support
            )
            or (
                normalized["od_ambiguity_index"] >= 0.35
                and support_strength >= 0.70
                and support_ratio >= 0.70
                and source_entropy >= 0.65
                and (source_count >= 3 or derived_source_mix_count >= 2)
            )
        )
    )
    normalized["context_strength"] = projected_context_strength
    return normalized


def refc_requires_full_stress_worlds(ambiguity_context: Mapping[str, Any] | None) -> bool:
    context = _normalised_ambiguity_context(ambiguity_context)
    return bool(context["is_hard_case"] or context["is_supported_ambiguity_case"])


def _family_context_boost(family: str, context: Mapping[str, Any]) -> float:
    ambiguity = _clamp01(context.get("od_ambiguity_index"))
    hard_case = _clamp01(context.get("od_hard_case_prior"))
    disagreement = _clamp01(context.get("od_engine_disagreement_prior"))
    spread = _clamp01(context.get("od_objective_spread"))
    toll_disagreement = _clamp01(context.get("od_toll_disagreement_rate"))
    support_strength = _clamp01(context.get("support_strength"), 0.0)
    path_density = min(1.0, max(0.0, (_as_float(context.get("od_candidate_path_count")) - 1.0) / 5.0))
    family_density = min(1.0, max(0.0, (_as_float(context.get("od_corridor_family_count")) - 1.0) / 3.0))
    margin_pressure = 1.0 - _clamp01(context.get("od_nominal_margin_proxy"), 1.0)
    general = (
        (0.24 * ambiguity)
        + (0.22 * hard_case)
        + (0.14 * spread)
        + (0.14 * path_density)
        + (0.10 * family_density)
        + (0.16 * margin_pressure)
    )
    family_specific = {
        "scenario": (0.24 * disagreement) + (0.08 * ambiguity),
        "toll": 0.52 * toll_disagreement,
        "terrain": 0.20 * spread,
        "fuel": 0.10 * spread,
        "carbon": 0.08 * spread,
        "weather": 0.22 * hard_case,
        "stochastic": (0.20 * hard_case) + (0.12 * disagreement),
    }.get(family, 0.0)
    return _clamp01((general + family_specific) * (0.65 + (0.35 * support_strength)))


def _targeted_stress_pack(
    *,
    routes: Sequence[Mapping[str, Any]] = (),
    families: Sequence[str],
    states: Sequence[str],
    seed: int,
    ambiguity_context: Mapping[str, Any],
    selector_weights: tuple[float, float, float] | None = None,
) -> list[WorldSample]:
    context = _normalised_ambiguity_context(ambiguity_context)
    if not families:
        return []
    effective_selector_weights = selector_weights or (1.0, 1.0, 1.0)
    pairwise_gap: dict[str, float] = {}
    winner_route_id: str | None = None
    runner_up_route_id: str | None = None
    stress_fraction = 0.0
    closeness = 0.0
    multi_frontier_pressure = 0.0
    if routes and len(routes) >= 2:
        ordered_routes = sorted(
            routes,
            key=lambda route: (
                _selector_score(route, _objective_vector(route), weights=effective_selector_weights),
                _route_id(route),
            ),
        )
        winner_route = ordered_routes[0]
        runner_up_route = ordered_routes[1]
        winner_route_id = _route_id(winner_route)
        runner_up_route_id = _route_id(runner_up_route)
        winner_score = _selector_score(winner_route, _objective_vector(winner_route), weights=effective_selector_weights)
        runner_up_score = _selector_score(
            runner_up_route,
            _objective_vector(runner_up_route),
            weights=effective_selector_weights,
        )
        score_gap = max(0.0, runner_up_score - winner_score)
        score_scale = max(abs(winner_score), abs(runner_up_score), 1.0)
        closeness = max(0.0, min(1.0, 1.0 - (score_gap / score_scale)))
        winner_tensor = _route_dependency_weights(winner_route, active_families=families)
        runner_tensor = _route_dependency_weights(runner_up_route, active_families=families)
        for family in families:
            exposure_gap = 0.0
            sensitivity = _family_sensitivity(family)
            winner_scale = _family_route_provenance_scale(winner_route, family)
            runner_scale = _family_route_provenance_scale(runner_up_route, family)
            for idx, objective in enumerate(OBJECTIVE_NAMES):
                objective_weight = _as_float(effective_selector_weights[idx])
                exposure_gap += objective_weight * max(
                    0.0,
                    (winner_tensor[objective].get(family, 0.0) * sensitivity[idx] * winner_scale)
                    - (runner_tensor[objective].get(family, 0.0) * sensitivity[idx] * runner_scale),
                )
            pairwise_gap[family] = exposure_gap
        multi_frontier_pressure = _clamp01(
            (0.48 * closeness)
            + (0.18 * min(1.0, max(0.0, len(routes) - 1.0) / 3.0))
            + (0.16 * _clamp01(context.get("od_ambiguity_family_density"), 0.0))
            + (0.10 * _clamp01(context.get("od_ambiguity_margin_pressure"), 0.0))
            + (0.08 * _clamp01(context.get("od_objective_spread"), 0.0))
        )
    priority_route: Mapping[str, Any] | None = None
    if routes:
        priority_route = ordered_routes[0] if len(routes) >= 2 else routes[0]
        winner_route_id = winner_route_id or _route_id(priority_route)
    route_exposure: dict[str, float] = {}
    if priority_route is not None:
        priority_tensor = _route_dependency_weights(priority_route, active_families=families)
        for family in families:
            exposure = 0.0
            sensitivity = _family_sensitivity(family)
            provenance_scale = _family_route_provenance_scale(priority_route, family)
            for idx, objective in enumerate(OBJECTIVE_NAMES):
                objective_weight = _as_float(effective_selector_weights[idx])
                exposure += (
                    objective_weight
                    * priority_tensor[objective].get(family, 0.0)
                    * sensitivity[idx]
                    * provenance_scale
                )
            route_exposure[family] = exposure
    support_strength = _clamp01(context.get("support_strength"), 0.0)
    support_ratio = _clamp01(context.get("od_ambiguity_support_ratio"), 0.0)
    source_entropy = _clamp01(context.get("od_ambiguity_source_entropy"), 0.0)
    support_dense = bool(
        support_strength >= 0.46
        and support_ratio >= 0.54
        and source_entropy >= 0.46
    )
    supported_ambiguity = bool(
        context.get("is_supported_ambiguity_case")
        and support_dense
        and (
            closeness >= 0.78
            or multi_frontier_pressure >= 0.30
            or max(pairwise_gap.values(), default=0.0) >= 0.03
            or len(routes) >= 3
        )
    )
    if not context["is_hard_case"] and not supported_ambiguity:
        return []
    case_kind = "hard_case" if context["is_hard_case"] else "supported_ambiguity"
    case_scale = 1.0 if case_kind == "hard_case" else 0.72
    if case_kind == "hard_case":
        case_scale = 1.0
        max_fraction = 0.60
    else:
        case_scale = 0.80
        max_fraction = 0.38
    stress_fraction = max(
        0.0,
        min(
            max_fraction,
            (0.10 if case_kind == "hard_case" else 0.03)
            + ((0.16 if case_kind == "hard_case" else 0.12) * context["context_strength"])
            + ((0.10 if case_kind == "hard_case" else 0.10) * support_strength)
            + ((0.10 if case_kind == "hard_case" else 0.09) * support_ratio)
            + ((0.08 if case_kind == "hard_case" else 0.07) * source_entropy)
            + ((0.16 if case_kind == "hard_case" else 0.12) * closeness)
            + ((0.12 if case_kind == "hard_case" else 0.10) * multi_frontier_pressure)
            + ((0.12 if case_kind == "hard_case" else 0.08) * (1.0 - _clamp01(context.get("od_nominal_margin_proxy"), 1.0))),
        ),
    )
    ranked = sorted(
        families,
        key=lambda family: (
            -(0.70 * _family_context_boost(family, context) + 1.35 * max(0.0, pairwise_gap.get(family, 0.0))),
            family,
        ),
    )
    target_family = max(
        families,
        key=lambda family: (
            route_exposure.get(family, 0.0),
            pairwise_gap.get(family, 0.0),
            _family_context_boost(family, context),
            family,
        ),
    )
    seeded = [target_family]
    for family in ranked:
        if family == target_family:
            continue
        if _family_context_boost(family, context) > 0.12:
            seeded.append(family)
    if len(seeded) == 1 and len(ranked) > 1:
        seeded.append(next(family for family in ranked if family != target_family))
    if case_kind == "supported_ambiguity" and len(seeded) > 1 and closeness >= 0.78:
        stress_fraction = min(max_fraction, stress_fraction + 0.04)
    severe_state = "severely_stale" if "severely_stale" in states else states[-1]
    low_conf_state = "low_confidence" if "low_confidence" in states else severe_state
    mild_state = "mildly_stale" if "mildly_stale" in states else severe_state
    pack_states: list[dict[str, str]] = []
    first_family = seeded[0]
    first_state = {family: "nominal" for family in families}
    first_state[first_family] = severe_state
    pack_states.append(first_state)
    if len(seeded) > 1:
        second_isolated_state = {family: "nominal" for family in families}
        second_isolated_state[seeded[1]] = severe_state
        pack_states.append(second_isolated_state)
        second_state = {family: "nominal" for family in families}
        second_state[seeded[0]] = mild_state
        second_state[seeded[1]] = severe_state
        pack_states.append(second_state)
    if len(seeded) > 2 and case_kind == "supported_ambiguity" and multi_frontier_pressure >= 0.24:
        third_isolated_state = {family: "nominal" for family in families}
        third_isolated_state[seeded[2]] = severe_state
        pack_states.append(third_isolated_state)
    winner_targeted_families = {seeded[0]}
    if len(seeded) > 1:
        winner_targeted_families.update(seeded[: min(len(seeded), 2)])
    if case_kind == "supported_ambiguity" and len(seeded) > 2 and multi_frontier_pressure >= 0.24:
        winner_targeted_families.add(seeded[2])
    if context["od_engine_disagreement_prior"] >= 0.4:
        disagreement_families = [family for family in ("scenario", "weather", "stochastic") if family in families]
        if disagreement_families:
            disagreement_state = {family: "nominal" for family in families}
            for family in disagreement_families[:2]:
                disagreement_state[family] = severe_state if family == disagreement_families[0] else low_conf_state
            pack_states.append(disagreement_state)
    if context["od_toll_disagreement_rate"] >= 0.4 and "toll" in families:
        toll_state = {family: "nominal" for family in families}
        toll_state["toll"] = severe_state
        pack_states.append(toll_state)
    if case_kind == "supported_ambiguity" and closeness >= 0.82 and len(seeded) > 1:
        combo_state = {family: "nominal" for family in families}
        combo_state[seeded[0]] = severe_state
        combo_state[seeded[1]] = severe_state
        if len(seeded) > 2:
            combo_state[seeded[2]] = low_conf_state
        pack_states.append(combo_state)
    if case_kind == "supported_ambiguity" and len(seeded) > 1 and multi_frontier_pressure >= 0.30:
        contrast_state = {family: "nominal" for family in families}
        contrast_state[seeded[0]] = severe_state
        contrast_state[seeded[-1]] = low_conf_state
        pack_states.append(contrast_state)
    if len(seeded) > 2:
        third_state = {family: "nominal" for family in families}
        third_state[seeded[0]] = severe_state
        third_state[seeded[1]] = severe_state
        third_state[seeded[2]] = low_conf_state
        pack_states.append(third_state)
    if support_strength >= 0.75 and multi_frontier_pressure >= 0.45 and len(seeded) > 1:
        reversal_state = {family: "nominal" for family in families}
        reversal_state[seeded[0]] = mild_state
        reversal_state[seeded[1]] = severe_state
        if "stochastic" in families and reversal_state.get("stochastic") == "nominal":
            reversal_state["stochastic"] = low_conf_state
        pack_states.append(reversal_state)
    repeat_budget = max(
        1,
        int(
            round(
                max(5.0 if case_kind == "hard_case" else 3.0, stress_fraction * (72.0 if case_kind == "hard_case" else 56.0))
            )
        ),
    )
    stress_factor = max(
        1.50 if case_kind == "hard_case" else 1.25,
        min(
            3.2 if case_kind == "hard_case" else 2.55,
            (1.62 if case_kind == "hard_case" else 1.28)
            + ((0.92 if case_kind == "hard_case" else 0.60) * context["context_strength"])
            + ((0.52 if case_kind == "hard_case" else 0.36) * support_strength)
            + ((0.72 if case_kind == "hard_case" else 0.44) * max(pairwise_gap.values(), default=0.0))
            + ((0.0 if case_kind == "hard_case" else 0.22) * multi_frontier_pressure),
        ),
    )
    pack: list[WorldSample] = []
    active_states = pack_states[: max(1, min(6 if case_kind == "hard_case" else 5, len(pack_states)))]
    allocated = 0
    for idx, state_map in enumerate(active_states):
        family_target_route_ids: dict[str, str] = {}
        for family, state_name in state_map.items():
            if state_name == "nominal":
                continue
            preferred_target = ""
            if winner_route_id and (
                max(0.0, pairwise_gap.get(family, 0.0)) > 0.0
                or route_exposure.get(family, 0.0) > 0.0
                or family in winner_targeted_families
            ):
                preferred_target = winner_route_id
            elif runner_up_route_id and max(0.0, pairwise_gap.get(family, 0.0)) <= 0.0:
                preferred_target = runner_up_route_id
            elif priority_route is not None:
                preferred_target = _route_id(priority_route)
            if preferred_target:
                family_target_route_ids[family] = preferred_target
        scoped_target_route_id = None
        scoped_targets = {route_id for route_id in family_target_route_ids.values() if route_id}
        if len(scoped_targets) == 1:
            scoped_target_route_id = next(iter(scoped_targets))
        state_gap = max(0.05, pairwise_gap.get(seeded[min(idx, len(seeded) - 1)], 0.0))
        state_repeat = max(
            1,
            int(
                round(
                    max(1.0, repeat_budget * min(1.0, 0.40 + state_gap) * case_scale)
                )
            ),
        )
        stressed_family_count = sum(1 for value in state_map.values() if value != "nominal")
        world_kind = f"{case_kind}_mixed_targeted" if stressed_family_count > 1 else f"{case_kind}_targeted"
        for repetition in range(state_repeat):
            world_id = _stable_hash(
                [
                    f"{case_kind}_pack",
                    str(seed),
                    str(idx),
                    str(repetition),
                    str(round(stress_factor, 6)),
                    json.dumps(state_map, sort_keys=True, separators=(",", ":")),
                ]
            )
            pack.append(
                WorldSample(
                    world_id=world_id,
                    states=state_map,
                    stress_factor=stress_factor,
                    world_kind=world_kind,
                    target_route_id=scoped_target_route_id,
                    target_route_ids=family_target_route_ids,
                )
            )
            allocated += 1
    if not pack and active_states:
        stressed_family_count = sum(1 for value in active_states[0].values() if value != "nominal")
        fallback_target_route_ids = {}
        if winner_route_id:
            fallback_target_route_ids = {
                family: winner_route_id
                for family, state_name in active_states[0].items()
                if state_name != "nominal"
            }
        world_id = _stable_hash(
            [
                f"{case_kind}_pack",
                str(seed),
                "fallback",
                str(round(stress_factor, 6)),
                json.dumps(active_states[0], sort_keys=True, separators=(",", ":")),
            ]
        )
        pack.append(
            WorldSample(
                world_id=world_id,
                states=active_states[0],
                stress_factor=stress_factor,
                world_kind=f"{case_kind}_mixed_targeted" if stressed_family_count > 1 else f"{case_kind}_targeted",
                target_route_id=winner_route_id,
                target_route_ids=fallback_target_route_ids,
            )
        )
    return pack


def _family_route_provenance_scale(route: Mapping[str, Any], family: str) -> float:
    rows = _family_provenance_rows([route], family)
    if not rows:
        return 1.0
    row = rows[0]
    confidence = _as_float(row.get("confidence"), 0.85)
    coverage = _as_float(row.get("coverage_ratio"), 1.0)
    freshness_present = 1.0 if str(row.get("freshness_timestamp_utc", "")).strip() else 0.0
    weakness = (
        (0.45 * (1.0 - confidence))
        + (0.25 * (1.0 - coverage))
        + (0.20 * (1.0 - freshness_present))
        + (0.10 * (1.0 if bool(row.get("fallback_used", False)) else 0.0))
    )
    # Even "good" evidence should still produce measurable fragility when two
    # routes are genuinely close; otherwise REFC collapses into a tautological
    # certificate on strict frontiers with small but meaningful trade-offs.
    return max(0.90, min(1.75, 0.97 + (1.05 * weakness)))


def _normalise_weights(weights: Mapping[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, _as_float(value)) for key, value in weights.items()}
    total = sum(positive.values())
    if total <= 0.0:
        return {key: 0.0 for key in positive}
    return {key: value / total for key, value in positive.items()}


def _amplify_dependency_contrast(weights: Mapping[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, _as_float(value)) for key, value in weights.items()}
    positive_values = [value for value in positive.values() if value > 0.0]
    if len(positive_values) <= 1:
        return positive
    mean_value = sum(positive_values) / float(len(positive_values))
    if mean_value <= 0.0:
        return positive
    spread = max(positive_values) - min(positive_values)
    if spread <= 0.0:
        return positive
    flatness = 1.0 - min(1.0, spread / max(mean_value, 1e-9))
    amplification = 1.0 + min(1.75, 0.35 + (1.4 * flatness))
    amplified: dict[str, float] = {}
    for family, value in positive.items():
        if value <= 0.0:
            amplified[family] = 0.0
            continue
        amplified[family] = max(0.0, mean_value + ((value - mean_value) * amplification))
    return amplified


def _route_dependency_weights(
    route: Mapping[str, Any],
    *,
    active_families: Sequence[str],
) -> dict[str, dict[str, float]]:
    tensor = _evidence_tensor(route)
    provenance = _route_provenance(route)
    raw_route_specific = provenance.get("dependency_weights", {})
    route_active_families = _route_active_families(route)
    operational_overrides = _route_operational_dependency_overrides(
        route,
        active_families=active_families,
    )
    out: dict[str, dict[str, float]] = {}
    for objective in OBJECTIVE_NAMES:
        objective_weights = {family: 0.0 for family in active_families}
        for family in active_families:
            if family in route_active_families:
                objective_weights[family] = DEFAULT_OBJECTIVE_BIAS[objective].get(family, 0.0)
        if isinstance(raw_route_specific, Mapping):
            for family, family_weights in raw_route_specific.items():
                if family not in active_families or not isinstance(family_weights, Mapping):
                    continue
                if objective in family_weights:
                    value = _as_float(family_weights.get(objective))
                    objective_weights[family] = value
        for family in active_families:
            family_weights = tensor.get(family, {})
            if objective in family_weights:
                objective_weights[family] = family_weights[objective]
        for family, override in operational_overrides.get(objective, {}).items():
            if family not in objective_weights:
                continue
            objective_weights[family] += max(0.0, _as_float(override))
        out[objective] = _normalise_weights(_amplify_dependency_contrast(objective_weights))
    return out


def _state_effect(state: str) -> tuple[float, float, float]:
    return DEFAULT_STATE_EFFECTS.get(state, DEFAULT_STATE_EFFECTS["nominal"])


def _family_sensitivity(family: str) -> tuple[float, float, float]:
    return DEFAULT_FAMILY_SENSITIVITY.get(family, (0.35, 0.35, 0.35))


def _selector_score(
    route: Mapping[str, Any],
    vector: Sequence[float],
    *,
    weights: tuple[float, float, float],
) -> float:
    if len(vector) < 3:
        raise ValueError("objective vector must have at least 3 entries")
    return (
        (_as_float(weights[0]) * _as_float(vector[0]))
        + (_as_float(weights[1]) * _as_float(vector[1]))
        + (_as_float(weights[2]) * _as_float(vector[2]))
        + _as_float(route.get("selector_bias", 0.0))
    )


def _world_signature(world: Mapping[str, Any]) -> str:
    return _stable_hash(
        [
            str(world.get("world_id", "")),
            str(round(_as_float(world.get("stress_factor"), 1.0), 6)),
            json.dumps(world.get("states", {}), sort_keys=True, separators=(",", ":"), default=str),
        ]
    )


def _world_score_map(
    routes: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    *,
    active_families: Sequence[str],
    selector_weights: tuple[float, float, float],
    selector_score_map_fn: SelectorScoreMapFn | None,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, float], str]:
    perturbed_by_id = {
        _route_id(route): _route_perturbed_objectives(route, world, active_families=active_families)
        for route in routes
    }
    if selector_score_map_fn is not None:
        raw_score_map = selector_score_map_fn(routes, perturbed_by_id)
        score_map = {
            route_id: _as_float(raw_score_map.get(route_id), float("inf"))
            for route_id in perturbed_by_id
        }
    else:
        score_map = {
            _route_id(route): _selector_score(
                route,
                perturbed_by_id[_route_id(route)],
                weights=selector_weights,
            )
            for route in routes
        }
    winner_id = min(score_map.items(), key=lambda item: (item[1], item[0]))[0]
    return perturbed_by_id, score_map, winner_id


@dataclass(frozen=True)
class EvidenceProvenance:
    family: str
    source: str
    freshness_timestamp_utc: str | None = None
    max_age_minutes: float | None = None
    signature: str | None = None
    confidence: float | None = None
    fallback_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorldSample:
    world_id: str
    states: dict[str, str]
    stress_factor: float = 1.0
    world_kind: str = "sampled"
    target_route_id: str | None = None
    target_route_ids: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "world_id": self.world_id,
            "states": dict(self.states),
            "stress_factor": float(self.stress_factor),
            "world_kind": str(self.world_kind),
        }
        if self.target_route_id:
            payload["target_route_id"] = str(self.target_route_id)
        if self.target_route_ids:
            payload["target_route_ids"] = {
                str(family): str(route_id)
                for family, route_id in self.target_route_ids.items()
                if str(family).strip() and str(route_id).strip()
            }
        return payload


@dataclass(frozen=True)
class CertificateConfig:
    seed: int = 0
    world_count: int = 64
    threshold: float = 0.67
    selector_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    state_catalog: tuple[str, ...] = EVIDENCE_STATES


@dataclass(frozen=True)
class CertificateResult:
    winner_id: str
    certificate: dict[str, float]
    threshold: float
    certified: bool
    selected_route_id: str
    route_scores: dict[str, list[float]]
    world_manifest: dict[str, Any]
    selector_config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner_id": self.winner_id,
            "certificate": dict(self.certificate),
            "threshold": self.threshold,
            "certified": self.certified,
            "selected_route_id": self.selected_route_id,
            "route_scores": {key: list(values) for key, values in self.route_scores.items()},
            "world_manifest": dict(self.world_manifest),
            "selector_config": dict(self.selector_config),
        }


@dataclass(frozen=True)
class FragilityResult:
    route_fragility_map: dict[str, dict[str, float]]
    competitor_fragility_breakdown: dict[str, dict[str, dict[str, int]]]
    value_of_refresh: dict[str, Any]
    route_fragility_details: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    evidence_snapshot_manifest: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_fragility_map": self.route_fragility_map,
            "competitor_fragility_breakdown": self.competitor_fragility_breakdown,
            "value_of_refresh": self.value_of_refresh,
            "route_fragility_details": self.route_fragility_details,
            "evidence_snapshot_manifest": self.evidence_snapshot_manifest,
        }


@dataclass(frozen=True)
class _EvaluatedWorldBundle:
    world_rows: list[dict[str, Any]]
    score_maps: list[dict[str, float]]
    winners: list[str]
    active_families: tuple[str, ...]
    unique_world_count: int = 0
    state_score_cache: dict[str, tuple[dict[str, float], str]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceValidationResult:
    status: str
    issues: list[dict[str, Any]]
    active_families: list[str]
    family_modes: dict[str, str]
    freshness_coverage: float
    live_family_count: int
    snapshot_family_count: int
    model_family_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issues": [dict(issue) for issue in self.issues],
            "active_families": list(self.active_families),
            "family_modes": dict(self.family_modes),
            "freshness_coverage": self.freshness_coverage,
            "live_family_count": self.live_family_count,
            "snapshot_family_count": self.snapshot_family_count,
            "model_family_count": self.model_family_count,
        }


def active_evidence_families(
    routes: Sequence[Mapping[str, Any]],
    *,
    configured_families: Sequence[str] | None = None,
) -> list[str]:
    configured = tuple(configured_families or EVIDENCE_FAMILIES)
    active: set[str] = set()
    for route in routes:
        tensor = _evidence_tensor(route)
        for family, objective_map in tensor.items():
            if family not in configured:
                continue
            if any(_as_float(weight) > 0.0 for weight in objective_map.values()):
                active.add(family)
        provenance = _route_provenance(route)
        active_families = provenance.get("active_families", [])
        if isinstance(active_families, Sequence) and not isinstance(active_families, (str, bytes)):
            families_iterable = active_families
        else:
            families_iterable = []
        for family in families_iterable:
            family_name = str(family)
            if family_name in configured:
                active.add(family_name)
    return sorted(active)


def _route_evidence_payload(route_or_provenance: Mapping[str, Any] | EvidenceProvenance | None) -> dict[str, Any]:
    if route_or_provenance is None:
        return {}
    if isinstance(route_or_provenance, EvidenceProvenance):
        return route_or_provenance.as_dict()
    payload = dict(route_or_provenance)
    if isinstance(payload.get("evidence_provenance"), Mapping):
        return dict(payload["evidence_provenance"])
    return payload


def _contains_strict_deny_token(*values: Any) -> bool:
    tokens = ("synthetic", "proxy", "fallback", "legacy", "bootstrap", "fixture")
    combined = " ".join(str(value) for value in values if value not in (None, "")).lower()
    return any(token in combined for token in tokens)


def validate_route_evidence_provenance(
    route_or_provenance: Mapping[str, Any] | EvidenceProvenance | None,
    *,
    allow_snapshot: bool = True,
    require_freshness: bool = True,
) -> EvidenceValidationResult:
    payload = _route_evidence_payload(route_or_provenance)
    families_raw = payload.get("families", [])
    families = families_raw if isinstance(families_raw, Sequence) and not isinstance(families_raw, (str, bytes)) else []
    active_families_raw = payload.get("active_families", [])
    active_families = (
        [str(family) for family in active_families_raw]
        if isinstance(active_families_raw, Sequence) and not isinstance(active_families_raw, (str, bytes))
        else []
    )
    issues: list[dict[str, Any]] = []
    family_modes: dict[str, str] = {}
    live_family_count = 0
    snapshot_family_count = 0
    model_family_count = 0
    freshness_count = 0
    for entry in families:
        if not isinstance(entry, Mapping):
            continue
        family = str(entry.get("family") or "").strip()
        if not family:
            continue
        active = bool(entry.get("active", True))
        source = str(entry.get("source") or "").strip()
        fallback_used = bool(entry.get("fallback_used", False))
        fallback_source = str(entry.get("fallback_source") or "").strip()
        freshness_timestamp = str(entry.get("freshness_timestamp_utc") or "").strip()
        signature = str(entry.get("signature") or "").strip()
        details = entry.get("details")
        combined_source = source if not isinstance(details, Mapping) else f"{source} {json.dumps(details, sort_keys=True, default=str)}"
        if fallback_used or _contains_strict_deny_token(source, fallback_source, combined_source):
            issues.append(
                {
                    "family": family,
                    "source": source,
                    "reason_code": "evidence_provenance_rejected",
                    "message": "Evidence provenance contains synthetic/proxy/fallback/legacy markers.",
                }
            )
            family_modes[family] = "rejected"
            continue
        if "snapshot" in source.lower() or "snapshot" in signature.lower():
            if not allow_snapshot:
                issues.append(
                    {
                        "family": family,
                        "source": source,
                        "reason_code": "evidence_snapshot_not_allowed",
                        "message": "Snapshot evidence is not allowed in this evaluation context.",
                    }
                )
                family_modes[family] = "rejected"
                continue
            family_modes[family] = "snapshot"
            snapshot_family_count += 1
            if active:
                freshness_count += 1
            continue
        if family == "stochastic":
            family_modes[family] = "model"
            model_family_count += 1
            continue
        if require_freshness and active and not freshness_timestamp:
            issues.append(
                {
                    "family": family,
                    "source": source,
                    "reason_code": "evidence_freshness_missing",
                    "message": "Active evidence family is missing a freshness timestamp.",
                }
            )
            family_modes[family] = "rejected"
            continue
        if active:
            freshness_count += 1
        family_modes[family] = "live"
        live_family_count += 1
    freshness_coverage = round(
        freshness_count / max(1, len([family for family in families if isinstance(family, Mapping) and bool(family.get("active", True))])),
        6,
    )
    status = "ok" if not issues else "rejected"
    return EvidenceValidationResult(
        status=status,
        issues=issues,
        active_families=list(active_families),
        family_modes=family_modes,
        freshness_coverage=freshness_coverage,
        live_family_count=live_family_count,
        snapshot_family_count=snapshot_family_count,
        model_family_count=model_family_count,
    )


def sample_world_manifest(
    *,
    active_families: Sequence[str],
    seed: int,
    world_count: int = 64,
    state_catalog: Sequence[str] = EVIDENCE_STATES,
    routes: Sequence[Mapping[str, Any]] = (),
    ambiguity_context: Mapping[str, Any] | None = None,
    selector_weights: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    # Deterministic world enumeration for exact replay. The manifest is derived
    # from (seed, world index, family) hashes rather than Monte Carlo draws so
    # evaluation runs remain byte-for-byte reproducible. This preserves the
    # reproducibility discipline of stress-testing workflows while still
    # covering a broad bounded set of evidence states.
    families = sorted({family for family in active_families if family in EVIDENCE_FAMILIES})
    states = tuple(state_catalog) if state_catalog else EVIDENCE_STATES
    if not states:
        raise ValueError("state catalog cannot be empty")
    requested_world_count = int(world_count)
    if requested_world_count <= 0:
        raise ValueError("world_count must be positive")
    context = _normalised_ambiguity_context(ambiguity_context)
    worlds: list[WorldSample] = []
    state_weights = {
        family: _family_state_weights(
            routes,
            family,
            state_catalog=states,
            ambiguity_context=context,
        )
        for family in families
    }
    for idx in range(requested_world_count):
        state_map: dict[str, str] = {}
        for family in families:
            digest = hashlib.sha1(f"{seed}|{idx}|{family}|{','.join(states)}".encode("utf-8")).digest()
            threshold = (int.from_bytes(digest[:8], "big") % 1_000_000) / 1_000_000.0
            cumulative = 0.0
            chosen_state = states[-1]
            for state in states:
                cumulative += float(state_weights[family].get(state, 0.0))
                if threshold <= cumulative:
                    chosen_state = state
                    break
            state_map[family] = chosen_state
        world_id = _stable_hash([str(seed), str(idx), json.dumps(state_map, sort_keys=True)])
        worlds.append(WorldSample(world_id=world_id, states=state_map))
    stress_pack = _targeted_stress_pack(
        routes=routes,
        families=families,
        states=states,
        seed=seed,
        ambiguity_context=context,
        selector_weights=selector_weights,
    )
    if stress_pack:
        worlds.extend(stress_pack)
    unique_signatures: set[str] = set()
    for world in worlds:
        unique_signatures.add(_world_signature(world.as_dict()))
    hard_case_stress_pack_count = sum(1 for world in stress_pack if str(world.world_kind).startswith("hard_case_"))
    supported_ambiguity_stress_pack_count = sum(
        1 for world in stress_pack if str(world.world_kind).startswith("supported_ambiguity_")
    )
    mixed_stress_pack_count = sum(1 for world in stress_pack if str(world.world_kind).endswith("_mixed_targeted"))
    single_family_stress_pack_count = len(stress_pack) - mixed_stress_pack_count
    stress_world_fraction = round(len(stress_pack) / float(max(1, len(worlds))), 6)
    hard_case_stress_world_fraction = round(
        hard_case_stress_pack_count / float(max(1, len(worlds))),
        6,
    )
    supported_ambiguity_stress_world_fraction = round(
        supported_ambiguity_stress_pack_count / float(max(1, len(worlds))),
        6,
    )
    payload = {
        "seed": int(seed),
        "requested_world_count": requested_world_count,
        # `world_count` is the effective weighted evaluation set. Hard-case
        # stress-pack duplicates intentionally remain here so certificate
        # frequencies reflect the targeted stress weighting, while
        # `unique_world_count` still reports the distinct world support.
        "world_count": len(worlds),
        "unique_world_count": len(unique_signatures),
        "active_families": families,
        "state_catalog": list(states),
        "state_weights": state_weights,
        "ambiguity_context": {
            **context,
            "targeted_stress_pack_count": len(stress_pack),
            "hard_case_stress_pack_count": hard_case_stress_pack_count,
            "supported_ambiguity_stress_pack_count": supported_ambiguity_stress_pack_count,
            "stress_world_fraction": stress_world_fraction,
            "refc_stress_world_fraction": stress_world_fraction,
            "hard_case_stress_world_fraction": hard_case_stress_world_fraction,
            "supported_ambiguity_stress_world_fraction": supported_ambiguity_stress_world_fraction,
        },
        "hard_case_stress_pack_count": hard_case_stress_pack_count,
        "supported_ambiguity_stress_pack_count": supported_ambiguity_stress_pack_count,
        "targeted_stress_pack_count": len(stress_pack),
        "mixed_targeted_stress_pack_count": mixed_stress_pack_count,
        "single_family_targeted_stress_pack_count": single_family_stress_pack_count,
        "world_reuse_rate": round((len(worlds) - len(unique_signatures)) / float(max(1, len(worlds))), 6),
        "stress_world_fraction": stress_world_fraction,
        "refc_stress_world_fraction": stress_world_fraction,
        "hard_case_stress_world_fraction": hard_case_stress_world_fraction,
        "supported_ambiguity_stress_world_fraction": supported_ambiguity_stress_world_fraction,
        "worlds": [world.as_dict() for world in worlds],
    }
    payload["manifest_hash"] = _stable_hash([json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)])
    return payload


def dependency_tensor(
    route: Mapping[str, Any],
    *,
    active_families: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    # Deterministic multilinear-style contribution decomposition so REFC can
    # explain how each evidence family perturbs each objective component.
    families = tuple(active_families or active_evidence_families([route]))
    tensor = _route_dependency_weights(route, active_families=families)
    return {objective: dict(weights) for objective, weights in tensor.items()}


def evidence_provenance_payload(
    *,
    family: str,
    source: str,
    freshness_timestamp_utc: str | None = None,
    max_age_minutes: float | None = None,
    signature: str | None = None,
    confidence: float | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    return EvidenceProvenance(
        family=family,
        source=source,
        freshness_timestamp_utc=freshness_timestamp_utc,
        max_age_minutes=max_age_minutes,
        signature=signature,
        confidence=confidence,
        fallback_used=fallback_used,
    ).as_dict()


def _route_perturbed_objectives(
    route: Mapping[str, Any],
    world: Mapping[str, Any],
    *,
    active_families: Sequence[str],
) -> tuple[float, float, float]:
    base = _objective_vector(route)
    tensor = _route_dependency_weights(route, active_families=active_families)
    route_id = _route_id(route)
    states = world.get("states", {})
    if not isinstance(states, Mapping):
        states = {}
    target_route_id = str(world.get("target_route_id", world.get("route_scope", "")) or "").strip()
    raw_target_route_ids = world.get("target_route_ids", world.get("route_scope_by_family", {}))
    target_route_ids: dict[str, str] = {}
    if isinstance(raw_target_route_ids, Mapping):
        target_route_ids = {
            str(family): str(target_id).strip()
            for family, target_id in raw_target_route_ids.items()
            if str(family).strip() and str(target_id).strip()
        }
    stress_factor = max(0.75, min(3.0, _as_float(world.get("stress_factor"), 1.0)))
    perturbed = [float(base[idx]) for idx in range(3)]
    for family in active_families:
        family_target_route_id = target_route_ids.get(str(family), target_route_id)
        if family_target_route_id and family_target_route_id != route_id:
            continue
        state = str(states.get(family, "nominal"))
        state_effect = _state_effect(state)
        sensitivity = _family_sensitivity(family)
        provenance_scale = _family_route_provenance_scale(route, family)
        for idx, objective in enumerate(OBJECTIVE_NAMES):
            weight = tensor[objective].get(family, 0.0)
            delta_ratio = state_effect[idx] * sensitivity[idx] * weight * provenance_scale * stress_factor
            perturbed[idx] = max(0.0, perturbed[idx] * (1.0 + delta_ratio))
    return float(perturbed[0]), float(perturbed[1]), float(perturbed[2])


def _evaluate_world_bundle(
    routes: Sequence[Mapping[str, Any]],
    worlds: Sequence[Mapping[str, Any]],
    *,
    active_families: Sequence[str],
    selector_weights: tuple[float, float, float],
    selector_score_map_fn: SelectorScoreMapFn | None,
    state_cache: dict[str, tuple[dict[str, float], str]] | None = None,
) -> _EvaluatedWorldBundle:
    world_rows = [dict(world) for world in worlds]
    score_maps: list[dict[str, float]] = []
    winners: list[str] = []
    evaluated_by_state = state_cache if state_cache is not None else {}
    for world in world_rows:
        raw_target_route_ids = world.get("target_route_ids", world.get("route_scope_by_family", {}))
        target_route_ids: dict[str, str] = {}
        if isinstance(raw_target_route_ids, Mapping):
            target_route_ids = {
                str(family): str(target_id).strip()
                for family, target_id in raw_target_route_ids.items()
                if str(family).strip() and str(target_id).strip()
            }
        state_signature = json.dumps(
            {
                "states": world.get("states", {}),
                "stress_factor": round(_as_float(world.get("stress_factor"), 1.0), 6),
                "target_route_id": str(world.get("target_route_id", world.get("route_scope", "")) or ""),
                "target_route_ids": target_route_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        cached = evaluated_by_state.get(state_signature)
        if cached is None:
            _, score_map, winner = _world_score_map(
                routes,
                world,
                active_families=active_families,
                selector_weights=selector_weights,
                selector_score_map_fn=selector_score_map_fn,
            )
            cached = ({str(route_id): float(score) for route_id, score in score_map.items()}, str(winner))
            evaluated_by_state[state_signature] = cached
        score_map, winner = cached
        score_maps.append(dict(score_map))
        winners.append(str(winner))
    return _EvaluatedWorldBundle(
        world_rows=world_rows,
        score_maps=score_maps,
        winners=winners,
        active_families=tuple(str(family) for family in active_families),
        unique_world_count=len(evaluated_by_state),
        state_score_cache=dict(evaluated_by_state),
    )


def evaluate_world_bundle(
    routes: Sequence[Mapping[str, Any]],
    worlds: Sequence[Mapping[str, Any]],
    *,
    active_families: Sequence[str],
    selector_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    selector_score_map_fn: SelectorScoreMapFn | None = None,
    state_cache: dict[str, tuple[dict[str, float], str]] | None = None,
) -> _EvaluatedWorldBundle:
    return _evaluate_world_bundle(
        routes,
        worlds,
        active_families=active_families,
        selector_weights=selector_weights,
        selector_score_map_fn=selector_score_map_fn,
        state_cache=state_cache,
    )


def _certificate_from_evaluated_bundle(
    routes: Sequence[Mapping[str, Any]],
    evaluated: _EvaluatedWorldBundle,
    *,
    selector_weights: tuple[float, float, float],
    threshold: float,
) -> CertificateResult:
    route_scores: dict[str, list[float]] = {_route_id(route): [] for route in routes}
    certificate_counts: dict[str, int] = {route_id: 0 for route_id in route_scores}
    for score_map, winner in zip(evaluated.score_maps, evaluated.winners, strict=True):
        for route_id, score in score_map.items():
            route_scores[route_id].append(float(score))
        certificate_counts[winner] += 1
    world_count = len(evaluated.world_rows)
    certificate = {route_id: certificate_counts[route_id] / float(world_count) for route_id in certificate_counts}
    winner_id = min(certificate.items(), key=lambda item: (-item[1], item[0]))[0]
    selected_route_id = min(
        routes,
        key=lambda route: (
            _selector_score(route, _objective_vector(route), weights=selector_weights),
            _route_id(route),
        ),
    )
    hard_case_stress_pack_count = sum(
        1 for world in evaluated.world_rows if str(world.get("world_kind", "")).strip().startswith("hard_case_")
    )
    supported_ambiguity_stress_pack_count = sum(
        1 for world in evaluated.world_rows if str(world.get("world_kind", "")).strip().startswith("supported_ambiguity_")
    )
    targeted_stress_pack_count = hard_case_stress_pack_count + supported_ambiguity_stress_pack_count
    world_manifest = {
        "world_count": len(evaluated.world_rows),
        "unique_world_count": int(evaluated.unique_world_count or len(evaluated.world_rows)),
        "active_families": list(evaluated.active_families),
        "worlds": evaluated.world_rows,
        "world_signatures": [_world_signature(world) for world in evaluated.world_rows],
        "world_reuse_rate": round(
            max(0, len(evaluated.world_rows) - int(evaluated.unique_world_count or len(evaluated.world_rows)))
            / float(max(1, len(evaluated.world_rows))),
            6,
        ),
        "targeted_stress_pack_count": targeted_stress_pack_count,
        "hard_case_stress_pack_count": hard_case_stress_pack_count,
        "supported_ambiguity_stress_pack_count": supported_ambiguity_stress_pack_count,
        "stress_world_fraction": round(targeted_stress_pack_count / float(max(1, len(evaluated.world_rows))), 6),
        "refc_stress_world_fraction": round(targeted_stress_pack_count / float(max(1, len(evaluated.world_rows))), 6),
        "hard_case_stress_world_fraction": round(
            hard_case_stress_pack_count / float(max(1, len(evaluated.world_rows))),
            6,
        ),
        "supported_ambiguity_stress_world_fraction": round(
            supported_ambiguity_stress_pack_count / float(max(1, len(evaluated.world_rows))),
            6,
        ),
        "manifest_hash": _stable_hash(
            [json.dumps(evaluated.world_rows, sort_keys=True, separators=(",", ":"), default=str)]
        ),
    }
    selector_config = {"selector_weights": list(selector_weights), "threshold": float(threshold)}
    certified = certificate.get(winner_id, 0.0) >= float(threshold)
    return CertificateResult(
        winner_id=winner_id,
        certificate=certificate,
        threshold=float(threshold),
        certified=certified,
        selected_route_id=_route_id(selected_route_id),
        route_scores=route_scores,
        world_manifest=world_manifest,
        selector_config=selector_config,
    )


def compute_certificate(
    routes: Sequence[Mapping[str, Any]],
    *,
    worlds: Sequence[Mapping[str, Any]],
    selector_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: float = 0.67,
    active_families: Sequence[str] | None = None,
    selector_score_map_fn: SelectorScoreMapFn | None = None,
    evaluated_bundle: _EvaluatedWorldBundle | None = None,
    ambiguity_context: Mapping[str, Any] | None = None,
) -> CertificateResult:
    # Winner frequency over the bounded world set is used as a sample-average
    # robustness certificate for the fixed selector, not as a posterior
    # probability claim about the unknown global optimum.
    if not routes:
        raise ValueError("routes cannot be empty")
    world_rows = [dict(world) for world in worlds]
    if not world_rows and evaluated_bundle is None:
        raise ValueError("worlds cannot be empty")
    families = tuple(active_families or active_evidence_families(routes))
    if not families:
        families = tuple(sorted({family for route in routes for family in _evidence_tensor(route)}))
    evaluated = evaluated_bundle or _evaluate_world_bundle(
        routes,
        world_rows,
        active_families=families,
        selector_weights=selector_weights,
        selector_score_map_fn=selector_score_map_fn,
    )
    result = _certificate_from_evaluated_bundle(
        routes,
        evaluated,
        selector_weights=selector_weights,
        threshold=threshold,
    )
    if ambiguity_context is not None:
        result.world_manifest["ambiguity_context"] = _normalised_ambiguity_context(
            {
                **dict(ambiguity_context),
                "targeted_stress_pack_count": result.world_manifest.get("targeted_stress_pack_count", 0),
                "hard_case_stress_pack_count": result.world_manifest.get("hard_case_stress_pack_count", 0),
                "supported_ambiguity_stress_pack_count": result.world_manifest.get(
                    "supported_ambiguity_stress_pack_count",
                    0,
                ),
                "stress_world_fraction": result.world_manifest.get("stress_world_fraction", 0.0),
                "refc_stress_world_fraction": result.world_manifest.get("refc_stress_world_fraction", 0.0),
                "hard_case_stress_world_fraction": result.world_manifest.get(
                    "hard_case_stress_world_fraction",
                    0.0,
                ),
                "supported_ambiguity_stress_world_fraction": result.world_manifest.get(
                    "supported_ambiguity_stress_world_fraction",
                    0.0,
                ),
            }
        )
    return result


def _refreshed_worlds(
    worlds: Sequence[Mapping[str, Any]],
    family: str,
    *,
    target_route_id: str | None = None,
) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    for world in worlds:
        states = dict(world.get("states", {}))
        if family in states:
            states[family] = "refreshed"
        row = {
            "world_id": world.get("world_id"),
            "states": states,
            "stress_factor": _as_float(world.get("stress_factor"), 1.0),
            "world_kind": world.get("world_kind"),
        }
        scope_route_id = target_route_id or world.get("target_route_id") or world.get("route_scope")
        if scope_route_id:
            row["target_route_id"] = str(scope_route_id)
        refreshed.append(row)
    return refreshed


def _stressed_worlds(
    worlds: Sequence[Mapping[str, Any]],
    family: str,
    *,
    stress_state: str,
    target_route_id: str | None = None,
) -> list[dict[str, Any]]:
    stressed: list[dict[str, Any]] = []
    for world in worlds:
        states = dict(world.get("states", {}))
        if family in states:
            states[family] = stress_state
        row = {
            "world_id": world.get("world_id"),
            "states": states,
            "stress_factor": _as_float(world.get("stress_factor"), 1.0),
            "world_kind": world.get("world_kind"),
        }
        scope_route_id = target_route_id or world.get("target_route_id") or world.get("route_scope")
        if scope_route_id:
            row["target_route_id"] = str(scope_route_id)
        stressed.append(row)
    return stressed


def _friction_from_certificates(
    base: float,
    refreshed: float,
) -> float:
    return max(0.0, refreshed - base)


def _near_tie_margin_signal(summary: Mapping[str, float]) -> float:
    signal = max(0.0, _as_float(summary.get("margin_stability_signal")))
    if signal <= 0.0 or signal >= 0.01:
        return 0.0
    return round(signal, 6)


def _runner_up_gap_summary(
    evaluated: _EvaluatedWorldBundle,
    route_id: str,
) -> dict[str, float]:
    gaps: list[float] = []
    for score_map in evaluated.score_maps:
        selected_score = score_map.get(route_id)
        if selected_score is None or not math.isfinite(float(selected_score)):
            continue
        competitor_scores = [
            float(score)
            for competitor_id, score in score_map.items()
            if competitor_id != route_id and math.isfinite(float(score))
        ]
        if not competitor_scores:
            continue
        gaps.append(float(min(competitor_scores) - float(selected_score)))
    if not gaps:
        return {
            "world_count": 0.0,
            "mean_runner_up_gap": 0.0,
            "min_runner_up_gap": 0.0,
            "max_runner_up_gap": 0.0,
            "positive_world_share": 0.0,
            "margin_stability_signal": 0.0,
        }
    positive_count = sum(1 for gap in gaps if gap > 0.0)
    mean_gap = sum(gaps) / float(len(gaps))
    min_gap = min(gaps)
    max_gap = max(gaps)
    stability_signal = max(0.0, mean_gap + (0.5 * (max_gap - min_gap)))
    return {
        "world_count": float(len(gaps)),
        "mean_runner_up_gap": round(mean_gap, 6),
        "min_runner_up_gap": round(min_gap, 6),
        "max_runner_up_gap": round(max_gap, 6),
        "positive_world_share": round(positive_count / float(len(gaps)), 6),
        "margin_stability_signal": round(stability_signal, 6),
    }


def _runner_up_gap_lift(
    baseline: Mapping[str, float],
    updated: Mapping[str, float],
) -> float:
    baseline_signal = _as_float(baseline.get("margin_stability_signal"), float("nan"))
    updated_signal = _as_float(updated.get("margin_stability_signal"), float("nan"))
    if math.isfinite(baseline_signal) and math.isfinite(updated_signal):
        raw_lift = max(0.0, updated_signal - baseline_signal)
        if raw_lift <= 1e-4:
            return 0.0
        signal_scale = max(
            0.01,
            abs(baseline_signal),
            abs(_as_float(baseline.get("mean_runner_up_gap"))),
            0.5 * abs(_as_float(updated.get("mean_runner_up_gap"))),
        )
        relative_lift = raw_lift / signal_scale
        bounded_relative_lift = 1.0 - math.exp(-2.5 * relative_lift)
        return min(0.01, max(raw_lift, bounded_relative_lift))
    raw_lift = max(0.0, _as_float(updated.get("mean_runner_up_gap")) - _as_float(baseline.get("mean_runner_up_gap")))
    if raw_lift <= 1e-4:
        return max(_near_tie_margin_signal(baseline), _near_tie_margin_signal(updated))
    signal_scale = max(
        0.01,
        abs(_as_float(baseline.get("mean_runner_up_gap"))),
        0.5 * abs(_as_float(updated.get("mean_runner_up_gap"))),
    )
    relative_lift = raw_lift / signal_scale
    bounded_relative_lift = 1.0 - math.exp(-2.5 * relative_lift)
    return min(0.01, max(raw_lift, bounded_relative_lift))


def _fragility_from_certificates(
    base: float,
    stressed: float,
) -> float:
    return max(0.0, base - stressed)


def _selector_weighted_family_exposure(
    route: Mapping[str, Any],
    family: str,
    *,
    active_families: Sequence[str],
    selector_weights: tuple[float, float, float],
) -> float:
    tensor = _route_dependency_weights(route, active_families=active_families)
    weight_total = 0.0
    weighted_exposure = 0.0
    for idx, objective in enumerate(OBJECTIVE_NAMES):
        objective_weight = max(
            0.0,
            _as_float(selector_weights[idx] if idx < len(selector_weights) else 1.0),
        )
        weighted_exposure += objective_weight * tensor.get(objective, {}).get(family, 0.0)
        weight_total += objective_weight
    if weight_total <= 0.0:
        return 0.0
    return _clamp01(weighted_exposure / weight_total)


def _winner_recoverable_fragility_floor(
    route: Mapping[str, Any],
    family: str,
    *,
    baseline_margin_summary: Mapping[str, float],
    active_families: Sequence[str],
    selector_weights: tuple[float, float, float],
    ambiguity_context: Mapping[str, Any] | None,
) -> float:
    context = _normalised_ambiguity_context(ambiguity_context)
    if not bool(context.get("is_hard_case") or context.get("is_supported_ambiguity_case")):
        return 0.0
    near_tie_signal = _near_tie_margin_signal(baseline_margin_summary)
    if near_tie_signal <= 0.0:
        return 0.0
    family_exposure = _selector_weighted_family_exposure(
        route,
        family,
        active_families=active_families,
        selector_weights=selector_weights,
    )
    if family_exposure <= 0.0:
        return 0.0
    support_strength = _clamp01(context.get("support_strength"), 0.0)
    context_strength = _clamp01(context.get("context_strength"), 0.0)
    margin_pressure = 1.0 - _clamp01(context.get("od_nominal_margin_proxy"), 1.0)
    provenance_scale = _family_route_provenance_scale(route, family)
    provenance_factor = max(0.72, min(1.0, provenance_scale / 1.25))
    floor = (
        near_tie_signal
        * (0.45 + (0.25 * support_strength) + (0.20 * context_strength) + (0.10 * margin_pressure))
        * max(0.18, family_exposure)
        * provenance_factor
    )
    return min(0.01, max(0.0, floor))


def _family_fully_refreshed_in_worlds(
    worlds: Sequence[Mapping[str, Any]],
    family: str,
) -> bool:
    relevant_worlds = 0
    refreshed_worlds = 0
    for world in worlds:
        states = world.get("states", {})
        if not isinstance(states, Mapping) or family not in states:
            continue
        relevant_worlds += 1
        if str(states.get(family, "")).strip() == "refreshed":
            refreshed_worlds += 1
    return relevant_worlds > 0 and refreshed_worlds == relevant_worlds


def _snapshot_payload_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    payload = None
    for key in ("snapshot_payload", "snapshot", "payload", "source_payload"):
        raw = entry.get(key)
        if isinstance(raw, Mapping):
            payload = dict(raw)
            break
    if payload is None:
        return {}
    payload_hash = _stable_hash([json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)])
    return {
        "snapshot_payload_hash": payload_hash,
        "snapshot_payload_keys": sorted(str(key) for key in payload),
    }


def _family_snapshot_manifest(
    routes: Sequence[Mapping[str, Any]],
    *,
    active_families: Sequence[str],
) -> dict[str, Any]:
    snapshots: dict[str, list[dict[str, Any]]] = {family: [] for family in active_families}
    for route in routes:
        route_id = _route_id(route)
        provenance = _route_provenance(route)
        raw_families = provenance.get("families", [])
        if not isinstance(raw_families, Sequence) or isinstance(raw_families, (str, bytes)):
            continue
        for entry in raw_families:
            if not isinstance(entry, Mapping):
                continue
            family = str(entry.get("family", "")).strip()
            if family not in snapshots:
                continue
            snapshot = {
                "source": str(entry.get("source", "unknown")),
                "freshness_timestamp_utc": entry.get("freshness_timestamp_utc"),
                "max_age_minutes": _as_float(entry.get("max_age_minutes")),
                "signature": entry.get("signature"),
                "confidence": _as_float(entry.get("confidence")),
                "coverage_ratio": _as_float(entry.get("coverage_ratio")),
                "fallback_used": bool(entry.get("fallback_used", False)),
                "fallback_source": entry.get("fallback_source"),
                "snapshot_id": entry.get("snapshot_id"),
                "route_id": route_id,
            }
            snapshot.update(_snapshot_payload_summary(entry))
            if snapshot not in snapshots[family]:
                snapshots[family].append(snapshot)
    payload = {
        "active_families": list(active_families),
        "route_ids": sorted(_route_id(route) for route in routes),
        "family_snapshots": {
            family: sorted(rows, key=lambda row: (str(row.get("source")), str(row.get("signature"))))
            for family, rows in snapshots.items()
        },
    }
    payload["manifest_hash"] = _stable_hash([json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)])
    return payload


def _certificate_matches_world_rows(
    certificate: CertificateResult | None,
    world_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if certificate is None:
        return False
    manifest = certificate.world_manifest if isinstance(certificate.world_manifest, Mapping) else {}
    if int(manifest.get("world_count", -1)) != len(world_rows):
        return False
    expected_hash = _stable_hash(
        [json.dumps([dict(world) for world in world_rows], sort_keys=True, separators=(",", ":"), default=str)]
    )
    return str(manifest.get("manifest_hash", "")) == expected_hash


def compute_fragility_maps(
    routes: Sequence[Mapping[str, Any]],
    *,
    worlds: Sequence[Mapping[str, Any]],
    selector_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    active_families: Sequence[str] | None = None,
    selected_route_id: str | None = None,
    selector_score_map_fn: SelectorScoreMapFn | None = None,
    evaluated_bundle: _EvaluatedWorldBundle | None = None,
    baseline_certificate: CertificateResult | None = None,
    ambiguity_context: Mapping[str, Any] | None = None,
) -> FragilityResult:
    # Family-level fragility is computed by isolated one-factor stress and
    # forced-refresh counterfactuals, mirroring classical sensitivity-analysis
    # practice and keeping attribution auditable.
    if not routes:
        raise ValueError("routes cannot be empty")
    families = tuple(active_families or active_evidence_families(routes))
    world_rows = [dict(world) for world in worlds]
    analysis_world_rows = [
        world
        for world in world_rows
        if not str(world.get("world_kind", "")).strip().endswith("_mixed_targeted")
    ]
    if not analysis_world_rows:
        analysis_world_rows = list(world_rows)
    shared_state_cache = (
        dict(evaluated_bundle.state_score_cache)
        if evaluated_bundle is not None and isinstance(evaluated_bundle.state_score_cache, dict)
        else {}
    )
    use_evaluated_bundle = (
        evaluated_bundle is not None
        and len(evaluated_bundle.world_rows) == len(analysis_world_rows)
        and not any(str(world.get("world_kind", "")).strip().endswith("_mixed_targeted") for world in evaluated_bundle.world_rows)
    )
    baseline_bundle = evaluated_bundle if use_evaluated_bundle else _evaluate_world_bundle(
        routes,
        analysis_world_rows,
        active_families=families,
        selector_weights=selector_weights,
        selector_score_map_fn=selector_score_map_fn,
        state_cache=shared_state_cache,
    )
    baseline = (
        baseline_certificate
        if _certificate_matches_world_rows(baseline_certificate, analysis_world_rows)
        else None
    )
    if baseline is None:
        baseline = _certificate_from_evaluated_bundle(
            routes,
            baseline_bundle,
            selector_weights=selector_weights,
            threshold=0.67,
        )
    baseline_certificate = baseline
    baseline = baseline_certificate
    refresh_baseline_bundle = (
        evaluated_bundle
        if evaluated_bundle is not None and len(evaluated_bundle.world_rows) == len(world_rows)
        else _evaluate_world_bundle(
            routes,
            world_rows,
            active_families=families,
            selector_weights=selector_weights,
            selector_score_map_fn=selector_score_map_fn,
            state_cache=shared_state_cache,
        )
    )
    refresh_baseline = (
        baseline_certificate
        if _certificate_matches_world_rows(baseline_certificate, world_rows)
        else _certificate_from_evaluated_bundle(
            routes,
            refresh_baseline_bundle,
            selector_weights=selector_weights,
            threshold=0.67,
        )
    )
    route_fragility_map: dict[str, dict[str, float]] = {}
    route_fragility_details: dict[str, dict[str, dict[str, float]]] = {}
    competitor_breakdown: dict[str, dict[str, dict[str, int]]] = {}
    stress_state = "severely_stale"
    context = _normalised_ambiguity_context(ambiguity_context)
    route_margin_baseline = {
        _route_id(route): _runner_up_gap_summary(baseline_bundle, _route_id(route))
        for route in routes
    }
    target_route_id = selected_route_id or refresh_baseline.winner_id
    target_route_stressed_bundles: dict[str, _EvaluatedWorldBundle] = {}
    target_route_stressed_certificates: dict[str, CertificateResult] = {}
    target_route_analysis_refreshed_bundles: dict[str, _EvaluatedWorldBundle] = {}
    target_route_analysis_refreshed_certificates: dict[str, CertificateResult] = {}
    target_route_refresh_bundles: dict[str, _EvaluatedWorldBundle] = {}
    target_route_refresh_certificates: dict[str, CertificateResult] = {}

    def _scope_world_rows(
        source_worlds: Sequence[Mapping[str, Any]],
        route_id: str,
    ) -> list[dict[str, Any]]:
        scoped_rows: list[dict[str, Any]] = []
        for world in source_worlds:
            row = dict(world)
            row["target_route_id"] = route_id
            scoped_rows.append(row)
        return scoped_rows

    for route in routes:
        route_id = _route_id(route)
        route_fragility_map[route_id] = {}
        route_fragility_details[route_id] = {}
        competitor_breakdown[route_id] = {}
        route_scoped_analysis_world_rows = _scope_world_rows(analysis_world_rows, route_id)
        route_scoped_full_world_rows = _scope_world_rows(world_rows, route_id)
        route_stressed_bundles: dict[str, _EvaluatedWorldBundle] = {}
        route_analysis_refreshed_bundles: dict[str, _EvaluatedWorldBundle] = {}
        route_refreshed_bundles: dict[str, _EvaluatedWorldBundle] = {}
        route_analysis_refreshed_certificates: dict[str, CertificateResult] = {}
        route_refreshed_certificates: dict[str, CertificateResult] = {}
        for family in families:
            stressed_world_rows = _stressed_worlds(
                route_scoped_analysis_world_rows,
                family,
                stress_state=stress_state,
                target_route_id=route_id,
            )
            analysis_refreshed_world_rows = _refreshed_worlds(
                route_scoped_analysis_world_rows,
                family,
                target_route_id=route_id,
            )
            refreshed_world_rows = _refreshed_worlds(
                route_scoped_full_world_rows,
                family,
                target_route_id=route_id,
            )
            stressed_bundle = _evaluate_world_bundle(
                routes,
                stressed_world_rows,
                active_families=families,
                selector_weights=selector_weights,
                selector_score_map_fn=selector_score_map_fn,
                state_cache=shared_state_cache,
            )
            stressed = _certificate_from_evaluated_bundle(
                routes,
                stressed_bundle,
                selector_weights=selector_weights,
                threshold=0.67,
            )
            analysis_refreshed_bundle = _evaluate_world_bundle(
                routes,
                analysis_refreshed_world_rows,
                active_families=families,
                selector_weights=selector_weights,
                selector_score_map_fn=selector_score_map_fn,
                state_cache=shared_state_cache,
            )
            refreshed = _certificate_from_evaluated_bundle(
                routes,
                analysis_refreshed_bundle,
                selector_weights=selector_weights,
                threshold=0.67,
            )
            refreshed_bundle = _evaluate_world_bundle(
                routes,
                refreshed_world_rows,
                active_families=families,
                selector_weights=selector_weights,
                selector_score_map_fn=selector_score_map_fn,
                state_cache=shared_state_cache,
            )
            refreshed_certificate = _certificate_from_evaluated_bundle(
                routes,
                refreshed_bundle,
                selector_weights=selector_weights,
                threshold=0.67,
            )
            route_stressed_bundles[family] = stressed_bundle
            route_analysis_refreshed_bundles[family] = analysis_refreshed_bundle
            route_refreshed_bundles[family] = refreshed_bundle
            route_analysis_refreshed_certificates[family] = refreshed
            route_refreshed_certificates[family] = refreshed_certificate
            if route_id == target_route_id:
                target_route_stressed_bundles[family] = stressed_bundle
                target_route_stressed_certificates[family] = stressed
                target_route_analysis_refreshed_bundles[family] = analysis_refreshed_bundle
                target_route_analysis_refreshed_certificates[family] = refreshed
                target_route_refresh_bundles[family] = refreshed_bundle
                target_route_refresh_certificates[family] = refreshed_certificate
            baseline_margin_summary = route_margin_baseline.get(route_id, {})
            stressed_margin_summary = _runner_up_gap_summary(stressed_bundle, route_id)
            refreshed_margin_summary = _runner_up_gap_summary(analysis_refreshed_bundle, route_id)
            margin_fragility = _runner_up_gap_lift(stressed_margin_summary, baseline_margin_summary)
            margin_refresh_gain = _runner_up_gap_lift(baseline_margin_summary, refreshed_margin_summary)
            certificate_refresh_gain = _friction_from_certificates(
                baseline.certificate.get(route_id, 0.0),
                refreshed.certificate.get(route_id, 0.0),
            )
            certificate_stress_recovery = _friction_from_certificates(
                stressed.certificate.get(route_id, 0.0),
                refreshed.certificate.get(route_id, 0.0),
            )
            margin_stress_recovery = _runner_up_gap_lift(stressed_margin_summary, refreshed_margin_summary)
            fragility_value = _fragility_from_certificates(
                baseline.certificate.get(route_id, 0.0),
                stressed.certificate.get(route_id, 0.0),
            )
            realized_fragility = max(fragility_value, margin_fragility)
            recoverable_winner_floor = 0.0
            if route_id == target_route_id and realized_fragility <= 1e-6:
                recoverable_winner_floor = _winner_recoverable_fragility_floor(
                    route,
                    family,
                    baseline_margin_summary=baseline_margin_summary,
                    active_families=families,
                    selector_weights=selector_weights,
                    ambiguity_context=context,
                )
            family_fully_refreshed = _family_fully_refreshed_in_worlds(route_scoped_full_world_rows, family)
            # Keep the recoverable floor as explanatory fragility, but do not let it
            # fabricate actionable refresh value without empirical counterfactual support.
            empirical_refresh_gain = max(
                certificate_refresh_gain,
                margin_refresh_gain,
                certificate_stress_recovery,
                margin_stress_recovery,
            )
            raw_refresh_gain = max(
                empirical_refresh_gain,
                recoverable_winner_floor,
            )
            actionable_refresh_gain = 0.0 if family_fully_refreshed else empirical_refresh_gain
            route_fragility_map[route_id][family] = max(realized_fragility, recoverable_winner_floor)
            route_fragility_details[route_id][family] = {
                "baseline_certificate": baseline.certificate.get(route_id, 0.0),
                "stressed_certificate": stressed.certificate.get(route_id, 0.0),
                "refreshed_certificate": refreshed.certificate.get(route_id, 0.0),
                "baseline_margin_summary": dict(baseline_margin_summary),
                "stressed_margin_summary": dict(stressed_margin_summary),
                "refreshed_margin_summary": dict(refreshed_margin_summary),
                "absolute_drop": fragility_value,
                "normalized_drop": (
                    fragility_value / max(1e-9, baseline.certificate.get(route_id, 0.0))
                    if baseline.certificate.get(route_id, 0.0) > 0.0
                    else 0.0
                ),
                "margin_fragility": margin_fragility,
                "margin_refresh_gain": margin_refresh_gain,
                "certificate_refresh_gain": certificate_refresh_gain,
                "certificate_stress_recovery": certificate_stress_recovery,
                "margin_stress_recovery": margin_stress_recovery,
                "winner_recoverable_floor": recoverable_winner_floor,
                "family_fully_refreshed": family_fully_refreshed,
                "raw_refresh_gain": raw_refresh_gain,
                "refresh_gain": actionable_refresh_gain,
            }
        for competitor in routes:
            competitor_id = _route_id(competitor)
            if competitor_id == route_id:
                continue
            competitor_breakdown[route_id][competitor_id] = {}
            for family in families:
                count = 0
                for route_scores in route_stressed_bundles[family].score_maps:
                    if route_scores.get(competitor_id, float("inf")) < route_scores.get(route_id, float("inf")):
                        count += 1
                competitor_breakdown[route_id][competitor_id][family] = count
    vor: dict[str, float] = {}
    for family in families:
        refreshed = target_route_refresh_certificates.get(family)
        refreshed_bundle = target_route_refresh_bundles.get(family)
        stressed = target_route_stressed_certificates.get(family)
        stressed_bundle = target_route_stressed_bundles.get(family)
        analysis_refreshed = target_route_analysis_refreshed_certificates.get(family)
        analysis_refreshed_bundle = target_route_analysis_refreshed_bundles.get(family)
        if (
            refreshed is None
            or refreshed_bundle is None
            or stressed is None
            or stressed_bundle is None
            or analysis_refreshed is None
            or analysis_refreshed_bundle is None
        ):
            continue
        refresh_margin_gain = _runner_up_gap_lift(
            _runner_up_gap_summary(refresh_baseline_bundle, target_route_id),
            _runner_up_gap_summary(refreshed_bundle, target_route_id),
        )
        stress_recovery_gain = _friction_from_certificates(
            stressed.certificate.get(target_route_id, 0.0),
            analysis_refreshed.certificate.get(target_route_id, 0.0),
        )
        stress_recovery_margin_gain = _runner_up_gap_lift(
            _runner_up_gap_summary(stressed_bundle, target_route_id),
            _runner_up_gap_summary(analysis_refreshed_bundle, target_route_id),
        )
        family_fully_refreshed = _family_fully_refreshed_in_worlds(world_rows, family)
        # The floor may explain fragility, but top-level VOR should only reflect
        # empirical refresh recovery and not a heuristic prior alone.
        empirical_refresh_gain = max(
            _friction_from_certificates(
                refresh_baseline.certificate.get(target_route_id, 0.0),
                refreshed.certificate.get(target_route_id, 0.0),
            ),
            refresh_margin_gain,
            stress_recovery_gain,
            stress_recovery_margin_gain,
        )
        raw_refresh_gain = max(
            empirical_refresh_gain,
            route_fragility_details.get(target_route_id, {}).get(family, {}).get("winner_recoverable_floor", 0.0),
        )
        vor[family] = 0.0 if family_fully_refreshed else empirical_refresh_gain
    ranked = sorted(vor.items(), key=lambda item: (-item[1], item[0]))
    return FragilityResult(
        route_fragility_map=route_fragility_map,
        competitor_fragility_breakdown=competitor_breakdown,
        value_of_refresh={
            "selected_route_id": target_route_id,
            "baseline_certificate": refresh_baseline.certificate.get(target_route_id, 0.0),
            "baseline_margin_summary": _runner_up_gap_summary(refresh_baseline_bundle, target_route_id),
            "fragility_stress_state": stress_state,
            "per_family_certificate": {
                family: target_route_refresh_certificates[family].certificate.get(target_route_id, 0.0)
                for family in families
            },
            "per_family_margin_summary": {
                family: _runner_up_gap_summary(target_route_refresh_bundles[family], target_route_id)
                for family in families
            },
            "ranking": [
                {"family": family, "vor": value}
                for family, value in ranked
            ],
            "top_refresh_family": ranked[0][0] if ranked else None,
            "top_refresh_gain": ranked[0][1] if ranked else 0.0,
        },
        route_fragility_details=route_fragility_details,
        evidence_snapshot_manifest={
            **_family_snapshot_manifest(routes, active_families=families),
            "ambiguity_context": _normalised_ambiguity_context(ambiguity_context),
            "baseline_unique_world_count": int(baseline_bundle.unique_world_count or len(baseline_bundle.world_rows)),
        },
    )


def rank_value_of_refresh(
    routes: Sequence[Mapping[str, Any]],
    *,
    worlds: Sequence[Mapping[str, Any]],
    selector_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    active_families: Sequence[str] | None = None,
    selected_route_id: str | None = None,
    selector_score_map_fn: SelectorScoreMapFn | None = None,
) -> dict[str, Any]:
    return compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=selector_weights,
        active_families=active_families,
        selected_route_id=selected_route_id,
        selector_score_map_fn=selector_score_map_fn,
    ).value_of_refresh
