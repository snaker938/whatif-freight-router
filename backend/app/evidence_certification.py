from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

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
    "mildly_stale": (0.015, 0.010, 0.012),
    "severely_stale": (0.045, 0.035, 0.030),
    "low_confidence": (0.030, 0.025, 0.020),
    "proxy": (0.060, 0.050, 0.045),
    "refreshed": (-0.015, -0.010, -0.008),
}

DEFAULT_FAMILY_SENSITIVITY: dict[str, tuple[float, float, float]] = {
    "scenario": (0.70, 0.30, 0.20),
    "toll": (0.05, 0.95, 0.10),
    "terrain": (0.45, 0.20, 0.70),
    "fuel": (0.10, 0.85, 0.25),
    "carbon": (0.00, 0.80, 0.65),
    "weather": (0.60, 0.15, 0.15),
    "stochastic": (0.55, 0.55, 0.45),
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


def _normalise_weights(weights: Mapping[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, _as_float(value)) for key, value in weights.items()}
    total = sum(positive.values())
    if total <= 0.0:
        return {key: 0.0 for key in positive}
    return {key: value / total for key, value in positive.items()}


def _route_dependency_weights(
    route: Mapping[str, Any],
    *,
    active_families: Sequence[str],
) -> dict[str, dict[str, float]]:
    tensor = _evidence_tensor(route)
    provenance = _route_provenance(route)
    out: dict[str, dict[str, float]] = {}
    for objective in OBJECTIVE_NAMES:
        objective_weights = {
            family: DEFAULT_OBJECTIVE_BIAS[objective].get(family, 0.0) for family in active_families
        }
        route_specific = provenance.get("dependency_weights", {})
        if isinstance(route_specific, Mapping):
            for family, family_weights in route_specific.items():
                if family not in active_families or not isinstance(family_weights, Mapping):
                    continue
                value = _as_float(family_weights.get(objective))
                if value > 0.0:
                    objective_weights[family] = value
        for family in active_families:
            family_weights = tensor.get(family, {})
            if objective in family_weights:
                objective_weights[family] = family_weights[objective]
        out[objective] = _normalise_weights(objective_weights)
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
            json.dumps(world.get("states", {}), sort_keys=True, separators=(",", ":"), default=str),
        ]
    )


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

    def as_dict(self) -> dict[str, Any]:
        return {"world_id": self.world_id, "states": dict(self.states)}


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_fragility_map": self.route_fragility_map,
            "competitor_fragility_breakdown": self.competitor_fragility_breakdown,
            "value_of_refresh": self.value_of_refresh,
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


def sample_world_manifest(
    *,
    active_families: Sequence[str],
    seed: int,
    world_count: int = 64,
    state_catalog: Sequence[str] = EVIDENCE_STATES,
) -> dict[str, Any]:
    families = sorted({family for family in active_families if family in EVIDENCE_FAMILIES})
    states = tuple(state_catalog) if state_catalog else EVIDENCE_STATES
    if not states:
        raise ValueError("state catalog cannot be empty")
    worlds: list[WorldSample] = []
    for idx in range(max(1, int(world_count))):
        state_map: dict[str, str] = {}
        for family in families:
            digest = hashlib.sha1(f"{seed}|{idx}|{family}|{','.join(states)}".encode("utf-8")).digest()
            state_index = int.from_bytes(digest[:8], "big") % len(states)
            state_map[family] = states[state_index]
        world_id = _stable_hash([str(seed), str(idx), json.dumps(state_map, sort_keys=True)])
        worlds.append(WorldSample(world_id=world_id, states=state_map))
    payload = {
        "seed": int(seed),
        "world_count": len(worlds),
        "active_families": families,
        "state_catalog": list(states),
        "worlds": [world.as_dict() for world in worlds],
    }
    payload["manifest_hash"] = _stable_hash([json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)])
    return payload


def dependency_tensor(
    route: Mapping[str, Any],
    *,
    active_families: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
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
    states = world.get("states", {})
    if not isinstance(states, Mapping):
        states = {}
    perturbed = [float(base[idx]) for idx in range(3)]
    for family in active_families:
        state = str(states.get(family, "nominal"))
        state_effect = _state_effect(state)
        sensitivity = _family_sensitivity(family)
        for idx, objective in enumerate(OBJECTIVE_NAMES):
            weight = tensor[objective].get(family, 0.0)
            delta_ratio = state_effect[idx] * sensitivity[idx] * weight
            perturbed[idx] = max(0.0, perturbed[idx] * (1.0 + delta_ratio))
    return float(perturbed[0]), float(perturbed[1]), float(perturbed[2])


def compute_certificate(
    routes: Sequence[Mapping[str, Any]],
    *,
    worlds: Sequence[Mapping[str, Any]],
    selector_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: float = 0.67,
    active_families: Sequence[str] | None = None,
    selector_score_map_fn: SelectorScoreMapFn | None = None,
) -> CertificateResult:
    if not routes:
        raise ValueError("routes cannot be empty")
    families = tuple(active_families or active_evidence_families(routes))
    if not families:
        families = tuple(sorted({family for route in routes for family in _evidence_tensor(route)}))
    route_scores: dict[str, list[float]] = {_route_id(route): [] for route in routes}
    certificate_counts: dict[str, int] = {route_id: 0 for route_id in route_scores}
    for world in worlds:
        perturbed_by_id: dict[str, tuple[float, float, float]] = {}
        for route in routes:
            route_id = _route_id(route)
            perturbed = _route_perturbed_objectives(route, world, active_families=families)
            perturbed_by_id[route_id] = perturbed
        if selector_score_map_fn is not None:
            raw_score_map = selector_score_map_fn(routes, perturbed_by_id)
            score_map = {
                route_id: _as_float(raw_score_map.get(route_id), float("inf"))
                for route_id in route_scores
            }
        else:
            score_map = {
                route_id: _selector_score(route, perturbed_by_id[route_id], weights=selector_weights)
                for route, route_id in zip(routes, route_scores, strict=True)
            }
        for route_id, score in score_map.items():
            route_scores[route_id].append(float(score))
        winner = min(score_map.items(), key=lambda item: (item[1], item[0]))[0]
        certificate_counts[winner] += 1
    world_count = max(1, len(worlds))
    certificate = {route_id: certificate_counts[route_id] / float(world_count) for route_id in certificate_counts}
    winner_id = min(
        certificate.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    selected_route_id = min(
        routes,
        key=lambda route: (
            _selector_score(route, _objective_vector(route), weights=selector_weights),
            _route_id(route),
        ),
    )
    world_manifest = {
        "world_count": len(worlds),
        "active_families": list(families),
        "worlds": [dict(world) for world in worlds],
        "manifest_hash": _stable_hash([json.dumps(list(worlds), sort_keys=True, separators=(",", ":"), default=str)]),
    }
    selector_config = {
        "selector_weights": list(selector_weights),
        "threshold": float(threshold),
    }
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


def _refreshed_worlds(
    worlds: Sequence[Mapping[str, Any]],
    family: str,
) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    for world in worlds:
        states = dict(world.get("states", {}))
        if family in states:
            states[family] = "refreshed"
        refreshed.append({"world_id": world.get("world_id"), "states": states})
    return refreshed


def _friction_from_certificates(
    base: float,
    refreshed: float,
) -> float:
    return max(0.0, refreshed - base)


def compute_fragility_maps(
    routes: Sequence[Mapping[str, Any]],
    *,
    worlds: Sequence[Mapping[str, Any]],
    selector_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    active_families: Sequence[str] | None = None,
    selected_route_id: str | None = None,
    selector_score_map_fn: SelectorScoreMapFn | None = None,
) -> FragilityResult:
    if not routes:
        raise ValueError("routes cannot be empty")
    families = tuple(active_families or active_evidence_families(routes))
    baseline = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=selector_weights,
        active_families=families,
        selector_score_map_fn=selector_score_map_fn,
    )
    route_fragility_map: dict[str, dict[str, float]] = {}
    competitor_breakdown: dict[str, dict[str, dict[str, int]]] = {}
    for route in routes:
        route_id = _route_id(route)
        route_fragility_map[route_id] = {}
        competitor_breakdown[route_id] = {}
        for family in families:
            refreshed_worlds = _refreshed_worlds(worlds, family)
            refreshed = compute_certificate(
                routes,
                worlds=refreshed_worlds,
                selector_weights=selector_weights,
                active_families=families,
                selector_score_map_fn=selector_score_map_fn,
            )
            route_fragility_map[route_id][family] = _friction_from_certificates(
                baseline.certificate.get(route_id, 0.0),
                refreshed.certificate.get(route_id, 0.0),
            )
        for competitor in routes:
            competitor_id = _route_id(competitor)
            if competitor_id == route_id:
                continue
            competitor_breakdown[route_id][competitor_id] = {}
            for family in families:
                count = 0
                for world in worlds:
                    states = dict(world.get("states", {}))
                    if family in states:
                        weak_state = states[family]
                        if weak_state == "nominal":
                            continue
                    perturbed_by_id = {
                        _route_id(row): _route_perturbed_objectives(row, world, active_families=families)
                        for row in routes
                    }
                    if selector_score_map_fn is not None:
                        raw_score_map = selector_score_map_fn(routes, perturbed_by_id)
                        route_scores = {
                            route_id: _as_float(raw_score_map.get(route_id), float("inf"))
                            for route_id in perturbed_by_id
                        }
                    else:
                        route_scores = {
                            _route_id(row): _selector_score(
                                row,
                                perturbed_by_id[_route_id(row)],
                                weights=selector_weights,
                            )
                            for row in routes
                        }
                    winner = min(route_scores.items(), key=lambda item: (item[1], item[0]))[0]
                    if winner == competitor_id:
                        count += 1
                competitor_breakdown[route_id][competitor_id][family] = count
    target_route_id = selected_route_id or baseline.winner_id
    vor: dict[str, float] = {}
    for family in families:
        refreshed_worlds = _refreshed_worlds(worlds, family)
        refreshed = compute_certificate(
            routes,
            worlds=refreshed_worlds,
            selector_weights=selector_weights,
            active_families=families,
            selector_score_map_fn=selector_score_map_fn,
        )
        vor[family] = _friction_from_certificates(
            baseline.certificate.get(target_route_id, 0.0),
            refreshed.certificate.get(target_route_id, 0.0),
        )
    ranked = sorted(vor.items(), key=lambda item: (-item[1], item[0]))
    return FragilityResult(
        route_fragility_map=route_fragility_map,
        competitor_fragility_breakdown=competitor_breakdown,
        value_of_refresh={
            "selected_route_id": target_route_id,
            "baseline_certificate": baseline.certificate.get(target_route_id, 0.0),
            "ranking": [
                {"family": family, "vor": value}
                for family, value in ranked
            ],
            "top_refresh_family": ranked[0][0] if ranked else None,
            "top_refresh_gain": ranked[0][1] if ranked else 0.0,
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
