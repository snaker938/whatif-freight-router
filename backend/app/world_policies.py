from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed != parsed:
        return float(default)
    return max(0.0, min(1.0, parsed))


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        return {str(key): 0.0 for key in weights}
    return {
        str(key): round(max(0.0, float(value)) / total, 6)
        for key, value in weights.items()
    }


def bundle_worlds(world_manifest: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(world_manifest, Mapping):
        return []
    raw_worlds = world_manifest.get("worlds", [])
    if not isinstance(raw_worlds, Sequence) or isinstance(raw_worlds, (str, bytes)):
        return []
    return [dict(world) for world in raw_worlds if isinstance(world, Mapping)]


def active_families(world_manifest: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(world_manifest, Mapping):
        return ()
    raw = world_manifest.get("active_families", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(sorted({str(family).strip() for family in raw if str(family).strip()}))


def world_kind_weights(worlds: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    counter: Counter[str] = Counter()
    for world in worlds:
        kind = str(world.get("world_kind", "sampled")).strip() or "sampled"
        counter[kind] += 1
    return normalize_weights(counter)


def family_state_weights(
    worlds: Sequence[Mapping[str, Any]],
    families: Sequence[str],
) -> dict[str, dict[str, float]]:
    family_counters: dict[str, Counter[str]] = {
        str(family): Counter() for family in families if str(family).strip()
    }
    for world in worlds:
        states = world.get("states", {})
        if not isinstance(states, Mapping):
            continue
        for family in family_counters:
            state = str(states.get(family, "nominal")).strip() or "nominal"
            family_counters[family][state] += 1
    return {
        family: normalize_weights(counter)
        for family, counter in family_counters.items()
    }


def targeting_fraction(worlds: Sequence[Mapping[str, Any]]) -> float:
    if not worlds:
        return 0.0
    targeted = 0
    for world in worlds:
        if str(world.get("target_route_id", "")).strip():
            targeted += 1
            continue
        raw_targets = world.get("target_route_ids", world.get("route_scope_by_family", {}))
        if isinstance(raw_targets, Mapping) and any(str(value).strip() for value in raw_targets.values()):
            targeted += 1
    return round(targeted / float(len(worlds)), 6)


def state_fraction(worlds: Sequence[Mapping[str, Any]], state_name: str) -> float:
    state_key = str(state_name).strip()
    if not worlds or not state_key:
        return 0.0
    total_slots = 0
    matching_slots = 0
    for world in worlds:
        states = world.get("states", {})
        if not isinstance(states, Mapping):
            continue
        for value in states.values():
            total_slots += 1
            if str(value).strip() == state_key:
                matching_slots += 1
    if total_slots <= 0:
        return 0.0
    return round(matching_slots / float(total_slots), 6)


def proxy_state_fraction(worlds: Sequence[Mapping[str, Any]]) -> float:
    return state_fraction(worlds, "proxy")


def refreshed_state_fraction(worlds: Sequence[Mapping[str, Any]]) -> float:
    return state_fraction(worlds, "refreshed")


def stress_world_fraction(world_manifest: Mapping[str, Any] | None) -> float:
    if not isinstance(world_manifest, Mapping):
        return 0.0
    candidates = (
        world_manifest.get("stress_world_fraction"),
        world_manifest.get("refc_stress_world_fraction"),
        world_manifest.get("hard_case_stress_world_fraction"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        return round(clamp01(candidate), 6)
    return 0.0


def effective_world_count(world_manifest: Mapping[str, Any] | None) -> int:
    if not isinstance(world_manifest, Mapping):
        return 0
    raw = world_manifest.get(
        "effective_world_count",
        world_manifest.get("world_count", len(bundle_worlds(world_manifest))),
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = len(bundle_worlds(world_manifest))
    return max(0, value)


def infer_world_policy(world_manifest: Mapping[str, Any] | None) -> str:
    if not isinstance(world_manifest, Mapping):
        return "unknown"
    status = str(world_manifest.get("status", "")).strip().lower()
    policy = str(world_manifest.get("world_count_policy", "")).strip().lower()
    stress_fraction = stress_world_fraction(world_manifest)
    reuse_rate = clamp01(world_manifest.get("world_reuse_rate"))
    if "shortcut" in status or "shortcut" in policy:
        return "shortcut"
    if stress_fraction > 0.0:
        return "targeted_stress"
    if reuse_rate > 0.0:
        return "replay_with_reuse"
    return "deterministic_replay"
