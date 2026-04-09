from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .world_policies import (
    active_families,
    bundle_worlds,
    clamp01,
    effective_world_count,
    family_state_weights,
    infer_world_policy,
    proxy_state_fraction,
    refreshed_state_fraction,
    stress_world_fraction,
    targeting_fraction,
    world_kind_weights,
)


@dataclass(frozen=True)
class ProbabilisticWorldBundle:
    bundle_id: str
    route_ids: tuple[str, ...]
    active_families: tuple[str, ...]
    world_count: int
    unique_world_count: int
    requested_world_count: int
    effective_world_count: int
    world_kind_weights: dict[str, float]
    family_state_weights: dict[str, dict[str, float]]
    targeted_route_fraction: float
    stress_world_fraction: float
    proxy_world_fraction: float
    refreshed_world_fraction: float
    world_reuse_rate: float
    policy: str
    manifest_hash: str | None = None

    @property
    def nominal_state_mass(self) -> float:
        if not self.family_state_weights:
            return 1.0
        masses = [
            float(state_weights.get("nominal", 0.0))
            for state_weights in self.family_state_weights.values()
        ]
        return round(sum(masses) / float(len(masses)), 6)

    @property
    def effective_support_mass(self) -> float:
        mass = (
            self.nominal_state_mass
            * max(0.2, 1.0 - (0.75 * self.proxy_world_fraction))
            * max(0.2, 1.0 - (0.35 * self.stress_world_fraction))
        )
        return round(clamp01(mass), 6)

    @property
    def multi_fidelity_mode(self) -> str:
        return "probabilistic" if self.world_count > 1 else "deterministic"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_manifest(
        cls,
        world_manifest: Mapping[str, Any] | None,
        *,
        route_ids: Sequence[str] = (),
    ) -> "ProbabilisticWorldBundle":
        manifest = dict(world_manifest) if isinstance(world_manifest, Mapping) else {}
        worlds = bundle_worlds(manifest)
        families = active_families(manifest)
        manifest_route_ids = {
            str(route_id).strip()
            for route_id in route_ids
            if str(route_id).strip()
        }
        selected_route_id = str(manifest.get("selected_route_id", "")).strip()
        if selected_route_id:
            manifest_route_ids.add(selected_route_id)
        for world in worlds:
            target_route_id = str(world.get("target_route_id", "")).strip()
            if target_route_id:
                manifest_route_ids.add(target_route_id)
            raw_targets = world.get("target_route_ids", world.get("route_scope_by_family", {}))
            if isinstance(raw_targets, Mapping):
                for route_id in raw_targets.values():
                    route_text = str(route_id).strip()
                    if route_text:
                        manifest_route_ids.add(route_text)
        world_count = max(
            len(worlds),
            int(manifest.get("world_count", len(worlds)) or len(worlds)),
        )
        unique_world_count = max(
            0,
            int(manifest.get("unique_world_count", world_count) or world_count),
        )
        requested_world_count = max(
            0,
            int(manifest.get("requested_world_count", world_count) or world_count),
        )
        return cls(
            bundle_id=str(manifest.get("manifest_hash", "probabilistic_bundle")).strip() or "probabilistic_bundle",
            route_ids=tuple(sorted(manifest_route_ids)),
            active_families=families,
            world_count=world_count,
            unique_world_count=unique_world_count,
            requested_world_count=requested_world_count,
            effective_world_count=effective_world_count(manifest),
            world_kind_weights=world_kind_weights(worlds),
            family_state_weights=family_state_weights(worlds, families),
            targeted_route_fraction=targeting_fraction(worlds),
            stress_world_fraction=stress_world_fraction(manifest),
            proxy_world_fraction=proxy_state_fraction(worlds),
            refreshed_world_fraction=refreshed_state_fraction(worlds),
            world_reuse_rate=clamp01(manifest.get("world_reuse_rate")),
            policy=infer_world_policy(manifest),
            manifest_hash=str(manifest.get("manifest_hash", "")).strip() or None,
        )
