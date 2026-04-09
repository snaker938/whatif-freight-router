from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .abstention import AbstentionRecord
from .certification_models import CertificationState


def _coerce_route_fragility_map(
    fragility: Mapping[str, Any] | None,
) -> dict[str, dict[str, float]]:
    if not isinstance(fragility, Mapping):
        return {}
    raw_route_map = fragility.get("route_fragility_map")
    candidate_route_map = raw_route_map if isinstance(raw_route_map, Mapping) else fragility
    route_fragility_map: dict[str, dict[str, float]] = {}
    for route_id, family_scores in candidate_route_map.items():
        if not isinstance(route_id, str) or not isinstance(family_scores, Mapping):
            continue
        numeric_scores: dict[str, float] = {}
        for family, score in family_scores.items():
            try:
                numeric_scores[str(family)] = float(score)
            except (TypeError, ValueError):
                continue
        if numeric_scores:
            route_fragility_map[route_id] = numeric_scores
    return route_fragility_map


def _derive_top_fragility_family(
    winner_id: str,
    fragility: Mapping[str, Any] | None,
) -> str | None:
    if not isinstance(fragility, Mapping):
        return None
    if fragility.get("top_refresh_family") is not None:
        return str(fragility.get("top_refresh_family"))
    route_fragility_map = _coerce_route_fragility_map(fragility)
    winner_scores = route_fragility_map.get(winner_id)
    if winner_scores is None and len(route_fragility_map) == 1:
        winner_scores = next(iter(route_fragility_map.values()))
    if not winner_scores:
        return None
    return min(
        winner_scores.items(),
        key=lambda item: (-float(item[1]), str(item[0])),
    )[0]


@dataclass(frozen=True)
class CertificateWitness:
    witness_id: str
    winner_id: str
    threshold: float
    certificate_value: float
    lower_bound: float
    upper_bound: float
    decision_status: str
    certified_route_ids: tuple[str, ...]
    best_challenger_id: str | None
    best_challenger_upper_bound: float
    support_strength: float
    proxy_world_fraction: float
    audit_correction: float
    stress_world_fraction: float
    top_fragility_family: str | None
    recommended_action: str
    reasons: tuple[str, ...]
    manifest_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_consistent_with_state(self, state: CertificationState) -> bool:
        return bool(
            self.winner_id == state.winner_id
            and self.decision_status == state.decision_region.status
            and self.certified_route_ids == state.certified_set.certified_route_ids
            and abs(self.certificate_value - state.certificate_map.get(state.winner_id, 0.0)) <= 1e-9
        )

    @classmethod
    def from_state(
        cls,
        state: CertificationState,
        *,
        fragility: Mapping[str, Any] | None = None,
        abstention: AbstentionRecord | None = None,
    ) -> "CertificateWitness":
        fragility_payload = dict(fragility) if isinstance(fragility, Mapping) else {}
        top_fragility_family = _derive_top_fragility_family(state.winner_id, fragility_payload)
        recommended_action = abstention.recommended_action if abstention is not None else (
            "hold" if state.certified else "review"
        )
        reasons = (
            (abstention.reason_code,)
            if abstention is not None
            else state.decision_region.reason_codes
        )
        witness_id = f"{state.manifest_hash or 'witness'}:{state.winner_id}:{state.decision_region.status}"
        return cls(
            witness_id=witness_id,
            winner_id=state.winner_id,
            threshold=state.threshold,
            certificate_value=state.certificate_map.get(state.winner_id, 0.0),
            lower_bound=state.winner_confidence.lower_bound,
            upper_bound=state.winner_confidence.upper_bound,
            decision_status=state.decision_region.status,
            certified_route_ids=state.certified_set.certified_route_ids,
            best_challenger_id=state.decision_region.best_challenger_id,
            best_challenger_upper_bound=state.decision_region.best_challenger_upper_bound,
            support_strength=state.support_state.support_strength,
            proxy_world_fraction=state.world_bundle.proxy_world_fraction,
            audit_correction=state.support_state.audit_correction,
            stress_world_fraction=state.world_bundle.stress_world_fraction,
            top_fragility_family=top_fragility_family,
            recommended_action=recommended_action,
            reasons=tuple(reasons),
            manifest_hash=state.manifest_hash,
        )
