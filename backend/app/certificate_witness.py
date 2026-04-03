from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .abstention import AbstentionRecord
from .certification_models import CertificationState


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
        top_fragility_family = None
        if fragility_payload.get("top_refresh_family") is not None:
            top_fragility_family = str(fragility_payload.get("top_refresh_family"))
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
