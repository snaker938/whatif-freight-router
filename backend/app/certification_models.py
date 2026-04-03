from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .audit_correction import AuditWorldBundle
from .certified_set import CertifiedSetState
from .confidence_sequences import WinnerConfidenceState
from .decision_region import DecisionRegionState
from .fidelity_model import ProbabilisticWorldBundle
from .flip_radius import FlipRadiusState
from .pairwise_gap_model import PairwiseGapState
from .support_model import WorldSupportState


@dataclass(frozen=True)
class CertificationState:
    winner_id: str
    threshold: float
    certificate_map: dict[str, float]
    world_bundle: ProbabilisticWorldBundle
    audit_bundle: AuditWorldBundle
    support_state: WorldSupportState
    winner_confidence: WinnerConfidenceState
    pairwise_gap_states: dict[str, PairwiseGapState]
    flip_radius: FlipRadiusState | None
    decision_region: DecisionRegionState
    certified_set: CertifiedSetState
    certified: bool
    certification_basis: str
    manifest_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["world_bundle"] = self.world_bundle.as_dict()
        payload["audit_bundle"] = self.audit_bundle.as_dict()
        payload["support_state"] = self.support_state.as_dict()
        payload["winner_confidence"] = self.winner_confidence.as_dict()
        payload["pairwise_gap_states"] = {
            route_id: state.as_dict()
            for route_id, state in self.pairwise_gap_states.items()
        }
        payload["flip_radius"] = self.flip_radius.as_dict() if self.flip_radius is not None else None
        payload["decision_region"] = self.decision_region.as_dict()
        payload["certified_set"] = self.certified_set.as_dict()
        return payload

    @classmethod
    def from_refc_outputs(
        cls,
        *,
        certificate: Mapping[str, float],
        threshold: float,
        world_manifest: Mapping[str, Any] | None = None,
        fragility: Mapping[str, Any] | None = None,
        evidence_snapshot_manifest: Mapping[str, Any] | None = None,
        ambiguity_context: Mapping[str, Any] | None = None,
        evidence_validation: Mapping[str, Any] | None = None,
    ) -> "CertificationState":
        certificate_map = {
            str(route_id): float(value)
            for route_id, value in certificate.items()
            if str(route_id).strip()
        }
        winner_id = min(
            certificate_map.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )[0]
        world_bundle = ProbabilisticWorldBundle.from_manifest(
            world_manifest,
            route_ids=tuple(certificate_map.keys()),
        )
        audit_bundle = AuditWorldBundle.from_sources(world_bundle, evidence_snapshot_manifest)
        support_state = WorldSupportState.from_inputs(
            ambiguity_context=ambiguity_context,
            evidence_validation=evidence_validation,
            audit_bundle=audit_bundle,
            probabilistic_bundle=world_bundle,
        )
        sample_count = max(1, world_bundle.effective_world_count or world_bundle.world_count or 1)
        winner_confidence = WinnerConfidenceState.from_point_estimate(
            winner_id,
            point_estimate=certificate_map.get(winner_id, 0.0),
            sample_count=sample_count,
            threshold=threshold,
            support_strength=support_state.support_strength,
            proxy_fraction=world_bundle.proxy_world_fraction,
        )
        challenger_confidences: list[WinnerConfidenceState] = []
        pairwise_gap_states: dict[str, PairwiseGapState] = {}
        for route_id, point_estimate in sorted(certificate_map.items()):
            if route_id == winner_id:
                continue
            challenger_state = WinnerConfidenceState.from_point_estimate(
                route_id,
                point_estimate=point_estimate,
                sample_count=sample_count,
                threshold=threshold,
                support_strength=support_state.support_strength,
                proxy_fraction=world_bundle.proxy_world_fraction,
            )
            challenger_confidences.append(challenger_state)
            pairwise_gap_states[route_id] = PairwiseGapState.from_certificate_gap(
                winner_id,
                route_id,
                winner_certificate=certificate_map.get(winner_id, 0.0),
                challenger_certificate=point_estimate,
                sample_count=sample_count,
                support_strength=support_state.support_strength,
                proxy_fraction=world_bundle.proxy_world_fraction,
            )
        best_pairwise = None
        if pairwise_gap_states:
            best_pairwise = min(
                pairwise_gap_states.values(),
                key=lambda state: (state.lower_gap, state.challenger_id),
            )
        flip_radius = (
            FlipRadiusState.from_pairwise_gap(
                best_pairwise,
                objective_scale=max(1.0 / float(sample_count), abs(best_pairwise.mean_gap), 1e-6),
            )
            if best_pairwise is not None
            else None
        )
        decision_region = DecisionRegionState.from_states(
            winner_confidence,
            challenger_confidences,
            list(pairwise_gap_states.values()),
            flip_radius,
            threshold=threshold,
        )
        confidence_states = [winner_confidence] + challenger_confidences
        certified_set = CertifiedSetState.from_confidence_states(
            confidence_states,
            threshold=threshold,
            winner_id=winner_id,
            decision_region=decision_region,
        )
        certified = decision_region.certified and (winner_id in certified_set.certified_route_ids)
        certification_basis = "threshold_and_pairwise" if certified else decision_region.status
        if fragility and isinstance(fragility, Mapping) and fragility.get("controller_ranking_basis"):
            certification_basis = f"{certification_basis}:{fragility.get('controller_ranking_basis')}"
        return cls(
            winner_id=winner_id,
            threshold=round(float(threshold), 6),
            certificate_map={route_id: round(value, 6) for route_id, value in certificate_map.items()},
            world_bundle=world_bundle,
            audit_bundle=audit_bundle,
            support_state=support_state,
            winner_confidence=winner_confidence,
            pairwise_gap_states=pairwise_gap_states,
            flip_radius=flip_radius,
            decision_region=decision_region,
            certified_set=certified_set,
            certified=certified,
            certification_basis=certification_basis,
            manifest_hash=world_bundle.manifest_hash,
        )
