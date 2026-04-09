from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .audit_correction import AuditWorldBundle
from .fidelity_model import ProbabilisticWorldBundle
from .world_policies import clamp01


def _float_or_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed != parsed:
        return float(default)
    return float(parsed)


@dataclass(frozen=True)
class WorldSupportState:
    support_strength: float
    source_support_strength: float
    ambiguity_support_ratio: float
    source_entropy: float
    source_count: int
    source_mix_count: int
    family_density: float
    margin_pressure: float
    provenance_coverage: float
    live_family_count: int
    snapshot_family_count: int
    model_family_count: int
    proxy_penalty: float
    audit_correction: float
    recommended_fidelity: str
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_support_sufficient(self, *, minimum: float = 0.6) -> bool:
        return self.support_strength >= float(minimum)

    @classmethod
    def from_inputs(
        cls,
        *,
        ambiguity_context: Mapping[str, Any] | None = None,
        evidence_validation: Mapping[str, Any] | None = None,
        audit_bundle: AuditWorldBundle | None = None,
        probabilistic_bundle: ProbabilisticWorldBundle | None = None,
    ) -> "WorldSupportState":
        context = dict(ambiguity_context) if isinstance(ambiguity_context, Mapping) else {}
        validation = dict(evidence_validation) if isinstance(evidence_validation, Mapping) else {}
        support_ratio = clamp01(context.get("od_ambiguity_support_ratio"))
        confidence = clamp01(context.get("od_ambiguity_confidence"))
        source_entropy = clamp01(context.get("od_ambiguity_source_entropy"))
        source_count = max(0, int(_float_or_default(context.get("od_ambiguity_source_count"), 0.0)))
        source_mix_count = max(0, int(_float_or_default(context.get("od_ambiguity_source_mix_count"), 0.0)))
        family_density = clamp01(context.get("od_ambiguity_family_density"))
        margin_pressure = clamp01(
            max(
                clamp01(context.get("od_ambiguity_margin_pressure")),
                1.0 - clamp01(context.get("od_nominal_margin_proxy"), 1.0),
            )
        )
        provenance_coverage = clamp01(validation.get("freshness_coverage"), 1.0)
        live_family_count = max(0, int(_float_or_default(validation.get("live_family_count"), 0.0)))
        snapshot_family_count = max(0, int(_float_or_default(validation.get("snapshot_family_count"), 0.0)))
        model_family_count = max(0, int(_float_or_default(validation.get("model_family_count"), 0.0)))
        explicit_source_support = context.get("od_ambiguity_source_support_strength")
        if explicit_source_support is None:
            source_support_strength = clamp01(
                (0.35 * support_ratio)
                + (0.25 * confidence)
                + (0.15 * min(1.0, source_count / 4.0))
                + (0.15 * min(1.0, source_mix_count / 3.0))
                + (0.10 * source_entropy),
            )
        else:
            source_support_strength = clamp01(explicit_source_support)
        proxy_penalty = 0.0
        if audit_bundle is not None:
            proxy_penalty = float(audit_bundle.correction_penalty)
        elif probabilistic_bundle is not None:
            proxy_penalty = float(probabilistic_bundle.proxy_world_fraction)
        audit_correction = float(audit_bundle.correction_scale) if audit_bundle is not None else 1.0
        live_total = live_family_count + snapshot_family_count + model_family_count
        live_share = (live_family_count / float(live_total)) if live_total > 0 else 0.0
        bundle_support = probabilistic_bundle.effective_support_mass if probabilistic_bundle is not None else 1.0
        support_strength = clamp01(
            (0.24 * source_support_strength)
            + (0.18 * support_ratio)
            + (0.12 * source_entropy)
            + (0.12 * provenance_coverage)
            + (0.10 * live_share)
            + (0.10 * bundle_support)
            + (0.08 * family_density)
            + (0.08 * margin_pressure)
            + (0.08 * audit_correction)
            - (0.20 * proxy_penalty),
        )
        if support_strength >= 0.75 and proxy_penalty <= 0.15:
            recommended_fidelity = "probabilistic"
        elif support_strength >= 0.45 and proxy_penalty <= 0.40:
            recommended_fidelity = "probabilistic_audit"
        else:
            recommended_fidelity = "audit_first"
        notes: list[str] = []
        if proxy_penalty > 0.0:
            notes.append("proxy_penalty")
        if provenance_coverage < 1.0:
            notes.append("partial_freshness")
        if snapshot_family_count > 0:
            notes.append("snapshot_evidence_present")
        return cls(
            support_strength=round(support_strength, 6),
            source_support_strength=round(source_support_strength, 6),
            ambiguity_support_ratio=round(support_ratio, 6),
            source_entropy=round(source_entropy, 6),
            source_count=source_count,
            source_mix_count=source_mix_count,
            family_density=round(family_density, 6),
            margin_pressure=round(margin_pressure, 6),
            provenance_coverage=round(provenance_coverage, 6),
            live_family_count=live_family_count,
            snapshot_family_count=snapshot_family_count,
            model_family_count=model_family_count,
            proxy_penalty=round(clamp01(proxy_penalty), 6),
            audit_correction=round(clamp01(audit_correction, 1.0), 6),
            recommended_fidelity=recommended_fidelity,
            notes=tuple(notes),
        )
