"""Shared REFC scaffold types for certification-native state and payloads.

These wrappers intentionally mirror evidence outputs without changing runtime
selection or stop semantics. They are kept JSON-friendly so the backend can
serialize them directly once wiring is added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import sys
from typing import Any, Mapping, Sequence


class SerializableRecord:
    """Mixin for small JSON-serializable dataclass wrappers."""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Sequence[Any] | None) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


@dataclass(frozen=True)
class WorldSupportState(SerializableRecord):
    support_flag: bool = True
    support_status: str = "unknown"
    support_reason: str | None = None
    support_bin: str | None = None
    out_of_support_reason: str | None = None
    coverage_ratio: float | None = None
    confidence: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbabilisticWorldBundle(SerializableRecord):
    world_count: int = 0
    unique_world_count: int = 0
    active_families: list[str] = field(default_factory=list)
    state_catalog: list[str] = field(default_factory=list)
    state_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    worlds: list[dict[str, Any]] = field(default_factory=list)
    world_reuse_rate: float = 0.0
    world_reuse_rate_within_manifest: float = 0.0
    world_reuse_rate_cross_request: float = 0.0
    certification_cache_reuse_origin: str = "miss"
    certification_cache_reuse_applied: bool = False
    manifest_hash: str | None = None
    support_state: WorldSupportState | None = None

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "ProbabilisticWorldBundle":
        payload = _mapping(manifest)
        support_payload = payload.get("support_state")
        support_state = (
            WorldSupportState(**_mapping(support_payload))
            if isinstance(support_payload, Mapping)
            else None
        )
        return cls(
            world_count=int(payload.get("world_count", 0) or 0),
            unique_world_count=int(payload.get("unique_world_count", 0) or 0),
            active_families=[str(family) for family in _sequence(payload.get("active_families"))],
            state_catalog=[str(state) for state in _sequence(payload.get("state_catalog"))],
            state_weights=_mapping(payload.get("state_weights")),
            worlds=[dict(world) for world in _sequence(payload.get("worlds")) if isinstance(world, Mapping)],
            world_reuse_rate=float(payload.get("world_reuse_rate", 0.0) or 0.0),
            world_reuse_rate_within_manifest=float(
                payload.get("world_reuse_rate_within_manifest", payload.get("world_reuse_rate", 0.0)) or 0.0
            ),
            world_reuse_rate_cross_request=float(payload.get("world_reuse_rate_cross_request", 0.0) or 0.0),
            certification_cache_reuse_origin=str(payload.get("certification_cache_reuse_origin", "miss")),
            certification_cache_reuse_applied=bool(payload.get("certification_cache_reuse_applied", False)),
            manifest_hash=str(payload.get("manifest_hash")) if payload.get("manifest_hash") is not None else None,
            support_state=support_state,
        )


@dataclass(frozen=True)
class AuditWorldBundle(SerializableRecord):
    audit_world_count: int = 0
    audited_route_pair_count: int = 0
    partially_audited_world_count: int = 0
    fully_audited_world_count: int = 0
    reused_world_count: int = 0
    corrected_world_count: int = 0
    support_condition: str = "unknown"
    calibration_version: str | None = None
    propensity_version: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProxyAuditRecord(SerializableRecord):
    route_id: str
    family: str
    proxy_value: float = 0.0
    audit_value: float = 0.0
    residual_bias: float = 0.0
    correction_active: bool = False
    propensity: float | None = None
    support_flag: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionValueEstimate(SerializableRecord):
    action_id: str
    action_family: str
    predicted_gain: float = 0.0
    realized_gain: float = 0.0
    cost: float = 0.0
    confidence: float | None = None
    rationale: str | None = None
    support_flag: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AbstentionRecord(SerializableRecord):
    abstention_type: str
    reason: str
    support_flag: bool = False
    reason_code: str | None = None
    certificate_gap: float | None = None
    budget_remaining: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CertificationDecisionRegion(SerializableRecord):
    route_id: str
    certified: bool = False
    best_challenger_id: str | None = None
    active_challenger_id: str | None = None
    nearest_certificate_boundary: str | None = None


@dataclass(frozen=True)
class CertificationWorldFidelitySummary(SerializableRecord):
    world_bundle_id: str | None = None
    multi_fidelity_mode: str | None = None
    policy: str | None = None
    world_count: int | None = None
    unique_world_count: int | None = None
    requested_world_count: int | None = None
    effective_world_count: int | None = None
    route_ids: list[str] = field(default_factory=list)
    active_families: list[str] = field(default_factory=list)
    stress_world_fraction: float | None = None
    world_reuse_rate: float | None = None
    manifest_hash: str | None = None
    notes: list[str] = field(default_factory=list)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decision_fidelity_summary_from_state(state: Any) -> Any | None:
    summary = getattr(state, "world_fidelity_summary", None)
    if summary is not None:
        return summary
    manifest = getattr(state, "world_manifest", None)
    if not isinstance(manifest, Mapping):
        return None
    return CertificationWorldFidelitySummary.from_manifest(
        manifest,
        route_ids=list(getattr(state, "top_route_ids", []) or []),
    )


def _install_main_compat_helpers() -> None:
    main_module = sys.modules.get("app.main")
    if main_module is not None and not hasattr(main_module, "_decision_fidelity_summary_from_state"):
        setattr(main_module, "_decision_fidelity_summary_from_state", _decision_fidelity_summary_from_state)


def _install_certificate_witness_compat() -> None:
    try:
        from .certificate_witness import CertificateWitness
    except Exception:
        return

    if not hasattr(CertificateWitness, "from_state"):

        @classmethod
        def from_state(cls, state: Any, *, fragility: Mapping[str, Any] | None = None) -> Any:
            selected_id = str(
                getattr(state, "selected_route_id", None)
                or getattr(state, "winner_id", None)
                or ""
            )
            decision_region = getattr(state, "decision_region", None)
            best_challenger_id = (
                getattr(decision_region, "best_challenger_id", None)
                or getattr(decision_region, "active_challenger_id", None)
            )
            fragility_payload = _mapping(fragility)
            route_fragility_map = _mapping(fragility_payload.get("route_fragility_map"))
            selected_fragility = _mapping(route_fragility_map.get(selected_id))
            active_families = [
                str(family)
                for family, value in sorted(
                    selected_fragility.items(),
                    key=lambda item: (-float(item[1] or 0.0), str(item[0])),
                )
                if _coerce_float_or_none(value) is not None and float(value) > 0.0
            ]
            active_challengers = [str(best_challenger_id)] if best_challenger_id else []
            witness_size = len(active_challengers) + len(active_families[:2])
            return cls(
                route_id=selected_id,
                active_challenger_ids=active_challengers,
                active_evidence_families=active_families[:2],
                action_steps=[],
                witness_size=witness_size,
                selected_certificate_basis=getattr(state, "selected_certificate_basis", None),
                nearest_certificate_boundary=getattr(
                    decision_region, "nearest_certificate_boundary", None
                ),
                active_challenger_count=len(active_challengers),
                active_evidence_family_count=len(active_families[:2]),
                support_flag=True,
                provenance={"recommended_action": "hold"},
            )

        setattr(CertificateWitness, "from_state", from_state)

    if not hasattr(CertificateWitness, "best_challenger_id"):
        setattr(
            CertificateWitness,
            "best_challenger_id",
            property(
                lambda self: (
                    self.active_challenger_ids[0]
                    if getattr(self, "active_challenger_ids", None)
                    else getattr(self, "targeted_challenger_route_id", None)
                )
            ),
        )
    if not hasattr(CertificateWitness, "top_fragility_family"):
        setattr(
            CertificateWitness,
            "top_fragility_family",
            property(
                lambda self: (
                    self.active_evidence_families[0]
                    if getattr(self, "active_evidence_families", None)
                    else None
                )
            ),
        )
    if not hasattr(CertificateWitness, "recommended_action"):
        setattr(
            CertificateWitness,
            "recommended_action",
            property(
                lambda self: _mapping(getattr(self, "provenance", None)).get(
                    "recommended_action", "hold"
                )
            ),
        )
    if not hasattr(CertificateWitness, "is_consistent_with_state"):

        def is_consistent_with_state(self: Any, state: Any) -> bool:
            decision_region = getattr(state, "decision_region", None)
            expected_challenger = (
                getattr(decision_region, "best_challenger_id", None)
                or getattr(decision_region, "active_challenger_id", None)
            )
            return bool(
                str(getattr(self, "route_id", "")) == str(getattr(state, "selected_route_id", ""))
                and (
                    expected_challenger is None
                    or getattr(self, "best_challenger_id", None) == str(expected_challenger)
                )
            )

        setattr(CertificateWitness, "is_consistent_with_state", is_consistent_with_state)


@classmethod
def _world_fidelity_from_manifest(
    cls,
    manifest: Mapping[str, Any],
    *,
    route_ids: Sequence[str] | None = None,
) -> "CertificationWorldFidelitySummary":
    payload = _mapping(manifest)
    world_count = _coerce_int(payload.get("world_count"), 0)
    unique_world_count = _coerce_int(payload.get("unique_world_count"), world_count)
    requested_world_count = _coerce_int(payload.get("requested_world_count"), world_count)
    effective_world_count = _coerce_int(
        payload.get("effective_world_count"),
        unique_world_count,
    )
    return cls(
        world_bundle_id=(
            str(payload.get("world_bundle_id"))
            if payload.get("world_bundle_id") is not None
            else None
        ),
        multi_fidelity_mode=(
            str(payload.get("multi_fidelity_mode"))
            if payload.get("multi_fidelity_mode") is not None
            else None
        ),
        policy=(
            str(payload.get("world_count_policy"))
            if payload.get("world_count_policy") is not None
            else None
        ),
        world_count=world_count,
        unique_world_count=unique_world_count,
        requested_world_count=requested_world_count,
        effective_world_count=effective_world_count,
        route_ids=[str(route_id) for route_id in _sequence(route_ids)],
        active_families=[str(family) for family in _sequence(payload.get("active_families"))],
        stress_world_fraction=_coerce_float_or_none(payload.get("stress_world_fraction")),
        world_reuse_rate=_coerce_float_or_none(payload.get("world_reuse_rate")),
        manifest_hash=(
            str(payload.get("manifest_hash")) if payload.get("manifest_hash") is not None else None
        ),
    )


CertificationWorldFidelitySummary.from_manifest = _world_fidelity_from_manifest  # type: ignore[attr-defined]


@dataclass(frozen=True)
class CertificationState(SerializableRecord):
    winner_id: str
    selected_route_id: str
    certificate: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0
    certified: bool = False
    selector_config: dict[str, Any] = field(default_factory=dict)
    world_manifest: dict[str, Any] = field(default_factory=dict)
    support_state: WorldSupportState | None = None
    selected_certificate_basis: str = "empirical"
    top_route_ids: list[str] = field(default_factory=list)
    witness: dict[str, Any] = field(default_factory=dict)
    decision_region: CertificationDecisionRegion | None = None
    world_fidelity_summary: CertificationWorldFidelitySummary | None = None

    @classmethod
    def from_refc_outputs(
        cls,
        *,
        certificate: Mapping[str, Any],
        threshold: float,
        world_manifest: Mapping[str, Any] | None = None,
        fragility: Mapping[str, Any] | None = None,
        evidence_snapshot_manifest: Mapping[str, Any] | None = None,
        ambiguity_context: Mapping[str, Any] | None = None,
        evidence_validation: Mapping[str, Any] | None = None,
        selector_config: Mapping[str, Any] | None = None,
        selected_certificate_basis: str = "empirical",
        **extra: Any,
    ) -> "CertificationState":
        _install_main_compat_helpers()
        _install_certificate_witness_compat()

        certificate_map = {
            str(route_id): float(value)
            for route_id, value in _mapping(certificate).items()
            if str(route_id).strip()
        }
        sorted_route_ids = [
            route_id
            for route_id, _value in sorted(
                certificate_map.items(),
                key=lambda item: (-float(item[1]), str(item[0])),
            )
        ]
        manifest_payload = _mapping(world_manifest)
        winner_id = sorted_route_ids[0] if sorted_route_ids else ""
        selected_route_id = str(manifest_payload.get("selected_route_id") or winner_id)
        winner_score = float(certificate_map.get(winner_id, 0.0))
        threshold_value = float(threshold)
        certified = bool(winner_score >= threshold_value)
        best_challenger_id = next(
            (route_id for route_id in sorted_route_ids if route_id != selected_route_id),
            None,
        )
        decision_region = CertificationDecisionRegion(
            route_id=selected_route_id,
            certified=certified,
            best_challenger_id=best_challenger_id,
            active_challenger_id=best_challenger_id,
            nearest_certificate_boundary=(
                f"{selected_route_id}:{best_challenger_id}" if best_challenger_id else None
            ),
        )
        fidelity_summary = CertificationWorldFidelitySummary.from_manifest(
            manifest_payload,
            route_ids=sorted_route_ids,
        )
        support_state = WorldSupportState(
            support_flag=True,
            support_status="in_support",
            coverage_ratio=_coerce_float_or_none(
                _mapping(evidence_validation).get("freshness_coverage")
            ),
            confidence=_coerce_float_or_none(
                _mapping(ambiguity_context).get("od_ambiguity_confidence")
            ),
            provenance={
                "fragility": _mapping(fragility),
                "evidence_snapshot_manifest": _mapping(evidence_snapshot_manifest),
                "ambiguity_context": _mapping(ambiguity_context),
                "evidence_validation": _mapping(evidence_validation),
            },
        )
        return cls(
            winner_id=winner_id,
            selected_route_id=selected_route_id,
            certificate=certificate_map,
            threshold=threshold_value,
            certified=certified,
            selector_config={**_mapping(selector_config), **_mapping(extra.get("selector_config"))},
            world_manifest=manifest_payload,
            support_state=support_state,
            selected_certificate_basis=str(selected_certificate_basis),
            top_route_ids=sorted_route_ids,
            witness={},
            decision_region=decision_region,
            world_fidelity_summary=fidelity_summary,
        )


_install_main_compat_helpers()
_install_certificate_witness_compat()



@dataclass(frozen=True)
class DecisionPackage(SerializableRecord):
    recommended_route: dict[str, Any] | None = None
    certified_set: list[dict[str, Any]] = field(default_factory=list)
    frontier_summary: dict[str, Any] = field(default_factory=dict)
    certificate_summary: dict[str, Any] = field(default_factory=dict)
    stability_summary: dict[str, Any] = field(default_factory=dict)
    preference_summary: dict[str, Any] = field(default_factory=dict)
    support_summary: dict[str, Any] = field(default_factory=dict)
    abstention_summary: dict[str, Any] = field(default_factory=dict)
    action_trace_summary: dict[str, Any] = field(default_factory=dict)
    witness_summary: dict[str, Any] = field(default_factory=dict)
    artifact_pointers: dict[str, str] = field(default_factory=dict)
    selected_certificate_basis: str | None = None
    terminal_type: str = "singleton"
    certification_state: CertificationState | None = None
    abstention_record: AbstentionRecord | None = None
