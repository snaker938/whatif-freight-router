"""Shared REFC scaffold types for certification-native state and payloads.

These wrappers intentionally mirror evidence outputs without changing runtime
selection or stop semantics. They are kept JSON-friendly so the backend can
serialize them directly once wiring is added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
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
