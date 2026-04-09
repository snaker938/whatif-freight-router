from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed != parsed:
        return float(default)
    return float(parsed)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _as_float(value, default)))


@dataclass(frozen=True)
class WorldSupportState:
    schema_version: str = "world-support-v1"
    support_flag: bool = False
    support_score: float = 0.0
    support_ratio: float = 0.0
    support_bin: str = "unspecified"
    calibration_bin: str = "unspecified"
    support_source: str = "unknown"
    out_of_support_reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbabilisticWorldBundle:
    schema_version: str = "probabilistic-world-bundle-v1"
    bundle_id: str = ""
    world_count: int = 0
    worlds: list[dict[str, Any]] = field(default_factory=list)
    support_state: WorldSupportState = field(default_factory=WorldSupportState)
    cache_mode: str = "cold"
    policy_name: str = "unspecified"
    policy_version: str = "v1"
    policy_hash: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditWorldBundle:
    schema_version: str = "audit-world-bundle-v1"
    bundle_id: str = ""
    audit_world_count: int = 0
    audit_worlds: list[dict[str, Any]] = field(default_factory=list)
    support_state: WorldSupportState = field(default_factory=WorldSupportState)
    cache_mode: str = "cold"
    policy_name: str = "unspecified"
    policy_version: str = "v1"
    policy_hash: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_world_support_state(
    *,
    support_score: float = 0.0,
    support_ratio: float = 0.0,
    support_bin: str = "unspecified",
    calibration_bin: str = "unspecified",
    support_source: str = "unknown",
    out_of_support_reason: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    support_threshold: float = 0.5,
) -> WorldSupportState:
    score = _clamp01(support_score)
    ratio = _clamp01(support_ratio)
    support_flag = score >= _clamp01(support_threshold) and out_of_support_reason is None
    return WorldSupportState(
        support_flag=support_flag,
        support_score=score,
        support_ratio=ratio,
        support_bin=str(support_bin or "unspecified"),
        calibration_bin=str(calibration_bin or "unspecified"),
        support_source=str(support_source or "unknown"),
        out_of_support_reason=out_of_support_reason,
        provenance=dict(provenance or {}),
    )


def build_probabilistic_world_bundle(
    *,
    bundle_id: str,
    worlds: list[dict[str, Any]] | None = None,
    support_state: WorldSupportState | None = None,
    cache_mode: str = "cold",
    policy_name: str = "unspecified",
    policy_version: str = "v1",
    policy_hash: str = "",
    provenance: Mapping[str, Any] | None = None,
) -> ProbabilisticWorldBundle:
    world_rows = list(worlds or [])
    return ProbabilisticWorldBundle(
        bundle_id=str(bundle_id or ""),
        world_count=len(world_rows),
        worlds=world_rows,
        support_state=support_state or WorldSupportState(),
        cache_mode=str(cache_mode or "cold"),
        policy_name=str(policy_name or "unspecified"),
        policy_version=str(policy_version or "v1"),
        policy_hash=str(policy_hash or ""),
        provenance=dict(provenance or {}),
    )


def build_audit_world_bundle(
    *,
    bundle_id: str,
    audit_worlds: list[dict[str, Any]] | None = None,
    support_state: WorldSupportState | None = None,
    cache_mode: str = "cold",
    policy_name: str = "unspecified",
    policy_version: str = "v1",
    policy_hash: str = "",
    provenance: Mapping[str, Any] | None = None,
) -> AuditWorldBundle:
    world_rows = list(audit_worlds or [])
    return AuditWorldBundle(
        bundle_id=str(bundle_id or ""),
        audit_world_count=len(world_rows),
        audit_worlds=world_rows,
        support_state=support_state or WorldSupportState(),
        cache_mode=str(cache_mode or "cold"),
        policy_name=str(policy_name or "unspecified"),
        policy_version=str(policy_version or "v1"),
        policy_hash=str(policy_hash or ""),
        provenance=dict(provenance or {}),
    )
