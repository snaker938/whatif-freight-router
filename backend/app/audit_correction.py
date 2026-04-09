from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .support_model import WorldSupportState


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
class LeakageSafeCorrectionMetadata:
    schema_version: str = "leakage-safe-correction-metadata-v1"
    model_name: str = "conservative_bias_correction"
    model_version: str = "v1"
    policy_hash: str = ""
    cross_fitted: bool = True
    out_of_fold_only: bool = True
    same_row_fit_prohibited: bool = True
    fold_count: int = 0
    training_rows: int = 0
    validation_rows: int = 0
    feature_names: list[str] = field(default_factory=list)
    training_scope: str = "unspecified"
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditPropensityMetadata:
    schema_version: str = "audit-propensity-metadata-v1"
    model_name: str = "conservative_audit_propensity"
    model_version: str = "v1"
    policy_hash: str = ""
    cross_fitted: bool = True
    out_of_fold_only: bool = True
    same_row_fit_prohibited: bool = True
    fold_count: int = 0
    training_rows: int = 0
    validation_rows: int = 0
    feature_names: list[str] = field(default_factory=list)
    training_scope: str = "unspecified"
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProxyAuditRecord:
    schema_version: str = "proxy-audit-record-v1"
    row_id: str = ""
    route_id: str = ""
    evidence_family: str = "unspecified"
    proxy_value: float = 0.0
    audited_value: float = 0.0
    residual_bias: float = 0.0
    absolute_residual: float = 0.0
    correction_factor: float = 1.0
    audit_probability: float = 0.0
    propensity_score: float = 0.0
    correction_applied: bool = False
    support_state: WorldSupportState = field(default_factory=WorldSupportState)
    correction_metadata: LeakageSafeCorrectionMetadata = field(
        default_factory=LeakageSafeCorrectionMetadata
    )
    propensity_metadata: AuditPropensityMetadata = field(default_factory=AuditPropensityMetadata)
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_leakage_safe_correction_metadata(
    *,
    model_version: str = "v1",
    policy_hash: str = "",
    fold_count: int = 0,
    training_rows: int = 0,
    validation_rows: int = 0,
    feature_names: list[str] | None = None,
    training_scope: str = "unspecified",
    cross_fitted: bool = True,
    out_of_fold_only: bool = True,
    same_row_fit_prohibited: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> LeakageSafeCorrectionMetadata:
    return LeakageSafeCorrectionMetadata(
        model_version=str(model_version or "v1"),
        policy_hash=str(policy_hash or ""),
        cross_fitted=bool(cross_fitted),
        out_of_fold_only=bool(out_of_fold_only),
        same_row_fit_prohibited=bool(same_row_fit_prohibited),
        fold_count=max(0, int(fold_count)),
        training_rows=max(0, int(training_rows)),
        validation_rows=max(0, int(validation_rows)),
        feature_names=list(feature_names or []),
        training_scope=str(training_scope or "unspecified"),
        provenance=dict(provenance or {}),
    )


def build_audit_propensity_metadata(
    *,
    model_version: str = "v1",
    policy_hash: str = "",
    fold_count: int = 0,
    training_rows: int = 0,
    validation_rows: int = 0,
    feature_names: list[str] | None = None,
    training_scope: str = "unspecified",
    cross_fitted: bool = True,
    out_of_fold_only: bool = True,
    same_row_fit_prohibited: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> AuditPropensityMetadata:
    return AuditPropensityMetadata(
        model_version=str(model_version or "v1"),
        policy_hash=str(policy_hash or ""),
        cross_fitted=bool(cross_fitted),
        out_of_fold_only=bool(out_of_fold_only),
        same_row_fit_prohibited=bool(same_row_fit_prohibited),
        fold_count=max(0, int(fold_count)),
        training_rows=max(0, int(training_rows)),
        validation_rows=max(0, int(validation_rows)),
        feature_names=list(feature_names or []),
        training_scope=str(training_scope or "unspecified"),
        provenance=dict(provenance or {}),
    )


def build_proxy_audit_record(
    *,
    row_id: str,
    route_id: str,
    evidence_family: str,
    proxy_value: float,
    audited_value: float,
    audit_probability: float = 0.0,
    propensity_score: float = 0.0,
    support_state: WorldSupportState | None = None,
    correction_metadata: LeakageSafeCorrectionMetadata | None = None,
    propensity_metadata: AuditPropensityMetadata | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ProxyAuditRecord:
    proxy = _as_float(proxy_value)
    audited = _as_float(audited_value)
    residual = audited - proxy
    correction_factor = audited / proxy if proxy not in (0.0, -0.0) else 1.0
    return ProxyAuditRecord(
        row_id=str(row_id or ""),
        route_id=str(route_id or ""),
        evidence_family=str(evidence_family or "unspecified"),
        proxy_value=proxy,
        audited_value=audited,
        residual_bias=residual,
        absolute_residual=abs(residual),
        correction_factor=correction_factor,
        audit_probability=_clamp01(audit_probability),
        propensity_score=_clamp01(propensity_score),
        correction_applied=abs(residual) > 0.0,
        support_state=support_state or WorldSupportState(),
        correction_metadata=correction_metadata or LeakageSafeCorrectionMetadata(),
        propensity_metadata=propensity_metadata or AuditPropensityMetadata(),
        provenance=dict(provenance or {}),
    )
