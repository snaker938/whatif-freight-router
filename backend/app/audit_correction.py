"""Pipeline stage: describe leakage-safe audit-correction outputs for support-aware evidence calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .support_model import (
    MultiFidelitySummary,
    WorldSupportState,
    build_multi_fidelity_summary,
)


_REQUIRED_REGIME_FEATURES: tuple[str, ...] = (
    "corridor_family",
    "ambiguity_regime",
    "support_regime",
    "evidence_family_regime",
    "engine_disagreement_regime",
    "candidate_density_or_pressure",
)

_FEATURE_ALIASES: dict[str, str] = {
    "corridor_family": "corridor_family",
    "corridor": "corridor_family",
    "ambiguity_regime": "ambiguity_regime",
    "ambiguity_band": "ambiguity_regime",
    "support_regime": "support_regime",
    "support_bin": "support_regime",
    "evidence_family_regime": "evidence_family_regime",
    "evidence_family": "evidence_family_regime",
    "engine_disagreement_regime": "engine_disagreement_regime",
    "engine_disagreement": "engine_disagreement_regime",
    "candidate_density_or_pressure": "candidate_density_or_pressure",
    "candidate_density": "candidate_density_or_pressure",
    "candidate_pressure": "candidate_density_or_pressure",
    "competitor_pressure": "candidate_density_or_pressure",
}


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


def _normalize_feature_names(feature_names: Sequence[str] | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for feature_name in list(feature_names or []):
        key = str(feature_name or "").strip().lower()
        if not key:
            continue
        canonical = _FEATURE_ALIASES.get(key, key)
        if canonical not in seen:
            ordered.append(canonical)
            seen.add(canonical)
    for required_name in _REQUIRED_REGIME_FEATURES:
        if required_name not in seen:
            ordered.append(required_name)
            seen.add(required_name)
    return ordered


def _leakage_safe_training(metadata: LeakageSafeCorrectionMetadata | AuditPropensityMetadata) -> bool:
    return bool(
        metadata.cross_fitted
        and metadata.out_of_fold_only
        and metadata.same_row_fit_prohibited
    )


def _pairwise_evaluation_tag(*, correction_applied: bool, audit_probability: float, propensity_score: float) -> str:
    if correction_applied:
        return "corrected_from_residual_model"
    if _clamp01(audit_probability) > 0.0 or _clamp01(propensity_score) > 0.0:
        return "fully_audited"
    return "proxy_only"


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
    pairwise_evaluation_tag: str = "proxy_only"
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
        feature_names=_normalize_feature_names(feature_names),
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
        feature_names=_normalize_feature_names(feature_names),
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
    resolved_correction_applied = abs(residual) > 0.0
    resolved_audit_probability = _clamp01(audit_probability)
    resolved_propensity_score = _clamp01(propensity_score)
    return ProxyAuditRecord(
        row_id=str(row_id or ""),
        route_id=str(route_id or ""),
        evidence_family=str(evidence_family or "unspecified"),
        proxy_value=proxy,
        audited_value=audited,
        residual_bias=residual,
        absolute_residual=abs(residual),
        correction_factor=correction_factor,
        audit_probability=resolved_audit_probability,
        propensity_score=resolved_propensity_score,
        correction_applied=resolved_correction_applied,
        pairwise_evaluation_tag=_pairwise_evaluation_tag(
            correction_applied=resolved_correction_applied,
            audit_probability=resolved_audit_probability,
            propensity_score=resolved_propensity_score,
        ),
        support_state=support_state or WorldSupportState(),
        correction_metadata=correction_metadata or LeakageSafeCorrectionMetadata(),
        propensity_metadata=propensity_metadata or AuditPropensityMetadata(),
        provenance=dict(provenance or {}),
    )


def summarize_proxy_audit_records(
    records: Sequence[ProxyAuditRecord],
    *,
    proxy_world_count: int = 0,
    audit_world_count: int | None = None,
    support_state: WorldSupportState | None = None,
    candidate_route_pair_count: int | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> MultiFidelitySummary:
    record_list = list(records)
    correction_active = any(record.correction_applied for record in record_list)
    correction_mass = sum(abs(float(record.residual_bias)) for record in record_list)
    propensity_scores = [float(record.propensity_score) for record in record_list]
    first_correction_meta = (
        record_list[0].correction_metadata if record_list else LeakageSafeCorrectionMetadata()
    )
    first_propensity_meta = (
        record_list[0].propensity_metadata if record_list else AuditPropensityMetadata()
    )
    correction_training_leakage_safe = bool(record_list) and _leakage_safe_training(first_correction_meta)
    propensity_training_leakage_safe = bool(record_list) and _leakage_safe_training(first_propensity_meta)
    resolved_support_state = (
        support_state
        or (record_list[0].support_state if record_list else WorldSupportState())
    )
    has_propensity_path = bool(record_list) and (
        bool(first_propensity_meta.model_version.strip())
        or any(score > 0.0 for score in propensity_scores)
    )
    if correction_active and has_propensity_path:
        correction_path_estimator = "doubly_robust_residual_correction"
    elif has_propensity_path:
        correction_path_estimator = "propensity_aware_residual_correction"
    elif correction_active:
        correction_path_estimator = "residual_correction_only"
    else:
        correction_path_estimator = "proxy_only"
    return build_multi_fidelity_summary(
        proxy_world_count=max(0, int(proxy_world_count)),
        audit_world_count=(
            max(0, int(audit_world_count))
            if audit_world_count is not None
            else len(record_list)
        ),
        support_state=resolved_support_state,
        proxy_bias_model_version=first_correction_meta.model_version,
        audit_propensity_version=first_propensity_meta.model_version,
        proxy_correction_active=correction_active,
        audit_correction_mass=correction_mass,
        correction_conditioning_features=first_correction_meta.feature_names,
        propensity_conditioning_features=first_propensity_meta.feature_names,
        correction_training_leakage_safe=correction_training_leakage_safe,
        propensity_training_leakage_safe=propensity_training_leakage_safe,
        correction_path_estimator=correction_path_estimator,
        audited_route_pair_count=len(record_list),
        candidate_route_pair_count=(
            candidate_route_pair_count
            if candidate_route_pair_count is not None
            else max(len(record_list), int(proxy_world_count) + len(record_list))
        ),
        propensity_scores=propensity_scores,
        provenance=dict(provenance or {"source": "proxy_audit_records"}),
    )
