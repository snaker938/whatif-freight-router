"""Pipeline stage: model world-support state and probabilistic bundles for certification gating."""

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


def _value_from(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _normalize_feature_names(feature_names: list[str] | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for feature_name in list(feature_names or []):
        key = str(feature_name or "").strip()
        if not key or key in seen:
            continue
        ordered.append(key)
        seen.add(key)
    return ordered


def _has_feature(feature_names: list[str], feature_name: str) -> bool:
    return feature_name in set(_normalize_feature_names(feature_names))


def _normalize_support_bin(value: Any, *, support_flag: bool) -> str:
    normalized = str(value or "unspecified").strip() or "unspecified"
    if not support_flag and normalized.lower() in {
        "supported",
        "in_support",
        "mid_support",
        "strong_support",
    }:
        return "weak_support"
    return normalized


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


@dataclass(frozen=True)
class PositivityDiagnostics:
    schema_version: str = "positivity-diagnostics-v1"
    audited_route_pair_count: int = 0
    audit_coverage_ratio: float = 0.0
    minimum_propensity: float = 0.0
    mean_propensity: float = 0.0
    maximum_propensity: float = 0.0
    positivity_ok: bool = False
    weak_overlap_detected: bool = False
    support_bin: str = "unspecified"
    support_condition: str = "unknown"
    recommendation: str = "collect_more_audits"
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiFidelitySummary:
    schema_version: str = "multi-fidelity-summary-v1"
    proxy_world_count: int = 0
    audit_world_count: int = 0
    proxy_bias_model_version: str = ""
    audit_propensity_version: str = ""
    proxy_correction_active: bool = False
    multi_fidelity_certificate_basis: str = "proxy_only"
    certification_evaluation_tag: str = "proxy_only"
    correction_conditioning_features: list[str] = field(default_factory=list)
    propensity_conditioning_features: list[str] = field(default_factory=list)
    conditions_on_corridor_family: bool = False
    conditions_on_ambiguity_regime: bool = False
    conditions_on_support_regime: bool = False
    conditions_on_evidence_family_regime: bool = False
    conditions_on_engine_disagreement_regime: bool = False
    conditions_on_candidate_density_or_pressure: bool = False
    correction_training_leakage_safe: bool = False
    propensity_training_leakage_safe: bool = False
    leakage_safe_training: bool = False
    correction_path_estimator: str = "proxy_only"
    proxy_only_fraction: float = 1.0
    audit_correction_mass: float = 0.0
    positivity_diagnostics: PositivityDiagnostics = field(default_factory=PositivityDiagnostics)
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
        support_bin=_normalize_support_bin(support_bin, support_flag=support_flag),
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


def build_positivity_diagnostics(
    *,
    audited_route_pair_count: int,
    candidate_route_pair_count: int | None = None,
    propensity_scores: list[float] | None = None,
    support_state: Any | None = None,
    min_propensity: float = 0.05,
    min_audit_coverage_ratio: float = 0.1,
    provenance: Mapping[str, Any] | None = None,
) -> PositivityDiagnostics:
    audited_count = max(0, int(audited_route_pair_count))
    candidate_count = max(audited_count, int(candidate_route_pair_count or audited_count))
    coverage_ratio = 0.0 if candidate_count <= 0 else audited_count / candidate_count
    propensities = [_clamp01(value) for value in list(propensity_scores or [])]
    minimum_propensity = min(propensities) if propensities else 0.0
    mean_propensity = sum(propensities) / len(propensities) if propensities else 0.0
    maximum_propensity = max(propensities) if propensities else 0.0
    support_bin = str(_value_from(support_state, "support_bin", "unspecified") or "unspecified")
    support_condition = (
        str(_value_from(support_state, "out_of_support_reason", "")).strip()
        or str(_value_from(support_state, "support_source", "unknown") or "unknown")
    )
    positivity_ok = (
        audited_count > 0
        and coverage_ratio >= _clamp01(min_audit_coverage_ratio)
        and minimum_propensity >= _clamp01(min_propensity)
    )
    weak_overlap_detected = audited_count > 0 and minimum_propensity < _clamp01(min_propensity)
    if audited_count <= 0:
        recommendation = "collect_initial_audits"
    elif coverage_ratio < _clamp01(min_audit_coverage_ratio):
        recommendation = "increase_audit_coverage"
    elif weak_overlap_detected:
        recommendation = "widen_support_before_proxy_certification"
    else:
        recommendation = "ready_for_support_aware_correction"
    return PositivityDiagnostics(
        audited_route_pair_count=audited_count,
        audit_coverage_ratio=coverage_ratio,
        minimum_propensity=minimum_propensity,
        mean_propensity=mean_propensity,
        maximum_propensity=maximum_propensity,
        positivity_ok=positivity_ok,
        weak_overlap_detected=weak_overlap_detected,
        support_bin=support_bin,
        support_condition=support_condition,
        recommendation=recommendation,
        provenance=dict(provenance or {}),
    )


def build_multi_fidelity_summary(
    *,
    probabilistic_world_bundle: Any | None = None,
    audit_world_bundle: Any | None = None,
    support_state: Any | None = None,
    proxy_world_count: int | None = None,
    audit_world_count: int | None = None,
    proxy_bias_model_version: str | None = None,
    audit_propensity_version: str | None = None,
    proxy_correction_active: bool = False,
    multi_fidelity_certificate_basis: str | None = None,
    certification_evaluation_tag: str | None = None,
    correction_conditioning_features: list[str] | None = None,
    propensity_conditioning_features: list[str] | None = None,
    correction_training_leakage_safe: bool | None = None,
    propensity_training_leakage_safe: bool | None = None,
    correction_path_estimator: str | None = None,
    proxy_only_fraction: float | None = None,
    audit_correction_mass: float = 0.0,
    positivity_diagnostics: PositivityDiagnostics | None = None,
    cache_mode: str | None = None,
    audited_route_pair_count: int | None = None,
    candidate_route_pair_count: int | None = None,
    propensity_scores: list[float] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> MultiFidelitySummary:
    resolved_proxy_world_count = max(
        0,
        int(
            proxy_world_count
            if proxy_world_count is not None
            else _value_from(probabilistic_world_bundle, "world_count", 0)
        ),
    )
    resolved_audit_world_count = max(
        0,
        int(
            audit_world_count
            if audit_world_count is not None
            else _value_from(audit_world_bundle, "audit_world_count", 0)
        ),
    )
    total_world_count = resolved_proxy_world_count + resolved_audit_world_count
    resolved_proxy_only_fraction = (
        _clamp01(proxy_only_fraction)
        if proxy_only_fraction is not None
        else (
            0.0
            if total_world_count <= 0
            else resolved_proxy_world_count / max(1, total_world_count)
        )
    )
    resolved_proxy_bias_model_version = str(
        proxy_bias_model_version
        or _value_from(probabilistic_world_bundle, "policy_version", "")
        or ""
    )
    resolved_audit_propensity_version = str(
        audit_propensity_version
        or _value_from(audit_world_bundle, "policy_version", "")
        or ""
    )
    resolved_cache_mode = str(
        cache_mode
        or _value_from(audit_world_bundle, "cache_mode", _value_from(probabilistic_world_bundle, "cache_mode", "cold"))
        or "cold"
    ).strip().lower()
    resolved_positivity = positivity_diagnostics or build_positivity_diagnostics(
        audited_route_pair_count=(
            audited_route_pair_count
            if audited_route_pair_count is not None
            else resolved_audit_world_count
        ),
        candidate_route_pair_count=(
            candidate_route_pair_count
            if candidate_route_pair_count is not None
            else total_world_count
        ),
        propensity_scores=propensity_scores,
        support_state=support_state,
        provenance={"source": "build_multi_fidelity_summary"},
    )
    if multi_fidelity_certificate_basis:
        basis = str(multi_fidelity_certificate_basis).strip()
    elif proxy_correction_active:
        basis = "corrected_from_residual_model"
    elif resolved_cache_mode != "cold" and total_world_count > 0:
        basis = "reused_from_cache"
    elif resolved_proxy_world_count > 0 and resolved_audit_world_count > 0:
        basis = "partially_audited"
    elif resolved_audit_world_count > 0:
        basis = "fully_audited"
    else:
        basis = "proxy_only"
    resolved_correction_features = _normalize_feature_names(correction_conditioning_features)
    resolved_propensity_features = _normalize_feature_names(propensity_conditioning_features)
    resolved_correction_training_leakage_safe = bool(
        correction_training_leakage_safe if correction_training_leakage_safe is not None else False
    )
    resolved_propensity_training_leakage_safe = bool(
        propensity_training_leakage_safe if propensity_training_leakage_safe is not None else False
    )
    if correction_path_estimator is not None:
        resolved_correction_path_estimator = str(correction_path_estimator).strip() or "proxy_only"
    elif resolved_propensity_training_leakage_safe and bool(resolved_audit_propensity_version.strip()):
        resolved_correction_path_estimator = (
            "doubly_robust_residual_correction"
            if bool(proxy_correction_active)
            else "propensity_aware_residual_correction"
        )
    elif bool(proxy_correction_active):
        resolved_correction_path_estimator = "residual_correction_only"
    else:
        resolved_correction_path_estimator = "proxy_only"
    resolved_evaluation_tag = (
        str(certification_evaluation_tag).strip()
        if certification_evaluation_tag is not None
        else basis
    ) or basis
    return MultiFidelitySummary(
        proxy_world_count=resolved_proxy_world_count,
        audit_world_count=resolved_audit_world_count,
        proxy_bias_model_version=resolved_proxy_bias_model_version,
        audit_propensity_version=resolved_audit_propensity_version,
        proxy_correction_active=bool(proxy_correction_active),
        multi_fidelity_certificate_basis=basis,
        certification_evaluation_tag=resolved_evaluation_tag,
        correction_conditioning_features=resolved_correction_features,
        propensity_conditioning_features=resolved_propensity_features,
        conditions_on_corridor_family=_has_feature(resolved_correction_features, "corridor_family"),
        conditions_on_ambiguity_regime=_has_feature(resolved_correction_features, "ambiguity_regime"),
        conditions_on_support_regime=_has_feature(resolved_correction_features, "support_regime"),
        conditions_on_evidence_family_regime=_has_feature(
            resolved_correction_features,
            "evidence_family_regime",
        ),
        conditions_on_engine_disagreement_regime=_has_feature(
            resolved_correction_features,
            "engine_disagreement_regime",
        ),
        conditions_on_candidate_density_or_pressure=_has_feature(
            resolved_correction_features,
            "candidate_density_or_pressure",
        ),
        correction_training_leakage_safe=resolved_correction_training_leakage_safe,
        propensity_training_leakage_safe=resolved_propensity_training_leakage_safe,
        leakage_safe_training=(
            resolved_correction_training_leakage_safe
            and resolved_propensity_training_leakage_safe
        ),
        correction_path_estimator=resolved_correction_path_estimator,
        proxy_only_fraction=resolved_proxy_only_fraction,
        audit_correction_mass=max(0.0, _as_float(audit_correction_mass)),
        positivity_diagnostics=resolved_positivity,
        provenance=dict(provenance or {}),
    )
