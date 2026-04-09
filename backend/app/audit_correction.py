from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .fidelity_model import ProbabilisticWorldBundle
from .world_policies import clamp01


def _mean_float(values: Sequence[Any], default: float) -> float:
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return float(default)
    return sum(parsed) / float(len(parsed))


def _contains_proxy_marker(*values: Any) -> bool:
    combined = " ".join(str(value) for value in values if value not in (None, "")).lower()
    return any(token in combined for token in ("proxy", "fallback", "synthetic", "bootstrap", "fixture"))


@dataclass(frozen=True)
class ProxyAuditRecord:
    family: str
    audit_mode: str
    proxy_fraction: float
    fallback_rate: float
    confidence_mean: float
    coverage_mean: float
    penalty: float
    requires_correction: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_family(
        cls,
        family: str,
        *,
        probabilistic_bundle: ProbabilisticWorldBundle,
        provenance_rows: Sequence[Mapping[str, Any]] = (),
    ) -> "ProxyAuditRecord":
        rows = [dict(row) for row in provenance_rows if isinstance(row, Mapping)]
        proxy_fraction = float(
            probabilistic_bundle.family_state_weights.get(family, {}).get("proxy", 0.0)
        )
        fallback_rate = 0.0
        if rows:
            fallback_rate = sum(1 for row in rows if bool(row.get("fallback_used", False))) / float(len(rows))
        confidence_mean = _mean_float((row.get("confidence") for row in rows), 1.0)
        coverage_mean = _mean_float((row.get("coverage_ratio") for row in rows), 1.0)
        proxy_marked = any(
            _contains_proxy_marker(
                row.get("source"),
                row.get("signature"),
                row.get("fallback_source"),
                row.get("snapshot_id"),
            )
            for row in rows
        )
        if proxy_marked or proxy_fraction > 0.0:
            audit_mode = "proxy"
        elif family == "stochastic":
            audit_mode = "model"
        elif any("snapshot" in str(row.get("source", "")).lower() for row in rows):
            audit_mode = "snapshot"
        else:
            audit_mode = "live"
        penalty = clamp01(
            (0.45 * proxy_fraction)
            + (0.20 * fallback_rate)
            + (0.20 * (1.0 - confidence_mean))
            + (0.15 * (1.0 - coverage_mean))
            + (0.10 if proxy_marked else 0.0),
        )
        notes: list[str] = []
        if proxy_marked:
            notes.append("provenance_proxy_marker")
        if fallback_rate > 0.0:
            notes.append("fallback_used")
        if coverage_mean < 0.75:
            notes.append("coverage_gap")
        if confidence_mean < 0.75:
            notes.append("confidence_gap")
        return cls(
            family=family,
            audit_mode=audit_mode,
            proxy_fraction=round(proxy_fraction, 6),
            fallback_rate=round(fallback_rate, 6),
            confidence_mean=round(clamp01(confidence_mean, 1.0), 6),
            coverage_mean=round(clamp01(coverage_mean, 1.0), 6),
            penalty=round(penalty, 6),
            requires_correction=bool(penalty > 0.15 or audit_mode == "proxy"),
            notes=tuple(notes),
        )


@dataclass(frozen=True)
class AuditWorldBundle:
    bundle_id: str
    audited_families: tuple[str, ...]
    proxy_records: tuple[ProxyAuditRecord, ...]
    proxy_family_count: int
    audited_world_fraction: float
    correction_scale: float
    recommended_policy: str
    manifest_hash: str | None = None

    @property
    def correction_penalty(self) -> float:
        return round(1.0 - self.correction_scale, 6)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["proxy_records"] = [record.as_dict() for record in self.proxy_records]
        return payload

    @classmethod
    def from_sources(
        cls,
        probabilistic_bundle: ProbabilisticWorldBundle,
        evidence_snapshot_manifest: Mapping[str, Any] | None = None,
    ) -> "AuditWorldBundle":
        manifest = (
            dict(evidence_snapshot_manifest)
            if isinstance(evidence_snapshot_manifest, Mapping)
            else {}
        )
        raw_family_snapshots = manifest.get("family_snapshots", {})
        family_snapshots = (
            dict(raw_family_snapshots)
            if isinstance(raw_family_snapshots, Mapping)
            else {}
        )
        records: list[ProxyAuditRecord] = []
        for family in probabilistic_bundle.active_families:
            rows = family_snapshots.get(family, [])
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                rows = []
            records.append(
                ProxyAuditRecord.from_family(
                    family,
                    probabilistic_bundle=probabilistic_bundle,
                    provenance_rows=[row for row in rows if isinstance(row, Mapping)],
                )
            )
        proxy_family_count = sum(1 for record in records if record.requires_correction)
        mean_penalty = _mean_float((record.penalty for record in records), 0.0)
        correction_scale = max(0.25, 1.0 - mean_penalty)
        recommended_policy = "audit_first" if proxy_family_count > 0 else "direct_use"
        return cls(
            bundle_id=f"{probabilistic_bundle.bundle_id}:audit",
            audited_families=tuple(record.family for record in records),
            proxy_records=tuple(records),
            proxy_family_count=proxy_family_count,
            audited_world_fraction=round(probabilistic_bundle.proxy_world_fraction, 6),
            correction_scale=round(clamp01(correction_scale, 1.0), 6),
            recommended_policy=recommended_policy,
            manifest_hash=str(manifest.get("manifest_hash", "")).strip() or probabilistic_bundle.manifest_hash,
        )
