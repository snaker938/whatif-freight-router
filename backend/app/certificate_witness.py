"""Witness objects for terminal REFC decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


def build_support_conditions(
    *,
    support_flag: bool,
    support_reason: str | None,
    selected_certificate_basis: str | None,
    support_bin: str | None = None,
) -> list[str]:
    conditions = [f"support_flag={support_flag}"]
    if support_reason:
        conditions.append(f"support_reason={support_reason}")
    if selected_certificate_basis:
        conditions.append(f"selected_certificate_basis={selected_certificate_basis}")
    if support_bin:
        conditions.append(f"support_bin={support_bin}")
    return conditions


def derive_witness_action_steps(
    *,
    active_challenger_ids: list[str],
    active_evidence_families: list[str],
    active_preference_constraints: list[str],
    support_flag: bool,
    support_reason: str | None,
    nearest_certificate_boundary: str | None,
) -> list[str]:
    steps: list[str] = []
    if not support_flag:
        steps.append(f"restore_support:{support_reason or 'out_of_support_world_model'}")
    if active_challenger_ids:
        steps.append(f"separate_from:{active_challenger_ids[0]}")
    for family in active_evidence_families[:2]:
        steps.append(f"refresh_family:{family}")
    for constraint in active_preference_constraints[:1]:
        steps.append(f"query_preference:{constraint}")
    if nearest_certificate_boundary:
        steps.append(f"inspect_boundary:{nearest_certificate_boundary}")

    deduped: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if step not in seen:
            deduped.append(step)
            seen.add(step)
    return deduped


@dataclass(frozen=True)
class CertificateWitness:
    route_id: str
    active_challenger_ids: list[str] = field(default_factory=list)
    active_evidence_families: list[str] = field(default_factory=list)
    active_preference_constraints: list[str] = field(default_factory=list)
    support_conditions: list[str] = field(default_factory=list)
    action_steps: list[str] = field(default_factory=list)
    witness_sparsity: float | None = None
    witness_size: int = 0
    explanation_sparsity: float | None = None
    selected_certificate_basis: str | None = None
    support_status: str | None = None
    support_bin: str | None = None
    support_reason: str | None = None
    calibration_bin: str | None = None
    calibration_policy_version: str | None = None
    nearest_certificate_boundary: str | None = None
    targeted_challenger_route_id: str | None = None
    active_challenger_count: int = 0
    active_evidence_family_count: int = 0
    active_preference_constraint_count: int = 0
    support_condition_count: int = 0
    action_step_count: int = 0
    atlas_kind: str | None = None
    root_cause_tags: list[str] = field(default_factory=list)
    support_flag: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
