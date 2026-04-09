from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _stable_json(payload: Mapping[str, Any] | None) -> str:
    normalized = dict(payload or {})
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)


def normalize_policy_name(policy_name: str) -> str:
    key = str(policy_name or "").strip().lower()
    return key or "unspecified"


def policy_version_tag(policy_name: str, version: str) -> str:
    name = normalize_policy_name(policy_name)
    ver = str(version or "").strip().lower() or "v1"
    return f"{name}:{ver}"


def policy_hash(
    policy_name: str,
    *,
    version: str = "v1",
    configuration: Mapping[str, Any] | None = None,
) -> str:
    material = "|".join(
        [
            normalize_policy_name(policy_name),
            str(version or "v1").strip().lower() or "v1",
            _stable_json(configuration),
        ]
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyFingerprint:
    schema_version: str
    policy_name: str
    policy_version: str
    policy_hash: str
    configuration: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_policy_fingerprint(
    policy_name: str,
    *,
    version: str = "v1",
    configuration: Mapping[str, Any] | None = None,
    schema_version: str = "policy-fingerprint-v1",
) -> PolicyFingerprint:
    normalized_config = dict(configuration or {})
    return PolicyFingerprint(
        schema_version=schema_version,
        policy_name=normalize_policy_name(policy_name),
        policy_version=str(version or "v1").strip().lower() or "v1",
        policy_hash=policy_hash(policy_name, version=version, configuration=normalized_config),
        configuration=normalized_config,
    )
