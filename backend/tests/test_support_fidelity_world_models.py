from __future__ import annotations

import json

from app.audit_correction import (
    build_audit_propensity_metadata,
    build_leakage_safe_correction_metadata,
    build_proxy_audit_record,
)
from app.fidelity_model import build_action_value_estimate
from app.support_model import (
    build_audit_world_bundle,
    build_probabilistic_world_bundle,
    build_world_support_state,
)
from app.world_policies import build_policy_fingerprint, policy_hash, policy_version_tag


def test_support_world_bundles_and_action_estimates_are_json_serializable() -> None:
    support_state = build_world_support_state(
        support_score=0.74,
        support_ratio=0.62,
        support_bin="in_support",
        calibration_bin="bin_3",
        support_source="proxy",
        provenance={"source": "unit-test"},
    )
    prob_bundle = build_probabilistic_world_bundle(
        bundle_id="prob-1",
        worlds=[{"world_id": "w1", "weight": 0.7}],
        support_state=support_state,
        cache_mode="warm",
        policy_name="world-policy",
        policy_version="v2",
        policy_hash="abc123",
        provenance={"mode": "probabilistic"},
    )
    audit_bundle = build_audit_world_bundle(
        bundle_id="audit-1",
        audit_worlds=[{"world_id": "a1", "weight": 1.0}],
        support_state=support_state,
        cache_mode="cold",
        policy_name="audit-policy",
        policy_version="v1",
        policy_hash="def456",
        provenance={"mode": "audit"},
    )
    action_estimate = build_action_value_estimate(
        action_id="refresh_top1_vor",
        expected_gain=12.5,
        expected_cost=3.5,
        confidence=0.82,
        support_weight=0.64,
        fidelity_class="audit",
        provenance={"row": "row-1"},
    )

    assert support_state.support_flag is True
    assert json.loads(json.dumps(support_state.as_dict()))["support_bin"] == "in_support"
    assert json.loads(json.dumps(prob_bundle.as_dict()))["world_count"] == 1
    assert json.loads(json.dumps(audit_bundle.as_dict()))["audit_world_count"] == 1
    encoded_action = json.loads(json.dumps(action_estimate.as_dict()))
    assert encoded_action["expected_net_gain"] == 9.0
    assert encoded_action["gain_per_cost"] > 0.0


def test_policy_helpers_are_stable_and_records_include_leakage_safe_metadata() -> None:
    fp = build_policy_fingerprint(
        "Proxy Bias Correction",
        version="V3",
        configuration={"alpha": 0.1, "beta": 2},
    )
    assert policy_version_tag("Proxy Bias Correction", "V3") == "proxy bias correction:v3"
    assert policy_hash("Proxy Bias Correction", version="V3", configuration={"alpha": 0.1, "beta": 2}) == fp.policy_hash

    correction_meta = build_leakage_safe_correction_metadata(
        model_version="v3",
        policy_hash=fp.policy_hash,
        fold_count=5,
        training_rows=100,
        validation_rows=20,
        feature_names=["corridor_family", "support_bin"],
        training_scope="cross_fit",
    )
    propensity_meta = build_audit_propensity_metadata(
        model_version="v2",
        policy_hash=fp.policy_hash,
        fold_count=5,
        training_rows=120,
        validation_rows=24,
        feature_names=["corridor_family", "support_bin"],
        training_scope="cross_fit",
    )
    record = build_proxy_audit_record(
        row_id="row-7",
        route_id="route-7",
        evidence_family="weather",
        proxy_value=100.0,
        audited_value=109.5,
        audit_probability=0.25,
        propensity_score=0.33,
        correction_metadata=correction_meta,
        propensity_metadata=propensity_meta,
        provenance={"source": "synthetic"},
    )

    payload = json.loads(json.dumps(record.as_dict()))
    assert payload["residual_bias"] == 9.5
    assert payload["absolute_residual"] == 9.5
    assert payload["correction_metadata"]["cross_fitted"] is True
    assert payload["correction_metadata"]["same_row_fit_prohibited"] is True
    assert payload["propensity_metadata"]["out_of_fold_only"] is True
    assert payload["support_state"]["schema_version"] == "world-support-v1"
