from __future__ import annotations

import json

from app.audit_correction import (
    build_audit_propensity_metadata,
    build_leakage_safe_correction_metadata,
    build_proxy_audit_record,
    summarize_proxy_audit_records,
)
from app.fidelity_model import build_action_value_estimate
from app.risk_model import build_risk_summary
from app.scenario import (
    ScenarioMode,
    ScenarioPolicy,
    build_scenario_support_summary,
)
from app.support_model import (
    build_audit_world_bundle,
    build_multi_fidelity_summary,
    build_probabilistic_world_bundle,
    build_positivity_diagnostics,
    build_world_support_state,
)
from app.uncertainty_model import build_world_bundle_summary
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
    assert payload["pairwise_evaluation_tag"] == "corrected_from_residual_model"
    assert payload["correction_metadata"]["cross_fitted"] is True
    assert payload["correction_metadata"]["same_row_fit_prohibited"] is True
    assert payload["propensity_metadata"]["out_of_fold_only"] is True
    assert payload["correction_metadata"]["feature_names"] == [
        "corridor_family",
        "support_regime",
        "ambiguity_regime",
        "evidence_family_regime",
        "engine_disagreement_regime",
        "candidate_density_or_pressure",
    ]
    assert payload["propensity_metadata"]["feature_names"] == [
        "corridor_family",
        "support_regime",
        "ambiguity_regime",
        "evidence_family_regime",
        "engine_disagreement_regime",
        "candidate_density_or_pressure",
    ]
    assert payload["support_state"]["schema_version"] == "world-support-v1"


def test_audit_correction_metadata_clamps_counts_and_preserves_scope_fields() -> None:
    correction_meta = build_leakage_safe_correction_metadata(
        model_version="proxy-v9",
        policy_hash="policy-123",
        fold_count=-5,
        training_rows=-100,
        validation_rows=-20,
        feature_names=["corridor_family", "support_bin"],
        training_scope="cross_fit",
        provenance={"source": "unit-test", "fold_strategy": "temporal"},
    )
    propensity_meta = build_audit_propensity_metadata(
        model_version="audit-v4",
        policy_hash="policy-123",
        fold_count=-3,
        training_rows=-60,
        validation_rows=-12,
        feature_names=["corridor_family", "support_bin"],
        training_scope="cross_fit",
        provenance={"source": "unit-test", "sampler": "audit"},
    )

    correction_payload = json.loads(json.dumps(correction_meta.as_dict()))
    propensity_payload = json.loads(json.dumps(propensity_meta.as_dict()))

    assert correction_payload["fold_count"] == 0
    assert correction_payload["training_rows"] == 0
    assert correction_payload["validation_rows"] == 0
    assert correction_payload["policy_hash"] == "policy-123"
    assert correction_payload["feature_names"] == [
        "corridor_family",
        "support_regime",
        "ambiguity_regime",
        "evidence_family_regime",
        "engine_disagreement_regime",
        "candidate_density_or_pressure",
    ]
    assert correction_payload["training_scope"] == "cross_fit"
    assert correction_payload["provenance"]["fold_strategy"] == "temporal"

    assert propensity_payload["fold_count"] == 0
    assert propensity_payload["training_rows"] == 0
    assert propensity_payload["validation_rows"] == 0
    assert propensity_payload["policy_hash"] == "policy-123"
    assert propensity_payload["feature_names"] == [
        "corridor_family",
        "support_regime",
        "ambiguity_regime",
        "evidence_family_regime",
        "engine_disagreement_regime",
        "candidate_density_or_pressure",
    ]
    assert propensity_payload["training_scope"] == "cross_fit"
    assert propensity_payload["provenance"]["sampler"] == "audit"


def test_multi_fidelity_summaries_surface_positivity_and_versions() -> None:
    support_state = build_world_support_state(
        support_score=0.84,
        support_ratio=0.78,
        support_bin="supported",
        calibration_bin="bin_2",
        support_source="audit_correction",
    )
    probabilistic_bundle = build_probabilistic_world_bundle(
        bundle_id="prob-2",
        worlds=[{"world_id": "w1"}, {"world_id": "w2"}, {"world_id": "w3"}, {"world_id": "w4"}],
        support_state=support_state,
        cache_mode="warm",
        policy_name="proxy-world-policy",
        policy_version="proxy-v4",
    )
    audit_bundle = build_audit_world_bundle(
        bundle_id="audit-2",
        audit_worlds=[{"world_id": "a1"}, {"world_id": "a2"}],
        support_state=support_state,
        cache_mode="cold",
        policy_name="audit-world-policy",
        policy_version="audit-v2",
    )
    positivity = build_positivity_diagnostics(
        audited_route_pair_count=2,
        candidate_route_pair_count=6,
        propensity_scores=[0.21, 0.44],
        support_state=support_state,
    )
    summary = build_multi_fidelity_summary(
        probabilistic_world_bundle=probabilistic_bundle,
        audit_world_bundle=audit_bundle,
        support_state=support_state,
        proxy_bias_model_version="proxy-v4",
        audit_propensity_version="audit-v2",
        proxy_correction_active=True,
        correction_conditioning_features=[
            "corridor_family",
            "ambiguity_regime",
            "support_regime",
            "evidence_family_regime",
            "engine_disagreement_regime",
            "candidate_density_or_pressure",
        ],
        propensity_conditioning_features=[
            "corridor_family",
            "ambiguity_regime",
            "support_regime",
            "evidence_family_regime",
            "engine_disagreement_regime",
            "candidate_density_or_pressure",
        ],
        correction_training_leakage_safe=True,
        propensity_training_leakage_safe=True,
        correction_path_estimator="doubly_robust_residual_correction",
        positivity_diagnostics=positivity,
        audit_correction_mass=7.5,
    )

    encoded = json.loads(json.dumps(summary.as_dict()))
    assert encoded["proxy_world_count"] == 4
    assert encoded["audit_world_count"] == 2
    assert encoded["proxy_bias_model_version"] == "proxy-v4"
    assert encoded["audit_propensity_version"] == "audit-v2"
    assert encoded["proxy_correction_active"] is True
    assert encoded["multi_fidelity_certificate_basis"] == "corrected_from_residual_model"
    assert encoded["certification_evaluation_tag"] == "corrected_from_residual_model"
    assert encoded["conditions_on_corridor_family"] is True
    assert encoded["conditions_on_ambiguity_regime"] is True
    assert encoded["conditions_on_support_regime"] is True
    assert encoded["conditions_on_evidence_family_regime"] is True
    assert encoded["conditions_on_engine_disagreement_regime"] is True
    assert encoded["conditions_on_candidate_density_or_pressure"] is True
    assert encoded["correction_training_leakage_safe"] is True
    assert encoded["propensity_training_leakage_safe"] is True
    assert encoded["leakage_safe_training"] is True
    assert encoded["correction_path_estimator"] == "doubly_robust_residual_correction"
    assert encoded["proxy_only_fraction"] == 4 / 6
    assert encoded["audit_correction_mass"] == 7.5
    assert encoded["positivity_diagnostics"]["positivity_ok"] is True


def test_out_of_support_worlds_downgrade_legacy_supported_bins() -> None:
    support_state = build_world_support_state(
        support_score=0.84,
        support_ratio=0.78,
        support_bin="supported",
        calibration_bin="bin_2",
        support_source="audit_correction",
        out_of_support_reason="out_of_support_world_model",
    )

    assert support_state.support_flag is False
    assert support_state.support_bin == "weak_support"
    assert json.loads(json.dumps(support_state.as_dict()))["support_bin"] == "weak_support"


def test_proxy_audit_summary_threads_through_world_scenario_and_risk_objects() -> None:
    support_state = build_world_support_state(
        support_score=0.79,
        support_ratio=0.68,
        support_bin="supported",
        calibration_bin="bin_3",
        support_source="unit-test",
    )
    probabilistic_bundle = build_probabilistic_world_bundle(
        bundle_id="prob-3",
        worlds=[{"world_id": "w1"}, {"world_id": "w2"}, {"world_id": "w3"}],
        support_state=support_state,
        cache_mode="warm",
        policy_version="proxy-v5",
    )
    audit_bundle = build_audit_world_bundle(
        bundle_id="audit-3",
        audit_worlds=[{"world_id": "a1"}],
        support_state=support_state,
        policy_version="audit-v3",
    )
    correction_meta = build_leakage_safe_correction_metadata(model_version="proxy-v5")
    propensity_meta = build_audit_propensity_metadata(model_version="audit-v3")
    records = [
        build_proxy_audit_record(
            row_id="row-1",
            route_id="route-1",
            evidence_family="weather",
            proxy_value=100.0,
            audited_value=110.0,
            audit_probability=0.35,
            propensity_score=0.28,
            support_state=support_state,
            correction_metadata=correction_meta,
            propensity_metadata=propensity_meta,
        ),
        build_proxy_audit_record(
            row_id="row-2",
            route_id="route-2",
            evidence_family="scenario",
            proxy_value=90.0,
            audited_value=96.0,
            audit_probability=0.42,
            propensity_score=0.31,
            support_state=support_state,
            correction_metadata=correction_meta,
            propensity_metadata=propensity_meta,
        ),
    ]
    summary = summarize_proxy_audit_records(
        records,
        proxy_world_count=3,
        audit_world_count=1,
        support_state=support_state,
    )
    world_bundle_summary = build_world_bundle_summary(
        manifest={
            "world_count": 3,
            "proxy_world_count": 3,
            "audit_world_count": 1,
            "proxy_bias_model_version": "proxy-v5",
            "audit_propensity_version": "audit-v3",
            "proxy_correction_active": True,
            "audit_correction_mass": summary.audit_correction_mass,
            "multi_fidelity_certificate_basis": "corrected_from_residual_model",
            "audit_propensity_scores": [0.28, 0.31],
        },
        support_state=support_state,
        audit_world_bundle=audit_bundle,
    )
    scenario_support = build_scenario_support_summary(
        ScenarioPolicy(
            duration_multiplier=1.0,
            incident_rate_multiplier=1.0,
            incident_delay_multiplier=1.0,
            fuel_consumption_multiplier=1.0,
            emissions_multiplier=1.0,
            stochastic_sigma_multiplier=1.0,
            source="unit-test",
            version="v1",
        ),
        mode=ScenarioMode.NO_SHARING,
        support_state=support_state,
        probabilistic_world_bundle=probabilistic_bundle,
        audit_world_bundle=audit_bundle,
    )
    risk_summary = build_risk_summary(
        duration_s=3600.0,
        monetary_cost=50.0,
        emissions_kg=20.0,
        distance_km=120.0,
        support_state=support_state,
        probabilistic_world_bundle=probabilistic_bundle,
        audit_world_bundle=audit_bundle,
    )

    assert summary.proxy_bias_model_version == "proxy-v5"
    assert summary.audit_propensity_version == "audit-v3"
    assert summary.positivity_diagnostics.audited_route_pair_count == 2
    assert summary.multi_fidelity_certificate_basis == "corrected_from_residual_model"
    assert summary.certification_evaluation_tag == "corrected_from_residual_model"
    assert summary.conditions_on_ambiguity_regime is True
    assert summary.conditions_on_evidence_family_regime is True
    assert summary.conditions_on_engine_disagreement_regime is True
    assert summary.conditions_on_candidate_density_or_pressure is True
    assert summary.correction_training_leakage_safe is True
    assert summary.propensity_training_leakage_safe is True
    assert summary.leakage_safe_training is True
    assert summary.correction_path_estimator == "doubly_robust_residual_correction"

    world_payload = world_bundle_summary.as_dict()
    assert world_payload["multi_fidelity_summary"]["proxy_world_count"] == 3
    assert world_payload["multi_fidelity_summary"]["audit_world_count"] == 1
    assert world_payload["multi_fidelity_summary"]["proxy_correction_active"] is True
    assert (
        world_payload["multi_fidelity_summary"]["certification_evaluation_tag"]
        == "corrected_from_residual_model"
    )

    scenario_payload = scenario_support.as_dict()
    assert scenario_payload["multi_fidelity_summary"]["proxy_world_count"] == 3
    assert scenario_payload["multi_fidelity_summary"]["audit_world_count"] == 1

    risk_payload = risk_summary.as_dict()
    assert risk_payload["multi_fidelity_summary"]["proxy_world_count"] == 3
    assert risk_payload["multi_fidelity_summary"]["audit_world_count"] == 1


def test_proxy_audit_record_and_summary_sanitize_sparse_edge_inputs() -> None:
    support_state = build_world_support_state(
        support_score=0.83,
        support_ratio=0.61,
        support_bin="supported",
        calibration_bin="bin_2",
        support_source="audit_correction",
    )
    correction_meta = build_leakage_safe_correction_metadata(
        model_version="proxy-v-edge",
        provenance={"source": "correction-edge"},
    )
    propensity_meta = build_audit_propensity_metadata(
        model_version="audit-v-edge",
        provenance={"source": "propensity-edge"},
    )
    record = build_proxy_audit_record(
        row_id="row-edge",
        route_id="route-edge",
        evidence_family="weather",
        proxy_value=0.0,
        audited_value=15.0,
        audit_probability=1.7,
        propensity_score=-0.2,
        support_state=support_state,
        correction_metadata=correction_meta,
        propensity_metadata=propensity_meta,
        provenance={"source": "edge-case"},
    )

    payload = json.loads(json.dumps(record.as_dict()))
    assert payload["residual_bias"] == 15.0
    assert payload["absolute_residual"] == 15.0
    assert payload["correction_factor"] == 1.0
    assert payload["audit_probability"] == 1.0
    assert payload["propensity_score"] == 0.0
    assert payload["correction_applied"] is True
    assert payload["pairwise_evaluation_tag"] == "corrected_from_residual_model"
    assert payload["correction_metadata"]["model_version"] == "proxy-v-edge"
    assert payload["propensity_metadata"]["model_version"] == "audit-v-edge"
    assert payload["provenance"]["source"] == "edge-case"

    summary = summarize_proxy_audit_records([record], proxy_world_count=3)

    assert summary.proxy_world_count == 3
    assert summary.audit_world_count == 1
    assert summary.proxy_bias_model_version == "proxy-v-edge"
    assert summary.audit_propensity_version == "audit-v-edge"
    assert summary.proxy_correction_active is True
    assert summary.audit_correction_mass == 15.0
    assert summary.proxy_only_fraction == 3 / 4
    assert summary.provenance["source"] == "proxy_audit_records"
    assert summary.certification_evaluation_tag == "corrected_from_residual_model"
    assert summary.correction_training_leakage_safe is True
    assert summary.propensity_training_leakage_safe is True
    assert summary.leakage_safe_training is True
    assert summary.correction_path_estimator == "doubly_robust_residual_correction"
    assert summary.positivity_diagnostics.audited_route_pair_count == 1
    assert summary.positivity_diagnostics.audit_coverage_ratio == 1 / 4
    assert summary.positivity_diagnostics.minimum_propensity == 0.0
    assert summary.positivity_diagnostics.weak_overlap_detected is True
    assert (
        summary.positivity_diagnostics.recommendation
        == "widen_support_before_proxy_certification"
    )


def test_proxy_audit_summary_requires_all_records_to_be_leakage_safe() -> None:
    support_state = build_world_support_state(
        support_score=0.83,
        support_ratio=0.61,
        support_bin="supported",
        calibration_bin="bin_2",
        support_source="audit_correction",
    )
    safe_correction_meta = build_leakage_safe_correction_metadata(
        model_version="proxy-v-safe",
        provenance={"source": "safe"},
    )
    safe_propensity_meta = build_audit_propensity_metadata(
        model_version="audit-v-safe",
        provenance={"source": "safe"},
    )
    unsafe_correction_meta = build_leakage_safe_correction_metadata(
        model_version="proxy-v-unsafe",
        cross_fitted=False,
        provenance={"source": "unsafe"},
    )
    unsafe_propensity_meta = build_audit_propensity_metadata(
        model_version="audit-v-unsafe",
        out_of_fold_only=False,
        provenance={"source": "unsafe"},
    )
    records = [
        build_proxy_audit_record(
            row_id="row-safe",
            route_id="route-safe",
            evidence_family="scenario",
            proxy_value=100.0,
            audited_value=103.0,
            audit_probability=0.41,
            propensity_score=0.32,
            support_state=support_state,
            correction_metadata=safe_correction_meta,
            propensity_metadata=safe_propensity_meta,
        ),
        build_proxy_audit_record(
            row_id="row-unsafe",
            route_id="route-unsafe",
            evidence_family="scenario",
            proxy_value=95.0,
            audited_value=99.0,
            audit_probability=0.43,
            propensity_score=0.28,
            support_state=support_state,
            correction_metadata=unsafe_correction_meta,
            propensity_metadata=unsafe_propensity_meta,
        ),
    ]

    summary = summarize_proxy_audit_records(records, proxy_world_count=4, audit_world_count=2)

    assert summary.correction_training_leakage_safe is False
    assert summary.propensity_training_leakage_safe is False
    assert summary.leakage_safe_training is False


def test_world_bundle_summary_infers_proxy_and_audit_counts_from_manifest_lists() -> None:
    support_state = build_world_support_state(
        support_score=0.82,
        support_ratio=0.74,
        support_bin="supported",
        calibration_bin="bin_3",
        support_source="audit_correction",
    )

    summary = build_world_bundle_summary(
        manifest={
            "world_count": 4,
            "probabilistic_worlds": [
                {"world_id": "p1", "world_kind": "sampled", "states": {"fuel": "nominal"}},
                {"world_id": "p2", "world_kind": "sampled", "states": {"fuel": "low_confidence"}},
                {"world_id": "p3", "world_kind": "sampled", "states": {"fuel": "refreshed"}},
            ],
            "audit_worlds": [
                {"world_id": "a1", "world_kind": "hard_case_targeted", "states": {"fuel": "nominal"}}
            ],
            "audited_worlds": [
                {"world_id": "p3", "world_kind": "sampled", "states": {"fuel": "refreshed"}},
                {"world_id": "a1", "world_kind": "hard_case_targeted", "states": {"fuel": "refreshed"}},
            ],
            "calibration_policy_version": "calibration-v2",
            "audit_propensity_version": "audit-v4",
        },
        support_state=support_state,
    )

    payload = summary.as_dict()
    assert payload["probabilistic_world_bundle"]["world_count"] == 3
    assert payload["multi_fidelity_summary"]["proxy_world_count"] == 3
    assert payload["multi_fidelity_summary"]["audit_world_count"] == 1
    assert payload["multi_fidelity_summary"]["positivity_diagnostics"]["audited_route_pair_count"] == 2
    assert payload["audit_world_bundle"]["audit_world_count"] == 1
    assert payload["audit_world_bundle"]["audited_route_pair_count"] == 2


def test_multi_fidelity_summary_emits_exact_evaluation_tags() -> None:
    proxy_only = build_multi_fidelity_summary(proxy_world_count=3, audit_world_count=0)
    partially_audited = build_multi_fidelity_summary(proxy_world_count=3, audit_world_count=1)
    fully_audited = build_multi_fidelity_summary(proxy_world_count=0, audit_world_count=2)
    reused_from_cache = build_multi_fidelity_summary(
        proxy_world_count=2,
        audit_world_count=0,
        cache_mode="warm",
    )
    corrected = build_multi_fidelity_summary(
        proxy_world_count=2,
        audit_world_count=1,
        proxy_correction_active=True,
    )

    assert proxy_only.certification_evaluation_tag == "proxy_only"
    assert partially_audited.certification_evaluation_tag == "partially_audited"
    assert fully_audited.certification_evaluation_tag == "fully_audited"
    assert reused_from_cache.certification_evaluation_tag == "reused_from_cache"
    assert corrected.certification_evaluation_tag == "corrected_from_residual_model"
