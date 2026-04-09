# Theorem Map

This repository slice does not yet publish a formal theorem package for the thesis pipeline. The map below therefore distinguishes the current supported surfaces from the proof slots that remain open.

Legend:

- `theorem-backed`: formally stated and supported by a proof package
- `empirical`: supported by tests, checked artifacts, or local evidence bundles
- `scaffold-only`: present as code or documentation structure, but not yet promoted to a formal claim

| ID | Claim family | Status | Code / doc surface | Test / artifact support | Notes |
| --- | --- | --- | --- | --- | --- |
| TM-00 | No theorem package is asserted for the current slice | scaffold-only | `docs/claim_matrix.md` | none | Placeholder entry so later theorem claims have an explicit home |
| TM-01 | Run-store artifact names and `schema_version` are stable enough to inspect and replay | empirical | `backend/app/run_store.py`, `backend/app/models.py` | `backend/tests/test_run_store_artifacts.py` | Verified in the current local packet |
| TM-02 | The thesis pipeline exposes explicit `legacy`, `dccs`, `dccs_refc`, and `voi` modes | scaffold-only | `README.md`, `docs/voi-pipeline-spec.md`, `backend/app/models.py` | existing serialization and runtime paths | Mode exposure is structural, not a theorem claim |
| TM-03 | DCCS candidate records carry envelope and safe-elimination provenance fields | scaffold-only | `backend/app/decision_critical.py`, `docs/voi-pipeline-spec.md`, `docs/thesis-codebase-report.md` | none in this packet | Treat as audited implementation detail until a proof package exists |
| TM-04 | REFC exposes certificate, fragility, and stop-certificate surfaces | scaffold-only | `backend/app/evidence_certification.py`, `backend/app/main.py`, `docs/voi-pipeline-spec.md` | current runtime artifact contract | Evidence surface only; no theorem-level guarantee is claimed here |
| TM-05 | `DecisionPackage` / route response fields make singleton, set, and abstention states explicit | scaffold-only | `backend/app/models.py`, `backend/app/main.py` | response serialization paths | Structural API contract, not a proof claim |
| TM-06 | Preference-state, query, and shrinkage surfaces are explicit and test-backed | empirical | `backend/app/preference_state.py`, `backend/app/preference_model.py`, `backend/app/preference_queries.py`, `backend/app/preference_update.py` | `backend/tests/test_preference_surface.py` | Current slice proves the serializable contract and conservative helper behavior, and the run-store names include `preference_state.json` and `preference_query_trace.json` |
| TM-07 | Support, fidelity, world-policy, and correction surfaces are explicit and test-backed | empirical | `backend/app/support_model.py`, `backend/app/fidelity_model.py`, `backend/app/world_policies.py`, `backend/app/audit_correction.py` | `backend/tests/test_support_fidelity_world_models.py` | These are current model and metadata surfaces, and the corresponding summary name is `world_support_summary.json` |
| TM-08 | REFC artifact names and replay surfaces are explicit and test-backed | empirical | `backend/app/run_store.py`, `backend/app/evidence_certification.py` | `backend/tests/test_refc_artifact_contract.py` | Includes `decision_package.json`, `winner_confidence_state.json`, `pairwise_gap_state.json`, `flip_radius_summary.json`, `decision_region_summary.json`, `certificate_witness.json`, and `certified_set_summary.json` |
| TM-09 | Live trace support, fidelity, and terminal metadata are explicit and test-backed | empirical | `backend/app/live_call_trace.py` | `backend/tests/test_route_cache_live_trace_state.py` | Trace state is inspectable and replayable, but it remains a structural surface rather than a theorem-backed guarantee |

## How To Extend This Map

When a later slice adds a genuine theorem or proposition, add a row with:

- a stable claim id
- the theorem statement or proposition name
- assumptions and scope
- code surface(s)
- at least one unit test
- at least one negative or property test
- at least one artifact field
- at least one evaluator metric

Until then, keep the map conservative and do not retrofit theorem language onto scaffold-only surfaces.
