# Claim Matrix

This document separates what the current slice actually supports from what is only scaffolded or still absent.

Legend:

- `theorem-backed`: a claim with a formal proof package, or an equally explicit theorem/proposition contract
- `empirical`: supported by checked artifacts, tests, or local evidence bundles
- `scaffold-only`: present in code or docs as a structural surface, but not yet backed by a formal proof package or a completed evaluation gate

## Current Slice

| Surface | Status | Claim class | Evidence | Notes |
| --- | --- | --- | --- | --- |
| README identity and thesis-pipeline framing | scaffold-only | descriptive | `README.md` | The README now states the thesis path is explicit backend support, not the default user-facing route flow |
| `docs/voi-pipeline-spec.md` pipeline modes and candidate lifecycle | scaffold-only | descriptive | `docs/voi-pipeline-spec.md` | The spec documents the current contract; it does not assert theorem-level guarantees |
| `backend/app/models.py` `DecisionPackage` and route response fields | scaffold-only | structural | `backend/app/models.py`, `backend/app/main.py` | The response shape is explicit, but the slice does not turn it into a theorem claim |
| `backend/app/run_store.py` artifact family and schema versioning | empirical | artifact contract | `backend/app/run_store.py`, `backend/tests/test_run_store_artifacts.py` | The artifact name set and `schema_version` behavior are asserted by test |
| Preference-state, query, and shrinkage surfaces | empirical | preference contract | `backend/app/preference_state.py`, `backend/app/preference_model.py`, `backend/app/preference_queries.py`, `backend/app/preference_update.py`, `backend/tests/test_preference_surface.py` | The preference state is serializable, the current tests lock the conservative query/shrinkage helpers, and the current run-store names include `preference_state.json` and `preference_query_trace.json` |
| Support, fidelity, world-policy, and correction surfaces | empirical | support/fidelity contract | `backend/app/support_model.py`, `backend/app/fidelity_model.py`, `backend/app/world_policies.py`, `backend/app/audit_correction.py`, `backend/tests/test_support_fidelity_world_models.py` | The support/world bundles and action-value metadata are currently test-backed implementation surfaces, with `world_support_summary.json` as the current run-store summary name |
| DCCS candidate envelope and safe-elimination provenance fields | scaffold-only | structural | `backend/app/decision_critical.py`, `docs/voi-pipeline-spec.md`, `docs/thesis-codebase-report.md` | These are current audit fields, not a formal safe-elimination theorem in this slice |
| REFC certificate / fragility / stop-certificate exposure | scaffold-only | structural | `backend/app/evidence_certification.py`, `backend/app/main.py`, `docs/voi-pipeline-spec.md` | The surfaces exist, but the report should still treat them as implementation evidence rather than theorem proof |
| REFC artifact contract and replay surfaces | empirical | artifact contract | `backend/app/run_store.py`, `backend/app/evidence_certification.py`, `backend/tests/test_refc_artifact_contract.py` | Current artifact names include `decision_package.json`, `winner_confidence_state.json`, `pairwise_gap_state.json`, `flip_radius_summary.json`, `decision_region_summary.json`, `certificate_witness.json`, and `certified_set_summary.json` |
| Live-trace support, fidelity, and terminal metadata | empirical | trace contract | `backend/app/live_call_trace.py`, `backend/tests/test_route_cache_live_trace_state.py` | Live trace snapshots currently carry support, fidelity, and terminal-type metadata for inspection rather than exposing a theorem-backed claim |
| Thesis report and pipeline docs bundle names | empirical | evidence bundle | `docs/thesis-codebase-report.md`, `docs/voi-pipeline-spec.md` | The docs cite checked local bundles and current artifact names |

## Theorem-Backed Surface

No theorem-backed claim is asserted for this slice.

If a later slice introduces a proof package, add a row here with:

- theorem or proposition id
- assumptions
- code surface
- artifact field(s)
- test coverage
- evaluator metric(s)

## Notes On Discipline

- Treat `scaffold-only` rows as implementation descriptions, not publication claims.
- Treat `empirical` rows as current evidence, not universal guarantees.
- Do not promote any row to `theorem-backed` unless the code, tests, and report all name the same claim explicitly.
