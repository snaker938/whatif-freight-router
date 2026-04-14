# Claim Matrix

WhatIf Freight Router is an auditable, tri-source, selective minimum-cost certification engine for freight-route recommendation under incomplete search, biased evidence, and ambiguous preferences.

This document is the detailed claim register for the current repository slice. It distinguishes certification-facing claims that are actually supported today from descriptive framing and still-open proof surfaces. It records implementation evidence and claim discipline, while `docs/theorem_map.md` separately tracks formal proof maturity.

## Claim Status Language

- `theorem-backed`: a claim with a formal proof package, or an equally explicit theorem/proposition contract.
- `empirical`: supported by checked artifacts, tests, code paths, or local evidence bundles.
- `heuristic-but-measured`: an engineering or heuristic component with checked measurement, artifact, or evaluator support, but no theorem-backed guarantee.
- `non-claim / descriptive only`: framing, navigation, negative scope, or implementation description rather than a performance, validity, or theorem claim.

When a row relies on support, calibration, or model assumptions, that conditionality is called out explicitly in the Notes field rather than by introducing a separate fifth primary status.

`docs/theorem_map.md` is the proof-maturity ledger. This matrix is the implementation-evidence ledger.
When no theorem or proposition id is currently published for a row, this file marks that explicitly as `none-published-in-this-slice` rather than fabricating one. Artifact-path cells use one of four explicit forms: exact checked local artifact files, exact reviewer-package index entries, artifact-family globs for runtime surface families, or an explicit `not-applicable` marker.

## Claim Discipline

- Treat certification guarantees as conditional when they rely on support, calibration, or model assumptions; record that conditionality in Notes rather than as a separate primary status label.
- Treat empirical rows as slice-scoped evidence, not universal dominance or universal validity claims.
- Treat `heuristic-but-measured` rows as measured engineering surfaces rather than theorem-backed guarantees.
- Treat `non-claim / descriptive only` rows as repository truth and navigation aids, not proof of metric gates.
- Treat any still-open `G11.*` or `P14.*` requirement as open unless a current artifact path closes it explicitly.
- Do not promote any row to `theorem-backed` unless the code, tests, artifacts, and report all name the same claim explicitly.

## Current Certification Slice

| Surface | Claim status | Theorem / proposition id | Evaluator metric(s) | Artifact path(s) | Evidence surfaces | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Certification-centric repository identity and scope | `non-claim / descriptive only` | `not-applicable` | `not-applicable` | `not-applicable (doc framing only)` | `claim_matrix.md`, `docs/thesis-codebase-report.md` | Identity statement only; not a performance or theorem claim. |
| The optimization objective is to minimize expected action cost to a justified terminal decision | `non-claim / descriptive only` | `not-applicable` | `not-applicable (objective-framing claim)` | `not-applicable (report framing only)` | `../docs/thesis-codebase-report.md`, `../claim_matrix.md` | Opening objective statement from the maintained report; it frames the system's decision target rather than claiming a metric win. |
| Search actions, evidence actions, and preference actions are kept explicit and separate throughout the redesign | `empirical` | `none-published-in-this-slice` | `voi_action_count`, `search_budget_used`, `evidence_budget_used` | `backend/out/**/voi_action_trace.json`, `backend/out/**/voi_stop_certificate.json`, `backend/out/**/index.json` | `backend/app/voi_controller.py`, `backend/app/models.py`, `backend/tests/test_decision_package_summary_payload.py` | Runtime/control-surface claim about explicit action-family separation. |
| Typed abstention classes are explicit named outcomes rather than free-form text | `empirical` | `none-published-in-this-slice` | `not-applicable (API/runtime contract claim)` | `backend/out/**/certificate_summary.json`, `backend/out/**/index.json` | `backend/app/abstention.py`, `backend/app/models.py`, `backend/tests/test_route_terminal_semantics.py` | Narrow claim about named abstention classes, separate from the broader terminal-outcome row below. |
| The project is UK-focused and implemented as a hybrid of a prepared OSRM engine, a FastAPI modeling backend, a Next.js frontend, and a calibrated asset layer | `non-claim / descriptive only` | `not-applicable` | `not-applicable` | `not-applicable (system-framing claim)` | `../docs/thesis-codebase-report.md`, `backend/app/main.py`, `frontend/app/page.tsx` | Opening architecture/scope sentence from the maintained report. |
| The system is framed as choosing the better operational decision under cost, delay risk, terrain, tolls, weather, live pressure, carbon, and user preferences rather than only the shortest road path | `non-claim / descriptive only` | `not-applicable` | `not-applicable` | `not-applicable (plain-English framing only)` | `../docs/thesis-codebase-report.md`, `../claim_matrix.md` | Plain-English decision framing claim from the report opening. |
| The backend is intentionally fail-closed across major readiness and live-data subsystems, preferring explicit reason codes over silent fallback from stale or unsupported data | `empirical` | `none-published-in-this-slice` | `strict_live_readiness_pass_rate`, `scenario_profile_unavailable_rate`, `route_evidence_ok_rate` | `backend/out/**/repo_asset_preflight.json`, `backend/out/**/evaluation_manifest.json`, `backend/out/**/index.json` | `backend/app/main.py`, `backend/app/model_data_errors.py`, `docs/strict-errors-reference.md` | Strictness/availability claim about current fail-closed behavior. |
| `dccs_refc` is the primary live `/route` path for thesis-facing non-waypoint requests; live `/route` rejects `pipeline_mode=legacy` and rejects waypoint requests, directing comparison traffic to `/route/baseline` and `/route/baseline/ors` | `empirical` | `none-published-in-this-slice` | `not-applicable (runtime-contract claim)` | `not-applicable (HTTP/runtime default rather than run artifact)` | `backend/app/settings.py`, `backend/app/main.py`, `docs/redesign-implementation-tracker.md` | Default-path claim narrowed to the current live `/route` slice. |
| User-facing terminal outcomes are limited to certified singleton, certified set, or typed abstention | `empirical` | `none-published-in-this-slice` | `not-applicable (API/runtime contract claim)` | `backend/out/**/certificate_summary.json`, `backend/out/**/certified_set_summary.json`, `backend/out/**/index.json` | `backend/app/models.py`, `backend/app/abstention.py`, `backend/tests/test_route_terminal_semantics.py` | Explicit terminal semantics claim. |
| `DecisionPackage` exposes frontier, certificate, stability, preference, support, abstention, witness, and artifact summary fields, and the live `/route` response surface now returns that shape directly while retaining compatibility fields for the current UI | `empirical` | `none-published-in-this-slice` | `not-applicable (API/schema contract claim)` | `not-applicable (public response contract rather than artifact file)` | `backend/app/models.py`, `backend/app/main.py` | Response-shape claim aligned to the current endpoint declaration and payload assembly. |
| Run-store artifact families and schema versioning are explicit and test-backed | `empirical` | `none-published-in-this-slice` | `evaluation_rerun_success_rate`, `required_artifact_count` | `backend/out/**/index.json`, `backend/out/**/index.md`, `backend/out/**/evaluation_manifest.json` | `backend/app/run_store.py`, `backend/tests/test_run_store_artifacts.py` | Current artifact inventory and schema-version behavior are asserted by test. |
| Named thesis pipeline modes (`legacy`, `dccs`, `dccs_refc`, `voi`) plus DCCS, REFC, and a VOI-style controller are explicit code-level surfaces rather than implicit script-only behavior | `empirical` | `none-published-in-this-slice` | `mean_certificate`, `voi_controller_engagement_rate`, `evaluation_rerun_success_rate` | `backend/out/**/evaluation_manifest.json`, `backend/out/**/thesis_summary.json`, `backend/out/**/index.json` | `backend/app/settings.py`, `backend/app/decision_critical.py`, `backend/app/evidence_certification.py`, `backend/app/voi_controller.py` | Pipeline-explicitness claim for the current thesis-facing runtime slice. |
| Academically recognizable methods are adapted into transparent engineering blends, and the modified profiles are not claimed here as novel theory | `non-claim / descriptive only` | `not-applicable` | `not-applicable` | `not-applicable (method-framing claim)` | `../docs/thesis-codebase-report.md`, `dissertation-math-overview.md`, `../claim_matrix.md` | Opening methods-framing sentence from the maintained report; it narrows theory claims rather than asserting a theorem result. |
| The current checked reviewer-facing headline evidence family includes the focused VOI variant summary table, cohort summary table, certificate-vs-variant figure source, and runtime-vs-variant figure source | `empirical` | `none-published-in-this-slice` | `not-applicable (reviewer-facing evidence-family claim)` | `../paper_artifact_index.json`, `reviewer_quickstart.md#focused-voi-headline-table-and-figure-commands` | `../paper_artifact_index.json`, `reviewer_quickstart.md` | Scoped only to the current checked local slice indexed in `paper_artifact_index.json`. |
| The current checked reviewer package maps the focused-VOI exported surfaces through the quickstart command block and maps the additional indexed focused-VOI aggregate table, focused preference-burden tables, and campaign-backed table/figure source surfaces through the root artifact index plus reviewer quickstart notes | `empirical` | `none-published-in-this-slice` | `not-applicable (reviewer-package traceability claim)` | `../paper_artifact_index.json`, `reviewer_quickstart.md#focused-voi-headline-table-and-figure-commands`, `reviewer_quickstart.md#focused-voi-additional-table-commands`, `reviewer_quickstart.md#latest-checked-campaign-table-and-figure-commands` | `../paper_artifact_index.json`, `reviewer_quickstart.md` | Narrow reviewer-traceability claim for the current checked local slice only; the quickstart export block does not materialize every indexed surface, while the newer checked full-suite assessment reports `publishable_on_current_evidence=false`, `adoption_claim_supported=false`, and `hot_rerun_all_green=true` at suite scope with blockers `dccs_hard_gates_not_all_green`, `refine_cost_forecast_gates_not_all_green`, and `voi_hard_gates_not_all_green`. |
| Preference-state, query, and shrinkage surfaces are explicit runtime objects | `empirical` | `none-published-in-this-slice` | `preference_query_count`, `preference_shrinkage`, `necessary_best_prob`, `possible_best_prob` | `backend/out/**/preference_state.json`, `backend/out/**/preference_query_trace.json`, `backend/out/**/certificate_summary.json` | `backend/app/preference_state.py`, `backend/app/preference_model.py`, `backend/app/preference_queries.py`, `backend/app/preference_update.py`, `backend/tests/test_preference_surface.py` | Structural and serialization claim for the current preference slice. |
| Support, fidelity, world-policy, and proxy/audit correction surfaces are explicit runtime objects | `empirical` | `none-published-in-this-slice` | `proxy_only_fraction`, `audit_world_count`, `support_flag` | `backend/out/**/world_support_summary.json`, `backend/out/**/sampled_world_manifest.json`, `backend/out/**/certificate_summary.json` | `backend/app/support_model.py`, `backend/app/fidelity_model.py`, `backend/app/world_policies.py`, `backend/app/audit_correction.py`, `backend/tests/test_support_fidelity_world_models.py` | This surface is runtime-explicit, but certification claims remain conditional on support, calibration, and overlap assumptions. |
| DCCS candidate-envelope and safe-elimination fields are exposed for audit and diagnosis | `non-claim / descriptive only` | `none-published-in-this-slice` | `safe_prune_rate`, `false_safe_prune_rate`, `search_completeness_score` | `backend/out/**/dccs_candidates.jsonl`, `backend/out/**/dccs_summary.json` | `backend/app/decision_critical.py`, `docs/voi-pipeline-spec.md`, `docs/thesis-codebase-report.md` | Exposed implementation surface, not a theorem claim in this slice. |
| REFC certificate, fragility, witness, and stop-certificate surfaces are exposed in runtime artifacts | `empirical` | `none-published-in-this-slice` | `mean_certificate`, `certificate_lcb`, `flip_radius_violation_rate`, `voi_controller_engagement_rate` | `backend/out/**/certificate_summary.json`, `backend/out/**/route_fragility_map.json`, `backend/out/**/certificate_witness.json`, `backend/out/**/voi_stop_certificate.json` | `backend/app/evidence_certification.py`, `backend/app/main.py`, `backend/tests/test_refc_artifact_contract.py` | Artifact-contract claim; any validity reading remains conditional on support, calibration, and the checked evidence family named in the linked artifacts. |
| Live trace captures support, fidelity, and terminal metadata for inspection | `empirical` | `none-published-in-this-slice` | `live_call_count`, `cache_hit_count`, `trace_row_count` | `backend/out/**/final_route_trace.json`, `backend/out/**/index.json` | `backend/app/live_call_trace.py`, `backend/tests/test_route_cache_live_trace_state.py` | Inspection/replay claim. |
| Hard redesign gates remain separate from this matrix and require current evidence before being called green | `non-claim / descriptive only` | `not-applicable` | `G11.*`, `P14.*` tracked externally | `not-applicable (claim-discipline rule)` | `docs/quality-gates-and-benchmarks.md` | Prevents this matrix from being mistaken for a gate report. |

## Current Reviewer Package Coverage

The current checked reviewer slice directly covers only the indexed summary and plot source surfaces listed in [`../paper_artifact_index.json`](../paper_artifact_index.json) and reproduced through [`reviewer_quickstart.md#focused-voi-headline-table-and-figure-commands`](reviewer_quickstart.md#focused-voi-headline-table-and-figure-commands).

All other rows in this file remain backed by wider code, test, API, or artifact-family evidence rather than by that exact reviewer package slice.

The focused reviewer slice is also single-seed only. The checked bundle records `seed_repeat_plan.headline_seed_repeat_required = true`, `configured_seed_count = 1`, and `status = single_seed_only`, so the current focused `C` evidence should not be read as seed-robust or threshold-stable publication evidence on its own.

## Theorem-Backed Surface

No `theorem-backed` claim is asserted for this slice.

That is a theorem-maturity statement, not a claim that the implementation evidence rows above are missing. The matrix above can record real code, test, and artifact evidence without promoting any row to `theorem-backed` status.

The named `partial-proof` package for the current slice now lives in [`theorem_package.md`](theorem_package.md) and [`theorem_map.md`](theorem_map.md). If a later slice promotes a family to `theorem-backed`, add or update the corresponding row here with:

- theorem or proposition id
- assumptions
- code surface
- artifact field(s)
- test coverage
- evaluator metric(s)

## Notes On Discipline

- Treat `non-claim / descriptive only` rows as implementation descriptions, not publication claims.
- Treat `empirical` rows as current evidence, not universal guarantees.
- Treat assumption-dependent rows as evidence surfaces that still depend on named assumptions and therefore must not be read as unconditional guarantees, even when their primary status is `empirical`.
- Do not promote any row to `theorem-backed` unless the code, tests, artifacts, and report all name the same claim explicitly.
- For reviewer-facing navigation, start at `claim_matrix.md` and then use `docs/DOCS_INDEX.md`.

## Related Docs

- [Top-Level Claim Matrix](../claim_matrix.md)
- [Theorem Map](theorem_map.md)
- [Evaluation Card](evaluation_card.md)
