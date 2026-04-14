# Frontend

WhatIf Freight Router is an auditable, tri-source, selective minimum-cost certification engine for freight-route recommendation under incomplete search, biased evidence, and ambiguous preferences.

The optimization objective is to minimize expected action cost to a justified terminal decision. The controller keeps three action families explicit and separate: search actions, evidence actions, and preference actions.

This directory contains the Next.js user interface for the current freight-router runtime.

## Current UI shape

- The live public `/route` contract consumed by the main page is `DecisionPackage`.
- Any flat route-response normalization that still exists in the page is legacy/internal compatibility glue for older payload shapes, not the primary public contract.
- The default primary runtime path behind the UI is the redesigned certification engine, with supported requests resolving through `dccs_refc`.
- Live `/route` rejects `pipeline_mode=legacy` and rejects waypoint requests.
- Baseline, ablation, replay, and historical-comparison traffic belongs on `/route/baseline` and `/route/baseline/ors`.
- The UI is inspecting a backend that keeps search actions, evidence actions, and preference actions explicit and separate.
- The visible shell is route-centric, but it now surfaces several read-only decision-state slices for certified singleton, certified set, and typed abstention outcomes.
- In the checked local direct-REFC smoke slice, a public `typed_abstention` payload can still carry normalized `certified_set_summary`, `winner_confidence_state`, `certificate_witness`, and artifact-pointer surfaces while emitted bundles preserve richer local certified-set artifacts.
- The typed abstention outcomes surfaced by the payload are `uncertified_due_to_search`, `uncertified_due_to_evidence`, `uncertified_due_to_preference`, `uncertified_due_to_out_of_support_world_model`, `uncertified_due_to_budget`, and `uncertified_due_to_model_assumption`.

## Current proof / governance surfaces

- `DecisionStateSummary` remains the read-only preference-inspection companion in the route result view. It shows terminal type, selected certificate basis, certified-set visibility, support summary, abstention state, and the typed `preference_state`, `preference_query_trace`, and `preference_summary` payloads as compatible-set summary, query evidence, shrinkage-over-time, `why this query`, and explicit `why no preference query was asked` messaging when the backend skips elicitation.
- `PreferenceElicitationPanel` is the live preference action surface alongside that summary. It consumes the current route/runtime payload and visible route metrics, supports pairwise, tradeoff-threshold, tradeoff-ratio, veto, and time-preserving guard questions, and syncs compatible-set updates and shrinkage trace payloads through the backend `/route/preference` runtime path rather than mock data.
- `RouteCertificationPanel` now shows controller context, an artifact-backed controller trace, support/governance fields, and an artifact-backed evidence-audit slice. It consumes the inline `world_support_summary` payload when present, augments it with proxied `voi_action_trace.json`, `voi_action_scores.csv`, `voi_stop_certificate.json`, `sampled_world_manifest.json`, `route_fragility_map.json`, and `value_of_refresh.json`, and offers frontend-generated CSV and SVG exports for the visible decision card, controller trace, and evidence summary. Those SVG exports are vector figure surfaces intended for PDF-ready placement, not direct PDF generation.
- `ProofDashboardPanel` is a separate proof-dashboard surface on the main page. It groups proof slices for `V0/A/B/C`, broad vs focused, cold vs hot, OSRM vs ORS, and theorem-to-artifact navigation, with direct artifact links and `RunInspector` entrypoints on the same tiles.
- The setup card now exposes canned proof-demo presets for safe singleton, certified set, support abstention, preference-sensitive, collapse-prone, and hot-rerun inspection. These presets only prefill existing request knobs; they do not fabricate backend outcomes.
- The proof dashboard can copy a text summary, export a CSV snapshot of the visible proof tiles, and generate a deterministic witness-driven explanation from the live payload fields. Full SVG/PDF-ready figure export is still a follow-up surface.
- Existing proof/navigation surfaces remain additive and read-only.

## What this README is and is not

- This is a conservative description of the current UI, not a claim that every redesign gate is green.
- The frontend now pairs read-only preference-trace visibility with a live preference elicitation panel that round-trips committed answers through the backend runtime; it still does not provide full controller editing or persisted cross-run preference workflow state.
- Keep new UI work aligned with the backend payloads already exposed by the live response contract.
