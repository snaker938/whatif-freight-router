# Frontend

WhatIf Freight Router is an auditable, tri-source, selective minimum-cost certification engine for freight-route recommendation under incomplete search, biased evidence, and ambiguous preferences.

The optimization objective is to minimize expected action cost to a justified terminal decision.

This directory contains the Next.js user interface for the current freight-router runtime.

## Current UI shape

- The main page consumes the live route payload and normalizes both flat route responses and wrapped decision-package responses.
- The default primary runtime path behind the UI is the redesigned certification engine, with supported requests resolving through `dccs_refc`.
- `legacy` remains the explicit comparator path, and the UI still reflects the current waypoint fallback to legacy while the default primary path stays on `dccs_refc`.
- The UI is inspecting a backend that keeps search actions, evidence actions, and preference actions explicit and separate.
- The visible shell is route-centric, but it now surfaces several read-only decision-state slices for certified singleton, certified set, and typed abstention outcomes.
- The typed abstention outcomes surfaced by the payload are `uncertified_due_to_search`, `uncertified_due_to_evidence`, `uncertified_due_to_preference`, `uncertified_due_to_out_of_support_world_model`, `uncertified_due_to_budget`, and `uncertified_due_to_model_assumption`.

## Current proof / governance surfaces

- `DecisionStateSummary` shows terminal type, selected certificate basis, certified-set visibility, support summary, preference runtime visibility, and abstention state.
- `RouteCertificationPanel` now shows controller context, support/governance fields, and an artifact-backed evidence-audit slice. It consumes the inline `world_support_summary` payload when present and augments it with proxied `sampled_world_manifest.json`, `route_fragility_map.json`, `value_of_refresh.json`, and `voi_stop_certificate.json`.
- Existing proof/navigation surfaces remain additive and read-only.

## What this README is and is not

- This is a conservative description of the current UI, not a claim that every redesign gate is green.
- The frontend exposes proof and governance summaries, but it does not provide interactive preference elicitation or controller editing.
- Keep new UI work aligned with the backend payloads already exposed by the live response contract.
