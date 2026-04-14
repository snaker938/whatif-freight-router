# Backend

WhatIf Freight Router is an auditable, tri-source, selective minimum-cost certification engine for freight-route recommendation under incomplete search, biased evidence, and ambiguous preferences.

The optimization objective is to minimize expected action cost to a justified terminal decision.

This directory contains the FastAPI service for the current freight-router runtime.

## Current runtime shape

- The default primary runtime path is the redesigned certification engine, with supported requests resolving through `dccs_refc`.
- `legacy` remains the explicit comparator path, and it still handles the current waypoint fallback while the default primary path stays on `dccs_refc`.
- The runtime keeps the three action families explicit and separate: search actions, evidence actions, and preference actions.
- The live route path returns a certification-oriented decision payload.
- Terminal outcomes are represented as certified singleton, certified set, or typed abstention.
- Typed abstention classes are `uncertified_due_to_search`, `uncertified_due_to_evidence`, `uncertified_due_to_preference`, `uncertified_due_to_out_of_support_world_model`, `uncertified_due_to_budget`, and `uncertified_due_to_model_assumption`.
- The top-level response model currently carries route, certificate, preference, support, witness, action-trace, and artifact metadata fields.
- The runtime writes artifact bundles for route computation, certificate/proof surfaces, preference state, support summaries, and provenance.

## Current evidence surfaces

- `DecisionPackage` is the public `/route` response contract. `RouteResponse` remains only as an internal compatibility model where older backend-only construction paths still need it.
- Preference state and preference query traces are exposed as runtime-visible payload fields and as artifacts.
- Certification/proof data includes certificate summary, certified-set summary, witness summary, and action-trace summary.
- Support/governance data includes support summaries and world-support summaries.

## What this README is and is not

- This is a factual map of the current codebase, not a publishability claim.
- The redesign is in progress, but broad hard gates are not yet asserted as green here.
- Any theorem, metric, or thesis claim must be checked against the backend tests, artifacts, and report text before being treated as complete.
