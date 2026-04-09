# Evaluation Card

This page is the evaluator-facing reference for the current thesis-style backend lanes, cohort labels, and checked artifacts.

It is intentionally conservative. It explains what the evaluator covers, where to look for current evidence, and what is still only partially evidenced or explicitly scaffolded.

## Purpose

The evaluator in this repository is not a single monolithic benchmark. It is a set of named lanes and evidence bundles that support different parts of the thesis pipeline:

- DCCS diagnostics
- REFC certification and fragility
- VOI controller behavior
- replay and hot-rerun checks
- preference, support, and calibration surfaces
- transfer and synthetic sanity checks

Use this page as a map, not as a claim of completion.

## Lane Definitions

The current docs and scripts support the following lane names:

- `broad cold proof`
- `focused REFC proof`
- `focused VOI proof`
- `DCCS diagnostic probe`
- `hot-rerun cold-source proof`
- `hot-rerun proof`
- `preference proof`
- `optional-stopping coverage`
- `proxy-audit calibration`
- `perturbation / flip-radius`
- `public transfer`
- `synthetic ground-truth`

These lane names are grounded in the quality-gates page and the thesis-pipeline spec. They are the right vocabulary for the current evaluator, but they should still be read as lane labels rather than universal theorem names.

## Cohort Semantics

The current evaluator and report surfaces reference the following cohorts or cohort-like slices:

- `collapse_prone`
- `osrm_brittle`
- `ors_brittle`
- `refresh_sensitive`
- `time_preserving_conflict`
- `low_ambiguity_fast_path`
- `preference_sensitive`
- `support_fragile`
- `audit_heavy`
- `proxy_friendly`

These cohort labels are useful because they keep the evidence organized around failure-sensitive cases rather than only around aggregate averages.

What is currently evidenced:

- the docs and report explicitly name these cohorts as part of the evaluation story
- the quality-gates page records current local artifacts and thesis bundles
- the claim matrix and theorem map distinguish scaffold-only surfaces from empirical ones

What is not yet evidenced as a blanket claim:

- that every cohort meets every headline threshold
- that every lane has the same maturity or the same completeness
- that cohort behavior generalizes outside the checked bundles

## What The Evaluator Currently Shows

The strongest current evidence is recorded in the following places:

- `docs/quality-gates-and-benchmarks.md`
  - latest local validation snapshot
  - CI lane definitions
  - minimum acceptance gates
  - quality thresholds
- `docs/thesis-codebase-report.md`
  - thesis-bundle evidence
  - limitations and scope notes
  - what the report does not overclaim
- `docs/sample-manifest.md`
  - run outputs and artifact bundle shape
  - manifest and report file names
- `docs/reproducibility-capsule.md`
  - repro controls and artifact provenance anchors
- `docs/claim_matrix.md`
  - current surfaces marked `scaffold-only`, `empirical`, or `theorem-backed`
- `docs/theorem_map.md`
  - theorem slots that are still open

The evaluator is therefore best understood as a set of evidence surfaces with different maturity levels, not as a single all-green scorecard.

## What Is Evidenced Now

The current repo does support the following evaluator-facing facts:

- strict preflight and readiness checks are present and produce checked local artifacts
- thesis-lane bundles exist with report and summary outputs
- the DCCS, REFC, and VOI pipeline vocabulary is explicit in docs and code
- current run-store artifacts include DCCS, REFC, VOI, support, and preference summaries
- the docs deliberately separate scaffold-only surfaces from empirical ones

Current artifact examples referenced by the docs:

- `backend/out/model_assets/preflight_live_runtime.json`
- `backend/out/model_assets/routing_graph_coverage_report.json`
- `backend/out/thesis_campaigns/*/campaign_report.md`
- `backend/out/thesis_campaigns/*/thesis_summary.json`
- `backend/out/thesis_campaigns/*/thesis_metrics.json`
- `backend/out/thesis_campaigns/*/methods_appendix.md`
- `backend/out/thesis_campaigns/*/evaluation_manifest.json`

## What Is Not Yet Green

Do not treat this repository slice as fully green on every published gate.

The current docs still acknowledge:

- limitations in scoped or universal claims
- support-sensitive failures that can still fail closed
- bounded local evidence rather than universal generalization
- scaffold-only REFC and DCCS surfaces in the current slice

That means this page should be used to orient reviewers, not to imply a completed proof package.

## Where Reviewers Should Look

For current evidence, reviewers should start with:

1. `docs/quality-gates-and-benchmarks.md` for the latest local validation and gate vocabulary
2. `docs/thesis-codebase-report.md` for the thesis narrative, limitations, and checked bundle references
3. `docs/sample-manifest.md` for artifact bundle names and retrieval flow
4. `docs/reproducibility-capsule.md` for provenance and repro anchors
5. `docs/claim_matrix.md` and `docs/theorem_map.md` for claim status and proof gaps

For backend scripts, the evaluator-facing entry points are currently:

- `backend/scripts/preflight_live_runtime.py`
- `backend/scripts/score_model_quality.py`
- `backend/scripts/benchmark_model_v2.py`
- `backend/scripts/benchmark_batch_pareto.py`
- `backend/scripts/validate_graph_coverage.py`
- `backend/scripts/run_thesis_evaluation.py`
- `backend/scripts/run_thesis_lane.py`

## Suggested Review Framing

When reviewing evaluator output, use these questions:

- Which lane produced the artifact?
- Which cohort slice does the row or bundle belong to?
- Is the evidence a checked artifact, a scaffold-only surface, or a theorem-backed claim?
- Does the doc point to a concrete file in `backend/out/` or only to a conceptual surface?
- Are the claims scoped to the observed bundle, or do they drift toward universal language?

## Bottom Line

The evaluator is already rich enough to support thesis-style review, but not every lane or cohort is fully mature.

The safe publication reading is:

- the lanes are explicit
- the cohorts are explicit
- the artifact paths are explicit
- the claim boundaries are still conservative

That is the right level of honesty for the current repository state.
