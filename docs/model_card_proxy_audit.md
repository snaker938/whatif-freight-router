# Proxy-Audit Model Card

This page documents the current proxy-audit / correction / propensity surface at the level the repository actually supports today.

It is intentionally conservative. It describes the current features, signals, and caveats, but it does **not** claim complete leakage-safe validation or unconditional overlap.

## Purpose

The repository uses proxy and audit evidence in support-aware routing and evaluation workflows. This card keeps the model-facing part of that story explicit:

- what signals the docs already mention
- what training or validation split language is documented
- what failure modes and caveats remain visible
- where the current evidence lives

## Current Surface

The current repo names support, fidelity, world-policy, and correction surfaces as explicit implementation areas.

Relevant backend modules include:

- `backend/app/support_model.py`
- `backend/app/fidelity_model.py`
- `backend/app/world_policies.py`
- `backend/app/audit_correction.py`
- `backend/app/uncertainty_model.py`
- `backend/app/risk_model.py`
- `backend/app/scenario.py`

The docs treat these as part of the support-aware routing stack rather than as a standalone machine-learning product.

## Signals And Features

The documentation already points to several signals that matter for proxy-audit and correction behavior:

- corridor family
- ambiguity regime
- support regime
- evidence-family regime
- engine-disagreement regime
- candidate density or pressure

The repo’s docs also reference the support/fidelity state family and the current runtime summaries that carry:

- support flags
- probabilistic world bundle summaries
- audit world bundle summaries
- proxy-only or audited provenance labels
- world-support summaries
- correction-related artifact names and summary surfaces

The key point is that the model surface is support-aware and evidence-family-aware, not a black-box one-number score.

## Documented Validation Language

The current docs do describe validation and calibration, but only at a bounded level.

What is documented:

- the support/fidelity/world-policy/correction surfaces are currently test-backed in the repo
- the quality-gates page records strict preflight evidence, current local validation, and benchmark evidence
- the claim matrix marks the support/fidelity/world-policy/correction surface as empirical
- the docs consistently point to checked artifacts and bundle outputs rather than a universal model guarantee

What is **not** documented as complete evidence:

- full leakage-safe validation for every proxy-audit path
- unconditional overlap in every regime
- theorem-backed guarantees for all correction behavior
- a universal statement that corrected estimates always dominate naive ones

## Failure Modes And Caveats

The docs explicitly warn that:

- live evidence can be stale, unavailable, biased, or incomplete
- proxy observations may stand in for fully observed ground truth
- support gaps and weak overlap can block or narrow what is safe to infer
- strict systems can fail closed rather than silently proceed
- ORS-like behavior may involve proxy fallback, which should not be confused with a fully validated reference engine

These are not defects hidden from the publication surface. They are part of the documented scope.

## Evidence And Artifact References

The current evidence for this surface lives in:

- `docs/thesis-codebase-report.md`
  - strictness and scope limitations
  - support and proxy caveats
  - local evidence claims
- `docs/quality-gates-and-benchmarks.md`
  - strict preflight evidence
  - CI lane definitions
  - minimum acceptance gates
  - quality thresholds
- `docs/model-assets-and-data-sources.md`
  - raw source families
  - live scenario and support assets
  - model asset provenance
- `docs/claim_matrix.md`
  - current empirical/scaffold-only status for support and correction surfaces
- `docs/theorem_map.md`
  - open theorem slots and non-theorem structural surfaces

Relevant runtime artifacts referenced by the docs include:

- `backend/out/model_assets/preflight_live_runtime.json`
- `backend/out/model_assets/routing_graph_coverage_report.json`
- `backend/out/corpus_ambiguity_refresh_summary.json`
- `backend/out/thesis_campaigns/*/evaluation_manifest.json`
- `backend/out/thesis_campaigns/*/thesis_metrics.json`
- `backend/out/thesis_campaigns/*/thesis_summary.json`
- `backend/out/thesis_campaigns/*/thesis_summary_by_cohort.json`

## What The Repo Can Claim Today

The strongest current claim is limited to this:

- the repository has explicit support, fidelity, world-policy, and correction surfaces
- the evaluator uses them in checked bundles and current docs
- the model card can point to those surfaces and their artifacts
- the current evidence is sufficient for scoped UK-focused analysis

It does **not** justify:

- complete leakage-free validation across all future data regimes
- universal calibration adequacy
- unconditional overlap or support sufficiency
- general-purpose deployment claims beyond the checked scope

## Review Questions

When reviewing this surface, ask:

- Is the claim tied to a checked artifact or only to a conceptual model?
- Is the validation language bounded to the current repo evidence?
- Does the doc distinguish proxy evidence from fully audited evidence?
- Are support and overlap caveats stated plainly?
- Is the claim scoped to the UK-focused thesis setting?

## Bottom Line

The current proxy-audit surface is explicit and test-backed at the implementation level, but the documentation only supports a cautious, scoped reading.

That is the correct publication posture for the current repository state: support-aware and evidence-rich, but not complete or universally validated.
