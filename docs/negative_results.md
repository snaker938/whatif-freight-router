# Negative Results

This document records where the current repository does **not** support strong claims, where the evidence is still partial, and where the repo/evaluation docs should stay scoped.

It is intentionally conservative. It is a negative-evidence companion to the positive evidence pages, not a replacement for them.

## What This Document Covers

The focus is on:

- unsupported or partially supported claim areas
- gate shortfalls and unproven thresholds
- failure modes that remain explicit in the code and docs
- scope limits that must stay visible in publication-facing writing

## What It Does Not Cover

This page does not repeat the main system description, the full certification/evaluation pipeline, or the artifact catalog.

For those surfaces, see:

- [Evaluation Card](evaluation_card.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
- [Claim Matrix](claim_matrix.md)
- [Theorem Map](theorem_map.md)

## Unsupported Or Unproven Claims

The current repo does **not** justify these as universal or unconditional claims:

- universal superiority over OSRM
- universal superiority over ORS
- unconditional validity outside supported UK-style scope
- unconditional deployability under all live-source conditions
- theorem-backed guarantees for the current REFC surfaces
- complete reviewer-package materialization or figure-ready PDF/SVG export for every indexed table/figure surface in the current checked slice

The maintained evaluation docs say this in different places. This page keeps the limitation visible in one place.

## Current Gate Shortfalls

The quality-gates page records the target gates and the latest evidence bundle, but not every gate is at a publishable threshold.

For this doc set, the safe default is: every `G11.*` and `P14.*` item is open unless another page cites a current artifact path, the measured value, the required threshold, and the required sample size. The reviewed pages in this slice do not yet do that for the full redesign gate set.

The current evidence set still leaves open:

- the checked full-suite companion is green for the applied-evidence publication/adoption framing: `publishable_on_current_evidence=true`, `adoption_claim_supported=true`, while DCCS, refine-cost forecasting, and VOI remain non-blocking algorithm diagnostics for strong certification
- do not state blanket green status for all `G11.*` or `P14.*` rows; a row is closed only when a maintained page cites the current artifact path, measured value, required threshold, and required sample size
- current checked full-suite evidence no longer treats sample size, optional stopping, perturbation, headline narrowing, hot rerun, or `P14.46-P14.50` runtime/report observability as negative results
- repeated-seed focused-VOI preference-burden support for `P14.17-P14.20` comes from `out/artifacts/full_suite_curated_latest_20260411_focused_voi_proof/` plus `_seed20260421` and `_seed20260522`; that family-specific closure supports the applied suite framing but does not make the overall suite deployment-ready or strong-certification-complete
- the staged `out/headline_exports/current_checked/table.focused_voi.preference_burden_*` source/provenance sidecars still cite the older focused `thesis_eval_20260331_r2_focused_voi/` single-seed bundle, so those staged sidecars should not be cited as repeated-seed proof until regenerated
- the older checked campaign baseline-fairness reviewer lane remains historical contrast; the maintained full-suite reviewer bundle now reports `fairness_failure_count=0`, so current adoption/publication wording should cite the full-suite verdict rather than that older campaign slice

The safe reading is that the repo can demonstrate credible local evidence, but not blanket success across every gate or cohort.

## Evidence-Bounded Limitations

The maintained evaluation docs explicitly frame several constraints as limitations rather than solved problems:

- UK-only operational scope
- unsupported-region behavior remains explicit
- baseline comparisons are local and scoped, not universal
- strict live preflight and readiness checks can fail when required inputs are stale or missing
- some model or asset families remain sensitivity points rather than fully closed proofs

This matters because publication text should not collapse these limits into a generic success narrative.

## Failure-Mode Summary

The repository’s failure handling is one of its strengths, but it is still evidence of limits rather than evidence of perfection.

Examples of negative or bounded behavior that remain visible in the docs and runtime:

- route requests can fail closed when live evidence is unavailable or unsupported
- terrain support can be insufficient for strict operation
- baseline providers can be unavailable or configured as proxies only
- strict readiness gates can stop the workflow early
- candidate triage and certification surfaces are explicit, but not yet theorem-backed in the current slice

## What To Say In External-Facing Evaluation Text

Use scoped language such as:

- the repository provides local evidence for the evaluated bundles
- the system is designed to fail closed under missing support
- the current evaluation claims are bounded by the checked artifacts and gates
- the REFC and DCCS surfaces are explicit implementation contracts, and the named theorem package remains `partial-proof` rather than `theorem-backed` in this slice

Avoid language such as:

- universal
- always
- provably optimal
- complete
- all conditions
- unconditional
- all required lanes are implemented
- `G11` is green
- `P14` is green
- seed-robust
- publishable

## Evidence Index

Use the following surfaces when writing or reviewing this document:

- `README.md`
  - Documentation section
  - Publication-facing docs links
- `docs/DOCS_INDEX.md`
  - docs navigation and maintenance hub
- `docs/evaluation_card.md`
  - `## What Is Not Yet Green`
  - `## Where Reviewers Should Look`
- `docs/quality-gates-and-benchmarks.md`
  - `## Latest Local Validation`
  - `## CI Lanes (Authoritative)`
  - `## Minimum Acceptance Gates`
  - `## Quality Thresholds`
- `docs/claim_matrix.md`
  - current slice rows marked `empirical`, `conditional`, or `descriptive`
  - `proved` rows are not asserted for the current slice
- `docs/theorem_map.md`
  - theorem slots that remain open
  - explicit structural surfaces that are not yet theorem-backed

Related publication docs:

- `docs/evaluation_card.md`
- `docs/data_card.md`
- `docs/model_card_proxy_audit.md`

## Minimal Review Checklist

Before treating a claim as publication-ready, check that:

- the claim appears in the positive evidence pages first
- a limitation or gate note exists if the claim is scoped
- any REFC or DCCS language is tied to actual checked artifacts or tests
- no universal wording slips in where the evidence is only local

## Bottom Line

The current repository supports a careful, evidence-backed story about a scoped UK freight-routing system with explicit failure modes and documented gates.

It does **not** yet support a blanket story of universal dominance, unconditional validity, or theorem-backed certainty for the current redesign surfaces.

## Related Docs

- [Claim Matrix](claim_matrix.md)
- [Evaluation Card](evaluation_card.md)
