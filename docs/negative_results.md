# Negative Results

This document records where the current repository does **not** support strong claims, where the evidence is still partial, and where the thesis/report should stay scoped.

It is intentionally conservative. It is a negative-evidence companion to the positive evidence pages, not a replacement for them.

## What This Document Covers

The focus is on:

- unsupported or partially supported claim areas
- gate shortfalls and unproven thresholds
- failure modes that remain explicit in the code and docs
- scope limits that must stay visible in publication-facing writing

## What It Does Not Cover

This page does not repeat the main system description, the full thesis pipeline, or the artifact catalog.

For those surfaces, see:

- [Thesis-Grade Codebase Report](thesis-codebase-report.md)
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

The thesis report already says this in different places. This page keeps the limitation visible in one place.

## Current Gate Shortfalls

The quality-gates page records the target gates and the latest evidence bundle, but not every gate is at a publishable threshold.

For this doc set, the safe default is: every `G11.*` and `P14.*` item is open unless another page cites a current artifact path, the measured value, the required threshold, and the required sample size. The reviewed pages in this slice do not yet do that for the full redesign gate set.

The current evidence set still leaves open:

- no reviewed page in this slice establishes green status for the DCCS, preference, multi-fidelity, REFC/selective-certification, VOI, global route-quality, runtime/reuse, evaluation-size, or test-floor `G11.*` families
- no reviewed page in this slice establishes `P14.*` statistical-discipline, seed-robustness, calibration-publishability, preference-burden, witness-sparsity, failure-atlas, baseline-fairness, artifact-packaging, sensitivity, or runtime-observability closure
- explicit suite-role registration now covers `broad cold proof`, `focused REFC proof`, `focused VOI proof`, `DCCS diagnostic probe`, `hot-rerun cold-source`, `hot-rerun`, `preference proof`, `optional-stopping coverage`, `proxy-audit calibration`, `perturbation / flip-radius`, `public transfer`, and `synthetic ground-truth`
- those newer suite roles still do not have checked artifact bundles cited on the reviewed pages in this slice, so role registration should not be confused with gate closure
- the current focused `C` evidence is still single-seed only (`seed_repeat_plan.headline_seed_repeat_required = true`, `configured_seed_count = 1`, `status = single_seed_only`), so `P14.8`, `P14.10`, and `P14.43` remain open even though the local checked bundle is useful as a single-run example
- the current checked `proxy_audit_calibration` bundle is proxy-only for `C` (`mean_audit_world_count = 0.0`, `mean_audited_route_pair_count = 0.0`, `proxy_only_fraction = 1.0`), so `P14.13` and `P14.14` remain open on current evidence and the audited-overlap `P5.2` / `P5.4` rows cannot be closed from this bundle alone

The safe reading is that the repo can demonstrate credible local evidence, but not blanket success across every gate or cohort.

## Evidence-Bounded Limitations

The thesis report explicitly frames several constraints as limitations rather than solved problems:

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

## What To Say In Publication Text

Use scoped language such as:

- the repository provides local evidence for the evaluated bundles
- the system is designed to fail closed under missing support
- the current thesis claims are bounded by the checked artifacts and gates
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
- `docs/thesis-codebase-report.md`
  - `### What this report will not overclaim`
  - `### Known limitations`
  - `## Appendix AE: Limitations, Risks, And Future-Work Directions`
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
- [Thesis-Grade Codebase Report](thesis-codebase-report.md)
