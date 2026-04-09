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

The thesis report already says this in different places. This page keeps the limitation visible in one place.

## Current Gate Shortfalls

The quality-gates page records the target gates and the latest evidence bundle, but not every gate is at a publishable threshold.

The current evidence set still leaves open:

- subsystem-dependent benchmark gaps
- support-sensitive behavior that can still fail closed
- fallback and readiness failures when live sources or assets are stale, missing, or unsupported
- thesis-lane claims that are scoped to specific bundles rather than universal across the whole problem space

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
- the REFC and DCCS surfaces are explicit implementation contracts, not formal proof packages in this slice

Avoid language such as:

- universal
- always
- provably optimal
- complete
- all conditions
- unconditional

## Evidence Index

Use the following surfaces when writing or reviewing this document:

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
  - current slice rows marked `scaffold-only`
  - current slice rows marked `empirical`
- `docs/theorem_map.md`
  - theorem slots that remain open
  - explicit structural surfaces that are not yet theorem-backed

## Minimal Review Checklist

Before treating a claim as publication-ready, check that:

- the claim appears in the positive evidence pages first
- a limitation or gate note exists if the claim is scoped
- any REFC or DCCS language is tied to actual checked artifacts or tests
- no universal wording slips in where the evidence is only local

## Bottom Line

The current repository supports a careful, evidence-backed story about a scoped UK freight-routing system with explicit failure modes and documented gates.

It does **not** yet support a blanket story of universal dominance, unconditional validity, or theorem-backed certainty for the current redesign surfaces.
