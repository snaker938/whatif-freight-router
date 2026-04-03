# Dissertation Math Overview

Last Updated: 2026-04-03  
Applies To: objective scoring, strict-frontier selection, DCCS triage, REFC certification, and VOI stopping

This page gives the high-level mathematical framing for the current thesis pipeline.

## Core Objective Family

Route candidates are evaluated across at least:

- duration
- monetary cost
- emissions

Risk-aware summaries may also expose uncertainty-derived quantities such as robust scores and quantile-based utilities.

The current API surface carries these values through `RouteOption.metrics`, `RouteOption.uncertainty`, and the route-selection response wrappers rather than flattening them into a single scalar.

## Strict Frontier

The strict frontier `F` is derived from the refined candidate set `R` and excludes dominated routes under the configured objective view.

This frontier is the object consumed by REFC and VOI. Presentation-layer backfill should not be treated as part of the certified frontier.

This page describes the maintained mathematical and state vocabulary for certification. It does not assume that every support or witness object is already wired into the public route-response surface.

## DCCS

DCCS is a budgeted triage stage that maps from `K_raw` to a selected refinement set inside `R`.

The thesis-facing quantities here are not only winner preservation, but also:

- frontier recall at budget
- decision-critical yield
- richness of nontrivial frontiers
- challenger survival into the refined strict frontier

## REFC

REFC defines certificate-style winner evidence over bounded world bundles plus explicit support and challenger-state summaries.

Key quantities:

- empirical certificate `C(r)`: conservative winner frequency over the bounded world bundle
- world-bundle fidelity, including probabilistic bundle state and audit correction state
- support strength derived from ambiguity support, provenance coverage, and proxy penalties
- winner confidence bounds and pairwise challenger-gap summaries
- flip radius and decision-region state for winner-vs-challenger separation
- certified-set state for routes whose lower bounds clear the threshold
- certification threshold
- winner fragility
- competitor-side fragility
- value of refresh by evidence family

The maintained certification layer now names these concepts explicitly through objects such as `ProbabilisticWorldBundle`, `AuditWorldBundle`, `WorldSupportState`, `WinnerConfidenceState`, `PairwiseGapState`, `FlipRadiusState`, `DecisionRegionState`, `CertifiedSetState`, `CertificationState`, `CertificateWitness`, and `AbstentionRecord`.

## VOI-AD2R

VOI-AD2R is a budgeted controller over admissible actions.

Key quantities:

- expected certificate lift or other admissible controller value
- evidence budget used
- search budget used
- stop threshold `tau_stop`
- explicit stop reason

The controller should stop honestly when no admissible action clears the threshold.

Current stop summaries also expose `search_completeness_score`, `search_completeness_gap`, and `credible_search_uncertainty` so the thesis can distinguish a deliberate stop from a premature one.

Those controller summaries should be read as public stop records, not as a replacement for the larger maintained support/certification state layer described above.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Expanded Math Appendix](math-appendix.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [VOI Thesis Pipeline Spec](voi-pipeline-spec.md)
