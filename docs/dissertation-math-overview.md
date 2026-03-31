# Dissertation Math Overview

Last Updated: 2026-03-31  
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

In the current response contract, the selected route and its frontier context are surfaced through `RouteResponse.selected`, `RouteResponse.candidates`, `RouteResponse.selected_certificate`, and `RouteResponse.voi_stop_summary`.

## DCCS

DCCS is a budgeted triage stage that maps from `K_raw` to a selected refinement set inside `R`.

The thesis-facing quantities here are not only winner preservation, but also:

- frontier recall at budget
- decision-critical yield
- richness of nontrivial frontiers
- challenger survival into the refined strict frontier

## REFC

REFC defines certificate-style winner evidence over deterministic sampled worlds.

Key quantities:

- `C(r)`: fraction of sampled worlds in which route `r` wins under the fixed selector
- certification threshold
- winner fragility
- competitor-side fragility
- value of refresh by evidence family

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

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Expanded Math Appendix](math-appendix.md)
- [Graph Theory Notes](appendix-graph-theory-notes.md)
- [VOI Thesis Pipeline Spec](voi-pipeline-spec.md)
