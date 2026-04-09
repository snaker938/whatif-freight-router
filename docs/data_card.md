# Data Card

This page is the corpus- and support-facing reference for the current thesis-style evaluation data.

It is intentionally conservative. It describes the OD corpus, support bins, and evidence artifacts the repository already documents today. It does **not** claim the corpus is complete, universal, or fully closed for all future evaluator use.

## Purpose

The repository uses a UK-oriented OD corpus together with support-aware slices to evaluate the routing pipeline. The data card keeps that corpus story explicit and bounded.

Use this page when you want to know:

- what the OD corpus represents
- how support or ambiguity slices are described today
- which checked artifacts hold the current corpus evidence
- where to look for raw input sources and derived bundle outputs

## Corpus Overview

The docs currently describe the OD corpus as a sampled or assembled set of UK origin-destination cases used for thesis evaluation and ambiguity-aware analysis.

The corpus is not presented here as a universal transport dataset. In the current repository, it is a bounded evidence set for the checked UK-focused workflows.

Current repository docs indicate that the corpus and related support labels are used for:

- distance and corridor binning
- ambiguity-aware evaluation slices
- scenario comparison and thesis-lane reporting
- cohort composition summaries in thesis bundles

## Support And Ambiguity Slices

The repo currently documents a support-aware and ambiguity-aware corpus story, but it stays descriptive rather than exhaustive.

Current surfaces referenced in the docs include:

- `od_ambiguity_support_ratio`
- `od_ambiguity_prior_strength`
- `od_ambiguity_source_count`
- `od_ambiguity_source_mix`
- `od_ambiguity_source_entropy`
- `od_ambiguity_confidence`
- `ambiguity_budget_prior`
- `ambiguity_budget_band`

These fields are useful because they explain why some rows are grouped as representative, ambiguity-heavy, or otherwise support-sensitive.

What is evidenced now:

- the corpus is stratified enough to support cohort composition and ambiguity summaries
- support-aware slices are visible in the thesis-lane outputs
- current docs connect these slices to evaluated bundles rather than treating them as abstract labels

What is not yet evidenced as a blanket claim:

- complete coverage of every possible OD pair
- universal stability of the support labels across all future corpus builds
- theorem-backed guarantees about corpus optimality or sufficiency

## Current Corpus Artifacts

The repository already documents the following corpus-related outputs:

- `backend/out/thesis_campaigns/*/od_corpus.csv`
- `backend/out/thesis_campaigns/*/od_corpus.json`
- `backend/out/thesis_campaigns/*/od_corpus_summary.json`
- `backend/out/thesis_campaigns/*/od_corpus_rejected.json`
- `backend/out/thesis_campaigns/*/cohort_composition.json`
- `backend/out/thesis_campaigns/*/thesis_summary_by_cohort.json`
- `backend/out/thesis_campaigns/*/thesis_metrics.json`

The docs also reference a corpus-ambiguity summary artifact:

- `backend/out/corpus_ambiguity_refresh_summary.json`

These files are the best source for current OD corpus composition and support-bin evidence.

## Current Evidence Shapes

The current docs indicate that the thesis bundles can report:

- row counts for the OD corpus
- cohort counts and composition
- ambiguity-index summaries
- support ratios and support-strength summaries
- per-cohort evaluation summaries

The docs also note that at least one current bundle includes:

- a `representative` cohort
- an `ambiguity` cohort

This means the evaluator is already using cohort semantics to explain the corpus, but the page should still treat those as checked-bundle descriptors rather than as permanent global classes.

## Raw Source Families

The corpus is built from or supported by the current UK raw evidence families documented elsewhere in the repo, including:

- `backend/data/raw/uk/scenario_live_observed.jsonl`
- `backend/data/raw/uk/scenario_mode_outcomes_observed.jsonl`
- `backend/data/raw/uk/stochastic_residuals_raw.csv`
- `backend/data/raw/uk/dft_counts_raw.csv`
- `backend/data/raw/uk/fuel_prices_raw.json`
- `backend/data/raw/uk/carbon_intensity_hourly_raw.json`
- `backend/data/raw/uk/toll_tariffs_operator_truth.json`
- `backend/data/raw/uk/toll_classification/*`
- `backend/data/raw/uk/toll_pricing/*`

The data card does not restate those source families in full. It only points to them as the upstream evidence base that the corpus and support bins depend on.

## What Is Evidenced Now

The current repo does support the following data-facing facts:

- the OD corpus is a real checked evaluation input, not a hypothetical placeholder
- cohort and support summaries are present in thesis bundles
- ambiguity/support fields are explicit in the docs and runtime outputs
- the corpus is scoped to UK-focused thesis evaluation rather than universal transport data

## What Is Not Yet Green

Do not treat this page as proof that:

- the corpus is complete
- the support bins are final for all future work
- every cohort has passed every possible threshold
- the evaluation data generalizes beyond the checked UK setting

Those claims would require stronger evidence than the current docs provide.

## Where Reviewers Should Look

For current evidence, reviewers should start with:

1. `docs/thesis-codebase-report.md`
   - OD corpus sections
   - ambiguity/corpus evidence sections
   - limitations and scope notes
2. `docs/sample-manifest.md`
   - artifact bundle names
   - `od_corpus.*` and `cohort_composition.json`
3. `docs/quality-gates-and-benchmarks.md`
   - current thesis-lane validation snapshot
   - strict preflight and benchmark evidence
4. `docs/model-assets-and-data-sources.md`
   - raw source families
   - runtime asset provenance
5. `docs/claim_matrix.md`
   - current slice status for corpus and support surfaces

## Suggested Review Framing

When reviewing corpus or support claims, use these questions:

- Is the claim about the corpus itself, or about a particular checked bundle?
- Does the doc point to a concrete artifact file, or only to a general concept?
- Is the support bin described as a current slice label or a permanent taxonomy?
- Are the notes scoped to UK-focused thesis evaluation, or drifting into universality?

## Bottom Line

The current repository supports a bounded UK OD corpus with explicit support and ambiguity slices.

It does **not** yet support a blanket claim of completeness, universality, or theorem-backed corpus sufficiency.
