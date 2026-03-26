# Methods Appendix

- Generated rows: 8
- Corpus hash: `44490ad7ee3b9b41e7d9060bb69d71cef8221b63b25c8054ed994de9b7134bbc`
- Variants: `V0=legacy, A=dccs, B=dccs_refc, C=voi`
- Matched search budget: `4`
- Evidence budget: `2`
- Certificate world count: `64`
- Certificate threshold: `0.8`
- Stop threshold: `0.02`
- Baseline refinement policy for V0: `corridor_uniform`
- Secondary baseline policy: `local_service`
- Secondary baseline snapshot mode: `off`
- Backend readiness timeout seconds: `1800.0`
- Backend readiness poll seconds: `5.0`
- Max alternatives: `8`
- Tolls enabled: `True`
- In-process backend: `True`
- Strict evidence policy: `no_synthetic_no_proxy_no_fallback`
- Backend URL: `http://localhost:8000`
- OSRM base URL: `http://localhost:5000`
- Local ORS base URL: `http://localhost:8082/ors`

Comparator honesty:
- `V0` is not a full-budget legacy solve in this thesis lane; it is a matched-budget legacy comparator whose expensive refinement stage is capped by the same `search_budget` used by the new pipeline variants.
- The thesis runner passes the same upstream ambiguity/support priors to every variant, including `V0`, so `V0` here should be read as an evaluator-informed legacy comparator rather than an uninformed public-API call with no corpus context.
- The secondary baseline is the self-hosted local openrouteservice engine when `ors_baseline_policy=local_service`; `repo_local` is retained only as an explicit fallback/debug comparator.
- Corpus ambiguity is treated as an upstream route-graph prior when available. Missing corpus priors are not backfilled unless `--auto-enrich-corpus-ambiguity` is explicitly enabled, so thesis runs do not hide extra route-graph probe cost inside evaluation runtime.
- Ambiguity-adaptive budgeting is deterministic and corpus-prior driven: low-ambiguity rows use smaller REFC/VOI budgets, while high-ambiguity rows keep larger search and certification budgets so the controller is still meaningfully stressed.

Metric definitions:
- Strict dominance compares duration, money, and CO2 with Pareto minimisation.
- Weighted win uses the same fixed selector weights as the route request.
- Balanced win averages clipped relative improvements across the three objectives.
- Frontier metrics include hypervolume, singleton baseline coverage, epsilon indicator, and diversity summaries.
- Ambiguity metrics include corpus-side OD ambiguity, engine-disagreement, and hard-case priors plus an observed ambiguity index derived from realized frontier multiplicity, near-tie mass, winner-margin compression, certificate shortfall, selector disagreement, diversity collapse, and controller intervention.
- Certification metrics include threshold margin, runner-up certificate gap, selector-vs-certificate disagreement, fragility entropy, competitor turnover, ambiguity-prior gap, unique-world reuse, and bounded hard-row stress-pack counts.
- DCCS metrics include score-label correlation, score-ranked frontier recall at used budget, corridor-family recall, realized diversity-collapse rate, supplemental challenger activation, and budget utilization.
- Controller metrics include action counts by type, controller engagement on stressed rows, initial-certificate stop rate, unnecessary VOI refine rate, shortcut rate, time-to-certification, and realized certificate lift.
- Cohort metrics split representative, ambiguity, a broader hard-case ambiguity-pressure cohort, and a stricter controller-stress cohort, then report ambiguity-vs-representative gaps for certificate, runtime, and resource utilization.
- Runtime metrics split algorithm route solve time from baseline acquisition time, report runtime ratios versus OSRM and local ORS, include readiness wait, graph warmup elapsed, warmup-overhead share, preflight-plus-warmup time, process RSS/VMS snapshots, route-cache hit rate, option-build stage cost and reuse, runtime-per-refined-candidate, runtime-per-frontier-member, memory-per-refined-candidate, and p50/p90/p95 distribution summaries plus controller action density.
- Direct incremental-value metrics compare A/B/C against the matched-budget V0 baseline on weighted utility margin, balanced gain, frontier hypervolume, certificate lift, and hard-case certificate lift.