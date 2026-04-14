# Tutorial Mode and Reporting

Last Updated: 2026-04-12
Applies To: frontend tutorial flows, reporting surfaces, and backend run artifact APIs

Current truth anchors for the live runtime and UI:

- [Backend README](../backend/README.md)
- [Frontend README](../frontend/README.md)
- [API Cookbook](api-cookbook.md)
- [Run and Operations Guide](run-and-operations.md)
- [Reviewer Quickstart](reviewer_quickstart.md)

## Tutorial Mode Scope

Tutorial flow guides users through:

- route and Pareto generation
- scenario compare
- departure optimization
- duty chain and experiment workflows
- artifact inspection
- baseline comparison and live diagnostics
- tutorial-aware map interactions

The current tutorial shell has four operational states: `blocked`, `chooser`, `running`, and `completed`. It is desktop-gated, persists progress locally, and can resume or restart from saved state when an unfinished walkthrough exists.

## How Tutorial Mode Works

- The guided overlay is implemented in `frontend/app/components/TutorialOverlay.tsx` and driven by the step catalog under `frontend/app/lib/tutorial/*`.
- The overlay can lock the map, lock only a sidebar section, or stay free depending on the active step's lock scope.
- When a step needs a precise target and the target is not yet available, the overlay shows a positioning/loading state before it promotes the card into the proper location.
- Manual confirmation is still supported for steps that need explicit acknowledgement, such as map pin confirmation and other marked actions.
- The setup area can prefill canonical tutorial inputs, including the Newcastle-to-London example, duty-chain stops, and the tutorial experiment bundle.
- Locale changes are available during the tutorial, but they only affect labels and formatting. They do not change route math.

## Backend Reporting Endpoints Used By The UI

The run inspector proxies a strict allowlist of report and artifact paths. Current core report endpoints are:

- `GET /runs/{run_id}/manifest`
- `GET /runs/{run_id}/scenario-manifest`
- `GET /runs/{run_id}/provenance`
- `GET /runs/{run_id}/signature`
- `GET /runs/{run_id}/scenario-signature`
- `GET /runs/{run_id}/artifacts`

Current artifact endpoints exposed through the frontend proxy are:

- `GET /runs/{run_id}/artifacts/results.json`
- `GET /runs/{run_id}/artifacts/results.csv`
- `GET /runs/{run_id}/artifacts/metadata.json`
- `GET /runs/{run_id}/artifacts/routes.geojson`
- `GET /runs/{run_id}/artifacts/results_summary.csv`
- `GET /runs/{run_id}/artifacts/report.pdf`
- `GET /runs/{run_id}/artifacts/dccs_candidates.jsonl`
- `GET /runs/{run_id}/artifacts/dccs_summary.json`
- `GET /runs/{run_id}/artifacts/refined_routes.jsonl`
- `GET /runs/{run_id}/artifacts/strict_frontier.jsonl`
- `GET /runs/{run_id}/artifacts/winner_summary.json`
- `GET /runs/{run_id}/artifacts/certificate_summary.json`
- `GET /runs/{run_id}/artifacts/route_fragility_map.json`
- `GET /runs/{run_id}/artifacts/competitor_fragility_breakdown.json`
- `GET /runs/{run_id}/artifacts/value_of_refresh.json`
- `GET /runs/{run_id}/artifacts/sampled_world_manifest.json`
- `GET /runs/{run_id}/artifacts/voi_action_trace.json`
- `GET /runs/{run_id}/artifacts/voi_action_scores.csv`
- `GET /runs/{run_id}/artifacts/voi_stop_certificate.json`
- `GET /runs/{run_id}/artifacts/final_route_trace.json`
- `GET /runs/{run_id}/artifacts/od_corpus.csv`
- `GET /runs/{run_id}/artifacts/od_corpus_summary.json`
- `GET /runs/{run_id}/artifacts/ors_snapshot.json`
- `GET /runs/{run_id}/artifacts/thesis_results.csv`
- `GET /runs/{run_id}/artifacts/thesis_summary.csv`
- `GET /runs/{run_id}/artifacts/methods_appendix.md`
- `GET /runs/{run_id}/artifacts/thesis_report.md`

The `frontend/app/api/runs/[runId]/[...subpath]/route.ts` proxy is intentionally allowlisted, so reporting links stay predictable instead of exposing arbitrary backend paths. In the current allowlist, `index.json` and `index.md` are bundle files you may inspect through the backend artifact route directly, but they are not exposed through this frontend proxy today.

For route-compute bundles, `index.json` is the machine-readable bundle manifest and `index.md` is the reviewer-readable companion summary. Thesis-like bundles may also carry the same pair when they were emitted or additively refreshed through the run-store path. Treat those files as stable artifact-list and artifact-presence entrypoints only; they do not imply committed PDF or SVG renders.

## Labeling Discipline For Reporting Surfaces

- Certification-facing summaries in the UI are conditional report surfaces: they reflect the support, bounded-world, and model-validity assumptions carried by the current response or artifact bundle, not assumption-free guarantees about reality.
- Baseline deltas, run summaries, and exported CSV rows are empirical summaries of the visible run or checked bundle unless a page explicitly cites a stronger artifact-backed invariant.
- `epic score`, `epic tier`, witness explanations, demo presets, and controller-score summaries are heuristic or organizational surfaces unless the underlying artifact explicitly marks a theorem-backed invariant.

## Current Reporting Panels

### Decision State / Preference Visibility

`frontend/app/components/DecisionStateSummary.tsx` remains the read-only preference-inspection surface inside the main route result view. It consumes the live `DecisionPackage` payload and stays visible for singleton, certified-set, and typed-abstention outcomes.

`frontend/app/components/PreferenceElicitationPanel.tsx` is the adjacent live elicitation surface in that same route view. It consumes the current route/runtime payload, lets the reviewer issue pairwise, tradeoff, veto, and time-preserving guard answers against the visible routes, and updates the in-memory compatible-set summary and shrinkage trace without switching to mock data.

The component currently renders:

- terminal type, certificate basis, certified-set membership/exclusion notes, and abstention class
- support summary and world-count/reuse visibility carried by the response payload
- typed preference payload summaries from:
  - `preference_state`
  - `preference_query_trace`
  - `preference_summary`
- compatible-set size and volume proxy
- necessary-best vs possible-best probability summaries when the payload provides them
- pairwise, tradeoff, veto, and time-guard query evidence when present
- shrinkage-over-time from the recorded preference trace
- explicit `why this query` and `why no preference query was asked` messaging when the backend provides those reasons

The live panel currently adds:

- pairwise choice between the selected route and a runtime challenger
- threshold and ratio tradeoff questions over the visible route metrics
- route-level veto submission
- time-preserving guard questions derived from the selected route duration
- a compatible-set region summary and shrinkage timeline that update immediately on the current runtime payload

### Route Certification / Decision Proof

`frontend/app/components/RouteCertificationPanel.tsx` is the current in-app certificate-inspection surface for singleton, certified-set, and typed-abstention outcomes. It keeps decision/support/controller context visible even when no singleton route geometry is available. The panel reports conditional certification artifacts from the active run; it does not upgrade those artifacts into assumption-free guarantees.

The panel currently renders:

- controller context from the live response payload (`terminal_type`, stop reason, search completeness, search gap, iteration and budget summaries)
- an artifact-backed controller trace with chosen actions, next-best unused action summaries, predicted vs realized certificate movement, action costs, and stop/abstain reasoning from:
  - `voi_action_trace.json`
  - `voi_action_scores.csv`
  - `voi_stop_certificate.json`
- support and governance context from the inline `world_support_summary` / `support_summary` payloads
- scenario/profile provenance, mode observation source, and mode projection ratio
- evidence audit metrics from proxied route artifacts:
  - `sampled_world_manifest.json`
  - `route_fragility_map.json`
  - `value_of_refresh.json`
- direct artifact links for the visible evidence-audit and controller metrics
- frontend-generated CSV and SVG exports for the visible decision card, controller trace, and evidence summary; the SVG exports are vector figure surfaces intended for PDF-ready placement rather than direct PDF rendering

The standalone `world_support_summary.json` artifact is still emitted by the backend run bundle, but the current frontend route-artifact proxy does not expose that file directly, so the panel uses the inline response payload for support/governance details today.

### Proof Dashboard / Demo Presets

`frontend/app/components/ProofDashboardPanel.tsx` is the separate proof-dashboard surface on the main page. It does not replace the route-level proof cards; it reorganizes the same run into reviewer-facing proof slices and bundle-entry links. In this page, "proof" means artifact-backed reviewer navigation, not automatic closure of theorem, publication, or adoption claims.

The dashboard currently renders:

- `V0 / A / B / C` proof slices for comparator, search/DCCS, certification/support, and VOI/controller families
- proof lenses for:
  - broad vs focused
  - cold vs hot
  - OSRM vs ORS
  - theorem-to-artifact navigation
- direct deep links to bundle and artifact files such as:
  - `index.json`
  - `index.md`
  - `dccs_summary.json`
  - `strict_frontier.jsonl`
  - `certificate_summary.json`
  - `route_fragility_map.json`
  - `sampled_world_manifest.json`
  - `voi_action_trace.json`
  - `voi_action_scores.csv`
  - `voi_stop_certificate.json`
- `RunInspector` buttons on the visible proof tiles
- `Copy Proof Summary` and `Export Dashboard CSV` actions grounded in the visible dashboard state
- a deterministic witness-driven explanation synthesized from the current terminal/support/controller/witness payload fields

The setup card also exposes canned proof-demo presets for:

- safe singleton
- certified set
- support abstention
- preference-sensitive inspection
- collapse-prone search inspection
- hot-rerun inspection

These presets only prefill existing request knobs. They are not synthetic result bundles, so the backend still determines the actual terminal outcome.

### Route Comparison

`frontend/app/components/RouteBaselineComparison.tsx` now reports a richer comparison than simple ETA alone. It shows:

- ETA improvement
- cost improvement
- CO2 improvement
- distance improvement
- `epic score`
- `epic tier`
- smart compute elapsed
- baseline fetch elapsed
- smart candidate count
- live-source coverage
- live calls observed

The comparison math is built from current route metrics and uses the existing baseline route as the reference. Positive percentages mean better when the metric is lower-is-better, which is why the panel labels the sign convention explicitly. Those deltas are empirical for the visible request or checked bundle, while `epic score` and `epic tier` remain heuristic summary labels for that comparison frame.

The map and summary panels currently support three baseline styles:

- OSRM baseline
- OpenRouteService baseline, including the proxy-backed variant
- academic reference selection

### Scenario Compare

`frontend/app/components/ScenarioComparison.tsx` compares no-sharing, partial-sharing, and full-sharing outcomes side by side. It also surfaces delta reason codes when a metric is missing, which makes it easier to explain when a comparison row is not fully populated.

### Experiment Reporting

`frontend/app/components/ExperimentManager.tsx` is the current save/replay surface for bundles. It supports catalog filtering by name, vehicle, scenario mode, and sort order, then lets the user load, open, edit metadata, replay, or delete a saved bundle.

### Oracle Quality Reporting

`frontend/app/components/OracleQualityDashboard.tsx` records source checks and reports:

- total checks
- source count
- stale threshold
- per-source pass rate
- schema failures
- signature failures
- stale count
- average latency
- last observed timestamp

The dashboard also exposes a CSV export at `GET /api/oracle/quality/dashboard.csv`.

The proof dashboard's CSV export is frontend-generated from the visible proof tiles. Full SVG/PDF-ready figure export for the decision-proof surfaces is still a later packet.

### Departure And Duty Reporting

`frontend/app/components/DepartureOptimizerChart.tsx` reports each tested departure time with the chosen route, score, ETA, cost, and CO2, then lets the user apply one result back into the active request.

`frontend/app/components/DutyChainPlanner.tsx` reports per-leg outcomes plus total metrics across the full chain, including optional total energy when the backend returns it.

### Segment Reporting

`frontend/app/components/SegmentBreakdown.tsx` shows a per-segment table for the selected route, defaults to a 40-row preview, and can expand to the full set. It can also copy the visible rows as CSV.

## Compute And Debug Reporting

The compute trace overlay now captures the operational story of a run:

- request ID
- stage timing
- retry and fallback behavior
- live-call trace summary
- graph diagnostics
- scenario coverage gate information
- live refresh gate information
- slowest calls
- optional AI diagnostic bundle

The UI also has direct access to:

- `POST /api/route/baseline`
- `POST /api/route/baseline/ors`
- `GET /api/debug/live-calls/{requestId}`
- `GET /api/health/ready`
- `GET /api/metrics`

## Practical Operator Flow

1. trigger compute action in UI
2. capture returned `run_id`
3. inspect manifest, provenance, and signatures
4. inspect route compare panels, scenario deltas, and live-call traces
5. download artifact files needed for reporting or debugging

## Local Commands

Frontend:

```powershell
pnpm -C frontend dev
```

Backend headless scenario smoke:

```powershell
uv run --project backend python backend/scripts/run_headless_scenario.py --input-json docs/examples/sample_batch_request.json
```

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Frontend Dev Tools Coverage](frontend-dev-tools.md)
- [Sample Manifest and Outputs](sample-manifest.md)
- [Backend APIs and Tooling](backend-api-tools.md)

