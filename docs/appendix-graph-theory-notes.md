# Graph Theory Notes (Current and Next Steps)

This note explains how the current implementation relates to graph-theory
concepts and what is required for future upgrades.

## 1) How OSRM output maps to path-level optimization

The backend currently asks OSRM for complete route alternatives between OD
pairs. Each route is already a path through OSRM's road graph.

The service then:

- extracts path-level metrics (duration, monetary proxy, emissions)
- performs dominance filtering on those path candidates
- applies weighted selection on the filtered set

So optimization is currently "path set post-processing", not in-graph dynamic
programming over labels.

## 2) Why this is candidate-based Pareto, not full MOSP

A full multi-objective shortest path (MOSP) algorithm explores graph states and
maintains non-dominated labels per node (or node-time state).

Current flow differs:

- route alternatives are generated first by OSRM heuristics
- Pareto is computed only over returned candidates
- no per-node label frontier is stored
- no exhaustive Pareto frontier guarantee over the full graph

Therefore, current Pareto quality depends on candidate diversity, not complete
graph search guarantees.

## 3) Concepts for future upgrade

### Label-setting and label-correcting

- Label-setting: permanently finalizes labels in an ordered progression under
  conditions that make this valid.
- Label-correcting: allows revisiting/improving labels and usually handles
  broader conditions but with higher runtime variance.

### Dominance pruning intuition

If label `A` has objective values no worse than label `B` in all dimensions and
strictly better in one, `B` can never lead to a Pareto-improving solution and
can be pruned.

### Practical frontier controls

- epsilon-dominance approximations
- bound-based pruning
- objective-specific lower bounds

These controls manage label explosion in dense networks.

## 4) Current limitations and transition path

Current limitations:

- no explicit graph builder with per-edge objective attributes stored in app
  data structures
- no exact multi-objective shortest-path guarantee
- scenario effects are applied as route-level multipliers, not edge-time events

Transition path:

1. Build/import internal graph with edge-level time/distance/cost/CO2 baseline.
2. Implement MOSP label engine with dominance pruning.
3. Add incident/weather time-dependent edge updates.
4. Keep current candidate mode as fallback and benchmark baseline.

## Assumptions and limitations

- This note is conceptual and aligns to current repository behavior, not an
  implemented MOSP engine.
- Complexity and guarantees for future algorithms depend on final state-space
  design (static graph vs time-expanded graph).
