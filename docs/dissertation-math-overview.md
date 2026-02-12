# Dissertation Math Overview

This page summarizes the mathematics currently implemented in the v0 backend.

## 1) Pareto dominance (minimization)

For objective vectors `a` and `b` with dimensions time, money, and CO2:

- `a` dominates `b` iff `a_i <= b_i` for every objective `i`
- and `a_j < b_j` for at least one objective `j`

The non-dominated set (Pareto set) is:

- all vectors for which no other vector dominates them

## 2) Current Pareto filtering complexity

The current `pareto_filter` compares each candidate with current non-dominated
items, giving worst-case complexity:

- `O(n^2)` time
- `O(n)` storage for retained non-dominated candidates

This is appropriate for the current small OSRM candidate sets.

## 3) Weighted-sum route selection

After Pareto candidates are available, final route selection uses normalized
weights:

- user weights: `(w_time, w_money, w_co2)`
- normalized weights: `(wt, wm, we) = (w_time, w_money, w_co2) / (w_time + w_money + w_co2)`
- if sum is zero, backend defaults to `(1/3, 1/3, 1/3)`

Each objective is min-max normalized:

- `norm(x) = 0` if `x_max <= x_min`
- else `norm(x) = (x - x_min) / (x_max - x_min)`

Score for route `r`:

- `score(r) = wt * norm(time_r) + wm * norm(money_r) + we * norm(co2_r)`

The route with smallest `score(r)` is selected.

## 4) Monetary proxy formula in v0

Per segment:

- `distance_km = distance_m / 1000`
- `speed_kmh = (distance_m / duration_s) * 3.6` (if duration > 0)
- `segment_fuel_cost = distance_km * vehicle.cost_per_km * speed_factor(speed_kmh)`

Route-level monetary proxy:

- `base_monetary = sum(segment_fuel_cost) + base_duration_h * vehicle.cost_per_hour * 0.35`
- `duration_s = base_duration_s * scenario_multiplier`
- `extra_time_h = max(duration_s - base_duration_s, 0) / 3600`
- `monetary = base_monetary + extra_time_h * vehicle.cost_per_hour * 0.35`

## 5) Emissions formula in v0

Per segment moving emissions:

- `segment_emissions = mass_tonnes * distance_km * emission_factor_kg_per_tkm * speed_factor(speed_kmh)`

Low-speed idling add-on (inside segment model):

- if `speed_kmh < 5`, add `duration_h * idle_emissions_kg_per_hour`

Scenario-delay idling add-on (route-level):

- add `extra_time_h * idle_emissions_kg_per_hour`

Total route CO2:

- sum of segment emissions and idling terms

## Assumptions and limitations

- `speed_factor` and cost formulas are placeholders for multi-objective behavior,
  not calibrated policy-grade models.
- Scenario handling currently uses fixed multipliers, not explicit incident
  simulation.
- Pareto filtering runs over fetched route candidates, not over a full
  multi-objective shortest-path graph search.
