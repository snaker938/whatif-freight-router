# Expanded Math Appendix

This appendix expands the definitions and derivations used by the current v0
implementation.

## Notation table

| Symbol | Meaning | Units |
|---|---|---|
| `r` | a candidate route | - |
| `i` | segment index within a route | - |
| `d_i` | segment distance | meters |
| `t_i` | segment duration | seconds |
| `d_i_km` | segment distance `d_i / 1000` | km |
| `v_i` | segment speed `(d_i / t_i) * 3.6` | km/h |
| `sf(v_i)` | speed factor function | scalar |
| `m` | vehicle mass | tonnes |
| `ef` | emission factor | kg CO2e / tonne-km |
| `idle_h` | idle emissions factor | kg CO2e / hour |
| `cpkm` | cost per km | currency / km |
| `cph` | cost per hour | currency / hour |
| `alpha` | driver time cost weight | scalar (0.35) |
| `T_r` | total route duration after scenario | seconds |
| `T_r_base` | base route duration before scenario | seconds |
| `M_r` | route monetary proxy | currency |
| `E_r` | route emissions | kg CO2e |

## Objective formulas

Segment emissions:

- `E_i_move = m * d_i_km * ef * sf(v_i)`

Low-speed idling inside segment function:

- if `v_i < 5`, `E_i_idle = (t_i / 3600) * idle_h`, else `0`

Base route emissions:

- `E_r_base = sum_i (E_i_move + E_i_idle)`

Scenario-adjusted duration:

- `T_r = T_r_base * scenario_multiplier`
- `extra_time_h = max(T_r - T_r_base, 0) / 3600`

Final route emissions:

- `E_r = E_r_base + extra_time_h * idle_h`

Segment monetary proxy:

- `M_i = d_i_km * cpkm * sf(v_i)`

Base route money:

- `M_r_base = sum_i M_i + (T_r_base / 3600) * cph * alpha`

Final route money:

- `M_r = M_r_base + extra_time_h * cph * alpha`

## Normalization and weighted selection derivation

Given candidate set `R` with objectives `T_r`, `M_r`, `E_r`:

1. Compute min/max for each objective across `R`.
2. Normalize each objective:
   - `norm(x_r) = 0` if `x_max <= x_min`
   - `norm(x_r) = (x_r - x_min) / (x_max - x_min)` otherwise
3. Normalize user weights:
   - `wt, wm, we = w_time, w_money, w_co2` divided by their sum
   - fallback to `1/3` each if sum is zero
4. Compute scalar score:
   - `S_r = wt * norm(T_r) + wm * norm(M_r) + we * norm(E_r)`
5. Choose route with minimum `S_r`.

## Complexity breakdown

### Candidate generation and fetch orchestration

- Spec count is constant-sized in current code:
  - 1 alternatives request
  - 4 exclude requests (`motorway`, `trunk`, `toll`, `ferry`)
  - 4 via-point requests
- Total fetch specs: 9
- Async fetch fan-out is bounded by semaphore (`batch_concurrency`, capped)

If each request returns up to `k` routes, route processing cost is about:

- `O(9 * k)` for dedupe/ranking with small constants

### Pareto filtering

For `n` built candidate options:

- dominance checks worst-case `O(n^2)`
- output sorting `O(n log n)`

With small `n` in this architecture, pairwise dominance remains practical.

## Assumptions and limitations

- Cost and emissions factors are heuristic placeholders.
- `speed_factor` is designed for visible trade-offs, not calibrated fuel physics.
- Current model omits toll tables, elevation, road grade, and stochastic travel
  time.
- Pareto operates on route candidates from OSRM, not full graph-state labels.
