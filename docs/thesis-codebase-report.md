# Thesis-Grade Codebase Report: whatif-freight-router

Date: March 6, 2026

Repository basis: current local working tree at `c:\Users\jmend\Documents\GitHub\whatif-freight-router`

Evidence policy: repo-local evidence only.

## Abstract

This project is a UK-focused freight-routing decision system built as a hybrid of four things: a prepared OSRM road-routing engine, a Python/FastAPI modeling backend, a Next.js single-page frontend, and a calibrated data/model asset layer that injects time-of-day, terrain, toll, fuel, carbon, uncertainty, and live scenario pressure into route selection. In plain language, the project does not only ask "what is the shortest road path?" It asks "which of the feasible UK freight routes is the better operational decision once cost, delay risk, terrain, tolls, weather, live pressure, carbon, and user preferences are added?"

The main engineering character of the repository is strictness. The backend is intentionally fail-closed for many subsystems: graph readiness, scenario data, terrain coverage, toll topology/tariffs, departure profiles, stochastic regimes, fuel and carbon feeds, and several signature/provenance checks. This means the software would rather stop with an explicit reason code than silently invent a result from stale or unsupported data. That design choice is one of the strongest thesis themes in the repo.

The second defining trait is that the project is not a pure academic implementation. It uses academically recognizable methods such as weighted-sum selection, augmented Tchebycheff scalarisation, VIKOR compromise ranking, Pareto dominance, epsilon-constraint filtering, A* heuristics, Yen-style K-shortest paths, quantiles, and CVaR. But those methods are adapted into transparent engineering blends that better fit freight-routing product needs. The code itself explicitly says the modified profiles are not claimed as novel theory.

## How To Read This Report

This report is intentionally dual-layered.

1. Each major chapter starts with plain-English explanations of what the subsystem does for a user.
2. The same chapter then drills into the exact implementation, math, data source, calibration, failure behavior, or file-level evidence.

If the goal is dissertation writing, the most useful reading order is:

1. System overview
2. Frontend capabilities
3. Backend routing pipeline
4. Routing mechanics
5. Math and algorithms
6. Physics and cost model
7. Live/strict data
8. Data and calibration
9. Comparison against OSRM/ORS baselines
10. Quality, reproducibility, and file inventory appendices

## Scope And Evidence Hierarchy

### Scope

The repository currently contains:

- 962 tracked files.
- 844 tracked files under `backend`, mostly because raw toll evidence and toll fixtures are intentionally stored in-repo.
- 72 tracked files under `frontend`.
- 24 tracked files under `docs`.
- 33 tracked backend data/build/benchmark scripts under `backend/scripts`.
- 39 backend runtime modules under `backend/app`.
- 71 tracked backend test files.
- 25 current frontend API proxy route files in the working tree.
- 22 current frontend component files in the working tree.

The current workspace also contains important locally present, currently untracked additions that affect the active codebase view in the IDE. The most important are:

- `frontend/app/api/route/baseline/route.ts`
- `frontend/app/api/route/baseline/ors/route.ts`
- `frontend/app/components/RouteBaselineComparison.tsx`
- `frontend/app/lib/baselineComparison.ts`
- `backend/tests/test_route_baseline_api.py`
- `backend/tests/test_pareto_backfill.py`

Because the user asked for the entire codebase, this report describes both the tracked repository and those active local additions when they materially affect behavior.

### Evidence hierarchy used in this report

Primary truth:

- backend runtime code in `backend/app`
- frontend runtime code in `frontend/app`

Secondary truth:

- backend tests in `backend/tests`
- repository docs in `docs`

Tertiary truth:

- generated outputs already present locally under `backend/out`
- CI workflow under `.github/workflows`
- supporting scripts under `backend/scripts` and `scripts`

When docs and code differ, code is treated as authoritative.

### What this report will not overclaim

The repository contains machinery to compare smart routes against:

- a plain OSRM baseline (`/route/baseline`)
- an ORS reference or ORS-proxy baseline (`/route/baseline/ors`)

However, the local repo does not contain a checked-in universal aggregate statement such as "the system beats OSRM by X% overall across all freight trips." The report therefore explains:

- how the baselines are computed
- what realism multipliers are applied
- why the smart system can outperform them in principle
- what local scripts and UI compare panels can measure

It does not state a single universal win-rate percentage without direct local evidence.

## Plain-English System Overview

### What problem the project solves

At the simplest level, the software helps a user choose freight routes between UK origin and destination points. But unlike a normal map app, it does not assume that the shortest route is the best route. A freight operator usually cares about several competing goals at the same time:

- arriving quickly
- paying less money
- producing less CO2
- avoiding toll surprises
- handling weather and incident pressure
- understanding uncertainty, not just average ETA
- comparing scenario assumptions such as no sharing, partial sharing, and full sharing

This repository turns those competing goals into a multi-objective routing workflow.

### What a user can do from the frontend

A user can:

- place an origin, destination, and optional intermediate stops on a UK map
- request either a single recommended route or a full Pareto set of trade-off routes
- compare routes under different scenario modes
- inspect segment-level breakdowns for distance, duration, cost, emissions, tolls, and incidents
- optimize departure time across a time window
- build duty chains across multiple stops
- run experiments and save scenario-comparison bundles
- inspect live-source diagnostics and strict readiness state
- view a baseline comparison against plain OSRM and ORS-style reference routing
- inspect artifact manifests, provenance, signatures, and quality dashboards
- define custom vehicles in addition to built-in profiles

### UK-only scope

This is not a world router. The code and assets are explicitly UK-centered:

- default PBF source is the Geofabrik United Kingdom extract
- terrain handling uses a UK bounding box and fail-closed policy
- departure profiles, bank holidays, toll data, stochastic regimes, and calibration assets are UK-specific
- graph coverage report bounding box is approximately latitude 49.75 to 61.10 and longitude -8.75 to 2.25
- non-UK terrain behavior is explicitly handled as unsupported in strict mode

In plain language, the project has been narrowed to one geography so the models can be deeper and stricter instead of being globally shallow.

## Full Stack And Deployment Architecture

### Stack summary

Backend:

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic
- httpx
- numpy, scipy, pandas
- networkx
- rasterio
- shapely, pyproj, rtree
- osmium, ijson

Frontend:

- Next.js 16.1.4
- React 19.2.3
- TypeScript
- Tailwind/PostCSS
- Leaflet and react-leaflet
- Chart.js

Routing engine:

- OSRM in Docker, using MLD preparation

Operations and packaging:

- Docker Compose
- PowerShell developer scripts
- GitHub Actions backend CI
- uv for Python environment and command execution

### Deployment shape

The default local architecture is a five-service Docker Compose pipeline plus local frontend tooling:

1. `osrm_download` downloads the configured region PBF.
2. `osrm` prepares cached OSRM data if needed, then serves routing on port 5000.
3. `osrm_ready` waits for OSRM readiness.
4. `backend` serves the FastAPI API on port 8000.
5. `frontend` serves the Next.js app on port 3000.

The root `docker-compose.yml` wires these together. The shell scripts under `osrm/scripts` manage PBF download caching and OSRM preprocessing reuse.

### Why OSRM is still present even though there is a custom route graph

The system is hybrid, not anti-OSRM.

OSRM is used for:

- road-level geometry
- leg annotations
- baseline route generation
- via-path refinement once candidate corridors are known

The custom route graph is used for:

- fast feasibility checks
- candidate diversification
- corridor exploration
- generating multiple structurally different candidate paths before OSRM refinement

In plain language, the graph layer is the route idea generator and feasibility gate; OSRM is the road-geometry and baseline engine.

### Runtime directories

Tracked source directories:

- `backend/app`: runtime backend
- `backend/assets/uk`: bundled UK model assets
- `backend/data/raw/uk`: raw evidence used to build assets
- `backend/scripts`: model builders, fetchers, validators, benchmarks
- `backend/tests`: behavioral lock-in
- `frontend/app`: full Next.js app
- `docs`: supporting documentation
- `osrm`: OSRM profile and scripts
- `scripts`: root operational helpers

Generated/runtime directories present locally:

- `backend/out/model_assets`
- `backend/out/artifacts`
- `backend/out/provenance`
- `backend/out/oracle_quality`
- `backend/out/logs`
- `backend/out/analysis`
- `backend/out/capsule`
- `backend/out/experiments`

The `.gitignore` deliberately excludes most of `backend/out` and heavy OSRM cache data, but those outputs are still valid local evidence for this report.

## Frontend Capability Chapter

### Plain-English view

The frontend is a single-page analyst workbench rather than a minimal route form. The user is not just submitting an origin and destination; they are exploring trade-offs, scenario variants, uncertainty, diagnostics, and reproducibility artifacts.

### Main frontend shell

The root page is `frontend/app/page.tsx`. It orchestrates:

- request building
- streaming Pareto mode
- single-route mode
- baseline overlays
- scenario comparisons
- departure optimization
- duty-chain planning
- experiments CRUD and comparison
- tutorial steps
- diagnostics and developer tools
- oracle-quality dashboard rendering

This makes the frontend effectively a control room for the backend's optimization pipeline.

### Map behavior

`frontend/app/components/MapView.tsx` is the central visual interaction surface. Important behaviors include:

- UK-bounded map navigation
- origin, destination, and stop markers
- alternative route overlays
- baseline route overlays
- ORS/reference baseline overlays
- segment and incident overlays
- route failure overlays
- tutorial highlighting
- time-lapse position display

In non-jargon terms, the map is not decorative. It is the main way a user sees how route choices physically diverge and how incidents or stops are attached to them.

### Scenario editor and advanced controls

`ScenarioParameterEditor.tsx` exposes a large number of user-adjustable routing controls:

- scenario mode: `no_sharing`, `partial_sharing`, `full_sharing`
- route count / max alternatives
- objective weights for time, money, and CO2
- Pareto method: dominance or epsilon-constraint
- epsilon thresholds
- departure time
- cost toggles such as toll usage, carbon price, fuel multiplier
- terrain profile: flat, rolling, hilly
- stochastic configuration: enabled, seed, sigma, samples
- optimization mode: expected value or robust
- risk aversion
- emissions context: fuel type, Euro class, ambient temperature
- weather configuration
- incident simulation configuration

This is unusually rich for a route planner. It turns the frontend into an experimental interface suitable for dissertation demonstrations.

### Baseline comparison UI

The current working tree adds a dedicated baseline-comparison view:

- `frontend/app/components/RouteBaselineComparison.tsx`
- `frontend/app/lib/baselineComparison.ts`
- `frontend/app/api/route/baseline/route.ts`
- `frontend/app/api/route/baseline/ors/route.ts`

The comparison layer computes:

- ETA delta
- cost delta
- CO2 delta
- distance delta
- "Epic score" based on balanced ETA, cost, and CO2 improvement
- qualitative tier labels: `Epic`, `Strong`, `Positive`, `Limited gain`

This is a presentation layer over backend comparison outputs, not a separate optimization engine. The important thesis point is that the project is explicitly set up to defend its route choice against simpler baselines.

### Other frontend workflows

Other user-facing components include:

- `ParetoChart.tsx`: trade-off chart for non-dominated solutions
- `ScenarioComparison.tsx`: multi-scenario result comparison
- `DepartureOptimizerChart.tsx`: best departure-slot exploration
- `DutyChainPlanner.tsx`: multi-stop chain planning
- `CounterfactualPanel.tsx`: what-if adjustments such as fuel/carbon/departure shifts
- `SegmentBreakdown.tsx`: per-segment accounting
- `ScenarioTimeLapse.tsx`: route evolution over time/scenario playback
- `EtaTimelineChart.tsx`: ETA stage decomposition
- `ExperimentManager.tsx`: saved scenario experiment bundles
- `OracleQualityDashboard.tsx`: source quality and freshness summaries
- `TutorialOverlay.tsx`: guided walkthrough
- devtools panels for batch runs, custom vehicles, ops diagnostics, run inspection, and signature verification

### Frontend API proxy pattern

The Next.js app does not talk to backend logic directly in components. It uses route handlers under `frontend/app/api/...` as a proxy layer. That layer:

- forwards requests to the FastAPI backend
- centralizes timeout handling
- classifies transport failures into reason codes
- forwards `x-route-request-id` headers for diagnostics
- relays NDJSON streams for streaming Pareto responses

The helper `frontend/app/lib/backendFetch.ts` defines:

- `computeAttemptTimeoutMs()` default 1,200,000 ms
- `computeRouteFallbackTimeoutMs()` default 900,000 ms
- transport reason codes such as `backend_headers_timeout`, `backend_body_timeout`, `backend_connection_reset`, and `backend_unreachable`

### Frontend type system

`frontend/app/lib/types.ts` mirrors backend models and is substantial. It defines:

- request and response shapes
- route option metrics
- stream event types
- strict reason-code unions
- quality dashboard types
- run artifact types
- vehicle profile types

This is important because the frontend is not loosely typed UI glue. It mirrors the backend contract closely enough to make streaming, diagnostics, and detailed panels reliable.

## Backend Request-Flow Chapter

### Plain-English summary

The backend is a staged route-computation pipeline. A request does not go straight from "origin/destination" to "best route". It passes through readiness gates, data refresh logic, candidate generation, OSRM refinement, physical and economic modeling, Pareto filtering, scalar ranking, and artifact persistence.

### Backend routing pipeline in prose-table form

| Stage | What happens | Why it exists | Main failure gates |
| --- | --- | --- | --- |
| 1. Request validation | Pydantic models validate coordinates, weights, toggles, and advanced configs | stop bad inputs early | validation errors |
| 2. Health/readiness context | strict graph and live-data readiness can be checked | protect strict runtime behavior | `routing_graph_warming_up`, `routing_graph_warmup_failed`, source unavailability |
| 3. Live refresh/prefetch | scenario, fuel, carbon, departure, stochastic, toll, and terrain prerequisites are refreshed depending on mode | make route compute use fresh data | `live_source_refresh_failed`, data-specific strict errors |
| 4. Graph OD feasibility | route graph checks whether origin/destination are connected and inside covered graph space | avoid expensive OSRM work for impossible or poor-coverage trips | disconnected OD, coverage gap, warmup problems |
| 5. Candidate path generation | route graph runs K-shortest / A*-assisted search and corridor rescue logic | produce structurally different candidate corridors | no path, timeout, state-budget exhaustion |
| 6. Candidate diversification | routes are deduplicated/prefiltered using signatures, corridor grouping, and family rules | prevent many tiny variants of the same route | reduced candidate set |
| 7. OSRM refinement | OSRM fetches actual via routes for graph candidates | obtain realistic geometry and per-leg annotations | OSRM fetch failure, timeouts |
| 8. Option build | each refined route is converted into a rich `RouteOption` with cost, emissions, uncertainty, terrain, toll, incidents, and summaries | turn geometry into decision-ready options | terrain/toll/fuel/carbon/scenario/stochastic failures |
| 9. Pareto processing | options are filtered by dominance or epsilon-constraint and can be backfilled | keep trade-offs rather than one naive optimum | `epsilon_infeasible`, empty candidate set |
| 10. Scalar selection | one representative route can be selected using academic or modified profiles | allow a single recommendation after multi-objective analysis | no surviving options |
| 11. Response and artifacts | result, diagnostics, manifests, signatures, provenance, and optional reports are persisted | reproducibility and auditability | artifact I/O failures |

### Important backend endpoints

Core user-facing compute endpoints are:

- `POST /route`
- `POST /pareto`
- `POST /pareto/stream`
- `POST /api/pareto/stream`
- `POST /route/baseline`
- `POST /route/baseline/ors`
- `POST /departure/optimize`
- `POST /duty/chain`
- `POST /scenario/compare`
- `POST /batch/import/csv`
- `POST /batch/pareto`

Supporting endpoints include:

- readiness and health
- vehicle profile CRUD
- cache stats and cache clear
- metrics
- experiments CRUD
- oracle quality checks/dashboard
- run artifact and signature endpoints
- debug live-call traces

The full endpoint appendix appears later in this report.

### Lifespan behavior and graph warmup

`backend/app/main.py` starts the app with a lifespan that:

- wires OSRM client dependencies
- triggers route-graph warmup
- exposes warmup status through `/health/ready`

This is what "warming up the graph" means in this repo. It means loading or preparing the large UK route graph so later route requests are not blocked by first-touch graph construction.

The warmup is not cosmetic. The graph coverage report in `backend/out/model_assets/routing_graph_coverage_report.json` shows the local graph evidence contains:

- 16,782,614 nodes
- 17,271,476 edges
- about 4123.27 MB graph size
- UK bounding box coverage

That is large enough that warmup policy materially affects system behavior.

## Routing Mechanics Chapter

### The hybrid routing design

The repo uses two routing layers:

1. Route graph layer
2. OSRM refinement layer

The route graph layer answers:

- Is there a plausible path?
- Are origin and destination within covered graph reach?
- What are several structurally different corridor candidates?

The OSRM layer answers:

- What is the concrete road geometry?
- What are the leg-level duration and distance annotations?
- What is the simple baseline route if we do not want enriched modeling?

### Route graph behavior

`backend/app/routing_graph.py` contains the graph engine. Important features visible in the code and tests:

- warmup state machine
- binary cache support
- giant-component and fragmentation checks
- nearest-node distance validation
- adaptive maximum hops based on trip length
- optional A* heuristic
- OD feasibility precheck
- candidate-budget control
- state-space rescue logic
- long-corridor bypass rules

The A* heuristic is explicitly cited in code comments as Hart et al. (1968).

### K-shortest path generation

`backend/app/k_shortest.py` implements a Yen-style K-shortest route search with:

- deadlines
- state budgets
- detour caps
- repeat control
- heuristic support

The code explicitly cites Yen (1971). In thesis language, the repository is using a recognizable K-shortest-path family, but with practical runtime-budget constraints added for a production-like setting.

### Candidate diversification

The backend does not blindly keep every route returned by the search. It performs route-family and corridor prefiltering. The reason is simple: five tiny variations of the same motorway path do not form a meaningful Pareto set for a user.

Relevant controls from `.env.example` include:

- `ROUTE_CANDIDATE_PREFILTER_MULTIPLIER=3`
- `ROUTE_CANDIDATE_PREFILTER_MULTIPLIER_LONG=2`
- `ROUTE_CANDIDATE_PREFILTER_LONG_DISTANCE_THRESHOLD_KM=180`
- `ROUTE_GRAPH_LONG_CORRIDOR_MAX_PATHS=4`

In non-jargon terms, the system intentionally spends more search effort than it finally shows, then compresses those raw candidates into a more diverse decision set.

### Long-corridor rescue logic

Several settings show explicit concern for long trips:

- `ROUTE_GRAPH_FAST_STARTUP_LONG_CORRIDOR_BYPASS_KM=120`
- `ROUTE_GRAPH_LONG_CORRIDOR_THRESHOLD_KM=150`
- `ROUTE_GRAPH_SKIP_INITIAL_SEARCH_LONG_CORRIDOR=true`
- `ROUTE_GRAPH_REDUCED_INITIAL_FOR_LONG_CORRIDOR=true`

This indicates an engineering lesson already learned in the repo: long freight corridors need different search tactics than short urban trips. The code therefore treats long corridors as a special regime rather than assuming one search budget fits all.

### Search budgets and fail-fast semantics

The graph search is strongly budgeted:

- max state budget
- per-hop state budget
- retry multiplier and cap
- initial, retry, and rescue timeouts
- OD feasibility timeout
- status check timeout

This is not only about speed. It is about bounded failure. Instead of hanging indefinitely, the backend tries to fail with a recognizable reason code.

### Route caching

`backend/app/route_cache.py` implements an in-memory TTL and LRU-like ordered cache with:

- configurable TTL
- max entry count
- hits/misses/evictions statistics
- deep-copy get/set behavior

This supports repeated route exploration in the frontend and exposes cache diagnostics through `/cache/stats`.

### OSRM client behavior

`backend/app/routing_osrm.py` wraps OSRM with:

- route API fetches
- support for boolean or numeric alternatives
- optional via points
- optional `exclude`
- bounded retry logic

The wrapper also compensates for practical issues such as alternative parameter compatibility and docker/host addressing hints. This is another example of the project being engineering-focused rather than purely theoretical.

### Baseline route mechanics

The current backend includes two explicit baseline endpoints:

- `POST /route/baseline`
- `POST /route/baseline/ors`

`/route/baseline`:

- uses OSRM directly
- bypasses strict enriched route-building logic
- returns a quick baseline note and realism multipliers

`/route/baseline/ors`:

- uses ORS if configured
- can fall back to an OSRM proxy baseline if ORS is unavailable and proxy fallback is enabled
- returns method labels such as `ors_reference` or `ors_proxy_baseline`

The repo-local test `backend/tests/test_route_baseline_api.py` explicitly checks that:

- OSRM baseline supports waypoints
- the OSRM baseline bypasses strict graph warmup gate
- ORS baseline can require an API key when proxy fallback is disabled
- ORS baseline can fall back to proxy mode when allowed

### Why this can outperform plain OSRM or ORS

In structural terms, the smart route can outperform base OSRM/ORS because it is optimizing a larger decision objective:

- not just shortest path
- not just one provider's notion of duration
- not just distance

It adds:

- scenario pressure
- departure-time contextualization
- terrain uplift
- toll topology and tariffs
- fuel and carbon pricing
- stochastic uncertainty and robust ranking
- Pareto selection instead of single-criterion shortest-path commitment

This does not guarantee every smart route is always faster or shorter. It means the system is capable of selecting routes that are operationally better once those extra costs and risks matter.

## Math And Algorithm Chapter

### Objective space

The main route objectives are:

- duration
- monetary cost
- emissions

Distance is also used in selection and diagnostics, even when it is not one of the core Pareto axes.

The route objective vector can therefore be thought of as:

`f(route) = [duration_s, monetary_cost, emissions_kg]`

and, for some ranking profiles, an auxiliary dimension:

`distance_km`

### Pareto dominance

`backend/app/pareto.py` implements standard Pareto dominance for minimization:

- route A dominates B if A is no worse in every objective
- and strictly better in at least one objective

This is the academically standard notion of non-dominance. The implementation is O(n^2), which the code explicitly says is acceptable because candidate sets are small after upstream filtering.

### Epsilon-constraint filtering

`backend/app/pareto_methods.py` supports epsilon-constraint filtering. In practical terms, this lets the user say something like:

- "show me routes whose cost is below this level"
- "or emissions below this threshold"
- "while still respecting trade-off structure"

This is useful in freight planning because some limits are contractual or regulatory rather than preference-based.

### Crowd-distance and knee-point logic

`backend/app/pareto_methods.py` also adds:

- crowding-distance truncation inspired by NSGA-II
- knee-score annotation

The code comments explicitly cite Deb et al. (2002) for NSGA-II crowding truncation. This matters because the repo is not just returning all non-dominated points. It is also trying to keep a useful spread and identify knee-like compromise routes.

### Academic selection formulas used in the repo

The route-selection comments in `backend/app/main.py` and the frontend mirror in `frontend/app/lib/weights.ts` explicitly identify the academic references:

- Weighted-sum: Marler and Arora (2010)
- Augmented Tchebycheff: Steuer and Choo (1983)
- VIKOR compromise ranking: Opricovic and Tzeng (2004)
- Distance criterion inspiration: Martins (1984)
- Knee-oriented signal inspiration: Branke et al. (2004)
- Entropy reward basis: Shannon (1948)

### Academic formulas versus engineering modifications

| Profile in repo | Academic baseline | Repo formula idea | Why it was modified |
| --- | --- | --- | --- |
| `academic_reference` | Weighted-sum | normalized weighted sum over duration, cost, emissions | simplest reference point |
| `academic_tchebycheff` | Augmented Tchebycheff | regret plus rho-weighted weighted-sum epsilon term | classic compromise/risk-of-worst-objective view |
| `academic_vikor` | VIKOR | Q score from utility S and regret R with parameter `v` | classic compromise ranking |
| `modified_hybrid` | weighted-sum plus regret/balance | weighted sum + regret weight + balance penalty | discourages lopsided routes |
| `modified_distance_aware` | weighted-sum foundation with added heuristics | weighted sum + regret + balance + distance + ETA-distance + knee - entropy reward | adds practical route-shape and compromise cues |
| `modified_vikor_distance` | VIKOR foundation with added heuristics | VIKOR Q + balance + distance + ETA-distance + knee - entropy reward | default profile; compromise score plus freight-specific tie-break signals |

### Exact extra terms added by the modified profiles

The modified profiles introduce the following engineering terms beyond the academic baselines:

- `balance`: standard-deviation-like spread between normalized objective values
- `nd`: normalized distance penalty
- `etaDistancePenalty`: square-root interaction between time and distance
- `kneePenalty`: average pairwise objective-gap penalty
- `entropyReward`: reward for more evenly distributed improvement across time, money, and CO2

In symbols, the default `modified_vikor_distance` implemented in the frontend mirror is:

`score = vikorQ + balanceWeight * balance + distanceWeight * nd + etaDistanceWeight * etaDistancePenalty + kneeWeight * kneePenalty - entropyWeight * entropyReward`

with defaults from `.env.example`:

- regret weight = 0.35
- balance weight = 0.10
- distance weight = 0.22
- ETA-distance weight = 0.18
- entropy weight = 0.08
- knee weight = 0.12
- Tchebycheff rho = 0.001
- VIKOR v = 0.5

The repo comments explicitly say these modified profiles are transparent engineering blends, not novel theory.

### Risk and robust optimization

The repo supports:

- `expected_value` optimization mode
- `robust` optimization mode

`backend/app/risk_model.py` provides:

- quantiles
- CVaR
- normalized weighted utility
- robust objective variants such as `cvar_excess`, `entropic`, and `downside_semivariance`

In plain English, expected-value mode chooses based on average outcomes. Robust mode penalizes routes whose bad-tail outcomes are too severe, even if their averages look good.

### Uncertainty model

`backend/app/uncertainty_model.py` converts deterministic route metrics into sampled distributions. It uses:

- deterministic seeding by route signature and departure slot
- local UK time-slot and day-kind context
- stochastic regime lookup
- posterior regime candidate fallback logic
- quantiles and CVaR summaries

The `UncertaintySummary` includes:

- means
- standard deviations
- q50, q90, q95
- cvar95
- utility mean, utility q95, utility cvar95
- robust score
- sample count and sigma clipping diagnostics

This is much more than "ETA +/- some number." The backend tracks how much sampling itself had to be clipped or normalized.

### Scenario context similarity math

`backend/app/scenario.py` builds context keys using:

- geohash5 / corridor bucket
- local hour
- day kind
- road mix
- vehicle class
- weather regime

The scenario profile asset records a weighted L1 similarity design with weights:

- geo distance 0.34
- hour distance 0.12
- day penalty 0.12
- weather penalty 0.12
- road penalty 0.16
- vehicle penalty 0.10
- road-mix distance 0.04

This is how the system decides which empirical context is "similar enough" to apply when an exact identity match is absent.

## Physics And Cost-Model Chapter

### Plain-English summary

The project tries to make route metrics physically meaningful. Distance and ETA come from routing, but cost and emissions are rebuilt from more specific drivers:

- vehicle profile
- segment speed
- grade/terrain
- fuel or electricity
- tolls
- carbon pricing
- departure-time uplift
- weather
- incidents

### Vehicle model

`backend/app/vehicles.py` defines the vehicle-profile schema. Built-in UK vehicle profiles include:

- `van`
- `rigid_hgv`
- `artic_hgv`
- `ev_hgv`

Each profile includes:

- mass
- cost per km
- cost per hour
- idle emissions
- powertrain
- toll class and axle class
- fuel surface class
- risk bucket
- stochastic bucket
- terrain parameters

This matters because the route is not scored in the abstract. It is scored for a specific freight vehicle.

### Fuel and energy model

`backend/app/fuel_energy_model.py` uses:

- fuel consumption surfaces
- uncertainty surfaces
- live fuel snapshot
- fuel type and ambient temperature
- load, speed, grade, and temperature interpolation

Hard-coded emissions constants visible in code include:

- diesel: 2.68 kg CO2 per liter
- petrol: 2.31 kg CO2 per liter
- LNG: 1.51 kg CO2 per liter

EVs are handled through:

- kWh per km
- grid CO2 intensity

This is a strong engineering choice: the project does not force EVs into an ICE emissions model.

### Terrain mechanics

`backend/app/terrain_physics.py` is where the repo becomes explicitly physical. It includes:

- gravity
- air density
- rolling resistance
- aerodynamic drag
- grade force
- drivetrain efficiency
- regenerative efficiency

At segment level, the duration multiplier is driven by force terms that depend on:

- vehicle mass
- rolling resistance coefficient
- drag area
- speed
- slope

In simplified form, the physics being approximated is:

- rolling force proportional to `m g c_rr`
- grade force proportional to `m g sin(theta)`
- aerodynamic force proportional to `0.5 rho CdA v^2`

The repo then converts those forces into duration and emissions multipliers rather than running a full vehicle dynamics simulation. This is an engineering simplification that preserves directional realism without turning the backend into a high-cost physics solver.

### Terrain coverage and fail-closed policy

`backend/app/terrain_dem.py` and `backend/app/terrain_dem_index.py` show several important design decisions:

- the route is densified into samples
- samples are taken from a terrain manifest or live DEM tiles
- UK coverage ratio is computed
- in strict mode, insufficient UK coverage raises a structured error
- non-UK routes can become `terrain_region_unsupported`

Defaults from `.env.example`:

- `TERRAIN_DEM_FAIL_CLOSED_UK=true`
- `TERRAIN_DEM_COVERAGE_MIN_UK=0.96`
- sample spacing 180 m
- longer routes use larger spacing and sample caps

In plain English, terrain is not optional decoration. If the system says terrain matters, it insists on actually having enough terrain data.

### Weather modifiers

`backend/app/weather_adapter.py` applies profile-based multipliers:

- clear: speed 1.00, incidents 1.00
- rain: speed 1.08, incidents 1.15
- storm: speed 1.20, incidents 1.50
- snow: speed 1.28, incidents 1.80
- fog: speed 1.12, incidents 1.25

These are scaled by intensity. They are not weather forecasts themselves; they are weather-to-routing impact transforms.

### Incident simulation

`backend/app/incident_simulator.py` supports synthetic events:

- dwell
- accident
- closure

using:

- rates per 100 km
- delay sizes
- seeded randomness by route key

Important strict-runtime nuance: synthetic incident simulation is disabled inside strict live route construction. That is a strong evidence point for the thesis. The project allows simulation as an experiment tool, but avoids mixing it into strict live operational outputs.

### Tolls

`backend/app/toll_engine.py` matches route geometry against UK toll-segment seeds and tariff rules. The system does not merely apply a flat toll multiplier. It uses:

- toll topology
- tariff rule tables
- route geometry matching
- confidence calibration

The toll confidence asset is explicitly empirical and logistic-calibrated, with reliability bins. This means the toll subsystem is trying to measure not just predicted toll cost, but how trustworthy that toll inference is.

### Carbon pricing

`backend/app/carbon_model.py` loads carbon schedule payloads and applies strict provenance checks. It rejects non-empirical provenance labels in strict mode if they contain terms such as:

- synthetic
- heuristic
- legacy
- interpolated
- simulated
- wobble

This is a clear example of the repo preferring no answer over a weak answer when strict policy is active.

### Counterfactuals

`build_option()` in `backend/app/main.py` generates counterfactual analyses such as:

- fuel +10%
- carbon price +0.10 per kg
- departure +2 hours
- scenario shift

This means the output is not only "here is the route." It is also "here is how sensitive this route is to plausible policy or market changes."

## Live/Strict Data Chapter

### Plain-English summary

The live/strict system is one of the central contributions of this repository. Many route planners quietly fall back to defaults when live feeds fail. This one is designed to expose, log, and often refuse that fallback.

### Strict fail-closed policy

The backend's reason-code contract in `backend/app/model_data_errors.py` includes frozen codes such as:

- `routing_graph_unavailable`
- `routing_graph_fragmented`
- `routing_graph_disconnected_od`
- `routing_graph_coverage_gap`
- `routing_graph_no_path`
- `routing_graph_precheck_timeout`
- `routing_graph_warming_up`
- `routing_graph_warmup_failed`
- `live_source_refresh_failed`
- `departure_profile_unavailable`
- `holiday_data_unavailable`
- `stochastic_calibration_unavailable`
- `scenario_profile_unavailable`
- `scenario_profile_invalid`
- `risk_normalization_unavailable`
- `risk_prior_unavailable`
- `terrain_region_unsupported`
- `terrain_dem_asset_unavailable`
- `terrain_dem_coverage_insufficient`
- `toll_topology_unavailable`
- `toll_tariff_unavailable`
- `toll_tariff_unresolved`
- `fuel_price_auth_unavailable`
- `fuel_price_source_unavailable`
- `vehicle_profile_unavailable`
- `vehicle_profile_invalid`
- `carbon_policy_unavailable`
- `carbon_intensity_unavailable`
- `epsilon_infeasible`
- `no_route_candidates`
- `baseline_route_unavailable`
- `baseline_provider_unconfigured`
- `model_asset_unavailable`

This frozen set is thesis-significant because it formalizes failure semantics as part of the public interface.

### Live source matrix

| Source family | Purpose | Repo evidence | Freshness / strict policy | Failure style |
| --- | --- | --- | --- | --- |
| WebTRIS | live traffic pressure | `live_scenario_context`, preflight summary | required for strict scenario context unless partial-source strictness is allowed | scenario/live refresh failure |
| Traffic England | incident/congestion pressure | `live_scenario_context`, preflight summary | required in strict live scenario context | scenario/live refresh failure |
| DfT raw counts | contextual traffic grounding and empirical departure building | live context plus raw/asset scripts | required for strict scenario/departure evidence | scenario/departure failure |
| Open-Meteo | current weather severity and regime | live scenario context | required for strict live scenario context | scenario/live refresh failure |
| GOV.UK bank holidays | day-kind correctness | bank-holiday loader and preflight | strict holiday source required | `holiday_data_unavailable` |
| GitHub raw scenario profiles | live scenario coefficients | `.env.example`, preflight summary | strict URL and freshness enforced | `scenario_profile_unavailable` or invalid |
| GitHub raw fuel prices | live fuel snapshot | `.env.example`, preflight | strict signature and max-age enforced | `fuel_price_source_unavailable` or auth failure |
| GitHub raw carbon schedule | policy price schedule | `.env.example`, carbon model | strict URL, uncertainty distribution, freshness, provenance checks | `carbon_policy_unavailable` |
| GitHub raw departure profiles | contextual time-of-day multipliers | `.env.example`, departure loader | strict URL/fallback policy | `departure_profile_unavailable` |
| GitHub raw stochastic regimes | route uncertainty regimes | `.env.example`, stochastic loader | strict URL/fallback policy | `stochastic_calibration_unavailable` |
| GitHub raw toll topology/tariffs | toll route matching and pricing | `.env.example`, toll loader | strict URL/fallback policy | `toll_topology_unavailable`, `toll_tariff_unavailable` |
| S3 terrain tiles | live DEM fallback/request-time terrain sampling | `.env.example`, terrain live loader | strict host allow-list and tile freshness; fallback disabled by default | terrain DEM failures |

### Preflight evidence currently present locally

`backend/out/model_assets/preflight_live_runtime.json` records a local strict preflight run at `2026-03-05T02:10:00Z` with:

- `required_ok: true`
- zero required failures
- scenario live context overall coverage 1.0
- scenario profile contexts: 192
- toll tariff rule count: 220
- toll topology segment count: 28
- stochastic regimes: 18
- departure profile region count: 11
- bank holiday count: 134
- carbon price per kg: 0.101

This is a valuable evidence artifact because it shows the strict runtime design was not only coded; it was exercised locally.

### Request-time live diagnostics

`backend/app/live_call_trace.py` records:

- expected external/live calls
- actual call entries
- cache hits and stale-cache usage
- request and response headers
- retry counts and backoff totals
- blocked stages and unmet expectations

The frontend can fetch this with `/debug/live-calls/{request_id}` and expose it in ops diagnostics.

This is one of the most thesis-worthy operational features in the repo. It turns "the route used live data" from a vague claim into inspectable evidence.

### Freshness and retry behavior

Important global retry settings in `.env.example`:

- max attempts = 6
- retry deadline = 30,000 ms
- backoff base = 200 ms
- backoff max = 2500 ms
- jitter = 150 ms
- retryable status codes = `429,500,502,503,504`

Important scenario strictness settings:

- `LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT=false`
- `LIVE_SCENARIO_MIN_SOURCE_COUNT_STRICT=4`
- `LIVE_SCENARIO_MIN_COVERAGE_OVERALL_STRICT=1.0`

Important terrain strictness settings:

- live terrain URL required in strict mode
- signed fallback disallowed by default
- allowed host list restricted to S3
- tile max age 7 days

### Why the repo is designed this way

The strict/live subsystem suggests a clear project rationale:

- freight decisions are high-cost enough that silent stale fallbacks are dangerous
- the user needs explicit reason codes, not ambiguous degraded behavior
- reproducibility matters, so feeds are paired with signatures, manifests, and provenance
- the thesis needs defensible claims about what data was or was not available

That design rationale is internally consistent across code, tests, docs, and artifacts.

## Data And Calibration Chapter

### Plain-English summary

The repository's intelligence is not only in algorithms. It is also in its calibrated UK data assets. The code repeatedly separates:

- raw evidence
- built assets
- compiled assets
- runtime outputs

This is a healthy data-science engineering pattern because it preserves provenance.

### Bundled UK asset families

The tracked UK asset directory `backend/assets/uk` contains 18 tracked files, including:

- carbon intensity hourly data
- carbon price schedule
- departure counts empirical CSV
- departure profiles
- fuel consumption surface
- fuel prices
- fuel uncertainty surface
- risk normalization references
- scenario profiles
- stochastic regimes
- stochastic residual priors
- stochastic residual empirical CSV
- terrain DEM grid
- toll confidence calibration
- toll tariffs
- toll topology
- default vehicle profiles
- a local OSM file placeholder/payload

### Raw evidence families

The tracked raw UK directory contains 342 tracked files, dominated by toll evidence:

- 220 raw toll-classification JSON files
- 105 raw toll-pricing JSON files
- scenario JSONL corpora
- DfT raw counts
- fuel raw JSON
- carbon intensity raw JSON
- stochastic residual raw CSV
- summary/state files for collectors

This is one of the clearest signs that the project tries to keep the model grounded in inspectable evidence rather than hiding everything behind opaque bundled assets.

### Scenario profile calibration facts

From `backend/assets/uk/scenario_profiles_uk.json`:

- version: `scenario_profiles_uk_v2_live`
- source: `free_live_apis+holdout_fit`
- calibration basis: `empirical_live_fit`
- context count: 192
- coverage: 1.0
- split strategy: `temporal_forward_plus_corridor_block`
- blocked corridor count in holdout metadata: 4
- observed mode row share: 0.5
- projection dominant context share: 0.5

The transform section shows fitted live-feature weights for:

- traffic pressure
- incident pressure
- weather pressure

and policy adjustment weights for:

- duration multiplier
- incident rate multiplier
- incident delay multiplier
- fuel consumption multiplier
- emissions multiplier
- stochastic sigma multiplier

### Sharing modes explained without jargon

The three scenario modes are:

- `no_sharing`: assume no meaningful coordination or resource-sharing improvement; keep route pressure relatively high.
- `partial_sharing`: assume some coordination benefit; reduce several penalties but not all the way to ideal.
- `full_sharing`: assume near-best coordination; penalties are pushed close to their floor values.

The asset's `mode_effect_scale` gives:

- `no_sharing = 1.0`
- `partial_sharing = 0.569279`
- `full_sharing = 0.05`

In a representative context from the asset:

- `no_sharing` duration multiplier is 1.474464
- `partial_sharing` duration multiplier is 1.246721
- `full_sharing` duration multiplier is 0.98

and similarly for incident rate, incident delay, fuel consumption, emissions, and stochastic sigma.

So the sharing modes do not merely rename scenarios. They alter multiple downstream multipliers that affect:

- ETA
- simulated incident frequency
- incident delay severity
- fuel consumption
- emissions
- uncertainty spread

### Stochastic calibration facts

From `backend/assets/uk/stochastic_regimes_uk.json`:

- empirical calibration basis
- holdout coverage 1.0
- PIT mean about 0.5166
- CRPS about 0.4709
- duration MAPE about 0.1401
- holdout rows 50,000

The asset stores multiple regimes with:

- factor scales
- correlations
- quantile mappings

This means the uncertainty system is not a single global Gaussian noise knob.

### Departure profile calibration facts

From `backend/assets/uk/departure_profiles_uk.json`:

- version `uk-v4-contextual-empirical`
- empirical calibration basis

The runtime departure model chooses profile rows by:

- UK-local day kind
- region bucket / geohash region
- road bucket
- route shape hints

This is much richer than a single rush-hour multiplier.

### Fuel and carbon facts

From `backend/assets/uk/fuel_prices_uk.json`:

- diesel 1.563 GBP/L
- petrol 1.51 GBP/L
- LNG 1.015 GBP/L
- grid electricity 0.248 GBP/kWh

From `backend/assets/uk/carbon_price_schedule_uk.json`:

- central/high/low schedules from 2025 to 2050
- uncertainty bands
- WTW/LCA scope factors
- EV grid-intensity by region and hour

### Risk normalization references

`backend/assets/uk/risk_normalization_refs_uk.json` supplies contextual reference scales for:

- duration
- money
- emissions

often per km and per vehicle class. This is important because robust and utility-based ranking depend on normalization. Without reference scales, one objective could numerically dominate simply because of units.

### Toll calibration facts

`backend/assets/uk/toll_confidence_calibration_uk.json` is labelled:

- version `uk-toll-confidence-v2-empirical`

and uses:

- fixture/test-backed empirical source
- logistic model coefficients
- reliability bins

That is better described as calibrated confidence scoring, not just a heuristic confidence guess.

### Vehicle profile facts

`backend/assets/uk/vehicle_profiles_uk.json` stores the built-in operational vehicle classes with:

- aliases
- terrain parameters
- toll classes
- fuel surface classes
- stochastic buckets
- risk buckets

This lets one route engine serve multiple fleet types without pretending they behave the same.

### Calibration matrix

| Calibration family | Local default / evidence | Practical effect |
| --- | --- | --- |
| baseline realism | OSRM duration 1.16, distance 1.13; ORS duration 1.24, distance 1.18 | makes baseline comparisons more conservative/operationally realistic |
| selection profile | `modified_vikor_distance` | default single-route selector after Pareto |
| modified selection weights | regret 0.35, balance 0.10, distance 0.22, ETA-distance 0.18, entropy 0.08, knee 0.12 | shapes which compromise route gets chosen |
| VIKOR parameter | `v = 0.5` | balances group utility and regret evenly |
| Tchebycheff epsilon term | `rho = 0.001` | small tie-break toward overall weighted performance |
| scenario mode effect scale | no sharing 1.0, partial 0.569279, full 0.05 | controls how strongly scenario modes compress penalties |
| scenario policy gain | 0.05 in asset transform blocks | scales how live pressures move multipliers |
| scenario context similarity weights | geo 0.34, hour 0.12, day 0.12, weather 0.12, road 0.16, vehicle 0.10, road-mix 0.04 | determines nearest-context matching |
| strict scenario source threshold | min source count 4, overall coverage 1.0 | decides whether live scenario context is acceptable |
| terrain coverage threshold | 0.96 UK minimum | fail-closed cutoff for terrain use |
| terrain sample spacing | 180 m default, 320 m long-route | detail versus runtime tradeoff |
| long-route terrain sample caps | 900 long-route, 1500 general cap | bounds DEM cost |
| live terrain freshness | 7-day tile max age | controls whether live DEM tiles count as valid |
| graph nearest-node threshold | 10,000 m max | coverage-gap fail condition |
| graph giant-component thresholds | 50,000 nodes and 0.20 ratio | graph integrity quality gate |
| graph search budgets | max state budget 1,200,000; retry cap 8,000,000 | controls graph search effort and rescue behavior |
| graph heuristics | A* on; heuristic max speed 220 kph | speeds candidate search |
| candidate prefilter multipliers | 3 normal, 2 long | controls over-generation before dedupe |
| segment caps | 160 normal, 40 long | truncates segment breakdown detail for large routes |
| stochastic sample controls | user-configurable seed/sigma/samples; asset holdout rows 50,000 | controls uncertainty smoothness and reproducibility |
| fuel price age | 14 days max in strict mode | determines fuel snapshot validity |
| carbon schedule validity | must include uncertainty distribution and empirical provenance | avoids using weak carbon schedules in strict mode |
| toll confidence calibration | empirical logistic coefficients and reliability bins | estimates how trustworthy toll assignments are |
| departure profile contextualization | region, road bucket, day kind, route shape | tunes time-of-day multiplier to route context |
| weather multipliers | clear/rain/storm/snow/fog profiles | transforms weather scenarios into ETA/incident changes |
| incident rates | dwell/accident/closure per 100 km and delay values | calibrates synthetic incident experiments |
| route Pareto backfill | enabled; minimum alternatives 6 | prevents very small frontiers from starving the UI |
| live retry policy | 6 attempts, 30 s deadline, 200-2500 ms backoff | balances strictness and practical recoverability |

## Comparison Chapter: Base OSRM, ORS, And The Smart Router

### What the project adds beyond base OSRM

Base OSRM gives:

- shortest or fastest route geometry
- leg distance and duration annotations
- alternative route support

This project adds:

- graph-based corridor diversification before OSRM refinement
- UK-specific scenario multipliers
- departure-time contextualization
- terrain and grade effects
- fuel and carbon pricing
- toll topology and tariff reasoning
- route uncertainty distributions
- robust objective support
- Pareto filtering and compromise selection
- run artifacts, provenance, manifests, and signatures

The simplest summary is that OSRM is a route engine, while this repository is a freight route decision system built around an OSRM route engine.

### What the project adds beyond ORS

ORS can provide an external reference route, but the local codebase still layers freight-specific decision logic around that baseline:

- ORS is treated as a baseline provider, not as the whole decision architecture
- ORS baseline can be disabled, proxied, or compared against
- the smart route still applies project-specific scoring, robustness, data strictness, and diagnostics

The thesis point is therefore not "OSRM bad, ORS bad." It is "provider routes are only the starting point for the project's freight decision logic."

### How the baselines are computed

The repo uses explicit baseline realism multipliers from `.env.example`:

- `ROUTE_BASELINE_DURATION_MULTIPLIER=1.16`
- `ROUTE_BASELINE_DISTANCE_MULTIPLIER=1.13`
- `ROUTE_ORS_BASELINE_DURATION_MULTIPLIER=1.24`
- `ROUTE_ORS_BASELINE_DISTANCE_MULTIPLIER=1.18`

These indicate that the project does not present raw provider outputs as final truth. It deliberately inflates them to form a more realistic baseline comparison frame.

In plain language, the code assumes a direct provider route is a useful reference, but not a fully operational freight estimate.

### What "better than the baseline" means in this repo

The frontend baseline-comparison layer and backend baseline endpoints frame improvement in terms of:

- ETA improvement
- cost improvement
- CO2 improvement
- distance change

The current frontend helper `baselineComparison.ts` computes a composite "Epic score" from normalized ETA, cost, and CO2 gains, then places the route in a tier.

This means the project's own notion of "beats the baseline" is not strictly shortest-time. It is explicitly multi-criteria.

### Evidence strength versus evidence gap

Strong local evidence:

- the system has dedicated OSRM and ORS baseline endpoints
- the frontend has explicit baseline-comparison UI
- local tests validate baseline endpoint behavior
- realism multipliers are configurable and checked in

Weak or absent local evidence:

- no single checked-in aggregate benchmark result states a universal percentage win over OSRM
- no local checked-in corpus summary proves one global improvement number over ORS

The defensible thesis wording is therefore:

- the repository contains explicit comparison machinery and user-facing comparison workflows
- it is architecturally capable of outperforming base routing once freight-specific costs and constraints matter
- but a universal percentage claim would need a separately reported benchmark corpus result

### Why the smart system can win in general

It can win generally because plain provider routes optimize a narrower surrogate target, while the smart system optimizes a broader operational target. If a route is slightly longer but avoids tolls, smoother terrain, peak departure pressure, or bad uncertainty tails, it can be better for freight operations even when it is not the absolute shortest geometry.

## Quality, Testing, And Reproducibility Chapter

### Quality philosophy

The repository shows a strong preference for testable, scriptable, notebook-free engineering. That is visible across:

- CI lanes
- deterministic script entry points
- build/fetch/validate scripts
- manifests and signatures
- separate raw-versus-built assets
- artifact persistence

### CI lanes

`../.github/workflows/backend-ci.yml` defines two important lanes:

`fast-lane`

- uses `STRICT_RUNTIME_TEST_BYPASS=1`
- runs a deterministic subset of backend tests
- targets quick regression detection

`strict-live-lane`

- uses `STRICT_RUNTIME_TEST_BYPASS=0`
- disables signed-fallback allowances for key feeds
- checks strict reason-code parity and strict subsystems such as terrain, departure, stochastic, metrics, and robust mode

This is a strong sign that the project intentionally tests both relaxed deterministic behavior and actual strict behavior.

### Test coverage themes

The 71 tracked test files cover the following major families:

- API and streaming behavior
- cost model and emissions
- counterfactuals
- departure optimization and profiles
- duty chains and multileg logic
- experiments and scenario comparison
- incidents and weather
- K-shortest search and graph feasibility
- live retry policy and call-trace rollups
- metrics and route cache behavior
- Pareto methods and strict frontier behavior
- robust optimization
- run stores and reporting
- strict reason-code contract
- terrain coverage, runtime budget, fail-closed, and non-UK behavior
- toll engine
- tooling scripts
- vehicle custom profiles

### Benchmark and quality scripts

The backend script layer includes:

- `backend/scripts/benchmark_model_v2.py`
- `backend/scripts/benchmark_batch_pareto.py`
- `backend/scripts/score_model_quality.py`
- `backend/scripts/run_sensitivity_analysis.py`
- `backend/scripts/run_robustness_analysis.py`
- `backend/scripts/check_eta_concept_drift.py`
- `backend/scripts/validate_graph_coverage.py`

Important local quality evidence from docs and scripts:

- warm-cache `/route` and `/pareto` p95 target under 2000 ms
- batch Pareto benchmark can run in `inprocess-fake` mode or against live backend
- model-quality scoring enforces 95-style thresholds across many subsystems
- robustness analysis varies seeds
- sensitivity analysis varies fuel, carbon, and toll assumptions

### Reproducibility mechanisms

The repo's reproducibility machinery includes:

- run manifests
- scenario manifests
- provenance logs
- signed payloads using HMAC-SHA256
- stored artifacts such as `backend/out/artifacts/<run_id>/results.json`, `backend/out/artifacts/<run_id>/results.csv`, `backend/out/artifacts/<run_id>/metadata.json`, `backend/out/artifacts/<run_id>/routes.geojson`, `backend/out/artifacts/<run_id>/results_summary.csv`, and `backend/out/artifacts/<run_id>/report.pdf`
- reproducibility capsule script `scripts/demo_repro_run.ps1`

`backend/app/signatures.py` uses canonical JSON serialization plus HMAC-SHA256. `backend/app/run_store.py` signs manifests. `backend/app/provenance_store.py` writes timestamped event histories.

This is more than "save the output." It is a chain of evidence about how the output was produced.

### Oracle quality dashboard

`backend/app/oracle_quality_store.py` writes:

- NDJSON check records
- a JSON summary
- dashboard CSV

and aggregates:

- pass rate
- schema failures
- signature failures
- stale counts
- average latency
- last observed times

That is a notable operational-quality feature because it lets the system score the reliability of the feeds it depends on.

### Notebook-free workflow

The docs explicitly document a notebook-free workflow. `notebooks/NOTEBOOKS_POLICY.md` reinforces that scripts, manifests, and checked artifacts are preferred over ad hoc analysis notebooks. For a thesis, this is useful because it shows a deliberate choice toward reproducible engineering pipelines rather than manual, fragile exploration.

## Design Rationale And Tradeoffs

### Why UK-only

The project is narrow by design. UK-only scope makes it feasible to:

- maintain stronger graph coverage checks
- use UK bank holidays
- calibrate departure profiles by UK corridor/region
- keep toll topology and tariff truth manageable
- make terrain fail-closed within a known bounding box

The tradeoff is obvious: geographical generality is sacrificed for model depth and reliability.

### Why strict/live-first

The repo repeatedly chooses explicit strictness over silent fallback. This reduces false confidence but increases the chance of surfacing an error instead of a route. For freight decision support, that is a reasonable tradeoff because a confident wrong answer can be worse than a visible failure.

### Why hybrid graph plus OSRM

Using only OSRM alternatives would make corridor diversification and early feasibility logic weaker. Using only a custom graph would make concrete road geometry and baseline comparison weaker. The hybrid model combines:

- graph-level exploration
- provider-level route realization

The tradeoff is architectural complexity, but the benefit is richer route-option generation.

### Why modify academic formulas

Pure academic scalarisation formulas are often too brittle for single-route product selection after Pareto generation. The repo adds distance, balance, knee, and entropy terms because real users need a representative route that looks operationally sensible, not only mathematically minimal under a single formula. The cost is theoretical purity. The benefit is better product behavior.

### Why so many diagnostics and artifacts

The project is trying to be explainable and auditable. That is why it stores:

- live call traces
- manifests
- signatures
- provenance
- reports
- quality dashboards
- artifact bundles

The tradeoff is more code and more moving parts. The benefit is stronger evidence and easier debugging.

### Known limitations

From the local evidence, the main limitations are:

- UK-only support
- heavy dependence on prepared assets and strict feed availability
- graph warmup and large graph size can dominate startup behavior
- no locally checked-in universal aggregate OSRM/ORS improvement percentage
- engineering blends in route selection are practical rather than formally optimal in a theorem-driven sense

## Appendix A: Backend Endpoint Inventory

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | basic root banner / smoke endpoint |
| `GET` | `/health` | simple health probe |
| `GET` | `/health/ready` | strict readiness, graph warmup, and live readiness detail |
| `GET` | `/debug/live-calls/{request_id}` | per-request live-call trace and expected-call rollup |
| `GET` | `/vehicles` | built-in vehicle list |
| `GET` | `/vehicles/custom` | locally stored custom vehicles |
| `POST` | `/vehicles/custom` | create custom vehicle |
| `PUT` | `/vehicles/custom/{vehicle_id}` | update custom vehicle |
| `DELETE` | `/vehicles/custom/{vehicle_id}` | delete custom vehicle |
| `GET` | `/metrics` | backend endpoint metrics snapshot |
| `GET` | `/cache/stats` | route-cache stats |
| `DELETE` | `/cache` | clear route cache |
| `POST` | `/oracle/quality/check` | record/check external-source quality |
| `GET` | `/oracle/quality/dashboard` | JSON oracle-quality dashboard |
| `GET` | /oracle/quality/dashboard.csv | CSV oracle-quality dashboard |
| `POST` | `/pareto` | compute Pareto routes as JSON |
| `POST` | `/pareto/stream` | stream Pareto events |
| `POST` | `/api/pareto/stream` | alternate streaming path |
| `POST` | `/route` | compute selected route and route set |
| `POST` | `/route/baseline` | OSRM quick baseline |
| `POST` | `/route/baseline/ors` | ORS reference or ORS-proxy baseline |
| `POST` | `/departure/optimize` | search best departure slot |
| `POST` | `/duty/chain` | multi-leg duty-chain composition |
| `POST` | `/scenario/compare` | compare route outputs across scenario modes |
| `GET` | `/experiments` | list experiment bundles |
| `POST` | `/experiments` | create experiment bundle |
| `GET` | `/experiments/{experiment_id}` | read one experiment bundle |
| `PUT` | `/experiments/{experiment_id}` | update one experiment bundle |
| `DELETE` | `/experiments/{experiment_id}` | delete experiment bundle |
| `POST` | `/experiments/{experiment_id}/compare` | run saved experiment comparison |
| `POST` | `/batch/import/csv` | parse/import batch OD CSV |
| `POST` | `/batch/pareto` | run batch Pareto over OD pairs |
| `GET` | `/runs/{run_id}/manifest` | signed run manifest |
| `GET` | `/runs/{run_id}/scenario-manifest` | signed scenario manifest |
| `GET` | `/runs/{run_id}/provenance` | provenance event log |
| `GET` | `/runs/{run_id}/signature` | run signature metadata |
| `GET` | `/runs/{run_id}/scenario-signature` | scenario signature metadata |
| `POST` | `/verify/signature` | verify payload signature |
| `GET` | `/runs/{run_id}/artifacts` | artifact listing |
| `GET` | `/runs/{run_id}/artifacts/results.json` | JSON batch results |
| `GET` | `/runs/{run_id}/artifacts/results.csv` | CSV batch results |
| `GET` | `/runs/{run_id}/artifacts/metadata.json` | run metadata |
| `GET` | `/runs/{run_id}/artifacts/routes.geojson` | route geometries |
| `GET` | `/runs/{run_id}/artifacts/results_summary.csv` | summarized metrics |
| `GET` | `/runs/{run_id}/artifacts/report.pdf` | generated PDF report |

## Appendix B: Frontend Proxy Endpoint Inventory

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | proxy backend health |
| `GET` | `/api/health/ready` | proxy backend readiness |
| `GET` | `/api/vehicles` | proxy built-in vehicles |
| `GET` | `/api/vehicles/custom` | proxy custom vehicles list |
| `POST` | `/api/vehicles/custom` | proxy custom vehicle creation |
| `PUT` | `/api/vehicles/custom/[vehicleId]` | proxy custom vehicle update |
| `DELETE` | `/api/vehicles/custom/[vehicleId]` | proxy custom vehicle delete |
| `POST` | `/api/route` | proxy route compute |
| `POST` | `/api/route/baseline` | proxy OSRM baseline |
| `POST` | `/api/route/baseline/ors` | proxy ORS/proxy baseline |
| `POST` | `/api/pareto` | proxy JSON Pareto compute |
| `POST` | `/api/pareto/stream` | proxy NDJSON Pareto stream |
| `POST` | `/api/scenario/compare` | proxy scenario compare |
| `POST` | `/api/departure/optimize` | proxy departure optimizer |
| `POST` | `/api/duty/chain` | proxy duty-chain compute |
| `GET` | `/api/experiments` | proxy experiment list |
| `POST` | `/api/experiments` | proxy experiment create |
| `GET` | `/api/experiments/[experimentId]` | proxy experiment read |
| `PUT` | `/api/experiments/[experimentId]` | proxy experiment update |
| `DELETE` | `/api/experiments/[experimentId]` | proxy experiment delete |
| `POST` | `/api/experiments/[experimentId]/compare` | proxy saved experiment compare |
| `POST` | `/api/batch/import/csv` | proxy CSV import |
| `POST` | `/api/batch/pareto` | proxy batch Pareto |
| `GET` | `/api/cache/stats` | proxy cache stats |
| `DELETE` | `/api/cache` | proxy cache clear |
| `GET` | `/api/metrics` | proxy metrics snapshot |
| `POST` | `/api/oracle/quality/check` | proxy oracle quality check |
| `GET` | `/api/oracle/quality/dashboard` | proxy oracle-quality JSON |
| `GET` | /api/oracle/quality/dashboard.csv | proxy oracle-quality CSV |
| `GET` | `/api/debug/live-calls/[requestId]` | proxy request trace |
| `GET` | `/api/runs/[runId]/[...subpath]` | proxy manifests, provenance, and artifacts |
| `POST` | `/api/verify/signature` | proxy signature verification |

## Appendix C: Frontend Control To Backend Field Mapping

| Frontend control | Backend field or endpoint | Plain-English meaning |
| --- | --- | --- |
| origin pin | `origin` | where the trip starts |
| destination pin | `destination` | where the trip ends |
| stop pins | `waypoints` or `stops` | intermediate via points or duty-chain stops |
| vehicle selector | `vehicle_type` | choose which freight vehicle profile to price and simulate |
| scenario mode | `scenario_mode` | choose assumed collaboration / pressure regime |
| time weight | `weights.time` | how strongly ETA should matter |
| money weight | `weights.money` | how strongly monetary cost should matter |
| CO2 weight | `weights.co2` | how strongly emissions should matter |
| max alternatives | `max_alternatives` | how many route options to retain |
| Pareto method | `pareto_method` | dominance or epsilon-based filtering |
| epsilon fields | `epsilon.duration_s`, `epsilon.monetary_cost`, `epsilon.emissions_kg` | hard-ish ceiling values for acceptable routes |
| departure time | `departure_time_utc` | when the trip is assumed to start |
| cost toggles | `cost_toggles` | turn tolls on/off and adjust carbon/fuel cost assumptions |
| terrain profile | `terrain_profile` | assume flatter, rolling, or hillier response floor |
| stochastic enabled | `stochastic.enabled` | whether to sample uncertainty |
| stochastic seed | `stochastic.seed` | deterministic replay seed |
| stochastic sigma | `stochastic.sigma` | noise magnitude |
| stochastic samples | `stochastic.samples` | Monte Carlo sample count |
| optimization mode | `optimization_mode` | average-value versus robust decision mode |
| risk aversion | `risk_aversion` | how strongly bad-tail outcomes matter |
| fuel type | `emissions_context.fuel_type` | diesel, petrol, LNG, or EV context |
| Euro class | `emissions_context.euro_class` | emissions standards context |
| ambient temperature | `emissions_context.ambient_temp_c` | temperature-sensitive energy/fuel adjustment |
| weather enabled/profile/intensity | `weather.*` | weather scenario fed into speed and incident modifiers |
| incident simulation settings | `incident_simulation.*` | synthetic incident experiment parameters |
| route button | `POST /route` | compute selected route |
| Pareto button | `POST /pareto` or `/pareto/stream` | compute the trade-off set |
| baseline button | `POST /route/baseline` | compute plain OSRM-style comparison route |
| ORS baseline button | `POST /route/baseline/ors` | compute ORS or proxy reference route |
| scenario compare | `POST /scenario/compare` | compare outputs across scenario modes |
| departure optimize | `POST /departure/optimize` | search time window for a better departure |
| duty chain | `POST /duty/chain` | chain multiple legs into one duty |
| experiments UI | `/experiments*` endpoints | store and replay scenario comparison setups |

## Appendix D: Request And Response Models

### Core backend models

| Model | Role |
| --- | --- |
| `LatLng` | latitude/longitude pair |
| `Waypoint` | optional via point with label |
| `Weights` | time/money/CO2 preference vector |
| `CostToggles` | toll, fuel, and carbon pricing switches |
| `EpsilonConstraints` | epsilon thresholds for Pareto filtering |
| `EmissionsContext` | fuel type, Euro class, ambient temperature |
| `WeatherImpactConfig` | weather profile and intensity |
| `IncidentSimulatorConfig` | synthetic incident rates and delays |
| `SimulatedIncidentEvent` | one generated or carried incident event |
| `TimeWindowConstraints` | departure optimization constraints |
| `StochasticConfig` | uncertainty enable/seed/sigma/sample count |
| `GeoJSONLineString` | route geometry contract |
| `RouteRequest` | single-route compute request |
| `ParetoRequest` | Pareto compute request |
| `ODPair` | one origin-destination pair for batch jobs |
| `BatchParetoRequest` | multi-pair route analysis request |
| `BatchCSVImportRequest` | CSV import helper request |
| `RouteMetrics` | distance, duration, cost, emissions, speed |
| `TerrainSummaryPayload` | terrain evidence summary embedded in a route |
| `ScenarioSummary` | applied scenario context and multipliers |
| `RouteOption` | a fully modeled route alternative |
| `RouteResponse` | selected route plus alternative set and metadata |
| `RouteBaselineResponse` | baseline route plus baseline method metadata |
| `ParetoResponse` | Pareto route set and diagnostics |
| `ScenarioCompareRequest/Response` | compare results across scenario modes |
| `DepartureOptimizeRequest/Response` | departure-window search contract |
| `DutyChainRequest/Response` | multi-stop duty-chain contract |
| `OracleFeedCheckInput/Record` | quality-check submission and result |
| `OracleQualityDashboardResponse` | source-quality dashboard |
| `ExperimentBundle*` | saved experiment inputs and catalog |
| `SignatureVerification*` | signature verification contract |

### Frontend mirrored type surface

`frontend/app/lib/types.ts` mirrors the backend and adds frontend-specific helper types such as:

- `ComputeMode`
- `PinNodeKind`
- `MapFailureOverlay`
- `ParetoStream*` event types
- `LiveCallTraceResponse`
- `RunManifestSummary`
- `RunArtifactsListResponse`
- `StrictErrorDetail`

This mirror is important because it keeps the frontend honest about what the backend actually returns.

## Appendix E: Operational Modes And Profiles

### Scenario modes

| Mode | Meaning |
| --- | --- |
| `no_sharing` | assume no coordination benefit; keep route pressure near observed baseline |
| `partial_sharing` | assume partial coordination benefit; reduce pressure multipliers partway |
| `full_sharing` | assume near-ideal coordination benefit; push multipliers toward floor values |

### Optimization modes

| Mode | Meaning |
| --- | --- |
| `expected_value` | rank routes by average/central outcomes |
| `robust` | penalize routes with worse tail-risk behavior |

### Pareto methods

| Method | Meaning |
| --- | --- |
| `dominance` | keep non-dominated routes |
| `epsilon_constraint` | keep routes that satisfy epsilon thresholds, then rank/filter |

### Selection math profiles

| Profile | Meaning |
| --- | --- |
| `academic_reference` | plain weighted-sum reference |
| `academic_tchebycheff` | augmented Tchebycheff scalarisation |
| `academic_vikor` | VIKOR compromise ranking |
| `modified_hybrid` | weighted-sum plus regret and balance |
| `modified_distance_aware` | weighted-sum blend plus distance/knee/entropy terms |
| `modified_vikor_distance` | VIKOR blend plus distance/knee/entropy terms; default in local config |

### Terrain profiles

| Profile | Meaning |
| --- | --- |
| `flat` | assume flat-route response floor |
| `rolling` | moderate terrain uplift |
| `hilly` | stronger terrain uplift |

## Appendix F: Environment Variable Families

The `.env.example` file is large. The most useful way to document it is by operational family rather than by raw alphabetical order.

### Core runtime and retry controls

| Variables | Effect |
| --- | --- |
| `LIVE_RUNTIME_DATA_ENABLED`, `STRICT_LIVE_DATA_REQUIRED` | turn strict live mode on/off |
| `LIVE_ROUTE_COMPUTE_REFRESH_MODE` | choose cache refresh strategy for route compute |
| `LIVE_ROUTE_COMPUTE_REQUIRE_ALL_EXPECTED` | require all expected live calls in strict route compute |
| `LIVE_ROUTE_COMPUTE_FORCE_NO_CACHE_HEADERS`, `LIVE_ROUTE_COMPUTE_FORCE_UNCACHED` | force uncached live fetch behavior |
| `LIVE_ROUTE_COMPUTE_PREFETCH_TIMEOUT_MS`, `LIVE_ROUTE_COMPUTE_PREFETCH_MAX_CONCURRENCY` | cap live prefetch latency and parallelism |
| `LIVE_DATA_CACHE_TTL_S`, `LIVE_SCENARIO_CACHE_TTL_SECONDS`, `ROUTE_CACHE_TTL_S` | cache TTLs |
| `LIVE_HTTP_MAX_ATTEMPTS`, `LIVE_HTTP_RETRY_DEADLINE_MS`, `LIVE_HTTP_RETRY_BACKOFF_BASE_MS`, `LIVE_HTTP_RETRY_BACKOFF_MAX_MS`, `LIVE_HTTP_RETRY_JITTER_MS`, `LIVE_HTTP_RETRY_RESPECT_RETRY_AFTER`, `LIVE_HTTP_RETRYABLE_STATUS_CODES` | bounded retry policy |

### Route graph, warmup, and search budgets

| Variables | Effect |
| --- | --- |
| `ROUTE_GRAPH_WARMUP_ON_STARTUP`, `ROUTE_GRAPH_WARMUP_FAILFAST`, `ROUTE_GRAPH_WARMUP_TIMEOUT_S` | graph warmup behavior |
| `ROUTE_GRAPH_FAST_STARTUP_ENABLED`, `ROUTE_GRAPH_FAST_STARTUP_LONG_CORRIDOR_BYPASS_KM` | startup shortcuts for large graphs/long corridors |
| `ROUTE_GRAPH_STATUS_CHECK_TIMEOUT_MS`, `ROUTE_GRAPH_OD_FEASIBILITY_TIMEOUT_MS`, `ROUTE_GRAPH_PRECHECK_TIMEOUT_FAIL_CLOSED` | readiness and OD precheck policy |
| `ROUTE_GRAPH_BINARY_CACHE_ENABLED` | use binary graph cache |
| `ROUTE_GRAPH_MAX_STATE_BUDGET`, `ROUTE_GRAPH_STATE_BUDGET_PER_HOP`, `ROUTE_GRAPH_STATE_BUDGET_RETRY_MULTIPLIER`, `ROUTE_GRAPH_STATE_BUDGET_RETRY_CAP` | search state-space budget |
| `ROUTE_GRAPH_SEARCH_INITIAL_TIMEOUT_MS`, `ROUTE_GRAPH_SEARCH_RETRY_TIMEOUT_MS`, `ROUTE_GRAPH_SEARCH_RESCUE_TIMEOUT_MS` | search deadlines |
| `ROUTE_GRAPH_STATE_SPACE_RESCUE_ENABLED`, `ROUTE_GRAPH_STATE_SPACE_RESCUE_MODE` | rescue mode |
| `ROUTE_GRAPH_LONG_CORRIDOR_THRESHOLD_KM`, `ROUTE_GRAPH_LONG_CORRIDOR_MAX_PATHS`, `ROUTE_GRAPH_SKIP_INITIAL_SEARCH_LONG_CORRIDOR` | long-corridor strategy |
| `ROUTE_GRAPH_MIN_GIANT_COMPONENT_NODES`, `ROUTE_GRAPH_MIN_GIANT_COMPONENT_RATIO`, `ROUTE_GRAPH_MAX_NEAREST_NODE_DISTANCE_M` | graph coverage quality gates |
| `ROUTE_GRAPH_OD_CANDIDATE_LIMIT`, `ROUTE_GRAPH_OD_CANDIDATE_MAX_RADIUS`, `ROUTE_GRAPH_MAX_HOPS`, `ROUTE_GRAPH_ADAPTIVE_HOPS_ENABLED`, `ROUTE_GRAPH_HOPS_PER_KM`, `ROUTE_GRAPH_HOPS_DETOUR_FACTOR`, `ROUTE_GRAPH_EDGE_LENGTH_ESTIMATE_M`, `ROUTE_GRAPH_HOPS_SAFETY_FACTOR`, `ROUTE_GRAPH_MAX_HOPS_CAP` | path-search breadth control |
| `ROUTE_GRAPH_A_STAR_HEURISTIC_ENABLED`, `ROUTE_GRAPH_HEURISTIC_MAX_SPEED_KPH` | A* heuristic use |

### Route option and baseline controls

| Variables | Effect |
| --- | --- |
| `ROUTE_CANDIDATE_PREFILTER_MULTIPLIER`, `ROUTE_CANDIDATE_PREFILTER_MULTIPLIER_LONG`, `ROUTE_CANDIDATE_PREFILTER_LONG_DISTANCE_THRESHOLD_KM` | candidate diversification compression |
| `ROUTE_OPTION_SEGMENT_CAP`, `ROUTE_OPTION_SEGMENT_CAP_LONG`, `ROUTE_OPTION_LONG_DISTANCE_THRESHOLD_KM` | segment-breakdown truncation |
| `ROUTE_OPTION_REUSE_SCENARIO_POLICY`, `ROUTE_OPTION_TOD_BUCKET_S`, `ROUTE_OPTION_ENERGY_SPEED_BIN_KPH`, `ROUTE_OPTION_ENERGY_GRADE_BIN_PCT` | route-option modeling details |
| `ROUTE_BASELINE_DURATION_MULTIPLIER`, `ROUTE_BASELINE_DISTANCE_MULTIPLIER` | OSRM baseline realism adjustments |
| `ROUTE_ORS_BASELINE_DURATION_MULTIPLIER`, `ROUTE_ORS_BASELINE_DISTANCE_MULTIPLIER`, `ROUTE_ORS_BASELINE_ALLOW_PROXY_FALLBACK` | ORS baseline realism and fallback policy |
| `ORS_DIRECTIONS_API_KEY`, `ORS_DIRECTIONS_URL_TEMPLATE`, `ORS_DIRECTIONS_TIMEOUT_MS`, `ORS_DIRECTIONS_PROFILE_DEFAULT`, `ORS_DIRECTIONS_PROFILE_HGV` | ORS provider configuration |
| `ROUTE_PARETO_BACKFILL_ENABLED`, `ROUTE_PARETO_BACKFILL_MIN_ALTERNATIVES` | backfill beyond strict frontier when needed |

### Selection math controls

| Variables | Effect |
| --- | --- |
| `ROUTE_SELECTION_MATH_PROFILE` | choose academic or modified scalar ranking profile |
| `ROUTE_SELECTION_MODIFIED_REGRET_WEIGHT` | regret term weight |
| `ROUTE_SELECTION_MODIFIED_BALANCE_WEIGHT` | balance term weight |
| `ROUTE_SELECTION_MODIFIED_DISTANCE_WEIGHT` | distance penalty weight |
| `ROUTE_SELECTION_MODIFIED_ETA_DISTANCE_WEIGHT` | ETA-distance interaction weight |
| `ROUTE_SELECTION_MODIFIED_ENTROPY_WEIGHT` | entropy reward weight |
| `ROUTE_SELECTION_MODIFIED_KNEE_WEIGHT` | knee penalty weight |
| `ROUTE_SELECTION_TCHEBYCHEFF_RHO` | augmented Tchebycheff epsilon coefficient |
| `ROUTE_SELECTION_VIKOR_V` | VIKOR compromise factor |

### Compute timeout and frontend degrade controls

| Variables | Effect |
| --- | --- |
| `ROUTE_COMPUTE_ATTEMPT_TIMEOUT_S`, `ROUTE_COMPUTE_SINGLE_ATTEMPT_TIMEOUT_S` | backend attempt budget |
| `COMPUTE_ATTEMPT_TIMEOUT_MS`, `COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS` | frontend/backend transport timeouts |
| `NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS`, `NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS`, `NEXT_PUBLIC_COMPUTE_DEGRADE_STEPS`, `NEXT_PUBLIC_ROUTE_GRAPH_WARMUP_BASELINE_MS` | frontend timeout and degrade UX |

### Terrain controls

| Variables | Effect |
| --- | --- |
| `TERRAIN_DEM_FAIL_CLOSED_UK`, `TERRAIN_DEM_COVERAGE_MIN_UK` | fail-closed terrain policy |
| `TERRAIN_SAMPLE_SPACING_M`, `TERRAIN_LONG_ROUTE_THRESHOLD_KM`, `TERRAIN_LONG_ROUTE_SAMPLE_SPACING_M`, `TERRAIN_LONG_ROUTE_MAX_SAMPLES_PER_ROUTE`, `TERRAIN_MAX_SAMPLES_PER_ROUTE`, `TERRAIN_SEGMENT_BOUNDARY_PROBE_MAX_SEGMENTS` | terrain sampling budgets |
| `LIVE_TERRAIN_DEM_URL_TEMPLATE`, `LIVE_TERRAIN_REQUIRE_URL_IN_STRICT`, `LIVE_TERRAIN_ALLOW_SIGNED_FALLBACK`, `LIVE_TERRAIN_ALLOWED_HOSTS`, `LIVE_TERRAIN_TILE_ZOOM`, `LIVE_TERRAIN_TILE_MAX_AGE_DAYS`, `LIVE_TERRAIN_CACHE_DIR`, `LIVE_TERRAIN_CACHE_MAX_TILES`, `LIVE_TERRAIN_CACHE_MAX_MB`, `LIVE_TERRAIN_FETCH_RETRIES`, `LIVE_TERRAIN_MAX_REMOTE_TILES_PER_ROUTE`, `LIVE_TERRAIN_CIRCUIT_BREAKER_FAILURES`, `LIVE_TERRAIN_CIRCUIT_BREAKER_COOLDOWN_S`, `LIVE_TERRAIN_ENABLE_IN_TESTS`, `LIVE_TERRAIN_PREFETCH_PROBE_FRACTIONS`, `LIVE_TERRAIN_PREFETCH_MIN_COVERED_POINTS` | live terrain tile behavior |

### Scenario, fuel, carbon, departure, stochastic, and toll source controls

| Variables | Effect |
| --- | --- |
| `LIVE_SCENARIO_COEFFICIENT_URL`, `LIVE_SCENARIO_REQUIRE_URL_IN_STRICT`, `LIVE_SCENARIO_ALLOW_SIGNED_FALLBACK`, `LIVE_SCENARIO_ALLOWED_HOSTS` | strict scenario source control |
| `LIVE_SCENARIO_WEBTRIS_NEAREST_SITES`, `LIVE_SCENARIO_DFT_MAX_PAGES`, `LIVE_SCENARIO_DFT_NEAREST_LIMIT`, `LIVE_SCENARIO_DFT_MIN_STATION_COUNT`, `LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES`, `LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT`, `LIVE_SCENARIO_MIN_SOURCE_COUNT_STRICT`, `LIVE_SCENARIO_MIN_COVERAGE_OVERALL_STRICT` | scenario coverage and freshness rules |
| `SCENARIO_MIN_OBSERVED_MODE_ROW_SHARE`, `SCENARIO_MAX_PROJECTION_DOMINANT_CONTEXT_SHARE`, `SCENARIO_REQUIRE_SIGNATURE` | scenario calibration acceptance rules |
| `LIVE_FUEL_PRICE_URL`, `LIVE_FUEL_AUTH_TOKEN`, `LIVE_FUEL_API_KEY`, `LIVE_FUEL_API_KEY_HEADER`, `LIVE_FUEL_REQUIRE_URL_IN_STRICT`, `LIVE_FUEL_ALLOW_SIGNED_FALLBACK`, `LIVE_FUEL_REQUIRE_SIGNATURE`, `LIVE_FUEL_ALLOWED_HOSTS`, `LIVE_FUEL_MAX_AGE_DAYS` | fuel snapshot source policy |
| `LIVE_CARBON_SCHEDULE_URL`, `LIVE_CARBON_REQUIRE_URL_IN_STRICT`, `LIVE_CARBON_ALLOW_SIGNED_FALLBACK`, `LIVE_CARBON_ALLOWED_HOSTS` | carbon schedule policy |
| `LIVE_BANK_HOLIDAYS_URL`, `LIVE_BANK_HOLIDAYS_ALLOWED_HOSTS` | UK holiday source policy |
| `LIVE_DEPARTURE_PROFILE_URL`, `LIVE_DEPARTURE_REQUIRE_URL_IN_STRICT`, `LIVE_DEPARTURE_ALLOW_SIGNED_FALLBACK`, `LIVE_DEPARTURE_ALLOWED_HOSTS` | departure profile source policy |
| `LIVE_STOCHASTIC_REGIMES_URL`, `LIVE_STOCHASTIC_REQUIRE_URL_IN_STRICT`, `LIVE_STOCHASTIC_ALLOW_SIGNED_FALLBACK`, `LIVE_STOCHASTIC_ALLOWED_HOSTS` | stochastic-regime source policy |
| `LIVE_TOLL_TOPOLOGY_URL`, `LIVE_TOLL_TOPOLOGY_REQUIRE_URL_IN_STRICT`, `LIVE_TOLL_TOPOLOGY_ALLOW_SIGNED_FALLBACK`, `LIVE_TOLL_TARIFFS_URL`, `LIVE_TOLL_TARIFFS_REQUIRE_URL_IN_STRICT`, `LIVE_TOLL_TARIFFS_ALLOW_SIGNED_FALLBACK`, `LIVE_TOLL_ALLOWED_HOSTS` | toll source policy |

### Dev/debug controls

| Variables | Effect |
| --- | --- |
| `DEV_ROUTE_DEBUG_CONSOLE_ENABLED`, `DEV_ROUTE_DEBUG_INCLUDE_SENSITIVE`, `DEV_ROUTE_DEBUG_MAX_CALLS_PER_REQUEST`, `DEV_ROUTE_DEBUG_TRACE_TTL_SECONDS`, `DEV_ROUTE_DEBUG_MAX_REQUEST_TRACES`, `DEV_ROUTE_DEBUG_RETURN_RAW_PAYLOADS`, `DEV_ROUTE_DEBUG_MAX_RAW_BODY_CHARS` | request-trace dev console behavior |
| `STRICT_RUNTIME_TEST_BYPASS` | deterministic test bypass switch |

## Appendix G: Strict Reason Codes

| Reason code | Meaning |
| --- | --- |
| `routing_graph_unavailable` | graph asset missing or unreadable |
| `routing_graph_fragmented` | graph giant-component quality gate failed |
| `routing_graph_disconnected_od` | origin and destination are disconnected in graph space |
| `routing_graph_coverage_gap` | nearest-node or coverage gap too large |
| `routing_graph_no_path` | graph search produced no usable path |
| `routing_graph_precheck_timeout` | OD feasibility precheck timed out |
| `routing_graph_deferred_load` | graph load postponed |
| `routing_graph_warming_up` | graph still loading |
| `routing_graph_warmup_failed` | graph warmup failed |
| `live_source_refresh_failed` | strict live refresh stage failed |
| `route_compute_timeout` | route compute exceeded allowed time |
| `departure_profile_unavailable` | departure profile missing/stale/invalid under strict policy |
| `holiday_data_unavailable` | holiday data missing/stale |
| `stochastic_calibration_unavailable` | stochastic regime asset unavailable |
| `scenario_profile_unavailable` | scenario profile feed/asset unavailable |
| `scenario_profile_invalid` | scenario profile failed validation |
| `risk_normalization_unavailable` | normalization references unavailable |
| `risk_prior_unavailable` | risk prior unavailable |
| `terrain_region_unsupported` | route is outside supported UK terrain region in strict mode |
| `terrain_dem_asset_unavailable` | terrain reader or assets unavailable |
| `terrain_dem_coverage_insufficient` | terrain coverage ratio below threshold |
| `toll_topology_unavailable` | toll topology missing |
| `toll_tariff_unavailable` | toll tariff source missing |
| `toll_tariff_unresolved` | route hit toll logic but tariff resolution failed |
| `fuel_price_auth_unavailable` | fuel source authentication failed |
| `fuel_price_source_unavailable` | fuel source missing/stale/invalid |
| `vehicle_profile_unavailable` | vehicle asset missing |
| `vehicle_profile_invalid` | vehicle profile failed schema or signature checks |
| `carbon_policy_unavailable` | carbon schedule missing/stale/invalid |
| `carbon_intensity_unavailable` | carbon intensity unavailable |
| `epsilon_infeasible` | epsilon constraints eliminate all options |
| `no_route_candidates` | no candidate routes survived |
| `baseline_route_unavailable` | baseline route provider failed |
| `baseline_provider_unconfigured` | ORS or other provider not configured |
| `model_asset_unavailable` | generic asset unavailability catch-all |

## Appendix H: Artifact Formats And Provenance

| Artifact | Produced by | Role |
| --- | --- | --- |
| run manifest | `run_store.write_manifest` | signed high-level run descriptor |
| scenario manifest | `run_store.write_scenario_manifest` | signed scenario-specific descriptor |
| provenance log | `provenance_store.write_provenance` | event timeline for how a run happened |
| `backend/out/artifacts/<run_id>/results.json` | batch/run artifact writer | full structured results |
| `backend/out/artifacts/<run_id>/results.csv` | batch/run artifact writer | tabular route metrics |
| `backend/out/artifacts/<run_id>/metadata.json` | batch/run artifact writer | run metadata |
| `backend/out/artifacts/<run_id>/routes.geojson` | batch/run artifact writer | geometry export |
| `backend/out/artifacts/<run_id>/results_summary.csv` | batch/reporting flow | summarized results |
| `backend/out/artifacts/<run_id>/report.pdf` | `backend/app/reporting.py` | human-readable report output |
| oracle checks NDJSON | `oracle_quality_store.append_check_record` | per-source quality record stream |
| oracle summary JSON/CSV | `oracle_quality_store.write_summary_artifacts` | dashboard outputs |
| model manifest | `backend/out/model_assets/manifest.json` | built asset list and checksums |
| preflight summary | `backend/out/model_assets/preflight_live_runtime.json` | strict readiness evidence |
| live publish summary | `backend/out/model_assets/live_publish_summary.json` | publish-to-runtime handoff evidence |

### Signature format

The repository uses:

- canonical JSON serialization
- HMAC-SHA256
- algorithm label `HMAC-SHA256`

This is used for manifest-style verification, not for general user authentication.

## Appendix I: Vehicle And Custom-Profile Schema

### Built-in profile shape

The built-in and custom profile schema in `backend/app/vehicles.py` includes:

- `id`
- `label`
- `mass_tonnes`
- `emission_factor_kg_per_tkm`
- `cost_per_km`
- `cost_per_hour`
- `idle_emissions_kg_per_hour`
- `powertrain`
- `ev_kwh_per_km`
- `grid_co2_kg_per_kwh`
- `schema_version`
- `vehicle_class`
- `toll_vehicle_class`
- `toll_axle_class`
- `fuel_surface_class`
- `risk_bucket`
- `stochastic_bucket`
- `terrain_params`
- `aliases`
- `profile_source`
- `profile_as_of_utc`

### Terrain parameter sub-schema

Each profile carries terrain parameters:

- `mass_kg`
- `c_rr`
- `drag_area_m2`
- `drivetrain_efficiency`
- `regen_efficiency`

This is what allows the physics/terrain subsystem to differ across vans, rigid trucks, artics, and EV freight vehicles.

## Appendix J: File Inventory

Importance legend used below:

- `core`: essential to understanding or running the system; not background.
- `supporting`: important supporting implementation or operations; can be treated as secondary background after core files.
- `reference`: explanatory or setup material; safe as background.
- `test-only`: behavior lock-in rather than runtime logic.
- `generated/runtime`: runtime evidence or outputs; useful for evaluation, not source logic.

### Root, repo, and top-level config files

| File | Importance | Purpose |
| --- | --- | --- |
| `.env.example` | `core` | master runtime configuration surface; the most important non-code file for strict/live behavior |
| `.gitattributes` | `reference` | line-ending policy for shell scripts, Docker files, YAML, and PowerShell |
| `.gitignore` | `reference` | declares caches, OSRM data, and runtime outputs as non-tracked noise |
| `../.github/workflows/backend-ci.yml` | `supporting` | defines fast and strict-live CI lanes |
| `../.vscode/settings.json` | `reference` | local editor performance and interpreter settings |
| `../README.md` | `supporting` | top-level architecture, startup flow, and operations guidance |
| `docker-compose.yml` | `core` | service graph for OSRM, backend, and frontend |
| `project-dump.txt` | `reference` | stale/high-level directory snapshot; useful background, not authoritative over current code |
| `data/.gitkeep` | `reference` | keeps placeholder data directory in Git |

### Documentation files

| File | Importance | Purpose |
| --- | --- | --- |
| `docs/.gitkeep` | `reference` | keeps docs directory present in Git |
| `docs/DOCS_INDEX.md` | `supporting` | entry point into the docs set |
| `docs/api-cookbook.md` | `supporting` | example API usage patterns |
| `docs/appendix-graph-theory-notes.md` | `reference` | graph-theory background notes for the route graph |
| `docs/backend-api-tools.md` | `core` | backend endpoint and tooling documentation |
| `docs/co2e-validation.md` | `supporting` | CO2e validation notes |
| `docs/dissertation-math-overview.md` | `supporting` | thesis-oriented math summary |
| `docs/eta-concept-drift.md` | `supporting` | ETA drift monitoring concepts |
| `docs/frontend-accessibility-i18n.md` | `supporting` | accessibility and i18n behavior |
| `docs/frontend-dev-tools.md` | `supporting` | frontend devtools and diagnostics panels |
| `docs/map-overlays-tooltips.md` | `supporting` | UI map overlay semantics |
| `docs/math-appendix.md` | `supporting` | mathematical appendix beyond high-level overview |
| `docs/model-assets-and-data-sources.md` | `core` | asset families, provenance, and source descriptions |
| `docs/performance-profiling-notes.md` | `supporting` | performance measurement notes |
| `docs/quality-gates-and-benchmarks.md` | `core` | explicit quality thresholds and benchmark expectations |
| `docs/reproducibility-capsule.md` | `supporting` | reproducibility capsule workflow |
| `docs/run-and-operations.md` | `supporting` | runtime operations guide |
| `docs/sample-manifest.md` | `supporting` | manifest example and explanation |
| `docs/strict-errors-reference.md` | `core` | strict reason-code reference |
| `docs/synthetic-incidents-weather.md` | `supporting` | incident simulation and weather overlay behavior |
| `docs/tutorial-and-reporting.md` | `supporting` | tutorial UX and reporting outputs |
| `docs/examples/sample_batch_request.json` | `reference` | batch request example payload |
| `docs/examples/sample_batch_response.json` | `reference` | batch response example payload |
| `docs/examples/sample_manifest.json` | `reference` | manifest example payload |

### Backend runtime modules (`backend/app`)

| File | Importance | Purpose |
| --- | --- | --- |
| `backend/app/__init__.py` | `reference` | package marker |
| `backend/app/calibration_loader.py` | `core` | loads and validates scenario, departure, toll, fuel, stochastic, and holiday assets |
| `backend/app/carbon_model.py` | `core` | strict carbon schedule loading and pricing context |
| `backend/app/departure_profile.py` | `core` | contextual departure-time multiplier logic |
| `backend/app/experiment_store.py` | `supporting` | filesystem-backed CRUD for saved experiments |
| `backend/app/fuel_energy_model.py` | `core` | fuel/energy/emissions computation over calibrated surfaces |
| `backend/app/incident_simulator.py` | `supporting` | seeded synthetic incident generation for experiments |
| `backend/app/k_shortest.py` | `core` | Yen-style K-shortest path search with budgets |
| `backend/app/live_call_trace.py` | `core` | request-scoped live-call tracing and rollups |
| `backend/app/live_data_sources.py` | `core` | bounded retry, caching, host allow-lists, and live feed fetches |
| `backend/app/logging_utils.py` | `supporting` | structured JSON logging setup |
| `backend/app/main.py` | `core` | FastAPI app, endpoint surface, route pipeline, selection logic |
| `backend/app/metrics_store.py` | `supporting` | in-memory endpoint metrics store |
| `backend/app/model_data_errors.py` | `core` | frozen strict reason-code contract |
| `backend/app/models.py` | `core` | Pydantic API and payload contracts |
| `backend/app/multileg_engine.py` | `core` | multi-leg duty-chain aggregation logic |
| `backend/app/objectives_emissions.py` | `supporting` | simpler/demo emissions-speed objective helpers |
| `backend/app/objectives_selection.py` | `supporting` | simple weighted-sum selector helper |
| `backend/app/oracle_quality_store.py` | `supporting` | source-quality dashboard persistence |
| `backend/app/pareto.py` | `core` | Pareto dominance filter |
| `backend/app/pareto_methods.py` | `core` | epsilon filter, crowding truncation, knee annotation |
| `backend/app/provenance_store.py` | `supporting` | run provenance event persistence |
| `backend/app/rbac.py` | `reference` | disabled RBAC shim preserving endpoint signatures |
| `backend/app/reporting.py` | `supporting` | PDF report generation |
| `backend/app/risk_model.py` | `core` | quantiles, CVaR, normalized utility, robust objective math |
| `backend/app/route_cache.py` | `supporting` | in-memory route-result cache with stats |
| `backend/app/routing_graph.py` | `core` | route graph load, warmup, status, and candidate generation |
| `backend/app/routing_osrm.py` | `core` | OSRM client wrapper and retry logic |
| `backend/app/run_store.py` | `supporting` | manifests and artifact writer paths |
| `backend/app/scenario.py` | `core` | scenario modes, context keys, and scenario policy application |
| `backend/app/settings.py` | `core` | validated runtime settings surface |
| `backend/app/signatures.py` | `supporting` | canonical payload signing and verification |
| `backend/app/terrain_dem.py` | `core` | terrain sampling, coverage checks, and route summaries |
| `backend/app/terrain_dem_index.py` | `core` | terrain manifest/live-tile indexing and tile fetch diagnostics |
| `backend/app/terrain_physics.py` | `core` | grade, drag, rolling resistance, and terrain uplift formulas |
| `backend/app/toll_engine.py` | `core` | toll matching, tariff application, and confidence handling |
| `backend/app/uncertainty_model.py` | `core` | stochastic regime application and uncertainty summaries |
| `backend/app/vehicles.py` | `core` | built-in/custom vehicle schema and persistence |
| `backend/app/weather_adapter.py` | `supporting` | weather-to-speed and weather-to-incident transforms |

### Backend scripts (`backend/scripts`)

| File | Importance | Purpose |
| --- | --- | --- |
| `backend/scripts/__init__.py` | `reference` | package marker |
| `backend/scripts/benchmark_batch_pareto.py` | `supporting` | measures batch Pareto runtime and memory |
| `backend/scripts/benchmark_model_v2.py` | `supporting` | route-option benchmark and p95 gate checker |
| `backend/scripts/build_departure_profiles_uk.py` | `supporting` | builds contextual departure profiles |
| `backend/scripts/build_model_assets.py` | `core` | compiles full model asset set and manifest |
| `backend/scripts/build_pricing_tables_uk.py` | `supporting` | toll pricing table builder |
| `backend/scripts/build_routing_graph_uk.py` | `core` | builds routing graph asset |
| `backend/scripts/build_scenario_profiles_uk.py` | `core` | builds scenario policy profiles from evidence |
| `backend/scripts/build_stochastic_calibration_uk.py` | `core` | builds stochastic regimes and priors |
| `backend/scripts/build_terrain_tiles_uk.py` | `supporting` | terrain tile build pipeline |
| `backend/scripts/check_eta_concept_drift.py` | `supporting` | concept-drift check for ETA behavior |
| `backend/scripts/collect_carbon_intensity_raw_uk.py` | `supporting` | fetch raw carbon intensity inputs |
| `backend/scripts/collect_dft_raw_counts_uk.py` | `supporting` | fetch raw DfT counts |
| `backend/scripts/collect_fuel_history_raw_uk.py` | `supporting` | fetch raw fuel-history inputs |
| `backend/scripts/collect_scenario_mode_outcomes_proxy_uk.py` | `supporting` | collect proxy outcomes for scenario modes |
| `backend/scripts/collect_stochastic_residuals_raw_uk.py` | `supporting` | build raw residual corpus |
| `backend/scripts/collect_toll_truth_raw_uk.py` | `supporting` | collect raw toll truth corpus |
| `backend/scripts/extract_osm_tolls_uk.py` | `supporting` | extract toll assets from OSM |
| `backend/scripts/fetch_carbon_intensity_uk.py` | `supporting` | build asset-ready carbon intensity data |
| `backend/scripts/fetch_dft_counts_uk.py` | `supporting` | transform DfT raw counts into empirical departure asset |
| `backend/scripts/fetch_fuel_history_uk.py` | `supporting` | build asset-ready fuel history snapshot |
| `backend/scripts/fetch_public_dem_tiles_uk.py` | `supporting` | fetch public DEM tiles |
| `backend/scripts/fetch_scenario_live_uk.py` | `core` | gather live scenario evidence |
| `backend/scripts/fetch_stochastic_residuals_uk.py` | `supporting` | build empirical stochastic residual asset |
| `backend/scripts/fetch_toll_truth_uk.py` | `supporting` | build toll truth artifacts |
| `backend/scripts/generate_run_report.py` | `supporting` | generate run report artifacts |
| `backend/scripts/preflight_live_runtime.py` | `core` | strict live runtime preflight |
| `backend/scripts/publish_live_artifacts_uk.py` | `core` | promote compiled assets to live runtime assets and signatures |
| `backend/scripts/run_headless_scenario.py` | `supporting` | run scenario computations without UI |
| `backend/scripts/run_robustness_analysis.py` | `supporting` | multi-seed robustness study |
| `backend/scripts/run_sensitivity_analysis.py` | `supporting` | fuel/carbon/toll sensitivity study |
| `backend/scripts/score_model_quality.py` | `core` | compute subsystem quality scores and evidence checks |
| `backend/scripts/validate_graph_coverage.py` | `core` | graph coverage validation and reporting |

### Frontend shell, config, and app files

| File | Importance | Purpose |
| --- | --- | --- |
| `frontend/.dockerignore` | `reference` | excludes frontend build noise from Docker context |
| `frontend/Dockerfile` | `supporting` | container image for frontend runtime |
| `frontend/next-env.d.ts` | `reference` | Next.js TypeScript environment declarations |
| `frontend/next.config.mjs` | `supporting` | standalone output and build configuration |
| `frontend/package.json` | `supporting` | frontend package manifest and scripts |
| `frontend/pnpm-lock.yaml` | `reference` | exact frontend dependency lockfile |
| `frontend/postcss.config.mjs` | `reference` | PostCSS/Tailwind pipeline config |
| `frontend/public/.gitkeep` | `reference` | placeholder public directory |
| `frontend/tsconfig.json` | `reference` | TypeScript compiler settings |
| `frontend/app/global.css` | `supporting` | global visual style system |
| `frontend/app/layout.tsx` | `supporting` | root layout and fonts |
| `frontend/app/page.tsx` | `core` | main single-page orchestration surface |

### Frontend API proxy route handlers

| File | Importance | Purpose |
| --- | --- | --- |
| `frontend/app/api/batch/import/csv/route.ts` | `supporting` | proxy CSV import |
| `frontend/app/api/batch/pareto/route.ts` | `supporting` | proxy batch Pareto |
| `frontend/app/api/cache/route.ts` | `supporting` | proxy cache clear |
| `frontend/app/api/cache/stats/route.ts` | `supporting` | proxy cache stats |
| `frontend/app/api/debug/live-calls/[requestId]/route.ts` | `supporting` | proxy request trace |
| `frontend/app/api/departure/optimize/route.ts` | `supporting` | proxy departure optimizer |
| `frontend/app/api/duty/chain/route.ts` | `supporting` | proxy duty-chain compute |
| `frontend/app/api/experiments/route.ts` | `supporting` | proxy experiment list/create |
| `frontend/app/api/experiments/[experimentId]/route.ts` | `supporting` | proxy experiment read/update/delete |
| `frontend/app/api/experiments/[experimentId]/compare/route.ts` | `supporting` | proxy experiment compare |
| `frontend/app/api/health/route.ts` | `supporting` | proxy health |
| `frontend/app/api/health/ready/route.ts` | `supporting` | proxy readiness |
| `frontend/app/api/metrics/route.ts` | `supporting` | proxy metrics |
| `frontend/app/api/oracle/quality/check/route.ts` | `supporting` | proxy oracle check |
| `frontend/app/api/oracle/quality/dashboard/route.ts` | `supporting` | proxy oracle dashboard JSON |
| `frontend/app/api/oracle/quality/dashboard.csv/route.ts` | `supporting` | proxy oracle dashboard CSV |
| `frontend/app/api/pareto/route.ts` | `core` | proxy JSON Pareto |
| `frontend/app/api/pareto/stream/route.ts` | `core` | proxy NDJSON Pareto |
| `frontend/app/api/route/route.ts` | `core` | proxy selected-route compute |
| `frontend/app/api/route/baseline/route.ts` | `supporting` | local working-tree proxy for OSRM baseline |
| `frontend/app/api/route/baseline/ors/route.ts` | `supporting` | local working-tree proxy for ORS baseline |
| `frontend/app/api/runs/[runId]/[...subpath]/route.ts` | `supporting` | proxy artifacts/manifests/provenance |
| `frontend/app/api/scenario/compare/route.ts` | `supporting` | proxy scenario compare |
| `frontend/app/api/vehicles/route.ts` | `supporting` | proxy built-in vehicles |
| `frontend/app/api/vehicles/custom/route.ts` | `supporting` | proxy custom vehicle list/create |
| `frontend/app/api/vehicles/custom/[vehicleId]/route.ts` | `supporting` | proxy custom vehicle update/delete |
| `frontend/app/api/verify/signature/route.ts` | `supporting` | proxy signature verification |

### Frontend components

| File | Importance | Purpose |
| --- | --- | --- |
| `frontend/app/components/CollapsibleCard.tsx` | `supporting` | generic collapsible panel wrapper |
| `frontend/app/components/CounterfactualPanel.tsx` | `supporting` | renders what-if deltas |
| `frontend/app/components/DepartureOptimizerChart.tsx` | `supporting` | departure-window chart |
| `frontend/app/components/DutyChainPlanner.tsx` | `supporting` | duty-chain UI |
| `frontend/app/components/EtaTimelineChart.tsx` | `supporting` | ETA stage visualization |
| `frontend/app/components/ExperimentManager.tsx` | `supporting` | experiment CRUD UI |
| `frontend/app/components/FieldInfo.tsx` | `supporting` | field help text display |
| `frontend/app/components/MapView.tsx` | `core` | UK map and route overlay surface |
| `frontend/app/components/OracleQualityDashboard.tsx` | `supporting` | source-quality dashboard UI |
| `frontend/app/components/ParetoChart.tsx` | `core` | Pareto visualization |
| `frontend/app/components/PinManager.tsx` | `supporting` | origin/destination/stop pin management |
| `frontend/app/components/RouteBaselineComparison.tsx` | `supporting` | local working-tree baseline comparison panel |
| `frontend/app/components/ScenarioComparison.tsx` | `supporting` | compare scenario outputs |
| `frontend/app/components/ScenarioParameterEditor.tsx` | `core` | advanced routing control panel |
| `frontend/app/components/ScenarioTimeLapse.tsx` | `supporting` | time-lapse display |
| `frontend/app/components/SegmentBreakdown.tsx` | `supporting` | segment-level metrics panel |
| `frontend/app/components/Select.tsx` | `supporting` | shared select widget |
| `frontend/app/components/TutorialOverlay.tsx` | `supporting` | guided walkthrough overlay |
| `frontend/app/components/devtools/BatchRunner.tsx` | `supporting` | batch-run devtool |
| `frontend/app/components/devtools/CustomVehicleManager.tsx` | `supporting` | custom vehicle devtool |
| `frontend/app/components/devtools/OpsDiagnosticsPanel.tsx` | `supporting` | ops diagnostics UI |
| `frontend/app/components/devtools/RunInspector.tsx` | `supporting` | artifact/run inspector |
| `frontend/app/components/devtools/SignatureVerifier.tsx` | `supporting` | signature verification UI |

### Frontend libraries and helpers

| File | Importance | Purpose |
| --- | --- | --- |
| `frontend/app/lib/api.ts` | `supporting` | fetch wrappers including NDJSON reader |
| `frontend/app/lib/backendFetch.ts` | `supporting` | backend base URL, timeout, and transport error classification |
| `frontend/app/lib/baselineComparison.ts` | `supporting` | local working-tree baseline comparison math |
| `frontend/app/lib/format.ts` | `supporting` | display formatting helpers |
| `frontend/app/lib/i18n.ts` | `supporting` | locale and translation helpers |
| `frontend/app/lib/mapOverlays.ts` | `supporting` | map overlay generation for incidents, stops, and segments |
| `frontend/app/lib/requestBuilders.ts` | `core` | strongly typed request constructors |
| `frontend/app/lib/sidebarHelpText.ts` | `supporting` | non-jargon descriptions of controls |
| `frontend/app/lib/types.ts` | `core` | frontend mirror of backend contracts |
| `frontend/app/lib/weights.ts` | `core` | frontend mirror of scalar selection math |
| `frontend/app/lib/tutorial/prefills.ts` | `supporting` | tutorial prefilled scenarios |
| `frontend/app/lib/tutorial/progress.ts` | `supporting` | tutorial progress state |
| `frontend/app/lib/tutorial/steps.ts` | `supporting` | tutorial step definitions |
| `frontend/app/lib/tutorial/types.ts` | `supporting` | tutorial type helpers |

### OSRM, notebooks, and root operational scripts

| File | Importance | Purpose |
| --- | --- | --- |
| `osrm/data/osrm/.gitkeep` | `reference` | placeholder prepared OSRM data dir |
| `osrm/data/pbf/.gitkeep` | `reference` | placeholder raw PBF dir |
| `osrm/profiles/car.lua` | `supporting` | wrapper around OSRM car profile |
| `osrm/scripts/download_pbf.sh` | `supporting` | download/cached-region PBF bootstrap |
| `osrm/scripts/run_osrm.sh` | `supporting` | OSRM extract/partition/customize/serve wrapper |
| `notebooks/NOTEBOOKS_POLICY.md` | `reference` | notebook-free workflow policy |
| `scripts/check_docs.py` | `supporting` | markdown/link/path/endpoint consistency checker |
| `scripts/clean.ps1` | `supporting` | remove caches and reset generated folders |
| `scripts/collect_true_empirical_public_uk.ps1` | `supporting` | end-to-end empirical data collection and asset build pipeline |
| `scripts/demo_repro_run.ps1` | `supporting` | reproducibility capsule demo |
| `scripts/dev.ps1` | `core` | full local startup orchestration |
| `scripts/run_backend_tests_safe.ps1` | `supporting` | guarded backend test runner |
| `scripts/serve_docs.ps1` | `reference` | lightweight docs HTTP server |

### Backend tests (part 1)

| File | Importance | Purpose |
| --- | --- | --- |
| `backend/tests/test_api_streaming.py` | `test-only` | streaming API contract |
| `backend/tests/test_app_package_and_errors.py` | `test-only` | app packaging and error normalization |
| `backend/tests/test_app_smoke_all.py` | `test-only` | broad smoke coverage |
| `backend/tests/test_app_test_matrix.py` | `test-only` | app test matrix execution |
| `backend/tests/test_batch_flow_integration.py` | `test-only` | end-to-end batch flow |
| `backend/tests/test_batch_import_csv.py` | `test-only` | CSV batch import behavior |
| `backend/tests/test_cost_model.py` | `test-only` | cost model and strict-source mapping |
| `backend/tests/test_counterfactuals.py` | `test-only` | counterfactual output generation |
| `backend/tests/test_departure_optimize.py` | `test-only` | departure optimization endpoint |
| `backend/tests/test_departure_profile_v2.py` | `test-only` | contextual departure profile behavior |
| `backend/tests/test_dev_preflight.py` | `test-only` | dev/preflight startup checks |
| `backend/tests/test_duty_chain.py` | `test-only` | duty-chain composition |
| `backend/tests/test_emissions_context.py` | `test-only` | emissions context adjustments |
| `backend/tests/test_emissions_models.py` | `test-only` | emissions-model correctness |
| `backend/tests/test_experiments.py` | `test-only` | experiment CRUD and replay |
| `backend/tests/test_explainability_compare_integration.py` | `test-only` | explanation and comparison integration |
| `backend/tests/test_incident_simulator.py` | `test-only` | synthetic incident generator |
| `backend/tests/test_k_shortest.py` | `test-only` | K-shortest search behavior |
| `backend/tests/test_live_bank_holidays_strict.py` | `test-only` | strict bank-holiday feed behavior |
| `backend/tests/test_live_call_trace_rollup.py` | `test-only` | live-call trace aggregation |
| `backend/tests/test_live_retry_policy.py` | `test-only` | retry and bounded-backoff behavior |
| `backend/tests/test_metrics.py` | `test-only` | metrics snapshot behavior |
| `backend/tests/test_models_weight_aliases.py` | `test-only` | model aliases and weight validation |
| `backend/tests/test_multileg_engine.py` | `test-only` | multileg aggregation logic |
| `backend/tests/test_oracle_quality.py` | `test-only` | oracle quality storage and rollups |
| `backend/tests/test_pareto.py` | `test-only` | Pareto fundamentals |
| `backend/tests/test_pareto_backfill.py` | `test-only` | local working-tree Pareto backfill behavior |
| `backend/tests/test_pareto_epsilon_knee.py` | `test-only` | epsilon and knee annotations |
| `backend/tests/test_pareto_strict_frontier.py` | `test-only` | strict frontier behavior |
| `backend/tests/test_path_selection_logic.py` | `test-only` | scalar route selection logic |
| `backend/tests/test_property_invariants.py` | `test-only` | general invariants and regression properties |
| `backend/tests/test_publish_live_artifacts.py` | `test-only` | publish-live artifact flow |
| `backend/tests/test_rbac_logging_live_sources.py` | `test-only` | RBAC shim, logging, and live source hooks |
| `backend/tests/test_robust_mode.py` | `test-only` | robust optimization behavior |
| `backend/tests/test_route_baseline_api.py` | `test-only` | local working-tree baseline endpoint behavior |
| `backend/tests/test_route_cache.py` | `test-only` | route-cache semantics |

### Backend tests (part 2)

| File | Importance | Purpose |
| --- | --- | --- |
| `backend/tests/test_route_graph_no_path_mapping.py` | `test-only` | no-path reason-code mapping |
| `backend/tests/test_route_graph_precheck_timeout.py` | `test-only` | OD precheck timeout behavior |
| `backend/tests/test_route_graph_reliability_budget_and_rescue.py` | `test-only` | reliability budgets and rescue logic |
| `backend/tests/test_route_options_prefetch_gate.py` | `test-only` | route prefetch/live-gate behavior |
| `backend/tests/test_routing_graph_adaptive_hops.py` | `test-only` | adaptive hop calculation |
| `backend/tests/test_routing_graph_feasibility.py` | `test-only` | graph feasibility checks |
| `backend/tests/test_routing_graph_loader.py` | `test-only` | graph loader behavior |
| `backend/tests/test_routing_graph_streaming_parse.py` | `test-only` | streaming graph parse behavior |
| `backend/tests/test_run_store_reporting.py` | `test-only` | artifact storage and reporting |
| `backend/tests/test_scenario_compare.py` | `test-only` | scenario compare endpoint |
| `backend/tests/test_scenario_context_probe_timeout.py` | `test-only` | scenario context probe timeout behavior |
| `backend/tests/test_scenario_profile_strict_thresholds.py` | `test-only` | strict scenario coverage thresholds |
| `backend/tests/test_scripts_builders_extended.py` | `test-only` | builder-script coverage |
| `backend/tests/test_scripts_fetchers_extended.py` | `test-only` | fetcher-script coverage |
| `backend/tests/test_scripts_quality_extended.py` | `test-only` | quality-script coverage |
| `backend/tests/test_scripts_smoke_all.py` | `test-only` | broad script smoke tests |
| `backend/tests/test_scripts_test_matrix.py` | `test-only` | script test matrix |
| `backend/tests/test_segment_breakdown.py` | `test-only` | segment-breakdown payloads |
| `backend/tests/test_signatures_api.py` | `test-only` | signature verification APIs |
| `backend/tests/test_stochastic_uncertainty.py` | `test-only` | stochastic uncertainty pipeline |
| `backend/tests/test_stores_and_terrain_index_unit.py` | `test-only` | store helpers and terrain index units |
| `backend/tests/test_strict_reason_code_contract.py` | `test-only` | frozen reason-code contract |
| `backend/tests/test_terrain_dem_coverage.py` | `test-only` | terrain coverage checks |
| `backend/tests/test_terrain_dem_index_live_sampling.py` | `test-only` | live tile sampling/index behavior |
| `backend/tests/test_terrain_fail_closed_uk.py` | `test-only` | UK fail-closed terrain policy |
| `backend/tests/test_terrain_non_uk_fallback.py` | `test-only` | non-UK terrain behavior |
| `backend/tests/test_terrain_physics_unit.py` | `test-only` | terrain-physics unit behavior |
| `backend/tests/test_terrain_physics_uplift.py` | `test-only` | terrain uplift effects |
| `backend/tests/test_terrain_runtime_budget.py` | `test-only` | terrain runtime-budget enforcement |
| `backend/tests/test_terrain_segment_grades.py` | `test-only` | grade extraction and segment behavior |
| `backend/tests/test_toll_engine_unit.py` | `test-only` | toll-engine correctness |
| `backend/tests/test_tooling_scripts.py` | `test-only` | root/tooling script behavior |
| `backend/tests/test_traffic_profiles.py` | `test-only` | traffic/departure/scenario profile behavior |
| `backend/tests/test_uncertainty_model_unit.py` | `test-only` | uncertainty-model unit coverage |
| `backend/tests/test_vehicle_custom.py` | `test-only` | custom vehicle CRUD and validation |
| `backend/tests/test_weather_adapter.py` | `test-only` | weather multiplier behavior |
| `backend/tests/test_weights.py` | `test-only` | selection-weight logic |

### Grouped repetitive file families

| Pattern / family | Count | Importance | Purpose |
| --- | --- | --- | --- |
| `backend/data/raw/uk/toll_classification/class_*.json` | 220 | `supporting` | raw toll classification evidence corpus |
| `backend/data/raw/uk/toll_pricing/price_*.json` plus proxy-price JSONs | 105 | `supporting` | raw toll pricing evidence corpus |
| `backend/tests/fixtures/toll_classification/class_*.json` | 220 | `test-only` | deterministic toll-classification fixtures |
| `backend/tests/fixtures/toll_pricing/price_*.json` plus proxy-price fixtures | 105 | `test-only` | deterministic toll-pricing fixtures |
| `backend/tests/fixtures/uk_routes/*.json` | 11 | `test-only` | representative UK route fixtures for tests/benchmarks |
| `backend/data/raw/uk/*.summary.json` and `*.state.json` | several | `supporting` | metadata about raw collection runs and freshness |
| `backend/out/model_assets/*` local generated evidence | local runtime set | `generated/runtime` | compiled assets, preflight summaries, publish summaries, and coverage reports |
| `backend/out/artifacts/*` local run folders | local runtime set | `generated/runtime` | saved results, metadata, route GeoJSON, summaries, and PDFs |
| `backend/out/provenance/*`, `backend/out/oracle_quality/*`, `backend/out/analysis/*`, `backend/out/capsule/*`, `backend/out/experiments/*` | local runtime set | `generated/runtime` | provenance, quality dashboards, analyses, reproducibility capsules, and saved experiments |

### Local working-tree additions worth noting

These files are present locally and materially affect the current codebase, even though they are not part of the tracked baseline:

| File | Importance | Purpose |
| --- | --- | --- |
| `frontend/app/api/route/baseline/route.ts` | `supporting` | adds explicit OSRM baseline proxy |
| `frontend/app/api/route/baseline/ors/route.ts` | `supporting` | adds explicit ORS baseline proxy |
| `frontend/app/components/RouteBaselineComparison.tsx` | `supporting` | baseline comparison panel |
| `frontend/app/lib/baselineComparison.ts` | `supporting` | composite comparison scoring |
| `backend/tests/test_route_baseline_api.py` | `test-only` | baseline endpoint regression coverage |
| `backend/tests/test_pareto_backfill.py` | `test-only` | Pareto backfill regression coverage |

## Appendix K: UK-Only Constraints Summary

| Constraint | Evidence | Practical meaning |
| --- | --- | --- |
| default PBF is UK extract | `.env.example` | road network source is UK-specific |
| terrain UK bounding box | `backend/app/terrain_dem.py`, coverage report | terrain strictness only guaranteed inside UK bounds |
| UK bank holidays | bank-holiday loader and URL | weekday/weekend/holiday logic is UK-specific |
| UK corridor buckets and geohash contexts | scenario/departure assets | traffic and departure calibration is geography-specific |
| UK toll topology/tariffs | toll assets | toll logic is curated for UK infrastructure |
| non-UK terrain fallback/error behavior | terrain tests | outside UK, strict terrain can fail rather than guess |

## Appendix L: Benchmark-Evidence Summary

| Evidence type | What exists locally | What it supports | What it does not support by itself |
| --- | --- | --- | --- |
| per-run comparison UI | baseline comparison component and endpoints | route-by-route smart vs baseline explanation | global performance claim across all freight trips |
| benchmark scripts | `backend/scripts/benchmark_model_v2.py`, `backend/scripts/benchmark_batch_pareto.py` | runtime, memory, and p95 gate measurement | universal route-quality superiority claim |
| sensitivity/robustness scripts | `backend/scripts/run_sensitivity_analysis.py`, `backend/scripts/run_robustness_analysis.py` | stability under changed prices/seeds | aggregate provider-beating percentage |
| quality scoring | `backend/scripts/score_model_quality.py` | internal subsystem quality thresholds | external market-wide superiority |
| local generated outputs | model asset reports and preflight summaries | evidence that the strict/live pipeline ran successfully | universal OSRM/ORS win-rate |

## Glossary

| Term | Non-jargon meaning |
| --- | --- |
| Pareto set | routes where none is strictly better on every objective at once |
| Pareto dominance | one route is at least as good on all objectives and better on at least one |
| epsilon constraint | a hard ceiling used to throw away routes that exceed a chosen limit |
| knee point | a route near the "best compromise" bend of a trade-off curve |
| weighted sum | a single score formed by multiplying each objective by a preference weight |
| augmented Tchebycheff | a ranking method that focuses on the worst normalized objective plus a small tie-break term |
| VIKOR | a compromise-ranking method balancing group utility and worst regret |
| robust mode | pick routes that behave better under bad-tail uncertainty, not only on average |
| CVaR | the average of the worst tail of outcomes, used as a risk measure |
| strict/live | fail-closed mode that requires fresh, valid live or signed data instead of silently guessing |
| graph warmup | preloading the huge UK route graph before serving requests |
| baseline route | a simpler provider route used for comparison, not the full smart model |
| no sharing | assume no coordination benefit; penalties stay relatively high |
| partial sharing | assume some coordination benefit; penalties reduce somewhat |
| full sharing | assume near-ideal coordination; penalties are pushed toward floor values |
| scenario pressure | live-context stress from traffic, incidents, and weather |
| toll confidence | calibrated trust score for predicted toll assignment |
| oracle quality | dashboard view of how healthy and fresh external data sources are |
| provenance | saved record of how a run or artifact was produced |
| manifest | signed summary describing a run or model asset bundle |
| corridor bucket | geographic/contextual region used for scenario and departure calibration |
| route backfill | adding ranked routes when the strict Pareto frontier would otherwise be too small |

## Closing Assessment

The core of this codebase is not any single algorithm. It is the combination of:

- hybrid graph-plus-OSRM routing
- calibrated UK asset layers
- strict live-data governance
- multi-objective and robust selection logic
- unusually strong diagnostics, manifests, signatures, and test coverage

For thesis purposes, the most defensible central claim is that this repository is a freight-routing decision platform rather than a plain road-router. Its novelty, in the engineering sense, comes from how these parts are integrated and made auditable, not from claiming brand-new optimization theory.

## Appendix M: Detailed Workflow Narratives

### M1. Single-Route Compute (`POST /route`)

- User intent: ask the system for one recommended route rather than a whole frontier.
- Frontend entry point: the main page state in `frontend/app/page.tsx`.
- Frontend request assembly: `frontend/app/lib/requestBuilders.ts` builds a `RouteRequest`.
- Frontend transport path: `frontend/app/api/route/route.ts`.
- Backend endpoint: `POST /route` in `backend/app/main.py`.
- Request contract: `RouteRequest` in `backend/app/models.py`.
- Mandatory user choices: origin, destination, vehicle type.
- Optional user choices: waypoints, weights, scenario mode, departure time, terrain profile, stochastic settings, weather, incident simulation, and optimization mode.
- Backend first validates all fields with Pydantic.
- Backend then checks whether the route graph is usable under strict mode.
- If graph warmup is still incomplete, the request can fail with `routing_graph_warming_up`.
- If graph quality or OD connectivity is bad, failure can be `routing_graph_fragmented`, `routing_graph_coverage_gap`, or `routing_graph_disconnected_od`.
- If strict live mode is active, live scenario coefficients and related data are refreshed or validated before option construction.
- Candidate generation is graph-led rather than provider-led.
- That means the system first asks "what different plausible corridors exist?" before it asks OSRM for road-realized paths.
- This differs from many ordinary navigation systems, which trust the provider's alternative list directly.
- Graph candidate generation uses A*-assisted search and Yen-style K-shortest logic with state budgets.
- Candidate diversification then removes near-duplicates.
- OSRM refinement converts those candidates into actual route geometries and leg annotations.
- `build_option()` turns each realized route into a fully modeled `RouteOption`.
- That step adds scenario multipliers, departure multipliers, weather effects, terrain uplift, toll cost, fuel/energy use, emissions, carbon cost, uncertainty summaries, and counterfactuals.
- The backend then applies Pareto filtering internally even for a single-route outcome, because the final recommendation is chosen from a candidate set rather than from one provider route.
- The final scalar choice uses the configured math profile, which is `modified_vikor_distance` by default.
- That profile is not a pure academic VIKOR implementation.
- It is VIKOR plus engineering terms for balance, distance, ETA-distance interaction, knee penalty, and entropy reward.
- The frontend receives the selected route and usually also supporting route alternatives and diagnostics.
- User-visible payoff: one route is shown, but that route is the end product of a multi-stage multi-objective process.
- Thesis significance: this is the clearest place where the project departs from textbook shortest-path routing into decision-support routing.

### M2. Pareto JSON Compute (`POST /pareto`)

- User intent: inspect trade-offs rather than accept one answer immediately.
- Frontend entry point: same page orchestration, but using Pareto mode.
- Frontend request builder: `buildParetoRequest()` in `frontend/app/lib/requestBuilders.ts`.
- Frontend transport path: `frontend/app/api/pareto/route.ts`.
- Backend endpoint: `POST /pareto`.
- Core backend stages up to option construction are similar to single-route compute.
- The critical difference is what happens after route options exist.
- Instead of collapsing the result set immediately, the backend emphasizes non-dominated or epsilon-feasible routes.
- If `pareto_method=dominance`, the system keeps routes that are not dominated on duration, money, and emissions.
- If `pareto_method=epsilon_constraint`, the system first throws away routes that violate user-imposed limits.
- This is academically meaningful because the project exposes real multi-objective semantics to the user rather than hiding them.
- The repo also annotates crowding and knee information so the frontier is not just a raw list.
- This is where the code's use of NSGA-II-style crowding ideas matters.
- The route set can then be backfilled if the strict frontier is too small and route backfill is enabled.
- The local working-tree test `backend/tests/test_pareto_backfill.py` confirms this behavior.
- User-visible result: a trade-off set that can be plotted and inspected.
- Frontend components like `ParetoChart.tsx` and `SegmentBreakdown.tsx` make this useful rather than abstract.
- The backend still computes a recommended route selection in some contexts, but the main artifact is the frontier.
- Failure semantics remain strict.
- If all routes fail epsilon filters, the system emits `epsilon_infeasible`.
- If all routes die before frontier construction, the system emits `no_route_candidates`.
- Thesis significance: this endpoint most directly reflects the project's multi-objective identity.

### M3. Pareto Stream (`POST /pareto/stream`, `POST /api/pareto/stream`)

- User intent: receive frontier results incrementally instead of waiting for one large JSON payload.
- Frontend transport helper: `postNDJSON()` in `frontend/app/lib/api.ts`.
- Frontend proxy route: `frontend/app/api/pareto/stream/route.ts`.
- Backend endpoints: `POST /pareto/stream` and `POST /api/pareto/stream`.
- Streaming is important for user trust on long computations.
- Instead of a blank wait, the UI can show progress and partial route events.
- The stream emits NDJSON lines.
- The frontend treats them as event objects rather than plain text.
- Event families in the frontend type mirror include:
- meta events
- route events
- fatal events
- done events
- If the backend encounters a terminal strict failure, the stream emits a fatal event in canonical form.
- The strict error docs explicitly define this shape.
- Streaming does not change the underlying routing algorithms.
- It changes the delivery and observability model.
- This is architecturally important because large frontier generation and strict live prefetch can take time.
- The stream path therefore reduces perceived latency even when actual backend work is unchanged.
- Thesis significance: this feature shows product-oriented engineering around a research-style optimization pipeline.

### M4. Scenario Compare (`POST /scenario/compare`)

- User intent: compare how route outcomes change when the assumed sharing regime changes.
- Frontend request path: `buildScenarioCompareRequest()` plus `/api/scenario/compare`.
- Backend endpoint: `POST /scenario/compare`.
- Scenario compare is not just a frontend diff over static routes.
- The backend reruns route modeling under scenario assumptions.
- That matters because scenario mode changes several multipliers:
- duration
- incident rate
- incident delay
- fuel consumption
- emissions
- stochastic sigma
- The scenario asset stores mode-specific profiles for each context.
- Context selection depends on location bucket, hour, day kind, road mix, vehicle class, and weather regime.
- This means scenario compare is context-conditioned rather than globally hard-coded.
- The frontend can therefore present a meaningful result like "partial sharing reduces ETA pressure less than full sharing in this corridor at this time."
- The asset also tracks whether a context is driven more by observed outcomes or by projection.
- Thesis significance: this is a clear example of how the repo combines live context and calibrated policy-like scenario modeling.

### M5. Departure Optimize (`POST /departure/optimize`)

- User intent: choose not only the route, but the best time to leave.
- Frontend control surface: window start, window end, and step size.
- Backend endpoint: `POST /departure/optimize`.
- Request model: `DepartureOptimizeRequest`.
- The backend samples candidate departure slots across the requested time window.
- For each slot it recomputes route modeling with contextual departure multipliers.
- The departure profile logic is UK-local and context-aware.
- It uses day kind, region bucket, road bucket, and route shape.
- That is deeper than a simple peak/off-peak toggle.
- The asset provides confidence envelopes as well as point values.
- The response returns ranked departure candidates and the best slot.
- This feature is operationally realistic for freight because dispatch timing can be as valuable as route shape.
- Thesis significance: this is a planning optimization layered on top of routing optimization.

### M6. Duty Chain (`POST /duty/chain`)

- User intent: plan a chain of stops rather than a single OD trip.
- Frontend component: `DutyChainPlanner.tsx`.
- Backend endpoint: `POST /duty/chain`.
- Request model: `DutyChainRequest`.
- The system computes each leg and then composes results through `backend/app/multileg_engine.py`.
- Aggregation covers geometry, distance, duration, monetary cost, emissions, uncertainty, terrain summaries, toll metadata, and incidents.
- This is not just concatenating polylines.
- It is re-aggregating modeled route properties into a chain-level result.
- Thesis significance: it shows that the project's abstractions scale beyond single trips.

### M7. Batch Pareto (`POST /batch/pareto`)

- User intent: analyze many OD pairs in one run.
- Frontend devtool: Batch Runner.
- Backend endpoint: `POST /batch/pareto`.
- Request model: `BatchParetoRequest`.
- Each pair can produce routes or an error string.
- Successful runs produce manifests, provenance, artifacts, CSVs, GeoJSON, and reports under `backend/out`.
- This is where the project's reproducibility architecture becomes most visible.
- The run is not ephemeral.
- It becomes a saved analysis bundle with a `run_id`.
- This is useful for thesis experiments because one batch run can create a citable artifact family.

### M8. Batch CSV Import (`POST /batch/import/csv`)

- User intent: upload many pairs from a spreadsheet-like source.
- Frontend devtool: Batch Runner.
- Backend endpoint: `POST /batch/import/csv`.
- Request model: `BatchCSVImportRequest`.
- The backend parses CSV text into OD pairs plus routing settings.
- This is a convenience layer, but an important one.
- It reduces manual request assembly and supports reproducible scenario studies from tabular data.
- Thesis significance: it makes the system suitable for corpus-scale evaluation, not only map clicking.

### M9. OSRM Baseline (`POST /route/baseline`)

- User intent: compare the smart route against a simpler provider-driven route.
- Frontend local proxy: `frontend/app/api/route/baseline/route.ts`.
- Backend endpoint: `POST /route/baseline`.
- Backend logic intentionally bypasses the strict graph-warmup gate for this baseline path.
- That choice is validated by local tests.
- The idea is to preserve a provider baseline even if the enriched route pipeline is not ready.
- The baseline still returns modeled metrics, not only raw polyline geometry.
- Realism multipliers are applied so the comparison is not unfairly optimistic toward the provider path.
- Thesis significance: this endpoint is essential for evaluation and narrative defense of the smart system.

### M10. ORS Baseline (`POST /route/baseline/ors`)

- User intent: compare against an external reference route, not only OSRM.
- Frontend local proxy: `frontend/app/api/route/baseline/ors/route.ts`.
- Backend endpoint: `POST /route/baseline/ors`.
- If ORS configuration is complete, the backend can return `ors_reference`.
- If ORS is unconfigured and proxy fallback is allowed, it can return `ors_proxy_baseline`.
- If ORS is unconfigured and fallback is disallowed, the endpoint can fail with `baseline_provider_unconfigured`.
- This is a clean example of explicit evaluation-path governance.
- The code does not hide the difference between a real ORS reference and an ORS-like proxy baseline.
- Thesis significance: comparison rigor is baked into the API surface.

### M11. Experiments Catalog And Replay

- User intent: save a comparison setup and rerun it later without rebuilding the request from scratch.
- Frontend component: `frontend/app/components/ExperimentManager.tsx`.
- Backend store: `backend/app/experiment_store.py`.
- API surface: `GET /experiments`, `POST /experiments`, `GET /experiments/{experiment_id}`, `PUT /experiments/{experiment_id}`, `DELETE /experiments/{experiment_id}`, and `POST /experiments/{experiment_id}/compare`.
- Storage model: experiment bundles are file-backed rather than database-backed.
- That means the experiment feature is lightweight and portable.
- Each experiment stores a `ScenarioCompareRequest`.
- This makes sense because scenario comparison is the most natural saved-analysis workflow.
- The store keeps an index file plus one JSON file per experiment.
- Filtering supports query text, vehicle type, scenario mode, and sort order.
- This is not a complex lab-notebook system.
- It is a practical, auditable saved-query system.
- Thesis significance: experiment management shows that the repo supports repeated analytical work, not only one-off demos.

### M12. Custom Vehicle Lifecycle

- User intent: represent a fleet vehicle that is not one of the built-in UK defaults.
- Frontend devtool: `CustomVehicleManager.tsx`.
- Backend endpoints: custom vehicle CRUD under `/vehicles/custom`.
- Runtime persistence: `backend/out/config/` for the runtime custom-vehicle store, with backend compatibility handling for versioned and legacy JSON filenames.
- Validation is strict.
- IDs must match a regex.
- Aliases are normalized and deduplicated.
- EV profiles must carry EV-specific fields.
- Toll classes and stochastic/risk buckets must be populated.
- This is not a cosmetic profile layer.
- It directly affects terrain physics, toll mapping, fuel surfaces, and uncertainty buckets.
- Thesis significance: the project is extensible to real fleet heterogeneity.

### M13. Oracle Quality Workflow

- User intent: assess whether the feeds the system relies on are healthy and trustworthy.
- Frontend component: `OracleQualityDashboard.tsx`.
- Backend endpoints: `POST /oracle/quality/check`, `GET /oracle/quality/dashboard`, `GET /oracle/quality/dashboard.csv`.
- Backend store: `backend/app/oracle_quality_store.py`.
- One record represents one quality observation about a source.
- The dashboard aggregates records by source.
- Aggregates include pass rate, schema failures, signature failures, stale counts, average latency, and last observed time.
- This is not route quality directly.
- It is input quality.
- That distinction matters for a thesis.
- A route engine can only be as trustworthy as the feeds it consumes.
- Thesis significance: the project includes observability of its evidence sources, not only observability of its outputs.

### M14. Ops Diagnostics Panel

- User intent: understand whether the system is healthy before or during route compute.
- Frontend component: `frontend/app/components/devtools/OpsDiagnosticsPanel.tsx`.
- Backend endpoints used: `GET /health`, `GET /metrics`, `GET /cache/stats`, `DELETE /cache`.
- This panel exposes operational state instead of algorithmic results.
- It helps distinguish "the route is bad" from "the service is unhealthy."
- This is especially useful in strict mode, where readiness can legitimately block compute.
- Thesis significance: operational observability is treated as part of the product, not as an external admin-only concern.

### M15. Run Inspector And Artifact Retrieval

- User intent: inspect what a prior run produced and why.
- Frontend devtool: `RunInspector.tsx`.
- Backend endpoints cover manifests, scenario manifests, provenance, signatures, artifact listing, and file downloads.
- The user flow is:
- compute something
- capture the `run_id`
- inspect manifests and provenance
- inspect or download artifacts
- This flow is documented in both tutorial/reporting docs and sample-manifest docs.
- The design is useful for dissertations because the UI can reproduce the same evidence chain a thesis appendix might cite.
- Thesis significance: this is one of the strongest reproducibility features in the whole repository.

### M16. Signature Verification Workflow

- User intent: verify that a manifest or payload has not been altered.
- Frontend devtool: `SignatureVerifier.tsx`.
- Backend endpoint: `POST /verify/signature`.
- Backend implementation: `backend/app/signatures.py`.
- Signature creation uses canonical JSON plus HMAC-SHA256.
- Verification returns both a validity boolean and the expected signature.
- This is simple but effective.
- It gives the system a built-in integrity check for exported artifacts.
- Thesis significance: this supports defensible reporting and archival claims.

### M17. Tutorial Mode

- User intent: learn the system by guided interaction rather than by reading docs first.
- Frontend implementation: `TutorialOverlay.tsx` plus tutorial helper files under `frontend/app/lib/tutorial`.
- Tutorial scope from docs includes:
- route and Pareto generation
- scenario compare
- departure optimization
- duty-chain and experiment workflows
- artifact inspection
- This means the tutorial is aligned with the system's actual analytical breadth.
- It is not only a map-clicking tour.
- Thesis significance: onboarding is part of system design, which matters if the thesis discusses usability or interpretability.

### M18. Map Overlay Workflow

- User intent: visually inspect what the modeled route is doing at route, segment, stop, and incident levels.
- Frontend implementation: `MapView.tsx` and `frontend/app/lib/mapOverlays.ts`.
- Overlay types documented in the docs are:
- stops
- incidents
- segments
- Segment tooltips can expose:
- distance
- duration
- grade or energy fields
- component costs such as time, fuel, toll, and carbon
- This is important because the route output is not a black box.
- The user can inspect which pieces of the route are driving the total score.
- Thesis significance: the visualization layer supports explainability, not only presentation.

### M19. Accessibility And i18n Workflow

- User intent: make the application usable across interaction styles and languages.
- Docs identify current locales `en` and `es`.
- Accessibility coverage includes:
- skip link to controls
- focus-visible styling
- live-region updates for async compute states
- keyboard-friendly form controls
- This matters in a strict asynchronous application because long-running tasks can otherwise become inaccessible or opaque.
- Thesis significance: even though this is not routing math, it is part of the project's engineering completeness.

### M20. Dev Live-Trace Workflow

- User intent: inspect what live requests the backend actually made during a route computation.
- Frontend route: `/api/debug/live-calls/[requestId]`.
- Backend route: `GET /debug/live-calls/{request_id}`.
- Trace records include:
- expected calls
- observed calls
- success/failure state
- retry counts
- cache hits
- stale-cache use
- blocked stages
- sensitive header behavior gated by dev settings
- This is one of the most operationally rich features in the codebase.
- It turns live data from a hidden dependency into an inspectable sub-process.
- Thesis significance: this is unusually strong instrumentation for a routing project.

### M21. Full Local Startup Workflow

- User intent: boot the complete system with OSRM, strict live preflight, backend, and frontend.
- Root command: `scripts/dev.ps1`.
- The script can create `.env` if missing.
- It starts Docker Desktop if needed.
- It boots the OSRM compose services.
- It waits for OSRM readiness.
- It runs strict live preflight.
- Only then does it launch backend and frontend.
- This is more disciplined than "start everything and hope."
- It reflects the repo's strict-first philosophy.
- Thesis significance: startup itself is treated as a validated pipeline.

### M22. Full Refresh / Rebuild Workflow

- User intent: rebuild the stack and assets end to end.
- Operational doc path: `docs/run-and-operations.md`.
- Root cleanup command: `scripts/clean.ps1`.
- Backend rebuild path uses:
- `backend/scripts/build_model_assets.py`
- `backend/scripts/publish_live_artifacts_uk.py`
- optional graph and terrain builders
- This workflow matters because the repo distinguishes:
- raw evidence
- built assets
- published runtime assets
- generated runtime outputs
- Thesis significance: rebuild semantics are explicit and reproducible.

### M23. Low-Resource Test Workflow

- User intent: run tests safely on constrained machines.
- Root script: `scripts/run_backend_tests_safe.ps1`.
- The script offers CPU, priority, timeout, memory, and output-stall controls.
- That is evidence that the maintainer expects heavy workloads.
- It also shows concern for reproducibility under limited hardware.
- Output summaries are written under `backend/out/test_runs/`.
- Thesis significance: the project treats resource-bounded validation as a first-class operational problem.

### M24. Docs Validation Workflow

- User intent: keep repo documentation aligned with code.
- Root script: `scripts/check_docs.py`.
- Checks include:
- markdown links
- orphan docs and related-doc sections
- referenced local paths
- endpoint parity against `backend/app/main.py`
- forbidden notebook references
- This is unusual but valuable.
- It means docs drift is itself testable.
- Thesis significance: the repository treats documentation as an auditable artifact rather than passive prose.

## Appendix N: Detailed Backend Runtime Module Narratives

### `backend/app/main.py`

- This is the orchestration center of the backend.
- It is where the FastAPI app is created.
- It owns the HTTP surface exposed to the frontend and to any direct client.
- It starts graph warmup during lifespan handling.
- It wires the OSRM client dependency.
- It contains the route-construction path that turns provider geometry into a modeled route option.
- It also contains the scalar route-selection logic and the baseline endpoints.
- If a thesis needs one file that represents the system's integration logic, this is that file.

### `backend/app/models.py`

- This file defines the public contract of the backend.
- Every important request and response shape is validated here.
- That includes route, Pareto, batch, scenario compare, departure optimize, duty chain, vehicle, signature, and oracle-quality payloads.
- The frontend mirrors many of these types.
- That makes this file central to interface stability.
- In thesis terms, it defines what the system believes a route request and route answer actually are.

### `backend/app/settings.py`

- This file turns environment variables into typed runtime settings.
- It is the place where strict defaults become operational policy.
- Many "why does the system behave this way?" answers resolve to settings defined here.
- It governs live-source URLs, retries, graph budgets, terrain thresholds, baseline multipliers, and selection weights.
- It is therefore both configuration and encoded design philosophy.
- The report's environment appendix is largely an interpretation of this file and `.env.example`.

### `backend/app/model_data_errors.py`

- This file freezes the canonical strict reason-code set.
- That makes failures part of the contract, not an ad hoc implementation detail.
- Normalization ensures that unknown internal exceptions do not leak inconsistent reason codes.
- The project therefore treats explainable failure as a feature.
- This is thesis-relevant because it supports auditability and repeatable error interpretation.

### `backend/app/routing_graph.py`

- This file is the graph-native routing layer.
- It loads the large UK graph asset.
- It tracks warmup state and graph readiness.
- It performs OD feasibility and candidate generation.
- It also enforces graph integrity gates such as giant-component checks and nearest-node coverage.
- This file is where the project most clearly differs from "call OSRM and trust whatever comes back."
- It is one of the codebase's strongest engineering contributions.

### `backend/app/k_shortest.py`

- This module implements K-shortest-path search under practical runtime limits.
- The comments tie it to Yen-style path enumeration.
- The implementation is not purely textbook.
- It adds deadlines, state budgets, detour caps, and heuristic support.
- Those changes matter because freight-scale graphs and strict runtime budgets make unconstrained academic algorithms impractical.
- This is a classic example of academic method plus engineering adaptation.

### `backend/app/routing_osrm.py`

- This file is the OSRM integration layer.
- It hides provider request mechanics from the rest of the app.
- It manages retries and parameter compatibility.
- It supports via points and alternative routes.
- It is used both for enriched route refinement and for baseline generation.
- In architectural terms, this is the boundary between provider routing and project-specific reasoning.

### `backend/app/pareto.py`

- This module encodes strict Pareto dominance semantics.
- It is simple on purpose.
- The code uses an O(n^2) filter because candidate sets are intentionally kept small enough for that to be acceptable.
- The module is academically conventional.
- Its value in the project is not novelty but correctness and clarity.

### `backend/app/pareto_methods.py`

- This file extends basic Pareto logic into a more usable frontier toolkit.
- It supports epsilon-constraint filtering.
- It adds crowding-distance style truncation.
- It annotates knee-like compromise points.
- This is where the frontier becomes user-facing rather than merely mathematically correct.
- It is also one of the repo files that most directly connects academic multi-objective ideas to practical UI needs.

### `backend/app/risk_model.py`

- This module defines how uncertainty summaries become decision signals.
- It includes quantiles, CVaR, normalized utility, and robust objective construction.
- It is central to the `robust` optimization mode.
- Without this file, uncertainty would be only descriptive.
- With it, uncertainty becomes actionable in selection.
- That is one of the strongest analytical features in the repository.

### `backend/app/uncertainty_model.py`

- This file transforms deterministic route metrics into sampled distributions.
- It links route signatures, departure slots, contextual buckets, and stochastic regimes.
- It computes the uncertainty summaries surfaced in route payloads.
- It also records clipping and diagnostic metadata about the sampling process itself.
- That makes the uncertainty layer inspectable rather than magical.
- In thesis terms, it is where route risk is operationalized.

### `backend/app/scenario.py`

- This module defines scenario modes and context construction.
- It translates route geography, time, vehicle, weather, and road mix into a scenario context key.
- It then maps that context to calibrated policy profiles for no, partial, or full sharing.
- This is a major differentiator of the project.
- The route is not only a road path; it is a path under a particular operational sharing assumption.
- That makes scenario logic a first-class modeling layer rather than a presentation toggle.

### `backend/app/departure_profile.py`

- This file turns departure time into a context-sensitive multiplier.
- It uses UK-local time.
- It distinguishes weekday, weekend, and holiday.
- It uses region buckets, road buckets, and route shape.
- This is a richer model than a generic rush-hour scalar.
- It is central to the departure optimization feature.

### `backend/app/fuel_energy_model.py`

- This module computes fuel use, energy use, and associated emissions from route segments.
- It uses calibrated surfaces rather than a single fixed factor.
- It differentiates among diesel, petrol, LNG, and EV contexts.
- It responds to load, speed, grade, and temperature.
- This is one of the core physical-economics modules in the project.
- It is also one reason the smart route can differ materially from a plain provider route.

### `backend/app/terrain_physics.py`

- This file encodes the simplified vehicle-physics layer.
- It models the effect of drag, rolling resistance, and slope on duration and emissions.
- It uses vehicle-specific terrain parameters.
- It is not a full dynamics simulator.
- It is an engineering abstraction designed to inject realistic directional effects at acceptable runtime cost.
- The thesis can present it as a physically informed uplift model rather than as raw terrain decoration.

### `backend/app/terrain_dem.py`

- This module turns route geometry into terrain summaries.
- It densifies paths, samples elevations, computes coverage, and derives ascent, descent, and grade histograms.
- It also enforces strict UK coverage policy.
- That means it is both a model module and a strictness gatekeeper.
- The route cannot claim terrain awareness unless this file can support it.
- This is important for methodological honesty.

### `backend/app/terrain_dem_index.py`

- This file manages terrain manifests and live tile fetching.
- It implements live tile cache behavior and terrain diagnostics.
- It also tracks circuit-breaker state for repeated live tile failures.
- This is a highly practical module.
- It shows how the project treats remote DEM tiles as an operational subsystem, not just a data file.
- Thesis significance: terrain strictness is backed by real runtime engineering.

### `backend/app/toll_engine.py`

- This file maps route geometry onto tolled infrastructure and tariffs.
- It is class-aware and confidence-aware.
- It depends on topology, pricing tables, and calibration.
- Without this file, cost modeling would miss an important freight cost driver.
- Its confidence layer is especially notable because it admits uncertainty in toll inference.
- That makes the cost model more honest.

### `backend/app/carbon_model.py`

- This module loads carbon schedules under strict provenance rules.
- It checks freshness.
- It checks that uncertainty distributions exist.
- It rejects suspect provenance labels in strict mode.
- This is an explicit defense against silently weak environmental cost inputs.
- Thesis significance: carbon pricing is treated as a governed data input, not a loose multiplier.

### `backend/app/weather_adapter.py`

- This file translates weather scenarios into routing multipliers.
- It is intentionally simple and profile-based.
- Its role is not to forecast weather.
- Its role is to convert weather labels and intensity into ETA and incident effects.
- This keeps the weather subsystem interpretable and cheap.
- It also integrates cleanly with scenario and incident logic.

### `backend/app/incident_simulator.py`

- This module generates controlled synthetic incident events.
- It is seeded for repeatability.
- It supports dwell, accident, and closure event families.
- It scales probabilities by route length and weather incident uplift.
- It is meant for controlled scenario experimentation, not for strict live truth.
- That distinction is central to the project's honesty about simulation versus observation.

### `backend/app/multileg_engine.py`

- This file aggregates multiple legs into duty-chain results.
- It combines metrics, geometry, uncertainty, terrain, tolls, and incident metadata.
- It allows route reasoning to scale from one leg to a sequence of operational legs.
- This is important for practical freight work, where routes are often part of a schedule chain.
- The module therefore extends the project from path planning toward duty planning.
- That broadens thesis scope beyond single-trip optimization.

### `backend/app/vehicles.py`

- This module defines built-in and custom vehicle semantics.
- It validates schema, aliases, EV fields, toll classes, stochastic buckets, and terrain parameters.
- It also persists custom profiles to runtime output storage.
- Vehicle heterogeneity is therefore a first-class concern.
- This matters because a van, rigid HGV, artic, and EV truck should not be scored identically.
- The module anchors that realism.

### `backend/app/route_cache.py`

- This file implements the in-memory route cache.
- It tracks hits, misses, and evictions.
- It enforces TTL and max entry count.
- It deep-copies payloads so callers do not mutate cache contents accidentally.
- The cache is operational rather than algorithmic, but it strongly affects user experience.
- It matters particularly during repeated experimentation from the frontend.

### `backend/app/live_data_sources.py`

- This is the backend's live-feed ingestion hub.
- It owns bounded retries, caching, host allow-lists, and request tracing hooks.
- It fetches scenario context inputs, fuel snapshots, toll payloads, departure profiles, bank holidays, stochastic regimes, and carbon schedules.
- It is one of the most operationally dense modules in the codebase.
- Many strict failures begin here.
- Thesis significance: this file embodies the project's live-data governance philosophy.

### `backend/app/live_call_trace.py`

- This module records what live calls were expected and what live calls actually happened.
- It stores per-request traces with summary rollups.
- It supports cache-hit suppression rules and blocked-stage explanations.
- This is unusually rich instrumentation for a routing project.
- It turns live-data use into an auditable trace.
- That is central to reproducibility and debugging.

### `backend/app/metrics_store.py`

- This module stores endpoint-level request counts and timing summaries.
- It is intentionally simple and in-memory.
- Its value is visibility, not analytics sophistication.
- It supports the Ops Diagnostics panel and quick health interpretation.
- The project does not need Prometheus-scale machinery for the use case shown here.
- But it still wants explicit endpoint timing accountability.

### `backend/app/oracle_quality_store.py`

- This file persists source-quality observations and builds a dashboard summary.
- It aggregates stale counts, pass rates, signature failures, and latency.
- It provides a quality layer for the project's upstream evidence sources.
- That is conceptually distinct from route quality.
- The design is useful in a thesis because it demonstrates concern for the reliability of model inputs.
- It also supports human-readable and CSV export paths.

### `backend/app/provenance_store.py`

- This module records run events as provenance logs.
- It is one of the main reasons the system can explain how a run was produced.
- Provenance is not the same as a result file.
- It is a process record.
- This distinction matters in rigorous reporting.
- The module is therefore a quiet but important reproducibility building block.

### `backend/app/run_store.py`

- This file writes manifests and artifact files for runs.
- It standardizes artifact names and output locations.
- It attaches signatures to manifests.
- It is central to batch workflows and artifact retrieval.
- In thesis terms, it operationalizes reproducible experiment outputs.
- It is where analytical runs become named, inspectable objects.

### `backend/app/signatures.py`

- This module handles signing and verifying canonical payloads.
- It uses HMAC-SHA256.
- The implementation is intentionally transparent.
- It is not an advanced cryptographic system.
- It is an integrity mechanism for manifests and related payloads.
- That level of simplicity is appropriate and defensible for the repository's needs.

### `backend/app/reporting.py`

- This file is responsible for report generation outputs such as PDF artifacts.
- It turns stored run artifacts into a more human-consumable summary object.
- This is important because raw JSON and CSV are not always the best thesis or operator-facing artifacts.
- The project therefore includes both machine-readable and human-readable outputs.
- Reporting is treated as part of the system, not as an external manual step.
- That strengthens the project's end-to-end completeness.

### `backend/app/experiment_store.py`

- This module manages stored experiment bundles on disk.
- It keeps indexing simple and file-based.
- It validates saved requests by reusing core models.
- That reduces drift between runtime behavior and saved-analysis behavior.
- The module is modest in complexity but important in workflow.
- It lets scenario-comparison studies persist across sessions.

### `backend/app/objectives_selection.py`

- This helper module contains a simpler weighted-sum selector.
- It is useful as a baseline or reference path.
- It is not the most advanced selection logic in the repo.
- But it matters because it provides a clean academic-style reference implementation.
- That is helpful for comparison against the modified profiles.
- It supports the thesis discussion of academic versus engineering scoring.

### `backend/app/objectives_emissions.py`

- This module provides a simpler emissions objective helper based on distance, speed, and idle behavior.
- It is more lightweight than the richer fuel/energy surface logic.
- Its presence shows that the repo evolved through simpler and richer modeling layers.
- This is useful in a thesis because it illustrates how the project moved beyond placeholder cost/emissions treatment.
- It can be described as a lower-fidelity reference helper.
- It is supporting rather than central.

### `backend/app/logging_utils.py`

- This file configures structured JSON logging.
- It writes to stdout and, if possible, to a local log directory.
- Logging is therefore machine-friendly and analysis-friendly.
- This supports post-hoc debugging and runtime evidence capture.
- The design is practical rather than ornate.
- It fits the rest of the repo's auditability theme.

### `backend/app/rbac.py`

- This module is a no-op RBAC shim.
- It preserves endpoint signatures while leaving access control effectively disabled.
- That tells a useful story about project scope.
- Security roles are recognized as a future concern, but they are not the core thesis focus here.
- The shim prevents invasive refactors later.
- It is a placeholder with structural value.
## Appendix O: Detailed Frontend Runtime Narratives

This appendix expands the frontend coverage from a feature checklist into a file-level explanation.

The point is not to pretend every frontend file is mathematically deep.

The point is to show that the frontend is an analytical instrument.

It does not merely display one route returned by the backend.

It actively shapes requests, exposes modeling choices, reveals diagnostics, stores experiments, surfaces provenance, and lets a user inspect why the backend behaved the way it did.

### Frontend Shell And Core Orchestration

### `frontend/app/page.tsx`

- This is the dominant orchestration file for the frontend.
- It contains the main page state, request lifecycle, fallback logic, chart binding, route selection logic, and many advanced workflow handlers.
- In practice, this file acts as a small application shell rather than a thin page wrapper.
- It mirrors important backend concepts directly in the UI: Pareto mode, batch mode, scenario compare, departure optimization, duty chain, custom vehicles, signatures, and diagnostics.
- It also implements the frontend-side degrade ladder described in docs: stream first, then JSON Pareto, then single-route fallback.
- That fallback logic matters because the backend can take significant time when strict live prefetch and graph-led candidate generation are active.
- The file therefore embodies the product decision that compute should be observable and resilient rather than opaque.
- It also mirrors selection math labels exposed to the user, including academic reference and modified profiles.
- That is important for a thesis because the UI is not hiding the difference between unchanged academic formulas and engineering blends.
- The page is therefore a presentation layer, an experiment console, and a method-comparison surface at the same time.

### `frontend/app/layout.tsx`

- This file supplies the application frame and metadata.
- In a thesis context, it is not algorithmic, but it is still part of the reproducibility story because it defines the top-level shell in which every workflow runs.
- It ensures the analytical page is consistently mounted.
- It is also the obvious place for future shared providers if the system grows.
- Its simplicity is a sign that the project keeps complexity in modeling logic rather than in unnecessary layout abstraction.
- That is a pragmatic choice for a research-heavy application.

### `frontend/app/global.css`

- This stylesheet defines the visual grammar of the interface.
- It matters more than it may first appear because the project exposes a dense amount of analytical detail.
- Good spacing, hierarchy, and status coloring are needed so diagnostics, route cards, charts, and dev tools remain readable.
- The file therefore contributes indirectly to explainability.
- A research UI that overwhelms the operator reduces the value of all backend rigor.
- This stylesheet supports the goal of making rich model output readable during long investigation sessions.

### Core Frontend Libraries

### `frontend/app/lib/types.ts`

- This file mirrors the backend contract into TypeScript.
- It is one of the most important frontend files because it prevents the UI from drifting away from the FastAPI response schema.
- The route option shape, stream event types, batch types, scenario compare payloads, duty-chain payloads, and quality dashboard types all live here.
- In plain English, this file is the frontend's promise about what a backend response should look like.
- In engineering terms, it reduces accidental schema mismatch and helps keep refactors honest.
- In thesis terms, it proves that the UI has a typed understanding of the backend's analytical surface.
- This also matters for the academic comparison controls because the frontend must know which fields exist for scalar scores, trade-off charts, and baseline explanations.

### `frontend/app/lib/api.ts`

- This file holds low-level frontend API helpers.
- It standardizes how JSON and NDJSON requests are issued from the browser-facing app layer.
- Streaming is especially important here because Pareto stream mode is not a trivial fetch-and-render pattern.
- The helper layer makes stream consumption predictable and reusable.
- In thesis language, this file is part of the reason the frontend can treat route generation as a staged process rather than a black box.
- It supports the project's emphasis on long-running but inspectable analytical requests.

### `frontend/app/lib/backendFetch.ts`

- This file centralizes the actual proxy fetch behavior toward the backend-facing routes.
- It gives the page shell one place to manage error translation, timeout expectations, and response parsing conventions.
- That is useful because strict failures use canonical reason codes and the frontend needs to preserve them clearly.
- The file therefore acts as a narrow translation boundary.
- It is not where routing decisions happen, but it is where backend truth is protected from ad hoc UI handling.
- That contributes to the consistency of error explanations throughout the application.

### `frontend/app/lib/requestBuilders.ts`

- This file converts UI state into backend request payloads.
- It is critical because the page offers many operator-facing controls whose backend names are not always identical to the human-facing labels.
- The builder layer ensures that route, Pareto, scenario compare, batch, departure, and duty-chain requests stay structurally correct.
- In plain English, it is the map from "what the user picked" to "what the backend expects."
- In methodological terms, it is also where repeated request construction becomes reproducible instead of manually assembled.
- That matters for dissertation screenshots and reruns because the same control state can be transformed consistently every time.

### `frontend/app/lib/weights.ts`

- This file mirrors the backend route-selection formulas in TypeScript.
- That design is unusually valuable for a thesis because it makes the mathematical choice visible on both sides of the stack.
- The file explicitly cites Marler and Arora (2010), Steuer and Choo (1983), Opricovic and Tzeng (2004), Martins (1984), Branke et al. (2004), and Shannon (1948).
- It also explicitly says the modified profiles are transparent engineering blends rather than claimed novel theory.
- The weighted sum, augmented Tchebycheff, academic VIKOR, modified hybrid, modified distance-aware, and modified VIKOR-distance profiles are implemented as frontend-visible logic.
- That means the frontend can compare routes using the same logic family as the backend instead of inventing a second interpretation.
- In practical terms, this file explains how a highlighted representative route is chosen from an already feasible Pareto set.
- In thesis terms, it is one of the clearest places where academic methods and local engineering modifications meet.

### `frontend/app/lib/baselineComparison.ts`

- This file computes or formats differences between the smart route and baseline routes.
- Its job is to make comparisons understandable rather than leaving the user to inspect raw numbers manually.
- This matters because the project is partly motivated by outperforming or out-explaining simpler baselines.
- The comparison helper turns metric deltas into a coherent narrative surface for the UI.
- It is therefore small but conceptually important.
- Without it, the baseline endpoints would exist, but the practical comparison workflow would be much weaker.

### `frontend/app/lib/format.ts`

- This file formats durations, currency, emissions, and similar output fields.
- It is not mathematically deep, but it is necessary for interpretability.
- Research interfaces fail quickly if users constantly decode raw seconds and floating-point costs.
- This file improves comprehension of backend outputs.
- It therefore supports explainability and presentation quality.
- The underlying model remains the same, but the human cost of reading it is lowered.

### `frontend/app/lib/i18n.ts`

- This file stores localized UI strings for English and Spanish.
- It demonstrates that the project does not treat accessibility and language support as afterthoughts.
- The content is directly relevant to route generation because translations exist for map setup, compute actions, and trade-off explanations.
- That means the analytical surface is intentionally exposed in more than one language.
- For a thesis, this supports a broader claim that the frontend is being designed as an operator tool, not only as a developer demo.
- It also shows that even highly technical workflows can be internationalized if the text is kept structured.

### `frontend/app/lib/sidebarHelpText.ts`

- This file contains contextual help text for controls and workflows.
- It matters because many controls are not self-evident to a casual user.
- Compute mode, Pareto method, selection profile, uncertainty settings, and baseline comparisons all benefit from plain-English explanation.
- This file is therefore part of the application's embedded documentation layer.
- In thesis terms, it reduces jargon distance between the model and the operator.
- It also means the main page can explain advanced behavior without forcing users to leave the UI.

### `frontend/app/lib/mapOverlays.ts`

- This file organizes the overlay data structures used to paint route-related information on the map.
- Overlays are not cosmetic in this project.
- They represent interpreted route state such as stops, incidents, and segment information.
- The overlay helper therefore links geometry, annotations, and tooltip content together.
- This matters because the project is not only about numeric ranking.
- It is also about being able to inspect where and why a route behaves differently across space.

### `frontend/app/lib/tutorial/types.ts`

- This file defines the structure of tutorial steps and progress.
- It supports a guided experience rather than leaving new users to discover advanced features by accident.
- The typed model ensures tutorial sequences are explicit and stable.
- That is especially useful in a project with many workflows beyond simple route compute.
- In thesis terms, it shows deliberate pedagogical design inside the product.
- The project is teaching its own analytical surface.

### `frontend/app/lib/tutorial/steps.ts`

- This file encodes the tutorial sequence itself.
- The steps cover map setup, compute mode choice, Pareto generation, route selection, and follow-on analytical flows.
- That means tutorial mode is aligned with the actual model architecture rather than being a generic onboarding overlay.
- The file documents what the project considers its core operator journey.
- In practical terms, it is also a useful high-level roadmap for dissertation writing because it reveals intended workflow order.
- The tutorial content is therefore both UX help and system self-description.

### `frontend/app/lib/tutorial/progress.ts`

- This file stores or computes tutorial completion progress.
- It matters because the guided flow is stateful.
- The system needs to know which instructional phase the user has completed and what should happen next.
- That persistence adds continuity to learning the interface.
- In thesis terms, it supports the argument that the frontend is built for repeated analytical use rather than one-shot demos.
- A tutorial that remembers progress is more operationally credible than a static overlay.

### `frontend/app/lib/tutorial/prefills.ts`

- This file provides prefilled values for tutorial scenarios.
- Those prefills make sure new users land in states that actually exercise the backend meaningfully.
- In plain English, it prevents a tutorial from asking the user to do something with invalid or empty inputs.
- In thesis terms, it is a subtle but important reproducibility aid.
- Guided screenshots and tutorial runs can begin from known-good states.
- That reduces noise when demonstrating the product in a dissertation or viva.

### Core Visual Components

### `frontend/app/components/MapView.tsx`

- This is the map rendering engine for the UI.
- It is one of the most important components because all route geometry, origin-destination placement, and overlay inspection ultimately happen here.
- The file includes careful viewport-fitting logic, occluder-aware padding, and debug tracing for fit behavior.
- That level of detail matters because a dense analytical screen can easily obscure route geometry if map layout is handled naively.
- The component is therefore doing more than drawing a polyline.
- It is managing how a complex, sidebar-heavy analysis screen still keeps the route readable.
- Tooltips and layer placement also turn backend segment metrics into spatial explanations.
- In thesis language, this component is where model output becomes geographic evidence.
- Its debug hooks are also notable because they show the repo treats frontend observability seriously, not only backend observability.

### `frontend/app/components/ParetoChart.tsx`

- This component visualizes the candidate frontier.
- It turns a list of route options into an interpretable trade-off space, currently emphasizing time versus CO2.
- The key point is that the user is not forced to read dozens of numeric cards to understand the frontier.
- A single chart point can represent a candidate route with a distinct objective compromise.
- This supports the core thesis argument that routing should not collapse to a single hidden optimum.
- It also makes the impact of Pareto filtering visible.
- The component is conceptually tied to the backend pareto.py and pareto_methods.py modules even though it is not computing the frontier itself.
- In plain English, it is how the user sees that more than one sensible route may exist.

### `frontend/app/components/SegmentBreakdown.tsx`

- This component displays segment-level breakdowns for a route.
- It translates raw backend segmentation into something a human can inspect.
- That matters because several modeled effects are segment-sensitive: terrain uplift, grade, fuel/energy use, and toll application.
- A route-level total can hide where the cost or risk is really coming from.
- This component helps the user see local causes rather than only global totals.
- In thesis terms, it improves explainability of the physical and economic model.
- It is especially useful when discussing why two geometrically similar routes still score differently.

### `frontend/app/components/CounterfactualPanel.tsx`

- This component exposes counterfactual route outputs.
- Counterfactuals matter because the backend can compute not just "what happened under chosen assumptions" but also "what would change if a modeled factor were absent or altered."
- The panel therefore supports explanatory analysis rather than only operational selection.
- In practical terms, it helps demonstrate the marginal effect of terrain, incidents, sharing assumptions, or similar model layers.
- In thesis writing, this is valuable because it separates mechanism from outcome.
- The component turns backend explanatory fields into a readable analyst surface.

### `frontend/app/components/RouteBaselineComparison.tsx`

- This component explains smart-versus-baseline differences.
- It is where the user sees whether the enriched route differs from OSRM or ORS in time, distance, money, or emissions terms.
- The component matters because the project's comparative claim should be auditable, not implied.
- It also helps avoid overclaiming.
- A user can inspect per-request differences rather than relying on vague statements about superiority.
- In thesis terms, the component operationalizes fairer baseline interpretation.

### `frontend/app/components/ScenarioComparison.tsx`

- This component renders scenario compare outputs.
- It is focused on how no-sharing, partial-sharing, and full-sharing assumptions alter modeled results.
- Its importance is conceptual as much as visual.
- The project treats scenario mode as a first-class modeling axis, and this component makes that axis inspectable.
- In plain English, it helps answer "how much does collaboration or coordination change the route outcome here?"
- In thesis terms, it visualizes policy-style scenario analysis inside the same routing platform.

### `frontend/app/components/ScenarioParameterEditor.tsx`

- This component exposes many of the advanced control surfaces behind scenario and optimization behavior.
- It is a bridge between non-technical user intent and technical backend fields.
- It covers inputs such as Pareto method, selection math profile, uncertainty options, scenario modes, and related advanced settings.
- The component therefore matters for methodological transparency.
- It does not hide the knobs that shape route generation.
- That makes the product more suitable for research than a sealed "recommend me a route" interface.

### `frontend/app/components/ScenarioTimeLapse.tsx`

- This component visualizes scenario changes across time.
- It helps the user see how route quality or scenario pressure evolves rather than reading one timestamp in isolation.
- In thesis terms, it reinforces that the project is dynamic and context-aware, not static.
- It pairs naturally with departure optimization and scenario compare workflows.
- The component is especially helpful when explaining why the same OD pair behaves differently under different departure slots.
- It converts time-conditioned calibration into an interpretable screen element.

### `frontend/app/components/DepartureOptimizerChart.tsx`

- This component displays departure optimization results.
- It visualizes the evaluated departure slots and the selected best departure time.
- That matters because departure optimization is not intuitive when represented only as a list of timestamps.
- A chart makes the cost/ETA landscape visible across the requested window.
- In thesis language, it reveals that departure optimization is a sampled comparative analysis, not magic.
- The component therefore supports both interpretability and operator trust.

### `frontend/app/components/EtaTimelineChart.tsx`

- This component focuses on ETA-oriented timeline interpretation.
- It helps turn modeled durations and uncertainty summaries into a time-based visual story.
- This is relevant when the user cares less about raw objective tables and more about "when will I likely arrive?"
- The component therefore surfaces one of the most operationally meaningful outputs from the backend.
- In thesis terms, it shows how stochastic summaries become planning information.
- It is part of the bridge from route science to dispatch practice.

### `frontend/app/components/DutyChainPlanner.tsx`

- This component handles multi-stop duty-chain planning on the frontend.
- It is important because duty planning is a genuinely different user task from single OD routing.
- The component collects stop sequences and displays chained results returned by the backend.
- In plain English, it is where the user asks for a whole work sequence rather than one trip.
- In thesis terms, it proves the project extends beyond isolated route selection into operational planning.
- It also demonstrates reuse of core route logic across a richer planning surface.

### `frontend/app/components/ExperimentManager.tsx`

- This component manages stored experiment bundles.
- It lets users save, browse, compare, and replay analytical scenarios.
- That is vital for dissertation and benchmarking workflows because repeated studies should not require manual re-entry.
- The component therefore turns one-off route analysis into a repeatable experiment catalog.
- It pairs closely with the backend experiment store.
- In thesis terms, it supports workflow continuity and comparative study design.

### `frontend/app/components/OracleQualityDashboard.tsx`

- This component renders the health of upstream data sources.
- It is deliberately separate from route quality.
- The dashboard focuses on source freshness, schema validity, signature issues, stale rates, and latency.
- That distinction is academically useful because it separates model-input reliability from route-output quality.
- The component is evidence that the project treats live data governance as a measurable subsystem.
- In plain English, it answers "can I trust the feeds the model is using right now?"

### `frontend/app/components/TutorialOverlay.tsx`

- This component runs the guided walkthrough layer.
- It is a pedagogical tool attached directly to live UI elements.
- That is important because the interface contains many advanced features that are easier to understand when introduced in a sequence.
- The component therefore improves discoverability of scenario compare, Pareto exploration, and related workflows.
- In thesis terms, it reduces the gap between implementation complexity and operator comprehension.
- It is a product-quality feature with research presentation value.

### `frontend/app/components/PinManager.tsx`

- This component manages origin, destination, and stop pins.
- It matters because map interaction is the first stage of almost every route workflow.
- Good pin management reduces input ambiguity and prevents downstream request errors.
- The component is simple conceptually, but it has high leverage over the whole system.
- In thesis terms, it is part of the user-input validation story.
- Reliable route science still depends on correctly captured geography.

### `frontend/app/components/FieldInfo.tsx`

- This component provides contextual explanatory text or metadata near input fields.
- It supports the same interpretability goal as sidebar help text but at a more local level.
- That is useful when advanced controls would otherwise be cryptic.
- In practical terms, it lowers the barrier to using features like scenario mode, stochastic sigma, or selection profiles.
- In thesis terms, it helps the UI carry some of the explanatory burden internally.
- This is important because the user may not have the rest of the documentation open.

### `frontend/app/components/Select.tsx`

- This is a reusable select/dropdown control component.
- Its role is infrastructural rather than algorithmic.
- However, many high-value model choices pass through it: vehicle type, scenario mode, selection math profile, and compute mode.
- Reuse here improves consistency across analytical controls.
- That matters for a dense UI where inconsistent controls would increase cognitive friction.
- The component is therefore a small but useful part of interface coherence.

### `frontend/app/components/CollapsibleCard.tsx`

- This component organizes information-dense panels into collapsible sections.
- It matters because the application exposes many optional analytical surfaces.
- Without progressive disclosure, the page would become visually unmanageable.
- The component therefore helps balance detail and usability.
- In thesis terms, it supports the argument that the frontend is built for deep analysis without forcing every detail to be open at all times.
- It is a structural aid for long-form route investigation.

### Development Tool Components

### `frontend/app/components/devtools/OpsDiagnosticsPanel.tsx`

- This component renders operational diagnostics such as health status, readiness, cache information, and request traces.
- It matters because strict/live routing can fail for reasons unrelated to route geometry.
- The panel makes those reasons inspectable.
- In plain English, it tells the user whether the system is blocked by graph warmup, stale live data, or another infrastructure issue.
- In thesis terms, it is a major part of the project's observability contribution.
- A research system that cannot explain its own blocked state is weaker than one that can.

### `frontend/app/components/devtools/RunInspector.tsx`

- This component lets users inspect stored run artifacts.
- It exposes manifests, provenance, signatures, and generated files for a chosen run identifier.
- That is crucial for reproducibility.
- It means a user can revisit what inputs, assets, and outputs defined a past analytical result.
- In thesis language, it operationalizes the evidence chain behind a claim.
- It is one of the strongest bridges between application UX and dissertation-grade documentation.

### `frontend/app/components/devtools/SignatureVerifier.tsx`

- This component provides a UI for signature checking.
- It sits on top of the backend verify-signature endpoint.
- Its existence matters because provenance is not only stored; it is also user-verifiable.
- In plain English, it lets the operator check whether a payload or manifest was altered.
- In thesis terms, this strengthens the integrity story around artifact exports.
- It turns a backend cryptographic utility into an accessible analytical control.

### `frontend/app/components/devtools/CustomVehicleManager.tsx`

- This component handles custom vehicle profile creation, editing, and deletion.
- It is important because the physical and cost models depend materially on vehicle parameters.
- The manager gives the frontend a controlled way to extend beyond bundled vehicle profiles.
- That means the system can support scenario-specific fleets rather than only defaults.
- In thesis terms, it is a strong example of configurable model inputs affecting downstream route evaluation.
- It also demonstrates that the repository is not hard-locked to one freight archetype.

### `frontend/app/components/devtools/BatchRunner.tsx`

- This component exposes batch Pareto execution through the UI.
- It matters because many serious evaluations are comparative, not single-route.
- The batch runner lets a user test multiple OD pairs and retrieve structured results.
- That supports benchmarking, sensitivity studies, and dissertation evidence gathering.
- In plain English, it turns the tool from a one-at-a-time router into a study runner.
- It is therefore essential for scaling analysis.

### Frontend API Proxy Routes

### `frontend/app/api/health/route.ts`

- This route proxies simple backend health status.
- It exists so the frontend can check backend availability without browser-side cross-origin complexity.
- The endpoint is lightweight but important to startup flow.
- It helps distinguish "backend not up" from "backend up but not ready for strict routing."
- In thesis terms, it supports operational transparency.
- A system that separates liveness from readiness is architecturally clearer.

### `frontend/app/api/health/ready/route.ts`

- This route proxies the richer readiness endpoint.
- It is more important than simple health because strict route compute depends on graph warmup and live-data freshness.
- The frontend uses it to gate compute and recommend user action.
- In plain English, it answers "is the smart router actually ready to serve strict requests?"
- This matters because readiness is a real analytical condition, not just server uptime.
- The proxy route carries that distinction to the UI cleanly.

### `frontend/app/api/route/route.ts`

- This is the proxy for single-route compute.
- It forwards the focused route request and returns the backend result.
- Its significance is that it is the final fallback in the frontend degrade chain.
- If streaming and Pareto JSON are unsuitable, this route still allows a direct recommendation workflow.
- In thesis terms, it demonstrates graceful degradation without changing the core modeling logic.
- The proxy keeps the browser-facing side simple while preserving backend strictness semantics.

### `frontend/app/api/pareto/route.ts`

- This route proxies JSON Pareto compute.
- It matters because JSON Pareto is the middle tier in the fallback ladder.
- The endpoint returns the full frontier in one response when streaming is not chosen or not suitable.
- In plain English, it is the non-streaming way to ask for multiple trade-off routes.
- The proxy preserves typed JSON handling and error transfer.
- It is a central route for the UI's multi-objective workflow.

### `frontend/app/api/pareto/stream/route.ts`

- This route proxies NDJSON Pareto streaming.
- It is one of the most distinctive frontend API routes because it supports progressive event delivery.
- The route makes long-running compute feel inspectable instead of frozen.
- It is also tied directly to the operational runbook's preferred compute sequence.
- In thesis terms, it demonstrates asynchronous analytical interaction rather than synchronous black-box querying.
- The proxy is a crucial part of the user's experience of complex route generation.

### `frontend/app/api/scenario/compare/route.ts`

- This route proxies scenario compare requests.
- It exists because scenario comparison is a first-class workflow, not just a parameter tweak inside another endpoint.
- The proxy keeps the frontend-side request structure stable while the backend computes multiple scenario results.
- In plain English, it asks how the answer changes when sharing assumptions change.
- In thesis terms, it is a route dedicated to policy-like comparative analysis.
- That is an unusually strong analytical feature for a routing UI.

### `frontend/app/api/departure/optimize/route.ts`

- This route proxies departure optimization.
- It is important because departure selection is handled as a sampled backend analysis, not as a static frontend heuristic.
- The route sends the time window and returns ranked slots.
- In plain English, it asks "when should I leave?" rather than only "how should I drive?"
- In thesis terms, it expands the routing problem into scheduling.
- The proxy gives the frontend clean access to that richer planning service.

### `frontend/app/api/duty/chain/route.ts`

- This route proxies duty-chain planning.
- Its presence shows that the frontend is not limited to one origin and one destination.
- The proxy packages multi-stop requests for the backend multileg engine.
- That matters because operational planning is often chain-based.
- In thesis terms, it broadens the problem scope from route choice to duty composition.
- The route is therefore strategically significant even if used less often than simple route compute.

### `frontend/app/api/batch/pareto/route.ts`

- This route proxies batch Pareto requests over multiple OD pairs.
- It supports scaled analytical studies.
- The frontend uses it for experiment-style workflows rather than interactive single-run decisions only.
- In plain English, it is how the user runs many trade-off searches in one command.
- In thesis terms, it is one of the main pathways for collecting broader evidence.
- The proxy route is therefore a bridge from UI to benchmark-like usage.

### `frontend/app/api/batch/import/csv/route.ts`

- This route accepts or proxies CSV-style batch imports.
- It exists so route study inputs can come from tabular operator data instead of manually entered pairs.
- That matters because realistic analytical work often starts from spreadsheets.
- The route therefore reduces friction in moving from external case lists into the router.
- In thesis terms, it supports reproducible studies using structured input corpora.
- It is a practical ingestion feature with high workflow value.

### `frontend/app/api/route/baseline/route.ts`

- This route proxies the OSRM baseline request.
- It exists because baseline comparison is a formal feature, not an afterthought.
- The frontend can therefore request a simpler reference route through a dedicated path.
- In plain English, it asks "what would the baseline provider-like route look like for this same trip?"
- In thesis terms, it enables per-case comparison without mixing baseline and smart logic on the client.
- The route supports disciplined benchmarking and explainability.

### `frontend/app/api/route/baseline/ors/route.ts`

- This route proxies the ORS baseline request.
- It is important because the project distinguishes between base OSRM and ORS-style references.
- The proxy makes the comparison surface broader than a single baseline family.
- That helps avoid overfitting the thesis narrative to one reference system.
- In plain English, it is the UI path for asking for the ORS comparison answer.
- The proxy preserves the backend's distinction between true ORS and proxy fallback behavior.

### `frontend/app/api/vehicles/route.ts`

- This route proxies retrieval of available vehicle profiles.
- It is needed because vehicle selection affects terrain physics, fuel use, emissions, and sometimes uncertainty bucketing.
- The frontend cannot expose meaningful vehicle controls without this contract.
- In thesis terms, it is part of the physical modeling input surface.
- It supports both bundled profile selection and validation of what profiles exist.
- The route turns backend vehicle catalogs into UI choices.

### `frontend/app/api/vehicles/custom/route.ts`

- This route proxies custom vehicle creation and listing.
- It matters because custom vehicles are the user-extensible part of the physical model.
- The route lets the frontend persist new profiles rather than treating them as temporary local state.
- In plain English, it is how the operator teaches the system about a new fleet archetype.
- In thesis terms, it demonstrates configurable model parameterization.
- It is a strong example of the frontend acting as a model-management tool.

### `frontend/app/api/vehicles/custom/[vehicleId]/route.ts`

- This route proxies updates and deletion for one custom vehicle.
- It complements the collection route by handling object-level management.
- That matters because analytical systems need controlled lifecycle operations for configurable inputs.
- In plain English, it is how an existing custom vehicle is edited or removed.
- In thesis terms, it supports disciplined management of user-defined physical assumptions.
- The route keeps custom-vehicle governance explicit.

### `frontend/app/api/experiments/route.ts`

- This route proxies experiment listing and creation.
- It underpins the experiment manager UI.
- The route matters because repeated analysis should not rely on browser memory alone.
- It provides a persistent catalog of saved analytical studies.
- In thesis language, it supports the move from ad hoc exploration to structured experimental workflow.
- It is a key route for longer-term comparative research use.

### `frontend/app/api/experiments/[experimentId]/route.ts`

- This route proxies fetch, update, or delete behavior for a specific stored experiment.
- It makes saved studies addressable and maintainable.
- That is useful because analytical comparisons evolve over time.
- The route gives the frontend a precise handle on one experiment record.
- In thesis terms, it supports versioned investigative workflow rather than disposable runs.
- It is a simple route with high workflow significance.

### `frontend/app/api/experiments/[experimentId]/compare/route.ts`

- This route proxies experiment comparison.
- It exists because saved experiments are not only stored; they can also be contrasted directly.
- That feature supports higher-level study design.
- In plain English, it helps answer "how did one saved scenario differ from another?"
- In thesis terms, it strengthens the comparative analysis story around the experiment store.
- The route is a natural extension of the repository's emphasis on reproducible comparison.

### `frontend/app/api/debug/live-calls/[requestId]/route.ts`

- This route proxies backend live-call traces for one request.
- It is extremely useful for development and dissertation explanation.
- The route turns upstream feed activity into a queryable artifact.
- In plain English, it helps answer "what live APIs did the backend actually touch for this route?"
- In thesis terms, it is one of the strongest transparency features in the whole frontend.
- It makes live-data dependence auditable rather than hidden.

### `frontend/app/api/oracle/quality/check/route.ts`

- This route proxies creation of oracle-quality observations.
- It supports active quality checks on upstream sources.
- The route is conceptually separate from route compute and that separation matters.
- In plain English, it is part of monitoring the health of what the model consumes.
- In thesis terms, it reinforces the distinction between input-quality monitoring and route-result ranking.
- The route contributes to live-data governance.

### `frontend/app/api/oracle/quality/dashboard/route.ts`

- This route proxies the oracle quality dashboard JSON.
- It provides the structured data used by the dashboard component.
- That matters because dashboarding is only as reliable as the feed behind it.
- In plain English, it fetches the aggregated source-health summary.
- In thesis terms, it exposes a governance subsystem dedicated to external inputs.
- It is part of the evidence chain for strict/live operation.

### `frontend/app/api/oracle/quality/dashboard.csv/route.ts`

- This route proxies CSV export of oracle-quality data.
- CSV export is operationally useful because it lets analysts move source-health records into spreadsheets or appendices.
- The route therefore extends monitoring into external reporting workflows.
- In plain English, it is the downloadable form of the source-health dashboard.
- In thesis terms, it supports citation and tabular analysis outside the app.
- It turns observability into a portable artifact.

### `frontend/app/api/runs/[runId]/[...subpath]/route.ts`

- This route proxies a family of run-artifact retrieval endpoints.
- It supports manifests, scenario manifests, provenance, signatures, listings, and artifact downloads behind one Next.js route pattern.
- That design keeps the frontend routing surface compact while still exposing many artifact operations.
- In plain English, it is the browser-facing door to stored run evidence.
- In thesis terms, it is central to reproducibility and report extraction.
- The route is one of the reasons the application can function as an evidence browser.

### `frontend/app/api/verify/signature/route.ts`

- This route proxies signature verification.
- It is the frontend counterpart to the backend integrity check.
- The route matters because artifact integrity should be testable from the UI, not only via CLI.
- In plain English, it lets a user submit a payload, signature, and secret for verification.
- In thesis terms, it strengthens the authenticity story around stored outputs.
- The route converts a backend utility into an accessible control.

### `frontend/app/api/cache/route.ts`

- This route proxies cache operations or cache introspection.
- It matters because the system uses caches for route reuse and live-source reuse.
- The proxy lets the frontend present or manage that behavior when appropriate.
- In plain English, it helps reveal whether cached state is affecting current runs.
- In thesis terms, it contributes to performance transparency.
- Cache behavior is an important part of interpreting repeated results.

### `frontend/app/api/cache/stats/route.ts`

- This route proxies route-cache statistics.
- It is useful for understanding hit rates, reuse patterns, and potential performance effects.
- That matters because route caching can change latency without changing the modeled route semantics.
- In plain English, it answers "is the system reusing recent route work?"
- In thesis terms, it supports a more precise discussion of performance than raw endpoint timing alone.
- The route is therefore part of the profiling surface.

### `frontend/app/api/metrics/route.ts`

- This route proxies backend metrics output.
- It exists because runtime metrics are part of the operational picture.
- The route helps the frontend expose health and performance counters without direct browser-to-backend handling.
- In plain English, it is a telemetry path.
- In thesis terms, it contributes to the observability stack rather than the route algorithm itself.
- That still matters because serious research software needs visible operational behavior.

### Frontend Architectural Interpretation

- The frontend as a whole is unusually dense for a research project because it does not stop at map display.
- It includes analytical controls, academic-method comparison, scenario analysis, artifact inspection, live-trace introspection, and reproducibility tooling.
- That breadth is important to state clearly in a thesis.
- The code is not simply a "demo UI" sitting on top of a strong backend.
- It is part of the methodology because it governs how users interrogate the backend and how they inspect evidence.
- The large `page.tsx` file is a tradeoff.
- It increases single-file complexity, but it also keeps the main analytical state visible in one place.
- That can be defended in a dissertation as a pragmatic research-interface choice rather than a scalable enterprise pattern.
- The typed request/response surface, mirrored selection math, and dedicated dev tools all strengthen the frontend's value as part of the scientific system.
- In short, the frontend is a route-analysis workbench, not just a route viewer.

## Appendix P: Detailed Backend Script Narratives

This appendix explains the backend script layer in more depth than the earlier file inventory.

These scripts matter because the project is not only a runtime application.

It is also a data-preparation, calibration, validation, and reproducibility pipeline.

The scripts explain how the live and strict runtime is made possible.

### Build-Orchestration Scripts

### `backend/scripts/build_model_assets.py`

- This is the umbrella asset-build script.
- It coordinates creation or refresh of the main runtime artifacts used by the strict backend.
- In plain English, it is the "prepare the system to be trustworthy" command.
- It matters because the runtime expects built tables, profiles, and supporting assets rather than raw data dumps.
- In thesis terms, it is one of the central reproducibility entry points.
- It also expresses the project's preference for scripted pipelines over manual notebook steps.

### `backend/scripts/publish_live_artifacts_uk.py`

- This script publishes generated artifacts into the tracked UK asset targets used by strict runtime.
- It connects build outputs to the files that live-source URLs and runtime loaders actually reference.
- In plain English, it is the promotion step from generated result to operational asset.
- This matters because a build that never gets published does not help the runtime.
- In thesis terms, the script marks the boundary between experimental generation and active deployment material.
- It therefore matters for provenance and version discipline.

### `backend/scripts/preflight_live_runtime.py`

- This script checks whether strict live runtime requirements are satisfied before service start or validation.
- It verifies that required live-backed sources are present, valid, fresh enough, and structurally acceptable.
- In plain English, it asks "if I start the backend now, will strict routing fail immediately because the inputs are stale or missing?"
- That is crucial in a fail-closed system.
- In thesis terms, this script is part of the governance model, not just operations glue.
- It externalizes readiness checks into a reproducible command rather than leaving them as implicit startup risk.

### `backend/scripts/validate_graph_coverage.py`

- This script checks routing-graph coverage and related structural quality.
- It helps ensure that the graph asset is not merely loadable but sufficiently usable.
- In plain English, it verifies that the graph actually covers the intended UK routing space in a defensible way.
- That matters because graph-led candidate generation is one of the project's main differentiators.
- In thesis terms, this script supports claims about graph availability and geographic adequacy.
- It is part of the evidence behind the coverage report.

### Core Asset Builders

### `backend/scripts/build_routing_graph_uk.py`

- This script compiles the UK routing graph from source data.
- It is one of the most important scripts in the whole repository.
- The runtime graph underpins graph-led candidate generation, warmup behavior, feasibility checks, and long-corridor rescue logic.
- In plain English, it builds the custom road-network brain that distinguishes the project from pure OSRM dependency.
- In thesis terms, this script is central to the system's hybrid architecture argument.
- It is also where UK-only scope becomes concretely embedded in an artifact.

### `backend/scripts/build_departure_profiles_uk.py`

- This script builds the departure-time profile asset.
- It turns observed or curated timing patterns into a structured departure multiplier table.
- In plain English, it teaches the backend how UK departure timing pressure changes across contexts.
- That matters because departure optimization requires more than raw routing.
- In thesis terms, the script operationalizes the shift from static trip modeling to schedule-aware trip modeling.
- It provides the data foundation for the departure optimizer endpoint.

### `backend/scripts/build_scenario_profiles_uk.py`

- This script builds the scenario profile asset for no-sharing, partial-sharing, and full-sharing modes.
- It is central to the scenario-analysis layer.
- In plain English, it learns or compiles how different sharing assumptions should change route pressure and outcome multipliers in different contexts.
- That matters because the scenario system is context-conditioned rather than hard-coded globally.
- In thesis terms, this script is a key part of the argument that the project extends beyond shortest path into policy-sensitive route modeling.
- It also turns live-observed scenario corpora into operational coefficients.

### `backend/scripts/build_stochastic_calibration_uk.py`

- This script builds the stochastic regimes and related uncertainty calibration artifacts.
- It matters because route uncertainty is not simulated with arbitrary noise.
- The runtime expects calibrated regimes, posterior context mappings, spreads, and copula structure.
- In plain English, this script teaches the system what "normal variation" and "bad tail behavior" look like under UK freight contexts.
- In thesis terms, it supports every statement about q95, CVaR, robust score, and correlated uncertainty.
- It is a foundational script for the uncertainty subsystem.

### `backend/scripts/build_pricing_tables_uk.py`

- This script builds pricing tables, especially around toll and related operational pricing surfaces.
- It takes raw or semi-raw pricing evidence and shapes it into runtime-usable tariff structures.
- In plain English, it turns messy pricing inputs into lookup-ready operational data.
- That matters because monetary cost modeling depends on consistent tariff logic rather than ad hoc price guesses.
- In thesis terms, this script helps make cost outputs traceable to maintained tables.
- It is part of the data-engineering layer behind route economics.

### `backend/scripts/build_terrain_tiles_uk.py`

- This script prepares terrain tiles or terrain-supporting assets for UK routing.
- Terrain matters because the project models grade-driven duration and emissions effects under strict coverage rules.
- In plain English, the script prepares the elevation evidence needed to stop terrain from being a hand-wavy adjustment.
- That matters because terrain is a genuine physical input, not just a scenario label.
- In thesis terms, this script supports the claim that terrain-aware routing is backed by actual geospatial data.
- It also contributes to fail-closed coverage control.

### Raw Data Collection And Fetch Scripts

### `backend/scripts/fetch_scenario_live_uk.py`

- This script fetches live scenario-related data for the UK context.
- It pulls the evidence families that later feed scenario profile construction and runtime refresh logic.
- In plain English, it is one of the main "bring live transport context into the project" scripts.
- It matters because scenario coefficients are not meant to be timeless constants.
- In thesis terms, this script links the project to real-world traffic, incident, and weather context feeds.
- It is part of the live-data ingestion story behind strict runtime.

### `backend/scripts/collect_scenario_mode_outcomes_proxy_uk.py`

- This script collects observed or proxy outcomes for scenario-mode calibration.
- It helps convert abstract sharing assumptions into empirically informed outcome patterns.
- In plain English, it gathers the evidence that lets no-sharing, partial-sharing, and full-sharing differ meaningfully.
- That matters because scenario modes need more than conceptual labels.
- In thesis terms, this script supports the calibration legitimacy of the scenario subsystem.
- It is one of the clearest places where operational assumptions are tied back to observed or proxy data.

### `backend/scripts/fetch_stochastic_residuals_uk.py`

- This script fetches stochastic residual evidence.
- Residual data is needed so uncertainty calibration reflects observed deviation behavior rather than arbitrary synthetic noise alone.
- In plain English, it collects the error-pattern evidence behind the uncertainty model.
- That matters because q95 and CVaR are only meaningful if the distributional assumptions are grounded.
- In thesis terms, this script strengthens the credibility of the robust-routing layer.
- It helps make the uncertainty model empirical rather than decorative.

### `backend/scripts/collect_stochastic_residuals_raw_uk.py`

- This script collects the raw residual corpus before final calibration shaping.
- It sits earlier in the pipeline than the final stochastic-regime builder.
- In plain English, it captures the rough observed mismatch data from which uncertainty tables are later learned.
- That matters because raw evidence provenance should be separable from processed calibration outputs.
- In thesis terms, it supports the chain from observation to calibrated uncertainty artifact.
- It is a useful script for data-lineage explanation.

### `backend/scripts/fetch_fuel_history_uk.py`

- This script fetches fuel history inputs used in fuel pricing or uncertainty layers.
- It matters because cost modeling depends on contemporary fuel conditions rather than timeless constants.
- In plain English, it collects the historical price evidence that later informs fuel-price assets.
- That matters for both route cost and uncertainty discussions.
- In thesis terms, it supports the economic realism of the model.
- It also links monetary outputs to time-sensitive external evidence.

### `backend/scripts/collect_fuel_history_raw_uk.py`

- This script preserves the raw side of fuel history collection.
- It is the data-lineage counterpart to the higher-level fetch/build pipeline.
- In plain English, it keeps the less-processed evidence around so the thesis can explain where the published fuel tables came from.
- That matters because transformed fuel assets should not appear from nowhere.
- In thesis terms, it supports provenance and reproducibility.
- It is part of the economic-data audit trail.

### `backend/scripts/fetch_carbon_intensity_uk.py`

- This script fetches carbon-intensity data.
- Carbon intensity is relevant because the project models emissions and carbon-cost surfaces, not just direct fuel burn.
- In plain English, it collects evidence for how carbon burden changes across time.
- That matters when the backend distinguishes direct emissions and monetized carbon cost.
- In thesis terms, it supports the environmental-accounting side of the router.
- It also helps explain how carbon schedule and intensity data interact.

### `backend/scripts/collect_carbon_intensity_raw_uk.py`

- This script stores the raw carbon-intensity evidence before final shaping.
- It gives the project a preserved raw layer beneath the published carbon asset.
- In plain English, it helps show the journey from downloaded carbon data to the runtime schedule or table.
- That matters because strict runtime likes validated artifacts, but the thesis still needs raw provenance.
- In thesis terms, this script is part of the environmental-data lineage.
- It supports defensible reporting of carbon inputs.

### `backend/scripts/fetch_dft_counts_uk.py`

- This script fetches Department for Transport count data or related traffic-count evidence.
- Such counts are relevant to corridor context, departure profiles, and scenario interpretation.
- In plain English, it gathers public traffic-count evidence that helps anchor the model in observed road use.
- That matters because UK corridor behavior should not be guessed completely in the dark.
- In thesis terms, it supports the contextual realism of the scenario and departure layers.
- It is part of the observation backbone behind UK-specific calibration.

### `backend/scripts/collect_dft_raw_counts_uk.py`

- This script stores the raw DfT count corpus and associated summaries.
- It preserves a less-processed form of the evidence used later in calibration.
- In plain English, it helps keep the raw traffic evidence visible and reproducible.
- That matters because transformed profile tables are easier to trust when their raw inputs are tracked.
- In thesis terms, this script strengthens the chain of evidence around UK traffic calibration.
- It is a practical provenance script.

### `backend/scripts/fetch_public_dem_tiles_uk.py`

- This script fetches public DEM tiles for terrain support.
- It is important because terrain coverage can come from live or prepared elevation tiles.
- In plain English, it acquires the raw elevation building blocks used to estimate grade.
- That matters because terrain-aware claims need real elevation evidence.
- In thesis terms, this script supports the physical-model side of the project.
- It also contributes to the UK-only geospatial pipeline.

### Toll And Road-Charge Scripts

### `backend/scripts/extract_osm_tolls_uk.py`

- This script extracts toll-related topology from UK OSM data.
- It helps identify where tolled infrastructure exists within the road network.
- In plain English, it is one of the scripts that turns map data into toll-aware route knowledge.
- That matters because toll modeling needs both tariff tables and route-topology awareness.
- In thesis terms, it supports the monetary realism of the route options.
- It is part of the structural side of toll calibration.

### `backend/scripts/fetch_toll_truth_uk.py`

- This script fetches operator or truth-style toll data.
- It captures pricing or classification evidence beyond what OSM topology alone can provide.
- In plain English, it helps answer "what are the real toll rules or charges that apply?"
- That matters because a tolled segment without a credible tariff is not enough for cost modeling.
- In thesis terms, this script supports the economic-grounding side of toll logic.
- It is part of the effort to move beyond simplistic toll flags.

### `backend/scripts/collect_toll_truth_raw_uk.py`

- This script stores the raw toll-truth collection outputs.
- It preserves the upstream evidence layer used to build the published toll tariff and confidence assets.
- In plain English, it is the raw toll archive script.
- That matters for provenance, debugging, and later recalibration.
- In thesis terms, it helps explain how operator truth and proxy pricing were blended into usable runtime tables.
- It is part of the toll audit trail.

### Analysis, Benchmark, And Validation Scripts

### `backend/scripts/score_model_quality.py`

- This script computes quality scores for asset subsystems.
- It is central to the quality-gate story documented elsewhere in the repo.
- In plain English, it asks whether the model inputs and subsystem outputs are good enough to trust.
- That matters because the project does not rely only on endpoint smoke tests.
- In thesis terms, this script is one of the clearest signs that the repository distinguishes correctness from quality.
- It provides structured, subsystem-aware evaluation rather than generic pass/fail behavior.

### `backend/scripts/benchmark_model_v2.py`

- This script runs benchmark measurements for the model runtime.
- It focuses on performance and operational behavior rather than on a single route example.
- In plain English, it helps measure how quickly the smart router behaves under benchmark conditions.
- That matters because a rich model that cannot respond within acceptable budgets becomes harder to use.
- In thesis terms, it supports claims about runtime feasibility and not just modeling sophistication.
- It is a key script in the performance narrative.

### `backend/scripts/benchmark_batch_pareto.py`

- This script benchmarks batch Pareto behavior.
- It is especially relevant because multi-objective and multi-pair analysis can be more demanding than single-route queries.
- In plain English, it asks how well the system handles many route-frontier computations in sequence or batch.
- That matters for study-scale use, not just interactive demo use.
- In thesis terms, it supports the practical viability of the batch-analysis workflow.
- It is closely tied to the reproducibility capsule and performance notes.

### `backend/scripts/check_eta_concept_drift.py`

- This script checks ETA concept drift.
- It matters because route-duration behavior can shift over time as live inputs, calibration assumptions, or external conditions change.
- In plain English, it helps detect when the system's ETA understanding may be drifting away from expected behavior.
- That matters for long-lived operational trust.
- In thesis terms, the script is evidence that the project treats temporal drift as an explicit risk.
- It extends the quality story beyond one-time calibration.

### `backend/scripts/run_sensitivity_analysis.py`

- This script runs one-factor sensitivity analysis for batch Pareto studies.
- It is important because model behavior should be explainable under controlled perturbation.
- In plain English, it tests how much outputs move when one parameter family is changed at a time.
- That matters because a thesis should be able to discuss robustness and parameter influence, not just default results.
- In thesis terms, this script supports the discussion of calibration leverage and tuning sensitivity.
- It is a formal tool for "what if this knob changes?" analysis.

### `backend/scripts/run_robustness_analysis.py`

- This script runs repeated-seed robustness analysis.
- It matters especially for stochastic outputs, where one seed should not be mistaken for a universal result.
- In plain English, it checks whether conclusions hold across multiple controlled randomness settings.
- That matters for credible reporting of uncertainty-aware routing.
- In thesis terms, it supports stronger claims about robust utility behavior.
- It complements sensitivity analysis by varying seed-driven uncertainty realizations instead of only model constants.

### `backend/scripts/run_headless_scenario.py`

- This script runs a scenario workflow without the frontend.
- It is useful for automation, smoke testing, and reproducible demonstrations.
- In plain English, it lets the thesis author prove that a scenario analysis can be reproduced from a saved input file alone.
- That matters because GUI-only workflows are harder to reproduce rigorously.
- In thesis terms, this script is a valuable bridge between interactive use and scripted reproducibility.
- It is one of the most dissertation-friendly scripts in the repository.

### `backend/scripts/generate_run_report.py`

- This script generates a report artifact from stored run outputs.
- It converts machine-readable run evidence into a more human-oriented report form.
- In plain English, it helps turn analysis results into something that can be read, shared, or attached to a thesis appendix.
- That matters because raw JSON is not always the best communication format.
- In thesis terms, it supports the final presentation layer of reproducible outputs.
- It is part of the system's end-to-end completeness.

### Supporting Script Notes

### `backend/scripts/__init__.py`

- This file marks the scripts directory as a package where needed.
- It is not analytically important on its own.
- Its value is structural.
- In plain English, it helps keep script imports predictable.
- In thesis terms, it is support infrastructure rather than substantive methodology.
- It is still part of the repository's operational hygiene.

## Appendix Q: Documentation Absorption Ledger

This appendix explains what each supporting markdown document contributes and how its content has been absorbed into the main thesis report.

The purpose is twofold.

First, it makes the main report stand alone.

Second, it records where the rest of the documentation was still useful as a source while preparing this thesis-grade master document.

### `docs/DOCS_INDEX.md`

- This file is the repo's documentation map.
- It matters because it reveals which topics the project treats as first-class: backend APIs, strict errors, quality gates, model assets, reproducibility, tutorial flows, accessibility, and math notes.
- In plain English, it is the table of contents for the engineering side of the project.
- Its content has been absorbed into this report by ensuring those same topic families each have a chapter or appendix.
- In thesis terms, the index helped verify coverage completeness rather than contributing unique algorithmic content.
- It is still useful as the repo-local navigation map for future maintenance.

### `docs/backend-api-tools.md`

- This document is the clearest compact description of the backend API contract.
- It lists route-producing endpoints, run-artifact endpoints, oracle-quality endpoints, and strict runtime assumptions.
- In plain English, it explains what the backend can do and what shape of request it expects.
- Its material has been absorbed into the endpoint inventories, workflow narratives, strict/live chapters, and frontend-control mapping appendix.
- The document was especially useful for confirming newer endpoints such as readiness checks, live-call trace retrieval, and baseline routes.
- In thesis terms, it acts as a contract-level companion to the runtime code.

### `docs/api-cookbook.md`

- This document provides reproducible command-line examples for the strict backend API.
- It demonstrates practical payload shapes for route, Pareto, stream, scenario compare, batch, duty chain, departure optimize, and signature verification workflows.
- In plain English, it shows how a user or tester would actually call the system from outside the UI.
- Its content has been absorbed into the workflow appendices and endpoint discussions in this report.
- The cookbook is especially useful for explaining that the system's features are not only theoretical; they are callable in reproducible ways.
- In thesis terms, it supplies a procedural view that complements the architecture chapters.

### `docs/run-and-operations.md`

- This is the main operational runbook.
- It explains prerequisites, environment setup, the compute fallback policy, graph warmup lifecycle, diagnostics panel, startup scripts, rebuild steps, runtime output locations, low-resource testing, and docs validation.
- In plain English, it is the "how to run this system safely" document.
- Its material has been absorbed into the deployment chapter, workflow narratives, readiness discussion, diagnostics discussion, and reproducibility chapter.
- It was especially important for capturing the strict compute degrade order of stream to JSON Pareto to single route with default steps 12 to 6 to 3.
- In thesis terms, this doc is where engineering operations and modeling rigor most visibly meet.

### `docs/model-assets-and-data-sources.md`

- This document maps runtime asset families to their origins, build scripts, and strict gates.
- It is one of the most important supporting docs for a thesis because it connects data provenance to runtime behavior.
- In plain English, it explains where the backend gets its graph, toll, departure, scenario, stochastic, terrain, fuel, carbon, and vehicle data.
- Its contents have been absorbed into the data-and-calibration chapter, live-source matrix, and file-inventory sections of this report.
- The document was especially valuable for distinguishing generated runtime assets from raw evidence families.
- In thesis terms, it provides the backbone for the project's data-governance story.

### `docs/quality-gates-and-benchmarks.md`

- This document describes the quality-gate sequence and benchmark expectations.
- It distinguishes subsystem quality scoring from runtime benchmarking and CI lanes.
- In plain English, it explains how the project checks whether its models and runtime are good enough, not merely whether the code runs.
- Its content has been absorbed into the quality, testing, and reproducibility chapter and the benchmark-evidence appendix.
- It is especially important because it separates strong local evidence from unsupported universal superiority claims.
- In thesis terms, it supports a more defensible evaluation narrative.

### `docs/reproducibility-capsule.md`

- This document explains how to produce deterministic or controlled runs and what artifacts should be archived.
- It lists commands, seeds, manifest files, signature files, and comparison checklists.
- In plain English, it is the short guide for proving that a result can be recreated.
- Its content has been absorbed into the reproducibility chapter, run-artifact workflows, and signature discussions in this report.
- It is especially important because it pushes the project beyond a one-off demo and toward research-grade rerun discipline.
- In thesis terms, it strengthens any claim that the system supports repeatable experiments.

### `docs/sample-manifest.md`

- This document shows where manifests and related outputs are written and how to retrieve them.
- It provides example artifact paths and retrieval flow.
- In plain English, it explains what a successful run leaves behind.
- Its content has been absorbed into the artifact-format appendix and the run-inspector workflow narrative.
- The document is useful because reproducibility is easiest to explain when concrete files are named.
- In thesis terms, it supports the evidence-chain discussion around manifests, signatures, and provenance.

### `docs/strict-errors-reference.md`

- This document defines the canonical strict error format and the major reason-code families.
- It is critical because strict/live behavior is central to the project's design philosophy.
- In plain English, it explains how the system fails when it refuses to guess.
- Its material has been absorbed into the strict/live chapter, reason-code appendix, and workflow narratives.
- The document is especially valuable because it standardizes how errors are interpreted across route, batch, and stream flows.
- In thesis terms, it is a core governance reference rather than mere troubleshooting material.

### `docs/performance-profiling-notes.md`

- This document focuses on performance budget checks and careful profiling practice.
- It explains benchmark entry points, warm-cache expectations, terrain live-fetch tuning, and safe profiling sequence.
- In plain English, it shows how to measure the system without contaminating the measurements.
- Its material has been absorbed into the quality and reproducibility chapter, especially the parts dealing with benchmark evidence and cache interpretation.
- The document is useful because it highlights that performance claims require controlled conditions.
- In thesis terms, it supports disciplined interpretation of runtime numbers.

### `docs/frontend-dev-tools.md`

- This document describes the frontend dev-tool surfaces such as ops diagnostics, custom vehicle manager, batch runner, run inspector, and signature verifier.
- In plain English, it explains the extra tools available beyond the main route page.
- Its content has been absorbed into the frontend capability chapter and the detailed frontend appendix.
- The document matters because these dev tools are not peripheral; they expose observability and reproducibility features central to the thesis.
- It also clarifies which backend endpoints those tools depend on.
- In thesis terms, it helps position the frontend as an investigative workbench.

### `docs/frontend-accessibility-i18n.md`

- This document records accessibility and internationalization behaviors such as English and Spanish locales, skip links, live regions, and keyboard-aware interactions.
- In plain English, it explains how the analytical interface stays usable for more than one audience or interaction style.
- Its material has been absorbed into the frontend workflow appendix and the frontend capability chapter.
- The document matters because long-running asynchronous compute is especially prone to accessibility failure if status updates are not announced properly.
- It also shows that the project considered operator usability, not only model sophistication.
- In thesis terms, it supports the claim that the UI is designed for real use, not just screenshots.

### `docs/map-overlays-tooltips.md`

- This document describes overlay types and tooltip semantics for stops, incidents, and segment details.
- In plain English, it explains what the map is trying to show beyond raw route geometry.
- Its content has been absorbed into the map overlay workflow narrative and the frontend appendix.
- It matters because route explanation in this system is spatial as well as numeric.
- The document is especially helpful for clarifying which metrics appear in tooltips and why.
- In thesis terms, it supports explainability of localized route effects.

### `docs/tutorial-and-reporting.md`

- This document links tutorial flows with report and artifact APIs.
- It covers scenario compare, departure optimization, duty chain, experiment workflows, and run-artifact retrieval.
- In plain English, it shows how user education and reporting fit into the same product surface.
- Its material has been absorbed into the tutorial workflow narrative, reporting discussions, and artifact chapters of this report.
- The document matters because it makes clear that tutorial mode is aligned with real workflows rather than being generic decoration.
- In thesis terms, it helps connect pedagogy, usability, and reproducibility.

### `docs/synthetic-incidents-weather.md`

- This document explains the weather block and incident simulation fields.
- It clarifies what a user can specify and how deterministic seeding interacts with those scenario inputs.
- In plain English, it describes how controlled weather and incident experiments are represented.
- Its content has been absorbed into the physics chapter, incident-simulation section, and workflow discussions.
- It matters because incident simulation is present but carefully scoped away from strict live truth.
- In thesis terms, it supports a nuanced distinction between controlled experimentation and live operational modeling.

### `docs/co2e-validation.md`

- This document focuses on validating CO2e-related outputs and strict carbon failures.
- It names targeted checks and carbon-source dependencies.
- In plain English, it explains how the emissions side of the model should be sanity-checked.
- Its material has been absorbed into the physics and cost-model chapter and the strict/live data chapter.
- It matters because emissions outputs are easy to overstate if their provenance and validation are not described.
- In thesis terms, this doc supports a more careful environmental-accounting argument.

### `docs/eta-concept-drift.md`

- This document explains drift checking for ETA behavior.
- It names inputs, outputs, metrics, and alerts for detecting changing ETA concept alignment.
- In plain English, it is about watching whether the meaning of the ETA model is moving over time.
- Its content has been absorbed into the quality and calibration chapters and the script appendix discussion of drift tooling.
- It matters because route-quality discussion should not assume timeless calibration.
- In thesis terms, it supports the argument that model maintenance risk was anticipated.

### `docs/dissertation-math-overview.md`

- This document gives a compact math-centered explanation of route scoring, Pareto selection, and robust utility.
- It is especially useful because it uses a dissertation-facing tone already.
- In plain English, it explains the objective family and the uncertainty-aware summary metrics.
- Its material has been absorbed into the math chapter and the later formula appendices of this report.
- The document was especially helpful for clearly stating invariants such as q50 less than or equal to q90 less than or equal to q95 less than or equal to CVaR95.
- In thesis terms, it is a direct precursor to the master report's math framing.

### `docs/math-appendix.md`

- This document provides additional mathematical notes and implementation-oriented statements about objectives and strict frontier filtering.
- In plain English, it explains the route-objective math at a level between the code and a formal derivation.
- Its content has been absorbed into the math chapter and the academic-versus-engineering discussion.
- It matters because it clarifies that Pareto filtering happens before downstream selection.
- That distinction is central to understanding why the single highlighted route is not just the shortest path.
- In thesis terms, it provides useful intermediate explanation between source code and prose report.

### `docs/appendix-graph-theory-notes.md`

- This document explains the graph-theoretic perspective behind candidate generation and frontier quality.
- It emphasizes that frontier quality is bounded by candidate-space coverage.
- In plain English, it says that you cannot get a good trade-off set if you never generated meaningful path alternatives.
- Its material has been absorbed into the routing mechanics chapter and the candidate-generation discussion.
- It matters because it supports the project's choice to invest heavily in graph-led path diversity.
- In thesis terms, it helps justify the hybrid routing design.

### Relationship Between Documents And Code

- The supporting docs are not treated as higher authority than code.
- Where code and docs could diverge, this thesis report follows the runtime code and notes the docs as secondary evidence.
- That hierarchy matters because research repositories often accumulate stale docs.
- In this project, the docs are still highly useful because many of them encode operational discipline, benchmark framing, and artifact workflow details that are harder to infer from code alone.
- The main thesis report now absorbs that material so the dissertation can stand on one uploaded document.
- The supporting docs remain valuable inside the repo for maintenance, onboarding, and targeted operational reference.

## Appendix R: Detailed Test Evidence Ledger

This appendix turns the test suite from a list of filenames into an evidence map.

The project has a large number of backend tests, and that matters for a thesis because code intent is not only described in comments and docs.

It is also locked in behaviorally by tests.

When this report says a strict failure is canonical, a frontier rule exists, or a workflow writes artifacts, that confidence is stronger when a corresponding test file exists.

### API, Packaging, And Runtime-Surface Tests

### `backend/tests/test_app_package_and_errors.py`

- This file locks the application package structure and error behavior at a broad level.
- It is useful because packaging errors or inconsistent exception surfacing can undermine the whole API surface.
- In plain English, it helps make sure the app imports and fails in expected ways.
- In thesis terms, it supports baseline runtime integrity rather than a specific modeling layer.

### `backend/tests/test_app_smoke_all.py`

- This file is a broad smoke test across the application surface.
- It matters because complex systems can break in many shallow ways before deeper algorithmic tests even run.
- In plain English, it asks whether the main system surface still basically works.
- In thesis terms, it supports confidence that the repo is operable end to end.

### `backend/tests/test_app_test_matrix.py`

- This file likely exercises multiple test-mode combinations or runtime matrices.
- It matters because strict bypass, live behavior, and configuration combinations can create edge-case regressions.
- In plain English, it helps make sure the application behaves across more than one test setting.
- In thesis terms, it supports configuration-surface credibility.

### `backend/tests/test_api_streaming.py`

- This file locks streaming API behavior.
- That is important because streamed Pareto output has a different contract from JSON responses.
- In plain English, it helps make sure stream events arrive in the expected shape and order.
- In thesis terms, it strengthens confidence in one of the project's most distinctive interactive features.

### `backend/tests/test_metrics.py`

- This file checks metrics-related behavior.
- Metrics matter because observability is a named part of the repo's design.
- In plain English, it helps verify that telemetry endpoints or metric recording behave sensibly.
- In thesis terms, it supports claims about monitoring and runtime visibility.

### `backend/tests/test_route_baseline_api.py`

- This file locks the route-baseline endpoints.
- That is crucial because smart-versus-baseline comparison is one of the repository's core evaluation surfaces.
- In plain English, it helps make sure baseline routes can still be requested and interpreted correctly.
- In thesis terms, it supports the fairness and availability of the comparison layer.

### `backend/tests/test_signatures_api.py`

- This file tests the signature-verification API surface.
- It matters because integrity checking should not only exist in helper code; it must also work over the actual endpoint contract.
- In plain English, it helps ensure manifest-signature workflows remain usable.
- In thesis terms, it supports the reproducibility and provenance story.

### `backend/tests/test_experiments.py`

- This file locks experiment-store behavior.
- That matters because saved analytical workflows are a key productivity and reproducibility feature.
- In plain English, it helps ensure experiments can be stored, loaded, filtered, or compared without drift.
- In thesis terms, it supports the claim that the application can sustain longitudinal analytical work.

### `backend/tests/test_batch_flow_integration.py`

- This file covers batch-flow integration.
- Batch behavior is important because the project is meant to support study-scale work, not only one route at a time.
- In plain English, it checks that multi-pair workflows actually complete coherently.
- In thesis terms, it supports benchmark, sensitivity, and corpus-style analysis claims.

### `backend/tests/test_batch_import_csv.py`

- This file tests CSV-driven batch import behavior.
- It matters because tabular input is a realistic operator and research workflow.
- In plain English, it helps ensure the system can turn CSV-like input into batch route requests correctly.
- In thesis terms, it supports reproducible case-study ingestion from spreadsheets or saved data.

### Graph And Candidate-Generation Tests

### `backend/tests/test_k_shortest.py`

- This file locks K-shortest path behavior.
- That is highly relevant because the candidate-generation layer depends on a Yen-style path enumeration family.
- In plain English, it checks that the backend can enumerate multiple path alternatives rather than collapsing to one route.
- In thesis terms, it supports the graph-diversity argument behind frontier quality.

### `backend/tests/test_routing_graph_loader.py`

- This file checks graph loading behavior.
- It matters because the graph asset is large and structurally important.
- In plain English, it helps ensure the graph can be parsed and mounted into runtime structures.
- In thesis terms, it supports the operational viability of the custom graph layer.

### `backend/tests/test_routing_graph_streaming_parse.py`

- This file likely checks memory-aware or streaming parse behavior for graph loading.
- That matters because a UK-scale graph can be too large for naive ingestion patterns.
- In plain English, it helps ensure the graph loader handles large assets in a controlled way.
- In thesis terms, it supports the engineering practicality of the graph design.

### `backend/tests/test_routing_graph_feasibility.py`

- This file locks OD feasibility logic.
- It matters because strict routing does not accept every input pair as valid just because coordinates were supplied.
- In plain English, it checks whether the system can determine if a start and end are connected well enough inside the graph.
- In thesis terms, it supports the fail-closed graph-readiness story.

### `backend/tests/test_routing_graph_adaptive_hops.py`

- This file checks adaptive-hop behavior.
- Adaptive hops matter because long trips should not be artificially constrained by the same hop budget as short trips.
- In plain English, it helps ensure graph search budgets scale with corridor length in the intended way.
- In thesis terms, it supports the practicality of using graph-led search over UK distances.

### `backend/tests/test_route_graph_no_path_mapping.py`

- This file checks how graph no-path outcomes are mapped into exposed failures.
- It matters because "no route" is not one undifferentiated condition in this project.
- In plain English, it helps ensure disconnected, fragmented, or coverage-gap situations are labeled clearly.
- In thesis terms, it supports the reason-code rigor of the graph layer.

### `backend/tests/test_route_graph_precheck_timeout.py`

- This file checks precheck timeout behavior for graph work.
- Timeouts matter because strict runtime is budgeted and must fail coherently under heavy conditions.
- In plain English, it helps ensure graph prechecks do not hang or fail opaquely.
- In thesis terms, it supports the engineering discipline around bounded compute.

### `backend/tests/test_route_graph_reliability_budget_and_rescue.py`

- This file is especially important because it covers reliability budgets and rescue logic.
- That maps directly onto long-corridor and state-space rescue behavior discussed earlier in the report.
- In plain English, it helps ensure the backend can widen or alter search strategy when the first graph attempt is too brittle.
- In thesis terms, it supports one of the project's most practical modifications of academic path search.

### `backend/tests/test_route_options_prefetch_gate.py`

- This file checks route-option prefetch gating.
- It matters because live-source refresh and strict gates occur before options are finalized.
- In plain English, it helps ensure option construction does not proceed when required upstream context has not been secured.
- In thesis terms, it supports the strict/live sequencing story.

### `backend/tests/test_route_cache.py`

- This file locks route-cache behavior.
- Caching matters for runtime performance and for interpreting repeated latency measurements.
- In plain English, it helps ensure route reuse works without corrupting semantics.
- In thesis terms, it supports the profiling and reproducibility discussion around warm versus cold behavior.

### Pareto And Selection Tests

### `backend/tests/test_pareto.py`

- This file covers core Pareto behavior.
- It is fundamental because the project's multi-objective identity depends on correct dominance logic.
- In plain English, it helps ensure dominated routes are not incorrectly kept and valid trade-off routes are not incorrectly removed.
- In thesis terms, it supports the mathematical legitimacy of the frontier layer.

### `backend/tests/test_pareto_epsilon_knee.py`

- This file checks epsilon-constraint and knee-related behavior.
- That matters because the project does more than plain dominance filtering.
- In plain English, it helps ensure epsilon feasibility and compromise-shape logic behave as intended.
- In thesis terms, it supports the richer frontier-curation story described in the math chapter.

### `backend/tests/test_pareto_strict_frontier.py`

- This file likely checks the strict frontier mode specifically.
- It matters because the repo distinguishes between strict non-dominated filtering and later backfill or selection stages.
- In plain English, it helps ensure the system can preserve a truly strict frontier when configured to do so.
- In thesis terms, it supports careful separation between academic filtering and engineering presentation logic.

### `backend/tests/test_pareto_backfill.py`

- This file checks route backfill behavior.
- Backfill matters because an extremely small frontier can be unhelpful for UI and decision support.
- In plain English, it helps ensure the system can add ranked alternatives when the true frontier would otherwise be too sparse.
- In thesis terms, it supports the project's pragmatic extension beyond pure frontier export.

### `backend/tests/test_path_selection_logic.py`

- This file tests the final route-selection logic.
- That is important because the thesis distinguishes clearly between frontier generation and representative-route selection.
- In plain English, it helps ensure the highlighted route is chosen according to the configured scalar profile.
- In thesis terms, it supports the academic-versus-modified-selector comparison.

### `backend/tests/test_weights.py`

- This file likely checks weight handling and normalization.
- Weight normalization matters because time, money, and emissions preferences must be made comparable.
- In plain English, it helps ensure user-provided weights are interpreted consistently.
- In thesis terms, it supports both UI preference handling and selection-math correctness.

### `backend/tests/test_models_weight_aliases.py`

- This file checks weight aliases in request models.
- It matters because user-facing or legacy field names may differ while still needing to map into the same internal structure.
- In plain English, it helps ensure route requests remain user-friendly without breaking the backend contract.
- In thesis terms, it supports interface stability and operator ergonomics.

### `backend/tests/test_property_invariants.py`

- This file checks invariant properties across modeled outputs.
- Invariant testing is important because complex systems can drift numerically without outright crashing.
- In plain English, it helps ensure certain relationships, bounds, or monotonic conditions continue to hold.
- In thesis terms, it strengthens confidence in the internal consistency of the model family.

### Scenario, Departure, And Duty Tests

### `backend/tests/test_scenario_compare.py`

- This file locks the scenario compare workflow.
- It matters because the project's sharing-mode analysis is one of its distinct features.
- In plain English, it helps ensure the system can compare no-sharing, partial-sharing, and full-sharing outputs coherently.
- In thesis terms, it supports the policy-sensitive scenario analysis layer.

### `backend/tests/test_scenario_profile_strict_thresholds.py`

- This file tests strict thresholds around scenario profiles.
- That matters because scenario data must meet freshness, coverage, and validation expectations under strict mode.
- In plain English, it helps ensure stale or incomplete scenario context is rejected properly.
- In thesis terms, it supports the fail-closed live-governance argument.

### `backend/tests/test_scenario_context_probe_timeout.py`

- This file checks scenario context probing under timeout conditions.
- It matters because context gathering is a precondition for some scenario-aware behavior.
- In plain English, it helps ensure context probing is bounded and fails clearly when too slow.
- In thesis terms, it supports operational realism around live-context lookups.

### `backend/tests/test_departure_optimize.py`

- This file locks the departure optimization endpoint.
- It matters because the endpoint is more complex than a single route request.
- In plain English, it helps ensure the system can sample candidate departures and rank them coherently.
- In thesis terms, it supports the scheduling-extension of the router.

### `backend/tests/test_departure_profile_v2.py`

- This file tests the departure profile asset behavior.
- It matters because departure-time uplift is a calibrated subsystem rather than a hard-coded rush-hour toggle.
- In plain English, it helps ensure departure multipliers behave the way the runtime expects.
- In thesis terms, it supports the data-backed departure-model claim.

### `backend/tests/test_duty_chain.py`

- This file locks the duty-chain workflow.
- It matters because chaining stops introduces different aggregation and sequencing logic from single-route analysis.
- In plain English, it helps ensure multi-stop planning produces coherent leg composition.
- In thesis terms, it supports the claim that the project extends into operational sequencing.

### `backend/tests/test_traffic_profiles.py`

- This file checks traffic-profile behavior.
- It matters because traffic interpretation feeds scenario and departure realism.
- In plain English, it helps ensure traffic-driven timing effects remain coherent.
- In thesis terms, it supports the contextual-calibration story.

### `backend/tests/test_live_bank_holidays_strict.py`

- This file checks strict handling of live bank-holiday inputs.
- Bank holidays matter because they alter UK traffic and departure context.
- In plain English, it helps ensure holiday-aware routing context is treated as governed input, not trivia.
- In thesis terms, it supports the UK-specific realism claim.

### `backend/tests/test_live_retry_policy.py`

- This file locks retry behavior for live-source access.
- It matters because strict runtime depends heavily on controlled retry semantics.
- In plain English, it helps ensure transient source errors are retried in the intended bounded way.
- In thesis terms, it supports live-data engineering rigor.

### `backend/tests/test_live_call_trace_rollup.py`

- This file checks live-call trace aggregation.
- It matters because one of the repo's differentiators is the ability to inspect what live calls actually happened.
- In plain English, it helps ensure those diagnostics remain meaningful and internally consistent.
- In thesis terms, it supports the transparency layer around upstream dependencies.

### `backend/tests/test_rbac_logging_live_sources.py`

- This file likely covers the no-op RBAC shim, structured logging, and live-source behavior together.
- It matters because peripheral infrastructure can still affect observability and runtime control.
- In plain English, it helps ensure admin-ish scaffolding and logging do not regress silently.
- In thesis terms, it supports operational robustness around the main model.

### `backend/tests/test_dev_preflight.py`

- This file checks development preflight behavior.
- That matters because startup safety and strict readiness are documented as operational requirements.
- In plain English, it helps ensure the repo catches bad live/runtime state before a user starts routing.
- In thesis terms, it supports the fail-closed startup story.

### Uncertainty And Risk Tests

### `backend/tests/test_stochastic_uncertainty.py`

- This file locks the stochastic uncertainty layer.
- It matters because the uncertainty model is one of the richest mathematical subsystems in the repo.
- In plain English, it helps ensure calibrated randomness, tail summaries, and route-risk fields remain coherent.
- In thesis terms, it supports claims about q95, CVaR, and robust-score output.

### `backend/tests/test_uncertainty_model_unit.py`

- This file checks the uncertainty model more locally at unit level.
- It matters because high-level integration tests alone can miss subtle statistical regressions.
- In plain English, it helps ensure regime resolution, seeding, and summary math behave as intended.
- In thesis terms, it supports the mathematical credibility of the risk layer.

### `backend/tests/test_robust_mode.py`

- This file checks robust-mode behavior specifically.
- It matters because robust routing is supposed to care about tail risk, not only mean performance.
- In plain English, it helps ensure routes with better averages but worse bad tails do not automatically win.
- In thesis terms, it supports the conceptual distinction between deterministic and robust recommendation.

### `backend/tests/test_emissions_context.py`

- This file checks emissions logic in contextual settings.
- It matters because emissions are affected by more than pure distance.
- In plain English, it helps ensure route context changes environmental outputs in the intended way.
- In thesis terms, it supports environmental realism.

### `backend/tests/test_emissions_models.py`

- This file checks emissions-model calculations more directly.
- It matters because CO2-related outputs are central objective fields in the router.
- In plain English, it helps ensure the emissions calculations themselves remain numerically sane.
- In thesis terms, it underpins the environmental objective dimension.

### `backend/tests/test_cost_model.py`

- This file checks monetary cost behavior.
- That matters because cost is one of the three core route objectives.
- In plain English, it helps ensure the system's money outputs remain coherent across modeled factors.
- In thesis terms, it supports the economic objective dimension.

### `backend/tests/test_counterfactuals.py`

- This file locks counterfactual output behavior.
- Counterfactuals matter because the project explains why routes differ, not just which route wins.
- In plain English, it helps ensure explanatory alternative metrics are produced consistently.
- In thesis terms, it strengthens the system's explainability claim.

### Terrain And Physics Tests

### `backend/tests/test_terrain_physics_unit.py`

- This file checks the terrain physics helper functions at unit level.
- It matters because rolling resistance, drag, grade effect, and related terms should behave predictably.
- In plain English, it helps ensure the terrain multipliers come from stable physical approximations.
- In thesis terms, it supports the physical-model component of the router.

### `backend/tests/test_terrain_physics_uplift.py`

- This file checks terrain uplift behavior.
- It matters because uphill and downhill effects should change route cost and emissions in plausible ways.
- In plain English, it helps ensure terrain actually moves route metrics in the intended direction.
- In thesis terms, it supports the claim that terrain meaningfully influences routing output.

### `backend/tests/test_terrain_segment_grades.py`

- This file checks grade extraction at segment level.
- It matters because terrain-aware modeling depends on local slope information rather than one route-wide guess.
- In plain English, it helps ensure grade values attached to route pieces are sensible.
- In thesis terms, it supports segment-level physical explainability.

### `backend/tests/test_terrain_dem_coverage.py`

- This file checks DEM coverage behavior.
- It matters because UK strict terrain logic is allowed to fail when coverage is insufficient.
- In plain English, it helps ensure the system knows when it has enough elevation support to trust terrain adjustments.
- In thesis terms, it supports fail-closed physical realism.

### `backend/tests/test_terrain_dem_index_live_sampling.py`

- This file checks terrain index behavior with live sampling.
- It matters because live tile lookup and cached terrain manifests are operationally complex.
- In plain English, it helps ensure the system can find and use elevation tiles correctly.
- In thesis terms, it supports the live-terrain engineering story.

### `backend/tests/test_terrain_fail_closed_uk.py`

- This file checks that terrain truly fails closed inside the UK when required conditions are not met.
- That matters because the report repeatedly claims terrain is governed strictly.
- In plain English, it helps ensure the backend does not quietly invent slope behavior when it lacks enough evidence.
- In thesis terms, it is one of the strongest supporting tests for the strict terrain claim.

### `backend/tests/test_terrain_non_uk_fallback.py`

- This file checks terrain behavior outside the main UK operating area.
- It matters because the project is UK-focused and should not overclaim non-UK reliability.
- In plain English, it helps ensure out-of-scope geography is handled explicitly rather than silently treated as in-scope.
- In thesis terms, it supports honest scope boundaries.

### `backend/tests/test_terrain_runtime_budget.py`

- This file checks runtime-budget behavior for terrain processing.
- Terrain can be computationally expensive if sampled too finely.
- In plain English, it helps ensure terrain processing stays bounded.
- In thesis terms, it supports the realism-versus-performance tradeoff discussion.

### `backend/tests/test_stores_and_terrain_index_unit.py`

- This file likely checks persistence helpers and terrain index utilities together.
- It matters because run stores and terrain indexes are both artifact-heavy support layers.
- In plain English, it helps ensure supporting infrastructure around terrain and storage behaves consistently.
- In thesis terms, it supports operational reliability around data-heavy subsystems.

### Toll, Vehicle, And Weather Tests

### `backend/tests/test_toll_engine_unit.py`

- This file checks toll-engine behavior directly.
- It matters because toll handling combines topology, tariffs, vehicle logic, and confidence calibration.
- In plain English, it helps ensure toll costs are attached correctly to routes.
- In thesis terms, it supports the realism of the monetary objective.

### `backend/tests/test_vehicle_custom.py`

- This file checks custom-vehicle behavior.
- It matters because custom vehicle profiles alter the physical and cost model.
- In plain English, it helps ensure user-defined fleet parameters are validated and used correctly.
- In thesis terms, it supports the configurable-vehicle claim.

### `backend/tests/test_weather_adapter.py`

- This file checks weather-profile handling.
- Weather affects duration pressure and incident multipliers.
- In plain English, it helps ensure weather scenarios map into route effects consistently.
- In thesis terms, it supports the external-context realism layer.

### Artifact, Provenance, And Reporting Tests

### `backend/tests/test_run_store_reporting.py`

- This file checks run storage and reporting behavior.
- It matters because successful analytical runs should become recoverable evidence objects.
- In plain English, it helps ensure manifests, artifacts, and reports are written coherently.
- In thesis terms, it underpins reproducibility.

### `backend/tests/test_oracle_quality.py`

- This file checks the oracle-quality store and dashboard behavior.
- It matters because the repository distinguishes source-quality monitoring from route-quality scoring.
- In plain English, it helps ensure upstream-feed health can be recorded and summarized reliably.
- In thesis terms, it supports the live-governance subsystem.

### `backend/tests/test_publish_live_artifacts.py`

- This file checks artifact publication behavior.
- It matters because generated calibration outputs must be promoted into the strict runtime asset set correctly.
- In plain English, it helps ensure the build-to-runtime handoff works.
- In thesis terms, it supports the asset lifecycle story.

### `backend/tests/test_tooling_scripts.py`

- This file checks tooling script behavior more generally.
- It matters because scripts are a major part of the repository's operational and calibration surface.
- In plain English, it helps ensure command-line support tools do what the docs say they do.
- In thesis terms, it supports reproducibility and maintainability claims.

### `backend/tests/test_scripts_builders_extended.py`

- This file extends script coverage for asset builders.
- It matters because builder regressions can quietly invalidate a runtime without breaking API syntax.
- In plain English, it helps ensure the build pipeline for model assets stays consistent.
- In thesis terms, it supports data-pipeline trust.

### `backend/tests/test_scripts_fetchers_extended.py`

- This file extends script coverage for fetchers.
- It matters because live and raw data collection are central to the repo's calibration pipeline.
- In plain English, it helps ensure upstream collection scripts still gather what later builders expect.
- In thesis terms, it supports source-provenance continuity.

### `backend/tests/test_scripts_quality_extended.py`

- This file extends script coverage for quality tooling.
- It matters because the project's quality-gate story depends on these scripts actually working.
- In plain English, it helps ensure quality scoring and related validations remain trustworthy.
- In thesis terms, it reinforces the evaluation layer.

### `backend/tests/test_scripts_smoke_all.py`

- This file is a broad smoke test across scripts.
- It matters because the repo relies heavily on scripted operations outside the runtime server.
- In plain English, it helps ensure the toolkit layer is not silently decaying.
- In thesis terms, it supports operational completeness.

### `backend/tests/test_scripts_test_matrix.py`

- This file likely checks scripts across multiple config combinations.
- It matters because scripts often behave differently under strict, bypass, or lightweight test settings.
- In plain English, it helps ensure the script surface is robust under varied modes.
- In thesis terms, it supports confidence in the repo's non-runtime tooling ecosystem.

### `backend/tests/test_strict_reason_code_contract.py`

- This file locks the canonical strict reason-code contract.
- It is one of the most thesis-important tests in the suite.
- In plain English, it helps ensure strict failures keep the same language and structure over time.
- In thesis terms, it underpins every discussion in this report about fail-closed semantics.

### Additional Integration And Regression Tests Worth Naming Explicitly

### `backend/tests/test_multileg_engine.py`

- This file checks the multileg engine directly.
- It matters because duty chains depend on stable leg aggregation behavior.
- In plain English, it helps ensure the engine that glues individual trips together works correctly.
- In thesis terms, it supports the move from route planning to duty planning.

### `backend/tests/test_route_baseline_api.py`

- This file checks baseline API behavior explicitly.
- It matters because comparison endpoints should not drift even if the smart-routing path evolves.
- In plain English, it helps ensure the control case remains available and interpretable.
- In thesis terms, it supports fair comparison infrastructure.

### `backend/tests/test_explainability_compare_integration.py`

- This file checks explainability-oriented comparison behavior in an integrated way.
- It matters because the repo claims to provide more than raw route output.
- In plain English, it helps ensure comparison and explanation surfaces line up sensibly.
- In thesis terms, it supports the explainability argument.

### `backend/tests/test_segment_breakdown.py`

- This file checks segment breakdown behavior.
- It matters because route-local explanations depend on consistent segment decomposition.
- In plain English, it helps ensure the pieces shown in the UI reflect the route model correctly.
- In thesis terms, it supports spatial explainability.

### `backend/tests/test_route_options_prefetch_gate.py`

- This file checks that route-option construction respects prefetch gating.
- It matters because the backend should not build rich outputs from incomplete live context.
- In plain English, it helps ensure the route is only assembled after required context checks pass.
- In thesis terms, it supports strict/live sequencing discipline.

### `backend/tests/test_metrics.py`

- This file checks internal metrics behavior.
- It matters because observability is treated as a first-class subsystem in the repo.
- In plain English, it helps ensure the backend's telemetry view remains coherent.
- In thesis terms, it supports runtime introspection claims.

### `backend/tests/test_route_cache.py`

- This file checks cache semantics directly.
- It matters because route caching affects performance interpretation and repeated route behavior.
- In plain English, it helps ensure cached results are reused safely.
- In thesis terms, it supports the profiling and reproducibility discussion.

### `backend/tests/test_incident_simulator.py`

- This file checks synthetic incident behavior.
- It matters because scenario experimentation should still be deterministic and controlled.
- In plain English, it helps ensure the disruption simulator behaves predictably.
- In thesis terms, it supports the experimental-analysis side of the system.

### `backend/tests/test_api_streaming.py`

- This file is worth naming again because streaming is such a distinctive interaction pattern in the project.
- It matters for event ordering, partial delivery, and fatal-stream error semantics.
- In plain English, it helps ensure streamed Pareto behaves like a real progressive workflow.
- In thesis terms, it supports asynchronous analytical transparency.

## Appendix S: Full Route-Compute Microtrace

This appendix narrates one route request from first user action to final artifact creation.

The purpose is to make the routing pipeline concrete.

Many readers understand the project more clearly when the stages are described in chronological order rather than as isolated modules.

The exact implementation has many branches, but the sequence below captures the dominant strict-runtime path.

### S1. User Input Capture In The Frontend

- The workflow begins when the user sets an origin and a destination on the map or through other UI controls.
- Optional inputs can include stops, a vehicle type, a custom vehicle, weights, a departure time, scenario mode, weather profile, incident controls, and uncertainty settings.
- At this stage the system still does not know whether a route is feasible or whether live data is fresh enough.
- The frontend is only collecting intent.
- That distinction matters because the UI does not simulate the backend locally.
- The thesis significance is that user choice is broad, but backend truth remains authoritative.

### S2. Request Construction

- The page state is transformed into a concrete request object by the request-builder helpers.
- This is where human-oriented control labels become backend field names.
- Weight sliders become numeric weights.
- Scenario-mode pickers become the literal no-sharing, partial-sharing, or full-sharing values expected by the backend.
- Departure windows become specific timestamps or optimization ranges.
- This stage is the frontend's last chance to produce a coherent, typed request before the backend validates it.

### S3. Frontend Proxy Routing

- The request is sent to a Next.js proxy route rather than directly from the browser to the backend.
- That keeps frontend networking simple and centralizes browser-facing error handling.
- It also allows NDJSON streaming to be handled in a controlled way for Pareto stream mode.
- The proxy does not perform route science itself.
- It is a transport and contract-preservation layer.
- In thesis terms, this separation helps keep browser concerns and routing-model concerns distinct.

### S4. FastAPI Validation

- Once the request reaches FastAPI, Pydantic models validate the payload.
- Required fields, literal enums, numeric ranges, and nested structures are checked here.
- This is the first hard gate in the backend.
- If validation fails, the system does not reach routing logic at all.
- That matters because many potential problems are interface errors rather than routing failures.
- In thesis terms, schema validation is part of the system's reliability story.

### S5. Runtime Readiness Check

- Before graph-led routing is attempted, the backend checks whether strict routing is ready.
- Graph warmup state matters here.
- If warmup is still running, the system can return `routing_graph_warming_up`.
- If warmup failed, the system can return `routing_graph_warmup_failed`.
- If the graph loaded but failed quality expectations, fragmentation-style errors can occur.
- This step is one of the clearest examples of fail-closed behavior in the repository.

### S6. Live-Trace Initialization

- If development diagnostics are enabled, a live-call trace record is prepared for the request.
- This trace is not the route result itself.
- It is a side-channel artifact describing expected and observed live-source activity.
- That matters because live-data dependence can otherwise be invisible.
- In thesis terms, trace initialization shows the project treats external data access as inspectable behavior.
- It turns hidden I/O into a first-class evidence stream.

### S7. Strict Live-Source Refresh And Validation

- Strict mode requires certain live-backed or published sources to be checked before route options are finalized.
- Scenario coefficients, fuel prices, carbon schedules, departure profiles, stochastic regimes, toll tables, and sometimes terrain-supporting assets are part of this universe.
- Each family has freshness, host-allow-list, or signature expectations.
- If a required family is stale or invalid, the route can fail before any path scoring happens.
- That is a deliberate design choice.
- In thesis terms, it demonstrates that the project would rather fail transparently than silently route on untrusted context.

### S8. Optional Context Probe

- The backend may run a bounded context probe depending on settings.
- The probe tries to gather quick context signals without committing to full route generation.
- This can improve later scenario or feasibility decisions.
- The probe itself is budgeted by time, state count, and path limits.
- That matters because context gathering should not become an unbounded hidden cost.
- In thesis terms, this is a practical engineering pre-step layered onto the main route computation.

### S9. Origin And Destination Snapping

- Coordinates supplied by the user must be mapped onto the routing graph and, later, onto road-realized geometry.
- Nearest-node distance thresholds are important here.
- If the graph cannot find a sufficiently close node, strict graph coverage errors can occur.
- This step sounds simple but is operationally important.
- Poor snapping logic can make valid user requests look disconnected.
- In thesis terms, snapping quality is part of geographic reliability.

### S10. OD Feasibility Screening

- After snapping, the backend evaluates whether the origin and destination look plausibly connected in the graph.
- This is not yet the full K-shortest route search.
- It is a feasibility gate.
- If connectivity fails here, the request can end with disconnected-OD or coverage-gap reason codes.
- That matters because expensive candidate generation should not start when the graph already knows the OD pair is invalid.
- In thesis terms, this is an efficiency and correctness safeguard.

### S11. Adaptive Search-Budget Construction

- The backend then computes effective hop and state budgets for the specific OD pair.
- Straight-line distance influences the effective hop ceiling.
- Safety factors and estimated edge length are used to avoid unrealistically small hop limits.
- Longer corridors can therefore receive a larger search envelope.
- This is a key practical modification relative to toy graph search.
- In thesis terms, it is an important engineering adaptation for UK-scale routing.

### S12. A* Heuristic Preparation

- If enabled, the routing graph prepares an admissible A* lower-bound heuristic toward the destination.
- The heuristic uses straight-line distance and maximum speed assumptions plus a minimum-cost-per-meter concept where available.
- The implementation notes cite Hart et al. (1968).
- This stage does not choose a path on its own.
- It accelerates pruning during search.
- In thesis terms, it is an academic search idea repurposed carefully inside a larger practical pipeline.

### S13. Yen-Style Candidate Enumeration

- The graph layer then uses Yen-style K-shortest path logic to enumerate alternative path families.
- This is where the system deliberately searches for more than one plausible corridor.
- The code cites Yen (1971).
- Runtime budgets matter heavily here because unrestricted K-shortest search on a national-scale graph can explode.
- The implementation therefore combines academic structure with bounded engineering constraints.
- In thesis terms, this is the main engine behind frontier breadth.

### S14. Long-Corridor Rescue Logic

- If early search is too brittle, long-corridor rescue logic can alter how search is attempted.
- State-space rescue, reduced-mode rescue, and long-distance bypass logic are examples.
- The point is not to guarantee every OD pair succeeds.
- The point is to avoid declaring failure too early when a wider or altered search could still work.
- This is one of the clearest places where the project differs from neat academic pseudocode.
- In thesis terms, rescue logic is a practical adaptation for hard real routes under time budgets.

### S15. Candidate Diversification And Prefilter

- The raw path list is not accepted blindly.
- Similar corridor families are filtered so the final candidate set has more structural variety.
- This matters because five tiny variants of the same motorway alignment do not give a user a useful trade-off set.
- Diversity here is a precursor to useful Pareto output.
- It is also a protection against search redundancy.
- In thesis terms, candidate quality is treated as more important than raw candidate count.

### S16. OSRM Refinement

- Once graph-led candidate corridors exist, OSRM is used to refine them into road-realized route geometry and annotations.
- This is the hybrid design in action.
- The graph provides structural diversity.
- OSRM provides road-network realization and leg detail.
- That means OSRM is still essential even though it is not trusted as the only route generator.
- In thesis terms, this is the key architectural compromise of the whole system.

### S17. Baseline Route Construction

- Baseline routes can also be constructed, either through OSRM baseline logic or ORS-related logic.
- These baselines are not identical to the smart route path.
- They exist so a reference solution can be compared against the enriched candidate pipeline.
- In plain English, the backend can compute a simpler answer for the same OD pair.
- That is crucial for evaluation.
- In thesis terms, it supports transparent comparison rather than isolated smart-routing claims.

### S18. Segment-Level Route Breakdown

- The refined route is then broken into segments for local modeling and explainability.
- This matters because terrain, tolls, incidents, and local speed assumptions are often segment-sensitive.
- A single route-wide multiplier would be cruder.
- Segment handling therefore supports more localized modeling.
- It also makes better tooltips and route inspection possible.
- In thesis terms, segmentation is the bridge from raw geometry to interpretable route mechanics.

### S19. Terrain Sampling

- If terrain-aware mode is active, the backend samples DEM support along the route.
- Coverage rules are strict inside the UK.
- If coverage falls below the configured threshold, the route can fail rather than silently guess terrain effects.
- Sample spacing depends on route length and terrain settings.
- Longer routes may use wider spacing and capped sample counts.
- In thesis terms, terrain is handled as a governed physical input, not a decorative heuristic.

### S20. Weather And Incident Effects

- Weather profiles and, in controlled scenarios, incident simulation influence route multipliers.
- Weather changes duration and incident pressure.
- Incidents can be simulated deterministically for experiments, but strict live truth paths do not silently replace live facts with synthetic incidents.
- That distinction matters a great deal.
- The project allows experimentation without confusing it with live operational truth.
- In thesis terms, this is a strong example of separating experimental knobs from strict runtime guarantees.

### S21. Toll Topology And Tariff Evaluation

- The toll engine checks whether route segments intersect tolled infrastructure.
- It then maps those intersections through topology and tariff data.
- Confidence calibration and proxy pricing logic also matter here.
- This is more sophisticated than a single yes-or-no toll flag.
- Monetary cost therefore depends on both route geometry and calibrated toll tables.
- In thesis terms, toll handling shows the broader pattern of topology plus pricing plus calibration.

### S22. Fuel And Energy Modeling

- The backend then computes route energy or fuel behavior using vehicle parameters, speed, terrain, and route decomposition.
- This is where vehicle profiles matter concretely.
- A van, rigid HGV, artic HGV, and EV HGV do not share the same physical assumptions.
- Fuel prices then translate physical consumption into monetary cost.
- The project therefore combines physics and price rather than collapsing cost into distance times a flat constant.
- In thesis terms, this is one of the richer realism layers over basic routing.

### S23. Emissions And Carbon Cost

- Emissions are derived from route and vehicle context plus fuel or energy modeling.
- Carbon schedules then convert emissions into a monetized carbon-cost dimension where applicable.
- This matters because the route output includes both direct environmental burden and a monetary environmental interpretation.
- The carbon schedule is treated as a governed input family.
- Strict failures can occur if required carbon data is unavailable.
- In thesis terms, environmental accounting is integrated into route evaluation rather than reported as an afterthought.

### S24. Departure Profile Application

- If a departure time is provided, the departure profile layer applies context-specific multipliers.
- These are UK-oriented and calibrated by region, time, and related context families.
- This means the same road geometry can score differently at different times of day.
- The departure layer is also what makes the departure optimization endpoint meaningful.
- Without it, departure optimization would be an empty loop over identical route estimates.
- In thesis terms, this is the timing-aware extension of the model.

### S25. Scenario Mode Application

- Scenario mode applies the no-sharing, partial-sharing, or full-sharing policy profile to the route context.
- This changes duration pressure, emissions multiplier, cost uplift, and uncertainty-related fields.
- Crucially, the size of the effect depends on context-conditioned profile data.
- That means full sharing is not a vague slogan.
- It is a calibrated mode-specific transformation applied inside the route build.
- In thesis terms, this is the project's policy-analysis layer.

### S26. Uncertainty Regime Resolution

- If uncertainty is active, the backend resolves a stochastic regime using corridor bucket, day kind, local slot, road mix, weather, and vehicle bucket.
- This is not a single universal noise model.
- It is a context-aware stochastic mapping backed by calibration artifacts.
- Posterior candidates are preferred where available.
- Fallback logic still exists, but strict settings can require posterior support.
- In thesis terms, this is one of the most sophisticated contextual layers in the repo.

### S27. Correlated Sampling And Tail Summaries

- Once a regime is chosen, correlated shocks are drawn using a cached Cholesky factor over a calibrated correlation matrix.
- Antithetic sampling is used to stabilize the Monte Carlo estimate.
- The backend then builds distributions for duration, money, and emissions.
- From those samples it computes q50, q90, q95, CVaR95, utility means, and robust scores.
- Sample and sigma clipping are also tracked explicitly.
- In thesis terms, this is uncertainty analysis as a quantified subsystem, not a vague confidence label.

### S28. Counterfactual Construction

- The route builder can also produce counterfactual interpretations.
- These are useful because they let the user see what would change if a factor such as terrain or incidents were absent or altered.
- Counterfactuals are explanatory rather than directly prescriptive.
- They help separate mechanism from headline score.
- This is valuable for thesis discussion because it allows local attribution rather than only end totals.
- In engineering terms, it turns a black-box score into a somewhat decomposable story.

### S29. Pareto Filtering

- Once candidate routes have full metrics, the backend applies Pareto filtering or epsilon-constraint filtering depending on request mode.
- Dominated routes are removed under the chosen objective framing.
- This is the moment where the system explicitly refuses to collapse everything to one weighted sum.
- The frontier stage is a major conceptual distinction between this project and single-objective shortest-path tooling.
- It is also where candidate diversity finally becomes visible as analytical value.
- In thesis terms, this is the core multi-objective step.

### S30. Frontier Curation And Backfill

- The system may apply additional frontier-curation logic such as crowding-based truncation or backfill.
- NSGA-II-style crowding ideas help keep spread across objective space.
- Backfill can add ranked routes when the strict frontier would otherwise be too small for useful comparison.
- This is a practical user-facing extension rather than a claim of new theory.
- It exists because a mathematically tiny frontier may be a poor product experience.
- In thesis terms, it is one of the clearest engineering compromises in the project.

### S31. Representative Route Selection

- For single-route workflows, one route is then selected from the feasible candidate set.
- This selection uses the configured math profile.
- The system may use a weighted-sum baseline, academic Tchebycheff, academic VIKOR, or modified engineering blends.
- The modified profiles add distance, ETA-distance interaction, balance, knee, and entropy terms.
- This stage is intentionally downstream of frontier formation.
- In thesis terms, it is important not to confuse selection with candidate generation.

### S32. Manifest, Provenance, And Signature Writing

- For workflows that persist results, the backend writes manifests, provenance records, signatures, and artifact files.
- This turns a route computation into an inspectable object rather than an ephemeral response.
- The written outputs can include JSON, CSV, GeoJSON, and report-friendly artifacts.
- Signatures protect integrity.
- Provenance records preserve source and artifact context.
- In thesis terms, this is where computation becomes evidence.

### S33. Frontend Rendering And Comparison

- The frontend receives the backend result and renders cards, charts, overlays, tables, and diagnostics.
- It may also request baseline comparisons, live-call traces, or run artifacts afterward.
- The user can then inspect segment breakdowns, scenario deltas, counterfactuals, or the Pareto chart.
- This means the user's analytical loop continues after the route arrives.
- The system is designed for interrogation, not just receipt.
- In thesis terms, the route result is the start of analysis, not the end.

### S34. Main Risks In This Pipeline

- A complex pipeline creates many possible failure points.
- The repository addresses this with reason codes, readiness gating, asset validation, traceability, and tests.
- Even so, graph quality, live-source freshness, calibration validity, and benchmark interpretation remain important risks.
- The system is transparent about these risks rather than pretending they do not exist.
- That honesty is methodologically valuable.
- In thesis terms, the pipeline is best described as rigorous but still bounded by input quality and engineering assumptions.

## Appendix T: Academic Methods And Engineering Modifications In Detail

This appendix restates the academic-method mapping in a slower and more explanatory way.

The repository itself is careful not to overclaim.

It names recognisable academic building blocks, cites them where implemented, and then clearly labels the modified selector profiles as engineering blends.

That is exactly the framing a thesis should preserve.

### T1. Weighted-Sum Scalarisation

- Weighted-sum selection is the simplest selector family used in the project.
- The implementation normalizes objective values within the candidate set and then computes a weighted combination of time, money, and emissions.
- The academic reference cited in code is Marler and Arora (2010).
- In plain English, the method asks for one number per route by mixing the user's preferences into one score.
- Its advantage is transparency.
- Its weakness is that it can hide compromise structure and become sensitive to scaling.

### T2. Augmented Tchebycheff Scalarisation

- The academic Tchebycheff profile uses weighted regret plus a small epsilon-like weighted-sum term.
- The repo cites Steuer and Choo (1983).
- In plain English, it pays attention to the worst normalized objective shortfall rather than only the average.
- This can be useful when one objective being very bad should matter strongly.
- The small rho term reduces tie brittleness.
- The implementation keeps this profile as an explicit academic reference baseline.

### T3. VIKOR Compromise Ranking

- Academic VIKOR combines group utility and individual regret into a compromise score.
- The repo cites Opricovic and Tzeng (2004).
- In code, weighted-sum-like utility and max-regret terms are normalized over the candidate set before being mixed by parameter `v`.
- In plain English, VIKOR asks for a route that balances collective goodness and worst-case disappointment.
- It is a natural candidate for selecting one representative route from a frontier.
- The repository preserves it both as an academic reference and as a base for a modified profile.

### T4. Distance As A Decision Signal

- Martins (1984) is cited as inspiration for using distance-oriented information in multi-objective routing choice.
- The repository adds explicit normalized distance terms to some modified selectors.
- In plain English, this says that two routes with similar objective scores may still differ in how geographically direct they are.
- Distance is not identical to duration.
- A shorter distance can imply fewer hidden complexities or a more intuitive route shape even when time is similar.
- The project uses distance not as a replacement objective but as an extra discriminating signal.

### T5. Knee-Oriented Preference

- The code cites Branke et al. (2004) for knee-oriented compromise intuition.
- A knee region is where improving one objective further starts harming others rapidly.
- In plain English, a knee-like route is often the "balanced compromise" a human decision-maker would find sensible.
- The repository uses a knee-penalty proxy based on disagreement among normalized objectives.
- This is not a full formal knee-detection algorithm.
- It is an interpretable engineering approximation.

### T6. Entropy Reward

- The code cites Shannon (1948) for an entropy-style reward idea.
- The repository computes improvement shares across time, money, and CO2, then rewards more balanced multi-objective improvement.
- In plain English, a route that improves several objectives together gets a small bonus over one that only wins on one axis.
- This is again an engineering extension.
- It is not presented as a canonical routing theorem.
- Its value is behavioral: it discourages narrow, one-axis wins from dominating the highlighted recommendation.

### T7. Modified Hybrid Selector

- The modified hybrid selector starts from weighted sum and adds regret and balance terms.
- This makes it more compromise-aware than plain weighted sum.
- In plain English, it still respects the user's preference weights, but it also dislikes routes that are very unbalanced across objectives.
- That is useful after Pareto filtering because several remaining routes can all be non-dominated yet qualitatively different.
- The selector is therefore better seen as a presentation-friendly route chooser than as a new optimization method.
- The repository is honest about that distinction.

### T8. Modified Distance-Aware Selector

- The modified distance-aware selector extends the hybrid style with distance, ETA-distance interaction, knee penalty, and entropy reward.
- In plain English, it asks for a route that is not only acceptable on average, but also operationally sensible and broadly improved.
- This profile is more product-oriented than academically pure.
- It is meant to pick a route a dispatcher might actually want to highlight.
- Its transparency is a strength.
- Its theoretical neatness is intentionally secondary.

### T9. Modified VIKOR-Distance Selector

- The default selector in local config is modified VIKOR-distance.
- It begins with academic VIKOR and then adds balance, distance, ETA-distance, knee, and entropy terms.
- In plain English, it uses a compromise-ranking skeleton but then nudges the outcome toward a more representative freight route.
- This is arguably the clearest embodiment of the repository's philosophy.
- Known academic structure is kept.
- Practical extra signals are layered on for product usefulness.

### T10. Why The Modified Profiles Are Defensible

- The modified profiles are defensible because they are transparent.
- Their extra terms are named explicitly.
- Their weights are configurable.
- The repo does not claim they are novel theory.
- Instead, it claims they are engineering blends for selecting one route from an already feasible set.
- That framing is rigorous and appropriate for a thesis.

### T11. Pareto Dominance

- The core Pareto filter uses standard minimization dominance.
- One route dominates another if it is no worse on all objectives and strictly better on at least one.
- In plain English, a dominated route has no rational claim to be preferred if all active objectives matter.
- This is the foundation of the route-set stage.
- It prevents the system from presenting obviously inferior options as meaningful alternatives.
- In thesis terms, it is the mathematically cleanest part of the multi-objective pipeline.

### T12. Epsilon-Constraint Filtering

- Epsilon-constraint mode lets the system keep routes that respect specified objective ceilings.
- This is useful when the user wants a practical feasibility envelope rather than pure dominance alone.
- In plain English, it means "show me routes that stay within these limits."
- This mode changes how the candidate set is filtered before export.
- It is especially useful for policy or compliance-like cases.
- In thesis terms, it broadens the kind of decision support the system can provide.

### T13. Crowding And Spread Preservation

- The pareto-methods module cites Deb et al. (2002) for NSGA-II crowding-distance truncation ideas.
- The repository uses spread-preservation logic to keep the exported frontier informative.
- In plain English, if there are too many non-dominated routes clustered in one tiny region, the system prefers a more spread-out subset.
- This helps charts and comparisons remain useful.
- It is not changing the dominance relation itself.
- It is curating the presentation of a rich frontier.

### T14. A* Heuristic

- The A* heuristic follows Hart et al. (1968) in spirit and citation.
- The heuristic is admissible because it is based on lower-bound travel assumptions.
- In plain English, it tells the graph search how far a node still seems from the goal in the best imaginable case.
- That allows better pruning.
- The repository uses it as an acceleration tool inside bounded search.
- It is an academic classic integrated into a practical freight-scale search setting.

### T15. Yen-Style K-Shortest Paths

- Yen's 1971 algorithm is a standard path-enumeration reference for multiple alternatives.
- The repository adapts this family to strict time and state budgets.
- In plain English, it keeps looking for next-best spur variations without pretending compute is free.
- This is essential because the project needs multiple plausible corridors, not only one shortest path.
- The result is recognisably academic in ancestry but operationally bounded in implementation.
- That is a recurring theme in the repo.

### T16. Quantiles

- Quantiles are used to summarize uncertainty samples.
- The repository computes interpolated quantiles for smoother tail behavior.
- In plain English, q50 is the median-style center, q90 and q95 describe increasingly bad tails.
- Quantiles matter because users often want understandable risk summaries rather than raw sample clouds.
- The code uses them as first-class uncertainty outputs.
- In thesis terms, they support human-readable stochastic interpretation.

### T17. CVaR

- CVaR, or conditional value at risk, is used as a tail-average risk measure.
- The implementation ensures CVaR is at least as large as the corresponding quantile threshold.
- In plain English, CVaR answers "if I am already in the bad tail, how bad is it on average?"
- This is stronger than only reporting q95.
- It is especially useful for operationally conservative decision-making.
- In thesis terms, CVaR is one of the clearest signs the uncertainty model is serious rather than cosmetic.

### T18. Robust Objective Families

- The risk model supports different robust-objective families such as CVaR excess, entropic-style penalty, and downside semivariance.
- In plain English, the system can penalize tail risk in more than one way.
- The default family is still a configured engineering choice rather than a universal theorem.
- That matters because different operators may interpret risk aversion differently.
- The code keeps this choice explicit.
- In thesis terms, it allows a more nuanced discussion of robust optimization.

### T19. Correlated Sampling And Antithetic Variates

- The uncertainty model draws correlated standard normals using a Cholesky factor.
- Antithetic pairing is then used to reduce variance in Monte Carlo estimation.
- In plain English, the system deliberately tries to make uncertainty summaries more stable for a given sample budget.
- That is an engineering-quality decision.
- It makes the stochastic layer more reproducible and less noisy.
- In thesis terms, it is a strong example of careful simulation design.

### T20. Physics-As-Model Rather Than Physics-As-Proof

- The terrain and energy modules are physically inspired engineering models.
- They use rolling resistance, aerodynamic drag, grade force, drivetrain efficiency, and regenerative recovery concepts.
- In plain English, the project uses enough vehicle physics to make terrain matter credibly.
- It does not claim to be a full vehicle-dynamics simulator.
- That limited but useful physicality is appropriate for route scoring.
- In thesis terms, this is a strong engineering compromise between realism and tractability.

## Appendix U: Detailed Variable And Calibration Catalogue

This appendix explains the runtime variable surface in more detail.

The point is not simply to list environment variable names.

The point is to explain what each family is tuning, what risk it mitigates, and what behavior it can change.

Because the project is strict, live, and graph-heavy, this configuration surface is part of the thesis story rather than an implementation footnote.

### U1. Source And Global Runtime Switches

### `REGION_PBF_URL`

- This variable points to the source OSM PBF used for regional graph preparation.
- In plain English, it identifies the base map extract from which the UK network is built.
- The variable matters because graph quality begins with source-data choice.
- In thesis terms, it marks the origin of the custom network layer.

### `LIVE_RUNTIME_DATA_ENABLED`

- This switch determines whether the runtime behaves as a live-data-aware system.
- In plain English, it turns the live-data machinery on or off.
- It matters because the repository can be run in more controlled or simplified modes for testing.
- In thesis terms, it is the top-level switch for the live-governance story.

### `STRICT_LIVE_DATA_REQUIRED`

- This variable controls whether the runtime is fail-closed on required live-backed inputs.
- In plain English, it decides whether missing or stale data should block route production.
- It is one of the defining variables of the repository's philosophy.
- In thesis terms, this is the configuration expression of "do not guess."

### U2. Route-Compute Refresh And Prefetch Controls

### `LIVE_ROUTE_COMPUTE_REFRESH_MODE`

- This variable selects how live inputs are refreshed during route compute.
- In plain English, it determines whether route requests actively refresh context or rely more on cached state.
- It matters because freshness and latency trade off against one another.
- In thesis terms, it is part of the balance between realism and responsiveness.

### `LIVE_ROUTE_COMPUTE_REQUIRE_ALL_EXPECTED`

- This variable determines whether all expected live calls must succeed for a strict route attempt.
- In plain English, it answers whether partial context is acceptable.
- Setting it strict protects completeness but raises failure sensitivity.
- In thesis terms, it tunes the completeness-versus-availability tradeoff.

### `LIVE_ROUTE_COMPUTE_FORCE_NO_CACHE_HEADERS`

- This variable controls whether route-compute live calls force cache-avoidance headers.
- In plain English, it tries to push upstream fetches toward fresher results.
- The benefit is freshness pressure.
- The risk is higher latency or rate-limit exposure.

### `LIVE_ROUTE_COMPUTE_FORCE_UNCACHED`

- This variable controls whether live-route compute should bypass caches more aggressively.
- In plain English, it is the stronger "fetch fresh, not reused" option.
- It matters because cached live context can be desirable for performance but undesirable for freshness-sensitive analysis.
- In thesis terms, it is a knob on the realism-versus-repeatability axis.

### `LIVE_ROUTE_COMPUTE_PREFETCH_TIMEOUT_MS`

- This variable sets the timeout budget for live prefetch activity during route compute.
- In plain English, it limits how long the system may wait to gather required live context.
- A low value protects latency but risks more prefetch failures.
- A high value improves completeness but can slow the user experience.

### `LIVE_ROUTE_COMPUTE_PREFETCH_MAX_CONCURRENCY`

- This variable limits how many live prefetch tasks can run at once.
- In plain English, it controls fan-out pressure against upstream sources and local resource usage.
- Higher concurrency can reduce elapsed wall-clock time.
- It can also increase rate-limit or burst-pressure risk.

### `LIVE_ROUTE_COMPUTE_PROBE_TERRAIN`

- This variable controls whether terrain is included in route-compute probing.
- In plain English, it decides whether terrain availability is checked early during route preparation.
- This matters because terrain strictness can be a route-blocking condition in UK mode.
- In thesis terms, it ties terrain into readiness rather than leaving it as a late surprise.

### U3. Cache And Retry Controls

### `LIVE_DATA_CACHE_TTL_S`

- This variable defines the time-to-live for generic live-data cache entries.
- In plain English, it determines how long fetched live context is considered reusable.
- Short TTLs improve freshness but raise call volume.
- Longer TTLs improve speed but reduce sensitivity to very recent change.

### `LIVE_SCENARIO_CACHE_TTL_SECONDS`

- This variable specializes cache lifetime for scenario-related live context.
- In plain English, it controls how long scenario inputs like traffic or related context may be reused.
- Scenario context is especially important because it influences sharing-mode pressure and related multipliers.
- In thesis terms, this variable tunes contextual freshness for one of the system's defining features.

### `ROUTE_CACHE_TTL_S`

- This variable sets the route-result cache lifetime.
- In plain English, it controls how long a previously computed route can be reused.
- It affects performance benchmarking and user-perceived responsiveness.
- In thesis terms, it matters whenever warm-cache and cold-cache results are compared.

### `LIVE_HTTP_MAX_ATTEMPTS`

- This variable caps retry count for live HTTP calls.
- In plain English, it says how stubborn the system should be before giving up on a feed.
- More attempts can improve resilience to transient failures.
- Too many attempts can increase delay and upstream load.

### `LIVE_HTTP_RETRY_DEADLINE_MS`

- This variable sets an overall deadline for HTTP retry behavior.
- In plain English, retries may happen, but not forever.
- It matters because retries without a total deadline can quietly destroy latency budgets.
- In thesis terms, it is part of bounded strictness.

### `LIVE_HTTP_RETRY_BACKOFF_BASE_MS`

- This variable sets the base backoff interval.
- In plain English, it controls the starting pause between retry attempts.
- It helps avoid hammering a failing or rate-limited source.
- This is a classic resilience tuning knob with clear operational significance.

### `LIVE_HTTP_RETRY_BACKOFF_MAX_MS`

- This variable caps the maximum backoff delay.
- In plain English, it prevents retries from sleeping for unboundedly long intervals.
- It is part of making retry behavior predictable.
- In thesis terms, it helps keep resilience bounded and measurable.

### `LIVE_HTTP_RETRY_JITTER_MS`

- This variable adds jitter to retries.
- In plain English, it helps avoid synchronized retry bursts.
- Jitter is a small operational detail with real reliability value.
- It is part of making live-source access more production-like.

### `LIVE_HTTP_RETRY_RESPECT_RETRY_AFTER`

- This variable controls whether `Retry-After` headers are obeyed.
- In plain English, it determines whether the system listens when a service asks it to slow down.
- Respecting these headers is more polite and often more stable.
- In thesis terms, it demonstrates responsible upstream interaction.

### `LIVE_HTTP_RETRYABLE_STATUS_CODES`

- This variable defines which HTTP statuses are treated as retryable.
- In plain English, it encodes what kinds of failure are considered temporary.
- That matters because not every error should trigger another attempt.
- It is part of explicit retry policy rather than ad hoc retry behavior.

### U4. Core Route Time Budgets

### `ROUTE_COMPUTE_ATTEMPT_TIMEOUT_S`

- This variable sets the total timeout for a route-compute attempt.
- In plain English, it is the ceiling for one strict attempt at route generation.
- It matters because route generation can involve live refresh, graph search, refinement, and full modeling.
- In thesis terms, it is one of the main "bounded compute" controls.

### `ROUTE_COMPUTE_SINGLE_ATTEMPT_TIMEOUT_S`

- This variable narrows timeout behavior for single-route attempts.
- In plain English, it lets focused route compute have its own budget separate from broader attempt logic.
- That matters because single-route fallback is supposed to be a quicker salvage path.
- In thesis terms, it supports controlled degradation rather than one uniform timeout for every mode.

### `ROUTE_CONTEXT_PROBE_ENABLED`

- This variable enables or disables context probing.
- In plain English, it decides whether the backend performs bounded early-context analysis.
- It matters because probes can improve context awareness but add compute.
- In thesis terms, it is a small but meaningful knob on preparatory intelligence.

### `ROUTE_CONTEXT_PROBE_TIMEOUT_MS`

- This variable limits context-probe duration.
- In plain English, it prevents context probing from becoming a hidden long-running subtask.
- It matters because probing is useful only when bounded.
- In thesis terms, it exemplifies the repo's preference for time-boxed substeps.

### `ROUTE_CONTEXT_PROBE_MAX_PATHS`

- This variable caps how many paths the context probe may examine.
- In plain English, it stops the probe from turning into a full alternative search.
- That keeps the probe lightweight.
- In thesis terms, it preserves the distinction between context gathering and route generation proper.

### `ROUTE_CONTEXT_PROBE_MAX_STATE_BUDGET`

- This variable caps the probe's search state budget.
- In plain English, it bounds how much graph-search work the probe may consume.
- It matters because state explosion can happen even in "small" helper tasks.
- In thesis terms, it reinforces disciplined computational budgeting.

### `ROUTE_CONTEXT_PROBE_MAX_HOPS`

- This variable sets a hop ceiling for the probe.
- In plain English, it limits path-length exploration during context probing.
- That matters because probe logic should remain conservative.
- It is another example of engineering guardrails around auxiliary search.

### U5. Graph Warmup Controls

### `ROUTE_GRAPH_WARMUP_ON_STARTUP`

- This variable controls whether the graph begins warming up as the backend starts.
- In plain English, it decides whether the system tries to prepare the graph before the first route request arrives.
- It matters because graph loading is expensive and should not surprise the first user.
- In thesis terms, it is part of the "warming up the graph" story requested by the user.

### `ROUTE_GRAPH_WARMUP_FAILFAST`

- This variable controls whether graph warmup errors should surface immediately as blocking conditions.
- In plain English, it answers whether the system should stop pretending route service is possible when warmup is broken.
- That matters because silent warmup failure would create misleading availability.
- In thesis terms, it supports explicit readiness semantics.

### `ROUTE_GRAPH_WARMUP_TIMEOUT_S`

- This variable sets how long warmup is allowed to continue before being treated as failed or timed out.
- In plain English, it bounds the graph-preparation window.
- A larger value is safer for huge assets but slower to fail.
- A smaller value increases responsiveness but can punish legitimate long loads.

### `ROUTE_GRAPH_FAST_STARTUP_ENABLED`

- This variable enables a faster readiness mode for startup.
- In plain English, it lets the graph become partially ready sooner under a reduced readiness interpretation.
- That can improve startup ergonomics.
- It also creates a need to distinguish fast-readiness from full-readiness carefully.

### `ROUTE_GRAPH_FAST_STARTUP_LONG_CORRIDOR_BYPASS_KM`

- This variable shapes fast-startup handling for longer corridors.
- In plain English, it says above what distance the fast-startup compromise may no longer be acceptable.
- That matters because long routes are where graph completeness matters most.
- In thesis terms, it is a speed-versus-coverage compromise knob.

### `ROUTE_GRAPH_STATUS_CHECK_TIMEOUT_MS`

- This variable limits how long graph-status checks may take.
- In plain English, even asking "are you ready?" has a budget.
- That matters because readiness checks should be cheap compared with full route compute.
- It is a small but real operational tuning control.

### U6. Graph Quality Gates

### `ROUTE_GRAPH_OD_FEASIBILITY_TIMEOUT_MS`

- This variable sets the timeout for OD feasibility checks.
- In plain English, the system has limited patience for determining whether an OD pair looks graph-connected.
- This matters because feasibility checks are useful only if they remain cheaper than full route generation.
- In thesis terms, it is another bounded-search safeguard.

### `ROUTE_GRAPH_PRECHECK_TIMEOUT_FAIL_CLOSED`

- This variable determines whether precheck timeout itself should cause strict failure.
- In plain English, it decides whether uncertainty about feasibility is treated as disqualifying.
- This is a philosophically important knob because it tunes how conservative strict mode is.
- In thesis terms, it is a direct strictness control on graph uncertainty.

### `ROUTE_GRAPH_BINARY_CACHE_ENABLED`

- This variable enables binary caching for graph data.
- In plain English, it allows preprocessed graph structures to load faster.
- That matters because graph warmup cost can otherwise dominate startup.
- In thesis terms, it is a performance optimization supporting operational practicality.

### `ROUTE_GRAPH_MIN_GIANT_COMPONENT_NODES`

- This variable defines the minimum size for the graph's giant component.
- In plain English, it protects against accepting a graph that is technically loaded but structurally too fragmented.
- This is a quality threshold, not a speed tweak.
- In thesis terms, it makes graph integrity measurable.

### `ROUTE_GRAPH_MIN_GIANT_COMPONENT_RATIO`

- This variable defines the minimum fraction of nodes that must belong to the largest component.
- In plain English, it asks whether the network is mostly one connected system rather than many disconnected islands.
- That matters because routing over a fragmented graph can be misleading.
- In thesis terms, it formalizes a graph-quality acceptance criterion.

### `ROUTE_GRAPH_MAX_NEAREST_NODE_DISTANCE_M`

- This variable caps how far a user coordinate may be from its nearest graph node.
- In plain English, it determines how far the system is willing to "snap" a request onto the network.
- Too small can reject legitimate near-network points.
- Too large can pretend bad geography is acceptable.

### U7. Search-Budget Controls

### `ROUTE_GRAPH_MAX_STATE_BUDGET`

- This variable sets the main search-state budget for graph route generation.
- In plain English, it limits how much search work a route can consume before being stopped.
- It is central to making graph-led candidate generation tractable.
- In thesis terms, it is one of the most important engineering bounds in the system.

### `ROUTE_GRAPH_STATE_BUDGET_PER_HOP`

- This variable scales allowed search effort with path length.
- In plain English, it recognizes that longer routes need more room to explore.
- This is a practical adaptation beyond textbook shortest-path descriptions.
- In thesis terms, it helps explain how the repo scales search to national distances.

### `ROUTE_GRAPH_STATE_BUDGET_RETRY_MULTIPLIER`

- This variable determines how much the state budget expands on retry.
- In plain English, it says how much more patient the system becomes after an initial search failure.
- It matters because rescue behavior should be deliberate, not arbitrary.
- In thesis terms, it is part of the controlled rescue strategy.

### `ROUTE_GRAPH_STATE_BUDGET_RETRY_CAP`

- This variable caps the expanded retry budget.
- In plain English, retries may grow the budget, but only up to a hard maximum.
- That protects the system from runaway rescue logic.
- It is another example of bounded engineering pragmatism.

### `ROUTE_GRAPH_SEARCH_INITIAL_TIMEOUT_MS`

- This variable limits the first graph search attempt.
- In plain English, it is the patience level for the initial, optimistic pass.
- A low value makes the system snappier but can trigger more retries.
- A high value reduces retries but can increase first-attempt delay.

### `ROUTE_GRAPH_SEARCH_RETRY_TIMEOUT_MS`

- This variable sets the timeout for the retry search.
- In plain English, it is the patience budget once the first search proved insufficient.
- This is part of making rescue behavior more generous than initial search.
- In thesis terms, it encodes graduated effort.

### `ROUTE_GRAPH_SEARCH_RESCUE_TIMEOUT_MS`

- This variable sets the timeout for the most explicit rescue search stage.
- In plain English, it is the budget for "last serious try" graph search.
- It matters especially on long corridors or brittle OD pairs.
- In thesis terms, it supports the narrative of structured escalation rather than binary success/fail.

### `ROUTE_GRAPH_STATE_SPACE_RESCUE_ENABLED`

- This variable turns state-space rescue on or off.
- In plain English, it decides whether the backend is allowed to escalate search strategy when normal search is too brittle.
- It matters because rescue logic is one of the system's practical differentiators.
- In thesis terms, it is a direct switch for an engineering modification beyond textbook search.

### `ROUTE_GRAPH_STATE_SPACE_RESCUE_MODE`

- This variable chooses the rescue mode, such as reduced-mode behavior.
- In plain English, it determines how the backend changes its search stance during rescue.
- The variable matters because rescue is not monolithic.
- In thesis terms, it exposes that the repo contains multiple practical fallback strategies inside graph search.

### `ROUTE_GRAPH_REDUCED_INITIAL_FOR_LONG_CORRIDOR`

- This variable allows a reduced initial mode for long-corridor cases.
- In plain English, it means the backend may use a more conservative first search on very long trips.
- This can reduce wasted work before escalation.
- In thesis terms, it shows long routes are treated as a distinct engineering regime.

### `ROUTE_GRAPH_LONG_CORRIDOR_THRESHOLD_KM`

- This variable defines when a trip counts as a long corridor.
- In plain English, it is the distance at which different graph-search tactics start to apply.
- The threshold matters because long-distance UK routing is where search complexity changes most.
- In thesis terms, it is one of the main "UK freight scale" tuning markers.

### `ROUTE_GRAPH_LONG_CORRIDOR_MAX_PATHS`

- This variable limits how many path alternatives are pursued in long-corridor mode.
- In plain English, it prevents long trips from exploding into an unmanageable number of path branches.
- This matters because route diversity is useful only while still computationally affordable.
- In thesis terms, it is a corridor-specific diversity cap.

### `ROUTE_GRAPH_SKIP_INITIAL_SEARCH_LONG_CORRIDOR`

- This variable allows the backend to skip a less useful initial search on long corridors.
- In plain English, it says some trips are so large that the first lightweight search is not worth attempting.
- That saves time in difficult cases.
- In thesis terms, it is a practical deviation from one-size-fits-all search flow.

### `ROUTE_GRAPH_SCENARIO_SEPARABILITY_FAIL`

- This variable controls whether lack of scenario separability should be treated as a failure.
- In plain English, it determines how strict the backend is when scenario-conditioned search behavior is not meaningfully distinct.
- This matters because scenario-aware routing can become less informative when all candidates collapse together.
- In thesis terms, it is a strictness knob on scenario expressiveness.

### `ROUTE_GRAPH_OD_CANDIDATE_LIMIT`

- This variable caps the OD candidate pool considered around origin and destination regions.
- In plain English, it limits how many graph anchor possibilities the search may consider.
- This matters because generous anchoring helps robustness but increases cost.
- In thesis terms, it is a tuning knob on snap robustness versus search expansion.

### `ROUTE_GRAPH_OD_CANDIDATE_MAX_RADIUS`

- This variable limits the radius within which OD graph candidates may be searched.
- In plain English, it constrains how far the system is willing to look for alternative graph anchors.
- A larger radius helps difficult edge cases but risks unrealistic snapping.
- In thesis terms, it is another geographic strictness control.

### `ROUTE_GRAPH_MAX_HOPS`

- This variable sets the base maximum hop count for graph search.
- In plain English, it is the default path-length ceiling before adaptive scaling.
- It matters because hop limits strongly shape what routes are even discoverable.
- In thesis terms, it is a core search-breadth control.

### `ROUTE_GRAPH_ADAPTIVE_HOPS_ENABLED`

- This variable turns adaptive-hop scaling on or off.
- In plain English, it decides whether trip distance changes the allowed path depth.
- This matters because fixed hop ceilings are poor fits for national-scale routing.
- In thesis terms, it is one of the clearest engineering adaptations for UK coverage.

### `ROUTE_GRAPH_HOPS_PER_KM`

- This variable scales hop allowance with straight-line distance.
- In plain English, it says how much route-depth budget should grow per kilometer.
- A higher value allows more detour complexity.
- In thesis terms, it is a practical parameter connecting geometry scale to graph search freedom.

### `ROUTE_GRAPH_HOPS_DETOUR_FACTOR`

- This variable inflates hop expectations to account for non-straight routes.
- In plain English, it recognizes that real roads do not follow a straight-line corridor perfectly.
- The variable matters because too little detour allowance can falsely rule out valid routes.
- In thesis terms, it is a realism multiplier for graph traversal.

### `ROUTE_GRAPH_EDGE_LENGTH_ESTIMATE_M`

- This variable estimates mean edge length for hop-floor computation.
- In plain English, it helps convert metric distance into a plausible minimum hop count.
- That matters because hop budgeting otherwise lacks a scale anchor.
- In thesis terms, it is a small but structurally important calibration constant.

### `ROUTE_GRAPH_HOPS_SAFETY_FACTOR`

- This variable inflates the hop floor for safety.
- In plain English, it gives the graph search extra headroom beyond the bare minimum estimate.
- That reduces brittle rejection of valid routes.
- In thesis terms, it is an explicit conservatism multiplier in search design.

### `ROUTE_GRAPH_MAX_HOPS_CAP`

- This variable caps adaptive-hop growth.
- In plain English, route depth may scale up, but not indefinitely.
- This prevents adaptive logic from becoming effectively unbounded.
- In thesis terms, it is the upper hard stop on hop liberalization.

### `ROUTE_GRAPH_A_STAR_HEURISTIC_ENABLED`

- This variable turns the A* lower-bound heuristic on or off.
- In plain English, it decides whether search is guided toward the destination by an admissible estimate.
- Heuristic guidance usually improves performance.
- In thesis terms, it switches one of the repo's named academic search ingredients.

### `ROUTE_GRAPH_HEURISTIC_MAX_SPEED_KPH`

- This variable sets the speed assumption used in the heuristic lower bound.
- In plain English, it defines how optimistic the time-based straight-line lower bound can be.
- The value matters because heuristic admissibility and pruning strength both depend on it.
- In thesis terms, it is a tuning constant inside the A* adaptation.

### `ROUTE_GRAPH_SEARCH_APPLY_SCENARIO_EDGE_COSTS`

- This variable controls whether scenario effects are injected directly into edge-search cost.
- In plain English, it decides whether scenario pressure influences path search itself or only later route modeling.
- This is a conceptually important choice.
- In thesis terms, it tunes how deeply scenario logic penetrates the routing stack.

### U8. Candidate Prefilter And Route-Option Controls

### `ROUTE_CANDIDATE_PREFILTER_MULTIPLIER`

- This variable sets how aggressively the raw candidate set is overgenerated before prefiltering.
- In plain English, it decides how much extra diversity is gathered before duplicates and near-duplicates are removed.
- A larger multiplier can improve frontier quality.
- It can also increase compute cost.

### `ROUTE_CANDIDATE_PREFILTER_MULTIPLIER_LONG`

- This variable is the long-corridor version of the same overgeneration control.
- In plain English, it recognizes that long trips need a separate diversity/computation balance.
- This matters because long-distance enumeration is more expensive.
- In thesis terms, it is another example of corridor-specific tuning.

### `ROUTE_CANDIDATE_PREFILTER_LONG_DISTANCE_THRESHOLD_KM`

- This variable says when the long-corridor prefilter multiplier starts to apply.
- In plain English, it is the cutoff where the system changes how aggressively it overgenerates alternatives.
- It matters because not every route needs the same diversity policy.
- In thesis terms, it is a scale-aware route-option tuning threshold.

### `ROUTE_OPTION_SEGMENT_CAP`

- This variable caps segment counts for ordinary route-option decomposition.
- In plain English, it limits how finely the backend will break a route into modeled pieces.
- More segments can improve fidelity.
- Fewer segments reduce compute and payload size.

### `ROUTE_OPTION_SEGMENT_CAP_LONG`

- This variable is the long-route version of the segment cap.
- In plain English, it keeps very long routes from creating huge breakdown payloads.
- That matters because segment-level explainability must remain tractable.
- In thesis terms, it is a fidelity-versus-scalability control.

### `ROUTE_OPTION_LONG_DISTANCE_THRESHOLD_KM`

- This variable defines when long-route segmentation rules should apply.
- In plain English, it marks the distance at which the backend starts simplifying route-option breakdown granularity.
- That matters for performance and artifact size.
- In thesis terms, it formalizes a distance-sensitive explainability compromise.

### `ROUTE_OPTION_REUSE_SCENARIO_POLICY`

- This variable controls reuse of scenario policy during route-option construction.
- In plain English, it decides whether scenario interpretation is recalculated or reused where appropriate.
- Reuse can improve efficiency.
- The variable matters because scenario calculation can be context-heavy.

### `ROUTE_OPTION_TOD_BUCKET_S`

- This variable sets the time-of-day bucketing resolution in seconds.
- In plain English, it determines how coarsely the route is grouped for time-sensitive modeling.
- Finer buckets can improve nuance.
- Coarser buckets reduce complexity and jitter.

### `ROUTE_OPTION_ENERGY_SPEED_BIN_KPH`

- This variable sets speed-bin granularity for energy modeling.
- In plain English, it controls how finely speed differences are represented when approximating energy use.
- Smaller bins can give more detail.
- Larger bins smooth the model and reduce surface complexity.

### `ROUTE_OPTION_ENERGY_GRADE_BIN_PCT`

- This variable sets grade-bin granularity for energy modeling.
- In plain English, it controls how finely slope differences are represented in route energy estimation.
- That matters because terrain effects can become noisy if the model is too granular.
- In thesis terms, it is part of physical-model discretization tuning.

### U9. Baseline Realism Controls

### `ROUTE_BASELINE_DURATION_MULTIPLIER`

- This variable inflates or calibrates OSRM baseline duration.
- In plain English, it adjusts the raw baseline so comparison is made against a more realistic reference rather than a naively optimistic provider number.
- This matters because unfair baselines distort evaluation.
- In thesis terms, it is a realism correction, not a claim of smart-routing superiority by fiat.

### `ROUTE_BASELINE_DISTANCE_MULTIPLIER`

- This variable adjusts OSRM baseline distance.
- In plain English, it is the distance-side counterpart to the baseline duration realism multiplier.
- It matters because simple provider routes may understate operational path distance.
- In thesis terms, it supports fairer route comparison framing.

### `ROUTE_ORS_BASELINE_DURATION_MULTIPLIER`

- This variable adjusts ORS baseline duration.
- In plain English, it plays the same realism role for the ORS-style comparison path.
- That matters because the repo treats OSRM and ORS as distinct reference families.
- In thesis terms, it helps keep cross-baseline comparisons explicit.

### `ROUTE_ORS_BASELINE_DISTANCE_MULTIPLIER`

- This variable adjusts ORS baseline distance.
- In plain English, it is the ORS-side distance realism correction.
- This matters for consistency between time and distance baseline treatment.
- In thesis terms, it is part of the "what counts as a fair baseline?" discussion.

### `ROUTE_ORS_BASELINE_ALLOW_PROXY_FALLBACK`

- This variable decides whether an ORS-like proxy fallback is allowed when true ORS is unconfigured.
- In plain English, it controls whether the system may still return an ORS comparison-style answer when direct ORS access is unavailable.
- This is important because the thesis should distinguish real ORS evidence from proxy reference behavior.
- The variable keeps that distinction configurable and explicit.

### U10. Selection-Math Controls

### `ROUTE_SELECTION_MATH_PROFILE`

- This variable chooses which scalar selection profile is used for the highlighted route.
- In plain English, it determines how one route is picked from the final feasible candidate set.
- This is one of the most thesis-important settings in the whole repo.
- It directly controls whether the system behaves like academic weighted sum, academic VIKOR, or a modified engineering selector.

### `ROUTE_SELECTION_MODIFIED_REGRET_WEIGHT`

- This variable weights the regret term in modified selectors.
- In plain English, it controls how strongly the system dislikes routes with one especially weak objective.
- Raising it makes compromise behavior more conservative.
- In thesis terms, it tunes one of the main engineering additions beyond pure weighted sum.

### `ROUTE_SELECTION_MODIFIED_BALANCE_WEIGHT`

- This variable weights balance penalty.
- In plain English, it controls how strongly the highlighted route should avoid objective imbalance.
- It matters because many non-dominated routes are acceptable but not equally "balanced."
- In thesis terms, it tunes the human-sensible compromise tendency of modified selectors.

### `ROUTE_SELECTION_MODIFIED_DISTANCE_WEIGHT`

- This variable weights normalized route distance inside modified selectors.
- In plain English, it rewards or penalizes geographic directness beyond time alone.
- It matters because freight users often care when a route looks unnecessarily long.
- In thesis terms, it is a concrete example of a non-classical engineering tie-breaker.

### `ROUTE_SELECTION_MODIFIED_ETA_DISTANCE_WEIGHT`

- This variable weights the ETA-distance interaction penalty.
- In plain English, it cares about the combination of slow and long, not just either one in isolation.
- This is a richer operational signal than pure duration or pure distance.
- In thesis terms, it shows the selector is tuned for route sensibility, not just formula purity.

### `ROUTE_SELECTION_MODIFIED_ENTROPY_WEIGHT`

- This variable controls the strength of the entropy reward.
- In plain English, it decides how much the selector should favor routes that improve several objectives together.
- Increasing it rewards broad improvement.
- In thesis terms, it tunes a deliberately interpretive engineering term.

### `ROUTE_SELECTION_MODIFIED_KNEE_WEIGHT`

- This variable controls knee-penalty strength.
- In plain English, it changes how strongly the selector prefers routes near a balanced compromise region.
- This matters because the "best-looking compromise" is not identical to minimal weighted sum.
- In thesis terms, it directly tunes the knee-oriented engineering extension.

### `ROUTE_SELECTION_TCHEBYCHEFF_RHO`

- This variable is the rho parameter for augmented Tchebycheff.
- In plain English, it is the small epsilon-like tie-break term added to regret.
- It matters because pure max-regret can be too brittle.
- In thesis terms, it preserves the standard augmented form of the academic baseline.

### `ROUTE_SELECTION_VIKOR_V`

- This variable is the `v` parameter in VIKOR.
- In plain English, it balances group utility and individual regret.
- A higher value leans toward overall utility.
- In thesis terms, it is the central tuning knob of the academic and modified VIKOR families.

### `ROUTE_PARETO_BACKFILL_ENABLED`

- This variable turns Pareto backfill on or off.
- In plain English, it decides whether extra ranked routes may be added when the strict frontier is too small.
- This matters because small frontiers can be mathematically correct but operationally unhelpful.
- In thesis terms, it switches a pragmatic product-oriented extension of frontier presentation.

### `ROUTE_PARETO_BACKFILL_MIN_ALTERNATIVES`

- This variable sets the minimum alternative count the system tries to preserve through backfill.
- In plain English, it answers how sparse is "too sparse" for the exported route set.
- This matters because route-comparison UX benefits from a non-trivial option count.
- In thesis terms, it is a direct tuning knob on frontier usability.

### U11. Development And Debug Controls

### `DEV_ROUTE_DEBUG_CONSOLE_ENABLED`

- This variable enables development-side route debug output.
- In plain English, it turns on richer console diagnostics around route requests.
- This matters when inspecting live-call behavior, attempts, and internal sequencing.
- In thesis terms, it supports development-time observability.

### `DEV_ROUTE_DEBUG_INCLUDE_SENSITIVE`

- This variable controls whether sensitive debug material is included.
- In plain English, it decides how much raw detail is visible in debug traces.
- It matters because observability and confidentiality can conflict.
- In thesis terms, it is a reminder that transparency must still be governed.

### `DEV_ROUTE_DEBUG_MAX_CALLS_PER_REQUEST`

- This variable caps how many live-call records may be stored per request trace.
- In plain English, it prevents debug tracing from exploding in memory or output size.
- This matters because detailed live-call logging can become heavy.
- In thesis terms, it is bounded observability.

### `DEV_ROUTE_DEBUG_TRACE_TTL_SECONDS`

- This variable controls how long request traces are retained.
- In plain English, it determines how long the live-call trace remains available for inspection.
- Longer retention helps debugging.
- Shorter retention limits memory and data exposure.

### `DEV_ROUTE_DEBUG_MAX_REQUEST_TRACES`

- This variable caps the number of stored request traces.
- In plain English, it prevents dev diagnostics from becoming an unbounded trace archive.
- This matters because debug features should not quietly turn into storage leaks.
- In thesis terms, it keeps observability operationally sane.

### `DEV_ROUTE_DEBUG_RETURN_RAW_PAYLOADS`

- This variable controls whether raw payload bodies are returned in debug contexts.
- In plain English, it is the switch between summarized diagnostics and raw upstream payload exposure.
- The setting matters because raw payloads are useful for debugging but heavier and more sensitive.
- In thesis terms, it highlights the project's careful handling of observability depth.

### `DEV_ROUTE_DEBUG_MAX_RAW_BODY_CHARS`

- This variable caps the size of returned raw debug payloads.
- In plain English, it prevents raw-body debugging from overwhelming the interface or logs.
- This is another bounded-observability guardrail.
- In thesis terms, it shows that even debug transparency is intentionally constrained.

### U12. Frontend And Attempt Timeout Controls

### `COMPUTE_ATTEMPT_TIMEOUT_MS`

- This variable sets the server-side compute-attempt timeout used by the frontend route handlers.
- In plain English, it defines how long a browser-facing attempt is allowed to run before the frontend degrades.
- This matters because the backend and frontend each have their own timeout perspective.
- In thesis terms, it is part of the user-experience shaping layer over strict compute.

### `COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS`

- This variable sets the server-side fallback timeout for degraded route attempts.
- In plain English, it defines the patience budget for fallback stages.
- This matters because a fallback is only helpful if it returns sooner than the failed attempt path.
- In thesis terms, it supports graceful degradation.

### `NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS`

- This variable exposes compute-attempt timeout to the browser side.
- In plain English, it lets the frontend know how long it should wait before considering an attempt timed out.
- This matters for coherent progress and fallback behavior.
- In thesis terms, it keeps the client and server aligned on compute patience.

### `NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS`

- This variable exposes fallback timeout to the frontend.
- In plain English, it tells the browser how long later-stage degraded attempts are allowed to run.
- This matters because the degrade ladder is a user-facing experience, not only a backend detail.
- In thesis terms, it supports a transparent fallback policy.

### `NEXT_PUBLIC_COMPUTE_DEGRADE_STEPS`

- This variable defines the alternative-count degrade steps, such as 12 to 6 to 3.
- In plain English, it tells the frontend how aggressively to simplify a failing request as it retries.
- This is one of the most visible performance/quality tradeoff knobs in the UI.
- In thesis terms, it turns the runbook's degrade policy into a concrete configuration artifact.

### `NEXT_PUBLIC_ROUTE_GRAPH_WARMUP_BASELINE_MS`

- This variable gives the frontend a warmup-related baseline expectation.
- In plain English, it helps the UI reason about graph warmup timing and readiness messaging.
- This matters because readiness UX benefits from having a timing frame of reference.
- In thesis terms, it connects startup engineering to user communication.

### U13. Terrain Strictness Controls

### `STRICT_RUNTIME_TEST_BYPASS`

- This variable controls whether strict runtime requirements are bypassed in controlled test contexts.
- In plain English, it allows the test suite to exercise paths that would otherwise be blocked by full strictness.
- This matters because a fully fail-closed system is harder to test comprehensively.
- In thesis terms, it is a testing accommodation, not a production philosophy change.

### `TERRAIN_DEM_FAIL_CLOSED_UK`

- This variable turns UK fail-closed terrain policy on or off.
- In plain English, it decides whether missing or insufficient terrain support should block UK route computation.
- This is one of the strongest realism controls in the repo.
- In thesis terms, it is a direct expression of physical-model strictness.

### `TERRAIN_DEM_COVERAGE_MIN_UK`

- This variable sets the minimum terrain coverage ratio required inside the UK.
- In plain English, it defines how much of the route must have valid elevation support before terrain effects are trusted.
- A higher value increases realism confidence but also route failure frequency.
- In thesis terms, it is a key threshold in the terrain-governance story.

### `TERRAIN_SAMPLE_SPACING_M`

- This variable sets default terrain sampling spacing.
- In plain English, it controls how often the route is probed for elevation information.
- Smaller spacing improves terrain fidelity.
- Larger spacing reduces cost but smooths away detail.

### `TERRAIN_LONG_ROUTE_THRESHOLD_KM`

- This variable defines when a route counts as "long" for terrain handling.
- In plain English, it marks the distance where terrain sampling rules change.
- This matters because very long routes need a different fidelity/performance balance.
- In thesis terms, it is another corridor-scale tuning threshold.

### `TERRAIN_LONG_ROUTE_SAMPLE_SPACING_M`

- This variable sets the sample spacing for long routes.
- In plain English, it allows long routes to use coarser terrain probing than short routes.
- This keeps terrain modeling computationally practical.
- In thesis terms, it is a deliberate approximation choice for scale management.

### `TERRAIN_LONG_ROUTE_MAX_SAMPLES_PER_ROUTE`

- This variable caps long-route terrain sample count.
- In plain English, it keeps large routes from producing huge terrain workloads.
- This matters because a strict terrain model can otherwise become operationally expensive.
- In thesis terms, it is a hard bound on physical-detail expansion.

### `TERRAIN_MAX_SAMPLES_PER_ROUTE`

- This variable caps total terrain samples for ordinary routes.
- In plain English, it prevents terrain probing from becoming arbitrarily dense.
- This matters for performance and memory.
- In thesis terms, it is an operational cap on physical fidelity.

### `TERRAIN_SEGMENT_BOUNDARY_PROBE_MAX_SEGMENTS`

- This variable caps how many segment boundaries are examined in terrain probing.
- In plain English, it limits how fine-grained segment-aware terrain checks can become.
- This matters because segmentation and terrain together can multiply complexity.
- In thesis terms, it is a control on explainability detail versus computation cost.

### U14. Live Terrain Controls

### `LIVE_TERRAIN_DEM_URL_TEMPLATE`

- This variable defines the live terrain tile URL pattern.
- In plain English, it tells the backend where remote elevation tiles are fetched from.
- This is critical for live or supplemental terrain coverage.
- In thesis terms, it anchors the live-terrain source family.

### `LIVE_TERRAIN_REQUIRE_URL_IN_STRICT`

- This variable decides whether a live terrain URL is mandatory under strict mode.
- In plain English, it determines whether terrain strictness permits a missing remote source configuration.
- This matters because fail-closed terrain depends on source availability.
- In thesis terms, it is a strict-governance rule for terrain sourcing.

### `LIVE_TERRAIN_ALLOW_SIGNED_FALLBACK`

- This variable decides whether a signed fallback terrain artifact may substitute for live terrain.
- In plain English, it tunes how strict terrain sourcing remains when live fetch is unavailable.
- Disallowing fallback is stricter and less forgiving.
- In thesis terms, it is a provenance-versus-availability tradeoff.

### `LIVE_TERRAIN_ALLOWED_HOSTS`

- This variable defines which terrain hosts are permitted.
- In plain English, it is a host allow-list for remote terrain data.
- This matters for source governance and safety.
- In thesis terms, it shows that live sourcing is restricted, not open-ended.

### `LIVE_TERRAIN_TILE_ZOOM`

- This variable defines the terrain tile zoom level.
- In plain English, it controls the spatial resolution of fetched terrain tiles.
- Higher zoom usually means finer spatial detail.
- In thesis terms, it influences the spatial granularity of live terrain support.

### `LIVE_TERRAIN_TILE_MAX_AGE_DAYS`

- This variable defines how old terrain tiles may be before being considered stale.
- In plain English, it sets freshness expectations for cached or fetched terrain support.
- This matters because even terrain caches are governed by age.
- In thesis terms, it contributes to the strictness of live physical context.

### `LIVE_TERRAIN_CACHE_DIR`

- This variable defines where live terrain tiles are cached locally.
- In plain English, it gives the backend a place to reuse fetched elevation support.
- This matters because repeated terrain fetches can be expensive.
- In thesis terms, it supports the performance side of live terrain.

### `LIVE_TERRAIN_CACHE_MAX_TILES`

- This variable caps how many live terrain tiles may be retained.
- In plain English, it prevents the terrain cache from growing without bound.
- This is a resource-governance setting.
- In thesis terms, it is part of operational sustainability for terrain support.

### `LIVE_TERRAIN_CACHE_MAX_MB`

- This variable caps live terrain cache size in megabytes.
- In plain English, it turns terrain reuse into a bounded storage cost.
- This matters because elevation tiles can accumulate quickly.
- In thesis terms, it keeps the live physical layer operationally manageable.

### `LIVE_TERRAIN_FETCH_RETRIES`

- This variable sets retry count for terrain tile fetches.
- In plain English, it says how many chances the backend gives remote terrain retrieval before giving up.
- This matters because live terrain is useful but should not dominate route latency.
- In thesis terms, it is bounded resilience for terrain sourcing.

### `LIVE_TERRAIN_MAX_REMOTE_TILES_PER_ROUTE`

- This variable caps how many remote terrain tiles a single route may fetch.
- In plain English, it limits how much one route can cost in remote elevation I/O.
- This is especially important on long routes.
- In thesis terms, it is a route-level budget on live physical context.

### `LIVE_TERRAIN_CIRCUIT_BREAKER_FAILURES`

- This variable sets the failure count that opens the terrain fetch circuit breaker.
- In plain English, repeated terrain-source failure can trigger a temporary stop in further fetch attempts.
- This protects the system and the upstream source.
- In thesis terms, it makes live terrain fetching more production-grade.

### `LIVE_TERRAIN_CIRCUIT_BREAKER_COOLDOWN_S`

- This variable sets the cooldown period for the terrain circuit breaker.
- In plain English, it determines how long the system waits before trying the terrain source again.
- This matters because persistent retries against a broken source are wasteful.
- In thesis terms, it is a resilience control on the live terrain path.

### `LIVE_TERRAIN_ENABLE_IN_TESTS`

- This variable decides whether live terrain fetching is allowed during tests.
- In plain English, it prevents tests from accidentally depending on remote terrain availability unless explicitly desired.
- This matters for deterministic and isolated testing.
- In thesis terms, it protects reproducibility in the test suite.

### `LIVE_TERRAIN_PREFETCH_PROBE_FRACTIONS`

- This variable defines where along the route terrain prefetch probes should be sampled.
- In plain English, it chooses representative fractions of the route for early terrain availability checks.
- This is a nuanced engineering control.
- In thesis terms, it shows how even terrain readiness is probed strategically rather than uniformly.

### `LIVE_TERRAIN_PREFETCH_MIN_COVERED_POINTS`

- This variable defines the minimum number of covered probe points needed during terrain prefetch.
- In plain English, it sets the threshold for saying "terrain seems present enough to continue."
- This matters because prefetch probing is an early confidence check.
- In thesis terms, it is a terrain-readiness gate parameter.

### U15. Live Scenario Controls

### `LIVE_SCENARIO_COEFFICIENT_URL`

- This variable points to the live or published scenario coefficient artifact.
- In plain English, it tells the backend where to fetch the current scenario profile data from.
- This is a central live-data dependency.
- In thesis terms, it is the main input location for sharing-mode realism.

### `LIVE_SCENARIO_REQUIRE_URL_IN_STRICT`

- This variable determines whether a scenario-coefficient URL is mandatory in strict mode.
- In plain English, it says strict routing cannot proceed without a configured scenario source.
- This matters because scenario logic is a core subsystem, not an optional garnish.
- In thesis terms, it is a strong strictness rule for context-driven sharing behavior.

### `LIVE_SCENARIO_ALLOW_SIGNED_FALLBACK`

- This variable controls whether signed fallback scenario artifacts are permitted.
- In plain English, it balances source freshness against fallback availability.
- Disallowing fallback is stricter and more fail-closed.
- In thesis terms, it is a direct configuration expression of scenario-data conservatism.

### `LIVE_SCENARIO_ALLOWED_HOSTS`

- This variable restricts which hosts scenario live calls may reach.
- In plain English, it is the allow-list for scenario-source domains.
- This matters for governance and source hygiene.
- In thesis terms, it makes live scenario ingestion explicitly curated.

### `LIVE_SCENARIO_WEBTRIS_NEAREST_SITES`

- This variable controls how many nearby WebTRIS sites are considered.
- In plain English, it tunes how much local traffic-sensor evidence is pulled into scenario context.
- More sites can improve robustness but also increase fetch work.
- In thesis terms, it is a detail-level knob on traffic evidence aggregation.

### `LIVE_SCENARIO_DFT_MAX_PAGES`

- This variable limits DfT pagination during scenario data collection.
- In plain English, it bounds how much paged source exploration is allowed.
- This matters because live scenario collection should remain bounded.
- In thesis terms, it is a resource control on public-source ingestion.

### `LIVE_SCENARIO_DFT_NEAREST_LIMIT`

- This variable caps how many nearby DfT stations are considered.
- In plain English, it controls the breadth of nearby official count evidence gathered.
- This affects scenario context richness and fetch cost.
- In thesis terms, it is another aggregation-breadth parameter in live scenario collection.

### `LIVE_SCENARIO_DFT_MIN_STATION_COUNT`

- This variable sets the minimum number of DfT stations required for strict scenario use.
- In plain English, it defines how much evidence is "enough" from this source family.
- This matters because sparse context should not masquerade as strong context.
- In thesis terms, it is a source-quality threshold for scenario inference.

### `LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES`

- This variable sets how old scenario coefficients may be.
- In plain English, it is the freshness window for scenario profiles.
- This is one of the most important live-governance thresholds in the repo.
- In thesis terms, it directly controls how quickly scenario knowledge becomes too stale to trust.

### `LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT`

- This variable decides whether partial live-source availability is acceptable in strict mode.
- In plain English, it controls whether the system can proceed with an incomplete scenario evidence bundle.
- This is a major completeness-versus-availability switch.
- In thesis terms, it shapes how hardline strict scenario routing really is.

### `LIVE_SCENARIO_MIN_SOURCE_COUNT_STRICT`

- This variable sets the minimum number of source families required for strict scenario context.
- In plain English, it defines how many independent context contributors must be present.
- This matters because scenario realism is meant to be multi-source.
- In thesis terms, it quantifies evidential sufficiency for live scenario use.

### `LIVE_SCENARIO_MIN_COVERAGE_OVERALL_STRICT`

- This variable sets the required overall coverage fraction for strict scenario context.
- In plain English, it defines how complete the combined scenario evidence must be.
- Local preflight evidence shows this can be set to full coverage.
- In thesis terms, it is one of the strongest expressions of fail-closed context policy.

### `SCENARIO_MIN_OBSERVED_MODE_ROW_SHARE`

- This variable sets the minimum observed row share for scenario mode evidence.
- In plain English, it controls how much direct observed support a scenario context should have before being treated as trustworthy.
- This matters because projection-heavy scenario calibration can become fragile.
- In thesis terms, it is a data-quality threshold inside scenario profile building.

### `SCENARIO_MAX_PROJECTION_DOMINANT_CONTEXT_SHARE`

- This variable limits how dominant projected rather than observed context may be.
- In plain English, it prevents a scenario context from being driven too heavily by inferred rather than observed evidence.
- This is an important calibration-quality guardrail.
- In thesis terms, it supports a more careful distinction between empirical and projected scenario behavior.

### U16. Fuel, Carbon, Departure, Stochastic, And Toll Source Controls

### `LIVE_FUEL_PRICE_URL`

- This variable points to the live or published fuel-price artifact.
- In plain English, it tells the backend where current fuel price data comes from.
- This matters because cost modeling depends on a governed price input.
- In thesis terms, it anchors the economic live-data path.

### `LIVE_FUEL_AUTH_TOKEN`

- This variable provides token-style authentication if the fuel source requires it.
- In plain English, it is one possible credential path for accessing fuel data.
- This matters because some live sources may not be openly readable.
- In thesis terms, it shows the repo anticipates governed source access.

### `LIVE_FUEL_API_KEY`

- This variable provides API-key authentication for fuel-price retrieval.
- In plain English, it is another route to authenticated fuel-source access.
- It matters because live economic feeds may require credentials.
- In thesis terms, it supports operational deployability of the fuel-data layer.

### `LIVE_FUEL_API_KEY_HEADER`

- This variable defines which header carries the fuel API key.
- In plain English, it tells the backend how to present credentials to the source.
- This is a small but necessary integration detail.
- In thesis terms, it is part of making live-source use reproducible and configurable.

### `LIVE_FUEL_REQUIRE_URL_IN_STRICT`

- This variable decides whether a fuel source URL is mandatory in strict mode.
- In plain English, strict cost realism cannot proceed without a configured fuel source.
- This matters because fuel is central to monetary cost.
- In thesis terms, it is a fail-closed rule for economic input.

### `LIVE_FUEL_ALLOW_SIGNED_FALLBACK`

- This variable controls whether signed fallback fuel data is allowed.
- In plain English, it balances freshness against fallback availability for fuel prices.
- Disallowing fallback is stricter.
- In thesis terms, it expresses conservatism in economic-source governance.

### `LIVE_FUEL_REQUIRE_SIGNATURE`

- This variable requires fuel data to carry a valid signature.
- In plain English, the backend may insist on integrity-checked fuel artifacts.
- This matters because economic inputs affect cost outputs directly.
- In thesis terms, it is one of the repo's clearest integrity controls on live-backed assets.

### `LIVE_FUEL_ALLOWED_HOSTS`

- This variable restricts which hosts may serve fuel data.
- In plain English, it is the allow-list for the fuel source.
- This matters for source hygiene and controlled provenance.
- In thesis terms, it reinforces that live fetching is curated, not unrestricted.

### `LIVE_FUEL_MAX_AGE_DAYS`

- This variable sets how old fuel data may be.
- In plain English, it defines the freshness window for fuel pricing.
- This matters because stale prices distort route cost.
- In thesis terms, it is a concrete freshness threshold on the money objective.

### `LIVE_CARBON_SCHEDULE_URL`

- This variable points to the carbon schedule artifact.
- In plain English, it tells the backend where environmental monetization policy comes from.
- This matters because carbon cost is a configured live-backed input, not a hard-coded constant.
- In thesis terms, it anchors the carbon-cost provenance path.

### `LIVE_CARBON_REQUIRE_URL_IN_STRICT`

- This variable decides whether strict mode requires a configured carbon schedule source.
- In plain English, it enforces that carbon monetization only happens with a known source.
- This matters for environmental-accounting credibility.
- In thesis terms, it is a strictness rule on carbon economics.

### `LIVE_CARBON_ALLOW_SIGNED_FALLBACK`

- This variable controls whether signed fallback carbon schedules are allowed.
- In plain English, it balances environmental-price freshness against fallback continuity.
- Disallowing fallback is the more conservative option.
- In thesis terms, it mirrors the broader strict/live governance pattern.

### `LIVE_CARBON_ALLOWED_HOSTS`

- This variable restricts allowed carbon-source hosts.
- In plain English, it governs where carbon schedule data may come from.
- This matters for provenance and deployment safety.
- In thesis terms, it keeps environmental inputs within curated domains.

### `LIVE_BANK_HOLIDAYS_URL`

- This variable points to the bank-holidays source.
- In plain English, it tells the backend where to fetch the holiday calendar that affects UK day-kind context.
- This matters because bank holidays influence departure and scenario interpretation.
- In thesis terms, it supports UK-specific contextual realism.

### `LIVE_BANK_HOLIDAYS_ALLOWED_HOSTS`

- This variable restricts allowed bank-holiday hosts.
- In plain English, it constrains the source of calendar truth.
- This matters because even seemingly simple context sources are governed explicitly.
- In thesis terms, it reinforces the system's consistent source-curation philosophy.

### `LIVE_DEPARTURE_PROFILE_URL`

- This variable points to the published departure profile artifact.
- In plain English, it tells the backend where the departure-time uplift model is fetched from.
- This matters because schedule-aware routing depends on a governed departure asset.
- In thesis terms, it anchors the live-backed departure-model path.

### `LIVE_DEPARTURE_REQUIRE_URL_IN_STRICT`

- This variable decides whether strict runtime requires a configured departure profile URL.
- In plain English, it blocks strict departure-aware routing when the departure source is undefined.
- This matters because departure optimization is only meaningful with a trusted profile.
- In thesis terms, it is a fail-closed rule for temporal calibration.

### `LIVE_DEPARTURE_ALLOW_SIGNED_FALLBACK`

- This variable controls whether signed fallback departure profiles are allowed.
- In plain English, it balances live freshness against fallback availability for departure context.
- This follows the same governance pattern as other live-backed assets.
- In thesis terms, it shows cross-subsystem consistency in source policy.

### `LIVE_DEPARTURE_ALLOWED_HOSTS`

- This variable restricts allowed hosts for departure profile retrieval.
- In plain English, it curates where departure calibration data may come from.
- This matters because temporal context is treated as governed model input.
- In thesis terms, it supports provenance discipline for schedule modeling.

### `LIVE_STOCHASTIC_REGIMES_URL`

- This variable points to the stochastic regimes artifact.
- In plain English, it tells the backend where uncertainty calibration comes from.
- This is foundational for q95, CVaR, and robust-score computation.
- In thesis terms, it anchors the live-backed uncertainty-calibration path.

### `LIVE_STOCHASTIC_REQUIRE_URL_IN_STRICT`

- This variable decides whether a stochastic source URL is mandatory in strict mode.
- In plain English, strict robust routing cannot proceed without a configured uncertainty source.
- This matters because the system should not fake calibrated risk.
- In thesis terms, it is a strictness rule for the stochastic subsystem.

### `LIVE_STOCHASTIC_ALLOW_SIGNED_FALLBACK`

- This variable controls whether signed stochastic-regime fallback is allowed.
- In plain English, it balances risk-calibration availability against source freshness.
- Disallowing fallback keeps the robust-routing story more conservative.
- In thesis terms, it expresses caution around uncertainty inputs.

### `LIVE_STOCHASTIC_ALLOWED_HOSTS`

- This variable restricts allowed hosts for stochastic-regime retrieval.
- In plain English, it curates where uncertainty calibration may come from.
- This matters because the risk layer is too important to source casually.
- In thesis terms, it is part of uncertainty-governance discipline.

### `LIVE_TOLL_TOPOLOGY_URL`

- This variable points to the toll-topology artifact.
- In plain English, it tells the backend where the structural toll map comes from.
- This matters because geometry-to-toll mapping depends on it.
- In thesis terms, it anchors one half of the toll model.

### `LIVE_TOLL_TOPOLOGY_REQUIRE_URL_IN_STRICT`

- This variable decides whether strict mode requires a toll-topology source URL.
- In plain English, strict toll modeling cannot proceed structurally without a defined topology source.
- This matters because toll realism is not reduced to tariffs alone.
- In thesis terms, it is a fail-closed rule on the topology side of the toll subsystem.

### `LIVE_TOLL_TOPOLOGY_ALLOW_SIGNED_FALLBACK`

- This variable controls whether signed fallback toll topology may be used.
- In plain English, it balances source continuity against topology freshness.
- This follows the repo-wide live governance pattern.
- In thesis terms, it shows consistency in how structured assets are governed.

### `LIVE_TOLL_TARIFFS_URL`

- This variable points to the toll tariffs artifact.
- In plain English, it tells the backend where toll prices come from.
- This matters because price realism depends on more than a topology map.
- In thesis terms, it anchors the monetary side of the toll subsystem.

### `LIVE_TOLL_TARIFFS_REQUIRE_URL_IN_STRICT`

- This variable decides whether strict mode requires a toll-tariff source URL.
- In plain English, the backend will not perform strict toll pricing without a defined tariff source.
- This matters because unknown toll costs should not silently become zero or guessed values.
- In thesis terms, it is a fail-closed rule on toll pricing.

### `LIVE_TOLL_TARIFFS_ALLOW_SIGNED_FALLBACK`

- This variable controls whether signed fallback toll tariffs are allowed.
- In plain English, it balances tariff freshness against tariff availability.
- This is another instance of the repository's cautious fallback policy.
- In thesis terms, it completes the toll-governance pattern.

### `LIVE_TOLL_ALLOWED_HOSTS`

- This variable restricts allowed toll-source hosts.
- In plain English, it controls where toll topology and tariff artifacts may be fetched from.
- This matters for provenance and supply-chain hygiene.
- In thesis terms, it keeps the toll subsystem's live inputs curated.

### `SCENARIO_REQUIRE_SIGNATURE`

- This variable requires scenario artifacts to carry valid signatures.
- In plain English, it insists on integrity checks for one of the most important live-backed assets in the repo.
- This matters because scenario coefficients materially change route outcomes.
- In thesis terms, it is a particularly strong integrity guard for the policy-analysis layer.

## Appendix V: Strict Reason Codes And Practical Mitigations

This appendix expands the earlier reason-code list into a more operational reading guide.

The project's strict failures are part of its methodology.

They are not merely technical error strings.

They express what the system refuses to assume when evidence is missing or invalid.

### `routing_graph_warming_up`

- This reason code means the graph is still loading or being prepared.
- In plain English, the backend is not yet ready to do strict graph-led routing.
- The route is blocked not because the origin and destination are bad, but because the routing substrate is incomplete.
- The practical mitigation is to wait for readiness and recheck `GET /health/ready`.
- In thesis terms, this code justifies the phrase "warming up the graph" as a real runtime phase.
- It also shows why liveness and readiness must be separated.

### `routing_graph_warmup_failed`

- This reason code means warmup did not complete successfully.
- In plain English, the graph-preparation phase broke or timed out badly enough that strict routing cannot trust the graph.
- This is stronger than temporary warmup-in-progress.
- The mitigation is to inspect warmup diagnostics, asset validity, and build outputs.
- In thesis terms, this is an integrity failure of the graph substrate.
- It protects the system from quietly routing on a broken graph.

### `routing_graph_fragmented`

- This reason code means the loaded graph failed connectivity quality expectations.
- In plain English, the graph exists, but too much of it is disconnected for strict trust.
- This usually points to build or source-quality issues rather than user input error.
- The mitigation is to inspect graph coverage and giant-component metrics.
- In thesis terms, it demonstrates that graph existence alone is not enough.
- Quality thresholds matter.

### `routing_graph_coverage_gap`

- This reason code means the requested OD geography is not adequately covered by the graph.
- In plain English, the route is asking the graph to work in an area it cannot represent well enough.
- This is especially important near edges or out-of-scope geography.
- The mitigation is to verify coordinates, graph coverage, and UK scope.
- In thesis terms, it is a geographically honest failure.
- It supports the report's insistence on UK-only operational confidence.

### `routing_graph_disconnected_od`

- This reason code means the origin and destination do not appear connected within the usable graph.
- In plain English, even if both points are on the graph, the backend cannot find a trusted graph connection between them.
- This is more specific than a generic "no route."
- The mitigation is to inspect snapping, connectivity, and graph build quality.
- In thesis terms, it shows that feasibility screening is explicit rather than implicit.
- It also keeps graph failures interpretable.

### `routing_graph_unavailable`

- This reason code means the graph asset is missing, invalid, or otherwise unusable.
- In plain English, the custom routing layer simply is not available.
- This is one of the strongest structural failure modes in the system.
- The mitigation is to rebuild or republish graph assets and confirm startup readiness.
- In thesis terms, it highlights how central the graph is to the smart-routing pipeline.
- Without it, the enriched candidate-generation layer cannot function as designed.

### `model_asset_unavailable`

- This is a more generic asset failure reason code.
- In plain English, some required model artifact could not be loaded or trusted.
- It matters because not every failure belongs neatly to one subsystem-specific label.
- The mitigation is to inspect asset manifests, build steps, and runtime publication state.
- In thesis terms, it reminds the reader that the system depends on a family of calibrated artifacts, not only one file.
- It is a catch-all guardrail, not a vague excuse.

### `departure_profile_unavailable`

- This reason code means the departure profile asset is missing or unusable.
- In plain English, the backend cannot trust its time-of-day uplift model.
- This matters especially for departure optimization and departure-time-specific route scoring.
- The mitigation is to rebuild or republish the departure profile asset and verify freshness.
- In thesis terms, it shows departure modeling is treated as a governed input family.
- The system does not silently replace missing departure knowledge with a fake average.

### `scenario_profile_unavailable`

- This reason code means the scenario profile or context could not be obtained reliably.
- In plain English, the backend cannot responsibly apply no-sharing, partial-sharing, or full-sharing behavior in the current context.
- This can be caused by missing live coefficients, missing context, or stale source state.
- The mitigation is to inspect live scenario sources, coefficient freshness, and source count rules.
- In thesis terms, it is central to the strict/live argument.
- Scenario analysis is only allowed when its inputs are trustworthy enough.

### `scenario_profile_invalid`

- This reason code means the scenario profile exists but failed validation.
- In plain English, the backend saw scenario data that did not meet schema or monotonicity expectations.
- This matters because bad scenario data can be worse than no scenario data.
- The mitigation is to inspect profile-generation outputs and validation logs.
- In thesis terms, it reinforces the idea that the system validates semantics, not only file presence.
- It is an example of correctness trumping convenience.

### `stochastic_calibration_unavailable`

- This reason code means the uncertainty subsystem lacks the calibrated regime information it needs.
- In plain English, the backend cannot produce trustworthy stochastic summaries for this context.
- That may be because the posterior model is missing, the regime cannot be matched, or calibration assets are absent.
- The mitigation is to inspect stochastic regime publication and context compatibility.
- In thesis terms, this code protects the integrity of robust and tail-risk outputs.
- It prevents fake precision in uncertainty reporting.

### `risk_prior_unavailable`

- This reason code means a risk-normalization or prior reference required by the risk layer is missing.
- In plain English, the backend cannot scale uncertainty-aware utility the way it expects.
- This matters because route utility is normalized against reference intensities rather than raw unit sums alone.
- The mitigation is to inspect risk normalization reference assets and configuration.
- In thesis terms, it supports the calibrated nature of robust utility.
- The system refuses to make up reference scales when they are absent.

### `vehicle_profile_unavailable`

- This reason code means the requested vehicle profile is unknown or missing.
- In plain English, the backend does not know the physical and cost assumptions for the selected vehicle.
- This matters because vehicle type affects terrain, fuel, emissions, and sometimes stochastic context.
- The mitigation is to use a built-in profile or create a valid custom profile first.
- In thesis terms, it shows vehicle realism is explicit and typed.
- The router is not pretending all freight vehicles are interchangeable.

### `vehicle_profile_invalid`

- This reason code means a vehicle profile was found but failed validation.
- In plain English, the system saw a vehicle definition that it cannot trust.
- Invalid physical parameters would contaminate multiple downstream models.
- The mitigation is to repair the custom profile schema or values.
- In thesis terms, this protects the physical-model layer from bad user-defined inputs.
- It is especially important because custom vehicles are a powerful but risky feature.

### `terrain_dem_asset_unavailable`

- This reason code means terrain DEM support is unavailable.
- In plain English, the backend cannot access the elevation evidence it needs.
- This can happen because local DEM assets are absent or live terrain retrieval is unavailable.
- The mitigation is to inspect DEM assets, live terrain URLs, and cache state.
- In thesis terms, it demonstrates that terrain-aware routing is only allowed with real support data.
- There is no silent imaginary hill model.

### `terrain_dem_coverage_insufficient`

- This reason code means terrain support exists but does not cover enough of the route.
- In plain English, the backend has some elevation information but not enough to trust the final terrain adjustment.
- This is especially important under UK fail-closed terrain settings.
- The mitigation is to improve coverage, relax policy intentionally, or accept route failure.
- In thesis terms, it is one of the best examples of "partial evidence is not enough."
- It shows a high bar for physical realism.

### `toll_topology_unavailable`

- This reason code means tolled-network structure cannot be trusted.
- In plain English, the backend does not know where toll infrastructure intersects the route well enough.
- Tariffs alone are not enough without topology.
- The mitigation is to inspect toll-topology assets and publication state.
- In thesis terms, it illustrates that monetary modeling depends on structural and pricing data together.
- This is a topology failure, not merely a price failure.

### `toll_tariffs_unavailable`

- This reason code means tariff data is missing or unusable.
- In plain English, the backend may know a route hits toll infrastructure but cannot price it credibly.
- This matters because toll realism needs actual charging data, not only toll flags.
- The mitigation is to inspect tariff assets, truth extraction, and publication state.
- In thesis terms, it protects the monetary objective from fake completeness.
- The system refuses to pretend an unknown tariff is zero.

### `fuel_price_unavailable`

- This reason code means the fuel-price source needed for cost modeling is unavailable or invalid.
- In plain English, the backend cannot translate fuel use into money the way strict runtime expects.
- This matters because the route cost objective is not just tolls and time; fuel is central.
- The mitigation is to inspect fuel-price publication, signatures, and freshness.
- In thesis terms, it supports the claim that route cost is data-driven.
- Missing fuel evidence blocks cost realism.

### `carbon_schedule_unavailable`

- This reason code means carbon pricing input is unavailable.
- In plain English, the backend cannot price environmental burden according to the configured carbon schedule.
- This affects carbon-cost reporting more directly than raw emissions.
- The mitigation is to inspect carbon schedule publication and strict URL settings.
- In thesis terms, it separates emissions modeling from environmental monetization.
- The system may know the CO2 but not the current carbon price context.

### `no_route_candidates`

- This reason code means the backend completed enough of the pipeline to conclude there are no viable candidates left.
- In plain English, route generation ran, but nothing survived the filters or feasibility conditions.
- This is different from a structural asset failure.
- The mitigation is to inspect graph feasibility, objective thresholds, and scenario/terrain strictness.
- In thesis terms, it is a substantive routing failure rather than a system-readiness failure.
- It tells the user the pipeline searched but could not produce a defensible answer.

### Why This Appendix Matters

- Strict reason codes are a design language.
- They tell the operator what sort of trust boundary was crossed.
- They also tell a thesis reader which subsystem made a route impossible.
- This is more rigorous than generic error text.
- It allows the report to discuss system failure behavior with precision.
- It is one of the clearest markers that the project was engineered for auditability.

## Appendix W: Data Provenance And Asset-Family Deep Dive

This appendix expands the earlier data chapter into a more explicit family-by-family provenance narrative.

The goal is to answer not only "what data exists?"

but also "why does this file exist, which runtime behavior depends on it, and what thesis claim can it support?"

The repository contains both source-like tracked assets and raw-data families that explain how those tracked assets were derived.

### W1. Routing Graph Asset Family

### `backend/assets/uk/uk-latest.osm`

- This is the primary UK OSM-derived graph-source asset tracked in the repository.
- It matters because the custom routing graph cannot exist without a regional road-network source.
- In plain English, this is the structural road map from which graph-led search begins.
- The routing graph builder script uses this family to create the compiled graph artifact used at runtime.
- In thesis terms, it is the data root of the hybrid routing architecture.
- Every claim about graph-led candidate generation ultimately depends on this source family.

### Local generated graph outputs under `backend/out/model_assets`

- The compiled graph itself is typically a generated runtime artifact rather than a tracked lightweight source file.
- Local evidence in the working tree includes graph coverage reporting with node count, edge count, size, and bounding box.
- In plain English, this is where the source road network becomes an operational search structure.
- The local coverage report gives concrete evidence for graph scale: millions of nodes and edges over a UK bounding box.
- In thesis terms, this local generated evidence is stronger than generic architectural prose because it demonstrates that the graph was actually built.
- It also supports statements about startup cost and warmup necessity.

### W2. Vehicle Profile Family

### `backend/assets/uk/vehicle_profiles_uk.json`

- This asset defines the built-in vehicle archetypes.
- The local repo shows profiles such as van, rigid HGV, artic HGV, and EV HGV.
- In plain English, it tells the router what kind of machine is traveling.
- This matters because mass, efficiency, energy use, and terrain response differ sharply across vehicle classes.
- In thesis terms, this file supports the claim that the model is fleet-aware rather than vehicle-agnostic.
- It is also the reference point for validating custom vehicle submissions.

### W3. Departure Profile Family

### `backend/assets/uk/departure_profiles_uk.json`

- This asset stores the departure-time uplift model used during route scoring and departure optimization.
- It is built rather than handwritten.
- In plain English, it tells the router how UK timing pressure changes by context.
- This file is why the departure optimizer can produce different answers across a time window.
- In thesis terms, it is one of the clearest data artifacts supporting schedule-aware routing.
- Without it, departure optimization would be mostly performative.

### `backend/assets/uk/departure_counts_empirical.csv`

- This empirical CSV is the observation-oriented companion to the published departure profile.
- It matters because a profile table is more credible when the underlying counts are preserved.
- In plain English, it shows some of the evidence behind the departure uplift model.
- This is especially useful in a thesis because it helps separate raw observation from processed profile.
- In thesis terms, it is part of the departure model's provenance chain.
- It supports claims that the profile is calibrated rather than imagined.

### W4. Scenario Profile Family

### `backend/assets/uk/scenario_profiles_uk.json`

- This is one of the most important assets in the whole repository.
- It stores the context-conditioned scenario multipliers for no-sharing, partial-sharing, and full-sharing.
- Local evidence shows a version string, similarity weights, and mode-effect scales.
- In plain English, it is the file that makes sharing modes operational rather than rhetorical.
- In thesis terms, it is central to the policy-sensitive identity of the project.
- The report's discussion of sharing multipliers, scenario context keys, and mode-specific pressure comes directly from this family.

### `backend/data/raw/uk/scenario_live_observed.jsonl`

- This raw JSONL corpus contains observed or collected scenario context data.
- It is useful because it preserves the live evidence base from which scenario profiles are built or refreshed.
- In plain English, it is part of the "what the world looked like" record behind scenario calibration.
- There is also a strict-oriented counterpart for stricter collection contexts.
- In thesis terms, this raw family supports claims that scenario coefficients are informed by collected context rather than static opinions.
- It is one of the repo's most important raw evidence stores.

### `backend/data/raw/uk/scenario_mode_outcomes_observed.jsonl`

- This raw family records observed or proxy outcomes for scenario-mode effects.
- It matters because the system needs evidence about what sharing assumptions should do to route outcomes.
- In plain English, it is the raw support for turning sharing modes into calibrated multipliers.
- The summary files alongside it help quantify data volume or coverage.
- In thesis terms, this family helps justify why no-sharing, partial-sharing, and full-sharing are not arbitrary labels.
- It is a bridge from scenario intuition to scenario parameterization.

### W5. Stochastic Calibration Family

### `backend/assets/uk/stochastic_regimes_uk.json`

- This asset stores the calibrated stochastic regimes used by the uncertainty model.
- It includes regime parameters, correlation structure references, and posterior context mapping support.
- Local evidence shows multiple regimes and holdout-quality summaries.
- In plain English, this is the backend's structured knowledge of how uncertainty behaves under different UK freight contexts.
- In thesis terms, it is foundational for q95, CVaR, robust score, and regime-resolution discussion.
- It is one of the richest data artifacts in the repository.

### `backend/assets/uk/stochastic_residual_priors_uk.json`

- This asset stores prior-style uncertainty references used by the stochastic or risk layer.
- It matters because the uncertainty subsystem needs more than one table.
- In plain English, it helps give the backend a prior expectation for residual behavior before or alongside richer posterior resolution.
- This supports stable uncertainty computation when context detail varies.
- In thesis terms, it contributes to the calibrated depth of the stochastic subsystem.
- It also helps explain why risk failures can refer to missing priors or calibration.

### `backend/assets/uk/stochastic_residuals_empirical.csv`

- This CSV preserves empirical residual evidence behind the stochastic calibration.
- It is important because processed regimes are easier to defend when their empirical source is preserved.
- In plain English, it is part of the observed mismatch history used to shape uncertainty behavior.
- The raw-data directory also contains a further-downstream raw residual corpus.
- In thesis terms, this family supports the data lineage of the risk model.
- It is evidence that the stochastic layer was calibrated from data, not invented wholesale.

### `backend/data/raw/uk/stochastic_residuals_raw.csv`

- This raw CSV is the earlier-stage collection side of the residual pipeline.
- It matters because preserving raw evidence helps debug or rebuild later calibrations.
- In plain English, it is the rough uncertainty evidence before final packaging into runtime assets.
- Summary files alongside it provide compact metadata about the collection.
- In thesis terms, it is part of the audit trail behind the stochastic model.
- It helps the report describe a full path from raw residuals to usable regimes.

### W6. Fuel And Energy Asset Family

### `backend/assets/uk/fuel_prices_uk.json`

- This asset stores the live-backed or published fuel-price snapshot used in cost modeling.
- Local evidence shows diesel, petrol, LNG, and grid-energy values.
- In plain English, it tells the router how much energy or fuel currently costs.
- This matters because route monetary cost depends on more than road length.
- In thesis terms, it supports economic realism and time-sensitive cost interpretation.
- It is also a strict-governed live source.

### `backend/assets/uk/fuel_consumption_surface_uk.json`

- This asset stores the fuel-consumption surface used by the richer cost model.
- It matters because fuel use depends on more than one scalar distance multiplier.
- In plain English, it is a lookup-style representation of how route conditions translate into consumption.
- This supports better cost and emissions estimation.
- In thesis terms, it is part of the engineering move away from simplistic distance times constant formulas.
- It helps explain why the router can respond to terrain and operating context.

### `backend/assets/uk/fuel_uncertainty_surface_uk.json`

- This asset stores uncertainty-oriented information related to fuel behavior.
- It matters because cost uncertainty can arise not only from traffic but also from energy-price or consumption uncertainty surfaces.
- In plain English, it helps the backend reason about the variability around fuel-driven cost.
- This strengthens the monetary side of the stochastic layer.
- In thesis terms, it contributes to the risk-aware economic model.
- It also illustrates how uncertainty is distributed across multiple objective families.

### `backend/data/raw/uk/fuel_prices_raw.json`

- This raw file preserves source fuel-price observations before final packaging.
- It matters because a published snapshot is more credible when the underlying collection is preserved.
- In plain English, it is the raw economic input behind the runtime fuel asset.
- Summary and seed-public files around it help explain how the final asset was assembled.
- In thesis terms, it supports fuel-price provenance.
- It is part of the monetary data audit trail.

### W7. Carbon Asset Family

### `backend/assets/uk/carbon_price_schedule_uk.json`

- This asset stores the carbon price schedule used for environmental monetization.
- Local evidence shows multiple schedule levels over future years.
- In plain English, it tells the system how to convert emissions into carbon-cost terms under a chosen schedule.
- This matters because emissions and carbon cost are related but not identical outputs.
- In thesis terms, it supports the environmental-economics side of the route model.
- It is also governed under strict live-source policy.

### `backend/assets/uk/carbon_intensity_hourly_uk.json`

- This asset stores carbon-intensity information over time.
- It matters because environmental burden can depend on temporal context, especially where electricity-related or time-varying assumptions matter.
- In plain English, it helps the system model environmental context more richly than one timeless carbon constant.
- This is especially relevant for EV-oriented or grid-related interpretations.
- In thesis terms, it extends the environmental model from emissions accounting toward temporal context.
- It also helps explain why carbon fetchers and collectors exist separately from fuel sources.

### `backend/data/raw/uk/carbon_intensity_hourly_raw.json`

- This raw family preserves collected carbon-intensity evidence.
- It matters because time-varying carbon assets should have a traceable upstream source.
- In plain English, it is the unprocessed environmental-context evidence behind the published intensity table.
- Summary files help quantify recency and collection structure.
- In thesis terms, it supports the provenance of temporal environmental inputs.
- It is part of the climate-data audit trail in the repo.

### W8. Toll Asset Family

### `backend/assets/uk/toll_topology_uk.json`

- This asset stores toll-topology structure.
- It matters because monetary toll treatment needs route-to-infrastructure mapping, not only tariff values.
- In plain English, it tells the router where tolled infrastructure is and how route segments relate to it.
- This is a structural map for the toll engine.
- In thesis terms, it supports realistic money modeling by connecting geometry to charging rules.
- It is one half of the toll story.

### `backend/assets/uk/toll_tariffs_uk.json`

- This asset stores toll tariffs.
- It is the pricing counterpart to toll topology.
- In plain English, it tells the router how much a tolled crossing or road should cost under supported assumptions.
- This matters because identifying a toll segment is not enough to compute money.
- In thesis terms, it is the value side of the toll model.
- It works in tandem with topology and confidence calibration.

### `backend/assets/uk/toll_confidence_calibration_uk.json`

- This asset stores confidence calibration for toll interpretation.
- It matters because not every toll inference has equal certainty.
- In plain English, it helps the system express how much trust it places in classified toll situations.
- This is a sophisticated feature that many simple route systems do not expose at all.
- In thesis terms, it supports the claim that the repo treats toll inference as uncertain, not binary-perfect.
- It adds calibration depth to the monetary model.

### `backend/data/raw/uk/toll_tariffs_operator_truth.json`

- This raw family preserves operator-truth style tariff evidence.
- It matters because published tariff tables should have a traceable raw source.
- In plain English, this is one of the upstream truth sets from which toll tariffs were built.
- Summary files around it help document collection state.
- In thesis terms, this family supports toll-price provenance.
- It is especially important when discussing how the local system differs from pure map-derived routing.

### `backend/data/raw/uk/toll_classification/*`

- This raw family contains many classification cases for toll interpretation.
- The volume of files reflects breadth of toll classification examples rather than needless duplication.
- In plain English, this family helps the project learn or validate how different toll situations should be identified.
- The matching test-fixture family mirrors this structure for regression protection.
- In thesis terms, it demonstrates depth in a subsystem that many routing projects oversimplify.
- The large family size is part of that evidence.

### `backend/data/raw/uk/toll_pricing/*`

- This raw family contains pricing cases and proxy pricing examples for toll scenarios.
- It matters because toll pricing requires case diversity, not one universal toll amount.
- In plain English, it is a catalogue of toll-price situations used in building or validating tariff behavior.
- The presence of proxy-price files shows the repo explicitly models some incomplete-truth situations.
- In thesis terms, it supports a nuanced discussion of toll pricing confidence and fallback.
- It is a major raw-evidence family for route cost realism.

### W9. Terrain Manifest Family

### `backend/assets/uk/terrain_dem_grid_uk.json`

- This asset stores terrain-grid or tile-manifest information for UK DEM support.
- It matters because terrain sampling needs a structured index of what elevation support exists.
- In plain English, it tells the backend how to find or interpret terrain coverage over the UK.
- This is a necessary support asset behind strict terrain coverage checking.
- In thesis terms, it underpins the physical realism layer of the router.
- It also supports explainability around coverage thresholds.

### W10. Risk-Normalization Family

### `backend/assets/uk/risk_normalization_refs_uk.json`

- This asset stores normalization references for the risk model.
- It matters because robust utility is computed on dimensionless normalized components rather than raw incomparable units alone.
- In plain English, it tells the system what "normal" duration, money, and emissions intensity look like for relevant contexts.
- This is crucial for meaningful weighted utility under uncertainty.
- In thesis terms, it makes the risk layer more principled and context-aware.
- It is one of the quieter but conceptually important assets in the repo.

### W11. Runtime Output Evidence Families

### `backend/out/manifests/*`

- This runtime family stores run manifests.
- It matters because successful analytical runs become inspectable records rather than transient response bodies.
- In plain English, manifests summarize what was run and under which assumptions or asset versions.
- They are central to reproducibility.
- In thesis terms, they are one of the strongest forms of local experimental evidence.
- They connect the routing engine to defensible archival output.

### `backend/out/scenario_manifests/*`

- This runtime family stores scenario-specific manifests.
- It matters because scenario-comparison workflows need their own archival trace.
- In plain English, it records how scenario-oriented runs were framed and what assets they depended on.
- This is useful for longer comparative studies.
- In thesis terms, it supports reproducible policy-mode analysis.
- It also helps keep scenario experiments distinct from ordinary route runs.

### `backend/out/provenance/*`

- This runtime family stores provenance logs.
- It matters because run artifacts are stronger when source and context evidence are recorded alongside them.
- In plain English, it is the trace of where the run's important inputs came from.
- Provenance is especially important in a live-data-aware system.
- In thesis terms, it is part of the chain of custody for analytical evidence.
- It supports claims about source governance.

### Why This Appendix Matters

- The repository uses a layered data strategy.
- There are tracked operational assets, raw collection families, and local runtime outputs.
- That layered structure is a strength for thesis writing because it supports both operational explanation and provenance explanation.
- It also makes clear that the project is not only code plus a map.
- It is a calibrated data system with explicit asset lifecycles.
- That is one of the main reasons it can support richer routing claims than a simple baseline router.

## Appendix X: Endpoint Semantics In Narrative Form

This appendix expands the endpoint inventories into a more thesis-friendly explanation of what each route actually means.

The goal is to help a reader understand the analytical function of each endpoint, not only its URL.

### X1. Core Health And Readiness Endpoints

### `GET /health`

- This endpoint answers whether the backend process is alive.
- It does not imply the smart-routing pipeline is fully ready.
- In plain English, it tells the user "the API is up."
- It is important operationally because a live process can still be unready for strict route service.
- In thesis terms, it supports the distinction between process health and model readiness.

### `GET /health/ready`

- This endpoint answers whether the backend is ready for strict route production.
- It includes graph-warmup status and strict-live diagnostics.
- In plain English, it tells the user whether the router can be trusted to answer right now.
- This is especially important in a graph-heavy, fail-closed system.
- In thesis terms, it is one of the most informative operational endpoints in the repo.

### X2. Core Route Endpoints

### `POST /route`

- This is the single-route endpoint.
- It returns one recommended route after running the candidate-generation, modeling, and selection pipeline.
- In plain English, it asks the system for the best highlighted route under the chosen settings.
- The backend still uses graph-led candidates and scalar selection internally rather than simple shortest path.
- In thesis terms, this endpoint is where the modified selectors matter most visibly.

### `POST /pareto`

- This endpoint returns a non-streaming Pareto-style result.
- It keeps multiple non-dominated or epsilon-feasible routes instead of only one selected route.
- In plain English, it asks for the trade-off set rather than the highlighted winner.
- This endpoint is central to the dissertation's multi-objective argument.
- In thesis terms, it is arguably the most academically expressive API route in the backend.

### `POST /pareto/stream`

- This endpoint streams Pareto results as events.
- It exposes progress and partial route delivery rather than forcing the user to wait for one final JSON payload.
- In plain English, it is the observable version of Pareto computation.
- This matters because frontier generation plus strict live refresh can take time.
- In thesis terms, it combines multi-objective routing with transparent asynchronous interaction.

### `POST /route/baseline`

- This endpoint returns the OSRM baseline-style route.
- It exists to support per-request comparison against a simpler reference path.
- In plain English, it asks "what would the baseline route look like for this same trip?"
- The endpoint is valuable because it lets comparison happen within the same product and artifact framework.
- In thesis terms, it supports disciplined baseline interpretation.

### `POST /route/baseline/ors`

- This endpoint returns the ORS-style baseline or permitted proxy fallback.
- It broadens the comparison surface beyond OSRM-only reference logic.
- In plain English, it asks for the ORS comparison answer while preserving explicit fallback semantics.
- This matters because the thesis should not pretend all baselines are identical.
- In thesis terms, it supports more nuanced benchmark framing.

### X3. Scenario And Planning Endpoints

### `POST /scenario/compare`

- This endpoint compares route outcomes across scenario modes.
- The route is evaluated under different sharing assumptions rather than only one chosen mode.
- In plain English, it asks how collaboration assumptions change the route answer.
- This is a powerful endpoint because it moves from optimization into comparative policy-style analysis.
- In thesis terms, it is one of the clearest signs that the project is more than a shortest-path service.

### `POST /departure/optimize`

- This endpoint evaluates candidate departure slots across a window.
- It recomputes route behavior under departure-sensitive context.
- In plain English, it asks not just "which route?" but also "when should I leave?"
- This is an operationally important extension of route planning.
- In thesis terms, it demonstrates schedule-aware optimization using UK departure profiles.

### `POST /duty/chain`

- This endpoint computes a chained multi-stop duty.
- It aggregates multiple legs into a single planning result.
- In plain English, it lets a user plan a sequence of stops rather than an isolated trip.
- This matters because real freight work is often chain-based.
- In thesis terms, it shows the project extending into duty planning.

### X4. Batch And Experiment Endpoints

### `POST /batch/pareto`

- This endpoint runs Pareto analysis across multiple OD pairs.
- It is aimed at studies, benchmarks, and repeated comparison rather than one-off interactive use only.
- In plain English, it asks the backend to solve many route-trade-off problems in one run.
- The endpoint often writes richer artifacts than a simple one-route response.
- In thesis terms, it is a major tool for empirical evaluation.

### `POST /batch/import/csv`

- This endpoint accepts batch input from CSV-like payloads.
- It matters because many real analyses begin from tables rather than map clicks.
- In plain English, it translates spreadsheet-style case lists into routed studies.
- This is important for reproducible dissertation case corpora.
- In thesis terms, it lowers the cost of moving from scenario list to measured study.

### `GET /experiments`

- This endpoint lists saved experiments.
- It exists so repeated analyses are browsable and reusable.
- In plain English, it lets the application remember past study setups.
- This matters for longer research cycles.
- In thesis terms, it supports a structured experimentation workflow.

### `POST /experiments`

- This endpoint creates a saved experiment bundle.
- It is the persistence side of the experiment manager.
- In plain English, it stores a study definition for later replay or comparison.
- This matters because many analytical investigations are iterative.
- In thesis terms, it supports comparative research continuity.

### `GET /experiments/{experiment_id}`

- This endpoint retrieves one stored experiment.
- It matters because saved analyses should be addressable, not anonymous.
- In plain English, it loads one specific study definition or record.
- This supports repeatability and revision.
- In thesis terms, it helps turn the system into a study-management tool.

### `DELETE /experiments/{experiment_id}`

- This endpoint deletes a stored experiment.
- It is operationally modest but workflow-important.
- In plain English, it lets users clean or revise their experiment catalog.
- This matters for maintaining an intelligible analytical workspace.
- In thesis terms, it supports disciplined study lifecycle management.

### `POST /experiments/{experiment_id}/compare`

- This endpoint compares a stored experiment against another target state.
- It matters because saved studies are most useful when they can be contrasted directly.
- In plain English, it helps ask how one named study differs from another.
- This supports longitudinal and scenario-based comparison.
- In thesis terms, it extends experiment storage into experiment analysis.

### X5. Vehicle And Input-Management Endpoints

### `GET /vehicles`

- This endpoint lists available vehicle profiles.
- It matters because vehicle choice changes route physics, cost, and sometimes risk context.
- In plain English, it tells the frontend what built-in fleet assumptions exist.
- This supports both operator choice and validation.
- In thesis terms, it exposes the configurable physical-model surface.

### `GET /vehicles/custom`

- This endpoint lists custom vehicles.
- It matters because user-defined fleet models are a supported feature rather than a hidden hack.
- In plain English, it shows what extra vehicle assumptions the user has added.
- This supports repeatable use of tailored vehicle physics.
- In thesis terms, it strengthens the configurable modeling claim.

### `POST /vehicles/custom`

- This endpoint creates a custom vehicle.
- It is one of the most direct ways users can alter physical assumptions in the system.
- In plain English, it lets the operator define a new fleet archetype.
- This is powerful and therefore tightly validated.
- In thesis terms, it supports the model-extensibility story.

### `PATCH /vehicles/custom/{vehicle_id}`

- This endpoint updates a custom vehicle.
- It matters because user-defined physical assumptions may need revision.
- In plain English, it edits an existing fleet profile.
- This supports stable long-term experimentation with evolving assumptions.
- In thesis terms, it keeps custom physics under controlled lifecycle management.

### `DELETE /vehicles/custom/{vehicle_id}`

- This endpoint removes a custom vehicle.
- It is operationally simple but important for data hygiene.
- In plain English, it cleans up user-defined fleet entries that are no longer wanted.
- This matters because stale custom profiles could clutter later studies.
- In thesis terms, it supports a maintainable customization workflow.

### X6. Artifact, Signature, And Provenance Endpoints

### `GET /runs/{run_id}/manifest`

- This endpoint retrieves the main run manifest.
- The manifest records what was run and under what conditions.
- In plain English, it is the summary identity card for a stored analysis run.
- This is central to reproducibility.
- In thesis terms, it is one of the strongest evidence endpoints in the project.

### `GET /runs/{run_id}/scenario-manifest`

- This endpoint retrieves a scenario-specific manifest.
- It matters because scenario studies often deserve their own archived explanation.
- In plain English, it is the scenario-analysis counterpart to the main run manifest.
- This supports repeatable policy-mode comparison.
- In thesis terms, it strengthens scenario reproducibility.

### `GET /runs/{run_id}/signature`

- This endpoint retrieves the manifest signature.
- It exists because reproducibility is stronger when integrity can be checked.
- In plain English, it gives the user the signature for the run summary.
- This supports verification and trust.
- In thesis terms, it links artifact storage to integrity assurance.

### `GET /runs/{run_id}/scenario-signature`

- This endpoint retrieves the scenario-manifest signature.
- It matters because scenario artifacts also need authenticity support.
- In plain English, it is the signature-side companion to scenario-manifest retrieval.
- This helps keep scenario analysis auditable.
- In thesis terms, it supports complete artifact integrity for comparative studies.

### `GET /runs/{run_id}/provenance`

- This endpoint retrieves provenance information for a run.
- Provenance is distinct from manifest summary.
- In plain English, it tells the user where important inputs came from.
- This matters especially in a live-source-aware system.
- In thesis terms, it is a key evidence endpoint for data-governance claims.

### `GET /runs/{run_id}/artifacts`

- This endpoint lists artifacts associated with a run.
- It matters because one analytical run may produce several outputs.
- In plain English, it tells the user what files exist for that run.
- This supports browsing, export, and appendix preparation.
- In thesis terms, it turns computation outputs into a discoverable evidence set.

### `GET /runs/{run_id}/artifacts/{artifact_name}`

- This endpoint downloads a specific artifact file.
- It matters because reproducibility should end in retrievable outputs, not just stored metadata.
- In plain English, it is the "give me the file" endpoint for run evidence.
- This supports direct use of GeoJSON, CSV, or reports in later analysis.
- In thesis terms, it helps connect the backend to dissertation-ready outputs.

### `POST /verify/signature`

- This endpoint verifies a payload against a provided signature and secret.
- It matters because integrity should be testable from the API, not only assumed.
- In plain English, it answers whether a payload appears unaltered under the signing rule.
- This strengthens trust in run artifacts.
- In thesis terms, it supports the repository's unusually strong artifact-integrity story.

### X7. Oracle-Quality And Debug Endpoints

### `POST /oracle/quality/check`

- This endpoint records or evaluates upstream source quality observations.
- It matters because the project tracks source health explicitly.
- In plain English, it is part of asking whether the data feeds behind the model are healthy.
- This is separate from route ranking.
- In thesis terms, it supports a mature live-governance layer.

### `GET /oracle/quality/dashboard`

- This endpoint returns aggregated source-quality status in JSON form.
- It powers the oracle-quality dashboard in the frontend.
- In plain English, it summarizes pass rates, stale counts, schema failures, and latency.
- This helps a user judge input trustworthiness.
- In thesis terms, it separates source governance from route quality itself.

### `GET /oracle/quality/dashboard.csv`

- This endpoint returns a CSV export of the same source-quality summary.
- It matters because research and operational review often need tabular exports.
- In plain English, it is the spreadsheet-friendly view of source health.
- This supports reporting and external analysis.
- In thesis terms, it makes the governance layer portable.

### `GET /debug/live-calls/{request_id}`

- This endpoint returns the live-call trace for a request.
- It matters because it exposes exactly what live calls were made or skipped.
- In plain English, it answers "what external data activity happened during this route?"
- This is one of the most transparent endpoints in the repo.
- In thesis terms, it is a major part of the system's auditability story.

### Why This Appendix Matters

- Endpoint inventories are useful for completeness.
- Endpoint narratives are better for thesis writing.
- They connect URLs to modeling purpose, evidence generation, and operational meaning.
- They also make clear that the backend is not one monolithic router.
- It is a family of routing, comparison, scheduling, artifact, governance, and observability services.
- That breadth is part of what makes the project thesis-worthy.

## Appendix Y: Design Rationale, Alternatives, And Threats To Validity

This appendix addresses the "why did you do it this way?" questions directly.

That matters because a thesis is not only a technical inventory.

It is also a justification of design choices.

The choices below are grounded in the codebase, docs, tests, and local artifacts already discussed.

### Y1. Why The Project Is UK-Only

- The repository is explicitly UK-scoped in its data assets, live-source assumptions, and calibration families.
- This is not a weakness by accident.
- It is a deliberate scope choice.
- The UK constraint allows consistent road-network sourcing, departure-profile interpretation, bank-holiday treatment, corridor buckets, and scenario calibration.
- In plain English, the project can be more realistic in one region because it is not pretending to model the whole world equally well.
- That is a strong research design choice.
- A thesis should usually prefer a well-calibrated bounded domain over a vague global ambition.
- The threat to validity is obvious: claims should not be generalized beyond UK-like contexts without new calibration and asset pipelines.

### Y2. Why Not Use Plain OSRM Alone

- Plain OSRM is excellent at shortest-path style routing and road realization.
- The project still uses OSRM for that reason.
- However, plain OSRM does not on its own provide graph-diverse candidate generation tuned for this thesis, Pareto frontier construction, scenario-conditioned multipliers, robust utility, governed live-data refresh, or route-specific provenance outputs.
- In plain English, OSRM is a strong engine, but it is not the whole research system.
- The repo therefore keeps OSRM where it is strongest and adds missing analytical layers around it.
- The threat to validity here is that the smart router should not claim to "replace OSRM" in every sense.
- It augments and reframes the route-selection problem rather than denying the value of provider routing.

### Y3. Why Use A Hybrid Graph Plus OSRM Design

- The hybrid design solves two different problems with two different tools.
- The custom graph is used to generate structurally different candidate corridors under explicit budgets.
- OSRM is then used to refine or realize those corridors into route geometry and detailed path annotations.
- In plain English, the graph asks "what different ways through the network are plausible?" while OSRM asks "what does this candidate look like as a road route?"
- This is a defensible decomposition.
- It lets the project escape provider single-path dependence without discarding provider strengths.
- The main threat is complexity.
- More moving parts mean more room for readiness and integration failures, which is why strict diagnostics and tests are so important.

### Y4. Why Warm The Graph On Startup

- The graph is large enough that loading it on the first user request would be operationally ugly and analytically misleading.
- Warmup makes readiness explicit.
- It also allows the backend to reject compute while the graph is incomplete rather than pretending to be ready.
- In plain English, warming up the graph is a way of paying the graph-readiness cost up front and transparently.
- This helps operator expectations and benchmark interpretation.
- The threat is startup time and memory pressure.
- The project addresses that with fast-startup modes, readiness reporting, and quality gates.
- In thesis terms, warmup is part of the engineering required for a national-scale graph asset.

### Y5. Why Strict Fail-Closed Live Data

- The project is unusually strict about stale or invalid live-backed data.
- That is intentional.
- If scenario coefficients, fuel prices, carbon schedules, or terrain coverage are part of the model, then silently guessing around their absence would weaken the meaning of the result.
- In plain English, a strict failure is better than a polished but dishonest answer.
- This design is especially appropriate for a dissertation because it keeps claims auditable.
- The threat is reduced availability.
- Some requests will fail rather than degrade smoothly.
- The thesis should present that as a deliberate tradeoff rather than a hidden flaw.

### Y6. Why Allow Controlled Synthetic Incidents But Keep Them Separate

- Incident simulation exists because experimentation often requires controlled shock scenarios.
- At the same time, strict live routing should not confuse synthetic incidents with observed live truth.
- The repository keeps those roles separate.
- In plain English, the system allows "what if a disruption happens?" without pretending that the simulation is the world.
- That is good research hygiene.
- The threat would be conflation of simulated and live evidence.
- The code and docs avoid that by preserving explicit workflow boundaries.

### Y7. Why Sharing Modes Are Context-Conditioned

- The repository could have used one global multiplier for no-sharing, partial-sharing, and full-sharing.
- It does not.
- Instead, it uses context-conditioned profiles with similarity weighting across geography, time, weather, road mix, and vehicle context.
- In plain English, the effect of coordination is allowed to depend on where and when the trip happens.
- That is far more defensible than a universal uplift constant.
- The threat is data hunger and calibration complexity.
- The repo accepts that complexity because scenario analysis is one of its signature contributions.

### Y8. Why Departure Optimization Is Separate From Route Selection

- Departure optimization is not just another slider inside single-route ranking.
- It is a repeated evaluation over a time window using departure-sensitive context.
- Keeping it as a dedicated endpoint clarifies that it is a scheduling problem layered on top of routing.
- In plain English, choosing when to leave is not the same question as choosing which road path to prefer.
- This separation improves clarity and API design.
- The threat is user confusion about which endpoint to use.
- The frontend and docs compensate with dedicated charts and workflow explanation.

### Y9. Why Batch Workflows Matter

- Many research conclusions are weak if they come from one OD pair.
- Batch Pareto and CSV import exist because broader case coverage matters.
- In plain English, the repo wants to support study-scale analysis, not just pretty demos.
- This is essential for benchmarking, sensitivity analysis, and scenario comparisons across a corpus of trips.
- The threat is that batch endpoints are harder to reason about and more resource-hungry.
- The artifact system and benchmark tooling exist partly to manage that complexity.
- In thesis terms, batch workflows are what make broader empirical claims even possible.

### Y10. Why The Modified Selectors Exist

- Pure academic scalarization methods are informative but can produce route recommendations that feel brittle or unrepresentative after a frontier has already been generated.
- The modified selectors add balance, distance, ETA-distance, knee, and entropy terms.
- In plain English, they try to pick a route a human planner would recognize as a sensible compromise.
- The repo is careful not to present these profiles as novel optimization theory.
- That honesty is a strength.
- The threat is a loss of theoretical purity.
- The benefit is improved operational sensibility for the highlighted route.
- In thesis terms, this is a pragmatic engineering extension rather than an academic overclaim.

### Y11. Why Keep Academic Baselines Alongside Modified Ones

- The repository could have hidden the academic selectors and only exposed the tuned ones.
- It does not.
- Keeping academic reference profiles visible is a methodological strength.
- It allows the thesis to compare unchanged formulas against local engineering blends on the same frontier.
- In plain English, it shows what is standard and what is custom.
- That improves transparency.
- The threat is interface complexity.
- The gain is intellectual honesty.

### Y12. Why Route Backfill Exists

- A strict Pareto frontier can be too small to be useful in an interface.
- The repo therefore supports backfill of ranked routes when configured.
- In plain English, it stops the UI from showing an impoverished route set when the true frontier collapses.
- This is clearly an engineering choice, not a theorem.
- The threat is that users may treat backfilled routes as if they were all equally "pure" frontier points.
- The report should therefore distinguish backfill from strict frontier routes explicitly.
- In thesis terms, backfill is a product-support compromise that keeps the system usable.

### Y13. Why So Much Observability Exists

- The project includes live-call tracing, run manifests, provenance logs, signatures, oracle-quality dashboards, readiness endpoints, and cache metrics.
- This is unusually rich for a routing project.
- In plain English, the system spends real effort explaining itself.
- That is not accidental.
- A strict, data-governed, multi-stage pipeline becomes much easier to trust when its internal state is visible.
- The threat is implementation complexity and interface density.
- The benefit is auditability.
- In thesis terms, observability is one of the repository's strongest engineering contributions.

### Y14. Why Signatures Matter

- Many projects stop at writing JSON results.
- This repo adds HMAC-based signatures.
- In plain English, it lets a user check whether a manifest or payload appears unchanged.
- This is not enterprise-grade PKI, and the code does not pretend otherwise.
- It is a simple integrity mechanism appropriate to the repo's scope.
- The threat would be overselling it as stronger security than it is.
- In thesis terms, signatures strengthen artifact integrity without making inflated cryptographic claims.

### Y15. Why The Frontend Is So Feature-Rich

- The frontend is more than a map because the backend itself is more than a route endpoint.
- If the repo includes Pareto exploration, scenario comparison, duty planning, experiment storage, provenance retrieval, and live diagnostics, then the UI must expose those capabilities.
- In plain English, a thin demo frontend would underrepresent the real system.
- The result is a denser page and some large orchestration files.
- That is a tradeoff.
- The threat is frontend complexity.
- The benefit is that the thesis can demonstrate the full system rather than a narrow slice of it.
- In research terms, the UI is part of the method, not just presentation polish.

### Y16. Why Accessibility And i18n Are Included

- Accessibility and i18n are not the central research novelty, but they are still meaningful.
- Async route computation with long waits can easily become inaccessible if status changes are not announced.
- A complex analytical tool can also become unnecessarily exclusionary if everything assumes one language and one interaction pattern.
- In plain English, these features make the system more usable and credible.
- The threat is extra implementation effort in a research project.
- The gain is a more serious operator-facing tool.
- In thesis terms, this strengthens the argument that the project was built for actual use rather than only for code inspection.

### Y17. Why The Repo Is Notebook-Light

- The workflow is intentionally script-first rather than notebook-first.
- That means asset builds, quality gates, benchmarks, and report generation are easier to rerun consistently.
- In plain English, the project prefers commands and artifacts over hidden notebook state.
- This is a good fit for reproducibility.
- The threat is a steeper learning curve for users who prefer exploratory notebooks.
- The benefit is much stronger operational repeatability.
- In thesis terms, this design supports cleaner evidence chains.

### Y18. Threats To Validity In Comparative Claims

- The project can compare a smart route against baseline routes on specific requests and through local benchmark machinery.
- However, the repo does not contain one universal checked-in win-rate proving superiority over OSRM or ORS everywhere.
- In plain English, local comparison capability is strong, but universal external superiority is not fully precomputed inside the repo.
- This is important to state clearly.
- The threat would be overgeneralization from case studies or limited benchmark sets.
- The thesis should therefore present smart-versus-baseline claims with careful scope language.
- That caution increases credibility.

### Y19. Threats To Validity In Live Data

- Live data can be stale, unavailable, biased, or incomplete.
- The repo addresses this with strict gating, host allow-lists, signatures, freshness checks, and dashboards.
- In plain English, the system knows live data is powerful but risky.
- The threat remains that good validation cannot fully guarantee real-world truth.
- The thesis should therefore distinguish validated input from guaranteed perfect reality.
- That is a mature position.
- It also matches the repository's design tone.

### Y20. Threats To Validity In Calibration

- Calibrated tables can age.
- Context buckets can be sparse.
- Some profile effects may rely on proxy observations rather than fully observed ground truth.
- The repo acknowledges this through drift checks, raw-data preservation, and build scripts.
- In plain English, calibration is treated as maintainable engineering, not eternal truth.
- The thesis should emphasize that asset freshness and recalibration matter.
- That framing is more defensible than pretending the calibrations are final.

### Y21. Threats To Validity In Scope

- The repo is freight-oriented, UK-oriented, and tuned around the included asset families.
- It is not a universal passenger-routing engine.
- It is not a globally calibrated routing oracle.
- In plain English, its strength comes from bounded ambition.
- The threat is that readers may extrapolate too widely from a strong bounded system.
- The thesis should resist that temptation.
- Strong bounded claims are better than weak universal ones.

### Y22. Why These Tradeoffs Still Form A Strong Thesis Project

- The project combines graph search, multi-objective routing, uncertainty, physical modeling, live-data governance, artifact integrity, and a capable frontend.
- It also distinguishes academic baselines from engineering modifications instead of blurring them.
- In plain English, it is a real systems project with both algorithmic and operational depth.
- The tradeoffs are visible and discussable.
- That is a strength, not a weakness.
- A thesis is stronger when it can explain why decisions were made and what was sacrificed.
- This repository gives plenty of such material.

## Appendix Z: What The User Can Actually Do And What Happens Under The Hood

This appendix is written in a deliberately user-centered way.

It answers a simple but important thesis question:

what can the operator actually do in the frontend, and what backend machinery is triggered when they do it?

### Z1. Set Origin And Destination On The Map

- The user can click or place origin and destination points on the map.
- This triggers frontend state updates, pin management, viewport logic, and later request construction.
- Backend-side, those coordinates become the seed for graph snapping and feasibility checks.
- In plain English, this is not just picking two dots.
- It is choosing the geographic problem the graph, OSRM, and all later models will solve.
- A route can change dramatically if these points move even slightly near different corridors or toll crossings.

### Z2. Add Intermediate Stops

- The user can add stop-like waypoints or multi-stop duty inputs depending on the workflow.
- This changes the problem from one direct OD path into a chained or constrained sequence.
- Backend-side, the multileg engine or duty-chain planner composes multiple route computations.
- In plain English, the user is now asking for an itinerary rather than a single trip.
- This changes route aggregation, timing, and potentially cost structure.
- It also increases the importance of manifests and artifact outputs because multi-stop studies are harder to inspect manually.

### Z3. Change Vehicle Type

- The user can switch between built-in vehicle profiles such as van, rigid HGV, artic HGV, and EV HGV.
- This alters terrain physics parameters, fuel behavior, emissions interpretation, and sometimes risk bucketing.
- Backend-side, vehicle profiles affect fuel-energy modeling, terrain multipliers, and contextual regime resolution.
- In plain English, the same road can look different to a lighter or heavier vehicle.
- This is one of the most direct ways a user changes the physics of the route.
- It is therefore a major thesis-relevant control, not a cosmetic label.

### Z4. Create A Custom Vehicle

- The user can define a custom vehicle in the dev-tool surface.
- That changes the physical assumptions even more deeply because mass, drag, efficiency, and related parameters can be adjusted.
- Backend-side, custom vehicles are validated and then treated as first-class profiles in later route modeling.
- In plain English, the user is teaching the model about a new fleet archetype.
- This can change energy use, route cost, emissions, and terrain response substantially.
- It is one of the most research-powerful features in the frontend.

### Z5. Change Objective Weights

- The user can change time, money, and CO2 weighting sliders.
- Backend-side, these weights influence scalar selection after feasible candidates exist.
- They do not change which roads are physically present in the graph.
- But they do change which route is highlighted among trade-off options.
- In plain English, this is the user's way of saying what kind of compromise they care about most.
- This is central to explaining the difference between Pareto exploration and single highlighted route selection.

### Z6. Switch Between Single-Route And Pareto Modes

- The user can request one route or a full candidate frontier.
- Backend-side, this changes whether the result is collapsed through a selector or exported as a multi-route trade-off set.
- It also changes what the frontend renders: a single card or a richer chart-and-candidate workflow.
- In plain English, the user chooses whether they want a recommendation or a trade-off landscape.
- This is a major conceptual choice in the whole system.
- It is one of the best demonstrations that the project is multi-objective by design.

### Z7. Choose Pareto JSON Versus Pareto Stream

- The user can receive Pareto results as one JSON payload or as a live stream of events.
- Backend-side, the route-compute core is similar, but response delivery and observability differ.
- Streaming exposes progress and partial visibility during longer computations.
- In plain English, the user chooses whether to wait silently or watch the frontier appear.
- This does not change the science of the route set directly.
- It changes the interaction model and transparency around that science.

### Z8. Switch Scenario Mode

- The user can choose no-sharing, partial-sharing, or full-sharing.
- Backend-side, this selects different scenario-mode transformations from the context-conditioned profile asset.
- This can change duration pressure, cost uplift, emissions multiplier, and uncertainty-related context.
- In plain English, the user is choosing an operational-coordination assumption.
- This is one of the strongest policy-style controls in the system.
- It makes the router suitable for comparative scenario work, not only optimization.

### Z9. Run Scenario Compare

- The user can run the explicit scenario compare workflow rather than choosing only one mode.
- Backend-side, multiple scenario evaluations are performed for the same trip context.
- The result is not only a route.
- It is a comparison across operational assumptions.
- In plain English, the user is asking "how would cooperation levels change the answer?"
- This is one of the clearest thesis-worthy features in the frontend.

### Z10. Choose A Departure Time

- The user can specify a concrete departure time.
- Backend-side, this affects local slot resolution, day-kind interpretation, departure profiles, and stochastic regime resolution.
- The same road geometry can therefore score differently at different times.
- In plain English, a route at 07:30 is not treated the same as a route at 14:00.
- This is important because time is not just an output in the system.
- Time is also an input.

### Z11. Run Departure Optimization

- The user can ask the system to search for the best departure within a time window.
- Backend-side, the route is recomputed over multiple slots using the same modeling layers.
- This is computationally richer than choosing one departure time.
- In plain English, the system is exploring "when should I leave?" rather than only "which road should I take?"
- This changes the problem from route ranking to schedule-aware route evaluation.
- It is a major extension over baseline routing tools.

### Z12. Enable Terrain Awareness

- The user can work with terrain-aware modeling where supported.
- Backend-side, DEM sampling, coverage checks, and terrain physics are activated.
- This can change duration, energy use, emissions, and sometimes route ordering.
- In plain English, hills and slopes start to matter.
- This is especially important for heavy freight and EV-like energy sensitivity.
- It is one of the strongest physical-model differentiators in the project.

### Z13. Change Weather Or Incident Inputs

- The user can supply weather profiles and, in controlled scenarios, incident settings.
- Backend-side, weather multipliers and incident behavior alter duration pressure and related modeled effects.
- Synthetic incidents remain separate from strict live truth.
- In plain English, the user can ask "what if conditions get worse?"
- This makes the system useful for contingency analysis.
- It is not just a fair-weather router.

### Z14. View Baseline Comparisons

- The user can request OSRM and ORS baseline comparisons.
- Backend-side, dedicated baseline endpoints produce reference routes and the frontend compares them with the smart route.
- This does not merely change display.
- It changes the analytical frame of the result.
- In plain English, the user can ask "what did the smart pipeline add compared with a simpler route provider?"
- This is vital for thesis evaluation and honest benchmarking.

### Z15. Inspect The Pareto Chart

- The user can click or inspect points in objective space.
- Backend-side, the route set already exists; the chart is how the user explores it.
- Selection from the chart changes which candidate's segment breakdown, overlays, and details are foregrounded.
- In plain English, the user can look at trade-offs rather than blindly trust the first recommendation.
- This is important for interpretability and multi-objective decision support.
- It makes the frontier tangible.

### Z16. Inspect Segment Breakdowns

- The user can drill into segment-level route details.
- Backend-side, this depends on route segmentation and the modeled per-segment metrics built during option construction.
- This helps explain where energy, tolls, or terrain effects come from.
- In plain English, it lets the user ask "which parts of the journey caused this score?"
- That moves the UI from route viewing into route explanation.
- It is especially useful in thesis illustrations.

### Z17. Inspect Counterfactuals

- The user can view counterfactual route information where available.
- Backend-side, counterfactual metrics are derived from the modeled route rather than fetched from a separate provider.
- This helps explain marginal effects of individual modeled layers.
- In plain English, it helps answer "what would this route look like if this factor changed or disappeared?"
- This supports mechanism-level discussion rather than only output comparison.
- It is one of the repo's more explainable-analysis features.

### Z18. Use The Duty-Chain Planner

- The user can plan a duty chain with multiple stops.
- Backend-side, the multileg engine aggregates leg results and produces chain-level outputs.
- This changes the system from a trip router into a work-sequence planner.
- In plain English, the user can ask for a day's sequence, not just one route.
- That broadens the operational significance of the tool considerably.
- It also makes artifacts and reports even more valuable because chain results are denser.

### Z19. Run Batch Pareto Studies

- The user can execute batch Pareto studies through the UI or API.
- Backend-side, the system iterates route analysis over many OD pairs.
- This is a study workflow rather than a single operational action.
- In plain English, the user can test the system on a set of journeys rather than one case.
- This supports benchmarking, sensitivity studies, and dissertation evidence gathering.
- It is one of the most research-useful capabilities in the whole frontend.

### Z20. Import CSV Case Lists

- The user can import cases in CSV-style form.
- Backend-side, those rows are converted into batch requests.
- This means route studies can start from tabular business or research data.
- In plain English, the app can absorb a spreadsheet and turn it into a routing experiment.
- This lowers the cost of building larger evaluation sets.
- It is therefore operationally and academically useful.

### Z21. Use The Experiment Manager

- The user can save and replay experiments.
- Backend-side, study definitions are persisted through the experiment store.
- This matters because analytical work often happens in sessions over time, not in one sitting.
- In plain English, the system can remember what the user was investigating.
- This helps turn ad hoc exploration into a repeatable workflow.
- It is a strong support feature for thesis writing and benchmarking.

### Z22. Inspect Run Manifests And Provenance

- The user can retrieve run manifests, scenario manifests, provenance, and signatures.
- Backend-side, these are read from persisted artifact families written during earlier computations.
- In plain English, the user can inspect the paperwork behind a result.
- This is one of the most unusual and valuable features in the whole project.
- It supports verification, archiving, and thesis appendix preparation.
- It turns routes into evidence objects.

### Z23. Verify Signatures

- The user can verify the integrity of a manifest or payload.
- Backend-side, HMAC verification checks the supplied payload, signature, and secret.
- In plain English, the user can ask whether a result appears to have been altered.
- This does not make the system a full security platform.
- But it does materially strengthen artifact trust.
- That is unusually strong for a research-routing repo.

### Z24. Inspect Live-Call Diagnostics

- The user can retrieve live-call traces for a request in development contexts.
- Backend-side, this comes from the live-call trace subsystem rather than the routing result itself.
- In plain English, the user can inspect what external data dependencies were actually touched.
- This is especially useful when a route fails due to stale or incomplete live context.
- It makes live data usage visible rather than mysterious.
- That is a major observability strength.

### Z25. View Oracle Quality

- The user can inspect the health of upstream sources through the oracle-quality dashboard.
- Backend-side, source-quality observations are aggregated and served separately from route results.
- In plain English, the user can inspect the trustworthiness of the data pipeline itself.
- This is conceptually distinct from asking which route is best.
- It supports a more mature operational understanding of the system.
- It also strengthens the thesis's live-data governance discussion.

### Why This Appendix Matters

- A long thesis report can become too code-centric if it only follows files and formulas.
- This appendix pulls the focus back to user action and analytical consequence.
- It shows that every important frontend control corresponds to real backend machinery.
- It also shows that the frontend is not superficial.
- It is how the system's modeling power becomes usable, inspectable, and comparable.
- That matters for both product understanding and dissertation writing.

## Appendix AA: What Actually Changes A Route Answer

This appendix is about causal leverage inside the system.

Many route controls look similar in a user interface.

They are not similar in modeling effect.

Some change the candidate set.

Some change only ranking.

Some change physics.

Some change data governance.

That distinction is important in a thesis.

### AA1. Origin And Destination Change Almost Everything

- Moving the origin or destination changes graph snapping, corridor choice, toll exposure, terrain profile, departure corridor bucket, and stochastic regime context.
- In plain English, geography is the deepest driver of route identity.
- This is why tiny map changes can shift the route onto a different motorway, crossing, or toll segment.
- In thesis terms, coordinate choice is the root causal input behind most other metrics.
- It changes more than any slider can.

### AA2. Stops Change The Problem Class

- Adding stops does not merely perturb a single-route score.
- It changes the combinational structure of the journey.
- Backend-side, this means multileg or duty-chain logic composes several route computations.
- In plain English, the system stops solving one path problem and starts solving a sequence problem.
- That is a major structural change, not a small preference tweak.

### AA3. Vehicle Type Changes Physics And Cost

- Vehicle type changes mass, drag area, rolling resistance assumptions, efficiency, and sometimes regenerative behavior.
- That affects terrain sensitivity, fuel use, emissions, and cost.
- In plain English, a van and an artic HGV do not experience the same route in the model.
- This is why vehicle selection is not a presentational choice.
- In thesis terms, it changes the physical model itself.

### AA4. Custom Vehicle Profiles Change The Model Even More

- A custom vehicle can move multiple physical parameters away from built-in defaults.
- That means user-defined vehicles can materially alter route energy and emissions behavior.
- In plain English, a custom vehicle is one of the strongest levers available to a user.
- This can change which route is favored under terrain or energy-aware conditions.
- In thesis terms, it is a high-impact calibration input.

### AA5. Objective Weights Change Selection More Than Generation

- Weight sliders mostly affect the final highlighted route rather than the underlying graph-search candidate generation.
- This is because frontier creation happens before scalar recommendation.
- In plain English, weights usually change "which candidate wins" more than "which candidates exist."
- That distinction matters for interpreting why the Pareto set can stay stable while the recommended route changes.
- In thesis terms, weights are largely a selection-layer control.

### AA6. Selection Math Profile Changes Highlight Logic

- Switching from weighted sum to VIKOR or a modified selector can change the recommended route even with identical frontier inputs.
- This happens because the profiles disagree about what a good compromise looks like.
- In plain English, the route can change because the decision rule changed, not because the road network changed.
- This is essential to explain in a dissertation.
- It shows why the repo separates academic baselines from modified profiles.

### AA7. Pareto Method Changes Exported Alternatives

- Dominance filtering and epsilon-constraint filtering do not always export the same route set.
- This means the visible frontier itself can change before selection occurs.
- In plain English, the user's definition of "acceptable trade-off" matters.
- Epsilon constraints can keep routes that pure dominance would treat differently.
- In thesis terms, this is a candidate-set shaping control, not just a ranking control.

### AA8. Departure Time Changes Context

- Departure time alters local hour bucket, day kind, departure-profile lookup, and sometimes stochastic regime mapping.
- The same route can therefore change ETA, cost, and uncertainty without any change in geometry.
- In plain English, timing changes the modeled world around the route.
- This is why schedule-aware routing is more than an extra timestamp field.
- In thesis terms, departure time is one of the strongest context levers in the system.

### AA9. Departure Optimization Changes The Search Space

- Departure optimization samples multiple departure times rather than scoring just one.
- The result can therefore differ from choosing one departure manually.
- In plain English, it widens the search from route choice alone to route-time combinations.
- This means the system is optimizing over more than one dimension of control.
- In thesis terms, it is a problem-class expansion.

### AA10. Scenario Mode Changes Policy Assumptions

- No-sharing, partial-sharing, and full-sharing alter modeled route multipliers through the scenario profile asset.
- These changes affect duration pressure, emissions multiplier, cost uplift, and uncertainty metadata.
- In plain English, the route answer changes because the assumed operating mode changed.
- This is not a pathfinding-only effect.
- In thesis terms, scenario mode changes the governing operational assumption behind the route.

### AA11. Scenario Context Quality Changes Whether Scenario Effects Are Allowed

- Even if a scenario mode is chosen, stale or incomplete live scenario evidence can block the route under strict mode.
- In plain English, not every requested scenario route is allowed to exist.
- The quality of the scenario input can determine whether the backend applies the policy layer at all.
- This is a governance effect rather than an optimization effect.
- In thesis terms, data quality can change route availability, not only route value.

### AA12. Terrain Profile Changes Physical Pressure

- Terrain-aware mode changes duration and emissions behavior using grade-sensitive approximations.
- Flat, rolling, and hilly intent affects how strongly terrain enters the model.
- In plain English, the user can tell the system how seriously to treat slope-driven penalty.
- This is not only aesthetic.
- It changes the physical cost surface of the route.

### AA13. Terrain Coverage Can Change Whether The Route Exists

- Under strict UK terrain policy, insufficient DEM coverage can block route production.
- In plain English, terrain is not always "best effort."
- Sometimes the absence of terrain evidence is enough to stop the route entirely.
- This is a strong form of model governance.
- In thesis terms, coverage thresholds are route-validity levers.

### AA14. Weather Changes Duration Pressure And Incident Sensitivity

- Weather profiles alter speed and incident multipliers.
- That can change both mean route metrics and stochastic behavior.
- In plain English, the same geometry becomes slower or riskier in worse weather.
- This is a context change, not a road-network change.
- In thesis terms, weather is part of the external operating state.

### AA15. Incident Simulation Changes Scenario Pressure In Experimental Modes

- Synthetic incidents change modeled disruption pressure in controlled analysis.
- They are not treated as live truth in strict production paths.
- In plain English, they answer "what if a disruption occurred here?"
- This can change route rankings, especially where alternatives differ in fragility.
- In thesis terms, incident simulation is an experimental lever.

### AA16. Fuel Price Changes Monetary Ranking

- Fuel price does not change the road geometry directly.
- It changes how expensive energy consumption looks.
- In plain English, expensive fuel can make longer or hillier routes look worse in money terms.
- This can flip recommendation ranking where cost is weighted strongly.
- In thesis terms, fuel is a major lever on the monetary objective.

### AA17. Carbon Schedule Changes Environmental Monetization

- Carbon schedule changes the price attached to emissions, not the raw emissions quantity itself.
- In plain English, the route may emit the same CO2 but look more or less costly under different carbon pricing assumptions.
- This matters if monetary interpretation includes carbon cost.
- It can change compromise ranking when money is an active concern.
- In thesis terms, it is a policy-cost lever rather than a physics lever.

### AA18. Toll Topology And Tariff Data Change Money Locally

- Toll-aware behavior depends on whether the route intersects tolled assets and how tariffs are priced.
- In plain English, two similar routes can differ sharply in money if one crosses chargeable infrastructure.
- This can alter frontier shape, not just the highlighted route.
- Toll confidence and proxy pricing also affect how strongly these differences are trusted.
- In thesis terms, toll data changes route economics spatially and discretely.

### AA19. Risk Aversion Changes Robust Ranking

- Risk aversion affects how much the robust objective penalizes tail risk above mean behavior.
- In plain English, the user or config can choose whether bad tails matter a little or a lot.
- This can change which route looks safest under uncertainty.
- It may leave deterministic rankings unchanged while robust rankings shift.
- In thesis terms, it is a preference lever over uncertainty interpretation.

### AA20. Sigma And Sample Count Change Uncertainty Detail

- Sigma affects uncertainty spread and sample count affects summary stability.
- In plain English, larger sigma makes the modeled world more volatile, while more samples make the estimates less noisy.
- These do not usually change geometry directly.
- They change how the route looks under risk.
- In thesis terms, they are uncertainty-shape controls.

### AA21. Corridor Bucket Changes Contextual Calibration

- Corridor buckets influence scenario matching, departure behavior, and stochastic regime resolution.
- In plain English, the system treats different UK corridors as behaviorally different.
- This matters because a motorway-heavy southern corridor is not assumed identical to every other corridor.
- Corridor identity can therefore change route multipliers and uncertainty structure.
- In thesis terms, this is one reason UK-specific calibration matters.

### AA22. Day Kind Changes Time Semantics

- Weekday, weekend, and holiday modes can map to different departure and stochastic contexts.
- In plain English, a route on a bank holiday is not evaluated as if it were a routine weekday morning.
- This changes contextual multipliers and sometimes uncertainty.
- It is a quiet but meaningful lever.
- In thesis terms, it supports calendar-aware realism.

### AA23. Road-Class Mix Changes Stochastic Regime Resolution

- The uncertainty model derives dominant road buckets such as motorway-heavy or trunk-heavy from route composition.
- In plain English, routes with different road-class mix can end up in different uncertainty regimes even at similar distances.
- This matters because not all roads carry the same volatility character.
- The result can change q95 and CVaR materially.
- In thesis terms, route composition changes risk, not just geometry.

### AA24. Graph Search Budgets Change Frontier Breadth

- Tight search budgets can reduce route diversity.
- More generous budgets can uncover more alternative corridors.
- In plain English, search patience affects the quality of the candidate set.
- This is one reason benchmark interpretation and budget tuning matter.
- In thesis terms, frontier quality is partly a computational-budget question.

### AA25. Rescue Logic Changes Whether Difficult Long Routes Succeed

- Rescue settings determine whether the backend escalates beyond a failed first search.
- In plain English, the route may exist under rescue settings and fail without them.
- This is a direct engineering effect on route availability.
- It matters particularly for long or awkward OD pairs.
- In thesis terms, rescue logic is a practical route-existence lever.

### AA26. Route Cache Changes Latency More Than Semantics

- Route caching usually changes how quickly a repeat request returns rather than the route meaning itself.
- In plain English, it affects performance more than modeling.
- However, cache presence does matter when interpreting benchmarks or demos.
- A cached result can make the system look much faster.
- In thesis terms, cache state is a performance confounder rather than a route-science confounder.

### AA27. Live Data Freshness Changes Availability

- Freshness windows can determine whether a route is allowed under strict mode.
- In plain English, the route may be blocked because the data is too old, even if the network itself is fine.
- This is a quality-governance effect, not an optimization effect.
- It matters because live realism is one of the repo's defining promises.
- In thesis terms, freshness is a route-validity lever.

### AA28. Source Allow-Lists Change Which Data Paths Are Legal

- Allowed-host settings do not change the route mathematically by themselves.
- They change which external data paths are permissible.
- In plain English, they constrain where the system is allowed to get truth from.
- This can indirectly change route availability if a source is configured but not allowed.
- In thesis terms, it is a governance lever on provenance.

### AA29. Signature Requirements Change Trust Threshold

- Signature requirements decide whether unsigned artifacts may be used.
- In plain English, they change the trust bar for live-backed or published assets.
- This can block route production if otherwise usable data lacks integrity proof.
- That is a deliberate choice.
- In thesis terms, integrity policy can change route availability.

### AA30. Frontend Degrade Steps Change User Experience, Not Core Math

- The degrade ladder changes how the UI responds to long or failing requests.
- In plain English, it changes whether the user gets a stream, a JSON frontier, or a simpler single-route attempt.
- The underlying backend models remain the same family.
- What changes is how much of that richness the user is still able to receive under pressure.
- In thesis terms, it is a presentation-and-resilience lever rather than an algorithmic one.

### Why This Appendix Matters

- Complex systems are easier to explain when their causal levers are separated.
- This appendix helps distinguish geometry changes, context changes, physics changes, governance changes, and selection-rule changes.
- That makes later thesis discussion much cleaner.
- A route can change because the road changed, because the policy assumption changed, because the data got fresher, or because the selector changed.
- Those are not the same kind of change.
- The repository is rich enough that this distinction matters.

## Appendix AB: Extended Plain-English Glossary For Thesis Writing

This appendix goes beyond the short glossary earlier in the report.

It is meant to help translate technical repository vocabulary into dissertation-friendly language.

Each term is explained in a way that preserves technical meaning without assuming specialist prior knowledge.

### `route graph`

- The route graph is the backend's custom network representation of UK roads.
- In plain English, it is the searchable road skeleton the smart router uses to discover candidate corridors.
- It is different from raw map tiles and different from the final route geometry shown in the UI.
- In thesis terms, it is the structure that makes graph-led path diversity possible.

### `graph warmup`

- Graph warmup is the startup process of loading and preparing that large custom road graph.
- In plain English, it is the system getting its road-search brain ready before serving strict route requests.
- Warmup exists because the graph is too large to treat as a trivial instant-on object.
- In thesis terms, it is a readiness phase, not just a loading annoyance.

### `giant component`

- The giant component is the largest connected piece of the graph.
- In plain English, it is the main body of roads that can all reach one another.
- If that component is too small relative to the graph, the graph is considered fragmented.
- In thesis terms, giant-component checks are a graph-quality safeguard.

### `coverage gap`

- A coverage gap means the graph or another data layer does not adequately cover the requested geography.
- In plain English, it means the system does not have enough trustworthy support for this place.
- This can apply to routing, terrain, or other context layers.
- In thesis terms, it is a scope or support failure, not a random glitch.

### `strict mode`

- Strict mode means the system prefers to fail rather than guess when important inputs are missing or invalid.
- In plain English, it is the "do not bluff" mode.
- This affects graph readiness, live data, terrain, and other governed subsystems.
- In thesis terms, strict mode is a methodology choice about trust.

### `live data`

- Live data means information fetched or refreshed from configured external or published sources during runtime use.
- In plain English, it is the up-to-date contextual evidence the router tries to use.
- Examples include scenario coefficients, fuel prices, bank holidays, and terrain tiles.
- In thesis terms, live data is what makes the model context-sensitive and temporally aware.

### `fail-closed`

- Fail-closed means the system blocks the route or subsystem when required evidence is absent.
- In plain English, it chooses "no answer" over "unsafe answer."
- This is the opposite of silently inventing fallback behavior.
- In thesis terms, fail-closed behavior is a trust-preserving design principle.

### `scenario mode`

- Scenario mode is the user's choice among no-sharing, partial-sharing, and full-sharing assumptions.
- In plain English, it is the chosen coordination or collaboration assumption for the trip.
- The backend maps that mode into context-conditioned multipliers.
- In thesis terms, scenario mode is a policy-style modeling layer.

### `no sharing`

- No sharing means the route is evaluated under the least collaborative operational assumption of the three built-in modes.
- In plain English, it is the baseline case where coordination benefits are not assumed.
- This does not mean "no route."
- It means "no special cooperation uplift."

### `partial sharing`

- Partial sharing means some coordination benefit is assumed, but not the maximum available in the modeled profiles.
- In plain English, it is the middle case between isolated operation and highly coordinated operation.
- It usually reduces pressure relative to no sharing, but not as much as full sharing.
- In thesis terms, it is the moderate policy scenario.

### `full sharing`

- Full sharing means the route is evaluated under the most coordinated of the three built-in modes.
- In plain English, it is the strongest assumed benefit from collaboration or shared operational behavior.
- Local scenario-profile evidence shows its effect scale is pushed closest to the floor among the three modes.
- In thesis terms, it is the optimistic coordination scenario, not a guaranteed real-world state.

### `Pareto set`

- The Pareto set is the collection of candidate routes that are not dominated on the chosen objectives.
- In plain English, it is the trade-off set where every remaining route is good at something.
- No single route in the set is strictly better than another on every objective at once.
- In thesis terms, it is the core multi-objective output of the system.

### `Pareto dominance`

- Pareto dominance means one route is at least as good on all objectives and better on at least one.
- In plain English, a dominated route is hard to defend because another route beats it overall.
- The backend uses this logic directly.
- In thesis terms, it is the mathematical filter behind the frontier.

### `epsilon constraint`

- Epsilon constraint means the system keeps routes that satisfy configured objective ceilings.
- In plain English, it is a "stay within these limits" rule.
- This can produce a different exported set from pure dominance filtering.
- In thesis terms, it is an alternative way of defining acceptable trade-off routes.

### `backfill`

- Backfill means adding extra ranked routes when the strict frontier is too small.
- In plain English, it is the system making the option set more usable for a human.
- Backfill does not pretend those extra routes are identical to strict frontier points.
- In thesis terms, it is a product-driven extension of a mathematically strict result.

### `selection profile`

- The selection profile is the rule used to pick one highlighted route from an already-feasible candidate set.
- In plain English, it is how the system chooses its final recommendation.
- This is separate from how candidates were generated.
- In thesis terms, it is crucial not to confuse selection with search.

### `weighted sum`

- Weighted sum is the simplest scalar route selector in the project.
- In plain English, it turns several objectives into one number using preference weights.
- It is easy to understand but can miss richer compromise structure.
- In thesis terms, it serves as an academic-style reference baseline.

### `VIKOR`

- VIKOR is a compromise-ranking method balancing group utility and worst regret.
- In plain English, it tries to find a route that is broadly good without being badly weak on one dimension.
- The repo keeps both academic and modified VIKOR-based selectors.
- In thesis terms, it is one of the main cited academic building blocks.

### `knee point`

- A knee point is a route near a high-curvature trade-off region where improving one objective further starts hurting the others more sharply.
- In plain English, it often feels like a balanced compromise route.
- The repo uses a simple knee-penalty proxy rather than a full formal knee-detection system.
- In thesis terms, knee preference is an engineering interpretation layer.

### `entropy reward`

- Entropy reward means rewarding routes that improve several objectives together rather than winning narrowly on one.
- In plain English, it slightly favors broad improvement over one-dimensional improvement.
- This is part of the modified selectors.
- In thesis terms, it is an engineering addition inspired by information-theoretic balance ideas.

### `A* heuristic`

- The A* heuristic is a lower-bound estimate of remaining route cost used to guide search.
- In plain English, it helps the graph search focus on promising directions.
- The repo's implementation cites Hart et al. (1968).
- In thesis terms, it is a classic search acceleration method embedded in the custom graph layer.

### `Yen K-shortest`

- Yen K-shortest refers to a family of algorithms for enumerating multiple good paths rather than only one shortest path.
- In plain English, it is part of how the system finds route alternatives.
- The repo adapts this idea with strict budgets and rescue logic.
- In thesis terms, it is one of the clearest academic ancestors in candidate generation.

### `route rescue`

- Route rescue is the backend's escalated search behavior when an initial graph search is too brittle or too constrained.
- In plain English, it is the system trying a smarter or more generous second attempt before giving up.
- Rescue is a practical extension, not a textbook primitive.
- In thesis terms, it is an engineering response to hard real-world route cases.

### `scenario context`

- Scenario context is the structured summary of geography, time, vehicle, weather, and road mix used to choose a scenario profile.
- In plain English, it is the description of the trip situation.
- The system uses that situation to look up calibrated scenario effects.
- In thesis terms, context is what makes sharing-mode effects specific rather than generic.

### `corridor bucket`

- A corridor bucket is a geography-aware context class used in calibration.
- In plain English, it is the system's way of saying "this trip belongs to this broad regional movement type."
- Corridor buckets influence scenario, departure, and uncertainty behavior.
- In thesis terms, they are one of the main mechanisms that make the repo UK-specific.

### `departure profile`

- A departure profile is the time-sensitive multiplier model used to adjust route behavior by departure context.
- In plain English, it tells the system how timing changes expected route pressure.
- It is used directly by the departure optimizer and route builder.
- In thesis terms, it is the engine behind schedule-aware routing.

### `stochastic regime`

- A stochastic regime is a calibrated uncertainty pattern tied to context.
- In plain English, it tells the system what kind of variability to expect for this sort of trip.
- Different regimes can have different spreads, correlations, and factor behavior.
- In thesis terms, regimes make uncertainty context-aware rather than one-size-fits-all.

### `quantile`

- A quantile is a cutoff in the distribution of sampled outcomes.
- In plain English, q95 is the level below which about 95 percent of sampled results lie.
- The repo uses q50, q90, and q95 heavily.
- In thesis terms, quantiles are core uncertainty summary statistics.

### `CVaR`

- CVaR is the average behavior in the bad tail beyond a chosen quantile threshold.
- In plain English, it says how bad things are on average once you are already in the nasty tail.
- This is more conservative than using q95 alone.
- In thesis terms, CVaR is one of the strongest risk measures in the repo.

### `robust score`

- The robust score is the uncertainty-aware utility after tail-risk penalization.
- In plain English, it is the "safe compromise" score under uncertainty.
- A route can have a good mean utility and still a worse robust score if its tail is ugly.
- In thesis terms, robust score is what turns uncertainty summaries into decision-relevant ranking.

### `risk aversion`

- Risk aversion controls how strongly the robust objective penalizes bad tails.
- In plain English, it is the dial for how much the user or config fears nasty outcomes.
- This matters most when routes differ in volatility rather than average.
- In thesis terms, it is the preference bridge from stochastic summary to decision rule.

### `terrain coverage`

- Terrain coverage is the fraction of the route for which the backend has usable elevation evidence.
- In plain English, it is how much of the journey the terrain model can actually see.
- Under strict UK settings, insufficient coverage can block the route.
- In thesis terms, coverage is a validity condition for terrain-aware routing.

### `counterfactual`

- A counterfactual is a "what would change if this modeled factor were different?" view of the route.
- In plain English, it is the system's explanatory alternate-world comparison.
- Counterfactuals help separate mechanism from total score.
- In thesis terms, they improve route explainability.

### `baseline route`

- A baseline route is a simpler reference answer used for comparison against the smart pipeline.
- In plain English, it is the control case.
- The repo supports both OSRM and ORS-style baselines.
- In thesis terms, baselines are essential for comparative evaluation.

### `manifest`

- A manifest is a stored summary of a run, its settings, and related artifact context.
- In plain English, it is the official record for one analytical result.
- Manifests make route runs reproducible and inspectable.
- In thesis terms, they are central evidence objects.

### `provenance`

- Provenance is the record of where the run's important inputs came from.
- In plain English, it is the source-history of the result.
- Provenance is especially important for live-backed and published assets.
- In thesis terms, it supports claims about traceability and trust.

### `signature`

- A signature is the integrity check attached to a manifest or related payload.
- In plain English, it helps detect whether stored result material was altered.
- The repo uses HMAC-based signing.
- In thesis terms, this is an integrity feature rather than a full security architecture.

### `oracle quality`

- Oracle quality is the repository's term for upstream source-health monitoring.
- In plain English, it is the scorecard for how healthy the input feeds are.
- It tracks stale counts, failures, and latency separately from route quality.
- In thesis terms, it is a governance layer over external data.

### `live-call trace`

- The live-call trace is the per-request record of expected and observed upstream calls.
- In plain English, it is the backend's memory of what external data work happened.
- This makes live dependency visible to users and developers.
- In thesis terms, it is a major observability artifact.

### `batch Pareto`

- Batch Pareto means running Pareto route analysis over many OD pairs.
- In plain English, it is the study-scale version of trade-off routing.
- This is used for evaluation, not just one-off route recommendation.
- In thesis terms, it enables broader empirical evidence.

### `duty chain`

- A duty chain is a multi-stop route sequence treated as one operational plan.
- In plain English, it is a shift or run of work rather than one journey.
- The backend composes multiple legs to build it.
- In thesis terms, it extends the project into operational scheduling territory.

### `tutorial mode`

- Tutorial mode is the guided walkthrough embedded in the frontend.
- In plain English, it teaches a user how to use the tool without leaving the app.
- It covers real workflows rather than generic onboarding.
- In thesis terms, it supports the usability side of a complex research system.

### Why This Appendix Matters

- A thesis report often has to serve readers with different technical backgrounds.
- This appendix helps the same document work for supervisors, examiners, developers, and operators.
- It keeps the report readable without dumbing the system down.
- That is important because the repository combines routing, optimization, live data, risk, and artifact governance in one place.

## Appendix AC: What It Means To Beat OSRM Or ORS In This Repository

This appendix is deliberately careful.

The repo can compare smart routes against baseline routes.

It can also benchmark and inspect those comparisons in local workflows.

What it does not provide as a checked-in fact is one universal, externally validated percentage by which it beats every baseline in all freight situations.

That distinction should be explicit in a thesis.

### AC1. There Are Several Different Meanings Of "Better"

- A route can be better in time.
- It can be better in money.
- It can be better in emissions.
- It can be better in robust tail risk.
- It can be better in explainability.
- It can be better in auditability.
- The project often improves some of these dimensions simultaneously, but not always.
- In thesis terms, "better" must be tied to a named objective or evaluation context.

### AC2. Baselines In This Repo Are Not Purely Raw Provider Outputs

- The repo includes realism multipliers for baseline duration and distance.
- This matters because a fair comparison should not depend on the baseline being unrealistically optimistic.
- In plain English, the baseline is adjusted to be a more believable operational reference.
- This is important to state because some readers may assume the baseline is a raw provider response.
- In thesis terms, the repo is trying to compare against a plausible reference, not a straw man.
- The exact multipliers are local configuration choices and should be reported as such.

### AC3. The Smart Router Can Beat Baselines By Generating Better Candidate Diversity

- Plain shortest-path systems often commit early to one dominant path family.
- The smart router first searches for multiple plausible corridors using the custom graph.
- In plain English, it tries not to get trapped in the first obvious corridor.
- This matters because some trade-off routes are only visible if candidate generation is diversity-aware.
- In thesis terms, better candidate breadth can improve frontier quality before any cost model is applied.
- This is one of the main conceptual advantages over provider-only routing.

### AC4. The Smart Router Can Beat Baselines By Scoring More Realistically

- The smart route is not judged only by raw distance or nominal ETA.
- It incorporates terrain effects, vehicle physics, toll logic, fuel prices, emissions, carbon cost, and scenario pressure.
- In plain English, it knows more about the route than a basic shortest-path provider does.
- That can change which route is truly better for freight.
- In thesis terms, realism layers can produce a more operationally meaningful winner than plain shortest path.
- This does not mean the geometry is always radically different.

### AC5. The Smart Router Can Beat Baselines By Being Risk-Aware

- Baselines are often deterministic.
- The smart router can summarize uncertainty through q95, CVaR, and robust score.
- In plain English, it can reject a route that looks good on average but is too ugly in the tail.
- That is an important kind of improvement for operational planning.
- In thesis terms, risk-aware superiority is different from mean-value superiority.
- The report should keep those comparison types separate.

### AC6. The Smart Router Can Beat Baselines By Being Scenario-Aware

- A baseline route usually ignores sharing-policy context.
- The smart router can evaluate the same trip under no-sharing, partial-sharing, and full-sharing assumptions.
- In plain English, it can answer a richer question.
- That means "better" sometimes refers to better policy sensitivity rather than just lower ETA.
- In thesis terms, scenario awareness is a capability improvement even when raw path geometry looks similar.
- This is important in discussing the novelty of the project.

### AC7. The Smart Router Can Beat Baselines By Being Departure-Aware

- If departure context matters, then a baseline computed for one moment may not reflect the best operational timing.
- The smart router can optimize over departure windows using departure profiles.
- In plain English, it can improve the timing of the trip, not just the path.
- This is a different kind of gain from path-only optimization.
- In thesis terms, the system expands the decision space.
- That should be presented as a core advantage over static baseline routing.

### AC8. The Smart Router Can Beat Baselines By Being More Explainable

- Even when the final path is similar, the smart system may be better because it explains why it chose the route.
- Segment breakdowns, counterfactuals, Pareto charts, and baseline deltas all contribute to this.
- In plain English, the user gets reasoning support rather than only a line on a map.
- This matters for dispatch trust and for dissertation evidence.
- In thesis terms, explainability is a legitimate axis of system superiority.
- It should not be ignored simply because it is not a raw travel metric.

### AC9. The Smart Router Can Beat Baselines By Producing Better Evidence

- Manifests, provenance, signatures, and oracle-quality diagnostics do not directly shorten travel time.
- But they do improve the quality of evidence around a route decision.
- In plain English, the system can prove more about how it got its answer.
- This is a form of engineering superiority in governance and reproducibility.
- In thesis terms, the project's contribution is not only route scoring but also route accountability.
- That is especially important for research-grade software.

### AC10. Why The Repo Avoids Universal Percentage Claims

- The repo contains comparison machinery and benchmark scripts.
- It does not include one checked-in, all-cases summary proving a universal average improvement over OSRM or ORS across all freight scenarios.
- In plain English, the system can compare itself carefully, but the report should not invent a global win-rate.
- This is an evidence limitation, not a system flaw.
- In thesis terms, local evidence supports per-run and benchmarked comparisons, not unlimited generalization.
- Saying this clearly increases credibility.

### AC11. What Local Evidence Does Support

- Local code and docs support that baseline endpoints exist and are intentionally compared.
- Local config supports that realism multipliers are applied.
- Local benchmark tooling supports that the project is designed to measure performance and quality systematically.
- Local runtime artifacts support that the strict/live model assets were built successfully.
- In plain English, the repo contains real comparison infrastructure, not vague aspirational comments.
- In thesis terms, that is enough to support measured, scoped claims.
- It is not enough to support universal external dominance claims.

### AC12. Why A Smart Route Might Lose To A Baseline In Some Cases

- Some trips are simple.
- Some corridors have one obvious dominant motorway path.
- Some context layers may not move the route enough to matter.
- Some weight settings may align closely with what a baseline already optimizes.
- In plain English, the smart router is not guaranteed to look different or better on every trip.
- A strong thesis should admit this.
- It should explain when the extra model depth matters most.

### AC13. When The Smart Router Is Most Likely To Matter

- The smart pipeline is most useful when there are genuine corridor alternatives.
- It is also more useful when toll exposure, terrain, emissions, cost, or risk meaningfully differ across those corridors.
- Scenario-sensitive or time-sensitive trips also magnify the value of the richer model.
- In plain English, the smart system matters most when the route problem is multidimensional.
- This is exactly when pure shortest path is least sufficient.
- In thesis terms, these are the conditions under which the project should be evaluated most seriously.
- That helps frame case-study selection.

### AC14. Why ORS Comparison Is Kept Separate From OSRM Comparison

- OSRM and ORS are not identical reference systems.
- Their profiles, routing behavior, and deployment assumptions differ.
- In plain English, comparing against two references is more informative than pretending all baselines are the same.
- The repo even distinguishes true ORS from ORS-like proxy fallback where needed.
- In thesis terms, this is a careful comparative design choice.
- It supports more honest evaluation language.

### AC15. Why Realism Multipliers Are Not Cheating

- A naive reader may worry that baseline realism multipliers unfairly handicap the baseline.
- The better interpretation is that the project is trying to avoid comparing a richly modeled route with an overly idealized reference.
- In plain English, the goal is a fairer operational comparison, not a fake victory.
- These multipliers are explicit, configurable, and reportable.
- In thesis terms, transparency makes them defensible.
- Hidden realism adjustments would be much harder to justify.

### AC16. Why Per-Run Comparison Is Still Valuable

- Even without a universal checked-in win-rate, individual case comparisons are useful.
- They reveal when terrain, tolls, sharing modes, or risk-awareness produce materially different choices.
- In plain English, a good case study can still teach the reader what the smart router adds.
- The key is not to oversell one case as a universal truth.
- In thesis terms, carefully chosen per-run comparisons remain legitimate evidence.
- They are especially strong when paired with manifests and diagnostics.

### AC17. Why Benchmark Scripts Matter For This Discussion

- Benchmark scripts let the repo measure runtime and, in some cases, comparison-oriented behavior systematically.
- They are not the same as one-off UI screenshots.
- In plain English, they provide a more repeatable way to test the system.
- This matters for any claim about practicality or repeatability.
- In thesis terms, benchmark scripts are an evidence amplifier.
- They make comparative evaluation more defensible than anecdote alone.

### AC18. How To Phrase The Comparison Claim Safely In A Thesis

- A safe claim is that the repository provides a richer, more context-aware, and more auditable routing system than a plain baseline route provider.
- Another safe claim is that local comparison machinery can show per-case advantages in route quality, realism, or explainability.
- A less safe claim would be that the system universally beats OSRM or ORS by a fixed percentage everywhere.
- In plain English, the thesis should be specific about what kind of improvement is being claimed.
- This is not a limitation of the project alone.
- It is a standard expectation of careful evaluation writing.
- The repo's own design encourages that caution.

### Why This Appendix Matters

- The question "does it beat OSRM/ORS?" is a natural one.
- But it is too coarse unless the meaning of "beat" is unpacked.
- This appendix gives that unpacking.
- It shows why the smart system can be better, when that difference matters, and where the evidence stops.
- That makes the overall thesis argument much more defensible.
- It also protects the report from overclaiming.

## Appendix AD: Thesis And Viva Question Bank

This appendix is written in question-and-answer form for dissertation use.

The answers are still based on repo-local evidence.

The goal is to make the report easier to mine for thesis paragraphs, viva preparation, and chapter-writing prompts.

### AD1. Why is this not just another map app?

- Because the system does much more than draw one route on a map.
- It builds a UK-specific graph, generates alternative corridors, applies physical and economic models, filters a Pareto frontier, ranks candidates, and stores reproducible artifacts.
- In plain English, the map is the surface, not the substance.
- In thesis terms, the project is a routing-analysis platform rather than a visualization toy.

### AD2. Why is the routing hybrid rather than purely graph-based or purely provider-based?

- Pure provider routing gives strong road realization but limited control over candidate diversity and rich post-modeling.
- Pure custom-graph routing would demand much more geometry and road-realization machinery.
- The hybrid design keeps the strengths of both.
- In thesis terms, it is a pragmatic systems decision, not indecision.

### AD3. Why is the project UK-only?

- Because UK-specific calibration is more defensible than pretending to have globally valid assumptions.
- The repo uses UK asset families, bank holidays, corridor buckets, and live-source assumptions.
- In plain English, it is trying to be good somewhere rather than vague everywhere.
- In thesis terms, bounded scope is a strength.

### AD4. What does “warming up the graph” really mean?

- It means loading, checking, and preparing the large UK route graph before strict routing begins.
- The graph is big enough that this is a real startup phase.
- In plain English, the system is getting its road-search memory ready.
- In thesis terms, warmup is part of operational readiness.

### AD5. Why can the system refuse to route even when the server is up?

- Because liveness and strict readiness are different.
- The graph may still be warming, a live source may be stale, or a required asset may be invalid.
- In plain English, “the app is running” does not mean “the model is safe to trust.”
- In thesis terms, this is a fail-closed design principle.

### AD6. What is the main mathematical heart of the project?

- The clearest mathematical heart is the multi-objective route evaluation pipeline.
- That includes Pareto dominance, epsilon constraints, scalar compromise ranking, and robust uncertainty summaries.
- In plain English, the project tries to keep several kinds of route quality visible at once.
- In thesis terms, it is a multi-objective decision-support system.

### AD7. What is the main engineering heart of the project?

- The main engineering heart is the way many subsystems are made operational together: graph warmup, live data, strict failures, artifacts, and a usable frontend.
- In plain English, the system is not only mathematically interesting.
- It is also operationally disciplined.
- In thesis terms, this is a systems-integration contribution.

### AD8. How does the project differ from a clean academic algorithm implementation?

- The repo uses academic building blocks but adds budgets, rescue logic, fallback ladders, integrity checks, and modified selectors.
- These are engineering responses to real constraints and usability needs.
- In plain English, the project turns theory into a usable freight-routing system.
- In thesis terms, it is an applied implementation with transparent modifications.

### AD9. Why are the modified selection formulas acceptable in a thesis?

- Because the code explicitly labels them as engineering blends rather than novel theory.
- The report can therefore discuss them honestly as pragmatic extensions built from cited components.
- In plain English, the project is transparent about what is standard and what is tuned.
- In thesis terms, that transparency makes the modifications defendable.

### AD10. What does no-sharing, partial-sharing, and full-sharing mean without jargon?

- No-sharing means no special coordination benefits are assumed.
- Partial-sharing means some cooperative uplift is assumed.
- Full-sharing means the strongest modeled coordination benefit is assumed.
- In thesis terms, they are three operational scenarios, not three pathfinding algorithms.

### AD11. Why is live data important here?

- Live data makes the model temporally aware.
- Scenario context, fuel prices, bank holidays, and similar inputs can change over time.
- In plain English, the system cares about when and under what current conditions the route is being evaluated.
- In thesis terms, live data is part of the realism layer.

### AD12. Why is live data dangerous?

- Because stale, partial, or invalid live data can produce confident but misleading results.
- The repo answers this with strict gating, signatures, allow-lists, and dashboards.
- In plain English, live data is powerful but easy to misuse.
- In thesis terms, that is why governance is central to the design.

### AD13. Why does the project need so many scripts?

- Because calibration, quality checks, fetchers, builders, and reproducibility steps are major parts of the system.
- The runtime API depends on those assets being built and validated correctly.
- In plain English, the scripts are how the model is fed, checked, and refreshed.
- In thesis terms, they are part of the system, not background clutter.

### AD14. Why are manifests and provenance a big deal?

- Because they convert route outputs into inspectable evidence objects.
- That is important for repeatability, debugging, and dissertation appendices.
- In plain English, they provide the paperwork behind the result.
- In thesis terms, they strengthen the credibility of every reported experiment.

### AD15. Why is the frontend so large?

- Because the frontend exposes far more than one route card.
- It covers Pareto exploration, scenario compare, departure optimization, duty chains, artifacts, signatures, and dev diagnostics.
- In plain English, the UI is large because the system is genuinely feature-rich.
- In thesis terms, the frontend is part of the methodological surface.

### AD16. What makes the project freight-specific rather than generic routing?

- Vehicle classes, terrain sensitivity, cost modeling, toll realism, and duty-chain support all push it toward freight use.
- The project is not only about fastest path.
- In plain English, it tries to think like freight planning rather than like a consumer sat-nav.
- In thesis terms, freight orientation is visible across the stack.

### AD17. Why do terrain and weather matter?

- They change how hard or costly a route is, not just how long it looks on a flat map.
- Heavy vehicles, long routes, and energy-sensitive cases are especially affected.
- In plain English, roads are not identical once hills and conditions are considered.
- In thesis terms, terrain and weather are key realism multipliers.

### AD18. Why does the system use uncertainty at all?

- Because average ETA or cost can hide operationally painful tail behavior.
- The repo computes q95, CVaR, and robust score so route comparison can include bad-case sensitivity.
- In plain English, it asks not only "what usually happens?" but also "how bad can it get?"
- In thesis terms, this is a risk-aware routing layer.

### AD19. What is the biggest strength of the project?

- Its biggest strength is breadth with coherence.
- Graph search, multi-objective selection, live governance, physical modeling, artifacts, and a rich frontend are all connected rather than isolated demos.
- In plain English, it behaves like a serious end-to-end system.
- In thesis terms, that systems coherence is a major contribution.

### AD20. What is the biggest weakness or limitation?

- The biggest limitation is bounded scope and evidence.
- The project is UK-specific, asset-dependent, and careful not to claim universal superiority over every baseline everywhere.
- In plain English, its strength comes from being well-scoped, but that also limits generalization.
- In thesis terms, this is a respectable limitation, not a fatal flaw.

### AD21. What is the best way to explain Pareto routing to a non-specialist?

- Say that the system keeps routes where each one is good in a different way.
- No remaining route is simply worse than another on every important dimension at once.
- In plain English, Pareto routing keeps meaningful trade-offs alive.
- In thesis terms, it is the core multi-objective idea.

### AD22. What is the best way to explain robust routing to a non-specialist?

- Say that the system cares about bad-case outcomes, not just average outcomes.
- A route that looks good most of the time but fails badly in the tail can be penalized.
- In plain English, robust routing prefers safer compromises under uncertainty.
- In thesis terms, it is the uncertainty-aware decision layer.

### AD23. What is the best way to explain the route graph to a non-specialist?

- Say that it is the backend's private road-search model for exploring different corridor possibilities.
- OSRM then helps turn those possibilities into realized road routes.
- In plain English, the graph is for finding alternatives, not only for drawing the final line.
- In thesis terms, it is the main engine behind route diversity.

### AD24. Why are there so many strict reason codes?

- Because different failures mean different things and require different mitigations.
- A stale scenario source is not the same problem as a disconnected graph or a missing terrain asset.
- In plain English, the system tries to fail specifically rather than vaguely.
- In thesis terms, that is part of its auditability.

### AD25. Why does the project care about signatures if it is not a security thesis?

- Because integrity still matters for reproducibility.
- A result that cannot be checked for tampering is weaker evidence.
- In plain English, signatures make artifacts more trustworthy.
- In thesis terms, they strengthen the evidence chain without changing the thesis domain.

### AD26. What is the fairest way to describe baseline comparisons?

- Say the repo provides local, per-run and scripted comparison capability against OSRM and ORS-style references.
- Do not claim one universal checked-in win-rate unless the evidence exists.
- In plain English, compare carefully and specifically.
- In thesis terms, scoped claims are better than inflated ones.

### AD27. Why does the project have an oracle-quality dashboard?

- Because source health and route quality are not the same thing.
- The dashboard helps the user judge whether the feeds behind the model are healthy.
- In plain English, it checks the trustworthiness of the inputs.
- In thesis terms, it is part of live-data governance.

### AD28. Why is there a tutorial mode in a technical project?

- Because a complex system is more valuable when people can actually learn it.
- The tutorial guides users through real workflows rather than leaving features hidden.
- In plain English, the project teaches its own interface.
- In thesis terms, this improves usability and demonstration quality.

### AD29. Why is batch analysis important in the thesis context?

- Because one or two routes are rarely enough for strong evaluation.
- Batch workflows let the project test many OD pairs and preserve broader evidence.
- In plain English, batch mode helps turn examples into studies.
- In thesis terms, it supports stronger empirical evaluation.

### AD30. If the thesis had to summarize the system in one sentence, what should it say?

- It is a UK-specific, graph-led, multi-objective freight-routing system that combines academic routing ideas with pragmatic engineering layers for live data, physical realism, robust risk handling, and reproducible evidence.
- In plain English, it is a smarter and more accountable router, not just a faster one.
- That sentence is broad enough to cover the stack but specific enough to stay defensible.
- In thesis terms, it is a strong abstract-level summary grounded in the repo.

## Appendix AE: Limitations, Risks, And Future-Work Directions

This appendix makes the report more academically balanced.

The repository is strong, but it is not unlimited.

A good thesis should state what remains hard, uncertain, or incomplete.

### AE1. Geographic Scope Remains Bounded

- The project is tuned for the UK and should be described that way.
- Extending it to another country would need new graph preparation, departure behavior, scenario context, toll assets, and likely new live-source policies.
- In plain English, the system does not become global merely by changing the map extract.
- In thesis terms, geographic transfer is future work, not a present claim.

### AE2. Calibration Can Age

- Scenario profiles, departure profiles, and stochastic regimes are only as current as the data and publication pipeline behind them.
- Drift tooling helps, but it does not magically eliminate temporal change.
- In plain English, calibrated realism still needs maintenance.
- In thesis terms, recalibration cadence is an important future-work consideration.

### AE3. Live Data Can Fail For Reasons Outside The Repo

- Even good code cannot control upstream outages, schema shifts, or rate limits.
- The repo handles this better than many systems by failing clearly and tracing calls.
- In plain English, live-data dependence remains an external risk.
- In thesis terms, upstream source volatility is a standing systems risk.

### AE4. Strictness Improves Trust But Reduces Availability

- Fail-closed behavior is methodologically strong.
- It also means some routes will fail instead of returning best-effort answers.
- In plain English, the system chooses honesty over always answering.
- In thesis terms, this is a defendable tradeoff, but still a tradeoff.

### AE5. Richer Models Mean More Configuration

- The repo has many environment variables and calibration surfaces.
- That flexibility is useful but increases deployment and tuning complexity.
- In plain English, the system is powerful partly because it has many knobs.
- In thesis terms, configuration management is a real operational challenge.

### AE6. The Frontend Carries Significant Complexity

- The large page orchestration file is practical, but it is also heavy.
- Future work could decompose state and workflows into smaller modules while preserving clarity.
- In plain English, the UI is effective but not minimal.
- In thesis terms, frontend maintainability is a reasonable future-improvement area.

### AE7. Global Superiority Over Baselines Is Not Proven In-Repo

- The project supports careful local comparison.
- It does not include one final universal number proving superiority across all freight use cases.
- In plain English, evidence is strong for scoped comparison, not unlimited generalization.
- In thesis terms, wider benchmark corpora remain future work.

### AE8. Physical Modeling Is Purposeful But Approximate

- Terrain and energy logic use physically informed engineering approximations.
- They are not full vehicle-dynamics simulations.
- In plain English, the model is realistic enough to matter, but not infinitely detailed.
- In thesis terms, higher-fidelity physics would be future work if justified by data and runtime budgets.

### AE9. Scenario Modes Still Depend On Modeling Assumptions

- Even with data support, no-sharing, partial-sharing, and full-sharing are scenario constructs rather than ground truth labels for every trip.
- Their value lies in disciplined comparison, not in pretending to observe all coordination effects perfectly.
- In plain English, the scenario layer is useful but interpretive.
- In thesis terms, scenario semantics should be discussed carefully and honestly.

### AE10. Uncertainty Quality Depends On Regime Coverage

- The stochastic model is richer where contexts map well to calibrated regimes.
- Sparse or weakly matched contexts can reduce confidence even when the system still computes.
- In plain English, risk estimates are best where the calibration knows the context well.
- In thesis terms, regime coverage is a natural future-work area.

### AE11. Operator Usability Still Needs Study

- The frontend is feature-rich, but the repo does not itself contain a formal user study.
- Tutorial mode, accessibility work, and help text are positive signs, but usability evidence could go further.
- In plain English, the interface looks serious, but user-research evidence could be deeper.
- In thesis terms, formal usability evaluation is an obvious extension.

### AE12. Cost Modeling Could Incorporate More Business Rules

- The current model already includes fuel, tolls, carbon cost, and physical effects.
- Real fleets may also care about labor rules, depot constraints, charging infrastructure, contractual penalties, and more.
- In plain English, there is room to widen the economics further.
- In thesis terms, richer operational business constraints are future-work candidates.

### AE13. Multi-Trip Network Effects Are Still Limited

- The duty-chain and batch features are important, but the repo is not a full fleet-network optimizer.
- It evaluates many trips and chains, but not a full dispatch network equilibrium.
- In plain English, it reaches toward operations planning without solving every fleet-allocation problem.
- In thesis terms, network-wide optimization remains future work.

### AE14. More External Benchmarking Would Strengthen The Story

- The repo's internal benchmarking and comparison scaffolding are strong.
- External benchmark corpora or third-party evaluations would strengthen generalization further.
- In plain English, more outside testing would be valuable.
- In thesis terms, independent evaluation is a high-value next step.

### AE15. Despite These Limits, The Current Scope Is Strong

- The project already combines enough algorithmic, physical, contextual, and operational depth to sustain a substantial thesis.
- The limitations are real but bounded and discussable.
- In plain English, the system is meaningfully ambitious without pretending to be complete.
- In thesis terms, that is exactly the kind of scope a strong dissertation can defend.

## Appendix AF: How Quality Is Actually Scored In The Repository

This appendix explains the quality-scoring machinery more explicitly.

The phrase "quality gate" can sound vague if it is not unpacked.

In this repository, quality is not one single number.

It is a set of subsystem scores, threshold checks, dropped-route gates, diversity gates, and benchmark expectations.

### AF1. Quality Is Scored Per Subsystem

- The backend quality script does not collapse everything into one unstructured pass/fail.
- It scores multiple subsystems independently.
- Local script evidence shows named score families including risk aversion, dominance, scenario profile, departure time, stochastic sampling, terrain profile, toll classification, fuel price, carbon price, and toll cost.
- In plain English, the repo asks "is each important modeling layer behaving well enough?" rather than only "did the overall script run?"
- In thesis terms, this is a much stronger evaluation design than a single smoke test.

### AF2. The Threshold Philosophy Is Explicit

- The quality script defines threshold targets for those subsystems.
- Local code shows a threshold value of 95 for each named subsystem in the quality-threshold map.
- In plain English, the default standard is intentionally high.
- This matters because the project is not satisfied by barely functional outputs.
- In thesis terms, the score threshold is part of the system's acceptance contract.

### AF3. Quality Also Includes Availability Gates

- Subsystem scores are not the only gate.
- The script also tracks dropped routes and compares them against a configured cap.
- Local docs state the strict default target is zero dropped routes.
- In plain English, a model that scores well only by silently discarding too many cases is not considered acceptable.
- In thesis terms, availability is part of quality.

### AF4. Diversity Gates Matter Too

- The quality script also checks fixture diversity and corridor diversity.
- That is significant because route-quality evidence is weaker when all tests come from too few patterns.
- In plain English, the repo wants to ensure quality is being judged over a meaningfully varied route set.
- This connects directly to the graph-theory note that frontier quality depends on candidate-space coverage.
- In thesis terms, input diversity is treated as part of evaluation legitimacy.

### AF5. Risk-Aversion Scoring

- The risk-aversion score checks whether robust decision behavior responds correctly to different uncertainty profiles.
- This matters because robust routing is one of the advanced claims in the project.
- In plain English, the repo wants to know whether routes with worse bad tails are penalized as expected.
- This is not just a mathematical curiosity.
- In thesis terms, it validates that the robust layer behaves like a real decision mechanism.

### AF6. Dominance Scoring

- The dominance score checks the Pareto layer.
- It matters because the project's multi-objective identity depends on dominance filtering being correct.
- In plain English, it tests whether obviously dominated routes are handled properly.
- This also supports later frontier curation and selector behavior.
- In thesis terms, it validates the core multi-objective logic.

### AF7. Scenario-Profile Scoring

- The scenario-profile score is one of the most important subsystem scores.
- Local docs state that it expects strict monotonicity across scenario modes and completeness of `scenario_summary` fields.
- The script also checks holdout fit behavior and observed-versus-projected context share.
- In plain English, the repo wants scenario effects to be ordered sensibly, data-backed, and complete in output.
- In thesis terms, this is a formal check on the policy-like scenario subsystem.

### AF8. Why Monotonicity Across Modes Matters

- Local docs explicitly state the expected mode ordering `no_sharing >= partial_sharing >= full_sharing`.
- This matters because the scenario modes are meant to represent increasing coordination benefit.
- In plain English, the system checks that the "more sharing" modes do not accidentally look worse in the wrong direction for core multipliers.
- That is an example of semantic validation, not only syntactic validation.
- In thesis terms, it is an important argument for calibrated interpretability.

### AF9. Holdout Metrics Matter In Scenario Quality

- The scenario builder and quality scorer refer to holdout behavior, separability, coverage, and row-share thresholds.
- Local script evidence shows strict checks around separability, MAPE, holdout coverage, observed-mode row share, projection-dominant context share, and identity collapse.
- In plain English, the repo tries to ensure the scenario layer has genuine predictive or structural value on held-out cases.
- This matters because scenario modeling is easy to overstate.
- In thesis terms, holdout scoring is a key defense against overfitting claims.

### AF10. Departure-Time Scoring

- The departure-time score checks the time-sensitive route layer.
- This matters because departure optimization should be grounded in a coherent departure profile, not an empty repeated loop.
- In plain English, the repo wants departure-aware routing to actually behave differently where it should.
- This supports the thesis claim that schedule matters in the model.
- In thesis terms, it validates the temporal-calibration subsystem.

### AF11. Stochastic-Sampling Scoring

- The stochastic-sampling score checks the uncertainty machinery.
- Local docs mention posterior regime probabilities, quantile mapping, and factor mappings for traffic, incident, weather, price, and eco.
- In plain English, the repo asks whether its uncertainty engine has the right kinds of calibrated moving parts and whether those parts behave sensibly.
- This matters because stochastic routing is easy to fake with arbitrary randomness.
- In thesis terms, it supports the seriousness of the risk layer.

### AF12. What Posterior Regime Checks Mean

- Posterior regime checks ensure there is a context-aware mapping from trip situation to uncertainty regime.
- In plain English, the system wants to know that not every route shares the same uncertainty personality.
- This matters because motorway-heavy weekday-peak freight should not necessarily look like all other trips.
- The posterior machinery is one of the deeper statistical features in the repo.
- In thesis terms, it is a context-rich uncertainty validation target.

### AF13. Clipping And Coverage Diagnostics Matter

- The stochastic docs and code mention clipping diagnostics and bounded factors.
- This matters because uncertainty models can become numerically unstable or unrealistic if tails are not governed.
- In plain English, the repo checks whether its uncertainty system stays within sane behavior ranges.
- This is one of the reasons the robust layer is more than a decorative checkbox.
- In thesis terms, bounded stochastic diagnostics are part of quality assurance.

### AF14. Terrain-Profile Scoring

- Terrain-profile scoring checks whether terrain-aware routing behaves sensibly under hilly versus flat assumptions.
- Local code uses hilly and flat route-option variants and inspects coverage and source signals.
- In plain English, the repo wants terrain to matter where it should and to fail clearly where it cannot be trusted.
- This matters because terrain is one of the richest physical realism layers in the project.
- In thesis terms, terrain quality scoring validates both physical effect and coverage governance.

### AF15. Runtime P95 Matters In Terrain Quality

- Local script evidence shows terrain-profile scoring also records runtime P95 and compares it against a two-second normalization target.
- That is important because terrain realism that destroys runtime usability is not acceptable.
- In plain English, terrain quality is judged partly by speed, not only by physical plausibility.
- This is a very practical engineering stance.
- In thesis terms, it shows that quality includes latency discipline.

### AF16. Toll-Classification Scoring

- Toll-classification scoring checks whether toll situations are identified correctly.
- This matters because toll cost depends first on recognizing tolled structure correctly.
- In plain English, before the system can price a toll, it has to know that a toll applies.
- This is a classification problem, not just a pricing problem.
- In thesis terms, it validates the structural side of the monetary model.

### AF17. Toll-Cost Scoring

- Toll-cost scoring checks whether money is attached correctly once toll structure is known.
- This is distinct from toll classification and is treated separately by the script.
- In plain English, the system separately validates "did I spot the toll?" and "did I price it correctly?"
- That separation is methodologically strong.
- In thesis terms, it makes the toll subsystem easier to defend.

### AF18. Fuel-Price Scoring

- Fuel-price scoring checks a mixture of source integrity and numeric behavior.
- Local code inspects quantile ordering for fuel liters and fuel cost, vehicle-profile completeness, source metadata, and response to multiplier changes.
- In plain English, the repo checks both that fuel data is trustworthy and that it changes route economics in the expected direction.
- This matters because cost realism is a major claim of the project.
- In thesis terms, the fuel-price score is central to the monetary-objective story.

### AF19. Why Quantile Ordering Shows Up In Quality Checks

- The docs explicitly call out monotone quantile ordering such as `p10 <= p50 <= p90`.
- This matters because uncertainty summaries are meaningless if their quantiles cross or behave incoherently.
- In plain English, the percentiles have to be in the right order for the uncertainty story to make sense.
- This is a small but important sanity condition.
- In thesis terms, it supports numerically credible uncertainty reporting.

### AF20. Carbon-Price Scoring

- Carbon-price scoring checks the environmental monetization layer.
- It matters because the repo distinguishes direct emissions from carbon-cost interpretation.
- In plain English, the system checks whether carbon pricing inputs and outputs behave sensibly as a policy-cost layer.
- This is important for route comparisons where money and environment interact.
- In thesis terms, it validates the environmental-economics part of the stack.

### AF21. Raw Evidence Requirements Matter

- The quality script defines required raw-evidence paths for several subsystems when strict live data is required.
- Local code names raw scenario, stochastic residual, DfT counts, fuel-price, carbon-intensity, toll-classification, and toll-pricing evidence families.
- In plain English, the repo does not want scored assets without traceable raw support.
- This is a strong provenance stance.
- In thesis terms, quality is tied to evidence lineage, not only output numbers.

### AF22. Quality Is Not Only About the Final Route Response

- Many scoring checks happen on the calibration assets or route-construction internals rather than on one final frontend screenshot.
- That matters because a route can look plausible while still sitting on weak calibration.
- In plain English, the repo tries to check the hidden machinery, not only the visible result.
- This is a strong engineering habit.
- In thesis terms, it supports deeper validity claims.

### AF23. Benchmarking And Quality Are Related But Distinct

- The docs clearly separate `backend/scripts/score_model_quality.py` from runtime benchmark scripts.
- Quality asks whether the subsystems behave correctly enough.
- Benchmarking asks how quickly or efficiently the system behaves.
- In plain English, correctness and speed are judged separately.
- In thesis terms, that separation produces cleaner evaluation logic.

### AF24. CI Uses This Structure

- Local docs state that CI lanes reflect these quality and strictness ideas.
- The fast lane uses bypassed strictness for quick regression confidence.
- The strict-live lane keeps signed fallbacks off and validates strict behavior.
- In plain English, the repo tests both day-to-day stability and full strict behavior.
- In thesis terms, CI mirrors the system's methodological commitments.

### AF25. Quality Scoring Supports Better Thesis Claims

- A thesis grounded in this repo can say more than "the code runs."
- It can say that multiple subsystems have explicit thresholds, evidence requirements, and drop-rate gates.
- In plain English, the project measures quality rather than assuming it.
- That does not prove perfection.
- But it does make the system more defensible than an unscored prototype.

### Why This Appendix Matters

- The user specifically asked about quality determination and scoring points.
- This appendix answers that directly.
- It shows that quality in this repo is formalized as a combination of subsystem metrics, thresholds, dropped-route gates, diversity checks, and benchmark expectations.
- That is exactly the kind of material that strengthens a thesis report.
- It also explains why the repo contains so much validation and testing machinery beyond the main API.

## Appendix AG: Physics, Energy, And Emissions Explained More Slowly

This appendix revisits the physical model in slower, thesis-friendly language.

The goal is not to turn the report into a vehicle-dynamics textbook.

The goal is to explain what physical ideas the project actually uses, what simplifications it accepts, and why that is still useful for routing.

### AG1. Why A Physical Model Is Needed At All

- A basic route engine can optimize by distance or nominal time without caring much about vehicle physics.
- This project wants to say more than "shortest path."
- It wants to estimate energy, cost, and emissions with some sensitivity to vehicle type, slope, and conditions.
- In plain English, physical modeling is what makes the route feel like a freight route rather than a geometry puzzle.
- In thesis terms, this is part of the realism layer that distinguishes the system from simpler baselines.

### AG2. The Model Is Physically Inspired, Not Fully Simulation-Grade

- The terrain physics module uses compact engineering approximations rather than a full dynamic vehicle simulator.
- That choice is deliberate.
- A route engine needs tractable computations over many route candidates.
- In plain English, the repo uses enough physics to matter, but not so much physics that the routing problem becomes intractable.
- In thesis terms, it is a purposeful approximation strategy.

### AG3. Vehicle Mass Matters

- Vehicle mass directly enters rolling resistance and grade-force terms.
- Heavier vehicles pay more for climbing and often carry larger resistance loads overall.
- In plain English, a hill hurts a heavy artic more than a light van in the model, because that is physically plausible.
- This is one reason vehicle profiles are not interchangeable.
- In thesis terms, mass is one of the core physical drivers in the route-energy layer.

### AG4. Rolling Resistance Matters

- Rolling resistance is modeled using mass, gravity, and a rolling-resistance coefficient.
- This gives a baseline force cost even on flat ground.
- In plain English, even a perfectly flat road still costs energy because tyres and road contact are not frictionless.
- This matters because route energy should not collapse to pure aerodynamic effects.
- In thesis terms, rolling resistance grounds the model in standard engineering intuition.

### AG5. Aerodynamic Drag Matters

- The model also includes an aerodynamic drag term through drag area and speed.
- Drag grows with the square of speed.
- In plain English, fast travel becomes disproportionately expensive because air resistance ramps up quickly.
- This matters for distinguishing route segments with different speed regimes.
- In thesis terms, it links route speed structure to energy realism.

### AG6. Grade Force Matters

- Grade force is proportional to vehicle mass, gravity, and road grade.
- Climbing therefore adds direct physical burden.
- Downhill travel can provide relief, but not infinite relief.
- In plain English, hills matter because gravity really does help or hurt the vehicle.
- In thesis terms, grade is the main reason terrain-aware routing can change route ranking.

### AG7. Downhill Relief Is Controlled

- The model does not let downhill segments create absurdly good outcomes.
- It caps how much negative grade can help, and it tempers recovery with regenerative or relief limits.
- In plain English, going downhill can help, but it does not magically make the trip free.
- This is an example of the repo preferring controlled realism to naive equation use.
- In thesis terms, it is a bounded-physics design choice.

### AG8. Drivetrain Efficiency Matters

- The terrain multiplier ultimately passes through drivetrain efficiency assumptions.
- This means not all physical force translates one-to-one into delivered energy use or duration uplift.
- In plain English, part of the route burden is mediated by how efficiently the vehicle turns energy into motion.
- This helps differentiate vehicles beyond raw mass.
- In thesis terms, drivetrain efficiency is a compact but important physical realism term.

### AG9. Regenerative Efficiency Matters

- The model includes a regenerative efficiency term, especially relevant for EV-like behavior.
- This means some downhill relief can be recovered rather than lost entirely.
- In plain English, some vehicle types can claw back more benefit from downhill sections than others.
- This matters when comparing conventional and electric-like profiles.
- In thesis terms, it makes EV-aware routing more than a label swap.

### AG10. Terrain Profiles Control Intensity, Not Geography

- The terrain profile setting such as flat, rolling, or hilly does not redraw the route.
- It changes how strongly terrain effects are weighted in the physical model.
- In plain English, it is a modeling-intent knob, not a map-editing knob.
- This matters because the same DEM evidence can be interpreted with different physical aggressiveness.
- In thesis terms, terrain profile is a realism-intensity control.

### AG11. Segment Duration Multiplier

- The terrain module computes a segment duration multiplier from force ratios and profile weighting.
- This is a compact way of turning physical burden into travel-time pressure.
- In plain English, physically harder segments become time-heavier in the model.
- This is not claiming a perfect speed model.
- In thesis terms, it is a route-duration approximation informed by physics.

### AG12. Route Emissions Multiplier

- The terrain module also produces an emissions multiplier from uphill and downhill grade summaries.
- Uphill burden tends to increase emissions pressure.
- Downhill relief can offset some of that depending on regenerative assumptions.
- In plain English, hillier routes can cost more environmentally even if distance looks similar.
- In thesis terms, this is one way the emissions objective gains route-specific realism.

### AG13. Why Terrain Coverage Rules Are Strict

- Terrain effects are only meaningful if elevation support is good enough.
- That is why the repo has a UK fail-closed coverage threshold.
- In plain English, the system refuses to talk confidently about hills it cannot actually see.
- This is an integrity choice.
- In thesis terms, physical realism is governed, not merely approximated.

### AG14. Fuel Use Is Not A Flat Distance Multiplier

- The richer fuel model does not reduce fuel use to distance times a constant.
- It uses vehicle class, speed, terrain, region multiplier, and price surfaces.
- In plain English, the cost of a route depends on how the journey is driven, not only how long it is.
- This matters because many freight routes trade distance against slope or speed.
- In thesis terms, fuel modeling is one of the clearest realism additions over baseline routing.

### AG15. Fuel Price And Fuel Consumption Are Separated

- The repo distinguishes how much fuel is consumed from how much that fuel currently costs.
- This is important because physical consumption and market price are not the same thing.
- In plain English, one module estimates usage and another layer prices it.
- That separation is methodologically cleaner.
- In thesis terms, it supports better economic interpretation of route changes.

### AG16. EV Handling Is Treated Explicitly

- EV-like routing is not simply diesel logic with different labels.
- The fuel-energy model checks powertrain and fuel type, and it uses grid price and regenerative logic where appropriate.
- In plain English, electric freight behavior is treated as a distinct modeling path.
- This matters because cost and emissions semantics differ from liquid fuels.
- In thesis terms, it broadens the fleet realism of the system.

### AG17. Emissions Are Route And Context Dependent

- Emissions in this repo depend on route conditions, vehicle context, and energy/fuel behavior.
- They are not only a direct function of distance.
- In plain English, a slower, hillier, or more stop-start route can look worse environmentally even if it is not much longer.
- This is important for meaningful multi-objective trade-offs.
- In thesis terms, it gives the emissions objective more explanatory power.

### AG18. Carbon Cost Is A Policy Layer Over Emissions

- Carbon cost is calculated from emissions using a configured schedule.
- This means the project separates physical emissions from economic interpretation of those emissions.
- In plain English, the route may emit the same CO2 while appearing more or less expensive under different carbon schedules.
- That is a useful distinction.
- In thesis terms, it keeps environmental accounting and environmental pricing conceptually separate.

### AG19. Weather Changes Physical Burden Indirectly

- Weather is not modeled as a full atmospheric vehicle simulator.
- Instead, it affects speed and incident multipliers, which then influence time, energy use, and emissions indirectly.
- In plain English, bad weather makes the route more difficult rather than rewriting the laws of physics inside the model.
- This is a practical compromise.
- In thesis terms, weather is a context pressure layer rather than a full physical sub-simulator.

### AG20. What This Means For Thesis Claims

- The physics layer is strong enough to justify saying the router is terrain-aware and vehicle-aware.
- It is not strong enough to justify claiming full high-fidelity vehicle simulation.
- That honest framing is important.
- In plain English, the physical model is useful, bounded, and purposeful.
- In thesis terms, that is exactly how it should be presented.

## Appendix AH: Uncertainty, Risk, And Robust Routing Explained More Slowly

This appendix revisits the uncertainty model in slower prose.

The aim is to explain how the repo moves from one nominal route estimate to a calibrated view of route risk.

### AH1. Why Uncertainty Exists In The Model

- Freight planning often cares about the bad case, not just the average case.
- A route that is usually good but occasionally terrible may be operationally unattractive.
- The repo responds by modeling route uncertainty explicitly.
- In plain English, it tries to quantify how shaky a route's outcome might be.
- In thesis terms, this is the basis of robust routing.

### AH2. Uncertainty Is Contextual, Not Universal

- The backend does not use one global randomness model for all trips.
- It resolves a stochastic regime using corridor, day kind, local time, road mix, weather, and vehicle bucket.
- In plain English, different kinds of trips are allowed to have different volatility patterns.
- This is more defensible than a one-size-fits-all noise term.
- In thesis terms, uncertainty is contextualized.

### AH3. Stable Seeding Matters

- The uncertainty model uses a stable seed built from route signature, departure slot, and user seed.
- This means the same route context can be replayed consistently.
- In plain English, randomness is controlled rather than chaotic.
- This supports reproducibility and debugging.
- In thesis terms, stable seeding is an important research-software feature.

### AH4. Departure Slotting Matters

- The route's departure time is converted into a deterministic local slot.
- This slot influences regime resolution and seeding.
- In plain English, leaving at 08:00 and leaving at 08:15 can matter in a structured way.
- This makes uncertainty time-aware.
- In thesis terms, it links temporal context and stochastic calibration.

### AH5. Holiday And Weekend Logic Matters

- The model classifies departure dates into weekday, weekend, or holiday.
- This affects regime selection.
- In plain English, uncertainty on a bank holiday is not treated like an ordinary weekday by default.
- This supports UK-specific realism.
- In thesis terms, it shows the risk model is calendar-aware.

### AH6. Posterior Regime Candidates Matter

- Where possible, the model uses posterior context-to-regime probabilities from calibration artifacts.
- In plain English, it asks the calibration which uncertainty regime best fits this trip context.
- This is a more principled approach than picking a generic regime by hand.
- It also supports strict checks when posterior calibration is required.
- In thesis terms, it makes uncertainty assignment data-driven.

### AH7. Fallback Regime Logic Still Exists

- If posterior mapping is unavailable or loosened, the model can fall back through a hierarchy of simpler regime candidates.
- In plain English, it tries context-rich matching first and broader matching later.
- This is a practical fallback, not a claim of equal fidelity.
- It helps the model remain usable while still preferring calibrated specificity.
- In thesis terms, it is another engineering compromise made explicit.

### AH8. Correlation Matters

- The model samples correlated shocks rather than independent random multipliers.
- This is important because traffic, incidents, weather, price, and eco effects are not truly independent in real life.
- In plain English, bad factors can move together.
- A realistic risk model should reflect that.
- In thesis terms, correlation is a major upgrade over naive independent sampling.

### AH9. Cholesky Factorization Matters

- The repo uses a cached Cholesky factor of the regime correlation matrix.
- This turns a desired correlation structure into a sampling mechanism.
- In plain English, it is the mathematical tool that lets the model draw linked random effects coherently.
- This is standard numerical practice used in a careful applied way.
- In thesis terms, it is part of the stochastic-engineering core.

### AH10. Antithetic Sampling Matters

- The model uses antithetic pairs of shocks.
- In plain English, for one random draw it also uses its sign-flipped counterpart.
- This helps reduce Monte Carlo noise for a given sample count.
- It is an efficiency and stability trick, not a new theory claim.
- In thesis terms, it is a strong sign of careful simulation practice.

### AH11. Sigma Is A Spread Control, Not A Magic Confidence Dial

- Sigma controls uncertainty spread before clipping and regime scaling.
- Larger sigma produces wider variation.
- In plain English, it tells the model how volatile the world should be around the nominal route.
- This changes risk summaries more than path geometry.
- In thesis terms, sigma is an uncertainty-shape parameter.

### AH12. Factor Clipping Matters

- The uncertainty model clips factors into bounded ranges.
- This prevents absurdly small or absurdly large multiplicative shocks.
- In plain English, the model is allowed to be uncertain, but not ridiculous.
- This is especially important in tails.
- In thesis terms, clipping is part of making uncertainty numerically credible.

### AH13. Quantile Mapping Matters

- Some regimes use quantile-mapping transforms for shocks.
- In plain English, the model can shape raw random draws into empirically informed multiplier behavior.
- This is more data-respecting than assuming pure lognormal-style scaling everywhere.
- It helps the tails look more like calibrated behavior.
- In thesis terms, it is an important link from data to simulation.

### AH14. Duration, Money, And Emissions Are Sampled Together

- The model samples route duration, monetary cost, and emissions together.
- It does not treat risk as a duration-only layer.
- In plain English, uncertainty is multi-objective too.
- This matters because a route can be risky economically or environmentally as well as temporally.
- In thesis terms, robust routing spans the full objective family.

### AH15. Utility Is Built From Normalized Components

- The risk model normalizes duration, money, and emissions using contextual reference intensities.
- This makes the utility dimensionless.
- In plain English, it gives the model a fairer way to combine different units under uncertainty.
- This is why risk modeling is not just "seconds plus pounds plus kilos."
- In thesis terms, normalization is essential to robust multi-objective utility.

### AH16. Quantiles And CVaR Serve Different Roles

- Quantiles such as q95 describe a threshold.
- CVaR describes the average behavior beyond that bad threshold.
- In plain English, q95 says where the bad tail starts, and CVaR says how bad it is once you are there.
- Both are useful.
- In thesis terms, the repo uses a richer tail language than many routing systems.

### AH17. Robust Score Converts Risk Into A Decision Signal

- The robust score takes mean utility and tail behavior and turns them into one risk-aware decision value.
- This can use CVaR excess, entropic-style, or downside-semivariance family logic.
- In plain English, it is how the route becomes "safe" or "unsafe" in ranking terms.
- This matters because summaries alone do not choose a route.
- In thesis terms, robust score is the decision-facing end of uncertainty modeling.

### AH18. Sample Caps And Clip Ratios Are Honest Diagnostics

- The model records sample-count clipping, sigma clipping, and factor clip rate.
- In plain English, it tells you when a requested uncertainty setup had to be tamed or bounded.
- This is important because hidden clipping would make uncertainty summaries harder to interpret.
- The repo makes those diagnostics explicit.
- In thesis terms, this supports honest risk reporting.

### AH19. Risk Can Change The Recommended Route Without Changing The Frontier

- Two routes may both remain feasible and non-dominated.
- But robust ranking can still favor one over the other if its tail behavior is better.
- In plain English, risk often changes the winner more than it changes the existence of options.
- This parallels how preference weights work at the selection layer.
- In thesis terms, it is another example of recommendation logic differing from candidate generation.

### AH20. What This Means For Thesis Claims

- The repo supports a credible claim that it is uncertainty-aware and capable of robust route interpretation.
- It does not support a claim of perfect probabilistic truth for every context.
- In plain English, it models route risk carefully, but within bounded calibration and engineering assumptions.
- That is a strong and defensible position.
- In thesis terms, the uncertainty layer is one of the report's strongest technical chapters when described this way.

## Related Docs

- [Documentation Index](DOCS_INDEX.md)
- [Backend APIs and Tooling](backend-api-tools.md)
- [Model Assets and Data Sources](model-assets-and-data-sources.md)
- [Quality Gates and Benchmarks](quality-gates-and-benchmarks.md)
