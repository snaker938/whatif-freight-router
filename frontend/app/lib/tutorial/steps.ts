import type { TutorialChapter, TutorialStep } from './types';

export const TUTORIAL_CHAPTERS: TutorialChapter[] = [
  {
    id: 'chapter_map',
    title: '1. Orientation and map basics',
    description: 'Set and manage pins, then use map-level controls and marker actions.',
  },
  {
    id: 'chapter_setup',
    title: '2. Setup and overlays',
    description: 'Configure vehicle/scenario context and verify map overlays.',
  },
  {
    id: 'chapter_advanced',
    title: '3. Advanced parameters',
    description: 'Tune optimization, Pareto filtering, uncertainty, terrain, and costs.',
  },
  {
    id: 'chapter_routes',
    title: '4. Compute and route analysis',
    description: 'Generate routes, inspect output panels, and use route-level tools.',
  },
  {
    id: 'chapter_compare',
    title: '5. Compare, departure optimization, and timeline playback',
    description: 'Run scenario deltas, optimize departure windows, and animate route playback.',
  },
  {
    id: 'chapter_ops',
    title: '6. Duty chain, oracle quality, and experiments',
    description: 'Run multi-leg planning, quality checks, and experiment lifecycle actions.',
  },
  {
    id: 'chapter_finish',
    title: '7. Completion recap',
    description: 'Confirm completion and keep restart available for future onboarding.',
  },
];

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: 'map_set_pins',
    chapterId: 'chapter_map',
    title: 'Set both route pins on the map',
    what:
      'Click the map to place Start and Destination pins. The tutorial pre-fills canonical coordinates, but you still need to actively place both pins to confirm map interaction.',
    impact:
      'All downstream computations use these coordinates. Every route metric, scenario delta, and artifact depends on this origin/destination pair.',
    targetIds: ['map.interactive'],
    required: [
      { actionId: 'map.set_origin', label: 'Place or update the Start pin.' },
      { actionId: 'map.set_destination', label: 'Place or update the Destination pin.' },
    ],
    prefillId: 'canonical_map',
  },
  {
    id: 'map_drag_pin',
    chapterId: 'chapter_map',
    title: 'Drag a pin to adjust coordinates',
    what:
      'Drag either marker once. This verifies precision editing after initial map placement.',
    impact:
      'Small coordinate changes can alter road options, ETA, and Pareto trade-offs; drag is the fastest refinement workflow.',
    targetIds: ['map.interactive'],
    required: [{ actionId: 'map.drag_marker', label: 'Drag any map marker once.' }],
  },
  {
    id: 'map_popup_actions',
    chapterId: 'chapter_map',
    title: 'Use marker popup controls',
    what:
      'Open a marker popup and use copy/close controls. Popup controls expose pin management shortcuts without leaving the map.',
    impact:
      'Popup actions speed operator workflows and reduce sidebar context switching during iterative planning.',
    targetIds: ['map.interactive'],
    required: [
      { actionId: 'map.popup_copy', label: 'Copy coordinates from any marker popup.' },
      { actionId: 'map.popup_close', label: 'Close a marker popup using its close action.' },
    ],
  },
  {
    id: 'map_destructive_actions',
    chapterId: 'chapter_map',
    title: 'Run destructive pin actions and recover',
    what:
      'Use remove and swap controls at least once, then re-establish both pins so compute remains available.',
    impact:
      'This demonstrates safe recovery from destructive actions and confirms pin lifecycle behavior end-to-end.',
    targetIds: ['map.interactive'],
    required: [
      { actionId: 'map.popup_remove', label: 'Remove one marker from popup controls.' },
      { actionId: 'map.popup_swap', label: 'Swap start/destination once from popup or setup controls.' },
      { actionId: 'map.set_destination', label: 'Restore destination after destructive actions.' },
    ],
    prefillId: 'canonical_map',
  },
  {
    id: 'setup_vehicle',
    chapterId: 'chapter_setup',
    title: 'Choose vehicle profile',
    what:
      'Open Vehicle type and choose a profile. The tutorial highlights profile options so you can see their operational meaning.',
    impact:
      'Vehicle profile changes cost coefficients and emissions factors, directly affecting ranking and selected-route metrics.',
    targetIds: ['setup.vehicle'],
    required: [{ actionId: 'setup.vehicle_select', label: 'Select a vehicle option.' }],
    prefillId: 'canonical_setup',
  },
  {
    id: 'setup_scenario',
    chapterId: 'chapter_setup',
    title: 'Choose scenario mode',
    what:
      'Use Scenario mode to choose policy assumptions (no/partial/full sharing).',
    impact:
      'Scenario mode alters delay assumptions and can shift ETA, cost, and emissions deltas across candidate routes.',
    targetIds: ['setup.scenario'],
    required: [{ actionId: 'setup.scenario_select', label: 'Select a scenario option.' }],
  },
  {
    id: 'setup_language',
    chapterId: 'chapter_setup',
    title: 'Confirm language/local formatting control',
    what:
      'Interact with Language once. This control changes UI copy and number/date formatting behavior.',
    impact:
      'Locale changes improve operator readability across regions and affect how metrics are presented.',
    targetIds: ['setup.language'],
    required: [{ actionId: 'setup.language_select', label: 'Change language selection once.' }],
  },
  {
    id: 'setup_api_token_optional',
    chapterId: 'chapter_setup',
    title: 'Optional API token behavior',
    what:
      'API token is optional unless RBAC-protected APIs are enabled. Enter a token or explicitly keep default empty.',
    impact:
      'When RBAC is enabled, token presence determines access. In default local mode, leaving it empty is valid.',
    targetIds: ['setup.api_token'],
    required: [],
    optional: {
      id: 'setup.api_token_decision',
      label: 'Provide API token input or explicitly keep the default empty value.',
      actionIds: ['setup.api_token_input'],
      defaultLabel: 'Keep API token empty (default).',
    },
  },
  {
    id: 'map_overlay_toggles',
    chapterId: 'chapter_setup',
    title: 'Toggle map overlays',
    what:
      'Use Stops, Incidents, and Segments map overlay toggles to verify map-level visibility controls.',
    impact:
      'Overlay visibility controls are critical for debugging route context and inspecting incidents/segment-level detail.',
    targetIds: ['map.overlay_controls'],
    required: [
      { actionId: 'map.overlay_stops_toggle', label: 'Toggle Stops overlay.' },
      { actionId: 'map.overlay_incidents_toggle', label: 'Toggle Incidents overlay.' },
      { actionId: 'map.overlay_segments_toggle', label: 'Toggle Segments overlay.' },
    ],
  },
  {
    id: 'advanced_optimization_and_risk',
    chapterId: 'chapter_advanced',
    title: 'Set optimization mode and risk aversion',
    what:
      'Choose robust mode and adjust risk aversion. Robust mode penalizes uncertainty relative to expected-value scoring.',
    impact:
      'Higher risk aversion steers selection toward less volatile options, even when mean ETA may be slightly higher.',
    targetIds: ['advanced.optimization_mode', 'advanced.risk_aversion'],
    required: [
      { actionId: 'advanced.optimization_mode_select', label: 'Select optimization mode.' },
      { actionId: 'advanced.risk_aversion_input', label: 'Update risk aversion value.' },
    ],
    prefillId: 'canonical_advanced',
  },
  {
    id: 'advanced_pareto_method',
    chapterId: 'chapter_advanced',
    title: 'Switch Pareto method and set epsilon bounds',
    what:
      'Change Pareto method to epsilon-constraint and fill all three epsilon thresholds.',
    impact:
      'Epsilon caps can filter out infeasible high-cost/high-emission/high-duration candidates before final nondominance ranking.',
    targetIds: ['advanced.pareto_method', 'advanced.epsilon_grid'],
    required: [
      { actionId: 'advanced.pareto_method_select', label: 'Select pareto method.' },
      { actionId: 'advanced.epsilon_duration_input', label: 'Set epsilon duration bound.' },
      { actionId: 'advanced.epsilon_money_input', label: 'Set epsilon monetary bound.' },
      { actionId: 'advanced.epsilon_emissions_input', label: 'Set epsilon emissions bound.' },
    ],
  },
  {
    id: 'advanced_departure_optional',
    chapterId: 'chapter_advanced',
    title: 'Optional departure time in route requests',
    what:
      'Departure time can be set explicitly or left unset. Explicit time improves reproducibility and profile consistency.',
    impact:
      'Time-dependent effects (for example profile multipliers) are anchored to this timestamp when present.',
    targetIds: ['advanced.departure_time'],
    required: [],
    optional: {
      id: 'advanced.departure_time_decision',
      label: 'Set departure time or explicitly keep default inferred behavior.',
      actionIds: ['advanced.departure_time_input'],
      defaultLabel: 'Keep departure time at default behavior.',
    },
  },
  {
    id: 'advanced_stochastic',
    chapterId: 'chapter_advanced',
    title: 'Configure stochastic sampling',
    what:
      'Enable stochastic mode and set seed/sigma/samples.',
    impact:
      'These controls determine uncertainty spread and reproducibility; they influence robust scoring and route confidence summaries.',
    targetIds: ['advanced.stochastic_toggle', 'advanced.stochastic_grid'],
    required: [
      { actionId: 'advanced.stochastic_toggle', label: 'Enable stochastic sampling.' },
      { actionId: 'advanced.stochastic_seed_input', label: 'Set stochastic seed.' },
      { actionId: 'advanced.stochastic_sigma_input', label: 'Set stochastic sigma.' },
      { actionId: 'advanced.stochastic_samples_input', label: 'Set sample count.' },
    ],
  },
  {
    id: 'advanced_terrain_and_cost',
    chapterId: 'chapter_advanced',
    title: 'Set terrain and monetary cost toggles',
    what:
      'Pick terrain profile, keep tolls enabled, and set fuel/carbon/toll pricing knobs.',
    impact:
      'Terrain and cost toggles alter duration/emissions/cost surfaces and can materially change selected-route ranking.',
    targetIds: ['advanced.terrain', 'advanced.cost_toggles'],
    required: [
      { actionId: 'advanced.terrain_select', label: 'Select terrain profile.' },
      { actionId: 'advanced.use_tolls_toggle', label: 'Toggle use tolls.' },
      { actionId: 'advanced.fuel_multiplier_input', label: 'Set fuel price multiplier.' },
      { actionId: 'advanced.carbon_price_input', label: 'Set carbon price per kg.' },
      { actionId: 'advanced.toll_per_km_input', label: 'Set toll cost per km.' },
    ],
  },
  {
    id: 'preferences_weights',
    chapterId: 'chapter_routes',
    title: 'Tune objective weights',
    what:
      'Adjust time, money, and CO2 sliders before compute.',
    impact:
      'Weights control weighted-selection preference over the Pareto set and determine which candidate auto-selects.',
    targetIds: ['preferences.weights'],
    required: [
      { actionId: 'pref.weight_time', label: 'Adjust Time weight.' },
      { actionId: 'pref.weight_money', label: 'Adjust Money weight.' },
      { actionId: 'pref.weight_co2', label: 'Adjust CO2 weight.' },
    ],
    prefillId: 'canonical_preferences',
  },
  {
    id: 'preferences_compute',
    chapterId: 'chapter_routes',
    title: 'Compute Pareto candidates',
    what:
      'Run Compute Pareto and wait for completion.',
    impact:
      'This produces the candidate route set that drives charting, selected-route metrics, segment breakdown, and downstream comparisons.',
    targetIds: ['preferences.compute_button'],
    required: [
      { actionId: 'pref.compute_pareto_click', label: 'Click Compute Pareto.' },
      { actionId: 'pref.compute_pareto_done', label: 'Wait for route computation to finish.' },
    ],
  },
  {
    id: 'routes_select_from_chart',
    chapterId: 'chapter_routes',
    title: 'Select a route from the Pareto chart',
    what:
      'Click a point in the Pareto chart.',
    impact:
      'Chart selection updates selected-route metrics and all dependent panels (timeline, segment table, counterfactuals).',
    targetIds: ['routes.chart'],
    required: [{ actionId: 'routes.select_chart', label: 'Select one route from the chart.' }],
  },
  {
    id: 'routes_select_and_rename',
    chapterId: 'chapter_routes',
    title: 'Select from card list and rename route',
    what:
      'Select a route card, start rename, and save the new label.',
    impact:
      'Human-readable route labels improve comparison and reporting when many candidates are generated.',
    targetIds: ['routes.list'],
    required: [
      { actionId: 'routes.select_card', label: 'Select route from card list.' },
      { actionId: 'routes.rename_start', label: 'Start rename flow.' },
      { actionId: 'routes.rename_save', label: 'Save renamed route label.' },
    ],
  },
  {
    id: 'routes_names_reset',
    chapterId: 'chapter_routes',
    title: 'Reset custom route names',
    what:
      'Use Reset names after renaming to confirm recoverability.',
    impact:
      'Reset avoids stale labeling errors and returns deterministic default naming.',
    targetIds: ['routes.reset_names'],
    required: [{ actionId: 'routes.reset_names', label: 'Click Reset names.' }],
  },
  {
    id: 'selected_segment_controls',
    chapterId: 'chapter_routes',
    title: 'Use segment breakdown controls',
    what:
      'Expand segment table, show more, show fewer, and collapse again.',
    impact:
      'Segment-level controls support granular inspection without overwhelming the sidebar by default.',
    targetIds: ['selected.segment_breakdown'],
    required: [
      { actionId: 'selected.segment_expand', label: 'Expand segment breakdown.' },
      { actionId: 'selected.segment_collapse', label: 'Collapse segment breakdown.' },
    ],
    optional: {
      id: 'selected.segment_rows_decision',
      label: 'Use Show all/Show fewer row controls, or explicitly keep compact preview mode.',
      actionIds: ['selected.segment_show_all', 'selected.segment_show_fewer'],
      defaultLabel: 'Keep compact preview mode.',
    },
  },
  {
    id: 'selected_read_panels',
    chapterId: 'chapter_routes',
    title: 'Review explainability panels',
    what:
      'Review ETA explanation, ETA timeline chart, and counterfactual panel output for the selected route.',
    impact:
      'These panels explain why ETA changed and what-if sensitivity around your current selected route.',
    targetIds: ['selected.route_panel'],
    required: [{ actionId: 'manual.selected_review', label: 'Mark selected-route explainability review complete.', kind: 'manual' }],
  },
  {
    id: 'scenario_compare_run',
    chapterId: 'chapter_compare',
    title: 'Run scenario comparison',
    what:
      'Execute Compare scenarios and wait for result table/deltas.',
    impact:
      'Scenario comparison quantifies no/partial/full sharing impacts using identical routing context.',
    targetIds: ['compare.section'],
    required: [
      { actionId: 'compare.run_click', label: 'Click Compare scenarios.' },
      { actionId: 'compare.run_done', label: 'Wait for comparison response.' },
    ],
  },
  {
    id: 'departure_controls',
    chapterId: 'chapter_compare',
    title: 'Set departure search window',
    what:
      'Edit start/end/step controls and optionally arrival constraints.',
    impact:
      'Window and step define explored departures; arrival constraints enforce feasibility bounds.',
    targetIds: ['departure.section'],
    required: [
      { actionId: 'dep.window_start_input', label: 'Set window start.' },
      { actionId: 'dep.window_end_input', label: 'Set window end.' },
      { actionId: 'dep.step_input', label: 'Set step minutes.' },
    ],
    optional: {
      id: 'dep.arrival_constraints_decision',
      label: 'Set earliest/latest arrival constraints or explicitly keep them blank.',
      actionIds: ['dep.earliest_input', 'dep.latest_input'],
      defaultLabel: 'Keep arrival constraints empty (default).',
    },
    prefillId: 'canonical_departure',
  },
  {
    id: 'departure_run_and_apply',
    chapterId: 'chapter_compare',
    title: 'Run optimization and apply candidate',
    what:
      'Run Optimize departures and apply one candidate back to advanced departure-time state.',
    impact:
      'Applying optimized departure creates a reproducible handoff from search output to route request input.',
    targetIds: ['departure.section'],
    required: [
      { actionId: 'dep.optimize_click', label: 'Click Optimize departures.' },
      { actionId: 'dep.optimize_done', label: 'Wait for optimization results.' },
      { actionId: 'dep.apply_departure', label: 'Apply one suggested departure.' },
    ],
  },
  {
    id: 'timelapse_controls',
    chapterId: 'chapter_compare',
    title: 'Use scenario time-lapse controls',
    what:
      'Play route animation, adjust speed, scrub timeline, and reset playback.',
    impact:
      'Time-lapse supports route communication and spatial sanity checks for selected candidates.',
    targetIds: ['timelapse.section'],
    required: [
      { actionId: 'timelapse.play', label: 'Start playback.' },
      { actionId: 'timelapse.speed_select', label: 'Change playback speed.' },
      { actionId: 'timelapse.scrubber_input', label: 'Move scrubber slider.' },
      { actionId: 'timelapse.reset', label: 'Reset playback.' },
    ],
  },
  {
    id: 'duty_chain',
    chapterId: 'chapter_ops',
    title: 'Run duty chain planner',
    what:
      'Populate stops and run a multi-leg chain.',
    impact:
      'Duty chaining aggregates one-vehicle multi-leg outcomes and highlights operational totals across legs.',
    targetIds: ['duty.section'],
    required: [
      { actionId: 'duty.stops_input', label: 'Edit duty stops input.' },
      { actionId: 'duty.run_click', label: 'Run duty chain.' },
      { actionId: 'duty.run_done', label: 'Wait for duty-chain response.' },
    ],
    prefillId: 'canonical_duty',
  },
  {
    id: 'oracle_fields',
    chapterId: 'chapter_ops',
    title: 'Configure oracle quality check fields',
    what:
      'Fill source/check fields, schema/signature flags, and telemetry inputs.',
    impact:
      'These values feed dashboard reliability KPIs and source-level health aggregation.',
    targetIds: ['oracle.section'],
    required: [
      { actionId: 'oracle.source_input', label: 'Set oracle source.' },
      { actionId: 'oracle.schema_toggle', label: 'Toggle schema valid.' },
      { actionId: 'oracle.signature_select', label: 'Set signature state.' },
      { actionId: 'oracle.freshness_input', label: 'Set freshness seconds.' },
      { actionId: 'oracle.latency_input', label: 'Set latency ms.' },
      { actionId: 'oracle.record_count_input', label: 'Set record count.' },
    ],
    optional: {
      id: 'oracle.error_note_decision',
      label: 'Enter an error note or explicitly leave it empty.',
      actionIds: ['oracle.error_note_input'],
      defaultLabel: 'Keep error note empty (default).',
    },
    prefillId: 'canonical_oracle',
  },
  {
    id: 'oracle_actions',
    chapterId: 'chapter_ops',
    title: 'Run oracle dashboard actions',
    what:
      'Refresh dashboard, record check, and trigger CSV download action.',
    impact:
      'This validates dashboard refresh flow, ingestion flow, and artifact export access.',
    targetIds: ['oracle.section'],
    required: [
      { actionId: 'oracle.refresh_click', label: 'Click Refresh dashboard.' },
      { actionId: 'oracle.record_check_click', label: 'Click Record check.' },
      { actionId: 'oracle.record_check_done', label: 'Wait for ingestion completion.' },
      { actionId: 'oracle.download_csv_click', label: 'Trigger dashboard CSV download link.' },
    ],
  },
  {
    id: 'experiments_filters',
    chapterId: 'chapter_ops',
    title: 'Use experiment catalog filters',
    what:
      'Interact with search/filter/sort and apply catalog filters.',
    impact:
      'Catalog controls scale experiment retrieval workflows when bundles grow over time.',
    targetIds: ['experiments.section'],
    required: [
      { actionId: 'exp.search_input', label: 'Edit catalog search.' },
      { actionId: 'exp.filter_vehicle_select', label: 'Select vehicle filter.' },
      { actionId: 'exp.filter_scenario_select', label: 'Select scenario filter.' },
      { actionId: 'exp.sort_select', label: 'Select catalog sort.' },
      { actionId: 'exp.apply_filters_click', label: 'Apply catalog filters.' },
    ],
  },
  {
    id: 'experiments_lifecycle',
    chapterId: 'chapter_ops',
    title: 'Run experiment lifecycle actions',
    what:
      'Save bundle, load it, replay compare, and delete bundle.',
    impact:
      'This covers full experiment lifecycle for reproducibility and quick scenario iteration.',
    targetIds: ['experiments.section'],
    required: [
      { actionId: 'exp.name_input', label: 'Set experiment name.' },
      { actionId: 'exp.description_input', label: 'Set experiment description.' },
      { actionId: 'exp.save_click', label: 'Save current bundle.' },
      { actionId: 'exp.load_click', label: 'Load saved bundle.' },
      { actionId: 'exp.replay_click', label: 'Run compare from bundle.' },
      { actionId: 'exp.delete_click', label: 'Delete saved bundle.' },
    ],
    prefillId: 'canonical_experiment',
  },
  {
    id: 'tutorial_completion',
    chapterId: 'chapter_finish',
    title: 'Complete tutorial and recap',
    what:
      'Confirm completion after exercising every major frontend feature group and destructive control path.',
    impact:
      'Completion marks onboarding done while still keeping manual restart available from Setup.',
    targetIds: ['setup.section'],
    required: [{ actionId: 'manual.final_recap', label: 'Mark recap as reviewed.', kind: 'manual' }],
    allowMissingTarget: true,
  },
];
