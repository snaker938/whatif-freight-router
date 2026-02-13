import type { TutorialChapter, TutorialStep } from './types';

export const TUTORIAL_CHAPTERS: TutorialChapter[] = [
  {
    id: 'chapter_map',
    title: '1. Orientation and map basics',
    description: 'Set and manage pins, then use marker actions for map-side operations.',
  },
  {
    id: 'chapter_setup',
    title: '2. Setup and core context',
    description: 'Configure vehicle/scenario context and optional API behavior.',
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
    title: 'Place Start near Newcastle and End near London',
    what:
      'Follow the guided map targets: place Start within the Newcastle zone first, then place End within the London zone. After both are placed, click each marker and drag each marker once.',
    impact:
      'Every downstream result is anchored to these coordinates. This strict placement demonstrates reproducible origin/destination setup before route optimization.',
    targetIds: ['map.interactive'],
    required: [
      {
        actionId: 'map.set_origin_newcastle',
        label: 'Place Start in the Newcastle guided zone.',
        details: 'Clicks outside the highlighted zone are rejected.',
      },
      {
        actionId: 'map.set_destination_london',
        label: 'Place End in the London guided zone.',
        details: 'After Start is valid, the map auto-centers London.',
      },
      {
        actionId: 'map.click_origin_marker',
        label: 'Click the Start marker once.',
        details: 'This verifies marker selection and popup focus.',
      },
      {
        actionId: 'map.click_destination_marker',
        label: 'Click the End marker once.',
        details: 'Use marker click interaction on the End pin.',
      },
      {
        actionId: 'map.drag_origin_marker',
        label: 'Drag the Start marker once.',
        details: 'Drag Start to a nearby location and drop.',
      },
      {
        actionId: 'map.drag_destination_marker',
        label: 'Drag the End marker once.',
        details: 'Drag End to a nearby location and drop.',
      },
    ],
    prefillId: 'clear_map',
    lockScope: 'map_only',
    mapGuide: {
      mode: 'pin_sequence',
      sequence: [
        {
          pin: 'origin',
          city: 'newcastle',
          lat: 54.9783,
          lon: -1.6178,
          radiusKm: 20,
          zoom: 11,
        },
        {
          pin: 'destination',
          city: 'london',
          lat: 51.5072,
          lon: -0.1276,
          radiusKm: 20,
          zoom: 10,
        },
      ],
    },
  },
  {
    id: 'map_popup_actions',
    chapterId: 'chapter_map',
    title: 'Use marker popups for fast pin operations',
    what:
      'Click a marker to open its popup, copy coordinates, then close the popup using the in-popup close action.',
    impact:
      'Popup actions let you inspect and manage markers without leaving map context, reducing context switching during route setup.',
    targetIds: ['map.interactive'],
    required: [
      { actionId: 'map.popup_copy', label: 'Copy coordinates from any marker popup.' },
      { actionId: 'map.popup_close', label: 'Close a marker popup using its close action.' },
    ],
    lockScope: 'map_only',
  },
  {
    id: 'map_stop_lifecycle',
    chapterId: 'chapter_map',
    title: 'Run full stop lifecycle actions on the map',
    what:
      'Add a midpoint stop from a marker popup, rename it, drag it to a nearby location, and delete it.',
    impact:
      'This proves stop lifecycle reliability: creation, editing, repositioning, and deletion all work without destabilizing Start/End pins.',
    targetIds: ['map.interactive'],
    required: [
      { actionId: 'map.add_stop_midpoint', label: 'Add or replace a midpoint stop from a pin popup.' },
      { actionId: 'map.rename_stop', label: 'Rename the stop from map popup controls.' },
      { actionId: 'map.drag_stop_marker', label: 'Drag the stop marker to a nearby location.' },
      { actionId: 'map.delete_stop', label: 'Delete the stop from stop controls.' },
    ],
    prefillId: 'canonical_map',
    lockScope: 'map_only',
  },
  {
    id: 'setup_vehicle',
    chapterId: 'chapter_setup',
    title: 'Choose a vehicle profile',
    what:
      'Open Vehicle type and actively pick a profile. Review the option hint text before selecting.',
    impact:
      'Vehicle profile changes cost/emissions coefficients and can reorder best-route selection under the same OD pair.',
    targetIds: ['setup.vehicle'],
    required: [{ actionId: 'setup.vehicle_select', label: 'Select a vehicle option.' }],
    prefillId: 'canonical_setup',
    activeSectionId: 'setup.section',
    lockScope: 'sidebar_section_only',
    allowedActions: ['setup.vehicle_option:*'],
  },
  {
    id: 'setup_scenario',
    chapterId: 'chapter_setup',
    title: 'Choose scenario mode',
    what:
      'Switch Scenario mode between sharing policies to understand available assumptions.',
    impact:
      'Scenario mode changes delay assumptions and can materially shift ETA, money, and CO2 values across candidates.',
    targetIds: ['setup.scenario'],
    required: [{ actionId: 'setup.scenario_select', label: 'Select a scenario option.' }],
    activeSectionId: 'setup.section',
    lockScope: 'sidebar_section_only',
    allowedActions: ['setup.scenario_option:*'],
  },
  {
    id: 'setup_language',
    chapterId: 'chapter_setup',
    title: 'Confirm language/local formatting control',
    what:
      'Change Language once to verify locale-aware labels and number/date presentation.',
    impact:
      'Locale improves operator readability and changes how metrics are rendered during review and reporting.',
    targetIds: ['setup.language'],
    required: [{ actionId: 'setup.language_select', label: 'Change language selection once.' }],
    activeSectionId: 'setup.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'setup_api_token_optional',
    chapterId: 'chapter_setup',
    title: 'Optional API token behavior',
    what:
      'API token is optional in local mode. Enter one if needed for protected environments, or explicitly keep it blank.',
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
    activeSectionId: 'setup.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'setup_map_actions',
    chapterId: 'chapter_setup',
    title: 'Use setup map action buttons',
    what:
      'Run setup map actions once: clear pins, restore sample pins, fit map to pins, then swap Start/End.',
    impact:
      'These quick actions speed up scenario setup and let operators recover, reframe, and reorder endpoints without manual re-entry.',
    targetIds: ['setup.section'],
    required: [
      { actionId: 'setup.clear_pins_button', label: 'Clear all pins once.' },
      { actionId: 'setup.sample_pins_button', label: 'Restore sample Start/End pins.' },
      { actionId: 'setup.fit_map_button', label: 'Fit map to current pins.' },
      { actionId: 'setup.swap_pins_button', label: 'Swap Start and End once.' },
    ],
    activeSectionId: 'setup.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'pins_sidebar_manage',
    chapterId: 'chapter_setup',
    title: 'Manage pin state from Pins and Stops',
    what:
      'Use the Pins and Stops section to select pins, add or replace a stop, rename the stop, and remove it.',
    impact:
      'Sidebar pin controls provide deterministic pin-state management without relying on map popups during constrained workflows.',
    targetIds: ['pins.section'],
    required: [
      { actionId: 'pins.sidebar_select', label: 'Select or deselect a pin from the sidebar rail/list.' },
      { actionId: 'pins.add_stop', label: 'Add or replace the midpoint stop from sidebar controls.' },
      { actionId: 'pins.rename_stop', label: 'Rename Stop #1 from the sidebar input.' },
      { actionId: 'pins.delete_stop', label: 'Delete the stop from sidebar controls.' },
    ],
    prefillId: 'canonical_map',
    activeSectionId: 'pins.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'advanced_optimization_and_risk',
    chapterId: 'chapter_advanced',
    title: 'Set optimization mode and risk aversion',
    what:
      'Set optimization mode and risk aversion explicitly. Compare expected-value versus robust behavior.',
    impact:
      'Higher risk aversion penalizes volatile options and can prioritize stability over minimum mean ETA.',
    targetIds: ['advanced.optimization_mode', 'advanced.risk_aversion'],
    required: [
      { actionId: 'advanced.optimization_mode_select', label: 'Select optimization mode.' },
      { actionId: 'advanced.risk_aversion_input', label: 'Update risk aversion value.' },
    ],
    prefillId: 'canonical_advanced',
    activeSectionId: 'advanced.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'advanced_pareto_method',
    chapterId: 'chapter_advanced',
    title: 'Switch Pareto method and set epsilon bounds',
    what:
      'Switch to epsilon-constraint mode and fill duration, monetary, and emissions caps.',
    impact:
      'Epsilon caps pre-filter infeasible candidates before nondominance selection, changing the final candidate set.',
    targetIds: ['advanced.pareto_method', 'advanced.epsilon_grid'],
    required: [
      { actionId: 'advanced.pareto_method_select', label: 'Select pareto method.' },
      { actionId: 'advanced.epsilon_duration_input', label: 'Set epsilon duration bound.' },
      { actionId: 'advanced.epsilon_money_input', label: 'Set epsilon monetary bound.' },
      { actionId: 'advanced.epsilon_emissions_input', label: 'Set epsilon emissions bound.' },
    ],
    activeSectionId: 'advanced.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'advanced_departure_optional',
    chapterId: 'chapter_advanced',
    title: 'Optional departure time in route requests',
    what:
      'Set a departure time or explicitly keep default behavior. This field is optional but important for reproducible runs.',
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
    activeSectionId: 'advanced.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'advanced_stochastic',
    chapterId: 'chapter_advanced',
    title: 'Configure stochastic sampling',
    what:
      'Enable stochastic sampling and set seed, sigma, and sample count.',
    impact:
      'These values control uncertainty spread and reproducibility, which directly influences robust ranking outcomes.',
    targetIds: ['advanced.stochastic_toggle', 'advanced.stochastic_grid'],
    required: [
      { actionId: 'advanced.stochastic_toggle', label: 'Enable stochastic sampling.' },
      { actionId: 'advanced.stochastic_seed_input', label: 'Set stochastic seed.' },
      { actionId: 'advanced.stochastic_sigma_input', label: 'Set stochastic sigma.' },
      { actionId: 'advanced.stochastic_samples_input', label: 'Set sample count.' },
    ],
    activeSectionId: 'advanced.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'advanced_terrain_and_cost',
    chapterId: 'chapter_advanced',
    title: 'Set terrain and monetary cost toggles',
    what:
      'Set terrain profile and cost toggles (tolls, fuel multiplier, carbon price, toll per km).',
    impact:
      'These knobs alter duration, emissions, and monetary surfaces, which can reorder selected-route outcomes.',
    targetIds: ['advanced.terrain', 'advanced.cost_toggles'],
    required: [
      { actionId: 'advanced.terrain_select', label: 'Select terrain profile.' },
      { actionId: 'advanced.use_tolls_toggle', label: 'Toggle use tolls.' },
      { actionId: 'advanced.fuel_multiplier_input', label: 'Set fuel price multiplier.' },
      { actionId: 'advanced.carbon_price_input', label: 'Set carbon price per kg.' },
      { actionId: 'advanced.toll_per_km_input', label: 'Set toll cost per km.' },
    ],
    activeSectionId: 'advanced.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'preferences_weights',
    chapterId: 'chapter_routes',
    title: 'Tune objective weights',
    what:
      'Adjust time, money, and CO2 sliders to set your optimization preference before computing routes.',
    impact:
      'Weights control weighted-selection preference over the Pareto set and determine which candidate auto-selects.',
    targetIds: ['preferences.weights'],
    required: [
      { actionId: 'pref.weight_time', label: 'Adjust Time weight.' },
      { actionId: 'pref.weight_money', label: 'Adjust Money weight.' },
      { actionId: 'pref.weight_co2', label: 'Adjust CO2 weight.' },
    ],
    prefillId: 'canonical_preferences',
    activeSectionId: 'preferences.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'preferences_compute',
    chapterId: 'chapter_routes',
    title: 'Compute Pareto candidates',
    what:
      'Run Compute Pareto and wait until candidate generation completes.',
    impact:
      'This produces the candidate route set that drives charting, selected-route metrics, segment breakdown, and downstream comparisons.',
    targetIds: ['preferences.compute_button'],
    required: [
      { actionId: 'pref.compute_pareto_click', label: 'Click Compute Pareto.' },
      { actionId: 'pref.compute_pareto_done', label: 'Wait for route computation to finish.' },
    ],
    activeSectionId: 'preferences.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'routes_sort_options',
    chapterId: 'chapter_routes',
    title: 'Apply route sorting control',
    what:
      'Change the route sort selector to inspect candidate ordering by ETA, cost, or emissions.',
    impact:
      'Sorting improves review flow and helps you inspect candidate sets from different operational priorities.',
    targetIds: ['routes.list'],
    required: [{ actionId: 'routes.sort_select', label: 'Change route sort order once.' }],
    activeSectionId: 'routes.list',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'routes_select_from_chart',
    chapterId: 'chapter_routes',
    title: 'Select a route from the Pareto chart',
    what:
      'Click any Pareto chart point to select a specific candidate.',
    impact:
      'Chart selection updates selected-route metrics and all dependent panels (timeline, segment table, counterfactuals).',
    targetIds: ['routes.chart'],
    required: [{ actionId: 'routes.select_chart', label: 'Select one route from the chart.' }],
    activeSectionId: 'routes.list',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'routes_select_and_rename',
    chapterId: 'chapter_routes',
    title: 'Select from card list and rename route',
    what:
      'Select a card, open rename, and save a custom route label.',
    impact:
      'Human-readable route labels improve comparison and reporting when many candidates are generated.',
    targetIds: ['routes.list'],
    required: [
      { actionId: 'routes.select_card', label: 'Select route from card list.' },
      { actionId: 'routes.rename_start', label: 'Start rename flow.' },
      { actionId: 'routes.rename_save', label: 'Save renamed route label.' },
    ],
    activeSectionId: 'routes.list',
    lockScope: 'sidebar_section_only',
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
    activeSectionId: 'routes.list',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'selected_panel_utilities',
    chapterId: 'chapter_routes',
    title: 'Use selected route utility actions',
    what:
      'Use the selected-route utility controls to fit the map to the active route and copy its metrics summary.',
    impact:
      'These controls speed QA and reporting by centering route context and extracting quick summaries.',
    targetIds: ['selected.route_panel'],
    required: [
      { actionId: 'selected.fit_route', label: 'Click Fit Route To Map.' },
      { actionId: 'selected.copy_summary', label: 'Copy selected-route summary once.' },
    ],
    activeSectionId: 'selected.route_panel',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'selected_segment_controls',
    chapterId: 'chapter_routes',
    title: 'Use segment breakdown controls',
    what:
      'Expand the segment table, inspect rows, then collapse again. Optionally use row expansion controls.',
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
    activeSectionId: 'selected.route_panel',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'selected_read_panels',
    chapterId: 'chapter_routes',
    title: 'Validate selected-route explainability panels',
    what:
      'Click inside the Selected route panel and confirm the explainability subpanels render for the active route.',
    impact:
      'These panels explain ETA deltas and sensitivity, which is essential for auditability and decision communication.',
    targetIds: ['selected.route_panel'],
    required: [
      {
        actionId: 'selected.panel_click',
        label: 'Click inside the Selected route panel.',
        details: 'Interact with the panel to confirm focus.',
      },
      {
        actionId: 'selected.panel_data_ready',
        label: 'Confirm selected-route metrics are available.',
        details: 'Route metrics must be present for explainability review.',
      },
      {
        actionId: 'selected.timeline_panel_visible',
        label: 'Confirm ETA timeline panel is visible.',
        details: 'Timeline chart section should render for the selected route.',
      },
      {
        actionId: 'selected.counterfactual_panel_visible',
        label: 'Confirm counterfactual panel is visible.',
        details: 'Counterfactual section should be shown for selected route context.',
      },
    ],
    activeSectionId: 'selected.route_panel',
    lockScope: 'sidebar_section_only',
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
    activeSectionId: 'compare.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'departure_controls',
    chapterId: 'chapter_compare',
    title: 'Set departure search window',
    what:
      'Edit search window start/end and step size, then optionally set arrival constraints.',
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
    activeSectionId: 'departure.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'departure_run_and_apply',
    chapterId: 'chapter_compare',
    title: 'Run optimization and apply candidate',
    what:
      'Run departure optimization and apply a suggested candidate back into advanced parameters.',
    impact:
      'Applying optimized departure creates a reproducible handoff from search output to route request input.',
    targetIds: ['departure.section'],
    required: [
      { actionId: 'dep.optimize_click', label: 'Click Optimize departures.' },
      { actionId: 'dep.optimize_done', label: 'Wait for optimization results.' },
      { actionId: 'dep.apply_departure', label: 'Apply one suggested departure.' },
    ],
    activeSectionId: 'departure.section',
    lockScope: 'sidebar_section_only',
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
      { actionId: 'timelapse.back_10', label: 'Use the back 10% jump control.' },
      { actionId: 'timelapse.forward_10', label: 'Use the forward 10% jump control.' },
      { actionId: 'timelapse.jump_start', label: 'Jump playback to route start.' },
      { actionId: 'timelapse.jump_end', label: 'Jump playback to route end.' },
      { actionId: 'timelapse.reset', label: 'Reset playback.' },
    ],
    activeSectionId: 'timelapse.section',
    lockScope: 'sidebar_section_only',
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
    activeSectionId: 'duty.section',
    lockScope: 'sidebar_section_only',
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
    activeSectionId: 'oracle.section',
    lockScope: 'sidebar_section_only',
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
    activeSectionId: 'oracle.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'oracle_utility_controls',
    chapterId: 'chapter_ops',
    title: 'Use oracle utility controls',
    what:
      'Load sample check values, then clear the form back to defaults.',
    impact:
      'Utility controls accelerate repeat testing and safe reset between oracle quality checks.',
    targetIds: ['oracle.section'],
    required: [
      { actionId: 'oracle.load_sample_click', label: 'Load sample oracle values once.' },
      { actionId: 'oracle.clear_form_click', label: 'Clear oracle form fields once.' },
    ],
    activeSectionId: 'oracle.section',
    lockScope: 'sidebar_section_only',
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
    activeSectionId: 'experiments.section',
    lockScope: 'sidebar_section_only',
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
    activeSectionId: 'experiments.section',
    lockScope: 'sidebar_section_only',
  },
  {
    id: 'experiments_utility_controls',
    chapterId: 'chapter_ops',
    title: 'Use experiment utility controls',
    what:
      'Refresh experiment catalog data and run the clear-filters shortcut.',
    impact:
      'Utility actions keep catalog views deterministic during repeated save/load cycles.',
    targetIds: ['experiments.section'],
    required: [
      { actionId: 'exp.refresh_click', label: 'Refresh experiment catalog once.' },
      { actionId: 'exp.clear_filters_click', label: 'Run clear-filters once.' },
    ],
    activeSectionId: 'experiments.section',
    lockScope: 'sidebar_section_only',
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
