'use client';

import { useEffect, useMemo, useState } from 'react';

import type {
  DecisionProofContext,
  PreferenceCompatibleSetSummary,
  PreferenceQuery,
  PreferenceQueryTrace,
  PreferenceShrinkageTrace,
  PairwisePreferenceQuery,
  PreferenceState,
  PreferenceSummary,
  RouteOption,
  ThresholdPreferenceQuery,
  TimeGuardPreferenceQuery,
  VetoPreferenceQuery,
} from '../lib/types';

type MetricKey = 'duration_s' | 'monetary_cost' | 'emissions_kg' | 'distance_km' | 'avg_speed_kmh';
type PreferenceChangeHandler = (nextPreferenceState: PreferenceState, selectedRouteId?: string | null) => void;

type Props = {
  locale: string;
  selectedRoute: RouteOption | null;
  candidateRoutes: RouteOption[];
  preferenceState?: PreferenceState | null;
  preferenceQueryTrace?: PreferenceQueryTrace | null;
  preferenceSummary?: PreferenceSummary | null;
  proofContext?: DecisionProofContext | null;
  syncPending?: boolean;
  syncError?: string | null;
  onPreferenceChange?: PreferenceChangeHandler;
};

const metricOrder: MetricKey[] = ['duration_s', 'monetary_cost', 'emissions_kg', 'distance_km', 'avg_speed_kmh'];

const metricLabel: Record<MetricKey, string> = {
  duration_s: 'Duration',
  monetary_cost: 'Cost',
  emissions_kg: 'Emissions',
  distance_km: 'Distance',
  avg_speed_kmh: 'Speed',
};

const metricDirection: Record<MetricKey, 'lower' | 'higher'> = {
  duration_s: 'lower',
  monetary_cost: 'lower',
  emissions_kg: 'lower',
  distance_km: 'lower',
  avg_speed_kmh: 'higher',
};

function cloneQuery(query: PreferenceQuery): PreferenceQuery {
  return query.query_type === 'pairwise'
    ? { ...query, weight_hint: query.weight_hint ? { ...query.weight_hint } : null }
    : { ...query };
}

function cloneRouteIds(routeIds: string[] | null | undefined): string[] | undefined {
  return routeIds ? [...routeIds] : undefined;
}

function cloneCompatibleSetSummary(summary: PreferenceCompatibleSetSummary | null | undefined): PreferenceCompatibleSetSummary | null {
  return summary
    ? {
        ...summary,
        route_ids: cloneRouteIds(summary.route_ids),
        necessary_best_route_ids: cloneRouteIds(summary.necessary_best_route_ids),
        possible_best_route_ids: cloneRouteIds(summary.possible_best_route_ids),
      }
    : null;
}

function cloneState(state: PreferenceState | null | undefined): PreferenceState {
  return {
    compatible_set_summary: cloneCompatibleSetSummary(state?.compatible_set_summary),
    compatible_weights: state?.compatible_weights ? state.compatible_weights.map((item) => ({ ...item })) : undefined,
    pairwise_constraints: state?.pairwise_constraints ? state.pairwise_constraints.map((item) => ({ ...item })) : undefined,
    threshold_constraints: state?.threshold_constraints ? state.threshold_constraints.map((item) => ({ ...item })) : undefined,
    ratio_constraints: state?.ratio_constraints ? state.ratio_constraints.map((item) => ({ ...item })) : undefined,
    veto_rules: state?.veto_rules ? state.veto_rules.map((item) => ({ ...item })) : undefined,
    time_preserving_guard_rules: state?.time_preserving_guard_rules ? state.time_preserving_guard_rules.map((item) => ({ ...item })) : undefined,
    query_history: state?.query_history ? state.query_history.map(cloneQuery) : undefined,
    shrinkage_trace: state?.shrinkage_trace ? state.shrinkage_trace.map((item) => ({ ...item })) : undefined,
    contradiction_record: state?.contradiction_record ? { ...state.contradiction_record } : null,
    derived_invariants: state?.derived_invariants ? { ...state.derived_invariants } : null,
    terminal_type: state?.terminal_type ?? null,
    preference_irrelevance_proven: state?.preference_irrelevance_proven ?? null,
    no_query_reason: state?.no_query_reason ?? null,
    no_preference_query_reason: state?.no_preference_query_reason ?? null,
    query_count: state?.query_count ?? null,
  };
}

function n(locale: string, value: unknown, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? new Intl.NumberFormat(locale, { maximumFractionDigits: digits }).format(value)
    : 'n/a';
}

function pct(locale: string, value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? new Intl.NumberFormat(locale, { style: 'percent', maximumFractionDigits: 1 }).format(value)
    : 'n/a';
}

function secs(locale: string, value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
  if (Math.abs(value) >= 3600) return `${n(locale, value / 3600, 1)} h`;
  if (Math.abs(value) >= 60) return `${n(locale, value / 60, 1)} min`;
  return `${n(locale, value, 0)} s`;
}

function routeMetric(route: RouteOption | null, metric: MetricKey): number {
  if (!route) return 0;
  return metric === 'duration_s'
    ? route.metrics.duration_s
    : metric === 'monetary_cost'
      ? route.metrics.monetary_cost
      : metric === 'emissions_kg'
        ? route.metrics.emissions_kg
        : metric === 'distance_km'
          ? route.metrics.distance_km
          : route.metrics.avg_speed_kmh;
}

function metricText(locale: string, metric: MetricKey, value: number): string {
  return metric === 'duration_s'
    ? secs(locale, value)
    : metric === 'distance_km'
      ? `${n(locale, value, 1)} km`
      : metric === 'avg_speed_kmh'
        ? `${n(locale, value, 1)} km/h`
        : n(locale, value, 2);
}

function routeSummary(locale: string, route: RouteOption | null): string {
  return route
    ? `${secs(locale, route.metrics.duration_s)} | ${n(locale, route.metrics.monetary_cost, 2)} cost | ${n(locale, route.metrics.emissions_kg, 2)} kg CO2`
    : 'n/a';
}

function dedupeRoutes(selectedRoute: RouteOption | null, candidateRoutes: RouteOption[]): RouteOption[] {
  const out: RouteOption[] = [];
  const seen = new Set<string>();
  const add = (route: RouteOption | null | undefined) => {
    if (!route?.id) return;
    const id = route.id.trim();
    if (!id || seen.has(id)) return;
    seen.add(id);
    out.push(route);
  };
  add(selectedRoute);
  candidateRoutes.forEach(add);
  return out;
}

function scoreRoute(route: RouteOption, routes: RouteOption[]): number {
  const metrics: MetricKey[] = ['duration_s', 'monetary_cost', 'emissions_kg', 'distance_km', 'avg_speed_kmh'];
  return metrics.reduce((sum, metric) => {
    const values = routes.map((item) => routeMetric(item, metric));
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (max === min) return sum + 0.5;
    const value = routeMetric(route, metric);
    const normalized = metricDirection[metric] === 'lower' ? 1 - (value - min) / (max - min) : (value - min) / (max - min);
    return sum + Math.max(0, Math.min(1, normalized));
  }, 0) / metrics.length;
}

function comparison(locale: string, winner: RouteOption | null, loser: RouteOption | null): string {
  if (!winner || !loser) return 'No comparison available.';
  const pieces = metricOrder
    .map((metric) => {
      const a = routeMetric(winner, metric);
      const b = routeMetric(loser, metric);
      const delta = metricDirection[metric] === 'lower' ? b - a : a - b;
      return delta > 0 ? `${metricLabel[metric]} ${metricDirection[metric] === 'lower' ? 'improves by' : 'leads by'} ${metricText(locale, metric, Math.abs(delta))}` : null;
    })
    .filter((value): value is string => Boolean(value));
  return pieces.length ? pieces.slice(0, 2).join('; ') : `${winner.id} and ${loser.id} are close on live metrics.`;
}

function pickRoutes(state: PreferenceState, routes: RouteOption[], selectedRouteId: string | null): PreferenceCompatibleSetSummary {
  const queries = state.query_history ?? [];
  const blocked = new Set<string>();
  for (const query of queries) {
    if (query.query_type === 'pairwise') blocked.add(query.challenger_route_id);
    if (query.query_type === 'threshold') {
      const route = routes.find((item) => item.id === query.route_id);
      const value = route ? routeMetric(route, query.metric_name as MetricKey) : null;
      const failed =
        value === null ||
        (query.direction === 'gte' ? value < query.threshold_value : value > query.threshold_value);
      if (failed) blocked.add(query.route_id);
    }
    if (query.query_type === 'ratio') {
      const route = routes.find((item) => item.id === query.route_id);
      const numerator = route ? routeMetric(route, query.numerator_metric as MetricKey) : null;
      const denominator = route ? routeMetric(route, query.denominator_metric as MetricKey) : null;
      const ratio = numerator !== null && denominator ? numerator / denominator : null;
      if (ratio === null || ratio < query.minimum_ratio) blocked.add(query.route_id);
    }
    if (query.query_type === 'veto' && query.active !== false) blocked.add(query.route_id);
    if (query.query_type === 'time_guard') {
      const route = routes.find((item) => item.id === query.route_id);
      if (!route) continue;
      const limit = query.max_travel_time_s ?? route.metrics.duration_s + (query.preserve_time_budget_s ?? 0);
      if (route.metrics.duration_s > limit) blocked.add(query.route_id);
    }
  }
  const survivors = routes.filter((route) => !blocked.has(route.id));
  const survivorIds = survivors.map((route) => route.id);
  const survivorCount = survivors.length;
  const selected = routes.find((route) => route.id === selectedRouteId) ?? routes[0] ?? null;
  const rank = selected ? [...survivors].sort((a, b) => scoreRoute(b, routes) - scoreRoute(a, routes)).findIndex((route) => route.id === selected.id) + 1 : 0;
  const preservedPossibleRouteIds =
    state.compatible_set_summary?.possible_best_route_ids?.filter((routeId) => survivorIds.includes(routeId)) ?? [];
  const preservedNecessaryRouteIds =
    state.compatible_set_summary?.necessary_best_route_ids?.filter((routeId) => survivorIds.includes(routeId)) ?? [];
  const possibleBestRouteIds = survivorCount
    ? preservedPossibleRouteIds.length
      ? preservedPossibleRouteIds
      : survivorIds
    : [];
  const necessaryBestRouteIds = survivorCount
    ? preservedNecessaryRouteIds.length
      ? preservedNecessaryRouteIds
      : survivorCount === 1
        ? survivorIds
        : []
    : [];
  const volumeProxy = routes.length ? Math.min(1, Math.max(0, (survivorCount / routes.length) * (survivorCount <= 1 ? 0.25 : 0.6))) : 0;
  return {
    route_ids: survivorIds,
    compatible_set_size: survivorCount,
    compatible_set_volume_proxy: volumeProxy,
    necessary_best_prob: survivorCount && rank === 1 ? 1 / survivorCount : 0,
    possible_best_prob: rank > 0 ? 1 / rank : 0,
    necessary_best_route_ids: necessaryBestRouteIds,
    possible_best_route_ids: possibleBestRouteIds,
    support_flag: survivorCount > 0,
    support_reason: survivorCount > 0 ? `${survivorCount} / ${routes.length} routes survive the live monotone filters.` : 'No route survives the current live monotone filters.',
  };
}

function finalize(state: PreferenceState, routes: RouteOption[], selectedRouteId: string | null): PreferenceState {
  const next = cloneState(state);
  const compatible = pickRoutes(next, routes, selectedRouteId);
  next.compatible_set_summary = compatible;
  next.query_history = next.query_history?.map(cloneQuery);
  next.shrinkage_trace = next.shrinkage_trace ? next.shrinkage_trace.map((item) => ({ ...item })) : [];
  next.query_count = next.query_history?.length ?? 0;
  next.terminal_type = compatible.compatible_set_size === 0 ? 'abstained' : compatible.compatible_set_size === 1 ? 'certified' : 'open';
  next.derived_invariants = {
    ...next.derived_invariants,
    live_runtime_payload: true,
    monotone_updates_only: true,
    pairwise_family_present: Boolean(next.pairwise_constraints?.length),
    tradeoff_family_present: Boolean(next.threshold_constraints?.length || next.ratio_constraints?.length),
    veto_family_present: Boolean(next.veto_rules?.length),
    time_guard_family_present: Boolean(next.time_preserving_guard_rules?.length),
  };
  return next;
}

function appendQuery(
  state: PreferenceState,
  query: PreferenceQuery,
  routes: RouteOption[],
  selectedRouteId: string | null,
): PreferenceState {
  const base = cloneState(state);
  base.query_history = [...(base.query_history ?? []), cloneQuery(query)];
  if (query.query_type === 'pairwise') {
    base.pairwise_constraints = [...(base.pairwise_constraints ?? []), { ...query }];
  }
  if (query.query_type === 'threshold') {
    base.threshold_constraints = [...(base.threshold_constraints ?? []), { ...query }];
  }
  if (query.query_type === 'ratio') {
    base.ratio_constraints = [...(base.ratio_constraints ?? []), { ...query }];
  }
  if (query.query_type === 'veto') {
    base.veto_rules = [...(base.veto_rules ?? []), { ...query }];
  }
  if (query.query_type === 'time_guard') {
    base.time_preserving_guard_rules = [...(base.time_preserving_guard_rules ?? []), { ...query }];
  }
  const before = pickRoutes(state, routes, selectedRouteId);
  const after = finalize(base, routes, selectedRouteId);
  after.shrinkage_trace = [
    ...(after.shrinkage_trace ?? []),
    {
      query_index: after.query_history?.length ?? 1,
      query_type: query.query_type,
      before_size: before.compatible_set_size ?? routes.length,
      after_size: after.compatible_set_summary?.compatible_set_size ?? 0,
      before_volume_proxy: before.compatible_set_volume_proxy ?? 0,
      after_volume_proxy: after.compatible_set_summary?.compatible_set_volume_proxy ?? 0,
      predicted_shrinkage: before.compatible_set_size
        ? Math.max(0, Math.min(1, 1 - (after.compatible_set_summary?.compatible_set_size ?? 0) / before.compatible_set_size))
        : null,
      realized_shrinkage: before.compatible_set_size
        ? Math.max(0, Math.min(1, 1 - (after.compatible_set_summary?.compatible_set_size ?? 0) / before.compatible_set_size))
        : null,
      target_route_id: 'route_id' in query ? query.route_id : query.query_type === 'pairwise' ? query.preferred_route_id : selectedRouteId,
      query_reason:
        query.query_type === 'pairwise'
          ? `pairwise ${query.preferred_route_id} over ${query.challenger_route_id}`
          : query.query_type === 'threshold'
            ? `threshold on ${query.metric_name}`
            : query.query_type === 'ratio'
              ? `ratio on ${query.numerator_metric}/${query.denominator_metric}`
              : query.query_type === 'veto'
                ? `veto ${query.veto_name}`
                : `time guard for ${query.route_id}`,
    },
  ];
  return finalize(after, routes, selectedRouteId);
}

function mergeInitialState(
  preferenceState: PreferenceState | null | undefined,
  preferenceTrace: PreferenceQueryTrace | null | undefined,
  preferenceSummary: PreferenceSummary | null | undefined,
): PreferenceState {
  const draft = cloneState(preferenceState ?? preferenceSummary?.preference_state ?? null);
  const traceCompatibleSetSummary = cloneCompatibleSetSummary(
    preferenceTrace?.compatible_set_summary ?? preferenceSummary?.compatible_set_summary ?? null,
  );
  if (!draft.compatible_set_summary) {
    draft.compatible_set_summary = traceCompatibleSetSummary;
  } else if (traceCompatibleSetSummary) {
    draft.compatible_set_summary = {
      ...traceCompatibleSetSummary,
      ...draft.compatible_set_summary,
      route_ids:
        draft.compatible_set_summary.route_ids?.length
          ? cloneRouteIds(draft.compatible_set_summary.route_ids)
          : cloneRouteIds(traceCompatibleSetSummary.route_ids),
      necessary_best_route_ids:
        draft.compatible_set_summary.necessary_best_route_ids?.length
          ? cloneRouteIds(draft.compatible_set_summary.necessary_best_route_ids)
          : cloneRouteIds(traceCompatibleSetSummary.necessary_best_route_ids),
      possible_best_route_ids:
        draft.compatible_set_summary.possible_best_route_ids?.length
          ? cloneRouteIds(draft.compatible_set_summary.possible_best_route_ids)
          : cloneRouteIds(traceCompatibleSetSummary.possible_best_route_ids),
    };
  }
  if (!draft.query_history?.length) {
    draft.query_history = preferenceTrace?.query_history?.map(cloneQuery) ?? [];
  }
  if (!draft.shrinkage_trace?.length) {
    draft.shrinkage_trace = preferenceTrace?.shrinkage_trace ? preferenceTrace.shrinkage_trace.map((item) => ({ ...item })) : [];
  }
  draft.query_count = draft.query_count ?? preferenceTrace?.query_count ?? preferenceSummary?.query_count ?? draft.query_history?.length ?? 0;
  draft.preference_irrelevance_proven = draft.preference_irrelevance_proven ?? preferenceTrace?.preference_irrelevance_proven ?? preferenceSummary?.preference_irrelevance_proven ?? null;
  draft.no_query_reason = draft.no_query_reason ?? preferenceTrace?.no_query_reason ?? preferenceSummary?.no_query_reason ?? null;
  draft.no_preference_query_reason = draft.no_preference_query_reason ?? preferenceTrace?.no_preference_query_reason ?? preferenceSummary?.no_preference_query_reason ?? null;
  draft.derived_invariants = { ...preferenceTrace?.derived_invariants, ...preferenceSummary?.derived_invariants, ...draft.derived_invariants };
  draft.contradiction_record = draft.contradiction_record ?? preferenceTrace?.contradiction_record ?? preferenceSummary?.contradiction_record ?? null;
  draft.terminal_type = draft.terminal_type ?? preferenceTrace?.terminal_type ?? null;
  return draft;
}

export default function PreferenceElicitationPanel({
  locale,
  selectedRoute,
  candidateRoutes,
  preferenceState,
  preferenceQueryTrace,
  preferenceSummary,
  proofContext,
  syncPending = false,
  syncError = null,
  onPreferenceChange,
}: Props) {
  const routes = useMemo(() => dedupeRoutes(selectedRoute, candidateRoutes), [selectedRoute, candidateRoutes]);
  const selectedRouteId = selectedRoute?.id?.trim() || preferenceQueryTrace?.selected_route_id?.trim() || null;
  const initialState = useMemo(
    () => finalize(mergeInitialState(preferenceState, preferenceQueryTrace, preferenceSummary), routes, selectedRouteId),
    [preferenceState, preferenceQueryTrace, preferenceSummary, routes, selectedRouteId],
  );
  const [draftState, setDraftState] = useState<PreferenceState>(initialState);
  const [pairwiseChallengerId, setPairwiseChallengerId] = useState<string>(() => routes.find((route) => route.id !== selectedRouteId)?.id ?? routes[0]?.id ?? '');
  const [focusRouteId, setFocusRouteId] = useState<string>(() => selectedRouteId ?? routes[0]?.id ?? '');
  const [thresholdRouteId, setThresholdRouteId] = useState<string>(() => selectedRouteId ?? routes[0]?.id ?? '');
  const [thresholdMetric, setThresholdMetric] = useState<MetricKey>('duration_s');
  const [thresholdDirection, setThresholdDirection] = useState<'lte' | 'gte'>('lte');
  const [thresholdValue, setThresholdValue] = useState<string>('');
  const [ratioRouteId, setRatioRouteId] = useState<string>(() => selectedRouteId ?? routes[0]?.id ?? '');
  const [ratioNumerator, setRatioNumerator] = useState<MetricKey>('duration_s');
  const [ratioDenominator, setRatioDenominator] = useState<MetricKey>('distance_km');
  const [ratioMinimum, setRatioMinimum] = useState<string>('');
  const [vetoRouteId, setVetoRouteId] = useState<string>(() => selectedRouteId ?? routes[0]?.id ?? '');
  const [vetoName, setVetoName] = useState<string>('user_defined_veto');
  const [guardRouteId, setGuardRouteId] = useState<string>(() => selectedRouteId ?? routes[0]?.id ?? '');
  const [guardSlackMinutes, setGuardSlackMinutes] = useState<string>('10');

  useEffect(() => {
    setDraftState(initialState);
    const defaultRoute = selectedRouteId ?? routes[0]?.id ?? '';
    setPairwiseChallengerId(routes.find((route) => route.id !== selectedRouteId)?.id ?? defaultRoute);
    setFocusRouteId(defaultRoute);
    setThresholdRouteId(defaultRoute);
    setRatioRouteId(defaultRoute);
    setVetoRouteId(defaultRoute);
    setGuardRouteId(defaultRoute);
    const baseThreshold = routeMetric(selectedRoute, thresholdMetric);
    setThresholdValue(baseThreshold ? n(locale, baseThreshold, 2) : '');
    const denominator = routeMetric(selectedRoute, ratioDenominator);
    const numerator = routeMetric(selectedRoute, ratioNumerator);
    setRatioMinimum(denominator ? n(locale, numerator / denominator * 1.05, 3) : '');
    setGuardSlackMinutes('10');
    setVetoName('user_defined_veto');
  }, [initialState, locale, ratioDenominator, ratioNumerator, routes, selectedRoute, selectedRouteId, thresholdMetric]);

  const queryHistory = draftState.query_history ?? [];
  const shrinkageTrace = draftState.shrinkage_trace ?? [];
  const compatibleSetSummary = draftState.compatible_set_summary ?? null;
  const compatibleRouteIds = compatibleSetSummary?.route_ids?.length ? compatibleSetSummary.route_ids : routes.map((route) => route.id);
  const possibleBestRouteIds = compatibleSetSummary?.possible_best_route_ids?.length
    ? compatibleSetSummary.possible_best_route_ids
    : compatibleRouteIds;
  const necessaryBestRouteIds = compatibleSetSummary?.necessary_best_route_ids?.length
    ? compatibleSetSummary.necessary_best_route_ids
    : [];
  const selectedCertificateBasis =
    preferenceSummary?.selected_certificate_basis ??
    preferenceQueryTrace?.selected_certificate_basis ??
    proofContext?.selected_certificate_basis ??
    null;
  const noQueryReason =
    preferenceQueryTrace?.no_preference_query_reason ??
    preferenceQueryTrace?.no_query_reason ??
    draftState.no_preference_query_reason ??
    draftState.no_query_reason ??
    null;
  const latestQuery = queryHistory.length ? queryHistory[queryHistory.length - 1] : null;
  const latestShrinkage = shrinkageTrace.length ? shrinkageTrace[shrinkageTrace.length - 1] : null;
  const whyThisQuery =
    preferenceQueryTrace?.query_selection_reason ??
    latestShrinkage?.query_reason ??
    (latestQuery ? `Latest live query type: ${latestQuery.query_type}` : null) ??
    noQueryReason ??
    null;
  const targetedChallengerId = preferenceQueryTrace?.targeted_challenger_route_id ?? null;
  const pairwiseQueries = queryHistory.filter((query) => query.query_type === 'pairwise');
  const thresholdQueries = queryHistory.filter((query) => query.query_type === 'threshold');
  const ratioQueries = queryHistory.filter((query) => query.query_type === 'ratio');
  const vetoQueries = queryHistory.filter((query) => query.query_type === 'veto');
  const guardQueries = queryHistory.filter((query) => query.query_type === 'time_guard');
  const controlsDisabled = syncPending;
  const snapshots = useMemo(() => routes.map((route) => ({ route, score: scoreRoute(route, routes) })).sort((a, b) => b.score - a.score), [routes]);
  const pairwiseChallenger = routes.find((route) => route.id === pairwiseChallengerId) ?? null;
  const focusRoute = routes.find((route) => route.id === focusRouteId) ?? selectedRoute;
  const thresholdRoute = routes.find((route) => route.id === thresholdRouteId) ?? selectedRoute;
  const ratioRoute = routes.find((route) => route.id === ratioRouteId) ?? selectedRoute;
  const vetoRoute = routes.find((route) => route.id === vetoRouteId) ?? selectedRoute;
  const guardRoute = routes.find((route) => route.id === guardRouteId) ?? selectedRoute;

  function commit(next: PreferenceState, selectedId?: string | null) {
    setDraftState(next);
    onPreferenceChange?.(next, selectedId ?? null);
  }

  function addPairwise(preferred: RouteOption, challenger: RouteOption) {
    commit(
      appendQuery(
        draftState,
        {
          query_type: 'pairwise',
          preferred_route_id: preferred.id,
          challenger_route_id: challenger.id,
          reason: comparison(locale, preferred, challenger),
          weight_hint: {
            duration_s: routeMetric(preferred, 'duration_s') - routeMetric(challenger, 'duration_s'),
            monetary_cost: routeMetric(preferred, 'monetary_cost') - routeMetric(challenger, 'monetary_cost'),
            emissions_kg: routeMetric(preferred, 'emissions_kg') - routeMetric(challenger, 'emissions_kg'),
          },
        },
        routes,
        preferred.id,
      ),
      preferred.id,
    );
  }

  function addThreshold() {
    const route = thresholdRoute;
    const value = Number.parseFloat(thresholdValue);
    if (!route || !Number.isFinite(value)) return;
    const query: ThresholdPreferenceQuery = {
      query_type: 'threshold',
      route_id: route.id,
      metric_name: thresholdMetric,
      threshold_value: value,
      direction: thresholdDirection,
      reason: `Threshold on ${metricLabel[thresholdMetric]} ${thresholdDirection === 'lte' ? '<=' : '>='} ${metricText(locale, thresholdMetric, value)}`,
    };
    commit(appendQuery(draftState, query, routes, route.id), route.id);
  }

  function addRatio() {
    const route = ratioRoute;
    const value = Number.parseFloat(ratioMinimum);
    if (!route || !Number.isFinite(value)) return;
    commit(
      appendQuery(
        draftState,
        {
          query_type: 'ratio',
          route_id: route.id,
          numerator_metric: ratioNumerator,
          denominator_metric: ratioDenominator,
          minimum_ratio: value,
          reason: `Tradeoff ratio ${metricLabel[ratioNumerator]}/${metricLabel[ratioDenominator]} >= ${n(locale, value, 3)}`,
        },
        routes,
        route.id,
      ),
      route.id,
    );
  }

  function addVeto() {
    const route = vetoRoute;
    if (!route || !vetoName.trim()) return;
    const query: VetoPreferenceQuery = {
      query_type: 'veto',
      route_id: route.id,
      veto_name: vetoName.trim(),
      active: true,
      reason: `Veto ${route.id}: ${vetoName.trim()}`,
    };
    commit(appendQuery(draftState, query, routes, route.id), route.id);
  }

  function addTimeGuard() {
    const route = guardRoute;
    const slack = Number.parseFloat(guardSlackMinutes);
    if (!route || !Number.isFinite(slack)) return;
    const query: TimeGuardPreferenceQuery = {
      query_type: 'time_guard',
      route_id: route.id,
      max_travel_time_s: route.metrics.duration_s + slack * 60,
      preserve_time_budget_s: slack * 60,
      reason: `Preserve ${route.id} travel time with ${n(locale, slack, 1)} minute slack`,
    };
    commit(appendQuery(draftState, query, routes, route.id), route.id);
  }

  return (
    <section className="pepRoot" aria-label="Preference elicitation panel">
      <header className="pepHeader">
        <div>
          <div className="pepEyebrow">Live preference elicitation</div>
          <h2>Preference Elicitation Panel</h2>
          <p className="pepIntro">
            This panel reads the live route and runtime payload. It keeps the preference state monotone in the active
            session and syncs committed pairwise, tradeoff, veto, and time-preserving guard answers through the backend
            preference runtime.
          </p>
          {syncPending ? <p className="pepStatus">Syncing the latest preference answer through the backend runtime…</p> : null}
          {syncError ? <p className="pepError">{syncError}</p> : null}
        </div>
        <div className="pepHeaderStats">
          <div><span>Selected</span><strong>{selectedRoute?.id ?? 'n/a'}</strong><em>{routeSummary(locale, selectedRoute)}</em></div>
          <div><span>Queries</span><strong>{n(locale, draftState.query_count ?? queryHistory.length, 0)}</strong><em>{draftState.terminal_type ?? 'open'}</em></div>
          <div><span>Compatible set</span><strong>{n(locale, compatibleSetSummary?.compatible_set_size ?? routes.length, 0)}</strong><em>{pct(locale, compatibleSetSummary?.compatible_set_volume_proxy ?? 0)}</em></div>
        </div>
      </header>

      <div className="pepGrid">
        <section className="pepCard pepWide">
          <h3>Why this query</h3>
          <p>{whyThisQuery ?? 'No live query reason has been emitted yet.'}</p>
          <div className="pepMetaLine">
            <span>Targeted challenger: {targetedChallengerId ?? 'n/a'}</span>
            <span>No-query reason: {noQueryReason ?? 'n/a'}</span>
            <span>Proof basis: {selectedCertificateBasis ?? 'n/a'}</span>
          </div>
        </section>

        <section className="pepCard">
          <h3>Live route table</h3>
          <div className="pepRouteTable">
            {snapshots.map(({ route, score }) => (
              <button
                type="button"
                key={route.id}
                className={`pepRouteRow ${route.id === selectedRouteId ? 'isSelected' : ''}`}
                disabled={controlsDisabled}
                onClick={() => {
                  setFocusRouteId(route.id);
                  setThresholdRouteId(route.id);
                  setRatioRouteId(route.id);
                  setVetoRouteId(route.id);
                  setGuardRouteId(route.id);
                }}
              >
                <strong>{route.id}</strong>
                <span>{routeSummary(locale, route)}</span>
                <em>score {n(locale, score, 3)}</em>
              </button>
            ))}
          </div>
        </section>

        <section className="pepCard">
          <h3>Pairwise question</h3>
          <label className="pepField">
            <span>Challenger route</span>
            <select value={pairwiseChallengerId} onChange={(event) => setPairwiseChallengerId(event.target.value)}>
              {routes
                .filter((route) => route.id !== selectedRouteId)
                .map((route) => (
                  <option key={route.id} value={route.id}>
                    {route.id}
                  </option>
                ))}
            </select>
          </label>
          <p className="pepPrompt">
            Would you keep <strong>{selectedRouteId ?? 'the selected route'}</strong> over{' '}
            <strong>{pairwiseChallenger?.id ?? 'the challenger'}</strong>?
          </p>
          <p className="pepSubtle">{comparison(locale, selectedRoute, pairwiseChallenger)}</p>
          <div className="pepButtons">
            <button
              type="button"
              disabled={controlsDisabled}
              onClick={() => selectedRoute && pairwiseChallenger && addPairwise(selectedRoute, pairwiseChallenger)}
            >
              Prefer selected
            </button>
            <button
              type="button"
              disabled={controlsDisabled}
              onClick={() => selectedRoute && pairwiseChallenger && addPairwise(pairwiseChallenger, selectedRoute)}
            >
              Prefer challenger
            </button>
          </div>
        </section>

        <section className="pepCard">
          <h3>Tradeoff question</h3>
          <div className="pepSubgrid">
            <label className="pepField">
              <span>Route</span>
              <select value={thresholdRouteId} onChange={(event) => setThresholdRouteId(event.target.value)}>
                {routes.map((route) => <option key={route.id} value={route.id}>{route.id}</option>)}
              </select>
            </label>
            <label className="pepField">
              <span>Metric</span>
              <select value={thresholdMetric} onChange={(event) => setThresholdMetric(event.target.value as MetricKey)}>
                {metricOrder.map((metric) => <option key={metric} value={metric}>{metricLabel[metric]}</option>)}
              </select>
            </label>
            <label className="pepField">
              <span>Direction</span>
              <select value={thresholdDirection} onChange={(event) => setThresholdDirection(event.target.value as 'lte' | 'gte')}>
                <option value="lte">At or below</option>
                <option value="gte">At or above</option>
              </select>
            </label>
            <label className="pepField">
              <span>Threshold ({metricLabel[thresholdMetric]})</span>
              <input value={thresholdValue} onChange={(event) => setThresholdValue(event.target.value)} inputMode="decimal" />
            </label>
          </div>
          <button type="button" disabled={controlsDisabled} onClick={addThreshold}>Add threshold rule</button>
          <div className="pepDivider" />
          <div className="pepSubgrid">
            <label className="pepField">
              <span>Route</span>
              <select value={ratioRouteId} onChange={(event) => setRatioRouteId(event.target.value)}>
                {routes.map((route) => <option key={route.id} value={route.id}>{route.id}</option>)}
              </select>
            </label>
            <label className="pepField">
              <span>Numerator</span>
              <select value={ratioNumerator} onChange={(event) => setRatioNumerator(event.target.value as MetricKey)}>
                {metricOrder.map((metric) => <option key={metric} value={metric}>{metricLabel[metric]}</option>)}
              </select>
            </label>
            <label className="pepField">
              <span>Denominator</span>
              <select value={ratioDenominator} onChange={(event) => setRatioDenominator(event.target.value as MetricKey)}>
                {metricOrder.filter((metric) => metric !== ratioNumerator).map((metric) => <option key={metric} value={metric}>{metricLabel[metric]}</option>)}
              </select>
            </label>
            <label className="pepField">
              <span>Minimum ratio</span>
              <input value={ratioMinimum} onChange={(event) => setRatioMinimum(event.target.value)} inputMode="decimal" />
            </label>
          </div>
          <button type="button" disabled={controlsDisabled} onClick={addRatio}>Add ratio rule</button>
        </section>

        <section className="pepCard">
          <h3>Veto setting</h3>
          <div className="pepSubgrid">
            <label className="pepField">
              <span>Route</span>
              <select value={vetoRouteId} onChange={(event) => setVetoRouteId(event.target.value)}>
                {routes.map((route) => <option key={route.id} value={route.id}>{route.id}</option>)}
              </select>
            </label>
            <label className="pepField">
              <span>Veto name</span>
              <input value={vetoName} onChange={(event) => setVetoName(event.target.value)} />
            </label>
          </div>
          <button type="button" disabled={controlsDisabled} onClick={addVeto}>Add veto rule</button>
        </section>

        <section className="pepCard">
          <h3>Time-preserving guard</h3>
          <div className="pepSubgrid">
            <label className="pepField">
              <span>Route</span>
              <select value={guardRouteId} onChange={(event) => setGuardRouteId(event.target.value)}>
                {routes.map((route) => <option key={route.id} value={route.id}>{route.id}</option>)}
              </select>
            </label>
            <label className="pepField">
              <span>Slack minutes</span>
              <input value={guardSlackMinutes} onChange={(event) => setGuardSlackMinutes(event.target.value)} inputMode="decimal" />
            </label>
          </div>
          <button type="button" disabled={controlsDisabled} onClick={addTimeGuard}>Add time-preserving guard</button>
        </section>

        <section className="pepCard pepWide">
          <h3>Compatible-set region summary</h3>
          <div className="pepStats">
            <div><span>Size</span><strong>{n(locale, compatibleSetSummary?.compatible_set_size ?? routes.length, 0)}</strong></div>
            <div><span>Volume proxy</span><strong>{pct(locale, compatibleSetSummary?.compatible_set_volume_proxy ?? 0)}</strong></div>
            <div><span>Necessary best</span><strong>{pct(locale, compatibleSetSummary?.necessary_best_prob ?? 0)}</strong></div>
            <div><span>Possible best</span><strong>{pct(locale, compatibleSetSummary?.possible_best_prob ?? 0)}</strong></div>
          </div>
          <p>{compatibleSetSummary?.support_reason ?? 'No compatible-set summary yet.'}</p>
          <div className="pepField">
            <span>Compatible routes</span>
            <div className="pepChips">
              {compatibleRouteIds.map((routeId) => (
                <span key={routeId}>{routeId}</span>
              ))}
            </div>
          </div>
          <div className="pepSubgrid">
            <div className="pepField">
              <span>Possible best routes</span>
              <div className="pepChips">
                {possibleBestRouteIds.map((routeId) => (
                  <span key={routeId}>{routeId}</span>
                ))}
              </div>
            </div>
            <div className="pepField">
              <span>Necessary best routes</span>
              <div className="pepChips">
                {(necessaryBestRouteIds.length ? necessaryBestRouteIds : ['None pinned yet']).map((routeId) => (
                  <span key={routeId}>{routeId}</span>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="pepCard">
          <h3>Shrinkage over time</h3>
          {shrinkageTrace.length ? (
            <ol className="pepTimeline">
              {shrinkageTrace.slice(-6).map((entry) => (
                <li key={`${entry.query_index}-${entry.query_type}`}>
                  <strong>Q{entry.query_index}</strong> {entry.query_type} | set {n(locale, entry.before_size, 0)} →{' '}
                  {n(locale, entry.after_size, 0)} | volume {pct(locale, entry.before_volume_proxy ?? 0)} →{' '}
                  {pct(locale, entry.after_volume_proxy ?? 0)}
                  <div>
                    predicted {pct(locale, entry.predicted_shrinkage ?? 0)} | realized {pct(locale, entry.realized_shrinkage ?? 0)}
                  </div>
                </li>
              ))}
            </ol>
          ) : (
            <p>No shrinkage trace is available yet.</p>
          )}
        </section>

        <section className="pepCard">
          <h3>Current live state</h3>
          <div className="pepStats">
            <div><span>Pairwise</span><strong>{n(locale, pairwiseQueries.length, 0)}</strong></div>
            <div><span>Tradeoff</span><strong>{n(locale, thresholdQueries.length + ratioQueries.length, 0)}</strong></div>
            <div><span>Veto</span><strong>{n(locale, vetoQueries.length, 0)}</strong></div>
            <div><span>Guard</span><strong>{n(locale, guardQueries.length, 0)}</strong></div>
          </div>
          <p>{latestQuery ? `Latest query: ${latestQuery.query_type}` : 'No live query has been issued yet.'}</p>
          <div className="pepChips">
            {draftState.derived_invariants ? Object.entries(draftState.derived_invariants).filter(([, active]) => active).map(([name]) => (
              <span key={name}>{name.replace(/_/g, ' ')}</span>
            )) : null}
          </div>
        </section>

        <section className="pepCard pepWide">
          <h3>Current route focus</h3>
          <p>Focus route: {focusRoute?.id ?? 'n/a'}</p>
          <p>Selected route metrics: {routeSummary(locale, selectedRoute)}</p>
          <p>Preference summary query count: {n(locale, preferenceSummary?.query_count ?? preferenceQueryTrace?.query_count ?? 0, 0)}</p>
          <p>Proof context: {selectedCertificateBasis ?? 'n/a'}{proofContext?.support_flag !== undefined ? ` · support ${proofContext.support_flag ? 'yes' : 'no'}` : ''}</p>
        </section>
      </div>

      <style jsx>{`
        .pepRoot {
          border: 1px solid rgba(121, 138, 160, 0.24);
          border-radius: 20px;
          background: linear-gradient(180deg, rgba(11, 18, 30, 0.98), rgba(8, 13, 20, 0.98));
          color: #eff6ff;
          padding: 20px;
        }
        .pepHeader {
          display: grid;
          grid-template-columns: minmax(0, 1.6fr) minmax(300px, 1fr);
          gap: 16px;
          margin-bottom: 16px;
        }
        .pepEyebrow {
          margin: 0 0 8px;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: #8fe7ff;
          font-size: 11px;
        }
        h2, h3, p {
          margin-top: 0;
        }
        h2 {
          margin-bottom: 10px;
          font-size: clamp(1.6rem, 3vw, 2.2rem);
        }
        .pepIntro {
          color: rgba(225, 236, 250, 0.84);
          line-height: 1.55;
          margin-bottom: 0;
        }
        .pepStatus,
        .pepError {
          margin-top: 10px;
          margin-bottom: 0;
          font-size: 13px;
        }
        .pepStatus {
          color: #8fe7ff;
        }
        .pepError {
          color: #ffb4b4;
        }
        .pepHeaderStats {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
        }
        .pepHeaderStats > div, .pepCard {
          border: 1px solid rgba(137, 155, 182, 0.2);
          background: rgba(15, 24, 36, 0.8);
          border-radius: 16px;
          padding: 14px;
        }
        .pepHeaderStats span, .pepStats span {
          display: block;
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: rgba(164, 183, 206, 0.82);
        }
        .pepHeaderStats strong, .pepStats strong {
          display: block;
          margin: 4px 0;
          font-size: 1.05rem;
        }
        .pepHeaderStats em {
          font-style: normal;
          color: rgba(203, 216, 233, 0.78);
          font-size: 13px;
        }
        .pepGrid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
        }
        .pepWide {
          grid-column: 1 / -1;
        }
        .pepField {
          display: grid;
          gap: 6px;
          margin-bottom: 10px;
        }
        .pepField span {
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: rgba(164, 183, 206, 0.82);
        }
        .pepField select, .pepField input, button {
          border-radius: 12px;
          border: 1px solid rgba(145, 164, 191, 0.28);
          background: rgba(7, 13, 20, 0.85);
          color: #eff6ff;
          padding: 10px 12px;
          font: inherit;
        }
        button {
          cursor: pointer;
          background: linear-gradient(180deg, rgba(58, 163, 124, 0.95), rgba(38, 128, 97, 0.95));
          color: #04130d;
          font-weight: 700;
          margin-right: 8px;
          margin-top: 4px;
        }
        .pepPrompt, .pepSubtle, .pepCard p {
          line-height: 1.5;
          color: rgba(225, 236, 250, 0.86);
        }
        .pepSubgrid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
        }
        .pepRouteTable {
          display: grid;
          gap: 8px;
        }
        .pepRouteRow {
          text-align: left;
          display: grid;
          gap: 3px;
          border-radius: 14px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(145, 164, 191, 0.18);
          color: inherit;
        }
        .pepRouteRow.isSelected {
          border-color: rgba(105, 225, 160, 0.6);
          background: rgba(76, 209, 132, 0.12);
        }
        .pepMetaLine, .pepStats, .pepChips {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
        }
        .pepChips span {
          border-radius: 999px;
          padding: 5px 9px;
          background: rgba(98, 216, 164, 0.12);
          border: 1px solid rgba(98, 216, 164, 0.22);
        }
        .pepDivider {
          height: 1px;
          background: rgba(145, 164, 191, 0.18);
          margin: 12px 0;
        }
        .pepTimeline {
          margin: 0;
          padding-left: 18px;
          display: grid;
          gap: 8px;
        }
        .pepTimeline li {
          color: rgba(225, 236, 250, 0.86);
        }
        @media (max-width: 900px) {
          .pepHeader, .pepHeaderStats, .pepGrid, .pepSubgrid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </section>
  );
}
