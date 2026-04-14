'use client';

import { useEffect, useMemo, useState, type CSSProperties } from 'react';

import FieldInfo from './FieldInfo';
import { formatMetricTooltip, type MetricTooltip } from './metricTooltip';
import type {
  ActionTraceSummary,
  DecisionRegionSummaryArtifact,
  FlipRadiusSummaryArtifact,
  PipelineMode,
  RouteCertificationSummary,
  RouteFragilityMapArtifact,
  RouteOption,
  SampledWorldManifestArtifact,
  ValueOfRefreshArtifact,
  VoiActionTraceArtifact,
  VoiStopCertificateArtifact,
  VoiStopSummary,
  VoiTraceAction,
  WitnessSummary,
  WorldSupportSummary,
} from '../lib/types';

type Props = {
  locale: string;
  route: RouteOption | null;
  runId?: string | null;
  pipelineMode?: PipelineMode;
  terminalType?: string | null;
  selectedCertificateBasis?: string | null;
  selectedCertificate?: RouteCertificationSummary | null;
  certificateSummary?: RouteCertificationSummary | Record<string, unknown> | null;
  voiStopSummary?: VoiStopSummary | null;
  actionTraceSummary?: ActionTraceSummary | null;
  witnessSummary?: WitnessSummary | null;
  worldSupportSummary?: WorldSupportSummary | null;
  supportSummary?: Record<string, unknown> | null;
  artifactPointers?: Record<string, string | null> | null;
  routeArtifactsEndpoint?: string | null;
  onOpenRunInspector?: (runId: string) => void;
};

type ArtifactState = {
  sampledWorldManifest: SampledWorldManifestArtifact | null;
  routeFragilityMap: RouteFragilityMapArtifact | null;
  flipRadiusSummary: FlipRadiusSummaryArtifact | null;
  decisionRegionSummary: DecisionRegionSummaryArtifact | null;
  valueOfRefresh: ValueOfRefreshArtifact | null;
  voiActionTrace: VoiActionTraceArtifact | null;
  voiStopCertificate: VoiStopCertificateArtifact | null;
  loading: boolean;
  error: string | null;
};

const inlineMetricLabelStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
};

function pct(locale: string, value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a';
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    maximumFractionDigits: 1,
    minimumFractionDigits: 0,
  }).format(value);
}

function n(locale: string, value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a';
  return new Intl.NumberFormat(locale, { maximumFractionDigits: 2 }).format(value);
}

function text(value: unknown): string {
  if (typeof value !== 'string') return 'n/a';
  const trimmed = value.trim();
  return trimmed || 'n/a';
}

function textOrNull(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed || null;
}

function recordValue(record: Record<string, unknown> | null | undefined, key: string): unknown {
  if (!record || !(key in record)) return undefined;
  return record[key];
}

function stringList(record: Record<string, unknown> | null | undefined, key: string): string[] {
  const value = recordValue(record, key);
  if (!Array.isArray(value)) return [];
  return value
    .map((entry) => (typeof entry === 'string' ? entry.trim() : ''))
    .filter(Boolean);
}

function numberOrNull(value: unknown): number | null | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : value === null ? null : undefined;
}

function average(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function csvCell(value: string): string {
  return /[",\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

function downloadTextFile(contents: string, fileName: string, mimeType: string): void {
  const blob = new Blob([contents], { type: mimeType });
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = href;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(href);
}

function escapeXml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function buildSvgFigure(title: string, subtitle: string, lines: string[]): string {
  const width = 960;
  const lineHeight = 24;
  const topOffset = 98;
  const height = Math.max(220, topOffset + lines.length * lineHeight + 36);
  const textLines = lines
    .map(
      (line, index) =>
        `<text x="40" y="${topOffset + index * lineHeight}" font-family="'Segoe UI', Arial, sans-serif" font-size="16" fill="#132238">${escapeXml(line)}</text>`,
    )
    .join('');
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">`,
    '<rect width="100%" height="100%" fill="#f8fafc" />',
    '<rect x="20" y="20" width="920" height="' + (height - 40) + '" rx="18" fill="#ffffff" stroke="#d6deea" stroke-width="2" />',
    `<text x="40" y="58" font-family="'Segoe UI', Arial, sans-serif" font-size="28" font-weight="700" fill="#132238">${escapeXml(title)}</text>`,
    `<text x="40" y="82" font-family="'Segoe UI', Arial, sans-serif" font-size="14" fill="#516074">${escapeXml(subtitle)}</text>`,
    textLines,
    '</svg>',
  ].join('');
}

function buildCsvDocument(headers: string[], rows: string[][]): string {
  return [headers.map(csvCell).join(','), ...rows.map((row) => row.map(csvCell).join(','))].join('\n');
}

function terminalLabel(terminalType: string | null | undefined): string {
  switch (terminalType) {
    case 'certified_singleton':
      return 'Certified singleton';
    case 'certified_set':
      return 'Certified set';
    case 'typed_abstention':
      return 'Typed abstention';
    default:
      return 'Decision payload';
  }
}

function statusLabel(flag: boolean | null | undefined): string {
  if (flag === true) return 'In support';
  if (flag === false) return 'Out of support';
  return 'Support unknown';
}

function artifactHref(
  routeArtifactsEndpoint: string | null | undefined,
  artifactPointer: string | null | undefined,
  fallbackName?: string,
  options?: { allowBaseResolution?: boolean },
): string | null {
  const allowBaseResolution = options?.allowBaseResolution ?? true;
  const raw = textOrNull(artifactPointer) ?? fallbackName ?? null;
  if (!raw) return null;
  if (/^(https?:)?\/\//i.test(raw) || raw.startsWith('/')) return raw;
  if (!allowBaseResolution || !routeArtifactsEndpoint) return null;
  const base = routeArtifactsEndpoint.replace(/\/$/, '');
  return `${base}/${raw}`;
}

async function fetchOptionalJson<T>(href: string, signal: AbortSignal): Promise<T | null> {
  const response = await fetch(href, { cache: 'no-store', signal });
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(`Failed to load ${href} (${response.status})`);
  }
  return (await response.json()) as T;
}

function actionFamilyForKind(kind: string | null | undefined): string {
  switch (kind) {
    case 'refine_top1_dccs':
    case 'refine_topk_dccs':
      return 'search';
    case 'refresh_top1_vor':
    case 'increase_stochastic_samples':
      return 'evidence';
    case 'stop':
      return 'terminal';
    default:
      return kind ? 'mixed' : 'n/a';
  }
}

function actionModalityForKind(kind: string | null | undefined): string {
  switch (kind) {
    case 'refine_top1_dccs':
      return 'refine_top1';
    case 'refine_topk_dccs':
      return 'refine_topk';
    case 'refresh_top1_vor':
      return 'refresh';
    case 'increase_stochastic_samples':
      return 'resample';
    case 'stop':
      return 'stop';
    default:
      return kind ?? 'n/a';
  }
}

function resolveActionFamily(action: VoiTraceAction | null | undefined): string {
  return text(action?.action_family) !== 'n/a'
    ? text(action?.action_family)
    : actionFamilyForKind(textOrNull(action?.kind));
}

function resolveActionModality(action: VoiTraceAction | null | undefined): string {
  return text(action?.action_modality) !== 'n/a'
    ? text(action?.action_modality)
    : actionModalityForKind(textOrNull(action?.kind));
}

function metricHelp(tooltip: MetricTooltip): string {
  return formatMetricTooltip(tooltip);
}

function metricLabel(label: string, tooltip: MetricTooltip) {
  return (
    <>
      {label}
      <FieldInfo text={metricHelp(tooltip)} />
    </>
  );
}

function inlineMetricLabel(label: string, tooltip: MetricTooltip) {
  return (
    <span style={inlineMetricLabelStyle}>
      <span>{label}</span>
      <FieldInfo text={metricHelp(tooltip)} />
    </span>
  );
}

export default function RouteCertificationPanel({
  locale,
  route,
  runId,
  pipelineMode = 'legacy',
  terminalType,
  selectedCertificateBasis,
  selectedCertificate,
  certificateSummary,
  voiStopSummary,
  actionTraceSummary,
  witnessSummary,
  worldSupportSummary,
  supportSummary,
  artifactPointers,
  routeArtifactsEndpoint,
  onOpenRunInspector,
}: Props) {
  const certification = selectedCertificate ?? route?.certification ?? null;
  const certificateSummaryRecord =
    certificateSummary && typeof certificateSummary === 'object'
      ? (certificateSummary as Record<string, unknown>)
      : null;
  const visible =
    Boolean(route) ||
    Boolean(certification) ||
    Boolean(certificateSummaryRecord) ||
    Boolean(worldSupportSummary) ||
    Boolean(supportSummary) ||
    Boolean(actionTraceSummary) ||
    Boolean(witnessSummary) ||
    Boolean(voiStopSummary) ||
    Boolean(runId) ||
    Boolean(routeArtifactsEndpoint);
  if (!visible) return null;

  const supportState = worldSupportSummary?.support_state ?? worldSupportSummary?.world_bundle_summary?.support_state ?? null;
  const activeFamilies =
    certification?.active_families ??
    route?.evidence_provenance?.active_families ??
    worldSupportSummary?.active_families ??
    stringList(supportSummary, 'active_families');
  const supportFlag =
    worldSupportSummary?.support_flag ??
    supportState?.support_flag ??
    (typeof recordValue(supportSummary, 'support_flag') === 'boolean'
      ? (recordValue(supportSummary, 'support_flag') as boolean)
      : undefined);
  const supportReason =
    textOrNull(worldSupportSummary?.support_reason) ??
    textOrNull(supportState?.support_reason) ??
    textOrNull(recordValue(supportSummary, 'support_reason'));
  const outOfSupportReason =
    textOrNull(supportState?.out_of_support_reason) ??
    textOrNull(recordValue(supportSummary, 'out_of_support_reason'));
  const supportStatus =
    textOrNull(supportState?.support_status) ?? textOrNull(recordValue(supportSummary, 'support_status'));
  const supportCalibrationBin =
    textOrNull(worldSupportSummary?.calibration_bin) ??
    textOrNull(supportState?.calibration_bin) ??
    textOrNull(recordValue(supportSummary, 'calibration_bin')) ??
    textOrNull(worldSupportSummary?.support_bin) ??
    textOrNull(supportState?.support_bin) ??
    textOrNull(recordValue(supportSummary, 'support_bin')) ??
    'n/a';
  const scenarioSummary = worldSupportSummary?.scenario_summary ?? route?.scenario_summary ?? null;
  const scenarioProvenanceBits = [
    textOrNull(scenarioSummary?.mode),
    textOrNull(scenarioSummary?.context_key),
    textOrNull(scenarioSummary?.source),
    textOrNull(scenarioSummary?.version),
  ].filter(Boolean) as string[];
  const artifactsBase = routeArtifactsEndpoint?.replace(/\/$/, '') ?? null;
  const manifestHref = artifactHref(
    artifactsBase,
    artifactPointers?.sampled_world_manifest ?? null,
    'sampled_world_manifest.json',
  );
  const fragilityHref = artifactHref(
    artifactsBase,
    artifactPointers?.route_fragility_map ?? null,
    'route_fragility_map.json',
  );
  const flipRadiusHref = artifactHref(
    artifactsBase,
    artifactPointers?.flip_radius_summary ?? null,
    'flip_radius_summary.json',
  );
  const decisionRegionHref = artifactHref(
    artifactsBase,
    artifactPointers?.decision_region_summary ?? null,
    'decision_region_summary.json',
  );
  const decisionPackageHref = artifactHref(
    artifactsBase,
    artifactPointers?.decision_package ?? null,
  );
  const certificateSummaryHref = artifactHref(
    artifactsBase,
    artifactPointers?.certificate_summary ?? null,
  );
  const certificateWitnessHref = artifactHref(
    artifactsBase,
    artifactPointers?.certificate_witness ?? null,
  );
  const valueOfRefreshHref = artifactHref(
    artifactsBase,
    artifactPointers?.value_of_refresh ?? null,
    'value_of_refresh.json',
  );
  const voiActionTraceHref = artifactHref(
    artifactsBase,
    artifactPointers?.voi_action_trace ?? null,
    'voi_action_trace.json',
  );
  const voiActionScoresHref = artifactHref(
    artifactsBase,
    artifactPointers?.voi_action_scores ?? null,
    'voi_action_scores.csv',
  );
  const voiStopHref = artifactHref(
    artifactsBase,
    artifactPointers?.voi_stop_certificate ?? null,
    'voi_stop_certificate.json',
  );
  const voiControllerStateHref = artifactHref(
    artifactsBase,
    artifactPointers?.voi_controller_state ?? null,
    'voi_controller_state.jsonl',
  );
  const worldSupportHref = artifactHref(
    artifactsBase,
    artifactPointers?.world_support_summary ?? null,
    undefined,
    { allowBaseResolution: false },
  );
  const [artifactState, setArtifactState] = useState<ArtifactState>({
    sampledWorldManifest: null,
    routeFragilityMap: null,
    flipRadiusSummary: null,
    decisionRegionSummary: null,
    valueOfRefresh: null,
    voiActionTrace: null,
    voiStopCertificate: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    const targets = [
      { key: 'sampledWorldManifest' as const, href: manifestHref, label: 'sampled_world_manifest.json' },
      { key: 'routeFragilityMap' as const, href: fragilityHref, label: 'route_fragility_map.json' },
      { key: 'flipRadiusSummary' as const, href: flipRadiusHref, label: 'flip_radius_summary.json' },
      { key: 'decisionRegionSummary' as const, href: decisionRegionHref, label: 'decision_region_summary.json' },
      { key: 'valueOfRefresh' as const, href: valueOfRefreshHref, label: 'value_of_refresh.json' },
      { key: 'voiActionTrace' as const, href: voiActionTraceHref, label: 'voi_action_trace.json' },
      { key: 'voiStopCertificate' as const, href: voiStopHref, label: 'voi_stop_certificate.json' },
    ].filter((target) => Boolean(target.href));
    if (!targets.length) {
      setArtifactState({
        sampledWorldManifest: null,
        routeFragilityMap: null,
        flipRadiusSummary: null,
        decisionRegionSummary: null,
        valueOfRefresh: null,
        voiActionTrace: null,
        voiStopCertificate: null,
        loading: false,
        error: null,
      });
      return;
    }
    const controller = new AbortController();
    setArtifactState((current) => ({ ...current, loading: true, error: null }));
    void Promise.allSettled(
      targets.map(async (target) => ({
        key: target.key,
        label: target.label,
        data: await fetchOptionalJson(target.href!, controller.signal),
      })),
    ).then((results) => {
      if (controller.signal.aborted) return;
      const next: ArtifactState = {
        sampledWorldManifest: null,
        routeFragilityMap: null,
        flipRadiusSummary: null,
        decisionRegionSummary: null,
        valueOfRefresh: null,
        voiActionTrace: null,
        voiStopCertificate: null,
        loading: false,
        error: null,
      };
      const failures: string[] = [];
      results.forEach((result, index) => {
        const label = targets[index]?.label ?? 'artifact';
        if (result.status === 'fulfilled') {
          if (result.value.key === 'sampledWorldManifest') {
            next.sampledWorldManifest = result.value.data as SampledWorldManifestArtifact | null;
          } else if (result.value.key === 'routeFragilityMap') {
            next.routeFragilityMap = result.value.data as RouteFragilityMapArtifact | null;
          } else if (result.value.key === 'flipRadiusSummary') {
            next.flipRadiusSummary = result.value.data as FlipRadiusSummaryArtifact | null;
          } else if (result.value.key === 'decisionRegionSummary') {
            next.decisionRegionSummary = result.value.data as DecisionRegionSummaryArtifact | null;
          } else if (result.value.key === 'valueOfRefresh') {
            next.valueOfRefresh = result.value.data as ValueOfRefreshArtifact | null;
          } else if (result.value.key === 'voiActionTrace') {
            next.voiActionTrace = result.value.data as VoiActionTraceArtifact | null;
          } else if (result.value.key === 'voiStopCertificate') {
            next.voiStopCertificate = result.value.data as VoiStopCertificateArtifact | null;
          }
          return;
        }
        failures.push(label);
      });
      if (failures.length) {
        next.error = `Some proof artifacts could not be loaded: ${failures.join(', ')}`;
      }
      setArtifactState(next);
    });
    return () => controller.abort();
  }, [decisionRegionHref, flipRadiusHref, fragilityHref, manifestHref, valueOfRefreshHref, voiActionTraceHref, voiStopHref]);

  const selectedRouteId = useMemo(
    () =>
      route?.id ??
      certification?.route_id ??
      witnessSummary?.route_id ??
      worldSupportSummary?.selected_route_id ??
      artifactState.sampledWorldManifest?.selected_route_id ??
      artifactState.valueOfRefresh?.selected_route_id ??
      null,
    [
      artifactState.sampledWorldManifest?.selected_route_id,
      artifactState.valueOfRefresh?.selected_route_id,
      certification?.route_id,
      route?.id,
      witnessSummary?.route_id,
      worldSupportSummary?.selected_route_id,
    ],
  );
  const probabilisticBundle = worldSupportSummary?.world_bundle_summary?.probabilistic_world_bundle ?? null;
  const auditBundle = worldSupportSummary?.world_bundle_summary?.audit_world_bundle ?? null;
  const probabilisticWorldCount =
    artifactState.sampledWorldManifest?.world_count ??
    probabilisticBundle?.world_count ??
    worldSupportSummary?.world_count ??
    numberOrNull(recordValue(supportSummary, 'world_count'));
  const uniqueWorldCount =
    artifactState.sampledWorldManifest?.unique_world_count ??
    probabilisticBundle?.unique_world_count ??
    worldSupportSummary?.unique_world_count ??
    numberOrNull(recordValue(supportSummary, 'unique_world_count'));
  const auditWorldCount = auditBundle?.audit_world_count ?? null;
  const auditPairCount = auditBundle?.audited_route_pair_count ?? null;
  const auditFullCount = auditBundle?.fully_audited_world_count ?? null;
  const auditPartialCount = auditBundle?.partially_audited_world_count ?? null;
  const auditReuseCount = auditBundle?.reused_world_count ?? null;
  const worldReuseRate =
    artifactState.sampledWorldManifest?.world_reuse_rate ??
    probabilisticBundle?.world_reuse_rate ??
    worldSupportSummary?.world_reuse_rate ??
    numberOrNull(recordValue(supportSummary, 'world_reuse_rate'));
  const stateWeights = artifactState.sampledWorldManifest?.state_weights ?? probabilisticBundle?.state_weights ?? null;
  const proxyShares = stateWeights
    ? Object.values(stateWeights)
        .map((weights) => numberOrNull(weights.proxy))
        .filter((value): value is number => value !== null && value !== undefined)
    : [];
  const proxyShareAverage = average(proxyShares);
  const proxyHeavyFamily = stateWeights
    ? Object.entries(stateWeights)
        .map(([family, weights]) => [family, numberOrNull(weights.proxy) ?? 0] as const)
        .sort((left, right) => right[1] - left[1])[0]
    : null;
  const worldKinds = artifactState.sampledWorldManifest?.worlds?.length
    ? Array.from(
        new Set(
          artifactState.sampledWorldManifest.worlds
            .map((world) => textOrNull(world.world_kind))
            .filter(Boolean) as string[],
        ),
      )
    : [];
  const routeFragility = selectedRouteId
    ? artifactState.routeFragilityMap?.[selectedRouteId] ?? null
    : artifactState.routeFragilityMap
      ? artifactState.routeFragilityMap[Object.keys(artifactState.routeFragilityMap)[0] ?? ''] ?? null
      : null;
  const fragilityEntries = routeFragility
    ? Object.entries(routeFragility).sort((left, right) => right[1] - left[1])
    : [];
  const dominantFragilityFamily =
    certification?.top_fragility_families?.[0] ??
    fragilityEntries[0]?.[0] ??
    textOrNull(artifactState.valueOfRefresh?.fragility_stress_state);
  const chosenRefreshFamily = certification?.top_value_of_refresh_family ?? null;
  const lowerConfidenceBound =
    certification?.certificate_lcb ??
    numberOrNull(recordValue(certificateSummaryRecord, 'certificate_lcb')) ??
    null;
  const nearestChallenger =
    textOrNull(artifactState.decisionRegionSummary?.active_challenger_id) ??
    witnessSummary?.active_challenger_ids?.[0] ??
    certification?.top_competitor_route_id ??
    textOrNull(recordValue(certificateSummaryRecord, 'top_competitor_route_id')) ??
    null;
  const minimumPairwiseGapLcb =
    certification?.minimum_pairwise_gap_lcb ??
    numberOrNull(recordValue(certificateSummaryRecord, 'minimum_pairwise_gap_lcb')) ??
    numberOrNull(recordValue(artifactState.decisionRegionSummary?.provenance ?? null, 'minimum_pairwise_gap_lcb')) ??
    null;
  const minimumFlipBudget =
    numberOrNull(artifactState.flipRadiusSummary?.minimum_flip_budget) ??
    numberOrNull(recordValue(artifactState.decisionRegionSummary?.provenance ?? null, 'minimum_flip_budget')) ??
    null;
  const empiricalRefreshFamily = artifactState.valueOfRefresh?.top_refresh_family ?? null;
  const controllerRefreshFamily = artifactState.valueOfRefresh?.top_refresh_family_controller ?? null;
  const controllerRankingBasis = artifactState.valueOfRefresh?.controller_ranking_basis ?? null;
  const baselineCertificate =
    artifactState.valueOfRefresh?.baseline_certificate ??
    artifactState.valueOfRefresh?.empirical_baseline_certificate ??
    artifactState.valueOfRefresh?.controller_baseline_certificate ??
    certification?.certificate ??
    null;
  const resolvedStopReason =
    artifactState.voiStopCertificate?.stop_reason ??
    voiStopSummary?.stop_reason ??
    actionTraceSummary?.stop_reason ??
    null;
  const resolvedSearchCompleteness =
    artifactState.voiStopCertificate?.search_completeness_score ??
    actionTraceSummary?.search_completeness_score ??
    null;
  const resolvedSearchGap =
    artifactState.voiStopCertificate?.search_completeness_gap ??
    actionTraceSummary?.search_completeness_gap ??
    null;
  const resolvedIterationCount =
    artifactState.voiStopCertificate?.iteration_count ??
    artifactState.voiActionTrace?.actions?.length ??
    voiStopSummary?.iteration_count ??
    null;
  const resolvedSearchBudgetUsed =
    artifactState.voiStopCertificate?.search_budget_used ??
    voiStopSummary?.search_budget_used ??
    null;
  const resolvedEvidenceBudgetUsed =
    artifactState.voiStopCertificate?.evidence_budget_used ??
    voiStopSummary?.evidence_budget_used ??
    null;
  const controllerState = artifactState.voiStopCertificate?.controller_state ?? null;
  const controllerBoundarySummary = controllerState?.active_certificate_boundary_summary ?? null;
  const controllerAuditSummary = controllerState?.audit_propensity_summary ?? null;
  const controllerCertificateLcb = controllerState?.certificate_lcb ?? lowerConfidenceBound;
  const controllerCertificateUcb = controllerState?.certificate_ucb ?? null;
  const controllerNecessaryBest = controllerState?.necessary_best_probability ?? null;
  const controllerPossibleBest = controllerState?.possible_best_probability ?? null;
  const controllerDeterministicFlipRadius = controllerState?.deterministic_local_flip_radius ?? null;
  const controllerProbabilisticFlipRadius = controllerState?.probabilistic_flip_radius ?? null;
  const controllerMinimumFlipBudget = controllerState?.minimum_flip_budget ?? minimumFlipBudget;
  const controllerCertifiedSetSize = controllerState?.certified_set_size ?? null;
  const controllerWeightSetVolume = controllerState?.weight_set_volume ?? null;
  const controllerWeightSetShrinkage = controllerState?.weight_set_shrinkage ?? null;
  const controllerUnresolvedFrontierMass = controllerState?.unresolved_possible_frontier_mass ?? null;
  const controllerUnresolvedWinnerMass = controllerState?.unresolved_possible_winner_mass ?? null;
  const controllerUnresolvedCriticalMass = controllerState?.unresolved_certificate_critical_mass ?? null;
  const controllerSupportFlag = controllerState?.support_flag ?? supportFlag ?? null;
  const controllerSupportReason = controllerState?.out_of_support_reason ?? outOfSupportReason ?? supportReason ?? null;
  const controllerProxyOnlyFraction = controllerState?.proxy_only_fraction ?? proxyShareAverage;
  const controllerBoundaryChallenger = controllerBoundarySummary?.active_challenger_id ?? nearestChallenger;
  const controllerBoundaryCount = controllerBoundarySummary?.challenger_count ?? null;
  const controllerBoundaryKind = controllerBoundarySummary?.certificate_boundary_kind ?? null;
  const controllerAuditCoverageRatio = controllerAuditSummary?.audit_coverage_ratio ?? null;
  const controllerMinimumPropensity = controllerAuditSummary?.minimum_propensity ?? null;
  const controllerMeanPropensity = controllerAuditSummary?.mean_propensity ?? null;
  const controllerPositivityOk = controllerAuditSummary?.positivity_ok ?? null;
  const controllerWeakOverlap = controllerAuditSummary?.weak_overlap_detected ?? null;
  const controllerCorrectionEstimator = controllerAuditSummary?.correction_path_estimator ?? null;
  const controllerEvaluationTag = controllerAuditSummary?.certification_evaluation_tag ?? null;
  const controllerRealizedRadiusDelta =
    artifactState.voiStopCertificate?.realized_delta_radius_or_flip_budget ?? null;
  const controllerRealizedPreferenceShrinkage =
    artifactState.voiStopCertificate?.realized_preference_shrinkage ?? null;
  const controllerRealizedCertifiedSetContraction =
    artifactState.voiStopCertificate?.realized_certified_set_contraction ?? null;
  const controllerHindsightNecessity =
    textOrNull(artifactState.voiStopCertificate?.hindsight_necessity_label) ?? null;
  const traceRows =
    artifactState.voiActionTrace?.actions ??
    artifactState.voiStopCertificate?.action_trace ??
    [];
  const bestRejectedAction =
    artifactState.voiStopCertificate?.best_rejected_action ??
    null;
  const terminalActionFamily =
    textOrNull(artifactState.voiStopCertificate?.terminal_action_family) ??
    actionFamilyForKind(textOrNull(artifactState.voiStopCertificate?.terminal_action_kind));
  const terminalActionModality =
    textOrNull(artifactState.voiStopCertificate?.terminal_action_modality) ??
    actionModalityForKind(textOrNull(artifactState.voiStopCertificate?.terminal_action_kind));
  const headerTone =
    terminalType === 'typed_abstention'
      ? 'mixed'
      : certification?.certified
        ? 'high'
        : 'mixed';
  const witnessTargetRouteId =
    route?.id ??
    witnessSummary?.route_id ??
    certification?.route_id ??
    selectedRouteId ??
    null;
  const witnessSupportNote = [supportReason, outOfSupportReason]
    .filter((value): value is string => Boolean(value))
    .join('; ');
  const witnessExplanation = [
    terminalType || witnessTargetRouteId
      ? `Terminal outcome: ${terminalLabel(terminalType)}${
          witnessTargetRouteId ? ` for ${witnessTargetRouteId}` : ''
        }.`
      : null,
    selectedCertificateBasis ? `Certificate basis: ${selectedCertificateBasis}.` : null,
    supportFlag !== null && supportFlag !== undefined
      ? `Support status: ${statusLabel(supportFlag)}${
          witnessSupportNote ? ` (${witnessSupportNote})` : ''
        }.`
      : witnessSupportNote
        ? `Support note: ${witnessSupportNote}.`
        : null,
    resolvedStopReason ? `Controller stop reason: ${text(resolvedStopReason)}.` : null,
    witnessSummary?.witness_size !== null && witnessSummary?.witness_size !== undefined
      ? `Witness size: ${n(locale, witnessSummary.witness_size)} atomic items.`
      : null,
    witnessSummary?.active_challenger_ids?.length
      ? `Active challengers: ${witnessSummary.active_challenger_ids.join(', ')}.`
      : null,
    witnessSummary?.active_evidence_families?.length
      ? `Active evidence families: ${witnessSummary.active_evidence_families.join(', ')}.`
      : null,
  ]
    .filter((value): value is string => Boolean(value))
    .join(' ');
  const exportSubtitle = `Pipeline ${pipelineMode.toUpperCase()}${runId ? ` | Run ${runId}` : ''}`;
  const decisionCardRows: Array<[string, string]> = [
    ['Terminal outcome', terminalLabel(terminalType)],
    ['Certificate basis', selectedCertificateBasis ?? 'n/a'],
    ['Certificate', pct(locale, certification?.certificate ?? null)],
    ['Threshold', pct(locale, certification?.threshold ?? null)],
    ['Lower confidence bound', pct(locale, lowerConfidenceBound)],
    ['Controller certificate LCB', pct(locale, controllerCertificateLcb)],
    ['Controller certificate UCB', pct(locale, controllerCertificateUcb)],
    ['Nearest challenger', nearestChallenger ?? 'n/a'],
    ['Minimum pairwise gap LCB', n(locale, minimumPairwiseGapLcb)],
    ['Minimum flip budget', n(locale, minimumFlipBudget)],
    ['Deterministic flip radius', n(locale, controllerDeterministicFlipRadius)],
    ['Probabilistic flip radius', n(locale, controllerProbabilisticFlipRadius)],
    ['Necessary-best probability', pct(locale, controllerNecessaryBest)],
    ['Possible-best probability', pct(locale, controllerPossibleBest)],
    ['Certified-set size', n(locale, controllerCertifiedSetSize)],
    ['Weight-set volume', n(locale, controllerWeightSetVolume)],
    ['Weight-set shrinkage', pct(locale, controllerWeightSetShrinkage)],
    ['Top refresh family', certification?.top_value_of_refresh_family ?? 'n/a'],
    ['Active evidence families', n(locale, activeFamilies.length)],
  ];
  const decisionCardCsv = buildCsvDocument(
    ['field', 'value'],
    decisionCardRows.map(([field, value]) => [field, value]),
  );
  const decisionCardSvg = buildSvgFigure(
    'Decision Card',
    exportSubtitle,
    decisionCardRows.map(([field, value]) => `${field}: ${value}`),
  );
  const controllerTraceSummaryRows: Array<[string, string]> = [
    ['Terminal outcome', terminalLabel(terminalType)],
    ['Stop reason', text(resolvedStopReason)],
    ['Search completeness', n(locale, resolvedSearchCompleteness)],
    ['Search gap', n(locale, resolvedSearchGap)],
    ['Iterations', n(locale, resolvedIterationCount)],
    ['Search budget used', n(locale, resolvedSearchBudgetUsed)],
    ['Evidence budget used', n(locale, resolvedEvidenceBudgetUsed)],
    ['Terminal action family', terminalActionFamily ?? 'n/a'],
    ['Terminal action modality', terminalActionModality ?? 'n/a'],
    ['Boundary challenger', controllerBoundaryChallenger ?? 'n/a'],
    ['Boundary kind', controllerBoundaryKind ?? 'n/a'],
    ['Boundary challenger count', n(locale, controllerBoundaryCount)],
    ['Unresolved frontier mass', pct(locale, controllerUnresolvedFrontierMass)],
    ['Unresolved winner mass', pct(locale, controllerUnresolvedWinnerMass)],
    ['Unresolved critical mass', pct(locale, controllerUnresolvedCriticalMass)],
    ['Realized radius delta', n(locale, controllerRealizedRadiusDelta)],
    ['Realized preference shrinkage', pct(locale, controllerRealizedPreferenceShrinkage)],
    ['Realized certified-set contraction', n(locale, controllerRealizedCertifiedSetContraction)],
    ['Hindsight necessity', controllerHindsightNecessity ?? 'n/a'],
  ];
  const controllerTraceRows = traceRows.map((traceRow) => {
    const chosenAction = traceRow.chosen_action ?? null;
    const feasibleActions = traceRow.feasible_actions ?? [];
    const nextUnusedAction =
      traceRow.next_best_unused_action ??
      feasibleActions.find((action) => action.action_id !== chosenAction?.action_id && action.kind !== 'stop') ??
      null;
    return [
      String(traceRow.iteration ?? ''),
      resolveActionFamily(chosenAction),
      resolveActionModality(chosenAction),
      text(chosenAction?.target),
      pct(locale, chosenAction?.predicted_delta_certificate ?? null),
      n(locale, chosenAction?.predicted_delta_margin ?? null),
      pct(locale, chosenAction?.predicted_winner_lcb_gain ?? null),
      n(locale, chosenAction?.predicted_gap_lcb_gain ?? null),
      n(locale, chosenAction?.predicted_radius_or_flip_budget_gain ?? null),
      pct(locale, chosenAction?.predicted_unresolved_mass_reduction ?? null),
      pct(locale, chosenAction?.predicted_preference_ambiguity_reduction ?? null),
      n(locale, chosenAction?.predicted_boundary_contraction ?? null),
      n(locale, chosenAction?.predicted_delta_radius_or_flip_budget ?? null),
      pct(locale, chosenAction?.predicted_preference_shrinkage ?? null),
      n(locale, chosenAction?.predicted_certified_set_contraction ?? null),
      n(locale, traceRow.realized_certificate_before),
      n(locale, traceRow.realized_certificate_after),
      n(locale, traceRow.realized_certificate_delta),
      n(locale, chosenAction?.cost_search ?? null),
      n(locale, chosenAction?.cost_evidence ?? null),
      n(locale, chosenAction?.q_score ?? null),
      text(chosenAction?.reason),
      nextUnusedAction ? `${resolveActionFamily(nextUnusedAction)} on ${text(nextUnusedAction.target)}` : 'n/a',
    ];
  });
  const controllerTraceCsv = buildCsvDocument(
    [
      'iteration',
      'chosen_action_family',
      'chosen_action_modality',
      'target',
      'predicted_certificate_delta',
      'predicted_gap_delta',
      'predicted_winner_lcb_gain',
      'predicted_gap_lcb_gain',
      'predicted_radius_or_flip_budget_gain',
      'predicted_unresolved_mass_reduction',
      'predicted_preference_ambiguity_reduction',
      'predicted_boundary_contraction',
      'predicted_delta_radius_or_flip_budget',
      'predicted_preference_shrinkage',
      'predicted_certified_set_contraction',
      'realized_certificate_before',
      'realized_certificate_after',
      'realized_certificate_delta',
      'search_cost',
      'evidence_cost',
      'q_score',
      'reason',
      'next_best_unused_action',
    ],
    controllerTraceRows.length ? controllerTraceRows : [Array.from({ length: 23 }, () => '')],
  );
  const controllerTraceSvg = buildSvgFigure(
    'Controller Trace',
    exportSubtitle,
    [
      ...controllerTraceSummaryRows.map(([field, value]) => `${field}: ${value}`),
      ...(traceRows.length
        ? traceRows.slice(0, 4).map((traceRow) => {
            const chosenAction = traceRow.chosen_action ?? null;
            return `Iteration ${n(locale, traceRow.iteration)}: ${resolveActionFamily(chosenAction)} / ${resolveActionModality(chosenAction)} on ${text(chosenAction?.target)}; predicted cert ${pct(locale, chosenAction?.predicted_delta_certificate ?? null)}; realized delta ${n(locale, traceRow.realized_certificate_delta)}.`;
          })
        : ['No controller action steps were recorded for this run.']),
    ],
  );
  const evidenceSummaryRows: Array<[string, string]> = [
    ['Support status', statusLabel(supportFlag)],
    ['Support note', supportStatus ?? supportReason ?? 'n/a'],
    ['Out-of-support warning', outOfSupportReason ?? 'n/a'],
    ['Calibration bin', supportCalibrationBin],
    ['Scenario / profile provenance', scenarioProvenanceBits.length ? scenarioProvenanceBits.join(' | ') : 'n/a'],
    ['Mode observation source', scenarioSummary?.mode_observation_source ?? 'n/a'],
    ['Projection ratio', pct(locale, scenarioSummary?.mode_projection_ratio ?? null)],
    ['Probabilistic worlds', n(locale, probabilisticWorldCount)],
    ['Unique worlds', n(locale, uniqueWorldCount)],
    ['Audit worlds', n(locale, auditWorldCount)],
    ['Proxy exposure', pct(locale, proxyShareAverage)],
    ['World reuse rate', pct(locale, worldReuseRate)],
    ['Dominant fragility family', dominantFragilityFamily ?? 'n/a'],
    ['Chosen refresh family', chosenRefreshFamily ?? 'n/a'],
    ['Controller refresh pick', controllerRefreshFamily ?? empiricalRefreshFamily ?? 'n/a'],
    ['Baseline certificate', pct(locale, baselineCertificate)],
    ['Controller support', statusLabel(controllerSupportFlag)],
    ['Controller support reason', controllerSupportReason ?? 'n/a'],
    ['Proxy-only fraction', pct(locale, controllerProxyOnlyFraction)],
    ['Audit coverage ratio', pct(locale, controllerAuditCoverageRatio)],
    ['Minimum propensity', pct(locale, controllerMinimumPropensity)],
    ['Mean propensity', pct(locale, controllerMeanPropensity)],
    ['Positivity ok', controllerPositivityOk === null ? 'n/a' : controllerPositivityOk ? 'yes' : 'no'],
    ['Weak overlap detected', controllerWeakOverlap === null ? 'n/a' : controllerWeakOverlap ? 'yes' : 'no'],
    ['Correction path estimator', controllerCorrectionEstimator ?? 'n/a'],
    ['Certification evaluation tag', controllerEvaluationTag ?? 'n/a'],
    ['Boundary challenger', controllerBoundaryChallenger ?? 'n/a'],
    ['Boundary kind', controllerBoundaryKind ?? 'n/a'],
  ];
  const evidenceSummaryCsv = buildCsvDocument(
    ['field', 'value'],
    evidenceSummaryRows.map(([field, value]) => [field, value]),
  );
  const evidenceSummarySvg = buildSvgFigure(
    'Evidence Summary',
    exportSubtitle,
    [
      ...evidenceSummaryRows.map(([field, value]) => `${field}: ${value}`),
      ...(fragilityEntries.length
        ? fragilityEntries.slice(0, 4).map(([family, fragility]) => `${family}: fragility ${pct(locale, fragility)}`)
        : ['No route-level fragility breakdown was available from the current artifacts.']),
    ],
  );

  return (
    <section className="baselineComparePanel">
      <div className="baselineComparePanel__head">
        <div className="baselineComparePanel__title">VOI / Certification</div>
        <div className={`baselineEpicScore baselineEpicScore--${headerTone}`}>
          {pipelineMode.toUpperCase()}
        </div>
      </div>
      <div className="baselineComparePanel__epicNote">
        Pipeline mode: <strong>{pipelineMode}</strong>
        {runId ? (
          <>
            {' '}
            | Run ID <code>{runId}</code>
          </>
        ) : null}
      </div>

      <div className="baselineComparePanel__tradeoff">
        Governance: basis {selectedCertificateBasis ?? 'n/a'}
        {witnessSummary?.witness_size !== undefined
          ? `; witness size ${n(locale, witnessSummary.witness_size)}`
          : ''}
        {witnessSummary?.active_challenger_ids?.length
          ? `; challengers ${witnessSummary.active_challenger_ids.join(', ')}`
          : ''}
        {witnessSummary?.active_evidence_families?.length
          ? `; evidence ${witnessSummary.active_evidence_families.join(', ')}`
          : ''}
        {actionTraceSummary?.stop_reason ? `; stop ${text(actionTraceSummary.stop_reason)}` : ''}
        {actionTraceSummary?.search_completeness_score !== undefined
          ? `; search ${n(locale, actionTraceSummary.search_completeness_score)}`
          : ''}
        {actionTraceSummary?.search_completeness_gap !== undefined
          ? `; gap ${n(locale, actionTraceSummary.search_completeness_gap)}`
          : ''}
      </div>

      {certification ? (
        <>
          <div className="baselineKpiGrid">
            <div className={`baselineKpi ${certification.certified ? 'isPositive' : 'isNegative'}`}>
              <div className="baselineKpi__label">
                {metricLabel('Certificate', {
                  definition: 'Empirical certificate value for the selected route or current decision payload.',
                  direction: 'Higher is better because more certificate mass supports the terminal decision.',
                  unit: 'probability or share',
                })}
              </div>
              <div className="baselineKpi__value">{pct(locale, certification.certificate)}</div>
              <div className="baselineKpi__meta">
                {inlineMetricLabel('Threshold', {
                  definition: 'Decision threshold the certificate was compared against for the current run.',
                  direction: 'Context only; this is the required bar rather than a better-or-worse score by itself.',
                  unit: 'probability or share threshold',
                })}{' '}
                {pct(locale, certification.threshold)} ({certification.certified ? 'Certified' : 'Uncertified'})
              </div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Lower Confidence Bound', {
                  definition:
                    'Lower confidence bound recorded in the certificate summary for the current winner.',
                  direction:
                    'Higher is better because a stronger lower bound indicates more conservative certificate support.',
                  unit: 'probability or share lower bound',
                })}
              </div>
              <div className="baselineKpi__value">{pct(locale, lowerConfidenceBound)}</div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Top Refresh Family', {
                  definition: 'Evidence family with the highest refresh value according to the certificate summary.',
                  direction: 'Context only; this identifies the next evidence family worth refreshing rather than ranking route quality.',
                  unit: 'categorical evidence-family label',
                })}
              </div>
              <div className="baselineKpi__value">{certification.top_value_of_refresh_family ?? 'n/a'}</div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Nearest Challenger', {
                  definition:
                    'Nearest active challenger named by the decision-region or certificate evidence surface.',
                  direction:
                    'Context only; this identifies the closest competing route rather than scoring route quality.',
                  unit: 'route identifier',
                })}
              </div>
              <div className="baselineKpi__value">{nearestChallenger ?? 'n/a'}</div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Minimum Pairwise Gap LCB', {
                  definition:
                    'Smallest pairwise gap lower confidence bound recorded for the current winner versus its challengers.',
                  direction:
                    'Higher is better because a wider certified gap means more separation from the nearest challenger.',
                  unit: 'certificate-gap lower bound',
                })}
              </div>
              <div className="baselineKpi__value">{n(locale, minimumPairwiseGapLcb)}</div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Minimum Flip Budget', {
                  definition:
                    'Smallest perturbation budget required to flip the current certification state according to the flip-radius surface.',
                  direction:
                    'Higher is better because it takes a larger perturbation to overturn the current decision.',
                  unit: 'flip-budget units',
                })}
              </div>
              <div className="baselineKpi__value">{n(locale, minimumFlipBudget)}</div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Active Evidence Families', {
                  definition: 'Count of evidence families still active in the current decision state.',
                  direction: 'Lower is usually better because fewer active evidence families remain on the proof boundary.',
                  unit: 'evidence-family count',
                })}
              </div>
              <div className="baselineKpi__value">{activeFamilies.length}</div>
            </div>
          </div>
          {certification.top_fragility_families?.length ? (
            <div className="baselineComparePanel__tradeoff">
              Fragility drivers: {certification.top_fragility_families.join(', ')}.
            </div>
          ) : null}
          <div className="actionGrid u-mt10">
            <button
              type="button"
              className="secondary"
              onClick={() =>
                downloadTextFile(
                  decisionCardCsv,
                  `decision-card-${runId ?? selectedRouteId ?? 'current'}.csv`,
                  'text/csv;charset=utf-8',
                )
              }
            >
              Decision Card CSV
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() =>
                downloadTextFile(
                  decisionCardSvg,
                  `decision-card-${runId ?? selectedRouteId ?? 'current'}.svg`,
                  'image/svg+xml;charset=utf-8',
                )
              }
            >
              Decision Card SVG
            </button>
          </div>
        </>
      ) : terminalType === 'typed_abstention' ? (
        <div className="baselineComparePanel__loading">
          No route was certified. Support and governance context below corresponds to a typed abstention outcome.
        </div>
      ) : terminalType === 'certified_set' ? (
        <div className="baselineComparePanel__loading">
          No singleton route was certified. Support and governance context below corresponds to a certified set outcome.
        </div>
      ) : (
        <div className="baselineComparePanel__loading">
          No certification summary was returned for the current decision payload. Support and governance context remain available below.
        </div>
      )}

      <div className="fieldLabel u-mb6 u-mt12">
        Witness Explanation
        <FieldInfo
          text={metricHelp({
            definition:
              'Deterministic human-readable explanation synthesized only from witness, support, terminal, and stop fields already present on this certificate panel.',
            direction:
              'Context only; this section explains the current decision state rather than ranking route quality.',
            unit: 'deterministic explanation text',
          })}
        />
      </div>
      <div className="baselineComparePanel__tradeoff">
        {witnessExplanation || 'No witness-driven explanation was available for the current decision payload.'}
      </div>

      <div className="fieldLabel u-mb6 u-mt12">
        Controller Context
        <FieldInfo
          text={metricHelp({
            definition: 'Summary of the terminal controller state for the current decision payload.',
            direction: 'Context only; this section explains how the controller stopped rather than ranking route quality.',
            unit: 'mixed controller state fields',
          })}
        />
      </div>
      <div className="baselineKpiGrid">
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Terminal Outcome', {
              definition: 'User-facing stop class returned by the certification engine.',
              direction: 'Context only; singleton, set, or abstention are different stop modes rather than a better/worse scale.',
              unit: 'categorical terminal class',
            })}
          </div>
          <div className="baselineKpi__value">{terminalLabel(terminalType)}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Stop Reason', {
              definition: 'Recorded reason for why the controller stopped acting or abstained.',
              direction: 'Context only; this explains the stop decision rather than ranking the run.',
              unit: 'categorical stop reason',
            })}
          </div>
          <div className="baselineKpi__value">{text(resolvedStopReason)}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Search Completeness', {
              definition: 'Search-completeness score reported at the terminal decision.',
              direction: 'Higher is better because more search deficiency has been resolved.',
              unit: 'search-completeness score',
            })}
          </div>
          <div className="baselineKpi__value">{n(locale, resolvedSearchCompleteness)}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Search Gap', {
              definition: 'Unresolved search-side shortfall remaining at the terminal decision.',
              direction: 'Lower is better because less search ambiguity remains.',
              unit: 'search-gap score',
            })}
          </div>
          <div className="baselineKpi__value">{n(locale, resolvedSearchGap)}</div>
        </div>
      </div>
      {controllerState ? (
        <>
          <div className="fieldLabel u-mb6 u-mt12">
            Controller Literal State
            <FieldInfo
              text={metricHelp({
                definition:
                  'Literal controller-state fields emitted in the terminal stop certificate and mirrored into the controller-state trace artifact.',
                direction:
                  'Context only; these fields expose proof boundary pressure, support, audit, and robustness state rather than one higher-is-better scalar.',
                unit: 'mixed controller-state fields',
              })}
            />
          </div>
          <div className="baselineComparePanel__tradeoff">
            Controller state exposes the certificate interval, active boundary summary, support regime,
            proxy-only share, audit-propensity summary, and unresolved proof mass directly from the
            emitted runtime state.
          </div>
          <div className="baselineKpiGrid">
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Certificate Interval', {
                  definition:
                    'Lower and upper certificate bounds emitted on the literal controller state for the terminal decision.',
                  direction:
                    'Higher lower bounds and tighter intervals are generally better because they indicate stronger terminal support.',
                  unit: 'certificate lower and upper bounds',
                })}
              </div>
              <div className="baselineKpi__value">
                {pct(locale, controllerCertificateLcb)} to {pct(locale, controllerCertificateUcb)}
              </div>
              <div className="baselineKpi__meta">
                {inlineMetricLabel('Necessary-best', {
                  definition: 'Probability mass under which the winner is already necessary-best.',
                  direction: 'Higher is better because more of the remaining state space already certifies the winner as necessary-best.',
                  unit: 'probability mass',
                })}{' '}
                {pct(locale, controllerNecessaryBest)} |{' '}
                {inlineMetricLabel('Possible-best', {
                  definition: 'Probability mass under which the winner remains possible-best.',
                  direction: 'Lower is usually better once a singleton is stable because less unresolved possible-best mass remains.',
                  unit: 'probability mass',
                })}{' '}
                {pct(locale, controllerPossibleBest)}
              </div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Boundary Pressure', {
                  definition:
                    'Active certificate-boundary challenger and challenger count recorded on the controller-state boundary summary.',
                  direction:
                    'Lower challenger counts are usually better because fewer routes remain active on the proof boundary.',
                  unit: 'challenger identity and count',
                })}
              </div>
              <div className="baselineKpi__value">{controllerBoundaryChallenger ?? 'n/a'}</div>
              <div className="baselineKpi__meta">
                {inlineMetricLabel('Boundary kind', {
                  definition: 'Categorical label for the active certificate-boundary regime.',
                  direction: 'Context only; this names the boundary regime rather than scoring the route.',
                  unit: 'categorical boundary label',
                })}{' '}
                {controllerBoundaryKind ?? 'n/a'} | challengers {n(locale, controllerBoundaryCount)}
              </div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Robustness Budget', {
                  definition:
                    'Robustness margin from the literal controller state, including deterministic and probabilistic flip-radius views and the minimum flip budget.',
                  direction:
                    'Higher is better because larger radii and budgets mean the terminal decision is harder to overturn.',
                  unit: 'flip radius or budget',
                })}
              </div>
              <div className="baselineKpi__value">{n(locale, controllerMinimumFlipBudget)}</div>
              <div className="baselineKpi__meta">
                deterministic {n(locale, controllerDeterministicFlipRadius)} | probabilistic{' '}
                {n(locale, controllerProbabilisticFlipRadius)}
              </div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Weight-Set State', {
                  definition:
                    'Literal weight-set volume proxy, shrinkage, and surviving certified-set size emitted by the controller state.',
                  direction:
                    'Lower surviving volume and larger shrinkage are usually better because the preference or proof region has tightened.',
                  unit: 'volume proxy, shrinkage fraction, and certified-set size',
                })}
              </div>
              <div className="baselineKpi__value">{n(locale, controllerWeightSetVolume)}</div>
              <div className="baselineKpi__meta">
                shrinkage {pct(locale, controllerWeightSetShrinkage)} | certified set{' '}
                {n(locale, controllerCertifiedSetSize)}
              </div>
            </div>
            <div className={`baselineKpi ${controllerSupportFlag === false ? 'isNegative' : controllerSupportFlag === true ? 'isPositive' : ''}`}>
              <div className="baselineKpi__label">
                {metricLabel('Controller Support', {
                  definition:
                    'Support flag and out-of-support reason emitted directly on the controller state rather than inferred from a separate support payload.',
                  direction:
                    'In-support is better for trustworthy certification; out-of-support means downstream claims should be interpreted more conservatively.',
                  unit: 'categorical support status and reason',
                })}
              </div>
              <div className="baselineKpi__value">{statusLabel(controllerSupportFlag)}</div>
              <div className="baselineKpi__meta">{controllerSupportReason ?? 'n/a'}</div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Proxy / Audit Posture', {
                  definition:
                    'Proxy-only share and audit-propensity coverage summary emitted directly on the controller state.',
                  direction:
                    'Lower proxy-only share and stronger coverage/propensity are usually better because less of the proof relies on cheap-only evidence.',
                  unit: 'proxy share and audit-propensity summary',
                })}
              </div>
              <div className="baselineKpi__value">{pct(locale, controllerProxyOnlyFraction)}</div>
              <div className="baselineKpi__meta">
                coverage {pct(locale, controllerAuditCoverageRatio)} | min prop {pct(locale, controllerMinimumPropensity)} | mean prop {pct(locale, controllerMeanPropensity)}
              </div>
            </div>
          </div>
          <ul className="baselineNotes">
            <li>
              Unresolved mass: frontier {pct(locale, controllerUnresolvedFrontierMass)}, winner {pct(locale, controllerUnresolvedWinnerMass)}, certificate-critical {pct(locale, controllerUnresolvedCriticalMass)}.
            </li>
            <li>
              Audit posture: positivity {controllerPositivityOk === null ? 'n/a' : controllerPositivityOk ? 'ok' : 'not ok'}, weak overlap {controllerWeakOverlap === null ? 'n/a' : controllerWeakOverlap ? 'detected' : 'not detected'}, corrected path {controllerCorrectionEstimator ?? 'n/a'}, evaluation tag {controllerEvaluationTag ?? 'n/a'}.
            </li>
            <li>
              Realized preference updates: radius delta {n(locale, controllerRealizedRadiusDelta)}, shrinkage {pct(locale, controllerRealizedPreferenceShrinkage)}, certified-set contraction {n(locale, controllerRealizedCertifiedSetContraction)}, hindsight necessity {controllerHindsightNecessity ?? 'n/a'}.
            </li>
          </ul>
        </>
      ) : null}
      <div className="fieldLabel u-mb6 u-mt12">
        Controller Trace
        <FieldInfo
          text={metricHelp({
            definition: 'Per-iteration controller action history and stop-certificate summary.',
            direction: 'Context only; this section explains actioning and stopping behavior.',
            unit: 'mixed controller trace fields',
          })}
        />
      </div>
      <div className="baselineKpiGrid">
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Iterations', {
              definition: 'Number of controller iterations recorded before the run stopped.',
              direction: 'Lower is usually better for cost and latency when matched decision quality is preserved.',
              unit: 'iteration count',
            })}
          </div>
          <div className="baselineKpi__value">{n(locale, resolvedIterationCount)}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Search Budget Used', {
              definition: 'Search budget consumed across the recorded controller run.',
              direction: 'Lower is usually better for efficiency when certification quality is held constant.',
              unit: 'search budget units',
            })}
          </div>
          <div className="baselineKpi__value">{n(locale, resolvedSearchBudgetUsed)}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Evidence Budget Used', {
              definition: 'Evidence budget consumed across the recorded controller run.',
              direction: 'Lower is usually better for efficiency when certification quality is held constant.',
              unit: 'evidence budget units',
            })}
          </div>
          <div className="baselineKpi__value">{n(locale, resolvedEvidenceBudgetUsed)}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Terminal Action', {
              definition: 'Family and modality of the terminal action recorded in the stop certificate.',
              direction: 'Context only; this names the stopping action rather than ranking the run.',
              unit: 'categorical action family and modality',
            })}
          </div>
          <div className="baselineKpi__value">
            {terminalActionFamily ?? 'n/a'}
          </div>
          <div className="baselineKpi__meta">
            {terminalActionModality ?? 'n/a'}
          </div>
        </div>
      </div>
      {traceRows.length ? (
        <div className="baselineComparePanel__artifactSection">
          {traceRows.map((traceRow) => {
            const chosenAction = traceRow.chosen_action ?? null;
            const feasibleActions = traceRow.feasible_actions ?? [];
            const nextUnusedAction =
              traceRow.next_best_unused_action ??
              feasibleActions.find((action) => action.action_id !== chosenAction?.action_id && action.kind !== 'stop') ??
              null;
            const traceKey =
              textOrNull(chosenAction?.action_id) ??
              textOrNull(nextUnusedAction?.action_id) ??
              String(traceRow.iteration ?? 'trace-row');
            return (
              <div key={`trace-${traceKey}`} className="baselineComparePanel__tradeoff">
                <strong>Iteration {n(locale, traceRow.iteration)}</strong>
                {chosenAction ? (
                  <>
                    {`: ${resolveActionFamily(chosenAction)} / ${resolveActionModality(chosenAction)} on ${text(chosenAction.target)};`}
                    {' '}predicted cert {pct(locale, chosenAction.predicted_delta_certificate ?? null)}
                    {`; predicted gap ${n(locale, chosenAction.predicted_delta_margin ?? null)}`}
                    {`; winner LCB ${pct(locale, chosenAction.predicted_winner_lcb_gain ?? null)}`}
                    {`; unresolved ${pct(locale, chosenAction.predicted_unresolved_mass_reduction ?? null)}`}
                    {`; pref ambiguity ${pct(locale, chosenAction.predicted_preference_ambiguity_reduction ?? null)}`}
                    {`; realized cert ${n(locale, traceRow.realized_certificate_before)} -> ${n(locale, traceRow.realized_certificate_after)}`}
                    {`; realized delta ${n(locale, traceRow.realized_certificate_delta)}`}
                    {`; cost S${n(locale, chosenAction.cost_search ?? null)} / E${n(locale, chosenAction.cost_evidence ?? null)}`}
                    {`; q ${n(locale, chosenAction.q_score ?? null)}`}
                    {`; reason ${text(chosenAction.reason)}`}
                    {chosenAction.preconditions?.length ? `; preconditions ${chosenAction.preconditions.join(', ')}` : ''}
                    {traceRow.realized_productive !== undefined && traceRow.realized_productive !== null
                      ? `; productive ${traceRow.realized_productive ? 'yes' : 'no'}`
                      : ''}
                  </>
                ) : (
                  <>: no chosen action recorded.</>
                )}
                {nextUnusedAction ? (
                  <>
                    {' '}Next best unused action: {resolveActionFamily(nextUnusedAction)} / {resolveActionModality(nextUnusedAction)} on{' '}
                    {text(nextUnusedAction.target)} with predicted cert {pct(locale, nextUnusedAction.predicted_delta_certificate ?? null)}.
                  </>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="baselineComparePanel__loading">
          No controller action steps were recorded for this run. The terminal stop certificate still explains why the controller stopped or abstained.
        </div>
      )}
      {bestRejectedAction ? (
        <div className="baselineComparePanel__tradeoff">
          Next best unused action:
          {' '}<strong>{resolveActionFamily(bestRejectedAction)} / {resolveActionModality(bestRejectedAction)}</strong>
          {' '}on {text(bestRejectedAction.target)}
          {`; predicted cert ${pct(locale, bestRejectedAction.predicted_delta_certificate ?? null)}`}
          {`; winner LCB ${pct(locale, bestRejectedAction.predicted_winner_lcb_gain ?? null)}`}
          {`; unresolved ${pct(locale, bestRejectedAction.predicted_unresolved_mass_reduction ?? null)}`}
          {`; q ${n(locale, bestRejectedAction.q_score ?? null)}`}
          {`; reason ${text(bestRejectedAction.reason)}`}
        </div>
      ) : null}
      {(voiActionTraceHref || voiActionScoresHref || voiStopHref || voiControllerStateHref) ? (
        <div className="actionGrid u-mt10">
          {voiActionTraceHref ? (
            <a className="secondary" href={voiActionTraceHref} target="_blank" rel="noreferrer">
              Action trace JSON
            </a>
          ) : null}
          {voiActionScoresHref ? (
            <a className="secondary" href={voiActionScoresHref} target="_blank" rel="noreferrer">
              Action scores CSV
            </a>
          ) : null}
          {voiStopHref ? (
            <a className="secondary" href={voiStopHref} target="_blank" rel="noreferrer">
              Stop certificate JSON
            </a>
          ) : null}
          {voiControllerStateHref ? (
            <a className="secondary" href={voiControllerStateHref} target="_blank" rel="noreferrer">
              Controller state JSONL
            </a>
          ) : null}
        </div>
      ) : null}
      <div className="actionGrid u-mt10">
        <button
          type="button"
          className="secondary"
          onClick={() =>
            downloadTextFile(
              controllerTraceCsv,
              `controller-trace-${runId ?? selectedRouteId ?? 'current'}.csv`,
              'text/csv;charset=utf-8',
            )
          }
        >
          Controller Trace CSV
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() =>
            downloadTextFile(
              controllerTraceSvg,
              `controller-trace-${runId ?? selectedRouteId ?? 'current'}.svg`,
              'image/svg+xml;charset=utf-8',
            )
          }
        >
          Controller Trace SVG
        </button>
      </div>

      <div className="fieldLabel u-mb6 u-mt12">
        Support &amp; Governance
        <FieldInfo
          text={metricHelp({
            definition: 'Support, scenario, and governance context for the current decision.',
            direction: 'Context only; these fields indicate calibration and provenance, not route superiority.',
            unit: 'mixed support and provenance fields',
          })}
        />
      </div>
      <div className="baselineKpiGrid">
        <div className={`baselineKpi ${supportFlag === false ? 'isNegative' : supportFlag === true ? 'isPositive' : ''}`}>
          <div className="baselineKpi__label">
            {metricLabel('Support Status', {
              definition: 'Whether the world model is considered in-support for this decision.',
              direction: 'In-support is better for trustworthiness; out-of-support means downgraded certification claims.',
              unit: 'categorical support status',
              note: 'The panel falls back to the inline response payload when the standalone support artifact is not proxied.',
            })}
          </div>
          <div className="baselineKpi__value">{statusLabel(supportFlag)}</div>
          <div className="baselineKpi__meta">{supportStatus ?? supportReason ?? 'No support status returned'}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Calibration Bin', {
              definition: 'Support/calibration bucket used for support checks and certificate interpretation.',
              direction: 'Context only; this labels the calibration regime rather than scoring quality.',
              unit: 'categorical calibration-bin label',
            })}
          </div>
          <div className="baselineKpi__value">{supportCalibrationBin}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Scenario / Profile Provenance', {
              definition: 'Scenario mode, context key, source, and version that produced the current decision payload.',
              direction: 'Context only; provenance supports auditability rather than better/worse ranking.',
              unit: 'categorical provenance label set',
            })}
          </div>
          <div className="baselineKpi__value">{scenarioSummary?.mode ?? 'n/a'}</div>
          <div className="baselineKpi__meta">{scenarioProvenanceBits.length ? scenarioProvenanceBits.join(' | ') : 'No scenario provenance returned'}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Mode Observation Source', {
              definition: 'Observation stream or scenario basis that informed the current mode projection.',
              direction: 'Context only; this identifies the source rather than ranking route quality.',
              unit: 'categorical source label',
            })}
          </div>
          <div className="baselineKpi__value">{scenarioSummary?.mode_observation_source ?? 'n/a'}</div>
          <div className="baselineKpi__meta">
            {inlineMetricLabel('Projection ratio', {
              definition: 'Ratio used by the scenario summary when projecting observed mode information into the current decision context.',
              direction: 'Context only; this is scenario metadata, not a better/worse quality score.',
              unit: 'projection ratio',
            })}{' '}
            {pct(locale, scenarioSummary?.mode_projection_ratio ?? null)}
          </div>
        </div>
      </div>
      {(supportReason || outOfSupportReason || scenarioSummary?.live_sources) ? (
        <div className="baselineComparePanel__tradeoff">
          {supportReason ? `Support reason: ${supportReason}. ` : ''}
          {outOfSupportReason ? `Out-of-support warning: ${outOfSupportReason}. ` : ''}
          {scenarioSummary?.live_sources ? `Live source mix: ${scenarioSummary.live_sources}.` : ''}
        </div>
      ) : null}

      <div className="fieldLabel u-mb6 u-mt12">
        Evidence Audit
        <FieldInfo
          text={metricHelp({
            definition: 'Artifact-backed evidence summary covering world counts, audit usage, proxy exposure, reuse, fragility, and refresh value.',
            direction: 'Context only; these are auditability and sensitivity fields rather than one better/worse scalar.',
            unit: 'mixed evidence-audit fields',
          })}
        />
      </div>
      <div className="baselineKpiGrid">
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Probabilistic Worlds', {
              definition: 'Sampled worlds contributing to the probabilistic certification view.',
              direction: 'Context only; more worlds can improve coverage but are not automatically better without considering cost.',
              unit: 'world count',
            })}
          </div>
          <div className="baselineKpi__value">{n(locale, probabilisticWorldCount)}</div>
          <div className="baselineKpi__meta">
            {inlineMetricLabel('Unique', {
              definition: 'Distinct probabilistic worlds after reuse and deduplication effects.',
              direction: 'Context only; higher unique counts mean broader distinct world coverage, not automatic superiority.',
              unit: 'unique world count',
            })}{' '}
            {n(locale, uniqueWorldCount)}
            {artifactState.sampledWorldManifest?.world_count_policy
              ? ` | Policy ${artifactState.sampledWorldManifest.world_count_policy}`
              : ''}
          </div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Audit Worlds', {
              definition: 'Deterministic or audited evidence worlds tracked separately from the cheaper probabilistic world bundle.',
              direction: 'Context only; more audited worlds can strengthen evidence but also cost more.',
              unit: 'audit world count',
            })}
          </div>
          <div className="baselineKpi__value">{n(locale, auditWorldCount)}</div>
          <div className="baselineKpi__meta">
            Full {n(locale, auditFullCount)} | Partial {n(locale, auditPartialCount)} | Pair audits {n(locale, auditPairCount)}
          </div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Proxy Exposure', {
              definition: 'Approximate share of evidence mass contributed by proxy states in the sampled-world bundle.',
              direction: 'Lower is usually better for audit strength because less evidence mass remains proxy-only.',
              unit: 'proxy exposure fraction',
            })}
          </div>
          <div className="baselineKpi__value">{pct(locale, proxyShareAverage)}</div>
          <div className="baselineKpi__meta">
            {proxyHeavyFamily ? `${proxyHeavyFamily[0]} peaks at ${pct(locale, proxyHeavyFamily[1])}` : 'No proxy weights returned'}
          </div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Reused Worlds', {
              definition: 'Share of the evidence world bundle reused rather than regenerated for the current decision.',
              direction: 'Context only; higher reuse indicates a hotter rerun posture, not necessarily better route quality.',
              unit: 'reuse fraction',
            })}
          </div>
          <div className="baselineKpi__value">{pct(locale, worldReuseRate)}</div>
          <div className="baselineKpi__meta">Audit reused {n(locale, auditReuseCount)}</div>
        </div>
      </div>
      {(activeFamilies.length || worldKinds.length) ? (
        <div className="baselineComparePanel__tradeoff">
          {activeFamilies.length ? `Active families: ${activeFamilies.join(', ')}.` : ''}
          {worldKinds.length ? ` World kinds: ${worldKinds.join(', ')}.` : ''}
        </div>
      ) : null}
      <div className="baselineKpiGrid">
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Dominant Fragility Family', {
              definition: 'Strongest route-level sensitivity family from the fragility map or certificate summary.',
              direction: 'Context only; this identifies the dominant risk channel rather than ranking route quality.',
              unit: 'categorical fragility-family label',
            })}
          </div>
          <div className="baselineKpi__value">{dominantFragilityFamily ?? 'n/a'}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Chosen Refresh Family', {
              definition: 'Refresh family selected in the certificate summary for follow-up evidence actioning.',
              direction: 'Context only; this names the selected evidence target rather than scoring route quality.',
              unit: 'categorical evidence-family label',
            })}
          </div>
          <div className="baselineKpi__value">{chosenRefreshFamily ?? 'n/a'}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Controller Refresh Pick', {
              definition: 'Top family in the controller-ranked value-of-refresh artifact when available.',
              direction: 'Context only; this identifies the controller-preferred refresh target rather than a higher-is-better score.',
              unit: 'categorical evidence-family label',
            })}
          </div>
          <div className="baselineKpi__value">{controllerRefreshFamily ?? empiricalRefreshFamily ?? 'n/a'}</div>
          <div className="baselineKpi__meta">{controllerRankingBasis ? `Basis ${controllerRankingBasis}` : 'No controller ranking basis returned'}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Baseline Certificate', {
              definition: 'Pre-refresh certificate value recorded by the value-of-refresh artifact before isolating family-level gains.',
              direction: 'Higher is better because the baseline certificate starts closer to a justified terminal decision.',
              unit: 'probability or share',
            })}
          </div>
          <div className="baselineKpi__value">{pct(locale, baselineCertificate)}</div>
        </div>
      </div>
      {fragilityEntries.length ? (
        <ul className="baselineNotes">
          {fragilityEntries.slice(0, 5).map(([family, fragility]) => (
            <li key={family}>
              {family}: fragility {pct(locale, fragility)}
            </li>
          ))}
        </ul>
      ) : (
        <div className="baselineComparePanel__loading">
          No route-level fragility breakdown was available from the current artifacts.
        </div>
      )}
      {artifactState.valueOfRefresh ? (
        <div className="baselineComparePanel__tradeoff">
          Refresh ranking:
          {artifactState.valueOfRefresh.top_refresh_family
            ? ` empirical ${artifactState.valueOfRefresh.top_refresh_family} (${pct(locale, artifactState.valueOfRefresh.top_refresh_gain)})`
            : ' empirical ranking unavailable'}
          {artifactState.valueOfRefresh.top_refresh_family_controller
            ? `; controller ${artifactState.valueOfRefresh.top_refresh_family_controller} (${pct(locale, artifactState.valueOfRefresh.top_refresh_gain_controller)})`
            : ''}
          {artifactState.valueOfRefresh.fragility_stress_state
            ? `; stress state ${artifactState.valueOfRefresh.fragility_stress_state}`
            : ''}
        </div>
      ) : null}
      <div className="actionGrid u-mt10">
        <button
          type="button"
          className="secondary"
          onClick={() =>
            downloadTextFile(
              evidenceSummaryCsv,
              `evidence-summary-${runId ?? selectedRouteId ?? 'current'}.csv`,
              'text/csv;charset=utf-8',
            )
          }
        >
          Evidence Summary CSV
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() =>
            downloadTextFile(
              evidenceSummarySvg,
              `evidence-summary-${runId ?? selectedRouteId ?? 'current'}.svg`,
              'image/svg+xml;charset=utf-8',
            )
          }
        >
          Evidence Summary SVG
        </button>
      </div>
      <div className="baselineComparePanel__tradeoff">
        SVG exports are vector figure surfaces intended for PDF-ready placement. Direct PDF rendering remains a bundle-level artifact concern rather than a route-panel export claim.
      </div>
      {artifactState.loading ? (
        <div className="baselineComparePanel__loading">Loading evidence audit artifacts...</div>
      ) : null}
      {artifactState.error ? (
        <div className="baselineComparePanel__loading">{artifactState.error}</div>
      ) : null}

      {voiStopSummary ? (
        <div className="baselineImpactGrid">
          <div>
            <div className="baselineImpactGrid__label">
              {metricLabel('Iterations', {
                definition: 'Controller iterations reported in the compact stop summary artifact.',
                direction: 'Lower is usually better for efficiency when decision quality is preserved.',
                unit: 'iteration count',
              })}
            </div>
            <div className="baselineImpactGrid__value">{n(locale, voiStopSummary.iteration_count)}</div>
          </div>
          <div>
            <div className="baselineImpactGrid__label">
              {metricLabel('Search budget used', {
                definition: 'Search budget consumed according to the compact stop summary artifact.',
                direction: 'Lower is usually better for efficiency when matched quality is preserved.',
                unit: 'search budget units',
              })}
            </div>
            <div className="baselineImpactGrid__value">{n(locale, voiStopSummary.search_budget_used)}</div>
          </div>
          <div>
            <div className="baselineImpactGrid__label">
              {metricLabel('Evidence budget used', {
                definition: 'Evidence budget consumed according to the compact stop summary artifact.',
                direction: 'Lower is usually better for efficiency when matched quality is preserved.',
                unit: 'evidence budget units',
              })}
            </div>
            <div className="baselineImpactGrid__value">{n(locale, voiStopSummary.evidence_budget_used)}</div>
          </div>
          <div>
            <div className="baselineImpactGrid__label">
              {metricLabel('Stop reason', {
                definition: 'Compact stop-summary reason for why the controller terminated.',
                direction: 'Context only; this explains the stop decision rather than ranking the run.',
                unit: 'categorical stop reason',
              })}
            </div>
            <div className="baselineImpactGrid__value">{voiStopSummary.stop_reason}</div>
          </div>
        </div>
      ) : null}

      {activeFamilies.length ? (
        <ul className="baselineNotes">
          <li>Active families: {activeFamilies.join(', ')}</li>
        </ul>
      ) : null}

      {(decisionPackageHref ||
        certificateSummaryHref ||
        certificateWitnessHref ||
        manifestHref ||
        fragilityHref ||
        valueOfRefreshHref ||
        voiStopHref ||
        worldSupportHref) ? (
        <div className="actionGrid u-mt10">
          {decisionPackageHref ? (
            <a className="secondary" href={decisionPackageHref} target="_blank" rel="noreferrer">
              Decision package
            </a>
          ) : null}
          {certificateSummaryHref ? (
            <a className="secondary" href={certificateSummaryHref} target="_blank" rel="noreferrer">
              Certificate summary
            </a>
          ) : null}
          {certificateWitnessHref ? (
            <a className="secondary" href={certificateWitnessHref} target="_blank" rel="noreferrer">
              Certificate witness
            </a>
          ) : null}
          {manifestHref ? (
            <a className="secondary" href={manifestHref} target="_blank" rel="noreferrer">
              Sampled world manifest
            </a>
          ) : null}
          {fragilityHref ? (
            <a className="secondary" href={fragilityHref} target="_blank" rel="noreferrer">
              Fragility map
            </a>
          ) : null}
          {valueOfRefreshHref ? (
            <a className="secondary" href={valueOfRefreshHref} target="_blank" rel="noreferrer">
              Value of refresh
            </a>
          ) : null}
          {voiStopHref ? (
            <a className="secondary" href={voiStopHref} target="_blank" rel="noreferrer">
              Stop certificate
            </a>
          ) : null}
          {worldSupportHref ? (
            <a className="secondary" href={worldSupportHref} target="_blank" rel="noreferrer">
              World support summary
            </a>
          ) : null}
        </div>
      ) : null}

      {runId && onOpenRunInspector ? (
        <div className="actionGrid u-mt10">
          <button type="button" className="secondary" onClick={() => onOpenRunInspector(runId)}>
            Open Run Inspector
          </button>
        </div>
      ) : null}
    </section>
  );
}
