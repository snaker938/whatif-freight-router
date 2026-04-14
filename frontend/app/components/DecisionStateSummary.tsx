'use client';

import type { CSSProperties } from 'react';

import FieldInfo from './FieldInfo';
import { formatMetricTooltip, type MetricTooltip } from './metricTooltip';
import type {
  DecisionProofContext,
  PreferenceCompatibleSetSummary,
  PreferenceQuery,
  PreferenceQueryTrace,
  PreferenceState,
  PreferenceSummary,
  RouteOption,
  WitnessSummary,
} from '../lib/types';

type SummaryRecord = Record<string, unknown> | null | undefined;

type Props = {
  locale: string;
  route: RouteOption | null;
  terminalType?: string | null;
  selectedCertificateBasis?: string | null;
  proofContext?: DecisionProofContext | null;
  routeManifestEndpoint?: string | null;
  routeArtifactsEndpoint?: string | null;
  routeProvenanceEndpoint?: string | null;
  certifiedSet?: RouteOption[] | null;
  certifiedSetSummary?: SummaryRecord;
  preferenceState?: PreferenceState | null;
  preferenceQueryTrace?: PreferenceQueryTrace | null;
  supportSummary?: SummaryRecord;
  worldSupportSummary?: SummaryRecord;
  preferenceSummary?: PreferenceSummary | null;
  abstentionSummary?: SummaryRecord;
  artifactPointers?: Record<string, string | null> | null;
  witnessSummary?: WitnessSummary | null;
};

const inlineMetricLabelStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
};

function n(locale: string, value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
  return new Intl.NumberFormat(locale, { maximumFractionDigits: 2 }).format(value);
}

function pct(locale: string, value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    maximumFractionDigits: 1,
    minimumFractionDigits: 0,
  }).format(value);
}

function numericOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
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

function toRecord(value: unknown): SummaryRecord {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function recordValue(record: SummaryRecord, key: string): unknown {
  if (!record || !(key in record)) return undefined;
  return record[key];
}

function stringList(record: SummaryRecord, key: string): string[] {
  const value = recordValue(record, key);
  if (!Array.isArray(value)) return [];
  return value
    .map((entry) => (typeof entry === 'string' ? entry.trim() : ''))
    .filter(Boolean);
}

function artifactEntries(
  artifactPointers: Record<string, string | null> | null | undefined,
): Array<[string, string]> {
  if (!artifactPointers) return [];
  return Object.entries(artifactPointers)
    .map(([key, value]) => [key, typeof value === 'string' ? value.trim() : ''] as [string, string])
    .filter(([, value]) => Boolean(value));
}

function labelizeArtifactKey(key: string): string {
  return key
    .split('_')
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(' ');
}

function artifactHref(base: string | null | undefined, artifactName: string): string | null {
  if (typeof base !== 'string') return null;
  const trimmed = base.trim();
  if (!trimmed) return null;
  return `${trimmed.replace(/\/$/, '')}/${encodeURIComponent(artifactName)}`;
}

function humanizeCode(value: string | null | undefined): string {
  if (!value) return 'n/a';
  return value
    .split(/[_-]+/)
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(' ');
}

function firstDefined<T>(...values: Array<T | null | undefined>): T | null {
  for (const value of values) {
    if (value !== undefined && value !== null) return value;
  }
  return null;
}

function supportStatusLabel(value: unknown): string {
  if (value === true) return 'supported';
  if (value === false) return 'unsupported';
  return 'n/a';
}

function normalizePreferenceState(
  preferenceState: PreferenceState | null | undefined,
  preferenceSummary: PreferenceSummary | null | undefined,
): PreferenceState | null {
  return preferenceState ?? preferenceSummary?.preference_state ?? null;
}

function normalizeCompatibleSetSummary(
  preferenceState: PreferenceState | null,
  preferenceTrace: PreferenceQueryTrace | null | undefined,
  preferenceSummary: PreferenceSummary | null | undefined,
): PreferenceCompatibleSetSummary | null {
  return (
    preferenceSummary?.compatible_set_summary ??
    preferenceTrace?.compatible_set_summary ??
    preferenceState?.compatible_set_summary ??
    null
  );
}

function formatPreferenceQuery(locale: string, query: PreferenceQuery): string {
  switch (query.query_type) {
    case 'pairwise':
      return `${query.preferred_route_id} over ${query.challenger_route_id}${
        query.reason ? `: ${query.reason}` : ''
      }`;
    case 'threshold':
      return `${query.route_id} ${query.metric_name} ${
        query.direction === 'gte' ? '>=' : '<='
      } ${n(locale, query.threshold_value)}${query.reason ? `: ${query.reason}` : ''}`;
    case 'ratio':
      return `${query.route_id} ${query.numerator_metric}/${query.denominator_metric} >= ${n(
        locale,
        query.minimum_ratio,
      )}${query.reason ? `: ${query.reason}` : ''}`;
    case 'veto':
      return `${query.route_id} ${query.active === false ? 'removes' : 'adds'} veto ${query.veto_name}${
        query.reason ? `: ${query.reason}` : ''
      }`;
    case 'time_guard':
      return `${query.route_id} preserves time budget${
        query.max_travel_time_s != null ? ` <= ${n(locale, query.max_travel_time_s)} s` : ''
      }${
        query.preserve_time_budget_s != null ? ` / guard ${n(locale, query.preserve_time_budget_s)} s` : ''
      }${query.reason ? `: ${query.reason}` : ''}`;
    default:
      return humanizeCode((query as { query_type?: string }).query_type ?? null);
  }
}

function preferenceNoQueryReason(reason: string | null | undefined): string | null {
  switch (reason) {
    case 'preference_contradiction_detected':
      return 'No preference query was asked because the current answers are contradictory.';
    case 'preference_support_insufficient':
      return 'No preference query was asked because support is too weak for a certificate-improving query.';
    case 'preference_irrelevance_proven':
      return 'No preference query was asked because preference irrelevance was already proven.';
    case 'singleton_frontier':
      return 'No preference query was asked because the surviving frontier had already collapsed to a singleton.';
    case 'no_preference_query_issued':
      return 'No preference query was asked because the controller did not identify a certificate-improving preference action.';
    default:
      return reason ? humanizeCode(reason) : null;
  }
}

function activeInvariantLabels(derivedInvariants: Record<string, boolean> | null | undefined): string[] {
  if (!derivedInvariants) return [];
  return Object.entries(derivedInvariants)
    .filter(([, active]) => active)
    .map(([name]) => humanizeCode(name))
    .filter((value) => value !== 'N/a');
}

function queryBucket(queries: PreferenceQuery[], type: PreferenceQuery['query_type']): PreferenceQuery[] {
  return queries.filter((query) => query.query_type === type);
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

export default function DecisionStateSummary({
  locale,
  route,
  terminalType,
  selectedCertificateBasis,
  proofContext,
  routeManifestEndpoint,
  routeArtifactsEndpoint,
  routeProvenanceEndpoint,
  certifiedSet,
  certifiedSetSummary,
  preferenceState,
  preferenceQueryTrace,
  supportSummary,
  worldSupportSummary,
  preferenceSummary,
  abstentionSummary,
  artifactPointers,
  witnessSummary,
}: Props) {
  const artifactList = artifactEntries(artifactPointers);
  const resolvedPreferenceState = normalizePreferenceState(preferenceState, preferenceSummary);
  const resolvedPreferenceTrace = preferenceQueryTrace ?? null;
  const compatibleSetSummary = normalizeCompatibleSetSummary(
    resolvedPreferenceState,
    resolvedPreferenceTrace,
    preferenceSummary,
  );
  const queryHistory = resolvedPreferenceTrace?.query_history ?? resolvedPreferenceState?.query_history ?? [];
  const shrinkageTrace = resolvedPreferenceTrace?.shrinkage_trace ?? resolvedPreferenceState?.shrinkage_trace ?? [];
  const queryCount =
    resolvedPreferenceTrace?.query_count ??
    resolvedPreferenceState?.query_count ??
    preferenceSummary?.query_count ??
    queryHistory.length;
  const latestQuery = queryHistory.length ? queryHistory[queryHistory.length - 1] : null;
  const latestShrinkage = shrinkageTrace.length ? shrinkageTrace[shrinkageTrace.length - 1] : null;
  const whyThisQuery =
    latestShrinkage?.query_reason ??
    textOrNull((latestQuery as { reason?: string | null } | null)?.reason) ??
    (latestQuery ? `Latest query type: ${humanizeCode(latestQuery.query_type)}` : null);
  const preferenceIrrelevanceProven =
    resolvedPreferenceState?.preference_irrelevance_proven ??
    resolvedPreferenceTrace?.preference_irrelevance_proven ??
    preferenceSummary?.preference_irrelevance_proven ??
    false;
  const rawNoPreferenceQueryReason =
    resolvedPreferenceState?.no_preference_query_reason ??
    resolvedPreferenceState?.no_query_reason ??
    resolvedPreferenceTrace?.no_preference_query_reason ??
    resolvedPreferenceTrace?.no_query_reason ??
    preferenceSummary?.no_preference_query_reason ??
    preferenceSummary?.no_query_reason ??
    null;
  const noQueryReason =
    (queryCount ?? 0) > 0 ? null : preferenceNoQueryReason(rawNoPreferenceQueryReason);
  const contradictionDetected = Boolean(
    resolvedPreferenceTrace?.contradiction_record?.contradiction_detected ??
      resolvedPreferenceState?.contradiction_record?.contradiction_detected ??
      preferenceSummary?.contradiction_record?.contradiction_detected,
  );
  const activeInvariants = activeInvariantLabels(
    resolvedPreferenceTrace?.derived_invariants ?? resolvedPreferenceState?.derived_invariants ?? null,
  );

  const pairwiseQueries = queryBucket(
    queryHistory.length
      ? queryHistory
      : ((resolvedPreferenceState?.pairwise_constraints as PreferenceQuery[] | undefined) ?? []),
    'pairwise',
  );
  const thresholdQueries = queryBucket(
    queryHistory.length
      ? queryHistory
      : ((resolvedPreferenceState?.threshold_constraints as PreferenceQuery[] | undefined) ?? []),
    'threshold',
  );
  const ratioQueries = queryBucket(
    queryHistory.length
      ? queryHistory
      : ((resolvedPreferenceState?.ratio_constraints as PreferenceQuery[] | undefined) ?? []),
    'ratio',
  );
  const vetoQueries = queryBucket(
    queryHistory.length
      ? queryHistory
      : ((resolvedPreferenceState?.veto_rules as PreferenceQuery[] | undefined) ?? []),
    'veto',
  );
  const timeGuardQueries = queryBucket(
    queryHistory.length
      ? queryHistory
      : ((resolvedPreferenceState?.time_preserving_guard_rules as PreferenceQuery[] | undefined) ?? []),
    'time_guard',
  );

  const supportRecord = firstDefined(
    supportSummary,
    toRecord(worldSupportSummary?.support_state),
    worldSupportSummary,
  );
  const proofSelectedCertificateBasis =
    proofContext?.selected_certificate_basis ?? selectedCertificateBasis ?? preferenceSummary?.selected_certificate_basis ?? null;
  const supportFlag = firstDefined(
    proofContext?.support_flag,
    recordValue(supportSummary, 'support_flag') as boolean | undefined,
    recordValue(toRecord(worldSupportSummary?.support_state), 'support_flag') as boolean | undefined,
    recordValue(worldSupportSummary, 'support_flag') as boolean | undefined,
  );
  const supportReason = firstDefined(
    textOrNull(proofContext?.out_of_support_reason),
    textOrNull(recordValue(supportSummary, 'support_reason')),
    textOrNull(recordValue(toRecord(worldSupportSummary?.support_state), 'support_reason')),
    textOrNull(recordValue(worldSupportSummary, 'support_reason')),
  );
  const certifiedSetMembers = stringList(certifiedSetSummary, 'member_route_ids');
  const certifiedSetExcluded = stringList(certifiedSetSummary, 'excluded_route_ids');
  const certifiedSetRouteIds =
    certifiedSetMembers.length > 0
      ? certifiedSetMembers
      : (certifiedSet ?? []).map((option) => option.id).filter(Boolean);
  const certifiedSetSummarySize = firstDefined(
    numericOrNull(recordValue(certifiedSetSummary, 'member_count')),
    numericOrNull(recordValue(certifiedSetSummary, 'set_size')),
  );
  const certifiedSetActive =
    terminalType === 'certified_set' ||
    certifiedSetRouteIds.length > 1 ||
    (certifiedSetSummarySize ?? 0) > 1;
  const preferenceVisible =
    Boolean(resolvedPreferenceState) ||
    Boolean(resolvedPreferenceTrace) ||
    Boolean(preferenceSummary) ||
    queryHistory.length > 0 ||
    shrinkageTrace.length > 0 ||
    Boolean(noQueryReason);

  const visible =
    Boolean(route) ||
    Boolean(terminalType) ||
    Boolean(proofSelectedCertificateBasis) ||
    Boolean(proofContext) ||
    Boolean(certifiedSet?.length) ||
    Boolean(certifiedSetSummary) ||
    Boolean(preferenceVisible) ||
    Boolean(abstentionSummary) ||
    Boolean(supportSummary) ||
    Boolean(worldSupportSummary) ||
    Boolean(witnessSummary) ||
    artifactList.length > 0;

  if (!visible) return null;

  const terminalLabel = terminalType ?? (route ? 'certified_singleton' : 'typed_abstention');
  const preferenceOutcomeLabel =
    terminalLabel === 'typed_abstention'
      ? 'abstained'
      : terminalLabel === 'certified_set' || terminalLabel === 'certified_singleton'
        ? 'certified'
        : 'n/a';
  const preferenceBurdenCertificateBasis = proofSelectedCertificateBasis;
  const preferenceBurdenCountLabel = `${n(locale, queryCount)} quer${
    queryCount === 1 ? 'y' : 'ies'
  }`;
  const preferenceBurdenSummary = [
    `Outcome ${preferenceOutcomeLabel}`,
    `Burden ${preferenceBurdenCountLabel}`,
    preferenceBurdenCertificateBasis ? `Basis ${preferenceBurdenCertificateBasis}` : null,
    preferenceIrrelevanceProven ? 'Irrelevance proven' : null,
    noQueryReason ? `No-query ${noQueryReason}` : null,
  ]
    .filter(Boolean)
    .join(' | ');
  const witnessTargetRouteId = route?.id ?? witnessSummary?.route_id ?? null;
  const witnessCertificateBasis =
    proofSelectedCertificateBasis ?? witnessSummary?.selected_certificate_basis ?? null;
  const typedAbstentionReasonCode = textOrNull(proofContext?.typed_abstention?.reason_code);
  const controllerBoundaryChallenger = textOrNull(proofContext?.controller_boundary_summary?.active_challenger_id);
  const explanationSources = [
    {
      label: 'Decision package',
      href: (() => {
        const artifactName = textOrNull(artifactPointers?.decision_package);
        return artifactName ? artifactHref(routeArtifactsEndpoint, artifactName) : null;
      })(),
    },
    {
      label: 'World support summary',
      href: (() => {
        const artifactName = textOrNull(artifactPointers?.world_support_summary);
        return artifactName ? artifactHref(routeArtifactsEndpoint, artifactName) : null;
      })(),
    },
    {
      label: 'Certificate witness',
      href: (() => {
        const artifactName = textOrNull(artifactPointers?.certificate_witness);
        return artifactName ? artifactHref(routeArtifactsEndpoint, artifactName) : null;
      })(),
    },
    {
      label: 'VOI stop certificate',
      href: (() => {
        const artifactName = textOrNull(artifactPointers?.voi_stop_certificate);
        return artifactName ? artifactHref(routeArtifactsEndpoint, artifactName) : null;
      })(),
    },
    {
      label: 'VOI controller state',
      href: (() => {
        const artifactName = textOrNull(artifactPointers?.voi_controller_state);
        return artifactName ? artifactHref(routeArtifactsEndpoint, artifactName) : null;
      })(),
    },
    { label: 'Manifest', href: textOrNull(routeManifestEndpoint) },
    { label: 'Provenance', href: textOrNull(routeProvenanceEndpoint) },
  ];
  const witnessExplanation = [
    terminalType || witnessTargetRouteId
      ? `Terminal outcome: ${terminalLabel}${
          witnessTargetRouteId ? ` for ${witnessTargetRouteId}` : ''
        }.`
      : null,
    witnessCertificateBasis ? `Certificate basis: ${witnessCertificateBasis}.` : null,
    supportFlag !== null && supportFlag !== undefined
      ? `Support status: ${supportStatusLabel(supportFlag)}${
          supportReason ? ` (${supportReason})` : ''
        }.`
      : supportReason
        ? `Support note: ${supportReason}.`
        : null,
    terminalLabel === 'typed_abstention' && typedAbstentionReasonCode
      ? `Typed abstention provenance: ${typedAbstentionReasonCode}.`
      : null,
    controllerBoundaryChallenger
      ? `Controller boundary challenger: ${controllerBoundaryChallenger}.`
      : null,
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

  return (
    <section className="baselineComparePanel">
      <div className="baselineComparePanel__head">
        <div className="baselineComparePanel__title">Decision State</div>
        <div className={`baselineEpicScore baselineEpicScore--${terminalLabel === 'typed_abstention' ? 'mixed' : 'high'}`}>
          {terminalLabel}
        </div>
      </div>

      <div className="baselineComparePanel__epicNote">
        {route ? (
          <>
            Route selected: <strong>{route.id}</strong>
          </>
        ) : (
          <>No certified route returned. Backend ended in a non-singleton or typed-abstention terminal state.</>
        )}
      </div>

      <div className="baselineKpiGrid">
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Selected Certificate Basis', {
              definition: 'Certificate basis currently selected for the live decision package.',
              direction: 'Context only; this names the active certification basis rather than scoring quality.',
              unit: 'categorical certificate-basis label',
            })}
          </div>
          <div className="baselineKpi__value">
            {proofSelectedCertificateBasis ?? 'n/a'}
          </div>
          <div className="baselineKpi__meta">
            Pipeline {preferenceSummary?.pipeline_mode ?? 'n/a'}
          </div>
        </div>
        <div className={`baselineKpi ${supportFlag === false ? 'isNegative' : 'isPositive'}`}>
          <div className="baselineKpi__label">
            {metricLabel('Support', {
              definition: 'Whether the current decision remains inside the world-model support regime.',
              direction: 'In-support is better for trustworthy certification; unsupported means weaker certification claims.',
              unit: 'categorical support status',
            })}
          </div>
          <div className="baselineKpi__value">{supportStatusLabel(supportFlag)}</div>
          <div className="baselineKpi__meta">{supportReason ?? 'n/a'}</div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('Worlds', {
              definition: 'Count of worlds represented by the current support summary or world bundle.',
              direction: 'Context only; larger world counts do not automatically mean a better decision.',
              unit: 'world count',
            })}
          </div>
          <div className="baselineKpi__value">{n(locale, recordValue(supportRecord, 'world_count'))}</div>
          <div className="baselineKpi__meta">
            {inlineMetricLabel('Unique', {
              definition: 'Number of distinct worlds after reuse and duplicate-collapse effects.',
              direction: 'Context only; higher unique counts indicate broader distinct world coverage, not automatic superiority.',
              unit: 'unique world count',
            })}{' '}
            {n(locale, recordValue(supportRecord, 'unique_world_count'))}
          </div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">
            {metricLabel('World Reuse', {
              definition: 'Share of the world bundle reused from prior computation instead of regenerated for this run.',
              direction: 'Context only; higher reuse means a hotter rerun, not necessarily a better routing outcome.',
              unit: 'reuse fraction',
            })}
          </div>
          <div className="baselineKpi__value">{pct(locale, recordValue(supportRecord, 'world_reuse_rate'))}</div>
          <div className="baselineKpi__meta">Reuse rate</div>
        </div>
      </div>

      <div className="baselineComparePanel__artifactSection">
        <div className="baselineComparePanel__tradeoff">
          Witness Explanation
          <FieldInfo text="Deterministic explanation assembled from emitted terminal, witness, certificate-basis, and support fields already present in this decision payload." />
        </div>
      <div className="baselineComparePanel__epicNote">
        {witnessExplanation || 'No witness-driven explanation was available for the current decision payload.'}
      </div>
      </div>

      <div className="baselineComparePanel__artifactSection">
        <div className="baselineComparePanel__tradeoff">
          Explanation Sources
          <FieldInfo text="Source-oriented links behind the witness explanation. Unavailable entries were not emitted in the current decision payload." />
        </div>
        <ul className="baselineNotes">
          {explanationSources.map((source) => (
            <li key={source.label}>
              <strong>{source.label}</strong>:{' '}
              {source.href ? <a href={source.href}>Open source</a> : 'Unavailable in current payload'}
            </li>
          ))}
        </ul>
      </div>

      {preferenceVisible ? (
        <div className="baselineComparePanel__artifactSection">
          <div className="baselineComparePanel__tradeoff">
            Preference Elicitation
            <FieldInfo text="Read-only view of the certificate-guided preference payload: compatible set summary, query evidence, shrinkage trace, and no-query justification when the controller skipped elicitation." />
          </div>

          <div className="baselineKpiGrid">
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Queries', {
                  definition: 'Preference queries recorded in the backend preference trace for this run.',
                  direction: 'Lower is usually better for user burden when certification quality is held constant.',
                  unit: 'query count',
                  note: 'This panel is read-only; it inspects the emitted trace instead of asking new questions.',
                })}
              </div>
              <div className="baselineKpi__value">{n(locale, queryCount)}</div>
              <div className="baselineKpi__meta">{preferenceBurdenSummary}</div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Compatible Set', {
                  definition: 'Routes that remain preference-compatible after the observed queries and constraints.',
                  direction: 'Lower is usually better because fewer preference-compatible survivors remain unresolved.',
                  unit: 'route count',
                })}
              </div>
              <div className="baselineKpi__value">{n(locale, compatibleSetSummary?.compatible_set_size)}</div>
              <div className="baselineKpi__meta">
                {compatibleSetSummary?.route_ids?.length
                  ? compatibleSetSummary.route_ids.join(', ')
                  : 'No compatible route list recorded'}
              </div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Volume Proxy', {
                  definition: 'Proxy for the remaining size of the compatible preference region.',
                  direction: 'Lower is better because elicited preferences have narrowed the compatible region more tightly.',
                  unit: 'compatible-set volume proxy',
                })}
              </div>
              <div className="baselineKpi__value">
                {n(locale, compatibleSetSummary?.compatible_set_volume_proxy)}
              </div>
              <div className="baselineKpi__meta">
                {inlineMetricLabel('Support', {
                  definition: 'Support status attached specifically to the compatible-set summary.',
                  direction: 'In-support is better for trustworthiness of the preference summary.',
                  unit: 'categorical support status',
                })}{' '}
                {supportStatusLabel(compatibleSetSummary?.support_flag)}
              </div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">
                {metricLabel('Necessary / Possible', {
                  definition:
                    'Necessary-best is the robust winner over the remaining compatible preference set; possible-best is the looser winner mass under that same set.',
                  direction:
                    'Higher necessary-best support and a tighter gap to possible-best indicate a more decisive preference-certificate state.',
                  unit: 'probability pair (necessary-best / possible-best)',
                })}
              </div>
              <div className="baselineKpi__value">
                {pct(locale, compatibleSetSummary?.necessary_best_prob)} /{' '}
                {pct(locale, compatibleSetSummary?.possible_best_prob)}
              </div>
              <div className="baselineKpi__meta">Necessary-best / possible-best</div>
            </div>
          </div>

          <ul className="baselineNotes">
            {preferenceSummary?.weights ? (
              <li>
                Weights snapshot: time {n(locale, preferenceSummary.weights.time)}, money{' '}
                {n(locale, preferenceSummary.weights.money)}, CO2 {n(locale, preferenceSummary.weights.co2)}
              </li>
            ) : null}
            {whyThisQuery ? <li>Why this query: {whyThisQuery}</li> : null}
            {noQueryReason ? (
              <li>
                Why no preference query was asked
                <FieldInfo
                  text={metricHelp({
                    definition: 'Controller-provided explanation for why no new preference query was issued on this run.',
                    direction: 'Context only; this explains the skipped action rather than ranking the run.',
                    unit: 'categorical controller rationale',
                  })}
                />
                :{' '}
                {noQueryReason}
              </li>
            ) : null}
                {preferenceIrrelevanceProven ? (
                  <li>Preference irrelevance proven: yes</li>
                ) : null}
            {contradictionDetected ? (
              <li>
                Contradiction detected: {resolvedPreferenceState?.contradiction_record?.contradiction_reasons?.join(', ') || 'yes'}
              </li>
            ) : null}
            {activeInvariants.length ? <li>Derived invariants: {activeInvariants.join(', ')}</li> : null}
            {textOrNull(compatibleSetSummary?.support_reason) ? (
              <li>Preference support note: {compatibleSetSummary?.support_reason}</li>
            ) : null}
          </ul>

          {shrinkageTrace.length ? (
            <>
              <div className="baselineComparePanel__tradeoff">
                Shrinkage Over Time
                <FieldInfo
                  text={metricHelp({
                    definition: 'Per-query change in compatible-set size and volume proxy over the elicitation trace.',
                    direction: 'Lower size/volume after each query is better because preference ambiguity is shrinking.',
                    unit: 'set size, volume proxy, and shrinkage fractions',
                  })}
                />
              </div>
              <ul className="baselineNotes">
                {shrinkageTrace.slice(-4).map((entry) => (
                  <li key={`${entry.query_index}-${entry.query_type}`}>
                    Q{entry.query_index} {humanizeCode(entry.query_type)}: set {n(locale, entry.before_size)} to{' '}
                    {n(locale, entry.after_size)}, volume {n(locale, entry.before_volume_proxy)} to{' '}
                    {n(locale, entry.after_volume_proxy)}, predicted {pct(locale, entry.predicted_shrinkage)}, realized{' '}
                    {pct(locale, entry.realized_shrinkage)}
                    {entry.query_reason ? `, reason ${entry.query_reason}` : ''}
                  </li>
                ))}
              </ul>
            </>
          ) : null}

          {pairwiseQueries.length ? (
            <>
              <div className="baselineComparePanel__tradeoff">Pairwise evidence</div>
              <ul className="baselineNotes">
                {pairwiseQueries.slice(-3).map((query, index) => (
                  <li key={`pairwise-${index}`}>{formatPreferenceQuery(locale, query)}</li>
                ))}
              </ul>
            </>
          ) : null}

          {thresholdQueries.length || ratioQueries.length ? (
            <>
              <div className="baselineComparePanel__tradeoff">Tradeoff evidence</div>
              <ul className="baselineNotes">
                {thresholdQueries.slice(-3).map((query, index) => (
                  <li key={`threshold-${index}`}>{formatPreferenceQuery(locale, query)}</li>
                ))}
                {ratioQueries.slice(-3).map((query, index) => (
                  <li key={`ratio-${index}`}>{formatPreferenceQuery(locale, query)}</li>
                ))}
              </ul>
            </>
          ) : null}

          {vetoQueries.length ? (
            <>
              <div className="baselineComparePanel__tradeoff">Veto evidence</div>
              <ul className="baselineNotes">
                {vetoQueries.slice(-3).map((query, index) => (
                  <li key={`veto-${index}`}>{formatPreferenceQuery(locale, query)}</li>
                ))}
              </ul>
            </>
          ) : null}

          {timeGuardQueries.length ? (
            <>
              <div className="baselineComparePanel__tradeoff">Time-guard evidence</div>
              <ul className="baselineNotes">
                {timeGuardQueries.slice(-3).map((query, index) => (
                  <li key={`time-guard-${index}`}>{formatPreferenceQuery(locale, query)}</li>
                ))}
              </ul>
            </>
          ) : null}
        </div>
      ) : null}

      {certifiedSetActive ? (
        <>
          <div className="baselineComparePanel__tradeoff">
            {inlineMetricLabel('Certified set count', {
              definition: 'Number of routes retained inside the certified set for the current decision outcome.',
              direction: 'Context only; set size reflects uncertainty and is not automatically better when larger or smaller.',
              unit: 'route count',
            })}{' '}
            {certifiedSetRouteIds.length || n(locale, recordValue(certifiedSetSummary, 'member_count'))}
            {recordValue(certifiedSetSummary, 'selected_route_id')
              ? (
                <>
                  {' | '}
                  {inlineMetricLabel('Selected route', {
                    definition: 'Route currently highlighted within the certified set summary.',
                    direction: 'Context only; this names the displayed route rather than ranking it.',
                    unit: 'route identifier',
                  })}{' '}
                  {text(recordValue(certifiedSetSummary, 'selected_route_id'))}
                </>
              )
              : route?.id
                ? (
                  <>
                    {' | '}
                    {inlineMetricLabel('Selected route', {
                      definition: 'Route currently highlighted within the certified set summary.',
                      direction: 'Context only; this names the displayed route rather than ranking it.',
                      unit: 'route identifier',
                    })}{' '}
                    {route.id}
                  </>
                )
                : ''}
            {recordValue(certifiedSetSummary, 'exclusion_basis')
              ? (
                <>
                  {' | '}
                  {inlineMetricLabel('Exclusion basis', {
                    definition: 'Rule or certificate basis used to exclude routes from the certified set.',
                    direction: 'Context only; this describes why routes were filtered out rather than scoring the chosen route.',
                    unit: 'categorical exclusion rule',
                  })}{' '}
                  {text(recordValue(certifiedSetSummary, 'exclusion_basis'))}
                </>
              )
              : ''}
          </div>
          <ul className="baselineNotes">
            {certifiedSetRouteIds.length ? (
              <li>
                {inlineMetricLabel('Members', {
                  definition: 'Route identifiers currently included in the certified set.',
                  direction: 'Context only; this enumerates set membership rather than ranking routes.',
                  unit: 'route identifier list',
                })}{' '}
                {certifiedSetRouteIds.join(', ')}
              </li>
            ) : null}
            {certifiedSetExcluded.length ? (
              <li>
                {inlineMetricLabel('Excluded', {
                  definition: 'Route identifiers explicitly excluded from the certified set.',
                  direction: 'Context only; this lists rejected routes rather than scoring the selected route.',
                  unit: 'route identifier list',
                })}{' '}
                {certifiedSetExcluded.join(', ')}
              </li>
            ) : null}
            {recordValue(certifiedSetSummary, 'frontier_count') !== undefined ? (
              <li>
                {inlineMetricLabel('Frontier count', {
                  definition: 'Number of routes on the surviving certified frontier after exclusions.',
                  direction: 'Context only; a larger frontier indicates more surviving ambiguity.',
                  unit: 'frontier route count',
                })}{' '}
                {n(locale, recordValue(certifiedSetSummary, 'frontier_count'))}
              </li>
            ) : null}
          </ul>
        </>
      ) : null}

      {abstentionSummary ? (
        <div className="baselineComparePanel__tradeoff">
          Typed abstention class: {humanizeCode(textOrNull(recordValue(abstentionSummary, 'reason_code')))}
          {recordValue(abstentionSummary, 'message') ? ` (${text(recordValue(abstentionSummary, 'message'))})` : ''}
        </div>
      ) : null}

      {artifactList.length ? (
        <div className="baselineComparePanel__artifactSection">
          <div className="baselineComparePanel__tradeoff">Proof navigation</div>
          {routeManifestEndpoint || routeArtifactsEndpoint || routeProvenanceEndpoint ? (
            <ul className="baselineNotes">
              {routeManifestEndpoint ? (
                <li>
                  <a href={routeManifestEndpoint}>Manifest</a>
                </li>
              ) : null}
              {routeArtifactsEndpoint ? (
                <li>
                  <a href={routeArtifactsEndpoint}>Artifacts index</a>
                </li>
              ) : null}
              {routeProvenanceEndpoint ? (
                <li>
                  <a href={routeProvenanceEndpoint}>Provenance</a>
                </li>
              ) : null}
            </ul>
          ) : null}
          <ul className="baselineNotes">
            {artifactList.map(([key, value]) => (
              <li key={key}>
                <strong>{labelizeArtifactKey(key)}</strong>:
                {artifactHref(routeArtifactsEndpoint, value) ? (
                  <>
                    {' '}
                    <a href={artifactHref(routeArtifactsEndpoint, value) ?? undefined}>{value}</a>
                  </>
                ) : (
                  <> {value}</>
                )}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
