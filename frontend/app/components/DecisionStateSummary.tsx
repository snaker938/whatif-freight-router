'use client';

import type { RouteOption } from '../lib/types';

type SummaryRecord = Record<string, unknown> | null | undefined;

type Props = {
  locale: string;
  route: RouteOption | null;
  terminalType?: string | null;
  selectedCertificateBasis?: string | null;
  routeManifestEndpoint?: string | null;
  routeArtifactsEndpoint?: string | null;
  routeProvenanceEndpoint?: string | null;
  certifiedSet?: RouteOption[] | null;
  certifiedSetSummary?: SummaryRecord;
  preferenceState?: SummaryRecord;
  preferenceQueryTrace?: SummaryRecord;
  supportSummary?: SummaryRecord;
  preferenceSummary?: SummaryRecord;
  abstentionSummary?: SummaryRecord;
  artifactPointers?: Record<string, string | null> | null;
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

function text(value: unknown): string {
  if (typeof value !== 'string') return 'n/a';
  const trimmed = value.trim();
  return trimmed || 'n/a';
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

function nestedRecord(record: SummaryRecord, key: string): SummaryRecord {
  const value = recordValue(record, key);
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as SummaryRecord) : null;
}

function artifactEntries(artifactPointers: Record<string, string | null> | null | undefined): Array<[string, string]> {
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

export default function DecisionStateSummary({
  locale,
  route,
  terminalType,
  selectedCertificateBasis,
  routeManifestEndpoint,
  routeArtifactsEndpoint,
  routeProvenanceEndpoint,
  certifiedSet,
  certifiedSetSummary,
  preferenceState,
  preferenceQueryTrace,
  supportSummary,
  preferenceSummary,
  abstentionSummary,
  artifactPointers,
}: Props) {
  const artifactList = artifactEntries(artifactPointers);
  const visible =
    Boolean(route) ||
    Boolean(terminalType) ||
    Boolean(selectedCertificateBasis) ||
    Boolean(certifiedSet?.length) ||
    Boolean(certifiedSetSummary) ||
    Boolean(preferenceState) ||
    Boolean(preferenceQueryTrace) ||
    Boolean(abstentionSummary) ||
    Boolean(supportSummary) ||
    Boolean(preferenceSummary) ||
    artifactList.length > 0;

  if (!visible) return null;

  const terminalLabel = terminalType ?? (route ? 'certified_singleton' : 'typed_abstention');
  const supportFlag = recordValue(supportSummary, 'support_flag');
  const certifiedSetMembers = stringList(certifiedSetSummary, 'member_route_ids');
  const certifiedSetExcluded = stringList(certifiedSetSummary, 'excluded_route_ids');
  const preferenceCompatibleSummary = nestedRecord(preferenceState, 'compatible_set_summary');
  const preferenceTraceTerminal = text(recordValue(preferenceQueryTrace, 'terminal_type'));
  const preferenceTraceSchema = text(recordValue(preferenceQueryTrace, 'schema_version'));
  const certifiedSetRouteIds =
    certifiedSetMembers.length > 0
      ? certifiedSetMembers
      : (certifiedSet ?? []).map((option) => option.id).filter(Boolean);
  const certifiedSetActive =
    terminalLabel === 'certified_set' || certifiedSetRouteIds.length > 1 || Boolean(certifiedSetSummary);

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
          <>
            No route selected. Backend returned a terminal abstention state.
          </>
        )}
      </div>

      <div className="baselineKpiGrid">
        <div className="baselineKpi">
          <div className="baselineKpi__label">Selected Certificate Basis</div>
          <div className="baselineKpi__value">
            {selectedCertificateBasis ?? text(recordValue(preferenceSummary, 'selected_certificate_basis'))}
          </div>
          <div className="baselineKpi__meta">
            Pipeline {text(recordValue(preferenceSummary, 'pipeline_mode'))}
          </div>
        </div>
        <div className={`baselineKpi ${supportFlag === false ? 'isNegative' : 'isPositive'}`}>
          <div className="baselineKpi__label">Support</div>
          <div className="baselineKpi__value">{supportFlag === undefined ? 'n/a' : supportFlag ? 'supported' : 'unsupported'}</div>
          <div className="baselineKpi__meta">
            {text(recordValue(supportSummary, 'support_reason'))}
          </div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">Worlds</div>
          <div className="baselineKpi__value">{n(locale, recordValue(supportSummary, 'world_count'))}</div>
          <div className="baselineKpi__meta">
            Unique {n(locale, recordValue(supportSummary, 'unique_world_count'))}
          </div>
        </div>
        <div className="baselineKpi">
          <div className="baselineKpi__label">World Reuse</div>
          <div className="baselineKpi__value">{pct(locale, recordValue(supportSummary, 'world_reuse_rate'))}</div>
          <div className="baselineKpi__meta">
            Reuse rate
          </div>
        </div>
      </div>

      {preferenceState || preferenceQueryTrace ? (
        <div className="baselineComparePanel__tradeoff">
          Preference runtime:
          <span style={{ marginLeft: 6 }}>
            terminal {text(recordValue(preferenceState, 'terminal_type'))}
          </span>
          <span style={{ marginLeft: 6 }}>
            queries {n(locale, recordValue(preferenceState, 'query_count'))}
          </span>
          <span style={{ marginLeft: 6 }}>
            set {n(locale, recordValue(preferenceCompatibleSummary, 'compatible_set_size'))}
            {recordValue(preferenceCompatibleSummary, 'support_flag') === undefined
              ? ''
              : ` (${recordValue(preferenceCompatibleSummary, 'support_flag') ? 'supported' : 'unsupported'})`}
          </span>
          <span style={{ marginLeft: 6 }}>
            trace {preferenceTraceSchema}
            {preferenceTraceTerminal ? ` / ${preferenceTraceTerminal}` : ''}
          </span>
        </div>
      ) : null}

      {certifiedSetActive ? (
        <div className="baselineComparePanel__tradeoff">
          Certified set: {certifiedSetRouteIds.length || n(locale, recordValue(certifiedSetSummary, 'member_count'))}
          {recordValue(certifiedSetSummary, 'selected_route_id')
            ? `; selected route ${text(recordValue(certifiedSetSummary, 'selected_route_id'))}`
            : route?.id
              ? `; selected route ${route.id}`
              : ''}
          {recordValue(certifiedSetSummary, 'exclusion_basis')
            ? `; exclusion basis ${text(recordValue(certifiedSetSummary, 'exclusion_basis'))}`
            : ''}
        </div>
      ) : null}

      {certifiedSetActive ? (
        <ul className="baselineNotes">
          {certifiedSetRouteIds.length ? <li>Members: {certifiedSetRouteIds.join(', ')}</li> : null}
          {certifiedSetExcluded.length ? <li>Excluded: {certifiedSetExcluded.join(', ')}</li> : null}
          {recordValue(certifiedSetSummary, 'frontier_count') !== undefined ? (
            <li>Frontier count: {n(locale, recordValue(certifiedSetSummary, 'frontier_count'))}</li>
          ) : null}
        </ul>
      ) : null}

      {abstentionSummary ? (
        <div className="baselineComparePanel__tradeoff">
          Abstention reason: {text(recordValue(abstentionSummary, 'reason_code'))}
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
