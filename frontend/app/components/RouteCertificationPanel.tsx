'use client';

import type {
  PipelineMode,
  RouteCertificationSummary,
  RouteOption,
  VoiStopSummary,
} from '../lib/types';

type Props = {
  locale: string;
  route: RouteOption | null;
  runId?: string | null;
  pipelineMode?: PipelineMode;
  selectedCertificateBasis?: string | null;
  selectedCertificate?: RouteCertificationSummary | null;
  voiStopSummary?: VoiStopSummary | null;
  actionTraceSummary?: Record<string, unknown> | null;
  witnessSummary?: Record<string, unknown> | null;
  worldSupportSummary?: Record<string, unknown> | null;
  onOpenRunInspector?: (runId: string) => void;
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

function nestedRecord(record: Record<string, unknown> | null | undefined, key: string): Record<string, unknown> | null {
  const value = recordValue(record, key);
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

export default function RouteCertificationPanel({
  locale,
  route,
  runId,
  pipelineMode = 'legacy',
  selectedCertificateBasis,
  selectedCertificate,
  voiStopSummary,
  actionTraceSummary,
  witnessSummary,
  worldSupportSummary,
  onOpenRunInspector,
}: Props) {
  if (!route) return null;

  const certification = selectedCertificate ?? route.certification ?? null;
  const activeFamilies = certification?.active_families ?? route.evidence_provenance?.active_families ?? [];
  const supportState = nestedRecord(worldSupportSummary, 'support_state');
  const supportCalibrationBin =
    text(recordValue(worldSupportSummary, 'calibration_bin')) !== 'n/a'
      ? text(recordValue(worldSupportSummary, 'calibration_bin'))
      : text(recordValue(supportState, 'calibration_bin')) !== 'n/a'
        ? text(recordValue(supportState, 'calibration_bin'))
        : text(recordValue(supportState, 'support_bin')) !== 'n/a'
          ? text(recordValue(supportState, 'support_bin'))
          : 'n/a';

  return (
    <section className="baselineComparePanel">
      <div className="baselineComparePanel__head">
        <div className="baselineComparePanel__title">VOI / Certification</div>
        <div className={`baselineEpicScore baselineEpicScore--${certification?.certified ? 'high' : 'mixed'}`}>
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
        {recordValue(witnessSummary, 'witness_size') !== undefined
          ? `; witness size ${n(locale, numberOrNull(recordValue(witnessSummary, 'witness_size')))}`
          : ''}
        {stringList(witnessSummary, 'active_challenger_ids').length
          ? `; challengers ${stringList(witnessSummary, 'active_challenger_ids').join(', ')}`
          : ''}
        {stringList(witnessSummary, 'active_evidence_families').length
          ? `; evidence ${stringList(witnessSummary, 'active_evidence_families').join(', ')}`
          : ''}
        {recordValue(actionTraceSummary, 'stop_reason') ? `; stop ${text(recordValue(actionTraceSummary, 'stop_reason'))}` : ''}
        {recordValue(actionTraceSummary, 'search_completeness_score') !== undefined
          ? `; search ${n(locale, numberOrNull(recordValue(actionTraceSummary, 'search_completeness_score')))}`
          : ''}
        {recordValue(actionTraceSummary, 'search_completeness_gap') !== undefined
          ? `; gap ${n(locale, numberOrNull(recordValue(actionTraceSummary, 'search_completeness_gap')))}`
          : ''}
      </div>

      {worldSupportSummary ? (
        <div className="baselineComparePanel__tradeoff">
          Support governance:
          {recordValue(worldSupportSummary, 'support_flag') !== undefined
            ? ` flag ${recordValue(worldSupportSummary, 'support_flag') ? 'supported' : 'unsupported'}`
            : ''}
          {recordValue(worldSupportSummary, 'support_reason') ? `; reason ${text(recordValue(worldSupportSummary, 'support_reason'))}` : ''}
          {supportCalibrationBin !== 'n/a' ? `; calibration ${supportCalibrationBin}` : ''}
          {recordValue(worldSupportSummary, 'world_count') !== undefined
            ? `; worlds ${n(locale, numberOrNull(recordValue(worldSupportSummary, 'world_count')))}`
            : ''}
          {recordValue(worldSupportSummary, 'unique_world_count') !== undefined
            ? `; unique ${n(locale, numberOrNull(recordValue(worldSupportSummary, 'unique_world_count')))}`
            : ''}
          {recordValue(worldSupportSummary, 'world_reuse_rate') !== undefined
            ? `; reuse ${pct(locale, numberOrNull(recordValue(worldSupportSummary, 'world_reuse_rate')))}`
            : ''}
        </div>
      ) : null}

      {certification ? (
        <>
          <div className="baselineKpiGrid">
            <div className={`baselineKpi ${certification.certified ? 'isPositive' : 'isNegative'}`}>
              <div className="baselineKpi__label">Certificate</div>
              <div className="baselineKpi__value">{pct(locale, certification.certificate)}</div>
              <div className="baselineKpi__meta">
                Threshold {pct(locale, certification.threshold)} ({certification.certified ? 'Certified' : 'Uncertified'})
              </div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">Top Refresh Family</div>
              <div className="baselineKpi__value">{certification.top_value_of_refresh_family ?? 'n/a'}</div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">Main Competitor</div>
              <div className="baselineKpi__value">{certification.top_competitor_route_id ?? 'n/a'}</div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">Active Evidence Families</div>
              <div className="baselineKpi__value">{activeFamilies.length}</div>
            </div>
          </div>
          {certification.top_fragility_families?.length ? (
            <div className="baselineComparePanel__tradeoff">
              Fragility drivers: {certification.top_fragility_families.join(', ')}.
            </div>
          ) : null}
        </>
      ) : (
        <div className="baselineComparePanel__loading">
          No certification summary was returned for this route. Legacy mode keeps the run artifacts but does not certify the winner.
        </div>
      )}

      {voiStopSummary ? (
        <div className="baselineImpactGrid">
          <div>
            <div className="baselineImpactGrid__label">Iterations</div>
            <div className="baselineImpactGrid__value">{n(locale, voiStopSummary.iteration_count)}</div>
          </div>
          <div>
            <div className="baselineImpactGrid__label">Search budget used</div>
            <div className="baselineImpactGrid__value">{n(locale, voiStopSummary.search_budget_used)}</div>
          </div>
          <div>
            <div className="baselineImpactGrid__label">Evidence budget used</div>
            <div className="baselineImpactGrid__value">{n(locale, voiStopSummary.evidence_budget_used)}</div>
          </div>
          <div>
            <div className="baselineImpactGrid__label">Stop reason</div>
            <div className="baselineImpactGrid__value">{voiStopSummary.stop_reason}</div>
          </div>
        </div>
      ) : null}

      {activeFamilies.length ? (
        <ul className="baselineNotes">
          <li>Active families: {activeFamilies.join(', ')}</li>
        </ul>
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
