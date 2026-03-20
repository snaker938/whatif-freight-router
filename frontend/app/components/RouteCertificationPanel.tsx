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
  selectedCertificate?: RouteCertificationSummary | null;
  voiStopSummary?: VoiStopSummary | null;
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

export default function RouteCertificationPanel({
  locale,
  route,
  runId,
  pipelineMode = 'legacy',
  selectedCertificate,
  voiStopSummary,
  onOpenRunInspector,
}: Props) {
  if (!route) return null;

  const certification = selectedCertificate ?? route.certification ?? null;
  const activeFamilies = certification?.active_families ?? route.evidence_provenance?.active_families ?? [];

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
