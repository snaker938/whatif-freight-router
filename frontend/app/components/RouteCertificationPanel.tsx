'use client';

import type {
  DecisionPackage,
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
  decisionPackage?: DecisionPackage | null;
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

function yn(value: boolean | null | undefined, yes = 'Yes', no = 'No'): string {
  return value ? yes : no;
}

function joinValues(values: Array<string | null | undefined> | null | undefined, empty = 'n/a'): string {
  const filtered = (values ?? []).filter((value): value is string => Boolean(value));
  return filtered.length ? filtered.join(', ') : empty;
}

function describeAmbiguityContext(
  ambiguityContext: Record<string, string | number | boolean | null> | null | undefined,
): string | null {
  const entries = Object.entries(ambiguityContext ?? {}).filter(([, value]) => value !== undefined);
  if (!entries.length) return null;
  return entries.map(([key, value]) => `${key}=${String(value)}`).join(', ');
}

export default function RouteCertificationPanel({
  locale,
  route,
  runId,
  pipelineMode = 'legacy',
  decisionPackage,
  selectedCertificate,
  voiStopSummary,
  onOpenRunInspector,
}: Props) {
  const certification = selectedCertificate ?? route?.certification ?? null;
  const activeFamilies = certification?.active_families ?? route?.evidence_provenance?.active_families ?? [];
  const certifiedSetSummary = decisionPackage?.certified_set_summary ?? null;
  const supportSummary = decisionPackage?.support_summary ?? null;
  const preferenceSummary = decisionPackage?.preference_summary ?? null;
  const abstentionSummary = decisionPackage?.abstention_summary ?? null;
  const witnessSummary = decisionPackage?.witness_summary ?? null;
  const controllerSummary = decisionPackage?.controller_summary ?? null;
  const laneManifest = decisionPackage?.lane_manifest ?? null;
  const ambiguitySummary = describeAmbiguityContext(certification?.ambiguity_context);
  const resolvedPipelineMode = decisionPackage?.pipeline_mode ?? pipelineMode;
  const certifiedSetRouteIds = certifiedSetSummary?.certified_route_ids ?? [];
  const frontierRouteIds = certifiedSetSummary?.frontier_route_ids ?? [];
  const isCertified = certification?.certified ?? certifiedSetSummary?.certified ?? false;

  if (!route && !decisionPackage && !certification && !voiStopSummary) return null;

  return (
    <section className="baselineComparePanel">
      <div className="baselineComparePanel__head">
        <div className="baselineComparePanel__title">VOI / Certification</div>
        <div className={`baselineEpicScore baselineEpicScore--${isCertified ? 'high' : 'mixed'}`}>
          {resolvedPipelineMode.toUpperCase()}
        </div>
      </div>
      <div className="baselineComparePanel__epicNote">
        Pipeline mode: <strong>{resolvedPipelineMode}</strong>
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
          {ambiguitySummary ? (
            <div className="baselineComparePanel__tradeoff">Ambiguity context: {ambiguitySummary}.</div>
          ) : null}
        </>
      ) : !decisionPackage ? (
        <div className="baselineComparePanel__loading">
          No certification summary was returned for this route. Legacy mode keeps the run artifacts but does not certify the winner.
        </div>
      ) : null}

      {decisionPackage ? (
        <>
          <div className="baselineComparePanel__tradeoff">
            Decision package <code>{decisionPackage.schema_version}</code> selected route{' '}
            <code>{decisionPackage.selected_route_id ?? certifiedSetSummary?.selected_route_id ?? route?.id ?? 'n/a'}</code>.
          </div>
          <div className="baselineKpiGrid">
            <div className={`baselineKpi ${certifiedSetSummary?.certified ? 'isPositive' : 'isNegative'}`}>
              <div className="baselineKpi__label">Certified Set</div>
              <div className="baselineKpi__value">{certifiedSetRouteIds.length}</div>
              <div className="baselineKpi__meta">
                Frontier {frontierRouteIds.length} | Gate {yn(certifiedSetSummary?.selective_gate_passed, 'Passed', 'Not passed')}
              </div>
            </div>
            <div className={`baselineKpi ${supportSummary?.satisfied ? 'isPositive' : 'isNegative'}`}>
              <div className="baselineKpi__label">Support</div>
              <div className="baselineKpi__value">
                {supportSummary
                  ? `${supportSummary.observed_source_count}/${supportSummary.required_source_count}`
                  : 'n/a'}
              </div>
              <div className="baselineKpi__meta">{supportSummary?.support_mode ?? 'n/a'}</div>
            </div>
            <div className={`baselineKpi ${abstentionSummary?.abstained ? 'isNegative' : 'isPositive'}`}>
              <div className="baselineKpi__label">Abstention</div>
              <div className="baselineKpi__value">
                {abstentionSummary
                  ? abstentionSummary.abstained
                    ? abstentionSummary.reason_code ?? 'Abstained'
                    : 'Clear'
                  : 'n/a'}
              </div>
              <div className="baselineKpi__meta">
                Retryable {abstentionSummary ? yn(abstentionSummary.retryable) : 'n/a'} | Blocking{' '}
                {abstentionSummary ? abstentionSummary.blocking_sources.length : 'n/a'}
              </div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">Witnesses</div>
              <div className="baselineKpi__value">{witnessSummary?.witness_route_ids.length ?? 0}</div>
              <div className="baselineKpi__meta">
                Challengers {witnessSummary?.challenger_route_ids.length ?? 0}
              </div>
            </div>
          </div>
          <ul className="baselineNotes">
            {preferenceSummary ? (
              <li>
                Preference: objective {preferenceSummary.objective_field} via {preferenceSummary.selector_policy}; selective{' '}
                {yn(preferenceSummary.selective, 'on', 'off')}; tie-break {joinValues(preferenceSummary.tie_break_order)}.
              </li>
            ) : null}
            {certifiedSetSummary ? (
              <li>
                Certified set: routes {joinValues(certifiedSetRouteIds)}; frontier {joinValues(frontierRouteIds)}; minimum-cost{' '}
                {certifiedSetSummary.minimum_cost_route_id ?? 'n/a'}; basis {certifiedSetSummary.certificate_basis}; certificate{' '}
                {pct(locale, certifiedSetSummary.certificate_value)} vs threshold{' '}
                {pct(locale, certifiedSetSummary.certificate_threshold)}.
              </li>
            ) : null}
            {supportSummary ? (
              <li>
                Support: {supportSummary.satisfied ? 'satisfied' : 'not satisfied'}; observed{' '}
                {supportSummary.observed_source_count}/{supportSummary.required_source_count}; mix{' '}
                {joinValues(supportSummary.source_mix)}; missing {joinValues(supportSummary.missing_sources)}; provenance{' '}
                {supportSummary.provenance_mode ?? 'n/a'}.
              </li>
            ) : null}
            {supportSummary?.sources.length ? (
              <li>
                Support sources:{' '}
                {supportSummary.sources
                  .map(
                    (source) =>
                      `${source.source_id} [${source.status}; ${source.present ? 'present' : 'missing'}${source.required ? '; required' : ''}]`,
                  )
                  .join('; ')}
                .
              </li>
            ) : null}
            {abstentionSummary ? (
              <li>
                Abstention: {abstentionSummary.abstained ? 'abstained' : 'not abstained'}; reason{' '}
                {abstentionSummary.reason_code ?? 'n/a'}; blocking {joinValues(abstentionSummary.blocking_sources)}; message{' '}
                {abstentionSummary.message ?? 'n/a'}.
              </li>
            ) : null}
            {witnessSummary ? (
              <li>
                Witness: primary {witnessSummary.primary_witness_route_id ?? 'n/a'}; witness routes{' '}
                {joinValues(witnessSummary.witness_route_ids)}; challengers {joinValues(witnessSummary.challenger_route_ids)}; source
                ids {joinValues(witnessSummary.witness_source_ids)}; worlds {n(locale, witnessSummary.witness_world_count)}.
              </li>
            ) : null}
            {controllerSummary ? (
              <li>
                Controller: mode {controllerSummary.controller_mode}; engaged {yn(controllerSummary.engaged)}; iterations{' '}
                {n(locale, controllerSummary.iteration_count)}; actions {n(locale, controllerSummary.action_count)}; stop reason{' '}
                {controllerSummary.stop_reason ?? 'n/a'}; budgets {n(locale, controllerSummary.search_budget_used)}/
                {n(locale, controllerSummary.evidence_budget_used)}.
              </li>
            ) : null}
            {laneManifest ? (
              <li>
                Lane: {laneManifest.lane_name ?? laneManifest.lane_id ?? 'n/a'} {laneManifest.lane_version ?? ''}. Artifacts{' '}
                {joinValues(laneManifest.artifact_names)}.
              </li>
            ) : null}
          </ul>
        </>
      ) : null}

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
          <div>
            <div className="baselineImpactGrid__label">Search completeness</div>
            <div className="baselineImpactGrid__value">{pct(locale, voiStopSummary.search_completeness_score)}</div>
          </div>
          <div>
            <div className="baselineImpactGrid__label">Search gap</div>
            <div className="baselineImpactGrid__value">{n(locale, voiStopSummary.search_completeness_gap)}</div>
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
