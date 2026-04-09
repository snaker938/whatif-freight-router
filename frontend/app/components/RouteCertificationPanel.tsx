'use client';

import type { ReactNode } from 'react';

import type {
  DecisionPackage,
  FrontendArtifactInspectionItem,
  PipelineMode,
  PreferenceQuerySummary,
  RouteCertificationSummary,
  RouteOption,
  VoiStopSummary,
} from '../lib/types';
import { buildFrontendArtifactInspectionGroups } from '../lib/types';

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

function describePreferenceQuery(query: PreferenceQuerySummary): string {
  const routeIds = joinValues(query.route_ids);
  const options = query.options.length ? query.options.join(' / ') : 'n/a';
  const metadata = Object.entries(query.metadata ?? {})
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(', ');
  return `${query.kind}: ${query.prompt} | options ${options} | routes ${routeIds}${metadata ? ` | ${metadata}` : ''}`;
}

function artifactStateLabel(item: FrontendArtifactInspectionItem): string {
  if (item.present) return 'available';
  if (item.listed) return 'listed by lane manifest';
  return item.expectation === 'guaranteed' ? 'expected after artifact fetch' : 'conditional';
}

function InspectionSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="u-mt10">
      <div className="baselineComparePanel__tradeoff">
        <strong>{title}</strong>
      </div>
      {children}
    </div>
  );
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
  const worldSupportSummary = decisionPackage?.world_support_summary ?? certification?.world_support_summary ?? null;
  const worldFidelitySummary = decisionPackage?.world_fidelity_summary ?? certification?.world_fidelity_summary ?? null;
  const certificationStateSummary =
    decisionPackage?.certification_state_summary ?? certification?.certification_state_summary ?? null;
  const preferenceSummary = decisionPackage?.preference_summary ?? null;
  const abstentionSummary = decisionPackage?.abstention_summary ?? null;
  const witnessSummary = decisionPackage?.witness_summary ?? null;
  const controllerSummary = decisionPackage?.controller_summary ?? null;
  const theoremHooks = decisionPackage?.theorem_hook_summary?.hooks ?? [];
  const laneManifest = decisionPackage?.lane_manifest ?? null;
  const ambiguitySummary = describeAmbiguityContext(certification?.ambiguity_context);
  const resolvedPipelineMode = decisionPackage?.pipeline_mode ?? pipelineMode;
  const certifiedSetRouteIds = certifiedSetSummary?.certified_route_ids ?? [];
  const frontierRouteIds = certifiedSetSummary?.frontier_route_ids ?? [];
  const isCertified = certification?.certified ?? certifiedSetSummary?.certified ?? false;
  const selectedRouteId = decisionPackage?.selected_route_id ?? certifiedSetSummary?.selected_route_id ?? route?.id ?? 'n/a';
  const artifactGroups = buildFrontendArtifactInspectionGroups({
    decisionPackage,
    listedArtifactNames: laneManifest?.artifact_names ?? null,
  });
  const availableArtifactCount = artifactGroups.reduce((total, group) => total + group.presentCount, 0);
  const listedArtifactCount = artifactGroups.reduce((total, group) => total + group.listedCount, 0);
  const missingArtifactCount = artifactGroups.reduce((total, group) => total + group.missingExpectedCount, 0);

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
            Decision package <code>{decisionPackage.schema_version}</code> selected route <code>{selectedRouteId}</code>.
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
                {supportSummary ? `${supportSummary.observed_source_count}/${supportSummary.required_source_count}` : 'n/a'}
              </div>
              <div className="baselineKpi__meta">{supportSummary?.support_mode ?? 'n/a'}</div>
            </div>
            <div className={`baselineKpi ${worldSupportSummary?.support_sufficient ? 'isPositive' : 'isNegative'}`}>
              <div className="baselineKpi__label">World Support</div>
              <div className="baselineKpi__value">{worldSupportSummary?.recommended_fidelity ?? 'n/a'}</div>
              <div className="baselineKpi__meta">
                Strength {pct(locale, worldSupportSummary?.support_strength)} | Proxy penalty {pct(locale, worldSupportSummary?.proxy_penalty)}
              </div>
            </div>
            <div className={`baselineKpi ${abstentionSummary?.abstained ? 'isNegative' : 'isPositive'}`}>
              <div className="baselineKpi__label">Abstention</div>
              <div className="baselineKpi__value">
                {abstentionSummary ? (abstentionSummary.abstained ? abstentionSummary.reason_code ?? 'Abstained' : 'Clear') : 'n/a'}
              </div>
              <div className="baselineKpi__meta">
                Retryable {abstentionSummary ? yn(abstentionSummary.retryable) : 'n/a'} | Blocking{' '}
                {abstentionSummary ? abstentionSummary.blocking_sources.length : 'n/a'}
              </div>
            </div>
            <div className={`baselineKpi ${decisionPackage.terminal_kind === 'typed_abstention' ? 'isNegative' : 'isPositive'}`}>
              <div className="baselineKpi__label">Terminal</div>
              <div className="baselineKpi__value">{decisionPackage.terminal_kind}</div>
              <div className="baselineKpi__meta">Selected {selectedRouteId}</div>
            </div>
            <div className={`baselineKpi ${worldFidelitySummary?.world_count ? 'isPositive' : 'isNegative'}`}>
              <div className="baselineKpi__label">World Fidelity</div>
              <div className="baselineKpi__value">{worldFidelitySummary?.multi_fidelity_mode ?? 'n/a'}</div>
              <div className="baselineKpi__meta">
                Worlds {worldFidelitySummary?.effective_world_count ?? 'n/a'} | Reuse {pct(locale, worldFidelitySummary?.world_reuse_rate)}
              </div>
            </div>
            <div className={`baselineKpi ${certificationStateSummary?.certified ? 'isPositive' : 'isNegative'}`}>
              <div className="baselineKpi__label">Cert State</div>
              <div className="baselineKpi__value">{certificationStateSummary?.certification_basis ?? 'n/a'}</div>
              <div className="baselineKpi__meta">
                Winner {certificationStateSummary?.winner_id ?? 'n/a'} | Abstained{' '}
                {certificationStateSummary ? yn(certificationStateSummary.abstained) : 'n/a'}
              </div>
            </div>
            <div className="baselineKpi">
              <div className="baselineKpi__label">Witnesses</div>
              <div className="baselineKpi__value">{witnessSummary?.witness_route_ids.length ?? 0}</div>
              <div className="baselineKpi__meta">Challengers {witnessSummary?.challenger_route_ids.length ?? 0}</div>
            </div>
            <div className={`baselineKpi ${missingArtifactCount ? 'isNegative' : 'isPositive'}`}>
              <div className="baselineKpi__label">Artifact Handoff</div>
              <div className="baselineKpi__value">{artifactGroups.length}</div>
              <div className="baselineKpi__meta">
                Available {availableArtifactCount} | Listed {listedArtifactCount} | Pending {missingArtifactCount}
              </div>
            </div>
          </div>

          <InspectionSection title="Terminal outcome">
            <ul className="baselineNotes">
              <li>
                Terminal kind <code>{decisionPackage.terminal_kind}</code>; selected route <code>{selectedRouteId}</code>; pipeline{' '}
                <code>{decisionPackage.pipeline_mode}</code>.
              </li>
              {certificationStateSummary ? (
                <li>
                  Certification state: winner {certificationStateSummary.winner_id ?? 'n/a'}; basis{' '}
                  {certificationStateSummary.certification_basis}; certified {yn(certificationStateSummary.certified, 'yes', 'no')}; support{' '}
                  {pct(locale, certificationStateSummary.support_strength)}; threshold {pct(locale, certificationStateSummary.threshold)}; region{' '}
                  {String(certificationStateSummary.decision_region?.status ?? 'n/a')}.
                </li>
              ) : null}
              {ambiguitySummary ? <li>Ambiguity context: {ambiguitySummary}.</li> : null}
            </ul>
          </InspectionSection>

          <InspectionSection title="Certified set and preferences">
            <ul className="baselineNotes">
              {certifiedSetSummary ? (
                <li>
                  Certified set: routes {joinValues(certifiedSetRouteIds)}; frontier {joinValues(frontierRouteIds)}; minimum-cost{' '}
                  {certifiedSetSummary.minimum_cost_route_id ?? 'n/a'}; basis {certifiedSetSummary.certificate_basis}; certificate{' '}
                  {pct(locale, certifiedSetSummary.certificate_value)} vs threshold {pct(locale, certifiedSetSummary.certificate_threshold)}.
                </li>
              ) : null}
              {preferenceSummary ? (
                <li>
                  Preference: objective {preferenceSummary.objective_field} via {preferenceSummary.selector_policy}; selective{' '}
                  {yn(preferenceSummary.selective, 'on', 'off')}; certified-only {yn(preferenceSummary.certified_only_required)}; time guard{' '}
                  {yn(preferenceSummary.time_guard_active)}; vetoes {joinValues(preferenceSummary.vetoed_targets)}; tie-break{' '}
                  {joinValues(preferenceSummary.tie_break_order)}.
                </li>
              ) : null}
              {preferenceSummary?.suggested_queries?.length ? (
                <li>Preference queries: {preferenceSummary.suggested_queries.map(describePreferenceQuery).join(' || ')}.</li>
              ) : null}
            </ul>
          </InspectionSection>

          <InspectionSection title="Support and fidelity">
            <ul className="baselineNotes">
              {supportSummary ? (
                <li>
                  Support: {supportSummary.satisfied ? 'satisfied' : 'not satisfied'}; observed{' '}
                  {supportSummary.observed_source_count}/{supportSummary.required_source_count}; mix{' '}
                  {joinValues(supportSummary.source_mix)}; missing {joinValues(supportSummary.missing_sources)}; provenance{' '}
                  {supportSummary.provenance_mode ?? 'n/a'}.
                </li>
              ) : null}
              {worldSupportSummary ? (
                <li>
                  World support: strength {pct(locale, worldSupportSummary.support_strength)}; source support{' '}
                  {pct(locale, worldSupportSummary.source_support_strength)}; fidelity {worldSupportSummary.recommended_fidelity}; proxy penalty{' '}
                  {pct(locale, worldSupportSummary.proxy_penalty)}; audit correction {pct(locale, worldSupportSummary.audit_correction)}; support sufficient{' '}
                  {yn(worldSupportSummary.support_sufficient, 'yes', 'no')}; notes {joinValues(worldSupportSummary.notes)}.
                </li>
              ) : null}
              {worldFidelitySummary ? (
                <li>
                  World fidelity: mode {worldFidelitySummary.multi_fidelity_mode}; policy {worldFidelitySummary.policy}; worlds{' '}
                  {n(locale, worldFidelitySummary.world_count)}/{n(locale, worldFidelitySummary.effective_world_count)} effective; proxy{' '}
                  {pct(locale, worldFidelitySummary.proxy_world_fraction)}; stress {pct(locale, worldFidelitySummary.stress_world_fraction)}; reuse{' '}
                  {pct(locale, worldFidelitySummary.world_reuse_rate)}; recommended policy {worldFidelitySummary.recommended_policy}; notes{' '}
                  {joinValues(worldFidelitySummary.notes)}.
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
            </ul>
          </InspectionSection>

          <InspectionSection title="Abstention and controller">
            <ul className="baselineNotes">
              {abstentionSummary ? (
                <li>
                  Abstention: {abstentionSummary.abstained ? 'abstained' : 'not abstained'}; reason{' '}
                  {abstentionSummary.reason_code ?? 'n/a'}; blocking {joinValues(abstentionSummary.blocking_sources)}; message{' '}
                  {abstentionSummary.message ?? 'n/a'}; type {abstentionSummary.abstention_type ?? 'n/a'}; recommended action{' '}
                  {abstentionSummary.recommended_action ?? 'n/a'}.
                </li>
              ) : (
                <li>
                  Abstention: terminal outcome is <code>{decisionPackage.terminal_kind}</code>; typed abstention artifacts stay conditional
                  unless the runtime actually abstains.
                </li>
              )}
              {controllerSummary ? (
                <li>
                  Controller: mode {controllerSummary.controller_mode}; engaged {yn(controllerSummary.engaged)}; iterations{' '}
                  {n(locale, controllerSummary.iteration_count)}; actions {n(locale, controllerSummary.action_count)}; stop reason{' '}
                  {controllerSummary.stop_reason ?? 'n/a'}; budgets {n(locale, controllerSummary.search_budget_used)}/
                  {n(locale, controllerSummary.evidence_budget_used)}.
                </li>
              ) : null}
              {voiStopSummary ? (
                <li>
                  Stop summary: route {voiStopSummary.final_route_id}; certificate {pct(locale, voiStopSummary.certificate)}; search completeness{' '}
                  {pct(locale, voiStopSummary.search_completeness_score)}; search gap {n(locale, voiStopSummary.search_completeness_gap)}.
                </li>
              ) : null}
            </ul>
          </InspectionSection>

          <InspectionSection title="Witnesses and theorem hooks">
            <ul className="baselineNotes">
              {witnessSummary ? (
                <li>
                  Witness: primary {witnessSummary.primary_witness_route_id ?? 'n/a'}; witness routes{' '}
                  {joinValues(witnessSummary.witness_route_ids)}; challengers {joinValues(witnessSummary.challenger_route_ids)}; source ids{' '}
                  {joinValues(witnessSummary.witness_source_ids)}; worlds {n(locale, witnessSummary.witness_world_count)}.
                </li>
              ) : (
                <li>Witness artifacts remain conditional until certification emits witness state for this run.</li>
              )}
              {theoremHooks.length ? (
                <li>
                  Theorem hooks:{' '}
                  {theoremHooks
                    .map((hook) => `${hook.hook_id} [${hook.status}${hook.artifact_name ? `; ${hook.artifact_name}` : ''}]`)
                    .join('; ')}
                  .
                </li>
              ) : (
                <li>No theorem-hook entries were attached to this decision package.</li>
              )}
              {laneManifest ? (
                <li>
                  Lane: {laneManifest.lane_name ?? laneManifest.lane_id ?? 'n/a'} {laneManifest.lane_version ?? ''}. Declared artifacts{' '}
                  {joinValues(laneManifest.artifact_names)}.
                </li>
              ) : null}
            </ul>
          </InspectionSection>

          <InspectionSection title="Proof and artifact handoff">
            <ul className="baselineNotes">
              <li>
                Run Inspector groups artifacts into route-core, decision-package, support, DCCS / REFC, controller / VOI,
                witness / fragility, theorem / lane, and evaluation families.
              </li>
              {artifactGroups.map((group) => (
                <li key={group.id}>
                  {group.label}: {group.description} Present {group.presentCount}; listed {group.listedCount}; pending{' '}
                  {group.missingExpectedCount}.{' '}
                  {group.items
                    .map((item) => `${item.name} [${artifactStateLabel(item)}; ${item.expectation}]`)
                    .join('; ')}
                  .
                </li>
              ))}
            </ul>
          </InspectionSection>
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
