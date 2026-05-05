'use client';

import type { ReactNode } from 'react';

import type { DecisionPackage, FrontendArtifactInspectionItem, RunArtifactsListResponse } from '../../lib/types';
import { buildFrontendArtifactInspectionGroups } from '../../lib/types';

type CoreDocKind = 'manifest' | 'scenarioManifest' | 'provenance' | 'signature' | 'scenarioSignature';

type Props = {
  runId: string;
  onRunIdChange: (runId: string) => void;
  loading: boolean;
  error: string | null;
  decisionPackage?: DecisionPackage | null;
  manifest: unknown | null;
  scenarioManifest: unknown | null;
  provenance: unknown | null;
  signature: unknown | null;
  scenarioSignature: unknown | null;
  artifacts: RunArtifactsListResponse | null;
  artifactPreviewName: string | null;
  artifactPreviewText: string | null;
  onLoadCore: () => void;
  onLoadArtifacts: () => void;
  onPreviewArtifact: (name: string) => void;
  onDownloadCore: (kind: CoreDocKind) => void;
  onDownloadArtifact: (name: string) => void;
};

function pretty(value: unknown): string {
  if (value == null) return '';
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function joinValues(values: Array<string | null | undefined> | null | undefined, empty = 'n/a'): string {
  const filtered = (values ?? []).filter((value): value is string => Boolean(value));
  return filtered.length ? filtered.join(', ') : empty;
}

function yesNo(value: boolean | null | undefined): string {
  return value ? 'yes' : 'no';
}

function artifactStateLabel(item: FrontendArtifactInspectionItem): string {
  if (item.present) return 'available';
  if (item.listed) return 'listed by lane manifest';
  return item.expectation === 'guaranteed' ? 'expected after artifact fetch' : 'conditional';
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <>
      <div className="tiny u-mt12">{title}</div>
      {children}
    </>
  );
}

export default function RunInspector({
  runId,
  onRunIdChange,
  loading,
  error,
  decisionPackage,
  manifest,
  scenarioManifest,
  provenance,
  signature,
  scenarioSignature,
  artifacts,
  artifactPreviewName,
  artifactPreviewText,
  onLoadCore,
  onLoadArtifacts,
  onPreviewArtifact,
  onDownloadCore,
  onDownloadArtifact,
}: Props) {
  const disabled = loading || !runId.trim();
  const supportSummary = decisionPackage?.support_summary ?? null;
  const worldSupportSummary = decisionPackage?.world_support_summary ?? null;
  const worldFidelitySummary = decisionPackage?.world_fidelity_summary ?? null;
  const certificationStateSummary = decisionPackage?.certification_state_summary ?? null;
  const certifiedSetSummary = decisionPackage?.certified_set_summary ?? null;
  const preferenceSummary = decisionPackage?.preference_summary ?? null;
  const abstentionSummary = decisionPackage?.abstention_summary ?? null;
  const witnessSummary = decisionPackage?.witness_summary ?? null;
  const controllerSummary = decisionPackage?.controller_summary ?? null;
  const theoremHooks = decisionPackage?.theorem_hook_summary?.hooks ?? [];
  const laneManifest = decisionPackage?.lane_manifest ?? null;
  const artifactGroups = buildFrontendArtifactInspectionGroups({
    decisionPackage,
    listedArtifactNames: laneManifest?.artifact_names ?? null,
    artifacts,
  });
  const availableArtifactCount = artifactGroups.reduce((total, group) => total + group.presentCount, 0);
  const listedArtifactCount = artifactGroups.reduce((total, group) => total + group.listedCount, 0);
  const pendingArtifactCount = artifactGroups.reduce((total, group) => total + group.missingExpectedCount, 0);

  return (
    <div className="devCard">
      <div className="devCard__head">
        <h4 className="devCard__title">Run Inspector</h4>
      </div>

      <div className="fieldLabel">Run ID</div>
      <input
        className="input"
        value={runId}
        onChange={(event) => onRunIdChange(event.target.value)}
        placeholder="Paste run_id"
        disabled={loading}
      />

      <div className="row u-mt10">
        <button type="button" className="secondary" onClick={onLoadCore} disabled={disabled}>
          {loading ? 'Loading...' : 'Inspect Core Docs'}
        </button>
        <button type="button" className="secondary" onClick={onLoadArtifacts} disabled={disabled}>
          {loading ? 'Loading...' : 'List Artifacts'}
        </button>
      </div>

      {error ? <div className="error">{error}</div> : null}

      {decisionPackage ? (
        <>
          <Section title="decision-package">
            <ul className="baselineNotes">
              <li>
                Package {decisionPackage.package_kind} schema {decisionPackage.schema_version} pipeline {decisionPackage.pipeline_mode}
                ; terminal {decisionPackage.terminal_kind}; selected{' '}
                {decisionPackage.selected_route_id ?? certifiedSetSummary?.selected_route_id ?? 'n/a'}.
              </li>
              {preferenceSummary ? (
                <li>
                  Preference: {preferenceSummary.objective_field} via {preferenceSummary.selector_policy}; selective{' '}
                  {yesNo(preferenceSummary.selective)}; tie-break {joinValues(preferenceSummary.tie_break_order)}.
                </li>
              ) : null}
              {certifiedSetSummary ? (
                <li>
                  Certified set: certified {yesNo(certifiedSetSummary.certified)}; routes{' '}
                  {joinValues(certifiedSetSummary.certified_route_ids)}; frontier {joinValues(certifiedSetSummary.frontier_route_ids)}; basis{' '}
                  {certifiedSetSummary.certificate_basis}; minimum-cost {certifiedSetSummary.minimum_cost_route_id ?? 'n/a'}.
                </li>
              ) : null}
              {supportSummary ? (
                <li>
                  Support: satisfied {yesNo(supportSummary.satisfied)}; observed {supportSummary.observed_source_count}/
                  {supportSummary.required_source_count}; mix {joinValues(supportSummary.source_mix)}; missing{' '}
                  {joinValues(supportSummary.missing_sources)}; provenance {supportSummary.provenance_mode ?? 'n/a'}.
                </li>
              ) : null}
              {worldSupportSummary ? (
                <li>
                  World support: strength {worldSupportSummary.support_strength ?? 'n/a'}; source support{' '}
                  {worldSupportSummary.source_support_strength ?? 'n/a'}; fidelity {worldSupportSummary.recommended_fidelity}; proxy penalty{' '}
                  {worldSupportSummary.proxy_penalty ?? 'n/a'}; audit correction {worldSupportSummary.audit_correction ?? 'n/a'}; support sufficient{' '}
                  {yesNo(worldSupportSummary.support_sufficient)}.
                </li>
              ) : null}
              {worldFidelitySummary ? (
                <li>
                  World fidelity: mode {worldFidelitySummary.multi_fidelity_mode}; policy {worldFidelitySummary.policy}; worlds{' '}
                  {worldFidelitySummary.effective_world_count}/{worldFidelitySummary.world_count} effective; proxy{' '}
                  {worldFidelitySummary.proxy_world_fraction ?? 'n/a'}; stress {worldFidelitySummary.stress_world_fraction ?? 'n/a'}; reuse{' '}
                  {worldFidelitySummary.world_reuse_rate ?? 'n/a'}; recommended policy {worldFidelitySummary.recommended_policy}.
                </li>
              ) : null}
              {certificationStateSummary ? (
                <li>
                  Certification state: winner {certificationStateSummary.winner_id ?? 'n/a'}; basis{' '}
                  {certificationStateSummary.certification_basis}; certified {yesNo(certificationStateSummary.certified)}; abstained{' '}
                  {yesNo(certificationStateSummary.abstained)}; support {certificationStateSummary.support_strength ?? 'n/a'}; set safe{' '}
                  {String(certificationStateSummary.certified_set?.safe ?? 'n/a')}.
                </li>
              ) : null}
              {abstentionSummary ? (
                <li>
                  Abstention: {abstentionSummary.abstained ? 'abstained' : 'clear'}; reason {abstentionSummary.reason_code ?? 'n/a'}; blocking{' '}
                  {joinValues(abstentionSummary.blocking_sources)}; retryable {yesNo(abstentionSummary.retryable)}.
                </li>
              ) : null}
              {witnessSummary ? (
                <li>
                  Witness: primary {witnessSummary.primary_witness_route_id ?? 'n/a'}; witness routes{' '}
                  {joinValues(witnessSummary.witness_route_ids)}; challengers {joinValues(witnessSummary.challenger_route_ids)}; source ids{' '}
                  {joinValues(witnessSummary.witness_source_ids)}; worlds {witnessSummary.witness_world_count ?? 'n/a'}.
                </li>
              ) : null}
              {controllerSummary ? (
                <li>
                  Controller: mode {controllerSummary.controller_mode}; engaged {yesNo(controllerSummary.engaged)}; iterations{' '}
                  {controllerSummary.iteration_count}; actions {controllerSummary.action_count}; stop reason{' '}
                  {controllerSummary.stop_reason ?? 'n/a'}; budgets {controllerSummary.search_budget_used}/
                  {controllerSummary.evidence_budget_used}.
                </li>
              ) : null}
              {laneManifest ? (
                <li>
                  Lane: {laneManifest.lane_name ?? laneManifest.lane_id ?? 'n/a'} {laneManifest.lane_version ?? ''}; artifacts{' '}
                  {joinValues(laneManifest.artifact_names)}.
                </li>
              ) : null}
              {theoremHooks.length ? (
                <li>
                  Theorem hooks:{' '}
                  {theoremHooks
                    .map((hook) => `${hook.hook_id} [${hook.status}${hook.artifact_name ? `; ${hook.artifact_name}` : ''}]`)
                    .join('; ')}
                  .
                </li>
              ) : null}
            </ul>
            <pre className="devPre">{pretty(decisionPackage.provenance)}</pre>
          </Section>
        </>
      ) : null}

      <div className="row u-mt10">
        <button type="button" className="ghostButton" onClick={() => onDownloadCore('manifest')} disabled={disabled}>
          Download Manifest
        </button>
        <button type="button" className="ghostButton" onClick={() => onDownloadCore('scenarioManifest')} disabled={disabled}>
          Download Scenario Manifest
        </button>
        <button type="button" className="ghostButton" onClick={() => onDownloadCore('provenance')} disabled={disabled}>
          Download Provenance
        </button>
      </div>

      <div className="row u-mt10">
        <button type="button" className="ghostButton" onClick={() => onDownloadCore('signature')} disabled={disabled}>
          Download Signature JSON
        </button>
        <button type="button" className="ghostButton" onClick={() => onDownloadCore('scenarioSignature')} disabled={disabled}>
          Download Scenario Signature JSON
        </button>
      </div>

      {artifactGroups.length ? (
        <Section title="artifact-inspection">
          <ul className="baselineNotes">
            <li>
              Artifact groups: {artifactGroups.length}; available {availableArtifactCount}; listed {listedArtifactCount}; pending{' '}
              {pendingArtifactCount}.
            </li>
            <li>
              Guaranteed artifacts are expected whenever a decision package exists; conditional artifacts depend on abstention, witness,
              controller, theorem-hook, lane, or evaluator state.
            </li>
          </ul>
          <ul className="routeList">
            {artifactGroups.map((group) => (
              <li key={group.id} className="routeCard" style={{ cursor: 'default' }}>
                <div className="routeCard__top">
                  <div className="routeCard__id">{group.label}</div>
                  <div className="routeCard__pill">
                    {group.presentCount} present / {group.items.length} tracked
                  </div>
                </div>
                <div className="tiny">{group.description}</div>
                <div className="tiny u-mt10">
                  Listed {group.listedCount} | Pending expected {group.missingExpectedCount}
                </div>
                <ul className="baselineNotes">
                  {group.items.map((item) => (
                    <li key={item.name}>
                      <strong>{item.label}</strong> <code>{item.name}</code>: {item.description}. Status {artifactStateLabel(item)}; expectation{' '}
                      {item.expectation}
                      {item.sizeBytes != null ? `; ${item.sizeBytes} bytes` : ''}.
                    </li>
                  ))}
                </ul>
                <div className="row">
                  {group.items
                    .filter((item) => item.present)
                    .slice(0, 4)
                    .map((item) => (
                      <button
                        key={item.name}
                        type="button"
                        className="secondary"
                        onClick={() => onPreviewArtifact(item.name)}
                        disabled={loading}
                      >
                        Inspect {item.label}
                      </button>
                    ))}
                  {group.items
                    .filter((item) => item.present)
                    .slice(0, 2)
                    .map((item) => (
                      <button
                        key={`${item.name}-download`}
                        type="button"
                        className="secondary"
                        onClick={() => onDownloadArtifact(item.name)}
                        disabled={loading}
                      >
                        Download {item.label}
                      </button>
                    ))}
                </div>
              </li>
            ))}
          </ul>
        </Section>
      ) : artifacts ? (
        <Section title="artifact-inspection">
          <div className="tiny">The run-store listing returned no mapped artifacts for this run.</div>
        </Section>
      ) : null}

      {manifest ? (
        <Section title="manifest">
          <pre className="devPre">{pretty(manifest)}</pre>
        </Section>
      ) : null}

      {scenarioManifest ? (
        <Section title="scenario-manifest">
          <pre className="devPre">{pretty(scenarioManifest)}</pre>
        </Section>
      ) : null}

      {provenance ? (
        <Section title="provenance">
          <pre className="devPre">{pretty(provenance)}</pre>
        </Section>
      ) : null}

      {signature ? (
        <Section title="signature">
          <pre className="devPre">{pretty(signature)}</pre>
        </Section>
      ) : null}

      {scenarioSignature ? (
        <Section title="scenario-signature">
          <pre className="devPre">{pretty(scenarioSignature)}</pre>
        </Section>
      ) : null}

      {artifactPreviewName && artifactPreviewText ? (
        <Section title={`artifact-preview: ${artifactPreviewName}`}>
          <pre className="devPre">{artifactPreviewText}</pre>
        </Section>
      ) : null}
    </div>
  );
}
