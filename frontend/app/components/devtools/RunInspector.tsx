'use client';

import type { DecisionPackage, RunArtifactsListResponse } from '../../lib/types';

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
  const certifiedSetSummary = decisionPackage?.certified_set_summary ?? null;
  const preferenceSummary = decisionPackage?.preference_summary ?? null;
  const abstentionSummary = decisionPackage?.abstention_summary ?? null;
  const witnessSummary = decisionPackage?.witness_summary ?? null;
  const controllerSummary = decisionPackage?.controller_summary ?? null;
  const theoremHooks = decisionPackage?.theorem_hook_summary?.hooks ?? [];
  const laneManifest = decisionPackage?.lane_manifest ?? null;

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
          <div className="tiny u-mt12">decision-package</div>
          <ul className="baselineNotes">
            <li>
              Package {decisionPackage.package_kind} schema {decisionPackage.schema_version} pipeline{' '}
              {decisionPackage.pipeline_mode} selected {decisionPackage.selected_route_id ?? certifiedSetSummary?.selected_route_id ?? 'n/a'}.
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
                Abstention: {abstentionSummary.abstained ? 'abstained' : 'clear'}; reason{' '}
                {abstentionSummary.reason_code ?? 'n/a'}; blocking {joinValues(abstentionSummary.blocking_sources)}; retryable{' '}
                {yesNo(abstentionSummary.retryable)}.
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
        </>
      ) : null}

      <div className="row u-mt10">
        <button type="button" className="ghostButton" onClick={() => onDownloadCore('manifest')} disabled={disabled}>
          Download Manifest
        </button>
        <button
          type="button"
          className="ghostButton"
          onClick={() => onDownloadCore('scenarioManifest')}
          disabled={disabled}
        >
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
        <button
          type="button"
          className="ghostButton"
          onClick={() => onDownloadCore('scenarioSignature')}
          disabled={disabled}
        >
          Download Scenario Signature JSON
        </button>
      </div>

      {manifest ? (
        <>
          <div className="tiny u-mt12">manifest</div>
          <pre className="devPre">{pretty(manifest)}</pre>
        </>
      ) : null}

      {scenarioManifest ? (
        <>
          <div className="tiny u-mt12">scenario-manifest</div>
          <pre className="devPre">{pretty(scenarioManifest)}</pre>
        </>
      ) : null}

      {provenance ? (
        <>
          <div className="tiny u-mt12">provenance</div>
          <pre className="devPre">{pretty(provenance)}</pre>
        </>
      ) : null}

      {signature ? (
        <>
          <div className="tiny u-mt12">signature</div>
          <pre className="devPre">{pretty(signature)}</pre>
        </>
      ) : null}

      {scenarioSignature ? (
        <>
          <div className="tiny u-mt12">scenario-signature</div>
          <pre className="devPre">{pretty(scenarioSignature)}</pre>
        </>
      ) : null}

      {artifacts ? (
        <>
          <div className="tiny u-mt12">Artifacts</div>
          <ul className="routeList">
            {artifacts.artifacts.map((artifact) => (
              <li key={artifact.name} className="routeCard" style={{ cursor: 'default' }}>
                <div className="routeCard__top">
                  <div className="routeCard__id">{artifact.name}</div>
                  <div className="routeCard__pill">{artifact.size_bytes} bytes</div>
                </div>
                <div className="row">
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => onPreviewArtifact(artifact.name)}
                    disabled={loading}
                  >
                    Inspect
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => onDownloadArtifact(artifact.name)}
                    disabled={loading}
                  >
                    Download
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      {artifactPreviewName && artifactPreviewText ? (
        <>
          <div className="tiny u-mt12">Artifact Preview: {artifactPreviewName}</div>
          <pre className="devPre">{artifactPreviewText}</pre>
        </>
      ) : null}
    </div>
  );
}
