'use client';

import type { RunArtifactsListResponse } from '../../lib/types';

type CoreDocKind = 'manifest' | 'scenarioManifest' | 'provenance' | 'signature' | 'scenarioSignature';

type Props = {
  runId: string;
  onRunIdChange: (runId: string) => void;
  loading: boolean;
  error: string | null;
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

export default function RunInspector({
  runId,
  onRunIdChange,
  loading,
  error,
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
