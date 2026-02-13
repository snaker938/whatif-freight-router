'use client';

import { useMemo, useState } from 'react';

import type {
  OracleFeedCheckInput,
  OracleFeedCheckRecord,
  OracleQualityDashboardResponse,
} from '../lib/types';

type Props = {
  dashboard: OracleQualityDashboardResponse | null;
  loading: boolean;
  ingesting: boolean;
  error: string | null;
  latestCheck: OracleFeedCheckRecord | null;
  disabled: boolean;
  onRefresh: () => void;
  onIngest: (payload: OracleFeedCheckInput) => Promise<void> | void;
};

export default function OracleQualityDashboard({
  dashboard,
  loading,
  ingesting,
  error,
  latestCheck,
  disabled,
  onRefresh,
  onIngest,
}: Props) {
  const [source, setSource] = useState('oracle_demo');
  const [schemaValid, setSchemaValid] = useState(true);
  const [signatureState, setSignatureState] = useState<'unknown' | 'valid' | 'invalid'>('unknown');
  const [freshnessS, setFreshnessS] = useState('');
  const [latencyMs, setLatencyMs] = useState('');
  const [recordCount, setRecordCount] = useState('');
  const [errorText, setErrorText] = useState('');

  const csvHref = useMemo(() => '/api/oracle/quality/dashboard.csv', []);

  async function handleIngest() {
    const trimmedSource = source.trim();
    if (!trimmedSource) return;

    const parsedFreshness =
      freshnessS.trim() === '' ? null : Number.isFinite(Number(freshnessS)) ? Number(freshnessS) : NaN;
    const parsedLatency =
      latencyMs.trim() === '' ? null : Number.isFinite(Number(latencyMs)) ? Number(latencyMs) : NaN;
    const parsedCount =
      recordCount.trim() === '' ? null : Number.isFinite(Number(recordCount)) ? Number(recordCount) : NaN;

    if (Number.isNaN(parsedFreshness) || (parsedFreshness !== null && parsedFreshness < 0)) return;
    if (Number.isNaN(parsedLatency) || (parsedLatency !== null && parsedLatency < 0)) return;
    if (
      Number.isNaN(parsedCount) ||
      (parsedCount !== null && (!Number.isInteger(parsedCount) || parsedCount < 0))
    ) {
      return;
    }

    await onIngest({
      source: trimmedSource,
      schema_valid: schemaValid,
      signature_valid:
        signatureState === 'unknown' ? null : signatureState === 'valid',
      freshness_s: parsedFreshness,
      latency_ms: parsedLatency,
      record_count: parsedCount,
      error: errorText.trim() || null,
    });
  }

  return (
    <section className="card">
      <div className="sectionTitleRow">
        <div className="sectionTitle">Oracle quality dashboard</div>
        <div className="row" style={{ marginTop: 0 }}>
          <button className="secondary" onClick={onRefresh} disabled={disabled || loading || ingesting}>
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
          <a className="secondary" href={csvHref} style={{ textDecoration: 'none' }}>
            Download CSV
          </a>
        </div>
      </div>

      <div className="helper">
        Record feed checks and inspect source-level pass rates, freshness, and signature health.
      </div>

      <label className="fieldLabel" htmlFor="oracle-source">
        Source
      </label>
      <input
        id="oracle-source"
        className="input"
        value={source}
        disabled={disabled || ingesting}
        onChange={(event) => setSource(event.target.value)}
      />

      <div className="checkboxRow">
        <input
          id="oracle-schema-valid"
          type="checkbox"
          checked={schemaValid}
          disabled={disabled || ingesting}
          onChange={(event) => setSchemaValid(event.target.checked)}
        />
        <label htmlFor="oracle-schema-valid">Schema valid</label>
      </div>

      <label className="fieldLabel" htmlFor="oracle-signature-state">
        Signature state
      </label>
      <select
        id="oracle-signature-state"
        className="input"
        value={signatureState}
        disabled={disabled || ingesting}
        onChange={(event) => setSignatureState(event.target.value as 'unknown' | 'valid' | 'invalid')}
      >
        <option value="unknown">Unknown</option>
        <option value="valid">Valid</option>
        <option value="invalid">Invalid</option>
      </select>

      <div className="advancedGrid">
        <label className="fieldLabel" htmlFor="oracle-freshness">
          Freshness seconds (optional)
        </label>
        <input
          id="oracle-freshness"
          className="input"
          type="number"
          min={0}
          step="any"
          value={freshnessS}
          disabled={disabled || ingesting}
          onChange={(event) => setFreshnessS(event.target.value)}
        />

        <label className="fieldLabel" htmlFor="oracle-latency">
          Latency ms (optional)
        </label>
        <input
          id="oracle-latency"
          className="input"
          type="number"
          min={0}
          step="any"
          value={latencyMs}
          disabled={disabled || ingesting}
          onChange={(event) => setLatencyMs(event.target.value)}
        />

        <label className="fieldLabel" htmlFor="oracle-record-count">
          Record count (optional)
        </label>
        <input
          id="oracle-record-count"
          className="input"
          type="number"
          min={0}
          step={1}
          value={recordCount}
          disabled={disabled || ingesting}
          onChange={(event) => setRecordCount(event.target.value)}
        />
      </div>

      <label className="fieldLabel" htmlFor="oracle-error-note">
        Error note (optional)
      </label>
      <input
        id="oracle-error-note"
        className="input"
        value={errorText}
        disabled={disabled || ingesting}
        onChange={(event) => setErrorText(event.target.value)}
      />

      <div className="row row--actions" style={{ marginTop: 10 }}>
        <button className="secondary" onClick={handleIngest} disabled={disabled || ingesting || loading}>
          {ingesting ? 'Recording...' : 'Record check'}
        </button>
      </div>

      {error ? <div className="error">{error}</div> : null}
      {latestCheck ? (
        <div className="tiny">
          Last check: {latestCheck.source} ({latestCheck.passed ? 'passed' : 'failed'}) at{' '}
          {new Date(latestCheck.ingested_at_utc).toLocaleString()}
        </div>
      ) : null}

      {dashboard ? (
        <div style={{ marginTop: 10 }}>
          <div className="metrics">
            <div className="metric">
              <div className="metric__label">Total checks</div>
              <div className="metric__value">{dashboard.total_checks}</div>
            </div>
            <div className="metric">
              <div className="metric__label">Sources</div>
              <div className="metric__value">{dashboard.source_count}</div>
            </div>
            <div className="metric">
              <div className="metric__label">Stale threshold</div>
              <div className="metric__value">{dashboard.stale_threshold_s.toFixed(0)} s</div>
            </div>
          </div>

          <ul className="routeList" style={{ marginTop: 10 }}>
            {dashboard.sources.map((item) => (
              <li key={item.source} className="routeCard" style={{ cursor: 'default' }}>
                <div className="routeCard__top">
                  <div className="routeCard__id">{item.source}</div>
                  <div className="routeCard__pill">{(item.pass_rate * 100).toFixed(1)}% pass</div>
                </div>
                <div className="routeCard__meta">
                  <span>checks {item.check_count}</span>
                  <span>schema fail {item.schema_failures}</span>
                  <span>signature fail {item.signature_failures}</span>
                  <span>stale {item.stale_count}</span>
                </div>
                <div className="tiny">
                  Avg latency: {item.avg_latency_ms !== null && item.avg_latency_ms !== undefined ? `${item.avg_latency_ms.toFixed(1)} ms` : 'n/a'} | Last observed:{' '}
                  {item.last_observed_at_utc
                    ? new Date(item.last_observed_at_utc).toLocaleString()
                    : 'n/a'}
                </div>
              </li>
            ))}
            {dashboard.sources.length === 0 ? <li className="helper">No checks ingested yet.</li> : null}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
