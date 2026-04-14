'use client';

import { useEffect, useMemo, useState } from 'react';

import CollapsibleCard from './CollapsibleCard';
import FieldInfo from './FieldInfo';
import { formatMetricTooltip, type MetricTooltip } from './metricTooltip';
import Select, { type SelectOption } from './Select';
import { formatDateTime, formatNumber } from '../lib/format';
import type { Locale } from '../lib/i18n';
import {
  SIDEBAR_DROPDOWN_OPTIONS_HELP,
  SIDEBAR_FIELD_HELP,
  SIDEBAR_SECTION_HINTS,
} from '../lib/sidebarHelpText';
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
  locale: Locale;
  tutorialResetNonce?: number;
  sectionControl?: {
    isOpen?: boolean;
    lockToggle?: boolean;
    tutorialLocked?: boolean;
  };
};

type SignatureState = 'unknown' | 'valid' | 'invalid';

const SIGNATURE_STATE_OPTIONS: SelectOption<SignatureState>[] = [
  { value: 'unknown', label: 'Unknown', description: 'Signature check not provided.' },
  { value: 'valid', label: 'Valid', description: 'Signature check passed.' },
  { value: 'invalid', label: 'Invalid', description: 'Signature check failed.' },
];

const inlineMetricLabelStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
} as const;

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

export default function OracleQualityDashboard({
  dashboard,
  loading,
  ingesting,
  error,
  latestCheck,
  disabled,
  onRefresh,
  onIngest,
  locale,
  tutorialResetNonce,
  sectionControl,
}: Props) {
  const [source, setSource] = useState('oracle_demo');
  const [schemaValid, setSchemaValid] = useState(true);
  const [signatureState, setSignatureState] = useState<SignatureState>('unknown');
  const [freshnessS, setFreshnessS] = useState('');
  const [latencyMs, setLatencyMs] = useState('');
  const [recordCount, setRecordCount] = useState('');
  const [errorText, setErrorText] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);

  const csvHref = useMemo(() => '/api/oracle/quality/dashboard.csv', []);

  useEffect(() => {
    if (typeof tutorialResetNonce !== 'number' || tutorialResetNonce <= 0) return;
    applySampleValues();
  }, [tutorialResetNonce]);

  function applySampleValues() {
    setSource('tutorial_oracle');
    setSchemaValid(true);
    setSignatureState('valid');
    setFreshnessS('120');
    setLatencyMs('85');
    setRecordCount('18');
    setErrorText('');
    setValidationError(null);
  }

  function clearFormValues() {
    setSource('');
    setSchemaValid(true);
    setSignatureState('unknown');
    setFreshnessS('');
    setLatencyMs('');
    setRecordCount('');
    setErrorText('');
    setValidationError(null);
  }

  async function handleIngest() {
    const trimmedSource = source.trim();
    if (!trimmedSource) {
      setValidationError('Source is required.');
      return;
    }

    const parsedFreshness =
      freshnessS.trim() === '' ? null : Number.isFinite(Number(freshnessS)) ? Number(freshnessS) : NaN;
    const parsedLatency =
      latencyMs.trim() === '' ? null : Number.isFinite(Number(latencyMs)) ? Number(latencyMs) : NaN;
    const parsedCount =
      recordCount.trim() === '' ? null : Number.isFinite(Number(recordCount)) ? Number(recordCount) : NaN;

    if (Number.isNaN(parsedFreshness) || (parsedFreshness !== null && parsedFreshness < 0)) {
      setValidationError('Freshness seconds must be a non-negative number.');
      return;
    }
    if (Number.isNaN(parsedLatency) || (parsedLatency !== null && parsedLatency < 0)) {
      setValidationError('Latency ms must be a non-negative number.');
      return;
    }
    if (
      Number.isNaN(parsedCount) ||
      (parsedCount !== null && (!Number.isInteger(parsedCount) || parsedCount < 0))
    ) {
      setValidationError('Record count must be a non-negative integer.');
      return;
    }

    setValidationError(null);
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
    <CollapsibleCard
      title="Oracle Quality Dashboard"
      hint={SIDEBAR_SECTION_HINTS.oracleQualityDashboard}
      dataTutorialId="oracle.section"
      isOpen={sectionControl?.isOpen}
      lockToggle={sectionControl?.lockToggle}
      tutorialLocked={sectionControl?.tutorialLocked}
    >
      <div className="helper">
        Record feed checks and inspect source-level pass rates, freshness, and signature health.
      </div>

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="oracle-source">
          Source
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.oracleSource} />
      </div>
      <input
        id="oracle-source"
        className="input"
        value={source}
        disabled={disabled || ingesting}
        onChange={(event) => {
          setSource(event.target.value);
          if (validationError) setValidationError(null);
        }}
        data-tutorial-action="oracle.source_input"
      />

      <div className="checkboxRow">
        <input
          id="oracle-schema-valid"
          type="checkbox"
          checked={schemaValid}
          disabled={disabled || ingesting}
          onChange={(event) => {
            setSchemaValid(event.target.checked);
            if (validationError) setValidationError(null);
          }}
          data-tutorial-action="oracle.schema_toggle"
        />
        <label htmlFor="oracle-schema-valid">Schema Valid</label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.schemaValid} />
      </div>

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="oracle-signature-state">
          Signature State
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.signatureState} />
      </div>
      <Select
        id="oracle-signature-state"
        ariaLabel="Signature state"
        value={signatureState}
        options={SIGNATURE_STATE_OPTIONS}
        disabled={disabled || ingesting}
        onChange={(next) => {
          setSignatureState(next);
          if (validationError) setValidationError(null);
        }}
        tutorialAction="oracle.signature_select"
      />
      <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.signatureState}</div>

      <div className="advancedGrid">
        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="oracle-freshness">
            Freshness Seconds (Optional)
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.freshnessSeconds} />
        </div>
        <input
          id="oracle-freshness"
          className="input"
          type="number"
          min={0}
          step="any"
          value={freshnessS}
          disabled={disabled || ingesting}
          onChange={(event) => {
            setFreshnessS(event.target.value);
            if (validationError) setValidationError(null);
          }}
          data-tutorial-action="oracle.freshness_input"
        />
        <div className="dropdownOptionsHint">Freshness Shows How Old Source Data Is At Check Time.</div>

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="oracle-latency">
            Latency ms (Optional)
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.latencyMs} />
        </div>
        <input
          id="oracle-latency"
          className="input"
          type="number"
          min={0}
          step="any"
          value={latencyMs}
          disabled={disabled || ingesting}
          onChange={(event) => {
            setLatencyMs(event.target.value);
            if (validationError) setValidationError(null);
          }}
          data-tutorial-action="oracle.latency_input"
        />

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="oracle-record-count">
            Record Count (Optional)
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.recordCount} />
        </div>
        <input
          id="oracle-record-count"
          className="input"
          type="number"
          min={0}
          step={1}
          value={recordCount}
          disabled={disabled || ingesting}
          onChange={(event) => {
            setRecordCount(event.target.value);
            if (validationError) setValidationError(null);
          }}
          data-tutorial-action="oracle.record_count_input"
        />
      </div>

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="oracle-error-note">
          Error Note (Optional)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.errorNote} />
      </div>
      <input
        id="oracle-error-note"
        className="input"
        value={errorText}
        disabled={disabled || ingesting}
        onChange={(event) => setErrorText(event.target.value)}
        data-tutorial-action="oracle.error_note_input"
      />

      <div className="actionGrid" style={{ marginTop: 12 }}>
        <button type="button"
          className="secondary"
          onClick={applySampleValues}
          disabled={disabled || loading || ingesting}
          data-tutorial-action="oracle.load_sample_click"
        >
          Load Sample
        </button>
        <button type="button"
          className="secondary"
          onClick={onRefresh}
          disabled={disabled || loading || ingesting}
          aria-label="Refresh oracle quality dashboard"
          data-tutorial-action="oracle.refresh_click"
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
        <a
          className="buttonLink"
          href={csvHref}
          aria-label="Download oracle dashboard CSV"
          data-tutorial-action="oracle.download_csv_click"
          target="_blank"
          rel="noreferrer"
        >
          Download CSV
        </a>
        <button type="button"
          className="secondary"
          onClick={clearFormValues}
          disabled={disabled || loading || ingesting}
          data-tutorial-action="oracle.clear_form_click"
        >
          Clear Form
        </button>
        <button type="button"
          className="secondary"
          onClick={handleIngest}
          disabled={disabled || ingesting || loading}
          data-tutorial-action="oracle.record_check_click"
        >
          {ingesting ? 'Recording...' : 'Record Check'}
        </button>
      </div>

      {error ? <div className="error">{error}</div> : null}
      {validationError ? <div className="error">{validationError}</div> : null}
      {latestCheck ? (
        <div className="tiny">
          Last Check: {latestCheck.source} ({latestCheck.passed ? 'Passed' : 'Failed'}) At{' '}
          {formatDateTime(latestCheck.ingested_at_utc, locale)}
        </div>
      ) : null}

      {dashboard ? (
        <div style={{ marginTop: 10 }}>
          <div className="metrics">
            <div className="metric">
              <div className="metric__label">
                {metricLabel('Total Checks', {
                  definition: 'Total oracle feed checks recorded in the dashboard summary.',
                  direction: 'Context only; more checks mean more monitoring history, not automatically better quality.',
                  unit: 'check count',
                })}
              </div>
              <div className="metric__value">
                {formatNumber(dashboard.total_checks, locale, { maximumFractionDigits: 0 })}
              </div>
            </div>
            <div className="metric">
              <div className="metric__label">
                {metricLabel('Sources', {
                  definition: 'Number of distinct oracle sources represented in the current dashboard summary.',
                  direction: 'Context only; more sources mean broader monitored coverage, not automatically better quality.',
                  unit: 'source count',
                })}
              </div>
              <div className="metric__value">
                {formatNumber(dashboard.source_count, locale, { maximumFractionDigits: 0 })}
              </div>
            </div>
            <div className="metric">
              <div className="metric__label">
                {metricLabel('Stale Threshold', {
                  definition: 'Freshness cutoff used to classify oracle checks as stale.',
                  direction: 'Context only; this is the configured monitoring threshold rather than a better/worse score.',
                  unit: 'seconds',
                })}
              </div>
              <div className="metric__value">
                {formatNumber(dashboard.stale_threshold_s, locale, { maximumFractionDigits: 0 })} s
              </div>
            </div>
          </div>

          <ul className="routeList" style={{ marginTop: 10 }}>
            {dashboard.sources.map((item) => (
              <li key={item.source} className="routeCard" style={{ cursor: 'default' }}>
                <div className="routeCard__top">
                  <div className="routeCard__id">{item.source}</div>
                  <div className="routeCard__pill">
                    {inlineMetricLabel('Pass rate', {
                      definition: 'Share of recorded checks for this source that passed.',
                      direction: 'Higher is better because more checks are succeeding.',
                      unit: 'pass-rate percent',
                    })}{' '}
                    {formatNumber(item.pass_rate * 100, locale, { maximumFractionDigits: 1 })}% Pass
                  </div>
                </div>
                <div className="routeCard__meta">
                  <span>
                    {inlineMetricLabel('Checks', {
                      definition: 'Number of recorded checks for this source.',
                      direction: 'Context only; more checks mean more monitoring history, not automatically better quality.',
                      unit: 'check count',
                    })}{' '}
                    {formatNumber(item.check_count, locale, { maximumFractionDigits: 0 })}
                  </span>
                  <span>
                    {inlineMetricLabel('Schema Fail', {
                      definition: 'Count of checks for this source that failed schema validation.',
                      direction: 'Lower is better because fewer schema failures occurred.',
                      unit: 'failure count',
                    })}{' '}
                    {formatNumber(item.schema_failures, locale, { maximumFractionDigits: 0 })}
                  </span>
                  <span>
                    {inlineMetricLabel('Signature Fail', {
                      definition: 'Count of checks for this source that failed signature validation.',
                      direction: 'Lower is better because fewer signature failures occurred.',
                      unit: 'failure count',
                    })}{' '}
                    {formatNumber(item.signature_failures, locale, { maximumFractionDigits: 0 })}
                  </span>
                  <span>
                    {inlineMetricLabel('Stale', {
                      definition: 'Count of checks for this source that exceeded the stale threshold.',
                      direction: 'Lower is better because fewer stale checks occurred.',
                      unit: 'stale-check count',
                    })}{' '}
                    {formatNumber(item.stale_count, locale, { maximumFractionDigits: 0 })}
                  </span>
                </div>
                <div className="tiny">
                  {inlineMetricLabel('Avg Latency', {
                    definition: 'Average observed latency for checks from this source.',
                    direction: 'Lower is better because checks complete faster.',
                    unit: 'milliseconds',
                  })}:{' '}
                  {item.avg_latency_ms !== null && item.avg_latency_ms !== undefined
                    ? `${formatNumber(item.avg_latency_ms, locale, { maximumFractionDigits: 1 })} ms`
                    : 'n/a'}{' '}
                  | Last Observed:{' '}
                  {item.last_observed_at_utc
                    ? formatDateTime(item.last_observed_at_utc, locale)
                    : 'n/a'}
                </div>
              </li>
            ))}
            {dashboard.sources.length === 0 ? <li className="helper">No Checks Ingested Yet.</li> : null}
          </ul>
        </div>
      ) : null}
    </CollapsibleCard>
  );
}
