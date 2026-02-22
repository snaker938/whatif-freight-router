'use client';

import { formatNumber } from '../lib/format';
import type { Locale } from '../lib/i18n';
import type { ScenarioCompareResponse } from '../lib/types';

type Props = {
  data: ScenarioCompareResponse | null;
  loading: boolean;
  error: string | null;
  locale: Locale;
  onInspectScenarioManifest?: (runId: string) => void;
  onInspectScenarioSignature?: (runId: string) => void;
  onOpenRunInspector?: (runId: string) => void;
};

function fmtDelta(value: number | null | undefined, locale: Locale, status?: string): string {
  if (status === 'missing') return 'n/a';
  if (value === undefined || value === null) return '-';
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${formatNumber(value, locale, { maximumFractionDigits: 2 })}`;
}

function scenarioLabel(mode: string): string {
  if (mode === 'no_sharing') return 'No Sharing';
  if (mode === 'partial_sharing') return 'Partial Sharing';
  if (mode === 'full_sharing') return 'Full Sharing';
  return mode;
}

function deltaReason(
  status: string | undefined,
  reasonCode: string | null | undefined,
  missingSource: string | null | undefined,
  reasonSource: string | null | undefined,
): string | undefined {
  if (status !== 'missing') return undefined;
  const tokens = [reasonCode, missingSource, reasonSource].filter(Boolean);
  if (!tokens.length) return 'metric_unavailable';
  return tokens.join(' | ');
}

export default function ScenarioComparison({
  data,
  loading,
  error,
  locale,
  onInspectScenarioManifest,
  onInspectScenarioSignature,
  onOpenRunInspector,
}: Props) {
  if (loading) {
    return <div className="helper">Comparing Scenarios...</div>;
  }
  if (error) {
    return <div className="error">{error}</div>;
  }
  if (!data) {
    return <div className="helper">Run Comparison To View No/Partial/Full Scenario Deltas.</div>;
  }
  if (!data.results?.length) {
    return <div className="helper">No Scenario Comparison Rows Were Returned For This Run.</div>;
  }
  const bestEtaMode = data.results.reduce<{ mode: string | null; duration: number }>(
    (acc, result) => {
      const duration = result.selected?.metrics?.duration_s;
      if (typeof duration !== 'number' || !Number.isFinite(duration)) return acc;
      if (duration < acc.duration) return { mode: result.scenario_mode, duration };
      return acc;
    },
    { mode: null, duration: Number.POSITIVE_INFINITY },
  ).mode;

  return (
    <div>
      <div className="tiny" style={{ marginBottom: 8 }}>
        Compare Run: {data.run_id}
      </div>
      <div className="tiny" style={{ marginBottom: 8 }}>
        Delta Baseline: {scenarioLabel(data.baseline_mode ?? 'no_sharing')}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <caption className="srOnly">Scenario Comparison Results Table</caption>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '6px 4px' }}>Scenario</th>
              <th style={{ textAlign: 'right', padding: '6px 4px' }}>ETA delta (s)</th>
              <th style={{ textAlign: 'right', padding: '6px 4px' }}>Cost delta (£)</th>
              <th style={{ textAlign: 'right', padding: '6px 4px' }}>CO2 delta (kg)</th>
            </tr>
          </thead>
          <tbody>
            {data.results.map((result) => {
              const delta =
                data.deltas[result.scenario_mode] ??
                ({} as NonNullable<ScenarioCompareResponse['deltas'][string]>);
              const isBestEta = bestEtaMode === result.scenario_mode;
              const durationReason = deltaReason(
                delta.duration_s_status,
                delta.duration_s_reason_code,
                delta.duration_s_missing_source,
                delta.duration_s_reason_source,
              );
              const moneyReason = deltaReason(
                delta.monetary_cost_status,
                delta.monetary_cost_reason_code,
                delta.monetary_cost_missing_source,
                delta.monetary_cost_reason_source,
              );
              const emissionsReason = deltaReason(
                delta.emissions_kg_status,
                delta.emissions_kg_reason_code,
                delta.emissions_kg_missing_source,
                delta.emissions_kg_reason_source,
              );
              return (
                <tr
                  key={result.scenario_mode}
                  className={isBestEta ? 'compareTable__row compareTable__row--best' : 'compareTable__row'}
                >
                  <td style={{ padding: '6px 4px' }}>{scenarioLabel(result.scenario_mode)}</td>
                  <td style={{ textAlign: 'right', padding: '6px 4px' }} title={durationReason}>
                    {fmtDelta(delta.duration_s_delta, locale, delta.duration_s_status)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '6px 4px' }} title={moneyReason}>
                    {fmtDelta(delta.monetary_cost_delta, locale, delta.monetary_cost_status)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '6px 4px' }} title={emissionsReason}>
                    {fmtDelta(delta.emissions_kg_delta, locale, delta.emissions_kg_status)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {bestEtaMode ? (
        <div className="tiny" style={{ marginTop: 8 }}>
          Fastest Scenario: {scenarioLabel(bestEtaMode)}
        </div>
      ) : null}
      <div className="tiny" style={{ marginTop: 8 }}>
        Manifest: {data.scenario_manifest_endpoint}
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <button
          type="button"
          className="secondary"
          onClick={() => onInspectScenarioManifest?.(data.run_id)}
          disabled={!onInspectScenarioManifest}
        >
          Inspect Scenario Manifest
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => onInspectScenarioSignature?.(data.run_id)}
          disabled={!onInspectScenarioSignature}
        >
          Inspect Scenario Signature
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => onOpenRunInspector?.(data.run_id)}
          disabled={!onOpenRunInspector}
        >
          Open Run Inspector
        </button>
      </div>
    </div>
  );
}
