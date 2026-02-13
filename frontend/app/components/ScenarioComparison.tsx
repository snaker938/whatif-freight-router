'use client';

import { formatNumber } from '../lib/format';
import type { Locale } from '../lib/i18n';
import type { ScenarioCompareResponse } from '../lib/types';

type Props = {
  data: ScenarioCompareResponse | null;
  loading: boolean;
  error: string | null;
  locale: Locale;
};

function fmtDelta(value: number | undefined, locale: Locale): string {
  if (value === undefined) return '-';
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${formatNumber(value, locale, { maximumFractionDigits: 2 })}`;
}

function scenarioLabel(mode: string): string {
  if (mode === 'no_sharing') return 'No Sharing';
  if (mode === 'partial_sharing') return 'Partial Sharing';
  if (mode === 'full_sharing') return 'Full Sharing';
  return mode;
}

export default function ScenarioComparison({ data, loading, error, locale }: Props) {
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
              const delta = data.deltas[result.scenario_mode] ?? {};
              const isBestEta = bestEtaMode === result.scenario_mode;
              return (
                <tr
                  key={result.scenario_mode}
                  className={isBestEta ? 'compareTable__row compareTable__row--best' : 'compareTable__row'}
                >
                  <td style={{ padding: '6px 4px' }}>{scenarioLabel(result.scenario_mode)}</td>
                  <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                    {fmtDelta(delta.duration_s_delta, locale)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                    {fmtDelta(delta.monetary_cost_delta, locale)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                    {fmtDelta(delta.emissions_kg_delta, locale)}
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
    </div>
  );
}
