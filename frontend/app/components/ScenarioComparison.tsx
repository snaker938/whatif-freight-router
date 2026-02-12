'use client';

import type { ScenarioCompareResponse } from '../lib/types';

type Props = {
  data: ScenarioCompareResponse | null;
  loading: boolean;
  error: string | null;
};

function fmtDelta(value: number | undefined): string {
  if (value === undefined) return '-';
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}`;
}

export default function ScenarioComparison({ data, loading, error }: Props) {
  if (loading) {
    return <div className="helper">Comparing scenarios...</div>;
  }
  if (error) {
    return <div className="error">{error}</div>;
  }
  if (!data) {
    return <div className="helper">Run comparison to view No/Partial/Full scenario deltas.</div>;
  }

  return (
    <div>
      <div className="tiny" style={{ marginBottom: 8 }}>
        Compare run: {data.run_id}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
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
              return (
                <tr key={result.scenario_mode}>
                  <td style={{ padding: '6px 4px' }}>{result.scenario_mode}</td>
                  <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                    {fmtDelta(delta.duration_s_delta)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                    {fmtDelta(delta.monetary_cost_delta)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                    {fmtDelta(delta.emissions_kg_delta)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="tiny" style={{ marginTop: 8 }}>
        Manifest: {data.scenario_manifest_endpoint}
      </div>
    </div>
  );
}

