'use client';

import type { RouteOption } from '../lib/types';

type Props = {
  route: RouteOption | null;
};

export default function SegmentBreakdown({ route }: Props) {
  const segments = Array.isArray(route?.segment_breakdown) ? route?.segment_breakdown : [];
  if (!segments?.length) return null;

  return (
    <div style={{ marginTop: 12 }}>
      <div className="fieldLabel" style={{ marginBottom: 6 }}>
        Per-segment cost breakdown
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '6px 4px' }}>Seg</th>
              <th style={{ textAlign: 'right', padding: '6px 4px' }}>Dist (km)</th>
              <th style={{ textAlign: 'right', padding: '6px 4px' }}>ETA (s)</th>
              <th style={{ textAlign: 'right', padding: '6px 4px' }}>Cost (£)</th>
              <th style={{ textAlign: 'right', padding: '6px 4px' }}>CO2 (kg)</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((segment, idx) => (
              <tr key={`${idx}-${segment.segment_index ?? idx}`}>
                <td style={{ padding: '6px 4px' }}>{Number(segment.segment_index ?? idx) + 1}</td>
                <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                  {Number(segment.distance_km ?? 0).toFixed(3)}
                </td>
                <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                  {Number(segment.duration_s ?? 0).toFixed(2)}
                </td>
                <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                  {Number(segment.monetary_cost ?? 0).toFixed(3)}
                </td>
                <td style={{ textAlign: 'right', padding: '6px 4px' }}>
                  {Number(segment.emissions_kg ?? 0).toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

