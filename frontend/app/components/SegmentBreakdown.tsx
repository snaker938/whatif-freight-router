'use client';

import { useEffect, useState } from 'react';

import type { RouteOption } from '../lib/types';

type Props = {
  route: RouteOption | null;
};

export default function SegmentBreakdown({ route }: Props) {
  const segments = Array.isArray(route?.segment_breakdown) ? route?.segment_breakdown : [];
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setExpanded(false);
  }, [route?.id]);

  if (!segments?.length) return null;

  return (
    <div className="segmentBreakdown">
      <div className="segmentBreakdown__header">
        <div className="segmentBreakdown__titleWrap">
          <div className="fieldLabel" style={{ margin: 0 }}>
            Per-segment cost breakdown
          </div>
          <span className="segmentBreakdown__count">{segments.length} segments</span>
        </div>
        <button
          type="button"
          className="ghostButton segmentBreakdown__toggle"
          onClick={() => setExpanded((prev) => !prev)}
          aria-expanded={expanded}
          aria-controls="segment-breakdown-scroll-region"
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>
      {!expanded ? (
        <div className="segmentBreakdown__collapsedHint">
          Expand to view all segment rows with independent scrolling.
        </div>
      ) : null}
      <div
        id="segment-breakdown-scroll-region"
        className={`segmentBreakdown__scrollWrap ${expanded ? 'isOpen' : ''}`}
      >
        <table className="segmentBreakdown__table">
          <thead>
            <tr>
              <th>Seg</th>
              <th className="segmentBreakdown__right">Dist (km)</th>
              <th className="segmentBreakdown__right">ETA (s)</th>
              <th className="segmentBreakdown__right">Cost (£)</th>
              <th className="segmentBreakdown__right">CO2 (kg)</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((segment, idx) => (
              <tr key={`${idx}-${segment.segment_index ?? idx}`}>
                <td>{Number(segment.segment_index ?? idx) + 1}</td>
                <td className="segmentBreakdown__right">
                  {Number(segment.distance_km ?? 0).toFixed(3)}
                </td>
                <td className="segmentBreakdown__right">
                  {Number(segment.duration_s ?? 0).toFixed(2)}
                </td>
                <td className="segmentBreakdown__right">
                  {Number(segment.monetary_cost ?? 0).toFixed(3)}
                </td>
                <td className="segmentBreakdown__right">
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
