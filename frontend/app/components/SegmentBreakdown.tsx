'use client';

import { useEffect, useState } from 'react';

import type { RouteOption } from '../lib/types';

type Props = {
  route: RouteOption | null;
};

const PREVIEW_ROW_COUNT = 40;

export default function SegmentBreakdown({ route }: Props) {
  const segments = Array.isArray(route?.segment_breakdown) ? route?.segment_breakdown : [];
  const [expanded, setExpanded] = useState(false);
  const [showAllRows, setShowAllRows] = useState(false);

  useEffect(() => {
    setExpanded(false);
    setShowAllRows(false);
  }, [route?.id]);

  if (!segments?.length) return null;
  const visibleSegments = showAllRows ? segments : segments.slice(0, PREVIEW_ROW_COUNT);
  const hiddenRowCount = Math.max(0, segments.length - visibleSegments.length);

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
          Expand to view a scrollable segment preview (not all rows by default).
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
            {visibleSegments.map((segment, idx) => (
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
      {expanded && hiddenRowCount > 0 ? (
        <div className="segmentBreakdown__footer">
          <span className="segmentBreakdown__footerText">
            Showing first {visibleSegments.length} of {segments.length} segments.
          </span>
          <button
            type="button"
            className="ghostButton segmentBreakdown__footerBtn"
            onClick={() => setShowAllRows(true)}
          >
            Show all rows
          </button>
        </div>
      ) : null}
      {expanded && showAllRows && segments.length > PREVIEW_ROW_COUNT ? (
        <div className="segmentBreakdown__footer">
          <span className="segmentBreakdown__footerText">All segment rows are currently visible.</span>
          <button
            type="button"
            className="ghostButton segmentBreakdown__footerBtn"
            onClick={() => setShowAllRows(false)}
          >
            Show fewer rows
          </button>
        </div>
      ) : null}
    </div>
  );
}
