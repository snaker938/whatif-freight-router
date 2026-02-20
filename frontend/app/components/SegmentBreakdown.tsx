'use client';

import { useEffect, useId, useState } from 'react';

import type { RouteOption } from '../lib/types';

type Props = {
  route: RouteOption | null;
  onTutorialAction?: (actionId: string) => void;
};

const PREVIEW_ROW_COUNT = 40;

export default function SegmentBreakdown({ route, onTutorialAction }: Props) {
  const segments = Array.isArray(route?.segment_breakdown) ? route?.segment_breakdown : [];
  const terrainSummary = route?.terrain_summary ?? null;
  const [expanded, setExpanded] = useState(false);
  const [showAllRows, setShowAllRows] = useState(false);
  const [copied, setCopied] = useState(false);
  const scrollRegionId = useId();

  useEffect(() => {
    setExpanded(false);
    setShowAllRows(false);
    setCopied(false);
  }, [route?.id]);

  if (!segments?.length) return null;
  const visibleSegments = showAllRows ? segments : segments.slice(0, PREVIEW_ROW_COUNT);
  const hiddenRowCount = Math.max(0, segments.length - visibleSegments.length);

  async function copyVisibleRowsCsv() {
    const header = ['segment_index', 'distance_km', 'duration_s', 'monetary_cost', 'emissions_kg'];
    const rows = visibleSegments.map((segment, idx) => [
      String(Number(segment.segment_index ?? idx) + 1),
      String(Number(segment.distance_km ?? 0)),
      String(Number(segment.duration_s ?? 0)),
      String(Number(segment.monetary_cost ?? 0)),
      String(Number(segment.emissions_kg ?? 0)),
    ]);
    const csv = [header, ...rows].map((line) => line.join(',')).join('\n');
    try {
      await navigator.clipboard.writeText(csv);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="segmentBreakdown" data-tutorial-id="selected.segment_breakdown">
      <div className="segmentBreakdown__header">
        <div className="segmentBreakdown__titleWrap">
          <div className="fieldLabel" style={{ margin: 0 }}>
            Per-Segment Cost Breakdown
          </div>
          <span className="segmentBreakdown__count">{segments.length} segments</span>
        </div>
        <button
          type="button"
          className="ghostButton segmentBreakdown__toggle"
          onClick={() => {
            setExpanded((prev) => {
              const next = !prev;
              if (!next) {
                setShowAllRows(false);
              }
              onTutorialAction?.(next ? 'selected.segment_expand' : 'selected.segment_collapse');
              return next;
            });
          }}
          aria-expanded={expanded}
          aria-controls={scrollRegionId}
          data-tutorial-action={expanded ? 'selected.segment_collapse' : 'selected.segment_expand'}
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>
      {terrainSummary ? (
        <div className="segmentBreakdown__collapsedHint">
          Terrain {terrainSummary.source} | Coverage {(terrainSummary.coverage_ratio * 100).toFixed(1)}% |
          Ascent {terrainSummary.ascent_m.toFixed(0)}m | Descent {terrainSummary.descent_m.toFixed(0)}m
        </div>
      ) : null}
      {!expanded ? (
        <div className="segmentBreakdown__collapsedHint">
          Expand To View A Scrollable Segment Preview (Not All Rows By Default).
        </div>
      ) : null}
      <div
        id={scrollRegionId}
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
            Showing First {visibleSegments.length} Of {segments.length} Segments.
          </span>
          <div className="segmentBreakdown__footerActions">
            <button
              type="button"
              className="ghostButton segmentBreakdown__footerBtn"
              onClick={copyVisibleRowsCsv}
            >
              {copied ? 'Copied' : 'Copy CSV'}
            </button>
            <button
              type="button"
              className="ghostButton segmentBreakdown__footerBtn"
              onClick={() => {
                setShowAllRows(true);
                onTutorialAction?.('selected.segment_show_all');
              }}
              data-tutorial-action="selected.segment_show_all"
            >
              Show All Rows
            </button>
          </div>
        </div>
      ) : null}
      {expanded && showAllRows && segments.length > PREVIEW_ROW_COUNT ? (
        <div className="segmentBreakdown__footer">
          <span className="segmentBreakdown__footerText">All Segment Rows Are Currently Visible.</span>
          <div className="segmentBreakdown__footerActions">
            <button
              type="button"
              className="ghostButton segmentBreakdown__footerBtn"
              onClick={copyVisibleRowsCsv}
            >
              {copied ? 'Copied' : 'Copy CSV'}
            </button>
            <button
              type="button"
              className="ghostButton segmentBreakdown__footerBtn"
              onClick={() => {
                setShowAllRows(false);
                onTutorialAction?.('selected.segment_show_fewer');
              }}
              data-tutorial-action="selected.segment_show_fewer"
            >
              Show Fewer Rows
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
