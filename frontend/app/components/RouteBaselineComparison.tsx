'use client';

import { useMemo } from 'react';

import type { BaselineComparison } from '../lib/baselineComparison';
import type { RouteOption } from '../lib/types';

type BaselineMeta = {
  method: string;
  compute_ms: number;
  notes: string[];
};

type LiveSummary = {
  total_calls: number;
  expected_satisfied: number;
  expected_total: number;
};

type Props = {
  title?: string;
  baselineLabel?: string;
  locale: string;
  smartRoute: RouteOption | null;
  baselineRoute: RouteOption | null;
  comparison: BaselineComparison | null;
  baselineLoading: boolean;
  baselineError: string | null;
  baselineMeta: BaselineMeta | null;
  smartComputeMs: number | null;
  candidateCount: number | null;
  liveSummary: LiveSummary | null;
};

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return 'n/a';
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const rem = total % 60;
  if (minutes === 0) return `${rem}s`;
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  }
  return rem > 0 ? `${minutes}m ${rem}s` : `${minutes}m`;
}

function formatMs(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return 'n/a';
  const seconds = ms / 1000;
  if (seconds < 1) return `${Math.round(ms)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return formatDuration(seconds);
}

function radarPoint(indexValue: number, angleRad: number, center: number, maxRadius: number): [number, number] {
  const radius = (clamp(indexValue, 50, 200) / 200) * maxRadius;
  return [center + radius * Math.cos(angleRad), center + radius * Math.sin(angleRad)];
}

function polygonPoints(values: number[], center: number, maxRadius: number): string {
  const angles = [-Math.PI / 2, 0, Math.PI / 2, Math.PI];
  return values
    .map((value, idx) => {
      const [x, y] = radarPoint(value, angles[idx], center, maxRadius);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
}

function ratioIndex(baselineValue: number, smartValue: number): number {
  const baseline = Math.max(baselineValue, 1e-6);
  const smart = Math.max(smartValue, 1e-6);
  return clamp((100 * baseline) / smart, 50, 200);
}

export default function RouteBaselineComparison({
  title = 'Baseline Comparison',
  baselineLabel = 'Baseline',
  locale,
  smartRoute,
  baselineRoute,
  comparison,
  baselineLoading,
  baselineError,
  baselineMeta,
  smartComputeMs,
  candidateCount,
  liveSummary,
}: Props) {
  const numberFmt2 = useMemo(
    () => new Intl.NumberFormat(locale, { maximumFractionDigits: 2, minimumFractionDigits: 0 }),
    [locale],
  );
  const numberFmt3 = useMemo(
    () => new Intl.NumberFormat(locale, { maximumFractionDigits: 3, minimumFractionDigits: 0 }),
    [locale],
  );
  const percentFmt = useMemo(
    () => new Intl.NumberFormat(locale, { maximumFractionDigits: 1, minimumFractionDigits: 0, signDisplay: 'exceptZero' }),
    [locale],
  );

  if (!smartRoute) return null;

  if (baselineLoading && !baselineRoute) {
    return (
      <section className="baselineComparePanel">
        <div className="baselineComparePanel__title">{title}</div>
        <div className="baselineComparePanel__loading">Fetching {baselineLabel.toLowerCase()} route...</div>
      </section>
    );
  }

  if (baselineError && !baselineRoute) {
    return (
      <section className="baselineComparePanel">
        <div className="baselineComparePanel__title">{title}</div>
        <div className="baselineComparePanel__warning">{baselineLabel} unavailable: {baselineError}</div>
      </section>
    );
  }

  if (!baselineRoute || !comparison) {
    return (
      <section className="baselineComparePanel">
        <div className="baselineComparePanel__title">{title}</div>
        <div className="baselineComparePanel__loading">{baselineLabel} is not available for this route yet.</div>
      </section>
    );
  }

  const etaBetter = comparison.etaDeltaS >= 0;
  const costBetter = comparison.costPct >= 0;
  const co2Better = comparison.co2Pct >= 0;
  const distanceBetter = comparison.distanceGainPct >= 0;
  const betterCount = [etaBetter, costBetter, co2Better, distanceBetter].filter(Boolean).length;

  const deltaRows = [
    { label: 'ETA improvement', value: comparison.etaPct },
    { label: 'Cost improvement', value: comparison.costPct },
    { label: 'CO2 improvement', value: comparison.co2Pct },
    { label: 'Distance improvement', value: comparison.distanceGainPct },
  ];
  const tradeoffUpsides: string[] = [];
  const tradeoffDownsides: string[] = [];
  if (etaBetter) tradeoffUpsides.push('faster ETA');
  else tradeoffDownsides.push('slower ETA');
  if (costBetter) tradeoffUpsides.push('lower cost');
  else tradeoffDownsides.push('higher cost');
  if (co2Better) tradeoffUpsides.push('lower CO2');
  else tradeoffDownsides.push('higher CO2');
  if (distanceBetter) tradeoffUpsides.push('shorter distance');
  else tradeoffDownsides.push('longer distance');
  const tradeoffSentence = `Selected by weighted objective: ${
    tradeoffUpsides.length ? tradeoffUpsides.join(', ') : 'no material improvements'
  }${
    tradeoffDownsides.length ? `, with tradeoffs in ${tradeoffDownsides.join(', ')}` : ''
  }.`;

  const smart = smartRoute.metrics;
  const baseline = baselineRoute.metrics;
  const smartRadar = [
    ratioIndex(baseline.duration_s, smart.duration_s),
    ratioIndex(baseline.monetary_cost, smart.monetary_cost),
    ratioIndex(baseline.emissions_kg, smart.emissions_kg),
    ratioIndex(baseline.distance_km, smart.distance_km),
  ];
  const baselineRadar = [100, 100, 100, 100];

  const smartPolygon = polygonPoints(smartRadar, 100, 76);
  const baselinePolygon = polygonPoints(baselineRadar, 100, 76);

  return (
    <section className="baselineComparePanel">
      <div className="baselineComparePanel__head">
        <div className="baselineComparePanel__title">{title}</div>
        <div className={`baselineEpicScore baselineEpicScore--${comparison.epicTier.replace(/\s+/g, '').toLowerCase()}`}>
          Epic score {comparison.epicScore} ({comparison.epicTier})
        </div>
      </div>
      <div className="baselineComparePanel__epicNote">
        Epic score is balanced across ETA, cost, and CO2. It is not ETA-only.
      </div>
      <div className="baselineComparePanel__epicNote">
        Net result: better on {betterCount}/4 metrics versus {baselineLabel}.
      </div>
      <div className="baselineComparePanel__signLegend">
        Percentages here are improvement vs {baselineLabel.toLowerCase()}: <strong>+ means better</strong>,{' '}
        <strong>- means worse</strong>. For ETA and distance, positive means faster/shorter.
      </div>

      {baselineError ? <div className="baselineComparePanel__warning">{baselineError}</div> : null}

      <div className="baselineKpiGrid">
        <div className={`baselineKpi ${etaBetter ? 'isPositive' : 'isNegative'}`}>
          <div className="baselineKpi__label">ETA improvement vs baseline</div>
          <div className="baselineKpi__value">
            {formatDuration(Math.abs(comparison.etaDeltaS))} {etaBetter ? 'faster' : 'slower'}
          </div>
          <div className="baselineKpi__meta">
            Improvement {percentFmt.format(comparison.etaPct)}% ({etaBetter ? 'Better' : 'Worse'})
          </div>
        </div>
        <div className={`baselineKpi ${costBetter ? 'isPositive' : 'isNegative'}`}>
          <div className="baselineKpi__label">Cost improvement vs baseline</div>
          <div className="baselineKpi__value">
            {numberFmt2.format(Math.abs(comparison.costDelta))} {costBetter ? 'lower' : 'higher'}
          </div>
          <div className="baselineKpi__meta">
            Improvement {percentFmt.format(comparison.costPct)}% ({costBetter ? 'Better' : 'Worse'})
          </div>
        </div>
        <div className={`baselineKpi ${co2Better ? 'isPositive' : 'isNegative'}`}>
          <div className="baselineKpi__label">CO2 improvement vs baseline</div>
          <div className="baselineKpi__value">
            {numberFmt3.format(Math.abs(comparison.co2Delta))} kg {co2Better ? 'lower' : 'higher'}
          </div>
          <div className="baselineKpi__meta">
            Improvement {percentFmt.format(comparison.co2Pct)}% ({co2Better ? 'Better' : 'Worse'})
          </div>
        </div>
        <div className={`baselineKpi ${distanceBetter ? 'isPositive' : 'isNegative'}`}>
          <div className="baselineKpi__label">Distance improvement vs baseline</div>
          <div className="baselineKpi__value">
            {numberFmt2.format(Math.abs(comparison.distanceDeltaKm))} km{' '}
            {distanceBetter ? 'shorter' : 'longer'}
          </div>
          <div className="baselineKpi__meta">
            Improvement {percentFmt.format(comparison.distanceGainPct)}% ({distanceBetter ? 'Better' : 'Worse'})
          </div>
        </div>
      </div>
      <div className="baselineComparePanel__tradeoff">{tradeoffSentence}</div>

      <div className="baselineImpactGrid">
        <div>
          <div className="baselineImpactGrid__label">Smart compute elapsed</div>
          <div className="baselineImpactGrid__value">{formatMs(smartComputeMs)}</div>
        </div>
        <div>
          <div className="baselineImpactGrid__label">Baseline fetch elapsed</div>
          <div className="baselineImpactGrid__value">{formatMs(baselineMeta?.compute_ms ?? null)}</div>
        </div>
        <div>
          <div className="baselineImpactGrid__label">Smart candidates</div>
          <div className="baselineImpactGrid__value">
            {candidateCount !== null && Number.isFinite(candidateCount) ? numberFmt2.format(candidateCount) : 'n/a'}
          </div>
        </div>
        <div>
          <div className="baselineImpactGrid__label">Live-source coverage</div>
          <div className="baselineImpactGrid__value">
            {liveSummary
              ? `${numberFmt2.format(liveSummary.expected_satisfied)}/${numberFmt2.format(liveSummary.expected_total)}`
              : 'n/a'}
          </div>
        </div>
      </div>
      {liveSummary ? (
        <div className="baselineImpactGrid__trace">Live calls observed: {numberFmt2.format(liveSummary.total_calls)}</div>
      ) : null}

      <div className="baselineChartsGrid">
        <div className="baselineChartCard">
          <div className="baselineChartCard__title">Improvement bars (% vs baseline, + is better)</div>
          <div className="baselineDeltaBars">
            {deltaRows.map((row) => {
              const value = clamp(row.value, -100, 100);
              const widthPct = Math.abs(value) * 0.5;
              const positive = value >= 0;
              return (
                <div key={row.label} className="baselineDeltaBars__row">
                  <div className="baselineDeltaBars__label">{row.label}</div>
                  <div className="baselineDeltaBars__track">
                    <div className="baselineDeltaBars__zero" />
                    <div
                      className={`baselineDeltaBars__bar ${positive ? 'isPositive' : 'isNegative'}`}
                      style={{
                        width: `${widthPct}%`,
                        left: positive ? '50%' : `${50 - widthPct}%`,
                      }}
                    />
                  </div>
                  <div className="baselineDeltaBars__value">
                    Improvement {percentFmt.format(row.value)}% ({positive ? 'Better' : 'Worse'})
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="baselineChartCard">
          <div className="baselineChartCard__title">Smart vs {baselineLabel} radar ({baselineLabel}=100)</div>
          <svg viewBox="0 0 200 200" className="baselineRadar" role="img" aria-label="Radar comparison">
            <circle cx="100" cy="100" r="19" className="baselineRadar__ring" />
            <circle cx="100" cy="100" r="38" className="baselineRadar__ring" />
            <circle cx="100" cy="100" r="57" className="baselineRadar__ring" />
            <circle cx="100" cy="100" r="76" className="baselineRadar__ring" />
            <line x1="100" y1="24" x2="100" y2="176" className="baselineRadar__axis" />
            <line x1="24" y1="100" x2="176" y2="100" className="baselineRadar__axis" />
            <polygon points={baselinePolygon} className="baselineRadar__baselinePoly" />
            <polygon points={smartPolygon} className="baselineRadar__smartPoly" />
            <text x="100" y="14" textAnchor="middle" className="baselineRadar__label">ETA</text>
            <text x="186" y="104" textAnchor="start" className="baselineRadar__label">Cost</text>
            <text x="100" y="194" textAnchor="middle" className="baselineRadar__label">CO2</text>
            <text x="14" y="104" textAnchor="end" className="baselineRadar__label">Dist</text>
          </svg>
          <div className="baselineRadarLegend">
            <span className="baselineRadarLegend__item baselineRadarLegend__item--smart">Smart route</span>
            <span className="baselineRadarLegend__item baselineRadarLegend__item--baseline">{baselineLabel}</span>
          </div>
        </div>
      </div>

      {baselineMeta?.notes?.length ? (
        <ul className="baselineNotes">
          {baselineMeta.notes.map((note, idx) => (
            <li key={`${idx}-${note}`}>{note}</li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
