'use client';

import FieldInfo from './FieldInfo';
import { formatMetricTooltip, type MetricTooltip } from './metricTooltip';
import type { RouteOption } from '../lib/types';

type Props = {
  route: RouteOption | null;
};

const inlineMetricLabelStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
} as const;

function metricHelp(tooltip: MetricTooltip): string {
  return formatMetricTooltip(tooltip);
}

function inlineMetricLabel(label: string, tooltip: MetricTooltip) {
  return (
    <span style={inlineMetricLabelStyle}>
      <span>{label}</span>
      <FieldInfo text={metricHelp(tooltip)} />
    </span>
  );
}

function toNum(value: string | number | boolean | undefined): number | null {
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export default function CounterfactualPanel({ route }: Props) {
  const rows = Array.isArray(route?.counterfactuals) ? route?.counterfactuals : [];
  if (!rows.length) return null;

  return (
    <div style={{ marginTop: 12 }}>
      <div className="fieldLabel" style={{ marginBottom: 6 }}>
        Counterfactuals
      </div>
      <ul className="routeList" style={{ marginTop: 6 }}>
        {rows.map((row, idx) => {
          const label = String(row.label ?? row.id ?? `Counterfactual ${idx + 1}`);
          const metric = String(row.metric ?? 'value');
          const delta = toNum(row.delta);
          const improves = Boolean(row.improves);
          return (
            <li key={`${idx}-${label}`} className="routeCard" style={{ cursor: 'default' }}>
              <div className="routeCard__top">
                <div className="routeCard__id">{label}</div>
                {delta !== null ? (
                  <div className="routeCard__pill">
                    {inlineMetricLabel('Delta', {
                      definition: 'Counterfactual change recorded for this scenario relative to the selected route outcome.',
                      direction: 'Interpretation depends on the named metric; the sign shows whether the counterfactual increased or decreased that metric.',
                      unit: `metric-specific unit for ${metric}`,
                    })}{' '}
                    {delta > 0 ? '+' : ''}
                    {delta.toFixed(3)} {metric}
                  </div>
                ) : null}
              </div>
              <div className="routeCard__meta">
                <span>
                  {inlineMetricLabel('Metric', {
                    definition: 'Underlying metric affected by this counterfactual change.',
                    direction: 'Context only; this names the affected metric rather than ranking the route.',
                    unit: 'metric identifier',
                  })}{' '}
                  {metric}
                </span>
                <span>
                  {inlineMetricLabel('Outcome effect', {
                    definition: 'Whether the counterfactual was marked as improving or worsening the selected outcome.',
                    direction: 'Improves outcome is better; worse outcome is worse.',
                    unit: 'categorical effect label',
                  })}{' '}
                  {improves ? 'Improves outcome' : 'Worse outcome'}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
