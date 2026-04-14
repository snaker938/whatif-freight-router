'use client';

import { Chart as ChartJS, Legend, LinearScale, PointElement, Title, Tooltip } from 'chart.js';
import { Scatter } from 'react-chartjs-2';

import FieldInfo from './FieldInfo';
import { formatMetricTooltip, type MetricTooltip } from './metricTooltip';
import type { RouteOption } from '../lib/types';

ChartJS.register(LinearScale, PointElement, Tooltip, Legend, Title);

type Props = {
  routes: RouteOption[];
  selectedId: string | null;
  labelsById: Record<string, string>;
  onSelect: (routeId: string) => void;
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

export default function ParetoChart({ routes, selectedId, labelsById, onSelect }: Props) {
  const points = routes.map((r) => ({
    x: r.metrics.duration_s / 60.0,
    y: r.metrics.emissions_kg,
    id: r.id,
    label: labelsById[r.id] ?? r.id,
    money: r.metrics.monetary_cost,
    isKnee: Boolean(r.is_knee),
  }));

  const data = {
    datasets: [
      {
        label: 'Pareto candidates',
        data: points as any[],
        pointRadius: (ctx: any) =>
          ctx?.raw?.id === selectedId ? 8 : ctx?.raw?.isKnee ? 6 : 5,
        pointHoverRadius: 10,
        pointBackgroundColor: (ctx: any) => {
          if (ctx?.raw?.id === selectedId) return 'rgba(6, 182, 212, 0.95)';
          if (ctx?.raw?.isKnee) return 'rgba(245, 158, 11, 0.95)';
          return 'rgba(255, 255, 255, 0.70)';
        },
        pointBorderColor: (ctx: any) =>
          ctx?.raw?.id === selectedId ? 'rgba(124, 58, 237, 0.95)' : 'rgba(255, 255, 255, 0.18)',
        pointBorderWidth: (ctx: any) => (ctx?.raw?.id === selectedId || ctx?.raw?.isKnee ? 2 : 1),
      },
    ],
  };

  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 450 },
    plugins: {
      title: {
        display: false,
        text: 'Pareto space: time vs CO₂',
        color: 'rgba(255, 255, 255, 0.85)',
        font: { size: 12, weight: '600' },
        padding: { top: 8, bottom: 6 },
      },
      tooltip: {
        callbacks: {
          title: (items: any[]) => {
            const raw = items?.[0]?.raw;
            return raw?.label ?? raw?.id ?? 'Route';
          },
          label: (context: any) => {
            const raw = context.raw;
            return [
              `Travel time: ${raw.x.toFixed(1)} min`,
              `Emissions: ${raw.y.toFixed(3)} kg CO2`,
              `Cost: £${raw.money.toFixed(2)}`,
              `Knee-point: ${raw.isKnee ? 'yes' : 'no'}`,
            ];
          },
        },
        titleColor: 'rgba(255, 255, 255, 0.92)',
        bodyColor: 'rgba(255, 255, 255, 0.85)',
        backgroundColor: 'rgba(10, 12, 20, 0.85)',
        borderColor: 'rgba(255, 255, 255, 0.16)',
        borderWidth: 1,
      },
      legend: { display: false },
    },
    scales: {
      x: {
        title: { display: false, text: 'Travel time (min)', color: 'rgba(255, 255, 255, 0.70)' },
        ticks: { color: 'rgba(255, 255, 255, 0.65)' },
        grid: { color: 'rgba(255, 255, 255, 0.08)' },
      },
      y: {
        title: { display: false, text: 'Emissions (kg CO₂)', color: 'rgba(255, 255, 255, 0.70)' },
        ticks: { color: 'rgba(255, 255, 255, 0.65)' },
        grid: { color: 'rgba(255, 255, 255, 0.08)' },
      },
    },
    onClick: (_event: any, elements: any[]) => {
      if (!elements?.length) return;
      const idx = elements[0].index;
      const chosen = points[idx];
      if (chosen?.id) onSelect(chosen.id);
    },
  };

  return (
    <div>
      <div className="routeCard__meta" style={{ marginBottom: 8 }}>
        <span>
          {inlineMetricLabel('Pareto space', {
            definition: 'Scatter view of candidate routes in time-emissions tradeoff space.',
            direction: 'Lower-left is better when minimizing both axes together.',
            unit: 'chart view',
          })}
        </span>
        <span>
          {inlineMetricLabel('Travel time', {
            definition: 'Route duration shown on the chart x-axis.',
            direction: 'Lower is usually better because the route completes sooner.',
            unit: 'minutes',
          })}
        </span>
        <span>
          {inlineMetricLabel('Emissions', {
            definition: 'Route CO2 shown on the chart y-axis.',
            direction: 'Lower is usually better because less CO2 is emitted.',
            unit: 'kilograms CO2',
          })}
        </span>
        <span>
          {inlineMetricLabel('Hover details', {
            definition: 'Per-point hover summary for travel time, emissions, cost, and knee-point status.',
            direction: 'Descriptive only; hover details explain the selected candidate rather than ranking it.',
            unit: 'mixed route metrics',
          })}
        </span>
      </div>
      <div style={{ height: 220 }}>
        <Scatter data={data} options={options} />
      </div>
    </div>
  );
}
