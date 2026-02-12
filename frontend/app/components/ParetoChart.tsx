'use client';

import { Chart as ChartJS, Legend, LinearScale, PointElement, Title, Tooltip } from 'chart.js';
import { Scatter } from 'react-chartjs-2';

import type { RouteOption } from '../lib/types';

ChartJS.register(LinearScale, PointElement, Tooltip, Legend, Title);

type Props = {
  routes: RouteOption[];
  selectedId: string | null;
  labelsById: Record<string, string>;
  onSelect: (routeId: string) => void;
};

export default function ParetoChart({ routes, selectedId, labelsById, onSelect }: Props) {
  const points = routes.map((r) => ({
    x: r.metrics.duration_s / 60.0,
    y: r.metrics.emissions_kg,
    id: r.id,
    label: labelsById[r.id] ?? r.id,
    money: r.metrics.monetary_cost,
  }));

  const data = {
    datasets: [
      {
        label: 'Pareto candidates',
        data: points as any[],
        pointRadius: (ctx: any) => (ctx?.raw?.id === selectedId ? 8 : 5),
        pointHoverRadius: 10,
        pointBackgroundColor: (ctx: any) =>
          ctx?.raw?.id === selectedId ? 'rgba(6, 182, 212, 0.95)' : 'rgba(255, 255, 255, 0.70)',
        pointBorderColor: (ctx: any) =>
          ctx?.raw?.id === selectedId ? 'rgba(124, 58, 237, 0.95)' : 'rgba(255, 255, 255, 0.18)',
        pointBorderWidth: (ctx: any) => (ctx?.raw?.id === selectedId ? 2 : 1),
      },
    ],
  };

  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 450 },
    plugins: {
      title: {
        display: true,
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
            return `time=${raw.x.toFixed(1)} min, CO₂=${raw.y.toFixed(3)} kg, £=${raw.money.toFixed(2)}`;
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
        title: { display: true, text: 'Travel time (min)', color: 'rgba(255, 255, 255, 0.70)' },
        ticks: { color: 'rgba(255, 255, 255, 0.65)' },
        grid: { color: 'rgba(255, 255, 255, 0.08)' },
      },
      y: {
        title: { display: true, text: 'Emissions (kg CO₂)', color: 'rgba(255, 255, 255, 0.70)' },
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
    <div style={{ height: 220 }}>
      <Scatter data={data} options={options} />
    </div>
  );
}
