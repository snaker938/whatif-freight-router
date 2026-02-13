'use client';

import { Chart as ChartJS, Legend, LineElement, LinearScale, PointElement, Tooltip } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { CategoryScale } from 'chart.js';

import type { RouteOption } from '../lib/types';

ChartJS.register(CategoryScale, LineElement, PointElement, LinearScale, Tooltip, Legend);

type Props = {
  route: RouteOption | null;
};

export default function EtaTimelineChart({ route }: Props) {
  const timeline = Array.isArray(route?.eta_timeline) ? route?.eta_timeline : [];
  if (!timeline?.length) {
    return null;
  }

  const labels = timeline.map((entry) => String(entry.stage ?? 'stage'));
  const durationsMin = timeline.map((entry) => Number(entry.duration_s ?? 0) / 60.0);

  const data = {
    labels,
    datasets: [
      {
        label: 'ETA progression (min)',
        data: durationsMin,
        borderColor: 'rgba(6, 182, 212, 0.95)',
        backgroundColor: 'rgba(6, 182, 212, 0.2)',
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 4,
      },
    ],
  };

  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx: any) => `${Number(ctx.parsed.y).toFixed(2)} min`,
        },
      },
    },
    scales: {
      y: {
        title: { display: true, text: 'Minutes' },
      },
    },
  };

  return (
    <div style={{ height: 190 }}>
      <Line data={data} options={options} />
    </div>
  );
}
