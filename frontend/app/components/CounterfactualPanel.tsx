'use client';

import type { RouteOption } from '../lib/types';

type Props = {
  route: RouteOption | null;
};

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
                    {delta > 0 ? '+' : ''}
                    {delta.toFixed(3)} {metric}
                  </div>
                ) : null}
              </div>
              <div className="routeCard__meta">
                <span>Metric: {metric}</span>
                <span>{improves ? 'Improves Outcome' : 'Worse Outcome'}</span>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
