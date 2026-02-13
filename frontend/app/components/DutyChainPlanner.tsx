'use client';

import type { DutyChainResponse } from '../lib/types';

type Props = {
  stopsText: string;
  onStopsTextChange: (value: string) => void;
  onRun: () => void;
  loading: boolean;
  error: string | null;
  data: DutyChainResponse | null;
  disabled: boolean;
};

export default function DutyChainPlanner({
  stopsText,
  onStopsTextChange,
  onRun,
  loading,
  error,
  data,
  disabled,
}: Props) {
  return (
    <section className="card">
      <div className="sectionTitleRow">
        <div className="sectionTitle">Duty chain planner</div>
        <button className="secondary" onClick={onRun} disabled={disabled || loading}>
          {loading ? 'Running...' : 'Run duty chain'}
        </button>
      </div>

      <div className="helper">
        Ordered stops, one per line: <code>lat,lon,label(optional)</code>.
      </div>
      <textarea
        className="input"
        style={{ minHeight: 110, marginTop: 8, resize: 'vertical' }}
        value={stopsText}
        disabled={disabled || loading}
        onChange={(event) => onStopsTextChange(event.target.value)}
      />

      {error ? <div className="error">{error}</div> : null}

      {data ? (
        <div style={{ marginTop: 10 }}>
          <div className="tiny">
            Legs: {data.leg_count} | Successful: {data.successful_leg_count}
          </div>
          <div className="metrics" style={{ marginTop: 10 }}>
            <div className="metric">
              <div className="metric__label">Total distance</div>
              <div className="metric__value">{data.total_metrics.distance_km.toFixed(2)} km</div>
            </div>
            <div className="metric">
              <div className="metric__label">Total duration</div>
              <div className="metric__value">{(data.total_metrics.duration_s / 60).toFixed(1)} min</div>
            </div>
            <div className="metric">
              <div className="metric__label">Total cost</div>
              <div className="metric__value">£{data.total_metrics.monetary_cost.toFixed(2)}</div>
            </div>
            <div className="metric">
              <div className="metric__label">Total CO2</div>
              <div className="metric__value">{data.total_metrics.emissions_kg.toFixed(3)} kg</div>
            </div>
            {data.total_metrics.energy_kwh !== null && data.total_metrics.energy_kwh !== undefined ? (
              <div className="metric">
                <div className="metric__label">Total energy</div>
                <div className="metric__value">{data.total_metrics.energy_kwh.toFixed(2)} kWh</div>
              </div>
            ) : null}
          </div>

          <ul className="routeList" style={{ marginTop: 10 }}>
            {data.legs.map((leg) => (
              <li key={`${leg.leg_index}-${leg.origin.lat}-${leg.destination.lat}`} className="routeCard" style={{ cursor: 'default' }}>
                <div className="routeCard__top">
                  <div className="routeCard__id">
                    Leg {leg.leg_index + 1}: {leg.origin.label ?? 'Origin'} {'->'}{' '}
                    {leg.destination.label ?? 'Destination'}
                  </div>
                  <div className="routeCard__pill">{leg.selected ? 'OK' : 'No route'}</div>
                </div>
                {leg.error ? <div className="error">{leg.error}</div> : null}
                {leg.selected ? (
                  <div className="routeCard__meta">
                    <span>{(leg.selected.metrics.duration_s / 60).toFixed(1)} min</span>
                    <span>£{leg.selected.metrics.monetary_cost.toFixed(2)}</span>
                    <span>{leg.selected.metrics.emissions_kg.toFixed(3)} kg CO2</span>
                  </div>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
