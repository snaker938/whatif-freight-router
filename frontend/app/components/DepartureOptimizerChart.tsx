'use client';

import type { DepartureOptimizeResponse } from '../lib/types';

type Props = {
  windowStartLocal: string;
  windowEndLocal: string;
  stepMinutes: number;
  loading: boolean;
  error: string | null;
  data: DepartureOptimizeResponse | null;
  disabled: boolean;
  onWindowStartChange: (value: string) => void;
  onWindowEndChange: (value: string) => void;
  onStepMinutesChange: (value: number) => void;
  onRun: () => void;
  onApplyDepartureTime: (isoUtc: string) => void;
};

export default function DepartureOptimizerChart({
  windowStartLocal,
  windowEndLocal,
  stepMinutes,
  loading,
  error,
  data,
  disabled,
  onWindowStartChange,
  onWindowEndChange,
  onStepMinutesChange,
  onRun,
  onApplyDepartureTime,
}: Props) {
  return (
    <section className="card">
      <div className="sectionTitleRow">
        <div className="sectionTitle">Departure optimization</div>
        <button className="secondary" onClick={onRun} disabled={disabled || loading}>
          {loading ? 'Optimizing...' : 'Optimize departures'}
        </button>
      </div>

      <label className="fieldLabel" htmlFor="dep-window-start">
        Window start (UTC)
      </label>
      <input
        id="dep-window-start"
        className="input"
        type="datetime-local"
        value={windowStartLocal}
        disabled={disabled || loading}
        onChange={(event) => onWindowStartChange(event.target.value)}
      />

      <label className="fieldLabel" htmlFor="dep-window-end">
        Window end (UTC)
      </label>
      <input
        id="dep-window-end"
        className="input"
        type="datetime-local"
        value={windowEndLocal}
        disabled={disabled || loading}
        onChange={(event) => onWindowEndChange(event.target.value)}
      />

      <label className="fieldLabel" htmlFor="dep-step">
        Step (minutes)
      </label>
      <input
        id="dep-step"
        className="input"
        type="number"
        min={5}
        max={720}
        step={5}
        value={stepMinutes}
        disabled={disabled || loading}
        onChange={(event) => onStepMinutesChange(Math.max(5, Number(event.target.value) || 5))}
      />

      {error ? <div className="error">{error}</div> : null}

      {data?.candidates?.length ? (
        <ul className="routeList" style={{ marginTop: 10 }}>
          {data.candidates.map((candidate) => (
            <li key={candidate.departure_time_utc} className="routeCard" style={{ cursor: 'default' }}>
              <div className="routeCard__top">
                <div className="routeCard__id">
                  {new Date(candidate.departure_time_utc).toLocaleString()} UTC
                </div>
                <div className="routeCard__pill">score {candidate.score.toFixed(4)}</div>
              </div>
              <div className="routeCard__meta">
                <span>{(candidate.selected.metrics.duration_s / 60).toFixed(1)} min</span>
                <span>£{candidate.selected.metrics.monetary_cost.toFixed(2)}</span>
                <span>{candidate.selected.metrics.emissions_kg.toFixed(3)} kg CO2</span>
              </div>
              <div className="row" style={{ marginTop: 10 }}>
                <button
                  className="secondary"
                  onClick={() => onApplyDepartureTime(candidate.departure_time_utc)}
                  disabled={disabled}
                >
                  Apply departure
                </button>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
