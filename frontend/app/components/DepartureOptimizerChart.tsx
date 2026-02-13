'use client';

import CollapsibleCard from './CollapsibleCard';
import FieldInfo from './FieldInfo';
import { formatDateTime, formatNumber } from '../lib/format';
import {
  SIDEBAR_FIELD_HELP,
  SIDEBAR_SECTION_HINTS,
} from '../lib/sidebarHelpText';
import type { Locale } from '../lib/i18n';
import type { DepartureOptimizeResponse } from '../lib/types';

type Props = {
  windowStartLocal: string;
  windowEndLocal: string;
  earliestArrivalLocal: string;
  latestArrivalLocal: string;
  stepMinutes: number;
  loading: boolean;
  error: string | null;
  data: DepartureOptimizeResponse | null;
  disabled: boolean;
  onWindowStartChange: (value: string) => void;
  onWindowEndChange: (value: string) => void;
  onEarliestArrivalChange: (value: string) => void;
  onLatestArrivalChange: (value: string) => void;
  onStepMinutesChange: (value: number) => void;
  onRun: () => void;
  onApplyDepartureTime: (isoUtc: string) => void;
  locale: Locale;
};

export default function DepartureOptimizerChart({
  windowStartLocal,
  windowEndLocal,
  earliestArrivalLocal,
  latestArrivalLocal,
  stepMinutes,
  loading,
  error,
  data,
  disabled,
  onWindowStartChange,
  onWindowEndChange,
  onEarliestArrivalChange,
  onLatestArrivalChange,
  onStepMinutesChange,
  onRun,
  onApplyDepartureTime,
  locale,
}: Props) {
  return (
    <CollapsibleCard
      title="Departure Optimization"
      hint={SIDEBAR_SECTION_HINTS.departureOptimization}
      dataTutorialId="departure.section"
    >
      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="dep-window-start">
          Window Start (UTC)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.windowStartEnd} />
      </div>
      <input
        id="dep-window-start"
        className="input"
        type="datetime-local"
        value={windowStartLocal}
        disabled={disabled || loading}
        onChange={(event) => onWindowStartChange(event.target.value)}
        data-tutorial-action="dep.window_start_input"
      />

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="dep-window-end">
          Window End (UTC)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.windowStartEnd} />
      </div>
      <input
        id="dep-window-end"
        className="input"
        type="datetime-local"
        value={windowEndLocal}
        disabled={disabled || loading}
        onChange={(event) => onWindowEndChange(event.target.value)}
        data-tutorial-action="dep.window_end_input"
      />

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="dep-step">
          Step (Minutes)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.stepMinutes} />
      </div>
      <input
        id="dep-step"
        className="input"
        type="number"
        min={5}
        max={720}
        step={5}
        value={stepMinutes}
        disabled={disabled || loading}
        onChange={(event) => {
          const parsed = Number(event.target.value);
          const safe = Number.isFinite(parsed) ? parsed : 5;
          const clamped = Math.min(720, Math.max(5, safe));
          onStepMinutesChange(Math.round(clamped));
        }}
        data-tutorial-action="dep.step_input"
      />

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="dep-earliest-arrival">
          Earliest Arrival (UTC, Optional)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.earliestLatestArrival} />
      </div>
      <input
        id="dep-earliest-arrival"
        className="input"
        type="datetime-local"
        value={earliestArrivalLocal}
        disabled={disabled || loading}
        onChange={(event) => onEarliestArrivalChange(event.target.value)}
        data-tutorial-action="dep.earliest_input"
      />

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="dep-latest-arrival">
          Latest Arrival (UTC, Optional)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.earliestLatestArrival} />
      </div>
      <input
        id="dep-latest-arrival"
        className="input"
        type="datetime-local"
        value={latestArrivalLocal}
        disabled={disabled || loading}
        onChange={(event) => onLatestArrivalChange(event.target.value)}
        data-tutorial-action="dep.latest_input"
      />

      <div className="row row--actions" style={{ marginTop: 12 }}>
        <button
          className="secondary"
          onClick={onRun}
          disabled={disabled || loading}
          data-tutorial-action="dep.optimize_click"
        >
          {loading ? 'Optimizing...' : 'Optimize Departures'}
        </button>
      </div>

      {error ? <div className="error">{error}</div> : null}

      {data?.candidates?.length ? (
        <ul className="routeList" style={{ marginTop: 10 }}>
          {data.candidates.map((candidate) => (
            <li key={candidate.departure_time_utc} className="routeCard" style={{ cursor: 'default' }}>
              <div className="routeCard__top">
                <div className="routeCard__id">
                  {formatDateTime(candidate.departure_time_utc, locale)} UTC
                </div>
                <div className="routeCard__pill">
                  Score {formatNumber(candidate.score, locale, { maximumFractionDigits: 4 })}
                </div>
              </div>
              <div className="routeCard__meta">
                <span>
                  {formatNumber(candidate.selected.metrics.duration_s / 60, locale, {
                    maximumFractionDigits: 1,
                  })}{' '}
                  min
                </span>
                <span>
                  £
                  {formatNumber(candidate.selected.metrics.monetary_cost, locale, {
                    maximumFractionDigits: 2,
                  })}
                </span>
                <span>
                  {formatNumber(candidate.selected.metrics.emissions_kg, locale, {
                    maximumFractionDigits: 3,
                  })}{' '}
                  kg CO2
                </span>
              </div>
              <div className="row" style={{ marginTop: 10 }}>
                <button
                  className="secondary"
                  onClick={() => onApplyDepartureTime(candidate.departure_time_utc)}
                  disabled={disabled}
                  data-tutorial-action="dep.apply_departure"
                >
                  Apply Departure
                </button>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
    </CollapsibleCard>
  );
}
