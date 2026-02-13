'use client';

import CollapsibleCard from './CollapsibleCard';
import FieldInfo from './FieldInfo';
import { formatNumber } from '../lib/format';
import type { Locale } from '../lib/i18n';
import { SIDEBAR_FIELD_HELP, SIDEBAR_SECTION_HINTS } from '../lib/sidebarHelpText';
import type { DutyChainResponse } from '../lib/types';

type Props = {
  stopsText: string;
  onStopsTextChange: (value: string) => void;
  onRun: () => void;
  loading: boolean;
  error: string | null;
  data: DutyChainResponse | null;
  disabled: boolean;
  locale: Locale;
};

export default function DutyChainPlanner({
  stopsText,
  onStopsTextChange,
  onRun,
  loading,
  error,
  data,
  disabled,
  locale,
}: Props) {
  return (
    <CollapsibleCard
      title="Duty chain planner"
      hint={SIDEBAR_SECTION_HINTS.dutyChainPlanner}
      dataTutorialId="duty.section"
    >
      <div className="sectionTitleRow" style={{ marginTop: 8 }}>
        <button
          className="secondary"
          onClick={onRun}
          disabled={disabled || loading}
          data-tutorial-action="duty.run_click"
        >
          {loading ? 'Running...' : 'Run duty chain'}
        </button>
      </div>

      <div className="helper">
        Ordered stops, one per line: <code>lat,lon,label(optional)</code>.
      </div>
      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="duty-stops-textarea">
          Stops input
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.dutyStopsTextarea} />
      </div>
      <textarea
        id="duty-stops-textarea"
        aria-label="Duty chain stops"
        className="input"
        style={{ minHeight: 110, marginTop: 8, resize: 'vertical' }}
        value={stopsText}
        disabled={disabled || loading}
        onChange={(event) => onStopsTextChange(event.target.value)}
        data-tutorial-action="duty.stops_input"
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
              <div className="metric__value">
                {formatNumber(data.total_metrics.distance_km, locale, { maximumFractionDigits: 2 })} km
              </div>
            </div>
            <div className="metric">
              <div className="metric__label">Total duration</div>
              <div className="metric__value">
                {formatNumber(data.total_metrics.duration_s / 60, locale, { maximumFractionDigits: 1 })}{' '}
                min
              </div>
            </div>
            <div className="metric">
              <div className="metric__label">Total cost</div>
              <div className="metric__value">
                £
                {formatNumber(data.total_metrics.monetary_cost, locale, {
                  maximumFractionDigits: 2,
                })}
              </div>
            </div>
            <div className="metric">
              <div className="metric__label">Total CO2</div>
              <div className="metric__value">
                {formatNumber(data.total_metrics.emissions_kg, locale, { maximumFractionDigits: 3 })} kg
              </div>
            </div>
            {data.total_metrics.energy_kwh !== null && data.total_metrics.energy_kwh !== undefined ? (
              <div className="metric">
                <div className="metric__label">Total energy</div>
                <div className="metric__value">
                  {formatNumber(data.total_metrics.energy_kwh, locale, { maximumFractionDigits: 2 })} kWh
                </div>
              </div>
            ) : null}
          </div>

          <ul className="routeList" style={{ marginTop: 10 }}>
            {data.legs.map((leg) => (
              <li
                key={`${leg.leg_index}-${leg.origin.lat}-${leg.origin.lon}-${leg.destination.lat}-${leg.destination.lon}`}
                className="routeCard"
                style={{ cursor: 'default' }}
              >
                <div className="routeCard__top">
                  <div className="routeCard__id">
                    Leg {leg.leg_index + 1}: {leg.origin.label ?? 'Origin'} {'->'}{' '}
                    {leg.destination.label ?? 'End'}
                  </div>
                  <div className="routeCard__pill">{leg.selected ? 'OK' : 'No route'}</div>
                </div>
                {leg.error ? <div className="error">{leg.error}</div> : null}
                {leg.selected ? (
                  <div className="routeCard__meta">
                    <span>
                      {formatNumber(leg.selected.metrics.duration_s / 60, locale, {
                        maximumFractionDigits: 1,
                      })}{' '}
                      min
                    </span>
                    <span>
                      £
                      {formatNumber(leg.selected.metrics.monetary_cost, locale, {
                        maximumFractionDigits: 2,
                      })}
                    </span>
                    <span>
                      {formatNumber(leg.selected.metrics.emissions_kg, locale, {
                        maximumFractionDigits: 3,
                      })}{' '}
                      kg CO2
                    </span>
                  </div>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </CollapsibleCard>
  );
}
