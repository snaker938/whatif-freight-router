'use client';

import CollapsibleCard from './CollapsibleCard';
import FieldInfo from './FieldInfo';
import { formatMetricTooltip, type MetricTooltip } from './metricTooltip';
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
  sectionControl?: {
    isOpen?: boolean;
    lockToggle?: boolean;
    tutorialLocked?: boolean;
  };
};

const inlineMetricLabelStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
} as const;

function metricHelp(tooltip: MetricTooltip): string {
  return formatMetricTooltip(tooltip);
}

function metricLabel(label: string, tooltip: MetricTooltip) {
  return (
    <>
      {label}
      <FieldInfo text={metricHelp(tooltip)} />
    </>
  );
}

function inlineMetricLabel(label: string, tooltip: MetricTooltip) {
  return (
    <span style={inlineMetricLabelStyle}>
      <span>{label}</span>
      <FieldInfo text={metricHelp(tooltip)} />
    </span>
  );
}

export default function DutyChainPlanner({
  stopsText,
  onStopsTextChange,
  onRun,
  loading,
  error,
  data,
  disabled,
  locale,
  sectionControl,
}: Props) {
  const nonEmptyLines = stopsText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  return (
    <CollapsibleCard
      title="Duty Chain Planner"
      hint={SIDEBAR_SECTION_HINTS.dutyChainPlanner}
      dataTutorialId="duty.section"
      isOpen={sectionControl?.isOpen}
      lockToggle={sectionControl?.lockToggle}
      tutorialLocked={sectionControl?.tutorialLocked}
    >
      <div className="helper">
        Ordered Stops, One Per Line: <code>lat,lon,label(optional)</code>.
      </div>
      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="duty-stops-textarea">
          Stops Input
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.dutyStopsTextarea} />
      </div>
      <textarea
        id="duty-stops-textarea"
        aria-label="Duty Chain Stops"
        className="input"
        style={{ minHeight: 110, marginTop: 8, resize: 'vertical' }}
        value={stopsText}
        disabled={disabled || loading}
        onChange={(event) => onStopsTextChange(event.target.value)}
        data-tutorial-action="duty.stops_input"
      />

      <div className="tiny">
        Lines: {nonEmptyLines.length} (Supports Up To 50 Stops Including Start/End)
      </div>
      {nonEmptyLines.length > 50 ? (
        <div className="error">Too Many Stops. Use 50 Or Fewer Rows.</div>
      ) : null}

      <div className="actionGrid actionGrid--single" style={{ marginTop: 12 }}>
        <button type="button"
          className="secondary"
          onClick={onRun}
          disabled={disabled || loading || nonEmptyLines.length > 50}
          data-tutorial-action="duty.run_click"
        >
          {loading ? 'Running...' : 'Run Duty Chain'}
        </button>
      </div>

      {error ? <div className="error">{error}</div> : null}

      {data ? (
        <div style={{ marginTop: 10 }}>
          <div className="tiny">
            {inlineMetricLabel('Legs', {
              definition: 'Total number of legs in the duty-chain run.',
              direction: 'Context only; more legs indicate a longer chain, not automatically better or worse.',
              unit: 'leg count',
            })}{' '}
            {data.leg_count} |{' '}
            {inlineMetricLabel('Successful Legs', {
              definition: 'Number of duty-chain legs that returned a route successfully.',
              direction: 'Higher is better because more planned legs were routed successfully.',
              unit: 'successful leg count',
            })}{' '}
            {data.successful_leg_count}
          </div>
          <div className="metrics" style={{ marginTop: 10 }}>
            <div className="metric">
              <div className="metric__label">
                {metricLabel('Total Distance', {
                  definition: 'Total routed distance across all duty-chain legs.',
                  direction: 'Lower is usually better when distance is a cost to minimize.',
                  unit: 'kilometers',
                })}
              </div>
              <div className="metric__value">
                {formatNumber(data.total_metrics.distance_km, locale, { maximumFractionDigits: 2 })} km
              </div>
            </div>
            <div className="metric">
              <div className="metric__label">
                {metricLabel('Total Duration', {
                  definition: 'Total travel time across all duty-chain legs.',
                  direction: 'Lower is usually better because the chain completes sooner.',
                  unit: 'minutes',
                })}
              </div>
              <div className="metric__value">
                {formatNumber(data.total_metrics.duration_s / 60, locale, { maximumFractionDigits: 1 })}{' '}
                min
              </div>
            </div>
            <div className="metric">
              <div className="metric__label">
                {metricLabel('Total Cost', {
                  definition: 'Proxy monetary cost across all duty-chain legs.',
                  direction: 'Lower is usually better because the chain costs less.',
                  unit: 'currency units',
                })}
              </div>
              <div className="metric__value">
                £
                {formatNumber(data.total_metrics.monetary_cost, locale, {
                  maximumFractionDigits: 2,
                })}
              </div>
            </div>
            <div className="metric">
              <div className="metric__label">
                {metricLabel('Total CO2', {
                  definition: 'Estimated total emissions across all duty-chain legs.',
                  direction: 'Lower is usually better because less CO2 is emitted.',
                  unit: 'kilograms CO2',
                })}
              </div>
              <div className="metric__value">
                {formatNumber(data.total_metrics.emissions_kg, locale, { maximumFractionDigits: 3 })} kg
              </div>
            </div>
            {data.total_metrics.energy_kwh !== null && data.total_metrics.energy_kwh !== undefined ? (
              <div className="metric">
                <div className="metric__label">
                  {metricLabel('Total Energy', {
                    definition: 'Estimated total energy use across all duty-chain legs.',
                    direction: 'Lower is usually better because the chain consumes less energy.',
                    unit: 'kilowatt-hours',
                  })}
                </div>
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
                    Leg {leg.leg_index + 1}: {leg.origin.label ?? 'Origin'} {'→'}{' '}
                    {leg.destination.label ?? 'End'}
                  </div>
                  <div className="routeCard__pill">
                    {inlineMetricLabel('Leg status', {
                      definition: 'Whether this duty-chain leg returned a route successfully.',
                      direction: 'OK is better because the leg routed successfully; No Route indicates failure.',
                      unit: 'categorical status',
                    })}{' '}
                    {leg.selected ? 'OK' : 'No Route'}
                  </div>
                </div>
                {leg.error ? <div className="error">{leg.error}</div> : null}
                {leg.selected ? (
                  <div className="routeCard__meta">
                    <span>
                      {inlineMetricLabel('Duration', {
                        definition: 'Travel time for this duty-chain leg.',
                        direction: 'Lower is usually better because the leg completes sooner.',
                        unit: 'minutes',
                      })}{' '}
                      {formatNumber(leg.selected.metrics.duration_s / 60, locale, {
                        maximumFractionDigits: 1,
                      })}{' '}
                      min
                    </span>
                    <span>
                      {inlineMetricLabel('Cost', {
                        definition: 'Proxy monetary cost for this duty-chain leg.',
                        direction: 'Lower is usually better because the leg costs less.',
                        unit: 'currency units',
                      })}{' '}
                      £
                      {formatNumber(leg.selected.metrics.monetary_cost, locale, {
                        maximumFractionDigits: 2,
                      })}
                    </span>
                    <span>
                      {inlineMetricLabel('CO2', {
                        definition: 'Estimated emissions for this duty-chain leg.',
                        direction: 'Lower is usually better because less CO2 is emitted.',
                        unit: 'kilograms CO2',
                      })}{' '}
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
