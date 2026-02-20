'use client';

import CollapsibleCard from './CollapsibleCard';
import FieldInfo from './FieldInfo';
import Select, { type SelectOption } from './Select';
import {
  SIDEBAR_DROPDOWN_OPTIONS_HELP,
  SIDEBAR_FIELD_HELP,
  SIDEBAR_SECTION_HINTS,
} from '../lib/sidebarHelpText';
import type {
  EuroClass,
  FuelType,
  OptimizationMode,
  ParetoMethod,
  TerrainProfile,
  WeatherProfile,
} from '../lib/types';

export type ScenarioAdvancedParams = {
  maxAlternatives: string;
  paretoMethod: ParetoMethod;
  epsilonDurationS: string;
  epsilonMonetaryCost: string;
  epsilonEmissionsKg: string;
  departureTimeUtcLocal: string;
  useTolls: boolean;
  fuelPriceMultiplier: string;
  carbonPricePerKg: string;
  tollCostPerKm: string;
  terrainProfile: TerrainProfile;
  optimizationMode: OptimizationMode;
  riskAversion: string;
  stochasticEnabled: boolean;
  stochasticSeed: string;
  stochasticSigma: string;
  stochasticSamples: string;
  fuelType: FuelType;
  euroClass: EuroClass;
  ambientTempC: string;
  weatherEnabled: boolean;
  weatherProfile: WeatherProfile;
  weatherIntensity: string;
  weatherIncidentUplift: boolean;
  incidentSimulationEnabled: boolean;
  incidentSeed: string;
  incidentDwellRatePer100km: string;
  incidentAccidentRatePer100km: string;
  incidentClosureRatePer100km: string;
  incidentDwellDelayS: string;
  incidentAccidentDelayS: string;
  incidentClosureDelayS: string;
  incidentMaxEventsPerRoute: string;
};

type Props = {
  value: ScenarioAdvancedParams;
  onChange: (next: ScenarioAdvancedParams) => void;
  disabled: boolean;
  validationError: string | null;
  sectionControl?: {
    isOpen?: boolean;
    lockToggle?: boolean;
    tutorialLocked?: boolean;
  };
};

const OPTIMIZATION_MODE_OPTIONS: SelectOption<OptimizationMode>[] = [
  { value: 'expected_value', label: 'Expected Value', description: 'Ranks by average objective values.' },
  { value: 'robust', label: 'Robust', description: 'Ranks by average plus risk penalty.' },
];

const PARETO_METHOD_OPTIONS: SelectOption<ParetoMethod>[] = [
  { value: 'dominance', label: 'Dominance', description: 'Keep only non-dominated routes.' },
  {
    value: 'epsilon_constraint',
    label: 'Epsilon Constraint',
    description: 'Apply caps, then choose non-dominated routes.',
  },
];

const TERRAIN_PROFILE_OPTIONS: SelectOption<TerrainProfile>[] = [
  { value: 'flat', label: 'Flat', description: 'Minimal gradient effect.' },
  { value: 'rolling', label: 'Rolling', description: 'Moderate gradient effect.' },
  { value: 'hilly', label: 'Hilly', description: 'Higher gradient penalty.' },
];

const FUEL_TYPE_OPTIONS: SelectOption<FuelType>[] = [
  { value: 'diesel', label: 'Diesel', description: 'Heavy-duty diesel default.' },
  { value: 'petrol', label: 'Petrol', description: 'Petrol spark ignition.' },
  { value: 'lng', label: 'LNG', description: 'Liquefied natural gas.' },
  { value: 'ev', label: 'EV', description: 'Battery electric powertrain.' },
];

const EURO_CLASS_OPTIONS: SelectOption<EuroClass>[] = [
  { value: 'euro4', label: 'Euro 4', description: 'Older emissions compliance.' },
  { value: 'euro5', label: 'Euro 5', description: 'Mid emissions compliance.' },
  { value: 'euro6', label: 'Euro 6', description: 'Latest emissions compliance baseline.' },
];

const WEATHER_PROFILE_OPTIONS: SelectOption<WeatherProfile>[] = [
  { value: 'clear', label: 'Clear', description: 'No weather penalty profile.' },
  { value: 'rain', label: 'Rain', description: 'Moderate weather impacts.' },
  { value: 'storm', label: 'Storm', description: 'Severe weather impacts.' },
  { value: 'snow', label: 'Snow', description: 'Low-speed high-variance weather.' },
  { value: 'fog', label: 'Fog', description: 'Visibility-constrained profile.' },
];

export default function ScenarioParameterEditor({
  value,
  onChange,
  disabled,
  validationError,
  sectionControl,
}: Props) {
  function patch<K extends keyof ScenarioAdvancedParams>(key: K, nextValue: ScenarioAdvancedParams[K]) {
    onChange({ ...value, [key]: nextValue });
  }

  return (
    <CollapsibleCard
      title="Advanced Parameters"
      hint={SIDEBAR_SECTION_HINTS.advancedParameters}
      dataTutorialId="advanced.section"
      isOpen={sectionControl?.isOpen}
      lockToggle={sectionControl?.lockToggle}
      tutorialLocked={sectionControl?.tutorialLocked}
    >

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="max-alternatives">
          Max Alternatives (1-48)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.maxAlternatives} />
      </div>
      <input
        id="max-alternatives"
        className="input"
        type="number"
        min={1}
        max={48}
        step={1}
        value={value.maxAlternatives}
        disabled={disabled}
        onChange={(event) => patch('maxAlternatives', event.target.value)}
      />

      <div className="fieldLabelRow">
        <div className="fieldLabel">Optimization Mode</div>
        <FieldInfo text={SIDEBAR_FIELD_HELP.optimizationMode} />
      </div>
      <Select
        id="optimization-mode"
        ariaLabel="Optimization mode"
        value={value.optimizationMode}
        options={OPTIMIZATION_MODE_OPTIONS}
        disabled={disabled}
        onChange={(next) => patch('optimizationMode', next)}
        tutorialId="advanced.optimization_mode"
        tutorialActionPrefix="advanced.optimization_mode_option"
      />
      <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.optimizationMode}</div>

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="risk-aversion">
          Risk Aversion
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.riskAversion} />
      </div>
      <input
        id="risk-aversion"
        className="input"
        type="number"
        min={0}
        step="any"
        value={value.riskAversion}
        disabled={disabled}
        onChange={(event) => patch('riskAversion', event.target.value)}
        data-tutorial-id="advanced.risk_aversion"
        data-tutorial-action="advanced.risk_aversion_input"
      />

      <div className="fieldLabelRow">
        <div className="fieldLabel">Pareto Method</div>
        <FieldInfo text={SIDEBAR_FIELD_HELP.paretoMethod} />
      </div>
      <Select
        id="pareto-method"
        ariaLabel="Pareto method"
        value={value.paretoMethod}
        options={PARETO_METHOD_OPTIONS}
        disabled={disabled}
        onChange={(next) => patch('paretoMethod', next)}
        tutorialId="advanced.pareto_method"
        tutorialActionPrefix="advanced.pareto_method_option"
      />
      <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.paretoMethod}</div>

      {value.paretoMethod === 'epsilon_constraint' && (
        <div className="advancedGrid" data-tutorial-id="advanced.epsilon_grid">
          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="epsilon-duration">
              Epsilon duration (s)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.epsilonDuration} />
          </div>
          <input
            id="epsilon-duration"
            className="input"
            type="number"
            min={0}
            step="any"
            placeholder="Optional upper bound"
            value={value.epsilonDurationS}
            disabled={disabled}
            onChange={(event) => patch('epsilonDurationS', event.target.value)}
            data-tutorial-action="advanced.epsilon_duration_input"
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="epsilon-money">
              Epsilon Monetary Cost
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.epsilonMonetaryCost} />
          </div>
          <input
            id="epsilon-money"
            className="input"
            type="number"
            min={0}
            step="any"
            placeholder="Optional upper bound"
            value={value.epsilonMonetaryCost}
            disabled={disabled}
            onChange={(event) => patch('epsilonMonetaryCost', event.target.value)}
            data-tutorial-action="advanced.epsilon_money_input"
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="epsilon-emissions">
              Epsilon Emissions (kg)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.epsilonEmissions} />
          </div>
          <input
            id="epsilon-emissions"
            className="input"
            type="number"
            min={0}
            step="any"
            placeholder="Optional upper bound"
            value={value.epsilonEmissionsKg}
            disabled={disabled}
            onChange={(event) => patch('epsilonEmissionsKg', event.target.value)}
            data-tutorial-action="advanced.epsilon_emissions_input"
          />
        </div>
      )}

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="departure-time-utc">
          Departure Time (UTC)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.departureTimeUtc} />
      </div>
      <input
        id="departure-time-utc"
        className="input"
        type="datetime-local"
        value={value.departureTimeUtcLocal}
        disabled={disabled}
        onChange={(event) => patch('departureTimeUtcLocal', event.target.value)}
        data-tutorial-id="advanced.departure_time"
        data-tutorial-action="advanced.departure_time_input"
      />

      <div className="checkboxRow">
        <input
          id="stochastic-enabled"
          type="checkbox"
          checked={value.stochasticEnabled}
          disabled={disabled}
          onChange={(event) => patch('stochasticEnabled', event.target.checked)}
          data-tutorial-id="advanced.stochastic_toggle"
          data-tutorial-action="advanced.stochastic_toggle"
        />
        <label htmlFor="stochastic-enabled">Enable Stochastic Travel-Time Sampling</label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.stochasticEnabled} />
      </div>

      {value.stochasticEnabled ? (
        <div className="advancedGrid" data-tutorial-id="advanced.stochastic_grid">
          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="stochastic-seed">
              Stochastic Seed (optional)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.stochasticSeed} />
          </div>
          <input
            id="stochastic-seed"
            className="input"
            type="number"
            step={1}
            value={value.stochasticSeed}
            disabled={disabled}
            onChange={(event) => patch('stochasticSeed', event.target.value)}
            data-tutorial-action="advanced.stochastic_seed_input"
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="stochastic-sigma">
              Stochastic Sigma (0-0.5)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.stochasticSigma} />
          </div>
          <input
            id="stochastic-sigma"
            className="input"
            type="number"
            min={0}
            max={0.5}
            step="any"
            value={value.stochasticSigma}
            disabled={disabled}
            onChange={(event) => patch('stochasticSigma', event.target.value)}
            data-tutorial-action="advanced.stochastic_sigma_input"
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="stochastic-samples">
              Stochastic Samples (5-200)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.stochasticSamples} />
          </div>
          <input
            id="stochastic-samples"
            className="input"
            type="number"
            min={5}
            max={200}
            step={1}
            value={value.stochasticSamples}
            disabled={disabled}
            onChange={(event) => patch('stochasticSamples', event.target.value)}
            data-tutorial-action="advanced.stochastic_samples_input"
          />
        </div>
      ) : null}

      <div className="fieldLabelRow">
        <div className="fieldLabel">Terrain Profile</div>
        <FieldInfo text={SIDEBAR_FIELD_HELP.terrainProfile} />
      </div>
      <Select
        id="terrain-profile"
        ariaLabel="Terrain profile"
        value={value.terrainProfile}
        options={TERRAIN_PROFILE_OPTIONS}
        disabled={disabled}
        onChange={(next) => patch('terrainProfile', next)}
        tutorialId="advanced.terrain"
        tutorialActionPrefix="advanced.terrain_option"
      />
      <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.terrainProfile}</div>

      <div className="checkboxRow">
        <input
          id="use-tolls"
          type="checkbox"
          checked={value.useTolls}
          disabled={disabled}
          onChange={(event) => patch('useTolls', event.target.checked)}
          data-tutorial-action="advanced.use_tolls_toggle"
        />
        <label htmlFor="use-tolls">Use Toll Costs</label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.useTolls} />
      </div>

      <div className="advancedGrid" data-tutorial-id="advanced.cost_toggles">
        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="fuel-price-mult">
            Fuel Price Multiplier
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.fuelPriceMultiplier} />
        </div>
        <input
          id="fuel-price-mult"
          className="input"
          type="number"
          min={0}
          step="any"
          value={value.fuelPriceMultiplier}
          disabled={disabled}
          onChange={(event) => patch('fuelPriceMultiplier', event.target.value)}
          data-tutorial-action="advanced.fuel_multiplier_input"
        />

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="carbon-price">
            Carbon Price (per kg)
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.carbonPrice} />
        </div>
        <input
          id="carbon-price"
          className="input"
          type="number"
          min={0}
          step="any"
          value={value.carbonPricePerKg}
          disabled={disabled}
          onChange={(event) => patch('carbonPricePerKg', event.target.value)}
          data-tutorial-action="advanced.carbon_price_input"
        />

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="toll-per-km">
            Toll Cost (per km)
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.tollCostPerKm} />
        </div>
        <input
          id="toll-per-km"
          className="input"
          type="number"
          min={0}
          step="any"
          value={value.tollCostPerKm}
          disabled={disabled}
          onChange={(event) => patch('tollCostPerKm', event.target.value)}
          data-tutorial-action="advanced.toll_per_km_input"
        />
      </div>

      <div className="fieldLabelRow">
        <div className="fieldLabel">Fuel Type</div>
        <FieldInfo text={SIDEBAR_FIELD_HELP.fuelType} />
      </div>
      <Select
        id="fuel-type"
        ariaLabel="Fuel type"
        value={value.fuelType}
        options={FUEL_TYPE_OPTIONS}
        disabled={disabled}
        onChange={(next) => patch('fuelType', next)}
      />

      <div className="fieldLabelRow">
        <div className="fieldLabel">Euro Class</div>
        <FieldInfo text={SIDEBAR_FIELD_HELP.euroClass} />
      </div>
      <Select
        id="euro-class"
        ariaLabel="Euro class"
        value={value.euroClass}
        options={EURO_CLASS_OPTIONS}
        disabled={disabled}
        onChange={(next) => patch('euroClass', next)}
      />

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="ambient-temp-c">
          Ambient Temp (°C)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.ambientTempC} />
      </div>
      <input
        id="ambient-temp-c"
        className="input"
        type="number"
        step="any"
        value={value.ambientTempC}
        disabled={disabled}
        onChange={(event) => patch('ambientTempC', event.target.value)}
      />

      <div className="checkboxRow">
        <input
          id="weather-enabled"
          type="checkbox"
          checked={value.weatherEnabled}
          disabled={disabled}
          onChange={(event) => patch('weatherEnabled', event.target.checked)}
        />
        <label htmlFor="weather-enabled">Enable Weather Impact Model</label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.weatherEnabled} />
      </div>

      {value.weatherEnabled ? (
        <div className="advancedGrid">
          <div className="fieldLabelRow">
            <div className="fieldLabel">Weather Profile</div>
            <FieldInfo text={SIDEBAR_FIELD_HELP.weatherProfile} />
          </div>
          <Select
            id="weather-profile"
            ariaLabel="Weather profile"
            value={value.weatherProfile}
            options={WEATHER_PROFILE_OPTIONS}
            disabled={disabled}
            onChange={(next) => patch('weatherProfile', next)}
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="weather-intensity">
              Weather Intensity (0-2)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.weatherIntensity} />
          </div>
          <input
            id="weather-intensity"
            className="input"
            type="number"
            min={0}
            max={2}
            step="any"
            value={value.weatherIntensity}
            disabled={disabled}
            onChange={(event) => patch('weatherIntensity', event.target.value)}
          />

          <div className="checkboxRow">
            <input
              id="weather-incident-uplift"
              type="checkbox"
              checked={value.weatherIncidentUplift}
              disabled={disabled}
              onChange={(event) => patch('weatherIncidentUplift', event.target.checked)}
            />
            <label htmlFor="weather-incident-uplift">Apply Incident Uplift Under Weather</label>
          </div>
        </div>
      ) : null}

      <div className="checkboxRow">
        <input
          id="incident-sim-enabled"
          type="checkbox"
          checked={value.incidentSimulationEnabled}
          disabled={disabled}
          onChange={(event) => patch('incidentSimulationEnabled', event.target.checked)}
        />
        <label htmlFor="incident-sim-enabled">Enable Incident Simulation</label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.incidentSimulationEnabled} />
      </div>

      {value.incidentSimulationEnabled ? (
        <div className="advancedGrid">
          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="incident-seed">
              Incident Seed (optional)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.incidentSeed} />
          </div>
          <input
            id="incident-seed"
            className="input"
            type="number"
            step={1}
            value={value.incidentSeed}
            disabled={disabled}
            onChange={(event) => patch('incidentSeed', event.target.value)}
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="incident-dwell-rate">
              Dwell Rate / 100km
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.incidentDwellRate} />
          </div>
          <input
            id="incident-dwell-rate"
            className="input"
            type="number"
            min={0}
            step="any"
            value={value.incidentDwellRatePer100km}
            disabled={disabled}
            onChange={(event) => patch('incidentDwellRatePer100km', event.target.value)}
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="incident-accident-rate">
              Accident Rate / 100km
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.incidentAccidentRate} />
          </div>
          <input
            id="incident-accident-rate"
            className="input"
            type="number"
            min={0}
            step="any"
            value={value.incidentAccidentRatePer100km}
            disabled={disabled}
            onChange={(event) => patch('incidentAccidentRatePer100km', event.target.value)}
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="incident-closure-rate">
              Closure Rate / 100km
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.incidentClosureRate} />
          </div>
          <input
            id="incident-closure-rate"
            className="input"
            type="number"
            min={0}
            step="any"
            value={value.incidentClosureRatePer100km}
            disabled={disabled}
            onChange={(event) => patch('incidentClosureRatePer100km', event.target.value)}
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="incident-dwell-delay">
              Dwell Delay (s)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.incidentDwellDelay} />
          </div>
          <input
            id="incident-dwell-delay"
            className="input"
            type="number"
            min={0}
            step="any"
            value={value.incidentDwellDelayS}
            disabled={disabled}
            onChange={(event) => patch('incidentDwellDelayS', event.target.value)}
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="incident-accident-delay">
              Accident Delay (s)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.incidentAccidentDelay} />
          </div>
          <input
            id="incident-accident-delay"
            className="input"
            type="number"
            min={0}
            step="any"
            value={value.incidentAccidentDelayS}
            disabled={disabled}
            onChange={(event) => patch('incidentAccidentDelayS', event.target.value)}
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="incident-closure-delay">
              Closure Delay (s)
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.incidentClosureDelay} />
          </div>
          <input
            id="incident-closure-delay"
            className="input"
            type="number"
            min={0}
            step="any"
            value={value.incidentClosureDelayS}
            disabled={disabled}
            onChange={(event) => patch('incidentClosureDelayS', event.target.value)}
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="incident-max-events">
              Max Events Per Route
            </label>
            <FieldInfo text={SIDEBAR_FIELD_HELP.incidentMaxEvents} />
          </div>
          <input
            id="incident-max-events"
            className="input"
            type="number"
            min={0}
            max={1000}
            step={1}
            value={value.incidentMaxEventsPerRoute}
            disabled={disabled}
            onChange={(event) => patch('incidentMaxEventsPerRoute', event.target.value)}
          />
        </div>
      ) : null}

      {validationError ? <div className="error">{validationError}</div> : null}
    </CollapsibleCard>
  );
}
