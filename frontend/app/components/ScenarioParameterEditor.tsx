'use client';

import FieldInfo from './FieldInfo';
import {
  SIDEBAR_DROPDOWN_OPTIONS_HELP,
  SIDEBAR_FIELD_HELP,
  SIDEBAR_SECTION_HINTS,
} from '../lib/sidebarHelpText';
import type { OptimizationMode, ParetoMethod, TerrainProfile } from '../lib/types';

export type ScenarioAdvancedParams = {
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
};

type Props = {
  value: ScenarioAdvancedParams;
  onChange: (next: ScenarioAdvancedParams) => void;
  disabled: boolean;
  validationError: string | null;
};

export default function ScenarioParameterEditor({
  value,
  onChange,
  disabled,
  validationError,
}: Props) {
  function patch<K extends keyof ScenarioAdvancedParams>(key: K, nextValue: ScenarioAdvancedParams[K]) {
    onChange({ ...value, [key]: nextValue });
  }

  return (
    <section className="card">
      <div className="sectionTitle">Advanced Parameters</div>
      <div className="sectionHint">{SIDEBAR_SECTION_HINTS.advancedParameters}</div>

      <div className="fieldLabelRow">
        <div className="fieldLabel">Optimization mode</div>
        <FieldInfo text={SIDEBAR_FIELD_HELP.optimizationMode} />
      </div>
      <select
        className="input"
        value={value.optimizationMode}
        disabled={disabled}
        onChange={(event) => patch('optimizationMode', event.target.value as OptimizationMode)}
      >
        <option value="expected_value">Expected value</option>
        <option value="robust">Robust</option>
      </select>
      <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.optimizationMode}</div>

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="risk-aversion">
          Risk aversion
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
      />

      <div className="fieldLabelRow">
        <div className="fieldLabel">Pareto method</div>
        <FieldInfo text={SIDEBAR_FIELD_HELP.paretoMethod} />
      </div>
      <select
        className="input"
        value={value.paretoMethod}
        disabled={disabled}
        onChange={(event) => patch('paretoMethod', event.target.value as ParetoMethod)}
      >
        <option value="dominance">Dominance</option>
        <option value="epsilon_constraint">Epsilon constraint</option>
      </select>
      <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.paretoMethod}</div>

      {value.paretoMethod === 'epsilon_constraint' && (
        <div className="advancedGrid">
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
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="epsilon-money">
              Epsilon monetary cost
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
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="epsilon-emissions">
              Epsilon emissions (kg)
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
          />
        </div>
      )}

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="departure-time-utc">
          Departure time (UTC)
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
      />

      <div className="checkboxRow">
        <input
          id="stochastic-enabled"
          type="checkbox"
          checked={value.stochasticEnabled}
          disabled={disabled}
          onChange={(event) => patch('stochasticEnabled', event.target.checked)}
        />
        <label htmlFor="stochastic-enabled">Enable stochastic travel-time sampling</label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.stochasticEnabled} />
      </div>

      {value.stochasticEnabled ? (
        <div className="advancedGrid">
          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="stochastic-seed">
              Stochastic seed (optional)
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
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="stochastic-sigma">
              Stochastic sigma (0-0.5)
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
          />

          <div className="fieldLabelRow">
            <label className="fieldLabel" htmlFor="stochastic-samples">
              Stochastic samples (5-200)
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
          />
        </div>
      ) : null}

      <div className="fieldLabelRow">
        <div className="fieldLabel">Terrain profile</div>
        <FieldInfo text={SIDEBAR_FIELD_HELP.terrainProfile} />
      </div>
      <select
        className="input"
        value={value.terrainProfile}
        disabled={disabled}
        onChange={(event) => patch('terrainProfile', event.target.value as TerrainProfile)}
      >
        <option value="flat">Flat</option>
        <option value="rolling">Rolling</option>
        <option value="hilly">Hilly</option>
      </select>
      <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.terrainProfile}</div>

      <div className="checkboxRow">
        <input
          id="use-tolls"
          type="checkbox"
          checked={value.useTolls}
          disabled={disabled}
          onChange={(event) => patch('useTolls', event.target.checked)}
        />
        <label htmlFor="use-tolls">Use toll costs</label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.useTolls} />
      </div>

      <div className="advancedGrid">
        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="fuel-price-mult">
            Fuel price multiplier
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
        />

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="carbon-price">
            Carbon price (per kg)
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
        />

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="toll-per-km">
            Toll cost (per km)
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
        />
      </div>

      {validationError ? <div className="error">{validationError}</div> : null}
    </section>
  );
}
