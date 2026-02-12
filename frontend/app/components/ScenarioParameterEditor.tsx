'use client';

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
      <div className="helper" style={{ marginBottom: 10 }}>
        Optional controls for Pareto/robust mode, epsilon bounds, departure time, stochastic sampling,
        cost toggles, and terrain profile.
      </div>

      <div className="fieldLabel">Optimization mode</div>
      <select
        className="input"
        value={value.optimizationMode}
        disabled={disabled}
        onChange={(event) => patch('optimizationMode', event.target.value as OptimizationMode)}
      >
        <option value="expected_value">Expected value</option>
        <option value="robust">Robust</option>
      </select>

      <label className="fieldLabel" htmlFor="risk-aversion">
        Risk aversion
      </label>
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

      <div className="fieldLabel">Pareto method</div>
      <select
        className="input"
        value={value.paretoMethod}
        disabled={disabled}
        onChange={(event) => patch('paretoMethod', event.target.value as ParetoMethod)}
      >
        <option value="dominance">Dominance</option>
        <option value="epsilon_constraint">Epsilon constraint</option>
      </select>

      {value.paretoMethod === 'epsilon_constraint' && (
        <div className="advancedGrid">
          <label className="fieldLabel" htmlFor="epsilon-duration">
            Epsilon duration (s)
          </label>
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

          <label className="fieldLabel" htmlFor="epsilon-money">
            Epsilon monetary cost
          </label>
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

          <label className="fieldLabel" htmlFor="epsilon-emissions">
            Epsilon emissions (kg)
          </label>
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

      <div className="fieldLabel">Departure time (UTC)</div>
      <input
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
      </div>

      {value.stochasticEnabled ? (
        <div className="advancedGrid">
          <label className="fieldLabel" htmlFor="stochastic-seed">
            Stochastic seed (optional)
          </label>
          <input
            id="stochastic-seed"
            className="input"
            type="number"
            step={1}
            value={value.stochasticSeed}
            disabled={disabled}
            onChange={(event) => patch('stochasticSeed', event.target.value)}
          />

          <label className="fieldLabel" htmlFor="stochastic-sigma">
            Stochastic sigma (0-0.5)
          </label>
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

          <label className="fieldLabel" htmlFor="stochastic-samples">
            Stochastic samples (5-200)
          </label>
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

      <div className="fieldLabel">Terrain profile</div>
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

      <div className="checkboxRow">
        <input
          id="use-tolls"
          type="checkbox"
          checked={value.useTolls}
          disabled={disabled}
          onChange={(event) => patch('useTolls', event.target.checked)}
        />
        <label htmlFor="use-tolls">Use toll costs</label>
      </div>

      <div className="advancedGrid">
        <label className="fieldLabel" htmlFor="fuel-price-mult">
          Fuel price multiplier
        </label>
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

        <label className="fieldLabel" htmlFor="carbon-price">
          Carbon price (per kg)
        </label>
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

        <label className="fieldLabel" htmlFor="toll-per-km">
          Toll cost (per km)
        </label>
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
