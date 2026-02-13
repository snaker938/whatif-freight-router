'use client';

import { useState } from 'react';

import type { ExperimentBundle, ExperimentCatalogSort, ScenarioMode } from '../lib/types';

type Props = {
  experiments: ExperimentBundle[];
  loading: boolean;
  error: string | null;
  canSave: boolean;
  disabled: boolean;
  onRefresh: () => void;
  onSave: (name: string, description: string) => Promise<void> | void;
  onLoad: (bundle: ExperimentBundle) => void;
  onDelete: (experimentId: string) => Promise<void> | void;
  onReplay: (experimentId: string) => Promise<void> | void;
  catalogQuery: string;
  catalogVehicleType: string;
  catalogScenarioMode: '' | ScenarioMode;
  catalogSort: ExperimentCatalogSort;
  vehicleOptions: Array<{ value: string; label: string }>;
  onCatalogQueryChange: (value: string) => void;
  onCatalogVehicleTypeChange: (value: string) => void;
  onCatalogScenarioModeChange: (value: '' | ScenarioMode) => void;
  onCatalogSortChange: (value: ExperimentCatalogSort) => void;
  onApplyCatalogFilters: () => void;
};

export default function ExperimentManager({
  experiments,
  loading,
  error,
  canSave,
  disabled,
  onRefresh,
  onSave,
  onLoad,
  onDelete,
  onReplay,
  catalogQuery,
  catalogVehicleType,
  catalogScenarioMode,
  catalogSort,
  vehicleOptions,
  onCatalogQueryChange,
  onCatalogVehicleTypeChange,
  onCatalogScenarioModeChange,
  onCatalogSortChange,
  onApplyCatalogFilters,
}: Props) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  async function handleSave() {
    const trimmed = name.trim();
    if (!trimmed) return;
    await onSave(trimmed, description.trim());
    setName('');
    setDescription('');
  }

  return (
    <section className="card">
      <div className="sectionTitleRow">
        <div className="sectionTitle">Experiments</div>
        <button className="secondary" onClick={onRefresh} disabled={loading || disabled}>
          Refresh
        </button>
      </div>

      <label className="fieldLabel" htmlFor="experiment-search">
        Catalog search
      </label>
      <input
        id="experiment-search"
        className="input"
        placeholder="Search by name, id, or description"
        value={catalogQuery}
        disabled={disabled || loading}
        onChange={(event) => onCatalogQueryChange(event.target.value)}
      />

      <div className="advancedGrid" style={{ marginTop: 8 }}>
        <label className="fieldLabel" htmlFor="experiment-filter-vehicle">
          Filter vehicle
        </label>
        <select
          id="experiment-filter-vehicle"
          className="input"
          value={catalogVehicleType}
          disabled={disabled || loading}
          onChange={(event) => onCatalogVehicleTypeChange(event.target.value)}
        >
          <option value="">All vehicles</option>
          {vehicleOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>

        <label className="fieldLabel" htmlFor="experiment-filter-scenario">
          Filter scenario
        </label>
        <select
          id="experiment-filter-scenario"
          className="input"
          value={catalogScenarioMode}
          disabled={disabled || loading}
          onChange={(event) => onCatalogScenarioModeChange(event.target.value as '' | ScenarioMode)}
        >
          <option value="">All scenarios</option>
          <option value="no_sharing">No sharing</option>
          <option value="partial_sharing">Partial sharing</option>
          <option value="full_sharing">Full sharing</option>
        </select>

        <label className="fieldLabel" htmlFor="experiment-sort">
          Sort
        </label>
        <select
          id="experiment-sort"
          className="input"
          value={catalogSort}
          disabled={disabled || loading}
          onChange={(event) => onCatalogSortChange(event.target.value as ExperimentCatalogSort)}
        >
          <option value="updated_desc">Updated (newest)</option>
          <option value="updated_asc">Updated (oldest)</option>
          <option value="name_asc">Name (A-Z)</option>
          <option value="name_desc">Name (Z-A)</option>
        </select>
      </div>

      <div className="row row--actions" style={{ marginTop: 10 }}>
        <button className="secondary" onClick={onApplyCatalogFilters} disabled={loading || disabled}>
          Apply filters
        </button>
      </div>

      <label className="fieldLabel" htmlFor="experiment-name">
        Name
      </label>
      <input
        id="experiment-name"
        className="input"
        placeholder="e.g. Morning baseline"
        value={name}
        disabled={disabled}
        onChange={(event) => setName(event.target.value)}
      />

      <label className="fieldLabel" htmlFor="experiment-description">
        Description (optional)
      </label>
      <input
        id="experiment-description"
        className="input"
        placeholder="Notes for this scenario setup"
        value={description}
        disabled={disabled}
        onChange={(event) => setDescription(event.target.value)}
      />

      <div className="row row--actions" style={{ marginTop: 10 }}>
        <button className="primary" onClick={handleSave} disabled={!canSave || !name.trim() || disabled}>
          Save current bundle
        </button>
      </div>

      {error ? <div className="error">{error}</div> : null}

      <ul className="routeList" style={{ marginTop: 10 }}>
        {experiments.length === 0 ? <li className="helper">No experiments match the current filters.</li> : null}
        {experiments.map((bundle) => (
          <li key={bundle.id} className="routeCard" style={{ cursor: 'default' }}>
            <div className="routeCard__top">
              <div className="routeCard__id">{bundle.name}</div>
              <div className="routeCard__pill">{new Date(bundle.updated_at).toLocaleString()}</div>
            </div>
            {bundle.description ? <div className="helper">{bundle.description}</div> : null}
            <div className="row" style={{ marginTop: 10 }}>
              <button className="secondary" onClick={() => onLoad(bundle)} disabled={disabled || loading}>
                Load
              </button>
              <button className="secondary" onClick={() => onReplay(bundle.id)} disabled={disabled || loading}>
                Run compare
              </button>
              <button className="secondary" onClick={() => onDelete(bundle.id)} disabled={disabled || loading}>
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
