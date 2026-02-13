'use client';

import { useEffect, useState } from 'react';

import FieldInfo from './FieldInfo';
import { formatDateTime } from '../lib/format';
import type { Locale } from '../lib/i18n';
import {
  SIDEBAR_DROPDOWN_OPTIONS_HELP,
  SIDEBAR_FIELD_HELP,
  SIDEBAR_SECTION_HINTS,
} from '../lib/sidebarHelpText';
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
  locale: Locale;
  defaultName?: string;
  defaultDescription?: string;
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
  locale,
  defaultName = '',
  defaultDescription = '',
}: Props) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  useEffect(() => {
    if (defaultName) {
      setName(defaultName);
    }
  }, [defaultName]);

  useEffect(() => {
    if (defaultDescription) {
      setDescription(defaultDescription);
    }
  }, [defaultDescription]);

  async function handleSave() {
    const trimmed = name.trim();
    if (!trimmed) return;
    await onSave(trimmed, description.trim());
    setName('');
    setDescription('');
  }

  return (
    <section className="card" data-tutorial-id="experiments.section">
      <div className="sectionTitleRow">
        <div className="sectionTitle">Experiments</div>
        <button
          className="secondary"
          onClick={onRefresh}
          disabled={loading || disabled}
          data-tutorial-action="exp.refresh_click"
        >
          Refresh
        </button>
      </div>
      <div className="sectionHint">{SIDEBAR_SECTION_HINTS.experiments}</div>

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="experiment-search">
          Catalog search
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.catalogSearch} />
      </div>
      <input
        id="experiment-search"
        className="input"
        placeholder="Search by name, id, or description"
        value={catalogQuery}
        disabled={disabled || loading}
        onChange={(event) => onCatalogQueryChange(event.target.value)}
        data-tutorial-action="exp.search_input"
      />

      <div className="advancedGrid" style={{ marginTop: 8 }}>
        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="experiment-filter-vehicle">
            Filter vehicle
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.filterVehicle} />
        </div>
        <select
          id="experiment-filter-vehicle"
          className="input"
          value={catalogVehicleType}
          disabled={disabled || loading}
          onChange={(event) => onCatalogVehicleTypeChange(event.target.value)}
          data-tutorial-action="exp.filter_vehicle_select"
        >
          <option value="">All vehicles</option>
          {vehicleOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="experiment-filter-scenario">
            Filter scenario
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.filterScenario} />
        </div>
        <select
          id="experiment-filter-scenario"
          className="input"
          value={catalogScenarioMode}
          disabled={disabled || loading}
          onChange={(event) => onCatalogScenarioModeChange(event.target.value as '' | ScenarioMode)}
          data-tutorial-action="exp.filter_scenario_select"
        >
          <option value="">All scenarios</option>
          <option value="no_sharing">No sharing</option>
          <option value="partial_sharing">Partial sharing</option>
          <option value="full_sharing">Full sharing</option>
        </select>
        <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.experimentFilterScenario}</div>

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="experiment-sort">
            Sort
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.sort} />
        </div>
        <select
          id="experiment-sort"
          className="input"
          value={catalogSort}
          disabled={disabled || loading}
          onChange={(event) => onCatalogSortChange(event.target.value as ExperimentCatalogSort)}
          data-tutorial-action="exp.sort_select"
        >
          <option value="updated_desc">Updated (newest)</option>
          <option value="updated_asc">Updated (oldest)</option>
          <option value="name_asc">Name (A-Z)</option>
          <option value="name_desc">Name (Z-A)</option>
        </select>
        <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.experimentSort}</div>
      </div>

      <div className="row row--actions" style={{ marginTop: 10 }}>
        <button
          className="secondary"
          onClick={onApplyCatalogFilters}
          disabled={loading || disabled}
          data-tutorial-action="exp.apply_filters_click"
        >
          Apply filters
        </button>
      </div>

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="experiment-name">
          Name
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.experimentName} />
      </div>
      <input
        id="experiment-name"
        className="input"
        placeholder="e.g. Morning baseline"
        value={name}
        disabled={disabled}
        onChange={(event) => setName(event.target.value)}
        data-tutorial-action="exp.name_input"
      />

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="experiment-description">
          Description (optional)
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.experimentDescription} />
      </div>
      <input
        id="experiment-description"
        className="input"
        placeholder="Notes for this scenario setup"
        value={description}
        disabled={disabled}
        onChange={(event) => setDescription(event.target.value)}
        data-tutorial-action="exp.description_input"
      />

      <div className="row row--actions" style={{ marginTop: 10 }}>
        <button
          className="primary"
          onClick={handleSave}
          disabled={!canSave || !name.trim() || disabled}
          data-tutorial-action="exp.save_click"
        >
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
                <div className="routeCard__pill">{formatDateTime(bundle.updated_at, locale)}</div>
              </div>
            {bundle.description ? <div className="helper">{bundle.description}</div> : null}
            <div className="row" style={{ marginTop: 10 }}>
              <button
                className="secondary"
                onClick={() => onLoad(bundle)}
                disabled={disabled || loading}
                data-tutorial-action="exp.load_click"
              >
                Load
              </button>
              <button
                className="secondary"
                onClick={() => onReplay(bundle.id)}
                disabled={disabled || loading}
                data-tutorial-action="exp.replay_click"
              >
                Run compare
              </button>
              <button
                className="secondary"
                onClick={() => onDelete(bundle.id)}
                disabled={disabled || loading}
                data-tutorial-action="exp.delete_click"
              >
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
