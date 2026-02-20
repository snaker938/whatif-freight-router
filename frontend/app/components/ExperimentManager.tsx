'use client';

import { useEffect, useState } from 'react';

import CollapsibleCard from './CollapsibleCard';
import FieldInfo from './FieldInfo';
import Select, { type SelectOption } from './Select';
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
  onOpen: (experimentId: string) => Promise<void> | void;
  onUpdateMetadata: (
    bundle: ExperimentBundle,
    next: { name: string; description?: string | null },
  ) => Promise<void> | void;
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
  tutorialResetNonce?: number;
  sectionControl?: {
    isOpen?: boolean;
    lockToggle?: boolean;
    tutorialLocked?: boolean;
  };
};

const SCENARIO_FILTER_OPTIONS: SelectOption<'' | ScenarioMode>[] = [
  { value: '', label: 'All Scenarios', description: 'No scenario filter.' },
  { value: 'no_sharing', label: 'No Sharing', description: 'Only no-sharing bundles.' },
  { value: 'partial_sharing', label: 'Partial Sharing', description: 'Only partial-sharing bundles.' },
  { value: 'full_sharing', label: 'Full Sharing', description: 'Only full-sharing bundles.' },
];

const EXPERIMENT_SORT_OPTIONS: SelectOption<ExperimentCatalogSort>[] = [
  { value: 'updated_desc', label: 'Updated (Newest)', description: 'Most recently updated first.' },
  { value: 'updated_asc', label: 'Updated (Oldest)', description: 'Least recently updated first.' },
  { value: 'name_asc', label: 'Name (A-Z)', description: 'Alphabetical ascending.' },
  { value: 'name_desc', label: 'Name (Z-A)', description: 'Alphabetical descending.' },
];

export default function ExperimentManager({
  experiments,
  loading,
  error,
  canSave,
  disabled,
  onRefresh,
  onSave,
  onLoad,
  onOpen,
  onUpdateMetadata,
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
  tutorialResetNonce,
  sectionControl,
}: Props) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const trimmedName = name.trim();
  const nameError =
    trimmedName.length > 80 ? 'Name Must Be 80 Characters Or Fewer.' : null;

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

  useEffect(() => {
    if (typeof tutorialResetNonce !== 'number') return;
    setName(defaultName ?? '');
    setDescription(defaultDescription ?? '');
  }, [defaultDescription, defaultName, tutorialResetNonce]);

  async function handleSave() {
    if (!trimmedName || nameError) return;
    await onSave(trimmedName, description.trim());
    setName('');
    setDescription('');
  }

  const vehicleFilterOptions: SelectOption<string>[] = [
    { value: '', label: 'All Vehicles', description: 'No vehicle filter.' },
    ...vehicleOptions.map((opt) => ({ value: opt.value, label: opt.label })),
  ];

  return (
    <CollapsibleCard
      title="Experiments"
      hint={SIDEBAR_SECTION_HINTS.experiments}
      dataTutorialId="experiments.section"
      isOpen={sectionControl?.isOpen}
      lockToggle={sectionControl?.lockToggle}
      tutorialLocked={sectionControl?.tutorialLocked}
    >
      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="experiment-search">
          Catalog Search
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.catalogSearch} />
      </div>
      <input
        id="experiment-search"
        className="input"
        placeholder="Search By Name, ID, Or Description"
        value={catalogQuery}
        disabled={disabled || loading}
        onChange={(event) => onCatalogQueryChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.preventDefault();
            onApplyCatalogFilters();
          }
        }}
        data-tutorial-action="exp.search_input"
      />

      <div className="advancedGrid" style={{ marginTop: 8 }}>
        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="experiment-filter-vehicle">
            Filter Vehicle
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.filterVehicle} />
        </div>
        <Select
          id="experiment-filter-vehicle"
          ariaLabel="Filter vehicle"
          value={catalogVehicleType}
          options={vehicleFilterOptions}
          disabled={disabled || loading}
          onChange={onCatalogVehicleTypeChange}
          tutorialAction="exp.filter_vehicle_select"
        />

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="experiment-filter-scenario">
            Filter Scenario
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.filterScenario} />
        </div>
        <Select
          id="experiment-filter-scenario"
          ariaLabel="Filter scenario"
          value={catalogScenarioMode}
          options={SCENARIO_FILTER_OPTIONS}
          disabled={disabled || loading}
          onChange={onCatalogScenarioModeChange}
          tutorialAction="exp.filter_scenario_select"
        />
        <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.experimentFilterScenario}</div>

        <div className="fieldLabelRow">
          <label className="fieldLabel" htmlFor="experiment-sort">
            Sort
          </label>
          <FieldInfo text={SIDEBAR_FIELD_HELP.sort} />
        </div>
        <Select
          id="experiment-sort"
          ariaLabel="Experiment sort order"
          value={catalogSort}
          options={EXPERIMENT_SORT_OPTIONS}
          disabled={disabled || loading}
          onChange={onCatalogSortChange}
          tutorialAction="exp.sort_select"
        />
        <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.experimentSort}</div>
      </div>

      <div className="actionGrid actionGrid--single" style={{ marginTop: 10 }}>
        <button type="button"
          className="secondary"
          onClick={onRefresh}
          disabled={loading || disabled}
          data-tutorial-action="exp.refresh_click"
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
        <button type="button"
          className="secondary"
          onClick={onApplyCatalogFilters}
          disabled={loading || disabled}
          data-tutorial-action="exp.apply_filters_click"
        >
          Apply Filters
        </button>
        <button type="button"
          className="secondary"
          onClick={() => {
            onCatalogQueryChange('');
            onCatalogVehicleTypeChange('');
            onCatalogScenarioModeChange('');
            onCatalogSortChange('updated_desc');
            onApplyCatalogFilters();
          }}
          disabled={loading || disabled}
          data-tutorial-action="exp.clear_filters_click"
        >
          Clear Filters
        </button>
      </div>

      <div className="tiny">
        Showing {experiments.length} Experiment{experiments.length === 1 ? '' : 's'}
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
      {nameError ? <div className="error">{nameError}</div> : null}

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="experiment-description">
          Description (Optional)
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

      <div className="actionGrid" style={{ marginTop: 10 }}>
        <button type="button"
          className="primary"
          onClick={handleSave}
          disabled={!canSave || !trimmedName || disabled || Boolean(nameError)}
          data-tutorial-action="exp.save_click"
        >
          Save Current Bundle
        </button>
      </div>

      {error ? <div className="error">{error}</div> : null}

      <ul className="routeList" style={{ marginTop: 10 }}>
        {experiments.length === 0 ? <li className="helper">No Experiments Match The Current Filters.</li> : null}
        {experiments.map((bundle) => (
          <li key={bundle.id} className="routeCard" style={{ cursor: 'default' }}>
            <div className="routeCard__top">
                <div className="routeCard__id">{bundle.name}</div>
                <div className="routeCard__pill">{formatDateTime(bundle.updated_at, locale)}</div>
              </div>
            {bundle.description ? <div className="helper">{bundle.description}</div> : null}
            <div className="row" style={{ marginTop: 10 }}>
              <button type="button"
                className="secondary"
                onClick={() => onLoad(bundle)}
                disabled={disabled || loading}
                data-tutorial-action="exp.load_click"
              >
                Load
              </button>
              <button
                type="button"
                className="secondary"
                onClick={() => onOpen(bundle.id)}
                disabled={disabled || loading}
              >
                Open (GET)
              </button>
              <button
                type="button"
                className="secondary"
                onClick={async () => {
                  const nextName = window.prompt('Experiment name', bundle.name);
                  if (nextName === null) return;
                  const nextDescription = window.prompt(
                    'Experiment description (optional)',
                    bundle.description ?? '',
                  );
                  if (nextDescription === null) return;
                  await onUpdateMetadata(bundle, {
                    name: nextName.trim() || bundle.name,
                    description: nextDescription.trim() || null,
                  });
                }}
                disabled={disabled || loading}
              >
                Edit Meta (PUT)
              </button>
                <button type="button"
                  className="secondary"
                  onClick={() => onReplay(bundle.id)}
                  disabled={disabled || loading}
                  data-tutorial-action="exp.replay_click"
                >
                  Run Compare
                </button>
              <button type="button"
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
    </CollapsibleCard>
  );
}
