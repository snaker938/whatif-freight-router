'use client';

import { useState } from 'react';

import type { ExperimentBundle } from '../lib/types';

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
