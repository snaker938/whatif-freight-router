'use client';

import { useState } from 'react';

import type {
  BatchCSVImportRequest,
  BatchParetoRequest,
  BatchParetoResponse,
} from '../../lib/types';

type Props = {
  loading: boolean;
  error: string | null;
  result: BatchParetoResponse | null;
  onRunPairs: (request: BatchParetoRequest) => Promise<void> | void;
  onRunCsv: (request: BatchCSVImportRequest) => Promise<void> | void;
};

const DEFAULT_BATCH_JSON = JSON.stringify(
  {
    pairs: [
      {
        origin: { lat: 51.5074, lon: -0.1278 },
        destination: { lat: 53.4808, lon: -2.2426 },
      },
    ],
    vehicle_type: 'rigid_hgv',
    scenario_mode: 'no_sharing',
    max_alternatives: 8,
  },
  null,
  2,
);

const DEFAULT_CSV = `origin_lat,origin_lon,destination_lat,destination_lon
51.5074,-0.1278,53.4808,-2.2426`;

const DEFAULT_CSV_OPTIONS = JSON.stringify(
  {
    vehicle_type: 'rigid_hgv',
    scenario_mode: 'no_sharing',
    max_alternatives: 8,
  },
  null,
  2,
);

export default function BatchRunner({ loading, error, result, onRunPairs, onRunCsv }: Props) {
  const [batchJson, setBatchJson] = useState(DEFAULT_BATCH_JSON);
  const [csvText, setCsvText] = useState(DEFAULT_CSV);
  const [csvOptionsJson, setCsvOptionsJson] = useState(DEFAULT_CSV_OPTIONS);
  const [localError, setLocalError] = useState<string | null>(null);

  async function runPairs() {
    try {
      setLocalError(null);
      const payload = JSON.parse(batchJson) as BatchParetoRequest;
      await onRunPairs(payload);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Invalid batch JSON payload.');
    }
  }

  async function runCsv() {
    try {
      setLocalError(null);
      const options = JSON.parse(csvOptionsJson) as Omit<BatchCSVImportRequest, 'csv_text'>;
      await onRunCsv({
        ...options,
        csv_text: csvText,
      });
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Invalid CSV options JSON payload.');
    }
  }

  return (
    <div className="devCard">
      <div className="devCard__head">
        <h4 className="devCard__title">Batch Runner</h4>
      </div>
      {error ? <div className="error">{error}</div> : null}
      {localError ? <div className="error">{localError}</div> : null}

      <div className="fieldLabel">POST /batch/pareto payload</div>
      <textarea
        className="input devTextarea"
        value={batchJson}
        disabled={loading}
        onChange={(event) => setBatchJson(event.target.value)}
      />
      <div className="actionGrid actionGrid--single u-mt10">
        <button type="button" className="secondary" onClick={runPairs} disabled={loading}>
          {loading ? 'Running...' : 'Run Batch Pareto'}
        </button>
      </div>

      <div className="fieldLabel u-mt12">POST /batch/import/csv csv_text</div>
      <textarea
        className="input devTextarea"
        value={csvText}
        disabled={loading}
        onChange={(event) => setCsvText(event.target.value)}
      />

      <div className="fieldLabel u-mt12">POST /batch/import/csv options JSON</div>
      <textarea
        className="input devTextarea"
        value={csvOptionsJson}
        disabled={loading}
        onChange={(event) => setCsvOptionsJson(event.target.value)}
      />
      <div className="actionGrid actionGrid--single u-mt10">
        <button type="button" className="secondary" onClick={runCsv} disabled={loading}>
          {loading ? 'Running...' : 'Run Batch CSV'}
        </button>
      </div>

      {result ? (
        <>
          <div className="tiny u-mt12">Run ID: {result.run_id}</div>
          <pre className="devPre">{JSON.stringify(result, null, 2)}</pre>
        </>
      ) : null}
    </div>
  );
}
