'use client';

import type {
  CacheStatsResponse,
  HealthResponse,
  MetricsResponse,
} from '../../lib/types';

type Props = {
  health: HealthResponse | null;
  metrics: MetricsResponse | null;
  cacheStats: CacheStatsResponse | null;
  loading: boolean;
  clearing: boolean;
  error: string | null;
  onRefresh: () => void;
  onClearCache: () => void;
};

function pretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export default function OpsDiagnosticsPanel({
  health,
  metrics,
  cacheStats,
  loading,
  clearing,
  error,
  onRefresh,
  onClearCache,
}: Props) {
  return (
    <div className="devCard">
      <div className="devCard__head">
        <h4 className="devCard__title">Ops Diagnostics</h4>
        <div className="row">
          <button type="button" className="secondary" disabled={loading || clearing} onClick={onRefresh}>
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
          <button type="button" className="secondary danger" disabled={loading || clearing} onClick={onClearCache}>
            {clearing ? 'Clearing...' : 'Clear Cache'}
          </button>
        </div>
      </div>

      {error ? <div className="error">{error}</div> : null}

      <div className="devGrid">
        <div>
          <div className="tiny">Health</div>
          <pre className="devPre">{pretty(health ?? { status: 'unknown' })}</pre>
        </div>
        <div>
          <div className="tiny">Cache Stats</div>
          <pre className="devPre">{pretty(cacheStats ?? {})}</pre>
        </div>
      </div>

      <div className="tiny">Metrics Snapshot</div>
      <pre className="devPre">{pretty(metrics ?? {})}</pre>
    </div>
  );
}
