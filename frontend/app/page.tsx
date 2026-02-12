'use client';
// frontend/app/page.tsx

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useMemo, useRef, useState, useTransition } from 'react';

import Select from './components/Select';
import { postJSON, postNDJSON } from './lib/api';
import type {
  LatLng,
  ParetoStreamEvent,
  RouteOption,
  ScenarioMode,
  ScenarioCompareResponse,
  VehicleListResponse,
  VehicleProfile,
} from './lib/types';
import { normaliseWeights, pickBestByWeightedSum, type WeightState } from './lib/weights';

type MarkerKind = 'origin' | 'destination';
type ProgressState = { done: number; total: number };

type MapViewProps = {
  origin: LatLng | null;
  destination: LatLng | null;

  selectedMarker: MarkerKind | null;

  route: RouteOption | null;

  onMapClick: (lat: number, lon: number) => void;
  onSelectMarker: (kind: MarkerKind | null) => void;
  onMoveMarker: (kind: MarkerKind, lat: number, lon: number) => void;
  onRemoveMarker: (kind: MarkerKind) => void;
  onSwapMarkers?: () => void;
};

const MapView = dynamic<MapViewProps>(() => import('./components/MapView'), {
  ssr: false,
  loading: () => <div className="mapPane" />,
});

type ParetoChartProps = {
  routes: RouteOption[];
  selectedId: string | null;
  labelsById: Record<string, string>;
  onSelect: (routeId: string) => void;
};

const ParetoChart = dynamic<ParetoChartProps>(() => import('./components/ParetoChart'), {
  ssr: false,
  loading: () => null,
});

const EtaTimelineChart = dynamic<{ route: RouteOption | null }>(
  () => import('./components/EtaTimelineChart'),
  {
    ssr: false,
    loading: () => null,
  },
);

const ScenarioComparison = dynamic<
  { data: ScenarioCompareResponse | null; loading: boolean; error: string | null }
>(() => import('./components/ScenarioComparison'), {
  ssr: false,
  loading: () => null,
});

function sortRoutesDeterministic(routes: RouteOption[]): RouteOption[] {
  return [...routes].sort((a, b) => {
    const byDuration = a.metrics.duration_s - b.metrics.duration_s;
    if (byDuration !== 0) return byDuration;
    return a.id.localeCompare(b.id);
  });
}

function dedupeWarnings(items: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];

  for (const item of items) {
    const trimmed = item.trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    out.push(trimmed);
  }

  return out;
}

function RoutesSkeleton() {
  return (
    <div className="routesSkeleton" aria-hidden="true">
      {[0, 1, 2].map((idx) => (
        <div className="routeSkeleton" key={idx}>
          <div className="routeSkeleton__row">
            <div className="routeSkeleton__line routeSkeleton__line--title shimmer" />
            <div className="routeSkeleton__line routeSkeleton__line--pill shimmer" />
          </div>
          <div className="routeSkeleton__line routeSkeleton__line--meta shimmer" />
        </div>
      ))}
    </div>
  );
}

function SidebarToggleIcon({ collapsed }: { collapsed: boolean }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.9"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`sidebarToggle__icon ${collapsed ? 'isCollapsed' : ''}`}
    >
      <rect x="3.5" y="4.5" width="17" height="15" rx="2.8" />
      <path d="M14 5v14" />
      <path d="M10.5 10l-2.5 2 2.5 2" />
    </svg>
  );
}

export default function Page() {
  const [origin, setOrigin] = useState<LatLng | null>(null);
  const [destination, setDestination] = useState<LatLng | null>(null);
  const [selectedMarker, setSelectedMarker] = useState<MarkerKind | null>(null);
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  const [vehicles, setVehicles] = useState<VehicleProfile[]>([]);
  const [vehicleType, setVehicleType] = useState<string>('rigid_hgv');
  const [scenarioMode, setScenarioMode] = useState<ScenarioMode>('no_sharing');

  const [weights, setWeights] = useState<WeightState>({ time: 60, money: 20, co2: 20 });

  const [paretoRoutes, setParetoRoutes] = useState<RouteOption[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [routeNames, setRouteNames] = useState<Record<string, string>>({});
  const [editingRouteId, setEditingRouteId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');

  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [showWarnings, setShowWarnings] = useState(false);
  const [scenarioCompare, setScenarioCompare] = useState<ScenarioCompareResponse | null>(null);
  const [scenarioCompareLoading, setScenarioCompareLoading] = useState(false);
  const [scenarioCompareError, setScenarioCompareError] = useState<string | null>(null);

  const [error, setError] = useState<string | null>(null);

  const [isPending, startTransition] = useTransition();

  const abortRef = useRef<AbortController | null>(null);
  const requestSeqRef = useRef(0);
  const routeBufferRef = useRef<RouteOption[]>([]);
  const flushTimerRef = useRef<number | null>(null);

  const clearFlushTimer = useCallback(() => {
    if (flushTimerRef.current !== null) {
      window.clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
  }, []);

  const flushRouteBuffer = useCallback(
    (seq: number) => {
      if (seq !== requestSeqRef.current) return;
      const pending = routeBufferRef.current;
      if (!pending.length) return;

      routeBufferRef.current = [];
      startTransition(() => {
        setParetoRoutes((prev) => {
          if (seq !== requestSeqRef.current) return prev;
          const seen = new Set(prev.map((route) => route.id));
          const merged = [...prev];

          for (const route of pending) {
            if (seen.has(route.id)) continue;
            seen.add(route.id);
            merged.push(route);
          }

          return merged;
        });
      });
    },
    [startTransition],
  );

  const scheduleRouteFlush = useCallback(
    (seq: number) => {
      if (flushTimerRef.current !== null) return;
      flushTimerRef.current = window.setTimeout(() => {
        flushTimerRef.current = null;
        flushRouteBuffer(seq);
      }, 80);
    },
    [flushRouteBuffer],
  );

  const flushRouteBufferNow = useCallback(
    (seq: number) => {
      clearFlushTimer();
      flushRouteBuffer(seq);
    },
    [clearFlushTimer, flushRouteBuffer],
  );

  const abortActiveCompute = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    routeBufferRef.current = [];
    clearFlushTimer();
  }, [clearFlushTimer]);

  const clearComputed = useCallback(() => {
    requestSeqRef.current += 1;
    abortActiveCompute();

    setLoading(false);
    setProgress(null);
    setWarnings([]);
    setShowWarnings(false);
    setError(null);
    setParetoRoutes([]);
    setSelectedId(null);

    setRouteNames({});
    setEditingRouteId(null);
    setEditingName('');
    setScenarioCompare(null);
    setScenarioCompareError(null);
    setScenarioCompareLoading(false);
  }, [abortActiveCompute]);

  const cancelCompute = useCallback(() => {
    requestSeqRef.current += 1;
    abortActiveCompute();
    setLoading(false);
    setProgress(null);
  }, [abortActiveCompute]);

  useEffect(() => {
    return () => {
      requestSeqRef.current += 1;
      abortActiveCompute();
    };
  }, [abortActiveCompute]);

  useEffect(() => {
    clearComputed();
  }, [vehicleType, scenarioMode, clearComputed]);

  useEffect(() => {
    const controller = new AbortController();

    void (async () => {
      try {
        const resp = await fetch('/api/vehicles', {
          signal: controller.signal,
          cache: 'no-store',
        });
        if (!resp.ok) return;

        const payload = (await resp.json()) as Partial<VehicleListResponse>;
        setVehicles(Array.isArray(payload.vehicles) ? payload.vehicles : []);
      } catch (e) {
        if (!controller.signal.aborted) {
          console.error('Failed to load vehicles:', e);
        }
      }
    })();

    return () => controller.abort();
  }, []);

  useEffect(() => {
    const best = pickBestByWeightedSum(paretoRoutes, weights);
    setSelectedId(best);
  }, [paretoRoutes, weights]);

  useEffect(() => {
    if (loading) return;

    startTransition(() => {
      setParetoRoutes((prev) => {
        const sorted = sortRoutesDeterministic(prev);
        const unchanged =
          prev.length === sorted.length && prev.every((route, idx) => route.id === sorted[idx]?.id);
        return unchanged ? prev : sorted;
      });
    });
  }, [loading, startTransition]);

  useEffect(() => {
    if (warnings.length === 0) {
      setShowWarnings(false);
    }
  }, [warnings.length]);

  const selectedRoute = useMemo(() => {
    if (!selectedId) return null;
    return paretoRoutes.find((route) => route.id === selectedId) ?? null;
  }, [paretoRoutes, selectedId]);

  const defaultLabelsById = useMemo(() => {
    const labels: Record<string, string> = {};
    for (let idx = 0; idx < paretoRoutes.length; idx += 1) {
      labels[paretoRoutes[idx].id] = `Route ${idx + 1}`;
    }
    return labels;
  }, [paretoRoutes]);

  const labelsById = useMemo(() => {
    const merged: Record<string, string> = { ...defaultLabelsById };
    for (const [routeId, name] of Object.entries(routeNames)) {
      const trimmed = name.trim();
      if (trimmed) merged[routeId] = trimmed;
    }
    return merged;
  }, [defaultLabelsById, routeNames]);

  const selectedLabel = selectedRoute ? labelsById[selectedRoute.id] ?? selectedRoute.id : null;

  const busy = loading || isPending;
  const canCompute = Boolean(origin && destination) && !busy;

  const progressText = progress ? `${Math.min(progress.done, progress.total)}/${progress.total}` : null;
  const progressPct =
    progress && progress.total > 0
      ? Math.max(0, Math.min(100, (progress.done / progress.total) * 100))
      : 0;
  const normalisedWeights = useMemo(() => normaliseWeights(weights), [weights]);

  const hasNameOverrides = Object.keys(routeNames).length > 0;

  const beginRename = useCallback(
    (routeId: string) => {
      if (busy) return;
      setEditingRouteId(routeId);
      setEditingName(routeNames[routeId] ?? labelsById[routeId] ?? '');
    },
    [busy, labelsById, routeNames],
  );

  const cancelRename = useCallback(() => {
    setEditingRouteId(null);
    setEditingName('');
  }, []);

  const commitRename = useCallback(() => {
    if (!editingRouteId) return;

    const trimmed = editingName.trim();
    setRouteNames((prev) => {
      const next = { ...prev };
      const defaultName = defaultLabelsById[editingRouteId];
      if (!trimmed || trimmed === defaultName) delete next[editingRouteId];
      else next[editingRouteId] = trimmed;
      return next;
    });

    setEditingRouteId(null);
    setEditingName('');
  }, [defaultLabelsById, editingName, editingRouteId]);

  const resetRouteNames = useCallback(() => {
    if (busy) return;
    setRouteNames({});
    setEditingRouteId(null);
    setEditingName('');
  }, [busy]);

  function handleMapClick(lat: number, lon: number) {
    setError(null);

    if (!origin) {
      setOrigin({ lat, lon });
      setSelectedMarker('origin');
      clearComputed();
      return;
    }

    if (!destination) {
      setDestination({ lat, lon });
      setSelectedMarker('destination');
      clearComputed();
      return;
    }

    if (selectedMarker === 'origin') {
      setOrigin({ lat, lon });
      clearComputed();
      return;
    }

    setDestination({ lat, lon });
    setSelectedMarker('destination');
    clearComputed();
  }

  function handleMoveMarker(kind: MarkerKind, lat: number, lon: number) {
    setError(null);

    if (kind === 'origin') setOrigin({ lat, lon });
    else setDestination({ lat, lon });

    setSelectedMarker(kind);
    clearComputed();
  }

  function handleRemoveMarker(kind: MarkerKind) {
    setError(null);

    if (kind === 'origin') setOrigin(null);
    else setDestination(null);

    setSelectedMarker(null);
    clearComputed();
  }

  function swapMarkers() {
    if (!origin || !destination) return;
    setOrigin(destination);
    setDestination(origin);
    setSelectedMarker(null);
    clearComputed();
  }

  function reset() {
    setOrigin(null);
    setDestination(null);
    setSelectedMarker(null);
    clearComputed();
    setError(null);
  }

  async function computePareto() {
    if (!origin || !destination) {
      setError('Click the map to set Start, then Destination.');
      return;
    }

    const seq = requestSeqRef.current + 1;
    requestSeqRef.current = seq;

    abortActiveCompute();

    const controller = new AbortController();
    abortRef.current = controller;

    routeBufferRef.current = [];
    clearFlushTimer();

    setLoading(true);
    setError(null);
    setProgress(null);
    setWarnings([]);
    setShowWarnings(false);
    setParetoRoutes([]);
    setSelectedId(null);
    setRouteNames({});
    setEditingRouteId(null);
    setEditingName('');
    setScenarioCompare(null);
    setScenarioCompareError(null);
    setScenarioCompareLoading(false);

    const body = {
      origin,
      destination,
      vehicle_type: vehicleType,
      scenario_mode: scenarioMode,
      max_alternatives: 5,
    };

    let sawDone = false;

    try {
      await postNDJSON<ParetoStreamEvent>('/api/pareto/stream', body, {
        signal: controller.signal,
        onEvent: (event) => {
          if (seq !== requestSeqRef.current) return;

          switch (event.type) {
            case 'meta': {
              setProgress({ done: 0, total: event.total });
              return;
            }

            case 'route': {
              routeBufferRef.current.push(event.route);
              scheduleRouteFlush(seq);
              setProgress({ done: event.done, total: event.total });
              return;
            }

            case 'error': {
              setProgress({ done: event.done, total: event.total });
              setWarnings((prev) => dedupeWarnings([...prev, event.message]));
              return;
            }

            case 'fatal': {
              setError(event.message || 'Route computation failed.');
              return;
            }

            case 'done': {
              sawDone = true;
              flushRouteBufferNow(seq);

              const finalRoutes = sortRoutesDeterministic(event.routes ?? []);
              startTransition(() => {
                setParetoRoutes(finalRoutes);
              });

              setProgress({ done: event.done, total: event.total });
              const doneWarnings = event.warnings ?? [];
              if (doneWarnings.length) {
                setWarnings((prev) => dedupeWarnings([...prev, ...doneWarnings]));
              }
              return;
            }

            default:
              return;
          }
        },
      });

      if (seq === requestSeqRef.current) {
        flushRouteBufferNow(seq);
        if (!sawDone) {
          startTransition(() => {
            setParetoRoutes((prev) => sortRoutesDeterministic(prev));
          });
        }
      }
    } catch (e: unknown) {
      if (seq !== requestSeqRef.current) return;
      if (controller.signal.aborted) return;
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      if (seq === requestSeqRef.current) {
        abortRef.current = null;
        setLoading(false);
      }
    }
  }

  async function compareScenarios() {
    if (!origin || !destination) {
      setScenarioCompareError('Set origin and destination before comparing scenarios.');
      return;
    }

    setScenarioCompareLoading(true);
    setScenarioCompareError(null);
    try {
      const body = {
        origin,
        destination,
        vehicle_type: vehicleType,
        max_alternatives: 5,
        weights: {
          time: weights.time,
          money: weights.money,
          co2: weights.co2,
        },
      };
      const payload = await postJSON<ScenarioCompareResponse>('/api/scenario/compare', body);
      setScenarioCompare(payload);
    } catch (e: unknown) {
      setScenarioCompareError(e instanceof Error ? e.message : 'Failed to compare scenarios');
    } finally {
      setScenarioCompareLoading(false);
    }
  }

  const m = selectedRoute?.metrics ?? null;

  const vehicleOptions = vehicles.length
    ? vehicles.map((v) => ({ value: v.id, label: v.label }))
    : [
        { value: 'van', label: 'Van' },
        { value: 'rigid_hgv', label: 'Rigid HGV' },
        { value: 'artic_hgv', label: 'Articulated HGV' },
      ];

  const scenarioOptions: { value: ScenarioMode; label: string; description: string }[] = [
    {
      value: 'no_sharing',
      label: 'No sharing',
      description: 'Baseline: no expected delay uplift.',
    },
    {
      value: 'partial_sharing',
      label: 'Partial sharing',
      description: 'Small delay uplift, some coordination overhead.',
    },
    {
      value: 'full_sharing',
      label: 'Full sharing',
      description: 'Best coordination: lowest expected delay uplift.',
    },
  ];

  const mapHint = (() => {
    if (!origin) return 'Click the map to set Start.';
    if (origin && !destination) return 'Now click the map to set Destination.';
    if (loading) {
      return progressText
        ? `Computing routes... (${progressText})`
        : 'Computing routes...';
    }
    return 'Compute Pareto to compare candidate routes. Use sliders to pick the best trade-off.';
  })();

  const showRoutesSection = loading || paretoRoutes.length > 0 || warnings.length > 0;
  const canCompareScenarios = Boolean(origin && destination) && !busy && !scenarioCompareLoading;
  const sidebarToggleLabel = isPanelCollapsed ? 'Extend sidebar' : 'Collapse sidebar';

  return (
    <div className="app">
      <div className="mapStage">
        <MapView
          origin={origin}
          destination={destination}
          selectedMarker={selectedMarker}
          route={selectedRoute}
          onMapClick={handleMapClick}
          onSelectMarker={setSelectedMarker}
          onMoveMarker={handleMoveMarker}
          onRemoveMarker={handleRemoveMarker}
          onSwapMarkers={swapMarkers}
        />

        <div className="mapHUD">
          {busy && (
            <div className="mapHUD__status" role="status" aria-live="polite">
              <div className="mapHUD__statusLine">
                <span className="spinner" />
                <span>
                  Computing routes{progressText ? ` (${progressText})` : '...'}
                </span>
              </div>

              {progress?.total ? (
                <div className="hudProgress" aria-hidden="true">
                  <div className="hudProgress__fill" style={{ width: `${progressPct}%` }} />
                </div>
              ) : null}
            </div>
          )}

          <div className="mapHUD__hint">{mapHint}</div>
        </div>
      </div>

      <button
        type="button"
        className={`sidebarToggle ${isPanelCollapsed ? 'isCollapsed' : ''}`}
        onClick={() => setIsPanelCollapsed((prev) => !prev)}
        aria-label={sidebarToggleLabel}
        aria-pressed={isPanelCollapsed}
        title={sidebarToggleLabel}
      >
        <SidebarToggleIcon collapsed={isPanelCollapsed} />
      </button>

      <aside className={`panel ${isPanelCollapsed ? 'isCollapsed' : ''}`} aria-hidden={isPanelCollapsed}>
            <header className="panelHeader">
              <div className="panelHeader__top">
                <h1>Carbon‑Aware Freight Router</h1>
                <div className="panelHeader__actions">
                  <span className="badge">v0</span>
                </div>
              </div>
              <p className="subtitle">
                Click the map to set Start, then Destination. Compute Pareto to generate candidate
                routes, then use the sliders to choose the best trade-off (time vs cost vs CO2).
              </p>
            </header>

            <div id="app-sidebar-content" className="panelBody">
              <section className="card">
                <div className="sectionTitle">Setup</div>

                <div className="fieldLabel">Vehicle type</div>
                <Select
                  ariaLabel="Vehicle type"
                  value={vehicleType}
                  options={vehicleOptions}
                  onChange={setVehicleType}
                  disabled={busy}
                />

                <div className="fieldLabel">Scenario mode</div>
                <Select
                  ariaLabel="Scenario mode"
                  value={scenarioMode}
                  options={scenarioOptions}
                  onChange={setScenarioMode}
                  disabled={busy}
                />

                <div className="helper">
                  Scenario mode applies a policy-based delay multiplier and adds idle emissions (and
                  driver time cost) for the extra delay. This is a lightweight stub that can be
                  swapped for a detailed simulator later.
                </div>

                <div className="row" style={{ marginTop: 12 }}>
                  <button
                    className="secondary"
                    onClick={swapMarkers}
                    disabled={!origin || !destination || busy}
                    title="Swap start and destination"
                  >
                    Swap pins
                  </button>
                  <button className="secondary" onClick={reset} disabled={busy}>
                    Clear pins
                  </button>
                </div>
              </section>

              <section className="card">
                <div className="sectionTitle">Preferences</div>

                <div className="sliderField">
                  <div className="sliderField__head">
                    <label htmlFor="weight-time">Time</label>
                    <span className="sliderField__value">{weights.time}</span>
                  </div>
                  <p id="weight-time-help" className="sliderField__desc">
                    Prioritises shorter delivery duration. Increase this to favour faster routes.
                  </p>
                  <input
                    id="weight-time"
                    type="range"
                    min={0}
                    max={100}
                    value={weights.time}
                    aria-describedby="weight-time-help"
                    onChange={(e) => setWeights((w) => ({ ...w, time: Number(e.target.value) }))}
                  />
                </div>

                <div className="sliderField">
                  <div className="sliderField__head">
                    <label htmlFor="weight-money">Money</label>
                    <span className="sliderField__value">{weights.money}</span>
                  </div>
                  <p id="weight-money-help" className="sliderField__desc">
                    Prioritises lower operating cost. Increase this to favour cheaper routes.
                  </p>
                  <input
                    id="weight-money"
                    type="range"
                    min={0}
                    max={100}
                    value={weights.money}
                    aria-describedby="weight-money-help"
                    onChange={(e) => setWeights((w) => ({ ...w, money: Number(e.target.value) }))}
                  />
                </div>

                <div className="sliderField">
                  <div className="sliderField__head">
                    <label htmlFor="weight-co2">CO2</label>
                    <span className="sliderField__value">{weights.co2}</span>
                  </div>
                  <p id="weight-co2-help" className="sliderField__desc">
                    Prioritises lower carbon emissions. Increase this to favour cleaner routes.
                  </p>
                  <input
                    id="weight-co2"
                    type="range"
                    min={0}
                    max={100}
                    value={weights.co2}
                    aria-describedby="weight-co2-help"
                    onChange={(e) => setWeights((w) => ({ ...w, co2: Number(e.target.value) }))}
                  />
                </div>

                <div className="row row--actions" style={{ marginTop: 12 }}>
                  <button className="primary" onClick={computePareto} disabled={!canCompute}>
                    {busy ? (
                      <span className="buttonLabel">
                        <span className="spinner spinner--inline" />
                        <span>
                          Computing...{progressText ? ` ${progressText}` : ''}
                        </span>
                      </span>
                    ) : (
                      'Compute Pareto'
                    )}
                  </button>

                  <button
                    className="secondary"
                    onClick={clearComputed}
                    disabled={busy || paretoRoutes.length === 0}
                  >
                    Clear results
                  </button>

                  {loading && (
                    <button className="secondary" onClick={cancelCompute}>
                      Cancel
                    </button>
                  )}
                </div>

                <div className="tiny">
                  Relative influence: Time {(normalisedWeights.time * 100).toFixed(0)}% | Money{' '}
                  {(normalisedWeights.money * 100).toFixed(0)}% | CO2{' '}
                  {(normalisedWeights.co2 * 100).toFixed(0)}%
                </div>

                {error && <div className="error">{error}</div>}
              </section>

          {m && (
            <section className="card">
              <div className="sectionTitle">Selected route</div>
              <div className="metrics">
                <div className="metric">
                  <div className="metric__label">Distance</div>
                  <div className="metric__value">{m.distance_km.toFixed(2)} km</div>
                </div>
                <div className="metric">
                  <div className="metric__label">Duration</div>
                  <div className="metric__value">{(m.duration_s / 60).toFixed(1)} min</div>
                </div>
                <div className="metric">
                  <div className="metric__label">£ (proxy)</div>
                  <div className="metric__value">{m.monetary_cost.toFixed(2)}</div>
                </div>
                <div className="metric">
                  <div className="metric__label">CO2</div>
                  <div className="metric__value">{m.emissions_kg.toFixed(3)} kg</div>
                </div>
                <div className="metric">
                  <div className="metric__label">Avg speed</div>
                  <div className="metric__value">{m.avg_speed_kmh.toFixed(1)} km/h</div>
                </div>
                {selectedLabel && (
                  <div className="metric">
                    <div className="metric__label">Route</div>
                    <div className="metric__value">{selectedLabel}</div>
                  </div>
                )}
              </div>

              {selectedRoute?.eta_explanations?.length ? (
                <div style={{ marginTop: 12 }}>
                  <div className="fieldLabel" style={{ marginBottom: 6 }}>
                    ETA explanation
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 16 }}>
                    {selectedRoute.eta_explanations.map((item, idx) => (
                      <li key={`${idx}-${item}`} className="tiny">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              <div style={{ marginTop: 12 }}>
                <EtaTimelineChart route={selectedRoute} />
              </div>
            </section>
          )}

          {showRoutesSection && (
            <section className={`card routesSection ${isPending ? 'isUpdating' : ''}`}>
              <div className="sectionTitleRow">
                <div className="sectionTitle">Routes</div>

                <div className="sectionTitleMeta">
                  {loading && <span className="statusPill">Computing {progressText ?? '...'}</span>}

                  {warnings.length > 0 && (
                    <button
                      type="button"
                      className={`warningPill warningPill--button ${showWarnings ? 'isOpen' : ''}`}
                      onClick={() => setShowWarnings((prev) => !prev)}
                      aria-expanded={showWarnings}
                      aria-controls="route-warning-list"
                    >
                      {warnings.length} warning{warnings.length === 1 ? '' : 's'}
                    </button>
                  )}

                  {hasNameOverrides && (
                    <button
                      type="button"
                      className="ghostButton"
                      onClick={resetRouteNames}
                      disabled={busy}
                    >
                      Reset names
                    </button>
                  )}
                </div>
              </div>

              {warnings.length > 0 && showWarnings && (
                <div
                  id="route-warning-list"
                  className="warningPanel"
                  role="region"
                  aria-labelledby="route-warning-title"
                >
                  <div id="route-warning-title" className="warningPanel__title">
                    Route generation warnings
                  </div>
                  <div className="warningPanel__hint">
                    Routing succeeded, but some candidate requests were skipped or failed.
                  </div>
                  <ul className="warningPanel__list">
                    {warnings.map((warning, idx) => (
                      <li key={`${idx}-${warning}`}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {loading && paretoRoutes.length === 0 ? (
                <RoutesSkeleton />
              ) : (
                <>
                  {paretoRoutes.length > 0 && (
                    <>
                      <div className="chartWrap">
                        <ParetoChart
                          routes={paretoRoutes}
                          selectedId={selectedId}
                          labelsById={labelsById}
                          onSelect={setSelectedId}
                        />
                      </div>

                      <div className="helper" style={{ marginTop: 10 }}>
                        Tip: click a point (or a route card) to lock a specific route. Moving
                        sliders will re-select the best route for your weights.
                        {paretoRoutes.length === 1 && (
                          <>
                            <br />
                            Only one route was generated for this pair - try moving the pins a
                            little.
                          </>
                        )}
                      </div>

                      <ul className="routeList">
                        {paretoRoutes.map((route, idx) => {
                          const label = labelsById[route.id] ?? `Route ${idx + 1}`;
                          const isEditing = editingRouteId === route.id;
                          const isSelected = route.id === selectedId;

                          return (
                            <li
                              key={route.id}
                              className={`routeCard ${isSelected ? 'isSelected' : ''}`}
                              role="button"
                              tabIndex={0}
                              aria-label={`Select ${label}`}
                              aria-pressed={isSelected}
                              onClick={() => setSelectedId(route.id)}
                              onKeyDown={(event) => {
                                if (event.key === 'Enter' || event.key === ' ') {
                                  event.preventDefault();
                                  setSelectedId(route.id);
                                }
                              }}
                            >
                              <div className="routeCard__top">
                                {isEditing ? (
                                  <div
                                    className="routeRename"
                                    onClick={(event) => event.stopPropagation()}
                                  >
                                    <input
                                      className="routeRename__input"
                                      value={editingName}
                                      onChange={(event) => setEditingName(event.target.value)}
                                      onKeyDown={(event) => {
                                        if (event.key === 'Enter') {
                                          event.preventDefault();
                                          commitRename();
                                        }
                                        if (event.key === 'Escape') {
                                          event.preventDefault();
                                          cancelRename();
                                        }
                                      }}
                                      autoFocus
                                      aria-label="Route name"
                                    />
                                    <button
                                      type="button"
                                      className="routeRename__btn"
                                      onClick={commitRename}
                                    >
                                      Save
                                    </button>
                                    <button
                                      type="button"
                                      className="routeRename__btn routeRename__btn--secondary"
                                      onClick={cancelRename}
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                ) : (
                                  <div className="routeCard__titleWrap">
                                    <div
                                      className="routeCard__id"
                                      onDoubleClick={(event) => {
                                        event.stopPropagation();
                                        beginRename(route.id);
                                      }}
                                    >
                                      {label}
                                    </div>

                                    <button
                                      type="button"
                                      className="routeCard__renameBtn"
                                      disabled={busy}
                                      onClick={(event) => {
                                        event.stopPropagation();
                                        beginRename(route.id);
                                      }}
                                    >
                                      Rename
                                    </button>
                                  </div>
                                )}

                                <div className="routeCard__pill">
                                  {(route.metrics.duration_s / 60).toFixed(1)} min
                                </div>
                              </div>

                              <div className="routeCard__meta">
                                <span>{route.metrics.emissions_kg.toFixed(3)} kg CO2</span>
                                <span>£{route.metrics.monetary_cost.toFixed(2)}</span>
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                    </>
                  )}
                </>
              )}
            </section>
          )}

          <section className="card">
            <div className="sectionTitleRow">
              <div className="sectionTitle">Scenario comparison</div>
              <button className="secondary" onClick={compareScenarios} disabled={!canCompareScenarios}>
                {scenarioCompareLoading ? 'Comparing...' : 'Compare scenarios'}
              </button>
            </div>
            <ScenarioComparison
              data={scenarioCompare}
              loading={scenarioCompareLoading}
              error={scenarioCompareError}
            />
          </section>
            </div>
      </aside>
    </div>
  );
}
