'use client';
// frontend/app/page.tsx

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useMemo, useRef, useState, useTransition } from 'react';

import Select from './components/Select';
import { postNDJSON } from './lib/api';
import type {
  LatLng,
  ParetoStreamEvent,
  RouteOption,
  ScenarioMode,
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

function sortRoutesDeterministic(routes: RouteOption[]): RouteOption[] {
  return [...routes].sort((a, b) => {
    const byDuration = a.metrics.duration_s - b.metrics.duration_s;
    if (byDuration !== 0) return byDuration;
    return a.id.localeCompare(b.id);
  });
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

export default function Page() {
  const [origin, setOrigin] = useState<LatLng | null>(null);
  const [destination, setDestination] = useState<LatLng | null>(null);
  const [selectedMarker, setSelectedMarker] = useState<MarkerKind | null>(null);

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
    setParetoRoutes([]);
    setSelectedId(null);

    setRouteNames({});
    setEditingRouteId(null);
    setEditingName('');
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
    fetch('/api/vehicles')
      .then((r) => r.json())
      .then((j: VehicleListResponse) => setVehicles(j.vehicles ?? []))
      .catch(() => {});
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
    setParetoRoutes([]);
    setSelectedId(null);
    setRouteNames({});
    setEditingRouteId(null);
    setEditingName('');

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
              setWarnings((prev) =>
                prev.includes(event.message) ? prev : [...prev, event.message],
              );
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
              if (event.warnings?.length) {
                setWarnings(event.warnings);
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
  const warningTitle = warnings.slice(0, 8).join('\n');

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

      <aside className="panel">
        <header className="panelHeader">
          <div className="panelHeader__top">
            <h1>Carbon‑Aware Freight Router</h1>
            <span className="badge">v0</span>
          </div>
          <p className="subtitle">
            Click the map to set Start, then Destination. Compute Pareto to generate candidate
            routes, then use the sliders to choose the best trade‑off (time vs cost vs CO₂).
          </p>
        </header>

        <div className="panelBody">
          <section className="card">
            <div className="sectionTitle">Setup</div>

            <label>Vehicle type</label>
            <Select
              ariaLabel="Vehicle type"
              value={vehicleType}
              options={vehicleOptions}
              onChange={setVehicleType}
              disabled={busy}
            />

            <label>Scenario mode</label>
            <Select
              ariaLabel="Scenario mode"
              value={scenarioMode}
              options={scenarioOptions}
              onChange={setScenarioMode}
              disabled={busy}
            />

            <div className="helper">
              Scenario mode applies a policy‑based delay multiplier and adds idle emissions (and
              driver time cost) for the extra delay. This is a lightweight stub that can be swapped
              for a detailed simulator later.
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

            <label>Time ({weights.time})</label>
            <input
              type="range"
              min={0}
              max={100}
              value={weights.time}
              onChange={(e) => setWeights((w) => ({ ...w, time: Number(e.target.value) }))}
            />

            <label>Money ({weights.money})</label>
            <input
              type="range"
              min={0}
              max={100}
              value={weights.money}
              onChange={(e) => setWeights((w) => ({ ...w, money: Number(e.target.value) }))}
            />

            <label>CO₂ ({weights.co2})</label>
            <input
              type="range"
              min={0}
              max={100}
              value={weights.co2}
              onChange={(e) => setWeights((w) => ({ ...w, co2: Number(e.target.value) }))}
            />

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

            <div className="tiny">Normalised: {JSON.stringify(normaliseWeights(weights))}</div>

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
                  <div className="metric__label">CO₂</div>
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
            </section>
          )}

          {showRoutesSection && (
            <section className={`card routesSection ${isPending ? 'isUpdating' : ''}`}>
              <div className="sectionTitleRow">
                <div className="sectionTitle">Routes</div>

                <div className="sectionTitleMeta">
                  {loading && (
                    <span className="statusPill">
                      Computing {progressText ?? '...'}
                    </span>
                  )}

                  {warnings.length > 0 && (
                    <span className="warningPill" title={warningTitle}>
                      {warnings.length} warning{warnings.length === 1 ? '' : 's'}
                    </span>
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

                          return (
                            <li
                              key={route.id}
                              className={`routeCard ${route.id === selectedId ? 'isSelected' : ''}`}
                              role="button"
                              tabIndex={0}
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
                                <span>{route.metrics.emissions_kg.toFixed(3)} kg CO₂</span>
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
        </div>
      </aside>
    </div>
  );
}
