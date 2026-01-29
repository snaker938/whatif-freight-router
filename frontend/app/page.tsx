'use client';
// frontend/app/page.tsx

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useMemo, useState } from 'react';

import Select from './components/Select';
import { postJSON } from './lib/api';
import type {
  LatLng,
  ParetoResponse,
  RouteOption,
  ScenarioMode,
  VehicleListResponse,
  VehicleProfile,
} from './lib/types';
import { normaliseWeights, pickBestByWeightedSum, type WeightState } from './lib/weights';

type MarkerKind = 'origin' | 'destination';

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
  onSelect: (routeId: string) => void;
};

const ParetoChart = dynamic<ParetoChartProps>(() => import('./components/ParetoChart'), {
  ssr: false,
  loading: () => null,
});

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

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canCompute = Boolean(origin && destination) && !loading;

  const clearComputed = useCallback(() => {
    setParetoRoutes([]);
    setSelectedId(null);
  }, []);

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

  const selectedRoute = useMemo(() => {
    if (!selectedId) return null;
    return paretoRoutes.find((r) => r.id === selectedId) ?? null;
  }, [paretoRoutes, selectedId]);

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
    setLoading(true);
    setError(null);

    try {
      const body = {
        origin,
        destination,
        vehicle_type: vehicleType,
        scenario_mode: scenarioMode,
        max_alternatives: 5,
      };

      const json = await postJSON<ParetoResponse>('/api/pareto', body);
      setParetoRoutes(json.routes);
    } catch (e: any) {
      setError(e?.message ?? 'Unknown error');
    } finally {
      setLoading(false);
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
    return 'Compute Pareto to compare candidate routes. Use sliders to pick the best trade-off.';
  })();

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
              disabled={loading}
            />

            <label>Scenario mode</label>
            <Select
              ariaLabel="Scenario mode"
              value={scenarioMode}
              options={scenarioOptions}
              onChange={setScenarioMode}
              disabled={loading}
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
                disabled={!origin || !destination || loading}
                title="Swap start and destination"
              >
                Swap pins
              </button>
              <button className="secondary" onClick={reset} disabled={loading}>
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

            <div className="row" style={{ marginTop: 12 }}>
              <button className="primary" onClick={computePareto} disabled={!canCompute}>
                {loading ? 'Computing…' : 'Compute Pareto'}
              </button>
              <button
                className="secondary"
                onClick={clearComputed}
                disabled={loading || paretoRoutes.length === 0}
              >
                Clear results
              </button>
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
                {selectedId && (
                  <div className="metric">
                    <div className="metric__label">Route ID</div>
                    <div className="metric__value">{selectedId}</div>
                  </div>
                )}
              </div>
            </section>
          )}

          {paretoRoutes.length > 0 && (
            <section className="card">
              <div className="sectionTitle">Pareto space</div>

              <div className="chartWrap">
                <ParetoChart
                  routes={paretoRoutes}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                />
              </div>

              <div className="helper" style={{ marginTop: 10 }}>
                Tip: click a point (or a route card) to lock a specific route. Moving sliders will
                re-select the best route for your weights.
                {paretoRoutes.length === 1 && (
                  <>
                    <br />
                    Only one route was generated for this pair — try moving the pins a little.
                  </>
                )}
              </div>

              <ul className="routeList">
                {paretoRoutes.map((r) => (
                  <li key={r.id}>
                    <button
                      type="button"
                      className={`routeCard ${r.id === selectedId ? 'isSelected' : ''}`}
                      onClick={() => setSelectedId(r.id)}
                    >
                      <div className="routeCard__top">
                        <div className="routeCard__id">{r.id}</div>
                        <div className="routeCard__pill">
                          {(r.metrics.duration_s / 60).toFixed(1)} min
                        </div>
                      </div>
                      <div className="routeCard__meta">
                        <span>{r.metrics.emissions_kg.toFixed(3)} kg CO₂</span>
                        <span>£{r.metrics.monetary_cost.toFixed(2)}</span>
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      </aside>
    </div>
  );
}
