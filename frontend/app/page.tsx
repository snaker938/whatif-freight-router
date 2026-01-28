'use client';

import dynamic from 'next/dynamic';
import { useEffect, useMemo, useState } from 'react';

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

type MapViewProps = {
  origin: LatLng | null;
  destination: LatLng | null;
  route: RouteOption | null;
  onMapClick: (lat: number, lon: number) => void;
};

const MapView = dynamic<MapViewProps>(() => import('./components/MapView'), {
  ssr: false,
  // Keep layout stable during SSR/prerender
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

  const [vehicles, setVehicles] = useState<VehicleProfile[]>([]);
  const [vehicleType, setVehicleType] = useState<string>('rigid_hgv');
  const [scenarioMode, setScenarioMode] = useState<ScenarioMode>('no_sharing');

  const [weights, setWeights] = useState<WeightState>({ time: 60, money: 20, co2: 20 });

  const [paretoRoutes, setParetoRoutes] = useState<RouteOption[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canCompute = Boolean(origin && destination) && !loading;

  // If the user changes scenario/vehicle after computing, invalidate the current route set.
  useEffect(() => {
    setParetoRoutes([]);
    setSelectedId(null);
  }, [vehicleType, scenarioMode]);

  useEffect(() => {
    // Load vehicles dynamically so backend changes auto-propagate
    fetch('/api/vehicles')
      .then((r) => r.json())
      .then((j: VehicleListResponse) => setVehicles(j.vehicles ?? []))
      .catch(() => {
        // Keep silent; UI still works with default select options
      });
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
      return;
    }
    if (!destination) {
      setDestination({ lat, lon });
      return;
    }

    // Quick iteration: update destination, clear computed routes
    setDestination({ lat, lon });
    setParetoRoutes([]);
    setSelectedId(null);
  }

  function reset() {
    setOrigin(null);
    setDestination(null);
    setParetoRoutes([]);
    setSelectedId(null);
    setError(null);
  }

  async function computePareto() {
    if (!origin || !destination) {
      setError('Click map to set origin then destination.');
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

  return (
    <div className="app">
      <div className="mapStage">
        <MapView
          origin={origin}
          destination={destination}
          route={selectedRoute}
          onMapClick={handleMapClick}
        />

        <div className="mapHUD">
          <div className="mapHUD__brand">
            <span className="mapHUD__dot" />
            <span>UK demo</span>
          </div>
          <div className="mapHUD__hint">
            {!origin && 'Click the map to drop an origin pin.'}
            {origin && !destination && 'Now click to drop the destination pin.'}
            {origin && destination && 'Adjust weights or recompute with a new destination.'}
          </div>
        </div>
      </div>

      <aside className="panel">
        <header className="panelHeader">
          <div className="panelHeader__top">
            <h1>Carbon‑Aware Freight Router</h1>
            <span className="badge">v0</span>
          </div>
          <p className="subtitle">
            Click map: origin → destination. Compute Pareto. Sliders re-select the best route among
            candidates.
          </p>
        </header>

        <div className="panelBody">
          <section className="card">
            <div className="sectionTitle">Setup</div>

            <label>Vehicle type</label>
            <select value={vehicleType} onChange={(e) => setVehicleType(e.target.value)}>
              {vehicleOptions.map((v) => (
                <option key={v.value} value={v.value}>
                  {v.label}
                </option>
              ))}
            </select>

            <label>Scenario mode</label>
            <select
              value={scenarioMode}
              onChange={(e) => setScenarioMode(e.target.value as ScenarioMode)}
            >
              <option value="no_sharing">No sharing</option>
              <option value="partial_sharing">Partial sharing</option>
              <option value="full_sharing">Full sharing</option>
            </select>

            <div className="helper">
              Scenario changes expected delay + idle emissions via policy (stub, replaceable with
              simulation later).
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
              <button className="secondary" onClick={reset} disabled={loading}>
                Reset
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
                    Only one route was returned for this pair — try a longer trip or different pins
                    to see more trade-offs.
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
