'use client';

import { useEffect, useMemo, useState } from 'react';

import type { VehicleProfile } from '../../lib/types';

type Props = {
  vehicles: VehicleProfile[];
  loading: boolean;
  saving: boolean;
  error: string | null;
  onRefresh: () => void;
  onCreate: (vehicle: VehicleProfile) => Promise<void> | void;
  onUpdate: (vehicleId: string, vehicle: VehicleProfile) => Promise<void> | void;
  onDelete: (vehicleId: string) => Promise<void> | void;
};

const DEFAULT_CREATE_JSON = JSON.stringify(
  {
    id: 'custom_hgv_demo',
    label: 'Custom HGV Demo',
    mass_tonnes: 18,
    emission_factor_kg_per_tkm: 0.095,
    cost_per_km: 1.4,
    cost_per_hour: 26,
    idle_emissions_kg_per_hour: 3.2,
    powertrain: 'ice',
    schema_version: 2,
    vehicle_class: 'rigid_hgv',
    toll_vehicle_class: 'rigid_hgv',
    toll_axle_class: '3to4',
    fuel_surface_class: 'rigid_hgv',
    risk_bucket: 'rigid_hgv',
    stochastic_bucket: 'rigid_hgv',
    terrain_params: {
      mass_kg: 26000,
      c_rr: 0.0082,
      drag_area_m2: 7.3,
      drivetrain_efficiency: 0.88,
      regen_efficiency: 0.14,
    },
    aliases: ['custom_hgv_alias'],
  },
  null,
  2,
);

function safeParseVehicle(json: string): VehicleProfile {
  const parsed = JSON.parse(json) as unknown;
  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Vehicle payload must be a JSON object.');
  }
  return parsed as VehicleProfile;
}

export default function CustomVehicleManager({
  vehicles,
  loading,
  saving,
  error,
  onRefresh,
  onCreate,
  onUpdate,
  onDelete,
}: Props) {
  const [selectedId, setSelectedId] = useState<string>('');
  const [createJson, setCreateJson] = useState(DEFAULT_CREATE_JSON);
  const [editJson, setEditJson] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);

  const selectedVehicle = useMemo(
    () => vehicles.find((vehicle) => vehicle.id === selectedId) ?? null,
    [selectedId, vehicles],
  );

  useEffect(() => {
    if (!selectedVehicle) {
      setEditJson('');
      return;
    }
    setEditJson(JSON.stringify(selectedVehicle, null, 2));
  }, [selectedVehicle]);

  async function handleCreate() {
    try {
      setLocalError(null);
      const vehicle = safeParseVehicle(createJson);
      await onCreate(vehicle);
      setSelectedId(vehicle.id);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Invalid vehicle JSON.');
    }
  }

  async function handleUpdate() {
    if (!selectedVehicle) return;
    try {
      setLocalError(null);
      const vehicle = safeParseVehicle(editJson);
      await onUpdate(selectedVehicle.id, vehicle);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Invalid vehicle JSON.');
    }
  }

  async function handleDelete() {
    if (!selectedVehicle) return;
    const confirmed = window.confirm(`Delete custom vehicle "${selectedVehicle.id}"?`);
    if (!confirmed) return;
    await onDelete(selectedVehicle.id);
    setSelectedId('');
  }

  return (
    <div className="devCard">
      <div className="devCard__head">
        <h4 className="devCard__title">Custom Vehicles</h4>
        <div className="row">
          <button type="button" className="secondary" onClick={onRefresh} disabled={loading || saving}>
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      {error ? <div className="error">{error}</div> : null}
      {localError ? <div className="error">{localError}</div> : null}

      <div className="fieldLabel">Create Vehicle (JSON)</div>
      <textarea
        className="input devTextarea"
        value={createJson}
        disabled={saving}
        onChange={(event) => setCreateJson(event.target.value)}
      />
      <div className="actionGrid actionGrid--single u-mt10">
        <button type="button" className="secondary" onClick={handleCreate} disabled={saving}>
          {saving ? 'Saving...' : 'Create Vehicle'}
        </button>
      </div>

      <div className="fieldLabelRow u-mt12">
        <label className="fieldLabel" htmlFor="custom-vehicle-select">
          Existing Custom Vehicle
        </label>
      </div>
      <select
        id="custom-vehicle-select"
        className="input"
        value={selectedId}
        onChange={(event) => setSelectedId(event.target.value)}
        disabled={loading || saving}
      >
        <option value="">Select vehicle</option>
        {vehicles.map((vehicle) => (
          <option key={vehicle.id} value={vehicle.id}>
            {vehicle.id}
          </option>
        ))}
      </select>

      {selectedVehicle ? (
        <>
          <div className="fieldLabel u-mt12">Edit Vehicle (JSON)</div>
          <textarea
            className="input devTextarea"
            value={editJson}
            disabled={saving}
            onChange={(event) => setEditJson(event.target.value)}
          />
          <div className="row u-mt10">
            <button type="button" className="secondary" onClick={handleUpdate} disabled={saving}>
              {saving ? 'Saving...' : 'Update Vehicle'}
            </button>
            <button type="button" className="secondary danger" onClick={handleDelete} disabled={saving}>
              Delete Vehicle
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}
