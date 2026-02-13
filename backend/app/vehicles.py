from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .settings import settings


class VehicleProfile(BaseModel):
    """Starter vehicle parameters.

    Replace placeholders with your validated factors (DEFRA/HBEFA/etc).
    Keep these here for now; later you can load from JSON/YAML.
    """

    id: str
    label: str

    mass_tonnes: float = Field(..., gt=0)
    emission_factor_kg_per_tkm: float = Field(..., gt=0)

    cost_per_km: float = Field(..., ge=0)
    cost_per_hour: float = Field(..., ge=0)

    # Used when scenarios add 'extra time' (idle/queuing) to a route
    idle_emissions_kg_per_hour: float = Field(default=0.0, ge=0)
    powertrain: Literal["ice", "ev"] = "ice"
    ev_kwh_per_km: float | None = Field(default=None, ge=0.0)
    grid_co2_kg_per_kwh: float | None = Field(default=None, ge=0.0)


VEHICLES: dict[str, VehicleProfile] = {
    "van": VehicleProfile(
        id="van",
        label="Delivery van",
        mass_tonnes=2.0,
        emission_factor_kg_per_tkm=0.18,
        cost_per_km=0.35,
        cost_per_hour=18.0,
        idle_emissions_kg_per_hour=1.2,
    ),
    "rigid_hgv": VehicleProfile(
        id="rigid_hgv",
        label="Rigid HGV",
        mass_tonnes=12.0,
        emission_factor_kg_per_tkm=0.10,
        cost_per_km=0.65,
        cost_per_hour=28.0,
        idle_emissions_kg_per_hour=3.5,
    ),
    "artic_hgv": VehicleProfile(
        id="artic_hgv",
        label="Articulated HGV",
        mass_tonnes=20.0,
        emission_factor_kg_per_tkm=0.08,
        cost_per_km=0.75,
        cost_per_hour=32.0,
        idle_emissions_kg_per_hour=4.5,
    ),
}

DEFAULT_VEHICLE_ID = "rigid_hgv"
VEHICLE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,47}$")


def _custom_vehicle_path() -> Path:
    p = Path(settings.out_dir) / "config" / "vehicles.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_custom_vehicles() -> dict[str, VehicleProfile]:
    path = _custom_vehicle_path()
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, list):
        return {}

    out: dict[str, VehicleProfile] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            vehicle = VehicleProfile.model_validate(item)
        except Exception:
            continue
        out[vehicle.id] = vehicle
    return out


def save_custom_vehicles(vehicles: dict[str, VehicleProfile]) -> Path:
    path = _custom_vehicle_path()
    payload = [vehicles[key].model_dump(mode="json") for key in sorted(vehicles)]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def all_vehicles() -> dict[str, VehicleProfile]:
    merged = dict(VEHICLES)
    merged.update(load_custom_vehicles())
    return merged


def list_custom_vehicles() -> list[VehicleProfile]:
    custom = load_custom_vehicles()
    return [custom[key] for key in sorted(custom)]


def _validate_custom_vehicle_id(vehicle_id: str) -> str:
    vid = vehicle_id.strip()
    if not VEHICLE_ID_RE.match(vid):
        raise ValueError("vehicle id must match ^[a-z][a-z0-9_-]{1,47}$")
    return vid


def create_custom_vehicle(vehicle: VehicleProfile) -> tuple[VehicleProfile, Path]:
    vid = _validate_custom_vehicle_id(vehicle.id)
    if vid in VEHICLES:
        raise ValueError("vehicle id collides with built-in profile")

    custom = load_custom_vehicles()
    if vid in custom:
        raise ValueError("vehicle id already exists")

    created = vehicle.model_copy(update={"id": vid})
    custom[vid] = created
    path = save_custom_vehicles(custom)
    return created, path


def update_custom_vehicle(vehicle_id: str, vehicle: VehicleProfile) -> tuple[VehicleProfile, Path]:
    vid = _validate_custom_vehicle_id(vehicle_id)
    if vid in VEHICLES:
        raise ValueError("built-in vehicles cannot be updated")
    if vehicle.id != vid:
        raise ValueError("body id must match path id")

    custom = load_custom_vehicles()
    if vid not in custom:
        raise KeyError("custom vehicle not found")

    updated = vehicle.model_copy(update={"id": vid})
    custom[vid] = updated
    path = save_custom_vehicles(custom)
    return updated, path


def delete_custom_vehicle(vehicle_id: str) -> tuple[str, Path]:
    vid = _validate_custom_vehicle_id(vehicle_id)
    if vid in VEHICLES:
        raise ValueError("built-in vehicles cannot be deleted")

    custom = load_custom_vehicles()
    if vid not in custom:
        raise KeyError("custom vehicle not found")

    custom.pop(vid, None)
    path = save_custom_vehicles(custom)
    return vid, path


def get_vehicle(vehicle_type: str) -> VehicleProfile:
    merged = all_vehicles()
    return merged.get(vehicle_type) or merged[DEFAULT_VEHICLE_ID]
