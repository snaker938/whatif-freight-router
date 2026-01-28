from __future__ import annotations

from pydantic import BaseModel, Field


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


def get_vehicle(vehicle_type: str) -> VehicleProfile:
    return VEHICLES.get(vehicle_type) or VEHICLES[DEFAULT_VEHICLE_ID]
