from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .model_data_errors import ModelDataError
from .settings import settings


VehicleClass = Literal["van", "rigid_hgv", "artic_hgv", "ev"]
Powertrain = Literal["ice", "ev"]
DEFAULT_VEHICLE_ID = "rigid_hgv"
VEHICLE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,47}$")


def _validate_vehicle_id_format(vehicle_id: str) -> str:
    vid = str(vehicle_id).strip()
    if not VEHICLE_ID_RE.match(vid):
        raise ValueError("vehicle id must match ^[a-z][a-z0-9_-]{1,47}$")
    return vid


class VehicleTerrainParamsModel(BaseModel):
    mass_kg: float = Field(..., gt=300.0, le=100_000.0)
    c_rr: float = Field(..., gt=0.001, le=0.05)
    drag_area_m2: float = Field(..., gt=0.5, le=25.0)
    drivetrain_efficiency: float = Field(..., gt=0.5, le=1.0)
    regen_efficiency: float = Field(..., ge=0.0, le=0.8)


def _default_rigid_terrain_params() -> VehicleTerrainParamsModel:
    return VehicleTerrainParamsModel(
        mass_kg=26_000.0,
        c_rr=0.0082,
        drag_area_m2=7.3,
        drivetrain_efficiency=0.88,
        regen_efficiency=0.14,
    )


class VehicleProfile(BaseModel):
    id: str
    label: str

    mass_tonnes: float = Field(..., gt=0)
    emission_factor_kg_per_tkm: float = Field(..., gt=0)
    cost_per_km: float = Field(..., ge=0)
    cost_per_hour: float = Field(..., ge=0)
    idle_emissions_kg_per_hour: float = Field(default=0.0, ge=0)

    powertrain: Powertrain = "ice"
    ev_kwh_per_km: float | None = Field(default=None, ge=0.0)
    grid_co2_kg_per_kwh: float | None = Field(default=None, ge=0.0)

    schema_version: int = 2
    vehicle_class: VehicleClass = "rigid_hgv"
    toll_vehicle_class: str = "rigid_hgv"
    toll_axle_class: str = "3to4"
    fuel_surface_class: VehicleClass = "rigid_hgv"
    risk_bucket: str = "rigid_hgv"
    stochastic_bucket: str = "rigid_hgv"
    terrain_params: VehicleTerrainParamsModel = Field(default_factory=_default_rigid_terrain_params)
    aliases: list[str] = Field(default_factory=list)
    profile_source: str = "custom"
    profile_as_of_utc: str | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_vehicle_id_format(value)

    @field_validator("aliases", mode="before")
    @classmethod
    def _normalize_aliases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("aliases must be a list")
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            alias = str(item).strip().lower()
            if not alias:
                continue
            if not VEHICLE_ID_RE.match(alias):
                raise ValueError(f"invalid alias '{alias}'")
            if alias in seen:
                continue
            out.append(alias)
            seen.add(alias)
        return out

    @model_validator(mode="after")
    def _validate_v2(self) -> "VehicleProfile":
        if self.schema_version < 2:
            raise ValueError("schema_version must be >=2")
        if self.id in self.aliases:
            raise ValueError("aliases must not include profile id")
        if self.powertrain == "ev" or self.vehicle_class == "ev" or self.fuel_surface_class == "ev":
            if self.ev_kwh_per_km is None or self.ev_kwh_per_km <= 0:
                raise ValueError("EV profiles require ev_kwh_per_km > 0")
            if self.grid_co2_kg_per_kwh is None or self.grid_co2_kg_per_kwh < 0:
                raise ValueError("EV profiles require grid_co2_kg_per_kwh >= 0")
        if not self.toll_vehicle_class.strip():
            raise ValueError("toll_vehicle_class is required")
        if not self.toll_axle_class.strip():
            raise ValueError("toll_axle_class is required")
        if not self.risk_bucket.strip():
            raise ValueError("risk_bucket is required")
        if not self.stochastic_bucket.strip():
            raise ValueError("stochastic_bucket is required")
        return self



def _strict_runtime_required() -> bool:
    return bool(settings.strict_live_data_required)


def _assets_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "uk"


def _model_asset_root() -> Path:
    return Path(settings.model_asset_dir)


def _builtin_vehicle_asset_path_candidates() -> list[Path]:
    return [
        _model_asset_root() / "vehicle_profiles_uk.json",
        _assets_root() / "vehicle_profiles_uk.json",
    ]


def _legacy_custom_vehicle_path() -> Path:
    p = Path(settings.out_dir) / "config" / "vehicles.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _custom_vehicle_path() -> Path:
    p = Path(settings.out_dir) / "config" / "vehicles_v2.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _terrain_defaults(vehicle_class: VehicleClass) -> dict[str, float]:
    if vehicle_class == "artic_hgv":
        return {
            "mass_kg": 38_000.0,
            "c_rr": 0.0075,
            "drag_area_m2": 8.2,
            "drivetrain_efficiency": 0.86,
            "regen_efficiency": 0.12,
        }
    if vehicle_class == "van":
        return {
            "mass_kg": 3_500.0,
            "c_rr": 0.0095,
            "drag_area_m2": 3.9,
            "drivetrain_efficiency": 0.89,
            "regen_efficiency": 0.18,
        }
    if vehicle_class == "ev":
        return {
            "mass_kg": 18_000.0,
            "c_rr": 0.0085,
            "drag_area_m2": 6.7,
            "drivetrain_efficiency": 0.91,
            "regen_efficiency": 0.40,
        }
    return {
        "mass_kg": 26_000.0,
        "c_rr": 0.0082,
        "drag_area_m2": 7.3,
        "drivetrain_efficiency": 0.88,
        "regen_efficiency": 0.14,
    }


def _infer_vehicle_class(raw: dict[str, Any]) -> VehicleClass:
    explicit = str(raw.get("vehicle_class", "")).strip().lower()
    if explicit in {"van", "rigid_hgv", "artic_hgv", "ev"}:
        return explicit  # type: ignore[return-value]
    powertrain = str(raw.get("powertrain", "ice")).strip().lower()
    if powertrain == "ev":
        return "ev"
    mass = float(raw.get("mass_tonnes", 0.0) or 0.0)
    if mass <= 3.5:
        return "van"
    if mass <= 18.0:
        return "rigid_hgv"
    return "artic_hgv"


def _default_toll_vehicle_class(klass: VehicleClass) -> str:
    if klass in {"van", "rigid_hgv", "artic_hgv"}:
        return klass
    return "rigid_hgv"


def _default_toll_axle_class(klass: VehicleClass) -> str:
    if klass == "van":
        return "2"
    if klass == "artic_hgv":
        return "5plus"
    return "3to4"


def _normalize_profile_payload(
    raw: dict[str, Any],
    *,
    source: str,
    as_of_utc: str | None,
) -> dict[str, Any]:
    klass = _infer_vehicle_class(raw)
    profile_source = str(raw.get("profile_source", "")).strip() or source
    payload = dict(raw)
    payload["id"] = str(raw.get("id", "")).strip().lower()
    payload["label"] = str(raw.get("label", payload["id"])).strip() or payload["id"]
    payload["powertrain"] = str(raw.get("powertrain", "ice")).strip().lower() or "ice"
    payload["schema_version"] = int(raw.get("schema_version", 2) or 2)
    payload["vehicle_class"] = str(raw.get("vehicle_class", klass)).strip().lower() or klass
    payload["toll_vehicle_class"] = (
        str(raw.get("toll_vehicle_class", _default_toll_vehicle_class(klass))).strip().lower()
        or _default_toll_vehicle_class(klass)
    )
    payload["toll_axle_class"] = (
        str(raw.get("toll_axle_class", _default_toll_axle_class(klass))).strip().lower()
        or _default_toll_axle_class(klass)
    )
    payload["fuel_surface_class"] = str(raw.get("fuel_surface_class", klass)).strip().lower() or klass
    payload["risk_bucket"] = str(raw.get("risk_bucket", klass)).strip().lower() or klass
    payload["stochastic_bucket"] = str(raw.get("stochastic_bucket", klass)).strip().lower() or klass
    if not isinstance(raw.get("terrain_params"), dict):
        payload["terrain_params"] = _terrain_defaults(klass)
    payload["aliases"] = raw.get("aliases", [])
    payload["profile_source"] = profile_source
    payload["profile_as_of_utc"] = str(raw.get("profile_as_of_utc", as_of_utc or "")).strip() or None
    return payload


def _validate_signature(payload: dict[str, Any], *, require_signature: bool) -> str | None:
    signature = str(payload.get("signature", "")).strip() or None
    if not require_signature:
        return signature
    if not signature:
        raise ModelDataError(
            reason_code="vehicle_profile_invalid",
            message="Vehicle profile asset signature is required in strict runtime.",
        )
    payload_no_sig = {k: v for k, v in payload.items() if k != "signature"}
    expected = hashlib.sha256(
        json.dumps(payload_no_sig, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if expected.lower() != signature.lower():
        raise ModelDataError(
            reason_code="vehicle_profile_invalid",
            message="Vehicle profile asset signature validation failed.",
            details={"expected_signature": expected},
        )
    return signature


def _bootstrap_builtin_profiles() -> dict[str, VehicleProfile]:
    rows = [
        {
            "id": "van",
            "label": "Delivery van",
            "mass_tonnes": 2.0,
            "emission_factor_kg_per_tkm": 0.18,
            "cost_per_km": 0.35,
            "cost_per_hour": 18.0,
            "idle_emissions_kg_per_hour": 1.2,
            "powertrain": "ice",
            "aliases": ["lgv", "delivery_van"],
        },
        {
            "id": "rigid_hgv",
            "label": "Rigid HGV",
            "mass_tonnes": 12.0,
            "emission_factor_kg_per_tkm": 0.10,
            "cost_per_km": 0.65,
            "cost_per_hour": 28.0,
            "idle_emissions_kg_per_hour": 3.5,
            "powertrain": "ice",
            "aliases": ["rigid", "hgv_rigid"],
        },
        {
            "id": "artic_hgv",
            "label": "Articulated HGV",
            "mass_tonnes": 20.0,
            "emission_factor_kg_per_tkm": 0.08,
            "cost_per_km": 0.75,
            "cost_per_hour": 32.0,
            "idle_emissions_kg_per_hour": 4.5,
            "powertrain": "ice",
            "aliases": ["artic", "tractor_trailer"],
        },
        {
            "id": "ev_hgv",
            "label": "EV HGV",
            "mass_tonnes": 18.0,
            "emission_factor_kg_per_tkm": 0.04,
            "cost_per_km": 0.55,
            "cost_per_hour": 29.0,
            "idle_emissions_kg_per_hour": 0.2,
            "powertrain": "ev",
            "ev_kwh_per_km": 1.2,
            "grid_co2_kg_per_kwh": 0.20,
            "vehicle_class": "ev",
            "fuel_surface_class": "ev",
            "aliases": ["ev", "electric_hgv"],
        },
    ]
    out: dict[str, VehicleProfile] = {}
    for row in rows:
        normalized = _normalize_profile_payload(
            row,
            source="bootstrap_defaults",
            as_of_utc=None,
        )
        out[normalized["id"]] = VehicleProfile.model_validate(normalized)
    return out


# Backward-compatible in-code defaults used by tests/bootstrap only.
VEHICLES: dict[str, VehicleProfile] = _bootstrap_builtin_profiles()


@lru_cache(maxsize=1)
def load_builtin_vehicles() -> dict[str, VehicleProfile]:
    for path in _builtin_vehicle_asset_path_candidates():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelDataError(
                reason_code="vehicle_profile_invalid",
                message=f"Vehicle profile asset '{path}' is not valid JSON.",
            ) from exc
        if not isinstance(payload, dict):
            raise ModelDataError(
                reason_code="vehicle_profile_invalid",
                message=f"Vehicle profile asset '{path}' has invalid schema.",
            )
        _validate_signature(
            payload,
            require_signature=bool(_strict_runtime_required()),
        )
        raw_profiles = payload.get("profiles")
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ModelDataError(
                reason_code="vehicle_profile_invalid",
                message=f"Vehicle profile asset '{path}' has no profiles.",
            )
        as_of_utc = str(payload.get("as_of_utc", payload.get("as_of", ""))).strip() or None
        out: dict[str, VehicleProfile] = {}
        for raw in raw_profiles:
            if not isinstance(raw, dict):
                continue
            normalized = _normalize_profile_payload(raw, source=str(path), as_of_utc=as_of_utc)
            try:
                profile = VehicleProfile.model_validate(normalized)
            except Exception as exc:
                raise ModelDataError(
                    reason_code="vehicle_profile_invalid",
                    message=f"Invalid built-in vehicle profile '{normalized.get('id', 'unknown')}'.",
                ) from exc
            out[profile.id] = profile
        if out:
            return out
    if _strict_runtime_required():
        raise ModelDataError(
            reason_code="vehicle_profile_unavailable",
            message="Built-in vehicle profile asset is required in strict runtime.",
        )
    return dict(VEHICLES)


def _load_custom_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def load_custom_vehicles() -> dict[str, VehicleProfile]:
    current_path = _custom_vehicle_path()
    rows = _load_custom_rows(current_path)
    source_path = current_path
    if not rows:
        legacy_rows = _load_custom_rows(_legacy_custom_vehicle_path())
        if legacy_rows:
            rows = legacy_rows
            source_path = _legacy_custom_vehicle_path()
    if not rows:
        return {}

    out: dict[str, VehicleProfile] = {}
    migrated_any = False
    for raw in rows:
        normalized = _normalize_profile_payload(raw, source=str(source_path), as_of_utc=None)
        try:
            profile = VehicleProfile.model_validate(normalized)
        except Exception:
            continue
        if int(raw.get("schema_version", 0) or 0) < 2:
            migrated_any = True
        out[profile.id] = profile
    if source_path != current_path and out:
        save_custom_vehicles(out)
    elif migrated_any and out:
        save_custom_vehicles(out)
    return out


def save_custom_vehicles(vehicles: dict[str, VehicleProfile]) -> Path:
    path = _custom_vehicle_path()
    payload = [vehicles[key].model_dump(mode="json") for key in sorted(vehicles)]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _alias_index(vehicles: dict[str, VehicleProfile]) -> dict[str, str]:
    index: dict[str, str] = {}
    for vehicle_id, profile in vehicles.items():
        key = vehicle_id.strip().lower()
        if key:
            index.setdefault(key, vehicle_id)
        for alias in profile.aliases:
            index.setdefault(alias.strip().lower(), vehicle_id)
    return index


def all_vehicles() -> dict[str, VehicleProfile]:
    merged = dict(load_builtin_vehicles())
    merged.update(load_custom_vehicles())
    return merged


def list_custom_vehicles() -> list[VehicleProfile]:
    custom = load_custom_vehicles()
    return [custom[key] for key in sorted(custom)]


def _validate_custom_vehicle_id(vehicle_id: str) -> str:
    return _validate_vehicle_id_format(vehicle_id)


def create_custom_vehicle(vehicle: VehicleProfile) -> tuple[VehicleProfile, Path]:
    vid = _validate_custom_vehicle_id(vehicle.id)
    builtins = load_builtin_vehicles()
    if vid in builtins:
        raise ValueError("vehicle id collides with built-in profile")
    alias_map = _alias_index(all_vehicles())
    if vid in alias_map:
        raise ValueError("vehicle id collides with existing vehicle alias")
    custom = load_custom_vehicles()
    if vid in custom:
        raise ValueError("vehicle id already exists")
    created = vehicle.model_copy(
        update={
            "id": vid,
            "schema_version": 2,
            "profile_source": "custom",
        }
    )
    custom[vid] = created
    path = save_custom_vehicles(custom)
    return created, path


def update_custom_vehicle(vehicle_id: str, vehicle: VehicleProfile) -> tuple[VehicleProfile, Path]:
    vid = _validate_custom_vehicle_id(vehicle_id)
    builtins = load_builtin_vehicles()
    if vid in builtins:
        raise ValueError("built-in vehicles cannot be updated")
    if vehicle.id.strip().lower() != vid:
        raise ValueError("body id must match path id")
    custom = load_custom_vehicles()
    if vid not in custom:
        raise KeyError("custom vehicle not found")
    alias_map = _alias_index({**builtins, **custom})
    for alias in vehicle.aliases:
        owner = alias_map.get(alias.strip().lower())
        if owner is not None and owner != vid:
            raise ValueError(f"alias '{alias}' collides with existing vehicle")
    updated = vehicle.model_copy(
        update={
            "id": vid,
            "schema_version": 2,
            "profile_source": "custom",
        }
    )
    custom[vid] = updated
    path = save_custom_vehicles(custom)
    return updated, path


def delete_custom_vehicle(vehicle_id: str) -> tuple[str, Path]:
    vid = _validate_custom_vehicle_id(vehicle_id)
    if vid in load_builtin_vehicles():
        raise ValueError("built-in vehicles cannot be deleted")
    custom = load_custom_vehicles()
    if vid not in custom:
        raise KeyError("custom vehicle not found")
    custom.pop(vid, None)
    path = save_custom_vehicles(custom)
    return vid, path


def resolve_vehicle_profile(vehicle_type: str) -> VehicleProfile:
    key = (vehicle_type or "").strip().lower()
    if not key:
        raise ModelDataError(
            reason_code="vehicle_profile_unavailable",
            message="Vehicle type must be provided in strict runtime.",
        )
    merged = all_vehicles()
    if key in merged:
        return merged[key]
    alias_target = _alias_index(merged).get(key)
    if alias_target and alias_target in merged:
        return merged[alias_target]
    raise ModelDataError(
        reason_code="vehicle_profile_unavailable",
        message=f"Unknown vehicle profile '{vehicle_type}'.",
        details={"vehicle_type": vehicle_type},
    )


def get_vehicle(vehicle_type: str) -> VehicleProfile:
    # Compatibility wrapper for existing call sites.
    return resolve_vehicle_profile(vehicle_type)
