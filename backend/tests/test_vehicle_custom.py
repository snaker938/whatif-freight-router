from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.settings import settings
from app.vehicles import load_custom_vehicles


def _vehicle_payload(vehicle_id: str = "fleet_demo") -> dict[str, object]:
    return {
        "id": vehicle_id,
        "label": "Fleet demo profile",
        "mass_tonnes": 9.5,
        "emission_factor_kg_per_tkm": 0.11,
        "cost_per_km": 0.58,
        "cost_per_hour": 26.0,
        "idle_emissions_kg_per_hour": 2.9,
    }


def test_custom_vehicle_crud_and_persistence(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))

    with TestClient(app) as client:
        create_resp = client.post("/vehicles/custom", json=_vehicle_payload("fleet_demo"))
        assert create_resp.status_code == 200
        assert create_resp.json()["vehicle"]["id"] == "fleet_demo"

        list_resp = client.get("/vehicles/custom")
        assert list_resp.status_code == 200
        ids = [v["id"] for v in list_resp.json()["vehicles"]]
        assert "fleet_demo" in ids

        merged_resp = client.get("/vehicles")
        assert merged_resp.status_code == 200
        merged_ids = [v["id"] for v in merged_resp.json()["vehicles"]]
        assert "fleet_demo" in merged_ids
        assert "rigid_hgv" in merged_ids

        updated = _vehicle_payload("fleet_demo")
        updated["cost_per_km"] = 0.72
        update_resp = client.put("/vehicles/custom/fleet_demo", json=updated)
        assert update_resp.status_code == 200
        assert update_resp.json()["vehicle"]["cost_per_km"] == 0.72

        delete_resp = client.delete("/vehicles/custom/fleet_demo")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted"] is True

    custom_after = load_custom_vehicles()
    assert "fleet_demo" not in custom_after


def test_custom_vehicle_collision_and_validation(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))

    with TestClient(app) as client:
        collision_resp = client.post("/vehicles/custom", json=_vehicle_payload("rigid_hgv"))
        assert collision_resp.status_code == 409

        bad_id_payload = _vehicle_payload("BAD-ID")
        bad_id_resp = client.post("/vehicles/custom", json=bad_id_payload)
        assert bad_id_resp.status_code == 422

        create_resp = client.post("/vehicles/custom", json=_vehicle_payload("custom_a"))
        assert create_resp.status_code == 200

        duplicate_resp = client.post("/vehicles/custom", json=_vehicle_payload("custom_a"))
        assert duplicate_resp.status_code == 409
