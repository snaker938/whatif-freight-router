from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app, osrm_client
from app.models import ScenarioCompareDelta, ScenarioCompareResult
from app.scenario import ScenarioMode
from app.settings import settings


class FakeOSRM:
    async def fetch_routes(self, **kwargs: Any) -> list[dict[str, Any]]:
        origin_lat = float(kwargs["origin_lat"])
        dest_lat = float(kwargs["dest_lat"])
        span = abs(origin_lat - dest_lat) + 0.05
        duration_s = 3_100.0 + (span * 120.0)
        distance_m = 70_000.0 + (span * 2_000.0)
        return [
            {
                "distance": distance_m,
                "duration": duration_s,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-1.89, 52.48], [-0.1276, 51.5072]],
                },
                "legs": [
                    {
                        "annotation": {
                            "distance": [distance_m / 2.0, distance_m / 2.0],
                            "duration": [duration_s / 2.0, duration_s / 2.0],
                        }
                    }
                ],
            }
        ]


def test_experiment_lifecycle_and_compare_replay(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))

    async def _fake_run_scenario_compare(*, req, osrm):
        _ = req, osrm
        results = [
            ScenarioCompareResult(scenario_mode=ScenarioMode.NO_SHARING, selected=None, candidates=[]),
            ScenarioCompareResult(scenario_mode=ScenarioMode.PARTIAL_SHARING, selected=None, candidates=[]),
            ScenarioCompareResult(scenario_mode=ScenarioMode.FULL_SHARING, selected=None, candidates=[]),
        ]
        deltas = {
            "partial_sharing": ScenarioCompareDelta(duration_s_delta=-120.0, monetary_cost_delta=-8.0, emissions_kg_delta=-5.0),
            "full_sharing": ScenarioCompareDelta(duration_s_delta=-210.0, monetary_cost_delta=-14.0, emissions_kg_delta=-9.0),
        }
        return results, deltas

    monkeypatch.setattr(main_module, "_run_scenario_compare", _fake_run_scenario_compare)
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()

    try:
        with TestClient(app) as client:
            payload = {
                "name": "Morning baseline",
                "description": "Primary scenario compare setup",
                "request": {
                    "origin": {"lat": 52.4862, "lon": -1.8904},
                    "destination": {"lat": 51.5072, "lon": -0.1276},
                    "vehicle_type": "rigid_hgv",
                    "weights": {"time": 1, "money": 1, "co2": 1},
                    "max_alternatives": 4,
                    "pareto_method": "epsilon_constraint",
                    "epsilon": {"duration_s": 8_000, "monetary_cost": 8_000, "emissions_kg": 8_000},
                    "terrain_profile": "rolling",
                },
            }

            create_resp = client.post("/experiments", json=payload)
            assert create_resp.status_code == 200
            created = create_resp.json()
            experiment_id = created["id"]
            assert created["name"] == "Morning baseline"
            assert created["request"]["terrain_profile"] == "rolling"

            list_resp = client.get("/experiments")
            assert list_resp.status_code == 200
            listed = list_resp.json()["experiments"]
            assert any(item["id"] == experiment_id for item in listed)

            get_resp = client.get(f"/experiments/{experiment_id}")
            assert get_resp.status_code == 200
            assert get_resp.json()["description"] == "Primary scenario compare setup"

            update_payload = dict(payload)
            update_payload["name"] = "Morning baseline v2"
            update_payload["description"] = "Updated"
            update_resp = client.put(f"/experiments/{experiment_id}", json=update_payload)
            assert update_resp.status_code == 200
            assert update_resp.json()["name"] == "Morning baseline v2"

            replay_resp = client.post(
                f"/experiments/{experiment_id}/compare",
                json={"overrides": {"max_alternatives": 3}},
            )
            assert replay_resp.status_code == 200
            replay = replay_resp.json()
            run_id = replay["run_id"]
            assert len(replay["results"]) == 3
            assert replay["scenario_manifest_endpoint"] == f"/runs/{run_id}/scenario-manifest"

            manifest_resp = client.get(f"/runs/{run_id}/scenario-manifest")
            assert manifest_resp.status_code == 200
            manifest = manifest_resp.json()
            assert manifest["source"]["experiment_id"] == experiment_id
            assert manifest["source"]["experiment_name"] == "Morning baseline v2"

            delete_resp = client.delete(f"/experiments/{experiment_id}")
            assert delete_resp.status_code == 200
            assert delete_resp.json()["deleted"] is True

            missing_resp = client.get(f"/experiments/{experiment_id}")
            assert missing_resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_experiment_catalog_filters_and_sorting(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))
    app.dependency_overrides[osrm_client] = lambda: FakeOSRM()

    try:
        with TestClient(app) as client:
            base_request = {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "weights": {"time": 1, "money": 1, "co2": 1},
                "max_alternatives": 3,
            }
            experiments = [
                ("Zulu run", "rigid_hgv", "no_sharing"),
                ("Alpha run", "van", "partial_sharing"),
                ("Echo run", "artic_hgv", "full_sharing"),
            ]
            for name, vehicle_type, scenario_mode in experiments:
                payload = {
                    "name": name,
                    "description": f"{name} description",
                    "request": {
                        **base_request,
                        "vehicle_type": vehicle_type,
                        "scenario_mode": scenario_mode,
                    },
                }
                create_resp = client.post("/experiments", json=payload)
                assert create_resp.status_code == 200

            all_resp = client.get("/experiments")
            assert all_resp.status_code == 200
            assert len(all_resp.json()["experiments"]) == 3

            search_resp = client.get("/experiments?q=alpha")
            assert search_resp.status_code == 200
            search_items = search_resp.json()["experiments"]
            assert len(search_items) == 1
            assert search_items[0]["name"] == "Alpha run"

            vehicle_resp = client.get("/experiments?vehicle_type=van")
            assert vehicle_resp.status_code == 200
            vehicle_items = vehicle_resp.json()["experiments"]
            assert len(vehicle_items) == 1
            assert vehicle_items[0]["request"]["vehicle_type"] == "van"

            scenario_resp = client.get("/experiments?scenario_mode=full_sharing")
            assert scenario_resp.status_code == 200
            scenario_items = scenario_resp.json()["experiments"]
            assert len(scenario_items) == 1
            assert scenario_items[0]["request"]["scenario_mode"] == "full_sharing"

            sort_resp = client.get("/experiments?sort=name_asc")
            assert sort_resp.status_code == 200
            sorted_names = [item["name"] for item in sort_resp.json()["experiments"]]
            assert sorted_names == ["Alpha run", "Echo run", "Zulu run"]

            bad_sort_resp = client.get("/experiments?sort=bad_sort")
            assert bad_sort_resp.status_code == 400
    finally:
        app.dependency_overrides.clear()
