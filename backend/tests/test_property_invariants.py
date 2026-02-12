from __future__ import annotations

import json
import random

from app.models import GeoJSONLineString, RouteMetrics, RouteOption
from app.objectives_selection import normalise_weights, pick_best_by_weighted_sum
from app.pareto import dominates, pareto_filter
from app.provenance_store import provenance_event, write_provenance
from app.route_cache import clear_route_cache, get_cached_routes, set_cached_routes
from app.settings import settings
from app.signatures import sign_payload, verify_payload_signature


def _make_option(option_id: str, *, duration_s: float, money: float, co2: float) -> RouteOption:
    return RouteOption(
        id=option_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(-1.0, 52.0), (-0.1, 51.5)]),
        metrics=RouteMetrics(
            distance_km=10.0,
            duration_s=duration_s,
            monetary_cost=money,
            emissions_kg=co2,
            avg_speed_kmh=40.0,
        ),
    )


def test_pareto_randomized_invariants() -> None:
    rng = random.Random(20260212)

    for _ in range(30):
        items = []
        for idx in range(25):
            vec = (
                round(rng.uniform(1.0, 100.0) + (idx * 1e-6), 6),
                round(rng.uniform(1.0, 100.0) + (idx * 1e-6), 6),
                round(rng.uniform(1.0, 100.0) + (idx * 1e-6), 6),
            )
            assert not dominates(vec, vec)
            items.append({"id": idx, "vec": vec})

        kept = pareto_filter(items, key=lambda item: item["vec"])
        kept_ids = {item["id"] for item in kept}

        for i, item_i in enumerate(kept):
            for j, item_j in enumerate(kept):
                if i == j:
                    continue
                assert not dominates(item_i["vec"], item_j["vec"])

        for item in items:
            if item["id"] in kept_ids:
                continue
            assert any(dominates(k["vec"], item["vec"]) for k in kept)


def test_weighted_selection_randomized_invariants() -> None:
    rng = random.Random(42)

    for _ in range(40):
        options = [
            _make_option(
                f"route_{idx}",
                duration_s=rng.uniform(1000.0, 6000.0),
                money=rng.uniform(100.0, 800.0),
                co2=rng.uniform(20.0, 200.0),
            )
            for idx in range(6)
        ]
        w_time = rng.uniform(0.0, 10.0)
        w_money = rng.uniform(0.0, 10.0)
        w_co2 = rng.uniform(0.0, 10.0)

        selected = pick_best_by_weighted_sum(
            options,
            w_time=w_time,
            w_money=w_money,
            w_co2=w_co2,
        )
        scaled = pick_best_by_weighted_sum(
            options,
            w_time=w_time * 7.0,
            w_money=w_money * 7.0,
            w_co2=w_co2 * 7.0,
        )
        assert selected.id == scaled.id

        wt, wm, we = normalise_weights(w_time, w_money, w_co2)
        assert abs((wt + wm + we) - 1.0) < 1e-9


def test_signature_and_cache_invariants_under_varied_inputs() -> None:
    rng = random.Random(99)
    clear_route_cache()

    for idx in range(12):
        payload = {
            "index": idx,
            "value": rng.randint(1, 1000),
            "nested": {"flag": idx % 2 == 0},
        }
        secret = f"secret-{idx}"
        signature = sign_payload(payload, secret=secret)
        valid, expected = verify_payload_signature(payload, signature, secret=secret)
        assert valid is True
        assert expected == signature

        tampered = dict(payload)
        tampered["value"] = int(payload["value"]) + 1
        valid_tampered, _ = verify_payload_signature(tampered, signature, secret=secret)
        assert valid_tampered is False

        key = f"cache_key_{idx}"
        value = (
            [{"distance": 1000.0 + idx, "duration": 100.0 + idx}],
            [f"warn_{idx}"],
            3,
        )
        set_cached_routes(key, value)
        assert get_cached_routes(key) == value

    assert get_cached_routes("missing_key") is None


def test_provenance_event_order_is_preserved(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path / "out"))

    run_id = "11111111-1111-1111-1111-111111111111"
    events = [
        provenance_event(run_id, "input_received", pair_count=2),
        provenance_event(run_id, "candidates_fetched", candidate_count=4),
        provenance_event(run_id, "options_built", option_count=4),
        provenance_event(run_id, "pareto_selected", pareto_count=2),
        provenance_event(run_id, "artifacts_written", artifact_count=5),
    ]
    path = write_provenance(run_id, events)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.exists()
    assert payload["event_count"] == 5
    ordered_events = [event["event"] for event in payload["events"]]
    assert ordered_events == [
        "input_received",
        "candidates_fetched",
        "options_built",
        "pareto_selected",
        "artifacts_written",
    ]
