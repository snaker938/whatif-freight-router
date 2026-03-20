from __future__ import annotations

from app.evidence_certification import (
    active_evidence_families,
    compute_certificate,
    compute_fragility_maps,
    dependency_tensor,
    rank_value_of_refresh,
    sample_world_manifest,
)


def _route(
    route_id: str,
    *,
    objective: tuple[float, float, float],
    evidence_tensor: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    return {
        "route_id": route_id,
        "objective_vector": objective,
        "evidence_tensor": evidence_tensor or {},
        "evidence_provenance": {
            "active_families": list((evidence_tensor or {}).keys()),
            "dependency_weights": {
                family: {"time": 1.0, "money": 1.0, "co2": 1.0}
                for family in (evidence_tensor or {})
            },
        },
    }


def test_sampled_world_manifest_is_seed_replayable() -> None:
    manifest_a = sample_world_manifest(
        active_families=["scenario", "toll", "weather"],
        seed=17,
        world_count=5,
    )
    manifest_b = sample_world_manifest(
        active_families=["scenario", "toll", "weather"],
        seed=17,
        world_count=5,
    )

    assert manifest_a == manifest_b
    assert manifest_a["active_families"] == ["scenario", "toll", "weather"]
    assert all(set(world["states"]) <= {"scenario", "toll", "weather"} for world in manifest_a["worlds"])


def test_dependency_tensor_is_bounded_and_normalised() -> None:
    route = _route(
        "route_a",
        objective=(10.0, 10.0, 10.0),
        evidence_tensor={"scenario": {"time": 1.0, "money": 1.0, "co2": 1.0}},
    )
    tensor = dependency_tensor(route, active_families=["scenario"])

    assert set(tensor) == {"time", "money", "co2"}
    assert tensor["time"]["scenario"] == 1.0
    assert tensor["money"]["scenario"] == 1.0
    assert tensor["co2"]["scenario"] == 1.0
    assert active_evidence_families([route]) == ["scenario"]


def test_certificate_and_fragility_outputs_are_hand_checkable() -> None:
    routes = [
        _route(
            "route_a",
            objective=(10.0, 10.0, 10.0),
            evidence_tensor={
                "scenario": {"time": 1.0, "money": 1.0, "co2": 1.0},
                "weather": {"time": 0.0, "money": 0.0, "co2": 0.0},
            },
        ),
        _route(
            "route_b",
            objective=(10.05, 10.05, 10.05),
            evidence_tensor={},
        ),
    ]
    worlds = [
        {"world_id": "w0", "states": {"scenario": "nominal"}},
        {"world_id": "w1", "states": {"scenario": "severely_stale"}},
        {"world_id": "w2", "states": {"scenario": "refreshed"}},
    ]

    certificate = compute_certificate(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        threshold=0.60,
        active_families=["scenario", "weather"],
    )
    fragility = compute_fragility_maps(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather"],
        selected_route_id="route_a",
    )
    vor = rank_value_of_refresh(
        routes,
        worlds=worlds,
        selector_weights=(1.0, 1.0, 1.0),
        active_families=["scenario", "weather"],
        selected_route_id="route_a",
    )

    assert certificate.winner_id == "route_a"
    assert certificate.certified is True
    assert certificate.certificate["route_a"] == 2 / 3
    assert certificate.certificate["route_b"] == 1 / 3
    assert fragility.route_fragility_map["route_a"]["scenario"] > 0.0
    assert fragility.route_fragility_map["route_a"]["weather"] == 0.0
    assert fragility.competitor_fragility_breakdown["route_a"]["route_b"]["scenario"] == 1
    assert fragility.value_of_refresh["top_refresh_family"] == "scenario"
    assert vor["top_refresh_family"] == "scenario"
    assert vor["ranking"][0]["family"] == "scenario"
