from __future__ import annotations

import pytest

from app.models import (
    GeoJSONLineString,
    LatLng,
    RouteMetrics,
    RouteOption,
    Waypoint,
)
from app.multileg_engine import compose_multileg_route_options


def _mk_option(
    option_id: str,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    distance_km: float,
    duration_s: float,
    money: float,
    co2: float,
    toll_confidence: float,
) -> RouteOption:
    return RouteOption(
        id=option_id,
        geometry=GeoJSONLineString(
            type="LineString",
            coordinates=[
                [start[1], start[0]],
                [end[1], end[0]],
            ],
        ),
        metrics=RouteMetrics(
            distance_km=distance_km,
            duration_s=duration_s,
            monetary_cost=money,
            emissions_kg=co2,
            avg_speed_kmh=distance_km / max(1e-6, duration_s / 3600.0),
        ),
        uncertainty={
            "mean_duration_s": duration_s,
            "std_duration_s": 0.0,
            "q50_duration_s": duration_s,
            "q90_duration_s": duration_s,
            "q95_duration_s": duration_s,
            "cvar95_duration_s": duration_s,
            "mean_monetary_cost": money,
            "std_monetary_cost": 0.0,
            "q50_monetary_cost": money,
            "q90_monetary_cost": money,
            "q95_monetary_cost": money,
            "cvar95_monetary_cost": money,
            "mean_emissions_kg": co2,
            "std_emissions_kg": 0.0,
            "q50_emissions_kg": co2,
            "q90_emissions_kg": co2,
            "q95_emissions_kg": co2,
            "cvar95_emissions_kg": co2,
            "robust_score": duration_s,
        },
        uncertainty_samples_meta={
            "sample_count": 16,
            "seed": f"seed-{option_id}",
        },
        toll_confidence=toll_confidence,
    )


@pytest.mark.anyio
async def test_multileg_composition_with_waypoint_chain() -> None:
    origin = LatLng(lat=54.9783, lon=-1.6178)
    waypoint = Waypoint(lat=53.4808, lon=-2.2426, label="Manchester")
    destination = LatLng(lat=51.5072, lon=-0.1276)

    async def leg_solver(
        leg_index: int,
        leg_origin: LatLng,
        leg_destination: LatLng,
    ) -> tuple[list[RouteOption], list[str], int]:
        if leg_index == 0:
            return (
                [
                    _mk_option(
                        "leg0_fast",
                        start=(leg_origin.lat, leg_origin.lon),
                        end=(leg_destination.lat, leg_destination.lon),
                        distance_km=240.0,
                        duration_s=10_100.0,
                        money=220.0,
                        co2=310.0,
                        toll_confidence=0.88,
                    ),
                    _mk_option(
                        "leg0_green",
                        start=(leg_origin.lat, leg_origin.lon),
                        end=(leg_destination.lat, leg_destination.lon),
                        distance_km=248.0,
                        duration_s=10_700.0,
                        money=205.0,
                        co2=280.0,
                        toll_confidence=0.82,
                    ),
                ],
                [],
                3,
            )
        return (
            [
                _mk_option(
                    "leg1_fast",
                    start=(leg_origin.lat, leg_origin.lon),
                    end=(leg_destination.lat, leg_destination.lon),
                    distance_km=325.0,
                    duration_s=13_800.0,
                    money=285.0,
                    co2=390.0,
                    toll_confidence=0.91,
                ),
                _mk_option(
                    "leg1_green",
                    start=(leg_origin.lat, leg_origin.lon),
                    end=(leg_destination.lat, leg_destination.lon),
                    distance_km=336.0,
                    duration_s=14_600.0,
                    money=272.0,
                    co2=355.0,
                    toll_confidence=0.86,
                ),
            ],
            [],
            3,
        )

    def pareto_selector(routes: list[RouteOption], limit: int) -> list[RouteOption]:
        ranked = sorted(
            routes,
            key=lambda route: (
                route.metrics.duration_s,
                route.metrics.monetary_cost,
                route.metrics.emissions_kg,
                route.id,
            ),
        )
        return ranked[: max(1, limit)]

    result = await compose_multileg_route_options(
        origin=origin,
        destination=destination,
        waypoints=[waypoint],
        max_alternatives=3,
        leg_solver=leg_solver,
        pareto_selector=pareto_selector,
        option_prefix="route",
    )

    assert result.options, "Expected composed multi-leg route options"
    assert result.candidate_fetches == 6
    top = result.options[0]
    assert top.legs is not None
    assert len(top.legs) == 2
    assert top.metrics.distance_km > 0
    assert top.metrics.duration_s > 0
    assert top.uncertainty is not None
    assert top.uncertainty_samples_meta is not None
    assert top.toll_confidence is not None
    assert len(top.geometry.coordinates) >= 3
