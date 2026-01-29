from __future__ import annotations

from .vehicles import VehicleProfile


def speed_factor(speed_kmh: float, *, eco_speed_kmh: float = 55.0) -> float:
    """Speed adjustment factor used for emissions (and optionally cost).

    Design goal (demo / UI):
      - avoid a single route dominating every alternative (which collapses the Pareto set to 1)
      - create visible *trade-offs* between time, money and CO₂

    This is a deliberately simple, placeholder curve:

      - below eco speed: only a mild penalty (stop/start is better handled by idle emissions)
      - above eco speed: stronger penalty (aerodynamic/rolling losses grow quickly)
      - very high speeds: an extra motorway+ surcharge

    It is not a validated emissions model, but it produces better multi-objective
    behaviour for the app.
    """
    if speed_kmh <= 0:
        return 1.0

    s = float(speed_kmh)
    eco = max(10.0, float(eco_speed_kmh))

    if s <= eco:
        # Mild penalty when crawling; near-zero speeds are handled by idle emissions.
        x = (eco - s) / eco
        factor = 1.0 + 0.18 * (x * x)
    else:
        # Stronger penalty above eco speed.
        x = (s - eco) / eco
        factor = 1.0 + 1.25 * (x * x)

    # Extra motorway+ surcharge
    if s > 90:
        factor += 0.30
    if s > 110:
        factor += 0.20

    # Keep a sane range for a demo.
    return min(max(factor, 1.0), 3.25)


def route_emissions_kg(
    *,
    vehicle: VehicleProfile,
    segment_distances_m: list[float],
    segment_durations_s: list[float],
    idle_speed_threshold_kmh: float = 5.0,
) -> float:
    """Compute total route emissions (kg CO2e) from OSRM segment annotations.

    Placeholder model:
      - moving emissions = mass_tonnes * distance_km * EF(kg/t-km) * speed_factor
      - optional idle/queuing emissions when speed is very low
    """
    if len(segment_distances_m) != len(segment_durations_s):
        raise ValueError("segment arrays must be same length")

    total = 0.0
    for d_m, t_s in zip(segment_distances_m, segment_durations_s, strict=True):
        d_m = max(float(d_m), 0.0)
        t_s = max(float(t_s), 0.0)

        d_km = d_m / 1000.0
        speed_kmh = (d_m / t_s) * 3.6 if t_s > 0 else 0.0

        # moving emissions (tonne-km baseline with speed factor)
        total += (
            vehicle.mass_tonnes
            * d_km
            * vehicle.emission_factor_kg_per_tkm
            * speed_factor(speed_kmh)
        )

        # Optional extra idle/queuing emissions for very low speeds
        if speed_kmh < idle_speed_threshold_kmh and t_s > 0:
            total += (t_s / 3600.0) * vehicle.idle_emissions_kg_per_hour

    return total
