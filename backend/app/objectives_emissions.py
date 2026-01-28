from __future__ import annotations

from .vehicles import VehicleProfile


def speed_factor(speed_kmh: float, *, eco_speed_kmh: float = 60.0) -> float:
    """Speed adjustment factor (starter placeholder, tuned for visible trade-offs).

    The earlier v0 version was mostly monotonic (slower => worse), which often caused
    a *single* route to dominate all OSRM alternatives, making the Pareto plot and
    sliders feel "broken".

    This v0.2 tweak uses a simple *U‑shaped* relationship:
      - very low speeds: stop/start + idling => higher emissions per km
      - moderate speeds (~eco_speed_kmh): best
      - very high speeds: aerodynamic/rolling losses => higher emissions per km

    It is still a placeholder (replace with a validated model later), but it produces
    more realistic multi-objective tension for the UI.
    """

    if speed_kmh <= 0:
        return 1.0

    s = float(speed_kmh)
    eco = max(10.0, float(eco_speed_kmh))

    # Quadratic penalty away from eco speed.
    x = (s - eco) / eco
    factor = 1.0 + 0.85 * (x * x)

    # Extra stop/start penalty at very low speeds (urban traffic).
    if s < 20:
        factor += 0.18
    if s < 10:
        factor += 0.22

    # Slight extra penalty at motorway+ speeds.
    if s > 90:
        factor += 0.12

    # Keep a sane range for a demo.
    return min(max(factor, 1.0), 2.75)


def route_emissions_kg(
    *,
    vehicle: VehicleProfile,
    segment_distances_m: list[float],
    segment_durations_s: list[float],
    idle_speed_threshold_kmh: float = 5.0,
) -> float:
    """Compute total route emissions (kg CO2e) from OSRM segment annotations.

    This is a deliberately simple placeholder model:
      - moving emissions = mass_tonnes * distance_km * EF(kg/t-km) * speed_factor
      - optional idle/queue emissions when speed is very low
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
