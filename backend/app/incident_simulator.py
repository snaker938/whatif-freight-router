from __future__ import annotations

import hashlib
import random

from .models import IncidentSimulatorConfig, SimulatedIncidentEvent


def _seed_for_route(config: IncidentSimulatorConfig, route_key: str) -> int:
    base_seed = config.seed if config.seed is not None else 0
    seed_material = f"{base_seed}|{route_key}"
    return int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:16], 16)


def simulate_incident_events(
    *,
    config: IncidentSimulatorConfig,
    segment_distances_m: list[float],
    segment_durations_s: list[float],
    weather_incident_multiplier: float = 1.0,
    route_key: str = "",
) -> list[SimulatedIncidentEvent]:
    if not config.enabled or config.max_events_per_route <= 0:
        return []

    if len(segment_distances_m) != len(segment_durations_s):
        return []

    rng = random.Random(_seed_for_route(config, route_key))
    weather_factor = max(0.5, float(weather_incident_multiplier))

    events: list[SimulatedIncidentEvent] = []
    cumulative_time_s = 0.0
    event_specs: list[tuple[str, float, float]] = [
        ("dwell", float(config.dwell_rate_per_100km), float(config.dwell_delay_s)),
        ("accident", float(config.accident_rate_per_100km), float(config.accident_delay_s)),
        ("closure", float(config.closure_rate_per_100km), float(config.closure_delay_s)),
    ]

    for idx, (distance_m, duration_s) in enumerate(zip(segment_distances_m, segment_durations_s, strict=True)):
        distance_km = max(0.0, float(distance_m) / 1000.0)
        segment_duration_s = max(0.0, float(duration_s))

        for event_type, rate_per_100km, base_delay_s in event_specs:
            if base_delay_s <= 0.0 or rate_per_100km <= 0.0:
                continue
            probability = min(0.98, (rate_per_100km * weather_factor * distance_km) / 100.0)
            if rng.random() >= probability:
                continue

            delay_scale = 0.75 + (rng.random() * 0.5)
            delay_s = max(0.0, base_delay_s * delay_scale)
            start_offset_s = cumulative_time_s
            if segment_duration_s > 0:
                start_offset_s += rng.random() * segment_duration_s

            raw_id = f"{route_key}|{event_type}|{idx}|{start_offset_s:.3f}|{delay_s:.3f}"
            event_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]
            events.append(
                SimulatedIncidentEvent(
                    event_id=event_id,
                    event_type=event_type,
                    segment_index=idx,
                    start_offset_s=round(start_offset_s, 3),
                    delay_s=round(delay_s, 3),
                )
            )

        cumulative_time_s += segment_duration_s

    events.sort(key=lambda event: (event.start_offset_s, event.segment_index, event.event_type, event.event_id))
    return events[: config.max_events_per_route]

