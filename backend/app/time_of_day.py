from __future__ import annotations

from datetime import datetime


def time_of_day_multiplier(departure_time_utc: datetime | None) -> float:
    """Deterministic traffic profile multiplier from UTC departure time.

    Profile bands are intentionally simple and stable for reproducible experiments.
    """
    if departure_time_utc is None:
        return 1.0

    hour = int(departure_time_utc.hour)

    if 7 <= hour <= 9:
        return 1.22
    if 16 <= hour <= 18:
        return 1.18
    if 10 <= hour <= 15:
        return 1.06
    if 19 <= hour <= 21:
        return 1.04
    if 0 <= hour <= 4:
        return 0.92
    return 1.0

