from __future__ import annotations

from datetime import UTC, datetime

from app.departure_profile import time_of_day_multiplier_uk


def test_departure_profile_weekday_vs_weekend_differs() -> None:
    weekday = time_of_day_multiplier_uk(datetime(2026, 2, 18, 8, 30, tzinfo=UTC))
    weekend = time_of_day_multiplier_uk(datetime(2026, 2, 22, 8, 30, tzinfo=UTC))
    assert weekday.profile_day == "weekday"
    assert weekend.profile_day == "weekend"
    assert weekday.multiplier != weekend.multiplier


def test_departure_profile_is_timezone_aware_and_deterministic() -> None:
    a = time_of_day_multiplier_uk(datetime(2026, 3, 29, 0, 30, tzinfo=UTC))
    b = time_of_day_multiplier_uk(datetime(2026, 3, 29, 0, 30, tzinfo=UTC))
    assert a.multiplier == b.multiplier
    assert a.local_time_iso == b.local_time_iso
    assert a.profile_source
