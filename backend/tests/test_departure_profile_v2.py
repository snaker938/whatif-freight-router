from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

import app.calibration_loader as calibration_loader
from app.departure_profile import time_of_day_multiplier_uk
from app.settings import settings


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _departure_profile_payload(now_iso: str) -> dict[str, object]:
    return {
        "version": "departure_live_test_v1",
        "calibration_basis": "empirical",
        "as_of_utc": now_iso,
        "weekday": [1.05] * 1440,
        "weekend": [0.92] * 1440,
        "holiday": [0.88] * 1440,
    }


def _bank_holidays_payload(now_iso: str) -> dict[str, object]:
    return {
        "as_of_utc": now_iso,
        "england-and-wales": {"events": [{"date": "2026-12-25"}]},
    }


@pytest.fixture(autouse=True)
def _strict_runtime_test_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("STRICT_RUNTIME_TEST_BYPASS") is None:
        monkeypatch.setenv("STRICT_RUNTIME_TEST_BYPASS", "1")
    now_iso = _now_iso()
    monkeypatch.setattr(settings, "live_departure_profile_url", "https://live.example/departure")
    monkeypatch.setattr(calibration_loader, "live_departure_profiles", lambda: _departure_profile_payload(now_iso))
    monkeypatch.setattr(calibration_loader, "live_bank_holidays", lambda: _bank_holidays_payload(now_iso))
    calibration_loader.load_departure_profile.cache_clear()
    calibration_loader.load_uk_bank_holidays.cache_clear()


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
