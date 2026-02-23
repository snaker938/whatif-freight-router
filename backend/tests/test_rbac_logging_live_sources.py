from __future__ import annotations

from datetime import UTC
from pathlib import Path

from starlette.requests import Request

import app.live_data_sources as live_data_sources
from app.logging_utils import _parse_level, get_logger, log_event
from app.rbac import require_role
from app.settings import settings


def test_require_role_is_noop_shim() -> None:
    req = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    require_role(req, "public")
    require_role(req, "user")
    require_role(req, "admin")


def test_logging_helpers_parse_levels_and_emit_event(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path))

    assert _parse_level("debug") > 0
    assert _parse_level("not_a_level") > 0

    logger1 = get_logger()
    handlers_before = len(logger1.handlers)
    logger2 = get_logger()
    assert logger1 is logger2
    assert len(logger2.handlers) == handlers_before

    log_event("unit_test_event", path="/health", status=200)


def test_live_data_source_url_and_time_helpers() -> None:
    assert live_data_sources._url_allowed("https://api.example.com/v1", allowed_hosts_raw="api.example.com")
    assert not live_data_sources._url_allowed("http://api.example.com/v1", allowed_hosts_raw="api.example.com")
    assert not live_data_sources._url_allowed("https://other.example.com/v1", allowed_hosts_raw="api.example.com")
    assert live_data_sources._url_allowed("https://free.example.com/v1", allowed_hosts_raw="*")

    parsed = live_data_sources._parse_as_of("2026-02-23T11:22:33Z")
    assert parsed is not None
    assert parsed.tzinfo == UTC
    assert parsed.hour == 11

    assert live_data_sources._parse_hour_value("2026-02-23T19:10:00Z") == 19
    assert live_data_sources._parse_hour_value("07:45") == 7


def test_live_data_source_extract_hourly_values_handles_nested_payload() -> None:
    payload = {
        "series": [
            {"timestamp": "2026-02-23T08:00:00Z", "diesel_price": 1.5},
            {"timestamp": "2026-02-23T09:00:00Z", "diesel_price": 1.6},
        ]
    }

    rows = live_data_sources._extract_hourly_values(payload, keys=("price", "diesel"))
    assert len(rows) >= 2
    hours = [hour for hour, _value in rows]
    assert 8 in hours
    assert 9 in hours
