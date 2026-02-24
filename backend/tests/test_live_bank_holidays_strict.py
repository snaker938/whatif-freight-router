from __future__ import annotations

from app.live_data_sources import live_bank_holidays


def test_live_bank_holidays_injects_as_of_from_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.live_data_sources._fetch_json",
        lambda **kwargs: {  # noqa: ARG005
            "england-and-wales": {"events": []},
            "_live_diagnostics": {"as_of_utc": "2026-02-24T12:00:00Z"},
        },
    )
    payload = live_bank_holidays()
    assert isinstance(payload, dict)
    assert payload.get("as_of_utc") == "2026-02-24T12:00:00Z"


def test_live_bank_holidays_preserves_existing_as_of(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.live_data_sources._fetch_json",
        lambda **kwargs: {  # noqa: ARG005
            "england-and-wales": {"events": []},
            "as_of_utc": "2026-02-23T01:02:03Z",
            "_live_diagnostics": {"as_of_utc": "2026-02-24T12:00:00Z"},
        },
    )
    payload = live_bank_holidays()
    assert isinstance(payload, dict)
    assert payload.get("as_of_utc") == "2026-02-23T01:02:03Z"


def test_live_bank_holidays_uses_dedicated_allowed_hosts(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_fetch_json(**kwargs):
        captured["allowed_hosts_csv"] = str(kwargs.get("allowed_hosts_csv", ""))
        return {"england-and-wales": {"events": []}}

    monkeypatch.setattr("app.live_data_sources._fetch_json", _fake_fetch_json)
    monkeypatch.setattr(
        "app.live_data_sources.settings.live_bank_holidays_allowed_hosts",
        "www.gov.uk,gov.uk",
    )
    payload = live_bank_holidays()
    assert isinstance(payload, dict)
    assert captured.get("allowed_hosts_csv") == "www.gov.uk,gov.uk"
