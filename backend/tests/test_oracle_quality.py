from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.settings import settings


def test_oracle_quality_dashboard_empty_then_populated(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "out_dir", str(out_dir))

    with TestClient(app) as client:
        initial = client.get("/oracle/quality/dashboard")
        assert initial.status_code == 200
        initial_payload = initial.json()
        assert initial_payload["total_checks"] == 0
        assert initial_payload["source_count"] == 0
        assert initial_payload["sources"] == []

        check_a_ok = client.post(
            "/oracle/quality/check",
            json={
                "source": "feed_a",
                "schema_valid": True,
                "signature_valid": True,
                "freshness_s": 120.0,
                "latency_ms": 20.0,
                "record_count": 12,
                "observed_at_utc": "2026-02-13T10:00:00Z",
            },
        )
        assert check_a_ok.status_code == 200
        assert check_a_ok.json()["passed"] is True

        check_a_bad = client.post(
            "/oracle/quality/check",
            json={
                "source": "feed_a",
                "schema_valid": False,
                "signature_valid": True,
                "freshness_s": 2200.0,
                "latency_ms": 40.0,
                "record_count": 5,
                "observed_at_utc": "2026-02-13T10:05:00Z",
                "error": "schema mismatch",
            },
        )
        assert check_a_bad.status_code == 200
        assert check_a_bad.json()["passed"] is False

        check_b_sig_fail = client.post(
            "/oracle/quality/check",
            json={
                "source": "feed_b",
                "schema_valid": True,
                "signature_valid": False,
                "freshness_s": 60.0,
                "latency_ms": 15.0,
                "record_count": 8,
                "observed_at_utc": "2026-02-13T10:10:00Z",
                "error": "signature invalid",
            },
        )
        assert check_b_sig_fail.status_code == 200
        assert check_b_sig_fail.json()["passed"] is False

        dashboard = client.get("/oracle/quality/dashboard")
        assert dashboard.status_code == 200
        payload = dashboard.json()
        assert payload["total_checks"] == 3
        assert payload["source_count"] == 2
        by_source = {item["source"]: item for item in payload["sources"]}

        feed_a = by_source["feed_a"]
        assert feed_a["check_count"] == 2
        assert feed_a["schema_failures"] == 1
        assert feed_a["signature_failures"] == 0
        assert feed_a["stale_count"] == 1
        assert abs(feed_a["pass_rate"] - 0.5) <= 1e-9
        assert abs(feed_a["avg_latency_ms"] - 30.0) <= 1e-9

        feed_b = by_source["feed_b"]
        assert feed_b["check_count"] == 1
        assert feed_b["schema_failures"] == 0
        assert feed_b["signature_failures"] == 1
        assert feed_b["stale_count"] == 0
        assert feed_b["pass_rate"] == 0.0

        csv_resp = client.get("/oracle/quality/dashboard.csv")
        assert csv_resp.status_code == 200
        assert csv_resp.headers["content-type"].startswith("text/csv")
        csv_text = csv_resp.text
        assert "source,check_count,pass_rate" in csv_text
        assert "feed_a" in csv_text
        assert "feed_b" in csv_text
