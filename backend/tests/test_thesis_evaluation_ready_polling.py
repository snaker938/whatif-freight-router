from __future__ import annotations

import httpx
import pytest

import scripts.run_thesis_evaluation as thesis_module


pytestmark = pytest.mark.thesis_results


def test_wait_for_backend_ready_accepts_eventual_ready_state() -> None:
    calls = {"count": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            payload = {
                "strict_route_ready": False,
                "route_graph": {"status": "warming_up", "state": "loading"},
                "strict_live": {"ok": True},
            }
        else:
            payload = {
                "strict_route_ready": True,
                "route_graph": {"status": "ok", "state": "ready"},
                "strict_live": {"ok": True},
            }
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(_handler), base_url="http://backend.test")
    try:
        payload = thesis_module._wait_for_backend_ready(
            client,
            backend_url="http://backend.test",
            timeout_seconds=5.0,
            poll_seconds=0.01,
        )
    finally:
        client.close()

    assert payload["strict_route_ready"] is True
    assert calls["count"] == 3
    assert payload["compute_ms"] >= 0.0


def test_wait_for_backend_ready_times_out_with_state_details() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "strict_route_ready": False,
                "route_graph": {"status": "warming_up", "state": "loading"},
                "strict_live": {"ok": True},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler), base_url="http://backend.test")
    try:
        with pytest.raises(
            RuntimeError,
            match="backend_not_ready: route_graph_status=warming_up route_graph_state=loading route_graph_phase=None strict_live_ok=True recommended_action=None",
        ):
            thesis_module._wait_for_backend_ready(
                client,
                backend_url="http://backend.test",
                timeout_seconds=0.03,
                poll_seconds=0.01,
            )
    finally:
        client.close()


def test_wait_for_backend_ready_handles_transient_http_failure() -> None:
    calls = {"count": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"detail": "warming"})
        return httpx.Response(
            200,
            json={
                "strict_route_ready": True,
                "route_graph": {"status": "ok", "state": "ready"},
                "strict_live": {"ok": True},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler), base_url="http://backend.test")
    try:
        payload = thesis_module._wait_for_backend_ready(
            client,
            backend_url="http://backend.test",
            timeout_seconds=1.0,
            poll_seconds=0.01,
        )
    finally:
        client.close()

    assert payload["strict_route_ready"] is True
    assert calls["count"] == 2
