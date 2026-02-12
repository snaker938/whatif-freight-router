from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock


@dataclass
class EndpointStats:
    request_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    max_duration_ms: float = 0.0


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._created_at = datetime.now(UTC).isoformat()
        self._endpoints: dict[str, EndpointStats] = {}

    def record(self, endpoint: str, *, duration_ms: float, error: bool = False) -> None:
        name = endpoint.strip() or "unknown"
        d_ms = max(float(duration_ms), 0.0)

        with self._lock:
            stats = self._endpoints.setdefault(name, EndpointStats())
            stats.request_count += 1
            if error:
                stats.error_count += 1
            stats.total_duration_ms += d_ms
            if d_ms > stats.max_duration_ms:
                stats.max_duration_ms = d_ms

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            endpoints: dict[str, dict[str, float | int]] = {}
            total_requests = 0
            total_errors = 0

            for name in sorted(self._endpoints):
                stats = self._endpoints[name]
                total_requests += stats.request_count
                total_errors += stats.error_count
                avg_duration_ms = (
                    stats.total_duration_ms / stats.request_count if stats.request_count else 0.0
                )
                endpoints[name] = {
                    "request_count": stats.request_count,
                    "error_count": stats.error_count,
                    "total_duration_ms": round(stats.total_duration_ms, 3),
                    "avg_duration_ms": round(avg_duration_ms, 3),
                    "max_duration_ms": round(stats.max_duration_ms, 3),
                }

            return {
                "created_at": self._created_at,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "endpoint_count": len(endpoints),
                "endpoints": endpoints,
            }

    def reset(self) -> None:
        with self._lock:
            self._created_at = datetime.now(UTC).isoformat()
            self._endpoints.clear()


METRICS = MetricsStore()


def record_request(endpoint: str, *, duration_ms: float, error: bool = False) -> None:
    METRICS.record(endpoint, duration_ms=duration_ms, error=error)


def metrics_snapshot() -> dict[str, object]:
    return METRICS.snapshot()


def reset_metrics() -> None:
    METRICS.reset()
