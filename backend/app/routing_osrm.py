# backend/app/routing_osrm.py
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

import httpx


class OSRMError(RuntimeError):
    pass


class OSRMRetryableError(OSRMError):
    """An OSRM error that is likely transient and safe to retry."""

    pass


AlternativesParam = bool | int


_RETRYABLE_STATUS: Final[set[int]] = {408, 425, 429, 500, 502, 503, 504}
_LOCALHOST_HOSTS: Final[set[str]] = {"localhost", "127.0.0.1"}


def _running_in_docker() -> bool:
    # Best-effort detection; used only for better defaults / hints.
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER") == "1"


def _format_osrm_error(resp: httpx.Response) -> str:
    """Best-effort decode of OSRM JSON error payloads."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            code = data.get("code")
            message = data.get("message")
            if code and message:
                return f"OSRM {resp.status_code} {code}: {message}"
            if code:
                return f"OSRM {resp.status_code} {code}"
            if message:
                return f"OSRM {resp.status_code}: {message}"
    except Exception:
        # fall through to text
        pass

    body = (resp.text or "").strip().replace("\n", " ")
    if len(body) > 240:
        body = body[:240] + "…"
    if body:
        return f"OSRM {resp.status_code}: {body}"
    return f"OSRM HTTP {resp.status_code}"


class OSRMClient:
    def __init__(self, *, base_url: str, profile: str = "driving") -> None:
        self.base_url = base_url.rstrip("/")
        self.profile = profile

        # OSRM is local in this stack, so keep connect timeout snappy.
        # IMPORTANT: trust_env=False prevents corporate proxy env vars (HTTP_PROXY/HTTPS_PROXY)
        # from hijacking requests to localhost / docker service names.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            trust_env=False,
            headers={"accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch_routes(
        self,
        *,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        alternatives: AlternativesParam = True,
        exclude: str | None = None,
        via: list[tuple[float, float]] | None = None,
        max_retries: int = 8,
    ) -> list[dict[str, Any]]:
        """Fetch routes from OSRM.

        Notes on `alternatives`:

        - OSRM's public API accepts `alternatives=true|false`.
        - Some OSRM builds also accept an integer (number of alternative routes).
          We try the integer form first (when provided) and fall back to boolean
          for compatibility.

        exclude:
          Optional comma-separated classes to exclude (e.g. "motorway", "toll", "ferry").
          Depends on profile supporting excludable classes.

        via:
          Optional list of via points as (lat, lon). If provided, OSRM will route:
            origin -> via[0] -> ... -> via[n-1] -> destination
        """
        coords_parts: list[str] = [f"{origin_lon},{origin_lat}"]
        if via:
            coords_parts.extend([f"{lon},{lat}" for (lat, lon) in via])
        coords_parts.append(f"{dest_lon},{dest_lat}")
        coords = ";".join(coords_parts)

        url = f"{self.base_url}/route/v1/{self.profile}/{coords}"

        # Decide what to send for the `alternatives` query param.
        alt_values: list[str]
        if isinstance(alternatives, bool):
            alt_values = ["true" if alternatives else "false"]
        else:
            try:
                alt_i = int(alternatives)
            except Exception:
                alt_i = 1

            if alt_i <= 1:
                alt_values = ["false"]
            else:
                # Try numeric first, then fall back to boolean for builds that
                # only accept true/false.
                alt_values = [str(alt_i), "true"]

        base_params: dict[str, str] = {
            "overview": "full",
            "geometries": "geojson",
            "annotations": "true",
            "steps": "true",
        }
        if exclude:
            base_params["exclude"] = exclude

        max_retries_i = max(1, int(max_retries))

        async def _call(params: dict[str, str]) -> list[dict[str, Any]]:
            last_err: Exception | None = None

            for attempt in range(max_retries_i):
                try:
                    resp = await self._client.get(url, params=params)

                    # Fast-fail on most 4xx: these are usually request errors
                    # (bad param, no segment, etc.)
                    if 400 <= resp.status_code < 500 and resp.status_code not in _RETRYABLE_STATUS:
                        raise OSRMError(_format_osrm_error(resp))

                    # Retryable HTTP errors
                    if resp.status_code in _RETRYABLE_STATUS:
                        raise OSRMRetryableError(_format_osrm_error(resp))

                    resp.raise_for_status()
                    data = resp.json()

                    if data.get("code") != "Ok":
                        raise OSRMError(
                            f"OSRM error code={data.get('code')} message={data.get('message')}"
                        )

                    routes = data.get("routes", [])
                    if not isinstance(routes, list) or not routes:
                        raise OSRMError("OSRM returned no routes")

                    return routes

                except OSRMRetryableError as e:
                    last_err = e
                except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as e:
                    last_err = e
                except httpx.HTTPStatusError as e:
                    # Non-retryable HTTP errors (already handled above for 4xx),
                    # but keep this as a safety net.
                    last_err = e
                    raise OSRMError(str(e)) from e

                if attempt < max_retries_i - 1:
                    await asyncio.sleep(min(0.25 * (2**attempt), 2.0))

            # httpx exceptions can stringify to "" (e.g. some timeouts), so include the type.
            if last_err is None:
                detail = "unknown error"
            else:
                msg = str(last_err).strip()
                detail = (
                    f"{type(last_err).__name__}: {msg}"
                    if msg
                    else f"{type(last_err).__name__}: {last_err!r}"
                )

            # Add a targeted hint for the most common misconfigurations.
            hint = ""
            try:
                host = urlparse(self.base_url).hostname or ""
            except Exception:
                host = ""

            if _running_in_docker() and host in _LOCALHOST_HOSTS:
                hint = (
                    " Hint: you're running inside a container; `localhost` points to that container. "
                    "In docker-compose, set OSRM_BASE_URL=http://osrm:5000."
                )
            elif (not _running_in_docker()) and host == "osrm":
                hint = (
                    " Hint: `osrm` is the docker-compose service name. "
                    "If you're running the backend directly on your host, set OSRM_BASE_URL=http://localhost:5000."
                )

            raise OSRMError(
                "OSRM request failed after "
                f"{max_retries_i} retries (base={self.base_url}): {detail}{hint}"
            )

        last_error: Exception | None = None
        for idx, alt in enumerate(alt_values):
            params = dict(base_params)
            params["alternatives"] = alt
            try:
                return await _call(params)
            except OSRMError as e:
                last_error = e
                # If OSRM doesn't accept numeric alternatives, fall back to boolean.
                if idx < len(alt_values) - 1:
                    continue
                raise

        # Should be unreachable, but keep type-checkers happy.
        raise OSRMError(str(last_error) if last_error else "OSRM request failed")


def extract_segment_annotations(route: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Return (distances_m, durations_s) concatenated across legs."""
    legs = route.get("legs", [])
    distances: list[float] = []
    durations: list[float] = []

    for leg in legs:
        ann = (leg or {}).get("annotation", {})
        d = ann.get("distance", [])
        t = ann.get("duration", [])
        if isinstance(d, list) and isinstance(t, list) and len(d) == len(t):
            distances.extend([float(x) for x in d])
            durations.extend([float(x) for x in t])

    if not distances or not durations or len(distances) != len(durations):
        raise OSRMError("Missing or invalid OSRM annotations (distance/duration)")
    return distances, durations
