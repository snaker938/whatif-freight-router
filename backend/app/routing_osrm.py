# backend/app/routing_osrm.py
from __future__ import annotations

import asyncio
from typing import Any, Final

import httpx


class OSRMError(RuntimeError):
    pass


class OSRMRetryableError(OSRMError):
    """An OSRM error that is likely transient and safe to retry."""

    pass


AlternativesParam = bool | int


_RETRYABLE_STATUS: Final[set[int]] = {408, 425, 429, 500, 502, 503, 504}


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
        # Keep connect timeout snappy; OSRM should be local in this stack.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

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

        - OSRM's public API uses a boolean `alternatives=true|false`.
        - Some builds accept an integer, but many (including common Docker images) do not.
          For robustness we treat any integer > 1 as `alternatives=true`.

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

        alt_bool = alternatives if isinstance(alternatives, bool) else int(alternatives) > 1

        params: dict[str, str] = {
            "alternatives": "true" if alt_bool else "false",
            "overview": "full",
            "geometries": "geojson",
            "annotations": "true",
        }
        if exclude:
            params["exclude"] = exclude

        max_retries_i = max(1, int(max_retries))
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

        raise OSRMError(f"OSRM request failed after {max_retries_i} retries: {last_err}")


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
