from __future__ import annotations

import asyncio
from typing import Any

import httpx


class OSRMError(RuntimeError):
    pass


class OSRMClient:
    def __init__(self, *, base_url: str, profile: str = "driving") -> None:
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self._client = httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch_routes(
        self,
        *,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        alternatives: bool = True,
        max_retries: int = 30,
    ) -> list[dict[str, Any]]:
        coords = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        url = f"{self.base_url}/route/v1/{self.profile}/{coords}"
        params = {
            "alternatives": "true" if alternatives else "false",
            "overview": "full",
            "geometries": "geojson",
            "annotations": "true",
        }

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = await self._client.get(url, params=params)
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
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(0.25 * (2**attempt), 2.5))

        raise OSRMError(f"OSRM request failed after {max_retries} retries: {last_err}")


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
