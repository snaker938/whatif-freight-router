from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


class ORSError(RuntimeError):
    pass


_LOCALHOST_HOSTS = {"localhost", "127.0.0.1"}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKER_COMPOSE_PATH = _REPO_ROOT / "docker-compose.yml"
_ORS_CONFIG_PATH = _REPO_ROOT / "ors" / "config" / "ors-config.yml"
_ORS_GRAPH_ROOT = _REPO_ROOT / "ors" / "data"
_SOURCE_PBF_PATH = _REPO_ROOT / "osrm" / "data" / "pbf" / "region.osm.pbf"
_SOURCE_PBF_URL_PATH = _REPO_ROOT / "osrm" / "data" / "pbf" / ".region_pbf_url"


def _running_in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER") == "1"


def _truncate(text: str, *, limit: int = 240) -> str:
    value = str(text or "").strip().replace("\n", " ")
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _file_state(path: Path, *, inline_hash_limit_bytes: int = 1_048_576) -> dict[str, Any]:
    state: dict[str, Any] = {
        "path": str(path),
        "exists": bool(path.exists()),
    }
    if not path.exists():
        return state
    stat = path.stat()
    state["size"] = int(stat.st_size)
    state["mtime_ns"] = int(stat.st_mtime_ns)
    if path.is_file() and int(stat.st_size) <= int(inline_hash_limit_bytes):
        state["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return state


def _parse_ors_compose_image() -> str | None:
    if not _DOCKER_COMPOSE_PATH.is_file():
        return None
    try:
        lines = _DOCKER_COMPOSE_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    in_ors_service = False
    for line in lines:
        if re.match(r"^\s{2}ors:\s*$", line):
            in_ors_service = True
            continue
        if in_ors_service and re.match(r"^\s{2}[A-Za-z0-9_-]+:\s*$", line):
            break
        if not in_ors_service:
            continue
        match = re.match(r"^\s{4}image:\s*['\"]?([^'\"]+)['\"]?\s*$", line)
        if match:
            return str(match.group(1)).strip() or None
    return None


def _parse_graph_build_info(build_info_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not build_info_path.is_file():
        return payload
    try:
        lines = build_info_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return payload
    for raw in lines:
        line = raw.strip()
        if line.startswith("graph_build_date:"):
            payload["graph_build_date"] = line.split(":", 1)[1].strip()
        elif line.startswith("osm_date:"):
            payload["osm_date"] = line.split(":", 1)[1].strip()
        elif line.startswith("graph_version:"):
            payload["graph_version"] = line.split(":", 1)[1].strip()
        elif line.startswith("graph_size_bytes:"):
            payload["graph_size_bytes"] = line.split(":", 1)[1].strip()
    return payload


def local_ors_runtime_manifest(
    *,
    base_url: str,
    profile: str,
    vehicle_type: str | None,
) -> dict[str, Any]:
    graph_dir = _ORS_GRAPH_ROOT / str(profile)
    graph_files: list[dict[str, Any]] = []
    latest_graph_mtime_ns = 0
    graph_total_bytes = 0
    if graph_dir.is_dir():
        for path in sorted(child for child in graph_dir.iterdir() if child.is_file()):
            stat = path.stat()
            entry = {
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
            if path.name in {"graph_build_info.yml", "properties", "stamp.txt"}:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            latest_graph_mtime_ns = max(latest_graph_mtime_ns, int(stat.st_mtime_ns))
            graph_total_bytes += int(stat.st_size)
            graph_files.append(entry)

    build_info_path = graph_dir / "graph_build_info.yml"
    build_info_state = _file_state(build_info_path)
    build_info = _parse_graph_build_info(build_info_path)
    stamp_path = graph_dir / "stamp.txt"
    stamp_text = None
    if stamp_path.is_file():
        try:
            stamp_text = stamp_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            stamp_text = None
    pbf_url = None
    if _SOURCE_PBF_URL_PATH.is_file():
        try:
            pbf_url = _SOURCE_PBF_URL_PATH.read_text(encoding="utf-8").strip() or None
        except OSError:
            pbf_url = None

    config_state = _file_state(_ORS_CONFIG_PATH)
    pbf_state = _file_state(_SOURCE_PBF_PATH, inline_hash_limit_bytes=0)
    identity_status = "graph_identity_verified"
    if not graph_dir.is_dir():
        identity_status = "graph_missing"
    elif not build_info_path.is_file():
        identity_status = "graph_build_info_missing"
    elif not bool(config_state.get("exists")):
        identity_status = "config_missing"
    elif not bool(pbf_state.get("exists")):
        identity_status = "source_pbf_missing"
    elif int(build_info_state.get("mtime_ns") or 0) < int(config_state.get("mtime_ns") or 0):
        identity_status = "graph_predates_config"
    elif int(build_info_state.get("mtime_ns") or 0) < int(pbf_state.get("mtime_ns") or 0):
        identity_status = "graph_predates_source_pbf"

    manifest: dict[str, Any] = {
        "engine": "openrouteservice",
        "base_url": str(base_url or "").strip(),
        "profile": str(profile or "").strip() or "driving-car",
        "vehicle_type": str(vehicle_type or "").strip() or "default",
        "compose_image": _parse_ors_compose_image(),
        "config": config_state,
        "source_pbf": {
            **pbf_state,
            "source_url": pbf_url,
        },
        "graph_dir": str(graph_dir),
        "graph_dir_exists": bool(graph_dir.is_dir()),
        "graph_file_count": len(graph_files),
        "graph_total_bytes": int(graph_total_bytes),
        "graph_listing_digest": hashlib.sha256(_canon(graph_files)).hexdigest() if graph_files else None,
        "graph_latest_mtime_ns": int(latest_graph_mtime_ns) if latest_graph_mtime_ns > 0 else None,
        "graph_build_info": {
            **build_info_state,
            **build_info,
        },
        "graph_stamp": stamp_text,
        "identity_status": identity_status,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    manifest["manifest_hash"] = hashlib.sha256(
        _canon({key: value for key, value in manifest.items() if key != "recorded_at"})
    ).hexdigest()
    return manifest


def _format_ors_error(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                message = error.get("message")
                if code and message:
                    return f"ORS {resp.status_code} {code}: {message}"
                if message:
                    return f"ORS {resp.status_code}: {message}"
            message = data.get("message")
            if message:
                return f"ORS {resp.status_code}: {message}"
    except Exception:
        pass
    body = _truncate(resp.text or "")
    if body:
        return f"ORS {resp.status_code}: {body}"
    return f"ORS HTTP {resp.status_code}"


@dataclass(frozen=True, init=False)
class ORSRoute:
    distance_m: float
    duration_s: float
    coordinates_lon_lat: list[tuple[float, float]]
    profile: str
    response_format: str = "geojson"

    def __init__(
        self,
        *,
        distance_m: float,
        duration_s: float,
        coordinates_lon_lat: list[tuple[float, float]] | None = None,
        coordinates: list[tuple[float, float]] | None = None,
        profile: str = "driving-hgv",
        response_format: str = "geojson",
    ) -> None:
        normalized_coordinates = coordinates_lon_lat if coordinates_lon_lat is not None else coordinates
        object.__setattr__(self, "distance_m", float(distance_m))
        object.__setattr__(self, "duration_s", float(duration_s))
        object.__setattr__(
            self,
            "coordinates_lon_lat",
            [(float(lon), float(lat)) for lon, lat in list(normalized_coordinates or [])],
        )
        object.__setattr__(self, "profile", str(profile))
        object.__setattr__(self, "response_format", str(response_format))

    def as_dict(self) -> dict[str, Any]:
        return {
            "distance_m": float(self.distance_m),
            "duration_s": float(self.duration_s),
            "coordinates_lon_lat": [(float(lon), float(lat)) for lon, lat in self.coordinates_lon_lat],
            "profile": self.profile,
            "response_format": self.response_format,
        }


class ORSClient:
    def __init__(self, *, base_url: str, timeout_ms: int = 25_000) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(max(1.0, float(timeout_ms) / 1000.0), connect=5.0),
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self, *, health_path: str = "/v2/health") -> dict[str, Any]:
        path = "/" + str(health_path or "/v2/health").lstrip("/")
        resp = await self._client.get(f"{self.base_url}{path}")
        if resp.status_code >= 400:
            raise ORSError(_format_ors_error(resp))
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ORSError("ORS health endpoint returned a non-object payload")
        return payload

    async def fetch_route(
        self,
        *,
        coordinates_lon_lat: list[tuple[float, float]],
        profile: str,
        vehicle_type: str | None = None,
    ) -> ORSRoute:
        if len(coordinates_lon_lat) < 2:
            raise ORSError("ORS request requires at least origin and destination coordinates")

        # Official openrouteservice docs expose local instances at
        # /ors/v2/directions and document local Swagger/API playground support:
        # https://github.com/GIScience/openrouteservice#usage
        # https://giscience.github.io/openrouteservice/api-reference/
        url = f"{self.base_url}/v2/directions/{profile}/geojson"
        body: dict[str, Any] = {
            "coordinates": [[float(lon), float(lat)] for lon, lat in coordinates_lon_lat],
            "instructions": False,
            "elevation": False,
        }
        normalized_vehicle_type = str(vehicle_type or "").strip().lower()
        if normalized_vehicle_type:
            body["options"] = {"vehicle_type": normalized_vehicle_type}

        try:
            resp = await self._client.post(url, json=body)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:
            message = str(exc).strip() or repr(exc)
            hint = ""
            try:
                host = urlparse(self.base_url).hostname or ""
            except Exception:
                host = ""
            if _running_in_docker() and host in _LOCALHOST_HOSTS:
                hint = (
                    " Hint: you're running inside a container; localhost points to that container. "
                    "In docker-compose, set ORS_BASE_URL=http://ors:8082/ors."
                )
            elif (not _running_in_docker()) and host == "ors":
                hint = (
                    " Hint: `ors` is the docker-compose service name. "
                    "If you're running on the host, set ORS_BASE_URL=http://localhost:8082/ors."
                )
            raise ORSError(f"ORS request failed (base={self.base_url}): {message}{hint}") from exc

        if resp.status_code >= 400:
            raise ORSError(_format_ors_error(resp))

        payload = resp.json()
        if not isinstance(payload, dict):
            raise ORSError("ORS directions response was not a JSON object")

        features = payload.get("features")
        if not isinstance(features, list) or not features:
            raise ORSError("ORS directions returned no features")
        feature = features[0] if isinstance(features[0], dict) else {}
        geometry = feature.get("geometry", {})
        coords_raw = geometry.get("coordinates", [])
        properties = feature.get("properties", {})
        summary = properties.get("summary", {})
        if not isinstance(coords_raw, list) or len(coords_raw) < 2:
            raise ORSError("ORS directions geometry is missing usable coordinates")
        coordinates = [
            (float(item[0]), float(item[1]))
            for item in coords_raw
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
        if len(coordinates) < 2:
            raise ORSError("ORS directions geometry did not contain enough coordinates")
        distance_m = float(summary.get("distance") or 0.0)
        duration_s = float(summary.get("duration") or 0.0)
        if distance_m <= 0.0 or duration_s <= 0.0:
            raise ORSError("ORS directions summary is missing positive distance/duration")
        return ORSRoute(
            distance_m=distance_m,
            duration_s=duration_s,
            coordinates_lon_lat=coordinates,
            profile=str(profile),
        )
