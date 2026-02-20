from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional in dev/CI
    import osmium
except Exception:  # pragma: no cover
    osmium = None  # type: ignore[assignment]


def _write_geojson(*, features: list[dict[str, Any]], output_geojson: Path) -> None:
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    output_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def _is_tolled_props(props: dict[str, Any]) -> bool:
    name = str(props.get("name", "")).strip().lower()
    road_class = str(props.get("road_class", "")).strip().lower()
    toll_tag = str(props.get("toll", props.get("osm_toll", ""))).strip().lower()
    barrier = str(props.get("barrier", props.get("osm_barrier", ""))).strip().lower()
    highway = str(props.get("highway", props.get("osm_highway", ""))).strip().lower()
    operator = str(props.get("operator", "")).strip().lower()
    return (
        "toll" in name
        or road_class in {"crossing", "motorway_toll", "tunnel_toll", "bridge_toll"}
        or toll_tag in {"yes", "hgv", "all"}
        or barrier == "toll_booth"
        or highway == "toll_gantry"
        or ("tunnel" in name and operator)
    )


def _extract_from_geojson(*, source_geojson: Path) -> list[dict[str, Any]]:
    payload = json.loads(source_geojson.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid source geojson payload")
    features = payload.get("features", [])
    if not isinstance(features, list):
        features = []
    filtered: list[dict] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties", {})
        if not isinstance(props, dict):
            props = {}
        if _is_tolled_props(props):
            filtered.append(feature)
    return filtered


def _way_coords(way_nodes: Iterable[Any]) -> list[list[float]]:
    out: list[list[float]] = []
    for node in way_nodes:
        try:
            lon = float(node.lon)
            lat = float(node.lat)
        except Exception:
            continue
        out.append([lon, lat])
    return out


def _extract_from_osm_pbf(*, source_pbf: Path) -> list[dict[str, Any]]:
    if osmium is None:
        if source_pbf.suffix.lower() == ".osm":
            return _extract_from_osm_xml(source_osm=source_pbf)
        raise RuntimeError("pyosmium is required for OSM PBF extraction.")

    class _RelationCollector(osmium.SimpleHandler):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.member_way_ids: set[int] = set()

        def relation(self, r: Any) -> None:
            tags = {str(k): str(v) for k, v in r.tags}
            props = {
                "name": tags.get("name", ""),
                "road_class": tags.get("road_class", tags.get("route", "default")).lower(),
                "toll": tags.get("toll", ""),
                "osm_toll": tags.get("toll", ""),
                "osm_highway": tags.get("highway", ""),
                "barrier": tags.get("barrier", ""),
                "operator": tags.get("operator", "default").lower(),
            }
            if not _is_tolled_props(props):
                return
            for member in r.members:
                if getattr(member, "type", None) == "w":
                    try:
                        self.member_way_ids.add(int(member.ref))
                    except Exception:
                        continue

    collector = _RelationCollector()
    collector.apply_file(str(source_pbf), locations=False)
    relation_way_ids = collector.member_way_ids

    class _TollHandler(osmium.SimpleHandler):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.features: list[dict[str, Any]] = []

        def way(self, w: Any) -> None:
            tags = {str(k): str(v) for k, v in w.tags}
            props = {
                "id": f"way/{w.id}",
                "crossing_id": f"way/{w.id}",
                "name": tags.get("name", ""),
                "operator": tags.get("operator", "default").lower(),
                "road_class": tags.get("road_class", tags.get("highway", "default")).lower(),
                "direction": tags.get("direction", tags.get("oneway", "both")).lower(),
                "toll": tags.get("toll", ""),
                "osm_toll": tags.get("toll", ""),
                "osm_highway": tags.get("highway", ""),
                "barrier": tags.get("barrier", ""),
            }
            way_id = int(w.id)
            if not _is_tolled_props(props) and way_id not in relation_way_ids:
                return
            coords = _way_coords(w.nodes)
            if len(coords) < 2:
                return
            self.features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "LineString", "coordinates": coords},
                }
            )

        def node(self, n: Any) -> None:
            tags = {str(k): str(v) for k, v in n.tags}
            props = {
                "id": f"node/{n.id}",
                "crossing_id": f"node/{n.id}",
                "name": tags.get("name", ""),
                "operator": tags.get("operator", "default").lower(),
                "road_class": tags.get("road_class", tags.get("highway", "crossing")).lower(),
                "direction": tags.get("direction", tags.get("oneway", "both")).lower(),
                "toll": tags.get("toll", ""),
                "osm_toll": tags.get("toll", ""),
                "osm_highway": tags.get("highway", ""),
                "barrier": tags.get("barrier", ""),
            }
            if not _is_tolled_props(props):
                return
            if props.get("barrier", "") != "toll_booth" and props.get("osm_highway", "") != "toll_gantry":
                return
            try:
                lon = float(n.lon)
                lat = float(n.lat)
            except Exception:
                return
            self.features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                }
            )

    handler = _TollHandler()
    handler.apply_file(str(source_pbf), locations=True)
    return handler.features


def _extract_from_osm_xml(*, source_osm: Path) -> list[dict[str, Any]]:
    tree = ET.parse(source_osm)
    root = tree.getroot()
    node_index: dict[str, tuple[float, float]] = {}
    node_tags: dict[str, dict[str, str]] = {}
    for node in root.findall("node"):
        node_id = node.attrib.get("id")
        if not node_id:
            continue
        try:
            lat = float(node.attrib.get("lat", "nan"))
            lon = float(node.attrib.get("lon", "nan"))
        except ValueError:
            continue
        node_index[node_id] = (lat, lon)
        tags: dict[str, str] = {}
        for child in list(node):
            if child.tag == "tag":
                key = str(child.attrib.get("k", "")).strip()
                value = str(child.attrib.get("v", "")).strip()
                if key:
                    tags[key] = value
        if tags:
            node_tags[node_id] = tags

    relation_way_ids: set[str] = set()
    for rel in root.findall("relation"):
        tags: dict[str, str] = {}
        member_way_refs: list[str] = []
        for child in list(rel):
            if child.tag == "tag":
                key = str(child.attrib.get("k", "")).strip()
                value = str(child.attrib.get("v", "")).strip()
                if key:
                    tags[key] = value
            elif child.tag == "member":
                if str(child.attrib.get("type", "")).strip() == "way":
                    ref = str(child.attrib.get("ref", "")).strip()
                    if ref:
                        member_way_refs.append(ref)
        props = {
            "name": tags.get("name", ""),
            "operator": tags.get("operator", "default").lower(),
            "road_class": tags.get("road_class", tags.get("route", "default")).lower(),
            "direction": tags.get("direction", tags.get("oneway", "both")).lower(),
            "toll": tags.get("toll", ""),
            "osm_toll": tags.get("toll", ""),
            "osm_highway": tags.get("highway", ""),
            "barrier": tags.get("barrier", ""),
        }
        if _is_tolled_props(props):
            relation_way_ids.update(member_way_refs)

    features: list[dict[str, Any]] = []
    for way in root.findall("way"):
        tags: dict[str, str] = {}
        refs: list[str] = []
        for child in list(way):
            if child.tag == "tag":
                key = str(child.attrib.get("k", "")).strip()
                value = str(child.attrib.get("v", "")).strip()
                if key:
                    tags[key] = value
            elif child.tag == "nd":
                ref = str(child.attrib.get("ref", "")).strip()
                if ref:
                    refs.append(ref)
        props = {
            "id": f"way/{way.attrib.get('id', '')}",
            "crossing_id": f"way/{way.attrib.get('id', '')}",
            "name": tags.get("name", ""),
            "operator": tags.get("operator", "default").lower(),
            "road_class": tags.get("road_class", tags.get("highway", "default")).lower(),
            "direction": tags.get("direction", tags.get("oneway", "both")).lower(),
            "toll": tags.get("toll", ""),
            "osm_toll": tags.get("toll", ""),
            "osm_highway": tags.get("highway", ""),
            "barrier": tags.get("barrier", ""),
        }
        if not _is_tolled_props(props) and str(way.attrib.get("id", "")) not in relation_way_ids:
            continue
        coords: list[list[float]] = []
        for ref in refs:
            coord = node_index.get(ref)
            if coord is None:
                continue
            lat, lon = coord
            coords.append([lon, lat])
        if len(coords) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )

    for node_id, tags in node_tags.items():
        props = {
            "id": f"node/{node_id}",
            "crossing_id": f"node/{node_id}",
            "name": tags.get("name", ""),
            "operator": tags.get("operator", "default").lower(),
            "road_class": tags.get("road_class", tags.get("highway", "crossing")).lower(),
            "direction": tags.get("direction", tags.get("oneway", "both")).lower(),
            "toll": tags.get("toll", ""),
            "osm_toll": tags.get("toll", ""),
            "osm_highway": tags.get("highway", ""),
            "barrier": tags.get("barrier", ""),
        }
        if not _is_tolled_props(props):
            continue
        if props.get("barrier", "") != "toll_booth" and props.get("osm_highway", "") != "toll_gantry":
            continue
        coord = node_index.get(node_id)
        if coord is None:
            continue
        lat, lon = coord
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )
    return features


def extract(*, source_geojson: Path, output_geojson: Path) -> None:
    suffix = source_geojson.suffix.lower()
    if suffix in {".pbf", ".osm"}:
        filtered = _extract_from_osm_pbf(source_pbf=source_geojson)
    else:
        filtered = _extract_from_geojson(source_geojson=source_geojson)
    _write_geojson(features=filtered, output_geojson=output_geojson)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract UK toll assets from OSM PBF/OSM (preferred) or GeoJSON.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("backend/assets/uk/uk-latest.osm.pbf"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/assets/uk/osm_toll_assets.geojson"),
    )
    args = parser.parse_args()
    extract(source_geojson=args.source, output_geojson=args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
