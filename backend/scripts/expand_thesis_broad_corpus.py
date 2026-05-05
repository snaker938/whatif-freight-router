from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENDPOINT_VARIANT_COUNT = 5
ENDPOINT_VARIANT_OFFSETS = (
    (0.045, 0.030),
    (-0.038, 0.026),
    (0.031, -0.034),
    (-0.028, -0.022),
    (0.022, 0.041),
    (-0.019, 0.044),
    (0.027, -0.047),
    (-0.041, -0.015),
)


@dataclass(frozen=True)
class LocationPrototype:
    key: str
    label: str
    lat: float
    lon: float
    region_bucket: str
    rows: tuple[dict[str, str], ...]


def _float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_") or "loc"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _distance_bin(distance_km: float) -> str:
    if distance_km < 100.0:
        return "0-100 km"
    if distance_km < 250.0:
        return "100-250 km"
    if distance_km < 500.0:
        return "250-500 km"
    return "500+ km"


def _prototype_label(row: dict[str, str], prefix: str) -> str:
    od_id = _text(row.get("od_id"))
    if "_" in od_id:
        parts = [part for part in od_id.split("_") if part]
        if prefix == "origin" and parts:
            return parts[0]
        if prefix == "destination" and len(parts) > 1:
            return parts[-1]
    region = _text(row.get(f"{prefix}_region_bucket"), "region")
    return f"{region}_{prefix}"


def _make_location_prototypes(rows: list[dict[str, str]], prefix: str) -> list[LocationPrototype]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        lat = round(_float(row[f"{prefix}_lat"]), 6)
        lon = round(_float(row[f"{prefix}_lon"]), 6)
        key = f"{lat:.6f},{lon:.6f}"
        grouped[key].append(row)

    prototypes: list[LocationPrototype] = []
    for key, members in grouped.items():
        lat_text, lon_text = key.split(",")
        lat = float(lat_text)
        lon = float(lon_text)
        label = _slug(_prototype_label(members[0], prefix))
        region_bucket = _text(members[0].get(f"{prefix}_region_bucket"), "unknown")
        prototypes.append(
            LocationPrototype(
                key=key,
                label=label,
                lat=lat,
                lon=lon,
                region_bucket=region_bucket,
                rows=tuple(members),
            )
        )
    prototypes.sort(key=lambda item: (item.region_bucket, item.label, item.key))
    return prototypes


def _augment_location_prototypes(
    prototypes: list[LocationPrototype],
    *,
    variant_count: int,
) -> list[LocationPrototype]:
    if variant_count <= 0:
        return list(prototypes)
    out = list(prototypes)
    seen_keys = {item.key for item in prototypes}
    for prototype_index, prototype in enumerate(prototypes):
        for variant_index in range(variant_count):
            delta_lat, delta_lon = ENDPOINT_VARIANT_OFFSETS[(prototype_index + variant_index) % len(ENDPOINT_VARIANT_OFFSETS)]
            scale = 1.0 + (variant_index * 0.18)
            lat = round(prototype.lat + (delta_lat * scale), 6)
            lon = round(prototype.lon + (delta_lon * scale), 6)
            key = f"{lat:.6f},{lon:.6f}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(
                LocationPrototype(
                    key=key,
                    label=f"{prototype.label}_alt{variant_index + 1}",
                    lat=lat,
                    lon=lon,
                    region_bucket=prototype.region_bucket,
                    rows=prototype.rows,
                )
            )
    out.sort(key=lambda item: (item.region_bucket, item.label, item.key))
    return out


def _mean_numeric(rows: list[dict[str, str]], field: str, default: float = 0.0) -> float:
    values = [_float(row.get(field), float("nan")) for row in rows]
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else default


def _max_numeric(rows: list[dict[str, str]], field: str, default: float = 0.0) -> float:
    values = [_float(row.get(field), float("nan")) for row in rows]
    clean = [value for value in values if math.isfinite(value)]
    return max(clean) if clean else default


def _any_text(rows: list[dict[str, str]], field: str, default: str = "") -> str:
    for row in rows:
        text = _text(row.get(field))
        if text:
            return text
    return default


def _merge_source_mix(rows: list[dict[str, str]]) -> tuple[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        mix = _text(row.get("od_ambiguity_source_mix") or row.get("ambiguity_prior_source"))
        if not mix:
            continue
        parsed: dict[str, Any] | None = None
        try:
            maybe = json.loads(mix)
            if isinstance(maybe, dict):
                parsed = maybe
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                counts[str(key)] += max(1, _int(value, 1))
        else:
            for token in [piece.strip() for piece in mix.split(",") if piece.strip()]:
                counts[token] += 1
    return (
        json.dumps(dict(sorted(counts.items())), sort_keys=True, separators=(",", ":")) if counts else "",
        len(counts),
    )


def _candidate_rows(
    origins: list[LocationPrototype],
    destinations: list[LocationPrototype],
    source_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    source_ambiguity = [_float(row.get("od_ambiguity_index"), _float(row.get("ambiguity_index"))) for row in source_rows]
    ambiguity_cut = sorted(source_ambiguity)[len(source_ambiguity) // 2] if source_ambiguity else 0.0
    out: list[dict[str, Any]] = []
    for origin in origins:
        for destination in destinations:
            if origin.key == destination.key:
                continue
            distance_km = round(_haversine_km(origin.lat, origin.lon, destination.lat, destination.lon), 6)
            if distance_km < 25.0:
                continue
            distance_bin = _distance_bin(distance_km)
            pair_rows = list(origin.rows) + list(destination.rows)
            source_mix, source_mix_count = _merge_source_mix(pair_rows)
            ambiguity_index = round(
                max(
                    _mean_numeric(pair_rows, "od_ambiguity_index"),
                    _mean_numeric(pair_rows, "ambiguity_index"),
                ),
                6,
            )
            corpus_group = "ambiguity" if ambiguity_index >= ambiguity_cut else "representative"
            row = {
                "od_id": f"{origin.label}_{destination.label}",
                "origin_lat": round(origin.lat, 6),
                "origin_lon": round(origin.lon, 6),
                "destination_lat": round(destination.lat, 6),
                "destination_lon": round(destination.lon, 6),
                "straight_line_km": distance_km,
                "distance_bin": distance_bin,
                "profile_id": "",
                "corpus_group": corpus_group,
                "corpus_kind": corpus_group,
                "weight_time": round(_mean_numeric(pair_rows, "weight_time", 1.0), 3),
                "weight_money": round(_mean_numeric(pair_rows, "weight_money", 1.0), 3),
                "weight_co2": round(_mean_numeric(pair_rows, "weight_co2", 1.0), 3),
                "scenario_mode": _any_text(pair_rows, "scenario_mode", "no_sharing"),
                "weather_profile": _any_text(pair_rows, "weather_profile", "clear"),
                "weather_intensity": round(_mean_numeric(pair_rows, "weather_intensity", 1.0), 3),
                "departure_time_utc": _any_text(pair_rows, "departure_time_utc", "2026-03-21T10:00:00Z"),
                "stochastic_enabled": str(_bool(_any_text(pair_rows, "stochastic_enabled", "true"), True)).lower(),
                "stochastic_samples": max(24, _int(_mean_numeric(pair_rows, "stochastic_samples", 48.0), 48)),
                "search_budget": max(4, _int(_mean_numeric(pair_rows, "search_budget", 5.0), 5)),
                "evidence_budget": max(2, _int(_mean_numeric(pair_rows, "evidence_budget", 2.0), 2)),
                "world_count": max(48, _int(_mean_numeric(pair_rows, "world_count", 72.0), 72)),
                "certificate_threshold": round(max(0.8, _mean_numeric(pair_rows, "certificate_threshold", 0.82)), 3),
                "tau_stop": round(max(0.01, _mean_numeric(pair_rows, "tau_stop", 0.018)), 3),
                "max_alternatives": max(8, _int(_mean_numeric(pair_rows, "max_alternatives", 9.0), 9)),
                "optimization_mode": _any_text(pair_rows, "optimization_mode", "robust"),
                "origin_region_bucket": origin.region_bucket,
                "destination_region_bucket": destination.region_bucket,
                "corridor_bucket": f"{origin.region_bucket}_to_{destination.region_bucket}",
                "trip_length_bin": distance_bin,
                "ambiguity_index": ambiguity_index,
                "od_ambiguity_index": ambiguity_index,
                "candidate_probe_path_count": max(2, _int(_mean_numeric(pair_rows, "candidate_probe_path_count", 4.0), 4)),
                "candidate_probe_corridor_family_count": max(1, _int(_mean_numeric(pair_rows, "candidate_probe_corridor_family_count", 1.0), 1)),
                "candidate_probe_objective_spread": round(_mean_numeric(pair_rows, "candidate_probe_objective_spread", 0.12), 6),
                "candidate_probe_nominal_margin": round(_mean_numeric(pair_rows, "candidate_probe_nominal_margin", 0.95), 6),
                "candidate_probe_engine_disagreement_prior": round(_mean_numeric(pair_rows, "candidate_probe_engine_disagreement_prior", 0.22), 6),
                "candidate_probe_toll_disagreement_rate": round(_mean_numeric(pair_rows, "candidate_probe_toll_disagreement_rate", 0.02), 6),
                "hard_case_prior": round(_mean_numeric(pair_rows, "hard_case_prior", 0.18), 6),
                "ambiguity_prior_source": _any_text(pair_rows, "ambiguity_prior_source", "expanded_broad_endpoint_recombination"),
                "ambiguity_prior_sample_count": max(2, _int(_mean_numeric(pair_rows, "ambiguity_prior_sample_count", 2.0), 2)),
                "ambiguity_prior_support_count": max(1, _int(_mean_numeric(pair_rows, "ambiguity_prior_support_count", 1.0), 1)),
                "od_ambiguity_confidence": round(_mean_numeric(pair_rows, "od_ambiguity_confidence", 0.72), 6),
                "od_ambiguity_source_count": max(1, source_mix_count),
                "od_ambiguity_source_mix": source_mix,
                "od_ambiguity_source_mix_count": max(1, source_mix_count),
                "od_ambiguity_source_support": source_mix,
                "od_ambiguity_source_support_strength": round(_mean_numeric(pair_rows, "od_ambiguity_source_support_strength", 0.62), 6),
                "od_ambiguity_source_entropy": round(_mean_numeric(pair_rows, "od_ambiguity_source_entropy", 0.68), 6),
                "od_ambiguity_support_ratio": round(_mean_numeric(pair_rows, "od_ambiguity_support_ratio", 0.7), 6),
                "od_ambiguity_prior_strength": round(_max_numeric(pair_rows, "od_ambiguity_prior_strength", ambiguity_index), 6),
                "od_ambiguity_family_density": round(_mean_numeric(pair_rows, "od_ambiguity_family_density", 0.22), 6),
                "od_ambiguity_margin_pressure": round(_mean_numeric(pair_rows, "od_ambiguity_margin_pressure", 0.18), 6),
                "od_ambiguity_spread_pressure": round(_mean_numeric(pair_rows, "od_ambiguity_spread_pressure", 0.14), 6),
                "od_ambiguity_toll_instability": round(_mean_numeric(pair_rows, "od_ambiguity_toll_instability", 0.04), 6),
                "ambiguity_budget_prior": round(max(ambiguity_index, _mean_numeric(pair_rows, "ambiguity_budget_prior", ambiguity_index)), 6),
                "ambiguity_budget_prior_gap": round(_mean_numeric(pair_rows, "ambiguity_budget_prior_gap", 0.0), 6),
                "budget_prior_exceeds_raw": str(_bool(_any_text(pair_rows, "budget_prior_exceeds_raw", "false"), False)).lower(),
                "row_overrides": "",
            }
            out.append(row)
    out.sort(
        key=lambda row: (
            -_float(row["od_ambiguity_index"]),
            -_float(row["straight_line_km"]),
            _text(row["od_id"]),
        )
    )
    return out


def _balanced_selection(rows: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    ordered_bins = ["0-100 km", "100-250 km", "250-500 km", "500+ km"]

    def take_rows(pool: list[dict[str, Any]], count: int, seen_ids: set[str]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            grouped[_text(row.get("distance_bin"), "unknown")].append(row)
        taken: list[dict[str, Any]] = []
        while len(taken) < count:
            progress = False
            for distance_bin in ordered_bins:
                batch = grouped.get(distance_bin, [])
                while batch:
                    row = batch.pop(0)
                    row_id = _text(row.get("od_id"))
                    if row_id in seen_ids:
                        continue
                    taken.append(row)
                    seen_ids.add(row_id)
                    progress = True
                    break
                if len(taken) >= count:
                    break
            if not progress:
                break
        return taken

    seen_ids: set[str] = set()
    ambiguity_rows = [row for row in rows if _text(row.get("corpus_group")) == "ambiguity"]
    representative_rows = [row for row in rows if _text(row.get("corpus_group")) == "representative"]
    target_ambiguity = min(len(ambiguity_rows), target_count // 2)
    target_representative = min(len(representative_rows), target_count // 2)

    selected = take_rows(ambiguity_rows, target_ambiguity, seen_ids)
    selected.extend(take_rows(representative_rows, target_representative, seen_ids))
    if len(selected) < target_count:
        selected.extend(take_rows(rows, target_count - len(selected), seen_ids))
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _split_rows_into_shards(rows: list[dict[str, Any]], shard_count: int) -> list[list[dict[str, Any]]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive when shard emission is requested")
    if shard_count > len(rows):
        raise ValueError("shard_count cannot exceed the number of selected rows")
    base_size, remainder = divmod(len(rows), shard_count)
    shards: list[list[dict[str, Any]]] = []
    cursor = 0
    for shard_index in range(shard_count):
        shard_size = base_size + (1 if shard_index < remainder else 0)
        shards.append(rows[cursor : cursor + shard_size])
        cursor += shard_size
    return shards


def _write_shards(
    rows: list[dict[str, Any]],
    *,
    output_csv: Path,
    shard_count: int,
    shard_output_dir: Path,
    shard_manifest_json: Path,
) -> dict[str, Any]:
    shard_output_dir.mkdir(parents=True, exist_ok=True)
    shards = _split_rows_into_shards(rows, shard_count)
    shard_entries: list[dict[str, Any]] = []
    for shard_index, shard_rows in enumerate(shards, start=1):
        shard_path = shard_output_dir / f"{output_csv.stem}_s{shard_index:02d}of{shard_count:02d}.csv"
        _write_csv(shard_path, shard_rows)
        od_ids = [_text(row.get("od_id")) for row in shard_rows]
        shard_entries.append(
            {
                "shard_index": shard_index,
                "shard_count": shard_count,
                "path": str(shard_path),
                "row_count": len(shard_rows),
                "od_ids": od_ids,
                "start_od_id": od_ids[0] if od_ids else "",
                "end_od_id": od_ids[-1] if od_ids else "",
            }
        )
    manifest = {
        "composition_type": "corpus_shards",
        "master_output_csv": str(output_csv),
        "shard_output_dir": str(shard_output_dir),
        "shard_count": shard_count,
        "total_row_count": len(rows),
        "shards": shard_entries,
    }
    shard_manifest_json.parent.mkdir(parents=True, exist_ok=True)
    shard_manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_expanded_corpus(
    input_csv: Path,
    output_csv: Path,
    summary_json: Path,
    target_count: int,
    *,
    shard_count: int = 0,
    shard_output_dir: Path | None = None,
    shard_manifest_json: Path | None = None,
) -> dict[str, Any]:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    base_origins = _make_location_prototypes(source_rows, "origin")
    base_destinations = _make_location_prototypes(source_rows, "destination")
    origins = _augment_location_prototypes(base_origins, variant_count=ENDPOINT_VARIANT_COUNT)
    destinations = _augment_location_prototypes(base_destinations, variant_count=ENDPOINT_VARIANT_COUNT)
    candidates = _candidate_rows(origins, destinations, source_rows)
    selected = _balanced_selection(candidates, target_count)
    if len(selected) < target_count:
        raise RuntimeError(f"Unable to synthesize {target_count} distinct OD rows from {len(candidates)} candidates.")
    _write_csv(output_csv, selected)
    shard_manifest: dict[str, Any] | None = None
    if shard_count > 0:
        if shard_output_dir is None or shard_manifest_json is None:
            raise ValueError("shard_output_dir and shard_manifest_json are required when shard_count is positive")
        shard_manifest = _write_shards(
            selected,
            output_csv=output_csv,
            shard_count=shard_count,
            shard_output_dir=shard_output_dir,
            shard_manifest_json=shard_manifest_json,
        )
    elif shard_output_dir is not None or shard_manifest_json is not None:
        raise ValueError("shard_output_dir and shard_manifest_json require shard_count > 0")
    summary = {
        "source_csv": str(input_csv),
        "output_csv": str(output_csv),
        "target_count": int(target_count),
        "selected_count": len(selected),
        "origin_prototype_count": len(base_origins),
        "destination_prototype_count": len(base_destinations),
        "augmented_origin_count": len(origins),
        "augmented_destination_count": len(destinations),
        "candidate_count": len(candidates),
        "distance_bin_counts": dict(sorted((bin_name, sum(1 for row in selected if row["distance_bin"] == bin_name)) for bin_name in {row["distance_bin"] for row in selected})),
        "corpus_group_counts": dict(sorted((group_name, sum(1 for row in selected if row["corpus_group"] == group_name)) for group_name in {row["corpus_group"] for row in selected})),
    }
    if shard_manifest is not None:
        summary["shard_manifest_json"] = str(shard_manifest_json)
        summary["shard_count"] = int(shard_count)
        summary["shard_row_counts"] = [int(entry["row_count"]) for entry in shard_manifest["shards"]]
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand the checked-in thesis broad corpus into a larger OD benchmark.")
    parser.add_argument("--input-csv", default="data/eval/uk_od_corpus_thesis_broad.csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--target-count", type=int, default=120)
    parser.add_argument("--shard-count", type=int, default=0)
    parser.add_argument("--shard-output-dir", default=None)
    parser.add_argument("--shard-manifest-json", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_expanded_corpus(
        input_csv=Path(args.input_csv).resolve(),
        output_csv=Path(args.output_csv).resolve(),
        summary_json=Path(args.summary_json).resolve(),
        target_count=int(args.target_count),
        shard_count=int(args.shard_count),
        shard_output_dir=Path(args.shard_output_dir).resolve() if args.shard_output_dir else None,
        shard_manifest_json=Path(args.shard_manifest_json).resolve() if args.shard_manifest_json else None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
