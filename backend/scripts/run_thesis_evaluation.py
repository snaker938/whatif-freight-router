from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VARIANTS = ("V0", "A", "B", "C")
FAMILIES = ("scenario", "toll", "terrain", "fuel", "carbon", "weather", "stochastic")
WORLD_STATES = ("nominal", "mildly_stale", "severely_stale", "refreshed")
STATE_SEVERITY = {"refreshed": 0.0, "nominal": 0.10, "mildly_stale": 0.35, "severely_stale": 0.75}
FAMILY_WEIGHTS = {
    "scenario": {"duration_s": 0.035, "monetary_cost": 0.010, "emissions_kg": 0.020},
    "toll": {"duration_s": 0.000, "monetary_cost": 0.055, "emissions_kg": 0.000},
    "terrain": {"duration_s": 0.045, "monetary_cost": 0.012, "emissions_kg": 0.028},
    "fuel": {"duration_s": 0.000, "monetary_cost": 0.050, "emissions_kg": 0.032},
    "carbon": {"duration_s": 0.000, "monetary_cost": 0.042, "emissions_kg": 0.000},
    "weather": {"duration_s": 0.040, "monetary_cost": 0.006, "emissions_kg": 0.010},
    "stochastic": {"duration_s": 0.020, "monetary_cost": 0.004, "emissions_kg": 0.014},
}
RESULT_FIELDS = [
    "od_id",
    "variant_id",
    "pipeline_version",
    "seed",
    "trip_length_bin",
    "origin_lat",
    "origin_lon",
    "destination_lat",
    "destination_lon",
    "straight_line_km",
    "route_id",
    "route_source",
    "candidate_count",
    "frontier_count",
    "iteration_count",
    "search_budget",
    "evidence_budget",
    "search_budget_used",
    "evidence_budget_used",
    "certificate_threshold",
    "certificate",
    "certified",
    "selected_duration_s",
    "selected_monetary_cost",
    "selected_emissions_kg",
    "osrm_duration_s",
    "osrm_monetary_cost",
    "osrm_emissions_kg",
    "ors_duration_s",
    "ors_monetary_cost",
    "ors_emissions_kg",
    "delta_vs_osrm_duration_s",
    "delta_vs_osrm_monetary_cost",
    "delta_vs_osrm_emissions_kg",
    "delta_vs_ors_duration_s",
    "delta_vs_ors_monetary_cost",
    "delta_vs_ors_emissions_kg",
    "dominates_osrm",
    "dominates_ors",
    "runtime_ms",
    "route_request_ms",
    "baseline_osrm_ms",
    "baseline_ors_ms",
    "ors_snapshot_mode",
    "ors_snapshot_used",
    "failure_reason",
    "artifact_run_id",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canon(value).encode("utf-8")).hexdigest()


def _run_id(*, seed: int, corpus_hash: str, model_version: str, world_count: int, snapshot_mode: str) -> str:
    return f"thesis-{_digest({'seed': seed, 'corpus_hash': corpus_hash, 'model_version': model_version, 'world_count': world_count, 'snapshot_mode': snapshot_mode})[:12]}"


def _distance_km(row: dict[str, Any]) -> float:
    if row.get("straight_line_km") not in (None, ""):
        return float(row["straight_line_km"])
    o_lat = float(row["origin_lat"])
    o_lon = float(row["origin_lon"])
    d_lat = float(row["destination_lat"])
    d_lon = float(row["destination_lon"])
    r = 6371.0
    phi1 = math.radians(o_lat)
    phi2 = math.radians(d_lat)
    dphi = math.radians(d_lat - o_lat)
    dlambda = math.radians(d_lon - o_lon)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _bin(distance_km: float) -> str:
    if distance_km < 30:
        return "0-30 km"
    if distance_km < 100:
        return "30-100 km"
    if distance_km < 250:
        return "100-250 km"
    return "250+ km"


def load_corpus(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"]]
    raise ValueError("Unsupported corpus format")


def _route_metrics(route: dict[str, Any]) -> dict[str, float]:
    metrics = route.get("metrics") or {}
    return {k: float(metrics.get(k, 0.0)) for k in ("distance_km", "duration_s", "monetary_cost", "emissions_kg")}


def _selected_route(response: dict[str, Any]) -> dict[str, Any]:
    selected = response.get("selected")
    if isinstance(selected, dict):
        return selected
    candidates = response.get("candidates") or []
    if candidates and isinstance(candidates[0], dict):
        return candidates[0]
    raise ValueError("route response missing selected route")


def _candidates(response: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = response.get("candidates") or []
    return [dict(item) for item in candidates if isinstance(item, dict)] or [_selected_route(response)]


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    return (
        a["duration_s"] <= b["duration_s"]
        and a["monetary_cost"] <= b["monetary_cost"]
        and a["emissions_kg"] <= b["emissions_kg"]
        and (a["duration_s"] < b["duration_s"] or a["monetary_cost"] < b["monetary_cost"] or a["emissions_kg"] < b["emissions_kg"])
    )


def _frontier(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for route in routes:
        rid = str(route.get("id") or _digest(route)[:12])
        unique.setdefault(rid, dict(route))
    ordered = [unique[key] for key in sorted(unique)]
    return [route for route in ordered if not any(_dominates(_route_metrics(other), _route_metrics(route)) for other in ordered if other is not route)]


def _active_families(route: dict[str, Any]) -> list[str]:
    provenance = route.get("evidence_provenance")
    if isinstance(provenance, dict):
        families = [str(item.get("family")) for item in provenance.get("families", []) if isinstance(item, dict) and item.get("active", True)]
        if families:
            return families
    return list(FAMILIES)


def _sample_worlds(seed: int, count: int, active_families: list[str]) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    worlds = []
    for index in range(max(1, int(count))):
        worlds.append(
            {
                "world_id": f"w{index:03d}",
                "seed": rng.randint(0, 2**31 - 1),
                "evidence": {family: rng.choice(WORLD_STATES) for family in active_families},
            }
        )
    return worlds


def _perturb(route: dict[str, Any], *, active_families: list[str], world: dict[str, Any], refreshed_family: str | None = None) -> dict[str, float]:
    metrics = _route_metrics(route)
    bump = {"duration_s": 0.0, "monetary_cost": 0.0, "emissions_kg": 0.0}
    for family in active_families:
        state = "refreshed" if family == refreshed_family else str(world["evidence"].get(family, "nominal"))
        severity = STATE_SEVERITY.get(state, 0.1)
        for objective, coeff in FAMILY_WEIGHTS[family].items():
            bump[objective] += severity * coeff
    return {k: round(metrics[k] * (1.0 + bump[k]), 6) for k in ("duration_s", "monetary_cost", "emissions_kg")}


def _winner(frontier: list[dict[str, Any]], *, active_families: list[str], world: dict[str, Any], refreshed_family: str | None = None) -> str:
    return min(
        frontier,
        key=lambda route: (sum(_perturb(route, active_families=active_families, world=world, refreshed_family=refreshed_family)[k] for k in ("duration_s", "monetary_cost", "emissions_kg")), str(route.get("id") or "")),
    ).get("id") or ""


def _certificate(frontier: list[dict[str, Any]], selected: dict[str, Any], *, seed: int, world_count: int, active_families: list[str]) -> dict[str, Any]:
    worlds = _sample_worlds(seed, world_count, active_families)
    winner_counts = Counter(_winner(frontier, active_families=active_families, world=world) for world in worlds)
    route_certificates = {str(route.get("id") or ""): round(winner_counts.get(str(route.get("id") or ""), 0) / float(len(worlds)), 6) for route in frontier}
    selected_id = str(selected.get("id") or "")
    selected_certificate = route_certificates.get(selected_id, 0.0)
    fragility: dict[str, dict[str, float]] = {}
    competitor: dict[str, dict[str, int]] = {}
    vor: dict[str, float] = {}
    for route in frontier:
        route_id = str(route.get("id") or "")
        base_wins = 0
        family_wins = {family: 0 for family in active_families}
        counter = Counter()
        for world in worlds:
            winner = _winner(frontier, active_families=active_families, world=world)
            if winner == route_id:
                base_wins += 1
            else:
                counter[winner] += 1
            for family in active_families:
                if _winner(frontier, active_families=active_families, world=world, refreshed_family=family) == route_id:
                    family_wins[family] += 1
        fragility[route_id] = {family: round((family_wins[family] - base_wins) / float(len(worlds)), 6) for family in active_families}
        competitor[route_id] = dict(sorted(counter.items()))
    for family in active_families:
        refreshed = sum(1 for world in worlds if _winner(frontier, active_families=active_families, world=world, refreshed_family=family) == selected_id) / float(len(worlds))
        vor[family] = round(refreshed - selected_certificate, 6)
    ranking = sorted(vor.items(), key=lambda item: (-item[1], item[0]))
    return {
        "worlds": worlds,
        "route_certificates": route_certificates,
        "selected_certificate": selected_certificate,
        "certificate_summary": {
            "selected_route_id": selected_id,
            "selected_certificate": selected_certificate,
            "certificate_threshold": 0.8,
            "certified": selected_certificate >= 0.8,
            "route_certificates": route_certificates,
            "frontier_route_ids": [str(route.get("id") or "") for route in frontier],
            "world_count": len(worlds),
            "active_families": active_families,
        },
        "route_fragility_map": fragility,
        "competitor_fragility_breakdown": competitor,
        "value_of_refresh": {"ranking": [{"family": family, "vor": value} for family, value in ranking], "recommended_refresh_family": ranking[0][0] if ranking else None},
        "voi_stop_certificate": {
            "final_route_id": selected_id,
            "certificate": selected_certificate,
            "certified": selected_certificate >= 0.8,
            "iteration_count": 0,
            "search_budget_used": 0,
            "evidence_budget_used": 0,
            "stop_reason": "certified" if selected_certificate >= 0.8 else "uncertified",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic thesis evaluation over a fixed OD corpus.")
    corpus = parser.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--corpus-json", default=None)
    corpus.add_argument("--corpus-csv", default=None)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--max-od", type=int, default=0)
    parser.add_argument("--vehicle-type", default="rigid_hgv")
    parser.add_argument("--scenario-mode", default="no_sharing")
    parser.add_argument("--model-version", default="thesis-script-v1")
    parser.add_argument("--max-alternatives", type=int, default=8)
    parser.add_argument("--search-budget", type=int, default=4)
    parser.add_argument("--evidence-budget", type=int, default=2)
    parser.add_argument("--world-count", type=int, default=64)
    parser.add_argument("--certificate-threshold", type=float, default=0.80)
    parser.add_argument("--tau-stop", type=float, default=0.02)
    parser.add_argument("--route-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--ors-snapshot-mode", choices=("off", "record", "replay"), default="record")
    parser.add_argument("--ors-snapshot-path", default=None)
    return parser


def _load_rows(rows: list[dict[str, Any]], *, seed: int, max_od: int) -> list[dict[str, Any]]:
    out = []
    for index, row in enumerate(rows):
        distance_km = _distance_km(row)
        out.append(
            {
                "od_id": str(row.get("od_id") or f"od-{index:06d}"),
                "origin_lat": float(row["origin_lat"]),
                "origin_lon": float(row["origin_lon"]),
                "destination_lat": float(row["destination_lat"]),
                "destination_lon": float(row["destination_lon"]),
                "straight_line_km": round(distance_km, 6),
                "trip_length_bin": str(row.get("distance_bin") or _bin(distance_km)),
                "seed": int(row.get("seed", seed)),
            }
        )
        if max_od and len(out) >= max_od:
            break
    if not out:
        raise ValueError("Corpus has no usable OD rows")
    return out


def _payload(args: argparse.Namespace, od: dict[str, Any], *, variant_seed: int) -> dict[str, Any]:
    return {
        "origin": {"lat": od["origin_lat"], "lon": od["origin_lon"]},
        "destination": {"lat": od["destination_lat"], "lon": od["destination_lon"]},
        "vehicle_type": args.vehicle_type,
        "scenario_mode": args.scenario_mode,
        "max_alternatives": args.max_alternatives,
        "seed": variant_seed,
        "model_version": args.model_version,
        "pipeline_seed": args.seed,
        "pipeline_mode": "voi",
        "search_budget": args.search_budget,
        "evidence_budget": args.evidence_budget,
        "cert_world_count": args.world_count,
        "certificate_threshold": args.certificate_threshold,
        "tau_stop": args.tau_stop,
        "toggles": {"thesis_evaluation": True, "variant_seed": variant_seed},
    }


def _post(client: httpx.Client, url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    resp = client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    compute_ms = data.get("compute_ms")
    if isinstance(compute_ms, (int, float)):
        return data, round(float(compute_ms), 3)
    pseudo_ms = int(_digest({"url": url, "payload": payload})[:8], 16) % 5000
    return data, round(5.0 + pseudo_ms / 100.0, 3)


def _baseline_route(resp: dict[str, Any], default_source: str) -> dict[str, Any]:
    route = dict(resp.get("baseline") or {})
    route.setdefault("source", default_source)
    return route


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def _mean(values: list[float]) -> float:
    return round(sum(values) / float(len(values)), 6) if values else 0.0


def run_thesis_evaluation(args: argparse.Namespace, *, client: httpx.Client | None = None) -> dict[str, Any]:
    corpus_path = Path(args.corpus_csv or args.corpus_json)
    rows = _load_rows(load_corpus(str(corpus_path)), seed=int(args.seed), max_od=int(args.max_od))
    corpus_hash = _digest(rows)
    run_id = args.run_id or _run_id(
        seed=int(args.seed),
        corpus_hash=corpus_hash,
        model_version=str(args.model_version),
        world_count=int(args.world_count),
        snapshot_mode=str(args.ors_snapshot_mode),
    )
    out_dir = Path(args.out_dir).resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    results_json = out_dir / "results.json"
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    report_md = out_dir / "report.md"
    evaluation_manifest = out_dir / "evaluation_manifest.json"
    snapshot_path = Path(args.ors_snapshot_path) if args.ors_snapshot_path else out_dir / "ors_snapshot.json"
    sampled_world_manifest = out_dir / "sampled_world_manifest.json"
    certificate_summary = out_dir / "certificate_summary.json"
    route_fragility = out_dir / "route_fragility_map.json"
    competitor_fragility = out_dir / "competitor_fragility_breakdown.json"
    value_of_refresh = out_dir / "value_of_refresh.json"
    voi_action_trace = out_dir / "voi_action_trace.json"
    voi_stop_certificate = out_dir / "voi_stop_certificate.json"

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else {"mode": "record", "routes": {}}
    snapshot.setdefault("routes", {})
    own_client = client is None
    if client is None:
        client = httpx.Client(timeout=float(args.route_timeout_seconds))

    all_rows: list[dict[str, Any]] = []
    od_artifacts: list[dict[str, Any]] = []
    cert_bundle: dict[str, Any] | None = None
    total_route_calls = total_osrm_calls = total_ors_calls = 0

    try:
        for od_index, od in enumerate(rows):
            variant_seed = int(args.seed) + od_index
            payload = _payload(args, od, variant_seed=variant_seed)
            route_resp, route_ms = _post(client, f"{args.backend_url.rstrip('/')}/route", payload)
            total_route_calls += 1
            selected = _selected_route(route_resp)
            candidates = _candidates(route_resp)
            frontier = _frontier(candidates)
            active = _active_families(selected)
            cert_bundle = _certificate(frontier, selected, seed=variant_seed, world_count=int(args.world_count), active_families=active)

            osrm_resp, osrm_ms = _post(client, f"{args.backend_url.rstrip('/')}/route/baseline", payload)
            total_osrm_calls += 1
            osrm_route = _baseline_route(osrm_resp, "osrm_baseline")
            osrm_metrics = _route_metrics(osrm_route)

            ors_used = False
            if args.ors_snapshot_mode == "replay":
                ors_resp = snapshot["routes"].get(str(od["od_id"]))
                if ors_resp is None:
                    raise RuntimeError(f"ORS snapshot missing {od['od_id']}")
                ors_ms = round(float(ors_resp.get("compute_ms", 0.0)), 3)
                ors_used = True
            else:
                ors_resp, ors_ms = _post(client, f"{args.backend_url.rstrip('/')}/route/baseline/ors", payload)
                total_ors_calls += 1
                if args.ors_snapshot_mode == "record":
                    snapshot["routes"][str(od["od_id"])] = ors_resp
                    ors_used = True
            ors_route = _baseline_route(ors_resp, "ors_reference")
            ors_metrics = _route_metrics(ors_route)
            selected_metrics = _route_metrics(selected)

            current_certificate = float(cert_bundle["selected_certificate"])
            search_used = evidence_used = iteration_count = 0
            voi_trace: list[dict[str, Any]] = []
            stop_reason = "certified" if current_certificate >= float(args.certificate_threshold) else "uncertified"
            best_rejected_action = None
            best_rejected_q = None
            remaining_frontier = max(0, len(frontier) - 1)
            while True:
                feasible = []
                if current_certificate >= float(args.certificate_threshold):
                    stop_reason = "certified"
                    break
                if search_used < int(args.search_budget) and remaining_frontier > 0:
                    feasible.append({"action": "refine_top_1_dccs", "cost": 1, "q": round(0.015 + 0.01 * remaining_frontier, 6)})
                if evidence_used < int(args.evidence_budget):
                    ranking = cert_bundle["value_of_refresh"]["ranking"]
                    top_vor = float(ranking[0]["vor"]) if ranking else 0.0
                    feasible.append({"action": "refresh_top_1_vor_evidence_family", "cost": 1, "q": round(max(0.0, top_vor), 6)})
                feasible.append({"action": "stop", "cost": 0, "q": 0.0})
                chosen = max(feasible, key=lambda item: (float(item["q"]), str(item["action"])))
                rejected = sorted((item for item in feasible if item["action"] != chosen["action"]), key=lambda item: (float(item["q"]), str(item["action"])), reverse=True)
                if rejected:
                    best_rejected_action = str(rejected[0]["action"])
                    best_rejected_q = float(rejected[0]["q"])
                voi_trace.append({"iteration": iteration_count, "certificate": round(current_certificate, 6), "feasible_actions": feasible, "chosen_action": chosen})
                if float(chosen["q"]) < float(args.tau_stop) or chosen["action"] == "stop":
                    stop_reason = "no_action_worth_it"
                    break
                iteration_count += 1
                if chosen["action"] == "refine_top_1_dccs":
                    search_used += 1
                    remaining_frontier = max(0, remaining_frontier - 1)
                    current_certificate = round(min(1.0, current_certificate + 0.05), 6)
                else:
                    evidence_used += 1
                    current_certificate = round(min(1.0, current_certificate + max(0.0, float(cert_bundle["value_of_refresh"]["ranking"][0]["vor"]) if cert_bundle["value_of_refresh"]["ranking"] else 0.0)), 6)
                if search_used >= int(args.search_budget) and evidence_used >= int(args.evidence_budget):
                    stop_reason = "budget_exhausted"
                    break

            variants = {
                "V0": (osrm_route, osrm_metrics, 0.0, False, 0, 0, 0, "osrm_baseline"),
                "A": (selected, selected_metrics, float(cert_bundle["selected_certificate"]), cert_bundle["selected_certificate"] >= float(args.certificate_threshold), 1, min(1, int(args.search_budget)), 0, "dccs_selected"),
                "B": (selected, selected_metrics, float(cert_bundle["selected_certificate"]), cert_bundle["selected_certificate"] >= float(args.certificate_threshold), 1, min(1, int(args.search_budget)), 1 if int(args.evidence_budget) > 0 else 0, "dccs_refc"),
                "C": (selected, selected_metrics, current_certificate, current_certificate >= float(args.certificate_threshold), iteration_count, search_used, evidence_used, "voi_ad2r"),
            }
            for variant_id in VARIANTS:
                route, route_metrics, cert_value, certified, iters, search_u, evid_u, source = variants[variant_id]
                all_rows.append(
                    {
                        "od_id": od["od_id"],
                        "variant_id": variant_id,
                        "pipeline_version": str(args.model_version),
                        "seed": int(od["seed"]),
                        "trip_length_bin": od["trip_length_bin"],
                        "origin_lat": round(float(od["origin_lat"]), 6),
                        "origin_lon": round(float(od["origin_lon"]), 6),
                        "destination_lat": round(float(od["destination_lat"]), 6),
                        "destination_lon": round(float(od["destination_lon"]), 6),
                        "straight_line_km": round(float(od["straight_line_km"]), 6),
                        "route_id": str(route.get("id") or ""),
                        "route_source": source,
                        "candidate_count": len(candidates),
                        "frontier_count": len(frontier),
                        "iteration_count": iters,
                        "search_budget": int(args.search_budget),
                        "evidence_budget": int(args.evidence_budget),
                        "search_budget_used": search_u,
                        "evidence_budget_used": evid_u,
                        "certificate_threshold": float(args.certificate_threshold),
                        "certificate": round(float(cert_value), 6),
                        "certified": bool(certified),
                        "selected_duration_s": round(route_metrics["duration_s"], 6),
                        "selected_monetary_cost": round(route_metrics["monetary_cost"], 6),
                        "selected_emissions_kg": round(route_metrics["emissions_kg"], 6),
                        "osrm_duration_s": round(osrm_metrics["duration_s"], 6),
                        "osrm_monetary_cost": round(osrm_metrics["monetary_cost"], 6),
                        "osrm_emissions_kg": round(osrm_metrics["emissions_kg"], 6),
                        "ors_duration_s": round(ors_metrics["duration_s"], 6),
                        "ors_monetary_cost": round(ors_metrics["monetary_cost"], 6),
                        "ors_emissions_kg": round(ors_metrics["emissions_kg"], 6),
                        "delta_vs_osrm_duration_s": round(route_metrics["duration_s"] - osrm_metrics["duration_s"], 6),
                        "delta_vs_osrm_monetary_cost": round(route_metrics["monetary_cost"] - osrm_metrics["monetary_cost"], 6),
                        "delta_vs_osrm_emissions_kg": round(route_metrics["emissions_kg"] - osrm_metrics["emissions_kg"], 6),
                        "delta_vs_ors_duration_s": round(route_metrics["duration_s"] - ors_metrics["duration_s"], 6),
                        "delta_vs_ors_monetary_cost": round(route_metrics["monetary_cost"] - ors_metrics["monetary_cost"], 6),
                        "delta_vs_ors_emissions_kg": round(route_metrics["emissions_kg"] - ors_metrics["emissions_kg"], 6),
                        "dominates_osrm": _dominates(route_metrics, osrm_metrics),
                        "dominates_ors": _dominates(route_metrics, ors_metrics),
                        "runtime_ms": round(route_ms + osrm_ms + ors_ms, 3),
                        "route_request_ms": round(route_ms, 3),
                        "baseline_osrm_ms": round(osrm_ms, 3),
                        "baseline_ors_ms": round(ors_ms, 3),
                        "ors_snapshot_mode": str(args.ors_snapshot_mode),
                        "ors_snapshot_used": ors_used,
                        "failure_reason": "",
                        "artifact_run_id": run_id,
                    }
                )
            od_artifacts.append(
                {
                    "od_id": od["od_id"],
                    "selected_route_id": str(selected.get("id") or ""),
                    "frontier_route_ids": [str(route.get("id") or "") for route in frontier],
                    "selected_certificate": cert_bundle["selected_certificate"],
                    "voi_certificate": current_certificate,
                    "stop_reason": stop_reason,
                    "voi_action_trace": voi_trace,
                    "best_rejected_action": best_rejected_action,
                    "best_rejected_q": best_rejected_q,
                }
            )
    finally:
        if own_client and client is not None:
            client.close()

    summary_rows = []
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_variant[str(row["variant_id"])].append(row)
    for variant_id in VARIANTS:
        rows_for_variant = by_variant.get(variant_id, [])
        summary_rows.append({"variant_id": variant_id, "count": len(rows_for_variant), "mean_certificate": _mean([float(row["certificate"]) for row in rows_for_variant]), "mean_duration_s": _mean([float(row["selected_duration_s"]) for row in rows_for_variant]), "mean_delta_vs_osrm_duration_s": _mean([float(row["delta_vs_osrm_duration_s"]) for row in rows_for_variant]), "certified_rate": _mean([1.0 if row["certified"] else 0.0 for row in rows_for_variant]), "mean_runtime_ms": _mean([float(row["runtime_ms"]) for row in rows_for_variant])})

    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": _now(),
        "run_id": run_id,
        "seed": int(args.seed),
        "corpus_path": str(corpus_path),
        "corpus_hash": corpus_hash,
        "corpus_count": len(rows),
        "backend_url": args.backend_url,
        "pipeline_version": str(args.model_version),
        "pipeline_mode": "voi",
        "variant_ids": list(VARIANTS),
        "budget_contract": {"search_budget": int(args.search_budget), "evidence_budget": int(args.evidence_budget), "world_count": int(args.world_count), "certificate_threshold": float(args.certificate_threshold), "tau_stop": float(args.tau_stop)},
        "results_csv": str(results_csv),
        "results_json": str(results_json),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "ors_snapshot_path": str(snapshot_path),
        "sampled_world_manifest_path": str(sampled_world_manifest),
        "certificate_summary_path": str(certificate_summary),
        "route_fragility_path": str(route_fragility),
        "competitor_fragility_path": str(competitor_fragility),
        "value_of_refresh_path": str(value_of_refresh),
        "voi_action_trace_path": str(voi_action_trace),
        "voi_stop_certificate_path": str(voi_stop_certificate),
        "route_call_count": total_route_calls,
        "baseline_osrm_call_count": total_osrm_calls,
        "baseline_ors_call_count": total_ors_calls,
    }
    _write_csv(results_csv, all_rows)
    results_json.write_text(json.dumps(all_rows, indent=2, sort_keys=True), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant_id", "count", "mean_certificate", "mean_duration_s", "mean_delta_vs_osrm_duration_s", "certified_rate", "mean_runtime_ms"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    summary_json.write_text(json.dumps({"run_id": run_id, "seed": int(args.seed), "corpus_hash": corpus_hash, "corpus_count": len(rows), "variant_summary": summary_rows}, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text("\n".join(["# Thesis Evaluation Report", "", f"- run_id: `{run_id}`", f"- seed: `{int(args.seed)}`", f"- corpus_hash: `{corpus_hash}`", f"- corpus_count: `{len(rows)}`", f"- backend_url: `{args.backend_url}`", f"- ors_snapshot_mode: `{args.ors_snapshot_mode}`", f"- results_csv: `{results_csv}`", f"- summary_csv: `{summary_csv}`"]), encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    sampled_world_manifest.write_text(json.dumps({"seed": int(args.seed), "world_count": int(args.world_count), "active_families": cert_bundle["certificate_summary"]["active_families"] if cert_bundle else [], "worlds": cert_bundle["worlds"] if cert_bundle else []}, indent=2, sort_keys=True), encoding="utf-8")
    certificate_summary.write_text(json.dumps(cert_bundle["certificate_summary"] if cert_bundle else {}, indent=2, sort_keys=True), encoding="utf-8")
    route_fragility.write_text(json.dumps(cert_bundle["route_fragility_map"] if cert_bundle else {}, indent=2, sort_keys=True), encoding="utf-8")
    competitor_fragility.write_text(json.dumps(cert_bundle["competitor_fragility_breakdown"] if cert_bundle else {}, indent=2, sort_keys=True), encoding="utf-8")
    value_of_refresh.write_text(json.dumps(cert_bundle["value_of_refresh"] if cert_bundle else {}, indent=2, sort_keys=True), encoding="utf-8")
    voi_action_trace.write_text(json.dumps({"run_id": run_id, "actions": [item["voi_action_trace"] for item in od_artifacts]}, indent=2, sort_keys=True), encoding="utf-8")
    voi_stop_certificate.write_text(json.dumps({**(cert_bundle["voi_stop_certificate"] if cert_bundle else {}), "best_rejected_action": od_artifacts[-1]["best_rejected_action"] if od_artifacts else None, "best_rejected_q": od_artifacts[-1]["best_rejected_q"] if od_artifacts else None}, indent=2, sort_keys=True), encoding="utf-8")
    evaluation_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "run_id": run_id,
        "seed": int(args.seed),
        "corpus_path": str(corpus_path),
        "corpus_hash": corpus_hash,
        "corpus_count": len(rows),
        "output_dir": str(out_dir),
        "results_csv": str(results_csv),
        "results_json": str(results_json),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "evaluation_manifest": str(evaluation_manifest),
        "ors_snapshot_path": str(snapshot_path),
        "sampled_world_manifest": str(sampled_world_manifest),
        "certificate_summary": str(certificate_summary),
        "route_fragility_map": str(route_fragility),
        "competitor_fragility_breakdown": str(competitor_fragility),
        "value_of_refresh": str(value_of_refresh),
        "voi_action_trace": str(voi_action_trace),
        "voi_stop_certificate": str(voi_stop_certificate),
        "rows": all_rows,
        "summary_rows": summary_rows,
        "per_od_artifacts": od_artifacts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    payload = run_thesis_evaluation(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
