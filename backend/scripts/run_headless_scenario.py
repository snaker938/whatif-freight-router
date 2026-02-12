from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import httpx


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run batch scenario headlessly and download manifest/artifacts."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-json", default=None)
    group.add_argument("--input-csv", default=None)
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--vehicle-type", default="rigid_hgv")
    parser.add_argument("--scenario-mode", default="no_sharing")
    parser.add_argument("--max-alternatives", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-version", default="headless-v1")
    parser.add_argument("--save-dir", default="out/headless")
    parser.add_argument("--summary-path", default=None)
    return parser


def load_payload_from_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    if "pairs" not in payload:
        raise ValueError("JSON payload must contain 'pairs'")
    return payload


def load_payload_from_csv(
    path: str,
    *,
    vehicle_type: str,
    scenario_mode: str,
    max_alternatives: int,
    seed: int | None,
    model_version: str,
) -> dict[str, Any]:
    pairs: list[dict[str, dict[str, float]]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"origin_lat", "origin_lon", "destination_lat", "destination_lon"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "CSV must include columns: origin_lat, origin_lon, destination_lat, destination_lon"
            )

        for row in reader:
            pairs.append(
                {
                    "origin": {
                        "lat": float(row["origin_lat"]),
                        "lon": float(row["origin_lon"]),
                    },
                    "destination": {
                        "lat": float(row["destination_lat"]),
                        "lon": float(row["destination_lon"]),
                    },
                }
            )

    if not pairs:
        raise ValueError("CSV input produced zero OD pairs")

    return {
        "pairs": pairs,
        "vehicle_type": vehicle_type,
        "scenario_mode": scenario_mode,
        "max_alternatives": max_alternatives,
        "seed": seed,
        "toggles": {"headless_mode": True},
        "model_version": model_version,
    }


def _download_file(client: httpx.Client, url: str, dest: Path) -> None:
    resp = client.get(url)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)


def execute_headless_run(
    payload: dict[str, Any],
    *,
    backend_url: str,
    save_dir: str,
    summary_path: str | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    base = backend_url.rstrip("/")
    own_client = client is None
    if client is None:
        client = httpx.Client(timeout=90.0)

    try:
        batch_resp = client.post(f"{base}/batch/pareto", json=payload)
        batch_resp.raise_for_status()
        batch_data = batch_resp.json()
        run_id = batch_data["run_id"]

        run_dir = Path(save_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest_url = f"{base}/runs/{run_id}/manifest"
        artifact_list_url = f"{base}/runs/{run_id}/artifacts"
        _download_file(client, manifest_url, run_dir / "manifest.json")

        list_resp = client.get(artifact_list_url)
        list_resp.raise_for_status()
        artifacts = list_resp.json()["artifacts"]

        downloaded: list[str] = []
        for item in artifacts:
            name = str(item["name"])
            _download_file(client, f"{base}/runs/{run_id}/artifacts/{name}", run_dir / name)
            downloaded.append(name)

        error_count = sum(1 for item in batch_data["results"] if item.get("error"))
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "pair_count": len(payload.get("pairs", [])),
            "error_count": error_count,
            "saved_dir": str(run_dir),
            "manifest_file": str(run_dir / "manifest.json"),
            "downloaded_artifacts": sorted(downloaded),
            "manifest_endpoint": f"/runs/{run_id}/manifest",
            "artifacts_endpoint": f"/runs/{run_id}/artifacts",
        }

        summary_file = (
            Path(summary_path)
            if summary_path
            else run_dir / f"headless_summary_{_utc_now_compact()}.json"
        )
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_file"] = str(summary_file)
        return summary
    finally:
        if own_client and client is not None:
            client.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.input_json:
        payload = load_payload_from_json(args.input_json)
    else:
        payload = load_payload_from_csv(
            args.input_csv,
            vehicle_type=args.vehicle_type,
            scenario_mode=args.scenario_mode,
            max_alternatives=args.max_alternatives,
            seed=args.seed,
            model_version=args.model_version,
        )

    summary = execute_headless_run(
        payload,
        backend_url=args.backend_url,
        save_dir=args.save_dir,
        summary_path=args.summary_path,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
