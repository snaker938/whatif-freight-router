from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from app.reporting import write_report_pdf


def _default_manifest_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "manifests" / f"{run_id}.json"


def _default_results_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "artifacts" / run_id / "results.json"


def _default_metadata_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "artifacts" / run_id / "metadata.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"required JSON file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON file: {path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def run_generate_report(args: argparse.Namespace) -> dict[str, Any]:
    run_id = str(args.run_id).strip()
    if not run_id:
        raise ValueError("run_id must not be empty")

    out_dir = Path(args.out_dir).resolve()
    manifest_path = (
        Path(args.manifest).resolve() if args.manifest else _default_manifest_path(out_dir, run_id)
    )
    results_path = (
        Path(args.results).resolve() if args.results else _default_results_path(out_dir, run_id)
    )
    metadata_path = (
        Path(args.metadata).resolve() if args.metadata else _default_metadata_path(out_dir, run_id)
    )

    manifest = _load_json(manifest_path)
    results = _load_json(results_path)
    metadata = _load_json(metadata_path)

    generated = write_report_pdf(run_id, manifest=manifest, metadata=metadata, results=results)
    output_path = Path(args.output).resolve() if args.output else generated.resolve()
    if output_path != generated.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(generated.read_bytes())

    return {
        "run_id": run_id,
        "manifest": str(manifest_path),
        "results": str(results_path),
        "metadata": str(metadata_path),
        "report_pdf": str(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report.pdf for an existing run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = run_generate_report(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
