from __future__ import annotations

from pathlib import Path
from typing import Any

from .run_store import artifact_dir_for_run


def _escape_pdf_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _report_lines(
    run_id: str,
    *,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    results: dict[str, Any],
) -> list[str]:
    request = manifest.get("request") if isinstance(manifest.get("request"), dict) else {}
    execution = manifest.get("execution") if isinstance(manifest.get("execution"), dict) else {}
    stochastic = execution.get("stochastic") if isinstance(execution.get("stochastic"), dict) else {}
    signature = manifest.get("signature") if isinstance(manifest.get("signature"), dict) else {}

    lines: list[str] = [
        "Freight Router Run Report",
        f"run_id: {run_id}",
        f"created_at: {manifest.get('created_at', 'n/a')}",
        "",
        "Run metadata",
        f"vehicle_type: {request.get('vehicle_type', 'n/a')}",
        f"scenario_mode: {request.get('scenario_mode', 'n/a')}",
        f"pair_count: {metadata.get('pair_count', execution.get('pair_count', 'n/a'))}",
        f"error_count: {metadata.get('error_count', execution.get('error_count', 'n/a'))}",
        f"duration_ms: {metadata.get('duration_ms', execution.get('duration_ms', 'n/a'))}",
        f"fallback_used: {metadata.get('fallback_used', execution.get('fallback_used', 'n/a'))}",
        f"optimization_mode: {execution.get('optimization_mode', 'expected_value')}",
        f"risk_aversion: {execution.get('risk_aversion', 1.0)}",
        (
            f"stochastic: enabled={stochastic.get('enabled', False)}, "
            f"sigma={stochastic.get('sigma', 0.08)}, samples={stochastic.get('samples', 25)}"
        ),
        "",
        "Selected/pareto route summary",
    ]

    result_rows = results.get("results")
    if not isinstance(result_rows, list):
        result_rows = []

    max_pairs = 10
    for idx, pair in enumerate(result_rows[:max_pairs]):
        pair_dict = pair if isinstance(pair, dict) else {}
        routes = pair_dict.get("routes")
        if not isinstance(routes, list) or not routes:
            lines.append(f"pair {idx}: no routes (error={pair_dict.get('error', 'n/a')})")
            continue
        route = routes[0] if isinstance(routes[0], dict) else {}
        metrics = route.get("metrics") if isinstance(route.get("metrics"), dict) else {}
        route_id = route.get("id", f"pair{idx}_route0")
        duration_s = _to_float(metrics.get("duration_s"))
        money = _to_float(metrics.get("monetary_cost"))
        co2 = _to_float(metrics.get("emissions_kg"))
        uncertainty = route.get("uncertainty")
        robust_suffix = ""
        if isinstance(uncertainty, dict) and "p95_duration_s" in uncertainty:
            robust_suffix = f", p95={_to_float(uncertainty.get('p95_duration_s')):.1f}s"
        lines.append(
            f"pair {idx}: {route_id} | duration={duration_s:.1f}s | cost={money:.2f} | co2={co2:.3f}kg{robust_suffix}"
        )

    if len(result_rows) > max_pairs:
        lines.append(f"... ({len(result_rows) - max_pairs} additional pairs omitted)")

    lines.extend(
        [
            "",
            "References",
            f"manifest_endpoint: {metadata.get('manifest_endpoint', f'/runs/{run_id}/manifest')}",
            f"artifacts_endpoint: {metadata.get('artifacts_endpoint', f'/runs/{run_id}/artifacts')}",
            f"signature_algorithm: {signature.get('algorithm', 'HMAC-SHA256')}",
            f"signature_prefix: {str(signature.get('signature', 'n/a'))[:24]}",
        ]
    )
    return lines


def _render_simple_pdf(lines: list[str]) -> bytes:
    rendered_lines = [line[:120] for line in lines]
    content_lines = ["BT", "/F1 11 Tf", "40 770 Td", "14 TL"]
    for i, line in enumerate(rendered_lines):
        escaped = _escape_pdf_text(line)
        if i == 0:
            content_lines.append(f"({escaped}) Tj")
        else:
            content_lines.append("T*")
            content_lines.append(f"({escaped}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("utf-8")

    objects: list[bytes] = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(
        b"5 0 obj << /Length "
        + str(len(stream)).encode("ascii")
        + b" >> stream\n"
        + stream
        + b"\nendstream endobj\n"
    )

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer << /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def write_report_pdf(
    run_id: str,
    *,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    results: dict[str, Any],
) -> Path:
    out_dir = artifact_dir_for_run(run_id)
    lines = _report_lines(run_id, manifest=manifest, metadata=metadata, results=results)
    pdf_bytes = _render_simple_pdf(lines)
    path = out_dir / "report.pdf"
    path.write_bytes(pdf_bytes)
    return path
