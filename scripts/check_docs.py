#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
ROOT_README = ROOT / "README.md"
DOC_INDEX = DOCS_DIR / "DOCS_INDEX.md"
BACKEND_MAIN = ROOT / "backend" / "app" / "main.py"
BACKEND_DOC = DOCS_DIR / "backend-api-tools.md"

MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
APP_ROUTE_RE = re.compile(r'@app\.(get|post|put|delete|patch)\("([^"]+)"')
DOC_ENDPOINT_RE = re.compile(r"`(GET|POST|PUT|DELETE|PATCH)\s+(/[^`]*)`")
FORBIDDEN_NOTEBOOK_RE = re.compile(
    r"(ipynb|jupyter notebook|freight_router_cookbook)", re.IGNORECASE
)
RELATED_DOCS_HEADING_RE = re.compile(r"^##\s+Related Docs\s*$", re.IGNORECASE | re.MULTILINE)
PY_CMD_PATH_RE = re.compile(r"(?:uv run python|python)\s+([A-Za-z0-9_./\\-]+\.py)")
PS1_CMD_PATH_RE = re.compile(r"(\.[\\/]+scripts[\\/][A-Za-z0-9_.-]+\.ps1)")
CLI_FILE_ARG_RE = re.compile(
    r"(?:--input-json|--input-csv|--output-dir|--source|--graph|--out-file)\s+([A-Za-z0-9_./\\-]+\.(?:json|csv|yaml|yml|geojson|pbf))"
)
CODE_FILE_LITERAL_RE = re.compile(
    r"`([A-Za-z0-9_./\\-]+\.(?:py|ps1|json|csv|yaml|yml|geojson|md))`"
)


def list_docs() -> list[Path]:
    return sorted(p for p in DOCS_DIR.glob("*.md") if p.is_file())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_anchor(link: str) -> str:
    if "#" in link:
        return link.split("#", 1)[0]
    return link


def normalize_candidate(value: str) -> str:
    value = value.strip().strip("`\"'.,:;()[]")
    value = value.replace("\\", "/")
    return value


def resolve_doc_link(source_file: Path, link: str) -> Path | None:
    link = link.strip()
    if not link or link.startswith("#"):
        return None
    if re.match(r"^[a-zA-Z]+://", link):
        return None
    target = strip_anchor(link)
    if not target:
        return None
    if target.startswith("/"):
        return ROOT / target.lstrip("/")
    return (source_file.parent / target).resolve()


def section_after_heading(text: str, heading_re: re.Pattern[str]) -> str:
    match = heading_re.search(text)
    if not match:
        return ""
    start = match.end()
    remainder = text[start:]
    next_heading = re.search(r"^##\s+", remainder, flags=re.MULTILINE)
    if not next_heading:
        return remainder
    return remainder[: next_heading.start()]


def run_link_check() -> list[str]:
    errors: list[str] = []
    files = [ROOT_README] + list_docs()
    for md_file in files:
        text = read_text(md_file)
        for raw_link in MD_LINK_RE.findall(text):
            target = resolve_doc_link(md_file, raw_link)
            if target is None:
                continue
            if not target.exists():
                errors.append(f"{md_file.relative_to(ROOT)} -> missing link target: {raw_link}")
    return errors


def run_orphan_check() -> list[str]:
    errors: list[str] = []
    docs = list_docs()
    if not DOC_INDEX.exists():
        return ["docs/DOCS_INDEX.md is missing"]

    index_text = read_text(DOC_INDEX)
    linked_docs: set[str] = set()
    for link in MD_LINK_RE.findall(index_text):
        resolved = resolve_doc_link(DOC_INDEX, link)
        if not resolved:
            continue
        if resolved.suffix.lower() == ".md" and resolved.is_file() and resolved.parent == DOCS_DIR:
            linked_docs.add(resolved.name)

    all_doc_names = {
        p.name
        for p in docs
        if p.name.lower() not in {"readme.md", DOC_INDEX.name.lower()}
    }
    missing_from_index = sorted(all_doc_names - linked_docs)
    for name in missing_from_index:
        errors.append(f"docs/{name} is not linked from docs/DOCS_INDEX.md")

    for doc in docs:
        text = read_text(doc)
        if not RELATED_DOCS_HEADING_RE.search(text):
            errors.append(f"{doc.relative_to(ROOT)} missing '## Related Docs' section")
            continue
        related_section = section_after_heading(text, RELATED_DOCS_HEADING_RE)
        related_links = [x for x in MD_LINK_RE.findall(related_section) if ".md" in x]
        if len(related_links) < 2:
            errors.append(f"{doc.relative_to(ROOT)} needs at least 2 links in Related Docs")
    return errors


def resolve_path_token(token: str, source_file: Path) -> Path | None:
    token = normalize_candidate(token)
    if not token:
        return None
    if any(x in token for x in ["*", "<", ">", "{", "}"]):
        return None
    if token.startswith("http://") or token.startswith("https://"):
        return None

    as_posix = token.replace("\\", "/")
    if as_posix.startswith("/"):
        return (ROOT / as_posix.lstrip("/")).resolve()
    if as_posix.startswith("./") or as_posix.startswith("../"):
        return (source_file.parent / as_posix).resolve()
    if as_posix.endswith(".md") and "/" not in as_posix:
        # doc-local references like `backend-api-tools.md`
        return (source_file.parent / as_posix).resolve()
    if as_posix in {"clean.ps1", "dev.ps1", "demo_repro_run.ps1"}:
        return (ROOT / "scripts" / as_posix).resolve()
    if as_posix.startswith("scripts/") and as_posix.endswith(".py"):
        root_path = (ROOT / as_posix).resolve()
        if root_path.exists():
            return root_path
        return (ROOT / "backend" / as_posix).resolve()
    return (ROOT / as_posix).resolve()


def run_path_check() -> list[str]:
    errors: list[str] = []
    files = [ROOT_README] + list_docs()
    seen: set[tuple[Path, str]] = set()

    for md_file in files:
        text = read_text(md_file)
        candidates = set(PY_CMD_PATH_RE.findall(text))
        candidates.update(PS1_CMD_PATH_RE.findall(text))
        candidates.update(CLI_FILE_ARG_RE.findall(text))
        candidates.update(CODE_FILE_LITERAL_RE.findall(text))

        for token in candidates:
            token_norm = normalize_candidate(token)
            if Path(token_norm.replace("\\", "/")).name in {
                "pairs.csv",
                "eta_observations.csv",
                "oracle_quality_dashboard.csv",
            }:
                continue
            key = (md_file, token_norm)
            if key in seen:
                continue
            seen.add(key)
            resolved = resolve_path_token(token_norm, md_file)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(
                    f"{md_file.relative_to(ROOT)} references missing path: {token_norm}"
                )
    return errors


def run_endpoint_check() -> list[str]:
    errors: list[str] = []
    if not BACKEND_MAIN.exists():
        return ["backend/app/main.py is missing"]
    if not BACKEND_DOC.exists():
        return ["docs/backend-api-tools.md is missing"]

    code_text = read_text(BACKEND_MAIN)
    doc_text = read_text(BACKEND_DOC)

    code_endpoints = {
        f"{method.upper()} {path}"
        for method, path in APP_ROUTE_RE.findall(code_text)
    }
    doc_endpoints = {
        f"{method.upper()} {path}"
        for method, path in DOC_ENDPOINT_RE.findall(doc_text)
    }

    missing = sorted(code_endpoints - doc_endpoints)
    extra = sorted(doc_endpoints - code_endpoints)
    for item in missing:
        errors.append(f"Undocumented endpoint in docs/backend-api-tools.md: {item}")
    for item in extra:
        errors.append(f"Documented endpoint not found in backend/app/main.py: {item}")
    return errors


def run_forbidden_notebook_check() -> list[str]:
    errors: list[str] = []
    files = [ROOT_README] + list_docs()
    for md_file in files:
        text = read_text(md_file)
        if FORBIDDEN_NOTEBOOK_RE.search(text):
            errors.append(f"{md_file.relative_to(ROOT)} contains forbidden notebook reference")
    return errors


def print_result(title: str, errors: list[str]) -> int:
    if not errors:
        print(f"[PASS] {title}")
        return 0
    print(f"[FAIL] {title} ({len(errors)} issues)")
    for err in errors:
        print(f"  - {err}")
    return len(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Documentation consistency checks")
    parser.add_argument("--check-links", action="store_true")
    parser.add_argument("--check-orphans", action="store_true")
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--check-endpoints", action="store_true")
    args = parser.parse_args()

    selected = {
        "links": args.check_links,
        "orphans": args.check_orphans,
        "paths": args.check_paths,
        "endpoints": args.check_endpoints,
    }
    if not any(selected.values()):
        selected = {k: True for k in selected}

    total = 0
    if selected["links"]:
        total += print_result("Markdown link validity", run_link_check())
    if selected["orphans"]:
        total += print_result("Docs index/orphan/related-doc coverage", run_orphan_check())
    if selected["paths"]:
        total += print_result("Referenced local path existence", run_path_check())
    if selected["endpoints"]:
        total += print_result("Endpoint parity against backend/app/main.py", run_endpoint_check())

    total += print_result("Notebook reference ban", run_forbidden_notebook_check())
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
