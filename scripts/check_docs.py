#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
AGENT_OPS_DIR = DOCS_DIR / "agent-ops"
ROOT_README = ROOT / "README.md"
BACKEND_README = ROOT / "backend" / "README.md"
FRONTEND_README = ROOT / "frontend" / "README.md"
ROOT_CLAIM_MATRIX = ROOT / "claim_matrix.md"
DOCS_CLAIM_MATRIX = DOCS_DIR / "claim_matrix.md"
THEOREM_MAP = DOCS_DIR / "theorem_map.md"
DOC_INDEX = DOCS_DIR / "DOCS_INDEX.md"
BACKEND_MAIN = ROOT / "backend" / "app" / "main.py"
BACKEND_DOC = DOCS_DIR / "backend-api-tools.md"
RUN_STORE = ROOT / "backend" / "app" / "run_store.py"
BASENAME_SEARCH_DIRS = (
    ROOT,
    ROOT / "scripts",
    ROOT / "backend" / "app",
    ROOT / "backend" / "scripts",
    ROOT / "backend" / "tests",
)
THESIS_BUNDLE_NAMES = {
    "baseline_smoke_summary.json",
    "campaign_report.md",
    "cohort_composition.json",
    "compare_r12_vs_r15_combo_summary.json",
    "dashboard.csv",
    "evaluation_manifest.json",
    "index.json",
    "initial_certificate_summary.json",
    "initial_competitor_fragility_breakdown.json",
    "initial_route_fragility_map.json",
    "initial_sampled_world_manifest.json",
    "initial_value_of_refresh.json",
    "inventory.csv",
    "inventory.json",
    "manifest.json",
    "methods_appendix.md",
    "od_corpus.csv",
    "od_corpus.json",
    "od_corpus_rejected.json",
    "od_corpus_summary.json",
    "refresh_manifest.json",
    "repo_asset_preflight.json",
    "replay_oracle_dashboard.csv",
    "replay_oracle_summary.json",
    "results_summary.csv",
    "summary.json",
    "thesis_metrics.json",
    "thesis_plots.json",
    "thesis_report.md",
    "thesis_results.csv",
    "thesis_summary.csv",
    "thesis_summary.json",
    "thesis_summary_by_cohort.json",
    "winner_summary.json",
}

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
THEOREM_ID_RE = re.compile(r"\b(?:THM|LB)-\d{2}\b")
PYTHON_ANCHOR_RE = re.compile(r"([A-Za-z0-9_./\\-]+\.py)((?:::[A-Za-z_][A-Za-z0-9_]*)+)")
EXTRA_MAINTAINED_MARKDOWN = (
    BACKEND_README,
    FRONTEND_README,
    ROOT_CLAIM_MATRIX,
)
THEOREM_TABLE_HEADING = "## Required Theorem / Proposition Families"
LOWER_BOUND_TABLE_HEADING = "## Required Lower-Bound / Impossibility Families"
ROOT_CLAIMS_HEADING = "## Headline Claims"
DETAIL_CLAIMS_HEADING = "## Current Certification Slice"
THEOREM_TEST_ANCHOR_COLUMNS = (
    "Unit test anchor",
    "Negative / property test anchor",
    "Constructive exact-synthetic example anchor",
    "Counterexample / assumption-failure example anchor",
)
LOWER_BOUND_TEST_ANCHOR_COLUMNS = (
    "Unit test anchor",
    "Negative / property test anchor",
)
COMMON_THEOREM_TABLE_COLUMNS = {
    "ID",
    "Family",
    "Current status",
    "Assumptions / scope",
    "Code objects",
    "Unit test anchor",
    "Negative / property test anchor",
    "Artifact field(s)",
    "Evaluator metric(s)",
    "Report appendix location",
    "Current gap",
}
REQUIRED_THEOREM_TABLE_COLUMNS = COMMON_THEOREM_TABLE_COLUMNS | set(
    THEOREM_TEST_ANCHOR_COLUMNS[2:]
)
REQUIRED_LOWER_BOUND_TABLE_COLUMNS = set(COMMON_THEOREM_TABLE_COLUMNS)
REQUIRED_CLAIM_TABLE_COLUMNS = {
    "Claim status",
    "Theorem / proposition id",
    "Evaluator metric(s)",
    "Artifact path(s)",
}
COMMON_THEOREM_MENTION_REQUIRED_COLUMNS = (
    "Artifact field(s)",
    "Evaluator metric(s)",
    "Report appendix location",
)
REQUIRED_THEOREM_IDS = {
    *(f"THM-{index:02d}" for index in range(1, 11)),
    *(f"LB-{index:02d}" for index in range(1, 5)),
}
ALLOWED_THEOREM_STATUSES = {
    "theorem-backed",
    "partial-proof",
    "empirical-surface",
    "scaffold-only",
}
ALLOWED_CLAIM_STATUSES = {
    "theorem-backed",
    "empirical",
    "heuristic-but-measured",
    "non-claim / descriptive only",
}
NON_THEOREM_REFERENCE_MARKERS = {"not-applicable", "none-published-in-this-slice"}
PROOF_REQUIRED_COLUMNS = {
    "Assumptions / scope",
    "Code objects",
    "Artifact field(s)",
    "Evaluator metric(s)",
    "Report appendix location",
}
PLACEHOLDER_MARKERS = (
    "not yet mapped",
    "not yet assigned",
    "not yet pinned",
    "not yet published",
    "still open",
)
THEOREMISH_NAME_RE = re.compile(
    r"(?im)^(?:[-*]\s*)?(?:Theorem|Proposition|Lower[- ]bound|Impossibility)(?:\s+family)?\s*:\s*([^\n]+)$"
)


def list_agent_ops_docs() -> list[Path]:
    if not AGENT_OPS_DIR.is_dir():
        return []
    return sorted(p for p in AGENT_OPS_DIR.glob("*.md") if p.is_file())


def list_docs() -> list[Path]:
    files = [*DOCS_DIR.glob("*.md"), *list_agent_ops_docs()]
    unique_files: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        unique_files.append(path)
    return sorted(unique_files, key=lambda item: item.as_posix())


def list_maintained_markdown() -> list[Path]:
    files = [ROOT_README, *EXTRA_MAINTAINED_MARKDOWN, *list_docs()]
    unique_files: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        unique_files.append(path)
    return unique_files


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def format_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_md_cell(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        value = value[1:-1]
    return value.strip()


def split_markdown_table_line(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    if not stripped:
        return []
    return [normalize_md_cell(cell) for cell in stripped.split("|")]


def extract_markdown_table(path: Path, heading: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = read_text(path).splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start_index = index + 1
            break
    if start_index is None:
        return [], []

    table_lines: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("|"):
            table_lines.append(line)
            continue
        if table_lines and stripped:
            break
    if len(table_lines) < 2:
        return [], []

    headers = split_markdown_table_line(table_lines[0])
    rows: list[dict[str, str]] = []
    for raw_line in table_lines[2:]:
        cells = split_markdown_table_line(raw_line)
        if not cells:
            continue
        if len(cells) != len(headers):
            cells.extend([""] * (len(headers) - len(cells)))
        rows.append(dict(zip(headers, cells)))
    return headers, rows


def load_theorem_map_rows() -> dict[str, dict[str, str]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    for heading in (THEOREM_TABLE_HEADING, LOWER_BOUND_TABLE_HEADING):
        _, rows = extract_markdown_table(THEOREM_MAP, heading)
        for row in rows:
            row_id = normalize_md_cell(row.get("ID", ""))
            if row_id:
                rows_by_id[row_id] = row
    return rows_by_id


def theorem_anchor_columns_for_row(row_id: str) -> tuple[str, ...]:
    if normalize_md_cell(row_id).startswith("THM-"):
        return THEOREM_TEST_ANCHOR_COLUMNS
    return LOWER_BOUND_TEST_ANCHOR_COLUMNS


def theorem_required_columns_for_heading(heading: str) -> set[str]:
    if heading == THEOREM_TABLE_HEADING:
        return REQUIRED_THEOREM_TABLE_COLUMNS
    return REQUIRED_LOWER_BOUND_TABLE_COLUMNS


def theorem_proof_required_columns_for_row(row_id: str) -> set[str]:
    return PROOF_REQUIRED_COLUMNS | set(theorem_anchor_columns_for_row(row_id))


def theorem_row_missing_mention_fields(row_id: str, row: dict[str, str]) -> list[str]:
    missing: list[str] = []
    required_columns = theorem_anchor_columns_for_row(row_id) + COMMON_THEOREM_MENTION_REQUIRED_COLUMNS
    for column in required_columns:
        value = normalize_md_cell(row.get(column, ""))
        if not value or contains_placeholder(value):
            missing.append(column)
    return missing


def theorem_family_index(rows_by_id: dict[str, dict[str, str]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for row_id, row in rows_by_id.items():
        family = normalize_md_cell(row.get("Family", ""))
        if family:
            index[family.casefold()] = row_id
    return index


def _mention_resolution_error(
    *,
    doc_path: Path,
    mention_label: str,
    theorem_id: str,
    row: dict[str, str],
) -> str | None:
    missing_columns = theorem_row_missing_mention_fields(theorem_id, row)
    if not missing_columns:
        return None
    return (
        f"{format_path(doc_path)} mentions {mention_label}, but theorem-map row {theorem_id} "
        f"lacks required mapping fields: {', '.join(missing_columns)}"
    )


def run_theorem_mention_consistency_check(
    rows_by_id: dict[str, dict[str, str]],
    *,
    markdown_files: list[Path] | None = None,
) -> list[str]:
    errors: list[str] = []
    files = markdown_files if markdown_files is not None else list_maintained_markdown()
    excluded = {
        THEOREM_MAP.resolve(),
        ROOT_CLAIM_MATRIX.resolve(),
        DOCS_CLAIM_MATRIX.resolve(),
    }
    family_index = theorem_family_index(rows_by_id)

    for md_file in files:
        resolved_file = md_file.resolve()
        if resolved_file in excluded:
            continue
        text = read_text(md_file)

        for theorem_id in sorted(set(THEOREM_ID_RE.findall(text))):
            theorem_row = rows_by_id.get(theorem_id)
            if theorem_row is None:
                errors.append(
                    f"{format_path(md_file)} mentions unknown theorem/proposition id: {theorem_id}"
                )
                continue
            error = _mention_resolution_error(
                doc_path=md_file,
                mention_label=f"theorem/proposition id {theorem_id}",
                theorem_id=theorem_id,
                row=theorem_row,
            )
            if error is not None:
                errors.append(error)

        for family_name, theorem_id in family_index.items():
            family_pattern = re.compile(re.escape(family_name), re.IGNORECASE)
            if family_pattern.search(text) is None:
                continue
            theorem_row = rows_by_id[theorem_id]
            error = _mention_resolution_error(
                doc_path=md_file,
                mention_label=f"theorem/proposition family '{normalize_md_cell(theorem_row.get('Family', ''))}'",
                theorem_id=theorem_id,
                row=theorem_row,
            )
            if error is not None:
                errors.append(error)

        for raw_name in THEOREMISH_NAME_RE.findall(text):
            normalized_name = normalize_md_cell(raw_name)
            if not normalized_name or extract_theorem_ids(normalized_name):
                continue
            theorem_id = family_index.get(normalized_name.casefold())
            if theorem_id is None:
                errors.append(
                    f"{format_path(md_file)} mentions theorem/proposition family '{normalized_name}' without a theorem-map row"
                )
                continue
            theorem_row = rows_by_id[theorem_id]
            error = _mention_resolution_error(
                doc_path=md_file,
                mention_label=f"theorem/proposition family '{normalized_name}'",
                theorem_id=theorem_id,
                row=theorem_row,
            )
            if error is not None:
                errors.append(error)

    return errors


def contains_placeholder(value: str) -> bool:
    normalized = normalize_md_cell(value).lower()
    return any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def extract_theorem_ids(value: str) -> list[str]:
    normalized = normalize_md_cell(value)
    if not normalized or normalized in NON_THEOREM_REFERENCE_MARKERS:
        return []
    return THEOREM_ID_RE.findall(normalized)


def extract_python_anchor_references(value: str) -> list[tuple[str, tuple[str, ...]]]:
    references: list[tuple[str, tuple[str, ...]]] = []
    for path_text, symbol_text in PYTHON_ANCHOR_RE.findall(value or ""):
        symbols = tuple(part for part in symbol_text.split("::") if part)
        if symbols:
            references.append((normalize_candidate(path_text), symbols))
    return references


def _find_python_symbol_in_nodes(nodes: list[ast.stmt], symbols: tuple[str, ...]) -> bool:
    current_nodes = nodes
    for index, symbol in enumerate(symbols):
        match = next(
            (
                node
                for node in current_nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == symbol
            ),
            None,
        )
        if match is None:
            return False
        if index == len(symbols) - 1:
            return True
        if not isinstance(match, ast.ClassDef):
            return False
        current_nodes = list(match.body)
    return False


def validate_theorem_anchor_references(
    rows_by_id: dict[str, dict[str, str]],
    *,
    theorem_map_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    source_path = theorem_map_path or THEOREM_MAP
    parsed_modules: dict[Path, ast.Module] = {}

    for row_id, row in rows_by_id.items():
        for column in theorem_anchor_columns_for_row(row_id):
            cell_value = normalize_md_cell(row.get(column, ""))
            if not cell_value:
                errors.append(
                    f"{format_path(source_path)} row {row_id} missing required cell for column: {column}"
                )
                continue
            references = extract_python_anchor_references(cell_value)
            if not references:
                errors.append(
                    f"{format_path(source_path)} row {row_id} column {column} "
                    "must include at least one anchored python reference (path.py::symbol)"
                )
                continue
            for path_text, symbols in references:
                resolved = resolve_path_token(path_text, source_path)
                if resolved is None or not resolved.exists():
                    errors.append(
                        f"{format_path(source_path)} row {row_id} column {column} "
                        f"references missing python path: {path_text}::{'::'.join(symbols)}"
                    )
                    continue
                if resolved.suffix != ".py":
                    errors.append(
                        f"{format_path(source_path)} row {row_id} column {column} "
                        f"references non-python anchor target: {path_text}::{'::'.join(symbols)}"
                    )
                    continue
                module = parsed_modules.get(resolved)
                if module is None:
                    try:
                        module = ast.parse(resolved.read_text(encoding="utf-8"))
                    except (OSError, SyntaxError) as exc:
                        errors.append(
                            f"{format_path(source_path)} row {row_id} column {column} "
                            f"could not parse python anchor target {format_path(resolved)}: {exc}"
                        )
                        continue
                    parsed_modules[resolved] = module
                if not _find_python_symbol_in_nodes(list(module.body), symbols):
                    errors.append(
                        f"{format_path(source_path)} row {row_id} column {column} "
                        f"references missing python symbol: {format_path(resolved)}::{'::'.join(symbols)}"
                    )
    return errors


def load_declared_artifact_names() -> set[str]:
    if not RUN_STORE.exists():
        return set()
    module = ast.parse(RUN_STORE.read_text(encoding="utf-8"))
    for node in module.body:
        value: ast.AST | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "ARTIFACT_FILES":
                value = node.value
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "ARTIFACT_FILES" for target in node.targets):
                value = node.value
        if value is None:
            continue
        parsed = ast.literal_eval(value)
        if isinstance(parsed, tuple | list):
            return {str(item) for item in parsed}
    return set()


GENERATED_ARTIFACT_NAMES = load_declared_artifact_names() | THESIS_BUNDLE_NAMES


def strip_anchor(link: str) -> str:
    if "#" in link:
        return link.split("#", 1)[0]
    return link


def normalize_candidate(value: str) -> str:
    value = value.strip().strip("`\"'.,:;()[]")
    value = value.replace("\\", "/")
    return value


def resolve_repo_basename(token: str) -> Path | None:
    if "/" in token:
        return None
    if token.lower() == "readme.md":
        return ROOT_README.resolve()
    for directory in BASENAME_SEARCH_DIRS:
        candidate = directory / token
        if candidate.exists():
            return candidate.resolve()
    return None


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
    files = list_maintained_markdown()
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

    agent_ops_root = AGENT_OPS_DIR.resolve()

    index_text = read_text(DOC_INDEX)
    linked_docs: set[Path] = set()
    for link in MD_LINK_RE.findall(index_text):
        resolved = resolve_doc_link(DOC_INDEX, link)
        if not resolved:
            continue
        resolved = resolved.resolve()
        if resolved.suffix.lower() == ".md" and resolved.is_file():
            linked_docs.add(resolved)

    docs_in_index = {
        p.resolve()
        for p in docs
        if p.name.lower() not in {"readme.md", DOC_INDEX.name.lower()}
        and not p.resolve().is_relative_to(agent_ops_root)
    }
    expected_extra_links = {path.resolve() for path in EXTRA_MAINTAINED_MARKDOWN if path.is_file()}
    missing_from_index = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (docs_in_index | expected_extra_links) - linked_docs
    )
    for path_text in missing_from_index:
        errors.append(f"{path_text} is not linked from docs/DOCS_INDEX.md")

    for doc in docs:
        if doc.resolve().is_relative_to(agent_ops_root):
            continue
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
    if "..." in token:
        return None
    if token.startswith("http://") or token.startswith("https://"):
        return None

    as_posix = token.replace("\\", "/")
    if as_posix.startswith("/"):
        return (ROOT / as_posix.lstrip("/")).resolve()
    if as_posix.startswith("./") or as_posix.startswith("../"):
        return (source_file.parent / as_posix).resolve()
    basename_match = resolve_repo_basename(as_posix)
    if basename_match is not None:
        return basename_match
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
    files = list_maintained_markdown()
    seen: set[tuple[Path, str]] = set()

    for md_file in files:
        text = read_text(md_file)
        candidates = set(PY_CMD_PATH_RE.findall(text))
        candidates.update(PS1_CMD_PATH_RE.findall(text))
        candidates.update(CLI_FILE_ARG_RE.findall(text))
        candidates.update(CODE_FILE_LITERAL_RE.findall(text))

        for token in candidates:
            token_norm = normalize_candidate(token)
            if "..." in token_norm:
                continue
            if Path(token_norm.replace("\\", "/")).name in {
                "pairs.csv",
                "eta_observations.csv",
                "oracle_quality_dashboard.csv",
            }:
                continue
            basename = Path(token_norm.replace("\\", "/")).name
            if "/" not in token_norm.replace("\\", "/") and basename in GENERATED_ARTIFACT_NAMES:
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
    files = list_maintained_markdown()
    for md_file in files:
        text = read_text(md_file)
        if FORBIDDEN_NOTEBOOK_RE.search(text):
            errors.append(f"{md_file.relative_to(ROOT)} contains forbidden notebook reference")
    return errors


def run_theorem_claim_consistency_check() -> list[str]:
    errors: list[str] = []

    theorem_headers, theorem_rows = extract_markdown_table(THEOREM_MAP, THEOREM_TABLE_HEADING)
    if not theorem_headers:
        return [f"{THEOREM_MAP.relative_to(ROOT)} missing '{THEOREM_TABLE_HEADING}' table"]
    lower_headers, lower_rows = extract_markdown_table(THEOREM_MAP, LOWER_BOUND_TABLE_HEADING)
    if not lower_headers:
        errors.append(f"{THEOREM_MAP.relative_to(ROOT)} missing '{LOWER_BOUND_TABLE_HEADING}' table")
        return errors

    for heading, headers in (
        (THEOREM_TABLE_HEADING, theorem_headers),
        (LOWER_BOUND_TABLE_HEADING, lower_headers),
    ):
        missing_columns = sorted(theorem_required_columns_for_heading(heading) - set(headers))
        for column in missing_columns:
            errors.append(
                f"{THEOREM_MAP.relative_to(ROOT)} table '{heading}' missing required column: {column}"
            )

    rows_by_id: dict[str, dict[str, str]] = {}
    duplicate_ids: set[str] = set()
    for row in [*theorem_rows, *lower_rows]:
        row_id = normalize_md_cell(row.get("ID", ""))
        if not row_id:
            errors.append(f"{THEOREM_MAP.relative_to(ROOT)} has a theorem-map row with no ID")
            continue
        if not THEOREM_ID_RE.fullmatch(row_id):
            errors.append(f"{THEOREM_MAP.relative_to(ROOT)} has invalid theorem-map ID: {row_id}")
            continue
        if row_id in rows_by_id:
            duplicate_ids.add(row_id)
            continue
        rows_by_id[row_id] = row
        status = normalize_md_cell(row.get("Current status", ""))
        if status not in ALLOWED_THEOREM_STATUSES:
            errors.append(
                f"{THEOREM_MAP.relative_to(ROOT)} row {row_id} has invalid current status: {status or '<missing>'}"
            )
        required_columns = theorem_required_columns_for_heading(
            THEOREM_TABLE_HEADING if row_id.startswith("THM-") else LOWER_BOUND_TABLE_HEADING
        )
        for column in required_columns:
            if not normalize_md_cell(row.get(column, "")):
                errors.append(
                    f"{THEOREM_MAP.relative_to(ROOT)} row {row_id} missing required cell for column: {column}"
                )
        if status == "theorem-backed":
            for column in theorem_proof_required_columns_for_row(row_id):
                if contains_placeholder(row.get(column, "")):
                    errors.append(
                        f"{THEOREM_MAP.relative_to(ROOT)} row {row_id} is theorem-backed but still has placeholder text in {column}"
                    )
    for duplicate_id in sorted(duplicate_ids):
        errors.append(f"{THEOREM_MAP.relative_to(ROOT)} contains duplicate theorem-map ID: {duplicate_id}")

    missing_ids = sorted(REQUIRED_THEOREM_IDS - set(rows_by_id))
    for row_id in missing_ids:
        errors.append(f"{THEOREM_MAP.relative_to(ROOT)} missing required theorem-map row: {row_id}")

    errors.extend(validate_theorem_anchor_references(rows_by_id))

    for claim_path, heading in (
        (ROOT_CLAIM_MATRIX, ROOT_CLAIMS_HEADING),
        (DOCS_CLAIM_MATRIX, DETAIL_CLAIMS_HEADING),
    ):
        headers, rows = extract_markdown_table(claim_path, heading)
        if not headers:
            errors.append(f"{claim_path.relative_to(ROOT)} missing '{heading}' table")
            continue
        missing_columns = sorted(REQUIRED_CLAIM_TABLE_COLUMNS - set(headers))
        for column in missing_columns:
            errors.append(
                f"{claim_path.relative_to(ROOT)} table '{heading}' missing required column: {column}"
            )
        for row_index, row in enumerate(rows, start=1):
            claim_status = normalize_md_cell(row.get("Claim status", ""))
            theorem_ref = normalize_md_cell(row.get("Theorem / proposition id", ""))
            if claim_status not in ALLOWED_CLAIM_STATUSES:
                errors.append(
                    f"{claim_path.relative_to(ROOT)} row {row_index} has invalid claim status: {claim_status or '<missing>'}"
                )
            theorem_ids = extract_theorem_ids(theorem_ref)
            if theorem_ref and theorem_ref not in NON_THEOREM_REFERENCE_MARKERS and not theorem_ids:
                errors.append(
                    f"{claim_path.relative_to(ROOT)} row {row_index} has unrecognized theorem/proposition id cell: {theorem_ref}"
                )
            for theorem_id in theorem_ids:
                theorem_row = rows_by_id.get(theorem_id)
                if theorem_row is None:
                    errors.append(
                        f"{claim_path.relative_to(ROOT)} row {row_index} references unknown theorem-map ID: {theorem_id}"
                    )
                    continue
                if claim_status == "theorem-backed":
                    theorem_status = normalize_md_cell(theorem_row.get("Current status", ""))
                    if theorem_status != "theorem-backed":
                        errors.append(
                            f"{claim_path.relative_to(ROOT)} row {row_index} is marked theorem-backed but theorem-map row {theorem_id} is {theorem_status or '<missing>'}"
                        )
            if claim_status == "theorem-backed" and theorem_ref in NON_THEOREM_REFERENCE_MARKERS:
                errors.append(
                    f"{claim_path.relative_to(ROOT)} row {row_index} is marked theorem-backed but does not name a theorem/proposition id"
                )

    errors.extend(run_theorem_mention_consistency_check(rows_by_id))
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
    parser.add_argument("--check-theorem-claims", action="store_true")
    args = parser.parse_args()

    selected = {
        "links": args.check_links,
        "orphans": args.check_orphans,
        "paths": args.check_paths,
        "endpoints": args.check_endpoints,
        "theorem_claims": args.check_theorem_claims,
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
    if selected["theorem_claims"]:
        total += print_result(
            "Theorem-map / claim-matrix semantic consistency",
            run_theorem_claim_consistency_check(),
        )

    total += print_result("Notebook reference ban", run_forbidden_notebook_check())
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
