from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_check_docs_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_docs.py"
    spec = importlib.util.spec_from_file_location("check_docs_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_docs_passes_repo_consistency_checks() -> None:
    module = _load_check_docs_module()

    assert module.run_link_check() == []
    assert module.run_orphan_check() == []
    assert module.run_path_check() == []
    assert module.run_endpoint_check() == []
    assert module.run_forbidden_notebook_check() == []


def test_check_docs_inventory_includes_maintained_non_docs_surfaces() -> None:
    module = _load_check_docs_module()

    maintained = {
        path.relative_to(module.ROOT).as_posix()
        for path in module.list_maintained_markdown()
    }

    assert {
        "backend/README.md",
        "frontend/README.md",
        "claim_matrix.md",
    }.issubset(maintained)


def test_docs_index_links_cover_maintained_non_docs_surfaces() -> None:
    module = _load_check_docs_module()

    index_text = module.read_text(module.DOC_INDEX)
    linked = set()
    for raw_link in module.MD_LINK_RE.findall(index_text):
        resolved = module.resolve_doc_link(module.DOC_INDEX, raw_link)
        if resolved is None or not resolved.exists():
            continue
        linked.add(resolved.resolve().relative_to(module.ROOT).as_posix())

    assert {
        "backend/README.md",
        "frontend/README.md",
        "claim_matrix.md",
    }.issubset(linked)
