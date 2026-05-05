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


def test_theorem_and_claim_docs_are_semantically_consistent() -> None:
    module = _load_check_docs_module()

    assert module.run_theorem_claim_consistency_check() == []


def test_theorem_map_inventory_covers_required_family_ids() -> None:
    module = _load_check_docs_module()

    theorem_rows = module.load_theorem_map_rows()

    assert module.REQUIRED_THEOREM_IDS.issubset(theorem_rows)


def test_theorem_mentions_fail_when_theorem_id_is_unknown(tmp_path: Path) -> None:
    module = _load_check_docs_module()

    note = tmp_path / "note.md"
    note.write_text("## Note\nThis section depends on THM-99.\n", encoding="utf-8")

    errors = module.run_theorem_mention_consistency_check(
        module.load_theorem_map_rows(),
        markdown_files=[note],
    )

    assert any("THM-99" in error for error in errors)


def test_theorem_mentions_fail_when_mapped_family_lacks_required_fields(tmp_path: Path) -> None:
    module = _load_check_docs_module()

    note = tmp_path / "note.md"
    note.write_text("## Note\nTheorem: Safe elimination\n", encoding="utf-8")

    errors = module.run_theorem_mention_consistency_check(
        {
            "THM-01": {
                "ID": "THM-01",
                "Family": "Safe elimination",
                "Unit test anchor": "backend/tests/test_dccs.py::test_candidate_ledger_is_stable_and_auditable",
                "Negative / property test anchor": "",
                "Artifact field(s)": "dccs_candidates.jsonl.safe_eliminated",
                "Evaluator metric(s)": "",
                "Report appendix location": "Appendix N",
            }
        },
        markdown_files=[note],
    )

    assert any("Negative / property test anchor" in error for error in errors)
    assert any("Evaluator metric(s)" in error for error in errors)


def test_theorem_anchor_validation_fails_when_constructive_or_counterexample_anchor_missing() -> None:
    module = _load_check_docs_module()

    rows = {
        "THM-01": {
            "ID": "THM-01",
            "Family": "Safe elimination",
            "Unit test anchor": "backend/tests/test_docs_check.py::test_check_docs_passes_repo_consistency_checks",
            "Negative / property test anchor": "backend/tests/test_docs_check.py::test_theorem_and_claim_docs_are_semantically_consistent",
            "Constructive exact-synthetic example anchor": "",
            "Counterexample / assumption-failure example anchor": "",
            "Artifact field(s)": "dccs_candidates.jsonl.safe_eliminated",
            "Evaluator metric(s)": "mean_dccs_false_safe_prune_rate",
            "Report appendix location": "Appendix N",
        }
    }

    errors = module.validate_theorem_anchor_references(rows)

    assert any("Constructive exact-synthetic example anchor" in error for error in errors)
    assert any("Counterexample / assumption-failure example anchor" in error for error in errors)


def test_theorem_anchor_validation_fails_when_symbol_is_missing() -> None:
    module = _load_check_docs_module()

    rows = {
        "THM-01": {
            "ID": "THM-01",
            "Family": "Safe elimination",
            "Unit test anchor": "backend/tests/test_docs_check.py::test_check_docs_passes_repo_consistency_checks",
            "Negative / property test anchor": "backend/tests/test_docs_check.py::test_theorem_and_claim_docs_are_semantically_consistent",
            "Constructive exact-synthetic example anchor": (
                "Constructive exact-synthetic: backend/tests/test_docs_check.py::missing_constructive_anchor"
            ),
            "Counterexample / assumption-failure example anchor": (
                "Counterexample: backend/tests/test_docs_check.py::test_theorem_mentions_fail_when_theorem_id_is_unknown"
            ),
            "Artifact field(s)": "dccs_candidates.jsonl.safe_eliminated",
            "Evaluator metric(s)": "mean_dccs_false_safe_prune_rate",
            "Report appendix location": "Appendix N",
        }
    }

    errors = module.validate_theorem_anchor_references(rows)

    assert any("missing python symbol" in error for error in errors)
    assert any("missing_constructive_anchor" in error for error in errors)
