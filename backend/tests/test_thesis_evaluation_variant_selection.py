from __future__ import annotations

import pytest

import scripts.run_thesis_evaluation as thesis_module


pytestmark = pytest.mark.thesis_results


def _args_for_variants(raw_variants: str | None = None):
    argv = [
        "--corpus-csv",
        "dummy.csv",
        "--backend-url",
        "http://backend.test",
    ]
    if raw_variants is not None:
        argv.extend(["--variants", raw_variants])
    return thesis_module._build_parser().parse_args(argv)


def test_selected_variant_specs_defaults_to_full_suite_order() -> None:
    args = _args_for_variants()

    specs = thesis_module._selected_variant_specs(args)

    assert [spec.variant_id for spec in specs] == ["A", "B", "C", "V0"]
    assert [spec.pipeline_mode for spec in specs] == ["dccs", "dccs_refc", "voi", "legacy"]


def test_selected_variant_specs_filters_requested_subset_in_canonical_order() -> None:
    args = _args_for_variants("B,A,V0")

    specs = thesis_module._selected_variant_specs(args)

    assert [spec.variant_id for spec in specs] == ["A", "B", "V0"]
    assert [spec.pipeline_mode for spec in specs] == ["dccs", "dccs_refc", "legacy"]


def test_selected_variant_specs_rejects_unknown_variants() -> None:
    args = _args_for_variants("A,Z")

    with pytest.raises(ValueError, match=r"unknown_variants:Z"):
        thesis_module._selected_variant_specs(args)
