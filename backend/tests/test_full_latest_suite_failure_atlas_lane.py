from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.run_full_latest_suite as full_suite_module
import scripts.run_thesis_evaluation as thesis_module


def test_full_latest_suite_default_corpus_counts_match_row_gate_multiplicity() -> None:
    args = full_suite_module._build_parser().parse_args([])

    assert full_suite_module.PIPELINE_VARIANT_COUNT == 4
    assert args.use_curated_corpora is True
    assert int(args.broad_count) == 50
    assert int(args.focused_count) == 15
    assert int(args.transfer_count) == 13
    assert int(args.synthetic_count) == 250
    assert int(args.optional_stopping_count) == 7500
    assert full_suite_module.CURATED_CORPUS_MINIMUM_ROWS == {
        "broad": 50,
        "focused": 15,
        "transfer": 13,
        "synthetic": 250,
    }
    assert full_suite_module._lane_corpus_key("optional_stopping_coverage") == "optional_stopping"
    assert full_suite_module._corpus_key_for_role("optional_stopping_coverage") == "optional_stopping"

    generated_args = full_suite_module._build_parser().parse_args(["--generate-corpora"])
    assert generated_args.use_curated_corpora is False


def test_optional_stopping_corpus_uses_lane_specific_7500_od_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = full_suite_module._build_parser().parse_args([])
    captured: dict[str, int] = {}

    def _fake_build_dual_od_corpora(
        *,
        seed: int,
        representative_count: int,
        ambiguous_count: int,
        bbox: object,
        max_attempts: int,
        acceptance_mode: str,
        probe_max_paths: int,
    ) -> dict[str, dict[str, list[dict[str, str]]]]:
        captured["seed"] = seed
        captured["representative_count"] = representative_count
        captured["ambiguous_count"] = ambiguous_count
        captured["max_attempts"] = max_attempts
        captured["probe_max_paths"] = probe_max_paths
        return {
            "representative": {"rows": [{"od_id": f"od_{idx:05d}"} for idx in range(representative_count)]},
            "ambiguous": {"rows": []},
        }

    def _fake_write_json_artifact(run_id: str, name: str, payload: object) -> Path:  # noqa: ARG001
        return tmp_path / name

    def _fake_persist_corpus(
        *,
        run_id: str,
        artifact_prefix: str,
        label: str,
        rows: list[dict[str, object]],
        summary_payload: object,
        source_summary_path: str,
    ) -> full_suite_module.CorpusArtifact:
        return full_suite_module.CorpusArtifact(
            key=artifact_prefix.replace("latest_corpus_", ""),
            label=label,
            row_count=len(rows),
            csv_path=str(tmp_path / f"{artifact_prefix}.csv"),
            json_path=str(tmp_path / f"{artifact_prefix}.json"),
            summary_path=str(tmp_path / f"{artifact_prefix}.summary.json"),
            source_summary_path=source_summary_path,
        )

    monkeypatch.setattr(full_suite_module, "build_dual_od_corpora", _fake_build_dual_od_corpora)
    monkeypatch.setattr(full_suite_module, "write_json_artifact", _fake_write_json_artifact)
    monkeypatch.setattr(full_suite_module, "_persist_corpus", _fake_persist_corpus)
    monkeypatch.setattr(full_suite_module, "_resolve_max_attempts", lambda _args: 1)

    corpus = full_suite_module._build_optional_stopping_corpus(args, suite_run_id="suite_optional_stopping")

    assert captured["representative_count"] == 7500
    assert captured["ambiguous_count"] == 20
    assert captured["max_attempts"] == 180000
    assert captured["probe_max_paths"] == 6
    assert corpus.key == "optional_stopping"
    assert corpus.row_count == 7500
    assert corpus.label == "Optional-stopping latest corpus"


def test_generated_focused_corpus_includes_support_fragile_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = full_suite_module._build_parser().parse_args(["--generate-corpora"])
    captured: dict[str, list[dict[str, object]]] = {}

    def _fake_build_dual_od_corpora(
        *,
        seed: int,
        representative_count: int,
        ambiguous_count: int,
        bbox: object,
        max_attempts: int,
        acceptance_mode: str,
        probe_max_paths: int,
    ) -> dict[str, dict[str, list[dict[str, object]]]]:
        representative_rows = [
            {
                "od_id": f"rep_{idx:04d}",
                "od_ambiguity_prior_strength": "0.8",
                "od_ambiguity_support_ratio": "0.8",
                "od_ambiguity_source_entropy": "0.8",
                "od_ambiguity_source_support_strength": "0.8",
                "od_ambiguity_confidence": "0.8",
                "od_ambiguity_source_count": "4",
                "od_ambiguity_source_mix": "a,b,c",
                "od_ambiguity_source_mix_count": "4",
                "od_ambiguity_family_density": "0.8",
                "candidate_probe_path_count": "6",
                "candidate_probe_corridor_family_count": "4",
            }
            for idx in range(representative_count)
        ]
        ambiguous_rows = [
            {
                "od_id": f"amb_{idx:04d}",
                "od_ambiguity_prior_strength": "0.7",
                "od_ambiguity_support_ratio": "0.7",
                "od_ambiguity_source_entropy": "0.7",
                "od_ambiguity_source_support_strength": "0.7",
                "od_ambiguity_confidence": "0.7",
                "od_ambiguity_source_count": "3",
                "od_ambiguity_source_mix": "a,b",
                "od_ambiguity_source_mix_count": "3",
                "od_ambiguity_family_density": "0.7",
                "candidate_probe_path_count": "5",
                "candidate_probe_corridor_family_count": "3",
            }
            for idx in range(ambiguous_count)
        ]
        return {
            "representative": {"rows": representative_rows},
            "ambiguous": {"rows": ambiguous_rows},
        }

    def _fake_read_csv_rows(path: Path) -> list[dict[str, object]]:
        if path == full_suite_module.SUPPORT_FRAGILE_SOURCE_CSV:
            return [
                {
                    "od_id": "weak_0001",
                    "od_ambiguity_prior_strength": "0.05",
                    "od_ambiguity_support_ratio": "0.05",
                    "od_ambiguity_source_entropy": "0.05",
                    "od_ambiguity_source_support_strength": "0.05",
                    "od_ambiguity_confidence": "0.05",
                    "od_ambiguity_source_count": "1",
                    "od_ambiguity_source_mix": "a",
                    "od_ambiguity_source_mix_count": "1",
                    "od_ambiguity_family_density": "0.05",
                    "candidate_probe_path_count": "1",
                    "candidate_probe_corridor_family_count": "1",
                },
                {
                    "od_id": "weak_0002",
                    "od_ambiguity_prior_strength": "0.08",
                    "od_ambiguity_support_ratio": "0.08",
                    "od_ambiguity_source_entropy": "0.08",
                    "od_ambiguity_source_support_strength": "0.08",
                    "od_ambiguity_confidence": "0.08",
                    "od_ambiguity_source_count": "1",
                    "od_ambiguity_source_mix": "a",
                    "od_ambiguity_source_mix_count": "1",
                    "od_ambiguity_family_density": "0.08",
                    "candidate_probe_path_count": "1",
                    "candidate_probe_corridor_family_count": "1",
                },
            ]
        return []

    def _fake_write_json_artifact(run_id: str, name: str, payload: object) -> Path:  # noqa: ARG001
        return tmp_path / name

    def _fake_persist_corpus(
        *,
        run_id: str,
        artifact_prefix: str,
        label: str,
        rows: list[dict[str, object]],
        summary_payload: object,
        source_summary_path: str,
    ) -> full_suite_module.CorpusArtifact:
        captured[artifact_prefix] = [dict(row) for row in rows]
        return full_suite_module.CorpusArtifact(
            key=artifact_prefix.replace("latest_corpus_", ""),
            label=label,
            row_count=len(rows),
            csv_path=str(tmp_path / f"{artifact_prefix}.csv"),
            json_path=str(tmp_path / f"{artifact_prefix}.json"),
            summary_path=str(tmp_path / f"{artifact_prefix}.summary.json"),
            source_summary_path=source_summary_path,
        )

    monkeypatch.setattr(full_suite_module, "build_dual_od_corpora", _fake_build_dual_od_corpora)
    monkeypatch.setattr(full_suite_module, "_read_csv_rows", _fake_read_csv_rows)
    monkeypatch.setattr(full_suite_module, "write_json_artifact", _fake_write_json_artifact)
    monkeypatch.setattr(full_suite_module, "_persist_corpus", _fake_persist_corpus)
    monkeypatch.setattr(full_suite_module, "_resolve_max_attempts", lambda _args: 1)

    full_suite_module._build_generated_corpora(args, suite_run_id="suite_generated_focus")

    focused_rows = captured["latest_corpus_focused"]
    assert len(focused_rows) == int(args.focused_count)
    weak_rows = [row for row in focused_rows if str(row.get("corpus_group")) == "support_fragile"]
    assert {str(row.get("od_id")) for row in weak_rows} == {"weak_0001", "weak_0002"}
    assert all(float(row["support_richness"]) <= full_suite_module.SUPPORT_FRAGILE_THRESHOLD for row in weak_rows)
    assert all(str(row.get("support_bin")) == "weak_support" for row in weak_rows)


def test_build_corpora_rebuilds_curated_focused_corpus_when_support_fragile_rows_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(full_suite_module.settings, "out_dir", str(out_dir))
    monkeypatch.setattr(full_suite_module, "CURATED_BASE_POOL_CSV", tmp_path / "missing_curated_base_pool.csv")
    monkeypatch.setattr(
        full_suite_module,
        "_build_optional_stopping_corpus",
        lambda *args, **kwargs: full_suite_module.CorpusArtifact(
            key="optional_stopping",
            label="Optional-stopping latest corpus",
            row_count=full_suite_module.DEFAULT_OPTIONAL_STOPPING_COUNT,
            csv_path=str(tmp_path / "optional.csv"),
            json_path=str(tmp_path / "optional.json"),
            summary_path=str(tmp_path / "optional.summary.json"),
            source_summary_path=str(tmp_path / "optional.source.json"),
        ),
    )

    broad_csv = tmp_path / "broad.csv"
    focused_csv = tmp_path / "focused.csv"
    transfer_csv = tmp_path / "transfer.csv"
    synthetic_csv = tmp_path / "synthetic.csv"

    broad_rows = [
        {"od_id": f"broad_{idx:04d}", "distance_bin": "0-100 km", "corridor_bucket": "south_to_south"}
        for idx in range(full_suite_module.DEFAULT_BROAD_COUNT)
    ]
    focused_rows = [
        {
            "od_id": f"focused_{idx:04d}",
            "distance_bin": "0-100 km",
            "corridor_bucket": "south_to_south",
            "corpus_group": "ambiguity",
            "od_ambiguity_prior_strength": "0.8",
            "od_ambiguity_support_ratio": "0.8",
            "od_ambiguity_source_entropy": "0.8",
            "od_ambiguity_source_support_strength": "0.8",
            "od_ambiguity_confidence": "0.8",
            "od_ambiguity_source_count": "4",
            "od_ambiguity_source_mix": "a,b,c",
            "od_ambiguity_source_mix_count": "4",
        }
        for idx in range(full_suite_module.DEFAULT_FOCUSED_COUNT)
    ]
    transfer_rows = [
        {"od_id": f"transfer_{idx:04d}", "distance_bin": "0-100 km", "corridor_bucket": "south_to_south"}
        for idx in range(full_suite_module.DEFAULT_TRANSFER_COUNT)
    ]
    synthetic_rows = [
        {"od_id": f"synthetic_{idx:04d}", "distance_bin": "0-100 km", "corridor_bucket": "south_to_south"}
        for idx in range(full_suite_module.DEFAULT_SYNTHETIC_COUNT)
    ]

    for path, rows in (
        (broad_csv, broad_rows),
        (focused_csv, focused_rows),
        (transfer_csv, transfer_rows),
        (synthetic_csv, synthetic_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    support_source_rows = [
        {
            "od_id": "weak_1001",
            "distance_bin": "0-100 km",
            "corridor_bucket": "scotland_to_scotland",
            "od_ambiguity_index": "0.62",
            "od_ambiguity_prior_strength": "0.05",
            "od_ambiguity_support_ratio": "0.05",
            "od_ambiguity_source_entropy": "0.05",
            "od_ambiguity_source_support_strength": "0.05",
            "od_ambiguity_confidence": "0.05",
            "od_ambiguity_source_count": "1",
            "od_ambiguity_source_mix": "a",
            "od_ambiguity_source_mix_count": "1",
        },
        {
            "od_id": "weak_1002",
            "distance_bin": "0-100 km",
            "corridor_bucket": "midlands_to_midlands",
            "od_ambiguity_index": "0.58",
            "od_ambiguity_prior_strength": "0.08",
            "od_ambiguity_support_ratio": "0.08",
            "od_ambiguity_source_entropy": "0.08",
            "od_ambiguity_source_support_strength": "0.08",
            "od_ambiguity_confidence": "0.08",
            "od_ambiguity_source_count": "1",
            "od_ambiguity_source_mix": "a",
            "od_ambiguity_source_mix_count": "1",
        },
        {
            "od_id": "weak_1003",
            "distance_bin": "0-100 km",
            "corridor_bucket": "scotland_to_scotland",
            "od_ambiguity_index": "0.55",
            "od_ambiguity_prior_strength": "0.10",
            "od_ambiguity_support_ratio": "0.10",
            "od_ambiguity_source_entropy": "0.10",
            "od_ambiguity_source_support_strength": "0.10",
            "od_ambiguity_confidence": "0.10",
            "od_ambiguity_source_count": "1",
            "od_ambiguity_source_mix": "a",
            "od_ambiguity_source_mix_count": "1",
        },
    ]

    support_source_csv = tmp_path / "support_fragile.csv"
    with support_source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(support_source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(support_source_rows)
    monkeypatch.setattr(full_suite_module, "SUPPORT_FRAGILE_SOURCE_CSV", support_source_csv)

    args = full_suite_module._build_parser().parse_args(
        [
            "--use-curated-corpora",
            "--broad-corpus-csv",
            str(broad_csv),
                "--focused-corpus-csv",
                str(focused_csv),
                "--transfer-corpus-csv",
                str(transfer_csv),
                "--synthetic-corpus-csv",
                str(synthetic_csv),
            ]
        )

    corpora = full_suite_module._build_corpora(args, suite_run_id="suite_curated_focused_mix")
    focused_payload = json.loads(Path(corpora["focused"].summary_path).read_text(encoding="utf-8"))

    assert focused_payload["selection_policy"] == "reused_curated_corpus_plus_support_fragile_slice"
    assert focused_payload["selected_by_cohort"]["support_fragile"] == 3
    weak_rows = [row for row in focused_payload["rows"] if row.get("corpus_group") == "support_fragile"]
    assert {row["od_id"] for row in weak_rows} == {"weak_1001", "weak_1002", "weak_1003"}
    assert all(float(row["support_richness"]) <= full_suite_module.SUPPORT_FRAGILE_THRESHOLD for row in weak_rows)


def test_build_corpora_prefers_curated_base_pool_when_defaults_need_scaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_pool_csv = tmp_path / "uk_od_corpus_seq02_combined_1204.csv"
    fieldnames = [
        "od_id",
        "origin_lat",
        "origin_lon",
        "destination_lat",
        "destination_lon",
        "distance_bin",
        "corridor_bucket",
        "profile_id",
        "corpus_group",
        "weight_time",
        "weight_money",
        "weight_co2",
        "scenario_mode",
        "weather_profile",
        "weather_intensity",
        "departure_time_utc",
        "stochastic_enabled",
        "stochastic_samples",
        "search_budget",
        "evidence_budget",
        "world_count",
        "certificate_threshold",
        "tau_stop",
        "max_alternatives",
        "optimization_mode",
        "ambiguity_index",
    ]
    with base_pool_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(300):
            writer.writerow(
                {
                    "od_id": f"od_{idx:04d}",
                    "origin_lat": "51.5",
                    "origin_lon": "-0.1",
                    "destination_lat": "52.5",
                    "destination_lon": "-1.1",
                    "distance_bin": "100-250 km",
                    "corridor_bucket": f"south_to_midlands|slot_{idx % 5}",
                    "profile_id": f"profile_{idx:04d}",
                    "corpus_group": "representative" if idx % 3 else "ambiguity",
                    "weight_time": "1.00",
                    "weight_money": "1.00",
                    "weight_co2": "1.00",
                    "scenario_mode": "no_sharing",
                    "weather_profile": "clear",
                    "weather_intensity": "1.0",
                    "departure_time_utc": "2026-03-21T09:00:00Z",
                    "stochastic_enabled": "true",
                    "stochastic_samples": "32",
                    "search_budget": "4",
                    "evidence_budget": "2",
                    "world_count": "64",
                    "certificate_threshold": "0.80",
                    "tau_stop": "0.020",
                    "max_alternatives": "8",
                    "optimization_mode": "expected_value",
                    "ambiguity_index": "0.2",
                }
            )

    monkeypatch.setattr(full_suite_module, "CURATED_BASE_POOL_CSV", base_pool_csv)
    monkeypatch.setattr(full_suite_module.settings, "out_dir", str(tmp_path / "out"))

    args = full_suite_module._build_parser().parse_args([])
    corpora = full_suite_module._build_corpora(args, suite_run_id="suite_curated_base_pool")

    assert {key: corpora[key].row_count for key in corpora} == {
        "broad": 50,
        "focused": 15,
        "transfer": 13,
        "synthetic": 250,
    }
    assert "curated-base-pool" in corpora["broad"].label
    assert Path(corpora["broad"].summary_path).exists()


def test_run_full_latest_suite_emits_complete_failure_atlas_lane_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _stub_corpus(key: str, label: str) -> full_suite_module.CorpusArtifact:
        csv_path = tmp_path / f"{key}.csv"
        json_path = tmp_path / f"{key}.json"
        summary_path = tmp_path / f"{key}.summary.json"
        source_summary_path = tmp_path / f"{key}.source_summary.json"
        _write_text(csv_path, "od_id,origin_lat,origin_lon,destination_lat,destination_lon\n")
        _write_json(json_path, {"rows": []})
        _write_json(summary_path, {"row_count": 0, "label": label})
        _write_json(source_summary_path, {"source": label})
        return full_suite_module.CorpusArtifact(
            key=key,
            label=label,
            row_count=0,
            csv_path=str(csv_path),
            json_path=str(json_path),
            summary_path=str(summary_path),
            source_summary_path=str(source_summary_path),
        )

    corpora = {
        "broad": _stub_corpus("broad", "Broad"),
        "focused": _stub_corpus("focused", "Focused"),
        "transfer": _stub_corpus("transfer", "Transfer"),
        "synthetic": _stub_corpus("synthetic", "Synthetic"),
    }

    def _route_bundle_artifacts(
        artifact_run_id: str,
        *,
        set_size: int,
        support_flag: bool,
        support_status: str,
        support_bin: str,
        singleton_not_justified_reasons: list[str] | None = None,
        excluded_route_safety_reasons: list[str] | None = None,
        stop_reason: str | None = None,
        abstention_reason_code: str | None = None,
    ) -> None:
        full_suite_module.write_json_artifact(
            artifact_run_id,
            "decision_region_summary.json",
            {
                "active_challenger_id": "route_b",
                "dominant_evidence_family": "weather",
                "support_status": support_status,
                "support_bin": support_bin,
                "calibration_bin": "empirical",
                "selected_certificate_basis": "empirical",
                "route_fragility_family_count": 1,
                "root_cause_tags": ["boundary:demo"],
            },
        )
        full_suite_module.write_json_artifact(
            artifact_run_id,
            "certificate_witness.json",
            {
                "support_status": support_status,
                "support_bin": support_bin,
                "calibration_bin": "empirical",
                "selected_certificate_basis": "empirical",
                "targeted_challenger_route_id": "route_b",
                "root_cause_tags": ["witness:demo"],
                "support_flag": support_flag,
            },
        )
        full_suite_module.write_json_artifact(
            artifact_run_id,
            "certified_set_summary.json",
            {
                "set_size": set_size,
                "support_flag": support_flag,
                "artifact_provenance": {"support_status": support_status},
                "witness": {
                    "singleton_not_justified_reasons": singleton_not_justified_reasons or [],
                    "excluded_route_safety_reasons": excluded_route_safety_reasons or [],
                    "outside_routes_safely_excluded": not bool(excluded_route_safety_reasons),
                },
            },
        )
        decision_package = {
            "schema_version": "1.0.0",
            "terminal_type": (
                "typed_abstention" if abstention_reason_code else ("certified_set" if set_size > 1 else "certified_singleton")
            ),
            "abstention": (
                {
                    "reason_code": abstention_reason_code,
                    "reason": abstention_reason_code,
                    "support_flag": support_flag,
                }
                if abstention_reason_code
                else {}
            ),
            "abstention_summary": (
                {
                    "reason_code": abstention_reason_code,
                    "support_flag": support_flag,
                }
                if abstention_reason_code
                else {}
            ),
        }
        full_suite_module.write_json_artifact(
            artifact_run_id,
            "decision_package.json",
            decision_package,
        )
        full_suite_module.write_json_artifact(
            artifact_run_id,
            "voi_stop_certificate.json",
            {"stop_reason": stop_reason},
        )
        full_suite_module.write_json_artifact(
            artifact_run_id,
            "results.json",
            {"artifact_run_id": artifact_run_id},
        )

    def _fake_run_thesis_evaluation(args: object, *, client=None) -> dict[str, object]:  # noqa: ARG001
        run_id = str(getattr(args, "run_id"))
        role = str(getattr(args, "evaluation_suite_role"))
        rows: list[dict[str, object]] = []
        if role == "focused_refc_proof":
            overlap_artifact = f"{run_id}_wrong_singleton_overlap"
            _route_bundle_artifacts(
                overlap_artifact,
                set_size=2,
                support_flag=False,
                support_status="unsupported",
                support_bin="weak_support",
                singleton_not_justified_reasons=["frontier_pairwise_gap_unresolved"],
                excluded_route_safety_reasons=["excluded_route_pairwise_gap_unresolved"],
                stop_reason="support_gap",
            )
            rows.append(
                {
                    "variant_id": "A",
                    "od_id": "focused-wrong-singleton-overlap",
                    "profile_id": "p0",
                    "artifact_run_id": overlap_artifact,
                    "terminal_type": "open",
                    "corpus_group": "ambiguity",
                }
            )
        else:
            support_artifact = f"{run_id}_support_downgrade"
            _route_bundle_artifacts(
                support_artifact,
                set_size=1,
                support_flag=True,
                support_status="",
                support_bin="weak_support",
            )
            rows.append(
                {
                    "variant_id": "C",
                    "od_id": "focused-support-downgrade",
                    "profile_id": "p2",
                    "artifact_run_id": support_artifact,
                    "failure_reason": "support_gap",
                    "terminal_type": "open",
                    "support_flag": True,
                    "support_richness": 0.41,
                    "corpus_group": "support_fragile",
                }
            )
            for index in range(6):
                abstention_artifact = f"{run_id}_abstention_{index}"
                _route_bundle_artifacts(
                    abstention_artifact,
                    set_size=0,
                    support_flag=True,
                    support_status="supported",
                    support_bin="supported",
                    stop_reason="budget_exhausted",
                    abstention_reason_code="uncertified_due_to_budget",
                )
                rows.append(
                    {
                        "variant_id": "C",
                        "od_id": f"focused-abstention-{index}",
                        "profile_id": f"p_abst_{index}",
                        "artifact_run_id": abstention_artifact,
                        "preference_terminal_type": "abstained",
                        "corpus_group": "controller_stress",
                    }
                )
        return {
            "run_id": run_id,
            "summary_rows": [
                {
                    "variant_id": "A" if role == "focused_refc_proof" else "C",
                    "pipeline_mode": "refc" if role == "focused_refc_proof" else "voi",
                    "row_count": len(rows),
                }
            ],
            "rows": rows,
            "lane_metadata": {
                "evaluation_suite": {
                    "role": role,
                    "scope": "focused",
                }
            },
        }

    def _fake_run_hot_rerun(args: object, *, client=None) -> dict[str, object]:  # noqa: ARG001
        hot_run_id = str(getattr(args, "hot_run_id"))
        comparison_json = full_suite_module.write_json_artifact(
            hot_run_id,
            "hot_rerun_vs_cold_comparison.json",
            {"rows": []},
        )
        gate_json = full_suite_module.write_json_artifact(
            hot_run_id,
            "hot_rerun_gate.json",
            {"all_green": True},
        )
        report_path = full_suite_module.write_text_artifact(hot_run_id, "hot_rerun_report.md", "# hot rerun\n")
        return {
            "hot_run_id": hot_run_id,
            "comparison_json": str(comparison_json),
            "comparison_csv": str(comparison_json).replace(".json", ".csv"),
            "gate_json": str(gate_json),
            "report_path": str(report_path),
            "hot_gate": {"all_green": True},
        }

    monkeypatch.setattr(full_suite_module, "DIRECT_SUITE_ROLES", ("focused_refc_proof", "focused_voi_proof"))
    monkeypatch.setattr(full_suite_module, "run_preflight", lambda output_path: {"strict_route_ready": True})  # noqa: ARG005
    monkeypatch.setattr(full_suite_module, "_build_corpora", lambda args, *, suite_run_id: corpora)  # noqa: ARG005
    monkeypatch.setattr(full_suite_module, "run_thesis_evaluation", _fake_run_thesis_evaluation)
    monkeypatch.setattr(full_suite_module, "run_hot_rerun_benchmark", _fake_run_hot_rerun)
    monkeypatch.setattr(full_suite_module.httpx, "Client", lambda *args, **kwargs: DummyClient())

    args = full_suite_module._build_parser().parse_args(
        [
            "--run-id",
            "suite_failure_atlas_lane",
            "--out-dir",
            str(out_dir),
            "--live-backend",
        ]
    )

    payload = full_suite_module.run_full_latest_suite(args)

    lane_metadata = json.loads(Path(payload["failure_atlas_lane_metadata_json"]).read_text(encoding="utf-8"))
    atlas_payload = json.loads(Path(payload["failure_atlas_json"]).read_text(encoding="utf-8"))
    atlas_markdown = Path(payload["failure_atlas_md"]).read_text(encoding="utf-8")
    index_payload = json.loads(Path(payload["index_json"]).read_text(encoding="utf-8"))
    results_payload = json.loads(Path(payload["results_json"]).read_text(encoding="utf-8"))

    assert lane_metadata["lane_id"] == "failure_atlas"
    assert lane_metadata["lane_status"] == "present_complete"
    assert lane_metadata["required_kind_presence"] == {
        "wrong_singleton": True,
        "support_downgrade": True,
        "abstention": True,
    }
    assert lane_metadata["required_kind_counts"] == {
        "wrong_singleton": 1,
        "support_downgrade": 2,
        "abstention": 6,
    }
    assert lane_metadata["coverage_class_counts"]["certified_set_violation"] == 1
    assert lane_metadata["coverage_class_counts"]["route_failure"] == 1
    assert lane_metadata["certified_set_violation_case_count"] == 1
    assert lane_metadata["abstention_class_counts"] == {"uncertified_due_to_budget": 6}
    assert lane_metadata["root_cause_family_counts"] == {
        "budget_cut": 6,
        "hidden_challenger": 0,
        "other": 0,
        "preference_ambiguity": 0,
        "proxy_bias": 0,
        "support_failure": 2,
    }
    assert lane_metadata["abstention_class_example_target"] == 5
    assert lane_metadata["source_lane_roles"] == ["focused_refc_proof", "focused_voi_proof"]
    assert lane_metadata["artifact_paths"]["lane_audit"] == payload["failure_atlas_json"]
    assert lane_metadata["artifact_paths"]["lane_report"] == payload["failure_atlas_md"]

    assert atlas_payload["lane_id"] == "failure_atlas"
    assert atlas_payload["counts_by_kind"] == {
        "abstention": 6,
        "wrong_singleton": 1,
        "support_downgrade": 1,
        "certified_set_violation": 0,
        "route_failure": 0,
    }
    assert atlas_payload["coverage_class_counts"] == {
        "abstention": 6,
        "wrong_singleton": 1,
        "support_downgrade": 2,
        "certified_set_violation": 1,
        "route_failure": 1,
    }
    assert atlas_payload["coverage_inclusion_status"]["certified_set_violation"]["detected_count"] == 1
    assert atlas_payload["coverage_inclusion_status"]["certified_set_violation"]["included_count"] == 1
    assert atlas_payload["certified_set_violation_case_count"] == 1
    assert atlas_payload["abstention_class_counts"] == {"uncertified_due_to_budget": 6}
    assert atlas_payload["root_cause_family_counts"] == {
        "budget_cut": 6,
        "hidden_challenger": 0,
        "other": 0,
        "preference_ambiguity": 0,
        "proxy_bias": 0,
        "support_failure": 2,
    }
    assert atlas_payload["abstention_class_documentation"] == [
        {
            "abstention_class": "uncertified_due_to_budget",
            "available_count": 6,
            "documented_count": 5,
            "documentation_target": 5,
            "documentation_complete": True,
        }
    ]
    abstention_examples = atlas_payload["abstention_class_examples"]["uncertified_due_to_budget"]
    assert len(abstention_examples) == 5

    overlap_row = next(row for row in atlas_payload["rows"] if row["od_id"] == "focused-wrong-singleton-overlap")
    assert overlap_row["coverage_classes"] == [
        "wrong_singleton",
        "support_downgrade",
        "certified_set_violation",
    ]
    assert overlap_row["row_id"] == "focused_refc_proof::A::focused-wrong-singleton-overlap::p0"
    assert overlap_row["cohort"] == "ambiguity"
    assert overlap_row["support_status"] == "unsupported"


def test_build_corpora_falls_back_to_fresh_generation_when_curated_corpora_are_short(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = full_suite_module._build_parser().parse_args(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--use-curated-corpora",
        ]
    )

    curated_counts = {
        "broad": 120,
        "focused": 40,
        "transfer": 20,
        "synthetic": 900,
    }

    def _short_curated_corpus(*, suite_run_id: str, artifact_prefix: str, label: str, csv_path: Path) -> full_suite_module.CorpusArtifact:  # noqa: ARG001
        key = artifact_prefix.replace("latest_corpus_", "")
        return full_suite_module.CorpusArtifact(
            key=key,
            label=label,
            row_count=curated_counts[key],
            csv_path=str(csv_path),
            json_path=str(tmp_path / f"{artifact_prefix}.json"),
            summary_path=str(tmp_path / f"{artifact_prefix}.summary.json"),
            source_summary_path=str(csv_path),
        )

    fresh_corpora = {
        "broad": full_suite_module.CorpusArtifact(
            key="broad",
            label="Broad representative latest corpus",
            row_count=200,
            csv_path=str(tmp_path / "fresh_broad.csv"),
            json_path=str(tmp_path / "fresh_broad.json"),
            summary_path=str(tmp_path / "fresh_broad.summary.json"),
            source_summary_path=str(tmp_path / "fresh_source.json"),
        ),
        "focused": full_suite_module.CorpusArtifact(
            key="focused",
            label="Focused ambiguity-heavy latest corpus",
            row_count=60,
            csv_path=str(tmp_path / "fresh_focused.csv"),
            json_path=str(tmp_path / "fresh_focused.json"),
            summary_path=str(tmp_path / "fresh_focused.summary.json"),
            source_summary_path=str(tmp_path / "fresh_source.json"),
        ),
        "transfer": full_suite_module.CorpusArtifact(
            key="transfer",
            label="Transfer latest corpus",
            row_count=50,
            csv_path=str(tmp_path / "fresh_transfer.csv"),
            json_path=str(tmp_path / "fresh_transfer.json"),
            summary_path=str(tmp_path / "fresh_transfer.summary.json"),
            source_summary_path=str(tmp_path / "fresh_transfer_source.json"),
        ),
        "synthetic": full_suite_module.CorpusArtifact(
            key="synthetic",
            label="Synthetic-lane latest corpus",
            row_count=1000,
            csv_path=str(tmp_path / "fresh_synthetic.csv"),
            json_path=str(tmp_path / "fresh_synthetic.json"),
            summary_path=str(tmp_path / "fresh_synthetic.summary.json"),
            source_summary_path=str(tmp_path / "fresh_source.json"),
        ),
    }

    generated_calls: list[str] = []

    def _fresh_generated_corpora(args: object, *, suite_run_id: str) -> dict[str, full_suite_module.CorpusArtifact]:  # noqa: ARG001
        generated_calls.append(str(suite_run_id))
        return fresh_corpora

    monkeypatch.setattr(full_suite_module, "_existing_corpus_artifact", _short_curated_corpus)
    monkeypatch.setattr(full_suite_module, "_build_generated_corpora", _fresh_generated_corpora)

    corpora = full_suite_module._build_corpora(args, suite_run_id="suite_floors")

    assert generated_calls == ["suite_floors"]
    assert corpora == fresh_corpora
    assert corpora["broad"].row_count >= 200
    assert corpora["focused"].row_count >= 60
    assert corpora["transfer"].row_count >= 50
    assert corpora["synthetic"].row_count >= 1000


def test_run_full_latest_suite_surfaces_threshold_sensitivity_lane_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _stub_corpus(key: str, label: str) -> full_suite_module.CorpusArtifact:
        csv_path = tmp_path / f"{key}.csv"
        json_path = tmp_path / f"{key}.json"
        summary_path = tmp_path / f"{key}.summary.json"
        source_summary_path = tmp_path / f"{key}.source_summary.json"
        _write_text(csv_path, "od_id,origin_lat,origin_lon,destination_lat,destination_lon\n")
        _write_json(json_path, {"rows": []})
        _write_json(summary_path, {"row_count": 0, "label": label})
        _write_json(source_summary_path, {"source": label})
        return full_suite_module.CorpusArtifact(
            key=key,
            label=label,
            row_count=0,
            csv_path=str(csv_path),
            json_path=str(json_path),
            summary_path=str(summary_path),
            source_summary_path=str(source_summary_path),
        )

    corpora = {
        "broad": _stub_corpus("broad", "Broad"),
        "focused": _stub_corpus("focused", "Focused"),
        "transfer": _stub_corpus("transfer", "Transfer"),
        "synthetic": _stub_corpus("synthetic", "Synthetic"),
    }

    def _fake_run_thesis_evaluation(args: object, *, client=None) -> dict[str, object]:  # noqa: ARG001
        run_id = str(getattr(args, "run_id"))
        role = str(getattr(args, "evaluation_suite_role"))
        if role == "threshold_sensitivity":
            summary_csv = full_suite_module.write_csv_artifact(
                run_id,
                "threshold_sensitivity_summary.csv",
                fieldnames=["variant_id", "threshold_parameter", "sweep_kind"],
                rows=[
                    {
                        "variant_id": "C",
                        "threshold_parameter": "certificate_threshold",
                        "sweep_kind": "configured",
                    }
                ],
            )
            summary_json = full_suite_module.write_json_artifact(
                run_id,
                "threshold_sensitivity_summary.json",
                {
                    "sweep_scheme": "one_factor_at_a_time",
                    "summary_rows": [
                        {
                            "variant_id": "C",
                            "threshold_parameter": "certificate_threshold",
                            "sweep_kind": "configured",
                        }
                    ],
                },
            )
            report_md = full_suite_module.write_text_artifact(
                run_id,
                "threshold_sensitivity_report.md",
                "# Threshold Sensitivity Report\n",
            )
            return {
                "run_id": run_id,
                "summary_rows": [
                    {
                        "variant_id": "C",
                        "pipeline_mode": "voi",
                        "row_count": 1,
                        "success_rate": 1.0,
                        "certified_rate": 1.0,
                        "mean_certificate": 0.91,
                        "weighted_win_rate_best_baseline": 1.0,
                        "dominance_win_rate_best_baseline": 1.0,
                        "dominance_win_rate_osrm": 1.0,
                        "dominance_win_rate_ors": 1.0,
                        "time_preserving_win_rate_best_baseline": 1.0,
                        "time_preserving_win_rate_osrm": 1.0,
                        "time_preserving_win_rate_ors": 1.0,
                        "mean_weighted_margin_vs_best_baseline": 3.5,
                        "mean_runtime_ratio_vs_osrm": 1.0,
                        "mean_runtime_ratio_vs_ors": 1.0,
                        "mean_runtime_p50_ms": 1.0,
                        "mean_runtime_p90_ms": 1.0,
                        "mean_runtime_p95_ms": 1.0,
                        "mean_process_rss_p90_mb": 1.0,
                        "median_preference_query_count": 0.0,
                        "p90_preference_query_count": 0.0,
                        "nontrivial_frontier_rate": 1.0,
                        "mean_dccs_false_safe_prune_rate": 0.0,
                        "mean_dccs_anti_collapse_success_rate": 1.0,
                        "mean_dccs_certificate_critical_hit_rate": 1.0,
                        "mean_dccs_time_preserving_challenger_coverage": 1.0,
                        "mean_dccs_dominance_likely_challenger_coverage": 1.0,
                        "mean_route_cache_hit_rate": 1.0,
                        "mean_option_build_cache_hit_rate": 1.0,
                        "mean_option_build_reuse_rate": 1.0,
                        "mean_refc_world_reuse_rate": 1.0,
                        "baseline_identity_verified_rate": 1.0,
                    }
                ],
                "lane_metadata": {
                    "evaluation_suite": {
                        "role": role,
                        "scope": "sensitivity",
                    }
                },
                "threshold_sensitivity_summary_csv": str(summary_csv),
                "threshold_sensitivity_summary_json": str(summary_json),
                "threshold_sensitivity_report": str(report_md),
            }
        return {
            "run_id": run_id,
            "summary_rows": [],
            "rows": [],
            "lane_metadata": {
                "evaluation_suite": {
                    "role": role,
                    "scope": "focused",
                }
            },
        }

    def _fake_run_hot_rerun(args: object, *, client=None) -> dict[str, object]:  # noqa: ARG001
        hot_run_id = str(getattr(args, "hot_run_id"))
        comparison_json = full_suite_module.write_json_artifact(
            hot_run_id,
            "hot_rerun_vs_cold_comparison.json",
            {"rows": []},
        )
        gate_json = full_suite_module.write_json_artifact(
            hot_run_id,
            "hot_rerun_gate.json",
            {"all_green": True},
        )
        report_path = full_suite_module.write_text_artifact(hot_run_id, "hot_rerun_report.md", "# hot rerun\n")
        return {
            "hot_run_id": hot_run_id,
            "comparison_json": str(comparison_json),
            "comparison_csv": str(comparison_json).replace(".json", ".csv"),
            "gate_json": str(gate_json),
            "report_path": str(report_path),
            "hot_gate": {"all_green": True},
        }

    monkeypatch.setattr(full_suite_module, "DIRECT_SUITE_ROLES", ("threshold_sensitivity",))
    monkeypatch.setattr(full_suite_module, "run_preflight", lambda output_path: {"strict_route_ready": True})  # noqa: ARG005
    monkeypatch.setattr(full_suite_module, "_build_corpora", lambda args, *, suite_run_id: corpora)  # noqa: ARG005
    monkeypatch.setattr(full_suite_module, "run_thesis_evaluation", _fake_run_thesis_evaluation)
    monkeypatch.setattr(full_suite_module, "run_hot_rerun_benchmark", _fake_run_hot_rerun)
    monkeypatch.setattr(full_suite_module.httpx, "Client", lambda *args, **kwargs: DummyClient())

    args = full_suite_module._build_parser().parse_args(
        [
            "--run-id",
            "suite_threshold_sensitivity_lane",
            "--out-dir",
            str(out_dir),
            "--live-backend",
        ]
    )

    payload = full_suite_module.run_full_latest_suite(args)

    results_payload = json.loads(Path(payload["results_json"]).read_text(encoding="utf-8"))
    lane_record = payload["lane_runs"]["threshold_sensitivity"]

    assert lane_record["lane_metadata"]["evaluation_suite"]["role"] == "threshold_sensitivity"
    assert lane_record["artifact_paths"]["threshold_sensitivity_summary_csv"].endswith("threshold_sensitivity_summary.csv")
    assert lane_record["artifact_paths"]["threshold_sensitivity_summary_json"].endswith("threshold_sensitivity_summary.json")
    assert lane_record["artifact_paths"]["threshold_sensitivity_report_md"].endswith("threshold_sensitivity_report.md")
    assert lane_record["artifact_paths"]["report_md"].endswith("thesis_report.md")
    assert Path(lane_record["artifact_paths"]["threshold_sensitivity_summary_csv"]).exists()
    assert Path(lane_record["artifact_paths"]["threshold_sensitivity_summary_json"]).exists()
    assert Path(lane_record["artifact_paths"]["threshold_sensitivity_report_md"]).exists()
    assert results_payload["lane_runs"]["threshold_sensitivity"]["artifact_paths"]["threshold_sensitivity_summary_csv"].endswith(
        "threshold_sensitivity_summary.csv"
    )
    assert results_payload["lane_runs"]["threshold_sensitivity"]["artifact_paths"]["threshold_sensitivity_report_md"].endswith(
        "threshold_sensitivity_report.md"
    )


def test_repair_failure_atlas_suite_root_republishes_root_surfaces(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    old_out_dir = full_suite_module.settings.out_dir
    full_suite_module.settings.out_dir = out_dir
    try:
        suite_run_id = "suite_failure_atlas_repair"
        focused_refc_run_id = f"{suite_run_id}_focused_refc_proof"
        focused_voi_run_id = f"{suite_run_id}_focused_voi_proof"

        def _write_route_bundle(
            artifact_run_id: str,
            *,
            set_size: int,
            support_flag: bool,
            support_status: str,
            support_bin: str,
            singleton_not_justified_reasons: list[str] | None = None,
            excluded_route_safety_reasons: list[str] | None = None,
            stop_reason: str | None = None,
            abstention_reason_code: str | None = None,
        ) -> None:
            full_suite_module.write_json_artifact(
                artifact_run_id,
                "decision_region_summary.json",
                {
                    "active_challenger_id": "route_b",
                    "dominant_evidence_family": "weather",
                    "support_status": support_status,
                    "support_bin": support_bin,
                    "calibration_bin": "empirical",
                    "selected_certificate_basis": "empirical",
                    "route_fragility_family_count": 1,
                    "root_cause_tags": ["boundary:demo"],
                },
            )
            full_suite_module.write_json_artifact(
                artifact_run_id,
                "certificate_witness.json",
                {
                    "support_status": support_status,
                    "support_bin": support_bin,
                    "calibration_bin": "empirical",
                    "selected_certificate_basis": "empirical",
                    "targeted_challenger_route_id": "route_b",
                    "root_cause_tags": ["witness:demo"],
                    "support_flag": support_flag,
                },
            )
            full_suite_module.write_json_artifact(
                artifact_run_id,
                "certified_set_summary.json",
                {
                    "set_size": set_size,
                    "support_flag": support_flag,
                    "artifact_provenance": {"support_status": support_status},
                    "witness": {
                        "singleton_not_justified_reasons": singleton_not_justified_reasons or [],
                        "excluded_route_safety_reasons": excluded_route_safety_reasons or [],
                        "outside_routes_safely_excluded": not bool(excluded_route_safety_reasons),
                    },
                },
            )
            decision_package = {
                "schema_version": "1.0.0",
                "terminal_type": (
                    "typed_abstention"
                    if abstention_reason_code
                    else ("certified_set" if set_size > 1 else "certified_singleton")
                ),
                "abstention": (
                    {
                        "reason_code": abstention_reason_code,
                        "reason": abstention_reason_code,
                        "support_flag": support_flag,
                    }
                    if abstention_reason_code
                    else {}
                ),
                "abstention_summary": (
                    {
                        "reason_code": abstention_reason_code,
                        "support_flag": support_flag,
                    }
                    if abstention_reason_code
                    else {}
                ),
            }
            full_suite_module.write_json_artifact(
                artifact_run_id,
                "decision_package.json",
                decision_package,
            )
            full_suite_module.write_json_artifact(
                artifact_run_id,
                "voi_stop_certificate.json",
                {"stop_reason": stop_reason},
            )
            full_suite_module.write_json_artifact(
                artifact_run_id,
                "results.json",
                {"artifact_run_id": artifact_run_id},
            )

        overlap_artifact = f"{focused_refc_run_id}_wrong_singleton_overlap"
        _write_route_bundle(
            overlap_artifact,
            set_size=2,
            support_flag=False,
            support_status="unsupported",
            support_bin="weak_support",
            singleton_not_justified_reasons=["frontier_pairwise_gap_unresolved"],
            excluded_route_safety_reasons=["excluded_route_pairwise_gap_unresolved"],
            stop_reason="support_gap",
        )
        refc_rows = [
            {
                "variant_id": "A",
                "od_id": "focused-wrong-singleton-overlap",
                "profile_id": "p0",
                "artifact_run_id": overlap_artifact,
                "terminal_type": "open",
                "corpus_group": "ambiguity",
            }
        ]
        full_suite_module.write_json_artifact(
            focused_refc_run_id,
            "results.json",
            {
                "run_id": focused_refc_run_id,
                "rows": refc_rows,
            },
        )

        voi_rows: list[dict[str, object]] = []
        support_artifact = f"{focused_voi_run_id}_support_downgrade"
        _write_route_bundle(
            support_artifact,
            set_size=1,
            support_flag=False,
            support_status="unsupported",
            support_bin="weak_support",
            stop_reason="support_gap",
        )
        voi_rows.append(
            {
                "variant_id": "C",
                "od_id": "focused-support-downgrade",
                "profile_id": "p2",
                "artifact_run_id": support_artifact,
                "failure_reason": "support_flag_false",
                "terminal_type": "open",
                "support_flag": False,
                "corpus_group": "support_fragile",
            }
        )
        for index in range(6):
            abstention_artifact = f"{focused_voi_run_id}_abstention_{index}"
            _write_route_bundle(
                abstention_artifact,
                set_size=0,
                support_flag=True,
                support_status="supported",
                support_bin="supported",
                stop_reason="budget_exhausted",
                abstention_reason_code="uncertified_due_to_budget",
            )
            voi_rows.append(
                {
                    "variant_id": "C",
                    "od_id": f"focused-abstention-{index}",
                    "profile_id": f"p_abst_{index}",
                    "artifact_run_id": abstention_artifact,
                    "preference_terminal_type": "abstained",
                    "corpus_group": "controller_stress",
                }
            )
        full_suite_module.write_json_artifact(
            focused_voi_run_id,
            "results.json",
            {
                "run_id": focused_voi_run_id,
                "rows": voi_rows,
            },
        )

        hot_run_id = f"{suite_run_id}_hot_rerun_hot"
        lane_runs = {
            "focused_refc_proof": {
                "status": "completed",
                "role": "focused_refc_proof",
                "run_id": focused_refc_run_id,
                "corpus_key": "focused",
                "artifact_paths": {
                    "results_json": str(full_suite_module.artifact_dir_for_run(focused_refc_run_id) / "results.json"),
                },
            },
            "focused_voi_proof": {
                "status": "completed",
                "role": "focused_voi_proof",
                "run_id": focused_voi_run_id,
                "corpus_key": "focused",
                "artifact_paths": {
                    "results_json": str(full_suite_module.artifact_dir_for_run(focused_voi_run_id) / "results.json"),
                },
            },
            "hot_rerun": {
                "status": "completed",
                "role": "hot_rerun",
                "run_id": hot_run_id,
                "hot_gate": {
                    "all_green": True,
                    "cold_run_id": f"{suite_run_id}_hot_rerun_cold",
                    "hot_run_id": hot_run_id,
                    "pair_run_id": f"{suite_run_id}_hot_rerun_pair",
                },
            },
        }

        full_suite_module.write_json_artifact(
            suite_run_id,
            "lane_publishability_summary.json",
            {
                "rows": [
                    {
                        "lane_role": "focused_refc_proof",
                        "variant_id": "A",
                    }
                ]
            },
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "universal_baseline_audit.json",
            {"rows": []},
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "sample_size_gate_summary.json",
            {"rows": []},
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "headline_seed_claims_summary.json",
            {"rows": []},
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "failure_atlas.json",
            {"rows": [{"stale": True}]},
        )
        full_suite_module.write_text_artifact(
            suite_run_id,
            "failure_atlas.md",
            "# stale atlas\n",
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "index.json",
            {
                "schema_version": "full-latest-suite-v1",
                "suite_run_id": suite_run_id,
                "lane_runs": lane_runs,
                "failure_atlas_json": str(full_suite_module.artifact_dir_for_run(suite_run_id) / "failure_atlas.json"),
                "failure_atlas_md": str(full_suite_module.artifact_dir_for_run(suite_run_id) / "failure_atlas.md"),
            },
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "results.json",
            {
                "schema_version": "full-latest-suite-v1",
                "suite_run_id": suite_run_id,
                "lane_runs": lane_runs,
                "failure_atlas_rows": [{"stale": True}],
                "failure_atlas": {"rows": [{"stale": True}]},
            },
        )
        full_suite_module.write_json_artifact(
            suite_run_id,
            "metadata.json",
            {
                "schema_version": "full-latest-suite-v1",
                "suite_run_id": suite_run_id,
            },
        )

        repaired = full_suite_module.repair_failure_atlas_suite_root(
            suite_run_id=suite_run_id,
            out_dir=out_dir,
        )

        repaired_lane_metadata = json.loads(Path(repaired["failure_atlas_lane_metadata_json"]).read_text(encoding="utf-8"))
        repaired_atlas = json.loads(Path(repaired["failure_atlas_json"]).read_text(encoding="utf-8"))
        repaired_results = json.loads(Path(repaired["results_json"]).read_text(encoding="utf-8"))
        repaired_index = json.loads(Path(repaired["index_json"]).read_text(encoding="utf-8"))
        repaired_metadata = json.loads(Path(repaired["metadata_json"]).read_text(encoding="utf-8"))
        repaired_verdict = json.loads(Path(repaired["publishability_verdict_json"]).read_text(encoding="utf-8"))

        assert repaired["repaired_roles"] == ["focused_refc_proof", "focused_voi_proof"]
        assert repaired_lane_metadata["lane_id"] == "failure_atlas"
        assert repaired_lane_metadata["lane_status"] == "present_complete"
        assert repaired_lane_metadata["coverage_class_counts"]["certified_set_violation"] == 1
        assert repaired_lane_metadata["abstention_class_example_target"] == 5
        assert repaired_atlas["lane_id"] == "failure_atlas"
        assert repaired_atlas["coverage_class_counts"]["abstention"] == 6
        assert repaired_results["failure_atlas_lane_metadata_json"] == repaired["failure_atlas_lane_metadata_json"]
        assert repaired_results["failure_atlas_lane_metadata"]["lane_id"] == "failure_atlas"
        assert repaired_results["failure_atlas"]["coverage_class_counts"]["certified_set_violation"] == 1
        assert repaired_index["failure_atlas_lane_metadata_json"] == repaired["failure_atlas_lane_metadata_json"]
        assert repaired_index["publishability_verdict_json"] == repaired["publishability_verdict_json"]
        assert repaired_metadata["failure_atlas_lane_id"] == "failure_atlas"
        assert repaired_metadata["failure_atlas_lane_status"] == "present_complete"
        assert repaired_metadata["failure_atlas_lane_metadata_json"] == repaired["failure_atlas_lane_metadata_json"]
        assert repaired_verdict["failure_atlas_case_count"] == 8
    finally:
        full_suite_module.settings.out_dir = old_out_dir


def test_baseline_audit_rows_capture_vehicle_restriction_and_feasibility_context(
    tmp_path: Path,
) -> None:
    corpus = full_suite_module.CorpusArtifact(
        key="broad",
        label="Broad curated corpus",
        row_count=1,
        csv_path=str(tmp_path / "broad.csv"),
        json_path=str(tmp_path / "broad.json"),
        summary_path=str(tmp_path / "broad.summary.json"),
        source_summary_path=str(tmp_path / "broad.source_summary.json"),
    )
    args = full_suite_module.argparse.Namespace(
        vehicle_type="rigid_hgv",
        departure_time_utc="2026-04-12T10:00:00Z",
        scenario_mode="no_sharing",
        disable_tolls=True,
        baseline_refinement_policy="corridor_uniform",
        ors_baseline_policy="local_service",
        ors_snapshot_mode="off",
        allow_proxy_ors=False,
        allow_evidence_fallbacks=False,
    )
    payload = {
        "run_id": "suite_fairness",
        "vehicle_type": "rigid_hgv",
        "scenario_mode": "no_sharing",
        "disable_tolls": True,
        "baseline_refinement_policy": "corridor_uniform",
        "ors_baseline_policy": "local_service",
        "ors_snapshot_mode": "off",
        "allow_proxy_ors": False,
        "allow_evidence_fallbacks": False,
        "baseline_smoke_summary": {
            "required_ok": True,
            "payload": {
                "origin": {"lat": 52.4862, "lon": -1.8904},
                "destination": {"lat": 51.5072, "lon": -0.1276},
                "vehicle_type": "rigid_hgv",
            },
            "osrm": {
                "ok": True,
                "provider_mode": "repo_local",
                "method": "osrm_engine_baseline",
            },
            "ors": {
                "ok": True,
                "provider_mode": "local_service",
                "method": "ors_local_engine_baseline",
            },
        },
        "rows": [
            {
                "variant_id": "A",
                "od_id": "od-1",
                "best_baseline_provider": "osrm",
                "ors_provider_mode": "local_service",
                "ors_graph_identity_status": "graph_identity_verified",
                "osrm_method": "osrm_engine_baseline",
                "ors_method": "ors_local_engine_baseline",
            }
        ],
        "summary_rows": [
            {
                "variant_id": "A",
                "pipeline_mode": "dccs",
                "row_count": 1,
                "baseline_identity_verified_rate": 1.0,
                "weighted_win_rate_best_baseline": 1.0,
                "dominance_win_rate_best_baseline": 1.0,
                "time_preserving_win_rate_best_baseline": 1.0,
                "mean_runtime_ratio_vs_osrm": 1.0,
                "mean_runtime_ratio_vs_ors": 1.0,
            }
        ],
    }

    audit_rows = full_suite_module._baseline_audit_rows_for_lane(
        role="broad_cold_proof",
        payload=payload,
        corpus=corpus,
        suite_args=args,
    )
    assert len(audit_rows) == 1
    audit_row = audit_rows[0]
    assert audit_row["matched_od_count"] == 1
    assert json.loads(audit_row["matched_od_ids_json"]) == ["od-1"]
    assert audit_row["matched_vehicle_type"] == "rigid_hgv"
    assert audit_row["baseline_smoke_required_ok"] is True
    assert json.loads(audit_row["matched_restriction_context_json"]) == {
        "allow_evidence_fallbacks": False,
        "allow_proxy_ors": False,
        "baseline_refinement_policy": "corridor_uniform",
        "disable_tolls": True,
        "ors_baseline_policy": "local_service",
        "ors_snapshot_mode": "off",
        "scenario_mode": "no_sharing",
    }
    feasibility_context = json.loads(audit_row["matched_route_feasibility_context_json"])
    assert feasibility_context["required_ok"] is True
    assert feasibility_context["vehicle_type"] == "rigid_hgv"
    assert feasibility_context["osrm_ok"] is True
    assert feasibility_context["ors_ok"] is True

    verdict = full_suite_module._publishability_verdict_payload(
        lane_publishability_rows=[
            {
                "lane_role": "broad_cold_proof",
                "variant_id": "A",
                "dominance_win_rate_best_baseline": 1.0,
                "dominance_win_rate_osrm": 1.0,
                "time_preserving_win_rate_best_baseline": 1.0,
                "time_preserving_win_rate_osrm": 1.0,
                "time_preserving_win_rate_ors": 1.0,
                "mean_weighted_margin_vs_best_baseline": 3.5,
                "baseline_identity_verified_rate": 1.0,
            }
        ],
        baseline_audit_rows=audit_rows,
        failure_atlas_rows=[],
        sample_size_rows=[],
        headline_seed_claim_rows=[],
        hot_payload={"hot_gate": {"all_green": True}},
    )
    assert verdict["fairness_failure_count"] == 0

    broken_verdict = full_suite_module._publishability_verdict_payload(
        lane_publishability_rows=[],
        baseline_audit_rows=[dict(audit_row, matched_vehicle_type="")],
        failure_atlas_rows=[],
        sample_size_rows=[],
        headline_seed_claim_rows=[],
        hot_payload={"hot_gate": {"all_green": True}},
    )
    assert broken_verdict["fairness_failure_count"] == 1


def test_build_failure_atlas_treats_proxy_only_positivity_failures_as_support_downgrades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full_suite_module.settings, "out_dir", str(tmp_path))
    artifact_run_id = "proxy_only_support_gap_case"

    full_suite_module.write_json_artifact(
        artifact_run_id,
        "decision_region_summary.json",
        {
            "active_challenger_id": "route_b",
            "dominant_evidence_family": "weather",
            "support_status": "supported",
            "support_bin": "supported",
            "calibration_bin": "empirical",
            "selected_certificate_basis": "empirical",
            "route_fragility_family_count": 1,
            "root_cause_tags": ["boundary:demo"],
        },
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "certificate_witness.json",
        {
            "support_status": "supported",
            "support_bin": "supported",
            "support_flag": True,
            "root_cause_tags": ["witness:demo"],
        },
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "certified_set_summary.json",
        {
            "set_size": 1,
            "support_flag": True,
            "artifact_provenance": {"support_status": "supported"},
            "witness": {
                "singleton_not_justified_reasons": [],
                "excluded_route_safety_reasons": [],
                "outside_routes_safely_excluded": True,
            },
        },
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "decision_package.json",
        {"schema_version": "1.0.0", "terminal_type": "certified_singleton"},
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "voi_stop_certificate.json",
        {"stop_reason": ""},
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "results.json",
        {"artifact_run_id": artifact_run_id},
    )

    payload = full_suite_module._build_failure_atlas(
        suite_run_id="suite_failure_atlas_proxy_gap",
        rows=[
            {
                "_suite_role": "focused_refc_proof",
                "_suite_lane_run_id": "focused_refc_proof_run",
                "variant_id": "A",
                "od_id": "proxy-gap-od",
                "profile_id": "profile-1",
                "artifact_run_id": artifact_run_id,
                "terminal_type": "open",
                "support_flag": True,
                "support_status": "supported",
                "support_bin": "supported",
                "positivity_ok": False,
                "proxy_only_fraction": 1.0,
                "audited_route_pair_count": 0,
            }
        ],
    )

    assert payload["counts_by_kind"]["support_downgrade"] == 1
    assert payload["coverage_class_counts"]["support_downgrade"] == 1
    assert payload["root_cause_family_counts"]["support_failure"] == 1
    row = payload["rows"][0]
    assert row["support_status"] == "unsupported"
    assert row["support_flag"] is False
    assert "support_failure" in row["root_cause_tags"]


def test_build_failure_atlas_prioritizes_support_downgrade_when_signals_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full_suite_module.settings, "out_dir", str(tmp_path))
    artifact_run_id = "support_downgrade_overlap_case"

    full_suite_module.write_json_artifact(
        artifact_run_id,
        "decision_region_summary.json",
        {
            "active_challenger_id": "route_b",
            "dominant_evidence_family": "weather",
            "support_status": "unsupported",
            "support_bin": "weak_support",
            "calibration_bin": "empirical",
            "selected_certificate_basis": "empirical",
            "route_fragility_family_count": 1,
            "root_cause_tags": ["boundary:demo"],
        },
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "certificate_witness.json",
        {
            "support_status": "unsupported",
            "support_bin": "weak_support",
            "support_flag": False,
            "singleton_not_justified_reasons": ["challenger_gap_nonpositive"],
            "root_cause_tags": ["witness:demo"],
        },
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "certified_set_summary.json",
        {
            "set_size": 1,
            "support_flag": False,
            "artifact_provenance": {"support_status": "unsupported"},
            "witness": {
                "singleton_not_justified_reasons": ["challenger_gap_nonpositive"],
                "excluded_route_safety_reasons": [],
                "outside_routes_safely_excluded": True,
            },
        },
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "decision_package.json",
        {"schema_version": "1.0.0", "terminal_type": "certified_singleton"},
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "voi_stop_certificate.json",
        {"stop_reason": ""},
    )
    full_suite_module.write_json_artifact(
        artifact_run_id,
        "results.json",
        {"artifact_run_id": artifact_run_id},
    )

    payload = full_suite_module._build_failure_atlas(
        suite_run_id="suite_failure_atlas_support_overlap",
        rows=[
            {
                "_suite_role": "focused_refc_proof",
                "_suite_lane_run_id": "focused_refc_proof_run",
                "variant_id": "B",
                "od_id": "support-overlap-od",
                "profile_id": "profile-1",
                "artifact_run_id": artifact_run_id,
                "terminal_type": "certified_singleton",
                "support_flag": False,
                "support_status": "unsupported",
                "support_bin": "weak_support",
                "singleton_not_justified": True,
            }
        ],
    )

    assert payload["counts_by_kind"]["support_downgrade"] == 1
    assert payload["coverage_class_counts"]["support_downgrade"] == 1
    assert payload["coverage_class_counts"]["wrong_singleton"] == 1
    assert payload["rows"][0]["atlas_kind"] == "support_downgrade"


def test_result_row_marks_focused_support_downgrade_from_weak_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full_suite_module.settings, "out_dir", str(tmp_path))
    args = full_suite_module._build_parser().parse_args([])
    spec = thesis_module.VariantSpec("C", "voi", "dccs")
    od = {
        "od_id": "focused-support-downgrade",
        "seed": 1,
        "trip_length_bin": "short",
        "origin_lat": 0.0,
        "origin_lon": 0.0,
        "destination_lat": 1.0,
        "destination_lon": 1.0,
        "straight_line_km": 1.0,
        "profile_id": "profile-1",
        "corpus_group": "support_fragile",
        "corpus_kind": "focused",
        "corridor_bucket": "bucket-1",
        "candidate_probe_path_count": "1",
        "candidate_probe_corridor_family_count": "1",
    }
    route_response = {
        "selected": {
            "id": "route-a",
            "duration_s": 10.0,
            "monetary_cost": 1.0,
            "emissions_kg": 1.0,
            "distance_km": 1.0,
        },
        "selected_certificate": {"certificate": 0.25, "certified": False},
        "candidates": [],
        "run_id": "run-focused-support-downgrade",
        "manifest_endpoint": "manifest",
        "artifacts_endpoint": "artifacts",
    }
    artifacts = {
        "dccs_summary.json": {},
        "certificate_summary.json": {},
        "certificate_witness.json": {},
        "certified_set_summary.json": {},
        "initial_certificate_summary.json": {},
        "flip_radius_summary.json": {},
        "route_fragility_map.json": {},
        "initial_route_fragility_map.json": {},
        "competitor_fragility_breakdown.json": {},
        "value_of_refresh.json": {},
        "initial_value_of_refresh.json": {},
        "sampled_world_manifest.json": {},
        "world_support_summary.json": {
            "support_flag": True,
            "support_status": "supported",
            "proxy_only_fraction": 1.0,
            "multi_fidelity_summary": {
                "proxy_only_fraction": 1.0,
                "audit_correction_mass": 0.0,
                "proxy_world_count": 1,
                "audit_world_count": 0,
                "correction_path_estimator": "toy",
                "multi_fidelity_certificate_basis": "toy",
                "proxy_bias_model_version": "toy",
                "audit_propensity_version": "toy",
                "proxy_correction_active": True,
                "correction_training_leakage_safe": True,
                "propensity_training_leakage_safe": True,
                "leakage_safe_training": True,
            },
            "positivity_diagnostics": {
                "weak_overlap_detected": True,
                "positivity_ok": False,
                "audited_route_pair_count": 0,
                "candidate_route_pair_count": 0,
                "audit_coverage_ratio": 0.0,
                "minimum_propensity": 0.0,
                "mean_propensity": 0.0,
                "maximum_propensity": 0.0,
            },
        },
        "voi_action_trace.json": {},
        "voi_stop_certificate.json": {},
        "final_route_trace.json": {},
    }
    osrm = thesis_module.BaselineResult(
        route={},
        metrics={"distance_km": 11.0, "duration_s": 11.0, "monetary_cost": 1.5, "emissions_kg": 1.2},
        method="osrm_engine_baseline",
        compute_ms=1.0,
    )
    ors = thesis_module.BaselineResult(
        route={},
        metrics={"distance_km": 12.0, "duration_s": 12.0, "monetary_cost": 1.7, "emissions_kg": 1.4},
        method="ors_local_engine_baseline",
        compute_ms=1.0,
    )

    row = thesis_module._result_row(
        args,
        od,
        spec,
        route_response,
        1.0,
        artifacts,
        osrm,
        ors,
        readiness_summary={"route_graph": {}},
    )

    assert row["failure_reason"] == "support_flag_false"
    assert row["support_flag"] is False
    assert row["support_status"] == "unsupported"
    assert row["out_of_support_reason"] == "out_of_support_world_model"
