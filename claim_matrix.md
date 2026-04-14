# Claim Matrix

What-If Freight Router is an auditable, tri-source, selective minimum-cost certification engine for freight-route recommendation under incomplete search, biased evidence, and ambiguous preferences.

This top-level file is the reviewer-facing claim index. It records current descriptive and empirical implementation evidence. The expanded rationale, caveats, and evidence notes live in [`docs/claim_matrix.md`](docs/claim_matrix.md). Formal proof maturity is tracked separately in `docs/theorem_map.md`.

## Claim Status Language

- `theorem-backed`: supported by an explicit theorem/proposition package.
- `empirical`: supported by checked code, tests, artifacts, or evaluator outputs.
- `heuristic-but-measured`: driven by heuristic logic that is instrumented and measured.
- `descriptive-only`: repository framing, navigation, or implementation description rather than a performance or validity claim.

## Headline Claims

| Headline claim | Status | Evidence surface | Notes |
| --- | --- | --- | --- |
| The repository is framed as a certification engine rather than a generic router. | descriptive-only | `docs/claim_matrix.md`, `docs/thesis-codebase-report.md` | Identity and scope statement, not a theorem or benchmark claim. |
| The default primary runtime for thesis-facing non-waypoint requests resolves through the redesigned certification path, while `legacy` remains the explicit comparator path and still handles the current waypoint fallback. | empirical | `backend/app/settings.py`, `backend/app/main.py`, `docs/redesign-implementation-tracker.md` | Runtime-default claim narrowed to current behavior. |
| User-facing terminal outcomes are limited to certified singleton, certified set, or typed abstention. | empirical | `backend/app/models.py`, `backend/app/abstention.py`, `backend/tests/test_route_terminal_semantics.py` | API/runtime contract claim. |
| The live `/route` response surface is `DecisionPackage`, with compatibility fields retained so the current UI can still read `selected`, `candidates`, and the summary families. | empirical | `backend/app/models.py`, `backend/app/main.py`, `backend/tests/test_refc_artifact_contract.py` | Response-contract claim aligned to the current endpoint declaration and payload assembly. |
| Preference, support, and multi-fidelity evidence state are first-class runtime surfaces. | empirical | `backend/app/preference_state.py`, `backend/app/support_model.py`, `backend/tests/test_support_fidelity_world_models.py` | Structural implementation claim. |
| REFC, VOI, and run-store artifact families are explicit and replayable. | empirical | `backend/app/run_store.py`, `backend/app/main.py`, `backend/tests/test_run_store_artifacts.py`, `backend/tests/test_refc_artifact_contract.py` | Artifact-contract claim. |
| No formal theorem package is currently published for this slice; theorem maturity is tracked separately from the implementation evidence recorded here. | descriptive-only | `docs/theorem_map.md`, `docs/claim_matrix.md` | Negative scope statement to prevent over-claiming and to separate proof maturity from implementation evidence. |
| Hard redesign gates are not represented here as green unless backed by current evidence elsewhere in the repo. | descriptive-only | `docs/quality-gates-and-benchmarks.md` | Claim-discipline statement. |

## Detailed Matrix

Use [`docs/claim_matrix.md`](docs/claim_matrix.md) for the expanded certification-oriented matrix and evidence notes.
