# Agent Ops

This directory is the durable companion documentation for the repo-scoped Codex operating system.

## What Exists

- Root operating manual: `AGENTS.md`
- Project Codex config: `.codex/config.toml`
- Custom role definitions: `.codex/agents/*.toml`
- Core coordination implementation: `tools/codex_coord_lib.py`
- Shared coordination CLI: `tools/codex_coord.py`
- Python lease wrapper: `tools/codex_run_python.py`
- Coordination tests: `tests/test_codex_coord.py`

The coordination CLI now includes first-class child-subagent lifecycle commands:

- `start-child`
- `heartbeat-child`
- `update-child`
- `note-child`
- `end-child`
- `replace-child`
- `ensure-six-subagents`

Child-only commands currently take the child session id via `--session-id`.

## Live Shared State

The canonical runtime state is not a tracked working-tree file.
When Git is available, the control plane lives under:

```text
<git-common-dir>/codex-coordination/
```

In this repository that resolves from `git rev-parse --git-common-dir`, which keeps the state shared across worktrees.

If Git is unavailable, the fallback runtime root is:

```text
.codex_tmp/codex-coordination/
```

## Checked-In Vs Runtime-Generated

Checked into the repo:

- `AGENTS.md`
- `.codex/config.toml`
- `.codex/agents/*.toml`
- `tools/codex_coord.py`
- `tools/codex_run_python.py`
- `docs/agent-ops/*.md`
- `tests/test_codex_coord.py`

Generated at runtime under the shared coordination root:

- `coordination.sqlite3`
- shared JSON snapshots for active parent sessions, active child sessions, work-item claims, file claims, Python leases, messages, and events
- a Markdown status board snapshot that exposes child compliance, live child count, missing child count, and unhealthy child count per parent
- per-session snapshot files with child rosters and child notes
- final Markdown and JSON archives under the archive area

## Quick Start

1. Inspect the board:

```powershell
python tools/codex_coord.py status
```

2. Inspect health and invariant failures:

```powershell
python tools/codex_coord.py doctor
```

3. Inspect the resolved shared runtime root:

```powershell
python tools/codex_coord.py runtime-path
```

4. Register a parent session:

```powershell
python tools/codex_coord.py start-parent --task-summary "Summarize and claim the next task" --task-scope reviewer-audit
```

5. Immediately register 6 real child subagents, one per slot, and give each child a real packet or explicit standby/watch duty:

```powershell
python tools/codex_coord.py start-child --parent-session-id <parent_session> --slot-id 1 --session-id <child_session> --role Planner --agent-name Planner --agent-kind planner --status active --summary "Own TASK-LEDGER and keep the work ledger current" --work-item-id TASK-LEDGER
python tools/codex_coord.py ensure-six-subagents --session-id <parent_session>
```

6. Claim work and files in the identity that will execute them. For delegated parent-controller work, that means the child session:

```powershell
python tools/codex_coord.py upsert-work-item --session-id <parent_session> --work-item-id TASK-01 --title "Sync backend and docs"
python tools/codex_coord.py claim-work --session-id <child_session> --work-item-id TASK-01 --status in_progress
python tools/codex_coord.py claim-file --session-id <child_session> --path docs/agent-ops/README.md --section-start-line 1 --section-end-line 40
```

7. Keep both parent and child liveness fresh while active:

```powershell
python tools/codex_coord.py heartbeat --session-id <parent_session> --note "coordinating active children"
python tools/codex_coord.py heartbeat-child --session-id <child_session> --note "working on TASK-01"
```

8. End cleanly. Replace stale or exited children before declaring the parent healthy, and use `end-parent` only after cleanup is complete:

```powershell
python tools/codex_coord.py end-child --session-id <child_session> --outcome completed --note "Packet closed and claims released"
python tools/codex_coord.py end-parent --session-id <parent_session> --outcome completed --note "Final cleanup complete"
```

See the sibling docs for protocol, lifecycle, Python leases, role definitions, and detailed command examples.
