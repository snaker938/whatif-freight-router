# Coordination Protocol

## Canonical Runtime Path

The control plane must resolve its live runtime directory from:

```powershell
git rev-parse --git-common-dir
```

The canonical shared runtime directory is:

```text
<git-common-dir>/codex-coordination/
```

If Git is unavailable, the documented fallback is:

```text
.codex_tmp/codex-coordination/
```

The fallback is untracked and ignored by the repo.

The core implementation lives in `tools/codex_coord_lib.py`; `tools/codex_coord.py` is the CLI wrapper that exposes the shared control plane to Codex sessions.

## Canonical State Model

The source of truth is one SQLite database:

```text
coordination.sqlite3
```

Human-readable JSON and Markdown snapshots are regenerated from the database after each mutation.
The store publishes a schema version and snapshot generation metadata so snapshot drift can be detected against the canonical SQLite state.

The store tracks:

- parent sessions
- child subagent sessions linked to parents
- slot projections per parent, each pointing at the currently active child for that parent-local slot when one exists
- work-item ledger rows and exclusive claims
- file claims and optional section claims
- Python leases
- parent checkpoints
- child notes and handoff breadcrumbs
- messages and inbox state
- event log entries
- archived finished sessions

Current session outcomes in the implementation are `active`, `completed`, `interrupted`, `crashed_or_lost`, `reaped`, `handed_off`, and `taken_over`.
Legacy aliases such as `finished`, `handoff`, and `stale_reaped` remain accepted at the CLI boundary and are normalized into the canonical outcomes above.

## Parent-Controller Invariant

Parent-controller mode requires exactly 6 live child subagents for every active parent session.

The live roster is evaluated from child session rows, not from role labels, `.codex/agents/*.toml`, `max_threads = 6`, or slot projections alone.
A child counts as live only when all of the following are true:

- the session is a registered child linked to the parent
- the child status is `active`
- the child heartbeat is fresh
- `role` is populated
- `activity_status` is populated
- the child has assigned work items, or it is on an explicit standby/watch status with a non-empty summary

`status`, `doctor`, and `ensure-six-subagents` surface the invariant through:

- `child_compliance`
- `required_child_count`
- `live_child_count`
- `missing_child_count`
- `unhealthy_child_count`
- `missing_child_slots`
- child roster tables that show child session id, role, activity status, health, and assigned work items

A parent may not complete successfully while the child roster is noncompliant.

## Work-Item Claims

Rules:

- Preserve provided checklist IDs exactly.
- If the prompt has no IDs, mint stable task-local IDs before delegation.
- Claims are keyed by `task_scope + raw_work_item_id`, not by the raw ID alone.
- Use `start-parent --task-scope <scope>` to deliberately join an existing task namespace; otherwise the store derives a stable scope from the prompt summary.
- A work-item claim is exclusive across active owners.
- Parent sessions may still own claims for undelegated or compatibility work, but delegated parent-controller execution should be claimed by the responsible child session.
- If a work item is already claimed, the next parent or child must choose other work or message the current owner.
- Valid states are `open`, `claimed`, `in_progress`, `blocked`, `qa`, and `closed`.

Stored fields:

- `work_item_id`
- `raw_work_item_id`
- `task_scope`
- `title`
- `source_ref`
- `status`
- `owner_session`
- `owner_slot`
- `created_by_session`
- `created_at`
- `claimed_at`
- `updated_at`
- `latest_note`
- `evidence`

Derived board and snapshot fields surface the owner context:

- `owner_parent_session`
- `owner_role`
- `owner_session_type`

## File Claims

Rules:

- Write claims are exclusive across active owners.
- Whole-file claims block all section claims on the same path.
- Line-range section claims may coexist only when they target the same file and non-overlapping spans.
- Label-only section claims remain conservative and are treated as conflicting when the store cannot prove they are disjoint.
- QA and read-only inspection remain claim-free.
- When work is delegated in parent-controller mode, the child that edits the file should own the file claim.

Stored fields:

- normalized repo-relative path
- optional `section_id`
- optional `section_start_line`
- optional `section_end_line`
- `mode`
- `owner_session`
- optional `owner_slot`
- `claimed_at`
- `last_heartbeat`
- `stale_after_seconds`
- `note`

Derived board and snapshot fields surface:

- `owner_parent_session`
- `owner_role`
- `owner_session_type`

## Messages

Messages support:

- targeted session handoffs
- broadcast notices
- claim conflict warnings
- unblock notes
- general coordination notes

Senders and recipients may be parent or child sessions.

Each message stores:

- sender session
- optional recipient session
- category
- subject
- body
- optional related work-item ID
- optional related repo path
- timestamp
- ack state
- archive state

## Child Notes And Parent Checkpoints

Parent checkpoints publish:

- parent session id
- task summary
- blockers
- next actions
- evidence paths
- the current slot projection table
- current work-item counts
- current file claims
- current Python leases
- durable resume context

Child subagents publish status, handoff, and evidence breadcrumbs through `note-child`.
Session snapshots and archives include child rosters and child notes so later parents can see which child owned which packet or file.

## Shared Snapshots

The runtime regenerates shared JSON snapshots for:

- active parent sessions
- active child sessions
- work-item claims
- file claims
- Python leases
- messages
- events
- snapshot manifest

The runtime also writes a Markdown status board and per-session snapshot files.
The archive directory stores final Markdown and JSON summaries for each parent and child session.

## Stale Takeover, Replacement, And Recovery

Default heartbeat interval guidance is 60 seconds.
Default stale threshold is 300 seconds.

Rules:

- Stale recovery must be explicit through `reap-stale`.
- A fresh claim may not be stolen silently.
- Stale child state is tracked separately from stale parent state.
- `replace-child` can replace one missing, stale, or exited child without tearing down the parent session.
- `end-child` cleans up child-owned file claims and Python leases, can return owned work to the parent, and marks the slot as needing replacement when appropriate.
- `resume-parent` creates a new active parent session from the latest durable checkpoint when interrupted work should continue without mutating the archived source session. The resumed parent must restore itself to 6 live children before taking new work.
- Reaping always drops stale file claims and closes stale Python leases.
- File claims are never auto-transferred; the new owner must claim files explicitly before editing.
- Every reap, replacement, and takeover is recorded in the event log and archived session summary.

## Store Health And Repair

The coordination CLI exposes:

- `ensure-six-subagents` for parent-level roster compliance checks
- `doctor` for schema/version checks, SQLite integrity checks, parent and child stale-risk reporting, missing-child violations, dead-PID lease reconciliation, and snapshot consistency inspection
- `repair` for non-destructive snapshot rebuilds and runtime visibility recovery

These commands operate on the canonical store. They do not delete live state and are safe to run while other sessions are active.
