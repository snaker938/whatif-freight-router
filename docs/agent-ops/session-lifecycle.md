# Session Lifecycle

## Startup

Every new parent session should do this in order:

1. Inspect the shared board with `python tools/codex_coord.py status`.
2. Inspect store health and invariant failures with `python tools/codex_coord.py doctor`.
3. Register the parent with `start-parent`, adding `--task-scope` when you intend to join an existing checklist namespace across parents.
4. Read inbox or broadcast messages from the status board or the shared messages snapshot.
5. Parse the prompt into work items, preserving provided IDs where they exist.
6. If the run is using parent-controller mode, register 6 real child subagents promptly with `start-child`, assign each child a work packet or explicit standby/watch duty, and confirm the roster with `ensure-six-subagents`.
7. Upsert or claim work before editing or launching heavy checks.

## Heartbeats

- Active parents should heartbeat with `heartbeat`.
- Active children should heartbeat their own child session with `heartbeat-child`.
- Refresh child heartbeats before long checks, after claim changes, and before a child becomes stale-risk.
- `heartbeat-child` updates the child session record and refreshes child-owned file-claim and open Python-lease heartbeats.

## Child Updates And Replacement

Use `update-child` whenever a live child changes:

- role
- activity status
- summary
- assigned work-item IDs

`update-slot` remains available as a compatibility shim that updates the slot projection or the active child currently attached to that slot.
Slots are parent-local child positions, not standalone workers.
The parent is responsible for keeping all 6 slots backed by live child sessions at all times.
If a child exits, goes stale, or becomes missing, use `replace-child` to restore compliance without tearing down the whole parent session.

## Checkpoints And Child Notes

Publish a parent checkpoint when:

- the task is decomposed
- a child changes phase
- a blocker appears or clears
- a long check starts or finishes
- the parent is about to end

Children should publish `note-child` breadcrumbs when they hit a meaningful status change, handoff, or evidence milestone.
Checkpoints and child notes are the shared-state mirror of what would otherwise be trapped inside one thread.

## Normal End

Normal shutdown is strict:

1. Write the final parent checkpoint.
2. End or deliberately replace any remaining child sessions.
3. Release all owned work-item claims.
4. Release all file claims.
5. Close all open Python leases.
6. Archive or clear messages tied to the session family.
7. Mark the parent `completed`, `interrupted`, `handed_off`, `reaped`, or `taken_over` as appropriate. Legacy aliases remain accepted and are normalized.
8. Remove the session from the active board.
9. Verify that no active child sessions, claims, or leases remain for that parent session.

The supported command is:

```powershell
python tools/codex_coord.py end-parent --session-id <session> --outcome completed
```

A parent may not finish with outcome `completed` while the 6-child roster is noncompliant.

## Interrupted Or Crashed Sessions

If a parent crashes and leaves active state behind:

1. Inspect the latest status snapshot and `doctor` output to confirm whether the parent, its children, or both are stale.
2. If the next parent should continue from the last durable checkpoint, prefer `resume-parent --from-session <old> --session-id <new>`.
3. Reap the stale source explicitly with `reap-stale` when its active claims or leases must be closed.
4. If another parent is taking over the task, pass `--takeover-session <new_session>`.
5. Restore the 6-child invariant with `start-child` or `replace-child` before taking new work.
6. Reclaim files explicitly before making edits.

Reaped and replaced sessions are archived so the shared board stays clear while the history remains visible.
