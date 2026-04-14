# Python Leases

## When To Register

Register a Python lease for any process that is:

- long-running
- memory-heavy
- benchmark-like
- evaluator-like
- or otherwise important enough that another session should be able to see it on the shared board

Short one-off standard-library commands do not need a lease.

## Preferred Launcher

Use the wrapper instead of launching Python directly for heavy work:

```powershell
python tools/codex_run_python.py --session-id <child_session> --purpose "targeted pytest slice" -- uv run --project backend pytest backend/tests/test_cost_model.py
```

When a child owns the work, pass the child session id. The board and snapshots will surface the child owner, parent session, and slot.

The wrapper:

- computes a default cap target of 5% of total RAM
- prefers a cgroup limit on POSIX when one is visible, so containerized runs do not inherit the host's full memory size by mistake
- records the lease in the shared coordination store
- forces common numeric-library thread counts to `1`
- applies the strongest practical memory limit it can find
- heartbeats the lease while the process runs
- closes the lease on normal exit or on handled termination signals

## Enforcement Methods

Best effort by platform:

- POSIX: `RLIMIT_AS` via `resource.setrlimit`, with cgroup memory limits preferred when available for reporting and cap calculation
- Windows: job object process-memory limit when available
- Fallback: record-only mode when OS enforcement is unavailable

The wrapper records the enforcement method actually used. It does not claim an enforced cap when only record-only mode was possible, and it leaves `memory_cap_percent` unset when total memory cannot be measured confidently.

## Lease Fields

Each lease stores:

- `lease_id`
- `owner_session`
- optional `owner_slot`
- `purpose`
- `command`
- `pid`
- `started_at`
- `last_heartbeat`
- `memory_cap_bytes`
- `memory_cap_percent`
- `enforcement_method`
- `status`
- `note`
- `closed_at`

Board and snapshot views also surface derived ownership context:

- `owner_parent_session`
- `owner_role`
- `owner_session_type`

The `note` field is the operator-facing breadcrumb. For wrapped launches it includes the launch context, the memory-cap summary, and the terminal exit or signal details when the lease closes.
`heartbeat-child` refreshes the `last_heartbeat` field for child-owned open leases.

## Clean Closure

The wrapper installs best-effort handlers for `SIGINT`, `SIGTERM`, and similar termination signals where the platform exposes them. It forwards the signal to the child, records the wrapper and child signal names in the lease note, and then closes the lease in `finally`.

The wrapper cannot cleanly recover `SIGKILL`, power loss, or a process that vanishes after launch without giving Python a chance to run cleanup. In those cases the parent still owns lease reconciliation and `end-child`, `replace-child`, `end-parent`, or a store-side reaper may need to close the stale record later.
