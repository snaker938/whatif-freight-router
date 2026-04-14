# Commands And Examples

## Start A Parent Session

```powershell
python tools/codex_coord.py start-parent --task-summary "Parse the prompt and initialize the work ledger" --task-scope release-audit
```

## Inspect The Shared Runtime Root

```powershell
python tools/codex_coord.py runtime-path
```

## Inspect Shared Status And Health

```powershell
python tools/codex_coord.py status
python tools/codex_coord.py status --format json
python tools/codex_coord.py doctor
python tools/codex_coord.py ensure-six-subagents --session-id <parent_session>
```

## Start And Maintain Real Child Subagents

Child-only lifecycle commands currently take the child session id via `--session-id`.

Register one real child in slot 2:

```powershell
python tools/codex_coord.py start-child --parent-session-id <parent_session> --slot-id 2 --session-id <child_session> --role Builder --agent-name Builder --agent-kind builder --status active --summary "Own TASK-01 implementation packet" --work-item-id TASK-01
```

Repeat `start-child` until all 6 slots are backed by real child sessions, then verify the roster:

```powershell
python tools/codex_coord.py ensure-six-subagents --session-id <parent_session>
```

Refresh liveness while active:

```powershell
python tools/codex_coord.py heartbeat --session-id <parent_session> --note "coordinating active child roster"
python tools/codex_coord.py heartbeat-child --session-id <child_session> --note "working on TASK-01"
```

Update or annotate a child:

```powershell
python tools/codex_coord.py update-child --session-id <child_session> --status blocked --summary "Waiting on fixture refresh" --work-item-id TASK-01 --note "blocker recorded"
python tools/codex_coord.py note-child --session-id <child_session> --category handoff --summary "Ready for QA review" --evidence-path docs/agent-ops/README.md
```

Replace one missing or stale child without tearing down the parent:

```powershell
python tools/codex_coord.py replace-child --parent-session-id <parent_session> --slot-id 2 --session-id <replacement_child_session> --from-child-session-id <old_child_session> --role Builder --agent-name Builder --agent-kind builder --status active --summary "Replacement child owns TASK-01" --work-item-id TASK-01
```

End a child cleanly when its packet is done:

```powershell
python tools/codex_coord.py end-child --session-id <child_session> --outcome completed --note "Packet closed and claims released"
```

## Register And Claim Work

Create or update the work item in the parent ledger, then claim it in the child that will execute it:

```powershell
python tools/codex_coord.py upsert-work-item --session-id <parent_session> --work-item-id TASK-01 --title "Sync backend and docs"
python tools/codex_coord.py claim-work --session-id <child_session> --work-item-id TASK-01 --status in_progress
```

Release a work-item claim when the packet is done or handed off:

```powershell
python tools/codex_coord.py release-work --session-id <child_session> --work-item-id TASK-01
```

## Claim A File Or Section

Claim files in the identity that will edit them:

```powershell
python tools/codex_coord.py claim-file --session-id <child_session> --path backend/app/main.py
python tools/codex_coord.py claim-file --session-id <child_session> --path docs/runbook.md --section-start-line 120 --section-end-line 200
```

Release a file claim when editing is finished:

```powershell
python tools/codex_coord.py release-file --session-id <child_session> --path docs/runbook.md --section-start-line 120 --section-end-line 200
```

`update-slot` remains available only as a compatibility shim for the parent-local slot projection. Prefer `update-child` for live child workflow changes.

## Post Or Acknowledge A Message

```powershell
python tools/codex_coord.py post-message --sender-session <child_session> --recipient-session <other_session> --category handoff --subject "Reclaim file after merge" --body "TASK-03 is ready for you."
python tools/codex_coord.py ack-message --session-id <other_session> --message-id 7
```

## Open A Python Lease

Direct lease registration:

```powershell
python tools/codex_coord.py open-python-lease --session-id <child_session> --purpose "one-off benchmark" --command "uv run --project backend pytest backend/tests/test_cost_model.py"
```

Preferred wrapped execution:

```powershell
python tools/codex_run_python.py --session-id <child_session> --purpose "focused backend pytest" -- uv run --project backend pytest backend/tests/test_cost_model.py
```

## Write A Checkpoint

```powershell
python tools/codex_coord.py checkpoint --session-id <parent_session> --task-summary "Mid-run sync" --blocker "waiting on route cache fixture" --next-action "rerun focused pytest slice" --evidence-path docs/agent-ops/README.md
```

## Resume Interrupted Work

```powershell
python tools/codex_coord.py resume-parent --from-session <old_session> --session-id <new_session>
python tools/codex_coord.py ensure-six-subagents --session-id <new_session>
```

## End A Session Cleanly

```powershell
python tools/codex_coord.py end-parent --session-id <parent_session> --outcome completed --note "Final checkpoint written and child cleanup complete"
```

## Reap Stale State

Release stale claims:

```powershell
python tools/codex_coord.py reap-stale --requestor-session <live_parent_session> --target-session <stale_session>
```

Reassign abandoned work items to a live session:

```powershell
python tools/codex_coord.py reap-stale --requestor-session <live_parent_session> --target-session <stale_session> --takeover-session <live_parent_session>
```

## Audit Or Repair The Runtime

```powershell
python tools/codex_coord.py doctor
python tools/codex_coord.py repair
```
