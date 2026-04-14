# Agent Roles

## Parent Responsibilities

The parent thread is the controller.
It owns:

- prompt parsing and work-item ledger creation
- claim and conflict coordination across the session family
- child spawning, assignment, and replacement
- parent checkpoints and final integration
- message handling
- verification routing
- cleanup

The parent should not become the main builder except for narrow mechanical integration.

## Six-Child Model

The default parent-controller pattern is one parent plus 6 real child subagents.
Slots remain parent-local child positions, but a slot only counts when a live registered child occupies it.

Recommended initial vocabulary:

1. Planner
2. Builder
3. QA
4. Docs Sync
5. Evaluator
6. Explorer or additional Builder/QA capacity depending on the task

Each child must keep its own heartbeat fresh, hold its own claims when it executes delegated work, and leave short notes the parent can verify quickly.

## Role Definitions

`planner`

- read-heavy
- decomposes work
- preserves or mints work-item IDs
- proposes claim-safe packets

`builder`

- makes the smallest defensible code or doc change
- owns file claims before editing
- avoids unrelated edits

`qa`

- verifies behavior and regressions
- runs focused checks first
- reopens work with evidence when needed

`docs_sync`

- updates maintained truth surfaces
- owns markdown claims before editing
- keeps implementation and documentation aligned

`evaluator`

- runs metrics, benchmarks, or targeted validation commands
- follows the low-memory Python lease policy
- records exact command and evidence

`repo_mapper`

- stays read-heavy
- maps code paths, commands, risky files, and docs surfaces
- supplies the parent with concrete impact areas

## How Future Prompts Should Think About Assignment

Future prompts should hand the parent a checklist or acceptance contract.
The parent then:

1. creates or syncs the work-item ledger
2. starts or updates 6 real child sessions and assigns explicit work-item IDs or standby/watch duties to each child
3. ensures the child that will execute a packet owns the required work and file claims
4. keeps `status`, `doctor`, and `ensure-six-subagents` green while the work is active
5. chooses the smallest verification surface that can close the packet
