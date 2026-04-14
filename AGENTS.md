# Codex Operating Manual

## Purpose And Scope
This file is the durable repo-scoped operating manual for Codex work in this repository.
Future prompts provide the task-specific checklist, bug report, or acceptance contract.
This file defines the always-on workflow, coordination, cleanup, and verification rules that every future session must follow.

## Instruction Precedence
1. The user prompt and any task-specific checklist win on task scope, acceptance criteria, and provided work-item IDs.
2. This `AGENTS.md` defines the default operating model for all Codex sessions in this repo.
3. More specific nested instructions only override this file where they are intentionally narrower for a subdirectory or workflow.

## Mandatory Startup Sequence
1. Read the shared coordination board first: `python tools/codex_coord.py status`.
2. Inspect store health and child-roster compliance: `python tools/codex_coord.py doctor`.
3. Register the parent session before substantive work: `python tools/codex_coord.py start-parent --task-summary "<summary>" [--task-scope "<scope>"]`.
4. Inspect active parent sessions, active child subagents, claimed work items, claimed files, active Python leases, and inbox messages from the shared snapshots under the Git common dir.
5. Parse the incoming prompt into a work-item ledger. Preserve provided IDs exactly; otherwise mint stable task-local IDs before delegation.
6. If the session is using the parent-controller workflow, promptly spawn and register exactly 6 real child subagents with `start-child`, give each child a real packet or an explicit standby/watch packet, and confirm compliance with `python tools/codex_coord.py ensure-six-subagents --session-id <parent_session_id>`.
7. Upsert or claim work before editing files or launching heavy checks.

## Parent-Controller Model
The parent thread is a coordinator and integrator, not the main implementer.
The parent owns decomposition, child spawning, packet assignment, health monitoring, replacement, verification routing, integration, and final cleanup.
Delegated execution belongs to real child subagents. Each child owns its own work-item claims, file claims, heartbeats, status notes, and clean shutdown.
Always assume another parent session may already be active or may start later. Refresh the board before taking new work, editing new files, running long checks, or replacing a child.

## Six-Child Model
Parent-controller sessions use exactly 6 parent-local child slots backed by exactly 6 live child sessions from startup until shutdown.
Slot ids remain parent-local coordination positions, but they are only projections of real child threads; slot rows alone do not prove compliance.
Cross-session references should prefer `parent_session_id + child_session_id`, with `slot_id` retained as a convenience key for the child position under that parent.
Recommended role vocabulary: `Planner`, `Builder`, `QA`, `Docs Sync`, `Evaluator`, `Explorer`.
A live child must satisfy all of the following:
- child identity exists and is linked to the parent
- session status is active
- heartbeat is fresh
- current role is set
- current activity status is set
- assigned work-item ids are present, or the child is on an explicit standby/watch packet with a non-empty summary
`status`, `doctor`, and `ensure-six-subagents` are the canonical compliance checks.
The existence of `.codex/agents/*.toml` files or `max_threads = 6` in `.codex/config.toml` does not prove that 6 live children exist.
A parent-controller session may not report healthy completion while missing or unhealthy children remain.

## Coordination Rules
- No source edits without an exclusive file claim owned by the session that will make the edit.
- No two parent sessions should knowingly work the same claimed work item at once.
- Resolve conflicts through claims or messages, never by racing edits.
- Refresh the shared board at startup, before starting new work, before editing a file, before running long checks, before replacing a child, and at every checkpoint.
- Active parents must heartbeat with `heartbeat`; active children must heartbeat with `heartbeat-child`.
- Use `note-child`, `post-message`, and `ack-message` for handoffs, warnings, unblocks, and claim conflicts.

## Work-Item Ledger Rules
- Preserve task or checklist IDs exactly when they are provided.
- If the prompt has no IDs, mint stable local IDs before delegation and keep them consistent across checkpoints.
- Every delegated work packet must map to explicit work-item IDs or to an explicit standby/watch duty for the assigned child.
- Valid work-item states are `open`, `claimed`, `in_progress`, `blocked`, `qa`, and `closed`.
- In parent-controller mode, delegated execution claims should be owned by the responsible child session rather than left parent-owned.

## File Claim Rules
- File write claims are exclusive across active claim owners.
- Large markdown or report files may use section claims when work is clearly disjoint.
- Read-only inspection and QA review do not require a write claim.
- Use repo-relative normalized paths in claims; do not claim files outside the repo root.
- When a parent delegates editing work, the child that performs the edit must own the file claim before editing.

## Python Lease Rules
- Any long-running or meaningfully heavy Python process must be registered.
- Prefer `python tools/codex_run_python.py --session-id <id> --purpose "<why>" -- <python command>` so the lease, RAM cap, and thread limits are tracked together. When a child owns the work, pass the child session id.
- The default target cap is 5% of total RAM with numeric-library thread counts forced to `1`.
- If OS-level enforcement is unavailable, record that fact instead of pretending the cap was enforced.

## Verification And Done Definition
- Prefer the smallest valid check first, then escalate only as required.
- Backend verification surface: `uv run --project backend pytest ...`.
- Frontend verification surface: `pnpm --dir frontend exec tsc --noEmit` and `pnpm --dir frontend build`.
- Docs consistency surface: `python scripts/check_docs.py`.
- Repo runtime smoke surface: `.\scripts\dev.ps1` for the full local stack.
- Relevant checks, docs sync, child cleanup, claim release, lease closure, and a final checkpoint are part of done.

## Cleanup And Stale-State Handling
- Normal end-of-session flow is: final checkpoint, explicitly end or replace active children as needed, release work claims, release file claims, close Python leases, archive or clear messages, then `end-parent`.
- No finished parent-controller session may leave active child sessions, claims, or open leases behind.
- Stale state must be handled explicitly with `reap-stale`. Do not silently steal fresh claims.
- Use `resume-parent` when interrupted work should continue under a new parent session with durable checkpoint context. The resumed parent must restore itself to 6 live child subagents before taking new work.
- Use `replace-child` when one child exits, goes stale, or needs reassignment without tearing down the whole parent session.
- Stale takeover may reassign abandoned work items, but new child owners must reacquire file claims before editing.

## Repo-Specific Guardrails
- Main code and docs surfaces are `backend/`, `frontend/`, `docs/`, `scripts/`, and `docker-compose.yml`.
- Treat `.env`, `backend/assets/uk/`, `backend/data/raw/uk/`, `backend/out/`, `out/`, `osrm/data/`, `ors/data/`, `backend/uv.lock`, and `frontend/pnpm-lock.yaml` as sensitive or generated surfaces.
- CI is currently backend-only via `.github/workflows/backend-ci.yml`; do not assume frontend CI coverage exists if you change the UI.

## Quick References
- Coordination overview: `docs/agent-ops/README.md`
- Protocol details: `docs/agent-ops/coordination-protocol.md`
- Lifecycle and cleanup: `docs/agent-ops/session-lifecycle.md`
- Python lease policy: `docs/agent-ops/python-leases.md`
- Role definitions: `docs/agent-ops/agent-roles.md`
- Commands and examples: `docs/agent-ops/commands-and-examples.md`
- Core coordination implementation: `tools/codex_coord_lib.py`
- Runtime root discovery: `python tools/codex_coord.py runtime-path`
- Child-roster compliance check: `python tools/codex_coord.py ensure-six-subagents --session-id <parent_session_id>`
- Health check: `python tools/codex_coord.py doctor`
- Snapshot repair: `python tools/codex_coord.py repair`
