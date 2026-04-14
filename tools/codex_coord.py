from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.codex_coord_lib import (  # noqa: E402
    CoordinationError,
    CoordinationStore,
    describe_runtime,
    load_repo_context,
    print_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repo-scoped Codex coordination control plane")
    parser.add_argument("--repo-root", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime_parser = subparsers.add_parser("runtime-path", help="Print the resolved runtime paths")
    runtime_parser.set_defaults(handler=lambda store, args: describe_runtime(store.context))

    start_parent = subparsers.add_parser("start-parent", help="Register a parent session")
    start_parent.add_argument("--session-id", default=None)
    start_parent.add_argument("--owner", default=None)
    start_parent.add_argument("--task-summary", required=True)
    start_parent.add_argument("--task-scope", default=None)
    start_parent.add_argument("--slots", type=int, default=6)
    start_parent.add_argument("--stale-after-seconds", type=int, default=300)
    start_parent.set_defaults(
        handler=lambda store, args: print_json(
            store.start_parent(
                task_summary=args.task_summary,
                session_id=args.session_id,
                owner=args.owner,
                task_scope=args.task_scope,
                slot_count=args.slots,
                stale_after_seconds=args.stale_after_seconds,
            )
        )
    )

    heartbeat = subparsers.add_parser("heartbeat", help="Refresh parent session liveness")
    heartbeat.add_argument("--session-id", required=True)
    heartbeat.add_argument("--note", default="")
    heartbeat.set_defaults(handler=lambda store, args: print_json(store.heartbeat(args.session_id, note=args.note)))

    start_child = subparsers.add_parser("start-child", help="Register a real child subagent under a parent session")
    start_child.add_argument("--parent-session-id", required=True)
    start_child.add_argument("--slot-id", required=True, type=int)
    start_child.add_argument("--session-id", "--child-session-id", dest="session_id", default=None)
    start_child.add_argument("--role", required=True)
    start_child.add_argument("--agent-name", required=True)
    start_child.add_argument("--agent-kind", default="")
    start_child.add_argument("--status", default="standby")
    start_child.add_argument("--summary", required=True)
    start_child.add_argument("--work-item-id", action="append", dest="work_item_ids", default=[])
    start_child.add_argument("--stale-after-seconds", type=int, default=None)
    start_child.add_argument("--task-summary", default=None)
    start_child.add_argument("--external-agent-id", default="")
    start_child.add_argument("--note", default="")
    start_child.set_defaults(
        handler=lambda store, args: print_json(
            store.start_child(
                parent_session_id=args.parent_session_id,
                slot_id=args.slot_id,
                child_session_id=args.session_id,
                role=args.role,
                agent_name=args.agent_name,
                agent_kind=args.agent_kind,
                activity_status=args.status,
                summary=args.summary,
                work_item_ids=args.work_item_ids,
                stale_after_seconds=args.stale_after_seconds,
                task_summary=args.task_summary,
                external_agent_id=args.external_agent_id,
                note=args.note,
            )
        )
    )

    heartbeat_child = subparsers.add_parser("heartbeat-child", help="Refresh one child subagent heartbeat")
    heartbeat_child.add_argument("--session-id", "--child-session-id", dest="session_id", required=True)
    heartbeat_child.add_argument("--note", default="")
    heartbeat_child.set_defaults(handler=lambda store, args: print_json(store.heartbeat_child(args.session_id, note=args.note)))

    update_child = subparsers.add_parser("update-child", help="Update a child subagent's role, status, or packet")
    update_child.add_argument("--session-id", "--child-session-id", dest="session_id", required=True)
    update_child.add_argument("--role", default=None)
    update_child.add_argument("--status", default=None)
    update_child.add_argument("--summary", default=None)
    update_child.add_argument("--agent-name", default=None)
    update_child.add_argument("--agent-kind", default=None)
    update_child.add_argument("--work-item-id", action="append", dest="work_item_ids", default=None)
    update_child.add_argument("--note", default="")
    update_child.set_defaults(
        handler=lambda store, args: print_json(
            store.update_child(
                child_session_id=args.session_id,
                role=args.role,
                activity_status=args.status,
                summary=args.summary,
                work_item_ids=args.work_item_ids,
                agent_name=args.agent_name,
                agent_kind=args.agent_kind,
                note=args.note,
            )
        )
    )

    note_child = subparsers.add_parser("note-child", help="Publish a child subagent note or handoff breadcrumb")
    note_child.add_argument("--session-id", "--child-session-id", dest="session_id", required=True)
    note_child.add_argument("--summary", required=True)
    note_child.add_argument("--category", default="note")
    note_child.add_argument("--evidence-path", action="append", default=[])
    note_child.set_defaults(
        handler=lambda store, args: print_json(
            store.note_child(
                child_session_id=args.session_id,
                summary=args.summary,
                category=args.category,
                evidence_paths=args.evidence_path,
            )
        )
    )

    end_child = subparsers.add_parser("end-child", help="End a child subagent session and clean its child-owned state")
    end_child.add_argument("--session-id", "--child-session-id", dest="session_id", required=True)
    end_child.add_argument("--outcome", default="completed")
    end_child.add_argument("--note", default="")
    end_child.add_argument("--release-work-to-parent", action=argparse.BooleanOptionalAction, default=True)
    end_child.set_defaults(
        handler=lambda store, args: print_json(
            store.end_child(
                child_session_id=args.session_id,
                outcome=args.outcome,
                note=args.note,
                release_work_to_parent=args.release_work_to_parent,
            )
        )
    )

    replace_child = subparsers.add_parser("replace-child", help="Replace one missing, stale, or exited child without tearing down the parent")
    replace_child.add_argument("--parent-session-id", required=True)
    replace_child.add_argument("--slot-id", required=True, type=int)
    replace_child.add_argument("--session-id", "--child-session-id", dest="session_id", default=None)
    replace_child.add_argument("--from-child-session-id", default=None)
    replace_child.add_argument("--role", required=True)
    replace_child.add_argument("--agent-name", required=True)
    replace_child.add_argument("--agent-kind", default="")
    replace_child.add_argument("--status", default="standby")
    replace_child.add_argument("--summary", required=True)
    replace_child.add_argument("--work-item-id", action="append", dest="work_item_ids", default=[])
    replace_child.add_argument("--external-agent-id", default="")
    replace_child.add_argument("--note", default="")
    replace_child.set_defaults(
        handler=lambda store, args: print_json(
            store.replace_child(
                parent_session_id=args.parent_session_id,
                slot_id=args.slot_id,
                child_session_id=args.session_id,
                from_child_session_id=args.from_child_session_id,
                role=args.role,
                agent_name=args.agent_name,
                agent_kind=args.agent_kind,
                activity_status=args.status,
                summary=args.summary,
                work_item_ids=args.work_item_ids,
                external_agent_id=args.external_agent_id,
                note=args.note,
            )
        )
    )

    ensure_six = subparsers.add_parser("ensure-six-subagents", help="Report whether a parent currently has 6 live registered child subagents")
    ensure_six.add_argument("--session-id", required=True)
    ensure_six.set_defaults(handler=lambda store, args: print_json(store.ensure_six_subagents(parent_session_id=args.session_id)))

    status = subparsers.add_parser("status", help="Render the current shared status board")
    status.add_argument("--format", choices=("markdown", "json"), default="markdown")
    status.set_defaults(
        handler=lambda store, args: print_json(store.status_json()) if args.format == "json" else store.status_text()
    )

    upsert_work_item = subparsers.add_parser("upsert-work-item", help="Add or update a work item in the shared ledger")
    upsert_work_item.add_argument("--session-id", required=True)
    upsert_work_item.add_argument("--work-item-id", required=True)
    upsert_work_item.add_argument("--title", required=True)
    upsert_work_item.add_argument("--source-ref", default="")
    upsert_work_item.add_argument("--status", default="open")
    upsert_work_item.add_argument("--note", default="")
    upsert_work_item.add_argument("--evidence", default="")
    upsert_work_item.set_defaults(
        handler=lambda store, args: print_json(
            store.upsert_work_item(
                session_id=args.session_id,
                work_item_id=args.work_item_id,
                title=args.title,
                source_ref=args.source_ref,
                status=args.status,
                note=args.note,
                evidence=args.evidence,
            )
        )
    )

    claim_work = subparsers.add_parser("claim-work", help="Exclusively claim a work item")
    claim_work.add_argument("--session-id", required=True)
    claim_work.add_argument("--work-item-id", required=True)
    claim_work.add_argument("--title", default=None)
    claim_work.add_argument("--source-ref", default="")
    claim_work.add_argument("--status", default="claimed")
    claim_work.add_argument("--slot-id", type=int, default=None)
    claim_work.add_argument("--note", default="")
    claim_work.add_argument("--evidence", default="")
    claim_work.set_defaults(
        handler=lambda store, args: print_json(
            store.claim_work(
                session_id=args.session_id,
                work_item_id=args.work_item_id,
                title=args.title,
                source_ref=args.source_ref,
                status=args.status,
                owner_slot=args.slot_id,
                note=args.note,
                evidence=args.evidence,
            )
        )
    )

    release_work = subparsers.add_parser("release-work", help="Release a work-item claim")
    release_work.add_argument("--session-id", required=True)
    release_work.add_argument("--work-item-id", required=True)
    release_work.add_argument("--status", default="open")
    release_work.add_argument("--note", default="")
    release_work.add_argument("--evidence", default="")
    release_work.set_defaults(
        handler=lambda store, args: print_json(
            store.release_work(
                session_id=args.session_id,
                work_item_id=args.work_item_id,
                status=args.status,
                note=args.note,
                evidence=args.evidence,
            )
        )
    )

    claim_file = subparsers.add_parser("claim-file", help="Exclusively claim a file or file section for writing")
    claim_file.add_argument("--session-id", required=True)
    claim_file.add_argument("--path", required=True)
    claim_file.add_argument("--section-id", default=None)
    claim_file.add_argument("--section-start-line", type=int, default=None)
    claim_file.add_argument("--section-end-line", type=int, default=None)
    claim_file.add_argument("--mode", default=None)
    claim_file.add_argument("--slot-id", type=int, default=None)
    claim_file.add_argument("--note", default="")
    claim_file.set_defaults(
        handler=lambda store, args: print_json(
            store.claim_file(
                session_id=args.session_id,
                path=args.path,
                section_id=args.section_id,
                section_start_line=args.section_start_line,
                section_end_line=args.section_end_line,
                mode=args.mode,
                owner_slot=args.slot_id,
                note=args.note,
            )
        )
    )

    release_file = subparsers.add_parser("release-file", help="Release a file claim")
    release_file.add_argument("--session-id", required=True)
    release_file.add_argument("--path", required=True)
    release_file.add_argument("--section-id", default=None)
    release_file.add_argument("--section-start-line", type=int, default=None)
    release_file.add_argument("--section-end-line", type=int, default=None)
    release_file.add_argument("--all-sections", action="store_true")
    release_file.set_defaults(
        handler=lambda store, args: print_json(
            store.release_file(
                session_id=args.session_id,
                path=args.path,
                section_id=args.section_id,
                section_start_line=args.section_start_line,
                section_end_line=args.section_end_line,
                all_sections=args.all_sections,
            )
        )
    )

    update_slot = subparsers.add_parser("update-slot", help="Compatibility shim that updates a slot projection or the active child in that slot")
    update_slot.add_argument("--session-id", required=True)
    update_slot.add_argument("--slot-id", required=True, type=int)
    update_slot.add_argument("--role", required=True)
    update_slot.add_argument("--status", required=True)
    update_slot.add_argument("--summary", required=True)
    update_slot.add_argument("--work-item-id", action="append", dest="work_item_ids", default=[])
    update_slot.set_defaults(
        handler=lambda store, args: print_json(
            store.update_slot(
                session_id=args.session_id,
                slot_id=args.slot_id,
                role=args.role,
                status=args.status,
                summary=args.summary,
                work_item_ids=args.work_item_ids,
            )
        )
    )

    post_message = subparsers.add_parser("post-message", help="Post a broadcast or session-targeted message")
    post_message.add_argument("--sender-session", required=True)
    post_message.add_argument("--recipient-session", default=None)
    post_message.add_argument("--category", default="note")
    post_message.add_argument("--subject", required=True)
    post_message.add_argument("--body", required=True)
    post_message.add_argument("--related-work-item-id", default=None)
    post_message.add_argument("--related-path", default=None)
    post_message.set_defaults(
        handler=lambda store, args: print_json(
            store.post_message(
                sender_session=args.sender_session,
                recipient_session=args.recipient_session,
                category=args.category,
                subject=args.subject,
                body=args.body,
                related_work_item_id=args.related_work_item_id,
                related_path=args.related_path,
            )
        )
    )

    ack_message = subparsers.add_parser("ack-message", help="Acknowledge an inbox message")
    ack_message.add_argument("--session-id", required=True)
    ack_message.add_argument("--message-id", required=True, type=int)
    ack_message.set_defaults(
        handler=lambda store, args: print_json(store.ack_message(session_id=args.session_id, message_id=args.message_id))
    )

    checkpoint = subparsers.add_parser("checkpoint", help="Publish a shared parent checkpoint")
    checkpoint.add_argument("--session-id", required=True)
    checkpoint.add_argument("--task-summary", default=None)
    checkpoint.add_argument("--blocker", action="append", default=[])
    checkpoint.add_argument("--next-action", action="append", default=[])
    checkpoint.add_argument("--evidence-path", action="append", default=[])
    checkpoint.add_argument("--note", default="")
    checkpoint.set_defaults(
        handler=lambda store, args: print_json(
            store.checkpoint(
                session_id=args.session_id,
                task_summary=args.task_summary,
                blockers=args.blocker,
                next_actions=args.next_action,
                evidence_paths=args.evidence_path,
                note=args.note,
            )
        )
    )

    open_lease = subparsers.add_parser("open-python-lease", help="Register a Python process lease")
    open_lease.add_argument("--session-id", required=True)
    open_lease.add_argument("--lease-id", default=None)
    open_lease.add_argument("--slot-id", type=int, default=None)
    open_lease.add_argument("--purpose", required=True)
    open_lease.add_argument("--command", required=True)
    open_lease.add_argument("--pid", type=int, default=None)
    open_lease.add_argument("--memory-cap-bytes", type=int, default=None)
    open_lease.add_argument("--memory-cap-percent", type=float, default=None)
    open_lease.add_argument("--enforcement-method", default="record_only")
    open_lease.add_argument("--status", default="open")
    open_lease.add_argument("--note", default="")
    open_lease.set_defaults(
        handler=lambda store, args: print_json(
            store.open_python_lease(
                session_id=args.session_id,
                lease_id=args.lease_id,
                owner_slot=args.slot_id,
                purpose=args.purpose,
                command=args.command,
                pid=args.pid,
                memory_cap_bytes=args.memory_cap_bytes,
                memory_cap_percent=args.memory_cap_percent,
                enforcement_method=args.enforcement_method,
                status=args.status,
                note=args.note,
            )
        )
    )

    close_lease = subparsers.add_parser("close-python-lease", help="Close a Python process lease")
    close_lease.add_argument("--session-id", required=True)
    close_lease.add_argument("--lease-id", required=True)
    close_lease.add_argument("--status", default="closed")
    close_lease.add_argument("--note", default="")
    close_lease.set_defaults(
        handler=lambda store, args: print_json(
            store.close_python_lease(
                session_id=args.session_id,
                lease_id=args.lease_id,
                status=args.status,
                note=args.note,
            )
        )
    )

    end_parent = subparsers.add_parser("end-parent", help="Write the final checkpoint and clean session state")
    end_parent.add_argument("--session-id", required=True)
    end_parent.add_argument("--outcome", default="completed")
    end_parent.add_argument("--task-summary", default=None)
    end_parent.add_argument("--blocker", action="append", default=[])
    end_parent.add_argument("--next-action", action="append", default=[])
    end_parent.add_argument("--evidence-path", action="append", default=[])
    end_parent.add_argument("--note", default="")
    end_parent.set_defaults(
        handler=lambda store, args: print_json(
            store.end_parent(
                session_id=args.session_id,
                outcome=args.outcome,
                task_summary=args.task_summary,
                blockers=args.blocker,
                next_actions=args.next_action,
                evidence_paths=args.evidence_path,
                note=args.note,
            )
        )
    )

    resume_parent = subparsers.add_parser("resume-parent", help="Create a new active parent session from a prior checkpoint")
    resume_parent.add_argument("--from-session-id", required=True)
    resume_parent.add_argument("--session-id", default=None)
    resume_parent.add_argument("--owner", default=None)
    resume_parent.add_argument("--stale-after-seconds", type=int, default=300)
    resume_parent.set_defaults(
        handler=lambda store, args: print_json(
            store.resume_parent(
                from_session_id=args.from_session_id,
                session_id=args.session_id,
                owner=args.owner,
                stale_after_seconds=args.stale_after_seconds,
            )
        )
    )

    reap = subparsers.add_parser("reap-stale", help="Reap stale parent sessions and abandoned claims")
    reap.add_argument("--requestor-session", required=True)
    reap.add_argument("--target-session", default=None)
    reap.add_argument("--takeover-session", default=None)
    reap.add_argument("--note", default="")
    reap.set_defaults(
        handler=lambda store, args: print_json(
            store.reap_stale(
                requestor_session=args.requestor_session,
                target_session=args.target_session,
                takeover_session=args.takeover_session,
                note=args.note,
            )
        )
    )

    doctor = subparsers.add_parser("doctor", help="Inspect store health, snapshots, and stale risks")
    doctor.set_defaults(handler=lambda store, args: print_json(store.doctor()))

    repair = subparsers.add_parser("repair", help="Safely rebuild snapshots and reconcile recoverable runtime drift")
    repair.set_defaults(handler=lambda store, args: print_json(store.repair()))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = CoordinationStore(load_repo_context(args.repo_root))
    try:
        output = args.handler(store, args)
    except CoordinationError as exc:
        print(f"coordination error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
