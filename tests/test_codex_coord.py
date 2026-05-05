from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COORD = ROOT / "tools" / "codex_coord.py"
RUN_PYTHON = ROOT / "tools" / "codex_run_python.py"


class CoordinationHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tempdir.name)
        subprocess.run(["git", "init"], cwd=self.repo_root, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "Codex Test"], cwd=self.repo_root, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "config", "user.email", "codex-test@example.com"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        (self.repo_root / "README.md").write_text("# temp repo\n", encoding="utf-8")
        (self.repo_root / "docs").mkdir()
        (self.repo_root / "docs" / "report.md").write_text("## Intro\n", encoding="utf-8")
        (self.repo_root / "src").mkdir()
        (self.repo_root / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.repo_root, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.runtime_root = self.repo_root / ".git" / "codex-coordination"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_coord(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.run_coord_at(self.repo_root, *args, check=check)

    def run_coord_at(
        self, repo_root: Path, *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, str(COORD), "--repo-root", str(repo_root), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if check and completed.returncode != 0:
            self.fail(f"coord failed: {' '.join(args)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
        return completed

    def run_python_wrapper(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, str(RUN_PYTHON), "--repo-root", str(self.repo_root), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if check and completed.returncode != 0:
            self.fail(
                f"python wrapper failed: {' '.join(args)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed

    def coord_json(self, *args: str) -> dict:
        return json.loads(self.run_coord(*args).stdout)

    def read_json(self, *parts: str) -> dict:
        return json.loads((self.runtime_root.joinpath(*parts)).read_text(encoding="utf-8"))

    def read_text(self, *parts: str) -> str:
        return self.runtime_root.joinpath(*parts).read_text(encoding="utf-8")

    def top_level_help(self) -> str:
        return self.run_coord("--help").stdout

    def command_help(self, *command: str) -> str:
        return self.run_coord(*command, "--help").stdout

    def runtime_path_details(self, repo_root: Path | None = None) -> dict[str, str]:
        completed = self.run_coord_at(repo_root or self.repo_root, "runtime-path")
        details: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            details[key.strip()] = value.strip()
        return details

    def start_parent(
        self,
        session_id: str,
        summary: str,
        *,
        stale_after_seconds: int = 300,
        task_scope: str | None = None,
    ) -> dict:
        args = [
            "start-parent",
            "--session-id",
            session_id,
            "--task-summary",
            summary,
            "--stale-after-seconds",
            str(stale_after_seconds),
        ]
        if task_scope:
            args.extend(["--task-scope", task_scope])
        return self.coord_json(*args)

    def start_child(
        self,
        parent_session_id: str,
        slot_id: int,
        session_id: str,
        *,
        role: str = "Evaluator",
        agent_name: str = "Evaluator",
        agent_kind: str = "evaluator",
        status: str = "standby",
        summary: str = "Standing by",
        work_item_ids: list[str] | None = None,
        note: str = "",
    ) -> dict:
        args = [
            "start-child",
            "--parent-session-id",
            parent_session_id,
            "--slot-id",
            str(slot_id),
            "--session-id",
            session_id,
            "--role",
            role,
            "--agent-name",
            agent_name,
            "--agent-kind",
            agent_kind,
            "--status",
            status,
            "--summary",
            summary,
        ]
        for work_item_id in work_item_ids or []:
            args.extend(["--work-item-id", work_item_id])
        if note:
            args.extend(["--note", note])
        return self.coord_json(*args)

    def register_full_child_roster(self, parent_session_id: str, *, prefix: str = "child") -> list[str]:
        child_ids: list[str] = []
        for slot_id in range(1, 7):
            child_session_id = f"{prefix}-{slot_id}"
            child_ids.append(child_session_id)
            self.start_child(
                parent_session_id,
                slot_id,
                child_session_id,
                role=f"Role-{slot_id}",
                agent_name=f"Agent-{slot_id}",
                agent_kind=f"kind-{slot_id}",
                status="standby",
                summary=f"Standing by in slot {slot_id}",
            )
        return child_ids

    def test_starting_two_parent_sessions_renders_the_active_board(self) -> None:
        self.start_parent("parent-a", "Map the first task")
        self.start_parent("parent-b", "Map the second task")

        board = self.run_coord("status").stdout
        active_sessions = self.read_json("snapshots", "active_sessions.json")

        self.assertIn("parent-a", board)
        self.assertIn("parent-b", board)
        self.assertEqual(2, len(active_sessions["sessions"]))

    def test_two_git_worktrees_share_the_same_runtime_and_board(self) -> None:
        worktree_tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(worktree_tempdir.cleanup)
        worktree_root = Path(worktree_tempdir.name) / "shared-worktree"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_root), "HEAD"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

        primary_runtime = self.runtime_path_details(self.repo_root)
        worktree_runtime = self.runtime_path_details(worktree_root)

        self.assertEqual(primary_runtime["git_common_dir"], worktree_runtime["git_common_dir"])
        self.assertEqual(primary_runtime["runtime_root"], worktree_runtime["runtime_root"])

        self.run_coord_at(self.repo_root, "start-parent", "--session-id", "parent-a", "--task-summary", "Primary worktree")
        self.run_coord_at(worktree_root, "start-parent", "--session-id", "parent-b", "--task-summary", "Secondary worktree")

        primary_board = self.run_coord_at(self.repo_root, "status").stdout
        worktree_board = self.run_coord_at(worktree_root, "status").stdout

        self.assertIn("parent-a", primary_board)
        self.assertIn("parent-b", primary_board)
        self.assertIn("parent-a", worktree_board)
        self.assertIn("parent-b", worktree_board)

    def test_conflicting_work_item_claims_allow_only_one_owner(self) -> None:
        self.start_parent("parent-a", "First parent", task_scope="shared-scope")
        self.start_parent("parent-b", "Second parent", task_scope="shared-scope")
        self.run_coord(
            "claim-work",
            "--session-id",
            "parent-a",
            "--work-item-id",
            "TASK-01",
            "--title",
            "Shared task",
        )

        blocked = self.run_coord(
            "claim-work",
            "--session-id",
            "parent-b",
            "--work-item-id",
            "TASK-01",
            "--title",
            "Shared task",
            check=False,
        )

        claims = self.read_json("snapshots", "work_item_claims.json")
        self.assertNotEqual(0, blocked.returncode)
        self.assertIn("already claimed", blocked.stderr)
        self.assertEqual("parent-a", claims["claimed_work_items"][0]["owner_session"])

    def test_conflicting_file_claims_allow_only_one_writer(self) -> None:
        self.start_parent("parent-a", "First parent")
        self.start_parent("parent-b", "Second parent")
        self.run_coord("claim-file", "--session-id", "parent-a", "--path", "src/main.py")

        blocked = self.run_coord(
            "claim-file",
            "--session-id",
            "parent-b",
            "--path",
            "src/main.py",
            check=False,
        )

        claims = self.read_json("snapshots", "file_claims.json")
        self.assertNotEqual(0, blocked.returncode)
        self.assertIn("file claim conflict", blocked.stderr)
        self.assertEqual("parent-a", claims["file_claims"][0]["owner_session"])

    def test_path_spellings_normalize_to_one_canonical_file_claim(self) -> None:
        self.start_parent("parent-a", "First parent")
        self.start_parent("parent-b", "Second parent")
        self.run_coord("claim-file", "--session-id", "parent-a", "--path", "src/main.py")

        relative_blocked = self.run_coord(
            "claim-file",
            "--session-id",
            "parent-b",
            "--path",
            r"src\main.py",
            check=False,
        )
        absolute_blocked = self.run_coord(
            "claim-file",
            "--session-id",
            "parent-b",
            "--path",
            str(self.repo_root / "src" / "main.py"),
            check=False,
        )

        claims = self.read_json("snapshots", "file_claims.json")
        self.assertNotEqual(0, relative_blocked.returncode)
        self.assertNotEqual(0, absolute_blocked.returncode)
        self.assertEqual("src/main.py", claims["file_claims"][0]["path"])

    def test_task_scope_namespaces_identical_raw_work_item_ids_when_supported(self) -> None:
        if "--task-scope" not in self.command_help("start-parent"):
            self.skipTest("task-scope support is not available yet")

        self.run_coord(
            "start-parent",
            "--session-id",
            "parent-a",
            "--task-summary",
            "First task",
            "--task-scope",
            "scope-a",
        )
        self.run_coord(
            "start-parent",
            "--session-id",
            "parent-b",
            "--task-summary",
            "Second task",
            "--task-scope",
            "scope-b",
        )
        self.run_coord(
            "claim-work",
            "--session-id",
            "parent-a",
            "--work-item-id",
            "TASK-01",
            "--title",
            "Scoped task",
        )
        second_claim = self.run_coord(
            "claim-work",
            "--session-id",
            "parent-b",
            "--work-item-id",
            "TASK-01",
            "--title",
            "Scoped task",
            check=False,
        )

        self.assertEqual(0, second_claim.returncode)
        work_snapshot = self.read_json("snapshots", "work_item_claims.json")
        self.assertEqual(2, len(work_snapshot["claimed_work_items"]))
        self.assertEqual({"scope-a", "scope-b"}, {item["task_scope"] for item in work_snapshot["claimed_work_items"]})

    def test_resume_parent_command_is_listed_before_we_depend_on_it(self) -> None:
        if "resume-parent" not in self.top_level_help():
            self.skipTest("resume-parent support is not available yet")
        self.assertIn("resume-parent", self.top_level_help())

    def test_child_lifecycle_commands_register_heartbeat_note_update_and_end(self) -> None:
        self.start_parent("parent-a", "Child lifecycle")
        self.start_child(
            "parent-a",
            1,
            "child-a",
            role="Evaluator",
            agent_name="Verifier",
            agent_kind="qa",
            status="standby",
            summary="Standing by for the next packet",
        )
        self.run_coord("heartbeat-child", "--session-id", "child-a", "--note", "child alive")
        self.run_coord(
            "note-child",
            "--session-id",
            "child-a",
            "--category",
            "status",
            "--summary",
            "Child heartbeat recorded",
        )
        self.run_coord(
            "update-child",
            "--session-id",
            "child-a",
            "--role",
            "Evaluator",
            "--status",
            "active",
            "--summary",
            "Working packet TASK-01",
            "--agent-name",
            "Verifier",
            "--agent-kind",
            "qa",
            "--work-item-id",
            "TASK-01",
        )
        pre_end_snapshot = self.coord_json("status", "--format", "json")
        self.run_coord("end-child", "--session-id", "child-a", "--outcome", "completed", "--note", "child done")

        child_snapshot = self.read_json("snapshots", "sessions", "child-a.json")
        archive_text = self.read_text("archive", "child-a", "final.md")

        self.assertEqual("child", child_snapshot["session"]["session_type"])
        self.assertEqual("completed", child_snapshot["session"]["status"])
        self.assertEqual("active", child_snapshot["session"]["activity_status"])
        self.assertEqual(["TASK-01"], pre_end_snapshot["sessions"]["child-a"]["session"]["work_item_ids"])
        self.assertEqual("status", pre_end_snapshot["sessions"]["child-a"]["latest_child_note"]["category"])
        self.assertEqual("Child heartbeat recorded", pre_end_snapshot["sessions"]["child-a"]["latest_child_note"]["summary"])
        self.assertEqual("end", child_snapshot["latest_child_note"]["category"])
        self.assertEqual("child done", child_snapshot["latest_child_note"]["summary"])
        self.assertIn("Child lifecycle", archive_text)
        self.assertIn("child-a", archive_text)

    def test_child_owned_work_and_file_claims_are_visible_under_child_identity(self) -> None:
        self.start_parent("parent-a", "Child ownership")
        self.start_child("parent-a", 1, "child-a", role="Evaluator", agent_name="Verifier", summary="Standing by")

        self.run_coord(
            "claim-work",
            "--session-id",
            "child-a",
            "--work-item-id",
            "TASK-01",
            "--title",
            "Child task",
        )
        self.run_coord("claim-file", "--session-id", "child-a", "--path", "src/main.py")

        snapshot = self.coord_json("status", "--format", "json")
        child_snapshot = snapshot["sessions"]["child-a"]

        self.assertEqual("child", child_snapshot["session"]["session_type"])
        self.assertEqual("child-a", snapshot["claimed_work_items"][0]["owner_session"])
        self.assertEqual("child", snapshot["claimed_work_items"][0]["owner_session_type"])
        self.assertEqual("child-a", snapshot["file_claims"][0]["owner_session"])
        self.assertEqual("child", snapshot["file_claims"][0]["owner_session_type"])
        self.assertEqual("child-a", snapshot["active_child_sessions"][0]["session"]["session_id"])
        self.assertEqual(1, len(child_snapshot["work_items"]))
        self.assertEqual("child-a", child_snapshot["work_items"][0]["owner_session"])
        self.assertEqual(1, len(child_snapshot["file_claims"]))

    def test_ensure_six_subagents_reports_missing_children_and_doctor_surfaces_noncompliance(self) -> None:
        self.start_parent("parent-a", "Real six invariant")
        self.start_child("parent-a", 1, "child-a", role="Evaluator", agent_name="Verifier", summary="Standing by")

        ensure_six = self.coord_json("ensure-six-subagents", "--session-id", "parent-a")
        board = self.run_coord("status").stdout
        doctor = self.coord_json("doctor")

        self.assertEqual(6, ensure_six["required_child_count"])
        self.assertEqual("noncompliant", ensure_six["child_compliance"])
        self.assertEqual(1, ensure_six["live_child_count"])
        self.assertEqual(5, ensure_six["missing_child_count"])
        self.assertEqual([2, 3, 4, 5, 6], ensure_six["missing_child_slots"])
        self.assertIn("| Slot | Child | Agent | Role | Status | Health | Work Items | Summary |", board)
        self.assertIn("Missing child count", board)
        self.assertEqual(1, doctor["noncompliant_parent_count"])
        self.assertEqual(1, len(doctor["missing_children"]))
        self.assertEqual("parent-a", doctor["missing_children"][0]["session_id"])
        self.assertEqual(5, doctor["missing_children"][0]["missing_child_count"])
        self.assertEqual([2, 3, 4, 5, 6], doctor["missing_children"][0]["missing_child_slots"])
        self.assertEqual(1, len(doctor["parent_child_invariant_violations"]))

    def test_replace_child_and_resume_restore_or_require_the_child_roster(self) -> None:
        self.start_parent("parent-a", "Replacement and resume", task_scope="resume-scope")
        self.register_full_child_roster("parent-a")
        self.run_coord("end-child", "--session-id", "child-6", "--outcome", "completed", "--note", "slot six completed")

        missing = self.coord_json("ensure-six-subagents", "--session-id", "parent-a")
        self.assertEqual(5, missing["live_child_count"])
        self.assertIn(6, missing["missing_child_slots"])

        replacement = self.coord_json(
            "replace-child",
            "--parent-session-id",
            "parent-a",
            "--slot-id",
            "6",
            "--session-id",
            "child-6-replacement",
            "--from-child-session-id",
            "child-6",
            "--role",
            "Evaluator",
            "--agent-name",
            "Bench",
            "--agent-kind",
            "evaluator",
            "--status",
            "standby",
            "--summary",
            "Replacement child for slot six",
        )
        restored = self.coord_json("ensure-six-subagents", "--session-id", "parent-a")

        self.run_coord("checkpoint", "--session-id", "parent-a", "--task-summary", "Replacement and resume", "--next-action", "resume later")
        self.run_coord("end-parent", "--session-id", "parent-a", "--outcome", "interrupted")

        resumed = self.coord_json("resume-parent", "--from-session-id", "parent-a", "--session-id", "parent-b")
        resumed_health = self.coord_json("ensure-six-subagents", "--session-id", "parent-b")
        resumed_snapshot = self.read_json("snapshots", "sessions", "parent-b.json")

        self.assertEqual("child-6-replacement", replacement["session_id"])
        self.assertEqual("compliant", restored["child_compliance"])
        self.assertEqual(6, restored["live_child_count"])
        self.assertEqual("parent-b", resumed["session_id"])
        self.assertEqual(6, resumed["required_child_count"])
        self.assertEqual("noncompliant", resumed_health["child_compliance"])
        self.assertEqual(0, resumed_health["live_child_count"])
        self.assertEqual(6, resumed_health["missing_child_count"])
        self.assertEqual([1, 2, 3, 4, 5, 6], resumed_health["missing_child_slots"])
        self.assertEqual(6, len(resumed_snapshot["slots"]))
        self.assertEqual("missing_child", resumed_snapshot["slots"][0]["status"])

    def test_end_parent_cascades_child_cleanup_and_archives_children(self) -> None:
        self.start_parent("parent-a", "Cleanup cascade")
        self.register_full_child_roster("parent-a", prefix="child-cleanup")
        self.run_coord(
            "claim-work",
            "--session-id",
            "child-cleanup-1",
            "--work-item-id",
            "TASK-01",
            "--title",
            "Child-owned task",
        )
        self.run_coord("claim-file", "--session-id", "child-cleanup-1", "--path", "src/main.py")

        self.run_coord("end-parent", "--session-id", "parent-a", "--outcome", "finished")

        active_sessions = self.read_json("snapshots", "active_sessions.json")
        parent_archive = self.read_text("archive", "parent-a", "final.md")
        child_archive = self.read_text("archive", "child-cleanup-1", "final.md")

        self.assertEqual([], active_sessions["sessions"])
        self.assertIn("Child Roster", parent_archive)
        self.assertIn("child-cleanup-1", parent_archive)
        self.assertIn("child-cleanup-1", child_archive)
        self.assertTrue((self.runtime_root / "archive" / "child-cleanup-1" / "final.md").is_file())

    def test_section_overlap_claims_are_rejected_when_range_claims_exist(self) -> None:
        help_text = self.command_help("claim-file")
        if "--section-start-line" not in help_text and "--section-end-line" not in help_text:
            self.skipTest("section-range support is not available yet")

        self.start_parent("parent-a", "First parent")
        self.start_parent("parent-b", "Second parent")
        self.run_coord(
            "claim-file",
            "--session-id",
            "parent-a",
            "--path",
            "docs/report.md",
            "--section-start-line",
            "1",
            "--section-end-line",
            "10",
        )
        blocked = self.run_coord(
            "claim-file",
            "--session-id",
            "parent-b",
            "--path",
            "docs/report.md",
            "--section-start-line",
            "8",
            "--section-end-line",
            "12",
            check=False,
        )

        self.assertNotEqual(0, blocked.returncode)

    def test_doctor_and_repair_commands_are_listed_when_available(self) -> None:
        help_text = self.top_level_help()
        if "doctor" not in help_text or "repair" not in help_text:
            self.skipTest("doctor/repair support is not available yet")
        self.assertIn("doctor", help_text)
        self.assertIn("repair", help_text)

    def test_normal_end_releases_work_and_file_claims(self) -> None:
        self.start_parent("parent-a", "Own and release work")
        self.register_full_child_roster("parent-a")
        self.run_coord("claim-work", "--session-id", "parent-a", "--work-item-id", "TASK-01", "--title", "Owned task")
        self.run_coord("claim-file", "--session-id", "parent-a", "--path", "src/main.py")
        self.run_coord("end-parent", "--session-id", "parent-a", "--outcome", "finished")

        work_snapshot = self.read_json("snapshots", "work_item_claims.json")
        file_snapshot = self.read_json("snapshots", "file_claims.json")
        session_snapshot = self.read_json("snapshots", "sessions", "parent-a.json")

        self.assertEqual([], work_snapshot["claimed_work_items"])
        self.assertEqual([], file_snapshot["file_claims"])
        self.assertEqual("completed", session_snapshot["session"]["status"])
        self.assertEqual("open", session_snapshot["work_items"][0]["status"])

    def test_stale_session_takeover_reassigns_work_and_reaps_file_claims(self) -> None:
        self.start_parent("parent-a", "Soon stale", stale_after_seconds=1)
        self.start_parent("parent-b", "Live parent")
        self.run_coord("claim-work", "--session-id", "parent-a", "--work-item-id", "TASK-01", "--title", "Takeover me")
        self.run_coord("claim-file", "--session-id", "parent-a", "--path", "src/main.py")

        time.sleep(2)
        self.run_coord(
            "reap-stale",
            "--requestor-session",
            "parent-b",
            "--target-session",
            "parent-a",
            "--takeover-session",
            "parent-b",
        )

        work_snapshot = self.read_json("snapshots", "work_item_claims.json")
        file_snapshot = self.read_json("snapshots", "file_claims.json")
        archive_text = self.read_text("archive", "parent-a", "final.md")

        self.assertEqual("parent-b", work_snapshot["claimed_work_items"][0]["owner_session"])
        self.assertEqual([], file_snapshot["file_claims"])
        self.assertIn("taken_over", archive_text)
        self.assertIn("Reaped by: `parent-b`", archive_text)

    def test_end_parent_is_idempotent_and_preserves_final_cleanup(self) -> None:
        self.start_parent("parent-a", "Idempotent cleanup")
        self.register_full_child_roster("parent-a")
        self.run_coord("claim-work", "--session-id", "parent-a", "--work-item-id", "TASK-01", "--title", "Cleanup task")
        self.run_coord("claim-file", "--session-id", "parent-a", "--path", "src/main.py")
        self.run_coord("end-parent", "--session-id", "parent-a", "--outcome", "finished")

        first_archive = self.read_text("archive", "parent-a", "final.md")
        second = self.run_coord("end-parent", "--session-id", "parent-a", "--outcome", "finished", check=False)

        self.assertEqual(0, second.returncode)
        self.assertEqual(first_archive, self.read_text("archive", "parent-a", "final.md"))
        self.assertEqual([], self.read_json("snapshots", "active_sessions.json")["sessions"])

    def test_python_lease_open_close_bookkeeping_via_wrapper(self) -> None:
        self.start_parent("parent-a", "Run python")
        completed = self.run_python_wrapper(
            "--session-id",
            "parent-a",
            "--purpose",
            "tiny command",
            "--",
            sys.executable,
            "-c",
            "print('ok')",
        )

        session_snapshot = self.read_json("snapshots", "sessions", "parent-a.json")
        active_leases = self.read_json("snapshots", "python_leases.json")

        self.assertEqual(0, completed.returncode)
        self.assertEqual([], active_leases["python_leases"])
        self.assertEqual("completed", session_snapshot["python_leases"][0]["status"])
        self.assertIsNotNone(session_snapshot["python_leases"][0]["closed_at"])

    def test_reap_stale_is_idempotent_for_an_already_reaped_session(self) -> None:
        self.start_parent("parent-a", "Soon stale", stale_after_seconds=1)
        self.start_parent("parent-b", "Live parent")
        self.run_coord("claim-work", "--session-id", "parent-a", "--work-item-id", "TASK-01", "--title", "Takeover me")
        self.run_coord("claim-file", "--session-id", "parent-a", "--path", "src/main.py")

        time.sleep(2)
        first = self.run_coord(
            "reap-stale",
            "--requestor-session",
            "parent-b",
            "--target-session",
            "parent-a",
            "--takeover-session",
            "parent-b",
        )
        second = self.run_coord(
            "reap-stale",
            "--requestor-session",
            "parent-b",
            "--target-session",
            "parent-a",
            "--takeover-session",
            "parent-b",
            check=False,
        )

        self.assertEqual(0, first.returncode)
        self.assertEqual(0, second.returncode)
        self.assertIn("taken_over", self.read_text("archive", "parent-a", "final.md"))
        self.assertIn("\"already_terminal\"", second.stdout)

    def test_resume_parent_rehydrates_a_new_session_from_checkpoint_context(self) -> None:
        self.start_parent("parent-a", "Interrupted task", task_scope="resume-scope")
        self.run_coord(
            "update-slot",
            "--session-id",
            "parent-a",
            "--slot-id",
            "2",
            "--role",
            "Builder",
            "--status",
            "active",
            "--summary",
            "Editing TASK-01",
            "--work-item-id",
            "TASK-01",
        )
        self.run_coord(
            "checkpoint",
            "--session-id",
            "parent-a",
            "--task-summary",
            "Interrupted task",
            "--next-action",
            "resume the builder lane",
            "--evidence-path",
            "src/main.py",
        )
        self.run_coord("end-parent", "--session-id", "parent-a", "--outcome", "interrupted")

        resumed = self.coord_json(
            "resume-parent",
            "--from-session-id",
            "parent-a",
            "--session-id",
            "parent-b",
        )
        session_snapshot = self.read_json("snapshots", "sessions", "parent-b.json")

        self.assertEqual("parent-b", resumed["session_id"])
        self.assertEqual("resume-scope", resumed["task_scope"])
        self.assertEqual("active", session_snapshot["session"]["status"])
        self.assertEqual("parent-a", session_snapshot["session"]["resume_from_session"])
        self.assertIn("Resumed from parent-a", session_snapshot["slots"][1]["summary"])

    def test_doctor_and_repair_rebuild_snapshot_manifest_and_report_store_health(self) -> None:
        self.start_parent("parent-a", "Doctor test")
        self.run_coord("status")

        manifest_path = self.runtime_root / "snapshots" / "snapshot_manifest.json"
        manifest_path.unlink()

        doctor_before = self.coord_json("doctor")
        repair = self.coord_json("repair")
        doctor_after = self.coord_json("doctor")

        self.assertIn("snapshot manifest is missing", doctor_before["snapshot_issues"])
        self.assertTrue(manifest_path.is_file())
        self.assertEqual([], doctor_after["snapshot_issues"])
        self.assertIn("snapshot_generation", repair)

    def test_status_board_rendering_includes_claims_messages_and_slots(self) -> None:
        self.start_parent("parent-a", "Render board")
        self.run_coord("claim-work", "--session-id", "parent-a", "--work-item-id", "TASK-01", "--title", "Visible task")
        self.run_coord("claim-file", "--session-id", "parent-a", "--path", "docs/report.md", "--section-id", "intro")
        self.run_coord(
            "update-slot",
            "--session-id",
            "parent-a",
            "--slot-id",
            "2",
            "--role",
            "Builder",
            "--status",
            "active",
            "--summary",
            "Working on TASK-01",
            "--work-item-id",
            "TASK-01",
        )
        self.run_coord(
            "post-message",
            "--sender-session",
            "parent-a",
            "--category",
            "note",
            "--subject",
            "Visible note",
            "--body",
            "This should appear on the board.",
        )

        board = self.read_text("snapshots", "status.md")

        self.assertIn("Active Parent Sessions", board)
        self.assertIn("Claimed Work Items", board)
        self.assertIn("File Claims", board)
        self.assertIn("Messages", board)
        self.assertIn("TASK-01", board)
        self.assertIn("Visible note", board)
        self.assertIn("parent-a", board)

    def test_cleanup_removes_finished_session_from_active_state(self) -> None:
        self.start_parent("parent-a", "Cleanup test")
        self.register_full_child_roster("parent-a")
        self.run_coord("claim-file", "--session-id", "parent-a", "--path", "src/main.py")
        self.run_coord(
            "open-python-lease",
            "--session-id",
            "parent-a",
            "--purpose",
            "hold lease",
            "--command",
            "python -c pass",
        )
        self.run_coord("end-parent", "--session-id", "parent-a", "--outcome", "finished")

        active_sessions = self.read_json("snapshots", "active_sessions.json")
        file_snapshot = self.read_json("snapshots", "file_claims.json")
        lease_snapshot = self.read_json("snapshots", "python_leases.json")

        self.assertEqual([], active_sessions["sessions"])
        self.assertEqual([], file_snapshot["file_claims"])
        self.assertEqual([], lease_snapshot["python_leases"])
        self.assertTrue((self.runtime_root / "archive" / "parent-a" / "final.md").is_file())


class DocsCheckerHarness(unittest.TestCase):
    @staticmethod
    def load_check_docs_module():
        spec = importlib.util.spec_from_file_location("check_docs_under_test", ROOT / "scripts" / "check_docs.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_agent_ops_docs_are_included_but_not_forced_into_top_level_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            docs_dir = root / "docs"
            agent_ops_dir = docs_dir / "agent-ops"
            agent_ops_dir.mkdir(parents=True)
            (root / "README.md").write_text("# temp repo\n", encoding="utf-8")
            (docs_dir / "DOCS_INDEX.md").write_text(
                "\n".join(
                    [
                        "# Documentation Index",
                        "",
                        "## Getting Started",
                        "",
                        "- [Root README](../README.md)",
                        "- [Codex Agent Ops](agent-ops/README.md)",
                        "",
                        "## Related Docs",
                        "",
                        "- [Root README](../README.md)",
                        "- [Codex Agent Ops](agent-ops/README.md)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (agent_ops_dir / "README.md").write_text("# Agent Ops\n", encoding="utf-8")
            (agent_ops_dir / "commands-and-examples.md").write_text("# Commands\n", encoding="utf-8")
            (agent_ops_dir / "session-lifecycle.md").write_text("# Lifecycle\n", encoding="utf-8")

            module = self.load_check_docs_module()
            module.ROOT = root
            module.DOCS_DIR = docs_dir
            module.AGENT_OPS_DIR = agent_ops_dir
            module.ROOT_README = root / "README.md"
            module.DOC_INDEX = docs_dir / "DOCS_INDEX.md"
            module.BACKEND_README = root / "backend" / "README.md"
            module.FRONTEND_README = root / "frontend" / "README.md"
            module.ROOT_CLAIM_MATRIX = root / "claim_matrix.md"
            module.DOCS_CLAIM_MATRIX = docs_dir / "claim_matrix.md"
            module.THEOREM_MAP = docs_dir / "theorem_map.md"
            module.BACKEND_MAIN = root / "backend" / "app" / "main.py"
            module.BACKEND_DOC = docs_dir / "backend-api-tools.md"
            module.RUN_STORE = root / "backend" / "app" / "run_store.py"
            module.EXTRA_MAINTAINED_MARKDOWN = ()

            maintained = module.list_maintained_markdown()

            self.assertIn(agent_ops_dir / "README.md", maintained)
            self.assertIn(agent_ops_dir / "commands-and-examples.md", maintained)
            self.assertIn(agent_ops_dir / "session-lifecycle.md", maintained)
            self.assertEqual([], module.run_orphan_check())


class AtomicWriteHarness(unittest.TestCase):
    def test_atomic_write_text_retries_transient_replace_permission_errors(self) -> None:
        from tools import codex_coord_lib as coord_lib

        with tempfile.TemporaryDirectory() as tempdir:
            target = Path(tempdir) / "snapshots" / "messages.json"
            real_replace = coord_lib.os.replace
            attempts = {"count": 0}

            def flaky_replace(src: str | Path, dst: str | Path) -> None:
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise PermissionError("transient sharing violation")
                real_replace(src, dst)

            with (
                mock.patch.object(coord_lib.os, "replace", side_effect=flaky_replace),
                mock.patch.object(coord_lib.time, "sleep", return_value=None) as sleep_mock,
            ):
                coord_lib.atomic_write_text(target, '{"message":"ok"}\n')

            self.assertEqual('{"message":"ok"}\n', target.read_text(encoding="utf-8"))
            self.assertEqual(3, attempts["count"])
            self.assertEqual(2, sleep_mock.call_count)


if __name__ == "__main__":
    unittest.main()
