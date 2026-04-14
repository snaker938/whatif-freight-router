from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import sqlite3
import subprocess
import textwrap
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator


RUNTIME_DIR_NAME = "codex-coordination"
FALLBACK_RUNTIME_DIR = ".codex_tmp/codex-coordination"
REQUIRED_CHILD_COUNT = 6
DEFAULT_SLOT_COUNT = REQUIRED_CHILD_COUNT
DEFAULT_STALE_AFTER_SECONDS = 300
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60
DEFAULT_PYTHON_MEMORY_CAP_PERCENT = 5.0
SCHEMA_VERSION = 3
WORK_ITEM_STATUSES = {"open", "claimed", "in_progress", "blocked", "qa", "closed"}
MESSAGE_CATEGORIES = {"handoff", "warning", "unblock", "claim-conflict", "note"}
SESSION_OUTCOMES = {"active", "completed", "interrupted", "crashed_or_lost", "reaped", "handed_off", "taken_over", "replaced"}
SESSION_OUTCOME_ALIASES = {
    "finished": "completed",
    "handoff": "handed_off",
    "stale_reaped": "reaped",
}
SESSION_TERMINAL_STATES = SESSION_OUTCOMES - {"active"}
CHILD_IDLE_ACTIVITY_STATUSES = {"standby", "watch", "reserve", "qa-watch"}
SNAPSHOT_FILES = (
    "active_sessions.json",
    "work_item_claims.json",
    "file_claims.json",
    "python_leases.json",
    "messages.json",
    "events.json",
    "snapshot_manifest.json",
    "status.md",
)


class CoordinationError(RuntimeError):
    """Raised when an operation would violate the coordination protocol."""


@dataclass(frozen=True)
class RepoContext:
    repo_root: Path
    runtime_root: Path
    git_common_dir: Path | None
    db_path: Path
    snapshots_dir: Path
    archive_dir: Path
    uses_git_common_dir: bool
    repo_identity: str
    path_case_insensitive: bool


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso(value: datetime | None = None) -> str:
    target = value or utc_now()
    return target.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def discover_repo_root(repo_root: str | Path | None = None) -> Path:
    base = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    try:
        return Path(git_output(base, "rev-parse", "--show-toplevel")).resolve()
    except Exception:
        return base


def discover_runtime_root(repo_root: Path) -> tuple[Path, Path | None, bool]:
    try:
        raw_common = git_output(repo_root, "rev-parse", "--git-common-dir")
    except Exception:
        return (repo_root / FALLBACK_RUNTIME_DIR).resolve(), None, False
    common_dir = Path(raw_common)
    if not common_dir.is_absolute():
        common_dir = (repo_root / common_dir).resolve()
    return (common_dir / RUNTIME_DIR_NAME).resolve(), common_dir.resolve(), True


def discover_repo_identity(repo_root: Path, git_common_dir: Path | None) -> str:
    if git_common_dir is not None:
        return str(git_common_dir)
    return str(repo_root)


def discover_path_case_insensitive(repo_root: Path) -> bool:
    try:
        raw = git_output(repo_root, "config", "--bool", "core.ignorecase").strip().lower()
        if raw in {"true", "false"}:
            return raw == "true"
    except Exception:
        pass
    return os.name == "nt"


def load_repo_context(repo_root: str | Path | None = None) -> RepoContext:
    resolved_root = discover_repo_root(repo_root)
    runtime_root, git_common_dir, uses_git_common_dir = discover_runtime_root(resolved_root)
    return RepoContext(
        repo_root=resolved_root,
        runtime_root=runtime_root,
        git_common_dir=git_common_dir,
        db_path=runtime_root / "coordination.sqlite3",
        snapshots_dir=runtime_root / "snapshots",
        archive_dir=runtime_root / "archive",
        uses_git_common_dir=uses_git_common_dir,
        repo_identity=discover_repo_identity(resolved_root, git_common_dir),
        path_case_insensitive=discover_path_case_insensitive(resolved_root),
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    try:
        for attempt in range(8):
            try:
                os.replace(temp_path, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def default_session_id(prefix: str = "parent") -> str:
    return f"{prefix}-{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def default_lease_id() -> str:
    return f"py-{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def append_note(existing: str, extra: str) -> str:
    existing = existing.strip()
    extra = extra.strip()
    if not extra:
        return existing
    if not existing:
        return extra
    return f"{existing}\n{extra}"


def normalize_session_outcome(value: str) -> str:
    normalized = value.strip()
    mapped = SESSION_OUTCOME_ALIASES.get(normalized, normalized)
    if mapped not in SESSION_OUTCOMES:
        raise CoordinationError(f"unsupported session outcome '{value}'")
    return mapped


def normalize_task_scope(raw_scope: str | None, *, task_summary: str) -> tuple[str, str]:
    if raw_scope and raw_scope.strip():
        slug = raw_scope.strip().lower().replace("\\", "/")
        slug = "-".join(part for part in slug.replace("/", "-").split())
        return slug or derive_task_scope(task_summary), "explicit"
    return derive_task_scope(task_summary), "auto"


def derive_task_scope(task_summary: str) -> str:
    normalized = " ".join(task_summary.strip().split()).lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"task-{digest}"


def normalize_work_item_id(raw_work_item_id: str) -> str:
    normalized = raw_work_item_id.strip()
    if not normalized:
        raise CoordinationError("work item id cannot be blank")
    return normalized


def make_work_item_key(task_scope: str, raw_work_item_id: str) -> str:
    return f"{task_scope}::{normalize_work_item_id(raw_work_item_id)}"


def normalize_repo_path(repo_root: Path, raw_path: str, *, case_insensitive: bool = False) -> str:
    repo_root = repo_root.resolve()
    candidate = Path(raw_path)
    absolute = (repo_root / candidate).resolve(strict=False) if not candidate.is_absolute() else candidate.resolve(strict=False)
    repo_text = os.path.normcase(str(repo_root)) if case_insensitive else str(repo_root)
    absolute_text = os.path.normcase(str(absolute)) if case_insensitive else str(absolute)
    try:
        common_root = os.path.commonpath([repo_text, absolute_text])
    except ValueError as exc:
        raise CoordinationError(f"path must stay inside the repository: {raw_path}") from exc
    if common_root != repo_text:
        raise CoordinationError(f"path must stay inside the repository: {raw_path}")
    relative_text = os.path.relpath(str(absolute), str(repo_root)).replace("\\", "/")
    normalized = Path(relative_text).as_posix()
    return normalized.casefold() if case_insensitive else normalized


def parse_json_or_default(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def pid_is_running(pid: int | None) -> bool | None:
    if pid is None or pid <= 0:
        return None
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return None
            return int(exit_code.value) == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return None
    return True


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


class CoordinationStore:
    def __init__(self, context: RepoContext) -> None:
        self.context = context
        self.context.runtime_root.mkdir(parents=True, exist_ok=True)
        self.context.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.context.archive_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.context.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _bootstrap(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            has_sessions = self._table_exists(connection, "sessions")
            if not has_sessions:
                self._create_latest_schema(connection)
                current_version = SCHEMA_VERSION
            elif current_version == 0:
                self._migrate_legacy_schema(connection)
                current_version = SCHEMA_VERSION
            elif current_version > SCHEMA_VERSION:
                raise CoordinationError(
                    f"coordination store schema {current_version} is newer than supported schema {SCHEMA_VERSION}"
                )
            elif current_version < SCHEMA_VERSION:
                self._migrate_legacy_schema(connection)
                current_version = SCHEMA_VERSION
            self._write_metadata(connection)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _table_exists(self, connection: sqlite3.Connection, table_name: str) -> bool:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _column_names(self, connection: sqlite3.Connection, table_name: str) -> set[str]:
        if not self._table_exists(connection, table_name):
            return set()
        return {row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})")}

    def _ensure_column(self, connection: sqlite3.Connection, table_name: str, column_sql: str, column_name: str) -> None:
        if column_name in self._column_names(connection, table_name):
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")

    def _create_latest_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_type TEXT NOT NULL DEFAULT 'parent',
                parent_session_id TEXT,
                child_slot_id INTEGER,
                repo_root TEXT NOT NULL,
                worktree_root TEXT NOT NULL,
                repo_identity TEXT NOT NULL,
                owner TEXT,
                task_summary TEXT NOT NULL,
                task_scope TEXT NOT NULL,
                task_scope_source TEXT NOT NULL DEFAULT 'auto',
                role TEXT NOT NULL DEFAULT '',
                agent_name TEXT NOT NULL DEFAULT '',
                agent_kind TEXT NOT NULL DEFAULT '',
                activity_status TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                work_item_ids_json TEXT NOT NULL DEFAULT '[]',
                external_agent_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_heartbeat TEXT NOT NULL,
                stale_after_seconds INTEGER NOT NULL,
                slot_count INTEGER NOT NULL DEFAULT 0,
                outcome TEXT,
                ended_at TEXT,
                note TEXT NOT NULL DEFAULT '',
                resume_from_session TEXT,
                resumed_by_session TEXT,
                reaped_by_session TEXT,
                takeover_by_session TEXT,
                replaced_by_session TEXT,
                replacement_for_session TEXT,
                terminal_reason TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_parent_session_id ON sessions(parent_session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_parent_slot_active
                ON sessions(parent_session_id, child_slot_id, status);
            CREATE TABLE IF NOT EXISTS slots (
                session_id TEXT NOT NULL,
                slot_id INTEGER NOT NULL,
                child_session_id TEXT,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                work_item_ids_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL,
                last_heartbeat TEXT NOT NULL,
                PRIMARY KEY (session_id, slot_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS work_items (
                work_item_id TEXT PRIMARY KEY,
                raw_work_item_id TEXT NOT NULL,
                task_scope TEXT NOT NULL,
                title TEXT NOT NULL,
                source_ref TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                owner_session TEXT,
                owner_slot INTEGER,
                created_by_session TEXT,
                created_at TEXT NOT NULL,
                claimed_at TEXT,
                updated_at TEXT NOT NULL,
                latest_note TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT ''
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_work_items_scope_raw_id ON work_items(task_scope, raw_work_item_id);
            CREATE TABLE IF NOT EXISTS file_claims (
                claim_key TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                section_id TEXT,
                section_start_line INTEGER,
                section_end_line INTEGER,
                mode TEXT NOT NULL,
                owner_session TEXT NOT NULL,
                owner_slot INTEGER,
                claimed_at TEXT NOT NULL,
                last_heartbeat TEXT NOT NULL,
                stale_after_seconds INTEGER NOT NULL,
                note TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_file_claims_path ON file_claims(path);
            CREATE TABLE IF NOT EXISTS python_leases (
                lease_id TEXT PRIMARY KEY,
                owner_session TEXT NOT NULL,
                owner_slot INTEGER,
                purpose TEXT NOT NULL,
                command TEXT NOT NULL,
                pid INTEGER,
                started_at TEXT NOT NULL,
                last_heartbeat TEXT NOT NULL,
                memory_cap_bytes INTEGER,
                memory_cap_percent REAL,
                enforcement_method TEXT NOT NULL,
                status TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                closed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_session TEXT NOT NULL,
                recipient_session TEXT,
                category TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                related_work_item_id TEXT,
                related_work_item_key TEXT,
                related_task_scope TEXT,
                related_path TEXT,
                created_at TEXT NOT NULL,
                ack_at TEXT,
                ack_by TEXT,
                archived_at TEXT
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                task_summary TEXT NOT NULL,
                task_scope TEXT NOT NULL DEFAULT '',
                blocker_json TEXT NOT NULL,
                next_actions_json TEXT NOT NULL,
                evidence_paths_json TEXT NOT NULL,
                resume_context_json TEXT NOT NULL DEFAULT '{}',
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS child_notes (
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                child_session_id TEXT NOT NULL,
                parent_session_id TEXT NOT NULL,
                child_slot_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                activity_status TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                work_item_ids_json TEXT NOT NULL DEFAULT '[]',
                evidence_paths_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                session_id TEXT,
                slot_id INTEGER,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

    def _migrate_legacy_schema(self, connection: sqlite3.Connection) -> None:
        self._ensure_column(connection, "sessions", "parent_session_id TEXT", "parent_session_id")
        self._ensure_column(connection, "sessions", "child_slot_id INTEGER", "child_slot_id")
        self._ensure_column(connection, "sessions", "worktree_root TEXT NOT NULL DEFAULT ''", "worktree_root")
        self._ensure_column(connection, "sessions", "repo_identity TEXT NOT NULL DEFAULT ''", "repo_identity")
        self._ensure_column(connection, "sessions", "task_scope TEXT NOT NULL DEFAULT ''", "task_scope")
        self._ensure_column(connection, "sessions", "task_scope_source TEXT NOT NULL DEFAULT 'legacy_auto'", "task_scope_source")
        self._ensure_column(connection, "sessions", "role TEXT NOT NULL DEFAULT ''", "role")
        self._ensure_column(connection, "sessions", "agent_name TEXT NOT NULL DEFAULT ''", "agent_name")
        self._ensure_column(connection, "sessions", "agent_kind TEXT NOT NULL DEFAULT ''", "agent_kind")
        self._ensure_column(connection, "sessions", "activity_status TEXT NOT NULL DEFAULT ''", "activity_status")
        self._ensure_column(connection, "sessions", "summary TEXT NOT NULL DEFAULT ''", "summary")
        self._ensure_column(connection, "sessions", "work_item_ids_json TEXT NOT NULL DEFAULT '[]'", "work_item_ids_json")
        self._ensure_column(connection, "sessions", "external_agent_id TEXT NOT NULL DEFAULT ''", "external_agent_id")
        self._ensure_column(connection, "sessions", "resume_from_session TEXT", "resume_from_session")
        self._ensure_column(connection, "sessions", "resumed_by_session TEXT", "resumed_by_session")
        self._ensure_column(connection, "sessions", "reaped_by_session TEXT", "reaped_by_session")
        self._ensure_column(connection, "sessions", "takeover_by_session TEXT", "takeover_by_session")
        self._ensure_column(connection, "sessions", "replaced_by_session TEXT", "replaced_by_session")
        self._ensure_column(connection, "sessions", "replacement_for_session TEXT", "replacement_for_session")
        self._ensure_column(connection, "sessions", "terminal_reason TEXT NOT NULL DEFAULT ''", "terminal_reason")

        self._ensure_column(connection, "work_items", "raw_work_item_id TEXT", "raw_work_item_id")
        self._ensure_column(connection, "work_items", "task_scope TEXT NOT NULL DEFAULT ''", "task_scope")

        self._ensure_column(connection, "slots", "child_session_id TEXT", "child_session_id")
        self._ensure_column(connection, "file_claims", "section_start_line INTEGER", "section_start_line")
        self._ensure_column(connection, "file_claims", "section_end_line INTEGER", "section_end_line")

        self._ensure_column(connection, "messages", "related_work_item_key TEXT", "related_work_item_key")
        self._ensure_column(connection, "messages", "related_task_scope TEXT", "related_task_scope")

        self._ensure_column(connection, "checkpoints", "task_scope TEXT NOT NULL DEFAULT ''", "task_scope")
        self._ensure_column(connection, "checkpoints", "resume_context_json TEXT NOT NULL DEFAULT '{}'", "resume_context_json")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS child_notes (
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                child_session_id TEXT NOT NULL,
                parent_session_id TEXT NOT NULL,
                child_slot_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                activity_status TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                work_item_ids_json TEXT NOT NULL DEFAULT '[]',
                evidence_paths_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_parent_session_id ON sessions(parent_session_id)")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_parent_slot_active ON sessions(parent_session_id, child_slot_id, status)"
        )

        connection.execute(
            """
            UPDATE sessions
            SET worktree_root = CASE WHEN worktree_root = '' THEN repo_root ELSE worktree_root END,
                repo_identity = CASE WHEN repo_identity = '' THEN ? ELSE repo_identity END,
                task_scope_source = CASE WHEN task_scope_source = '' THEN 'legacy_auto' ELSE task_scope_source END,
                work_item_ids_json = COALESCE(NULLIF(work_item_ids_json, ''), '[]'),
                external_agent_id = COALESCE(external_agent_id, ''),
                role = COALESCE(role, ''),
                agent_name = COALESCE(agent_name, ''),
                agent_kind = COALESCE(agent_kind, ''),
                activity_status = COALESCE(activity_status, ''),
                summary = COALESCE(summary, '')
            """,
            (self.context.repo_identity,),
        )
        legacy_sessions = connection.execute("SELECT session_id, task_summary, task_scope, status, outcome FROM sessions").fetchall()
        for row in legacy_sessions:
            scope = row["task_scope"] or derive_task_scope(row["task_summary"] or row["session_id"])
            canonical_status = normalize_session_outcome(row["status"]) if row["status"] != "active" else "active"
            canonical_outcome = normalize_session_outcome(row["outcome"]) if row["outcome"] else (canonical_status if canonical_status != "active" else None)
            connection.execute(
                """
                UPDATE sessions
                SET task_scope = ?, status = ?, outcome = COALESCE(?, outcome)
                WHERE session_id = ?
                """,
                (scope, canonical_status, canonical_outcome, row["session_id"]),
            )
        scope_by_session = {
            row["session_id"]: row["task_scope"]
            for row in connection.execute("SELECT session_id, task_scope FROM sessions")
        }
        legacy_items = connection.execute(
            """
            SELECT work_item_id, raw_work_item_id, task_scope, created_by_session, owner_session
            FROM work_items
            ORDER BY created_at ASC
            """
        ).fetchall()
        for item in legacy_items:
            raw_work_item_id = item["raw_work_item_id"] or item["work_item_id"]
            scope = item["task_scope"] or scope_by_session.get(item["owner_session"]) or scope_by_session.get(item["created_by_session"]) or derive_task_scope("legacy-shared-task")
            connection.execute(
                """
                UPDATE work_items
                SET raw_work_item_id = ?, task_scope = ?, work_item_id = ?
                WHERE work_item_id = ?
                """,
                (
                    raw_work_item_id,
                    scope,
                    make_work_item_key(scope, raw_work_item_id),
                    item["work_item_id"],
                ),
            )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_work_items_scope_raw_id ON work_items(task_scope, raw_work_item_id)"
        )
        connection.execute(
            """
            UPDATE messages
            SET related_task_scope = COALESCE(related_task_scope, (
                    SELECT task_scope FROM sessions WHERE session_id = messages.sender_session
                )),
                related_work_item_key = CASE
                    WHEN related_work_item_id IS NULL THEN related_work_item_key
                    WHEN related_work_item_key IS NOT NULL THEN related_work_item_key
                    ELSE COALESCE((
                        SELECT work_item_id
                        FROM work_items
                        WHERE raw_work_item_id = messages.related_work_item_id
                          AND task_scope = COALESCE(messages.related_task_scope, (
                              SELECT task_scope FROM sessions WHERE session_id = messages.sender_session
                          ))
                    ), related_work_item_key)
                END
            """
        )
        connection.execute(
            """
            UPDATE checkpoints
            SET task_scope = COALESCE(NULLIF(task_scope, ''), (
                    SELECT task_scope FROM sessions WHERE session_id = checkpoints.session_id
                )),
                resume_context_json = COALESCE(NULLIF(resume_context_json, ''), '{}')
            """
        )

    def _write_metadata(self, connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", str(SCHEMA_VERSION)),
                ("repo_root", str(self.context.repo_root)),
                ("repo_identity", self.context.repo_identity),
                ("runtime_root", str(self.context.runtime_root)),
                ("git_common_dir", str(self.context.git_common_dir) if self.context.git_common_dir else ""),
                ("uses_git_common_dir", "true" if self.context.uses_git_common_dir else "false"),
                ("default_stale_after_seconds", str(DEFAULT_STALE_AFTER_SECONDS)),
                ("default_heartbeat_interval_seconds", str(DEFAULT_HEARTBEAT_INTERVAL_SECONDS)),
                ("default_python_memory_cap_percent", str(DEFAULT_PYTHON_MEMORY_CAP_PERCENT)),
                ("required_child_count", str(REQUIRED_CHILD_COUNT)),
                ("path_case_insensitive", "true" if self.context.path_case_insensitive else "false"),
            ),
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _record_event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
        slot_id: int | None = None,
        created_at: str | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(event_type, session_id, slot_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_type, session_id, slot_id, json.dumps(payload, sort_keys=True), created_at or utc_iso()),
        )

    def _require_session_row(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        *,
        active_only: bool = False,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise CoordinationError(f"session '{session_id}' is not registered")
        if active_only and row["status"] != "active":
            raise CoordinationError(f"session '{session_id}' is not active")
        return row

    def _require_parent_session_row(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        *,
        active_only: bool = False,
    ) -> sqlite3.Row:
        row = self._require_session_row(connection, session_id, active_only=active_only)
        if row["session_type"] != "parent":
            raise CoordinationError(f"session '{session_id}' is not a parent session")
        return row

    def _require_child_session_row(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        *,
        active_only: bool = False,
    ) -> sqlite3.Row:
        row = self._require_session_row(connection, session_id, active_only=active_only)
        if row["session_type"] != "child":
            raise CoordinationError(f"session '{session_id}' is not a child session")
        return row

    def _parse_session_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        payload = row if isinstance(row, dict) else row_to_dict(row)
        parsed = dict(payload)
        parsed["work_item_ids"] = parse_json_or_default(parsed.pop("work_item_ids_json", "[]"), [])
        parsed["is_parent_session"] = parsed.get("session_type") == "parent"
        parsed["is_child_session"] = parsed.get("session_type") == "child"
        return parsed

    def _session_parent_id(self, connection: sqlite3.Connection, session_id: str) -> str:
        row = self._require_session_row(connection, session_id, active_only=False)
        return row["parent_session_id"] or row["session_id"]

    def _descendant_child_rows(self, connection: sqlite3.Connection, parent_session_id: str) -> list[sqlite3.Row]:
        return connection.execute(
            """
            SELECT * FROM sessions
            WHERE parent_session_id = ?
            ORDER BY child_slot_id ASC, started_at ASC, session_id ASC
            """,
            (parent_session_id,),
        ).fetchall()

    def _family_session_ids(self, connection: sqlite3.Connection, session_id: str) -> list[str]:
        row = self._require_session_row(connection, session_id, active_only=False)
        if row["session_type"] == "parent":
            return [row["session_id"], *[child["session_id"] for child in self._descendant_child_rows(connection, row["session_id"])]]
        return [row["session_id"]]

    def _active_child_row_for_slot(
        self,
        connection: sqlite3.Connection,
        *,
        parent_session_id: str,
        slot_id: int,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT * FROM sessions
            WHERE parent_session_id = ?
              AND child_slot_id = ?
              AND session_type = 'child'
              AND status = 'active'
            ORDER BY started_at DESC, session_id DESC
            LIMIT 1
            """,
            (parent_session_id, slot_id),
        ).fetchone()

    def _resolve_actor_session(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        owner_slot: int | None = None,
        active_only: bool = True,
    ) -> sqlite3.Row:
        row = self._require_session_row(connection, session_id, active_only=active_only)
        if row["session_type"] == "child":
            return row
        if owner_slot is None:
            return row
        parent = self._require_parent_session_row(connection, session_id, active_only=active_only)
        child = self._active_child_row_for_slot(connection, parent_session_id=parent["session_id"], slot_id=owner_slot)
        if child is None:
            raise CoordinationError(
                f"parent session '{session_id}' has no active child registered for slot {owner_slot}"
            )
        child_payload = self._parse_session_row(child)
        child_payload.update(self._session_staleness(child_payload))
        if child_payload["stale_signals"]:
            raise CoordinationError(
                f"slot {owner_slot} for parent '{session_id}' is assigned to stale child '{child['session_id']}'"
            )
        return child

    def _sync_slot_row_from_child(
        self,
        connection: sqlite3.Connection,
        *,
        child_session: sqlite3.Row | dict[str, Any],
        timestamp: str,
    ) -> None:
        child = self._parse_session_row(child_session)
        parent_session_id = child.get("parent_session_id")
        slot_id = int(child.get("child_slot_id") or 0)
        if not parent_session_id or slot_id < 1:
            return
        summary = child.get("summary") or "Awaiting assignment"
        connection.execute(
            """
            INSERT INTO slots(session_id, slot_id, child_session_id, role, status, summary, work_item_ids_json, updated_at, last_heartbeat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, slot_id) DO UPDATE SET
                child_session_id = excluded.child_session_id,
                role = excluded.role,
                status = excluded.status,
                summary = excluded.summary,
                work_item_ids_json = excluded.work_item_ids_json,
                updated_at = excluded.updated_at,
                last_heartbeat = excluded.last_heartbeat
            """,
            (
                parent_session_id,
                slot_id,
                child["session_id"],
                child.get("role") or "Unassigned",
                child.get("activity_status") or "active",
                summary,
                json.dumps(child.get("work_item_ids") or []),
                timestamp,
                child.get("last_heartbeat") or timestamp,
            ),
        )

    def _mark_slot_missing(
        self,
        connection: sqlite3.Connection,
        *,
        parent_session_id: str,
        slot_id: int,
        summary: str,
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO slots(session_id, slot_id, child_session_id, role, status, summary, work_item_ids_json, updated_at, last_heartbeat)
            VALUES (?, ?, NULL, 'Unassigned', 'missing_child', ?, '[]', ?, ?)
            ON CONFLICT(session_id, slot_id) DO UPDATE SET
                child_session_id = NULL,
                status = 'missing_child',
                summary = excluded.summary,
                work_item_ids_json = '[]',
                updated_at = excluded.updated_at,
                last_heartbeat = excluded.last_heartbeat
            """,
            (parent_session_id, slot_id, summary.strip(), timestamp, timestamp),
        )

    def _record_child_note(
        self,
        connection: sqlite3.Connection,
        *,
        child_session: sqlite3.Row,
        category: str,
        summary: str,
        evidence_paths: list[str] | None = None,
        timestamp: str,
    ) -> None:
        evidence_paths = evidence_paths or []
        connection.execute(
            """
            INSERT INTO child_notes(
                child_session_id, parent_session_id, child_slot_id, category, activity_status,
                role, summary, work_item_ids_json, evidence_paths_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                child_session["session_id"],
                child_session["parent_session_id"],
                int(child_session["child_slot_id"]),
                category.strip(),
                child_session["activity_status"],
                child_session["role"],
                summary.strip(),
                child_session["work_item_ids_json"] or "[]",
                json.dumps(evidence_paths),
                timestamp,
            ),
        )

    def _child_note_rows(
        self,
        connection: sqlite3.Connection,
        *,
        parent_session_id: str | None = None,
        child_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if child_session_id:
            rows = connection.execute(
                "SELECT * FROM child_notes WHERE child_session_id = ? ORDER BY created_at DESC, note_id DESC",
                (child_session_id,),
            ).fetchall()
        elif parent_session_id:
            rows = connection.execute(
                "SELECT * FROM child_notes WHERE parent_session_id = ? ORDER BY child_slot_id ASC, created_at DESC, note_id DESC",
                (parent_session_id,),
            ).fetchall()
        else:
            rows = connection.execute("SELECT * FROM child_notes ORDER BY created_at DESC, note_id DESC").fetchall()
        payloads = [row_to_dict(row) for row in rows]
        for payload in payloads:
            payload["work_item_ids"] = parse_json_or_default(payload.pop("work_item_ids_json"), [])
            payload["evidence_paths"] = parse_json_or_default(payload.pop("evidence_paths_json"), [])
        return payloads

    def _child_health(self, child: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        has_assignment = bool(child.get("work_item_ids")) or (
            child.get("activity_status") in CHILD_IDLE_ACTIVITY_STATUSES and bool((child.get("summary") or "").strip())
        )
        if child.get("status") != "active":
            issues.append("child_exited_unreplaced")
        if child.get("stale_signals"):
            issues.extend(child["stale_signals"])
        if not (child.get("role") or "").strip():
            issues.append("missing_role")
        if not (child.get("activity_status") or "").strip():
            issues.append("missing_status")
        if not has_assignment:
            issues.append("child_without_packet")
        live = not issues
        if live:
            health = "live"
        elif "heartbeat_overdue" in issues:
            health = "stale_child"
        elif child.get("status") == "active":
            health = "replacement_needed"
        else:
            health = "exited"
        return {
            "health": health,
            "health_issues": issues,
            "has_assignment": has_assignment,
            "is_live_child": live,
        }

    def _parent_child_health(
        self,
        connection: sqlite3.Connection,
        *,
        parent_session_id: str,
        required_child_count: int,
    ) -> dict[str, Any]:
        active_children_by_slot: dict[int, list[dict[str, Any]]] = {}
        unhealthy_children: list[dict[str, Any]] = []
        live_child_count = 0
        roster: list[dict[str, Any]] = []
        for child_row in self._descendant_child_rows(connection, parent_session_id):
            child = self._parse_session_row(child_row)
            child.update(self._session_staleness(child))
            child.update(self._child_health(child))
            slot_id = int(child.get("child_slot_id") or 0)
            if slot_id:
                active_children_by_slot.setdefault(slot_id, []).append(child)
            if child["status"] == "active" and not child["is_live_child"]:
                unhealthy_children.append(child)
        missing_slots: list[int] = []
        slot_rows = {
            int(row["slot_id"]): self._parse_slot_row(row)
            for row in connection.execute(
                "SELECT * FROM slots WHERE session_id = ? ORDER BY slot_id ASC",
                (parent_session_id,),
            ).fetchall()
        }
        for slot_id in range(1, required_child_count + 1):
            live_child = next(
                (
                    child
                    for child in active_children_by_slot.get(slot_id, [])
                    if child["status"] == "active" and child["is_live_child"]
                ),
                None,
            )
            slot_payload = slot_rows.get(slot_id, {"slot_id": slot_id, "role": "Unassigned", "status": "missing_child", "summary": "No slot row"})
            if live_child is None:
                missing_slots.append(slot_id)
                roster.append(
                    {
                        **slot_payload,
                        "slot_id": slot_id,
                        "child_session_id": live_child["session_id"] if live_child else slot_payload.get("child_session_id"),
                        "agent_name": live_child["agent_name"] if live_child else "",
                        "agent_kind": live_child["agent_kind"] if live_child else "",
                        "activity_status": live_child["activity_status"] if live_child else "missing_child",
                        "health": live_child["health"] if live_child else "missing_child",
                        "health_issues": live_child["health_issues"] if live_child else ["missing_child"],
                        "work_item_ids": live_child["work_item_ids"] if live_child else [],
                        "last_heartbeat": live_child["last_heartbeat"] if live_child else slot_payload.get("last_heartbeat"),
                        "summary": live_child["summary"] if live_child else slot_payload.get("summary", "No live child registered"),
                    }
                )
                continue
            live_child_count += 1
            roster.append(
                {
                    **slot_payload,
                    "slot_id": slot_id,
                    "child_session_id": live_child["session_id"],
                    "agent_name": live_child["agent_name"],
                    "agent_kind": live_child["agent_kind"],
                    "activity_status": live_child["activity_status"],
                    "health": live_child["health"],
                    "health_issues": live_child["health_issues"],
                    "work_item_ids": live_child["work_item_ids"],
                    "last_heartbeat": live_child["last_heartbeat"],
                    "summary": live_child["summary"],
                }
            )
        missing_child_count = len(missing_slots)
        return {
            "required_child_count": required_child_count,
            "live_child_count": live_child_count,
            "missing_child_count": missing_child_count,
            "unhealthy_child_count": len(unhealthy_children),
            "missing_child_slots": missing_slots,
            "unhealthy_children": unhealthy_children,
            "child_roster": roster,
            "is_child_roster_compliant": live_child_count == required_child_count and not unhealthy_children,
            "child_compliance": "compliant" if live_child_count == required_child_count and not unhealthy_children else "noncompliant",
        }

    def start_parent(
        self,
        *,
        task_summary: str,
        session_id: str | None = None,
        owner: str | None = None,
        task_scope: str | None = None,
        slot_count: int = DEFAULT_SLOT_COUNT,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    ) -> dict[str, Any]:
        if not task_summary.strip():
            raise CoordinationError("task summary cannot be blank")
        if slot_count != REQUIRED_CHILD_COUNT:
            raise CoordinationError(
                f"parent-controller sessions require exactly {REQUIRED_CHILD_COUNT} child slots; received {slot_count}"
            )
        chosen_session_id = session_id or default_session_id()
        chosen_scope, scope_source = normalize_task_scope(task_scope, task_summary=task_summary)
        timestamp = utc_iso()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (chosen_session_id,),
            ).fetchone()
            if existing is not None and existing["status"] != "active":
                raise CoordinationError(
                    f"session '{chosen_session_id}' already exists with status {existing['status']}"
                )
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO sessions(
                        session_id, session_type, parent_session_id, child_slot_id, repo_root, worktree_root, repo_identity,
                        owner, task_summary, task_scope, task_scope_source, role, agent_name, agent_kind,
                        activity_status, summary, work_item_ids_json, external_agent_id, status, started_at,
                        last_heartbeat, stale_after_seconds, slot_count
                    )
                    VALUES (?, 'parent', NULL, NULL, ?, ?, ?, ?, ?, ?, ?, '', '', '', '', '', '[]', '', 'active', ?, ?, ?, ?)
                    """,
                    (
                        chosen_session_id,
                        str(self.context.repo_root),
                        str(self.context.repo_root),
                        self.context.repo_identity,
                        owner,
                        task_summary.strip(),
                        chosen_scope,
                        scope_source,
                        timestamp,
                        timestamp,
                        stale_after_seconds,
                        slot_count,
                    ),
                )
            else:
                if existing["task_scope"] and existing["task_scope"] != chosen_scope:
                    raise CoordinationError(
                        f"session '{chosen_session_id}' already uses task scope '{existing['task_scope']}', not '{chosen_scope}'"
                    )
                connection.execute(
                    """
                    UPDATE sessions
                    SET owner = COALESCE(?, owner), task_summary = ?, task_scope = ?, task_scope_source = ?,
                        repo_root = ?, worktree_root = ?, repo_identity = ?, last_heartbeat = ?, slot_count = ?,
                        parent_session_id = NULL, child_slot_id = NULL, session_type = 'parent'
                    WHERE session_id = ?
                    """,
                    (
                        owner,
                        task_summary.strip(),
                        chosen_scope,
                        existing["task_scope_source"] or scope_source,
                        str(self.context.repo_root),
                        str(self.context.repo_root),
                        self.context.repo_identity,
                        timestamp,
                        slot_count,
                        chosen_session_id,
                    ),
                )
            for slot_id in range(1, slot_count + 1):
                child = self._active_child_row_for_slot(connection, parent_session_id=chosen_session_id, slot_id=slot_id)
                if child is not None:
                    self._sync_slot_row_from_child(connection, child_session=child, timestamp=timestamp)
                else:
                    self._mark_slot_missing(
                        connection,
                        parent_session_id=chosen_session_id,
                        slot_id=slot_id,
                        summary="No live child registered",
                        timestamp=timestamp,
                    )
            self._record_event(
                connection,
                "start_parent",
                {
                    "task_summary": task_summary.strip(),
                    "owner": owner,
                    "task_scope": chosen_scope,
                    "task_scope_source": scope_source,
                    "slot_count": slot_count,
                    "stale_after_seconds": stale_after_seconds,
                },
                session_id=chosen_session_id,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "session_id": chosen_session_id,
            "task_summary": task_summary.strip(),
            "task_scope": chosen_scope,
            "slot_count": slot_count,
            "required_child_count": slot_count,
            "runtime_root": str(self.context.runtime_root),
            "db_path": str(self.context.db_path),
            "status": "active",
        }

    def heartbeat(self, session_id: str, *, note: str = "") -> dict[str, Any]:
        timestamp = utc_iso()
        with self.transaction() as connection:
            row = self._require_parent_session_row(connection, session_id, active_only=True)
            connection.execute(
                "UPDATE sessions SET last_heartbeat = ?, note = ? WHERE session_id = ?",
                (timestamp, append_note(row["note"], note), session_id),
            )
            connection.execute(
                "UPDATE file_claims SET last_heartbeat = ? WHERE owner_session = ?",
                (timestamp, session_id),
            )
            connection.execute(
                "UPDATE python_leases SET last_heartbeat = ? WHERE owner_session = ? AND closed_at IS NULL",
                (timestamp, session_id),
            )
            self._record_event(
                connection,
                "heartbeat",
                {"note": note},
                session_id=session_id,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {"session_id": session_id, "last_heartbeat": timestamp}

    def start_child(
        self,
        *,
        parent_session_id: str,
        slot_id: int,
        role: str,
        agent_name: str,
        agent_kind: str = "",
        activity_status: str = "standby",
        summary: str = "",
        work_item_ids: list[str] | None = None,
        child_session_id: str | None = None,
        stale_after_seconds: int | None = None,
        task_summary: str | None = None,
        external_agent_id: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        if slot_id < 1 or slot_id > REQUIRED_CHILD_COUNT:
            raise CoordinationError(f"child slot ids must stay within 1..{REQUIRED_CHILD_COUNT}")
        if not role.strip():
            raise CoordinationError("child role cannot be blank")
        if not agent_name.strip():
            raise CoordinationError("child agent name cannot be blank")
        chosen_child_session_id = child_session_id or default_session_id("child")
        work_item_ids = work_item_ids or []
        if not work_item_ids and activity_status not in CHILD_IDLE_ACTIVITY_STATUSES:
            raise CoordinationError(
                f"child '{chosen_child_session_id}' must have assigned work items or an explicit standby/watch status"
            )
        timestamp = utc_iso()
        with self.transaction() as connection:
            parent = self._require_parent_session_row(connection, parent_session_id, active_only=True)
            existing = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (chosen_child_session_id,),
            ).fetchone()
            if existing is not None:
                raise CoordinationError(f"session '{chosen_child_session_id}' already exists")
            incumbent = self._active_child_row_for_slot(connection, parent_session_id=parent_session_id, slot_id=slot_id)
            if incumbent is not None:
                raise CoordinationError(
                    f"slot {slot_id} for parent '{parent_session_id}' is already occupied by child '{incumbent['session_id']}'"
                )
            connection.execute(
                """
                INSERT INTO sessions(
                    session_id, session_type, parent_session_id, child_slot_id, repo_root, worktree_root, repo_identity,
                    owner, task_summary, task_scope, task_scope_source, role, agent_name, agent_kind, activity_status,
                    summary, work_item_ids_json, external_agent_id, status, started_at, last_heartbeat, stale_after_seconds,
                    slot_count, note
                )
                VALUES (?, 'child', ?, ?, ?, ?, ?, ?, ?, ?, 'parent-linked', ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, 0, ?)
                """,
                (
                    chosen_child_session_id,
                    parent_session_id,
                    slot_id,
                    str(self.context.repo_root),
                    str(self.context.repo_root),
                    self.context.repo_identity,
                    parent["owner"],
                    (task_summary or parent["task_summary"]).strip(),
                    parent["task_scope"],
                    role.strip(),
                    agent_name.strip(),
                    (agent_kind or agent_name).strip(),
                    activity_status.strip(),
                    (summary or "Standby watch packet").strip(),
                    json.dumps(work_item_ids),
                    external_agent_id.strip(),
                    timestamp,
                    timestamp,
                    stale_after_seconds or int(parent["stale_after_seconds"]),
                    note.strip(),
                ),
            )
            child = self._require_child_session_row(connection, chosen_child_session_id, active_only=True)
            self._sync_slot_row_from_child(connection, child_session=child, timestamp=timestamp)
            self._record_child_note(
                connection,
                child_session=child,
                category="start",
                summary=child["summary"],
                timestamp=timestamp,
            )
            self._record_event(
                connection,
                "start_child",
                {
                    "parent_session_id": parent_session_id,
                    "child_session_id": chosen_child_session_id,
                    "slot_id": slot_id,
                    "role": role.strip(),
                    "agent_name": agent_name.strip(),
                    "agent_kind": (agent_kind or agent_name).strip(),
                    "activity_status": activity_status.strip(),
                    "work_item_ids": work_item_ids,
                    "external_agent_id": external_agent_id.strip(),
                },
                session_id=chosen_child_session_id,
                slot_id=slot_id,
                created_at=timestamp,
            )
        snapshot = self.refresh_snapshots()
        return {
            "session_id": chosen_child_session_id,
            "parent_session_id": parent_session_id,
            "slot_id": slot_id,
            "role": role.strip(),
            "agent_name": agent_name.strip(),
            "agent_kind": (agent_kind or agent_name).strip(),
            "activity_status": activity_status.strip(),
            "summary": (summary or "Standby watch packet").strip(),
            "work_item_ids": work_item_ids,
            "child_roster_health": snapshot["sessions"][parent_session_id]["child_health"],
        }

    def heartbeat_child(self, child_session_id: str, *, note: str = "") -> dict[str, Any]:
        timestamp = utc_iso()
        with self.transaction() as connection:
            child = self._require_child_session_row(connection, child_session_id, active_only=True)
            connection.execute(
                "UPDATE sessions SET last_heartbeat = ?, note = ? WHERE session_id = ?",
                (timestamp, append_note(child["note"], note), child_session_id),
            )
            connection.execute(
                "UPDATE file_claims SET last_heartbeat = ? WHERE owner_session = ?",
                (timestamp, child_session_id),
            )
            connection.execute(
                "UPDATE python_leases SET last_heartbeat = ? WHERE owner_session = ? AND closed_at IS NULL",
                (timestamp, child_session_id),
            )
            refreshed_child = self._require_child_session_row(connection, child_session_id, active_only=True)
            self._sync_slot_row_from_child(connection, child_session=refreshed_child, timestamp=timestamp)
            self._record_event(
                connection,
                "heartbeat_child",
                {"note": note},
                session_id=child_session_id,
                slot_id=int(child["child_slot_id"]),
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {"session_id": child_session_id, "last_heartbeat": timestamp}

    def update_child(
        self,
        *,
        child_session_id: str,
        role: str | None = None,
        activity_status: str | None = None,
        summary: str | None = None,
        work_item_ids: list[str] | None = None,
        agent_name: str | None = None,
        agent_kind: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        timestamp = utc_iso()
        with self.transaction() as connection:
            child = self._require_child_session_row(connection, child_session_id, active_only=True)
            next_role = (role or child["role"]).strip()
            next_activity_status = (activity_status or child["activity_status"]).strip()
            next_summary = (summary or child["summary"]).strip()
            next_work_item_ids = work_item_ids if work_item_ids is not None else parse_json_or_default(child["work_item_ids_json"], [])
            next_agent_name = (agent_name or child["agent_name"]).strip()
            next_agent_kind = (agent_kind or child["agent_kind"] or next_agent_name).strip()
            if not next_work_item_ids and next_activity_status not in CHILD_IDLE_ACTIVITY_STATUSES:
                raise CoordinationError(
                    f"child '{child_session_id}' must have assigned work items or an explicit standby/watch status"
                )
            connection.execute(
                """
                UPDATE sessions
                SET role = ?, activity_status = ?, summary = ?, work_item_ids_json = ?, agent_name = ?, agent_kind = ?,
                    last_heartbeat = ?, note = ?
                WHERE session_id = ?
                """,
                (
                    next_role,
                    next_activity_status,
                    next_summary,
                    json.dumps(next_work_item_ids),
                    next_agent_name,
                    next_agent_kind,
                    timestamp,
                    append_note(child["note"], note),
                    child_session_id,
                ),
            )
            refreshed_child = self._require_child_session_row(connection, child_session_id, active_only=True)
            self._sync_slot_row_from_child(connection, child_session=refreshed_child, timestamp=timestamp)
            self._record_event(
                connection,
                "update_child",
                {
                    "role": next_role,
                    "activity_status": next_activity_status,
                    "summary": next_summary,
                    "work_item_ids": next_work_item_ids,
                    "agent_name": next_agent_name,
                    "agent_kind": next_agent_kind,
                },
                session_id=child_session_id,
                slot_id=int(refreshed_child["child_slot_id"]),
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "session_id": child_session_id,
            "role": next_role,
            "activity_status": next_activity_status,
            "summary": next_summary,
            "work_item_ids": next_work_item_ids,
            "agent_name": next_agent_name,
            "agent_kind": next_agent_kind,
        }

    def note_child(
        self,
        *,
        child_session_id: str,
        summary: str,
        category: str = "note",
        evidence_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_evidence = [
            normalize_repo_path(self.context.repo_root, path, case_insensitive=self.context.path_case_insensitive)
            for path in (evidence_paths or [])
        ]
        timestamp = utc_iso()
        with self.transaction() as connection:
            child = self._require_child_session_row(connection, child_session_id, active_only=True)
            connection.execute(
                "UPDATE sessions SET summary = ?, last_heartbeat = ? WHERE session_id = ?",
                (summary.strip(), timestamp, child_session_id),
            )
            refreshed_child = self._require_child_session_row(connection, child_session_id, active_only=True)
            self._sync_slot_row_from_child(connection, child_session=refreshed_child, timestamp=timestamp)
            self._record_child_note(
                connection,
                child_session=refreshed_child,
                category=category,
                summary=summary.strip(),
                evidence_paths=normalized_evidence,
                timestamp=timestamp,
            )
            self._record_event(
                connection,
                "note_child",
                {
                    "category": category,
                    "summary": summary.strip(),
                    "evidence_paths": normalized_evidence,
                },
                session_id=child_session_id,
                slot_id=int(refreshed_child["child_slot_id"]),
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "session_id": child_session_id,
            "category": category,
            "summary": summary.strip(),
            "evidence_paths": normalized_evidence,
            "created_at": timestamp,
        }

    def end_child(
        self,
        *,
        child_session_id: str,
        outcome: str = "completed",
        note: str = "",
        release_work_to_parent: bool = True,
    ) -> dict[str, Any]:
        normalized_outcome = normalize_session_outcome(outcome)
        timestamp = utc_iso()
        with self.transaction() as connection:
            child = self._require_child_session_row(connection, child_session_id, active_only=False)
            if child["status"] != "active":
                response = {
                    "session_id": child_session_id,
                    "outcome": child["outcome"] or child["status"],
                    "ended_at": child["ended_at"] or child["last_heartbeat"],
                    "already_ended": True,
                }
            else:
                parent_session_id = child["parent_session_id"]
                slot_id = int(child["child_slot_id"])
                owned_work_items = connection.execute(
                    "SELECT * FROM work_items WHERE owner_session = ? ORDER BY task_scope ASC, raw_work_item_id ASC",
                    (child_session_id,),
                ).fetchall()
                if release_work_to_parent and parent_session_id:
                    for item in owned_work_items:
                        next_status = item["status"] if item["status"] in {"blocked", "qa", "closed"} else "claimed"
                        connection.execute(
                            """
                            UPDATE work_items
                            SET owner_session = ?, owner_slot = ?, status = ?, updated_at = ?, latest_note = ?
                            WHERE work_item_id = ?
                            """,
                            (
                                parent_session_id,
                                slot_id,
                                next_status,
                                timestamp,
                                append_note(
                                    item["latest_note"],
                                    f"Returned from child {child_session_id} during end-child ({normalized_outcome}).",
                                ),
                                item["work_item_id"],
                            ),
                        )
                connection.execute("DELETE FROM file_claims WHERE owner_session = ?", (child_session_id,))
                connection.execute(
                    """
                    UPDATE python_leases
                    SET status = CASE WHEN closed_at IS NULL THEN 'released_by_end_child' ELSE status END,
                        closed_at = COALESCE(closed_at, ?),
                        last_heartbeat = ?,
                        note = CASE
                            WHEN note = '' THEN ?
                            ELSE note || CHAR(10) || ?
                        END
                    WHERE owner_session = ?
                    """,
                    (
                        timestamp,
                        timestamp,
                        f"Closed during end-child ({normalized_outcome}).",
                        f"Closed during end-child ({normalized_outcome}).",
                        child_session_id,
                    ),
                )
                connection.execute(
                    """
                    UPDATE messages
                    SET archived_at = COALESCE(archived_at, ?)
                    WHERE archived_at IS NULL AND (sender_session = ? OR recipient_session = ?)
                    """,
                    (timestamp, child_session_id, child_session_id),
                )
                connection.execute(
                    """
                    UPDATE sessions
                    SET status = ?, outcome = ?, ended_at = ?, last_heartbeat = ?, terminal_reason = ?, note = ?
                    WHERE session_id = ?
                    """,
                    (
                        normalized_outcome,
                        normalized_outcome,
                        timestamp,
                        timestamp,
                        note.strip(),
                        append_note(child["note"], note),
                        child_session_id,
                    ),
                )
                current_slot = connection.execute(
                    "SELECT * FROM slots WHERE session_id = ? AND slot_id = ?",
                    (parent_session_id, slot_id),
                ).fetchone()
                if current_slot is not None and current_slot["child_session_id"] == child_session_id:
                    self._mark_slot_missing(
                        connection,
                        parent_session_id=parent_session_id,
                        slot_id=slot_id,
                        summary=f"Child {child_session_id} ended; replacement needed",
                        timestamp=timestamp,
                    )
                ended_child = self._require_child_session_row(connection, child_session_id, active_only=False)
                self._record_child_note(
                    connection,
                    child_session=ended_child,
                    category="end",
                    summary=note.strip() or f"Child ended with outcome {normalized_outcome}",
                    timestamp=timestamp,
                )
                self._record_event(
                    connection,
                    "end_child",
                    {
                        "parent_session_id": parent_session_id,
                        "slot_id": slot_id,
                        "outcome": normalized_outcome,
                        "release_work_to_parent": release_work_to_parent,
                    },
                    session_id=child_session_id,
                    slot_id=slot_id,
                    created_at=timestamp,
                )
                response = {
                    "session_id": child_session_id,
                    "parent_session_id": parent_session_id,
                    "slot_id": slot_id,
                    "outcome": normalized_outcome,
                    "ended_at": timestamp,
                }
        self.refresh_snapshots()
        self.write_archive(child_session_id)
        return response

    def replace_child(
        self,
        *,
        parent_session_id: str,
        slot_id: int,
        role: str,
        agent_name: str,
        agent_kind: str = "",
        activity_status: str = "standby",
        summary: str = "",
        work_item_ids: list[str] | None = None,
        child_session_id: str | None = None,
        from_child_session_id: str | None = None,
        external_agent_id: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        if slot_id < 1 or slot_id > REQUIRED_CHILD_COUNT:
            raise CoordinationError(f"child slot ids must stay within 1..{REQUIRED_CHILD_COUNT}")
        chosen_child_session_id = child_session_id or default_session_id("child")
        timestamp = utc_iso()
        archived_child_id: str | None = None
        with self.transaction() as connection:
            parent = self._require_parent_session_row(connection, parent_session_id, active_only=True)
            existing = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (chosen_child_session_id,),
            ).fetchone()
            if existing is not None:
                raise CoordinationError(f"session '{chosen_child_session_id}' already exists")
            old_child = None
            if from_child_session_id:
                old_child = self._require_child_session_row(connection, from_child_session_id, active_only=False)
                if old_child["parent_session_id"] != parent_session_id or int(old_child["child_slot_id"]) != slot_id:
                    raise CoordinationError(
                        f"child '{from_child_session_id}' does not belong to parent '{parent_session_id}' slot {slot_id}"
                    )
            else:
                old_child = self._active_child_row_for_slot(connection, parent_session_id=parent_session_id, slot_id=slot_id)
            work_rows = []
            if old_child is not None:
                work_rows = connection.execute(
                    "SELECT * FROM work_items WHERE owner_session = ? ORDER BY task_scope ASC, raw_work_item_id ASC",
                    (old_child["session_id"],),
                ).fetchall()
                if old_child["status"] == "active":
                    connection.execute("DELETE FROM file_claims WHERE owner_session = ?", (old_child["session_id"],))
                    connection.execute(
                        """
                        UPDATE python_leases
                        SET status = CASE WHEN closed_at IS NULL THEN 'replaced' ELSE status END,
                            closed_at = COALESCE(closed_at, ?),
                            last_heartbeat = ?,
                            note = CASE
                                WHEN note = '' THEN ?
                                ELSE note || CHAR(10) || ?
                            END
                        WHERE owner_session = ?
                        """,
                        (
                            timestamp,
                            timestamp,
                            f"Closed during child replacement by {chosen_child_session_id}.",
                            f"Closed during child replacement by {chosen_child_session_id}.",
                            old_child["session_id"],
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE messages
                        SET archived_at = COALESCE(archived_at, ?)
                        WHERE archived_at IS NULL AND (sender_session = ? OR recipient_session = ?)
                        """,
                        (timestamp, old_child["session_id"], old_child["session_id"]),
                    )
                    connection.execute(
                        """
                        UPDATE sessions
                        SET status = 'replaced', outcome = 'replaced', ended_at = ?, last_heartbeat = ?,
                            replaced_by_session = ?, terminal_reason = ?, note = ?
                        WHERE session_id = ?
                        """,
                        (
                            timestamp,
                            timestamp,
                            chosen_child_session_id,
                            note.strip() or f"Replaced in slot {slot_id} by {chosen_child_session_id}",
                            append_note(old_child["note"], note or f"Replaced by {chosen_child_session_id}."),
                            old_child["session_id"],
                        ),
                    )
                    archived_child_id = old_child["session_id"]
            inherited_work_item_ids = work_item_ids or [row["raw_work_item_id"] for row in work_rows]
            if not inherited_work_item_ids and activity_status not in CHILD_IDLE_ACTIVITY_STATUSES:
                raise CoordinationError(
                    f"replacement child '{chosen_child_session_id}' must have assigned work items or an explicit standby/watch status"
                )
            connection.execute(
                """
                INSERT INTO sessions(
                    session_id, session_type, parent_session_id, child_slot_id, repo_root, worktree_root, repo_identity,
                    owner, task_summary, task_scope, task_scope_source, role, agent_name, agent_kind, activity_status,
                    summary, work_item_ids_json, external_agent_id, status, started_at, last_heartbeat, stale_after_seconds,
                    slot_count, note, replacement_for_session
                )
                VALUES (?, 'child', ?, ?, ?, ?, ?, ?, ?, ?, 'parent-linked', ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, 0, ?, ?)
                """,
                (
                    chosen_child_session_id,
                    parent_session_id,
                    slot_id,
                    str(self.context.repo_root),
                    str(self.context.repo_root),
                    self.context.repo_identity,
                    parent["owner"],
                    parent["task_summary"],
                    parent["task_scope"],
                    role.strip(),
                    agent_name.strip(),
                    (agent_kind or agent_name).strip(),
                    activity_status.strip(),
                    (summary or "Replacement standby packet").strip(),
                    json.dumps(inherited_work_item_ids),
                    external_agent_id.strip(),
                    timestamp,
                    timestamp,
                    int(parent["stale_after_seconds"]),
                    note.strip(),
                    old_child["session_id"] if old_child is not None else None,
                ),
            )
            for item in work_rows:
                next_status = item["status"] if item["status"] in {"blocked", "qa", "closed", "in_progress"} else "claimed"
                connection.execute(
                    """
                    UPDATE work_items
                    SET owner_session = ?, owner_slot = ?, status = ?, updated_at = ?, latest_note = ?
                    WHERE work_item_id = ?
                    """,
                    (
                        chosen_child_session_id,
                        slot_id,
                        next_status,
                        timestamp,
                        append_note(
                            item["latest_note"],
                            f"Transferred from {old_child['session_id'] if old_child is not None else parent_session_id} to replacement child {chosen_child_session_id}.",
                        ),
                        item["work_item_id"],
                    ),
                )
            new_child = self._require_child_session_row(connection, chosen_child_session_id, active_only=True)
            self._sync_slot_row_from_child(connection, child_session=new_child, timestamp=timestamp)
            self._record_child_note(
                connection,
                child_session=new_child,
                category="replace",
                summary=new_child["summary"],
                timestamp=timestamp,
            )
            self._record_event(
                connection,
                "replace_child",
                {
                    "parent_session_id": parent_session_id,
                    "slot_id": slot_id,
                    "from_child_session_id": old_child["session_id"] if old_child is not None else None,
                    "child_session_id": chosen_child_session_id,
                    "role": role.strip(),
                    "agent_name": agent_name.strip(),
                    "activity_status": activity_status.strip(),
                    "work_item_ids": inherited_work_item_ids,
                },
                session_id=chosen_child_session_id,
                slot_id=slot_id,
                created_at=timestamp,
            )
        snapshot = self.refresh_snapshots()
        if archived_child_id:
            self.write_archive(archived_child_id)
        return {
            "session_id": chosen_child_session_id,
            "parent_session_id": parent_session_id,
            "slot_id": slot_id,
            "replaced_child_session_id": old_child["session_id"] if old_child is not None else None,
            "child_roster_health": snapshot["sessions"][parent_session_id]["child_health"],
        }

    def ensure_six_subagents(self, *, parent_session_id: str) -> dict[str, Any]:
        with self.transaction() as connection:
            parent = self._require_parent_session_row(connection, parent_session_id, active_only=True)
            return {
                "session_id": parent_session_id,
                **self._parent_child_health(
                    connection,
                    parent_session_id=parent_session_id,
                    required_child_count=int(parent["slot_count"] or REQUIRED_CHILD_COUNT),
                ),
            }

    def _work_item_identity(self, connection: sqlite3.Connection, session_id: str, raw_work_item_id: str) -> tuple[str, str]:
        session = self._require_session_row(connection, session_id)
        task_scope = session["task_scope"] or derive_task_scope(session["task_summary"])
        normalized_work_item_id = normalize_work_item_id(raw_work_item_id)
        return task_scope, make_work_item_key(task_scope, normalized_work_item_id)

    def _get_work_item_row(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        raw_work_item_id: str,
    ) -> tuple[sqlite3.Row | None, str, str]:
        task_scope, work_item_key = self._work_item_identity(connection, session_id, raw_work_item_id)
        row = connection.execute(
            "SELECT * FROM work_items WHERE work_item_id = ?",
            (work_item_key,),
        ).fetchone()
        return row, task_scope, work_item_key

    def upsert_work_item(
        self,
        *,
        session_id: str,
        work_item_id: str,
        title: str,
        source_ref: str = "",
        status: str = "open",
        note: str = "",
        evidence: str = "",
    ) -> dict[str, Any]:
        if status not in WORK_ITEM_STATUSES:
            raise CoordinationError(f"unsupported work-item status '{status}'")
        timestamp = utc_iso()
        normalized_work_item_id = normalize_work_item_id(work_item_id)
        with self.transaction() as connection:
            self._require_session_row(connection, session_id, active_only=True)
            existing, task_scope, work_item_key = self._get_work_item_row(
                connection,
                session_id=session_id,
                raw_work_item_id=normalized_work_item_id,
            )
            if existing and existing["owner_session"] and existing["owner_session"] != session_id:
                raise CoordinationError(
                    f"work item '{normalized_work_item_id}' in task scope '{task_scope}' is claimed by session '{existing['owner_session']}'"
                )
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO work_items(
                        work_item_id, raw_work_item_id, task_scope, title, source_ref, status,
                        owner_session, owner_slot, created_by_session, created_at, claimed_at,
                        updated_at, latest_note, evidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        work_item_key,
                        normalized_work_item_id,
                        task_scope,
                        title.strip(),
                        source_ref.strip(),
                        status,
                        session_id,
                        timestamp,
                        timestamp,
                        note.strip(),
                        evidence.strip(),
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE work_items
                    SET title = ?, source_ref = ?, status = ?, updated_at = ?, latest_note = ?, evidence = ?
                    WHERE work_item_id = ?
                    """,
                    (
                        title.strip(),
                        source_ref.strip(),
                        status,
                        timestamp,
                        append_note(existing["latest_note"], note),
                        evidence.strip() or existing["evidence"],
                        work_item_key,
                    ),
                )
            self._record_event(
                connection,
                "upsert_work_item",
                {
                    "task_scope": task_scope,
                    "work_item_id": normalized_work_item_id,
                    "work_item_key": work_item_key,
                    "status": status,
                    "title": title.strip(),
                },
                session_id=session_id,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {"work_item_id": normalized_work_item_id, "work_item_key": work_item_key, "task_scope": task_scope, "status": status}

    def claim_work(
        self,
        *,
        session_id: str,
        work_item_id: str,
        title: str | None = None,
        source_ref: str = "",
        status: str = "claimed",
        owner_slot: int | None = None,
        note: str = "",
        evidence: str = "",
    ) -> dict[str, Any]:
        if status not in WORK_ITEM_STATUSES or status == "open":
            raise CoordinationError("claim-work requires a claimed/in-progress/blocked/qa/closed status")
        timestamp = utc_iso()
        normalized_work_item_id = normalize_work_item_id(work_item_id)
        with self.transaction() as connection:
            actor = self._resolve_actor_session(connection, session_id=session_id, owner_slot=owner_slot, active_only=True)
            actor_slot = int(actor["child_slot_id"]) if actor["session_type"] == "child" else owner_slot
            existing, task_scope, work_item_key = self._get_work_item_row(
                connection,
                session_id=actor["session_id"],
                raw_work_item_id=normalized_work_item_id,
            )
            if existing and existing["owner_session"] and existing["owner_session"] != actor["session_id"]:
                raise CoordinationError(
                    f"work item '{normalized_work_item_id}' in task scope '{task_scope}' is already claimed by session '{existing['owner_session']}'"
                )
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO work_items(
                        work_item_id, raw_work_item_id, task_scope, title, source_ref, status,
                        owner_session, owner_slot, created_by_session, created_at, claimed_at,
                        updated_at, latest_note, evidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        work_item_key,
                        normalized_work_item_id,
                        task_scope,
                        (title or normalized_work_item_id).strip(),
                        source_ref.strip(),
                        status,
                        actor["session_id"],
                        actor_slot,
                        actor["session_id"],
                        timestamp,
                        timestamp,
                        timestamp,
                        note.strip(),
                        evidence.strip(),
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE work_items
                    SET title = ?, source_ref = ?, status = ?, owner_session = ?, owner_slot = ?,
                        claimed_at = COALESCE(claimed_at, ?), updated_at = ?, latest_note = ?, evidence = ?
                    WHERE work_item_id = ?
                    """,
                    (
                        (title or existing["title"]).strip(),
                        source_ref.strip() or existing["source_ref"],
                        status,
                        actor["session_id"],
                        actor_slot,
                        timestamp,
                        timestamp,
                        append_note(existing["latest_note"], note),
                        evidence.strip() or existing["evidence"],
                        work_item_key,
                    ),
                )
            self._record_event(
                connection,
                "claim_work",
                {
                    "task_scope": task_scope,
                    "work_item_id": normalized_work_item_id,
                    "work_item_key": work_item_key,
                    "status": status,
                    "owner_slot": actor_slot,
                },
                session_id=actor["session_id"],
                slot_id=actor_slot,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "work_item_id": normalized_work_item_id,
            "work_item_key": work_item_key,
            "task_scope": task_scope,
            "owner_session": actor["session_id"],
            "owner_slot": actor_slot,
            "status": status,
        }

    def release_work(
        self,
        *,
        session_id: str,
        work_item_id: str,
        status: str = "open",
        note: str = "",
        evidence: str = "",
    ) -> dict[str, Any]:
        if status not in WORK_ITEM_STATUSES:
            raise CoordinationError(f"unsupported work-item status '{status}'")
        timestamp = utc_iso()
        normalized_work_item_id = normalize_work_item_id(work_item_id)
        with self.transaction() as connection:
            self._require_session_row(connection, session_id)
            family_session_ids = self._family_session_ids(connection, session_id)
            existing, task_scope, work_item_key = self._get_work_item_row(
                connection,
                session_id=session_id,
                raw_work_item_id=normalized_work_item_id,
            )
            if existing is None:
                raise CoordinationError(f"work item '{normalized_work_item_id}' does not exist in task scope '{task_scope}'")
            if existing["owner_session"] not in family_session_ids:
                raise CoordinationError(
                    f"work item '{normalized_work_item_id}' is owned by '{existing['owner_session']}'"
                )
            connection.execute(
                """
                UPDATE work_items
                SET owner_session = NULL, owner_slot = NULL, status = ?, updated_at = ?,
                    latest_note = ?, evidence = ?
                WHERE work_item_id = ?
                """,
                (
                    status,
                    timestamp,
                    append_note(existing["latest_note"], note),
                    evidence.strip() or existing["evidence"],
                    work_item_key,
                ),
            )
            self._record_event(
                connection,
                "release_work",
                {
                    "task_scope": task_scope,
                    "work_item_id": normalized_work_item_id,
                    "work_item_key": work_item_key,
                    "status": status,
                },
                session_id=session_id,
                slot_id=existing["owner_slot"],
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "work_item_id": normalized_work_item_id,
            "work_item_key": work_item_key,
            "task_scope": task_scope,
            "status": status,
            "owner_session": None,
        }

    def _find_conflicting_file_claims(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        normalized_path: str,
        section_id: str | None,
        section_start_line: int | None,
        section_end_line: int | None,
    ) -> list[dict[str, Any]]:
        rows = [
            row_to_dict(row)
            for row in connection.execute(
                "SELECT * FROM file_claims WHERE path = ? AND owner_session != ?",
                (normalized_path, session_id),
            ).fetchall()
        ]
        conflicts: list[dict[str, Any]] = []
        for row in rows:
            existing_has_range = row.get("section_start_line") is not None and row.get("section_end_line") is not None
            new_has_range = section_start_line is not None and section_end_line is not None
            if section_id is None:
                conflicts.append(row)
                continue
            if row.get("section_id") is None:
                conflicts.append(row)
                continue
            if new_has_range and existing_has_range:
                starts_before_end = int(section_start_line) <= int(row["section_end_line"])
                ends_after_start = int(section_end_line) >= int(row["section_start_line"])
                if starts_before_end and ends_after_start:
                    conflicts.append(row)
                continue
            if not new_has_range and not existing_has_range and row.get("section_id") != section_id:
                continue
            conflicts.append(row)
        return conflicts

    def _normalize_section_range(
        self,
        *,
        section_start_line: int | None,
        section_end_line: int | None,
    ) -> tuple[int | None, int | None]:
        if section_start_line is None and section_end_line is None:
            return None, None
        if section_start_line is None or section_end_line is None:
            raise CoordinationError("section line claims require both --section-start-line and --section-end-line")
        if section_start_line < 1 or section_end_line < section_start_line:
            raise CoordinationError("invalid section line range")
        return section_start_line, section_end_line

    def _file_claim_key(
        self,
        *,
        normalized_path: str,
        section_id: str | None,
        section_start_line: int | None,
        section_end_line: int | None,
    ) -> str:
        if section_id is None:
            return normalized_path
        if section_start_line is not None and section_end_line is not None:
            return f"{normalized_path}::{section_id}::{section_start_line}-{section_end_line}"
        return f"{normalized_path}::{section_id}"

    def claim_file(
        self,
        *,
        session_id: str,
        path: str,
        section_id: str | None = None,
        section_start_line: int | None = None,
        section_end_line: int | None = None,
        mode: str | None = None,
        owner_slot: int | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        timestamp = utc_iso()
        normalized_path = normalize_repo_path(
            self.context.repo_root,
            path,
            case_insensitive=self.context.path_case_insensitive,
        )
        section_start_line, section_end_line = self._normalize_section_range(
            section_start_line=section_start_line,
            section_end_line=section_end_line,
        )
        chosen_mode = mode or ("section-write" if section_id else "write")
        claim_key = self._file_claim_key(
            normalized_path=normalized_path,
            section_id=section_id,
            section_start_line=section_start_line,
            section_end_line=section_end_line,
        )
        with self.transaction() as connection:
            actor = self._resolve_actor_session(connection, session_id=session_id, owner_slot=owner_slot, active_only=True)
            actor_slot = int(actor["child_slot_id"]) if actor["session_type"] == "child" else owner_slot
            conflicts = self._find_conflicting_file_claims(
                connection,
                session_id=actor["session_id"],
                normalized_path=normalized_path,
                section_id=section_id,
                section_start_line=section_start_line,
                section_end_line=section_end_line,
            )
            if conflicts:
                owners = ", ".join(
                    f"{item['owner_session']}:{item['section_id'] or '*'}"
                    + (
                        f"[{item['section_start_line']}-{item['section_end_line']}]"
                        if item.get("section_start_line") is not None and item.get("section_end_line") is not None
                        else ""
                    )
                    for item in conflicts
                )
                raise CoordinationError(f"file claim conflict on '{normalized_path}' with {owners}")
            if section_id is None:
                connection.execute(
                    "DELETE FROM file_claims WHERE owner_session = ? AND path = ?",
                    (actor["session_id"], normalized_path),
                )
            existing = connection.execute(
                "SELECT * FROM file_claims WHERE claim_key = ?",
                (claim_key,),
            ).fetchone()
            if existing and existing["owner_session"] != actor["session_id"]:
                raise CoordinationError(
                    f"file claim '{claim_key}' is owned by session '{existing['owner_session']}'"
                )
            connection.execute(
                """
                INSERT INTO file_claims(
                    claim_key, path, section_id, section_start_line, section_end_line, mode, owner_session, owner_slot,
                    claimed_at, last_heartbeat, stale_after_seconds, note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_key) DO UPDATE SET
                    mode = excluded.mode,
                    owner_session = excluded.owner_session,
                    owner_slot = excluded.owner_slot,
                    section_start_line = excluded.section_start_line,
                    section_end_line = excluded.section_end_line,
                    last_heartbeat = excluded.last_heartbeat,
                    stale_after_seconds = excluded.stale_after_seconds,
                    note = excluded.note
                """,
                (
                    claim_key,
                    normalized_path,
                    section_id,
                    section_start_line,
                    section_end_line,
                    chosen_mode,
                    actor["session_id"],
                    actor_slot,
                    timestamp,
                    timestamp,
                    int(actor["stale_after_seconds"]),
                    note.strip(),
                ),
            )
            self._record_event(
                connection,
                "claim_file",
                {
                    "path": normalized_path,
                    "section_id": section_id,
                    "section_start_line": section_start_line,
                    "section_end_line": section_end_line,
                    "mode": chosen_mode,
                    "owner_slot": actor_slot,
                },
                session_id=actor["session_id"],
                slot_id=actor_slot,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "path": normalized_path,
            "section_id": section_id,
            "section_start_line": section_start_line,
            "section_end_line": section_end_line,
            "mode": chosen_mode,
            "owner_session": actor["session_id"],
            "owner_slot": actor_slot,
        }

    def release_file(
        self,
        *,
        session_id: str,
        path: str,
        section_id: str | None = None,
        section_start_line: int | None = None,
        section_end_line: int | None = None,
        all_sections: bool = False,
    ) -> dict[str, Any]:
        normalized_path = normalize_repo_path(
            self.context.repo_root,
            path,
            case_insensitive=self.context.path_case_insensitive,
        )
        section_start_line, section_end_line = self._normalize_section_range(
            section_start_line=section_start_line,
            section_end_line=section_end_line,
        )
        timestamp = utc_iso()
        with self.transaction() as connection:
            self._require_session_row(connection, session_id)
            owner_ids = self._family_session_ids(connection, session_id)
            placeholders = ", ".join("?" for _ in owner_ids)
            if section_id is not None and section_start_line is not None:
                deleted = connection.execute(
                    f"""
                    DELETE FROM file_claims
                    WHERE owner_session IN ({placeholders}) AND path = ? AND section_id = ? AND section_start_line = ? AND section_end_line = ?
                    """,
                    (*owner_ids, normalized_path, section_id, section_start_line, section_end_line),
                ).rowcount
            elif section_id is not None:
                deleted = connection.execute(
                    f"""
                    DELETE FROM file_claims
                    WHERE owner_session IN ({placeholders}) AND path = ? AND section_id = ? AND section_start_line IS NULL AND section_end_line IS NULL
                    """,
                    (*owner_ids, normalized_path, section_id),
                ).rowcount
            elif all_sections:
                deleted = connection.execute(
                    f"DELETE FROM file_claims WHERE owner_session IN ({placeholders}) AND path = ?",
                    (*owner_ids, normalized_path),
                ).rowcount
            else:
                deleted = connection.execute(
                    f"""
                    DELETE FROM file_claims
                    WHERE owner_session IN ({placeholders}) AND path = ? AND section_id IS NULL
                    """,
                    (*owner_ids, normalized_path),
                ).rowcount
            self._record_event(
                connection,
                "release_file",
                {
                    "path": normalized_path,
                    "section_id": section_id,
                    "section_start_line": section_start_line,
                    "section_end_line": section_end_line,
                    "all_sections": all_sections,
                    "deleted": deleted,
                },
                session_id=session_id,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "path": normalized_path,
            "section_id": section_id,
            "section_start_line": section_start_line,
            "section_end_line": section_end_line,
            "all_sections": all_sections,
            "released": int(deleted),
        }

    def update_slot(
        self,
        *,
        session_id: str,
        slot_id: int,
        role: str,
        status: str,
        summary: str,
        work_item_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if slot_id < 1:
            raise CoordinationError("slot ids start at 1")
        if slot_id > REQUIRED_CHILD_COUNT:
            raise CoordinationError(f"slot ids must stay within 1..{REQUIRED_CHILD_COUNT}")
        timestamp = utc_iso()
        with self.transaction() as connection:
            session = self._require_parent_session_row(connection, session_id, active_only=True)
            active_child = self._active_child_row_for_slot(connection, parent_session_id=session_id, slot_id=slot_id)
            if active_child is not None:
                connection.execute(
                    """
                    UPDATE sessions
                    SET role = ?, activity_status = ?, summary = ?, work_item_ids_json = ?, last_heartbeat = ?
                    WHERE session_id = ?
                    """,
                    (
                        role.strip(),
                        status.strip(),
                        summary.strip(),
                        json.dumps(work_item_ids or []),
                        timestamp,
                        active_child["session_id"],
                    ),
                )
                refreshed_child = self._require_child_session_row(connection, active_child["session_id"], active_only=True)
                self._sync_slot_row_from_child(connection, child_session=refreshed_child, timestamp=timestamp)
            else:
                connection.execute(
                    """
                    INSERT INTO slots(session_id, slot_id, child_session_id, role, status, summary, work_item_ids_json, updated_at, last_heartbeat)
                    VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, slot_id) DO UPDATE SET
                        child_session_id = NULL,
                        role = excluded.role,
                        status = excluded.status,
                        summary = excluded.summary,
                        work_item_ids_json = excluded.work_item_ids_json,
                        updated_at = excluded.updated_at,
                        last_heartbeat = excluded.last_heartbeat
                    """,
                    (
                        session_id,
                        slot_id,
                        role.strip(),
                        status.strip(),
                        summary.strip(),
                        json.dumps(work_item_ids or []),
                        timestamp,
                        timestamp,
                    ),
                )
            self._record_event(
                connection,
                "update_slot",
                {
                    "slot_id": slot_id,
                    "role": role.strip(),
                    "status": status.strip(),
                    "summary": summary.strip(),
                    "work_item_ids": work_item_ids or [],
                    "child_session_id": active_child["session_id"] if active_child is not None else None,
                },
                session_id=session_id,
                slot_id=slot_id,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "session_id": session_id,
            "slot_id": slot_id,
            "role": role.strip(),
            "status": status.strip(),
            "summary": summary.strip(),
            "work_item_ids": work_item_ids or [],
        }

    def _parse_slot_row(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = row_to_dict(row)
        payload["work_item_ids"] = parse_json_or_default(payload.pop("work_item_ids_json"), [])
        return payload

    def _parse_work_item_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        payload = row if isinstance(row, dict) else row_to_dict(row)
        payload = dict(payload)
        canonical_key = payload.get("work_item_id", "")
        payload["work_item_key"] = canonical_key
        payload["work_item_id"] = payload.get("raw_work_item_id") or canonical_key
        return payload

    def _parse_message_row(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = row_to_dict(row)
        payload["is_unread"] = payload["ack_at"] is None
        return payload

    def _parse_checkpoint_row(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = row_to_dict(row)
        payload["blockers"] = parse_json_or_default(payload.pop("blocker_json"), [])
        payload["next_actions"] = parse_json_or_default(payload.pop("next_actions_json"), [])
        payload["evidence_paths"] = parse_json_or_default(payload.pop("evidence_paths_json"), [])
        payload["resume_context"] = parse_json_or_default(payload.pop("resume_context_json"), {})
        return payload

    def _parse_event_row(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = row_to_dict(row)
        payload["payload"] = parse_json_or_default(payload.pop("payload_json"), {})
        return payload

    def _session_slots(self, connection: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
        return [
            self._parse_slot_row(row)
            for row in connection.execute(
                "SELECT * FROM slots WHERE session_id = ? ORDER BY slot_id ASC",
                (session_id,),
            ).fetchall()
        ]

    def _session_work_items(self, connection: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
        family_session_ids = self._family_session_ids(connection, session_id)
        placeholders = ", ".join("?" for _ in family_session_ids)
        return [
            self._parse_work_item_row(row)
            for row in connection.execute(
                f"""
                SELECT * FROM work_items
                WHERE created_by_session IN ({placeholders}) OR owner_session IN ({placeholders})
                ORDER BY task_scope ASC, raw_work_item_id ASC
                """,
                (*family_session_ids, *family_session_ids),
            ).fetchall()
        ]

    def _session_file_claims(self, connection: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
        family_session_ids = self._family_session_ids(connection, session_id)
        placeholders = ", ".join("?" for _ in family_session_ids)
        return [
            row_to_dict(row)
            for row in connection.execute(
                f"SELECT * FROM file_claims WHERE owner_session IN ({placeholders}) ORDER BY path, section_id, section_start_line",
                tuple(family_session_ids),
            ).fetchall()
        ]

    def _session_python_leases(self, connection: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
        family_session_ids = self._family_session_ids(connection, session_id)
        placeholders = ", ".join("?" for _ in family_session_ids)
        return [
            row_to_dict(row)
            for row in connection.execute(
                f"SELECT * FROM python_leases WHERE owner_session IN ({placeholders}) ORDER BY started_at ASC",
                tuple(family_session_ids),
            ).fetchall()
        ]

    def _session_messages(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        *,
        include_archived: bool,
    ) -> list[dict[str, Any]]:
        where_archived = "" if include_archived else "AND archived_at IS NULL"
        family_session_ids = self._family_session_ids(connection, session_id)
        placeholders = ", ".join("?" for _ in family_session_ids)
        return [
            self._parse_message_row(row)
            for row in connection.execute(
                f"""
                SELECT * FROM messages
                WHERE (sender_session IN ({placeholders}) OR recipient_session IN ({placeholders}) OR recipient_session IS NULL)
                {where_archived}
                ORDER BY created_at ASC
                """,
                (*family_session_ids, *family_session_ids),
            ).fetchall()
        ]

    def _work_item_counts(self, work_items: list[dict[str, Any]]) -> dict[str, int]:
        counts = {status: 0 for status in WORK_ITEM_STATUSES}
        for item in work_items:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        return counts

    def _build_resume_context(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
        *,
        task_summary: str,
        blockers: list[str],
        next_actions: list[str],
        evidence_paths: list[str],
    ) -> dict[str, Any]:
        session_id = session["session_id"]
        slots = self._session_slots(connection, session_id)
        work_items = self._session_work_items(connection, session_id)
        file_claims = self._session_file_claims(connection, session_id)
        python_leases = self._session_python_leases(connection, session_id)
        child_rows = [self._parse_session_row(row) for row in self._descendant_child_rows(connection, session_id)]
        for child in child_rows:
            child.update(self._session_staleness(child))
            child.update(self._child_health(child))
        child_health = self._parent_child_health(
            connection,
            parent_session_id=session_id,
            required_child_count=int(session["slot_count"] or REQUIRED_CHILD_COUNT),
        )
        return {
            "parent_session_id": session_id,
            "task_summary": task_summary,
            "task_scope": session["task_scope"] or derive_task_scope(task_summary),
            "slot_table": slots,
            "child_roster": child_rows,
            "child_health": child_health,
            "child_notes": self._child_note_rows(connection, parent_session_id=session_id),
            "work_items": work_items,
            "claimed_work_items": [item for item in work_items if item.get("owner_session")],
            "file_claims": file_claims,
            "active_python_leases": [lease for lease in python_leases if lease.get("closed_at") is None],
            "latest_evidence_paths": evidence_paths,
            "blockers": blockers,
            "suggested_next_actions": next_actions,
            "work_item_counts": self._work_item_counts(work_items),
        }

    def post_message(
        self,
        *,
        sender_session: str,
        subject: str,
        body: str,
        recipient_session: str | None = None,
        category: str = "note",
        related_work_item_id: str | None = None,
        related_path: str | None = None,
    ) -> dict[str, Any]:
        if category not in MESSAGE_CATEGORIES:
            raise CoordinationError(f"unsupported message category '{category}'")
        if not subject.strip() or not body.strip():
            raise CoordinationError("message subject and body must be non-empty")
        normalized_path = (
            normalize_repo_path(self.context.repo_root, related_path, case_insensitive=self.context.path_case_insensitive)
            if related_path
            else None
        )
        timestamp = utc_iso()
        with self.transaction() as connection:
            sender = self._require_session_row(connection, sender_session)
            if recipient_session:
                self._require_session_row(connection, recipient_session)
            normalized_related_work_item_id = normalize_work_item_id(related_work_item_id) if related_work_item_id else None
            related_task_scope = sender["task_scope"] if normalized_related_work_item_id else None
            related_work_item_key = (
                make_work_item_key(related_task_scope, normalized_related_work_item_id)
                if normalized_related_work_item_id and related_task_scope
                else None
            )
            cursor = connection.execute(
                """
                INSERT INTO messages(
                    sender_session, recipient_session, category, subject, body,
                    related_work_item_id, related_work_item_key, related_task_scope, related_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sender_session,
                    recipient_session,
                    category,
                    subject.strip(),
                    body.strip(),
                    normalized_related_work_item_id,
                    related_work_item_key,
                    related_task_scope,
                    normalized_path,
                    timestamp,
                ),
            )
            message_id = int(cursor.lastrowid)
            self._record_event(
                connection,
                "post_message",
                {
                    "message_id": message_id,
                    "recipient_session": recipient_session,
                    "category": category,
                    "subject": subject.strip(),
                    "related_work_item_id": normalized_related_work_item_id,
                    "related_work_item_key": related_work_item_key,
                    "related_task_scope": related_task_scope,
                },
                session_id=sender_session,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {"message_id": message_id, "category": category, "recipient_session": recipient_session}

    def ack_message(self, *, session_id: str, message_id: int) -> dict[str, Any]:
        timestamp = utc_iso()
        with self.transaction() as connection:
            self._require_session_row(connection, session_id)
            message = connection.execute(
                "SELECT * FROM messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if message is None:
                raise CoordinationError(f"message '{message_id}' does not exist")
            if message["recipient_session"] and message["recipient_session"] != session_id:
                raise CoordinationError(
                    f"message '{message_id}' is targeted to session '{message['recipient_session']}', not '{session_id}'"
                )
            if message["ack_at"]:
                if message["ack_by"] == session_id:
                    return {"message_id": message_id, "ack_by": session_id, "ack_at": message["ack_at"], "already_acked": True}
                raise CoordinationError(
                    f"message '{message_id}' was already acknowledged by session '{message['ack_by']}'"
                )
            connection.execute(
                "UPDATE messages SET ack_at = ?, ack_by = ? WHERE message_id = ?",
                (timestamp, session_id, message_id),
            )
            self._record_event(
                connection,
                "ack_message",
                {"message_id": message_id},
                session_id=session_id,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {"message_id": message_id, "ack_by": session_id, "ack_at": timestamp}

    def checkpoint(
        self,
        *,
        session_id: str,
        task_summary: str | None = None,
        blockers: list[str] | None = None,
        next_actions: list[str] | None = None,
        evidence_paths: list[str] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        blockers = blockers or []
        next_actions = next_actions or []
        normalized_evidence = [
            normalize_repo_path(self.context.repo_root, path, case_insensitive=self.context.path_case_insensitive)
            for path in (evidence_paths or [])
        ]
        timestamp = utc_iso()
        with self.transaction() as connection:
            session = self._require_parent_session_row(connection, session_id, active_only=True)
            summary = (task_summary or session["task_summary"]).strip()
            resume_context = self._build_resume_context(
                connection,
                session,
                task_summary=summary,
                blockers=blockers,
                next_actions=next_actions,
                evidence_paths=normalized_evidence,
            )
            connection.execute(
                """
                INSERT INTO checkpoints(
                    session_id, task_summary, task_scope, blocker_json, next_actions_json,
                    evidence_paths_json, resume_context_json, note, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    summary,
                    session["task_scope"] or derive_task_scope(summary),
                    json.dumps(blockers),
                    json.dumps(next_actions),
                    json.dumps(normalized_evidence),
                    json.dumps(resume_context, sort_keys=True),
                    note.strip(),
                    timestamp,
                ),
            )
            connection.execute(
                """
                UPDATE sessions
                SET task_summary = ?, note = ?, last_heartbeat = ?, repo_root = ?, worktree_root = ?, repo_identity = ?
                WHERE session_id = ?
                """,
                (
                    summary,
                    append_note(session["note"], note),
                    timestamp,
                    str(self.context.repo_root),
                    str(self.context.repo_root),
                    self.context.repo_identity,
                    session_id,
                ),
            )
            self._record_event(
                connection,
                "checkpoint",
                {
                    "task_scope": session["task_scope"] or derive_task_scope(summary),
                    "task_summary": summary,
                    "blockers": blockers,
                    "next_actions": next_actions,
                    "evidence_paths": normalized_evidence,
                    "resume_context_keys": sorted(resume_context.keys()),
                },
                session_id=session_id,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "session_id": session_id,
            "task_summary": task_summary or "",
            "task_scope": session["task_scope"] or derive_task_scope(summary),
            "created_at": timestamp,
            "resume_context": resume_context,
        }

    def open_python_lease(
        self,
        *,
        session_id: str,
        purpose: str,
        command: str,
        owner_slot: int | None = None,
        lease_id: str | None = None,
        pid: int | None = None,
        memory_cap_bytes: int | None = None,
        memory_cap_percent: float | None = None,
        enforcement_method: str = "record_only",
        status: str = "open",
        note: str = "",
    ) -> dict[str, Any]:
        if not purpose.strip() or not command.strip():
            raise CoordinationError("python lease purpose and command must be non-empty")
        chosen_lease_id = lease_id or default_lease_id()
        timestamp = utc_iso()
        with self.transaction() as connection:
            actor = self._resolve_actor_session(connection, session_id=session_id, owner_slot=owner_slot, active_only=True)
            actor_slot = int(actor["child_slot_id"]) if actor["session_type"] == "child" else owner_slot
            existing = connection.execute(
                "SELECT * FROM python_leases WHERE lease_id = ?",
                (chosen_lease_id,),
            ).fetchone()
            if existing is not None:
                raise CoordinationError(f"python lease '{chosen_lease_id}' already exists")
            connection.execute(
                """
                INSERT INTO python_leases(
                    lease_id, owner_session, owner_slot, purpose, command, pid,
                    started_at, last_heartbeat, memory_cap_bytes, memory_cap_percent,
                    enforcement_method, status, note, closed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    chosen_lease_id,
                    actor["session_id"],
                    actor_slot,
                    purpose.strip(),
                    command.strip(),
                    pid,
                    timestamp,
                    timestamp,
                    memory_cap_bytes,
                    memory_cap_percent,
                    enforcement_method.strip(),
                    status.strip(),
                    note.strip(),
                ),
            )
            self._record_event(
                connection,
                "open_python_lease",
                {
                    "lease_id": chosen_lease_id,
                    "owner_slot": actor_slot,
                    "pid": pid,
                    "purpose": purpose.strip(),
                    "memory_cap_bytes": memory_cap_bytes,
                    "memory_cap_percent": memory_cap_percent,
                    "enforcement_method": enforcement_method.strip(),
                    "status": status.strip(),
                },
                session_id=actor["session_id"],
                slot_id=actor_slot,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "lease_id": chosen_lease_id,
            "status": status.strip(),
            "owner_session": actor["session_id"],
            "owner_slot": actor_slot,
        }

    def _reconcile_python_leases(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT * FROM python_leases
            WHERE closed_at IS NULL
              AND (? IS NULL OR owner_session = ?)
            ORDER BY started_at ASC
            """,
            (session_id, session_id),
        ).fetchall()
        reconciled: list[dict[str, Any]] = []
        timestamp = utc_iso()
        for lease in rows:
            live = pid_is_running(lease["pid"])
            if live is not False:
                continue
            next_status = "lost_pid"
            note = append_note(lease["note"], f"Reconciled dead PID {lease['pid']} at {timestamp}.")
            connection.execute(
                """
                UPDATE python_leases
                SET status = ?, last_heartbeat = ?, note = ?, closed_at = COALESCE(closed_at, ?)
                WHERE lease_id = ?
                """,
                (next_status, timestamp, note, timestamp, lease["lease_id"]),
            )
            self._record_event(
                connection,
                "reconcile_python_lease",
                {
                    "lease_id": lease["lease_id"],
                    "pid": lease["pid"],
                    "status": next_status,
                },
                session_id=lease["owner_session"],
                slot_id=lease["owner_slot"],
                created_at=timestamp,
            )
            reconciled.append({"lease_id": lease["lease_id"], "status": next_status, "pid": lease["pid"]})
        return reconciled

    def touch_python_lease(
        self,
        *,
        lease_id: str,
        status: str | None = None,
        pid: int | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        timestamp = utc_iso()
        with self.transaction() as connection:
            lease = connection.execute(
                "SELECT * FROM python_leases WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
            if lease is None:
                raise CoordinationError(f"python lease '{lease_id}' does not exist")
            if lease["closed_at"]:
                raise CoordinationError(f"python lease '{lease_id}' is already closed")
            connection.execute(
                """
                UPDATE python_leases
                SET status = ?, pid = COALESCE(?, pid), last_heartbeat = ?, note = ?
                WHERE lease_id = ?
                """,
                (
                    status or lease["status"],
                    pid,
                    timestamp,
                    append_note(lease["note"], note),
                    lease_id,
                ),
            )
            self._record_event(
                connection,
                "touch_python_lease",
                {"lease_id": lease_id, "status": status, "pid": pid},
                session_id=lease["owner_session"],
                slot_id=lease["owner_slot"],
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {"lease_id": lease_id, "status": status or "", "pid": pid}

    def close_python_lease(
        self,
        *,
        session_id: str,
        lease_id: str,
        status: str = "closed",
        note: str = "",
    ) -> dict[str, Any]:
        timestamp = utc_iso()
        with self.transaction() as connection:
            self._require_session_row(connection, session_id)
            family_session_ids = self._family_session_ids(connection, session_id)
            lease = connection.execute(
                "SELECT * FROM python_leases WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
            if lease is None:
                raise CoordinationError(f"python lease '{lease_id}' does not exist")
            if lease["owner_session"] not in family_session_ids:
                raise CoordinationError(
                    f"python lease '{lease_id}' is owned by session '{lease['owner_session']}'"
                )
            if lease["closed_at"]:
                return {
                    "lease_id": lease_id,
                    "status": lease["status"],
                    "closed_at": lease["closed_at"],
                    "already_closed": True,
                }
            connection.execute(
                """
                UPDATE python_leases
                SET status = ?, last_heartbeat = ?, note = ?, closed_at = COALESCE(closed_at, ?)
                WHERE lease_id = ?
                """,
                (
                    status.strip(),
                    timestamp,
                    append_note(lease["note"], note),
                    timestamp,
                    lease_id,
                ),
            )
            self._record_event(
                connection,
                "close_python_lease",
                {"lease_id": lease_id, "status": status.strip()},
                session_id=session_id,
                slot_id=lease["owner_slot"],
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {"lease_id": lease_id, "status": status.strip(), "closed_at": timestamp}

    def end_parent(
        self,
        *,
        session_id: str,
        outcome: str = "completed",
        task_summary: str | None = None,
        blockers: list[str] | None = None,
        next_actions: list[str] | None = None,
        evidence_paths: list[str] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        normalized_outcome = normalize_session_outcome(outcome)
        blockers = blockers or []
        next_actions = next_actions or []
        normalized_evidence = [
            normalize_repo_path(self.context.repo_root, path, case_insensitive=self.context.path_case_insensitive)
            for path in (evidence_paths or [])
        ]
        timestamp = utc_iso()
        response: dict[str, Any]
        child_archives: list[str] = []
        with self.transaction() as connection:
            session = self._require_parent_session_row(connection, session_id, active_only=False)
            if session["status"] != "active":
                response = {
                    "session_id": session_id,
                    "outcome": session["outcome"] or session["status"],
                    "ended_at": session["ended_at"] or session["last_heartbeat"],
                    "already_ended": True,
                }
            else:
                child_health = self._parent_child_health(
                    connection,
                    parent_session_id=session_id,
                    required_child_count=int(session["slot_count"] or REQUIRED_CHILD_COUNT),
                )
                if normalized_outcome == "completed" and not child_health["is_child_roster_compliant"]:
                    raise CoordinationError(
                        f"parent session '{session_id}' cannot complete while child roster is noncompliant "
                        f"({child_health['live_child_count']}/{child_health['required_child_count']} live children)"
                    )
                summary = (task_summary or session["task_summary"]).strip()
                resume_context = self._build_resume_context(
                    connection,
                    session,
                    task_summary=summary,
                    blockers=blockers,
                    next_actions=next_actions,
                    evidence_paths=normalized_evidence,
                )
                connection.execute(
                    """
                    INSERT INTO checkpoints(
                        session_id, task_summary, task_scope, blocker_json, next_actions_json,
                        evidence_paths_json, resume_context_json, note, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        summary,
                        session["task_scope"] or derive_task_scope(summary),
                        json.dumps(blockers),
                        json.dumps(next_actions),
                        json.dumps(normalized_evidence),
                        json.dumps(resume_context, sort_keys=True),
                        append_note("final checkpoint", note),
                        timestamp,
                    ),
                )
                family_session_ids = self._family_session_ids(connection, session_id)
                placeholders = ", ".join("?" for _ in family_session_ids)
                active_children = [
                    row
                    for row in self._descendant_child_rows(connection, session_id)
                    if row["status"] == "active"
                ]
                for child in active_children:
                    connection.execute("DELETE FROM file_claims WHERE owner_session = ?", (child["session_id"],))
                    connection.execute(
                        """
                        UPDATE python_leases
                        SET status = CASE WHEN closed_at IS NULL THEN 'released_by_end_parent' ELSE status END,
                            closed_at = COALESCE(closed_at, ?),
                            last_heartbeat = ?,
                            note = CASE
                                WHEN note = '' THEN ?
                                ELSE note || CHAR(10) || ?
                            END
                        WHERE owner_session = ?
                        """,
                        (
                            timestamp,
                            timestamp,
                            f"Closed during parent cleanup ({normalized_outcome}).",
                            f"Closed during parent cleanup ({normalized_outcome}).",
                            child["session_id"],
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE messages
                        SET archived_at = COALESCE(archived_at, ?)
                        WHERE archived_at IS NULL AND (sender_session = ? OR recipient_session = ?)
                        """,
                        (timestamp, child["session_id"], child["session_id"]),
                    )
                    connection.execute(
                        """
                        UPDATE sessions
                        SET status = ?, outcome = ?, ended_at = ?, last_heartbeat = ?, terminal_reason = ?, note = ?
                        WHERE session_id = ?
                        """,
                        (
                            normalized_outcome,
                            normalized_outcome,
                            timestamp,
                            timestamp,
                            f"Closed during end-parent for {session_id}.",
                            append_note(child["note"], f"Closed during end-parent for {session_id}."),
                            child["session_id"],
                        ),
                    )
                    ended_child = self._require_child_session_row(connection, child["session_id"], active_only=False)
                    self._record_child_note(
                        connection,
                        child_session=ended_child,
                        category="end",
                        summary=f"Closed during end-parent for {session_id}.",
                        timestamp=timestamp,
                    )
                    self._record_event(
                        connection,
                        "end_child",
                        {
                            "parent_session_id": session_id,
                            "slot_id": child["child_slot_id"],
                            "outcome": normalized_outcome,
                            "release_work_to_parent": False,
                        },
                        session_id=child["session_id"],
                        slot_id=int(child["child_slot_id"]),
                        created_at=timestamp,
                    )
                    child_archives.append(child["session_id"])
                connection.execute(
                    f"""
                    UPDATE work_items
                    SET owner_session = NULL,
                        owner_slot = NULL,
                        status = CASE WHEN status = 'closed' THEN 'closed' ELSE 'open' END,
                        updated_at = ?,
                        latest_note = CASE
                            WHEN latest_note = '' THEN ?
                            ELSE latest_note || CHAR(10) || ?
                        END
                    WHERE owner_session IN ({placeholders})
                    """,
                    (
                        timestamp,
                        f"Released during end-parent ({normalized_outcome}).",
                        f"Released during end-parent ({normalized_outcome}).",
                        *family_session_ids,
                    ),
                )
                connection.execute(
                    f"DELETE FROM file_claims WHERE owner_session IN ({placeholders})",
                    tuple(family_session_ids),
                )
                connection.execute(
                    f"""
                    UPDATE python_leases
                    SET status = CASE WHEN closed_at IS NULL THEN 'released_by_end_parent' ELSE status END,
                        closed_at = COALESCE(closed_at, ?),
                        last_heartbeat = ?,
                        note = CASE
                            WHEN note = '' THEN ?
                            ELSE note || CHAR(10) || ?
                        END
                    WHERE owner_session IN ({placeholders})
                    """,
                    (
                        timestamp,
                        timestamp,
                        f"Closed during end-parent ({normalized_outcome}).",
                        f"Closed during end-parent ({normalized_outcome}).",
                        *family_session_ids,
                    ),
                )
                connection.execute(
                    f"""
                    UPDATE messages
                    SET archived_at = COALESCE(archived_at, ?)
                    WHERE archived_at IS NULL
                      AND (sender_session IN ({placeholders}) OR recipient_session IN ({placeholders}))
                    """,
                    (timestamp, *family_session_ids, *family_session_ids),
                )
                connection.execute(
                    """
                    UPDATE sessions
                    SET status = ?, outcome = ?, task_summary = ?, ended_at = ?, last_heartbeat = ?, note = ?,
                        terminal_reason = ?, repo_root = ?, worktree_root = ?, repo_identity = ?
                    WHERE session_id = ?
                    """,
                    (
                        normalized_outcome,
                        normalized_outcome,
                        summary,
                        timestamp,
                        timestamp,
                        append_note(session["note"], note),
                        note.strip(),
                        str(self.context.repo_root),
                        str(self.context.repo_root),
                        self.context.repo_identity,
                        session_id,
                    ),
                )
                self._record_event(
                    connection,
                    "end_parent",
                    {
                        "outcome": normalized_outcome,
                        "task_summary": summary,
                        "task_scope": session["task_scope"] or derive_task_scope(summary),
                        "blockers": blockers,
                        "next_actions": next_actions,
                        "evidence_paths": normalized_evidence,
                        "resume_context_keys": sorted(resume_context.keys()),
                    },
                    session_id=session_id,
                    created_at=timestamp,
                )
                response = {"session_id": session_id, "outcome": normalized_outcome, "ended_at": timestamp}
        self.refresh_snapshots()
        for child_session_id in child_archives:
            self.write_archive(child_session_id)
        self.write_archive(session_id)
        return response

    def resume_parent(
        self,
        *,
        from_session_id: str,
        session_id: str | None = None,
        owner: str | None = None,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    ) -> dict[str, Any]:
        chosen_session_id = session_id or default_session_id()
        timestamp = utc_iso()
        with self.transaction() as connection:
            source = self._require_parent_session_row(connection, from_session_id, active_only=False)
            if source["status"] == "active":
                raise CoordinationError(
                    f"session '{from_session_id}' is still active; reap stale state or end it before resuming"
                )
            checkpoint = connection.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC, checkpoint_id DESC LIMIT 1",
                (from_session_id,),
            ).fetchone()
            if checkpoint is None:
                raise CoordinationError(f"session '{from_session_id}' has no checkpoint to resume from")
            parsed_checkpoint = self._parse_checkpoint_row(checkpoint)
            resume_context = parsed_checkpoint["resume_context"]
            summary = parsed_checkpoint["task_summary"]
            task_scope = source["task_scope"] or parsed_checkpoint["task_scope"] or derive_task_scope(summary)
            slot_count = REQUIRED_CHILD_COUNT
            existing = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (chosen_session_id,),
            ).fetchone()
            if existing is not None:
                raise CoordinationError(f"session '{chosen_session_id}' already exists")
            connection.execute(
                """
                INSERT INTO sessions(
                    session_id, session_type, parent_session_id, child_slot_id, repo_root, worktree_root, repo_identity,
                    owner, task_summary, task_scope, task_scope_source, role, agent_name, agent_kind,
                    activity_status, summary, work_item_ids_json, external_agent_id, status, started_at,
                    last_heartbeat, stale_after_seconds, slot_count, outcome, ended_at, note, resume_from_session
                )
                VALUES (?, 'parent', NULL, NULL, ?, ?, ?, ?, ?, ?, 'resume', '', '', '', '', '', '[]', '', 'active', ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    chosen_session_id,
                    str(self.context.repo_root),
                    str(self.context.repo_root),
                    self.context.repo_identity,
                    owner,
                    summary,
                    task_scope,
                    timestamp,
                    timestamp,
                    stale_after_seconds,
                    slot_count,
                    f"Resumed from {from_session_id}.",
                    from_session_id,
                ),
            )
            slot_table = resume_context.get("slot_table") or []
            for slot_id in range(1, slot_count + 1):
                prior = next((item for item in slot_table if int(item.get("slot_id", 0)) == slot_id), None)
                connection.execute(
                    """
                    INSERT INTO slots(session_id, slot_id, child_session_id, role, status, summary, work_item_ids_json, updated_at, last_heartbeat)
                    VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chosen_session_id,
                        slot_id,
                        (prior or {}).get("role", "Unassigned"),
                        "missing_child",
                        f"Resumed from {from_session_id}: replace or start live child for slot {slot_id}. Prior summary: {(prior or {}).get('summary', 'Awaiting assignment')}",
                        json.dumps((prior or {}).get("work_item_ids", [])),
                        timestamp,
                        timestamp,
                    ),
                )
            connection.execute(
                "UPDATE sessions SET resumed_by_session = ? WHERE session_id = ?",
                (chosen_session_id, from_session_id),
            )
            self._record_event(
                connection,
                "resume_parent",
                {
                    "from_session_id": from_session_id,
                    "task_scope": task_scope,
                    "slot_count": slot_count,
                },
                session_id=chosen_session_id,
                created_at=timestamp,
            )
        self.refresh_snapshots()
        return {
            "session_id": chosen_session_id,
            "task_summary": summary,
            "task_scope": task_scope,
            "resume_from_session": from_session_id,
            "slot_count": slot_count,
            "required_child_count": slot_count,
            "resume_context": resume_context,
        }

    def reap_stale(
        self,
        *,
        requestor_session: str,
        target_session: str | None = None,
        takeover_session: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        timestamp = utc_iso()
        reaped: list[dict[str, Any]] = []
        already_terminal: list[dict[str, Any]] = []
        terminal_target: str | None = None
        child_archives: list[str] = []
        with self.transaction() as connection:
            if requestor_session != "manual":
                self._require_session_row(connection, requestor_session)
            if takeover_session:
                self._require_parent_session_row(connection, takeover_session, active_only=True)
            if target_session:
                target_row = self._require_parent_session_row(connection, target_session, active_only=False)
                if target_row["status"] != "active":
                    already_terminal.append(
                        {
                            "session_id": target_session,
                            "status": target_row["status"],
                            "ended_at": target_row["ended_at"] or target_row["last_heartbeat"],
                        }
                    )
                    terminal_target = target_session
            candidates = connection.execute(
                "SELECT * FROM sessions WHERE status = 'active' AND session_type = 'parent' ORDER BY started_at ASC"
            ).fetchall()
            for row in ([] if terminal_target else candidates):
                if target_session and row["session_id"] != target_session:
                    continue
                cutoff = parse_utc(row["last_heartbeat"]) + timedelta(seconds=int(row["stale_after_seconds"]))
                worktree_root = Path((row["worktree_root"] or row["repo_root"] or self.context.repo_root))
                worktree_exists = worktree_root.exists()
                stale_reasons: list[str] = []
                if cutoff < utc_now():
                    stale_reasons.append("heartbeat_overdue")
                if not worktree_exists:
                    stale_reasons.append("worktree_missing")
                if not stale_reasons:
                    if target_session == row["session_id"]:
                        raise CoordinationError(
                            f"session '{target_session}' is still fresh until {utc_iso(cutoff)}"
                        )
                    continue
                target = row["session_id"]
                family_session_ids = self._family_session_ids(connection, target)
                family_placeholders = ", ".join("?" for _ in family_session_ids)
                active_children = [
                    child
                    for child in self._descendant_child_rows(connection, target)
                    if child["status"] == "active"
                ]
                work_items = connection.execute(
                    f"SELECT * FROM work_items WHERE owner_session IN ({family_placeholders})",
                    tuple(family_session_ids),
                ).fetchall()
                if takeover_session:
                    for item in work_items:
                        next_status = item["status"] if item["status"] in {"blocked", "qa"} else "claimed"
                        connection.execute(
                            """
                            UPDATE work_items
                            SET owner_session = ?, owner_slot = NULL, status = ?,
                                claimed_at = ?, updated_at = ?, latest_note = ?
                            WHERE work_item_id = ?
                            """,
                            (
                                takeover_session,
                                next_status,
                                timestamp,
                                timestamp,
                                append_note(item["latest_note"], f"Taken over from stale session {target} by {takeover_session}."),
                                item["work_item_id"],
                            ),
                        )
                else:
                    for item in work_items:
                        next_status = "closed" if item["status"] == "closed" else "open"
                        connection.execute(
                            """
                            UPDATE work_items
                            SET owner_session = NULL, owner_slot = NULL, status = ?, updated_at = ?, latest_note = ?
                            WHERE work_item_id = ?
                            """,
                            (
                                next_status,
                                timestamp,
                                append_note(item["latest_note"], f"Released after stale reap of session {target}."),
                                item["work_item_id"],
                            ),
                        )
                connection.execute(
                    f"DELETE FROM file_claims WHERE owner_session IN ({family_placeholders})",
                    tuple(family_session_ids),
                )
                connection.execute(
                    f"""
                    UPDATE python_leases
                    SET status = CASE WHEN closed_at IS NULL THEN 'reaped' ELSE status END,
                        closed_at = COALESCE(closed_at, ?),
                        last_heartbeat = ?,
                        note = CASE
                            WHEN note = '' THEN ?
                            ELSE note || CHAR(10) || ?
                        END
                    WHERE owner_session IN ({family_placeholders})
                    """,
                    (
                        timestamp,
                        timestamp,
                        f"Closed during stale reap by {requestor_session}.",
                        f"Closed during stale reap by {requestor_session}.",
                        *family_session_ids,
                    ),
                )
                connection.execute(
                    f"""
                    UPDATE messages
                    SET archived_at = COALESCE(archived_at, ?)
                    WHERE archived_at IS NULL
                      AND (sender_session IN ({family_placeholders}) OR recipient_session IN ({family_placeholders}))
                    """,
                    (timestamp, *family_session_ids, *family_session_ids),
                )
                for child in active_children:
                    connection.execute(
                        """
                        UPDATE sessions
                        SET status = ?, outcome = ?, ended_at = ?, last_heartbeat = ?, reaped_by_session = ?,
                            takeover_by_session = ?, terminal_reason = ?, note = CASE
                                WHEN note = '' THEN ?
                                ELSE note || CHAR(10) || ?
                            END
                        WHERE session_id = ?
                        """,
                        (
                            "taken_over" if takeover_session else "reaped",
                            "taken_over" if takeover_session else "reaped",
                            timestamp,
                            timestamp,
                            requestor_session,
                            takeover_session,
                            f"Parent {target} was reaped as stale by {requestor_session}.",
                            f"Reaped with parent {target} by {requestor_session}. {note}".strip(),
                            f"Reaped with parent {target} by {requestor_session}. {note}".strip(),
                            child["session_id"],
                        ),
                    )
                    self._record_event(
                        connection,
                        "reap_stale_child",
                        {
                            "requestor_session": requestor_session,
                            "target_parent_session": target,
                            "target_child_session": child["session_id"],
                            "takeover_session": takeover_session,
                            "stale_reasons": stale_reasons,
                        },
                        session_id=child["session_id"],
                        slot_id=int(child["child_slot_id"]),
                        created_at=timestamp,
                    )
                    child_archives.append(child["session_id"])
                connection.execute(
                    """
                    UPDATE sessions
                    SET status = ?,
                        outcome = ?,
                        ended_at = ?,
                        last_heartbeat = ?,
                        reaped_by_session = ?,
                        takeover_by_session = ?,
                        terminal_reason = ?,
                        note = CASE
                            WHEN note = '' THEN ?
                            ELSE note || CHAR(10) || ?
                        END
                    WHERE session_id = ?
                    """,
                    (
                        "taken_over" if takeover_session else "reaped",
                        "taken_over" if takeover_session else "reaped",
                        timestamp,
                        timestamp,
                        requestor_session,
                        takeover_session,
                        f"Explicit stale reap by {requestor_session}; reasons={','.join(stale_reasons)}. {note}".strip(),
                        f"Reaped as stale by {requestor_session}. {note}".strip(),
                        f"Reaped as stale by {requestor_session}. {note}".strip(),
                        target,
                    ),
                )
                self._record_event(
                    connection,
                    "reap_stale",
                    {
                        "requestor_session": requestor_session,
                        "target_session": target,
                        "takeover_session": takeover_session,
                        "stale_reasons": stale_reasons,
                        "stale_deadline": utc_iso(cutoff),
                    },
                    session_id=target,
                    created_at=timestamp,
                )
                if takeover_session:
                    self._record_event(
                        connection,
                        "takeover_session_adopted",
                        {
                            "from_session": target,
                            "reassigned_work_item_keys": [item["work_item_id"] for item in work_items],
                        },
                        session_id=takeover_session,
                        created_at=timestamp,
                    )
                reaped.append(
                    {
                        "session_id": target,
                        "takeover_session": takeover_session,
                        "reaped_at": timestamp,
                        "stale_reasons": stale_reasons,
                        "stale_deadline": utc_iso(cutoff),
                    }
                )
            if target_session and not reaped and not terminal_target:
                raise CoordinationError(f"no stale session matched '{target_session}'")
        self.refresh_snapshots()
        if terminal_target:
            self.write_archive(terminal_target)
            return {"reaped": [], "count": 0, "already_terminal": already_terminal}
        for child_session_id in child_archives:
            self.write_archive(child_session_id)
        for item in reaped:
            self.write_archive(item["session_id"])
        return {"reaped": reaped, "count": len(reaped), "already_terminal": already_terminal}

    def _session_staleness(self, session: dict[str, Any]) -> dict[str, Any]:
        worktree_root = Path(session.get("worktree_root") or session.get("repo_root") or self.context.repo_root)
        cutoff = parse_utc(session["last_heartbeat"]) + timedelta(seconds=int(session["stale_after_seconds"]))
        seconds_until_stale = int((cutoff - utc_now()).total_seconds())
        stale_signals: list[str] = []
        if session["status"] == "active" and seconds_until_stale < 0:
            stale_signals.append("heartbeat_overdue")
        if session["status"] == "active" and not worktree_root.exists():
            stale_signals.append("worktree_missing")
        risk = "stale" if stale_signals else ("at_risk" if session["status"] == "active" and seconds_until_stale <= DEFAULT_HEARTBEAT_INTERVAL_SECONDS else "fresh")
        return {
            "stale_deadline": utc_iso(cutoff),
            "seconds_until_stale": seconds_until_stale,
            "stale_signals": stale_signals,
            "stale_risk": risk,
            "worktree_exists": worktree_root.exists(),
        }

    def _decorate_python_lease(self, lease: dict[str, Any]) -> dict[str, Any]:
        payload = dict(lease)
        if payload.get("closed_at"):
            payload["health"] = "closed"
        elif payload.get("pid") is None:
            payload["health"] = "missing_pid"
        else:
            liveness = pid_is_running(int(payload["pid"]))
            payload["health"] = "running" if liveness is True else "dead_pid" if liveness is False else "unknown_pid"
        return payload

    def doctor(self) -> dict[str, Any]:
        snapshot = self.collect_status()
        with self._connect() as connection:
            integrity_row = connection.execute("PRAGMA integrity_check").fetchone()
            schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            ended_sessions = [
                row_to_dict(row)
                for row in connection.execute("SELECT session_id, status, ended_at, last_heartbeat FROM sessions WHERE status != 'active'")
            ]
        integrity_ok = integrity_row is not None and integrity_row[0] == "ok"
        manifest_path = self.context.snapshots_dir / "snapshot_manifest.json"
        snapshot_issues: list[str] = []
        if not manifest_path.exists():
            snapshot_issues.append("snapshot manifest is missing")
        else:
            manifest = parse_json_or_default(manifest_path.read_text(encoding="utf-8"), {})
            for name in ("active_sessions.json", "work_item_claims.json", "file_claims.json", "python_leases.json", "messages.json", "events.json"):
                payload_path = self.context.snapshots_dir / name
                if not payload_path.exists():
                    snapshot_issues.append(f"missing snapshot file: {name}")
                    continue
                payload = parse_json_or_default(payload_path.read_text(encoding="utf-8"), {})
                if payload.get("snapshot_generation") != manifest.get("snapshot_generation"):
                    snapshot_issues.append(f"snapshot generation drift detected in {name}")
                if payload.get("state_revision") != manifest.get("state_revision"):
                    snapshot_issues.append(f"snapshot state revision drift detected in {name}")
        stale_parent_sessions = [
            {
                "session_id": item["session"]["session_id"],
                "stale_signals": item["session"]["stale_signals"],
                "stale_deadline": item["session"]["stale_deadline"],
            }
            for item in snapshot["active_sessions"]
            if item["session"]["stale_signals"]
        ]
        stale_child_sessions = [
            {
                "session_id": item["session"]["session_id"],
                "parent_session_id": item["session"]["parent_session_id"],
                "slot_id": item["session"]["child_slot_id"],
                "stale_signals": item["session"]["stale_signals"],
                "stale_deadline": item["session"]["stale_deadline"],
            }
            for item in snapshot.get("active_child_sessions", [])
            if item["session"]["stale_signals"]
        ]
        stale_claims = [
            claim
            for claim in snapshot["file_claims"]
            if snapshot["sessions"].get(claim["owner_session"], {}).get("session", {}).get("stale_signals")
        ]
        stale_python_leases = [
            lease
            for lease in snapshot["python_leases"]
            if lease.get("status") == "lost_pid" or lease.get("health") in {"dead_pid", "unknown_pid"}
        ]
        orphaned_archives = []
        for session in ended_sessions:
            archive_root = self.context.archive_dir / session["session_id"]
            if not (archive_root / "final.json").exists() or not (archive_root / "final.md").exists():
                orphaned_archives.append(session["session_id"])
        child_claims_at_risk = [
            claim
            for claim in snapshot["file_claims"]
            if claim.get("owner_child_session")
            and (
                snapshot["sessions"].get(claim["owner_session"], {}).get("session", {}).get("stale_signals")
                or snapshot["sessions"]
                .get(claim.get("owner_parent_session") or "", {})
                .get("child_health", {})
                .get("child_compliance")
                == "noncompliant"
            )
        ]
        return {
            "generated_at": utc_iso(),
            "repo_identity": self.context.repo_identity,
            "runtime_root": str(self.context.runtime_root),
            "db_path": str(self.context.db_path),
            "schema_version": schema_version,
            "schema_version_ok": schema_version == SCHEMA_VERSION,
            "integrity_check": integrity_row[0] if integrity_row else "missing",
            "integrity_ok": integrity_ok,
            "snapshot_issues": snapshot_issues,
            "stale_sessions": stale_parent_sessions,
            "stale_parent_sessions": stale_parent_sessions,
            "stale_child_sessions": stale_child_sessions,
            "stale_claims": stale_claims,
            "stale_python_leases": stale_python_leases,
            "parent_child_invariant_violations": snapshot.get("parent_child_invariant_violations", []),
            "missing_children": [
                {
                    "session_id": item["session_id"],
                    "missing_child_count": item["missing_child_count"],
                    "missing_child_slots": item["missing_child_slots"],
                }
                for item in snapshot.get("parent_child_invariant_violations", [])
                if item["missing_child_count"] > 0
            ],
            "unhealthy_children": [
                child
                for item in snapshot.get("parent_child_invariant_violations", [])
                for child in item["unhealthy_children"]
            ],
            "child_claims_at_risk": child_claims_at_risk,
            "orphaned_archives": orphaned_archives,
            "active_session_count": len(snapshot["active_sessions"]),
            "active_child_session_count": len(snapshot.get("active_child_sessions", [])),
            "noncompliant_parent_count": len(snapshot.get("parent_child_invariant_violations", [])),
            "claimed_work_item_count": len(snapshot["claimed_work_items"]),
            "open_python_lease_count": len(snapshot["python_leases"]),
        }

    def repair(self) -> dict[str, Any]:
        with self.transaction() as connection:
            reconciled_python_leases = self._reconcile_python_leases(connection)
            ended_sessions = [
                row["session_id"]
                for row in connection.execute("SELECT session_id FROM sessions WHERE status != 'active'").fetchall()
            ]
        snapshot = self.refresh_snapshots()
        archived_sessions: list[str] = []
        for session_id in ended_sessions:
            archive_root = self.context.archive_dir / session_id
            if not (archive_root / "final.json").exists() or not (archive_root / "final.md").exists():
                self.write_archive(session_id)
                archived_sessions.append(session_id)
        return {
            "reconciled_python_leases": reconciled_python_leases,
            "archived_sessions": archived_sessions,
            "snapshot_generation": snapshot["snapshot_generation"],
            "state_revision": snapshot["state_revision"],
        }

    def collect_status(self) -> dict[str, Any]:
        with self.transaction() as connection:
            reconciled_python_leases = self._reconcile_python_leases(connection)
        with self._connect() as connection:
            schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            state_revision = int(connection.execute("SELECT COALESCE(MAX(event_id), 0) FROM events").fetchone()[0])
            messages = [
                self._parse_message_row(row)
                for row in connection.execute("SELECT * FROM messages WHERE archived_at IS NULL ORDER BY created_at ASC").fetchall()
            ]
            checkpoints_by_session: dict[str, list[dict[str, Any]]] = {}
            for row in connection.execute("SELECT * FROM checkpoints ORDER BY created_at DESC, checkpoint_id DESC").fetchall():
                parsed = self._parse_checkpoint_row(row)
                checkpoints_by_session.setdefault(parsed["session_id"], []).append(parsed)
            events = [
                self._parse_event_row(row)
                for row in connection.execute("SELECT * FROM events ORDER BY event_id DESC LIMIT 200").fetchall()
            ]
            raw_sessions = [
                self._parse_session_row(row)
                for row in connection.execute("SELECT * FROM sessions ORDER BY started_at ASC, session_id ASC").fetchall()
            ]
            for session in raw_sessions:
                session["task_scope"] = session.get("task_scope") or derive_task_scope(session["task_summary"])
                session["worktree_root"] = session.get("worktree_root") or session["repo_root"]
                session["repo_identity"] = session.get("repo_identity") or self.context.repo_identity
                session.update(self._session_staleness(session))
            sessions_by_id = {session["session_id"]: session for session in raw_sessions}

            def enrich_owner(payload: dict[str, Any]) -> dict[str, Any]:
                enriched = dict(payload)
                owner = sessions_by_id.get(enriched.get("owner_session"))
                if owner is None:
                    enriched["owner_session_type"] = None
                    enriched["owner_parent_session"] = None
                    enriched["owner_child_session"] = None
                    enriched["owner_agent_name"] = ""
                    enriched["owner_role"] = ""
                else:
                    enriched["owner_session_type"] = owner["session_type"]
                    enriched["owner_parent_session"] = owner["parent_session_id"] or owner["session_id"]
                    enriched["owner_child_session"] = owner["session_id"] if owner["session_type"] == "child" else None
                    enriched["owner_agent_name"] = owner.get("agent_name") or ""
                    enriched["owner_role"] = owner.get("role") or ""
                return enriched

            all_work_items = [
                enrich_owner(self._parse_work_item_row(row))
                for row in connection.execute("SELECT * FROM work_items ORDER BY task_scope ASC, raw_work_item_id ASC").fetchall()
            ]
            file_claims = [
                enrich_owner(row_to_dict(row))
                for row in connection.execute(
                    "SELECT * FROM file_claims ORDER BY path ASC, section_id ASC, section_start_line ASC"
                ).fetchall()
            ]
            all_python_leases = [
                enrich_owner(self._decorate_python_lease(row_to_dict(row)))
                for row in connection.execute("SELECT * FROM python_leases ORDER BY started_at ASC").fetchall()
            ]
            child_notes_by_child: dict[str, list[dict[str, Any]]] = {}
            child_notes_by_parent: dict[str, list[dict[str, Any]]] = {}
            for note in self._child_note_rows(connection):
                child_notes_by_child.setdefault(note["child_session_id"], []).append(note)
                child_notes_by_parent.setdefault(note["parent_session_id"], []).append(note)
            session_snapshots: dict[str, dict[str, Any]] = {}
            active_sessions: list[dict[str, Any]] = []
            active_child_sessions: list[dict[str, Any]] = []
            parent_child_invariant_violations: list[dict[str, Any]] = []
            for session in raw_sessions:
                session_id = session["session_id"]
                session_work = [enrich_owner(item) for item in self._session_work_items(connection, session_id)]
                session_file_claims = [enrich_owner(item) for item in self._session_file_claims(connection, session_id)]
                session_python_leases = [enrich_owner(self._decorate_python_lease(item)) for item in self._session_python_leases(connection, session_id)]
                inbox_messages = [message for message in messages if message["recipient_session"] in (None, session_id)]
                unread_messages = [message for message in inbox_messages if message["ack_at"] is None]
                handoff_messages = [message for message in inbox_messages if message["category"] == "handoff"]
                child_health = None
                child_roster: list[dict[str, Any]] = []
                if session["session_type"] == "parent":
                    child_health = self._parent_child_health(
                        connection,
                        parent_session_id=session_id,
                        required_child_count=int(session["slot_count"] or REQUIRED_CHILD_COUNT),
                    )
                    child_roster = []
                    for child_row in self._descendant_child_rows(connection, session_id):
                        child = sessions_by_id[child_row["session_id"]]
                        child.update(self._child_health(child))
                        child_roster.append(child)
                    session["child_health"] = child_health
                    if session["status"] == "active" and not child_health["is_child_roster_compliant"]:
                        parent_child_invariant_violations.append(
                            {
                                "session_id": session_id,
                                "required_child_count": child_health["required_child_count"],
                                "live_child_count": child_health["live_child_count"],
                                "missing_child_count": child_health["missing_child_count"],
                                "missing_child_slots": child_health["missing_child_slots"],
                                "unhealthy_child_count": child_health["unhealthy_child_count"],
                                "unhealthy_children": [
                                    {
                                        "session_id": child["session_id"],
                                        "slot_id": child["child_slot_id"],
                                        "health": child["health"],
                                        "health_issues": child["health_issues"],
                                    }
                                    for child in child_health["unhealthy_children"]
                                ],
                            }
                        )
                session_snapshot = {
                    "session": session,
                    "slots": self._session_slots(connection, session_id),
                    "children": child_roster,
                    "child_notes": child_notes_by_parent.get(session_id, []) if session["session_type"] == "parent" else child_notes_by_child.get(session_id, []),
                    "child_health": child_health,
                    "work_items": session_work,
                    "work_item_counts": self._work_item_counts(session_work),
                    "file_claims": session_file_claims,
                    "python_leases": session_python_leases,
                    "messages": inbox_messages,
                    "unread_messages": unread_messages,
                    "handoff_messages": handoff_messages,
                    "latest_checkpoint": checkpoints_by_session.get(session_id, [None])[0] if session["session_type"] == "parent" else None,
                    "latest_child_note": (child_notes_by_child.get(session_id) or [None])[0] if session["session_type"] == "child" else None,
                }
                session_snapshots[session_id] = session_snapshot
                if session["status"] == "active" and session["session_type"] == "parent":
                    active_sessions.append(session_snapshot)
                elif session["status"] == "active" and session["session_type"] == "child":
                    active_child_sessions.append(session_snapshot)
        return {
            "generated_at": utc_iso(),
            "repo_root": str(self.context.repo_root),
            "repo_identity": self.context.repo_identity,
            "runtime_root": str(self.context.runtime_root),
            "git_common_dir": str(self.context.git_common_dir) if self.context.git_common_dir else None,
            "uses_git_common_dir": self.context.uses_git_common_dir,
            "db_path": str(self.context.db_path),
            "schema_version": schema_version,
            "state_revision": state_revision,
            "reconciled_python_leases": reconciled_python_leases,
            "active_sessions": active_sessions,
            "active_child_sessions": active_child_sessions,
            "sessions": session_snapshots,
            "claimed_work_items": [item for item in all_work_items if item["owner_session"]],
            "work_items": all_work_items,
            "file_claims": file_claims,
            "python_leases": [lease for lease in all_python_leases if lease["closed_at"] is None],
            "messages": messages,
            "unread_messages": [message for message in messages if message["ack_at"] is None],
            "handoff_messages": [message for message in messages if message["category"] == "handoff"],
            "events": events,
            "parent_child_invariant_violations": parent_child_invariant_violations,
        }

    def render_status_board(self, snapshot: dict[str, Any]) -> str:
        lines = [
            "# Codex Coordination Status",
            "",
            f"- Generated: `{snapshot['generated_at']}`",
            f"- Snapshot generation: `{snapshot.get('snapshot_generation', '<pending>')}`",
            f"- State revision: `{snapshot['state_revision']}`",
            f"- Schema version: `{snapshot['schema_version']}`",
            f"- Repo identity: `{snapshot['repo_identity']}`",
            f"- Worktree root: `{snapshot['repo_root']}`",
            f"- Runtime root: `{snapshot['runtime_root']}`",
            f"- Database: `{snapshot['db_path']}`",
            f"- Git common dir: `{snapshot['git_common_dir'] or 'fallback local runtime'}`",
            "",
            "## Active Parent Sessions",
        ]
        if not snapshot["active_sessions"]:
            lines.extend(["", "No active parent sessions."])
        else:
            lines.extend(
                [
                    "",
                    "| Session | Scope | Heartbeat | Compliance | Live Children | Missing | Unhealthy | Stale Risk | Worktree | Open | Claimed | In Progress | Blocked | QA | Closed |",
                    "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for item in snapshot["active_sessions"]:
                session = item["session"]
                counts = item["work_item_counts"]
                child_health = item["child_health"] or {}
                stale = ", ".join(session["stale_signals"]) if session["stale_signals"] else session["stale_risk"]
                worktree = "ok" if session["worktree_exists"] else "missing"
                lines.append(
                    "| {session_id} | {scope} | {heartbeat} | {compliance} | {live} / {required} | {missing} | {unhealthy} | {stale} | {worktree} | {open} | {claimed} | {in_progress} | {blocked} | {qa} | {closed} |".format(
                        session_id=session["session_id"],
                        scope=session["task_scope"],
                        heartbeat=session["last_heartbeat"],
                        compliance=child_health.get("child_compliance", "noncompliant"),
                        live=child_health.get("live_child_count", 0),
                        required=child_health.get("required_child_count", REQUIRED_CHILD_COUNT),
                        missing=child_health.get("missing_child_count", REQUIRED_CHILD_COUNT),
                        unhealthy=child_health.get("unhealthy_child_count", 0),
                        stale=stale,
                        worktree=worktree,
                        open=counts.get("open", 0),
                        claimed=counts.get("claimed", 0),
                        in_progress=counts.get("in_progress", 0),
                        blocked=counts.get("blocked", 0),
                        qa=counts.get("qa", 0),
                        closed=counts.get("closed", 0),
                    )
                )
            for item in snapshot["active_sessions"]:
                session = item["session"]
                child_health = item["child_health"] or {}
                lines.extend(
                    [
                        "",
                        f"### Session: `{session['session_id']}`",
                        "",
                        f"- Task: {session['task_summary']}",
                        f"- Scope: `{session['task_scope']}`",
                        f"- Stale deadline: `{session['stale_deadline']}`",
                        f"- Seconds until stale: `{session['seconds_until_stale']}`",
                        f"- Child compliance: `{child_health.get('child_compliance', 'noncompliant')}`",
                        f"- Required child count: `{child_health.get('required_child_count', REQUIRED_CHILD_COUNT)}`",
                        f"- Live child count: `{child_health.get('live_child_count', 0)}`",
                        f"- Missing child count: `{child_health.get('missing_child_count', REQUIRED_CHILD_COUNT)}`",
                        f"- Unhealthy child count: `{child_health.get('unhealthy_child_count', 0)}`",
                        "",
                        "| Slot | Child | Agent | Role | Status | Health | Work Items | Summary |",
                        "| --- | --- | --- | --- | --- | --- | --- | --- |",
                    ]
                )
                for slot in child_health.get("child_roster", []):
                    joined_ids = ", ".join(slot["work_item_ids"]) if slot.get("work_item_ids") else "-"
                    agent_name = slot.get("agent_name") or slot.get("child_session_id") or "-"
                    lines.append(
                        f"| {slot['slot_id']} | {slot.get('child_session_id') or '-'} | {agent_name} | {slot.get('role') or '-'} | {slot.get('activity_status') or slot.get('status') or '-'} | {slot.get('health') or '-'} | {joined_ids} | {(slot.get('summary') or '').replace('|', '/')} |"
                    )
                checkpoint = item["latest_checkpoint"]
                if checkpoint:
                    blockers = ", ".join(checkpoint["blockers"]) if checkpoint["blockers"] else "-"
                    next_actions = ", ".join(checkpoint["next_actions"]) if checkpoint["next_actions"] else "-"
                    lines.extend(["", f"- Latest checkpoint: `{checkpoint['created_at']}`", f"- Blockers: {blockers}", f"- Next actions: {next_actions}"])
        lines.extend(["", "## Unread Inbox"])
        if not snapshot["unread_messages"]:
            lines.extend(["", "No unread inbox or broadcast messages."])
        else:
            lines.extend(["", "| ID | To | Category | Subject | Related |", "| --- | --- | --- | --- | --- |"])
            for item in snapshot["unread_messages"]:
                target = item["recipient_session"] or "broadcast"
                related = item["related_work_item_key"] or item["related_path"] or "-"
                lines.append(f"| {item['message_id']} | {target} | {item['category']} | {item['subject'].replace('|', '/')} | {related} |")
        lines.extend(["", "## Claimed Work Items"])
        if not snapshot["claimed_work_items"]:
            lines.extend(["", "No claimed work items."])
        else:
            lines.extend(
                [
                    "",
                    "| Scope | Work Item | Status | Owner | Parent | Slot | Role | Title | Updated |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for item in snapshot["claimed_work_items"]:
                lines.append(
                    f"| {item['task_scope']} | {item['work_item_id']} | {item['status']} | {item['owner_session']} | {item.get('owner_parent_session') or '-'} | {item['owner_slot'] or '-'} | {item.get('owner_role') or '-'} | {item['title'].replace('|', '/')} | {item['updated_at']} |"
                )
        lines.extend(["", "## File Claims"])
        if not snapshot["file_claims"]:
            lines.extend(["", "No active file claims."])
        else:
            lines.extend(
                [
                    "",
                    "| Path | Section | Lines | Owner | Parent | Slot | Mode | Heartbeat |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for item in snapshot["file_claims"]:
                line_range = (
                    f"{item['section_start_line']}-{item['section_end_line']}"
                    if item["section_start_line"] is not None and item["section_end_line"] is not None
                    else "-"
                )
                lines.append(
                    f"| {item['path']} | {item['section_id'] or '-'} | {line_range} | {item['owner_session']} | {item.get('owner_parent_session') or '-'} | {item['owner_slot'] or '-'} | {item['mode']} | {item['last_heartbeat']} |"
                )
        lines.extend(["", "## Python Leases"])
        if not snapshot["python_leases"]:
            lines.extend(["", "No active Python leases."])
        else:
            lines.extend(
                [
                    "",
                    "| Lease | Owner | Parent | Slot | Status | Health | PID | Memory Cap | Method | Purpose |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for item in snapshot["python_leases"]:
                if item["memory_cap_bytes"] is not None:
                    memory_cap = f"{item['memory_cap_bytes']} bytes"
                elif item["memory_cap_percent"] is not None:
                    memory_cap = f"{item['memory_cap_percent']}%"
                else:
                    memory_cap = "unknown"
                lines.append(
                    f"| {item['lease_id']} | {item['owner_session']} | {item.get('owner_parent_session') or '-'} | {item['owner_slot'] or '-'} | {item['status']} | {item['health']} | {item['pid'] or '-'} | {memory_cap} | {item['enforcement_method']} | {item['purpose'].replace('|', '/')} |"
                )
        lines.extend(["", "## Handoff Messages"])
        if not snapshot["handoff_messages"]:
            lines.extend(["", "No handoff messages."])
        else:
            lines.extend(["", "| ID | To | Subject | Ack |", "| --- | --- | --- | --- |"])
            for item in snapshot["handoff_messages"]:
                lines.append(f"| {item['message_id']} | {item['recipient_session'] or 'broadcast'} | {item['subject'].replace('|', '/')} | {item['ack_by'] or 'pending'} |")
        return "\n".join(lines).rstrip() + "\n"

    def refresh_snapshots(self) -> dict[str, Any]:
        snapshot = self.collect_status()
        generation = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        snapshot["snapshot_generation"] = generation
        session_dir = self.context.snapshots_dir / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        marker_fields = {
            "generated_at": snapshot["generated_at"],
            "snapshot_generation": generation,
            "state_revision": snapshot["state_revision"],
            "schema_version": snapshot["schema_version"],
        }
        atomic_write_json(
            self.context.snapshots_dir / "active_sessions.json",
            {
                **marker_fields,
                "sessions": snapshot["active_sessions"],
                "active_child_sessions": snapshot.get("active_child_sessions", []),
                "parent_child_invariant_violations": snapshot.get("parent_child_invariant_violations", []),
            },
        )
        atomic_write_json(self.context.snapshots_dir / "work_item_claims.json", {**marker_fields, "claimed_work_items": snapshot["claimed_work_items"], "all_work_items": snapshot["work_items"]})
        atomic_write_json(self.context.snapshots_dir / "file_claims.json", {**marker_fields, "file_claims": snapshot["file_claims"]})
        atomic_write_json(self.context.snapshots_dir / "python_leases.json", {**marker_fields, "python_leases": snapshot["python_leases"], "reconciled_python_leases": snapshot["reconciled_python_leases"]})
        atomic_write_json(self.context.snapshots_dir / "messages.json", {**marker_fields, "messages": snapshot["messages"], "unread_messages": snapshot["unread_messages"]})
        atomic_write_json(self.context.snapshots_dir / "events.json", {**marker_fields, "events": snapshot["events"]})
        for session_id, payload in snapshot["sessions"].items():
            atomic_write_json(session_dir / f"{session_id}.json", {**marker_fields, **payload})
        atomic_write_text(self.context.snapshots_dir / "status.md", self.render_status_board(snapshot))
        atomic_write_json(
            self.context.snapshots_dir / "snapshot_manifest.json",
            {
                **marker_fields,
                "files": list(SNAPSHOT_FILES),
                "session_count": len(snapshot["sessions"]),
                "active_session_count": len(snapshot["active_sessions"]),
                "active_child_session_count": len(snapshot.get("active_child_sessions", [])),
            },
        )
        return snapshot

    def write_archive(self, session_id: str) -> None:
        with self._connect() as connection:
            session_row = self._require_session_row(connection, session_id, active_only=False)
            session = self._parse_session_row(session_row)
            session["task_scope"] = session.get("task_scope") or derive_task_scope(session["task_summary"])
            session["worktree_root"] = session.get("worktree_root") or session["repo_root"]
            session["repo_identity"] = session.get("repo_identity") or self.context.repo_identity
            session.update(self._session_staleness(session))
            work_items = self._session_work_items(connection, session_id)
            payload = {
                "session": session,
                "slots": self._session_slots(connection, session_id) if session["session_type"] == "parent" else [],
                "children": [],
                "child_notes": [],
                "child_health": None,
                "work_items": work_items,
                "work_item_counts": self._work_item_counts(work_items),
                "file_claims": self._session_file_claims(connection, session_id),
                "python_leases": [self._decorate_python_lease(item) for item in self._session_python_leases(connection, session_id)],
                "messages": self._session_messages(connection, session_id, include_archived=True),
                "latest_checkpoint": None,
                "latest_child_note": None,
            }
            if session["session_type"] == "parent":
                children = [self._parse_session_row(row) for row in self._descendant_child_rows(connection, session_id)]
                for child in children:
                    child.update(self._session_staleness(child))
                    child.update(self._child_health(child))
                payload["children"] = children
                payload["child_notes"] = self._child_note_rows(connection, parent_session_id=session_id)
                payload["child_health"] = self._parent_child_health(
                    connection,
                    parent_session_id=session_id,
                    required_child_count=int(session["slot_count"] or REQUIRED_CHILD_COUNT),
                )
            else:
                payload["latest_child_note"] = (self._child_note_rows(connection, child_session_id=session_id) or [None])[0]
            checkpoint = connection.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC, checkpoint_id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if checkpoint is not None:
                payload["latest_checkpoint"] = self._parse_checkpoint_row(checkpoint)
        archive_root = self.context.archive_dir / session_id
        archive_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(archive_root / "final.json", payload)
        session = payload["session"]
        checkpoint = payload["latest_checkpoint"]
        lines = [
            f"# Session Archive: {session_id}",
            "",
            f"- Session type: `{session['session_type']}`",
            f"- Status: `{session['status']}`",
            f"- Outcome: `{session['outcome'] or session['status']}`",
            f"- Task scope: `{session['task_scope']}`",
            f"- Started: `{session['started_at']}`",
            f"- Ended: `{session['ended_at'] or session['last_heartbeat']}`",
            f"- Task summary: {session['task_summary']}",
            f"- Parent session: `{session.get('parent_session_id') or '-'}`",
            f"- Child slot: `{session.get('child_slot_id') or '-'}`",
            f"- Agent name: `{session.get('agent_name') or '-'}`",
            f"- Agent kind: `{session.get('agent_kind') or '-'}`",
            f"- Role: `{session.get('role') or '-'}`",
            f"- Activity status: `{session.get('activity_status') or '-'}`",
            f"- Resumed from: `{session.get('resume_from_session') or '-'}`",
            f"- Resumed by: `{session.get('resumed_by_session') or '-'}`",
            f"- Reaped by: `{session.get('reaped_by_session') or '-'}`",
            f"- Taken over by: `{session.get('takeover_by_session') or '-'}`",
            f"- Replaced by: `{session.get('replaced_by_session') or '-'}`",
            f"- Replacement for: `{session.get('replacement_for_session') or '-'}`",
            f"- Terminal reason: {session.get('terminal_reason') or '-'}",
            "",
            "## Work Item Counts",
            "",
            "| Open | Claimed | In Progress | Blocked | QA | Closed |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            "| {open} | {claimed} | {in_progress} | {blocked} | {qa} | {closed} |".format(
                open=payload["work_item_counts"].get("open", 0),
                claimed=payload["work_item_counts"].get("claimed", 0),
                in_progress=payload["work_item_counts"].get("in_progress", 0),
                blocked=payload["work_item_counts"].get("blocked", 0),
                qa=payload["work_item_counts"].get("qa", 0),
                closed=payload["work_item_counts"].get("closed", 0),
            ),
            "",
            "## Latest Checkpoint",
            "",
        ]
        if checkpoint:
            resume_context = checkpoint["resume_context"]
            lines.extend([
                f"- Created: `{checkpoint['created_at']}`",
                f"- Blockers: {', '.join(checkpoint['blockers']) if checkpoint['blockers'] else '-'}",
                f"- Next actions: {', '.join(checkpoint['next_actions']) if checkpoint['next_actions'] else '-'}",
                f"- Evidence: {', '.join(checkpoint['evidence_paths']) if checkpoint['evidence_paths'] else '-'}",
                f"- Resume files claimed: `{len(resume_context.get('file_claims', []))}`",
                f"- Resume active Python leases: `{len(resume_context.get('active_python_leases', []))}`",
            ])
        else:
            lines.append("No checkpoints were recorded.")
        if payload["child_health"] is not None:
            child_health = payload["child_health"]
            lines.extend(
                [
                    "",
                    "## Child Roster",
                    "",
                    f"- Compliance: `{child_health['child_compliance']}`",
                    f"- Required child count: `{child_health['required_child_count']}`",
                    f"- Live child count: `{child_health['live_child_count']}`",
                    f"- Missing child count: `{child_health['missing_child_count']}`",
                    f"- Unhealthy child count: `{child_health['unhealthy_child_count']}`",
                    "",
                ]
            )
            if payload["children"]:
                lines.extend(
                    [
                        "| Slot | Child | Outcome | Health | Role | Status | Work Items | Summary |",
                        "| --- | --- | --- | --- | --- | --- | --- | --- |",
                    ]
                )
                for child in payload["children"]:
                    joined_ids = ", ".join(child["work_item_ids"]) if child["work_item_ids"] else "-"
                    lines.append(
                        f"| {child.get('child_slot_id') or '-'} | {child['session_id']} | {child.get('outcome') or child['status']} | {child.get('health') or '-'} | {child.get('role') or '-'} | {child.get('activity_status') or '-'} | {joined_ids} | {(child.get('summary') or '').replace('|', '/')} |"
                    )
            else:
                lines.append("No child sessions were recorded.")
        elif payload["latest_child_note"] is not None:
            note = payload["latest_child_note"]
            lines.extend(
                [
                    "",
                    "## Latest Child Note",
                    "",
                    f"- Created: `{note['created_at']}`",
                    f"- Category: `{note['category']}`",
                    f"- Summary: {note['summary']}",
                ]
            )
        lines.extend(["", "## Messages", ""])
        if not payload["messages"]:
            lines.append("No messages were recorded.")
        else:
            lines.extend(["| ID | Direction | Category | Subject | Ack |", "| --- | --- | --- | --- | --- |"])
            for item in payload["messages"]:
                direction = "outbound" if item["sender_session"] == session_id else "inbound"
                lines.append(f"| {item['message_id']} | {direction} | {item['category']} | {item['subject'].replace('|', '/')} | {item['ack_by'] or 'pending'} |")
        lines.extend([
            "",
            "## Cleanup Result",
            "",
            f"- Remaining file claims: `{len(payload['file_claims'])}`",
            f"- Remaining active Python leases: `{len([lease for lease in payload['python_leases'] if lease['closed_at'] is None])}`",
            "",
        ])
        atomic_write_text(archive_root / "final.md", "\n".join(lines).rstrip() + "\n")

    def status_text(self) -> str:
        return self.render_status_board(self.refresh_snapshots())

    def status_json(self) -> dict[str, Any]:
        return self.refresh_snapshots()


def print_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def describe_runtime(context: RepoContext) -> str:
    return textwrap.dedent(
        f"""
        repo_root={context.repo_root}
        runtime_root={context.runtime_root}
        git_common_dir={context.git_common_dir or '<fallback local runtime>'}
        db_path={context.db_path}
        """
    ).strip()
