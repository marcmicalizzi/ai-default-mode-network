from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath


def json_text(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)


def write_durable(path: Path, value):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json_text(value))
        f.flush()
        os.fsync(f.fileno())


def memory_path(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or len(value) > 512:
        raise ValueError("memory path must be an absolute logical path of at most 512 characters")
    if "\\" in value or any(ord(c) < 32 for c in value) or ".." in value.split("/"):
        raise ValueError("invalid memory path")
    path = str(PurePosixPath(value))
    if path == "/":
        raise ValueError("a memory needs a name")
    return path


class InstanceLock:
    """OS advisory lock: process death releases it; stale filenames do not block restart."""
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.file = (root / "instance.lock").open("a+b")
        try:
            if os.fstat(self.file.fileno()).st_size == 0:
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError("this instance is already open in another process") from exc

    def close(self):
        if not self.file.closed:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.RLock()
        self.closed = False
        self.db = sqlite3.connect(self.root / "runtime.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY, directory TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, action_id TEXT UNIQUE NOT NULL, content TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS message_destinations (
                message_id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                participant_id TEXT NOT NULL, in_reply_to INTEGER);
            CREATE TABLE IF NOT EXISTS memories (
                path TEXT PRIMARY KEY, content TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS memory_versions (
                path TEXT NOT NULL, revision INTEGER NOT NULL, content TEXT,
                operation TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(path,revision));
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS event_keys (
                key TEXT PRIMARY KEY, event_id INTEGER NOT NULL REFERENCES events(id));
            CREATE TABLE IF NOT EXISTS learning_plans (
                revision TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS sleep_recipes (
                revision TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS sleep_executions (
                revision TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS sleep_runs (
                id TEXT PRIMARY KEY, phase TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS activity_intents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL, created REAL NOT NULL);
        ''')
        # Archive existing current values once when opening an older database.
        self.db.execute("""INSERT INTO memory_versions(path,revision,content,operation,created)
            SELECT path,1,content,'baseline',updated FROM memories m
            WHERE NOT EXISTS(SELECT 1 FROM memory_versions v WHERE v.path=m.path)""")
        self.db.commit()

    @contextmanager
    def transaction(self):
        with self.mutex:
            with self.db:
                yield self.db

    def enqueue(self, kind, payload, now=None, idempotency_key=None):
        with self.transaction() as db:
            return self._enqueue(db, kind, payload, now, idempotency_key)

    def _enqueue(self, db, kind, payload, now=None, idempotency_key=None):
        """Insert inside the caller's transaction, without an inner commit."""
        encoded = json_text(payload)
        if idempotency_key is not None:
            if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 256:
                raise ValueError("idempotency_key must be a string of 1 to 256 characters")
            prior = db.execute("SELECT e.* FROM events e JOIN event_keys k ON e.id=k.event_id WHERE k.key=?",
                               (idempotency_key,)).fetchone()
            if prior:
                if prior["kind"] != kind or prior["payload"] != encoded:
                    raise ValueError("idempotency key already used with different content")
                return prior["id"]
        event_id = db.execute("INSERT INTO events(kind,payload,created) VALUES(?,?,?)",
                              (kind, encoded, time.time() if now is None else now)).lastrowid
        if idempotency_key is not None:
            db.execute("INSERT INTO event_keys VALUES(?,?)", (idempotency_key, event_id))
        return event_id

    def next_event(self, cursor):
        with self.mutex:
            row = self.db.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT 1", (cursor,)).fetchone()
            return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    def latest(self):
        with self.mutex:
            row = self.db.execute("SELECT directory FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
            return self.root / "checkpoints" / row[0] if row else None

    def record(self, kind, payload, now=None):
        with self.transaction() as db:
            db.execute("INSERT INTO records(kind,payload,created) VALUES(?,?,?)",
                       (kind, json_text(payload), time.time() if now is None else now))

    def commit_checkpoint(self, directory: str, effects: list[dict], now: float, events=()):
        # State pointer and every externally observable action commit together.
        with self.transaction() as db:
            for event in events:
                db.execute("INSERT INTO events(kind,payload,created) VALUES(?,?,?)",
                           (event["kind"], json_text(event["payload"]), event["created"]))
            for effect in effects:
                op = effect["op"]
                if op.startswith("memory_") and "expected_revision" in effect:
                    if self.memory_revision(effect["path"]) != effect["expected_revision"]:
                        raise ValueError("memory revision changed before commit")
                if op == "send_message":
                    message_id = db.execute("INSERT INTO messages(action_id,content,created) VALUES(?,?,?)",
                                           (effect["action_id"], effect["content"], now)).lastrowid
                    if "conversation_id" in effect:
                        db.execute("INSERT INTO message_destinations VALUES(?,?,?,?)",
                                   (message_id, effect["conversation_id"], effect["participant_id"], effect.get("in_reply_to")))
                elif op in {"events_delivered", "close_conversation", "reopen_conversation",
                            "block_participant", "unblock_participant", "contact_decide"}:
                    from .conversations import commit_effect
                    commit_effect(db, effect, now)
                elif op == "memory_write":
                    db.execute("INSERT INTO memories VALUES(?,?,?) ON CONFLICT(path) DO UPDATE SET content=excluded.content, updated=excluded.updated",
                               (effect["path"], effect["content"], now))
                    self._archive_memory(db, effect["path"], effect["content"], op, now)
                elif op == "memory_delete":
                    db.execute("DELETE FROM memories WHERE path=?", (effect["path"],))
                    self._archive_memory(db, effect["path"], None, op, now)
                elif op == "memory_move":
                    content = self.memory_read(effect["path"])
                    db.execute("UPDATE memories SET path=?,updated=? WHERE path=?",
                               (effect["destination"], now, effect["path"]))
                    self._archive_memory(db, effect["path"], None, op, now)
                    self._archive_memory(db, effect["destination"], content, op, now)
                elif op == "prompt_record":
                    db.execute("INSERT INTO records(kind,payload,created) VALUES(?,?,?)",
                               ("prompt_decision", json_text(effect["record"]), now))
                elif op in {"learning_plan_create", "learning_plan_withdraw"}:
                    from .learning import commit_effect
                    commit_effect(db, effect, now)
                elif op == "image_permission":
                    from .attachments import commit_effect
                    commit_effect(db, effect, now)
                elif op in {"learning_compile", "learning_candidate_prepare", "learning_execution_decide", "deep_sleep"}:
                    from .sleep_plans import commit_effect
                    commit_effect(db, effect, now, directory)
                else:
                    raise ValueError(f"unknown staged effect {op}")
            db.execute("INSERT INTO checkpoints(directory,created) VALUES(?,?)", (directory, now))
            # The single inference owner captured all pending activity choices.
            # Clear only in the same transaction that publishes their snapshot.
            db.execute("DELETE FROM activity_intents")

    def activity_intent(self):
        with self.mutex:
            row = self.db.execute("SELECT id,payload FROM activity_intents ORDER BY id DESC LIMIT 1").fetchone()
            return {"revision": row["id"], **json.loads(row["payload"])} if row else None

    def put_activity_intent(self, payload, now):
        with self.transaction() as db:
            latest = db.execute("SELECT directory FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
            if latest is None or payload["checkpoint"] != latest[0]:
                raise ValueError("activity intent must refer to the current checkpoint")
            revision = db.execute("INSERT INTO activity_intents(payload,created) VALUES(?,?)",
                                  (json_text(payload), now)).lastrowid
            db.execute("DELETE FROM activity_intents WHERE id<?", (revision,))
            return revision

    def _archive_memory(self, db, path, content, operation, now):
        db.execute("INSERT INTO memory_versions VALUES(?,?,?,?,?)",
                   (path, self.memory_next_revision(path), content, operation, now))

    def memory_next_revision(self, path):
        with self.mutex:
            return self.db.execute("SELECT COALESCE(MAX(revision),0)+1 FROM memory_versions WHERE path=?", (path,)).fetchone()[0]

    def memory_revision(self, path):
        with self.mutex:
            if not self.db.execute("SELECT 1 FROM memories WHERE path=?", (path,)).fetchone():
                return 0
            return self.memory_next_revision(path) - 1

    def memory_history(self, path, offset=0, limit=20):
        with self.mutex:
            return [dict(r) for r in self.db.execute("""SELECT revision,operation,created,LENGTH(content) AS characters
                FROM memory_versions WHERE path=? ORDER BY revision DESC LIMIT ? OFFSET ?""", (path, limit, offset))]

    def memory_read(self, path, revision=None):
        with self.mutex:
            row = (self.db.execute("SELECT content FROM memories WHERE path=?", (path,)).fetchone() if revision is None else
                   self.db.execute("SELECT content FROM memory_versions WHERE path=? AND revision=?", (path, revision)).fetchone())
            if row is None:
                raise ValueError("memory does not exist")
            if row[0] is None:
                raise ValueError("this revision records a removal; read an earlier revision")
            return row[0]

    def memory_list(self, prefix="/", offset=0, limit=50):
        with self.mutex:
            return [r[0] for r in self.db.execute(
                "SELECT path FROM memories WHERE substr(path,1,?)=? ORDER BY path LIMIT ? OFFSET ?",
                (len(prefix), prefix, limit, offset))]

    def messages(self, after=0, limit=200, conversation_id=None):
        with self.mutex:
            rows = self.db.execute('''SELECT m.*, d.conversation_id, d.participant_id, d.in_reply_to
                FROM messages m LEFT JOIN message_destinations d ON d.message_id=m.id
                WHERE m.id>? AND (? IS NULL OR d.conversation_id=?) ORDER BY m.id LIMIT ?''',
                (after, conversation_id, conversation_id, limit)).fetchall()
            return [{k: v for k, v in dict(r).items()
                     if r["conversation_id"] is not None or k not in {"conversation_id", "participant_id", "in_reply_to"}}
                    for r in rows]

    def close(self):
        with self.mutex:
            if not self.closed:
                self.db.close()
                self.closed = True
