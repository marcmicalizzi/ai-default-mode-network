"""Per-chat relay cursors, input receipts and immutable source bindings."""
import hashlib
import sqlite3


class MultiBridgeLedger:
    def __init__(self, path, instance_id, namespace):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS source (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), instance_id TEXT NOT NULL, namespace TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS bindings (
                chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, instance_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL UNIQUE, participant_id TEXT NOT NULL, cursor INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS receipts (
                chat_id TEXT NOT NULL, message_id TEXT NOT NULL, digest TEXT NOT NULL,
                assistant_id TEXT NOT NULL, event_id INTEGER, PRIMARY KEY(chat_id,message_id));
            CREATE TABLE IF NOT EXISTS contact_updates (chat_id TEXT PRIMARY KEY, status TEXT NOT NULL);
        ''')
        prior = self.db.execute("SELECT instance_id,namespace FROM source").fetchone()
        if prior and tuple(prior) != (instance_id, namespace):
            self.db.close()
            raise ValueError("relay ledger belongs to another instance or WebUI source")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO source VALUES(1,?,?)", (instance_id, namespace))
        self.instance_id = instance_id

    def binding(self, chat_id=None):
        row = self.db.execute("SELECT * FROM bindings WHERE chat_id=?", (chat_id,)).fetchone()
        return dict(row) if row else None

    def bindings(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM bindings ORDER BY chat_id")]

    def bind(self, chat_id, user_id, conversation_id, participant_id):
        expected = dict(chat_id=chat_id, user_id=user_id, instance_id=self.instance_id,
                        conversation_id=conversation_id, participant_id=participant_id)
        with self.db:
            prior = self.binding(chat_id)
            if prior and any(prior[k] != v for k, v in expected.items()):
                raise ValueError("chat binding cannot change owner or destination")
            self.db.execute("INSERT OR IGNORE INTO bindings(chat_id,user_id,instance_id,conversation_id,participant_id) VALUES(?,?,?,?,?)", tuple(expected.values()))
        return self.binding(chat_id)

    def receipt(self, chat_id, message_id, content, assistant_id, *, digest=None):
        digest = digest or hashlib.sha256(content.encode()).hexdigest()
        with self.db:
            prior = self.get_receipt(message_id, chat_id)
            if prior and (prior["digest"] != digest or prior["assistant_id"] != assistant_id):
                raise ValueError("editing or regenerating an accepted message cannot rewind DMN")
            self.db.execute("INSERT OR IGNORE INTO receipts VALUES(?,?,?,?,NULL)", (chat_id, message_id, digest, assistant_id))
        return prior

    def get_receipt(self, message_id, chat_id=None):
        row = self.db.execute("SELECT * FROM receipts WHERE chat_id=? AND message_id=?", (chat_id, message_id)).fetchone()
        return dict(row) if row else None

    def accepted(self, message_id, event_id, chat_id=None):
        with self.db:
            self.db.execute("UPDATE receipts SET event_id=? WHERE chat_id=? AND message_id=?", (event_id, chat_id, message_id))

    def advance(self, chat_id, cursor):
        with self.db:
            self.db.execute("UPDATE bindings SET cursor=max(cursor,?) WHERE chat_id=?", (cursor, chat_id))

    def placeholders(self, chat_id=None):
        return {row[0] for row in self.db.execute("SELECT assistant_id FROM receipts WHERE chat_id=? AND event_id IS NOT NULL", (chat_id,))}

    def contact_status(self, chat_id):
        row = self.db.execute("SELECT status FROM contact_updates WHERE chat_id=?", (chat_id,)).fetchone()
        return row[0] if row else None

    def save_contact_status(self, chat_id, status):
        with self.db:
            self.db.execute("INSERT INTO contact_updates VALUES(?,?) ON CONFLICT(chat_id) DO UPDATE SET status=excluded.status", (chat_id, status))

    def close(self):
        self.db.close()
