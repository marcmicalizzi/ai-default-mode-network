import copy
from contextlib import closing
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dmn.adoption import adopt_openwebui, validate_adopted_message
from dmn.backend import DemoBackend, sha256_file
from dmn.bridge import BridgeLedger
from dmn.config import Config
from dmn.runtime import Runtime
from dmn.storage import write_durable


class AdoptionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.instance = self.root / "instance"
        self.capture = self.root / "capture"
        self.capture.mkdir()
        self.database = self.root / "webui.db"
        self.nodes = {"u": {"id": "u", "role": "user", "content": "earlier", "parentId": None},
                      "a": {"id": "a", "role": "assistant", "content": "reply", "parentId": "u"}}
        self.chat = {"history": {"messages": self.nodes, "currentId": "a"}}
        source = {"id": "chat", "user_id": "owner", "chat": self.chat}
        write_durable(self.capture / "source-chat.json", source)
        config = Config(backend="demo", n_ctx=16384)
        r = Runtime(self.instance, config, DemoBackend(config), prepare_only=True)
        self.id = r.state["instance_id"]
        # Synthetic import marker for orchestration; no claim of native import.
        r.state["initial_context"] = {"fixture": True}
        r.checkpoint()
        r.close()
        (self.instance / "import").mkdir()
        (self.instance / "import/provider-request.json").write_text('{"fixture":true}')
        write_durable(self.capture / "report.json", {"source_chat_sha256": sha256_file(self.capture / "source-chat.json"),
            "provider_request_sha256": sha256_file(self.instance / "import/provider-request.json"), "source_leaf_id": "a"})
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE chat(id,user_id,chat,current_message_id)")
            db.execute("CREATE TABLE chat_message(id,chat_id,role,content,parent_id,output,files,context_summary)")
            db.execute("INSERT INTO chat VALUES(?,?,?,?)", ("chat", "owner", json.dumps(self.chat), "a"))
            for key, node in self.nodes.items():
                db.execute("INSERT INTO chat_message VALUES(?,?,?,?,?,NULL,NULL,NULL)", ("chat-" + key, "chat", node["role"], json.dumps(node["content"]), node["parentId"]))

    def tearDown(self):
        self.temp.cleanup()

    def test_explicit_adoption_is_idempotent_and_does_not_edit_source(self):
        before = self.database.read_bytes()
        for _ in range(2):
            result = adopt_openwebui(self.instance, self.database, self.capture)
            self.assertFalse(result["generation_started"])
        self.assertEqual(self.database.read_bytes(), before)
        ledger = BridgeLedger(self.root / "dmn-bridge/relay.sqlite3")
        try:
            self.assertEqual(ledger.binding()["instance_id"], self.id)
        finally:
            ledger.close()

    def test_changed_source_or_wrong_import_refuses_binding(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("UPDATE chat SET current_message_id='other'")
        with self.assertRaisesRegex(ValueError, "changed"):
            adopt_openwebui(self.instance, self.database, self.capture)
        self.assertFalse((self.root / "dmn-bridge/relay.sqlite3").exists())

    def test_historical_inputs_wrong_branches_and_edits_are_rejected(self):
        evidence = {"source_messages": copy.deepcopy(self.nodes), "source_leaf_id": "a"}
        validate_adopted_message(evidence, self.chat, {"id": "new", "parentId": "a"})
        for message in (self.nodes["u"], {"id": "new", "parentId": "u"}, {"id": "new", "parentId": None}):
            with self.assertRaises(ValueError):
                validate_adopted_message(evidence, self.chat, message)
        self.nodes["u"]["content"] = "edited"
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_adopted_message(evidence, self.chat, {"id": "new", "parentId": "a"})

    def test_normalized_history_change_is_not_hidden_by_old_json(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("UPDATE chat_message SET content=? WHERE id='chat-u'", (json.dumps("edited"),))
        with self.assertRaisesRegex(ValueError, "normalized"):
            adopt_openwebui(self.instance, self.database, self.capture)
