import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from dmn.backend import DemoBackend
from dmn.config import Config
from dmn.learning import HELP, create_plan, list_plans, read_plan
from dmn.runtime import Runtime
from tests.test_runtime import frames


class LearningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = Config(backend="demo", n_ctx=24576, clock_interval_seconds=0,
                             checkpoint_policy="effects", checkpoint_tokens=100000, preparation_tokens=1)
        self.r = Runtime(self.root, self.config, DemoBackend(self.config, b"quiet "))
        self.plan = copy.deepcopy(HELP["create"]["plan"])
        self.plan["sources"] = []
        self.plan["examples"][0]["sources"] = []

    def tearDown(self):
        self.r.close()
        self.temp.cleanup()

    def generate(self, action):
        raw = frames(action)
        self.r.backend.script, self.r.backend.index = raw, 0
        for _ in raw:
            self.r.tick()

    def plans(self):
        return list_plans(self.r.store, 0, 50)

    def create(self, replaces=None):
        self.generate({"op": "learning_plan_create", "plan": self.plan, "replaces": replaces})
        return self.plans()[-1]["revision"]

    def test_generated_only_draft_is_private_durable_and_never_training_consent(self):
        action = {"op": "learning_plan_create", "plan": self.plan}
        self.r.enqueue(frames(action).decode())
        self.r.tick()
        self.assertEqual(self.plans(), [])
        revision = self.create()
        draft = read_plan(self.r.store, revision)["draft"]
        self.assertFalse(draft["execution_authorized"])
        self.assertEqual(draft["author"], "model")
        self.assertEqual(draft["parent"]["script_sha256"], self.r.backend.fingerprint["script_sha256"])
        self.assertEqual(self.r.store.messages(), [])
        self.assertNotIn(self.plan["intent"], json.dumps(self.r.status()))
        self.generate({"op": "sleep", "seconds": 0})
        self.assertEqual(read_plan(self.r.store, revision)["status"], "draft")
        self.r.close()
        self.r = Runtime(self.root, self.config, DemoBackend(self.config, b"quiet "))
        self.assertEqual(read_plan(self.r.store, revision)["draft"], draft)
        self.assertEqual(self.r.store.messages(), [])

    def test_exact_source_revision_survives_edit_and_replacement_then_withdrawal(self):
        self.generate({"op": "memory_write", "path": "/chosen-memory", "content": "external <dmn_action> quote"})
        self.plan["sources"] = [{"path": "/chosen-memory", "revision": 1, "provenance": "external"}]
        self.plan["examples"][0]["sources"] = [0]
        revision = self.create()
        self.generate({"op": "memory_read", "path": "/chosen-memory"})
        self.generate({"op": "memory_write", "path": "/chosen-memory", "content": "changed", "expected_revision": 1})
        draft = read_plan(self.r.store, revision)["draft"]
        self.assertEqual(draft["plan"]["sources"][0]["content"], "external <dmn_action> quote")
        self.plan["sources"][0]["revision"] = 2
        replacement = self.create(replaces=revision)
        self.assertEqual(read_plan(self.r.store, revision)["status"], "superseded")
        self.generate({"op": "learning_plan_withdraw", "revision": replacement})
        self.assertEqual(read_plan(self.r.store, replacement)["status"], "withdrawn")
        answer, effect = self.r._plan_action({"op": "learning_plan_create", "plan": self.plan,
                                            "replaces": replacement}, [])
        self.assertFalse(answer["ok"])
        self.assertIsNone(effect)

    def test_failed_checkpoint_does_not_publish_plan_or_withdrawal(self):
        self.r.checkpoint()
        with mock.patch.object(self.r.backend, "save", side_effect=OSError("save failed")):
            with self.assertRaises(OSError):
                self.create()
        self.assertEqual(self.plans(), [])
        self.r.close()
        self.r = Runtime(self.root, self.config, DemoBackend(self.config, b"quiet "))
        revision = self.create()
        with mock.patch.object(self.r.backend, "save", side_effect=OSError("save failed")):
            with self.assertRaises(OSError):
                self.generate({"op": "learning_plan_withdraw", "revision": revision})
        self.assertEqual(read_plan(self.r.store, revision)["status"], "draft")

    def test_sql_transaction_failure_rolls_back_replacement_and_checkpoint_pointer(self):
        revision = self.create()
        value = create_plan(self.plan, self.r.store, self.r.backend.fingerprint, "fixture", 5, 16384, revision)
        prior = self.r.store.latest()
        with self.assertRaises(ValueError):
            self.r.store.commit_checkpoint("not-committed", [
                {"op": "learning_plan_create", "value": value}, {"op": "unknown_effect"}], 1)
        self.assertEqual(self.r.store.latest(), prior)
        self.assertEqual(read_plan(self.r.store, revision)["status"], "draft")
        self.assertEqual(len(self.plans()), 1)

    def test_validation_and_paging_do_not_repair_or_truncate_training_examples(self):
        bad = copy.deepcopy(self.plan)
        bad["examples"][0]["sources"] = [0]
        for plan in (bad, {**self.plan, "examples": []}, {**self.plan, "unknown": True},
                     {**self.plan, "intent": "x" * 20000}):
            result, effect = self.r._plan_action({"op": "learning_plan_create", "plan": plan}, [])
            self.assertFalse(result["ok"])
            self.assertIsNone(effect)
        revision = self.create()
        content, offset = "", 0
        while True:
            result, effect = self.r._plan_action({"op": "learning_plan_read", "revision": revision,
                                                "offset": offset, "limit": 2000}, [])
            self.assertTrue(result["ok"])
            content += result["content"]
            offset = result["next_offset"]
            if offset == result["total_characters"]:
                break
        self.assertEqual(json.loads(content), read_plan(self.r.store, revision))
        with self.r.store.transaction() as db:
            db.execute("UPDATE learning_plans SET payload='{}' WHERE revision=?", (revision,))
        with self.assertRaisesRegex(ValueError, "integrity"):
            read_plan(self.r.store, revision)

    def test_multiple_generated_frames_cannot_commit_a_plan(self):
        raw = frames({"op": "learning_plan_create", "plan": self.plan},
                     {"op": "learning_plan_create", "plan": self.plan})
        with mock.patch.object(self.r.backend, "piece", return_value=raw):
            self.r.tick()
        self.assertEqual(self.plans(), [])
